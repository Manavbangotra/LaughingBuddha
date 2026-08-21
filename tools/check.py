#!/usr/bin/env python3
"""
check.py — the book's quality gates.

  check.py                 run every gate over the whole book
  check.py --part 7        restrict to one part
  check.py --gate code     run one gate
  check.py --list          list the gates

A book of this size is written across many sessions, and the failure mode that
matters is not a typo — it is *drift*: a symbol that changes meaning in Part XV,
a chapter that quietly depends on one written eighty chapters later, a citation
nobody ever checked, a code listing that stopped running three refactors ago.
Every gate here exists to make one species of drift impossible to miss.

Exit status is 0 only when no gate reports an error. Warnings never fail the
build; they are printed so they can be triaged.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxml import html as lhtml

from bookdata import ROOT, Book, Doc, split_frontmatter
from build import Builder

VENV_PY = ROOT / ".venv" / "bin" / "python"

# --- chapter templates --------------------------------------------------------

SECTIONS_FULL = [
    "Learning Objectives", "Why This Matters", "Prerequisites",
    "Intuitive Explanation", "Formal Explanation", "Mathematical Foundation",
    "Internal Mechanics", "Implementation", "Practical Example",
    "Production Considerations", "Common Mistakes", "Failure Modes",
    "Alternatives", "Evaluation", "Advanced Concepts",
    "Connection to Previous Chapters", "Exercises", "Interview Questions",
    "Research Questions", "Chapter Summary", "Further Reading",
]

SECTIONS_FOCUSED = [
    "Learning Objectives", "Why This Matters", "Prerequisites",
    "Intuitive Explanation", "Formal Explanation", "Mathematical Foundation",
    "Implementation", "Practical Example", "Common Mistakes",
    "Connection to Previous Chapters", "Exercises", "Chapter Summary",
]

TEMPLATE = {"full": SECTIONS_FULL, "focused": SECTIONS_FOCUSED}

REQUIRED_FRONTMATTER = ["id", "number", "part", "tier", "status"]
VALID_STATUS = {"draft", "reviewed", "final"}


# =============================================================================
# Result plumbing
# =============================================================================

@dataclass
class Report:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


class Context:
    """Shared state so gates parse and render the book only once."""

    def __init__(self, parts: list[int] | None):
        self.book = Book()
        self.parts = parts
        self._builder: Builder | None = None

    def in_scope(self, d: Doc) -> bool:
        return self.parts is None or (d.part in self.parts)

    def docs(self) -> list[Doc]:
        return [d for d in self.book.docs
                if d.exists and not d.generated and self.in_scope(d)]

    def chapters(self) -> list[Doc]:
        return [d for d in self.docs() if d.kind == "chapter"]

    @property
    def builder(self) -> Builder:
        if self._builder is None:
            b = Builder()
            b.render_all(verbose=False)
            self._builder = b
        return self._builder

    def headings(self, doc: Doc) -> list[tuple[int, str]]:
        """Top-level (##) and sub (###) headings from the raw Markdown."""
        _, body = split_frontmatter(doc.path.read_text(encoding="utf-8"))
        out, fence = [], None
        for line in body.split("\n"):
            fm = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
            if fm:
                tok = fm.group(1)[0] * 3
                fence = None if fence and line.strip().startswith(fence) else (fence or tok)
                continue
            if fence:
                continue
            hm = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if hm:
                out.append((len(hm.group(1)), hm.group(2).strip()))
        return out


# =============================================================================
# Gates
# =============================================================================

def gate_frontmatter(ctx: Context) -> Report:
    """Every chapter declares the metadata the rest of the toolchain relies on."""
    r = Report("frontmatter")
    for d in ctx.docs():
        if d.kind != "chapter":
            continue
        meta = d.meta
        if "_frontmatter_error" in meta:
            r.error(f"{d.slug}: unparseable frontmatter — {meta['_frontmatter_error']}")
            continue
        for key in REQUIRED_FRONTMATTER:
            if key not in meta:
                r.error(f"{d.slug}: frontmatter missing '{key}'")
        if meta.get("id") != d.id:
            r.error(f"{d.slug}: frontmatter id '{meta.get('id')}' != book.yaml id '{d.id}'")
        if meta.get("number") != d.number:
            r.error(f"{d.slug}: frontmatter number {meta.get('number')} != "
                    f"computed chapter number {d.number}")
        if meta.get("tier") not in (None, d.tier):
            r.error(f"{d.slug}: tier '{meta.get('tier')}' != part tier '{d.tier}'")
        if meta.get("status") not in VALID_STATUS:
            r.error(f"{d.slug}: status '{meta.get('status')}' not in {sorted(VALID_STATUS)}")
        if not meta.get("provides"):
            r.warn(f"{d.slug}: declares no 'provides' concepts")
    return r


def gate_structure(ctx: Context) -> Report:
    """Chapters contain every section their depth tier requires, in order."""
    r = Report("structure")
    for d in ctx.chapters():
        want = TEMPLATE.get(d.tier or "full", SECTIONS_FULL)
        got = [t for lvl, t in ctx.headings(d) if lvl == 2]
        # Section headings are written as "## 7. Internal Mechanics".
        stripped = [re.sub(r"^\d+\.\s*", "", h).strip() for h in got]
        missing = [s for s in want if s not in stripped]
        if missing:
            r.error(f"{d.slug}: missing {d.tier} sections: {', '.join(missing)}")
        order = [stripped.index(s) for s in want if s in stripped]
        if order != sorted(order):
            r.error(f"{d.slug}: template sections are out of order")
        extra = [s for s in stripped if s not in want]
        if extra:
            r.note(f"{d.slug}: extra sections: {', '.join(extra)}")
    return r


def gate_depth(ctx: Context) -> Report:
    """No chapter is quietly compressed below its tier's floor."""
    r = Report("depth")
    b = ctx.builder
    for d in ctx.chapters():
        rendered = b.rendered.get(d.id)
        if rendered is None:
            continue
        floor = ctx.book.tier_spec(d.tier or "full")["min_words"]
        if rendered.words < floor:
            severity = r.error if d.meta.get("status") in ("reviewed", "final") else r.warn
            severity(f"{d.slug}: {rendered.words:,} words is below the "
                     f"{d.tier} floor of {floor:,}")
    return r


def gate_prereqs(ctx: Context) -> Report:
    """The dependency graph is acyclic and never points forward in reading order."""
    r = Report("prereqs")
    book = ctx.book
    # A concept is available once any chapter that provides it has been read.
    provider: dict[str, Doc] = {}
    for d in book.docs:
        if not d.exists:
            continue
        for concept in d.meta.get("provides", []) or []:
            if concept in provider:
                r.error(f"concept '{concept}' provided by both "
                        f"{provider[concept].slug} and {d.slug}")
            provider[concept] = d

    edges: dict[str, set[str]] = defaultdict(set)
    for d in ctx.docs():
        for req in d.meta.get("requires", []) or []:
            src = book.by_id.get(req) or provider.get(req)
            if src is None:
                r.error(f"{d.slug}: requires '{req}', which no chapter provides "
                        f"and no chapter id matches")
                continue
            edges[d.id].add(src.id)
            if src.order > d.order:
                r.error(f"{d.slug} ({d.label}) requires '{req}' from "
                        f"{src.slug} ({src.label}), which comes later in "
                        f"reading order")

    # Cycle detection over whatever edges exist.
    colour: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        colour[node] = 1
        for nxt in edges.get(node, ()):
            if colour.get(nxt) == 1:
                cyc = " -> ".join(stack[stack.index(nxt):] + [nxt]) if nxt in stack \
                    else f"{node} -> {nxt}"
                r.error(f"dependency cycle: {cyc}")
            elif colour.get(nxt, 0) == 0:
                visit(nxt, stack + [nxt])
        colour[node] = 2

    for node in list(edges):
        if colour.get(node, 0) == 0:
            visit(node, [node])
    return r


def gate_references(ctx: Context) -> Report:
    """No cross-reference, equation label, or figure label dangles."""
    r = Report("references")
    b = ctx.builder
    for doc_id, ref in b.resolver.dangling:
        r.error(f"{doc_id}: unresolved reference {{{{{ref}}}}}")
    if b.resolver.pending:
        by_doc: dict[str, int] = defaultdict(int)
        for doc_id, _ in b.resolver.pending:
            by_doc[doc_id] += 1
        r.note(f"{len(b.resolver.pending)} references point at chapters not yet "
               f"written ({len(by_doc)} documents affected)")
    for rendered in b.rendered.values():
        for issue in rendered.issues:
            r.error(f"{rendered.doc.slug}: {issue}")

    # A macro that never became a placeholder leaves its literal source in the
    # output, which no other gate sees — the resolver only inspects placeholders
    # it created. Scan the rendered HTML for surviving {{...}} outside code.
    leak = re.compile(r"\{\{[a-z]+:[^}]{1,80}\}\}")
    for rendered in b.rendered.values():
        if not ctx.in_scope(rendered.doc):
            continue
        tree = lhtml.fragment_fromstring(rendered.body, create_parent="div")
        for el in tree.iter("code", "pre"):
            el.clear()
        for hit in set(leak.findall(" ".join(tree.itertext()))):
            r.error(f"{rendered.doc.slug}: macro {hit} survived into the "
                    f"rendered output — it was never parsed")
    return r


def gate_citations(ctx: Context) -> Report:
    """Every cited key exists in the bibliography and has been verified."""
    r = Report("citations")
    bib = ctx.book.bibliography
    b = ctx.builder
    cited: set[str] = set()
    for rendered in b.rendered.values():
        if not ctx.in_scope(rendered.doc):
            continue
        for key in rendered.citations:
            cited.add(key)
            entry = bib.get(key)
            if entry is None:
                r.error(f"{rendered.doc.slug}: cites '{key}', absent from "
                        f"data/bibliography.yaml")
            elif not entry.get("verified"):
                r.warn(f"{rendered.doc.slug}: cites '{key}', which is not yet "
                       f"verified against a primary source")
    for key, entry in bib.items():
        if entry.get("verified") and not entry.get("verified_via"):
            r.error(f"bibliography '{key}': marked verified but records no "
                    f"verified_via source")
        if key not in cited and ctx.parts is None:
            r.note(f"bibliography '{key}' is never cited")
    n_ver = sum(1 for e in bib.values() if e.get("verified"))
    if bib:
        r.note(f"{n_ver}/{len(bib)} bibliography entries verified")
    return r


def gate_terminology(ctx: Context) -> Report:
    """Glossary and notation are internally consistent and actually used."""
    r = Report("terminology")
    terms = ctx.book.glossary
    syms = ctx.book.notation

    surfaces: dict[str, str] = {}
    for tid, e in terms.items():
        if not e.get("term"):
            r.error(f"glossary '{tid}': missing 'term'")
        if not e.get("intuitive"):
            r.error(f"glossary '{tid}': missing 'intuitive' definition")
        if not e.get("formal"):
            r.warn(f"glossary '{tid}': missing 'formal' definition")
        for name in [e.get("term", tid)] + list(e.get("aliases") or []):
            low = name.lower()
            if low in surfaces and surfaces[low] != tid:
                r.error(f"glossary: surface form '{name}' claimed by both "
                        f"'{surfaces[low]}' and '{tid}'")
            surfaces[low] = tid
        for ref in e.get("see_also") or []:
            if ref not in terms:
                r.error(f"glossary '{tid}': see_also '{ref}' is not a term")
        origin = e.get("introduced_in")
        if origin and origin not in ctx.book.by_id:
            r.error(f"glossary '{tid}': introduced_in '{origin}' is not a document id")

    for sid, e in syms.items():
        if not e.get("tex"):
            r.error(f"notation '{sid}': missing 'tex'")
        if not e.get("meaning"):
            r.error(f"notation '{sid}': missing 'meaning'")
    if terms:
        r.note(f"{len(terms)} glossary terms, {len(syms)} notation symbols")
    return r


def gate_rendering(ctx: Context) -> Report:
    """Maths and diagrams all rendered without falling back to error output."""
    r = Report("rendering")
    b = ctx.builder
    for tex, err in b.math.errors:
        r.error(f"KaTeX failed on {tex[:70]!r}: {err[:160]}")
    for src, err in b.mermaid.errors:
        r.error(f"Mermaid failed on {src[:70]!r}: {err[:200]}")
    return r


def gate_code(ctx: Context) -> Report:
    """Tier A listings execute; Tier B listings at least parse and resolve."""
    r = Report("code")
    b = ctx.builder
    blocks = [(rd.doc, blk) for rd in b.rendered.values()
              for blk in rd.code_blocks if ctx.in_scope(rd.doc)]
    if not blocks:
        return r

    b.extract_code_quiet()
    tier_a = [(d, blk) for d, blk in blocks if blk["tier"] == "A"]
    tier_b = [(d, blk) for d, blk in blocks if blk["tier"] == "B"]

    for d, blk in blocks:
        try:
            ast.parse(blk["source"])
        except SyntaxError as exc:
            r.error(f"{d.slug} [{blk['name']}]: syntax error line {exc.lineno}: {exc.msg}")

    passed = 0
    for d, blk in tier_a:
        path = ROOT / blk["path"]
        if not path.exists():
            r.error(f"{d.slug} [{blk['name']}]: extracted file missing")
            continue
        proc = subprocess.run([str(VENV_PY), str(path)], capture_output=True,
                              text=True, cwd=path.parent, timeout=1200)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().split("\n")[-4:]
            r.error(f"{d.slug} [{blk['name']}] tier A failed:\n        "
                    + "\n        ".join(tail))
        else:
            passed += 1

    # Tier B cannot run here (GPU, API keys, or large weights), but an import
    # that does not exist is still a bug the reader would hit immediately.
    for d, blk in tier_b:
        for mod in _imports(blk["source"]):
            if not _resolvable(mod):
                r.note(f"{d.slug} [{blk['name']}]: tier B imports '{mod}', "
                       f"not installed locally (expected for GPU/API code)")

    r.note(f"tier A: {passed}/{len(tier_a)} executed successfully; "
           f"tier B: {len(tier_b)} static-checked")
    return r


def _imports(src: str) -> set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


_RESOLVE_CACHE: dict[str, bool] = {}


def _resolvable(mod: str) -> bool:
    if mod not in _RESOLVE_CACHE:
        import importlib.util
        try:
            _RESOLVE_CACHE[mod] = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            _RESOLVE_CACHE[mod] = False
    return _RESOLVE_CACHE[mod]


def gate_maturity(ctx: Context) -> Report:
    """Named products and models carry an explicit maturity label somewhere."""
    r = Report("maturity")
    labels = set(ctx.book.cfg.get("maturity_labels", []))
    for d in ctx.chapters():
        _, body = split_frontmatter(d.path.read_text(encoding="utf-8"))
        used = set(re.findall(r"\{\{maturity:([^}]+)\}\}", body))
        bad = {u for u in used if u.strip() not in labels}
        for u in bad:
            r.error(f"{d.slug}: unknown maturity label '{u}'")
    return r


GATES = {
    "frontmatter": gate_frontmatter,
    "structure": gate_structure,
    "depth": gate_depth,
    "prereqs": gate_prereqs,
    "references": gate_references,
    "citations": gate_citations,
    "terminology": gate_terminology,
    "rendering": gate_rendering,
    "maturity": gate_maturity,
    "code": gate_code,
}


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Run the book's quality gates.")
    ap.add_argument("--part", type=int, action="append", dest="parts")
    ap.add_argument("--gate", action="append", dest="gates")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print informational notes")
    args = ap.parse_args()

    if args.list:
        for name, fn in GATES.items():
            print(f"  {name:14s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    names = args.gates or list(GATES)
    unknown = [n for n in names if n not in GATES]
    if unknown:
        print(f"unknown gate(s): {', '.join(unknown)}")
        return 2

    ctx = Context(args.parts)
    scope = f"part {', '.join(map(str, args.parts))}" if args.parts else "whole book"
    print(f"Checking {scope} — {len(ctx.docs())} written documents\n")

    n_err = n_warn = 0
    for name in names:
        rep = GATES[name](ctx)
        n_err += len(rep.errors)
        n_warn += len(rep.warnings)
        mark = "✓" if rep.ok else "✗"
        summary = []
        if rep.errors:
            summary.append(f"{len(rep.errors)} error(s)")
        if rep.warnings:
            summary.append(f"{len(rep.warnings)} warning(s)")
        print(f"  {mark} {name:14s} {', '.join(summary) if summary else 'clean'}")
        for msg in rep.errors:
            print(f"      ERROR  {msg}")
        for msg in rep.warnings:
            print(f"      warn   {msg}")
        if args.verbose:
            for msg in rep.notes:
                print(f"      note   {msg}")

    print()
    if n_err:
        print(f"FAILED — {n_err} error(s), {n_warn} warning(s)")
        return 1
    print(f"PASSED — 0 errors, {n_warn} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
