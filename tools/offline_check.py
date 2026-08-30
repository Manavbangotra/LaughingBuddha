#!/usr/bin/env python3
"""Offline stand-in for tools/check.py gates that do not need node or lxml.

Covers: frontmatter, structure, depth (approximate prose word count),
prereqs, references, citations, maturity. Rendering and the code gate are run
separately (the listings are executed directly).
"""
from __future__ import annotations
import re, sys, argparse
from pathlib import Path

BOOK = Path("C:/Github/LaughingBuddha")
sys.path.insert(0, str(BOOK / "tools"))
from bookdata import Book, split_frontmatter  # noqa: E402

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

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def strip_code(body):
    """Return (prose-only text, [(info, source)]) for fenced blocks."""
    out, blocks, fence, buf, info = [], [], None, [], ""
    for line in body.split("\n"):
        m = FENCE.match(line)
        if m:
            tok = m.group(1)[0] * 3
            if fence and line.strip().startswith(fence):
                blocks.append((info, "\n".join(buf)))
                fence, buf, info = None, [], ""
            elif not fence:
                fence, info = tok, line.strip().lstrip("`~")
            continue
        (buf if fence else out).append(line)
    return "\n".join(out), blocks


def prose_words(body):
    text, _ = strip_code(body)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", " ", text)
    text = re.sub(r"\{\{[a-z]+:([^}]*)\}\}", r" \1 ", text)
    text = re.sub(r"\(eq:[\w-]+\)", " ", text)
    return len(text.split())


def slug(text):
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_]+", "-", s) or "section"


def headings(body):
    text, _ = strip_code(body)
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, action="append", dest="parts")
    ap.add_argument("--words", action="store_true", help="print prose word counts")
    args = ap.parse_args()

    book = Book()
    docs = [d for d in book.docs if d.exists and not d.generated]
    scope = [d for d in docs if args.parts is None or d.part in args.parts]
    errors, warnings = [], []

    labels = set()
    provider = {}
    for d in docs:
        raw = d.path.read_text(encoding="utf-8")
        _, body = split_frontmatter(raw)
        text, blocks = strip_code(body)
        for name in re.findall(r"\(\s*(eq:[\w-]+)\s*\)", body):
            labels.add(name)
        for info, _src in blocks:
            m = re.search(r"\{#(fig:[\w-]+)", info)
            if m:
                labels.add(m.group(1))
        for name in re.findall(r"^\{#(tbl:[\w-]+)", text, flags=re.M):
            labels.add(name)
        seen = set()
        for lvl, h in headings(body):
            if lvl == 2:
                s, base, n = slug(h), slug(h), 2
                while s in seen:
                    s = "%s-%d" % (base, n)
                    n += 1
                seen.add(s)
                labels.add("sec:" + s)
        for c in (d.meta.get("provides") or []):
            if c in provider:
                errors.append("concept '%s' provided by both %s and %s"
                              % (c, provider[c].slug, d.slug))
            provider[c] = d

    bib = book.bibliography
    maturity_labels = set(book.cfg.get("maturity_labels", []))

    for d in scope:
        raw = d.path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(raw)
        tag = d.slug

        if d.kind == "chapter":
            for key in ("id", "number", "part", "tier", "status"):
                if key not in meta:
                    errors.append("%s: frontmatter missing '%s'" % (tag, key))
            if meta.get("id") != d.id:
                errors.append("%s: id '%s' != book.yaml '%s'" % (tag, meta.get("id"), d.id))
            if meta.get("number") != d.number:
                errors.append("%s: number %s != computed %s" % (tag, meta.get("number"), d.number))
            if meta.get("tier") not in (None, d.tier):
                errors.append("%s: tier '%s' != part tier '%s'" % (tag, meta.get("tier"), d.tier))
            if meta.get("status") not in {"draft", "reviewed", "final"}:
                errors.append("%s: bad status '%s'" % (tag, meta.get("status")))

            want = TEMPLATE.get(d.tier or "full", SECTIONS_FULL)
            got = [re.sub(r"^\d+\.\s*", "", h).strip()
                   for lvl, h in headings(body) if lvl == 2]
            missing = [s for s in want if s not in got]
            if missing:
                errors.append("%s: missing sections: %s" % (tag, ", ".join(missing)))
            order = [got.index(s) for s in want if s in got]
            if order != sorted(order):
                errors.append("%s: sections out of order" % tag)
            extra = [s for s in got if s not in want]
            if extra:
                warnings.append("%s: extra sections: %s" % (tag, ", ".join(extra)))

            floor = book.tier_spec(d.tier or "full")["min_words"]
            w = prose_words(body)
            if args.words:
                print("  %-34s %6d prose words (floor %d)" % (tag, w, floor))
            if w < floor:
                warnings.append("%s: ~%d prose words below the %s floor of %d"
                                % (tag, w, d.tier, floor))

        for req in (d.meta.get("requires") or []):
            src = book.by_id.get(req) or provider.get(req)
            if src is None:
                errors.append("%s: requires '%s' which nothing provides" % (tag, req))
            elif src.order > d.order:
                errors.append("%s: requires '%s' from %s, later in reading order"
                              % (tag, req, src.slug))

        text, _ = strip_code(body)
        for kind, arg in re.findall(r"\{\{([a-z]+):([^}]+)\}\}", text):
            arg = arg.strip()
            if kind in ("ch", "proj"):
                if arg not in book.by_id:
                    errors.append("%s: {{%s:%s}} — no such document id" % (tag, kind, arg))
            elif kind == "part":
                try:
                    book.part(int(arg))
                except Exception:
                    errors.append("%s: {{part:%s}} — no such part" % (tag, arg))
            elif kind == "cite":
                for key in [k.strip() for k in arg.split(",")]:
                    if key not in bib:
                        errors.append("%s: cites '%s', absent from bibliography" % (tag, key))
                    elif not bib[key].get("verified"):
                        warnings.append("%s: cites unverified '%s'" % (tag, key))
            elif kind == "term":
                if arg not in book.glossary:
                    errors.append("%s: {{term:%s}} — not a glossary id" % (tag, arg))
            elif kind == "maturity":
                if arg not in maturity_labels:
                    errors.append("%s: unknown maturity label '%s'" % (tag, arg))
            elif kind in ("eq", "fig", "tbl", "sec"):
                if "%s:%s" % (kind, arg) not in labels:
                    errors.append("%s: dangling {{%s:%s}}" % (tag, kind, arg))
            else:
                errors.append("%s: unknown macro kind '%s'" % (tag, kind))

        for key in (d.meta.get("citations") or []):
            if key not in bib:
                errors.append("%s: frontmatter cites '%s', absent from bibliography" % (tag, key))

    print("\nlint: %d document(s) in scope\n" % len(scope))
    for e in errors:
        print("  ERROR  " + e)
    for w in warnings:
        print("  warn   " + w)
    print("\n%s — %d error(s), %d warning(s)"
          % ("FAILED" if errors else "PASSED", len(errors), len(warnings)))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
