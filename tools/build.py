#!/usr/bin/env python3
"""
build.py — assemble the book.

  build.py html                 render the whole site into build/html
  build.py pdf [--part N ...]   render per-part PDFs (all parts by default)
  build.py merge                merge part PDFs into one book with a paged TOC
  build.py code                 extract code blocks into code/
  build.py all                  html + code + pdf + merge

Rendering happens once. The PDF path reuses the same HTML bodies, re-scoping
ids so an entire part can live in a single document without anchor collisions.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jinja2 import Environment, FileSystemLoader, select_autoescape
from lxml import html as lhtml

from bookdata import BUILD, DATA, ROOT, Book, Doc
from external import MathRenderer, MermaidRenderer
from generated import build_generated_docs
from render import DocRenderer, Rendered, ReferenceResolver

HTML_OUT = BUILD / "html"
PDF_OUT = BUILD / "pdf"
TEMPLATES = ROOT / "templates"
ASSETS = ROOT / "assets"
PAGEDJS = ROOT / "node_modules" / ".bin" / "pagedjs-cli"


# =============================================================================
# Link strategy
# =============================================================================

def href_html(from_doc: Doc, to_doc: Doc | None, anchor: str | None) -> str:
    if to_doc is None:
        return f"#{anchor}" if anchor else "#"
    if to_doc.id == from_doc.id:
        return f"#{anchor}" if anchor else "#top"
    return f"{to_doc.slug}.html" + (f"#{anchor}" if anchor else "")


_ID_ATTR = re.compile(r'\bid="([^"]+)"')


def pdfify(body: str, slug: str) -> str:
    """Re-scope ids and links so many documents can share one PDF document.

    SVG subtrees are left alone. Mermaid scopes its own stylesheet to the svg
    root's id (`#mmd-<hash> .node rect { … }`), so renaming that id silently
    detaches every rule and the diagram renders with initial values — black
    fills on black strokes. Those ids are already content-hash unique, so they
    need no extra namespacing.
    """
    tree = lhtml.fragment_fromstring(body, create_parent="div")
    for el in tree.xpath("//*[@id][not(ancestor-or-self::svg)]"):
        el.set("id", f"{slug}--{el.get('id')}")
    for a in tree.xpath("//a[@href][not(ancestor::svg)]"):
        href = a.get("href")
        if href.startswith("#"):
            a.set("href", f"#{slug}--{href[1:]}")
        elif href.endswith(".html"):
            a.set("href", f"#{href[:-5]}--top")
        elif ".html#" in href:
            f, _, frag = href.partition(".html#")
            a.set("href", f"#{f}--{frag}")
    return DocRenderer._serialise(tree)


# =============================================================================
# Build
# =============================================================================

class Builder:
    def __init__(self) -> None:
        self.book = Book()
        self.math = MathRenderer()
        self.mermaid = MermaidRenderer()
        self.rendered: dict[str, Rendered] = {}
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES)),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True, lstrip_blocks=True,
        )
        self.env.globals["book"] = self.book.cfg
        self.env.globals["build_date"] = date.today().isoformat()

    # -- phase 1 + 2 ----------------------------------------------------------

    def render_all(self, verbose: bool = True) -> None:
        t0 = time.time()
        docs = [d for d in self.book.docs if d.exists and not d.generated]
        for d in docs:
            r = DocRenderer(self.book, d, self.math, self.mermaid).render()
            self.rendered[d.id] = r
            if verbose:
                print(f"  rendered {d.slug:44s} {r.words:>6,} words "
                      f"{len(r.code_blocks):>2} code")

        # Generated documents (TOC, glossary, bibliography, notation) are built
        # from the same data the checks read, so they cannot drift.
        for r in build_generated_docs(self.book, self.rendered, self.math):
            self.rendered[r.doc.id] = r

        self.resolver = ReferenceResolver(self.book, self.rendered)
        for r in self.rendered.values():
            r.body = self.resolver.resolve(r, href_html)

        if verbose:
            print(f"  phase 2 resolved refs in {time.time() - t0:.1f}s "
                  f"({len(self.rendered)} documents)")
            if self.mermaid.rendered or self.mermaid.cache_hits:
                print(f"  diagrams: {self.mermaid.rendered} rendered, "
                      f"{self.mermaid.cache_hits} cached")

    # -- html site ------------------------------------------------------------

    def build_html(self) -> None:
        HTML_OUT.mkdir(parents=True, exist_ok=True)
        for sub in ("css", "js", "fonts"):
            src = ASSETS / sub
            if src.exists():
                shutil.copytree(src, HTML_OUT / sub, dirs_exist_ok=True)
        # KaTeX ships its own stylesheet and web fonts; vendor them so the site
        # renders maths with no network access.
        katex_dist = ROOT / "node_modules" / "katex" / "dist"
        if katex_dist.exists():
            dest = HTML_OUT / "vendor" / "katex"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(katex_dist / "katex.min.css", dest / "katex.min.css")
            shutil.copytree(katex_dist / "fonts", dest / "fonts", dirs_exist_ok=True)

        nav = self._nav()
        page = self.env.get_template("page.html.j2")

        for r in self.rendered.values():
            prev, nxt = self.book.neighbours(r.doc)
            out = page.render(
                doc=r.doc, body=r.body, toc=r.toc, nav=nav,
                prev=prev, next=nxt, words=r.words,
                part=self.book.part(r.doc.part) if r.doc.part else None,
            )
            (HTML_OUT / r.doc.out_name).write_text(out, encoding="utf-8")

        index = self.env.get_template("index.html.j2")
        (HTML_OUT / "index.html").write_text(
            index.render(nav=nav, stats=self.stats()), encoding="utf-8")

        (HTML_OUT / "search-index.json").write_text(
            json.dumps(self._search_index(), separators=(",", ":")),
            encoding="utf-8")
        print(f"  wrote {len(self.rendered) + 1} html files -> {HTML_OUT}")

    def _nav(self) -> list[dict[str, Any]]:
        """Sidebar tree: front matter, parts with their docs, capstone, appendices."""
        groups: list[dict[str, Any]] = []

        def entry(d: Doc) -> dict[str, Any]:
            return {"id": d.id, "title": d.title, "label": d.label,
                    "href": f"{d.slug}.html", "kind": d.kind,
                    "exists": d.id in self.rendered,
                    "number": d.number, "status": d.status}

        fm = [d for d in self.book.docs if d.kind == "frontmatter"]
        groups.append({"title": "Front Matter", "roman": "", "items": [entry(d) for d in fm]})

        for p in self.book.parts:
            items = [entry(d) for d in p.docs]
            items += [entry(d) for d in self.book.docs
                      if d.kind == "project" and d.part == p.number]
            groups.append({"title": p.title, "roman": p.roman,
                           "number": p.number, "summary": p.summary, "items": items})

        cap = [d for d in self.book.docs if d.kind == "capstone"]
        if cap:
            groups.append({"title": self.book.cfg["capstone"]["title"],
                           "roman": "", "items": [entry(d) for d in cap]})
        app = [d for d in self.book.docs if d.kind == "appendix"]
        groups.append({"title": "Appendices", "roman": "",
                       "items": [entry(d) for d in app]})
        return groups

    def _search_index(self) -> list[dict[str, Any]]:
        idx = []
        for r in self.rendered.values():
            tree = lhtml.fragment_fromstring(r.body, create_parent="div")
            for el in tree.iter("pre", "svg", "script", "style"):
                el.clear()
            text = re.sub(r"\s+", " ", " ".join(tree.itertext())).strip()
            idx.append({
                "u": r.doc.out_name, "t": r.doc.title, "l": r.doc.label,
                "h": [h["text"] for h in r.toc],
                "b": text[:14000],
            })
        return idx

    # -- pdf ------------------------------------------------------------------

    def build_pdf(self, parts: list[int] | None = None) -> list[Path]:
        PDF_OUT.mkdir(parents=True, exist_ok=True)
        tpl = self.env.get_template("pdf.html.j2")
        outputs: list[Path] = []

        for group in self._pdf_groups(parts):
            docs = [self.rendered[d.id] for d in group["docs"] if d.id in self.rendered]
            if not docs:
                continue
            bodies = [{"doc": r.doc, "body": pdfify(r.body, r.doc.slug)} for r in docs]
            html_path = BUILD / "cache" / f"pdf-{group['key']}.html"
            html_path.write_text(
                tpl.render(
                    title=group["title"], subtitle=group.get("subtitle", ""),
                    kicker=group.get("kicker", ""),
                    opener_title=group.get("opener_title", group["title"]),
                    is_front=(group["key"] == "front-matter"),
                    sections=bodies, css=self._inline_css(),
                    katex_css=ROOT / "node_modules" / "katex" / "dist" / "katex.min.css",
                ),
                encoding="utf-8")

            pdf_path = PDF_OUT / f"{group['key']}.pdf"
            if self._run_paged(html_path, pdf_path):
                outputs.append(pdf_path)
                print(f"  pdf {pdf_path.name:32s} {self._pages(pdf_path):>5} pages")
        return outputs

    def _pdf_groups(self, parts: list[int] | None) -> list[dict[str, Any]]:
        groups = []
        if parts is None:
            fm = [d for d in self.book.docs if d.kind == "frontmatter"]
            if any(d.id in self.rendered for d in fm):
                groups.append({"key": "front-matter", "title": self.book.cfg["title"],
                               "subtitle": self.book.cfg["subtitle"], "docs": fm})
        for p in self.book.parts:
            if parts is not None and p.number not in parts:
                continue
            docs = list(p.docs) + [d for d in self.book.docs
                                   if d.kind == "project" and d.part == p.number]
            groups.append({"key": f"part-{p.number:02d}",
                           "title": f"Part {p.roman} — {p.title}",
                           "kicker": f"Part {p.roman}", "opener_title": p.title,
                           "subtitle": p.summary, "docs": docs})
        if parts is None:
            cap = [d for d in self.book.docs if d.kind == "capstone"]
            if any(d.id in self.rendered for d in cap):
                groups.append({"key": "capstone",
                               "title": self.book.cfg["capstone"]["title"], "docs": cap})
            app = [d for d in self.book.docs if d.kind == "appendix"]
            if any(d.id in self.rendered for d in app):
                groups.append({"key": "appendices", "title": "Appendices", "docs": app})
        return groups

    def _inline_css(self) -> str:
        parts = []
        for name in ("book.css", "print.css"):
            f = ASSETS / "css" / name
            if f.exists():
                parts.append(f.read_text(encoding="utf-8"))
        return "\n".join(parts)

    def _run_paged(self, src: Path, out: Path) -> bool:
        cmd = [str(PAGEDJS), str(src), "-o", str(out),
               "--browserArgs", "--no-sandbox,--disable-dev-shm-usage",
               "--outline-tags", "h1,h2", "-t", "180000"]
        env = {"PUPPETEER_EXECUTABLE_PATH": "/usr/bin/google-chrome",
               "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
        if proc.returncode != 0 or not out.exists():
            print(f"  ! pagedjs failed for {src.name}:\n"
                  f"{(proc.stderr or proc.stdout)[-1500:]}")
            return False
        return True

    @staticmethod
    def _pages(pdf: Path) -> int:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)

    def merge_pdf(self) -> Path | None:
        """Concatenate the part PDFs and add a bookmark outline."""
        from pypdf import PdfWriter, PdfReader
        order = ["front-matter"] + [f"part-{p.number:02d}" for p in self.book.parts] \
                + ["capstone", "appendices"]
        files = [PDF_OUT / f"{k}.pdf" for k in order if (PDF_OUT / f"{k}.pdf").exists()]
        if not files:
            print("  nothing to merge")
            return None
        writer = PdfWriter()
        for f in files:
            start = len(writer.pages)
            reader = PdfReader(str(f))
            for pg in reader.pages:
                writer.add_page(pg)
            writer.add_outline_item(f.stem.replace("-", " ").title(), start)
        out = PDF_OUT / "machines-that-learn-and-act.pdf"
        with out.open("wb") as fh:
            writer.write(fh)
        print(f"  merged {len(files)} PDFs -> {out.name} ({len(writer.pages)} pages)")
        return out

    # -- code extraction ------------------------------------------------------

    def extract_code_quiet(self) -> int:
        return self.extract_code(verbose=False)

    def extract_code(self, verbose: bool = True) -> int:
        n = 0
        for r in self.rendered.values():
            for blk in r.code_blocks:
                path = ROOT / blk["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                header = (f"# Extracted from: {r.doc.label} — {r.doc.title}\n"
                          f"# Source: src/.../{r.doc.slug}.md   Tier: {blk['tier']}\n"
                          f"# Regenerate with: make code  (do not edit by hand)\n\n")
                path.write_text(header + blk["source"], encoding="utf-8")
                n += 1
        if verbose:
            print(f"  extracted {n} code files -> code/")
        return n

    # -- manifest -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        words = sum(r.words for r in self.rendered.values())
        chapters = self.book.chapters
        written = [c for c in chapters if c.id in self.rendered]
        return {
            "chapters_total": len(chapters),
            "chapters_written": len(written),
            "words": words,
            "documents": len(self.rendered),
            "diagrams": self.mermaid.rendered + self.mermaid.cache_hits,
            "code_blocks": sum(len(r.code_blocks) for r in self.rendered.values()),
        }

    def write_manifest(self) -> None:
        rows = []
        for d in self.book.docs:
            r = self.rendered.get(d.id)
            rows.append({
                "id": d.id, "kind": d.kind, "part": d.part, "number": d.number,
                "title": d.title, "slug": d.slug, "tier": d.tier,
                "status": d.status, "exists": d.exists,
                "words": r.words if r else 0,
                "code": len(r.code_blocks) if r else 0,
                "citations": sorted(r.citations) if r else [],
                "requires": d.meta.get("requires", []),
                "provides": d.meta.get("provides", []),
            })
        DATA.mkdir(exist_ok=True)
        (DATA / "manifest.json").write_text(
            json.dumps({"generated": date.today().isoformat(),
                        "stats": self.stats(), "documents": rows}, indent=2),
            encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Build the book.")
    ap.add_argument("target", choices=["html", "pdf", "merge", "code", "all"])
    ap.add_argument("--part", type=int, action="append", dest="parts")
    args = ap.parse_args()

    b = Builder()
    print(f"Building «{b.book.cfg['title']}»")
    b.render_all()

    if args.target in ("html", "all"):
        b.build_html()
    if args.target in ("code", "all"):
        b.extract_code()
    if args.target in ("pdf", "all"):
        b.build_pdf(args.parts)
    if args.target in ("merge", "all") and not args.parts:
        b.merge_pdf()

    b.write_manifest()
    s = b.stats()
    print(f"\n  {s['chapters_written']}/{s['chapters_total']} chapters · "
          f"{s['words']:,} words · {s['diagrams']} diagrams · "
          f"{s['code_blocks']} code blocks")

    if b.math.errors:
        print(f"  ! {len(b.math.errors)} KaTeX errors")
        for tex, err in b.math.errors[:5]:
            print(f"      {tex[:60]!r}: {err[:100]}")
    if b.mermaid.errors:
        print(f"  ! {len(b.mermaid.errors)} Mermaid errors")
        for src, err in b.mermaid.errors[:3]:
            print(f"      {src[:60]!r}: {err[:200]}")
    if b.resolver.dangling:
        print(f"  ! {len(b.resolver.dangling)} dangling references")
        for doc, ref in b.resolver.dangling[:10]:
            print(f"      {doc}: {ref}")
    if b.resolver.pending:
        print(f"  · {len(b.resolver.pending)} references to chapters not yet "
              f"written (rendered as plain text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
