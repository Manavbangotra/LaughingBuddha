#!/usr/bin/env python3
"""
report.py — progress dashboard.

Prints where the book stands: which parts are written, how many words, how much
code is verified, how many citations are checked, and what the next unwritten
chapter is. Reads data/manifest.json when it is current, otherwise renders.

  report.py            summary by part
  report.py --part 7   chapter-level detail for one part
  report.py --todo     the next chapters to write, in reading order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bookdata import Book
from build import Builder

BAR = "█"
DIM = "·"


def bar(frac: float, width: int = 18) -> str:
    filled = round(frac * width)
    return BAR * filled + DIM * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser(description="Book progress dashboard.")
    ap.add_argument("--part", type=int)
    ap.add_argument("--todo", action="store_true")
    args = ap.parse_args()

    book = Book()
    b = Builder()
    b.render_all(verbose=False)

    if args.todo:
        return _todo(book, b)
    if args.part:
        return _part_detail(book, b, args.part)
    return _summary(book, b)


def _summary(book: Book, b: Builder) -> int:
    print(f"\n  {book.cfg['title']}")
    print(f"  {book.cfg['subtitle']}\n")

    total_words = total_written = total_chapters = 0
    rows = []
    for p in book.parts:
        written = [c for c in p.chapters if c.id in b.rendered]
        words = sum(b.rendered[c.id].words for c in written)
        rows.append((p, len(written), len(p.chapters), words))
        total_words += words
        total_written += len(written)
        total_chapters += len(p.chapters)

    print(f"  {'PART':<6} {'TITLE':<34} {'CHAPTERS':>9}  {'':18} {'WORDS':>9}")
    print("  " + "─" * 82)
    for p, n, tot, words in rows:
        frac = n / tot if tot else 0
        mark = "✓" if n == tot else " "
        print(f"  {p.roman:<6} {p.title[:34]:<34} {n:>4}/{tot:<4} "
              f"{bar(frac)} {words:>9,} {mark}")
    print("  " + "─" * 82)

    extras = [d for d in book.docs
              if d.kind in ("project", "capstone", "appendix", "frontmatter")]
    extras_written = [d for d in extras if d.id in b.rendered]
    extra_words = sum(b.rendered[d.id].words for d in extras_written)

    frac = total_written / total_chapters if total_chapters else 0
    print(f"  {'':6} {'Chapters':<34} {total_written:>4}/{total_chapters:<4} "
          f"{bar(frac)} {total_words:>9,}")
    print(f"  {'':6} {'Front matter, projects, appendices':<34} "
          f"{len(extras_written):>4}/{len(extras):<4} "
          f"{bar(len(extras_written)/len(extras) if extras else 0)} {extra_words:>9,}")

    grand = total_words + extra_words
    pages = round(grand / 340)
    code_a = code_b = 0
    for r in b.rendered.values():
        for blk in r.code_blocks:
            if blk["tier"] == "A":
                code_a += 1
            elif blk["tier"] == "B":
                code_b += 1

    bib = book.bibliography
    verified = sum(1 for e in bib.values() if e.get("verified"))

    print()
    print(f"  Total          {grand:>9,} words  ≈ {pages:,} pages")
    print(f"  Diagrams       {b.mermaid.rendered + b.mermaid.cache_hits:>9,}")
    print(f"  Code listings  {code_a + code_b:>9,}   "
          f"({code_a} executed, {code_b} static-checked)")
    print(f"  Glossary       {len(book.glossary):>9,} terms")
    print(f"  Notation       {len(book.notation):>9,} symbols")
    print(f"  Bibliography   {len(bib):>9,} entries   ({verified} verified)")
    print()

    nxt = _next_unwritten(book, b)
    if nxt:
        print(f"  Next to write: {nxt.label} — {nxt.title}")
        print(f"                 src/{nxt.path.relative_to(nxt.path.parents[1])}")
    else:
        print("  Every planned document is written.")
    print()
    return 0


def _part_detail(book: Book, b: Builder, number: int) -> int:
    p = book.part(number)
    print(f"\n  Part {p.roman} — {p.title}   [{p.tier} tier]\n")
    floor = book.tier_spec(p.tier)["min_words"]
    print(f"  {'#':>4}  {'CHAPTER':<50} {'WORDS':>8} {'STATUS':<10} CODE")
    print("  " + "─" * 88)
    for c in p.chapters:
        r = b.rendered.get(c.id)
        if r is None:
            print(f"  {c.number:>4}  {c.title[:50]:<50} {'—':>8} {'unwritten':<10}")
            continue
        flag = " " if r.words >= floor else "!"
        code = "".join(sorted(blk["tier"] for blk in r.code_blocks))
        print(f"  {c.number:>4}  {c.title[:50]:<50} {r.words:>8,}{flag}"
              f"{c.status:<10} {code}")
    print("  " + "─" * 88)
    projects = [d for d in book.docs if d.kind == "project" and d.part == number]
    for d in projects:
        r = b.rendered.get(d.id)
        print(f"  {'P':>4}  {d.title[:50]:<50} "
              f"{(f'{r.words:,}' if r else '—'):>8}  {d.status}")
    print()
    return 0


def _todo(book: Book, b: Builder) -> int:
    print("\n  Next unwritten documents, in reading order:\n")
    n = 0
    for d in book.docs:
        if d.generated or d.id in b.rendered:
            continue
        rel = d.path.relative_to(d.path.parents[2]) if len(d.path.parents) > 2 else d.path
        print(f"  {d.label:<14} {d.title[:52]:<52} src/{rel}")
        n += 1
        if n >= 25:
            print("  …")
            break
    print()
    return 0


def _next_unwritten(book: Book, b: Builder):
    for d in book.docs:
        if not d.generated and d.id not in b.rendered:
            return d
    return None


if __name__ == "__main__":
    raise SystemExit(main())
