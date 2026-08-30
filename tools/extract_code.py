#!/usr/bin/env python3
"""Standalone stand-in for `make code`, for environments without markdown-it.

tools/build.py imports lxml at module scope, so `build.py code` cannot run here.
This reproduces the same extraction contract exactly:

  * only ```python / ```py fences whose attrs carry tier A or B are extracted
  * the index in the filename counts extracted blocks only, 1-based, per doc
  * path   = code/part-NN/<slug>/<nn>-<stem>.py
  * stem   = name attr, else snippet-<nn>
  * header = the four-line banner build.py writes, including the PEP 263 cookie

Doc metadata comes from tools/bookdata.py, so slugs, titles and labels are the
real ones rather than a re-derivation. Verified by regenerating the parts that
were extracted with the real toolchain and diffing byte-for-byte.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
while not (ROOT / "book.yaml").exists():
    if ROOT.parent == ROOT:
        sys.exit("book.yaml not found (pass the repo root as argv[1])")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "tools"))
from bookdata import Book  # noqa: E402

CODE_DIR = ROOT / "code"

_ATTR_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
_ID_RE = re.compile(r"#([A-Za-z0-9][\w:-]*)")
_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")


def parse_attrs(info: str) -> dict[str, str]:
    """Verbatim from tools/render.py."""
    attrs: dict[str, str] = {}
    body = info.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    for m in _ID_RE.finditer(body):
        attrs["id"] = m.group(1)
    for m in _ATTR_RE.finditer(body):
        attrs[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return attrs


def split_info(info: str) -> tuple[str, dict[str, str]]:
    """Verbatim from tools/render.py."""
    info = info.strip()
    if not info:
        return "", {}
    brace = info.find("{")
    if brace == -1:
        parts = info.split(None, 1)
        return parts[0], (parse_attrs(parts[1]) if len(parts) > 1 else {})
    return info[:brace].strip(), parse_attrs(info[brace:])


def fences(text: str):
    """Yield (info, source) for every fenced block, CommonMark-style.

    A closing fence is the same character, at least as long as the opener, with
    nothing after it. Anything else inside the block is content, so a nested
    ``` inside a python listing does not terminate it early.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = _OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, marker, info = m.group(1), m.group(2), m.group(3)
        char, length = marker[0], len(marker)
        if char == "`" and "`" in info:
            i += 1                      # inline code run, not a fence
            continue
        body: list[str] = []
        i += 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            closing = (stripped
                       and stripped == char * len(stripped)
                       and len(stripped) >= length
                       and len(line) - len(line.lstrip(" ")) <= 3)
            if closing:
                break
            body.append(line[len(indent):] if line.startswith(indent) else line)
            i += 1
        i += 1
        yield info, "\n".join(body) + "\n"


def main() -> int:
    book = Book(ROOT / "book.yaml")
    docs = [d for d in book.docs if d.exists]
    written, per_part = 0, {}

    for doc in docs:
        text = doc.path.read_text(encoding="utf-8")
        n = 0
        for info, source in fences(text):
            lang, attrs = split_info(info)
            tier = attrs.get("tier", "C").upper()
            if lang not in ("python", "py") or tier not in ("A", "B"):
                continue
            n += 1
            stem = attrs.get("name", "") or f"snippet-{n:02d}"
            part = f"part-{doc.part:02d}" if doc.part else "misc"
            path = CODE_DIR / part / doc.slug / f"{n:02d}-{stem}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            header = (f"# -*- coding: utf-8 -*-\n"
                      f"# Extracted from: {doc.label} — {doc.title}\n"
                      f"# Source: src/.../{doc.slug}.md   Tier: {tier}\n"
                      f"# Regenerate with: make code  (do not edit by hand)\n\n")
            path.write_text(header + source, encoding="utf-8")
            written += 1
            per_part[part] = per_part.get(part, 0) + 1

    for part in sorted(per_part):
        print(f"  {part}  {per_part[part]:>4} files")
    print(f"\n  extracted {written} code files -> code/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
