"""
bookdata.py — the book's structural model.

Loads book.yaml, discovers chapter source files, computes the canonical reading
order and global numbering, and exposes lookup tables that every other tool
(renderer, checks, report) builds on. Nothing else in the toolchain is allowed
to decide chapter order or numbering; it all comes from here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
BUILD = ROOT / "build"

# --- frontmatter -------------------------------------------------------------

_FM_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a YAML frontmatter block off the top of a Markdown document.

    Returns (metadata, body). A document with no frontmatter yields ({}, text)
    rather than raising, so partially written chapters still render.
    """
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    # Keep line numbers aligned with the source file so error messages point at
    # the right line: replace the frontmatter with an equal number of newlines.
    consumed = text[: m.end()]
    return meta, "\n" * consumed.count("\n") + text[m.end() :]


# --- document model ----------------------------------------------------------

ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
    9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI",
    22: "XXII", 23: "XXIII", 24: "XXIV", 25: "XXV", 26: "XXVI", 27: "XXVII",
    28: "XXVIII",
}


@dataclass
class Doc:
    """One rendered unit of the book: a chapter, project, appendix, or divider."""

    id: str
    slug: str
    title: str
    kind: str            # chapter | project | capstone | appendix | part-intro
                         # | assessment | frontmatter
    path: Path
    part: int | None = None      # owning part number, when applicable
    tier: str | None = None      # focused | full  (chapters only)
    number: int | None = None    # global chapter number (chapters only)
    order: int = 0               # position in reading order
    generated: bool = False      # produced by the build, not hand-written

    # populated from the file's own frontmatter once it exists
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def label(self) -> str:
        """Human-facing designator used in headings and cross-references."""
        if self.kind == "chapter":
            return f"Chapter {self.number}"
        if self.kind == "project":
            return f"Project {self.id.split('-')[-1].lstrip('0')}"
        if self.kind == "capstone":
            return "Capstone"
        if self.kind == "appendix":
            return f"Appendix {self.appendix_letter}"
        if self.kind == "part-intro":
            return f"Part {ROMAN[self.part]}"
        return self.title

    appendix_letter: str = ""

    @property
    def out_name(self) -> str:
        return f"{self.slug}.html"

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        return str(self.meta.get("status", "draft"))


@dataclass
class Part:
    number: int
    roman: str
    dir: str
    title: str
    tier: str
    summary: str
    chapters: list[Doc] = field(default_factory=list)
    intro: Doc | None = None
    assessment: Doc | None = None

    @property
    def docs(self) -> list[Doc]:
        out: list[Doc] = []
        if self.intro:
            out.append(self.intro)
        out.extend(self.chapters)
        if self.assessment:
            out.append(self.assessment)
        return out


class Book:
    """The whole book: configuration plus the ordered document list."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or (ROOT / "book.yaml")
        self.cfg: dict[str, Any] = yaml.safe_load(self.config_path.read_text())
        self.parts: list[Part] = []
        self.docs: list[Doc] = []
        self.by_id: dict[str, Doc] = {}
        self._build_structure()
        self._load_meta()

    # -- construction ---------------------------------------------------------

    def _build_structure(self) -> None:
        order = 0
        chapter_no = 0

        def add(doc: Doc) -> Doc:
            nonlocal order
            doc.order = order
            order += 1
            if doc.id in self.by_id:
                raise ValueError(f"duplicate document id: {doc.id}")
            self.by_id[doc.id] = doc
            self.docs.append(doc)
            return doc

        # Front matter -------------------------------------------------------
        for fid, slug, title in [
            ("fm-title", "00-title", "Title Page"),
            ("fm-preface", "01-preface", "Preface"),
            ("fm-how-to-read", "02-how-to-read", "How to Read This Book"),
            ("fm-toc", "03-contents", "Contents"),
            ("fm-architecture", "04-architecture", "Book Architecture and Dependency Graph"),
            ("fm-notation", "05-notation", "Notation and Conventions"),
        ]:
            add(Doc(id=fid, slug=slug, title=title, kind="frontmatter",
                    path=SRC / "frontmatter" / f"{slug}.md",
                    generated=(fid == "fm-toc")))

        # Projects grouped by the part they follow -----------------------------
        projects_by_part: dict[int, list[dict[str, Any]]] = {}
        for p in self.cfg.get("projects", []):
            projects_by_part.setdefault(int(p["after_part"]), []).append(p)

        # Parts ---------------------------------------------------------------
        for pc in self.cfg["parts"]:
            part = Part(
                number=int(pc["number"]), roman=pc["roman"], dir=pc["dir"],
                title=pc["title"], tier=pc["tier"],
                summary=(pc.get("summary") or "").strip(),
            )
            pdir = SRC / part.dir

            part.intro = add(Doc(
                id=f"part-{part.number:02d}-intro", slug=f"{part.dir}-intro",
                title=part.title, kind="part-intro", path=pdir / "_part.md",
                part=part.number,
            ))

            for cc in pc["chapters"]:
                chapter_no += 1
                part.chapters.append(add(Doc(
                    id=cc["id"], slug=cc["slug"], title=cc["title"],
                    kind="chapter", path=pdir / f"{cc['slug']}.md",
                    part=part.number, tier=part.tier, number=chapter_no,
                )))

            part.assessment = add(Doc(
                id=f"part-{part.number:02d}-assessment",
                slug=f"{part.dir}-assessment",
                title=f"Part {part.roman} — Knowledge Check and Assignments",
                kind="assessment", path=pdir / "_assessment.md",
                part=part.number,
            ))

            for p in projects_by_part.get(part.number, []):
                add(Doc(id=p["id"], slug=p["slug"], title=p["title"],
                        kind="project", path=SRC / "projects" / f"{p['slug']}.md",
                        part=part.number))

            self.parts.append(part)

        # Capstone -------------------------------------------------------------
        for cc in self.cfg["capstone"]["chapters"]:
            add(Doc(id=cc["id"], slug=cc["slug"], title=cc["title"],
                    kind="capstone", path=SRC / "capstone" / f"{cc['slug']}.md"))

        # Appendices -----------------------------------------------------------
        for i, ac in enumerate(self.cfg["appendices"]):
            d = Doc(id=ac["id"], slug=ac["slug"], title=ac["title"],
                    kind="appendix", path=SRC / "appendices" / f"{ac['slug']}.md",
                    generated=bool(ac.get("generated")))
            d.appendix_letter = chr(ord("A") + i)
            add(d)

        self.n_chapters = chapter_no

    def _load_meta(self) -> None:
        for d in self.docs:
            if d.exists:
                try:
                    d.meta, _ = split_frontmatter(d.path.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001 - surfaced by checks
                    d.meta = {"_frontmatter_error": str(exc)}

    # -- lookup ---------------------------------------------------------------

    @property
    def chapters(self) -> list[Doc]:
        return [d for d in self.docs if d.kind == "chapter"]

    @property
    def written(self) -> list[Doc]:
        return [d for d in self.docs if d.exists]

    def part(self, number: int) -> Part:
        for p in self.parts:
            if p.number == number:
                return p
        raise KeyError(f"no part {number}")

    def neighbours(self, doc: Doc) -> tuple[Doc | None, Doc | None]:
        """Previous and next *existing* documents in reading order."""
        existing = [d for d in self.docs if d.exists or d.generated]
        try:
            i = existing.index(doc)
        except ValueError:
            return None, None
        return (existing[i - 1] if i > 0 else None,
                existing[i + 1] if i + 1 < len(existing) else None)

    def tier_spec(self, tier: str) -> dict[str, Any]:
        return self.cfg["tiers"][tier]

    # -- auxiliary data --------------------------------------------------------

    def load_yaml(self, name: str) -> dict[str, Any]:
        p = DATA / name
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    @property
    def glossary(self) -> dict[str, Any]:
        return self.load_yaml("glossary.yaml").get("terms", {})

    @property
    def notation(self) -> dict[str, Any]:
        return self.load_yaml("notation.yaml").get("symbols", {})

    @property
    def bibliography(self) -> dict[str, Any]:
        return self.load_yaml("bibliography.yaml").get("entries", {})


def load() -> Book:
    return Book()
