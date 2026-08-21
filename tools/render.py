"""
render.py — Markdown -> HTML for one document, plus the book-wide reference
resolution pass.

Rendering is deliberately two-phase:

  Phase 1 (per document)  parse, render, and emit *placeholders* for anything
                          that needs knowledge the document does not have:
                          cross-references to other chapters, citations,
                          glossary links.
  Phase 2 (book-wide)     with every document's labels collected, walk the DOM
                          and fill the placeholders in.

That split is what lets Chapter 63 reference an equation defined in Chapter 90
without a chicken-and-egg problem, and it is why the build can report a dangling
reference as an error instead of silently emitting broken text.

Authoring conventions this module implements
--------------------------------------------
  Display math with a label      $$\\n ... \\n$$ (eq:name)
  Diagram                        ```mermaid {#fig:name caption="..."}
  Code with a verification tier  ```python {tier=A name=softmax-demo}
  Table label (line before)      {#tbl:name caption="..."}
  Callout                        > NOTE: ...   (also WARNING, IMPORTANT,
                                 PRODUCTION TIP, RESEARCH NOTE, MATH NOTE,
                                 HISTORY)
  Cross-reference                {{ch:tf-multi-head}} {{eq:name}} {{fig:name}}
                                 {{tbl:name}} {{sec:name}} {{part:12}}
  Citation                       {{cite:vaswani2017}}
  Glossary term (forced)         {{term:attention}}
  Maturity badge                 {{maturity:EMERGING}}
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree, html as lhtml
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from bookdata import ROOT, Book, Doc, split_frontmatter

CODE_DIR = ROOT / "code"

CALLOUT_KINDS = {
    "NOTE": "note",
    "WARNING": "warning",
    "IMPORTANT": "important",
    "PRODUCTION TIP": "production",
    "RESEARCH NOTE": "research",
    "MATH NOTE": "math",
    "HISTORY": "history",
}
CALLOUT_TITLES = {
    "note": "Note", "warning": "Warning", "important": "Important",
    "production": "Production Tip", "research": "Research Note",
    "math": "Math Note", "history": "History",
}

MATURITY_CLASS = {
    "ESTABLISHED": "established", "MATURE": "mature", "EMERGING": "emerging",
    "EXPERIMENTAL": "experimental", "RESEARCH FRONTIER": "frontier",
}


# =============================================================================
# Results
# =============================================================================

# Sentinel returned by ReferenceResolver._lookup when a reference names a real,
# planned chapter whose file has not been written yet.
PENDING = object()


@dataclass
class Label:
    """A numbered, referenceable thing: an equation, figure, table, or section."""
    kind: str          # eq | fig | tbl | sec
    name: str          # bare name, e.g. "softmax"
    number: str        # display number, e.g. "63.4"
    doc_id: str
    anchor: str
    caption: str = ""


@dataclass
class Rendered:
    doc: Doc
    body: str
    toc: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, Label] = field(default_factory=dict)
    words: int = 0
    code_blocks: list[dict[str, Any]] = field(default_factory=list)
    citations: set[str] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)


# =============================================================================
# Markdown preprocessing (fence-aware, line-based)
# =============================================================================

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_TBL_LABEL_RE = re.compile(r'^\{#(tbl:[A-Za-z0-9][\w-]*)((?:\s+\w+="[^"]*")*)\s*\}\s*$')


def preprocess(text: str) -> str:
    """Rewrite table-label lines into HTML markers, skipping fenced code."""
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if m:
            tok = m.group(1)[0] * 3
            if fence is None:
                fence = tok
            elif line.strip().startswith(fence):
                fence = None
            out.append(line)
            continue
        if fence is None:
            tm = _TBL_LABEL_RE.match(line.strip())
            if tm:
                attrs = parse_attrs(tm.group(2) or "")
                cap = html.escape(attrs.get("caption", ""), quote=True)
                out.append(
                    f'<!--TBLLABEL name="{tm.group(1)}" caption="{cap}"-->')
                continue
        out.append(line)
    return "\n".join(out)


_LINESPAN_RE = re.compile(r'<span id="ln-(\d+)">')
_PYG_WRAPPER_RE = re.compile(r"<pre[^>]*>(.*)</pre>", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
_ID_RE = re.compile(r"#([A-Za-z0-9][\w:-]*)")


def parse_attrs(info: str) -> dict[str, str]:
    """Parse a fence info suffix like `{#fig:x caption="y" tier=A}`."""
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
    """Split a fence info string into (language, attributes)."""
    info = info.strip()
    if not info:
        return "", {}
    brace = info.find("{")
    if brace == -1:
        parts = info.split(None, 1)
        return parts[0], (parse_attrs(parts[1]) if len(parts) > 1 else {})
    return info[:brace].strip(), parse_attrs(info[brace:])


# =============================================================================
# Token-stream transforms
# =============================================================================

_CALLOUT_RE = re.compile(
    r"^(" + "|".join(re.escape(k) for k in CALLOUT_KINDS) + r")\s*:\s*",
    re.IGNORECASE)


def transform_callouts(tokens: list[Token]) -> None:
    """Turn `> NOTE: ...` blockquotes into semantic <aside> callouts.

    The first inline token inside the quote carries the marker; the marker text
    is stripped and the surrounding blockquote tokens are retyped so the default
    renderer emits an aside instead.
    """
    depth_stack: list[int] = []
    for i, tok in enumerate(tokens):
        if tok.type == "blockquote_open":
            kind = _detect_callout(tokens, i)
            depth_stack.append(0)
            if kind:
                tok.type = "callout_open"
                tok.meta = {"kind": kind}
        elif tok.type == "blockquote_close":
            if depth_stack:
                depth_stack.pop()
            # Match with the nearest unclosed callout_open.
            for j in range(i - 1, -1, -1):
                if tokens[j].type == "callout_open" and not tokens[j].meta.get("closed"):
                    tokens[j].meta["closed"] = True
                    tok.type = "callout_close"
                    break
                if tokens[j].type == "blockquote_open":
                    break


def _detect_callout(tokens: list[Token], open_idx: int) -> str | None:
    """Inspect the first inline token of a blockquote for a callout marker."""
    for tok in tokens[open_idx + 1 : open_idx + 4]:
        if tok.type == "inline":
            m = _CALLOUT_RE.match(tok.content)
            if not m:
                return None
            kind = CALLOUT_KINDS[m.group(1).upper()]
            tok.content = tok.content[m.end():]
            if tok.children:
                first = tok.children[0]
                if first.type == "text":
                    first.content = _CALLOUT_RE.sub("", first.content, count=1)
            return kind
        if tok.type not in ("paragraph_open",):
            return None
    return None


# =============================================================================
# The renderer
# =============================================================================

class DocRenderer:
    """Renders one document. Instantiate per document; not reusable."""

    def __init__(self, book: Book, doc: Doc, math, mermaid, extract_code: bool = True):
        self.book = book
        self.doc = doc
        self.math = math
        self.mermaid = mermaid
        self.extract_code = extract_code

        self.chapter_key = str(doc.number) if doc.number else doc.slug
        self.counters = {"eq": 0, "fig": 0, "tbl": 0}
        self.labels: dict[str, Label] = {}
        self.math_queue: list[tuple[str, bool]] = []
        self.code_blocks: list[dict[str, Any]] = []
        self.issues: list[str] = []
        self._anchors: set[str] = set()

        self.md = self._make_md()

    # -- markdown-it wiring ---------------------------------------------------

    def _make_md(self) -> MarkdownIt:
        md = MarkdownIt("commonmark", {"typographer": True, "html": True})
        md.enable(["table", "strikethrough", "replacements", "smartquotes"])
        md.use(dollarmath_plugin, allow_labels=True, double_inline=True)
        md.use(deflist_plugin)
        md.use(footnote_plugin)

        r = md.renderer.rules
        r["fence"] = self._rule_fence
        r["math_inline"] = self._rule_math_inline
        r["math_inline_double"] = self._rule_math_inline_double
        r["math_block"] = self._rule_math_block
        r["math_block_label"] = self._rule_math_block
        r["callout_open"] = self._rule_callout_open
        r["callout_close"] = lambda *a: "</div></aside>\n"
        return md

    # -- fences ---------------------------------------------------------------

    def _rule_fence(self, tokens, idx, options, env) -> str:
        tok = tokens[idx]
        lang, attrs = split_info(tok.info or "")
        if lang == "mermaid":
            return self._render_diagram(tok.content, attrs)
        return self._render_code(tok.content, lang, attrs)

    def _render_diagram(self, source: str, attrs: dict[str, str]) -> str:
        svg = self.mermaid.render(source)
        caption = attrs.get("caption", "")
        ref = attrs.get("id", "")
        num = ""
        anchor = ""
        if ref.startswith("fig:"):
            self.counters["fig"] += 1
            num = f"{self.chapter_key}.{self.counters['fig']}"
            anchor = f"fig-{ref[4:]}"
            self.labels[ref] = Label("fig", ref[4:], num, self.doc.id, anchor, caption)
        cap_html = ""
        if caption or num:
            lead = f'<span class="fig-num">Figure {num}</span> ' if num else ""
            cap_html = (f'<figcaption>{lead}'
                        f'{self.md.renderInline(caption) if caption else ""}'
                        f'</figcaption>')
        idattr = f' id="{anchor}"' if anchor else ""
        return (f'<figure class="diagram"{idattr}>'
                f'<div class="diagram-svg">{svg}</div>{cap_html}</figure>\n')

    def _render_code(self, source: str, lang: str, attrs: dict[str, str]) -> str:
        lang = lang or "text"
        tier = attrs.get("tier", "C").upper()
        name = attrs.get("name", "")

        if self.extract_code and lang in ("python", "py") and tier in ("A", "B"):
            self._extract(source, name, tier)

        try:
            lexer = get_lexer_by_name(lang, stripall=False)
        except ClassNotFound:
            lexer = get_lexer_by_name("text")

        # linespans makes Pygments wrap each *source* line in its own element,
        # splitting multi-line tokens (docstrings, triple-quoted strings)
        # correctly. Turning those into block rows keeps the line numbers locked
        # to the code even in print, where long lines must soft-wrap — a
        # separate gutter column silently drifts out of alignment there.
        #
        # linespans is only applied when nowrap is off, so the wrapper markup is
        # generated and then stripped rather than suppressed up front.
        formatter = HtmlFormatter(linespans="ln")
        body = highlight(source, lexer, formatter)
        m = _PYG_WRAPPER_RE.search(body)
        if m:
            body = m.group(1)
        body = _LINESPAN_RE.sub(r'<span class="cl" data-ln="\1">', body)

        badge = ""
        tier_cfg = self.book.cfg.get("code_tiers", {}).get(tier, {})
        if tier_cfg.get("badge"):
            badge = (f'<p class="code-badge tier-{tier.lower()}">'
                     f'{html.escape(tier_cfg["badge"])}</p>')
        header = ""
        if name:
            header = f'<p class="code-name">{html.escape(name)}</p>'

        n_lines = len(source.rstrip("\n").split("\n"))
        long = " long" if n_lines > 30 else ""
        return (f'<div class="codeblock{long} lang-{html.escape(lang)}">{header}'
                f'<pre class="code"><code class="language-{html.escape(lang)}">'
                f'{body}</code></pre>{badge}</div>\n')

    def _extract(self, source: str, name: str, tier: str) -> None:
        n = len(self.code_blocks) + 1
        stem = name or f"snippet-{n:02d}"
        part = f"part-{self.doc.part:02d}" if self.doc.part else "misc"
        path = CODE_DIR / part / self.doc.slug / f"{n:02d}-{stem}.py"
        self.code_blocks.append({
            "path": str(path.relative_to(ROOT)), "tier": tier,
            "name": stem, "source": source, "doc": self.doc.id,
        })

    # -- math -----------------------------------------------------------------

    def _queue(self, tex: str, display: bool) -> int:
        self.math_queue.append((tex, display))
        return len(self.math_queue) - 1

    def _rule_math_inline(self, tokens, idx, options, env) -> str:
        return f"\x00MATH{self._queue(tokens[idx].content, False)}\x00"

    def _rule_math_inline_double(self, tokens, idx, options, env) -> str:
        return (f'<span class="math-display-inline">'
                f"\x00MATH{self._queue(tokens[idx].content, True)}\x00</span>")

    def _rule_math_block(self, tokens, idx, options, env) -> str:
        tok = tokens[idx]
        slot = self._queue(tok.content.strip(), True)
        ref = (tok.info or "").strip()
        num = ""
        anchor = ""
        if ref.startswith("eq:"):
            self.counters["eq"] += 1
            num = f"{self.chapter_key}.{self.counters['eq']}"
            anchor = f"eq-{ref[3:]}"
            self.labels[ref] = Label("eq", ref[3:], num, self.doc.id, anchor)
        idattr = f' id="{anchor}"' if anchor else ""
        tag = f'<span class="eq-num">({num})</span>' if num else ""
        cls = "equation numbered" if num else "equation"
        return (f'<div class="{cls}"{idattr}>'
                f'<div class="eq-body">\x00MATH{slot}\x00</div>{tag}</div>\n')

    # -- callouts -------------------------------------------------------------

    def _rule_callout_open(self, tokens, idx, options, env) -> str:
        kind = tokens[idx].meta.get("kind", "note")
        title = CALLOUT_TITLES.get(kind, kind.title())
        return (f'<aside class="callout callout-{kind}">'
                f'<p class="callout-title">{title}</p><div class="callout-body">')

    # -- main entry -----------------------------------------------------------

    def render(self) -> Rendered:
        raw = self.doc.path.read_text(encoding="utf-8")
        _, body_md = split_frontmatter(raw)
        body_md = preprocess(body_md)

        tokens = self.md.parse(body_md)
        transform_callouts(tokens)
        html_out = self.md.renderer.render(tokens, self.md.options, {})

        html_out = self._substitute_math(html_out)
        tree = self._dom_pass(html_out)
        body = self._serialise(tree)

        return Rendered(
            doc=self.doc, body=body, toc=self._toc, labels=self.labels,
            words=self._words, code_blocks=self.code_blocks,
            citations=self._citations, issues=self.issues,
        )

    def _substitute_math(self, s: str) -> str:
        if not self.math_queue:
            return s
        rendered = self.math.render_many(self.math_queue)
        def sub(m: re.Match[str]) -> str:
            return rendered[int(m.group(1))]
        return re.sub(r"\x00MATH(\d+)\x00", sub, s)

    # -- DOM pass -------------------------------------------------------------

    def _dom_pass(self, body_html: str):
        tree = lhtml.fragment_fromstring(body_html, create_parent="div")
        self._toc: list[dict[str, Any]] = []
        self._citations: set[str] = set()

        self._number_headings(tree)
        self._attach_table_labels(tree)
        self._wrap_tables(tree)
        self._mark_macros(tree)
        self._words = self._count_words(tree)
        return tree

    def _slug(self, text: str) -> str:
        s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
        s = re.sub(r"[\s_]+", "-", s) or "section"
        base, n = s, 2
        while s in self._anchors:
            s = f"{base}-{n}"
            n += 1
        self._anchors.add(s)
        return s

    @staticmethod
    def _heading_text(h) -> str:
        """Plain text of a heading, with rendered maths reduced to its source.

        A KaTeX span tree contains the expression twice — as MathML and as
        per-glyph HTML — so naive itertext() on a heading containing maths
        yields doubled gibberish. The TeX annotation KaTeX embeds is the only
        faithful plain-text form available.
        """
        parts: list[str] = []

        def walk(node) -> None:
            if not isinstance(node.tag, str):
                return
            cls = node.get("class") or ""
            if "katex" in cls:
                tex = node.xpath(
                    './/*[local-name()="annotation"][@encoding="application/x-tex"]/text()')
                if tex:
                    parts.append(re.sub(r"[\\{}]", "", tex[0]).strip())
                parts.append(node.tail or "")
                return
            parts.append(node.text or "")
            for child in node:
                walk(child)
            parts.append(node.tail or "")

        parts.append(h.text or "")
        for child in h:
            walk(child)
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    @staticmethod
    def _heading_html(h) -> str:
        """Inner markup of a heading, so the on-page contents keeps real maths."""
        out = [html.escape(h.text or "")]
        for child in h:
            if (child.get("class") or "") == "hanchor":
                continue
            out.append(lhtml.tostring(child, encoding="unicode"))
        return "".join(out).strip()

    def _number_headings(self, tree) -> None:
        for h in tree.iter("h1", "h2", "h3", "h4", "h5", "h6"):
            text = self._heading_text(h)
            anchor = h.get("id") or self._slug(text)
            h.set("id", anchor)
            level = int(h.tag[1])
            if level <= 3:
                self._toc.append({"level": level, "text": text,
                                  "html": self._heading_html(h),
                                  "anchor": anchor})
            if level == 2:
                self.labels[f"sec:{anchor}"] = Label(
                    "sec", anchor, "", self.doc.id, anchor, text)
            link = etree.SubElement(h, "a")
            link.set("class", "hanchor")
            link.set("href", f"#{anchor}")
            link.set("aria-hidden", "true")
            link.text = "#"

    def _attach_table_labels(self, tree) -> None:
        """Bind `<!--TBLLABEL-->` markers to the table that follows them."""
        for comment in tree.xpath("//comment()"):
            data = (comment.text or "").strip()
            if not data.startswith("TBLLABEL"):
                continue
            attrs = parse_attrs(data[len("TBLLABEL"):])
            nxt = comment.getnext()
            while nxt is not None and nxt.tag is etree.Comment:
                nxt = nxt.getnext()
            if nxt is not None and nxt.tag == "table":
                name = attrs.get("name", "")
                self.counters["tbl"] += 1
                num = f"{self.chapter_key}.{self.counters['tbl']}"
                anchor = f"tbl-{name.split(':', 1)[-1]}"
                nxt.set("data-label", name)
                nxt.set("data-number", num)
                nxt.set("data-caption", attrs.get("caption", ""))
                nxt.set("id", anchor)
                if name:
                    self.labels[name] = Label(
                        "tbl", name.split(":", 1)[-1], num, self.doc.id,
                        anchor, attrs.get("caption", ""))
            parent = comment.getparent()
            if parent is not None:
                parent.remove(comment)

    def _wrap_tables(self, tree) -> None:
        """Wrap tables so wide ones scroll on screen and shrink in print."""
        for table in list(tree.iter("table")):
            parent = table.getparent()
            if parent is None or parent.get("class") == "table-scroll":
                continue
            idx = parent.index(table)
            fig = etree.Element("figure")
            fig.set("class", "table-figure")
            if table.get("id"):
                fig.set("id", table.get("id"))
                del table.attrib["id"]
            scroll = etree.SubElement(fig, "div")
            scroll.set("class", "table-scroll")
            parent.insert(idx, fig)
            scroll.append(table)
            num, cap = table.get("data-number"), table.get("data-caption")
            if num or cap:
                fc = etree.SubElement(fig, "figcaption")
                lead = etree.SubElement(fc, "span")
                lead.set("class", "tbl-num")
                lead.text = f"Table {num}" if num else ""
                lead.tail = f" {cap}" if cap else ""

    def _mark_macros(self, tree) -> None:
        """Replace {{...}} macros in text nodes with placeholder elements.

        Skips code, pre, and existing anchors so a literal `{{ch:x}}` inside a
        code sample survives intact.
        """
        SKIP = {"code", "pre", "script", "style", "a"}
        pattern = re.compile(r"\{\{([a-z]+):([^}]+)\}\}")

        for node in list(tree.iter()):
            if not isinstance(node.tag, str):
                continue
            # A skipped element's *text* is literal content to leave alone, but
            # its *tail* is ordinary prose belonging to the parent's flow. Not
            # processing tails here silently dropped every macro that happened
            # to follow an inline <code> span or a link.
            attrs = ("tail",) if node.tag in SKIP else ("text", "tail")
            for attr in attrs:
                text = getattr(node, attr)
                if not text or "{{" not in text:
                    continue
                if attr == "tail" and node.getparent() is None:
                    continue
                self._split_macros(node, attr, text, pattern)

    def _split_macros(self, node, attr, text, pattern) -> None:
        parts = list(pattern.finditer(text))
        if not parts:
            return
        parent = node.getparent() if attr == "tail" else node
        insert_at = (parent.index(node) + 1) if attr == "tail" else 0

        setattr(node, attr, text[: parts[0].start()])
        cursor = insert_at
        for i, m in enumerate(parts):
            kind, arg = m.group(1), m.group(2).strip()
            el = self._macro_element(kind, arg)
            tail_end = parts[i + 1].start() if i + 1 < len(parts) else len(text)
            el.tail = text[m.end() : tail_end]
            parent.insert(cursor, el)
            cursor += 1

    def _macro_element(self, kind: str, arg: str):
        el = etree.Element("span")
        if kind == "cite":
            keys = [k.strip() for k in arg.split(",") if k.strip()]
            self._citations.update(keys)
            el.set("class", "xref cite")
            el.set("data-keys", ",".join(keys))
        elif kind == "maturity":
            key = arg.upper()
            el.tag = "span"
            el.set("class", f"maturity maturity-{MATURITY_CLASS.get(key, 'emerging')}")
            el.text = key
        elif kind in ("ch", "eq", "fig", "tbl", "sec", "part", "term", "proj"):
            el.set("class", "xref")
            el.set("data-kind", kind)
            el.set("data-arg", arg)
        else:
            el.set("class", "xref broken")
            el.text = f"{{{{{kind}:{arg}}}}}"
            self.issues.append(f"unknown macro kind '{kind}'")
        return el

    def _count_words(self, tree) -> int:
        """Count prose words only.

        Code, diagrams and rendered maths are excluded. Maths matters here: a
        KaTeX span tree carries the expression twice — once as MathML text and
        once as per-glyph HTML spans — so counting it naively can double a
        chapter's apparent length and defeat the depth gate.
        """
        SKIP_TAGS = {"code", "pre", "script", "style", "svg", "math"}

        def walk(node) -> int:
            if not isinstance(node.tag, str):
                return 0
            cls = node.get("class") or ""
            if node.tag in SKIP_TAGS or "katex" in cls:
                # Skip the subtree entirely; the tail belongs to the parent's
                # flow, so it is still counted by the caller below.
                return len((node.tail or "").split())
            n = len((node.text or "").split()) + len((node.tail or "").split())
            for child in node:
                n += walk(child)
            return n

        return walk(tree)

    @staticmethod
    def _serialise(tree) -> str:
        inner = (tree.text or "")
        inner += "".join(
            lhtml.tostring(c, encoding="unicode") for c in tree)
        return inner


# =============================================================================
# Phase 2 — book-wide reference resolution
# =============================================================================

class ReferenceResolver:
    """Fills the placeholders left by phase 1, using the whole book's labels."""

    def __init__(self, book: Book, rendered: dict[str, Rendered]):
        self.book = book
        self.rendered = rendered
        self.glossary = book.glossary
        self.bib = book.bibliography
        self.labels: dict[str, tuple[Label, Doc]] = {}
        for r in rendered.values():
            for key, lab in r.labels.items():
                self.labels[key] = (lab, r.doc)
        self.dangling: list[tuple[str, str]] = []
        self.pending: list[tuple[str, str]] = []
        self._alias_index = self._build_alias_index()

    def _build_alias_index(self) -> list[tuple[re.Pattern[str], str]]:
        idx: list[tuple[re.Pattern[str], str]] = []
        for tid, entry in self.glossary.items():
            # Terms whose surface form is a common English word ("key", "value",
            # "token") set autolink: false. They still resolve through an
            # explicit {{term:...}}, but auto-linking them would pepper every
            # chapter with links on incidental uses of the word.
            if entry.get("autolink") is False:
                continue
            names = [entry.get("term", tid)] + list(entry.get("aliases", []) or [])
            for n in names:
                if len(n) < 3:
                    continue
                idx.append((re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE), tid))
        # Longest surface form first so "multi-head attention" wins over "attention".
        idx.sort(key=lambda p: -len(p[0].pattern))
        return idx

    def resolve(self, r: Rendered, href_for) -> str:
        tree = lhtml.fragment_fromstring(r.body, create_parent="div")
        self._resolve_xrefs(tree, r, href_for)
        self._link_glossary(tree, r, href_for)
        return DocRenderer._serialise(tree)

    # -- cross references ------------------------------------------------------

    def _resolve_xrefs(self, tree, r: Rendered, href_for) -> None:
        for el in tree.xpath('//span[contains(@class,"xref")]'):
            if "cite" in (el.get("class") or ""):
                self._render_citation(el, href_for)
                continue
            kind, arg = el.get("data-kind"), el.get("data-arg")
            if not kind:
                continue
            text, href = self._lookup(kind, arg, r, href_for)
            if href is None:
                el.set("class", "xref broken")
                el.text = text
                self.dangling.append((r.doc.id, f"{kind}:{arg}"))
                continue
            if href is PENDING:
                # The target is a real, planned chapter that simply is not
                # written yet. Render the designator without a dead link rather
                # than pretending the page exists.
                el.set("class", "xref xref-pending")
                el.set("title", "This chapter is planned but not yet written")
                el.text = text
                self.pending.append((r.doc.id, f"{kind}:{arg}"))
                continue
            a = etree.Element("a")
            a.set("href", href)
            a.set("class", f"xref xref-{kind}")
            a.text = text
            a.tail = el.tail
            el.getparent().replace(el, a)

    def _lookup(self, kind: str, arg: str, r: Rendered, href_for):
        if kind == "ch":
            doc = self.book.by_id.get(arg)
            if not doc:
                return f"[missing chapter: {arg}]", None
            if doc.id not in self.rendered:
                return doc.label, PENDING
            return doc.label, href_for(r.doc, doc, None)
        if kind == "proj":
            doc = self.book.by_id.get(arg)
            if not doc:
                return f"[missing project: {arg}]", None
            if doc.id not in self.rendered:
                return doc.label, PENDING
            return doc.label, href_for(r.doc, doc, None)
        if kind == "part":
            try:
                p = self.book.part(int(arg))
            except (ValueError, KeyError):
                return f"[missing part: {arg}]", None
            intro = p.intro
            if intro is None or intro.id not in self.rendered:
                return f"Part {p.roman}", PENDING
            return f"Part {p.roman}", href_for(r.doc, intro, None)
        if kind == "term":
            entry = self.glossary.get(arg)
            if not entry:
                return f"[missing term: {arg}]", None
            gloss = self.book.by_id.get("app-glossary")
            return entry.get("term", arg), href_for(r.doc, gloss, f"term-{arg}")

        key = f"{kind}:{arg}"
        hit = self.labels.get(key)
        if not hit:
            return f"[missing {kind}: {arg}]", None
        lab, doc = hit
        names = {"eq": "Equation", "fig": "Figure", "tbl": "Table", "sec": ""}
        prefix = names.get(kind, "")
        if kind == "sec":
            text = lab.caption or arg
        else:
            text = f"{prefix} {lab.number}".strip()
            if doc.id != r.doc.id:
                text += f" ({doc.label})"
        return text, href_for(r.doc, doc, lab.anchor)

    def _render_citation(self, el, href_for) -> None:
        keys = [k for k in (el.get("data-keys") or "").split(",") if k]
        papers = self.book.by_id.get("app-papers")
        span = etree.Element("span")
        span.set("class", "citation")
        span.tail = el.tail
        span.text = "["
        for i, key in enumerate(keys):
            entry = self.bib.get(key)
            a = etree.SubElement(span, "a")
            a.set("href", f"{papers.slug}.html#bib-{key}" if papers else f"#bib-{key}")
            if entry:
                a.set("class", "cite-ok" if entry.get("verified") else "cite-unverified")
                a.text = self._short(entry, key)
                if not entry.get("verified"):
                    a.text += " [UNVERIFIED]"
            else:
                a.set("class", "cite-missing")
                a.text = f"{key} [UNVERIFIED CITATION]"
            a.tail = "; " if i < len(keys) - 1 else "]"
        el.getparent().replace(el, span)

    @staticmethod
    def _short(entry: dict[str, Any], key: str) -> str:
        authors = entry.get("authors") or []
        year = entry.get("year", "n.d.")
        if not authors:
            return f"{key}, {year}"
        first = str(authors[0]).split()[-1]
        if len(authors) == 1:
            return f"{first} {year}"
        if len(authors) == 2:
            return f"{first} & {str(authors[1]).split()[-1]} {year}"
        return f"{first} et al. {year}"

    # -- glossary --------------------------------------------------------------

    def _link_glossary(self, tree, r: Rendered, href_for) -> None:
        """Link the first mention of each glossary term in this document."""
        if not self._alias_index or r.doc.kind == "appendix":
            return
        gloss = self.book.by_id.get("app-glossary")
        if gloss is None:
            return
        seen: set[str] = set()
        SKIP = {"code", "pre", "script", "style", "a", "dfn", "h1", "h2",
                "h3", "h4", "h5", "h6", "svg", "figcaption"}

        for node in list(tree.iter()):
            if not isinstance(node.tag, str) or node.tag in SKIP:
                continue
            if node.tag == "aside":
                continue
            if not node.text or len(seen) == len(self._alias_index):
                continue
            if any(a.tag in SKIP for a in node.iterancestors()
                   if isinstance(a.tag, str)):
                continue
            self._link_in_text(node, r, gloss, seen, href_for)

    def _link_in_text(self, node, r, gloss, seen, href_for) -> None:
        for rx, tid in self._alias_index:
            if tid in seen:
                continue
            text = node.text or ""
            m = rx.search(text)
            if not m:
                continue
            seen.add(tid)
            dfn = etree.Element("dfn")
            dfn.set("class", "term")
            a = etree.SubElement(dfn, "a")
            a.set("href", href_for(r.doc, gloss, f"term-{tid}"))
            a.text = m.group(0)
            dfn.tail = text[m.end():]
            node.text = text[: m.start()]
            node.insert(0, dfn)
            return  # one substitution per node keeps offsets valid
