"""
generated.py — documents the build produces rather than an author writing them.

The table of contents, glossary, acronym list, notation reference and annotated
bibliography are all derived from book.yaml and data/*.yaml. Generating them
means they cannot fall out of step with the prose: adding a chapter or a term
updates every one of these pages on the next build.
"""

from __future__ import annotations

import html
from typing import Any

from bookdata import Book, Doc
from render import Rendered


def _e(s: Any) -> str:
    return html.escape(str(s or ""))


def build_generated_docs(book: Book, rendered: dict[str, Rendered],
                         math=None) -> list[Rendered]:
    out: list[Rendered] = []
    for doc_id, fn in [
        ("fm-toc", _toc),
        ("app-glossary", _glossary),
        ("app-acronyms", _acronyms),
        ("app-notation", _notation),
        ("app-papers", _papers),
    ]:
        doc = book.by_id.get(doc_id)
        if doc is None:
            continue
        body = fn(book, rendered, math)
        out.append(Rendered(doc=doc, body=body,
                            words=len(_strip(body).split())))
    return out


def _strip(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", s)


# =============================================================================

def _toc(book: Book, rendered: dict[str, Rendered], math=None) -> str:
    p: list[str] = ["<h1>Contents</h1>"]
    p.append('<p class="toc-lead">Chapters marked <em>pending</em> are planned '
             'but not yet written. Reading order is top to bottom.</p>')

    def li(d: Doc) -> str:
        state = "" if d.id in rendered else ' <span class="pending">pending</span>'
        label = f'<span class="toc-label">{_e(d.label)}</span> ' \
            if d.kind in ("chapter", "project", "appendix") else ""
        href = f'{d.slug}.html'
        link = (f'<a href="{href}">{_e(d.title)}</a>' if d.id in rendered
                else f'<span class="toc-dim">{_e(d.title)}</span>')
        return f'<li class="toc-{d.kind}">{label}{link}{state}</li>'

    fm = [d for d in book.docs if d.kind == "frontmatter" and d.id != "fm-toc"]
    p.append("<h2>Front Matter</h2><ul class='toc'>")
    p += [li(d) for d in fm]
    p.append("</ul>")

    for part in book.parts:
        p.append(f'<h2>Part {part.roman} — {_e(part.title)}</h2>')
        if part.summary:
            p.append(f'<p class="part-summary">{_e(part.summary)}</p>')
        p.append("<ul class='toc'>")
        p += [li(d) for d in part.chapters]
        projects = [d for d in book.docs
                    if d.kind == "project" and d.part == part.number]
        p += [li(d) for d in projects]
        p.append("</ul>")

    cap = [d for d in book.docs if d.kind == "capstone"]
    if cap:
        p.append(f'<h2>{_e(book.cfg["capstone"]["title"])}</h2><ul class="toc">')
        p += [li(d) for d in cap]
        p.append("</ul>")

    p.append("<h2>Appendices</h2><ul class='toc'>")
    p += [li(d) for d in book.docs if d.kind == "appendix"]
    p.append("</ul>")
    return "\n".join(p)


# =============================================================================

def _glossary(book: Book, rendered: dict[str, Rendered], math=None) -> str:
    terms = book.glossary
    p = ["<h1>Complete Glossary</h1>",
         '<p>Every technical term the book defines, with an intuitive gloss '
         'first and a precise statement second. The first mention of a term in '
         'any chapter links here.</p>']
    if not terms:
        return "\n".join(p + ["<p><em>No terms defined yet.</em></p>"])

    by_letter: dict[str, list[tuple[str, dict]]] = {}
    for tid, e in sorted(terms.items(), key=lambda kv: kv[1].get("term", kv[0]).lower()):
        by_letter.setdefault(e.get("term", tid)[0].upper(), []).append((tid, e))

    p.append('<nav class="alpha-index">' + " ".join(
        f'<a href="#letter-{L}">{L}</a>' for L in sorted(by_letter)) + "</nav>")

    for L in sorted(by_letter):
        p.append(f'<h2 id="letter-{L}">{L}</h2><dl class="glossary">')
        for tid, e in by_letter[L]:
            p.append(f'<dt id="term-{_e(tid)}">{_e(e.get("term", tid))}')
            if e.get("aliases"):
                p.append(f' <span class="aliases">also: '
                         f'{_e(", ".join(e["aliases"]))}</span>')
            p.append("</dt><dd>")
            if e.get("intuitive"):
                p.append(f'<p class="gloss-intuitive">{_e(e["intuitive"])}</p>')
            if e.get("formal"):
                p.append(f'<p class="gloss-formal">{_e(e["formal"])}</p>')
            meta = []
            if e.get("introduced_in"):
                d = book.by_id.get(e["introduced_in"])
                if d:
                    meta.append(f'Introduced in <a href="{d.slug}.html">'
                                f'{_e(d.label)}</a>')
            if e.get("see_also"):
                links = ", ".join(
                    f'<a href="#term-{_e(s)}">{_e(terms.get(s, {}).get("term", s))}</a>'
                    for s in e["see_also"] if s in terms)
                if links:
                    meta.append(f"See also: {links}")
            if meta:
                p.append(f'<p class="gloss-meta">{" · ".join(meta)}</p>')
            p.append("</dd>")
        p.append("</dl>")
    return "\n".join(p)


def _acronyms(book: Book, rendered: dict[str, Rendered], math=None) -> str:
    rows = []
    for tid, e in book.glossary.items():
        if e.get("acronym"):
            rows.append((e["acronym"], e.get("term", tid), tid))
    rows.sort()
    p = ["<h1>Acronym Dictionary</h1>"]
    if not rows:
        return "\n".join(p + ["<p><em>No acronyms recorded yet.</em></p>"])
    p.append('<table class="acronyms"><thead><tr><th>Acronym</th>'
             "<th>Expansion</th></tr></thead><tbody>")
    for ac, term, tid in rows:
        p.append(f'<tr><td><code>{_e(ac)}</code></td>'
                 f'<td><a href="{book.by_id["app-glossary"].slug}.html#term-'
                 f'{_e(tid)}">{_e(term)}</a></td></tr>')
    p.append("</tbody></table>")
    return "\n".join(p)


def _notation(book: Book, rendered: dict[str, Rendered], math=None) -> str:
    syms = book.notation
    p = ["<h1>Notation Reference</h1>",
         "<p>Symbols keep these meanings for the whole book. Where a research "
         "paper uses a different convention, the chapter says so explicitly.</p>"]
    if not syms:
        return "\n".join(p + ["<p><em>Notation not yet defined.</em></p>"])
    groups: dict[str, list] = {}
    for sid, e in syms.items():
        groups.setdefault(e.get("group", "General"), []).append((sid, e))

    # Symbols are TeX, so they go through the same KaTeX pass as chapter maths —
    # the reference page must render identically to the chapters it documents.
    ordered = [(g, sid, e) for g in sorted(groups)
               for sid, e in sorted(groups[g], key=lambda kv: kv[0])]
    if math is not None:
        html_syms = math.render_many([(e.get("tex", sid), False)
                                      for _, sid, e in ordered])
    else:
        html_syms = [f'<code>{_e(e.get("tex", sid))}</code>' for _, sid, e in ordered]

    i = 0
    for g in sorted(groups):
        p.append(f"<h2>{_e(g)}</h2>")
        p.append('<table class="notation"><thead><tr><th>Symbol</th>'
                 "<th>Meaning</th><th>Notes</th></tr></thead><tbody>")
        for sid, e in sorted(groups[g], key=lambda kv: kv[0]):
            p.append(f'<tr id="sym-{_e(sid)}"><td class="sym">{html_syms[i]}</td>'
                     f'<td>{_e(e.get("meaning", ""))}</td>'
                     f'<td>{_e(e.get("notes", ""))}</td></tr>')
            i += 1
        p.append("</tbody></table>")
    return "\n".join(p)


def _papers(book: Book, rendered: dict[str, Rendered], math=None) -> str:
    bib = book.bibliography
    p = ["<h1>Annotated Bibliography</h1>",
         "<p>Each entry records what the work contributed, why it mattered, and "
         "what descends from it. Entries not yet checked against a primary "
         "source are marked <strong>UNVERIFIED</strong> and must not be cited "
         "as established.</p>"]
    if not bib:
        return "\n".join(p + ["<p><em>No references recorded yet.</em></p>"])

    cited: dict[str, list[str]] = {}
    for r in rendered.values():
        for key in r.citations:
            cited.setdefault(key, []).append(r.doc.label)

    for key, e in sorted(bib.items(), key=lambda kv: (kv[1].get("year", 0), kv[0])):
        ver = "" if e.get("verified") else ' <span class="unverified">UNVERIFIED</span>'
        p.append(f'<div class="bib-entry" id="bib-{_e(key)}">')
        p.append(f'<h3>{_e(e.get("title", key))}{ver}</h3>')
        auth = ", ".join(e.get("authors", []) or []) or "unknown authors"
        venue = e.get("venue", "")
        p.append(f'<p class="bib-meta">{_e(auth)} · {_e(e.get("year", "n.d."))}'
                 + (f" · {_e(venue)}" if venue else "") + "</p>")
        links = []
        if e.get("arxiv"):
            links.append(f'<a href="https://arxiv.org/abs/{_e(e["arxiv"])}">'
                         f'arXiv:{_e(e["arxiv"])}</a>')
        if e.get("doi"):
            links.append(f'<a href="https://doi.org/{_e(e["doi"])}">doi</a>')
        if e.get("url"):
            links.append(f'<a href="{_e(e["url"])}">link</a>')
        if links:
            p.append(f'<p class="bib-links">{" · ".join(links)}</p>')
        for field, label in [("contribution", "Core contribution"),
                             ("why_it_mattered", "Why it mattered"),
                             ("what_changed_after", "What changed after"),
                             ("descendants", "Modern descendants")]:
            v = e.get(field)
            if not v:
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            p.append(f'<p><strong>{label}.</strong> {_e(v)}</p>')
        if key in cited:
            p.append(f'<p class="bib-cited">Cited in: '
                     f'{_e(", ".join(sorted(set(cited[key]))))}</p>')
        p.append("</div>")
    return "\n".join(p)
