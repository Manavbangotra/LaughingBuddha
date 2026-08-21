# Machines That Learn and Act

A complete technical textbook of modern AI — from mathematical foundations to
production agentic systems. 28 parts, 240 chapters, 15 projects, a six-chapter
capstone, and 15 appendices.

Source is Markdown. The deliverables are a self-contained HTML site and a
paginated PDF, both produced by the build in `tools/`.

## Quick start

```bash
make deps          # python venv + node toolchain (needs network, once)
make html          # -> build/html/index.html
make pdf           # -> build/pdf/part-NN.pdf
make book          # -> build/pdf/machines-that-learn-and-act.pdf
make check         # all quality gates; exit 0 means clean
make report        # progress dashboard
make serve         # preview at http://localhost:8000
```

`make all` runs code extraction, HTML, per-part PDFs, and the merge.

## Layout

```
book.yaml                 structure, ordering, depth tiers — the single source of truth
data/
  bibliography.yaml       verified citations; nothing is cited from memory
  glossary.yaml           every defined term, one definition each
  notation.yaml           the fixed symbol table
  manifest.json           generated: per-document status
src/
  frontmatter/            title, preface, how-to-read, architecture, notation
  part-NN-*/              _part.md, chapters, _assessment.md
  projects/ capstone/ appendices/
code/                     generated: every tagged code block, extracted and runnable
research/                 per-part web-research notes feeding each writing pass
templates/                chapter templates + HTML/PDF Jinja templates
tools/                    the build system and the quality gates
build/                    generated: html/, pdf/, and the render caches
```

## Authoring conventions

| What | Syntax |
|---|---|
| Labelled equation | `$$ ... $$ (eq:name)` |
| Diagram | ` ```mermaid {#fig:name caption="..."} ` |
| Code | ` ```python {tier=A name=slug} ` |
| Table label | `{#tbl:name caption="..."}` on the line before the table |
| Cross-reference | `{{ch:id}}` `{{eq:name}}` `{{fig:name}}` `{{tbl:name}}` `{{sec:anchor}}` `{{part:7}}` |
| Citation | `{{cite:bibkey}}` |
| Glossary term (forced) | `{{term:id}}` |
| Maturity label | `{{maturity:EMERGING}}` |
| Callout | `> NOTE:` — also `WARNING`, `IMPORTANT`, `PRODUCTION TIP`, `RESEARCH NOTE`, `MATH NOTE`, `HISTORY` |

Chapter templates are in `templates/chapter-full.md` and
`templates/chapter-focused.md`. Parts I–V use the focused (12-section) template;
Parts VI–XXVIII use the full (21-section) template.

### Code tiers

- **A** — executed by `make check` in the project venv; must exit 0.
- **B** — needs a GPU, API key, or large weights. Syntax- and import-checked,
  reviewed by hand, and labelled in the book as *not executed locally*.
- **C** — illustrative fragment, marked as not standalone.

## Quality gates

`make check` runs ten gates. All must pass before a part is marked complete.

| Gate | What it prevents |
|---|---|
| `frontmatter` | Metadata drifting from `book.yaml` |
| `structure` | Missing or reordered template sections |
| `depth` | A chapter being quietly compressed below its tier floor |
| `prereqs` | Forward dependencies and cycles in the concept graph |
| `references` | Dangling `{{...}}` cross-references |
| `citations` | Citing a key that is absent or unverified |
| `terminology` | A term defined twice, or a surface form claimed by two terms |
| `rendering` | Maths or diagrams silently failing to render |
| `maturity` | Unknown maturity labels |
| `code` | Tier A listings that no longer run |

Run one gate with `--gate NAME`, or restrict to a part with `--part N`.

## The per-part writing cycle

1. **Research** — verify the part's citations against primary sources; record
   findings in `research/part-NN-notes.md` and entries in `data/bibliography.yaml`.
2. **Outline** — chapter stubs with `requires`/`provides` frontmatter.
3. **Write** the chapters at the part's tier.
4. `make code` — extract listings.
5. `make check --part NN` — fix every error.
6. **Assessment** — knowledge check, practical assignment, advanced challenge,
   interview preparation in `_assessment.md`.
7. `make part PART=NN` — build HTML and the part PDF.

## Requirements

Python 3.12+, Node 20+, and a Chrome or Chromium binary at
`/usr/bin/google-chrome` (used headlessly by Mermaid and Paged.js; override via
`PUPPETEER_EXECUTABLE_PATH` and `tools/puppeteer.json`).

PyTorch is installed from the CPU-only index; see `requirements.txt`.
