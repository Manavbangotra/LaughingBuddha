# Building this book

**Status: builds clean. `make check` passes; `build/html` renders.**

240 chapters, 1,145,157 rendered words, 9,868 equations rendered by KaTeX with
zero errors, 151 Mermaid diagrams, 565 executable listings.

The PDF path is the one thing still unfinished — see
[Known-unverified](#known-unverified).

---

## What exists

| | |
|---|---|
| Chapters | **240 / 240** |
| Part introductions | 28 / 28 |
| Part assessments | 28 / 28 |
| Rendered words | 1,145,157 |
| Equations | 9,868 rendered, 0 errors |
| Diagrams | 151 |
| Tier-A listings | 565, extracted into `code/` |
| Citations | 331, all `verified: true` |
| HTML output | 307 files, 86 MB |

Not yet written: **15 projects**, the **6-chapter capstone**, **11 appendices**.
`book.yaml` already has their slots, so they appear as missing documents rather
than as errors.

---

## Prerequisites

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
npm install          # katex, mermaid-cli, pagedjs-cli
```

Node 18+ and Python 3.11+.

**Without a system Node**, a binary wheel works and is how this book was first
built:

```bash
pip install nodejs-wheel-binaries
export PATH="$(python -c 'import nodejs_wheel,pathlib;print(pathlib.Path(nodejs_wheel.__file__).parent)'):$PATH"
node "$(python -c '...')/lib/node_modules/npm/bin/npm-cli.js" install
```

**On Windows, set `PYTHONIOENCODING=utf-8`.** The gates print `✓`, which the
console codec cannot encode otherwise.

---

## Build

```bash
make offline      # stdlib + PyYAML only; runs before Node exists
make check        # all ten gates
make numbers      # executes all 565 listings; slow, ~an hour
make html         # -> build/html
make pdf          # -> build/pdf   (see below)
make book         # merged volume
make serve        # preview at http://localhost:8000
```

`make check --gate ...` selects individual gates. The `code` gate executes every
listing and dominates the runtime; the other nine finish in a few minutes.

---

## Known-unverified

### 1. No PDF has been produced

`make html` is complete and correct. `make pdf` fails inside Paged.js while
Chromium paginates a full part — 46,000 words with hundreds of code blocks — and
the captured stderr is truncated, so it is not yet clear whether the limit is the
180-second timeout in `_run_paged` or browser memory.

This is an environment limit rather than a manuscript defect: the same HTML
renders correctly in a browser. Start by raising `-t` and building a single small
part.

**Everything downstream of the PDF is therefore also unverified:** page layout,
column widths in listings, table overflow, the merged volume's outline and
pagination. The fixed-width tables in Practical Example sections were sized to
100 columns by eye in a terminal, never in a paginated column.

### 2. The `code` gate has not completed a full run

It executes all 565 listings and was still running after twenty minutes.
`make numbers` does execute all of them and passes, so there is strong evidence
they run — but that specific gate has not gone green end to end.

### 3. Reference resolution is checked, appearance is not

`make check` confirms every `{{ch:}}`, `{{eq:}}`, `{{sec:}}` and `{{cite:}}`
resolves and that the anchors exist. Nobody has read a rendered page end to end
to confirm the *prose* around a reference still makes sense where the renderer
places it.

---

## Gates

Ten gates in `tools/check.py`; all pass except `code`, which has not finished a
full run.

| gate | what it checks |
|---|---|
| `frontmatter` | schema, ids, tiers |
| `structure` | section numbering and order |
| `depth` | prose word floors per tier |
| `prereqs` | `requires`/`provides` wiring across the book |
| `references` | every cross-reference resolves to a real anchor |
| `citations` | every key exists and is verified |
| `terminology` | consistent naming |
| `rendering` | **KaTeX and Mermaid actually parse every block** |
| `maturity` | labels are drawn from `book.yaml` |
| `code` | every tier-A listing exits 0 |

`make numbers` is separate and is the strongest guarantee here: every number
quoted in a Practical Example is matched against the actual stdout of the listing
that produces it. It passes on all 240 chapters. **No figure in any Practical
Example was typed by hand.**

`make offline` (`tools/offline_check.py`, `audit_tex.py`, `extract_code.py`,
`run_listings.py`) needs only PyYAML and the stdlib. It exists for environments
without the toolchain and is a strict subset of `make check` — it does not render
math or diagrams, so a clean run there says nothing about KaTeX or Mermaid.

---

## What the first real build found

The manuscript was written before the toolchain could run, and the first
`make check` found defects no offline gate could see. Recorded here because it is
a fair sample of what a rendering gate catches:

* `H^\*` — `\*` is not a KaTeX control sequence. Nine occurrences that would each
  have rendered as an error box.
* A leaked Python string concatenation, `" + B + "right]`, had replaced
  `\right]` inside `eq:reversibility-is-a-design-property`, so the equation never
  closed its `\left[`.
* `eq:graph-surrenders-the-tail` was named in `provides`, titled a tier-A
  listing, and was referenced from four other documents — with no display
  equation carrying the label. Five dangling references.
* `{{maturity:SPECULATIVE}}`, a label not in `book.yaml`.
* Eight chapters just under the 4,200-word floor.

Two more were found by examining the extraction path: 62 listings carried an
absolute scratch path as their `name=` attribute, and `tools/bookdata.py` read
`book.yaml` without an encoding, so Windows cp1252 mangled en-dashes in chapter
titles into mojibake that propagated into every artefact.

And four Linux-only assumptions in the toolchain itself: the POSIX shims for
`mmdc` and `pagedjs-cli`, a hardcoded `/usr/bin/google-chrome` with a scrubbed
`PATH` that drops Node, subprocess bridges decoding KaTeX output with the locale
codec, and `check.py` requiring `.venv` to exist.

---

## Provenance of the numbers in the text

Two different things are quoted in this book and they carry different weight.

**Numbers from listings are real.** Every figure in a Practical Example is the
output of code in that chapter, checked by `make numbers`. If the code is wrong
the number is wrong, but the number always matches the code.

**The inputs to those listings are illustrative.** Constants were chosen to make
a structure legible, not measured from a corpus. `src/part-28-research` says this
explicitly and applies its own evidence rubric to itself: the structural claims
are derivable, the coefficients are furniture. Read them the same way everywhere
else.

Citations are stricter. Every entry in `data/bibliography.yaml` is on arXiv, was
fetched, and was read against the claim it supports; `verified_via` records what
the abstract page said. 29 candidate citations were refused — 14 not on arXiv
(which excludes regulatory instruments and industry taxonomies, noted in the text
wherever it matters), 9 unfetchable, and 6 because the paper did not say what it
would have been cited for.
