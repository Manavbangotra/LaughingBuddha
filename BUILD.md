# Building this book

**Status: the manuscript is complete and has never been rendered.**

Every chapter is written and passes the checks that can run without a build
toolchain. No HTML or PDF has been produced, because the machine the manuscript
was written on had neither Node nor the full Python build stack. This file
records exactly what was verified, what was not, and what to expect on the
first real build.

Read [Known-unverified](#known-unverified) before you trust anything in
`build/`.

---

## What exists

| | |
|---|---|
| Chapters | **240 / 240** |
| Part introductions | 28 / 28 |
| Part assessments | 28 / 28 |
| Prose | ~1,746,800 words |
| Tier-A listings | 565, extracted into `code/` |
| Citations | 331, all `verified: true` |
| Display equations | 3,554, carrying 2,908 labels |
| Mermaid diagrams | 151 |

Not yet written: **15 projects**, the **6-chapter capstone**, **11 appendices**.
`book.yaml` already has their slots, so they appear in the build as missing
documents rather than as errors.

---

## Prerequisites

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
npm install          # katex, mermaid-cli, pagedjs-cli
```

Node 18+ and Python 3.11+. `npm install` is not optional: math, diagrams and
PDF pagination are all Node-side, and they are precisely the parts that have
never run.

---

## The first build, in the order that finds problems soonest

```bash
make offline      # stdlib + PyYAML only; should pass on a fresh clone
make check        # the real gate — EXPECT FAILURES HERE FIRST
make numbers      # executes all 565 listings; slow, ~an hour
make html
make pdf
make book
```

`make offline` passing means very little about `make check`. It is a strict
subset that skips exactly the parts that were never exercised.

---

## Known-unverified

These are ordered by how likely they are to bite, and every one of them is a
consequence of the toolchain being absent rather than of a known defect.

### 1. No equation has ever been rendered

3,554 display equations and a large number of inline ones have been written
against the KaTeX subset without KaTeX ever parsing them. A systematic error —
an unsupported macro, an environment KaTeX does not implement — would be
present in all 240 chapters and nothing has looked for it.

`tools/audit_tex.py` catches only one specific failure: LaTeX backslashes
collapsing into control characters (`\t`, `\r`, `\f`, `\b`) when a listing was
written through a shell heredoc. It reports CLEAN on all 301 files. That is a
much narrower claim than "the math renders".

**Expect the first `make check` to fail here.** Fix the macro, re-run, repeat.

### 2. No diagram has ever been rendered

151 Mermaid blocks, never parsed by mermaid-cli. Same argument: a systematic
syntax error is plausible and unmeasured.

### 3. Page layout is entirely untested

No PDF exists. Column widths in the fixed-width listing output were chosen to
fit 100 characters, which was checked by eye in a terminal, not in a paginated
column. Wide tables in Practical Example sections are the most likely place for
overflow.

### 4. `tools/report.py` cannot run without lxml

It imports `build.py`, which imports lxml at module scope. Not a defect, just a
thing that will fail on a partial install.

### 5. Cross-part reference resolution is only half-checked

`tools/offline_check.py` verifies that every `{{ch:...}}`, `{{eq:...}}`,
`{{sec:...}}` and `{{cite:...}}` resolves to something that exists. It does not
build the anchor graph the real renderer builds, so an anchor that exists but
lands in the wrong document would pass here and fail in `make check`.

---

## What *was* verified, and how

Offline substitutes, written because the real gates could not run. They are in
`tools/` and wired into `make offline`.

| tool | what it actually checks |
|---|---|
| `offline_check.py` | frontmatter schema, section numbering and order, `requires`/`provides` wiring, every cross-reference and citation key resolves, prose word floors per tier |
| `audit_tex.py` | control characters from collapsed LaTeX backslashes |
| `extract_code.py` | stand-in for `make code` — see below |
| `run_listings.py` | executes one chapter's tier-A listings and reports failures |
| `verify_numbers.py` | **the real tool.** Every number quoted in a Practical Example is matched against the actual stdout of the listing that produces it |

`verify_numbers.py` is the strongest guarantee in the repository and it does run
here. It passes on all 240 chapters. No figure in any Practical Example was
typed by hand; each one was read back from an executed listing.

### On `extract_code.py`

`tools/build.py` imports lxml at module scope, so `build.py code` cannot run
without the full stack. `tools/extract_code.py` reproduces the same contract —
same fence filter, same 1-based per-document index, same paths, same four-line
banner including the PEP 263 cookie.

It was validated by regenerating the 309 files that the real toolchain had
produced before and diffing byte-for-byte. After the encoding fix below, the
match is exact. All 565 extracted files parse under `ast.parse`, with no
filename collisions.

Once you have the real toolchain, prefer `make code`. `extract_code.py` exists
for environments that do not.

---

## Fixed while writing this file

Two defects that only surfaced when the build path was examined:

**Absolute paths leaked into 62 fence headers.** The splice tool used to place
listings into chapters wrote its full input path as the listing's `name=`
attribute, so 62 listings across parts 24–28 carried
`name=C:/Users/.../scratchpad/xx1`. This would have produced invalid extraction
paths, crashed `make code` on Windows, and printed an absolute scratch path as
the visible listing name in the rendered book. Each has been renamed to the
semantic anchor the listing establishes — for example
`name=compliance-cost-is-a-step-function`. No path leaks remain in `src/`.

**`tools/bookdata.py` read `book.yaml` without an encoding.** On Windows the
default is cp1252, which turns the en-dash in a chapter title into mojibake that
then propagates into every extracted file's banner and, on a real build, into
every rendered heading. Now reads UTF-8 explicitly.

Both were invisible to the offline gates and would have been caught by the first
real build. They are a fair indication of the class of thing still waiting in
the unverified list above.

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
