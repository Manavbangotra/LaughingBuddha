# Part II — Python for Data and AI: research notes

Research pass run 2026-08-13, before writing.

## State of the ecosystem, and what it changes

Unlike Part I, Part II covers material that genuinely moves. Three findings from
this pass changed what the chapters teach.

### 1. Packaging has consolidated around `pyproject.toml` and `uv`

Verified against the uv documentation and corroborated across several 2026
ecosystem surveys. The situation the previous generation of tutorials describes
— `requirements.txt`, `virtualenv`, `pip`, with `conda` or `poetry` as the
alternatives — is no longer the default advice for new projects.

The current position is: **`pyproject.toml` is the standard project manifest**,
lock files are expected, and `uv` (a single Rust binary from Astral) subsumes
`pip`, `virtualenv`, `pyenv` and much of `poetry` at 10-100× the speed.

This directly reshapes {{ch:py-environments}}. That chapter now teaches
`pyproject.toml` as the primary artefact and `uv` as the primary tool, while
still covering `venv` and `pip` — because they are in the standard library,
because every existing project uses them, and because a reader who understands
only the fast wrapper will be stuck the first time they meet a repository that
predates it. The chapter is explicit that `uv` is
{{maturity:EMERGING}} rather than established: it is dominant in new projects
and young enough that a book should not assume it will still be dominant in
five years.

> The durable content is the *concepts* — isolation, resolution, locking,
> reproducibility — and those are taught first, with tools presented as
> implementations of them. That is the "make fundamentals permanent, quarantine
> the 2026-specific" principle from the book's architecture applied to the part
> where it matters most.

### 2. Python version status

Verified against the official CPython developer guide on 2026-08-13:

| Version | Status | EOL |
|---|---|---|
| 3.15 | prerelease | — |
| 3.14 | bugfix (latest stable, released 2025-10-07) | Oct 2030 |
| 3.13 | bugfix | Oct 2029 |
| 3.12 | security only | Oct 2028 |
| 3.11 | security only | Oct 2027 |
| 3.10 | security only | Oct 2026 |
| 3.9 | end of life 2025-10-31 | — |

The book targets **3.12+** and says so. Worth noting honestly: this repository's
own virtual environment runs 3.12.3, which is in security-only status. Every
Tier A listing in Part II was executed on it. Nothing in Part II depends on a
feature newer than 3.10, so the code runs on anything currently supported.

### 3. Free-threading is real but not yet the default

PEP 703 (Sam Gross, sponsored by Łukasz Langa) is **Final** — accepted by the
Steering Council — with a first implementation target of Python 3.13, available
through a separate `--disable-gil` build. Reported overhead is 5-6%
single-threaded and 7-8% multi-threaded in that build; default builds are
unchanged.

This matters for {{ch:py-engineering}}, which covers concurrency. The
long-standing advice — "threads for I/O, processes for CPU, because the GIL" —
is still correct for the default build a reader will have, and the chapter
teaches it as such. But it now carries an explicit
{{maturity:EXPERIMENTAL}} note that the constraint is being removed rather than
being a permanent fact about Python. Presenting the GIL as immutable would date
badly, and presenting free-threading as available would be wrong for almost
every reader today.

## Library versions used

Pinned in `requirements.txt` and used for every Tier A listing:
NumPy 2.2.3, pandas 2.2.3, Matplotlib 3.10.0, SciPy 1.15.2, PyTorch 2.13.0
(CPU), pytest 9.1.1.

NumPy is on the 2.x line, which changed several long-standing behaviours from
1.x — most visibly the scalar representation in `repr` and the promotion rules
(NEP 50). Chapter 16 mentions this where a reader following an older tutorial
would otherwise be confused, and does not dwell on it.

## References checked

All verified against primary or official sources on 2026-08-13.

| Key | What | Checked against |
|---|---|---|
| `harris2020` | NumPy, Nature 585:357-362 | numpy.org/citing-numpy — full 26-author list, DOI |
| `mckinney2010` | pandas, SciPy 2010 proceedings | pandas.pydata.org/about/citing.html — pages 56-61, DOI, editors |
| `hunter2007` | Matplotlib, CiSE 9(3):90-95 | Crossref API record for DOI 10.1109/MCSE.2007.55 |
| `pep8` | Style Guide for Python Code | peps.python.org/pep-0008 — authors, created 2001-07-05, Status: Active |
| `pep484` | Type Hints | peps.python.org/pep-0484 — authors, targets 3.5, Status: Final |
| `pep703` | Optional GIL | peps.python.org/pep-0703 — author, Status: Final, 3.13 target, experimental |

Nature and IEEE both refused direct fetches (auth redirect and 403). The NumPy
and Matplotlib citations were therefore verified through the projects' own
official citation page and the Crossref record respectively — both authoritative
for bibliographic metadata, and both recorded in `verified_via`.

## Deliberate omissions

- **`conda`.** Still widely used in scientific computing and unavoidable for
  some GPU stacks, so it is mentioned and its niche explained — but it is not
  taught as a primary workflow, because for the projects in this book it is
  heavier than what is needed.
- **`poetry`, `pipenv`, `pdm`.** Named once in a comparison table. Teaching four
  package managers would waste the reader's time.
- **Python 2 differences.** Gone. It has been dead for six years.
- **Deep OOP.** Chapter 14 covers classes to the depth this book actually uses —
  `dataclass`, a little inheritance, `__init__`/`__repr__`/`__call__`, context
  managers. Metaclasses, descriptors and the full MRO are named and skipped;
  nothing later in the book needs them.
- **`async` beyond what serving requires.** Chapter 20 teaches the concurrency
  model and enough `asyncio` to understand FastAPI in {{part:22}}. Event-loop
  internals are not covered.

## Chapter-level notes

**Ch 13 (fundamentals)** assumes programming experience, not Python experience.
It moves fast and concentrates on the things that trip up people arriving from
other languages: mutability and aliasing, the mutable-default-argument trap,
truthiness, and comprehensions.

**Ch 16 (NumPy)** is the most important chapter in the part. Vectorisation,
broadcasting, views versus copies, and dtype are the concepts that make
everything from {{part:4}} onward tractable, and the views-versus-copies
distinction is the single most common source of silent bugs in numerical Python.

**Ch 17 (pandas)** deliberately teaches the sharp edges — `SettingWithCopyWarning`,
index alignment, the merge-cardinality trap — rather than presenting a clean
tour. Those are what a practitioner actually hits.

**Ch 18 (visualisation)** teaches the object-oriented Matplotlib interface, not
`pyplot` state, because the state machine breaks down the moment you build
anything programmatic.

**Ch 19 (I/O, APIs, SQL)** uses only the standard library for its runnable code —
`json`, `urllib`, `sqlite3`, and a local `http.server` in a background thread —
so every listing is Tier A and executes offline. Real HTTP against a third-party
API is Tier B and labelled as not executed.
