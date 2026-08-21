---
id: part-02-intro
status: final
---

## What this part is for

Python is the language of machine learning for historical rather than technical
reasons. It is slow, dynamically typed, and its concurrency story is awkward.
It won because NumPy {{cite:harris2020}} gave it a fast array, pandas
{{cite:mckinney2010}} gave it a dataframe, Matplotlib {{cite:hunter2007}} gave
it plots, and by the time deep learning arrived the ecosystem was already there.

That history explains the shape of this part. The Python you need for AI work is
not general-purpose Python. It is a specific dialect in which the language
itself is mostly glue, and the real work happens inside compiled libraries you
drive from it. Learning to write fast Python for this domain is largely learning
*not* to write Python — to express computation as array operations that execute
in C, rather than as loops that execute in the interpreter.

Eight chapters, aimed at a reader who can already program.

## What is here

- **Chapters 13–15** — the language and the environment. Enough Python to be
  fluent, the parts of its object model that surprise people, and how to set up
  a project that will still install in a year.
- **Chapters 16–18** — the numerical stack. NumPy, pandas and Matplotlib, which
  are the three libraries you will use every day for the rest of the book.
- **Chapters 19–20** — getting data in and out, and the engineering practices
  that separate a script from a system.

```mermaid {#fig:part2-deps caption="Dependencies within Part II. NumPy is the hinge: pandas is built on it, Matplotlib consumes it, and every framework from Part VI onward speaks its array interface."}
graph LR
  C13[13 · Fundamentals] --> C14[14 · Functions & classes]
  C14 --> C15[15 · Environments]
  C13 --> C16[16 · NumPy]
  C16 --> C17[17 · pandas]
  C16 --> C18[18 · Visualization]
  C17 --> C18
  C14 --> C19[19 · Files, APIs, SQL]
  C17 --> C19
  C15 --> C20[20 · Engineering]
  C14 --> C20
  C16 --> C20
```

If you already write Python professionally, read {{ch:py-numpy}} carefully
anyway — the views-versus-copies and broadcasting material is where experienced
programmers most often have confident misconceptions — then skim the rest and
come back to {{ch:py-environments}} when you set up your first project.

## What this part deliberately does not cover

Metaclasses, descriptors, the full method resolution order, and most of Python's
deeper object protocol. They are interesting and nothing in the remaining
twenty-six parts needs them.

Four competing package managers. {{ch:py-environments}} teaches the concepts —
isolation, resolution, locking — and then one modern toolchain, with the others
named in a comparison table.

Python 2. It has been dead since 2020.

Web frameworks beyond what {{part:22}} needs. FastAPI appears there, in
context, once there is a system worth serving.

## A note on how fast this material moves

Part I is permanent. Part II is not, and the difference is worth stating
plainly.

Between the first edition of most Python-for-data books and today, the
recommended way to manage a project has changed at least three times. The
current answer — a `pyproject.toml` manifest, a lock file, and a single fast
resolver — is better than what came before and will itself be superseded.

This part is therefore written so the durable half carries the weight. What a
virtual environment *is*, why dependency resolution is hard, why a lock file is
not the same as a requirements list, what vectorisation buys and why, what a
view shares with its parent array — none of that will change. Specific tools are
labelled with their maturity and confined to clearly marked sections, so that
when the tooling turns over again you can replace those sections without
disturbing anything around them.

> NOTE: All code in this part was executed on Python 3.12 with NumPy 2.2,
> pandas 2.2 and Matplotlib 3.10, the versions pinned in the repository. The
> current stable Python at the time of writing is 3.14; nothing here depends on
> a feature newer than 3.10, so the listings run on any supported version.

## What you should be able to do at the end

Read and write idiomatic Python without being surprised by mutability or
aliasing. Set up a project whose environment someone else can reproduce
exactly. Replace a loop over a million elements with an array expression, and
explain the speedup. Predict whether an operation returns a view or a copy.
Load, clean, join and aggregate a real dataset in pandas without corrupting it.
Produce a figure that communicates rather than decorates. Read a JSON API and a
SQL database safely. Write a test that will catch a regression, a log line that
will be useful at three in the morning, and a profile that tells you where the
time actually went.

The assignment at the end of this part exercises all of it at once.
