---
id: part-02-assessment
status: final
---

## How to use this

Same structure as {{part:1}}: a knowledge check, a substantial assignment, an
open-ended challenge, and interview preparation.

The assignment is the part that matters. Part II's material only becomes real
when you build something that has to keep working, and the project below is
deliberately the smallest thing that forces every chapter into use at once.

---

## Knowledge Check

Twenty questions. Answer without looking anything up; the chapter is noted for
checking.

1. What does `b = a` do when `a` is a list? What does `b.append(1)` then do to
   `a`, and why? [{{ch:py-fundamentals}}]

2. Why is a mutable default argument a bug? Show the fix.
   [{{ch:py-fundamentals}}]

3. Give a case where `if not x:` and `if x is None:` behave differently and the
   difference matters. [{{ch:py-fundamentals}}]

4. What is the complexity of `x in lst` versus `x in set`? What algorithmic
   difference does that make inside a loop? [{{ch:py-fundamentals}}]

5. `@d` applied to `f` is equivalent to what single line? What does
   `functools.wraps` preserve, and what breaks without it?
   [{{ch:py-functions-classes}}]

6. Why must a `@contextmanager` generator wrap its yield in `try/finally`?
   [{{ch:py-functions-classes}}]

7. Are type hints enforced at runtime? What are they for?
   [{{ch:py-functions-classes}}]

8. What is the difference between declared constraints and a lock file? Which
   does a published library commit? [{{ch:py-environments}}]

9. Why is dependency resolution NP-complete? [{{ch:py-environments}}]

10. Name four things beyond code and dependencies that must be controlled for a
    reproducible experiment. [{{ch:py-environments}}]

11. Why is a NumPy array 50-100× faster than a list for numerical work? Give two
    distinct reasons. [{{ch:py-numpy}}]

12. Which of these return a view and which a copy: `a[1:4]`, `a[a > 0]`,
    `a[[0, 2]]`, `a.T`, `a.reshape(2, 5)`? [{{ch:py-numpy}}]

13. What shape results from broadcasting `(1000, 1)` with `(1000,)`? Why is that
    dangerous? [{{ch:py-numpy}}]

14. What does `axis=1` mean in `a.sum(axis=1)`? State the general rule.
    [{{ch:py-numpy}}]

15. Why is `a.T` free while `np.ascontiguousarray(a.T)` is not?
    [{{ch:py-numpy}}]

16. Why does pandas align on index labels rather than position? Give one benefit
    and one hazard. [{{ch:py-pandas}}]

17. Why is `df[df.a > 0]["b"] = 1` unreliable? What is the correct form?
    [{{ch:py-pandas}}]

18. What does `validate="one_to_one"` protect against, and what goes wrong
    without it? [{{ch:py-pandas}}]

19. Four datasets share their mean, variance, correlation and regression line.
    What does that tell you about reporting summary statistics?
    [{{ch:py-visualization}}]

20. Why must a bar chart's axis start at zero when a line chart's need not?
    [{{ch:py-visualization}}]

**Bonus.** Why does retry without jitter make an outage worse? And: for a
CPU-bound pure-Python workload, when are processes *slower* than sequential
execution? [{{ch:py-io-apis-sql}}, {{ch:py-engineering}}]

---

## Practical Assignment

### Build a reproducible data pipeline, properly packaged

Not a notebook. A package someone else can install, run, and trust.

**The task**

Ingest data from a JSON API and a CSV file, clean and join them, engineer
features, produce a diagnostic report, and persist the result — with the whole
thing reproducible from a lock file and a seed.

```text
pipeline/
├── pyproject.toml            ← deps, tool config, all of it
├── uv.lock  (or requirements.lock)   ← committed
├── README.md
├── src/pipeline/
│   ├── __init__.py
│   ├── config.py             ← dataclasses, not dicts
│   ├── ingest.py             ← API client + CSV reader
│   ├── clean.py              ← pure transformations
│   ├── features.py           ← pure transformations
│   ├── report.py             ← figures
│   └── cli.py                ← the impure shell
├── tests/
│   ├── conftest.py           ← fixtures
│   ├── test_clean.py
│   ├── test_features.py
│   └── test_ingest.py        ← against a local mock server
└── data/                     ← gitignored
```

**Requirements, by chapter**

*Structure and environment* ({{ch:py-environments}}). `src/` layout, installable
with `pip install -e .`, all tooling configured in `pyproject.toml`, lock file
committed. `seed_everything` covering every generator in play. A
`fingerprint()` recording Python version, platform, package versions and git
commit, written alongside every output.

*Language* ({{ch:py-fundamentals}}, {{ch:py-functions-classes}}). Ingestion is a
generator pipeline that never holds the full dataset in memory. Configuration is
a `dataclass`. Transformations are pure and never mutate their inputs — write a
test that proves it. Cross-cutting concerns (timing, retry) are decorators.
Resources are released by context managers.

*Ingestion* ({{ch:py-io-apis-sql}}). A client with retry, exponential backoff
with jitter, `Retry-After` handling, and cursor pagination exposed as a
generator. Non-retryable statuses must fail immediately. CSV read with explicit
dtypes. Output written as Parquet, not CSV, and a note in the README on why.

*Transformation* ({{ch:py-numpy}}, {{ch:py-pandas}}). No Python loops over rows —
if you write one, justify it in a comment. Every join passes `validate=` and
asserts the row count. Missing values handled with a documented decision and an
indicator column where appropriate. At least one group-relative feature via
`transform`. Final frame optimised for memory with a before/after measurement.

*Reporting* ({{ch:py-visualization}}). A multi-panel figure built from functions
that each take an `ax`. Agg backend, figures closed. Log scales where the data
spans orders of magnitude. No truncated bar axes.

*Engineering* ({{ch:py-engineering}}). Structured logging throughout, level
controlled by config. A test suite covering shapes, invariants, edge cases
(empty input, single row, all-identical values, unseen category), determinism,
and one regression test. `ruff` and `mypy` clean. A profile of the full run,
with the top hotspot identified and either optimised or explained.

**Acceptance criteria**

- `pip install -e .` then `python -m pipeline.cli --config configs/demo.toml`
  works from a clean environment.
- `pytest` passes from outside the project directory.
- `ruff check` and `mypy src/` report nothing.
- Two runs with the same seed produce byte-identical output; two runs with
  different seeds do not.
- The test suite catches a deliberately introduced bug — demonstrate this by
  breaking something and showing which test fails.
- The README states the memory reduction achieved, the profile's top hotspot,
  and one thing you would do differently at 100× the data.

---

## Advanced Challenge

### Make a slow pipeline fast, and prove it

Take the pipeline you built, or any script of your own that processes a
meaningful amount of data, and optimise it with measurement at every step.

**Part A — Establish the baseline.** Build a benchmark that is representative
and repeatable. Report a distribution, not a single number
({{ch:math-inference}}) — run it enough times to give a confidence interval, and
say how many runs that took.

**Part B — Profile and predict.** Profile with `cProfile` and a line profiler.
Before optimising anything, use {{eq:amdahl}} to predict the maximum achievable
speedup from each of the top three hotspots. Write the predictions down first.

**Part C — Optimise in order of predicted value.** Work down your list. After
each change, re-measure and compare the achieved speedup against the prediction.
Where they disagree, explain why — that disagreement is where the interesting
learning is.

**Part D — Exhaust the single-threaded options before parallelising.** Try, in
order: algorithmic change, vectorisation, better dtypes, avoiding intermediate
allocations. Only then consider concurrency, and use
{{eq:processes}} to predict whether it will help before you write it.

**Part E — Find the point where it breaks.** Scale the input until something
fails — memory, time, or accuracy. Characterise the limit and say what
architectural change would move it. This is the question {{part:23}} answers at
a larger scale.

**Deliverable.** A report with the baseline distribution, the profile, the
predictions, the achieved results, and an honest account of where your
predictions were wrong.

---

## Interview Preparation

### Junior

1. What is the difference between a list and a tuple?
2. What does a virtual environment do?
3. What is a decorator?
4. Why use NumPy instead of Python lists for numerical work?
5. What is the difference between `.loc` and `.iloc`?
6. How do you handle missing values in pandas?
7. What is a unit test and why write one?

### Mid-level

8. Explain the mutable-default-argument trap and why it happens.
9. When does NumPy return a view rather than a copy, and why does it matter?
10. Explain broadcasting. Give an example where it succeeds but is wrong.
11. What is `SettingWithCopyWarning` telling you?
12. What is the difference between `agg` and `transform` in a groupby?
13. Why is retry without jitter dangerous?
14. How do you prevent SQL injection, and why is the fix also faster?
15. What is the GIL and when does it matter?

### Senior

16. How would you make a machine-learning experiment reproducible? Enumerate
    everything that must be controlled.
17. A colleague's pipeline is slow. Walk through your approach.
18. When do you choose threads, processes, and async? Give the cost model.
19. What do you test in ML code, given that outputs are stochastic?
20. Explain the N+1 query problem and how you would detect it in an unfamiliar
    codebase.
21. A join silently doubled your row count. How would you have caught it, and
    what does it do to downstream metrics?
22. Why is dependency resolution hard, and what does a lock file actually
    guarantee?

### Systems and judgement

23. Your team's notebooks are unreproducible and the models cannot be rebuilt.
    Describe what you would change, in what order, and what you would deliver
    first.
24. You must process a dataset ten times larger than memory, on one machine.
    Describe your approach and its limits.
25. When is pandas the wrong tool, and what would you reach for instead?

---

## Before moving on

You are ready for {{part:3}} when you can, without reference:

- Predict whether an operation mutates or rebinds, and whether it returns a view
  or a copy.
- Set up a project someone else can reproduce exactly.
- Replace a loop over a million elements with an array expression and explain
  the speedup.
- Join two tables without corrupting the row count, and prove it.
- Produce a diagnostic figure that answers a question a number cannot.
- Write a test that will catch a regression in a year, and a log line that will
  be useful at three in the morning.
- Choose between threads, processes, async and vectorisation from the shape of
  the workload.

{{part:3}} assumes all of this. From here on, code listings will not explain
what a DataFrame is or why the loop was vectorised.
