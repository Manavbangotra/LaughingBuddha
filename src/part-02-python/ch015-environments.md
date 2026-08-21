---
id: py-environments
number: 15
part: II
tier: focused
status: reviewed
requires: [py-functions-classes]
provides: [virtual-environment, dependency-resolution, lock-file,
           semantic-versioning, reproducibility-term, random-seed]
citations: [pep8]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what a virtual environment is and why isolation is not optional.
2. Explain why dependency resolution is genuinely hard, and what a resolver
   actually does.
3. Distinguish declared constraints from a lock file, and say why both exist.
4. Read and write a `pyproject.toml`.
5. Choose between the available toolchains and justify the choice.
6. Lay out a project so that imports work from anywhere.
7. Make an ML experiment reproducible, and enumerate the sources of
   nondeterminism that remain.

## 2. Why This Matters

"It works on my machine" is the oldest failure in software, and machine learning
has an aggravated form of it. A model is a function of code, data, hyperparameters
*and* library versions, and the last of those is the one people forget to
record. A NumPy minor release changes a default; a scikit-learn upgrade changes
a random-number stream; a CUDA version bump changes a kernel's summation order
by a few bits. Any of these can move a result, and none of them are in your git
history.

The concrete costs are worth naming. An experiment you cannot rerun is not a
result, it is an anecdote. A model you cannot rebuild is a liability the moment
it needs retraining. A deployment whose environment differs from the training
environment is where the interesting production bugs live
({{ch:ops-versioning}}).

There is also a plain practical reason. Two projects will eventually need
incompatible versions of the same library, and without isolation, installing one
breaks the other.

> NOTE: This chapter's *concepts* are durable — isolation, resolution, locking,
> reproducibility. Its *tools* are the fastest-moving material in the book. The
> chapter teaches the concepts first and confines specific tools to
> {{sec:5-formal-explanation}}, so the tooling half can be replaced without
> disturbing the rest.

## 3. Prerequisites

{{ch:py-functions-classes}} for modules, packages and imports.

## 4. Intuitive Explanation

### 4.1 Isolation

Installing a package system-wide puts it in one shared directory. Every project
sees the same version. When project A needs `pandas 1.5` and project B needs
`pandas 2.2`, one of them is broken, and you find out at import time.

A {{term:virtual-environment}} is a private copy of that directory per project.
Activating it puts its `bin` first on `PATH`, so `python` and `pip` resolve to
the project's own copies and installs land in the project's own
`site-packages`.

That is the entire mechanism. It is a directory and a `PATH` manipulation — no
containers, no virtualisation, nothing clever.

```text
~/.venvs/project-a/          ~/.venvs/project-b/
  bin/python  ──────────┐      bin/python
  lib/site-packages/    │      lib/site-packages/
    pandas 1.5.3        │        pandas 2.2.3
    numpy 1.24.0        │        numpy 2.2.3
                        │
   PATH puts one of these first; that decides which `python` you get
```

### 4.2 Resolution is a search problem

You ask for `pandas` and `scikit-learn`. Each depends on `numpy`, with
different constraints. The installer must find one version of `numpy`
satisfying both — and then repeat for every transitive dependency, where a
choice made for one package constrains what is available for another.

This is a constraint-satisfaction problem, and in general it is **NP-complete**.
Real resolvers backtrack, and on a large dependency graph they can take minutes
or fail with a conflict that has no solution at all.

Two consequences follow. Resolution is slow, which is why a faster resolver was
a big enough improvement to reorganise the ecosystem around it. And resolution
is *not deterministic across time*: run the same install six months later and
new releases will have appeared, so you may get different versions. That second
point is what lock files exist for.

### 4.3 Constraints versus locks

These are different artefacts doing different jobs, and conflating them is the
most common error in this area.

{#tbl:constraints-vs-lock caption="Declared constraints and a lock file answer different questions. A project needs both."}

| | Declared constraints | Lock file |
|---|---|---|
| Lives in | `pyproject.toml` | `uv.lock`, `poetry.lock`, `requirements.txt` (pinned) |
| Says | "I need pandas 2.x" | "install exactly pandas 2.2.3, hash abc…" |
| Written by | a human | the resolver |
| Includes transitive deps | no | yes, every one |
| Purpose | express what you actually need | recreate an exact environment |
| Committed to git | yes | yes, for applications |

A `requirements.txt` containing `pandas>=2.0` is a constraint file pretending to
be a lock file, and it is why environments drift. A lock file pins the complete
transitive graph — often two hundred packages for a project that declared five.

> IMPORTANT: For a **library** you publish, keep constraints loose so consumers
> can resolve alongside their other dependencies, and do not commit a lock file.
> For an **application** or an experiment, pin hard and commit the lock. Machine
> learning projects are almost always the second case.

## 5. Formal Explanation

### 5.1 Creating an environment

The standard-library way, available everywhere with no installation:

```bash {tier=C name=venv-stdlib}
python -m venv .venv                 # create
source .venv/bin/activate            # activate (Linux/macOS)
.venv\Scripts\activate               # activate (Windows)
python -m pip install pandas         # install into it
deactivate                           # leave
```

Two conventions worth adopting. Put the environment in `.venv` inside the
project, so it is obvious which project it belongs to, and add `.venv/` to
`.gitignore` — an environment is build output, not source.

> PRODUCTION TIP: Prefer `python -m pip install` over bare `pip install`. The
> `-m` form guarantees you are using the pip belonging to the interpreter you
> just invoked. A bare `pip` may be a different one from elsewhere on `PATH`,
> and installing into the wrong environment then wondering why the import fails
> is a rite of passage worth skipping.

### 5.2 `pyproject.toml`

The standard project manifest. It replaces `setup.py`, `setup.cfg`,
`requirements.txt`, and most tool-specific config files.

```toml {tier=C name=pyproject}
[project]
name = "my-ml-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0,<3",
    "pandas>=2.2,<3",
    "scikit-learn>=1.5",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
```

One file, holding metadata, dependencies, build configuration and tool settings.
Note the version constraints: `>=2.0,<3` says "any 2.x", which relies on
{{term:semantic-versioning}} — the convention that a major-version bump signals
a breaking change. It is a promise maintainers make, not a rule anything
enforces, and it is broken often enough that a lock file is still necessary.

### 5.3 Toolchains

{#tbl:toolchains caption="Python environment toolchains as of 2026. The concepts are identical; they differ in speed, scope, and how much they do for you."}

| Tool | Scope | Notes |
|---|---|---|
| `venv` + `pip` | environments, installing | Standard library. Always available. No locking, no resolution guarantees. |
| `uv` | everything: Python versions, environments, resolution, locking, running | A single Rust binary. 10-100× faster than pip. {{maturity:EMERGING}} |
| `poetry` | environments, resolution, locking, publishing | Mature, slower, its own config dialect predating some standards. {{maturity:MATURE}} |
| `conda` / `mamba` | environments including non-Python binaries | Handles CUDA, MKL, compilers. Heavier. Its niche is real. {{maturity:MATURE}} |

For a new project in 2026, `uv` is the recommendation:

```bash {tier=C name=uv-workflow}
uv init my-project            # create pyproject.toml and a project skeleton
uv add pandas scikit-learn    # add deps, resolve, install, update the lock
uv add --dev pytest ruff      # development-only dependencies
uv run python train.py        # run inside the environment, no activation
uv sync                       # recreate the environment exactly from the lock
```

`uv sync` is the important one: given `uv.lock`, it reproduces the environment
byte-for-byte on another machine.

> WARNING: `uv` is labelled {{maturity:EMERGING}} deliberately. It is dominant
> among new projects and is a genuine improvement, but it is young, and the
> previous three "obvious" answers to this problem were also obvious at the
> time. Learn what the tool *does* — resolve, lock, sync — so that switching
> costs you an afternoon rather than a re-education. The `venv` and `pip` route
> is not going away; it is in the standard library and every existing project
> uses it.

**`conda` deserves its niche.** It installs non-Python binaries — CUDA
toolkits, MKL, compilers — which pip cannot. If your GPU stack demands it, use
it. For everything in this book up to {{part:23}}, it is more than you need.

### 5.4 Project layout

```text
my-ml-project/
├── pyproject.toml
├── uv.lock                  ← committed
├── README.md
├── .gitignore               ← .venv/, __pycache__/, data/, *.ipynb_checkpoints
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       └── train.py
├── tests/
│   └── test_features.py
├── notebooks/               ← exploration only; nothing imports from here
├── data/                    ← gitignored; tracked separately (Chapter 206)
└── configs/
```

The `src/` layout is worth adopting over putting the package at the top level.
With `src/`, the package can only be imported if it has been *installed* —
usually with `pip install -e .` or `uv sync`. That forces you to notice a
missing dependency or a broken package configuration immediately, rather than
discovering it when someone else clones the repository.

> PRODUCTION TIP: Never manipulate `sys.path` to make imports work. It is a
> symptom of a project that has not been packaged, and it fails differently in
> notebooks, in tests, and in production. Install the package instead — an
> editable install (`pip install -e .`) points at your working tree, so edits
> take effect without reinstalling.

### 5.5 Reproducibility

{{term:reproducibility-term}} in machine learning needs more than a lock file.
Six things must be controlled:

1. **Code** — git commit.
2. **Dependencies** — lock file.
3. **Python and system libraries** — recorded, or containerised.
4. **Data** — versioned or content-hashed ({{ch:ops-versioning}}).
5. **Configuration** — hyperparameters recorded with the run.
6. **Randomness** — a fixed {{term:random-seed}} for every generator in play.

Point 6 is the one people implement halfway. There are typically several
independent generators: Python's `random`, NumPy's, the framework's, and the
data loader's per-worker generators. Seeding one is not seeding the others.

```python {tier=C name=seeding}
import os, random
import numpy as np

def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

> WARNING: Even with every seed fixed, GPU results are frequently not
> bit-identical between runs. Many CUDA kernels use atomic accumulation, whose
> summation order depends on thread scheduling, and floating-point addition is
> not associative. Frameworks offer a deterministic mode that forces slower,
> ordered kernels. Expect to pay for it in throughput, and decide deliberately
> whether you need bit-exactness or only statistical reproducibility
> ({{ch:mle-reproducibility}}).

## 6. Mathematical Foundation

### 6.1 Why resolution is hard

Model the problem formally. Let $P$ be the set of packages and $V_p$ the
versions available for package $p$. A solution is an assignment choosing at most
one version per package, satisfying every constraint contributed by every chosen
version.

The subtlety is that **constraints depend on the choices**: selecting
`pandas 2.2.3` introduces its requirement `numpy>=1.26`, whereas selecting
`pandas 1.5.3` introduces a different one. The constraint set is not known in
advance; it is discovered as the search proceeds.

This is equivalent to boolean satisfiability, and therefore NP-complete. The
search space is

$$
\prod_{p \in P}\big(\lvert V_p \rvert + 1\big)
$$ (eq:resolution-space)

where the $+1$ allows for omitting an optional package. With 200 transitive
dependencies averaging 50 released versions each, that is $51^{200}$ — a number
with 341 digits.

Real resolvers do not enumerate it. They use backtracking with conflict-driven
clause learning, exactly as SAT solvers do: on hitting a conflict, derive a
constraint explaining it, and use that to prune large regions of the search
space. The practical consequences you will observe are that resolution is
sometimes slow, that error messages are sometimes long chains of "because X
requires Y which requires Z", and that some requirement sets genuinely have no
solution.

> MATH NOTE: This also explains why installing packages one at a time can leave
> you in a state that a single combined resolution would have avoided. Each
> `pip install` resolves against what is already present rather than
> re-solving the whole problem, so an early choice can foreclose a later one.
> Declare all dependencies together and let the resolver see the whole problem.

### 6.2 What semantic versioning promises

Under {{term:semantic-versioning}}, `MAJOR.MINOR.PATCH`:

$$
\text{MAJOR} \uparrow \;\Rightarrow\; \text{breaking change}
$$
$$
\text{MINOR} \uparrow \;\Rightarrow\; \text{backward-compatible addition}
$$
$$
\text{PATCH} \uparrow \;\Rightarrow\; \text{backward-compatible fix}
$$

The constraint `>=2.2,<3` therefore means "any release I should be compatible
with".

Two caveats keep this from being sufficient on its own. It is a **social
convention**, not a technical guarantee — maintainers break it, sometimes
accidentally. And "breaking" is judged from the maintainer's perspective: a
bug fix that changes a floating-point result in the sixteenth digit is a patch
release by their standards and can change your model's output.

That gap between the promise and reality is exactly the space a lock file
occupies.

### 6.3 Costing reproducibility

What each level of rigour buys, and what it costs:

{#tbl:reproducibility-levels caption="Levels of reproducibility. Most research code sits at level 1 and reports results as though it were at level 4."}

| Level | Controls | Recreates | Cost |
|---|---|---|---|
| 0 | nothing | nothing | none |
| 1 | code (git) | the logic | none |
| 2 | + lock file | the environment | minutes to set up |
| 3 | + seeds | the exact run, on CPU | a few lines |
| 4 | + deterministic kernels | the exact run, on GPU | 10-30% throughput |
| 5 | + container image | the whole system | build and registry overhead |

Level 3 is the right default for research. Level 4 is worth it when debugging a
discrepancy or certifying a result. Level 5 is what production needs
({{ch:inf-kubernetes}}).

The honest observation is that most published machine-learning results sit at
level 1 while being reported with a precision that would require level 4.

## 7. Implementation

```python {tier=A name=environment-inspection}
"""Inspecting an environment, and demonstrating what reproducibility requires.

Everything here runs against the environment actually executing it — the same
introspection you should record with every experiment.
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np

# --- where am I running? -----------------------------------------------------
print("=" * 66)
print("environment")
print("=" * 66)
print(f"python executable : {sys.executable}")
print(f"python version    : {sys.version.split()[0]}")
print(f"platform          : {platform.platform()}")
print(f"in a virtualenv   : {sys.prefix != sys.base_prefix}")
print(f"site-packages     : "
      f"{[p for p in sys.path if 'site-packages' in p][:1]}")

# --- what is installed? ------------------------------------------------------
interesting = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
               "torch", "pytest"]
print(f"\n{'package':<16} {'version':>12}")
installed = {}
for name in interesting:
    try:
        v = md.version(name)
        installed[name] = v
    except md.PackageNotFoundError:
        v = "not installed"
    print(f"{name:<16} {v:>12}")

print(f"\ntotal distributions installed: "
      f"{len(list(md.distributions()))}")

# --- a dependency graph is bigger than you declared -------------------------
def requirements_of(pkg):
    try:
        return md.requires(pkg) or []
    except md.PackageNotFoundError:
        return []


print(f"\ndirect requirements of pandas:")
for req in requirements_of("pandas")[:6]:
    print(f"  {req}")
print("  ...each of which has its own, transitively — which is what a")
print("  resolver must satisfy simultaneously (eq. 15.1).")

# --- an environment fingerprint you can record with a run -------------------
def environment_fingerprint() -> dict:
    """Everything needed to explain a result, minus the data."""
    dists = sorted(
        (d.metadata["Name"], d.version)
        for d in md.distributions()
        if d.metadata["Name"]
    )
    blob = json.dumps(dists, sort_keys=True).encode()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_packages": len(dists),
        "env_hash": hashlib.sha256(blob).hexdigest()[:16],
    }


fp = environment_fingerprint()
print(f"\nfingerprint to store alongside every experiment:")
for k, v in fp.items():
    print(f"  {k:<12} {v}")
print("Two runs with different env_hash values are not comparable.")

# --- seeding: one generator is not all of them -------------------------------
print("\n" + "=" * 66)
print("reproducibility: seeding every generator, not just one")
print("=" * 66)


def partial_seed(seed=42):
    random.seed(seed)                 # only stdlib random


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)              # legacy global NumPy state
    return np.random.default_rng(seed)   # and a modern explicit generator


def draw():
    return (round(random.random(), 6),
            round(float(np.random.random()), 6))


partial_seed(); a = draw()
partial_seed(); b = draw()
print(f"seeding only random: {a} then {b}")
print(f"  stdlib matches: {a[0] == b[0]}, numpy matches: {a[1] == b[1]}"
      "   <- numpy drifted")

seed_everything(); c = draw()
seed_everything(); d = draw()
print(f"seeding both       : {c} then {d}")
print(f"  stdlib matches: {c[0] == d[0]}, numpy matches: {c[1] == d[1]}")
assert c == d

# --- prefer an explicit Generator to the global legacy state ----------------
print("\nglobal np.random.seed sets hidden process-wide state; a Generator")
print("is explicit and cannot be disturbed by a library you called:")
rng1 = np.random.default_rng(7)
rng2 = np.random.default_rng(7)
print(f"  rng1: {np.round(rng1.normal(size=3), 4)}")
np.random.seed(999)                   # a library does this behind your back
print(f"  rng2: {np.round(rng2.normal(size=3), 4)}   <- unaffected")

# --- floating-point non-associativity, the root of GPU nondeterminism -------
print("\n" + "=" * 66)
print("why fixed seeds are still not bit-identical on a GPU")
print("=" * 66)
rng = np.random.default_rng(0)
vals = rng.normal(size=100_000).astype(np.float32)
forward = np.float32(0.0)
for v in vals:
    forward += v
backward = np.float32(0.0)
for v in vals[::-1]:
    backward += v
print(f"summing forwards : {forward:.10f}")
print(f"summing backwards: {backward:.10f}")
print(f"identical        : {forward == backward}")
print("Floating-point addition is not associative. GPU kernels accumulate in")
print("a thread-scheduling-dependent order, so the sum differs run to run —")
print("with every seed fixed. Deterministic mode forces an order, and costs")
print("throughput.")

# --- what pip freeze gives you, and what it does not -------------------------
print("\n" + "=" * 66)
print("pinned list vs lock file")
print("=" * 66)
out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                     capture_output=True, text=True, timeout=120)
lines = [l for l in out.stdout.splitlines() if l and not l.startswith("-e")]
print(f"pip freeze lists {len(lines)} pinned packages, e.g.:")
for line in lines[:4]:
    print(f"  {line}")
print("\nThis pins versions but records no hashes, no resolution metadata,")
print("and no marker for which packages you actually asked for. A real lock")
print("file records all three, which is what makes `sync` exact.")
```

## 8. Practical Example

Setting up a project properly takes ten minutes once and saves days later. The
following builds one end to end, in a temporary directory, and verifies that it
works.

```python {tier=A name=project-scaffold}
"""Scaffold a correctly-structured project and prove the layout works.

Creates the src/ layout of section 5.4, installs the package into the current
environment in editable mode, and imports it — demonstrating why the src/
layout catches packaging mistakes that a flat layout hides.
"""
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="scaffold-"))
pkg = root / "src" / "demo_project"
pkg.mkdir(parents=True)
(root / "tests").mkdir()

# --- pyproject.toml: the single manifest ------------------------------------
(root / "pyproject.toml").write_text(textwrap.dedent("""
    [project]
    name = "demo-project"
    version = "0.1.0"
    requires-python = ">=3.10"
    dependencies = ["numpy>=1.24"]

    [project.optional-dependencies]
    dev = ["pytest>=8"]

    [build-system]
    requires = ["setuptools>=68"]
    build-backend = "setuptools.build_meta"

    [tool.setuptools.packages.find]
    where = ["src"]
""").lstrip())

(pkg / "__init__.py").write_text('__version__ = "0.1.0"\n')
(pkg / "features.py").write_text(textwrap.dedent('''
    """A pure transformation — trivially testable (Chapter 14)."""
    import numpy as np


    def standardise(x: np.ndarray) -> np.ndarray:
        """Centre and scale to unit variance."""
        x = np.asarray(x, dtype=float)
        std = x.std()
        return (x - x.mean()) / (std if std else 1.0)
''').lstrip())

(root / "tests" / "test_features.py").write_text(textwrap.dedent('''
    import numpy as np
    from demo_project.features import standardise


    def test_standardise_gives_zero_mean_unit_variance():
        out = standardise([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(out.mean(), 0.0)
        assert np.isclose(out.std(), 1.0)


    def test_standardise_handles_constant_input():
        out = standardise([5.0, 5.0, 5.0])
        assert np.allclose(out, 0.0)
''').lstrip())

(root / ".gitignore").write_text(".venv/\n__pycache__/\n*.egg-info/\ndata/\n")

print("created project:")
for p in sorted(root.rglob("*")):
    if "__pycache__" not in str(p) and "egg-info" not in str(p):
        print(f"  {p.relative_to(root)}")

# --- the src/ layout means the package is NOT importable until installed ----
probe = [sys.executable, "-c", "import demo_project; print('imported')"]
before = subprocess.run(probe, cwd=root, capture_output=True, text=True)
print(f"\nimport from the project root BEFORE installing: "
      f"{'succeeded' if before.returncode == 0 else 'failed (as intended)'}")
print("  With a flat layout this would have succeeded by accident, hiding a")
print("  packaging error until someone else cloned the repository.")

# --- install it, editable ----------------------------------------------------
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"],
    cwd=root, capture_output=True, text=True, timeout=600)
print(f"\neditable install: {'ok' if install.returncode == 0 else 'FAILED'}")
if install.returncode != 0:
    print(install.stderr[-500:])

after = subprocess.run(probe, cwd=root, capture_output=True, text=True)
print(f"import after installing: {after.stdout.strip() or after.stderr[-200:]}")

# --- and now the tests run from anywhere, not just the project root ---------
tests = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"),
                        "-q", "--no-header"],
                       cwd=tempfile.gettempdir(), capture_output=True,
                       text=True, timeout=600)
print(f"\npytest run from a DIFFERENT directory:")
print("  " + "\n  ".join(tests.stdout.strip().splitlines()[-3:]))

# --- editable means edits take effect without reinstalling ------------------
(pkg / "features.py").write_text(
    (pkg / "features.py").read_text().replace('"""Centre and scale to unit variance."""',
                                              '"""EDITED docstring."""'))
check = subprocess.run(
    [sys.executable, "-c",
     "from demo_project.features import standardise; print(standardise.__doc__)"],
    cwd=tempfile.gettempdir(), capture_output=True, text=True)
print(f"\nafter editing the source, with no reinstall: {check.stdout.strip()!r}")

# --- clean up ----------------------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                "demo-project"], capture_output=True, timeout=300)
import shutil
shutil.rmtree(root, ignore_errors=True)
print("\nuninstalled and cleaned up.")
```

## 9. Common Mistakes

**Installing into the system Python.** Breaks other projects and sometimes the
operating system's own tooling. Always use an environment.

**Committing `.venv/`.** It is build output, it is large, and it is
platform-specific.

**Treating `requirements.txt` with `>=` as a lock file.** It pins nothing.
Environments drift silently.

**Not committing the lock file for an application.** Then nobody, including
future you, can recreate the environment.

**Committing a lock file for a published library.** It forces your resolution
onto consumers who have their own constraints.

**Using bare `pip` instead of `python -m pip`.** You may install into a
different environment than you think.

**Manipulating `sys.path` to fix imports.** Install the package instead.

**Seeding only one generator.** Python's `random`, NumPy's global state, an
explicit `Generator`, and the framework's are all separate.

**Using `np.random.seed` rather than an explicit `Generator`.** The global state
is process-wide and any library you call can disturb it.

**Assuming fixed seeds give bit-identical GPU results.** They do not, for the
floating-point reason demonstrated in {{sec:7-implementation}}.

**Installing packages one at a time and expecting a coherent resolution.** Each
install resolves against the current state, not the whole problem.

## 10. Connection to Previous Chapters

{{ch:py-functions-classes}} covered modules and imports; this chapter packages
them so those imports resolve from anywhere. {{ch:py-fundamentals}} noted that
data pipelines must be reproducible; {{sec:5-formal-explanation}} lists what that
actually requires.

{{ch:math-inference}} is why reproducibility matters more than it appears:
comparing two runs is a statistical claim, and if the environments differ, the
comparison has an uncontrolled variable in it.

Forward within Part II: {{ch:py-engineering}} adds testing, linting and type
checking, all configured in the same `pyproject.toml`.

Beyond Part II: {{ch:mle-reproducibility}} extends this to experiment tracking
and dataset versioning; {{ch:ops-versioning}} to model and data versioning in
production; {{ch:inf-kubernetes}} to containers, which are level 5 of
{{tbl:reproducibility-levels}}.

Code style throughout follows {{cite:pep8}}.

## 11. Exercises

**Beginner**

1. Create a virtual environment, activate it, install `numpy`, and verify with
   `sys.prefix` that you are inside it.
2. Write a minimal `pyproject.toml` declaring one dependency and requiring
   Python 3.12 or newer.
3. Explain the difference between `pandas>=2.0` and `pandas==2.2.3`.
4. What belongs in `.gitignore` for a Python ML project? List six entries and
   justify each.
5. Why is `python -m pip install` safer than `pip install`?

**Intermediate**

6. Given constraints `A>=1.0`, `A<2.0` and `A==1.5`, is there a solution? Now
   add `A>1.5`.
7. Explain why installing packages one at a time can reach a state that a single
   resolution would have avoided.
8. Write `seed_everything` covering `random`, NumPy and — if installed — PyTorch,
   and demonstrate it works.
9. Explain the `src/` layout's advantage over a flat one, with a concrete failure
   the flat layout hides.
10. Your colleague's results differ from yours on identical code and data. List
    six candidate causes in the order you would check them.
11. When should a lock file be committed, and when should it not?

**Advanced**

12. Explain why dependency resolution is NP-complete, by sketching a reduction
    from SAT.
13. Semantic versioning is a social convention. Describe a realistic scenario
    where a patch release changes an ML result, and how a lock file helps.
14. Design a scheme recording everything needed to reproduce an experiment.
    State what you would store, where, and what you deliberately would not.
15. Explain why deterministic GPU kernels cost throughput, in terms of the
    associativity result in {{sec:7-implementation}}.

**Implementation**

16. Scaffold a project with the `src/` layout, install it editable, and write a
    test that imports it. Verify the test runs from outside the project
    directory.
17. Write a script producing a JSON environment fingerprint — Python version,
    platform, all package versions, git commit — and attach it to a run.
18. Write a function comparing two fingerprints and reporting exactly which
    packages differ.
19. Demonstrate empirically that `np.random.seed` is process-wide by having one
    function disturb another's stream, then show `default_rng` is immune.

**Reasoning**

20. Most published ML results are not reproducible at level 4 of
    {{tbl:reproducibility-levels}}. Is that acceptable? Distinguish research
    from production.
21. `uv` is faster and better than what preceded it, and is also young. How
    should a team decide whether to adopt a tool in that position?

## 12. Chapter Summary

A virtual environment is a per-project package directory selected by putting its
`bin` first on `PATH`. Isolation is not optional: two projects will eventually
need incompatible versions of the same library.

Dependency resolution is a constraint-satisfaction problem, equivalent to SAT
and therefore NP-complete, with the added difficulty that constraints are
discovered as choices are made. This is why resolvers are slow, why their error
messages are long, and why installing packages one at a time can reach a state
a single resolution would have avoided.

Declared constraints and a lock file are different artefacts. Constraints say
what you need; a lock file records the exact resolved graph, hashes included, so
an environment can be recreated. Applications commit both; published libraries
commit only the constraints.

`pyproject.toml` is the standard manifest, holding metadata, dependencies, build
configuration and tool settings in one file. Toolchains differ in speed and
scope but implement the same concepts; `uv` is currently the fastest and most
unified, and is young enough to be treated as an implementation of the concepts
rather than as the concepts themselves.

The `src/` layout forces the package to be installed before it can be imported,
which surfaces packaging errors immediately instead of when someone else clones
the repository.

Reproducibility requires six things: code, dependencies, interpreter, data,
configuration, and seeds — and seeding one generator is not seeding them all.
Even then, GPU results are frequently not bit-identical, because floating-point
addition is not associative and kernel accumulation order depends on thread
scheduling. Deterministic mode fixes this and costs throughput.
