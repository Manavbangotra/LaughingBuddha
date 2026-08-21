---
id: py-engineering
number: 20
part: II
tier: focused
status: reviewed
requires: [py-environments, py-functions-classes, py-numpy]
provides: [structured-logging, observability-term, unit-test, test-fixture,
           property-based-testing, regression-test, profiling, memoisation,
           global-interpreter-lock, concurrency, parallelism, coroutine,
           event-loop, static-analysis]
citations: [pep8, pep484, pep703]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Debug systematically with `pdb` and structured reasoning rather than
   scattered print statements.
2. Configure logging properly, and explain why logging is not printing.
3. Write tests that will still be useful in a year, including fixtures and
   property-based tests.
4. Explain what to test in a machine-learning codebase, where outputs are
   stochastic.
5. Profile before optimising, and interpret the output.
6. Distinguish concurrency from parallelism, and choose threads, processes or
   async correctly.
7. Explain what the GIL constrains and what is changing about it.
8. Apply static analysis to catch errors before running anything.

## 2. Why This Matters

Everything to this point has been about making code work. This chapter is about
making it keep working — which is a different problem and, over the life of a
project, a larger one.

Machine-learning code is unusually hostile to the standard practices, for three
reasons worth naming.

**Failures are silent.** A web application that breaks returns a 500. A training
pipeline that breaks returns a model with slightly worse accuracy, which is
indistinguishable from a bad hyperparameter. The absence of loud failure is
precisely why the discipline matters more here, not less.

**Feedback loops are long.** A bug that surfaces forty minutes into a training
run costs forty minutes each time you probe it. Static analysis and a fast test
suite are worth far more when the alternative is that slow.

**Outputs are stochastic.** You cannot assert that a model achieves 94.2%
accuracy. {{sec:5-formal-explanation}} covers what you test instead, and it is
not obvious.

The practices in this chapter — logging, testing, profiling, and choosing the
right concurrency model — are the difference between a project that accelerates
as it grows and one that slows down.

## 3. Prerequisites

{{ch:py-environments}} for project structure and `pyproject.toml`, which is
where these tools are configured. {{ch:py-functions-classes}} for decorators,
which is what a pytest fixture is. {{ch:py-numpy}} for the vectorisation that
profiling will usually tell you to reach for.

## 4. Intuitive Explanation

### 4.1 Debugging is narrowing, not staring

The instinct on encountering a bug is to read the code until the error becomes
apparent. That works for simple bugs and fails for the interesting ones.

The reliable method is **bisection on the state space**. You know the input is
correct and the output is wrong; the fault is somewhere between. Check the
middle. That halves the search space, and repeating it finds the fault in
$\log_2 n$ steps rather than by inspiration.

`print` is a legitimate tool for this and is often the fastest one. But
`breakpoint()` drops you into an interactive debugger at that line, where you can
inspect any variable, evaluate expressions, and step forward — without editing,
re-running, and waiting.

> PRODUCTION TIP: `breakpoint()` is a builtin since Python 3.7 and honours the
> `PYTHONBREAKPOINT` environment variable — setting it to `0` disables every
> breakpoint in the codebase without editing anything. That is what makes it
> safe to leave one in a rarely-taken error branch.

### 4.2 Logging is not printing

`print` writes a string to stdout. Logging attaches a **severity**, a
**source**, and a **timestamp**, and lets you decide at runtime what to keep.

```python {tier=C name=logging-basic}
logger.debug("cache lookup for %s", key)      # verbose; off in production
logger.info("epoch %d complete, loss=%.4f", epoch, loss)
logger.warning("retrying after %s", exc)      # unexpected but handled
logger.error("failed to load checkpoint")     # something broke
logger.exception("unhandled")                 # error + full traceback
```

The gain is control. In development you enable `DEBUG` and see everything; in
production you enable `INFO` and the debug calls cost almost nothing. You cannot
do that with `print` without editing code.

{{term:structured-logging}} goes further and emits records as fields rather than
sentences. `logger.info("trained", extra={"epoch": 3, "loss": 0.21})` produces
something a log system can filter and aggregate — "show me every run where loss
exceeded 0.5" — which is impossible with formatted prose.

### 4.3 Testing stochastic code

The obvious objection to testing ML code is that outputs are not deterministic.
The resolution is to test **properties and contracts**, not values.

{#tbl:what-to-test caption="What to test in a machine-learning codebase. The left column is testable; asserting a specific accuracy is not."}

| Test | Example |
|---|---|
| Shapes | model output is `(batch, n_classes)` |
| Invariants | probabilities sum to 1; loss is non-negative |
| Boundaries | empty input, single row, all-identical values |
| Determinism | same seed gives the same result |
| Equivalence | vectorised version matches the loop version |
| Gradients | analytic gradient matches finite differences |
| Regression | a fixed input still produces the recorded output |
| Direction | loss decreases over a few training steps |

The last one is the closest you get to "does it learn", and it is a good smoke
test: train for twenty steps on ten examples and assert the loss went down. It
catches a large fraction of wiring errors in seconds.

### 4.4 Measure before optimising

Intuitions about where time goes are wrong often enough that guessing is not a
strategy. {{term:profiling}} measures it.

The usual outcome is that 90% of the time is in one place you did not suspect,
and the code you were about to optimise contributes 2%. Optimising that 2% can
at best make the program 2% faster — an instance of Amdahl's law, quantified in
{{sec:6-mathematical-foundation}}.

## 5. Formal Explanation

### 5.1 Logging configuration

```python {tier=C name=logging-config}
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
```

Three rules matter.

**Use `logging.getLogger(__name__)`**, not the root logger. Loggers form a
hierarchy following module names, so you can turn up verbosity for one
subpackage without drowning in everything else.

**Use lazy formatting** — `logger.debug("value %s", x)`, not
`logger.debug(f"value {x}")`. With the f-string, the formatting happens whether
or not the message is emitted. In a hot loop with `DEBUG` disabled, that is real
cost for output nobody sees.

**Libraries should not configure logging.** A library adds a
`NullHandler` and lets the application decide. A library that calls
`basicConfig` hijacks the application's configuration.

### 5.2 Tests with pytest

```python {tier=C name=pytest-basics}
import pytest

def test_standardise_gives_unit_variance():
    out = standardise([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(out.std(), 1.0)

@pytest.fixture
def sample_frame():
    """Shared setup. Teardown goes after the yield."""
    return pd.DataFrame({"a": [1, 2, 3]})

@pytest.mark.parametrize("n,expected", [(0, 1), (1, 1), (5, 120)])
def test_factorial(n, expected):
    assert factorial(n) == expected

def test_raises_on_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        factorial(-1)
```

A {{term:test-fixture}} is a decorated function providing setup, and pytest
injects it by parameter name. `parametrize` runs the same test over many inputs,
reporting each separately.

### 5.3 Property-based testing

Example-based tests check the cases you thought of.
{{term:property-based-testing}} states a property that should hold for *all*
inputs and lets the framework search for a counterexample.

```python {tier=C name=hypothesis-style}
# with the `hypothesis` library
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
def test_standardise_always_centres(values):
    assume(len(set(values)) > 1)
    assert abs(standardise(values).mean()) < 1e-9
```

The framework generates hundreds of inputs, including the adversarial ones you
would not write — empty lists, single elements, all-identical values, enormous
values, negative zero — and *shrinks* any failure to a minimal reproducing case.

{{sec:7-implementation}} implements a small version from scratch, since the
mechanism is worth seeing.

### 5.4 Profiling

Three tools, three questions:

```bash {tier=C name=profiling-tools}
python -m cProfile -s cumtime train.py     # which functions, cumulatively
python -m timeit -s "setup" "statement"    # microbenchmark one line
python -m tracemalloc                      # where memory is allocated
```

`cProfile` gives per-function call counts and time. The column that matters is
usually `cumtime` — total time including callees — because it finds the
expensive subtree rather than the expensive leaf.

A line profiler (`line_profiler`) narrows to statements within a function, which
is usually the next question.

> IMPORTANT: Profile a *representative* workload. Profiling on 100 rows when
> production has 10 million measures startup overhead, not the thing you care
> about. Profilers also add overhead of their own — `cProfile` can slow code
> several-fold — so use it to find *relative* hotspots, not absolute timings.

### 5.5 Concurrency and the GIL

Three models, and choosing correctly matters more than optimising within one.

{#tbl:concurrency-models caption="Choosing a concurrency model. The question is always whether the work is waiting or computing."}

| Model | Good for | Parallel? | Cost |
|---|---|---|---|
| `threading` | I/O-bound: files, network, database | no (GIL) | shared memory, races |
| `multiprocessing` | CPU-bound pure Python | yes | process overhead, pickling |
| `asyncio` | many concurrent I/O operations | no | requires async all the way down |
| NumPy / native libs | CPU-bound numeric work | **yes** | already done for you |

The {{term:global-interpreter-lock}} allows only one thread to execute Python
bytecode at a time. Threads therefore do not speed up pure-Python computation.
They do help with I/O, because a thread waiting on a socket releases the GIL.

The fourth row is the one people forget. NumPy releases the GIL during its
compiled operations, so a matrix multiply genuinely runs in parallel across
cores. For numerical work the answer is usually not "add threads" but "use
arrays" ({{ch:py-numpy}}).

> RESEARCH NOTE: PEP 703 {{cite:pep703}} makes the GIL optional. It is
> **Final** — accepted — with a first implementation in Python 3.13 via a
> separate `--disable-gil` build, at a reported 5-6% single-threaded overhead.
> Default builds are unchanged, so the advice above still holds for almost every
> reader today. {{maturity:EXPERIMENTAL}} The point worth carrying forward is
> that the GIL is a property of one implementation at one moment, not a
> permanent fact about Python.

{{term:concurrency}} is structuring a program so several things are in progress;
{{term:parallelism}} is executing them simultaneously. Async gives concurrency
without parallelism, and that is exactly right for I/O.

### 5.6 Static analysis

{{term:static-analysis}} finds errors without running anything:

- **`ruff`** — linting and formatting, extremely fast, replaces several older
  tools. Catches unused imports, mutable defaults ({{ch:py-fundamentals}}),
  shadowed builtins.
- **`mypy` or `pyright`** — type checking against the annotations of
  {{cite:pep484}}.

Both configure in `pyproject.toml` and run in CI. For code with a forty-minute
feedback loop, catching a type error in one second is a large multiple of its
cost.

## 6. Mathematical Foundation

### 6.1 Amdahl's law

If a fraction $p$ of a program's runtime is improved by a factor $s$, the
overall speedup is

$$
S = \frac{1}{(1 - p) + \dfrac{p}{s}}
$$ (eq:amdahl)

and the limit as $s \to \infty$ is

$$
S_{\max} = \frac{1}{1 - p}
$$ (eq:amdahl-limit)

The consequences are stark and are why profiling comes first.

{#tbl:amdahl caption="Maximum achievable speedup by fraction optimised. Making a 20% component infinitely fast yields 1.25×."}

| Fraction optimised | Speedup at $s = 10$ | Limit as $s \to \infty$ |
|---|---|---|
| 10% | 1.10× | 1.11× |
| 50% | 1.82× | 2× |
| 90% | 5.26× | 10× |
| 99% | 9.17× | 100× |

Optimising a component responsible for 10% of runtime cannot make the program
more than 11% faster no matter how good the optimisation. Finding the 90%
component is worth vastly more than any amount of cleverness applied to the
wrong one — which is the whole argument for measuring first.

The same equation governs parallelisation, with $p$ the parallelisable fraction:
a workload that is 95% parallel caps at 20× however many cores you add. This
recurs in {{ch:inf-parallelism}}.

### 6.2 What a test suite is actually worth

A test's value is not in the bug it catches today. It is insurance against
future change.

Let $c_w$ be the cost of writing a test, $c_r$ its cost per run, $n$ the number
of runs over the project's life, $p$ the probability it eventually catches a
regression, and $c_b$ the cost of that regression reaching production. The test
is worth writing when

$$
p \cdot c_b > c_w + n \cdot c_r
$$ (eq:test-value)

Two things follow. Since $c_r$ is seconds and $c_b$ for a silently-wrong model
in production is large, even a small $p$ justifies the test. And since $c_w$
appears once while $c_b$ scales with how long the code lives, **tests are worth
more on code that will be changed often** and comparatively little on a one-off
script.

This is why the right answer for a throwaway notebook and for a training
pipeline differ, and why "always test everything" is as wrong as "never test".

### 6.3 The cost model for concurrency choices

For $n$ tasks each spending $t_{io}$ waiting and $t_{cpu}$ computing:

$$
T_{\text{sequential}} = n(t_{io} + t_{cpu})
$$ (eq:sequential)

$$
T_{\text{threads}} \approx \max(t_{io}) + n\,t_{cpu}
\qquad\text{(GIL serialises the CPU part)}
$$ (eq:threads)

$$
T_{\text{processes}} \approx \frac{n(t_{io} + t_{cpu})}{k} + n\,c_{\text{ipc}}
$$ (eq:processes)

with $k$ workers and $c_{\text{ipc}}$ the per-task cost of pickling arguments
and results across the process boundary.

Reading these off gives the decision rule. When $t_{io} \gg t_{cpu}$, threads
win — the waiting overlaps and the serialised CPU portion is negligible. When
$t_{cpu} \gg t_{io}$, processes win, *provided* $c_{\text{ipc}}$ is small
relative to $t_{cpu}$. When the per-task work is small, $n\,c_{\text{ipc}}$
dominates and multiprocessing is **slower than sequential** — a result that
surprises people and that {{sec:7-implementation}} demonstrates.

## 7. Implementation

```python {tier=A name=testing-and-profiling}
"""Logging, testing, profiling, and Amdahl's law — measured.
"""
import cProfile
import io
import logging
import pstats
import random
import time

import numpy as np

# --- logging: levels, lazy formatting, and the hierarchy --------------------
print("=" * 66)
print("logging")
print("=" * 66)

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)-8s %(name)s | %(message)s",
                    force=True)
logger = logging.getLogger("demo.pipeline")

logger.debug("this is suppressed at INFO level")
logger.info("epoch %d complete, loss=%.4f", 3, 0.2143)
logger.warning("retrying after %s", "connection reset")

# Lazy formatting is not a style preference; it is a cost.
class Expensive:
    def __str__(self):
        time.sleep(0.001)          # pretend this is costly to render
        return "expensive-value"


t0 = time.perf_counter()
for _ in range(200):
    logger.debug("value is %s", Expensive())     # __str__ never called
lazy = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(200):
    logger.debug(f"value is {Expensive()}")      # __str__ ALWAYS called
eager = time.perf_counter() - t0

print(f"\n200 suppressed DEBUG calls:")
print(f"  lazy  'msg %s', arg : {lazy*1000:>7.1f} ms")
print(f"  eager f'msg {{arg}}'  : {eager*1000:>7.1f} ms   <- {eager/lazy:.0f}x")
print("The f-string formats the message even though nothing is emitted.")

# --- a minimal property-based tester ----------------------------------------
print("\n" + "=" * 66)
print("property-based testing, from scratch")
print("=" * 66)


def standardise(values):
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    return (arr - arr.mean()) / (std if std else 1.0)


def check_property(prop, generator, trials=400, seed=0):
    """Search for a counterexample, then shrink it to something minimal."""
    rng = random.Random(seed)
    for _ in range(trials):
        case = generator(rng)
        try:
            if prop(case):
                continue
        except Exception:
            pass
        # Found a failure. Shrink by repeatedly trying smaller inputs.
        shrunk = case
        improved = True
        while improved:
            improved = False
            for candidate in ([shrunk[:len(shrunk)//2], shrunk[1:],
                               shrunk[:-1]] if len(shrunk) > 1 else []):
                try:
                    ok = prop(candidate)
                except Exception:
                    ok = False
                if not ok and len(candidate) < len(shrunk):
                    shrunk, improved = candidate, True
                    break
        return shrunk
    return None


def gen_floats(rng):
    n = rng.randint(1, 8)
    return [rng.choice([rng.uniform(-100, 100), 0.0, 1e9, -1e9])
            for _ in range(n)]


# Property 1: standardising always centres the data. TRUE, and it passes.
cx = check_property(lambda v: abs(standardise(v).mean()) < 1e-6,
                    gen_floats)
print(f"property 'mean is zero'      : "
      f"{'PASSED' if cx is None else f'failed on {cx}'}")

# Property 2: the result always has unit variance. FALSE — constant input.
cx = check_property(lambda v: abs(standardise(v).std() - 1.0) < 1e-6,
                    gen_floats)
print(f"property 'std is one'        : "
      f"{'PASSED' if cx is None else f'FAILED, shrunk to {cx}'}")
print("  The framework found the constant-input case, which the guard in")
print("  standardise() handles by returning zeros — a real edge case that an")
print("  example-based test would only cover if you had thought of it.")

# --- gradient checking as a test (Chapter 11) --------------------------------
print("\n" + "=" * 66)
print("testing what is testable in ML code")
print("=" * 66)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss_and_grad(w, X, y):
    p = np.clip(sigmoid(X @ w), 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = X.T @ (p - y) / len(y)
    return loss, grad


rng = np.random.default_rng(0)
X = rng.normal(size=(40, 5))
y = (rng.random(40) < 0.5).astype(float)
w = rng.normal(size=5)

# Test 1: shapes and invariants.
loss, grad = loss_and_grad(w, X, y)
assert grad.shape == w.shape, "gradient must match parameter shape"
assert loss >= 0, "cross-entropy is non-negative"
print(f"shape and invariant tests    : PASSED (loss={loss:.4f})")

# Test 2: the analytic gradient matches finite differences.
h = 1e-6
numeric = np.array([
    (loss_and_grad(w + h * np.eye(5)[i], X, y)[0]
     - loss_and_grad(w - h * np.eye(5)[i], X, y)[0]) / (2 * h)
    for i in range(5)])
err = np.abs(grad - numeric).max()
assert err < 1e-6, f"gradient check failed: {err}"
print(f"gradient check               : PASSED (max error {err:.2e})")

# Test 3: determinism under a fixed seed.
def run(seed):
    r = np.random.default_rng(seed)
    return float(r.normal(size=100).mean())


assert run(42) == run(42)
print(f"determinism under fixed seed : PASSED")

# Test 4: loss decreases — the cheapest useful smoke test.
w_train = np.zeros(5)
losses = []
for _ in range(30):
    l, g = loss_and_grad(w_train, X, y)
    losses.append(l)
    w_train -= 0.5 * g
assert losses[-1] < losses[0], "loss must decrease"
print(f"loss decreases over 30 steps : PASSED "
      f"({losses[0]:.4f} -> {losses[-1]:.4f})")

# --- profiling, and eq. 20.1 -------------------------------------------------
print("\n" + "=" * 66)
print("profile before optimising")
print("=" * 66)


def slow_component(n):
    return sum(i * i for i in range(n))          # the dominant cost


def fast_component(n):
    return sum(range(n))                          # the minor cost


def workload():
    total = 0
    for _ in range(30):
        total += slow_component(20_000)
        total += fast_component(20_000)
    return total


buf = io.StringIO()
profiler = cProfile.Profile()
profiler.enable()
workload()
profiler.disable()
pstats.Stats(profiler, stream=buf).sort_stats("cumtime").print_stats(6)
lines = [l for l in buf.getvalue().splitlines()
         if "component" in l or "cumtime" in l]
print("\n".join(lines[:4]))

# Measure the true split, then apply Amdahl.
t0 = time.perf_counter()
for _ in range(30):
    slow_component(20_000)
t_slow = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(30):
    fast_component(20_000)
t_fast = time.perf_counter() - t0
p_slow = t_slow / (t_slow + t_fast)

print(f"\nmeasured split: slow_component is {p_slow:.1%} of runtime")
print(f"\n{'optimise':<18} {'by 10x':>9} {'by inf':>9}   (eq. 20.1)")
for label, p in (("the slow part", p_slow), ("the fast part", 1 - p_slow)):
    s10 = 1 / ((1 - p) + p / 10)
    smax = 1 / (1 - p)
    print(f"{label:<18} {s10:>8.2f}x {smax:>8.2f}x")
print("\nMaking the fast component infinitely fast barely helps. This is why")
print("guessing where the time goes is not a strategy.")
```

## 8. Practical Example

Choosing a concurrency model is a decision with a cost model behind it, and the
surprising cases are worth measuring rather than assuming.

```python {tier=A name=concurrency-choice}
"""Threads, processes, async and NumPy — measured against eqs. 20.4-20.6.

Includes the case people get wrong: multiprocessing being slower than
sequential when per-task work is small.
"""
import asyncio
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

N_TASKS = 8
IO_SECONDS = 0.12


# --- an I/O-bound task: mostly waiting ---------------------------------------
def io_task(_):
    time.sleep(IO_SECONDS)          # sleep releases the GIL
    return 1


# --- a CPU-bound task: pure Python arithmetic --------------------------------
def cpu_task(n):
    total = 0.0
    for i in range(1, n):
        total += math.sqrt(i) * math.sin(i)
    return total


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


print("=" * 70)
print(f"I/O-bound: {N_TASKS} tasks x {IO_SECONDS}s of waiting")
print("=" * 70)

seq = timeit(lambda: [io_task(i) for i in range(N_TASKS)])
thr = timeit(lambda: list(ThreadPoolExecutor(N_TASKS).map(io_task,
                                                          range(N_TASKS))))


async def async_task():
    await asyncio.sleep(IO_SECONDS)
    return 1


async def async_all():
    return await asyncio.gather(*(async_task() for _ in range(N_TASKS)))


asy = timeit(lambda: asyncio.run(async_all()))

print(f"{'sequential':<16} {seq:>7.3f}s   (eq. 20.4: n*t_io = "
      f"{N_TASKS*IO_SECONDS:.2f}s)")
print(f"{'threads':<16} {thr:>7.3f}s   {seq/thr:>5.1f}x faster")
print(f"{'asyncio':<16} {asy:>7.3f}s   {seq/asy:>5.1f}x faster")
print("\nBoth overlap the waiting. The GIL is irrelevant here because a")
print("sleeping thread is not executing bytecode (eq. 20.5).")

print("\n" + "=" * 70)
print("CPU-bound pure Python")
print("=" * 70)

WORK = 400_000
seq_cpu = timeit(lambda: [cpu_task(WORK) for _ in range(N_TASKS)])
thr_cpu = timeit(lambda: list(ThreadPoolExecutor(N_TASKS).map(
    cpu_task, [WORK] * N_TASKS)))

n_cores = mp.cpu_count()
if sys.platform != "win32":
    proc_cpu = timeit(lambda: list(ProcessPoolExecutor(
        min(n_cores, N_TASKS)).map(cpu_task, [WORK] * N_TASKS)))
else:
    proc_cpu = float("nan")

print(f"{'sequential':<16} {seq_cpu:>7.3f}s")
print(f"{'threads':<16} {thr_cpu:>7.3f}s   {seq_cpu/thr_cpu:>5.2f}x "
      f"<- no speedup: the GIL serialises bytecode")
print(f"{'processes':<16} {proc_cpu:>7.3f}s   {seq_cpu/proc_cpu:>5.2f}x "
      f"<- real parallelism across {n_cores} cores")

print("\n" + "=" * 70)
print("the case people get wrong: small tasks in processes")
print("=" * 70)

TINY = 300
tiny_seq = timeit(lambda: [cpu_task(TINY) for _ in range(2000)])
if sys.platform != "win32":
    tiny_proc = timeit(lambda: list(ProcessPoolExecutor(n_cores).map(
        cpu_task, [TINY] * 2000, chunksize=1)))
else:
    tiny_proc = float("nan")

print(f"2000 tiny tasks, sequential : {tiny_seq:>7.3f}s")
print(f"2000 tiny tasks, processes  : {tiny_proc:>7.3f}s   "
      f"{'SLOWER' if tiny_proc > tiny_seq else 'faster'} "
      f"({tiny_proc/tiny_seq:.1f}x)")
print("\nEach task must be pickled, sent to a worker, and the result sent")
print("back. When per-task work is smaller than that overhead, eq. 20.6's")
print("n*c_ipc term dominates and parallelism costs more than it saves.")
print("Raising chunksize amortises it — but the real fix is bigger tasks.")

print("\n" + "=" * 70)
print("the option people forget: don't use Python for the loop")
print("=" * 70)


def cpu_task_numpy(n):
    i = np.arange(1, n, dtype=np.float64)
    return float((np.sqrt(i) * np.sin(i)).sum())


t_np = timeit(lambda: [cpu_task_numpy(WORK) for _ in range(N_TASKS)])
same = math.isclose(cpu_task(WORK), cpu_task_numpy(WORK), rel_tol=1e-9)
print(f"{'numpy, 1 thread':<18} {t_np:>7.3f}s   {seq_cpu/t_np:>5.1f}x vs "
      f"sequential Python")
print(f"{'(processes were)':<18} {proc_cpu:>7.3f}s   {seq_cpu/proc_cpu:>5.1f}x")
print(f"identical result: {same}")
print("\nVectorising beat multiprocessing on one core, with no IPC, no")
print("pickling and no pool to manage. For numerical work the first question")
print("is not 'how do I parallelise this loop' but 'why is there a loop'.")

print("\n" + "=" * 70)
print("decision rule")
print("=" * 70)
print("  waiting on I/O            -> threads or asyncio")
print("  numeric computation       -> NumPy (already parallel, GIL released)")
print("  CPU-bound pure Python,")
print("    large tasks             -> processes")
print("    small tasks             -> batch them first, or stay sequential")
```

## 9. Common Mistakes

**Debugging by adding prints and re-running.** Use `breakpoint()`.

**Using `print` instead of logging.** No levels, no source, no runtime control.

**Eager f-strings in log calls.** The message is formatted even when suppressed.

**Configuring logging inside a library.** Hijacks the application's setup.

**Asserting exact metric values in tests.** Test properties, shapes,
invariants and direction instead.

**Testing only the happy path.** Empty input, single row and constant input are
where the bugs are.

**Optimising before profiling.** Amdahl's law: the wrong 10% caps you at 1.11×.

**Profiling an unrepresentative workload.** Measures startup, not the hot path.

**Using threads for CPU-bound Python.** The GIL serialises it; you get overhead
and no speedup.

**Using processes for small tasks.** IPC overhead exceeds the work.

**Reaching for concurrency before vectorisation.** For numerical work, NumPy is
usually both simpler and faster.

**Skipping static analysis.** One second in CI against forty minutes into a
training run.

**Bare `except`.** {{ch:py-fundamentals}}, and it hides exactly the failures
logging exists to surface.

## 10. Connection to Previous Chapters

{{ch:py-environments}} supplied the `pyproject.toml` where pytest, ruff and mypy
are all configured, and the reproducibility discipline that testing depends on.
{{ch:py-functions-classes}} supplied decorators — a pytest fixture is one — and
the purity that makes functions testable in a single line.
{{ch:py-numpy}} supplied vectorisation, which the concurrency comparison in
{{sec:8-practical-example}} shows is usually the better answer.
{{ch:math-derivatives}} supplied the gradient check used as a test.

Beyond Part II: {{ch:mle-reproducibility}} extends testing to experiments;
{{ch:ops-observability}} extends logging to distributed tracing;
{{ch:ev-framework}} builds a full evaluation harness on these foundations;
{{ch:inf-parallelism}} applies Amdahl's law to model parallelism; and
{{ch:sd-async}} builds on the async model for serving.

Style follows {{cite:pep8}}; annotations {{cite:pep484}}; the GIL discussion
tracks {{cite:pep703}}.

## 11. Exercises

**Beginner**

1. Add logging at four levels to a function and demonstrate filtering by level.
2. Write three pytest tests for a function, including one using
   `pytest.raises`.
3. Use `breakpoint()` to inspect a variable mid-function.
4. Profile a script with `cProfile` and identify the largest `cumtime` entry.
5. Explain the difference between concurrency and parallelism in one sentence
   each.

**Intermediate**

6. Write a fixture creating a temporary directory, and a test using it. Confirm
   cleanup happens.
7. Parametrize a test over eight input cases including two edge cases.
8. Measure the cost of eager versus lazy log formatting in a hot loop.
9. Use {{eq:amdahl}} to compute the speedup from making a 40% component 5×
   faster, and the limit.
10. Benchmark threads against processes for an I/O task and a CPU task, and
    explain both results.
11. Write a gradient check as a pytest test for a function of your own.

**Advanced**

12. Implement a property-based tester with shrinking, and use it to find a bug
    in a function you wrote.
13. Find the task size at which multiprocessing overtakes sequential execution
    on your machine, and explain it with {{eq:processes}}.
14. Profile a NumPy-heavy script and explain why `cProfile` attributes so little
    time to the array operations.
15. Write a pytest plugin or fixture that fails any test taking longer than a
    threshold.
16. Explain what removing the GIL {{cite:pep703}} would and would not change
    about the results in {{sec:8-practical-example}}.

**Implementation**

17. Add a full test suite to the pipeline from {{ch:py-functions-classes}} —
    shapes, invariants, edge cases, determinism, and a regression test.
18. Configure `ruff` and `mypy` in `pyproject.toml`, run them over your code,
    and fix everything they report.
19. Build a `@timed` decorator emitting structured log records, and aggregate a
    run's output into a per-function summary.
20. Take a slow script, profile it, optimise the top item, re-profile, and
    document the actual speedup against Amdahl's prediction.

**Reasoning**

21. Using {{eq:test-value}}, argue for a testing policy that treats exploratory
    notebooks and production pipelines differently.
22. ML failures are silent. Which practice in this chapter does the most to
    address that, and why?

## 12. Chapter Summary

Debugging is bisection on the state space, not inspection. `breakpoint()` beats
adding prints and re-running, and it can be globally disabled through the
environment.

Logging differs from printing by carrying severity, source and timestamp, and by
allowing runtime control of what is kept. Use `getLogger(__name__)`, use lazy
`%s` formatting so suppressed messages cost nothing, and never configure logging
inside a library.

Machine-learning code is testable despite stochastic outputs, because you test
properties rather than values: shapes, invariants, boundaries, determinism under
a fixed seed, equivalence between implementations, gradient checks, regression
fixtures, and the direction of the loss. Property-based testing generates the
adversarial inputs you would not have written and shrinks failures to minimal
cases.

Profile before optimising. Amdahl's law bounds the speedup from improving a
fraction $p$ of runtime at $1/(1-p)$, so optimising a 10% component cannot
exceed 1.11× however good the optimisation. Finding the right component is worth
more than any cleverness applied to the wrong one.

Concurrency is structure; parallelism is simultaneous execution. Threads help
with I/O because waiting releases the GIL, and do not help with pure-Python
computation because the GIL serialises bytecode. Processes give real parallelism
at the cost of pickling, which makes them *slower than sequential* when per-task
work is small. For numerical work the right answer is usually neither: NumPy
releases the GIL and vectorisation beats multiprocessing on one core.

The GIL is a property of one implementation at one moment. PEP 703 is accepted
and shipping experimentally, so the constraint is being removed — gradually
enough that the advice above still holds for the build almost every reader has.

Static analysis catches errors in seconds that would otherwise surface forty
minutes into a training run, which is the highest-leverage ratio in this chapter.
