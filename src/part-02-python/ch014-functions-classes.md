---
id: py-functions-classes
number: 14
part: II
tier: focused
status: reviewed
requires: [py-fundamentals]
provides: [pure-function, side-effect, decorator, context-manager, dataclass,
           type-hint, protocol-term, duck-typing, namespace, module-term]
citations: [pep8, pep484]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish pure functions from those with side effects, and explain why the
   distinction governs testability.
2. Use Python's argument forms correctly: positional, keyword, defaults,
   `*args`, `**kwargs`, and keyword-only parameters.
3. Write and read type hints, and explain what they do and do not enforce.
4. Choose appropriately between a function, a `dataclass`, and a full class.
5. Write decorators, including ones that take arguments, and explain what
   `functools.wraps` preserves.
6. Implement context managers both as classes and with `contextlib`.
7. Explain duck typing and structural protocols, and when each is preferable to
   inheritance.
8. Organise code into modules and packages with imports that do not break.

## 2. Why This Matters

The difference between a notebook that produced a result once and a system that
produces it reliably is mostly the material in this chapter.

Three specific reasons.

**Purity determines testability.** A function whose output depends only on its
inputs can be tested with one line and cached for free. A function that reads a
global, writes a file, and mutates its argument can only be tested by
constructing the world around it. In data pipelines this is not an aesthetic
preference: an impure transformation is the usual mechanism by which a bug in
one stage silently corrupts another ({{ch:mle-pipelines}}).

**Type hints catch the errors this field actually produces.** Python will not
stop you passing a `(batch, seq, dim)` tensor to something expecting
`(batch, dim)`. A static checker will, in a second, rather than forty minutes
into a training run {{cite:pep484}}.

**Decorators and context managers are how cross-cutting concerns get handled.**
Timing, retrying, caching, logging, and resource cleanup all appear repeatedly
from {{part:22}} onward, and all are implemented with the two constructs in
{{sec:5-formal-explanation}}. Every framework you will use is built from them,
so reading framework source requires knowing them.

## 3. Prerequisites

{{ch:py-fundamentals}} for the object model, mutability and aliasing — a
function's behaviour with respect to its arguments follows directly from those.

## 4. Intuitive Explanation

### 4.1 Purity is a property worth designing for

A {{term:pure-function}} depends only on its arguments and changes nothing
outside itself:

```python {tier=C name=pure-vs-impure}
def normalise(values, mean, std):        # pure
    return [(v - mean) / std for v in values]

TOTAL = 0
def accumulate(values):                  # impure: reads and writes a global,
    global TOTAL                         # and its result depends on history
    TOTAL += sum(values)
    return TOTAL
```

The first can be tested by calling it. The second can only be tested by
controlling `TOTAL` first, and it returns different answers for identical
inputs.

{{term:side-effect}}s are not forbidden — a program with no side effects does
nothing observable. The discipline is to *concentrate* them: keep the
transformations pure and push the file writing, the logging and the mutation to
the edges. That is what makes the middle of a pipeline testable.

### 4.2 Classes: use fewer than you think

Programmers arriving from Java reach for classes reflexively. Python's culture
does not, and for data work the culture is right.

Use a **function** when you transform inputs to outputs. That is most of the
time.

Use a **{{term:dataclass}}** when you need to carry a fixed set of related
fields around. It generates the constructor, the repr and equality for you.

Use a **full class** when you have state that genuinely evolves and behaviour
that depends on it — a model being fitted, a connection pool, a stateful
tokeniser.

A class whose only method is `run()` and whose constructor just stores its
arguments is a function that has been made harder to test.

### 4.3 Decorators wrap behaviour around functions

A {{term:decorator}} takes a function and returns a replacement:

```python {tier=C name=decorator-shape}
@timed
def train(epochs):
    ...

# is exactly
train = timed(train)
```

That is the whole mechanism. `@timed` is syntax for reassignment.

Decorators are how you add timing, caching, retries, logging or validation to
many functions without editing any of them. You will meet them constantly:
`@property`, `@dataclass`, `@functools.cache`, `@pytest.fixture`,
`@app.get("/predict")` in FastAPI, `@torch.no_grad()`.

### 4.4 Context managers guarantee cleanup

```python {tier=C name=context-manager-shape}
with open("data.csv") as f:
    process(f)
# f is closed here — even if process() raised
```

The guarantee is the point. A {{term:context-manager}}'s `__exit__` runs on
every path out of the block, including exceptions and early returns. Anything
that must be released — files, locks, database connections, GPU memory, a
temporarily changed setting — belongs in one.

## 5. Formal Explanation

### 5.1 Function arguments

Python's parameter system is richer than most, and the pieces have specific
roles.

```python {tier=C name=argument-forms}
def f(pos, /, normal, *args, kwonly, **kwargs):
    ...
```

- `pos` before `/` is **positional-only** — callers cannot use `pos=`.
- `normal` may be passed either way.
- `*args` collects extra positional arguments into a tuple.
- `kwonly` after `*` is **keyword-only** — it must be named.
- `**kwargs` collects extra keyword arguments into a dict.

The practically important one is **keyword-only**. A function with several
boolean or numeric options is unreadable when called positionally:

```python {tier=C name=keyword-only-motivation}
train(model, data, True, False, 0.1)         # what are these?
train(model, data, shuffle=True, verbose=False, lr=0.1)   # clear
```

Putting a bare `*` in the signature makes the second form mandatory. It is worth
doing for any parameter whose meaning is not obvious from its value.

> IMPORTANT: Defaults are evaluated **once**, at definition time
> ({{ch:py-fundamentals}}). Never use a mutable default; use `None` and create
> the object inside.

### 5.2 Type hints

{{term:type-hint}}s annotate expected types. Python **does not enforce them** —
they are metadata, stored in `__annotations__`, ignored by the interpreter
{{cite:pep484}}.

```python {tier=C name=type-hints}
def batch(items: list[str], size: int = 32) -> list[list[str]]:
    ...

def load(path: str | None = None) -> dict[str, float]:
    ...
```

They earn their place three ways: a static checker (`mypy`, `pyright`) finds
mismatches before you run anything; editors give real completion; and they
document the contract in the place most likely to stay current.

> WARNING: An annotation is not validation. `def f(x: int)` called with a string
> runs happily until something fails downstream, usually somewhere unrelated.
> When you need runtime enforcement — at an API boundary, or parsing config —
> use a validating library such as Pydantic ({{ch:sd-apis-auth}}), which reads
> the annotations and actually checks them.

### 5.3 Classes, and the three tiers

```python {tier=C name=three-tiers}
# 1. a function — a transformation
def tokenize(text: str) -> list[str]:
    return text.lower().split()

# 2. a dataclass — grouped data
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    tags: list[str] = field(default_factory=list)   # NOT tags: list = []

# 3. a full class — evolving state plus behaviour
class RunningMean:
    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0

    def update(self, x: float) -> None:
        self._n += 1
        self._mean += (x - self._mean) / self._n     # Welford; see Chapter 9

    @property
    def value(self) -> float:
        return self._mean
```

Note `field(default_factory=list)` in the dataclass. It is the mutable-default
trap of {{ch:py-fundamentals}} again, and `@dataclass` raises an error if you
write `tags: list = []` — one of the few places Python protects you from it.

The dunder methods worth knowing for this book: `__init__` (construction),
`__repr__` (debugging output — always define one), `__len__`, `__getitem__`
(indexing, which is what makes a PyTorch `Dataset` work), `__call__` (making an
instance callable, which is how every PyTorch `Module` is used), and
`__enter__`/`__exit__` (context managers).

### 5.4 Duck typing and protocols

{{term:duck-typing}} means an object is acceptable if it supports the operations
used, regardless of its declared type. Anything with `__len__` and
`__getitem__` works as a sequence; anything with `read()` works where a file is
expected.

The modern way to make that checkable is a {{term:protocol-term}}:

```python {tier=C name=protocol}
from typing import Protocol

class Estimator(Protocol):
    def fit(self, X, y) -> "Estimator": ...
    def predict(self, X): ...

def evaluate(model: Estimator, X, y) -> float:
    return score(model.predict(X), y)
```

Any object with `fit` and `predict` satisfies `Estimator` — no inheritance, no
registration. A static checker verifies the match structurally. This is how
scikit-learn's ecosystem works, and why third-party estimators drop into it
without importing anything from scikit-learn.

### 5.5 Composition over inheritance

Inheritance is the tool most over-reached for by programmers arriving from
Java or C++, and the reason is worth understanding rather than treating as
style advice.

Inheritance couples a subclass to its parent's *implementation*, not just its
interface. A subclass can be broken by a change to a parent method it never
mentions, because it may depend on the order in which the parent calls its own
methods — the fragile base class problem. That coupling is invisible in the
subclass's source, which is what makes it expensive.

Composition holds a reference instead of extending:

```python {tier=C name=composition}
# inheritance: Trainer IS a Logger, and inherits everything it does
class Trainer(Logger):
    def step(self):
        self.log("stepping")        # where is log defined? which override wins?

# composition: Trainer HAS a logger, and uses exactly what it needs
class Trainer:
    def __init__(self, logger):
        self._logger = logger       # any object with .log() works

    def step(self):
        self._logger.log("stepping")
```

The composed version can be tested with a fake logger, can swap
implementations at runtime, and states its dependency in the constructor
signature where a reader will see it. Combined with a `Protocol`
({{sec:5-formal-explanation}} above), it gets static checking too, without any
inheritance relationship existing.

Inheritance remains the right tool in two situations. When a framework requires
it — subclassing `torch.nn.Module` or `Exception` is how you participate in
those systems ({{ch:dl-forward}}). And when there is a genuine
*is-a* relationship with shared implementation that varies only at named
extension points, which is the template-method pattern.

The practical test: if you are inheriting to reuse code rather than to *be*
the parent type, compose instead.

### 5.6 Modules and packages

A {{term:module-term}} is a `.py` file; a package is a directory of them. Each
has its own {{term:namespace}}, and name resolution follows the **LEGB** order:
Local, Enclosing, Global, Built-in.

Import conventions {{cite:pep8}}:

```python {tier=C name=import-style}
import numpy as np                    # absolute, aliased by convention
from mypkg.data import load_csv       # absolute, specific
from .utils import clean              # relative, within a package
```

Two rules avoid most import pain. **Prefer absolute imports** — relative ones
break when a module is run directly. **Never use `from x import *`** — it
pollutes the namespace and makes it impossible to tell where a name came from.

> PRODUCTION TIP: Circular imports — module A importing B while B imports A —
> usually indicate that a shared concept belongs in a third module. Deferring
> the import into a function body works and is a patch, not a fix. Extract the
> shared piece instead.

## 6. Mathematical Foundation

### 6.1 What a decorator actually does

Decorator syntax is pure sugar:

$$
\texttt{@d}\;\;\texttt{def f(): ...}
\qquad\equiv\qquad
\texttt{f = d(f)}
$$ (eq:decorator-equivalence)

So a decorator is any callable taking a function and returning a replacement.
The standard shape:

```python {tier=C name=decorator-anatomy}
import functools

def timed(func):
    @functools.wraps(func)                  # preserve identity — see below
    def wrapper(*args, **kwargs):           # accept ANY signature
        start = time.perf_counter()
        result = func(*args, **kwargs)      # call the original
        print(f"{func.__name__}: {time.perf_counter()-start:.4f}s")
        return result
    return wrapper                          # replaces the original
```

`*args, **kwargs` is what lets one decorator wrap functions of any signature.

`functools.wraps` copies `__name__`, `__doc__`, `__module__` and
`__wrapped__` from the original onto the wrapper. Without it the decorated
function reports itself as `wrapper`, which breaks documentation tools,
debuggers, and anything that introspects — including pytest's test collection.
It is not optional.

A decorator that **takes arguments** needs one more layer, because
`@retry(times=3)` is a *call* whose result is then used as the decorator:

$$
\texttt{@d(a)}\;\;\texttt{def f(): ...}
\qquad\equiv\qquad
\texttt{f = d(a)(f)}
$$ (eq:parameterised-decorator)

Three nested functions: the outer takes the parameters, the middle takes the
function, the inner does the work.

### 6.2 The context-manager protocol

```python {tier=C name=context-protocol}
class Resource:
    def __enter__(self):
        self.handle = acquire()
        return self.handle            # bound to the `as` name

    def __exit__(self, exc_type, exc_value, traceback):
        release(self.handle)
        return False                  # False => propagate any exception
```

`__exit__` receives exception information if the block raised, and `None` three
times if it did not. Its **return value decides whether the exception
propagates**: falsy propagates, truthy suppresses. Returning `True`
unconditionally silently swallows every exception in the block, which is almost
never what you want.

`contextlib.contextmanager` expresses the same thing as a generator:

```python {tier=C name=contextlib-form}
from contextlib import contextmanager

@contextmanager
def resource():
    handle = acquire()
    try:
        yield handle          # everything before is __enter__
    finally:
        release(handle)       # everything after is __exit__
```

The `try/finally` is essential. Without it, an exception in the block skips the
cleanup entirely — which defeats the entire purpose of the construct.

### 6.3 Why purity enables caching

{{term:memoisation}} is only correct for pure functions, and the reason is
precise. Caching replaces a call with a stored result:

$$
\texttt{cache[args]} \;\longrightarrow\; f(\texttt{args})
$$

This is sound exactly when $f$ is **referentially transparent** — when replacing
any call with its result leaves program behaviour unchanged. An impure function
fails this by definition: its side effects are part of what the call does, and
they do not happen on a cache hit.

Concretely, caching a function that writes to a database means the write happens
once and then silently stops happening. `functools.cache` cannot detect this,
and there is no error.

## 7. Implementation

```python {tier=A name=decorators-and-managers}
"""Decorators, context managers, dataclasses and protocols — working code for
each, with the failure modes demonstrated rather than described.
"""
import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

# --- eq. 14.1: a decorator is just reassignment -----------------------------
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_seconds = time.perf_counter() - start
        return result
    wrapper.last_seconds = None
    return wrapper


@timed
def slow_sum(n: int) -> int:
    """Add up the first n integers, slowly and on purpose."""
    return sum(range(n))


slow_sum(2_000_000)
print(f"slow_sum took {slow_sum.last_seconds*1000:.2f} ms")

# --- why functools.wraps matters --------------------------------------------
def naive_decorator(func):
    def wrapper(*a, **k):
        return func(*a, **k)
    return wrapper              # no @wraps


@naive_decorator
def documented(x):
    """This docstring should survive decoration."""
    return x


print(f"\nwithout @wraps: name={documented.__name__!r}, "
      f"doc={documented.__doc__!r}")
print(f"with    @wraps: name={slow_sum.__name__!r}, "
      f"doc={slow_sum.__doc__!r}")
print("Losing __name__ breaks debuggers, docs tools and pytest collection.")

# --- eq. 14.2: a decorator that takes arguments needs three layers ----------
def retry(times: int, exceptions=(Exception,)):
    """Retry a flaky call, with the attempt count reported."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == times:
                        raise
                    print(f"    attempt {attempt} failed ({exc}); retrying")
        return wrapper
    return decorator


calls = {"n": 0}


@retry(times=4, exceptions=(ValueError,))
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ValueError("transient")
    return f"succeeded on attempt {calls['n']}"


print(f"\n{flaky()}")

# --- section 6.3: caching is only correct for PURE functions ----------------
@functools.cache
def pure_square(n: int) -> int:
    return n * n


side_effects = []


@functools.cache
def impure_log(n: int) -> int:
    side_effects.append(n)          # a side effect, silently cached away
    return n * n


for _ in range(5):
    pure_square(7)
    impure_log(7)

print(f"\npure_square called 5 times, cache info: {pure_square.cache_info()}")
print(f"impure_log called 5 times, side effects recorded: {side_effects}")
print("The side effect happened ONCE. Caching silently changed behaviour —")
print("which is why memoisation requires referential transparency.")

# --- context managers, both forms -------------------------------------------
class Timer:
    """Class form: __enter__ / __exit__."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.elapsed = time.perf_counter() - self.start
        # Returning a falsy value lets any exception propagate.
        return False


with Timer() as t:
    sum(range(500_000))
print(f"\nclass-based timer: {t.elapsed*1000:.2f} ms")


@contextmanager
def temporary_setting(store: dict, key: str, value):
    """Generator form: everything before yield is enter, after is exit."""
    missing = object()
    previous = store.get(key, missing)
    store[key] = value
    try:
        yield store
    finally:
        # The finally is essential: without it an exception skips the restore.
        if previous is missing:
            del store[key]
        else:
            store[key] = previous


config = {"mode": "train"}
with temporary_setting(config, "mode", "eval"):
    print(f"inside the block : {config}")
print(f"after the block  : {config}")

# ...and it restores correctly even when the block raises.
try:
    with temporary_setting(config, "mode", "debug"):
        raise RuntimeError("boom")
except RuntimeError:
    pass
print(f"after an exception: {config}   <- still restored")

# --- dataclasses -------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 10
    # A mutable default MUST use default_factory; a bare [] is a TypeError.
    tags: list[str] = field(default_factory=list)


cfg = TrainConfig(lr=3e-4, tags=["baseline"])
print(f"\n{cfg}")
print(f"as a dict: {asdict(cfg)}")
print(f"equality is by value: {TrainConfig() == TrainConfig()}")

try:
    @dataclass
    class Broken:
        items: list = []          # the trap of Chapter 13
except ValueError as exc:
    print(f"\ndataclass rejects a mutable default: {str(exc)[:64]}...")

# --- duck typing and protocols ----------------------------------------------
from typing import Protocol


class Scorer(Protocol):
    def score(self, y_true, y_pred) -> float: ...


class Accuracy:
    def score(self, y_true, y_pred) -> float:
        return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


class MeanAbsoluteError:
    def score(self, y_true, y_pred) -> float:
        return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def report(scorers: list[Scorer], y_true, y_pred) -> None:
    for s in scorers:
        print(f"  {type(s).__name__:<20} {s.score(y_true, y_pred):.4f}")


print("\nprotocol-based dispatch — neither class inherits from anything:")
report([Accuracy(), MeanAbsoluteError()], [1, 0, 1, 1], [1, 0, 0, 1])

# --- __call__ makes an instance behave like a function ----------------------
class Scaler:
    """This is exactly the shape of a PyTorch Module (Part VI)."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def __call__(self, xs):
        return [x * self.factor for x in xs]

    def __repr__(self) -> str:
        return f"Scaler(factor={self.factor})"


double = Scaler(2.0)
print(f"\n{double} applied to [1, 2, 3]: {double([1, 2, 3])}")
print(f"callable(double) = {callable(double)}")
```

## 8. Practical Example

A small, well-structured module is the deliverable this chapter is really about.
The following is the shape almost every data-processing component in this book
takes.

```python {tier=A name=well-structured-module}
"""A feature-pipeline component built the way the rest of the book will.

Pure transformations in the middle, side effects at the edges, configuration in
a dataclass, cross-cutting concerns in decorators, and a protocol so that
alternative implementations drop in without inheritance.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Protocol

# ---------------------------------------------------------------- decorators

def instrumented(func):
    """Record call count and cumulative time — a cross-cutting concern kept
    out of the transformation logic itself."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        wrapper.calls += 1
        wrapper.seconds += time.perf_counter() - start
        return out
    wrapper.calls = 0
    wrapper.seconds = 0.0
    return wrapper


# ------------------------------------------------------------------ protocol

class Transform(Protocol):
    """Anything with this shape can go in a pipeline. No base class needed."""

    def fit(self, rows: list[dict]) -> "Transform": ...
    def apply(self, row: dict) -> dict: ...


# ----------------------------------------------------------------- transforms

@dataclass
class Standardise:
    """Centre and scale one numeric column. State is learned in fit()."""

    column: str
    mean_: float | None = None
    std_: float | None = None

    def fit(self, rows: list[dict]) -> "Standardise":
        values = [r[self.column] for r in rows if r.get(self.column) is not None]
        n = len(values)
        self.mean_ = sum(values) / n
        var = sum((v - self.mean_) ** 2 for v in values) / max(n - 1, 1)
        self.std_ = var ** 0.5 or 1.0
        return self

    @instrumented
    def apply(self, row: dict) -> dict:
        if self.mean_ is None:
            raise RuntimeError("Standardise.apply called before fit")
        out = dict(row)                       # never mutate the caller's row
        v = row.get(self.column)
        out[self.column] = None if v is None else (v - self.mean_) / self.std_
        return out


@dataclass
class OneHot:
    """Expand a categorical column into indicator columns."""

    column: str
    categories_: list[str] = field(default_factory=list)

    def fit(self, rows: list[dict]) -> "OneHot":
        self.categories_ = sorted({r[self.column] for r in rows
                                   if r.get(self.column) is not None})
        return self

    @instrumented
    def apply(self, row: dict) -> dict:
        out = {k: v for k, v in row.items() if k != self.column}
        for c in self.categories_:
            out[f"{self.column}={c}"] = int(row.get(self.column) == c)
        return out


# ------------------------------------------------------------------ pipeline

@dataclass
class Pipeline:
    steps: list[Transform]

    def fit(self, rows: list[dict]) -> "Pipeline":
        # Each step is fitted on the output of the previous one.
        current = rows
        for step in self.steps:
            step.fit(current)
            current = [step.apply(r) for r in current]
        return self

    def apply(self, rows: list[dict]) -> list[dict]:
        for step in self.steps:
            rows = [step.apply(r) for r in rows]
        return rows


# ---------------------------------------------------------------------- usage

TRAIN = [
    {"age": 25, "city": "london", "score": 1},
    {"age": 40, "city": "leeds", "score": 0},
    {"age": 35, "city": "london", "score": 1},
    {"age": 50, "city": "bristol", "score": 0},
    {"age": None, "city": "leeds", "score": 1},
]
TEST = [{"age": 30, "city": "london", "score": 1},
        {"age": 45, "city": "cardiff", "score": 0}]   # unseen category

pipeline = Pipeline([Standardise("age"), OneHot("city")]).fit(TRAIN)

print("fitted state:")
print(f"  Standardise: mean={pipeline.steps[0].mean_:.2f}, "
      f"std={pipeline.steps[0].std_:.2f}")
print(f"  OneHot     : {pipeline.steps[1].categories_}")

print("\ntransformed test rows:")
for row in pipeline.apply(TEST):
    age = row["age"]
    print(f"  age={age if age is None else round(age, 3):<7} "
          f"{ {k: v for k, v in row.items() if k.startswith('city=')} }")

print("\nNote the unseen category 'cardiff' becomes all zeros rather than")
print("raising — a deliberate choice, and one that must be tested (Ch. 20).")

# The input was never mutated: transformations are pure with respect to caller
# state, which is what makes them safe to reuse and reorder.
print(f"\noriginal TEST unchanged: {TEST[0]}")
assert TEST[0] == {"age": 30, "city": "london", "score": 1}

print(f"\ninstrumentation, collected by the decorator without the transform "
      f"knowing:")
for step in pipeline.steps:
    fn = step.apply
    print(f"  {type(step).__name__:<14} {fn.calls:>3} calls, "
          f"{fn.seconds*1e6:>7.1f} us total")
```

## 9. Common Mistakes

**Mutable default arguments.** Covered in {{ch:py-fundamentals}} and worth
repeating because it recurs in dataclass fields, where `default_factory` is
required.

**Omitting `functools.wraps`.** The decorated function loses its name and
docstring, which breaks introspection, documentation, and pytest.

**Returning `True` from `__exit__`.** Silently swallows every exception in the
block.

**Omitting `try/finally` in a `@contextmanager`.** Cleanup is skipped exactly
when it matters most — on the exception path.

**Caching an impure function.** The side effects stop happening after the first
call, with no error.

**Writing a class where a function would do.** If it has one method and a
constructor that only stores arguments, it is a function with extra steps.

**Believing type hints are enforced.** They are not. Use a checker in CI, and a
validating library at runtime boundaries.

**Mutating an argument inside a transformation.** The caller's data changes
underneath them. Copy first, as `Standardise.apply` does above.

**`from module import *`.** Destroys the ability to tell where a name came from.

**Deep inheritance hierarchies.** Composition and protocols solve almost every
problem inheritance is reached for, with less coupling.

## 10. Connection to Previous Chapters

{{ch:py-fundamentals}} established the object model that explains argument
passing, the mutable-default trap that reappears here in dataclass fields, and
the exception handling that `__exit__` interacts with.

Forward within Part II: {{ch:py-environments}} packages the modules of
{{sec:5-formal-explanation}} into installable projects. {{ch:py-engineering}}
tests them, and uses `@pytest.fixture` — a decorator — plus type checking as
static analysis.

Beyond Part II: the `fit`/`apply` protocol in {{sec:8-practical-example}} is
scikit-learn's interface, formalised in {{ch:mle-pipelines}}. `__call__` is how
every PyTorch module is invoked ({{ch:dl-forward}}), and `__len__`/`__getitem__`
is the `Dataset` interface. Decorators appear as `@torch.no_grad()`, as FastAPI
route handlers ({{ch:sd-apis-auth}}), and as retry logic against flaky APIs
({{ch:sd-fault-tolerance}}). Context managers appear wherever a resource must be
released.

Style follows {{cite:pep8}}; annotations follow {{cite:pep484}}.

## 11. Exercises

**Beginner**

1. Write a pure function converting Celsius to Fahrenheit, and an impure one
   that logs each conversion to a global list. State which is easier to test and
   why.
2. Write a function with one positional-only, one normal, and one keyword-only
   parameter, then call it correctly.
3. Add type hints to a function taking a list of strings and an integer and
   returning a dict of string to float.
4. Convert a class with two fields and a repr into a `dataclass`.
5. Use `contextlib.contextmanager` to write a context manager that prints on
   entry and exit.

**Intermediate**

6. Write a `@log_calls` decorator recording arguments and return values, using
   `functools.wraps`. Demonstrate what breaks without it.
7. Write a decorator taking a parameter — `@repeat(3)` — and explain the three
   nested functions.
8. Implement a context manager that changes the working directory and restores
   it, correctly on the exception path.
9. Define a `Protocol` with two methods and two unrelated classes satisfying it.
   Verify with a type checker if you have one.
10. Explain why `@dataclass` rejects `items: list = []` but plain classes do
    not.
11. Add a `Transform` to the pipeline in {{sec:8-practical-example}} that clips
    a numeric column to a fitted range.

**Advanced**

12. Write a decorator that caches results with a time-to-live, and explain what
    makes it safe or unsafe compared with `functools.cache`.
13. Implement a context manager that is also usable as a decorator — as
    `contextlib.ContextDecorator` does — and explain the mechanism.
14. Explain what `functools.wraps` copies and why `__wrapped__` in particular
    matters for introspection.
15. Give a case where inheritance is genuinely the right tool over composition,
    and justify it.
16. Write a decorator that validates arguments against their type hints at
    runtime, using `typing.get_type_hints`. Discuss the performance cost.

**Implementation**

17. Build a `Pipeline` supporting `fit_apply`, saving fitted state to JSON, and
    loading it back. Verify a round trip.
18. Write a `@retry` decorator with exponential backoff and jitter
    ({{ch:py-io-apis-sql}}), and test it against a function that fails a
    controlled number of times.
19. Take a function of your own with side effects and refactor it into a pure
    core plus a thin impure shell. Write tests for the core that need no setup.
20. Profile the overhead a decorator adds to a trivial function called a million
    times, and say when that cost would matter.

**Reasoning**

21. Python does not enforce type hints. Argue for and against making them
    enforced at runtime by default.
22. The pipeline above returns all-zero indicators for an unseen category rather
    than raising. Argue both positions, and say what you would do differently in
    a training context versus a serving one.

## 12. Chapter Summary

Pure functions depend only on their inputs and change nothing outside
themselves, which makes them testable in one line and safely cacheable. Side
effects are unavoidable but should be concentrated at the edges, leaving the
transformations in the middle pure.

Python's parameter system supports positional-only, keyword-only, `*args` and
`**kwargs`. Keyword-only parameters are worth using for any option whose meaning
is not evident from its value. Defaults are evaluated once at definition, so
mutable defaults are a trap — including in dataclass fields, where
`default_factory` is required.

Type hints are metadata, not enforcement. They pay for themselves through static
checking, editor support, and documentation that stays current; runtime
validation needs a library that reads them deliberately.

Choose a function for a transformation, a dataclass for grouped fields, and a
full class only for genuinely evolving state. A class with one method is a
function made harder to test.

A decorator is a callable that takes a function and returns a replacement, and
`@d` is exactly `f = d(f)`. Always use `functools.wraps`. Parameterised
decorators need three nested layers because `@d(a)` is `d(a)(f)`.

Context managers guarantee cleanup on every exit path. In the generator form the
`try/finally` is essential; `__exit__` returning a truthy value silently
suppresses exceptions.

Duck typing accepts any object supporting the operations used. `Protocol` makes
that structurally checkable without inheritance, which is why scikit-learn's
`fit`/`predict` interface works across unrelated libraries.
