# -*- coding: utf-8 -*-
# Extracted from: Chapter 14 — Functions, Classes, and Modules
# Source: src/.../ch014-functions-classes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
