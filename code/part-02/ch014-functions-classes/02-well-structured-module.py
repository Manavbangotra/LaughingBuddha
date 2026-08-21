# Extracted from: Chapter 14 — Functions, Classes, and Modules
# Source: src/.../ch014-functions-classes.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
