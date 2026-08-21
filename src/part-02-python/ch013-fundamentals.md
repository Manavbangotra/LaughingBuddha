---
id: py-fundamentals
number: 13
part: II
tier: focused
status: reviewed
requires: [math-notation]
provides: [mutability, aliasing, shallow-copy, python-object-model,
           comprehension, iterator, generator, lazy-evaluation, truthiness]
citations: [pep8]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain Python's object model: that names bind to objects and assignment
   never copies.
2. Predict whether an operation mutates an object or creates a new one, and
   trace aliasing through function calls.
3. Explain and avoid the mutable-default-argument trap.
4. Distinguish shallow from deep copies and know when each is needed.
5. Write comprehensions fluently and know when a loop is clearer.
6. Explain lazy evaluation and use generators to process data larger than
   memory.
7. Apply Python's truthiness rules, and recognise where they cause bugs.
8. Handle exceptions with the right granularity, and say why bare `except` is
   almost always wrong.

## 2. Why This Matters

This chapter assumes you can already program. Its job is not to teach you what a
loop is; it is to cover the handful of things about Python that reliably
surprise people arriving from other languages, and that cause bugs which do not
announce themselves.

Almost all of those come from one design decision: **Python names are references
to objects, and assignment binds names rather than copying values.** In a
language with value semantics, passing a list to a function and passing an
integer behave the same way. In Python they do not, and the difference is
invisible until a function you called quietly modified data you thought you
owned.

The second theme is laziness. Data in this book will routinely be larger than
memory, and the difference between a program that works and one that dies is
often the difference between building a list and yielding from a generator.

The third is that Python is slow, and knowing *why* tells you what to do about
it. Every operation goes through dynamic type dispatch, every integer is a heap
object, and the interpreter overhead per operation is on the order of tens of
nanoseconds. That is fine for orchestration and catastrophic in an inner loop —
which is the entire motivation for {{ch:py-numpy}}.

## 3. Prerequisites

Programming experience in some language. {{ch:math-notation}} for the notation
conventions used in comments and docstrings.

You do not need prior Python.

## 4. Intuitive Explanation

### 4.1 Names are labels, not boxes

The mental model most people bring from C, Java or Go is that a variable is a
box holding a value, and assignment puts a value in the box. Python does not
work that way, and holding the wrong model produces confident wrong predictions.

In Python, **objects live somewhere, and names are labels stuck to them.**
Assignment moves a label; it never copies the thing labelled.

```text
a = [1, 2, 3]        a ──▶ [1, 2, 3]
b = a                a ──▶ [1, 2, 3] ◀── b     (one object, two labels)
b.append(4)          a ──▶ [1, 2, 3, 4] ◀── b  (both see it)
b = [9]              a ──▶ [1, 2, 3, 4]        (b's label moved)
                     b ──▶ [9]
```

Two operations look similar and are entirely different. `b.append(4)` **mutates
the object** both names point at. `b = [9]` **rebinds the name** and leaves the
object alone. The first is visible through `a`; the second is not.

### 4.2 Mutability decides whether this matters

If every object were immutable, aliasing would be invisible — you could never
observe the difference. {{term:mutability}} is what makes it matter.

{#tbl:mutability caption="Which built-in types can be changed in place. The immutable column is safe to alias freely; the mutable column is where the surprises live."}

| Immutable | Mutable |
|---|---|
| `int`, `float`, `bool`, `complex` | `list` |
| `str`, `bytes` | `dict` |
| `tuple`, `frozenset` | `set` |
| `None` | most user-defined classes |

This explains the behaviour that puzzles newcomers most:

```python {tier=C name=mutability-contrast}
def add_one(n):        # n is an int — immutable
    n += 1             # rebinds the local name; caller unaffected
    return n

def append_one(xs):    # xs is a list — mutable
    xs.append(1)       # mutates the caller's object
```

Nothing inconsistent is happening. Both functions receive a reference; the first
rebinds a local name, the second mutates a shared object. The difference is the
type, not the calling convention.

### 4.3 Comprehensions

A {{term:comprehension}} replaces the build-a-list-in-a-loop pattern:

```python {tier=C name=comprehension-forms}
squares = []                      # the loop form
for x in range(10):
    if x % 2 == 0:
        squares.append(x ** 2)

squares = [x ** 2 for x in range(10) if x % 2 == 0]   # the comprehension
```

They read left to right as *what to produce*, *what to iterate*, *what to keep*.
There are four forms — list `[]`, set `{}`, dict `{k: v}`, and generator `()` —
and the generator form is the one that matters most for large data, because it
produces values on demand instead of building the whole result.

Use a comprehension when it fits on one line and reads cleanly. When it needs
two conditions, a nested loop, and a ternary, write the loop; comprehension is
not a virtue in itself.

### 4.4 Laziness

The difference between these two lines is the difference between a program that
runs and one that exhausts memory:

```python {tier=C name=lazy-vs-eager}
total = sum([line_length(l) for l in open("huge.txt")])   # builds a full list
total = sum(line_length(l) for l in open("huge.txt"))     # streams, one at a time
```

One bracket. The first materialises every value before summing; the second
yields them one at a time and discards each after use. On a hundred-gigabyte
file the first fails and the second does not.

This is {{term:lazy-evaluation}}, and it is the reason {{term:generator}}s exist.

## 5. Formal Explanation

### 5.1 The object model

Every value in Python is an object with three properties:

- **Identity** — its address, fixed for its lifetime, returned by `id()`.
  Compared with `is`.
- **Type** — fixed at creation, returned by `type()`.
- **Value** — the data, mutable or not depending on the type. Compared with
  `==`.

A name is a binding in a {{term:namespace}} mapping to an object reference. The
consequences:

$$
\texttt{a = b} \;\Longrightarrow\; \texttt{a is b} \;\text{is True}
$$ (eq:assignment-aliases)

Assignment creates an alias. It never copies.

> IMPORTANT: `is` compares identity, `==` compares value. They coincide often
> enough to hide the difference and then diverge at the worst moment. Use `==`
> for values and reserve `is` for `None`, `True` and `False` — the singletons
> where identity is the intended test.

### 5.2 Function arguments

Python passes arguments by *object reference*, sometimes called call-by-sharing.
The parameter name is a new binding to the same object.

Therefore:

- **Rebinding a parameter** (`x = something`) affects only the local name.
- **Mutating the object** (`x.append(...)`, `x[0] = ...`) is visible to the
  caller.

The most consequential instance of this is the **mutable default argument**:

```python {tier=C name=mutable-default-trap}
def collect(item, into=[]):     # WRONG
    into.append(item)
    return into
```

The default is evaluated **once**, when the function is defined, not on each
call. Every call that omits `into` shares the same list, which accumulates
across calls indefinitely. The fix is the standard idiom:

```python {tier=C name=mutable-default-fix}
def collect(item, into=None):
    if into is None:
        into = []
    into.append(item)
    return into
```

> WARNING: This is the single most common Python bug that survives code review,
> because the function looks correct and behaves correctly the first time it is
> called. It is caught by linters; run one ({{ch:py-engineering}}).

### 5.3 Copying

Three levels, and choosing the wrong one is a real source of bugs:

$$
\text{alias} \subset \text{shallow copy} \subset \text{deep copy}
$$ (eq:copy-levels)

- **Alias** (`b = a`) — one object, two names.
- **{{term:shallow-copy}}** (`a.copy()`, `list(a)`, `a[:]`) — new container,
  *same* elements. Mutating an element is still shared.
- **Deep copy** (`copy.deepcopy(a)`) — recursively new everything. Correct, and
  slow.

For a flat list of immutable values, shallow is sufficient. For nested
structures — a list of lists, a dict of dicts, a config object — it is not, and
the failure is silent.

### 5.4 Truthiness

Python evaluates any object in a boolean context. Falsy values:

$$
\texttt{False},\; \texttt{None},\; 0,\; 0.0,\; \texttt{""},\; [\,],\; \{\},\; ()
$$ (eq:falsy)

Everything else is truthy.

This makes `if items:` a natural way to say "if there are any items". It also
creates a real hazard:

```python {tier=C name=truthiness-trap}
def apply_discount(price, discount=None):
    if not discount:            # WRONG: 0.0 is falsy
        discount = DEFAULT_DISCOUNT
    return price * (1 - discount)
```

A deliberate discount of `0.0` is indistinguishable from "not supplied", so it
gets silently replaced. The fix is to test for the condition you actually mean:
`if discount is None:`.

> IMPORTANT: This pattern recurs throughout data work. A measured value of zero,
> an empty result set, and a missing field are three different things, and
> truthiness collapses them into one. Whenever the difference matters — and in
> data it usually does — test explicitly. {{ch:py-pandas}} meets the same
> problem again with `NaN`.

### 5.5 Iterators and generators

An {{term:iterator}} implements `__iter__` and `__next__`, raising
`StopIteration` when exhausted. A `for` loop is sugar for driving one.

Iterators are **consumed once**. This catches people:

```python {tier=C name=iterator-exhaustion}
squares = (x**2 for x in range(5))
print(list(squares))    # [0, 1, 4, 9, 16]
print(list(squares))    # []  — already exhausted
```

A {{term:generator}} is the easy way to write an iterator: a function containing
`yield`, which suspends at each yield and resumes on the next request.

```python {tier=C name=generator-basic}
def read_records(path):
    with open(path) as f:
        for line in f:                  # files are themselves iterators
            record = parse(line)
            if record.is_valid:
                yield record            # one at a time, constant memory
```

Memory use is $O(1)$ in the number of records rather than $O(n)$. For the
datasets in {{part:3}} onward, that is frequently the difference between
possible and impossible.

### 5.6 Exceptions

Python's convention is *easier to ask forgiveness than permission*: attempt the
operation and handle failure, rather than checking preconditions first.

```python {tier=C name=eafp}
try:
    value = config["timeout"]
except KeyError:
    value = DEFAULT_TIMEOUT
```

Two rules matter.

**Catch the narrowest exception that could occur.** A bare `except:` catches
everything, including `KeyboardInterrupt` and `SystemExit`, turning
Ctrl-C into a silent no-op and hiding programming errors as though they were
expected conditions.

**Never silence an exception without recording it.** `except Exception: pass` is
how a system fails invisibly for six months.

## 6. Mathematical Foundation

### 6.1 Why Python is slow, quantified

Understanding the cost model tells you what to do about it, and the reason is
not mysterious.

Adding two integers in C is one machine instruction, roughly a nanosecond.
Adding two integers in Python involves: looking up the names in a dictionary,
checking the types, dispatching to the `int.__add__` implementation, allocating
a new heap object for the result because integers are immutable, and updating
reference counts.

That is on the order of **50-100 nanoseconds** — a factor of 50 or more.

The consequence for a loop over $n$ elements:

$$
T_{\text{Python}} \approx n \cdot c_{\text{interp}},
\qquad
T_{\text{NumPy}} \approx c_{\text{call}} + n \cdot c_{\text{machine}}
$$ (eq:loop-cost)

with $c_{\text{interp}} \approx 50\text{-}100$ ns and
$c_{\text{machine}} \approx 1$ ns. The fixed call overhead $c_{\text{call}}$ is
a few microseconds, so:

- For small $n$, the fixed cost dominates and Python may actually win.
- For large $n$, the ratio approaches
  $c_{\text{interp}}/c_{\text{machine}} \approx 50\text{-}100\times$.

{{sec:7-implementation}} measures this. The crossover is typically around a few
hundred elements.

> IMPORTANT: This is the whole argument of {{ch:py-numpy}}, stated numerically.
> Fast Python is Python where the loops happen inside compiled code. The
> language's job is to arrange the data and issue the calls.

### 6.2 Complexity of the built-in types

Choosing the wrong container turns a linear algorithm into a quadratic one, and
the code looks identical.

{#tbl:complexity caption="Average-case complexity of common operations. The membership-test row is the one that most often turns an O(n) algorithm into O(n²)."}

| Operation | `list` | `dict` / `set` |
|---|---|---|
| index / key lookup | $O(1)$ | $O(1)$ |
| membership `x in c` | $O(n)$ | $O(1)$ |
| append / insert-at-end | $O(1)$ amortised | $O(1)$ |
| insert / delete at front | $O(n)$ | $O(1)$ |
| iteration | $O(n)$ | $O(n)$ |

The classic mistake:

```python {tier=C name=membership-quadratic}
seen = []                       # O(n) membership
for item in items:              # n iterations
    if item not in seen:        # => O(n^2) overall
        seen.append(item)
```

Changing `seen = []` to `seen = set()` makes it $O(n)$. On a million items that
is the difference between a fraction of a second and several hours.

Amortised $O(1)$ for `append` means individual appends are occasionally $O(n)$
when the list grows its buffer, but the average over many appends is constant.
For deletion or insertion at the *front*, use `collections.deque`, which is
$O(1)$ at both ends.

## 7. Implementation

```python {tier=A name=object-model-and-cost}
"""Python's object model, its traps, and its cost model — all measured.
"""
import copy
import sys
import time

# --- eq. 13.1: assignment aliases, it does not copy -------------------------
a = [1, 2, 3]
b = a
print(f"a is b            : {a is b}")
b.append(4)
print(f"after b.append(4) : a = {a}   <- a changed too")
b = [9]
print(f"after b = [9]     : a = {a}, b = {b}   <- rebinding did not")

# Immutable objects cannot show this, because they cannot be mutated at all.
s = "hello"
t = s
t += " world"                     # creates a NEW string; s is untouched
print(f"strings           : s = {s!r}, t = {t!r}")

# --- the mutable default argument trap --------------------------------------
def collect_broken(item, into=[]):
    into.append(item)
    return into


def collect_fixed(item, into=None):
    if into is None:
        into = []
    into.append(item)
    return into


print(f"\nbroken: {collect_broken('a')} {collect_broken('b')} "
      f"{collect_broken('c')}   <- accumulates across calls")
print(f"fixed : {collect_fixed('a')} {collect_fixed('b')} "
      f"{collect_fixed('c')}   <- fresh each time")
print(f"the default is bound once, at definition: "
      f"{collect_broken.__defaults__}")

# --- eq. 13.2: three levels of copying --------------------------------------
original = [[1, 2], [3, 4]]
alias = original
shallow = copy.copy(original)          # or list(original), or original[:]
deep = copy.deepcopy(original)

original[0].append(99)                 # mutate a NESTED element

print(f"\nafter mutating a nested element of the original:")
print(f"  original : {original}")
print(f"  alias    : {alias}      <- same object")
print(f"  shallow  : {shallow}      <- new outer list, SHARED inner lists")
print(f"  deep     : {deep}            <- fully independent")
assert shallow[0] is original[0] and deep[0] is not original[0]

# --- truthiness, and where it bites -----------------------------------------
print(f"\n{'value':>10} {'truthy?':>9}")
for v in (0, 0.0, "", [], {}, None, False, "0", [0], 0.1):
    print(f"{v!r:>10} {bool(v):>9}")

DEFAULT_DISCOUNT = 0.10


def price_broken(price, discount=None):
    if not discount:                   # 0.0 is falsy!
        discount = DEFAULT_DISCOUNT
    return price * (1 - discount)


def price_fixed(price, discount=None):
    if discount is None:
        discount = DEFAULT_DISCOUNT
    return price * (1 - discount)


print(f"\ncustomer explicitly gets a 0% discount on a price of 100:")
print(f"  broken: {price_broken(100, 0.0):.2f}   <- silently applied 10%")
print(f"  fixed : {price_fixed(100, 0.0):.2f}  <- correct")

# --- iterators are consumed once --------------------------------------------
gen = (x**2 for x in range(5))
print(f"\nfirst  list(gen): {list(gen)}")
print(f"second list(gen): {list(gen)}   <- exhausted, silently empty")

# --- generators use constant memory -----------------------------------------
n = 2_000_000
eager = [x * 2 for x in range(n)]
lazy = (x * 2 for x in range(n))
print(f"\nlist of {n:,} ints : {sys.getsizeof(eager)/1e6:>8.2f} MB")
print(f"equivalent generator: {sys.getsizeof(lazy)/1e6:>8.6f} MB")
print(f"both sum to the same value: {sum(lazy) == sum(eager)}")
del eager

# --- eq. 13.3: the interpreter's per-operation cost -------------------------
print("\n" + "=" * 66)
print("why Python is slow: per-operation interpreter overhead")
print("=" * 66)

import numpy as np

print(f"{'n':>10} {'python loop':>14} {'numpy':>12} {'speedup':>10}")
for n in (100, 1_000, 100_000, 2_000_000):
    xs = list(range(n))
    arr = np.arange(n)

    t0 = time.perf_counter()
    total_py = 0
    for x in xs:
        total_py += x * 2
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    total_np = int((arr * 2).sum())
    t_np = time.perf_counter() - t0

    assert total_py == total_np
    print(f"{n:>10,} {t_py*1e3:>12.3f}ms {t_np*1e3:>10.3f}ms "
          f"{t_py/max(t_np,1e-9):>9.1f}x")

print("\nAt small n the fixed call overhead dominates and NumPy can lose.")
print("At large n the ratio approaches the interpreter overhead itself.")

# --- table 13.2: container choice changes the complexity class --------------
print("\n" + "=" * 66)
print("membership testing: list is O(n), set is O(1)")
print("=" * 66)
print(f"{'items':>9} {'list (s)':>11} {'set (s)':>10} {'ratio':>9}")
for n in (1_000, 4_000, 16_000):
    items = list(range(n)) * 2

    t0 = time.perf_counter()
    seen_list = []
    for it in items:
        if it not in seen_list:
            seen_list.append(it)
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    seen_set, out = set(), []
    for it in items:
        if it not in seen_set:
            seen_set.add(it)
            out.append(it)
    t_set = time.perf_counter() - t0

    assert seen_list == out
    print(f"{n:>9,} {t_list:>11.4f} {t_set:>10.4f} "
          f"{t_list/max(t_set,1e-9):>8.0f}x")
print("\nQuadrupling n roughly sixteen-folds the list time and quadruples")
print("the set time — the signature of O(n^2) against O(n).")
```

## 8. Practical Example

Streaming a file too large for memory is the canonical use of generators, and it
shows how they compose into a pipeline.

```python {tier=A name=streaming-pipeline}
"""A generator pipeline: parse, filter, aggregate — in constant memory.

Each stage yields to the next, so at most one record exists at a time no matter
how large the input. This is the shape most data-ingestion code should have.
"""
import io
import json
from collections import Counter

# Stand in for a file too large to load; a real one would be open(path).
RAW = "\n".join(
    json.dumps({"user": f"u{i % 7}", "event": ["click", "view", "buy"][i % 3],
                "amount": (i % 13) * 1.5})
    for i in range(10_000)
)


def read_lines(handle):
    """Stage 1: yield lines. Files are already iterators; this makes it explicit."""
    for line in handle:
        line = line.strip()
        if line:
            yield line


def parse_json(lines):
    """Stage 2: decode, skipping malformed records rather than crashing."""
    for lineno, line in enumerate(lines, start=1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Narrow exception, and the failure is recorded rather than hidden.
            print(f"  skipping malformed line {lineno}")


def only_purchases(records):
    """Stage 3: filter."""
    for r in records:
        if r.get("event") == "buy":
            yield r


def summarise(records):
    """Stage 4: aggregate. This is the only stage that accumulates."""
    totals, counts = Counter(), Counter()
    for r in records:
        totals[r["user"]] += r["amount"]
        counts[r["user"]] += 1
    return totals, counts


# The pipeline is assembled but nothing has executed yet — generators are lazy.
handle = io.StringIO(RAW)
pipeline = only_purchases(parse_json(read_lines(handle)))
print(f"pipeline object: {type(pipeline).__name__}  <- nothing computed yet")

totals, counts = summarise(pipeline)      # execution happens here

print(f"\n{'user':>6} {'purchases':>11} {'total':>10} {'mean':>9}")
for user in sorted(totals):
    print(f"{user:>6} {counts[user]:>11} {totals[user]:>10.2f} "
          f"{totals[user]/counts[user]:>9.2f}")

# --- the same pipeline handles malformed input without special-casing -------
DIRTY = RAW.split("\n")[:5] + ["{not json}", ""] + RAW.split("\n")[5:10]
print("\nre-running over input containing a malformed line:")
t2, c2 = summarise(only_purchases(parse_json(read_lines(iter(DIRTY)))))
print(f"  processed {sum(c2.values())} purchases despite the bad record")

# --- itertools composes generators without materialising anything -----------
import itertools

handle = io.StringIO(RAW)
first_three = list(itertools.islice(parse_json(read_lines(handle)), 3))
print(f"\nfirst 3 records via islice (the rest never parsed):")
for r in first_three:
    print(f"  {r}")

# Chunking a stream into batches — the shape every training loop needs.
def batched(iterable, size):
    it = iter(iterable)
    while batch := list(itertools.islice(it, size)):
        yield batch


handle = io.StringIO(RAW)
sizes = [len(b) for b in batched(parse_json(read_lines(handle)), 3000)]
print(f"\nbatch sizes from a 10,000-record stream: {sizes}")
print("Constant memory throughout — no stage ever held more than one batch.")
```

## 9. Common Mistakes

**Assuming assignment copies.** It aliases. Use `.copy()` or
`copy.deepcopy()` when you need independence.

**Mutable default arguments.** Evaluated once at definition. Use `None` and
create inside.

**Shallow-copying nested structures.** The inner objects are still shared.

**Using `not x` when you mean `x is None`.** Zero, empty string and empty list
are all falsy, and in data work all three are meaningful values.

**Using `is` to compare values.** It compares identity. Small integers and short
strings are cached, so `a is b` may be `True` for equal values *sometimes* —
which is worse than never, because it hides the bug during testing.

**Reusing an exhausted iterator.** It silently yields nothing. If you need two
passes, materialise a list or re-create the generator.

**Bare `except:`.** Catches `KeyboardInterrupt` and `SystemExit` too. Catch
`Exception` at worst, and something narrower whenever you can name it.

**Silencing exceptions with `pass`.** At minimum, log them
({{ch:py-engineering}}).

**Membership tests against a list in a loop.** $O(n^2)$. Use a set.

**Modifying a list while iterating over it.** Skips elements, and no error is
raised. Iterate over a copy, or build a new list.

**Writing loops where an array expression would do.** The whole of
{{ch:py-numpy}}.

## 10. Connection to Previous Chapters

{{ch:math-notation}} established the notation conventions this book's code
comments follow, and the correspondence between mathematical summation and
Python loops.

Forward within Part II: {{ch:py-functions-classes}} builds on the object model
here to cover functions, classes and modules. {{ch:py-numpy}} is the answer to
{{sec:6-mathematical-foundation}}'s cost analysis — the same computation moved
into compiled code. {{ch:py-pandas}} meets the mutability question again in the
form of `SettingWithCopyWarning`, which is aliasing wearing a different hat.
{{ch:py-engineering}} covers the linters that catch most of
{{sec:9-common-mistakes}} automatically.

Beyond Part II: the generator pipeline in {{sec:8-practical-example}} is the
shape of every data-loading path from {{part:3}} onward, and the batching
function at its end is what a training loop consumes ({{ch:dl-optimizers}}).

{{cite:pep8}} is the style guide the code in this book follows.

## 11. Exercises

**Beginner**

1. Predict the output, then check: `a = [1,2]; b = a; b += [3]; print(a)`.
   Now with `b = b + [3]`. Explain the difference.
2. Write a comprehension producing the squares of odd numbers below 20.
3. Which of these are falsy: `0`, `"0"`, `[]`, `[0]`, `{}`, `None`, `" "`?
4. Convert a loop that builds a list of parsed lines into a generator function.
5. What does `x is y` test, and when should you use it?

**Intermediate**

6. Write a function demonstrating the mutable-default trap, then fix it. Explain
   why `__defaults__` reveals the cause.
7. Build a nested structure where a shallow copy causes a bug, and show that a
   deep copy fixes it.
8. Rewrite an $O(n^2)$ deduplication as $O(n)$ while preserving order.
9. Write a generator yielding fixed-size batches from any iterable, handling a
   final partial batch correctly.
10. Explain why `for x in items: items.remove(x)` misbehaves, and give two
    correct alternatives.
11. Write a function that reads a file and returns the mean line length, in
    constant memory.

**Advanced**

12. Implement an iterator class supporting `__iter__` and `__next__` without
    using `yield`, then rewrite it as a generator. Compare the line counts.
13. Explain why `sys.getsizeof` on a generator is constant regardless of how
    many values it will produce.
14. Time list-append against a preallocated list against a NumPy array for
    $n \in \{10^3, 10^5, 10^7\}$. Explain each crossover.
15. Python caches small integers. Find the range experimentally, and explain why
    relying on that caching is a bug.

**Implementation**

16. Build a generator pipeline that reads a CSV, filters rows, transforms a
    column, and writes the result, never holding more than one row in memory.
17. Write `deep_equal(a, b)` comparing nested structures by value, and a test
    suite distinguishing it from `==` and `is`.
18. Implement an LRU cache decorator with an explicit dictionary, then compare it
    against `functools.lru_cache` for correctness and speed.
19. Reproduce the timing table in {{sec:7-implementation}} on your own machine
    and identify the crossover point where NumPy overtakes a Python loop.

**Reasoning**

20. Python is slow, yet it dominates machine learning. Reconcile these, using
    {{eq:loop-cost}}.
21. Truthiness makes `if items:` read well but conflates zero with absent. Was
    that a good language design decision? Argue both sides.

## 12. Chapter Summary

Python names are labels bound to objects; assignment moves labels and never
copies. Whether that matters depends on mutability: rebinding a parameter is
local, but mutating a shared object is visible to the caller. This one rule
explains aliasing, the mutable-default-argument trap, and most of the surprises
that Python holds for programmers arriving from value-semantics languages.

Copying comes in three strengths — alias, shallow, deep. Shallow copies share
their nested contents, which is correct for flat data and silently wrong for
nested structures.

Truthiness treats zero, empty containers and `None` alike. That reads well and
is a genuine hazard in data work, where a measured zero, an empty result and a
missing value are three different things. Test with `is None` when you mean
absence.

Iterators produce values on demand and are consumed once. Generators are the
convenient way to write them, and turn an $O(n)$-memory pipeline into an $O(1)$
one — which is what makes datasets larger than memory tractable.

Python is roughly 50-100× slower per operation than compiled code, because every
operation carries dynamic dispatch, heap allocation and reference counting. That
cost is fixed per *operation*, not per unit of data, which is precisely why the
answer is to move the loop into compiled code rather than to write cleverer
Python — the subject of {{ch:py-numpy}}.

Container choice changes complexity class. Membership testing is $O(n)$ in a
list and $O(1)$ in a set, and confusing the two silently converts a linear
algorithm into a quadratic one.
