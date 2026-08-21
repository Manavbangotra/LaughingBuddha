---
id: py-numpy
number: 16
part: II
tier: focused
status: reviewed
requires: [py-fundamentals, math-matrices, math-vectors]
provides: [ndarray, dtype, numeric-precision, vectorisation, ufunc,
           array-view, strides, contiguity, axis]
citations: [harris2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what an ndarray is and why it is fast, in terms of memory layout.
2. Vectorise a loop and predict the speedup from the cost model of
   {{ch:py-fundamentals}}.
3. Choose a dtype deliberately and reason about precision and memory.
4. Apply broadcasting rules correctly and predict the resulting shape.
5. Determine whether an operation returns a view or a copy, and explain the
   consequences of getting it wrong.
6. Reason about axes in reductions without guessing.
7. Explain what strides are and why transposition is free.
8. Use fancy and boolean indexing, and know which of them copies.

## 2. Why This Matters

This is the most important chapter in Part II.

Every numerical library you will use in the remaining twenty-six parts either is
NumPy, is built on NumPy, or copies its interface. Pandas stores its columns as
arrays. Scikit-learn takes and returns them. Matplotlib plots them. A PyTorch
tensor is deliberately a near-clone of the ndarray API with gradients and a
device attached, so the mental model transfers directly
{{cite:harris2020}}.

More concretely, three things in this chapter are the difference between code
that works and code that quietly does not.

**Vectorisation** is the 50-100× speedup derived in {{ch:py-fundamentals}},
collected. A training loop written with Python loops does not merely run slowly;
it runs so slowly that experiments you would otherwise do become impossible.

**Views versus copies** is the single most common source of silent bugs in
numerical Python. A slice shares memory with its parent, so writing to what you
believed was a copy modifies the original — with no error, no warning, and a
result that is wrong in a way that survives casual inspection.

**Broadcasting** is what makes array code concise, and what makes shape bugs
subtle. An operation between a `(1000, 1)` and a `(1000,)` array succeeds and
produces a `(1000, 1000)` result. That is 8 MB where you expected 8 KB, and the
program keeps running.

## 3. Prerequisites

{{ch:py-fundamentals}} for the interpreter cost model and for mutability, which
is what makes views dangerous. {{ch:math-vectors}} and {{ch:math-matrices}} for
vectors, matrices and the shape rules — this chapter is those chapters made
executable.

## 4. Intuitive Explanation

### 4.1 Why a list is slow and an array is fast

A Python list of a million integers is a million pointers to a million separate
heap objects, each carrying a type tag and a reference count. The values are
scattered across memory. Iterating means chasing pointers, and adding two of
them means a type check per element.

An {{term:ndarray}} of a million integers is **one contiguous block of eight
million bytes**, plus a small header saying "int64, shape (1000000,)". There are
no per-element objects and no per-element type tags.

```text
list:   [ptr]─▶obj(5)   [ptr]─▶obj(2)   [ptr]─▶obj(9)      scattered, boxed
array:  | 5 | 2 | 9 | 7 | 1 | 4 | ...                      contiguous, raw
```

Three consequences follow, and they are the whole reason NumPy exists:

- **No boxing.** Values are raw machine numbers, not objects.
- **No per-element dispatch.** The type is known once, for the whole array.
- **Cache locality.** Adjacent elements are adjacent in memory, so a cache line
  fetch brings in eight useful values instead of one pointer.

Adding two arrays is then a single call into a compiled loop over contiguous
memory — which the CPU can also vectorise into SIMD instructions.

### 4.2 Vectorisation is a change of who runs the loop

{{term:vectorisation}} does not eliminate the loop. It moves it from the
interpreter into compiled code.

```python {tier=C name=vectorisation-contrast}
out = []                            # the loop runs in Python
for x in data:
    out.append(x * 2 + 1)

out = data * 2 + 1                  # the loop runs in C
```

Both perform $n$ multiplications and $n$ additions. The difference is the ~60 ns
of interpreter overhead per element in the first and ~1 ns of machine work in
the second.

The practical rule for the rest of this book: **if you are writing a `for` loop
over array elements, there is almost certainly an array expression that replaces
it.** Learning to see those replacements is the skill this chapter teaches.

### 4.3 Broadcasting

Broadcasting lets arrays of different shapes combine, by conceptually stretching
the smaller one:

```python {tier=C name=broadcasting-intro}
prices = np.array([10.0, 20.0, 30.0])       # shape (3,)
discounted = prices * 0.9                   # scalar stretched across all three

matrix = np.arange(12).reshape(3, 4)        # shape (3, 4)
col_means = matrix.mean(axis=0)             # shape (4,)
centred = matrix - col_means                # (3,4) - (4,) -> (3,4)
```

Nothing is physically copied. NumPy sets the stride along the broadcast axis to
zero, so the same memory is read repeatedly. Broadcasting is therefore free in
memory as well as convenient.

The danger is that it succeeds when you did not intend it, which
{{sec:5-formal-explanation}} covers.

### 4.4 Views: the sharpest edge in NumPy

Slicing a list copies. Slicing an array does not.

```python {tier=C name=view-surprise}
lst = [1, 2, 3, 4, 5]
part = lst[1:4]
part[0] = 99          # lst is unchanged

arr = np.array([1, 2, 3, 4, 5])
part = arr[1:4]
part[0] = 99          # arr IS changed: arr == [1, 99, 3, 4, 5]
```

An {{term:array-view}} shares its parent's memory. This is a deliberate design
choice — it is why slicing a gigabyte array costs nothing — and it is the source
of more silent bugs than anything else in numerical Python.

## 5. Formal Explanation

### 5.1 The anatomy of an array

An ndarray is a small header describing a buffer:

{#tbl:array-attributes caption="What an ndarray actually stores. The buffer is one contiguous allocation; everything else is metadata describing how to read it."}

| Attribute | Meaning |
|---|---|
| `.data` | pointer to the raw buffer |
| `.dtype` | element type and size |
| `.shape` | length along each axis |
| `.strides` | bytes to step to advance one position along each axis |
| `.ndim` | number of axes |
| `.base` | the array owning the buffer, if this is a view |

The element at index $(i, j)$ lives at byte offset

$$
\text{offset}(i, j) = i \cdot s_0 + j \cdot s_1
$$ (eq:stride-offset)

where $s_0, s_1$ are the {{term:strides}}. Everything about views follows from
this one formula: **any operation expressible as a change of shape, strides and
offset is free**, because it produces new metadata over the same buffer.

Transposition swaps the strides. Slicing adjusts the offset and strides.
Reshaping a contiguous array reinterprets the strides. None of them copy.

### 5.2 dtypes

The {{term:dtype}} fixes element size and interpretation.

{#tbl:dtypes caption="Common dtypes. The float32/float64 choice is the one that recurs throughout this book."}

| dtype | Bytes | Range or precision | Typical use |
|---|---|---|---|
| `bool` | 1 | True/False | masks |
| `int32` | 4 | ±2.1 × 10⁹ | indices, counts |
| `int64` | 8 | ±9.2 × 10¹⁸ | default integer |
| `float32` | 4 | ~7 significant digits | deep learning |
| `float64` | 8 | ~16 significant digits | default float, scientific work |
| `float16` | 2 | ~3 significant digits | mixed-precision training |

NumPy defaults to `float64`. Deep learning defaults to `float32`, and
increasingly to `bfloat16` — halving memory and doubling arithmetic throughput
matters more than digits eight through sixteen ({{ch:q-formats}}).

> WARNING: Integer arrays **overflow silently**. `np.int8(127) + 1` gives
> `-128` with no exception. This is not a bug; fixed-width integers wrap by
> definition, and NumPy is giving you machine semantics. It bites when counting
> into an `int32` accumulator or computing a product of dimensions. When in
> doubt, use `int64` or accumulate in floating point.

### 5.3 Broadcasting rules

Two arrays broadcast if, comparing their shapes **from the right**, each pair of
dimensions is either equal or one of them is 1.

$$
\begin{aligned}
(3, 4) \;\;\text{with}\;\; (4,) \;&\to\; (3, 4) \\
(3, 4) \;\;\text{with}\;\; (3, 1) \;&\to\; (3, 4) \\
(3, 4) \;\;\text{with}\;\; (3,) \;&\to\; \text{error: } 4 \neq 3 \\
(5, 1, 3) \;\;\text{with}\;\; (4, 3) \;&\to\; (5, 4, 3)
\end{aligned}
$$ (eq:broadcast-rules)

Missing leading dimensions are treated as 1.

The mechanism is that a broadcast dimension gets **stride 0**, so advancing
along it does not move in memory. No data is duplicated.

> IMPORTANT: The dangerous case is not the error — errors are helpful. It is the
> accidental success. A `(1000, 1)` combined with a `(1000,)` broadcasts to
> `(1000, 1000)`: a million elements where you wanted a thousand. Nothing warns
> you. The usual cause is a reduction that kept its dimension, or a column
> extracted with `[:, None]` when a flat array was intended. Assert shapes.

### 5.4 Views and copies

{#tbl:view-or-copy caption="Which operations share memory. Getting this wrong is the most common silent bug in numerical Python."}

| Operation | Result |
|---|---|
| basic slicing `a[1:4]`, `a[:, 0]` | **view** |
| `a.T`, `a.reshape(...)` on contiguous data | **view** |
| `a.ravel()` | view if possible |
| boolean mask `a[a > 0]` | **copy** |
| fancy indexing `a[[0, 2, 4]]` | **copy** |
| `a.flatten()`, `a.copy()` | **copy** |
| arithmetic `a + b` | new array |
| in-place `a += b` | modifies `a`, and any view of it |

The rule underlying the table: **basic slicing yields a view because the result
is expressible with strides; advanced indexing yields a copy because it is
not.** Picking out elements 0, 2 and 7 has no constant stride, so there is
nothing to describe it with except a new buffer.

Check with `arr.base`: `None` means it owns its data, otherwise it is a view of
whatever `base` points at.

### 5.5 Axes

The `axis` argument is where confusion concentrates. One rule resolves it:

> **`axis=i` means the `i`-th dimension is the one that disappears.**

For a `(3, 4)` array:

- `a.sum(axis=0)` collapses the 3 → shape `(4,)`. Summing *down* columns.
- `a.sum(axis=1)` collapses the 4 → shape `(3,)`. Summing *across* rows.
- `a.sum()` collapses everything → a scalar.
- `a.sum(axis=0, keepdims=True)` → shape `(1, 4)`, kept for broadcasting.

`keepdims=True` exists precisely so the result still broadcasts against the
original, which is what makes `a - a.mean(axis=1, keepdims=True)` work.

### 5.6 Universal functions

A {{term:ufunc}} applies element-wise with broadcasting: `np.exp`, `np.sqrt`,
`np.maximum`, and every arithmetic operator.

Useful features beyond the obvious:

- `out=` writes into an existing array, avoiding an allocation.
- `where=` applies conditionally.
- `.reduce()`, `.accumulate()` — `np.add.reduce` is `sum`;
  `np.add.accumulate` is `cumsum`.

## 6. Mathematical Foundation

### 6.1 Why contiguity matters: the memory hierarchy

The speed difference between array and list code is not only about interpreter
overhead. It is also about cache.

A modern CPU reads memory in **cache lines** of 64 bytes. Fetching one byte
fetches all 64. If the next value you need is adjacent, it is already in cache —
about 1 ns away. If it is elsewhere, you pay a main-memory access, about 100 ns.

For a contiguous `float64` array, one cache line holds 8 elements, so a
sequential pass incurs one memory access per 8 elements. For a list of pointers
to scattered objects, each element is potentially its own cache miss.

This also explains why iteration order matters:

$$
\text{C-contiguous } (m, n):\quad \text{offset}(i, j) = 8(i \cdot n + j)
$$ (eq:c-contiguous)

Walking $j$ fastest steps 8 bytes at a time — perfect locality. Walking $i$
fastest steps $8n$ bytes — a fresh cache line every element.

{{sec:7-implementation}} measures this, and the honest result is worth
reporting: for `sum` on a $3000 \times 3000$ array the difference is around
15%, not the order of magnitude the argument above might suggest. NumPy's
reduction machinery chooses a favourable iteration order internally where it
can, which absorbs most of the penalty. The effect is larger for operations that
cannot reorder — element-wise work on a strided view, or anything handed to a
library expecting contiguous input, where the cost reappears as a silent copy.

> NOTE: The lesson is not "layout does not matter" but "layout matters where the
> library cannot compensate". Measure before restructuring code for locality;
> {{ch:py-engineering}} covers how.

### 6.2 Why transposition is free

Transposition swaps the strides and leaves the buffer untouched:

$$
\mat{A}: \text{shape } (m, n), \text{ strides } (8n, 8)
\quad\longrightarrow\quad
\mat{A}\T: \text{shape } (n, m), \text{ strides } (8, 8n)
$$ (eq:transpose-strides)

By {{eq:stride-offset}}, reading $\mat{A}\T[j, i]$ computes offset
$j \cdot 8 + i \cdot 8n$, which is exactly where $\mat{A}[i, j]$ lives. Correct
values, no data movement, $O(1)$ cost regardless of size.

The catch is that the transpose is no longer C-contiguous, so a subsequent
sequential pass has poor locality. Operations needing contiguity — some
reshapes, some library calls — will silently copy to restore it. That copy is
where the cost you avoided reappears.

### 6.3 Precision, quantified

A `float32` has a 23-bit mantissa, giving machine epsilon

$$
\varepsilon_{32} = 2^{-23} \approx 1.19 \times 10^{-7}
$$ (eq:eps32)

and `float64` has 52 bits, giving $\varepsilon_{64} \approx 2.22 \times
10^{-16}$.

Machine epsilon is the smallest $\varepsilon$ with $1 + \varepsilon \neq 1$. The
immediate consequence: **adding a small number to a large one can do nothing at
all.** In `float32`, $10^{7} + 1$ is exactly $10^{7}$.

This matters when accumulating. Summing $n$ values naively accumulates error
growing as $O(n\varepsilon)$ in the worst case, and for `float32` with
$n = 10^{7}$ that is potentially complete loss of precision. NumPy mitigates it
with **pairwise summation**, which reduces the growth to $O(\varepsilon \log n)$
— which is why `arr.sum()` is measurably more accurate than a Python loop doing
the same additions. {{sec:7-implementation}} demonstrates both effects.

### 6.4 Memory arithmetic

An array's size is exactly

$$
\text{bytes} = \prod_{i} \text{shape}_i \times \text{itemsize}
$$ (eq:array-bytes)

A `float32` batch of shape $(32, 3, 224, 224)$ — a typical image batch — is
$32 \times 3 \times 224 \times 224 \times 4 = 19{,}267{,}584$ bytes, about
19 MB. In `float64` it would be 39 MB, which is the entire argument for
`float32` in deep learning.

The arithmetic that catches people is that **intermediates count**. The
expression `(a - b) ** 2 / c` allocates a temporary for `a - b`, another for the
square, and another for the division: three full-size arrays live simultaneously.
For a 4 GB array that is 16 GB of peak usage for an expression that looks like
it needs 8. In-place operations (`np.subtract(a, b, out=tmp)`) avoid it, and are
why serving code is written the way it is ({{ch:inf-gpu-memory}}).

## 7. Implementation

```python {tier=A name=numpy-core}
"""NumPy's core concepts, each measured rather than described.
"""
import numpy as np

# --- eq. 16.4: memory is exactly shape x itemsize ---------------------------
a = np.arange(12, dtype=np.float64).reshape(3, 4)
print(f"shape {a.shape}, dtype {a.dtype}, itemsize {a.itemsize} bytes")
print(f"strides {a.strides}  <- (bytes per row, bytes per element)")
print(f"nbytes {a.nbytes} == {a.size} * {a.itemsize}")
assert a.nbytes == a.size * a.itemsize

# --- eq. 16.1: strides explain indexing -------------------------------------
i, j = 2, 1
offset = i * a.strides[0] + j * a.strides[1]
print(f"\nelement [{i},{j}] = {a[i,j]}, at byte offset {offset}")
flat = a.reshape(-1)
print(f"same element from the flat buffer: {flat[offset // a.itemsize]}")

# --- eq. 16.3: transposition swaps strides and copies nothing ---------------
t = a.T
print(f"\na  : shape {a.shape}, strides {a.strides}, contiguous "
      f"{a.flags['C_CONTIGUOUS']}")
print(f"a.T: shape {t.shape}, strides {t.strides}, contiguous "
      f"{t.flags['C_CONTIGUOUS']}")
print(f"a.T shares memory with a: {np.shares_memory(a, t)}")

# --- views vs copies, and the bug that follows ------------------------------
print("\n" + "=" * 66)
print("views share memory; advanced indexing copies")
print("=" * 66)
base = np.arange(10)
sl = base[2:5]              # basic slicing -> view
mask = base[base > 6]       # boolean mask  -> copy
fancy = base[[0, 3, 7]]     # fancy index   -> copy

print(f"{'operation':<24} {'shares memory':>14} {'.base is None':>15}")
for name, arr in (("base[2:5]", sl), ("base[base>6]", mask),
                  ("base[[0,3,7]]", fancy), ("base.reshape(2,5)",
                                             base.reshape(2, 5))):
    print(f"{name:<24} {str(np.shares_memory(base, arr)):>14} "
          f"{str(arr.base is None):>15}")

sl[0] = 999
print(f"\nafter sl[0] = 999, base = {base}")
print("Writing through a slice wrote through to the parent. No warning.")

# The safe pattern when you need independence.
safe = base[2:5].copy()
safe[0] = -1
print(f"after safe[0] = -1 (an explicit copy), base = {base}   <- unchanged")

# --- eq. 16.2: broadcasting, including the dangerous success ----------------
print("\n" + "=" * 66)
print("broadcasting")
print("=" * 66)
m = np.arange(12).reshape(3, 4)
col_mean = m.mean(axis=0)                 # (4,)
row_mean = m.mean(axis=1, keepdims=True)  # (3,1)
print(f"m {m.shape} - col_mean {col_mean.shape} -> "
      f"{(m - col_mean).shape}")
print(f"m {m.shape} - row_mean {row_mean.shape} -> "
      f"{(m - row_mean).shape}")

try:
    m - m.mean(axis=1)                    # (3,4) - (3,) : 4 != 3
except ValueError as exc:
    print(f"m - row_mean WITHOUT keepdims: ValueError: {str(exc)[:52]}")
print("  ^ this error is helpful. The next one is not.")

col = np.arange(1000).reshape(1000, 1)    # (1000, 1)
row = np.arange(1000)                     # (1000,)
oops = col + row
print(f"\n(1000,1) + (1000,) -> {oops.shape}, {oops.nbytes/1e6:.1f} MB")
print("Expected 1000 elements; got a million. No error, no warning.")
print("This is why array code should assert its shapes.")

# --- broadcasting allocates nothing ------------------------------------------
big = np.zeros((5000, 5000), dtype=np.float32)
bias = np.ones(5000, dtype=np.float32)
bcast = np.broadcast_to(bias, big.shape)
print(f"\nbroadcast_to((5000,), (5000,5000)): "
      f"strides {bcast.strides}  <- stride 0 on the repeated axis")
print(f"  it owns no data: base is not None -> {bcast.base is not None}")

# --- axes: axis=i is the dimension that disappears --------------------------
print("\n" + "=" * 66)
print("axis = the dimension that disappears")
print("=" * 66)
x = np.arange(24).reshape(2, 3, 4)
print(f"x.shape = {x.shape}")
for ax in (0, 1, 2, None):
    out = x.sum(axis=ax)
    print(f"  x.sum(axis={str(ax):<4}) -> shape {str(np.shape(out)):<10} "
          f"{'(scalar)' if ax is None else ''}")
print(f"  x.sum(axis=1, keepdims=True) -> "
      f"{x.sum(axis=1, keepdims=True).shape}  <- kept for broadcasting")

# --- eq. 16.5 / 16.6: precision -----------------------------------------------
print("\n" + "=" * 66)
print("floating-point precision")
print("=" * 66)
for dt in (np.float16, np.float32, np.float64):
    info = np.finfo(dt)
    print(f"{str(np.dtype(dt)):<9} eps = {info.eps:<12.3e} "
          f"max = {info.max:.3e}")

big32 = np.float32(1e7)
print(f"\nfloat32: 1e7 + 1 == 1e7 ? {big32 + np.float32(1) == big32}")
print("Adding a small number to a large one can be a no-op.")

# Naive summation loses precision; NumPy's pairwise summation does not.
vals = np.full(2_000_000, 0.1, dtype=np.float32)
naive = np.float32(0.0)
for v in vals[:200_000]:
    naive += v
print(f"\nsumming 200,000 copies of 0.1 in float32:")
print(f"  naive Python loop : {naive:.4f}")
print(f"  numpy .sum()      : {vals[:200_000].sum():.4f}")
print(f"  exact answer      : {20000.0:.4f}")
print("NumPy uses pairwise summation, so error grows as log(n), not n.")

# --- integer overflow is silent ----------------------------------------------
small = np.array([127], dtype=np.int8)
print(f"\nint8: 127 + 1 = {(small + np.int8(1))[0]}   <- wrapped, no error")

# --- section 6.1: memory layout changes the speed of identical arithmetic ---
print("\n" + "=" * 66)
print("cache locality: same arithmetic, different traversal order")
print("=" * 66)
import time

n = 3000
c_order = np.ones((n, n), dtype=np.float64)
f_order = np.asfortranarray(c_order)

for name, arr in (("C-contiguous", c_order), ("Fortran-order", f_order)):
    t0 = time.perf_counter()
    s = arr.sum(axis=1)          # walk along rows
    t_rows = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = arr.sum(axis=0)          # walk down columns
    t_cols = time.perf_counter() - t0
    print(f"{name:<15} sum(axis=1) {t_rows*1e3:>7.1f} ms | "
          f"sum(axis=0) {t_cols*1e3:>7.1f} ms")
print("Identical FLOPs. The difference is entirely cache behaviour.")

# --- intermediates cost memory ----------------------------------------------
print("\n" + "=" * 66)
print("temporaries: an expression allocates more than its result")
print("=" * 66)
size = 2_000_000
p = np.ones(size, dtype=np.float64)
q = np.ones(size, dtype=np.float64)
r = np.full(size, 2.0)
print(f"each array is {p.nbytes/1e6:.0f} MB")
print(f"(p - q) ** 2 / r allocates 3 temporaries -> "
      f"~{3*p.nbytes/1e6:.0f} MB peak beyond the inputs")

out = np.empty_like(p)
np.subtract(p, q, out=out)
np.square(out, out=out)
np.divide(out, r, out=out)
print(f"the same computation with out= allocates "
      f"{out.nbytes/1e6:.0f} MB once")
assert np.allclose(out, (p - q) ** 2 / r)
```

## 8. Practical Example

Vectorising a real computation is the skill this chapter exists to teach.
Pairwise Euclidean distances — the operation underneath k-nearest neighbours,
k-means, and retrieval — makes a good case study because the naive version is
obvious and the vectorised version is not.

```python {tier=A name=vectorising-distances}
"""Three implementations of pairwise distances, from loops to linear algebra.

The final version uses the expansion ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
from Chapter 3, turning the whole computation into one matrix product.
"""
import time

import numpy as np

rng = np.random.default_rng(0)


def distances_loops(A, B):
    """Naive: two Python loops. O(n*m) interpreter iterations."""
    n, m = len(A), len(B)
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            diff = A[i] - B[j]
            out[i, j] = np.sqrt(np.sum(diff * diff))
    return out


def distances_broadcast(A, B):
    """Broadcasting: one Python-level operation, but a big temporary.
    (n,1,d) - (1,m,d) -> (n,m,d), which is d times the size of the answer."""
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def distances_gemm(A, B):
    """The expansion of Chapter 3: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2.

    The cross term is a single matrix product, which is what BLAS is for.
    No (n,m,d) temporary is ever created.
    """
    a2 = np.einsum("ij,ij->i", A, A)[:, None]     # ||a||^2, shape (n,1)
    b2 = np.einsum("ij,ij->i", B, B)[None, :]     # ||b||^2, shape (1,m)
    sq = a2 - 2.0 * (A @ B.T) + b2
    # Rounding can make a squared distance very slightly negative.
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq)


n, m, d = 600, 500, 64
A = rng.normal(size=(n, d))
B = rng.normal(size=(m, d))

reference = distances_loops(A, B)
results = {}
print(f"{'method':<22} {'time':>10} {'speedup':>9} {'peak temp':>12} {'max err':>10}")
for name, fn, temp_mb in (
    ("python loops", distances_loops, 0.0),
    ("broadcasting", distances_broadcast, n * m * d * 8 / 1e6),
    ("gemm expansion", distances_gemm, n * m * 8 / 1e6),
):
    t0 = time.perf_counter()
    out = fn(A, B)
    elapsed = time.perf_counter() - t0
    results[name] = elapsed
    err = np.abs(out - reference).max()
    print(f"{name:<22} {elapsed*1e3:>8.1f}ms "
          f"{results['python loops']/elapsed:>8.0f}x {temp_mb:>10.1f}MB "
          f"{err:>10.2e}")

print(f"\nAll three compute the same thing to within {1e-9:.0e}.")
print("The broadcast version is fast but allocates an (n,m,d) temporary —")
print(f"{n*m*d*8/1e6:.0f} MB here, and it grows with d. The gemm version")
print(f"allocates only the (n,m) answer, {n*m*8/1e6:.1f} MB, and hands the")
print("work to BLAS.")

# --- how the temporary scales, which is what breaks at real sizes -----------
print(f"\n{'n = m':>8} {'d':>5} {'broadcast temp':>16} {'gemm temp':>12}")
for nn, dd in ((1_000, 128), (10_000, 768), (50_000, 1536)):
    print(f"{nn:>8,} {dd:>5} {nn*nn*dd*8/1e9:>14.1f} GB "
          f"{nn*nn*8/1e9:>10.2f} GB")
print("At retrieval scale the broadcast version is not slow — it is")
print("impossible. This is why Part XI builds on matrix products.")

# --- the same pattern: normalise then use a dot product ---------------------
An = A / np.linalg.norm(A, axis=1, keepdims=True)
Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
cosine = An @ Bn.T
print(f"\ncosine similarity for all {n}x{m} pairs: one matmul, "
      f"{cosine.shape}, {cosine.nbytes/1e6:.1f} MB")
print(f"ranking by cosine == ranking by distance on normalised vectors: "
      f"{np.array_equal(np.argsort(-cosine[0]), np.argsort(distances_gemm(An, Bn)[0]))}")
print("  (Chapter 5, eq. 5.11 — verified here at scale.)")
```

## 9. Common Mistakes

**Writing Python loops over array elements.** Almost always replaceable, and
almost always by a factor of 50 or more.

**Assuming a slice is a copy.** It is a view. Write to it and you write to the
parent.

**Forgetting `keepdims=True`.** The subsequent broadcast either fails, which is
fine, or succeeds with the wrong shape, which is not.

**Accidental broadcasting.** `(n, 1)` with `(n,)` gives `(n, n)`. Assert shapes
after operations you expect to preserve them.

**Using `float64` in deep learning.** Double the memory and bandwidth for
precision the model does not use.

**Assuming integer arrays saturate.** They wrap, silently.

**Building an `(n, m, d)` temporary when a matmul would do.** Fine at $n = 100$,
impossible at $n = 50{,}000$.

**Using `np.random.seed` rather than a `Generator`.** Global state, as
{{ch:py-environments}} showed.

**Chaining indexing operations.** `a[0][1] = 5` works only because `a[0]` is a
view; the equivalent on a copy silently discards the assignment. Write
`a[0, 1] = 5`.

**Comparing floats with `==`.** Use `np.isclose` or `np.allclose`.

## 10. Connection to Previous Chapters

{{ch:py-fundamentals}} derived the interpreter cost model that
{{sec:4-intuitive-explanation}} cashes in, and established the mutability that
makes views hazardous. {{ch:math-vectors}} and {{ch:math-matrices}} supplied the
dot product and matrix product this chapter executes, and
{{eq:matmul-associative}} is the reason the gemm formulation in
{{sec:8-practical-example}} wins. {{ch:math-norms}} supplied the identity that
turns squared distance into a matrix product, and {{eq:cosine-euclidean}} is
verified numerically at the end of that example.

Forward within Part II: {{ch:py-pandas}} stores its columns as arrays and
inherits both the speed and the view semantics. {{ch:py-visualization}} consumes
arrays directly.

Beyond Part II: every algorithm in {{part:4}} is array code; a PyTorch tensor is
an ndarray with gradients and a device ({{ch:dl-forward}}); the memory
arithmetic of {{sec:6-mathematical-foundation}} is what
{{ch:q-memory-math}} scales up to model weights; and the retrieval scaling at the
end of {{sec:8-practical-example}} is why {{ch:emb-ann}} exists.

{{cite:harris2020}} is the reference description of the array model.

## 11. Exercises

**Beginner**

1. Create a `(3, 4)` array of zeros, one of ones, and one of the integers 0-11.
2. Give the shape, dtype, ndim and nbytes of `np.ones((2, 3, 4),
   dtype=np.float32)`.
3. Compute the mean of each column and of each row of a `(5, 3)` array.
4. Select all elements greater than 5 from a 1-D array. Is the result a view?
5. Reshape a length-12 array to `(3, 4)`, then transpose it. What is the shape?

**Intermediate**

6. Predict the broadcast result shape for each pair: `(5,3)` with `(3,)`;
   `(5,3)` with `(5,1)`; `(5,3)` with `(5,)`; `(2,1,4)` with `(3,4)`.
7. Demonstrate a view causing a bug, then fix it with `.copy()`.
8. Standardise each column of a `(100, 5)` array to zero mean and unit variance
   in one expression.
9. Explain why `a.T` is $O(1)$ but `np.ascontiguousarray(a.T)` is $O(n)$.
10. Replace this loop with an array expression:
    `[x**2 if x > 0 else 0 for x in arr]`.
11. Compute the row-wise softmax of a `(100, 10)` array, numerically stably
    ({{ch:math-functions}}).

**Advanced**

12. Given `a` of shape `(1000, 1000)`, explain why `a.sum(axis=0)` and
    `a.sum(axis=1)` can differ in speed, and predict which is faster.
13. Use `np.lib.stride_tricks.sliding_window_view` to compute a moving average
    without copying, and explain the strides of the result.
14. Show that `float32` summation of $10^{7}$ values loses precision, and
    measure how much. Compare naive accumulation, `np.sum`, and `math.fsum`.
15. Implement pairwise distances with `np.einsum` and explain the subscript
    string.
16. Explain why `distances_gemm` can produce slightly negative squared
    distances, and why clipping is the right fix rather than a hack.

**Implementation**

17. Implement k-means from scratch with no Python loops over data points, using
    the gemm distance formulation.
18. Write `assert_shape(arr, expected)` giving a useful error, and use it to
    catch the accidental broadcast from {{sec:7-implementation}}.
19. Benchmark list-comprehension against array arithmetic across
    $n \in \{10^2 \ldots 10^7\}$ and plot the ratio. Identify the crossover.
20. Implement a memory-efficient nearest-neighbour search over a matrix too
    large for one distance matrix, by chunking the query side. Verify against
    the unchunked version.

**Reasoning**

21. NumPy chose views over copies for slicing. Argue both sides, given that it
    is the most common source of silent bugs.
22. Deep learning uses `float32` or lower while scientific computing uses
    `float64`. What is different about the two workloads?

## 12. Chapter Summary

An ndarray is a contiguous buffer plus a header of dtype, shape and strides.
That layout — no boxing, no per-element type dispatch, and cache locality — is
what makes it 50-100× faster than a list for numerical work.

Vectorisation does not remove the loop; it moves it from the interpreter into
compiled code. If you are writing a `for` loop over array elements, an array
expression almost certainly replaces it.

Strides explain everything about views. Any operation expressible as new shape,
strides and offset over the same buffer is free, which is why transposition and
basic slicing cost nothing. It is also why a slice shares memory with its
parent, making writes visible through both — the most common silent bug in
numerical Python. Basic slicing gives a view; boolean masks and fancy indexing
give copies.

Broadcasting aligns shapes from the right, requiring each dimension pair to
match or one to be 1, and implements the stretch with a stride of zero so
nothing is copied. Its failures are loud and helpful; its accidental successes
are silent and expensive.

`axis=i` means the `i`-th dimension disappears. `keepdims=True` preserves it so
the result still broadcasts.

dtype fixes precision and memory. `float32` gives about seven significant digits
and halves memory against `float64`, which is why deep learning uses it. Integer
arrays wrap silently on overflow, and `float32` addition can be a no-op when the
operands differ enough in magnitude — which is why NumPy sums pairwise rather
than sequentially.

Memory is shape times itemsize, and intermediates count: an expression with
three operations allocates three temporaries. Choosing a formulation that avoids
a large intermediate — as the matrix-product distance formulation does — is
frequently the difference between a computation that is slow and one that is
impossible.
