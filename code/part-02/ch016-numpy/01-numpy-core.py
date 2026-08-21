# Extracted from: Chapter 16 — NumPy: Arrays, Broadcasting, and Vectorized Computation
# Source: src/.../ch016-numpy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
