# Extracted from: Chapter 46 — Reproducibility, Experiment Tracking, and Versioning
# Source: src/.../ch046-reproducibility.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Break a run's reproducibility, then fix one cause at a time.
"""
import hashlib
import os
import sys

import numpy as np

# --- the run under test -----------------------------------------------------
FEATURES = {"amount", "tenure", "n_txn", "region"}     # a SET, deliberately


def build_matrix(data, feature_names):
    """Assemble a design matrix by iterating a set of feature names.

    The bug: set iteration order is not guaranteed across processes when
    PYTHONHASHSEED is randomised, so the COLUMNS come out in a different
    order — and any model that is not permutation-invariant sees different
    data. Nothing errors.
    """
    return np.column_stack([data[k] for k in feature_names])


def make_data(n=600, seed=0):
    rs = np.random.default_rng(seed)
    return {
        "amount": rs.lognormal(4.0, 0.6, n),
        "tenure": rs.uniform(0, 120, n),
        "n_txn": rs.poisson(6, n).astype(float),
        "region": rs.integers(0, 5, n).astype(float),
    }


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64)
                          .tobytes()).hexdigest()[:12]


print("=" * 72)
print("source 1: iteration order of an unordered collection")
print("=" * 72)
data = make_data()
print(f"PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED', '<unset>')}")
print(f"set iteration order in THIS process: {list(FEATURES)}")
m_set = build_matrix(data, FEATURES)
m_sorted = build_matrix(data, sorted(FEATURES))
print(f"  digest, set order    : {digest(m_set)}")
print(f"  digest, sorted order : {digest(m_sorted)}")
print(f"  identical            : {digest(m_set) == digest(m_sorted)}")
print("\nWithin one process a set's order is stable, so this reproduces")
print("perfectly all afternoon and differs tomorrow — or on a colleague's")
print("machine, or in CI. Sorting the names costs nothing and removes the")
print("whole class. If an ordering matters, make it explicit.")

# --- source 2: one global seed is not enough --------------------------------
print("\n" + "=" * 72)
print("source 2: one global seed, consumed in a different ORDER")
print("=" * 72)


def train_v1(seed):
    """Two components draw from one stream: init, then augmentation."""
    rs = np.random.default_rng(seed)
    init = rs.normal(size=4)
    aug = rs.normal(size=3)
    return init, aug


def train_v2(seed):
    """Someone adds a shuffle before the init. Nothing else changed."""
    rs = np.random.default_rng(seed)
    _shuffle = rs.permutation(10)            # the new line
    init = rs.normal(size=4)
    aug = rs.normal(size=3)
    return init, aug


i1, a1 = train_v1(42)
i2, a2 = train_v2(42)
print("same seed (42), one line added upstream:")
print(f"  v1 init: {np.round(i1, 4)}")
print(f"  v2 init: {np.round(i2, 4)}")
print(f"  identical: {np.allclose(i1, i2)}")

# the fix: derive an independent stream per component (eq. 46.2)
def streams(master, *names):
    """Eq. 46.2 via SeedSequence.spawn_key — each name gets a stable,
    independent stream that does not depend on consumption order."""
    ss = np.random.SeedSequence(master)
    return {n: np.random.default_rng(
        np.random.SeedSequence(entropy=master,
                               spawn_key=(int.from_bytes(
                                   hashlib.sha256(n.encode()).digest()[:4],
                                   "big"),)))
        for n in names}


def train_v3(seed, extra_shuffle=False):
    st = streams(seed, "init", "augment", "shuffle")
    if extra_shuffle:
        st["shuffle"].permutation(10)
    return st["init"].normal(size=4), st["augment"].normal(size=3)


i3a, a3a = train_v3(42, extra_shuffle=False)
i3b, a3b = train_v3(42, extra_shuffle=True)
print("\nwith per-component streams (eq. 46.2), same change:")
print(f"  without the shuffle: {np.round(i3a, 4)}")
print(f"  with the shuffle   : {np.round(i3b, 4)}")
print(f"  identical: {np.allclose(i3a, i3b)}")
print("\nAdding a component no longer perturbs the others. That is the whole")
print("point: streams are keyed by NAME, so they are independent of the")
print("order in which code happens to run.")

# --- source 3: floating-point summation order -------------------------------
print("\n" + "=" * 72)
print("source 3: floating-point addition is not associative (eq. 46.3)")
print("=" * 72)
rs = np.random.default_rng(7)
x = rs.normal(0, 1, 100_000) * rs.choice([1.0, 1e6], 100_000)

sequential = 0.0
for v in x:
    sequential += v
numpy_pairwise = float(np.sum(x))
shuffled = float(np.sum(rs.permutation(x)))
chunked = float(sum(np.sum(c) for c in np.array_split(x, 8)))

print(f"  Python loop, in order   : {sequential:.10f}")
print(f"  np.sum (pairwise)       : {numpy_pairwise:.10f}")
print(f"  np.sum of a permutation : {shuffled:.10f}")
print(f"  8 chunks summed         : {chunked:.10f}")
vals = [sequential, numpy_pairwise, shuffled, chunked]
print(f"\n  spread: {max(vals) - min(vals):.3e}  "
      f"(relative: {(max(vals) - min(vals)) / max(abs(np.mean(vals)), 1):.3e})")
print("\nFour correct ways to add the same numbers, four different answers.")
print("Eq. 46.3 says the error grows with n and depends on the order, and a")
print("parallel reduction chooses its order by whichever thread finishes")
print("first.")

# --- ...and how that becomes a different model (section 6.2) ----------------
print("\n" + "=" * 72)
print("how 1e-16 becomes a different model (section 6.2)")
print("=" * 72)


def best_split(x, y, jitter=0.0, return_gains=False):
    """Return the chosen threshold. Ties are broken by floating-point noise
    exactly as they are inside a real tree."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    cs = np.cumsum(ys)
    n = len(ys)
    k = np.arange(1, n)
    left = cs[:-1] / k
    right = (cs[-1] - cs[:-1]) / (n - k)
    gain = k * left ** 2 + (n - k) * right ** 2
    if jitter:
        # a RELATIVE perturbation of the size a different reduction order
        # would produce
        rsj = np.random.default_rng(abs(hash(gain.tobytes())) % (2 ** 32))
        gain = gain * (1.0 + jitter * rsj.standard_normal(gain.shape))
    if return_gains:
        return None, gain
    i = int(np.argmax(gain))
    return 0.5 * (xs[i] + xs[i + 1]), float(gain[i])


# (a) how close must two gains be before 1e-16 flips the choice?
rs = np.random.default_rng(3)
print("How large must a perturbation be before it changes which split is")
print("chosen? Feature values are continuous here, so exact ties are rare.\n")
print(f"{'relative perturbation':>22} {'nodes flipped (of 4,000)':>26}")
for jitter in (1e-16, 1e-12, 1e-8, 1e-4, 1e-2):
    rs = np.random.default_rng(3)
    n_flip = 0
    for t in range(4000):
        xv = rs.uniform(0, 1, 60)
        yv = (rs.random(60) < 0.5).astype(float)
        t0, _ = best_split(xv, yv, jitter=0.0)
        t1, _ = best_split(xv, yv, jitter=jitter)
        n_flip += (t0 != t1)
    print(f"{jitter:>22.0e} {n_flip:>26,}")

print("\nRead the shape of that column, which is not what one would guess.")
print("A perturbation at MACHINE EPSILON already flips about 1% of nodes,")
print("and the rate barely moves as the perturbation grows through eight")
print("orders of magnitude — 38, 67, 67, 72 — before finally jumping at")
print("1e-2.")
print("\nThat flatness is the tell. Those flips are not the perturbation")
print("overpowering a genuine gap; they are near-exact ties being resolved")
print("differently, and a tie is broken just as effectively by 1e-16 as by")
print("1e-4. Only at 1e-2 does the noise become large enough to cross real")
print("gaps between distinguishable candidates, and the count jumps")
print("twenty-fold.")
print("\nSo floating-point nondeterminism is a TIE-BREAKING mechanism, not a")
print("perturbation mechanism, and its effect does not scale with its size.")

# (b) ...but exact ties are common, and then 1e-16 is enough
print("\n" + "=" * 72)
print("...unless there are EXACT ties, which coarse features produce")
print("constantly")
print("=" * 72)
rs = np.random.default_rng(5)
n_flip, n_tied, n_trials = 0, 0, 4000
for t in range(n_trials):
    xv = rs.integers(0, 4, 40).astype(float)     # 4 distinct values
    yv = (rs.random(40) < 0.5).astype(float)
    t0, g0 = best_split(xv, yv, jitter=0.0)
    # is the best gain achieved more than once, to machine precision?
    _, gains = best_split(xv, yv, jitter=0.0, return_gains=True)
    n_tied += int(np.sum(np.abs(gains - gains.max())
                         <= 1e-15 * abs(gains.max())) > 1)
    t1, _ = best_split(xv, yv, jitter=1e-16)
    n_flip += (t0 != t1)

print(f"features take 4 distinct values, {n_trials:,} nodes:")
print(f"  nodes with an EXACT tie for best gain : {n_tied:,} "
      f"({n_tied / n_trials:.1%})")
print(f"  nodes flipped by a 1e-16 perturbation : {n_flip:,} "
      f"({n_flip / n_trials:.2%})")
print(f"  expected flips in a 500-tree forest with 1,000 nodes each:")
print(f"    {n_flip / n_trials * 500 * 1000:,.0f}")

print("\nCoarse features make ties common rather than accidental: 3.3% of")
print("nodes here have two or more splits tied to machine precision, and")
print("about 1% flip under a 1e-16 perturbation. Exact ties appear wherever")
print("a feature is not continuous — categorical codes, counts, binned")
print("values, one-hot columns, anything integer-valued — which is most")
print("tabular data.")
print("\nOne flipped split changes every subtree beneath it, and the")
print("expected count above is not small. This is why bitwise")
print("reproducibility (level 3+) needs the thread count and reduction order")
print("fixed, not merely the seeds.")

print("\nThe general rule from section 6.2: floating-point nondeterminism is")
print("harmless in continuous computations and dangerous wherever a discrete")
print("decision — an argmax, a threshold, an early-stopping test — sits")
print("downstream of one.")
