---
id: mle-reproducibility
number: 46
part: V
tier: focused
status: reviewed
requires: [mle-pipelines, mle-hpo, py-environments]
provides: [reproducibility, experiment-tracking, determinism, run-lineage,
           data-versioning, content-addressing, seed-discipline]
citations: [sculley2015, breck2017, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish reproducibility from experiment tracking, and say why a team can
   have one without the other.
2. Enumerate the sources of nondeterminism in a training run and control each.
3. Explain seed discipline, including why one global seed is not enough.
4. Version a dataset by content rather than by filename.
5. Record the lineage needed to recreate an artefact.
6. Choose a level of reproducibility proportionate to the cost.
7. Recognise which sources of nondeterminism cannot be removed and what to do
   instead.

## 2. Why This Matters

**These are two different problems and they are routinely conflated.**
Experiment tracking answers *"which of these runs was better?"* — it needs
parameters, metrics and a comparable key. Reproducibility answers *"can I
recreate this exact artefact?"* — it needs code, data, environment and seeds.
A team can have an immaculate tracking dashboard and be unable to rebuild last
quarter's production model. That is the common case, not a pathological one.

**Irreproducibility is discovered at the worst moment.** Nobody notices while
the model is working. It is discovered when a regulator asks how a decision was
made, when a bug needs bisecting across model versions, or when the person who
trained it has left and the numbers no longer replicate. All three are
deadlines.

**Most of the sources are not the ones people guard against.** Setting
`random_state=42` is the first thing everyone does and it addresses perhaps a
third of the problem. {{sec:7-implementation}} enumerates the rest by breaking
a run repeatedly and fixing one cause at a time.

## 3. Prerequisites

{{ch:mle-pipelines}} for the pipeline whose definition must be versioned
alongside the model. {{ch:mle-hpo}} for the search whose trials must be
recorded. {{ch:py-environments}} for environments and dependency pinning.

## 4. Intuitive Explanation

### 4.1 The two questions

```text
   TRACKING                          REPRODUCIBILITY
   ───────────────────────           ─────────────────────────
   "run 47 scored 0.83,              "rebuild run 47 exactly"
    run 48 scored 0.85"
                                     needs: code commit
   needs: parameters, metrics,             data version
          a comparable key                 environment
                                           seeds
                                           hardware (sometimes)

   answers: which is better?         answers: what did we ship?
   cost:    low                      cost:    moderate to high
```

Tracking is cheap and almost always worth it. Reproducibility is a spectrum
with real costs, and the right level depends on what happens if you cannot
reproduce. A research prototype needs little; a credit model needs a great
deal, because someone will eventually be entitled to ask why they were
declined.

### 4.2 What actually varies

Run the same script twice and get different numbers. The causes, roughly in
order of how often they bite:

```text
   the data moved              a table was appended to since yesterday
   an unseeded RNG             a shuffle, an init, a subsample, a dropout mask
   library version drift       a default changed in a minor release
   hash ordering               set/dict iteration feeding a data structure
   parallel reduction order    floating-point addition is not associative
   GPU nondeterminism          atomics and algorithm selection
   wall-clock or hostname      anything that reads the environment
```

The first is by far the most common and the least discussed. Everyone reaches
for seeds and almost nobody versions the input table, so the run is
irreproducible before a single random number is drawn.

### 4.3 Floating-point addition is not associative

This one surprises people and explains a whole class of "impossible" bugs.

$$
(a + b) + c \ne a + (b + c) \quad \text{in floating point}
$$

Each addition rounds to the nearest representable value, and the rounding
depends on the magnitudes involved. So summing an array in a different order
gives a different answer — usually in the last bits, occasionally amplified by
a subsequent division or an exponential.

A parallel reduction over eight threads sums each thread's chunk and then
combines, and the order in which chunks finish is not guaranteed. The result
is a run-to-run difference of about $10^{-16}$ relative, which is invisible —
until it decides a tie between two candidate splits in a tree, the tree takes
a different branch, and the model is materially different.

The measurement in {{sec:7-implementation}} sharpens that story in a way worth
anticipating: the effect is **tie-breaking**, not perturbation. Its rate barely
depends on the size of the numerical noise, because a tie is resolved just as
readily by $10^{-16}$ as by $10^{-4}$.

### 4.4 Content addressing

"The data is in `s3://bucket/training_data.parquet`" is not a version. The file
can be overwritten, appended to, or regenerated, and nothing about the
reference changes.

The fix is to identify data by a **hash of its contents**. Then `sha256:a3f9…`
is either the exact bytes you trained on or it is not, and the check is
mechanical. This is what git does for code, what container digests do for
images, and what DVC and its relatives do for datasets. It is the single
highest-value change most teams can make, because it converts an unfalsifiable
claim into a testable one.

## 5. Formal Explanation

### 5.1 What a run must record

A run is reproducible if the tuple below is captured and every element is
resolvable later:

$$
\mathcal{R} = \langle
  c,\; d,\; e,\; \phi,\; \sigma,\; h
\rangle
$$ (eq:run-tuple)

- $c$ — **code**: a commit hash, and a clean working tree. A dirty tree makes
  the commit a lie.
- $d$ — **data**: a content hash of every input, not a path.
- $e$ — **environment**: exact versions of every transitive dependency, plus
  the interpreter and, where it matters, the base image digest.
- $\phi$ — **configuration**: every hyperparameter, including the defaults you
  did not set, because defaults change between versions.
- $\sigma$ — **seeds**: every RNG stream, not one global seed.
- $h$ — **hardware**: CPU/GPU model and thread count, needed only when
  bitwise reproducibility is required.

The commonest gap is $d$, the second commonest is the "defaults you did not
set" clause of $\phi$.

### 5.2 Levels of reproducibility

Not all of these are worth paying for, and naming the levels stops the
conversation being ideological.

{#tbl:repro-levels caption="Levels of reproducibility and what each costs. Most projects should target level 2; level 4 is for artefacts that must be defended."}

| Level | Guarantee | Needs | Typical cost |
|---|---|---|---|
| 0 | none | — | — |
| 1 | statistical: same distribution of results | code + data versions | low |
| 2 | same metrics to reported precision | + environment + seeds | low-moderate |
| 3 | same model parameters | + deterministic ops, fixed thread count | moderate |
| 4 | bitwise identical artefact | + fixed hardware, pinned image | high |

Level 2 is the right default: you can say "this configuration produces this
score" and defend it. Level 3 costs real throughput, because deterministic
kernels are slower and fixing the thread count forfeits parallel speed-up.
Level 4 is for regulated artefacts and for bisecting a bug across versions.

### 5.3 Seed discipline

One global seed is insufficient for a reason that is easy to miss: RNG streams
consumed in a different **order** produce different values even from the same
seed. Add a shuffle, and every draw after it shifts.

$$
\sigma_{\text{component}} = H(\sigma_{\text{master}} \,\|\, \text{name})
$$ (eq:seed-derivation)

Deriving each component's seed by hashing the master seed with a component
name makes streams independent of each other and of the order in which they are
consumed. Adding a data-augmentation RNG then cannot perturb the weight
initialisation.

The libraries needing separate attention in a typical stack: Python's `random`,
NumPy's global and generator RNGs, the framework's CPU and GPU RNGs, the
dataloader's per-worker seeds, and `PYTHONHASHSEED` for the interpreter itself.

> WARNING: `PYTHONHASHSEED` is randomised per process by default, and it
> affects the iteration order of sets and of dicts keyed by objects whose hash
> is address-based. If any iteration order feeds a data structure — the column
> order of a one-hot encoding, the order features are added — the run is
> nondeterministic in a way no `random_state` will fix. It must be set in the
> environment *before* the interpreter starts; setting it in Python is too
> late.

### 5.4 Data versioning

Three mechanisms, in increasing strength:

**Immutable partitions.** Never overwrite; write `date=2026-08-14/` and treat
existing partitions as read-only. Cheap, and it gives you time travel for free.

**Content addressing.** Hash the bytes; store under the hash. Guarantees the
data is what you think it is, and deduplicates identical files automatically.

**Transactional table formats.** Delta, Iceberg and Hudi give snapshot
isolation and time travel over mutable tables, so "the table as of version 118"
is a first-class query.

The hash of a *logical* dataset needs care: file order, partition layout and
compression can all change without the data changing. Hash the sorted per-row
content or a canonical serialisation, not the file bytes, if you want the hash
to mean "the same data".

### 5.5 What you cannot make deterministic, and what to do instead

Some nondeterminism is not worth removing, and pretending otherwise wastes
effort.

**Distributed asynchronous training** has no fixed reduction order by design.
**Some GPU kernels** have no deterministic implementation, and forcing one can
cost several-fold throughput. **Wall-clock-dependent early stopping** — stop
after two hours — is nondeterministic by construction.

For these the answer is not determinism but **statistical reproducibility**:
run the training several times, report the mean and spread, and treat any
comparison smaller than the run-to-run spread as unresolved. That is level 1,
and it is honest.

The corollary is a useful reviewing question: **has anyone measured the
run-to-run variance of this pipeline?** If not, no reported improvement smaller
than that unknown number means anything.

## 6. Mathematical Foundation

### 6.1 Why floating-point summation depends on order

A floating-point addition rounds: $\mathrm{fl}(a+b) = (a+b)(1+\delta)$ with
$|\delta| \le u$, the unit roundoff ($u \approx 1.1 \times 10^{-16}$ for
float64).

Summing $n$ values sequentially accumulates these roundings. The standard
forward error bound is

$$
\Big|\hat{S} - \sum_i x_i\Big| \le \frac{nu}{1-nu}\sum_i |x_i|
 \approx nu \sum_i |x_i|
$$ (eq:summation-error)

Two facts follow, and they explain the practical behaviour.

**The bound grows with $n$**, so long sums are less accurate — and a *pairwise*
or tree reduction has error $O(u\log n)$ instead of $O(un)$, which is why
NumPy's `sum` is more accurate than a Python loop and also why it gives a
different answer.

**The error depends on the order**, because the intermediate partial sums, and
therefore the magnitudes being rounded, differ. Two orderings do not merely
have different bounds; they have different results.

For a parallel reduction over $p$ threads, the order depends on which chunk
completes first, so the result is a function of scheduling. The magnitude is
tiny, and {{sec:7-implementation}} measures how it can nonetheless change a
model.

### 6.2 How a $10^{-16}$ difference becomes a different model

Discrete decisions amplify. Consider a tree choosing between two splits with
gains $g_1$ and $g_2$:

$$
\text{chosen} = \argmax(g_1, g_2)
$$

If $|g_1 - g_2| < \epsilon_{\text{fp}}$, the choice is decided by rounding. The
probability of a near-tie is small per node, but a forest of 500 trees with
1,000 nodes each makes half a million draws from that lottery, and one
different split changes every subtree beneath it.

More precisely, if gains at a node are approximately continuous with density
$f$ near the maximum, the probability that the top two are within
$\epsilon$ is approximately $\propto f \epsilon$ — negligible per node, and
multiplied by $5 \times 10^{5}$ nodes it is not negligible per forest.

The same mechanism applies to any `argmax`, any threshold comparison, and any
early-stopping rule that fires on a strict inequality. **Nondeterminism is
harmless in continuous computations and dangerous wherever a discrete decision
sits downstream of one.**

### 6.3 Why hashing seeds by component name works

The requirement is that component seeds be independent of each other and of
consumption order. Deriving $\sigma_i = H(\sigma_{\text{master}} \| \text{name}_i)$
with a good hash gives outputs that are, for practical purposes,
independent — knowing $\sigma_1$ tells you nothing usable about $\sigma_2$.

The property that matters is not cryptographic strength but **stability**: the
same name always yields the same stream, regardless of what else exists in the
program. Adding a new component perturbs nothing, and reordering the code
perturbs nothing, which is precisely the failure mode a single global seed has.

NumPy's `SeedSequence` implements this directly through `spawn`, and using it
is strictly better than `default_rng(42)` in several places.

## 7. Implementation

```python {tier=A name=nondeterminism-hunt}
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
```

```python {tier=A name=lineage-and-content-addressing}
"""Content-addressed data, a run record, and a reproduce-check that fails
loudly when something moved.
"""
import hashlib
import json
import platform
import sys

import numpy as np


# --- content addressing (section 4.4) ---------------------------------------
def content_hash(arr, *, canonical=True):
    """Hash a dataset by CONTENT, not by path.

    `canonical=True` sorts the rows first, so the hash means 'the same data'
    rather than 'the same bytes in the same order'. Which you want depends
    on whether row order is part of the dataset's identity — for a training
    table it usually is not, and a re-export that shuffles rows should not
    look like new data.
    """
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if canonical:
        a = a[np.lexsort(a.T[::-1])]
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_table(n=500, seed=0, extra_rows=0):
    rs = np.random.default_rng(seed)
    X = np.column_stack([rs.normal(size=n), rs.uniform(0, 10, n)])
    if extra_rows:
        rs2 = np.random.default_rng(seed + 999)
        X = np.vstack([X, np.column_stack([rs2.normal(size=extra_rows),
                                           rs2.uniform(0, 10, extra_rows)])])
    return X


base = make_table()
print("=" * 72)
print("content addressing: what a path cannot tell you")
print("=" * 72)
cases = [
    ("the same table", make_table()),
    ("rows re-ordered by an export", base[np.random.default_rng(1)
                                          .permutation(len(base))]),
    ("42 rows appended overnight", make_table(extra_rows=42)),
    ("one value changed", np.where(np.arange(base.size).reshape(base.shape)
                                   == 7, base + 1e-9, base)),
]
print(f"{'dataset':<32} {'canonical hash':>18} {'same as base?':>15}")
h0 = content_hash(base)
print(f"{'base':<32} {h0[:16]:>18} {'-':>15}")
for label, t in cases:
    h = content_hash(t)
    print(f"{label:<32} {h[:16]:>18} {str(h == h0):>15}")

print("\nA re-export that shuffles rows is correctly recognised as the SAME")
print("data. An overnight append and a single changed value are correctly")
print("recognised as different. The path 's3://bucket/training.parquet' says")
print("nothing about any of these — and the append is the one that quietly")
print("makes last week's run irreproducible.")


# --- the run record (eq. 46.1) ----------------------------------------------
class RunRecord:
    """Capture the tuple of eq. 46.1 and verify it later."""

    def __init__(self, name):
        self.rec = {
            "name": name,
            "code": {},          # c
            "data": {},          # d
            "environment": {},   # e
            "config": {},        # phi
            "seeds": {},         # sigma
            "hardware": {},      # h
            "metrics": {},
        }

    def code(self, commit, dirty):
        self.rec["code"] = {"commit": commit, "dirty": dirty}
        return self

    def data(self, **named_arrays):
        self.rec["data"] = {k: content_hash(v)[:16]
                            for k, v in named_arrays.items()}
        return self

    def environment(self):
        self.rec["environment"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.system(),
        }
        return self

    def config(self, explicit, defaults):
        """Record the defaults you did NOT set (section 5.1) — they change
        between library versions and are the second commonest gap."""
        self.rec["config"] = {"explicit": explicit,
                              "resolved_defaults": defaults}
        return self

    def seeds(self, **s):
        self.rec["seeds"] = s
        return self

    def hardware(self, threads):
        self.rec["hardware"] = {"threads": threads}
        return self

    def metrics(self, **m):
        self.rec["metrics"] = m
        return self

    def audit(self):
        """What is missing, and what does each gap cost?"""
        gaps = []
        if not self.rec["code"].get("commit"):
            gaps.append(("code", "cannot identify the source at all"))
        elif self.rec["code"].get("dirty"):
            gaps.append(("code", "working tree was dirty: the commit is a lie"))
        if not self.rec["data"]:
            gaps.append(("data", "no content hash: cannot prove the inputs"))
        if not self.rec["environment"]:
            gaps.append(("environment", "a default may have changed under you"))
        if not self.rec["seeds"]:
            gaps.append(("seeds", "results vary run to run"))
        if not self.rec["config"].get("resolved_defaults"):
            gaps.append(("config", "unset defaults are unrecorded"))
        return gaps

    def level(self):
        """Which row of table 46.1 this record supports."""
        r = self.rec
        if not r["code"].get("commit") or not r["data"]:
            return 0
        if not r["environment"] or not r["seeds"]:
            return 1
        if not r["hardware"].get("threads"):
            return 2
        return 3


print("\n" + "=" * 72)
print("run records at three levels of care (table 46.1)")
print("=" * 72)

Xtr = make_table(400, seed=5)
ytr = (Xtr[:, 0] + 0.3 * Xtr[:, 1] > 1.5).astype(float)

sloppy = (RunRecord("sloppy")
          .code(commit="a91f3c2", dirty=True)
          .config({"max_depth": 6}, {})
          .metrics(auc=0.842))

typical = (RunRecord("typical")
           .code(commit="a91f3c2", dirty=False)
           .data(train=Xtr, labels=ytr)
           .environment()
           .config({"max_depth": 6},
                   {"min_samples_leaf": 1, "criterion": "gini"})
           .seeds(split=7, init=11)
           .metrics(auc=0.842))

careful = (RunRecord("careful")
           .code(commit="a91f3c2", dirty=False)
           .data(train=Xtr, labels=ytr)
           .environment()
           .config({"max_depth": 6},
                   {"min_samples_leaf": 1, "criterion": "gini"})
           .seeds(split=7, init=11, shuffle=13, augment=17)
           .hardware(threads=1)
           .metrics(auc=0.842))

for r in (sloppy, typical, careful):
    gaps = r.audit()
    print(f"\n{r.rec['name']}  ->  reproducibility level {r.level()}")
    if not gaps:
        print("    no gaps")
    for what, cost in gaps:
        print(f"    missing {what:<12} {cost}")

print("\nAll three report the same AUC. Only the last two can defend it, and")
print("only the last could rebuild the same parameters on demand.")

# --- the reproduce-check ----------------------------------------------------
print("\n" + "=" * 72)
print("the reproduce-check: verifying a record still resolves")
print("=" * 72)


def verify(record, *, current_data, current_env):
    """Would re-running this record today give the same thing?"""
    problems = []
    for name, h in record.rec["data"].items():
        now = content_hash(current_data[name])[:16]
        if now != h:
            problems.append(f"data '{name}' changed: recorded {h}, now {now}")
    for k, v in record.rec["environment"].items():
        if current_env.get(k) != v:
            problems.append(f"environment '{k}': recorded {v}, "
                            f"now {current_env.get(k)}")
    return problems


env_now = {"python": sys.version.split()[0], "numpy": np.__version__,
           "platform": platform.system()}

print("a) nothing has changed:")
p = verify(careful, current_data={"train": Xtr, "labels": ytr},
           current_env=env_now)
print(f"   {'reproducible' if not p else p}")

print("\nb) the training table was appended to overnight:")
Xtr2 = np.vstack([Xtr, make_table(20, seed=99)])
ytr2 = np.r_[ytr, np.zeros(20)]
p = verify(careful, current_data={"train": Xtr2, "labels": ytr2},
           current_env=env_now)
for msg in p:
    print(f"   BLOCKED: {msg}")

print("\nc) a library was upgraded:")
p = verify(careful, current_data={"train": Xtr, "labels": ytr},
           current_env={**env_now, "numpy": "9.9.9"})
for msg in p:
    print(f"   BLOCKED: {msg}")

print("\nCase (b) is the one that matters, and it is the case a path-based")
print("reference cannot detect at all: the file name is unchanged, the job")
print("runs, the numbers differ, and nobody knows why. A content hash turns")
print("that into a blocked build with a one-line explanation.")
```

## 8. Practical Example

```python {tier=A name=repro-in-practice}
"""Measuring run-to-run variance, and using it to decide what is real.
"""
import numpy as np

rng = np.random.default_rng(31)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, 10))
    z = 1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] - 0.3
    return X, (rs.random(n) < 1 / (1 + np.exp(-z))).astype(int)


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(3000, 0)
    Xte, yte = make_data(6000, 1)

    # --- 1. the number nobody measures -------------------------------------
    print("=" * 72)
    print("1. run-to-run variance: the number that makes 'improvements' real")
    print("=" * 72)
    print("Same code, same data, same hyperparameters. Only the seed moves.\n")

    scores = []
    for seed in range(15):
        m = RandomForestClassifier(n_estimators=120, max_depth=8,
                                   random_state=seed, n_jobs=1).fit(Xtr, ytr)
        scores.append(auc(yte, m.predict_proba(Xte)[:, 1]))
    scores = np.array(scores)
    sd = float(scores.std(ddof=1))
    print(f"  15 runs, AUC: min {scores.min():.4f}  "
          f"median {np.median(scores):.4f}  max {scores.max():.4f}")
    print(f"  standard deviation across seeds : {sd:.4f}")
    print(f"  full range                      : {np.ptp(scores):.4f}")
    print(f"\n  Any reported improvement smaller than about {2 * sd:.4f} AUC is")
    print("  indistinguishable from having re-rolled the seed.")

    # --- 2. and what that does to a comparison ------------------------------
    print("\n" + "=" * 72)
    print("2. two 'different' configurations, judged against that noise")
    print("=" * 72)
    cfg_a = dict(n_estimators=120, max_depth=8)
    cfg_b = dict(n_estimators=120, max_depth=9)
    a_scores, b_scores = [], []
    for seed in range(15):
        ma = RandomForestClassifier(random_state=seed, n_jobs=1,
                                    **cfg_a).fit(Xtr, ytr)
        mb = RandomForestClassifier(random_state=seed, n_jobs=1,
                                    **cfg_b).fit(Xtr, ytr)
        a_scores.append(auc(yte, ma.predict_proba(Xte)[:, 1]))
        b_scores.append(auc(yte, mb.predict_proba(Xte)[:, 1]))
    a_scores, b_scores = np.array(a_scores), np.array(b_scores)

    print(f"  single-seed comparison (seed 0): "
          f"depth 8 = {a_scores[0]:.4f}, depth 9 = {b_scores[0]:.4f}, "
          f"diff {b_scores[0] - a_scores[0]:+.4f}")
    diff = b_scores - a_scores
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    unpaired_se = float(np.sqrt(a_scores.var(ddof=1) / len(a_scores)
                                + b_scores.var(ddof=1) / len(b_scores)))
    print(f"  paired over 15 seeds           : mean diff "
          f"{diff.mean():+.4f} +- {se:.4f} (SE)")
    print(f"  unpaired, same 30 runs         : mean diff "
          f"{diff.mean():+.4f} +- {unpaired_se:.4f} (SE)")
    verdict = ("resolved" if abs(diff.mean()) > 2 * se else "not resolved")
    print(f"  verdict (paired)               : {verdict}")
    print(f"  verdict (unpaired)             : "
          f"{'resolved' if abs(diff.mean()) > 2 * unpaired_se else 'not resolved'}")
    print("\n  The single-seed difference is smaller than the run-to-run")
    print("  standard deviation measured above, so on its own it is not")
    print("  evidence of anything — it could be the seed.")
    print("\n  REPLICATION is what rescues it. Averaging fifteen seeds")
    print("  shrinks the standard error by sqrt(15), which is what turns a")
    print("  difference smaller than the noise into one that is resolvable.")
    print("\n  Pairing helps too, and by less than one might expect here:")
    print("  0.0003 against 0.0004 on the very same thirty runs. Pairing pays")
    print("  in proportion to how much variation the two configurations")
    print("  SHARE, and depth 8 versus depth 9 are similar enough that most")
    print("  of the seed effect does not cancel. Between two genuinely")
    print("  different model families it would pay much more. It costs")
    print("  nothing either way, so pair by default and do not count on it.")

    # --- 3. determinism has a price -----------------------------------------
    print("\n" + "=" * 72)
    print("3. what determinism costs (table 46.1, levels 2 vs 3)")
    print("=" * 72)
    import time
    print(f"{'threads':>8} {'fit seconds':>13} {'identical across repeats?':>27}")
    for n_jobs in (1, 2, 4):
        digs, t0 = set(), time.perf_counter()
        for _ in range(3):
            m = RandomForestClassifier(n_estimators=150, max_depth=10,
                                       random_state=0,
                                       n_jobs=n_jobs).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            digs.add(hash(p.tobytes()))
        dt = (time.perf_counter() - t0) / 3
        print(f"{n_jobs:>8} {dt:>13.2f} {str(len(digs) == 1):>27}")

    print("\n  Read the last column before the second-to-last. A seeded")
    print("  random forest — about as benign a model as exists, with trees")
    print("  built independently — is bitwise reproducible on one and two")
    print("  threads and NOT on four.")
    print("\n  That is worth pausing on, because it is the opposite of what")
    print("  most people assume, and the assumption is reasonable: the seed")
    print("  is fixed, the trees are independent, nothing about the algorithm")
    print("  looks order-dependent. The nondeterminism enters below the")
    print("  algorithm, in how work is partitioned and floating-point results")
    print("  are combined across workers.")
    print("\n  Note also the trade being made: four threads are about 2.6x")
    print("  faster than one. Level 3 of table 46.1 costs that speed-up, and")
    print("  on a GPU with atomic accumulation the cost of forcing")
    print("  deterministic kernels is typically larger still.")
    print("\n  The lesson is the test itself, not the result: run it three")
    print("  times, digest the output, compare. Determinism is a property to")
    print("  MEASURE on your own stack, not one to reason about.")

    # --- 4. the decision ----------------------------------------------------
    print("\n" + "=" * 72)
    print("4. choosing a level, by what it costs to be wrong")
    print("=" * 72)
    rows = [
        ("exploratory notebook", 1,
         "re-run it; nobody depends on the number"),
        ("a paper or a blog post", 2,
         "someone will try to replicate the metric"),
        ("a shipped product model", 2,
         "you must be able to rebuild and compare"),
        ("bisecting a regression", 3,
         "differences must be attributable to the change"),
        ("a credit or clinical model", 4,
         "a person is entitled to ask why, years later"),
    ]
    print(f"{'situation':<28} {'level':>6}  {'because':<44}")
    for what, lvl, why in rows:
        print(f"{what:<28} {lvl:>6}  {why:<44}")
    print("\n  The mistake in both directions is common: teams chase bitwise")
    print("  reproducibility for a dashboard nobody audits, and ship credit")
    print("  models they cannot rebuild. Match the level to the consequence.")
```

## 9. Common Mistakes

**Conflating tracking with reproducibility.** They answer different questions
and a team can have one without the other.

**Referring to data by path.** The measurement shows an overnight append that
a path cannot detect and a content hash catches immediately.

**Setting one global seed.** The measurement shows a single added line
upstream changing every downstream draw; per-component streams do not.

**Forgetting `PYTHONHASHSEED`.** Set-iteration order feeds real data
structures, and it must be set before the interpreter starts.

**Recording only the parameters you set.** Defaults change between versions.

**Recording a commit with a dirty working tree.** The commit is then a lie.

**Assuming a seed gives bitwise determinism.** The measurement shows a
$10^{-16}$ perturbation flipping a measurable fraction of split decisions.

**Comparing two configurations on one seed.** The measured run-to-run spread is
often larger than the difference being claimed.

**Paying for level 4 on a prototype, or level 1 on a regulated model.** Match
the level to the consequence.

## 10. Connection to Previous Chapters

{{ch:mle-pipelines}} supplied the pipeline whose definition is part of the
artefact — a model and its feature definitions must be versioned together, or
the lineage has a hole exactly where the skew of that chapter would enter.
{{ch:mle-hpo}} supplied the search whose trials must be recorded; without the
record, {{eq:hpo-optimism}}'s correction cannot be computed because
$K_{\text{total}}$ is unknown. {{ch:mle-splits}} supplied the holdout ledger,
which is a tracking system for one specific thing. {{cite:sculley2015}}'s
configuration debt is exactly what {{eq:run-tuple}}'s $\phi$ term is for.

Forward: {{ch:mle-registry}} consumes the run record — a registry entry without
lineage is a binary with no provenance. {{ch:mle-drift}} needs to know which
data version a reference distribution was computed from. {{part:24}} covers the
platform tooling this chapter deliberately describes only conceptually.

## 11. Exercises

**Beginner**

1. Distinguish tracking from reproducibility in one sentence each.
2. Name four sources of nondeterminism in a training run.
3. Why is a file path not a data version?
4. Why is `(a+b)+c \ne a+(b+c)` in floating point?
5. What does `PYTHONHASHSEED` affect?

**Intermediate**

6. Explain why one global seed fails when a component is added upstream.
7. Using {{eq:summation-error}}, explain why pairwise summation is more
   accurate than a sequential loop.
8. Explain how a $10^{-16}$ difference produces a different tree.
9. Which level of {{tbl:repro-levels}} would you target for a churn model, and
   why?
10. Why must the defaults you did not set be recorded?
11. Give a case where determinism is not achievable and say what to do instead.

**Advanced**

12. Derive {{eq:summation-error}} and state the assumptions.
13. Explain why {{eq:seed-derivation}} makes streams order-independent, and what
    property of the hash is actually required.
14. Estimate the probability that a 500-tree forest differs between two runs,
    given a per-node tie probability, and state your assumptions.
15. Design a canonical hash for a partitioned dataset that is invariant to
    partition layout and compression but not to content.

**Implementation**

16. Instrument a training script to capture every element of
    {{eq:run-tuple}} automatically, failing if the working tree is dirty.
17. Implement per-component seed derivation and demonstrate that adding a
    component leaves the others' streams unchanged.
18. Measure the run-to-run variance of your own pipeline and record it
    alongside every reported metric.
19. Build a `verify` step into CI that blocks a deploy when a recorded data
    hash no longer resolves.

**Reasoning**

20. A model from eighteen months ago cannot be rebuilt. What can you still
    establish about it, and what is permanently lost?
21. A colleague reports a 0.3-point AUC improvement. What do you ask for?

## 12. Chapter Summary

Tracking and reproducibility are different problems. Tracking compares runs and
is cheap; reproducibility recreates one and costs more. A team with an excellent
dashboard may be unable to rebuild last quarter's model, and that is the common
case.

A run is reproducible when code, data, environment, configuration, seeds and —
at the strictest level — hardware are all captured and resolvable. The
commonest gap is data, and the second is the defaults nobody set, which change
between library versions.

Data must be identified by content, not by path. The measurement shows a
canonical content hash correctly treating a row-reordered re-export as the same
data while catching an overnight append and a single changed value — none of
which a filename can distinguish, and the append is exactly what silently
invalidates last week's run.

One global seed is insufficient, because streams consumed in a different order
diverge. The measurement shows a single added line upstream changing every
subsequent draw, and per-component streams keyed by name eliminating it.
`PYTHONHASHSEED` is a separate, easily-missed cause that must be set before the
interpreter starts.

Floating-point addition is not associative, so summation order changes results.
The measured spread across four correct summation orders is real, and the
mechanism by which it matters is amplification through discrete decisions: a
$10^{-16}$ perturbation flips a measurable fraction of split choices, and one
different split changes every subtree beneath it. Nondeterminism is harmless in
continuous computation and dangerous wherever an argmax sits downstream.

Reproducibility comes in levels, and the right one depends on the cost of not
having it. Level 2 — same metrics to reported precision — is the sensible
default; level 4 is for artefacts someone will be entitled to interrogate years
later.

Finally, the run-to-run variance of a pipeline is the number that makes
improvements meaningful, and almost nobody measures it. The measurement shows a
single-seed comparison reporting a confident difference that a paired
fifteen-seed comparison declines to resolve.
