# Extracted from: Chapter 42 — Anomaly Detection Methods
# Source: src/.../ch042-anomaly.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The application every deployed model needs: monitoring its own inputs.
"""
import numpy as np

rng = np.random.default_rng(31)


def c_factor(n):
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def build_itree(X, depth, max_depth, rs):
    n = len(X)
    if depth >= max_depth or n <= 1:
        return {"size": n}
    j = int(rs.integers(0, X.shape[1]))
    lo, hi = X[:, j].min(), X[:, j].max()
    if hi - lo < 1e-12:
        return {"size": n}
    p = float(rs.uniform(lo, hi))
    m = X[:, j] < p
    if m.all() or (~m).all():
        return {"size": n}
    return {"f": j, "p": p, "l": build_itree(X[m], depth + 1, max_depth, rs),
            "r": build_itree(X[~m], depth + 1, max_depth, rs)}


def path_length(node, x):
    d = 0
    while "f" in node:
        node = node["l"] if x[node["f"]] < node["p"] else node["r"]
        d += 1
    return d + c_factor(node["size"])


class IForest:
    """NOVELTY detection: fit on clean reference data, score new data."""

    def __init__(self, n_trees=120, sample_size=256, seed=0):
        self.n_trees, self.m, self.seed = n_trees, sample_size, seed

    def fit(self, X):
        rs = np.random.default_rng(self.seed)
        m = min(self.m, len(X))
        self.c = c_factor(m)
        depth = int(np.ceil(np.log2(max(m, 2))))
        self.trees = [build_itree(X[rs.choice(len(X), m, replace=False)],
                                  0, depth, rs) for _ in range(self.n_trees)]
        return self

    def score(self, X):
        h = np.array([[path_length(t, x) for t in self.trees] for x in X])
        return 2.0 ** (-h.mean(1) / max(self.c, 1e-12))


# --- the reference distribution the model was trained on --------------------
def make_reference(n):
    age = rng.normal(42, 12, n)
    income = rng.lognormal(10.4, 0.5, n)
    tenure = rng.exponential(4.0, n)
    n_txn = rng.poisson(14, n)
    # a STRONG correlation, as real feature sets have: income tracks age
    income = np.exp(9.4 + 0.030 * age + rng.normal(0, 0.22, n))
    return np.column_stack([age, np.log(income), tenure, n_txn])


NAMES = ["age", "log_income", "tenure", "n_txn"]
ref = make_reference(4000)
mu, sd = ref.mean(0), ref.std(0)
ref_s = (ref - mu) / sd

det = IForest(seed=3).fit(ref_s)
ref_scores = det.score(ref_s)
# calibrate the alarm on the REFERENCE data: flag the top 1%
THRESH = float(np.quantile(ref_scores, 0.99))
print(f"reference scores: mean {ref_scores.mean():.4f}, "
      f"99th percentile {THRESH:.4f}")
print(f"false-alarm rate on the reference itself: "
      f"{float((ref_scores > THRESH).mean()):.4f}  (1% by construction)\n")


# A second, complementary signal. Isolation forest finds points in SPARSE
# regions; Mahalanobis finds points that violate the correlation structure,
# including points sitting implausibly at the joint mode. They fail on
# different things, which is the reason to run both.
S_ref = np.cov(ref_s, rowvar=False) + 1e-6 * np.eye(ref_s.shape[1])
S_inv = np.linalg.inv(S_ref)


def maha(batch_s):
    d = batch_s - ref_s.mean(0)
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, S_inv, d), 0))


MAHA_THRESH = float(np.quantile(maha(ref_s), 0.99))


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def check(batch, label):
    bs = (batch - mu) / sd
    r_if = float((det.score(bs) > THRESH).mean())
    r_mh = float((maha(bs) > MAHA_THRESH).mean())
    ks = max(ks_stat(ref[:, j], batch[:, j]) for j in range(ref.shape[1]))
    fires = (r_if > 0.05, r_mh > 0.05, ks > 0.15)
    verdict = "ALARM" if any(fires) else "ok"
    print(f"{label:<40} {r_if:>9.3f} {r_mh:>12.3f} {ks:>8.3f}   {verdict}")
    return r_if, r_mh, ks


print("=" * 72)
print("monitoring the input distribution")
print("=" * 72)
print("Three signals with three different definitions of 'wrong'. The two")
print("flag rates are calibrated to 1% on the reference, so 0.01 is the")
print("no-drift baseline; the KS column is the largest per-feature")
print("two-sample statistic, where ~0.03 is the no-drift baseline.\n")
print(f"{'incoming batch':<40} {'iForest':>9} {'Mahalanobis':>12} "
      f"{'max KS':>8}")
print("-" * 78)

check(make_reference(1500), "same distribution (the control)")

# 1. a feature silently changes units
b = make_reference(1500)
b[:, 2] *= 30.0                       # tenure switched from years to months
check(b, "unit change: tenure years -> months")

# 2. a slow covariate shift
for shift in (0.25, 0.5, 1.0, 2.0):
    b = make_reference(1500)
    b[:, 0] += shift * 12             # the population ages
    check(b, f"covariate shift: mean age +{shift * 12:.0f} years")

# 3. an upstream bug fills a column with a constant
b = make_reference(1500)
b[:, 1] = np.log(35000.0)
check(b, "upstream bug: log_income constant")

# 4. the correlation breaks while every MARGINAL is preserved
b = make_reference(1500)
b[:, 1] = rng.permutation(b[:, 1])
check(b, "correlation broken, marginals identical")

# 5. missing values imputed with the mean
b = make_reference(1500)
idx = rng.choice(1500, 500, replace=False)
b[idx, 3] = ref[:, 3].mean()
check(b, "33% of n_txn mean-imputed upstream")

print("\nRead across the columns: each signal catches what the others miss,")
print("and the division of labour follows directly from what each one")
print("measures.")
print("\nThe ISOLATION FOREST scores points in SPARSE regions, so it fires")
print("on inputs that land where no reference data lives — the unit change,")
print("most obviously. It is nearly blind to the constant-income bug, and")
print("that is not a defect but a consequence: a column collapsed to one")
print("central value puts every row in the DENSEST region there is. A")
print("detector built to find isolated points cannot find")
print("over-concentrated ones.")
print("\nMAHALANOBIS measures departure from the joint structure, so it")
print("fires when the correlation is broken even though every marginal is")
print("untouched — a point ordinary in every column but wrong in the")
print("covariance is far from the centre in the whitened space (eq. 42.8).")
print("\nPER-FEATURE KS is the crudest of the three and the only one that")
print("reliably catches a collapsed or mean-imputed column, because those")
print("change a marginal DISTRIBUTION without moving anything into a sparse")
print("region or breaking a correlation.")
print("\nThe practical lesson is stronger than 'add a multivariate check':")
print("run several detectors with DIFFERENT definitions of anomalous,")
print("because each is blind to roughly what the others are built for.")

# --- what a per-feature monitor would have seen -----------------------------
print("\n" + "=" * 72)
print("what per-feature monitoring misses")
print("=" * 72)


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic — the usual drift check."""
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


broken = make_reference(1500)
broken[:, 1] = rng.permutation(broken[:, 1])
print(f"{'feature':<14} {'KS statistic vs reference':>27} {'verdict':<12}")
for j, nm in enumerate(NAMES):
    d = ks_stat(ref[:, j], broken[:, j])
    print(f"{nm:<14} {d:>27.4f} "
          f"{('drift' if d > 0.1 else 'no drift'):<12}")

bs = (broken - mu) / sd
print(f"\nisolation forest flag rate : "
      f"{float((det.score(bs) > THRESH).mean()):.4f}   (reference 0.0100)")
print(f"Mahalanobis flag rate      : "
      f"{float((maha(bs) > MAHA_THRESH).mean()):.4f}   (reference 0.0100)")
print("\nEvery per-feature KS test says 'no drift', and every one of them is")
print("CORRECT — the marginals really are unchanged. A wall of per-feature")
print("histograms would show nothing whatsoever.")
print("\nThe joint detector sees it, and note which joint detector: the one")
print("whose definition of anomalous is 'violates the correlation")
print("structure'. Input monitoring needs at least one multivariate check,")
print("and it needs the right kind.")

# --- the honest limits ------------------------------------------------------
print("\n" + "=" * 72)
print("what this does NOT tell you")
print("=" * 72)
b = make_reference(1500)
b_shift = b.copy()
b_shift[:, 0] += 2.0                  # a small age shift: within normal range
r = check(b_shift, "small shift, entirely within the normal range")
print("\nA drift small enough to sit inside the reference distribution is")
print("invisible to a novelty detector by construction, and it can still")
print("move a model's calibration. Input monitoring catches inputs the model")
print("has never seen; it does not catch a model whose relationship to the")
print("target has changed. For that you need outcomes, and outcomes arrive")
print("late — which is the subject of Chapter 179.")

print("\nAnd the reverse failure: a HIGH flag rate is not proof of a problem.")
print("A legitimate new customer segment looks exactly like an anomaly to a")
print("detector fitted on last quarter's population. The detector tells you")
print("the input distribution moved, and a human decides whether that is a")
print("bug or a business.")
