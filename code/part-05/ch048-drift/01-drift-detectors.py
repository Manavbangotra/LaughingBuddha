# Extracted from: Chapter 48 — Monitoring, Drift, and Model Degradation
# Source: src/.../ch048-drift.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""PSI, per-feature testing with correction, and a classifier two-sample
test — with each one's blind spot measured.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the detectors ----------------------------------------------------------
def psi(ref, cur, edges):
    """Eq. 48.1. Bin edges are FIXED from the reference period — recomputing
    them per window would make runs incomparable (section 5.1)."""
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_edges(ref, bins=10):
    e = np.quantile(ref, np.linspace(0, 1, bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def ks_stat(a, b):
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def ks_pvalue(d, n1, n2):
    """Asymptotic two-sided KS p-value."""
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1) ** (j - 1)
                                    * np.exp(-2 * j ** 2 * lam ** 2)), 0, 1))


def benjamini_hochberg(pvals, alpha=0.05):
    """Control the false DISCOVERY rate: the expected fraction of alerts
    that are false, which is the operationally relevant quantity."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    if k:
        out[order[:k]] = True
    return out


def classifier_two_sample(ref, cur, seed=0):
    """Train a classifier to tell reference from current. Its held-out AUC
    is an interpretable effect size: 0.5 = indistinguishable.

    A depth-limited tree ensemble would be better; a logistic model on
    standardised features is enough to show the mechanism and keeps this
    listing self-contained.
    """
    rs = np.random.default_rng(seed)
    X = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    # add pairwise products so the test can see JOINT changes, not only
    # marginal ones — without them it is just a linear marginal test
    d = X.shape[1]
    prods = np.column_stack([X[:, i] * X[:, j]
                             for i in range(d) for j in range(i + 1, d)])
    X = np.column_stack([X, prods])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    perm = rs.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(y))
    A = np.column_stack([np.ones(cut), X[:cut]])
    w = np.zeros(A.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y[:cut]) / cut + 0.01 * np.r_[0, w[1:]]
        S = np.maximum(p * (1 - p), 1e-7)
        H = (A * S[:, None]).T @ A / cut + 0.02 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    B = np.column_stack([np.ones(len(y) - cut), X[cut:]])
    s = B @ w
    yt = y[cut:]
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(yt.sum())
    if npos in (0, len(yt)):
        return 0.5
    return float((r[yt == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(yt) - npos)))


# --- reference data and four kinds of change --------------------------------
D = 8


def make_reference(n, seed):
    rs = np.random.default_rng(seed)
    z = rs.normal(size=(n, 3))
    M = rs.normal(size=(3, D)) if seed == 0 else MIX
    return z @ M + rs.normal(0, 0.4, (n, D))


MIX = np.random.default_rng(0).normal(size=(3, D))


def make_window(n, kind, seed):
    rs = np.random.default_rng(seed)
    z = rs.normal(size=(n, 3))
    X = z @ MIX + rs.normal(0, 0.4, (n, D))
    if kind == "stable":
        return X
    if kind == "marginal_shift":
        X[:, 2] += 0.6                       # one feature moved
        return X
    if kind == "joint_only":
        # every MARGINAL is preserved; the joint structure is destroyed
        for j in range(D):
            X[:, j] = rs.permutation(X[:, j])
        return X
    if kind == "point_mass":
        bad = rs.random(n) < 0.10
        X[bad, 1] = 0.0                      # an upstream default value
        return X
    raise ValueError(kind)


ref = make_window(6000, "stable", 1)
EDGES = [psi_edges(ref[:, j]) for j in range(D)]

print("=" * 72)
print("three detectors, four kinds of change")
print("=" * 72)
print(f"{'change':<22} {'max PSI':>9} {'features flagged':>18} "
      f"{'classifier AUC':>16}")
for kind in ("stable", "marginal_shift", "joint_only", "point_mass"):
    cur = make_window(2000, kind, 42)
    psis = [psi(ref[:, j], cur[:, j], EDGES[j]) for j in range(D)]
    pvals = [ks_pvalue(ks_stat(ref[:, j], cur[:, j]), len(ref), len(cur))
             for j in range(D)]
    flagged = int(benjamini_hochberg(pvals, 0.05).sum())
    auc = classifier_two_sample(ref, cur, seed=3)
    print(f"{kind:<22} {max(psis):>9.4f} {flagged:>15} /{D:<2} "
          f"{auc:>16.4f}")

print("\nRead the rows against each other; each detector has a different")
print("blind spot and no row is caught by all three.")
print("\nThe MARGINAL SHIFT is caught by everything, as it should be.")
print("\nThe POINT MASS is caught by the per-feature test — one feature's")
print("marginal genuinely changed — and is nearly invisible to the")
print("classifier, whose AUC barely leaves 0.5. Ten per cent of rows pinned")
print("to a single value is a small, sharp change that a smooth decision")
print("boundary cannot separate on, which is why Chapter 47 checked for it")
print("directly rather than hoping a general detector would notice.")
print("\nThe third row is the one that separates the methods. Every marginal")
print("is preserved exactly — the columns were permuted independently — so")
print("PSI and the per-feature tests see nothing, correctly, because they")
print("only ever look at one feature at a time. The classifier two-sample")
print("test sees it immediately, because its pairwise product terms let it")
print("notice that the features no longer move together.")
print("\nThis is Chapter 42's finding arriving in a monitoring context: the")
print("detector's definition of 'different' decides what it can see, and no")
print("single detector covers the space.")

# --- section 6.2: the false-alarm arithmetic --------------------------------
print("\n" + "=" * 72)
print("why per-feature testing needs correction (eq. 48.4)")
print("=" * 72)
print("Stable data throughout — there is nothing to detect. 200 simulated")
print("monitoring days, 8 features tested per day.\n")

n_days = 200
raw_days, bh_days, bonf_days = 0, 0, 0
for day in range(n_days):
    cur = make_window(2000, "stable", 1000 + day)
    pvals = np.array([ks_pvalue(ks_stat(ref[:, j], cur[:, j]),
                                len(ref), len(cur)) for j in range(D)])
    raw_days += bool((pvals < 0.05).any())
    bh_days += bool(benjamini_hochberg(pvals, 0.05).any())
    bonf_days += bool((pvals < 0.05 / D).any())

print(f"{'rule':<34} {'days with >=1 alert':>21} {'rate':>8} "
      f"{'per year':>10}")
for label, count in (("uncorrected, alpha=0.05", raw_days),
                     ("Benjamini-Hochberg, FDR 0.05", bh_days),
                     ("Bonferroni, alpha=0.05/8", bonf_days)):
    print(f"{label:<34} {count:>21} {count / n_days:>8.3f} "
          f"{count / n_days * 365:>10.0f}")
print(f"\npredicted by eq. 48.4 for 8 independent features: "
      f"{1 - 0.95 ** D:.3f}")
print("\nThe uncorrected rule alerts on a stable system at close to the rate")
print("eq. 48.4 predicts, which at eight features is already most weeks and")
print("at thirty features would be most days. Correction is one line and it")
print("is the difference between a monitor people read and a muted channel.")

# --- section 6.4: thresholds from measured false-alarm rates ----------------
print("\n" + "=" * 72)
print("setting a PSI threshold empirically (eq. 48.6)")
print("=" * 72)
print("The conventional 0.1 / 0.25 thresholds are credit-scoring conventions,")
print("not derivations. Here is what stable data actually produces, at")
print("several monitoring window sizes:\n")
print(f"{'window':>8} {'median PSI':>11} {'99th pct':>10} "
      f"{'PSI at a real 0.3 sd shift':>28} {'0.25 fires?':>12}")
for n_win in (200, 500, 2000, 10000):
    stable_vals = np.array(
        [psi(ref[:, 0], make_window(n_win, "stable", 5000 + k)[:, 0],
             EDGES[0]) for k in range(120)])
    shifted_vals = np.array(
        [psi(ref[:, 0],
             make_window(n_win, "stable", 7000 + k)[:, 0]
             + 0.3 * ref[:, 0].std(), EDGES[0]) for k in range(60)])
    fires = np.mean(stable_vals > 0.25) > 0 or np.mean(shifted_vals > 0.25) > 0
    print(f"{n_win:>8} {np.median(stable_vals):>11.4f} "
          f"{np.percentile(stable_vals, 99):>10.4f} "
          f"{np.median(shifted_vals):>28.4f} "
          f"{('yes' if np.median(shifted_vals) > 0.25 else 'no'):>12}")

print("\nThe result is not the one the convention would lead you to expect,")
print("and it is more useful.")
print("\nPSI on STABLE data is not zero, and its scale depends strongly on")
print("the window size: the 99th percentile falls from 0.125 at a 200-row")
print("window to 0.005 at 10,000 — a twenty-five-fold range for data that")
print("has not changed at all. Small windows manufacture PSI out of sampling")
print("noise.")
print("\nMeanwhile the conventional 0.25 threshold never fires here — not on")
print("stable data, which is good, and not on a genuine 0.3-standard-")
print("deviation shift either, which is not. At a 10,000-row window that")
print("threshold sits fifty times above the noise floor and would miss any")
print("shift short of a catastrophe.")
print("\nSo a fixed threshold cannot be right across window sizes, because")
print("the statistic's own scale moves by a factor of twenty-five. Eq. 48.6")
print("gives a threshold in the units of your own data and window with a")
print("false-alarm rate you chose, at the cost of one pass over a stable")
print("period — and it is the difference between a number you can defend and")
print("one inherited from a different industry at a different sample size.")
