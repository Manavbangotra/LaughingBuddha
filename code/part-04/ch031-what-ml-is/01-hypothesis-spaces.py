# Extracted from: Chapter 31 — What Machine Learning Is: Learning Paradigms
# Source: src/.../ch031-what-ml-is.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Hypothesis spaces, the memorisation baseline, and no free lunch — measured.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 6.1: training error is nearly uninformative --------------------
print("=" * 72)
print("a memoriser: zero training error, chance generalisation")
print("=" * 72)


class Memoriser:
    """Stores every training pair; guesses on anything unseen."""

    def fit(self, X, y):
        self.table = {tuple(np.round(row, 6)): label
                      for row, label in zip(X, y)}
        self.default = int(round(y.mean()))
        return self

    def predict(self, X):
        return np.array([self.table.get(tuple(np.round(r, 6)), self.default)
                         for r in X])


n, d = 400, 8
X = rng.normal(size=(n, d))
y = (rng.random(n) < 0.5).astype(int)         # a coin flip: NO signal

split = n // 2
m = Memoriser().fit(X[:split], y[:split])
print(f"training accuracy : {(m.predict(X[:split]) == y[:split]).mean():.4f}")
print(f"test accuracy     : {(m.predict(X[split:]) == y[split:]).mean():.4f}")
print("Perfect on data it has seen, chance on data it has not. Training")
print("error alone cannot distinguish learning from memorising.")

# --- section 4.3: the hypothesis space decides what is reachable ------------
print("\n" + "=" * 72)
print("a model can only find what its hypothesis space contains")
print("=" * 72)

x = np.linspace(-3, 3, 600)
targets = {
    "linear":     2 * x + 1,
    "quadratic":  x ** 2,
    "step":       np.where(x > 0, 3.0, -1.0),
    "sinusoid":   3 * np.sin(2 * x),
}


def fit_poly(x, y, degree):
    A = np.vander(x, degree + 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ beta


def fit_stumps(x, y, n_splits=8):
    """A depth-limited axis-aligned partition — a tiny 'tree' hypothesis space."""
    edges = np.quantile(x, np.linspace(0, 1, n_splits + 1))
    out = np.empty_like(y)
    for i in range(n_splits):
        m_ = (x >= edges[i]) & (x <= edges[i + 1])
        if m_.any():
            out[m_] = y[m_].mean()
    return out


print(f"{'target':<12} {'linear model R2':>17} {'8-region steps R2':>19}")
for name, y_t in targets.items():
    def r2(pred):
        return 1 - np.sum((y_t - pred) ** 2) / np.sum((y_t - y_t.mean()) ** 2)
    print(f"{name:<12} {r2(fit_poly(x, y_t, 1)):>17.4f} "
          f"{r2(fit_stumps(x, y_t)):>19.4f}")

print("\nThe linear space contains the linear target exactly (R2 = 1) and is")
print("worthless on the quadratic (R2 = 0.0000 — a symmetric target has zero")
print("linear component, so the best line is the mean). Note that the line")
print("still scores 0.75 on the step: a wrong hypothesis space is usually not")
print("zero-skill, which is exactly what makes underfitting hard to notice.")

# --- section 6.2: no free lunch, by enumeration -----------------------------
print("\n" + "=" * 72)
print("no free lunch, verified by enumerating every target function")
print("=" * 72)

n_points, n_seen = 12, 6
seen, unseen = np.arange(n_seen), np.arange(n_seen, n_points)

algorithms = {
    "always 0":        lambda tr_y, k: np.zeros(k, dtype=int),
    "always 1":        lambda tr_y, k: np.ones(k, dtype=int),
    "majority of seen": lambda tr_y, k: np.full(k, int(tr_y.mean() >= 0.5)),
    "alternating":     lambda tr_y, k: np.arange(k) % 2,
    "random":          lambda tr_y, k: rng.integers(0, 2, k),
}

n_functions = 2 ** n_points
scores = {name: 0 for name in algorithms}
for code in range(n_functions):
    truth = np.array([(code >> i) & 1 for i in range(n_points)])
    for name, alg in algorithms.items():
        pred = alg(truth[seen], len(unseen))
        scores[name] += (pred == truth[unseen]).mean()

print(f"averaged over all {n_functions:,} possible target functions on "
      f"{n_points} points:\n")
print(f"{'algorithm':<20} {'off-training-set accuracy':>27}")
for name, total in scores.items():
    print(f"{name:<20} {total / n_functions:>27.4f}")
print("\nEvery deterministic algorithm scores exactly 0.5000, including the")
print("sensible ones; the randomised one lands within sampling error of it.")
print("An algorithm's value comes entirely from real problems NOT being")
print("drawn uniformly from this set (section 6.2).")

# --- ...and the same algorithms on a STRUCTURED subset ----------------------
print("\nnow restricted to 'smooth' targets — those with at most 2 label")
print("changes along the ordering, which is what real problems look like:")
smooth = []
for code in range(n_functions):
    truth = np.array([(code >> i) & 1 for i in range(n_points)])
    if np.sum(np.abs(np.diff(truth))) <= 2:
        smooth.append(truth)

scores2 = {name: 0 for name in algorithms}
for truth in smooth:
    for name, alg in algorithms.items():
        pred = alg(truth[seen], len(unseen))
        scores2[name] += (pred == truth[unseen]).mean()

print(f"\n{len(smooth)} smooth functions out of {n_functions:,}\n")
print(f"{'algorithm':<20} {'off-training-set accuracy':>27}")
for name, total in scores2.items():
    print(f"{name:<20} {total / len(smooth):>27.4f}")
print("\nExactly one algorithm rises above chance: the only one that LOOKS AT")
print("the training labels. 'always 0', 'always 1' and 'alternating' ignore")
print("the data, so structure in the target cannot help them. Structure in")
print("the world plus an algorithm that exploits it is what beats chance —")
print("neither alone does (section 6.2).")

# --- sample complexity (section 6.3) ----------------------------------------
print("\n" + "=" * 72)
print("capacity and sample size (section 6.3)")
print("=" * 72)
print(f"{'|F| (hypothesis space)':>24} {'N for gap<=0.05':>17} "
      f"{'N for gap<=0.01':>17}")
for M in (10, 10**3, 10**6, 10**12):
    n05 = (np.log(M) + np.log(1 / 0.05)) / (2 * 0.05 ** 2)
    n01 = (np.log(M) + np.log(1 / 0.05)) / (2 * 0.01 ** 2)
    print(f"{M:>24,} {n05:>17,.0f} {n01:>17,.0f}")
print("\nA million-fold larger hypothesis space costs about 3x the data, not a")
print("million times it — that is the log M. Tightening the gap fivefold")
print("costs 25x, the usual 1/eps^2.")
