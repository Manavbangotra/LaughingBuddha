# -*- coding: utf-8 -*-
# Extracted from: Chapter 31 — What Machine Learning Is: Learning Paradigms
# Source: src/.../ch031-what-ml-is.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Four datasets with different structure, four model families.

Each family wins on the data whose structure matches its assumption, and none
wins everywhere — no free lunch, made concrete.
"""
import numpy as np

rng = np.random.default_rng(3)
n = 1200


def make_datasets(n):
    """Four binary problems with deliberately different geometry."""
    out = {}

    # 1. linearly separable
    X = rng.normal(size=(n, 2))
    out["linear boundary"] = (X, (X[:, 0] + X[:, 1] > 0).astype(int))

    # 2. axis-aligned rectangle — natural for trees
    X = rng.uniform(-3, 3, (n, 2))
    out["axis-aligned box"] = (X, ((np.abs(X[:, 0]) < 1.2)
                                   & (np.abs(X[:, 1]) < 1.2)).astype(int))

    # 3. concentric rings — needs a nonlinear boundary
    r = rng.uniform(0, 3, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    out["concentric rings"] = (X, (r > 1.6).astype(int))

    # 4. XOR — no single linear split works
    X = rng.normal(size=(n, 2))
    out["XOR"] = (X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int))
    return out


# --- four tiny model families, each with one clear assumption ---------------
def fit_linear(Xtr, ytr, Xte):
    """Assumption: the boundary is a straight line."""
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    w = np.zeros(A.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        w -= 0.5 * (A.T @ (p - ytr) / len(ytr))
    B = np.column_stack([np.ones(len(Xte)), Xte])
    return (1 / (1 + np.exp(-np.clip(B @ w, -30, 30))) > 0.5).astype(int)


def fit_stump_grid(Xtr, ytr, Xte, bins=6):
    """Assumption: the boundary is axis-aligned (a coarse tree)."""
    edges = [np.quantile(Xtr[:, j], np.linspace(0, 1, bins + 1)) for j in (0, 1)]
    def cell(X):
        return tuple(np.clip(np.digitize(X[:, j], edges[j][1:-1]), 0, bins - 1)
                     for j in (0, 1))
    ctr = cell(Xtr)
    table = {}
    for i in range(len(Xtr)):
        table.setdefault((ctr[0][i], ctr[1][i]), []).append(ytr[i])
    table = {k: int(np.mean(v) >= 0.5) for k, v in table.items()}
    cte = cell(Xte)
    default = int(ytr.mean() >= 0.5)
    return np.array([table.get((cte[0][i], cte[1][i]), default)
                     for i in range(len(Xte))])


def fit_knn(Xtr, ytr, Xte, k=9):
    """Assumption: nearby points share labels."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d, k, axis=1)[:, :k]
    return (ytr[nn].mean(axis=1) >= 0.5).astype(int)


def fit_radial(Xtr, ytr, Xte):
    """Assumption: the boundary depends only on distance from the origin."""
    rtr = np.linalg.norm(Xtr, axis=1)[:, None]
    rte = np.linalg.norm(Xte, axis=1)[:, None]
    return fit_linear(rtr, ytr, rte)


models = {"linear": fit_linear, "axis-aligned": fit_stump_grid,
          "kNN": fit_knn, "radial": fit_radial}


N_TEST = 4000          # held out at a fixed size so the two tables compare


def bake_off(n_train, label):
    """Train on n_train rows, always evaluate on the same-size held-out set."""
    datasets = make_datasets(n_train + N_TEST)
    print(f"\n{label}")
    print(f"{'dataset':<20} " + " ".join(f"{m:>13}" for m in models))
    print("-" * 72)
    best_of = {}
    for dname, (X, y) in datasets.items():
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        row, accs = [], {}
        for mname, fn in models.items():
            acc = (fn(Xtr, ytr, Xte) == yte).mean()
            accs[mname] = acc
            row.append(f"{acc:>13.3f}")
        best_of[dname] = (max(accs, key=accs.get), max(accs.values()))
        print(f"{dname:<20} " + " ".join(row))
    for dname, (mname, acc) in best_of.items():
        print(f"  best on {dname:<20} {mname:<14} {acc:.3f}")
    return best_of


big = bake_off(2000, "2000 training rows, 2 features — plenty of data")

print("\nkNN wins or ties nearly everything. That is not an accident and not a")
print("recommendation: in two dimensions with two thousand training points,")
print("'nearby points share labels' is almost always true, so the weakest")
print("assumption is the best one. Assumptions earn their keep when data is")
print("scarce:")

small = bake_off(40, "40 training rows, same four problems, same test set")

print("\nNow they separate, and the ordering changes. With forty points kNN")
print("no longer has neighbours close enough to trust, while the models that")
print("assume a shape need only a handful of points to locate it. A strong")
print("assumption is cheap when it is right and expensive when it is wrong —")
print("that is the whole trade (section 5.3).\n")
for dname in big:
    print(f"  {dname:<20} 2000 rows: {big[dname][0]:<14}"
          f"   40 rows: {small[dname][0]}")

# --- and the assumption is visible in what each one CANNOT do ---------------
print("\n" + "=" * 72)
print("what each assumption rules out")
print("=" * 72)
datasets = make_datasets(n)
X, y = datasets["XOR"]
cut = int(0.7 * n)
lin_acc = (fit_linear(X[:cut], y[:cut], X[cut:]) == y[cut:]).mean()
print(f"linear model on XOR: {lin_acc:.3f}  (chance is "
      f"{max(y[cut:].mean(), 1 - y[cut:].mean()):.3f})")
print("No amount of data fixes this: XOR is not in the linear hypothesis")
print("space at all. It is underfitting caused by the choice, not the fit.")

X2 = np.column_stack([X, X[:, 0] * X[:, 1]])       # add the interaction
lin_acc2 = (fit_linear(X2[:cut], y[:cut], X2[cut:]) == y[cut:]).mean()
print(f"\nlinear model on XOR + an interaction feature: {lin_acc2:.3f}")
print("Feature engineering (Chapter 27) enlarged the hypothesis space to")
print("contain the answer. That is the same lever as changing model family.")
