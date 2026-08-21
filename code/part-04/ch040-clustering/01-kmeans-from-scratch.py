# Extracted from: Chapter 40 — Clustering: K-Means, DBSCAN, and Hierarchical Methods
# Source: src/.../ch040-clustering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""k-means from scratch: Lloyd's algorithm, k-means++, and the three failures.
"""
import numpy as np

rng = np.random.default_rng(0)


def kmeans(X, k, init="++", n_init=10, max_iter=300, tol=1e-9, seed=0):
    """Lloyd's algorithm (eq. 40.2) with restarts; returns the best run."""
    rs = np.random.default_rng(seed)
    best = (None, None, np.inf, 0)
    for run in range(n_init):
        C = (kmeanspp_init(X, k, rs) if init == "++"
             else X[rs.choice(len(X), k, replace=False)].copy())
        for it in range(max_iter):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = C.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:
                    newC[j] = X[rs.integers(0, len(X))]   # re-seed empty
            shift = float(((newC - C) ** 2).sum())
            C = newC
            if shift < tol:
                break
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(X)), lab].sum())        # eq. 40.1
        if J < best[2]:
            best = (C, lab, J, it + 1)
    return best


def kmeanspp_init(X, k, rs):
    """Seed proportional to D^2 (section 5.2)."""
    C = [X[rs.integers(0, len(X))]]
    for _ in range(1, k):
        D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(-1),
                    axis=1)
        total = D2.sum()
        p = D2 / total if total > 0 else np.full(len(X), 1 / len(X))
        C.append(X[rs.choice(len(X), p=p)])
    return np.array(C)


def make_blobs(n, centres, spread=0.6):
    centres = np.asarray(centres, float)
    lab = rng.integers(0, len(centres), n)
    return centres[lab] + rng.normal(0, spread, (n, centres.shape[1])), lab


# --- k-means++ vs random initialisation -------------------------------------
print("=" * 72)
print("initialisation decides the local optimum you land in (section 5.2)")
print("=" * 72)
# eight well-separated blobs: random seeding will double up on some
centres8 = [[0, 0], [10, 0], [20, 0], [30, 0],
            [0, 10], [10, 10], [20, 10], [30, 10]]
X8, _ = make_blobs(1600, centres8, spread=0.8)

print(f"{'init':>10} {'restarts':>10} {'best WCSS':>12} {'mean WCSS':>12} "
      f"{'worst WCSS':>12}")
for init in ("random", "++"):
    for n_init in (1, 10):
        Js = [kmeans(X8, 8, init=init, n_init=1, seed=100 + s)[2]
              for s in range(20)]
        if n_init == 1:
            print(f"{init:>10} {1:>10} {min(Js):>12.1f} "
                  f"{np.mean(Js):>12.1f} {max(Js):>12.1f}")
        else:
            best_of_10 = [min(Js[i:i + 10]) for i in (0, 10)]
            print(f"{init:>10} {10:>10} {min(best_of_10):>12.1f} "
                  f"{np.mean(best_of_10):>12.1f} {max(best_of_10):>12.1f}")

print("\nWith a single random start the objective varies enormously across")
print("seeds — some runs place two centroids in one blob and leave another")
print("empty, and Lloyd's algorithm cannot recover, because both of its")
print("steps only ever DECREASE the objective (section 6.1). k-means++ is")
print("far more consistent, and restarts help both. Never run k-means once.")

# --- section 6.1: the objective never increases -----------------------------
print("\n" + "=" * 72)
print("both steps of Lloyd's algorithm decrease J (section 6.1)")
print("=" * 72)
X3, _ = make_blobs(600, [[0, 0], [4, 4], [8, 0]], spread=1.0)
rs = np.random.default_rng(3)
C = X3[rs.choice(len(X3), 3, replace=False)].copy()
print(f"{'iter':>5} {'J after assign':>16} {'J after update':>16}")
for it in range(7):
    d = ((X3[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    J_assign = float(d[np.arange(len(X3)), lab].sum())
    for j in range(3):
        if (lab == j).any():
            C[j] = X3[lab == j].mean(0)
    d2 = ((X3[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    J_update = float(d2[np.arange(len(X3)), lab].sum())
    print(f"{it:>5} {J_assign:>16.4f} {J_update:>16.4f}")
print("\nMonotone decrease, in both columns, at every step. Since there are")
print("finitely many assignments the algorithm must terminate — and it")
print("terminates at whichever local optimum the initialisation led to.")

# --- section 4.2: the three assumptions, and the three failures -------------
print("\n" + "=" * 72)
print("k-means' three assumptions, each violated in turn (section 4.2)")
print("=" * 72)


def purity(true_lab, pred_lab):
    """Fraction correct under the best matching of predicted to true labels."""
    total = 0
    for p in np.unique(pred_lab):
        m = pred_lab == p
        if m.any():
            total += np.bincount(true_lab[m]).max()
    return total / len(true_lab)


# 1. elongated clusters
t = rng.uniform(-6, 6, 800)
Xe = np.column_stack([t, 0.25 * t + rng.normal(0, 0.35, 800)])
Xe = np.vstack([Xe, np.column_stack([t, 0.25 * t + 4
                                     + rng.normal(0, 0.35, 800)])])
ye = np.r_[np.zeros(800, int), np.ones(800, int)]

# 2. different sizes
Xs = np.vstack([rng.normal([0, 0], 0.5, (1500, 2)),
                rng.normal([5, 0], 0.5, (60, 2)),
                rng.normal([5, 3], 0.5, (60, 2))])
ys = np.r_[np.zeros(1500, int), np.ones(60, int), np.full(60, 2)]

# 3. different densities
Xd = np.vstack([rng.normal([0, 0], 0.3, (600, 2)),
                rng.normal([4, 0], 1.8, (600, 2))])
yd = np.r_[np.zeros(600, int), np.ones(600, int)]

# 4. the case it is built for
Xg, yg = make_blobs(1200, [[0, 0], [6, 0], [3, 5]], spread=0.9)

print(f"{'geometry':<34} {'k':>3} {'purity':>8} {'verdict':<22}")
for name, Xc, yc, k in (
        ("spherical, equal size (its home)", Xg, yg, 3),
        ("elongated parallel bands", Xe, ye, 2),
        ("one huge cluster, two tiny", Xs, ys, 3),
        ("very different densities", Xd, yd, 2)):
    _, lab, _, _ = kmeans(Xc, k, n_init=10, seed=7)
    pu = purity(yc, lab)
    verdict = ("recovers it" if pu > 0.95 else
               "partly" if pu > 0.75 else "fails")
    print(f"{name:<34} {k:>3} {pu:>8.4f} {verdict:<22}")

print("\nThe shape assumption is the one that bites, and it bites hard: on")
print("parallel elongated bands k-means scores barely above chance. That")
print("follows directly from eq. 40.7 — the boundary between two k-means")
print("clusters is a HYPERPLANE, so the cells are convex polyhedra, and no")
print("plane separates two bands running alongside each other. It cuts them")
print("crosswise instead.")
print("\nThe other two assumptions are worth being honest about. Unequal")
print("SIZE did not hurt here, because the clusters were also far apart —")
print("the textbook warning applies when a large cluster is close enough to")
print("small ones that splitting it costs less than merging them. Unequal")
print("DENSITY cost about ten points. Neither is in the same category as")
print("the shape failure.")
print("\nThe usable summary: k-means is a Voronoi partition. If your")
print("clusters are not roughly convex blobs, that is the assumption to")
print("check first, and the other two are second-order.")

# --- the failure that matters most: no structure at all ---------------------
print("\n" + "=" * 72)
print("k-means finds clusters in pure noise, and looks confident doing it")
print("=" * 72)


def silhouette(X, lab):
    """Mean silhouette (eq. 40.3). O(N^2) — fine at this size."""
    D = np.sqrt(np.maximum(
        ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    out = np.zeros(len(X))
    labs = np.unique(lab)
    if len(labs) < 2:
        return 0.0
    for i in range(len(X)):
        own = lab == i * 0 + lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


X_noise = rng.uniform(0, 1, (500, 2))         # uniform. No clusters exist.
print("500 points drawn uniformly from a square — there is no structure.\n")
print(f"{'k':>4} {'WCSS':>10} {'mean silhouette':>17} {'reported as':<26}")
for k in (2, 3, 4, 5, 8):
    _, lab, J, _ = kmeans(X_noise, k, n_init=10, seed=5)
    s = silhouette(X_noise, lab)
    verdict = ("'reasonable structure'" if s > 0.35 else
               "'weak but present'" if s > 0.25 else "'no structure'")
    print(f"{k:>4} {J:>10.2f} {s:>17.4f} {verdict:<26}")

print("\nEvery k returns k clusters with a positive silhouette. Nothing in")
print("the algorithm or the metric can say 'there is nothing here'. The")
print("elbow method is no help either: WCSS falls smoothly and monotonically")
print("with k, as it must, since more centroids can only reduce eq. 40.1.")
print("\nThis is the single most common way clustering is misused, and it is")
print("why section 5.4 insists on a reference distribution, a stability")
print("check, or an external outcome.")
