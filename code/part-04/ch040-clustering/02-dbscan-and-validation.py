# -*- coding: utf-8 -*-
# Extracted from: Chapter 40 — Clustering: K-Means, DBSCAN, and Hierarchical Methods
# Source: src/.../ch040-clustering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""DBSCAN, hierarchical linkage, and honest cluster validation.
"""
import numpy as np

rng = np.random.default_rng(11)


def pairwise(A, B=None):
    B = A if B is None else B
    return np.sqrt(np.maximum(
        ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0))


def dbscan(X, eps, min_samples):
    """Section 5.3, literally: core points, then transitive closure."""
    D = pairwise(X)
    neigh = [np.flatnonzero(row <= eps) for row in D]
    core = np.array([len(nb) >= min_samples for nb in neigh])
    labels = np.full(len(X), -1)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        stack, labels[i] = [i], cid
        while stack:
            p = stack.pop()
            for q in neigh[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    if core[q]:                 # only core points expand
                        stack.append(q)
        cid += 1
    return labels, core


def kmeans(X, k, n_init=10, seed=0):
    rs = np.random.default_rng(seed)
    best = (None, np.inf)
    for _ in range(n_init):
        C = [X[rs.integers(0, len(X))]]
        for _ in range(1, k):
            D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2
                         ).sum(-1), axis=1)
            tot = D2.sum()
            C.append(X[rs.choice(len(X),
                                 p=D2 / tot if tot > 0 else None)])
        C = np.array(C)
        for _ in range(200):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(X)), lab].sum())
        if J < best[1]:
            best = (lab, J)
    return best[0]


def silhouette(X, lab):
    mask = lab >= 0                        # noise points are not scored
    X, lab = X[mask], lab[mask]
    labs = np.unique(lab)
    if len(labs) < 2:
        return float("nan")
    D = pairwise(X)
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


def adjusted_rand(a, b):
    """External index: needs true labels, and is therefore trustworthy."""
    from itertools import product
    la, lb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in lb] for i in la])

    def c2(x):
        return x * (x - 1) / 2
    sum_ij = c2(cont).sum()
    sum_i = c2(cont.sum(1)).sum()
    sum_j = c2(cont.sum(0)).sum()
    n2 = c2(len(a))
    exp = sum_i * sum_j / n2
    mx = 0.5 * (sum_i + sum_j)
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


# --- two crescents: the canonical case where shape decides ------------------
def make_moons(n, noise=0.09):
    t = rng.uniform(0, np.pi, n)
    top = np.column_stack([np.cos(t), np.sin(t)])
    bot = np.column_stack([1 - np.cos(t), 0.35 - np.sin(t)])
    X = np.vstack([top, bot]) + rng.normal(0, noise, (2 * n, 2))
    return X, np.r_[np.zeros(n, int), np.ones(n, int)]


Xm, ym = make_moons(300)

print("=" * 72)
print("the metric decides the winner before the data is seen (section 6.4)")
print("=" * 72)

km_lab = kmeans(Xm, 2, seed=1)
db_lab, _ = dbscan(Xm, eps=0.22, min_samples=5)

print(f"{'method':<22} {'clusters':>9} {'noise':>7} "
      f"{'silhouette':>12} {'adj. Rand (truth)':>19}")
for name, lab in (("k-means, k=2", km_lab), ("DBSCAN", db_lab)):
    nc = len(set(lab[lab >= 0]))
    print(f"{name:<22} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{silhouette(Xm, lab):>12.4f} {adjusted_rand(ym, lab):>19.4f}")

print("\nRead the two right-hand columns against each other. DBSCAN recovers")
print("the true crescents almost exactly — adjusted Rand near 1.0 — and the")
print("SILHOUETTE PREFERS K-MEANS, which is simply wrong.")
print("\nThe reason is eq. 40.3. A point in the middle of a crescent is far")
print("from the far tip of its OWN crescent (large a) and close to the other")
print("crescent curling around it (small b), so the correct clustering")
print("scores badly on a metric built out of compactness. An internal index")
print("cannot arbitrate between algorithms that assume different shapes: it")
print("is a model of what a good answer looks like, and here it is the wrong")
print("model.")

# --- eps: the elbow in the k-distance plot ----------------------------------
print("\n" + "=" * 72)
print("choosing eps from the k-distance curve (section 5.3)")
print("=" * 72)
m = 5
D = pairwise(Xm)
kdist = np.sort(np.sort(D, axis=1)[:, m])
qs = [0.5, 0.7, 0.85, 0.92, 0.99, 1.0]
eps_extra = [0.30, 0.50]
print(f"{'quantile of 5-NN distance':>26} {'eps':>8} {'clusters':>9} "
      f"{'noise':>7} {'adj. Rand':>11}")
for q in qs:
    e = float(np.quantile(kdist, q))
    lab, _ = dbscan(Xm, eps=e, min_samples=m)
    nc = len(set(lab[lab >= 0]))
    print(f"{q:>26.2f} {e:>8.4f} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{adjusted_rand(ym, lab):>11.4f}")
for e in eps_extra:                        # beyond the k-distance range
    lab, _ = dbscan(Xm, eps=e, min_samples=m)
    nc = len(set(lab[lab >= 0]))
    print(f"{'(beyond the curve)':>26} {e:>8.4f} {nc:>9} "
          f"{int((lab == -1).sum()):>7} {adjusted_rand(ym, lab):>11.4f}")

print("\nToo small and the data fragments into 21 clusters with a sixth of")
print("the points discarded as noise. Too large and the two crescents merge")
print("into one, which shows up as a cluster count of 1 and an adjusted")
print("Rand near zero. The usable window here runs from roughly the 85th to")
print("the 99th percentile of the 5-NN distance — real, but narrow, and it")
print("has to be found. This sensitivity is DBSCAN's main practical")
print("weakness, and the mirror image of k-means needing k.")

# --- DBSCAN's own assumption: uniform density -------------------------------
print("\n" + "=" * 72)
print("DBSCAN assumes uniform density, and fails when that is false")
print("=" * 72)
Xv = np.vstack([rng.normal([0, 0], 0.12, (400, 2)),
                rng.normal([1.4, 0], 0.12, (400, 2)),
                rng.normal([9, 0], 2.20, (400, 2))])
yv = np.r_[np.zeros(400, int), np.ones(400, int), np.full(400, 2)]
print("two tight clusters and one diffuse one\n")
print(f"{'eps':>7} {'clusters':>9} {'noise':>7} {'tight pair kept':>17} "
      f"{'diffuse kept':>14} {'what happened':<28}")
for e in (0.08, 0.15, 0.3, 0.6, 1.2, 2.0):
    lab, _ = dbscan(Xv, eps=e, min_samples=6)
    nc = len(set(lab[lab >= 0]))
    frac_noise = (lab == -1).mean()
    # did the two tight clusters stay separate, and was the diffuse one
    # recovered as a cluster rather than discarded as noise?
    t1, t2 = lab[:400], lab[400:800]
    kept_apart = (len(set(t1[t1 >= 0]) & set(t2[t2 >= 0])) == 0
                  and (t1 >= 0).mean() > 0.5 and (t2 >= 0).mean() > 0.5)
    diffuse_kept = float((lab[800:] >= 0).mean())
    note = ("diffuse cluster mostly noise" if frac_noise > 0.25
            else "tight clusters merged" if nc < 3
            else "over-fragmented" if nc > 4 else "")
    print(f"{e:>7} {nc:>9} {int((lab == -1).sum()):>7} "
          f"{('yes' if kept_apart else 'NO'):>17} "
          f"{diffuse_kept:>14.2f} {note:<28}")
best_e = None
print("\nRead the two middle columns together — that is the whole point.")
print("Small eps keeps the tight pair apart and throws the diffuse cluster")
print("away as noise. Large eps recovers the diffuse cluster and fuses the")
print("pair into one. There is no setting that does both cleanly.")
print("\nThe least-bad compromise is eps = 0.6, and look at what it costs:")
print("the pair survives and 78% of the diffuse cluster is kept, but the")
print("run reports SEVEN clusters — the diffuse region has been chopped")
print("into pieces wherever a local thin patch appeared. Three real")
print("clusters, and no eps returns three.")
print("\nThe two tight clusters are 1.4 apart with spread 0.12; the diffuse")
print("one has spread 2.2. Small eps keeps the tight pair separate and")
print("throws the diffuse cluster away as noise; large eps captures the")
print("diffuse cluster and glues the tight pair together. There is no")
print("value that does both, because DBSCAN has ONE density scale and the")
print("data has two.")
print("\nThis is the same shape as the impossibility in section 6.3 of the")
print("next chapter, and it is what HDBSCAN exists for: it builds a")
print("hierarchy over all eps simultaneously and extracts clusters that are")
print("stable across a range of scales, rather than committing to one.")

# --- section 5.4: stability, the useful label-free check --------------------
print("\n" + "=" * 72)
print("stability: does the clustering survive resampling? (section 5.4)")
print("=" * 72)


def resample_stability(X, k, n_boot=12, seed=0):
    """Mean adjusted Rand between clusterings of independent resamples,
    computed on the points the two resamples share.

    A clustering that reflects real structure reproduces; one fitted to
    noise does not. Needs no labels and assumes no shape. Note the metric:
    'how often are two points together' rises mechanically with k, because
    most pairs are apart in every run — ARI corrects for chance and does
    not have that defect.
    """
    rs = np.random.default_rng(seed)
    n = len(X)
    runs = []
    for _ in range(n_boot):
        idx = np.unique(rs.integers(0, n, n))
        runs.append((idx, kmeans(X[idx], k, n_init=3,
                                 seed=int(rs.integers(0, 10 ** 6)))))
    scores = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            ia, la = runs[a]
            ib, lb = runs[b]
            shared = np.intersect1d(ia, ib)
            if len(shared) < 20:
                continue
            pa = la[np.searchsorted(ia, shared)]
            pb = lb[np.searchsorted(ib, shared)]
            scores.append(adjusted_rand(pa, pb))
    return float(np.mean(scores)) if scores else float("nan")


X_real, _ = make_moons(220, noise=0.06)
X_blobs = np.vstack([rng.normal([0, 0], 0.6, (220, 2)),
                     rng.normal([5, 0], 0.6, (220, 2)),
                     rng.normal([2.5, 4], 0.6, (220, 2))])
X_null = rng.uniform(0, 1, (440, 2))

print(f"{'data':<34} {'k':>3} {'silhouette':>12} "
      f"{'resample stability':>20}")
for name, Xc, k in (("three real blobs", X_blobs, 3),
                    ("two crescents (k-means is wrong)", X_real, 2),
                    ("uniform noise (no structure)", X_null, 3)):
    lab = kmeans(Xc, k, seed=2)
    print(f"{name:<34} {k:>3} {silhouette(Xc, lab):>12.4f} "
          f"{resample_stability(Xc, k):>20.4f}")

print("\nStability separates genuine structure from an arbitrary partition of")
print("noise without needing labels or assuming a shape. It is not proof")
print("that the clusters MEAN anything — only that they are reproducible —")
print("but it reliably catches the case that a silhouette cannot.")
