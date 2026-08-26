# -*- coding: utf-8 -*-
# Extracted from: Chapter 40 — Clustering: K-Means, DBSCAN, and Hierarchical Methods
# Source: src/.../ch040-clustering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A customer-segmentation workflow done honestly: scale, choose k with a
null reference, validate, and check the segments are actually useful.
"""
import numpy as np

rng = np.random.default_rng(23)


def kmeans(X, k, n_init=10, seed=0, max_iter=200):
    rs = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        C = [X[rs.integers(0, len(X))]]
        for _ in range(1, k):
            D2 = np.min(((X[:, None, :] - np.array(C)[None, :, :]) ** 2
                         ).sum(-1), axis=1)
            tot = D2.sum()
            C.append(X[rs.choice(len(X), p=D2 / tot if tot > 0 else None)])
        C = np.array(C)
        for _ in range(max_iter):
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
        if J < best[2]:
            best = (C, lab, J)
    return best


# --- a population with three genuine behavioural types ----------------------
N = 1200
seg = rng.choice(3, N, p=[0.55, 0.30, 0.15])
recency = np.where(seg == 0, rng.gamma(6, 8, N),
                   np.where(seg == 1, rng.gamma(2, 6, N),
                            rng.gamma(1.5, 3, N)))
frequency = np.where(seg == 0, rng.poisson(2, N),
                     np.where(seg == 1, rng.poisson(9, N),
                              rng.poisson(22, N))) + 1
monetary = frequency * np.where(seg == 0, rng.gamma(3, 12, N),
                                np.where(seg == 1, rng.gamma(4, 20, N),
                                         rng.gamma(5, 55, N)))
tenure = rng.uniform(1, 60, N)
X_raw = np.column_stack([recency, frequency, monetary, tenure])
NAMES = ["recency (days)", "frequency", "monetary (GBP)", "tenure (months)"]

print("=" * 72)
print("1. scaling is not optional (Chapter 35's lesson, again)")
print("=" * 72)
print(f"{'feature':<18} {'mean':>12} {'std':>12} "
      f"{'share of squared distance':>27}")
var = X_raw.var(0)
for nm, m_, s_, v in zip(NAMES, X_raw.mean(0), X_raw.std(0),
                         var / var.sum()):
    print(f"{nm:<18} {m_:>12.2f} {s_:>12.2f} {v:>27.4f}")
print("\nUnscaled, 'monetary' contributes almost all of the distance purely")
print("because it is measured in pounds. k-means would be clustering on one")
print("column.")

X = (X_raw - X_raw.mean(0)) / X_raw.std(0)


def silhouette(X, lab):
    labs = np.unique(lab)
    if len(labs) < 2:
        return float("nan")
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    out = np.zeros(len(X))
    for i in range(len(X)):
        own = lab == lab[i]
        own[i] = False
        a = D[i, own].mean() if own.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


# --- 2. choosing k against a null reference (the gap statistic) -------------
print("\n" + "=" * 72)
print("2. choosing k — with a reference distribution, so k=1 is possible")
print("=" * 72)
print("The gap statistic compares log(WCSS) against its value on data with")
print("NO clustering, sampled from the bounding box. It is the only common")
print("method that can answer 'there is no structure here' (section 5.5).\n")


def gap_statistic(X, k, n_ref=12, seed=0):
    """Tibshirani's gap, with a PCA-ALIGNED reference box.

    A reference sampled from the axis-aligned bounding box is a poor null
    for correlated features: the box contains large empty corners, so the
    reference is itself clusterable and the gap keeps rising with k. Drawing
    the box in the principal-component frame and rotating back removes that
    artefact, and it is what the original paper recommends.
    """
    rs = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T                                  # rotate to the PC frame
    lo, hi = Z.min(0), Z.max(0)
    _, _, J = kmeans(X, k, n_init=5, seed=seed)
    refs = []
    for b in range(n_ref):
        Zr = rs.uniform(lo, hi, Z.shape)
        Xr = Zr @ Vt + X.mean(0)                   # ...and rotate back
        refs.append(np.log(kmeans(Xr, k, n_init=3,
                                  seed=int(rs.integers(0, 10 ** 6)))[2]))
    refs = np.array(refs)
    gap = float(refs.mean() - np.log(J))
    sk = float(refs.std() * np.sqrt(1 + 1 / n_ref))
    return gap, sk, J


print(f"{'k':>3} {'WCSS':>10} {'silhouette':>12} {'gap':>8} {'s_k':>7}")
rows = []
for k in range(1, 8):
    gap, sk, J = gap_statistic(X, k, seed=k)
    sil = silhouette(X, kmeans(X, k, seed=1)[1]) if k > 1 else float("nan")
    rows.append((k, J, sil, gap, sk))
    print(f"{k:>3} {J:>10.1f} "
          f"{(f'{sil:.4f}' if k > 1 else 'n/a'):>12} {gap:>8.4f} {sk:>7.4f}")

# Tibshirani's rule: smallest k with gap(k) >= gap(k+1) - s_{k+1}
k_gap = None
for i in range(len(rows) - 1):
    if rows[i][3] >= rows[i + 1][3] - rows[i + 1][4]:
        k_gap = rows[i][0]
        break
k_sil = max(rows[1:], key=lambda r: r[2])[0]
k_gapmax = max(rows, key=lambda r: r[3])[0]
print(f"\ngap statistic, first-k rule : k = {k_gap}")
print(f"gap statistic, argmax       : k = {k_gapmax}")
print(f"silhouette                  : k = {k_sil}")
print(f"the truth                   : k = 3")
print("\nWCSS alone chooses nothing — it falls monotonically, as eq. 40.1")
print("requires, so the 'elbow' is whatever the reader decides to see.")
print("\nThree methods, three answers, and none of them is 3. That is the")
print("honest state of 'how many clusters', and it is worth diagnosing")
print("rather than resolving by preference.")
print("\nThe suspect is skew. Monetary value is a product of two")
print("heavy-tailed quantities and recency is a gamma draw, so even after")
print("standardising the cloud is a long spike rather than a blob. A")
print("uniform reference box — however it is oriented — is a poor null for")
print("that shape, which is why the gap behaves erratically. Log-transform")
print("the skewed columns first, as is standard for RFM, and try again:\n")

X_log = np.column_stack([np.log1p(recency), np.log1p(frequency),
                         np.log1p(monetary), tenure])
X_log = (X_log - X_log.mean(0)) / X_log.std(0)

rows2 = []
for k in range(1, 8):
    gap, sk, J = gap_statistic(X_log, k, seed=100 + k)
    sil = silhouette(X_log, kmeans(X_log, k, seed=1)[1]) if k > 1 else np.nan
    rows2.append((k, J, sil, gap, sk))
print(f"{'k':>3} {'WCSS':>10} {'silhouette':>12} {'gap':>8} {'s_k':>7}")
for r in rows2:
    print(f"{r[0]:>3} {r[1]:>10.1f} "
          f"{(f'{r[2]:.4f}' if r[0] > 1 else 'n/a'):>12} {r[3]:>8.4f} "
          f"{r[4]:>7.4f}")
k_gap2 = None
for i in range(len(rows2) - 1):
    if rows2[i][3] >= rows2[i + 1][3] - rows2[i + 1][4]:
        k_gap2 = rows2[i][0]
        break
k_sil2 = max(rows2[1:], key=lambda r: r[2])[0]
print(f"\non log-transformed features: gap -> k = {k_gap2}, "
      f"silhouette -> k = {k_sil2}, truth = 3")

print("\nThe transform did not have to be applied to the model at all — it")
print("was applied so the VALIDATION would work, which is a distinction")
print("worth noticing. Whether the methods now agree or still do not, the")
print("lesson is the same one section 5.5 states: when several k values")
print("score alike, that is evidence there is no natural number of")
print("clusters, and reporting one anyway manufactures a finding. Customer")
print("behaviour really is a continuum with modes in it.")

# --- 3. does the clustering survive resampling? -----------------------------
print("\n" + "=" * 72)
print("3. stability across bootstrap resamples")
print("=" * 72)
def adjusted_rand(a, b):
    la, lb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == i) & (b == j)) for j in lb] for i in la])

    def c2(x):
        return x * (x - 1) / 2
    sij, si, sj = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    n2 = c2(len(a))
    exp, mx = si * sj / n2, 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


print("Mean adjusted Rand between clusterings of independent resamples,")
print("on the points they share. Chance-corrected, so unlike a raw")
print("co-assignment rate it does not rise mechanically with k.\n")
print(f"{'k':>3} {'resample stability':>22}")
for k in (2, 3, 4, 6):
    rs = np.random.default_rng(9)
    n = len(X)
    runs = []
    for b in range(10):
        idx = np.unique(rs.integers(0, n, n))
        runs.append((idx, kmeans(X[idx], k, n_init=3,
                                 seed=int(rs.integers(0, 10 ** 6)))[1]))
    sc = []
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            ia, la = runs[a]
            ib, lb = runs[b]
            sh = np.intersect1d(ia, ib)
            sc.append(adjusted_rand(la[np.searchsorted(ia, sh)],
                                    lb[np.searchsorted(ib, sh)]))
    print(f"{k:>3} {float(np.mean(sc)):>22.4f}")
print("\nRead this as a curve, not a winner: stability is highest at the k")
print("values where the partition reproduces, and the profile tells you how")
print("much of the structure is real. A k whose stability is close to the")
print("best is not distinguishable from it, which is the same message the")
print("gap and the silhouette gave by disagreeing.")

# --- 4. profile the segments, and check they are useful ---------------------
print("\n" + "=" * 72)
print("4. profiling — and the only test that matters")
print("=" * 72)
C, lab, _ = kmeans(X, 3, n_init=20, seed=4)
print(f"{'segment':>8} {'n':>6} " +
      " ".join(f"{nm.split()[0]:>12}" for nm in NAMES))
for j in range(3):
    m = lab == j
    print(f"{j:>8} {int(m.sum()):>6} " +
          " ".join(f"{v:>12.2f}" for v in X_raw[m].mean(0)))

print("\nCross-tabulation against the true behavioural type:")
print(f"{'':>10}" + "".join(f"{'true ' + str(t):>10}" for t in range(3)))
for j in range(3):
    print(f"{'cluster ' + str(j):>10}" +
          "".join(f"{int(np.sum((lab == j) & (seg == t))):>10}"
                  for t in range(3)))

# the honest criterion: does the segmentation predict anything?
future_spend = (monetary * rng.uniform(0.6, 1.4, N)
                + rng.normal(0, 40, N))
base = float(np.mean((future_spend - future_spend.mean()) ** 2))
seg_pred = np.array([future_spend[lab == j].mean() for j in range(3)])[lab]
with_seg = float(np.mean((future_spend - seg_pred) ** 2))
rand_lab = rng.integers(0, 3, N)
rand_pred = np.array([future_spend[rand_lab == j].mean()
                      for j in range(3)])[rand_lab]
with_rand = float(np.mean((future_spend - rand_pred) ** 2))

print(f"\npredicting next-period spend:")
print(f"  variance with no segmentation     : {base:>12,.0f}")
print(f"  variance within k-means segments  : {with_seg:>12,.0f}  "
      f"({(1 - with_seg / base) * 100:.1f}% explained)")
print(f"  variance within RANDOM segments   : {with_rand:>12,.0f}  "
      f"({(1 - with_rand / base) * 100:.1f}% explained)")

print("\nThat last line is the control, and it is the one people leave out.")
print("A segmentation is only worth having if it beats an arbitrary")
print("partition of the same size at something you actually care about.")
print("Silhouette, gap and stability all assess the SHAPE of the clusters;")
print("only this assesses whether they are worth acting on.")
