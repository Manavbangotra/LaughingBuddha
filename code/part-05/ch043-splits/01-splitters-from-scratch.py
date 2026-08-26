# -*- coding: utf-8 -*-
# Extracted from: Chapter 43 — Splits, Cross-Validation, and Honest Evaluation
# Source: src/.../ch043-splits.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Group, time and shifted splitters from scratch, and what each leak costs.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the splitters ----------------------------------------------------------
def random_split(n, frac=0.75, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def group_split(groups, frac=0.75, seed=0):
    """Split on the GROUP level, so no group appears on both sides."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    keep = set(perm[:int(frac * len(uniq))].tolist())
    mask = np.array([g in keep for g in groups])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def time_split(times, frac=0.75, embargo=0.0):
    """Everything before the cut trains; everything after `embargo` validates.

    The embargo is the gap of section 6.2 — rows inside it are DISCARDED, not
    assigned to either side, because they are contaminated in both directions.
    """
    order = np.argsort(times, kind="mergesort")
    cut_t = np.quantile(times, frac)
    tr = order[times[order] <= cut_t]
    va = order[times[order] > cut_t + embargo]
    return tr, va


def group_kfold(groups, k=5, seed=0):
    """K folds in which each group appears in exactly one validation fold."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    buckets = np.array_split(perm, k)
    out = []
    for b in buckets:
        held = set(b.tolist())
        m = np.array([g in held for g in groups])
        out.append((np.flatnonzero(~m), np.flatnonzero(m)))
    return out


def expanding_window(times, n_splits=5, embargo=0.0):
    """Train on everything up to t, validate on the next block, advance."""
    order = np.argsort(times, kind="mergesort")
    ts = times[order]
    edges = np.quantile(ts, np.linspace(0.4, 1.0, n_splits + 1))
    out = []
    for i in range(n_splits):
        tr = order[ts <= edges[i]]
        va = order[(ts > edges[i] + embargo) & (ts <= edges[i + 1])]
        if len(tr) and len(va):
            out.append((tr, va))
    return out


# --- data with a genuine group effect (eq. 43.1) ----------------------------
def make_grouped(n_groups=150, rows_per_group=20, icc=0.4, seed=1):
    """y = f(x) + u_g + eps, with the group-effect variance set to hit a
    target intraclass correlation (eq. 43.4).

    Crucially the FEATURES are group-correlated too: each customer has a
    characteristic profile and their rows scatter around it. That is what
    real grouped data looks like, and it is what lets a flexible model
    recognise 'this row belongs to a customer I have seen' and recall that
    customer's effect. Without it there is nothing for the model to key on
    and no leak to measure.
    """
    rs = np.random.default_rng(seed)
    sig_e = 1.0
    sig_u = np.sqrt(icc / (1 - icc)) * sig_e if icc < 1 else 10.0
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(groups)
    centre = rs.normal(0, 1.0, (n_groups, 6))[groups]     # customer profile
    X = centre + rs.normal(0, 0.25, (n, 6))               # tight scatter
    u = rs.normal(0, sig_u, n_groups)[groups]
    f = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3]
    y = f + u + rs.normal(0, sig_e, n)
    return X, y, groups, f


def fit_knn(Xtr, ytr, Xte, k=5):
    """A flexible model — capacity is what makes a grouped leak visible
    (Chapter 28 measured the same thing)."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("=" * 72)
print("what a grouped leak is worth, as a function of the ICC (eq. 43.4)")
print("=" * 72)
print("y = f(x) + u_g + eps. A random split puts rows from the SAME group on")
print("both sides, so the model can estimate u_g and reuse it. Production")
print("never allows that, because in production the group is new.\n")
print(f"{'ICC':>6} {'random-split RMSE':>19} {'grouped-split RMSE':>20} "
      f"{'optimism':>10} {'upper bound':>12}")
for icc in (0.0, 0.1, 0.3, 0.5, 0.7):
    X, y, g, f = make_grouped(icc=icc)
    tr, va = random_split(len(y), seed=3)
    r_rand = rmse(fit_knn(X[tr], y[tr], X[va]), y[va])
    tr, va = group_split(g, seed=3)
    r_grp = rmse(fit_knn(X[tr], y[tr], X[va]), y[va])
    # eq. 43.2 / 43.3: the ratio of error variances is ~ 1 + sig_u^2/sig_e^2
    predicted = np.sqrt(1 / (1 - icc)) if icc < 1 else np.inf
    print(f"{icc:>6.1f} {r_rand:>19.4f} {r_grp:>20.4f} "
          f"{r_grp / r_rand:>10.3f}x {predicted:>10.3f}x")

print("\nThe optimism column is the factor by which the random split")
print("understates the error. At ICC = 0 the two splits agree, exactly as")
print("they should — there is no group effect to leak. As the ICC rises the")
print("gap opens, monotonically, and nothing in the output warns you it is")
print("happening.")
print("\nThe last column is the simple bound from eqs. 43.2-43.3, and the")
print("measured optimism EXCEEDS it at every ICC. That is worth")
print("understanding rather than waving at, because the reason makes the")
print("problem worse than the algebra suggests.")
print("\nEq. 43.3 assumes that on an unseen group the model contributes")
print("nothing — it falls back to zero and simply eats the group effect. A")
print("nearest-neighbour model does something worse: it borrows the effect")
print("of whichever group happens to be nearby in feature space. That is not")
print("a missing estimate, it is a WRONG one, and it adds roughly a second")
print("factor of the group-effect variance rather than one.")
print("\nSo eq. 43.3 is a floor on the damage, not a ceiling. Any model that")
print("generalises across groups by similarity — which is most of them —")
print("will do worse on a genuinely new group than the algebra predicts.")

# --- effective sample size (eq. 43.5) ---------------------------------------
print("\n" + "=" * 72)
print("the other half: your confidence intervals are too narrow (eq. 43.5)")
print("=" * 72)
N, m = 100_000, 200
print(f"{'ICC':>6} {'nominal N':>11} {'effective N':>13} "
      f"{'CI too narrow by':>18}")
for icc in (0.0, 0.05, 0.1, 0.3, 0.5):
    n_eff = N / (1 + (m - 1) * icc)
    print(f"{icc:>6.2f} {N:>11,} {n_eff:>13,.0f} "
          f"{np.sqrt(N / n_eff):>17.1f}x")
print(f"\n({N:,} rows from {N // m:,} groups of {m}.) At an ICC of 0.3 the")
print("effective sample size is under 1,700 and every interval computed from")
print("the nominal N is about eight times too narrow. This is Chapter 22's")
print("design effect, arriving as an engineering problem.")

# --- time splits and the embargo (section 6.2) ------------------------------
print("\n" + "=" * 72)
print("time splits: the embargo, and why max(w, d) (section 6.2)")
print("=" * 72)


def make_temporal(n=4000, feature_window=30, label_delay=45, seed=2):
    """A trailing-window feature and a label that resolves `label_delay`
    days later — so a row at time t encodes information from t + delay."""
    rs = np.random.default_rng(seed)
    t = np.sort(rs.uniform(0, 700, n))
    latent = np.sin(t / 60.0) + rs.normal(0, 0.3, n)
    # trailing-window mean: the feature at t depends on the past w days
    feat = np.array([latent[(t > ti - feature_window) & (t <= ti)].mean()
                     if ((t > ti - feature_window) & (t <= ti)).any()
                     else 0.0 for ti in t])
    # the label depends on the FUTURE `delay` days, which is what leaks
    lab = np.array([latent[(t > ti) & (t <= ti + label_delay)].mean()
                    if ((t > ti) & (t <= ti + label_delay)).any()
                    else 0.0 for ti in t])
    y = 2.0 * lab + rs.normal(0, 0.25, n)
    # time is a feature here, which is realistic (recency, tenure, day of
    # week) and is also what lets a model exploit temporal adjacency at all
    X = np.column_stack([feat, t / 100.0, rs.normal(size=(n, 2))])
    return X, y, t, lab


Xt, yt, tt, lab_t = make_temporal()
print("feature window w = 30 days, label delay d = 45 days, so the embargo")
print("should be max(w, d) = 45 days.")
print("\nThe VALIDATION WINDOW IS HELD FIXED throughout — only the training")
print("side changes, by dropping rows within `embargo` days of the")
print("validation start. Varying both at once would compare scores on")
print("different data and measure nothing.\n")

cut_t = float(np.quantile(tt, 0.7))
val_mask = (tt > cut_t) & (tt <= cut_t + 120)      # fixed validation window
va = np.flatnonzero(val_mask)
print(f"fixed validation window: {int(val_mask.sum())} rows in "
      f"({cut_t:.0f}, {cut_t + 120:.0f}] days")

print("\nRather than read the leak off a downstream score — where it")
print("competes with every other effect — measure the CONTAMINATION")
print("directly. A training row is contaminated if its own windows reach")
print("into the validation period: its feature window (t-w, t] or its label")
print("window (t, t+d].\n")
print(f"{'embargo':>9} {'train rows':>11} {'feature-window':>15} "
      f"{'label-window':>13} {'either':>8}")
print(f"{'(days)':>9} {'':>11} {'overlap':>15} {'overlap':>13} {'':>8}")
W, DELAY = 30, 45
for emb in (0, 15, 30, 45, 60):
    tr = np.flatnonzero(tt <= cut_t - emb)
    ts = tt[tr]
    feat_bad = float(np.mean(ts + W > cut_t))      # feature window crosses cut
    lab_bad = float(np.mean(ts + DELAY > cut_t))   # label resolves after cut
    either = float(np.mean((ts + W > cut_t) | (ts + DELAY > cut_t)))
    print(f"{emb:>9} {len(tr):>11} {feat_bad:>15.4f} {lab_bad:>13.4f} "
          f"{either:>8.4f}")

print("\nBoth columns reach exactly zero at an embargo of 45 days, and not")
print("before: the feature-window contamination clears at 30 and the")
print("label-window contamination needs 45. The binding constraint is")
print("max(w, d), which is eq. 43.6 confirmed by counting rather than")
print("argued.")
print("\nNote which one binds. The feature window is the one people think")
print("of; the label window is longer here and is the one usually forgotten,")
print("because it is not visible anywhere in the feature engineering code.")

# ...and what the contamination is worth, using a model that can exploit it
print("\nAnd what the contamination buys a model that can use it — one with")
print("time as a feature, so temporally adjacent rows are its neighbours:\n")
print(f"{'embargo':>9} {'validation RMSE':>18} {'vs clean':>10}")
clean = None
scores = {}
for emb in (0, 15, 30, 45, 60):
    tr = np.flatnonzero(tt <= cut_t - emb)
    scores[emb] = rmse(fit_knn(Xt[tr], yt[tr], Xt[va]), yt[va])
for emb in (0, 15, 30, 45, 60):
    delta = scores[emb] - scores[45]
    print(f"{emb:>9} {scores[emb]:>18.4f} {delta:>+10.4f}")
print("\nThe embargoed score is the honest one. Whether the un-embargoed")
print("score looks better or worse here depends on a second effect pulling")
print("the other way — the dropped rows are also the most RECENT, and")
print("recency helps a time-indexed model. That confound is exactly why the")
print("contamination count above is the better evidence: it measures the")
print("mechanism, not a downstream number that several effects move at once.")
