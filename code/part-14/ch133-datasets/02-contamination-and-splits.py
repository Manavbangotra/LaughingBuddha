# -*- coding: utf-8 -*-
# Extracted from: Chapter 133 — Dataset Creation for Fine-Tuning
# Source: src/.../ch133-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Contamination: why a random split lies, and why deduplication does not fix it.

Fine-tuning data is rarely a set of independent examples. It arrives in clusters
-- the same question asked three ways, a template instantiated per customer, a
document and its summary. Split that pool at random and members of one cluster
land on both sides, so the held-out score measures memorisation and reports it as
generalisation (eq:leakage-inflates).

The standard response is deduplication against a similarity threshold. This
listing measures how much of the problem that removes, and finds two failures at
once: what the threshold catches is not what leaks, and what it discards is not a
random sample of the test set.
"""
import numpy as np

rng = np.random.default_rng(167)

D, NF = 8, 400             # input dim, random-feature width (enough to memorise)
N_CLUST = 1500
REPS, REPS2 = 20, 8


def make_pool(dup_rate, far_share=0.5):
    """Clusters share an ANSWER. Some members are surface-similar to each other
    (a paraphrase), some are surface-DIFFERENT but still share the answer (the
    same question posed from another angle). Both are contamination."""
    centres = rng.normal(size=(N_CLUST, D))
    w = rng.normal(size=D)
    y_c = (np.sin(1.2 * centres @ w / np.sqrt(D)) + 0.5 * centres[:, 0]
           > 0).astype(int)

    xs, ys, gs = [], [], []
    for g in range(N_CLUST):
        n_extra = rng.integers(1, 4) if rng.random() < dup_rate else 0
        for j in range(1 + n_extra):
            eps = 0.05 if (j == 0 or rng.random() > far_share) else 0.9
            xs.append(centres[g] + eps * rng.normal(size=D))
            ys.append(y_c[g]); gs.append(g)
    return np.array(xs), np.array(ys), np.array(gs)


W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def fit_predict(Xtr, ytr, Xte, lam=3e-5):
    """Low ridge: the model CAN memorise, which is the point."""
    P = feat(Xtr)
    A = P.T @ P + lam * len(Xtr) * np.eye(NF)
    c = np.linalg.solve(A, P.T @ (2.0 * ytr - 1))
    return (feat(Xte) @ c > 0).astype(int)


def split_random(n, frac=0.3):
    idx = rng.permutation(n)
    k = int(frac * n)
    return idx[k:], idx[:k]


def split_group(g, frac=0.3):
    groups = np.unique(g)
    gp = rng.permutation(groups)
    held = set(gp[:int(frac * len(groups))].tolist())
    mask = np.array([x in held for x in g])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


print("A pool of clustered examples: members of a cluster share the answer.\n")
print(f"{'duplication':>12}{'random split':>15}{'group split':>14}"
      f"{'inflation':>12}")
print(f"{'rate':>12}{'reports':>15}{'reports':>14}{'':>12}")
print("-" * 53)

table = {}
for dr in (0.0, 0.1, 0.25, 0.5, 0.9):
    ar, ag = [], []
    for _ in range(REPS):
        X, y, g = make_pool(dr)
        tr, te = split_random(len(y))
        ar.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).mean())
        tr, te = split_group(g)
        ag.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).mean())
    a_rand, a_grp = float(np.mean(ar)), float(np.mean(ag))
    table[dr] = (a_rand, a_grp)
    print(f"{dr:>12.0%}{a_rand:>15.3f}{a_grp:>14.3f}{a_rand - a_grp:>+12.3f}")

print("\n\nDoes decontamination by distance threshold recover the truth?\n")
truth = table[0.5][1]
print(f"{'threshold':>10}{'% of test':>12}{'reports':>10}{'error vs':>13}"
      f"{'share of kept':>15}")
print(f"{'':>10}{'discarded':>12}{'':>10}{'group split':>13}"
      f"{'still leaked':>15}")
print("-" * 60)

TAUS = (0.0, 0.3, 0.6, 1.0, 2.0)
agg = {t: [[], [], []] for t in TAUS}
for _ in range(REPS2):
    X, y, g = make_pool(0.5)
    tr, te = split_random(len(y))
    d2 = ((X[te] ** 2).sum(1)[:, None] + (X[tr] ** 2).sum(1)[None, :]
          - 2.0 * X[te] @ X[tr].T)
    d = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    leaked = np.isin(g[te], g[tr])          # ground truth: shares a cluster
    for tau in TAUS:
        keep = d > tau
        if keep.sum() < 40:
            continue
        a = (fit_predict(X[tr], y[tr], X[te][keep]) == y[te][keep]).mean()
        agg[tau][0].append(1 - keep.mean())
        agg[tau][1].append(a)
        agg[tau][2].append(leaked[keep].mean())

rows = {}
for tau in TAUS:
    rm, ac, rs = (float(np.mean(v)) for v in agg[tau])
    rows[tau] = (rm, ac, rs)
    print(f"{tau:>10.1f}{rm:>12.1%}{ac:>10.3f}{ac - truth:>+13.3f}{rs:>15.1%}")

t0, t3, t20 = rows[0.0], rows[0.3], rows[2.0]
print(f"""
The first table is the size of the problem, and its top row is the control: with
no duplication the two splits agree to {abs(table[0.0][0]-table[0.0][1]):.3f}, as
they must, because with singleton clusters the two procedures are the same
procedure.

Add clustering and they separate, monotonically. At a 50% duplication rate the
random split reports {table[0.5][0]:.3f} where the group split reports
{table[0.5][1]:.3f}, an inflation of {table[0.5][0]-table[0.5][1]:+.3f}; at 90% it
is {table[0.9][0]-table[0.9][1]:+.3f}.

Nothing about the model changed between those two numbers. The same features, the
same training procedure, the same pool -- a different assignment of clusters to
sides. The random split measures how well the model recalls answers it was shown
and reports it as how well the model answers new questions
(eq:leakage-inflates).

This is the most common way a fine-tuning result turns out to be fictional, and
it is invisible from the inside: the loss curve looks healthy, the held-out score
looks good, and the number is simply about something other than what it claims.

The second table is the part that decides what to do about it, and it fails in
two directions at once.

Read the last column first. At a threshold of 0.3, {t3[0]:.0%} of the test set has
been discarded and {t3[2]:.0%} of what remains STILL shares a cluster with a
training example. The threshold removed the surface-similar duplicates and left
the rest, because half of each cluster's members were built to be surface-
DIFFERENT while sharing the answer -- the same question from another angle, a
different document about the same fact. Those are contamination by every meaning
that matters, and they sit far away in input space, so no threshold on surface
distance will find them (eq:distance-misses-semantics).

The bottom row settles it. Push the threshold to 2.0 and {t20[0]:.0%} of the test
set is thrown away -- and the leaked share of what remains goes UP, to
{t20[2]:.0%}. That is not a plateau, it is the wrong direction. An aggressive
distance filter preferentially removes the surface-similar contamination and
preferentially KEEPS the semantic contamination, so the harder you scrub, the
more concentrated in real leakage the surviving test set becomes.

Now read the error column, which is the failure people do not anticipate.
Decontamination does not converge on the truth from above; it swings PAST it.
Undecontaminated, the split over-reports by {t0[1]-truth:+.3f}. After thresholding
it reports {t3[1]-truth:+.3f} against the group split, while half the remaining
test set is still leaked.

Both effects are present at once and they partly cancel, which is worse than
either alone, because the cancellation is accidental. Removing test examples that
are near training examples does not remove a random sample of the test set: it
removes the ones the model finds easy, so what survives is harder than the task
is. The number you get is a leaked score on an unrepresentative subset, and there
is no reason for it to land anywhere in particular.

So the fix is not a better threshold. It is to split by GROUP, using the
provenance you had before the examples became vectors: the document they came
from, the customer, the template, the source URL, the ticket. That information is
free at collection time and largely unrecoverable afterwards, which is the whole
practical lesson of this chapter.

Write down the group key when you build the dataset. If you did not, the honest
options are to reconstruct provenance or to report that your held-out numbers
carry an unknown bias -- and the second is far more common than the literature
would suggest.""")
