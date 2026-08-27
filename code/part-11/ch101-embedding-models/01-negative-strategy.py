# -*- coding: utf-8 -*-
# Extracted from: Chapter 101 — Embedding Models: Training, Choosing, and Evaluating
# Source: src/.../ch101-embedding-models.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What the negatives teach: random against mined-hard, on a tight bottleneck.

Each item's latent vector has two parts:

  COARSE -- a high-variance group identity (60 groups). Separating two items from
            DIFFERENT groups only requires these directions.
  FINE   -- a low-variance individual identity. Separating two items from the
            SAME group requires these.

The encoder is deliberately given fewer output dimensions than the latent space,
so it must CHOOSE which directions to preserve -- eq:capacity-allocation. We
report two evaluations: retrieval against the whole test corpus, and retrieval
restricted to the query's own group, which is the fine-grained task.

Three runs per strategy, because the variance is itself part of the result.
"""
import numpy as np
import statistics

rng = np.random.default_rng(17)

COARSE, FINE, OBS, EMB = 8, 6, 48, 5      # EMB < COARSE + FINE: a real bottleneck
C_SCALE, F_SCALE = 4.0, 0.35
N, N_DUP, N_GROUP, TAU = 4000, 1000, 60, 0.07

proj = rng.normal(size=(COARSE + FINE, OBS)) / np.sqrt(COARSE + FINE)
offset = rng.normal(size=OBS) * 2.0
q_shift = rng.normal(size=OBS) * 0.4


def latents(n):
    g = rng.integers(0, N_GROUP, size=n)
    coarse = rng.normal(size=(N_GROUP, COARSE))[g] * C_SCALE
    fine = rng.normal(size=(n, FINE)) * F_SCALE
    return np.hstack([coarse, fine]), g


Z_tr, G_tr = latents(N)
# Some items have a near-duplicate: SAME fine identity (so genuinely equivalent),
# DIFFERENT coarse surface. These are the unlabelled relevant documents of
# eq:false-negative -- the ones a filter on labelled positives cannot see.
dup_src = rng.choice(N, N_DUP, replace=False)
new_coarse = rng.normal(size=(N_DUP, COARSE)) * C_SCALE
Z_dup = np.hstack([new_coarse,
                   Z_tr[dup_src][:, COARSE:] + rng.normal(scale=0.02,
                                                          size=(N_DUP, FINE))])
Z = np.vstack([Z_tr, Z_dup])
partner = np.full(len(Z), -1)
partner[N:] = dup_src
partner[dup_src] = np.arange(N, N + N_DUP)


def views(z):
    b = z @ proj + offset
    return (b + q_shift + rng.normal(scale=0.10, size=b.shape),
            b + rng.normal(scale=0.10, size=b.shape))


Q, D = views(Z)
Z_te, G_te = latents(1500)
Q_te, D_te = views(Z_te)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def train(strategy, steps=1200, batch=128, n_neg=32, lr=0.5):
    """Fit a linear encoder with eq:infonce-explicit; returns W and the
    measured false-negative rate among the negatives actually used."""
    W = rng.normal(scale=0.05, size=(OBS, EMB))
    fn_hits, fn_slots = 0, 0
    for _ in range(steps):
        anchors = rng.choice(len(Z), batch, replace=False)

        if strategy == "random":
            neg = rng.integers(0, len(Z), size=(batch, n_neg))
        else:
            pool = rng.choice(len(Z), 800, replace=False)
            sc = unit(Q[anchors] @ W) @ unit(D[pool] @ W).T
            sc[pool[None, :] == anchors[:, None]] = -np.inf      # drop self
            neg = pool[np.argpartition(-sc, n_neg, axis=1)[:, :n_neg]]

        pm = partner[anchors]
        fn_hits += int(np.sum(neg == pm[:, None]))
        fn_slots += neg.size

        A, P, Ng = Q[anchors], D[anchors], D[neg]
        Za_r, Zp_r, Zn_r = A @ W, P @ W, Ng @ W
        na = np.linalg.norm(Za_r, axis=1, keepdims=True)
        np_ = np.linalg.norm(Zp_r, axis=1, keepdims=True)
        nn = np.linalg.norm(Zn_r, axis=2, keepdims=True)
        Za, Zp, Zn = Za_r / na, Zp_r / np_, Zn_r / nn

        logits = np.concatenate([np.sum(Za * Zp, axis=1, keepdims=True),
                                 np.einsum('bd,bnd->bn', Za, Zn)], axis=1) / TAU
        logits -= logits.max(axis=1, keepdims=True)
        Pr = np.exp(logits)
        Pr /= Pr.sum(axis=1, keepdims=True)

        g = Pr.copy()
        g[:, 0] -= 1.0                       # the positive is column 0
        g /= batch * TAU
        dZa = g[:, 0:1] * Zp + np.einsum('bn,bnd->bd', g[:, 1:], Zn)
        dZp = g[:, 0:1] * Za
        dZn = g[:, 1:, None] * Za[:, None, :]

        def through_norm(dZ, Zx, nx):
            return (dZ - Zx * np.sum(dZ * Zx, axis=-1, keepdims=True)) / nx

        W -= lr * (A.T @ through_norm(dZa, Za, na)
                   + P.T @ through_norm(dZp, Zp, np_)
                   + np.einsum('bnd,bne->de', Ng, through_norm(dZn, Zn, nn)))
    return W, fn_hits / fn_slots


def evaluate(W):
    a, b = unit(Q_te @ W), unit(D_te @ W)
    S = a @ b.T
    overall = float(np.mean(np.argmax(S, axis=1) == np.arange(len(a))))
    M = S.copy()
    M[G_te[None, :] != G_te[:, None]] = -np.inf     # same-group candidates only
    within = float(np.mean(np.argmax(M, axis=1) == np.arange(len(a))))
    return overall, within


print(f"{'negatives':<16}{'acc, whole corpus':>19}{'acc, within group':>19}"
      f"{'mined false-neg':>17}")
print("-" * 71)
rows = {}
for strategy in ["random", "mined hard"]:
    overalls, withins, fns = [], [], []
    for _ in range(3):
        W, fn_rate = train(strategy)
        o, w = evaluate(W)
        overalls.append(o)
        withins.append(w)
        fns.append(fn_rate)
    sd = statistics.pstdev(withins)
    rows[strategy] = (statistics.mean(overalls), statistics.mean(withins), sd)
    print(f"{strategy:<16}{statistics.mean(overalls):>19.4f}"
          f"{statistics.mean(withins):>19.4f}{100 * statistics.mean(fns):>16.2f}%"
          f"   (within-group sd {sd:.4f})")

d_all = 100 * (rows["mined hard"][0] - rows["random"][0])
d_within = 100 * (rows["mined hard"][1] - rows["random"][1])
sd_ratio = rows["random"][2] / rows["mined hard"][2]

print(f"""
Two numbers, and the second one is the lesson.

Mined negatives beat random ones on the whole-corpus task by {d_all:.1f} points
-- worth having. On the WITHIN-GROUP task they beat them by {d_within:.1f}, about
{d_within / d_all:.1f} times as much. That gap between the gaps is
eq:capacity-allocation: with a bottleneck this
tight the encoder must choose which latent directions to keep, random negatives
only ever ask it to separate different groups, and the high-variance coarse
directions are enough for that. Only mined negatives force it to spend capacity
on the fine directions.

The practical consequence is uncomfortable. If your evaluation set is drawn
uniformly from the corpus, most of its pairs are easy, and you will measure the
{d_all:.1f}-point version of an intervention that delivered {d_within:.1f} on the
queries users actually send -- the confusable ones.

Now the variance column. Random negatives are not merely worse on average; their
run-to-run spread is {sd_ratio:.1f} times larger. Whether the model ever learns
the fine distinction depends on how many informative negatives happened to be
drawn. Mined negatives make that deterministic. A training procedure whose
outcome varies this much between seeds is not one you can A/B test cheaply.

Finally, the false-negative column. These are genuinely equivalent documents --
same fine identity, different surface -- and mining picks them up at well under
one percent of negative slots. Not because the filter caught them: there is no
filter here. Mining scores candidates under the CURRENT model, and a model whose
capacity has gone to surface features cannot see that two differently-worded
documents say the same thing, so it never ranks them highly enough to mine. That
is why false negatives are a late-training and iterative-mining hazard rather
than a round-one one -- and why the number to watch is this rate, per round.""")
