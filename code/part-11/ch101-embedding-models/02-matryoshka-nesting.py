# -*- coding: utf-8 -*-
# Extracted from: Chapter 101 — Embedding Models: Training, Choosing, and Evaluating
# Source: src/.../ch101-embedding-models.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Dimension as a serving-time choice: nested training against truncation.

Three ways to obtain a k-dimensional embedding for k in {4, 8, 16, 32}:

  plain, truncated  -- train once at 32, keep the first k coordinates, re-normalise
  matryoshka        -- train once with eq:matryoshka-loss over all four widths,
                       then keep the first k
  trained at dim    -- train a separate model at each k (the upper bound, and
                       four times the training cost)
"""
import numpy as np

rng = np.random.default_rng(23)

D_LAT, D_OBS, D_EMB = 40, 96, 32
N_TRAIN, N_TEST, TAU = 6000, 3000, 0.07
NEST = [4, 8, 16, 32]

proj = rng.normal(size=(D_LAT, D_OBS)) / np.sqrt(D_LAT)
offset = rng.normal(size=D_OBS) * 2.0
q_shift = rng.normal(size=D_OBS) * 0.4


def views(z):
    b = z @ proj + offset
    return (b + q_shift + rng.normal(scale=0.55, size=b.shape),
            b + rng.normal(scale=0.55, size=b.shape))


Q, D = views(rng.normal(size=(N_TRAIN, D_LAT)))
Q_te, D_te = views(rng.normal(size=(N_TEST, D_LAT)))


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def grad(W, A, P, dims):
    """Gradient of the sum of InfoNCE over the given prefixes (eq:matryoshka-loss).
    Passing a single-element list recovers ordinary contrastive training."""
    dW = np.zeros_like(W)
    Za_full, Zp_full = A @ W, P @ W
    for d in dims:
        Za_r, Zp_r = Za_full[:, :d], Zp_full[:, :d]
        na = np.linalg.norm(Za_r, axis=1, keepdims=True)
        np_ = np.linalg.norm(Zp_r, axis=1, keepdims=True)
        Za, Zp = Za_r / na, Zp_r / np_
        logits = Za @ Zp.T / TAU
        logits -= logits.max(axis=1, keepdims=True)
        Pr = np.exp(logits)
        Pr /= Pr.sum(axis=1, keepdims=True)
        G = Pr.copy()
        G[np.arange(len(G)), np.arange(len(G))] -= 1.0
        G /= len(G) * TAU

        def through_norm(dZ, Zx, nx):
            return (dZ - Zx * np.sum(dZ * Zx, axis=1, keepdims=True)) / nx

        # Only the first d columns of W receive gradient from this prefix.
        dW[:, :d] += (A.T @ through_norm(G @ Zp, Za, na)
                      + P.T @ through_norm(G.T @ Za, Zp, np_))
    return dW


def train(dims, steps=1500, batch=256, lr=0.5):
    W = rng.normal(scale=0.05, size=(D_OBS, D_EMB))
    for _ in range(steps):
        i = rng.choice(N_TRAIN, batch, replace=False)
        W -= lr * grad(W, Q[i], D[i], dims)
    return W


def accuracy(W, d):
    a, b = unit((Q_te @ W)[:, :d]), unit((D_te @ W)[:, :d])
    return float(np.mean(np.argmax(a @ b.T, axis=1) == np.arange(len(a))))


W_plain = train([D_EMB])                       # one model, full width only
W_mrl = train(NEST)                            # one model, all widths
W_sep = {d: train([d]) for d in NEST}          # four models

print(f"{'dim':>5}{'plain, truncated':>19}{'matryoshka':>13}{'trained at dim':>16}"
      f"{'index memory':>15}")
print("-" * 68)
for d in NEST:
    rel = d / D_EMB
    print(f"{d:>5}{accuracy(W_plain, d):>19.4f}{accuracy(W_mrl, d):>13.4f}"
          f"{accuracy(W_sep[d], d):>16.4f}{rel:>14.0%}")

print("""
Read across each row. The matryoshka column tracks the trained-at-dim column
closely at every width -- one model reproduces what four separate models
achieve, which is the entire claim, and it means the width can be chosen after
training rather than before.

Now read the plain-truncated column against it. Truncating an ordinarily-trained
embedding is much worse at the narrow end. Nothing in ordinary contrastive
training asks the first eight coordinates to be useful ALONE, so they are not;
the information is spread across all thirty-two, and cutting the vector destroys
it. Matryoshka's only difference is that eq:matryoshka-loss asks each prefix to
work on its own.

The last row is the price, and it should be stated rather than hidden: at full
width the matryoshka model is slightly WORSE than one trained at full width
alone. The nesting is not free -- the widest representation shares capacity with
every narrower one. That trade is usually worth it, because the memory column
shows what the narrow end buys: an eight-dimensional index is a quarter of the
memory and a quarter of the distance arithmetic, forever, on every query.""")
