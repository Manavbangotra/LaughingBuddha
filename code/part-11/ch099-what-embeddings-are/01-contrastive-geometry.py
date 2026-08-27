# -*- coding: utf-8 -*-
# Extracted from: Chapter 99 — What Embeddings Are: Representation Learning Revisited
# Source: src/.../ch099-what-embeddings-are.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Contrastive training on a toy corpus: what InfoNCE does to geometry.

Each ITEM has a latent vector. It is observed twice -- once as a "query" view
and once as a "document" view -- through a shared random projection, plus:

  * a large constant OFFSET shared by everything, which is what makes raw
    observations anisotropic, exactly as a language model's hidden states are;
  * a constant Q_SHIFT applied to queries only, which is the query/document
    asymmetry of eq:ranking-constraint in its simplest possible form.

Retrieval task: given a query view, find its own document view among 1,200
candidates. We compare four encoders on identical data and report anisotropy
(eq:mean-cosine), the scale-free positive margin, alignment (eq:alignment),
uniformity (eq:uniformity), and accuracy@1.
"""
import numpy as np

rng = np.random.default_rng(11)

D_LAT, D_OBS, D_EMB = 12, 64, 16
N_TRAIN, N_TEST = 6000, 1200
TAU = 0.07

proj = rng.normal(size=(D_LAT, D_OBS)) / np.sqrt(D_LAT)
offset = rng.normal(size=D_OBS) * 3.0          # shared by queries AND documents
q_shift = rng.normal(size=D_OBS) * 0.8         # queries only


def sample(n):
    z = rng.normal(size=(n, D_LAT))
    base = z @ proj + offset
    q = base + q_shift + rng.normal(scale=0.55, size=(n, D_OBS))
    d = base + rng.normal(scale=0.55, size=(n, D_OBS))
    return q, d


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


Q_tr, D_tr = sample(N_TRAIN)
Q_te, D_te = sample(N_TEST)
mu_all = np.concatenate([Q_tr, D_tr]).mean(axis=0)     # one mean for everything
mu_q, mu_d = Q_tr.mean(axis=0), D_tr.mean(axis=0)      # one mean per side


def mean_cosine(emb, n=800):
    """Mean pairwise cosine of the corpus (eq:mean-cosine)."""
    i, j = rng.choice(len(emb), n), rng.choice(len(emb), n)
    k = i != j
    return float(np.mean(np.sum(emb[i[k]] * emb[j[k]], axis=1)))


def margin(qe, de):
    """Mean positive cosine minus mean random cosine: scale-free dynamic range."""
    return float(np.mean(np.sum(qe * de, axis=1))) - mean_cosine(de)


def alignment(qe, de):
    """E||f(q) - f(d+)||^2 over true pairs (eq:alignment)."""
    return float(np.mean(np.sum((qe - de) ** 2, axis=1)))


def uniformity(emb, n=800):
    """log E exp(-2||f(x) - f(y)||^2) over random pairs (eq:uniformity)."""
    i, j = rng.choice(len(emb), n), rng.choice(len(emb), n)
    k = i != j
    d2 = np.sum((emb[i[k]] - emb[j[k]]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * d2))))


def accuracy(qe, de):
    """Is a query's nearest document its own partner, out of all N_TEST?"""
    return float(np.mean(np.argmax(qe @ de.T, axis=1) == np.arange(len(qe))))


# ---- The trained encoder: a linear map fitted with InfoNCE (eq:infonce) ------
W = rng.normal(scale=0.05, size=(D_OBS, D_EMB))


def infonce_step(W, batch_q, batch_d, lr):
    """One InfoNCE step with in-batch negatives; explicit gradient, no autograd."""
    Zq_raw, Zd_raw = batch_q @ W, batch_d @ W
    nq = np.linalg.norm(Zq_raw, axis=1, keepdims=True)
    nd = np.linalg.norm(Zd_raw, axis=1, keepdims=True)
    Zq, Zd = Zq_raw / nq, Zd_raw / nd

    logits = Zq @ Zd.T / TAU
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(np.clip(np.diag(P), 1e-12, None)))

    G = P.copy()                                   # dL/dlogits (eq:infonce-gradient)
    G[np.arange(len(G)), np.arange(len(G))] -= 1.0
    G /= len(G) * TAU
    dZq, dZd = G @ Zd, G.T @ Zq

    def through_norm(dZ, Z, n):                    # backprop through L2 normalise
        return (dZ - Z * np.sum(dZ * Z, axis=1, keepdims=True)) / n

    dW = (batch_q.T @ through_norm(dZq, Zq, nq)
          + batch_d.T @ through_norm(dZd, Zd, nd))
    return W - lr * dW, loss


BATCH, STEPS, LR = 256, 3000, 0.5
for step in range(STEPS + 1):
    idx = rng.choice(N_TRAIN, BATCH, replace=False)
    W, loss = infonce_step(W, Q_tr[idx], D_tr[idx], LR)
    if step % 1000 == 0:
        print(f"  step {step:4d}  InfoNCE loss {loss:.4f}"
              f"   (chance = log {BATCH} = {np.log(BATCH):.3f})")

encoders = {
    "raw (by-product)":     (unit(Q_te),        unit(D_te)),
    "centred (global mean)": (unit(Q_te - mu_all), unit(D_te - mu_all)),
    "centred (per side)":   (unit(Q_te - mu_q), unit(D_te - mu_d)),
    "trained (InfoNCE)":    (unit(Q_te @ W),    unit(D_te @ W)),
}

print(f"\n{'encoder':<22}{'mean cos':>10}{'margin':>9}{'align':>8}"
      f"{'uniform':>9}{'acc@1':>8}")
print("-" * 66)
for name, (qe, de) in encoders.items():
    print(f"{name:<22}{mean_cosine(de):>10.4f}{margin(qe, de):>9.4f}"
          f"{alignment(qe, de):>8.4f}{uniformity(de):>9.4f}{accuracy(qe, de):>8.4f}")

print("""
Read the ALIGN column last, and read it sceptically. The raw embeddings have the
BEST alignment of the four -- and the worst retrieval but one. That is not a
paradox, it is the point: alignment is an absolute squared distance, and raw
vectors are crammed into a narrow cone (mean cosine 0.90) where EVERY pair is
close, positives included. Alignment measured on its own rewards collapse.

The MARGIN column is the scale-free version and it ranks the encoders correctly.
It is the gap eq:ranking-constraint actually cares about: how much closer a true
pair is than a random one.

Now compare the two centrings, which is the result worth taking away. Removing
one global mean fixes the anisotropy -- and makes retrieval WORSE than doing
nothing. Removing a mean per side fixes it and improves retrieval substantially.
The difference is Q_SHIFT: queries and documents have different distributions,
the global mean centres neither of them, and once the large shared offset is
gone that asymmetry is a much larger fraction of what remains. Anisotropy was
masking it.

So the cheap geometric fix is real but conditional. It requires knowing that
queries and documents are different populations -- which is the same fact that
motivates two towers and query/passage prefixes in section 5. And only the
TRAINED encoder closes the gap, because only it was told which pairs go
together.""")
