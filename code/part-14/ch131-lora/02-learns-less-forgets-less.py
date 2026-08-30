# -*- coding: utf-8 -*-
# Extracted from: Chapter 131 — LoRA and the Low-Rank Hypothesis
# Source: src/.../ch131-lora.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""LoRA learns less and forgets less. Both halves, from one sweep.

cite:biderman2024loralearnsless is the controlled comparison the folklore needed.
The finding is not that LoRA matches full fine-tuning, and not that it is worse.
It is a TRADE with a single mechanism behind both halves: LoRA constrains how far
the weights can move, so it captures less of the new task AND disturbs less of
the old one (eq:rank-constrains-movement).

This listing measures both on one sweep. A base model is optimal for task A; it
is then adapted to task B at a range of LoRA ranks and with an unconstrained
fine-tune, and BOTH tasks are evaluated after. Task B's ideal update has a
decaying spectrum rather than a hard rank, which is what a real adaptation looks
like -- so there is no rank at which the trade disappears.
"""
import numpy as np

rng = np.random.default_rng(151)

D = 96
N, STEPS, LR = 5000, 700, 0.05


def spectrum_delta(decay=1.0):
    """An update whose singular values decay smoothly: no hard rank, so every
    additional rank captures a little more and moves a little further."""
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    V, _ = np.linalg.qr(rng.normal(size=(D, D)))
    s = 1.0 / (1.0 + np.arange(D)) ** decay
    s = s / s[0] * 1.1
    return U @ np.diag(s) @ V.T


W_A = rng.normal(size=(D, D)) / np.sqrt(D)          # base model: optimal for A
DELTA = spectrum_delta(decay=1.0)
W_B = W_A + DELTA                                    # target for task B

X_B = rng.normal(size=(N, D)); Y_B = X_B @ W_B
X_A = rng.normal(size=(N, D)); Y_A = X_A @ W_A


def rel(pred, Y):
    return float(np.linalg.norm(pred - Y) / np.linalg.norm(Y))


def fit_lora(rank):
    A = rng.normal(size=(D, rank)) / np.sqrt(D)
    B = np.zeros((rank, D))
    for _ in range(STEPS):
        G = 2.0 * (X_B @ (W_A + A @ B) - Y_B) / N
        GD = X_B.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= LR * gA
        B -= LR * gB
    return A @ B


def fit_full():
    return np.linalg.lstsq(X_B, Y_B - X_B @ W_A, rcond=None)[0]


RANKS = (1, 2, 4, 8, 16, 32, 64)

print(f"{D}x{D} layer. Base model is exact on task A. Task B's ideal update has")
print("a decaying spectrum, so no finite rank captures all of it.\n")
print(f"{'adaptation':>14}{'trainable':>12}{'task B':>10}{'task A':>10}"
      f"{'update':>10}{'captured':>11}")
print(f"{'':>14}{'params':>12}{'error':>10}{'error':>10}{'norm':>10}"
      f"{'of delta':>11}")
print("-" * 68)

rows = {}
full_norm = np.linalg.norm(DELTA)
for r in RANKS:
    d = fit_lora(r)
    b = rel(X_B @ (W_A + d), Y_B)
    a = rel(X_A @ (W_A + d), Y_A)
    rows[r] = (b, a, float(np.linalg.norm(d)))
    print(f"{'LoRA r=' + str(r):>14}{2 * D * r:>12,}{b:>10.4f}{a:>10.4f}"
          f"{np.linalg.norm(d):>10.3f}{np.linalg.norm(d)/full_norm:>11.2f}")

dfull = fit_full()
bf = rel(X_B @ (W_A + dfull), Y_B)
af = rel(X_A @ (W_A + dfull), Y_A)
rows["full"] = (bf, af, float(np.linalg.norm(dfull)))
print(f"{'full FT':>14}{D * D:>12,}{bf:>10.4f}{af:>10.4f}"
      f"{np.linalg.norm(dfull):>10.3f}{np.linalg.norm(dfull)/full_norm:>11.2f}")

r1, r64, fl = rows[1], rows[64], rows["full"]
print(f"""
Read the two error columns together, because reading either alone produces a
wrong conclusion.

Down the task-B column, error falls monotonically with rank: {r1[0]:.4f} at rank
1 to {r64[0]:.4f} at rank 64, and {fl[0]:.4f} unconstrained. LoRA LEARNS LESS,
and it learns less by an amount that shrinks as rank grows but never reaches
zero, because task B's ideal update has a decaying spectrum and any finite rank
truncates it (eq:eckart-young).

Down the task-A column, error RISES with rank, in the same order: {r1[1]:.4f} at
rank 1 to {r64[1]:.4f} at rank 64 and {fl[1]:.4f} unconstrained. LoRA FORGETS
LESS, and it forgets less for exactly the same reason it learns less.

That is the whole finding, and the two halves are not independent observations
that happen to point the same way. They are one mechanism seen twice, and the
update-norm column is the mechanism: {r1[2]:.3f} at rank 1 rising to
{fl[2]:.3f} unconstrained. Rank bounds how far the weights can travel from the
base model (eq:rank-constrains-movement), and how far they travel determines both
how much of the new task they can reach and how much of the old one they disturb.

So "is LoRA as good as full fine-tuning?" is the wrong question, because it has
two answers pointing in opposite directions. The right question has two halves
that can be asked separately: how much NEW capability does this task require, and
how much OLD capability must survive?

That reframing is what makes the choice decidable. A task needing a large,
genuinely novel capability wants rank -- or full fine-tuning -- and will pay in
forgetting. A task that adjusts style or format on a model whose general ability
must be preserved wants low rank, and the capability it gives up was capability it
did not need. The two are different points on one curve rather than competing
techniques.

Note the last column, which prices the parameter argument honestly. Rank 64 has
{2 * D * 64 / (D * D):.1f} times the trainable parameters of a full fine-tune of
this layer, because 2*D*r exceeds D*D once r passes D/2. LoRA's parameter saving
is real at small rank and evaporates at large rank, so "10,000x fewer parameters"
is a statement about a particular rank on a particular model shape rather than a
property of the method. If your task needs high rank, LoRA is not saving you
much, and the choice should be made on forgetting instead.""")
