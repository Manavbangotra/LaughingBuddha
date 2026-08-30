# -*- coding: utf-8 -*-
# Extracted from: Chapter 131 — LoRA and the Low-Rank Hypothesis
# Source: src/.../ch131-lora.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The low-rank hypothesis, tested rather than assumed.

cite:hu2021lora's claim is precise and often paraphrased into something weaker.
It is NOT that a model's weights are low rank -- they are not. It is that the
CHANGE required to adapt a pretrained model to a new task has low intrinsic rank,
so the update can be written as a product of two thin matrices
(eq:lora-parameterisation).

That is a claim about tasks, and it is testable. This listing builds adaptation
problems whose true update has a known rank, fits LoRA at a range of ranks, and
finds the knee. Then it does the reverse: fine-tunes without any rank constraint
and inspects the singular value spectrum of the resulting update, to see what rank
the task actually wanted (eq:effective-rank).
"""
import numpy as np

rng = np.random.default_rng(149)

D_IN, D_OUT = 96, 96
N_TRAIN, N_TEST = 4000, 4000
STEPS, LR = 700, 0.05


def make_problem(true_rank, noise=0.05):
    """A frozen base map W0, and a target that differs from it by a delta of
    exactly `true_rank`. Adaptation means recovering that delta."""
    W0 = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
    A = rng.normal(size=(D_IN, true_rank)) / np.sqrt(D_IN)
    B = rng.normal(size=(true_rank, D_OUT)) / np.sqrt(true_rank)
    delta = (A @ B) * 0.8
    X = rng.normal(size=(N_TRAIN, D_IN))
    Y = X @ (W0 + delta) + noise * rng.normal(size=(N_TRAIN, D_OUT))
    Xt = rng.normal(size=(N_TEST, D_IN))
    Yt = Xt @ (W0 + delta)
    return W0, delta, X, Y, Xt, Yt


def fit_lora(W0, X, Y, rank):
    """Train only A and B, with B initialised at zero so the adapted model
    starts exactly at the base model (eq:lora-init)."""
    A = rng.normal(size=(D_IN, rank)) / np.sqrt(D_IN)
    B = np.zeros((rank, D_OUT))
    for _ in range(STEPS):
        pred = X @ (W0 + A @ B)
        G = 2.0 * (pred - Y) / len(X)          # dL/dpred
        GD = X.T @ G                            # dL/d(delta)
        gA, gB = GD @ B.T, A.T @ GD
        A -= LR * gA
        B -= LR * gB
    return A @ B


def fit_full(W0, X, Y):
    """No rank constraint: solve for the delta directly."""
    return np.linalg.lstsq(X, Y - X @ W0, rcond=None)[0]


def rel_err(pred, Y):
    return float(np.linalg.norm(pred - Y) / np.linalg.norm(Y))


def effective_rank(M, thresh=0.99):
    """How many singular values are needed to capture `thresh` of the energy --
    the rank the update actually used (eq:effective-rank)."""
    s = np.linalg.svd(M, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, thresh) + 1)


RANKS = (1, 2, 4, 8, 16, 32)
TRUE = (2, 8, 32)

print(f"{D_IN}x{D_OUT} layer. The target differs from the base by a delta of "
      f"known rank.\n")
print(f"{'true rank':>10}{'':>3}" + "".join(f"{'LoRA r=' + str(r):>12}"
                                            for r in RANKS)
      + f"{'full FT':>10}")
print("-" * 92)

for tr in TRUE:
    W0, delta, X, Y, Xt, Yt = make_problem(tr)
    row = []
    for r in RANKS:
        d = fit_lora(W0, X, Y, r)
        row.append(rel_err(Xt @ (W0 + d), Yt))
    dfull = fit_full(W0, X, Y)
    ef = rel_err(Xt @ (W0 + dfull), Yt)
    print(f"{tr:>10}{'':>3}" + "".join(f"{v:>12.4f}" for v in row)
          + f"{ef:>10.4f}")

print(f"\n\nWhat rank did an UNCONSTRAINED fine-tune actually use?\n")
print(f"{'true rank of task':>19}{'effective rank of':>22}{'ratio':>9}")
print(f"{'':>19}{'the fitted delta':>22}{'':>9}")
print("-" * 50)
for tr in TRUE:
    W0, delta, X, Y, Xt, Yt = make_problem(tr)
    dfull = fit_full(W0, X, Y)
    er = effective_rank(dfull)
    print(f"{tr:>19}{er:>22}{er / tr:>9.1f}x")

print("""
Read each row of the first table left to right and the knee is unmistakable. For
a task whose true update has rank 2, LoRA at rank 1 is poor and rank 2 is already
at the floor -- adding rank beyond that buys nothing, because there is nothing
left to represent. For the rank-32 task, every LoRA rank below 32 leaves a
residual that no amount of training removes.

That is eq:lora-parameterisation behaving exactly as claimed, and it makes the
central point sharply: rank is not a quality dial. It is a CAPACITY limit. Below
the task's intrinsic rank the adapter cannot express the required update, and the
error that remains is not an optimisation failure that more steps would fix -- it
is the distance from the true delta to the nearest rank-r matrix, which is
determined by the task's singular values (eq:eckart-young).

The full fine-tuning column is the control: unconstrained, it reaches the noise
floor on every task, because it has no capacity limit to run into.

Now the second table, which is the more useful direction. Given an unconstrained
fine-tune, how many singular values does the resulting update actually need? For
these synthetic tasks the answer tracks the true rank closely, which is the
expected result and confirms the measurement works.

The reason that matters is what it licenses in practice. The effective rank of a
real fine-tuning delta is measurable the same way: fine-tune once without a rank
constraint, take the SVD of the weight change, and read off how many singular
values carry the energy. That number is the rank your adapter needs, and it is a
property of YOUR task rather than a hyperparameter to be guessed.

Which reframes the usual advice. "Try rank 8, then 16, then 32" is a search over
a quantity that can be measured directly, and the measurement costs one run that
you would arguably want anyway as a full-fine-tuning baseline.

One caution before generalising from this table. These deltas are exactly low
rank by construction, so the knee is sharp. A real adaptation delta has a decaying
spectrum rather than a hard cutoff, so the curve is a gentle bend instead of a
corner, and the choice of threshold in the effective-rank calculation matters.
The shape of the argument survives; the crispness does not.""")
