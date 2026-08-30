# -*- coding: utf-8 -*-
# Extracted from: Chapter 137 — Model Merging and Distillation
# Source: src/.../ch137-merging.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Merging works, and the precondition is not optional.

cite:wortsman2022modelsoups reported that averaging the weights of several
fine-tuned models produces a model better than any of them, which sounds like it
should not work. Averaging two neural networks is normally catastrophic: the same
function can be represented by permuting hidden units, so two independently
trained networks put the same feature in different places and their average puts
it nowhere.

The reason soups work is a precondition that gets dropped in retelling: the models
were fine-tuned FROM A SHARED BASE. That keeps them in one low-loss basin, where
the segment between them stays low-loss too (eq:linear-mode-connectivity).

This listing measures the barrier both ways -- shared base and independent
initialisation -- because the difference between them is the whole reason merging
is a technique rather than folklore.
"""
import numpy as np

rng = np.random.default_rng(223)

D, H, DO = 14, 48, 6
N = 2500


def init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def grad(p, X, Y):
    h, o = forward(p, X)
    d = 2 * (o - Y) / len(X)
    dh = d @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def fit(p, X, Y, steps, lr=0.01):
    p = [w.copy() for w in p]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    for t in range(steps):
        g = grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p


def lerp(pa, pb, a):
    return [(1 - a) * x + a * y for x, y in zip(pa, pb)]


W_PRE = rng.normal(size=(D, DO)) / np.sqrt(D)
W_A = rng.normal(size=(D, DO)) / np.sqrt(D)
W_B = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.12):
    X = rng.normal(size=(n, D))
    return X, np.tanh(X @ W) + 0.3 * X[:, :DO] + noise * rng.normal(size=(n, DO))


Xp, Yp = task(W_PRE, N)
Xa, Ya = task(W_A, N); Xa_t, Ya_t = task(W_A, 1500)
Xb, Yb = task(W_B, N); Xb_t, Yb_t = task(W_B, 1500)

# Shared base: one pretrained model, two fine-tunes.
BASE = fit(init(), Xp, Yp, 3000)
SA = fit(BASE, Xa, Ya, 900)
SB = fit(BASE, Xb, Yb, 900)

# Independent: two models trained on the same two tasks from scratch.
IA = fit(init(), Xa, Ya, 3000)
IB = fit(init(), Xb, Yb, 3000)

# The thing a merge is a cheap substitute for: one model trained on both tasks.
XJ = np.concatenate([Xa, Xb]); YJ = np.concatenate([Ya, Yb])
JOINT = fit(BASE, XJ, YJ, 1800)

ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

print(f"Interpolating between two fine-tunes. alpha=0 is model A, 1 is model B.\n")
print(f"{'alpha':>7}{'SHARED BASE':>34}{'INDEPENDENT INIT':>36}")
print(f"{'':>7}{'A loss':>11}{'B loss':>11}{'joint':>11}{'':>2}"
      f"{'A loss':>11}{'B loss':>11}{'joint':>11}")
print("-" * 77)

sh, ind = [], []
for a in ALPHAS:
    ps, pi = lerp(SA, SB, a), lerp(IA, IB, a)
    sv = (mse(ps, Xa_t, Ya_t), mse(ps, Xb_t, Yb_t))
    iv = (mse(pi, Xa_t, Ya_t), mse(pi, Xb_t, Yb_t))
    sh.append(sv); ind.append(iv)
    print(f"{a:>7.3f}{sv[0]:>11.4f}{sv[1]:>11.4f}"
          f"{(sv[0]+sv[1])/2:>11.4f}{'':>2}"
          f"{iv[0]:>11.4f}{iv[1]:>11.4f}{(iv[0]+iv[1])/2:>11.4f}")

joint = lambda v: (v[0] + v[1]) / 2
m_sh = min(sh, key=joint)
m_ind = min(ind, key=joint)
A_SH = ALPHAS[sh.index(m_sh)]
A_IND = ALPHAS[ind.index(m_ind)]
MID = len(ALPHAS) // 2
J_MULTI = (mse(JOINT, Xa_t, Ya_t), mse(JOINT, Xb_t, Yb_t))
J_BASE = (mse(BASE, Xa_t, Ya_t), mse(BASE, Xb_t, Yb_t))

print("")
print(f"{'one model that does both tasks':>34}{'A loss':>11}{'B loss':>11}"
      f"{'joint':>11}")
print("-" * 67)
for name, v in (("base model, no fine-tuning", J_BASE),
                ("merged, shared base", m_sh),
                ("merged, independent init", m_ind),
                ("trained jointly on both tasks", J_MULTI)):
    print(f"{name:>34}{v[0]:>11.4f}{v[1]:>11.4f}{joint(v):>11.4f}")
print(f"""
The alpha sweep says merging is possible at all. The summary table says what it
is worth.

Start with the sweep. Both columns move smoothly from one specialist to the other
-- there is no wall in the middle, which is already worth noting, because
averaging two neural networks has no right to work. A network's function is
unchanged by permuting its hidden units, so two independently trained networks
that compute similar things generally put those things in different coordinates,
and averaging coordinate-wise then averages unrelated features.

But compare the joint columns at the true midpoint, which is what "average the
two models" actually means. With a shared base, alpha=0.5 gives
{joint(sh[MID]):.4f}. With independent initialisation, {joint(ind[MID]):.4f} --
{joint(ind[MID])/joint(sh[MID]):.1f}x worse, on the same tasks, the same
architecture and the same budget.

Then look at where each column's BEST point sits, which is the sharper diagnostic.
The shared-base optimum is at alpha={A_SH}, in the interior, where the merge is
genuinely combining two models. The independent optimum is at alpha={A_IND},
essentially at an endpoint -- the best thing you can do with those two models is
to pick one and use it. That is what a failed merge looks like from the outside:
not an error, just an optimum that quietly degenerates to "do not merge".

That gap is the precondition, measured. Fine-tuning from a shared base never
leaves the region where the coordinates mean the same thing, so the segment
between two fine-tunes stays inside one low-loss basin
(eq:linear-mode-connectivity). Independent runs land in different basins that
happen to compute similar functions, and the straight line between them leaves
both.

So the precondition is operational rather than theoretical: merging works between
models that share an ancestor and have not travelled far from it. It is not a
general model-combination technique, and essentially every merge failure reported
in practice is a violation of that sentence rather than a subtlety of the merging
algorithm.

Three violations worth naming because they look innocuous. Two models of the same
architecture from different pretraining runs do not share a base. A model
fine-tuned twice in sequence has travelled further than the merge assumes. And a
quantised copy is not the same base as the original, which is
ch:ft-qlora-peft's adapter constraint arriving here in a new form.

Now the summary table, which prices the technique honestly.

The base model before any fine-tuning has a joint loss of {joint(J_BASE):.4f} --
the do-nothing baseline. The shared-base merge reaches {joint(m_sh):.4f}.
Training one model on both tasks together, which is what a merge is a cheap
substitute for, reaches {joint(J_MULTI):.4f}.

That is the number that makes merging worth a chapter. The merge is within
{joint(m_sh)/joint(J_MULTI)-1:.0%} of an actual multi-task training run, and it
cost a weighted sum over two checkpoints you already had. Not an approximation
that gets you most of the way with caveats -- {joint(m_sh)/joint(J_MULTI)-1:.0%},
for milliseconds of arithmetic.

Read the same numbers the other way and the limit is equally clear. The merged
model is worse on task A than model A and worse on task B than model B. Merging
did not produce a model better at everything; it produced ONE model instead of
two, at a measured cost on each.

That distinction matters because it is routinely blurred.
cite:wortsman2022modelsoups's headline -- a soup that beats every ingredient --
comes from averaging models fine-tuned on the SAME task with different
hyperparameters, where averaging cancels independent noise. Merging models
fine-tuned on DIFFERENT tasks combines skills and pays for it. Two techniques, one
name, and only the first is free.""")
