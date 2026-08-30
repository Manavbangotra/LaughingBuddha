# -*- coding: utf-8 -*-
# Extracted from: Chapter 136 — Training Configuration, Catastrophic Forgetting, and Overfitting
# Source: src/.../ch136-training-config.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The exchange rate between new capability and old capability.

Every fine-tuning run has a stopping rule, and in practice it is the target
task's validation loss. That rule optimises one of the two things a fine-tune
changes and is silent about the other.

This listing tracks both. A network is trained to convergence on task A, then
fine-tuned on task B, and after every few steps BOTH are measured -- along with
the distance travelled from the base weights, which ch:ft-lora's
eq:forgetting-quadratic says should govern the damage.

The column to watch is the last one: how much task-B capability the run is buying
per unit of task-A capability it is spending, at each point along the way
(eq:capability-exchange-rate).
"""
import numpy as np

rng = np.random.default_rng(211)

D, H, DO = 16, 64, 8
N = 3000


def mlp_init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def loss_grad(p, X, Y):
    h, o = forward(p, X)
    d = (o - Y) / len(X)
    g3, g4 = h.T @ (2 * d), (2 * d).sum(0)
    dh = (2 * d) @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), g3, g4]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def train(p, X, Y, steps, lr, snap=None, Xa=None, Ya=None, base=None):
    """Plain Adam, full batch, so the trajectory is deterministic and the
    distance travelled is a clean quantity to reason about."""
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    hist = []
    for t in range(steps + 1):
        if snap and t % snap == 0:
            dist = np.sqrt(sum(((a - b) ** 2).sum()
                               for a, b in zip(p, base))) if base else 0.0
            hist.append((t, mse(p, X, Y), mse(p, Xa, Ya), dist))
        g = loss_grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p, hist


WA = rng.normal(size=(D, DO)) / np.sqrt(D)
WB = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.15):
    """Observation noise gives the base model a nonzero loss floor, so
    'degradation' is measured against something meaningful."""
    X = rng.normal(size=(n, D))
    return X, (np.tanh(X @ W) + 0.3 * X[:, :DO]
               + noise * rng.normal(size=(n, DO)))


Xa, Ya = task(WA, N)
Xb, Yb = task(WB, N)
Xa_te, Ya_te = task(WA, 2000)
Xb_te, Yb_te = task(WB, 2000)

base, _ = train(mlp_init(), Xa, Ya, 4000, 0.01)
base = [w.copy() for w in base]
A0, B0 = mse(base, Xa_te, Ya_te), mse(base, Xb_te, Yb_te)
print(f"Base model trained on task A: A loss {A0:.4f}, B loss {B0:.4f}.\n")

p = [w.copy() for w in base]
_, hist = train(p, Xb, Yb, 57, 0.0015, snap=3,
                Xa=Xa_te, Ya=Ya_te, base=base)

B_BEST = min(r[1] for r in hist)
A_WORST = max(r[2] for r in hist)


def gained(lb):
    return (B0 - lb) / (B0 - B_BEST)


def lost(la):
    return (la - A0) / (A_WORST - A0)


print(f"{'step':>6}{'B loss':>9}{'A loss':>9}{'B gained':>10}{'A lost':>9}"
      f"{'||delta||':>11}{'marginal':>11}")
print(f"{'':>6}{'':>9}{'':>9}{'':>10}{'':>9}{'':>11}{'exchange':>11}")
print("-" * 65)
prev = None
for t, lb, la, dist in hist:
    g, l = gained(lb), lost(la)
    rate = ""
    if prev is not None:
        dl = l - prev[1]
        rate = f"{(g - prev[0]) / dl:>11.1f}" if dl > 1e-9 else f"{'--':>11}"
    print(f"{t:>6}{lb:>9.4f}{la:>9.4f}{g:>10.1%}{l:>9.1%}{dist:>11.3f}{rate}")
    prev = (g, l)

b_only = min(hist, key=lambda r: r[1])
cheap = [r for r in hist if lost(r[2]) <= 0.15]
best_cheap = max(cheap, key=lambda r: gained(r[1]))
print(f"""
Read the last column down the page, because it is the whole listing.

Early in the run each unit of task-A capability spent buys several units of
task-B capability. Late in the run it buys a fraction of one. The exchange rate
is not constant, it does not degrade gently, and it crosses 1.0 well before the
target task's loss curve gives any sign of stopping.

That shape follows from the two quantities rather than from this setup. Task A's
loss is quadratic in the distance travelled from its minimum
(ch:ft-lora's eq:forgetting-quadratic), so at small distances it is nearly flat --
the first steps cost almost nothing. Task B's loss starts far from its own minimum
and falls fast and then slower, concave in the same distance. Quadratic against
concave gives exactly this: early steps nearly free, late steps nearly pure
damage (eq:capability-exchange-rate).

The numbers make the trade concrete. By step {best_cheap[0]}, task B has gained
{gained(best_cheap[1]):.0%} of everything it will ever gain, for
{lost(best_cheap[2]):.0%} of the task-A capability it will eventually destroy. The
remaining {1-gained(best_cheap[1]):.0%} of task B costs the other
{1-lost(best_cheap[2]):.0%} of task A.

That is not a marginal call. It is most of the benefit for a small fraction of the
cost, followed by a small benefit for most of the cost, and the usual stopping
rule takes the second deal without being asked.

The standard criterion -- lowest validation loss on the target task -- stops at
step {b_only[0]}, having given up {lost(b_only[2]):.0%} of task A. Nothing about
that checkpoint is wrong if task A does not matter. The problem is that the
decision was never made: the target task's validation curve chose the trade
silently, and it has no information about the thing being traded away.

Which makes the fix cheap and specific. The base-capability evaluation you need
already exists -- it is whatever told you the base model was good enough to start
from. Run it at every checkpoint. It costs a small fraction of the training budget
and converts an invisible exchange into a visible one, with an exchange rate you
can read off directly.

Then choose deliberately. If the base capability is the product and the fine-tune
is a refinement, stop early where the rate is favourable. If the base model is
scaffolding for a narrow deployment and general ability is irrelevant, train to
convergence and ignore the column. Both are legitimate. Picking by default is not,
and picking by default is the current standard.

One thread left for the next listing. Both effects track the distance column, and
if forgetting is a function of HOW FAR the weights moved rather than of how they
got there, then learning rate and epoch count are not independent levers at all --
they are two ways of setting one quantity, and most training-configuration advice
is about that quantity without saying so.""")
