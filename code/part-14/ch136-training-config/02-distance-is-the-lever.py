# -*- coding: utf-8 -*-
# Extracted from: Chapter 136 — Training Configuration, Catastrophic Forgetting, and Overfitting
# Source: src/.../ch136-training-config.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Learning rate and epochs set one quantity. Here is what actually moves the curve.

The standard advice for reducing catastrophic forgetting is to lower the learning
rate and train for fewer epochs. The previous listing suggested why that advice
behaves oddly in practice: both effects tracked the DISTANCE travelled from the
base weights, and learning rate and epoch count are two ways of setting one
distance.

This listing tests that directly, then asks the question that follows -- if
configuration mostly sets how far you go rather than what going that far costs,
what changes the cost?
"""
# --- setup, identical to the previous listing -----------------------
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
# --- end of shared setup --------------------------------------------

base = [w.copy() for w in base]
A0, B0 = mse(base, Xa_te, Ya_te), mse(base, Xb_te, Yb_te)


def run(lr, steps, rehearse=0.0, anchor=0.0, snap=2):
    """Fine-tune on B. `rehearse` mixes task-A examples back into training;
    `anchor` penalises distance from the base weights."""
    p = [w.copy() for w in base]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    out = []
    n_a = int(len(Xb) * rehearse)
    Xm = np.concatenate([Xb, Xa[:n_a]]) if n_a else Xb
    Ym = np.concatenate([Yb, Ya[:n_a]]) if n_a else Yb
    for t in range(steps + 1):
        if t % snap == 0:
            d = np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(p, base)))
            out.append((d, mse(p, Xb_te, Yb_te), mse(p, Xa_te, Ya_te)))
        g = loss_grad(p, Xm, Ym)
        for i in range(4):
            if anchor:
                g[i] = g[i] + anchor * (p[i] - base[i])
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return out


REF = run(0.0015, 300)
B_BEST = min(r[1] for r in REF)
A_WORST = max(r[2] for r in REF)


def gained(lb):
    return (B0 - lb) / (B0 - B_BEST)


def lost(la):
    return (la - A0) / (A_WORST - A0)


print("Does HOW you travel matter, or only how far?\n")
CONFIGS = [(0.0004, 320, 2), (0.0015, 100, 1), (0.0050, 32, 1)]
TARGETS = (0.3, 0.6, 0.9, 1.2, 1.5)
print(f"{'distance':>9}" + "".join(f"{'lr=' + str(lr):>22}"
                                   for lr, _, _ in CONFIGS))
print(f"{'travelled':>9}" + "".join(f"{'B gained':>11}{'A lost':>11}"
                                    for _ in CONFIGS))
print("-" * 75)

traj = {lr: run(lr, st, snap=sn) for lr, st, sn in CONFIGS}
spread, dmg = [], []
for tgt in TARGETS:
    cells = []
    for lr, _, _ in CONFIGS:
        r = min(traj[lr], key=lambda x: abs(x[0] - tgt))
        cells.append((gained(r[1]), lost(r[2])))
    spread.append(max(c[1] for c in cells) - min(c[1] for c in cells))
    dmg.append([c[1] for c in cells])
    print(f"{tgt:>9.1f}" + "".join(f"{g:>11.1%}{l:>11.1%}" for g, l in cells))

print("\n\nWhat DOES move the curve? Compared at matched task-B gain.\n")
STRATS = [("plain fine-tune", dict()),
          ("lower LR, more steps", dict(lr=0.0004, steps=400)),
          ("rehearsal: 20% task-A data", dict(rehearse=0.20)),
          ("rehearsal: equal parts A and B", dict(rehearse=1.00)),
          ("anchor to base weights", dict(anchor=3.0))]

print(f"{'strategy':>32}{'A lost at':>12}{'A lost at':>12}{'A lost at':>12}")
print(f"{'':>32}{'50% of B':>12}{'75% of B':>12}{'90% of B':>12}")
print("-" * 68)
res = {}
for name, kw in STRATS:
    lr = kw.pop("lr", 0.0015)
    st = kw.pop("steps", 300)
    tr = run(lr, st, **kw)
    row = []
    for want in (0.50, 0.75, 0.90):
        hit = [r for r in tr if gained(r[1]) >= want]
        row.append(lost(hit[0][2]) if hit else float("nan"))
    res[name] = row
    print(f"{name:>32}" + "".join(f"{'--':>12}" if np.isnan(v)
                                  else f"{v:>12.1%}" for v in row))

pl = res["plain fine-tune"]
lo = res["lower LR, more steps"]
r2 = res["rehearsal: 20% task-A data"]
r1 = res["rehearsal: equal parts A and B"]
an = res["anchor to base weights"]
print(f"""
The first table is the negative result, and it is the more useful half.

Three learning rates spanning an order of magnitude, with step counts chosen so
each covers the same ground. At short distances the three columns agree closely:
at 0.6 the task-A damage is {tuple(f'{v:.1%}' for v in dmg[1])}. The spread widens
with distance -- at 1.5 it is {tuple(f'{v:.1%}' for v in dmg[-1])} -- so the
fastest configuration is somewhat worse per unit of ground covered, which is a
real effect and a modest one.

The headline is the agreement, not the residual. Across an order of magnitude in
learning rate, distance travelled predicts damage far better than configuration
does. Learning rate and epoch count are not two independent levers on forgetting:
they are mostly two ways of setting one quantity, and the quantity is distance
from the base weights (eq:distance-is-the-lever).

That explains why the standard advice behaves inconsistently in practice. "Lower
the learning rate to forget less" works if you also stop after the same number of
steps, because then you have travelled less. Run the lower learning rate to
convergence, as anyone chasing target-task quality eventually does, and you arrive
at nearly the same place by a slower route.

The second table compares at matched task-B gain rather than matched distance, so
these are genuine trade-off comparisons. Every intervention helps, and the sizes
are the story.

At 75% of task B: the plain run costs {pl[1]:.1%} of task A. Lowering the learning
rate and training longer costs {lo[1]:.1%} -- the small path-efficiency effect the
first table predicted. Rehearsal at 20% costs {r2[1]:.1%}, at equal parts
{r1[1]:.1%}. Anchoring to the base weights costs {an[1]:.1%}.

The best of them improves on the plain run by a factor of
{pl[1]/min(lo[1], r2[1], r1[1], an[1]):.2f}. That is worth having and it is not
the big lever, and saying so is the point of running the comparison rather than
recommending a favourite.

Put it next to the previous listing to see what the big lever is. Stopping early
took 59% of task B for 15% of task A, where training to the target task's best
validation loss took essentially all of task B for essentially all of task A. The
stopping decision moves the damage by roughly a factor of four. Every mitigation
in this table moves it by about {pl[1]/min(lo[1], r2[1], r1[1], an[1]):.1f}.

So the hierarchy is: decide where to stop, then mitigate, then configure -- and
the usual ordering of effort is the exact reverse.

Two notes on the mitigations, since they are the part people can act on today.

Rehearsal is the one that changes what a given distance COSTS rather than how much
of it you cover, because the gradient now contains a term pulling toward task A's
minimum. It is also the cheapest to try, and teams skip it for the wrong reason:
you do not need the original pretraining corpus. Any data exercising the
capability you want to keep will do -- a few thousand general
instruction-following examples, held out from the fine-tuning task. The 20% row is
one line of a data loader.

Anchoring performs comparably here while treating every direction as equally
expensive, which rehearsal does not -- rehearsal implicitly knows which directions
task A cares about. That gap is exactly what cite:kirkpatrick2017ewc's Fisher
weighting exists to close, and it is why an importance-weighted penalty should in
principle beat both. It is also why nobody uses one: rehearsal gets most of the
benefit for none of the machinery.""")
