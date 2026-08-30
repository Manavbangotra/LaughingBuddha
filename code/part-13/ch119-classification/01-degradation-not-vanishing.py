# -*- coding: utf-8 -*-
# Extracted from: Chapter 119 — Image Classification and the ResNet Lineage
# Source: src/.../ch119-classification.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The degradation problem, and why "vanishing gradients" is the wrong diagnosis.

cite:he2016resnet's motivating observation is easy to state and easy to
misremember. Adding layers to a plain deep network made it worse -- not on test
error, which would be overfitting, but on TRAINING error, which cannot be.

That rules out capacity. A deeper network can express everything a shallower one
can: set the extra layers to the identity and you have the shallower network
exactly (eq:identity-embedding). So the deeper model's higher training loss is a
statement about OPTIMISATION, not about what the architecture can represent.

This listing reproduces degradation on a task small enough to check, and measures
the gradient at the same time -- because if vanishing gradients were the whole
story, the gradient measurement would show it.
"""
import numpy as np

rng = np.random.default_rng(11)

D_IN, WIDTH, N_CLASS = 24, 32, 4
N_TRAIN = 4000
EPOCHS, BATCH, LR = 40, 64, 0.05


def teacher_data(n):
    """A fixed random two-layer teacher, so the task is definitely learnable by
    a small network and any failure is the optimiser's."""
    g = np.random.default_rng(0)
    A = g.normal(size=(D_IN, 16)) / np.sqrt(D_IN)
    B = g.normal(size=(16, N_CLASS)) / 4
    X = rng.normal(size=(n, D_IN))
    y = (np.maximum(X @ A, 0) @ B).argmax(axis=1)
    return X, y


class Net:
    def __init__(self, depth, residual):
        self.depth, self.residual = depth, residual
        # A residual stream accumulates one branch's variance per layer, so its
        # scale grows like sqrt(depth) and a deep stack overflows
        # (eq:residual-variance). A real ResNet controls this with normalisation
        # inside the branch; scaling the branch by 1/sqrt(depth) does the same
        # job in one line. It is not optional -- without it, depth 32 diverges.
        self.scale = 1.0 / np.sqrt(depth) if residual else 1.0
        self.Win = rng.normal(scale=np.sqrt(2 / D_IN), size=(D_IN, WIDTH))
        self.W = [rng.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, WIDTH))
                  for _ in range(depth)]
        self.b = [np.zeros(WIDTH) for _ in range(depth)]
        self.Wout = rng.normal(scale=np.sqrt(2 / WIDTH), size=(WIDTH, N_CLASS))
        self.bout = np.zeros(N_CLASS)

    def forward(self, X):
        self.hs, self.zs = [], []
        h = np.maximum(X @ self.Win, 0)
        self.x = X
        for W, b in zip(self.W, self.b):
            self.hs.append(h)
            z = h @ W + b
            self.zs.append(z)
            a = np.maximum(z, 0)
            # The ONLY difference between the two architectures.
            h = h + self.scale * a if self.residual else a
        self.hlast = h
        return h @ self.Wout + self.bout

    def backward(self, g, lr):
        gWout, gbout = self.hlast.T @ g, g.sum(axis=0)
        gh = g @ self.Wout.T
        first_grad = None
        for i in reversed(range(self.depth)):
            ga = gh * self.scale                # residual: identity path carries gh
            gz = ga * (self.zs[i] > 0)
            gW = self.hs[i].T @ gz
            gb = gz.sum(axis=0)
            gh = gz @ self.W[i].T + (gh if self.residual else 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb
            if i == 0:
                first_grad = float(np.linalg.norm(gW))
        self.Wout -= lr * gWout
        self.bout -= lr * gbout
        return first_grad


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return loss, g / len(y)


X, y = teacher_data(N_TRAIN)

print(f"{N_TRAIN} training points, width {WIDTH}, {EPOCHS} epochs. Every number "
      f"below is\nTRAINING performance -- no test set is involved, so nothing "
      f"here is overfitting.\n")
print(f"{'depth':>7}{'plain loss':>13}{'plain acc':>12}{'':>4}"
      f"{'residual loss':>15}{'residual acc':>14}{'':>4}{'grad ratio':>12}")
print("-" * 82)

rows = {}
for depth in (2, 4, 8, 16, 32):
    out = {}
    for residual in (False, True):
        rng2 = np.random.default_rng(11)
        globals()['rng'] = rng2
        net = Net(depth, residual)
        gnorm = []
        for ep in range(EPOCHS):
            order = rng2.permutation(N_TRAIN)
            for s in range(0, N_TRAIN, BATCH):
                b = order[s:s + BATCH]
                logits = net.forward(X[b])
                _, g = softmax_ce(logits, y[b])
                fg = net.backward(g, LR)
                if ep == 0:
                    gnorm.append(fg)
        logits = net.forward(X)
        loss, _ = softmax_ce(logits, y)
        acc = float((logits.argmax(axis=1) == y).mean())
        out[residual] = (loss, acc, float(np.mean(gnorm)))
    rows[depth] = out
    ratio = out[True][2] / max(out[False][2], 1e-12)
    print(f"{depth:>7}{out[False][0]:>13.4f}{out[False][1]:>12.3f}{'':>4}"
          f"{out[True][0]:>15.4f}{out[True][1]:>14.3f}{'':>4}{ratio:>12.1f}")

p2, p32 = rows[2][False], rows[32][False]
r2, r32 = rows[2][True], rows[32][True]
print(f"""
Read the plain columns downward first, and remember these are TRAINING numbers.
Loss falls from {p2[0]:.4f} at depth 2 to {rows[4][False][0]:.4f} at depth 4 --
depth is helping -- and then reverses, rising to {p32[0]:.4f} at depth 32 with
training accuracy down to {p32[1]:.3f}. The deep model is not overfitting. It
never fitted. It is worse at the task it was directly optimised on.

That single observation is what makes degradation interesting, and it is why
"deeper is harder to train" is an incomplete explanation. Consider what the
depth-32 network could do: set twenty-four of its layers to the identity and it
becomes the depth-8 network exactly (eq:identity-embedding). The solution is
inside the hypothesis space, it is reachable, and the optimiser does not find it.
The failure is in the SEARCH, not in the space.

The residual columns are the same task, width, optimiser and epoch count, with
one line different. Loss at depth 32 is {r32[0]:.4f} against the plain network's
{p32[0]:.4f}, and accuracy holds at {r32[1]:.3f} against {p32[1]:.3f}. The
residual column is nearly FLAT in depth, which is the real claim: it is not that
skips make deep networks better, it is that they stop depth from making things
worse.

Now read the top two rows, because they are the part an enthusiastic account
would omit. At depth 2 and depth 4 the plain network WINS -- {rows[4][False][0]:.4f}
against {rows[4][True][0]:.4f}. Residual connections are not free: the branch is
scaled down to keep the stream stable, so at shallow depth the architecture is
paying for insurance it does not need. The crossover here is around depth 8, and
below it the skip is a small cost rather than a benefit.

Finally the gradient ratio column, where the usual explanation gets tested rather
than repeated. It reports how much larger the first layer's gradient is with skips
than without, early in training. It rises with depth, so skips do improve gradient
flow and the vanishing-gradient story is not wrong.

It is just too small to be the whole story. At depth 32 the ratio is only
{rows[32][True][2] / rows[32][False][2]:.1f} -- less than a factor of two -- while
the outcome gap is the difference between a network that learned the task and one
that mostly did not. A modest change in gradient magnitude is being asked to
explain a large change in result, and it cannot.

What the skip also changes is WHICH FUNCTION IS EASY. In a plain layer the
identity is a particular setting of the weights that has to be found. In a
residual block the identity is what you get when the weights are ZERO -- which is
where initialisation starts and where weight decay pulls (eq:identity-is-default).
The optimiser no longer has to discover how to do nothing.

So the residual connection fixes two things at once, and the one usually quoted
is the smaller one.""")
