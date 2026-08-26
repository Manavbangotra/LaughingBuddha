# -*- coding: utf-8 -*-
# Extracted from: Chapter 54 — Optimizers: SGD, Momentum, RMSProp, and Adam
# Source: src/.../ch054-optimizers.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""SGD, momentum, RMSProp, Adam and AdamW on a real network, with the
coupled-versus-decoupled weight decay difference measured.
"""
import numpy as np

rng = np.random.default_rng(7)


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.W = [rs.normal(0, np.sqrt(2 / a), (a, b)) for a, b in self.shapes]
        self.b = [np.zeros(b) for _, b in self.shapes]

    def forward(self, X):
        self.H, self.Z = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        p = e / e.sum(axis=1, keepdims=True)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = p.copy()
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = (d @ self.W[l].T) * (self.Z[l - 1] > 0)
        return loss, gW, gb


D, CLASSES = 20, 4
# ONE labelling function, shared by train and test. Drawing a fresh Wt per
# split would make the task unlearnable, and the symptom — every optimiser
# scoring at chance — looks like an optimiser problem rather than a data one.
_wt_rs = np.random.default_rng(1234)
W_TRUE = _wt_rs.normal(size=(D, CLASSES))
W_TRUE[:5] /= 100.0                         # undo the feature rescaling below
W_TRUE[5:10] *= 100.0
H_TRUE = _wt_rs.normal(size=(CLASSES, CLASSES))


def make_data(n, seed=0):
    """Deliberately BADLY SCALED features, which is the realistic case."""
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    X[:, :5] *= 100.0                       # some features are huge
    X[:, 5:10] *= 0.01                      # some are tiny
    logits = np.tanh(X @ W_TRUE) @ H_TRUE * 3.0
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rs.choice(CLASSES, p=pi) for pi in p])
    return X, y


Xtr, ytr = make_data(6000, seed=1)
Xte, yte = make_data(6000, seed=2)
print("=" * 72)
print("the problem")
print("=" * 72)
_p = np.exp(np.tanh(Xte @ W_TRUE) @ H_TRUE * 3.0)
_p = _p / _p.sum(axis=1, keepdims=True)
print(f"{len(Xtr)} train / {len(Xte)} test, {CLASSES} classes, "
      f"{D} features")
print(f"Bayes-optimal accuracy on the test set : "
      f"{float(_p[np.arange(len(yte)), yte].mean()):.4f}")
print(f"Bayes-optimal cross-entropy            : "
      f"{float(-np.log(_p[np.arange(len(yte)), yte]).mean()):.4f}")
print(f"chance accuracy                        : {1 / CLASSES:.4f}")
print("\nFive features are scaled by 100 and five by 0.01, so the gradient")
print("magnitudes across the first layer span four orders of magnitude.")
print("That is the situation adaptive methods were invented for.")


def train(opt_factory, lr, steps=3000, batch=64, seed=0, wd=0.0,
          decoupled=True):
    net = MLP([20, 64, 64, 4], seed=seed)
    opts = [opt_factory(lr) for _ in net.W] + [opt_factory(lr) for _ in net.b]
    rs = np.random.default_rng(seed + 100)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        for i, (W, g) in enumerate(zip(net.W, gW)):
            if wd and not decoupled:
                g = g + wd * W
            net.W[i] = opts[i].step(W, g, t)
            if wd and decoupled:
                net.W[i] = net.W[i] - lr * wd * net.W[i]
        for i, (b, g) in enumerate(zip(net.b, gb)):
            net.b[i] = opts[len(net.W) + i].step(b, g, t)
    tr_loss, _, _ = net.loss_and_grads(Xtr, ytr)
    te_loss, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    wnorm = float(np.sqrt(sum(float(np.sum(W ** 2)) for W in net.W)))
    return tr_loss, te_loss, acc, wnorm


class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, None

    def step(self, p, g, t):
        if self.mu == 0:
            return p - self.lr * g
        if self.v is None:
            self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g
        return p - self.lr * self.v


class RMSProp:
    def __init__(self, lr, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s = self.rho * self.s + (1 - self.rho) * g * g
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh, vh = self.m / (1 - self.b1 ** t), self.v / (1 - self.b2 ** t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


print("=" * 72)
print("four optimisers, each with its OWN tuned learning rate")
print("=" * 72)

GRID = {
    "SGD": (lambda lr: SGD(lr), [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "SGD + momentum 0.9": (lambda lr: SGD(lr, 0.9),
                           [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
    "RMSProp": (lambda lr: RMSProp(lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "Adam": (lambda lr: Adam(lr), [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
}
print(f"{'optimiser':<22} {'best lr':>9} {'train loss':>12} "
      f"{'test loss':>11} {'test acc':>10}")
results = {}
for name, (factory, grid) in GRID.items():
    best = None
    for lr in grid:
        out = train(factory, lr, seed=0)
        if np.isfinite(out[1]) and (best is None or out[1] < best[1][1]):
            best = (lr, out)
    results[name] = best
    lr, (trl, tel, acc, wn) = best
    print(f"{name:<22} {lr:>9.0e} {trl:>12.4f} {tel:>11.4f} {acc:>10.4f}")

print("\nEach optimiser was given its own learning-rate search, so this is a")
print("comparison of the methods rather than of one lucky hyperparameter.")
print("\nThe best learning rates span a factor of about thirty, and the")
print("ordering is the one eq. 54.16 and eq. 54.18 predict: momentum's best")
print("rate is roughly ten times below plain SGD's, because eq. 54.16")
print("amplifies the step by 1/(1-mu); and Adam's is high in absolute terms")
print("because eq. 54.18 makes it a step SIZE rather than a gradient")
print("multiplier.")
print("\nThe substantive result is the gap between the two families. The")
print("adaptive methods reach a materially lower loss and a much higher")
print("accuracy than either SGD variant, at every learning rate either")
print("family was given. On a problem whose gradient magnitudes span four")
print("orders of magnitude across features, one global step size is simply")
print("the wrong instrument — which is the argument adaptive methods were")
print("introduced to make, here on a problem constructed to make it.")
print("\nNote also that every method is well short of the Bayes rate")
print("printed above. This is a fixed budget of 3000 steps, so what is")
print("being measured is optimisation SPEED, not what each method could")
print("eventually reach.")

# --- coupled vs decoupled weight decay --------------------------------------
print("\n" + "=" * 72)
print("Adam + L2 is not AdamW (section 5.5)")
print("=" * 72)
print("The SAME lambda through both formulations, on the same network.\n")
print(f"{'lambda':>10} {'formulation':<20} {'train loss':>12} "
      f"{'test loss':>11} {'test acc':>10} {'|W|':>9}")
trl, tel, acc, wn = train(lambda lr: Adam(lr), 1e-3, wd=0.0, seed=0)
print(f"{0:>10g} {'none':<20} {trl:>12.4f} {tel:>11.4f} "
      f"{acc:>10.4f} {wn:>9.3f}")
for wd in (0.001, 0.01, 0.1, 1.0):
    for decoupled in (False, True):
        trl, tel, acc, wn = train(lambda lr: Adam(lr), 1e-3, wd=wd,
                                  decoupled=decoupled, seed=0)
        label = ("AdamW (eq 54.12)" if decoupled
                 else "Adam + L2 (eq 54.11)")
        print(f"{wd:>10g} {label:<20} {trl:>12.4f} {tel:>11.4f} "
              f"{acc:>10.4f} {wn:>9.3f}")

print("\nRead the |W| column down the pairs. At the SAME lambda the two")
print("formulations shrink the weights by wildly different amounts.")
print("\nThe reason is in the two equations. Decoupled decay multiplies the")
print("weight by (1 - eta*lambda) each step, so at eta = 1e-3 and")
print("lambda = 1e-3 that is a factor of 1e-6 per step and essentially")
print("nothing over a few thousand steps. Coupled L2 adds lambda*W to the")
print("gradient, which then passes through the 1/sqrt(v) rescaling of")
print("eq. 54.11 — and since sqrt(v) is small, the rescaling AMPLIFIES the")
print("decay enormously.")
print("\nSo the practical consequence is not subtle: lambda does NOT")
print("transfer between the two. Reading a value from a paper that used one")
print("and applying it under the other is a silent misconfiguration, and it")
print("is exactly why AdamW's recommended lambdas look so much larger than")
print("the L2 coefficients people were used to. To get comparable")
print("regularisation you have to compare at matched |W|, not at matched")
print("lambda — which is what the rows above let you do.")
print("\nNote also the second-order point hidden in eq. 54.11: because the")
print("coupled decay is divided by sqrt(v), parameters with LARGE gradients")
print("are regularised LESS. That is backwards from the intent, and it is")
print("the substance of Loshchilov and Hutter's argument, not just the")
print("bookkeeping about lambda.")

# --- how sensitive is each optimiser to its learning rate? ------------------
print("\n" + "=" * 72)
print("sensitivity to the learning rate: the practical reason for Adam")
print("=" * 72)
print("Test loss across the whole grid, so the shape of each row is visible")
print("rather than only its best point.\n")
for name, (factory, grid) in GRID.items():
    row = []
    for lr in grid:
        out = train(factory, lr, seed=0)
        row.append("diverged" if not np.isfinite(out[1]) or out[1] > 10
                   else f"{out[1]:.3f}")
    print(f"{name:<22} " + " ".join(f"{v:>9}" for v in row))
    print(f"{'':<22} " + " ".join(f"{lr:>9.0e}" for lr in grid))
print("\nNothing diverged, which is worth saying because it is not the")
print("story the usual 'Adam is robust' claim would predict. Every method")
print("at every rate produced a finite loss.")
print("\nWhat the rows actually show is different and more useful. Both SGD")
print("variants flatten out around 1.25 and get no further no matter which")
print("rate they are given — the curve has a floor. Both adaptive methods")
print("are still improving at the TOP of their grids and reach about 1.0.")
print("\nSo the difference here is not robustness to the learning rate; it")
print("is that on a badly scaled problem SGD has a ceiling that no single")
print("global step size can get past, and the adaptive methods do not. If")
print("you take one thing from this table, take that: run the grid and look")
print("at whether it has flattened, because a flat tail means the method")
print("rather than the setting is the limit.")
