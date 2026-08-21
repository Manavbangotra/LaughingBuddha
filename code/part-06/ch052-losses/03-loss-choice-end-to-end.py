# Extracted from: Chapter 52 — Loss Functions
# Source: src/.../ch052-losses.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""One dataset, four losses, on a real network: the choice measured on data
that violates the Gaussian assumption of section 6.1.
"""
import numpy as np

rng = np.random.default_rng(17)


# --- a regression problem with heavy-tailed noise ---------------------------
def make_regression(n, noise, seed):
    """Clean signal; noise is either Gaussian or heavy-tailed."""
    rs = np.random.default_rng(seed)
    X = rs.uniform(-2, 2, (n, 3))
    signal = (np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
              - 0.8 * X[:, 0] * X[:, 2])
    if noise == "gaussian":
        eps = rs.normal(0, 0.3, n)
    else:                                     # 5% at 20x the scale
        eps = rs.normal(0, 0.3, n)
        idx = rs.choice(n, size=n // 20, replace=False)
        eps[idx] = rs.normal(0, 6.0, len(idx))
    return X, signal + eps, signal


class MLP:
    """Two hidden layers, hand-written backward, one loss slot."""

    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        self.h = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            self.h.append(h)
        return h

    def backward(self, dout, lr):
        for i in reversed(range(len(self.W))):
            hin = self.h[i]
            gW = hin.T @ dout / len(dout)
            gb = dout.mean(axis=0)
            if i > 0:
                dout = (dout @ self.W[i].T) * (self.h[i] > 0)
            self.W[i] -= lr * gW
            self.b[i] -= lr * gb


LOSS_GRADS = {
    "squared": lambda p, t: 2 * (p - t),
    "absolute": lambda p, t: np.sign(p - t),
    "huber(1.0)": lambda p, t: np.clip(p - t, -1.0, 1.0),
    "huber(0.3)": lambda p, t: np.clip(p - t, -0.3, 0.3),
}


def train(X, y, loss, steps=4000, lr=0.02, batch=64, seed=0):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    rs = np.random.default_rng(seed + 1)
    grad = LOSS_GRADS[loss]
    for _ in range(steps):
        idx = rs.integers(0, len(X), batch)
        pred = net.forward(X[idx])
        net.backward(grad(pred, y[idx, None]), lr)
    return net


print("=" * 72)
print("the loss choice under two noise models (section 6.1)")
print("=" * 72)
print("The SAME clean signal; only the noise distribution differs. Error is")
print("measured against the noise-free signal, so we can see what each loss")
print("actually recovered rather than how well it fitted the noise.\n")

for noise in ("gaussian", "heavy-tailed"):
    Xtr, ytr, _ = make_regression(3000, noise, 21)
    Xte, yte, clean_te = make_regression(3000, noise, 22)
    print(f"{noise} noise")
    print(f"  {'loss':<14} {'RMSE vs noisy y':>17} {'RMSE vs clean':>15} "
          f"{'MAE vs clean':>14}")
    for loss in LOSS_GRADS:
        net = train(Xtr, ytr, loss)
        p = net.forward(Xte)[:, 0]
        rmse_noisy = float(np.sqrt(np.mean((p - yte) ** 2)))
        rmse_clean = float(np.sqrt(np.mean((p - clean_te) ** 2)))
        mae_clean = float(np.mean(np.abs(p - clean_te)))
        print(f"  {loss:<14} {rmse_noisy:>17.4f} {rmse_clean:>15.4f} "
              f"{mae_clean:>14.4f}")
    print()

print("Read the 'RMSE vs clean' column: it measures what we actually want,")
print("which is how well each loss recovered the underlying signal.")
print("\nUnder Gaussian noise squared error is the maximum-likelihood")
print("estimator (section 6.1) and it wins, as it should. Under heavy-tailed")
print("noise its assumption is violated and it loses to the robust losses.")
print("\nNote what delta does to Huber. At delta=1.0 it is the best loss in")
print("the heavy-tailed setting and second-worst in the Gaussian one; at")
print("delta=0.3 it is mediocre in both. Too small a delta throws away the")
print("efficiency of squared error on the bulk of the data in exchange for")
print("robustness it does not need there. Delta is a real hyperparameter")
print("and 'use Huber' is not by itself a decision.")
print("\nNote the trap in the 'RMSE vs noisy y' column: it is the metric you")
print("would actually compute in production, since the clean signal is")
print("unobservable, and squared error is favoured by it BY CONSTRUCTION —")
print("evaluating with the same functional form you trained with is not a")
print("neutral comparison.")

# --- classification: the pairing, on a network ------------------------------
print("=" * 72)
print("the loss-activation pairing, on a real network (section 6.5)")
print("=" * 72)


def make_clf(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, 8))
    logit = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.9 * X[:, 0] * X[:, 2]
    y = (rs.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, y


Xc, yc = make_clf(4000, 31)
Xcv, ycv = make_clf(4000, 32)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def eval_clf(net, X, y):
    s = net.forward(X)[:, 0]
    p = sigmoid(s)
    nll = -np.mean(y * np.log(np.clip(p, 1e-12, 1))
                   + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    acc = ((p > 0.5) == y).mean()
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return nll, acc, auc, p.mean()


def train_clf_traced(X, y, Xv, yv, pairing, bias, steps=4000, lr=0.05,
                     batch=64, seed=0, checkpoints=(0, 100, 400, 1000, 4000)):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    net.b[-1] += bias                    # start CONFIDENTLY predicting 1
    rs = np.random.default_rng(seed + 1)
    out = {}
    for t in range(steps + 1):
        if t in checkpoints:
            out[t] = eval_clf(net, Xv, yv)
        p = sigmoid(net.forward(X[(idx := rs.integers(0, len(X), batch))]))
        tgt = y[idx, None]
        if pairing == "cross-entropy":
            dz = p - tgt                               # eq. 52.15
        else:                                          # squared error
            dz = 2 * (p - tgt) * p * (1 - p)           # eq. 52.17
        net.backward(dz, lr)
    return out


for bias in (4.0, 8.0):
    p0 = 1 / (1 + np.exp(-bias))
    print(f"output bias +{bias:.0f}: the network starts predicting p = "
          f"{p0:.6f} for everything.")
    print(f"  eq. 52.17's damping factor p(1-p) is {p0 * (1 - p0):.2e} there, "
          f"so squared error\n  starts with a gradient "
          f"{1 / (p0 * (1 - p0)):.0f}x smaller than cross-entropy's.\n")
    print(f"  {'step':>6}  {'cross-entropy NLL':>18} {'acc':>7}   "
          f"{'squared-error NLL':>18} {'acc':>7}")
    ce = train_clf_traced(Xc, yc, Xcv, ycv, "cross-entropy", bias)
    ms = train_clf_traced(Xc, yc, Xcv, ycv, "squared error", bias)
    for t in sorted(ce):
        print(f"  {t:>6}  {ce[t][0]:>18.4f} {ce[t][1]:>7.4f}   "
              f"{ms[t][0]:>18.4f} {ms[t][1]:>7.4f}")
    print()

print(f"base rate: {ycv.mean():.4f}")
print("\nRead the early rows, not the last one. Eq. 52.17 is a statement")
print("about the gradient in the saturated region, so it predicts a")
print("difference in how fast each pairing ESCAPES that region — not")
print("necessarily a difference in where they end up after a long run.")
print("\nAt bias +4 the damping factor is around 0.018, which slows squared")
print("error down without stopping it. At bias +8 it is around 3e-4, and the")
print("gap in the early rows is the whole point of the chapter: the")
print("cross-entropy network is already learning while the squared-error one")
print("has barely moved.")
print("\nBe careful about the final row. Given enough steps the squared-")
print("error network can catch up, and on this problem it does. That is")
print("worth saying plainly rather than hiding: the wrong pairing is a")
print("severe slowdown at initialisation, not an impossibility. In a real")
print("network with many saturating units, at a depth where the damping")
print("factors multiply, 'severe slowdown' becomes 'does not train' —")
print("which is the Chapter 50 argument, applied to the output layer.")

# --- label smoothing, measured ----------------------------------------------
print("\n" + "=" * 72)
print("label smoothing: what it costs and what it buys (eq. 52.20)")
print("=" * 72)


def train_smoothed(X, y, eps, steps=4000, lr=0.05, batch=64, seed=0):
    net = MLP([X.shape[1], 48, 48, 1], seed=seed)
    rs = np.random.default_rng(seed + 1)
    for _ in range(steps):
        idx = rs.integers(0, len(X), batch)
        p = sigmoid(net.forward(X[idx]))
        t = y[idx, None] * (1 - eps) + eps / 2.0       # binary: C = 2
        net.backward(p - t, lr)
    return net


print(f"{'epsilon':>9} {'val NLL':>9} {'AUC':>8} {'ECE':>8} "
      f"{'max |logit|':>13} {'mean |logit|':>14}")
for epsA in (0.0, 0.05, 0.1, 0.2):
    net = train_smoothed(Xc, yc, epsA)
    s = net.forward(Xcv)[:, 0]
    p = sigmoid(s)
    nll = -np.mean(ycv * np.log(np.clip(p, 1e-12, 1))
                   + (1 - ycv) * np.log(np.clip(1 - p, 1e-12, 1)))
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos, nneg = ycv.sum(), (1 - ycv).sum()
    auc = (ranks[ycv == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # expected calibration error, 10 bins
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - ycv[bins == b].mean())
              * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    print(f"{epsA:>9.2f} {nll:>9.4f} {auc:>8.4f} {ece:>8.4f} "
          f"{np.abs(s).max():>13.3f} {np.abs(s).mean():>14.3f}")

print("\nThe logit columns are the mechanism of eq. 52.20 made visible: with")
print("no smoothing the optimum is unreachable and the logits keep growing;")
print("any epsilon caps them at a finite value that shrinks as epsilon rises.")
print("\nNote that NLL here is computed against the TRUE hard labels, so a")
print("smoothed model is being penalised for exactly the under-confidence it")
print("was asked to produce. That is the honest way to score it, rather than")
print("against its own smoothed target — and it still comes out ahead at")
print("moderate epsilon, with the calibration error roughly halving.")
print("\nThe trade is visible in the last row: push epsilon far enough and")
print("the enforced under-confidence starts costing more than the")
print("regularisation buys. AUC barely moves throughout, which is the same")
print("pattern as the class-weighting experiment — these are interventions")
print("on the probabilities, not on the ranking.")
