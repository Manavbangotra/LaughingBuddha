# Extracted from: Chapter 58 — Regularization, Dropout, Overfitting, and Underfitting
# Source: src/.../ch058-regularization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every regulariser in the chapter on the same problem, at the same
budget, with the data-size regime varied — because that is what decides.
"""
import numpy as np

rng = np.random.default_rng(11)

D, C = 20, 5
_a = np.random.default_rng(321)
T1, T2 = _a.normal(size=(D, 14)), _a.normal(size=(14, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ T1) @ T2 * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


# One pool split in half. Val and test must be EXCHANGEABLE, or the
# early-stopping measurement below picks up the difference between two
# separately drawn sets instead of the selection effect it is testing.
_Xpool, _ypool = make_data(24000, 90)
Xva, yva = _Xpool[:12000], _ypool[:12000]
Xte, yte = _Xpool[12000:], _ypool[12000:]
_p = np.exp(np.tanh(Xte @ T1) @ T2 * 1.5)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Net:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X, p_drop=0.0, rs=None):
        self.H, self.Z, self.M = [X], [], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            if i < len(self.W) - 1:
                h = np.maximum(0.0, z)
                if p_drop > 0 and rs is not None:
                    m = (rs.random(h.shape) >= p_drop) / (1 - p_drop)
                    h, keep = h * m, m
                else:
                    keep = None
                self.M.append(keep)
            else:
                h, keep = z, None
                self.M.append(keep)
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y, p_drop=0.0, rs=None):
        logits = self.forward(X, p_drop, rs)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ self.W[l].T
                if self.M[l - 1] is not None:
                    d = d * self.M[l - 1]
                d = d * (self.Z[l - 1] > 0)
        return loss, gW, gb


def evaluate(net, X, y):
    loss, _, _ = net.loss_and_grads(X, y)
    acc = float((net.forward(X).argmax(axis=1) == y).mean())
    return loss, acc


def train(Xtr, ytr, wd=0.0, p_drop=0.0, aug=0.0, label_smooth=0.0,
          early_stop=False, steps=6000, lr=2e-3, batch=64, seed=0,
          eval_every=250):
    net = Net([D, 128, 128, C], seed=seed)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 60)
    best = (np.inf, None, 0)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), min(batch, len(Xtr)))
        xb, yb = Xtr[idx], ytr[idx]
        if aug:                       # Gaussian jitter: the tabular analogue
            xb = xb + rs.normal(0, aug, xb.shape)
        logits = net.forward(xb, p_drop, rs)
        mm = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - mm)
        d = e / e.sum(axis=1, keepdims=True)
        onehot = np.eye(C)[yb]
        if label_smooth:
            onehot = onehot * (1 - label_smooth) + label_smooth / C
        d = (d - onehot) / len(xb)
        gW, gb = [None] * len(net.W), [None] * len(net.W)
        for l in reversed(range(len(net.W))):
            gW[l] = net.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = d @ net.W[l].T
                if net.M[l - 1] is not None:
                    d = d * net.M[l - 1]
                d = d * (net.Z[l - 1] > 0)
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if wd and pp.ndim == 2:
                pp -= lr * wd * pp
        if early_stop and t % eval_every == 0:
            vl, _ = evaluate(net, Xva, yva)
            if vl < best[0]:
                best = (vl, [p.copy() for p in params], t)
    if early_stop and best[1] is not None:
        for pp, saved in zip(params, best[1]):
            pp[...] = saved
    tr_loss, tr_acc = evaluate(net, Xtr, ytr)
    te_loss, te_acc = evaluate(net, Xte, yte)
    return tr_loss, te_loss, tr_acc, te_acc, best[2]


RECIPES = {
    "none": {},
    "weight decay 0.01": {"wd": 0.01},
    "weight decay 0.1": {"wd": 0.1},
    "dropout 0.2": {"p_drop": 0.2},
    "dropout 0.5": {"p_drop": 0.5},
    "input noise 0.3": {"aug": 0.3},
    "label smoothing 0.1": {"label_smooth": 0.1},
    "early stopping": {"early_stop": True},
    "wd 0.01 + drop 0.2 + noise": {"wd": 0.01, "p_drop": 0.2, "aug": 0.3},
}

print("=" * 72)
print("the same regularisers in two data regimes")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}")
print("Excess test loss above that floor; lower is better.\n")

for n_train, label in ((500, "SMALL: 500 training examples"),
                       (20000, "LARGE: 20000 training examples")):
    Xtr, ytr = make_data(n_train, 1)
    print(f"{label}   ({n_train} examples, "
          f"{sum(W.size for W in Net([D, 128, 128, C]).W):,} weights)")
    print(f"  {'recipe':<28} {'train loss':>11} {'excess test':>12} "
          f"{'test acc':>10} {'gap':>8}")
    base = None
    for name, kw in RECIPES.items():
        trl, tel, tra, tea, stopped = train(Xtr, ytr, **kw)
        if base is None:
            base = tel - BAYES
        note = f"  (stopped @{stopped})" if kw.get("early_stop") else ""
        print(f"  {name:<28} {trl:>11.4f} {tel - BAYES:>12.4f} "
              f"{tea:>10.4f} {tel - trl:>8.4f}{note}")
    print()

print("Read the GAP column first — the train/test difference, which is")
print("what regularisation exists to reduce.")
print("\nAt 500 examples the network has far more weights than data, the")
print("unregularised gap is enormous, and every technique has something to")
print("work with. At 20000 the unregularised gap is several times smaller")
print("before any technique is applied.")
print("\nThe honest headline is in the 'none' rows: going from 500 to 20000")
print("examples improved the excess test loss by more than any regulariser")
print("achieved within either regime. MORE DATA BEAT EVERY TECHNIQUE ON")
print("THIS TABLE, which is the comparison people skip and the one that")
print("usually decides.")
print("\nRegularisation still helps at the larger size — this is not a")
print("regime where it stops mattering — but the amount available to gain")
print("has shrunk with the gap, and the techniques that cost accuracy by")
print("removing capacity are correspondingly closer to breaking even.")
print("\nExtrapolate that trend and you get the change in practice noted in")
print("section 5.6. A large language model sees each token roughly once,")
print("so its gap is near zero by construction and the techniques aimed at")
print("preventing memorisation have nothing left to prevent — which is why")
print("dropout largely disappeared from them without anyone deciding it was")
print("a bad idea.")

# --- early stopping's honest accounting -------------------------------------
print("=" * 72)
print("early stopping consumes the validation set (section 5.3)")
print("=" * 72)
Xtr, ytr = make_data(500, 1)
print("Validation and test are two halves of ONE pool, so they are")
print("exchangeable and any systematic difference is the selection effect.\n")
print(f"{'seed':>6} {'best VAL loss':>15} {'TEST loss at that point':>25} "
      f"{'optimism':>10}")
opt = []
for seed in range(5):
    net = Net([D, 128, 128, C], seed=seed)
    params = net.W + net.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 60)
    best_v, best_t = np.inf, None
    LR = 2e-3
    for t in range(1, 6001):
        idx = rs.integers(0, len(Xtr), 64)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        for i, (pp, g) in enumerate(zip(params, gW + gb)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= LR * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0:
            vl, _ = evaluate(net, Xva, yva)
            if vl < best_v:
                best_v = vl
                best_t = evaluate(net, Xte, yte)[0]
    opt.append(best_t - best_v)
    print(f"{seed:>6} {best_v:>15.4f} {best_t:>25.4f} "
          f"{best_t - best_v:>10.4f}")
print(f"\nmean optimism: {float(np.mean(opt)):+.4f} "
      f"(positive means the validation number FLATTERS the model)")
print(f"consistent in sign across seeds: "
      f"{all(o > 0 for o in opt) or all(o < 0 for o in opt)}")
print("\nThe two sets are exchangeable by construction, so in the absence")
print("of any selection they should agree up to sampling noise. The")
print("systematic component is the price of having CHOSEN the stopping")
print("step by looking at the validation set — the selection effect of")
print("Chapter 43, arriving through a different door.")
print("\nThe magnitude here is small, and that is worth saying rather than")
print("overstating: with a validation set of this size and checkpoints")
print("every 250 steps, there are few opportunities to overfit the split.")
print("It grows with the number of decisions taken on the same set, which")
print("in a real project is not one but dozens.")
print("\nThe fix is Chapter 43's: a third split that no decision touches.")
