# Extracted from: Chapter 55 — Learning-Rate Schedules and Warmup
# Source: src/.../ch055-lr-schedules.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Six schedules on the same network at the same budget, and the two
couplings that catch people out: the total step budget and batch size.
"""
import numpy as np

rng = np.random.default_rng(11)


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

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
        z = self.forward(X)
        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - z[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = (d @ self.W[l].T) * (self.Z[l - 1] > 0)
        return loss, gW, gb


D, C = 24, 5
_rs = np.random.default_rng(99)
W1T = _rs.normal(size=(D, 16))
W2T = _rs.normal(size=(16, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ W1T) @ W2T * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # vectorised categorical sampling: inverse-CDF on one uniform per row
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


# 60k training examples against ~6k parameters: large enough that the
# comparison measures optimisation rather than which schedule happens to
# early-stop most. At 8k the network overfits and every row's test loss
# RISES with training, which measures regularisation instead.
Xtr, ytr = make_data(60000, 1)
Xte, yte = make_data(20000, 2)
_p = np.exp(np.tanh(Xte @ W1T) @ W2T * 1.5)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Adam:
    def __init__(self, shape, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v = np.zeros(shape), np.zeros(shape)
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, p, g, t, lr):
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        return p - lr * mh / (np.sqrt(vh) + self.eps)


def cosine(t, T, e0, e_min=0.0):
    return e_min + 0.5 * (e0 - e_min) * (1 + np.cos(np.pi * min(t, T) / T))


SCHED = {
    "constant": lambda t, T, e0: e0,
    "step x0.1 at 50/75%": lambda t, T, e0: e0 * (
        0.1 ** ((t >= T // 2) + (t >= 3 * T // 4))),
    "exponential to 1%": lambda t, T, e0: e0 * 0.01 ** (t / T),
    "linear to 0": lambda t, T, e0: e0 * max(0.0, 1 - t / T),
    "cosine to 0": cosine,
    "warmup 5% + cosine": lambda t, T, e0: (
        e0 * (t + 1) / (T // 20) if t < T // 20
        else cosine(t - T // 20, T - T // 20, e0)),
}


def train(sched, steps=20000, lr=3e-3, batch=128, seed=0, trace=()):
    net = MLP([D, 64, 64, C], seed=seed)
    opts = [Adam(W.shape) for W in net.W] + [Adam(b.shape) for b in net.b]
    rs = np.random.default_rng(seed + 50)
    hist = {}
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        eta = sched(t - 1, steps, lr)
        for i, (W, g) in enumerate(zip(net.W, gW)):
            net.W[i] = opts[i].step(W, g, t, eta)
        for i, (b, g) in enumerate(zip(net.b, gb)):
            net.b[i] = opts[len(net.W) + i].step(b, g, t, eta)
        if t in trace:
            hist[t] = net.loss_and_grads(Xte, yte)[0]
    te, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    return te, acc, hist


print("=" * 72)
print("six schedules, same budget, same peak learning rate")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}\n")
TRACE = (1000, 4000, 8000, 14000, 20000)
print(f"{'schedule':<22} " + " ".join(f"{f'@{x}':>9}" for x in TRACE)
      + f" {'test acc':>10} {'excess':>9}")
for name, fn in SCHED.items():
    te, acc, hist = train(fn, trace=TRACE)
    print(f"{name:<22} " + " ".join(f"{hist[x]:>9.4f}" for x in TRACE)
          + f" {acc:>10.4f} {te - BAYES:>9.4f}")

print("\nThe 'excess' column is the test loss above the Bayes floor, which")
print("is the only fair way to compare — the irreducible part of the loss")
print("is the same for every row and including it compresses the")
print("differences.")
print("\nRead the trace columns left to right. Early in the run the")
print("constant schedule is ahead of every decaying one, because it takes")
print("the largest steps and the network is still far from anything good.")
print("The decaying schedules catch and pass it later, as their noise")
print("floors come down — eq. 55.10 acting on a real network.")
print("\nThe crossover point is the thing to notice, and it is why the")
print("budget matters. Through the first few thousand steps the six rows")
print("are within a few thousandths of each other — a comparison stopped")
print("there would report that the schedule makes no difference. By 20000")
print("steps the best row's excess loss is a third lower than the worst's.")
print("Nothing about the schedules changed; the budget did.")
print("\nCompare this also with the one-dimensional experiment above, which")
print("preferred the FASTEST decay of all. There the only obstacle was the")
print("noise floor. Here there is genuine optimisation to do as well, so a")
print("schedule that decays too early — exponential-to-1% — gives up")
print("progress it cannot recover. Cosine's shape sits between the two")
print("pressures, which is what it was designed to do.")

# --- the budget coupling ----------------------------------------------------
print("\n" + "=" * 72)
print("a cosine schedule must know the budget in advance (section 5.5)")
print("=" * 72)
print("Same total steps run in every row; only the T the schedule was")
print("PLANNED for differs.\n")
print(f"{'planned T':>11} {'actual steps':>13} {'final lr / peak':>17} "
      f"{'test loss':>11} {'excess':>9}")
ACTUAL = 20000
for planned in (20000, 40000, 100000, 1000000):
    fn = lambda t, T, e0, P=planned: cosine(t, P, e0)
    te, acc, _ = train(fn, steps=ACTUAL)
    print(f"{planned:>11} {ACTUAL:>13} "
          f"{cosine(ACTUAL, planned, 1.0):>17.4f} {te:>11.4f} "
          f"{te - BAYES:>9.4f}")

print("\nRead the 'final lr / peak' column alongside the excess. A cosine")
print("planned for fifty times its actual budget has barely decayed at all")
print("— it ends at almost its full peak rate — so it IS a constant")
print("schedule wearing a cosine's name, and it lands on the constant")
print("schedule's excess loss from the previous table.")
print("\nThis is the practical trap, and it is worth stating in both")
print("directions. You cannot stop a cosine run early and get the model the")
print("schedule was going to give you, because the decay that does the work")
print("is all at the end. And you cannot extend a run that finished without")
print("re-planning, because the schedule is already at zero.")
print("\nInverse-square-root exists precisely because it does not have this")
print("problem: eq. 55.4 is a function of t alone, so it is meaningful at")
print("any stopping point. That is why it reappeared for very large models,")
print("where the budget is decided while the run is already going.")

# --- the batch-size coupling ------------------------------------------------
print("\n" + "=" * 72)
print("the linear scaling rule (eq. 55.6)")
print("=" * 72)
print("Batch size k times larger at 1/k the steps — the SAME number of")
print("examples seen. Does scaling the rate by k preserve the result?\n")
print(f"{'batch':>7} {'steps':>7} {'lr rule':<16} {'lr':>9} "
      f"{'test loss':>11} {'excess':>9}")
BASE_B, BASE_LR, BASE_STEPS = 32, 1e-3, 32000
for k in (1, 4, 16, 64):
    for rule, lr in (("unscaled", BASE_LR),
                     ("linear (x k)", BASE_LR * k),
                     ("sqrt (x sqrt k)", BASE_LR * np.sqrt(k))):
        if k == 1 and rule != "unscaled":
            continue
        te, acc, _ = train(SCHED["cosine to 0"], steps=BASE_STEPS // k,
                           lr=lr, batch=BASE_B * k)
        print(f"{BASE_B * k:>7} {BASE_STEPS // k:>7} {rule:<16} {lr:>9.1e} "
              f"{te:>11.4f} {te - BAYES:>9.4f}")

print("\nEvery row sees the same number of examples, so the question is")
print("purely which learning rate makes a large-batch run behave like the")
print("small-batch one.")
print("\nRead the unscaled rows down the table first. As the batch grows the")
print("run gets fewer steps at the same step size, so it covers less ground")
print("and the excess loss degrades — which is the failure eq. 55.6 exists")
print("to prevent.")
print("\nBetween the two rules, linear scaling wins at every batch size")
print("here and the square-root rule undercorrects — it recovers part of")
print("the loss and not all of it. That is eq. 55.13 working as derived.")
print("\nOne honest complication. The linearly scaled runs at k = 4 and")
print("k = 16 do not merely match the k = 1 baseline, they BEAT it, which")
print("no scaling rule promises. The explanation is that the baseline's")
print("1e-3 was itself below the best rate for batch 32, so scaling it up")
print("improved the run on its own merits as well as compensating for the")
print("batch. A clean test of eq. 55.6 needs the baseline rate tuned first,")
print("and this table conflates the two effects at small k.")
print("\nWhat is not conflated is the trend at large k. By k = 64 even the")
print("linearly scaled run has fallen behind the baseline, which is the")
print("regime where eq. 55.13's assumption — that the gradient does not")
print("change over the k steps being merged — has stopped holding. That is")
print("where warmup becomes necessary and, beyond it, where LARS and LAMB")
print("replace the rule entirely.")
