# -*- coding: utf-8 -*-
# Extracted from: Chapter 122 — Vision Transformers
# Source: src/.../ch122-vision-transformers.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What the convolutional prior buys, and what it forbids.

cite:dosovitskiy2021vit's claim is routinely quoted without its condition. A
transformer over image patches beats a convolutional network GIVEN ENOUGH DATA,
and underperforms it below that, because the convolution's locality and
translation-equivariance are assumptions the transformer has to learn from
examples instead (eq:prior-as-data).

An assumption is not free in either direction. This listing runs two tasks:

  TASK A  which shape is present?   -- translation-invariant, exactly what
                                       eq:pooling-invariance was built for
  TASK B  which half is it in?      -- a question ABOUT position

A conv net with global pooling is invariant to translation by construction, so on
task B it is not merely worse, it is structurally incapable: eq:invariance-forbids
says its output cannot depend on something its representation discarded. That is
the cost of a prior, and it is usually left out of the comparison.
"""
import numpy as np

rng = np.random.default_rng(41)

H = 24
P = 4                     # patch side for the transformer
T = (H // P) ** 2         # tokens
DP = P * P                # raw patch dimension
DM = 24                   # model width
K = 5                     # conv kernel / shape template size
HID = 24

TPL = np.zeros((3, K, K))
TPL[0, 2, :] = 1; TPL[0, :, 2] = 1            # cross
TPL[1, 0, :] = 1; TPL[1, :, 0] = 1            # corner
TPL[2, 1:4, 1:4] = 1                          # block


def place(img, t, r, c):
    img[r:r + K, c:c + K] = np.maximum(img[r:r + K, c:c + K], TPL[t])


def task_a(n):
    """Which shape is present? One shape, anywhere."""
    X = np.zeros((n, H, H)); y = rng.integers(0, 3, size=n)
    for i in range(n):
        r, c = rng.integers(0, H - K + 1, size=2)
        place(X[i], y[i], r, c)
    return X + 0.06 * rng.normal(size=X.shape), y


def task_b(n):
    """Which half of the image is the shape in?

    One shape, its TYPE chosen at random and independent of the label, so the
    two classes contain exactly the same distribution of image CONTENT and
    differ only in where that content sits. A model whose features are
    invariant to translation therefore cannot separate them at all -- not
    poorly, but provably at chance, because eq:pooling-invariance says its
    representation is identical for the two classes.
    """
    X = np.zeros((n, H, H)); y = np.zeros(n, dtype=int)
    for i in range(n):
        t = int(rng.integers(0, 3))
        left = int(rng.integers(0, 2))
        # Both bands sit well inside the frame. That matters: a shape near an
        # edge produces a different set of PARTIAL window views than the same
        # shape near the opposite edge, so translation equivariance is only
        # exact away from the borders (ch:mm-cv-fundamentals). Keeping both
        # bands interior removes that leak, so the only remaining difference
        # between the classes really is position.
        c = int(rng.integers(4, 7)) if left else int(rng.integers(H - K - 6, H - K - 3))
        r = int(rng.integers(4, H - K - 3))
        place(X[i], t, r, c)
        y[i] = left
    return X + 0.06 * rng.normal(size=X.shape), y


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean(), \
        (p - np.eye(p.shape[1])[y]) / len(y)


class Conv:
    """Shared filters at every position, then a global max over positions --
    translation-equivariant then translation-INVARIANT (eq:pooling-invariance)."""

    def __init__(self, ncls):
        self.F = rng.normal(scale=np.sqrt(2 / (K * K)), size=(K * K, HID))
        self.bf = np.zeros(HID)
        self.W = rng.normal(scale=np.sqrt(2 / HID), size=(HID, ncls))
        self.b = np.zeros(ncls)

    def patches(self, X):
        n, S = len(X), H - K + 1
        out = np.empty((n, S * S, K * K))
        for i in range(S):
            for j in range(S):
                out[:, i * S + j] = X[:, i:i + K, j:j + K].reshape(n, -1)
        return out

    def forward(self, X):
        self.p = self.patches(X)
        self.a = np.maximum(self.p @ self.F + self.bf, 0)
        self.arg = self.a.argmax(axis=1)
        self.h = np.take_along_axis(self.a, self.arg[:, None, :], axis=1)[:, 0]
        return self.h @ self.W + self.b

    def step(self, g, lr):
        gW, gb = self.h.T @ g, g.sum(0)
        gh = (g @ self.W.T) * (self.h > 0)
        gF = np.zeros_like(self.F)
        for c in range(HID):
            gF[:, c] = self.p[np.arange(len(gh)), self.arg[:, c]].T @ gh[:, c]
        self.F -= lr * gF; self.bf -= lr * gh.sum(0)
        self.W -= lr * gW; self.b -= lr * gb


class ViT:
    """Patch embed, learned position embeddings, one self-attention block, mean
    pool. No locality prior and no translation equivariance -- position is an
    explicit learned input rather than an architectural assumption."""

    def __init__(self, ncls):
        s = np.sqrt(2 / DP)
        self.We = rng.normal(scale=s, size=(DP, DM)); self.be = np.zeros(DM)
        self.Pe = rng.normal(scale=0.1, size=(T, DM))
        sm = np.sqrt(1 / DM)
        self.Wq = rng.normal(scale=sm, size=(DM, DM))
        self.Wk = rng.normal(scale=sm, size=(DM, DM))
        self.Wv = rng.normal(scale=sm, size=(DM, DM))
        self.Wo = rng.normal(scale=np.sqrt(2 / DM), size=(DM, ncls))
        self.bo = np.zeros(ncls)

    def tokens(self, X):
        n = len(X)
        g = H // P
        out = np.empty((n, T, DP))
        for i in range(g):
            for j in range(g):
                out[:, i * g + j] = X[:, i*P:(i+1)*P, j*P:(j+1)*P].reshape(n, -1)
        return out

    def forward(self, X):
        self.tk = self.tokens(X)
        self.Z = self.tk @ self.We + self.be + self.Pe
        self.Q, self.Kx, self.V = self.Z @ self.Wq, self.Z @ self.Wk, self.Z @ self.Wv
        S = self.Q @ self.Kx.transpose(0, 2, 1) / np.sqrt(DM)
        S -= S.max(axis=-1, keepdims=True)
        A = np.exp(S); self.A = A / A.sum(axis=-1, keepdims=True)
        self.O = self.A @ self.V
        self.Hh = self.Z + self.O
        self.pool = self.Hh.mean(axis=1)
        return self.pool @ self.Wo + self.bo

    def step(self, g, lr):
        gWo, gbo = self.pool.T @ g, g.sum(0)
        gH = np.repeat((g @ self.Wo.T)[:, None, :], T, axis=1) / T
        gZ, gO = gH.copy(), gH.copy()
        gA = gO @ self.V.transpose(0, 2, 1)
        gV = self.A.transpose(0, 2, 1) @ gO
        gS = self.A * (gA - (gA * self.A).sum(axis=-1, keepdims=True))
        gS /= np.sqrt(DM)
        gQ = gS @ self.Kx
        gK = gS.transpose(0, 2, 1) @ self.Q
        gWq = (self.Z.transpose(0, 2, 1) @ gQ).sum(0)
        gWk = (self.Z.transpose(0, 2, 1) @ gK).sum(0)
        gWv = (self.Z.transpose(0, 2, 1) @ gV).sum(0)
        gZ += gQ @ self.Wq.T + gK @ self.Wk.T + gV @ self.Wv.T
        gWe = (self.tk.transpose(0, 2, 1) @ gZ).sum(0)
        gbe = gZ.sum(axis=(0, 1))
        gPe = gZ.sum(axis=0)
        for p, gp in ((self.Wo, gWo), (self.bo, gbo), (self.Wq, gWq),
                      (self.Wk, gWk), (self.Wv, gWv), (self.We, gWe),
                      (self.be, gbe), (self.Pe, gPe)):
            p -= lr * gp


def run(model_cls, task, n_train, ncls, epochs=40, lr=0.05):
    Xtr, ytr = task(n_train)
    Xte, yte = task(1200)
    m = model_cls(ncls)
    for _ in range(epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, 32):
            b = order[s:s + 32]
            _, g = softmax_ce(m.forward(Xtr[b]), ytr[b])
            m.step(g, lr)
    return float((m.forward(Xte).argmax(1) == yte).mean())


SIZES = (100, 400, 1600, 6400)
print(f"{H}x{H} images. Conv: shared {K}x{K} filters + global max pool.")
print(f"ViT: {P}x{P} patches, learned positions, 1 attention block, mean pool.\n")
print(f"{'train size':>11}{'':>3}{'TASK A: which shape':>26}{'':>4}"
      f"{'TASK B: which half':>26}")
print(f"{'':>11}{'':>3}{'conv':>12}{'ViT':>14}{'':>4}{'conv':>12}{'ViT':>14}")
print("-" * 74)

res = {}
for n in SIZES:
    a_c = run(Conv, task_a, n, 3)
    a_v = run(ViT, task_a, n, 3)
    b_c = run(Conv, task_b, n, 2)
    b_v = run(ViT, task_b, n, 2)
    res[n] = (a_c, a_v, b_c, b_v)
    print(f"{n:>11}{'':>3}{a_c:>12.3f}{a_v:>14.3f}{'':>4}{b_c:>12.3f}{b_v:>14.3f}")

lo, hi = SIZES[0], SIZES[-1]
print(f"""
Task A is the case the convolutional prior was designed for, and the sample
efficiency shows. At {lo} training examples the conv model reaches
{res[lo][0]:.3f} while the transformer manages {res[lo][1]:.3f} -- chance for a
three-way choice. The conv model already knows that a shape means the same thing
wherever it appears; the transformer is still learning that from examples, and by
{hi} examples it has reached {res[hi][1]:.3f} and is still climbing.

Look at the SHAPE of the transformer's column, because it is characteristic. It
does not improve gradually from 100 to 1600 -- it sits at chance, learning
nothing, and then rises steeply. It is not slowly approximating the prior; it is
failing until it has enough data to discover the regularity, then learning
quickly. eq:prior-as-data's exchange rate here is at least 64x, on a task whose
entire content IS the prior.

That is cite:dosovitskiy2021vit's claim with its condition attached, and the
condition is the part that decides which architecture to use. Quoting
"transformers beat convnets" without the data scale drops the only operative
detail.

Task B is what the prior FORBIDS, and it is the half of the comparison usually
missing. The question is about position -- which half of the frame the shape is
in -- and the shape's TYPE is random and independent of the label, so the two
classes contain identical image content and differ only in where it sits. The
conv model's global max pool discarded position by construction, so
eq:invariance-forbids says the two classes have identical representations.

The measurement is flat at chance -- {res[lo][2]:.3f}, {res[400][2]:.3f},
{res[1600][2]:.3f}, {res[hi][2]:.3f} -- across a 64-fold increase in training
data. That is not a model that needs more examples. More data cannot help, more
depth cannot help, and a better optimiser cannot help, because the function is
outside the hypothesis space. The transformer reaches {res[hi][3]:.3f} on the
same task with the same budget, because its position embeddings are an INPUT
rather than an assumption: it can learn to be translation-invariant when that is
right, and learn not to be when it is not.

So the trade is not "prior versus no prior, with data deciding". It is: a prior is
a constraint; constraints buy sample efficiency inside their scope and cost
everything outside it; and the transformer's advantage at scale is partly that it
has fewer constraints to be wrong about. That also explains why hybrid designs
keep reappearing -- a convolutional stem where the assumption holds, attention
above for the relations a convolution cannot represent.""")
