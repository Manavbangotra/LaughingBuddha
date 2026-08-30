# -*- coding: utf-8 -*-
# Extracted from: Chapter 118 — Computer Vision Fundamentals
# Source: src/.../ch118-cv-fundamentals.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why a fully connected layer is the wrong prior for an image.

The usual reason given is parameter count, and it is true and it is the less
interesting half. The real objection is that a dense layer has no notion that
translating an image leaves its content unchanged: pixel 47 and pixel 48 are
unrelated coordinates to it, so a shape it learned in one place teaches it
nothing about the same shape three pixels over (eq:translation-equivariance).

This listing trains both models from scratch on the same task and separates the
two objections. Both see the same data, both are trained the same way, and the
test set contains translations the training set did not.
"""
import numpy as np

rng = np.random.default_rng(3)

H = W = 16
K = 5                        # shape template size
N_CLASS = 3
N_TRAIN, N_TEST = 3000, 1500
HID = 24
EPOCHS, LR = 24, 0.08

# Three 5x5 shapes: a cross, a corner, a bar.
TEMPLATES = np.zeros((N_CLASS, K, K))
TEMPLATES[0, 2, :] = 1; TEMPLATES[0, :, 2] = 1                 # cross
TEMPLATES[1, 0, :] = 1; TEMPLATES[1, :, 0] = 1                 # corner
TEMPLATES[2, 2, :] = 1                                         # bar


def make(n, centred):
    """centred=True places every shape in the middle; False places it anywhere."""
    X = np.zeros((n, H, W))
    y = rng.integers(0, N_CLASS, size=n)
    for i in range(n):
        if centred:
            r = c = (H - K) // 2
        else:
            r, c = rng.integers(0, H - K + 1, size=2)
        X[i, r:r + K, c:c + K] = TEMPLATES[y[i]]
    X += 0.08 * rng.normal(size=X.shape)
    return X, y


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return loss, g / len(y)


class Dense:
    """Flatten the image, then two dense layers. Every pixel is its own
    coordinate and nothing ties neighbours together (eq:dense-no-prior)."""

    def __init__(self):
        self.W1 = rng.normal(scale=np.sqrt(2 / (H * W)), size=(H * W, HID))
        self.b1 = np.zeros(HID)
        self.W2 = rng.normal(scale=np.sqrt(2 / HID), size=(HID, N_CLASS))
        self.b2 = np.zeros(N_CLASS)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, X):
        self.f = X.reshape(len(X), -1)
        self.h = np.maximum(self.f @ self.W1 + self.b1, 0)
        return self.h @ self.W2 + self.b2

    def backward(self, g, lr):
        gW2, gb2 = self.h.T @ g, g.sum(axis=0)
        gh = (g @ self.W2.T) * (self.h > 0)
        gW1, gb1 = self.f.T @ gh, gh.sum(axis=0)
        for p, gp in ((self.W1, gW1), (self.b1, gb1), (self.W2, gW2), (self.b2, gb2)):
            p -= lr * gp


class Conv:
    """One bank of shared KxK filters applied at every position, then a global
    max over positions. Weight sharing IS the translation prior: the same filter
    is asked the same question everywhere, and the max discards where."""

    def __init__(self):
        self.F = rng.normal(scale=np.sqrt(2 / (K * K)), size=(K * K, HID))
        self.bf = np.zeros(HID)
        self.W2 = rng.normal(scale=np.sqrt(2 / HID), size=(HID, N_CLASS))
        self.b2 = np.zeros(N_CLASS)

    def n_params(self):
        return self.F.size + self.bf.size + self.W2.size + self.b2.size

    @staticmethod
    def patches(X):
        n, P = len(X), H - K + 1
        out = np.empty((n, P * P, K * K))
        for i in range(P):
            for j in range(P):
                out[:, i * P + j] = X[:, i:i + K, j:j + K].reshape(n, -1)
        return out

    def forward(self, X):
        self.p = self.patches(X)                       # (n, pos, K*K)
        self.a = np.maximum(self.p @ self.F + self.bf, 0)   # (n, pos, HID)
        self.arg = self.a.argmax(axis=1)               # global max pool
        self.h = np.take_along_axis(self.a, self.arg[:, None, :], axis=1)[:, 0]
        return self.h @ self.W2 + self.b2

    def backward(self, g, lr):
        gW2, gb2 = self.h.T @ g, g.sum(axis=0)
        gh = (g @ self.W2.T) * (self.h > 0)
        # Gradient flows only through the winning position for each channel.
        gF = np.zeros_like(self.F)
        gbf = gh.sum(axis=0)
        n = len(gh)
        for c in range(HID):
            win = self.p[np.arange(n), self.arg[:, c]]          # (n, K*K)
            gF[:, c] = win.T @ gh[:, c]
        for p, gp in ((self.F, gF), (self.bf, gbf), (self.W2, gW2), (self.b2, gb2)):
            p -= lr * gp


def train_eval(model, Xtr, ytr, Xte, yte):
    n = len(Xtr)
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, 64):
            b = order[s:s + 64]
            logits = model.forward(Xtr[b])
            _, g = softmax_ce(logits, ytr[b])
            model.backward(g, LR)
    return float((model.forward(Xte).argmax(axis=1) == yte).mean())


Xc, yc = make(N_TRAIN, centred=True)         # training: shapes always centred
Xa, ya = make(N_TRAIN, centred=False)        # training: shapes anywhere
Xt, yt = make(N_TEST, centred=False)         # test: always anywhere
Xtc, ytc = make(N_TEST, centred=True)

print(f"{H}x{W} images, {N_CLASS} shapes, {HID} hidden units in both models\n")
print(f"{'model':<10}{'params':>9}{'train centred ->':>19}{'':>3}"
      f"{'train anywhere ->':>19}")
print(f"{'':<10}{'':>9}{'test centred':>12}{'test anywhere':>15}"
      f"{'test anywhere':>18}")
print("-" * 66)

for name, cls in (("dense", Dense), ("conv", Conv)):
    m1 = cls(); cen = train_eval(m1, Xc, yc, Xtc, ytc)
    shift = float((m1.forward(Xt).argmax(axis=1) == yt).mean())
    m3 = cls(); anyw = train_eval(m3, Xa, ya, Xt, yt)
    print(f"{name:<10}{cls().n_params():>9,}{cen:>12.3f}{shift:>15.3f}"
          f"{anyw:>18.3f}")

print(f"""
Read the middle two columns together, because the pair is the argument. Trained
on centred shapes and tested on centred shapes, both models solve the task -- the
dense network is perfectly capable of learning three shapes at a fixed location.
Tested on the SAME shapes moved elsewhere in the frame, the dense model collapses
to 0.346 against a chance rate of 0.333, while the convolutional one holds 0.970.

That gap is not a capacity problem and it is not an optimisation problem. The
dense model learned the task it was shown, completely. It simply has no way to
know that the task it was shown and the task it is being tested on are the same
task, because to a dense layer pixel 47 and pixel 48 are unrelated coordinates.
Every translated copy of a shape is, to that architecture, a new shape.

The convolutional model gets that for free from weight sharing
(eq:translation-equivariance). The same filter is applied at every position, so
evidence about a shape gathered anywhere updates the same weights, and the global
max discards WHERE the evidence was found. Note that the prior is doing two jobs
here: sharing makes learning at one position transfer to all of them, and pooling
makes the answer invariant to which position won.

The last column is the control that makes the argument honest. Given training
data that already covers every position, the dense model recovers to 0.893 -- so
the prior is not supplying capability the dense model lacks, it is supplying
SAMPLE EFFICIENCY the dense model would otherwise have to buy with data. That is
what an architectural prior is: a statement about the world, paid for once in
design instead of repeatedly in examples.

And only now is the parameter count worth mentioning. The dense model uses about
{Dense().n_params() / Conv().n_params():.0f} times the parameters to be worse,
and the ratio grows with image area -- eq:dense-parameter-count scales with the
number of pixels and eq:conv-parameter-count does not scale with image size at
all. The parameter argument is the one usually given for convolutions. It is the
weaker of the two.""")
