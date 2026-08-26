# -*- coding: utf-8 -*-
# Extracted from: Chapter 61 — Autoencoders and Representation Learning
# Source: src/.../ch061-autoencoders.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The variational autoencoder: why the reparameterisation trick is
necessary, what the KL term buys, and posterior collapse.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.3: two gradient estimators -----------------------------------
print("=" * 72)
print("why reparameterisation and not the score function (eqs. 61.13-14)")
print("=" * 72)
print("Both estimate d/dmu E_{z~N(mu,1)}[f(z)] and both are UNBIASED. The")
print("question is their variance.\n")


def compare_estimators(mu, n_samples, f, df, trials=2000, seed=0):
    rs = np.random.default_rng(seed)
    score, path = [], []
    for _ in range(trials):
        eps = rs.normal(size=n_samples)
        z = mu + eps
        # eq. 61.13: f(z) * d/dmu log N(z; mu, 1) = f(z) * (z - mu)
        score.append(float(np.mean(f(z) * (z - mu))))
        # eq. 61.14: d/dmu f(mu + eps) = f'(mu + eps)
        path.append(float(np.mean(df(z))))
    return np.array(score), np.array(path)


f = lambda z: z ** 2
df = lambda z: 2 * z
MU = 1.5
true_grad = 2 * MU                          # d/dmu E[(mu+eps)^2] = 2 mu
print(f"f(z) = z^2, mu = {MU}, true gradient = {true_grad}\n")
print(f"{'samples':>9} {'score-fn mean':>15} {'score-fn sd':>13} "
      f"{'pathwise mean':>15} {'pathwise sd':>13} {'variance ratio':>16}")
for n in (1, 4, 16, 64):
    sc, pa = compare_estimators(MU, n, f, df)
    print(f"{n:>9} {sc.mean():>15.4f} {sc.std():>13.4f} "
          f"{pa.mean():>15.4f} {pa.std():>13.4f} "
          f"{(sc.std() / max(pa.std(), 1e-12)) ** 2:>16.1f}x")

print("\nBoth means sit on the true gradient — both estimators are")
print("unbiased, as eqs. 61.13 and 61.14 say — and the pathwise estimator's")
print("variance is an order of magnitude lower in one dimension.")

# and how it scales with DIMENSION, which is the case that matters
print("\nOne dimension understates it. A VAE's latent has many, and the")
print("score-function estimator's variance grows with that number while")
print("the pathwise estimator's does not:\n")
print(f"{'latent dim':>12} {'score-fn sd':>14} {'pathwise sd':>14} "
      f"{'variance ratio':>16}")
for d in (1, 4, 16, 64, 256):
    rs = np.random.default_rng(3)
    mu = np.full(d, 0.3)
    sc, pa = [], []
    for _ in range(1500):
        eps = rs.normal(size=d)
        z = mu + eps
        fv = float(np.sum(z ** 2))          # scalar objective, as a loss is
        sc.append(fv * (z - mu))            # eq. 61.13, per coordinate
        pa.append(2 * z)                    # eq. 61.14
    sc, pa = np.array(sc), np.array(pa)
    s_sd = float(sc.std(axis=0).mean())
    p_sd = float(pa.std(axis=0).mean())
    print(f"{d:>12} {s_sd:>14.4f} {p_sd:>14.4f} "
          f"{(s_sd / p_sd) ** 2:>16.1f}x")

print("\nThe reason is what each estimator uses. The score-function")
print("estimator sees only f's VALUE — one scalar — and has to infer a")
print("direction in d dimensions from the correlation between that scalar")
print("and the displacement. The pathwise estimator uses f's GRADIENT,")
print("which is a d-dimensional object that already points the right way.")
print("\nSo the score-function estimator is extracting d numbers' worth of")
print("information from one number, and its variance grows accordingly.")
print("At a realistic latent size the gap is several orders of magnitude.")
print("\nThat is why VAEs are trainable, and why the reparameterisation")
print("trick appears far outside generative modelling — anywhere a gradient")
print("has to pass through a sampling step.")

# --- a working VAE ----------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class VAE:
    """Eqs. 61.6-61.8, with the reparameterisation of eq. 61.10."""

    def __init__(self, d, k, hidden=64, seed=0):
        rs = np.random.default_rng(seed)
        self.We = rs.normal(0, np.sqrt(2 / d), (d, hidden))
        self.be = np.zeros(hidden)
        self.Wmu = rs.normal(0, np.sqrt(1 / hidden), (hidden, k))
        self.bmu = np.zeros(k)
        self.Wlv = rs.normal(0, np.sqrt(1 / hidden), (hidden, k))
        self.blv = np.zeros(k)
        self.Wd1 = rs.normal(0, np.sqrt(2 / k), (k, hidden))
        self.bd1 = np.zeros(hidden)
        self.Wd2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, d))
        self.bd2 = np.zeros(d)
        self.k = k

    def params(self):
        return [self.We, self.be, self.Wmu, self.bmu, self.Wlv, self.blv,
                self.Wd1, self.bd1, self.Wd2, self.bd2]

    def encode(self, X):
        h = np.tanh(X @ self.We + self.be)
        return h, h @ self.Wmu + self.bmu, h @ self.Wlv + self.blv

    def decode(self, z):
        h = np.tanh(z @ self.Wd1 + self.bd1)
        return h, h @ self.Wd2 + self.bd2

    def loss_and_grads(self, X, beta=1.0, rs=None):
        n = len(X)
        h, mu, logvar = self.encode(X)
        sd = np.exp(0.5 * logvar)
        eps = rs.normal(size=mu.shape)
        z = mu + sd * eps                                   # eq. 61.10
        hd, xr = self.decode(z)
        rec = float(np.sum((xr - X) ** 2) / n)
        kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1)  # eq. 61.9
        kl = float(kl_per.sum() / n)
        # gradients
        dxr = 2 * (xr - X) / n
        gWd2, gbd2 = hd.T @ dxr, dxr.sum(axis=0)
        dhd = (dxr @ self.Wd2.T) * (1 - hd ** 2)
        gWd1, gbd1 = z.T @ dhd, dhd.sum(axis=0)
        dz = dhd @ self.Wd1.T
        dmu = dz + beta * mu / n                            # rec + KL
        dlogvar = dz * (0.5 * sd * eps) + beta * 0.5 * (
            np.exp(logvar) - 1) / n
        gWmu, gbmu = h.T @ dmu, dmu.sum(axis=0)
        gWlv, gblv = h.T @ dlogvar, dlogvar.sum(axis=0)
        dh = (dmu @ self.Wmu.T + dlogvar @ self.Wlv.T) * (1 - h ** 2)
        gWe, gbe = X.T @ dh, dh.sum(axis=0)
        return (rec, kl,
                [gWe, gbe, gWmu, gbmu, gWlv, gblv, gWd1, gbd1, gWd2, gbd2])


def make_blobs(n, d=16, seed=0):
    rs = np.random.default_rng(seed)
    centres = np.random.default_rng(42).normal(size=(4, d)) * 2.0
    idx = rs.integers(0, 4, n)
    return centres[idx] + rs.normal(0, 0.5, (n, d)), idx


Xv, yv = make_blobs(8000, seed=1)
Xvt, yvt = make_blobs(4000, seed=2)
Xv = Xv - Xv.mean(axis=0)
Xvt = Xvt - Xvt.mean(axis=0)


def train_vae(net, X, beta=1.0, steps=5000, lr=2e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 3)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, _, gs = net.loss_and_grads(xb, beta, rs)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


# --- section 4.3 and 5.4: what the KL term buys -----------------------------
print("\n" + "=" * 72)
print("what the KL term buys: a latent space you can SAMPLE from (5.4)")
print("=" * 72)
print("beta = 0 is a plain autoencoder; beta = 1 is a VAE. Read the last")
print("two columns: how close the aggregate code distribution is to the")
print("N(0, I) prior we would sample from.\n")
print(f"{'beta':>7} {'test recon MSE':>16} {'KL':>9} {'active dims':>13} "
      f"{'|mean(z)|':>11} {'mean sd(z)':>12} {'decode-random MSE':>19}")
rs_eval = np.random.default_rng(11)
for beta in (0.0, 0.1, 1.0, 4.0, 20.0):
    net = train_vae(VAE(16, 6, seed=7), Xv, beta=beta)
    h, mu, logvar = net.encode(Xvt)
    z = mu + np.exp(0.5 * logvar) * rs_eval.normal(size=mu.shape)
    _, xr = net.decode(z)
    rec = float(np.mean((xr - Xvt) ** 2))
    kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).mean(axis=0)
    active = int((kl_per > 0.01).sum())
    # decode points drawn from the PRIOR: does the decoder know what to do?
    zp = rs_eval.normal(size=(2000, 6))
    _, xp = net.decode(zp)
    # nearest real point: how far are the generated samples from the data?
    d2 = ((xp[:, None, :] - Xvt[None, :400, :]) ** 2).sum(axis=2)
    gen = float(d2.min(axis=1).mean() / Xvt.shape[1])
    print(f"{beta:>7.1f} {rec:>16.5f} {kl_per.sum():>9.3f} {active:>13} "
          f"{np.abs(mu.mean(axis=0)).max():>11.4f} "
          f"{np.exp(0.5 * logvar).mean():>12.4f} {gen:>19.5f}")

print("\nAt beta = 0 there is no pressure toward the prior at all: the codes")
print("go wherever reconstruction wants them. Reconstruction is best, and")
print("the last column — how close a point decoded from a PRIOR sample")
print("lands to real data — is worst. That is section 4.3's failure,")
print("measured: the decoder was never trained on the codes you are")
print("sampling.")
print("\nAs beta rises the codes are pulled toward N(0, I) — watch mean")
print("sd(z) climb toward 1 and the KL fall — the decoder sees something")
print("much closer to what you will actually sample from, and generated")
print("points land near the data.")
print("\nBut the last column is NOT monotone in beta, and the reason is the")
print("active-dims column beside it. Past a certain point the KL term wins")
print("outright: latent dimensions collapse to the prior, the model has")
print("nothing left to condition on, and its samples get worse again")
print("because it can no longer represent the data rather than because it")
print("cannot sample.")
print("\nSo beta is a dial between two failure modes, not a monotone knob.")
print("Too small and the decoder has never seen the codes you sample; too")
print("large and there is nothing in the codes to decode. The next")
print("experiment measures the second failure directly.")

# --- section 6.4: posterior collapse ----------------------------------------
print("\n" + "=" * 72)
print("posterior collapse: the latent stops carrying information (6.4)")
print("=" * 72)
print("Per-dimension KL. A collapsed dimension has KL near zero, meaning")
print("mu = 0 and sd = 1 for every input — it has become the prior and")
print("carries nothing.\n")
print(f"{'beta':>7} {'active dims (of 6)':>20} {'per-dimension KL':>44}")
for beta in (0.1, 1.0, 4.0, 20.0, 100.0):
    net = train_vae(VAE(16, 6, seed=7), Xv, beta=beta)
    _, mu, logvar = net.encode(Xvt)
    kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).mean(axis=0)
    active = int((kl_per > 0.01).sum())
    print(f"{beta:>7.1f} {active:>20} "
          f"{np.array2string(np.sort(kl_per)[::-1], precision=3, suppress_small=True):>44}")

print("\nAt large beta the KL term dominates and the optimiser finds it")
print("cheaper to make every dimension match the prior exactly than to use")
print("any of them. The active-dimension count falls, and at the extreme")
print("the model reconstructs the data mean and nothing else.")
print("\nThis is the OPTIMUM of the objective, not a failure to converge —")
print("which is what makes it insidious. The loss goes down, the training")
print("looks healthy, and the latent variable the whole model was built")
print("around has quietly stopped existing.")
print("\nThe per-dimension KL is the diagnostic, and it costs one line.")
print("Anyone training a VAE should be logging the active-unit count.")
