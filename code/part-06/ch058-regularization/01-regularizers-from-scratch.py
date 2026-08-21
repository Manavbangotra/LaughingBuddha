# Extracted from: Chapter 58 — Regularization, Dropout, Overfitting, and Underfitting
# Source: src/.../ch058-regularization.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Dropout, weight decay and early stopping, each measured against the
theory that predicts what it should do.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 5.1: inverted dropout ------------------------------------------
def dropout(h, p, rs, training=True):
    """Eq. 58.1. The 1/(1-p) is what makes inference a no-op."""
    if not training or p == 0.0:
        return h
    mask = (rs.random(h.shape) >= p) / (1.0 - p)
    return h * mask


print("=" * 72)
print("inverted dropout: the expectation is preserved (eq. 58.1)")
print("=" * 72)
h = rng.random((2000, 64)) * 2.0
rs = np.random.default_rng(1)
print(f"{'p':>6} {'E[dropout(h)] / h':>20} {'sd / h  (measured)':>21} "
      f"{'predicted sqrt(p/(1-p))':>25}")
for p in (0.0, 0.1, 0.3, 0.5, 0.8):
    samples = np.array([dropout(h, p, rs) for _ in range(200)])
    ratio_mean = float((samples.mean(axis=0) / h).mean())
    ratio_sd = float((samples.std(axis=0) / h).mean())
    pred = np.sqrt(p / (1 - p)) if p < 1 else np.inf
    print(f"{p:>6.1f} {ratio_mean:>20.5f} {ratio_sd:>21.5f} {pred:>25.5f}")

print("\nThe mean ratio is 1.000 at every rate, which is the point of the")
print("1/(1-p) scaling: inference needs no adjustment because training")
print("already matched the expectation.")
print("\nThe standard deviation matches eq. 58.2's sqrt(p/(1-p)) exactly.")
print("At p = 0.5 the injected noise has the same magnitude as the signal,")
print("and the noise is PROPORTIONAL to each activation — large activations")
print("are perturbed more, which is a data-dependent perturbation rather")
print("than additive noise.")

# --- section 6.1: dropout is an L2 penalty for a linear model ---------------
print("\n" + "=" * 72)
print("for a LINEAR model, dropout IS an L2 penalty (eq. 58.5)")
print("=" * 72)
N, D = 400, 12
Xl = rng.normal(size=(N, D))
Xl[:, :4] *= 3.0                                  # some features larger
w_true = rng.normal(size=D)
yl = Xl @ w_true + rng.normal(0, 0.5, N)


def fit_dropout(X, y, p, steps=6000, lr=0.02, seed=0, n_masks=1):
    rs = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        Xd = X if p == 0 else X * ((rs.random(X.shape) >= p) / (1 - p))
        w -= lr * (Xd.T @ (Xd @ w - y)) / len(X)
    return w


def fit_ridge_weighted(X, y, p):
    """Eq. 58.5's closed form: ridge with per-feature weight ||x_j||^2."""
    lam = p / (1 - p)
    Pen = np.diag(lam * (X ** 2).sum(axis=0))
    return np.linalg.solve(X.T @ X + Pen, X.T @ y)


print(f"{'p':>6} {'|w| dropout':>13} {'|w| weighted ridge':>20} "
      f"{'max |diff|':>12} {'cos similarity':>16}")
for p in (0.0, 0.1, 0.3, 0.5):
    wd = fit_dropout(Xl, yl, p)
    wr = fit_ridge_weighted(Xl, yl, p) if p > 0 else np.linalg.lstsq(
        Xl, yl, rcond=None)[0]
    cos = float(wd @ wr / (np.linalg.norm(wd) * np.linalg.norm(wr)))
    print(f"{p:>6.1f} {np.linalg.norm(wd):>13.5f} {np.linalg.norm(wr):>20.5f} "
          f"{np.abs(wd - wr).max():>12.5f} {cos:>16.6f}")

print("\nThe two agree closely: SGD with dropout converges to the solution")
print("of the weighted ridge problem eq. 58.5 predicts, without ridge ever")
print("being written down. The residual difference is the sampling noise")
print("in a finite number of masks.")
print("\nNote the weighting. The penalty is per-feature and proportional to")
print("||x_j||^2, so the four features scaled by 3 are penalised nine times")
print("as hard as the rest. Plain ridge would treat them alike, so dropout")
print("is a DATA-DEPENDENT ridge rather than a plain one.")
print("\nThis is exact for a linear model. For a deep one the expectation")
print("over masks does not factor and the correspondence is suggestive")
print("rather than derived — which is worth remembering when the intuition")
print("'dropout is like L2' is applied to a network.")

# --- section 6.3: early stopping approximates ridge -------------------------
print("\n" + "=" * 72)
print("early stopping approximates ridge with alpha = 1/(eta*t) (eq. 58.11)")
print("=" * 72)


def gd_trajectory(X, y, lr, steps):
    w = np.zeros(X.shape[1])
    out = {}
    for t in range(1, steps + 1):
        w -= lr * (X.T @ (X @ w - y)) / len(X)
        out[t] = w.copy()
    return out


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X / len(X) + alpha * np.eye(X.shape[1]),
                           X.T @ y / len(X))


lr = 0.01
traj = gd_trajectory(Xl, yl, lr, 20000)
print(f"{'steps t':>9} {'predicted alpha':>17} {'|w_gd|':>10} "
      f"{'|w_ridge|':>11} {'cos(w_gd, w_ridge)':>20} {'max |diff|':>12}")
for t in (50, 200, 1000, 5000, 20000):
    a = 1.0 / (lr * t)
    wg = traj[t]
    wr = ridge(Xl, yl, a)
    cos = float(wg @ wr / (np.linalg.norm(wg) * np.linalg.norm(wr)))
    print(f"{t:>9} {a:>17.5f} {np.linalg.norm(wg):>10.4f} "
          f"{np.linalg.norm(wr):>11.4f} {cos:>20.6f} "
          f"{np.abs(wg - wr).max():>12.5f}")

print("\nThe correspondence of eq. 58.11 holds well: stopping at step t")
print("gives a solution close to ridge at alpha = 1/(eta*t), and both norms")
print("grow together as t increases and the implied penalty weakens.")
print("\nSo early stopping is not a separate idea from weight decay — on a")
print("quadratic they are the same regulariser expressed two ways, one as a")
print("penalty and one as a budget of steps. The approximation is exact")
print("only for a quadratic, and the direction of the effect is robust: FEWER")
print("STEPS MEANS MORE REGULARISATION.")

# --- section 6.2: weight decay in a scale-invariant layer -------------------
print("\n" + "=" * 72)
print("weight decay sets an equilibrium norm, not a smaller function (6.2)")
print("=" * 72)
print("A scale-invariant layer: the loss depends only on W/|W|, so the")
print("gradient is orthogonal to W and eq. 58.8 says |W| can only GROW.\n")


def scale_invariant_run(lam, steps=20000, lr=0.05, d=32, seed=2,
                        noise=1.0):
    """A scale-invariant loss with a STOCHASTIC gradient, so the run never
    converges and the norm growth of eq. 58.8 is not confounded with the
    gradient decaying to zero."""
    rs = np.random.default_rng(seed)
    W = rs.normal(0, 1.0, d)
    hist = []
    for t in range(1, steps + 1):
        u = W / np.linalg.norm(W)
        target = rs.normal(0, 1.0, d)             # a fresh target each step
        target /= np.linalg.norm(target)
        g_u = u - target
        g = (g_u - u * (g_u @ u)) / np.linalg.norm(W)   # radial part removed
        W = W - lr * g - lr * lam * W             # eq. 58.6
        if t in (1, 100, 1000, 5000, 20000):
            hist.append((t, float(np.linalg.norm(W))))
    return hist


STEPS = (1, 100, 1000, 5000, 20000)
print(f"{'lambda':>9} " + " ".join(f"{f'|W| @{t}':>11}" for t in STEPS)
      + f" {'1/|W|^2 @20000':>16}")
for lam in (0.0, 0.0003, 0.003, 0.03):
    h_ = scale_invariant_run(lam)
    final = h_[-1][1]
    print(f"{lam:>9.4f} " + " ".join(f"{v:>11.4f}" for _, v in h_)
          + f" {1.0 / final ** 2:>16.5f}")

print("\nAt lambda = 0 the norm only ever GROWS — monotonically at every")
print("checkpoint, and never once down. That is eq. 58.8: the gradient is")
print("orthogonal to W, so each step adds to the norm by Pythagoras and")
print("nothing subtracts. It grows slowly, because the increment is")
print("eta^2|g|^2 and |g| itself falls as 1/|W|, but the direction is")
print("forced and there is no equilibrium without decay.")
print("\nWith decay the norm settles at an equilibrium where the shrinkage")
print("balances that growth, and the equilibrium is lower for larger")
print("lambda. The last column is the effective learning rate multiplier at")
print("that equilibrium, which spans two orders of magnitude across the")
print("table.")
print("\nThat is the mechanism. In a normalised network weight decay is not")
print("shrinking the function — eq. 57.10 says the function does not depend")
print("on |W| at all — it is setting the step size. Which is why lambda")
print("matters there despite the classical argument saying it should do")
print("nothing.")
