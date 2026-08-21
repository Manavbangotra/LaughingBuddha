# Extracted from: Chapter 54 — Optimizers: SGD, Momentum, RMSProp, and Adam
# Source: src/.../ch054-optimizers.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every optimiser in this chapter, implemented from its equations and
compared on a problem whose difficulty we control.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the optimisers ---------------------------------------------------------
class SGD:
    name = "SGD"

    def __init__(self, lr=0.01, momentum=0.0, nesterov=False):
        self.lr, self.mu, self.nesterov = lr, momentum, nesterov
        self.v = None

    def step(self, p, g, t):
        if self.mu == 0.0:
            return p - self.lr * g                       # eq. 54.1
        if self.v is None:
            self.v = np.zeros_like(p)
        self.v = self.mu * self.v + g                    # eq. 54.2
        d = (g + self.mu * self.v) if self.nesterov else self.v
        return p - self.lr * d


class AdaGrad:
    name = "AdaGrad"

    def __init__(self, lr=0.1, eps=1e-8):
        self.lr, self.eps, self.s = lr, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s += g * g                                  # eq. 54.4
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class RMSProp:
    name = "RMSProp"

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, None

    def step(self, p, g, t):
        if self.s is None:
            self.s = np.zeros_like(p)
        self.s = self.rho * self.s + (1 - self.rho) * g * g    # eq. 54.5
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8,
                 weight_decay=0.0, decoupled=True, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decoupled, self.bc = weight_decay, decoupled, \
            bias_correction
        self.m = self.v = None

    @property
    def name(self):
        if self.wd == 0.0:
            return "Adam" if self.bc else "Adam (no bias correction)"
        return "AdamW" if self.decoupled else "Adam + coupled L2"

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        if self.wd and not self.decoupled:
            g = g + self.wd * p                          # eq. 54.11
        self.m = self.b1 * self.m + (1 - self.b1) * g    # eq. 54.6
        self.v = self.b2 * self.v + (1 - self.b2) * g * g   # eq. 54.7
        if self.bc:
            mh = self.m / (1 - self.b1 ** t)             # eq. 54.8
            vh = self.v / (1 - self.b2 ** t)
        else:
            mh, vh = self.m, self.v
        p = p - self.lr * mh / (np.sqrt(vh) + self.eps)  # eq. 54.9
        if self.wd and self.decoupled:
            p = p - self.lr * self.wd * p                # eq. 54.12
        return p


# --- an ill-conditioned quadratic, where the theory is exact ----------------
def quadratic(kappa, dim=50, seed=0):
    """L = 0.5 x' A x with eigenvalues log-spaced over [1, kappa]."""
    rs = np.random.default_rng(seed)
    evals = np.logspace(0, np.log10(kappa), dim)
    Q, _ = np.linalg.qr(rs.normal(size=(dim, dim)))
    A = Q @ np.diag(evals) @ Q.T
    return A, evals


def run_quadratic(A, opt, steps=400, seed=1, noise=0.0):
    rs = np.random.default_rng(seed)
    x = rs.normal(size=len(A))
    x = x / np.linalg.norm(x) * 5.0
    losses = []
    for t in range(1, steps + 1):
        g = A @ x
        if noise:
            g = g + rs.normal(0, noise * np.linalg.norm(g) / np.sqrt(len(g)),
                              len(g))
        losses.append(0.5 * float(x @ A @ x))
        x = opt.step(x, g, t)
        if not np.all(np.isfinite(x)):
            return losses + [np.inf] * (steps - len(losses))
    return losses


print("=" * 72)
print("momentum buys a square root in the condition number (eq. 54.17)")
print("=" * 72)
print("A quadratic with log-spaced eigenvalues. Each method gets ITS OWN")
print("theoretically optimal settings, which is the only fair comparison:")
print("  gradient descent   eta = 2/(alpha+beta)")
print("  heavy ball         eta = 4/(sqrt(alpha)+sqrt(beta))^2,")
print("                      mu = ((sqrt(k)-1)/(sqrt(k)+1))^2")
print("Giving momentum the SAME eta as gradient descent and then dividing")
print("by (1-mu) — a natural-looking choice — exactly cancels eq. 54.16's")
print("amplification and produces no speedup at all.\n")
print(f"{'kappa':>8} {'GD steps':>10} {'momentum steps':>16} "
      f"{'measured speedup':>18} {'predicted sqrt(k)':>19}")
for kappa in (10, 100, 1000, 10000):
    A, evals = quadratic(kappa, seed=0)
    a, b = evals.min(), evals.max()
    reach = {}
    opts = {
        "gd": SGD(lr=2.0 / (a + b)),
        "mom": SGD(lr=4.0 / (np.sqrt(a) + np.sqrt(b)) ** 2,
                   momentum=((np.sqrt(kappa) - 1)
                             / (np.sqrt(kappa) + 1)) ** 2),
    }
    for label, opt in opts.items():
        ls = run_quadratic(A, opt, steps=300000 if kappa > 999 else 30000)
        reach[label] = next((i for i, v in enumerate(ls)
                             if v < 1e-6 * ls[0]), None)
    if reach["gd"] and reach["mom"]:
        print(f"{kappa:>8} {reach['gd']:>10} {reach['mom']:>16} "
              f"{reach['gd'] / reach['mom']:>17.1f}x "
              f"{np.sqrt(kappa):>18.1f}")
    else:
        print(f"{kappa:>8} {str(reach['gd']):>10} {str(reach['mom']):>16}")

print("\nThe measured speedup is consistently about HALF the predicted")
print("sqrt(kappa), and it grows with kappa in the same proportion at every")
print("row. That is the right kind of agreement to expect: eq. 54.17 is an")
print("asymptotic rate, the constant in front of it is not one, and the")
print("spectrum here is a full log-spaced range rather than the two-point")
print("worst case on which the bound is tight.")
print("\nThe scaling is what matters. Momentum buys a factor of two at")
print("kappa = 10 and a factor of nearly fifty at kappa = 10000: it is")
print("worth almost nothing on a well-conditioned problem and enormous on")
print("a badly conditioned one, which is exactly eq. 54.17's claim.")

# --- eq. 54.16: momentum amplifies the step ---------------------------------
print("\n" + "=" * 72)
print("momentum and learning rate are not independent (eq. 54.16)")
print("=" * 72)
print(f"{'mu':>6} {'velocity after 2000 steps':>27} "
      f"{'predicted 1/(1-mu)':>20}")
for mu in (0.0, 0.5, 0.9, 0.99):
    v = 0.0
    for _ in range(2000):
        v = mu * v + 1.0                          # eq. 54.2 with g = 1
    print(f"{mu:>6.2f} {v:>27.4f} {1 / (1 - mu):>20.4f}")
print("\nThe asymptotic velocity is exactly 1/(1-mu) times the gradient.")
print("Raising momentum from 0.9 to 0.99 therefore multiplies the effective")
print("step by TEN at a fixed learning rate — which is why 'I increased")
print("momentum and it diverged' is not a mystery.")

# --- section 5.4: bias correction -------------------------------------------
print("\n" + "=" * 72)
print("what bias correction actually prevents (section 5.4)")
print("=" * 72)
print("A constant unit gradient. Without correction, m warms up in ~10 steps")
print("and v in ~1000, so the RATIO is far too large early.\n")
print(f"{'step':>6} {'m':>10} {'v':>12} {'step w/ correction':>20} "
      f"{'step WITHOUT':>14} {'ratio':>10}")
b1, b2 = 0.9, 0.999
m = v = 0.0
for t in range(1, 3001):
    g = 1.0
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    with_bc = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    without = m / (np.sqrt(v) + 1e-8)
    if t in (1, 2, 5, 20, 100, 500, 1000, 3000):
        print(f"{t:>6} {m:>10.4f} {v:>12.6f} {with_bc:>20.4f} "
              f"{without:>14.4f} {without / with_bc:>9.2f}x")

print("\nRead the last two columns. WITH correction the step is 1.0 from the")
print("very first iteration, which is what eq. 54.9 is supposed to give for")
print("a consistent gradient. WITHOUT it the step is over THREE times too")
print("large at step 1 and stays inflated for hundreds of iterations.")
print("\nThe asymmetry in section 5.4 is the cause: m warms up on a timescale")
print("of 1/(1-b1) = 10 steps and v on 1/(1-b2) = 1000, so the denominator")
print("is suppressed for far longer than the numerator. The uncorrected")
print("update is not a slow start — it is an overshoot, and on a real")
print("network it is a divergence.")
