# Extracted from: Chapter 54 — Optimizers: SGD, Momentum, RMSProp, and Adam
# Source: src/.../ch054-optimizers.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Two properties that explain why Adam is the default: scale invariance and
the bounded step. Both measured rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(1)


class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None

    def step(self, p, g, t):
        if self.m is None:
            self.m = np.zeros_like(p)
            self.v = np.zeros_like(p)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


# --- eq. 54.18: scale invariance -------------------------------------------
print("=" * 72)
print("Adam is invariant to the scale of the loss; SGD is not (eq. 54.18)")
print("=" * 72)
A = np.diag(np.logspace(0, 2, 20))
x0 = rng.normal(size=20)


def trajectory(opt_factory, scale, steps=60):
    """Stopped mid-flight: a converged run would hide the difference."""
    x = x0.copy()
    opt = opt_factory()
    for t in range(1, steps + 1):
        g = scale * (A @ x)
        x = opt.step(x, g, t)
    return x


SCALES = (0.001, 1.0, 1000.0)
rows = {}
for scale in SCALES:
    xa = trajectory(lambda: Adam(lr=0.05), scale)
    xs_ = x0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, 61):
            xs_ = xs_ - 0.005 * scale * (A @ xs_)
    rows[scale] = (float(np.linalg.norm(xa)),
                   float(np.linalg.norm(xs_)) if np.all(np.isfinite(xs_))
                   else float("inf"))

ref_a, ref_s = rows[1.0]
print(f"{'loss scale':>12} {'Adam |x| @60':>15} {'vs scale=1':>12}   "
      f"{'SGD |x| @60':>14} {'vs scale=1':>12}")
for scale in SCALES:
    na, ns = rows[scale]
    print(f"{scale:>12g} {na:>15.8f} {na / ref_a:>11.4f}x   "
          f"{ns:>14.6g} {ns / ref_s:>11.4g}x")

print("\nAdam is at the SAME point after sixty steps whatever the loss")
print("scale: the c in eq. 54.18 cancels between numerator and denominator.")
print("SGD is not. At one thousandth the scale it has barely moved, and at")
print("a thousand times it has diverged — all at one learning rate.")
print("\nThis is why Adam needs so much less retuning when the loss changes")
print("form, when a scaling factor is introduced for mixed precision, or")
print("when the architecture changes the gradient magnitudes.")

# --- eq. 54.19: the bounded step -------------------------------------------
print("\n" + "=" * 72)
print("Adam's per-parameter step is bounded; SGD's is not (eq. 54.19)")
print("=" * 72)
print("A gradient of magnitude 1, then a single SPIKE, then back to 1.\n")

opt = Adam(lr=0.01)
p = np.zeros(1)
print(f"{'step':>6} {'gradient':>12} {'|Adam move|':>14} {'|SGD move|':>14}")
lr_sgd = 0.01
for t in range(1, 41):
    g = np.array([1000.0]) if t == 20 else np.array([1.0])
    before = p.copy()
    p = opt.step(p, g, t)
    move_adam = float(abs(p - before).item())
    move_sgd = float(abs(lr_sgd * g).item())
    if t in (1, 5, 19, 20, 21, 25, 40):
        print(f"{t:>6} {g.item():>12.1f} {move_adam:>14.6f} "
              f"{move_sgd:>14.6f}")

print("\nThe spike is a thousand times the usual gradient. SGD moves a")
print("thousand times further, and on a real network that single step")
print("destroys the parameter.")
print("\nAdam's move on the spike step did not grow at all — it HALVED.")
print("That is worth more than the bound of eq. 54.19 promised, and the")
print("reason is the timescale asymmetry again: v jumps by the full")
print("(1-b2)*g^2 immediately, while m is an average over roughly ten")
print("steps and barely notices one outlier. The denominator reacts faster")
print("than the numerator, so a gradient spike makes Adam MORE cautious")
print("rather than less.")
print("\nThe steps then stay small for a long time afterwards — a tenth of")
print("normal by step 40 — because v remembers the spike for about")
print("1/(1-b2) = 1000 steps. That is the cost of the protection, and it")
print("is why gradient clipping is still worth having: it stops the spike")
print("from entering v in the first place.")
print("\nThis combination is the strongest single argument for Adam as a")
print("default. One learning rate caps how far any parameter can move in")
print("one step, so a bad batch cannot wreck the model.")

# --- the price: what Adam costs in memory (table 54.1) ----------------------
print("\n" + "=" * 72)
print("what the state costs (table 54.1)")
print("=" * 72)
print(f"{'model':<14} {'params':>10} {'SGD':>9} {'SGD+mom':>9} "
      f"{'Adam':>9} {'serve bf16':>11} {'train/serve':>12}")
for label, P in (("small MLP", 1e6), ("BERT-base", 1.1e8),
                 ("7B model", 7e9), ("70B model", 7e10)):
    gb = lambda mult: P * mult / 1e9
    print(f"{label:<14} {P:>10.0e} {gb(8):>8.1f}G {gb(12):>8.1f}G "
          f"{gb(16):>8.1f}G {gb(2):>10.1f}G {gb(16) / gb(2):>11.0f}x")
print("\n(4 bytes each for weights, gradients and each moment; serving needs")
print(" only the weights, in bf16. Mixed-precision training comes to the")
print(" same total as the fp32 column, because the bf16 weight and gradient")
print(" copies save exactly what the fp32 master copy costs.)")
print("\nThe 7B row is the one to remember: 112 GB of optimiser-related")
print("memory before a single activation is stored, for a model whose")
print("weights are 28 GB. Serving it needs 14 GB in bf16. That factor of")
print("eight between serving and training is why optimiser-state sharding")
print("exists, and it is arithmetic rather than mystery.")
