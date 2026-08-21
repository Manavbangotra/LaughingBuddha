# Extracted from: Chapter 55 — Learning-Rate Schedules and Warmup
# Source: src/.../ch055-lr-schedules.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why warmup is needed with Adam, quantified from eq. 55.11, and the
learning-rate range test.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.4: Adam's variance estimate early in training ----------------
def effective_sample_size(t, b2=0.999):
    """Eq. 55.11: the effective number of samples in the EMA at step t."""
    num = (1 - b2 ** t) ** 2 * (1 + b2)
    den = (1 - b2 ** (2 * t)) * (1 - b2)
    return num / den


print("=" * 72)
print("Adam's second moment is estimated from very few samples early")
print("=" * 72)
print(f"{'step':>8} {'effective n':>14} {'rel. sd of v':>15} "
      f"{'rel. sd of 1/sqrt(v)':>22}")
for t in (1, 2, 5, 10, 50, 100, 500, 1000, 5000):
    n = effective_sample_size(t)
    rel = np.sqrt(2.0 / n)
    print(f"{t:>8} {n:>14.1f} {rel:>14.1%} {rel / 2:>21.1%}")

print("\nAt step 1 the variance estimate comes from a single sample and has")
print("a relative error above 100%. The update divides by its square root,")
print("so half that error passes straight into the step size.")
print("\nBy step 100 the effective sample size is in the tens and the error")
print("is manageable; by step 1000 it has essentially converged to the")
print("asymptotic (1+b2)/(1-b2) = 2000. That is the quantitative case for")
print("warmup, and it also predicts the right LENGTH: a few hundred to a")
print("couple of thousand steps, which is what recipes use.")

# --- the consequence, measured on real Adam updates -------------------------
print("\n" + "=" * 72)
print("what that does to the actual step sizes")
print("=" * 72)


def adam_steps(grad_sd=1.0, steps=2000, lr=1e-3, b1=0.9, b2=0.999,
               warmup=0, seed=0):
    """Track |update| for a single parameter under noisy zero-mean gradients."""
    rs = np.random.default_rng(seed)
    m = v = 0.0
    out = []
    for t in range(1, steps + 1):
        g = rs.normal(0, grad_sd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        eta = lr * min(1.0, t / warmup) if warmup else lr
        out.append(abs(eta * mh / (np.sqrt(vh) + 1e-8)))
    return np.array(out)


print("Pure-noise gradients (mean zero), so every step is wasted motion and")
print("the only question is HOW FAR the parameter wanders.\n")
print(f"{'warmup':>8} {'max |step| in first 100':>25} "
      f"{'total distance, steps 1-100':>29} {'steps 100-2000':>16}")
for warmup in (0, 100, 500, 2000):
    st = adam_steps(warmup=warmup, seed=3)
    print(f"{warmup:>8} {st[:100].max():>25.3e} {st[:100].sum():>29.3e} "
          f"{st[100:].sum():>16.3e}")

print("\nWith no warmup the largest early step is far bigger than anything")
print("that follows, and the parameter travels a long way on gradients that")
print("carry no signal at all. Warmup suppresses exactly that window and")
print("leaves the rest of training untouched.")
print("\nThe reason is eq. 55.11: with a handful of samples, sqrt(v) can be")
print("far below the true gradient scale, and the update divides by it.")

# --- section 6.4 on a real optimisation -------------------------------------
print("\n" + "=" * 72)
print("warmup on a badly conditioned problem")
print("=" * 72)


def quad(kappa=1000, dim=40, seed=0):
    rs = np.random.default_rng(seed)
    evals = np.logspace(0, np.log10(kappa), dim)
    Q, _ = np.linalg.qr(rs.normal(size=(dim, dim)))
    return Q @ np.diag(evals) @ Q.T


A = quad()
x0 = rng.normal(size=40) * 2.0


def run_adam(lr, warmup, steps=1500, noise=2.0, seed=5):
    rs = np.random.default_rng(seed)
    x = x0.copy()
    m = v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    losses = []
    for t in range(1, steps + 1):
        g = A @ x + rs.normal(0, noise, len(x))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        eta = lr * min(1.0, t / warmup) if warmup else lr
        x = x - eta * mh / (np.sqrt(vh) + 1e-8)
        losses.append(0.5 * float(x @ A @ x))
        if not np.isfinite(losses[-1]):
            return losses + [np.inf] * (steps - len(losses))
    return losses


print(f"{'lr':>8} {'warmup':>8} {'loss @100':>13} {'loss @500':>13} "
      f"{'loss @1500':>13}")
for lr in (0.05, 0.3):
    for warmup in (0, 100, 300):
        ls = run_adam(lr, warmup)
        fmt = lambda v: ("diverged" if not np.isfinite(v) else f"{v:.4f}")
        print(f"{lr:>8.2f} {warmup:>8} {fmt(ls[99]):>13} "
              f"{fmt(ls[499]):>13} {fmt(ls[1499]):>13}")

print("\nThe two learning rates give OPPOSITE answers, and that is the")
print("useful result.")
print("\nAt lr = 0.05 warmup only costs. The rate was never dangerous, so")
print("suppressing the early steps threw away progress and the run was")
print("still behind at step 1500.")
print("\nAt lr = 0.30 warmup pays. Without it the run reaches a plateau it")
print("never leaves; with 300 steps of warmup it ends materially lower.")
print("The early steps at that rate did damage the run could not undo.")
print("\nSo warmup is not free and it is not universally good. It is")
print("insurance against the specific failure of taking large steps while")
print("eq. 55.11's variance estimate is unreliable, and its value is")
print("proportional to how close the rate is to the edge. The practical")
print("reading: warm up when you are pushing the learning rate, which for")
print("large models you almost always are.")

# --- the learning-rate range test -------------------------------------------
print("\n" + "=" * 72)
print("the learning-rate range test (section 4.4)")
print("=" * 72)
print("Increase the rate exponentially over 400 steps and watch the loss.\n")


def range_test(lo=1e-5, hi=3.0, steps=400, noise=2.0, seed=7):
    rs = np.random.default_rng(seed)
    x = x0.copy()
    m = v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    out = []
    for t in range(1, steps + 1):
        lr = lo * (hi / lo) ** ((t - 1) / (steps - 1))
        g = A @ x + rs.normal(0, noise, len(x))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (
            np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        loss = 0.5 * float(x @ A @ x)
        out.append((lr, loss if np.isfinite(loss) else np.inf))
        if not np.isfinite(loss) or loss > 1e6 * out[0][1]:
            break
    return out


curve = range_test()
lrs = np.array([c[0] for c in curve])
ls = np.array([c[1] for c in curve])

# Centred moving average. mode="same" zero-pads, which corrupts a half-window
# at each end and produced a spurious "steepest descent" at the last point —
# so smooth with mode="valid" and keep the indices that are actually defined.
W = 9
H = W // 2
smooth = np.convolve(np.nan_to_num(ls, posinf=1e12),
                     np.ones(W) / W, mode="valid")
lrs_v = lrs[H:len(lrs) - H]
dl = np.gradient(np.log(np.clip(smooth, 1e-12, None)), np.log(lrs_v))

print(f"{'lr':>10} {'loss':>14} {'d(log loss)/d(log lr)':>24}")
idx = np.linspace(0, len(lrs_v) - 1, 14).astype(int)
for i in idx:
    print(f"{lrs_v[i]:>10.2e} {smooth[i]:>14.4f} {dl[i]:>24.3f}")

# Steepest descent is by definition BEFORE the minimum, so search only there.
# Searching the whole array picks up spurious dips in the rising tail, where
# the loss is changing by orders of magnitude between adjacent points.
bottom = int(np.argmin(smooth))
steepest = int(np.argmin(dl[:bottom + 1]))
print(f"\nsteepest descent at lr = {lrs_v[steepest]:.3e}")
print(f"minimum loss reached at lr = {lrs_v[bottom]:.3e}")
print(f"ratio between them        = {lrs_v[bottom] / lrs_v[steepest]:.0f}x")
print(f"\nThe standard advice is to take the rate of STEEPEST DESCENT rather")
print(f"than the one at the minimum, because by the time the loss stops")
print(f"falling the rate is already marginal. This test cost "
      f"{len(curve)} steps.")
print("\nBe honest about what it gives you: an order of magnitude, not a")
print("value. It is a way to skip the part of a grid search that is")
print("obviously wrong, and no substitute for a short sweep around the")
print("answer it suggests.")
