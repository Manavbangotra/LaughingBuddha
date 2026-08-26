# Extracted from: Chapter 82 — Scaling Laws: Parameters, Data, and Compute
# Source: src/.../ch082-scaling-laws.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Training compute is paid once; inference is paid forever. Where is the optimum?"""
import numpy as np
from scipy.optimize import minimize_scalar

E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def tokens_for_loss(N, target):
    """Invert eq:chinchilla-form for D — equation (eq:d-of-n)."""
    residual = target - E - A / N ** ALPHA
    if residual <= 0:
        return np.inf              # this N cannot reach the target at any D
    return (B / residual) ** (1 / BETA)


TARGET_LOSS = 2.10

# The hard floor from section 6.3: the smallest N that can reach the target.
n_floor = (A / (TARGET_LOSS - E)) ** (1 / ALPHA)
print(f"target loss {TARGET_LOSS}")
print(f"model-size floor (no D suffices below this): {n_floor / 1e9:.2f}B\n")

print(f"{'serving R (tokens)':>20} {'N*':>10} {'D*':>12} {'D*/N*':>8} "
      f"{'train FLOPs':>13} {'infer FLOPs':>13}")
for R in (0, 1e11, 1e12, 1e13, 1e14, 1e15):
    def total_cost(log_n):
        N = np.exp(log_n)
        D = tokens_for_loss(N, TARGET_LOSS)
        if not np.isfinite(D):
            return 1e30
        return 6 * N * D + 2 * N * R

    res = minimize_scalar(total_cost,
                          bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                          method="bounded")
    N = float(np.exp(res.x))
    D = tokens_for_loss(N, TARGET_LOSS)
    print(f"{R:>20.0e} {N / 1e9:>9.2f}B {D / 1e9:>11.0f}B {D / N:>8.0f} "
          f"{6 * N * D:>13.2e} {2 * N * R:>13.2e}")

print("""
At R = 0 the answer is the compute-optimal one: minimise training cost alone.
As serving volume grows the optimum slides toward a SMALLER model trained on
MORE tokens, because the inference term scales with N and not with D. By
R = 10^14 generated tokens the ratio is far past Chinchilla's 20 — which is the
regime real deployments are in, and the argument LLaMA acted on.

Note the floor: no amount of data reaches the target below a certain model
size, so the optimum approaches that floor from above and stops.""")
