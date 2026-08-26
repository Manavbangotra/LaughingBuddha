# Extracted from: Chapter 82 — Scaling Laws: Parameters, Data, and Compute
# Source: src/.../ch082-scaling-laws.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Turning a cluster booking and a traffic forecast into N and D."""
import numpy as np
from scipy.optimize import minimize_scalar

DEVICES, DEVICE_FLOPS, MFU = 512, 1e15, 0.45
WEEKS = 6
REQUESTS_PER_MONTH = 50e6
TOKENS_PER_REQUEST = 600           # prompt + completion
MONTHS_DEPLOYED = 24
UNIQUE_TOKENS = 1.5e12             # from the corpus audit, ch:fm-datasets

E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28

C_train = DEVICES * DEVICE_FLOPS * MFU * WEEKS * 7 * 86_400
R = REQUESTS_PER_MONTH * TOKENS_PER_REQUEST * MONTHS_DEPLOYED

print(f"training budget : {C_train:.3e} FLOPs ({WEEKS} weeks x {DEVICES} devices)")
print(f"serving forecast: {R:.3e} tokens over {MONTHS_DEPLOYED} months")
print(f"unique tokens   : {UNIQUE_TOKENS:.2e} available\n")


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def solve(C, cap_D=None):
    """Best (N, D) on the C = 6ND constraint, optionally capping D."""
    def objective(log_n):
        N = np.exp(log_n)
        D = C / (6 * N)
        if cap_D is not None and D > cap_D:
            return 1e30                      # infeasible: not enough tokens
        return loss(N, D)
    lo = np.log(C / (6 * cap_D)) if cap_D else np.log(1e7)
    r = minimize_scalar(objective, bounds=(lo, np.log(1e12)), method="bounded")
    N = float(np.exp(r.x))
    return N, C / (6 * N)


# --- allocation 1: minimise training loss alone (compute-optimal) -----------
n_co, d_co = solve(C_train)

# --- allocation 2: minimise LIFETIME FLOPs at the loss the first achieves ---
# This is eq:lifetime-cost-2 with a target loss, and it is the honest
# comparison: same quality, different total cost.
target = loss(n_co, d_co)
n_floor = (A / (target - E)) ** (1 / ALPHA)


def tokens_for_loss(N, L0):
    residual = L0 - E - A / N ** ALPHA
    return np.inf if residual <= 0 else (B / residual) ** (1 / BETA)


def lifetime(log_n):
    N = np.exp(log_n)
    D = tokens_for_loss(N, target)
    return 1e30 if not np.isfinite(D) else 6 * N * D + 2 * N * R


r = minimize_scalar(lifetime, bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                    method="bounded")
n_ia = float(np.exp(r.x))
d_ia = tokens_for_loss(n_ia, target)

print(f"both allocations reach loss {target:.4f}; model-size floor for that "
      f"loss is {n_floor / 1e9:.2f}B\n")
print(f"{'allocation':<20} {'N':>9} {'D':>11} {'D/N':>7} "
      f"{'train FLOPs':>12} {'infer FLOPs':>12} {'lifetime':>12}")
for label, N, D in [("compute-optimal", n_co, d_co),
                    ("inference-aware", n_ia, d_ia)]:
    tr, inf = 6 * N * D, 2 * N * R
    print(f"{label:<20} {N / 1e9:>8.2f}B {D / 1e9:>10.0f}B {D / N:>7.0f} "
          f"{tr:>12.2e} {inf:>12.2e} {tr + inf:>12.2e}")

saving = (6 * n_co * d_co + 2 * n_co * R) - (6 * n_ia * d_ia + 2 * n_ia * R)
base = 6 * n_co * d_co + 2 * n_co * R
print(f"\nlifetime saving from the inference-aware choice: {saving:.2e} FLOPs "
      f"({saving / base:.1%})")
print(f"inference is {2 * n_co * R / base:.1%} of lifetime cost at this traffic "
      f"— which is why the correction is small HERE.")

# When does the correction actually matter? Sweep the traffic forecast.
print(f"\n{'requests/month':>16} {'infer share':>12} {'N*':>9} {'D*/N*':>7} "
      f"{'saving':>8}")
for rpm in (5e6, 5e7, 5e8, 5e9, 5e10):
    R_i = rpm * TOKENS_PER_REQUEST * MONTHS_DEPLOYED

    def lifetime_i(log_n):
        N = np.exp(log_n)
        D = tokens_for_loss(N, target)
        return 1e30 if not np.isfinite(D) else 6 * N * D + 2 * N * R_i

    ri = minimize_scalar(lifetime_i, bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                         method="bounded")
    N_i = float(np.exp(ri.x))
    D_i = tokens_for_loss(N_i, target)
    base_i = 6 * n_co * d_co + 2 * n_co * R_i
    save_i = base_i - (6 * N_i * D_i + 2 * N_i * R_i)
    print(f"{rpm:>16.0e} {2 * n_co * R_i / base_i:>11.1%} {N_i / 1e9:>8.1f}B "
          f"{D_i / N_i:>7.0f} {save_i / base_i:>7.1%}")

print("\nThe correction is worth having only once inference is a material "
      "share of lifetime cost. Below that it is a rounding error, and above it "
      "it dominates — which is why LLaMA's regime and a research run's regime "
      "give genuinely different answers.")

# --- correction 2 from fig:scaling-decisions: is there enough unique data? --
print()
for label, D in [("compute-optimal", d_co), ("inference-aware", d_ia)]:
    epochs = D / UNIQUE_TOKENS
    verdict = ("fits in unique data" if epochs <= 1
               else f"needs {epochs:.1f} epochs of repetition")
    print(f"{label:<20} D = {D / 1e12:>6.2f}T -> {verdict}")

n_cap, d_cap = solve(C_train, cap_D=UNIQUE_TOKENS)
print(f"\nre-solved with D capped at the unique-token supply:")
print(f"  N = {n_cap / 1e9:.2f}B, D = {d_cap / 1e12:.2f}T, "
      f"loss = {loss(n_cap, d_cap):.4f} "
      f"(vs {target:.4f} uncapped — a penalty of "
      f"{loss(n_cap, d_cap) - target:+.4f})")

print("""
Three inputs decided this and only one is a fact about machine learning: the
compute budget, the traffic forecast, and the size of the deduplicated corpus.

Note which one actually bound here. At this traffic level the inference-aware
correction moved N by under 10%, while the data cap forced a 2.4x change in
model size and cost real loss. The chapter's headline correction was the least
important of the three for THIS deployment — and the only way to know that was
to compute all three.""")
