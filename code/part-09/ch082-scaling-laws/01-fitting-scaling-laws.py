# Extracted from: Chapter 82 — Scaling Laws: Parameters, Data, and Compute
# Source: src/.../ch082-scaling-laws.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fit L(N,D) = E + A/N^a + B/D^b and recover the compute-optimal allocation."""
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

rng = np.random.default_rng(0)

# Ground truth we will try to recover from noisy "measurements".
TRUE = dict(E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28)


def loss_true(N, D):
    return (TRUE["E"] + TRUE["A"] / N ** TRUE["alpha"]
            + TRUE["B"] / D ** TRUE["beta"])


# A sweep: a grid of model sizes and token counts, as a real fit would use.
Ns = np.array([3e7, 1e8, 3e8, 1e9, 3e9, 1e10])
Ds = np.array([1e9, 3e9, 1e10, 3e10, 1e11, 3e11])
grid_N, grid_D = np.meshgrid(Ns, Ds, indexing="ij")
obs = loss_true(grid_N, grid_D) * (1 + 0.01 * rng.normal(size=grid_N.shape))

print(f"sweep: {len(Ns)} model sizes x {len(Ds)} token counts "
      f"= {obs.size} runs, 1% observation noise\n")


def model(X, E, A, alpha, B, beta):
    N, D = X
    return E + A / N ** alpha + B / D ** beta


popt, _ = curve_fit(
    model, (grid_N.ravel(), grid_D.ravel()), obs.ravel(),
    p0=[1.0, 100.0, 0.3, 100.0, 0.3],
    bounds=([0, 0, 0.05, 0, 0.05], [10, 1e5, 1.0, 1e5, 1.0]), maxfev=200_000)

names = ["E", "A", "alpha", "B", "beta"]
print(f"{'parameter':>10} {'true':>12} {'recovered':>12} {'error':>10}")
for name, fitted in zip(names, popt):
    truth = TRUE[name]
    print(f"{name:>10} {truth:>12.4f} {fitted:>12.4f} "
          f"{abs(fitted - truth) / truth:>9.1%}")

E_f, A_f, a_f, B_f, b_f = popt

# --- the allocation, from the fitted exponents (eq:n-optimal) ---------------
print(f"\nexponent-implied scaling: N* ~ C^{b_f / (a_f + b_f):.3f}, "
      f"D* ~ C^{a_f / (a_f + b_f):.3f}")
print(f"(Chinchilla reports ~0.5 and ~0.5; Kaplan reported 0.73 and 0.27)")


def optimal_split(C):
    """Minimise the fitted loss along the C = 6ND constraint."""
    def nl(log_n):
        N = np.exp(log_n)
        D = C / (6 * N)
        return E_f + A_f / N ** a_f + B_f / D ** b_f
    r = minimize_scalar(nl, bounds=(np.log(1e6), np.log(1e13)), method="bounded")
    N = float(np.exp(r.x))
    return N, C / (6 * N), float(r.fun)


print(f"\n{'budget C':>12} {'N*':>12} {'D*':>14} {'D*/N*':>8} {'loss':>8}")
for C in (1e19, 1e20, 1e21, 1e22, 1e23, 1e24):
    N, D, L = optimal_split(C)
    print(f"{C:>12.0e} {N / 1e9:>11.2f}B {D / 1e9:>13.0f}B {D / N:>8.1f} "
          f"{L:>8.4f}")

print(f"""
Nothing here assumed a ratio: the sweep was fitted and the allocation fell out
of the exponents. Two things to read off.

First, both scaling exponents are near 1/2 ({b_f / (a_f + b_f):.2f} and """
      f"""{a_f / (a_f + b_f):.2f}), which is Chinchilla's headline — parameters
and tokens grow together, not parameters three times faster.

Second, D*/N* is NOT constant. It drifts upward with budget, because alpha and
beta are close but not equal: D*/N* scales as C^{(a_f - b_f) / (a_f + b_f):.3f}.
The famous "20 tokens per parameter" is the value at the scale Chinchilla
itself was trained at, not a law. Quoting it at a budget three orders of
magnitude away is an extrapolation, and this column is what it extrapolates
to.""")

# --- the fitting trap of section 6.1 ---------------------------------------
# Hold D fixed and vary N, so the exponent being recovered is alpha itself.
D_fixed = 3e11
N_sweep = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
L_sweep = loss_true(N_sweep, D_fixed)

# Wrong: regress log L on log N, ignoring the irreducible floor.
naive_slope, _ = np.polyfit(np.log(N_sweep), np.log(L_sweep), 1)

# Right: subtract the floor first, then the relationship is a true power law.
floor = E_f + B_f / D_fixed ** b_f          # everything not attributable to N
corrected_slope, _ = np.polyfit(np.log(N_sweep), np.log(L_sweep - floor), 1)

print(f"\nrecovering alpha (true value {TRUE['alpha']:.3f}) at fixed D:")
print(f"  naive     log L      vs log N : {-naive_slope:.4f}  "
      f"({-naive_slope / TRUE['alpha'] - 1:+.0%} error)")
print(f"  corrected log(L-floor) vs log N: {-corrected_slope:.4f}  "
      f"({-corrected_slope / TRUE['alpha'] - 1:+.0%} error)")
print("The naive fit is badly biased DOWNWARD, because most of L is a floor "
      "that does not respond to N at all. Subtract the floor first.")
