# -*- coding: utf-8 -*-
# Extracted from: Chapter 10 — Statistical Inference, Sampling, and Hypothesis Testing
# Source: src/.../ch010-inference.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Statistical inference by simulation — every claim in the chapter checked by
drawing many samples rather than trusting the formula.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- the sampling distribution, made visible --------------------------------
TRUE_ACC = 0.94
print("Drawing 20,000 test sets of 1,000 examples from a model whose TRUE")
print(f"accuracy is exactly {TRUE_ACC}:\n")

for n in (100, 1_000, 10_000):
    samples = rng.binomial(n, TRUE_ACC, size=20_000) / n
    se_predicted = np.sqrt(TRUE_ACC * (1 - TRUE_ACC) / n)
    print(f"  n = {n:>6}: observed accuracies range "
          f"{samples.min():.3f} to {samples.max():.3f}, "
          f"sd {samples.std():.4f} (predicted {se_predicted:.4f})")

print("\nA single measured accuracy is one draw from that spread.")

# --- eq. 10.6: the CLT works regardless of the underlying distribution ------
print(f"\n{'source distribution':<26} {'skew of X':>10} "
      f"{'skew of mean(n=200)':>21}")
sources = {
    "uniform":     lambda k: rng.uniform(0, 1, k),
    "exponential": lambda k: rng.exponential(1.0, k),
    "bernoulli":   lambda k: (rng.random(k) < 0.1).astype(float),
    "bimodal":     lambda k: np.where(rng.random(k) < 0.5,
                                      rng.normal(-3, 0.4, k),
                                      rng.normal(3, 0.4, k)),
}


def skew(v):
    z = (v - v.mean()) / v.std()
    return float((z ** 3).mean())


for name, draw in sources.items():
    raw = draw(200_000)
    means = np.array([draw(200).mean() for _ in range(4000)])
    print(f"{name:<26} {skew(raw):>10.3f} {skew(means):>21.3f}")
print("Whatever the source, the distribution of the MEAN is nearly symmetric.")

# --- eq. 10.9: do confidence intervals actually cover 95% of the time? ------
print("\nCoverage check: build 20,000 intervals and count how many contain")
print("the true value.\n")
for n in (30, 100, 1000):
    covered = 0
    trials = 20_000
    for _ in range(trials):
        s = rng.binomial(n, TRUE_ACC) / n
        se = np.sqrt(max(s * (1 - s), 1e-12) / n)
        if s - 1.96 * se <= TRUE_ACC <= s + 1.96 * se:
            covered += 1
    print(f"  n = {n:>4}: {covered/trials:.1%} of 95% intervals covered the truth")
print("At small n the normal approximation under-covers — the interval is")
print("too narrow, and the nominal 95% is a lie.")

# --- section 6.2: comparing two models --------------------------------------
n_test = 1000
acc_a, acc_b = 0.942, 0.938
se_a = np.sqrt(acc_a * (1 - acc_a) / n_test)
se_b = np.sqrt(acc_b * (1 - acc_b) / n_test)
se_diff = np.sqrt(se_a**2 + se_b**2)
z = (acc_a - acc_b) / se_diff

print(f"\nmodel A: {acc_a:.3f} +/- {1.96*se_a:.4f}")
print(f"model B: {acc_b:.3f} +/- {1.96*se_b:.4f}")
print(f"difference {acc_a-acc_b:.4f}, SE of difference {se_diff:.4f}, "
      f"z = {z:.2f}")
print(f"|z| < 1.96, so the difference is indistinguishable from noise.")


# --- eq. 10.13: how much data would settle it? ------------------------------
def required_n(delta, p_bar, alpha_z=1.96, beta_z=0.84):
    return 2 * (alpha_z + beta_z) ** 2 * p_bar * (1 - p_bar) / delta ** 2


print(f"\n{'effect to detect':>18} {'n per group':>14}")
for delta in (0.004, 0.01, 0.02, 0.05):
    print(f"{delta:>17.1%} {required_n(delta, 0.94):>14,.0f}")
print("Halving the effect quadruples the sample — the delta^2 in eq. 10.13.")

# --- eq. 10.14: multiple comparisons ----------------------------------------
print(f"\n{'tests':>7} {'P(>=1 false positive)':>24} {'simulated':>11}")
for m in (1, 5, 20, 100):
    analytic = 1 - 0.95 ** m
    sims = rng.random((20_000, m)) < 0.05        # each test, under a true null
    simulated = sims.any(axis=1).mean()
    print(f"{m:>7} {analytic:>23.1%} {simulated:>11.1%}")

# The selection effect: the winner of a large search is optimistic.
print("\nSelection bias in hyperparameter search.")
print("50 configurations, ALL with identical true accuracy of 0.90:\n")
for n_val in (200, 2000):
    winners_val, winners_test = [], []
    for _ in range(2000):
        val = rng.binomial(n_val, 0.90, size=50) / n_val
        best = int(np.argmax(val))
        winners_val.append(val[best])
        # The winner re-measured on fresh data of the same size.
        winners_test.append(rng.binomial(n_val, 0.90) / n_val)
    print(f"  validation set n = {n_val}:")
    print(f"    winner's validation score : {np.mean(winners_val):.4f}")
    print(f"    same model on fresh data  : {np.mean(winners_test):.4f}")
    print(f"    optimism                  : "
          f"{np.mean(winners_val) - np.mean(winners_test):+.4f}")
print("\nEvery configuration was equally good. The gap is pure selection")
print("bias — which is exactly why a held-out test set is not optional.")
