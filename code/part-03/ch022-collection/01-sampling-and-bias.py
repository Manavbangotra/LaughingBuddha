# -*- coding: utf-8 -*-
# Extracted from: Chapter 22 — Data Collection, Ingestion, and Storage
# Source: src/.../ch022-collection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Selection bias, the design effect, and why more data does not help.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- eq. 22.3: bias does not shrink with n ----------------------------------
print("=" * 70)
print("selection bias is not a small-sample problem")
print("=" * 70)

TRUE_MEAN = 50.0


def biased_sample(n):
    """Users are sampled with probability increasing in their value —
    a convenience sample of the engaged."""
    pop = rng.normal(TRUE_MEAN, 15, n * 6)
    p_include = 1 / (1 + np.exp(-(pop - 50) / 8))      # value-dependent
    keep = rng.random(len(pop)) < p_include
    return pop[keep][:n], pop


print(f"{'n':>10} {'sample mean':>13} {'95% CI half-width':>19} "
      f"{'covers 50?':>12}")
for n in (100, 1_000, 10_000, 100_000):
    sample, _ = biased_sample(n)
    m = sample.mean()
    hw = 1.96 * sample.std(ddof=1) / np.sqrt(len(sample))
    covers = abs(m - TRUE_MEAN) < hw
    print(f"{n:>10,} {m:>13.3f} {hw:>19.3f} {str(covers):>12}")

print("\nThe interval narrows and the estimate does not approach 50. More")
print("data makes the wrong answer more confident (eq. 22.3).")

# --- eq. 22.4: survivorship bias, worked ------------------------------------
print("\n" + "=" * 70)
print("survivorship bias")
print("=" * 70)
n_funds = 20_000
true_return = rng.normal(0.036, 0.12, n_funds)
# Funds with poor returns are more likely to close.
p_survive = 1 / (1 + np.exp(-(true_return - 0.0) / 0.06))
survived = rng.random(n_funds) < p_survive

print(f"all funds        : mean return {true_return.mean():>7.3%}  "
      f"n={n_funds:,}")
print(f"survivors only   : mean return {true_return[survived].mean():>7.3%}  "
      f"n={survived.sum():,}")
print(f"closed funds     : mean return {true_return[~survived].mean():>7.3%}  "
      f"n={(~survived).sum():,}")

# Verify eq. 22.3 numerically.
p_out = (~survived).mean()
predicted_bias = p_out * (true_return[survived].mean()
                          - true_return[~survived].mean())
actual_bias = true_return[survived].mean() - true_return.mean()
print(f"\nbias predicted by eq. 22.3 : {predicted_bias:>7.4f}")
print(f"bias measured              : {actual_bias:>7.4f}")
assert abs(predicted_bias - actual_bias) < 1e-9
print("Analysing only survivors overstates returns by "
      f"{actual_bias*100:.1f} percentage points.")

# --- eq. 22.1: stratification reduces variance ------------------------------
print("\n" + "=" * 70)
print("stratified vs simple random sampling")
print("=" * 70)

# Three strata with very different means — the case where stratifying helps.
strata = {"small": (0.60, 20.0, 5.0), "medium": (0.30, 60.0, 8.0),
          "large": (0.10, 150.0, 20.0)}
population = np.concatenate([
    rng.normal(mu, sd, int(400_000 * w)) for w, mu, sd in strata.values()])
labels = np.concatenate([
    np.full(int(400_000 * w), name) for name, (w, _, _) in strata.items()])
true_mean = population.mean()

n_sample = 600
srs_means, strat_means = [], []
for _ in range(2000):
    idx = rng.choice(len(population), n_sample, replace=False)
    srs_means.append(population[idx].mean())

    total = 0.0
    for name, (w, _, _) in strata.items():
        pool = population[labels == name]
        take = max(2, int(n_sample * w))
        total += w * rng.choice(pool, take, replace=False).mean()
    strat_means.append(total)

print(f"true population mean        : {true_mean:.3f}")
print(f"{'scheme':<22} {'mean of estimates':>18} {'sd of estimates':>17}")
print(f"{'simple random':<22} {np.mean(srs_means):>18.3f} "
      f"{np.std(srs_means):>17.3f}")
print(f"{'stratified':<22} {np.mean(strat_means):>18.3f} "
      f"{np.std(strat_means):>17.3f}")
print(f"\nvariance reduction: {np.var(srs_means)/np.var(strat_means):.1f}x "
      f"for the same sample size (eq. 22.1)")
print("Both are unbiased; stratifying removes the luck of the draw across")
print("strata, which is where most of the variance was.")

# --- eq. 22.2: the design effect --------------------------------------------
print("\n" + "=" * 70)
print("clustered data: 'we have millions of rows' is not millions of samples")
print("=" * 70)

n_users, per_user = 40_000, 50
user_effect = rng.normal(0, 6.0, n_users)          # persistent per-user level
noise = rng.normal(0, 9.2, (n_users, per_user))
events = user_effect[:, None] + noise
flat = events.ravel()

# Intra-cluster correlation: share of variance that is between-user.
var_between = user_effect.var()
var_within = noise.var()
rho = var_between / (var_between + var_within)
deff = 1 + (per_user - 1) * rho

naive_se = flat.std(ddof=1) / np.sqrt(len(flat))
user_means = events.mean(axis=1)
correct_se = user_means.std(ddof=1) / np.sqrt(n_users)

print(f"{len(flat):,} events from {n_users:,} users ({per_user} each)")
print(f"intra-cluster correlation rho : {rho:.3f}")
print(f"design effect (eq. 22.2)      : {deff:.1f}")
print(f"effective sample size         : {len(flat)/deff:,.0f}")
print(f"\nnaive SE (events independent) : {naive_se:.5f}")
print(f"correct SE (user-level)       : {correct_se:.5f}")
print(f"understatement factor         : {correct_se/naive_se:.1f}x "
      f"(sqrt(DEFF) = {np.sqrt(deff):.1f})")
print("\nA 95% interval computed the naive way is about a quarter of its")
print("correct width. The arithmetic is right; the independence assumption")
print("is not.")
