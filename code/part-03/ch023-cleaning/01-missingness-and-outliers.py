# -*- coding: utf-8 -*-
# Extracted from: Chapter 23 — Data Cleaning: Missing Values, Outliers, and Feature Types
# Source: src/.../ch023-cleaning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Missingness mechanisms, imputation distortion, and robust outlier detection.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

# --- eq. 23.1 / 23.2: what mean imputation costs ----------------------------
print("=" * 70)
print("mean imputation shrinks variance and attenuates correlation")
print("=" * 70)

n = 200_000
x = rng.normal(50, 10, n)
y = 0.7 * x + rng.normal(0, 7.14, n)          # correlation ~0.7
true_rho = np.corrcoef(x, y)[0, 1]
true_var = x.var()

print(f"complete data: var(x) = {true_var:.2f}, corr = {true_rho:.4f}\n")
print(f"{'missing':>8} {'var after':>11} {'predicted':>11} "
      f"{'corr after':>12} {'predicted':>11}")
for f in (0.1, 0.2, 0.4, 0.6):
    xm = x.copy()
    mask = rng.random(n) < f                   # MCAR
    xm[mask] = np.nan
    filled = np.where(np.isnan(xm), np.nanmean(xm), xm)
    obs_f = mask.mean()
    print(f"{obs_f:>8.1%} {filled.var():>11.2f} {(1-obs_f)*true_var:>11.2f} "
          f"{np.corrcoef(filled, y)[0,1]:>12.4f} "
          f"{true_rho*np.sqrt(1-obs_f):>11.4f}")

print("\nBoth match eqs. 23.1 and 23.2. Every coefficient fitted on imputed")
print("data is biased toward zero by a factor you can compute in advance.")

# --- the three mechanisms behave differently --------------------------------
print("\n" + "=" * 70)
print("MCAR / MAR / MNAR: dropping rows is only safe for one of them")
print("=" * 70)

income = rng.lognormal(10.5, 0.6, n)
age = rng.uniform(18, 80, n)
true_mean = income.mean()

mcar = rng.random(n) < 0.3                                     # unrelated
mar = rng.random(n) < (age - 18) / 62 * 0.6                    # depends on age
mnar = rng.random(n) < 1 / (1 + np.exp(-(income - 60_000) / 20_000))

print(f"true mean income: £{true_mean:,.0f}\n")
print(f"{'mechanism':<8} {'% missing':>10} {'mean after dropping':>21} "
      f"{'bias':>10}")
for name, mask in (("MCAR", mcar), ("MAR", mar), ("MNAR", mnar)):
    kept = income[~mask].mean()
    print(f"{name:<8} {mask.mean():>10.1%} £{kept:>19,.0f} "
          f"{kept - true_mean:>+10,.0f}")

print("\nMCAR: unbiased. MAR: biased here because age correlates with income,")
print("and conditioning on age would fix it. MNAR: badly biased and NOT")
print("fixable from the observed data — the high earners are simply absent.")

# --- masking: outliers hide from the z-score rule ---------------------------
print("\n" + "=" * 70)
print("masking: the z-score rule fails on the outliers it looks for")
print("=" * 70)


def zscore_flags(v, thresh=3.0):
    z = (v - v.mean()) / v.std(ddof=0)
    return np.abs(z) > thresh


def robust_flags(v, thresh=3.5):
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        mad = 1e-9
    z = 0.6745 * (v - med) / mad               # eq. 23.5
    return np.abs(z) > thresh


base = rng.normal(10, 1, 40)
for n_out, label in ((1, "one outlier"), (5, "five outliers")):
    data = np.concatenate([base, np.full(n_out, 1000.0)])
    zf, rf = zscore_flags(data), robust_flags(data)
    z_of_outlier = ((data - data.mean()) / data.std(ddof=0))[-1]
    med, mad = np.median(data), np.median(np.abs(data - np.median(data)))
    r_of_outlier = 0.6745 * (1000 - med) / max(mad, 1e-9)
    print(f"\n{label} of 1000 among 40 values near 10:")
    print(f"  mean {data.mean():>8.1f}, sd {data.std(ddof=0):>8.1f}  "
          f"-> z-score of the outlier {z_of_outlier:>6.2f}")
    print(f"  median {med:>6.1f}, MAD {mad:>7.2f}  "
          f"-> robust z {r_of_outlier:>10.1f}")
    print(f"  z-score rule flags {zf.sum()}/{n_out}, "
          f"robust rule flags {rf.sum()}/{n_out}")

print("\nWith five outliers the z-score rule finds NONE of them: they have")
print("inflated the standard deviation enough to hide themselves. The robust")
print("rule is unaffected because the median and MAD ignore them.")

# --- eq. 23.6: Tukey fences on skewed data ----------------------------------
print("\n" + "=" * 70)
print("Tukey's fences assume symmetry")
print("=" * 70)


def tukey_flags(v):
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return (v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)


normal_data = rng.normal(0, 1, 100_000)
skewed = rng.lognormal(0, 1, 100_000)          # legitimately right-skewed
logged = np.log(skewed)

print(f"{'data':<28} {'% flagged by Tukey':>20}")
print(f"{'normal':<28} {tukey_flags(normal_data).mean():>19.2%}")
print(f"{'lognormal (raw)':<28} {tukey_flags(skewed).mean():>19.2%}")
print(f"{'lognormal (log-transformed)':<28} {tukey_flags(logged).mean():>19.2%}")
print("\nOn skewed data the rule flags 4% of perfectly legitimate values.")
print("Log-transform first, or use quantile clipping instead.")

# --- winsorisation keeps the row and limits its influence -------------------
print("\n" + "=" * 70)
print("winsorising vs dropping")
print("=" * 70)
revenue = np.concatenate([rng.gamma(2, 100, 5000), [250_000, 310_000]])
lo, hi = np.percentile(revenue, [1, 99])
wins = np.clip(revenue, lo, hi)
dropped = revenue[(revenue >= lo) & (revenue <= hi)]

print(f"{'treatment':<16} {'n':>7} {'mean':>10} {'sd':>10} {'total':>12}")
for label, v in (("raw", revenue), ("winsorised", wins), ("dropped", dropped)):
    print(f"{label:<16} {len(v):>7,} {v.mean():>10.1f} {v.std():>10.1f} "
          f"{v.sum():>12,.0f}")
lost_rows = len(revenue) - len(dropped)
lost_value = revenue.sum() - dropped.sum()
print(f"\nDropping removes {lost_rows} rows and £{lost_value:,.0f} of recorded")
print("revenue. Winsorising keeps every customer and caps their leverage,")
print(f"costing £{revenue.sum() - wins.sum():,.0f} of recorded value instead.")
print("Which is right depends entirely on whether those two orders were real.")
