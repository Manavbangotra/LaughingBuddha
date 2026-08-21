# Extracted from: Chapter 24 — Exploratory Data Analysis and Visualization
# Source: src/.../ch024-eda.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A systematic EDA pass, and the structure aggregates hide.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
pd.set_option("display.width", 110)

# --- a dataset with several things worth finding ----------------------------
n = 20_000
device = rng.choice(["desktop", "mobile"], n, p=[0.35, 0.65])
# Two populations with genuinely different behaviour.
session = np.where(device == "desktop",
                   rng.lognormal(5.6, 0.5, n),
                   rng.lognormal(4.2, 0.6, n))
spend = np.where(device == "desktop",
                 rng.gamma(3, 40, n), rng.gamma(2, 12, n))
# A sentinel that survived cleaning, and a capped column.
age = rng.integers(18, 80, n).astype(float)
age[rng.random(n) < 0.03] = 0                       # 0 used as "unknown"
rating = np.clip(rng.normal(3.9, 1.1, n).round(), 1, 5)

df = pd.DataFrame({"device": device, "session_s": session.round(1),
                   "spend": spend.round(2), "age": age, "rating": rating})
df["converted"] = (rng.random(n) <
                   np.where(device == "desktop", 0.08, 0.01)).astype(int)

# --- step 1-2: shape and quality --------------------------------------------
print("=" * 72)
print("1-2. shape and quality")
print("=" * 72)
print(f"rows {len(df):,}  columns {df.shape[1]}  "
      f"memory {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
print(f"\n{'column':<12} {'dtype':<10} {'null%':>7} {'nunique':>9} "
      f"{'constant?':>10}")
for c in df.columns:
    s = df[c]
    print(f"{c:<12} {str(s.dtype):<10} {s.isna().mean():>6.1%} "
          f"{s.nunique():>9,} {str(s.nunique() <= 1):>10}")

# --- step 3: univariate, with the diagnostics of section 6 ------------------
print("\n" + "=" * 72)
print("3. univariate: skew, spikes, granularity")
print("=" * 72)
print(f"{'column':<12} {'mean':>10} {'median':>10} {'mean/med':>9} "
      f"{'skew':>8} {'top value share':>17}")
for c in ["session_s", "spend", "age", "rating"]:
    s = df[c]
    z = (s - s.mean()) / s.std()
    skew = (z ** 3).mean()
    top = s.value_counts(normalize=True).iloc[0]
    top_val = s.value_counts().index[0]
    print(f"{c:<12} {s.mean():>10.2f} {s.median():>10.2f} "
          f"{s.mean()/s.median():>9.2f} {skew:>8.2f} "
          f"{top:>10.1%} at {top_val:g}")

print("\nfindings:")
print("  session_s and spend are right-skewed (mean/median > 1): report")
print("    medians, and expect the mean to be driven by a few large values.")
print(f"  age has {(df['age'] == 0).mean():.1%} of values at exactly 0 —")
print("    impossible for an age, and a sentinel that survived cleaning.")
print("  rating takes only integer values 1-5: it is ordinal, not continuous.")

# eq. 24.4: how far the mean/median ratio goes on lognormal data
print(f"\n{'sigma':>7} {'mean/median (eq. 24.4)':>24} {'simulated':>11}")
for sigma in (0.5, 1.0, 1.5, 2.0):
    sample = rng.lognormal(0, sigma, 400_000)
    print(f"{sigma:>7.1f} {np.exp(sigma**2/2):>24.2f} "
          f"{sample.mean()/np.median(sample):>11.2f}")

# --- step 5-6: bivariate, and the decile view --------------------------------
print("\n" + "=" * 72)
print("5. numeric feature vs binary target: the decile view")
print("=" * 72)
df["spend_decile"] = pd.qcut(df["spend"], 10, labels=False, duplicates="drop")
by_decile = df.groupby("spend_decile").agg(
    n=("converted", "size"), rate=("converted", "mean"),
    spend_lo=("spend", "min"), spend_hi=("spend", "max"))
by_decile["se"] = np.sqrt(by_decile["rate"] * (1 - by_decile["rate"])
                          / by_decile["n"])
print(f"{'decile':>7} {'spend range':>20} {'n':>7} {'conv rate':>11} "
      f"{'95% CI':>18}")
for i, r in by_decile.iterrows():
    lo, hi = r["rate"] - 1.96*r["se"], r["rate"] + 1.96*r["se"]
    print(f"{i:>7} {f'{r.spend_lo:>7.0f}-{r.spend_hi:<7.0f}':>20} "
          f"{int(r.n):>7,} {r['rate']:>10.2%} "
          f"{f'[{lo:.2%}, {hi:.2%}]':>18}")
print("\nThe rate rises with spend — but the intervals overlap heavily between")
print("adjacent deciles, so only the overall trend is supported, not any")
print("individual step (Chapter 10).")

# --- step 8: disaggregate before believing the aggregate --------------------
print("\n" + "=" * 72)
print("8. segments: the aggregate hides the mechanism")
print("=" * 72)
overall = np.corrcoef(df["session_s"], df["converted"])[0, 1]
print(f"overall corr(session_s, converted) = {overall:+.4f}")
print(f"\n{'device':<10} {'n':>8} {'mean session':>14} {'conv rate':>11} "
      f"{'within-group corr':>19}")
for dev, g in df.groupby("device"):
    r = np.corrcoef(g["session_s"], g["converted"])[0, 1]
    print(f"{dev:<10} {len(g):>8,} {g['session_s'].mean():>14.1f} "
          f"{g['converted'].mean():>10.2%} {r:>19.4f}")

print("\nThe overall correlation is driven almost entirely by device: desktop")
print("users have both longer sessions and higher conversion. Within each")
print("device the association is far weaker. Concluding 'longer sessions")
print("cause conversion' from the pooled number would be wrong (Chapter 25).")

# --- eq. 24.7: correlation vs rank correlation vs mutual information --------
print("\n" + "=" * 72)
print("what correlation misses")
print("=" * 72)

m = 20_000
x = rng.uniform(-3, 3, m)
relationships = {
    "linear":        3 * x + rng.normal(0, 1, m),
    "monotonic":     x ** 11 + rng.normal(0, 1, m),   # strictly increasing
    "quadratic (U)": x ** 2 + rng.normal(0, 1, m),
    "independent":   rng.normal(0, 1, m),
}


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1]


def mutual_info(a, b, bins=24):
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


# The diagnostic is the RELATIONSHIP between the three numbers, not any
# absolute threshold: Spearman exceeding Pearson means monotonic-but-curved,
# and mutual information without either means non-monotonic dependence.
print(f"{'relationship':<16} {'Pearson':>9} {'Spearman':>10} "
      f"{'MI':>7} {'S-P gap':>9}  {'verdict'}")
for name, y in relationships.items():
    pe, sp, mi = np.corrcoef(x, y)[0, 1], spearman(x, y), mutual_info(x, y)
    gap = abs(sp) - abs(pe)
    if mi < 0.05:
        verdict = "genuinely independent"
    elif abs(pe) < 0.15 and abs(sp) < 0.15:
        verdict = "dependent but NON-MONOTONIC — both correlations blind"
    elif gap > 0.08:
        verdict = "monotonic but curved — Spearman beats Pearson"
    else:
        verdict = "linear — Pearson is adequate"
    print(f"{name:<16} {pe:>+9.3f} {sp:>+10.3f} {mi:>7.3f} {gap:>+9.3f}  "
          f"{verdict}")

print("\nThe quadratic row is the one that matters: Pearson and Spearman both")
print("report near zero, and mutual information detects it. A correlation")
print("screen would discard this feature entirely (Chapter 27).")
