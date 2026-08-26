# -*- coding: utf-8 -*-
# Extracted from: Chapter 21 — What Data Science Actually Is
# Source: src/.../ch021-what-data-science-is.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A provenance and sanity audit, run before any analysis.

Most of what this finds is invisible in a .head() and fatal downstream.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)

# A dataset with several realistic, individually-plausible defects.
n = 30_000
start = pd.to_datetime("2025-06-01")
df = pd.DataFrame({
    "user_id": rng.integers(1, 12_000, n),
    "event_time": start + pd.to_timedelta(rng.integers(0, 300, n), "D"),
    "plan": rng.choice(["free", "pro", "enterprise"], n, p=[.7, .25, .05]),
    "spend": rng.gamma(2, 40, n).round(2),
    "region": rng.choice(["EU", "NA", "APAC"], n, p=[.4, .45, .15]),
})
# Defect 1: a column that was only backfilled from a certain date.
df["nps_score"] = np.where(df["event_time"] < "2025-09-01", np.nan,
                           rng.integers(0, 11, n))
# Defect 2: a field populated only AFTER the outcome — target leakage.
df["churned"] = rng.random(n) < 0.18
df["cancellation_reason"] = np.where(df["churned"],
                                     rng.choice(["price", "bug", "moved"], n),
                                     None)
# Defect 3: a duplicated batch, from an ingestion retry.
df = pd.concat([df, df.iloc[:900]], ignore_index=True)


def audit(df, time_col=None, id_col=None, target=None):
    """Ask the questions of section 4.3 mechanically."""
    print(f"rows {len(df):,}  columns {df.shape[1]}\n")

    print(f"{'column':<22} {'dtype':<12} {'null%':>7} {'nunique':>9} "
          f"{'sample':<20}")
    for c in df.columns:
        s = df[c]
        sample = str(s.dropna().iloc[0])[:18] if s.notna().any() else "—"
        print(f"{c:<22} {str(s.dtype):<12} {s.isna().mean():>6.1%} "
              f"{s.nunique():>9,} {sample:<20}")

    print(f"\nexact duplicate rows: {df.duplicated().sum():,}")
    if id_col:
        dup_ids = df[id_col].duplicated().sum()
        print(f"repeated {id_col}: {dup_ids:,} "
              f"({dup_ids/len(df):.1%}) — is one row per {id_col} expected?")

    # Missingness that varies over time almost always means a schema change.
    if time_col:
        print(f"\nmissingness over time (a jump means the column changed):")
        monthly = df.set_index(time_col).resample("MS").apply(
            lambda g: g.isna().mean())
        for c in df.columns:
            if df[c].isna().any():
                series = monthly[c].dropna()
                if len(series) > 1 and series.max() - series.min() > 0.4:
                    first_full = series[series < 0.5].index.min()
                    print(f"  {c}: {series.min():.0%}–{series.max():.0%} "
                          f"across months; only populated from "
                          f"{first_full.date()}")

    # A column almost perfectly determined by the target is a leak.
    if target:
        print(f"\ncolumns suspiciously related to '{target}':")
        for c in df.columns:
            if c == target:
                continue
            present = df[c].notna()
            if present.nunique() < 2:
                continue
            agreement = max((present == df[target]).mean(),
                            (present != df[target]).mean())
            if agreement > 0.95:
                print(f"  {c}: presence agrees with {target} "
                      f"{agreement:.1%} of the time  <- likely target leakage")


audit(df, time_col="event_time", id_col="user_id", target="churned")

print("\n" + "=" * 68)
print("what the audit found, and what each means")
print("=" * 68)
print("1. 900 duplicate rows — an ingestion retry. Every sum and count is")
print("   inflated by 3% until they are removed.")
print("2. nps_score is entirely missing before 2025-09-01 — the column was")
print("   added mid-stream. Training on the full history teaches the model")
print("   that 'missing NPS' means 'earlier period', which is a date proxy.")
print("3. cancellation_reason is populated exactly when churned is true. It")
print("   is a consequence of the target, not a predictor of it. Including it")
print("   gives a model that appears near-perfect and is useless.")
print("\nNone of these are visible in df.head(), and all three would have")
print("produced a confident, wrong result. Chapters 23 and 28 handle them.")
