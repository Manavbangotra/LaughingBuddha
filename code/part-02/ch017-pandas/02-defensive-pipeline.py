# -*- coding: utf-8 -*-
# Extracted from: Chapter 17 — Pandas: DataFrames, Joins, and Data Wrangling
# Source: src/.../ch017-pandas.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Cleaning and joining a messy dataset, defensively.

Every step that could silently change the row count or the semantics asserts
what it expects. This is the difference between a pipeline you can trust and a
notebook that produced a number once.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# --- deliberately messy inputs ----------------------------------------------
n = 800
events = pd.DataFrame({
    "user_id": rng.integers(1, 200, n),
    "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 60, n), "D"),
    "amount": np.where(rng.random(n) < 0.06, np.nan, rng.gamma(2, 25, n)),
    "channel": rng.choice(["web", "app", "Web", " app ", None], n,
                          p=[0.4, 0.35, 0.1, 0.1, 0.05]),
})
users = pd.DataFrame({
    "user_id": np.arange(1, 200),
    "segment": rng.choice(["free", "pro", "enterprise"], 199),
})
# A realistic data-quality defect: two duplicated user records.
users = pd.concat([users, users.iloc[[5, 20]]], ignore_index=True)

print(f"events: {len(events)} rows | users: {len(users)} rows")


def check(df, expected_rows=None, name=""):
    """Assert what we believe, loudly."""
    if expected_rows is not None and len(df) != expected_rows:
        raise AssertionError(
            f"{name}: expected {expected_rows} rows, got {len(df)}")
    return df


# --- step 1: normalise the categorical column -------------------------------
before = len(events)
events["channel"] = (events["channel"].str.strip().str.lower())
print(f"\nchannel values after normalising: "
      f"{sorted(events['channel'].dropna().unique())}")
print(f"  'Web' and ' app ' merged into existing categories; "
      f"{events['channel'].isna().sum()} genuinely missing")
check(events, before, "normalise")

# --- step 2: missing amounts — decide deliberately, and record the decision -
missing_amount = events["amount"].isna()
print(f"\n{missing_amount.sum()} rows have a missing amount "
      f"({missing_amount.mean():.1%})")

# Is missingness related to anything? If so, dropping biases the result.
by_channel = events.assign(m=missing_amount).groupby(
    "channel", dropna=False)["m"].mean()
print(f"missing rate by channel:\n{by_channel.round(3).to_dict()}")
print("  Roughly uniform, so dropping is defensible here — but we keep an")
print("  indicator rather than silently discarding the information.")

events["amount_missing"] = missing_amount.astype("int8")
events["amount"] = events["amount"].fillna(events["amount"].median())

# --- step 3: the join, validated --------------------------------------------
print(f"\njoining events to users...")
try:
    events.merge(users, on="user_id", how="left", validate="many_to_one")
except pd.errors.MergeError as exc:
    print(f"  BLOCKED by validate=: {str(exc)[:70]}")
    print("  The users table has duplicate user_id values. Without validate=,")
    print("  this join would have silently inflated the event count.")

dupes = users["user_id"].duplicated().sum()
print(f"  found {dupes} duplicate user rows; de-duplicating")
users_clean = users.drop_duplicates(subset="user_id", keep="first")

joined = events.merge(users_clean, on="user_id", how="left",
                      validate="many_to_one", indicator=True)
check(joined, len(events), "join")
print(f"  joined cleanly: {len(joined)} rows, unchanged")
print(f"  match status: {joined['_merge'].value_counts().to_dict()}")
joined = joined.drop(columns="_merge")

# --- step 4: aggregate --------------------------------------------------------
summary = (joined
           .groupby(["segment", "channel"], dropna=False, observed=True)
           .agg(events=("user_id", "size"),
                users=("user_id", "nunique"),
                revenue=("amount", "sum"),
                mean_amount=("amount", "mean"))
           .round(2)
           .sort_values("revenue", ascending=False))

print(f"\n{summary.head(8)}")

# The aggregate must reconcile with the source — a cheap, powerful check.
assert summary["events"].sum() == len(joined)
assert np.isclose(summary["revenue"].sum(), joined["amount"].sum())
print(f"\nreconciliation: event count and revenue both tie back to the source")

# --- step 5: a group-relative feature, via transform ------------------------
joined["segment_mean"] = joined.groupby("segment")["amount"].transform("mean")
joined["amount_vs_segment"] = joined["amount"] - joined["segment_mean"]
print(f"\ngroup-relative feature (transform keeps the original shape):")
print(joined[["user_id", "segment", "amount", "segment_mean",
              "amount_vs_segment"]].head(4).round(2).to_string(index=False))

# --- step 6: shrink before writing ------------------------------------------
final = joined.astype({"segment": "category", "channel": "category",
                       "amount": "float32", "segment_mean": "float32",
                       "amount_vs_segment": "float32"})
print(f"\nmemory: {joined.memory_usage(deep=True).sum()/1e6:.2f} MB -> "
      f"{final.memory_usage(deep=True).sum()/1e6:.2f} MB")
