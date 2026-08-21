# Extracted from: Chapter 17 — Pandas: DataFrames, Joins, and Data Wrangling
# Source: src/.../ch017-pandas.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Pandas' core mechanics and its three sharpest edges, all demonstrated.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 100)
rng = np.random.default_rng(0)

# --- a DataFrame is aligned columns of independent dtypes -------------------
df = pd.DataFrame({
    "user": ["u1", "u2", "u3", "u4", "u5"],
    "age": [25, 40, 35, 50, np.nan],
    "city": ["london", "leeds", "london", "bristol", "leeds"],
    "score": [0.82, np.nan, 0.91, 0.45, 0.77],
})
print(df)
print(f"\ndtypes:\n{df.dtypes}")
print(f"\nage became float64 because it holds a NaN — classic int columns")
print("cannot represent missing values.")
print(f"with the nullable dtype: "
      f"{df['age'].astype('Int64').dtype}  <- capital I, keeps integers")

# --- eq. 17.1: alignment is by LABEL, not position --------------------------
print("\n" + "=" * 66)
print("index alignment")
print("=" * 66)
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["z", "y", "x"])
print(f"a:\n{a.to_dict()}\nb:\n{b.to_dict()}")
print(f"a + b (matched by label):\n{(a + b).to_dict()}")
print("  x:31 = 1+30, not 1+10. Position was ignored.")

c = pd.Series([1, 2, 3], index=["x", "y", "w"])
print(f"\nwith non-overlapping indexes, a + c:\n{(a + c).to_dict()}")
print(f"  union has {len((a + c))} labels, intersection has "
      f"{len(set(a.index) & set(c.index))} non-null (eq. 17.2)")

# --- chained indexing: the assignment that may go nowhere -------------------
print("\n" + "=" * 66)
print("chained indexing")
print("=" * 66)
work = df.copy()
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        work[work["age"] > 30]["score"] = 0.0        # chained: two operations
    except Exception as exc:
        print(f"  raised: {type(exc).__name__}")
changed = work.loc[work["age"] > 30, "score"].tolist()
print(f"after chained assignment, scores are {changed}")
print("  The write went into a temporary. The original is untouched.")

work.loc[work["age"] > 30, "score"] = 0.0            # single .loc: correct
print(f"after .loc assignment,     scores are "
      f"{work.loc[work['age'] > 30, 'score'].tolist()}")

# --- missing data ------------------------------------------------------------
print("\n" + "=" * 66)
print("missing data")
print("=" * 66)
print(f"NaN == NaN            : {np.nan == np.nan}   <- never use == for NaN")
print(f"df['score'].isna()    : {df['score'].isna().tolist()}")
print(f"nulls per column      : {df.isna().sum().to_dict()}")

filled_mean = df["score"].fillna(df["score"].mean())
print(f"\noriginal score std    : {df['score'].std():.4f}")
print(f"after mean imputation : {filled_mean.std():.4f}   <- variance shrank")
print("Mean imputation always understates variance and distorts correlations.")

# The indicator-column approach keeps the information that a value was absent.
augmented = df.assign(score_missing=df["score"].isna().astype(int),
                      score=df["score"].fillna(df["score"].median()))
print(f"\nindicator approach keeps the signal:\n"
      f"{augmented[['user', 'score', 'score_missing']].to_string(index=False)}")

# --- split-apply-combine ------------------------------------------------------
print("\n" + "=" * 66)
print("split-apply-combine: agg reduces, transform broadcasts")
print("=" * 66)
agg = df.groupby("city").agg(n=("user", "size"),
                             mean_score=("score", "mean"),
                             max_age=("age", "max"))
print(f"agg (one row per group):\n{agg}")

df2 = df.assign(city_mean=df.groupby("city")["score"].transform("mean"))
df2["vs_city"] = df2["score"] - df2["city_mean"]
print(f"\ntransform (same shape as input — a group-relative feature):")
print(df2[["user", "city", "score", "city_mean", "vs_city"]].to_string(index=False))

# groupby drops NaN keys unless told otherwise.
with_nan = df.assign(city=df["city"].where(df.index != 0))
print(f"\nrows: {len(with_nan)}, "
      f"grouped (default): {with_nan.groupby('city').size().sum()}, "
      f"dropna=False: {with_nan.groupby('city', dropna=False).size().sum()}")
print("  A row silently vanished. Pass dropna=False when a missing key means")
print("  something.")

# --- eq. 17.3: join cardinality ---------------------------------------------
print("\n" + "=" * 66)
print("join cardinality: the silent row explosion")
print("=" * 66)
users = pd.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]})
orders = pd.DataFrame({"user_id": [1, 1, 2, 2, 2, 3],
                       "amount": [10, 20, 30, 40, 50, 60]})

one_to_many = users.merge(orders, on="user_id", how="inner",
                          validate="one_to_many")
print(f"users {len(users)} x orders {len(orders)} -> "
      f"{len(one_to_many)} rows (one-to-many, as expected)")

# Now duplicate a user — a realistic data-quality problem.
dirty = pd.concat([users, users.iloc[[1]]], ignore_index=True)
bad = dirty.merge(orders, on="user_id", how="inner")
print(f"\nwith user 2 duplicated in the LEFT table:")
print(f"  rows: {len(bad)} (was {len(one_to_many)}) — eq. 17.3 predicts "
      f"{1*2 + 2*3 + 1*1}")
print(f"  total amount: {bad['amount'].sum()} vs true "
      f"{orders['amount'].sum()}   <- inflated")

try:
    dirty.merge(orders, on="user_id", how="inner", validate="one_to_many")
except pd.errors.MergeError as exc:
    print(f"\nvalidate= catches it: MergeError: {str(exc)[:60]}")

# indicator= surfaces unmatched keys instead of silently dropping them.
partial = pd.DataFrame({"user_id": [1, 99], "tier": ["gold", "silver"]})
checked = users.merge(partial, on="user_id", how="outer", indicator=True)
print(f"\nindicator=True shows what matched:\n"
      f"{checked['_merge'].value_counts().to_dict()}")

# --- reshaping ----------------------------------------------------------------
print("\n" + "=" * 66)
print("wide <-> long")
print("=" * 66)
wide = pd.DataFrame({"city": ["london", "leeds"],
                     "2023": [10, 8], "2024": [12, 9]})
long = wide.melt(id_vars="city", var_name="year", value_name="value")
print(f"wide:\n{wide.to_string(index=False)}")
print(f"\nlong (tidy):\n{long.to_string(index=False)}")
back = long.pivot(index="city", columns="year", values="value").reset_index()
back.columns.name = None
print(f"\nafter the round trip:\n{back.to_string(index=False)}")

# pivot SORTS the index, so the rows come back in a different order. The data
# is identical; a naive .equals() comparison would report otherwise.
naive = back.set_index("city").equals(wide.set_index("city"))
sorted_cmp = (back.set_index("city").sort_index()
              .equals(wide.set_index("city").sort_index()))
print(f"naive .equals()      : {naive}   <- misleading: row order differs")
print(f"compared after sorting: {sorted_cmp}   <- the data is preserved")
assert sorted_cmp
print("pivot returns a sorted index. Comparing frames without accounting for")
print("order is a common way to convince yourself a correct transform is broken.")

# --- eq. 17.4: memory, and what category buys -------------------------------
print("\n" + "=" * 66)
print("memory: dtypes are not a detail")
print("=" * 66)
n = 500_000
big = pd.DataFrame({
    "city": rng.choice(["london", "leeds", "bristol", "cardiff",
                        "manchester"], n),
    "count": rng.integers(0, 100, n),
    "ratio": rng.random(n),
})
before = big.memory_usage(deep=True).sum()

tuned = big.astype({"city": "category", "count": "int8", "ratio": "float32"})
after = tuned.memory_usage(deep=True).sum()

print(f"{'column':<10} {'original':>14} {'optimised':>14}")
for col in big.columns:
    print(f"{col:<10} {big[col].memory_usage(deep=True)/1e6:>12.2f} MB "
          f"{tuned[col].memory_usage(deep=True)/1e6:>12.2f} MB")
print(f"{'total':<10} {before/1e6:>12.2f} MB {after/1e6:>12.2f} MB "
      f"  ({before/after:.1f}x smaller)")
print("\nThe city column dominates: object dtype stores a pointer per row to a")
print("boxed Python string. category stores one small integer per row plus")
print("five strings total (eq. 17.4).")

# Values are unchanged; only the representation differs.
assert (tuned["city"].astype(str) == big["city"]).all()
assert (tuned["count"] == big["count"]).all()
print("...and the values are identical.")
