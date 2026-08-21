# Extracted from: Chapter 23 — Data Cleaning: Missing Values, Outliers, and Feature Types
# Source: src/.../ch023-cleaning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A cleaning pipeline as a fitted transformation.

Every learned quantity — medians, categories, bounds — is a parameter fitted on
training data and applied unchanged to anything else. That is what makes the
validation score honest and the production behaviour predictable.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

rng = np.random.default_rng(5)


@dataclass
class Cleaner:
    """Fit on train only; transform anything."""

    numeric: list[str]
    categorical: list[str]
    clip_quantiles: tuple[float, float] = (0.01, 0.99)

    medians_: dict = field(default_factory=dict)
    bounds_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        for c in self.numeric:
            s = pd.to_numeric(df[c], errors="coerce")
            self.medians_[c] = float(s.median())
            self.bounds_[c] = tuple(
                float(v) for v in s.quantile(self.clip_quantiles))
        for c in self.categorical:
            norm = df[c].astype("string").str.strip().str.lower()
            self.categories_[c] = sorted(norm.dropna().unique())
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("transform called before fit")
        out = df.copy()
        for c in self.numeric:
            s = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}_missing"] = s.isna().astype("int8")   # keep the signal
            lo, hi = self.bounds_[c]
            out[c] = s.fillna(self.medians_[c]).clip(lo, hi)
        for c in self.categorical:
            norm = out[c].astype("string").str.strip().str.lower()
            known = set(self.categories_[c])
            out[f"{c}_unseen"] = (~norm.isin(known) & norm.notna()).astype("int8")
            out[c] = norm.where(norm.isin(known), "__other__").fillna("__missing__")
        return out


# --- a messy dataset with realistic defects ---------------------------------
def make_data(n, seed, shift=False):
    r = np.random.default_rng(seed)
    income = r.lognormal(10.4 + (0.2 if shift else 0), 0.6, n)
    income[r.random(n) < 0.18] = np.nan                     # missing
    return pd.DataFrame({
        "income": income,
        "age": np.where(r.random(n) < 0.05, 999, r.integers(18, 85, n)),
        "city": r.choice(["London", " london", "LEEDS", "Bristol", None], n,
                         p=[.3, .1, .3, .25, .05]),
        "target": (r.random(n) < 0.3).astype(int),
    })


train = make_data(6000, seed=1)
test = make_data(2000, seed=2, shift=True)      # deliberately shifted

print("BEFORE cleaning:")
print(f"  income nulls   : {train['income'].isna().mean():.1%}")
print(f"  age == 999     : {(train['age'] == 999).sum()} sentinel values")
print(f"  city variants  : {sorted(train['city'].dropna().unique())}")

# The sentinel must be handled before anything statistical touches it.
for d in (train, test):
    d.loc[d["age"] == 999, "age"] = np.nan

cleaner = Cleaner(numeric=["income", "age"], categorical=["city"]).fit(train)

print(f"\nfitted parameters (learned from TRAIN only):")
print(f"  income median  : £{cleaner.medians_['income']:,.0f}")
print(f"  income bounds  : £{cleaner.bounds_['income'][0]:,.0f} – "
      f"£{cleaner.bounds_['income'][1]:,.0f}")
print(f"  known cities   : {cleaner.categories_['city']}")

train_c = cleaner.transform(train)
test_c = cleaner.transform(test)

print(f"\nAFTER cleaning:")
print(f"  nulls remaining     : {int(train_c.isna().sum().sum())}")
print(f"  city variants       : {sorted(train_c['city'].unique())}")
print(f"  indicator columns   : "
      f"{[c for c in train_c.columns if c.endswith(('_missing', '_unseen'))]}")

# --- the test set is transformed with TRAIN parameters ----------------------
print(f"\nthe test set is shifted upward, and is clipped using the TRAIN")
print(f"bounds — which is exactly what will happen in production:")
raw_test_income = pd.to_numeric(test["income"], errors="coerce")
print(f"  test raw max      : £{raw_test_income.max():,.0f}")
print(f"  test after clip   : £{test_c['income'].max():,.0f}  "
      f"(train's 99th percentile)")
print(f"  fraction clipped  : "
      f"{(raw_test_income > cleaner.bounds_['income'][1]).mean():.1%}")

# --- leakage: what fitting on everything would have cost --------------------
print("\n" + "=" * 70)
print("why the fit/transform split matters")
print("=" * 70)
combined = pd.concat([train, test], ignore_index=True)
leaky = Cleaner(numeric=["income", "age"], categorical=["city"]).fit(combined)
print(f"median fitted on train only : £{cleaner.medians_['income']:,.0f}")
print(f"median fitted on everything : £{leaky.medians_['income']:,.0f}")
print(f"difference                  : "
      f"£{leaky.medians_['income'] - cleaner.medians_['income']:,.0f}")
print("\nSmall here, and it is the wrong thing to measure. The leaky version")
print("used information from the test set, so its validation score no longer")
print("estimates production performance — by an amount you cannot know.")
print("The discipline should not depend on how much you got away with.")

# --- transform is deterministic and idempotent ------------------------------
again = cleaner.transform(test)
assert again.equals(test_c)
print(f"\ntransform is deterministic: repeated application is identical.")
