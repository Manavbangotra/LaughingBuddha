# Extracted from: Chapter 22 — Data Collection, Ingestion, and Storage
# Source: src/.../ch022-collection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A data contract enforced at ingestion, catching drift before it spreads.

Each violation is a change that produces plausible wrong numbers rather than
an error, which is why it must be checked explicitly.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)


class ContractViolation(Exception):
    pass


CONTRACT = {
    "fields": {
        "user_id":   {"dtype": "int64",    "nullable": False, "unique": True},
        "signup_at": {"dtype": "datetime", "nullable": False},
        "plan":      {"dtype": "object",   "allowed": {"free", "pro",
                                                       "enterprise"}},
        "spend_gbp": {"dtype": "float",    "min": 0.0, "max": 50_000.0,
                      "max_null_frac": 0.20},
    },
    "rows": {"min": 800, "max": 5_000},
}


def validate(df: pd.DataFrame, contract: dict, name: str = "batch") -> None:
    problems = []

    n = len(df)
    lo, hi = contract["rows"]["min"], contract["rows"]["max"]
    if not lo <= n <= hi:
        problems.append(f"row count {n:,} outside expected [{lo:,}, {hi:,}]")

    for field, rules in contract["fields"].items():
        if field not in df.columns:
            problems.append(f"{field}: missing entirely")
            continue
        s = df[field]

        if rules["dtype"] == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
            problems.append(f"{field}: expected datetime, got {s.dtype}")
        elif rules["dtype"] == "int64" and not pd.api.types.is_integer_dtype(s):
            problems.append(f"{field}: expected integer, got {s.dtype}")
        elif rules["dtype"] == "float" and not pd.api.types.is_float_dtype(s):
            problems.append(f"{field}: expected float, got {s.dtype}")

        if not rules.get("nullable", True) and s.isna().any():
            problems.append(f"{field}: {s.isna().sum():,} nulls, none allowed")
        if "max_null_frac" in rules and s.isna().mean() > rules["max_null_frac"]:
            problems.append(f"{field}: null fraction {s.isna().mean():.1%} "
                            f"exceeds {rules['max_null_frac']:.0%}")
        if rules.get("unique") and s.duplicated().any():
            problems.append(f"{field}: {s.duplicated().sum():,} duplicates, "
                            f"must be unique")
        if "allowed" in rules:
            unexpected = set(s.dropna().unique()) - rules["allowed"]
            if unexpected:
                problems.append(f"{field}: unexpected values {sorted(unexpected)}")
        if "min" in rules and s.dropna().lt(rules["min"]).any():
            problems.append(f"{field}: values below {rules['min']}")
        if "max" in rules and s.dropna().gt(rules["max"]).any():
            worst = s.max()
            problems.append(f"{field}: max {worst:,.0f} exceeds {rules['max']:,.0f}")

    if problems:
        raise ContractViolation(
            f"{name}: {len(problems)} violation(s)\n  - "
            + "\n  - ".join(problems))
    print(f"{name}: passed ({n:,} rows)")


def make_batch(n=2000, **defects):
    df = pd.DataFrame({
        "user_id": np.arange(1, n + 1),
        "signup_at": pd.to_datetime("2026-01-01")
                     + pd.to_timedelta(rng.integers(0, 200, n), "D"),
        "plan": rng.choice(["free", "pro", "enterprise"], n, p=[.7, .25, .05]),
        "spend_gbp": np.where(rng.random(n) < 0.08, np.nan,
                              rng.gamma(2, 60, n).round(2)),
    })
    if defects.get("units_changed"):        # pounds became pence
        df["spend_gbp"] *= 100
    if defects.get("new_category"):
        df.loc[df.index[:40], "plan"] = "team"
    if defects.get("duplicated_batch"):
        df = pd.concat([df, df.iloc[:300]], ignore_index=True)
    if defects.get("more_nulls"):
        mask = rng.random(len(df)) < 0.35
        df.loc[mask, "spend_gbp"] = np.nan
    return df


print("a clean batch:")
validate(make_batch(), CONTRACT, "2026-08-13")

print("\nnow the four drifts from table 22.1 that do NOT raise an error on")
print("their own — each would silently corrupt downstream numbers:\n")

for label, defect in [
    ("units changed (GBP -> pence)", {"units_changed": True}),
    ("new category appeared", {"new_category": True}),
    ("batch loaded twice", {"duplicated_batch": True}),
    ("upstream started dropping values", {"more_nulls": True}),
]:
    try:
        validate(make_batch(**defect), CONTRACT, label)
    except ContractViolation as exc:
        first = str(exc).splitlines()[1].strip()
        print(f"  {label:<34} CAUGHT: {first[2:]}")

print("\nWithout the contract:")
bad = make_batch(units_changed=True)
clean = make_batch()
print(f"  mean spend, clean batch  : £{clean['spend_gbp'].mean():>12,.2f}")
print(f"  mean spend, units changed: £{bad['spend_gbp'].mean():>12,.2f}")
print("  Both are plausible-looking numbers. Only one is in pounds.")

# --- idempotent loading ------------------------------------------------------
print("\n" + "=" * 70)
print("idempotent loading: the job WILL run twice")
print("=" * 70)

warehouse = pd.DataFrame(columns=["user_id", "day", "spend"])
batch = pd.DataFrame({"user_id": [1, 2, 3], "day": "2026-08-13",
                      "spend": [10.0, 20.0, 30.0]})


def naive_append(wh, b):
    return pd.concat([wh, b], ignore_index=True)


def partition_overwrite(wh, b, day):
    kept = wh[wh["day"] != day]
    return pd.concat([kept, b], ignore_index=True)


naive = warehouse.copy()
idem = warehouse.copy()
for attempt in range(1, 4):
    naive = naive_append(naive, batch)
    idem = partition_overwrite(idem, batch, "2026-08-13")
    print(f"  after run {attempt}: naive total £{naive['spend'].sum():>6.0f}, "
          f"idempotent total £{idem['spend'].sum():>6.0f}")

print("\nThe naive load triples revenue after two retries. The partition")
print("overwrite is unchanged — which is what makes a rerun safe.")
