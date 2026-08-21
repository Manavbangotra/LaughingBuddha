# Extracted from: Chapter 24 — Exploratory Data Analysis and Visualization
# Source: src/.../ch024-eda.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An EDA profiler that prioritises its findings.

Generating every statistic is easy and is what automated tools do. Deciding
which of forty observations deserves attention is the part that is still
manual, so this profiler assigns a severity and sorts.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)


def profile(df: pd.DataFrame, target: str | None = None,
            time_col: str | None = None) -> pd.DataFrame:
    """Return findings ordered by severity rather than by column."""
    findings = []

    def add(severity, column, finding, action):
        findings.append({"severity": severity, "column": column,
                         "finding": finding, "suggested action": action})

    n = len(df)
    if df.duplicated().sum():
        add(3, "—", f"{df.duplicated().sum():,} exact duplicate rows",
            "deduplicate; check ingestion idempotency")

    for c in df.columns:
        s = df[c]
        nunique, null_frac = s.nunique(dropna=True), s.isna().mean()

        if nunique <= 1:
            add(2, c, "constant column", "drop — carries no information")
            continue
        if null_frac > 0.5:
            add(2, c, f"{null_frac:.0%} missing",
                "consider dropping; check why it is absent")
        elif null_frac > 0:
            add(1, c, f"{null_frac:.1%} missing", "impute + indicator column")

        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            if len(v) < 10:
                continue
            # Impossible or sentinel-looking values.
            top_share = v.value_counts(normalize=True).iloc[0]
            top_val = v.value_counts().index[0]
            if top_share > 0.02 and top_val in (0, -1, -999, 999, 9999):
                add(3, c, f"{top_share:.1%} of values equal {top_val:g}",
                    "likely a sentinel — verify before treating as numeric")
            if v.median() != 0 and v.mean() / max(v.median(), 1e-9) > 1.5:
                add(1, c, f"right-skewed (mean/median "
                          f"{v.mean()/v.median():.1f})",
                    "report median; consider a log transform")
            if nunique < 15 and pd.api.types.is_integer_dtype(s):
                add(1, c, f"only {nunique} distinct integer values",
                    "probably ordinal or categorical, not continuous")
        else:
            if nunique > 0.5 * n:
                add(2, c, f"cardinality {nunique:,} of {n:,} rows",
                    "identifier-like — exclude, or use target encoding")
            # Categories that differ only by case or whitespace.
            norm = s.astype("string").str.strip().str.lower()
            if norm.nunique() < nunique:
                add(3, c, f"{nunique - norm.nunique()} categories collapse "
                          f"after trim/lowercase",
                    "normalise before encoding — these are the same value")

        if target and c != target and pd.api.types.is_numeric_dtype(s):
            paired = df[[c, target]].dropna()
            if len(paired) > 30 and paired[c].nunique() > 1:
                r = np.corrcoef(paired[c], paired[target])[0, 1]
                if abs(r) > 0.95:
                    add(3, c, f"correlation {r:+.3f} with the target",
                        "suspiciously high — check for leakage (Chapter 28)")

        if time_col and c != time_col and df[c].isna().any():
            monthly = df.set_index(time_col)[c].resample("MS").apply(
                lambda g: g.isna().mean()).dropna()
            if len(monthly) > 1 and monthly.max() - monthly.min() > 0.4:
                add(3, c, "null rate changes sharply over time",
                    "schema change or backfill — check provenance")

    out = pd.DataFrame(findings).sort_values(
        ["severity", "column"], ascending=[False, True])
    return out.reset_index(drop=True)


# --- a dataset with a planted problem of each severity ----------------------
n = 12_000
frame = pd.DataFrame({
    "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 210, n), "D"),
    "user_id": rng.integers(1, 40_000, n),
    "country": rng.choice(["GB", "gb ", "US", "FR"], n, p=[.4, .1, .3, .2]),
    "revenue": rng.lognormal(3.2, 1.1, n).round(2),
    "age": np.where(rng.random(n) < 0.04, -999,
                    rng.integers(18, 80, n)).astype(float),
    "rating": rng.integers(1, 6, n),
    "region": "EMEA",                                     # constant
    "target": rng.integers(0, 2, n),
})
frame["score"] = frame["target"] * 100 + rng.normal(0, 1, n)   # leak
frame["nps"] = np.where(frame["ts"] < "2026-04-01", np.nan,
                        rng.integers(0, 11, n))                # backfilled
frame = pd.concat([frame, frame.iloc[:150]], ignore_index=True)  # duplicates

report = profile(frame, target="target", time_col="ts")

labels = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}
print(f"{'sev':<7} {'column':<10} {'finding':<46} {'action'}")
print("-" * 118)
for _, r in report.iterrows():
    print(f"{labels[r.severity]:<7} {r['column']:<10} {r['finding']:<46} "
          f"{r['suggested action']}")

print(f"\n{len(report)} findings, {(report.severity == 3).sum()} high severity.")
print("\nThe ranking is the contribution. An automated profiler would report")
print("all of these with equal weight, plus fifty more distributions. Knowing")
print("that 'score correlates 0.999 with the target' outranks 'revenue is")
print("skewed' is judgement, and it is what decides whether the next hour is")
print("well spent.")
