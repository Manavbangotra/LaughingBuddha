# -*- coding: utf-8 -*-
# Extracted from: Chapter 21 — What Data Science Actually Is
# Source: src/.../ch021-what-data-science-is.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Framing and feasibility: the arithmetic that should precede a project.

Nothing here is modelling. All of it decides whether modelling is worth doing.
"""
import numpy as np
import pandas as pd

# --- eq. 21.4: is the question answerable with the traffic available? -------
def required_n(delta, p_bar, alpha_z=1.96, beta_z=0.84):
    """Per-variant sample size to detect an absolute difference delta."""
    return 2 * (alpha_z + beta_z) ** 2 * p_bar * (1 - p_bar) / delta ** 2


print("Feasibility before any modelling: how long must an experiment run?\n")
weekly_traffic = 2000
print(f"{'baseline':>9} {'target lift':>12} {'n per arm':>11} "
      f"{'weeks at 2k/wk':>16}")
for p_bar in (0.05, 0.20):
    for delta in (0.005, 0.01, 0.02, 0.05):
        n = required_n(delta, p_bar)
        weeks = 2 * n / weekly_traffic          # two arms
        flag = "  <- infeasible" if weeks > 12 else ""
        print(f"{p_bar:>9.1%} {delta:>12.1%} {n:>11,.0f} "
              f"{weeks:>16.1f}{flag}")

print("\nDetecting a 0.5pp lift on a 5% baseline needs ~30k per arm — thirty")
print("weeks of traffic here. That answer costs two minutes and can stop an")
print("infeasible project before anyone writes a model.")

# --- the effort distribution, measured on a real-ish pipeline ---------------
print("\n" + "=" * 68)
print("where the effort actually goes")
print("=" * 68)

rng = np.random.default_rng(0)
n = 40_000

# A deliberately messy source, of the kind that actually arrives.
raw = pd.DataFrame({
    "user_id": rng.integers(1, 9000, n),
    "signup": pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 500, n), "D"),
    "channel": rng.choice(["web", "Web", "app", " app", None], n,
                          p=[0.35, 0.1, 0.3, 0.1, 0.15]),
    "revenue": np.where(rng.random(n) < 0.12, np.nan,
                        rng.gamma(2, 30, n)),
    "country": rng.choice(["GB", "gb", "US", "FR", ""], n,
                          p=[0.3, 0.1, 0.3, 0.2, 0.1]),
})

issues = {
    "rows": len(raw),
    "duplicate user_ids": int(raw["user_id"].duplicated().sum()),
    "missing revenue": int(raw["revenue"].isna().sum()),
    "missing channel": int(raw["channel"].isna().sum()),
    "channel variants": raw["channel"].dropna().nunique(),
    "country variants": raw["country"].nunique(),
    "empty-string countries": int((raw["country"] == "").sum()),
}
print("what an unexamined dataset contains:")
for k, v in issues.items():
    print(f"  {k:<26} {v:>8,}")

cleaned = raw.assign(
    channel=raw["channel"].str.strip().str.lower(),
    country=raw["country"].str.upper().replace("", np.nan),
)
print(f"\nafter two lines of normalisation:")
print(f"  channel variants           {cleaned['channel'].dropna().nunique():>8}"
      f"   (was {issues['channel variants']})")
print(f"  country variants           {cleaned['country'].dropna().nunique():>8}"
      f"   (was {issues['country variants']})")
print("Those variants would have become separate categories in a one-hot")
print("encoding, splitting the same signal across duplicate columns.")

# --- eq. 21.2/21.3: simulating the two kinds of drift -----------------------
print("\n" + "=" * 68)
print("data drift vs concept drift")
print("=" * 68)

def fit_threshold(x, y):
    """A trivial model: the threshold on x that best separates the classes."""
    candidates = np.quantile(x, np.linspace(0.05, 0.95, 60))
    accs = [((x > t) == y).mean() for t in candidates]
    return candidates[int(np.argmax(accs))]


# Training world: y = 1 when x is large.
x_train = rng.normal(50, 10, 20_000)
y_train = (x_train + rng.normal(0, 3, 20_000)) > 55
thr = fit_threshold(x_train, y_train)
base_acc = ((x_train > thr) == y_train).mean()
print(f"trained threshold {thr:.2f}, training accuracy {base_acc:.3f}")

# Covariate shift: inputs move, the RELATIONSHIP is unchanged.
x_cov = rng.normal(62, 10, 20_000)                       # p(x) moved
y_cov = (x_cov + rng.normal(0, 3, 20_000)) > 55          # p(y|x) same
acc_cov = ((x_cov > thr) == y_cov).mean()

# Concept drift: inputs look identical, the relationship inverted.
x_con = rng.normal(50, 10, 20_000)                       # p(x) unchanged
y_con = (x_con + rng.normal(0, 3, 20_000)) < 45          # p(y|x) CHANGED
acc_con = ((x_con > thr) == y_con).mean()

print(f"\n{'scenario':<22} {'mean(x)':>9} {'accuracy':>10} "
      f"{'detectable from x alone?':>26}")
print(f"{'training':<22} {x_train.mean():>9.2f} {base_acc:>10.3f} "
      f"{'—':>26}")
print(f"{'covariate shift':<22} {x_cov.mean():>9.2f} {acc_cov:>10.3f} "
      f"{'yes: the mean moved':>26}")
print(f"{'concept drift':<22} {x_con.mean():>9.2f} {acc_con:>10.3f} "
      f"{'NO: inputs look normal':>26}")

print("\nMonitoring the input distribution catches the first and is blind to")
print("the second — which is the more damaging of the two, and only shows up")
print("once labels arrive (Chapter 48).")

# --- the cost of not identifying the decision -------------------------------
print("\n" + "=" * 68)
print("framing: does the answer change a decision?")
print("=" * 68)
questions = [
    ("Which segment has the highest churn?",
     "yes — determines where retention spend goes"),
    ("What is our average session length?",
     "no decision attached — a number for a slide"),
    ("Will this user churn in 30 days?",
     "yes — triggers an intervention, if one exists"),
    ("How accurate could a churn model be?",
     "only if the accuracy threshold for acting is agreed first"),
]
for q, verdict in questions:
    print(f"  {q:<42} {verdict}")
print("\nAn analysis with no decision attached has the same value as a wrong")
print("one, and costs the same to produce.")
