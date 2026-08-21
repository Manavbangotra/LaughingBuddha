# Extracted from: Chapter 27 — Feature Engineering and Feature Selection
# Source: src/.../ch027-feature-engineering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A feature pipeline with a single definition used by both paths.

The same function computes features for training and for a single serving
request, which is what eliminates training-serving skew (section 5.5).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)


@dataclass
class FeaturePipeline:
    """One definition, fitted on train, applied identically everywhere."""

    smoothing: float = 20.0
    n_folds: int = 5

    global_mean_: float = 0.0
    city_stats_: pd.DataFrame = field(default_factory=pd.DataFrame)
    fitted_: bool = False

    # ---- the single shared definition of the deterministic features -------
    @staticmethod
    def _base_features(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["tenure_days"] = (as_of - df["signup"]).dt.days
        f["recency_days"] = (as_of - df["last_order"]).dt.days
        f["avg_order_value"] = df["total_spend"] / df["n_orders"]
        f["orders_per_month"] = df["n_orders"] / np.maximum(
            f["tenure_days"] / 30.44, 0.1)
        f["expected_gap"] = f["tenure_days"] / df["n_orders"]
        f["recency_vs_own_gap"] = f["recency_days"] / np.maximum(
            f["expected_gap"], 1.0)
        f["log_spend"] = np.log1p(df["total_spend"])
        return f

    def fit(self, df: pd.DataFrame, y: np.ndarray,
            as_of: pd.Timestamp) -> "FeaturePipeline":
        self.global_mean_ = float(y.mean())
        tmp = pd.DataFrame({"city": df["city"].to_numpy(), "y": y})
        agg = tmp.groupby("city")["y"].agg(["count", "mean"])
        # eq. 27.1, computed on the full training set for use at serving time
        agg["enc"] = ((agg["count"] * agg["mean"]
                       + self.smoothing * self.global_mean_)
                      / (agg["count"] + self.smoothing))
        self.city_stats_ = agg
        self.fitted_ = True
        return self

    def transform_train(self, df, y, as_of) -> pd.DataFrame:
        """Training features: the city encoding is computed OUT OF FOLD."""
        if not self.fitted_:
            raise RuntimeError("fit first")
        f = self._base_features(df, as_of)
        fold = rng.integers(0, self.n_folds, len(df))
        enc = np.empty(len(df))
        for k in range(self.n_folds):
            mask = fold == k
            other = pd.DataFrame({"city": df["city"].to_numpy()[~mask],
                                  "y": y[~mask]})
            a = other.groupby("city")["y"].agg(["count", "mean"])
            gm = other["y"].mean()
            a["enc"] = (a["count"] * a["mean"] + self.smoothing * gm) / \
                       (a["count"] + self.smoothing)
            enc[mask] = pd.Series(df["city"].to_numpy()[mask]).map(
                a["enc"]).fillna(gm).to_numpy()
        f["city_target_enc"] = enc
        return f

    def transform_serve(self, df, as_of) -> pd.DataFrame:
        """Serving features: the encoding uses the FULL training statistics.

        This asymmetry is correct — a serving row contributed nothing to those
        statistics, so there is nothing to leak (section 5.2)."""
        if not self.fitted_:
            raise RuntimeError("fit first")
        f = self._base_features(df, as_of)
        f["city_target_enc"] = (df["city"].map(self.city_stats_["enc"])
                                .fillna(self.global_mean_).to_numpy())
        return f


# --- data ---------------------------------------------------------------------
def make(n, seed):
    r = np.random.default_rng(seed)
    as_of = pd.Timestamp("2026-08-13")
    d = pd.DataFrame({
        "signup": as_of - pd.to_timedelta(r.integers(40, 900, n), "D"),
        "last_order": as_of - pd.to_timedelta(r.integers(0, 250, n), "D"),
        "n_orders": r.poisson(7, n) + 1,
        "total_spend": r.gamma(3, 90, n).round(2),
        "city": r.choice([f"city_{i}" for i in range(60)], n),
    })
    tenure = (as_of - d["signup"]).dt.days
    recency = (as_of - d["last_order"]).dt.days
    p = 1 / (1 + np.exp(-(recency / np.maximum(tenure / d["n_orders"], 1) - 1.2)))
    return d, (r.random(n) < p).astype(int), as_of


train_df, y_train, as_of = make(12_000, 1)
serve_df, y_serve, _ = make(4_000, 2)

pipe = FeaturePipeline().fit(train_df, y_train, as_of)
F_train = pipe.transform_train(train_df, y_train, as_of)
F_serve = pipe.transform_serve(serve_df, as_of)

print("feature columns:", list(F_train.columns))
print(f"\ntrain shape {F_train.shape}, serve shape {F_serve.shape}")
print(f"same columns, same order: "
      f"{list(F_train.columns) == list(F_serve.columns)}")

# --- the point: one definition means no skew --------------------------------
print("\n" + "=" * 72)
print("no training-serving skew: the deterministic features are computed by")
print("the SAME function in both paths")
print("=" * 72)
single_row = serve_df.iloc[[0]]
batch_version = pipe.transform_serve(serve_df, as_of).iloc[[0]]
single_version = pipe.transform_serve(single_row, as_of)
matching = np.allclose(batch_version.to_numpy(), single_version.to_numpy())
print(f"batch and single-row serving produce identical features: {matching}")
assert matching

print(f"\n{'feature':<22} {'train mean':>12} {'serve mean':>12} {'ratio':>8}")
for c in F_train.columns:
    tr, se = F_train[c].mean(), F_serve[c].mean()
    print(f"{c:<22} {tr:>12.3f} {se:>12.3f} {se/tr:>8.3f}")
print("\nRatios near 1 indicate the two paths agree. A ratio far from 1 in")
print("production is the signature of training-serving skew, and monitoring")
print("it is cheap (Chapter 48).")

# --- unseen categories are handled explicitly -------------------------------
novel = serve_df.iloc[[0]].copy()
novel["city"] = "city_never_seen"
out = pipe.transform_serve(novel, as_of)
print(f"\nunseen city -> encoding falls back to the global mean "
      f"{out['city_target_enc'].iloc[0]:.4f} "
      f"(global {pipe.global_mean_:.4f})")
print("An explicit, tested fallback — not a silent NaN that fails downstream.")
