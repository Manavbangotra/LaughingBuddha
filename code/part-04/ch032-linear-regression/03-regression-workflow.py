# Extracted from: Chapter 32 — Linear Regression from First Principles
# Source: src/.../ch032-linear-regression.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A complete regression workflow: fit, diagnose, regularise, interpret.

Synthetic housing data with a deliberately planted nonlinearity, planted
heteroscedasticity, and a pair of collinear features — the three things
residual plots are supposed to catch.
"""
import numpy as np

rng = np.random.default_rng(7)
n = 900

area = rng.gamma(9.0, 12.0, n)                       # sq metres
bedrooms = np.clip(np.round(area / 45 + rng.normal(0, 0.6, n)), 1, 6)
area_sqft = area * 10.7639                           # collinear by construction
age = rng.uniform(0, 90, n)
dist_centre = rng.gamma(2.0, 3.0, n)                 # km

# the truth: LOG price is linear; price itself is not, and noise scales with it
log_price = (11.0
             + 0.0060 * area
             + 0.030 * bedrooms
             - 0.0035 * age
             - 0.045 * dist_centre
             + rng.normal(0, 0.18, n))
price = np.exp(log_price)

names = ["area", "bedrooms", "area_sqft", "age", "dist_centre"]
X = np.column_stack([area, bedrooms, area_sqft, age, dist_centre])
cut = 700
Xtr, Xte, ytr, yte = X[:cut], X[cut:], price[:cut], price[cut:]


def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def r2(pred, y):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


# --- step 1: fit naively and look at the diagnostics ------------------------
b = fit(Xtr, ytr)
fitted = predict(b, Xtr)
resid = ytr - fitted
print("=" * 72)
print("step 1 — fit on the raw target")
print("=" * 72)
print(f"train R2 {r2(fitted, ytr):.4f}   test R2 {r2(predict(b, Xte), yte):.4f}")

# nonlinearity: are residuals systematically signed across the fitted range?
# heteroscedasticity: does residual spread grow with the fitted value?
# Both are read off the same binning of the residuals by fitted value.
order = np.argsort(fitted)
bins = np.array_split(resid[order], 6)
print("\nresiduals binned by fitted value — the only plot that matters:")
print(f"{'sextile':>8} {'mean resid':>14} {'std resid':>14}")
for i, t in enumerate(bins):
    print(f"{i + 1:>8} {t.mean():>14,.0f} {t.std():>14,.0f}")

signs = "".join("+" if t.mean() > 0 else "-" for t in bins)
print(f"\nsign pattern of the mean residual: {signs}")
print("Runs of one sign, not an alternating jumble. The model under-predicts")
print("at both ends of its own range and over-predicts in the middle — the")
print("signature of fitting a straight line to a convex relationship. A")
print("correctly specified model cannot be systematically wrong in one")
print("region of its own predictions, so this is a specification error.")

spread_ratio = bins[-1].std() / bins[0].std()
print(f"\nresidual spread, highest bin / lowest bin = {spread_ratio:.2f}   "
      f"({'heteroscedastic' if spread_ratio > 2 else 'roughly constant'})")
print("Constant-variance errors were assumed by Gauss-Markov (section 5.2).")
print("Violating it does not bias the coefficients — it invalidates every")
print("standard error and confidence interval computed from this fit.")

# --- step 2: collinearity ---------------------------------------------------
print("\n" + "=" * 72)
print("step 2 — variance inflation factors (eq. 32.12)")
print("=" * 72)


def vif(Xf, names):
    for j in range(Xf.shape[1]):
        others = np.column_stack([np.ones(len(Xf)), np.delete(Xf, j, axis=1)])
        pred = others @ np.linalg.lstsq(others, Xf[:, j], rcond=None)[0]
        ss_res = np.sum((Xf[:, j] - pred) ** 2)
        ss_tot = np.sum((Xf[:, j] - Xf[:, j].mean()) ** 2)
        rsq = 1 - ss_res / ss_tot
        v = np.inf if rsq > 1 - 1e-12 else 1 / (1 - rsq)
        flag = "  <-- collinear" if v > 10 else ""
        print(f"  {names[j]:<14} R2 {rsq:>8.5f}   VIF {v:>12,.1f}{flag}")


vif(Xtr, names)
print("\narea and area_sqft are the same measurement in different units.")
print("Coefficients on both are meaningless; predictions are unaffected.")

keep = [0, 1, 3, 4]
b2 = fit(Xtr[:, keep], ytr)
print(f"\nafter dropping area_sqft: test R2 "
      f"{r2(predict(b2, Xte[:, keep]), yte):.4f} "
      f"(was {r2(predict(b, Xte), yte):.4f}) — essentially unchanged")
print(f"coefficient on area: {b[1]:>12,.1f} with the duplicate present")
print(f"                     {b2[1]:>12,.1f} with it removed")

# --- step 3: fix the functional form ----------------------------------------
print("\n" + "=" * 72)
print("step 3 — model log(price) instead, as the diagnostics suggested")
print("=" * 72)
b3 = fit(Xtr[:, keep], np.log(ytr))
log_fit = predict(b3, Xtr[:, keep])
log_res = np.log(ytr) - log_fit
lo = np.array_split(log_res[np.argsort(log_fit)], 6)
print("mean residual by sextile:", " ".join(f"{t.mean():>8.4f}" for t in lo))
print("std  residual by sextile:", " ".join(f"{t.std():>8.4f}" for t in lo))
print("sign pattern:            ",
      "".join("+" if t.mean() > 0 else "-" for t in lo))
print(f"spread ratio high/low = {lo[-1].std() / lo[0].std():.2f}")

pred_price = np.exp(predict(b3, Xte[:, keep]))
print(f"\ntest R2 on the price scale: {r2(pred_price, yte):.4f} "
      f"(linear model gave {r2(predict(b2, Xte[:, keep]), yte):.4f})")
print("Both diagnostics are now flat, and accuracy improved. The residual")
print("plot told us the functional form was wrong before any tuning.")

# --- step 4: interpret ------------------------------------------------------
print("\n" + "=" * 72)
print("step 4 — interpreting the coefficients")
print("=" * 72)
kept_names = [names[j] for j in keep]
truth = {"area": 0.0060, "bedrooms": 0.030, "age": -0.0035,
         "dist_centre": -0.045}
print(f"{'feature':<14} {'coef':>10} {'true':>10} "
      f"{'interpretation (log model)':<40}")
for j, nm in enumerate(kept_names):
    pct = (np.exp(b3[j + 1]) - 1) * 100
    print(f"{nm:<14} {b3[j + 1]:>10.5f} {truth[nm]:>10.5f} "
          f"{'+1 unit -> ' + format(pct, '+.2f') + '% in price':<40}")

# standardised coefficients answer a DIFFERENT question
sd = Xtr[:, keep].std(0)
print(f"\n{'feature':<14} {'raw coef':>10} {'per-SD effect':>15}")
for j, nm in enumerate(kept_names):
    print(f"{nm:<14} {b3[j + 1]:>10.5f} {b3[j + 1] * sd[j]:>15.5f}")
print("\nRaw coefficients answer 'what does one more unit do'. Standardised")
print("ones answer 'which feature moves the prediction most across its")
print("observed range'. They rank features differently and neither is the")
print("importance of a feature in any causal sense (Chapter 30).")
