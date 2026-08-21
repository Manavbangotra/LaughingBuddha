# Extracted from: Chapter 26 — Experiment Design and A/B Testing
# Source: src/.../ch026-experiments.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A complete experiment: design, sanity checks, analysis, decision.

The randomisation-unit error is demonstrated explicitly, because it is the
most common way an experiment's conclusion is wrong while looking correct.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# --- the design, fixed in advance -------------------------------------------
DESIGN = {
    "hypothesis": "a simplified checkout raises completion",
    "primary": "checkout completion rate per user",
    "guardrails": ["p95 latency", "support contacts per user"],
    "baseline": 0.22,
    "mde": 0.015,
    "alpha": 0.05,
    "power": 0.80,
}
za = stats.norm.ppf(1 - DESIGN["alpha"] / 2)
zb = stats.norm.ppf(DESIGN["power"])
p = DESIGN["baseline"]
n_required = int(np.ceil(2 * (za + zb) ** 2 * p * (1 - p) / DESIGN["mde"] ** 2))

print("PRE-REGISTERED DESIGN")
for k, v in DESIGN.items():
    print(f"  {k:<12} {v}")
print(f"  {'n per arm':<12} {n_required:,}")

# --- simulate the experiment -------------------------------------------------
n_users = n_required
TRUE_LIFT = 0.018                       # a real effect, slightly above the MDE

user_prop = rng.beta(2, 6, n_users * 2)          # per-user baseline propensity
assign = rng.random(n_users * 2) < 0.5
sessions = rng.poisson(6, n_users * 2) + 1       # multiple sessions per user

completed, total_sessions, latency, support = [], [], [], []
for i in range(n_users * 2):
    p_i = np.clip(user_prop[i] + (TRUE_LIFT if assign[i] else 0), 0, 1)
    s = sessions[i]
    c = rng.binomial(s, p_i)
    completed.append(c)
    total_sessions.append(s)
    latency.append(rng.gamma(3, 60) * (1.22 if assign[i] else 1.0))  # slower!
    support.append(rng.poisson(0.05))

completed = np.array(completed); total_sessions = np.array(total_sessions)
latency = np.array(latency); support = np.array(support)

# --- sanity checks before looking at the result -----------------------------
print("\n" + "=" * 72)
print("SANITY CHECKS")
print("=" * 72)
n_a, n_b = int((~assign).sum()), int(assign.sum())
chi2 = ((n_a - (n_a+n_b)/2) ** 2 / ((n_a+n_b)/2)
        + (n_b - (n_a+n_b)/2) ** 2 / ((n_a+n_b)/2))
srm_p = 1 - stats.chi2.cdf(chi2, 1)
print(f"  sample ratio    : {n_a:,} / {n_b:,}  p = {srm_p:.3f}  "
      f"{'OK' if srm_p > 0.001 else 'BROKEN'}")

pre_a, pre_b = user_prop[~assign].mean(), user_prop[assign].mean()
pre_se = np.sqrt(user_prop[~assign].var()/n_a + user_prop[assign].var()/n_b)
pre_z = (pre_b - pre_a) / pre_se
print(f"  pre-period balance: {pre_a:.4f} vs {pre_b:.4f}, z = {pre_z:+.2f}  "
      f"{'OK' if abs(pre_z) < 3 else 'IMBALANCED'}")

# --- the randomisation-unit error -------------------------------------------
print("\n" + "=" * 72)
print("ANALYSIS — and why the unit matters")
print("=" * 72)

# WRONG: analyse per session, when randomisation was per user.
sess_a = completed[~assign].sum() / total_sessions[~assign].sum()
sess_b = completed[assign].sum() / total_sessions[assign].sum()
n_sess_a, n_sess_b = total_sessions[~assign].sum(), total_sessions[assign].sum()
se_sess = np.sqrt(sess_a*(1-sess_a)/n_sess_a + sess_b*(1-sess_b)/n_sess_b)

# RIGHT: aggregate to one value per user first.
rate_a = (completed / total_sessions)[~assign]
rate_b = (completed / total_sessions)[assign]
se_user = np.sqrt(rate_a.var(ddof=1)/len(rate_a) + rate_b.var(ddof=1)/len(rate_b))
diff_user = rate_b.mean() - rate_a.mean()

print(f"{'analysis unit':<18} {'estimate':>10} {'std error':>11} "
      f"{'95% CI':>22} {'z':>7}")
print(f"{'per session (WRONG)':<18} {sess_b-sess_a:>+10.4f} {se_sess:>11.5f} "
      f"{f'[{sess_b-sess_a-1.96*se_sess:+.4f}, {sess_b-sess_a+1.96*se_sess:+.4f}]':>22} "
      f"{(sess_b-sess_a)/se_sess:>7.1f}")
print(f"{'per user (RIGHT)':<18} {diff_user:>+10.4f} {se_user:>11.5f} "
      f"{f'[{diff_user-1.96*se_user:+.4f}, {diff_user+1.96*se_user:+.4f}]':>22} "
      f"{diff_user/se_user:>7.1f}")
print(f"\nThe per-session interval is {se_user/se_sess:.1f}x too narrow. Both")
print("estimate a similar effect; only one reports honest uncertainty.")
print("Randomisation was per user, so analysis must be per user (section 6.2).")

# --- guardrails ---------------------------------------------------------------
print("\n" + "=" * 72)
print("GUARDRAILS")
print("=" * 72)
lat_a, lat_b = np.percentile(latency[~assign], 95), np.percentile(latency[assign], 95)
sup_a, sup_b = support[~assign].mean(), support[assign].mean()
sup_se = np.sqrt(support[~assign].var()/n_a + support[assign].var()/n_b)

print(f"  p95 latency        : {lat_a:.0f} ms -> {lat_b:.0f} ms  "
      f"({(lat_b/lat_a - 1):+.1%})   "
      f"{'BREACH' if lat_b/lat_a - 1 > 0.05 else 'ok'}")
print(f"  support per user   : {sup_a:.4f} -> {sup_b:.4f}  "
      f"(z = {(sup_b-sup_a)/sup_se:+.2f})   ok")

# --- the decision -------------------------------------------------------------
print("\n" + "=" * 72)
print("DECISION")
print("=" * 72)
significant = abs(diff_user / se_user) > 1.96
practical = diff_user > DESIGN["mde"]
guardrail_ok = (lat_b / lat_a - 1) <= 0.05

print(f"  primary metric significant : {significant}")
print(f"  effect exceeds the MDE     : {practical} "
      f"({diff_user:+.4f} vs {DESIGN['mde']:+.4f})")
print(f"  guardrails passed          : {guardrail_ok}")
print(f"\n  -> {'SHIP' if (significant and practical and guardrail_ok) else 'DO NOT SHIP'}")
print("\nThe primary metric moved in the right direction and is statistically")
print("significant. The latency guardrail failed. Without the guardrail this")
print("would have shipped a change that trades checkout completion against")
print("page speed — a trade nobody agreed to make.")
