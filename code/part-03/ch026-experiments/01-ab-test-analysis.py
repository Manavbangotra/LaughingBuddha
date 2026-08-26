# -*- coding: utf-8 -*-
# Extracted from: Chapter 26 — Experiment Design and A/B Testing
# Source: src/.../ch026-experiments.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Designing and analysing an A/B test, with the failure modes measured.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# --- eq. 26.1: sample size before anything else -----------------------------
def sample_size(baseline, mde, alpha=0.05, power=0.80):
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    p = baseline
    return int(np.ceil(2 * (za + zb) ** 2 * p * (1 - p) / mde ** 2))


print("=" * 72)
print("design: how much traffic does this question need?")
print("=" * 72)
print(f"{'baseline':>9} {'MDE':>7} {'n per arm':>11} {'days at 5k/day':>16}")
for baseline in (0.05, 0.15):
    for mde in (0.005, 0.01, 0.02):
        n = sample_size(baseline, mde)
        print(f"{baseline:>9.0%} {mde:>7.1%} {n:>11,} {2*n/5000:>16.1f}")
print("\nHalving the MDE quadruples the cost (the delta^2 in eq. 26.1).")

# --- sanity check: sample ratio mismatch ------------------------------------
print("\n" + "=" * 72)
print("sanity check 1: sample ratio mismatch (eq. 26.3)")
print("=" * 72)


def srm_check(n_a, n_b, expected=0.5):
    total = n_a + n_b
    exp_a, exp_b = total * expected, total * (1 - expected)
    chi2 = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    return chi2, 1 - stats.chi2.cdf(chi2, df=1)


print(f"{'split':<22} {'chi2':>9} {'p-value':>11} {'verdict'}")
for label, (a, b) in {
    "50000 / 50000 (clean)": (50_000, 50_000),
    "50000 / 49800 (noise)": (50_000, 49_800),
    "50000 / 48500 (BUG)":   (50_000, 48_500),
}.items():
    chi2, p = srm_check(a, b)
    verdict = "BROKEN — discard" if p < 0.001 else "fine"
    print(f"{label:<22} {chi2:>9.2f} {p:>11.2e} {verdict}")
print("\nAn SRM is a bug report, not a finding. A 1.5% shortfall in one arm")
print("has a p-value near zero at this scale and means users were lost.")

# --- section 6.1: peeking, measured -----------------------------------------
print("\n" + "=" * 72)
print("peeking: the false-positive rate under a NULL effect")
print("=" * 72)


def run_experiment(n_total, true_lift, peeks, baseline=0.10, alpha=0.05):
    """Return True if the experiment 'wins' under the given peeking schedule."""
    a = rng.random(n_total) < baseline
    b = rng.random(n_total) < baseline + true_lift
    checkpoints = np.linspace(n_total // peeks, n_total, peeks).astype(int)
    for n in checkpoints:
        pa, pb = a[:n].mean(), b[:n].mean()
        se = np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
        if se > 0 and abs(pb - pa) / se > stats.norm.ppf(1 - alpha / 2):
            return True
    return False


N, TRIALS = 20_000, 1500
print(f"{'peeks':>7} {'false-positive rate':>22} {'inflation vs 5%':>18}")
for peeks in (1, 2, 5, 10, 20):
    hits = sum(run_experiment(N, 0.0, peeks) for _ in range(TRIALS))
    rate = hits / TRIALS
    print(f"{peeks:>7} {rate:>21.1%} {rate/0.05:>17.1f}x")

print("\nThere is no real effect in any of these runs. Checking twenty times")
print("instead of once turns a 5% error rate into something several times")
print("larger — the stopping rule selects for the noisiest moment.")

# --- eq. 26.4: analysis, reported properly ----------------------------------
print("\n" + "=" * 72)
print("analysis: report the interval, not the p-value")
print("=" * 72)


def analyse(conv_a, n_a, conv_b, n_b, alpha=0.05):
    pa, pb = conv_a / n_a, conv_b / n_b
    diff = pb - pa
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {"a": pa, "b": pb, "diff": diff, "ci": (diff - half, diff + half),
            "p": p, "rel": diff / pa}


scenarios = {
    "clear win":        (4800, 60_000, 5250, 60_000),
    "no effect, tight": (6000, 80_000, 6030, 80_000),
    "underpowered":     (140,   1_500,  160,   1_500),
}
for label, (ca, na, cb, nb) in scenarios.items():
    r = analyse(ca, na, cb, nb)
    lo, hi = r["ci"]
    sig = "significant" if r["p"] < 0.05 else "not significant"
    print(f"\n{label}:")
    print(f"  A {r['a']:.3%}  B {r['b']:.3%}  "
          f"diff {r['diff']:+.3%} ({r['rel']:+.1%} relative)")
    print(f"  95% CI [{lo:+.3%}, {hi:+.3%}]   p = {r['p']:.4f}  ({sig})")

print("\nThe second and third are both 'not significant' and mean completely")
print("different things. The tight one rules out any effect above ~0.2pp;")
print("the underpowered one rules out nothing and should not be reported as")
print("evidence of no effect.")

# --- eq. 26.6: CUPED variance reduction -------------------------------------
print("\n" + "=" * 72)
print("CUPED: free precision from pre-experiment data (eq. 26.6)")
print("=" * 72)

n = 40_000
for rho in (0.3, 0.5, 0.7, 0.9):
    pre = rng.normal(100, 25, n)
    noise_sd = 25 * np.sqrt(1 / rho ** 2 - 1)
    post = pre + rng.normal(0, noise_sd, n)
    assign = rng.random(n) < 0.5
    post = post + assign * 2.0                       # a true +2.0 effect

    naive = post[assign].mean() - post[~assign].mean()
    naive_se = np.sqrt(post[assign].var(ddof=1) / assign.sum()
                       + post[~assign].var(ddof=1) / (~assign).sum())

    theta = np.cov(post, pre)[0, 1] / pre.var(ddof=1)
    adj = post - theta * (pre - pre.mean())
    cuped = adj[assign].mean() - adj[~assign].mean()
    cuped_se = np.sqrt(adj[assign].var(ddof=1) / assign.sum()
                       + adj[~assign].var(ddof=1) / (~assign).sum())

    actual_rho = np.corrcoef(pre, post)[0, 1]
    print(f"rho={actual_rho:.2f}  naive {naive:+.3f} +/- {1.96*naive_se:.3f}   "
          f"CUPED {cuped:+.3f} +/- {1.96*cuped_se:.3f}   "
          f"variance x{(cuped_se/naive_se)**2:.2f} "
          f"(predicted {1-actual_rho**2:.2f})")

print("\nBoth estimate the same effect. CUPED's variance reduction of")
print("1 - rho^2 means at rho=0.7 you need roughly half the traffic.")
