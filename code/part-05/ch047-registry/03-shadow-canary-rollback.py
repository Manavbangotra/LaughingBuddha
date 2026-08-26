# -*- coding: utf-8 -*-
# Extracted from: Chapter 47 — Model Registry and the Deployment Handoff
# Source: src/.../ch047-registry.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Shadow, canary and rollback — what each one can actually detect.
"""
import numpy as np

rng = np.random.default_rng(11)

N_DAILY = 1_000_000


# --- section 6.2: what a canary is powered to detect ------------------------
def required_n(delta, sigma, power_z=0.84, alpha_z=1.96):
    """Eq. 47.5: observations needed per arm to detect `delta`."""
    return 2 * (alpha_z + power_z) ** 2 * sigma ** 2 / delta ** 2


print("=" * 72)
print("what a canary can detect, and what it cannot (eq. 47.5)")
print("=" * 72)
print(f"traffic {N_DAILY:,}/day; the canary serves a fraction f of it\n")
print(f"{'signal':<26} {'delta':>8} {'sigma':>7} {'n needed':>12} "
      f"{'f=1%: hours':>13} {'f=5%: hours':>13}")
signals = [
    ("error rate 0.1% -> 2%", 0.019, 0.14),
    ("p99 latency +30ms", 30.0, 45.0),
    ("mean prediction +0.05", 0.05, 0.30),
    ("conversion -1 point", 0.010, 0.30),
    ("conversion -0.2 points", 0.002, 0.30),
]
for name, delta, sigma in signals:
    n = required_n(delta, sigma)
    h1 = n / (0.01 * N_DAILY / 24)
    h5 = n / (0.05 * N_DAILY / 24)
    print(f"{name:<26} {delta:>8.3f} {sigma:>7.2f} {n:>12,.0f} "
          f"{h1:>13.1f} {h5:>13.1f}")

print("\nThe split is stark and it is the practical content of this section.")
print("Breakage — errors, latency, a prediction distribution that moved —")
print("has a large effect size relative to its noise and is detectable in")
print("MINUTES at 1% of traffic. A one-point conversion regression needs a")
print("day and a half at 1%; a fifth of a point needs five weeks.")
print("\nSo a canary detects BREAKAGE, not DEGRADATION. Watching a business")
print("metric on a 1% canary and concluding 'no regression' after an hour is")
print("not evidence of anything — the experiment had no power to see one.")

# --- shadow: what running without acting can tell you -----------------------
print("\n" + "=" * 72)
print("shadow deployment: comparing distributions without acting")
print("=" * 72)


def score_batch(model_kind, n, rs):
    """Simulate a scoring pass; returns (predictions, latencies_ms)."""
    if model_kind == "incumbent":
        p = rs.beta(2.0, 12.0, n)
        lat = rs.gamma(4.0, 3.0, n)
    elif model_kind == "candidate_ok":
        p = rs.beta(2.1, 12.0, n)
        lat = rs.gamma(4.2, 3.1, n)
    elif model_kind == "candidate_shifted":
        p = rs.beta(3.4, 9.0, n)            # scores much higher on average
        lat = rs.gamma(4.1, 3.0, n)
    else:                                    # a pipeline-version mismatch
        p = rs.beta(2.0, 12.0, n)
        bad = rs.random(n) < 0.08            # 8% get a default feature value
        p[bad] = 0.5                         # ...so they all score identically
        lat = rs.gamma(4.0, 3.0, n)
    return p, lat


def point_mass(p, tol=1e-9):
    """Largest fraction of predictions taking a single identical value.

    A distribution summary like PSI can miss this entirely — a spike inside
    an existing bin barely moves the bin's mass — and it is the signature of
    a pipeline mismatch, where some rows fall back to a default feature and
    therefore all score the same. Worth checking on its own.
    """
    vals, counts = np.unique(np.round(p / tol) * tol, return_counts=True)
    return float(counts.max() / len(p)), float(vals[counts.argmax()])


def psi(ref, cur, bins=10):
    """Population stability index between two prediction distributions."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


rs = np.random.default_rng(5)
n_shadow = 20_000
p_inc, lat_inc = score_batch("incumbent", n_shadow, rs)

print(f"shadow run over {n_shadow:,} real requests, outputs discarded\n")
pm_inc, _ = point_mass(p_inc)
print(f"{'candidate':<26} {'mean pred':>10} {'PSI':>8} {'point mass':>11} "
      f"{'p99 ms':>8} {'verdict':<26}")
print(f"{'incumbent (reference)':<26} {p_inc.mean():>10.4f} {0.0:>8.4f} "
      f"{pm_inc:>11.4f} {np.percentile(lat_inc, 99):>8.1f} {'-':<26}")
for kind, label in (("candidate_ok", "v7  refinement"),
                    ("candidate_shifted", "v8  large shift"),
                    ("candidate_broken", "v9  pipeline mismatch")):
    p_c, lat_c = score_batch(kind, n_shadow, rs)
    ps = psi(p_inc, p_c)
    pm, pm_val = point_mass(p_c)
    p99 = float(np.percentile(lat_c, 99))
    if pm > 10 * max(pm_inc, 1e-4):
        verdict = f"point mass at {pm_val:.2f}"
    elif ps >= 0.25:
        verdict = "distribution moved too far"
    elif ps >= 0.1:
        verdict = "investigate"
    else:
        verdict = "looks like a refinement"
    print(f"{label:<26} {p_c.mean():>10.4f} {ps:>8.4f} {pm:>11.4f} "
          f"{p99:>8.1f} {verdict:<26}")
print("\nNo user was affected by any of this, and two of the three candidates")
print("are already disqualified.")
print("\nNote that they were caught by DIFFERENT checks, and that the second")
print("one needed a check PSI does not provide. v8's whole distribution")
print("moved, which PSI reports loudly. v9's did not: 92% of its predictions")
print("are perfectly normal and 8% are pinned to exactly 0.50 because those")
print("rows fell back to a default feature. That spike sits inside an")
print("existing bin, so PSI barely registers it — 0.04, comfortably in")
print("'looks like a refinement' territory.")
print("\nA point-mass check finds it immediately, because 8% of predictions")
print("taking one identical value is not something a working model does. The")
print("general lesson: a distributional summary can be blind to a")
print("degenerate mode, and the pipeline-mismatch failure of section 4.2")
print("produces exactly that shape. Check for spikes as well as for shift.")
print("\nWhat shadow mode cannot tell you is whether the candidate is")
print("BETTER. Nothing acted on its predictions, so there are no outcomes to")
print("compare.")

# --- rollback: bounding the damage ------------------------------------------
print("\n" + "=" * 72)
print("rollback: the only property that bounds the cost")
print("=" * 72)


def incident_cost(detect_min, rollback_min, rate_per_min, cost_per_bad):
    return (detect_min + rollback_min) * rate_per_min * cost_per_bad


RATE = N_DAILY / (24 * 60)
COST_PER_BAD = 0.40
print(f"traffic {RATE:,.0f} requests/min; each bad decision costs "
      f"GBP {COST_PER_BAD:.2f}\n")
print(f"{'setup':<38} {'detect':>8} {'rollback':>9} {'incident cost':>15}")
setups = [
    ("previous artefact kept warm", 4, 1),
    ("rebuild and redeploy from CI", 4, 25),
    ("rebuild, plus pipeline revert", 4, 40),
    ("no automated detection", 240, 25),
]
for label, det, rb in setups:
    print(f"{label:<38} {det:>6} m {rb:>7} m "
          f"{incident_cost(det, rb, RATE, COST_PER_BAD):>13,.0f}")

print("\nEverything upstream of this chapter reduces the PROBABILITY of a bad")
print("deploy. Rollback bounds its DURATION, and duration is what the cost")
print("is proportional to. Keeping the previous artefact warm turns a")
print("half-hour incident into a five-minute one for the price of some idle")
print("memory.")
print("\nThe last row is the one to notice: with no automated detection, the")
print("rollback speed barely matters. Detection time dominates, which is")
print("why Chapter 48 exists and why the ML Test Score puts monitoring in")
print("its own category.")

# --- what cannot be rolled back ---------------------------------------------
print("\n" + "=" * 72)
print("what rollback does NOT undo (section 5.5)")
print("=" * 72)
cases = [
    ("a ranking shown to users", "reverted on the next request", "yes"),
    ("a fraud score used to decline", "the decline already happened", "no"),
    ("an email that was sent", "irreversible", "no"),
    ("a price that was quoted", "may be contractually binding", "no"),
    ("a recommendation logged as training data",
     "poisons the next model too", "no, and it compounds"),
]
print(f"{'effect of a prediction':<38} {'after rollback':<34} "
      f"{'undone?':<18}")
for what, after, undone in cases:
    print(f"{what:<38} {after:<34} {undone:<18}")

print("\nWhere the answer is 'no', rollback is not a safety net and the gate")
print("has to carry the weight instead. The last row is the worst: a model")
print("whose outputs become training data for its successor has a feedback")
print("loop (Chapter 30), so a bad deployment contaminates future models")
print("even after it has been reverted.")
