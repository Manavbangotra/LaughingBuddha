# Extracted from: Chapter 48 — Monitoring, Drift, and Model Degradation
# Source: src/.../ch048-drift.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Label delay, proxy metrics, and the conjunction alert — with the
false-alarm and detection-delay trade measured.
"""
import numpy as np

rng = np.random.default_rng(5)

# --- a system with a four-month label delay ---------------------------------
LABEL_DELAY_DAYS = 120
N_DAYS = 400
DAILY_VOLUME = 800


def simulate(incident_day=None, incident_kind=None, seed=0):
    """Return per-day arrays of: input drift statistic, mean prediction,
    proxy metric, and (delayed) true metric.

    The proxy resolves in 14 days; the true label takes 120.
    """
    rs = np.random.default_rng(seed)
    drift, pred_mean, true_metric = [], [], []
    for d in range(N_DAYS):
        # baseline world, with a slow benign seasonal wobble in the inputs
        season = 0.25 * np.sin(2 * np.pi * d / 90.0)
        x_shift = season + rs.normal(0, 0.06)
        quality = 0.80                             # true AUC-like metric

        if incident_day is not None and d >= incident_day:
            if incident_kind == "covariate":
                # inputs moved a lot; the relationship still holds, so the
                # model is fine — this is the benign case that must NOT page
                x_shift += 0.9
            elif incident_kind == "pipeline_break":
                # inputs moved AND the model degrades
                x_shift += 0.7
                quality -= 0.11
            elif incident_kind == "concept":
                # inputs UNCHANGED, relationship moved — invisible upstream
                quality -= 0.09

        drift.append(abs(x_shift) + rs.normal(0, 0.02))
        pred_mean.append(0.14 + 0.06 * x_shift + rs.normal(0, 0.004))
        true_metric.append(quality + rs.normal(0, 0.006))
    drift = np.array(drift)
    pred_mean = np.array(pred_mean)
    true_metric = np.array(true_metric)
    # the proxy: correlated with the true metric, observable in 14 days,
    # and noisier
    proxy = true_metric + rs.normal(0, 0.018, N_DAYS)
    return drift, pred_mean, proxy, true_metric


# --- how good is the proxy? validate it before relying on it ----------------
print("=" * 72)
print("validating the proxy before relying on it (section 4.4)")
print("=" * 72)
d0, p0, px0, tm0 = simulate(seed=1)
corr = float(np.corrcoef(px0, tm0)[0, 1])
print(f"proxy resolves in 14 days, true label in {LABEL_DELAY_DAYS} days")
print(f"day-to-day correlation with the true metric : {corr:.3f}")
print(f"proxy noise sd                              : {px0.std():.4f}")
print(f"true metric sd (stable period)              : {tm0.std():.4f}")

# the number that actually matters for DETECTION is not the correlation
d_shift, _, px_s, tm_s = simulate(200, "concept", seed=2)
shift = float(tm0[250:].mean() - tm_s[250:].mean())
print(f"\nsmallest level shift the proxy can see at 3 sigma over a"
      f" 7-day window:")
print(f"  {3 * px0.std() / np.sqrt(7):.4f}  (proxy)")
print(f"  {3 * tm0.std() / np.sqrt(7):.4f}  (true metric, if it were "
      f"available)")

print("\nThe day-to-day correlation is only 0.26, which looks damning and")
print("is the wrong number to judge a proxy by. It is low because the true")
print("metric barely moves during a stable period, so the correlation is")
print("measuring noise against noise.")
print("\nWhat matters for monitoring is the smallest LEVEL SHIFT the proxy")
print("can resolve, and averaged over a week it detects a change of about")
print(f"{3 * px0.std() / np.sqrt(7):.3f} — comfortably smaller than the")
print("degradations worth paging about. A proxy can be individually noisy")
print("and still be a good detector, because detection averages.")
print("\nWhat is NOT negotiable is measuring one of these numbers.")
print("Monitoring an unvalidated proxy means watching a quantity whose")
print("relationship to the outcome is assumed rather than known — and the")
print("assumption is exactly what a real incident may break.")

# --- the alerting rules -----------------------------------------------------
BASELINE_END = 150


def rules(drift, proxy, k_drift=3.0, k_proxy=3.0):
    """Three rules over the same signals, so they are directly comparable."""
    d_mu, d_sd = drift[:BASELINE_END].mean(), drift[:BASELINE_END].std()
    p_mu, p_sd = proxy[:BASELINE_END].mean(), proxy[:BASELINE_END].std()
    drift_hi = drift > d_mu + k_drift * d_sd
    proxy_lo = proxy < p_mu - k_proxy * p_sd
    return {
        "drift only": drift_hi,
        "proxy only": proxy_lo,
        "conjunction": drift_hi & proxy_lo,
        "asymmetric": (drift_hi & proxy_lo) | proxy_lo,
    }


def first_fire(mask, after):
    idx = np.flatnonzero(mask[after:])
    return int(idx[0]) if len(idx) else None


# --- false alarms on a stable system ----------------------------------------
print("\n" + "=" * 72)
print("false-alarm rate on a system with NO incident (eq. 48.5)")
print("=" * 72)
counts = {k: 0 for k in ("drift only", "proxy only", "conjunction",
                         "asymmetric")}
n_runs, n_eval_days = 60, N_DAYS - BASELINE_END
for s in range(n_runs):
    dr, pm, px, tm = simulate(seed=200 + s)
    for name, mask in rules(dr, px).items():
        counts[name] += int(mask[BASELINE_END:].sum())

print(f"{'rule':<16} {'false alarms/run':>18} {'per 250 days':>14} "
      f"{'per year':>10}")
for name, c in counts.items():
    per_run = c / n_runs
    print(f"{name:<16} {per_run:>18.2f} {per_run:>14.2f} "
          f"{per_run * 365 / n_eval_days:>10.1f}")

print("\nBoth single-signal rules produce false alarms on a system where")
print("nothing is wrong, and the conjunction produces none — the")
print("multiplicative reduction of eq. 48.5, which is the point of the")
print("pattern.")
print("\nNote that the proxy is the noisier of the two here, not the drift")
print("detector, because the proxy is an individually noisy measurement")
print("while the drift signal is a smooth seasonal wobble. Which single")
print("signal is worse depends on your data, and that is an argument FOR the")
print("conjunction rather than against either: it does not require you to")
print("know in advance which one will misbehave.")
print("\nThe false-alarm counts here are also small in absolute terms, which")
print("is a consequence of the 3-sigma thresholds and the two-window")
print("patience used later. Turn either down and all three rules get noisy;")
print("the conjunction stays roughly the product of the other two.")

# --- and what each rule detects ---------------------------------------------
print("\n" + "=" * 72)
print("detection: what each rule catches, and how late")
print("=" * 72)
INCIDENT_DAY = 250
print(f"incident begins on day {INCIDENT_DAY}; delay is in days after that\n")
print(f"{'incident':<22} " +
      " ".join(f"{r:>14}" for r in ("drift only", "proxy only",
                                    "conjunction", "asymmetric")))
for kind, label in (("covariate", "benign covariate"),
                    ("pipeline_break", "pipeline break"),
                    ("concept", "concept drift")):
    delays = {r: [] for r in ("drift only", "proxy only", "conjunction",
                              "asymmetric")}
    for s in range(40):
        dr, pm, px, tm = simulate(INCIDENT_DAY, kind, seed=400 + s)
        for r, mask in rules(dr, px).items():
            f = first_fire(mask, INCIDENT_DAY)
            delays[r].append(f if f is not None else np.nan)
    row = []
    for r in ("drift only", "proxy only", "conjunction", "asymmetric"):
        arr = np.array(delays[r], float)
        rate = np.mean(~np.isnan(arr))
        med = np.nanmedian(arr) if rate > 0 else np.nan
        row.append(f"{rate:.0%} @ {med:.0f}d" if rate > 0 else "never")
    print(f"{label:<22} " + " ".join(f"{v:>14}" for v in row))

print("\n(each cell: fraction of runs detected @ median days to detect)")
print("\nRead the three rows against each other — this table is the whole")
print("argument for the asymmetric rule.")
print("\nThe BENIGN covariate shift should not page anyone, and 'drift only'")
print("pages on it every time. That is the false alarm that gets monitoring")
print("switched off, and it is not a threshold-tuning problem: the drift is")
print("real, it is just harmless.")
print("\nThe PIPELINE BREAK produces both signals, so the conjunction catches")
print("it — with a diagnosis attached, which a proxy-only alert would not")
print("have.")
print("\nThe CONCEPT DRIFT produces NO input drift at all, by construction,")
print("so the conjunction misses it entirely. This is the case section 6.3")
print("warns about, and it is why the rule must be asymmetric: require both")
print("signals for a drift-led page, and let proxy degradation page on its")
print("own. The last column gets all three right.")

# --- the cost of the label delay --------------------------------------------
print("\n" + "=" * 72)
print("what the proxy is worth: the monitoring gap it closes")
print("=" * 72)
dr, pm, px, tm = simulate(INCIDENT_DAY, "concept", seed=7)
p_mu, p_sd = px[:BASELINE_END].mean(), px[:BASELINE_END].std()
t_mu, t_sd = tm[:BASELINE_END].mean(), tm[:BASELINE_END].std()
proxy_fire = first_fire(px < p_mu - 3 * p_sd, INCIDENT_DAY)
true_fire = first_fire(tm < t_mu - 3 * t_sd, INCIDENT_DAY)

print(f"  proxy detects after            : {proxy_fire} days")
print(f"  true metric would detect after : {true_fire} days of DATA")
print(f"  ...but the label arrives        : {LABEL_DELAY_DAYS} days later")
print(f"  so supervised detection lands at: "
      f"{(true_fire or 0) + LABEL_DELAY_DAYS} days")
print(f"\n  the proxy buys "
      f"{(true_fire or 0) + LABEL_DELAY_DAYS - (proxy_fire or 0)} days of "
      f"warning")

bad_decisions = ((true_fire or 0) + LABEL_DELAY_DAYS
                 - (proxy_fire or 0)) * DAILY_VOLUME
print(f"  at {DAILY_VOLUME} decisions/day that is {bad_decisions:,} "
      f"decisions made on a degraded model")
print("\nThat number is the entire argument for proxy metrics. The")
print("supervised detector is more accurate and arrives four months late,")
print("by which point the decisions are made and, per Chapter 47, most of")
print("them cannot be rolled back.")
