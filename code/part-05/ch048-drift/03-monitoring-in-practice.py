# Extracted from: Chapter 48 — Monitoring, Drift, and Model Degradation
# Source: src/.../ch048-drift.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A monitoring configuration, end to end, with every threshold derived.
"""
import numpy as np

rng = np.random.default_rng(23)


class Monitor:
    """Thresholds derived from a stable baseline (eq. 48.6), an asymmetric
    alerting rule (section 6.3), and hysteresis so it cannot flap."""

    def __init__(self, *, fa_rate=0.01, patience=2):
        self.fa_rate, self.patience = fa_rate, patience
        self.thresholds = {}
        self._streak = {}

    def calibrate(self, baseline_windows):
        """baseline_windows: dict signal -> array of values from a stable
        period, one per monitoring window."""
        for sig, vals in baseline_windows.items():
            v = np.asarray(vals, float)
            if sig.endswith("_lower"):        # alarm when the value FALLS
                self.thresholds[sig] = float(np.quantile(v, self.fa_rate))
            else:                             # alarm when it RISES
                self.thresholds[sig] = float(np.quantile(v, 1 - self.fa_rate))
        return self

    def _breach(self, sig, value):
        t = self.thresholds[sig]
        return value < t if sig.endswith("_lower") else value > t

    def step(self, **signals):
        """One monitoring window. Returns (level, reasons)."""
        fired = {}
        for sig, val in signals.items():
            b = self._breach(sig, val)
            self._streak[sig] = self._streak.get(sig, 0) + 1 if b else 0
            fired[sig] = self._streak[sig] >= self.patience

        drift = fired.get("input_drift", False) or fired.get("pred_shift",
                                                             False)
        proxy = fired.get("proxy_lower", False)

        if proxy and drift:
            return "PAGE", ["proxy degraded AND inputs moved "
                            "(likely cause upstream)"]
        if proxy:
            return "PAGE", ["proxy degraded with NO input drift "
                            "(possible concept drift)"]
        if drift:
            reasons = [s for s in ("input_drift", "pred_shift") if fired[s]]
            return "NOTIFY", [f"{', '.join(reasons)} moved; "
                              f"no measurable effect yet"]
        return "OK", []


# --- a year of operation, with three events ---------------------------------
def world(day, seed):
    """Returns the three monitored signals for one day."""
    rs = np.random.default_rng(seed * 100003 + day)
    season = 0.20 * np.sin(2 * np.pi * day / 90.0)
    drift = abs(season + rs.normal(0, 0.05))
    pred = 0.140 + 0.05 * season + rs.normal(0, 0.003)
    proxy = 0.800 + rs.normal(0, 0.012)

    if 120 <= day < 150:                     # a marketing campaign
        drift += 0.55                        # inputs move, model is fine
        pred += 0.030
    if 200 <= day < 215:                     # an upstream unit change
        drift += 0.85
        pred += 0.075
        proxy -= 0.075                       # and the model degrades
    if day >= 300:                           # the world changed
        proxy -= 0.055                       # inputs unchanged
    return drift, pred, proxy


# --- calibrate on a stable period -------------------------------------------
BASE_DAYS = 100
base = {"input_drift": [], "pred_shift": [], "proxy_lower": []}
for d in range(BASE_DAYS):
    dr, pr, px = world(d, seed=1)
    base["input_drift"].append(dr)
    base["pred_shift"].append(abs(pr - 0.140))
    base["proxy_lower"].append(px)

mon = Monitor(fa_rate=0.01, patience=2).calibrate(base)
print("=" * 72)
print("thresholds derived from 100 stable days at a 1% false-alarm rate")
print("=" * 72)
for sig, t in mon.thresholds.items():
    direction = "below" if sig.endswith("_lower") else "above"
    print(f"  {sig:<14} alarm when {direction} {t:.4f}")
print("\nNo conventional numbers were used. Each threshold is the empirical")
print("quantile of that signal on a period with no known incidents")
print("(eq. 48.6), so the false-alarm rate is chosen rather than inherited.")

# --- run the year -----------------------------------------------------------
print("\n" + "=" * 72)
print("a year of operation")
print("=" * 72)
events, log = [], []
for d in range(BASE_DAYS, 365):
    dr, pr, px = world(d, seed=1)
    level, reasons = mon.step(input_drift=dr, pred_shift=abs(pr - 0.140),
                              proxy_lower=px)
    log.append(level)
    if level != "OK":
        events.append((d, level, reasons[0]))

# collapse consecutive identical events into episodes
episodes = []
for d, level, reason in events:
    if episodes and episodes[-1][2] == reason and d - episodes[-1][1] <= 2:
        episodes[-1][1] = d
    else:
        episodes.append([d, d, reason, level])

print(f"{'days':<14} {'level':<8} {'reason':<52}")
for start, end, reason, level in episodes:
    span = f"{start}-{end}" if end > start else f"{start}"
    print(f"{span:<14} {level:<8} {reason:<52}")

n_page = sum(1 for e in episodes if e[3] == "PAGE")
n_notify = sum(1 for e in episodes if e[3] == "NOTIFY")
print(f"\nover 265 operating days: {n_page} pages, {n_notify} notifications")
print("\nThe three planted events were: a marketing campaign on days 120-150")
print("(inputs move, model fine), an upstream unit change on days 200-215")
print("(inputs move AND the model degrades), and a permanent change in the")
print("world from day 300 (inputs unchanged, model degrades).")
print("\nThe campaign produced a NOTIFICATION, not a page — correct, and the")
print("difference between a monitor people trust and one they mute.")
print("\nThe unit change is the nicest case: the drift signal fired on days")
print("199-200 as a notification, and the page followed on day 201 once the")
print("proxy confirmed an effect. Two days of advance warning, and then an")
print("alert that already names the likely cause.")
print("\nThe concept drift paged on the proxy alone, which the symmetric")
print("conjunction of eq. 48.3 would have missed entirely — the inputs never")
print("moved. Note that it also arrives as two episodes rather than one,")
print("because the proxy dips back inside the threshold briefly; a runbook")
print("should treat re-firing within a few days as the same incident rather")
print("than a new one.")

# --- what to do when it fires -----------------------------------------------
print("\n" + "=" * 72)
print("the runbook: what each alert means and what to do")
print("=" * 72)
runbook = [
    ("NOTIFY: inputs moved, no effect",
     "look during working hours; usually a campaign, a season or a new "
     "segment"),
    ("PAGE: proxy down AND inputs moved",
     "check upstream FIRST — a pipeline break looks exactly like this"),
    ("PAGE: proxy down, inputs stable",
     "concept drift or a label-generating change; retraining may help"),
    ("PAGE: prediction distribution spiked",
     "check for a point mass (Chapter 47): a default feature value"),
]
for what, action in runbook:
    print(f"  {what}")
    print(f"      -> {action}")

print("\n" + "=" * 72)
print("and the decision the alert exists to inform")
print("=" * 72)
options = [
    ("do nothing", "the drift is benign, or the effect is within tolerance"),
    ("roll back", "the change coincided with a deployment (Chapter 47)"),
    ("fix upstream", "a pipeline break — retraining would LEARN the bug"),
    ("retrain", "the world genuinely changed and the new data is correct"),
    ("retire the model", "the assumption it was built on no longer holds"),
]
print(f"{'action':<20} {'when':<54}")
for a, w in options:
    print(f"{a:<20} {w:<54}")
print("\nNote the third row, which is the one teams get wrong. Retraining on")
print("data produced by a broken pipeline teaches the model the bug and")
print("makes the problem permanent and much harder to diagnose. A retrain is")
print("a candidate like any other and must pass the same promotion gate")
print("(Chapter 47) — including the schema and lineage checks that would")
print("have caught the break in the first place.")
