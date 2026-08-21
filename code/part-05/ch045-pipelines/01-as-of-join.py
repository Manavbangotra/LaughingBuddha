# Extracted from: Chapter 45 — Data and Feature Pipelines
# Source: src/.../ch045-pipelines.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The time-travel bug, built deliberately, and the as-of join that fixes it.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- a small event-sourced world --------------------------------------------
# Customers accumulate transactions. We predict whether a customer defaults
# in the 30 days after a decision date. The feature is their PEAK SPEND TO
# DATE — a running maximum, which is the kind of feature that makes time
# travel dangerous, because a spike caused by the label never washes out.
N_CUST, HORIZON = 600, 30.0


def simulate():
    """Return (feature_log, decisions) with explicit timestamps.

    feature_log rows are (customer, available_at, peak_spend), where
    peak_spend is the customer's RUNNING MAXIMUM spend to date. A running
    statistic is the realistic case and the dangerous one: once a value
    enters it, it never leaves, so a spike caused by the label event
    contaminates every later reading of the feature.

    `available_at` is when the value became QUERYABLE, which lags the event
    by a batch delay — that distinction is section 5.1's second detail and
    is where most as-of joins go wrong.
    """
    log, decisions = [], []
    for c in range(N_CUST):
        risk = rng.beta(2, 5)
        # riskier customers genuinely spend more, so the HONEST feature has
        # real signal — otherwise the comparison below measures nothing
        base = 100.0 + 900.0 * risk + 150.0 * rng.random()
        peak, t = base, 0.0
        while t < 300:
            t += rng.exponential(9.0)
            if t >= 300:
                break
            peak = max(peak, base * (1.0 + abs(rng.normal(0.0, 0.25))))
            log.append((c, t + 1.0, peak))           # 1 day to become visible
        t_dec = float(rng.uniform(120, 240))
        default = rng.random() < risk
        decisions.append((c, t_dec, int(default)))
        if default:
            # distress: a large spike shortly BEFORE the default event, which
            # is AFTER the decision date. It permanently raises the running
            # maximum, so any later read of the feature carries it.
            t_default = t_dec + rng.uniform(1, HORIZON)
            t_spike = t_default - rng.uniform(0.5, 4.0)
            if t_spike > 0:
                peak = max(peak, base * (3.0 + 2.0 * rng.random()))
                log.append((c, t_spike + 1.0, peak))
                # and it persists in every subsequent reading
                for tk in np.arange(t_spike + 5.0, 300.0, 9.0):
                    log.append((c, tk + 1.0, peak))
    log = np.array(sorted(log, key=lambda r: (r[0], r[1])), dtype=float)
    dec = np.array(decisions, dtype=float)
    return log, dec


log, dec = simulate()
print(f"{len(log):,} feature-log rows, {len(dec):,} decisions, "
      f"default rate {dec[:, 2].mean():.3f}")


# --- three joins: one correct, two subtly wrong -----------------------------
def join_latest(log, dec):
    """WRONG. Takes each customer's most recent value overall — i.e. the
    value as of NOW, which for a training row is the future. This is what a
    plain groupby-last produces, and it is the commonest form of the bug."""
    out = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        out[i] = rows[-1, 2] if len(rows) else np.nan
    return out


def join_as_of_event_time(log, dec, event_lag=1.0):
    """SUBTLY WRONG. Correct as-of semantics, but keyed on EVENT time rather
    than availability time — so it uses values that had not yet landed."""
    out = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        event_t = rows[:, 1] - event_lag              # undo the batch delay
        ok = rows[event_t < t]
        out[i] = ok[-1, 2] if len(ok) else np.nan
    return out


def join_as_of(log, dec):
    """CORRECT (eq. 45.1): most recent value whose AVAILABILITY time is
    strictly before the decision time."""
    out = np.empty(len(dec))
    age = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        ok = rows[rows[:, 1] < t]                     # strict inequality
        if len(ok):
            out[i] = ok[-1, 2]
            age[i] = t - ok[-1, 1]                    # section 5.1: return age
        else:
            out[i], age[i] = np.nan, np.inf
    return out, age


x_latest = join_latest(log, dec)
x_event = join_as_of_event_time(log, dec)
x_pit, age = join_as_of(log, dec)
y = dec[:, 2].astype(int)

print(f"\nfeature staleness under the correct join: "
      f"median {np.median(age[np.isfinite(age)]):.1f} days, "
      f"p95 {np.percentile(age[np.isfinite(age)], 95):.1f} days")


# --- what each join is worth ------------------------------------------------
def auc(y, s):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


print("\n" + "=" * 72)
print("what each join reports, and what it would actually deliver")
print("=" * 72)
print(f"{'join':<38} {'validation AUC':>15} {'verdict':<22}")
for name, x, verdict in (
        ("groupby-last (value as of NOW)", x_latest, "time travel"),
        ("as-of on EVENT time", x_event, "1-day time travel"),
        ("as-of on AVAILABILITY time", x_pit, "correct")):
    print(f"{name:<38} {auc(y, x):>15.4f} {verdict:<22}")

print("\nAll three joins run without error, produce a plausibly-distributed")
print("numeric feature, and would pass any schema check.")
print("\nThe first inflates AUC by about twenty points. It is the commonest")
print("form of the bug and, mercifully, the easiest to spot afterwards —")
print("a feature that good usually is too good.")
print("\nThe middle row is the dangerous one. Its as-of logic is CORRECT —")
print("most recent value strictly before the decision — and it is still")
print("wrong, because it keyed on the time the transaction HAPPENED rather")
print("than the time it became queryable. A one-day error, invisible in the")
print("code, and exactly the size a nightly batch boundary or a timezone")
print("mistake produces.")
print("\nIts inflation is about one point. That smallness is the lesson, not")
print("a reprieve: the size of a time-travel leak is set by how much")
print("label-carrying information falls inside the window you accidentally")
print("included. One day catches only the spikes that happen to land in it")
print("here. Shorten the label horizon from thirty days to one — a")
print("same-session conversion, a next-hour failure — and that same one-day")
print("error becomes the twenty-point row instead.")

# --- and it survives cross-validation (section 6.1) -------------------------
print("\n" + "=" * 72)
print("why cross-validation does not catch it (section 6.1)")
print("=" * 72)
folds = np.array_split(np.random.default_rng(1).permutation(len(y)), 5)
print(f"{'join':<38} " + " ".join(f"{'fold ' + str(i + 1):>8}"
                                  for i in range(5)) + f" {'spread':>8}")
for name, x in (("groupby-last", x_latest), ("as-of on availability", x_pit)):
    scores = [auc(y[f], x[f]) for f in folds]
    print(f"{name:<38} " + " ".join(f"{v:>8.3f}" for v in scores) +
          f" {max(scores) - min(scores):>8.3f}")

print("\nThe leaking join's folds do not merely agree — they agree MORE")
print("TIGHTLY than the correct join's, by a factor of about two on the")
print("spread. That is not a fluke: the leaked feature is a strong, clean")
print("signal, so every fold recovers it easily and they all land in the")
print("same place. The honest feature is weaker, so folds disagree more.")
print("\nSo the diagnostic people actually use — 'the folds agree, the")
print("estimate is stable, I trust it' — points the wrong way. Consistency")
print("across folds measures how reliably the signal is recoverable, not")
print("whether the signal should exist.")
print("\nThere is no split that detects a bad join. The join has to be right.")
