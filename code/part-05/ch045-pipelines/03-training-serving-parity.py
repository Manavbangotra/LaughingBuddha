# -*- coding: utf-8 -*-
# Extracted from: Chapter 45 — Data and Feature Pipelines
# Source: src/.../ch045-pipelines.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""One definition, two paths, and a parity test that proves they agree.
"""
import numpy as np

rng = np.random.default_rng(21)

# --- the shared definition: ONE implementation, called by both paths --------
FEATURE_SPEC = {
    "avg_order_30d": {"window_days": 30, "agg": "mean", "default": 0.0},
    "n_orders_30d":  {"window_days": 30, "agg": "count", "default": 0.0},
    "max_order_90d": {"window_days": 90, "agg": "max", "default": 0.0},
}


def compute_features(events, as_of, spec=FEATURE_SPEC):
    """The single source of truth.

    `events` is an (n, 2) array of (available_at, amount) for ONE entity,
    already filtered to that entity. `as_of` is the decision time.

    Both the training job and the serving path call this. That is what
    removes code divergence (section 4.2) — and note that it removes ONLY
    code divergence. The other three causes are still live.
    """
    out = {}
    for name, s in spec.items():
        lo = as_of - s["window_days"]
        m = (events[:, 0] < as_of) & (events[:, 0] >= lo)   # strict, eq. 45.1
        vals = events[m, 1]
        if len(vals) == 0:
            out[name] = s["default"]
        elif s["agg"] == "mean":
            out[name] = float(vals.mean())
        elif s["agg"] == "count":
            out[name] = float(len(vals))
        else:
            out[name] = float(vals.max())
    return out


# --- a world of entities and events -----------------------------------------
def make_world(n_entities=300, seed=0):
    rs = np.random.default_rng(seed)
    events = {}
    for e in range(n_entities):
        n_ev = rs.poisson(28) + 3
        t = np.sort(rs.uniform(0, 365, n_ev))
        amt = rs.lognormal(3.6, 0.6, n_ev)
        events[e] = np.column_stack([t, amt])
    return events


world = make_world()


# --- the TRAINING path: batch, over history ---------------------------------
def build_training_rows(world, decisions):
    rows = []
    for e, t in decisions:
        rows.append(compute_features(world[e], t))
    return rows


# --- the SERVING path: one entity, now, from a cache ------------------------
class OnlineStore:
    """A cache refreshed on a schedule. Its staleness is the point."""

    def __init__(self, world, refresh_every=1.0, seed=0):
        self.world, self.refresh_every = world, refresh_every
        self.rs = np.random.default_rng(seed)

    def get_events(self, entity, now):
        """Return the events visible to serving, which is everything up to
        the last refresh — NOT up to `now`."""
        last_refresh = now - self.rs.uniform(0, self.refresh_every)
        ev = self.world[entity]
        return ev[ev[:, 0] < last_refresh], now - last_refresh


def serve(store, entity, now):
    ev, staleness = store.get_events(entity, now)
    feats = compute_features(ev, now)
    return feats, staleness


# --- the parity test --------------------------------------------------------
print("=" * 72)
print("parity test: do the two paths agree on the same inputs?")
print("=" * 72)
print("This is the test that should run in CI. Take real decision points,")
print("compute features both ways, and require agreement.\n")

decisions = [(int(e), float(t))
             for e, t in zip(rng.integers(0, 300, 400),
                             rng.uniform(120, 360, 400))]

# First: identical inputs, identical code. Parity must be EXACT.
store_fresh = OnlineStore(world, refresh_every=0.0, seed=1)
mismatches, max_rel = 0, 0.0
for e, t in decisions:
    train_f = compute_features(world[e], t)
    serve_f, _ = serve(store_fresh, e, t)
    for k in FEATURE_SPEC:
        denom = max(abs(train_f[k]), 1e-9)
        rel = abs(train_f[k] - serve_f[k]) / denom
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            mismatches += 1
print(f"  shared definition, zero staleness:")
print(f"    feature values compared : {len(decisions) * len(FEATURE_SPEC):,}")
print(f"    mismatches              : {mismatches}")
print(f"    max relative difference : {max_rel:.2e}")
print("  -> code divergence is eliminated by construction, because there is")
print("     only one implementation. This is what a feature store buys.")

# --- now the three causes a shared definition does NOT fix ------------------
print("\n" + "=" * 72)
print("the three causes a shared definition does NOT fix (section 4.2)")
print("=" * 72)

# 1. FRESHNESS
print("\n1. freshness — the online store is refreshed on a schedule")
print(f"{'refresh interval':>18} {'mean staleness':>16} "
      f"{'mean |rel. error|':>19} {'rows differing':>16}")
for label_h, interval in (("0 h", 0.0), ("1 h", 1 / 24), ("6 h", 6 / 24),
                          ("24 h", 1.0)):
    errs, diff, stales = [], 0, []
    st = OnlineStore(world, refresh_every=interval, seed=2)
    for e, t in decisions:
        train_f = compute_features(world[e], t)
        serve_f, stale = serve(st, e, t)
        stales.append(stale)
        for k in FEATURE_SPEC:
            denom = max(abs(train_f[k]), 1e-9)
            r = abs(train_f[k] - serve_f[k]) / denom
            errs.append(r)
            diff += r > 1e-9
    print(f"{label_h:>18} {np.mean(stales) * 24:>13.2f} h "
          f"{np.mean(errs):>19.4f} "
          f"{diff / (len(decisions) * len(FEATURE_SPEC)):>15.1%}")

print("\n   Same code, same definition, growing disagreement. Eq. 45.4 says")
print("   the damage is proportional to how fast the feature moves, which")
print("   is why a 30-day mean tolerates a day of staleness better than a")
print("   count does.")

# per-feature, to make that concrete
print("\n   per-feature, at a 24-hour refresh:")
st = OnlineStore(world, refresh_every=1.0, seed=3)
per = {k: [] for k in FEATURE_SPEC}
for e, t in decisions:
    train_f = compute_features(world[e], t)
    serve_f, _ = serve(st, e, t)
    for k in FEATURE_SPEC:
        per[k].append(abs(train_f[k] - serve_f[k]) / max(abs(train_f[k]), 1e-9))
for k, v in per.items():
    print(f"     {k:<16} mean relative error {np.mean(v):>8.4f}")

# 2. TIME TRAVEL — a shared definition called with the wrong timestamp
print("\n2. time travel — the SAME function, called with as_of = now + 1 day")
bad = [compute_features(world[e], t + 1.0) for e, t in decisions[:200]]
good = [compute_features(world[e], t) for e, t in decisions[:200]]
diff = np.mean([b["n_orders_30d"] != g["n_orders_30d"]
                for b, g in zip(bad, good)])
print(f"   rows whose 30-day order count changed: {diff:.1%}")
print("   The definition is shared and correct. The CALLER passed a")
print("   timestamp one day late, and about a sixth of the rows now contain")
print("   information from the future. No shared implementation prevents")
print("   this; only a correct as-of join does.")

# 3. AVAILABILITY
print("\n3. availability — a feature the warehouse can compute and serving")
print("   cannot within its latency budget")
LATENCY_BUDGET_MS = 50.0
COST_MS = {"avg_order_30d": 3.0, "n_orders_30d": 2.0, "max_order_90d": 9.0,
           "pct_rank_vs_cohort_365d": 140.0}
print(f"\n{'feature':<28} {'serving cost':>13} {'within budget?':>16}")
for k, c in COST_MS.items():
    print(f"{k:<28} {c:>10.0f} ms {'yes' if c < LATENCY_BUDGET_MS else 'NO':>16}")
print(f"\n   The last feature is perfectly computable offline and is 2.8x the")
print(f"   entire {LATENCY_BUDGET_MS:.0f}ms budget on its own. Discovering that")
print("   AFTER training on it means either dropping the feature and")
print("   retraining, or serving a default the model has never seen. The")
print("   check belongs at feature-definition time.")

# --- the summary table ------------------------------------------------------
print("\n" + "=" * 72)
print("what fixes what")
print("=" * 72)
rows = [
    ("code divergence", "shared definition", "SOLVED — measured above"),
    ("time travel", "as-of join on availability time", "caller must be right"),
    ("freshness", "match the staleness distributions", "manage, not eliminate"),
    ("availability", "check the latency budget up front", "a design constraint"),
]
print(f"{'cause':<20} {'remedy':<36} {'status':<26}")
for c, r, st_ in rows:
    print(f"{c:<20} {r:<36} {st_:<26}")
print("\nA feature store is a good way to get the first row and a convenient")
print("way to get the second. It does not get you the third or fourth, and")
print("the decision to adopt one should turn on how many independent")
print("consumers read the features and whether anything serves them in real")
print("time — not on how sophisticated the team wishes to appear.")
