# Extracted from: Chapter 47 — Model Registry and the Deployment Handoff
# Source: src/.../ch047-registry.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A registry with stages, and a promotion gate that is executable policy.
"""
import json
import time

import numpy as np

rng = np.random.default_rng(0)


# --- the entry (eq. 47.1) ---------------------------------------------------
class RegistryEntry:
    def __init__(self, name, version):
        self.e = {
            "name": name, "version": version, "stage": "registered",
            "artefact": None, "pipeline_version": None, "schema": None,
            "lineage": {}, "eval": {}, "contract": {}, "audit": [],
        }

    def set(self, **kw):
        self.e.update(kw)
        return self

    def log(self, action, actor, detail):
        self.e["audit"].append({"action": action, "actor": actor,
                                "detail": detail, "seq": len(self.e["audit"])})
        return self


# --- the gate (eq. 47.4) ----------------------------------------------------
def gate(candidate, incumbent, *, margin, floors, latency_budget_ms,
         max_dist_shift=0.15):
    """Each check returns (name, passed, reason). The gate returns all of
    them, not just the first failure — a gate that stops at the first
    problem makes people fix issues one deploy at a time."""
    results = []
    ce, ie = candidate.e, (incumbent.e if incumbent else None)

    # 1. aggregate floor
    auc = ce["eval"].get("auc")
    results.append(("aggregate_auc_floor", auc is not None
                    and auc >= floors["auc"],
                    f"auc={auc} floor={floors['auc']}"))

    # 2. per-slice floors — the check that catches 'improved on average'
    slices = ce["eval"].get("slices", {})
    worst = min(slices.items(), key=lambda kv: kv[1]) if slices else None
    results.append(("slice_auc_floor",
                    bool(slices) and worst[1] >= floors["slice_auc"],
                    f"worst slice {worst[0]}={worst[1]:.4f} "
                    f"floor={floors['slice_auc']}" if worst else "no slices"))

    # 3. non-inferiority, NOT superiority (section 5.4)
    if ie:
        delta = auc - ie["eval"]["auc"]
        results.append(("non_inferior_to_incumbent", delta >= -margin,
                        f"delta={delta:+.4f} margin=-{margin:.4f}"))
    else:
        results.append(("non_inferior_to_incumbent", True, "no incumbent"))

    # 4. calibration
    ece = ce["eval"].get("ece")
    results.append(("calibration", ece is not None and ece <= floors["ece"],
                    f"ece={ece} max={floors['ece']}"))

    # 5. prediction-distribution shift vs incumbent
    if ie and "pred_mean" in ce["eval"] and "pred_mean" in ie["eval"]:
        shift = abs(ce["eval"]["pred_mean"] - ie["eval"]["pred_mean"])
        rel = shift / max(ie["eval"]["pred_mean"], 1e-9)
        results.append(("prediction_distribution", rel <= max_dist_shift,
                        f"mean prediction moved {rel:.1%} "
                        f"(max {max_dist_shift:.0%})"))
    else:
        results.append(("prediction_distribution", True, "not comparable"))

    # 6. latency
    p99 = ce["eval"].get("p99_latency_ms")
    results.append(("latency_p99", p99 is not None and p99 <= latency_budget_ms,
                    f"p99={p99}ms budget={latency_budget_ms}ms"))

    # 7. schema conformance — catches a pipeline/model version mismatch
    ok_schema = (ce["schema"] is not None
                 and (ie is None or ce["schema"] == ie["schema"]
                      or ce.get("schema_change_approved")))
    results.append(("schema_conformance", ok_schema,
                    "schema matches incumbent or change is approved"))

    # 8. lineage completeness
    need = {"commit", "data_hash", "environment", "seeds", "pipeline_version"}
    have = set(ce["lineage"]) | ({"pipeline_version"}
                                 if ce["pipeline_version"] else set())
    missing = need - have
    results.append(("lineage_complete", not missing,
                    f"missing: {sorted(missing)}" if missing else "complete"))

    return results


def report(results, label):
    passed = all(r[1] for r in results)
    print(f"\n{label}: {'PROMOTE' if passed else 'BLOCKED'}")
    for name, ok, reason in results:
        print(f"    {'PASS' if ok else 'FAIL'}  {name:<28} {reason}")
    return passed


# --- an incumbent and four candidates, each with one realistic problem ------
FLOORS = {"auc": 0.78, "slice_auc": 0.70, "ece": 0.05}
BUDGET_MS = 50.0

incumbent = (RegistryEntry("churn", 6).set(
    stage="production", pipeline_version=12, schema=["a", "b", "c", "d"],
    lineage={"commit": "aa11", "data_hash": "d1", "environment": "e1",
             "seeds": {"split": 7}},
    eval={"auc": 0.8120, "ece": 0.021, "pred_mean": 0.140,
          "p99_latency_ms": 31.0,
          "slices": {"new_customer": 0.744, "tenured": 0.828,
                     "high_value": 0.791}}))

candidates = {
    "v7  clean improvement": RegistryEntry("churn", 7).set(
        pipeline_version=12, schema=["a", "b", "c", "d"],
        lineage={"commit": "bb22", "data_hash": "d2", "environment": "e1",
                 "seeds": {"split": 7}},
        eval={"auc": 0.8240, "ece": 0.024, "pred_mean": 0.146,
              "p99_latency_ms": 33.0,
              "slices": {"new_customer": 0.762, "tenured": 0.839,
                         "high_value": 0.804}}),

    "v8  better overall, broke a slice": RegistryEntry("churn", 8).set(
        pipeline_version=12, schema=["a", "b", "c", "d"],
        lineage={"commit": "cc33", "data_hash": "d3", "environment": "e1",
                 "seeds": {"split": 7}},
        eval={"auc": 0.8310, "ece": 0.027, "pred_mean": 0.151,
              "p99_latency_ms": 35.0,
              "slices": {"new_customer": 0.661, "tenured": 0.858,
                         "high_value": 0.842}}),

    "v9  accurate but uncalibrated and slow": RegistryEntry("churn", 9).set(
        pipeline_version=12, schema=["a", "b", "c", "d"],
        lineage={"commit": "dd44", "data_hash": "d4", "environment": "e1",
                 "seeds": {"split": 7}},
        eval={"auc": 0.8390, "ece": 0.118, "pred_mean": 0.262,
              "p99_latency_ms": 96.0,
              "slices": {"new_customer": 0.771, "tenured": 0.851,
                         "high_value": 0.826}}),

    "v10 retrain on fresh data, same score": RegistryEntry("churn", 10).set(
        pipeline_version=13, schema=["a", "b", "c", "d", "e"],
        lineage={"commit": "ee55", "data_hash": "d5", "environment": "e1"},
        eval={"auc": 0.8102, "ece": 0.020, "pred_mean": 0.139,
              "p99_latency_ms": 30.0,
              "slices": {"new_customer": 0.741, "tenured": 0.826,
                         "high_value": 0.788}}),
}

# the margin comes from the metric's own noise (eq. 47.6), not from taste
SIGMA_RUN, SIGMA_EVAL = 0.0021, 0.0043
MARGIN = 2 * float(np.sqrt(SIGMA_RUN ** 2 + SIGMA_EVAL ** 2))

print("=" * 72)
print("the promotion gate, against four realistic candidates")
print("=" * 72)
print(f"non-inferiority margin = 2 * sqrt(sigma_run^2 + sigma_eval^2)")
print(f"                       = 2 * sqrt({SIGMA_RUN}^2 + {SIGMA_EVAL}^2) "
      f"= {MARGIN:.4f}")
print("  -> derived from measured noise (eq. 47.6), not chosen as a round")
print("     number. A margin below the noise makes the gate flaky, which is")
print("     worse than having no gate at all.")

for label, cand in candidates.items():
    res = gate(cand, incumbent, margin=MARGIN, floors=FLOORS,
               latency_budget_ms=BUDGET_MS)
    report(res, label)

print("\n" + "=" * 72)
print("what each candidate teaches")
print("=" * 72)
lessons = [
    ("v7", "passes everything. This is the boring case, and most releases "
           "should look like it."),
    ("v8", "AUC improved by 1.9 points and the new-customer slice fell 8.3. "
           "An aggregate-only gate would have shipped it."),
    ("v9", "the most accurate candidate of the four, and unshippable: its "
           "probabilities are meaningless and it misses the latency budget."),
    ("v10", "no better than the incumbent, and that is FINE — it is a "
            "retrain on fresher data. It is blocked for a different reason: "
            "the pipeline and schema changed without approval, and its "
            "lineage is incomplete."),
]
for v, why in lessons:
    print(f"  {v:<5} {why}")

print("\nNote which checks did the work. The metric floors caught nothing")
print("that a human would not have caught; the SLICE floor, the calibration")
print("check, the latency budget and the schema check caught things a review")
print("meeting reliably misses, because they require looking at numbers")
print("nobody thinks to ask for.")
