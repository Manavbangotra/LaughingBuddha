---
id: mle-registry
number: 47
part: V
tier: focused
status: reviewed
requires: [mle-reproducibility, mle-pipelines, ml-metrics]
provides: [model-registry, promotion-gate, model-card, shadow-deployment,
           canary-release, ml-test-score, deployment-handoff, rollback]
citations: [breck2017, sculley2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Specify what a registry entry must contain for someone who did not train
   the model to deploy it safely.
2. Implement a promotion gate as executable policy rather than a review
   meeting.
3. Explain the ML Test Score and why its aggregate is a minimum rather than a
   sum.
4. Distinguish shadow deployment, canary release and A/B testing, and say what
   each can and cannot establish.
5. Design a rollback that works, including for the cases where it does not.
6. Write a model card that is useful rather than ceremonial.
7. Explain why the artefact is the model *plus its pipeline*.

## 2. Why This Matters

This is the chapter about a boundary rather than a technique, and boundaries
are where systems fail.

**The person deploying the model is usually not the person who trained it**,
and often not the same person six months later. Everything they need must be in
the artefact, because the alternative is a conversation that will not happen.
A registry is the mechanism for making tacit knowledge explicit before it
evaporates.

**"Approved" has to mean something checkable.** A review meeting that ends in
approval is not a gate; it is an opinion with a timestamp. {{cite:breck2017}}
made production readiness a scored rubric precisely so the question has an
answer that survives the meeting, and the most useful property of that rubric
is that it is diagnostic — a low score says *which* area is neglected.

**Rollback is the only safety property you actually get.** Everything upstream
reduces the probability of a bad deployment; rollback bounds its duration. A
system that cannot roll back in minutes has no upper bound on the cost of being
wrong, and models fail in ways tests do not catch.

## 3. Prerequisites

{{ch:mle-reproducibility}} for the lineage a registry entry carries — an entry
without it is a binary with no provenance. {{ch:mle-pipelines}} for why the
pipeline is part of the artefact. {{ch:ml-metrics}} for the thresholds a gate
enforces and for slice-based evaluation.

## 4. Intuitive Explanation

### 4.1 The handoff problem

```text
   TRAINING SIDE                       SERVING SIDE
   ─────────────────                   ──────────────────
   knows: what it was trained on       knows: latency budget
          which slices are weak               traffic mix
          what the threshold means            rollback procedure
          what it must NOT be used for
                    │
                    ▼
            ┌───────────────┐
            │   REGISTRY    │   everything the serving side needs
            │  entry v7     │   must be written down here
            └───────────────┘
```

The failure is not that people are careless. It is that the training side knows
a dozen things that feel obvious at the time — the threshold was chosen for a
particular cost ratio, the model is weak on new customers, it expects the
feature pipeline at version 12 — and none of them is obvious to anyone else, or
to the same person a year later.

### 4.2 The artefact is not the weights

A serialised model is not deployable on its own. The deployable unit is:

- the model weights;
- the **exact feature pipeline** that produced its inputs
  ({{ch:mle-pipelines}});
- the input schema, with types and ranges;
- the output contract — what the number means, and its calibration;
- the decision threshold and the reasoning behind it;
- the lineage ({{ch:mle-reproducibility}});
- the evaluation, disaggregated by slice;
- the known limitations.

Ship the weights alone and you have shipped a function whose domain is
undocumented. The commonest production incident in this category is a model
served with a pipeline one version out of step — everything runs, and the
inputs mean something slightly different from what the model was trained on.

### 4.3 A gate is code

Compare two ways of deciding whether a model may ship:

```text
   A MEETING                          A GATE
   ───────────────────                ─────────────────────────────
   "looks good to me"                 assert auc >= 0.78
   "did we check the new segment?"    assert min(slice_aucs) >= 0.70
   "how fast is it?"                  assert p99_latency_ms <= 50
   "is it better than the old one?"   assert auc >= incumbent_auc - 0.005
                                      assert schema == expected_schema
                                      assert lineage_complete()

   outcome: an opinion                outcome: a record with reasons
```

The gate is not better because it is stricter. It is better because it is
**the same every time**, it runs when nobody is watching, and its failures
carry a reason. A meeting produces a decision; a gate produces evidence.

The thresholds still require judgement — someone must decide that 0.78 is the
floor — but that judgement is made once, in the open, and is then applied
consistently.

### 4.4 Three ways to try a model on real traffic

Routinely confused, and they answer different questions:

**Shadow.** The candidate scores real traffic; its output is discarded.
Answers: does it run, how fast, and how does its prediction distribution
compare? Cannot answer: is it better, because nothing acts on it.

**Canary.** The candidate serves a small fraction of traffic for real.
Answers: does it break anything in production. Cannot answer: is it better,
usually, because the slice is too small to resolve a metric difference.

**A/B test.** Randomised assignment with enough traffic and time to measure an
outcome. Answers: is it better. Costs: exposure of real users to the worse arm,
and time — often weeks, because the outcome that matters is delayed.

The sequence is not optional-in-any-order: shadow catches crashes and latency
cheaply, canary catches integration failures, and only the A/B test measures
value. Skipping to the A/B test means discovering an integration bug with real
traffic on it.

## 5. Formal Explanation

### 5.1 The registry entry

$$
\mathcal{E} = \langle
  \text{artefact},\; \text{pipeline},\; \text{schema},\; \mathcal{R},\;
  \text{eval},\; \text{contract},\; \text{stage},\; \text{audit}
\rangle
$$ (eq:registry-entry)

with $\mathcal{R}$ the run record of {{eq:run-tuple}}. Two fields deserve
expansion because they are the ones usually thin.

**Evaluation must be disaggregated.** A single aggregate number hides exactly
the failure that will hurt: the model is fine overall and poor on the segment
that generates the complaints. The entry should carry per-slice metrics for
every slice anyone will later ask about — and deciding that list in advance is
part of the work.

**The contract states what the output means.** Is it a calibrated probability
or a score? What threshold, and derived from what cost ratio
({{ch:ml-logistic}})? What is the expected input rate, and what should the
caller do on a timeout?

### 5.2 Stages and transitions

A registry's states are a small state machine, and the value is in the
transitions being recorded rather than in the names.

```text
   registered ──gate──▶ staging ──gate+manual──▶ production ──▶ archived
        │                  │                          │
        └──────────────────┴───── rejected ◀──────────┘
```

Every transition records who or what triggered it, which gate results
justified it, and when. That log is what makes "why is this model live?" an
answerable question — and it is the artefact an auditor actually wants.

### 5.3 The ML Test Score

{{cite:breck2017}} defines twenty-eight tests in four categories: features and
data, model development, ML infrastructure, and monitoring. Each scores 0
(not done), 0.5 (documented but manual) or 1 (automated and repeated).

The scoring rule is the interesting part:

$$
\text{score} = \min_{c \in \text{categories}} \sum_{t \in c} s_t
$$ (eq:ml-test-score)

**The minimum across categories, not the sum.** That choice encodes the paper's
central claim: a system with impeccable model development and no monitoring is
not a well-tested system, it is an untested system with a good model in it.
Excellence in one area cannot compensate for absence in another, because the
failure modes are independent.

The empirical regularity is that teams score highest on model development —
the part that resembles research — and lowest on monitoring, which is why the
minimum bites there.

### 5.4 The promotion gate

A gate is a predicate over the candidate, the incumbent and the entry:

$$
\text{promote} \iff
  \bigwedge_{i} g_i(\text{candidate},\; \text{incumbent},\; \mathcal{E})
$$ (eq:promotion-gate)

The checks worth having, and what each catches:

{#tbl:gate-checks caption="Promotion-gate checks. The first three are about the model being good; the rest are about it being deployable, which is where gates usually find problems."}

| Check | Catches |
|---|---|
| Aggregate metric floor | an outright regression |
| Per-slice metric floors | a model that improved on average and broke a segment |
| Non-inferiority to incumbent | a change that is not an improvement |
| Calibration | a model whose probabilities stopped meaning anything |
| Prediction-distribution shift vs incumbent | a change too large to be a refinement |
| Latency at p99 | a model that cannot be served |
| Schema conformance | a pipeline/model version mismatch |
| Lineage completeness | an artefact that cannot be rebuilt |

> IMPORTANT: **Non-inferiority, not superiority.** Requiring a candidate to
> beat the incumbent sounds right and blocks legitimate changes — a
> simplification, a dependency removal, a retrain on fresher data with the same
> score. The useful gate is "not meaningfully worse, on any slice", with the
> margin set by the metric's own standard error rather than by taste. This is
> {{ch:mle-reproducibility}}'s run-to-run variance being used for something.

### 5.5 Rollback

The property to design for, because it bounds the damage.

**Keep the previous artefact warm.** Rollback is then a routing change of
seconds, not a rebuild of minutes.

**Roll back the pipeline with the model.** They are one unit; reverting the
model alone recreates the version mismatch of
{{sec:4-intuitive-explanation}}.

**Know what cannot be rolled back.** This is the part usually missed. If the
model's outputs were written somewhere — decisions taken, emails sent, prices
set, a recommendation feedback loop trained on ({{ch:ds-recsys}}) — then
reverting the model does not revert its effects. For those systems the gate
must be stricter, because rollback is not a safety net.

## 6. Mathematical Foundation

### 6.1 Why the minimum, not the sum

Model a system as failing if any of $k$ independent areas fails, with area $i$
having failure probability $p_i$ decreasing in its test coverage $s_i$:

$$
\Prob(\text{system fails}) = 1 - \prod_{i=1}^{k}(1 - p_i(s_i))
$$ (eq:system-failure)

Take logs: $\log \Prob(\text{no failure}) = \sum_i \log(1 - p_i)$. For small
$p_i$ this is approximately $-\sum_i p_i$, so system reliability is dominated
by the **largest** $p_i$ — the least-tested area.

Increasing $s$ in an already-well-tested area reduces an already-small $p_i$
and barely moves the total. A sum-based score would reward that; the minimum
does not. {{eq:ml-test-score}} is therefore the scoring rule matching the
failure model, which is a nicer justification than "it makes people do
monitoring".

### 6.2 What a canary can and cannot detect

A canary serving fraction $f$ of $N$ daily requests observes $fN$ outcomes. To
detect a degradation of size $\delta$ in a metric with per-observation standard
deviation $\sigma$, at 80% power and 5% significance, the requirement from
{{ch:math-inference}} is approximately

$$
fN \ge \frac{2(z_{0.975} + z_{0.80})^{2}\sigma^{2}}{\delta^{2}}
 \approx \frac{15.7\,\sigma^{2}}{\delta^{2}}
$$ (eq:canary-power)

At $f = 0.01$, $N = 10^{6}$ requests per day, a conversion metric with
$\sigma \approx 0.3$, and a degradation of one percentage point
($\delta = 0.01$): the requirement is about 14,000 observations against 10,000
available, so a full day of canary is marginal for a one-point effect and
hopeless for a smaller one.

Two conclusions, and they are the practical content:

**A canary detects breakage, not degradation.** Errors, timeouts and crashes
have huge effect sizes and appear in minutes. A one-percent metric regression
does not, and pretending the canary is checking for it is the mistake.

**The metric you monitor during a canary should be one with a large effect
size** — error rate, latency, prediction distribution — not the business
outcome, which is both delayed and low-powered at that traffic share.

### 6.3 Setting the non-inferiority margin

The margin should come from the metric's own noise, not from a round number.
With run-to-run standard deviation $\sigma_{\text{run}}$
({{ch:mle-reproducibility}}) and evaluation standard error
$\sigma_{\text{eval}}$, the total is

$$
\sigma_{\text{total}} = \sqrt{\sigma_{\text{run}}^{2}
                            + \sigma_{\text{eval}}^{2}}
$$ (eq:margin-noise)

and a margin of $2\sigma_{\text{total}}$ blocks changes that are
distinguishably worse while admitting those that are not.

Setting the margin below $\sigma_{\text{total}}$ produces a gate that fails
randomly, which is worse than no gate: teams learn to re-run it until it
passes, and the gate stops carrying information. **A flaky gate is
actively harmful**, because it trains people to ignore gates.

## 7. Implementation

```python {tier=A name=registry-and-gate}
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
```

```python {tier=A name=ml-test-score}
"""The ML Test Score (eq. 47.3), and why the aggregate is a minimum.
"""
import numpy as np

# --- the rubric, abbreviated to the checks that carry the most weight -------
RUBRIC = {
    "data": [
        "feature expectations are captured in a schema",
        "all features are beneficial (tested, not assumed)",
        "no feature costs more than it is worth",
        "features comply with policy and privacy requirements",
        "the data pipeline has appropriate privacy controls",
        "new features can be added quickly",
        "all input feature code is tested",
    ],
    "model": [
        "model specs are reviewed and version-controlled",
        "offline and online metrics correlate",
        "all hyperparameters have been tuned",
        "the impact of model staleness is known",
        "a simpler model is not better",
        "model quality is sufficient on all important data slices",
        "the model has been tested for inclusion (fairness)",
    ],
    "infrastructure": [
        "training is reproducible",
        "model specs are unit tested",
        "the full ML pipeline is integration tested",
        "model quality is validated before serving",
        "the model allows debugging by observing a single example",
        "models are canaried before serving",
        "serving models can be rolled back",
    ],
    "monitoring": [
        "dependency changes result in notification",
        "data invariants hold in training and serving",
        "training and serving features compute the same values",
        "models are not too stale",
        "the model is numerically stable",
        "compute performance has not regressed",
        "prediction quality has not regressed",
    ],
}


def score(answers):
    """0 = not done, 0.5 = documented but manual, 1 = automated (eq. 47.3)."""
    per_cat = {c: sum(answers[c]) for c in RUBRIC}
    return per_cat, min(per_cat.values())


def band(total):
    if total < 1:
        return "more of a research project than a product"
    if total < 3:
        return "not enough testing"
    if total < 5:
        return "reasonable, but investment warranted"
    if total < 7:
        return "strong levels of automated testing"
    return "exceptional"


# --- three teams, all of whom believe they test well ------------------------
teams = {
    "the research-strong team": {
        "data": [1, 0.5, 0, 0.5, 0.5, 1, 0.5],
        "model": [1, 0.5, 1, 0.5, 1, 1, 0.5],
        "infrastructure": [1, 0.5, 0.5, 0.5, 0.5, 0, 0.5],
        "monitoring": [0, 0.5, 0, 0, 0, 0.5, 0],
    },
    "the platform-strong team": {
        "data": [1, 0, 0, 1, 1, 1, 1],
        "model": [0.5, 0, 0.5, 0, 0, 0, 0],
        "infrastructure": [1, 1, 1, 1, 0.5, 1, 1],
        "monitoring": [1, 1, 0.5, 1, 0.5, 1, 0.5],
    },
    "the balanced team": {
        "data": [1, 0.5, 0.5, 1, 1, 0.5, 1],
        "model": [1, 0.5, 1, 0.5, 0.5, 1, 0.5],
        "infrastructure": [1, 0.5, 1, 1, 0.5, 1, 1],
        "monitoring": [0.5, 1, 1, 0.5, 0.5, 1, 0.5],
    },
}

print("=" * 72)
print("the ML Test Score: minimum across categories, not sum (eq. 47.3)")
print("=" * 72)
print(f"{'team':<28} " + " ".join(f"{c[:6]:>7}" for c in RUBRIC) +
      f" {'SUM':>6} {'SCORE':>7}")
for name, ans in teams.items():
    per_cat, total = score(ans)
    print(f"{name:<28} " + " ".join(f"{per_cat[c]:>7.1f}" for c in RUBRIC) +
          f" {sum(per_cat.values()):>6.1f} {total:>7.1f}")

print(f"\n{'team':<28} {'verdict':<44}")
for name, ans in teams.items():
    _, total = score(ans)
    print(f"{name:<28} {band(total):<44}")

print("\nThe first two teams differ by four points of SUM — 14 against 18,")
print("a 29% gap that a sum-based rubric would treat as meaningful — and")
print("score IDENTICALLY at 1.0. The score is right and the sum is not:")
print("both have one nearly-empty category, and it makes no difference to")
print("their reliability which category it is.")
print("\nThe research-strong team has excellent model development and")
print("almost no monitoring, so it will fail silently. The platform-strong")
print("team has excellent infrastructure and almost no model validation, so")
print("it will deploy a bad model quickly and reliably. Neither is a")
print("well-tested system, and only the minimum says so.")
print("\nSection 6.1 is why the minimum is the right rule rather than a")
print("rhetorical one: if a system fails when ANY area fails, total")
print("reliability is dominated by the weakest area, and improving an")
print("already-strong one barely moves it.")

# --- and the arithmetic behind that (section 6.1) ---------------------------
print("\n" + "=" * 72)
print("why the minimum, quantified (eq. 47.5)")
print("=" * 72)
print("Model each category's failure probability as falling with its score:")
print("  p_i = 0.30 * exp(-0.45 * s_i)\n")


def p_fail(s):
    return 0.30 * np.exp(-0.45 * s)


print(f"{'team':<28} " + " ".join(f"{'p(' + c[:4] + ')':>9}" for c in RUBRIC)
      + f" {'p(system)':>11}")
for name, ans in teams.items():
    per_cat, _ = score(ans)
    ps = [p_fail(per_cat[c]) for c in RUBRIC]
    p_sys = 1 - np.prod([1 - p for p in ps])
    print(f"{name:<28} " + " ".join(f"{p:>9.3f}" for p in ps) +
          f" {p_sys:>11.3f}")

print("\nNow the counterfactual that makes the point. Take the")
print("research-strong team and give it one more point of testing, spent in")
print("two different places:\n")
base = teams["the research-strong team"]
per_cat, _ = score(base)
p_sys_base = 1 - np.prod([1 - p_fail(per_cat[c]) for c in RUBRIC])
print(f"{'where the point is spent':<34} {'p(system fails)':>17} "
      f"{'improvement':>13}")
print(f"{'nowhere (baseline)':<34} {p_sys_base:>17.4f} {'-':>13}")
for target in ("model", "monitoring"):
    pc = dict(per_cat)
    pc[target] += 1.0
    p_sys = 1 - np.prod([1 - p_fail(pc[c]) for c in RUBRIC])
    print(f"{'into ' + target + ' (already ' + f'{per_cat[target]:.1f}' + ')':<34} "
          f"{p_sys:>17.4f} {p_sys_base - p_sys:>13.4f}")

print("\nThe same unit of effort buys several times more reliability when")
print("spent on the weakest category. A sum-based score is indifferent")
print("between the two; the minimum directs the effort where eq. 47.5 says")
print("it pays. That is the whole argument, and it is why the rubric's most")
print("useful output is not the number but WHICH category produced it.")
```

## 8. Practical Example

```python {tier=A name=shadow-canary-rollback}
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
```

## 9. Common Mistakes

**Shipping the weights without the pipeline.** They are one deployable unit;
a version mismatch runs cleanly and means something different.

**Requiring superiority rather than non-inferiority.** It blocks legitimate
retrains and simplifications.

**Setting a margin below the measured noise.** The gate becomes flaky, and a
flaky gate trains people to ignore gates.

**Gating on the aggregate metric only.** The measured `v8` candidate improved
by 1.9 points overall and lost 8.3 on a slice.

**Treating a canary as a metric test.** The power calculation shows a
one-point conversion regression needs most of a day at 1% of traffic.

**A review meeting instead of a gate.** It produces an opinion, not evidence,
and it is not the same twice.

**No warm previous artefact.** The measured incident cost is dominated by
rollback time when detection is fast.

**Assuming rollback undoes the damage.** Declines, emails, prices and logged
training data are not reverted.

**Optimising the ML Test Score category you are already good at.** The
measured counterfactual shows the same effort buying several times more
reliability in the weakest category.

## 10. Connection to Previous Chapters

{{ch:mle-reproducibility}} supplied the lineage that {{eq:registry-entry}}
carries, and the run-to-run variance that {{eq:margin-noise}} turns into a
non-inferiority margin — that measurement finally has an operational use here.
{{ch:mle-pipelines}} established that the pipeline is part of the artefact,
which the schema check enforces. {{ch:ml-metrics}} supplied the slice-based
evaluation and the calibration check that caught the measured `v9`.
{{ch:ml-logistic}} supplied the cost-derived threshold the contract must state.
{{ch:math-inference}} supplied the power calculation of {{eq:canary-power}}.

Forward: {{ch:mle-drift}} is the monitoring category that the measured ML Test
Score identifies as most teams' weakest, and the detection time that dominates
the measured incident cost. {{ch:ds-recsys}}'s feedback loop is why some
deployments cannot be rolled back. {{part:24}} covers the serving platform;
{{ch:rai-regulation}} returns to model cards and documentation as a governance
obligation rather than an engineering one.

## 11. Exercises

**Beginner**

1. Name six things a registry entry must contain besides the weights.
2. Why is the deployable unit the model plus its pipeline?
3. Distinguish shadow, canary and A/B testing in one sentence each.
4. Why non-inferiority rather than superiority?
5. What does a model card tell a reader that a metric does not?

**Intermediate**

6. Explain why {{eq:ml-test-score}} takes a minimum rather than a sum.
7. Using {{eq:canary-power}}, compute the observations needed to detect a
   0.5-point conversion change with $\sigma = 0.3$.
8. Why does a margin below $\sigma_{\text{total}}$ make a gate harmful?
9. Give three cases where rollback does not undo the damage.
10. Which gate checks would have caught the measured `v9` candidate?
11. Why should a gate report all failures rather than the first?

**Advanced**

12. Derive {{eq:system-failure}}'s implication that the weakest category
    dominates, and state where the independence assumption fails.
13. Derive {{eq:canary-power}} from the two-sample test of
    {{ch:math-inference}}.
14. Design a promotion gate for a model whose outputs become training data,
    given that rollback is ineffective.
15. Propose a scoring rule better than {{eq:ml-test-score}} for a system whose
    categories are *not* independent, and justify it.

**Implementation**

16. Extend the gate with a fairness check across protected slices and decide
    what the failure mode should be.
17. Implement automatic rollback triggered by a monitored error rate, with
    hysteresis so it cannot oscillate.
18. Build a registry that refuses to accept an entry whose lineage does not
    resolve, using {{ch:mle-reproducibility}}'s verifier.
19. Score your own project against the twenty-eight tests and identify the
    binding category.

**Reasoning**

20. A candidate is 3 points better overall and 6 points worse on a slice
    representing 4% of traffic. What do you do, and what do you need to know?
21. Your gate has failed spuriously three times this month. What is the
    problem, and what do you change?

## 12. Chapter Summary

A registry exists because the person deploying a model is not the person who
trained it. The entry must carry the artefact, the exact pipeline, the schema,
the lineage, disaggregated evaluation, the output contract, the stage, and an
audit trail of transitions. The deployable unit is the model *plus its
pipeline*; shipping the weights alone ships a function with an undocumented
domain.

A promotion gate is executable policy, and its value is that it is the same
every time and produces evidence rather than an opinion. The measured example
shows which checks earn their place: aggregate metric floors caught nothing a
human would have missed, while the per-slice floor caught a candidate that
improved 1.9 points overall and lost 8.3 on new customers, the calibration and
latency checks disqualified the most accurate candidate of the four, and the
schema check caught a pipeline mismatch.

The margin for non-inferiority should come from measured noise —
$2\sqrt{\sigma_{\text{run}}^{2} + \sigma_{\text{eval}}^{2}}$ — not from a round
number. A margin below the noise makes the gate flaky, and a flaky gate is
worse than none because it teaches people to ignore gates.

The ML Test Score aggregates as a **minimum** across four categories, and
{{eq:system-failure}} justifies that: if a system fails when any area fails,
reliability is dominated by the weakest area. The measured counterfactual makes
it concrete — the same unit of testing effort buys several times more
reliability spent on the weakest category than on the strongest — and the
measured teams show the research-strong one having the second-highest sum and
the lowest score, correctly identified as a system that will fail silently.

Shadow, canary and A/B answer different questions. The power calculation is the
key result: breakage has a large effect size and is detectable in minutes at 1%
of traffic, while a one-point conversion regression needs most of a day and a
fifth of a point needs weeks. **A canary detects breakage, not degradation**,
and watching a business metric on a small canary for an hour is not evidence.

Rollback is the only property that bounds the cost of being wrong, because cost
is proportional to duration. Keeping the previous artefact warm converts a
half-hour incident into a five-minute one. But where the model's outputs have
already caused effects — a decline, an email, a price, or a logged row that
becomes training data — rollback does not undo them, and the gate must carry
the weight instead.
