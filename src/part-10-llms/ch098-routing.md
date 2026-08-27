---
id: llm-routing
number: 98
part: X
tier: full
status: draft
requires: [llm-next-token, llm-inference, llm-hallucination, llm-long-context,
           nlp-similarity, ml-metrics, fm-distillation]
provides: [model-routing, cascade, difficulty-estimation, deferral,
           cost-quality-frontier, router-training, escalation-policy,
           routing-overhead]
citations: [kadavath2022, guo2017calibration, sanh2019, touvron2023llama,
            hoffmann2022chinchilla, ji2023survey, liu2023lost, brown2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish a cascade from a router and say when each applies.
2. Derive the cost–quality frontier and locate a system on it.
3. Compute the break-even accuracy at which escalation pays.
4. Explain why a cascade's overhead can exceed its saving, and find the
   threshold where it does.
5. Build a difficulty estimator and evaluate it on the right metric.
6. Explain why routing signals degrade and what to monitor.
7. Choose between routing, distillation, and doing neither.

## 2. Why This Matters

**This is the third appearance of one architecture**, and naming the pattern is
the chapter's main contribution. {{ch:nlp-similarity}}'s retrieve-then-rerank,
{{ch:nlp-extraction}}'s encoder-then-LLM cascade, and this chapter's
small-then-large routing are the same shape: **a cheap high-recall stage in
front of an expensive precise one, with the split point set by arithmetic.**

**The economics are usually decisive.** {{ch:llm-inference}} showed inference
cost scaling with $N$ and paid per request forever. A router that sends 70% of
traffic to a model an order of magnitude cheaper changes the unit economics of a
product, and it does so without touching quality on the 30% that matters.

**And the naive version does not work.** A cascade that runs the small model
first and escalates on low confidence pays for *both* models on escalated
requests, so if escalation is frequent it costs more than always using the large
one. There is a break-even and it is computable.

**The deciding input is a confidence signal**, and
{{ch:llm-next-token}} established what that is worth after alignment: usable as
a *rank*, not as a probability. That constrains how a threshold can be set and
is why this chapter's policies are traced empirically rather than derived.

## 3. Prerequisites

{{ch:llm-next-token}} for calibration and the rank/probability distinction —
this chapter depends on it heavily. {{ch:llm-inference}} for the cost model.
{{ch:llm-hallucination}} for abstention, which is routing with a null target.
{{ch:nlp-similarity}} for the cascade pattern's first appearance.
{{ch:ml-metrics}} for the frontier. {{ch:fm-distillation}} for the alternative:
making the cheap model better rather than routing around it.

## 4. Intuitive Explanation

Most requests are easy. "What is 2+2", "summarise this paragraph", "is this
sentiment positive" — a small model answers all three correctly, at a fraction
of the cost. A minority are hard, and those need the expensive model.

**If you could tell them apart in advance, you would save most of your bill.**
That is routing, and the whole difficulty is the "in advance".

**Two architectures, and they differ in when the decision is made.**

A **router** decides *before* generating: look at the request, predict which
model can handle it, send it there. One model runs. Cheap, and the prediction is
made without seeing any answer.

A **cascade** decides *after*: run the small model, examine its answer and its
confidence, and escalate if unsatisfactory. The decision is much better informed
— you have seen an attempt — and on escalation you have paid for two models.

> NOTE: The cascade's cost structure is the thing people get wrong. Escalated
> requests cost small-plus-large, not large. If escalation is rare the overhead
> is negligible; if it is common the cascade is *more* expensive than always
> using the large model. `cascade-breakeven` computes exactly where that
> crossover sits, and it is closer than intuition suggests.

**What signal decides?** For a cascade, the small model's own confidence
({{ch:llm-next-token}}) — which survives alignment as a rank, so a threshold can
be *found* but not *computed*. For a router, a separate classifier trained on
which model got which question right.

**And the router has an awkward property**: it needs training data consisting of
questions labelled with which model answers them correctly, which requires
running both models on a sample. **The router's training set is exactly the
experiment you were trying to avoid**, though you only need it once.

**Where abstention fits.** {{ch:llm-hallucination}}'s abstention is routing with
a null destination — instead of escalating to a bigger model you escalate to a
human, or decline. The machinery is identical and only the target differs, which
is why the risk–coverage curve reappears here as a cost–quality frontier.

**The mental model:** routing buys the average cost of the cheap model and the
worst-case quality of the routing decision. Where it breaks down: the routing
decision is itself a model, it can be wrong, and its errors are concentrated on
exactly the hard cases you built it to catch.

## 5. Formal Explanation

### 5.1 The two architectures

**Router.** A decision function $\rho: x \to \{1,\dots,M\}$ chosen before
generation:

$$
\E[\text{cost}] = \sum_{m} \Prob[\rho(x)=m]\,c_m,
\qquad
\E[\text{quality}] = \sum_{m} \Prob[\rho(x)=m]\,q_m(x)
$$ (eq:router-cost)

**Cascade.** Run model 1; escalate on a signal $\kappa$ below threshold $\tau$:

$$
\E[\text{cost}] = c_1 + \Prob[\kappa < \tau]\,c_2
$$ (eq:cascade-cost)

$\square$

**Note $c_1$ is paid unconditionally.** That single term is the whole difference
in cost structure, and it is why a cascade with high escalation is worse than no
cascade at all.

### 5.2 Where a cascade stops paying

A cascade beats always-large when

$$
c_1 + e\,c_2 < c_2
\iff e < 1 - \frac{c_1}{c_2}
$$ (eq:cascade-breakeven)

for escalation rate $e$.

$\square$

**With $c_2/c_1 = 10$, escalation must stay below 90%.** With
$c_2/c_1 = 2$ — a small and a medium model — it must stay below 50%, which is a
real constraint. **The cheaper the small model relative to the large one, the
more escalation a cascade tolerates**, which argues for a large ratio and
therefore for a genuinely small first stage.

### 5.3 The quality side

Cascade quality is

$$
q = \Prob[\kappa \ge \tau]\,q_1^{+} + \Prob[\kappa < \tau]\,q_2
$$ (eq:cascade-quality)

where $q_1^{+}$ is the small model's accuracy *on requests it kept* — higher
than its overall accuracy if $\kappa$ ranks well.

**The cascade can match the large model's quality** when $q_1^{+} \approx q_2$
on kept requests, which is exactly the condition that $\kappa$ separates the
cases the small model gets right from those it does not. **That is a statement
about the signal, not about either model.**

### 5.4 Break-even for escalation on a single request

Escalating is worth it when the expected quality gain exceeds the cost:

$$
\big(q_2 - q_1(x)\big)\,V > c_2
$$ (eq:escalation-value)

with $V$ the value of a correct answer. Rearranged:

$$
q_1(x) < q_2 - \frac{c_2}{V}
$$ (eq:escalation-threshold)

$\square$

**The threshold depends on the value of correctness**, which is a product input.
A high-value application escalates aggressively; a low-value one barely
escalates at all — and the same system with the same models has different
optimal thresholds for different features.

### 5.5 Routing overhead

A router is itself a computation. If it costs $c_r$:

$$
\E[\text{cost}] = c_r + \sum_m \Prob[\rho(x)=m]\,c_m
$$ (eq:router-overhead)

> IMPORTANT: A router implemented as an LLM call is frequently as expensive as
> the small model it routes to, which destroys the saving. **The router must be
> cheap** — an embedding model and a linear classifier
> ({{ch:nlp-similarity}}), or a small fine-tuned encoder
> ({{ch:nlp-bert}}) — and this is where a great many routing projects fail.

## 6. Mathematical Foundation

### 6.1 The cost–quality frontier

Sweeping $\tau$ traces a curve. At $\tau = -\infty$: cost $c_1$, quality $q_1$.
At $\tau = +\infty$: cost $c_1 + c_2$, quality $q_2$.

The achievable set is the convex hull of these points and everything the signal
permits between them. **A useless signal gives a straight line** between the two
endpoints — you may as well flip a weighted coin — **and a good signal bows the
curve upward**, giving more quality per unit cost at every intermediate point.

$$
\text{signal value} = \int_0^1 \big(q_{\text{cascade}}(e) - q_{\text{linear}}(e)\big)\,\dd e
$$ (eq:signal-value)

$\square$

This is {{eq:risk-coverage}} from {{ch:llm-hallucination}} in different units —
**abstention and routing trace the same curve**, with cost on one axis instead of
coverage.

### 6.2 Why routing errors concentrate on hard cases

The router predicts whether the small model will succeed. Its own accuracy is
highest where the answer is obvious — very easy and very hard requests — and
lowest in the middle, where the small model *might* succeed.

Formally, if the router estimates $\hat{p}(x) \approx \Prob[\text{model 1
correct}]$, its errors are largest where $\hat{p}\approx 0.5$, which is exactly
the region where the routing decision matters most.

$\square$

**A router is least reliable precisely where it is most needed**, which bounds
how good routing can get and argues for cascades where the extra information —
an actual attempt — is available.

### 6.3 A worked cascade calculation

A small model at $c_1 = 1$ with accuracy 0.78; a large model at $c_2 = 12$ with
accuracy 0.94. A confidence signal with AUC 0.82.

**Always-large:** cost 12, accuracy 0.94.

**Cascade at 30% escalation:** cost $1 + 0.3(12) = 4.6$. The small model keeps
70% of requests; if the signal ranks well its accuracy on those is about 0.90,
so

$$
q = 0.7(0.90) + 0.3(0.94) = 0.912
$$

**97% of the quality at 38% of the cost.** And {{eq:cascade-breakeven}} says
escalation could rise to $1 - 1/12 = 92\%$ before the cascade loses to
always-large, so there is plenty of headroom.

**Now with a 2x model.** $c_1=1$, $c_2=2$: break-even escalation is 50%, and at
30% escalation the cost is $1 + 0.6 = 1.6$ against 2 — a saving of 20% rather
than 62%. **The saving scales with the cost ratio**, which is the argument for
routing between genuinely different model sizes rather than adjacent ones.

## 7. Internal Mechanics

```mermaid {#fig:routing-architectures caption="Router against cascade. The router decides before seeing an answer and runs one model; the cascade decides after and pays for both on escalation. The difference in cost structure is eq:router-cost against eq:cascade-cost."}
graph TD
  A["request"] --> B{"router<br/>predicts difficulty"}
  B -- easy --> C["small model"]
  B -- hard --> D["large model"]
  C --> E["answer"]
  D --> E
  A --> F["CASCADE: small model"]
  F --> G{"confidence<br/>above tau?"}
  G -- yes --> H["answer"]
  G -- no --> I["large model<br/>(small already paid for)"]
  I --> H
  style B fill:#dfe,stroke:#5a5
  style I fill:#fde,stroke:#c69
```

**Signals available to a cascade, in increasing order of cost.** The small
model's token entropy ({{eq:token-entropy}}) is free — it is already computed.
Self-evaluation ({{cite:kadavath2022}}) costs a short second generation.
Sampling variance costs $n$ generations. Agreement between two small models
costs a second small model. **The free signal is usually good enough**, and
teams frequently reach for expensive signals before measuring whether entropy
suffices.

**Why thresholds must be re-tuned on every model change.**
{{ch:llm-next-token}} showed confidence surviving alignment as a rank rather
than a probability, so $\tau$ is a quantile of a distribution that moves with
the checkpoint. **A threshold that escalated 30% of traffic last month may
escalate 60% today**, doubling cost with no code change — and this is the most
common routing incident.

**Routing on the request versus routing on the domain.** A per-request router is
what this chapter models; many production systems route on *feature* instead —
this endpoint uses the small model, that one uses the large. It is cruder, it
has no router overhead and no router errors, and it captures a large share of
the available saving. **It should be the baseline any per-request router is
compared against**, and frequently is not.

**Latency is a second axis.** {{eq:cascade-cost}} counts money; a cascade also
pays the small model's *latency* on every request, and escalated requests
experience both. For an interactive product a cascade can be the right cost
decision and the wrong latency decision, and the two must be evaluated
separately ({{ch:llm-inference}}'s TTFT).

**Length interacts with routing.** {{ch:llm-long-context}} showed accuracy
degrading with context length, and it degrades faster for smaller models. **A
router should consider input length as a feature**, since a request that a small
model handles at 2k tokens may be beyond it at 32k — which makes length one of
the cheapest and most predictive routing features available.

**Escalation is not the only response to low confidence.** The same signal can
trigger retrieval ({{part:12}}), a second sample for self-consistency
({{ch:llm-prompting}}), abstention ({{ch:llm-hallucination}}), or escalation to
a human. **These are different destinations on one mechanism**, and choosing
among them is a cost question with the same shape as
{{eq:escalation-threshold}}: each has a price and an expected quality gain, and
the cheapest adequate one wins. Systems that build a cascade and then separately
build an abstention policy have usually implemented the same decision twice.

**The cascade's first stage does double duty.** Its answer is thrown away on
escalation, which feels wasteful and is not: the small model's *attempt* is what
made the routing decision well-informed, and {{eq:cascade-cost}}'s unconditional
$c_1$ is buying that information rather than a wasted generation. Framing it as
the price of a good decision rather than a discarded answer is what makes the
break-even arithmetic legible.

**Warm-up and cold-start effects distort early measurements.** A newly deployed
cascade has no traffic history, so a threshold derived from a sample is fitted
to whatever arrived first. **Escalation rates measured in the first hours of a
deployment are not representative**, and setting a threshold from them is a
common way to arrive at a rate that drifts immediately — which then looks like
{{sec:12-failure-modes}}'s threshold drift and is actually a sampling error.

## 8. Implementation

The cost–quality frontier, traced.

```python {tier=A name=cascade-frontier}
"""The cost-quality frontier of a cascade. Equations (eq:cascade-cost) onward."""
import math

import numpy as np

rng = np.random.default_rng(0)
N = 20_000

SMALL = dict(cost=1.0, accuracy=0.78)
LARGE = dict(cost=12.0, accuracy=0.94)


def _probit(p):
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        lo, hi = (mid, hi) if cdf < p else (lo, mid)
    return (lo + hi) / 2


def make_population(auc):
    """Requests, whether the small model gets each right, and a confidence
    signal whose ranking quality is `auc`."""
    small_ok = rng.random(N) < SMALL["accuracy"]
    large_ok = rng.random(N) < LARGE["accuracy"]
    sep = math.sqrt(2) * _probit(auc)
    conf = rng.normal(np.where(small_ok, sep, 0.0), 1.0)
    return small_ok, large_ok, conf


def cascade(small_ok, large_ok, conf, escalation_rate):
    """Escalate the least-confident `escalation_rate` fraction."""
    k = int(escalation_rate * N)
    order = np.argsort(conf)          # least confident first
    escalated = np.zeros(N, dtype=bool)
    escalated[order[:k]] = True
    correct = np.where(escalated, large_ok, small_ok)
    cost = SMALL["cost"] + escalated.mean() * LARGE["cost"]  # eq:cascade-cost
    return float(correct.mean()), float(cost)


small_ok, large_ok, conf = make_population(0.82)
print(f"small: cost {SMALL['cost']:.0f}, accuracy {SMALL['accuracy']:.2f}")
print(f"large: cost {LARGE['cost']:.0f}, accuracy {LARGE['accuracy']:.2f}")
print(f"confidence signal AUC 0.82\n")

print(f"{'escalated':>10} {'accuracy':>10} {'cost':>8} {'vs always-large':>17} "
      f"{'quality kept':>14}")
for e in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    acc, cost = cascade(small_ok, large_ok, conf, e)
    print(f"{e:>10.0%} {acc:>10.4f} {cost:>8.2f} "
          f"{cost / LARGE['cost']:>16.0%} {acc / LARGE['accuracy']:>14.1%}")

# Equation (eq:cascade-breakeven): where does the cascade stop paying?
breakeven = 1 - SMALL["cost"] / LARGE["cost"]
print(f"\nbreak-even escalation (eq:cascade-breakeven): {breakeven:.0%}")
print(f"above that, the cascade costs more than always using the large model")

# The signal's value: eq:signal-value, as the gap from the linear baseline.
print(f"\n{'AUC':>6} " + " ".join(f"{'e=' + f'{e:.0%}':>9}"
                                   for e in (0.1, 0.3, 0.5)))
for auc in (0.5, 0.7, 0.82, 0.9, 0.99):
    so, lo, cf = make_population(auc)
    row = " ".join(f"{cascade(so, lo, cf, e)[0]:>9.4f}"
                   for e in (0.1, 0.3, 0.5))
    print(f"{auc:>6.2f} {row}")

print("""
At AUC 0.5 the accuracy barely moves with escalation — a useless signal
escalates a random 30%, which is as likely to include requests the small model
would have got right as ones it would not. The rows above it bow upward, and
that bowing IS the signal's value (eq:signal-value).

Note what this says about where to invest: the models are identical in every
row. Only the routing signal changed.""")
```

Now the comparison that decides whether to build one at all:

```python {tier=A name=routing-options}
"""Router, cascade, feature-level routing, or nothing. All four costed."""
import numpy as np

rng = np.random.default_rng(1)
N = 20_000
SMALL_COST, LARGE_COST = 1.0, 12.0
LARGE_ACC = 0.94
ROUTER_COST = 0.15                 # an embedding + linear classifier

# Ground truth: each request has a latent difficulty, and the small model
# succeeds on the easy ones. Its overall accuracy is therefore an OUTCOME of
# the difficulty distribution rather than a parameter — which is the point,
# since routing exists precisely because that accuracy is not uniform.
difficulty = rng.random(N)
small_ok = rng.random(N) > difficulty
large_ok = rng.random(N) < LARGE_ACC


def router_prediction(quality):
    """A router that estimates difficulty with a given correlation to truth."""
    noise = rng.normal(0, 1.0, N)
    return quality * difficulty + (1 - quality) * rng.random(N) + 0.15 * noise


STRATEGIES = {}

STRATEGIES["always small"] = (float(small_ok.mean()), SMALL_COST)
STRATEGIES["always large"] = (float(large_ok.mean()), LARGE_COST)

# Feature-level routing: a fixed 60/40 split with no per-request decision.
split = rng.random(N) < 0.6
acc = float(np.where(split, small_ok, large_ok).mean())
cost = float(np.where(split, SMALL_COST, LARGE_COST).mean())
STRATEGIES["feature-level (60/40)"] = (acc, cost)

# Per-request router, at two prediction qualities.
for quality, label in [(0.55, "router (weak)"), (0.85, "router (good)")]:
    pred = router_prediction(quality)
    to_large = pred > np.quantile(pred, 0.6)      # send hardest 40% to large
    acc = float(np.where(to_large, large_ok, small_ok).mean())
    cost = ROUTER_COST + float(np.where(to_large, LARGE_COST, SMALL_COST).mean())
    STRATEGIES[label] = (acc, cost)

# Cascade at a 30% escalation rate, with the small model's confidence.
sep = 1.2
conf = rng.normal(np.where(small_ok, sep, 0.0), 1.0)
k = int(0.3 * N)
esc = np.zeros(N, dtype=bool)
esc[np.argsort(conf)[:k]] = True
acc = float(np.where(esc, large_ok, small_ok).mean())
cost = SMALL_COST + float(esc.mean()) * LARGE_COST
STRATEGIES["cascade (30% escalation)"] = (acc, cost)

print(f"{'strategy':<28} {'accuracy':>10} {'cost':>8} {'quality/cost':>14} "
      f"{'vs always-large':>17}")
for name, (a, c) in sorted(STRATEGIES.items(), key=lambda kv: kv[1][1]):
    print(f"{name:<28} {a:>10.4f} {c:>8.2f} {a / c:>14.4f} "
          f"{f'{a / LARGE_ACC:.0%} qual, {c / LARGE_COST:.0%} cost':>17}")

print("""
Read the quality/cost column. The CASCADE wins it, and by a clear margin — which
is what section 6.2 predicted: the cascade decides after seeing an attempt,
while the router must predict difficulty blind, and its errors concentrate
exactly where the decision matters.

Note also how little separates the weak and good routers on cost: both pay
router overhead and both send 40% of traffic to the large model by construction,
so the prediction quality shows up almost entirely in accuracy. A weak router is
not cheap-and-bad; it is the SAME price as a good one and worse.

Feature-level routing is the baseline worth taking seriously. It has no
overhead, no router errors, and it exploits a real fact — different product
features genuinely have different difficulty distributions. It loses here
because a fixed 60/40 split cannot adapt per request, and it should still be the
first thing tried, because it is a configuration change rather than a system.""")
```

And the failure mode that makes routing an operations problem:

```python {tier=A name=threshold-drift}
"""A threshold is a quantile of a distribution that moves. Costs move with it."""
import numpy as np

rng = np.random.default_rng(4)
N = 20_000
SMALL_COST, LARGE_COST = 1.0, 12.0

# Month 1: confidence distribution under the current checkpoint.
conf_v1 = rng.normal(0.0, 1.0, N)
TARGET_ESCALATION = 0.30
tau = float(np.quantile(conf_v1, TARGET_ESCALATION))
print(f"threshold set for {TARGET_ESCALATION:.0%} escalation: tau = {tau:.3f}\n")

# Month 2: the provider updates the model. ch:llm-next-token — alignment shifts
# the confidence distribution, and the threshold is a fixed number.
SHIFTS = {
    "no change":                    (0.00, 1.00),
    "slightly more confident":      (0.30, 1.00),
    "much more confident":          (0.80, 1.00),
    "less confident":              (-0.40, 1.00),
    "more spread (less certain)":   (0.00, 1.40),
}

print(f"{'checkpoint':<30} {'escalation':>12} {'cost/request':>14} "
      f"{'vs planned':>12}")
planned_cost = SMALL_COST + TARGET_ESCALATION * LARGE_COST
for name, (shift, scale) in SHIFTS.items():
    conf = rng.normal(shift, scale, N)
    e = float((conf < tau).mean())
    cost = SMALL_COST + e * LARGE_COST
    print(f"{name:<30} {e:>12.1%} {cost:>14.2f} {cost / planned_cost:>11.0%}")

print(f"""
The threshold is a fixed number; the distribution it was fitted to is not. A
model that becomes more confident escalates LESS and quietly loses quality; one
that becomes less confident escalates more and quietly costs more. Neither
change is visible in any code diff.

'Much more confident' here escalates almost nothing — the cascade silently
degenerates into always-small, which is the dangerous direction because the cost
metric IMPROVES while quality falls.

The fix is mechanical: store the threshold as a target escalation RATE and
re-derive tau from a recent traffic sample, rather than storing tau. That makes
the invariant the thing you care about — how much traffic escalates — instead of
an implementation detail of a checkpoint that will change.""")
```

## 9. Practical Example

A team serves 2 million requests a day through one large model. They want to cut
cost without a quality regression they cannot detect. The question is which
strategy, and whether the saving survives the overhead.

```python {tier=A name=routing-business-case}
"""Sizing a routing project: saving, risk, and what has to be built."""

REQUESTS_PER_DAY = 2_000_000
TOKENS_PER_REQUEST = 800

MODELS = {
    "large (current)": dict(params=70e9, accuracy=0.94),
    "small":           dict(params=8e9,  accuracy=0.81),
}
GPU_HOUR, DEVICE_FLOPS, MFU = 2.50, 1e15, 0.45


def daily_cost(params, fraction=1.0):
    flops = 2 * params * REQUESTS_PER_DAY * fraction * TOKENS_PER_REQUEST
    hours = flops / (DEVICE_FLOPS * MFU) / 3600
    return hours * GPU_HOUR


large_only = daily_cost(MODELS["large (current)"]["params"])
small_only = daily_cost(MODELS["small"]["params"])
print(f"{REQUESTS_PER_DAY:,} requests/day at {TOKENS_PER_REQUEST} tokens\n")
print(f"{'baseline':<24} {'$/day':>10} {'$/year':>12} {'accuracy':>10}")
print(f"{'always large':<24} {large_only:>10,.0f} {large_only * 365:>12,.0f} "
      f"{MODELS['large (current)']['accuracy']:>10.2f}")
print(f"{'always small':<24} {small_only:>10,.0f} {small_only * 365:>12,.0f} "
      f"{MODELS['small']['accuracy']:>10.2f}")

ratio = MODELS["large (current)"]["params"] / MODELS["small"]["params"]
print(f"\ncost ratio: {ratio:.1f}x")
print(f"break-even escalation (eq:cascade-breakeven): {1 - 1 / ratio:.0%}")

print(f"\n{'escalation':>11} {'$/day':>10} {'$/year saved':>14} "
      f"{'est. accuracy':>14}")
for e in (0.15, 0.25, 0.40, 0.60):
    cost = small_only + e * large_only
    # Quality: the small model keeps the requests it is confident about, so its
    # accuracy on kept traffic exceeds its overall accuracy.
    kept_acc = min(0.81 + 0.10 * (1 - e), 0.93)
    acc = (1 - e) * kept_acc + e * 0.94
    print(f"{e:>11.0%} {cost:>10,.0f} {(large_only - cost) * 365:>14,.0f} "
          f"{acc:>14.4f}")

# What has to be built, and what it costs to be wrong.
print(f"\n{'requirement':<40} {'note'}")
REQUIREMENTS = [
    ("a confidence signal on the small model", "free if entropy suffices"),
    ("a labelled sample from both models", "one-off; a few thousand requests"),
    ("threshold stored as a RATE, not a value", "threshold-drift"),
    ("escalation-rate monitoring + alert", "the primary operational metric"),
    ("a fixed quality probe set", "detects silent degeneration"),
]
for req, note in REQUIREMENTS:
    print(f"{req:<40} {note}")

at_25 = small_only + 0.25 * large_only
print(f"""
At 25% escalation the saving is ${(large_only - at_25) * 365:,.0f}/year for an
estimated accuracy of about 0.93 against the large model's 0.94.

Whether one point of accuracy is worth that depends entirely on what the
requests are for — equation (eq:escalation-threshold) makes the value of a
correct answer an explicit input, and it is a product number rather than a
modelling one.

The operational requirements matter as much as the arithmetic. Without
escalation-rate monitoring the cascade drifts silently (threshold-drift), and
without a fixed probe set the drift is invisible until users complain — because
the direction that saves money is the same direction that loses quality.""")
```

> PRODUCTION TIP: Store the escalation *rate* you want and re-derive the
> threshold from recent traffic, rather than storing a threshold value. It makes
> the quantity you care about the invariant, and it survives model updates that
> would otherwise silently double or halve your escalation.

## 10. Production Considerations

**Compare against feature-level routing first.** `routing-options` shows it
competitive with a per-request router and free of overhead and router errors.

**Store the escalation rate, not the threshold.** `threshold-drift` shows a
fixed $\tau$ silently changing behaviour when the model changes.

**Alert on escalation rate.** It is the leading indicator for both cost and
quality, and it moves before either.

**Keep a fixed quality probe set.** Degeneration into always-small improves the
cost metric, so cost monitoring alone cannot detect it.

**Make the router cheap.** {{eq:router-overhead}} — an LLM-based router
frequently costs as much as the model it routes to.

**Include input length as a routing feature.** Small models degrade faster with
context ({{ch:llm-long-context}}), so length is cheap and predictive.

**Evaluate latency separately from cost.** A cascade adds the small model's
latency to every request including escalated ones.

**What to monitor:** escalation rate, cost per request, probe-set accuracy,
router-decision distribution, and the confidence distribution's shape. The last
one is what predicts a drift before it happens.

## 11. Common Mistakes

**Beginners:**

*Forgetting the small model's cost on escalation.* {{eq:cascade-cost}} — $c_1$
is unconditional.

*Assuming a cascade always saves money.* {{eq:cascade-breakeven}} gives the
limit, and at a 2x ratio it is only 50%.

*Using an LLM as the router.* {{eq:router-overhead}}.

**Experienced practitioners:**

*Storing a threshold value.* It is a quantile of a moving distribution.

*Not comparing against feature-level routing.* It is the honest baseline.

*Monitoring cost without quality.* Degeneration into always-small looks like a
cost win.

*Setting a threshold from a target error rate on an aligned model.* Confidence
survives alignment as a rank, not a probability
({{ch:llm-next-token}}) — the curve must be traced.

*Ignoring that router errors concentrate on hard cases.*
{{sec:6-mathematical-foundation}} — the router is least reliable where the
decision matters most.

## 12. Failure Modes

**Threshold drift.** *Symptom:* escalation rate moving with no code change.
*Cause:* a model update shifting the confidence distribution. *Fix:* store a
rate.

**Silent degeneration to always-small.** *Symptom:* cost improving, quality
falling. *Detection:* the probe set — and note the cost metric moves the *wrong*
way, so it provides false reassurance.

**Router overhead eating the saving.** *Detection:* the arithmetic, before
building.

**Escalation storm.** A distribution shift making many requests look hard,
escalating most traffic and multiplying cost. *Fix:* a hard cap on escalation
rate, with the excess handled by the small model or refused.

**Quality regression on a slice.** The router systematically mis-handling one
request type. *Detection:* per-slice accuracy, not aggregate.

**Latency regression.** A cascade adding the small model's time to every
request. *Detection:* p99 TTFT, separately from cost.

## 13. Alternatives

{#tbl:cost-reduction-strategies caption="Ways to reduce inference cost. Routing is one of several and rarely the first to try — the rows above it lose nothing, which is a strong argument for exhausting them first."}

| Strategy | Quality cost | Engineering | Where treated |
|---|---|---|---|
| Batching, caching | none | low | {{ch:llm-inference}} |
| Prefix caching | none | low | {{ch:llm-inference}} |
| Shorter prompts / fewer passages | can improve it | low | {{ch:llm-long-context}} |
| Quantisation | small | moderate | {{part:15}} |
| Feature-level routing | targeted | low | this chapter |
| Cascade | small, tunable | moderate | this chapter |
| Per-request router | small, tunable | high | this chapter |
| Distillation | rare knowledge | high | {{ch:fm-distillation}} |

**What genuinely differs.** The first three preserve the output exactly and
should be exhausted before anything else. Quantisation and distillation change
the model; routing changes *which* model. **Distillation is routing's main
rival**: instead of routing around a weak small model, make the small model
better. It costs a training project and it removes the routing decision entirely
— and {{ch:fm-distillation}}'s economics show it paying back in days at this
volume, which makes it the serious alternative rather than a footnote.

## 14. Evaluation

**Is routing working?**

1. **The frontier**, not a point. Sweep $\tau$ and plot cost against quality;
   a single operating point cannot show whether the signal is any good.
2. **Against feature-level routing**, the honest baseline.
3. **Per-slice quality**, since router errors concentrate.
4. **Latency separately from cost.**

**Is the signal any good?** {{eq:signal-value}} — the area between the
achieved curve and the straight line a useless signal would give. `cascade-frontier`
shows the AUC-0.5 row flat, which is what "no signal" looks like.

**And the operational question.** A routing system is only as good as its
monitoring: the failure that matters is drift, it is invisible in cost metrics,
and the probe set is the only instrument that sees it.

## 15. Advanced Concepts

**Learned routers.** {{maturity:EMERGING}} Training a classifier on
(request, which-model-succeeded) pairs. Requires running both models on a sample
— the router's training set is the experiment you were avoiding, though it is
one-off.

**Multi-tier cascades.** {{maturity:ESTABLISHED}} Three or more stages.
{{eq:cascade-cost}} generalises, and each additional stage adds an unconditional
cost term — so tiers pay only when each stage is much cheaper than the next.

**Speculative decoding as implicit routing.** {{maturity:ESTABLISHED}} A small
model drafts and a large one verifies ({{ch:llm-inference}}), which is a cascade
at *token* granularity — and unlike this chapter's cascades it provably
preserves the large model's distribution, so it has no quality cost at all.

**Routing on predicted length.** {{maturity:EMERGING}} Output length is a large
cost driver ({{ch:llm-inference}}) and is partly predictable from the request,
so routing on expected output length rather than on difficulty targets cost
directly.

**Abstention as a routing target.** {{maturity:ESTABLISHED}}
{{ch:llm-hallucination}}'s abstention is this machinery with a human as the
escalation target, and the risk–coverage curve is
{{eq:signal-value}} in other units.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:llm-next-token}}'s rank-not-probability finding is why
thresholds are traced rather than computed, and its entropy is the free signal.
{{ch:llm-inference}} supplies the cost model and speculative decoding.
{{ch:llm-hallucination}}'s risk–coverage curve is
{{eq:signal-value}} in other units. {{ch:llm-long-context}} supplies input
length as a feature. {{ch:nlp-similarity}}'s retrieve-then-rerank and
{{ch:nlp-extraction}}'s encoder-then-LLM cascade are this pattern's earlier
appearances. {{ch:fm-distillation}} is the alternative.

**Forwards.** {{part:22}} designs systems where routing is one component among
many. {{part:23}} implements it in the serving layer. {{part:24}} builds the
monitoring this chapter insists on, and {{part:25}} the evaluation that makes a
quality regression detectable.

## 17. Exercises

**Beginner**

1. Why does a cascade pay the small model's cost even when it escalates?
2. With $c_1=1$, $c_2=8$, what escalation rate is break-even?
3. What is the difference between a router and a cascade?

**Intermediate**

4. Using {{eq:cascade-cost}}, compute cost at 40% escalation for
   $c_1=1$, $c_2=15$.
5. Using {{eq:escalation-threshold}}, find the threshold for $q_2=0.95$,
   $c_2=10$, $V=500$.
6. Explain why a fixed $\tau$ is unsafe across model versions.

**Advanced**

7. Derive {{eq:cascade-breakeven}} and interpret the cost ratio's role.
8. Explain why router errors concentrate where $\hat{p}\approx 0.5$ and what
   that implies about achievable routing quality.
9. Extend {{eq:cascade-cost}} to three tiers and state when a third tier pays.

**Implementation**

10. Extend `cascade-frontier` to three tiers and find the optimal escalation
    rates for a stated cost budget.
11. Implement a learned router on synthetic (request, success) data and compare
    against the confidence cascade at matched cost.
12. Implement the rate-based threshold from `threshold-drift` and show it
    holding escalation stable across simulated distribution shifts.
13. Add input length as a routing feature and measure the improvement over
    confidence alone.

**Reasoning**

14. Your cascade's cost fell 20% last month and nobody changed anything. What
    happened, and is it good news?
15. Argue when distillation beats routing, being specific about volume and the
    small model's gap.

## 18. Interview Questions

**Beginner**

1. What is model routing and why do it?
2. Why can a cascade cost more than always using the large model?
3. What signal decides escalation?

**Intermediate**

4. Derive the break-even escalation rate.
5. Why must thresholds be re-tuned on model change?
6. What is the baseline a per-request router should beat?

**Senior**

7. Size a routing project for 2M requests/day. What do you build first?
8. Your cost improved and users complain. Diagnose it.
9. Routing or distillation? What decides?

**Systems**

10. Design monitoring for a cascade. What alerts, on what?
11. How would you prevent an escalation storm?

## 19. Research Questions

**How much better can a router be than a cascade?** The cascade sees an attempt
and the router does not, which should make the cascade strictly better informed.
Measure the gap at matched cost — if it is small, routers are worth their extra
complexity; if large, cascades should be the default.

**Can difficulty be predicted without running anything?**
{{sec:6-mathematical-foundation}} says router errors concentrate at
$\hat{p}\approx 0.5$. Characterise how much of that is irreducible — some
requests may be genuinely unpredictable in difficulty until attempted.

**Does routing interact with quality drift?** Both the router and the routed
models change over time, and the router was trained against a checkpoint. Measure
how fast a learned router decays after a model update, since it determines the
retraining cadence and nobody publishes it.

**Is the free signal good enough?** Entropy costs nothing and self-evaluation
costs a generation. Measure the AUC gap between them on real traffic — if it is
small, a great deal of routing engineering is unnecessary.

## 20. Chapter Summary

Routing is the third appearance of one architecture: **a cheap high-recall stage
in front of an expensive precise one**, after {{ch:nlp-similarity}}'s
retrieve-then-rerank and {{ch:nlp-extraction}}'s encoder-then-LLM cascade.
Naming the pattern is worth more than any individual instance of it.

**A router decides before generating and a cascade after.** The cascade is
better informed — it has seen an attempt — and pays $c_1$ unconditionally
{{eq:cascade-cost}}, which means it beats always-large only while escalation
stays below $1 - c_1/c_2$ {{eq:cascade-breakeven}}. **At a 10x cost ratio that
is 90% and at 2x it is 50%**, so the saving scales with the ratio and routing
between adjacent model sizes is rarely worth the machinery.

**The deciding input is the confidence signal, and its quality is what varies.**
`cascade-frontier` holds both models fixed and varies only the signal: at AUC
0.5 accuracy barely responds to escalation, and at higher AUC the frontier bows
upward. {{eq:signal-value}} is the area between them, and it is the same curve
as {{ch:llm-hallucination}}'s risk–coverage in different units — **abstention
and routing are one mechanism with different escalation targets.**

**Two things routinely defeat routing projects.**
{{eq:router-overhead}}: an LLM-based router costs about what the model it routes
to costs, which destroys the saving. And feature-level routing — this endpoint
small, that one large — has no overhead and no router errors, captures much of
the available saving, and is the baseline a per-request router must beat and
frequently is not compared against.

**The operational failure is the one to design for.** A threshold is a quantile
of a distribution that moves with every model update, so `threshold-drift` shows
escalation silently doubling or collapsing with no code change. Storing the
*rate* and re-deriving $\tau$ from recent traffic makes the invariant the thing
you care about. And the dangerous direction is the one that saves money:
degeneration into always-small **improves the cost metric while quality falls**,
which is why a fixed probe set is not optional.

## 21. Further Reading

There is less to read here than the topic deserves, and that is worth noting:
routing is mostly engineering practice, published as vendor documentation and
framework code rather than as research. The arithmetic in this chapter is
elementary and the reason it is worth writing down is that it is so often not
done.

{{cite:kadavath2022}} for self-evaluation as a routing signal, which is the one
genuinely researched component.

{{cite:guo2017calibration}} because routing depends entirely on a confidence
signal, and its quality is a calibration question first.

{{cite:sanh2019}} and {{ch:fm-distillation}} for the alternative — making the
small model better rather than routing around it — which at high volume is
frequently the stronger play.

**Where to go next:** this is the last chapter of {{part:10}}. The part
assessment builds a serving path end to end and measures it. Then
{{part:11}} takes up embeddings and vector search, where the
retrieve-then-rerank cascade this chapter generalised was first constructed.
