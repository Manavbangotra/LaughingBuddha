---
id: fm-emergence
number: 83
part: IX
tier: full
status: draft
requires: [fm-scaling-laws, fm-pretraining, ml-metrics, math-probability,
           ds-experiments, fm-what-they-are]
provides: [emergent-ability, discontinuous-metric, metric-artefact,
           capability-forecasting, breakthroughness, grokking,
           per-step-accuracy, capability-versus-loss]
citations: [wei2022emergent, schaeffer2023, brown2020, kaplan2020scaling,
            hoffmann2022chinchilla, bommasani2021, lee2022dedup]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the definition of an emergent ability and identify its two clauses.
2. Derive why exact-match scoring over $k$ independent steps produces a sharp
   curve from a smooth underlying improvement.
3. Reproduce {{cite:schaeffer2023}}'s argument as a simulation and explain
   exactly what it does and does not establish.
4. Distinguish a discontinuity in a model's capability from a discontinuity in
   the metric's view of it.
5. Evaluate an emergence claim by asking the three questions that decide it.
6. Explain why the distinction matters for product requirements even if the
   underlying quantity is smooth.
7. State honestly what is settled and what is not.

## 2. Why This Matters

**This is the chapter where scaling laws stop working.**
{{ch:fm-scaling-laws}} predicts loss to remarkable accuracy. It says nothing
about whether the model can write correct code, and the step from one to the
other is the single largest gap in the field's ability to plan.

**The disagreement here is live and both sides are partly right**, which makes
it unusually good practice. {{cite:wei2022emergent}} documented abilities
appearing sharply with scale; {{cite:schaeffer2023}} argued the sharpness is
manufactured by the choice of metric. Reading both carefully teaches something
more durable than either result: **a measurement choice can create a
qualitative phenomenon out of a quantitative one.**

**It has consequences well outside the technical literature.** Unpredictable
capability jumps are a premise in most arguments about AI risk and most
arguments for pre-deployment evaluation. If emergence is largely a metric
artefact, those arguments need restating — not abandoning, but restating. That
is a high-stakes inference resting on a measurement convention, which is exactly
the kind of thing worth being careful about.

**And the practical version survives the debate entirely.** If your product
requires valid JSON, the metric *is* exact match, and a discontinuity in exact
match is real for you regardless of what the per-token accuracy is doing
underneath. {{sec:9-practical-example}} makes that concrete.

## 3. Prerequisites

{{ch:fm-scaling-laws}} for what scaling predicts well, and the fitted curves
this chapter contrasts capability against. {{ch:fm-pretraining}} for loss as the
training signal. {{ch:ml-metrics}} for metric selection — this chapter is that
topic at its most consequential. {{ch:math-probability}} for the independent
trials in {{sec:6-mathematical-foundation}}. {{ch:ds-experiments}} for what it
takes to establish that an effect is real. {{ch:fm-what-they-are}} for the
evidence-quality habit, which this chapter needs more than any other.

## 4. Intuitive Explanation

Plot a model family's loss against scale and you get a smooth curve
({{ch:fm-scaling-laws}}). Plot its accuracy on three-digit multiplication
against scale and you may get something different: flat at zero, flat at zero,
flat at zero — and then, past some size, rising steeply.

{{cite:wei2022emergent}} catalogued dozens of tasks with that shape and named
the phenomenon: an **emergent ability** is one not present in smaller models and
not predictable by extrapolating from them.

The second clause is the important one. Nobody disputes that big models do
things small models cannot. The claim with teeth is **unpredictability** — that
you could not have seen it coming from the smaller runs.

**Then {{cite:schaeffer2023}} asked an awkward question: what if the model is
improving smoothly and the metric is not?**

Consider three-digit multiplication scored by exact match. To get the answer
right the model must produce every digit correctly. Suppose per-digit accuracy
improves smoothly with scale — 0.3, then 0.6, then 0.9. Exact match over, say,
five digits is then $0.3^5 = 0.002$, $0.6^5 = 0.08$, $0.9^5 = 0.59$.

**Smooth input, sharp output.** The exponent did it. Nothing about the model
changed discontinuously; the scoring function raised a smoothly rising number to
a power, and a power of a number below 1 stays near zero until the number gets
close to 1.

> NOTE: This is not a subtle statistical point. It is the observation that
> $x^k$ looks like a step function for large $k$, applied to a number that is
> rising smoothly. {{sec:8-implementation}} is twenty lines of numpy and is more
> convincing than any amount of argument.

**So is emergence real?** Both of these are true:

- Many published emergence curves flatten out under a continuous metric. The
  sharpness in those cases is a property of the scoring, not the model.
- {{cite:schaeffer2023}} does not show that nothing is ever discontinuous, and
  it cannot — it shows that a particular common measurement practice
  manufactures sharpness, which is a claim about the evidence rather than about
  models.

**And the practical fact is unaffected.** If what you need is a compiling
program or a parseable JSON object, exact match is not a badly chosen metric —
it is the requirement. The discontinuity is real in the only sense that matters
to a deployment.

**The mental model:** capability as measured is a composition of two functions —
how good the model actually is, and how the metric maps that onto a number. A
sharp curve tells you the composition is sharp and does not tell you which
factor is responsible. Where it breaks down: separating them requires a
continuous metric to exist for your task, and for some tasks — did the proof
verify, did the code run — there genuinely is no meaningful partial credit.

## 5. Formal Explanation

### 5.1 The definition

{{cite:wei2022emergent}}: an ability is **emergent** if it is

1. not present in smaller models, and
2. not predictable by extrapolating the performance of smaller models.

Formally, for a performance measure $P$ over scale $s$, emergence at $s^*$
requires $P(s)\approx P_{\text{chance}}$ for $s < s^*$ and $P(s) \gg
P_{\text{chance}}$ for $s > s^*$, with the pre-$s^*$ observations carrying no
signal about the post-$s^*$ values.

**Clause 2 is the falsifiable one**, and it is a claim about a *predictor*, not
about a model. That is what makes it vulnerable to the metric argument: change
the measure $P$ and the predictability changes.

### 5.2 Why exact match manufactures sharpness

Let a task require $k$ sub-steps, each succeeding independently with probability
$p(s)$ that improves smoothly with scale. Exact-match accuracy is

$$
P_{\text{exact}}(s) = p(s)^{k}
$$ (eq:exact-match-composition)

Take the derivative with respect to $p$:

$$
\frac{\dd P_{\text{exact}}}{\dd p} = k\,p^{k-1}
$$ (eq:exact-match-derivative)

At $p = 0.5$ and $k = 10$ this is $10\times0.5^{9} \approx 0.02$; at $p = 0.95$
it is $10\times0.95^9 \approx 6.3$. **The metric's sensitivity to the underlying
quantity varies by more than two orders of magnitude across the range**, being
almost flat where $p$ is small and steep where $p$ approaches 1.

That is a sigmoid in disguise, and it is what produces the characteristic
emergence plot.

### 5.3 The continuous alternative

Score the same task by *per-step* accuracy:

$$
P_{\text{cont}}(s) = p(s)
$$ (eq:continuous-metric)

or by token-level edit distance, or by log-likelihood of the correct answer.
Each is a monotone but *non-explosive* function of the same underlying $p$, so
each traces the underlying smoothness rather than amplifying it.

**{{cite:schaeffer2023}}'s empirical claim** is that when published emergence
tasks are rescored this way, the sharp transitions largely disappear.

> IMPORTANT: Note precisely what that establishes. It shows the *evidence* for
> unpredictability was weaker than it appeared, because the metric was doing
> work that was attributed to the model. It does not show that models improve
> smoothly in every respect, and it does not show that clause 2 is false in
> general. Those are different claims and the literature frequently conflates
> them.

### 5.4 The three questions

Given any emergence claim, three questions decide how much to believe it:

1. **What is the metric, and is it discontinuous in the underlying quantity?**
   Exact match, multiple-choice accuracy with a threshold, and pass@1 on
   programs all are. Log-likelihood and per-step accuracy are not.
2. **How many scale points, and how are they spaced?** Emergence claims often
   rest on three or four model sizes an order of magnitude apart. A sharp curve
   through four points is weak evidence for a sharp function.
3. **Is the task contaminated?** {{cite:lee2022dedup}}'s train/test overlap
   applies here with force: a large model that memorised the test set will show
   exactly the emergence signature, and the signature is indistinguishable from
   capability without a contamination audit.

**Question 3 is the one most often skipped** and is the most damaging when it
applies, because contamination correlates with scale — larger models are usually
trained on more data, and more data means more chance of overlap.

### 5.5 What scaling laws can and cannot forecast

{{ch:fm-scaling-laws}}'s laws predict $L(N,D)$ accurately. Capability is some
unknown function $g$ of the loss and the task:

$$
P_{\text{task}} = g\big(L(N,D),\ \text{task}\big)
$$ (eq:capability-from-loss)

**The laws give the argument of $g$ and say nothing about $g$.** If $g$ is
smooth and monotone, capability is forecastable from loss. If $g$ has a
threshold — which {{eq:exact-match-composition}} shows the *metric* can supply
even when the model does not — it is not.

This is the precise sense in which capability forecasting is unsolved, and it
is why {{part:25}} treats evaluation as a discipline rather than a step.

## 6. Mathematical Foundation

### 6.1 The sharpness of a power

Let $P = p^k$ with $p\in(0,1)$. Define the scale over which $P$ traverses most
of its range: the values of $p$ giving $P = 0.1$ and $P = 0.9$ are

$$
p_{0.1} = 0.1^{1/k},\qquad p_{0.9} = 0.9^{1/k}
$$ (eq:transition-width)

so the transition width in $p$ is

$$
\Delta p = 0.9^{1/k} - 0.1^{1/k}
$$

For $k=1$, $\Delta p = 0.8$ — the whole range. For $k = 10$,
$\Delta p = 0.9895 - 0.7943 = 0.195$. For $k = 50$,
$\Delta p = 0.99789 - 0.95499 = 0.043$.

$\square$

**The transition narrows as $1/k$ roughly**, so a task requiring 50 correct
steps compresses the entire visible transition into 4% of the underlying range.
Sample the scale axis at four points and you will almost certainly see a step.

### 6.2 Why the appearance depends on sampling

Suppose $p(s)$ rises linearly in $\log s$ across the sampled range, and you
measure at $m$ evenly log-spaced scales. The probability that at least one
sample lands inside the transition window is roughly

$$
\Prob[\text{see the transition}] \approx 1 - \big(1 - \Delta p\big)^{m}
$$ (eq:sampling-transition)

For $k=50$ ($\Delta p = 0.043$) and $m = 4$ scale points:
$1 - 0.957^4 = 0.16$. **There is an 84% chance that no sampled model lands in
the transition**, in which case the plot shows a flat line at zero followed by a
jump — the emergence signature — purely from sparse sampling of a continuous
function.

$\square$

This is why question 2 in {{sec:5-formal-explanation}} matters, and why
{{cite:schaeffer2023}}'s rescoring is persuasive: it is not merely a different
number, it is a metric whose transition is wide enough to be sampled.

### 6.3 A worked case

Three-digit multiplication, treating each output digit as a step, $k = 6$.
Suppose per-digit accuracy at three model scales is $0.55$, $0.75$, $0.92$.

$$
P_{\text{exact}} = 0.55^6 = 0.028,\quad 0.75^6 = 0.178,\quad 0.92^6 = 0.606
$$

Plotted, exact-match rises by a factor of 21 across the range while per-digit
accuracy rises by a factor of 1.7. **The same three models, the same three
measurements, and two qualitatively different stories** — one of a sudden
capability appearing, one of steady improvement.

Neither number is wrong. They answer different questions: "can it do the task"
and "how good is it getting".

## 7. Internal Mechanics

```mermaid {#fig:emergence-mechanism caption="How a smooth capability becomes a sharp curve. The model's underlying competence improves continuously with scale; the scoring function composes that with a power, and the composition is what gets plotted and named."}
graph LR
  A["scale s"] --> B["loss L(N,D)<br/>smooth, ch:fm-scaling-laws"]
  B --> C["per-step competence p(s)<br/>smooth, monotone"]
  C --> D{"scoring<br/>function"}
  D -->|"per-step accuracy<br/>P = p"| E["smooth curve<br/>no emergence"]
  D -->|"exact match<br/>P = p^k"| F["sharp curve<br/>'emergence'"]
  style F fill:#fde,stroke:#c69
  style E fill:#dfe,stroke:#5a5
```

**Why this is easy to miss in practice.** The person plotting the curve did not
choose a discontinuous metric to manufacture an effect; they chose the metric the
task naturally comes with. Multiplication is scored by whether the answer is
right. Code is scored by whether it runs. The discontinuity enters through the
task definition, not through an analytical choice, which is why it went
unremarked for so long.

**Where genuinely discontinuous behaviour would come from.** The metric argument
does not exclude real phase transitions. Candidates that have been proposed:
a circuit forming abruptly during training (grokking-like dynamics), a
capability requiring a minimum representational width, or in-context learning
becoming available once a certain structure exists. **None of these has been
demonstrated to produce an unpredictable jump in a continuous metric at scale**,
which is the evidential situation as of writing.

**Contamination as a confound.** A larger model trained on more data has a
higher chance of having seen the test items ({{ch:fm-datasets}}). Memorisation
produces a jump from chance to high accuracy, and it correlates with scale.
Distinguishing this from capability requires the contamination audit that most
emergence papers do not report.

## 8. Implementation

The central demonstration of the chapter, and it is short.

```python {tier=A name=emergence-from-metric}
"""One smoothly improving model family, two metrics, two different stories."""
import numpy as np

# Model scale, log-spaced over four orders of magnitude.
scales = np.logspace(7, 11, 40)          # 10M to 100B parameters

# Per-step competence improves SMOOTHLY and monotonically with log scale.
# Nothing here is discontinuous. This is the ground truth.
p = 1 / (1 + np.exp(-(np.log10(scales) - 9.0) * 1.6))

K = 12                                    # sub-steps the task requires
exact = p ** K                            # equation (eq:exact-match-composition)

print("A single smooth p(s), scored two ways\n")
print(f"{'params':>10} {'per-step p':>12} {'exact match p^12':>18}")
for i in range(0, len(scales), 6):
    print(f"{scales[i]:>10.1e} {p[i]:>12.4f} {exact[i]:>18.6f}")


def sharpness(y, x):
    """Fraction of the total rise that happens in the steepest 10% of x."""
    y = (y - y.min()) / (y.max() - y.min())
    n = max(1, len(x) // 10)
    gains = [y[i + n] - y[i] for i in range(len(y) - n)]
    return max(gains)


print(f"\nlargest rise within any 10% window of log-scale:")
print(f"  per-step accuracy : {sharpness(p, scales):.3f}")
print(f"  exact match       : {sharpness(exact, scales):.3f}")

# Equation (eq:transition-width): how wide is the transition in p?
for k in (1, 6, 12, 50):
    lo, hi = 0.1 ** (1 / k), 0.9 ** (1 / k)
    print(f"k={k:>3}: exact match goes 0.1 -> 0.9 as p goes "
          f"{lo:.4f} -> {hi:.4f}  (width {hi - lo:.4f})")

# Equation (eq:sampling-transition): what a sparse scale sweep sees.
print(f"\nWhat a study with only a few model sizes observes (k={K}):")
lo, hi = 0.1 ** (1 / K), 0.9 ** (1 / K)
width = hi - lo
for m in (3, 4, 6, 10, 40):
    idx = np.linspace(0, len(scales) - 1, m).astype(int)
    in_window = int(((p[idx] > lo) & (p[idx] < hi)).sum())
    prob = 1 - (1 - width) ** m
    print(f"  {m:>3} model sizes: {in_window} land inside the transition "
          f"(predicted chance of >=1: {prob:.0%})")

print("""
Read the last block carefully. With three or four model sizes — which is what
most emergence studies have — it is likely that NO sampled model lands in the
transition window. The plot then shows a flat line at chance followed by a
jump, and the jump is an artefact of sampling a continuous function sparsely
with a metric that compresses its transition into a few per cent of the range.

The model in this listing improves perfectly smoothly. There is no emergence
anywhere in it. Everything sharp on the right-hand plot was put there by the
exponent.""")
```

Now the other half of the argument, which the chapter must give equal weight:
a genuinely discontinuous capability, to show the test can distinguish them.

```python {tier=A name=real-versus-apparent-discontinuity}
"""Can the continuous metric tell a real jump from a metric artefact? Yes."""
import numpy as np

scales = np.logspace(7, 11, 40)
log_s = np.log10(scales)
K = 12

# Case A: smooth competence, discontinuous METRIC (the Schaeffer scenario).
p_smooth = 1 / (1 + np.exp(-(log_s - 9.0) * 1.6))
metric_A = p_smooth ** K

# Case B: genuinely discontinuous competence — the model acquires the ability
# at a threshold — scored with the SAME exact-match metric.
p_step = np.where(log_s < 9.0, 0.05, 0.95)
metric_B = p_step ** K

# Case C: the same genuine discontinuity, scored continuously.
metric_C = p_step


def transition_decades(y, x_log):
    """How many decades of scale the curve needs to go from 10% to 90% of its
    range. Narrow means sharp. This is the right measure: unlike a per-step
    jump, it does not depend on how densely the scale axis was sampled."""
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    lo = x_log[np.argmax(y >= 0.1)]
    hi = x_log[np.argmax(y >= 0.9)]
    return float(hi - lo)


print(f"{'case':<44} {'metric':<8} {'10%->90% width (decades)':>26}")
rows = [
    ("A: smooth ability, exact-match metric", "p^12", metric_A),
    ("A: smooth ability, continuous metric", "p", p_smooth),
    ("B: real discontinuity, exact-match metric", "p^12", metric_B),
    ("C: real discontinuity, continuous metric", "p", metric_C),
]
for label, m, y in rows:
    print(f"{label:<44} {m:<8} {transition_decades(y, log_s):>26.3f}")

print("""
Compare row 1 against row 2, then row 3 against row 4.

The exact-match metric more than halves the apparent transition width of a
perfectly smooth ability (2.46 decades -> 1.03). Sample that at four model
sizes an order of magnitude apart and it reads as a step. The sharpness was
added by the exponent.

The real discontinuity measures 0.00 decades under BOTH metrics. No rescoring
smooths a competence that genuinely stepped, which is what makes the continuous
metric a test rather than a way of explaining emergence away: row 2 and row 4
differ by everything, while row 1 and row 3 are both narrow enough to be
mistaken for each other on a sparse sweep.""")

# The exact-match metric compresses the smooth ability's transition...
assert transition_decades(metric_A, log_s) < 0.5 * transition_decades(p_smooth, log_s), \
    "p^k must narrow the transition of a smooth ability"
# ...but a genuine step stays a step however it is scored.
assert transition_decades(metric_C, log_s) < 0.2, \
    "a real discontinuity survives rescoring"
assert transition_decades(p_smooth, log_s) > 1.0, \
    "the smooth ability must remain wide under a continuous metric"
```

And the confound that most emergence studies do not control for:

```python {tier=A name=contamination-mimics-emergence}
"""Memorisation produces the emergence signature, and correlates with scale."""
import numpy as np

rng = np.random.default_rng(0)
scales = np.logspace(7, 11, 12)
log_s = np.log10(scales)
N_TEST = 400

# Genuine capability: smooth, and modest even at the largest scale.
true_skill = 0.15 + 0.35 / (1 + np.exp(-(log_s - 10.2) * 2.0))

# Contamination: larger models are trained on more data, so the chance that a
# given test item was seen rises with scale (ch:fm-datasets).
seen_fraction = np.clip((log_s - 8.5) / 3.0, 0, 1) * 0.55

print(f"{'params':>10} {'true skill':>11} {'% test seen':>12} "
      f"{'observed':>10} {'inflation':>10}")
observed = []
for i, s in enumerate(scales):
    seen = rng.random(N_TEST) < seen_fraction[i]
    correct = np.where(seen, rng.random(N_TEST) < 0.97,       # memorised
                       rng.random(N_TEST) < true_skill[i])    # actually solved
    obs = correct.mean()
    observed.append(obs)
    print(f"{s:>10.1e} {true_skill[i]:>11.3f} {seen_fraction[i]:>11.1%} "
          f"{obs:>10.3f} {obs - true_skill[i]:>10.3f}")

observed = np.array(observed)
print(f"\ntrue skill rises   {true_skill[0]:.3f} -> {true_skill[-1]:.3f} "
      f"({true_skill[-1] / true_skill[0]:.1f}x)")
print(f"observed rises     {observed[0]:.3f} -> {observed[-1]:.3f} "
      f"({observed[-1] / observed[0]:.1f}x)")
print(f"largest jump: true {np.max(np.diff(true_skill)):.3f}, "
      f"observed {np.max(np.diff(observed)):.3f}")

print("""
The observed curve rises far faster than the underlying skill, and it does so
because contamination CORRELATES WITH SCALE — a bigger model saw more data, so
it saw more of the test set. This is a third explanation for a sharp curve,
alongside a real jump and a metric artefact, and it is the one least often
audited.

Note that no rescoring detects it. A continuous metric would show the same
inflation, because the model really is producing the right answers. Only a
contamination audit against the training corpus separates this case, and for
frontier models that corpus is not public.""")
```

## 9. Practical Example

A team is deciding whether their product feature is viable. The feature needs
the model to emit a valid JSON object with seven required fields, and the
downstream system rejects anything malformed. They have measurements from three
model sizes and want to know whether the next size up will work.

This is exactly the situation the chapter is about, and the answer is not
"emergence is a myth, extrapolate the smooth metric".

```python {tier=A name=forecasting-a-threshold-requirement}
"""Forecasting an all-or-nothing product requirement from continuous measurements."""
import numpy as np

FIELDS = 7                       # every one must be correct
SIZES = np.array([1e9, 7e9, 70e9])
NEXT_SIZE = 400e9

# What the team measured: per-field correctness, which is continuous and
# therefore forecastable — this is the metric to instrument.
per_field = np.array([0.72, 0.89, 0.968])
observed_valid = per_field ** FIELDS

print(f"{'params':>9} {'per-field':>11} {'valid JSON':>12} "
      f"{'usable?':>9}")
for s, pf, v in zip(SIZES, per_field, observed_valid):
    print(f"{s:>9.0e} {pf:>11.3f} {v:>12.3f} {str(v > 0.95):>9}")

# Forecast the CONTINUOUS quantity, then compose the metric. Fit in logit
# space, where the smooth improvement is close to linear in log scale.
def logit(x):
    return np.log(x / (1 - x))


slope, intercept = np.polyfit(np.log10(SIZES), logit(per_field), 1)
pred_logit = slope * np.log10(NEXT_SIZE) + intercept
pred_field = 1 / (1 + np.exp(-pred_logit))
pred_valid = pred_field ** FIELDS

print(f"\nforecast at {NEXT_SIZE:.0e} parameters:")
print(f"  per-field correctness : {pred_field:.4f}  (extrapolated)")
print(f"  valid-JSON rate       : {pred_valid:.4f}  (composed)")

# The naive alternative: extrapolate the discontinuous metric directly.
naive_slope, naive_int = np.polyfit(np.log10(SIZES), observed_valid, 1)
naive_pred = naive_slope * np.log10(NEXT_SIZE) + naive_int
print(f"  naive extrapolation of valid-JSON rate: {naive_pred:.4f}")
print(f"  (composing the smooth forecast gives {pred_valid:.4f} — the naive "
      f"line is fitted to a curve that is not straight)")

# How much per-field accuracy does the requirement actually need?
for target in (0.90, 0.95, 0.99):
    needed = target ** (1 / FIELDS)
    print(f"\nto reach {target:.0%} valid JSON, per-field must reach "
          f"{needed:.4f}")
    print(f"  currently {per_field[-1]:.4f} at 70B -> the gap is "
          f"{needed - per_field[-1]:+.4f}")

print("""
Two lessons, and the second is the one teams get wrong.

Forecast the CONTINUOUS quantity and compose the metric afterwards. Per-field
accuracy extrapolates sensibly in logit space; the valid-JSON rate does not
extrapolate at all, because it is a seventh power and fitting a straight line
to it is meaningless.

But the REQUIREMENT is still all-or-nothing. Knowing that the sharpness is a
metric artefact does not soften the product constraint one bit — it just tells
you which number to measure and forecast. The discontinuity is real for the
deployment even though it is not real in the model.""")
```

> PRODUCTION TIP: Instrument the continuous quantity even when you only care
> about the threshold. A team tracking only "valid JSON rate" sees 0.12, 0.44,
> 0.80 and cannot forecast; a team tracking per-field accuracy sees 0.72, 0.89,
> 0.968 and can. It is the same three experiments and one of them is
> predictive.

## 10. Production Considerations

**Report both metrics, always.** A dashboard showing only pass/fail cannot
distinguish a model that is nearly right from one that is nowhere near, and the
two have completely different implications for whether to keep investing.

**Emergence claims should not drive roadmaps.** "The next model will be able to
do X" based on a sharp curve through four points is not a forecast. Base
planning on continuous metrics you can extrapolate, and treat threshold
crossings as things you verify rather than predict.

**Audit contamination before believing a jump.** For your own evaluations, check
overlap against the training corpus if you have it. For hosted models, prefer
tasks constructed after the training cutoff — this is the only defence available
and it is the reason to maintain private evaluation sets ({{part:25}}).

**Budget for the threshold being crossed at an unknown scale.** If a feature
requires an all-or-nothing capability, the honest plan includes what happens if
the next model does not clear it: a fallback, a narrower scope, or a different
architecture such as constrained decoding ({{ch:llm-structured-output}}), which
turns the threshold into a guarantee rather than a hope.

**What to monitor:** per-step or per-field accuracy alongside the pass rate, and
the gap between them. A widening gap means the model is getting closer without
crossing, which is exactly the information a threshold metric hides.

## 11. Common Mistakes

**Beginners:**

*Reading a sharp curve as a sharp capability.* {{eq:exact-match-composition}}
shows the metric can supply the sharpness. Ask what the metric is before
interpreting the shape.

*Extrapolating a discontinuous metric.* Fitting a line to $p^k$ is fitting a
line to a curve that is not one. Forecast $p$ and compose.

*Concluding emergence is debunked.* {{cite:schaeffer2023}} shows the evidence
was weaker than claimed, not that no discontinuity exists.

**Experienced practitioners:**

*Comparing models on threshold metrics only.* Two models at 0% pass rate can be
very far apart in competence, and the metric cannot see it.

*Skipping the contamination audit.* It is the explanation that correlates with
scale, mimics emergence exactly, and survives rescoring — the
`contamination-mimics-emergence` listing shows a curve that no metric choice
can debunk.

*Treating four scale points as a curve.* {{eq:sampling-transition}} says a
sparse sweep is likely to miss the transition entirely, and a missed transition
looks like a step.

*Assuming the continuous metric exists.* For "did the proof verify" there may be
no meaningful partial credit, in which case the metric argument does not apply
and the threshold is the only available measurement.

## 12. Failure Modes

**Metric-manufactured emergence.** A sharp curve attributed to the model.
*Detection:* rescore with a continuous metric, as in
`real-versus-apparent-discontinuity`. *Consequence:* forecasts and risk
arguments built on an artefact.

**Contamination-driven emergence.** Memorisation rising with scale.
*Detection:* $n$-gram overlap against the training corpus
({{ch:fm-datasets}}), or evaluation on post-cutoff data. *This is the failure
that survives rescoring and is therefore the most dangerous.*

**Sparse-sampling illusion.** Too few model sizes to resolve a transition.
*Detection:* count the scale points and compute {{eq:sampling-transition}}.

**Planning on an unverified threshold.** A roadmap assuming the next model
crosses a bar. *Symptom:* a feature that cannot ship and has no fallback.
*Mitigation:* {{sec:10-production-considerations}}'s fallback requirement.

**Selection over tasks.** With enough tasks and enough metrics, some will show
sharp curves by chance. *Detection:* was the task chosen before or after the
curve was seen? Emergence papers that survey many tasks and report the striking
ones have a multiple-comparisons problem that is rarely quantified.

## 13. Alternatives

{#tbl:emergence-explanations caption="Explanations for a sharp capability curve, and how to distinguish them. Only the last is a claim about the model; the first three are claims about the measurement, and they must be excluded before the last is considered."}

| Explanation | Mechanism | Test | Survives rescoring |
|---|---|---|---|
| Metric composition | $P = p^k$ amplifies a smooth $p$ | rescore continuously | no |
| Sparse sampling | transition falls between scale points | add scale points | no |
| Contamination | memorisation rising with data volume | overlap audit, post-cutoff data | **yes** |
| Real discontinuity | competence genuinely steps | continuous metric still jumps | **yes** |

**What differs versus what merely looks different.** The first two are
measurement artefacts and dissolve under better measurement. The third is a real
effect on the *observed* number produced by something other than capability. Only
the fourth is emergence in the sense {{cite:wei2022emergent}} intended, and it
is the one for which the evidence is thinnest — precisely because the first
three have to be excluded first, and usually are not.

**The related phenomenon of grokking** — a model's test accuracy jumping long
after training loss converged — is genuinely discontinuous in training time and
is well documented at small scale on algorithmic tasks. Whether it is the same
phenomenon as scale-emergence is unresolved, and the two are frequently
conflated because both produce a step.

## 14. Evaluation

**Evaluating a capability claim.** The three questions of
{{sec:5-formal-explanation}}, in order: what is the metric, how many scale
points, and is it contaminated. A claim that does not answer all three is not
evaluable, and most published ones answer at most one.

**Evaluating your own model's progress.**

1. **Track a continuous metric** even when the requirement is a threshold.
2. **Report the pass rate alongside it**, and the gap.
3. **Hold out post-cutoff data** so contamination cannot explain improvement.
4. **Use enough scale points** if you are claiming anything about scaling —
   {{eq:sampling-transition}} is the calculation for how many.

**On the standing question of this book.** *What was held fixed?* For emergence,
the answer is usually: the metric was held fixed at whatever the task came with,
and no one asked what that choice was doing. The pattern is
{{cite:levy2015}} and {{cite:kaplan2020scaling}} again — a nuisance variable
carrying a conclusion.

## 15. Advanced Concepts

**Grokking.** {{maturity:EMERGING}} Sudden generalisation long after training
loss plateaus, on algorithmic tasks at small scale. Genuinely discontinuous in
training time; its relationship to scale-emergence is unresolved.

**Predictable-from-loss capability.** {{maturity:EMERGING}} Some benchmarks
correlate tightly enough with pretraining loss to be forecast from it. Where
this holds, {{eq:capability-from-loss}}'s $g$ is effectively smooth and
forecasting works — identifying which tasks these are is useful and
under-studied.

**Breakthroughness metrics.** {{maturity:RESEARCH FRONTIER}} Quantifying how
much of a curve's sharpness is attributable to the metric, so emergence claims
can be compared rather than asserted.

**Phase transitions in representation.** {{maturity:RESEARCH FRONTIER}} Whether
specific circuits form abruptly during training, which would be a mechanistic
basis for genuine discontinuity. Interpretability work at small scale is
suggestive and does not yet scale.

**Inverse scaling.** {{maturity:ESTABLISHED}} Tasks where larger models do
*worse*, which is a useful corrective: it demonstrates that scale is not
uniformly beneficial and that the relationship between loss and capability is
not monotone for every task.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-scaling-laws}} predicts loss precisely and this chapter
is about the gap between that and capability — {{eq:capability-from-loss}} is
the formal statement of the gap. {{ch:ml-metrics}} established that the unit of
evaluation is a modelling decision; {{ch:nlp-extraction}} showed it costing 0.86
against 0.67 on the same predictions, and this chapter shows the same choice
manufacturing a qualitative phenomenon. {{ch:fm-datasets}}'s contamination is
the confound that survives every rescoring. {{ch:ds-experiments}} supplies the
standard for establishing an effect is real.

**Forwards.** {{part:25}} builds evaluation into a discipline, with this
chapter's three questions as part of its foundation.
{{ch:llm-structured-output}} turns the JSON threshold of
{{sec:9-practical-example}} from a hope into a guarantee via constrained
decoding — which is the engineering answer to an all-or-nothing requirement.
{{ch:res-scaling}} returns to what scaling does and does not predict with the
frontier's current evidence.

## 17. Exercises

**Beginner**

1. A task needs 8 independent correct steps. Per-step accuracy goes from 0.6 to
   0.9. What does exact-match accuracy do?
2. State the two clauses of the emergence definition and say which is
   falsifiable.
3. Why can a sharp curve not distinguish a real jump from a metric artefact?

**Intermediate**

4. Using {{eq:transition-width}}, compute the transition width for $k = 20$ and
   $k = 100$.
5. With $k = 20$ and 5 log-spaced model sizes, use
   {{eq:sampling-transition}} to estimate the chance of observing the
   transition.
6. Explain why contamination correlates with scale and why that matters here.

**Advanced**

7. Derive {{eq:transition-width}} and show the width scales approximately as
   $1/k$ for large $k$.
8. Construct a task and metric where a genuinely smooth capability produces a
   sharp curve that rescoring cannot fix. What property does your metric have?
9. Design a study that would establish a genuine discontinuity, addressing all
   three questions of {{sec:5-formal-explanation}}.

**Implementation**

10. Extend `emergence-from-metric` with a third metric — token-level edit
    distance — and show it also traces the smooth underlying curve.
11. Implement a "breakthroughness" score quantifying how much of a curve's
    sharpness is attributable to the metric, and apply it to both cases in
    `real-versus-apparent-discontinuity`.
12. Extend `contamination-mimics-emergence` with an $n$-gram audit that
    successfully identifies the contaminated items, and show the corrected curve.
13. Reproduce the sparse-sampling illusion: sample the smooth curve at 3, 4, 6
    and 20 points and show how the apparent sharpness changes with sampling
    alone.

**Reasoning**

14. Emergence is a premise in many AI-risk arguments. If it is largely a metric
    artefact, which of those arguments survive and which need restating? Be
    specific.
15. A product requires 99% valid JSON. Explain why the metric debate changes
    your measurement strategy but not your requirement.

## 18. Interview Questions

**Beginner**

1. What is an emergent ability?
2. Why does exact-match scoring produce sharp curves?
3. What does scaling law predict, and what does it not?

**Intermediate**

4. Explain {{cite:schaeffer2023}}'s argument and what it does not establish.
5. How would you evaluate a claim that a capability emerged at some scale?
6. Why is contamination especially dangerous for emergence claims?

**Senior**

7. Your product needs an all-or-nothing capability. How do you forecast whether
   the next model will have it?
8. A colleague shows a sharp capability curve across four model sizes. What do
   you ask?
9. How should emergence affect a technical roadmap? How should it not?

**Systems**

10. Design an evaluation harness that cannot be fooled by metric-manufactured
    emergence.
11. How would you maintain evaluation sets that stay uncontaminated as models
    improve?

## 19. Research Questions

**How much published emergence survives all three tests?**
{{cite:schaeffer2023}} applied the metric test. Nobody has systematically
applied the sampling and contamination tests to the same task set. The fraction
surviving all three is the number the debate actually needs and it is not known.

**Which tasks are forecastable from loss?** {{eq:capability-from-loss}}'s $g$ is
smooth for some tasks and not others. Characterising which — by task structure
rather than case by case — would turn capability forecasting from an art into a
method.

**Is grokking the same phenomenon?** Both produce steps, one in training time
and one in scale. Determining whether they share a mechanism, or merely a shape,
is answerable at small scale and would resolve a persistent conflation.

**Can breakthroughness be measured before the fact?** Given a task and a metric,
predict how sharp the observed curve will be — from $k$, the sampling density,
and the underlying improvement rate. If this works, emergence claims become
checkable arithmetic rather than plots.

## 20. Chapter Summary

An emergent ability is defined as one not present in smaller models and not
predictable from them. The second clause is the falsifiable one, and it is a
claim about a *predictor* rather than about a model — which is what makes it
vulnerable to the choice of measure.

**A metric can manufacture a discontinuity from a smooth improvement.** When a
task requires $k$ correct sub-steps and is scored by exact match, the observed
score is $p^k$ {{eq:exact-match-composition}}, whose sensitivity to $p$ varies by
orders of magnitude across the range. The transition compresses into a window of
width $0.9^{1/k} - 0.1^{1/k}$ {{eq:transition-width}} — about 4% of the range at
$k = 50$. With the three or four model sizes a typical study has,
{{eq:sampling-transition}} says the transition will probably fall entirely
between sampled points, producing a flat line and a jump from a perfectly smooth
function. `emergence-from-metric` shows exactly this with no discontinuity
anywhere in the model.

**What that establishes, precisely:** the evidence for unpredictability was
weaker than it appeared, because work attributed to the model was being done by
the scoring. It does not establish that nothing is discontinuous.
`real-versus-apparent-discontinuity` shows the test that separates them — under
a continuous metric a genuine step still steps, while an artefact flattens.

**And there is a third explanation that survives every rescoring.**
Contamination correlates with scale, because larger models are trained on more
data and more data means more test overlap. Memorisation produces the emergence
signature exactly, and no metric choice detects it — only an audit against the
training corpus, which for frontier models is not public.

**The practical requirement is untouched by the debate.** If a product needs
valid JSON with seven fields, exact match is not a poor metric choice — it is
the requirement, and the threshold is real for the deployment however smooth the
model's underlying competence is. The correct response is not to abandon the
threshold but to *instrument the continuous quantity and compose*: per-field
accuracy extrapolates, the seventh power does not.

## 21. Further Reading

{{cite:wei2022emergent}} should be read for its definition in §2 and its figures.
The figures are the evidence and they are worth studying with
{{eq:exact-match-composition}} in mind — ask, for each panel, what the metric is
and how many sub-steps the task requires.

{{cite:schaeffer2023}} is short and its §3 is the argument. Read it immediately
after Wei et al., and be careful to distinguish what it demonstrates about the
evidence from what it is often reported as demonstrating about models.

{{cite:hoffmann2022chinchilla}} is relevant here as the contrast: loss is
predictable, which is why the unpredictability of capability is surprising
rather than expected.

{{cite:lee2022dedup}}, once more, for the contamination mechanism. It is not an
emergence paper and it supplies the explanation that neither emergence paper
controls for.

**Where to go next:** {{ch:fm-instruction-tuning}} leaves the question of what
the base model can do and takes up the first correction stage — teaching a text
continuer that a request should be answered.
