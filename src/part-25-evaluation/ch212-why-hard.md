---
id: ev-why-hard
number: 212
part: XXV
tier: full
status: draft
requires: [semantic-failure-has-no-instrument, evaluation-sets-decay-silently,
           refresh-beats-growth, uniform-sampling-misses-rare-failures]
provides: [metric-choice-manufactures-the-finding, discontinuity-hides-progress,
           reference-scoring-penalises-valid-answers, agreement-caps-measurable-quality]
citations: [schaeffer2023mirage, hendrycks2020mmlu, rein2023gpqa, chen2021humaneval,
            jimenez2023swebench]
---

## 1. Learning Objectives

By the end of this chapter you will be able to show that raising a smoothly improving
capability to a power produces an apparent threshold, and locate that threshold from the
exponent alone; explain why a discontinuous metric cannot support the decisions evaluation
exists to support; compute the share of *correct* answers a reference-based metric marks
wrong from the size of the acceptable-answer space; explain why adding references does not
fix it and why execution-based grading does; and derive the ceiling that annotator
agreement places on any automated metric's correlation with quality.

## 2. Why This Matters

Take one capability that improves smoothly with scale — no jumps anywhere — and score a
five-token answer by exact match. The reported score reads **0.0200 at 3B and 0.9139 at
180B**, and every chart of it shows a threshold that does not exist in the underlying
curve. Change the answer length to two tokens and the "emergence" moves to **3.0B**; change
it to twelve and it moves to **20.0B** ({{eq:metric-choice-manufactures-the-finding}}).
{{cite:schaeffer2023mirage}} showed this is not a hypothetical: a large class of published
emergent abilities disappears under a continuous metric, and the effect can be manufactured
on demand.

The cost is not aesthetic. Extrapolating one scale step, the continuous metric errs by
**14%** and the discontinuous one by **48%** ({{eq:discontinuity-hides-progress}}) — and
the discontinuous metric's worst errors are at the low end, where a programme is decided.
Across five generations of genuine compounding progress, exact match reports `no
capability` four times.

The second problem is worse because there is no metric that escapes it. Most useful tasks
have no ground truth — they have a space of acceptable answers, and the reference is one
draw from it. On summarisation, a single-reference metric marks **99.4%** of the model's
*correct* answers wrong; on email drafting, **99.99%**
({{eq:reference-scoring-penalises-valid-answers}}). Five references cover **2.8%** of a
summarisation space.

And the human judgement that would settle it is noisy. At **81%** raw annotator agreement,
no metric can correlate with true quality above **0.79**, so a metric already at 0.71 has
**0.08** of headroom ({{eq:agreement-caps-measurable-quality}}) — and below 74% agreement
the headroom is negative.

## 3. Prerequisites

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is the operational
form of this chapter's problem: the reason there is no instrument is that the quantity has
no ground truth to instrument against, which is what {{sec:9-practical-example}} measures.

{{eq:evaluation-sets-decay-silently}} and {{eq:refresh-beats-growth}} from
{{ch:ops-prompt-versioning}} established that an evaluation set's *coverage* decays while
its reported score does not. This chapter adds the prior problem: even a perfectly current
set can be scored by a metric that reports something other than quality.

{{eq:uniform-sampling-misses-rare-failures}} from {{ch:ops-observability}} governs how the
evaluation set gets built in the first place, and its bias compounds with everything here.

{{cite:hendrycks2020mmlu}} is the canonical broad benchmark and the source of a finding
usually dropped when its headline score is quoted: models are poorly calibrated and
"frequently do not know when they are wrong."

## 4. Intuitive Explanation

There are two hard problems in evaluating AI systems, and neither is the one people expect.

The expected problem is that the systems are complicated. That is not it. Complicated
systems are evaluated routinely — aircraft, compilers, drug candidates — and the difficulty
is engineering rather than conceptual.

**The first real problem is that the metric is a choice, and below a certain performance
level the choice determines the finding.**

Here is the mechanism, and it is arithmetic rather than anything deep. Suppose your task
has a five-token answer and the model has to get all five right. Suppose the model's
per-token accuracy improves smoothly with scale — 2.6%, 7.4%, 21.5%, 45.7%, 69.6% — a clean
curve with no jumps.

Now score by exact match. Exact match is per-token accuracy raised to the fifth power:
0.0000, 0.0000, 0.0005, 0.0200, 0.1637. That curve looks like nothing, nothing, nothing,
a flicker, and then a capability arriving.

Nothing arrived. You raised a smooth curve to the fifth power, and exponentiation turns
gentle curves into thresholds. Where the threshold appears to sit is set by the exponent —
which is to say by *how long the answer is* — and not by the model at all. Two-token answers
"emerge" at 3B parameters; twelve-token answers "emerge" at 20B. Same model, same
capability, same day.

This is not a thought experiment. {{cite:schaeffer2023mirage}} tested it against the
InstructGPT and GPT-3 families, replicated it in a meta-analysis of BIG-Bench, and then
manufactured never-before-seen emergent abilities in vision tasks by choosing metrics for
the purpose. When you can produce a phenomenon on demand by changing how you measure, the
phenomenon belongs to the measurement.

The practical damage is in what you cannot do with a metric that reads zero. You cannot
extrapolate it, because zero has no slope. You cannot compare two candidate models that
both score zero. You cannot set a target, forecast a generation, or tell a funder that
things are going well. Every question evaluation exists to answer requires a metric with a
derivative, and the metrics people find most interpretable — did it get the answer right? —
are precisely the ones whose derivative vanishes where the decisions are made.

Which yields an unintuitive rule: **choose a metric for its derivative when tracking
progress and for its interpretability when deciding acceptance**, and accept that these are
two different metrics. Most teams own one and use it for both.

**The second real problem is that for most tasks there is no ground truth to compare
against.**

A classifier has a label. `positive` or `negative`, and one of them is right. Everything
about evaluation is easy in that world, which is why the textbook chapter on evaluation is
about classifiers.

Now ask the model to summarise a paragraph. What is the right answer? There is no right
answer — there is a *space* of faithful summaries, easily hundreds of them, and whichever
one your annotator happened to write is one draw from that space.

So when you compare the model's summary against the reference, you are not asking "is this
correct?" You are asking "is this the same one?" If there are a hundred and eighty
acceptable summaries and the model produces a good one at random, it matches the reference
about once in a hundred and eighty tries.

Which means the metric marks 99.4% of the model's correct answers wrong. Not 99.4% of its
answers — 99.4% of the ones that were *right*.

The obvious response is to collect more references, and the arithmetic of that response is
unkind. Coverage is references divided by the size of the answer space. Five references
cover 2.8% of a summarisation space. A hundred references — which is a serious annotation
programme, weeks of work — cover 56% of summarisation and 0.67% of email drafting. Cost
grows linearly and coverage does not, so the approach works exactly when the answer space is
small, which is exactly when you did not need it.

Partial-credit metrics — n-gram overlap, embedding similarity — soften this rather than
solving it. They stop requiring an exact match and start measuring proximity to one
arbitrary draw. That is better, and it is still a measurement of "how close is this to the
particular correct answer we happened to write down," which is not the question anybody
wanted answered.

There is exactly one clean escape, and it is worth naming precisely because it is
underused. **If you can state an acceptance predicate instead of writing an answer, the
problem disappears.** A unit test does not sample the space of correct programs — it
*defines* it. Every program that passes is correct, however different from every other one.
That is why {{cite:chen2021humaneval}}'s functional-correctness grading and
{{cite:jimenez2023swebench}}'s test-graded GitHub issues are the most trustworthy numbers in
this part of the book, and it is why the first question in any evaluation design should be
"can this be checked rather than compared?"

For the tasks where it cannot, there is a further ceiling, and it is the one that ends most
arguments about metric quality.

Suppose you validate an automated metric by correlating it against human ratings. How high
can that correlation go? Not to 1.0, because the human ratings are noisy — two annotators
looking at the same output disagree some of the time. If they agree 81% of the time, the
reliability of a single human label is about 0.62, and the highest correlation any metric
can achieve with the underlying truth is the square root of that: 0.79.

So a metric correlating at 0.71 has eight points of headroom, not twenty-nine. And below
74% agreement the headroom goes negative, which means **your metric is already better than
the labels you are grading it against**, and every genuine improvement from here will be
measured as a regression.

That is routinely misdiagnosed. A metric that stops improving against human labels has
either stopped improving or hit the annotation ceiling, and from the metric's side these are
indistinguishable. Telling them apart requires measuring annotator agreement, which costs
one double-labelled sample and is skipped almost universally.

## 5. Formal Explanation

**Metric-induced thresholds.** Let $p(s)$ be an underlying per-unit capability, smooth and
monotone in scale $s$. An exact-match metric over a $k$-unit answer reports $M_k(s) =
p(s)^k$. Then

$$\frac{d M_k}{d s} = k\, p^{k-1} \frac{dp}{ds},$$

which is vanishingly small wherever $p$ is small and $k$ is large, and grows sharply once
$p$ approaches 1. The curve's inflection is located where $p(s) \approx (1 - 1/k)$, so its
apparent threshold is set by $k$ — a property of the answer format — and not by any property
of the model. Composing a smooth function with a high power is the whole mechanism.

**The extrapolation loss.** For linear extrapolation in $\log s$, the relative error of
predicting $M(s_{i+1})$ from $M(s_{i-1}), M(s_i)$ scales with the curvature of $M$ over the
interval divided by $M$ itself. For $M_k = p^k$ the curvature term carries a factor $k(k-1)$
while the denominator carries $p^k$, so relative error is worst exactly where $p$ is small.
A metric whose relative extrapolation error is largest in the region where programmes are
decided is not a progress signal.

**The reference-sampling loss.** Let $A$ be the set of acceptable answers for an item and
$R \subseteq A$ the reference set, $|R| = r$. If the model's output is uniform over $A$
conditional on being acceptable, then $\Pr[\text{scored correct}] = q \cdot \min(1, r/|A|)$
where $q$ is true accuracy. The metric is therefore $q$ scaled by a task-dependent constant
$r/|A|$, which preserves ordering within a task and destroys both level and cross-task
comparability. It also destroys ordering *between* systems whenever $|A|$ differs by system
— which it does whenever one system is more expansive than another.

**The attenuation ceiling.** If human labels have reliability $\rho$ (the correlation
between two independent labellings of the same item), then the correlation between any
metric and the noisy label is bounded by $\sqrt{\rho}$ times its correlation with the true
score. Hence $r_{\max} = \sqrt{\rho}$, and a metric's measured correlation against human
labels understates its true correlation by that factor.

## 6. Mathematical Foundation

The composition that manufactures a threshold:

$$M_k(s) = p(s)^k, \qquad \frac{\partial\, s^\star}{\partial k} > 0, \qquad \frac{\partial\, s^\star}{\partial(\text{model})} \;\text{ is not the term that moved}$$ (eq:metric-choice-manufactures-the-finding)

where $s^\star$ is the scale at which $M_k$ first exceeds a fixed threshold. At $k=2$,
$s^\star = 3$B; at $k=5$, $8$B; at $k=12$, $20$B, on one unchanged capability curve.

The extrapolation consequence:

$$\varepsilon_{\text{rel}}(M) = \frac{|\hat{M}(s_{i+1}) - M(s_{i+1})|}{M(s_{i+1})}, \qquad \bar\varepsilon(p) = 14\%, \quad \bar\varepsilon(p^5) = 48\%$$ (eq:discontinuity-hides-progress)

with the discontinuous metric's error concentrated at small $s$, where it reads near zero
and carries no gradient.

Reference-based scoring as a scaled measurement:

$$\mathbb{E}[\text{score}] = q \cdot \min\!\left(1, \frac{r}{|A|}\right), \qquad \text{valid-but-marked-wrong} = 1 - \min\!\left(1, \frac{r}{|A|}\right)$$ (eq:reference-scoring-penalises-valid-answers)

At $|A| = 180$ and $r = 1$: **99.4%** of correct answers marked wrong. At $|A| = 15{,}000$:
**99.99%**.

And the ceiling on validation:

$$r_{\max} = \sqrt{\rho}, \qquad \rho = \kappa = \frac{p_o - p_e}{1 - p_e}$$ (eq:agreement-caps-measurable-quality)

At $p_o = 0.81$, $p_e = 0.50$: $\kappa = 0.62$ and $r_{\max} = 0.79$.

## 7. Internal Mechanics

Why does the exact-match metric survive despite all of this? Because it is the only metric
that is *unarguable*. When a stakeholder asks whether the system got the answer right, exact
match answers yes or no and nobody negotiates. Every continuous metric invites the question
"but what does 0.457 mean?", and that question has no short answer.

So the metric that is hardest to defend statistically is the easiest to defend socially,
and evaluation designs are chosen in meetings. This is the same selection pressure that
made p99 latency the standard tail metric and error rate the standard reliability metric:
legibility wins, and the cost of legibility is paid later by whoever has to forecast.

The answer-space problem has a subtler origin. Nobody decides that a task has 180 acceptable
answers; the number is a property of the task that never gets written down. And the
annotation process actively hides it: an annotator is asked to *write the answer*, produces
one, and the artefact that reaches the evaluation harness is a single string with no
indication that four hundred others would have done. **The reference set records a draw and
discards the distribution**, and every downstream metric then treats the draw as the
distribution.

That framing suggests the diagnostic. Ask two annotators to answer the same open-ended item
independently and compare. If they write substantially different answers and both are
acceptable, $|A| > 1$ and every reference-based number you have is scaled by an unknown
constant. This costs one afternoon and is almost never done, because the annotation process
is designed to produce references rather than to characterise the space they were drawn
from.

The attenuation ceiling has a mechanical consequence that is easy to miss. Because
$r_{\max} = \sqrt{\rho}$, the ceiling falls *slowly* at first and then quickly: reliability
0.90 gives 0.95, but reliability 0.32 gives 0.57. So a modest deterioration in annotation
quality — a new vendor, a rushed batch, an ambiguous guideline revision — moves the ceiling
much more than it moves the agreement statistic that would reveal it. Teams monitoring raw
agreement see it drift from 88% to 81% and read that as a small change; the metric ceiling
moved from 0.87 to 0.79, which is most of the headroom.

Finally, the two problems interact in a way neither section covers alone. A metric with a
vanishing derivative is usually chosen *because* the reference-based alternative was
unconvincing — exact match at least means something — so the task's large answer space is
what drove the team toward the discontinuous metric in the first place. The two failures are
not independent; the second causes the first.

## 8. Implementation

The first listing takes one smooth capability and measures it six ways.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ha1}
"""The metric is not a lens on the finding. Below a point, it is the finding.

cite:schaeffer2023mirage showed that a large class of reported "emergent abilities"
disappears when the metric is changed from a discontinuous one to a continuous one, and
that the effect can be manufactured on demand in unrelated domains.

This listing reproduces the mechanism from first principles. One smoothly improving
underlying capability, measured six ways, produces six different stories about when the
capability appeared -- and two of them say it never did
(eq:metric-choice-manufactures-the-finding).

The consequence is not philosophical. A team tracking the discontinuous metric sees a
flat line for several generations of work that was in fact improving steadily, and the
flat line is what gets reported to whoever decides whether to continue
(eq:discontinuity-hides-progress).
"""
import math

SCALES = [0.1, 0.3, 1.0, 3.0, 8.0, 20.0, 70.0, 180.0, 400.0]   # billions of parameters
ANSWER_TOKENS = 5           # the task's answer is five tokens; all must be right


def per_token(scale):
    """Underlying capability: smooth, monotone, no discontinuity anywhere."""
    return 1.0 / (1.0 + math.exp(-(math.log10(scale) - 0.55) * 2.35))


print("One underlying capability, improving smoothly with scale. No jumps.")
print(f"The task needs {ANSWER_TOKENS} tokens and all of them must be right.")
print()
print(f"{'params (B)':>12}{'per-token':>12}{'exact match':>14}"
      f"{'token accuracy':>17}{'log-likelihood':>17}")
print("-" * 72)
tab = {}
for s in SCALES:
    p = per_token(s)
    em = p ** ANSWER_TOKENS
    ll = ANSWER_TOKENS * math.log(p)
    tab[s] = (p, em, p, ll)
    print(f"{s:>12.1f}{p:>12.3f}{em:>14.4f}{p:>17.3f}{ll:>17.3f}")

print()
print("Same numbers. The middle column is the one that gets reported.")

print()
print()
print("When did the capability 'appear'? Each metric answers differently.")
print()
THRESH = 0.05               # "it works" is conventionally somewhere around here
print(f"{'metric':>26}{'first scale above 5%':>23}{'value one step earlier':>25}")
print("-" * 74)


def first_above(f, thresh):
    for s in SCALES:
        if f(s) >= thresh:
            return s
    return None


METRICS = [
    ("exact match (all 5 right)", lambda s: per_token(s) ** ANSWER_TOKENS),
    ("exact match, 12 tokens",    lambda s: per_token(s) ** 12),
    ("exact match, 2 tokens",     lambda s: per_token(s) ** 2),
    ("mean token accuracy",       lambda s: per_token(s)),
    ("normalised log-likelihood", lambda s: 1.0 + math.log(per_token(s)) / 6.0),
]
emerge = {}
for name, f in METRICS:
    s = first_above(f, THRESH)
    prev = SCALES[SCALES.index(s) - 1] if s and SCALES.index(s) > 0 else None
    emerge[name] = (s, f(prev) if prev else 0.0)
    print(f"{name:>26}{(str(s) + 'B') if s else 'never':>23}"
          f"{(f(prev) if prev else 0.0):>25.4f}")

print()
print()
print("How abrupt each metric looks: largest ratio between adjacent scales.")
print()
print(f"{'metric':>26}{'largest jump':>15}{'looks like':>28}")
print("-" * 69)
jumps = {}
for name, f in METRICS:
    best = 1.0
    for a, b in zip(SCALES, SCALES[1:]):
        va, vb = f(a), f(b)
        if va > 1e-9:
            best = max(best, vb / va)
    jumps[name] = best
    verdict = ("a phase change" if best > 20 else
               "a sharp gain" if best > 5 else
               "steady progress")
    print(f"{name:>26}{best:>14.1f}x{verdict:>28}")

print()
print()
print("The predictability test: extrapolate each metric one scale step from its")
print("own last two points, and compare against what actually happened.")
print()
print(f"{'known up to':>14}{'next scale':>13}"
      f"{'token acc: pred/act':>22}{'rel err':>10}"
      f"{'exact: pred/act':>20}{'rel err':>10}")
print("-" * 89)


def step(f, i):
    """Linear extrapolation in log-scale from points i-1, i to point i+1."""
    x0, x1, x2 = (math.log10(SCALES[i - 1]), math.log10(SCALES[i]),
                  math.log10(SCALES[i + 1]))
    y0, y1 = f(SCALES[i - 1]), f(SCALES[i])
    return y1 + (y1 - y0) / (x1 - x0) * (x2 - x1), f(SCALES[i + 1])


err_c, err_d = [], []
for i in range(1, len(SCALES) - 1):
    pc, ac = step(per_token, i)
    pd, ad = step(lambda s: per_token(s) ** ANSWER_TOKENS, i)
    rc, rd = abs(pc - ac) / ac, abs(pd - ad) / max(ad, 1e-9)
    err_c.append(rc)
    err_d.append(rd)
    print(f"{SCALES[i]:>13.1f}B{SCALES[i + 1]:>12.1f}B"
          f"{pc:>12.3f}/{ac:<9.3f}{rc:>10.0%}"
          f"{pd:>10.4f}/{ad:<8.4f}{rd:>10.0%}")
print("-" * 89)
mean_c, mean_d = sum(err_c) / len(err_c), sum(err_d) / len(err_d)
print(f"{'MEAN RELATIVE ERROR':>27}{mean_c:>32.0%}{mean_d:>30.0%}")

print()
print()
print("What a team sees, tracking one metric across five generations of work.")
print()
GEN = [(0.1, "gen 1"), (0.3, "gen 2"), (1.0, "gen 3"), (3.0, "gen 4"),
       (8.0, "gen 5")]
print(f"{'':>10}{'exact match':>14}{'reported as':>26}"
      f"{'token accuracy':>17}{'reported as':>22}")
print("-" * 89)
for s, label in GEN:
    p = per_token(s)
    em = p ** ANSWER_TOKENS
    a = "no capability" if em < 0.01 else ("marginal" if em < 0.1 else "works")
    b = ("no capability" if p < 0.15 else
         "clear progress" if p < 0.6 else "works")
    print(f"{label:>10}{em:>14.4f}{a:>26}{p:>17.3f}{b:>22}")

print()
print()
print("And the cost of the wrong choice, in decisions rather than numbers.")
print()
print(f"{'decision':>34}{'under exact match':>20}{'under token accuracy':>23}")
print("-" * 77)
DECISIONS = [
    ("continue funding after gen 3",  "no",  "yes"),
    ("forecast gen 5 from gen 4",     "no",  "yes"),
    ("attribute the gen-5 gain",      "to scale", "to steady progress"),
    ("set a target for gen 6",        "unable",   "extrapolable"),
    ("compare two 1B candidates",     "tied near 0", "separable"),
]
for d, a, b in DECISIONS:
    print(f"{d:>34}{a:>20}{b:>23}")

print(f"""
The first table is the entire mechanism and it fits in three columns. One capability,
improving smoothly -- {tab[0.3][0]:.3f} to {tab[180.0][0]:.3f} per token across the range
-- and an exact-match score that reads {tab[3.0][1]:.4f} at 3B and
{tab[180.0][1]:.4f} at 180B.

Nothing jumped. The exponent did the work: **raising a smooth curve to the fifth power
produces a curve that looks like a threshold** (eq:metric-choice-manufactures-the-finding),
and the threshold's location is a property of the exponent rather than of the model.

The emergence table makes that concrete by varying only the answer length. The same
capability "appears" at {emerge['exact match, 2 tokens'][0]}B if the answer is two tokens
and {emerge["exact match, 12 tokens"][0]}B
if it is twelve. **The task did not change. The formatting of the answer changed**, and with
it the published finding about where the ability emerged.

The abruptness table is what a reader of the chart would conclude. Exact match at 5 tokens
jumps {jumps['exact match (all 5 right)']:.1f}x between adjacent scales, which reads as a
phase change; mean token accuracy jumps {jumps['mean token accuracy']:.1f}x, which reads as
steady progress. Same underlying numbers, two incompatible scientific claims, and the
choice between them was made when somebody decided how to score the answer.

The predictability table is the practical loss and it is larger than the aesthetic one.
Extrapolating one scale step from the previous two points, the continuous metric is off by
**{mean_c:.0%}** on average and the discontinuous one by **{mean_d:.0%}** --
{mean_d / mean_c:.0f} times worse, from the same data
(eq:discontinuity-hides-progress). And the errors are worst exactly where the decision
gets made: at the low end, where the discontinuous metric reads near zero and therefore
carries no gradient to extrapolate along.

That is not a subtlety about charts. It means the discontinuous metric is unusable for
exactly the decisions evaluation exists to support: is this working, is it improving, and
how much more of this do we need?

The generations table says what that looks like inside an organisation. Under exact match,
generations one through four all report `no capability` or `marginal`. Under token
accuracy, generation three already reports `clear progress` and generation four confirms
it. **Four consecutive review cycles of real, measurable, compounding progress reported as
nothing happening** -- and a programme cancelled after generation three on the first metric
would have been continued on the second, with the same model in both rooms.

The decisions table is the summary worth carrying. Under a discontinuous metric you cannot
fund, forecast, attribute, target, or compare. Under a continuous one you can do all five.
That is the case for choosing metrics by their *derivative* rather than by their
interpretability, which is close to the opposite of how metrics are usually chosen.

One caution. Continuous metrics are not automatically better -- a per-token accuracy that
is high while the answer is wrong is measuring something the user does not receive. The
claim here is narrower and it is about *when* to use which: **a discontinuous metric is the
right acceptance test and the wrong progress signal**, and most teams own one metric and
use it for both.""")
```

## 9. Practical Example

One capability, no discontinuities anywhere, scored five ways:

```
  params (B)   per-token   exact match   token accuracy   log-likelihood
------------------------------------------------------------------------
         0.1       0.026        0.0000            0.026          -18.342
         1.0       0.215        0.0005            0.215           -7.676
         3.0       0.457        0.0200            0.457           -3.912
         8.0       0.696        0.1637            0.696           -1.810
        20.0       0.854        0.4538            0.854           -0.790
       180.0       0.982        0.9139            0.982           -0.090
```

The middle column is the one that gets reported, and it is the third column raised to the
fifth power. **Nothing jumped; the exponent did the work.**

```
                    metric   first scale above 5%   value one step earlier
--------------------------------------------------------------------------
 exact match (all 5 right)                   8.0B                   0.0200
    exact match, 12 tokens                  20.0B                   0.0130
     exact match, 2 tokens                   3.0B                   0.0464
       mean token accuracy                   0.3B                   0.0255
 normalised log-likelihood                   0.1B                   0.0000
```

The same capability "emerges" at **3.0B** with a two-token answer and **20.0B** with a
twelve-token one ({{eq:metric-choice-manufactures-the-finding}}). The task did not change —
the answer format did.

```
                    metric   largest jump                  looks like
---------------------------------------------------------------------
 exact match (all 5 right)         210.4x              a phase change
    exact match, 12 tokens        8367.7x              a phase change
     exact match, 2 tokens           8.5x                a sharp gain
       mean token accuracy           2.9x             steady progress
 normalised log-likelihood           1.5x             steady progress
```

Same numbers, two incompatible scientific claims, and the choice between them was made when
somebody decided how to score the answer.

```
   known up to   next scale   token acc: pred/act   rel err     exact: pred/act   rel err
-----------------------------------------------------------------------------------------
          1.0B         3.0B       0.344/0.457           25%    0.0009/0.0200         96%
          3.0B         8.0B       0.673/0.696            3%    0.0374/0.1637         77%
          8.0B        20.0B       0.920/0.854            8%    0.2979/0.4538         34%
         20.0B        70.0B       1.069/0.955           12%    0.8504/0.7923          7%
-----------------------------------------------------------------------------------------
        MEAN RELATIVE ERROR                             14%                           48%
```

**14% against 48%** ({{eq:discontinuity-hides-progress}}), with the discontinuous metric's
errors concentrated at the low end — where funding decisions are made.

```
             exact match               reported as   token accuracy           reported as
-----------------------------------------------------------------------------------------
     gen 1        0.0000             no capability            0.026         no capability
     gen 2        0.0000             no capability            0.074         no capability
     gen 3        0.0005             no capability            0.215        clear progress
     gen 4        0.0200                  marginal            0.457        clear progress
     gen 5        0.1637                     works            0.696                 works
```

**Four consecutive review cycles of real compounding progress reported as nothing
happening.** A programme cancelled after generation three on the first metric would have
been continued on the second, with the same model in both rooms.

The second listing measures the problem no metric choice escapes.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ha2}
"""For most useful tasks there is no ground truth, only one sample from a set of them.

A classifier has a label. A summariser does not -- it has a space of acceptable summaries,
and whichever one a human wrote down is a draw from that space rather than the truth about
it.

Reference-based scoring compares the model's draw against the reference's draw, so it
penalises every correct answer that happens to be a different one
(eq:reference-scoring-penalises-valid-answers).

And the human judgement that would settle it is itself noisy, which puts a ceiling on how
well any automated metric can correlate with quality no matter how good the metric is
(eq:agreement-caps-measurable-quality).

This listing measures both, prices the standard workarounds, and finds the one case where
the problem genuinely goes away.
"""
import math

# (task, |A| = size of the acceptable-answer space, note)
TASKS = [
    ("classify sentiment",          1.0,   "the label is the answer"),
    ("extract the invoice date",    2.0,   "two valid formats"),
    ("name the capital city",       1.4,   "occasional alias"),
    ("write a SQL query",          24.0,   "many correct queries"),
    ("summarise a paragraph",     180.0,   "many faithful summaries"),
    ("explain a concept",        2400.0,   "many correct explanations"),
    ("draft a reply email",     15000.0,   "many acceptable replies"),
]
TRUE_ACCURACY = 0.78          # share of the model's answers that ARE acceptable

print("Reference-based scoring credits the model only when its draw from the")
print("acceptable set matches the reference's draw.")
print()
print(f"{'task':>26}{'|A|':>10}{'P(match), 1 ref':>18}"
      f"{'measured':>11}{'valid answers scored wrong':>28}")
print("-" * 93)
tab = {}
for name, A, note in TASKS:
    hit = min(1.0, 1.0 / A)
    measured = TRUE_ACCURACY * hit
    tab[name] = (A, hit, measured, 1.0 - hit)
    print(f"{name:>26}{A:>10.1f}{hit:>18.3f}"
          f"{measured:>11.3f}{1.0 - hit:>28.2%}")

print()
print(f"true accuracy is {TRUE_ACCURACY:.0%} in every row. The last column is the")
print("share of the model's CORRECT answers that the metric marks wrong.")

print()
print()
print("Adding references buys coverage, and the curve is unkind.")
print()
print(f"{'references':>12}", end="")
for name, A, note in TASKS[3:]:
    print(f"{name.split()[-1]:>14}", end="")
print()
print("-" * 68)
multi = {}
for R in (1, 3, 5, 10, 25, 100):
    print(f"{R:>12}", end="")
    for name, A, note in TASKS[3:]:
        cov = min(1.0, R / A)
        multi[(R, name)] = cov
        print(f"{cov:>14.1%}", end="")
    print()

print()
print("A hundred references cover a summarisation space and do not touch an")
print("email space. Labelling cost is linear; coverage is not.")

print()
print()
print("What each standard workaround actually replaces |A| with.")
print()
print(f"{'approach':>26}{'what it measures':>34}{'penalty on task 5':>20}")
print("-" * 80)
SUMM_A = 180.0
APPROACHES = [
    ("single reference, exact",  "match to one arbitrary draw",   1 - 1 / SUMM_A),
    ("5 references, exact",      "match to five draws",           1 - 5 / SUMM_A),
    ("n-gram overlap",           "surface form near one draw",    0.42),
    ("embedding similarity",     "semantic distance to one draw", 0.29),
    ("LLM judge",                "the judge's acceptability set", 0.17),
    ("execution / unit tests",   "whether it works",              0.02),
]
appr = {}
for name, what, pen in APPROACHES:
    appr[name] = pen
    print(f"{name:>26}{what:>34}{pen:>20.1%}")

print()
print("Only the last one changes the problem instead of approximating it.")

print()
print()
print("The other ceiling: human labels are noisy, so no metric can correlate")
print("with quality better than the labels do with themselves.")
print()
print(f"{'annotator agreement':>21}{'reliability':>14}{'metric ceiling':>17}"
      f"{'best reported r':>18}{'headroom':>11}")
print("-" * 81)
REPORTED_R = 0.71             # a good automated metric's correlation with human scores
ceil = {}
for obs in (0.95, 0.88, 0.81, 0.74, 0.66):
    chance = 0.50
    kappa = (obs - chance) / (1 - chance)
    rel = kappa                       # treat kappa as the reliability of one label
    c = math.sqrt(max(rel, 0.0))      # attenuation: r_max = sqrt(reliability)
    ceil[obs] = (kappa, c)
    print(f"{obs:>21.0%}{kappa:>14.2f}{c:>17.2f}"
          f"{REPORTED_R:>18.2f}{c - REPORTED_R:>11.2f}")

print()
print("Below 81% raw agreement, a metric correlating at 0.71 is already at the")
print("ceiling -- and improving the metric cannot help.")

print()
print()
print("Putting both together: what a reported score means.")
print()
print(f"{'measurement design':>30}{'reports':>10}{'true':>8}"
      f"{'level usable?':>16}{'ranking usable?':>18}")
print("-" * 82)
DESIGNS = [
    ("single-reference exact match",  TRUE_ACCURACY / SUMM_A, "no",  "within task"),
    ("n-gram overlap",                TRUE_ACCURACY * 0.58,   "no",  "within task"),
    ("LLM judge",                     TRUE_ACCURACY * 0.83,   "approximately", "yes"),
    ("execution",                     TRUE_ACCURACY * 0.98,   "yes", "yes"),
    ("human, 2 annotators",           TRUE_ACCURACY * 0.94,   "yes", "yes"),
]
for name, rep, lvl, rank in DESIGNS:
    print(f"{name:>30}{rep:>10.3f}{TRUE_ACCURACY:>8.2f}{lvl:>16}{rank:>18}")

print()
print()
print("And the case where even the ranking fails: two systems whose answer")
print("spaces differ, which is what happens when one is more verbose.")
print()
print(f"{'system':>12}{'true quality':>15}{'|A| of its outputs':>21}"
      f"{'single-ref score':>18}{'true rank':>11}{'measured rank':>15}")
print("-" * 92)
SYSTEMS = [("terse", 0.72, 95.0), ("verbose", 0.78, 420.0)]
scores = {}
for name, q, A in SYSTEMS:
    scores[name] = (q, A, q / A)
by_true = sorted(SYSTEMS, key=lambda s: -s[1])
by_meas = sorted(SYSTEMS, key=lambda s: -scores[s[0]][2])
for name, q, A in SYSTEMS:
    tr = [s[0] for s in by_true].index(name) + 1
    mr = [s[0] for s in by_meas].index(name) + 1
    print(f"{name:>12}{q:>15.2f}{A:>21.0f}{scores[name][2]:>18.5f}"
          f"{tr:>11}{mr:>15}")
best_true = max(SYSTEMS, key=lambda s: s[1])[0]
best_meas = max(scores, key=lambda k: scores[k][2])
print()
print(f"better system: {best_true}    better score: {best_meas}")
print(f"score ratio: {scores[best_meas][2] / scores[best_true][2]:.1f}x the wrong way")

print(f"""
The first table is the mechanism and it needs one sentence. True accuracy is
{TRUE_ACCURACY:.0%} in every row, and the reported number ranges from
{tab['classify sentiment'][2]:.3f} to {tab['draft a reply email'][2]:.5f}
(eq:reference-scoring-penalises-valid-answers).

The difference between the rows is not model quality. It is **how many correct answers the
task has**, and the metric divides by that number.

Notice which tasks sit at each end. Classification is the one place reference scoring is
exact, and classification is also the task nobody deploys a language model for. The tasks
people actually ship -- summarise, explain, reply -- have acceptable-answer spaces in the
hundreds or thousands, and on those the single-reference metric marks
{tab['summarise a paragraph'][3]:.1%} to {tab['draft a reply email'][3]:.2%} of *correct*
answers wrong.

The multi-reference table is the standard fix and the table shows why it does not scale.
Five references cover {multi[(5, 'summarise a paragraph')]:.1%} of a summarisation space
and {multi[(5, 'draft a reply email')]:.2%} of an email space. A hundred references --
which is a serious annotation programme -- reach
{multi[(100, 'summarise a paragraph')]:.0%} and
{multi[(100, 'draft a reply email')]:.2%} respectively.

**Labelling cost is linear in R and coverage is R over |A|**, so the approach works exactly
when |A| is small, which is exactly when you did not need it.

The workaround table is the honest survey. Overlap and embedding metrics reduce the penalty
from {appr['single reference, exact']:.0%} to {appr['n-gram overlap']:.0%} and
{appr['embedding similarity']:.0%} by giving partial credit -- but read the middle column:
they are measuring *proximity to one arbitrary draw*, not acceptability. A judge does
better at {appr['LLM judge']:.0%} because it evaluates against a learned acceptability
boundary rather than a sample, which is ch:ev-llm-judge's subject and its own set of
problems.

Only execution changes the question. A unit test does not sample the acceptable set --
**it defines it**, collapsing |A| to one equivalence class by construction, which is why
cite:chen2021humaneval's pass@k and cite:jimenez2023swebench's test-graded issues are the
most trustworthy numbers in this book's evaluation chapters.

The lesson generalises past code: **wherever you can state an acceptance predicate instead
of writing an answer, do that.** It is usually possible for more tasks than teams assume,
and it is almost never the first thing tried.

The agreement table is the second ceiling and it is the one that ends arguments about
metric quality. At {0.81:.0%} raw agreement between annotators, kappa is
{ceil[0.81][0]:.2f} and the highest correlation any metric can achieve with the true
quality is {ceil[0.81][1]:.2f} (eq:agreement-caps-measurable-quality). A metric already
correlating at {REPORTED_R:.2f} has {ceil[0.81][1] - REPORTED_R:.2f} of headroom.

Below that agreement level the headroom is negative, which means **the metric is already
performing better than the labels it is being validated against**, and every further
improvement will be measured as a regression.

That is worth stating plainly because it is routinely misdiagnosed. A metric that stops
improving against human labels has either stopped improving or hit the annotation ceiling,
and the two look identical from the metric's side. The way to tell them apart is to measure
annotator agreement, which costs a double-labelled sample and is skipped almost universally.

The design table converts both results into what a score is good for. A single-reference
score of {TRUE_ACCURACY / SUMM_A:.4f} against a true {TRUE_ACCURACY:.2f} carries no usable
*level* -- you cannot tell a stakeholder the system is right {TRUE_ACCURACY / SUMM_A:.1%} of
the time -- but it may still rank two systems correctly on the same task.

Which is the defence usually offered for these metrics, and the last table is where it
fails. Two systems, true quality {SYSTEMS[0][1]:.2f} and {SYSTEMS[1][1]:.2f}, differing in
verbosity so that their answer spaces are {SYSTEMS[0][2]:.0f} and {SYSTEMS[1][2]:.0f}. The
better system scores {scores['verbose'][2] / scores['terse'][2]:.1f} times *lower*.

**Reference-based scoring is order-preserving only when the compared systems have the same
answer-space size**, and nothing about the comparison guarantees that -- in fact any change
that makes a system more expansive violates it. Which means a metric that has ranked
correctly for two years can invert the first time somebody makes the model more helpful.""")
```

```
                      task       |A|   P(match), 1 ref   measured  valid answers scored wrong
---------------------------------------------------------------------------------------------
        classify sentiment       1.0             1.000      0.780                       0.00%
  extract the invoice date       2.0             0.500      0.390                      50.00%
         write a SQL query      24.0             0.042      0.033                      95.83%
     summarise a paragraph     180.0             0.006      0.004                      99.44%
       draft a reply email   15000.0             0.000      0.000                      99.99%
```

True accuracy is **78%** in every row and the reported number ranges from **0.780** to
**0.00005** ({{eq:reference-scoring-penalises-valid-answers}}). Classification is the one
place reference scoring is exact, and it is the task nobody deploys a language model for.

```
  references         query     paragraph       concept         email
--------------------------------------------------------------------
           1          4.2%          0.6%          0.0%          0.0%
           5         20.8%          2.8%          0.2%          0.0%
          25        100.0%         13.9%          1.0%          0.2%
         100        100.0%         55.6%          4.2%          0.7%
```

Cost is linear in references and coverage is $r/|A|$ — so **the approach works exactly when
the answer space is small, which is exactly when you did not need it.**

```
                  approach                  what it measures   penalty on task 5
--------------------------------------------------------------------------------
   single reference, exact       match to one arbitrary draw               99.4%
            n-gram overlap        surface form near one draw               42.0%
      embedding similarity     semantic distance to one draw               29.0%
                 LLM judge     the judge's acceptability set               17.0%
    execution / unit tests                  whether it works                2.0%
```

Read the middle column. Overlap and embedding metrics measure *proximity to one arbitrary
draw*. **Only execution changes the question** — a test does not sample the acceptable set,
it defines it, which is why {{cite:chen2021humaneval}} and {{cite:jimenez2023swebench}}
produce the most trustworthy numbers in this part.

```
  annotator agreement   reliability   metric ceiling   best reported r   headroom
---------------------------------------------------------------------------------
                  95%          0.90             0.95              0.71       0.24
                  88%          0.76             0.87              0.71       0.16
                  81%          0.62             0.79              0.71       0.08
                  74%          0.48             0.69              0.71      -0.02
                  66%          0.32             0.57              0.71      -0.14
```

At 81% agreement no metric can exceed **0.79** ({{eq:agreement-caps-measurable-quality}});
below 74% the headroom is negative and **further improvement is measured as regression.**

```
      system   true quality   |A| of its outputs  single-ref score  true rank  measured rank
--------------------------------------------------------------------------------------------
       terse           0.72                   95           0.00758          2              1
     verbose           0.78                  420           0.00186          1              2
```

The defence usually offered for reference metrics — the level is meaningless but the ranking
holds — fails whenever answer-space size differs by system. The better system scores
**4.1× lower**, and any change that makes a model more expansive triggers it.

## 10. Production Considerations

Own two metrics, not one: a continuous one for tracking and a discontinuous one for
acceptance. They answer different questions and neither substitutes.

Before believing an emergence claim, ask for the answer length. The apparent threshold moves
with the exponent, and the exponent is the answer format.

Measure your answer-space size before choosing a metric. Two annotators, the same open-ended
items, independently — one afternoon, and it tells you whether any reference-based number
you own carries a level.

Ask whether the task can be checked rather than compared. Execution-graded evaluation is
available for more tasks than teams assume and is almost never the first thing tried.

Measure annotator agreement whenever you validate a metric against human labels. Without it
you cannot distinguish a metric that has stopped improving from one that has hit the
ceiling.

Never compare reference-based scores across tasks. The scale factor is $r/|A|$ and it
differs by orders of magnitude between rows of the same report.

Watch verbosity when comparing systems on reference metrics. A more expansive model has a
larger answer space and will score lower for being better.

## 11. Common Mistakes

**Reading a threshold as a property of the model.** It is a property of the exponent, and
the exponent is the answer length.

**Using one metric for progress and acceptance.** A metric that reads zero has no
derivative, and every progress question needs one.

**Comparing reference-based scores across tasks.** The scale factor differs by a factor of
thousands between classification and drafting.

**Adding references to fix coverage.** Cost is linear, coverage is $r/|A|$, and the tasks
that need it most are the ones where it works least.

**Validating a metric against human labels without measuring agreement.** The ceiling may
already be below the metric's current correlation.

**Treating a reference as the answer.** It is one draw from the answer space, and the space
was discarded when the reference was written.

## 12. Failure Modes

**Programme cancelled at the flat part of the curve.** Four review cycles of real progress
report `no capability`, and the decision is correct given the metric.

**Emergence claim that does not replicate.** A threshold reported under one answer format
vanishes under another, and neither team's model changed.

**Metric improvements that read as regressions.** The annotation ceiling was passed two
quarters ago, and every genuine gain since has been measured downward.

**Ranking inversion on a helpfulness change.** A model made more expansive drops on the
reference metric, is reverted, and the reversion is recorded as a quality win.

**Annotation quality drift that nobody prices.** Raw agreement falls from 88% to 81% and is
read as a small change; the metric ceiling fell from 0.87 to 0.79.

**Benchmark score quoted without a human baseline.** A 39% is reported as poor and a 72% as
good, with no idea which side of the expert ceiling either sits on.

## 13. Alternatives

**Execution-based grading.** State an acceptance predicate; the answer space collapses to
one equivalence class. The only clean escape, and available only where correctness is
checkable.

**Pairwise preference.** Ask which of two outputs is better rather than how good one is.
Sidesteps the level problem entirely, keeps the ranking, and is what
{{ch:ev-llm-judge}} and the arena leaderboards are built on.

**LLM-as-judge.** Replace the reference draw with a learned acceptability boundary. Better
than proximity metrics, and it inherits the judge's biases and its own ceiling.

**Rubric scoring by humans.** Score against explicit criteria rather than a reference. Raises
agreement and therefore the ceiling, and costs annotation time per item rather than per
dataset.

**Behavioural test suites.** Capability-by-capability minimum functionality tests rather than
one aggregate score. Trades a single number for a matrix, which is harder to report and much
harder to game.

## 14. Evaluation

Estimate $|A|$ for your top three task types by double-writing open-ended answers. Report the
scale factor $r/|A|$ alongside every reference-based number you publish.

Publish annotator agreement alongside any metric validated against human labels, and compute
the implied ceiling from it.

Recompute your headline metric under a continuous alternative and compare the two curves over
the last four model versions. If the story differs, you have this chapter's problem.

Test your metric's ranking stability against verbosity: score the same model with a
"be more thorough" instruction and see whether it drops.

For any emergence claim in your own results, vary the answer length and report whether the
threshold moves.

## 15. Advanced Concepts

The uniformity assumption in the reference-sampling model is generous in one direction and
harsh in another. Models do not sample uniformly from the acceptable set — they concentrate
on high-probability answers, and so do human annotators, so the match rate is higher than
$1/|A|$ for tasks with a strongly peaked answer distribution. That correction shrinks the
penalty for conventional tasks and leaves it intact for open-ended ones, since the whole
character of an open-ended task is a flat answer distribution. The direction is worth
knowing: **the model overstates the penalty exactly where the penalty is least important.**

The attenuation formula $r_{\max} = \sqrt{\rho}$ assumes annotator errors are independent
of the item, which they are not. Some items are genuinely ambiguous and every annotator
disagrees about them; others are clear and everyone agrees. That means the reliability is
itself a mixture, and the correct ceiling is higher for the clear stratum and lower for the
ambiguous one. A metric validated on the pooled set is being graded against a mixture whose
composition it cannot see — which is
{{eq:biased-sampling-distorts-composition}} from {{ch:ops-observability}} reappearing inside
the label set rather than the trace set. The practical remedy is to report agreement and
correlation by stratum, which almost no evaluation harness supports.

There is a deeper point underneath the metric-choice result that {{cite:schaeffer2023mirage}}
states carefully and readers often overstate. The finding is not that emergent abilities do
not exist; it is that a specific class of reported ones is a metric artefact. A genuinely
discontinuous underlying capability would show up under a continuous metric too. So the
correct inference from a threshold is *not* "this is an artefact" but "this is
unidentifiable from this metric" — and the way to resolve it is to re-measure continuously,
which costs nothing because the log-probabilities were computed anyway.

Finally, the two problems in this chapter have opposite remedies, which is why teams tend to
solve one and worsen the other. The remedy for metric-induced thresholds is a *softer*
metric — partial credit, continuous scoring, per-token accuracy. The remedy for the
reference-sampling penalty is a *harder* one — an acceptance predicate, an execution test, a
judge with a boundary. A team that softens its metric to recover a derivative has usually
moved further from measuring acceptability, and a team that hardens it to measure
acceptability has usually lost the derivative. Owning both explicitly is the only way out,
and it is why {{ch:ev-framework}} treats an evaluation framework as a *set* of instruments
rather than a scoring function.

## 16. Connection to Previous Chapters

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} said there is no
instrument for semantic failure. This chapter says why: the quantity has no ground truth to
instrument against, only a space of acceptable values from which references are drawn.

{{eq:evaluation-sets-decay-silently}} and {{eq:refresh-beats-growth}} from
{{ch:ops-prompt-versioning}} addressed whether the set still represents production. This
chapter addresses the prior question of whether the score means anything even when it does.

{{eq:uniform-sampling-misses-rare-failures}} from {{ch:ops-observability}} governs how
evaluation items are selected, and its bias compounds with the label-noise stratification in
{{sec:15-advanced-concepts}}.

{{cite:hendrycks2020mmlu}}'s calibration finding — that models frequently do not know when
they are wrong — is the empirical companion to this chapter's argument, and it is the part of
that paper that gets dropped when the accuracy number is quoted.

## 17. Exercises

1. Take a capability curve of your own and score it under exact match at three answer
   lengths. Where does each report the capability emerging?

2. Estimate $|A|$ for one of your open-ended tasks by having two annotators answer the same
   twenty items independently. What is the implied scale factor on your current metric?

3. Compute the metric ceiling implied by your annotator agreement. How much headroom does
   your best metric actually have?

4. Find a task in your system currently graded by comparison and rewrite it as an acceptance
   predicate. How much of the task does the predicate cover?

5. Model a non-uniform answer distribution (Zipfian over $|A|$) and recompute the
   single-reference match rate. How much does {{sec:15-advanced-concepts}}'s correction move
   it for your tasks?

## 18. Interview Questions

1. A benchmark shows a capability appearing sharply at 20B parameters. What do you ask
   first?

2. Why can a metric that reads zero not support a funding decision?

3. Our summarisation model scores 0.004 on exact match. Is it bad?

4. We doubled our reference set and the score barely moved. Why?

5. Our metric has correlated with human ratings at 0.71 for a year and will not improve.
   What are the two explanations and how do you tell them apart?

6. When is a reference-based metric's ranking trustworthy, and when does it invert?

## 19. Research Questions

1. How large is $|A|$ empirically for common production tasks, and how peaked is the answer
   distribution over it?

2. How much of the reported emergence literature survives re-measurement under continuous
   metrics beyond the cases {{cite:schaeffer2023mirage}} tested?

3. Can annotator reliability be estimated per item cheaply enough to stratify metric
   validation in practice?

4. For which task families can an acceptance predicate be written, and what fraction of the
   acceptable set does a realistic predicate actually admit?

## 20. Chapter Summary

Evaluating AI systems is hard for two reasons, and neither is complexity.

**The metric is a choice and below a performance level it is the finding.** One smooth
capability scored by exact match on a five-token answer reads **0.0200 at 3B and 0.9139 at
180B**; the same capability "emerges" at **3.0B** with two-token answers and **20.0B** with
twelve ({{eq:metric-choice-manufactures-the-finding}}). {{cite:schaeffer2023mirage}}
demonstrated this on real model families and then manufactured the effect on demand.

The cost is decisions, not charts. Extrapolating one scale step, the continuous metric errs
**14%** and the discontinuous one **48%**, worst at the low end
({{eq:discontinuity-hides-progress}}) — and across five generations of compounding progress,
exact match reports `no capability` four times.

**And most tasks have no ground truth, only a draw from a space of acceptable answers.** A
single-reference metric marks **99.4%** of correct summaries and **99.99%** of correct emails
wrong ({{eq:reference-scoring-penalises-valid-answers}}). Five references cover **2.8%** of a
summarisation space; a hundred cover **0.67%** of an email space. Only execution-based
grading escapes, because a test defines the acceptable set rather than sampling it.

Even the human judgement that would settle it is bounded. At **81%** annotator agreement the
metric ceiling is **0.79** ({{eq:agreement-caps-measurable-quality}}), and below **74%** it
sits under the metric's current correlation — so improvements read as regressions.

The two problems have opposite remedies, which is why teams solve one and worsen the other:
thresholds want a softer metric and reference penalties want a harder one. That tension is
not resolvable inside a single score, and pretending otherwise is what produces an
evaluation that is trusted, stable, and about something else.

Carry forward: **choose the metric for its derivative when tracking and its predicate when
accepting**, and **a reference is a draw, not the truth**.

## 21. Further Reading

- {{cite:schaeffer2023mirage}} — the demonstration that metric choice manufactures
  emergence, including on tasks chosen for the purpose.
- {{cite:hendrycks2020mmlu}} — the canonical broad benchmark, and a calibration result more
  durable than the accuracy figure it is quoted for.
- {{cite:rein2023gpqa}} — expert and non-expert human baselines measured under the same
  conditions as the models, which is what gives a score a scale.
- {{cite:chen2021humaneval}} — functional correctness as the grading standard, and the gap
  between one sample and a hundred.
- {{cite:jimenez2023swebench}} — real issues graded by real test suites, the cleanest
  large-scale instance of an acceptance predicate.
