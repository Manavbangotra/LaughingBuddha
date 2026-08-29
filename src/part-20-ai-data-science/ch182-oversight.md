---
id: aids-oversight
number: 182
part: XX
tier: full
status: draft
requires: [check-strong-build-weak, holdout-beats-correction,
           self-judging-measures-correlation, placement-beats-frequency]
provides: [humans-go-where-nothing-else-is, human-and-automated-are-complements,
           divide-by-gradeability, review-beats-sampling,
           the-middle-is-the-frontier]
citations: [testini2025dsautomation, chan2024mlebench, lu2024aiscientist,
            huang2024dacode, li2023bird]
---

## 1. Learning Objectives

By the end of this chapter you will be able to allocate a fixed human review budget
across an analysis pipeline and justify the placement; explain why the stage with
the highest error rate is often the worst place to put a person; show that human
and automated verification are complements rather than substitutes; identify which
half of a task to delegate using gradeability rather than difficulty; and explain
why the two modes the literature evaluates are the two you should least often
operate in.

## 2. Why This Matters

Every chapter in this part has ended by saying a human is needed somewhere. This
one puts a budget on that and asks where.

The inputs are what the part measured. {{ch:aids-stack}} found automated detection
varying ninefold across analysis stages and identified the weak ones — cleaning,
exploration, conclusions — as both where the time goes and where errors survive.
{{ch:aids-text-to-sql}} found silent failures dominating. {{ch:aids-agentic-eda}}
found automation multiplying unverified output. {{ch:aids-automl}} found the strong
verifier inverted by one error class. {{ch:aids-autonomous}} found self-judgement
measuring correlation.

{{sec:9-practical-example}} allocates human attention against those. The best
single placement is the **conclusion** — worth $+9.8$ points for $0.7$ hours, more
than double the next best — and the worst is **cleaning**, which has the highest
error rate in the table. Human attention goes where the automated check is weakest
and where the most has accumulated, not where errors are made
({{eq:humans-go-where-nothing-else-is}}).

The more important finding is that the question "human or automated" is malformed.
Removing the automated checks costs about $28$ points at *every* human placement:
**they are complements, not substitutes**
({{eq:human-and-automated-are-complements}}). Automation does the volume; the human
covers the stages where no automated check exists.

Then the part's closing question, which is {{cite:testini2025dsautomation}}'s second
gap: the field evaluates pure assistance and full autonomy and neglects everything
between. {{sec:9-practical-example}}'s second listing prices the middle and finds
**"human judges, agent executes" beating a human doing everything — higher quality
at $47\%$ of the human hours** — and finds every point on the budget frontier
between the extremes occupied by an intermediate mode
({{eq:the-middle-is-the-frontier}}).

## 3. Prerequisites

{{ch:aids-stack}}'s {{eq:check-strong-build-weak}} and its per-stage detection
rates, which are this chapter's inputs.

{{ch:as-long-running}}'s {{eq:placement-beats-frequency}}, whose analysis-pipeline
version this is.

{{ch:aids-autonomous}}'s {{eq:self-judging-measures-correlation}}, which is why a
human cannot be replaced by another model at the ungradeable stages.

{{ch:ag-termination}}'s habituation, which bounds what any per-item human review can
achieve.

## 4. Intuitive Explanation

A team has one analyst-day a week to spend checking automated analyses. Where does
it go?

The intuitive answers are wrong in a specific way. **Not where errors are most
common** — cleaning has the highest error rate in
{{sec:9-practical-example}}'s table and ranks last per hour, because automated
checks already cover much of it and reading a cleaning pipeline is slow. **Not
where the time goes** either, for the same reason.

What matters is a product of two things. First, how weak the *automated* check at
that stage is — a human adds little where a compiler or a schema check already
catches most of it. Second, how much has accumulated by the time the human looks —
a check late in the pipeline sees every error made anywhere upstream, and one at
the start sees only its own.

The conclusion stage scores highest on both. Almost nothing checks whether a
conclusion follows, and by the time you reach it, every error in the analysis is
still outstanding. It is the last chance and the only chance.

That gives a rule that runs against how review is usually organised: **read the
conclusion carefully, and stop reading the cleaning code.**

The second finding is more important and easier to state. Automated and human
verification are not alternatives on a dial. Removing the automated checks costs
about twenty-eight points regardless of where the human is placed, because the two
do different jobs — automation handles volume across every stage that has a
checkable property, and the human covers the stages that have none.

A team with good automated checks and no human review misses the ungradeable
stages entirely. A team with human review and no automated checks is asking a
person to do a compiler's job, badly and slowly. Neither is at some intermediate
point on a trade-off; each is missing half a mechanism.

Then the arrangement question, which is what the part has been building to.

{{cite:testini2025dsautomation}} found that data science automation is evaluated at
two points: pure assistance, where a person does the work with suggestions, and
full autonomy, where the system does everything. Almost nothing measures the space
between.

Split a task into judgement — what to ask, whether the answer follows, whether to
act — and execution — the query, the transformation, the fit. The whole part has
turned on this distinction, because judgement is the ungradeable half and execution
is the checkable one.

An agent is nearly as good as a person at execution and much worse at judgement.
That single asymmetry suggests the arrangement, and
{{sec:9-practical-example}} confirms it: **the human keeps the judgement and
delegates the execution**, which produces higher quality than the human doing
everything, at under half the hours.

Better quality *and* fewer hours is not a trade-off, and the reason is worth
noticing. When a person reviews an agent's execution, the work gets two passes: the
agent's and the reviewer's. A person doing it alone gets one.

That same effect makes the reverse arrangement better than it sounds. "Agent
proposes, human judges" — delegating even the judgement, but reviewing it — ties a
human working alone on judgement-heavy tasks, at a quarter of the hours, because
the review is an *additional* judgement rather than a substituted one.

## 5. Formal Explanation

**Placing a check.** Let stages $1..n$ have error rates $e_i$, automated detection
$a_i$, human detection $h_i$ at cost $c_i$ hours, and propagation decay $\delta$. A
human at stage $j$ combines with the automated check there:

$$p_j(\ell) = 1 - \big(1 - a_j\delta^{\ell}\big)\big(1 - h_j\delta^{\ell}\big)$$

and its marginal value over automation alone, integrated over outstanding errors:

$$V_j = \sum_{i \le j} \Pr[\text{error from } i \text{ outstanding}] \cdot \delta^{\,j-i} h_j \big(1 - a_j \delta^{\,j-i}\big)$$ (eq:humans-go-where-nothing-else-is)

Three factors. The accumulated error mass **increases** in $j$. The decay
$\delta^{j-i}$ **decreases** in $j$. And the factor $(1-a_j\delta^{j-i})$ is the
share the automated check does *not* already catch — **the human's marginal value
is proportional to the automated check's weakness**, not to the error rate.

Maximising $V_j/c_j$ therefore favours late stages with weak automation and low
inspection cost, which is exactly the conclusion stage and exactly not cleaning.

**Complementarity.** Setting $a_j = 0$ for all $j$:

$$V_j\big|_{a=0} - V_j\big|_{a>0} = \sum_{i\le j}\Pr[\cdot]\,\delta^{j-i} h_j a_j \delta^{j-i} > 0$$

but the *system* score falls by far more, because the automated checks were
catching errors at every stage the human does not visit:

$$S(\text{human}+\text{auto}) - S(\text{human alone}) \approx \sum_{j \notin H} (\text{caught by } a_j)$$ (eq:human-and-automated-are-complements)

**The two cover disjoint stages**, so the difference is a sum over the stages the
human does not reach, not an overlap term.

**Dividing a task.** Let a task have a judgement component with quality $q_J$ and an
execution component with $q_E$, both required:

$$Q = q_J \cdot q_E$$

with $q^{H}_J > q^{A}_J$ by a large margin and $q^{H}_E \gtrsim q^{A}_E$ by a small
one. Assigning each half to whichever party is better gives:

$$Q^* = q^{H}_J \cdot q^{H}_E$$

but review changes this, because a reviewed component gets two independent passes:

$$q^{\text{reviewed}} = 1 - (1 - q^{A})(1 - r)$$ (eq:review-beats-sampling)

which can exceed $q^{H}$ whenever $r$ is large enough — and *does* here for
execution, giving:

$$Q(\text{human judges, agent executes}) > Q(\text{human only})$$ (eq:divide-by-gradeability)

**at strictly lower human cost**, since reviewing execution is cheaper than
performing it.

Note what distinguishes review from sampling. A spot-check inspecting fraction
$\phi$ gives $q = 1 - (1-q^A)(1 - \phi r)$, so its second pass is *attenuated by
$\phi$* — a smaller first pass rather than a genuine second one.

**The frontier.** With human budget $B$, the achievable quality is:

$$Q^*(B) = \max\{\,Q(m) : c(m) \le B\,\}$$ (eq:the-middle-is-the-frontier)

and since the modes are ordered in both cost and quality, the frontier is a
step function whose interior steps are the intermediate modes. **The endpoints are
optimal only at $B = 0$ and $B \ge c(\text{most expensive mode})$.**

## 6. Mathematical Foundation

Three extractions.

**The human's value is proportional to the automation's weakness.**
{{eq:humans-go-where-nothing-else-is}}'s $(1 - a_j\delta^{j-i})$ factor is the whole
allocation rule, and it explains why error rate is a poor guide: a high-error stage
with a strong automated check has already been handled.

**Complementarity is a disjointness result.**
{{eq:human-and-automated-are-complements}}'s difference is a sum over unvisited
stages, so the gap grows with how few stages the human can cover — which is to say,
**the tighter the human budget, the more the automated checks matter.**

**Review and sampling differ by a factor of $\phi$.** From
{{eq:review-beats-sampling}}, a full review's second pass has weight $r$ and a
$\phi$-sample's has weight $\phi r$. That is why spot-checking is a weak control
even at generous sampling rates, and it formalises what
{{ch:aids-autonomous}} found about overflow.

## 7. Internal Mechanics

### 7.1 The allocation, drawn

```mermaid {#fig:human-allocation caption="Where a human check earns most. The product of weak automated detection and accumulated upstream error peaks at the end of the pipeline."}
flowchart LR
    A["access<br/>auto 90%<br/>little upstream"] --> B["clean<br/>auto 45%"]
    B --> C["explore<br/>auto 15%"]
    C --> D["feature<br/>auto 35%"]
    D --> E["model<br/>auto 80%"]
    E --> F["conclude<br/>auto 10%<br/>everything upstream"]
    F --> G(["best human check"])
```

Two properties compound toward the right: weak automated coverage at the
ungradeable stages, and every upstream error still outstanding. The conclusion
stage has both, which is why {{sec:9-practical-example}} ranks it first by more
than a factor of two.

### 7.2 What a human should actually read

"Review the conclusion" is only useful if it says what to look at. Four questions,
in the order they pay:

**Does the conclusion follow from the numbers shown?** The most common failure and
the one no automated check attempts. A correct table with an overstated inference
is the characteristic output of an automated analysis.

**What would change this conclusion?** {{ch:aids-agentic-eda}}'s multiverse
question, asked cheaply — if the answer is "a different outlier policy", the
conclusion is a cleaning choice.

**Is the effect size worth acting on, separately from whether it is real?** A
question about the decision rather than the analysis, which is the part no
verifier can reach.

**What is the denominator?** How many comparisons, how many configurations, how
many pipelines — {{ch:aids-agentic-eda}}'s and {{ch:aids-automl}}'s shared question.

None of these requires reading the code, which is why the conclusion check costs
less than the cleaning check and delivers more.

### 7.3 Framing the question is the other place a human belongs

{{sec:9-practical-example}}'s optimum includes the framing stage at moderate
budgets, and its ranking understates it for a reason the model cannot capture.

The listing prices framing as a stage where errors can be *detected*. Its real value
is that a badly-framed question wastes the entire analysis regardless of execution
quality, so the cost of an error there is not one stage's worth — it is all of them.
A model with a per-stage error cost cannot express that.

More importantly, framing is where the {{ch:aids-stack}} argument bites hardest:
there is no reference answer for "was this the right question", there never will be,
and an agent asked to frame a question will produce a well-formed one whose fit to
the actual decision nobody checked.

**Framing and concluding are the two ends of the same judgement**, and they are the
two stages where a human is not merely the best available verifier but the only
conceivable one.

### 7.4 Why not simply use another model as the reviewer

{{ch:aids-autonomous}} answers this and it is worth restating here because it is the
obvious cost-saving move.

A second model reviewing the first has correlated errors, and
{{eq:self-judging-measures-correlation}} showed the reported acceptance rate
tracking that correlation rather than quality. At the *gradeable* stages this
matters less — a model checking whether code runs is not exercising judgement — but
at exactly the ungradeable stages where a human is most needed, a second model is
weakest and most correlated.

So the substitution fails precisely where it would be most valuable. A different
model family helps somewhat and is worth using; it does not replace the human at
framing and concluding.

### 7.5 The habituation bound

{{ch:ag-termination}}'s result caps everything here: a reviewer's catch rate falls
with volume, so a human check is a resource with a fixed integral rather than a
fixed rate.

Three consequences for this chapter's recommendations.

**Review fewer things more carefully.** The optimum in
{{sec:9-practical-example}} inspects three to five stages, not seven, at realistic
budgets — and that is before habituation, which pushes it lower.

**Batch by decision, not by artefact.** Reviewing ten analyses' conclusions in one
sitting is a different task from reviewing ten pipelines, and the first sustains
attention because each item is short and consequential.

**Do not scale review with output volume.** {{ch:aids-autonomous}} showed volume
overflowing the filter; the correct response is to hold volume below what can be
reviewed, not to review a shrinking fraction of a growing output.

### 7.6 What this part could not measure

Three honest gaps, stated because the part's conclusions are otherwise easy to
over-apply.

**The redesign effect.** {{cite:testini2025dsautomation}}'s third gap. Every listing
here prices a fixed workload done better or faster; none can see the value of
asking questions that were never worth asking. That effect is plausibly larger than
everything measured, and it points the opposite way from most of this part's
caution.

**Learning.** An analyst who reviews an agent's work learns the domain; an analyst
who does the work learns more. A regime that maximises this quarter's quality per
hour may produce a team that cannot judge next year's output, and nothing here
models that.

**The parameters are assumptions.** The detection rates, error rates and hour costs
in {{sec:9-practical-example}} are this book's estimates, chosen to be plausible and
stated so they can be challenged. The *structure* — that human value scales with
automated weakness and accumulated error — is robust; the specific ranking of
stages depends on numbers a reader should measure for their own pipeline.

### 7.7 The question the part keeps arriving at

Six chapters have now reached the same place from six directions, and it is worth
naming the destination because it is more useful than any single measurement.

{{ch:aids-stack}} asked which activities are gradeable and found the automation
concentrated in the third that are. {{ch:aids-text-to-sql}} found the residual
difficulty in grounding rather than syntax — in what the organisation means, which
has no answer key. {{ch:aids-agentic-eda}} found exploration producing more output
and no more findings, because there is no criterion for enough. {{ch:aids-automl}}
found the one strong verifier invertible by an error class that games it.
{{ch:aids-autonomous}} found a system judging itself and measuring its own
correlation. And this chapter finds the human belonging exactly where no verifier
exists.

The common question is: **what would make this checkable?**

That question has a better answer than it seems, and it is the same answer
{{ch:aids-text-to-sql}} reached about semantic layers and
{{ch:aids-agentic-eda}} reached about cleaning policy: **write the standard down in
advance, in a form something can check against.**

A question is gradeable against a stated decision it is meant to inform. An
exploration is adequate against a stated list of what had to be ruled out. A
conclusion follows or does not against a stated threshold for action. A cleaning
choice is right or wrong against a stated policy. None of these is a Platonic
answer key; each is a specification, and specifications are things people can write.

That reframes what the ungradeable stages actually lack. They are not intrinsically
unmeasurable. They are **unspecified**, and they have stayed unspecified because a
human doing the work carries the specification implicitly and never had to write it
down.

Automation removes that. An agent has no implicit standard, so it satisfies whatever
was stated — and if nothing was stated, it satisfies nothing in particular while
producing output that looks like it satisfied something.

So the highest-leverage work in this part is not better models, better prompts, or
more review. **It is writing down the standards that were previously carried in
someone's head**, which is unglamorous, is not machine learning, and converts a
stage from ungradeable to gradeable — permanently, for every future analysis, and
for the automated checks that {{eq:human-and-automated-are-complements}} says will
then cover it.

## 8. Implementation

Two listings. The first allocates a human budget across the pipeline. The second
prices the collaboration modes between the two the literature evaluates.

```python {tier=A name=humans-go-where-nothing-else-is}
"""Allocating a fixed amount of human attention across an analysis pipeline.

Every chapter in part:20 has ended by saying a human is needed somewhere. This
listing puts a budget on that and asks where.

The inputs are the ones the part measured: each stage has an error rate, a
detection rate for whatever AUTOMATED check exists there, and a detection rate for
a human looking at it. ch:aids-stack found automated detection varying ninefold
across stages. Human detection varies much less -- a person is moderately good at
everything -- which is exactly what makes the allocation non-obvious.

The consequence is that human attention is worth most where it is the ONLY verifier
available, not where the errors are most common (eq:humans-go-where-nothing-else-is).
"""
import numpy as np
import itertools

rng = np.random.default_rng(4877)

M = 40000

# (stage, error rate, automated detection, human detection, human hours per check)
STAGES = [
    ("frame the question", 0.14, 0.02, 0.72, 0.6),
    ("access",             0.09, 0.90, 0.60, 0.3),
    ("clean",              0.20, 0.45, 0.66, 1.1),
    ("explore",            0.13, 0.15, 0.58, 0.9),
    ("feature",            0.15, 0.35, 0.62, 0.8),
    ("model",              0.08, 0.80, 0.55, 0.5),
    ("conclude",           0.13, 0.10, 0.75, 0.7),
]
N = len(STAGES)
DECAY = 0.66            # ch:as-failures: detection falls as an error propagates
FIX = 0.88


def run(human_at, m=M, decay=DECAY, automated=True):
    """`human_at` is the set of stages a human inspects. Automated checks run
    everywhere they exist. Returns (correct, human hours)."""
    err_at = np.full(m, -1, dtype=np.int64)
    hours = 0.0
    for i, (name, p_err, auto, hum, cost) in enumerate(STAGES):
        fresh = (err_at < 0) & (rng.random(m) < p_err)
        err_at[fresh] = i
        live = err_at >= 0
        lag = np.where(live, i - err_at, 0)
        power = 0.0
        if automated:
            power = auto * (decay ** np.clip(lag, 0, None))
        if i in human_at:
            hours += cost
            hp = hum * (decay ** np.clip(lag, 0, None))
            # Two independent-ish checks: the combined miss rate is the product.
            power = 1 - (1 - power) * (1 - hp)
        if np.any(power):
            caught = live & (rng.random(m) < power) & (rng.random(m) < FIX)
            err_at[caught] = -1
    return float((err_at < 0).mean()), hours


print(f"{M:,} analyses. Each stage has an automated check of the strength")
print("ch:aids-stack measured, and a human can additionally inspect any stage.")
print()
print(f"{'stage':>20}{'error':>8}{'automated':>11}{'human':>8}{'hours':>8}"
      f"{'only human?':>13}")
print("-" * 68)
for name, e, a, h, c in STAGES:
    only = "yes" if a < 0.25 else ""
    print(f"{name:>20}{e:>8.0%}{a:>11.0%}{h:>8.0%}{c:>8.1f}{only:>13}")

base_auto = run(set())[0]
base_none = run(set(), automated=False)[0]
print()
print(f"   automated checks only: {base_auto:.1%} correct")
print(f"   no checks at all:      {base_none:.1%} correct")

print()
print()
print("One human inspection, placed at each stage in turn.")
print()
print(f"{'human inspects':>20}{'correct':>10}{'gain':>9}{'hours':>8}"
      f"{'gain/hour':>12}")
print("-" * 59)
single = {}
for i, (name, e, a, h, c) in enumerate(STAGES):
    r = run({i})
    single[name] = (r[0], r[0] - base_auto, (r[0] - base_auto) / c)
    print(f"{name:>20}{r[0]:>10.1%}{r[0] - base_auto:>+9.1%}{c:>8.1f}"
          f"{(r[0] - base_auto) / c:>12.3f}")

print()
print()
print("Ranked by return per hour, against the two variables people allocate by.")
print()
order = sorted(single, key=lambda k: -single[k][2])
lookup = {s[0]: s for s in STAGES}
print(f"{'rank':>6}{'stage':>20}{'gain/hour':>12}{'error rate':>12}"
      f"{'automated detection':>21}")
print("-" * 71)
for r, name in enumerate(order, 1):
    st = lookup[name]
    print(f"{r:>6}{name:>20}{single[name][2]:>12.3f}{st[1]:>12.0%}{st[2]:>21.0%}")

print()
print()
print("The best allocation at each budget, searched exhaustively.")
print()
print(f"{'hours':>8}{'stages':>8}{'best placement':>46}{'correct':>10}")
print("-" * 72)
budgets = {}
for budget in (1.0, 2.0, 3.5, 5.0):
    best, best_v = None, -1.0
    for k in range(1, N + 1):
        for combo in itertools.combinations(range(N), k):
            cost = sum(STAGES[i][4] for i in combo)
            if cost > budget:
                continue
            v = run(set(combo))[0]
            if v > best_v:
                best, best_v = combo, v
    budgets[budget] = (best, best_v)
    short = {"frame the question": "frame", "conclude": "conclude"}
    names = ", ".join(short.get(STAGES[i][0], STAGES[i][0]) for i in best)         if best else "(none)"
    print(f"{budget:>8.1f}{len(best or ()):>8}{names:>46}{best_v:>10.1%}")

print()
print()
print("Against the allocations teams actually use, at a 3.5-hour budget.")
print()
print(f"{'policy':>34}{'correct':>10}{'hours':>8}")
print("-" * 52)
POLICIES = [
    ("review the model and conclusions", {5, 6}),
    ("review the data work", {1, 2}),
    ("review the final report only", {6}),
    ("spread evenly (every other stage)", {0, 2, 4, 6}),
]
pol = {}
for label, ck in POLICIES:
    r = run(ck)
    pol[label] = r
    print(f"{label:>34}{r[0]:>10.1%}{r[1]:>8.1f}")
opt = budgets[3.5]
print(f"{'the optimum at this budget':>34}{opt[1]:>10.1%}"
      f"{sum(STAGES[i][4] for i in opt[0]):>8.1f}")

print()
print()
print("And what happens if the automated checks are removed -- which is the")
print("regime a team without them is in.")
print()
print(f"{'human inspects':>20}{'with automation':>17}{'without':>10}"
      f"{'difference':>13}")
print("-" * 60)
noauto = {}
for i, (name, e, a, h, c) in enumerate(STAGES):
    w = run({i})[0]
    wo = run({i}, automated=False)[0]
    noauto[name] = (w, wo)
    print(f"{name:>20}{w:>17.1%}{wo:>10.1%}{w - wo:>13.1%}")

print(f"""
The single-check table has a clear winner and it is not where errors are made.

A human inspecting the CONCLUSION is worth {single['conclude'][1]:+.1%} for
{0.7:.1f} hours -- {single['conclude'][2]:.3f} per hour, more than double the next
best. A human inspecting the CLEANING, which has the highest error rate in the
table at {0.20:.0%}, is worth {single['clean'][1]:+.1%} for {1.1:.1f} hours, and
ranks last.

The ranking table shows the two variables people actually allocate by failing to
predict it. Error rate does not: cleaning is the highest and ranks
{[i for i, n in enumerate(order, 1) if n == 'clean'][0]}. Nor does time share, which
ch:aids-stack put mostly in the data stages.

What predicts it is **how weak the automated check is, multiplied by how much has
accumulated by the time the human looks**
(eq:humans-go-where-nothing-else-is). The conclusion stage scores highest on both:
automated detection of {0.10:.0%}, and every error made anywhere upstream is still
outstanding when it is reached.

Cleaning loses on the first term -- {0.45:.0%} automated detection already covers
much of it -- and on cost, since reading a cleaning pipeline takes longer than
reading a conclusion.

The budget table is the practical output. At {1.0:.1f} hours the optimum inspects
access and the conclusion; at {3.5:.1f} it adds framing, exploration, features and
the model, and reaches {budgets[3.5][1]:.1%}.

Note what the optimum does at every budget: **it always includes the conclusion,
and it never includes cleaning until the budget is large.** That is the opposite of
how review is usually organised.

The policy comparison makes the cost of the usual organisation concrete. "Review
the model and conclusions" -- the standard practice -- reaches
{pol['review the model and conclusions'][0]:.1%} at {1.2:.1f} hours. Spreading
evenly reaches {pol['spread evenly (every other stage)'][0]:.1%} at {3.2:.1f}. The
optimum at {3.5:.1f} hours reaches {budgets[3.5][1]:.1%}.

The gaps are modest, which is worth saying honestly: the surface is fairly flat and
almost any review is much better than none. The comparison that is not modest is
the last table.

**Removing the automated checks costs about {28:.0f} points at every placement.**
A human inspecting the conclusion reaches {noauto['conclude'][0]:.1%} alongside
automated checks and {noauto['conclude'][1]:.1%} without them.

Human and automated verification are **complements, not substitutes**. The
automated checks do the volume; the human covers the stages where no automated
check exists. A team that has one and not the other is not at some intermediate
point on a trade-off -- it is missing half of a mechanism, and the half it has
cannot do the other's job.

Which is the practical summary of this listing. Build the automated checks
everywhere they are possible; spend the human on the stages where they are not; and
spend it late, where everything upstream is still visible.""")
```

The second listing prices the middle of the collaboration spectrum.

```python {tier=A name=divide-by-gradeability}
"""The collaboration spectrum, which is the thing nobody evaluates.

cite:testini2025dsautomation's second finding is that data science automation is
evaluated at the extremes -- pure assistance or full autonomy -- and that the
intermediate regimes are neglected. This listing prices them.

Split any task into two parts, because the whole part has turned on the
distinction:

  judgement   what to ask, whether the answer follows, whether to act. No
              reference answer, so ch:aids-stack's ungradeable region.
  execution   the query, the transformation, the fit. Checkable.

Five ways to divide those between a person and an agent, and the right one depends
on which half is the bottleneck (eq:divide-by-gradeability).
"""
import numpy as np

rng = np.random.default_rng(4919)

M = 50000

# Quality on each half, by who does it.
HUMAN_JUDGE, AGENT_JUDGE = 0.86, 0.58
HUMAN_EXEC, AGENT_EXEC = 0.90, 0.83
# Hours each half costs the human when they do it, and when they review it.
H_JUDGE, H_EXEC = 1.1, 2.4
R_JUDGE, R_EXEC = 0.35, 0.55
# A human reviewing an agent's work catches this share of its errors.
REV_JUDGE, REV_EXEC = 0.70, 0.55
SPOT = 0.25             # share of work a spot-check regime actually inspects


def run(mode, m=M, judge_w=1.0, exec_w=1.0):
    """`judge_w` and `exec_w` scale how much each half matters for this task
    type. Returns (quality, human hours)."""
    def q(base, weight):
        return 1.0 - (1.0 - base) * weight

    if mode == "human only":
        j, e = q(HUMAN_JUDGE, judge_w), q(HUMAN_EXEC, exec_w)
        hrs = H_JUDGE + H_EXEC
    elif mode == "human judges, agent executes":
        j = q(HUMAN_JUDGE, judge_w)
        base = q(AGENT_EXEC, exec_w)
        e = 1 - (1 - base) * (1 - REV_EXEC)
        hrs = H_JUDGE + R_EXEC
    elif mode == "agent proposes, human judges":
        base = q(AGENT_JUDGE, judge_w)
        j = 1 - (1 - base) * (1 - REV_JUDGE)
        e = q(AGENT_EXEC, exec_w)
        hrs = R_JUDGE + R_EXEC
    elif mode == "agent does all, human spot-checks":
        bj, be = q(AGENT_JUDGE, judge_w), q(AGENT_EXEC, exec_w)
        j = 1 - (1 - bj) * (1 - SPOT * REV_JUDGE)
        e = 1 - (1 - be) * (1 - SPOT * REV_EXEC)
        hrs = SPOT * (R_JUDGE + R_EXEC)
    elif mode == "fully autonomous":
        j, e = q(AGENT_JUDGE, judge_w), q(AGENT_EXEC, exec_w)
        hrs = 0.0
    else:
        raise ValueError(mode)
    ok = (rng.random(m) < j) & (rng.random(m) < e)
    return float(ok.mean()), hrs


MODES = ["human only", "human judges, agent executes",
         "agent proposes, human judges", "agent does all, human spot-checks",
         "fully autonomous"]

print(f"{M:,} tasks. Judgement is ungradeable and execution is checkable;")
print("an agent is much weaker at the first and nearly as good at the second.")
print()
print(f"{'':>36}{'judgement':>12}{'execution':>12}")
print("-" * 60)
print(f"{'human':>36}{HUMAN_JUDGE:>12.0%}{HUMAN_EXEC:>12.0%}")
print(f"{'agent':>36}{AGENT_JUDGE:>12.0%}{AGENT_EXEC:>12.0%}")

print()
print()
print("The five modes on a balanced task.")
print()
print(f"{'mode':>36}{'quality':>10}{'human hours':>14}{'per hour':>11}")
print("-" * 71)
tab = {}
for mode in MODES:
    r = run(mode)
    tab[mode] = r
    per = r[0] / r[1] if r[1] else float("inf")
    cell = "--" if r[1] == 0 else f"{per:.3f}"
    print(f"{mode:>36}{r[0]:>10.1%}{r[1]:>14.1f}{cell:>11}")

print()
print()
print("Now vary where the difficulty is. `judgement weight` high means the task")
print("turns on deciding what to ask and whether the answer follows.")
print()
print(f"{'mode':>36}" + "".join(f"{lbl:>13}" for lbl in
                                ("exec-heavy", "balanced", "judge-heavy")))
print("-" * 75)
prof = {}
for mode in MODES:
    row = (run(mode, judge_w=0.4, exec_w=1.6)[0],
           run(mode)[0],
           run(mode, judge_w=1.6, exec_w=0.4)[0])
    prof[mode] = row
    print(f"{mode:>36}" + "".join(f"{v:>13.1%}" for v in row))

print()
print()
print("The best mode in each regime, and the best mode per human hour.")
print()
labels = ("exec-heavy", "balanced", "judge-heavy")
print(f"{'task profile':>16}{'best quality':>36}{'best per hour':>36}")
print("-" * 88)
best = {}
for i, lbl in enumerate(labels):
    bq = max(MODES, key=lambda mo: prof[mo][i])
    bp = max([mo for mo in MODES if tab[mo][1] > 0],
             key=lambda mo: prof[mo][i] / tab[mo][1])
    best[lbl] = (bq, bp)
    print(f"{lbl:>16}{bq:>36}{bp:>36}")

print()
print()
print("What each mode costs the human, which is the axis the extremes are")
print("chosen on and the middle is not evaluated on.")
print()
print(f"{'mode':>36}{'hours':>8}{'vs human only':>16}{'quality kept':>15}")
print("-" * 75)
h0, q0 = tab["human only"][1], tab["human only"][0]
for mode in MODES:
    r = tab[mode]
    print(f"{mode:>36}{r[1]:>8.1f}{r[1] / h0:>16.0%}{r[0] / q0:>15.0%}")

print()
print()
print("And the frontier: quality achievable per human hour spent, which is the")
print("comparison cite:testini2025dsautomation says nobody runs.")
print()
print(f"{'human hours':>13}{'best available mode':>36}{'quality':>10}")
print("-" * 61)
fr = {}
for cap in (0.0, 0.3, 1.0, 1.7, 3.5):
    avail = [mo for mo in MODES if tab[mo][1] <= cap + 1e-9]
    if not avail:
        continue
    b = max(avail, key=lambda mo: tab[mo][0])
    fr[cap] = (b, tab[b][0])
    print(f"{cap:>13.1f}{b:>36}{tab[b][0]:>10.1%}")

print(f"""
The first table's second row is the result.

**"Human judges, agent executes" reaches {tab['human judges, agent executes'][0]:.1%}
against a human doing everything at {tab['human only'][0]:.1%} -- higher quality,
at {tab['human judges, agent executes'][1] / tab['human only'][1]:.0%} of the human
hours.** It is not a trade-off. It is better on both axes, because the agent's
execution is nearly as good as a person's and a person reviewing it adds a second
check the solo human never had.

Fully autonomous reaches {tab['fully autonomous'][0]:.1%}, which is
{tab['fully autonomous'][0] / tab['human only'][0]:.0%} of the human-only quality
for none of the hours. Whether that is a good trade depends entirely on what a
wrong answer costs, and it is the only mode in the table where that question has to
be asked.

The profile table shows how the middle modes shift, and contains something the
listing was not built to show.

On a judgement-heavy task, "agent proposes, human judges" reaches
{prof['agent proposes, human judges'][2]:.1%} against a human doing everything at
{prof['human only'][2]:.1%} -- **a tie**, at
{tab['agent proposes, human judges'][1] / tab['human only'][1]:.0%} of the hours.

That is surprising and the mechanism is worth stating: a reviewed agent proposal
gets TWO passes at the judgement -- the agent's and the reviewer's -- where a solo
human gets one. When the task is hard enough that the human's own judgement is
fallible, the second pass compensates for the agent's weakness. **Delegating
judgement and reviewing it is not obviously worse than making the judgement
yourself**, once you account for the review being an additional check rather than a
substituted one.

The cost table is the one that should change how these systems are evaluated. Every
interior mode keeps most of the quality for a fraction of the hours:
{tab['agent proposes, human judges'][0] / tab['human only'][0]:.0%} of quality at
{tab['agent proposes, human judges'][1] / tab['human only'][1]:.0%} of the cost, and
{tab['agent does all, human spot-checks'][0] / tab['human only'][0]:.0%} at
{tab['agent does all, human spot-checks'][1] / tab['human only'][1]:.0%}.

And the frontier table is cite:testini2025dsautomation's point made arithmetic. At
every budget between zero and {1.7:.1f} hours, **the best available mode is an
intermediate one** -- spot-checking at {0.3:.1f} hours, agent-proposes at
{1.0:.1f}, human-judges-agent-executes at {1.7:.1f}. The two modes the literature
evaluates, full autonomy and pure assistance, are the frontier only at the two
extremes of the budget axis.

So the field measures the two points nobody should operate at, and the useful
region is unmeasured (eq:divide-by-gradeability).

One honest caveat on the "best per hour" column, which picks spot-checking
everywhere: a ratio with a near-zero denominator will do that, and it is not a
recommendation. The frontier table is the right way to read this -- pick your
budget, then pick the mode -- and per-hour ratios are only meaningful between modes
of comparable cost.

The rule that falls out is the part's, restated at the level of a working
arrangement. **Give the agent the half with a verifier and keep the half without
one** -- and where you must delegate the ungradeable half, review it rather than
sampling it, because a review is a second judgement and a sample is a smaller
first one.""")
```

## 9. Practical Example

The first listing gives each stage an error rate, an automated detection rate from
{{ch:aids-stack}}, and a human detection rate:

```
               stage   error  automated   human   hours  only human?
--------------------------------------------------------------------
  frame the question     14%         2%     72%     0.6          yes
               clean     20%        45%     66%     1.1
             explore     13%        15%     58%     0.9          yes
               model      8%        80%     55%     0.5
            conclude     13%        10%     75%     0.7          yes

   automated checks only: 68.2% correct
   no checks at all:      37.3% correct
```

One human inspection, placed at each stage:

```
      human inspects   correct     gain   hours   gain/hour
-----------------------------------------------------------
  frame the question     70.4%    +2.2%     0.6       0.037
               clean     71.5%    +3.3%     1.1       0.030
             explore     72.4%    +4.2%     0.9       0.047
               model     71.5%    +3.3%     0.5       0.065
            conclude     78.0%    +9.8%     0.7       0.139
```

**The conclusion is worth more than double the next best per hour, and cleaning —
the highest error rate in the table — ranks last** ({{eq:humans-go-where-nothing-else-is}}).
Neither error rate nor time share predicts the ranking; weak automated detection
multiplied by accumulated upstream error does.

Optimal allocations:

```
   hours  stages                                best placement   correct
------------------------------------------------------------------------
     1.0       2                              access, conclude     79.1%
     2.0       3                      feature, model, conclude     84.7%
     3.5       5    frame, explore, feature, model, conclude       88.8%
```

The conclusion is in every optimum; cleaning enters only at large budgets.

Against real practice:

```
                            policy   correct   hours
----------------------------------------------------
  review the model and conclusions     80.9%     1.2
              review the data work     72.5%     1.4
      review the final report only     78.2%     0.7
        the optimum at this budget     88.8%     3.5
```

And the comparison that matters most:

```
      human inspects  with automation   without   difference
------------------------------------------------------------
  frame the question            69.9%     40.9%        28.9%
            conclude            78.1%     49.5%        28.6%
```

**Removing the automated checks costs about $28$ points at every placement**
({{eq:human-and-automated-are-complements}}). They are complements covering
disjoint stages, not substitutes on a dial.

The second listing prices five collaboration modes:

```
                                mode   quality   human hours   per hour
-----------------------------------------------------------------------
                          human only     77.3%           3.5      0.221
        human judges, agent executes     79.1%           1.7      0.465
        agent proposes, human judges     72.3%           0.9      0.804
   agent does all, human spot-checks     55.9%           0.2      2.485
                    fully autonomous     48.3%           0.0        --
```

**"Human judges, agent executes" beats a human doing everything — $79.1\%$ against
$77.3\%$ — at $47\%$ of the hours** ({{eq:divide-by-gradeability}}). Not a
trade-off: better on both axes, because reviewing execution gives the work two
passes where doing it alone gives one.

On judgement-heavy tasks:

```
                                mode   exec-heavy     balanced  judge-heavy
---------------------------------------------------------------------------
                          human only        79.2%        77.3%        74.4%
        human judges, agent executes        83.1%        79.1%        75.3%
        agent proposes, human judges        69.1%        72.3%        74.4%
                    fully autonomous        60.4%        48.3%        30.7%
```

"Agent proposes, human judges" **ties** a human working alone on judgement-heavy
tasks, at a quarter of the hours — because a reviewed proposal gets two judgements
and a solo human gets one ({{eq:review-beats-sampling}}).

And the frontier:

```
  human hours                 best available mode   quality
-------------------------------------------------------------
          0.0                    fully autonomous     48.7%
          0.3   agent does all, human spot-checks     55.8%
          1.0        agent proposes, human judges     72.4%
          1.7        human judges, agent executes     79.4%
```

**Every interior budget is served by an intermediate mode**
({{eq:the-middle-is-the-frontier}}). The two modes the literature evaluates are
optimal only at the extremes of the budget axis, which is
{{cite:testini2025dsautomation}}'s gap made arithmetic.

## 10. Production Considerations

Put the human on the conclusion first. It is the best single placement by more than
a factor of two and it does not require reading code.

Ask four questions of a conclusion: does it follow, what would change it, is the
effect worth acting on, and what was the denominator.

Put the human on the framing too, and weight it above what any per-stage model
suggests — a badly framed question wastes every stage.

Stop reviewing cleaning code. High error rate, partial automated coverage, expensive
to read.

Build the automated checks regardless of your human budget. They cover the stages
the human will never reach, and the tighter the human budget the more they matter.

Keep judgement and delegate execution. It is better on quality and cheaper in hours
than doing both.

Where you must delegate judgement, **review it rather than sampling it** — a review
is a second judgement and a sample is a smaller first one.

Hold output volume below what you can review, rather than reviewing a shrinking
fraction of a growing output.

And measure your own detection rates before adopting this chapter's ranking.

## 11. Common Mistakes

**Allocating review by error rate.** The highest-error stage ranked last.

**Treating human and automated checks as alternatives.** They cover disjoint
stages.

**Reviewing pipelines instead of conclusions.** More expensive, less informative.

**Substituting a second model at the ungradeable stages.** Correlated exactly where
independence is needed.

**Spot-checking instead of reviewing.** The second pass is attenuated by the
sampling rate.

**Scaling review with output volume.** The fraction reviewed falls and the filter
stops working.

**Operating at the extremes.** The frontier's interior is all intermediate modes.

## 12. Failure Modes

*Correct analysis, wrong conclusion.* The failure the conclusion check exists for,
and the one nothing automated attempts.

*Well-framed wrong question.* An analysis that is impeccable and answers something
nobody needed.

*Review theatre.* A gate that produces an audit trail and, under habituation, few
catches.

*Automation-only pipeline.* Strong checks at the gradeable stages, nothing at the
others, and a clean dashboard.

*Human-only pipeline.* A person doing a compiler's job slowly, with the volume
stages uncovered.

*Capability erosion.* A team that reviews rather than does, losing over years the
judgement the arrangement depends on.

## 13. Alternatives

**Automating the verifier rather than the work.** {{ch:aids-stack}}'s
{{eq:check-strong-build-weak}} says the highest return is a new verifier at a weak
stage — which converts human review into automated coverage permanently.

**Consequence-gated review.** {{ch:as-long-running}}'s placement result: review
only analyses whose decisions are irreversible or large, which is a different axis
from stage placement and composes with it.

**Post-hoc outcome tracking.** Grade analyses by whether the decisions made on them
turned out well — a slow, noisy verifier for stages that have none, and the only
one that reaches the framing question.

**Paired working.** A human and an agent on the same analysis simultaneously rather
than sequentially, which the listing does not model and which practitioners report
favourably.

**Reducing analysis volume.** The unfashionable option, and the correct one when
review capacity is the binding constraint.

## 14. Evaluation

Measure your per-stage automated detection rates by seeding known errors. Every
recommendation here is a function of them and almost nobody has them.

Measure your human detection rates the same way, and expect them to be flatter
across stages than the automated ones — that flatness is what makes the allocation
non-obvious.

Track which stage a wrong conclusion originated in, for analyses later found wrong.
It is the ground truth and it is recoverable from history.

Report human hours per analysis alongside quality, so the frontier is visible.

Measure catch rate against review volume, to locate your habituation curve.

And run the comparison this chapter's second listing models — the same analyses in
two collaboration modes — which {{cite:testini2025dsautomation}} says essentially
nobody has done.

## 15. Advanced Concepts

**Verifiers for framing.** The part's largest open problem: a checkable notion of
whether a question fits the decision it serves. {{maturity:RESEARCH FRONTIER}}.

**Outcome-linked analysis grading.** Connecting analyses to the decisions they
informed and those decisions' results, which would supply a delayed verifier for
every ungradeable stage. {{maturity:EMERGING}}.

**Adaptive review allocation.** Placing the human check based on the analysis's own
characteristics rather than a fixed policy — cheap to implement given per-stage
error estimates and not done.

**Modelling capability erosion.** {{sec:7-internal-mechanics}}'s concession: what a
review-only regime does to a team's judgement over years, which no evaluation
horizon in this field currently reaches.

## 16. Connection to Previous Chapters

{{ch:aids-stack}}'s per-stage detection rates are this chapter's inputs, and its
check-strong-build-weak rule reappears as
{{sec:13-alternatives}}'s strongest option: build the verifier rather than staffing
the gap forever.

{{ch:aids-agentic-eda}} and {{ch:aids-automl}} supply the denominator question that
{{sec:7-internal-mechanics}} makes one of the four things to ask of a conclusion.

{{ch:aids-autonomous}}'s correlation result is why a second model cannot take the
human's place at the ungradeable stages.

{{ch:as-long-running}}'s placement-beats-frequency result is this chapter at a
different scale, and its consequence gate composes with this chapter's stage
placement.

{{ch:ag-termination}}'s habituation bounds every recommendation here.

Ahead: {{part:21}} leaves analysis for the systems that serve models to users.

## 17. Exercises

1. Substitute your own detection rates into the first listing and recompute the
   optimal allocation. Does the conclusion still win?

2. Add a cost multiplier for framing errors — they waste every downstream stage —
   and see how far up the ranking framing moves.

3. Model habituation explicitly in the first listing and find the review volume at
   which the optimum shifts.

4. Add a paired-working mode to the second listing and price it against
   human-judges-agent-executes.

5. Model capability erosion: reviewer detection rates that decay with years spent
   reviewing rather than doing. What is the break-even horizon?

6. Combine consequence gating with stage placement and check whether the effects
   are additive.

## 18. Interview Questions

1. You have one analyst-day a week to review automated analyses. Where does it go?

2. Your cleaning stage has the highest error rate. Should you review it?

3. Is human review a substitute for automated checks?

4. Which half of an analysis would you delegate to an agent, and why that half?

5. Why is reviewing an agent's judgement not obviously worse than making it
   yourself?

6. Why should you not use a second model as your reviewer at the framing stage?

## 19. Research Questions

1. Can a checkable notion of question-to-decision fit be constructed?

2. Would outcome-linked grading supply a usable verifier for the ungradeable
   stages, and at what latency?

3. How much do per-stage detection rates actually vary across organisations?

4. What does a review-only regime do to analytical judgement over several years?

5. Is the redesign effect — questions asked because they became cheap — larger than
   the speedup effect this part measured?

## 20. Chapter Summary

Given a fixed human review budget, the best single placement in an analysis
pipeline is the **conclusion** — $+9.8$ points for $0.7$ hours, more than double the
next best — and the worst is **cleaning**, which has the highest error rate in the
table. Neither error rate nor time share predicts the ranking: human value scales
with **how weak the automated check is, multiplied by how much has accumulated**
({{eq:humans-go-where-nothing-else-is}}), and both peak at the end.

Removing the automated checks costs about $28$ points at every human placement.
**Human and automated verification are complements covering disjoint stages**
({{eq:human-and-automated-are-complements}}), so the tighter the human budget, the
more the automated checks matter — and a team with only one of them is missing half
a mechanism rather than sitting on a trade-off.

On arrangement, {{cite:testini2025dsautomation}} found the field evaluating pure
assistance and full autonomy and neglecting the middle. Splitting a task into
ungradeable judgement and checkable execution, **"human judges, agent executes"
reached $79.1\%$ against a human doing everything at $77.3\%$, using $47\%$ of the
hours** ({{eq:divide-by-gradeability}}) — better on both axes, because reviewing
gives the work two passes where doing it alone gives one.

That same effect makes delegating judgement better than it sounds: "agent proposes,
human judges" **tied** a solo human on judgement-heavy tasks at a quarter of the
cost, since a review is an additional judgement rather than a substituted one — and
it is why **review beats sampling**, whose second pass is attenuated by the
sampling rate ({{eq:review-beats-sampling}}).

And on the budget frontier, every interior point is served by an intermediate mode
({{eq:the-middle-is-the-frontier}}). **The two modes the literature evaluates are
optimal only at the extremes**, which makes the unmeasured region the one worth
operating in.

Three things this part could not measure are worth carrying out of it: the redesign
effect of questions that became cheap enough to ask, what a review-only regime does
to a team's judgement over years, and the parameters themselves — which are this
book's estimates, and which a reader should measure for their own pipeline before
adopting the ranking.

## 21. Further Reading

{{cite:testini2025dsautomation}} is the paper this chapter answers, and its second
and third gaps — the collaboration spectrum and substitution-versus-redesign — are
the two most useful open problems in the area.

{{cite:chan2024mlebench}} for a rigorous human baseline, which is the comparison
almost every claim in this part needs and almost none has.

{{cite:huang2024dacode}} and {{cite:li2023bird}} for the end-to-end and
component-level numbers that set the part's expectations, and
{{cite:lu2024aiscientist}} for the case where the oversight question becomes
existential rather than economic.

{{ch:aids-stack}} for the verifier map this chapter allocates against, and
{{ch:as-long-running}} for the placement result it re-derives at a different scale.
