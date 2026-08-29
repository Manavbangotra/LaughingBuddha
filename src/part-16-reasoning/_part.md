---
id: part-16-intro
status: final
---

## What this part is for

{{part:15}} made a model run. This part is about making it *think*, and it is the
part of the book where the gap between what a technique is claimed to do and what
it measurably does is widest.

**The hazard here is that "reasoning" is a word applied to any system that answers
problems which look like they need reasoning.** That is not a definition, and a
part organised around the word teaches nothing. Organised around *mechanisms* —
what intermediate tokens buy, what a selector can recover, what a critic can
compute about its own output — it teaches something that survives the next model
release.

> **The rule adopted for this part: every claim is a measurement, and every
> listing reports what it found rather than what it was built to show.** Five of
> the fourteen listings in this part contradicted the hypothesis they were written
> for. Those are the most useful pages here, and they are left as they came out.

## The organising idea

**Every chapter answers one question: what is the binding constraint, and can you
buy your way past it?**

```text
   CHAPTER                   WHAT IT BUYS               WHAT THEN BINDS
   ───────────────────────   ────────────────────────   ─────────────────────────
   146 (the measurement)     a test that separates      nothing you can buy
   147 intermediate tokens   serial computation         per-step reliability
   148 more samples          coverage                   the selector
   149 aggregation           some of the coverage       the generator's mode
   150 supervision           a better selector          the annotation budget
   151 an executable check   ALL of the coverage        the specification
   152 (the measurement)     a number you can trust     the rendering you chose
```

The through-line, and it is the reason the chapters are in this order:
**everything in this part is bounded by verification, and verification is the
component nobody budgets for.** Sampling produces candidates a mediocre verifier
cannot cash in. Voting is a verifier made of the generator's own mode. Reflection
is a verifier made of the generator itself. Reward models are verifiers trained on
a signal that may be biased. A test suite is a verifier that is exact and
incomplete. The subject of {{part:16}} is not reasoning. It is checking.

**And a second through-line emerged that was not planned.** In chapter after
chapter, the property that decided the outcome was not the quality of a component
but the *shape of its errors*:

| Chapter | The quoted property | What actually decided it |
|---|---|---|
| {{ch:rsn-cot}} | chain-of-thought helps | whether the task has serial depth |
| {{ch:rsn-test-time-compute}} | sample budget | whether errors are systematic |
| {{ch:rsn-self-consistency}} | critic accuracy | critic–solver error covariance |
| {{ch:rsn-supervision}} | process vs outcome | the lucky-chain rate |
| {{ch:rsn-tool-assisted}} | using a tool | how many boundary crossings |
| {{ch:rsn-benchmarks}} | the score | the variance nobody reports |

**That is not a coincidence about reasoning.** It is what happens when a field
measures components in isolation and deploys them in composition. Accuracy is a
marginal; every result in this part turned on a conditional.

## Ten things worth knowing before you start

**Intermediate tokens buy serial steps, not intelligence.** A forward pass through
$L$ layers performs $L$ sequential operations; emitting a token and reading it
back buys another $L$. {{cite:merrill2024cotexpressive}} proved the stratification
— logarithmic, linear and polynomial token budgets land in different complexity
classes — and {{ch:rsn-cot}} measures the consequence: direct models are **100%
inside the trained range of step counts and scatter 10.7–56.3% outside it**, while
the same model with chain-of-thought is 100% at every length.

**And the chain is not the reason for the answer.** {{ch:rsn-cot}}'s second listing
builds two heads with two objectives and no term tying them. The shortcut carries
**53.3% of the answer head's weight, and zeroing it leaves accuracy at 100%** —
the model learned the correct computation and the shortcut *overrides* it. Stated
reason quality stays flat at ~28% while accuracy goes 100% → 6.3%.

**Sampling and selecting are different purchases.** A 256× budget took coverage
from **9.4% to 89.2%** while the majority vote went **9.4% to 8.8%**. A control
generator matched to the same single-sample accuracy, differing only in having
random rather than systematic errors, voted **9.9% to 99.6%**. Same accuracy, same
task, a 90-point swing from error *shape*.

**Systematic error caps coverage too.** The same comparison: 89.2% against 100%.
It defeats the generator and the selector at once, and no budget clears it.

**Self-consistency is argmax versus sum.** Greedy decoding follows the most likely
path; voting returns the answer with the most total mass
({{cite:wang2023selfconsistency}}). {{ch:rsn-self-consistency}} measures **+8.4
points over greedy and +20.7 over a temperature-matched single sample** — and
finds the optimal temperature for the vote is not the optimal temperature for one
answer.

**Reflection is self-consistency, sequentially and for more money.** Two critics
with **identical confusion matrices** produced different outcomes, so the variable
is covariance, not competence. The self-critic accepted a correct proposal
**10.6% of the time on problems whose modal answer was wrong** — inverted exactly
where it was needed — and revising toward one's own critique reached **36.5%
against a majority vote of 36.8%**.

**Outcome supervision is biased, not noisy.** A fourfold increase in outcome labels
moved accuracy **+0.6 points** against process supervision's **+3.1**: the outcome
model had learned everything its signal contained. And the advantage of process
supervision **reverses** when a wrong derivation rarely reaches a right answer,
which reconciles {{cite:uesato2022process}} with {{cite:lightman2023verify}}.

**An executable check changes the shape of the curve, not its height.** A learned
verifier flattened at **72.3%** over a 128× budget; a complete check tracked
coverage to **100%**. Then the constraint moves: an incomplete specification caps
accuracy at **66%, 37% or 22%** while the pass rate goes to **100%** — a green run
on every problem while shipping defects 78% of the time.

**A tool is a boundary, and crossings are what cost.** The same tool called per
step **lost at every chain length** (0.97× down to 0.38×); called once per problem
it crossed over at k=5 and reached **1.34×**. The parse cost is paid once per
crossing; the translation advantage compounds per step.

**A benchmark score is a draw.** Two models whose true abilities differ by 1.2
points produced a measured gap with standard deviation 1.4, and **the worse model
won on 21.5% of renderings**. Averaging eight renderings fixes it; more trials per
item does not, because **96.7% of the variance was the wording**.

## What this part deliberately does not cover

**How reasoning models are trained.** {{cite:deepseek2025r1}}'s RL recipe is named
in {{ch:rsn-supervision}} and developed in {{part:9}}'s alignment material. This
part is about what the behaviour *is* and what it costs at inference.

**Agents.** Multi-step tool use with state, planning and environments is
{{part:17}}. {{ch:rsn-tool-assisted}} stops at a single tool and the arithmetic of
calling it.

**Serving reasoning models.** The economics are {{part:15}}'s and are used here
rather than rederived — reasoning tokens are decode-phase output tokens, and that
is what makes long traces expensive.

**Evaluation infrastructure.** {{ch:rsn-benchmarks}} is about what a score means,
not about how to run an eval harness, which is {{part:25}}'s.

## How to read it

{{ch:rsn-vs-generation}} and {{ch:rsn-cot}} are the foundation. In particular
{{eq:chain-accuracy-compounds}} and the invariance criterion are used in every
subsequent chapter, and {{eq:trace-and-answer-are-untied}} is the reason
{{ch:rsn-self-consistency}}'s reflection result comes out the way it does.

{{ch:rsn-test-time-compute}} and {{ch:rsn-self-consistency}} are a single argument
in two steps: what sampling buys, and what aggregation recovers from it. **Read
them in order.**

{{ch:rsn-supervision}} and {{ch:rsn-tool-assisted}} are the constructive half — how
to build the verifier everything else has been waiting on, and what happens when
it is exact.

{{ch:rsn-benchmarks}} closes by asking what any of the preceding numbers mean, and
it is deliberately last: it applies {{ch:rsn-vs-generation}}'s test to the
measurements the rest of the part relied on.

> **One thing to notice on a second reading**: several listings in this part
> failed to reproduce the magnitude of a published result while reproducing its
> direction — {{ch:rsn-supervision}} gets 20.7% against 19.5% on a metric where
> {{cite:uesato2022process}} reports 14.0% against 3.4%. The diagnosis is given
> each time and it is always the same shape: **a synthetic model of a component is
> a lower bound on what a well-built one does**, and the honest thing is to name
> which half of a result you have actually shown.
