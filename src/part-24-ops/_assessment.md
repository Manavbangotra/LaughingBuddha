---
id: part-24-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is a
**control audit**, because this part's rule was that a control is judged by what it bounds
rather than what it reports, and the commonest finding is that a control everyone trusts
bounds nothing. The challenge problems are open-ended. The interview section is what to
rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The lifecycle**

1. Decompose a lifecycle period into work and waiting. Why is the waiting share so hard to
   measure, and what does that predict about which one gets optimised?
2. State {{eq:lifecycle-period-is-wait-not-work}}. If work is 18% of the calendar, what does
   halving team throughput do to the period?
3. Fifteen changes are in flight and a live defect is 53 days old. Explain why this makes
   the loop period a *diagnostic* property rather than a throughput one.
4. Rework makes expected effort 2.08× a clean pass and lands almost entirely on one stage
   with a rework probability of zero. Explain how both facts are true at once.
5. Moving detection from canary to evaluation is worth 1.03× and moving it to a zero-return
   stage is worth 1.23×. State {{eq:shift-to-shorter-return-not-earlier}} and say what is
   wrong with "shift left" as usually phrased.

**Versioning**

6. Why is reproducibility a product over artefact coverage rather than a sum? What does
   that imply about a programme that covers nine of ten artefacts?
7. Application code is versioned 99% of the time and the retrieval corpus 12%, while the
   corpus changes output 88% of the time. What does the product come to, and why?
8. State {{eq:partial-coverage-buys-little}}. Half the effort buys 10.30%. Why is the value
   curve convex?
9. The same artefact list justified as incident tooling has a *concave* value curve. What
   changed, and what does that mean for how a backlog should be argued for?
10. The candidate space is 66,960 combinations and diagnosis cost is its logarithm. Why does
    each artefact pinned pay independently under this objective and not under the other?
11. The corpus is first by exposure and sixth by payback. Which ordering should a team with
    no reproducibility programme follow, and why?

**Deployment**

12. Derive why a canary's detection time for a semantic signal is inversely proportional to
    its share. Which term does the canary size divide?
13. State {{eq:exposure-is-invariant-to-canary-size}}. Why do share and duration cancel, and
    what does a small canary actually limit?
14. The optimum canary is 20% for the modelled failure, 50% for subtle regressions and 10%
    for obvious ones. What quantity determines which regime you are in?
15. A rollback recovers 10% of a bad deploy's damage. Enumerate the categories that make up
    the other 90% and say which are recoverable at all.
16. State {{eq:reversibility-is-a-design-property}}. Give two design decisions that change
    the 31% unrecoverable share.
17. Why do a small canary and a rollback compose *badly* — 87% permanent damage at 1%
    against 46% at 50%?

**Observability**

18. Why does standard tracing resolve only 14% of investigations for a semantic failure?
    Name the four field classes it records and what they have in common.
19. `verifier score` resolves 17% of investigations in 40 bytes; `retrieved doc text`
    resolves 31% in 46,000. What does that say about ranking fields by size?
20. State {{eq:cheap-fields-carry-most-attribution}}. Why are the densest fields also the
    privacy-conservative choice?
21. Uniform sampling captures exactly the sampling rate's share of every failure mode. Why
    is that a tautology, and why is it still the thing teams get wrong?
22. Biasing toward flagged requests captures 22× more failures and inverts the ranking.
    State {{eq:biased-sampling-distorts-composition}} and give the operating rule that
    follows.

**Prompt and evaluation-set versioning**

23. Application code escapes at 8.3% through five gates; a prompt escapes at 100% through
    zero. What does the model assume about the two defect rates before gating, and why does
    that assumption matter to the argument?
24. Prompts are 63% of changes and 88% of escaped defects. Derive the ratio and explain what
    it is attributable to.
25. State {{eq:format-check-is-the-cheapest-gate}}. What exactly does a format check assert,
    and why does it work when golden-output testing does not?
26. An evaluation set at 3.5% weekly drift covers 16% of traffic after a year. Why does its
    reported pass rate stay at 93%?
27. State {{eq:refresh-beats-growth}}. Why does quadrupling the set not substitute for
    refreshing it? Answer in terms of variance and bias.
28. Rolling replacement of 25% monthly holds 75% coverage for a quarter of the cost of full
    regeneration. Name the second advantage that does not appear in the cost table.

**Agent tracing**

29. One engineer localises 14 of 3,780 failing traces a day. Why does the chapter call human
    triage a sampling strategy rather than partial coverage?
30. State {{eq:triage-capacity-is-the-binding-constraint}}. Twenty-five engineers plus full
    automated coverage leave 79.8% untriaged. Which term dominates, and why does neither
    channel rescue the other?
31. Automated triage is 11% accurate. Argue both that it is worth running and that it is
    dangerous, and say what decides between the two.
32. State {{eq:structure-improves-both-channels}}. Why is trace structure the only variable
    in the model with this property?
33. The cause of an agent failure sits 2.7 steps back on average. Why does cost grow
    *superlinearly* in that distance rather than linearly?
34. Per-step correctness checks catch 27%. Explain the mechanism in one sentence, and
    connect it to {{eq:semantic-failure-has-no-instrument}}.
35. Step boundaries and tool arguments take triage from 26.4 to 16.9 minutes for 1.5 units of
    effort. What do those two fields have in common that the expensive ones do not?
36. Replay localises in 11 minutes, better than a raw trace. State
    {{eq:record-beats-replay}} and say why recording is nonetheless the primary path.

**Governance**

37. State {{eq:budget-overrun-is-set-by-feedback-delay}}. Where does the budget limit appear
    in the expression, and what follows?
38. Why is a fast approximate spend meter a better control than an exact billing export?
    Name the two jobs and say why one instrument cannot do both.
39. After fixing detection, 81% of the remaining incident loss is collateral. State
    {{eq:attribution-precedes-control}} and explain why detection and attribution are
    sequential rather than alternative.
40. The mean per-request agent cost sits at the 84th percentile. Derive why, from the
    stopping-time distribution and the growth of cost with step index.
41. A slower dependency, a prompt edit and a harder customer segment move monthly spend 51%.
    Why does no cost-control system have an opinion about any of them?
42. A step limit does not remove the cost tail. What does it do instead, and what shows this
    in the percentile table?
43. State {{eq:per-request-cap-beats-aggregate-budget}}. Why is the ratio of spend removed to
    successes lost so favourable, and what assumption is that resting on?
44. An 8× cap implies \$4.10 per success forgone. What would you compare that number
    against, and where does the comparison figure come from?

## Assignment: a control audit

Take an AI system you operate or can instrument. Produce a written audit with six sections.

**1. The control inventory.** List every control your team believes bounds something —
budgets, canaries, alerts, evaluation gates, rate limits, step limits, dashboards. For each,
write one sentence stating what it *bounds*, not what it reports. Mark every row where the
honest answer is "nothing".

**2. The four measurements.** Compute, from your own system: your lifecycle period and its
work share; your reproducibility product across the artefact list in
{{ch:ops-versioning}}; the share of investigations your current telemetry resolves; and your
per-request cost distribution's mean-to-median ratio. Each is an afternoon. State which of
the four you already knew.

**3. The delay audit.** For each control in section 1 that does bound something, measure its
feedback delay — from the event to the first signal to the first possible action. Multiply
by the relevant rate to get what the control permits. Following {{ch:ops-governance}}'s
method, this is the number that matters and it is almost never on the runbook.

**4. Two corrections.** Pick the two controls with the worst permitted-loss figures and
compute what fixing each would buy, in the units your organisation uses. Rank by gain per
week of work. At least one of them should be a measurement or format change rather than a
capacity increase; if neither is, say why your system is the exception.

**5. The attribution test.** Pick a plausible incident — a spend spike, a quality
regression, a latency shift — and walk through what your current instrumentation would let
you attribute it to, step by step, using only fields you actually retain. Stop at the first
question you cannot answer.

**6. What you could not measure.** Every quantity this audit needed that your telemetry does
not retain. As in the audits for {{part:22}} and {{part:23}}, this is the most valuable
section, and the reason to do the exercise before an incident rather than during one.

Length: six to ten pages and a spreadsheet.

## Challenge problems

**A. One result in four currencies.** {{eq:rework-cost-is-set-by-detection-lateness}},
{{eq:detection-time-sets-the-blast-radius}}, {{eq:cause-distance-drives-triage-cost}} and
{{eq:budget-overrun-is-set-by-feedback-delay}} are arguably the same claim. Write the
general form, derive each of the four as a special case, and identify what the general form
predicts that no individual chapter states.

**B. Correlated gates.** Every model in this part assumes independence — between gates,
between recorded fields, between detectors. Take three of them, model realistic correlation,
and determine which of the part's recommendations survive and which reverse.

**C. The mid-run stuck detector.** {{ch:ops-governance}} argues the stuck mode is detectable
during a run, and {{ch:ops-agent-tracing}} lists the fields such a detector would consume.
Specify the detector: its inputs, its trigger, its false-positive cost. Estimate how much
earlier it fires than a cost cap and what it replaces.

**D. Stratified refresh and stratified fidelity.** {{ch:ops-prompt-versioning}} suggests
refreshing high-stakes evaluation slices more often; {{ch:ops-agent-tracing}} suggests
outcome-triggered trace fidelity. Both are biased sampling justified by purpose. Design a
single stratification policy serving both, and state precisely which downstream statistics
it invalidates.

**E. Reversibility as a design budget.** {{ch:ops-deployment}} finds 31% of deploy damage
unrecoverable. Treat reversibility as a budget to be allocated across derived stores, caches,
notifications and written records. What does the optimal allocation look like, and what does
it cost against the 69% baseline?

**F. The economics of the trace format.** {{ch:ops-agent-tracing}} finds a format change
worth more than tripling the triage team. Build the full business case — instrumentation
cost, storage cost, retention policy, the triage capacity gained, and the automated-channel
gain — and find the retention period that maximises net value.

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "We shifted testing left and nothing improved." — Return trip, not detection point, with
   the 1.03× number.
2. "Everything is in version control." — The artefact list, then the product, then the 0.27%.
3. "We roll out to 1% first, so the blast radius is small." — Detection time divides by
   share, exposure is invariant, and the 93,855.
4. "We rolled it back." — What rollback restores, then the 10%, then which categories are
   permanent.
5. "We have full tracing." — Control flow versus payload, then the 14%, then the four dense
   fields.
6. "Our evaluation suite has passed at 93% for two years." — Coverage decay, then why the
   reported number cannot move.
7. "We alarm on every failed tool call." — The causing step succeeded, then the 27%.
8. "We need more engineers to triage agent failures." — Capacity arithmetic, then minutes per
   trace, then the format change.
9. "Our AI budget was exceeded by 30%." — Loop delay times burn rate, then attribution, then
   what the budget bounded.
10. "Our mean cost per request is 2.7× the median." — The mean is a tail statistic, then the
    per-request cap.

The pattern across all ten: **ask what the control bounds, then measure its delay, then
check whether the quantity it reports is the quantity that binds.** That is the part in one
sentence and the question worth having ready.
