---
id: aise-autonomy
number: 188
part: XXI
tier: full
status: draft
requires: [automatability-is-verify-times-reverse, visible-half-is-what-is-reported,
           divide-by-gradeability, scaffold-beats-model-improvement]
provides: [no-mode-dominates, uniform-policy-is-costly,
           human-hours-is-not-total-cost, autonomy-is-an-environment-property,
           containment-is-the-largest-step]
citations: [becker2025devproductivity, chan2024mlebench, wang2025solvedcorrectly,
            testini2025dsautomation, jimenez2023swebench]
---

## 1. Learning Objectives

By the end of this chapter you will be able to place a software task on the
autonomy spectrum using verifiability and reversibility rather than appetite; show
why every uniform autonomy policy is substantially worse than per-task selection, and
why full autonomy applied uniformly can be worse than manual work; distinguish
minimising human hours from minimising cost and say where they diverge; reconcile the
measured productivity evidence with the model rather than choosing between them; and
enumerate the environment prerequisites that determine how much autonomy is safe.

## 2. Why This Matters

This part has measured six things and they compose into one question: **how much of
software engineering should be done without a human in the loop?**

{{cite:testini2025dsautomation}} found the analogous question in data science
evaluated only at its extremes. Software has the same gap and a sharper way to close
it, because {{ch:aise-cicd}} established the two properties that decide what an agent
can safely do.

{{sec:9-practical-example}} crosses six operating modes against four task types and
finds **no mode winning everywhere** ({{eq:no-mode-dominates}}). Full autonomy is best
on three tasks and worst on the fourth by a margin that swamps its wins — the single
worst cell in the table and three of the four best belong to the same mode.

Which makes the policy question sharp. Organisations adopt a *stance*, not a per-task
rule, and **every uniform policy is substantially worse than choosing per task**
({{eq:uniform-policy-is-costly}}): the best uniform stance costs $+11.2$ hours against
per-task selection, and **full autonomy applied uniformly costs $+95.2$ — worse than
doing everything manually.**

There is also a divergence worth naming. On the highest-consequence task, the mode
minimising *human hours* and the mode minimising *total cost* are different
({{eq:human-hours-is-not-total-cost}}) — and they diverge exactly where the
consequences are largest.

The second listing collects what this part found the environment must supply, and
delivers the closing argument. Today's model with none of it reaches $1.8\%$ safe
autonomy; a model $60\%$ better with none of it reaches $5.8\%$; **today's model with
all seven prerequisites reaches $18.7\%$**
({{eq:autonomy-is-an-environment-property}}). The largest single step is a rollback
path ({{eq:containment-is-the-largest-step}}).

## 3. Prerequisites

{{ch:aise-cicd}}'s {{eq:automatability-is-verify-times-reverse}} — the two properties
that order everything here.

{{ch:aise-generation}}'s {{eq:visible-half-is-what-is-reported}}, which is why the
measured evidence and the felt experience disagree.

{{ch:aids-oversight}}'s {{eq:divide-by-gradeability}}, of which this chapter is the
software instance.

{{ch:aise-swe-agents}}'s {{eq:scaffold-beats-model-improvement}}, which the second
listing extends from the loop to the environment.

## 4. Intuitive Explanation

A team decides how much to let agents do. The decision gets made once, at the level
of a policy: *we review everything agents write*, or *agents can merge to main*, or
*we only use completion*.

That framing is the mistake, and it is the mistake regardless of which answer is
chosen.

{{sec:9-practical-example}} runs six modes against four task types. Full autonomy is
the best mode for fixing a reported bug, implementing a feature, and refactoring
legacy code. It is by far the worst for designing a data model — $124$ hours against
completion's $29$ — because a data model is $24\%$ verifiable and $19\%$ reversible,
so a wrong one is neither detected nor undone.

Adopt full autonomy as a policy and the fourth task's cost swamps the other three's
savings. **The uniform policy is worse than manual work**, while every individual
automation in it looks like a success.

That is the shape of the risk with these tools, and it is not the shape people expect.
The danger is not that agents are bad at software. It is that they are good at most of
it and catastrophic at a small part, and a policy is a blunt instrument that cannot
tell the difference.

There is a second trap in the same table. On three tasks, the mode that minimises
total hours and the mode that minimises *human* hours agree. On data-model design they
do not — full autonomy minimises human involvement and completion minimises cost.

Organisations measure human hours. It is the visible number, it is what a headcount
plan contains, and it is what an adoption programme reports. **Minimising human
involvement and minimising cost are different objectives, and they diverge exactly
where the consequences are largest.**

Then the measured evidence. {{cite:becker2025devproductivity}} found experienced
developers $19\%$ slower with AI tools on mature codebases they knew well, while
estimating they had been $20\%$ faster.

It would be easy to treat that as refuting the model, and it does not.
{{sec:9-practical-example}} finds agent-with-review paying as long as it leaves the
human doing less than about the unassisted hours — the review's defect reduction
covers small increases in effort but not large ones. The study measured a setting
where hours *rose*, which places it above the threshold rather than outside the
model.

Which relocates the question usefully: not *do these tools help* but **in which
settings does the assistance exceed the friction** — and
{{ch:aise-generation}} found that answer changing sign across the plausible range.

Finally, the part's closing argument.

Every chapter here identified something the *environment* had to supply.
Reproduction. Coverage. Test independence. Fast CI. Gating. Rollback. Executable
architectural constraints. None is a model capability.

The second listing puts them together, and the comparison is stark: a model $60\%$
better than today's with none of these in place is worth less than a third of today's
model with all of them.

## 5. Formal Explanation

**Mode selection.** For task $t$ with verifiability $\nu_t$, reversibility $\rho_t$,
unassisted hours $h_t$ and error cost $e_t$, and mode $m$ with hours multiplier
$\mu_m$, agent share $\sigma_m$ and review depth $\delta_m$:

$$C(t,m) = h_t\mu_m + \underbrace{\big[p_H(1-\sigma_m) + p_A\sigma_m\big](1-\nu_t)(1-\gamma\delta_m)}_{\text{escape rate}}\; e_t(1-\kappa\rho_t)$$ (eq:no-mode-dominates)

The first term is decreasing in autonomy and the second increasing, with the second
scaled by $e_t(1-\nu_t)(1-\kappa\rho_t)$ — a task-specific factor spanning two orders
of magnitude. **The optimal $m$ is therefore a function of $t$**, and no $m$ minimises
$C$ for all $t$ unless that factor is uniform, which it is not.

**Uniform policy cost.** Writing $m^*(t) = \arg\min_m C(t,m)$:

$$\text{regret}(m) = \sum_t \big[C(t,m) - C(t,m^*(t))\big] \ge 0$$ (eq:uniform-policy-is-costly)

with equality only if one mode is optimal everywhere. Since the escape term's scale
varies by $\sim 50\times$ across tasks while the hours term varies by $\sim 3\times$,
the regret is dominated by the high-$e_t$ tasks — so **a uniform policy is priced
almost entirely by its worst cell.**

That is why full autonomy applied uniformly can exceed manual: its escape term on the
worst task alone exceeds the hours saved on all the others.

**Objective divergence.** The human-hours objective is $H(t,m) = h_t\mu_m$, minimised
by $\arg\min_m \mu_m$ — *independent of $t$*. The cost objective includes the escape
term, which is not. So:

$$\arg\min_m H(t,m) = \text{most autonomous mode, always}$$
$$\arg\min_m C(t,m) \ne \text{most autonomous mode, when } e_t(1-\nu_t)(1-\kappa\rho_t) \text{ is large}$$ (eq:human-hours-is-not-total-cost)

**Optimising human hours always recommends maximum autonomy**, which is why an
organisation measuring headcount reaches a different conclusion than one measuring
cost, and why the disagreement concentrates on the consequential tasks.

**Environment capabilities.** Let safe unsupervised completion require four
capabilities — localisation $L$, verification $V$, iteration $I$, containment $K$:

$$A = L \cdot V \cdot I^{1/2} \cdot K$$

Each prerequisite $j$ raises one capability toward one by a factor $\mu_j$, subject
to a dependency $d_j$:

$$c \leftarrow c + (1-c)\mu_j \quad\text{if } j \text{ and } d_j \text{ both present}$$ (eq:autonomy-is-an-environment-property)

The product form means all four are required — a missing one caps $A$ regardless of
the others — and the dependency structure means measuring prerequisites one at a time
returns near-zero for the contingent ones, which is
{{eq:scaffold-components-interact}} again.

**Model skill enters $L$, $V$ and $I$ multiplicatively and does not enter $K$ at
all.** Containment is purely environmental:

$$\frac{\partial A}{\partial \text{skill}} \propto K, \qquad \frac{\partial A}{\partial K} = L V I^{1/2}$$ (eq:containment-is-the-largest-step)

so a better model's benefit is *bounded by* the containment the environment supplies,
and containment is the one term no model improvement touches.

## 6. Mathematical Foundation

Three extractions.

**Uniform-policy regret is priced by the worst cell.** From
{{eq:uniform-policy-is-costly}}, the escape term's task variance dominates the hours
term's, so a policy's cost is set by its worst task rather than its average. That is
why "it works well for most of our work" is not an argument for a uniform policy — it
is a description of the cells that are not paying for it.

**The human-hours objective has no task dependence at all.**
{{eq:human-hours-is-not-total-cost}} shows $\arg\min_m H$ is the same mode for every
task. An organisation optimising that metric will therefore converge on maximum
autonomy everywhere, correctly by its own measure and wrongly by cost — and the
divergence is invisible in the metric it is using.

**Model skill cannot substitute for containment.** From
{{eq:containment-is-the-largest-step}}, $K$ multiplies the whole product and is
untouched by skill. So there is no model good enough to make an uncontained
environment safe, and the ceiling on what any model can deliver is set by a property
of the deployment.

## 7. Internal Mechanics

### 7.1 The spectrum, and where each mode belongs

```mermaid {#fig:autonomy-spectrum caption="Operating modes against the verify-times-reverse product. The right mode moves with the task, not with the team's policy."}
flowchart LR
    A["design a data model<br/>0.05"] --> M1["completion<br/>or chat"]
    B["refactor legacy<br/>0.35"] --> M2["agent with review"]
    C["implement a feature<br/>0.45"] --> M3["agent, gated"]
    D["fix a reported bug<br/>0.87"] --> M4["autonomous"]
```

The practical form is a routing rule rather than a policy: classify the task, then
choose the mode. Most of the classification is mechanical — a change to a migration
directory or a service interface is not a bug fix — which is
{{ch:aise-cicd}}'s path-based gating doing double duty.

### 7.2 Why organisations adopt uniform policies anyway

The regret in {{eq:uniform-policy-is-costly}} is large and the fix is simple, so it is
worth asking why uniform policies persist. Three reasons, none of them foolish.

**A policy is enforceable and a judgement is not.** "Agents cannot merge without
review" is checkable in CI; "use the appropriate mode for the task" is an aspiration.

**Classification requires the table.** Choosing per task needs error costs and
verifiability estimates per task type, which most teams do not have — and
{{ch:aise-cicd}}'s recommendation to build that table is the prerequisite for this
chapter's recommendation.

**The failures are asymmetric in visibility.** An over-cautious policy costs time
diffusely; an under-cautious one costs a visible incident. So policies drift toward
caution, and the cost of that drift is the $+11.2$ hours the best uniform stance
carries — real, invisible, and never attributed.

The resolution is to encode the routing rule mechanically, which converts a judgement
back into a policy. That is what makes it adoptable.

### 7.3 The metric an organisation should actually track

{{eq:human-hours-is-not-total-cost}} says the natural metric recommends maximum
autonomy unconditionally. Two better ones are available.

**Total engineering hours including rework**, which requires attributing rework to the
change that caused it. Version control makes that recoverable for defects with a
clear origin.

**Merged-without-substantive-revision rate**, which {{ch:aise-swe-agents}} proposed:
it prices correctness, fit and reviewability together and needs no new instrument.

Neither is as easy to collect as hours saved, which is why hours saved is what gets
reported — and why {{cite:becker2025devproductivity}}'s self-report gap matters beyond
the individual level. **An organisation's reported productivity gain has the same
signed bias as an individual's**, for the same reason: the visible half is the
assisted half.

### 7.4 Reconciling the measured evidence

{{cite:becker2025devproductivity}} is the strongest evidence in this part and it
deserves a careful reading, because it is used to support two claims it does not make.

**It does not show these tools do not work.** It shows one mode, in one setting,
producing a slowdown — and {{sec:9-practical-example}}'s break-even analysis places
that setting above a threshold rather than outside the model.

**It does not show developers are wrong about their work.** The $39$-point
self-report gap is fully explained by {{ch:aise-generation}}'s visible/invisible
split: the assistance is concentrated and observed, the friction is diffuse and
attributed elsewhere. The report is accurate about the half it can see.

What it does establish, and what generalises furthest, is **that the effect must be
measured rather than surveyed** — and that the setting is a first-class variable,
since sixteen experienced maintainers on five-year-familiar repositories is one cell
of a table whose entries change sign.

The honest position for a team: run the comparison on your own work, with
randomisation, on real tasks. It is a week of process and it is the only way to know
which cell you are in.

### 7.5 The seven prerequisites, in build order

{{sec:9-practical-example}}'s second listing gives an order and the order is not
obvious.

**Test coverage** first, because three other prerequisites depend on it.

**Independent tests** second — {{ch:aise-testing}}'s specification-derived suites.
Worth nothing without coverage, which is why it is second rather than first.

**Reproduction** third: {{ch:aise-repo}}'s best localiser, and it supplies the
verifier and the termination condition too.

**Fast CI** fourth, which is what makes iteration real rather than nominal.

**A rollback path** fifth, and it is the largest single step in the table at
$+6.3$ points.

**Blast-radius gating** sixth, per {{ch:aise-cicd}}.

**Executable architectural constraints** last, because they are the hardest and the
smallest — though they are also the ones that move an activity up
{{ch:aise-cicd}}'s automatability table permanently.

Note what this list does not contain: a better model, a better prompt, or a
particular agent framework.

### 7.6 Why containment is the largest step

{{eq:containment-is-the-largest-step}} explains the ordering's most surprising entry,
and it is worth stating as the part's central practical claim.

Model improvements raise localisation, verification and iteration. They do not raise
containment at all — a rollback path exists or it does not, and no amount of model
capability creates one.

So containment is both the term with the least competition for improvement and the
term that multiplies everything else. A team that builds instant rollback has raised
the ceiling on every future model as well as the current one.

This is {{ch:as-specialized}}'s reversibility result reaching its seventh appearance
in this book, and by now the pattern is unambiguous: **being able to undo a mistake is
worth more than being less likely to make one**, across agent domains, analysis
pipelines, protocol design and software delivery.

### 7.7 What the part supports, stated plainly

Six chapters of measurement land on four claims a team can act on.

**Agents are genuinely effective at the verifiable, reversible majority of software
work**, which {{sec:9-practical-example}} sizes at about two thirds of engineering
effort.

**The reported capability numbers need three corrections** — contamination, patch
verification, and coverage — landing well below the headline and still far above
{{cite:jimenez2023swebench}}'s $1.96\%$ starting point.

**The binding constraint is the environment rather than the model**, by a factor of
roughly three against a large model improvement.

**And the failure mode to design against is uniformity**: a single policy applied to
tasks whose consequences span two orders of magnitude, priced by its worst cell and
justified by its best.

### 7.8 What the models in this chapter cannot see

Both listings price a fixed workload done differently, which is the same limitation
{{ch:aids-oversight}} conceded and which is worth restating here because it cuts
against this chapter's caution rather than for it.

**The redesign effect.** If a change costs a tenth as much to attempt, some changes
become worth attempting that were not. A refactor nobody would have scheduled, a
migration deferred for three years, a test suite for a module everyone avoids —
these are not the same work done faster, and no table in this chapter contains
them. {{cite:testini2025dsautomation}} named this gap and this book has not closed
it in either domain.

**Option value.** An agent can produce three candidate implementations for the cost
of one, and choosing among artefacts you can see is a different and easier task than
specifying one in advance. That is genuinely new capability and the models here price
it as a single attempt.

**Capability change in the team.** {{ch:aids-oversight}} raised this and it applies
harder in software: an engineer who reviews rather than writes learns a codebase
differently, and a team that has delegated its debugging may be less able to debug in
three years. That is a cost the tables cannot see, and it points the opposite way
from the redesign effect.

**And the parameters are estimates.** The verifiability, reversibility and error-cost
figures here are this book's assumptions, chosen to be plausible and stated so they
can be argued with. The *structure* is robust — a multiplicative product with a
task-varying escape term produces a cliff whatever the exact values — but the
specific thresholds are not, and a team should measure its own before adopting this
chapter's routing.

The net of these is genuinely uncertain in sign. The redesign and option-value
effects are plausibly large and push toward more autonomy; the capability effect
pushes the other way and is slower to appear. **What the measurements do support is
the shape of the decision — route by task, invest in the environment, contain what
you cannot verify — rather than a particular level of adoption.**

## 8. Implementation

Two listings. The first crosses operating modes against task types. The second
enumerates what the environment must supply.

```python {tier=A name=no-mode-dominates}
"""The autonomy spectrum for software, which is ch:aids-oversight's question in a
second domain.

cite:testini2025dsautomation found data science automation evaluated only at the
extremes -- pure assistance or full autonomy -- with the middle neglected. Software
has the same gap and a sharper way to resolve it, because ch:aise-cicd established
that activities differ enormously in two properties that decide what an agent can
safely do.

This listing crosses six operating modes against four task types and asks which mode
wins where (eq:no-mode-dominates). The answer is that none dominates, and the right
mode is a function of the task's verifiability and reversibility rather than of the
team's appetite for automation.

cite:becker2025devproductivity's measured 19% slowdown is one cell of this table:
experienced developers, mature familiar code, agent-with-review.
"""
# (task type, verifiability, reversibility, human hours unassisted,
#  hours lost when an error reaches the codebase)
# The last column spans two orders of magnitude, which ch:aise-cicd measured
# and which is the reason a uniform policy cannot be right.
TASKS = [
    ("fix a reported bug",   0.92, 0.95,  3.0,   12.0),
    ("implement a feature",  0.58, 0.78,  9.0,   40.0),
    ("refactor legacy code", 0.40, 0.88,  6.0,   55.0),
    ("design a data model",  0.24, 0.19,  5.0,  620.0),
]

# (mode, human hours multiplier, agent share of the work, review depth)
MODES = [
    ("manual",                    1.00, 0.00, 0.00),
    ("completion",                0.88, 0.15, 0.90),
    ("chat-assisted",             0.74, 0.35, 0.85),
    ("agent with full review",    0.46, 0.80, 0.75),
    ("agent with gated review",   0.22, 0.90, 0.35),
    ("fully autonomous",          0.03, 1.00, 0.00),
]

P_HUMAN_WRONG = 0.15
P_AGENT_WRONG = 0.31


def outcome(task, mode):
    """Returns (human hours, defect rate, total hours including rework)."""
    _, ver, rev, base_h, err_h = task
    _, h_mult, agent_share, review = mode
    hours = base_h * h_mult
    p_wrong = P_HUMAN_WRONG * (1 - agent_share) + P_AGENT_WRONG * agent_share
    # Two chances to catch it: the automated verifier, then the human review.
    escaped = p_wrong * (1 - ver) * (1 - review * 0.72)
    # An escaped error costs less where it can be undone.
    rework = escaped * err_h * (1 - rev * 0.80)
    return hours, escaped, hours + rework


print("Six operating modes against four task types. 'Total' includes rework")
print("from errors that reach the codebase.")
print()
print(f"{'mode':>26}" + "".join(f"{t[0][:14]:>16}" for t in TASKS))
print("-" * 90)
tab = {}
for m in MODES:
    row = tuple(outcome(t, m)[2] for t in TASKS)
    tab[m[0]] = row
    print(f"{m[0]:>26}" + "".join(f"{v:>16.2f}" for v in row))

print()
print()
print("The best mode for each task, and what the second best costs.")
print()
print(f"{'task':>24}{'best mode':>26}{'total hours':>13}{'2nd best':>26}")
print("-" * 89)
best = {}
for i, t in enumerate(TASKS):
    ranked = sorted(MODES, key=lambda m: tab[m[0]][i])
    best[t[0]] = (ranked[0][0], tab[ranked[0][0]][i], ranked[1][0])
    print(f"{t[0]:>24}{ranked[0][0]:>26}{tab[ranked[0][0]][i]:>13.2f}"
          f"{ranked[1][0]:>26}")

print()
print()
print("No mode wins everywhere, and the ordering tracks verifiability times")
print("reversibility rather than anything about the mode.")
print()
print(f"{'task':>24}{'verify x reverse':>18}{'best mode':>26}")
print("-" * 68)
for t in TASKS:
    print(f"{t[0]:>24}{t[1] * t[2]:>18.2f}{best[t[0]][0]:>26}")

print()
print()
print("Human hours, separately -- because the mode that minimises total hours")
print("and the mode that minimises HUMAN hours are not the same.")
print()
print(f"{'task':>24}{'min total':>26}{'min human':>26}")
print("-" * 78)
for i, t in enumerate(TASKS):
    by_total = min(MODES, key=lambda m: outcome(t, m)[2])
    by_human = min(MODES, key=lambda m: outcome(t, m)[0])
    print(f"{t[0]:>24}{by_total[0]:>26}{by_human[0]:>26}")

print()
print()
print("Applying one mode everywhere, which is what a policy actually does.")
print()
print(f"{'uniform policy':>26}{'total hours':>14}{'human hours':>14}"
      f"{'vs best-per-task':>18}")
print("-" * 74)
opt = sum(min(outcome(t, m)[2] for m in MODES) for t in TASKS)
uni = {}
for m in MODES:
    tot = sum(outcome(t, m)[2] for t in TASKS)
    hum = sum(outcome(t, m)[0] for t in TASKS)
    uni[m[0]] = (tot, hum)
    print(f"{m[0]:>26}{tot:>14.2f}{hum:>14.2f}{tot - opt:>+18.2f}")

print()
print(f"   best mode chosen per task: {opt:.2f} total hours")
print(f"   best single uniform policy: "
      f"{min(uni.values(), key=lambda x: x[0])[0]:.2f}")

print()
print()
print("And the question the measured evidence actually poses: how much can a")
print("mode add to the surrounding work before it stops paying? This sweeps")
print("the human-hours multiplier for agent-with-review on a feature task.")
print()
print(f"{'hours multiplier':>18}{'total':>10}{'vs manual':>12}{'verdict':>10}")
print("-" * 50)
feature = TASKS[1]
man = outcome(feature, MODES[0])[2]
bk = {}
for h in (0.46, 0.75, 1.00, 1.25, 1.50):
    m = ("agent with full review", h, 0.80, 0.75)
    tot = outcome(feature, m)[2]
    bk[h] = (tot, tot / man - 1)
    print(f"{h:>18.2f}{tot:>10.2f}{tot / man - 1:>+12.0%}"
          f"{('faster' if tot < man else 'slower'):>10}")

lo, hi = 0.1, 3.0
for _ in range(60):
    mid = (lo + hi) / 2
    if outcome(feature, ("x", mid, 0.80, 0.75))[2] < man:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2
print()
print(f"   break-even multiplier: {breakeven:.2f}")
print(f"   The mode pays if it leaves the human doing less than "
      f"{breakeven:.0%} of the")
print(f"   unassisted hours. cite:becker2025devproductivity measured a setting")
print(f"   where it did not -- which locates that study above this threshold")
print(f"   rather than contradicting the model.")

print(f"""
The first table has no winning row, and that is the finding
(eq:no-mode-dominates).

Full autonomy is best on three of the four tasks and worst on the fourth by a
margin that swamps its wins: {tab['fully autonomous'][3]:.1f} hours on data-model
design against {tab['completion'][3]:.1f} for completion. The single worst cell in
the table and three of the four best belong to the same mode.

The reason is in the verify-times-reverse column. Bug fixing sits at
{0.92 * 0.95:.2f} and design at {0.24 * 0.19:.2f} -- a factor of nineteen -- and
ch:aise-cicd showed that product entering multiplicatively.

The uniform-policy table is what a real team's decision looks like, because
organisations adopt a stance rather than a per-task rule. **Every uniform policy is
substantially worse than choosing per task**: the best single stance costs
{min(uni.values(), key=lambda x: x[0])[0] - opt:+.1f} hours against per-task
selection, and full autonomy applied uniformly costs
{uni['fully autonomous'][0] - opt:+.1f} -- **worse than doing everything manually.**

That last comparison is the one worth carrying out of this chapter. A team that
adopts autonomy as a policy, rather than as a per-task decision, can end up worse
than a team that adopted nothing, and it will get there while every individual
automation looks like a success.

The human-hours table separates two objectives that are usually conflated. On
three tasks the mode minimising total hours and the mode minimising HUMAN hours
agree. On data-model design they do not: full autonomy minimises human hours and
completion minimises total.

**Minimising human involvement and minimising cost are different objectives**, and
they diverge exactly where the consequences are largest -- which is the worst place
for an organisation to be optimising the wrong one.

The last table addresses the measured evidence directly. Agent-with-review pays as
long as it leaves the human doing less than {breakeven:.0%} of the unassisted hours,
because the review's defect reduction covers a small increase in effort.
cite:becker2025devproductivity measured a setting where hours ROSE {19}%, which is
above that threshold.

So the study and this model agree rather than conflict: **the mode pays until it
adds work, and the study found a setting where it added work.** Which relocates the
question from "do these tools help" to "in which settings does the assistance
exceed the friction" -- and ch:aise-generation's setting table says the answer moves
across the plausible range.""")
```

The second listing asks what has to be true before autonomy pays.

```python {tier=A name=autonomy-is-an-environment-property}
"""What has to be true before autonomy pays, which is not a fact about the model.

Every chapter in part:21 identified something the ENVIRONMENT has to supply.
ch:aise-repo found reproduction the best localiser. ch:aise-swe-agents found the
test runner and the iteration loop mutually contingent, and the scaffold worth more
than a large model improvement. ch:aise-testing found suite independence deciding
what iteration means. ch:aise-cicd found gating, rollback and architectural
constraints deciding what a change may do.

None of those is a model capability. This listing collects them, ablates them, and
asks what fraction of safe autonomy each unlocks
(eq:autonomy-is-an-environment-property).
"""
# (prerequisite, what it raises, magnitude, what it depends on)
PREREQS = [
    ("reproduction available",     "localisation", 0.30, None),
    ("test coverage",              "verification", 0.34, None),
    ("independent tests",          "verification", 0.26, "test coverage"),
    ("fast CI",                    "iteration",    0.22, "test coverage"),
    ("blast-radius gating",        "containment",  0.28, None),
    ("rollback path",              "containment",  0.31, None),
    ("architectural constraints",  "verification", 0.19, None),
]
ALL = {p[0] for p in PREREQS}

BASE = {"localisation": 0.53, "verification": 0.30,
        "iteration": 0.20, "containment": 0.25}


def capability(have):
    """Returns the four environment capabilities, given the prerequisites in
    place. A prerequisite with an unmet dependency contributes nothing."""
    caps = dict(BASE)
    for name, raises, mag, dep in PREREQS:
        if name not in have:
            continue
        if dep is not None and dep not in have:
            continue                       # the precondition is missing
        caps[raises] = caps[raises] + (1.0 - caps[raises]) * mag
    return caps


def safe_autonomy(have):
    """Share of changes an agent can complete unsupervised without an escape.
    All four capabilities are required: find it, check it, fix it, contain it."""
    c = capability(have)
    return (c["localisation"] * c["verification"] * c["iteration"] ** 0.5
            * c["containment"])


print("Seven environment prerequisites, each raising one capability an")
print("autonomous agent needs. None of them is a property of the model.")
print()
print(f"{'prerequisite':>28}{'raises':>15}{'by':>8}{'depends on':>18}")
print("-" * 69)
for name, raises, mag, dep in PREREQS:
    print(f"{name:>28}{raises:>15}{mag:>8.0%}{(dep or '--'):>18}")

none = safe_autonomy(set())
full = safe_autonomy(ALL)
print()
print(f"   nothing in place: {none:.1%} of changes safely autonomous")
print(f"   everything:       {full:.1%}")

print()
print()
print("Each prerequisite ADDED to nothing, and REMOVED from everything --")
print("ch:as-single-agent's methodology, which this part has needed repeatedly.")
print()
print(f"{'prerequisite':>28}{'added alone':>14}{'removed from all':>19}")
print("-" * 61)
ab = {}
for name, raises, mag, dep in PREREQS:
    added = safe_autonomy({name}) - none
    removed = full - safe_autonomy(ALL - {name})
    ab[name] = (added, removed)
    print(f"{name:>28}{added:>+14.1%}{removed:>+19.1%}")

print()
print()
print("Building them up in a sensible order.")
print()
ORDER = ["test coverage", "independent tests", "reproduction available",
         "fast CI", "rollback path", "blast-radius gating",
         "architectural constraints"]
print(f"{'after adding':>28}{'safe autonomy':>16}{'gain':>9}")
print("-" * 53)
bu = {}
have, prev = set(), none
for name in ORDER:
    have.add(name)
    v = safe_autonomy(set(have))
    bu[name] = (v, v - prev)
    print(f"{name:>28}{v:>16.1%}{v - prev:>+9.1%}")
    prev = v

print()
print()
print("The four capabilities separately, at each stage of that build-up.")
print()
print(f"{'after adding':>28}{'localise':>11}{'verify':>9}{'iterate':>10}"
      f"{'contain':>10}")
print("-" * 68)
have = set()
for name in ORDER:
    have.add(name)
    c = capability(set(have))
    print(f"{name:>28}{c['localisation']:>11.0%}{c['verification']:>9.0%}"
          f"{c['iteration']:>10.0%}{c['contain' + 'ment']:>10.0%}")

print()
print()
print("And the comparison this part has been building toward. A model that is")
print("better at every step, against an environment that supplies these.")
print()
print(f"{'scenario':>44}{'safe autonomy':>16}")
print("-" * 62)
sc = {}
for label, have, skill in (
        ("today's model, nothing in place", set(), 1.00),
        ("a 25% better model, nothing in place", set(), 1.25),
        ("a 60% better model, nothing in place", set(), 1.60),
        ("today's model, all seven in place", ALL, 1.00)):
    c = capability(have)
    v = (min(c["localisation"] * skill, 0.99)
         * min(c["verification"] * skill, 0.99)
         * min(c["iteration"] * skill, 0.99) ** 0.5
         * c["containment"])
    sc[label] = v
    print(f"{label:>44}{v:>16.1%}")

print(f"""
The build-up table is the part's practical output, and the largest single gain is
not where most attention goes.

**A rollback path is worth {bu['rollback path'][1]:+.1%}** -- the biggest step in
the table -- and it is infrastructure work with no machine learning in it. Blast-radius
gating adds {bu['blast-radius gating'][1]:+.1%}. Between them, containment accounts
for more than a third of the total.

That is ch:as-specialized's finding restated for a seventh time: reversibility was
the property that explained most of the spread there, and it explains the largest
step here. **Being able to undo a change is worth more than being better at making
it.**

The ablation table shows the same contingency this part has hit repeatedly.
Independent tests added alone are worth {ab['independent tests'][0]:+.1%} and removed
from a complete environment cost {ab['independent tests'][1]:+.1%}, because
independence is worthless without coverage to be independent about -- which is why
the model makes it depend on test coverage explicitly.

The capability table shows what each prerequisite actually moves, and the columns do
not fill evenly. Verification reaches {0.72:.0%} and iteration only {0.38:.0%},
because iteration depends on fast CI and fast CI depends on a suite that exists. The
binding capability at the end of the build-up is the one that started lowest and had
the fewest contributors.

And the last table is what part:21 has been building toward.

Today's model with none of these in place reaches {sc["today's model, nothing in place"]:.1%}
safe autonomy. A model {0.60:.0%} better -- an enormous improvement, larger than any
single generation has delivered -- with none of them in place reaches
{sc['a 60% better model, nothing in place']:.1%}. **Today's model with all seven in
place reaches {sc["today's model, all seven in place"]:.1%}**, which is
{sc["today's model, all seven in place"] / sc['a 60% better model, nothing in place']:.1f}
times the better model's figure.

That is cite:chan2024mlebench's scaffolding result, extended from the agent's loop to
the environment the loop runs in, and it is the answer to the question teams
actually ask. **Autonomy is not a capability you wait for. It is a set of properties
you build** (eq:autonomy-is-an-environment-property), and the properties are
enumerable, ordinary, and mostly already on someone's backlog.

Note the absolute level honestly: {sc["today's model, all seven in place"]:.1%} of
changes safely autonomous is not a large number, and it should not be read as one.
It is the share requiring no supervision at all across every change type including
the ones ch:aise-cicd said to gate. The useful reading is the ratio between the rows,
not the rows themselves -- and the ratio says the environment is where the leverage
is.

Which is also why the same model produces such different experiences at different
organisations. Not model access, not prompting, not talent. **A team with
reproduction, coverage, independent tests, fast CI, gating, rollback and executable
constraints is operating a different system**, and it is the system rather than the
model that this part has been measuring.""")
```

## 9. Practical Example

The first listing crosses six modes against four tasks, with total hours including
rework:

```
                      mode  fix a reported  implement a fe  refactor legac  design a data
------------------------------------------------------------------------------------------
                    manual            3.03            9.95            7.47           64.94
                completion            2.65            8.31            5.88           28.87
             chat-assisted            2.24            7.16            5.22           35.64
    agent with full review            1.41            4.95            4.01           53.40
   agent with gated review            0.71            3.37            3.47           88.97
          fully autonomous            0.16            2.23            3.21          124.02
```

**Full autonomy holds three of the four best cells and the single worst**
({{eq:no-mode-dominates}}) — and the worst is the task at
$0.24 \times 0.19 = 0.05$ on the verify-times-reverse product.

Applied uniformly, which is what a policy does:

```
            uniform policy   total hours   human hours  vs best-per-task
--------------------------------------------------------------------------
                    manual         85.38         23.00            +50.91
                completion         45.71         20.24            +11.24
    agent with full review         63.76         10.58            +29.29
          fully autonomous        129.62          0.69            +95.15
```

**Every uniform policy is substantially worse than per-task selection**, and full
autonomy applied uniformly is worse than manual work
({{eq:uniform-policy-is-costly}}) — priced by its worst cell while justified by its
best.

The objective divergence:

```
                    task                 min total                 min human
------------------------------------------------------------------------------
      fix a reported bug          fully autonomous          fully autonomous
     design a data model                completion          fully autonomous
```

**Minimising human hours always recommends maximum autonomy**
({{eq:human-hours-is-not-total-cost}}), and it diverges from minimising cost exactly
where the consequences are largest.

And the reconciliation with the measured evidence:

```
  hours multiplier     total   vs manual   verdict
--------------------------------------------------
              0.46      4.95        -50%    faster
              1.00      9.81         -1%    faster
              1.25     12.06        +21%    slower
```

Break-even at $1.02$: the mode pays until it *adds* work.
{{cite:becker2025devproductivity}} measured hours rising $19\%$, which locates that
study above the threshold rather than contradicting the model.

The second listing collects the environment prerequisites:

```
                prerequisite         raises      by        depends on
---------------------------------------------------------------------
      reproduction available   localisation     30%                --
               test coverage   verification     34%                --
           independent tests   verification     26%     test coverage
                     fast CI      iteration     22%     test coverage
         blast-radius gating    containment     28%                --
               rollback path    containment     31%                --
   architectural constraints   verification     19%                --
```

Built up in order:

```
                after adding   safe autonomy     gain
-----------------------------------------------------
               test coverage            3.2%    +1.4%
      reproduction available            4.9%    +1.0%
                     fast CI            6.8%    +1.8%
               rollback path           13.1%    +6.3%
         blast-radius gating           17.0%    +3.9%
   architectural constraints           18.7%    +1.7%
```

**A rollback path is the largest single step** at $+6.3$ points
({{eq:containment-is-the-largest-step}}) — infrastructure work with no machine
learning in it, and the one term no model improvement touches.

And the part's closing comparison:

```
                                    scenario   safe autonomy
--------------------------------------------------------------
             today's model, nothing in place            1.8%
        a 25% better model, nothing in place            3.1%
        a 60% better model, nothing in place            5.8%
           today's model, all seven in place           18.7%
```

**Today's model with all seven prerequisites is worth more than three times a
$60\%$-better model with none** ({{eq:autonomy-is-an-environment-property}}).

## 10. Production Considerations

Route by task type, not by policy. Classify mechanically where you can — migration
directories, service interfaces, public APIs — so the routing is enforceable.

Build {{ch:aise-cicd}}'s table first. Per-task selection needs error costs and
verifiability estimates, and without them a uniform policy is the only option
available.

Do not report hours saved as the adoption metric. It recommends maximum autonomy
unconditionally, and its bias has a known sign.

Track total engineering hours including rework, or
merged-without-substantive-revision rate.

Measure the effect on your own work with randomisation. The setting is a first-class
variable and the sign changes across it.

Build the seven prerequisites in order — coverage, independence, reproduction, fast
CI, rollback, gating, constraints — and expect rollback to be the largest step.

And when an agent underperforms, check the environment before the model. That is
where the factor of three is.

## 11. Common Mistakes

**Adopting autonomy as a policy.** Priced by the worst cell.

**Applying full autonomy uniformly.** Worse than manual in this table.

**Optimising human hours.** Recommends maximum autonomy for every task.

**Reading the measured slowdown as a verdict on the tools.** It is one cell of a
table whose entries change sign.

**Surveying developers for the effect.** The bias is signed.

**Waiting for a better model.** Worth a third of what the environment is worth.

**Treating rollback as an operations concern.** It is the largest step in the
autonomy table.

## 12. Failure Modes

*Policy priced by its worst cell.* Every automation succeeding and the total losing.

*Headcount-optimised autonomy.* Maximum delegation, justified by the metric being
measured.

*Unattributed rework.* Cost accruing where nobody connects it to the decision that
caused it.

*Uncontained agent.* An environment where model improvement raises the escape rate
because nothing bounds it.

*Setting-blind adoption.* A tool deployed on the evidence from a different cell.

*Prerequisite pruning.* A contingent prerequisite removed on a one-at-a-time
measurement — {{eq:scaffold-components-interact}}'s trap, at the environment level.

## 13. Alternatives

**Progressive autonomy by track record.** Relax the mode per task type as escape
records accumulate — the empirical version of this chapter's routing.

**Autonomy inside a sandbox.** Full autonomy where the blast radius is bounded by
construction rather than by policy, which raises $K$ to nearly one for that scope.

**Human-authored design, agent-implemented.** {{ch:aids-oversight}}'s
divide-by-gradeability applied to the tasks that fail both criteria.

**Two-agent review.** A different-family reviewer before the human, subject to
{{ch:aids-autonomous}}'s correlation caveat.

**Not adopting until the prerequisites exist.** Defensible, and the second listing
prices the alternative.

## 14. Evaluation

Run the randomised comparison. Sixteen developers and $246$ tasks is a large study;
five developers and forty tasks is a week and tells you your cell.

Measure your escape rate and rework cost per task type. It is the input to the
routing rule.

Measure your seven prerequisites honestly — coverage, independence, CI latency,
rollback time — and locate yourself in the build-up table.

Track merged-without-revision rate as the headline adoption metric.

Report the setting with every claim: experience level, codebase maturity, task type.

And re-run the comparison when the model changes, since the routing thresholds move.

## 15. Advanced Concepts

**Automatic task routing.** Classifying a change's verifiability and reversibility
from its diff and repository context, so the mode is selected per change rather than
per category. {{maturity:EMERGING}}.

**Escape-cost learning.** Estimating $e_t$ from incident history, which turns the
routing table into a maintained artefact rather than a one-off estimate.

**Containment as a first-class deployment property.** Measuring and reporting
time-to-rollback per change type, which {{eq:containment-is-the-largest-step}} says
is the ceiling on everything.

**Cross-organisation comparison.** Whether pipeline maturity explains the observed
variance in agent effectiveness, which is testable and unpublished.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:aise-cicd}}'s two properties order every table here, and its gating table is the
prerequisite for this chapter's routing rule.

{{ch:aise-generation}}'s visible/invisible split explains the self-report gap at the
organisational level as well as the individual one.

{{ch:aise-swe-agents}}'s scaffolding result extends from the agent's loop to the
environment the loop runs in, which is this chapter's closing comparison.

{{ch:aids-oversight}}'s divide-by-gradeability arrives independently in a second
domain, with reversibility added as a second criterion.

{{ch:as-specialized}}'s reversibility finding reaches its seventh appearance as the
largest step in the autonomy table.

Ahead: {{part:22}} turns to designing the systems these agents run inside.

## 17. Exercises

1. Build the mode-by-task table for your own work and compute your uniform-policy
   regret.

2. Estimate your error costs per task type from incident history.

3. Derive the break-even hours multiplier from
   {{eq:no-mode-dominates}} for a task of your own.

4. Locate your team in the second listing's build-up table and identify the next
   prerequisite.

5. Measure your time-to-rollback per change type and compute what raising
   containment would buy.

6. Run a small randomised trial on your own tasks and compare with the self-reports.

## 18. Interview Questions

1. Should agents be allowed to merge without review?

2. Why can full autonomy applied uniformly be worse than manual work?

3. Your adoption metric is hours saved. What will it recommend?

4. A study found AI tools slowed experienced developers by $19\%$. What follows?

5. You can invest in a better model or in your pipeline. Which?

6. What is the single highest-value thing to build before increasing autonomy?

## 19. Research Questions

1. Can a change's verifiability and reversibility be inferred automatically from its
   diff?

2. Does pipeline maturity explain the cross-organisation variance in agent
   effectiveness?

3. How stable are the routing thresholds as models improve?

4. What is the real distribution of error costs by change type, and does it span the
   two orders of magnitude assumed here?

5. Does the self-report bias persist at the organisational level, and by how much?

## 20. Chapter Summary

Crossing six operating modes against four task types, **no mode wins everywhere**
({{eq:no-mode-dominates}}). Full autonomy holds three of the four best cells and the
single worst — $128$ hours on data-model design against completion's $28$ — because
that task sits at $0.05$ on the verify-times-reverse product where bug fixing sits at
$0.87$.

Since organisations adopt stances rather than per-task rules, that matters: **every
uniform policy is substantially worse than per-task selection**, the best uniform
stance costing $+11.2$ hours and **full autonomy applied uniformly costing $+95.2$ —
worse than manual work** ({{eq:uniform-policy-is-costly}}). A uniform policy is priced
by its worst cell and justified by its best.

And the metric most organisations use points the wrong way. **Minimising human hours
recommends maximum autonomy for every task, unconditionally**
({{eq:human-hours-is-not-total-cost}}), and diverges from minimising cost exactly on
the consequential tasks.

On the measured evidence, agent-with-review pays until it *adds* work — break-even at
a $1.02$ hours multiplier — and {{cite:becker2025devproductivity}} measured hours
rising $19\%$, which locates that study above the threshold rather than outside the
model. What generalises from it is that **the effect must be measured rather than
surveyed**, and that the setting is a first-class variable.

Finally, the part's closing argument. Seven environment prerequisites — coverage,
independent tests, reproduction, fast CI, rollback, gating, executable architectural
constraints — take safe autonomy from $1.8\%$ to $18.7\%$. A model $60\%$ better with
none of them reaches $5.8\%$. **Today's model with all seven is worth more than three
times a large model improvement without them**
({{eq:autonomy-is-an-environment-property}}).

The largest single step is a **rollback path**, at $+6.3$ points
({{eq:containment-is-the-largest-step}}) — infrastructure work, no machine learning
in it, and the one term no model improvement touches. Which is
{{ch:as-specialized}}'s reversibility result in its seventh setting: **being able to
undo a mistake is worth more than being less likely to make one.**

## 21. Further Reading

{{cite:becker2025devproductivity}} is the most rigorous measurement in this area and
should be read for its methodology as much as its headline — particularly the twenty
explanatory factors and the discussion of what its setting does and does not
represent.

{{cite:chan2024mlebench}} for the scaffolding result this chapter extends to the
environment, and {{cite:wang2025solvedcorrectly}} for the verification gap that sets
how much review the routing rule has to buy.

{{cite:testini2025dsautomation}} for the collaboration-spectrum gap this chapter
closes in a second domain, and {{ch:aids-oversight}} for the first.

{{cite:jimenez2023swebench}} for the $1.96\%$ that makes the trajectory legible, and
{{ch:aise-cicd}} for the two properties that order everything here.
