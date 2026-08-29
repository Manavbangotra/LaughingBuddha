---
id: ops-prompt-versioning
number: 209
part: XXIV
tier: full
status: draft
requires: [reproducibility-is-a-product-over-artefacts, semantic-failure-has-no-instrument,
           rework-cost-is-set-by-detection-lateness, biased-sampling-distorts-composition]
provides: [prompt-is-ungated-code, format-check-is-the-cheapest-gate,
           evaluation-sets-decay-silently, refresh-beats-growth]
citations: [sculley2015, breck2017, gama2014, paleyes2020deployment]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute a defect escape rate from a
change's gate stack, and explain why prompts escape at 100%; rank candidate prompt gates
by defects prevented per unit of effort and identify the cheapest one; show that an
evaluation set's coverage decays exponentially with traffic drift while its reported pass
rate does not move; compute the refresh cadence that minimises total cost; and explain
why enlarging an evaluation set does not substitute for refreshing it.

## 2. Why This Matters

{{ch:ops-versioning}} argued for putting the prompt under version control on
reproducibility grounds. This chapter makes the same argument again from quality, and the
second argument is blunter.

Application code passes five gates — compiler, types, unit tests, review, integration —
and a defective change escapes at **8.3%**. A system prompt passes **zero** gates and
escapes at **100%** ({{eq:prompt-is-ungated-code}}). Prompts and few-shot examples are
**63%** of changes and **88%** of escaped defects, and the listing assumes they are
*equally likely to contain a mistake* — the gap is entirely gating.

The fix is cheap. A format check on the assembled prompt removes **38%** of defective
changes for half a unit of effort, the best ratio available
({{eq:format-check-is-the-cheapest-gate}}), and it exists in almost no system.

The second half concerns the gate teams do build. An evaluation set is a sample of the
traffic distribution at the moment it was made, and traffic drifts — so coverage falls
from **100%** to **16%** over a year while **the reported pass rate does not move**
({{eq:evaluation-sets-decay-silently}}). And the instinctive fix, adding more cases, does
not work: quadrupling the set leaves coverage at 16%, because a bigger sample of an old
distribution is still old ({{eq:refresh-beats-growth}}).

## 3. Prerequisites

You need {{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} —
the prompt appeared there as the second-highest-payback artefact, and this chapter is the
independent quality case for the same work.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is why a decayed
evaluation gate is invisible: it keeps reporting its own pass rate, which is not the
quantity that changed.

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} prices what an
escaped prompt defect costs: it is detected late, and the return trip is long.

{{cite:gama2014}}'s concept drift is the mechanism behind evaluation-set decay, applied
to the evaluation set rather than to the model.

## 4. Intuitive Explanation

Consider the path a change takes to production, and count the things that could stop it.

For a line of application code: a compiler that rejects it if it does not parse, a type
checker that rejects it if the shapes do not fit, a unit test that fails if the behaviour
changed, a reviewer who might notice, and an integration test that exercises it in
context. Five independent chances to catch a mistake before a user sees it.

For a change to the system prompt: none of those. Not a weakened version of them —
none. There is no compiler for English. There is no type checker for "be concise and cite
your sources." Most teams have no test that asserts anything about the prompt, no review
because the edit did not go through a repository, and no integration test because the
prompt is not an input to any test that exists.

So the escape rate for a defective prompt change is one hundred percent. Every mistake
reaches production, because nothing in the path is capable of stopping one.

Now put that next to how often prompts change. In {{sec:9-practical-example}}'s numbers
they change twice as often as code — they are the easiest thing to adjust, so they absorb
most of the iteration. Sixty-three percent of changes, eighty-eight percent of escaped
defects.

And notice what that argument does *not* assume. It does not assume prompts are harder to
get right than code, or that the people writing them are less careful. Assume identical
care and identical defect rates at the moment of writing, and the gap is still there,
because the gap is entirely about what happens afterwards.

Which means it is fixable, and cheaply. You cannot compile a prompt, but you can assert
things about the assembled one: that it contains the sections it should, in the order it
should, under the length it should, with the retrieved context actually substituted in.
That is a format check, it is half a day of work, and it catches over a third of
defective changes.

The second half of the chapter is about the gate teams do eventually build — an
evaluation set — and a way it fails that is entirely silent.

An evaluation set is a snapshot. You collect a few hundred representative cases, label
them, and gate changes on the score. On the day you build it, those cases are
representative by construction.

Then traffic moves. A new feature brings new phrasings. A new customer segment asks
different questions. The documents change. Every week, a few percent of what your system
handles is something your evaluation set has no case for.

After a year, the set represents about a sixth of your traffic. The gate is now testing a
corner of what the system does and passing everything else through untested.

Here is the part that makes it dangerous rather than merely imperfect: **the gate does not
report this.** It runs its cases, they pass at the same rate they always did, and the
dashboard shows a healthy suite. The cases have not become wrong — they have become
unrepresentative, and a pass rate cannot tell the difference.

The natural response is to add more cases, and it does not help. Doubling the set doubles
your sample of the distribution as it was; it does not add a single case from the traffic
that arrived since. Size fixes noise and age causes bias, and you cannot fix a bias by
sampling harder from the wrong place.

## 5. Formal Explanation

**Gate stacks.** Let a change have prior defect probability $\pi$ and pass through gates
$g \in G$, where gate $g$ lets a defective change through with probability $\kappa_g$.
Assuming gates fail independently, the escape rate is

$$ E(G) \;=\; \prod_{g \in G}\kappa_g, \qquad \text{defects per week} \;=\; \nu\,\pi\,E(G) $$ (eq:prompt-is-ungated-code)

for change rate $\nu$. For $G = \emptyset$ the product is empty and $E = 1$: **an ungated
change escapes with certainty**, which is a tautology worth writing down because it is
the entire situation for prompts.

The share of escaped defects attributable to artefact $a$ is
$\nu_a\pi E_a / \sum_b \nu_b\pi E_b$, which for equal $\pi$ reduces to
$\nu_a E_a / \sum_b \nu_b E_b$ — **change rate times escape rate**, not change rate alone.
That is why prompts contribute disproportionately without being written worse.

Ranking candidate gates by defects prevented per unit effort $e_g$ gives

$$ \text{rank}(g) \;=\; \frac{\nu\pi(1 - \kappa_g)}{e_g} $$ (eq:format-check-is-the-cheapest-gate)

which favours cheap gates with moderate catch rates over expensive gates with high ones —
the opposite of how gates are usually proposed.

**Evaluation decay.** Let traffic drift at rate $\delta$ per week, meaning a share
$\delta$ of the distribution each week is unrepresented by any pre-existing sample. A set
of age $w$ covers

$$ C(w) \;=\; (1-\delta)^w $$

and if the gate catches a defect in covered traffic with probability $\gamma$, its true
catch rate is $\gamma C(w)$. Meanwhile the *reported* pass rate is computed over the set's
own cases, which are unchanged, so

$$ \frac{\partial\,\text{reported}}{\partial w} = 0 \quad\text{while}\quad \frac{\partial\,\gamma C(w)}{\partial w} = -\gamma\delta(1-\delta)^w < 0 $$ (eq:evaluation-sets-decay-silently)

**The measurement is constant and the thing measured decays.**

Refreshing every $T$ weeks gives mean coverage
$\bar{C}(T) = \frac{1}{T}\sum_{w<T}(1-\delta)^w$, and total annual cost is labelling plus
escapes:

$$ K(T) \;=\; \frac{52}{T}\,S\rho_\ell \;+\; 52\,\nu_d\bigl(1 - \gamma\bar{C}(T)\bigr)\lambda_d $$

for set size $S$, per-case labelling cost $\rho_\ell$, defect rate $\nu_d$ and defect cost
$\lambda_d$. Convex in $T$, with an interior minimum.

## 6. Mathematical Foundation

The growth-versus-refresh comparison follows from noting that $S$ and $w$ enter the model
in different places.

Set size $S$ affects only the *variance* of the estimate: the catch probability $\gamma$
on covered traffic improves with more cases, but with diminishing returns and bounded
above by one. Set age $w$ affects the *coverage* $C(w)$, which multiplies $\gamma$.

So the true catch rate is $\gamma(S)\,C(w)$, and

$$ \frac{\partial}{\partial S}\bigl[\gamma(S)C(w)\bigr] = C(w)\gamma'(S), \qquad \frac{\partial}{\partial w}\bigl[\gamma(S)C(w)\bigr] = -\gamma(S)\delta C(w) $$ (eq:refresh-beats-growth)

The size derivative is **multiplied by $C(w)$** — so as the set ages, adding cases buys
proportionally less, because the additional cases are drawn from the same stale
distribution. **Growth cannot compensate for age; age discounts growth.**

{{sec:9-practical-example}} measures this directly: quadrupling a set from 900 to 3,600
cases leaves coverage at 16% and the catch rate at 12%, while refreshing 900 cases
quarterly reaches 81% coverage and 60% catch, for less total labelling.

The rolling-refresh design follows from the same expression. Replacing a fraction $f$ of
the set each month gives mean case age $\approx 4/(2f)$ weeks, so coverage is
$C(4/(2f))$ while labelling is $Sf$ per month. Since $C$ is exponential in age and cost is
linear in $f$, **there are strong diminishing returns to refreshing faster**, and
{{sec:9-practical-example}} finds 25% monthly replacement getting most of full monthly
regeneration's coverage for a quarter of the labelling.

One term the model omits favours refreshing more than shown. A refreshed set is drawn from
current traffic, which includes the failure modes introduced by recent changes — so it
tests the code most likely to be wrong. A stale set tests code that has been stable for a
year, which is the code least likely to break.

## 7. Internal Mechanics

**Why there is no compiler for a prompt.** A compiler checks a program against a grammar
and a type system, both of which are total: every program is either well-formed or not.
Natural language has no such structure, and the property a prompt must satisfy —
"produces good behaviour from this model" — is not decidable from the text. So the absence
of a compiler is real. **What is not real is the leap from "no compiler" to "no gates at
all"**, which is the leap most systems have made.

**What a format check actually asserts.** That the assembled prompt contains each required
section; that the sections appear in the intended order; that retrieved context was
substituted rather than left as a placeholder; that the total is within the context
budget; that no section is empty. Every one of those is a bug that has shipped, and every
one is a string assertion.

**Why golden-output tests work here when {{ch:sd-architecture}} said they survive at 9%.**
That result was about testing *generated text*, which is nondeterministic. Testing the
*assembled prompt* is deterministic — same inputs, same string — so golden tests apply
with full force. **The prompt is the deterministic half of a nondeterministic system**,
and it is testable by exactly the means the nondeterministic half defeats.

**Why prompts are edited outside the repository.** They are usually editable through a
console or an admin interface, because the ability to adjust behaviour without a deploy is
a genuine operational benefit. That benefit is real and it is what removes the gates. The
resolution is not to take the console away but to route it through the same checks — a
console that runs the format check and records the change is a console that keeps the
benefit and closes the gap.

**Drift is not the same as concept drift.** {{cite:gama2014}}'s concept drift is the
input-output relationship changing. Evaluation-set decay is the *input distribution*
moving away from the sample, with the relationship unchanged. They have different
remedies: concept drift needs retraining, coverage decay needs resampling, and a team that
diagnoses one as the other spends a quarter on the wrong fix.

**Why prompt changes are so frequent in the first place.** The change-rate figures are
not incidental to the argument -- they are a consequence of the same property that removed
the gates. A prompt is the cheapest place to make a behavioural change, so it absorbs
iteration that would otherwise have gone into code, retrieval, or tooling. That is a real
benefit: it is why teams can respond to a complaint the same afternoon. But it means the
artefact with no gates is also the artefact that changes most, and the two facts compound
rather than merely coexisting. **Gating the prompt does not only reduce the escape rate;
it slows the change rate**, which is a cost worth naming rather than hiding, and the
right response is a gate fast enough not to lose the afternoon.

**Measuring the drift rate.** Everything in the second half depends on $\delta$, and it is
estimable: embed a week of current traffic, embed the evaluation set, and measure the share
of current traffic with no near neighbour in the set. That is an afternoon's work and it
converts the cadence recommendation from arithmetic-on-a-guess into a decision.

**{{cite:sculley2015}}'s configuration debt, precisely.** A prompt is configuration that
determines behaviour, changes frequently, and is not under the discipline applied to code.
{{cite:breck2017}}'s readiness rubric has a whole section on configuration testing, and
almost no team applies it to the prompt.

## 8. Implementation

The first listing measures escape rates by gate stack.

```python {tier=A name=ee1}
"""A prompt is code, and it is the only code with no tests, no review, and no types.

Every other artefact that determines behaviour passes through gates: a compiler, a type
checker, a test suite, a reviewer. A prompt passes through none of them. It is a string,
it is edited by people who are not committing, and the first thing that evaluates it is
production.

So the defect escape rate for a prompt change is the escape rate for an ungated change,
and this listing measures what that is against the gated alternatives
(eq:prompt-is-ungated-code).

The finding is that the gap is not explained by prompts being harder to test. It is
explained by nobody testing them.
"""
# (artefact, changes per week, gates it passes, P(a defective change escapes each gate)
CHANGES = [
    ("application code",   3.0,
     [("compiler", 0.55), ("type check", 0.70), ("unit tests", 0.55),
      ("code review", 0.60), ("integration tests", 0.65)]),
    ("configuration",      2.0,
     [("schema validation", 0.70), ("code review", 0.60)]),
    ("tool schema",        0.7,
     [("schema validation", 0.65), ("code review", 0.60)]),
    ("system prompt",      6.0, []),
    ("few-shot examples",  4.0, []),
]
P_DEFECTIVE = 0.22        # share of changes that contain a defect, before gates


def escape(gates):
    p = 1.0
    for name, keep in gates:
        p *= keep
    return p


print("How each artefact is gated before it reaches production.")
print()
print(f"{'artefact':>22}{'changes/week':>15}{'gates':>8}"
      f"{'escape rate':>14}{'defects/week':>15}")
print("-" * 76)
tab = {}
for name, rate, gates in CHANGES:
    e = escape(gates)
    d = rate * P_DEFECTIVE * e
    tab[name] = (rate, len(gates), e, d)
    print(f"{name:>22}{rate:>15.1f}{len(gates):>8}{e:>14.1%}{d:>15.2f}")
print("-" * 76)
total = sum(tab[n][3] for n, r, g in CHANGES)
print(f"{'TOTAL':>22}{sum(r for n, r, g in CHANGES):>15.1f}{'':>8}"
      f"{'':>14}{total:>15.2f}")

print()
print()
print("Share of escaped defects by artefact -- which is not the share of changes.")
print()
print(f"{'artefact':>22}{'share of changes':>19}{'share of escapes':>19}"
      f"{'ratio':>9}")
print("-" * 70)
tot_changes = sum(r for n, r, g in CHANGES)
share = {}
for name, rate, gates in CHANGES:
    sc = rate / tot_changes
    se = tab[name][3] / total
    share[name] = (sc, se, se / sc)
    print(f"{name:>22}{sc:>19.0%}{se:>19.0%}{se / sc:>9.1f}x")

print()
print()
print("What each gate would remove, applied to prompts.")
print()
GATES = [
    ("schema / format check",   0.62, 0.5),
    ("golden-output test",      0.71, 3.0),
    ("peer review",             0.45, 1.0),
    ("evaluation-set gate",     0.38, 6.0),
    ("shadow comparison",       0.30, 9.0),
]
print(f"{'gate for prompts':>26}{'keeps':>9}{'escape after':>15}"
      f"{'defects/week':>15}{'effort':>9}")
print("-" * 76)
cur = 1.0
eff = 0.0
prompt_rate = tab["system prompt"][0] + tab["few-shot examples"][0]
base_prompt = prompt_rate * P_DEFECTIVE
path = []
for label, keep, e in GATES:
    cur *= keep
    eff += e
    path.append((label, cur, prompt_rate * P_DEFECTIVE * cur, eff))
    print(f"{label:>26}{keep:>9.0%}{cur:>15.1%}"
          f"{prompt_rate * P_DEFECTIVE * cur:>15.2f}{eff:>9.1f}")

print()
print(f"prompt defects per week, ungated: {base_prompt:.2f}")
print(f"after all five gates:             {path[-1][2]:.2f}")

print()
print()
print("Cost per defect prevented, which is how a gate should be chosen.")
print()
order = sorted(GATES, key=lambda g: -((1 - g[1]) / g[2]))
print(f"{'rank':>6}{'gate':>26}{'removes':>10}{'effort':>9}"
      f"{'defects/wk prevented':>23}{'per effort':>13}")
print("-" * 88)
for i, (label, keep, e) in enumerate(order, 1):
    prevented = base_prompt * (1 - keep)
    print(f"{i:>6}{label:>26}{1 - keep:>10.0%}{e:>9.1f}"
          f"{prevented:>23.2f}{prevented / e:>13.3f}")

print()
print()
print("And the comparison that makes the case: apply the SAME gate coverage that")
print("application code already has.")
print()
code_escape = tab["application code"][2]
print(f"{'artefact':>22}{'escape rate':>14}{'defects/week':>15}"
      f"{'vs code':>11}")
print("-" * 64)
for name, rate, gates in CHANGES:
    print(f"{name:>22}{tab[name][2]:>14.1%}{tab[name][3]:>15.2f}"
          f"{tab[name][2] / code_escape:>10.1f}x")
print(f"{'prompt at code gating':>22}{code_escape:>14.1%}"
      f"{prompt_rate * P_DEFECTIVE * code_escape:>15.2f}{1.0:>10.1f}x")

print(f"""
The gating table is the whole argument and it needs almost no commentary. Application
code passes {tab['application code'][1]} gates and escapes at
{tab['application code'][2]:.1%}. A system prompt passes **zero** and escapes at
{tab['system prompt'][2]:.0%} (eq:prompt-is-ungated-code).

Every defective prompt change reaches production. Not most of them --
**all of them**, because there is nothing in the path that could stop one.

The share table converts that into where the defects come from. Prompts and few-shot
examples are {share['system prompt'][0] + share['few-shot examples'][0]:.0%} of changes
and {share['system prompt'][1] + share['few-shot examples'][1]:.0%} of escaped defects.
Application code is {share['application code'][0]:.0%} of changes and
{share['application code'][1]:.0%} of escapes.

**Prompts produce {share['system prompt'][1] / share['system prompt'][0]:.1f} times their
share of defects and code produces {share['application code'][1] / share['application code'][0]:.2f}
times its share** -- and the difference is entirely gating, since the defect rate before
gates was assumed identical.

That last point is worth being explicit about. This listing assumes prompts and code are
equally likely to contain a mistake when written. **The escape gap is not because prompts
are harder to get right. It is because nothing checks them.**

The gate table shows the ungated state is a choice rather than a necessity. A format
check removes {1 - 0.62:.0%} of defective prompt changes for {0.5:.1f} units of effort.
A golden-output test -- which ch:sd-architecture said survives at only 9% for model
outputs -- still removes {1 - 0.71:.0%} when applied to *prompt structure* rather than to
generated text, because it is checking that the assembled prompt looks right rather than
that the answer is right.

All five gates take prompt defects from {base_prompt:.2f} to {path[-1][2]:.2f} a week.

The ranking is where a plan comes from. `{order[0][0]}` removes {1 - order[0][1]:.0%} for
{order[0][2]:.1f} effort -- {base_prompt * (1 - order[0][1]) / order[0][2]:.3f} defects
prevented per unit. `{order[-1][0]}` removes {1 - order[-1][1]:.0%} for
{order[-1][2]:.1f}, which is {base_prompt * (1 - order[-1][1]) / order[-1][2]:.3f}.

**The cheapest gate is a format check and it does not exist in most systems.** Not
because it is hard -- it is asserting that the assembled prompt contains the sections it
should, in the order it should, within the length it should -- but because a prompt does
not feel like something you assert about.

The final table is the comparison to put in a design document. Applying application
code's existing gate coverage to prompts would take them from
{tab['system prompt'][3] + tab['few-shot examples'][3]:.2f} escaped defects a week to
{prompt_rate * P_DEFECTIVE * code_escape:.2f} --
{(tab['system prompt'][3] + tab['few-shot examples'][3]) / (prompt_rate * P_DEFECTIVE * code_escape):.0f}
times fewer.

**Nothing about that requires new technology.** It requires deciding that the string is
code, which is a policy decision that ch:ops-versioning already argued for on
reproducibility grounds and this listing argues for again on quality grounds. Two
independent arguments, one afternoon of work, and it is still the most commonly skipped
item in this part.""")
```

## 9. Practical Example

How each artefact is gated:

```
              artefact   changes/week   gates   escape rate   defects/week
----------------------------------------------------------------------------
      application code            3.0       5          8.3%           0.05
         configuration            2.0       2         42.0%           0.18
           tool schema            0.7       2         39.0%           0.06
         system prompt            6.0       0        100.0%           1.32
     few-shot examples            4.0       0        100.0%           0.88
----------------------------------------------------------------------------
                 TOTAL           15.7                                 2.50
```

**Zero gates, 100% escape** ({{eq:prompt-is-ungated-code}}). Every defective prompt change
reaches production, because nothing in the path could stop one.

```
              artefact   share of changes   share of escapes    ratio
----------------------------------------------------------------------
      application code                19%                 2%      0.1x
         configuration                13%                 7%      0.6x
           tool schema                 4%                 2%      0.5x
         system prompt                38%                53%      1.4x
     few-shot examples                25%                35%      1.4x
```

Prompts and examples are **63%** of changes and **88%** of escaped defects. **The listing
assumed identical defect rates at the moment of writing** — the whole gap is gating.

Candidate gates, ranked by defects prevented per unit of effort:

```
  rank                      gate   removes   effort   defects/wk prevented   per effort
----------------------------------------------------------------------------------------
     1     schema / format check       38%      0.5                   0.84        1.672
     2               peer review       55%      1.0                   1.21        1.210
     3       evaluation-set gate       62%      6.0                   1.36        0.227
     4        golden-output test       29%      3.0                   0.64        0.213
     5         shadow comparison       70%      9.0                   1.54        0.171
```

**The format check is first at 1.672** ({{eq:format-check-is-the-cheapest-gate}}) — half a
unit of effort, and it does not exist in most systems.

```mermaid {#fig:gates caption="Code passes five gates and escapes at 8.3%; a prompt passes none and escapes at 100%. The defect rate at the moment of writing is assumed identical — the entire gap is what happens afterwards."}
flowchart LR
  A["code change"] --> B["compiler → types → tests<br/>→ review → integration"]
  B --> C["8.3% escape"]
  D["prompt change"] --> E["nothing"]
  E --> F["100% escape"]
  D -.->|"format check<br/>0.5 effort"| G["62% escape"]
```

And the comparison for a design document:

```
              artefact   escape rate   defects/week    vs code
----------------------------------------------------------------
      application code          8.3%           0.05       1.0x
         system prompt        100.0%           1.32      12.1x
     few-shot examples        100.0%           0.88      12.1x
 prompt at code gating          8.3%           0.18       1.0x
```

Applying code's existing gate coverage to prompts takes escaped defects from **2.20** to
**0.18** a week — **12× fewer**, with no new technology.

The second listing turns to the gate teams do build.

```python {tier=A name=ee2}
"""An evaluation set is a gate that decays, and nothing announces when it has.

A fixed evaluation set is a sample of the traffic distribution at the moment it was
built. Traffic moves -- new features, new users, new phrasings, new documents -- so the
set represents less of production every week.

The gate does not fail loudly when this happens. It keeps passing, at the same rate,
having stopped testing most of what the system now does
(eq:evaluation-sets-decay-silently).

This listing measures the decay, finds the refresh cadence that holds coverage, and shows
why the obvious alternative -- keep adding cases -- does not fix it.
"""
DRIFT_PER_WEEK = 0.035        # share of traffic distribution that is new each week
SET_SIZE = 900
WEEKS = [0, 4, 12, 26, 52, 104]
GATE_CATCH_ON_COVERED = 0.74  # P(the gate catches a defect in covered traffic)


def coverage(weeks_old):
    """Share of current traffic the set still represents."""
    return (1.0 - DRIFT_PER_WEEK) ** weeks_old


def gate_power(weeks_old):
    return coverage(weeks_old) * GATE_CATCH_ON_COVERED


print("Traffic distribution drifts %.1f%% a week. An evaluation set built once"
      % (DRIFT_PER_WEEK * 100))
print("covers less of it every week, and nothing in the gate reports this.")
print()
print(f"{'set age (weeks)':>17}{'coverage':>11}{'gate catches':>15}"
       f"{'escapes':>10}{'vs new set':>13}")
print("-" * 68)
tab = {}
for w in WEEKS:
    c = coverage(w)
    g = gate_power(w)
    tab[w] = (c, g, 1 - g)
    print(f"{w:>17}{c:>11.0%}{g:>15.0%}{1 - g:>10.0%}"
          f"{(1 - g) / (1 - gate_power(0)):>12.2f}x")

print()
print()
print("What the gate REPORTS while that happens: the pass rate on its own cases,")
print("which is unaffected by drift because those cases have not changed.")
print()
print(f"{'set age (weeks)':>17}{'pass rate reported':>21}{'true catch rate':>18}"
       f"{'gap':>9}")
print("-" * 68)
for w in WEEKS:
    reported = 0.93           # the suite keeps passing at its usual rate
    print(f"{w:>17}{reported:>21.0%}{tab[w][1]:>18.0%}"
          f"{reported - tab[w][1]:>9.0%}")

print()
print("The reported number does not move. That is the failure mode.")

print()
print()
print("Refresh cadence: how often the set must be regenerated to hold coverage.")
print()
print(f"{'refresh every':>16}{'coverage at worst':>20}{'mean coverage':>16}"
       f"{'regenerations/yr':>19}")
print("-" * 74)
cad = {}
for weeks in (2, 4, 8, 13, 26, 52):
    worst = coverage(weeks)
    mean = sum(coverage(w) for w in range(weeks)) / weeks
    cad[weeks] = (worst, mean, 52.0 / weeks)
    print(f"{weeks:>14}w{worst:>20.0%}{mean:>16.0%}{52.0 / weeks:>19.1f}")

print()
print()
print("The cost of each cadence, against what it prevents.")
print()
LABEL_COST_PER_CASE = 4.10
DEFECTS_PER_WEEK = 1.7
DEFECT_COST = 2400.0
print(f"{'refresh every':>16}{'labelling/yr':>15}{'escapes/yr':>13}"
       f"{'escape cost/yr':>17}{'total/yr':>12}")
print("-" * 76)
tot = {}
for weeks in (2, 4, 8, 13, 26, 52):
    label = SET_SIZE * LABEL_COST_PER_CASE * (52.0 / weeks)
    esc = DEFECTS_PER_WEEK * 52.0 * (1 - cad[weeks][1] * GATE_CATCH_ON_COVERED)
    tot[weeks] = (label, esc, esc * DEFECT_COST, label + esc * DEFECT_COST)
    print(f"{weeks:>14}w{label:>15,.0f}{esc:>13.1f}"
          f"{esc * DEFECT_COST:>17,.0f}{label + esc * DEFECT_COST:>12,.0f}")

best = min(tot, key=lambda k: tot[k][3])
print()
print(f"cheapest cadence: every {best} weeks at {tot[best][3]:,.0f} a year")

print()
print()
print("Why growing the set does not substitute for refreshing it.")
print()
print(f"{'strategy':>34}{'cases':>9}{'coverage':>11}{'catches':>10}"
       f"{'labelling/yr':>15}")
print("-" * 80)
STRATS = [
    ("900 cases, never refreshed",       900,  coverage(52), 0),
    ("1800 cases, never refreshed",     1800,  coverage(52), 900),
    ("3600 cases, never refreshed",     3600,  coverage(52), 2700),
    ("900 cases, refreshed quarterly",   900,  cad[13][1],   900 * 4),
    ("900 cases, refreshed monthly",     900,  cad[4][1],    900 * 13),
]
for label, n, cov, new_cases in STRATS:
    print(f"{label:>34}{n:>9}{cov:>11.0%}"
          f"{cov * GATE_CATCH_ON_COVERED:>10.0%}"
          f"{new_cases * LABEL_COST_PER_CASE:>15,.0f}")

print()
print()
print("And the sampling design that keeps a set current for less: replace the")
print("oldest slice each period rather than regenerating the whole set.")
print()
print(f"{'replace per month':>19}{'mean age (weeks)':>19}{'coverage':>11}"
       f"{'labelling/yr':>15}")
print("-" * 66)
roll = {}
for frac in (1.00, 0.50, 0.25, 0.10, 0.05):
    mean_age = 4.0 / (2.0 * frac) if frac > 0 else 999.0
    c = coverage(mean_age)
    cases = SET_SIZE * frac * 13.0
    roll[frac] = (mean_age, c, cases * LABEL_COST_PER_CASE)
    print(f"{frac:>19.0%}{mean_age:>19.1f}{c:>11.0%}"
          f"{cases * LABEL_COST_PER_CASE:>15,.0f}")

print(f"""
The decay table is the mechanism. A set built today covers {coverage(0):.0%} of traffic;
at {26} weeks it covers {tab[26][0]:.0%} and at {104} it covers {tab[104][0]:.0%}
(eq:evaluation-sets-decay-silently).

The gate's catch rate falls with it -- from {tab[0][1]:.0%} to {tab[104][1]:.0%} -- so
after two years a suite that was catching three quarters of defects catches under a
fifth.

The second table is why nobody notices. **The gate keeps reporting the same pass rate**,
because it is running the same cases against a system that still handles those cases.
Its own measurement is unaffected by the drift. The reported number stays at
{0.93:.0%} while the true catch rate falls to {tab[104][1]:.0%} -- a gap of
{0.93 - tab[104][1]:.0%}, with nothing to indicate it.

That is ch:sd-architecture's pattern once more, and this instance is particularly clean:
**the instrument is measuring exactly what it was built to measure, and what it was built
to measure stopped being the question.**

The cadence table gives the fix and its price. Refreshing every {4} weeks holds mean
coverage at {cad[4][1]:.0%}; every {26} weeks holds {cad[26][1]:.0%}; annually,
{cad[52][1]:.0%}.

The cost table finds the optimum at **every {best} weeks**, at {tot[best][3]:,.0f} a
year against {tot[52][3]:,.0f} for annual refresh -- driven by escape cost rather than
labelling cost, since labelling {SET_SIZE} cases is
{SET_SIZE * LABEL_COST_PER_CASE:,.0f} a time and one escaped defect is
{DEFECT_COST:,.0f}.

The growth table is the intervention teams reach for instead, and it does not work.
Quadrupling the set to {3600} cases without refreshing leaves coverage at
{coverage(52):.0%} and the catch rate at {coverage(52) * GATE_CATCH_ON_COVERED:.0%},
because **a bigger sample of an old distribution is still a sample of an old
distribution**. Refreshing {900} cases quarterly reaches
{cad[13][1] * GATE_CATCH_ON_COVERED:.0%} for less labelling than the quadrupling cost.

Size answers a variance question and age answers a bias question, and adding cases
addresses the wrong one.

The rolling table is the design that gets the coverage cheaply. Replacing
{0.25:.0%} of the set each month keeps mean case age at {roll[0.25][0]:.1f} weeks and
coverage at {roll[0.25][1]:.0%}, for {roll[0.25][2]:,.0f} a year of labelling --
against {roll[1.0][2]:,.0f} for full monthly regeneration.

**A rolling refresh gets most of a full regeneration's coverage for a quarter of the
labelling**, and it has a second advantage the table does not show: it produces a steady
small labelling workload rather than a periodic large one, which is the difference
between a process that survives and one that gets skipped when a quarter is busy.

One caution about all of this. The drift rate is the parameter everything depends on and
it is the one nobody measures. It is estimable -- compare the embedding distribution of
this month's traffic against the evaluation set's -- and until it is measured, the
cadence above is arithmetic on a guess.""")
```

```
  set age (weeks)   coverage   gate catches   escapes   vs new set
--------------------------------------------------------------------
                0       100%            74%       26%        1.00x
                4        87%            64%       36%        1.38x
               12        65%            48%       52%        1.99x
               26        40%            29%       71%        2.72x
               52        16%            12%       88%        3.40x
              104         2%             2%       98%        3.78x
```

At 3.5% weekly drift, a one-year-old set covers **16%** of traffic and catches **12%** of
defects.

And the reason nobody notices:

```
  set age (weeks)   pass rate reported   true catch rate      gap
--------------------------------------------------------------------
                0                  93%               74%      19%
               26                  93%               29%      64%
               52                  93%               12%      81%
              104                  93%                2%      91%
```

**The reported number does not move**
({{eq:evaluation-sets-decay-silently}}). The suite runs the same cases against a system
that still handles them; its own measurement is unaffected by the drift.

Refresh cadence and its cost:

```
   refresh every   coverage at worst   mean coverage   regenerations/yr
--------------------------------------------------------------------------
             2w                 93%             98%               26.0
             4w                 87%             95%               13.0
             8w                 75%             89%                6.5
            13w                 63%             81%                4.0
            26w                 40%             66%                2.0
            52w                 16%             46%                1.0
```

And why adding cases is not a substitute:

```
                          strategy    cases   coverage   catches   labelling/yr
--------------------------------------------------------------------------------
        900 cases, never refreshed      900        16%       12%               0
       1800 cases, never refreshed     1800        16%       12%           3,690
       3600 cases, never refreshed     3600        16%       12%          11,070
    900 cases, refreshed quarterly      900        81%       60%          14,760
      900 cases, refreshed monthly      900        95%       70%          47,970
```

**Quadrupling the set leaves coverage at 16%** ({{eq:refresh-beats-growth}}), because a
bigger sample of an old distribution is still old. Size answers a variance question and
age answers a bias question.

Rolling replacement gets the coverage cheaply:

```
  replace per month   mean age (weeks)   coverage   labelling/yr
------------------------------------------------------------------
               100%                2.0        93%         47,970
                50%                4.0        87%         23,985
                25%                8.0        75%         11,993
                10%               20.0        49%          4,797
                 5%               40.0        24%          2,398
```

**25% monthly replacement holds 75% coverage for 11,993 a year** against 47,970 for full
monthly regeneration — and it produces a steady small workload rather than a periodic
large one, which is the difference between a process that survives and one that gets
skipped in a busy quarter.

## 10. Production Considerations

Add a format check on the assembled prompt this week. It is the highest-ratio gate
available, it is string assertions, and it does not exist in most systems.

Route console prompt edits through the same checks as committed ones. The console's
benefit is real; losing the gates is not a necessary price for it.

Apply golden-output tests to the assembled prompt, not the generated text. The prompt is
deterministic, so the technique {{ch:sd-architecture}} found broken for outputs works at
full strength here.

Measure your drift rate before choosing a refresh cadence. Embed a week of traffic against
the evaluation set and count the unmatched share; without it the cadence is arithmetic on
a guess.

Refresh on a rolling basis rather than regenerating. A quarter of the labelling for most
of the coverage, and a workload that survives busy quarters.

Report evaluation-set *coverage* alongside pass rate, permanently. The pass rate cannot
decay and the coverage can, so a dashboard showing only the first is showing the half that
cannot go wrong.

Stop treating set size as the quality lever. It addresses variance; the problem is bias,
and the two do not substitute.

## 11. Common Mistakes

**Concluding prompts cannot be gated because they cannot be compiled.** The leap from no
compiler to no gates is the error.

**Assuming prompt defects reflect carelessness.** Identical care, identical defect rate,
and the escape rate still differs by 12×.

**Choosing gates by catch rate.** Rank by catch per unit effort; the cheapest gate wins.

**Reading a stable pass rate as a healthy gate.** It cannot move, which is the problem.

**Growing the evaluation set instead of refreshing it.** Size fixes variance and the
problem is age.

**Diagnosing coverage decay as concept drift.** Different mechanism, different remedy, one
wasted quarter.

## 12. Failure Modes

**Silent gate death.** The evaluation suite passes at its usual rate for two years while
catching almost nothing, and the team's confidence in it rises with its age.

**Prompt edited to pass the gate.** Someone adjusts a prompt until the evaluation set
scores well, which is overfitting to a sample of a distribution that has moved — the worst
of both problems.

**Format check that checks the template.** Asserting on the pre-substitution template
rather than the assembled prompt misses exactly the class of bug the check exists for.

**Refresh that resamples from cached traffic.** Regenerating the set from a stored corpus
rather than from live traffic reproduces the old distribution and the coverage does not
move.

**Console change with no record.** The prompt is edited, the behaviour shifts, and
{{ch:ops-versioning}}'s candidate space grows by an artefact nobody knows changed —
so the eventual investigation is searching a space it does not know the size of.

## 13. Alternatives

**Prompt as code, no console.** Removes the gap entirely by removing the ungated path;
costs the operational flexibility that made the console attractive.

**Continuous evaluation on live traffic.** Score a sample of production answers instead of
maintaining a fixed set. No decay by construction, and it requires
{{ch:sd-fault-tolerance}}'s judge and gives no pre-deployment gate.

**Held-out slice of recent traffic.** A rolling window used as the evaluation set, labelled
continuously. This is the rolling-refresh design taken to its limit and it is probably the
right answer for most teams.

**Synthetic case generation.** Generate evaluation cases from a specification rather than
sampling traffic. Cheap to refresh and it tests the specification rather than reality,
which is a different and narrower guarantee.

**Accept the decay and re-baseline on incident.** Rebuild the set whenever an escaped
defect reveals a gap. Reactive, cheap, and it means the set is always one incident behind.

## 14. Evaluation

Report gate coverage per artefact — which gates each change type passes. It is a table
anyone can produce in an hour and most teams have never drawn.

Measure the actual escape rate by artefact from incident history. The prior here is
illustrative; yours is countable.

Publish evaluation-set coverage as a first-class metric alongside pass rate, and alert on
it falling below a threshold.

Track case age distribution, not just set size. A set with a healthy mean age and a stale
tail behaves differently from one uniformly refreshed.

Test the format check against known past prompt defects. If it would not have caught the
last three, it is asserting the wrong things.

## 15. Advanced Concepts

The gate-independence assumption is optimistic in a familiar way. A reviewer and a test
suite often miss the same defect, because both are looking for what the author was thinking
about rather than what they were not — so $\prod\kappa_g$ understates the true escape rate.
The correction matters most for the well-gated artefacts, since more gates means more
opportunity for correlated blind spots, and it means the code-versus-prompt gap in
{{sec:9-practical-example}} is somewhat overstated. The qualitative conclusion is
unaffected: zero gates escape at one hundred percent under any correlation structure.

The drift model treats coverage as a scalar, but drift is not uniform across the
distribution. Some regions of traffic are stable for years — the core use case, the
common phrasings — while others turn over monthly. That means a fixed evaluation set
retains full coverage of the stable core and loses the periphery entirely, so the true
picture is not uniform decay but **a set that increasingly tests only what was already
working**. That is worse than the scalar model suggests, because the periphery is where
new features and new failure modes live.

The cost model also treats every escaped defect as equally expensive, which
{{ch:sd-routing-caching}} already showed is false: a wrong answer on a low-stakes surface
and a wrong answer in a price quotation differ by orders of magnitude. Since the
evaluation set can be stratified by surface, the refresh cadence should be too -- refresh
the high-stakes slice monthly and the low-stakes slice annually, rather than picking one
cadence for a set that mixes them. That is a strictly better allocation of the same
labelling budget, and it requires knowing which surface each case belongs to, which most
evaluation sets do not record.

There is a connection to {{ch:ops-observability}}'s sampling result that neither chapter
develops. An evaluation set built by sampling production traffic inherits whatever bias the
sampling had — so a set built from a biased trace sample has the detector's recall profile
baked into it, and will systematically under-test the failure modes the detector cannot
see. **Building the evaluation set from the uniform stratum is therefore not a detail but
a requirement**, and it is one more reason to keep that stratum.

## 16. Connection to Previous Chapters

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} ranked the
prompt second by payback for reproducibility. This chapter reaches the same conclusion from
quality, which is two independent arguments for the same afternoon of work.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is why the decayed
gate is silent — and this is a particularly clean instance, since the instrument is
measuring exactly what it was built to measure.

{{eq:rework-cost-is-set-by-detection-lateness}} from {{ch:ops-lifecycle}} prices an escaped
prompt defect: it is found late, and late means a long return trip.

{{eq:biased-sampling-distorts-composition}} from {{ch:ops-observability}} bears on how the
evaluation set is built, and {{sec:15-advanced-concepts}} draws the link.

## 17. Exercises

1. Count the gates each artefact in your system passes. What is the escape rate for each,
   and what share of your incidents does each produce?

2. Write the format check for your own assembled prompt. How many assertions, and would it
   have caught your last three prompt defects?

3. Derive the refresh cadence that minimises total cost for your set size, labelling cost,
   and defect cost.

4. Estimate your drift rate by embedding a week of traffic against your evaluation set.
   What cadence does it imply?

5. Model non-uniform drift with a stable core and a fast-turnover periphery. How much worse
   is coverage of the periphery than the scalar model suggests?

## 18. Interview Questions

1. Why do prompt changes cause disproportionately many incidents, assuming equal care?

2. What is the cheapest gate you could add to a prompt, and what does it assert?

3. Our evaluation suite has passed at 93% for two years. Is that reassuring?

4. We doubled our evaluation set and defects still escape. Why?

5. How would you measure whether your evaluation set still represents production?

6. Gating prompts would slow down how fast we can respond to complaints. How would
   you design a gate that keeps the response time?

## 19. Research Questions

1. How correlated are gate failures in practice, and how far does that close the
   code-versus-prompt gap?

2. What is the real drift rate for production AI traffic, and how much does it vary by
   surface?

3. How much of an evaluation set's decay is concentrated in the periphery, and does a
   stratified refresh outperform a uniform one?

4. Can prompt defects be predicted from the diff — a static analysis for prompts — well
   enough to gate on?

## 20. Chapter Summary

Application code passes five gates and escapes at **8.3%**. A prompt passes zero and
escapes at **100%** ({{eq:prompt-is-ungated-code}}) — every defective change reaches
production. Prompts and few-shot examples are **63%** of changes and **88%** of escaped
defects, with identical assumed defect rates at the moment of writing.

The cheapest fix is a format check on the assembled prompt: **38%** of defective changes
removed for half a unit of effort, the best ratio available
({{eq:format-check-is-the-cheapest-gate}}). Applying code's full gate coverage takes prompt
defects from **2.20** to **0.18** a week — **12× fewer**, with no new technology.

An evaluation set decays. At 3.5% weekly drift, coverage falls from **100%** to **16%**
over a year and the catch rate from **74%** to **12%** — while **the reported pass rate
stays at 93%** ({{eq:evaluation-sets-decay-silently}}), because the cases have not changed.

And growth does not substitute for refresh: quadrupling to 3,600 cases leaves coverage at
**16%**, while refreshing 900 quarterly reaches **81%** for less labelling
({{eq:refresh-beats-growth}}). Rolling replacement of **25%** monthly holds **75%**
coverage for a quarter of full regeneration's cost.

Both halves describe a control that is absent rather than broken. There is no failing
prompt gate to fix, because there is no prompt gate; there is no alarm on evaluation
coverage, because coverage is not measured. That is a harder class of problem to notice
than a control performing badly, and it is why both survive in otherwise disciplined
teams: nothing is red, and nothing was ever green either.

Carry forward: **the prompt is code and it has no gates**, and **report evaluation
coverage, not just pass rate**.

## 21. Further Reading

- {{cite:sculley2015}} — configuration debt, of which the ungated prompt is the sharpest
  modern case.
- {{cite:breck2017}} — a readiness rubric with a configuration-testing section almost
  nobody applies to prompts.
- {{cite:gama2014}} — concept drift, whose mechanism differs from coverage decay and is
  routinely confused with it.
- {{cite:paleyes2020deployment}} — obstacles at every stage, many of which are gates that
  were never built.
