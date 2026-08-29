---
id: as-specialized
number: 168
part: XVIII
tier: full
status: draft
requires: [horizon-changes-the-failure, gate-on-consequence,
           distinctness-not-count, verifier-quality-ceiling]
provides: [verifier-sets-the-ceiling, retry-needs-a-verifier,
           reversibility-dominates, domain-properties-are-complementary,
           specialization-is-affordance-building]
citations: [zhou2024webarena, liu2024agentbench, cemri2025mast,
            schick2023toolformer, shinn2023reflexion, yao2023react]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a domain's difficulty
predicts agent success less well than its verifier does; state the condition under
which retry is a correction rather than a fresh sample; rank the affordances that
separate coding agents from computer-use agents by measured effect rather than by
intuition; explain why individual improvements to hard-domain agents so often
measure as worthless; and describe what specialising an agent for a domain actually
consists of.

## 2. Why This Matters

Every survey of agent capability reports the same ordering: coding agents work,
data agents mostly work, research agents are unreliable, browser and computer-use
agents are experimental. The explanation attached is difficulty — those tasks are
harder, the model is not good enough yet, wait for the next one.

{{sec:9-practical-example}} tests that explanation and finds it does not survive
contact with its own premise. Across five domain profiles, per-step difficulty
correlates $0.71$ with task success. **Verifier quality correlates $0.96$.**

The clearest case is two rows of one table. A research profile has the *higher*
per-step success — $90\%$ against coding's $82\%$ — and reaches $59.0\%$ task
success against coding's $94.3\%$. On the variable everyone reaches for, research is
the easier domain, and it loses by thirty-five points.

The difference is that a coding agent has a compiler. It can tell a bad step from a
good one $97\%$ of the time; a research agent, checking sources against each other,
manages $45\%$. And {{ch:ag-recovery}}'s retry needs something to retry *against*:
with a good verifier four retries are worth $+71$ points, and with a poor one the
same four are worth $+14$ ({{eq:retry-needs-a-verifier}}).

The second listing asks the same question about computer-use and produces a result
it was not built to produce. It was written to argue that partial observability —
you cannot really see a screen — matters more than action-space size. Reversibility
beat both ({{eq:reversibility-dominates}}), and more importantly the three turned
out to be complementary rather than substitutable: interventions worth $+1.0$ and
$+0.3$ points alone were worth $+54.5$ together
({{eq:domain-properties-are-complementary}}).

Which explains a pattern otherwise hard to account for — that incremental
improvements to hard-domain agents so often measure as nothing, right up until they
suddenly measure as everything.

## 3. Prerequisites

{{ch:ag-recovery}}'s retry, whose value this chapter makes conditional on a
verifier.

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}}, extended here: the
*affordances* are the interface, and a domain that offers no undo is an interface
problem before it is a model problem.

{{ch:ag-termination}}'s {{eq:gate-on-consequence}}, which is where irreversibility
goes when it cannot be removed.

{{ch:rsn-test-time-compute}}'s verifier-quality ceiling, which found sampled
compute capped by the verifier that selects among the samples. This chapter is that
result transposed from reasoning to action, where the ceiling is set by whether the
domain lets you check a step at all.

{{ch:as-long-running}} for the oversight placement result this chapter's second
listing points back at.

## 4. Intuitive Explanation

Ask why coding agents work better than browser agents and you will usually be told
that code is more structured, or that there is more code in the training data, or
that clicking is just harder.

Consider instead what happens after a mistake in each.

The coding agent writes a function, runs the tests, and sees a red line with a stack
trace. It knows something is wrong, roughly where, and often why. It edits and runs
again. If the edit made things worse, `git checkout` puts everything back.

The browser agent clicks something, and a page loads. Is that the right page? It
looks like a page. There is no test suite for "did that click accomplish what I
intended", so the agent's evidence is a screenshot that mostly looks like the
screenshot it expected. If the click was wrong — submitted the form, deleted the
item, sent the message — there is no `git checkout`.

Neither of those differences is about difficulty. They are about what the domain
*affords*: whether you can tell that you erred, and whether you can undo it.

That reframing has a consequence for retry, which is the mechanism most reliability
work rests on. Retry works by re-running the thing that failed. If you cannot tell
which thing failed, you re-run at random, and re-running at random is just sampling
again — you get the base rate back, not a correction. {{sec:9-practical-example}}
measures exactly that gap.

So the ordering of domains is largely an ordering of verifiers, and "specialising an
agent for this domain" mostly means building the verifier the domain does not
supply.

The second listing pushes on the browser and computer-use end and finds something
sharper. Three things separate those domains from coding: a much bigger action
space, much worse observations, and no undo. The intuition — mine included, which is
why the listing was written that way — is that observation is the binding one.

It is not. Undo explains the most spread. And the counterfactuals are the real
result: fixing *any one* of the three for a computer-use profile is worth almost
nothing, and fixing two is worth fifty-four points.

That is what a system with several simultaneously-binding constraints looks like.
Relieving one while the others still bind produces no measurable improvement, which
makes a research programme look stalled while it is in fact accumulating the
components of a discontinuity.

## 5. Formal Explanation

Let a task require $n$ steps, each succeeding with probability $p$. A verifier flags
a failed step with probability $d$ (detection) and a good step with probability
$\phi$ (false alarm). With up to $r$ retries per step, the per-step success becomes:

$$p_r = p + (1-p)\,d\,\big(1 - (1 - p)^{r}\big) - \varepsilon(\phi)$$ (eq:retry-needs-a-verifier)

with $\varepsilon(\phi)$ the waste from re-running good steps. The retry term is
*multiplied by $d$*. At $d \to 0$ the whole benefit vanishes regardless of $r$:
**retry without detection is a fresh sample, not a correction.**

Task success is $p_r^{\,n}$, so:

$$\frac{\partial \log S}{\partial d} = n\,\frac{(1-p)(1-(1-p)^r)}{p_r} \qquad\text{versus}\qquad \frac{\partial \log S}{\partial p} = \frac{n}{p_r}\Big(1 - d(1-(1-p)^r) + \ldots\Big)$$ (eq:verifier-sets-the-ceiling)

Both are $O(n)$, so neither dominates asymptotically — this is not a claim that
difficulty is irrelevant. The claim is empirical and about *range*: across observed
domains, $d$ varies from $0.25$ to $0.97$ while $p$ varies from $0.72$ to $0.90$,
and $d$ is the one under a team's control.

Now the second listing. Let the agent choose among $N$ candidate actions, observe
the state faithfully with probability $f$, and undo a mistake with probability $u$.
Model action choice as skill-weighted over candidates:

$$\Pr[\text{right}] = f \cdot \frac{s}{s + \log_2 N} + (1-f)\cdot\frac{1}{N}$$ (eq:action-choice-under-partial-observation)

Note $N$ enters logarithmically under a faithful observation and linearly under an
unfaithful one — the action space hurts far more when you cannot see. That is
already an interaction rather than two separate effects.

Undo enters differently. A wrong action costs a turn if undoable and costs progress
if not, so the effective progress rate is:

$$\dot{\pi} = \Pr[\text{right}] - (1-u)\big(1 - \Pr[\text{right}]\big)$$ (eq:reversibility-dominates)

which crosses zero at $\Pr[\text{right}] = (1-u)/(2-u)$. **Below that threshold the
run makes negative progress and no budget saves it** — which is why the low-$u$
profiles collapse rather than degrade.

Composing these, the sensitivity to any one variable depends on the others:

$$\frac{\partial S}{\partial f}\Big|_{u \text{ small}} \approx 0, \qquad \frac{\partial S}{\partial u}\Big|_{f \text{ small}} \approx 0$$ (eq:domain-properties-are-complementary)

**Each partial derivative is near zero while another constraint binds**, so the
joint effect exceeds the sum of the marginal effects by a large factor —
$+54.5$ against $+1.3$ in {{sec:9-practical-example}}.

The practical form of all this:

$$\text{specialisation} = \text{raise } d,\ \text{raise } f,\ \text{raise } u \quad\text{— jointly}$$ (eq:specialization-is-affordance-building)

none of which is a property of the model.

## 6. Mathematical Foundation

Three extractions.

**Detection multiplies the retry term, so it gates every recovery mechanism.** From
{{eq:retry-needs-a-verifier}}, anything that works by noticing and re-doing —
{{cite:shinn2023reflexion}}'s reflection, {{ch:ag-recovery}}'s replanning,
{{ch:ag-planning}}'s checkpoints — inherits the same factor $d$. A domain with weak
detection does not merely have weaker retry; it has weaker *everything downstream of
noticing*, which is most of {{part:17}}.

**The action space enters logarithmically when you can see and linearly when you
cannot.** {{eq:action-choice-under-partial-observation}} says the "huge action
space" complaint is really a complaint about the interaction. A well-observed
thousand-action interface is not much worse than a well-observed forty-action one;
a poorly-observed one is catastrophically worse. This is why accessibility trees
help browser agents more than the raw action-count reduction predicts.

**There is a progress threshold, not a gradient.** From
{{eq:reversibility-dominates}}, when $\Pr[\text{right}]$ falls below
$(1-u)/(2-u)$ the expected progress is negative and more budget makes things worse
rather than better. That is a qualitative change of regime, and it explains why the
weak profiles in {{sec:9-practical-example}} sit at $0.0\%$ rather than at some small
positive number.

## 7. Internal Mechanics

### 7.1 The five domains as affordance profiles

```mermaid {#fig:domain-affordances caption="Domains ordered by what they afford rather than by difficulty. The ordering of agent success follows this diagram, not a difficulty ranking."}
flowchart TD
    C["coding<br/>verifier: compiler + tests<br/>observation: exact<br/>undo: version control"]
    D["data<br/>verifier: schema + types<br/>observation: exact<br/>undo: transactions"]
    R["research<br/>verifier: source agreement<br/>observation: good<br/>undo: nothing to undo"]
    B["browser<br/>verifier: page changed<br/>observation: rendered<br/>undo: rarely"]
    U["computer-use<br/>verifier: screenshot<br/>observation: pixels<br/>undo: no"]
    C --> D --> R --> B --> U
```

Reading the diagram as a difficulty ranking is the mistake this chapter exists to
correct. It is an affordance ranking, and the success ordering follows it because
the affordances are what the reliability machinery runs on.

Note that research breaks the pattern in an instructive way: its observations are
good and there is little to undo, but its *verifier* is weak, and that alone drops it
below data.

### 7.2 Building a verifier where the domain does not supply one

The chapter's main practical claim is that this is the work. Four patterns, roughly
in order of strength.

**Execute something.** The strongest verifiers all reduce to running code and
checking the result. Where a domain permits any executable check — a query that must
return rows, a config that must parse, a URL that must resolve — that check is worth
more than any amount of model-based review.

**Check an invariant.** Weaker but always available: properties that must hold
regardless of the task. Row counts that should not decrease, totals that should
reconcile, a document that should still be valid JSON.

**Cross-check independent sources.** The research domain's option, and its $45\%$
detection in {{sec:9-practical-example}} is the honest rate — it catches
disagreement, which is a subset of error.

**Ask a model.** The weakest and the most commonly deployed, with the caution
{{part:16}} attached: a model checking its own work shares its own blind spots, so
measured detection on *your* error distribution is the only number that means
anything.

### 7.3 Improving observation fidelity is an engineering problem

Browser agents improved substantially when accessibility trees and structured DOM
extraction replaced raw screenshots, and
{{eq:action-choice-under-partial-observation}} says why the gain exceeded what the
action-count reduction alone would predict: fidelity and action space interact.

The general move is to replace a *rendering* with the *state that produced it*. A
screenshot is a lossy projection of a UI tree; the UI tree is available. A
terminal scrollback is a lossy projection of process state. Wherever the underlying
structured state can be exposed, exposing it raises $f$ directly, and
{{cite:zhou2024webarena}}'s environment is built on exactly that observation.

### 7.4 Manufacturing an undo

Where a domain has no undo, one can often be built, and the second listing says this
is the highest-value single affordance.

**Transactions**, where the substrate supports them — the data domain's advantage,
and the reason it outperforms its raw difficulty.

**Snapshots.** A copy-on-write filesystem, a database snapshot, a VM checkpoint
turns an irreversible environment into a reversible one at storage cost. This is
{{ch:as-state-machines}}'s durable checkpoint used for rollback rather than resume.

**Compensating actions.** A recorded inverse per operation — delete the row that was
inserted, send the correction. Weaker than real undo because compensation can fail,
and it is what {{ch:as-state-machines}} recommended for non-idempotent effects.

**Staging.** Act on a copy and promote when verified. This converts an irreversible
domain into a reversible one plus a single irreversible promotion step, and that one
step is where {{eq:gate-on-consequence}} puts the human.

Staging deserves emphasis because it composes the whole chapter: it raises $u$ across
almost every step, concentrates irreversibility into one place, and makes that place
cheap to gate.

### 7.5 Why incremental progress looks like no progress

{{eq:domain-properties-are-complementary}} has a research-management consequence
worth stating explicitly.

A team improving a computer-use agent's observation fidelity from $62\%$ to $97\%$ —
a serious engineering effort — would measure $+0.3$ points. The natural conclusion is
that the approach is wrong. The correct conclusion is that undo is still binding and
the fidelity gain is stored, not lost.

**In a multiply-constrained domain, the marginal value of relieving one constraint is
near zero, and the marginal value of relieving the last one is enormous.** Evaluating
such work by its isolated ablation will systematically defund the components of the
eventual jump. The alternative is to evaluate against a profile where the *other*
constraints are artificially relieved — which measures the contribution rather than
the current marginal effect.

### 7.6 Where specialisation genuinely is about the model

Not everything is affordances, and two things are not.

**Domain vocabulary and conventions.** An agent that does not know a field's idioms
proposes actions that are wrong in ways no verifier catches, because they are
plausible. This is real and it is what fine-tuning addresses.

**Long-tail tool competence.** {{cite:schick2023toolformer}}'s concern: knowing
*when* to reach for a tool. This degrades with tool count in ways
{{ch:ag-tool-calling}} measured, and it is not fixed by better verification.

Both are genuine. Neither explains the ordering in
{{sec:9-practical-example}}'s first table, which is the point of putting them here
rather than at the front.

### 7.7 Benchmarks measure affordances too

{{cite:zhou2024webarena}} and {{cite:liu2024agentbench}} report scores per
environment, and those environments differ in exactly the variables this chapter
isolates — verifier availability most of all, since a benchmark needs an automatic
success check and therefore *only exists* where detection is possible.

That is a selection effect worth naming. A benchmark can only score a domain where
someone built a verifier, so benchmark coverage is biased toward the domains agents
already do well in, and the hardest domains are underrepresented *because* they are
hard for the reason this chapter identifies.

### 7.8 This is the verifier ceiling, moved from reasoning to action

{{ch:rsn-test-time-compute}} found that sampling more reasoning traces buys very
little unless something can pick the good one, and that the selector's quality caps
the whole method. Sixty-four samples with a weak
verifier lose to eight with a strong one.

{{eq:retry-needs-a-verifier}} is the same statement about actions. Retrying a step
generates more attempts; detection is what turns attempts into progress. Both are
instances of one structure: a generate-and-select loop whose ceiling is the selector,
not the generator.

Stating it that way makes the domain ordering less surprising. It is not that
browser tasks are mysteriously hard — it is that the browser domain is a
generate-and-select loop in which nobody has built the select half, and
{{ch:rsn-test-time-compute}} already measured what that costs in a setting where the
tasks were easy and well understood.

It also predicts where effort pays. In reasoning, the productive work of the last
few years was largely verifier work — process supervision, outcome reward models,
executable checks. **The same prediction for agents is that the productive work is
detection**, and the domains that got good are the ones where detection came free
with the substrate.

Which is a slightly deflating account of why coding agents succeeded first, and
probably the right one: not that code is what these models are best at, but that
`pytest` exists.

## 8. Implementation

Two listings. The first tests difficulty against verifier quality as predictors of
success. The second separates action space, observation fidelity and reversibility.

```python {tier=A name=verifier-sets-the-ceiling}
"""What actually separates a coding agent from a browser agent.

The usual account is difficulty: coding is 'easier' for agents than computer-use
because the tasks are more structured. This listing proposes a different variable
and measures which one predicts success.

Every domain differs in whether the agent can CHECK its own work, and how well:

  coding        a compiler and a test suite -- cheap, fast, near-perfect
  data          schema and type checks -- cheap, partial
  research      other sources agreeing -- expensive, weak
  browser       the page changed, somehow -- cheap, very weak
  computer-use  a screenshot -- cheap, very weak

ch:ag-recovery's retry needs a verifier to retry AGAINST. Without one, a retry is
a fresh sample rather than a correction (eq:retry-needs-a-verifier), so a domain's
ceiling may be set by its verifier rather than by its difficulty
(eq:verifier-sets-the-ceiling).
"""
import numpy as np

rng = np.random.default_rng(3719)

M = 40000
STEPS = 12
RETRIES = 4

# (name, per-step success, verifier detects a bad step, verifier false alarm)
DOMAINS = [
    ("coding",       0.82, 0.97, 0.02),
    ("data",         0.86, 0.80, 0.06),
    ("research",     0.90, 0.45, 0.15),
    ("browser",      0.78, 0.30, 0.10),
    ("computer-use", 0.72, 0.25, 0.12),
]


def run(p_step, p_detect, p_fa, m=M, steps=STEPS, retries=RETRIES):
    """Each step succeeds with p_step. A verifier flags bad steps with p_detect
    and good ones with p_fa. A flagged step is retried, up to `retries` times.
    An undetected bad step is carried forward and the task is wrong."""
    ok = np.ones(m, dtype=bool)
    cost = np.zeros(m, dtype=np.int64)
    for _ in range(steps):
        live = np.flatnonzero(ok)
        if not len(live):
            break
        good = rng.random(len(live)) < p_step
        cost[live] += 1
        # Retry loop: only fires on a flagged step.
        for _ in range(retries):
            bad = ~good
            flagged = np.where(bad, rng.random(len(live)) < p_detect,
                               rng.random(len(live)) < p_fa)
            redo = flagged
            if not redo.any():
                break
            cost[live[redo]] += 1
            fresh = rng.random(int(redo.sum())) < p_step
            good = good.copy()
            good[redo] = fresh
        ok[live[~good]] = False
    return float(ok.mean()), float(cost.mean())


print(f"{M:,} tasks of {STEPS} steps, up to {RETRIES} retries per step.")
print("Per-step success is the domain's raw difficulty; detection is how well")
print("the domain lets an agent tell a bad step from a good one.")
print()
print(f"{'domain':>14}{'step success':>14}{'detection':>11}{'task success':>14}"
      f"{'steps':>8}")
print("-" * 61)
tab = {}
for name, ps, pd, pf in DOMAINS:
    r = run(ps, pd, pf)
    tab[name] = (ps, pd, r[0], r[1])
    print(f"{name:>14}{ps:>14.0%}{pd:>11.0%}{r[0]:>14.1%}{r[1]:>8.1f}")

print()
print()
print("Which variable predicts the outcome? Correlations across the five domains:")
print()
xs = np.array([v[0] for v in tab.values()])
ds = np.array([v[1] for v in tab.values()])
ys = np.array([v[2] for v in tab.values()])
c_diff = float(np.corrcoef(xs, ys)[0, 1])
c_ver = float(np.corrcoef(ds, ys)[0, 1])
print(f"{'per-step success (difficulty) vs task success':>50}{c_diff:>10.2f}")
print(f"{'verifier detection vs task success':>50}{c_ver:>10.2f}")

print()
print()
print("The controlled version: hold difficulty fixed and sweep the verifier,")
print("then hold the verifier fixed and sweep difficulty.")
print()
print(f"{'detection':>11}{'task success':>14}      {'step success':>14}"
      f"{'task success':>14}")
print("-" * 69)
sweep_v, sweep_d = {}, {}
DET = (0.25, 0.45, 0.70, 0.90, 0.97)
DIF = (0.72, 0.78, 0.82, 0.86, 0.90)
for pd, ps in zip(DET, DIF):
    a = run(0.80, pd, 0.08)[0]
    b = run(ps, 0.60, 0.08)[0]
    sweep_v[pd] = a
    sweep_d[ps] = b
    print(f"{pd:>11.0%}{a:>14.1%}      {ps:>14.0%}{b:>14.1%}")

print()
print()
print("And what retries are worth with and without a verifier -- the mechanism")
print("behind the whole table.")
print()
print(f"{'retries':>9}{'detection 97%':>16}{'detection 60%':>16}"
      f"{'detection 25%':>16}")
print("-" * 57)
rt = {}
for k in (0, 1, 2, 4, 8):
    row = tuple(run(0.80, pd, 0.08, retries=k)[0] for pd in (0.97, 0.60, 0.25))
    rt[k] = row
    print(f"{k:>9}{row[0]:>16.1%}{row[1]:>16.1%}{row[2]:>16.1%}")

print(f"""
Compare the coding and research rows, which is the whole listing in two lines.

Research has the HIGHER per-step success -- {tab['research'][0]:.0%} against coding's
{tab['coding'][0]:.0%} -- and the lower task success:
{tab['research'][2]:.1%} against {tab['coding'][2]:.1%}. On the variable everyone
reaches for, research is the easier domain, and it loses by
{tab['coding'][2] - tab['research'][2]:.1f} points.

The difference is the other column. A coding agent has a compiler and a test suite:
it can tell a bad step from a good one {tab['coding'][1]:.0%} of the time. A research
agent has other sources agreeing, at {tab['research'][1]:.0%}.

The correlations make it quantitative. Across the five domains, per-step difficulty
correlates {c_diff:.2f} with task success and verifier detection correlates
{c_ver:.2f}. **A domain's ceiling is set more by whether the agent can check its own
work than by how hard the work is** (eq:verifier-sets-the-ceiling).

Five points is a small sample and the profiles are hand-set, so the controlled
sweep matters more. Holding difficulty at {0.80:.0%} and moving detection from
{0.25:.0%} to {0.97:.0%} moves task success from {sweep_v[0.25]:.1%} to
{sweep_v[0.97]:.1%}. Holding detection at {0.60:.0%} and moving difficulty across
its full observed range moves it from {sweep_d[0.72]:.1%} to {sweep_d[0.90]:.1%}.

Both matter. The verifier range is wider, and -- more usefully -- **the verifier is
the one you can build.** Per-step difficulty is a property of the domain and the
model. Detection is a property of the tooling you wrap around it, and a test suite
is something a team can write this week.

The last table shows the mechanism, and it is ch:ag-recovery's with a condition
attached. At {0.97:.0%} detection, going from {0} to {4} retries is worth
{rt[4][0] - rt[0][0]:+.1%}. At {0.25:.0%} detection the same retries are worth
{rt[4][2] - rt[0][2]:+.1%}.

**A retry without a verifier is a fresh sample rather than a correction**
(eq:retry-needs-a-verifier). It cannot preferentially re-run the steps that went
wrong, so it re-runs everything at the same rate and buys far less. Every
reliability mechanism in part:17 that depends on noticing a failure inherits this
condition, which is why the domains at the bottom of the first table are hard in a
way that more retries do not fix.

So the practical reading of 'specialising an agent for a domain' is narrower than it
sounds. It is mostly not prompt engineering and mostly not model choice. **It is
building the domain's verifier**, and where the domain does not offer one cheaply,
that is the work.""")
```

The second listing asks what makes computer-use hard.

```python {tier=A name=reversibility-dominates}
"""Why computer-use is hard, tested against three candidate explanations.

The usual explanation is the action space: a screen has thousands of clickable
points and an API has twelve endpoints, so of course clicking is harder.

This listing was written to test a second explanation against it -- that the real
problem is whether the agent can SEE the state it is acting on, since a screenshot
is a partial, stale, ambiguous observation and a filesystem read is not.

It also includes a third, added mostly for completeness: whether a mistake can be
undone. Coding has version control, so a bad edit costs a step. A sent email, a
deleted row, a submitted form cannot be taken back.

The measurement puts the third one first, which is not what the listing was built
to show (eq:reversibility-dominates), and finds the three strongly complementary
rather than substitutable (eq:domain-properties-are-complementary).
"""
import numpy as np

rng = np.random.default_rng(3767)

M = 40000
GOAL_STEPS = 8
BUDGET = 40
P_FATAL = 0.05     # share of un-undone mistakes that cannot be recovered


def run(n_actions, fidelity, undo, m=M, goal=GOAL_STEPS, budget=BUDGET,
        skill=3.2):
    """The agent needs `goal` correct actions. Each turn it observes the state;
    with probability `fidelity` the observation is faithful, otherwise it is
    misleading. Action choice is skill-weighted over `n_actions` candidates.

    A wrong action is undone with probability `undo`, costing only the turn.
    Otherwise it leaves damage: progress falls back a step, and a small share of
    un-undone mistakes are unrecoverable outright."""
    # Probability of the right action given a faithful observation: a
    # skill-weighted softmax over n candidates.
    p_right_true = skill / (skill + np.log2(n_actions))
    # Given a misleading observation, the agent is choosing at chance.
    p_right_false = 1.0 / n_actions
    prog = np.zeros(m, dtype=np.int64)
    alive = np.ones(m, dtype=bool)
    used = np.zeros(m, dtype=np.int64)
    for _ in range(budget):
        live = np.flatnonzero(alive & (prog < goal))
        if not len(live):
            break
        used[live] += 1
        faithful = rng.random(len(live)) < fidelity
        p = np.where(faithful, p_right_true, p_right_false)
        right = rng.random(len(live)) < p
        prog[live[right]] += 1
        wrong = live[~right]
        if len(wrong):
            stuck = wrong[rng.random(len(wrong)) >= undo]
            prog[stuck] = np.maximum(prog[stuck] - 1, 0)
            fatal = stuck[rng.random(len(stuck)) < P_FATAL]
            alive[fatal] = False
    done = alive & (prog >= goal)
    return float(done.mean()), float(used.mean())


# (name, action space, observation fidelity, undo probability)
PROFILES = [
    ("coding",       40,   0.97, 0.99),
    ("data",         25,   0.93, 0.90),
    ("research",    200,   0.85, 0.97),
    ("browser",     600,   0.70, 0.55),
    ("computer-use", 4000, 0.62, 0.40),
]

print(f"{M:,} tasks needing {GOAL_STEPS} correct actions within {BUDGET} turns.")
print()
print(f"{'domain':>14}{'actions':>9}{'fidelity':>10}{'undo':>7}"
      f"{'success':>10}{'turns':>8}")
print("-" * 58)
prof = {}
for name, n, f, u in PROFILES:
    r = run(n, f, u)
    prof[name] = (n, f, u, r[0], r[1])
    print(f"{name:>14}{n:>9}{f:>10.0%}{u:>7.0%}{r[0]:>10.1%}{r[1]:>8.1f}")

print()
print()
print("One variable at a time, the other two held at the coding profile.")
print()
print(f"{'action space':>14}{'success':>10}   {'fidelity':>10}{'success':>10}"
      f"   {'undo':>8}{'success':>10}")
print("-" * 68)
NS = (40, 200, 600, 1500, 4000)
FS = (0.97, 0.90, 0.80, 0.70, 0.62)
US = (0.99, 0.90, 0.75, 0.55, 0.40)
sw_n, sw_f, sw_u = {}, {}, {}
for n, f, u in zip(NS, FS, US):
    a = run(n, 0.97, 0.99)[0]
    b = run(40, f, 0.99)[0]
    c = run(40, 0.97, u)[0]
    sw_n[n], sw_f[f], sw_u[u] = a, b, c
    print(f"{n:>14}{a:>10.1%}   {f:>10.0%}{b:>10.1%}   {u:>8.0%}{c:>10.1%}")

print()
print()
print("Ranges over each sweep -- how much of the spread each variable explains.")
print()
rng_n = max(sw_n.values()) - min(sw_n.values())
rng_f = max(sw_f.values()) - min(sw_f.values())
rng_u = max(sw_u.values()) - min(sw_u.values())
for label, v in (("action space", rng_n), ("observation fidelity", rng_f),
                 ("undo availability", rng_u)):
    print(f"{label:>24}{v:>10.1%}")

print()
print()
print("The counterfactual that matters for tooling: give the computer-use")
print("profile ONE of the other domains' properties at a time.")
print()
n0, f0, u0 = 4000, 0.62, 0.40
print(f"{'computer-use, plus...':>28}{'success':>10}{'gain':>9}")
print("-" * 47)
cu = {}
base = run(n0, f0, u0)[0]
cu["baseline"] = base
print(f"{'(baseline)':>28}{base:>10.1%}{'--':>9}")
for label, kw in [("a 40-action interface", dict(n_actions=40)),
                  ("faithful observations", dict(fidelity=0.97)),
                  ("reliable undo", dict(undo=0.99)),
                  ("observations + undo", dict(fidelity=0.97, undo=0.99))]:
    args = dict(n_actions=n0, fidelity=f0, undo=u0)
    args.update(kw)
    v = run(**args)[0]
    cu[label] = v
    print(f"{label:>28}{v:>10.1%}{v - base:>+9.1%}")

print(f"""
The first table reproduces the ordering everyone expects, and the sweep underneath
it disagrees about why.

Held one at a time from the coding profile, moving the action space from
{40} to {4000} costs {sw_n[4000] - sw_n[40]:.1%}. Moving observation fidelity across
the full observed range costs {sw_f[0.62] - sw_f[0.97]:.1%}. Moving undo
availability costs {sw_u[0.40] - sw_u[0.99]:.1%}.

**Reversibility explains the largest share of the spread**
(eq:reversibility-dominates), and observation fidelity -- the variable this listing
was written to promote -- explains the smallest. That is worth stating plainly
because the fidelity story is the one usually told about computer-use, and at these
parameters it is the weakest of the three.

The counterfactual table is where it becomes actionable, and it contains a stronger
result than the sweep.

Give the computer-use profile a {40}-action interface and nothing else: it goes
from {cu['baseline']:.1%} to {cu['a 40-action interface']:.1%}. Give it faithful
observations and nothing else: {cu['faithful observations']:.1%}. Give it reliable
undo alone: {cu['reliable undo']:.1%}.

Give it faithful observations AND reliable undo: {cu['observations + undo']:.1%}.

**The properties are complementary rather than substitutable**
(eq:domain-properties-are-complementary). Two interventions worth
{cu['a 40-action interface'] - cu['baseline']:.1%} and
{cu['faithful observations'] - cu['baseline']:.1%} on their own are worth
{cu['observations + undo'] - cu['baseline']:.1%} together, because each one is
useless while another is binding. Seeing the state does not help if a mistaken
action cannot be taken back; being able to take actions back does not help if you
cannot see whether they were mistaken.

That explains something otherwise puzzling about this class of system: individual
improvements to computer-use agents often measure as near-worthless, and then a
combination measures as transformative. It is not that the individual measurements
were wrong. **In a domain with several binding constraints, the marginal value of
relieving one of them is near zero**, which makes incremental progress look like no
progress until the last constraint goes.

The practical reading is the same as the previous listing's and points at the same
place. Specialising for a domain is mostly not model work. It is building the
missing affordances -- a verifier, a faithful observation, an undo -- and building
them together, because relieving one at a time will not show up in the numbers.

And where the domain genuinely cannot offer an undo, ch:ag-termination's answer
stands: that is exactly where the human gate goes, and ch:as-long-running found
placing gates on those steps worth an eightfold review budget.""")
```

## 9. Practical Example

The first listing runs twelve-step tasks across five domain profiles, with up to
four retries per step:

```
        domain  step success  detection  task success   steps
-------------------------------------------------------------
        coding           82%        97%         94.3%    15.3
          data           86%        80%         85.4%    15.8
      research           90%        45%         59.0%    16.0
       browser           78%        30%         19.5%     9.8
  computer-use           72%        25%          6.5%     7.4
```

Research has the highest per-step success in the table and finishes third.
**On difficulty it is the easiest domain, and it loses to coding by $35.3$ points.**

```
     per-step success (difficulty) vs task success      0.71
                verifier detection vs task success      0.96
```

Five hand-set profiles is a small sample, so the controlled sweep matters more:

```
  detection  task success        step success  task success
---------------------------------------------------------------------
        25%         21.2%                 72%         42.1%
        70%         66.5%                 82%         62.9%
        97%         78.1%                 90%         79.8%
```

Both variables matter; the verifier's range is wider
({{eq:verifier-sets-the-ceiling}}). And the verifier is the one a team can build —
difficulty is a property of the domain and the model.

The mechanism:

```
  retries   detection 97%   detection 60%   detection 25%
---------------------------------------------------------
        0            6.9%            6.7%            6.8%
        2           71.0%           39.4%           13.7%
        8           78.2%           66.7%           31.2%
```

Four retries are worth $+71.0$ points at $97\%$ detection and $+14.1$ at $25\%$.
**A retry without a verifier is a fresh sample rather than a correction**
({{eq:retry-needs-a-verifier}}) — and every mechanism in {{part:17}} that works by
noticing a failure inherits the same factor.

The second listing separates three candidate explanations for the browser and
computer-use end:

```
        domain  actions  fidelity   undo   success   turns
----------------------------------------------------------
        coding       40       97%    99%     98.3%    22.1
      research      200       85%    97%     71.8%    31.2
       browser      600       70%    55%      0.4%    28.5
  computer-use     4000       62%    40%      0.0%    25.0
```

Swept one at a time from the coding profile:

```
            action space     43.5%
    observation fidelity     23.4%
       undo availability     83.3%
```

**Reversibility explains the largest share of the spread**
({{eq:reversibility-dominates}}), and observation fidelity — the variable this
listing was written to promote — explains the smallest.

The counterfactuals are the stronger result:

```
       computer-use, plus...   success     gain
-----------------------------------------------
                  (baseline)      0.0%       --
       a 40-action interface      1.0%    +1.0%
       faithful observations      0.3%    +0.3%
               reliable undo     12.3%   +12.3%
         observations + undo     54.5%   +54.5%
```

Interventions worth $+1.0$ and $+0.3$ alone are worth $+54.5$ together.
**The properties are complementary rather than substitutable**
({{eq:domain-properties-are-complementary}}): seeing the state does not help if a
mistake cannot be undone, and undo does not help if you cannot see that you erred.

Which accounts for a pattern otherwise puzzling — that improvements to hard-domain
agents so often measure as nothing until, abruptly, they measure as everything.

## 10. Production Considerations

Before choosing a model for a domain, write down its detection rate. It predicts
outcomes better than anything else you will measure, and it is the number that says
whether the project is a modelling problem or an engineering one.

Build the verifier first. Execute something if you can; check an invariant if you
cannot; measure whatever you build against your own error distribution rather than
trusting a claimed rate.

Expose structured state instead of renderings. A UI tree beats a screenshot by more
than the action-count reduction predicts.

Manufacture an undo — transactions, snapshots, compensating actions, or staging.
Staging is usually the best of these because it concentrates irreversibility into a
single promotable step.

Put the human gate on that step, per {{eq:gate-on-consequence}} and
{{ch:as-long-running}}'s placement result.

Do not evaluate affordance work by its isolated ablation while other constraints
bind. Measure it against a profile with the others relieved.

And treat "wait for a better model" as a decision with a cost: for a
detection-limited domain it is waiting for the wrong thing.

## 11. Common Mistakes

**Explaining domain ordering by difficulty.** Detection correlates $0.96$;
difficulty $0.71$, and the research row inverts.

**Adding retries to a domain with no verifier.** They buy the base rate back, not a
correction.

**Trusting a claimed detection rate.** Measure it on your errors; a verifier that
shares the generator's blind spots reports well and catches little.

**Treating screenshots as observations.** They are lossy projections of state that is
usually available structured.

**Concluding an affordance improvement failed because its ablation is flat.** In a
multiply-constrained domain that is what a working component looks like.

**Leaving irreversibility distributed.** Staging concentrates it into one gateable
step, which is worth more than any per-step caution.

## 12. Failure Modes

*Silent acceptance of wrong work.* The characteristic weak-verifier failure — the
agent finishes confidently because nothing said otherwise.

*Negative progress.* Below {{eq:reversibility-dominates}}'s threshold the run
destroys more than it builds, and more budget makes it worse.

*Verifier blind-spot collapse.* A model-based verifier that agrees with the
generator, reporting high confidence at low detection.

*Plausible domain errors.* Actions that are wrong by domain convention and pass every
structural check — the case {{sec:7-internal-mechanics}} concedes is genuinely a
model problem.

*Benchmark-shaped capability.* Progress concentrated in domains where verifiers exist
because those are the domains that can be scored — a selection effect, not a
capability boundary.

## 13. Alternatives

**Restricting the domain to its verifiable subset.** Ship the part you can check;
this is what most successful agent products actually did.

**Human-as-verifier.** Substitute a person for the missing detection, with
{{ch:ag-termination}}'s habituation caveat and {{ch:as-long-running}}'s placement
rule.

**Fine-tuning for domain conventions.** Addresses the residual this chapter concedes
is real, and does not address detection.

**Building the environment rather than the agent.** {{cite:zhou2024webarena}}'s
approach: if the environment exposes structured state and supports reset, the agent
problem gets easier without touching the agent.

**Not automating the domain yet.** For a domain with no verifier, no undo, and no
structured observation, the honest answer is that the engineering to make it
tractable has not been done, and doing that engineering is the project.

## 14. Evaluation

Measure detection rate directly: seed known-bad steps and count how many your
verifier flags, and count false alarms on known-good ones. Both numbers, not one.

Report per-step and task-level success separately. The gap between them *is* the
verifier's contribution and it is the number this chapter is about.

Measure undo coverage: the fraction of your agent's actions that can actually be
reversed, from the tool audit {{ch:as-state-machines}} already asked for.

Evaluate affordance improvements with other constraints relieved, or you will
measure zero and conclude wrongly.

And check where you sit relative to
{{eq:reversibility-dominates}}'s progress threshold before spending on budget.

## 15. Advanced Concepts

**Learned verifiers for unverifiable domains.** Training a detector on human
judgements of agent trajectories, which is {{part:16}}'s process supervision applied
to actions rather than reasoning. The measured detection rate is the whole question.
{{maturity:EMERGING}}.

**Automatic affordance discovery.** An agent that determines for itself which of its
actions are reversible, by trying and observing, rather than being told.
{{maturity:EXPERIMENTAL}}.

**Environment instrumentation as a research target.** {{sec:13-alternatives}}'s
observation taken seriously: much of what looks like agent capability research would
be more efficiently pursued as environment engineering.

**Predicting the discontinuity.** {{eq:domain-properties-are-complementary}} implies
a domain's jump happens when its last binding constraint lifts. Identifying which
constraint is currently binding, from measurements rather than intuition, would make
that predictable. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-recovery}}'s retry acquires its precondition here: it needs detection, and
without it the mechanism is sampling.

{{part:16}}'s verification asymmetry is the premise that makes verifier-building
worth more than generation improvements, and this chapter is that argument applied to
domains.

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}} generalises: the affordances
are the interface, and they are what the tooling can change.

{{ch:ag-termination}}'s consequence gate is where irreversibility goes when staging
cannot remove it, and {{ch:as-long-running}} measured what placing it well is worth.

{{ch:as-state-machines}}'s tool audit — idempotent? reversible? — turns out to be the
input to this chapter's central decision as well as to that one's.

Ahead: {{ch:as-failures}} closes the part with the failure modes that appear only
when agents talk to each other, and several of this chapter's affordance gaps
reappear there as coordination problems.

## 17. Exercises

1. Measure your own domain's detection rate by seeding known-bad steps. Where does it
   put you in the first table?

2. Vary the false-alarm rate in the first listing. At what point does a verifier
   become net-harmful?

3. Implement staging in the second listing — actions on a copy, one promotion step —
   and measure the computer-use profile.

4. Find the progress threshold from {{eq:reversibility-dominates}} numerically and
   check it against the profiles that collapse.

5. Make the action space enter only through
   {{eq:action-choice-under-partial-observation}}'s faithful branch. How much of the
   action-space effect was really an interaction with fidelity?

6. Design an evaluation that would correctly value an affordance improvement while
   other constraints bind, and test it on the fidelity intervention.

## 18. Interview Questions

1. Why do coding agents work better than research agents when research tasks have
   higher per-step success rates?

2. What does retry require in order to be a correction rather than a resample?

3. Your computer-use agent got a large fidelity improvement and the eval did not
   move. What do you conclude?

4. How would you manufacture an undo for a domain that has none?

5. Where do you put the human gate once you have staging?

6. What would make you say a domain is not ready to automate?

## 19. Research Questions

1. How well do learned verifiers transfer across domains, and what detection rate do
   they achieve on unseen error distributions?

2. Can an agent determine reversibility empirically without incurring the
   irreversible costs while learning?

3. What fraction of measured agent capability gains over the last several years are
   attributable to environment instrumentation rather than to models?

4. Can the currently-binding constraint in a domain be identified from trajectory
   data?

5. Does the benchmark selection effect in {{sec:7-internal-mechanics}} materially
   distort the field's picture of where agents work?

## 20. Chapter Summary

The usual explanation for why coding agents work and computer-use agents do not is
difficulty. Across five domain profiles, per-step difficulty correlates $0.71$ with
task success and **verifier detection correlates $0.96$**
({{eq:verifier-sets-the-ceiling}}). Research, with the highest per-step success in
the table, finishes thirty-five points behind coding because it can check itself
$45\%$ of the time against coding's $97\%$.

The mechanism is that detection multiplies the retry term: four retries are worth
$+71.0$ points at high detection and $+14.1$ at low
({{eq:retry-needs-a-verifier}}). **A retry without a verifier is a fresh sample, not
a correction** — and every mechanism in {{part:17}} that works by noticing a failure
inherits the same factor.

Separating the hard end into action space, observation fidelity and reversibility,
**reversibility explained the largest share** at $83.3\%$ against fidelity's
$23.4\%$ ({{eq:reversibility-dominates}}) — which is not what the listing was
written to show.

The stronger result is that the three are complementary: fixing the action space
alone was worth $+1.0$, faithful observations alone $+0.3$, and observations plus
undo $+54.5$ ({{eq:domain-properties-are-complementary}}). **In a multiply-
constrained domain the marginal value of relieving one constraint is near zero**,
which is why incremental work on hard domains measures as nothing until it abruptly
measures as everything — and why evaluating such work by isolated ablation
systematically defunds the components of the eventual jump.

So specialising an agent for a domain is mostly not prompt engineering and mostly
not model choice. **It is building the affordances the domain does not supply**
({{eq:specialization-is-affordance-building}}) — a verifier, a faithful observation,
an undo — and building them together.

## 21. Further Reading

{{cite:zhou2024webarena}} for a browser environment built on structured state rather
than pixels, which is {{sec:7-internal-mechanics}}'s fidelity argument made
concrete, and {{cite:liu2024agentbench}} for the cross-domain comparison this
chapter reinterprets.

{{cite:shinn2023reflexion}} for a recovery mechanism whose value is gated by
detection in exactly the way {{eq:retry-needs-a-verifier}} describes, and
{{cite:yao2023react}} for the loop it wraps.

{{cite:schick2023toolformer}} for the tool-selection competence
{{sec:7-internal-mechanics}} concedes is genuinely a model property, and
{{cite:cemri2025mast}} for failure taxonomy across domains.

{{ch:as-state-machines}} for the tool audit that feeds this chapter's central
decision, and {{ch:rsn-test-time-compute}} for the verifier ceiling this chapter transposes
from reasoning to action.
