---
id: as-roles
number: 164
part: XVIII
tier: full
status: draft
requires: [handoff-is-a-bottleneck, decorrelation-is-the-variable,
           feedback-must-be-external]
provides: [roles-are-prompts, critic-must-beat-more-attempts,
           advise-not-gate-roles, capability-roles-not-label-roles,
           role-budget-cost, checker-critic-crossover]
citations: [cemri2025mast, du2023debate, huang2024selfcorrect,
            shinn2023reflexion, greshake2023indirect, liu2024agentbench]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say what a role label does and does
not supply; show that adding roles costs budget twice and buys nothing unless the
role-bearer is a different *system*; identify the critic as the one role with a
mechanism, and price it against the alternatives it competes with; configure a
critic to advise rather than gate, and say why; and state the condition under which
role separation genuinely earns its handoffs.

## 2. Why This Matters

Supervisor, worker, planner, critic. The taxonomy is universal, appears in every
framework, and is almost never accompanied by a mechanism. This chapter supplies
the measurement.

{{sec:9-practical-example}} runs a role-structured three-agent system against one
agent at equal cost and finds it *worse*: $18.2\%$ against $35.1\%$. And it finds
the same agent merely *switching between role prompts* also worse, at $22.8\%$ —
so the loss is not the handoffs alone.

**Roles cost budget twice.** Once because planning and criticism consume calls that
could have been attempts, and again because distributing them across agents costs
{{ch:as-multi-agent}}'s handoffs. Neither cost is offset by anything the label
supplies.

The exception is the row that makes the chapter. The same three agents, with a
critic whose errors are *decorrelated* from the worker's, reach $45.5\%$. What
produced that gain was not the label "critic"; it was a reviewer that fails in
different places — {{ch:rsn-self-consistency}}'s covariance term, arriving for the
fourth time in this book.

**A role is a prompt, and a prompt does not decorrelate anything.** A critic
implemented as the same model with different instructions shares nearly all the
worker's blind spots, and {{sec:9-practical-example}} measures that configuration
underperforming no critic at all.

The critic is nonetheless the one role with a genuine mechanism, and the second
listing prices it properly. Without a selector, extra attempts buy *nothing* — you
keep the last one — so a critic is not competing with "more tries", it is competing
with other selectors. Against that framing it does real work: $54.9\% \to 91.0\%$.
And it should be configured to **advise rather than gate**, because a gating critic
can veto correct work and an advising one cannot.

## 3. Prerequisites

You need {{ch:as-multi-agent}}'s handoff cost and its equal-cost comparison
discipline, because both apply directly here.

From {{ch:rsn-self-consistency}}, the correlated-critic result. This chapter is that
result applied to an organisational chart, and the finding is the same.

From {{ch:ag-recovery}}, the advise-versus-gate distinction: a weak signal wired to
a gate is worse than no signal. A critic is exactly such a signal.

From {{ch:rsn-test-time-compute}}, the coverage/selection decomposition, because the
second listing turns out to be that decomposition with the critic in the selector
slot.

## 4. Intuitive Explanation

Ask what a role is, mechanically.

In a framework, a role is a system prompt. The "planner" is a model instance told
to produce a plan; the "critic" is a model instance told to find problems. The
underlying computation is the same model, the same weights, the same failure modes.

That matters because {{ch:as-single-agent}} established what a second agent can
contribute: decorrelation, and nothing else. A prompt does not decorrelate. The
critic-prompted instance is wrong about the same things the worker-prompted
instance is wrong about, because they are the same system asked different questions.

So the default expectation should be that roles buy nothing — and
{{sec:9-practical-example}} finds they buy less than nothing, because they cost.

The first cost is budget. A fixed number of model calls is available; roles spend
some of them on planning and criticism rather than on attempts. If planning and
criticism do not improve the attempts by more than the attempts they displaced,
the trade loses.

The second cost is handoffs. Putting the roles in separate agents means the plan
crosses a boundary to reach the worker and the work crosses another to reach the
critic, and {{ch:as-multi-agent}} measured what that costs.

Now the exception, because there is one and it is important.

A critic *is* a mechanism, not just a job description. Its purpose is to select:
to look at a result and say whether it is good. And selection is the scarcest
component in this book — {{ch:rsn-test-time-compute}} showed coverage is worthless
without it, {{ch:rsn-supervision}} was about building one, {{ch:rsn-tool-assisted}}
was about what happens when you have a perfect one.

That reframes the comparison. A critic is not competing with "spend the budget on
more attempts", because without a selector more attempts do not help — you have no
way to know which attempt to keep, so you keep the last one and the extra calls
were wasted. A critic is competing with *other selectors*: an executable check, a
verifier model, a human.

Against that framing the critic does substantial work, and the second listing
measures it. But two things about how it is configured matter more than its
quality.

The first is decorrelation, again. A critic that shares the worker's blind spots
approves exactly the errors that matter.

The second is whether it *gates*. A critic that can send work back is also a critic
that can send back *correct* work, and {{ch:ag-recovery}}'s asymmetry applies: a
signal that only conditions a rework has a floor at doing nothing, and one that can
also block has no floor. Since a critic's false-positive rate is the error nobody
measures, advising is the safe configuration.

Finally, where roles genuinely earn their handoffs. If the role-bearer carries
different *capabilities* — a reader that cannot act, an actor that cannot read
private data — then it is a different system rather than a different prompt. It
decorrelates because its access is different, and it contains, because the blast
radius partitions. That is {{ch:ag-security}}'s capability split arriving as an
organisational chart, and it is the strongest case for role separation in this book.

## 5. Formal Explanation

Let a role-structured system spend a fraction $\beta$ of its budget $B$ on
role-specific work (planning, criticism) and $1 - \beta$ on attempts, with $h$
handoffs to distribute the roles. Against a single agent spending all of $B$:

$$S_{\text{roles}} = f\big((1-\beta)B\big) \cdot g(\text{critic}) \cdot (wr)^{h}, \qquad S_{\text{solo}} = f(B)$$ (eq:role-budget-cost)

Three factors, two of which are less than one. Roles win only if
$g(\text{critic})$ exceeds the product of the budget loss and the handoff loss —
and $g$ is where the decorrelation lives.

Model the critic by its true-positive rate $\tau$, false-positive rate $\phi$, and
the fraction $\rho$ of the worker's blind spots it shares:

$$g = 1 + \underbrace{(1-\rho)\,\tau\,\Pr[\text{wrong}]\,\pi_{\text{fix}}}_{\text{catches}} - \underbrace{\phi\,\Pr[\text{right}]\,\pi_{\text{break}}}_{\text{vetoes good work}}$$ (eq:roles-are-prompts)

The $(1-\rho)$ on the positive term and its absence on the negative one is the
whole asymmetry. **A critic's benefit is scaled by its decorrelation and its harm
is not**, so a highly correlated critic contributes almost none of the first term
and all of the second.

For a same-model, different-prompt critic $\rho \approx 1$, so $g < 1$ and
{{eq:role-budget-cost}} is a product of three factors below one.

Now the critic's proper comparison. Without a selector, $n$ attempts yield the
*last* attempt's success $p$, not the coverage $1 - (1-p)^n$:

$$S_{\text{no selector}} = p \quad\text{(independent of } n\text{)}, \qquad S_{\text{oracle}} = 1 - (1-p)^{n}$$ (eq:critic-must-beat-more-attempts)

The gap between those is what any selector is competing for, and a critic captures
a fraction of it. That reframes the question from "is a critic worth the calls" to
"which selector cashes the most coverage per call".

Finally the gating asymmetry, following {{ch:ag-recovery}}. Write $A$ for advising
(rework only when flagged, never block) and $G$ for gating (a flag sends work back
regardless):

$$S_A \ge S_{\text{no critic}} \text{ always}, \qquad S_G - S_A = -\phi\,\Pr[\text{right}]\,\pi_{\text{break}}$$ (eq:advise-not-gate-roles)

**Advising has a floor and gating does not**, and the gap is exactly the
false-positive term. {{sec:9-practical-example}} measures it widening from $0.0$ to
$3.4$ points as $\phi$ goes from $0\%$ to $40\%$.

And the capability case:

$$\rho_{\text{prompt}} \approx 1, \qquad \rho_{\text{capability}} \ll 1$$ (eq:capability-roles-not-label-roles)

because an agent with different access sees different evidence and therefore fails
in different places. That is a structural decorrelation rather than an instructed
one, and it is the only kind a role can supply.

## 6. Mathematical Foundation

Three extractions.

**The break-even correlation is computable.** Setting
{{eq:role-budget-cost}}'s product equal to one and solving for $\rho$ gives the
correlation below which roles pay. {{sec:9-practical-example}} finds it below
$0.6$ for its parameters, and a same-model critic sits far above that. **Measure
$\rho$ before adopting a critic**, by running both prompts on the same failing
cases and counting agreement on which are wrong.

**Gating's cost is unbounded in $\phi$ and advising's is zero.** From
{{eq:advise-not-gate-roles}}, $\partial S_G/\partial\phi < 0$ and
$\partial S_A/\partial\phi = 0$. Since $\phi$ drifts — a critic prompt that was
calibrated on one distribution objects more on another — the advising
configuration is also the one that degrades gracefully.

**The checker/critic crossover is a ceiling effect.** An executable check with
coverage $c$ caps at $c$; a critic given repeated attempts does not.
{{sec:9-practical-example}} measures the checker ahead at a budget of two
($77.1\%$ against $54.9\%$) and the critic ahead at sixteen ($99.8\%$ against
$96.4\%$). **An imperfect checker is better at small budgets and worse at large
ones**, which inverts the usual assumption that a real checker always dominates.

One caveat on the model. It treats planning as pure overhead — budget spent that
does not improve the attempts. {{ch:ag-planning}} found planning-as-structure
genuinely valuable, through checkpoints, so a planner role that produces *verifiable
boundaries* is contributing something this listing does not represent. The
distinction is the same one that chapter drew: a plan as prediction is overhead,
and a plan as structure is not.

## 7. Internal Mechanics

### 7.1 Where each role's value actually comes from

```mermaid {#fig:roles caption="The standard four roles, annotated with the mechanism each supplies. Only two of the four supply anything a prompt cannot."}
flowchart LR
    P[planner] -->|structure, if it emits checkpoints| W[worker]
    W -->|the work| C[critic]
    C -->|selection, if decorrelated| O[output]
    S[supervisor] -.->|routing, if capabilities differ| W
```

**Planner**: valuable if it emits checkpoints ({{ch:ag-planning}}), overhead if it
emits predictions.

**Worker**: the only role that is doing the task, and the one the other three take
budget from.

**Critic**: valuable in proportion to $(1-\rho)$ ({{eq:roles-are-prompts}}), and
harmful if configured to gate.

**Supervisor**: routing, which {{ch:ag-what-is-an-agent}} priced — valuable when the
routes have genuinely different capabilities, decorative otherwise.

### 7.2 Why a same-model critic shares blind spots

The critic reads the worker's output and judges it. That judgement is a forward
pass over the same weights that produced the output, so it is a function of the
same learned associations. Where the worker's error came from a systematic
misunderstanding, the critic shares the misunderstanding and reads the output as
correct.

This is {{ch:rsn-self-consistency}}'s result and {{cite:huang2024selfcorrect}}'s,
and putting the critic in a separate process does not change it. **What changes it
is a different model, different training lineage, or different access to
evidence** — the third being the cheapest and the most overlooked.

### 7.3 Making a critic decorrelate

Three routes, in increasing order of effect.

**Different evidence.** Give the critic access to something the worker did not
have — the test output, a retrieved document, a second data source. Its errors now
depend on different inputs.

**Different model.** {{eq:capability-roles-not-label-roles}}'s $\rho$ is minimised
across model families. A smaller, different-family critic often outperforms a
same-family one of any size for this reason.

**A real check.** {{ch:rsn-tool-assisted}}'s executable verifier has $\rho = 0$ by
construction on the property it checks, and it is the strongest critic available
where one exists.

Note that "a different prompt" appears nowhere on that list, and it is the standard
implementation.

### 7.4 Capability roles

The strongest role separation in this book is {{ch:ag-security}}'s reader/actor
split, and {{sec:9-practical-example}} shows it doing double duty: it decorrelates,
because the two agents see different evidence, *and* it halves the composed blast
radius.

That is the one arrangement where the handoff cost is clearly paid for, and it is
worth noting that the justification is security first and performance second. A
team that adopts it for containment gets the decorrelation as a side effect.

### 7.5 What the supervisor is for

In most implementations, nothing measurable. It routes, which is
{{ch:ag-what-is-an-agent}}'s router — valuable when there are genuinely different
things to route to, and an extra model call when there are not.

The exception is budget arbitration. {{ch:ag-termination}} found pooled budgets
beating per-task ones, and a supervisor holding the pool is a natural place to
implement that. **A supervisor that allocates is doing something; a supervisor that
merely delegates is a handoff.**

### 7.6 Why the role taxonomy persists anyway

If roles buy so little, it is worth asking why every framework ships them, and the
answer is that they are solving a different problem from the one this chapter
measures.

**They are a decomposition aid for the person writing the system.** "Planner,
worker, critic" is a way of thinking about what a system has to do, and thinking
about it that way produces better prompts than thinking about it as one
undifferentiated instruction. That benefit is real and it accrues at design time,
not at run time — which means it is fully available from one agent switching
prompts, with no handoffs.

**They are an observability structure.** A trace segmented by role is easier to
read than a trace that is one long conversation, and
{{ch:ag-what-is-an-agent}} established that an agent's debugging surface is mostly
its trace. That benefit is also available without separate agents, from labelling
the segments.

**And they map onto how teams divide ownership.** Different people own the planner
prompt and the critic prompt, and separate agents make that boundary enforceable.
This is a genuine organisational benefit with a measurable engineering cost, and it
should be recognised as the trade it is rather than justified on performance
grounds.

So the recommendation is not to abandon the vocabulary. It is to **keep the roles
as a design and observability structure and stop paying handoffs for them**, unless
the role-bearer carries different capabilities — at which point the handoff is
buying containment and the decorrelation arrives free.

That distinction — roles as a way of thinking versus roles as a deployment topology
— is what {{sec:9-practical-example}}'s middle two rows separate, and it is the
practical output of the chapter.

### 7.7 The one number to measure before adopting any role

Everything in this chapter reduces to $ho$, and it is measurable in an
afternoon on data you already have.

Take a hundred cases your worker got wrong. Run the critic prompt on each and
record whether it flags them. Take a hundred it got right and do the same. The
first gives you $	au$, the second gives you $\phi$, and the correlation follows
from comparing the critic's flags against a difficulty ranking of the cases: a
decorrelated critic catches hard cases at roughly the rate it catches easy ones,
and a correlated one catches almost none of the hard ones.

That last test is the discriminating one, and it is the one nobody runs. A critic
with a respectable aggregate $	au$ can have a $	au$ near zero on the cases where
the worker failed systematically — which are precisely the cases the critic was
added to catch. **Aggregate critic accuracy is not evidence about critic value**,
which is {{ch:ag-loop}}'s lesson about the stopping classifier and
{{ch:rsn-supervision}}'s about reward models, arriving once more.

If the conditional $	au$ on hard cases is low, the critic is decoration and the
three routes in {{sec:7-internal-mechanics}} are what to try next. If it is high,
the critic is a genuine second opinion and the handoff may be worth paying.

## 8. Implementation

Two listings. The first compares role-structured designs against one agent at equal
cost. The second prices the critic against the other selectors it competes with.

```python {tier=A name=roles-are-prompts}
"""Do role labels do anything?

Supervisor, worker, planner, critic. The taxonomy is universal and the mechanism
is rarely stated. This listing asks whether a role LABEL buys anything, by
comparing a role-structured multi-agent system against one agent running the same
prompts in sequence (eq:roles-are-prompts).

The comparison is deliberately unfair to the multi-agent version in exactly one
way, which is the way that matters: both spend the same number of model calls.
"""
import numpy as np

rng = np.random.default_rng(3079)

M = 40000
WORK = 8
BUDGET = 24
P_ORD = 0.93
P_SHARE_STICKY = 0.20
P_STICKY_FRESH = 0.32
P_HANDOFF = 0.884       # measured in ch:as-multi-agent
P_CRITIC_TP = 0.72      # a critic notices a real error
P_CRITIC_FP = 0.12      # a critic objects to correct work


def attempt(sticky, tries, diversity, m):
    ok = np.zeros(m, dtype=bool)
    fresh = np.ones(m, dtype=bool)
    for _ in range(tries):
        p = np.where(sticky, np.where(fresh, P_STICKY_FRESH, 0.05), P_ORD)
        ok |= (~ok) & (rng.random(m) < p)
        fresh = rng.random(m) < diversity
    return ok


def system(kind, m=M, work=WORK, budget=BUDGET, corr=0.9, diversity=0.0):
    """kind:
      solo       one agent, one prompt, all the budget
      solo_roles one agent switching prompts (plan, do, check) -- no handoff
      roles      separate planner/worker/critic agents, handoffs between them
      roles_dec  the same, with a critic whose errors are decorrelated (corr)
    """
    sticky = rng.random((m, work)) < P_SHARE_STICKY
    if kind == "solo":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= attempt(sticky[:, j], tries, diversity, m)
        return float(ok.mean())

    # Every role-bearing design spends part of its budget on planning and
    # criticism rather than on doing.
    do_budget = int(budget * 0.6)
    tries = max(1, do_budget // work)
    ok = np.ones(m, dtype=bool)
    for j in range(work):
        ok &= attempt(sticky[:, j], tries, diversity, m)

    # The critic reviews the result. Its errors are correlated with the worker's
    # unless it is genuinely a different system.
    shares = rng.random(m) < corr
    caught = (~ok) & (rng.random(m) < P_CRITIC_TP) & ~shares
    objected = ok & (rng.random(m) < P_CRITIC_FP)
    # A caught error gets one more pass; a false objection wastes one.
    ok |= caught & (rng.random(m) < P_ORD)
    ok &= ~(objected & (rng.random(m) < 0.35))     # some rework breaks things

    if kind in ("roles", "roles_dec"):
        # planner -> worker and worker -> critic are two handoffs.
        ok &= rng.random(m) < P_HANDOFF
        ok &= rng.random(m) < P_HANDOFF
    return float(ok.mean())


DESIGNS = [("one agent, one prompt", dict(kind="solo")),
           ("one agent, role prompts", dict(kind="solo_roles")),
           ("three agents, roles", dict(kind="roles")),
           ("three agents, decorrelated critic",
            dict(kind="roles_dec", corr=0.25))]

print(f"{M:,} tasks, {WORK} steps, {BUDGET} model calls for every design.")
print(f"Role-bearing designs spend 40% of the budget on planning and criticism.")
print(f"A critic catches a real error {P_CRITIC_TP:.0%} of the time and objects")
print(f"to correct work {P_CRITIC_FP:.0%} of the time. Each handoff costs")
print(f"{P_HANDOFF}.")
print()
print(f"{'design':>36}{'completed':>12}{'agents':>9}{'handoffs':>11}")
print("-" * 68)
res = {}
for name, kw in DESIGNS:
    v = system(**kw)
    res[name] = v
    n_ag = 1 if kw["kind"].startswith("solo") else 3
    n_ho = 0 if kw["kind"].startswith("solo") else 2
    print(f"{name:>36}{v:>12.1%}{n_ag:>9}{n_ho:>11}")

print()
print()
print("The critic is the role with a mechanism. Sweep how correlated its errors")
print("are with the worker's, holding everything else fixed.")
print()
print(f"{'critic correlation':>20}{'three agents':>15}{'one agent':>13}"
      f"{'best':>13}")
print("-" * 61)
solo = res["one agent, one prompt"]
cc = {}
for c in (1.0, 0.9, 0.6, 0.3, 0.0):
    a = system(kind="roles", corr=c)
    cc[c] = a
    best = "roles" if a > solo else "one agent"
    print(f"{c:>20.1f}{a:>15.1%}{solo:>13.1%}{best:>13}")

print()
print()
print("What the budget split costs. Role designs spend some of it on roles;")
print("sweep how much.")
print()
print(f"{'spent on doing':>16}{'three agents':>15}{'one agent, roles':>19}")
print("-" * 50)
sp = {}
for frac in (0.4, 0.6, 0.8, 1.0):
    a = system(kind="roles", corr=0.25, budget=int(BUDGET))
    # recompute with an explicit do-share by scaling the budget passed through
    tries_budget = int(BUDGET * frac)
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tr = max(1, tries_budget // WORK)
    ok = np.ones(M, dtype=bool)
    for j in range(WORK):
        ok &= attempt(sticky[:, j], tr, 0.0, M)
    shares = rng.random(M) < 0.25
    caught = (~ok) & (rng.random(M) < P_CRITIC_TP) & ~shares
    ok2 = ok | (caught & (rng.random(M) < P_ORD))
    with_ho = ok2 & (rng.random(M) < P_HANDOFF) & (rng.random(M) < P_HANDOFF)
    sp[frac] = (float(with_ho.mean()), float(ok2.mean()))
    print(f"{frac:>16.0%}{sp[frac][0]:>15.1%}{sp[frac][1]:>19.1%}")

print()
print()
print("And what roles buy when they carry different CAPABILITIES rather than")
print("different labels -- ch:ag-security's partition, as an architecture.")
print()
print(f"{'arrangement':>34}{'completed':>12}{'blast radius':>15}")
print("-" * 61)
cap = {}
for name, corr, radius in [("one agent, all capabilities", 0.9, 8),
                           ("three role agents, all capabilities", 0.9, 8),
                           ("three agents, split capabilities", 0.25, 4)]:
    v = system(kind="roles" if "three" in name else "solo", corr=corr)
    cap[name] = (v, radius)
    print(f"{name:>34}{v:>12.1%}{radius:>15}")

print(f"""
The first table is the comparison, and the middle two rows are the finding.

One agent with one prompt: {res['one agent, one prompt']:.1%}. The SAME agent
switching between a planning prompt, a working prompt and a checking prompt:
{res['one agent, role prompts']:.1%}. Three agents with those roles:
{res['three agents, roles']:.1%}.

**Adding roles made it worse, twice.** Once by spending {0.4:.0%} of the budget on
planning and criticism instead of on doing, and again by paying two handoffs to
distribute those roles across agents.

The last row is the exception and it is the whole chapter:
{res['three agents, decorrelated critic']:.1%}, from the same three agents with a
critic whose errors are decorrelated from the worker's.

So a role is not a mechanism. **A role is a prompt, and a prompt does not
decorrelate anything** (eq:roles-are-prompts). What produced the gain in the last
row was not the label "critic"; it was a reviewer that fails in different places,
which is ch:rsn-self-consistency's finding for the fourth time in this book.

The second table isolates that. Sweeping the critic's correlation with the worker
from {1.0} down to {0.0}, the three-agent design goes from {cc[1.0]:.1%} to
{cc[0.0]:.1%}, crossing the single agent's {solo:.1%} somewhere below {0.6}.

**A critic that shares most of the worker's blind spots is worse than no critic**,
because it costs budget and catches little -- and a critic implemented as the same
model with a different prompt shares almost everything. That is the standard
implementation.

The third table prices the budget split directly. Spending {0.4:.0%} of the budget
on doing gives {sp[0.4][1]:.1%} for the single agent and {sp[0.4][0]:.1%} with the
handoffs; spending {1.0:.0%} gives {sp[1.0][1]:.1%} and {sp[1.0][0]:.1%}.

Two costs, cleanly separated: **the budget the roles consume, and the handoffs
they require.** Neither is a property of having roles conceptually; both are
properties of implementing them as separate agents.

The fourth table is where roles genuinely earn their place, and it is not a
performance argument. Three agents with split CAPABILITIES reach
{cap['three agents, split capabilities'][0]:.1%} -- roughly the same as the
decorrelated-critic row, because splitting capabilities also decorrelates -- and
they halve the blast radius, from {cap['one agent, all capabilities'][1]} composed
risks to {cap['three agents, split capabilities'][1]}.

That is ch:ag-security's capability partition arriving as an architecture, and it
is the strongest case for role separation in this book: **a reader that cannot act
and an actor that cannot read private data are different SYSTEMS, not different
prompts** -- so they decorrelate, and they contain.

Which gives the rule. Roles are worth having when they carry different
capabilities, different credentials, or different model lineage. They are worth
nothing when they carry different instructions to the same model, and they cost
budget and handoffs either way.""")
```

The second listing isolates the critic.

```python {tier=A name=critic-must-beat-more-attempts}
"""The critic role, priced against the thing it is competing with.

The critic is the only role in the standard taxonomy with a mechanism rather than
a job description: it is supposed to catch errors the worker missed. That makes it
ch:rsn-self-consistency's critic problem and ch:ag-recovery's gating problem at
the same time, and both said the same thing -- the value is decorrelation, and a
weak signal that GATES is worse than no signal.

This listing puts a critic agent against the things it is competing with, and the
comparison is sharper than it first looks: more attempts are only worth having if
something can SELECT among them, and the critic is that something. So this is
ch:rsn-test-time-compute's coverage/selection decomposition with the critic in the
selector slot (eq:critic-must-beat-more-attempts).
"""
import numpy as np

rng = np.random.default_rng(3163)

M = 60000
P_TASK = 0.55           # a single attempt produces a correct result
BUDGET = 6              # model calls available per task


def outcome(design, budget=BUDGET, tp=0.72, fp=0.12, corr=0.85, check=0.0,
            m=M):
    """design:
      attempts   spend everything on attempts, keep the last
      critic     spend half on attempts, half on a critic that gates rework
      advise     the critic conditions a rework but never blocks a good result
      checker    an executable check of coverage `check` selects among attempts
    """
    if design == "attempts":
        # No selector: you keep the last attempt, so extra attempts buy nothing.
        return float(np.mean(rng.random(m) < P_TASK))
    if design == "coverage":
        # The ceiling: an oracle selector picking any correct attempt.
        ok = np.zeros(m, dtype=bool)
        for _ in range(budget):
            ok |= (~ok) & (rng.random(m) < P_TASK)
        return float(ok.mean())

    half = budget // 2
    ok = np.zeros(m, dtype=bool)
    for _ in range(half):
        ok |= (~ok) & (rng.random(m) < P_TASK)

    shares = rng.random(m) < corr
    flags_bad = (~ok) & (rng.random(m) < tp) & ~shares
    flags_good = ok & (rng.random(m) < fp)

    if design == "critic":
        # A flagged result is reworked; a good result flagged is also reworked
        # and may be broken by the rework.
        rework = flags_bad | flags_good
        redo = np.zeros(m, dtype=bool)
        for _ in range(budget - half):
            redo |= (~redo) & (rng.random(m) < P_TASK)
        out = np.where(rework, redo, ok)
        return float(out.mean())

    if design == "advise":
        # Only genuinely-flagged failures are reworked; good work is never
        # blocked, so a false positive costs nothing.
        redo = np.zeros(m, dtype=bool)
        for _ in range(budget - half):
            redo |= (~redo) & (rng.random(m) < P_TASK)
        return float((ok | (flags_bad & redo)).mean())

    if design == "checker":
        # An executable check of coverage `check` selects a correct attempt if
        # one exists among all `budget` attempts.
        got = np.zeros(m, dtype=bool)
        for _ in range(budget):
            got |= (~got) & (rng.random(m) < P_TASK)
        detected = got & (rng.random(m) < check)
        return float((detected | (got & (rng.random(m) < 0.3))).mean())
    raise ValueError(design)


print(f"{M:,} tasks. One attempt is correct {P_TASK:.0%} of the time;")
print(f"{BUDGET} model calls available. A critic catches a real failure 72% of")
print("the time, objects to good work 12%, and shares 85% of the worker's")
print("blind spots.")
print()
print(f"{'design':>34}{'completed':>12}{'vs no selector':>16}")
print("-" * 60)
base = outcome("attempts")
res = {}
for name, kw in [("more attempts, no selector", dict(design="attempts")),
                 ("half on a gating critic", dict(design="critic")),
                 ("half on an advising critic", dict(design="advise")),
                 ("executable check, 95% coverage",
                  dict(design="checker", check=0.95)),
                 ("oracle selector (the ceiling)", dict(design="coverage"))]:
    v = outcome(**kw)
    res[name] = v
    print(f"{name:>34}{v:>12.1%}{v - base:>+16.1%}")

print()
print()
print("The gating critic, swept over how correlated it is with the worker.")
print()
print(f"{'correlation':>13}{'gating':>10}{'advising':>11}{'no selector':>13}"
      f"{'best':>12}")
print("-" * 57)
cc = {}
for c in (0.95, 0.85, 0.6, 0.3, 0.0):
    g = outcome("critic", corr=c)
    a = outcome("advise", corr=c)
    cc[c] = (g, a)
    best = max([("gating", g), ("advising", a), ("none", base)],
               key=lambda x: x[1])[0]
    print(f"{c:>13.2f}{g:>10.1%}{a:>11.1%}{base:>13.1%}{best:>12}")

print()
print()
print("And swept over the critic's false-positive rate, which is what makes")
print("gating dangerous.")
print()
print(f"{'false positives':>17}{'gating':>10}{'advising':>11}{'gap':>9}")
print("-" * 47)
ff = {}
for f in (0.0, 0.05, 0.12, 0.25, 0.40):
    g = outcome("critic", fp=f, corr=0.3)
    a = outcome("advise", fp=f, corr=0.3)
    ff[f] = (g, a)
    print(f"{f:>17.0%}{g:>10.1%}{a:>11.1%}{a - g:>+9.1%}")

print()
print()
print("How the comparison moves with the budget. Without a selector the budget")
print("buys nothing; every other column is a way of cashing coverage in.")
print()
print(f"{'budget':>8}{'no selector':>13}{'gating':>10}{'advising':>11}"
      f"{'checker':>10}{'oracle':>10}")
print("-" * 50)
bd = {}
for b in (2, 4, 6, 10, 16):
    row = (outcome("attempts", budget=b), outcome("critic", budget=b),
           outcome("advise", budget=b),
           outcome("checker", budget=b, check=0.95),
           outcome("coverage", budget=b))
    bd[b] = row
    print(f"{b:>8}{row[0]:>13.1%}{row[1]:>10.1%}{row[2]:>11.1%}"
          f"{row[3]:>10.1%}{row[4]:>10.1%}")

print(f"""
The first table reframes what a critic is for, and the first row is the reframing.

**Without a selector, more attempts buy nothing.** You keep the last one, so the
budget column is flat at {res['more attempts, no selector']:.1%} no matter how
much you spend. That is ch:rsn-test-time-compute's coverage with nothing to cash
it in, and it is the situation a critic exists to fix.

An oracle selector reaches {res['oracle selector (the ceiling)']:.1%} -- the
ceiling. A gating critic reaches {res['half on a gating critic']:.1%}, an advising
one {res['half on an advising critic']:.1%}, and an executable check
{res['executable check, 95% coverage']:.1%}.

So the critic is doing real work: {res['half on a gating critic'] - res['more attempts, no selector']:+.1%}
over having no selector at all. **The critic role is a selector, and selectors are
the scarce component in this book** -- which is why this is the one role in the
standard taxonomy with a mechanism.

The second table sweeps its correlation with the worker, and the advising column
is consistently ahead. At {0.85} correlation -- which is what a critic implemented
as the same model with a different prompt actually is -- gating gives
{cc[0.85][0]:.1%} and advising {cc[0.85][1]:.1%}. At {0.0}, {cc[0.0][0]:.1%}
against {cc[0.0][1]:.1%}.

The third table says why. Sweeping the critic's false-positive rate, the gap
between gating and advising grows from {ff[0.0][1] - ff[0.0][0]:+.1%} at
{0:.0%} to {ff[0.4][1] - ff[0.4][0]:+.1%} at {0.4:.0%}.

**A gating critic can veto correct work; an advising one cannot.** That is
ch:ag-recovery's asymmetry exactly: a signal that only conditions a rework has a
floor at doing nothing, and one that also blocks has no floor
(eq:critic-must-beat-more-attempts). Since a critic's false-positive rate is the
error it is least often measured on, the advising configuration is the safe
default.

The last table is the one to size a critic against, and it contains a crossover
worth noticing.

At a budget of {2} the checker leads at {bd[2][3]:.1%} against the critic's
{bd[2][1]:.1%}. At {16} the critic leads, {bd[16][1]:.1%} against
{bd[16][3]:.1%} -- because the executable check's coverage is capped at
{0.95:.0%} and the critic gets repeated chances at the same task.

**An imperfect executable check is better at small budgets and worse at large
ones**, which is the opposite of the usual assumption that a real checker always
dominates a model-based one. The reason is that a check with fixed coverage has a
ceiling and a critic with repeated attempts does not.

The practical reading: use the executable check where it exists, keep a critic for
the part it does not cover, and configure the critic to advise rather than to
gate. And note that every column except the first requires the same thing -- a
selector -- which is ch:rsn-test-time-compute's conclusion arriving as an
organisational chart.""")
```

## 9. Practical Example

The first listing gives every design twenty-four model calls on an eight-step task.

```
                              design   completed   agents   handoffs
--------------------------------------------------------------------
               one agent, one prompt       35.1%        1          0
             one agent, role prompts       22.8%        1          0
                 three agents, roles       18.2%        3          2
   three agents, decorrelated critic       45.5%        3          2
```

**Adding roles made it worse, twice.** Once from the budget split ($35.1\% \to
22.8\%$, with no handoffs involved) and again from distributing them across agents
($22.8\% \to 18.2\%$).

The last row is the exception: the same three agents, with a critic whose errors are
decorrelated, reach $45.5\%$. **What produced that was not the label** — it was a
reviewer that fails in different places.

Sweeping that directly:

```
  critic correlation   three agents    one agent         best
-------------------------------------------------------------
                 1.0          13.7%        35.1%    one agent
                 0.6          31.0%        35.1%    one agent
                 0.3          44.0%        35.1%        roles
                 0.0          56.2%        35.1%        roles
```

The crossover is below $0.6$, and a critic implemented as the same model with a
different prompt sits far above it. **That configuration is worse than no critic**
({{eq:roles-are-prompts}}), and it is the standard one.

The budget split alone:

```
  spent on doing   three agents   one agent, roles
--------------------------------------------------
             40%          46.2%              59.2%
            100%          52.7%              67.5%
```

And where roles genuinely pay:

```
                       arrangement   completed   blast radius
-------------------------------------------------------------
       one agent, all capabilities       34.8%              8
three role agents, all capabilities       17.5%              8
  three agents, split capabilities       45.6%              4
```

Splitting *capabilities* rather than labels reaches $45.6\%$ **and** halves the
composed blast radius. That is {{ch:ag-security}}'s partition doing double duty
({{eq:capability-roles-not-label-roles}}).

The second listing isolates the critic and reframes what it competes with:

```
                            design   completed   vs no selector
---------------------------------------------------------------
        more attempts, no selector       54.9%           +0.0%
           half on a gating critic       91.0%          +36.1%
        half on an advising critic       91.6%          +36.7%
    executable check, 95% coverage       95.4%          +40.5%
     oracle selector (the ceiling)       99.2%          +44.3%
```

**Without a selector, more attempts buy nothing** — you keep the last one, so the
column is flat at $54.9\%$ at every budget. A critic is therefore not competing
with "more tries"; it is competing with other selectors, and against that framing
it does substantial work ({{eq:critic-must-beat-more-attempts}}).

Advising beats gating at every correlation, and the gap is the false-positive term:

```
  false positives    gating   advising      gap
-----------------------------------------------
               0%     95.0%      95.0%    +0.0%
              12%     94.2%      95.0%    +0.8%
              40%     91.7%      95.1%    +3.4%
```

**A gating critic can veto correct work; an advising one cannot**
({{eq:advise-not-gate-roles}}). Since $\phi$ is the error nobody measures, advising
is the safe default.

And a crossover worth noticing:

```
  budget  no selector    gating   advising   checker    oracle
--------------------------------------------------------------
       2        54.8%     54.9%      57.6%     77.1%     79.8%
       6        54.9%     91.0%      91.6%     95.4%     99.2%
      16        55.3%     99.8%      99.8%     96.4%    100.0%
```

The executable check leads at small budgets and the critic leads at large ones,
because a check with fixed coverage has a ceiling and a critic with repeated
attempts does not. **An imperfect checker is better cheaply and worse expensively**,
which inverts the usual assumption.

## 10. Production Considerations

Measure your critic's correlation with your worker before adopting it. Run both
prompts on the same failing cases and count agreement on which are wrong. Below
about $0.6$ it pays; above, it costs.

Decorrelate by evidence, model, or check — never by prompt. Giving the critic access
to something the worker did not have is the cheapest of the three.

Configure critics to advise, not gate. The floor property
({{eq:advise-not-gate-roles}}) matters more than the accuracy, and it degrades
gracefully when $\phi$ drifts.

Do not put roles in separate agents unless they carry different capabilities. The
same prompts run sequentially by one agent avoid the handoffs and lose nothing.

If you have a planner, make it emit checkpoints rather than predictions
({{ch:ag-planning}}). That is the difference between overhead and structure.

If you have a supervisor, give it the budget pool ({{ch:ag-termination}}). A
supervisor that allocates is doing something; one that delegates is a handoff.

And adopt the reader/actor split for containment; the decorrelation comes free.

## 11. Common Mistakes

**Implementing a critic as the same model with a different prompt.** $\rho \approx
1$, so it contributes none of the benefit and all of the harm.

**Letting the critic gate.** It can veto correct work, and the cost grows with a
false-positive rate nobody tracks.

**Putting roles in separate agents by default.** Two handoffs for prompts that could
have run sequentially.

**Comparing a role-structured system against a single agent at equal *agents*.**
The comparison must be at equal cost.

**Treating "more attempts" as the critic's competitor.** Without a selector, extra
attempts buy nothing — the competitor is another selector.

**Assuming an executable check always beats a model critic.** True at small budgets,
false at large ones.

**A planner that predicts rather than structures.** Pure budget overhead
({{ch:ag-planning}}).

## 12. Failure Modes

*Agreeable critic.* A same-model critic approves the worker's systematic errors and
objects to unfamiliar-looking correct work. Both directions are correlated with the
thing that matters.

*Rework churn.* A gating critic and a worker disagreeing repeatedly, consuming the
budget — {{ch:ag-loop}}'s non-productive cycle at the level of an organisation, and
one of {{cite:cemri2025mast}}'s inter-agent misalignment modes.

*Budget starvation of the worker.* Roles consuming enough of the budget that the
attempts that actually do the task become too few.

*Lost deduplication across the critic boundary.* The reworking agent repeats what
the original already tried, because the failure set did not cross
({{ch:as-multi-agent}}).

*Supervisor as a bottleneck.* Every decision routed through one agent, adding a
handoff per step and serialising work that could have run concurrently.

## 13. Alternatives

**One agent, sequential prompts.** All of the role structure, none of the handoffs.
{{sec:9-practical-example}} finds it ahead of the three-agent version.

**An executable check instead of a critic.** {{ch:rsn-tool-assisted}}: $\rho = 0$ on
the checked property, and the strongest selector available where one exists.

**A different-family critic.** The cheapest genuine decorrelation, and often a
smaller model suffices because the job is discrimination rather than generation.

**Capability partition.** {{ch:ag-security}}: decorrelates and contains, and is the
one role separation this book recommends without qualification.

**No critic, more coverage plus a checker.** Where a check exists, spend the
critic's budget on attempts and let the check select.

## 14. Evaluation

Measure $\rho$ between any two agents you plan to pair. It is the only variable in
{{eq:roles-are-prompts}} and it is estimable from paired traces.

Measure your critic's $\tau$ and $\phi$ separately, and report $\phi$ — it is the
one that decides advise-versus-gate and it is almost never published.

Compare role structures at equal *cost* and against one agent running the same
prompts sequentially, not against a bare loop.

Report the budget split: what fraction of calls went to attempts versus to roles.
{{eq:role-budget-cost}}'s first factor is usually invisible.

And run the checker/critic comparison at *your* budget, since
{{sec:9-practical-example}} shows the ordering reverses.

## 15. Advanced Concepts

**Measuring $\rho$ cheaply.** Two systems' error correlation is estimable from
agreement on a labelled set, and the estimate transfers across tasks better than
either system's accuracy does. This should be a standard number and is not.
{{maturity:EMERGING}}.

**Evidence-based decorrelation.** Giving a critic access the worker lacked is the
cheapest route to low $\rho$, and it is under-explored relative to the
different-model route. The design question is which evidence decorrelates most per
unit of cost.

**Adaptive advise/gate.** {{eq:advise-not-gate-roles}} says gating is safe only when
$\phi$ is small, and $\phi$ is measurable online. A critic that gates when confident
and advises otherwise is a small policy layer nobody ships.

**Roles as capability contracts.** If a role were a declared capability set rather
than a prompt, {{eq:capability-roles-not-label-roles}} would be enforceable and
{{ch:ag-security}}'s containment would follow from the architecture. That is a
framework design question and {{maturity:RESEARCH FRONTIER}} in practice.

## 16. Connection to Previous Chapters

{{ch:rsn-self-consistency}}'s correlated critic appears here for the fourth time,
and this chapter is the version where it applies to an organisational chart rather
than to a prompt.

{{ch:ag-recovery}}'s advise-versus-gate asymmetry transfers directly, and
{{eq:advise-not-gate-roles}} is that result restated for a critic agent.

{{ch:as-multi-agent}}'s handoff cost is the second of the two costs roles impose,
and its equal-cost discipline is what makes the comparison honest.

{{ch:ag-security}}'s capability partition is the one role separation with an
unqualified justification, and {{sec:9-practical-example}} shows it doing double
duty.

Ahead: {{ch:as-graph}} makes the role structure explicit and asks what that buys;
{{ch:as-failures}} returns to {{cite:cemri2025mast}}'s inter-agent misalignment
category, whose mechanisms are this chapter's failure modes.

## 17. Exercises

1. Solve {{eq:role-budget-cost}} for the break-even $\rho$ and check it against the
   measured crossover below $0.6$.

2. Add a planner that emits checkpoints rather than predictions, using
   {{ch:ag-planning}}'s mechanism. How much of the role overhead does that recover?

3. Sweep the critic's true-positive rate at fixed $\rho = 0.9$. Can a *better*
   same-model critic ever beat no critic?

4. Implement the adaptive advise/gate policy from
   {{sec:15-advanced-concepts}} and measure it against both fixed policies.

5. Find the budget at which the checker and critic cross for a checker with $80\%$
   coverage rather than $95\%$.

6. Estimate $\rho$ between your own worker and critic prompts on fifty failing
   cases. Which side of the crossover are you on?

## 18. Interview Questions

1. What does a role label supply that a prompt does not?

2. Your critic is the same model with a different system prompt. What is it worth?

3. Should a critic be able to send work back? Why or why not?

4. A critic costs half your budget. What is it competing with?

5. When does putting roles in separate agents pay for the handoffs?

6. When is an executable check worse than a model critic?

## 19. Research Questions

1. What is $\rho$ empirically between same-model different-prompt agents, and how
   much do the three decorrelation routes move it?

2. Which evidence decorrelates a critic most per unit of cost?

3. Can $\phi$ be estimated online well enough to drive an adaptive advise/gate
   policy?

4. Does a checkpoint-emitting planner recover enough of the role overhead to make
   planner separation worthwhile?

5. Could roles be expressed as enforced capability contracts, and would that make
   {{ch:ag-security}}'s containment a property of the architecture?

## 20. Chapter Summary

A role is a prompt, and a prompt does not decorrelate anything
({{eq:roles-are-prompts}}). {{sec:9-practical-example}} measures a role-structured
three-agent system at $18.2\%$ against one agent's $35.1\%$ at equal cost, and the
same agent merely switching role prompts at $22.8\%$ — so **roles cost budget
twice**, once for the calls they consume and once for the handoffs they require
({{eq:role-budget-cost}}).

The exception is decorrelation: the same three agents with a critic whose errors are
independent reach $45.5\%$. The crossover in correlation is below $0.6$, and a
same-model different-prompt critic sits far above it — **worse than no critic at
all**.

The critic is nonetheless the one role with a mechanism, and its proper comparison
is not "more attempts" but "other selectors", because without a selector extra
attempts buy nothing ({{eq:critic-must-beat-more-attempts}}). Against that framing
it takes $54.9\%$ to $91.0\%$. Configure it to **advise rather than gate** —
advising has a floor and gating does not, and the gap grows with a false-positive
rate nobody measures ({{eq:advise-not-gate-roles}}).

An imperfect executable check beats a critic at small budgets and loses at large
ones, because a fixed-coverage check has a ceiling and repeated criticism does not.

And role separation earns its handoffs in exactly one case: when the roles carry
different **capabilities** rather than different labels
({{eq:capability-roles-not-label-roles}}). A reader that cannot act and an actor
that cannot read private data are different systems — they decorrelate *and* they
halve the composed blast radius, which is {{ch:ag-security}}'s partition doing
double duty.

## 21. Further Reading

{{cite:cemri2025mast}}'s inter-agent misalignment category is this chapter's failure
modes catalogued from real traces, and worth reading against
{{sec:12-failure-modes}}.

{{cite:huang2024selfcorrect}} for why a same-model critic cannot decorrelate, and
{{ch:rsn-self-consistency}} for the measurement.

{{cite:du2023debate}} for the one multi-agent design whose critique mechanism this
chapter's model does not capture — and which is where a debate's real value sits.

{{ch:ag-security}} for the capability partition, which is the role separation this
chapter ends up recommending on grounds that are not about performance at all.
