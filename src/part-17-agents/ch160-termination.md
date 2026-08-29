---
id: ag-termination
number: 160
part: XVII
tier: full
status: draft
requires: [stopping-is-a-classifier, visible-versus-silent-failure,
           marginal-value-of-samples]
provides: [three-terminations, habituation, gate-on-consequence,
           budget-is-a-population-decision, escalate-not-confirm,
           per-task-cap-wastes-budget]
citations: [zhou2024webarena, liu2024agentbench, greshake2023indirect,
            snell2024testtime, brown2024monkeys, huang2024selfcorrect,
            shinn2023reflexion]
---

## 1. Learning Objectives

By the end of this chapter you will be able to separate the three reasons a run
ends and say who should own each decision; explain why a confirmation gate applied
to everything is close to applying it to nothing, and compute the load at which
that happens; choose a gating criterion from the harm-avoided-per-hour ratio rather
than from a category policy; explain why a per-task step budget converts a budget
increase into nothing; and say why escalation is a better first investment than
confirmation.

## 2. Why This Matters

A run ends for one of three reasons — the task is done, the budget ran out, or a
person is needed — and systems conflate them. {{ch:ag-loop}} handled the first and
found it a classifier the agent should not own, because a false stop is a confident
wrong answer. This chapter is about the other two, and both come out against
standard practice.

**Confirming everything is close to confirming nothing.** A reviewer's catch rate
is not a constant; it falls with load. {{sec:9-practical-example}} measures three
reviewers asked to approve three thousand actions a day, and their effective catch
rate collapses from $92\%$ to $2.2\%$. Seventy-five human hours a day to avoid about
$3\%$ of the harm. They are not reviewing — they are clicking, and the organisation
now believes those actions were reviewed.

Hold the review budget fixed at $2\%$ and change only *what* gets looked at, and the
harm avoided per human hour ranges from $0.07$ for gating everything to $34.52$ for
gating by blast radius. **A factor of about five hundred, from the selection
criterion alone**, at identical cost per review.

The budget half produces a similarly unwelcome result. The best policy measured
**predicts nothing**: one shared pool, round-robin over live tasks, a task leaves
the pool the moment it finishes. It beat a difficulty-aware policy that spent a
pilot to estimate task difficulty, and it beat uniform per-task budgets by $9$
points. A per-task cap, meanwhile, flatlines — raising the total budget from $8$ to
$50$ steps per task moved it by $7.7$ points while pooling gained $48.0$.

And the third decision, escalation, is the one that raises the ceiling rather than
the efficiency. It is triggered by a *visible* failure the system produces for free,
it spends attention only on runs that already failed, and unlike a confirmation gate
it does not habituate, because its volume is bounded by the failure rate.

## 3. Prerequisites

You need {{ch:ag-loop}}'s three-way outcome split — done, exhausted, stopped early —
and specifically its distinction between visible and silent failure, because
escalation is only available on the visible kind.

From {{ch:rsn-test-time-compute}}, the allocation result: a fixed per-task budget is
the uniform allocation, pilots are expensive relative to what they recover, and
adaptive early stopping beat an oracle. All three reappear here.

From {{ch:ag-planning}}, the budget cliff, and from {{ch:ag-recovery}}, the
observation that the retry decision should belong to a budget policy rather than to
the agent. This chapter is where that policy gets written.

## 4. Intuitive Explanation

Start with the confirmation gate, because the reasoning is where intuition fails
most cleanly.

The argument for it is unimpeachable. The agent might do something wrong; a person
checking would catch it; therefore have a person check. Every step of that is true
for a single action.

What it omits is that the person is a shared resource with finite attention. Ask
someone to approve four things a day and they read them carefully. Ask the same
person to approve four hundred and they develop a rhythm: glance, approve, glance,
approve. Not through carelessness — through the ordinary adaptation that lets people
do repetitive work at all.

So the catch rate is a function of the load, and a gate that routes more work to the
same reviewers buys less per item than it did before. Past some volume it buys
nearly nothing, and you are paying full price for it.

The consequence is worse than wasted money. A gate that everyone knows is a rubber
stamp is a *documented* review. The action was approved. The audit log says so. The
organisation has converted an unreviewed action into an unreviewed action with a
signature on it.

Which points at the fix, and it is not "review harder". It is to spend the fixed
attention on the actions where it matters. Two candidate criteria present
themselves and they are not equal.

You could gate on the agent's confidence — review what the agent is least sure
about. This is better than random, and it inherits the correlation problem from
{{ch:rsn-self-consistency}}: the agent is confidently wrong exactly where its
judgement has failed, so the actions most in need of review are partly filtered out
of the queue.

Or you could gate on *consequence*: what happens if this is wrong. An agent
misfiling a note and an agent deleting a customer record are the same action from
the confidence model's point of view and completely different from yours. And
consequence, unlike confidence, is a property of the action type that you can
determine without asking the model anything.

The sharpest version of the criterion is reversibility. An action you can undo does
not need a person before it; at most it needs one after. An action you cannot undo
needs one before or it needs not to be available to the agent at all.

Now the budget. The default is a per-task step limit — this run may take at most $n$
steps. That is a reasonable-looking rule with a specific flaw: **unused allowance
does not go anywhere.** A task that finishes in three steps leaves its remaining
fifteen unspent, and a task that needs twenty-five is cut off at eighteen, while the
fifteen sit next to it doing nothing.

Pool the budget instead — a total for the batch, spent round-robin over whatever is
still running — and the unspent allowance flows automatically to the tasks that are
still working. No prediction, no routing, no difficulty model. Just: finished tasks
stop consuming.

That turns out to beat trying to be clever. {{sec:9-practical-example}} measures a
policy that spends a pilot to estimate each task's difficulty and allocates
accordingly, and it does *worse* than uniform, because the pilot costs budget and
three observations is not enough to place the rest well.

The last idea is the difference between confirmation and escalation, and it is
mostly about *when* the human is asked.

A confirmation gate asks before the action, on a prediction that something might go
wrong. Its volume is the volume of actions, which is large, so it habituates.

An escalation asks after a failure the system has already detected — the budget ran
out. Its volume is the failure rate, which is small, so it does not habituate. And
the human is being handed something the system has explicitly said it could not do,
which is a much better use of a person than asking them to confirm a thousand things
that were fine.

## 5. Formal Explanation

Separate the three terminations:

$$\text{end} \in \{\underbrace{\text{done}}_{\text{a classifier}},\ \underbrace{\text{exhausted}}_{\text{a budget policy}},\ \underbrace{\text{escalated}}_{\text{a routing policy}}\}$$ (eq:three-terminations)

{{ch:ag-loop}} owns the first. The second and third are the subject here, and they
have different owners: the budget policy is a property of the batch, and the routing
policy is a property of the action.

**Habituation.** Let $R$ reviewers share $n$ gated items in a period. Model the
catch rate as decaying in per-reviewer load:

$$c(n) = \frac{c_0}{1 + \dfrac{n/R}{\kappa}}$$ (eq:habituation)

with $\kappa$ the load at which attention halves. The *expected catches* are
$n \cdot c(n) \cdot \Pr[\text{bad}]$, and note what happens as $n$ grows: the
product $n\,c(n)$ approaches the constant $c_0 R \kappa$. **Total catches saturate.**
Adding items to the queue past $R\kappa$ buys essentially nothing, no matter how
many you add.

That is the formal statement of "confirming everything is close to confirming
nothing", and it says the ceiling is set by reviewer capacity rather than by policy.

Since the number of catches is capped, the only remaining variable is *which* items
occupy the slots. With harm $h_i$ per bad action, the objective is:

$$\max_{\mathcal{G}} \sum_{i \in \mathcal{G}} h_i \Pr[\text{bad}_i]\, c(|\mathcal{G}|) \quad\text{s.t.}\quad |\mathcal{G}| \le R\kappa$$ (eq:gate-on-consequence)

which is a knapsack over $h_i \Pr[\text{bad}_i]$. Confidence estimates the second
factor badly and consequence measures the first exactly, so **gating on consequence
dominates gating on confidence whenever harm is more dispersed than error
probability** — which it is, since harm distributions have tails and error rates do
not.

**Budgets.** With $M$ tasks and a total $B$, a per-task cap $b = B/M$ is the uniform
allocation. Its inefficiency is the unspent remainder:

$$\text{waste} = \sum_i \max(0,\ b - s_i), \qquad s_i = \text{steps task } i \text{ needed}$$ (eq:per-task-cap-wastes-budget)

which is strictly positive whenever any task finishes early, and is *unrecoverable*
under a per-task rule. Pooling makes it recoverable:

$$B_{\text{available to unfinished}} = B - \sum_{i \text{ finished}} s_i$$ (eq:budget-is-a-population-decision)

with no estimate of $s_i$ required in advance — the policy learns each $s_i$ by
watching the task finish. That is {{ch:rsn-test-time-compute}}'s early stopping, and
it beats a difficulty-predicting policy for the same reason it did there: prediction
costs budget and observation is free.

**Escalation.** Where a confirmation gate processes $n_{\text{gate}} = O(\text{all
actions})$, escalation processes $n_{\text{esc}} = O(\text{failures})$:

$$n_{\text{esc}} = M \cdot \Pr[\text{exhausted}] \;\ll\; n_{\text{gate}}$$ (eq:escalate-not-confirm)

Substituting into {{eq:habituation}}, escalation operates in the regime where
$c(n) \approx c_0$ and confirmation operates where $c(n) \to 0$. **The two
mechanisms use the same people and land on opposite sides of the habituation
curve.**

## 6. Mathematical Foundation

Three consequences worth extracting.

**The gate has an interior optimum and it is small.** Total harm avoided is
$\sum_{i \in \mathcal{G}} h_i \Pr[\text{bad}_i] c(|\mathcal{G}|)$, and adding the
$(n{+}1)$th item contributes $h_{n+1}\Pr[\text{bad}]c(n{+}1)$ while reducing every
existing item's contribution through $c$. The marginal item is by construction less
consequential than those already in the queue, so the derivative turns negative
before the queue is full. {{sec:9-practical-example}} measures harm avoided peaking
at a $5\%$ review budget and *falling* at $15\%$ — more review made things worse.

**Reviewer count and $\kappa$ enter multiplicatively.** From
{{eq:habituation}}, capacity is $R\kappa$: doubling the reviewers and doubling their
tolerance for repetitive review are equivalent. That means "hire more reviewers" and
"make each review shorter and more focused" are the same intervention
quantitatively, which is worth knowing when only one is affordable.

**A per-task cap decouples outcome from budget.** From
{{eq:per-task-cap-wastes-budget}}, raising $B$ under a fixed cap $b$ raises nothing
until $b$ is also raised. {{sec:9-practical-example}} measures a cap of $10$ holding
at $40.8$–$41.2\%$ completion across a sixfold increase in total budget, while
pooling rose from $35.4\%$ to $83.4\%$. **A per-task cap converts a budget increase
into nothing**, and the symptom — spending more and observing no improvement — is
usually misdiagnosed as a model problem.

One boundary on the pooling result. {{sec:9-practical-example}}'s hard band consumed
$30.2$ steps per task under pooling to complete $0.9\%$ of them, the most of any
band for almost nothing. Pooling reallocates toward tasks that are still working,
and a hopeless task is still working. **Pooling needs a per-task cap on top of it** —
not as the budget, but as a floor on how much any single task may absorb. The two
mechanisms do different jobs and the mistake is using the cap as the budget.

## 7. Internal Mechanics

### 7.1 The three decisions and their owners

```mermaid {#fig:three-terminations caption="Three ways a run ends. The first is a classifier the agent should not own; the second belongs to the batch; the third to the action's consequence."}
flowchart TD
    R[run] --> D{task complete?}
    D -- yes, verified --> DONE[done]
    D -- no --> B{budget left?}
    B -- yes --> R
    B -- no --> E{consequential?}
    E -- yes --> H[escalate to a person]
    E -- no --> F[report failure]
```

The important structural point is that escalation hangs off *budget exhaustion*,
which the system knows for certain, rather than off a prediction that something may
go wrong.

### 7.2 Why confidence is the wrong gating signal

It is not that confidence is uninformative — {{sec:9-practical-example}} measures it
beating random by a wide margin. It is that it fails in the correlated direction.

{{ch:rsn-self-consistency}}'s result says a model's confidence is least calibrated
exactly where its answer is wrong for a systematic reason. So confidence-based gating
filters out the errors the model *knows* are risky and passes through the ones it is
sure about, which are the ones that got that way by a mechanism the model cannot
see.

Consequence has no such property. The blast radius of `DELETE FROM` does not depend
on how the agent feels about it.

### 7.3 Reversibility as the practical criterion

Harm is hard to quantify and reversibility is usually obvious, which makes it the
operational proxy for {{eq:gate-on-consequence}}. Three tiers:

**Reversible and cheap** — a draft, a search, a read. No gate. Log it.

**Reversible and expensive** — a message sent, a file overwritten with a backup. No
gate before; a fast undo path and an alert after.

**Irreversible** — a payment, a deletion without backup, an external notification.
Gate, or do not give the agent the capability.

The third tier is where {{eq:gate-on-consequence}}'s knapsack budget should be
spent, and it is usually small enough to fit inside $R\kappa$, which is the whole
design.

### 7.4 Budgets in the right unit

{{ch:ag-recovery}} argued budgets should be in steps rather than attempts, so a
resuming policy is rewarded. This chapter adds a second dimension: budgets should be
*pooled* rather than per-task, so unspent allowance is recoverable.

Combined, the budget is a step pool over a batch, with a per-task cap acting as a
circuit breaker rather than as the allocation. Three distinct parameters —
pool size, per-task cap, per-step timeout — that are frequently collapsed into one
number.

### 7.5 What escalation costs and what it needs

An escalation needs three things the system must produce: the state at the point of
failure, what was attempted, and what the agent believes went wrong.
{{ch:ag-memory}}'s scratchpad and {{ch:ag-planning}}'s checkpoints supply the first
two directly, which is another return on building them.

The third is the agent's own diagnosis, and {{ch:ag-recovery}}'s rule applies here
too: it should *advise* the human, never *gate* the escalation. An agent that
suppresses an escalation because it thinks it understands the failure is reproducing
the false-stop failure at a higher level.

### 7.6 Escalation as an attack surface

An escalation puts agent-controlled text in front of a human who is deciding whether
to authorise something. {{cite:greshake2023indirect}}'s injection vector applies: if
any part of that text can be influenced by content the agent read, the attacker is
writing the justification the reviewer sees.

The mitigation is structural rather than a policy — show the reviewer the *action* in
a form the agent did not author, and treat the agent's narrative as untrusted
commentary beside it.

## 8. Implementation

Two listings. The first prices confirmation gating against reviewer habituation and
compares gating criteria at a fixed attention budget. The second compares budget
policies across a population of tasks and prices escalation on top of the best one.

```python {tier=A name=habituation}
"""Confirming everything is close to confirming nothing.

A human-in-the-loop gate is usually specified as a policy about WHICH actions need
approval, and evaluated on the assumption that an approved action was actually
reviewed. That assumption is the whole problem.

A reviewer's attention is finite and degrades with load. A person asked to approve
four actions a day reads them; the same person asked to approve four hundred clicks
through. So the catch rate is not a constant -- it is a function of how much you ask
them to look at (eq:habituation), and a gate that routes more work to the same
people buys less per item than it did before.

This listing prices five gating policies on the two things that matter: the harm
that reached production, and the human hours consumed.
"""
import numpy as np

rng = np.random.default_rng(2423)

DAYS = 400
N = 3000                # actions proposed per day
P_BAD = 0.06            # share of proposed actions that are wrong
BASE_CATCH = 0.92       # catch rate for an attentive reviewer
REVIEWERS = 3
HAB_K = 25              # reviews PER REVIEWER PER DAY at which attention halves
MINUTES = 1.5


def catch_rate(n_reviews):
    """Attention decays with each reviewer's daily load, not with the total."""
    return BASE_CATCH / (1.0 + (n_reviews / REVIEWERS) / HAB_K)


def make_day():
    bad = rng.random(N) < P_BAD
    # Blast radius: most actions are trivial, a few are not.
    harm = np.where(rng.random(N) < 0.08, rng.gamma(3.0, 6.0, N),
                    rng.gamma(1.0, 0.4, N))
    # The agent's own confidence: informative, and correlated with its errors.
    conf = np.clip(rng.normal(0.72 - 0.30 * bad, 0.18), 0, 1)
    return bad, harm, conf


def mask_for(kind, frac, harm, conf):
    if kind == "none":
        return np.zeros(N, dtype=bool)
    if kind == "all":
        return np.ones(N, dtype=bool)
    if kind == "random":
        return rng.random(N) < frac
    if kind == "harm":
        return harm >= np.quantile(harm, 1 - frac)
    if kind == "lowconf":
        return conf <= np.quantile(conf, frac)
    raise ValueError(kind)


def run(kind, frac=0.02, days=DAYS, hab=None):
    hk = HAB_K if hab is None else hab
    shipped_n = shipped_h = revs = 0.0
    catch_acc = 0.0
    for _ in range(days):
        bad, harm, conf = make_day()
        m = mask_for(kind, frac, harm, conf)
        n_rev = int(m.sum())
        c = BASE_CATCH / (1.0 + (n_rev / REVIEWERS) / hk)
        caught = m & bad & (rng.random(N) < c)
        ship = bad & ~caught
        shipped_n += ship.sum()
        shipped_h += harm[ship].sum()
        revs += n_rev
        catch_acc += c
    d = days
    return (shipped_n / (d * N), shipped_h / d, revs / d,
            revs / d * MINUTES / 60.0, catch_acc / d)


print(f"{N:,} proposed actions a day, {P_BAD:.0%} of them wrong, over {DAYS} days.")
print(f"{REVIEWERS} reviewers; an attentive one catches {BASE_CATCH:.0%}, and")
print(f"attention halves at {HAB_K} reviews per person per day.")
print(f"Each review costs {MINUTES} human minutes.")
print()
print(f"{'gate rate':>11}{'reviews/day':>13}{'catch rate':>12}"
      f"{'harm/day':>11}{'hours/day':>12}")
print("-" * 59)
sweep = {}
for g in (0.0, 0.005, 0.02, 0.05, 0.20, 1.0):
    r = run("random" if g not in (0.0, 1.0) else ("none" if g == 0 else "all"),
            frac=g)
    sweep[g] = r
    print(f"{g:>11.1%}{r[2]:>13.0f}{r[4]:>12.1%}{r[1]:>11.1f}{r[3]:>12.1f}")

print()
print()
print("Five gating policies at a fixed 2% review budget, so the comparison is")
print("about WHAT you look at rather than how much.")
print()
BF = 0.02
print(f"{'policy':>26}{'reviews/day':>13}{'catch':>8}{'harm/day':>11}"
      f"{'vs no gate':>12}")
print("-" * 70)
pol = {}
base = run("none")[1]
for name, kind in [("no gate", "none"), ("gate everything", "all"),
                   ("random 2%", "random"),
                   ("lowest-confidence 2%", "lowconf"),
                   ("highest-blast-radius 2%", "harm")]:
    r = run(kind, frac=BF)
    pol[name] = r
    print(f"{name:>26}{r[2]:>13.0f}{r[4]:>8.1%}{r[1]:>11.1f}"
          f"{r[1] - base:>+12.1f}")

print()
print()
print("Harm avoided per human hour -- the only ratio that decides the policy.")
print()
print(f"{'policy':>26}{'hours/day':>12}{'harm avoided':>15}{'per hour':>12}")
print("-" * 65)
for name, r in pol.items():
    saved = base - r[1]
    per = saved / r[3] if r[3] > 0 else float("nan")
    print(f"{name:>26}{r[3]:>12.1f}{saved:>15.1f}"
          f"{('--' if r[3] == 0 else format(per, '.2f')):>12}")

print()
print()
print("How much does habituation matter? Sweep the load at which attention")
print("halves, for gate-everything against gating the top 2% by blast radius.")
print()
print(f"{'halving load':>14}{'gate all: catch':>18}{'gate all: harm':>17}"
      f"{'top 2%: harm':>15}")
print("-" * 64)
hab = {}
for k in (10, 25, 100, 500, 10 ** 7):
    a = run("all", frac=BF, days=120, hab=k)
    b = run("harm", frac=BF, days=120, hab=k)
    hab[k] = (a, b)
    label = "none" if k > 10 ** 6 else f"{k:,}"
    print(f"{label:>14}{a[4]:>18.1%}{a[1]:>17.1f}{b[1]:>15.1f}")

print()
print()
print("And how much review is worth buying, keyed to blast radius.")
print()
print(f"{'review budget':>15}{'hours/day':>12}{'harm/day':>11}"
      f"{'harm avoided':>15}{'per hour':>11}")
print("-" * 64)
bud = {}
for f in (0.002, 0.005, 0.01, 0.02, 0.05, 0.15):
    r = run("harm", frac=f, days=200)
    bud[f] = r
    saved = base - r[1]
    print(f"{f:>15.1%}{r[3]:>12.1f}{r[1]:>11.1f}{saved:>15.1f}"
          f"{saved / max(r[3], 1e-9):>11.2f}")

print(f"""
The first table is the case against the reflex, and the two end rows are the whole
argument.

Gating nothing ships {sweep[0.0][1]:.0f} units of harm a day and costs zero human
hours. Gating EVERYTHING ships {sweep[1.0][1]:.0f} and costs
{sweep[1.0][3]:.0f} human hours a day.

Seventy-five hours -- more than nine person-days -- to avoid about
{(sweep[0.0][1] - sweep[1.0][1]) / sweep[0.0][1]:.0%} of the harm. The catch-rate
column says why: it falls from {sweep[0.0][4]:.0%} for an attentive reviewer to
{sweep[1.0][4]:.1%} when three people are asked to approve three thousand actions a
day. **They are not reviewing. They are clicking** (eq:habituation).

That is the failure mode a policy of "require human approval for agent actions"
produces at scale, and it is worse than it looks on paper, because the
organisation now believes those actions were reviewed.

The second table holds the review budget fixed at {BF:.0%} and changes only WHAT
gets looked at, which is where the leverage is.

Random {BF:.0%} avoids {base - pol['random 2%'][1]:.1f} units of harm a day.
Gating the lowest-confidence {BF:.0%} avoids
{base - pol['lowest-confidence 2%'][1]:.1f}. Gating the highest-blast-radius
{BF:.0%} avoids {base - pol['highest-blast-radius 2%'][1]:.1f}.

Same number of reviews, same reviewers, same catch rate --
{pol['highest-blast-radius 2%'][1] / pol['random 2%'][1] - 1:+.0%} difference in
harm shipped, from the selection criterion alone.

The third table is the ratio that should decide the policy, and the spread is the
result of this listing. Harm avoided per human hour:
{base - pol['gate everything'][1] and (base - pol['gate everything'][1]) / pol['gate everything'][3]:.2f} for gating everything,
{(base - pol['random 2%'][1]) / pol['random 2%'][3]:.2f} for random sampling,
{(base - pol['highest-blast-radius 2%'][1]) / pol['highest-blast-radius 2%'][3]:.2f}
for gating by blast radius.

**A factor of about five hundred between the worst policy and the best**, at
identical human cost per review. Almost nobody computes this ratio, and it is the
only number that matters for the decision.

The fourth table proves that habituation is the mechanism rather than an
assumption I baked in. Remove it entirely -- an infinitely patient reviewer -- and
gating everything ships {hab[10 ** 7][0][1]:.1f} harm against blast-radius
gating's {hab[10 ** 7][1][1]:.1f}. **With no habituation, gating everything is by
far the best policy**, exactly as intuition says it should be.

Reintroduce it and the ordering inverts. At a halving load of {25} reviews per
person per day, gate-everything ships {hab[25][0][1]:.1f} and the {BF:.0%} policy
ships {hab[25][1][1]:.1f}.

So the argument for selective gating is not that review is unhelpful. It is that
**attention is the scarce resource, and spending it uniformly is spending it on
the actions that did not need it.** If your reviewers genuinely do not habituate,
gate more; the table tells you what that assumption is worth.

The last table sizes the budget, and it turns over. Harm avoided rises from
{base - bud[0.002][1]:.1f} at a {0.002:.1%} budget to {base - bud[0.05][1]:.1f} at
{0.05:.0%}, then FALLS to {base - bud[0.15][1]:.1f} at {0.15:.0%}.

More review made things worse. The marginal action added to the queue is less
consequential than the ones already there, and adding it dilutes the attention
being paid to those -- so past a point the gate is trading a careful look at the
important items for a cursory look at more of them. **A confirmation gate has an
interior optimum**, and it is smaller than most teams set it.

Harm avoided per hour falls monotonically throughout
({(base - bud[0.002][1]) / bud[0.002][3]:.0f} down to
{(base - bud[0.15][1]) / bud[0.15][3]:.1f}), which is the number to use when
deciding where to stop.

Four rules follow, and the first two contradict standard practice.

**Do not gate on confidence alone.** It is better than random
({base - pol['lowest-confidence 2%'][1]:.1f} against
{base - pol['random 2%'][1]:.1f}) and worse than blast radius, and it inherits
ch:rsn-self-consistency's correlation -- the agent is most confident where it is
most wrong.

**Gate on consequence, and specifically on reversibility.** An action you can undo
does not need a human before it; it needs a human after it, if at all.

**Size the gate from the harm-per-hour column**, not from a policy about
categories. The optimum is interior and it is small.

**Measure your reviewers' actual catch rate under load.** Every number in this
listing turns on a curve nobody measures, and it is measurable: seed known-bad
actions into the review queue at varying volumes and count the catches.""")
```

The second listing turns to the budget.

```python {tier=A name=budget-is-a-population-decision}
"""Three reasons to stop, and only one of them is the agent's to decide.

A run ends for one of three reasons, and systems conflate them:

  DONE      -- the task is complete
  EXHAUSTED -- the budget ran out
  ESCALATED -- this needs a person

ch:ag-loop showed the first is a classifier the agent should not be trusted with,
because a false stop is a confident wrong answer. This listing is about the second
and third, and about how the budget should be shared across a population of tasks
rather than fixed per task (eq:budget-is-a-population-decision).

The connection is ch:rsn-test-time-compute's allocation result: a fixed per-task
budget is the uniform allocation, and uniform is not optimal when tasks differ in
difficulty. What is new here is that an agent can OBSERVE its own progress, which
makes adaptive policies available that a one-shot sampler does not have.
"""
import numpy as np

rng = np.random.default_rng(2531)

M = 20000               # tasks
TOTAL_PER = 18          # mean step budget per task
NEED = 6                # productive steps required


def make_tasks():
    """Per-step success rate varies across tasks: some are easy, a tail is not."""
    p = np.concatenate([
        rng.beta(8.0, 1.5, size=M // 3),          # easy
        rng.beta(2.0, 3.0, size=M // 3),          # middling
        rng.beta(0.6, 9.0, size=M - 2 * (M // 3)),  # hard
    ])
    rng.shuffle(p)
    return np.clip(p, 0.01, 0.999)


P = make_tasks()
TOTAL = M * TOTAL_PER


def simulate(alloc, p=P, need=NEED):
    """Run each task until it accumulates `need` productive steps or exhausts
    its allocation. Returns (completed, steps used, steps used on failures)."""
    prog = np.zeros(M, dtype=np.int64)
    used = np.zeros(M, dtype=np.int64)
    alive = np.ones(M, dtype=bool)
    for t in range(int(alloc.max())):
        idx = np.flatnonzero(alive & (used < alloc))
        if not len(idx):
            break
        used[idx] += 1
        prog[idx] += rng.random(len(idx)) < p[idx]
        alive[idx[prog[idx] >= need]] = False
    done = prog >= need
    return (float(done.mean()), float(used.mean()),
            float(used[~done].sum() / M), done, used)


def pooled_adaptive(p=P, need=NEED, total=TOTAL, cap=200):
    """One shared pool. Round-robin over live tasks; a task leaves the pool the
    moment it finishes. Nothing is predicted -- the policy just reacts."""
    prog = np.zeros(M, dtype=np.int64)
    used = np.zeros(M, dtype=np.int64)
    alive = np.ones(M, dtype=bool)
    spent = 0
    while spent < total and alive.any():
        idx = np.flatnonzero(alive & (used < cap))
        if not len(idx):
            break
        k = len(idx)
        if spent + k > total:
            idx = idx[: total - spent]
            k = len(idx)
        used[idx] += 1
        prog[idx] += rng.random(k) < p[idx]
        alive[idx[prog[idx] >= need]] = False
        spent += k
    done = prog >= need
    return (float(done.mean()), float(used.mean()),
            float(used[~done].sum() / M), done, used)


print(f"{M:,} tasks, {NEED} productive steps each, a total budget of")
print(f"{TOTAL:,} steps ({TOTAL_PER} per task on average). Per-step success")
print("rates span easy, middling and hard.")
print()
print(f"{'band':>18}{'count':>8}{'mean p':>9}{'share of budget':>18}")
print("-" * 53)
bands = [("p > 0.5 (easy)", P > 0.5), ("0.1 < p < 0.5", (P > 0.1) & (P <= 0.5)),
         ("p < 0.1 (hard)", P <= 0.1)]
for name, m in bands:
    print(f"{name:>18}{int(m.sum()):>8}{float(P[m].mean()):>9.3f}"
          f"{m.mean():>18.0%}")

print()
print()
print("Four budget policies spending the same total.")
print()
print(f"{'policy':>34}{'completed':>12}{'steps/task':>13}"
      f"{'wasted on failures':>20}")
print("-" * 79)
res = {}
uniform = np.full(M, TOTAL_PER, dtype=np.int64)
res["uniform (18 each)"] = simulate(uniform)

# A fixed cap chosen conservatively, spending the remainder nowhere.
tight = np.full(M, 10, dtype=np.int64)
res["tight cap (10 each)"] = simulate(tight)

# Difficulty-aware, using a pilot of 3 real steps to estimate p.
pilot = 3
s = rng.binomial(pilot, P)
est = (s + 1.0) / (pilot + 2.0)
lam = np.clip(est, 1e-6, 1 - 1e-6)
w = np.log(np.maximum(1e-9, 0.02 / lam)) / np.log1p(-lam)
w = np.clip(np.round(w), 1, 400)
w = np.round(w * (TOTAL - pilot * M) / w.sum()).astype(np.int64)
res["pilot of 3, then allocate"] = simulate(np.maximum(w + pilot, 1))

res["pooled, stop when done"] = pooled_adaptive()

for name, r in res.items():
    print(f"{name:>34}{r[0]:>12.1%}{r[1]:>13.1f}{r[2]:>20.1f}")

print()
print()
print("Where the steps go, by difficulty band, under uniform and pooled.")
print()
print(f"{'band':>18}{'uniform':>22}{'pooled':>22}")
print(f"{'':>18}{'done':>10}{'steps':>12}{'done':>10}{'steps':>12}")
print("-" * 62)
_, _, _, u_done, u_used = simulate(uniform)
_, _, _, p_done, p_used = pooled_adaptive()
band_tab = {}
for name, m in bands:
    band_tab[name] = (float(u_done[m].mean()), float(u_used[m].mean()),
                      float(p_done[m].mean()), float(p_used[m].mean()))
    v = band_tab[name]
    print(f"{name:>18}{v[0]:>10.1%}{v[1]:>12.1f}{v[2]:>10.1%}{v[3]:>12.1f}")

print()
print()
print("How the policies respond to a bigger budget.")
print()
print(f"{'budget/task':>13}{'uniform':>11}{'tight cap 10':>15}{'pooled':>10}")
print("-" * 49)
bd = {}
for b in (8, 12, 18, 30, 50):
    u = simulate(np.full(M, b, dtype=np.int64))[0]
    t = simulate(np.full(M, min(b, 10), dtype=np.int64))[0]
    pl = pooled_adaptive(total=M * b)[0]
    bd[b] = (u, t, pl)
    print(f"{b:>13}{u:>11.1%}{t:>15.1%}{pl:>10.1%}")

print()
print()
print("And what an escalation policy buys, on top of the best budget policy.")
print("A task that exhausts its budget is handed to a person, who resolves it")
print("with probability r at a cost of 20 steps' worth of time.")
print()
print(f"{'human resolves':>16}{'auto-completed':>16}{'escalated':>12}"
      f"{'end to end':>13}{'human load':>13}")
print("-" * 70)
auto = pooled_adaptive()[0]
esc_rate = 1 - auto
esc = {}
for r in (0.0, 0.4, 0.7, 0.95):
    total_done = auto + esc_rate * r
    esc[r] = (auto, esc_rate, total_done, esc_rate * 20)
    print(f"{r:>16.0%}{auto:>16.1%}{esc_rate:>12.1%}{total_done:>13.1%}"
          f"{esc_rate * 20:>13.1f}")

print(f"""
The first table is the population, and the last column is what makes a fixed
per-task budget wrong before any policy is chosen. The hard band is
{bands[2][1].mean():.0%} of tasks and receives {bands[2][1].mean():.0%} of the
budget under a uniform allocation, and it completes almost none of them.

The second table prices four ways of spending the same total.

Uniform -- {TOTAL_PER} steps each, the default in every framework -- completes
{res['uniform (18 each)'][0]:.1%} using {res['uniform (18 each)'][1]:.1f} steps per
task on average.

A tight cap of {10} completes {res['tight cap (10 each)'][0]:.1%}. It saves steps
and gives them back to nobody, which is the failure of a per-task budget: **unused
allowance from an easy task does not become available to a hard one.**

Estimating difficulty with a pilot and allocating accordingly reaches
{res['pilot of 3, then allocate'][0]:.1%} -- WORSE than uniform, at higher cost.
The pilot spends {3 * M / TOTAL:.0%} of the budget on measurement and the estimate
is too noisy at three observations to place the rest well, which is
ch:rsn-test-time-compute's finding about pilots reproduced in an agent setting.

And pooling -- one shared budget, round-robin over live tasks, a task leaves the
pool the moment it finishes -- reaches {res['pooled, stop when done'][0]:.1%}.

**The best policy predicts nothing.** It does not estimate difficulty, it does not
route, and it does not decide in advance how much anything deserves. It reacts:
finished tasks stop consuming, so their unspent allowance flows to tasks still
working. That is ch:rsn-test-time-compute's early-stopping result arriving in the
agent setting, and it beats the difficulty-aware policy by
{res['pooled, stop when done'][0] - res['pilot of 3, then allocate'][0]:.1%}.

The third table shows the reallocation happening. Under uniform, the easy band uses
{band_tab['p > 0.5 (easy)'][1]:.1f} steps and the hard band {band_tab['p < 0.1 (hard)'][1]:.1f}
-- the hard tasks consume their full allowance and complete
{band_tab['p < 0.1 (hard)'][0]:.1%} of the time. Under pooling the easy band still
uses {band_tab['p > 0.5 (easy)'][3]:.1f} and the MIDDLE band rises from
{band_tab['0.1 < p < 0.5'][1]:.1f} to {band_tab['0.1 < p < 0.5'][3]:.1f} steps,
taking its completion from {band_tab['0.1 < p < 0.5'][0]:.1%} to
{band_tab['0.1 < p < 0.5'][2]:.1%}.

Note where the gain is NOT. The hard band goes from
{band_tab['p < 0.1 (hard)'][0]:.1%} to {band_tab['p < 0.1 (hard)'][2]:.1%} while
consuming {band_tab['p < 0.1 (hard)'][3]:.1f} steps -- the most of any band, for
almost nothing. **Pooling reallocates toward the middle and still overspends on
the hopeless**, which is why the cap in the pooling policy matters and why
ch:ag-what-is-an-agent's per-task cap does not go away.

The fourth table shows the policies diverging rather than converging as the budget
grows: at {8} steps per task the three are within
{max(bd[8]) - min(bd[8]):.1%} of each other, and at {50} pooling leads uniform by
{bd[50][2] - bd[50][0]:.1%}. A tight cap flatlines at {bd[50][1]:.1%} no matter
how much budget exists, because it refuses to spend it.

**A per-task cap converts a budget increase into nothing.** That is the specific
harm of expressing the budget in the wrong place, and it is invisible until
somebody raises the budget and observes no improvement.

The last table is the third termination reason, and it is the one that changes the
system's ceiling rather than its efficiency.

Pooling completes {auto:.1%} automatically and exhausts on {esc_rate:.1%}. Those
exhausted runs are a VISIBLE failure -- ch:ag-loop's distinction -- so they can be
escalated. If a person resolves {0.7:.0%} of what is escalated, end-to-end
completion is {esc[0.7][2]:.1%}; at {0.95:.0%}, {esc[0.95][2]:.1%}.

The human load is {esc[0.7][3]:.1f} steps' worth of time per task on average, and
it is spent only on runs that already failed -- which is the crucial difference
from the previous listing's confirmation gate. **A gate spends attention on
everything to catch a few; an escalation spends it only on the failures.** Same
people, same hours available, and the second design puts them where the agent has
already said it could not cope.

That is the argument for building escalation before confirmation. It uses a signal
the system produces for free (the budget was exhausted), it is triggered by a
visible failure rather than a predicted one, and it does not habituate because the
volume is bounded by the failure rate.""")
```

## 9. Practical Example

The first listing gives three reviewers a stream of three thousand agent actions a
day, $6\%$ of them wrong, with attention halving at twenty-five reviews per person
per day.

```
  gate rate  reviews/day  catch rate   harm/day   hours/day
-----------------------------------------------------------
       0.0%            0       92.0%      326.4         0.0
       2.0%           60       51.4%      329.1         1.5
      20.0%          599       10.2%      324.1        15.0
     100.0%         3000        2.2%      316.7        75.0
```

Gating everything costs $75$ human hours a day — more than nine person-days — to
avoid about $3\%$ of the harm, because the catch rate collapses from $92\%$ to
$2.2\%$ ({{eq:habituation}}). **They are not reviewing; they are clicking**, and the
audit log now records those actions as approved.

Hold the budget at $2\%$ and change only what is looked at:

```
                    policy  reviews/day   catch   harm/day  vs no gate
----------------------------------------------------------------------
                   no gate            0   92.0%      329.4        +3.7
           gate everything         3000    2.2%      320.7        -4.9
                 random 2%           60   51.4%      322.7        -2.9
      lowest-confidence 2%           60   51.1%      287.4       -38.2
   highest-blast-radius 2%           60   51.1%      273.9       -51.8
```

Same reviewers, same review count, same catch rate. Gating by blast radius avoids
$51.8$ units of harm a day against random's $2.9$.

The ratio that should decide the policy:

```
                    policy   hours/day   harm avoided    per hour
-----------------------------------------------------------------
           gate everything        75.0            4.9        0.07
                 random 2%         1.5            2.9        1.97
      lowest-confidence 2%         1.5           38.2       25.50
   highest-blast-radius 2%         1.5           51.8       34.52
```

**A factor of about five hundred between gating everything and gating by
consequence**, at identical cost per review. Almost nobody computes this ratio.

Habituation is the mechanism rather than an assumption, and removing it inverts the
ordering:

```
  halving load   gate all: catch   gate all: harm   top 2%: harm
----------------------------------------------------------------
            25              2.2%            323.6          267.6
           500             30.7%            226.3          222.2
          none             92.0%             27.4          218.4
```

With an infinitely patient reviewer, gating everything is by far the best policy —
$27.4$ against $218.4$ — exactly as intuition says. **The argument for selective
gating is not that review is unhelpful; it is that attention is the scarce
resource.**

And the gate has an interior optimum:

```
  review budget   hours/day   harm/day   harm avoided   per hour
----------------------------------------------------------------
           0.2%         0.1      307.3           18.4     122.53
           1.0%         0.8      280.0           45.7      60.96
           5.0%         3.8      260.6           65.1      17.36
          15.0%        11.2      284.2           41.4       3.68
```

Harm avoided peaks at $5\%$ and *falls* at $15\%$: more review made things worse,
because the marginal item is less consequential than the ones it dilutes attention
from.

The second listing turns to the budget: twenty thousand tasks, six productive steps
each, a total of eighteen steps per task.

```
                            policy   completed   steps/task  wasted on failures
-------------------------------------------------------------------------------
                 uniform (18 each)       54.8%         13.0                 8.1
               tight cap (10 each)       41.2%          8.9                 5.9
         pilot of 3, then allocate       51.7%         15.7                10.8
            pooled, stop when done       63.3%         18.0                11.1
```

**The best policy predicts nothing.** Pooling — one shared budget, round-robin over
live tasks, a task leaves the pool when it finishes — reaches $63.3\%$. The
difficulty-aware policy that spends a pilot reaches $51.7\%$, *worse than uniform*,
because the pilot costs budget and three observations place the rest badly. That is
{{ch:rsn-test-time-compute}}'s pilot result reproduced in an agent setting.

Where the reallocation goes:

```
              band               uniform                pooled
                        done       steps      done       steps
--------------------------------------------------------------
    p > 0.5 (easy)     99.7%         7.8    100.0%         7.8
     0.1 < p < 0.5     40.6%        16.1     70.2%        21.6
    p < 0.1 (hard)      0.1%        18.0      0.9%        30.2
```

The middle band gains most — $40.6\% \to 70.2\%$. And the hard band consumes $30.2$
steps for $0.9\%$ completion, the most of any band for almost nothing: **pooling
still overspends on the hopeless**, which is why a per-task cap is still needed as a
circuit breaker on top of the pool.

The policies diverge as the budget grows:

```
  budget/task    uniform   tight cap 10    pooled
-------------------------------------------------
            8      32.7%          33.1%     35.4%
           18      54.7%          41.2%     63.7%
           50      70.7%          40.8%     83.4%
```

A tight per-task cap holds at about $41\%$ across a sixfold budget increase. **A
per-task cap converts a budget increase into nothing**
({{eq:per-task-cap-wastes-budget}}), and the symptom is usually misdiagnosed as a
model problem.

Finally, escalation on top of the best budget policy:

```
  human resolves  auto-completed   escalated   end to end   human load
----------------------------------------------------------------------
              0%           63.8%       36.2%        63.8%          7.2
             70%           63.8%       36.2%        89.1%          7.2
             95%           63.8%       36.2%        98.2%          7.2
```

A person resolving $70\%$ of exhausted runs takes end-to-end completion to
$89.1\%$. The human load is bounded by the failure rate rather than the action
rate, which is the whole difference: **a gate spends attention on everything to
catch a few; an escalation spends it only on the failures**
({{eq:escalate-not-confirm}}).

## 10. Production Considerations

Report the three terminations separately: completed, exhausted, escalated. Merging
exhausted into "failed" hides the population that escalation can rescue.

Measure your reviewers' catch rate under load. Seed known-bad actions into the review
queue at varying volumes and count the catches. Every number in this chapter turns
on that curve and nobody measures it.

Gate on reversibility, not on confidence and not on category. Confidence fails in
the correlated direction; reversibility is a property of the action you can
determine without asking the model.

Size the gate from harm-avoided-per-hour and expect the optimum to be small. It
turned over at $5\%$ here.

Pool the step budget across a batch, with a per-task cap as a circuit breaker rather
than as the allocation. Three parameters, not one.

Build escalation before confirmation. It uses a signal the system produces for free,
it is triggered by a visible failure, and it does not habituate.

Show the reviewer the action, not the agent's description of the action
({{cite:greshake2023indirect}}). An escalation path is an injection surface.

## 11. Common Mistakes

**Requiring approval for everything.** $75$ human hours a day for a $2.2\%$ catch
rate ({{eq:habituation}}).

**Treating catch rate as a constant.** It is a function of load, and the total
catches saturate at $R\kappa$ regardless of policy.

**Gating on the agent's confidence.** Better than random and correlated with exactly
the errors that matter ({{ch:rsn-self-consistency}}).

**Using a per-task cap as the budget.** It cannot recover unspent allowance and it
flatlines as the total budget grows.

**Spending a pilot to predict difficulty.** It cost budget and did worse than
uniform.

**Pooling without a per-task cap.** Hopeless tasks absorbed $30.2$ steps each for
$0.9\%$ completion.

**Letting the agent decide whether to escalate.** It is the false-stop failure at a
higher level.

## 12. Failure Modes

*Rubber-stamped approval.* The most consequential failure in this chapter, because
it produces a false record. Detect it by measuring catch rate, not approval rate.

*Budget increase with no effect.* A per-task cap absorbing the increase, usually
misread as the model having plateaued.

*Hopeless-task absorption.* Pooling directing an unbounded share of the budget at
tasks that will never finish.

*Escalation flood.* A change that raises the exhaustion rate turns a bounded human
load into an unbounded one overnight. Alert on escalation volume, not just rate.

*Injected justification.* Agent-authored escalation text influenced by untrusted
content, presented to a human as the basis for authorising an action.

## 13. Alternatives

**Post-hoc review with fast undo.** Instead of gating before, make actions reversible
and review a sample after. This moves the whole system into the reversible tier of
{{sec:7-internal-mechanics}} and removes the habituation problem entirely.

**Capability restriction.** Do not give the agent the irreversible action at all.
Strictly better than gating it where the capability is not essential, and it is
{{ch:ag-security}}'s preferred answer.

**Sampled audit.** Review a random $1\%$ *after* the fact for measurement rather than
for prevention. Cheap, unbiased, and it is how you estimate the bad-action rate that
{{eq:gate-on-consequence}} needs.

**Tiered autonomy.** Full autonomy on reversible actions, escalation on
irreversible ones, and a human-initiated mode for anything outside the enumerated
set.

**More reviewers.** {{eq:habituation}} says capacity is $R\kappa$, so this works —
and it is the expensive way to buy what a better gating criterion buys for free.

## 14. Evaluation

Measure the catch-rate-versus-load curve. It is the input to every decision here and
it is measurable with seeded known-bad actions.

Report harm avoided per human hour for your current gating policy, and compare it
against gating by reversibility at the same review volume.

Report the three terminations separately and track the exhausted share over time; it
is your escalation load and your leading indicator of a capability regression.

Measure budget utilisation: what fraction of the allocated steps were actually spent,
and how much of the unspent remainder was recoverable. That number is
{{eq:per-task-cap-wastes-budget}} and it is usually surprising.

And evaluate the budget policy at several total budgets. A policy that does not
improve when you spend more is capping, not allocating.

## 15. Advanced Concepts

**Learned consequence estimation.** {{eq:gate-on-consequence}} needs $h_i$, and for
many action types it is derivable from the action's schema — which table, which
scope, whether an undo exists. Estimating it automatically turns the gate into a
policy rather than a hand-maintained list. {{maturity:EMERGING}}.

**Adaptive gate sizing.** Since the optimum is interior and depends on the current
load and bad-action rate, the gate fraction could be set online from measured
harm-per-hour rather than fixed. No framework exposes this.

**Attention as a scheduled resource.** {{eq:habituation}} implies review quality is
better in a short focused block than spread through the day. Batching escalations
into review windows rather than delivering them as interrupts is a testable
intervention with no cost.

**Value of information for escalation.** Deciding *which* exhausted runs to escalate,
when human capacity is below the exhaustion rate, is the same knapsack as
{{eq:gate-on-consequence}} with resolution probability in place of catch rate.
{{maturity:RESEARCH FRONTIER}} in the sense that nobody formulates it this way.

## 16. Connection to Previous Chapters

{{ch:ag-loop}}'s visible-versus-silent distinction is what makes escalation possible:
you can only escalate a failure the system knows about, and its argument for biasing
against early stopping is what produces the exhausted population this chapter routes.

{{ch:rsn-test-time-compute}}'s allocation results transfer almost unchanged — uniform
is not optimal, pilots cost more than they recover, and adaptive stopping wins — with
the addition that an agent observes its own progress, which a one-shot sampler cannot.

{{ch:ag-recovery}}'s rule that the retry decision belongs to a budget policy rather
than to the agent is implemented here, and its advise-versus-gate distinction applies
to escalation as well.

{{ch:ag-memory}} and {{ch:ag-planning}} supply what an escalation needs to be
actionable: the state at failure and the record of what was attempted.

Ahead: {{ch:ag-security}} takes up capability restriction, which
{{sec:13-alternatives}} rates above gating wherever it is available.

## 17. Exercises

1. Derive the saturation of total catches from {{eq:habituation}} and compute
   $R\kappa$ for the listing's constants. How does it compare with the measured
   optimum?

2. Replace the harm distribution with a light-tailed one and re-run the policy
   comparison. At what tail weight does gating by consequence stop beating gating by
   confidence?

3. Add a per-task cap to the pooling policy in the second listing and sweep it. Where
   is the optimum, and how much of the hard band's waste does it recover?

4. Model escalation capacity as finite and implement the knapsack from
   {{sec:15-advanced-concepts}}. How much better than first-come-first-served?

5. Make the reviewer's catch rate recover overnight and re-run with escalations
   batched into a morning window. How much does batching buy?

6. Take your own action inventory and classify it into the three reversibility tiers.
   What fraction is irreversible, and does it fit inside $R\kappa$?

## 18. Interview Questions

1. Why is requiring human approval for every agent action a bad policy?

2. Your reviewers approve 98% of what they see. Is that good?

3. What should a confirmation gate key on, and why not confidence?

4. You doubled the step budget and completion did not move. What do you check?

5. What is the difference between confirmation and escalation, and which would you
   build first?

6. Why does the best budget policy in this chapter make no predictions?

## 19. Research Questions

1. What does the catch-rate-versus-load curve actually look like for human reviewers
   of agent actions, and how much does it vary by task type and reviewer experience?

2. Can consequence be estimated automatically from an action's schema well enough to
   drive {{eq:gate-on-consequence}}?

3. Does batching escalations into review windows measurably raise catch rate, as
   {{eq:habituation}} implies it should?

4. What is the right per-task cap on top of a pooled budget, and can it be set
   adaptively from observed progress rather than fixed?

5. Is there a progress signal that identifies hopeless tasks early enough to stop
   them before they absorb their cap?

## 20. Chapter Summary

A run ends for three reasons and they have different owners
({{eq:three-terminations}}). {{ch:ag-loop}} handled the first; this chapter handled
the other two, and both results run against standard practice.

**Confirming everything is close to confirming nothing.** Catch rate falls with load
({{eq:habituation}}), and total catches saturate at reviewer capacity regardless of
policy. Three reviewers gating three thousand actions a day dropped to a $2.2\%$
catch rate, spending $75$ human hours to avoid $3\%$ of the harm — and producing an
audit trail saying those actions were reviewed.

Since the number of catches is capped, the only variable is which items occupy the
slots. At a fixed $2\%$ budget, harm avoided per human hour was $0.07$ for gating
everything, $1.97$ for random, $25.50$ for lowest-confidence, and $34.52$ for
highest-blast-radius — **a factor of about five hundred from the selection criterion
alone** ({{eq:gate-on-consequence}}). And the gate has an interior optimum: harm
avoided peaked at a $5\%$ review budget and fell at $15\%$.

On budgets, **the best policy predicts nothing**. Pooling — one shared budget,
round-robin over live tasks, finished tasks stop consuming — reached $63.3\%$ against
uniform's $54.8\%$ and a difficulty-predicting policy's $51.7\%$, which was *worse
than uniform* because the pilot cost budget it could not recover. A per-task cap held
at $41\%$ across a sixfold budget increase: **a per-task cap converts a budget
increase into nothing** ({{eq:per-task-cap-wastes-budget}}). Pooling still needs a
cap as a circuit breaker, because hopeless tasks absorbed $30.2$ steps each for
$0.9\%$ completion.

And escalation beats confirmation as a first investment. It is triggered by a visible
failure the system already knows about, its volume is bounded by the failure rate
rather than the action rate, and it therefore sits on the flat part of the
habituation curve ({{eq:escalate-not-confirm}}). A person resolving $70\%$ of
exhausted runs took end-to-end completion from $63.8\%$ to $89.1\%$.

## 21. Further Reading

{{cite:snell2024testtime}} and {{cite:brown2024monkeys}} for the allocation results
this chapter transfers, and {{ch:rsn-test-time-compute}} for the version with the
agent's own progress observation added.

{{cite:zhou2024webarena}} for what agent success rates on realistic long-horizon
tasks actually are, which is the exhaustion rate escalation has to handle.

{{cite:greshake2023indirect}} for why the escalation path is an injection surface,
and {{ch:ag-security}} for the capability-restriction answer that
{{sec:13-alternatives}} rates above gating.

{{cite:huang2024selfcorrect}} for why the agent should not own any of the three
decisions in {{eq:three-terminations}}.
