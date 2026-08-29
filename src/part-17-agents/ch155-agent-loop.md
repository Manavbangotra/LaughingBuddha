---
id: ag-loop
number: 155
part: XVII
tier: full
status: draft
requires: [control-location, four-decisions, error-message-as-selector]
provides: [loop-is-not-a-chain, stopping-is-a-classifier,
           asymmetric-stopping-errors, no-progress-signal,
           context-change-breaks-loops, visible-versus-silent-failure]
citations: [yao2023react, shinn2023reflexion, liu2024agentbench,
            zhou2024webarena, huang2024selfcorrect, schick2023toolformer]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a loop with slack is
*not* a chain, and why per-step accuracy therefore buys speed rather than success;
identify the stopping decision as a separate classifier with two asymmetric error
rates, and say which one dominates and why; distinguish a visible failure from a
silent one and design for the difference; explain why an agent that is stuck tends
to stay stuck, in terms of what the policy conditions on; and rank the available
loop-breaking interventions by measured effect.

## 2. Why This Matters

{{ch:ag-what-is-an-agent}} established that an agent chooses its own actions
including when to stop, and {{ch:ag-tool-calling}} decomposed the quality of one
action. This chapter puts them in a loop, and the loop turns out to behave
nothing like the chain the previous part's arithmetic would suggest.

**A chain has to get every step right, and its accuracy is $p^k$. A loop only has
to get *enough* steps right eventually.** With slack in the horizon, a failed step
is retried rather than fatal, so {{eq:chain-accuracy-compounds}} stops governing.
{{sec:9-practical-example}} measures this directly: with a perfect stopping
judgement, an agent at $75\%$ per action and one at $99\%$ both complete $100\%$
of tasks. The only difference is $7.99$ steps against $6.06$.

That is a genuinely surprising result the first time you see it, and it relocates
the problem. If per-action accuracy is not what determines success, something else
is — and {{sec:9-practical-example}} finds it in the decision nobody treats as a
component: **the judgement that the task is done.**

That judgement is a binary classifier with two error rates, and the chapter's
central finding is that they are wildly asymmetric. Improving the model's actions
from $90\%$ to $99\%$ bought $+2.4$ points. Improving completion *recognition*
from $85\%$ to $95\%$ bought $+0.1$. Cutting *false stops* from $5\%$ to $1\%$
bought $+19.6$. One parameter is worth roughly eight times the other two combined,
because a missed completion gets retried every step and a false stop is terminal.

The false-stop direction is also the one that produces the worse failure. Running
out of budget is *visible* — the system knows, and can escalate. Stopping early is
*silent*: the agent returns a confident answer having done part of the work. Those
are summed into one "did not succeed" number by almost every agent evaluation, and
they have completely different costs.

The second half of the chapter is about where the steps go when they are not
making progress, and the mechanism is more mundane than the phrase "the agent got
stuck in a loop" suggests. After a failed action, the context is nearly what it
was before — the failure appended an observation and removed nothing — so a policy
conditioned on that context produces nearly the same action. **A loop is a fixed
point of the policy, not a malfunction.** Which tells you what fixes it: not a
better policy, but a changed input.

## 3. Prerequisites

You need {{ch:ag-what-is-an-agent}}'s framing and its cost-distribution result,
and {{ch:ag-tool-calling}}'s decomposition of a single action — in particular the
error-message finding, which reappears here as a loop property rather than a call
property.

From {{ch:rsn-self-consistency}}, the correlated-critic result: the agent's
judgement that it is finished comes from the same model that did the work, which
is what makes {{sec:9-practical-example}}'s numbers optimistic.

Basic Markov chains: absorbing states, expected time to absorption, and what it
means for a chain to have a positive escape probability. Nothing beyond that.

## 4. Intuitive Explanation

Start with the surprise, because it reframes everything after it.

You would expect an agent taking twelve steps at $90\%$ each to succeed about
$28\%$ of the time. That is the chain arithmetic from {{ch:rsn-cot}} and it is
wrong here, for a reason that is obvious once stated: **an agent that fails a step
takes another one.** It does not need twelve consecutive successes; it needs
twelve successes at some point within its budget, and a budget of twenty-five
gives it a lot of room.

So a loop with slack converts a reliability problem into a cost problem. A less
accurate agent does not fail more often; it takes more steps. That is a much
better trade than the chain arithmetic suggests, and it is the reason agent loops
work at all.

But it means the success rate has to be decided by something else, and the
something else is the decision to stop.

Consider what "the agent decides it is done" actually is. It is a classifier
running every step, answering "is the task complete?" from the context. Like any
classifier it has two error rates, and here they behave completely differently.

If it *misses* a completion — the work is done and it does not notice — nothing much
happens. It takes another step, and asks again. And again. Every remaining step is
another chance to notice, so a missed completion is a delay rather than a failure.
An agent that recognises completion only half the time still finishes almost
always; it just takes longer.

If it *falsely* declares completion — the work is not done and it stops — the run is
over. There is no next step in which to reconsider. One error gets unlimited
retries and the other gets one shot at ruining the task.

That asymmetry is the chapter's main practical output, and it points against the
intuitive setting. Most systems tune the stopping decision toward stopping
promptly, because an agent that keeps checking feels indecisive. The arithmetic
says the opposite: **bias heavily toward not stopping, and let the budget end the
run.**

The reason is not only the success rate. It is what the two failures look like
from outside. An agent that exhausts its budget has failed *visibly* — the system
knows it hit the cap, and can retry, escalate, or tell the user. An agent that
stopped early has failed *silently*: it returns an answer, confidently, having
done half the job. For an agent with write access that difference is the whole
risk profile.

The second idea concerns the steps that are not making progress.

The phrase "the agent got stuck in a loop" implies something went wrong. Nothing
did. The agent picked an action from its context; the action failed; the failure
added a line to the context; the agent picked an action from the new context,
which is nearly the old context, and picked the same one.

Understanding it that way tells you what the fix is. A better model still produces
the same action from the same context — it just reaches the stuck state less
often. What breaks the loop is **changing the input**, and every effective
technique in this part is a version of that: refuse to re-issue an action already
tried; raise the temperature after a failure; return an error message that
actually says something; write a note to a scratchpad; replan.

{{sec:9-practical-example}} prices them, and the cheapest one wins: refusing to
retry an already-failed action beats improving the model from $82\%$ to $96\%$,
and it is about fifteen lines of code.

## 5. Formal Explanation

Model the loop as an absorbing Markov chain. Let the state be the number of
productive steps completed, $j \in \{0, \ldots, m\}$, with $m$ the number the task
requires. Each step advances with probability $p$:

$$\Pr[j \to j+1] = p, \qquad \Pr[j \to j] = 1 - p$$ (eq:loop-is-not-a-chain)

The number of steps to reach $m$ is a negative binomial with mean $m/p$, and the
probability of reaching it within a horizon $H$ is:

$$\Pr[\text{complete within } H] = \sum_{i=m}^{H} \binom{i-1}{m-1} p^{m}(1-p)^{i-m}$$ (eq:completion-within-horizon)

Compare with a chain's $p^m$. The chain's success falls geometrically in $m$; the
loop's approaches 1 as $H$ grows, for any $p > 0$. **Slack converts reliability
into cost**, and the expected cost is $m/p$ — linear in $1/p$, not exponential.

Now add the stopping decision. It is a classifier fired every step, with
sensitivity $\alpha$ (declares done when done) and false-positive rate $\beta$
(declares done when not):

$$\Pr[\text{stop} \mid \text{done}] = \alpha, \qquad \Pr[\text{stop} \mid \text{not done}] = \beta$$ (eq:stopping-is-a-classifier)

The two errors enter the outcome completely differently. Once the task is done,
each remaining step is an independent chance to notice, so the probability of
*eventually* stopping correctly within $r$ remaining steps is
$1 - (1-\alpha)^{r}$ — which is close to 1 for any $\alpha$ bounded away from zero
and any reasonable $r$. But a false stop is absorbing: the probability of
surviving $t$ not-yet-done steps without one is $(1-\beta)^{t}$, and there is no
recovery.

$$P(\text{success}) \approx \underbrace{(1-\beta)^{\,t}}_{\text{survive to completion}} \times \underbrace{\big(1 - (1-\alpha)^{r}\big)}_{\text{then notice}}$$ (eq:asymmetric-stopping-errors)

The first factor decays geometrically in the number of steps taken; the second
saturates. **They should never be traded against each other at equal weight**, and
a single "stopping accuracy" number does exactly that.

For the non-productive cycles, model the policy's action as a function of the
context. After a failed action the context changes by one appended observation, so
write the probability that the policy selects a *different* action as an
increasing function of how much that observation changed the state:

$$\Pr[\text{repeat}] = \sigma\big(\text{stickiness}\big), \qquad \text{stickiness} \propto \frac{1}{\Delta(\text{context})}$$ (eq:no-progress-signal)

This is the formal content of "a loop is a fixed point of the policy". It also
identifies the intervention: every loop-breaking technique increases
$\Delta(\text{context})$, and they differ only in how.

$$\Delta \uparrow \;\;\text{via}\;\; \{\text{dedupe, temperature, informative errors, scratchpad, replanning}\}$$ (eq:context-change-breaks-loops)

Finally, the outcome taxonomy. A run ends in one of three states, and the middle
one is usually merged into the third:

$$\{\text{correct}\},\quad \{\text{stopped early: silently wrong}\},\quad \{\text{budget exhausted: visibly failed}\}$$ (eq:visible-versus-silent-failure)

The second and third have different costs, different detectability, and different
remedies. Reporting them as one number discards the distinction that matters most
for anything with side effects.

## 6. Mathematical Foundation

Two consequences of {{eq:asymmetric-stopping-errors}} deserve stating explicitly,
because they invert common practice.

**The relative weight of the two errors scales with the run length.** The false-stop
factor is $(1-\beta)^t$ where $t$ is the number of not-yet-done steps, so a longer
task multiplies $\beta$'s damage while $\alpha$'s saturation is unchanged. A
stopping threshold tuned on short tasks is mis-tuned for long ones, in the
dangerous direction.

**The optimal threshold is not the accuracy-maximising one.** Since a missed stop
costs a step and a false stop costs the task, the loss-minimising operating point
sits far from the point that maximises classification accuracy. If a step costs
$c_s$ and a wrong answer costs $c_w$, the threshold should satisfy roughly
$\beta/\alpha \approx c_s/c_w$, and for most agent products $c_w/c_s$ is in the
hundreds.

Now the horizon. {{eq:completion-within-horizon}} says more slack always helps,
which makes "raise the budget" look like a universal fix. Two things bound it.

The first is cost, and {{ch:ag-what-is-an-agent}} covered it: the horizon sets the
worst case, and worst cases dominate capacity plans.

The second is that slack does not help against a *stuck* agent, which is the point
of {{eq:no-progress-signal}}. If the policy has a fixed point, additional steps are
spent at that fixed point. {{sec:9-practical-example}} measures a naive agent
needing horizon $25$ to reach what a de-duplicating agent reaches at $8$, spending
$11.01$ steps against $6.98$. **The horizon buys completion at roughly 1.6 times
the step cost of fixing the loop**, and it is the intervention that looks free
because it requires no code.

One caveat on all the arithmetic here. {{eq:stopping-is-a-classifier}} treats
$\alpha$ and $\beta$ as constants, and in a real agent the same model produces
both the actions and the completion judgement. {{ch:rsn-self-consistency}}'s
correlated-critic result then applies: $\beta$ will be highest on precisely the
tasks the agent handled badly, because that is where its own judgement is least
reliable. Every number in {{sec:9-practical-example}} is therefore optimistic in
the direction that matters, and the structural fix is the same one that chapter
reached — a completion check that is not the agent.

## 7. Internal Mechanics

### 7.1 The loop, and what is actually in the state

```mermaid {#fig:loop-state caption="One iteration. The state carried forward is the context window, which grows monotonically -- nothing is ever removed by the loop itself."}
flowchart TD
    S[context: goal + history] --> P[model call]
    P --> A{stop?}
    A -- no --> T[choose and execute action]
    T --> O[append observation]
    O --> S
    A -- yes --> R[return]
```

Two properties of that diagram do most of the work in this chapter. The stop test
runs *every* iteration, which is why a missed stop is cheap and a false stop is
not. And the state is append-only, which is why a failure barely changes it.

### 7.2 Why the append-only context causes repetition

The loop never removes anything. A failed action leaves behind: the action, and
whatever the tool said. If the tool said little, the context after the failure is
almost the context before it, plus a line.

A policy is a function of its context. Nearly the same context, nearly the same
distribution over actions, and at low temperature nearly the same action. So the
agent tries it again — and the second failure adds another nearly-identical line.

This is why the fix has to change the state rather than improve the policy, and it
is also why {{ch:ag-tool-calling}}'s error-message result reappears here. An error
that names the field and lists valid values changes the context *a lot*; an error
that says `error` changes it almost not at all. The same string that made retries
succeed also makes loops break, for the same reason.

### 7.3 Three loop-breaking mechanisms, and what they cost

**Deduplication** maintains the set of actions already attempted and removes them
from consideration. Strongest measured effect, and it requires deciding when two
actions are "the same" — which is easy for exact tool-call matches and hard for
near-duplicates with different arguments.

**Temperature on failure** samples more diversely after a failed step. Weaker than
deduplication but needs no notion of action identity, and it degrades gracefully.
{{sec:9-practical-example}} measures it recovering most of deduplication's
benefit.

**Progress checkpoints** require the agent to state what changed after each step,
and abort if nothing did. This is the only one of the three that also detects the
case where actions succeed and accomplish nothing, which the model in
{{sec:9-practical-example}} does not cover.

### 7.4 Where the stopping signal should come from

Since $\beta$ dominates and the agent's own judgement is correlated with its
errors, the highest-value change is to make the completion check something other
than the agent. In rough order of strength:

An **executable check** — the tests pass, the record exists, the file parses. This
is {{ch:rsn-tool-assisted}}'s $q=1$ case, and where it exists $\beta$ goes to
zero.

A **separate model** with a different prompt lineage, which is
{{eq:recoverable-mass}}'s decorrelation argument applied to termination.

A **structural condition** — the plan's steps are all marked complete — which is
weaker but cheap, and it is one of {{ch:ag-planning}}'s few defensible benefits.

The agent's own say-so is the weakest option and the default in most systems.

### 7.5 What the loop costs to serve

Every iteration is a full model call with a context that has grown since the last
one, so the per-step cost rises through the run: {{part:15}}'s KV cache growing,
re-read on every step. A twelve-step run does not cost twelve times a one-step
run; it costs more, and the growth is why the p99 in
{{ch:ag-what-is-an-agent}}'s cost distribution is worse than its step count
suggests.

This gives a second, independent reason to prefer fixing loops over raising
horizons: wasted steps late in a run are the most expensive steps in it.

## 8. Implementation

Two listings. The first isolates the stopping decision from the action quality and
prices them against each other. The second measures where the steps go when the
agent is not making progress, and compares three loop-breaking interventions
against a model improvement.

```python {tier=A name=stopping-is-a-classifier}
"""The stopping decision is a separate classifier, and its errors dominate.

An agent loop is an absorbing Markov chain: states, transitions, and a "done"
state you hope to reach. ch:ag-what-is-an-agent measured the cost distribution.
This listing measures the thing that determines whether the chain absorbs at all,
and it is not the quality of the actions (eq:stopping-is-a-classifier).

An agent decides to stop by judging that the task is complete. That judgement is a
binary classifier with two error rates, and they cause opposite failures: stopping
early when the work is not done, and never stopping when it is. Systems report
"task success" and tune the actions, and the actions may not be where the loss is.
"""
import numpy as np

rng = np.random.default_rng(1721)

N = 60000
NEED = 6              # productive steps a task requires
P_ACT = 0.90          # a step makes progress
HORIZON = 25


def run(p_act, tpr, fpr, horizon=HORIZON, need=NEED):
    """tpr: chance of correctly recognising the task IS done.
    fpr: chance of wrongly declaring it done when it is not.
    Returns (succeeded, stopped_early, ran_out, steps)."""
    prog = np.zeros(N, dtype=np.int64)
    steps = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    early = np.zeros(N, dtype=bool)
    good = np.zeros(N, dtype=bool)
    for _ in range(horizon):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        prog[idx] += (rng.random(len(idx)) < p_act)
        done = prog[idx] >= need
        # The stopping classifier fires on every step, on both kinds of state.
        u = rng.random(len(idx))
        stop = np.where(done, u < tpr, u < fpr)
        good[idx[stop & done]] = True
        early[idx[stop & ~done]] = True
        alive[idx[stop]] = False
    return good, early, ~good & ~early, steps


print(f"A task needs {NEED} productive steps; a step makes progress")
print(f"{P_ACT:.0%} of the time. The agent stops when it JUDGES the task done.")
print(f"Horizon {HORIZON}.")
print()
print("First: a perfect stopping judgement, to isolate the action quality.")
print()
print(f"{'p(action)':>11}{'succeeded':>12}{'ran out':>10}{'mean steps':>13}")
print("-" * 46)
act_tab = {}
for pa in (0.75, 0.85, 0.90, 0.95, 0.99):
    g, e, r, s = run(pa, 1.0, 0.0)
    act_tab[pa] = (float(g.mean()), float(r.mean()), float(s.mean()))
    print(f"{pa:>11.0%}{act_tab[pa][0]:>12.1%}{act_tab[pa][1]:>10.1%}"
          f"{act_tab[pa][2]:>13.2f}")

print()
print()
print("Now hold the actions fixed and vary the stopping judgement instead.")
print(f"Actions are {P_ACT:.0%} accurate throughout.")
print()
print(f"{'recognises':>12}{'false':>9}{'succeeded':>12}{'stopped':>10}"
      f"{'ran':>8}{'mean':>8}")
print(f"{'done':>12}{'stop':>9}{'':>12}{'early':>10}{'out':>8}{'steps':>8}")
print("-" * 59)
stop_tab = {}
CASES = [(1.00, 0.00), (0.95, 0.02), (0.85, 0.05), (0.70, 0.10),
         (0.50, 0.02), (0.95, 0.15)]
# (0.95, 0.02) and (0.50, 0.02) differ only in recognition, which is the
# comparison the narrative turns on.
for tpr, fpr in CASES:
    g, e, r, s = run(P_ACT, tpr, fpr)
    stop_tab[(tpr, fpr)] = (float(g.mean()), float(e.mean()), float(r.mean()),
                            float(s.mean()))
    v = stop_tab[(tpr, fpr)]
    print(f"{tpr:>12.0%}{fpr:>9.0%}{v[0]:>12.1%}{v[1]:>10.1%}{v[2]:>8.1%}"
          f"{v[3]:>8.2f}")

print()
print()
print("Which is the better place to spend? Equal-sized improvements to the")
print("action quality and to the stopping judgement, from a common baseline.")
print()
BASE = (P_ACT, 0.85, 0.05)
g, e, r, s = run(*BASE)
base_succ = float(g.mean())
print(f"{'intervention':>40}{'succeeded':>12}{'change':>10}")
print("-" * 62)
print(f"{'baseline (act 90%, tpr 85%, fpr 5%)':>40}{base_succ:>12.1%}"
      f"{0.0:>+10.1%}")
spend = {}
for name, args in [
        ("actions 90% -> 95%", (0.95, 0.85, 0.05)),
        ("actions 90% -> 99%", (0.99, 0.85, 0.05)),
        ("recognises done 85% -> 95%", (P_ACT, 0.95, 0.05)),
        ("false stops 5% -> 1%", (P_ACT, 0.85, 0.01)),
        ("both stopping fixes", (P_ACT, 0.95, 0.01))]:
    g, e, r, s = run(*args)
    spend[name] = float(g.mean())
    print(f"{name:>40}{spend[name]:>12.1%}{spend[name] - base_succ:>+10.1%}")

print()
print()
print("A false stop is not a failure to finish -- it is a WRONG ANSWER returned")
print("confidently. Split the outcomes by what the user actually receives.")
print()
print(f"{'false stop rate':>17}{'correct':>10}{'confidently':>14}{'visibly':>10}")
print(f"{'':>17}{'answer':>10}{'wrong':>14}{'failed':>10}")
print("-" * 51)
fs_tab = {}
for fpr in (0.0, 0.01, 0.03, 0.05, 0.10, 0.20):
    g, e, r, s = run(P_ACT, 0.90, fpr)
    fs_tab[fpr] = (float(g.mean()), float(e.mean()), float(r.mean()))
    print(f"{fpr:>17.0%}{fs_tab[fpr][0]:>10.1%}{fs_tab[fpr][1]:>14.1%}"
          f"{fs_tab[fpr][2]:>10.1%}")

print(f"""
The first table is the loop working as advertised, and it is the comparison every
agent system reports. With a perfect stopping judgement, EVERY action quality
reaches {act_tab[0.75][0]:.1%}. Not approximately -- exactly, at
{0.75:.0%} per action and at {0.99:.0%}. The only thing that changes is the number
of steps taken: {act_tab[0.75][2]:.2f} against {act_tab[0.99][2]:.2f}.

That is worth sitting with, because it contradicts the intuition carried over from
ch:rsn-cot. A chain has to get every step right and its accuracy is $p^k$. **A
loop only has to get ENOUGH steps right eventually**, so with slack in the horizon
it converts a reliability problem into a cost problem. Per-action accuracy buys
speed, not success.

Which means the loop's success has to be decided somewhere else, and the second
table finds where.

Holding actions at {P_ACT:.0%}, a stopping judgement that recognises completion
{0.85:.0%} of the time with a {0.05:.0%} false-stop rate scores
{stop_tab[(0.85, 0.05)][0]:.1%} against a perfect judgement's
{stop_tab[(1.0, 0.0)][0]:.1%}. **The actions did not change, and
{stop_tab[(1.0, 0.0)][0] - stop_tab[(0.85, 0.05)][0]:.1%} of the outcome was
decided by a classifier nobody was measuring.**

The third table prices the interventions, and the result was not the one I
expected.

Improving actions from {P_ACT:.0%} to {0.99:.0%} -- about the most you could hope
for, and a large investment -- buys
{spend['actions 90% -> 99%'] - base_succ:+.1%}. Improving completion RECOGNITION
from {0.85:.0%} to {0.95:.0%} buys
{spend['recognises done 85% -> 95%'] - base_succ:+.1%}, which is nothing. Cutting
FALSE STOPS from {0.05:.0%} to {0.01:.0%} buys
{spend['false stops 5% -> 1%'] - base_succ:+.1%}.

One parameter is worth roughly eight times the other two combined, and the reason
is structural rather than numerical. **The two stopping errors are not
symmetric.**

A missed completion is recoverable. The agent does not notice it is done, takes
another step, and gets another chance to notice -- and another, every step until
the horizon. Over a run with slack, a recognition rate of {0.5:.0%} and one of
{0.95:.0%} produce almost the same outcome
({stop_tab[(0.5, 0.02)][0]:.1%} against {stop_tab[(0.95, 0.02)][0]:.1%}); the
low-recognition agent simply takes longer ({stop_tab[(0.5, 0.02)][3]:.2f} steps
against {stop_tab[(0.95, 0.02)][3]:.2f}).

A false stop is terminal. It ends the run, and there is no next step in which to
correct it. So one error gets retried at every opportunity and the other gets one
shot at ruining the task, and their per-step rates should never be compared
directly (eq:stopping-is-a-classifier).

This also explains the {0.5:.0%}-recognition row, which looks anomalous until you
see it: {stop_tab[(0.5, 0.02)][0]:.1%} success from an agent that recognises
completion only half the time. Half of a lot of chances is still enough chances.

The fourth table is why the false-stop direction deserves separate treatment
beyond its size, and it is about what the user receives rather than what the
metric records.

At a {0.05:.0%} false-stop rate, {fs_tab[0.05][1]:.1%} of runs end early and
{fs_tab[0.05][2]:.1%} exhaust the horizon. At {0.20:.0%} it is
{fs_tab[0.2][1]:.1%} against {fs_tab[0.2][2]:.1%}.

Those two failures are usually summed into one "did not succeed" number, and they
are not comparable. Exhausting the horizon is a VISIBLE failure: the budget was
hit, the system knows it, it can retry or escalate. Stopping early is INVISIBLE:
the agent returns an answer, confidently, having done part of the work. **A
visible failure costs a retry; a confident wrong answer costs whatever the wrong
answer causes**, and for an agent with write access that is ch:ag-security's
subject.

So the design conclusion is a threshold, and it points against the intuitive
setting. **Bias the stopping classifier heavily toward not stopping, and let the
budget end the run.** Missed completions cost steps, which are cheap and bounded
by the horizon. False stops cost correctness, and they are not recoverable. Most
systems tune this the other way, because an agent that stops promptly feels
better than one that keeps checking.

Two caveats, and the second is the larger.

The horizon is doing a great deal of work here. All of the "a missed stop is
recoverable" argument depends on there being steps left, so at a horizon close to
{NEED} the two error directions become comparable and the argument weakens. The
right reading is that the asymmetry is a function of slack, and slack is a design
parameter.

And this models completion detection as a classifier with a fixed operating point.
In a real agent that judgement comes from the same model that took the actions,
which is ch:rsn-self-consistency's correlated critic: the false-stop rate will be
highest exactly on the tasks the agent handled badly, because that is where its
own judgement is least reliable. **The numbers here are optimistic in precisely
the direction that matters**, and the fix is the same one that chapter reached --
a completion check that is not the agent.""")
```

The second listing turns to the steps that are not productive.

```python {tier=A name=no-progress-signal}
"""Where the steps go: progress, repetition, and the cheapest fix for a loop.

The previous listing showed the stopping decision dominating success. This one is
about the other half of the loop's arithmetic: what the steps are SPENT on, and
why an agent that is not making progress usually keeps not making progress
(eq:no-progress-signal).

The mechanism is specific. An agent chooses its next action from the context, and
after a failed action the context is nearly the same as before it -- the failure
added an observation and removed nothing. So the same context produces the same
action, and the loop repeats. That is not a mysterious pathology; it is what a
policy does when its input barely changed.

Three interventions are measured against it, and the cheapest one wins.
"""
import numpy as np

rng = np.random.default_rng(1811)

N = 80000
NEED = 6
HORIZON = 25
P_ACT = 0.82           # a fresh action makes progress
STICK = 0.75           # after a failure, chance of repeating the same action


def run(mode, p_act=P_ACT, stick=STICK, horizon=HORIZON):
    """mode:
       'naive'    -- repeat-prone: a failed action is likely to be retried as is
       'dedupe'   -- an action already tried and failed is not tried again
       'temp'     -- after a failure, sample a different action with prob 1-stick
                     (i.e. raise temperature only when stuck)
       'ideal'    -- never repeats
    """
    prog = np.zeros(N, dtype=np.int64)
    wasted = np.zeros(N, dtype=np.int64)
    repeats = np.zeros(N, dtype=np.int64)
    failed_last = np.zeros(N, dtype=bool)
    tried_bad = np.zeros(N, dtype=np.int64)     # how many distinct duds tried
    alive = np.ones(N, dtype=bool)
    steps = np.zeros(N, dtype=np.int64)
    for _ in range(horizon):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        if mode == "naive":
            rep = failed_last[idx] & (rng.random(len(idx)) < stick)
        elif mode == "temp":
            rep = failed_last[idx] & (rng.random(len(idx)) < stick * 0.35)
        else:
            rep = np.zeros(len(idx), dtype=bool)
        # A repeated action repeats its outcome: it already failed.
        ok = np.where(rep, False, rng.random(len(idx)) < p_act)
        if mode == "dedupe":
            # Ruling out duds raises the chance the next fresh action works.
            boost = 1.0 + 0.06 * np.minimum(tried_bad[idx], 5)
            ok = rng.random(len(idx)) < np.minimum(p_act * boost, 0.99)
        prog[idx[ok]] += 1
        repeats[idx[rep]] += 1
        wasted[idx[~ok & ~rep]] += 1
        tried_bad[idx[~ok & ~rep]] += 1
        failed_last[idx] = ~ok
        fin = prog[idx] >= NEED
        alive[idx[fin]] = False
    done = prog >= NEED
    return (float(done.mean()), float(steps.mean()), float(repeats.mean()),
            float(wasted.mean()))


MODES = [("naive (repeats on failure)", "naive"),
         ("raise temperature when stuck", "temp"),
         ("do not retry a failed action", "dedupe"),
         ("never repeats (ideal)", "ideal")]

print(f"A task needs {NEED} productive steps in a horizon of {HORIZON}. A fresh")
print(f"action works {P_ACT:.0%} of the time. After a failure the naive agent")
print(f"retries the same action {STICK:.0%} of the time, because its context")
print("barely changed.")
print()
print(f"{'loop policy':>32}{'completed':>12}{'steps':>9}{'repeats':>10}"
      f"{'wasted':>9}")
print("-" * 72)
res = {}
for name, m in MODES:
    r = run(m)
    res[name] = r
    print(f"{name:>32}{r[0]:>12.1%}{r[1]:>9.2f}{r[2]:>10.2f}{r[3]:>9.2f}")

print()
print()
print("How much of the horizon does repetition consume? Sweep the stickiness --")
print("how strongly a failed action pulls the agent to try it again.")
print()
print(f"{'stickiness':>12}{'completed':>12}{'repeats':>10}{'wasted':>17}")
print(f"{'':>12}{'':>12}{'per run':>10}{'share of steps':>17}")
print("-" * 51)
st_tab = {}
for st in (0.0, 0.25, 0.50, 0.75, 0.90):
    r = run("naive", stick=st)
    st_tab[st] = r
    print(f"{st:>12.0%}{r[0]:>12.1%}{r[2]:>10.2f}"
          f"{(r[2] + r[3]) / r[1]:>17.1%}")

print()
print()
print("Does a bigger horizon fix it? Naive against dedupe, horizon swept.")
print()
print(f"{'horizon':>9}{'naive':>22}{'dedupe':>20}")
print(f"{'':>9}{'completed':>12}{'steps':>10}{'completed':>11}{'steps':>9}")
print("-" * 51)
hz, hzd = {}, {}
for h in (8, 10, 15, 25, 40, 60):
    r = run("naive", horizon=h)
    d = run("dedupe", horizon=h)
    hz[h], hzd[h] = r, d
    print(f"{h:>9}{r[0]:>12.1%}{r[1]:>10.2f}{d[0]:>11.1%}{d[1]:>9.2f}")

print()
print()
print("And the comparison that decides where to spend: a better model against")
print("a loop-detection rule, at the same horizon.")
print()
print(f"{'change':>40}{'completed':>12}{'steps':>9}")
print("-" * 61)
base = run("naive")
opts = [("baseline (naive, action 82%)", ("naive", P_ACT)),
        ("action 82% -> 90%", ("naive", 0.90)),
        ("action 82% -> 96%", ("naive", 0.96)),
        ("keep 82%, add dedupe", ("dedupe", P_ACT)),
        ("keep 82%, temperature on failure", ("temp", P_ACT))]
cmp_ = {}
for name, (m, pa) in opts:
    r = run(m, p_act=pa)
    cmp_[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[1]:>9.2f}")

nv = res["naive (repeats on failure)"]
dd = res["do not retry a failed action"]
tp = res["raise temperature when stuck"]
idl = res["never repeats (ideal)"]
print(f"""
The first table is where the steps go, and the repeats column is the whole
subject.

The naive agent completes {nv[0]:.1%} of tasks, spending {nv[1]:.2f} steps of
which {nv[2]:.2f} are repeats of an action that already failed. That is
{nv[2] / nv[1]:.0%} of its budget spent re-running something it has already
watched fail.

The mechanism is not exotic. **After a failed action the context is nearly
unchanged** -- the failure appended an observation and removed nothing -- so a
policy conditioned on that context produces nearly the same action. A loop is a
fixed point of the policy, not a bug in it, and describing it as "the agent got
confused" gets the causality backwards.

The second table sweeps how strongly a failure pulls the agent back to the same
action. At {0:.0%} stickiness the agent completes {st_tab[0.0][0]:.1%}; at
{0.9:.0%} it completes {st_tab[0.9][0]:.1%} and wastes
{(st_tab[0.9][2] + st_tab[0.9][3]) / st_tab[0.9][1]:.0%} of its steps.

Note that stickiness is not a property anybody chose. It is the degree to which
a failure changes the context, and a tool that returns a terse error changes it
less than one that returns a specific fault -- which is ch:ag-tool-calling's
error-message result arriving as a loop property. **The error message is also a
loop-breaking mechanism**, and that is a second, independent reason to write it.

The third table is the response most teams reach for first, and the comparison
column is what makes it the wrong one.

Raising the naive agent's horizon does work: {hz[8][0]:.1%} at {8} steps,
{hz[25][0]:.1%} at {25}, {hz[60][0]:.1%} at {60}. So "give it more budget" is not
useless advice.

But look at what dedupe achieves at each horizon. At {8} steps dedupe completes
{hzd[8][0]:.1%} against naive's {hz[8][0]:.1%}; at {10} it is already
{hzd[10][0]:.1%} against {hz[10][0]:.1%}, and it stays there.

**The naive agent needs a horizon of {25} to reach what dedupe reaches at {8}**,
and it spends {hz[25][1]:.2f} steps doing it against dedupe's {hzd[8][1]:.2f} --
{hz[25][1] / hzd[8][1]:.1f} times the cost for a slightly worse result. Dedupe's
step count is flat at {hzd[60][1]:.2f} across every horizon, because it never
needed the extra room.

So the horizon does buy completion, and it buys it inefficiently. **A bigger
budget buys a stuck agent more time to be stuck**, and it is the intervention
that looks free because it requires no code and appears in no diff.

The fourth table is the comparison that matters, and the ordering is decisive.

Improving the action from {P_ACT:.0%} to {0.96:.0%} -- a large model
investment -- takes completion from {cmp_['baseline (naive, action 82%)'][0]:.1%}
to {cmp_['action 82% -> 96%'][0]:.1%}. Keeping the {P_ACT:.0%} model and simply
refusing to re-issue an action that already failed takes it to
{cmp_['keep 82%, add dedupe'][0]:.1%}, in {cmp_['keep 82%, add dedupe'][1]:.2f}
steps against the baseline's {cmp_['baseline (naive, action 82%)'][1]:.2f}.

**A loop-detection rule beats a large model improvement, and it is about fifteen
lines of code.** It is not doing anything clever: it maintains the set of actions
already attempted in this run and removes them from consideration. The reason it
works so well is that it attacks the term that a better model does not -- a more
accurate policy still produces the same action from the same context, so it gets
stuck less often but exits no faster once it is.

Raising the temperature on failure -- a softer version of the same idea, sampling
a different action rather than forbidding the old one -- reaches
{cmp_['keep 82%, temperature on failure'][0]:.1%}. Most of the benefit, none of
the bookkeeping, and it degrades gracefully when the "same action" is hard to
define, which in a real agent it often is.

The general shape is worth naming because it recurs through the rest of this
part. **An agent's failures divide into ones a better policy fixes and ones only
a change of state fixes.** Repetition is the second kind: the policy is behaving
correctly given its input, and the fix is to change the input. Every effective
loop-breaking technique in this part -- deduplication, temperature on failure,
informative errors, replanning, an explicit scratchpad -- is a way of making the
context after a failure genuinely different from the context before it.""")
```

## 9. Practical Example

The first listing gives a task requiring six productive steps, a horizon of
twenty-five, and a stopping judgement with a settable operating point.

```
  p(action)   succeeded   ran out   mean steps
----------------------------------------------
        75%      100.0%      0.0%         7.99
        90%      100.0%      0.0%         6.66
        99%      100.0%      0.0%         6.06
```

With a perfect stopping judgement, *every* action quality reaches $100\%$. Not
approximately — exactly, at $75\%$ per action and at $99\%$. The only thing that
changes is the step count.

That contradicts the intuition carried from {{ch:rsn-cot}}, and the reason is
{{eq:loop-is-not-a-chain}}: a chain needs every step right, a loop needs enough
steps right eventually. **Per-action accuracy buys speed, not success**, whenever
there is slack.

So success is decided somewhere else:

```
  recognises    false   succeeded   stopped     ran    mean
        done     stop                 early     out   steps
-----------------------------------------------------------
        100%       0%      100.0%      0.0%    0.0%    6.68
         95%       2%       89.3%     10.7%    0.0%    6.35
         85%       5%       74.7%     25.3%    0.0%    5.91
         50%       2%       89.2%     10.8%    0.0%    7.18
         95%      15%       40.0%     60.0%    0.0%    4.41
```

The actions never change across that table. A judgement at $85\%$ recognition and
$5\%$ false stops scores $74.7\%$ against a perfect judgement's $100\%$:
**$25.3$ points of the outcome decided by a classifier nobody was measuring.**

Pricing the interventions gave the result I did not expect:

```
                            intervention   succeeded    change
--------------------------------------------------------------
     baseline (act 90%, tpr 85%, fpr 5%)       74.8%     +0.0%
                      actions 90% -> 99%       77.2%     +2.4%
              recognises done 85% -> 95%       74.9%     +0.1%
                    false stops 5% -> 1%       94.4%    +19.6%
```

One parameter is worth about eight times the other two combined, and the reason is
structural. Compare the $95\%/2\%$ row with the $50\%/2\%$ row above: recognition
of $50\%$ scores $89.2\%$ against $95\%$ recognition's $89.3\%$ — indistinguishable
— while taking $7.18$ steps instead of $6.35$. **A missed completion is a delay; a
false stop is terminal** ({{eq:asymmetric-stopping-errors}}).

And the two failures are not interchangeable:

```
  false stop rate   correct   confidently   visibly
                     answer         wrong    failed
---------------------------------------------------
               1%     94.6%          5.4%      0.0%
               5%     75.0%         25.0%      0.0%
              20%     28.9%         71.1%      0.0%
```

Every failure here is a *confident wrong answer*, not a visible one. That is what
tuning a stopping classifier toward promptness produces, and it is why the design
conclusion is to bias heavily against stopping and let the budget end the run
({{eq:visible-versus-silent-failure}}).

The second listing asks where the steps go. A fresh action works $82\%$ of the
time; after a failure the naive agent retries the same action $75\%$ of the time.

```
                     loop policy   completed    steps   repeats   wasted
------------------------------------------------------------------------
      naive (repeats on failure)       95.9%    11.01      3.80     1.30
    raise temperature when stuck      100.0%     7.78      0.46     1.32
    do not retry a failed action      100.0%     7.06      0.00     1.06
           never repeats (ideal)      100.0%     7.32      0.00     1.32
```

The naive agent spends $3.80$ of its $11.01$ steps — $35\%$ of its budget —
re-running something it has already watched fail. Not because anything
malfunctioned: **after a failed action the context is nearly unchanged**, so the
policy produces nearly the same action ({{eq:no-progress-signal}}).

Stickiness is not a property anyone chose. It is how much a failure changes the
context, which makes {{ch:ag-tool-calling}}'s error message a loop-breaking
mechanism as well as a retry-conditioning one:

```
  stickiness   completed   repeats           wasted
                           per run   share of steps
---------------------------------------------------
          0%      100.0%      0.00            18.1%
         50%       99.9%      1.32            30.5%
         90%       73.9%      8.36            64.6%
```

The response most teams reach for first is a bigger budget, and the comparison
column shows why it is the wrong one:

```
  horizon                 naive              dedupe
            completed     steps  completed    steps
---------------------------------------------------
        8       46.2%      7.31      93.6%     6.98
       10       58.8%      8.32     100.0%     7.06
       25       95.9%     11.01     100.0%     7.06
       60      100.0%     11.29     100.0%     7.06
```

The naive agent needs a horizon of $25$ to approach what dedupe reaches at $8$,
spending $11.01$ steps against $6.98$. Dedupe's step count is flat across every
horizon because it never needed the extra room. **A bigger budget buys a stuck
agent more time to be stuck.**

Finally, the comparison that decides where to spend:

```
                                  change   completed    steps
-------------------------------------------------------------
            baseline (naive, action 82%)       95.8%    11.01
                       action 82% -> 90%       98.9%     8.61
                       action 82% -> 96%       99.8%     6.98
                    keep 82%, add dedupe      100.0%     7.06
```

**A loop-detection rule beats a large model improvement**, at about fifteen lines
of code. It attacks the term a better model does not: a more accurate policy still
produces the same action from the same context, so it gets stuck less often and
exits no faster once it is.

## 10. Production Considerations

Expose the stopping decision as a component with a threshold, not as a sentence in
a prompt. It is the highest-leverage parameter in the loop and most systems do not
have a name for it.

Bias it hard against stopping. {{sec:6-mathematical-foundation}} gives the
threshold as roughly $\beta/\alpha \approx c_s/c_w$, and for most products the
cost of a wrong answer exceeds the cost of a step by two orders of magnitude.

Make the completion check something other than the agent wherever possible: an
executable check, a separate model, or a structural condition. This is the only
change that attacks $\beta$ at its source.

Report three outcomes, not two: succeeded, stopped early, budget exhausted. The
middle one is a confident wrong answer and it is invisible when merged with the
third.

Add deduplication. Maintain the set of attempted actions and refuse to re-issue
one. {{sec:9-practical-example}} makes it the highest-return change in this
chapter and it needs no model.

Raise temperature after a failure rather than globally. It gets most of
deduplication's benefit without requiring a notion of action identity.

Write informative tool errors — again. They raise first-retry success
({{ch:ag-tool-calling}}) *and* they break loops, for the same reason.

Track repeats per run as a first-class metric. It is the leading indicator of the
failure this chapter is about, and it is trivially computable from a trace.

## 11. Common Mistakes

**Applying chain arithmetic to a loop.** $p^k$ is wrong when the agent can retry;
with slack, per-step accuracy sets cost rather than success
({{eq:loop-is-not-a-chain}}).

**Tuning the stopping decision for promptness.** It trades a delay for a confident
wrong answer, at about eight-to-one against
({{eq:asymmetric-stopping-errors}}).

**Reporting one "stopping accuracy".** It averages two errors with very different
consequences.

**Raising the horizon to fix a stuck agent.** {{sec:9-practical-example}}: 1.6
times the step cost of fixing the loop, for a slightly worse result.

**Merging "stopped early" into "failed".** The first is silent and the second is
visible, and only the second can be escalated.

**Assuming the agent knows when it is done.** That judgement comes from the model
that did the work, so it is least reliable exactly where it matters
({{ch:rsn-self-consistency}}).

## 12. Failure Modes

*Silent early stop.* The agent returns a partial result confidently. No error, no
alert, and the metric records it identically to a budget exhaustion.

*Action repetition.* The dominant consumer of steps in an unprotected loop —
$35\%$ of the budget in {{sec:9-practical-example}}, rising to $65\%$ at high
stickiness.

*Productive-looking non-progress.* Actions succeed and accomplish nothing: a
search that returns results nobody needed, a file read that changes no decision.
Deduplication does not catch this; a progress checkpoint does.

*Late-run cost blowup.* Wasted steps occur late, when the context is longest, so
they are the most expensive steps in the run ({{part:15}}).

*Threshold drift across task lengths.* A stopping threshold tuned on short tasks
is mis-tuned for long ones in the dangerous direction, because $\beta$'s damage
scales with run length.

## 13. Alternatives

**A fixed pipeline.** {{ch:ag-what-is-an-agent}}: if the tail mass does not justify
autonomy, none of this chapter's problems need solving.

**Bounded action sets.** Restricting the action space shrinks the state space
enough that repetition can be detected structurally rather than heuristically.

**Externalised progress state.** Keep an explicit task list outside the context and
mark items complete. This gives a structural completion condition and makes
progress checkable, and it is the defensible core of {{ch:ag-planning}}.

**Verifier-gated stopping.** Where an executable check exists,
{{ch:rsn-tool-assisted}}'s $q = 1$ sets $\beta$ to zero, which
{{sec:9-practical-example}} says is worth more than any other change available.

**Human confirmation before returning.** Converts a silent failure into a delay,
at the cost of attention. {{ch:ag-termination}} prices it.

## 14. Evaluation

Measure $\alpha$ and $\beta$ separately, by running to completion with the stop
suppressed and recording when the agent *would* have stopped. Both numbers are
recoverable from traces you can already collect.

Report the three-way outcome split
({{eq:visible-versus-silent-failure}}) rather than a success rate.

Report steps at p50, p90 and p99, and repeats per run. The second is the
loop-health metric and the first is the capacity metric.

Evaluate at several horizons. A system's behaviour at its budget tells you whether
the budget is binding, and a completion rate that keeps rising with horizon means
the agent is stuck rather than slow.

And evaluate stopping on *long* tasks specifically, because that is where $\beta$
does the most damage and where a threshold tuned elsewhere fails.

## 15. Advanced Concepts

**Learned stopping.** Treat completion detection as a trained classifier over
trajectory features rather than a prompted judgement. It is a small supervised
problem with abundant labels — every completed trace is one — and it decorrelates
the judgement from the policy. {{maturity:EMERGING}} and unusually tractable.

**Progress as a measurable quantity.** Everything in
{{sec:12-failure-modes}}'s productive-looking non-progress needs a definition of
progress that is not "an action succeeded". Distance-to-goal estimates,
state-diff sizes, and information-gain proxies are all candidates and none is
standard. {{maturity:RESEARCH FRONTIER}}.

**Adaptive horizons.** {{ch:rsn-test-time-compute}}'s allocation result applies: a
fixed per-task budget is the uniform allocation, and difficulty-aware or
outcome-adaptive budgets beat it. The catch is the same one — it needs a signal
that the task is going well, which is the quantity above.

**The correlation ceiling.** $\beta$'s dependence on the agent's own reliability
({{eq:correlated-critic}}) bounds how good a self-judged stopping decision can be,
and quantifying that bound per task family would say when an external check is
mandatory rather than merely better.

## 16. Connection to Previous Chapters

{{ch:rsn-cot}}'s compounding result does *not* transfer to a loop with slack, and
{{eq:loop-is-not-a-chain}} says why. That is one of the few places in this book
where an earlier chapter's arithmetic stops applying, and it is worth noticing as
such.

{{ch:ag-tool-calling}}'s error-message finding reappears as
{{eq:context-change-breaks-loops}}: the same string that conditions a retry also
breaks a repetition cycle, because both are about how much the failure changed the
context.

{{ch:rsn-self-consistency}}'s correlated critic is the ceiling on self-judged
stopping, and {{ch:rsn-tool-assisted}}'s executable check is the escape from it.

{{ch:ag-what-is-an-agent}}'s cost distribution is what
{{sec:9-practical-example}}'s repeats column explains: the p99 is long because
stuck runs spend their whole budget.

Ahead: {{ch:ag-react}} asks whether interleaving reasoning between these steps
helps; {{ch:ag-recovery}} takes up what to do when the loop detects it is going
wrong; and {{ch:ag-termination}} develops the budget and the human gate.

## 17. Exercises

1. Derive the loss-minimising stopping threshold from
   {{eq:asymmetric-stopping-errors}} for step cost $c_s$ and wrong-answer cost
   $c_w$, and compute it for $c_w/c_s = 100$.

2. In the first listing, make $\beta$ depend on progress — higher when the agent
   has done less — and re-run. How much of the optimism does that remove?

3. Set the horizon to exactly `NEED` and re-run the stopping sweep. Show that the
   two error directions become comparable, and explain why.

4. Add "productive-looking non-progress" to the second listing: actions that
   succeed but do not advance. Which of the three interventions still helps?

5. Measure how deduplication's benefit changes as the action space grows. At what
   size does forbidding tried actions stop being useful?

6. Take an agent trace you own, count repeated actions, and compute what fraction
   of steps they represent.

## 18. Interview Questions

1. An agent takes 12 steps at 90% each. What is its success rate?

2. Why is a missed completion cheap and a false stop expensive?

3. Your agent's success rate is 75% and its actions are 95% accurate. Where would
   you look?

4. What is the difference between "ran out of budget" and "stopped early", and why
   does it matter more than the success rate?

5. Why does an agent repeat a failing action, and what is the cheapest fix?

6. When does raising the step budget not help?

## 19. Research Questions

1. Can completion detection be trained as a separate classifier that generalises
   across tasks, and how much of the correlation with the policy survives?

2. What is a task-independent measure of *progress*, and does it detect the
   productive-looking non-progress that deduplication misses?

3. How does the optimal stopping threshold vary with task length, and can it be
   set adaptively from within a run?

4. Deduplication requires an action-identity relation. What is the right one for
   near-duplicate calls with different arguments, and does the choice change the
   measured benefit?

5. Does the loop-versus-chain distinction hold for tasks where a failed action has
   side effects that cannot be undone — and what replaces the slack argument there?

## 20. Chapter Summary

**A loop with slack is not a chain.** With a perfect stopping judgement, an agent
at $75\%$ per action and one at $99\%$ both completed $100\%$ of tasks, differing
only in steps ($7.99$ against $6.06$). {{eq:loop-is-not-a-chain}}: retries convert
reliability into cost, and {{ch:rsn-cot}}'s $p^k$ stops governing.

So success is decided by the stopping judgement, which is a classifier with two
error rates ({{eq:stopping-is-a-classifier}}) that are wildly asymmetric.
Improving actions $90\% \to 99\%$ bought $+2.4$ points; improving completion
recognition $85\% \to 95\%$ bought $+0.1$; cutting false stops $5\% \to 1\%$
bought $+19.6$. **A missed completion is retried every remaining step; a false
stop is terminal** ({{eq:asymmetric-stopping-errors}}). An agent recognising
completion half the time scored the same as one recognising it $95\%$ of the time,
and simply took longer.

The two failures are also not interchangeable. Budget exhaustion is *visible* and
can be escalated; stopping early is a confident wrong answer
({{eq:visible-versus-silent-failure}}). So bias the classifier hard against
stopping and let the budget end the run — which is the opposite of how promptness
is usually tuned.

On the steps themselves: a naive agent spent $35\%$ of its budget re-running
actions that had already failed, because **after a failure the context is nearly
unchanged and a policy on nearly the same context produces nearly the same
action** ({{eq:no-progress-signal}}). A loop is a fixed point, not a malfunction.

Which says the fix changes the input, not the policy
({{eq:context-change-breaks-loops}}). Refusing to re-issue a failed action beat
improving the model from $82\%$ to $96\%$, at fifteen lines of code; raising
temperature on failure recovered most of it without needing action identity; and
raising the horizon bought the same completion at $1.6\times$ the step cost.

## 21. Further Reading

{{cite:yao2023react}} is the loop this chapter formalises, and
{{ch:ag-react}} reads it against the interleaving cost.

{{cite:shinn2023reflexion}} is the loop with a memory of past failures, which is
{{eq:context-change-breaks-loops}} implemented as episodic state —
{{ch:ag-recovery}} takes it apart.

{{cite:liu2024agentbench}} identifies long-horizon consistency as the agent
bottleneck across eight environments, which is this chapter's subject stated as an
empirical finding.

{{cite:huang2024selfcorrect}} for why the agent's own judgement of its progress is
the weakest available signal, and {{ch:rsn-tool-assisted}} for what to replace it
with.
