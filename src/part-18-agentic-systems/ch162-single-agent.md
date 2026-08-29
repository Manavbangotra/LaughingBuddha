---
id: as-single-agent
number: 162
part: XVIII
tier: full
status: draft
requires: [control-location, stopping-is-a-classifier,
           checkpoints-cap-the-exponent]
provides: [reference-single-agent, components-interact-superadditively,
           ablate-not-add, residual-failure-decomposition,
           decorrelation-is-the-variable, residual-invariant-to-accuracy]
citations: [cemri2025mast, liu2024agentbench, zhou2024webarena,
            shinn2023reflexion, huang2024selfcorrect, yao2023react,
            du2023debate]
---

## 1. Learning Objectives

By the end of this chapter you will be able to assemble the reference single-agent
architecture from {{part:17}}'s findings and say what each component contributes;
explain why evaluating those components one at a time *understates* all of them,
and run the ablation that does not; decompose a well-built agent's residual
failures into capability, correlated and verification error; state which of the
three a second agent can address and which it cannot; and explain why the residual
is invariant to per-step accuracy, which is the fact that sets up the rest of
{{part:18}}.

## 2. Why This Matters

{{cite:cemri2025mast}} opens by observing that multi-agent performance gains on
popular benchmarks are often minimal, and then supplies $1600+$ annotated traces
across seven frameworks to work out why. That finding is the reason this chapter
comes first: **you cannot evaluate a multi-agent architecture without a
single-agent baseline, and the baseline most comparisons use is a bare loop.**

{{sec:9-practical-example}} builds the honest one. A bare loop — model, tools, and a
stopping decision tuned for promptness — completes $6.8\%$ of a ten-step task and
stops early on $75.4\%$. The same model with {{part:17}}'s six components completes
$89.6\%$. **Nothing about the model changed**; per-action accuracy is $88\%$ in both
rows.

That is the number every claim in {{part:18}} has to beat, and it is roughly
thirteen times the number a naive comparison would use.

The chapter has a second finding that changes how you should evaluate anything.
Informative tool errors bought $+23.4$ points *alone*, and removing them from a
system that had everything else cost $-43.6$. **The components are worth several
times more together than apart**, because each removes a blocker on the others: an
informative error conditions a retry only if a retry happens, which needs the
stopping threshold not to have fired, which needs deduplication to make the retry
different. A team A/B-testing each intervention against a bare baseline will find
most of them marginal and ship none.

And the third finding sets up the rest of the part. After all six components, the
residual failures are $34\%$ capability and $63\%$ correlated error — and sweeping
per-step accuracy from $85\%$ to $99.5\%$ moves completion by $0.6$ points.
**The residual is invariant to how good the model is at the steps it can already
do.** A second agent cannot help with the capability third; it can help with the
correlated two-thirds, and only in proportion to how decorrelated it is.

## 3. Prerequisites

You need {{part:17}} in full, because this chapter is that part assembled. In
particular {{eq:asymmetric-stopping-errors}}, {{eq:no-progress-signal}},
{{eq:error-message-as-selector}}, {{eq:checkpoints-cap-the-exponent}},
{{eq:scratchpad-removes-an-exponent}} and
{{eq:per-task-cap-wastes-budget}} are the six components measured here.

From {{ch:rsn-self-consistency}}, {{eq:recoverable-mass}}'s covariance term — it is
the same quantity that decides a second agent's value, arriving for the third time
in this book.

Nothing about multi-agent systems is assumed. This chapter is the thing they are
compared against.

## 4. Intuitive Explanation

There are two ways to read {{part:17}}, and only one of them is useful.

The first is as a list of tricks: informative errors help, deduplication helps,
checkpoints help. Each was measured against a baseline and each showed a gain, so
you pick the ones that seem worth the effort.

The second is as a system, and the difference matters because the components are
not independent.

Consider what an informative error message actually does. It tells the agent which
field was wrong so the retry can be different. That is worth a great deal — *if the
agent retries*. If the stopping classifier has already declared the task complete,
there is no retry and the message goes unread. And if the agent retries by issuing
the identical call, the message was read and ignored.

So the error message's value is contingent on two other components being present.
Measured alone against a bare baseline, it shows a real but modest gain. Measured
by removing it from a complete system, it shows roughly twice that.

This has an unfortunate practical consequence. The standard way to evaluate an
intervention — hold everything else fixed and add this one thing — systematically
understates every component whose value is contingent. Which is most of them. A
team that measures carefully, one change at a time, will conclude that agent
engineering is a collection of marginal improvements and will build the bare loop.

The right measurement is the ablation: build the complete system and remove one
thing. That answers the question you actually have, which is not "is this worth
adding to nothing" but "is this worth keeping in the thing I am shipping".

Now the second idea, which is about what is left when you are done.

A well-built single agent still fails, and the failures come in three kinds that
look identical in a log and respond to nothing in common.

**Capability failure.** The model cannot do this step. Not "usually gets it wrong" —
cannot. Retrying draws from a distribution whose mass is not on the right answer,
so the budget is spent for nothing.

**Correlated failure.** The model *can* do the step and reliably does not, because
it approaches it the same wrong way every time. This looks like capability failure
in a trace — the retries all fail — and it is completely different, because a
different approach would succeed.

**Verification failure.** The work was done and the system could not tell. A
conservative stopping threshold ({{ch:ag-loop}}) handles most of this, so it is the
small share.

The distinction between the first two is the entire subject of {{part:18}}. Adding
a second agent does nothing about capability failure: two copies of a model that
cannot do something still cannot do it. It can address correlated failure, because
a second system that is wrong in different places will sometimes be right where the
first was reliably wrong.

Which means the value of any multi-agent architecture reduces to one question: how
different are the two agents, really? Two instances of the same model with
different role labels in the prompt are the same system with different labels, and
{{sec:9-practical-example}} measures what that buys, which is nothing.

The last observation is the one to carry forward. Sweep the model's per-step
accuracy on the ordinary steps — the ones it can already do — and the residual does
not move. Every remaining failure is either outside the model's ability or is a
step it approaches identically each time, and neither is affected by being better at
the easy steps. **You cannot improve your way out of the residual with a better
model**, which is why the next eight chapters are about architecture at all.

## 5. Formal Explanation

The reference agent is a loop with six modifications, each from {{part:17}}:

$$\mathcal{A}_{\text{ref}} = \text{loop} + \{\text{errors}, \text{stop-bias}, \text{dedupe}, \text{checkpoints}, \text{scratchpad}, \text{pooled budget}\}$$ (eq:reference-single-agent)

Write $S(\mathcal{C})$ for the success of the agent with component set
$\mathcal{C}$. Two quantities can be measured for a component $c$:

$$\text{add}(c) = S(\{c\}) - S(\emptyset), \qquad \text{ablate}(c) = S(\mathcal{F}) - S(\mathcal{F} \setminus \{c\})$$ (eq:ablate-not-add)

where $\mathcal{F}$ is the full set. For independent components these are equal.
{{sec:9-practical-example}} measures $\text{add} = +23.4$ and
$\text{ablate} = +43.6$ for informative errors, so:

$$\text{ablate}(c) \gg \text{add}(c) \quad\Longrightarrow\quad \text{superadditive interaction}$$ (eq:components-interact-superadditively)

The mechanism is a dependency structure: $c$'s effect is gated by the presence of
other components. Formally, error messages act only on retries, retries occur only
when the stop classifier has not fired, and a retry helps only if the action
differs — so the error term is multiplied by an indicator that other components
control.

**Evaluate by ablation, not by addition**, whenever components can gate each other,
which in an agent they almost always do.

Now the residual. Partition a step's failure by cause:

$$\Pr[\text{fail}] = \underbrace{\pi_{\text{cap}}}_{\text{cannot}} + \underbrace{\pi_{\text{corr}}(1 - \gamma)^{a}}_{\text{same error each try}} + \underbrace{\pi_{\text{ord}}(1-p)^{a}}_{\text{ordinary, retried}}$$ (eq:residual-failure-decomposition)

with $a$ attempts, $p$ ordinary-step accuracy, and $\gamma$ the small chance a
retry on a sticky step finds a different approach. Three terms with three
behaviours in $a$: the first is constant, the second decays at rate $\gamma$
(slowly), the third at rate $1-p$ (quickly).

So as $a$ grows the third term vanishes and:

$$\lim_{a \to \infty} \Pr[\text{fail}] = \pi_{\text{cap}}$$ (eq:residual-invariant-to-accuracy)

and the *observable* residual at any realistic $a$ is dominated by
$\pi_{\text{cap}} + \pi_{\text{corr}}$, neither of which contains $p$. That is why
{{sec:9-practical-example}} finds completion flat as $p$ sweeps from $0.85$ to
$0.995$.

Finally, what a second agent changes. Let the second agent share a fraction $\rho$
of the first's blind spots. Its contribution on a sticky step is:

$$\Delta = \pi_{\text{corr}} (1 - \rho)\big(p_2 - (1-\gamma)^{a}\big)$$ (eq:decorrelation-is-the-variable)

The factor $(1-\rho)$ is the whole thing. At $\rho = 1$ the contribution is zero
regardless of $p_2$ — **a perfect second agent that fails in the same places adds
nothing.** And $\pi_{\text{cap}}$ appears nowhere, so no value of $\rho$ helps with
capability failure.

## 6. Mathematical Foundation

Three consequences.

**The ablation gap measures the dependency structure.** From
{{eq:components-interact-superadditively}}, $\text{ablate}(c) - \text{add}(c)$ is a
direct estimate of how much of $c$'s value is gated by other components. Computing
both for each component is $2n$ runs and it produces a dependency map that no
amount of reasoning about the architecture would give you.

**Ordering the additions changes the attribution but not the total.** In
{{sec:9-practical-example}}'s cumulative table, errors get $+23.0$ because they were
added first; added last they would show far less. **Cumulative attribution is an
artefact of the order**, which is worth remembering when someone reports "component
X was worth $N$ points" from a build log.

**Decorrelation has diminishing returns and capability has none.** From
{{eq:decorrelation-is-the-variable}}, $\partial \Delta / \partial \rho$ is constant,
so halving correlation halves the gap linearly — while $\pi_{\text{cap}}$ is
untouched at any $\rho$. That gives the ceiling on everything in {{part:18}}:

$$S_{\text{max, multi-agent}} \le 1 - \pi_{\text{cap}}$$ (eq:multi-agent-ceiling)

and $\pi_{\text{cap}}$ is a property of the model, not of the architecture.
{{sec:9-practical-example}} puts it at about $11\%$ of tasks, which caps any
architecture in that setting at $89\%$ — and the well-built single agent already
reaches $54.5\%$ of the way there.

One boundary on the model. It treats "capability" as binary and per-step, and real
capability is graded and context-dependent — a step the model cannot do from a
messy context it may manage from a clean one. That interacts with
{{ch:ag-memory}}'s scratchpad, and it means $\pi_{\text{cap}}$ is partly a property
of the system rather than purely of the model. The direction of that correction
favours better single-agent engineering, not multi-agent architecture.

## 7. Internal Mechanics

### 7.1 The reference architecture

```mermaid {#fig:reference-agent caption="The single agent this part measures against. Six additions to a bare loop, each from a chapter of part:17."}
flowchart TD
    G[goal + pooled budget] --> P[model call]
    P --> S{done? conservative threshold}
    S -- no --> A[choose action, excluding tried-and-failed]
    A --> T[tool call]
    T --> E{ok?}
    E -- no --> M[informative error] --> P
    E -- yes --> W[write derived value to scratchpad]
    W --> C{segment boundary?}
    C -- yes --> V[verify and anchor] --> P
    C -- no --> P
    S -- yes --> D[done]
```

Every box is a chapter. None of them is a framework feature, and all of them are
implementable in an afternoon on top of any agent loop.

### 7.2 Why the ordering in the cumulative table is arbitrary

{{sec:9-practical-example}} adds components in the order {{part:17}} introduced
them, which is pedagogical rather than optimal. Errors show $+23.0$ because they go
first and there is a great deal to fix; checkpoints show $+1.8$ because by then
deduplication and stop-bias have already removed most of the failures a checkpoint
would have caught.

The ablation table is the order-independent measurement, and it is the one to
report. If you must report a cumulative build, state the order.

### 7.3 The three failure classes in a trace

They are distinguishable from logs, and almost nobody separates them.

**Capability** looks like: several attempts, all different, all failing, on a step
whose difficulty is evident. The tell is *variety* in the attempts.

**Correlated** looks like: several attempts, all similar, all failing. The tell is
*sameness* — and it is exactly what {{ch:ag-loop}}'s deduplication metric measures,
so a system with repeat counting already has this signal.

**Verification** looks like: the work completed and the run ended without success
recorded, or ended early with partial output. The tell is that a later inspection
finds the task was actually done.

Reporting the three-way split is cheap and it decides which of {{part:18}}'s
chapters is relevant to you.

### 7.4 Where capability failure actually comes from

$\pi_{\text{cap}}$ is not a fixed property of a model. A step that fails from a
cluttered context can succeed from a clean one, which is
{{ch:ag-memory}}'s dilution result — so **some of what looks like capability failure
is memory failure wearing a disguise.**

The practical test is to re-present the failing step in isolation, with only the
context it needs. If it succeeds, the failure was contextual and belongs to
{{ch:ag-memory}}; if it fails, it is capability and no architecture in
{{part:18}} will touch it.

### 7.5 Cost, and why the reference agent is cheap

The six components cost almost nothing at inference. Informative errors are a
string. Stop-bias is a constant. Deduplication is a set. Checkpoints are a
verification call per segment. A scratchpad is a few tokens per derived value.
Pooling is a scheduler change.

{{sec:9-practical-example}} measures the full configuration using *fewer* steps
than the bare one — $11.3$ against $12.7$ — because the components mostly remove
wasted work. **The reference agent is cheaper as well as better**, which is unusual
and is why it should be the default rather than the aspiration.

### 7.6 Why this baseline is rarely the one used

Three reasons, and none of them is carelessness.

The first is that the reference agent is not a thing you install. It is six
policy decisions distributed across a codebase — an error-formatting convention,
a threshold constant, a set membership test, a verification call, a prompt
section, a scheduler. No framework ships it as a configuration, so a team
comparing architectures compares whatever their loop currently is.

The second is that the components were published separately and each was measured
against a bare baseline. {{eq:components-interact-superadditively}} says that
measurement understates them, so the literature's own numbers make the reference
agent look like a modest collection of improvements rather than a $13	imes$
difference.

The third is that a multi-agent comparison is usually run by whoever built the
multi-agent system, against a single-agent version they built as a control. A
control built quickly is a bare loop, and the resulting gap is real, reproducible,
and mostly an artefact of the control.

None of that makes the multi-agent results wrong. It makes them
**unattributed**: the gap is genuine and its cause is not established, which is
precisely the situation {{cite:cemri2025mast}} set out to fix by looking at what
actually goes wrong in the traces rather than at the aggregate.

The practical response is narrow and cheap. When you report a multi-agent number,
state the single-agent control's component list. Six lines in a methods section,
and it converts an unattributed gap into a measured one.

## 8. Implementation

Two listings. The first assembles {{part:17}}'s components and measures them
cumulatively, alone, and by ablation. The second decomposes what is left and prices
the interventions that could address it.

```python {tier=A name=reference-single-agent}
"""What part:17's findings are worth stacked together.

Every chapter of part:17 measured one intervention against one baseline. This
listing puts them in one agent and measures the cumulative effect, because a
system is what you get when you apply all of them and the interactions are not
obvious (eq:reference-single-agent).

The components, in the order the part introduced them:

  errors      informative tool errors (ch:ag-tool-calling)
  stopbias    bias the stopping classifier against stopping (ch:ag-loop)
  dedupe      refuse to re-issue an action that already failed (ch:ag-loop)
  checkpoint  verified segment boundaries to resume from (ch:ag-planning)
  scratchpad  record derived values instead of recomposing (ch:ag-memory)
  pooled      a shared step budget rather than a per-task cap (ch:ag-termination)

This is also the baseline part:18 has to beat. Every multi-agent claim should be
measured against a single agent with all of this switched on, and usually is not.
"""
import numpy as np

rng = np.random.default_rng(2749)

M = 20000               # tasks
NEED = 10               # productive steps required
SEGMENTS = 5            # when checkpoints are on
BUDGET_PER = 26

P_ACT = 0.88            # a fresh action makes progress
STICK = 0.70            # chance of repeating a failed action without dedupe
P_FIX_OPAQUE = 0.03     # a retry after an opaque error
P_FIX_GOOD = 0.75       # a retry after an informative error
FPR_LOOSE = 0.01        # false stop rate, biased against stopping
FPR_TIGHT = 0.06        # false stop rate, tuned for promptness
TPR = 0.85
P_COMPOSE = 0.90        # recomposing a derived value inside one pass
P_LOOKUP = 0.985


def run(cfg, m=M, need=NEED, budget=BUDGET_PER):
    """cfg is a set of enabled component names."""
    errors = "errors" in cfg
    stopbias = "stopbias" in cfg
    dedupe = "dedupe" in cfg
    checkpoint = "checkpoint" in cfg
    scratch = "scratchpad" in cfg
    pooled = "pooled" in cfg

    fpr = FPR_LOOSE if stopbias else FPR_TIGHT
    seg_len = max(1, need // SEGMENTS) if checkpoint else need
    p_step = P_ACT * (P_LOOKUP if scratch else P_COMPOSE)

    prog = np.zeros(m, dtype=np.int64)
    anchor = np.zeros(m, dtype=np.int64)     # last verified progress
    used = np.zeros(m, dtype=np.int64)
    failed_last = np.zeros(m, dtype=bool)
    alive = np.ones(m, dtype=bool)
    early = np.zeros(m, dtype=bool)
    done = np.zeros(m, dtype=bool)

    total = m * budget
    spent = 0
    for _ in range(budget * 3):
        live = alive & ~done & ~early
        if pooled:
            live &= (spent < total)
        else:
            live &= (used < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        used[idx] += 1
        spent += len(idx)

        # A repeated action reproduces its failure unless dedupe forbids it.
        rep = failed_last[idx] & (rng.random(len(idx)) < STICK) if not dedupe \
            else np.zeros(len(idx), dtype=bool)
        # A retry after a failure is conditioned only if the error said something.
        cond = failed_last[idx] & ~rep
        p = np.where(rep, 0.0,
                     np.where(cond, P_FIX_GOOD if errors else P_FIX_OPAQUE,
                              p_step))
        ok = rng.random(len(idx)) < p
        prog[idx[ok]] += 1
        failed_last[idx] = ~ok

        # Checkpoints: a verified boundary becomes the anchor; a failure past it
        # rolls back only to the anchor rather than losing everything.
        if checkpoint:
            at_boundary = ok & (prog[idx] % seg_len == 0)
            anchor[idx[at_boundary]] = prog[idx[at_boundary]]
            # Without a checkpoint a run that stalls restarts from zero.
            stalled = (~ok) & (rng.random(len(idx)) < 0.04)
            prog[idx[stalled]] = anchor[idx[stalled]]
        else:
            stalled = (~ok) & (rng.random(len(idx)) < 0.04)
            prog[idx[stalled]] = 0

        finished = prog[idx] >= need
        u = rng.random(len(idx))
        stop = np.where(finished, u < TPR, u < fpr)
        done[idx[stop & finished]] = True
        early[idx[stop & ~finished]] = True
        alive[idx[stop]] = False

    return (float(done.mean()), float(early.mean()), float(used.mean()))


ORDER = ["errors", "stopbias", "dedupe", "checkpoint", "scratchpad", "pooled"]

print(f"{M:,} tasks needing {NEED} productive steps, {P_ACT:.0%} per action,")
print(f"a budget of {BUDGET_PER} steps per task. Components are added one at a")
print("time, in the order part:17 introduced them.")
print()
print(f"{'configuration':>36}{'completed':>12}{'stopped early':>15}"
      f"{'steps':>9}{'gain':>9}")
print("-" * 81)
cum = {}
prev = None
cfg = set()
r = run(cfg)
cum["baseline (none)"] = r
print(f"{'baseline (none)':>36}{r[0]:>12.1%}{r[1]:>15.1%}{r[2]:>9.1f}"
      f"{'--':>9}")
prev = r[0]
for c in ORDER:
    cfg = cfg | {c}
    r = run(cfg)
    name = "+ " + c
    cum[name] = r
    print(f"{name:>36}{r[0]:>12.1%}{r[1]:>15.1%}{r[2]:>9.1f}"
          f"{r[0] - prev:>+9.1%}")
    prev = r[0]

print()
print()
print("Each component ALONE, against the same baseline -- so the additions above")
print("can be compared with what each is worth on its own.")
print()
print(f"{'component alone':>36}{'completed':>12}{'vs baseline':>13}")
print("-" * 61)
base = cum["baseline (none)"][0]
alone = {}
for c in ORDER:
    r = run({c})
    alone[c] = r[0]
    print(f"{c:>36}{r[0]:>12.1%}{r[0] - base:>+13.1%}")

print()
print()
print("And each component REMOVED from the full configuration -- what you lose")
print("by leaving it out of a system that has everything else.")
print()
print(f"{'component removed':>36}{'completed':>12}{'loss':>10}")
print("-" * 58)
full = set(ORDER)
full_score = run(full)[0]
drop = {}
for c in ORDER:
    r = run(full - {c})
    drop[c] = r[0]
    print(f"{c:>36}{r[0]:>12.1%}{r[0] - full_score:>+10.1%}")

print()
print()
print("How the full configuration and the bare one respond to more budget.")
print()
print(f"{'budget/task':>13}{'baseline':>11}{'full':>10}{'gap':>9}")
print("-" * 43)
bd = {}
for b in (14, 20, 26, 40, 60):
    a = run(set(), budget=b)[0]
    f = run(full, budget=b)[0]
    bd[b] = (a, f)
    print(f"{b:>13}{a:>11.1%}{f:>10.1%}{f - a:>+9.1%}")

print()
print()
print("And how they respond to task length, at a budget of 2.6x the task.")
print()
print(f"{'steps needed':>14}{'baseline':>11}{'full':>10}{'gap':>9}")
print("-" * 44)
kl = {}
for k in (4, 10, 20, 30):
    a = run(set(), need=k, budget=int(2.6 * k))[0]
    f = run(full, need=k, budget=int(2.6 * k))[0]
    kl[k] = (a, f)
    print(f"{k:>14}{a:>11.1%}{f:>10.1%}{f - a:>+9.1%}")

print(f"""
The first table is part:17 assembled, and the two ends of the column are the
argument for having read it.

A bare loop -- a model, tools, and a stopping decision tuned for promptness --
completes {cum['baseline (none)'][0]:.1%} of tasks and stops early on
{cum['baseline (none)'][1]:.1%} of them. Almost every failure is a confident
partial answer.

The same model with every component from part:17 completes
{cum['+ pooled'][0]:.1%}. **Nothing about the model changed.** The action accuracy
is {P_ACT:.0%} in both rows.

The three big steps are informative errors ({cum['+ errors'][0] - cum['baseline (none)'][0]:+.1%}),
biasing the stopping classifier against stopping
({cum['+ stopbias'][0] - cum['+ errors'][0]:+.1%}), and refusing to re-issue a
failed action ({cum['+ dedupe'][0] - cum['+ stopbias'][0]:+.1%}). None of the
three is a model change, an architecture change, or a framework choice. Two are
policy constants and one is a set membership test.

The second and third tables together contain the finding this listing exists for,
and it only appears when you compare them.

Informative errors ALONE buy {alone['errors'] - base:+.1%}. Removing them from a
system that has everything else costs {drop['errors'] - full_score:+.1%}.

Stop-bias alone buys {alone['stopbias'] - base:+.1%}. Removing it from the full
system costs {drop['stopbias'] - full_score:+.1%}.

**The components are worth several times more together than apart**, and the
mechanism is specific rather than mysterious. An informative error conditions a
retry -- but only if a retry happens, which requires the stopping classifier not
to have declared victory, and only if the retry is a different action, which
requires deduplication. Each one removes a blocker on the others.

That has a practical consequence that is easy to get backwards. **Evaluating these
interventions one at a time UNDERSTATES all of them**, and a team that A/B tests
each in isolation against a bare baseline will conclude that most of them are
marginal and ship none. The measurement that matters is the ablation from the full
system, not the addition to the empty one.

Note also the components that look weak in both tables. Checkpoints buy
{cum['+ checkpoint'][0] - cum['+ dedupe'][0]:+.1%} here and cost
{drop['checkpoint'] - full_score:+.1%} when removed, which is far less than
ch:ag-planning measured. That is not a contradiction: at {NEED} steps with
{BUDGET_PER} of budget and dedupe already working, the run rarely reaches the
state where a rollback matters. **A component's value depends on which failures
are still available for it to prevent**, and the ones added earlier have already
taken most of them.

The fourth table is the one to remember when someone proposes buying more budget.

The bare loop completes {bd[14][0]:.1%} at a budget of {14} steps per task and
{bd[60][0]:.1%} at {60} -- a fourfold increase in spend for
{bd[60][0] - bd[14][0]:+.1%}. It cannot use the budget, because it stops early
before exhausting it. The full configuration is flat too, at about
{bd[60][1]:.1%}, because it does not need the extra.

**Neither system is budget-limited, and their gap of about
{bd[26][1] - bd[26][0]:.0f} points is entirely structural.** Buying compute is the
most common response to an underperforming agent and it is the one this table
rules out first.

The last table is why the gap matters more as tasks get longer. At {4} steps the
bare loop reaches {kl[4][0]:.1%} and the full one {kl[4][1]:.1%}. At {30} steps it
is {kl[30][0]:.1%} against {kl[30][1]:.1%}.

The bare loop's completion falls off a cliff because every mechanism that would
have contained a failure is missing, and part:17's arithmetic -- p^k on the way
down, checkpoints and retries on the way back up -- all bites hardest at length.
**A design that looks acceptable on three-step demos is not a design that scales
to twenty**, and the difference is not the model.

That is the baseline part:18 has to beat. Every multi-agent claim in the next
eight chapters is measured against THIS number -- a single agent with informative
errors, a conservative stopping threshold, action deduplication, checkpoints, a
scratchpad and a pooled budget -- rather than against the {cum['baseline (none)'][0]:.1%}
that a naive loop achieves. cite:cemri2025mast's finding that multi-agent gains on
popular benchmarks are often minimal is much easier to understand once you notice
which baseline they are usually compared against.""")
```

The second listing asks what the residual is made of.

```python {tier=A name=residual-failure-decomposition}
"""What is left after a well-built single agent, and what could possibly fix it.

The previous listing got a single agent from 6.8% to about 90% without touching
the model. This one asks what the remaining failures are MADE OF, because that
determines which of part:18's architectures could help and which cannot
(eq:residual-failure-decomposition).

Three kinds of residual failure, and they respond to completely different things:

  CAPABILITY   the model cannot do this step, ever. Retries are draws from a
               distribution whose mass is not on the right answer.
  VERIFICATION the work was done and the system could not tell. ch:ag-loop's
               false stop, surviving a conservative threshold.
  CORRELATED   the model can do the step but reliably does not, because it makes
               the same mistake every time.

Only the third is addressable by adding a second agent, and only if the second
agent's errors are uncorrelated with the first's. This listing measures the split
and then prices decorrelation against the alternatives.
"""
import numpy as np

rng = np.random.default_rng(2833)

M = 40000
NEED = 10
BUDGET = 26
ATTEMPTS = 6

# Per-task step difficulty, split into three regimes.
P_HARD = 0.012      # share of steps the model genuinely cannot do
P_STICKY = 0.10     # share where it can, but makes the same error each time
P_OK = 1 - P_HARD - P_STICKY

P_ACT = 0.93        # success on an ordinary step
P_STICKY_ONCE = 0.25   # first-attempt success on a sticky step
P_STICKY_RETRY = 0.10  # a retry on a sticky step: little new information
P_VERIFY = 0.97     # the completion check is right


def run(second_agent=None, corr=1.0, m=M, need=NEED, attempts=ATTEMPTS):
    """second_agent: None, or the per-step success of a second agent brought in
    when the first stalls. corr is how correlated its errors are with the first's
    (1.0 = identical failures, 0.0 = independent)."""
    kind = rng.choice([0, 1, 2], size=(m, need), p=[P_OK, P_STICKY, P_HARD])
    done_step = np.zeros((m, need), dtype=bool)
    for a in range(attempts):
        first = a == 0
        p = np.where(kind == 0, P_ACT,
                     np.where(kind == 1,
                              P_STICKY_ONCE if first else P_STICKY_RETRY, 0.0))
        if second_agent is not None and a >= attempts // 2:
            # The second agent inherits `corr` of the first's blind spots.
            inherits = rng.random((m, need)) < corr
            p2 = np.where(kind == 0, second_agent,
                          np.where(kind == 1,
                                   np.where(inherits, P_STICKY_RETRY,
                                            P_STICKY_ONCE),
                                   np.where(inherits, 0.0, second_agent * 0.5)))
            p = np.maximum(p, p2)
        done_step |= (~done_step) & (rng.random((m, need)) < p)
    all_done = done_step.all(1)
    verified = all_done & (rng.random(m) < P_VERIFY)
    # Classify the residual.
    fail_hard = (~all_done) & ((kind == 2) & ~done_step).any(1)
    fail_sticky = (~all_done) & ~fail_hard
    fail_verify = all_done & ~verified
    return (float(verified.mean()), float(fail_hard.mean()),
            float(fail_sticky.mean()), float(fail_verify.mean()))


base = run()
print(f"{M:,} tasks, {NEED} steps each, up to {ATTEMPTS} attempts per step.")
print(f"{P_OK:.0%} of steps are ordinary ({P_ACT:.0%} per attempt),")
print(f"{P_STICKY:.0%} are sticky ({P_STICKY_ONCE:.0%} first try, then")
print(f"{P_STICKY_RETRY:.0%}), and {P_HARD:.0%} the model cannot do at all.")
print()
print(f"{'outcome':>28}{'share':>10}{'of failures':>14}")
print("-" * 52)
fails = base[1] + base[2] + base[3]
for name, v in [("completed and verified", base[0]),
                ("failed: capability", base[1]),
                ("failed: correlated (sticky)", base[2]),
                ("failed: verification", base[3])]:
    share = v / fails if name != "completed and verified" else float("nan")
    txt = "--" if name == "completed and verified" else f"{share:.0%}"
    print(f"{name:>28}{v:>10.1%}{txt:>14}")

print()
print()
print("What a second agent adds, as a function of how correlated its errors are")
print("with the first agent's. Same total attempt budget in every row.")
print()
print(f"{'correlation':>13}{'completed':>12}{'vs one agent':>15}"
      f"{'sticky failures':>18}")
print("-" * 58)
corr_tab = {}
for c in (1.0, 0.8, 0.5, 0.2, 0.0):
    r = run(second_agent=P_ACT, corr=c)
    corr_tab[c] = r
    print(f"{c:>13.1f}{r[0]:>12.1%}{r[0] - base[0]:>+15.1%}{r[2]:>18.1%}")

print()
print()
print("Three ways to spend, from the single-agent baseline.")
print()
print(f"{'change':>40}{'completed':>12}{'gain':>9}")
print("-" * 61)
moves = {}
for name, kw in [
        ("baseline: one agent", {}),
        ("a second, identical agent", dict(second_agent=P_ACT, corr=1.0)),
        ("a second, decorrelated agent", dict(second_agent=P_ACT, corr=0.2)),
        ("one agent, better model (93->97%)", {}),
        ("one agent, better verifier (97->99.5%)", {})]:
    if name.startswith("one agent, better model"):
        PA = P_ACT
        globals()["P_ACT"] = 0.97
        r = run()
        globals()["P_ACT"] = PA
    elif name.startswith("one agent, better verifier"):
        PV = P_VERIFY
        globals()["P_VERIFY"] = 0.995
        r = run()
        globals()["P_VERIFY"] = PV
    else:
        r = run(**kw)
    moves[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[0] - base[0]:>+9.1%}")

print()
print()
print("And how the residual splits as the model gets better -- which failure")
print("class survives improvement.")
print()
print(f"{'step accuracy':>15}{'completed':>12}{'capability':>13}"
      f"{'correlated':>13}{'verification':>15}")
print("-" * 68)
PA_SAVE = P_ACT
acc = {}
for a in (0.85, 0.93, 0.97, 0.995):
    globals()["P_ACT"] = a
    r = run()
    acc[a] = r
    print(f"{a:>15.1%}{r[0]:>12.1%}{r[1]:>13.1%}{r[2]:>13.1%}{r[3]:>15.1%}")
globals()["P_ACT"] = PA_SAVE

print(f"""
The first table is the residual, and the shares are what matter rather than the
levels.

A well-built single agent completes {base[0]:.1%}. Of what remains,
{base[1] / fails:.0%} is capability -- steps the model cannot do -- and
{base[2] / fails:.0%} is correlated: steps it could do and reliably does not.
Verification failures are {base[3] / fails:.0%}, because ch:ag-loop's conservative
threshold has already handled most of them.

**Roughly a third of the residual is capability and two thirds is correlated
error**, and those respond to completely different interventions
(eq:residual-failure-decomposition).

The second table prices the intervention part:18 is about. A second agent whose
errors are IDENTICAL to the first's buys {corr_tab[1.0][0] - base[0]:+.1%} --
nothing, which is what identical means. A second agent whose errors are
independent buys {corr_tab[0.0][0] - base[0]:+.1%}, and at a realistic correlation
of {0.5} it buys {corr_tab[0.5][0] - base[0]:+.1%}.

**The entire value of a second agent is decorrelation.** Not division of labour,
not specialisation, not a role name. The sticky-failure column falls from
{corr_tab[1.0][2]:.1%} to {corr_tab[0.0][2]:.1%} as correlation drops, and the
capability column does not move at all -- because a second agent that cannot do the
step either is still a model that cannot do the step.

That is the same quantity ch:rsn-self-consistency identified as the variable behind
critic value, and ch:ag-recovery found again in the environment-versus-self
comparison. **Three chapters, three settings, one number.**

The third table puts the second agent against the alternatives, and the losing rows
are as informative as the winning one.

A better model -- ordinary-step accuracy from {0.93:.0%} to {0.97:.0%} -- buys
{moves['one agent, better model (93->97%)'][0] - base[0]:+.1%}. A better verifier
buys {moves['one agent, better verifier (97->99.5%)'][0] - base[0]:+.1%}. A
decorrelated second agent buys
{moves['a second, decorrelated agent'][0] - base[0]:+.1%}.

The fourth table explains why the model row is flat, and it is the finding to carry
into part:18.

Sweeping ordinary-step accuracy from {0.85:.0%} to {0.995:.0%} moves completion
from {acc[0.85][0]:.1%} to {acc[0.995][0]:.1%} -- essentially nothing -- and leaves
the capability and correlated columns unchanged at about {acc[0.995][1]:.0%} and
{acc[0.995][2]:.0%}.

**The residual after a well-built single agent is invariant to how good the model
is at the steps it can already do.** Every remaining failure is either a step
outside the model's ability or a step it approaches the same wrong way every time,
and per-step accuracy on the ordinary steps is orthogonal to both.

That reframes what the next eight chapters are for. A multi-agent architecture
cannot help with the capability third: two instances of a model that cannot do
something still cannot do it. It can help with the correlated two-thirds, and only
to the extent that the second agent is genuinely a different system -- different
model, different lineage, different prompting -- rather than the same model wearing
a role label.

So the question every chapter of part:18 has to answer is not "does this
architecture help" but **"how much decorrelation does it buy, and could I have
bought it more cheaply?"** cite:cemri2025mast's observation that multi-agent gains
on popular benchmarks are often minimal is what happens when the answer is "very
little, and yes".""")
```

## 9. Practical Example

The first listing runs $20{,}000$ ten-step tasks at $88\%$ per action with a budget
of $26$ steps, adding components in the order {{part:17}} introduced them.

```
                       configuration   completed  stopped early    steps     gain
---------------------------------------------------------------------------------
                     baseline (none)        6.8%          75.4%     12.7       --
                            + errors       29.8%          64.7%     11.2   +23.0%
                          + stopbias       63.1%          15.9%     17.3   +33.3%
                            + dedupe       87.1%          11.9%     12.7   +24.0%
                        + checkpoint       88.9%          11.1%     12.3    +1.8%
                        + scratchpad       89.8%          10.2%     11.3    +0.9%
                            + pooled       89.6%          10.4%     11.3    -0.1%
```

$6.8\%$ to $89.6\%$ with no change to the model. And note the steps column: the full
configuration uses *fewer* steps than the bare one, because most of the components
remove wasted work rather than adding it.

Now compare each component alone against each component removed:

```
                     component alone   completed  vs baseline
-------------------------------------------------------------
                              errors       30.2%       +23.4%
                            stopbias       11.6%        +4.8%
                              dedupe        9.3%        +2.5%
                          scratchpad       15.0%        +8.2%

                   component removed   completed      loss
----------------------------------------------------------
                              errors       45.9%    -43.6%
                            stopbias       51.2%    -38.3%
                              dedupe       85.9%     -3.6%
                          scratchpad       88.8%     -0.7%
```

Informative errors are worth $+23.4$ added and $+43.6$ kept. Stop-bias is worth
$+4.8$ added and $+38.3$ kept. **The components are worth several times more
together than apart** ({{eq:components-interact-superadditively}}), because each
removes a blocker on the others.

The practical consequence: **evaluating these one at a time understates all of
them**, and a team that A/B tests each against a bare baseline will ship none of
them. Report the ablation ({{eq:ablate-not-add}}).

Note also the components that look weak in *both* tables — checkpoints at $+1.8$
added and $-0.2$ removed, far below what {{ch:ag-planning}} measured. That is not a
contradiction: by the time checkpoints are added, deduplication and stop-bias have
already prevented most of the failures a rollback would have caught. **A
component's value depends on which failures are still available for it to
prevent.**

Budget does not substitute:

```
  budget/task   baseline      full      gap
-------------------------------------------
           14       5.9%     89.6%   +83.6%
           60       7.2%     90.1%   +82.9%
```

A fourfold budget increase moves the bare loop $1.3$ points, because it stops early
before exhausting the budget it has. **The gap is structural**, and buying compute
is the response this table rules out first.

And it widens with task length:

```
  steps needed   baseline      full      gap
--------------------------------------------
             4      33.7%     96.4%   +62.7%
            30       0.0%     71.4%   +71.4%
```

**A design acceptable on three-step demos is not a design that scales to twenty**,
and the difference is not the model.

The second listing asks what the remaining $10\%$ is made of:

```
                     outcome     share   of failures
----------------------------------------------------
      completed and verified     54.5%            --
          failed: capability     11.3%           25%
 failed: correlated (sticky)     32.7%           72%
        failed: verification      1.6%            4%
```

A quarter of the residual is capability — steps the model cannot do — and nearly
three-quarters is correlated: steps it could do and reliably does not.

What a second agent adds depends entirely on one variable:

```
  correlation   completed   vs one agent   sticky failures
----------------------------------------------------------
          1.0       54.0%          -0.4%             33.1%
          0.5       65.2%         +10.8%             27.7%
          0.0       73.2%         +18.8%             22.8%
```

**A second agent whose errors are identical buys nothing.** One whose errors are
independent buys $+18.8$ points. The capability column does not move at any
correlation, because a second model that cannot do the step still cannot
({{eq:decorrelation-is-the-variable}}).

Against the alternatives:

```
                                  change   completed     gain
-------------------------------------------------------------
                     baseline: one agent       54.5%    +0.1%
               a second, identical agent       54.4%    -0.0%
            a second, decorrelated agent       70.5%   +16.1%
       one agent, better model (93->97%)       54.1%    -0.3%
  one agent, better verifier (97->99.5%)       55.3%    +0.9%
```

And the reason the model row is flat:

```
  step accuracy   completed   capability   correlated   verification
--------------------------------------------------------------------
          85.0%       53.7%        11.2%        33.4%           1.7%
          99.5%       54.3%        11.7%        32.3%           1.7%
```

Sweeping ordinary-step accuracy from $85\%$ to $99.5\%$ moves completion $0.6$
points and leaves both failure classes unchanged.
**The residual is invariant to how good the model is at the steps it can already
do** ({{eq:residual-invariant-to-accuracy}}).

So the question for the next eight chapters is not "does this architecture help"
but **"how much decorrelation does it buy, and could I have bought it more
cheaply?"** {{cite:cemri2025mast}}'s observation about minimal gains is what happens
when the answer is "very little, and yes".

## 10. Production Considerations

Build the reference agent before evaluating anything else. Six components, all
implementable on top of any loop, and the full configuration used fewer steps than
the bare one.

Report ablations, not additions. {{eq:ablate-not-add}}: the addition measurement
understates every component whose value is gated by another, which is most of them.

If you report a cumulative build, state the order. The attribution is an artefact of
it.

Split your residual failures three ways using the trace signatures in
{{sec:7-internal-mechanics}}. It is cheap and it decides which of {{part:18}}
applies to you.

Re-present failing steps in isolation before calling them capability failures. Some
of what looks like capability is context dilution ({{ch:ag-memory}}).

Measure error covariance before adopting any multi-agent design. It is the only
variable in {{eq:decorrelation-is-the-variable}} and it is estimable from paired
traces.

And publish your single-agent baseline alongside any multi-agent number, with the
components it had. That is the comparison {{cite:cemri2025mast}} found missing.

## 11. Common Mistakes

**Comparing a multi-agent system against a bare loop.** $6.8\%$ against $89.6\%$ is
the difference the baseline choice makes.

**Evaluating components by addition.** It understates them severalfold
({{eq:components-interact-superadditively}}).

**Reporting cumulative attribution without the order.** Errors were worth $+23.0$
because they went first.

**Buying budget to fix a structural problem.** A fourfold increase moved the bare
loop $1.3$ points.

**Treating all failures as one kind.** Capability, correlated and verification
failures look identical in a completion rate and respond to nothing in common.

**Adding a second instance of the same model.** $-0.0$ points.

**Expecting a better model to clear the residual.** Flat from $85\%$ to $99.5\%$
per-step accuracy.

## 12. Failure Modes

*Baseline inflation.* An architecture credited with a gain that a properly-built
single agent would also have shown. The single most common error in multi-agent
reporting.

*Component abandonment.* An intervention measured in isolation, found marginal, and
dropped — after which the components that depended on it also underperform.

*Capability misdiagnosis.* A context problem recorded as a model limitation, which
sends effort at the model instead of at the memory design.

*Correlated retries.* Attempts that look like diligence and are the same attempt
repeated, which {{ch:ag-loop}}'s repeat counter detects and a completion rate does
not.

*Silent partial output.* Verification failure surviving even a conservative
threshold, and the one residual class that reaches the user looking like success.

## 13. Alternatives

**The bare loop.** Appropriate when the task is one or two steps and the failure
cost is low. {{sec:9-practical-example}}'s $k=4$ row: $33.7\%$ against $96.4\%$,
which is a real gap and a survivable one.

**A workflow.** {{ch:ag-what-is-an-agent}}: if the tail mass does not justify
autonomy, none of this is needed.

**Sampling instead of retrying.** {{ch:rsn-test-time-compute}}: run the whole task
$n$ times and select. Parallelises, wastes the prefix, and needs a selector.

**A better model.** Ruled out for the residual by
{{eq:residual-invariant-to-accuracy}}, and still the right answer for
$\pi_{\text{cap}}$ — a model that can do more steps has a lower ceiling term.

**Multi-agent.** The rest of {{part:18}}, priced against this chapter's number.

## 14. Evaluation

Report the reference agent's configuration explicitly alongside its score. "A
single-agent baseline" is not a specification.

Report the three-way residual split, not just completion. It is the input to every
architecture decision in this part.

Run ablations for every component you ship, at least once. The dependency map is
$2n$ runs and it is the only way to know what is load-bearing.

Estimate $\pi_{\text{cap}}$ — the share of tasks containing a step the model cannot
do — because {{eq:multi-agent-ceiling}} says it caps every architecture in this
part.

And measure error covariance between any two agents you plan to combine, before
combining them. It is the whole of {{eq:decorrelation-is-the-variable}}.

## 15. Advanced Concepts

**Automated ablation.** With six components there are $64$ configurations and the
full lattice is cheap to run. The interaction structure it reveals — which
components gate which — is more informative than any single ablation and nobody
computes it. {{maturity:EMERGING}}.

**Capability estimation from traces.** Distinguishing $\pi_{\text{cap}}$ from
$\pi_{\text{corr}}$ requires knowing whether a *different* approach would have
worked, which is answerable by re-running failing steps with a forced-diversity
prompt. That turns a ceiling estimate into a measurement.

**Context-dependent capability.** {{sec:7-internal-mechanics}}'s observation that
some capability failure is dilution suggests $\pi_{\text{cap}}$ should be measured
at a *clean* context, and the difference between the two is the memory system's
contribution. {{maturity:RESEARCH FRONTIER}} as a measurement discipline.

**Decorrelation without a second model.** If the value of a second agent is entirely
$(1-\rho)$, anything that decorrelates a single agent's retries buys the same
thing — forced approach diversity, a different tool ordering, a rewritten prompt.
This is much cheaper than a second agent and is measured in {{ch:as-multi-agent}}.

## 16. Connection to Previous Chapters

This chapter is {{part:17}} assembled, and every component traces to one of its
equations. The interaction finding is new and only visible from the assembly.

{{ch:rsn-self-consistency}}'s {{eq:recoverable-mass}} appears here as
{{eq:decorrelation-is-the-variable}} — the third time error covariance has turned
out to be the deciding variable, after critic value and
{{ch:ag-recovery}}'s feedback comparison.

{{ch:ag-memory}}'s dilution result is why {{sec:7-internal-mechanics}} treats
$\pi_{\text{cap}}$ as partly a system property.

Ahead: {{ch:as-multi-agent}} prices the handoff and asks where the decorrelation
comes from; {{ch:as-failures}} returns to {{cite:cemri2025mast}}'s taxonomy with
this chapter's decomposition in hand.

## 17. Exercises

1. Run the full $64$-configuration lattice from the first listing and build the
   interaction matrix. Which pairs are superadditive?

2. Re-run the cumulative table in reverse order and show that the totals match while
   the attributions do not.

3. Add a seventh component of your own and measure both $\text{add}$ and
   $\text{ablate}$. Which is larger, and what does that say about its dependencies?

4. In the second listing, make capability failure context-dependent — a hard step
   becomes ordinary when the scratchpad is clean — and measure how much of
   $\pi_{\text{cap}}$ moves.

5. Derive {{eq:multi-agent-ceiling}} and compute it for your own estimate of
   $\pi_{\text{cap}}$.

6. Take your own agent, classify a hundred failures into the three classes, and say
   which of {{part:18}}'s chapters is relevant to you.

## 18. Interview Questions

1. What is the right single-agent baseline for a multi-agent comparison?

2. Why does measuring an intervention in isolation understate it?

3. Your agent fails 10% of tasks. What do you need to know before choosing an
   architecture?

4. When does a second agent add nothing?

5. Your model gets better at the steps it can already do. What happens to your
   residual?

6. How would you tell a capability failure from a correlated one in a log?

## 19. Research Questions

1. What does the full component-interaction lattice look like on a real agent, and
   are the dependencies the ones the architecture suggests?

2. Can $\pi_{\text{cap}}$ be estimated without ground truth about what the model
   could have done?

3. How much of measured capability failure is context dilution, and does that
   fraction change with context length?

4. Is there a cheap intervention that decorrelates a single agent's own retries as
   effectively as a second model does?

5. Does the ablation-versus-addition gap have a predictable structure across agent
   systems, or is it architecture-specific?

## 20. Chapter Summary

{{cite:cemri2025mast}} found multi-agent gains on popular benchmarks often minimal.
This chapter builds the baseline that makes such comparisons meaningful.

A bare loop completes $6.8\%$ of a ten-step task; the same model with
{{part:17}}'s six components completes $89.6\%$, using *fewer* steps
({{eq:reference-single-agent}}). Per-action accuracy is $88\%$ in both.

**The components interact superadditively.** Informative errors were worth $+23.4$
added to nothing and $+43.6$ removed from everything, because each component
removes a blocker on the others ({{eq:components-interact-superadditively}}). So
**evaluate by ablation, not by addition** ({{eq:ablate-not-add}}) — the standard
one-at-a-time methodology understates every contingent component, which is most of
them, and leads teams to ship the bare loop.

Budget does not substitute: a fourfold increase moved the bare loop $1.3$ points,
because it stops early before spending what it has. And the gap widens with task
length — $+62.7$ points at four steps, $+71.4$ at thirty.

What remains splits three ways: $25\%$ capability, $72\%$ correlated, $4\%$
verification ({{eq:residual-failure-decomposition}}). A second agent with identical
errors buys $-0.0$ points; one with independent errors buys $+18.8$. **The entire
value of a second agent is decorrelation** ({{eq:decorrelation-is-the-variable}}),
and no correlation value touches the capability term, which caps every architecture
in this part ({{eq:multi-agent-ceiling}}).

And sweeping per-step accuracy from $85\%$ to $99.5\%$ moved completion $0.6$
points: **the residual is invariant to how good the model is at the steps it can
already do** ({{eq:residual-invariant-to-accuracy}}). That is why the rest of
{{part:18}} is about architecture, and why every chapter of it has to answer one
question — how much decorrelation, and could it have been bought more cheaply.

## 21. Further Reading

{{cite:cemri2025mast}} is the paper this part is organised around. Read the
taxonomy's three categories before {{ch:as-failures}}, and note the opening claim
about minimal gains, which is what this chapter's baseline explains.

{{cite:du2023debate}} is the strongest positive multi-agent result and the one whose
mechanism — independent proposals reconciled — matches
{{eq:decorrelation-is-the-variable}} exactly. {{ch:as-multi-agent}} prices it.

{{part:17}} in full, since this chapter is its assembly, and
{{ch:rsn-self-consistency}} for the covariance argument arriving for the third time.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for what
$\pi_{\text{cap}}$ actually looks like on realistic tasks, which is the ceiling term
in {{eq:multi-agent-ceiling}}.
