---
id: as-state-machines
number: 166
part: XVIII
tier: full
status: draft
requires: [checkpoints-cap-the-exponent, graph-bounds-the-paths,
           context-change-breaks-loops]
provides: [replay-needs-idempotence, dedup-key-is-the-fix,
           state-must-be-sufficient, tried-set-is-the-missing-field,
           checkpoint-frequency-trade, durability-degrades-with-crash-rate]
citations: [cemri2025mast, liu2024agentbench, greshake2023indirect,
            shinn2023reflexion, zhou2024webarena, yao2023react]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what an at-least-once
execution guarantee actually guarantees, and why that is a statement about side
effects; give every non-idempotent step a deduplication key derived correctly;
choose a checkpoint frequency from the replay-window trade rather than by default;
list what a durable state must contain for a *resume* to be correct rather than
merely to restart; and explain why the field that matters most is the one no
workflow engine persists.

## 2. Why This Matters

{{ch:ag-planning}} showed checkpoints are the largest lever available to an agent.
Durable execution is that mechanism with the state written to storage, so a restart
can follow a process crash rather than only a logical failure. The frameworks call
this at-least-once execution and treat it as a solved problem.

It is not, for a reason {{sec:9-practical-example}} makes concrete. **An
at-least-once execution guarantee is an at-least-once side effect guarantee.** The
engine replays the step; whether replaying is harmless is a property of the step,
and the engine cannot tell. With no idempotent steps, $33.5\%$ of runs end with a
duplicated effect.

The fix that works is not more checkpoints — though those help, taking corruption
from $40.2\%$ to $0.0\%$ as the checkpoint interval falls from twelve steps to one.
It is a **deduplication key**: the engine records that a particular effect already
happened and suppresses the repeat. That takes every idempotence level to $100\%$,
including zero.

The second half is about what the durable state must *contain*, and it produces the
chapter's most useful finding. Ablating fields from a complete state, the largest
loss by far is the **tried set** — {{ch:ag-loop}}'s record of which actions already
failed — at $-38.6$ points. Position costs $-3.3$.

**The fields workflow engines persist by default are not the ones that decide
whether a resume produces the right answer.** Position is what the *engine* needs to
know where to restart. The tried set, the derived values and the verbatim goal are
what the *agent* needs in order to be the same agent it was.

## 3. Prerequisites

You need {{ch:ag-planning}}'s checkpoints, because durable execution is that
mechanism plus persistence, and its budget-cliff warning applies here too.

From {{ch:ag-loop}}, the deduplication set and
{{eq:context-change-breaks-loops}} — a resume that has lost the failure record
recreates the repetition failure with a clean context that makes the wrong approach
look new.

From {{ch:ag-memory}}, the scratchpad: derived values and the verbatim goal are its
mechanisms, and durability is what makes them survive a restart.

From {{ch:as-graph}}, the static-graph idea: a state machine is a graph whose edges
are checkable predicates, so $p_e = 1$ and the path bound holds without the branch
penalty.

## 4. Intuitive Explanation

A long-running agent will be interrupted. The process is deployed over, the machine
is preempted, the network drops. Durable execution is the answer: write the state
somewhere, and on restart pick up where you left off.

The engines that do this make a guarantee, and the guarantee is usually
*at-least-once*: every step will execute at least one time. That is the honest thing
to promise, because exactly-once is impossible in a distributed system for reasons
older than any of this.

Here is what at-least-once means when the step has a side effect. The step sends an
email. The engine records that the step ran. The machine dies between those two
operations. On restart the engine sees no record, replays the step, and sends the
email again.

Nothing malfunctioned. The guarantee was honoured. The customer got two emails.

So the correctness of a durable agent depends entirely on a property of its steps
that nobody writes down: **what happens if this runs twice?** A read is fine. A
"set status to approved" is fine. A "send", "charge", "append", or "increment" is
not.

Two responses, and they are not equivalent.

The first is to checkpoint more often, so less gets replayed. This works and it
shrinks the problem rather than removing it: the window between the last durable
point and the crash is the window that replays, and making it one step means at most
one duplicate. It costs a durable write per checkpoint.

The second is a deduplication key. Before the effect, the engine writes a record
keyed by the run and the step; the effect handler checks the key and refuses to act
twice. That eliminates the duplicate rather than narrowing it, and
{{sec:9-practical-example}} measures it taking corruption to zero at every level of
natural idempotence.

The key has to be derived from the run and the step, not generated fresh — a fresh
key on replay is a different key, and the whole mechanism turns off. That is the
most common way to implement this wrong.

Now the second question, which is what the durable state has to hold.

The obvious answer is position: which step we reached. Every engine stores this,
because it is what the engine needs to know where to resume.

But consider what the *agent* needs. It had a set of approaches it had already tried
and eliminated. It had values it derived from several inputs. It had the user's
original request, which it may since have summarised. A resume that restores only
the position gives you an agent at step seven with no memory of why it is not at
step four — and it will happily re-derive the approach the crashed run had already
ruled out.

{{sec:9-practical-example}} measures that. Omitting the tried set from an otherwise
complete durable state costs $38.6$ points, an order of magnitude more than
omitting the position.

Which is worth sitting with, because it is the field no workflow engine persists.
Engines store what engines need. The agent's working knowledge is application state,
and if you do not write it down deliberately, a resume produces a fresh agent
wearing the old one's position.

## 5. Formal Explanation

Let a workflow have $n$ steps, crash probability $c$ after each, and a durable
checkpoint every $\kappa$ steps. On a crash, the run resumes from the last
checkpoint, so the expected number of steps replayed is:

$$\mathbb{E}[\text{replayed} \mid \text{crash}] = \frac{\kappa - 1}{2}$$ (eq:checkpoint-frequency-trade)

increasing in $\kappa$. Each replayed step with a side effect duplicates it unless
the step is idempotent, so with an idempotent fraction $\iota$:

$$\Pr[\text{corrupt}] \approx 1 - \big(1 - c\,(1-\iota)\big)^{\,n \cdot \frac{\kappa-1}{2\kappa}}$$ (eq:replay-needs-idempotence)

Two levers appear, and they act differently. Raising $\iota$ removes the harm;
lowering $\kappa$ reduces the exposure. The first is a property of the tools and the
second is a configuration.

A deduplication key changes the first. If the engine writes a key before the effect
and the handler is keyed, the replay is suppressed and the *effective* idempotent
fraction becomes one:

$$\iota_{\text{eff}} = 1 \quad\Longrightarrow\quad \Pr[\text{corrupt}] = 0$$ (eq:dedup-key-is-the-fix)

independent of $\kappa$ and of $c$. **That is why keys dominate checkpoint
frequency**: they remove a term rather than shrink it.

The key must satisfy one property, and it is where implementations fail:

$$k = f(\text{run id}, \text{step id}) \quad\text{— deterministic across replays}$$ (eq:key-must-be-deterministic)

A key generated at effect time — a UUID, a timestamp — differs on replay and
suppresses nothing.

Now sufficiency of state. Write the agent's working state as a tuple and let a
resume reconstruct it from what was persisted:

$$\sigma = \langle \text{pos}, \text{outputs}, \text{tried}, \text{derived}, \text{goal} \rangle$$ (eq:state-must-be-sufficient)

A resume is *correct* if the reconstructed $\hat\sigma$ leads to the same outcome
distribution as $\sigma$ would have. Each omitted field degrades that differently:

- omitting **pos** costs re-execution, which is expensive and not incorrect;
- omitting **outputs** costs re-execution of completed work;
- omitting **tried** costs re-exploration of eliminated approaches, which is
  {{eq:context-change-breaks-loops}} in reverse — the context after the resume no
  longer differs from the context before the failure;
- omitting **derived** costs recomposition, at {{ch:ag-memory}}'s
  {{eq:scratchpad-removes-an-exponent}} rate;
- omitting **goal** costs drift toward a paraphrase.

Only the first two are re-execution costs. **The last three are correctness costs**,
and they are the three no engine stores.

$$\text{engine state} = \langle \text{pos}, \text{outputs}\rangle, \qquad \text{agent state} = \langle \text{tried}, \text{derived}, \text{goal}\rangle$$ (eq:tried-set-is-the-missing-field)

Finally, the sensitivity to the environment. Every quantity above is multiplied by
$c$, so:

$$\frac{\partial}{\partial c}\big(S_{\text{full}} - S_{\text{partial}}\big) > 0$$ (eq:durability-degrades-with-crash-rate)

**The gap between a good durable design and a poor one widens with the crash
rate** — which means a design validated in a stable environment is untested, and
fails exactly when durability was supposed to help.

## 6. Mathematical Foundation

Three extractions.

**Keys and checkpoints are not substitutes.** From
{{eq:replay-needs-idempotence}} and {{eq:dedup-key-is-the-fix}}, checkpoint
frequency scales the exposure and a key eliminates the harm.
{{sec:9-practical-example}} measures keys worth $+33.7$ points at zero natural
idempotence and checkpointing every step worth $+40.2$ — comparable, but the key
holds at every crash rate while the checkpoint's benefit erodes. And a key costs one
write per effect where per-step checkpointing costs one write per step.

**The right checkpoint interval follows from the write cost.** Minimising
$\kappa$-dependent replay cost plus $n/\kappa$ write cost gives an interior optimum
at $\kappa \propto \sqrt{w/c}$ for write cost $w$. For cheap writes and frequent
crashes, checkpoint every step; for expensive writes and rare crashes, less often.
This is a two-parameter calculation nobody does.

**Cost comparisons must be conditioned on completion.** In
{{sec:9-practical-example}}'s last table, persisting nothing uses fewer steps than
persisting everything — because most of its runs died early. **A broken durability
design looks efficient**, and comparing steps-per-run across designs with different
completion rates is meaningless.

One caveat on the model. It treats "outputs" as mattering only through how far a
resume must rewind, which makes its ablation read as zero. In a real system the
outputs are what lets the resumed run continue at all — you cannot proceed from step
seven without knowing what steps one to six produced. The listing therefore
understates that field, and the ordering to take from it is between the three
*agent-state* fields rather than between all five.

## 7. Internal Mechanics

### 7.1 The three writes around a side effect

```mermaid {#fig:durable-effect caption="The ordering that makes replay safe. The key is written before the effect, so a crash between them replays into a suppressed handler rather than a duplicate."}
flowchart LR
    A[step begins] --> K[write dedup key]
    K --> E[perform effect]
    E --> R[record completion]
    R --> N[next step]
    K -. crash here .-> S[replay: key exists, suppressed]
    E -. crash here .-> S
```

The order matters. Key first, then effect, then completion record. A crash anywhere
after the key leaves the key present, so the replay is suppressed. A system that
writes the key *after* the effect has a window in which the effect happened and the
key does not exist, which is the bug this design exists to prevent.

### 7.2 What makes a step idempotent

Four patterns, in decreasing order of how often they are available.

**Assignment rather than mutation.** "Set status to approved" is idempotent; "advance
status" is not. Most state transitions can be written either way and the idempotent
form is usually no harder.

**Conditional effects.** "Send unless already sent" requires a record, which is a
deduplication key by another name.

**Natural keys in the downstream system.** Many APIs accept an idempotency key
precisely for this reason, and an agent tool that does not pass one through is
discarding a guarantee that was on offer.

**Compensating actions.** Where an effect cannot be made idempotent, a recorded
inverse lets a duplicate be undone — this is {{ch:ag-planning}}'s rollback, and it is
the fallback when the other three fail.

### 7.3 Persisting the agent's state, specifically

{{eq:tried-set-is-the-missing-field}} says three fields matter and no engine stores
them. Concretely:

**tried** — a list of (action, arguments, outcome) for every failed attempt. This is
already in memory if you implemented {{ch:ag-loop}}'s deduplication; durability is
writing it out.

**derived** — the scratchpad from {{ch:ag-memory}}, with stable keys so a resume can
look values up rather than re-derive them.

**goal** — the original request verbatim. Not a summary, because a summary is a
lossy re-encoding and a resume that reads it inherits the loss permanently.

All three are small, all three are cheap to serialise, and all three are the
difference between resuming and restarting-from-a-position.

### 7.4 A state machine is a graph with $p_e = 1$

{{ch:as-graph}} found a graph's branch decisions costing $p_e^{\beta}$ on every
request, and separated static graphs — whose edges are checkable predicates over
typed state — from dynamic ones whose edges are model judgements.

A state machine is the static case taken seriously. Its transitions are functions of
the persisted state, so they are deterministic, so $p_e = 1$ and
{{eq:branch-count-is-an-exponent}}'s penalty vanishes. It keeps the path bound and
loses the branch cost.

What it gives up is the tail: a state machine can only be in states someone declared.
That is {{eq:graph-surrenders-the-tail}} in its strongest form, and it is why this
design belongs to workflows with agent-filled nodes rather than to agents with a
workflow bolted on.

### 7.5 Events, and why ordering is a correctness property

Event-driven agent systems add a second replay hazard: events may arrive out of
order or more than once, so a handler must be safe under both. That is the same
idempotence requirement with an extra clause about ordering.

The practical rule is to make handlers commutative where possible — the resulting
state does not depend on the order — and where it is not possible, to sequence
explicitly with a version or a sequence number rather than relying on delivery order.
An agent that reasons over an out-of-order event history reaches a state no design
anticipated, which is one of {{cite:cemri2025mast}}'s system-design failure modes.

### 7.6 Durability and untrusted content

A durable store is a persistence boundary, and {{ch:ag-memory}}'s warning applies
with more force: content written from untrusted input survives the run, is retrieved
on resume, and is trusted because it is "our own state".
{{cite:greshake2023indirect}}'s injection vector reaches further when the target is
durable.

Record provenance alongside the value, and treat a resumed run's state with the same
suspicion as any other retrieved content.

### 7.7 Resume is not restart, and the difference is measurable

The distinction this chapter keeps returning to deserves a name, because the two
words are used interchangeably and they describe different things.

A **restart** puts the run back at a position with the engine's state intact. It is
what every workflow framework provides and it is a statement about progress: no
completed work is redone, and the process continues from where it stopped.

A **resume** puts the *agent* back where it was. It requires that the agent's
working knowledge survived — what it had ruled out, what it had derived, what it was
actually asked. It is a statement about continuity of reasoning, and no framework
provides it because the framework does not know what that knowledge is.

The gap between them is exactly {{eq:tried-set-is-the-missing-field}}, and
{{sec:9-practical-example}} prices it at $38.6$ points for the largest field alone.
A system that restarts perfectly and resumes badly will show clean operational
metrics — no lost work, no failed executions — while producing worse answers than a
system with no durability at all, because the restarted agent burns its remaining
budget re-walking a path it had already abandoned.

That is a genuinely awkward failure to detect, because every dashboard the framework
gives you measures restart quality. The measurement that would catch it is in
{{sec:14-evaluation}}: compare the answer quality of runs that crashed and resumed
against runs that did not, matched on task. If durability is working, the
distributions overlap. If only restart is working, the resumed runs are visibly
worse and nothing in the operational telemetry says so.

## 8. Implementation

Two listings. The first measures what replay costs when steps are not idempotent,
and prices the two responses. The second sweeps what is in the durable state and
measures whether the resume is correct.

```python {tier=A name=replay-needs-idempotence}
"""Durable execution is a checkpoint that survives a crash. Replay is not free.

ch:ag-planning's checkpoint lets a failed segment restart from a known-good state.
Durable execution is the same mechanism with the state written to storage, so the
restart can follow a process crash rather than only a logical failure.

The complication is that an agent step usually has a SIDE EFFECT, and replaying it
does it again. A workflow engine that guarantees at-least-once execution therefore
guarantees at-least-once side effects, and the correctness of a resume depends
entirely on how many of the replayed steps are idempotent
(eq:replay-needs-idempotence).

This listing counts the replays and the duplicate effects they cause.
"""
import numpy as np

rng = np.random.default_rng(3413)

M = 40000
STEPS = 12
P_CRASH = 0.05          # chance of a crash after any given step
MAX_ITERS = 200


def run(idem_frac, ck_every, m=M, steps=STEPS, crash=P_CRASH, keys=False):
    """Walk the workflow. A crash rolls the position back to the last durable
    checkpoint; every step between the checkpoint and the crash point is then
    executed a second time. A replayed non-idempotent step duplicates its side
    effect, unless a deduplication key suppresses it."""
    idem = rng.random((m, steps)) < idem_frac
    pos = np.zeros(m, dtype=np.int64)
    anchor = np.zeros(m, dtype=np.int64)
    ran = np.zeros((m, steps), dtype=bool)
    dupes = np.zeros(m, dtype=np.int64)
    corrupt = np.zeros(m, dtype=bool)
    steps_taken = np.zeros(m, dtype=np.int64)
    rows = np.arange(m)
    for _ in range(MAX_ITERS):
        live = (pos < steps) & ~corrupt
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        j = pos[idx]
        steps_taken[idx] += 1
        replay = ran[idx, j]
        bad = replay & (~idem[idx, j]) & (not keys)
        dupes[idx] += bad
        corrupt[idx[bad]] = True
        ran[idx, j] = True
        pos[idx] += 1
        # A checkpoint durably records progress.
        at_ck = (pos[idx] % ck_every) == 0
        anchor[idx[at_ck]] = pos[idx[at_ck]]
        # A crash rolls back to the last durable point.
        crashed = rng.random(len(idx)) < crash
        pos[idx[crashed]] = anchor[idx[crashed]]
    done = (pos >= steps) & ~corrupt
    return (float(done.mean()), float(corrupt.mean()), float(dupes.mean()),
            float(steps_taken.mean()))


print(f"{M:,} runs of a {STEPS}-step workflow, {P_CRASH:.0%} crash chance after")
print("each step. On resume the run replays from the last durable checkpoint,")
print("so every step between the checkpoint and the crash executes twice.")
print()
print(f"{'idempotent steps':>18}{'completed':>12}{'corrupted':>12}"
      f"{'duplicates':>13}{'steps run':>12}")
print("-" * 67)
tab = {}
for f in (0.0, 0.5, 0.8, 0.95, 1.0):
    r = run(f, 3)
    tab[f] = r
    print(f"{f:>18.0%}{r[0]:>12.1%}{r[1]:>12.1%}{r[2]:>13.2f}{r[3]:>12.1f}")

print()
print()
print("Checkpoint frequency decides how much gets replayed, so it decides how")
print("many duplicate side effects a crash causes.")
print()
print(f"{'checkpoint every':>18}{'completed':>12}{'corrupted':>12}"
      f"{'duplicates':>13}{'steps run':>12}")
print("-" * 67)
ck = {}
for c in (1, 2, 3, 6, 12):
    r = run(0.5, c)
    ck[c] = r
    print(f"{c:>18}{r[0]:>12.1%}{r[1]:>12.1%}{r[2]:>13.2f}{r[3]:>12.1f}")

print()
print()
print("A deduplication key makes a non-idempotent step effectively idempotent:")
print("the engine records that the effect happened and suppresses the repeat.")
print()
print(f"{'idempotent steps':>18}{'no key':>12}{'with key':>12}{'gain':>9}")
print("-" * 51)
dk = {}
for f in (0.0, 0.5, 0.8, 1.0):
    a = run(f, 3)[0]
    b = run(f, 3, keys=True)[0]
    dk[f] = (a, b)
    print(f"{f:>18.0%}{a:>12.1%}{b:>12.1%}{b - a:>+9.1%}")

print()
print()
print("And how it all moves with the crash rate, which is what durability is")
print("bought to survive.")
print()
print(f"{'crash rate':>12}{'ck every 12':>14}{'ck every 3':>13}"
      f"{'ck every 1':>13}{'ck 3 + keys':>14}")
print("-" * 66)
cr = {}
for c in (0.01, 0.05, 0.15, 0.30):
    row = (run(0.5, 12, crash=c)[0], run(0.5, 3, crash=c)[0],
           run(0.5, 1, crash=c)[0], run(0.5, 3, crash=c, keys=True)[0])
    cr[c] = row
    print(f"{c:>12.0%}{row[0]:>14.1%}{row[1]:>13.1%}{row[2]:>13.1%}"
          f"{row[3]:>14.1%}")

print(f"""
The first table is the cost of replay, and the first row is the case a workflow
engine's guarantee does not cover.

With no idempotent steps, {tab[0.0][1]:.1%} of runs end corrupted -- a duplicate
side effect somewhere -- against {tab[1.0][1]:.1%} when every step is idempotent.
The completion column tracks it inversely: {tab[0.0][0]:.1%} to {tab[1.0][0]:.1%}.

**An at-least-once execution guarantee is an at-least-once SIDE EFFECT guarantee**
(eq:replay-needs-idempotence), and the engine cannot tell the difference. It
replays the step; whether that is harmless is a property of the step.

The second table is the knob most teams reach for, and it works. Checkpointing
after every step gives {ck[1][0]:.1%} and checkpointing only at the end gives
{ck[12][0]:.1%}, because the amount replayed after a crash is exactly the distance
back to the last durable point.

It is not free: the steps-run column goes {ck[12][3]:.1f} to {ck[1][3]:.1f}, and
each checkpoint is a durable write. **Checkpoint frequency trades write cost
against replay blast radius**, and it is the same trade ch:ag-planning found
between segment count and verification overhead.

The third table is the fix that actually solves the problem rather than shrinking
it. A deduplication key -- the engine records that a particular effect already
happened and suppresses the repeat -- takes every idempotence level to
{dk[0.0][1]:.1%}.

**A key makes a non-idempotent step idempotent from the engine's point of view**,
and it is the difference between mitigating replay and eliminating it. Note the
size: {dk[0.0][1] - dk[0.0][0]:+.1%} at zero natural idempotence, against the
{ck[1][0] - ck[12][0]:+.1%} that checkpointing every step buys.

The last table is why this matters more as systems get less reliable. At
{0.01:.0%} crash rate the coarse-checkpoint design still reaches {cr[0.01][0]:.1%}
and the difference between designs looks academic. At {0.30:.0%} it is
{cr[0.3][0]:.1%} against {cr[0.3][3]:.1%} with keys.

**Durability is bought to survive crashes and its own correctness degrades with
the crash rate** unless the replay is made safe. A design validated in a stable
environment will fail in an unstable one in a way that looks like the environment's
fault.

Three rules follow.

**Ask of every step: what happens if this runs twice?** That question, per tool,
is the entire content of durable-execution correctness, and it is answerable at
design time rather than discovered in an incident.

**Give every non-idempotent effect a deduplication key**, derived from the run and
the step rather than generated fresh -- a fresh key on replay is not a key.

**Checkpoint often enough that the replay window is small**, and treat the write
cost as the price of a smaller blast radius rather than as overhead.""")
```

The second listing asks what has to be persisted.

```python {tier=A name=state-must-be-sufficient}
"""What has to be in the durable state for a resume to be correct.

A checkpoint that records the wrong things produces a resume that looks successful
and is not. This listing sweeps what is persisted and measures how often the
resumed run reaches the right answer (eq:state-must-be-sufficient).

The candidate fields come from part:17, which is not a coincidence -- they are the
same artefacts ch:ag-memory and ch:ag-planning said to build, and durability is
what makes them survive a restart:

  position   how far the run got
  outputs    what each completed step produced
  tried      which actions failed (ch:ag-loop's deduplication set)
  derived    values computed from several inputs (ch:ag-memory's scratchpad)
  goal       the original request, verbatim rather than summarised
"""
import numpy as np

rng = np.random.default_rng(3491)

M = 60000
STEPS = 10
P_CRASH = 0.10
P_STEP = 0.95
P_RECOMPUTE = 0.86      # re-deriving a lost derived value
P_REDISCOVER = 0.55     # re-learning which actions fail, per lost entry
P_REGOAL = 0.90         # reconstructing the goal from a summary

FIELDS = ["position", "outputs", "tried", "derived", "goal"]


def run(persisted, m=M, steps=STEPS, crash=P_CRASH):
    have = set(persisted)
    ok = np.ones(m, dtype=bool)
    pos = np.zeros(m, dtype=np.int64)
    work = np.zeros(m, dtype=np.int64)
    resumes = np.zeros(m, dtype=np.int64)
    for _ in range(steps * 6):
        live = ok & (pos < steps) & (work < steps * 5)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        work[idx] += 1
        p = np.full(len(idx), P_STEP)
        good = rng.random(len(idx)) < p
        pos[idx[good]] += 1
        crashed = rng.random(len(idx)) < crash
        c = idx[crashed]
        if len(c):
            resumes[c] += 1
            # Without position, the run restarts from zero.
            if "position" not in have:
                pos[c] = 0
            # Without outputs, completed work must be redone.
            elif "outputs" not in have:
                pos[c] = np.maximum(pos[c] - 2, 0)
            # Without the tried set, the resumed run re-explores dead ends.
            if "tried" not in have:
                ok[c] &= rng.random(len(c)) < P_REDISCOVER
            # Without derived values, they are recomputed, sometimes wrongly.
            if "derived" not in have:
                ok[c] &= rng.random(len(c)) < P_RECOMPUTE
            # Without the verbatim goal, the run continues toward a paraphrase.
            if "goal" not in have:
                ok[c] &= rng.random(len(c)) < P_REGOAL
    done = ok & (pos >= steps)
    return float(done.mean()), float(work.mean()), float(resumes.mean())


print(f"{M:,} runs, {STEPS} steps, {P_CRASH:.0%} crash rate per step.")
print("Fields are added to the durable state one at a time.")
print()
print(f"{'persisted state':>44}{'completed':>12}{'steps':>9}{'gain':>9}")
print("-" * 76)
cum = {}
have = []
r = run(have)
cum["(nothing durable)"] = r
print(f"{'(nothing durable)':>44}{r[0]:>12.1%}{r[1]:>9.1f}{'--':>9}")
prev = r[0]
for f in FIELDS:
    have.append(f)
    r = run(have)
    cum["+ " + f] = r
    print(f"{('+ ' + f):>44}{r[0]:>12.1%}{r[1]:>9.1f}{r[0] - prev:>+9.1%}")
    prev = r[0]

print()
print()
print("Each field REMOVED from a complete state -- what you lose by omitting it")
print("from a system that persists everything else.")
print()
print(f"{'field omitted':>44}{'completed':>12}{'loss':>10}")
print("-" * 68)
full = run(FIELDS)
drop = {}
for f in FIELDS:
    r = run([x for x in FIELDS if x != f])
    drop[f] = r[0]
    print(f"{f:>44}{r[0]:>12.1%}{r[0] - full[0]:>+10.1%}")

print()
print()
print("How the ranking changes with the crash rate, since a rare crash makes")
print("every field look unnecessary.")
print()
print(f"{'crash rate':>12}{'nothing':>11}{'position only':>16}"
      f"{'position+outputs':>19}{'everything':>13}")
print("-" * 71)
cr = {}
for c in (0.02, 0.10, 0.25, 0.45):
    row = (run([], crash=c)[0], run(["position"], crash=c)[0],
           run(["position", "outputs"], crash=c)[0],
           run(FIELDS, crash=c)[0])
    cr[c] = row
    print(f"{c:>12.0%}{row[0]:>11.1%}{row[1]:>16.1%}{row[2]:>19.1%}"
          f"{row[3]:>13.1%}")

print()
print()
print("And the cost side: what persisting everything saves in re-executed work.")
print()
print(f"{'persisted state':>28}{'completed':>12}{'steps used':>13}"
       f"{'resumes':>10}")
print("-" * 63)
for name, fields in [("nothing", []), ("position only", ["position"]),
                     ("position + outputs", ["position", "outputs"]),
                     ("everything", FIELDS)]:
    r = run(fields)
    print(f"{name:>28}{r[0]:>12.1%}{r[1]:>13.1f}{r[2]:>10.2f}")

print(f"""
The first table adds durable fields one at a time and the totals are unremarkable
until you compare them with the second, which is where the finding is.

Removing the TRIED set -- ch:ag-loop's record of which actions already failed --
from a state that persists everything else costs
{drop['tried'] - full[0]:+.1%}. That is by far the largest single loss, and it is
the field no workflow engine persists.

The reason is ch:ag-loop's, transplanted. A resumed run with no memory of what
failed re-derives the same wrong approach the crashed run had already eliminated,
and it does so with a fresh context that makes the wrong approach look new. **A
resume without the failure set is a run that has forgotten why it was going the
way it was going** (eq:state-must-be-sufficient).

Derived values cost {drop['derived'] - full[0]:+.1%} and the verbatim goal
{drop['goal'] - full[0]:+.1%}. Both are ch:ag-memory's mechanisms needing to
survive a restart, and both are cheap to write and easy to omit.

Position costs {drop['position'] - full[0]:+.1%}, which is much less than its
prominence suggests -- and OUTPUTS costs {drop['outputs'] - full[0]:+.1%}, which is
zero. That second number is an artefact of this model rather than a finding: here
outputs only matter through their effect on how far back a resume must go, which
position already captures. In a real system outputs are what makes the resumed run
able to continue at all, and the listing understates them.

**The ordering to take away is that the fields workflow engines persist by default
-- position and outputs -- are not the ones that decide whether a resume produces
the right answer.** Position is what the engine needs to know where to restart.
The tried set, the derived values and the goal are what the AGENT needs to be the
same agent it was.

The third table shows the ranking depending on the crash rate, which is the reason
this is easy to miss. At {0.02:.0%} crashes, persisting nothing gives {cr[0.02][0]:.1%}
and persisting everything {cr[0.02][3]:.1%} -- a gap of
{cr[0.02][3] - cr[0.02][0]:.1%}, noticeable but survivable. At {0.45:.0%} it is
{cr[0.45][0]:.1%} against {cr[0.45][3]:.1%}.

**A durable-state design validated in a low-crash environment is untested**, and
the failure appears exactly when the environment degrades -- which is when
durability was supposed to help.

The last table prices it. Persisting everything uses {run(FIELDS)[1]:.1f} steps
against {run([])[1]:.1f} for persisting nothing, and completes
{run(FIELDS)[0] - run([])[0]:+.1%} more. The extra steps are not overhead; they are
runs that got far enough to need them, and the "nothing" row is cheap because most
of its runs died early.

**Cost comparisons between durability designs have to be conditioned on
completion**, or the broken design looks efficient.""")
```

## 9. Practical Example

The first listing runs a twelve-step workflow with a $5\%$ crash chance after each
step, resuming from the last durable checkpoint.

```
  idempotent steps   completed   corrupted   duplicates   steps run
-------------------------------------------------------------------
                0%       66.5%       33.5%         0.34        10.2
               50%       77.5%       22.5%         0.23        11.1
               95%       97.2%        2.8%         0.03        12.5
              100%      100.0%        0.0%         0.00        12.6
```

**An at-least-once execution guarantee is an at-least-once side effect guarantee**
({{eq:replay-needs-idempotence}}). Nothing malfunctioned in the $33.5\%$; the engine
honoured its contract and the effect happened twice.

Checkpoint frequency shrinks the exposure:

```
  checkpoint every   completed   corrupted   duplicates   steps run
-------------------------------------------------------------------
                 1      100.0%        0.0%         0.00        12.0
                 3       77.8%       22.2%         0.22        11.1
                12       59.8%       40.2%         0.40        10.5
```

The replay window is the distance back to the last durable point
({{eq:checkpoint-frequency-trade}}), and it costs a write per checkpoint.

A deduplication key removes it:

```
  idempotent steps      no key    with key     gain
---------------------------------------------------
                0%       66.3%      100.0%   +33.7%
               50%       77.6%      100.0%   +22.4%
              100%      100.0%      100.0%    +0.0%
```

**A key makes a non-idempotent step idempotent from the engine's point of view**
({{eq:dedup-key-is-the-fix}}) — it removes the term rather than shrinking it, and it
costs one write per *effect* rather than one per *step*.

And the environment sensitivity:

```
  crash rate   ck every 12   ck every 3   ck every 1   ck 3 + keys
------------------------------------------------------------------
          1%         90.7%        95.1%       100.0%        100.0%
          5%         60.4%        77.5%       100.0%        100.0%
         30%          3.3%        20.1%       100.0%        100.0%
```

At $1\%$ crashes the difference between designs looks academic; at $30\%$ it is
$3.3\%$ against $100\%$. **A durability design validated in a stable environment is
untested** ({{eq:durability-degrades-with-crash-rate}}).

The second listing sweeps what is persisted, and the ablation is where the finding
is:

```
                               field omitted   completed      loss
--------------------------------------------------------------------
                                    position       96.7%     -3.3%
                                     outputs      100.0%     +0.0%
                                       tried       61.4%    -38.6%
                                     derived       86.2%    -13.8%
                                        goal       90.0%    -10.0%
```

Omitting the **tried set** costs $38.6$ points — an order of magnitude more than
omitting the position. A resumed run with no memory of what failed re-derives the
approach the crashed run had eliminated, with a fresh context that makes it look new
({{eq:context-change-breaks-loops}} in reverse).

Derived values cost $13.8$ and the verbatim goal $10.0$. Both are
{{ch:ag-memory}}'s mechanisms needing to survive a restart.

The `outputs` row reading zero is an artefact of this model rather than a finding —
here outputs matter only through rewind distance, which position already captures.
In a real system they are what lets the resume continue at all.

**The three fields that decide correctness are the three no engine stores**
({{eq:tried-set-is-the-missing-field}}). Position and outputs are what the *engine*
needs; tried, derived and goal are what the *agent* needs.

```
  crash rate    nothing   position only   position+outputs   everything
-----------------------------------------------------------------------
          2%      88.0%           88.5%              88.4%       100.0%
         45%       0.3%            0.8%               4.3%       100.0%
```

And the cost side, which contains a trap:

```
             persisted state   completed   steps used   resumes
---------------------------------------------------------------
                     nothing       46.5%          9.4      0.93
                  everything      100.0%         10.5      1.05
```

Persisting nothing uses *fewer* steps — because most of its runs died early.
**Cost comparisons between durability designs must be conditioned on completion**,
or the broken design looks efficient.

## 10. Production Considerations

Ask of every tool: what happens if this runs twice? That question, answered per
tool, is the whole content of durable-execution correctness and it is answerable at
design time.

Give every non-idempotent effect a deduplication key derived from the run and step
ids ({{eq:key-must-be-deterministic}}), written *before* the effect.

Pass idempotency keys through to downstream APIs that accept them. A tool that drops
one is discarding a guarantee that was on offer.

Persist the tried set, the derived values and the verbatim goal — not just position
and outputs. Three small fields, and they are what the ablation says decide
correctness.

Store the goal verbatim. A summary is a lossy re-encoding and a resume inherits the
loss permanently.

Choose the checkpoint interval from write cost and crash rate rather than by
default, and remember it is a mitigation where a key is a fix.

Test at an elevated crash rate. {{eq:durability-degrades-with-crash-rate}} says a
design validated at $1\%$ tells you nothing about $30\%$.

And record provenance on durable state. It outlives the run, and
{{cite:greshake2023indirect}}'s injection reaches further when the target persists.

## 11. Common Mistakes

**Treating at-least-once as a correctness guarantee.** It is a delivery guarantee;
correctness is a property of your steps.

**Generating the deduplication key at effect time.** A fresh key on replay
suppresses nothing.

**Writing the key after the effect.** Leaves a window where the effect happened and
the key does not exist.

**Persisting only position and outputs.** The three fields that decide correctness
are the other three.

**Storing a summarised goal.** The loss becomes permanent at the first resume.

**Comparing durability designs on steps-per-run.** The broken one looks cheaper
because its runs die early.

**Validating durability at production's normal crash rate.** The gap widens exactly
when you need it.

## 12. Failure Modes

*Duplicate side effects.* The canonical one — two emails, two charges, two rows. No
error anywhere; the engine did what it promised.

*Amnesiac resume.* A run continues from the right position having forgotten what it
had ruled out, and repeats the eliminated approach with apparent confidence.

*Goal drift across resumes.* Each restart reads a slightly lossier summary, and a
long-running task ends up answering a related question.

*Out-of-order event application.* A handler that assumes ordering reaches a state no
design anticipated — one of {{cite:cemri2025mast}}'s system-design modes.

*Persisted injection.* Untrusted content written to durable state and read back on
resume as trusted fact.

## 13. Alternatives

**No durability, just retry from scratch.** Correct if steps are cheap and effects
are idempotent, and it avoids every problem in this chapter.

**Compensating transactions.** Where effects cannot be made idempotent, record the
inverse and undo duplicates. More machinery, and the only option for genuinely
irreversible effects.

**Externalised state in a database.** The agent reads and writes typed state rather
than carrying it; durability becomes the database's problem and
{{ch:as-multi-agent}}'s handoff cost falls too.

**A state machine.** {{sec:7-internal-mechanics}}: transitions as predicates over
persisted state, $p_e = 1$, at the cost of {{eq:graph-surrenders-the-tail}}.

**Shorter runs.** The cheapest way to reduce crash exposure is to have less to
crash during, which is {{ch:ag-planning}}'s decomposition arriving for an
operational reason.

## 14. Evaluation

Audit every tool for idempotence and publish the list. It is the input to every
decision here and it is a table, not a project.

Measure your actual crash and preemption rate, and test durability at several times
it.

Measure resume *correctness*, not resume *success* — whether the resumed run reaches
the right answer, not whether it finishes. The two diverge exactly where this
chapter's fields are missing.

Count duplicate effects directly, by looking downstream: how many messages, rows or
charges have the same logical origin. It is the only direct measurement of
{{eq:replay-needs-idempotence}}.

And condition every cost comparison on completion.

## 15. Advanced Concepts

**Deriving idempotence from tool schemas.** Whether a tool mutates, assigns, or
appends is usually inferable from its signature, so the audit in
{{sec:14-evaluation}} could be generated rather than written.
{{maturity:EMERGING}}.

**Agent state as a first-class engine concept.** Workflow engines persist position
and outputs because those are what the engine needs. An engine that also persisted a
declared agent-state schema would make {{eq:tried-set-is-the-missing-field}}'s
omission structurally impossible, and none does.

**Commutative handler design.** For event-driven systems, designing handlers so the
final state is order-independent removes an entire class of replay hazard, and the
conditions under which agent steps can be made commutative are not well
characterised.

**Provenance-carrying durable state.** Combining {{ch:ag-security}}'s provenance
question with persistence: if durable state recorded where each value came from,
trust decisions on resume could be structural. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-planning}}'s checkpoint is this chapter's mechanism, with persistence added
and its budget-cliff warning intact.

{{ch:ag-loop}}'s deduplication set turns out to be the most valuable durable field,
and {{eq:context-change-breaks-loops}} explains why: losing it makes the context
after the resume identical to the context before the failure.

{{ch:ag-memory}}'s scratchpad and verbatim goal are the other two agent-state
fields, and {{eq:scratchpad-removes-an-exponent}} is the rate at which losing them
costs.

{{ch:as-graph}}'s static/dynamic distinction is resolved here: a state machine is
the static case, with $p_e = 1$ and the tail surrendered.

Ahead: {{ch:as-long-running}} takes up what happens over long horizons when no
single step is wrong; {{ch:as-failures}} returns to
{{cite:cemri2025mast}}'s system-design category, of which this chapter's failures
are instances.

## 17. Exercises

1. Derive the optimal checkpoint interval from
   {{eq:checkpoint-frequency-trade}} plus a write cost, and compute it for the
   listing's crash rate.

2. Implement the wrong key ordering — key written after the effect — and measure the
   window it leaves.

3. Add "outputs" as a genuine requirement in the second listing: a resume without
   them cannot continue at all. How does the ablation ordering change?

4. Model goal drift as compounding across resumes rather than being a one-off, and
   measure a long-running task's answer quality against resume count.

5. Add out-of-order event delivery and find which handlers must be commutative for
   correctness.

6. Audit your own tools for idempotence and compute your $\iota$. Where does that
   put you in the first table?

## 18. Interview Questions

1. What does at-least-once execution actually guarantee?

2. Your workflow engine replays a step that sends an email. Whose bug is it?

3. Why is a deduplication key better than checkpointing more often?

4. Where must the key be written relative to the effect, and why?

5. Your durable state has position and outputs. What is missing?

6. Design A completes 47% using 9.4 steps; design B completes 100% using 10.5. Which
   is cheaper?

## 19. Research Questions

1. Can tool idempotence be inferred from schemas reliably enough to generate the
   audit?

2. What would a workflow engine with a first-class agent-state schema look like, and
   would declaring it prevent {{eq:tried-set-is-the-missing-field}}'s omission?

3. Under what conditions can agent steps be made commutative, and how much of the
   event-ordering hazard does that remove?

4. How much does goal drift across resumes actually compound in long-running
   systems?

5. Could durable state carry provenance well enough to make post-resume trust
   decisions structural?

## 20. Chapter Summary

Durable execution is {{ch:ag-planning}}'s checkpoint with persistence, and its
guarantee is narrower than it sounds. **An at-least-once execution guarantee is an
at-least-once side effect guarantee** ({{eq:replay-needs-idempotence}}): with no
idempotent steps, $33.5\%$ of runs ended with a duplicated effect and nothing
malfunctioned.

Checkpointing more often shrinks the replay window — corruption fell from $40.2\%$
to $0.0\%$ as the interval went from twelve steps to one — and it costs a write per
step. A **deduplication key** removes the term instead of shrinking it, taking every
idempotence level to $100\%$ at one write per *effect*
({{eq:dedup-key-is-the-fix}}). It must be derived from the run and step ids and
written *before* the effect ({{eq:key-must-be-deterministic}}).

The gap between designs widens with the crash rate — $90.7\%$ against $100\%$ at
$1\%$ crashes, $3.3\%$ against $100\%$ at $30\%$ — so **a durability design
validated in a stable environment is untested**
({{eq:durability-degrades-with-crash-rate}}).

On what to persist, the ablation is decisive. Omitting the **tried set** costs
$38.6$ points, against $3.3$ for omitting the position. Derived values cost $13.8$
and the verbatim goal $10.0$. **The three fields that decide correctness are the
three no workflow engine stores** ({{eq:tried-set-is-the-missing-field}}) — because
engines persist what engines need, and the agent's working knowledge is application
state.

And a warning about cost: persisting nothing used *fewer* steps than persisting
everything, because most of its runs died early. **Cost comparisons must be
conditioned on completion.**

## 21. Further Reading

{{cite:cemri2025mast}}'s system-design failure category contains this chapter's
failures observed in real traces, and is worth reading against
{{sec:12-failure-modes}}.

{{ch:ag-planning}} for the checkpoint mechanism and its budget cliff, and
{{ch:ag-loop}} for the deduplication set that turns out to be the field that
matters.

{{ch:as-graph}} for the static/dynamic edge distinction this chapter resolves, and
{{cite:greshake2023indirect}} for why durable state is a persistence boundary for
untrusted content as well as for progress.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for the long-horizon
settings where crash exposure accumulates enough to make any of this bind.
