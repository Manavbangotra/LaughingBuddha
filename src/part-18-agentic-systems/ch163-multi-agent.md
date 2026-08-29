---
id: as-multi-agent
number: 163
part: XVIII
tier: full
status: draft
requires: [residual-failure-decomposition, boundary-crossing-cost,
           decorrelation-is-the-variable]
provides: [handoff-is-a-bottleneck, schema-not-story, decorrelate-cheaply,
           equal-cost-comparison, specialisation-needs-no-retries,
           agents-are-a-cost]
citations: [du2023debate, cemri2025mast, gao2023pal, yao2023react,
            liu2024agentbench, shinn2023reflexion, huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to price a handoff and show that its
cost is geometric in the number of handoffs; state the single cheapest mitigation
and why it is the same one {{ch:ag-tool-calling}} recommended; run the equal-cost
comparison that multi-agent claims usually omit; explain why a single agent forced
to vary its approach can beat two agents debating, even at zero shared blind spots;
and say the one condition under which specialisation across agents does pay.

## 2. Why This Matters

{{ch:as-single-agent}} established that the entire value of a second agent is
decorrelation, and that no architecture touches the capability term. This chapter
asks what the second agent *costs*, and then whether the decorrelation could have
been bought without it.

The cost is a handoff, and it is larger than it looks. Passing work between agents
is a serialisation: the sender compresses its state into a message and the receiver
reconstructs enough to continue, and both halves are lossy.
{{sec:9-practical-example}} holds the work fixed and varies only how many times it
changes hands: one agent completes $99.8\%$, four agents $68.6\%$, twelve agents
$25.7\%$. The fit to a geometric law is essentially exact, with an implied
per-handoff factor of $0.884$.

**Success falls geometrically in the number of times control changes hands** — the
same shape as {{ch:rsn-tool-assisted}}'s boundary crossings, with a worse constant,
because an agent's state is larger and less structured than a tool call's
arguments. An architecture diagram with six agents in a chain starts at $0.884^5$
of a single agent's success before anyone has made a mistake.

The mitigation is the one {{ch:ag-tool-calling}} already recommended, at a bigger
boundary. Give the handoff a schema instead of prose and twelve agents go from
$25.7\%$ to $62.7\%$.

Then the harder question. {{cite:du2023debate}} is the strongest published case for
multiple agents: instances propose and debate over rounds, and mathematical and
factual accuracy improve. {{sec:9-practical-example}} runs it against a single agent
at *equal call budget* and finds the single agent ahead — $49.5\%$ against
$27.1\%$ — and still ahead when the two debaters share *zero* blind spots
($49.5\%$ against $39.5\%$).

**Decorrelation is a property of the samples, not of the agents.** Two instances
decorrelate because they are different systems; one instance decorrelates because
it was made to try a different approach. Both produce varied attempts, and only one
halves its own budget to do it.

## 3. Prerequisites

You need {{ch:as-single-agent}}'s residual decomposition and
{{eq:decorrelation-is-the-variable}} — this chapter is about where that
decorrelation comes from and what it costs.

From {{ch:rsn-tool-assisted}}, the boundary-crossing arithmetic. A handoff is the
same object with a larger payload, and the equation transfers unchanged.

From {{ch:ag-tool-calling}}, the schema result: constraining what crosses a
boundary raises the crossing's reliability, and the effect is raised to the power
of the crossing count.

And {{ch:ag-loop}}'s observation that a loop with slack is not a chain, because it
turns out to be the reason specialisation usually does not pay.

## 4. Intuitive Explanation

Two agents working on one task must, at some point, hand it over. That handoff is
the thing to reason about, and it is easy to underestimate because it looks like a
message.

Consider what actually has to happen. The first agent holds a working state:
what it has tried, what it learned, what the current situation is, what remains.
Most of that is in its context window as accumulated history. To hand over, it must
compress that into something the second agent can read — a summary, a status
object, a message. Compression loses things.

Then the second agent must reconstruct enough of the state to continue. It reads a
description of a situation it did not experience. Reconstruction loses things too.

Both halves are lossy, and the losses multiply across handoffs. Two handoffs are
not twice the cost of one in the way that two steps are twice the work — they are
the *square* of the per-handoff survival probability, because the task has to
survive both.

That is exactly the arithmetic {{ch:rsn-tool-assisted}} found for tool calls, and
the constant is worse here for a specific reason: a tool call's argument is a small
structured object and an agent's state is a large unstructured one. Which points
straight at the mitigation. Give the handoff a schema — required fields, an explicit
list of what has been tried, a typed status — and both halves become much more
reliable. {{sec:9-practical-example}} measures that recovering most of the loss.

Now the harder question, which is whether the handoff buys anything.

The usual justification is specialisation: agent two is better at its part than
agent one would have been. That is a real effect and it competes against a real
cost, so it is an arithmetic question.

And it usually loses, for a reason that is not obvious. If the agent can *retry* a
failed step, then a step it would have got wrong on the first attempt it probably
gets right on the second. A specialist's advantage is concentrated on first
attempts, and first attempts are the ones retries already cover.
{{sec:9-practical-example}} measures a specialist edge of up to ten points buying
nothing at all when three attempts are available — and buying a great deal when only
one is.

**Specialisation pays where retries are unavailable**, which means where actions
have side effects, where the budget is tight, or where a step is expensive. That is
a much narrower condition than "the sub-tasks are different".

The second justification is ensembling: several agents attempt the same thing and
their disagreement is informative. This has a real theoretical basis —
{{ch:as-single-agent}} showed decorrelation is the only thing that helps the
residual — and it is the mechanism behind {{cite:du2023debate}}.

But decorrelation is about the *attempts*, not about who makes them. An agent asked
to solve a step a second time, with instructions to approach it differently, is
generating a decorrelated attempt. It costs one model call, the same as an attempt
by a second agent — except that the second agent's design halved the budget to
afford two of everything.

So the comparison at equal cost comes out in an uncomfortable place. One agent
spending twenty-four calls on varied attempts beats two agents spending twelve each
on independent ones, and it beats them even when the two agents are perfectly
independent, because the budget halving costs more than the independence buys.

None of that makes {{cite:du2023debate}} wrong. Their procedure has instances *read
each other's reasoning and revise*, which is an extra mechanism this chapter's model
does not include. What the measurement establishes is narrower and still useful:
the ensembling half of debate is available more cheaply from one agent, so a
debate's gains must come from the critique half to justify its cost.

## 5. Formal Explanation

A handoff is a write followed by a read, each imperfect. With write reliability
$w$ and read reliability $r$, a task crossing $h$ boundaries survives with:

$$S(h) = S(0) \cdot (w\,r)^{h}$$ (eq:handoff-is-a-bottleneck)

geometric in $h$, exactly as {{ch:rsn-tool-assisted}}'s boundary crossings.
{{sec:9-practical-example}} fits this with residuals under $0.3$ points across
$h \in \{0, \ldots, 11\}$, so it is not an approximation in this setting.

Two consequences follow immediately. **The number of agents is an exponent**, so an
architecture's agent count is a first-order design parameter rather than a
presentational one. And $wr$ appears raised to $h$, so:

$$\frac{\partial \log S}{\partial \log(wr)} = h$$ (eq:schema-not-story)

**handoff quality is leveraged by the handoff count**, which makes it the single
most valuable number in a multi-agent design and the one least often measured.
Structuring the payload raises $w$ and $r$ together and is therefore worth $h$ times
whatever it costs.

Now the two justifications for paying $(wr)^h$.

**Specialisation.** Let the specialist have per-step accuracy $p + \epsilon$ against
a generalist's $p$, with $a$ attempts available. A step succeeds with
$1 - (1-p)^a$, so the specialist's contribution is:

$$\Delta_{\text{spec}} = (1-p)^{a} - (1-p-\epsilon)^{a}$$ (eq:specialisation-needs-no-retries)

which shrinks rapidly in $a$: at $a = 1$ it is $\epsilon$, and at $a = 3$ with
$p = 0.94$ it is under $0.001$. **Retries eat the specialist's edge before the
handoff can pay for it**, which is {{ch:ag-loop}}'s slack argument applied to
architecture. Specialisation is worth its handoff only when $a$ is small.

**Ensembling.** Let two instances share a fraction $\rho$ of blind spots and split a
budget $B$ between them. Each gets $B/2$ attempts. Against one agent with $B$
attempts and forced diversity $d$:

$$S_{\text{pair}} = 1 - q^{B/2}\big(\rho + (1-\rho)q^{B/2}\big), \qquad S_{\text{solo}} = 1 - \prod_{t=1}^{B}q_t(d)$$ (eq:decorrelate-cheaply)

where $q_t(d)$ falls with $d$ because a diverse retry is a fresh draw rather than a
repeat. The solo agent has twice the exponent; the pair has the $(1-\rho)$ factor.
**Doubling the exponent beats removing a factor bounded by one**, which is why
{{sec:9-practical-example}} finds the solo agent ahead even at $\rho = 0$.

Finally, the comparison discipline. Every claim in {{part:18}} should be evaluated
as:

$$\text{gain} = S_{\text{architecture}}(B) - S_{\text{reference single agent}}(B)$$ (eq:equal-cost-comparison)

at the *same* $B$, against {{ch:as-single-agent}}'s reference configuration rather
than a bare loop. {{cite:cemri2025mast}}'s observation about minimal gains is what
{{eq:equal-cost-comparison}} produces when it is finally run.

## 6. Mathematical Foundation

Three extractions.

**The handoff penalty is constant in task size.** From
{{eq:handoff-is-a-bottleneck}}, $S(h)/S(0) = (wr)^h$ contains no work term, and
{{sec:9-practical-example}} confirms it: three handoffs cost $31.0$ points on a
six-step task and $30.8$ on a forty-step one. **You cannot amortise a handoff by
giving each agent more to do** — the cost is per boundary, not per unit of work.

That is worth stating because it inverts a common design instinct. "Give each agent
a bigger chunk so the handoffs matter less" reduces the *count* of handoffs, which
helps, and does nothing about their individual cost.

**The break-even specialist edge grows with attempts.** Setting
{{eq:specialisation-needs-no-retries}} equal to the handoff cost $1 - wr$ and
solving for $\epsilon$ gives a threshold that rises steeply in $a$. At $a=1$ a
modest edge suffices; at $a=3$ the required edge exceeds what a specialist plausibly
has. **Check your retry budget before arguing for specialists.**

**The pair's disadvantage widens with budget.** From
{{eq:decorrelate-cheaply}}, the solo agent's exponent is $B$ and the pair's is
$B/2$, so the gap grows as $B$ rises — measured at $9.9$ points at $B=16$ and
$45.0$ at $B=64$. **Spending more makes the single-agent design relatively better**,
which is the opposite of the usual expectation that bigger budgets justify more
elaborate architectures.

One caveat on the ensembling model, and it is the important one. It represents a
debate as "either instance solves it", which captures ensembling and omits
critique. {{cite:du2023debate}}'s instances read each other's reasoning and revise,
and that is a mechanism with no counterpart here. The listing therefore bounds the
*ensembling* contribution and says nothing about the critique contribution — so a
debate's measured gains, where they exist, are evidence about critique rather than
about having two agents.

That also gives the falsifiable prediction: a debate whose instances do not read
each other should perform like the pair column, and one that does should exceed it
by whatever critique is worth. That comparison is cheap and rarely run.

## 7. Internal Mechanics

### 7.1 What crosses the boundary

```mermaid {#fig:handoff caption="A handoff is a write and a read, each lossy. The dashed content is what does not survive: the reasoning, the dead ends, the tacit context."}
flowchart LR
    A[agent 1 state] --> W[serialise]
    W --> Msg[message]
    Msg --> R[reconstruct]
    R --> B[agent 2 state]
    A -. history .-x Msg
    A -. failed attempts .-x Msg
    A -. tacit context .-x Msg
```

The three dashed edges are where the cost lives, and the third is the one a schema
cannot fix. What agent one *understood about the situation* has no field.

### 7.2 What a handoff schema should contain

Four things, in decreasing order of what {{sec:9-practical-example}}'s structured
column suggests they are worth:

**The goal restated**, so the receiver is not inferring it from a partial trace.

**What has been tried and failed.** This is {{ch:ag-loop}}'s deduplication set,
which otherwise resets at every handoff — a second agent with no record of the first
agent's failures will repeat them.

**Verified state**, which is {{ch:ag-planning}}'s checkpoint. A boundary is a
natural place to anchor, and a handoff that carries an anchor makes the receiver's
failures recoverable.

**Open questions**, meaning what the sender could not determine. Without this the
receiver cannot distinguish "not attempted" from "attempted and inconclusive".

Note that three of the four are artefacts other chapters already told you to build.
A system with {{part:17}}'s components has most of a handoff schema lying around.

### 7.3 Why the deduplication set is the expensive omission

Of the four, the failed-attempts list has an outsized effect, because losing it
recreates {{ch:ag-loop}}'s repetition failure at each boundary. The receiver's
context is *fresh*, which sounds good and means it has no record of what does not
work — so it re-derives the same wrong approach the sender already eliminated.

That is a specific, common, and easily fixed multi-agent pathology, and it is one of
the {{cite:cemri2025mast}} inter-agent-misalignment category's mechanisms.

### 7.4 Forced diversity, and what it assumes

The single-agent column in {{sec:9-practical-example}} assumes an agent can be made
to take a genuinely different approach on demand. That is the load-bearing
assumption of this chapter and it is testable: sample a stuck step twice with a
diversity-forcing instruction and measure how often the two attempts differ
meaningfully.

If they do not — if the model returns essentially the same approach with different
wording — the single-agent numbers are optimistic and the multi-agent case
strengthens. **This is the measurement that decides the chapter's conclusion for
your system**, and it takes an afternoon.

### 7.5 Cost and latency

Two agents are two model calls where one would have done, plus the handoff's own
call. Sequential handoffs are serial, so latency is additive, and each agent
re-establishes context from scratch, so prefill is paid repeatedly rather than
amortised across a growing cache ({{part:15}}).

The one genuine architectural advantage of multiple agents is that *independent*
work parallelises. Where two agents genuinely do not need each other's output, they
run concurrently and the wall-clock cost halves. That is a real benefit and it is
about latency rather than accuracy, and it should be argued for on those terms.

## 8. Implementation

Two listings. The first prices the handoff by holding the work fixed and varying
only how many times it changes hands. The second runs four designs at equal call
budget and asks where the decorrelation comes from.

```python {tier=A name=handoff-is-a-bottleneck}
"""What a handoff costs, and why the number of agents should be minimised.

Passing work between agents is a serialisation. The sending agent compresses its
state into a message; the receiving agent reconstructs enough of it to continue.
Both halves are lossy, and the loss is multiplicative in the number of handoffs
(eq:handoff-is-a-bottleneck).

This is ch:rsn-cot's token bottleneck at the level of a whole agent, and
ch:rsn-tool-assisted's boundary-crossing cost with a bigger boundary. The
arithmetic is the same and the constant is worse, because an agent's state is
larger and less structured than a tool call's arguments.

This listing holds the total work fixed and varies only how many times it changes
hands.
"""
import numpy as np

rng = np.random.default_rng(2917)

M = 40000
WORK = 12               # total productive steps the task needs
P_STEP = 0.94           # a step succeeds
ATTEMPTS = 3            # retries available per step


def run(handoffs, p_write=0.93, p_read=0.95, structured=False, m=M,
        work=WORK, p_step=P_STEP, attempts=ATTEMPTS):
    """The work is split into `handoffs + 1` stretches. Between stretches the
    state is serialised and reconstructed. `structured` means the state has a
    schema rather than being prose, which raises both halves."""
    if structured:
        p_write = 1 - (1 - p_write) * 0.35
        p_read = 1 - (1 - p_read) * 0.35
    ok = np.ones(m, dtype=bool)
    stretch = max(1, work // (handoffs + 1))
    for h in range(handoffs + 1):
        n = stretch if h < handoffs else work - stretch * handoffs
        n = max(n, 0)
        for _ in range(n):
            got = np.zeros(m, dtype=bool)
            for _a in range(attempts):
                got |= (~got) & (rng.random(m) < p_step)
            ok &= got
        if h < handoffs:
            ok &= rng.random(m) < p_write
            ok &= rng.random(m) < p_read
    return float(ok.mean())


print(f"A task needing {WORK} productive steps at {P_STEP:.0%} each, with")
print(f"{ATTEMPTS} attempts per step. The work is split across agents; between")
print("stretches the state is written out and read back in.")
print()
print(f"{'handoffs':>10}{'agents':>9}{'steps each':>13}{'prose state':>14}"
      f"{'structured state':>19}")
print("-" * 65)
tab = {}
for h in (0, 1, 2, 3, 5, 11):
    a = run(h)
    b = run(h, structured=True)
    tab[h] = (a, b)
    print(f"{h:>10}{h + 1:>9}{max(1, WORK // (h + 1)):>13}{a:>14.1%}"
          f"{b:>19.1%}")

print()
print()
print("The handoff penalty in isolation: same work, same steps, only the")
print("serialisation quality varies.")
print()
print(f"{'write x read':>14}{'1 handoff':>12}{'3 handoffs':>13}"
      f"{'11 handoffs':>14}")
print("-" * 53)
q_tab = {}
for pw, pr in ((0.99, 0.99), (0.95, 0.97), (0.93, 0.95), (0.88, 0.90),
               (0.80, 0.85)):
    row = [run(h, p_write=pw, p_read=pr) for h in (1, 3, 11)]
    q_tab[(pw, pr)] = row
    print(f"{pw * pr:>14.1%}{row[0]:>12.1%}{row[1]:>13.1%}{row[2]:>14.1%}")

print()
print()
print("Is the handoff cost really multiplicative? Fit success against handoff")
print("count and check the implied per-handoff factor.")
print()
print(f"{'handoffs':>10}{'measured':>11}{'no-handoff x f^h':>19}{'residual':>11}")
print("-" * 51)
f0 = tab[0][0]
fac = (tab[1][0] / f0)
for h in (0, 1, 2, 3, 5, 11):
    pred = f0 * fac ** h
    print(f"{h:>10}{tab[h][0]:>11.1%}{pred:>19.1%}"
          f"{tab[h][0] - pred:>+11.1%}")
print()
print(f"  implied per-handoff factor: {fac:.3f}")

print()
print()
print("What a handoff has to buy to be worth taking. A specialist is better at")
print("its stretch; how much better does it have to be? Shown with retries")
print("available and without, because retries already saturate an easy step.")
print()
print(f"{'specialist edge':>17}{'3 attempts':>26}{'1 attempt':>24}")
print(f"{'':>17}{'no handoff':>13}{'3 handoffs':>13}{'no handoff':>12}"
      f"{'3 handoffs':>12}")
print("-" * 67)
edge = {}
for e in (0.0, 0.02, 0.04, 0.06, 0.10):
    ps = min(P_STEP + e, 0.999)
    a0 = run(0, p_step=P_STEP)
    a3 = run(3, p_step=ps)
    b0 = run(0, p_step=P_STEP, attempts=1)
    b3 = run(3, p_step=ps, attempts=1)
    edge[e] = (a0, a3, b0, b3)
    print(f"{e:>17.0%}{a0:>13.1%}{a3:>13.1%}{b0:>12.1%}{b3:>12.1%}")

print()
print()
print("And how it scales with task size, at a fixed 3 handoffs.")
print()
print(f"{'work steps':>12}{'no handoff':>13}{'3 handoffs':>13}{'loss':>9}")
print("-" * 47)
wk = {}
for w in (6, 12, 24, 40):
    a = run(0, work=w)
    b = run(3, work=w)
    wk[w] = (a, b)
    print(f"{w:>12}{a:>13.1%}{b:>13.1%}{b - a:>+9.1%}")

print(f"""
The first table is the cost, and the shape of the column is the finding.

The same {WORK} steps of work: one agent completes {tab[0][0]:.1%}, two agents
{tab[1][0]:.1%}, four {tab[3][0]:.1%}, twelve {tab[11][0]:.1%}. **Nothing about
the work changed.** The only difference is how many times the state was written
out and read back in.

The structured column is the cheapest available mitigation. Giving the handoff a
schema rather than prose takes twelve agents from {tab[11][0]:.1%} to
{tab[11][1]:.1%}, and three handoffs from {tab[3][0]:.1%} to {tab[3][1]:.1%}. That
is the same intervention ch:ag-tool-calling recommended for tool arguments, at a
larger boundary: **a handoff is a tool call whose argument is an entire working
state, and constraining its shape helps for the same reason.**

The third table checks whether the cost is really multiplicative, and it is
almost exactly so. Predicting from the single-handoff factor of {fac:.3f} raised
to the handoff count reproduces every measured value within
{max(abs(tab[h][0] - f0 * fac ** h) for h in (0, 1, 2, 3, 5, 11)):.1%}
(eq:handoff-is-a-bottleneck).

So the rule is the one ch:rsn-tool-assisted reached about tool boundaries, with a
worse constant: **success falls geometrically in the number of times control
changes hands.** An architecture diagram with six agents in a chain is
{fac:.3f}^5 = {fac ** 5:.2f} of the success of the same work done by one, before
any of them has done anything wrong.

The second table says what the constant depends on, and it is entirely the
serialisation quality. At {0.98:.0%} write-times-read the three-handoff
configuration reaches {q_tab[(0.99, 0.99)][1]:.1%}; at {0.68:.0%} it reaches
{q_tab[(0.80, 0.85)][1]:.1%}. At eleven handoffs the same range is
{q_tab[(0.99, 0.99)][2]:.1%} to {q_tab[(0.80, 0.85)][2]:.1%}.

**Handoff quality is raised to the power of the handoff count**, which means it is
the single most leveraged number in a multi-agent design and the one least often
measured.

The fourth table asks the question that decides whether any of this is worth it:
if the receiving agent is BETTER at its stretch, does the specialisation pay for
the handoff?

With retries available, no. At every specialist edge from {0:.0%} to {0.10:.0%},
three handoffs land at about {edge[0.10][1]:.1%} against a single agent's
{edge[0.10][0]:.1%}. The edge buys nothing because **retries have already
saturated the steps** -- ch:ag-loop's finding that a loop with slack is not a
chain, arriving as an argument against specialisation.

Without retries the picture reverses. A single agent reaches
{edge[0.0][2]:.1%}, three handoffs with no edge reach {edge[0.0][3]:.1%}, and
three handoffs with a {0.04:.0%} edge reach {edge[0.04][3]:.1%} -- ahead.

**Specialisation pays only where retries are unavailable**, and retries are
unavailable when actions have side effects, when the budget is tight, or when the
step is expensive. That is a much narrower condition than the usual case for
splitting work across specialists, and it is checkable: if your agent can retry a
failed step, a specialist's per-step edge is being spent on steps that would have
succeeded on the second try anyway.

The last table confirms the loss is a property of the handoffs rather than of the
task. At {6} steps of work three handoffs cost {wk[6][1] - wk[6][0]:.1%}; at
{40} steps they cost {wk[40][1] - wk[40][0]:.1%}. **The penalty is constant in task
size**, because it depends on how many boundaries there are and not on how much
work sits between them.

Which gives the design rule: **hand off as few times as the work allows, and when
you must, hand off a schema rather than a story.** The number of agents in a
system is a cost, not a feature, and an architecture that adds one should be able
to say what it buys against a factor of {fac:.3f}.""")
```

The second listing puts the architectures on one budget.

```python {tier=A name=decorrelate-cheaply}
"""Where the decorrelation comes from, and whether it needs a second agent.

ch:as-single-agent found that the entire value of a second agent is decorrelation.
cite:du2023debate is the strongest published case that it is real: several
instances propose and debate over several rounds and factual and mathematical
accuracy improve.

But decorrelation is a property of the SAMPLES, not of the agents. A single agent
that is forced to approach a stuck step differently is decorrelating its own
retries, and it pays no handoff cost to do it (eq:decorrelate-cheaply).

This listing puts four designs on the same call budget: one agent retrying
normally, one agent retrying with forced approach diversity, two agents debating,
and two agents dividing the work. The comparison is at EQUAL COST, which is the
comparison cite:cemri2025mast says is usually missing.
"""
import numpy as np

rng = np.random.default_rng(3001)

M = 40000
WORK = 8
BUDGET = 24             # model calls available per task, for every design
P_ORD = 0.94            # an ordinary step
P_STICKY_FIRST = 0.30   # a sticky step, first approach
P_SHARE_STICKY = 0.22   # share of steps that are sticky
P_HANDOFF = 0.884       # per-handoff factor, measured in the previous listing


def solve_step(sticky, tries, diversity, m):
    """Attempt one step `tries` times. `diversity` is the chance a retry takes a
    genuinely different approach rather than repeating the last one."""
    ok = np.zeros(m, dtype=bool)
    fresh = np.ones(m, dtype=bool)         # is this attempt a new approach?
    for t in range(tries):
        p = np.where(sticky,
                     np.where(fresh, P_STICKY_FIRST, 0.04),
                     P_ORD)
        ok |= (~ok) & (rng.random(m) < p)
        fresh = rng.random(m) < diversity
    return ok


def design(kind, m=M, work=WORK, budget=BUDGET):
    sticky = rng.random((m, work)) < P_SHARE_STICKY
    if kind == "single":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.0, m)
        return float(ok.mean())
    if kind == "single_diverse":
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.85, m)
        return float(ok.mean())
    if kind == "debate":
        # Two instances propose independently and reconcile. Half the budget
        # each; a step succeeds if either instance solves it.
        tries = budget // (2 * work)
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            a = solve_step(sticky[:, j], tries, 0.0, m)
            b = solve_step(sticky[:, j], tries, 0.0, m)
            # Independent instances share the sticky blind spot only partly.
            shared = rng.random(m) < 0.45
            ok &= a | (b & ~shared)
        return float(ok.mean())
    if kind == "divide":
        # Two agents take half the work each, with one handoff between them.
        tries = budget // work
        ok = np.ones(m, dtype=bool)
        for j in range(work):
            ok &= solve_step(sticky[:, j], tries, 0.0, m)
        ok &= rng.random(m) < P_HANDOFF
        return float(ok.mean())
    raise ValueError(kind)


DESIGNS = [("one agent, plain retries", "single"),
           ("one agent, forced diversity", "single_diverse"),
           ("two agents, debate", "debate"),
           ("two agents, divided work", "divide")]

print(f"{M:,} tasks, {WORK} steps, {BUDGET} model calls for every design.")
print(f"{P_SHARE_STICKY:.0%} of steps are sticky: {P_STICKY_FIRST:.0%} on a fresh")
print("approach and near-zero on a repeat. Two instances share 45% of the")
print(f"sticky blind spots. A handoff costs a factor of {P_HANDOFF}.")
print()
print(f"{'design':>32}{'completed':>12}{'calls':>9}{'agents':>9}")
print("-" * 62)
res = {}
for name, k in DESIGNS:
    v = design(k)
    res[name] = v
    n_ag = 1 if k.startswith("single") else 2
    print(f"{name:>32}{v:>12.1%}{BUDGET:>9}{n_ag:>9}")

print()
print()
print("How much diversity does a single agent need to match two debating ones?")
print()
print(f"{'diversity':>11}{'one agent':>12}{'two debating':>15}{'gap':>9}")
print("-" * 47)
deb = res["two agents, debate"]
div = {}
for d in (0.0, 0.25, 0.5, 0.75, 0.95):
    ok = np.ones(M, dtype=bool)
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tries = BUDGET // WORK
    for j in range(WORK):
        ok &= solve_step(sticky[:, j], tries, d, M)
    v = float(ok.mean())
    div[d] = v
    print(f"{d:>11.0%}{v:>12.1%}{deb:>15.1%}{v - deb:>+9.1%}")

print()
print()
print("And how the comparison moves with how much the two instances share.")
print("A debate between two copies of the same model shares almost everything.")
print()
print(f"{'shared blind spots':>20}{'two debating':>15}{'one, diverse':>15}"
      f"{'better':>16}")
print("-" * 66)
sh = {}
single_div = res["one agent, forced diversity"]
for s in (0.95, 0.75, 0.45, 0.20, 0.0):
    sticky = rng.random((M, WORK)) < P_SHARE_STICKY
    tries = BUDGET // (2 * WORK)
    ok = np.ones(M, dtype=bool)
    for j in range(WORK):
        a = solve_step(sticky[:, j], tries, 0.0, M)
        b = solve_step(sticky[:, j], tries, 0.0, M)
        shared = rng.random(M) < s
        ok &= a | (b & ~shared)
    v = float(ok.mean())
    sh[s] = v
    best = "debate" if v > single_div else "one agent"
    print(f"{s:>20.0%}{v:>15.1%}{single_div:>15.1%}{best:>16}")

print()
print()
print("The same four designs at several budgets, since halving the budget per")
print("instance is what a two-agent design actually does.")
print()
print(f"{'budget':>8}" + "".join(f"{n.split(',')[0] + ' ' + n.split(',')[1][:8]:>22}"
                                 for n, _ in DESIGNS[:2])
      + f"{'debate':>10}{'divided':>10}")
print("-" * 74)
bd = {}
for b in (16, 24, 40, 64):
    row = [design(k, budget=b) for _, k in DESIGNS]
    bd[b] = row
    print(f"{b:>8}{row[0]:>22.1%}{row[1]:>22.1%}{row[2]:>10.1%}{row[3]:>10.1%}")

print(f"""
The first table is the comparison at equal cost, and the ordering is the result.

One agent with plain retries: {res['one agent, plain retries']:.1%}. Two agents
debating: {res['two agents, debate']:.1%}. Two agents dividing the work:
{res['two agents, divided work']:.1%}. **One agent with forced approach diversity:
{res['one agent, forced diversity']:.1%}.**

The single agent wins by {res['one agent, forced diversity'] - res['two agents, debate']:.1%}
over the debating pair, at the same {BUDGET} calls, with no handoff and no second
system to operate.

The mechanism is arithmetic rather than subtle. **Decorrelation is a property of
the SAMPLES, not of the agents** (eq:decorrelate-cheaply). Two instances
decorrelate because they are different systems; one instance decorrelates because
it was made to try a different approach. Both produce varied attempts, and only
one of them halves its own budget to do it.

The second table quantifies how much diversity a single agent needs to match the
pair. At {0:.0%} forced diversity it already scores {div[0.0]:.1%} against the
debate's {deb:.1%} -- ahead, purely because it kept its whole budget. At
{0.5:.0%} it is {div[0.5]:.1%}, and at {0.95:.0%}, {div[0.95]:.1%}.

The third table is the one that should settle an architecture argument. It sweeps
how much the two debating instances share, from {0.95:.0%} -- two copies of the
same model, which is what a debate between two prompts of one model actually is --
down to {0:.0%}, which is two genuinely independent systems.

At {0.95:.0%} shared blind spots the debate reaches {sh[0.95]:.1%}. At
{0:.0%} -- perfect independence, which no real pair achieves -- it reaches
{sh[0.0]:.1%}. The single diverse agent scores {single_div:.1%}.

**The debate loses at every level of independence**, including at zero, because
the budget halving costs more than the decorrelation buys. That is the honest form
of this listing's finding and it is stronger than expected: it is not that
multi-agent debate is a marginal improvement, it is that at equal cost it is
behind a single agent that varies its own approach.

The fourth table shows the gap is not an artefact of the budget. At {16} calls the
single diverse agent leads by {bd[16][1] - bd[16][2]:.1%}; at {64} by
{bd[64][1] - bd[64][2]:.1%}. **The gap widens with budget**, because the single
agent gets the whole increase and the pair gets half each.

Three honest caveats, and the second is substantial.

This model gives debate no mechanism beyond "either instance solves the step".
cite:du2023debate's actual procedure has instances READ each other's reasoning and
revise, which is a real additional mechanism this listing does not represent --
and their reported gains on mathematical and factual tasks are real. What the
listing establishes is narrower: **the ensembling half of debate is available more
cheaply from one agent**, so a debate's gains have to come from the critique half
to be worth its cost.

Second, "forced diversity" is an assumption here rather than a measurement. The
listing supposes an agent CAN be made to take a genuinely different approach on
demand, at {0.85:.0%} reliability. That is the load-bearing assumption and it is
the one to test on your own system: sample a stuck step twice with a
diversity-forcing prompt and measure how often the two attempts are actually
different. If they are not, the single-agent column in the first table is
optimistic and the argument weakens.

Third, none of this touches the capability term. ch:as-single-agent's
eq:multi-agent-ceiling applies to every column here: a step no instance can do is
not solved by varying the approach or by adding an instance.

So the question to put to any multi-agent proposal is not whether it decorrelates.
It is **whether it decorrelates more per model call than forcing one agent to vary
its approach** -- and this listing says the bar is higher than the architecture
diagrams suggest.""")
```

## 9. Practical Example

The first listing gives a task needing twelve productive steps at $94\%$ each with
three attempts per step, and varies only how many times the state changes hands.

```
  handoffs   agents   steps each   prose state   structured state
-----------------------------------------------------------------
         0        1           12         99.8%              99.7%
         1        2            6         88.2%              95.7%
         3        4            3         68.6%              87.7%
        11       12            1         25.7%              62.7%
```

Same work, same steps: $99.8\%$ with one agent, $25.7\%$ with twelve. And the
structured column is the cheapest mitigation available — a schema instead of prose
takes twelve agents from $25.7\%$ to $62.7\%$
({{eq:schema-not-story}}).

The cost is geometric, and almost exactly so:

```
  handoffs   measured   no-handoff x f^h   residual
---------------------------------------------------
         0      99.8%              99.8%      +0.0%
         3      68.6%              68.9%      -0.3%
        11      25.7%              25.6%      +0.0%

  implied per-handoff factor: 0.884
```

Every measured value is reproduced by $S(0) \cdot 0.884^{h}$ within $0.3$ points
({{eq:handoff-is-a-bottleneck}}). **An architecture with six agents in a chain
starts at $0.884^5 = 0.54$ of a single agent's success**, before anyone has made a
mistake.

Handoff quality is leveraged by the count:

```
  write x read   1 handoff   3 handoffs   11 handoffs
-----------------------------------------------------
         98.0%       97.8%        93.8%         79.9%
         88.3%       88.2%        68.6%         26.0%
         68.0%       67.7%        31.5%          1.5%
```

And the question that decides whether the handoff is worth taking:

```
  specialist edge                3 attempts               1 attempt
                    no handoff   3 handoffs  no handoff  3 handoffs
-------------------------------------------------------------------
               0%        99.7%        69.2%       48.0%       32.3%
               4%        99.7%        68.8%       47.7%       54.2%
              10%        99.7%        69.6%       47.5%       68.1%
```

With retries available, a specialist edge of up to ten points buys **nothing** —
three handoffs sit at about $69\%$ regardless. Without retries, a four-point edge
already makes the split worthwhile. **Specialisation pays only where retries are
unavailable** ({{eq:specialisation-needs-no-retries}}), because retries already
cover the first attempts where a specialist's advantage lives.

And the penalty does not amortise:

```
  work steps   no handoff   3 handoffs     loss
-----------------------------------------------
           6        99.9%        68.7%   -31.2%
          40        99.1%        68.3%   -30.8%
```

Constant in task size. Giving each agent more to do does not make its handoff
cheaper.

The second listing puts four designs on the same twenty-four calls:

```
                          design   completed    calls   agents
--------------------------------------------------------------
        one agent, plain retries       29.2%       24        1
     one agent, forced diversity       49.5%       24        1
              two agents, debate       27.1%       24        2
        two agents, divided work       25.5%       24        2
```

**The single agent with forced approach diversity wins by $22.4$ points over the
debating pair**, at equal cost, with no handoff and no second system to operate.

And it wins even against perfect independence:

```
  shared blind spots   two debating   one, diverse          better
------------------------------------------------------------------
                 95%          17.2%          49.5%       one agent
                 45%          27.2%          49.5%       one agent
                  0%          39.5%          49.5%       one agent
```

At zero shared blind spots — two genuinely independent systems, which no real pair
achieves — the debate reaches $39.5\%$ against the solo agent's $49.5\%$. **The
budget halving costs more than the independence buys**
({{eq:decorrelate-cheaply}}).

The gap widens with budget:

```
  budget    one agent plain    one agent forced     debate     divided
----------------------------------------------------------------------
      16              27.1%               37.0%      27.3%       24.0%
      64              37.5%               85.9%      40.9%       33.0%
```

$9.7$ points at sixteen calls, $45.0$ at sixty-four — because the solo agent gets
the whole increase and each member of the pair gets half.

Three caveats, and the second is substantial. This model gives debate no mechanism
beyond "either instance solves it", while {{cite:du2023debate}}'s instances read
each other's reasoning and revise — a real mechanism with no counterpart here.
**What the listing establishes is that the ensembling half of debate is available
more cheaply from one agent**, so a debate's gains must come from the critique half.
Second, forced diversity is an assumption here rather than a measurement, and it is
the one to test on your own system. Third,
{{eq:multi-agent-ceiling}} still applies: no column touches the capability term.

## 10. Production Considerations

Count your handoffs and treat the count as an exponent. Six agents in a chain is
$0.884^5$ before anything goes wrong.

Give every handoff a schema: goal, tried-and-failed, verified state, open
questions. Three of the four are artefacts {{part:17}} already told you to build.

Carry the deduplication set across the boundary. Losing it recreates
{{ch:ag-loop}}'s repetition failure at every handoff, and it is one of the cheapest
multi-agent bugs to fix.

Measure your write and read reliability directly: hand a state over and ask the
receiver to answer questions only the sender knew. That number, raised to your
handoff count, is your architecture's ceiling.

Check your retry budget before arguing for specialists.
{{eq:specialisation-needs-no-retries}} says the edge is eaten by retries, and three
attempts was enough to eliminate a ten-point edge entirely.

Test forced diversity on your own model before accepting this chapter's conclusion.
Sample a stuck step twice with a diversity instruction and measure whether the
attempts actually differ.

And run {{eq:equal-cost-comparison}} for every architecture claim: same budget,
against the reference single agent, not against a bare loop.

## 11. Common Mistakes

**Adding agents without counting handoffs.** The cost is geometric and the count is
the exponent.

**Handing off prose.** $25.7\%$ against $62.7\%$ at twelve handoffs.

**Losing the failed-attempt list at the boundary.** The receiver re-derives what the
sender already eliminated.

**Arguing for specialists in a system with retries.** A ten-point edge bought
nothing at three attempts.

**Comparing architectures at equal agents rather than equal cost.** Two agents at
full budget each is not a comparison; it is twice the spend.

**Assuming debate's gains are the ensembling.** The ensembling is cheaper from one
agent; the gains, where real, are the critique.

**Amortising a handoff by giving each agent more work.** The penalty is constant in
task size.

## 12. Failure Modes

*Context amnesia at the boundary.* The receiver lacks something the sender knew and
did not think to serialise. The tell is a receiver re-doing work.

*Repeated elimination.* Each agent independently rules out the same wrong approach,
because the deduplication set did not cross.

*Budget dilution.* An n-agent design silently gives each agent $1/n$ of the
attempts, and the loss is attributed to the task being hard.

*Handoff loops.* Two agents passing work back and forth, each finding it
under-specified — {{cite:cemri2025mast}}'s inter-agent misalignment, and a
{{ch:ag-loop}} non-productive cycle with a larger period.

*Latency stacking.* Sequential handoffs are additive in wall clock and each agent
re-pays prefill ({{part:15}}).

## 13. Alternatives

**One agent, forced diversity.** {{sec:9-practical-example}}'s winner, and the
default this chapter recommends.

**One agent with checkpoints.** {{ch:ag-planning}}: the decomposition benefit
without the handoff, since a checkpoint is a boundary the same agent crosses.

**Parallel independent agents.** Where the sub-tasks genuinely do not depend on each
other, there is no handoff and the wall clock halves. This is the strongest
multi-agent case and it is about latency.

**Debate with critique.** {{cite:du2023debate}}'s actual procedure, whose value this
chapter's model does not measure. Worth running against the solo-diverse baseline.

**Two genuinely different models.** The $\rho$ in
{{eq:decorrelate-cheaply}} is smallest across model families, so if you are going to
pay for a second agent, pay for a different one rather than a second instance.

## 14. Evaluation

Report the handoff count as an architecture parameter alongside accuracy.

Measure $w$ and $r$ separately — serialise a state, then quiz the receiver on facts
only the sender had. Their product raised to your handoff count is the ceiling.

Compare at equal *cost*, not equal agents, and against the reference single agent
({{eq:equal-cost-comparison}}).

Measure error covariance between your agents before combining them. At $\rho$ near
one, {{eq:decorrelate-cheaply}} says the pair adds nothing.

And for any debate design, run the ablation that removes the critique — instances
that propose but do not read each other. The difference is what the debate is
actually buying.

## 15. Advanced Concepts

**Learned handoff schemas.** The fields that matter are discoverable: serialise a
state, measure which receiver failures correlate with which omitted fields, and add
those. This is a small supervised problem with abundant traces.
{{maturity:EMERGING}}.

**Shared state instead of messages.** If both agents read and write one structured
store rather than passing messages, $w$ and $r$ collapse toward one and
{{eq:handoff-is-a-bottleneck}}'s exponent stops mattering. The cost is coupling, and
it is the design {{ch:as-state-machines}} develops.

**Measuring diversity directly.** {{sec:7-internal-mechanics}}'s forced-diversity
assumption is measurable as a distance between successive attempts, and the same
measurement would let a system detect when it has stopped varying — which is
{{ch:ag-loop}}'s repetition signal in a more general form.

**Cross-family ensembles.** {{eq:decorrelate-cheaply}}'s $\rho$ is the only
variable, and it is minimised across genuinely different model families rather than
across prompts. Quantifying $\rho$ between families would say what a heterogeneous
ensemble is actually worth. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:rsn-tool-assisted}}'s boundary-crossing arithmetic transfers to handoffs
unchanged, with a worse constant — this chapter is that result at the scale of a
whole agent.

{{ch:ag-tool-calling}}'s schema recommendation reappears as
{{eq:schema-not-story}}, and the leverage is the same: constraining what crosses a
boundary is worth the crossing count.

{{ch:ag-loop}}'s slack argument is why specialisation usually loses, and its
deduplication set is the handoff field that matters most.

{{ch:as-single-agent}}'s {{eq:decorrelation-is-the-variable}} is what this chapter
prices, and {{eq:multi-agent-ceiling}} still bounds every design here.

Ahead: {{ch:as-roles}} asks whether role labels supply any decorrelation;
{{ch:as-state-machines}} develops the shared-state alternative to messages; and
{{ch:as-failures}} returns to {{cite:cemri2025mast}}'s taxonomy.

## 17. Exercises

1. Derive the break-even specialist edge from
   {{eq:specialisation-needs-no-retries}} and $1 - wr$, and compute it for
   $a \in \{1, 2, 3\}$.

2. Add a fifth design to the second listing: two agents that share a deduplication
   set across the handoff. How much of the handoff penalty does that recover?

3. Model shared state instead of messages — both agents read and write one store —
   and show what happens to {{eq:handoff-is-a-bottleneck}}'s exponent.

4. Add a critique mechanism to the debate design: each instance reads the other's
   attempt and revises. How strong must critique be for debate to beat the solo
   diverse agent?

5. Sweep forced diversity below $0.5$ and find where the solo agent stops beating
   the pair. Is that a plausible diversity level for your model?

6. Take a real multi-agent system you have access to, count its handoffs, and
   estimate $wr$ by quizzing receivers. What is its ceiling?

## 18. Interview Questions

1. What does a handoff cost, and how does the cost scale with the number of agents?

2. Your architecture has six agents in a chain. What is your ceiling before anyone
   makes a mistake?

3. When does splitting work across specialists pay?

4. Two agents debate and beat one agent. What comparison would you want to see?

5. Why can one agent decorrelate its own attempts more cheaply than two agents can?

6. What is the single most important field in a handoff message?

## 19. Research Questions

1. What are $w$ and $r$ empirically for real agent handoffs, and how much does a
   schema move them?

2. Which handoff fields matter most, and can the schema be learned from receiver
   failures?

3. How much of {{cite:du2023debate}}'s measured gain is ensembling and how much is
   critique? The ablation is cheap and has not been published.

4. What is $\rho$ between model families, and does a heterogeneous ensemble justify
   its cost where a homogeneous one does not?

5. Can forced approach diversity be made reliable, and is the diversity it produces
   comparable to the diversity between two instances?

## 20. Chapter Summary

A handoff is a lossy write followed by a lossy read, and its cost is geometric in
the count ({{eq:handoff-is-a-bottleneck}}). Holding work fixed, one agent completed
$99.8\%$, four $68.6\%$, twelve $25.7\%$ — fitted by $S(0)\cdot 0.884^h$ with
residuals under $0.3$ points. **The number of agents is an exponent**, and six in a
chain starts at $0.54$ of a single agent before anything goes wrong.

Handoff quality is leveraged by the count ({{eq:schema-not-story}}), so structuring
the payload is worth $h$ times its cost: a schema instead of prose took twelve
agents from $25.7\%$ to $62.7\%$. The field that matters most is the
failed-attempt list, because losing it recreates {{ch:ag-loop}}'s repetition failure
at every boundary.

Specialisation rarely pays for that. A specialist edge of up to ten points bought
*nothing* with three attempts available, because retries already cover the first
attempts where the edge lives ({{eq:specialisation-needs-no-retries}}). **It pays
only where retries are unavailable** — side effects, tight budgets, expensive steps.

And ensembling does not need a second agent. At equal cost, one agent with forced
approach diversity reached $49.5\%$ against two agents debating at $27.1\%$ — and
still led at $39.5\%$ when the debaters shared *zero* blind spots, because halving
the budget costs more than independence buys ({{eq:decorrelate-cheaply}}). The gap
widened with budget, from $9.7$ points to $45.0$.

The honest boundary: this model represents debate as ensembling and omits
{{cite:du2023debate}}'s critique mechanism. **The ensembling half is available more
cheaply from one agent**, so a debate's real gains are evidence about critique — and
the ablation that would separate them is cheap and unpublished.

So: count handoffs, hand off schemas, carry the failure set across, check your retry
budget before invoking specialists, and compare at equal cost against the reference
single agent ({{eq:equal-cost-comparison}}).

## 21. Further Reading

{{cite:du2023debate}} is the strongest case for multiple agents, and worth reading
with {{sec:6-mathematical-foundation}}'s decomposition in hand: the question is
which half of the procedure the gains come from.

{{cite:cemri2025mast}} for the failure taxonomy, and specifically for the
inter-agent misalignment category, whose mechanisms are the handoff pathologies this
chapter measures.

{{ch:rsn-tool-assisted}} for the boundary-crossing arithmetic this chapter inherits,
and {{ch:ag-tool-calling}} for the schema result at the smaller boundary.

{{cite:liu2024agentbench}} for the long-horizon settings where handoff counts get
large enough for {{eq:handoff-is-a-bottleneck}} to dominate.
