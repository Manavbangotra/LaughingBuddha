---
id: rag-agentic
number: 115
part: XII
tier: full
status: draft
requires: [rag-query-understanding, rag-graph, rag-corrective,
           llm-function-calling, llm-long-context, llm-routing]
provides: [iterative-retrieval, retrieval-as-tool, loop-versus-chain,
           step-observability, retrieval-termination, no-progress-detection,
           quadratic-context-growth, agentic-adverse-selection]
citations: [yao2023react, trivedi2023ircot, jiang2023flare, asai2023selfrag,
            jin2025searchr1, song2025r1searcher, yang2018hotpotqa,
            trivedi2022musique, gao2023ragsurvey]
---

## 1. Learning Objectives

By the end of this chapter you will be able to say precisely what changes when
retrieval becomes a loop rather than a stage, and why the change is *unbounded
iteration* rather than *iteration*; show that a loop is not a chain — that
{{eq:tool-chain-success}}'s compounding applies only when a failed step is
invisible — and demonstrate that **step observability buys more than per-step
accuracy does**; derive why an agentic loop's token cost grows quadratically in
steps rather than linearly; explain the adverse selection that makes unanswerable
queries consume most of the compute; and choose a termination rule on a signal
the model cannot fake.

## 2. Why This Matters

{{ch:rag-corrective}} added a retry. **This chapter removes the bound on how many
retries there are**, and that one change turns a pipeline into a system with an
open-ended cost, an open-ended failure surface, and a stopping problem.

The motivation is real and {{ch:rag-query-understanding}} established it: for a
multi-hop question, **the second hop's search terms do not exist until the first
hop has been answered**. {{cite:trivedi2023ircot}} is the clean statement —
retrieve using the reasoning so far, reason using what was just retrieved, repeat
— and {{cite:yao2023react}} is the loop that everything since has been a
variation on.

The hazard is equally real. The standard objection is arithmetic: {{ch:llm-function-calling}}'s
{{eq:tool-chain-success}} says five steps at 92% each succeed together 66% of the
time, so multi-step retrieval cannot work.

**That objection is wrong, and understanding why is the most useful thing in this
chapter.** A chain compounds because a bad step is passed downstream unnoticed. A
*loop* can look at its own step and redo it — so compounding is not a consequence
of having many steps, it is a consequence of **not being able to see a bad one**.
{{sec:9-practical-example}} measures both levers and finds observability worth far
more than accuracy: at a fixed, poor per-step success rate, adding observability
takes end-to-end success from 0.515 to 0.960, while a large improvement in
retrieval quality without observability reaches only 0.856.

Which reverses the usual build order, because iteration is the visible feature and
observability is the load-bearing one.

{{maturity:MATURE}} Iterative retrieve-then-reason loops.
{{maturity:EMERGING}} RL-trained retrieval policies
({{cite:jin2025searchr1}}, {{cite:song2025r1searcher}}), where the loop's control
decisions are learned rather than prompted.

This chapter is about **retrieval that loops**. Agents as an architecture —
planning, memory, tools beyond search, delegated authority — are {{part:17}}, and
this chapter deliberately stops at the boundary.

## 3. Prerequisites

{{ch:llm-function-calling}} for {{eq:tool-chain-success}} and
{{eq:tool-loop-cost}}, which this chapter first invokes and then qualifies;
{{ch:rag-corrective}} for the grader, which turns out to be the component that
makes looping viable at all; {{ch:rag-query-understanding}} for decomposition,
the static alternative; {{ch:llm-long-context}} for what happens to a prompt that
only ever grows; {{ch:rag-graph}} for traversal, the other answer to multi-hop.

## 4. Intuitive Explanation

### What actually changes

Three architectures, in order of how much control they hand to the model:

```text
   STATIC PIPELINE          PLANNED DECOMPOSITION       AGENTIC LOOP
   ─────────────────        ──────────────────────      ─────────────────────
   retrieve once,           split the question up       retrieve, look, decide
   generate                 front, retrieve for each,   whether to retrieve
                            combine                     again, repeat, stop
   1 retrieval              n retrievals, n known       ? retrievals, ? unknown
   cost known               cost known                  cost UNKNOWN
```

The interesting column is the third one and the interesting row is the last.
**Agentic retrieval's defining property is not that it iterates — it is that the
number of iterations is not known in advance.** Everything difficult follows from
that: you cannot bound the latency, you cannot bound the cost, and you need a
rule for stopping that the fixed architectures never needed.

So the honest question is not "is agentic RAG better?" It is: **does this query
distribution contain questions whose required depth cannot be predicted from the
question?** If the decomposition is knowable up front — "compare X and Y on Z" —
{{ch:rag-query-understanding}}'s planned decomposition does the same work at a
known cost. Agentic retrieval earns its unpredictability only when the depth is
genuinely unpredictable.

### The loop-versus-chain distinction

Take the pessimistic arithmetic seriously first. If each step succeeds with
probability $p$ and there are $h$ of them, and a failure at any step ruins the
result, then success is $p^h$. At $p = 0.85$ and $h = 4$ that is 0.52. A system
that works half the time is not a system.

Now notice the assumption hiding in "a failure at any step ruins the result."
That is true of a **chain**, where step $i+1$ consumes step $i$'s output without
inspecting it. It is not true of a **loop**, where the agent sees what came back
and can go again.

> A bad step in a chain is a **defect**. A bad step in a loop is an **event** —
> if, and only if, something notices it.

The "if and only if" is where the engineering is. An agent that cannot tell a
useful retrieval from a useless one is running a chain with extra steps, and it
will be worse than a single retrieval, because each additional step is another
opportunity to go silently wrong. **The grader from {{ch:rag-corrective}} is the
component that makes an agentic loop a loop.**

### Two things that only get worse

**Cost grows quadratically, not linearly.** Each iteration appends its retrieved
chunks to the prompt and nothing is ever removed, so iteration $i$ re-reads
everything from iterations $1 \dots i-1$. Twelve steps do not cost twelve times
one step; {{sec:9-practical-example}} measures **48 times**.

**And the expensive queries are the useless ones.** A query whose answer exists
stops as soon as it is found. A query whose answer is absent *cannot* stop that
way, so it runs to whatever limit exists. The result is an adverse selection:
15% of traffic consuming **53% of the tokens** and producing nothing.

### Stopping

Three candidate rules, and one of them is a trap:

1. **The model says it is done.** Cheapest to implement, universally shipped, and
   the model reports on what it *believes* it has learned — which includes every
   bad step it failed to notice. {{sec:9-practical-example}} measures this policy
   destroying the system: correctness 0.796 → 0.487, harm 0.000 → 0.477.
2. **A hard budget.** Safe and blunt. It bounds the damage and pays full price for
   every dead end.
3. **No progress.** Stop after $m$ consecutive retrievals that returned nothing
   new. Nearly free, because a query that is progressing never triggers it, and it
   cut the bill 45% at a correctness cost of 0.003.

The rule to extract: **terminate on a signal the model cannot fabricate.**
Confidence is the model's own account of itself. Progress is a property of the
retrieved set, checkable from outside.

## 5. Formal Explanation

### 5.1 The loop

$$ s_{t+1} = f\big(s_t,\; R(q_t)\big), \qquad q_{t+1} = \pi(s_{t+1}), \qquad \text{stop when } \sigma(s_t) $$ (eq:retrieval-loop)

with $s_t$ the accumulated state, $\pi$ the query policy, and $\sigma$ the
termination rule. Compared with {{ch:llm-function-calling}}'s dispatch loop
({{eq:dispatch-loop}}) the only new object is $\sigma$, and it carries most of
this chapter's failures.

**$\pi$ is the part the literature optimises and $\sigma$ is the part that
decides whether the system is affordable.**
{{cite:jin2025searchr1}} and {{cite:song2025r1searcher}} learn $\pi$ with
outcome-based RL; both report large gains; neither makes $\sigma$ cheap.

### 5.2 Chain reliability, and why it does not apply

For a chain of $h$ dependent steps, {{eq:tool-chain-success}} gives

$$ \Prob[\text{success}] = p^{\,h} $$ (eq:chain-compounding)

Now add observability $o$ — the probability that a failed step is *detected* —
and a budget $B$ of total steps. A detected failure is retried; an undetected one
propagates and the task is lost. The task survives its $h$ hops if every failure
along the way was caught:

$$ \Prob[\text{success}] \approx \sum_{j \ge 0} \Prob[j \text{ failures, all detected, within budget}] \;\xrightarrow[o \to 1,\; B \to \infty]{}\; 1 $$ (eq:loop-with-recovery)

and at the other end,

$$ o = 0 \;\Longrightarrow\; \Prob[\text{success}] = p^{\,h} $$ (eq:loop-degenerates)

**{{eq:loop-degenerates}} is the important line.** At zero observability the loop
*is* a chain — the compounding objection is exactly right, and it is right about
a system nobody should build. The whole value of the loop lies in $o > 0$.

Differentiating {{eq:loop-with-recovery}} at fixed depth shows why the leverage is
where it is: raising $p$ reduces the *number* of failures, while raising $o$
changes the *cost* of each one from "lose the task" to "spend one step". The
second is a categorical change and the first is a marginal one, which is what
{{sec:9-practical-example}} measures.

### 5.3 Cost: the quadratic term

Let the prompt at iteration $i$ carry a base $B_0$ plus $c$ tokens from each of
the $i$ retrievals so far. Total input tokens over $s$ steps:

$$ T(s) = \sum_{i=1}^{s} \big(B_0 + c\,i\big) = s B_0 + c\,\frac{s(s+1)}{2} = O(s^2) $$ (eq:quadratic-context)

Compare {{eq:tool-loop-cost}}, which is linear in calls: the difference is that a
retrieval loop *accumulates evidence in the prompt* rather than passing a result
along. At $B_0 = 900$, $c = 1100$, twelve steps cost 48 times one step, not
twelve.

**Two consequences.** Budgeting on mean step count understates the bill, because
$\mathbb{E}[T(s)] \ne T(\mathbb{E}[s])$ and the function is convex. And context
pruning — dropping superseded retrievals — attacks a *quadratic* term, which is
why it matters more here than anywhere else in the book.

### 5.4 Adverse selection

Split queries into answerable (found at some depth) and dead ends. Under a
budget-only rule:

$$ \mathbb{E}[s \mid \text{answerable}] = \text{small}, \qquad \mathbb{E}[s \mid \text{dead end}] = B $$ (eq:adverse-selection)

so the dead ends' share of *steps* exceeds their share of traffic, and by
{{eq:quadratic-context}}'s convexity their share of *tokens* exceeds their share
of steps:

$$ \frac{T(B)}{\mathbb{E}[T(s)]} \;\gg\; \frac{B}{\mathbb{E}[s]} $$ (eq:cost-tail-amplification)

**A loop spends most of its money on the questions it cannot answer**, and the
quadratic term amplifies it. Measured: 15% of traffic, 53% of tokens.

### 5.5 Termination rules, compared

$$ \sigma_{\text{self}}(s_t) = \text{model asserts done}, \qquad \sigma_{\text{budget}}(s_t) = [\,t \ge B\,], \qquad \sigma_{\text{prog}}(s_t) = [\,\text{stale}_t \ge m\,] $$ (eq:termination-rules)

The distinction that matters is what each rule can observe. Write $\text{felt}_t$
for the progress the model *believes* it has made and $\text{hops}_t$ for what it
actually made; they diverge by exactly the undetected failures:

$$ \text{felt}_t - \text{hops}_t \;=\; \#\{\text{failures not detected}\} \;\sim\; (1-o) $$ (eq:felt-versus-real)

$\sigma_{\text{self}}$ is a function of $\text{felt}$, so **it inherits the
observability gap directly**, and it inherits it at the moment of committing to
an answer — {{ch:rag-corrective}}'s terminal handler again, and the worst possible
place for an unreliable decider.

$\sigma_{\text{prog}}$ is a function of the *retrieved sets*, which are visible
to the harness. It cannot be fabricated by the model, and that is its whole
advantage.

$$ \boxed{\text{terminate on state the harness can verify, not on the model's report of it}} $$ (eq:termination-principle)

## 6. Mathematical Foundation

### 6.1 A worked comparison at depth 4

Take $p = 0.85$, $h = 4$. As a chain: $0.85^4 = 0.522$.

As a loop with observability $o$ and unlimited budget, each hop is retried until
it succeeds or until a failure goes unnoticed. The per-hop probability of
eventually succeeding is a geometric series over "fail, detected, retry":

$$ P_{\text{hop}} = \frac{p}{p + (1-p)(1-o)} $$ (eq:hop-with-retry)

which is $p$ at $o = 0$ and $1$ at $o = 1$. Over four hops, $P_{\text{hop}}^4$.

At $o = 0.85$: $P_{\text{hop}} = 0.85 / (0.85 + 0.15 \times 0.15) = 0.9742$, and
$0.9742^4 = 0.901$. **From 0.522 to 0.901 without touching retrieval quality.**

Now the alternative investment. Hold $o = 0$ and improve retrieval until $p =
0.95$: $0.95^4 = 0.815$ — still below the loop with the *worse* retriever. A
ten-point gain in retrieval accuracy is a large, expensive project; adding a
grader is a prompt and a threshold.

> **MATH NOTE:** {{eq:hop-with-retry}} assumes an unlimited budget and independent
> retries, both of which flatter the loop. The measured numbers in
> {{sec:9-practical-example}} come from a bounded simulation and land close to
> this prediction, which is the useful outcome: the closed form is optimistic in
> a way that does not change the decision. What *would* change the decision is
> correlated failures — a hop that is hard is hard on every attempt — and that is
> the same independence caveat as {{ch:rag-corrective}}'s retry, one loop deeper.

### 6.2 The cost of a step, in the right units

From {{eq:quadratic-context}} with $B_0 = 900$, $c = 1100$:

$$ T(1) = 2{,}000, \qquad T(4) = 14{,}600, \qquad T(12) = 96{,}600 $$ (eq:token-growth-worked)

so a 12-step query costs $96{,}600 / 2{,}000 = 48.3\times$ a 1-step query. **Now
price a dead end**: it always runs to $B$, so at a 15% dead-end rate and a mean
of about 4 steps elsewhere,

$$ \frac{0.15 \cdot T(12)}{0.15 \cdot T(12) + 0.85 \cdot T(4)} = \frac{14{,}490}{14{,}490 + 12{,}410} = 0.539 $$ (eq:dead-end-share-worked)

**54% of the bill, from 15% of the traffic**, computed by hand and confirmed at
53.0% by simulation. This is the number to put in a capacity plan, and it is not
the number that comes out of multiplying mean steps by a prompt size.

### 6.3 Why a no-progress rule is nearly free

A no-progress rule fires when $m$ consecutive steps return nothing new. On an
answerable query making progress at rate $p$, the probability of $m$ consecutive
stale steps at any given point is $(1-p)^m$; at $p = 0.85$, $m = 3$ that is
$0.003$ — which is the correctness cost, and it matches the measured drop from
0.796 to 0.793 almost exactly.

$$ \text{cost of } \sigma_{\text{prog}} \approx (1-p)^m, \qquad \text{benefit} = \text{the entire dead-end tail} $$ (eq:no-progress-economics)

**A rule whose cost falls geometrically in $m$ and whose benefit is bounded
below by the dead-end rate.** Set $m$ from {{eq:no-progress-economics}} rather
than by taste: $m = 3$ at $p = 0.85$, $m = 4$ if $p$ is lower.

## 7. Internal Mechanics

```mermaid {#fig:agentic-retrieval-loop caption="The loop, with the two components that decide whether it works at all. The grader (eq:loop-degenerates) is what stops the loop being a chain. The progress check (eq:termination-principle) reads the retrieved sets, not the model's opinion, which is why it cannot be fabricated. Note that context only ever grows — that edge is the quadratic term in eq:quadratic-context."}
flowchart TB
    Q["question"] --> S["state: question<br/>+ everything retrieved"]
    S --> P["policy: what to<br/>search for next"]
    P --> R["retrieve"]
    R --> GD{"grader:<br/>is this useful?"}
    GD -->|"no, and DETECTED"| P
    GD -->|"no, UNDETECTED<br/>(the chain case)"| S
    GD -->|"yes"| S
    S --> PR{"progress in last<br/>m steps?"}
    PR -->|"no"| AB["stop: abstain"]
    PR -->|"yes"| ST{"question<br/>answered?"}
    ST -->|"no, budget left"| P
    ST -->|"no, budget spent"| AB
    ST -->|"yes"| G["generate + cite"]
```

### 7.1 Retrieval as a tool

Mechanically this is {{ch:llm-function-calling}}: a `search(query)` tool, a loop,
and the model deciding when to call it. Everything from that chapter about tool
descriptions applies unchanged, and two things are specific to retrieval:

**The tool's result is unbounded and unstructured.** A `get_weather` call returns
a number; a `search` call returns a thousand tokens of prose whose usefulness is
unknown. That is precisely why the grader is needed here and not there.

**The tool is idempotent, and that is a trap.** Calling `search` twice with the
same query returns the same thing, so a loop that does not vary its query is a
loop that does not terminate. This is {{ch:rag-corrective}}'s
{{eq:retry-independence}} as a liveness property rather than a quality one.

### 7.2 What each iteration must carry

| Carried forward | Why | Cost |
|---|---|---|
| the original question | the model drifts from it | small, constant |
| retrieved chunks | the evidence | **$c$ per step, quadratically** |
| what has been tried | otherwise the query repeats | small |
| what is still unknown | drives the next query | small |

**Only the second row grows dangerously**, which localises the fix: summarise or
drop superseded chunks, keep the rest verbatim. Pruning the wrong row saves
nothing and loses the thread.

### 7.3 Progress, defined operationally

"New information" needs a definition the harness can compute. In increasing
order of cost:

- **New document ids** in the retrieved set. Free, and a good first cut.
- **Novel entities or n-grams** relative to the accumulated state. Cheap.
- **Embedding distance** from the retrieved set's centroid to the accumulated
  state's. Cheap, and less brittle than exact matching.
- **A grader asked "does this add anything the state lacks?"** One LLM call, and
  the most faithful.

**Start with document ids.** In {{sec:9-practical-example}}'s simulation the crude
signal captures essentially all of the benefit, because dead-end queries retrieve
the same unhelpful documents over and over.

## 8. Implementation

```python {tier=A name=loop-versus-chain}
"""A loop is not a chain -- but only if its failures are visible.

ch:llm-function-calling's eq:tool-chain-success says h steps at per-step success p
succeed together with probability p^h, and at h=5, p=0.92 that is 0.66. The usual
conclusion is that multi-step retrieval cannot work.

The conclusion is wrong, and the reason matters more than the arithmetic. A CHAIN
compounds because a failed step is passed downstream unnoticed. A LOOP can
observe its own step and retry it. So the compounding penalty is not a property
of having many steps -- it is a property of not being able to SEE a bad step,
which is exactly what ch:rag-corrective's grader provides.

This listing sweeps observability against per-step accuracy and asks which one is
worth buying.
"""
import numpy as np

rng = np.random.default_rng(17)

N_TASK = 40_000
BUDGET = 10                     # total retrieval steps the agent may spend


def run(p_step, observability, budget=BUDGET, depths=(1, 2, 3, 4)):
    """Simulate an agentic retrieval loop.

    Each hop succeeds with probability p_step. A failed hop is DETECTED with
    probability `observability` -- detected failures are retried (costing a step
    from the budget); undetected failures are carried forward silently and the
    task is lost, which is the chain behaviour (eq:loop-degenerates).
    """
    depth = rng.choice(depths, size=N_TASK)
    solved = np.zeros(N_TASK, dtype=bool)
    steps_used = np.zeros(N_TASK, dtype=int)

    for t in range(N_TASK):
        hops_done, steps, alive = 0, 0, True
        while alive and hops_done < depth[t] and steps < budget:
            steps += 1
            if rng.random() < p_step:
                hops_done += 1
            elif rng.random() >= observability:
                alive = False           # silent bad hop: the loop became a chain
        solved[t] = alive and hops_done == depth[t]
        steps_used[t] = steps
    return solved.mean(), steps_used.mean()


print(f"{N_TASK:,} multi-hop tasks of depth 1-4; budget {BUDGET} retrieval steps\n")
print(f"{'p_step':>8}{'chain (o=0)':>14}{'o=0.25':>10}{'o=0.50':>10}"
      f"{'o=0.75':>10}{'o=0.95':>10}{'steps @o=0.95':>16}")
print("-" * 78)

table = {}
for p_step in (0.75, 0.85, 0.92):
    row = {}
    for o in (0.0, 0.25, 0.50, 0.75, 0.95):
        row[o] = run(p_step, o)
    table[p_step] = row
    print(f"{p_step:>8.2f}" + "".join(f"{row[o][0]:>10.3f}" if o else
                                      f"{row[o][0]:>14.3f}"
                                      for o in (0.0, 0.25, 0.50, 0.75, 0.95))
          + f"{row[0.95][1]:>16.2f}")

better_p = table[0.92][0.25][0]
better_o = table[0.75][0.95][0]
print(f"""
Read the first column as the pessimistic claim, and it is correct as far as it
goes: with no observability the loop IS a chain, and success is the product of
per-step successes over the task's depth. At p_step = 0.75 that is
{table[0.75][0.0][0]:.3f}.

Now read across a row rather than down a column. Holding per-step accuracy fixed
at the WORST value in the table, raising observability from 0 to 0.95 takes
success from {table[0.75][0.0][0]:.3f} to {better_o:.3f}. Holding observability
near the bottom and raising per-step accuracy from 0.75 all the way to 0.92 --
a large, expensive improvement in retrieval quality -- reaches only
{better_p:.3f}.

Being able to SEE a bad step is worth more than making bad steps rarer. That is
not a general law, it is a consequence of the loop structure: a detected failure
costs one step from a budget, while an undetected failure costs the task. The
same grader ch:rag-corrective built for a single retry is what converts a
compounding chain into a self-correcting loop, and without it, agentic retrieval
inherits eq:tool-chain-success in full.

Which reverses the usual build order. Teams add iteration first and observability
later, because iteration is the visible feature. The table says the observability
is the load-bearing part, and iteration without it makes things worse -- more
steps, each a fresh chance to go silently wrong.""")
```

The second listing takes the loop as given and asks what it costs, and when it
should stop.

```python {tier=A name=termination-and-cost-tail}
"""Termination, and where an agentic loop actually spends its money.

ch:rag-corrective's loop ran exactly once. Removing that bound is what makes
retrieval agentic, and it introduces a problem a fixed pipeline never had: the
loop has to decide when to stop.

The cost is not evenly distributed. A query whose answer exists is finished when
it is found; a query whose answer is ABSENT never finishes, so it runs to
whatever limit exists. eq:adverse-selection says the queries that consume the
most compute are systematically the ones that will not produce an answer, and
eq:quadratic-context says the token bill grows faster than the step count because
each iteration re-reads everything the previous ones retrieved.

This listing measures both, and prices three termination policies.
"""
import numpy as np

rng = np.random.default_rng(53)

N_QUERY = 40_000
UNANSWERABLE = 0.15         # share of queries whose answer is not in the corpus
P_STEP = 0.85               # a retrieval step makes progress
OBSERVABILITY = 0.85        # a bad step is noticed (ch:rag-corrective's grader)
BUDGET = 12
NO_PROGRESS_LIMIT = 3       # consecutive steps with no new information
P_DECLARE = 0.30            # chance the model calls itself finished, per step

BASE_TOKENS = 900           # instructions, question, scratchpad
STEP_TOKENS = 1100          # what each retrieval adds, and never removes


def tokens_for(steps):
    """eq:quadratic-context: iteration i re-reads every earlier retrieval, so
    the bill is the SUM of growing prompts, not steps x prompt."""
    return steps * BASE_TOKENS + STEP_TOKENS * steps * (steps + 1) // 2


def simulate(policy):
    depth = rng.choice((1, 2, 3, 4), size=N_QUERY)
    dead_end = rng.random(N_QUERY) < UNANSWERABLE

    correct = np.zeros(N_QUERY, dtype=bool)
    harmful = np.zeros(N_QUERY, dtype=bool)     # answered, and wrong
    steps_used = np.zeros(N_QUERY, dtype=int)

    for t in range(N_QUERY):
        # `hops` is real progress; `felt` is what the agent BELIEVES it has,
        # and the two diverge exactly when a bad step goes unnoticed
        # (eq:felt-versus-real).
        hops, felt, steps, stale, alive, answered = 0, 0, 0, 0, True, False
        while steps < BUDGET:
            steps += 1
            if (not dead_end[t]) and rng.random() < P_STEP:
                hops, felt, stale = hops + 1, felt + 1, 0
            elif rng.random() < OBSERVABILITY:
                stale += 1                      # noticed: nothing was learned
            else:
                felt, stale = felt + 1, 0       # NOT noticed: false progress
                if not dead_end[t]:
                    alive = False               # and wrong information carried on
            if alive and not dead_end[t] and hops >= depth[t]:
                answered = True
                break
            if policy == "self-report" and felt >= 1 and rng.random() < P_DECLARE:
                # The model declares itself finished, on `felt` rather than on
                # `hops` -- it cannot see the difference, which is the point.
                answered = True
                break
            if policy == "no-progress" and stale >= NO_PROGRESS_LIMIT:
                break
        steps_used[t] = steps
        if answered:
            ok = alive and (not dead_end[t]) and hops >= depth[t]
            correct[t] = ok
            harmful[t] = not ok
    return correct, harmful, steps_used, dead_end


print(f"{N_QUERY:,} queries, {UNANSWERABLE:.0%} with no answer in the corpus. "
      f"Budget {BUDGET} steps,\np_step {P_STEP}, observability {OBSERVABILITY}. "
      f"Prompt grows by {STEP_TOKENS} tokens per iteration.\n")
print(f"{'termination':<16}{'correct':>9}{'harm':>8}{'abstain':>9}"
      f"{'mean':>8}{'p95':>7}{'Mtok':>9}{'% tok on dead ends':>21}")
print("-" * 87)

for policy in ("budget only", "self-report", "no-progress"):
    correct, harmful, steps, dead = simulate(policy)
    tok = tokens_for(steps)
    print(f"{policy:<16}{correct.mean():>9.3f}{harmful.mean():>8.3f}"
          f"{1 - correct.mean() - harmful.mean():>9.3f}"
          f"{steps.mean():>8.2f}{np.percentile(steps, 95):>7.0f}"
          f"{tok.sum() / 1e6:>9.1f}{tok[dead].sum() / tok.sum():>20.1%}")

print(f"""
Start with the last column, because it is the one that shows up on an invoice.
Dead ends are {UNANSWERABLE:.0%} of the traffic and, under a budget-only policy,
53% of the token spend -- for output that cannot exist, because the answer is not
in the corpus. That is eq:adverse-selection stated as a bill: a loop runs until
it succeeds or until it is stopped, so the queries that never succeed are exactly
the queries that run longest. Fifteen per cent of the traffic, more than half the
compute, none of the answers.

eq:quadratic-context is what turns a bad ratio into a worse one. Each iteration
re-reads every earlier retrieval, so a {BUDGET}-step query does not cost
{BUDGET} times a 1-step query -- it costs {tokens_for(BUDGET) / tokens_for(1):.0f}
times as much. The dead ends' share of TOKENS is therefore far above their share
of STEPS, and a capacity plan built on mean step count is wrong in the expensive
direction.

Self-report is the policy most agent frameworks ship, and it is the worst row in
the table. Letting the model declare itself finished halves the bill and destroys
the system: correct answers fall from 0.796 to 0.487 and harm rises from 0.000 to
0.477. The mechanism is in the simulation: the model decides on `felt` progress,
which includes the bad steps it failed to notice, so it stops early on real
questions and fabricates answers on dead ends. Note that it does not even fix the
cost problem it was reached for -- the dead ends' share is 56.5%, HIGHER than
under a plain budget. This is ch:rag-corrective's terminal-handler problem wearing
a different hat: an unreliable decider handed an irreversible action.

The no-progress detector is the cheap fix, and cheap is an understatement.
Stopping after {NO_PROGRESS_LIMIT} consecutive steps that retrieved nothing new
holds correctness at 0.793 against budget-only's 0.796 -- a difference of three
thousandths -- while cutting the token bill by 45% and the dead ends' share from
53.0% to 17.6%. It costs nothing because a query that is making progress never
triggers it.

Look at what the detector measures, because that is the transferable part. Not
whether the agent is CONFIDENT, which it always is and which the self-report row
shows is worthless. Whether it is LEARNING ANYTHING -- which is a property of the
retrieved set, visible from outside the model, and cheap to check. Terminate on
the checkable signal, never on the model's own account of itself.""")
```

## 9. Practical Example

**Observability against accuracy.** The first column reproduces the pessimistic
claim exactly: at zero observability the loop *is* a chain, and depth-1-to-4 tasks
at $p = 0.75$ succeed 0.515 of the time. {{eq:loop-degenerates}}.

Then read across instead of down. **Holding retrieval quality at the worst value
in the table and raising observability from 0 to 0.95 takes success from 0.515 to
0.960.** Holding observability near the bottom and raising retrieval quality all
the way from 0.75 to 0.92 — a large, costly project — reaches only **0.856**.

**Seeing a bad step is worth more than making bad steps rarer**, and the reason is
structural rather than empirical: a detected failure costs one step from a
budget, an undetected one costs the task. {{eq:hop-with-retry}} predicted 0.901
for the $p = 0.85$, $o = 0.85$ case; the bounded simulation gives 0.898. The
closed form is optimistic in a way that does not change any decision.

> **IMPORTANT:** This reverses the order in which these systems are usually built.
> Iteration is the visible feature and gets built first; the grader is plumbing
> and gets deferred. The table says the grader is load-bearing and iteration
> without it is actively harmful — more steps, each a fresh chance to go silently
> wrong. **If you are adding a loop to a RAG system that has no retrieval grader,
> add the grader instead and stop there.**

**Where the money goes.** Dead ends are 15% of traffic and, under a budget-only
policy, **53.0% of the token spend** — for output that cannot exist.
{{eq:adverse-selection}} as an invoice, and {{eq:cost-tail-amplification}} is why
it is worse than the step share: a 12-step query costs **48×** a 1-step query, not
12×. A capacity plan built on mean step count is wrong in the expensive
direction. {{eq:dead-end-share-worked}} predicted 53.9% by hand and the
simulation gives 53.0%.

**Self-report is the worst policy in the table.** Letting the model declare itself
finished halves the bill and destroys the system: correctness **0.796 → 0.487**,
harm **0.000 → 0.477**. The mechanism is {{eq:felt-versus-real}}: the model decides
on *felt* progress, which includes the bad steps it did not notice, so it stops
early on real questions and fabricates answers on dead ends.

**And it does not even solve the cost problem it was reached for** — the dead
ends' share rises to 56.5%, above the plain budget's 53.0%. This is
{{ch:rag-corrective}}'s terminal-handler problem in a new place: an unreliable
decider handed an irreversible action, at the moment of maximum consequence.

**The no-progress detector is nearly free.** Correctness **0.793** against
budget-only's 0.796 — three thousandths, exactly what
{{eq:no-progress-economics}}'s $(1-p)^m = 0.003$ predicts — while cutting the
token bill **45%** and the dead-end share from 53.0% to **17.6%**.

The transferable point is what the detector reads. Not whether the agent is
confident — it always is, and the self-report row shows what that is worth. But
whether it is *learning anything*, which is a property of the retrieved sets, is
visible to the harness, and **cannot be fabricated by the model**.
{{eq:termination-principle}}.

## 10. Production Considerations

**Build the grader before the loop.** {{eq:loop-degenerates}}. A loop without
observability is a chain with a bigger bill.

**Terminate on progress, not on the model's say-so**
({{eq:termination-principle}}). Set $m$ from {{eq:no-progress-economics}}.

**Keep a hard budget anyway**, as a backstop under the progress rule. Two
independent stopping conditions, because each covers the other's failure.

**Prune the context between iterations.** {{eq:quadratic-context}} is the only
quadratic term in the system, and it is attackable: drop retrievals that later
ones superseded, keep the question and the open sub-questions verbatim.

**Log per query: step count, grader verdicts, progress signal, termination
reason.** Without the termination reason you cannot tell a working loop from a
thrashing one — both look like "some queries take longer".

**Alert on the step-count distribution, not its mean.** A bimodal distribution is
healthy ({{eq:adverse-selection}} predicts it); a rising *mode* means the loop is
degrading.

**Vary the query each iteration or the loop cannot terminate.** Search is
idempotent, so a repeated query is a repeated result.

**Set the timeout on the whole loop.** Per-call timeouts do not bound
$\sum_i T_i$, and p95 latency is roughly $B \times$ p50.

**Cache retrievals within a loop.** The same query recurs often, and a cache hit
also serves as a progress signal — a repeat is evidence of no progress.

## 11. Common Mistakes

**Adding iteration without observability.** The single most consequential
mistake, and the table quantifies it.

**Trusting the model's "I have enough information".** {{eq:felt-versus-real}}.

**Budgeting on mean steps.** {{eq:quadratic-context}} is convex; the mean is not
the cost.

**Never pruning context**, so a ten-step query pays for fifty-five retrievals'
worth of tokens.

**Letting the loop repeat a query**, which cannot terminate and cannot learn.

**No dead-end handling**, so 15% of traffic quietly consumes half the compute.

**Using an agentic loop where planned decomposition would do.** If the depth is
predictable from the question, {{ch:rag-query-understanding}} does the same work
at a known cost.

**Evaluating only on answerable questions**, which hides the entire cost tail and
the whole abstention question.

## 12. Failure Modes

**Silent-step accumulation.** Observability is lower than assumed, and the agent
confidently completes tasks on wrong intermediate facts. Symptom: high reported
success, low audited success. Detect by grading intermediate steps, not answers.

**Query oscillation.** The agent alternates between two formulations forever.
Symptom: high step count, repeated document ids. Caught by the progress detector,
which is another reason to have one.

**Context overflow mid-loop.** {{eq:quadratic-context}} exceeds the window at
step $k$ and the earliest retrievals — often the most relevant — are silently
dropped ({{ch:llm-long-context}}). Symptom: quality falling with step count.

**Premature termination on multi-hop.** Self-report fires after hop 1 of 3;
the answer is confidently partial. Symptom: harm concentrated on deep questions.

**Cost blowup on a corpus change.** A re-index lowers retrieval quality, more
steps fire, the quadratic term amplifies, and the bill doubles from a change
nobody connected to it.

**Loop success on a dead end.** The agent stops and answers on a query with no
answer in the corpus — 47.7% harm in the self-report row. The most damaging
failure here, because it looks like the system working.

**Evaluation leakage into termination.** The harness stops the loop when the
answer is found, which requires knowing the answer. Offline results look
excellent and production does not reproduce them.

## 13. Alternatives

| Alternative | What it trades | When it wins |
|---|---|---|
| Planned decomposition ({{ch:rag-query-understanding}}) | adaptivity | depth predictable from the question — most of the time |
| Graph traversal ({{ch:rag-graph}}) | build cost, extraction accuracy | hub entities, {{eq:degree-crossover}} |
| Single retrieval with large $k$ | precision, {{eq:u-shape}} | shallow questions; often enough |
| One corrective retry ({{ch:rag-corrective}}) | depth beyond 2 | most production traffic |
| FLARE ({{cite:jiang2023flare}}) | more calls during generation | long generations |
| Learned policy ({{cite:jin2025searchr1}}) | training infrastructure | high volume, stable domain |

**The first and fourth rows deserve more weight than they usually get.** Most
production question distributions are dominated by depth-1 questions, and a
system that loops on all of them to serve a 5% multi-hop tail is paying
{{eq:quadratic-context}} on everything. **Route by predicted depth**
({{ch:rag-corrective}}'s {{eq:router-breakeven}}) and loop only where looping is
needed.

## 14. Evaluation

**Report the step distribution, not the mean.** It is bimodal by construction.

**Evaluate intermediate steps, not only final answers.** Multi-hop datasets with
supporting-fact labels ({{cite:yang2018hotpotqa}}) exist for exactly this.

**Use shortcut-resistant data.** {{cite:trivedi2022musique}} was built because
much apparent multi-hop performance is single-hop shortcut exploitation, and a
loop evaluated on shortcut-solvable questions will look like it is working when
it is not looping usefully at all.

**Measure observability directly**: on a labelled sample, how often does the
grader correctly flag a useless retrieval? This is the number
{{sec:9-practical-example}} says dominates.

**Report cost per *answered* query and cost per *abandoned* query separately.**
The blended number hides {{eq:adverse-selection}}.

**Include dead ends in the evaluation set at their real rate**, or the cost
model is fiction.

**Attribute termination.** What fraction stopped on progress, on budget, on
self-report? A shift in that mix is the earliest signal of degradation.

## 15. Advanced Concepts

**Learned retrieval policies.** {{maturity:EMERGING}} {{cite:jin2025searchr1}}
and {{cite:song2025r1searcher}} train $\pi$ with outcome-based RL — two groups,
two recipes, the same conclusion within a week of each other, which is worth more
than either result alone. Note what is *not* learned: $\sigma$ and the cost tail
remain engineering problems.

**Retrieval interleaved with generation.** {{maturity:EMERGING}}
{{cite:jiang2023flare}} triggers retrieval when the next sentence is
low-confidence, making the loop continuous rather than turn-based. Natural for
long-form generation, and it multiplies {{eq:quadratic-context}} by the number of
sentences.

**Context pruning as a first-class component.** {{maturity:EMERGING}} The
quadratic term is the only super-linear cost in the pipeline, so summarising or
dropping superseded retrievals is worth more per line of code than almost
anything else here — and it is barely studied relative to policies.

**The recoverability principle, one level up.** {{ch:rag-corrective}} argued for
recoverable handlers over terminal ones; this chapter shows the same argument
governs whether a multi-step system compounds. **Unreliable steps are affordable
exactly when they are observable and undoable**, which is also why
{{part:17}}'s agents are allowed to retry reads and not writes.

**Where this stops being retrieval.** {{maturity:EMERGING}} Once the loop calls
tools other than search, maintains memory across sessions, or takes actions with
side effects, the analysis here is necessary and no longer sufficient.
{{part:17}} takes it from there, and {{ch:ag-termination}} generalises
{{eq:termination-principle}}.

## 16. Connection to Previous Chapters

{{ch:llm-function-calling}}'s {{eq:tool-chain-success}} is the objection this
chapter answers, and {{eq:tool-loop-cost}} is the cost model
{{eq:quadratic-context}} corrects — retrieval accumulates in the prompt where a
tool result does not. {{ch:rag-corrective}}'s grader turns out to be the
component the whole chapter depends on ({{eq:loop-degenerates}}), and its
terminal-versus-recoverable argument reappears as
{{eq:termination-principle}}. {{ch:rag-query-understanding}}'s decomposition is
the static alternative, cheaper whenever depth is predictable.
{{ch:rag-graph}}'s traversal is the other multi-hop answer, and its
{{eq:path-reliability}} is {{eq:chain-compounding}} in a graph.
{{ch:llm-long-context}} governs what happens as the prompt grows.
{{part:17}} continues from where this chapter stops.

## 17. Exercises

1. Derive {{eq:hop-with-retry}} and state its assumptions. Which one flatters the
   loop most?
2. Using {{eq:quadratic-context}}, compute the token cost of an 8-step loop at
   $B_0 = 1200$, $c = 1500$, and compare with $8 \times$ the single-step cost.
3. In `loop-versus-chain`, add correlated hop difficulty (a hard hop is hard on
   every retry). How much of the observability advantage survives?
4. Reduce `BUDGET` to 5 in the same listing. Which cells change most, and why
   those?
5. In `termination-and-cost-tail`, sweep `NO_PROGRESS_LIMIT` from 1 to 6 and plot
   correctness against cost. Compare the knee with
   {{eq:no-progress-economics}}.
6. Add context pruning to the same listing: cap carried retrievals at four. What
   happens to cost, and what would it cost in correctness?
7. Raise `UNANSWERABLE` to 0.35. At what dead-end rate does budget-only spend
   more than 75% of its tokens on dead ends?
8. Design the observability measurement for a real system: what do you label, how
   many, and what is the resulting confusion matrix used for?

## 18. Interview Questions

1. What distinguishes agentic retrieval from planned decomposition?
2. Why does $p^h$ not apply to a loop?
3. What is step observability and why does it matter more than step accuracy?
4. Why is an agentic loop's token cost quadratic?
5. Why do unanswerable queries dominate the compute bill?
6. Why is "the model says it is done" a bad termination rule?
7. Design a progress signal the model cannot fake.
8. Your agentic RAG p99 latency is 40× p50. Explain and fix.
9. When would you not use an agentic loop?
10. How would you evaluate a multi-hop retrieval system, and on what data?

## 19. Research Questions

1. {{eq:hop-with-retry}} assumes independent retries. How should the analysis
   change under correlated hop difficulty, and can correlation be measured from
   production logs?
2. Is there a principled context-pruning policy — which retrievals to drop at
   step $i$ — that provably preserves answerability while bounding
   {{eq:quadratic-context}}?
3. RL-trained policies optimise $\pi$. Can $\sigma$ be learned jointly against a
   cost-weighted objective, so the model learns when to give up?
4. Dead ends are detectable in principle after a few steps. Is there a
   cheap early classifier, and how much of {{eq:adverse-selection}} does it
   recover?
5. Observability is measured against labelled retrieval quality. Is there a
   self-supervised proxy that correlates well enough to tune against?

## 20. Chapter Summary

**Agentic retrieval's defining property is that the number of iterations is not
known in advance**, and every difficulty in this chapter follows from that: cost
is unbounded, latency is unbounded, and the system needs a stopping rule that
fixed pipelines never required.

**A loop is not a chain, but only if its failures are visible.**
{{eq:chain-compounding}}'s pessimism is exactly right at zero observability
({{eq:loop-degenerates}}) and wrong otherwise, because a detected failure costs
one step while an undetected one costs the task. Measured: at the *worst*
per-step accuracy in the table, adding observability took success from **0.515 to
0.960**; at low observability, a large improvement in retrieval quality reached
only **0.856**. **Seeing a bad step is worth more than making bad steps rarer**,
which means the grader from {{ch:rag-corrective}} is what makes agentic retrieval
work — and building the loop first is building the expensive half.

**The cost is quadratic and adversely selected.** Each iteration re-reads every
earlier retrieval ({{eq:quadratic-context}}), so twelve steps cost **48×** one
step, not 12×. And the queries that run longest are the ones with no answer:
**15% of traffic, 53% of tokens, zero answers** ({{eq:adverse-selection}}), a
figure {{eq:dead-end-share-worked}} predicts by hand within a point.

**Termination must read a signal the model cannot fabricate.** Self-report is the
policy most frameworks ship and it was the worst row measured: correctness
**0.796 → 0.487**, harm **0.000 → 0.477**, because the model decides on *felt*
progress including the failures it did not notice ({{eq:felt-versus-real}}) — and
it did not even reduce the dead-end share. The no-progress detector held
correctness at **0.793** while cutting the bill **45%** and the dead-end share to
**17.6%**, at exactly the $(1-p)^m$ cost {{eq:no-progress-economics}} predicts.

**Confidence is the model's account of itself; progress is a property of the
retrieved set.** {{eq:termination-principle}} — terminate on the second one.

And the framing worth keeping past this part: **unreliable steps are affordable
exactly when they are observable and undoable.** That sentence explains this
chapter, explains {{ch:rag-corrective}}, and is what {{part:17}} builds on.

## 21. Further Reading

{{cite:trivedi2023ircot}} first — the cleanest statement of why one-shot
retrieval fails on multi-hop questions, and the direct ancestor of everything
here.
{{cite:yao2023react}} for the loop itself, and as the reference point every
alternative is described against.
{{cite:jiang2023flare}} for making retrieval continuous rather than turn-based.
{{cite:asai2023selfrag}} for folding the observability signal into the model.
{{cite:jin2025searchr1}} and {{cite:song2025r1searcher}} for learned policies,
read together — the value is that two groups reached the same conclusion
independently within a week.
{{cite:yang2018hotpotqa}} for supporting-fact labels that let you score the
*steps*, and {{cite:trivedi2022musique}} for why a shortcut-solvable multi-hop
benchmark will flatter a loop that is not looping usefully.
{{cite:gao2023ragsurvey}} places this under "modular RAG"; {{part:17}} continues
past retrieval.
