---
id: ag-what-is-an-agent
number: 153
part: XVII
tier: full
status: draft
requires: [per-step-error-compounding, tokens-as-working-memory,
           coverage-selection-decomposition]
provides: [control-location, tail-mass-decides, autonomy-costs-variance,
           path-explosion, statistical-correctness-argument,
           failure-to-call-ratio]
citations: [yao2023react, liu2024agentbench, zhou2024webarena,
            schick2023toolformer, shinn2023reflexion, sprague2024tocot,
            brown2024monkeys, huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the one property that
distinguishes an agent from a workflow, and why it is binary rather than a
spectrum; compute the tail mass at which an agent's success overtakes a router's,
and explain why the cost-efficiency crossover is somewhere else entirely; state
the ratio that decides between them for your product; describe an agent's cost as
a distribution rather than a number, and say what a step budget buys; and explain
why a workflow's correctness argument is coverage and an agent's cannot be.

## 2. Why This Matters

"Agent" is the most product-marketed word in this book, and most writing about it
describes frameworks. A part organised around frameworks would be obsolete before
it rendered, so this one is organised around a single structural property, and
this chapter is where that property gets named.

**The property is who chooses the next step.** In a workflow, the control flow is
written by a human and the model fills in slots. In an agent, the model chooses
each action, including when to stop. That is not a spectrum — it is a fact about
where control lives — and it predicts everything else: a workflow fails where its
author did not anticipate a case, and an agent fails by looping, wandering, or
stopping early.

This matters because the two have *different correctness arguments*, and teams
routinely apply the wrong one. A workflow can be verified by enumerating its
paths; {{sec:9-practical-example}} counts an agent's paths at horizon 12 and gets
$531{,}441$, of which a $500$-case test suite covers $0.09\%$. That is not a
testing-effort problem. It is a statement about the size of a set.

It also matters because the decision between them is usually made on taste and is
actually a measurement. {{sec:9-practical-example}} finds an agent overtaking a
six-branch router on success at about $20\%$ tail mass — the share of requests
that do not fit any shape somebody enumerated — and *never* overtaking it on
success per model call over the whole range swept. Those two crossovers being in
different places is the practically important part: which one governs depends on
whether a failed task or a model call is the expensive thing in your product, and
that ratio differs by orders of magnitude between a chat assistant and a coding
agent.

And it matters because expectations are miscalibrated. {{cite:zhou2024webarena}}
built an environment of real, functional websites and measured the best GPT-4
based agent at a $14.41\%$ end-to-end success rate against human performance of
$78.24\%$. {{cite:liu2024agentbench}} identified the bottleneck across eight
environments as long-horizon reasoning, decision-making and instruction-following
— not as any single missing capability. Those are the numbers a design should be
built against.

## 3. Prerequisites

You need {{ch:rsn-cot}}'s compounding result. An agent takes many steps and its
success is multiplicative in them, so {{eq:chain-accuracy-compounds}} is the
governing equation of this entire part.

You need {{ch:rsn-tool-assisted}}'s boundary-crossing arithmetic, because an agent
step is a boundary crossing and the cost accounting there transfers directly.

From {{ch:rsn-test-time-compute}}, the coverage/selection decomposition recurs
here as the difference between an agent being *able* to solve a task and its
reliably *choosing* to.

No framework knowledge is assumed, and none is taught. Everything here is about
the shape of the computation.

## 4. Intuitive Explanation

Consider three ways to handle a customer request.

The first writes the steps down. Look up the account, check the order status,
draft a reply. A human decided that sequence, a model fills in the pieces, and
every request goes through the same path. This is a pipeline, and its
characteristic property is that you know in advance exactly what will happen.

The second adds a fork. Classify the request into one of six kinds, then run the
pipeline for that kind. A human still wrote all six pipelines and decided that
there are six; the model now makes one decision, which is *which* pipeline. This
is a router, and it handles six shapes instead of one at the cost of a
classification that can be wrong.

The third gives the model the tools and the goal and lets it decide. Look up the
account — then, having seen the account, decide what to do next. Maybe check the
order, maybe check a refund policy, maybe ask a question. Nobody wrote the
sequence, and different requests produce different sequences. This is an agent.

The difference between the second and third is not degree. In the router, a human
enumerated the possibilities and the model picked from the list. In the agent, the
model constructs the sequence, which means it can construct sequences nobody
thought of — including bad ones.

That is the whole trade, and both halves are measurable.

**The benefit is tail coverage.** A router can only run flows that exist, so its
ceiling is the fraction of your traffic that fits a shape you enumerated. If
$95\%$ of requests are one of six kinds, a router handles $95\%$ and an agent's
per-step error tax buys you nothing. If half your requests are one-off
combinations, no amount of enumeration catches up, and the tax is what you pay to
stop trying.

**The cost is that every step is a chance to go wrong, and the errors multiply.**
A five-step task at $93\%$ per step completes $70\%$ of the time. Not because any
step is unreliable — $93\%$ sounds fine — but because five of them in a row is
$0.93^5$. This is {{ch:rsn-cot}}'s arithmetic and it is the reason agent demos
look better than agent deployments: a demo is three steps and a deployment is
twelve.

There are two further costs that averages hide, and they are the reason this
chapter has two listings rather than one.

The first is that an agent's cost is a *distribution*. A router spends exactly two
model calls on every request. An agent spends however many steps it takes, and
that has a tail — retries, wandering, and the occasional run that circles for a
long time. {{sec:9-practical-example}} measures a mean of $6.29$ steps with a p99
of $17$ and an observed maximum of $40$. A capacity plan built on the mean is
wrong by that factor for exactly the requests that hurt.

The second is that you cannot test an agent the way you test a workflow. Six
branches means six paths, and six tests cover them. Twelve steps with three
possible outcomes each means half a million paths. You are not going to cover
that, and the useful response is not to test harder but to notice that paths are
wildly unequal in probability: at horizon eight, ten paths out of $6{,}561$
account for $85\%$ of runs. So an agent's correctness argument is *statistical*
— cover the mass, not the paths — and it comes with an explicit uncovered
remainder that a workflow's does not.

That remainder matters more than its size suggests, because the rare paths are
exactly where the agent does something surprising with its tools. Which is why
{{ch:ag-security}}'s response is about limiting consequences rather than
improving coverage.

## 5. Formal Explanation

Define the three architectures by who determines the action sequence
$a_1, \ldots, a_k$.

$$\text{pipeline: } a_i = f_i(\text{state}), \quad \text{router: } a_i = f_{r(x)}^{(i)}, \quad \text{agent: } a_i \sim \pi_\theta(\cdot \mid x, a_{1:i-1}, o_{1:i-1})$$ (eq:control-location)

In the first two, $f$ is written by a human and the model's contribution is
bounded — filling slots, or choosing $r(x)$ from a finite set. In the third, the
policy $\pi_\theta$ is the model, the action space is whatever the tools permit,
and the sequence length $k$ is itself chosen by $\pi_\theta$.

**That last clause is the definition.** An agent chooses when to stop, which is why
termination is a problem for agents and not for workflows, and it is
{{ch:ag-termination}}'s subject.

Now success. Let $h$ be the fraction of tasks matching an enumerated shape (the
head), so $1 - h$ is the tail mass. Let $p_r$ be routing accuracy, $p$ per-step
agent accuracy, and $k_{\text{head}}$, $k_{\text{tail}}$ the steps each kind of
task needs. Then:

$$S_{\text{router}} = h \cdot p_r, \qquad S_{\text{agent}} = h\, p^{\,k_{\text{head}}} + (1-h)\, p^{\,k_{\text{tail}}}$$ (eq:tail-mass-decides)

The router's success is linear in head mass with a hard ceiling at $p_r$; the
agent's is a weighted average of two compounding terms and has no ceiling in $h$.
Setting them equal gives the crossover tail mass, and it moves in exactly two
ways: **left as $p$ rises** (a more reliable agent is worth using sooner), and
**right as you enumerate more shapes** (every shape added converts tail into
head).

Cost is where the architectures separate more sharply. The router's is a constant;
the agent's is a random variable:

$$C_{\text{router}} = 2, \qquad C_{\text{agent}} = K, \quad K \text{ random}$$ (eq:autonomy-costs-variance)

with $K$ having a geometric-like tail whenever the loop can enter non-productive
states. The decision statistic is not average success or average cost but their
ratio against the cost of failure. Let $c_f$ be the cost of a failed task and
$c_m$ the cost of a model call. Then an agent is preferable when:

$$\big(S_{\text{agent}} - S_{\text{router}}\big)\, c_f \;>\; \big(\mathbb{E}[K] - 2\big)\, c_m$$ (eq:failure-to-call-ratio)

which rearranges to a condition on $c_f / c_m$ alone. **That ratio is the number
to know about your product**, and it is the reason the same measurement supports
opposite decisions for a high-volume assistant and a coding agent.

Finally, testability. With $m$ possible outcomes per step and horizon $k$, the
number of distinct execution paths is:

$$P(k) = m^{k}$$ (eq:path-explosion)

and a test suite of size $T$ covers $\min(T, P)/P$ of them, which goes to zero
fast. But paths carry unequal probability, so define the *mass coverage* of the
$N$ commonest paths:

$$M(N) = \sum_{j=1}^{N} \pi_{(j)}, \qquad \pi_{(1)} \ge \pi_{(2)} \ge \cdots$$ (eq:statistical-correctness-argument)

$M$ rises far faster than $N/P$, and the achievable correctness argument for an
agent is a statement about $M$ rather than about path coverage — with an explicit
$1 - M$ remainder that a workflow does not have.

## 6. Mathematical Foundation

Three properties of {{eq:tail-mass-decides}} are worth extracting.

**The crossover is more sensitive to task length than to per-step accuracy.**
$\partial S_{\text{agent}} / \partial k = p^k \ln p$, which for $p$ near 1 is
approximately $-(1-p) p^k$ — small — while $\partial S / \partial p = k p^{k-1}$
carries a factor of $k$. So a longer task hurts more than a slightly worse model,
and the practical lever is decomposition: two five-step tasks succeed more often
than one ten-step task, which is {{ch:ag-planning}}'s justification if it has one.

**The router's ceiling is hard and the agent's is soft.** No investment in a router
raises $S$ above $p_r$ while tail mass is positive; enumerating more shapes moves
mass rather than raising the ceiling. This asymmetry is why the argument
eventually goes the agent's way in any domain where the input space is open, and
why it never does in domains where it is closed.

**Cost and benefit are correlated in the wrong direction.** Tail tasks are both
rarer and longer, so the agent's cost rises precisely where its advantage does.
{{sec:9-practical-example}} measures average calls going from $3.00$ to $5.10$
across the tail sweep while the agent's advantage grows — which is why the
efficiency crossover never arrives in that range.

Now the cost distribution. Model the loop as a Markov chain with a productive
state, a non-productive state entered with probability $q$ per step, and escape
probability $e$. Time spent in non-productive cycles is geometric with mean $1/e$,
so:

$$\mathbb{E}[K] \approx \frac{k^{*}}{p} \Big(1 + \frac{q}{e}\Big)$$ (eq:expected-steps)

The expectation is finite whenever $e > 0$, so the common claim that an unbudgeted
agent has infinite expected cost is *false* under this model, and
{{sec:9-practical-example}} says so explicitly. What is unbounded is the maximum:
the longest run you observe grows like $\log N$ in the number of requests served,
so a p100 cannot be stated without also stating a traffic volume.

That is a weaker argument for budgets than the usual one and it is still
decisive. A budget lets you state a worst case without reference to traffic, and
{{sec:9-practical-example}} prices it: capping at $12$ steps costs $4.1$ points of
completion and caps the worst case at $12$ instead of $40$.

## 7. Internal Mechanics

### 7.1 What the loop actually is

Strip the framework away and an agent is:

```mermaid {#fig:agent-loop caption="The agent loop. The only structural difference from a workflow is that the model, not the author, decides whether to take the branch back."}
flowchart LR
    G[goal + state] --> M[model call]
    M --> D{action or stop?}
    D -- action --> T[execute tool]
    T --> O[observation]
    O --> G
    D -- stop --> R[result]
```

Every framework is an implementation of that diagram with opinions about state
representation, error handling and observability. The diagram is what to reason
about; the opinions are what to compare when choosing a library, and they date.

Note what is *not* in the diagram: any guarantee that the branch back is taken a
bounded number of times. That absence is the whole of {{ch:ag-termination}}.

### 7.2 Why the model's step accuracy is not the number you think

The per-step accuracy $p$ in {{eq:tail-mass-decides}} is not the model's accuracy
on a benchmark. It is the probability that, given the current state, the model
emits an action that makes progress — which folds together tool selection,
argument construction, result interpretation, and the judgement of what to do
next. {{cite:schick2023toolformer}} decomposes the first three, and
{{ch:ag-tool-calling}} measures them separately, because they fail at very
different rates.

The practical consequence: $p$ is a property of the *agent system*, not of the
model, and most of the levers on it are in the tool design rather than in the
prompt.

### 7.3 Where the state lives

A workflow's state is a data structure the author defined. An agent's state is
whatever is in the context window, which means it is text, it is lossy, and it
grows with every step.

Two consequences follow, and both are {{ch:ag-memory}}'s subject. The context
grows linearly in steps, so a long run gets expensive in exactly the way
{{part:15}} describes — a growing KV cache re-read on every step. And the state is
not typed, so nothing prevents an inconsistent state, which is one of the ways a
loop enters {{eq:expected-steps}}'s non-productive region.

### 7.4 The correctness argument, stated precisely

For a workflow, "we tested it" means: the set of execution paths is small, we
enumerated it, and each path was checked. That is a coverage claim and it is
verifiable.

For an agent, {{eq:path-explosion}} makes that claim unavailable, and the honest
substitute is: we sampled runs from the real task distribution, the sample covers
$M$ of the probability mass, and the remaining $1 - M$ contains behaviours we have
not observed. {{sec:9-practical-example}} computes $M$ for a horizon-8 loop and
finds ten paths covering $85.4\%$ of runs — so the substitute is much less
hopeless than {{eq:path-explosion}} alone suggests.

But state it as a probability claim rather than as a coverage one. The difference
matters when someone asks whether the agent can do something specific, because the
answer for a workflow is "no, here is the path enumeration" and for an agent it is
"we have not seen it, at this sample size".

### 7.5 Why the rare paths are the dangerous ones

The uncovered mass is, by construction, the unusual outcomes. And unusual outcomes
are where an agent takes an unusual action — an unexpected tool, an unexpected
argument, a sequence nobody anticipated.

So the untested fraction of an agent's behaviour is systematically the fraction
with the largest blast radius, which inverts the usual testing intuition that
uncovered code is uncovered because it does not matter. This is the argument for
{{ch:ag-security}}'s design response: constrain what an action *can do*
rather than trying to predict which action will be taken.

## 8. Implementation

Two listings. The first measures the architecture choice as a function of tail
mass and of cost. The second measures the two things averages hide: the shape of
an agent's cost distribution, and what happens to testability.

```python {tier=A name=control-location}
"""Pipeline, router, agent: the choice is a measurement about your traffic.

The three architectures differ in ONE thing -- who chooses the next step. A
pipeline's control flow is written by a human. A router picks among human-written
flows. An agent chooses each action itself, including when to stop.

That difference has a cost and a benefit, and both are computable
(eq:control-location). The cost is that an agent's per-step choices compound
(ch:rsn-cot's eq:chain-accuracy-compounds arriving in a new place). The benefit is
that it can handle task shapes nobody anticipated.

So the question "should this be an agent" is not an architecture preference. It
is a question about how much of your traffic falls outside the shapes you
enumerated, and this listing puts a number on the crossover.
"""
import numpy as np

rng = np.random.default_rng(1451)

N_TASKS = 40000
N_SHAPES = 6            # task shapes the team enumerated and built flows for
STEPS_HEAD = 3          # steps a head task needs
STEPS_TAIL = 6          # steps a tail task needs (they are the awkward ones)
P_STEP = 0.93           # agent's per-step action accuracy
P_ROUTE = 0.94          # router's classification accuracy
COST_STEP = 1.0         # one model call
COST_FLOW = 1.0         # a hand-written flow is one call, no deliberation


def run(tail_mass, n=N_TASKS):
    """One draw of the task distribution, run through all three architectures."""
    is_tail = rng.random(n) < tail_mass
    shape = rng.integers(0, N_SHAPES, size=n)          # for head tasks
    steps = np.where(is_tail, STEPS_TAIL, STEPS_HEAD)

    # PIPELINE: one hand-written flow for the commonest shape. It is correct on
    # that shape and wrong on everything else, and it costs one call either way.
    pipe_ok = (~is_tail) & (shape == 0)
    pipe_cost = np.full(n, COST_FLOW)

    # ROUTER: classify into one of the known shapes, then run that flow. Correct
    # when the task is a known shape AND the classification is right. Costs one
    # call to route plus one to run.
    routed_right = rng.random(n) < P_ROUTE
    rout_ok = (~is_tail) & routed_right
    rout_cost = np.full(n, 2 * COST_FLOW)

    # AGENT: chooses each action. It can address any shape, and every step is a
    # chance to go wrong. Cost is one call per step.
    agent_ok = (rng.random((n, STEPS_TAIL)) < P_STEP)
    agent_ok = np.array([agent_ok[i, :steps[i]].all() for i in range(n)])
    agent_cost = steps * COST_STEP

    return {
        "pipeline": (float(pipe_ok.mean()), float(pipe_cost.mean())),
        "router": (float(rout_ok.mean()), float(rout_cost.mean())),
        "agent": (float(agent_ok.mean()), float(agent_cost.mean())),
    }


TAILS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

print(f"{N_TASKS} tasks. {N_SHAPES} task shapes were enumerated and given")
print(f"hand-written flows; the rest is tail. Head tasks take {STEPS_HEAD}")
print(f"steps, tail tasks {STEPS_TAIL}. The agent is {P_STEP:.0%} accurate per")
print(f"step; the router is {P_ROUTE:.0%} accurate at classifying.")
print()
print(f"{'tail mass':>11}{'pipeline':>22}{'router':>20}{'agent':>20}")
print(f"{'':>11}{'success':>11}{'cost':>11}{'success':>10}{'cost':>10}"
      f"{'success':>10}{'cost':>10}")
print("-" * 73)

res = {}
for t in TAILS:
    r = run(t)
    res[t] = r
    print(f"{t:>11.0%}{r['pipeline'][0]:>11.1%}{r['pipeline'][1]:>11.2f}"
          f"{r['router'][0]:>10.1%}{r['router'][1]:>10.2f}"
          f"{r['agent'][0]:>10.1%}{r['agent'][1]:>10.2f}")

print()
print()
print("Success per unit of cost -- what you get for each model call.")
print()
print(f"{'tail mass':>11}{'pipeline':>12}{'router':>10}{'agent':>10}"
      f"{'best':>12}")
print("-" * 55)
eff = {}
for t in TAILS:
    r = res[t]
    e = {k: r[k][0] / r[k][1] for k in r}
    eff[t] = e
    best = max(e, key=e.get)
    print(f"{t:>11.0%}{e['pipeline']:>12.3f}{e['router']:>10.3f}"
          f"{e['agent']:>10.3f}{best:>12}")

print()
print()
print("Where does the agent overtake on raw success? Sweep finely.")
print()
print(f"{'tail mass':>11}{'router':>10}{'agent':>10}{'gap':>10}")
print("-" * 41)
fine = {}
for t in (0.08, 0.12, 0.16, 0.20, 0.24, 0.30):
    r = run(t, n=60000)
    fine[t] = (r["router"][0], r["agent"][0])
    print(f"{t:>11.0%}{r['router'][0]:>10.1%}{r['agent'][0]:>10.1%}"
          f"{r['agent'][0] - r['router'][0]:>+10.1%}")

cross = [t for t in sorted(fine) if fine[t][1] > fine[t][0]]
head_step = P_STEP ** STEPS_HEAD
tail_step = P_STEP ** STEPS_TAIL
print(f"""
The first table is the trade in its simplest form, and the two ends of the tail
column are the whole argument.

At {0:.0%} tail mass the pipeline succeeds {res[0.0]['pipeline'][0]:.1%} of the
time -- it only handles one of the {N_SHAPES} shapes -- the router
{res[0.0]['router'][0]:.1%}, and the agent {res[0.0]['agent'][0]:.1%}. The router
wins comfortably, and it wins because every task IS one of the shapes somebody
enumerated. There is nothing for autonomy to buy.

At {0.5:.0%} tail mass the same three are {res[0.5]['pipeline'][0]:.1%},
{res[0.5]['router'][0]:.1%} and {res[0.5]['agent'][0]:.1%}. The router's ceiling
is the head mass, by construction: it can only run flows that exist. The agent
has no such ceiling and pays for it per step.

{'The agent overtakes the router at a tail mass of about ' + format(cross[0], '.0%') + '.' if cross else 'The agent does not overtake the router over the range swept.'}

That crossover is the number to compute for your own traffic, and it moves with
two things you can measure. It moves LEFT as your agent's per-step accuracy
rises: at {P_STEP:.0%} per step a {STEPS_HEAD}-step task completes
{head_step:.1%} of the time and a {STEPS_TAIL}-step task {tail_step:.1%}. And it
moves RIGHT as you enumerate more shapes, because every shape you add converts
tail mass into head mass.

**So "should this be an agent" is a question with a numeric answer, and the
answer is mostly about your traffic rather than about your model.** A team whose
requests fall into six shapes should write six flows. A team whose requests have
a long tail of one-off combinations cannot enumerate their way out, and the
per-step tax is what they pay to avoid trying.

The second table adds cost, and it does not merely shift the crossover -- it
removes it over the whole range swept.

Per model call the router is the most efficient of the three at every tail mass
here: {eff[0.0]['router']:.3f} against the agent's {eff[0.0]['agent']:.3f} at
zero tail, and {eff[0.7]['router']:.3f} against {eff[0.7]['agent']:.3f} at
{0.7:.0%} -- where the agent is winning on raw success by
{res[0.7]['agent'][0] - res[0.7]['router'][0]:+.1%}. The agent spends
{res[0.0]['agent'][1]:.1f} calls at zero tail and {res[0.7]['agent'][1]:.1f} at
{0.7:.0%}, because tail tasks are longer as well as rarer, so its cost rises
exactly where its advantage does.

**The success crossover and the cost-efficiency crossover are in completely
different places, and on this cost model the second one never arrives.** Which
you should use depends on whether model calls or failed tasks are the expensive
thing, and those differ by orders of magnitude between products. For a
high-volume assistant the calls dominate and the router is correct well past the
success crossover. For an agent that files support tickets or writes code, a
failed task costs a human's time and the calls are a rounding error.

That is a more useful framing than "agents are expensive", because it says what
to measure: **the ratio of the cost of a failed task to the cost of a model
call.** Below roughly {res[0.2]['agent'][1] / res[0.2]['router'][1]:.1f} -- the
call-count ratio at the success crossover -- the router wins on both counts.

One thing this listing deliberately does not model, and it is the largest
omission: the agent's cost here is deterministic given the task length. Real
agent loops retry, wander, and occasionally do not stop at all, which turns cost
into a heavy-tailed distribution and makes the mean a poor summary. That is the
next listing's subject, and it moves the recommendation further toward the
router than this table suggests.""")
```

The second listing looks at the same agent through two lenses that a mean hides.

```python {tier=A name=autonomy-costs-variance}
"""The two things autonomy costs that a mean does not show.

The previous listing compared architectures on average success and average cost.
Averages are the wrong summary for an agent, and this listing measures the two
reasons (eq:autonomy-costs-variance).

The first is that an agent's cost is a DISTRIBUTION, not a number. A router
spends exactly two calls on every request. An agent spends however many steps it
takes, and "however many it takes" has a tail: retries, wandering, and the case
where it does not converge at all. Capacity planning against a mean under-
provisions by whatever the tail is.

The second is that an agent has more distinct execution paths than you can
enumerate, which changes what "tested" can mean. A pipeline has one path. A
router has one per branch. An agent's path count grows exponentially in the
horizon, and this listing computes what fraction of it a test suite covers.
"""
import numpy as np

rng = np.random.default_rng(1523)

N_RUNS = 200000
P_STEP = 0.93           # a step does the right thing
P_STUCK = 0.06          # a step enters a non-productive cycle
STEPS_NEEDED = 5        # productive steps required to finish
P_ESCAPE = 0.35         # chance of leaving a non-productive cycle each step


def simulate(budget):
    """One agent run. Each step either makes progress, gets stuck, or is a wasted
    retry. Returns (completed, steps_used)."""
    done = np.zeros(N_RUNS, dtype=bool)
    steps = np.zeros(N_RUNS, dtype=np.int64)
    progress = np.zeros(N_RUNS, dtype=np.int64)
    stuck = np.zeros(N_RUNS, dtype=bool)
    alive = np.ones(N_RUNS, dtype=bool)
    for _ in range(budget):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        steps[idx] += 1
        u = rng.random(len(idx))
        # Runs in a non-productive cycle either escape or burn a step.
        st = stuck[idx]
        esc = st & (u < P_ESCAPE)
        stuck[idx[esc]] = False
        # Runs not stuck make progress, get stuck, or waste the step.
        free = ~st
        v = rng.random(len(idx))
        adv = free & (v < P_STEP)
        newstuck = free & (v >= P_STEP) & (v < P_STEP + P_STUCK)
        progress[idx[adv]] += 1
        stuck[idx[newstuck]] = True
        fin = progress[idx] >= STEPS_NEEDED
        done[idx[fin]] = True
        alive[idx[fin]] = False
    return done, steps


BUDGETS = [5, 8, 12, 20, 40, 100]

print(f"An agent needs {STEPS_NEEDED} productive steps. Each step makes progress")
print(f"{P_STEP:.0%} of the time, enters a non-productive cycle {P_STUCK:.0%} of")
print(f"the time, and otherwise is wasted; a cycle is escaped {P_ESCAPE:.0%} of")
print("the time per step.")
print()
print(f"{'step budget':>13}{'completed':>12}{'mean':>8}{'p50':>7}{'p90':>7}"
      f"{'p99':>7}{'max':>7}")
print(f"{'':>13}{'':>12}{'steps':>8}{'':>7}{'':>7}{'':>7}{'':>7}")
print("-" * 61)

tab = {}
for b in BUDGETS:
    done, steps = simulate(b)
    q = np.percentile(steps, [50, 90, 99])
    tab[b] = (float(done.mean()), float(steps.mean()), q[0], q[1], q[2],
              int(steps.max()))
    print(f"{b:>13}{tab[b][0]:>12.1%}{tab[b][1]:>8.2f}{q[0]:>7.0f}{q[1]:>7.0f}"
          f"{q[2]:>7.0f}{tab[b][5]:>7}")

print()
print()
print("What does the budget cost in completions, and what does it buy in")
print("predictability? Compare each budget against the largest.")
print()
ref = BUDGETS[-1]
print(f"{'step budget':>13}{'completions':>14}{'p99 cost':>11}"
      f"{'worst case':>13}")
print(f"{'':>13}{'lost':>14}{'vs b=' + str(ref):>11}{'cost':>13}")
print("-" * 51)
for b in BUDGETS:
    print(f"{b:>13}{tab[b][0] - tab[ref][0]:>+14.1%}"
          f"{tab[b][4] - tab[ref][4]:>+11.0f}{b:>13}")

print()
print()
print("How many distinct execution paths are there? A path is a sequence of")
print("per-step outcomes, and a test case exercises one of them.")
print()
K_OUT = 3               # outcomes per step: progress, stuck, wasted
N_TESTS = 500
print(f"{'architecture':>22}{'distinct paths':>17}{'covered by':>13}"
      f"{'coverage':>11}")
print(f"{'':>22}{'':>17}{str(N_TESTS) + ' tests':>13}{'':>11}")
print("-" * 63)
arch = [("pipeline", 1), ("router, 6 branches", 6)]
for h in (3, 5, 8, 12):
    arch.append((f"agent, horizon {h}", K_OUT ** h))
for name, paths in arch:
    cov = min(N_TESTS, paths)
    print(f"{name:>22}{paths:>17,}{cov:>13,}{cov / paths:>11.2%}")

print()
print()
print("Paths are not equally likely. What share of PROBABILITY MASS do the")
print("commonest paths cover, for an agent at horizon 8?")
print()
H = 8
probs = np.array([P_STEP, P_STUCK, 1 - P_STEP - P_STUCK])
mass = np.array([1.0])
for _ in range(H):
    mass = np.outer(mass, probs).ravel()
mass = np.sort(mass)[::-1]
cum = np.cumsum(mass)
print(f"{'commonest N paths':>20}{'share of runs':>16}"
      f"{'share of paths':>17}")
print("-" * 53)
for n in (1, 10, 50, 200, 1000, len(mass)):
    print(f"{n:>20,}{cum[n - 1]:>16.1%}{n / len(mass):>17.2%}")

hit_90 = int(np.searchsorted(cum, 0.90) + 1)
hit_99 = int(np.searchsorted(cum, 0.99) + 1)
print(f"""
The first table is the shape of an agent's cost, and the columns to compare are
the mean and the p99.

At a budget of {BUDGETS[-1]} steps the agent completes {tab[ref][0]:.1%} of tasks
using {tab[ref][1]:.2f} steps on average -- and a p99 of {tab[ref][4]:.0f} and a
worst case of {tab[ref][5]}. **The tail is {tab[ref][4] / tab[ref][1]:.1f} times
the mean**, and capacity planned against the mean under-provisions by that factor
for the requests that matter most, which are the slow ones.

A router's equivalent row would read: mean 2, p50 2, p90 2, p99 2, max 2. That is
the difference autonomy makes to a capacity plan, and it is invisible in any
comparison of averages.

The second table is what a step budget buys, and it is the most useful thing in
this listing because the trade is so lopsided.

Cutting the budget from {BUDGETS[-1]} to {20} costs
{tab[20][0] - tab[ref][0]:+.1%} in completions and takes the p99 from
{tab[ref][4]:.0f} steps to {tab[20][4]:.0f}. Cutting to {12} costs
{tab[12][0] - tab[ref][0]:+.1%} and caps the worst case at {12}.

**A budget converts a worst case you do not control into one you do, at a small
cost in completions**, and that is the argument for having one.

Be precise about what the uncapped case is, because the usual telling
overstates it. The run-length distribution here has a geometric tail rather than
an infinite one: escape probability is positive, so every run terminates with
probability 1, and across {N_RUNS:,} runs the longest observed was
{tab[ref][5]} steps -- {tab[ref][5] / tab[ref][1]:.1f} times the mean. The
expected cost is finite. What is not bounded is the MAXIMUM, which grows
logarithmically with how many requests you serve: the worst run you will see this
year is a function of your traffic volume, not of your system.

That is the honest form of the argument, and it is still decisive for a capacity
plan. You cannot state a p100 without stating a request count, and a budget lets
you state one without either.

The third table is the other thing autonomy costs, and it is about what testing
can mean.

A pipeline has one execution path and {N_TESTS} tests cover it. A six-branch
router has six, and {N_TESTS} tests cover them all. An agent at horizon
{8} has {K_OUT ** 8:,} distinct paths and {N_TESTS} tests cover
{N_TESTS / K_OUT ** 8:.2%} of them; at horizon {12} it is
{N_TESTS / K_OUT ** 12:.4%}.

**A workflow can be verified by enumerating its paths and an agent cannot.** That
is not a matter of testing harder. It is a statement about the size of the set,
and it means the two architectures need different correctness arguments: a
workflow's is coverage, and an agent's has to be something else.

The fourth table says what that something else is, and it is the reason the
situation is not hopeless.

Paths are wildly unequal in probability. At horizon {H}, the commonest
{hit_90:,} paths -- {hit_90 / len(mass):.2%} of the total -- account for
{0.9:.0%} of runs, and {hit_99:,} paths account for {0.99:.0%}. The exponential
blow-up is real and almost all of it is in outcomes that essentially never
happen.

So the achievable correctness argument for an agent is **statistical rather than
exhaustive**: sample from the real distribution of runs, cover the mass rather
than the paths, and accept that the remaining {1 - 0.99:.0%} contains behaviours
nobody has ever seen. That is a weaker guarantee than a workflow's and it is not
nothing, and stating it in those terms is more honest than either "we tested it"
or "it cannot be tested".

One consequence worth drawing out. Because the uncovered mass is where the
unusual outcomes live, and because unusual outcomes are exactly where an agent
does something surprising with its tools, **the untested fraction of an agent's
behaviour is systematically the dangerous fraction**. That is
ch:ag-security's subject, and it is why the design response there is about
limiting consequences rather than about improving coverage.""")
```

## 9. Practical Example

The first listing runs $40{,}000$ tasks through all three architectures. Six task
shapes were enumerated and given hand-written flows; the rest is tail. Head tasks
take three steps, tail tasks six. The agent is $93\%$ accurate per step and the
router $94\%$ accurate at classifying.

```
  tail mass              pipeline              router               agent
               success       cost   success      cost   success      cost
-------------------------------------------------------------------------
         0%      16.5%       1.00     94.0%      2.00     80.5%      3.00
        10%      15.1%       1.00     84.5%      2.00     78.6%      3.30
        20%      13.2%       1.00     75.3%      2.00     77.6%      3.60
        50%       8.3%       1.00     46.7%      2.00     72.5%      4.51
        70%       4.9%       1.00     28.3%      2.00     69.4%      5.10
```

At zero tail mass the router wins comfortably: every task *is* one of the shapes
somebody enumerated, and there is nothing for autonomy to buy. At $70\%$ the
router is at $28.3\%$ against the agent's $69.4\%$, because the router's ceiling
is the head mass by construction.

A finer sweep puts the success crossover at about $20\%$ tail mass:

```
  tail mass    router     agent       gap
-----------------------------------------
        12%     82.8%     78.5%     -4.2%
        16%     79.1%     77.9%     -1.2%
        20%     75.4%     77.5%     +2.2%
        30%     65.9%     75.9%    +10.0%
```

That crossover moves left as per-step accuracy rises and right as you enumerate
more shapes ({{eq:tail-mass-decides}}). **So "should this be an agent" has a
numeric answer, and it is mostly about your traffic rather than your model.**

The cost table does not merely shift the crossover — it removes it:

```
  tail mass    pipeline    router     agent        best
-------------------------------------------------------
         0%       0.165     0.470     0.268      router
        20%       0.133     0.377     0.216      router
        70%       0.049     0.141     0.136      router
```

Per model call the router is the most efficient of the three at every tail mass
swept, including at $70\%$ where the agent is winning on raw success by $41$
points. The agent's cost rises from $3.00$ to $5.10$ calls exactly where its
advantage grows, because tail tasks are longer as well as rarer.

**The success crossover and the cost-efficiency crossover are in different places,
and on this cost model the second one never arrives.** Which governs depends on
$c_f / c_m$ ({{eq:failure-to-call-ratio}}): for a high-volume assistant the calls
dominate and the router is right well past the success crossover; for an agent
that files tickets or writes code, a failed task costs a human's time and the
calls are a rounding error.

The second listing gives the agent five productive steps to complete, with a $6\%$
chance per step of entering a non-productive cycle escaped $35\%$ of the time.

```
  step budget   completed    mean    p50    p90    p99    max
-------------------------------------------------------------
            8       85.7%    5.77      5      8      8      8
           12       95.9%    6.16      5     10     12     12
           20       99.7%    6.28      5     10     17     20
          100      100.0%    6.29      5     10     17     40
```

At a budget of $100$ the mean is $6.29$ steps, the p99 is $17$, and the observed
maximum over $200{,}000$ runs is $40$. A router's row would read $2, 2, 2, 2, 2$.
**That is what autonomy does to a capacity plan, and it is invisible in any
comparison of averages.**

Capping at $12$ steps costs $4.1$ points of completion and caps the worst case at
$12$. Capping at $20$ costs $0.3$ points.

Be precise about what the uncapped case is, because the usual telling overstates
it. The distribution has a geometric tail, not an infinite one — every run
terminates with probability 1, and the expected cost is finite
({{eq:expected-steps}}). What is unbounded is the *maximum*, which grows with
traffic volume. You cannot state a p100 without stating a request count, and a
budget lets you state one without either.

Then testability:

```
          architecture   distinct paths   covered by   coverage
                                           500 tests           
---------------------------------------------------------------
              pipeline                1            1    100.00%
    router, 6 branches                6            6    100.00%
      agent, horizon 5              243          243    100.00%
      agent, horizon 8            6,561          500      7.62%
     agent, horizon 12          531,441          500      0.09%
```

**A workflow can be verified by enumerating its paths and an agent cannot** — not
as a matter of effort, but of set size. The two architectures need different
correctness arguments.

And the reason the situation is not hopeless:

```
   commonest N paths   share of runs   share of paths
-----------------------------------------------------
                   1           56.0%            0.02%
                  10           85.4%            0.15%
                  50           96.4%            0.76%
                 200           99.4%            3.05%
```

At horizon 8, ten paths out of $6{,}561$ cover $85.4\%$ of runs and two hundred
cover $99.4\%$. The exponential blow-up is real and almost all of it is in
outcomes that essentially never happen. So the achievable argument is statistical
({{eq:statistical-correctness-argument}}): cover the mass, and state the
remainder explicitly.

## 10. Production Considerations

Measure your tail mass before choosing an architecture. Sample a few hundred real
requests, classify them against the shapes you have flows for, and compute the
fraction that fits none. That number and {{eq:tail-mass-decides}} decide the
question.

Compute $c_f / c_m$ for your product. A failed task that costs a human ten
minutes and a model call that costs a fraction of a cent put you in a completely
different regime from a free-tier assistant, and the same measurement supports
opposite decisions.

Report agent cost as a distribution. p50, p90, p99 and observed max, not a mean.
Capacity plans and rate limits should be built on the p99.

Set a step budget always. {{sec:9-practical-example}} prices it at a few points of
completion for a bounded worst case, and the alternative is a worst case that is a
function of your traffic volume.

Instrument the loop, not just the outcome. Steps taken, tools called, and whether
the run terminated by success or by budget — because the budget-terminated
fraction is the metric that tells you whether the budget is set right.

Do not convert a workflow into an agent because agents are interesting. Convert it
when the tail mass justifies it, and expect to keep the workflow for the head:
the two compose, and a router with an agent as its fallback branch is usually
better than either alone.

## 11. Common Mistakes

**Treating agent-versus-workflow as a spectrum.** It is a fact about who chooses
the next action ({{eq:control-location}}), and the two have different correctness
arguments. Blurring it means applying a coverage argument to something that cannot
support one.

**Comparing architectures on average cost.** A router's cost is a constant and an
agent's is a distribution with a p99 nearly three times its mean.

**Building an agent for a closed input space.** If your requests fit six shapes, a
router beats an agent on both success and cost, and no amount of prompt work
changes that.

**Reading demo performance as deployment performance.** A three-step demo at $93\%$
per step completes $80\%$ of the time; a twelve-step deployment completes $42\%$.
The model did not get worse.

**Running without a step budget.** The expected cost is finite and the maximum is
a function of traffic ({{eq:expected-steps}}). Without a cap you cannot state a
worst case at all.

**Claiming an agent is "tested".** State the mass covered and the sample size.
{{eq:path-explosion}} makes the coverage claim unavailable.

## 12. Failure Modes

*Silent tail failure.* An agent handles the head well and fails on the tail, which
is where it was supposed to earn its cost. Because the tail is by definition rare,
aggregate metrics look fine.

*Cost blowup under load.* The p99 run costs three times the mean, so a traffic
spike does not raise cost linearly — it raises it by whatever fraction of requests
hit the tail, and those requests also occupy capacity longest.

*Non-productive cycles.* The loop repeats an action, gets the same observation, and
repeats it again. {{eq:expected-steps}}'s $q/e$ term, and the reason budgets
matter more than they look.

*Untested behaviour with a large blast radius.* The uncovered probability mass is
the unusual actions, which are the consequential ones
({{sec:7-internal-mechanics}}).

*Enumeration creep.* A team adds shapes to a router until the routing itself
becomes unreliable. Routing accuracy is a classification problem over a growing
label set, and it degrades.

## 13. Alternatives

**A router with a fallback.** Handle the enumerated head with flows and route the
rest to an agent. This gets the router's cost profile on most traffic and the
agent's coverage on the tail, and is usually the right first architecture.

**Constrained agents.** Give the model action choice but restrict the action space
and the horizon. This shrinks {{eq:path-explosion}} enough to make partial
enumeration meaningful, and it is the practical middle ground.

**A workflow with model-filled slots.** {{part:8}}'s constrained decoding makes the
model's contribution structurally valid by construction, which removes a large
class of {{ch:ag-tool-calling}}'s failures.

**Human in the loop.** Where $c_f$ is very large, a human confirmation converts a
failure into a delay. {{ch:ag-termination}} prices this.

**Not automating the tail.** Sometimes the correct answer. If the tail is $5\%$ and
each case costs a human two minutes, an agent that handles it at $70\%$ and needs
oversight may not be cheaper.

## 14. Evaluation

Evaluate on the task distribution you actually have, weighted by frequency. An
agent evaluated on curated tasks is measured on the head;
{{cite:zhou2024webarena}}'s $14.41\%$ against a human $78.24\%$ is what happens
when the tasks are long-horizon and realistic.

Report success *and* the cost distribution *and* the budget-termination rate.
Three numbers; the third one tells you whether your budget is binding.

Break failure down by cause: wrong tool, wrong arguments, wrong interpretation,
wrong next step, never terminated. The aggregate hides which of the four is
costing you, and {{ch:ag-tool-calling}} measures them separately for that reason.

Measure the mass coverage of your evaluation set
({{eq:statistical-correctness-argument}}) — what fraction of real runs follow a
path you have tested. That is the honest form of a coverage claim.

And compare against the router baseline every time. It is cheap, and
{{sec:9-practical-example}} says it wins more often than the architecture
discussion implies.

## 15. Advanced Concepts

**Decomposition as the main lever.** From
{{sec:6-mathematical-foundation}}, task length hurts more than per-step accuracy
does. Splitting a ten-step task into two five-step tasks with a checkpoint between
them raises completion substantially, and it is a cheaper intervention than
improving the model. This is the strongest argument for structure inside an agent
and it is {{ch:ag-planning}}'s to make or fail to make.

**Agents as search.** {{ch:rsn-test-time-compute}}'s decomposition applies: an
agent that can retry is sampling, and its coverage grows with attempts. What it
lacks is a selector — the agent must decide whether it succeeded, which is
{{eq:correlated-critic}}'s problem again unless the environment tells it.

**The horizon at which everything changes.** {{cite:liu2024agentbench}} identifies
long-horizon consistency as the bottleneck. There is a horizon beyond which every
current system fails, it is task-dependent, and measuring it for your task is more
informative than any aggregate benchmark. {{maturity:EMERGING}} as a practice.

**Compositional guarantees.** If a workflow's correctness is compositional — each
step verified, so the whole is verified — an agent's is not, because the step set
is not fixed. Recovering any compositional guarantee for an agent requires
constraining the action space enough to enumerate it, which is
{{sec:13-alternatives}}'s constrained-agent option and the only known route.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:rsn-cot}}'s {{eq:chain-accuracy-compounds}} is the governing equation of this
part, and {{eq:tail-mass-decides}} is it applied to two task lengths at once.

{{ch:rsn-tool-assisted}}'s boundary-crossing arithmetic explains why an agent step
is expensive and why fewer, larger steps beat more, smaller ones — the same
argument that made one program beat twelve calls.

{{ch:rsn-self-consistency}}'s correlated-critic result is why an agent cannot
reliably judge its own progress, which is the mechanism behind
{{eq:expected-steps}}'s non-productive cycles and the subject of
{{ch:ag-recovery}}.

Ahead: {{ch:ag-tool-calling}} decomposes the per-step accuracy this chapter
treated as a single number; {{ch:ag-loop}} develops {{eq:expected-steps}}
properly; and {{ch:ag-security}} takes up the observation that the untested
mass is the dangerous mass.

## 17. Exercises

1. Solve {{eq:tail-mass-decides}} for the crossover tail mass symbolically, and
   check it against the listing's measured $20\%$.

2. Add a fourth architecture to the first listing: a router whose fallback branch
   is the agent. Measure it across the tail sweep and say where it wins.

3. Derive {{eq:failure-to-call-ratio}} and compute the threshold $c_f/c_m$ at each
   tail mass in the listing.

4. In the second listing, vary $P_{\text{ESCAPE}}$ down toward zero and observe
   what happens to the mean and the maximum. At what point does the mean become an
   unusable summary?

5. Compute the mass coverage {{eq:statistical-correctness-argument}} for horizon
   12 and find how many paths cover $99\%$. Compare with horizon 8.

6. Take a workflow you own, sample 200 real requests, and measure its tail mass.
   Decide whether it should be an agent, and write down which number decided it.

## 18. Interview Questions

1. What distinguishes an agent from a workflow? Answer in one sentence.

2. Your router handles 94% of requests correctly. When is an agent better, and
   what would you measure to find out?

3. Why can an agent win on success and lose on cost at the same tail mass?

4. What is wrong with reporting an agent's average cost?

5. Why can you not test an agent the way you test a pipeline, and what do you do
   instead?

6. A three-step agent demo works at 80%. What do you expect from a twelve-step
   deployment, and why?

## 19. Research Questions

1. Can tail mass be estimated cheaply from a request log without hand-classifying
   against a shape inventory?

2. {{eq:path-explosion}} makes exhaustive verification impossible. Is there a
   constrained action-space design that recovers a compositional guarantee without
   giving up the coverage that motivated the agent?

3. What determines the horizon at which long-horizon consistency collapses
   ({{cite:liu2024agentbench}}), and can it be predicted from a model's
   single-step properties?

4. Is the mass-coverage statistic in {{eq:statistical-correctness-argument}} stable
   between an evaluation set and production traffic, and if not, what does that
   imply about agent evaluation?

5. Non-productive cycles are the dominant cost term in {{eq:expected-steps}}. What
   detects one from inside the loop, given that the agent's own judgement is
   correlated with the error that caused it?

## 20. Chapter Summary

An agent differs from a workflow in exactly one property: **the model chooses the
next action, including whether to stop** ({{eq:control-location}}). That is
binary, not a spectrum, and it determines everything else.

The benefit is tail coverage and the cost is compounding. A router's success is
capped at its head mass; an agent's is a weighted sum of $p^{k}$ terms with no
such ceiling ({{eq:tail-mass-decides}}). {{sec:9-practical-example}} measures the
success crossover at about $20\%$ tail mass — and finds the router more efficient
per model call at *every* tail mass swept, including where the agent leads on
success by $41$ points. The two crossovers are in different places, and which
governs is decided by $c_f/c_m$, the cost of a failed task over the cost of a
model call.

Averages hide two things. An agent's cost is a distribution: mean $6.29$ steps,
p99 of $17$, observed maximum $40$, against a router's constant $2$. And an
agent's execution paths number $m^k$ ({{eq:path-explosion}}) — $531{,}441$ at
horizon 12, of which 500 tests cover $0.09\%$.

A step budget prices out well: capping at 12 steps costs $4.1$ points of
completion and caps the worst case at 12 instead of 40. The usual justification
overstates the case — the expected cost is finite ({{eq:expected-steps}}) — but the
*maximum* grows with traffic volume, so a budget is what lets you state a worst
case at all.

And the correctness argument has to change. Paths are wildly unequal: ten of
$6{,}561$ cover $85.4\%$ of runs. So an agent's argument is statistical — cover the
mass, state the remainder ({{eq:statistical-correctness-argument}}) — and the
remainder matters more than its size, because rare paths are where an agent takes
unusual actions with real tools.

## 21. Further Reading

{{cite:zhou2024webarena}} is the calibration. Read it for the environment design
and for the $14.41\%$ against $78.24\%$, which is the most useful single number in
this part.

{{cite:liu2024agentbench}} is the diagnosis: the bottleneck is long-horizon
consistency rather than any single capability, across eight environments. Its
observation that code training helps some agent tasks and hurts others is worth
noticing too.

{{cite:yao2023react}} is the architecture the next few chapters build on, and
{{ch:ag-react}} reads it against {{ch:rsn-tool-assisted}}'s boundary-crossing
cost.

{{cite:schick2023toolformer}} for the four-decision decomposition of tool use that
{{ch:ag-tool-calling}} measures, and {{cite:shinn2023reflexion}} for the feedback
loop {{ch:ag-recovery}} takes apart.
