---
id: as-graph
number: 165
part: XVIII
tier: full
status: draft
requires: [path-explosion, statistical-correctness-argument, tail-mass-decides]
provides: [graph-bounds-the-paths, graph-surrenders-the-tail,
           branch-count-is-an-exponent, escape-hatch-unbounds-again,
           graph-is-a-router-over-paths, readability-is-the-real-benefit]
citations: [cemri2025mast, liu2024agentbench, zhou2024webarena, yao2023react,
            greshake2023indirect, du2023debate]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the one property a graph
supplies that a loop cannot, and price it in tests; show that the testability
advantage is largest exactly where an agent's behaviour is most varied; explain
why a graph loses to a free loop even at zero tail mass; compute what branch count
does to reliability; and say what an escape hatch does to the property the graph
was adopted for.

## 2. Why This Matters

Graph orchestration is usually sold on reliability — a declared control flow, so
the system does what you drew. {{sec:9-practical-example}} finds the opposite at
its parameters: at *zero* tail mass, where every request fits a drawn route and the
graph should be at its strongest, the graph scores $66.1\%$ against a free loop's
$75.1\%$.

The reason is that a graph does not remove the model's uncertainty. It *relocates*
it, from "which action" to "which edge" — and an edge choice is still a
classification the model makes. Three branch points at $96\%$ is $88.5\%$ before
any step executes, and **branch count is an exponent**: ten branches at $90\%$
takes the graph to $25.9\%$.

What a graph does supply is real and singular. It bounds the path set. Six nodes
with two edges each gives $32$ paths, and a $400$-case suite covers all of them,
against $43$ million for a horizon-16 loop. **That is a property; everything else
about graph orchestration is a convenience.**

But the advantage is smaller than the path counts suggest, and it is largest in the
wrong place. Applying {{ch:ag-what-is-an-agent}}'s correction — that paths are
wildly unequal, so mass matters rather than count — a horizon-10 loop already covers
$81.1\%$ of runs with $400$ tests. And the gap widens precisely as behaviour gets
*more varied*: at low concentration the graph covers $90.0\%$ of mass and the loop
$12.7\%$. **A graph's testability advantage is largest exactly where you wanted an
agent.**

Then the cost the chapter was written to measure. A graph can only run routes its
author drew, so it is {{ch:ag-what-is-an-agent}}'s router with a larger enumerable
set. At $50\%$ tail mass it scores $33.2\%$ against the loop's $65.5\%$. And the
obvious fix — an escape hatch to a free loop — converts the bounded path set back
into an unbounded one at exactly the rate it is used.

## 3. Prerequisites

You need {{ch:ag-what-is-an-agent}}'s path-explosion result and its mass-coverage
correction, because this chapter is both applied to a different control structure.

Its tail-mass arithmetic ({{eq:tail-mass-decides}}) is what
{{sec:9-practical-example}}'s second listing reuses — a graph is a router over
paths, so the same equation governs.

From {{ch:as-multi-agent}}, the observation that a count in an exponent is a
first-order design parameter. Branch count plays the role handoff count played
there.

## 4. Intuitive Explanation

A graph orchestration declares the control flow. Nodes are steps, edges are
transitions, and conditions on edges decide which transition fires. The model still
does the work inside the nodes; what changes is that the *sequence* is drawn rather
than chosen freely.

The appeal is obvious and it is genuinely about testing. A free loop can produce
any sequence of actions its horizon allows, which is an astronomically large set —
{{ch:ag-what-is-an-agent}} counted half a million paths at horizon twelve. A graph
can produce only the routes its edges permit, which is a set you can enumerate,
list in a test plan, and check off.

That is a real property and it is the only one. Everything else attributed to
graphs — clearer code, better observability, easier onboarding — follows from having
written the flow down, and would follow equally from any other explicit
representation.

Now three things that complicate it, in increasing order of importance.

**The path count overstates the problem it solves.** Paths are not equally likely.
A loop's astronomically large path set is mostly outcomes that essentially never
occur, and a few hundred tests already cover most of the runs. So the honest
comparison is not $32$ against $43$ million; it is $100\%$ mass coverage against
$77\%$, which is a meaningful difference and not a categorical one.

**And the advantage sits in the wrong place.** The graph's edge is largest when
behaviour is varied — when runs spread across many routes rather than concentrating
on a few. But varied behaviour is what an agent is *for*. If your runs concentrate
on a handful of routes, the loop is nearly as testable as the graph and you might
as well have written a workflow. If they do not, the graph's routes will not cover
them.

**And the branches are their own error source.** This is the one the chapter did not
anticipate. Every branch point in a drawn route is a decision, and the model makes
it. Getting an edge right is not free: it is a classification over a set the author
fixed, which is a different problem from choosing an action but not obviously an
easier one. A route with three branches at $96\%$ each has already lost $11.5\%$
before any node executes, and richly-branched graphs pay this on every request.

So the graph's declared control flow does not eliminate the model's judgement. It
moves the judgement to the edges and adds up the errors, which is why
{{sec:9-practical-example}} finds it behind a free loop even in its best case.

The last idea is what happens when you notice the tail problem and add an escape
hatch — a fallback that hands unroutable requests to a free agent. It works, in the
sense that the tail is handled. And it dissolves the property you bought the graph
for: the runs that take the hatch are unenumerable, and they are the unusual ones,
which {{ch:ag-what-is-an-agent}} noted are the consequential ones.

You end up with a testable core and an untestable periphery, where the periphery is
where the interesting failures live. That is a defensible position and it should be
stated as one rather than described as "a graph with a fallback".

## 5. Formal Explanation

For a free loop with $m$ outcomes per step and horizon $k$, the reachable path set
is $m^k$. For a graph with $n$ nodes and branching factor $b$:

$$|\Pi_{\text{loop}}| = m^{k}, \qquad |\Pi_{\text{graph}}| = b^{\,n-1}$$ (eq:graph-bounds-the-paths)

Both are exponential; the graph's exponent is under the author's control and the
loop's is not. That is the entire structural difference and it is why a graph is a
*bound* rather than an improvement.

Coverage of a test suite of size $T$ is $\min(T, |\Pi|)/|\Pi|$, which is the wrong
statistic. Following {{eq:statistical-correctness-argument}}, use mass:

$$M(T) = \sum_{j=1}^{T}\pi_{(j)}, \qquad \pi_{(j)} \propto j^{-s}$$ (eq:mass-under-skew)

with $s$ the concentration of run probability. $\partial M/\partial s > 0$ for both
shapes, and the loop's derivative is larger — so the graph's advantage
$M_{\text{graph}} - M_{\text{loop}}$ *shrinks* as runs concentrate.
{{sec:9-practical-example}} measures the gap at $77.3$ points when $s = 0.6$ and
$0.0$ when $s = 2.2$.

**The graph's testability advantage is largest at low concentration**, which is the
regime where a loop's flexibility is being used.

Now reliability. Let a drawn route have $\beta$ branch points, each taken correctly
with probability $p_e$, and $k_h$ steps at $p$:

$$S_{\text{graph}} = h \cdot p_e^{\,\beta} \cdot p^{\,k_h}, \qquad S_{\text{loop}} = h\,p^{\,k_h} + (1-h)\,p^{\,k_t}$$ (eq:branch-count-is-an-exponent)

with $h$ the head mass. Two differences: the graph carries a $p_e^{\beta}$ factor
the loop does not, and the loop carries a tail term the graph does not.

So the graph wins only when:

$$p_e^{\,\beta} \;>\; 1 + \frac{(1-h)\,p^{\,k_t}}{h\,p^{\,k_h}}$$ (eq:graph-is-a-router-over-paths)

whose right-hand side exceeds one for any $h < 1$, and whose left-hand side is at
most one. **The graph cannot win on this comparison at any positive tail mass**, and
at zero tail mass it wins only if $p_e = 1$. That is the formal version of
{{sec:9-practical-example}}'s first row.

Finally, the escape hatch. If a fraction $e$ of runs leave the graph for a free
loop, the enumerable share of behaviour is:

$$\text{enumerable} = 1 - e$$ (eq:escape-hatch-unbounds-again)

linear and unforgiving. A hatch used on $15\%$ of runs leaves $85\%$ of behaviour
enumerable — and those $15\%$ are, by construction, the runs no route anticipated.

## 6. Mathematical Foundation

Three extractions.

**Branch reliability should be measured, and almost never is.** $p_e$ in
{{eq:branch-count-is-an-exponent}} is a classification accuracy over the edge set
at each node, and it is directly measurable from traces: at each branch, did the
run take the edge a human would have chosen? {{sec:9-practical-example}} shows
$p_e = 0.90$ with ten branches taking the graph to $25.9\%$, so this is a
first-order number.

**Branch count is a design parameter with an exponent, like handoff count.** From
{{eq:branch-count-is-an-exponent}}, $\partial \log S/\partial \beta = \log p_e$, so
each branch costs a constant multiplicative factor. A graph refactored to have
fewer, wider decision points is strictly better than one with many narrow ones at
the same $p_e$ — and the instinct when a graph misroutes is to add more conditions,
which makes it worse.

**The escape-hatch trade is quantifiable and should be stated.** From
{{eq:escape-hatch-unbounds-again}}, a system with a hatch has a testability claim of
$1 - e$ and a capability claim of the full loop. Reporting only the first is the
usual practice and it is misleading, because the uncovered $e$ is the unusual
behaviour.

One caveat on the model. It treats a graph's nodes as having the same per-step
reliability as a loop's, which is generous to the loop in one respect and to the
graph in another. A node with a narrower job may well execute more reliably — that
is the specialisation argument from {{ch:as-multi-agent}}, which found retries
eating the edge. And a loop that can revisit a step has retries a rigid graph does
not. Both corrections are second-order relative to $p_e^{\beta}$.

## 7. Internal Mechanics

### 7.1 Where the model still decides

```mermaid {#fig:graph-decisions caption="A graph moves the model's judgement from the action to the edge. The dashed decisions are the ones a graph adds; they do not exist in a free loop."}
flowchart TD
    A[start] --> B[node: fetch]
    B --> C{edge condition}
    C -- path 1 --> D[node: transform]
    C -- path 2 --> E[node: escalate]
    D --> F{edge condition}
    F -- path 1 --> G[node: write]
    F -- path 2 --> E
    C -.model decides.-> C
    F -.model decides.-> F
```

The dashed annotations are the point. A graph does not remove judgement; it
schedules it.

### 7.2 What makes an edge condition reliable

The same things that make a tool call reliable ({{ch:ag-tool-calling}}), and for the
same reason — an edge condition is a classification over an enumerated set.

**Fewer alternatives per branch.** A three-way branch is easier than a nine-way one,
which argues for hierarchy over breadth.

**Distinguishable conditions.** {{ch:ag-tool-calling}}'s overlap result transfers:
two edges whose conditions describe similar situations are two edges the model will
confuse, and inventories of edges accrete the same way inventories of tools do.

**Structural conditions where possible.** An edge keyed to a typed field — status is
`failed`, the record exists — has $p_e = 1$. An edge keyed to a judgement does not.
**The cheapest way to raise $p_e$ is to make conditions checkable rather than
inferable**, and most graph frameworks permit both without distinguishing them.

### 7.3 The escape hatch, honestly

A hatch is the right design and it should be instrumented rather than hidden. Three
things to record:

**The rate.** {{eq:escape-hatch-unbounds-again}} makes it your testability claim.

**What took it.** A cluster of similar requests taking the hatch is a route you
should draw; a scatter is genuine tail.

**What happened after.** Hatch runs are the unenumerated behaviour, so they deserve
more logging than graph runs, not less.

### 7.4 Graphs and the failure taxonomy

{{cite:cemri2025mast}}'s system-design category is largely about control flow that
does not match the task, and a graph makes those failures *visible* — a misrouted
request is a wrong edge, which is inspectable, rather than a wrong action buried in
a trace.

That is a genuine operational benefit and it is about diagnosis rather than
prevention. A graph does not misroute less; it misroutes legibly.

### 7.5 What a graph is really for

The measurements in this chapter are unfavourable, and graph orchestration remains
the right choice for many systems. The reason is not in the tables.

**The control flow becomes an artefact a human can read, review, version and diff.**
On a system several people maintain, that is worth a great deal: a change to the
flow is a change to a file, reviewable in the ordinary way, rather than a change to
a prompt whose effect on behaviour is unpredictable.

That is a software-engineering property, not a reliability one, and it should be
argued for on those terms. A team that adopts a graph for maintainability and
accepts $p_e^{\beta}$ as its cost has made a defensible trade. A team that adopts
one expecting higher success rates has not measured $p_e$.

### 7.6 The two graphs people mean

"Graph orchestration" covers two designs that behave very differently, and
conflating them is why the reliability argument survives despite measurements like
this chapter's.

**The static graph** has edges whose conditions are checkable predicates over typed
state: a status field, a record's existence, a numeric threshold. Its routing is
deterministic given the state, so the branch term in
{{eq:branch-count-is-an-exponent}} disappears entirely and its only cost is the
tail. This is a state machine, it is genuinely more reliable than a free loop on
the traffic it covers, and it is {{ch:as-state-machines}}'s subject.

**The dynamic graph** has edges whose conditions are model judgements: "if the
user seems to be asking about billing", "if the result looks incomplete". Its
routing is a classification, it carries the full $p_e^{eta}$ penalty, and it is
what {{sec:9-practical-example}} measures.

Most deployed graphs are a mixture, and the mixture's reliability is the product
over both kinds of edge — so a graph with eight structural edges and two judged
ones pays $p_e^2$, not $p_e^{10}$.

That gives the practical instrument. **Count your judged edges separately from your
structural ones**, because only the first appear in the exponent. A refactoring
that converts a judged edge into a structural one — by having an earlier node write
a typed field rather than leaving the decision implicit — is worth a full factor of
$p_e$, and it is usually available.

It also explains the disagreement between this chapter and practitioners'
experience. A team whose graph is mostly structural edges is running something
closer to {{ch:as-state-machines}}'s design and correctly reports it as reliable.
A team whose edges are judgements is running the design measured here.

### 7.7 Why the tail argument is weaker than it looks

{{eq:graph-surrenders-the-tail}} treats an unroutable request as a failure, which
is the right model for a pure graph and is pessimistic for a real deployment in one
specific way: **a graph fails unroutable requests VISIBLY.**

That is {{ch:ag-loop}}'s distinction arriving again. A free loop given a request it
cannot handle wanders, consumes budget, and may return a confident wrong answer. A
graph given a request no edge matches has nowhere to go and says so.

So the tail comparison in {{sec:9-practical-example}} understates the graph on the
axis that matters most for anything with side effects. Its $33.2\%$ at high tail
mass is $33.2\%$ correct and $66.8\%$ *refused*, where the loop's $65.5\%$ is
$65.5\%$ correct and $34.5\%$ mixed between refusal and confident error.

Which is a real point in the graph's favour and it is a containment argument rather
than a capability one — the same shape as {{ch:ag-security}}'s. A graph is a
system that can only do what it was drawn to do, and for an agent with write access
that is worth something the success column does not show.

## 8. Implementation

Two listings. The first counts what each shape can reach and what a test budget
covers. The second measures what the graph gives up.

```python {tier=A name=graph-bounds-the-paths}
"""What a graph buys, in the only currency it can pay in.

A graph orchestration declares the control flow: nodes, edges, and the conditions
on them. The claim is usually about reliability, and the mechanism is about
TESTABILITY -- ch:ag-what-is-an-agent showed a free-running loop has more execution
paths than any test suite can cover, and a graph has as many as its edges allow
(eq:graph-bounds-the-paths).

This listing counts what each shape can reach and what a fixed test budget covers,
and then asks the question that decides whether the trade is worth taking: how much
of the real task distribution does the graph's path set actually contain?
"""
import numpy as np

rng = np.random.default_rng(3251)

TESTS = 400
OUTCOMES = 3            # per-step outcomes in a free loop


def loop_paths(horizon):
    return OUTCOMES ** horizon


def graph_paths(nodes, avg_out):
    """Paths through a DAG of `nodes` with `avg_out` outgoing edges each, up to
    the longest route. This is an upper bound on distinct executions."""
    return int(avg_out ** max(1, nodes - 1))


print(f"A test suite of {TESTS} cases. A free loop has {OUTCOMES} outcomes per")
print("step; a graph has as many paths as its edges permit.")
print()
print(f"{'shape':>30}{'paths':>14}{'covered':>11}{'coverage':>11}")
print("-" * 66)
shapes = [("graph, 6 nodes, 2 edges each", graph_paths(6, 2)),
          ("graph, 10 nodes, 2 edges each", graph_paths(10, 2)),
          ("graph, 10 nodes, 3 edges each", graph_paths(10, 3)),
          ("free loop, horizon 6", loop_paths(6)),
          ("free loop, horizon 10", loop_paths(10)),
          ("free loop, horizon 16", loop_paths(16))]
cov = {}
for name, p in shapes:
    c = min(TESTS, p)
    cov[name] = (p, c / p)
    print(f"{name:>30}{p:>14,}{c:>11,}{c / p:>11.2%}")

print()
print()
print("Coverage is the wrong metric -- ch:ag-what-is-an-agent showed paths are")
print("wildly unequal. What share of RUNS does a test suite cover?")
print()


def mass_covered(paths, tests, skew):
    """Paths ranked by probability with a Zipf-like skew; how much probability
    mass do the commonest `tests` of them hold?"""
    n = min(paths, 2_000_000)
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-skew)
    w /= w.sum()
    return float(w[:min(tests, n)].sum())


print(f"{'shape':>30}{'paths':>12}{'path coverage':>16}{'mass covered':>15}")
print("-" * 73)
mc = {}
for name, p in shapes:
    m = mass_covered(p, TESTS, 1.2)
    mc[name] = m
    print(f"{name:>30}{p:>12,}{min(TESTS, p) / p:>16.2%}{m:>15.1%}")

print()
print()
print("How much mass a test suite covers, by how concentrated the runs are.")
print()
print(f"{'skew':>8}{'graph, 10 nodes':>18}{'loop, horizon 10':>19}"
      f"{'loop, horizon 16':>19}")
print("-" * 64)
sk = {}
for s in (0.6, 0.9, 1.2, 1.6, 2.2):
    row = (mass_covered(graph_paths(10, 2), TESTS, s),
           mass_covered(loop_paths(10), TESTS, s),
           mass_covered(loop_paths(16), TESTS, s))
    sk[s] = row
    print(f"{s:>8.1f}{row[0]:>18.1%}{row[1]:>19.1%}{row[2]:>19.1%}")

print()
print()
print("And how many tests each shape needs for 99% mass coverage.")
print()
print(f"{'shape':>30}{'tests for 90%':>16}{'for 99%':>11}{'for 99.9%':>12}")
print("-" * 69)
need = {}


def tests_for(paths, target, skew=1.2):
    n = min(paths, 2_000_000)
    r = np.arange(1, n + 1, dtype=float)
    w = r ** (-skew)
    w /= w.sum()
    c = np.cumsum(w)
    idx = int(np.searchsorted(c, target)) + 1
    return min(idx, n)


for name, p in shapes:
    need[name] = tuple(tests_for(p, t) for t in (0.90, 0.99, 0.999))
    print(f"{name:>30}{need[name][0]:>16,}{need[name][1]:>11,}"
          f"{need[name][2]:>12,}")

print(f"""
The first table is the claim, and it holds. A graph with {6} nodes and two edges
each has {cov['graph, 6 nodes, 2 edges each'][0]} distinct paths, and
{TESTS} tests cover all of them. A free loop at horizon {16} has
{cov['free loop, horizon 16'][0]:,}, of which the same suite covers
{cov['free loop, horizon 16'][1]:.2%}.

**A graph bounds the path set** (eq:graph-bounds-the-paths), and that is the one
thing it does that a loop cannot. Everything else about graph orchestration is a
convenience; this is a property.

The second table applies ch:ag-what-is-an-agent's correction, which is that path
coverage is the wrong statistic because paths are wildly unequal. On mass, the
free loop at horizon {10} is at {mc['free loop, horizon 10']:.1%} and the graph at
{mc['graph, 10 nodes, 2 edges each']:.1%}.

That narrows the gap considerably. The loop is not as untestable as its path count
suggests, because most runs take one of a few routes -- which was
ch:ag-what-is-an-agent's finding and it survives here.

The third table says when the gap matters, and it is entirely about how
concentrated the runs are. At skew {2.2} -- runs overwhelmingly following a few
routes -- the loop covers {sk[2.2][1]:.1%} of mass with {TESTS} tests and the graph
{sk[2.2][0]:.1%}. At skew {0.6}, where behaviour is genuinely varied, the loop
covers {sk[0.6][1]:.1%} and the graph {sk[0.6][0]:.1%}.

**A graph's testability advantage is largest exactly where an agent's behaviour is
most varied** -- which is where you wanted an agent. That tension is the chapter,
and the next listing prices the other side of it.

The last table is the practical form. Reaching {0.99:.0%} mass coverage needs
{need['graph, 10 nodes, 2 edges each'][1]:,} tests for a ten-node graph and
{need['free loop, horizon 10'][1]:,} for a horizon-10 loop; at horizon {16} it is
{need['free loop, horizon 16'][1]:,}.

Those are the numbers to put beside a testing budget. A graph makes exhaustive-ish
testing a purchasing decision; a loop makes it impossible and leaves you with
sampling and an explicit uncovered remainder.""")
```

The second listing prices the cost.

```python {tier=A name=graph-surrenders-the-tail}
"""What a graph gives up, which is the thing that motivated the agent.

The previous listing measured what a graph buys: a path set small enough to test.
This one measures what it costs, and the cost is the same quantity
ch:ag-what-is-an-agent used to justify having an agent at all -- tail coverage
(eq:graph-surrenders-the-tail).

A graph can only execute routes its author drew. A request needing a sequence
nobody anticipated fails, exactly as a router fails on a shape nobody enumerated.
So a graph is a router over paths rather than over flows, and the same arithmetic
applies with a larger enumerable set.
"""
import numpy as np

rng = np.random.default_rng(3331)

M = 60000
P_STEP = 0.93
STEPS_HEAD = 4
STEPS_TAIL = 8
P_EDGE = 0.96           # the graph picks the right edge at a branch
BRANCHES = 3            # branch points on a typical route


def run(tail_mass, shape, m=M, escape=0.0):
    """shape: 'graph' | 'loop' | 'hybrid'. `escape` is the share of tail
    requests a hybrid routes out of the graph and into a free loop."""
    is_tail = rng.random(m) < tail_mass
    steps = np.where(is_tail, STEPS_TAIL, STEPS_HEAD)
    if shape == "loop":
        ok = (rng.random((m, STEPS_TAIL)) < P_STEP)
        ok = np.array([ok[i, :steps[i]].all() for i in range(m)])
        return float(ok.mean()), float(steps.mean())
    if shape == "graph":
        # Head requests follow a drawn route; each branch may be mis-taken.
        edges_ok = (rng.random((m, BRANCHES)) < P_EDGE).all(1)
        body_ok = (rng.random((m, STEPS_HEAD)) < P_STEP).all(1)
        ok = (~is_tail) & edges_ok & body_ok
        return float(ok.mean()), float(np.where(is_tail, 1.0,
                                                STEPS_HEAD).mean())
    if shape == "hybrid":
        edges_ok = (rng.random((m, BRANCHES)) < P_EDGE).all(1)
        body_ok = (rng.random((m, STEPS_HEAD)) < P_STEP).all(1)
        graph_ok = (~is_tail) & edges_ok & body_ok
        routed = is_tail & (rng.random(m) < escape)
        loop_ok = (rng.random((m, STEPS_TAIL)) < P_STEP).all(1)
        ok = graph_ok | (routed & loop_ok)
        cost = np.where(is_tail, np.where(routed, STEPS_TAIL + 1, 1.0),
                        STEPS_HEAD)
        return float(ok.mean()), float(cost.mean())
    raise ValueError(shape)


TAILS = [0.0, 0.05, 0.15, 0.30, 0.50]

print(f"{M:,} requests. Head requests take {STEPS_HEAD} steps on a drawn route")
print(f"with {BRANCHES} branch points ({P_EDGE:.0%} per branch); tail requests")
print(f"take {STEPS_TAIL} steps and no route exists for them. Steps are")
print(f"{P_STEP:.0%} reliable.")
print()
print(f"{'tail mass':>11}{'graph':>20}{'free loop':>20}")
print(f"{'':>11}{'success':>11}{'steps':>9}{'success':>11}{'steps':>9}")
print("-" * 51)
tab = {}
for t in TAILS:
    g = run(t, "graph")
    l = run(t, "loop")
    tab[t] = (g, l)
    print(f"{t:>11.0%}{g[0]:>11.1%}{g[1]:>9.1f}{l[0]:>11.1%}{l[1]:>9.1f}")

print()
print()
print("The hybrid: a graph for the head, an escape hatch to a free loop for")
print("anything the graph cannot route.")
print()
print(f"{'tail mass':>11}{'graph':>9}{'loop':>9}{'hybrid':>10}{'steps':>9}"
      f"{'best':>10}")
print("-" * 58)
hy = {}
for t in TAILS:
    g = tab[t][0][0]
    l = tab[t][1][0]
    h = run(t, "hybrid", escape=1.0)
    hy[t] = h
    best = max([("graph", g), ("loop", l), ("hybrid", h[0])],
               key=lambda x: x[1])[0]
    print(f"{t:>11.0%}{g:>9.1%}{l:>9.1%}{h[0]:>10.1%}{h[1]:>9.1f}{best:>10}")

print()
print()
print("What the hybrid costs in testability, since the escape hatch is a free")
print("loop and therefore unenumerable. Share of RUNS that stay in the graph:")
print()
print(f"{'tail mass':>11}{'runs in graph':>16}{'runs in the loop':>19}"
      f"{'enumerable share':>19}")
print("-" * 65)
for t in TAILS:
    print(f"{t:>11.0%}{1 - t:>16.1%}{t:>19.1%}{1 - t:>19.1%}")

print()
print()
print("And how branch reliability limits the graph, independent of the tail.")
print()
print(f"{'branches':>10}" + "".join(f"{'edge ' + format(p, '.0%'):>13}"
                                    for p in (0.99, 0.96, 0.90))
      + f"{'free loop':>12}")
print("-" * 61)
br = {}
BR_SAVE = BRANCHES
for b in (1, 3, 6, 10):
    globals()["BRANCHES"] = b
    row = []
    for pe in (0.99, 0.96, 0.90):
        PE = P_EDGE
        globals()["P_EDGE"] = pe
        row.append(run(0.0, "graph")[0])
        globals()["P_EDGE"] = PE
    globals()["BRANCHES"] = BR_SAVE
    br[b] = row
    print(f"{b:>10}" + "".join(f"{v:>13.1%}" for v in row)
          + f"{tab[0.0][1][0]:>12.1%}")

print(f"""
The first table contains a result the chapter was not written to expect, and it is
in the first row rather than the last.

At {0:.0%} tail mass -- every request fits a route the author drew, which is the
graph's best case -- the graph scores {tab[0.0][0][0]:.1%} against the free loop's
{tab[0.0][1][0]:.1%}. **The graph is behind where it should be strongest.**

The reason is the branch decisions. A drawn route has branch points, and each is a
decision that can go wrong: {BRANCHES} branches at {P_EDGE:.0%} is
{P_EDGE ** BRANCHES:.1%} before any step executes. **A graph does not remove the
model's uncertainty; it relocates it from "which action" to "which edge"** -- and
an edge choice is a classification over a set the author fixed, which is not
obviously easier.

The last table isolates that. One branch at {0.99:.0%} gives {br[1][0]:.1%},
essentially matching the loop. Ten branches at {0.90:.0%} gives {br[10][2]:.1%}.
**Branch count is an exponent**, exactly as handoff count was in
ch:as-multi-agent, and a richly-branched graph pays it on every request.

Then the tail, which is the cost the chapter was written to measure. At
{0.5:.0%} tail mass the graph scores {tab[0.5][0][0]:.1%} against the loop's
{tab[0.5][1][0]:.1%}, because a request needing a sequence nobody drew has nowhere
to go (eq:graph-surrenders-the-tail). That is ch:ag-what-is-an-agent's router
result with a larger enumerable set: **a graph is a router over paths.**

The second table adds the obvious fix, and it does not rescue the design here. A
hybrid -- graph for the head, escape to a free loop for anything unroutable --
reaches {hy[0.15][0]:.1%} at {0.15:.0%} tail against the pure loop's
{tab[0.15][1][0]:.1%}. It beats the pure graph everywhere and beats the loop
nowhere, because the graph half is itself behind.

The third table is what the hybrid costs in the currency the graph was bought
with. At {0.15:.0%} tail mass, {0.15:.0%} of runs leave the graph and enter an
unenumerable loop, so the enumerable share of behaviour is {0.85:.0%}. **An escape
hatch converts a bounded path set back into an unbounded one at exactly the rate
it is used**, and the runs that use it are the unusual ones -- which
ch:ag-what-is-an-agent noted are the consequential ones.

So the honest summary of these two listings is narrower than the usual case for
graphs, and it is not about reliability.

**A graph buys testability and pays for it twice**: in branch decisions, which cost
{1 - P_EDGE ** BRANCHES:.1%} on every request at this branching factor, and in tail
coverage, which is the thing that motivated using an agent at all.

Where that trade is worth taking is where the path set is genuinely small and the
tail is genuinely thin -- a workflow with a handful of branches, which is
ch:ag-what-is-an-agent's router by another name. Where the tail is thick, the graph
is buying testability with the capability that justified the system.

And there is one thing a graph buys that neither listing measures and that is
probably its strongest justification in practice: **the control flow becomes an
artefact a human can read, review, version and diff.** That is not a reliability
property and it is worth a great deal on a system several people maintain. It
should be argued for on those terms rather than on the numbers above.""")
```

## 9. Practical Example

The first listing counts paths and coverage against a $400$-case test suite.

```
                         shape         paths    covered   coverage
------------------------------------------------------------------
  graph, 6 nodes, 2 edges each            32         32    100.00%
 graph, 10 nodes, 2 edges each           512        400     78.12%
          free loop, horizon 6           729        400     54.87%
         free loop, horizon 16    43,046,721        400      0.00%
```

**A graph bounds the path set** ({{eq:graph-bounds-the-paths}}) and that is the one
thing it does that a loop cannot.

But path coverage is the wrong statistic:

```
                         shape       paths   path coverage   mass covered
-------------------------------------------------------------------------
 graph, 10 nodes, 2 edges each         512          78.12%          98.3%
         free loop, horizon 10      59,049           0.68%          81.1%
         free loop, horizon 16  43,046,721           0.00%          76.8%
```

The loop is far more testable than its path count suggests, because most runs take
a few routes. And the advantage sits in the wrong place:

```
    skew   graph, 10 nodes   loop, horizon 10   loop, horizon 16
----------------------------------------------------------------
     0.6             90.0%              12.7%               3.1%
     1.2             98.3%              81.1%              76.8%
     2.2            100.0%             100.0%             100.0%
```

At high concentration the two are identical; at low concentration the graph leads by
$77.3$ points. **The graph's testability advantage is largest exactly where
behaviour is most varied** — which is where you wanted an agent
({{eq:mass-under-skew}}).

The second listing measures the cost, and the first row is the surprise:

```
  tail mass               graph           free loop
               success    steps    success    steps
---------------------------------------------------
         0%      66.1%      4.0      75.1%      4.0
        15%      56.1%      3.5      71.9%      4.6
        50%      33.2%      2.5      65.5%      6.0
```

At *zero* tail mass — the graph's best case — it scores $66.1\%$ against $75.1\%$.
**The graph is behind where it should be strongest**, because a drawn route has
branch points and each is a decision that can go wrong: three branches at $96\%$ is
$88.5\%$ before any step runs.

```
  branches     edge 99%     edge 96%     edge 90%   free loop
-------------------------------------------------------------
         1        74.4%        71.7%        67.4%       75.1%
         3        72.4%        66.1%        54.9%       75.1%
        10        67.6%        49.5%        25.9%       75.1%
```

**Branch count is an exponent** ({{eq:branch-count-is-an-exponent}}), exactly as
handoff count was in {{ch:as-multi-agent}}. And note the instinct a misrouting graph
provokes — add more conditions — makes this worse.

The escape hatch does not rescue it:

```
  tail mass    graph     loop    hybrid    steps      best
----------------------------------------------------------
         0%    66.1%    75.1%     66.0%      4.0      loop
        15%    56.1%    71.9%     64.6%      4.8      loop
        50%    33.2%    65.5%     61.3%      6.5      loop
```

The hybrid beats the pure graph everywhere and the loop nowhere, because the graph
half is itself behind. And it costs the property the graph was bought for:

```
  tail mass   runs in graph   runs in the loop   enumerable share
-----------------------------------------------------------------
        15%           85.0%              15.0%              85.0%
        50%           50.0%              50.0%              50.0%
```

**An escape hatch unbounds the path set at exactly the rate it is used**
({{eq:escape-hatch-unbounds-again}}), and the runs that use it are the unusual ones.

So: a graph buys testability and pays for it twice — in branch decisions, and in the
tail coverage that motivated using an agent. The trade is worth taking where the
path set is genuinely small and the tail genuinely thin, which is
{{ch:ag-what-is-an-agent}}'s router by another name.

## 10. Production Considerations

Measure $p_e$ per branch from traces: did the run take the edge a human would have
chosen? It is a first-order number and it is not in any dashboard.

Count your branches and treat the count as an exponent. Refactor toward fewer,
wider decision points rather than more, narrower ones.

Make edge conditions structural wherever possible — a typed field rather than a
judgement. That sets $p_e = 1$ for those edges and is the cheapest available
improvement.

Check your edge conditions for overlap, the way {{ch:ag-tool-calling}} checks tool
descriptions. Edge inventories accrete the same way.

Instrument the escape hatch: its rate is your testability claim, and clusters within
it are routes you should draw.

Report mass coverage, not path coverage, and report it alongside the hatch rate.
"$100\%$ of paths tested" over $85\%$ of runs is a different claim.

And adopt graphs for maintainability if that is the reason. It is a good reason, and
it survives the numbers in this chapter in a way that the reliability argument does
not.

## 11. Common Mistakes

**Adopting a graph for reliability.** It lost to a free loop at zero tail mass.

**Adding conditions when a graph misroutes.** Branch count is an exponent
({{eq:branch-count-is-an-exponent}}).

**Reporting path coverage.** Paths are unequal; mass is the statistic.

**Ignoring the escape-hatch rate.** It is $1 -$ your testability claim.

**Edge conditions phrased as judgements where a field would do.** Free $p_e$ left
on the table.

**Assuming the graph's advantage grows with the system.** It shrinks as runs
concentrate and grows as they spread — the opposite of the usual intuition.

**Never measuring $p_e$.** Everything in {{eq:branch-count-is-an-exponent}} turns on
it.

## 12. Failure Modes

*Silent misrouting.* A wrong edge sends the request down a valid route that answers
a different question. Visible in the graph, which is its diagnosis benefit, and only
if anyone looks.

*Condition overlap.* Two edges matching the same situation, resolved arbitrarily —
{{ch:ag-tool-calling}}'s distinctness problem at the edge set.

*Hatch creep.* The escape hatch handling a growing share of traffic, quietly
converting a graph system into a loop system with extra steps.

*Branch proliferation.* Each misroute prompting a new condition, compounding
$p_e^{\beta}$.

*Dead routes.* Drawn paths that no traffic takes, which cost review effort and
nothing else — the graph's version of an unused tool.

## 13. Alternatives

**A free loop with checkpoints.** {{ch:ag-planning}}: structure without edges, so no
$p_e^{\beta}$ term.

**A router with an agent fallback.** {{ch:ag-what-is-an-agent}}: the same trade at a
coarser grain, with fewer decision points.

**Structural edges only.** A graph whose conditions are all typed fields has
$p_e = 1$ and keeps the path bound — this is a state machine, and it is
{{ch:as-state-machines}}'s subject.

**Explicit flow, free execution.** Write the flow down for humans and let the agent
run freely, using the diagram as documentation and evaluation structure rather than
as control. Captures {{sec:7-internal-mechanics}}'s real benefit at none of the cost.

**A workflow.** If the tail is thin enough for a graph, it may be thin enough for a
pipeline.

## 14. Evaluation

Report mass coverage and the number of tests, not path coverage.

Report the escape-hatch rate as part of any testability claim.

Measure $p_e$ per branch and report the product $p_e^{\beta}$ — it is your
route-selection ceiling.

Evaluate at your measured run concentration. {{eq:mass-under-skew}} says the
graph's advantage depends on it entirely, and it is estimable from a trace sample.

And compare against a free loop at equal budget, as {{ch:as-multi-agent}}'s
discipline requires — the graph's step savings on unroutable requests are not a
benefit if those requests failed.

## 15. Advanced Concepts

**Learned edge conditions.** $p_e$ is a classification accuracy, so it is trainable
from traces where a human labelled the correct edge. Almost nobody does this, and it
is the highest-leverage improvement available to a graph system.
{{maturity:EMERGING}}.

**Graph synthesis from traces.** Clustering hatch runs by route and promoting
frequent ones to drawn edges converts tail into head automatically. That is
{{eq:tail-mass-decides}}'s "enumerate more shapes" made continuous.

**Hierarchical graphs.** Reducing $\beta$ per level while keeping total expressivity
attacks the exponent directly, and it is the structural response to
{{eq:branch-count-is-an-exponent}}.

**Verified edges.** If every edge condition were a checkable predicate over typed
state, $p_e^{\beta} = 1$ and the graph's only cost would be the tail. That is a
state machine, and the question of how much agent capability survives full
typing is {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-what-is-an-agent}}'s path explosion and mass-coverage correction are both
applied here, and its tail-mass equation governs
{{eq:graph-is-a-router-over-paths}} — a graph is a router over paths.

{{ch:as-multi-agent}}'s "a count in an exponent is a design parameter" recurs as
branch count, and the refactoring advice is the same.

{{ch:ag-tool-calling}}'s distinctness result transfers to edge conditions, which
accrete and overlap exactly as tool descriptions do.

{{ch:ag-planning}}'s checkpoints are the alternative structure — bounded recovery
without bounded routes.

Ahead: {{ch:as-state-machines}} takes the structural-edge idea to its conclusion,
where $p_e = 1$ and durability becomes the subject.

## 17. Exercises

1. Solve {{eq:graph-is-a-router-over-paths}} for the $p_e$ at which a graph matches
   a loop at zero tail mass, for $\beta \in \{1, 3, 10\}$.

2. Add retries within graph nodes and re-run the first table. How much of the
   graph's deficit does that recover?

3. Measure mass coverage for a real trace set of your own and estimate the skew.
   Which side of {{eq:mass-under-skew}}'s crossover are you on?

4. Implement the graph-synthesis idea: cluster hatch runs and promote frequent
   routes. How much tail becomes head after one round?

5. Model hierarchical graphs — two levels of three branches instead of one level of
   nine — and quantify the reduction in $p_e^{\beta}$.

6. Take a graph you own, count its branches, and estimate $p_e$ by checking a
   hundred routings against your own judgement.

## 18. Interview Questions

1. What does a graph give you that a loop cannot?

2. Your graph misroutes. Should you add a condition?

3. Why can a graph lose to a free loop even when every request fits a drawn route?

4. Your system reports 100% path coverage. What else do you need to know?

5. When is a graph's testability advantage largest, and why is that awkward?

6. What does an escape hatch cost?

## 19. Research Questions

1. What is $p_e$ empirically for edge conditions in deployed graph systems, and how
   much do structural conditions raise it?

2. Can edge conditions be trained from labelled traces, and how much does that move
   $p_e^{\beta}$?

3. Does automatic graph synthesis from hatch clusters converge, or does the tail
   regenerate as fast as it is drawn?

4. How much agent capability survives requiring every edge condition to be a
   checkable predicate?

5. What is the real run-concentration skew in production agent systems, and does it
   justify the graph's testability premium?

## 20. Chapter Summary

A graph bounds the path set ({{eq:graph-bounds-the-paths}}) — $32$ paths against a
loop's $43$ million — and that is the one property it supplies. Everything else is a
convenience.

The advantage is smaller than the counts suggest and sits in the wrong place. On
mass coverage, a horizon-10 loop already covers $81.1\%$ of runs with $400$ tests,
and the graph's edge *grows* as behaviour gets more varied
({{eq:mass-under-skew}}) — which is the regime an agent exists for.

At its parameters the graph loses on reliability too, and at zero tail mass:
$66.1\%$ against $75.1\%$. **A graph does not remove the model's judgement; it
relocates it to the edges**, and three branches at $96\%$ costs $11.5\%$ before any
step runs. **Branch count is an exponent** ({{eq:branch-count-is-an-exponent}}) —
ten branches at $90\%$ takes the graph to $25.9\%$ — and the instinct to add
conditions when a graph misroutes makes it worse.

Then the tail: $33.2\%$ against $65.5\%$ at $50\%$ tail mass, because a graph can
only run routes its author drew ({{eq:graph-is-a-router-over-paths}}). An escape
hatch handles it and dissolves the property the graph was bought for —
**enumerable behaviour falls linearly in hatch rate**
({{eq:escape-hatch-unbounds-again}}), and the runs taking the hatch are the unusual,
consequential ones.

So the honest case for graph orchestration is not reliability and not coverage. It
is that **the control flow becomes an artefact a human can read, review, version and
diff** — a software-engineering property worth a great deal on a system several
people maintain, and one that survives the numbers here in a way the reliability
argument does not.

## 21. Further Reading

{{cite:cemri2025mast}}'s system-design failure category is largely control flow that
does not fit the task, and a graph makes those failures legible — which is the
diagnosis benefit {{sec:7-internal-mechanics}} separates from prevention.

{{ch:ag-what-is-an-agent}} for the path and mass arithmetic this chapter reuses, and
{{ch:as-multi-agent}} for the exponent-in-a-count pattern.

{{cite:zhou2024webarena}} and {{cite:liu2024agentbench}} for environments whose run
concentration would decide {{eq:mass-under-skew}} in practice — the measurement this
chapter says nobody makes.

{{ch:as-state-machines}} next, for what happens when every edge condition is
required to be checkable.
