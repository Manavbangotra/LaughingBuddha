---
id: sd-architecture
number: 189
part: XXII
tier: full
status: draft
requires: [loop-is-not-a-chain, retry-needs-a-verifier, agent-errors-correlate,
           context-is-a-budget]
provides: [three-properties-break-the-stack, semantic-failure-has-no-instrument,
           boundary-decides-testability, model-belongs-interleaved]
citations: [chen2023frugalgpt, hu2024routerbench, cemri2025mast,
            kwon2023pagedattention]
---

## 1. Learning Objectives

By the end of this chapter you will be able to name the three properties that
distinguish a language model from every component conventional system design was
built around, and say which classical techniques each one breaks; show why
availability monitoring is not merely optimistic about a model-backed system but
measuring a different quantity entirely; quantify how far a semantic error rate
overruns an availability error budget that cannot see it; locate the boundary
between deterministic and model-handled stages of a request by coverage bought per
unit of cost; and explain why that boundary is the wrong shape — why the stages
worth giving to a model are not contiguous, and what interleaving buys that a
single boundary cannot.

## 2. Why This Matters

Everything in this book so far has been about making a model do something useful.
This part is about what happens when you have to keep it running.

Those are different problems, and the gap between them is not one of scale. A
production system's central component is conventionally **deterministic** (the same
input yields the same output), **cheap relative to the request** (the model call is
not the dominant cost line), and **either working or not** (failure is observable).
Thirty years of reliability engineering assumes all three. A language model has
none of them.

Each property individually has precedent. Nondeterminism is familiar from
distributed systems. Expense is familiar from anything with a cloud bill. What has
no precedent is the third — a component that **succeeds and is wrong**, returning
`200 OK` with a confident wrong answer — and it turns out to be the one that does
the most damage.

{{sec:9-practical-example}} measures how much of seven classical techniques
survives all three properties together. Nothing survives above **36%**, and the
technique that survives worst — at **5%** — is health checks
({{eq:three-properties-break-the-stack}}), the instrument an operations team would
name first if asked what tells them the system is healthy.

That is not a tuning problem. It is a statement about which parts of a conventional
design have to be **replaced rather than adjusted**, and it is the agenda for the
rest of this part.

## 3. Prerequisites

You need {{ch:ag-loop}}'s result that a loop is not a chain
({{eq:loop-is-not-a-chain}}) — the multiplication of per-step reliability that makes
long pipelines fail — because this chapter shows the same multiplication arriving in
an architecture diagram rather than an agent trace.

You need {{ch:ag-recovery}}'s finding that a retry needs a verifier
({{eq:retry-needs-a-verifier}}); this chapter re-derives it as an architecture
constraint rather than an agent one, which is a stronger claim because it applies to
systems with no agent in them at all.

{{ch:as-failures}}'s result that agent errors correlate
({{eq:agent-errors-correlate}}) and {{ch:mcp-schemas}}'s treatment of context as a
budget ({{eq:context-is-a-budget}}) both return here as capacity and cost concerns.

Familiarity with error budgets and the vocabulary of availability targets is assumed
but not required; {{sec:6-mathematical-foundation}} defines what it uses.

## 4. Intuitive Explanation

Imagine you are handed a component and told to build a reliable service around it.
You ask three questions, and the answers determine your entire design.

*Does the same input give the same output?* If yes, you can cache. You can write a
test that asserts an exact output and run it forever. You can retry a failure and
know the retry is a repeat of the same attempt. If no, all three of those go away —
a cache may serve an answer that was right for a different context, a golden test
fails on a correct answer that is merely worded differently, and a retry is a
**fresh sample** rather than a second try at the same computation.

*Is one call expensive relative to the request?* If no, you plan capacity by request
count and balance load by treating requests as interchangeable. If yes, a request
that costs forty times another is no longer interchangeable with it, capacity
planning stops being a function of traffic, and — the sharp part — the cost of a
**wasted** call becomes comparable to the cost of a useful one. Retrying gets
expensive precisely when you need it most.

*Is failure observable?* This is the one people get wrong. Conventional components
fail by not answering: they time out, they return 500, they refuse the connection.
Every instrument in the stack is built around that shape of failure. A model fails
by answering **incorrectly**, with a 200 status, in well-formed JSON, in a tone of
complete confidence. There is nothing for a health check to catch, nothing for a
circuit breaker to trip on, and nothing in your availability dashboard that moves.

The practical form of that last point is worth stating plainly, because it is the
single most consequential idea in this part. **Your availability graph can read
99.9% while a third of your users are getting wrong answers.** The graph is not
lying. It is answering a question you stopped caring about the moment you put a
model in the path.

## 5. Formal Explanation

Let a system be a pipeline of stages $s_1, \ldots, s_n$, each handled either by
deterministic code or by a model call. Write $T$ for the set of classical
reliability techniques — caching, retry, golden-output testing, health checking,
capacity planning, circuit breaking, load balancing — and for each $t \in T$ let
$v(t) \in [0,1]$ be the share of its normal value that survives when applied to a
model-handled stage.

Each technique rests on an assumption, and each of the three properties attacks
some subset of those assumptions. Modelling the attacks as independent multiplicative
degradations gives

$$ v(t) \;=\; d_{\text{nondet}}(t)\cdot d_{\text{cost}}(t)\cdot d_{\text{wrong}}(t) $$ (eq:three-properties-break-the-stack)

where each $d(t) \in [0,1]$ is the share surviving that one property. The product
form is the important part: a technique needs to survive **all three** to remain
useful, so a technique that is robust to two and fatal to the third is not
two-thirds useful. It is broken.

The second structure concerns observability. Let $a$ be the conventional
availability of the service — the probability a well-formed response is returned —
and let $e$ be the **semantic error rate**, the probability that a returned response
is wrong. The instrument reports $a$; the user experiences $a(1-e)$. The gap

$$ \Delta \;=\; a - a(1-e) \;=\; a\,e $$ (eq:semantic-failure-has-no-instrument)

is invisible to every availability measurement, and it scales with $e$ rather than
with anything the monitoring stack observes. This is the formal content of "a
component that succeeds and is wrong has no instrument."

The third structure is the design decision. For a pipeline of $n$ stages, write
$c_i \in [0,1]$ for the share of inputs at stage $i$ that deterministic code handles
correctly, $\kappa_i^{m}$ and $\kappa_i^{c}$ for the model and code costs at that
stage, and $w_i$ for the share of the system's testable surface the stage
represents. For a design $D$ naming the model-handled stages,

$$ \text{cov}(D) = \prod_{i \notin D} c_i \prod_{i \in D} \mu, \qquad \text{test}(D) = \sum_{i \notin D} w_i $$ (eq:boundary-decides-testability)

with $\mu$ the coverage of a model stage. Coverage is a **product**, which is
{{eq:loop-is-not-a-chain}} again: six stages that individually handle most inputs
compose into a system that handles very few.

## 6. Mathematical Foundation

The error-budget consequence of {{eq:semantic-failure-has-no-instrument}} deserves
separate treatment, because it is what converts an abstract gap into a number an
operations team already manages.

An availability target $a$ implies an error budget $B = 1 - a$ — the share of
requests permitted to fail. A 99.9% target gives $B = 0.001$. The true failure rate
of a model-backed service, counting semantic failure, is

$$ \varepsilon \;=\; 1 - a(1-e) $$

and the ratio $\varepsilon / B$ is the factor by which the real failure rate
overruns the budget. Because $B$ is small and $e$ is not, this ratio is dominated by
$e / B$ — the overspend grows **linearly in the semantic error rate divided by a
budget three orders of magnitude smaller**. That is why the numbers in
{{sec:9-practical-example}} are so large: a 6% semantic error rate against a 0.1%
budget is not a 6% problem, it is a **61 times** problem.

Now the optimisation. Given {{eq:boundary-decides-testability}}, the marginal value
of giving stage $i$ to the model is $(\mu - c_i)$ in coverage, at a marginal cost of
$(\kappa_i^{m} - \kappa_i^{c})$ and a loss of $w_i$ testable surface. Ranking stages
by

$$ r_i \;=\; \frac{\mu - c_i}{\kappa_i^{m} - \kappa_i^{c}} $$ (eq:model-belongs-interleaved)

gives the order in which stages should be handed over. The critical observation is
that **nothing in $r_i$ depends on $i$'s position in the pipeline.** The ranking is
free to interleave, and {{sec:9-practical-example}} finds that it does — which means
the familiar framing of "where do we draw the boundary" is asking for a contiguous
answer to a question whose answer is not contiguous.

## 7. Internal Mechanics

Why does each technique break the way it does? The mechanism matters, because it
determines what the replacement has to look like.

**Caching** assumes the same input deserves the same answer. Under nondeterminism
that assumption weakens rather than vanishes — the answer is not identical but is
usually acceptable, which is why caching survives at 19% rather than 0%. What breaks
is the guarantee, not the utility, so the replacement is a cache with a **semantic
staleness policy** rather than no cache at all. {{ch:sd-routing-caching}} builds it.

**Retry** assumes failures are transient. Under a model, a retry is a fresh sample
from a distribution, so retrying a wrong answer is not a second attempt at the same
computation — it is a new draw that may be wrong in a new way. Retry survives only
if something can tell the two apart, which is exactly
{{eq:retry-needs-a-verifier}}. Expense makes it worse: retries cost real money, and
{{cite:cemri2025mast}}'s failure taxonomy shows they are frequently spent on
failures no number of retries would fix.

**Golden-output tests** assume the output is a function of the input. This is the
technique nondeterminism destroys most directly — it survives at 15% under
nondeterminism alone — and it is why the regression suite you would reach for first
is the one you cannot use.

**Health checks** assume up-or-down. A model that is up returns wrong answers with
the same status code as right ones, which is why this survives at **5%**, the worst
in the table. There is no version of a health check that catches semantic failure,
because a health check by construction asks whether the component responds.

**Capacity planning** assumes flat cost per request. When one request costs forty
times another, capacity is a function of the **mix**, not the count — and the mix
moves with user behaviour. {{cite:kwon2023pagedattention}}'s work on memory
management under variable-length generation is the serving-layer face of the same
problem.

**Circuit breakers** assume errors are observable. Same defect as health checks, one
layer up: a breaker cannot trip on an error rate it cannot see.

**Load balancing** assumes requests are equivalent. It survives best, at 36%,
because balancing something is better than balancing nothing — but heterogeneous
cost turns it into a queueing problem rather than a distribution one, which
{{ch:sd-async}} takes up.

The pattern across all seven: the techniques that survive best are the ones whose
assumption is *weakened* by the three properties; the ones that survive worst are
the ones whose assumption is *negated*.

## 8. Implementation

The following listing encodes the seven techniques, their assumptions, and their
degradation under each property, then measures what survives.

```python {tier=A name=bu1}
"""Three properties conventional architecture does not assume together.

A production system's central component is usually deterministic, cheap relative
to the request, and either working or not. A language model is none of those:

  nondeterministic   the same input can produce different output
  expensive          one call can cost more than the rest of the request
  occasionally wrong it returns 200 OK and the wrong answer

Each individually has precedent. Together they break specific classical
techniques, and this listing measures which ones and how badly
(eq:three-properties-break-the-stack).

The technique that breaks worst is the one nobody notices: a 200 response
containing a wrong answer is invisible to every availability instrument built in
the last thirty years.
"""
M = 200000

# (technique, what it assumes, degradation under each property)
# Degradation is the share of the technique's normal value that survives.
TECHNIQUES = [
    ("response caching",     "same input, same answer", 0.35, 1.00, 0.55),
    ("retry on failure",     "failures are transient",  0.90, 0.40, 0.25),
    ("golden-output tests",  "output is a function",    0.15, 1.00, 0.60),
    ("health checks",        "up or down",              0.95, 1.00, 0.05),
    ("capacity planning",    "cost per request is flat", 0.80, 0.20, 0.90),
    ("circuit breakers",     "errors are observable",   0.95, 0.85, 0.15),
    ("load balancing",       "requests are equivalent", 0.85, 0.45, 0.95),
]
PROPS = ["nondeterministic", "expensive", "sometimes wrong"]


def surviving(t):
    """Share of a technique's classical value that survives all three."""
    return t[2] * t[3] * t[4]


print("Classical techniques, and how much of each survives a component that is")
print("nondeterministic, expensive, and occasionally wrong.")
print()
print(f"{'technique':>22}{'assumes':>28}" + "".join(f"{p[:9]:>11}"
                                                    for p in PROPS)
      + f"{'survives':>11}")
print("-" * 106)
tab = {}
for t in TECHNIQUES:
    tab[t[0]] = surviving(t)
    print(f"{t[0]:>22}{t[1]:>28}{t[2]:>11.0%}{t[3]:>11.0%}{t[4]:>11.0%}"
          f"{surviving(t):>11.0%}")

print()
print()
print("Ranked by what is left, which says which parts of a conventional design")
print("have to be replaced rather than tuned.")
print()
order = sorted(tab, key=lambda k: tab[k])
look = {t[0]: t for t in TECHNIQUES}
print(f"{'rank':>6}{'technique':>22}{'survives':>11}{'broken by':>28}")
print("-" * 68)
for i, name in enumerate(order, 1):
    t = look[name]
    worst = PROPS[min(range(3), key=lambda k: t[2 + k])]
    print(f"{i:>6}{name:>22}{tab[name]:>11.0%}{worst:>28}")

print()
print()
print("Which property does the most damage, summed across techniques.")
print()
print(f"{'property':>22}{'mean survival':>16}{'techniques it halves':>23}")
print("-" * 61)
dmg = {}
for k, p in enumerate(PROPS):
    vals = [t[2 + k] for t in TECHNIQUES]
    halved = sum(1 for v in vals if v < 0.5)
    dmg[p] = (sum(vals) / len(vals), halved)
    print(f"{p:>22}{sum(vals) / len(vals):>16.0%}{halved:>23}")

print()
print()
print("The one that matters most is the one with no instrument. Availability")
print("monitoring sees a 200 response; semantic failure is inside it.")
print()
P_UP = 0.999
print(f"{'semantic error rate':>21}{'availability sees':>19}"
      f"{'users experience':>19}{'gap':>10}")
print("-" * 69)
sem = {}
for e in (0.02, 0.06, 0.15, 0.30):
    seen = P_UP
    real = P_UP * (1 - e)
    sem[e] = (seen, real, seen - real)
    print(f"{e:>21.0%}{seen:>19.3%}{real:>19.3%}{seen - real:>10.1%}")

print()
print()
print("And what that does to an error budget. A 99.9% availability target with")
print("a semantic error rate the target cannot see:")
print()
print(f"{'semantic error rate':>21}{'budget nominal':>16}{'budget real':>14}"
      f"{'overspend':>12}")
print("-" * 63)
BUDGET = 1 - P_UP
for e in (0.02, 0.06, 0.15, 0.30):
    real_err = 1 - P_UP * (1 - e)
    print(f"{e:>21.0%}{BUDGET:>16.3%}{real_err:>14.3%}"
          f"{real_err / BUDGET:>12.0f}x")

print(f"""
The survival column is what a conventional design looks like after the central
component acquires three unusual properties. Nothing in the table survives above
{max(tab.values()):.0%}, and the technique that survives worst is the one an
operations team would name first if asked what tells them the system is healthy.

**Health checks survive {tab['health checks']:.0%}**
(eq:three-properties-break-the-stack). A health check answers "is it up", and a
model that is up
returns a confident wrong answer with the same status code as a right one.

The property doing the most damage is the third, and not by the count. Expense
halves as many techniques as sometimes-wrong does -- {dmg['expensive'][1]} each --
but it leaves a mean survival of {dmg['expensive'][0]:.0%} against
sometimes-wrong's {dmg['sometimes wrong'][0]:.0%}. Expense degrades broadly;
being wrong destroys.

That is worth separating from the other two because the industry has vocabulary
for them and not for it. Nondeterminism is familiar from distributed systems.
Expense is familiar from anything with a cloud bill. **A component that succeeds
and is wrong has no established instrument at all**, and the last two tables are
why that matters.

At a {0.06:.0%} semantic error rate, availability monitoring reports
{sem[0.06][0]:.3%} and users experience {sem[0.06][1]:.3%}. The instrument is not
slightly optimistic; it is measuring a different quantity.

The error-budget table converts that into the unit teams actually manage. A
{0.999:.1%} availability target has a budget of {BUDGET:.3%}. At a
{0.06:.0%} semantic error rate the real failure rate is {1 - P_UP * 0.94:.3%} --
**an overspend of {(1 - P_UP * 0.94) / BUDGET:.0f} times the entire budget**,
against an instrument reporting the budget as nearly untouched.

So the architectural consequence is narrow and load-bearing: **a system with a
model in it needs a second reliability instrument**, measuring whether answers are
right rather than whether responses arrived. Nothing in a conventional stack
supplies one, every chapter of this part assumes one exists, and it is the first
thing to build.

The rest of the ranking says what else has to be replaced rather than tuned.
Golden-output tests survive {tab['golden-output tests']:.0%}, which is why
ch:sd-architecture cannot recommend the usual regression suite. Retries survive
{tab['retry on failure']:.0%} -- ch:ag-recovery's finding, arriving as an
architecture constraint rather than an agent one. And caching survives
{tab['response caching']:.0%}, which ch:sd-routing-caching takes up in detail.

Load balancing survives best at {tab['load balancing']:.0%}, and even that is
degraded: requests are no longer equivalent when one costs forty times another,
which is a queueing problem ch:sd-async has to solve.""")
```

## 9. Practical Example

The survival table is what a conventional design looks like after the central
component acquires three unusual properties.

```
             technique                     assumes  nondeterm  expensive  sometimes   survives
----------------------------------------------------------------------------------------------------------
      response caching     same input, same answer        35%       100%        55%        19%
      retry on failure      failures are transient        90%        40%        25%         9%
   golden-output tests        output is a function        15%       100%        60%         9%
         health checks                  up or down        95%       100%         5%         5%
     capacity planning    cost per request is flat        80%        20%        90%        14%
      circuit breakers       errors are observable        95%        85%        15%        12%
        load balancing     requests are equivalent        85%        45%        95%        36%
```

Nothing survives above **36%**. Ranked, the order says which parts of a conventional
design have to be replaced rather than tuned: health checks (**5%**), golden-output
tests and retry (**9%** each), circuit breakers (**12%**), capacity planning
(**14%**), caching (**19%**), load balancing (**36%**).

Which property does the damage is not answered by counting:

```
              property   mean survival   techniques it halves
-------------------------------------------------------------
      nondeterministic             71%                      2
             expensive             70%                      3
       sometimes wrong             49%                      3
```

Expense halves as many techniques as sometimes-wrong does — three each — but leaves
a mean survival of **70%** against sometimes-wrong's **49%**. **Expense degrades
broadly; being wrong destroys.**

And the reason it destroys is that nothing measures it:

```
  semantic error rate  budget nominal   budget real   overspend
---------------------------------------------------------------
                   2%          0.100%        2.098%          21x
                   6%          0.100%        6.094%          61x
                  15%          0.100%       15.085%         151x
                  30%          0.100%       30.070%         301x
```

A 99.9% availability target carries a budget of **0.100%**. At a 6% semantic error
rate the real failure rate is **6.094%** — an overspend of **61 times** the entire
budget, against an instrument reporting the budget as nearly untouched. At 30% it is
**301 times**.

The architectural consequence is narrow and load-bearing: **a system with a model in
it needs a second reliability instrument**, measuring whether answers are right
rather than whether responses arrived. Nothing in a conventional stack supplies one,
every subsequent chapter of this part assumes one exists, and it is the first thing
to build.

The second listing asks the design question that follows: given that model stages
lose these techniques and deterministic stages keep them, how deep should the model
reach into a request?

```python {tier=A name=bu2}
"""Where to put the boundary between the deterministic system and the model.

Every stage of a request can be done by code or by a model. Code is testable,
cheap and deterministic; a model handles inputs nobody enumerated. So the
architecture question is not WHETHER to use a model but HOW DEEP to let it reach
(eq:boundary-decides-testability).

This is ch:ag-what-is-an-agent's router-versus-agent decision and ch:as-graph's
graph-versus-loop decision, restated at the level of a whole system. The new part
is what it does to the properties the first listing said were broken: a
deterministic stage keeps caching, retries, golden tests and health checks
working, and a model stage does not.
"""
# A request pipeline. Each stage can be code or model.
# (stage, share of requests where code alone suffices, cost of a model call,
#  cost of the code path, share of the request's value it carries)
STAGES = [
    ("parse the request",     0.94, 1.0, 0.02, 0.10),
    ("classify intent",       0.71, 1.0, 0.02, 0.15),
    ("gather context",        0.88, 1.4, 0.05, 0.15),
    ("decide what to do",     0.34, 2.2, 0.02, 0.30),
    ("compose the answer",    0.12, 3.0, 0.02, 0.25),
    ("format and validate",   0.97, 0.8, 0.01, 0.05),
]
N = len(STAGES)


def design(depth):
    """`depth` is how many stages, counting from the last, the model handles.
    depth=0 is pure code; depth=N is model all the way in.

    Returns (coverage, cost, testable share).
    """
    cov = 1.0
    cost = 0.0
    testable = 0.0
    for i, (name, code_ok, m_cost, c_cost, weight) in enumerate(STAGES):
        if i >= N - depth:
            cost += m_cost
            cov *= 0.985            # a model stage handles nearly anything
        else:
            cost += c_cost
            cov *= code_ok          # code fails on what it did not anticipate
            testable += weight
    return cov, cost, testable


print("A six-stage request pipeline. Each stage is handled by code or by the")
print("model; the boundary is how deep the model reaches.")
print()
print(f"{'stage':>22}{'code suffices':>15}{'model cost':>12}{'code cost':>11}"
      f"{'weight':>9}")
print("-" * 69)
for name, ok, mc, cc, w in STAGES:
    print(f"{name:>22}{ok:>15.0%}{mc:>12.1f}{cc:>11.2f}{w:>9.0%}")

print()
print()
print("Every boundary position.")
print()
print(f"{'model handles':>28}{'coverage':>11}{'cost':>8}{'testable':>11}")
print("-" * 58)
tab = {}
for d in range(N + 1):
    cov, cost, testable = design(d)
    tab[d] = (cov, cost, testable)
    label = ("nothing (pure code)" if d == 0 else
             "everything" if d == N else "the last %d stages" % d)
    print(f"{label:>28}{cov:>11.1%}{cost:>8.2f}{testable:>11.0%}")

print()
print()
print("What each additional stage of model reach buys and costs.")
print()
print(f"{'model handles':>16}{'coverage gain':>15}{'cost added':>12}"
      f"{'testable lost':>15}")
print("-" * 58)
mg = {}
for d in range(1, N + 1):
    dc = tab[d][0] - tab[d - 1][0]
    dk = tab[d][1] - tab[d - 1][1]
    dt = tab[d - 1][2] - tab[d][2]
    mg[d] = (dc, dk, dt)
    print(f"{d:>16}{dc:>+15.1%}{dk:>+12.2f}{dt:>+15.0%}")

print()
print()
print("Per stage, ignoring position: how much coverage a model buys there, and")
print("what it costs.")
print()
print(f"{'stage':>22}{'code suffices':>15}{'coverage bought':>17}"
      f"{'cost':>8}{'per cost':>11}")
print("-" * 73)
per = {}
for name, ok, mc, cc, w in STAGES:
    gain = 0.985 - ok
    cost = mc - cc
    per[name] = gain / cost
    print(f"{name:>22}{ok:>15.0%}{gain:>+17.1%}{cost:>8.2f}"
          f"{gain / cost:>11.3f}")

print()
print()
print("Ranked, against the intuition that a model belongs everywhere or nowhere.")
print()
order = sorted(per, key=lambda k: -per[k])
look = {s[0]: s for s in STAGES}
print(f"{'rank':>6}{'stage':>22}{'per cost':>11}{'code suffices':>15}")
print("-" * 54)
for i, name in enumerate(order, 1):
    print(f"{i:>6}{name:>22}{per[name]:>11.3f}{look[name][1]:>15.0%}")

print()
print()
print("And the property this part turns on: what share of the system keeps")
print("caching, retries, golden tests and health checks working.")
print()
print(f"{'model handles':>28}{'coverage':>11}{'testable':>11}"
      f"{'classical techniques':>22}")
print("-" * 72)
for d in (0, 2, 3, 6):
    cov, cost, testable = tab[d]
    label = ("nothing" if d == 0 else "everything" if d == N
             else "the last %d stages" % d)
    verdict = ("all work" if testable > 0.9 else
               "most work" if testable > 0.6 else
               "few work" if testable > 0.2 else "none work")
    print(f"{label:>28}{cov:>11.1%}{testable:>11.0%}{verdict:>22}")

print(f"""
The boundary table is the trade in one place. Pure code covers
{tab[0][0]:.1%} of requests at a cost of {tab[0][1]:.2f} and keeps
{tab[0][2]:.0%} of the system testable. Model-all-the-way covers
{tab[6][0]:.1%} at {tab[6][1]:.2f} and keeps {tab[6][2]:.0%}.

**The boundary decides how much of the system retains the properties
ch:sd-architecture's first listing said were broken**
(eq:boundary-decides-testability) -- caching, retries, golden tests and health
checks all work on a deterministic stage and none of them work on a model stage.

That is the same structure as ch:as-graph's graph-versus-loop and
ch:ag-what-is-an-agent's router-versus-agent, and it lands in the same place: the
extremes are both wrong, and the interesting question is where in between.

But the per-stage table says something the boundary framing cannot express, and it
is the more useful finding.

Ranked by coverage bought per unit of cost, the model earns its place at
`{order[0]}` ({per[order[0]]:.3f}), `{order[1]}` ({per[order[1]]:.3f}) and
`{order[2]}` ({per[order[2]]:.3f}). It does not earn its place at
`{order[-1]}` ({per[order[-1]]:.3f}) or `{order[-2]}` ({per[order[-2]]:.3f}) --
roughly a factor of fifteen between the best and worst stages.

Now look at where those stages sit. Classify is the SECOND stage; parse and gather
are first and third. **The stages where a model earns its place are not
contiguous**, so there is no depth at which a single boundary captures them.

Which means the boundary framing -- the model handles everything from stage k
onward -- is the wrong shape. The right design **interleaves**: deterministic
parsing, model classification, deterministic retrieval, model decision, model
composition, deterministic validation.

That has a practical consequence worth stating plainly. Each deterministic stage
between model stages is a place where the classical techniques come back:
you can cache the retrieval, unit-test the parser, health-check the validator, and
retry the deterministic parts safely. **Interleaving does not merely cost less
than model-everywhere; it restores testability in the gaps**, and the gaps are
where the operational instruments live.

The last table prices the extremes honestly. Pure code covers
{tab[0][0]:.1%} of requests, which is not a viable product -- six stages at
{0.94:.0%} to {0.12:.0%} multiply badly, and that multiplication is
ch:ag-loop's chain arriving in an architecture diagram. Model-everywhere covers
{tab[6][0]:.1%} and leaves nothing testable.

The design that puts the model at the three stages that earn it covers most of the
gap and keeps the other three stages instrumented -- which is the recommendation,
and it is neither of the two positions the debate is usually conducted between.""")
```

Sweeping the boundary across a six-stage pipeline gives:

```
               model handles   coverage   testable  classical techniques
------------------------------------------------------------------------
                     nothing       2.3%       100%              all work
           the last 2 stages      19.4%        70%             most work
           the last 3 stages      56.1%        40%              few work
                  everything      91.3%         0%             none work
```

Pure code covers **2.3%** of requests — six stages multiplying, which is
{{eq:loop-is-not-a-chain}} in an architecture diagram — and keeps everything
testable. Model-everywhere covers **91.3%** and keeps nothing.

But ranking stages by coverage bought per unit of cost
({{eq:model-belongs-interleaved}}) says something the boundary framing cannot
express:

```
  rank                 stage   per cost  code suffices
------------------------------------------------------
     1     decide what to do      0.296            34%
     2    compose the answer      0.290            12%
     3       classify intent      0.281            71%
     4        gather context      0.078            88%
     5     parse the request      0.046            94%
     6   format and validate      0.019            97%
```

The top three are separated from the bottom three by a factor of roughly four, and
best from worst by about **fifteen**. Now look at where they sit: `classify intent`
is the **second** stage, while `parse the request` and `gather context` — first and
third — are in the bottom half. **The stages where a model earns its place are not
contiguous**, so no single boundary depth captures them.

The right design **interleaves**: deterministic parsing, model classification,
deterministic retrieval, model decision, model composition, deterministic
validation.

```mermaid {#fig:interleaved caption="The stages worth giving to a model are not contiguous, so the deterministic stages survive as gaps between them, and each gap restores the classical techniques."}
flowchart LR
  A["parse<br/>code"] --> B["classify<br/>model"]
  B --> C["gather context<br/>code"]
  C --> D["decide<br/>model"]
  D --> E["compose<br/>model"]
  E --> F["validate<br/>code"]
  A -.- A1["testable, cacheable"]
  C -.- C1["testable, cacheable"]
  F -.- F1["testable, cacheable"]
```

That has a consequence worth stating plainly. Each deterministic stage between model
stages is a place where the classical techniques **come back**: you can cache the
retrieval, unit-test the parser, health-check the validator, and retry the
deterministic parts safely. Interleaving does not merely cost less than
model-everywhere — **it restores testability in the gaps**, and the gaps are where
the operational instruments live.

## 10. Production Considerations

The second instrument is the first thing to build, and it is not optional. In
practice it takes one of three forms: a sampled human review stream, an
automated verifier over a subset of requests, or a proxy signal (user retries,
thumbs-down rates, escalation to a human channel) that correlates with semantic
failure. All three are worse than an availability graph; all three are infinitely
better than nothing.

Instrument the **gaps**, not just the ends. The interleaved design's deterministic
stages are the only places where conventional observability means what it says, so
that is where to put assertions, cache-hit metrics, and health checks. A trace that
records only the model calls throws away the part of the system you can actually
reason about.

Cost per request is a distribution, not a number. Budget and alert on percentiles.
A mean cost that looks fine can hide a tail where a small share of requests consumes
most of the spend, and that tail moves with user behaviour rather than with traffic.

One organisational note, because it decides whether any of this survives contact
with a real team. The availability graph has an owner, a dashboard, a paging
policy, and usually a quarterly review. The semantic error rate typically has
none of those on the day it is first measured, which means it degrades quietly
while the instrument that *does* have an owner keeps reporting green. Give the
second instrument the same institutional weight as the first — a named owner, a
target, and a place in the same review — or it becomes a graph nobody reads.

A related point about rollout. Because the two instruments measure different
quantities, a deploy can improve one and degrade the other, and the usual
canary logic will not catch it: a canary that watches latency and error rate
will happily promote a build that answers faster and wronger. Canaries for
model-backed systems need the semantic signal in the promotion gate, which in
turn means the signal has to be fast enough to gate on. That is a real
constraint on how the second instrument is built, and it is worth designing for
before the first bad deploy rather than after.

Expect the boundary to move. As models get cheaper, $r_i$ rises for every stage and
more of them cross the threshold; as a domain's deterministic tooling improves,
$c_i$ rises and stages cross back. The ranking is worth recomputing quarterly, not
once.

## 11. Common Mistakes

**Reading the availability graph as a health signal.** It is a real measurement of a
question that stopped being the important one. The overspend table is what it is
hiding.

**Drawing a single boundary because the diagram is prettier.** The ranking
interleaves; a contiguous boundary either gives the model stages it does not earn or
withholds ones it does.

**Retrying a semantic failure.** Without a verifier, a retry is a fresh sample that
costs money and may be wrong in a new way ({{eq:retry-needs-a-verifier}}).

**Planning capacity on request count.** With forty-to-one cost heterogeneity,
capacity is a function of the mix.

**Treating the three properties as a single problem.** They break different
techniques and demand different replacements; the survival table's value is in the
per-property columns, not the total.

## 12. Failure Modes

**Silent semantic drift.** Model or prompt changes shift the error rate with no
instrument moving. Detected only by the second instrument, which is why its absence
is a failure mode rather than a gap.

**Cache poisoning by context mismatch.** A cached answer that was correct for one
context is served into another. Grows with cache hit rate, so it worsens exactly as
the cache starts paying off.

**Retry storms under semantic failure.** A failure that retries cannot fix consumes
budget until a cost ceiling stops it — the expensive property amplifying the
sometimes-wrong one.

**Tail-driven capacity exhaustion.** A shift in request mix toward expensive stages
exhausts capacity at unchanged traffic, which every count-based dashboard reports as
normal.

**Boundary creep.** Stages migrate to the model individually, each defensible,
until nothing deterministic remains between them and the testable gaps are gone.

## 13. Alternatives

**Deterministic-only.** Covers 2.3% in the listing's pipeline. Viable only where the
input space is genuinely enumerable, which is rarer than it looks.

**Model-only.** Covers 91.3% and keeps nothing testable. Defensible for prototypes
and for systems whose failure cost is low; indefensible once someone is on call.

**Cascade routing.** Rather than fixing which stages use a model, route each request
to a cheap or expensive model by difficulty. {{cite:chen2023frugalgpt}} reports
matching the best individual model with up to 98% cost reduction, or +4% accuracy at
equal cost — but the cascade only works if something judges when the cheap answer
suffices, which is the verifier problem wearing a cost hat.
{{cite:hu2024routerbench}} made the trade-off measurable rather than assertable with
over 405,000 precomputed inference outcomes. {{ch:sd-routing-caching}} takes this
up; it composes with interleaving rather than replacing it.

**Human-in-the-loop at every model stage.** Restores an instrument at the cost of
throughput. Sensible where per-error cost is high; the autonomy analysis of
{{ch:aise-autonomy}} applies directly.

## 14. Evaluation

Measure **semantic** error rate, not availability — sampled and reviewed if it
cannot be automated. Report it beside availability, never instead of it; they are
different quantities and the pairing is the point.

Track the **overspend ratio** $\varepsilon / B$ explicitly. It converts a semantic
error rate into the unit the on-call rotation already manages, and it is the number
that makes the case for the second instrument to people who own budgets.

For the boundary, measure $c_i$ per stage — the share of inputs deterministic code
handles correctly — against real traffic rather than estimates. Everything in
{{eq:model-belongs-interleaved}} follows from $c_i$, and it is the parameter most
often guessed.

Track **testable share** as a first-class architectural metric. A design that
improves coverage while driving testable share to zero has traded a measurable
property for an unmeasurable one, and nothing in a conventional dashboard will say
so.

## 15. Advanced Concepts

The independence assumption in {{eq:three-properties-break-the-stack}} is a
simplification. The properties interact: expense amplifies the cost of
sometimes-wrong (a wrong answer wastes an expensive call *and* requires another),
and nondeterminism amplifies both. A more faithful model would use a
super-multiplicative form; the qualitative ranking survives, but the survival
figures are optimistic.

The coverage model treats $\mu$ as constant across stages, which understates the
case for interleaving. In reality a model stage bounded by deterministic stages on
both sides has a **narrower input distribution** than one embedded in a chain of
model stages, so its effective coverage is higher. Interleaving improves the model
stages as well as preserving the deterministic ones.

There is also a question the cost model in {{eq:model-belongs-interleaved}}
sidesteps. It prices a stage by the cost of the call, but the cost that matters
operationally is the cost of the call *plus the expected cost of being wrong at
that stage*, and those are not proportional. A wrong intent classification is
cheap to detect and cheap to recover from, because the next deterministic stage
sees an intent it cannot serve. A wrong composition is expensive to detect and
lands directly in front of the user. Weighting each stage by the recoverability
of its failures — rather than by the price of its call — changes the ranking,
and generally moves the model toward stages whose errors a later deterministic
stage can catch. This is the same asymmetry {{ch:ag-termination}} used to decide
when to stop, arriving as a placement decision instead of a stopping one.

{{eq:agent-errors-correlate}} bears on the composition: if consecutive model stages
fail together rather than independently, the product form in
{{eq:boundary-decides-testability}} is optimistic for model-heavy designs, and a
deterministic stage between two model stages does more than restore techniques — it
**breaks the correlation** by re-grounding the input.

## 16. Connection to Previous Chapters

{{eq:loop-is-not-a-chain}} from {{ch:ag-loop}} appears here as the coverage product:
six stages at 94% to 12% compose to 2.3%. The result is the same; the setting is a
request pipeline rather than an agent trace, which is what makes it an architecture
constraint.

{{eq:retry-needs-a-verifier}} from {{ch:ag-recovery}} is re-derived from the
survival table at 9%. That it arrives independently, from a model of classical
techniques rather than of agents, is the stronger form of the claim.

{{eq:context-is-a-budget}} from {{ch:mcp-schemas}} becomes the cost heterogeneity
that breaks capacity planning and load balancing.

{{cite:cemri2025mast}}'s failure taxonomy supplies the empirical shape of the
failures that the second instrument has to catch.

## 17. Exercises

1. Recompute the survival table with a super-multiplicative interaction between
   *expensive* and *sometimes wrong*. Does the ranking change, or only the values?

2. Take a service you work on. Estimate $c_i$ per stage from real traffic and rank
   by {{eq:model-belongs-interleaved}}. Is the resulting design contiguous?

3. Derive the semantic error rate at which the overspend ratio reaches 100 for a
   99.95% availability target. Compare to the 99.9% case.

4. Modify the second listing so a model stage bounded by deterministic stages has
   higher $\mu$ than one following another model stage. How much does that shift the
   optimum toward interleaving?

5. Design a second instrument for a system you know. State its sampling rate, its
   cost, and the smallest semantic error rate change it could detect within a week.

## 18. Interview Questions

1. Your availability dashboard reads 99.95% and users are complaining the answers
   are wrong. What is the dashboard measuring, and what would you build?

2. Why is retrying a model call different from retrying a database call?

3. A colleague proposes "let the model handle everything from intent classification
   onward." What is wrong with the *shape* of that proposal, independent of where
   the boundary sits?

4. How does cost heterogeneity change load balancing, and what does that make it?

5. Which survives better under the three properties — caching or health checks — and
   why is the answer counterintuitive?

## 19. Research Questions

1. Can semantic error rate be estimated online without a verifier, from proxy
   signals alone, with enough precision to drive an error budget?

2. Is there a principled way to choose $\mu$ per stage from the input distribution's
   entropy, rather than assuming it constant?

3. How much of the correlation between consecutive model failures does a
   deterministic stage between them actually break? The claim in
   {{sec:15-advanced-concepts}} is plausible and unmeasured.

4. Does the interleaving result hold when routing ({{cite:chen2023frugalgpt}}) is
   composed with it, or do cascades change the per-stage ranking?

## 20. Chapter Summary

A language model is nondeterministic, expensive, and occasionally wrong, and
conventional system design assumes none of those. Modelling the damage as
independent multiplicative degradations
({{eq:three-properties-break-the-stack}}) leaves nothing above **36%** and health
checks at **5%**.

Being wrong does the most damage — not by count, since expense halves as many
techniques, but by depth: **49%** mean survival against expense's **70%**. Expense
degrades broadly; being wrong destroys, because it is the only one of the three with
no instrument ({{eq:semantic-failure-has-no-instrument}}). A 6% semantic error rate
overspends a 99.9% availability budget by **61 times** while the dashboard reads
99.900%.

The design response is a boundary between deterministic and model-handled stages
({{eq:boundary-decides-testability}}), and the boundary sweep gives the familiar
extremes: **2.3%** coverage fully testable, **91.3%** coverage not testable at all.

But ranking stages by coverage per unit cost ({{eq:model-belongs-interleaved}})
shows the stages worth giving to a model are **not contiguous** — `classify intent`
ranks third while the stages surrounding it rank fourth and fifth. So the right
design interleaves, and interleaving restores the classical techniques in the
deterministic gaps rather than merely costing less.

Two things to carry forward: **build the second instrument first**, and **instrument
the gaps**.

## 21. Further Reading

- {{cite:chen2023frugalgpt}} — cascade routing, and the cost case for a judge.
- {{cite:hu2024routerbench}} — 405,000+ precomputed outcomes, making the
  cost-quality frontier measurable.
- {{cite:cemri2025mast}} — the empirical failure taxonomy behind the second
  instrument.
- {{cite:kwon2023pagedattention}} — memory management under variable-length
  generation; cost heterogeneity at the serving layer.
