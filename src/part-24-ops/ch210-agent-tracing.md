---
id: ops-agent-tracing
number: 210
part: XXIV
tier: full
status: draft
requires: [semantic-failure-has-no-instrument, attribution-needs-payload-not-timing,
           uniform-sampling-misses-rare-failures, reproducibility-is-a-product-over-artefacts]
provides: [triage-capacity-is-the-binding-constraint, structure-improves-both-channels,
           cause-distance-drives-triage-cost, record-beats-replay]
citations: [deshpande2025trail, cemri2025mast, sculley2015, breck2017]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute triage capacity against failing-trace
arrival rate and show that neither human nor automated triage scales to it; explain why
trace *structure* is the only lever that improves both channels simultaneously; measure the
distance in steps between where an agent failure becomes visible and where it was caused;
explain why per-step correctness monitoring catches almost none of these failures; rank
recorded trace fields by minutes of triage saved per unit of instrumentation effort; and
decide when recording state is preferable to replaying the run.

## 2. Why This Matters

An agent service at 42,000 requests a day with a 9% failure rate produces **3,780 failing
traces a day**. One engineer, at 26 minutes a trace, localises **14 of them — 0.4%**.
Twenty-five engineers reach **9.2%**. Clearing a single day's failures takes **273
engineer-days** ({{eq:triage-capacity-is-the-binding-constraint}}).

The automated channel does not rescue this. {{cite:deshpande2025trail}} built a benchmark
for precisely this task — localising the issue inside an annotated agent trace — and the
best model tested reached **11%**. Running it over every failing trace costs $1,285 a day
and correctly localises **416**, leaving **3,364** wrong or unlocalised. Twenty-five
engineers *plus* full automated coverage still leaves **79.8%** of failing traces untriaged.

So the lever is not capacity in either channel. It is minutes per trace and accuracy per
trace, and both are set by what the trace contains
({{eq:structure-improves-both-channels}}). Adding explicit step boundaries, tool inputs and
outputs, intermediate state, and causal links takes automated localisation from **11% to
44%** — **4.0×** — and human triage from **26.4 minutes to 4.8**.

The second half asks what the structure must contain, and the answer follows from a
property agent failures have and single-turn failures do not: the failure becomes visible
at one step and the cause sits **2.7 steps earlier on average**
({{eq:cause-distance-drives-triage-cost}}). Per-step correctness monitoring catches **27%**
of these, because in **86%** of cases the causing step *succeeded*.

## 3. Prerequisites

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is the ancestor of
this chapter's central negative result. Per-step monitoring of an agent is a health check
by another name, and it fails for the same reason: the step returned, in the expected
shape, with the wrong content.

{{eq:attribution-needs-payload-not-timing}} from {{ch:ops-observability}} established that
timing telemetry cannot attribute semantic failures and payload can. This chapter applies
that to a sequence rather than a single call, where the payload that matters is
*intermediate* rather than terminal.

{{eq:uniform-sampling-misses-rare-failures}} from the same chapter decides which traces get
recorded at full fidelity, and {{sec:10-production-considerations}} connects the two.

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is what
replay requires, and its scarcity is why recording wins.

{{cite:cemri2025mast}}'s taxonomy of multi-agent failure modes is the empirical backdrop:
the failures are distributed across the run, not concentrated at its end.

## 4. Intuitive Explanation

A single-turn request has one place to look. The prompt went in, the answer came out, and
if the answer is wrong the wrongness is in the model, the prompt, or the retrieved context.
Three candidates, all visible, all in one record.

An agent request is a sequence. It plans, calls a tool, reads the result, decides what to do
next, calls another tool, revises the plan, and eventually answers. Seven steps on average
in the numbers used here. When the answer is wrong, the wrongness entered somewhere in that
sequence and then propagated forward through every subsequent decision.

That changes the question from *what failed* to *where did this go wrong*, and the second
question has a search space where the first had a list.

Now consider the volume. A service handling 42,000 agent requests a day with a 9% failure
rate produces nearly four thousand failing traces daily, each seven steps long. Reading one
carefully takes something like twenty-six minutes — you are reconstructing a chain of
reasoning from its residue. An engineer with six productive hours can read fourteen.

Fourteen against three thousand seven hundred and eighty.

This is not a staffing shortfall to be closed. Clearing one day's failures exhaustively
takes 273 engineer-days, which at market rates is fifty-three million dollars a year for a
single day's backlog to stay flat. Nobody is going to fund that, which means the honest
description of human triage here is not *partial coverage* but **sampling** — and it samples
at whatever rate the headcount happens to allow, which is a strange way to choose what you
look at.

The obvious alternative is to have a model read the traces. {{cite:deshpande2025trail}}
measured how well that works on a benchmark built for the task, and the best model reached
eleven percent. Eleven percent is not a partial solution either. It means nearly nine in
ten automated diagnoses are wrong, and a wrong diagnosis is worse than no diagnosis, because
someone acts on it.

So both channels are far short and neither scales. What is left?

Minutes per trace, and accuracy per trace. Both are properties of the trace, not of the
reader. And the striking thing — the reason this chapter exists — is that the *same*
structural properties improve both. A trace with explicit step boundaries, recorded tool
arguments and results, saved intermediate state, and links showing which step used which
earlier output is faster for a human to read *and* more tractable for a model to localise
in. Human triage falls from 26.4 minutes to 4.8; model accuracy rises from 11% to 44%.

Trace structure is the only lever in this chapter that improves both channels at once, and
it is the one nobody budgets for, because a trace format does not look like a reliability
investment. It looks like logging.

The second half of the chapter asks what the structure has to contain, and the answer comes
from measuring the thing that makes agent failures hard.

Take a failing agent run and ask: at which step did the failure become *visible*, and at
which step was it *caused*? Those are usually not the same step. The answer was wrong
because a decision was wrong because a tool returned something odd because the arguments
were built from a retrieval that missed. The visible failure is at the end; the cause is
several steps back. On the distribution used here, 2.7 steps back on average.

That distance is the whole cost. A cause at the visible step takes 3.1 minutes to find. A
cause six steps back takes 67.3, because every intervening step has to be reconstructed —
you have to work out what each one must have been holding in order to produce what the next
one did.

And here is why the obvious instrumentation does not help. The natural response to "the
cause is at an earlier step" is "then check every step." Validate each tool call, assert on
each output, alarm when a step fails.

That catches 27% of these failures, and the reason is in the structure of the remaining 73%:
**the causing step succeeded.** A tool that returns well-formed, plausible, wrong data has
not failed. A retrieval that returns documents — just not the right documents — has not
failed. A plan that is wrong from the start executes perfectly, step by correct step. A
state write that corrupts the working set succeeded as a write.

Only one category — the tool that genuinely errored — is caught reliably, and it is 14% of
failures.

That is {{ch:sd-architecture}}'s third property arriving in agent form, and it settles the
design question. If per-step checks cannot find the cause, the trace has to preserve enough
that someone can find it afterwards. Which means recording what each step *held*, not just
whether it *worked*.

The good news in the numbers is that the two cheapest fields do most of the work — and both
are things the agent framework already has in memory at the moment it discards them. Step
boundaries and tool arguments are not derived or inferred. They are variables that existed
and were not written down.

## 5. Formal Explanation

Let $R$ be requests per day and $f$ the failure rate, so the arrival rate of failing traces
is $\Lambda = R f$. Let $m$ be minutes to localise one trace, $h$ productive triage hours
per engineer per day, and $N$ engineers. Human triage capacity is $N h \cdot 60 / m$ traces
per day, and the covered share is that capacity divided by $\Lambda$.

The automated channel has a different shape. It has effectively unlimited throughput at a
per-trace cost $c$, but an accuracy $A < 1$ — so covering a share $\sigma$ of traces
localises $\Lambda \sigma A$ correctly and leaves $\Lambda \sigma (1 - A)$ with a *wrong*
diagnosis, which is not the same as leaving them alone.

Combining the channels, humans take what automation could not localise, up to capacity, so
the untriaged residue is

$$\Lambda - \Lambda \sigma A - \min\left(\frac{Nh \cdot 60}{m},\; \Lambda(1 - \sigma A)\right).$$

Both terms subtracted are small: the first because $A$ is small, the second because $Nh
\cdot 60 / m$ is small relative to $\Lambda$. Neither $N$ nor $\sigma$ can be raised enough
to matter, which is the result.

What *can* move is $m$ and $A$, and both are functions of the trace's structure $S$. Write
$m(S)$ and $A(S)$. The human channel improves as $m$ falls; the automated channel improves
as $A$ rises; and the empirical claim of this chapter is that the same $S$ does both.

For the second half: let $d$ be the number of steps between the visible failure and its
cause, distributed with probabilities $p_d$. Localisation cost grows superlinearly in $d$,
because each intervening step must be reconstructed and reconstruction gets harder as the
chain lengthens — modelled here as $T(d) = t_0 (d+1)(1 + \beta d)$.

Recording a field $k$ removes a share $\gamma_k$ of the reconstruction work, because the
state no longer has to be inferred. Applying a set of fields multiplies:
$T' = T \prod_k (1 - \gamma_k)$, and each field costs effort $e_k$, so the build order is by
$\gamma_k / e_k$.

Finally, replay. If the run is reproducible — every artefact pinned, in
{{ch:ops-versioning}}'s sense — you can re-execute with full logging and observe the
intermediate state directly, which is cheaper than reading any trace. If it is not, replay
is unavailable at any price, and recording is not the cheaper option but the only one.

## 6. Mathematical Foundation

Triage capacity against arrival rate, the constraint that governs the chapter:

$$C = \frac{N h \cdot 60}{m} \quad \text{against} \quad \Lambda = R f, \qquad \text{covered} = \frac{C}{\Lambda}$$ (eq:triage-capacity-is-the-binding-constraint)

With $R = 42{,}000$, $f = 0.09$, $h = 6$, $m = 26$: one engineer covers $0.4\%$ and
twenty-five cover $9.2\%$. The automated channel adds $\sigma A$ with $A = 0.11$
{{cite:deshpande2025trail}}, which is bounded above by $11\%$ however much is spent.

Both channels are functions of the same structural variable, in opposite directions:

$$\frac{\partial m(S)}{\partial S} < 0 \quad \text{and} \quad \frac{\partial A(S)}{\partial S} > 0 \;\Longrightarrow\; \frac{\partial}{\partial S}\left[\frac{Nh \cdot 60}{m(S)} + \Lambda \sigma A(S)\right] > 0$$ (eq:structure-improves-both-channels)

This is the only term in the model with that property. Adding engineers raises the first
term and leaves the second untouched; buying more automated coverage raises the second and
leaves the first untouched.

Localisation cost as a function of cause distance:

$$T(d) = t_0 (d + 1)(1 + \beta d), \qquad \bar{T} = \sum_d p_d \, T(d), \qquad \bar{T}' = \bar{T} \prod_k (1 - \gamma_k)$$ (eq:cause-distance-drives-triage-cost)

With $t_0 = 3.1$ and $\beta = 0.35$: $T(0) = 3.1$ minutes, $T(6) = 67.3$, and $\bar{T} =
26.4$. The per-step check catches $\sum_d p_d \kappa_d = 27\%$, where $\kappa_d$ is the
probability the causing step visibly failed.

And the choice between recording and replaying, which is not a cost comparison in most
systems:

$$T_{\text{best}} = \begin{cases} \min(T_{\text{replay}}, \bar{T}') & \text{if } \rho = 1 \\[4pt] \bar{T}' & \text{if } \rho < 1 \end{cases}$$ (eq:record-beats-replay)

where $\rho$ is {{ch:ops-versioning}}'s artefact coverage. Replay localises in 11 minutes
against a structured trace's 4.8 — but only at $\rho = 1$, which that chapter found is
rarely reached.

## 7. Internal Mechanics

Why does an agent framework end up with an unreadable trace by default? Not by decision.
Three mechanisms, each locally reasonable.

**The framework logs at the boundary it owns.** An agent library emits a span per LLM call
and a span per tool call, because those are the operations it performs. It does not emit the
agent's *state* between them, because state is a local variable in a loop rather than an
operation. So the trace records every interaction with an external system and none of the
reasoning that connected them — which is exactly inverted from where the causes sit.

**Tool results are large and get truncated.** A retrieval returns eight documents; the
logger writes the first 512 characters. The truncation is sensible for volume and fatal for
triage, because "retrieval missed the fact" — 21% of failures here, sitting three steps back
— is diagnosable only by seeing what retrieval actually returned. This is
{{ch:ops-observability}}'s payload-versus-timing result recurring at a finer grain: the
field is present but emptied.

**Causal links are never materialised because they were never explicit.** When step five
uses a value that step two produced, the connection exists in the agent's context window and
nowhere else. Reconstructing it afterwards means reading step five's prompt and matching
substrings against earlier outputs, which is most of what those 26 minutes are spent on. The
framework knew the link at the moment it built the prompt; it simply had no field to put it
in.

The cost curve's superlinearity comes from the same place. Finding a cause one step back
means reading one extra step. Finding one six steps back means reading six extra steps *and*
holding a hypothesis about each one's state while you evaluate the next, because none of
them recorded what they held. The $(1 + \beta d)$ term is the cost of that mental
bookkeeping, and it is precisely the term that recorded state removes.

There is a second-order effect worth naming. Because the trace is unreadable, teams triage
by re-running the agent on the same input and watching — which works when the run is
reproducible and produces a *different* run when it is not. A non-reproducible re-run that
happens to succeed is read as "not reproducible, closing," and a real failure mode is
retired by a coin flip.

## 8. Implementation

The first listing measures both triage channels against the arrival rate.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ef1}
"""Agent traces arrive faster than anything can triage them, and nothing can triage them.

A single-turn request produces one span worth reading. An agent request produces a
sequence of steps, each with a tool call, a result, and a model decision -- and the
failure is somewhere in the sequence rather than at a point.

cite:deshpande2025trail built a benchmark for exactly this task, localising the issue
inside an annotated agent trace, and the best model tested reached **11%**. Humans do
better and not by enough to matter at volume
(eq:triage-capacity-is-the-binding-constraint).

This listing measures the arrival rate against both triage channels and finds neither
scales, which makes trace STRUCTURE the only lever left.
"""
REQUESTS_PER_DAY = 42000.0
FAIL_RATE = 0.09              # agent requests that end badly
STEPS_MEAN = 7.4
HUMAN_MINUTES_PER_TRACE = 26.0
HUMAN_HOURS_PER_DAY = 6.0     # productive triage hours per engineer
MODEL_ACCURACY = 0.11         # cite:deshpande2025trail, best model on TRAIL
MODEL_COST_PER_TRACE = 0.34

failing = REQUESTS_PER_DAY * FAIL_RATE
print("An agent service at %.0f requests a day, %.0f%% ending badly."
      % (REQUESTS_PER_DAY, FAIL_RATE * 100))
print("That is %.0f failing traces a day, averaging %.1f steps each."
      % (failing, STEPS_MEAN))
print()
print("Human triage capacity, at %.0f minutes a trace." % HUMAN_MINUTES_PER_TRACE)
print()
print(f"{'engineers':>11}{'traces/day':>13}{'share of failures':>20}"
      f"{'days to clear one day':>24}")
print("-" * 70)
cap = {}
for n in (1, 2, 5, 10, 25, 100):
    per_day = n * HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE
    cap[n] = (per_day, per_day / failing)
    print(f"{n:>11}{per_day:>13.0f}{per_day / failing:>20.1%}"
          f"{failing / per_day:>24.1f}")

print()
print()
print("Automated triage: cheap per trace, and it is right %.0f%% of the time."
      % (MODEL_ACCURACY * 100))
print()
print(f"{'coverage':>11}{'traces/day':>13}{'cost/day':>11}"
      f"{'correctly localised':>22}{'wrong or unlocalised':>23}")
print("-" * 82)
auto = {}
for cov in (0.05, 0.25, 0.50, 1.00):
    n = failing * cov
    correct = n * MODEL_ACCURACY
    auto[cov] = (n, n * MODEL_COST_PER_TRACE, correct, n - correct)
    print(f"{cov:>11.0%}{n:>13.0f}{n * MODEL_COST_PER_TRACE:>11.0f}"
          f"{correct:>22.0f}{n - correct:>23.0f}")

print()
print()
print("Both channels together, and what is left untriaged.")
print()
print(f"{'engineers':>11}{'auto coverage':>15}{'localised/day':>16}"
      f"{'untriaged/day':>16}{'share untriaged':>18}")
print("-" * 78)
both = {}
for n in (2, 10, 25):
    for cov in (0.0, 0.5, 1.0):
        auto_correct = failing * cov * MODEL_ACCURACY
        # Humans triage what automation could not localise, up to capacity.
        remaining = failing - auto_correct
        human = min(cap[n][0], remaining)
        left = remaining - human
        both[(n, cov)] = (auto_correct + human, left)
        print(f"{n:>11}{cov:>15.0%}{auto_correct + human:>16.0f}"
              f"{left:>16.0f}{left / failing:>18.1%}")

print()
print()
print("The arithmetic that matters: how much triage effort a single percentage")
print("point of failure rate costs.")
print()
print(f"{'failure rate':>14}{'traces/day':>13}{'engineers to clear':>21}"
      f"{'annual cost':>14}")
print("-" * 64)
ENG_COST = 195000.0
for fr in (0.01, 0.03, 0.05, 0.09, 0.15):
    f = REQUESTS_PER_DAY * fr
    n_eng = f / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE)
    print(f"{fr:>14.0%}{f:>13.0f}{n_eng:>21.0f}{n_eng * ENG_COST:>14,.0f}")

print()
print("Clearing every failing trace is not a staffing plan. It is a different")
print("company.")

print()
print()
print("So the lever is not capacity. It is minutes per trace -- which structure")
print("controls.")
print()
print(f"{'minutes/trace':>15}{'traces/engineer/day':>22}"
      f"{'engineers for 10%':>20}{'engineers for 50%':>20}")
print("-" * 78)
mins = {}
for m in (26.0, 14.0, 8.0, 4.0, 1.5):
    per_eng = HUMAN_HOURS_PER_DAY * 60.0 / m
    mins[m] = per_eng
    print(f"{m:>15.1f}{per_eng:>22.0f}"
          f"{failing * 0.10 / per_eng:>20.1f}{failing * 0.50 / per_eng:>20.1f}")

print()
print()
print("And the same lever applied to the automated channel: a model localises")
print("better on a trace that is structured for it.")
print()
print(f"{'trace quality':>32}{'model accuracy':>17}{'localised/day at 100%':>24}"
      f"{'vs baseline':>13}")
print("-" * 88)
QUAL = [
    ("raw log lines",           0.11),
    ("+ explicit step boundaries", 0.19),
    ("+ tool inputs and outputs", 0.28),
    ("+ recorded intermediate state", 0.37),
    ("+ causal links between steps", 0.44),
]
qual = {}
for label, acc in QUAL:
    n = failing * acc
    qual[label] = (acc, n)
    print(f"{label:>32}{acc:>17.0%}{n:>24.0f}"
          f"{acc / MODEL_ACCURACY:>12.1f}x")

print(f"""
The capacity table is the first thing to look at and it settles the staffing question
immediately. At {HUMAN_MINUTES_PER_TRACE:.0f} minutes a trace, one engineer triages
{cap[1][0]:.0f} traces a day against {failing:.0f} failing ones -- **{cap[1][1]:.1%} of
them**. Twenty-five engineers reach {cap[25][1]:.1%}.

Clearing a single day's failures would take {failing / cap[1][0]:.0f} engineer-days
(eq:triage-capacity-is-the-binding-constraint). **Human triage is not a partial solution
here. It is a sampling strategy**, and it samples at whatever rate the headcount
happens to allow.

The automated table is the obvious alternative and cite:deshpande2025trail measured its
ceiling. At {MODEL_ACCURACY:.0%} accuracy, running automated triage over every failing
trace costs {auto[1.0][1]:.0f} a day and correctly localises {auto[1.0][2]:.0f} of
{failing:.0f} -- leaving {auto[1.0][3]:.0f} either wrong or unlocalised.

That is not a system anyone can act on. **An eleven percent localisation rate means
nearly nine in ten automated diagnoses are wrong**, and a wrong diagnosis is worse than
no diagnosis because someone acts on it.

The combined table shows the two channels do not rescue each other. With
{25} engineers and full automated coverage, {both[(25, 1.0)][1] / failing:.0%} of failing
traces are still untriaged. The automation removes {MODEL_ACCURACY:.0%} and the humans
remove what they can reach, and the sum is far short.

The failure-rate table is where this becomes a design constraint rather than an
operations problem. Every percentage point of failure rate costs
{REQUESTS_PER_DAY * 0.01 / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE):.0f}
engineers to triage exhaustively. At {0.09:.0%}, exhaustive triage is
{failing / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE):.0f} engineers and
{failing / (HUMAN_HOURS_PER_DAY * 60.0 / HUMAN_MINUTES_PER_TRACE) * ENG_COST / 1e6:.1f}
million a year.

**Nobody is going to staff that**, which means the question is never "how do we triage
everything" but "what do we do with the tiny share we can look at" -- and
ch:ops-observability's sampling result then decides which share that is.

The minutes-per-trace table is the lever that is actually available. Cutting triage from
{26.0:.0f} minutes to {8.0:.0f} takes one engineer from {mins[26.0]:.0f} traces a day to
{mins[8.0]:.0f}, so the engineers needed to cover a tenth of failures falls from
{failing * 0.10 / mins[26.0]:.1f} to {failing * 0.10 / mins[8.0]:.1f}.

**That is a {mins[8.0] / mins[26.0]:.1f}x capacity gain from making traces easier to
read**, and unlike headcount it compounds with the automated channel.

Which the last table shows. The same structural properties that make a trace fast for a
human to read make it tractable for a model: explicit step boundaries, recorded tool
inputs and outputs, intermediate state, causal links. Adding them takes automated
localisation from {MODEL_ACCURACY:.0%} to
{qual['+ causal links between steps'][0]:.0%} --
{qual['+ causal links between steps'][0] / MODEL_ACCURACY:.1f} times more traces
correctly localised, for the same model.

**Trace structure is the only lever that improves both channels at once**, and it is
the one nobody budgets for, because a trace format does not look like a reliability
investment. ch:ops-agent-tracing's second listing takes up what the structure has to
contain.""")
```

## 9. Practical Example

Human triage capacity against 3,780 failing traces a day:

```
  engineers   traces/day   share of failures   days to clear one day
----------------------------------------------------------------------
          1           14                0.4%                   273.0
          2           28                0.7%                   136.5
          5           69                1.8%                    54.6
         10          138                3.7%                    27.3
         25          346                9.2%                    10.9
        100         1385               36.6%                     2.7
```

One engineer covers **0.4%**; twenty-five cover **9.2%**
({{eq:triage-capacity-is-the-binding-constraint}}). Clearing one day's failures takes
**273 engineer-days**, so human triage here is not partial coverage — it is a sampling
strategy whose rate is set by headcount.

```
   coverage   traces/day   cost/day   correctly localised   wrong or unlocalised
----------------------------------------------------------------------------------
         5%          189         64                    21                    168
        25%          945        321                   104                    841
        50%         1890        643                   208                   1682
       100%         3780       1285                   416                   3364
```

At {{cite:deshpande2025trail}}'s **11%**, automated triage over every failing trace costs
$1,285 a day and correctly localises **416 of 3,780**. The other **3,364** are wrong or
unlocalised, and a wrong diagnosis is acted on.

```
  engineers  auto coverage   localised/day   untriaged/day   share untriaged
------------------------------------------------------------------------------
          2             0%              28            3752             99.3%
         10           100%             554            3226             85.3%
         25             0%             346            3434             90.8%
         25           100%             762            3018             79.8%
```

Twenty-five engineers **and** full automated coverage still leave **79.8%** untriaged. The
channels do not rescue each other.

```
  failure rate   traces/day   engineers to clear   annual cost
----------------------------------------------------------------
            1%          420                   30     5,915,000
            3%         1260                   91    17,745,000
            9%         3780                  273    53,235,000
           15%         6300                  455    88,725,000
```

Every point of failure rate costs **30 engineers** to triage exhaustively. At 9% that is
**273 engineers and $53.2M a year** — which nobody will staff, so the question is never
"how do we triage everything."

```
  minutes/trace   traces/engineer/day   engineers for 10%   engineers for 50%
------------------------------------------------------------------------------
           26.0                    14                27.3               136.5
           14.0                    26                14.7                73.5
            8.0                    45                 8.4                42.0
            4.0                    90                 4.2                21.0
            1.5                   240                 1.6                 7.9
```

Cutting triage from 26 to 8 minutes takes one engineer from 14 traces a day to 45 — a
**3.2× capacity gain from making traces easier to read**, and unlike headcount it compounds
with the automated channel.

```
                   trace quality   model accuracy   localised/day at 100%  vs baseline
----------------------------------------------------------------------------------------
                   raw log lines              11%                     416         1.0x
      + explicit step boundaries              19%                     718         1.7x
       + tool inputs and outputs              28%                    1058         2.5x
   + recorded intermediate state              37%                    1399         3.4x
    + causal links between steps              44%                    1663         4.0x
```

The same properties lift automated localisation from **11% to 44%** — **4.0×** for the same
model ({{eq:structure-improves-both-channels}}).

The second listing measures what the structure has to contain.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ef2}
"""In an agent trace, the failure is a step and the cause is an earlier step.

A single-turn failure has one place to look. An agent failure has a chain: the answer was
wrong because a decision was wrong because a tool returned something unexpected because
the arguments were built from a retrieval that missed.

So localisation is not "which step failed" -- it is "how far back from the visible
failure does the cause sit", and the search cost grows with that distance
(eq:cause-distance-drives-triage-cost).

This listing measures the distance distribution, finds what the trace must record to
close it, and shows why per-step correctness monitoring does not find these at all.
"""
import math

# (cause type, share of failures, steps back from the visible failure,
#  P(a per-step check would have caught it at the step where it happened))
CAUSES = [
    ("tool returned an error",        0.14, 0, 0.94),
    ("tool returned wrong data",      0.19, 2, 0.11),
    ("arguments built wrongly",       0.16, 1, 0.42),
    ("retrieval missed the fact",     0.21, 3, 0.08),
    ("plan was wrong from the start", 0.17, 6, 0.05),
    ("state corrupted mid-run",       0.13, 4, 0.22),
]
STEPS_MEAN = 7.4
BASE_MINUTES_PER_STEP = 3.1

print("Where the cause of an agent failure actually sits, relative to the step")
print("where the failure became visible.")
print()
print(f"{'cause':>32}{'share':>9}{'steps back':>13}"
      f"{'per-step check catches':>25}")
print("-" * 80)
tab = {}
for name, share, back, catch in CAUSES:
    tab[name] = (share, back, catch)
    print(f"{name:>32}{share:>9.0%}{back:>13}{catch:>25.0%}")

mean_back = sum(s * b for n, s, b, c in CAUSES)
caught = sum(s * c for n, s, b, c in CAUSES)
print()
print(f"mean distance from visible failure to cause: {mean_back:.1f} steps")
print(f"share a per-step correctness check would catch: {caught:.0%}")

print()
print()
print("Why per-step checks miss most of it: the steps that CAUSE failures mostly")
print("succeed at the time.")
print()
print(f"{'cause':>32}{'step succeeded?':>18}  {'why the check passes':<40}")
print("-" * 92)
WHY = {
    "tool returned an error":        ("no",  "it did not"),
    "tool returned wrong data":      ("yes", "well-formed, plausible, wrong"),
    "arguments built wrongly":       ("yes", "valid arguments, wrong ones"),
    "retrieval missed the fact":     ("yes", "returned documents, not the right ones"),
    "plan was wrong from the start": ("yes", "each step executed correctly"),
    "state corrupted mid-run":       ("yes", "the write succeeded"),
}
for name, share, back, catch in CAUSES:
    ok, why = WHY[name]
    print(f"{name:>32}{ok:>18}  {why:<40}")

print()
print()
print("Triage cost by distance. Without recorded intermediate state, each step")
print("back must be reconstructed by re-reading and inferring.")
print()
print(f"{'steps back':>12}{'share of failures':>20}{'minutes to localise':>22}"
      f"{'weighted':>11}")
print("-" * 68)
raw_total = 0.0
for back in sorted(set(b for n, s, b, c in CAUSES)):
    share = sum(s for n, s, b, c in CAUSES if b == back)
    mins = BASE_MINUTES_PER_STEP * (back + 1) * (1.0 + 0.35 * back)
    raw_total += share * mins
    print(f"{back:>12}{share:>20.0%}{mins:>22.1f}{share * mins:>11.1f}")
print("-" * 68)
print(f"{'MEAN':>12}{1.0:>20.0%}{'':>22}{raw_total:>11.1f}")

print()
print()
print("What each recorded field removes from that cost.")
print()
FIELDS = [
    ("explicit step boundaries",   0.18, 0.5),
    ("tool arguments as sent",     0.22, 1.0),
    ("tool results as received",   0.26, 1.5),
    ("agent state after each step", 0.31, 4.0),
    ("the plan, and revisions to it", 0.15, 2.0),
    ("causal links (this used that)", 0.34, 7.0),
]
print(f"{'field':>32}{'cuts triage by':>17}{'effort':>9}"
      f"{'minutes after':>16}{'per effort':>13}")
print("-" * 88)
cur = raw_total
per = {}
for name, cut, eff in FIELDS:
    saved = cur * cut
    per[name] = (cut, eff, saved, saved / eff)
    print(f"{name:>32}{cut:>17.0%}{eff:>9.1f}"
          f"{cur * (1 - cut):>16.1f}{saved / eff:>13.2f}")

print()
print()
print("Building them in payback order.")
print()
order = sorted(FIELDS, key=lambda f: -((raw_total * f[1]) / f[2]))
print(f"{'after adding':>32}{'minutes to localise':>22}{'effort so far':>16}"
      f"{'vs raw':>10}")
print("-" * 82)
cur = raw_total
eff = 0.0
path = []
for name, cut, e in order:
    cur *= (1 - cut)
    eff += e
    path.append((name, cur, eff))
    print(f"{name:>32}{cur:>22.1f}{eff:>16.1f}{cur / raw_total:>9.2f}x")

print()
print()
print("What that does to the triage capacity from the previous listing.")
print()
FAILING = 42000.0 * 0.09
HUMAN_MIN_DAY = 6.0 * 60.0
print(f"{'trace structure':>32}{'minutes/trace':>16}"
      f"{'traces/engineer/day':>22}{'engineers for 25%':>20}")
print("-" * 92)
for label, mins in (("raw", raw_total),
                    ("+ top two fields", path[1][1]),
                    ("+ top four fields", path[3][1]),
                    ("everything", path[-1][1])):
    per_eng = HUMAN_MIN_DAY / mins
    print(f"{label:>32}{mins:>16.1f}{per_eng:>22.0f}"
          f"{FAILING * 0.25 / per_eng:>20.1f}")

print()
print()
print("And the alternative to recording: re-run the agent and watch. This is the")
print("only way to recover state that was never written down.")
print()
print(f"{'approach':>32}{'minutes':>10}  {'needs':<36}")
print("-" * 82)
REPLAY = [
    ("read the raw trace",             raw_total, "nothing"),
    ("read a structured trace",        path[-1][1], "instrumentation"),
    ("re-run with full logging",       11.0, "reproducibility, ch:ops-versioning"),
    ("re-run and step through",        34.0, "reproducibility and an engineer"),
]
for label, mins, needs in REPLAY:
    print(f"{label:>32}{mins:>10.1f}  {needs:<36}")

print(f"""
The distance table is the structural difference between an agent failure and any other
kind. The failure becomes visible at one step and the cause sits **{mean_back:.1f} steps
earlier on average** (eq:cause-distance-drives-triage-cost), with the largest single
category -- `{max(CAUSES, key=lambda c: c[1])[0]}` at
{max(CAUSES, key=lambda c: c[1])[1]:.0%} -- sitting
{max(CAUSES, key=lambda c: c[1])[2]} steps back.

The second table is why the obvious instrumentation does not help. Per-step correctness
monitoring -- check each tool call, validate each output -- catches **{caught:.0%}** of
these, and the reason is in the last column: **the causing step succeeded.** A tool that
returns well-formed wrong data has not failed. A retrieval that returns documents has
not failed. A plan whose every step executes correctly has not failed.

Only `{CAUSES[0][0]}` at {CAUSES[0][1]:.0%} is caught reliably, because it is the one
category where something actually errored.

That is ch:sd-architecture's third property arriving in agent form: **the step
succeeded and was wrong**, and per-step checks are health checks by another name.

The cost table converts distance into minutes. A failure whose cause is at the visible
step takes {BASE_MINUTES_PER_STEP * 1 * 1.0:.1f} minutes; one six steps back takes
{BASE_MINUTES_PER_STEP * 7 * (1 + 0.35 * 6):.1f}, because each intervening step has to
be reconstructed by reading and inferring what it must have held. The weighted mean is
**{raw_total:.1f} minutes**.

The field table is the intervention. `{order[0][0]}` cuts triage by {order[0][1]:.0%}
for {order[0][2]:.1f} units of effort -- {raw_total * order[0][1] / order[0][2]:.2f}
minutes saved per unit, the best available. `{order[-1][0]}` cuts
{order[-1][1]:.0%} for {order[-1][2]:.1f}.

Built in payback order, the top two fields take triage from {raw_total:.1f} minutes to
{path[1][1]:.1f} for {path[1][2]:.1f} units of effort. All six reach {path[-1][1]:.1f}.

**The two cheapest fields do most of the work**, and both are things the agent framework
already has in memory at the moment it discards them. Step boundaries and tool arguments
are not derived or inferred -- they are variables that existed and were not written down.

The capacity table closes the loop with the previous listing. At raw traces, an engineer
localises {HUMAN_MIN_DAY / raw_total:.0f} a day and covering a quarter of failures needs
{FAILING * 0.25 / (HUMAN_MIN_DAY / raw_total):.1f} engineers. With the top four fields it
is {HUMAN_MIN_DAY / path[3][1]:.0f} a day and
{FAILING * 0.25 / (HUMAN_MIN_DAY / path[3][1]):.1f} engineers.

**A trace format change is worth more than tripling the team**, which is not how trace
formats are usually justified.

The last table is the honest bound. Re-running the agent with full logging localises in
{11.0:.0f} minutes -- better than any structured trace -- and it requires
ch:ops-versioning's reproducibility, which the same team probably does not have. So the
recording approach is not merely a cheaper alternative to replay; **for most teams it is
the only alternative**, because replay requires an artefact-pinning programme that
chapter found is usually incomplete.

Which gives the ordering: pin the artefacts if you can, and until then record the state,
because a trace you can read is the fallback for a run you cannot reproduce.""")
```

Where the cause sits relative to the visible failure:

```
                           cause    share   steps back   per-step check catches
--------------------------------------------------------------------------------
          tool returned an error      14%            0                      94%
        tool returned wrong data      19%            2                      11%
         arguments built wrongly      16%            1                      42%
       retrieval missed the fact      21%            3                       8%
   plan was wrong from the start      17%            6                       5%
         state corrupted mid-run      13%            4                      22%

mean distance from visible failure to cause: 2.7 steps
share a per-step correctness check would catch: 27%
```

The cause sits **2.7 steps** earlier on average, and per-step correctness checks catch
**27%**. The reason is the next table:

```
                           cause   step succeeded?  why the check passes
--------------------------------------------------------------------------------------
          tool returned an error                no  it did not
        tool returned wrong data               yes  well-formed, plausible, wrong
         arguments built wrongly               yes  valid arguments, wrong ones
       retrieval missed the fact               yes  returned documents, not the right ones
   plan was wrong from the start               yes  each step executed correctly
         state corrupted mid-run               yes  the write succeeded
```

**The causing step succeeded** in five of six categories, covering 86% of failures. Only the
tool that genuinely errored is caught reliably.

```
  steps back   share of failures   minutes to localise   weighted
--------------------------------------------------------------------
           0                 14%                   3.1        0.4
           1                 16%                   8.4        1.3
           2                 19%                  15.8        3.0
           3                 21%                  25.4        5.3
           4                 13%                  37.2        4.8
           6                 17%                  67.3       11.4
--------------------------------------------------------------------
        MEAN                100%                             26.4
```

Distance converts to minutes: **3.1** at the visible step, **67.3** six steps back, weighted
mean **26.4** ({{eq:cause-distance-drives-triage-cost}}).

```
                           field   cuts triage by   effort   minutes after   per effort
----------------------------------------------------------------------------------------
        explicit step boundaries              18%      0.5            21.6         9.50
          tool arguments as sent              22%      1.0            20.6         5.81
        tool results as received              26%      1.5            19.5         4.57
     agent state after each step              31%      4.0            18.2         2.05
   the plan, and revisions to it              15%      2.0            22.4         1.98
   causal links (this used that)              34%      7.0            17.4         1.28
```

```
                    after adding   minutes to localise   effort so far    vs raw
----------------------------------------------------------------------------------
        explicit step boundaries                  21.6             0.5     0.82x
          tool arguments as sent                  16.9             1.5     0.64x
        tool results as received                  12.5             3.0     0.47x
     agent state after each step                   8.6             7.0     0.33x
   the plan, and revisions to it                   7.3             9.0     0.28x
   causal links (this used that)                   4.8            16.0     0.18x
```

**The two cheapest fields do most of the work** — 26.4 minutes to 16.9 for 1.5 units of
effort — and both are variables the framework already holds and discards.

```
                 trace structure   minutes/trace   traces/engineer/day   engineers for 25%
--------------------------------------------------------------------------------------------
                             raw            26.4                    14                69.3
                + top two fields            16.9                    21                44.3
               + top four fields             8.6                    42                22.6
                      everything             4.8                    74                12.7
```

Covering a quarter of failures needs **69.3 engineers** on raw traces and **22.6** with the
top four fields. **A trace format change is worth more than tripling the team.**

```
                        approach   minutes  needs
----------------------------------------------------------------------------------
              read the raw trace      26.4  nothing
         read a structured trace       4.8  instrumentation
        re-run with full logging      11.0  reproducibility, ch:ops-versioning
         re-run and step through      34.0  reproducibility and an engineer
```

Replay localises in **11 minutes** — worse than a fully structured trace and better than a
raw one — but only where {{ch:ops-versioning}}'s artefact coverage is complete
({{eq:record-beats-replay}}). For most teams it is unavailable at any price.

## 10. Production Considerations

Emit step boundaries and tool arguments first. They are 1.5 units of effort for 36% of the
triage cost, and the framework has both in memory at the moment it drops them.

Record tool results untruncated for failing traces. Truncation is the correct default for
volume and fatal for the 21% of failures that are retrieval misses; the resolution is a
fidelity tier, not a global limit.

Materialise causal links at prompt-assembly time. The link between step five and step two
exists when the prompt is built and nowhere afterwards, so it is nearly free to record then
and expensive to reconstruct ever after.

Do not build per-step correctness alarms as the primary control. They catch 27%, and the
73% they miss are the categories where the step succeeded — which is where the effort
should go instead.

Tier trace fidelity by outcome, not uniformly. Full-fidelity traces on failures and a
uniform stratum on the rest, which is {{ch:ops-observability}}'s stratified design applied
to a different field set.

Report cause-distance distribution as a metric. It is the single number that says whether
your instrumentation is aimed at the right steps, and it is computable from resolved
incidents.

Treat replay as a bonus rather than a plan. It is faster than a raw trace and it requires an
artefact-pinning programme most teams have not finished.

## 11. Common Mistakes

**Reading a per-step alarm as coverage.** It covers 27%, and the miss is systematic rather
than random.

**Truncating tool results uniformly.** The truncation removes exactly the payload that
diagnoses the largest failure category.

**Choosing trace fields by completeness.** Rank by minutes saved per unit of effort; the
two cheapest fields give most of the gain.

**Budgeting triage as headcount.** Twenty-five engineers reach 9.2%; a format change reaches
further for a fraction of the cost.

**Trusting an automated diagnosis at 11%.** Nine in ten are wrong and someone acts on them.

**Assuming replay is available.** It requires reproducibility that {{ch:ops-versioning}}
found is usually incomplete.

## 12. Failure Modes

**Non-reproducible re-run closes the ticket.** The agent is re-run, succeeds by chance, and
the failure mode is retired by a coin flip.

**Structured trace with empty payloads.** Every field is present and truncated, so the trace
looks well-instrumented and diagnoses nothing.

**Automated triage feeding a runbook.** The 11% localisation is wired to an automatic
remediation, and 89% of the time the remediation is applied to the wrong component.

**Causal links inferred post hoc.** Substring matching between prompts and earlier outputs
produces links that are usually right and occasionally confidently wrong, which is worse
than none.

**Sampling that drops the failures.** A uniform trace sample at low rate misses the rare
categories entirely, which is {{eq:uniform-sampling-misses-rare-failures}} applied to
traces.

**Triage capacity mistaken for failure rate.** The number of localised failures is read as
the number of failures, so a team that triages 9% believes its failure rate is a tenth of
what it is.

## 13. Alternatives

**Full session recording.** Capture everything at full fidelity for every request. No
distance problem and a storage bill that grows with traffic rather than with failures.

**Deterministic replay infrastructure.** Pin every artefact and record every non-deterministic
draw so any run re-executes exactly. The best answer available and the most expensive
prerequisite.

**Step-level assertions with semantic checks.** Validate tool outputs against expected
properties rather than schemas. Raises the 27% and requires knowing what to assert, which is
the same problem one level down.

**Trace-level LLM judge.** Score the whole trace rather than localise within it. Cheaper and
answers a different question — whether it failed, not where.

**Human review of a stratified sample.** Accept that coverage is a sample and choose the
sample deliberately rather than by capacity. Cheap, honest, and it forecloses nothing.

## 14. Evaluation

Measure your cause-distance distribution from resolved incidents. It is the parameter every
field-ranking in this chapter depends on and it is countable from tickets you already have.

Time your engineers on real traces before and after adding a field. The 18% and 22% here are
illustrative; yours are measurable in an afternoon.

Track the share of failing traces that receive any triage at all, and publish it. Most teams
do not know this number, and it is usually below ten percent.

Test automated localisation against your own annotated traces rather than trusting a
benchmark figure. {{cite:deshpande2025trail}}'s 11% is a ceiling on a specific corpus.

Audit truncation. Sample failing traces and count how many have a tool result that was cut
before the diagnostic content.

## 15. Advanced Concepts

The independence assumed between recorded fields is generous. Step boundaries and causal
links overlap substantially — once you know where each step begins and what it called, some
of the causal structure is recoverable by inspection — so the multiplicative model
$\prod_k(1 - \gamma_k)$ overstates the combined effect. The correction bites hardest at the
end of the build order, which means the last two fields are worth less than the table
suggests and the first two, which overlap least, are worth roughly what it says. The
practical conclusion is unchanged and slightly strengthened: build the cheap fields, and
treat the expensive ones as optional.

The cause-distance distribution is treated as fixed, but it is a property of the agent's
design as much as of its failures. An agent that re-reads its full history at every step
propagates errors further than one that carries a compact state, because a corrupted item
stays in context and keeps influencing decisions. That means **architecture choices move the
distance distribution**, and a design that bounds context length is buying diagnosability as
well as cost. Neither this chapter nor {{ch:sd-architecture}} models that link, and it is
the most interesting unexplored direction here.

There is also a selection problem in the 11% figure that deserves stating. A benchmark of
annotated agent traces is built from failures somebody could annotate, which requires that
somebody could localise them — so the benchmark is drawn from the localisable tail of the
distribution, and the true accuracy over *all* production failures is likely lower than 11%
rather than higher. The direction of the bias is the unhelpful one.

Finally, the interaction with {{ch:ops-observability}}'s sampling result is sharper for
traces than for spans. A trace is expensive enough that nobody records all of them at full
fidelity, and the failing ones are exactly the ones worth recording — so the natural design
is outcome-triggered fidelity, which is biased sampling by construction. That is acceptable
for triage, where you want the failures, and unacceptable for any rate estimate computed
from the same store, which is the mistake {{eq:biased-sampling-distorts-composition}}
describes. Keep the two stores distinct or the triage corpus will silently become the
statistics corpus.

## 16. Connection to Previous Chapters

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is why per-step checks
catch 27%: the causing step succeeded, which is the same property that defeats health checks
one layer down.

{{eq:attribution-needs-payload-not-timing}} from {{ch:ops-observability}} generalises here
from a single call to a sequence, where the payload that attributes is *intermediate* state
rather than terminal output.

{{eq:uniform-sampling-misses-rare-failures}} from the same chapter decides which traces get
full fidelity, and {{sec:15-advanced-concepts}} draws out why the natural answer for traces
is biased by construction.

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is what replay
requires, and its scarcity is what makes recording the primary path rather than the fallback.

## 17. Exercises

1. Compute your own triage capacity and coverage from request volume, failure rate, and
   engineer count. What share of failing traces does anyone look at?

2. Take ten resolved agent incidents and measure the cause distance in each. How does the
   distribution compare to the one used here?

3. Rank the fields your traces do not currently record by estimated minutes saved per unit
   of instrumentation effort.

4. Audit truncation in your failing traces. What share lost the tool result that would have
   diagnosed them?

5. Model an agent that carries a compact state rather than full history. How much does the
   cause-distance distribution shift, and what does that do to $\bar{T}$?

## 18. Interview Questions

1. Why does adding engineers not solve agent triage, and what does?

2. We alarm on every failed tool call. What share of our failures will that catch?

3. Where does the cause of an agent failure usually sit relative to where it becomes
   visible, and why does that matter for instrumentation?

4. Automated trace triage is 11% accurate. Is it worth running?

5. Which two trace fields would you add first, and why those?

6. When is replaying the run better than reading the trace?

## 19. Research Questions

1. How does agent architecture — context carry-over, state compaction, planning depth —
   change the cause-distance distribution?

2. How much of the gap between 11% and human localisation accuracy is closed by trace
   structure alone, measured rather than modelled?

3. Are annotated agent-trace benchmarks biased toward localisable failures, and by how much?

4. Can causal links between steps be recorded reliably at prompt-assembly time across
   heterogeneous agent frameworks?

## 20. Chapter Summary

At 42,000 requests a day and a 9% failure rate, **3,780 traces fail daily**. One engineer
covers **0.4%**, twenty-five cover **9.2%**, and clearing a single day takes **273
engineer-days** ({{eq:triage-capacity-is-the-binding-constraint}}). Automated triage at
{{cite:deshpande2025trail}}'s **11%** correctly localises **416** and misdiagnoses **3,364**.
Twenty-five engineers plus full automation leave **79.8%** untriaged.

The lever is trace structure, because it is the only variable that improves both channels:
human triage from **26.4 minutes to 4.8**, model accuracy from **11% to 44%**
({{eq:structure-improves-both-channels}}).

What the structure must hold follows from where the causes sit. The visible failure and its
cause are **2.7 steps** apart on average, cost rises from **3.1 minutes** at distance zero to
**67.3** at distance six ({{eq:cause-distance-drives-triage-cost}}), and per-step correctness
checks catch only **27%** because in 86% of cases **the causing step succeeded**.

Step boundaries and tool arguments — 1.5 units of effort — take triage from **26.4 to 16.9
minutes**. All six fields reach **4.8**, at which point covering a quarter of failures needs
**22.6 engineers instead of 69.3**. Replay is faster than a raw trace and requires
reproducibility most teams do not have ({{eq:record-beats-replay}}).

The uncomfortable part is that none of this is technology. Every field in the ranking is a
variable the agent framework holds and then drops, and the reason it drops them is that a
trace format has never looked like a reliability investment. It looks like logging, it gets
logging's budget, and then a team staffs a triage function it cannot afford to solve a
problem a serialiser would have solved.

Carry forward: **triage capacity is the binding constraint and structure is the only lever
on it**, and **the causing step succeeded, so record what it held**.

## 21. Further Reading

- {{cite:deshpande2025trail}} — the benchmark that measures automated trace localisation and
  finds the ceiling this chapter builds on.
- {{cite:cemri2025mast}} — a taxonomy of multi-agent failure modes, showing how they
  distribute across a run.
- {{cite:sculley2015}} — the debt framing, of which discarded intermediate state is a clean
  instance.
- {{cite:breck2017}} — a readiness rubric whose monitoring section predates agents and still
  asks the right questions.
