---
id: ag-planning
number: 157
part: XVII
tier: full
status: draft
requires: [observation-informativeness, loop-is-not-a-chain, control-location]
provides: [disjoint-losses, drift-versus-quality, checkpoints-cap-the-exponent,
           checkpoint-is-a-classifier, budget-split-coupling,
           plan-as-structure]
citations: [liu2024agentbench, yao2023react, zhou2024webarena, gao2023pal,
            shinn2023reflexion, huang2024selfcorrect, sprague2024tocot]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose a plan's failure into two
disjoint losses — the plan being wrong and the world having moved — and compute
which one dominates for your task; explain why improving plan quality and
replanning more often do not substitute for each other; state what a checkpoint
does to the exponent that governs success, and why that is the largest lever in
this part; predict when adding checkpoints will make a system *worse*; and
articulate the justification for planning that survives the measurements, which is
not accuracy.

## 2. Why This Matters

Planning is the most confidently recommended technique in agent engineering and
the one with the weakest supporting measurement. This chapter tries to price it,
and gets two answers pointing in opposite directions.

The first is discouraging. A plan is a prediction about a sequence of states the
agent has not seen, made with the least information it will ever have.
{{cite:liu2024agentbench}} identifies long-horizon reasoning as a primary obstacle
across eight environments — which means planning asks models to do precisely the
thing they are worst at, and then commits to the answer.
{{sec:9-practical-example}} measures a 12-step task at $90\%$ plan quality
completing $11.4\%$ of the time, and shows that neither of the obvious rescues
dominates: better plans and more frequent replanning fix *disjoint* losses, so
which is worth more depends entirely on which loss is larger for you.

The second is the most useful result in {{part:17}}. Cut the task into segments
with a verified checkpoint between them and the same agent, the same model, the
same $90\%$ per step goes from $48.7\%$ to $97.5\%$. Nothing about the reasoning
changed. What changed is the exponent: an undivided task needs twelve consecutive
successes, and a segmented one needs two, retried.

**Splitting the task beat improving the model from $90\%$ to $95\%$ per step**
($88.7\%$ against $78.7\%$), and the advantage grows with task length — from
$+15.5$ points at six steps to $+92.4$ at thirty-two.

Those two findings together give the chapter its shape. Planning-as-prediction is
a weak lever working on the hardest available sub-problem. Planning-as-*structure*
is the strongest lever available, and the structure that matters is not the
sequence of actions. It is the set of points at which you can verify where you are
and retry from there.

There is a failure mode attached, and it is abrupt rather than gradual. At a step
budget of $14$, splitting a 12-step task into four segments scores **zero** — the
checkpoint overhead does not fit, so no run ever finishes. Budget and split have
to be chosen together.

## 3. Prerequisites

You need {{ch:ag-what-is-an-agent}}'s observation that task length hurts more than
per-step accuracy, because the entire second half of this chapter is that
observation turned into an intervention.

From {{ch:ag-react}}, the observation-informativeness result: this chapter's
"drift" is the same quantity, and the two listings agree from different
directions.

From {{ch:ag-loop}}, the stopping classifier and its two asymmetric errors — a
checkpoint is that classifier again, and the false-pass direction is the one that
causes the damage here too.

And {{ch:rsn-tool-assisted}}'s executable-check argument, which turns up in
{{sec:7-internal-mechanics}} as a structural requirement rather than a
recommendation.

## 4. Intuitive Explanation

There are two completely different things called planning, and they have different
justifications.

The first is **planning as prediction**: work out the sequence of actions before
you start. Its appeal is obvious — you think once instead of at every step — and its
weakness is equally obvious once stated. You are predicting a sequence of states
you have not observed, using the least information you will ever have about the
task. If the world does not match the prediction, the plan is not slightly wrong;
every step after the mismatch is addressing a state that no longer exists.

Two things can rescue it, and they are usually presented as alternatives. Make the
plan better. Or rewrite it more often.

Here is the part worth internalising: **those two fix different problems, and
neither can do the other's job.**

Suppose the plan is perfect and the world drifts. Every step the planner wrote was
right when written; the environment moved underneath it. No amount of planning
skill helps — a better planner would have written the same plan. Only replanning
recovers.

Now suppose the plan is imperfect and the world is completely static. Rewriting the
plan gives you a fresh draw from the same flawed planner, re-syncing with a world
that never moved. Replanning buys nothing. Only a better planner helps.

So the question "should I invest in plan quality or replanning frequency" has no
general answer. It has an arithmetic answer: compute both losses and attack the
bigger one. {{sec:9-practical-example}} does the arithmetic and finds quality
winning at its parameters — which is the opposite of the outline this chapter was
written from, and a useful reminder that the recommendation is parameter-dependent
rather than principled.

The second thing called planning is **planning as structure**, and it is where the
value actually is.

Consider what a twelve-step task asks of an agent: twelve things in a row, all
correct. At $90\%$ each that is $0.9^{12}$, about $28\%$. Now put a checkpoint
after every second step — something you can *verify*, so that if the pair went
wrong you retry just that pair from a state you know is good.

The task now asks for two things in a row, six times, with retries. The exponent
that governs success is no longer the task length; it is the segment length. That
is why the measured jump is so large, and it has nothing to do with the model
being smarter about the plan.

Two costs come with it, and both matter.

A checkpoint has to be *checked*, which is a call, so cutting too finely spends
more on verification than the verification saves. There is an optimum, and
{{sec:9-practical-example}} finds twelve segments doing worse than six.

And a checkpoint is a classifier, with {{ch:ag-loop}}'s two errors. The dangerous
one is passing bad work: the retry mechanism restores to the last state the
checkpoint *approved*, so an error waved through is an error you can never undo.
More segments means more checkpoints means more chances to be fooled, which is why
the false-pass penalty grows with the number of segments.

That gives the design rule the chapter ends on. The checkpoint should be an
executable check wherever one exists, because {{ch:rsn-tool-assisted}}'s $q = 1$ is
what makes the trade unambiguously good. A checkpoint that is the agent's own
judgement inherits {{ch:rsn-self-consistency}}'s correlation and is least reliable
exactly where the segment went wrong.

## 5. Formal Explanation

Let a plan have $k$ steps, per-step planning quality $q$, and let the environment
depart from the plan's assumptions with probability $\delta$ per step. Once
departed, planned steps are wrong until a replan re-syncs.

With a replan every $r$ steps, success factorises into two independent terms:

$$S(q, r) = \underbrace{q^{\,k}}_{\text{plan quality}} \times \underbrace{\big[(1-\delta)^{\,r}\big]^{k/r}}_{\text{drift within segments}}$$ (eq:disjoint-losses)

Look at what each term contains. The first has no $\delta$ and no $r$; the second
has no $q$. **The two losses are disjoint**, which is why they compose and why
neither substitutes for the other:

$$\frac{\partial S}{\partial q} \text{ is independent of } r, \qquad \frac{\partial S}{\partial r} \text{ is independent of } q$$ (eq:drift-versus-quality)

The practical decision follows from comparing their magnitudes rather than from a
preference. The quality loss is $1 - q^k$; the drift loss, with replanning every
$r$, is $1 - (1-\delta)^{r}$ per segment. Attack whichever is larger, and expect
the answer to flip as $q$, $\delta$ or $k$ moves.

Note also that {{eq:disjoint-losses}}'s drift term is not improved by $r < 1$'s
limit in the way intuition suggests: at $r = 1$ it becomes $(1-\delta)^{k}$ — one
step's worth of drift exposure per step — which is the floor, not zero. **Fully
reactive execution does not eliminate drift; it minimises exposure to it.**

Now decomposition. Split $k$ steps into $m$ segments of length $\ell = k/m$, with a
checkpoint after each. A segment succeeds if all $\ell$ steps succeed; a failed
segment is retried from the last verified state. With enough budget, the
probability a segment eventually passes approaches 1, and the whole task's success
is governed by the checkpoint rather than by the steps:

$$S_{\text{seg}} = \prod_{i=1}^{m} \Pr[\text{segment } i \text{ eventually verified correct}]$$ (eq:checkpoints-cap-the-exponent)

Contrast with the undivided case $S = p^{k}$. **The exponent that governs success
changes from the task length to the segment length**, because retries make the
segment's own failure recoverable.

The checkpoint is not free. With sensitivity $\alpha_c$ and false-pass rate
$\beta_c$, a segment can be approved while wrong, and that error is unrecoverable
— the retry restores to an approved state:

$$S_{\text{seg}} \approx \prod_{i=1}^{m} \big(1 - \beta_c \cdot \Pr[\text{segment wrong at approval}]\big)$$ (eq:checkpoint-is-a-classifier)

which decays in $m$. **Decomposition trades one exponent for another**: it removes
$k$ from the success exponent and adds $m$. The trade is good precisely while the
checkpoint is more reliable than a step.

Finally, budget. Each segment costs $\ell$ steps per attempt plus one verification
call, and a failed segment is retried. Expected consumption is:

$$\mathbb{E}[\text{budget}] \approx m\left(\frac{\ell}{p^{\ell}} + 1\right)$$ (eq:budget-split-coupling)

The $+1$ per segment is the verification overhead, and it grows linearly in $m$
while the $\ell/p^{\ell}$ term falls. Below a budget threshold the overhead does
not fit at all, and the system does not degrade — it stops completing anything.
**Budget and split must be chosen together**, and {{sec:9-practical-example}}
measures a four-way split scoring exactly zero at a budget where a single segment
scores $28.4\%$.

## 6. Mathematical Foundation

Three consequences worth extracting.

**The optimal split has a closed-ish form.** Minimising
{{eq:budget-split-coupling}} over $\ell$ with a checkpoint cost $c$ gives a
first-order condition balancing the retry cost $\ell/p^{\ell}$ against the
verification cost $c \cdot k/\ell$. The retry term grows super-linearly in $\ell$
(because $p^{\ell}$ shrinks geometrically) while the verification term falls as
$1/\ell$, so the optimum is interior and sits at short segments — two to four
steps at the reliabilities agents actually have. That matches
{{sec:9-practical-example}}'s measured optimum of six segments over twelve steps.

**The decomposition gain grows with $k$.** Undivided success is $p^{k}$ and
segmented success is approximately constant in $k$ for fixed $\ell$, so the gap
widens without bound. {{sec:9-practical-example}} measures $+15.5$ points at
$k=6$ and $+92.4$ at $k=32$. This is the precise sense in which
{{cite:liu2024agentbench}}'s long-horizon bottleneck is *structural*: the models
are not failing to reason over long horizons so much as being asked to succeed
$k$ times consecutively when they could be asked to succeed $\ell$ times
consecutively, $m$ times.

**The checkpoint's false-pass rate is the binding constraint at high $m$.** From
{{eq:checkpoint-is-a-classifier}}, the penalty is roughly $m\beta_c$ times the
chance of an unnoticed error, so it scales linearly in the number of segments
while the benefit saturates. That gives a second, independent reason for an
interior optimum, and it is the reason a fine split with a weak checkpoint is
worse than a coarse split with a strong one.

One caveat about {{eq:disjoint-losses}}. It gives a replan a free, perfect
re-sync. A real replan is a fresh long-horizon prediction with the same quality
$q$ as the first one, *and* it requires detecting that drift occurred — which
{{ch:ag-react}} measured as the hard part. Both make the replanning column in
{{sec:9-practical-example}} optimistic, and the direction of the bias favours the
intervention this chapter is already sceptical about.

## 7. Internal Mechanics

### 7.1 What a plan is made of

```mermaid {#fig:plan-structure caption="Two readings of the same plan. The action sequence is a prediction; the checkpoints are structure. The measurements say the second is worth more."}
flowchart LR
    subgraph pred [as prediction]
        A1[step 1] --> A2[step 2] --> A3[step 3] --> A4[step 4]
    end
    subgraph struct [as structure]
        B1[steps 1-2] --> C1{{verify}}
        C1 --> B2[steps 3-4]
        B2 --> C2{{verify}}
        C1 -. retry .-> B1
        C2 -. retry .-> B2
    end
```

Most plan representations record only the top row. Recording the bottom row costs
nothing extra at planning time — it is a decision about where the boundaries are —
and it is what {{eq:checkpoints-cap-the-exponent}} operates on.

### 7.2 What makes a good checkpoint

A checkpoint must be a statement about *state*, not about *action*. "Called the
search tool" is not checkable; "the customer record is loaded and has a non-empty
address" is. The distinction matters because the retry restores to a state, and a
state you cannot describe is a state you cannot restore to.

In rough order of strength:

**Executable**: a query returns the expected row, the file parses, the tests pass.
{{ch:rsn-tool-assisted}}'s $q=1$ case, and the only one where
{{eq:checkpoint-is-a-classifier}}'s penalty vanishes.

**Structural**: a required field is populated, an invariant holds. Cheap, and
covers what happens to be typed.

**Judged**: the model reads the state and decides. Most general and subject to
{{ch:rsn-self-consistency}}'s correlation — least reliable exactly when the segment
went wrong, which is the worst possible failure profile for
{{eq:checkpoint-is-a-classifier}}.

### 7.3 Plans as expectation records

{{ch:ag-react}}'s surprise detector needed something to compare an observation
against. A plan that records *expected outcomes* alongside actions supplies it for
free, converting drift detection from a judgement into a comparison.

This is under-used and close to costless: the planner is already reasoning about
what each step will produce, and writing it down turns a private inference into a
checkable artefact. It is also, not coincidentally, what turns a plan step into a
checkpoint.

### 7.4 Hierarchical plans

A coarse plan whose steps expand into finer plans when reached has two effects on
the arithmetic. It lowers effective drift at the top level, because coarse steps
are less specific and therefore less falsifiable by a changed detail. And it
supplies a natural segmentation, because each coarse step is a checkpoint boundary.

The risk is that a coarse step is not *checkable* — "handle the refund" is not a
state — so hierarchy buys the segmentation only if each level's boundaries are
expressed as verifiable conditions. Hierarchy without checkable boundaries is
decomposition on paper and $p^{k}$ in practice.

### 7.5 Where the budget goes

{{eq:budget-split-coupling}} says verification cost grows linearly in the number of
segments while retry savings saturate. In serving terms ({{part:15}}), each
checkpoint is an extra model call or an extra tool round trip, on a context that
has been growing since the run started.

The practical consequence is the cliff in {{sec:9-practical-example}}: adding
checkpoints without raising the budget can take a system from working to
completing nothing, and it will look like the checkpoints broke the agent rather
than that they exhausted the allowance.

## 8. Implementation

Two listings. The first separates the two things that could rescue a plan and
measures which loss each one fixes. The second measures what checkpoints do to the
exponent, and what they cost.

```python {tier=A name=disjoint-losses}
"""Is a better plan worth more than a more frequent one?

cite:liu2024agentbench identifies long-horizon reasoning as a primary obstacle for
agents across eight environments. A plan is long-horizon reasoning in its purest
form: a prediction, made before any observation, about a sequence of states the
agent has not seen.

So planning asks models to do the thing they are worst at, and then commits to the
answer. This listing separates the two things that could rescue that -- making the
plan better, and rewriting it more often -- and asks which is worth more
(eq:replan-rate-dominates-plan-quality).

The environment drifts: at each step there is some chance the world is not what
the plan assumed. Once that has happened, every subsequent planned step is
addressing a state that no longer exists, until somebody replans.
"""
import numpy as np

rng = np.random.default_rng(2003)

N = 60000
K = 12                  # steps in the task
DRIFT = 0.08            # chance per step that the state departs from the plan


def run(quality, replan_every, k=K, drift=DRIFT):
    """quality: chance a planned step is correct GIVEN the assumed state holds.
    replan_every: rewrite the plan every r steps. r=1 is fully reactive,
    r=k is plan-once-and-execute."""
    ok = np.ones(N, dtype=bool)
    stale = np.zeros(N, dtype=bool)      # the plan no longer matches reality
    calls = np.ones(N)                   # the initial plan
    for i in range(k):
        if i > 0 and i % replan_every == 0:
            stale[:] = False             # a rewrite re-syncs with the world
            calls += 1
        # A step succeeds if the plan is right AND the plan still applies.
        good = (rng.random(N) < quality) & ~stale
        ok &= good
        stale |= rng.random(N) < drift
        calls += 1
    return float(ok.mean()), float(calls.mean())


QUALITIES = [0.90, 0.94, 0.97, 0.99]
RATES = [12, 6, 4, 2, 1]

print(f"A {K}-step task. The world departs from the plan's assumptions")
print(f"{DRIFT:.0%} of the time per step. `replan every r` rewrites the plan")
print("every r steps; r=12 is plan-once, r=1 is fully reactive.")
print()
print(f"{'plan quality':>14}" + "".join(f"{'r=' + str(r):>10}" for r in RATES))
print("-" * 64)
tab = {}
for q in QUALITIES:
    row = {}
    for r in RATES:
        row[r] = run(q, r)
        tab[(q, r)] = row[r]
    print(f"{q:>14.0%}" + "".join(f"{row[r][0]:>10.1%}" for r in RATES))

print()
print()
print("The same grid, in model calls -- replanning is not free.")
print()
print(f"{'plan quality':>14}" + "".join(f"{'r=' + str(r):>10}" for r in RATES))
print("-" * 64)
for q in QUALITIES:
    print(f"{q:>14.0%}" + "".join(f"{tab[(q, r)][1]:>10.1f}" for r in RATES))

print()
print()
print("Two ways to spend, from a common baseline of quality 90%, r=12.")
print()
base_s, base_c = tab[(0.90, 12)]
print(f"{'change':>38}{'success':>11}{'gain':>9}{'calls':>9}")
print("-" * 67)
moves = [("baseline: quality 90%, plan once", (0.90, 12)),
         ("quality 90% -> 99% (plan once)", (0.99, 12)),
         ("keep 90%, replan every 4 steps", (0.90, 4)),
         ("keep 90%, replan every 2 steps", (0.90, 2)),
         ("keep 90%, replan every step", (0.90, 1)),
         ("quality 99% AND replan every 2", (0.99, 2))]
mv = {}
for name, key in moves:
    s, c = tab[key]
    mv[name] = (s, c)
    print(f"{name:>38}{s:>11.1%}{s - base_s:>+9.1%}{c:>9.1f}")

print()
print()
print("How much of the loss is drift, and how much is the plan being wrong?")
print("Hold one at zero and vary the other, at r=12 and r=2.")
print()
print(f"{'condition':>34}{'r=12':>10}{'r=2':>10}")
print("-" * 54)
iso = {}
for name, q, d in [("perfect plan, no drift", 1.0, 0.0),
                   ("perfect plan, drift 8%", 1.0, DRIFT),
                   ("quality 90%, no drift", 0.90, 0.0),
                   ("quality 90%, drift 8%", 0.90, DRIFT)]:
    a = run(q, 12, drift=d)[0]
    b = run(q, 2, drift=d)[0]
    iso[name] = (a, b)
    print(f"{name:>34}{a:>10.1%}{b:>10.1%}")

print()
print()
print("And how the answer moves with how volatile the environment is.")
print()
print(f"{'drift':>8}{'plan once':>12}{'replan /4':>12}{'replan /1':>12}"
      f"{'best':>12}")
print("-" * 56)
dr = {}
for d in (0.0, 0.02, 0.05, 0.10, 0.20):
    a = run(0.94, 12, drift=d)[0]
    b = run(0.94, 4, drift=d)[0]
    c = run(0.94, 1, drift=d)[0]
    dr[d] = (a, b, c)
    best = ["plan once", "replan /4", "replan /1"][int(np.argmax([a, b, c]))]
    print(f"{d:>8.0%}{a:>12.1%}{b:>12.1%}{c:>12.1%}{best:>12}")

print(f"""
The first table is the comparison, and it did not come out the way the chapter
was outlined to expect.

Along the top row -- plan quality {QUALITIES[0]:.0%} -- going from planning once
to replanning every step takes success from {tab[(0.90, 12)][0]:.1%} to
{tab[(0.90, 1)][0]:.1%}, a gain of {tab[(0.90, 1)][0] - tab[(0.90, 12)][0]:.1%}.

Down the left column -- plan once -- going from quality {QUALITIES[0]:.0%} to
{QUALITIES[-1]:.0%} takes it from {tab[(0.90, 12)][0]:.1%} to
{tab[(0.99, 12)][0]:.1%}, a gain of
{tab[(0.99, 12)][0] - tab[(0.90, 12)][0]:.1%}.

**Plan quality moves the number further than replanning does**, at these
parameters, and the intuition that a frequently-rewritten mediocre plan beats a
good one is simply wrong here. The interesting question is why, and the third
table answers it.

The second table prices replanning first, because it is cheap: every {4} steps
costs {tab[(0.90, 4)][1] - tab[(0.90, 12)][1]:.1f} extra calls on a {K}-step
task, every step costs {tab[(0.90, 1)][1] - tab[(0.90, 12)][1]:.1f}. Quality is
free in call terms and expensive in every other sense -- it is a better model, a
better prompt, or a better planner, none of which is a configuration change.

Note the last row: quality {0.99:.0%} AND replanning every {2} steps reaches
{tab[(0.99, 2)][0]:.1%}, against {tab[(0.99, 12)][0]:.1%} for quality alone and
{tab[(0.90, 2)][0]:.1%} for replanning alone. **The two compose**, which is the
first clue that they are not competing.

The third table is the one to take away, because it separates the two losses
instead of comparing them.

With a perfect plan and no drift, success is
{iso['perfect plan, no drift'][0]:.1%}. Introduce {DRIFT:.0%} drift and a perfect
plan falls to {iso['perfect plan, drift 8%'][0]:.1%}, recovering to
{iso['perfect plan, drift 8%'][1]:.1%} when replanned every {2} steps. That is
the DRIFT loss: the plan was correct when written and the world moved. Replanning
addresses it and plan quality cannot -- there is no plan skilful enough to predict
an unobserved change.

Now the other axis. Quality {0.9:.0%} with NO drift scores
{iso['quality 90%, no drift'][0]:.1%} at r={12} and
{iso['quality 90%, no drift'][1]:.1%} at r={2}. Replanning changes nothing,
because a rewrite re-syncs the plan with a world that never moved. That is the
QUALITY loss, and only a better planner touches it.

**The two interventions fix disjoint losses**, which is why they compose in the
second table and why neither dominates in general. Which one is worth more is
decided by which loss is bigger, and that is arithmetic you can do in advance:
the quality loss is $1 - q^{{k}}$ and the drift loss is governed by
$1 - (1-\delta)^{{r}}$ per planning segment. At {0.9:.0%} quality over {K} steps
the first is {1 - 0.9 ** K:.0%}; at {DRIFT:.0%} drift over {12} steps the second
is {1 - (1 - DRIFT) ** 12:.0%}. The quality term is larger, so quality wins --
and at {0.99:.0%} quality it would not be.

That is the chapter's actual finding, and it is more useful than a
recommendation: **compute the two losses before choosing which to attack**, and
expect the answer to flip as either parameter moves.

The fourth table sweeps the drift side to show the flip. At {0:.0%} drift,
planning once scores {dr[0.0][0]:.1%} against reactive's {dr[0.0][2]:.1%} --
identical, and the extra {tab[(0.94, 1)][1] - tab[(0.94, 12)][1]:.0f} calls
bought nothing whatsoever. At {0.2:.0%} drift it is {dr[0.2][0]:.1%} against
{dr[0.2][2]:.1%}.

That agrees with ch:ag-react's informativeness sweep from a different direction,
and the two together are the durable version of this part's architecture advice:
**the volatility of the environment decides how often to replan, and the
capability of the model decides how good the plan is, and neither substitutes for
the other.**

Two honest caveats on what this listing does not show.

It gives replanning a free, perfect re-sync: a rewrite always restores the plan to
match the world. A real replan is a fresh long-horizon prediction with the same
quality {QUALITIES[0]:.0%}--{QUALITIES[-1]:.0%} problem as the first one, and it
also needs to DETECT that drift occurred, which ch:ag-react measured as the hard
part. Both make the replanning column optimistic.

And it treats plan quality as a free parameter. cite:liu2024agentbench identifies
long-horizon reasoning as a primary agent bottleneck, which is precisely the
capability the quality column represents -- so the column that wins here is the
one that is hardest to move. **Planning's problem is not that better plans are not
worth having. It is that a plan is the most long-horizon thing an agent does, made
with the least information it will ever have.**

Which leaves the justification this listing cannot score, and it is not accuracy.
A plan is a STRUCTURE: inspectable before execution, checkable during it, and a
place for a human to intervene. ch:ag-loop needed exactly such a structure for a
completion condition that is not the agent's own judgement, and ch:ag-termination
will need one for a budget. **The plan earns its place as an artefact other
components can use, rather than as a prediction to be followed.**""")
```

The second listing leaves the plan's content alone and changes its structure.

```python {tier=A name=checkpoints-cap-the-exponent}
"""Decomposition: why a plan's structure is worth more than its content.

ch:ag-what-is-an-agent found that task LENGTH hurts more than per-step accuracy,
because success is a per-step base raised to k. That suggests an intervention the
previous listing could not measure: do not make the plan better, make the task
shorter -- by cutting it into segments with a checkable boundary between them
(eq:checkpoints-cap-the-exponent).

A checkpoint does one thing. It converts a failure that loses the whole run into
one that loses the current segment, because the segment can be retried from a
state that is known good. The exponent that governs success stops being the task
length and becomes the SEGMENT length.

The cost is that a checkpoint must be verified, which is a call, and the
verification is itself a classifier with the two asymmetric errors ch:ag-loop
measured.
"""
import numpy as np

rng = np.random.default_rng(2087)

N = 40000
K = 12                  # total steps
P_STEP = 0.90           # a step succeeds
BUDGET = 30             # total step budget including retries


def run(segments, p_step=P_STEP, ck_tpr=1.0, ck_fpr=0.0, budget=BUDGET, k=K):
    """Split k steps into `segments` equal parts. After each segment a
    checkpoint verifies it: ck_tpr is the chance a good segment is passed,
    ck_fpr the chance a bad one is passed anyway (and its error carried on)."""
    seg_len = k // segments
    ok = np.ones(N, dtype=bool)
    spent = np.zeros(N, dtype=np.int64)
    alive = np.ones(N, dtype=bool)
    for _ in range(segments):
        seg_done = np.zeros(N, dtype=bool)
        corrupted = np.zeros(N, dtype=bool)
        for _attempt in range(budget):          # retry the segment until pass
            live = alive & ~seg_done & (spent + seg_len <= budget)
            idx = np.flatnonzero(live)
            if not len(idx):
                break
            spent[idx] += seg_len
            good = (rng.random((len(idx), seg_len)) < p_step).all(1)
            u = rng.random(len(idx))
            passed = np.where(good, u < ck_tpr, u < ck_fpr)
            seg_done[idx[passed]] = True
            corrupted[idx[passed & ~good]] = True
            spent[idx] += 1                      # the checkpoint call itself
        alive &= seg_done
        ok &= seg_done & ~corrupted
    return float(ok.mean()), float(spent.mean())


print(f"A {K}-step task, {P_STEP:.0%} per step, budget {BUDGET} steps. Split into")
print("equal segments with a verified checkpoint between them; a failed segment")
print("is retried from the last good state.")
print()
print(f"{'segments':>10}{'steps each':>13}{'completed':>12}{'steps used':>13}")
print("-" * 48)
seg_tab = {}
for m in (1, 2, 3, 4, 6, 12):
    seg_tab[m] = run(m)
    print(f"{m:>10}{K // m:>13}{seg_tab[m][0]:>12.1%}{seg_tab[m][1]:>13.1f}")

print()
print()
print("The same, holding the checkpoint imperfect. A checkpoint that passes bad")
print("work carries the error forward and the task fails anyway.")
print()
print(f"{'segments':>10}{'perfect':>11}{'fpr 5%':>10}{'fpr 15%':>10}"
      f"{'tpr 85%':>11}")
print("-" * 52)
ck_tab = {}
for m in (1, 2, 3, 4, 6):
    a = run(m)[0]
    b = run(m, ck_fpr=0.05)[0]
    c = run(m, ck_fpr=0.15)[0]
    d = run(m, ck_tpr=0.85)[0]
    ck_tab[m] = (a, b, c, d)
    print(f"{m:>10}{a:>11.1%}{b:>10.1%}{c:>10.1%}{d:>11.1%}")

print()
print()
print("Decomposition against a better model, from the same baseline.")
print()
print(f"{'change':>40}{'completed':>12}{'steps':>9}")
print("-" * 61)
mv = {}
for name, args in [("baseline: 1 segment, step 90%", (1, 0.90)),
                   ("step 90% -> 95%, still 1 segment", (1, 0.95)),
                   ("step 90% -> 99%, still 1 segment", (1, 0.99)),
                   ("keep 90%, split into 3 segments", (3, 0.90)),
                   ("keep 90%, split into 4 segments", (4, 0.90)),
                   ("step 95% AND 4 segments", (4, 0.95))]:
    r = run(args[0], p_step=args[1])
    mv[name] = r
    print(f"{name:>40}{r[0]:>12.1%}{r[1]:>9.1f}")

print()
print()
print("Does the best split depend on the budget? Sweep both.")
print()
print(f"{'budget':>8}" + "".join(f"{str(m) + ' seg':>10}" for m in (1, 2, 3, 4, 6))
      + f"{'best':>8}")
print("-" * 66)
bd = {}
for b in (14, 18, 24, 30, 45):
    row = [run(m, budget=b)[0] for m in (1, 2, 3, 4, 6)]
    bd[b] = row
    best = [1, 2, 3, 4, 6][int(np.argmax(row))]
    print(f"{b:>8}" + "".join(f"{v:>10.1%}" for v in row) + f"{best:>8}")

print()
print()
print("And how it moves with task length, at a budget of 2.5x the task.")
print()
print(f"{'steps k':>9}{'1 segment':>12}{'k/4 segments':>15}{'gain':>9}")
print("-" * 45)
kl = {}
for k in (6, 12, 20, 32):
    a = run(1, budget=int(2.5 * k), k=k)[0]
    b = run(max(k // 3, 1), budget=int(2.5 * k), k=k)[0]
    kl[k] = (a, b)
    print(f"{k:>9}{a:>12.1%}{b:>15.1%}{b - a:>+9.1%}")

print(f"""
The first table is the effect, and the size of it is the point.

The same task, the same model, the same {P_STEP:.0%} per step: {seg_tab[1][0]:.1%}
undivided, {seg_tab[6][0]:.1%} split into six segments of two steps. Nothing about
the agent changed. What changed is the exponent -- an undivided task needs
{K} consecutive successes and a segmented one needs {K // 6}, retried
(eq:checkpoints-cap-the-exponent).

Note that it turns over: {seg_tab[12][0]:.1%} at twelve segments, below the
{seg_tab[6][0]:.1%} at six. A checkpoint costs a call, and at one step per segment
the verification overhead is as large as the work -- {seg_tab[12][1]:.1f} steps
against {seg_tab[6][1]:.1f}. **There is an interior optimum in how finely to cut**,
and it is set by the ratio of checkpoint cost to segment length.

The second table is the thing that makes this harder than it looks, and it is the
reason decomposition is not free.

Every checkpoint is a classifier, with ch:ag-loop's two asymmetric errors. A
checkpoint that PASSES BAD WORK carries the error into the next segment, where it
cannot be repaired -- the retry mechanism only restores to the last state the
checkpoint approved. At six segments, a {0.15:.0%} false-pass rate takes
completion from {ck_tab[6][0]:.1%} to {ck_tab[6][2]:.1%}.

And the damage grows with the number of segments, because more segments means more
checkpoints to fool. Compare the {0.15:.0%} column down the rows: the gap from
perfect widens from {ck_tab[1][0] - ck_tab[1][2]:.1%} at one segment to
{ck_tab[6][0] - ck_tab[6][2]:.1%} at six.

**So decomposition trades one exponent for another.** It removes the task length
from the success exponent and adds the checkpoint count. That is a good trade only
while the checkpoint is more reliable than a step, which is exactly why the
checkpoint should be an executable check rather than a judgement --
ch:rsn-tool-assisted's argument arriving as a structural requirement.

The third table prices it against the alternative everyone reaches for first.

Improving the model from {0.90:.0%} to {0.95:.0%} per step takes completion from
{mv['baseline: 1 segment, step 90%'][0]:.1%} to
{mv['step 90% -> 95%, still 1 segment'][0]:.1%}. Keeping the {0.90:.0%} model and
cutting the task into three takes it to
{mv['keep 90%, split into 3 segments'][0]:.1%}.

**Splitting the task beats a five-point model improvement**, and it is a change to
the prompt and the control flow rather than to the model. The two also compose:
{0.95:.0%} steps with four segments reaches
{mv['step 95% AND 4 segments'][0]:.1%}.

The fourth table is the caveat that matters most in production, and it is a
failure mode rather than a diminishing return.

At a budget of {14} steps, splitting into four segments scores
{bd[14][3]:.1%}. Not "less than one segment" -- zero. The checkpoint calls plus
the segment retries do not fit in the budget at all, so no run ever finishes.
**Decomposition consumes budget before it saves any**, and a system that adds
checkpoints without raising the step budget can go from working to completely
broken in one change.

Above the threshold the ordering reverses hard: at budget {30} the best split is
{6} segments at {bd[30][4]:.1%} against one segment's {bd[30][0]:.1%}. So the
budget and the split have to be chosen together, and neither is meaningful alone.

The last table is the reason this is the most important lever in the part. The
gain from decomposition grows with task length: {kl[6][1] - kl[6][0]:+.1%} at
{6} steps, {kl[12][1] - kl[12][0]:+.1%} at {12}, {kl[32][1] - kl[32][0]:+.1%} at
{32}, where an undivided task completes {kl[32][0]:.1%} of the time and a
segmented one {kl[32][1]:.1%}.

That is the direct consequence of what it does to the exponent, and it says
something specific about cite:liu2024agentbench's finding that long-horizon
consistency is the agent bottleneck. **The bottleneck is not that models cannot
reason over long horizons. It is that nothing was checkpointing them**, so a
twenty-step task was being asked to succeed twenty times consecutively rather than
five times consecutively, four times.

Which is the honest resolution of this chapter's two halves. The previous listing
found planning-as-prediction to be a weak lever, working on the capability models
are worst at. This one finds planning-as-STRUCTURE to be the strongest lever
available -- and the structure that matters is not the sequence of actions. It is
the set of points at which you can verify where you are and retry from there.

**A plan whose steps are checkable is worth far more than a plan whose steps are
correct**, and if you have to choose which property to optimise, the arithmetic
above says which one.""")
```

## 9. Practical Example

The first listing runs a 12-step task where the world departs from the plan's
assumptions $8\%$ of the time per step.

```
  plan quality      r=12       r=6       r=4       r=2       r=1
----------------------------------------------------------------
           90%     11.4%     12.4%     13.6%     17.1%     28.0%
           94%     19.1%     20.8%     22.6%     28.8%     47.9%
           97%     27.8%     30.2%     32.9%     42.0%     69.2%
           99%     35.0%     38.5%     41.9%     53.6%     88.5%
```

Reading across the top row, replanning every step instead of once buys $+16.6$
points. Reading down the left column, quality $90\% \to 99\%$ buys $+23.6$.
**Plan quality moves the number further than replanning does at these
parameters** — the opposite of the outline this chapter started from.

Why is in the decomposition:

```
                         condition      r=12       r=2
------------------------------------------------------
            perfect plan, no drift    100.0%    100.0%
            perfect plan, drift 8%     39.9%     60.6%
             quality 90%, no drift     28.2%     28.1%
             quality 90%, drift 8%     11.1%     17.2%
```

A perfect plan loses $60.1$ points to drift alone, recovering to $60.6\%$ when
replanned every two steps — that loss is replanning's to fix and plan quality
cannot touch it. Quality $90\%$ with *no* drift scores $28.2\%$ at $r=12$ and
$28.1\%$ at $r=2$ — replanning buys nothing, because a rewrite re-syncs with a
world that never moved.

**The two fix disjoint losses** ({{eq:disjoint-losses}}), which is why they compose
— quality $99\%$ *and* replanning every two steps reaches $53.6\%$ — and why
neither dominates in general. At $90\%$ over 12 steps the quality loss is $72\%$
and the drift loss over 12 steps is $63\%$; the larger one wins, and at $99\%$
quality it would be the other way.

Sweeping volatility confirms it agrees with {{ch:ag-react}}'s informativeness
result from a different direction:

```
   drift   plan once   replan /4   replan /1        best
--------------------------------------------------------
      0%       47.2%       47.4%       47.3%   replan /4
      5%       27.1%       30.3%       47.8%   replan /1
     20%        9.4%       14.6%       47.4%   replan /1
```

At zero drift the extra eleven calls bought nothing whatsoever.

The second listing leaves the plan's content alone and changes its structure. Same
task, same $90\%$ per step, split into segments with a verified checkpoint between
them:

```
  segments   steps each   completed   steps used
------------------------------------------------
         1           12       48.7%         22.3
         2            6       73.3%         22.2
         4            3       90.9%         21.3
         6            2       97.5%         22.1
        12            1       94.6%         26.5
```

$48.7\% \to 97.5\%$ with no change to the agent. The exponent governing success
went from the task length to the segment length
({{eq:checkpoints-cap-the-exponent}}). And it turns over at twelve segments,
because verification overhead becomes as large as the work — $26.5$ steps against
$22.1$.

Against the alternative teams reach for first:

```
                                  change   completed    steps
-------------------------------------------------------------
           baseline: 1 segment, step 90%       48.9%     22.3
        step 90% -> 95%, still 1 segment       78.7%     19.0
         keep 90%, split into 3 segments       88.7%     21.8
                 step 95% AND 4 segments       99.0%     18.6
```

**Splitting the task beats a five-point model improvement**, and it is a control-flow
change rather than a model change. The two compose.

The checkpoint is not free, and the false-pass direction is the dangerous one:

```
  segments    perfect    fpr 5%   fpr 15%    tpr 85%
----------------------------------------------------
         1      48.3%     47.6%     45.5%      42.9%
         3      88.7%     83.7%     74.6%      75.7%
         6      97.4%     91.4%     80.3%      82.8%
```

The gap from perfect widens from $2.8$ points at one segment to $17.1$ at six,
because more segments means more checkpoints to fool
({{eq:checkpoint-is-a-classifier}}). A checkpoint that approves bad work carries
the error into a state the retry mechanism treats as good.

And the failure mode that matters most in production:

```
  budget     1 seg     2 seg     3 seg     4 seg     6 seg    best
------------------------------------------------------------------
      14     28.4%     28.3%     28.1%      0.0%      0.0%       1
      24     28.2%     54.9%     77.7%     79.5%     81.9%       6
      45     62.7%     91.7%     99.0%     99.8%    100.0%       6
```

At a budget of $14$, four segments scores **zero** — not "less than one segment".
The checkpoint calls plus retries do not fit, so nothing ever finishes
({{eq:budget-split-coupling}}). **Adding checkpoints without raising the budget
can take a system from working to completing nothing in one change.**

Finally, why this is the largest lever in {{part:17}}:

```
  steps k   1 segment   k/4 segments     gain
---------------------------------------------
        6       78.0%          93.5%   +15.5%
       12       48.9%          90.8%   +41.9%
       32        6.8%          99.2%   +92.4%
```

The gain grows without bound in task length, which says something specific about
{{cite:liu2024agentbench}}'s long-horizon bottleneck: **the models are not failing
to reason over long horizons so much as being asked to succeed thirty-two times
consecutively when they could be asked to succeed three times consecutively,
eleven times.**

## 10. Production Considerations

Compute both losses before choosing what to fix. The quality loss is $1 - q^k$ and
the drift loss is set by how often the world departs from the plan. They are
disjoint ({{eq:disjoint-losses}}) and the larger one is where the return is.

Segment every task longer than about four steps. This is the highest-return change
in {{part:17}} and it is a control-flow change, not a model change.

Choose the split and the budget together. {{eq:budget-split-coupling}} has a cliff,
and crossing it produces a system that completes nothing while looking like a
checkpoint bug.

Make every checkpoint a statement about *state*, not about action taken. A state
you cannot describe is a state you cannot restore to.

Prefer executable checkpoints. {{eq:checkpoint-is-a-classifier}}'s penalty scales
with the number of segments, so a fine split needs a strong check — and a
model-judged checkpoint inherits {{ch:rsn-self-consistency}}'s correlation exactly
where the segment went wrong.

Record expected outcomes in the plan, not just actions. It converts
{{ch:ag-react}}'s drift detection from a judgement into a comparison, and it is
what turns a plan step into a checkpoint.

Track the false-pass rate of your checkpoints as a first-class metric. It is the
term that binds at high segment counts, and it is invisible in end-to-end success
until it is large.

## 11. Common Mistakes

**Treating plan quality and replan frequency as substitutes.** They fix disjoint
losses ({{eq:drift-versus-quality}}) and compose.

**Investing in better plans when drift dominates.** A perfect plan lost $60.1$
points to drift alone in {{sec:9-practical-example}}.

**Replanning frequently when the environment is static.** At zero drift, eleven
extra calls bought $+0.1$ points.

**Splitting without raising the budget.** Zero completion at budget $14$ with four
segments.

**Splitting too finely.** Twelve segments scored below six, because verification
overhead exceeded the retry saving.

**Using the agent's own judgement as the checkpoint.** It is least reliable exactly
where the segment went wrong, and its errors are unrecoverable.

**Hierarchy without checkable boundaries.** Coarse steps that are not states give
you decomposition on paper and $p^k$ in practice.

## 12. Failure Modes

*Stale-plan execution.* Drift occurred, was not detected, and the remaining steps
run against a state that no longer exists. Every step "succeeds".

*Approved corruption.* A checkpoint passes bad work; the retry mechanism now
treats that state as good and can never restore past it. The unrecoverable failure
in {{eq:checkpoint-is-a-classifier}}.

*Budget starvation from checkpoints.* The cliff in
{{eq:budget-split-coupling}} — abrupt, total, and easy to misdiagnose.

*Checkpoint theatre.* Boundaries that are declared but not verified, which cost the
call and provide none of the retry benefit.

*Plan-quality investment with no measurement.* Effort spent on the hardest
sub-problem models have ({{cite:liu2024agentbench}}) without first checking whether
the quality loss is the binding one.

## 13. Alternatives

**Reactive execution.** {{ch:ag-react}}: no plan at all, decide each step from the
current state. Correct when drift is high, and it forfeits the structure that
{{sec:9-practical-example}}'s second listing shows is the real prize.

**Replan on surprise.** {{ch:ag-react}}'s hybrid, which dominated both pure
strategies there. Composes with decomposition rather than competing.

**Program generation.** {{cite:gao2023pal}}: where the whole task is computational
and the environment static, emit one program. The extreme of the planning end.

**Externally supplied plans.** A human or a workflow supplies the structure and the
agent fills in segments. This removes the quality loss entirely and keeps the
checkpoint benefit, and it is what a router-with-agent-fallback
({{ch:ag-what-is-an-agent}}) amounts to.

**No decomposition, bigger budget.** {{sec:9-practical-example}} says this works —
$62.7\%$ at budget $45$ undivided — and costs far more than segmenting to reach the
same place.

## 14. Evaluation

Measure drift directly: for a sample of traces, how often did an observation
invalidate the plan's assumption? That number and $1 - q^k$ decide where to invest.

Measure your checkpoints' sensitivity and false-pass rate separately, as
{{ch:ag-loop}} argued for the stopping decision and for the same reason: they cause
opposite failures and the second one is unrecoverable.

Report completion against the *pair* (segments, budget), never against either
alone. The cliff in {{eq:budget-split-coupling}} makes single-axis reporting
misleading.

Sweep task length. The decomposition benefit grows with $k$, so an evaluation on
short tasks understates the most important lever in this part.

And evaluate a segmented system against an unsegmented one at *equal budget*, not
equal steps — checkpoints consume the allowance.

## 15. Advanced Concepts

**Optimal segmentation from measured reliabilities.** Given $p$, $c$ and $k$,
{{eq:budget-split-coupling}} has an interior minimum that can be solved
numerically. Almost nobody does this, and it is a two-line computation over
numbers most teams already log.

**Non-uniform segments.** Steps are not equally reliable, so equal-length segments
are not optimal. Placing checkpoints after the *least* reliable steps concentrates
retry capacity where the failures are, and the optimisation is a simple dynamic
program. {{maturity:EMERGING}}.

**Checkpoints as a training signal.** A verified segment boundary is exactly
{{ch:rsn-supervision}}'s process label, generated for free by the execution
infrastructure. An agent that checkpoints is producing step-level supervision as a
by-product, which is the cheapest route to a process reward model this book has
described.

**Recoverability as a plan property.** {{eq:checkpoint-is-a-classifier}}'s
unrecoverable failure exists because state cannot be rolled back past an approval.
Systems with genuine rollback — transactions, snapshots, undo — change the
arithmetic completely, and characterising which agent domains admit it is
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-what-is-an-agent}}'s observation that length hurts more than accuracy is
what {{eq:checkpoints-cap-the-exponent}} exploits, and this chapter is the payoff
for that section.

{{ch:ag-react}}'s informativeness and this chapter's drift are the same quantity
measured two ways, and the two listings agree — which is worth noting because they
were built independently.

{{ch:ag-loop}}'s stopping classifier is structurally identical to the checkpoint,
including the asymmetry: the false-pass direction is unrecoverable in both.

{{ch:rsn-tool-assisted}}'s executable check is what makes
{{eq:checkpoint-is-a-classifier}}'s penalty vanish, which is why it appears here as
a requirement rather than a preference.

Ahead: {{ch:ag-memory}} takes up what a plan and its checkpoints should record;
{{ch:ag-recovery}} generalises the retry mechanism; and
{{ch:ag-termination}} uses the plan as the structure its budget and its
human gate hang from.

## 17. Exercises

1. Derive the optimal segment length from {{eq:budget-split-coupling}} for
   $p = 0.9$, checkpoint cost $1$, and $k = 12$. Compare with the measured optimum.

2. Make the replan in the first listing imperfect — a fresh plan at the same
   quality $q$ — and re-measure. How much of replanning's advantage survives?

3. Add non-uniform step reliability to the second listing and place checkpoints
   greedily after the weakest steps. How much better than equal splits?

4. Find the budget threshold below which four segments beats one, and show it
   matches {{eq:budget-split-coupling}}'s prediction.

5. Model rollback: allow the retry to restore past an approved-but-corrupt state.
   How much of {{eq:checkpoint-is-a-classifier}}'s penalty disappears?

6. Take a task your agent performs and write down its checkpoints as *state*
   assertions. How many of your plan's steps produce a checkable state?

## 18. Interview Questions

1. Your agent's plans are often wrong. Should you improve the planner or replan
   more often? What decides it?

2. Why does replanning buy nothing in a static environment?

3. A 12-step task at 90% per step succeeds 28% of the time. How do you get to 90%
   without changing the model?

4. Why can adding checkpoints make an agent complete *fewer* tasks?

5. What is the difference between a plan step and a checkpoint?

6. Why is a checkpoint that wrongly approves worse than one that wrongly rejects?

## 19. Research Questions

1. Can drift be estimated from traces without knowing what was knowable in
   advance, and does the estimate transfer across task families?

2. What is the optimal non-uniform checkpoint placement given per-step reliability
   estimates, and how much does it beat equal splits in practice?

3. Do checkpoint verdicts generated during execution make usable process labels
   ({{ch:rsn-supervision}}), and how does their bias compare with human step
   annotation?

4. Which agent domains admit genuine rollback, and how much of
   {{eq:checkpoint-is-a-classifier}}'s unrecoverable penalty does it remove?

5. Is there a plan representation in which drift invalidates only dependent steps
   rather than all subsequent ones, and what does that do to
   {{eq:disjoint-losses}}?

## 20. Chapter Summary

Planning fails in two disjoint ways: the plan is wrong, or the world moved
({{eq:disjoint-losses}}). Plan quality fixes the first and cannot touch the
second; replanning fixes the second and cannot touch the first. In
{{sec:9-practical-example}} a perfect plan still lost $60.1$ points to drift, and
a $90\%$-quality plan in a static world gained $-0.1$ points from replanning every
two steps instead of never. **They compose and neither dominates** — quality won at
the listing's parameters, which is the opposite of what the chapter was outlined to
find, and the transferable output is the arithmetic rather than the
recommendation.

The second half is where the value is. Cutting a 12-step task into six verified
segments took the same agent from $48.7\%$ to $97.5\%$, because the exponent
governing success changes from the task length to the segment length
({{eq:checkpoints-cap-the-exponent}}). That **beat improving the model from $90\%$
to $95\%$ per step**, and the advantage grows with task length — $+15.5$ points at
$k=6$, $+92.4$ at $k=32$.

Two costs. A checkpoint is a classifier, and one that approves bad work creates an
unrecoverable error because the retry restores to an approved state; the penalty
scales with the number of segments ({{eq:checkpoint-is-a-classifier}}), which is
why fine splits need executable checks. And checkpoints consume budget linearly, so
{{eq:budget-split-coupling}} has a cliff: four segments at a budget of $14$
completed **zero** tasks where one segment completed $28.4\%$.

So the justification for planning that survives the measurements is not accuracy.
**A plan whose steps are checkable is worth far more than a plan whose steps are
correct**, and the structure it supplies — verification points, restore points, and
a place for a human to intervene — is what {{ch:ag-loop}}, {{ch:ag-recovery}} and
{{ch:ag-termination}} all need and cannot build for themselves.

## 21. Further Reading

{{cite:liu2024agentbench}} for the long-horizon bottleneck, and worth re-reading
after this chapter: the finding reads differently once you notice that nothing in
those environments was checkpointing.

{{cite:yao2023react}} and {{ch:ag-react}} for the reactive end of the axis, whose
informativeness measurement agrees with this chapter's drift sweep from a
different direction.

{{cite:zhou2024webarena}} for what long-horizon tasks with real state actually
look like, which is the setting where {{eq:checkpoints-cap-the-exponent}} matters
most.

{{cite:shinn2023reflexion}} for the retry mechanism this chapter's checkpoints
enable, taken up properly in {{ch:ag-recovery}}.
