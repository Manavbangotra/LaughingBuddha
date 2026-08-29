---
id: aise-generation
number: 183
part: XXI
tier: full
status: draft
requires: [amdahl-bounds-the-stack, execution-is-not-correctness,
           free-check-before-paid-check, habituation]
provides: [acceptance-is-not-correctness, generated-code-is-under-reviewed,
           ratio-decides-acceptance, writing-is-a-small-share,
           visible-half-is-what-is-reported]
citations: [becker2025devproductivity, jimenez2023swebench, chan2024mlebench,
            testini2025dsautomation, huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why acceptance rate is not
a quality measure and what it cannot see; state why generated code is reviewed
less carefully than written code and what that costs; apply the ratio rule that
decides which suggestions to accept; compute the Amdahl bound on automating the
writing stage; and explain the mechanism behind a measured $39$-point gap between
what developers report about these tools and what measurement shows.

## 2. Why This Matters

Code is the domain with the strongest verifier in this book. Tests execute. A
compiler rejects malformed programs. {{ch:as-specialized}} found a domain's ceiling
set by whether the agent can check its own work, and by that standard software
engineering should be the success story — and largely is.

This part is about how far that gets you, and it opens with the narrowest case:
suggesting the next few lines.

Code completion is measured by **acceptance rate**, the share of suggestions a
developer keeps. {{sec:9-practical-example}} shows why that is not a quality
measure. A suggestion can be rejected, accepted and correct, or accepted and subtly
wrong — and acceptance rate cannot distinguish the last two
({{eq:acceptance-is-not-correctness}}). Across the sweep, apparent seconds-per-block
falls the whole way while total cost including later debugging *rises*.

A second effect compounds it. **Generated code is reviewed less carefully than
written code** ({{eq:generated-code-is-under-reviewed}}) — not from laziness, but
because reading is cheaper than writing and finished-looking code presents as a
completed artefact rather than a draft. Letting review depth fall from full to a
tenth cut apparent time by $20\%$ and raised true cost by $65\%$.

Which produces the chapter's most useful measurement. At high acceptance with
shallow review, the *apparent* saving is $+66\%$ and the *true* saving is $-171\%$
— **opposite signs**, with the developer having access only to the first.

That is the mechanism behind a real result.
{{cite:becker2025devproductivity}} ran a randomised trial: sixteen experienced
open-source developers, $246$ tasks, mature repositories they averaged five years
on. They forecast AI would make them $24\%$ faster, estimated afterwards that it had
made them $20\%$ faster, and were measured **$19\%$ slower**
({{eq:visible-half-is-what-is-reported}}).

## 3. Prerequisites

{{ch:aids-stack}}'s {{eq:amdahl-bounds-the-stack}} — this chapter applies it to a
coding task and finds writing to be $15\%$ of the work.

{{ch:aids-text-to-sql}}'s {{eq:execution-is-not-correctness}} and
{{eq:free-check-before-paid-check}}, whose structure recurs: the damaging failure
is the one that looks fine, and the cheap check is the one nobody runs.

{{ch:ag-termination}}'s {{eq:habituation}}, which is why review depth falls with
volume rather than staying put.

## 4. Intuitive Explanation

A developer is typing. A suggestion appears. They read it, it looks right, they
press tab.

Three things could be true. It is correct, and they saved thirty seconds. It is
obviously wrong, and they rejected it for the cost of a glance. Or it is *plausibly*
wrong — the right shape, the wrong boundary condition, a subtly different
comparison — and they have just accepted a defect that will cost fifteen minutes to
find in three weeks.

Acceptance rate counts the first and third together. It is a measure of how
persuasive suggestions are, which correlates with quality and is not the same thing.

Now the second effect, which is the one worth internalising.

Writing code requires deciding what it should do. Reading code requires deciding
whether you agree with what it does — a lower bar, reached faster, and one that a
tired developer reaches by default. And a suggestion arrives *looking finished*:
formatted, plausible, complete. A blank line demands a decision; a plausible
function demands agreement.

So review depth falls, and it falls further as suggestions become routine, which is
{{ch:ag-termination}}'s habituation in an editor. {{sec:9-practical-example}} finds
that costing more than the acceptance rate does.

Put the two together and you get a system where the felt experience and the actual
effect point in different directions. The saving is immediate, concentrated and
visible: you watched the tool write the function. The cost is delayed, diffuse and
attributed to something else — the forty-minute debugging session does not announce
that twelve of its minutes belong to a suggestion accepted an hour ago.

Then the second listing asks where the time in a coding task actually goes, and the
answer is {{ch:aids-stack}}'s answer for data science, transposed.

Writing the code is about $15\%$ of the task. Understanding the issue, locating
where the change belongs, getting it actually working, and landing it are the rest.
So making writing *free* — not fast, free — produces about a $1.17\times$ speedup.

That is the whole Amdahl argument and it is unkind to the stage everyone means by
"AI coding". It is also not the interesting half.

The interesting half is that the stages the tool does not speed up are stages it can
*slow down*. Reviewing generated code takes longer than reviewing your own. Debugging
code you did not write takes longer than debugging code you did. Integrating a change
whose reasoning you did not follow takes longer.

{{sec:9-practical-example}} finds the break-even: the tool pays off if it adds less
than about $26\%$ to review, debugging and integration, and costs time if it adds
more.

Which is why the effect changes sign across settings. An expert on a mature codebase
they know has fast writing already and enormous constraint knowledge the tool lacks —
the least favourable cell. A novice, or anyone on unfamiliar code or greenfield work,
is in a much better one. **The randomised trial studied the hardest cell**, which is
what makes its result important and what stops it generalising to every setting.

## 5. Formal Explanation

**Acceptance and correctness.** Let a suggestion be correct with probability $p$,
accepted with probability $a$, and reviewed with depth $d \in [0,1]$ where a full
review catches a fraction $c_0 d$ of incorrect accepted suggestions. Per block, with
$T$ the time to write it by hand, $g$ the glance cost, $r$ the full review cost, and
$D$ the cost of an escaped defect:

$$\mathbb{E}[\text{apparent}] = g + (1-a)T + a\,r d, \qquad \mathbb{E}[\text{true}] = \mathbb{E}[\text{apparent}] + a(1-p)(1 - c_0 d)\,D$$ (eq:acceptance-is-not-correctness)

The apparent term is decreasing in $a$ and in $-d$; the escape term is increasing in
both. **The two components move in opposite directions in both control variables**,
which is why the felt and measured quantities diverge.

**Review depth.** Differentiating with respect to $d$:

$$\frac{\partial}{\partial d}\mathbb{E}[\text{apparent}] = ar > 0, \qquad \frac{\partial}{\partial d}\mathbb{E}[\text{true}] = ar - a(1-p)c_0 D$$ (eq:generated-code-is-under-reviewed)

which is negative whenever $D > r/((1-p)c_0)$ — that is, whenever an escaped defect
costs more than a review divided by the chance the review would have caught it. For
realistic values that inequality holds comfortably, so **deeper review is always
worth it and always feels worse.**

**The ratio rule.** The optimal acceptance rate depends on $T$ and $D$ only through
their ratio. Accepting is worth it when the expected saving exceeds the expected
escape cost:

$$T - rd > (1-p)(1 - c_0 d)\,D \quad\Longleftrightarrow\quad \frac{D}{T} < \frac{1 - rd/T}{(1-p)(1-c_0 d)}$$ (eq:ratio-decides-acceptance)

**Accept when the defect-to-saving ratio is small**, which means accepting long
mechanical blocks and rejecting short subtle ones — because the numerator scales
with what the suggestion saves and the denominator does not.

**Amdahl on the task.** With stage times $t_i$ and multipliers $\mu_i$:

$$S = \frac{\sum_i t_i}{\sum_i t_i \mu_i}$$ (eq:writing-is-a-small-share)

Setting $\mu_{\text{write}} = 0$ and all others to $1$ bounds the speedup at
$1/(1 - t_{\text{write}}/\sum t_i)$. At a $15\%$ share that is $1.17$.

**The reported quantity.** Partition stages into visible $V$ and invisible $I$. A
developer's self-report tracks $V$:

$$\hat{S} = \frac{\sum_{i \in V} t_i}{\sum_{i \in V} t_i\mu_i}, \qquad S = \frac{\sum_i t_i}{\sum_i t_i \mu_i}$$ (eq:visible-half-is-what-is-reported)

and since $\mu_i < 1$ on $V$ and $\mu_i > 1$ on $I$ by construction, $\hat{S} > S$
always. **The self-report is biased upward by an amount equal to the friction it
cannot observe** — and it is a faithful report of the half it can.

The break-even friction $\phi^*$ solves
$\sum_V t_i \mu_i + \phi^* \sum_I t_i = \sum_i t_i$, giving:

$$\phi^* = 1 + \frac{\sum_V t_i (1 - \mu_i)}{\sum_I t_i}$$

## 6. Mathematical Foundation

Three extractions.

**Both control variables have opposite signs on the two objectives.** From
{{eq:acceptance-is-not-correctness}}, raising acceptance and lowering review depth
each improve the felt quantity and worsen the true one. A system whose operator
optimises what they feel will therefore move to a corner that is bad on both counts,
and will experience the trip as improvement.

**The optimum depends on a ratio, not a rate.**
{{eq:ratio-decides-acceptance}} contains $D/T$, so "what acceptance rate should I
have" has no answer — it depends on the size of the suggestion and the cost of a
defect in this codebase, both of which vary within a single working day.

**The self-report bias is signed, not noisy.**
{{eq:visible-half-is-what-is-reported}} gives $\hat S > S$ unconditionally under the
stated sign pattern. That is stronger than "self-reports are unreliable": it says
they are wrong in a known direction, so a survey of developers cannot be used as
evidence about the effect even in aggregate.

## 7. Internal Mechanics

### 7.1 The three outcomes of a suggestion

```mermaid {#fig:suggestion-outcomes caption="What acceptance rate can and cannot see. The two accepted branches are indistinguishable at the moment of acceptance, which is the moment the metric is recorded."}
flowchart TD
    S[suggestion appears] --> R{does it look right?}
    R -->|no| REJ["rejected<br/>cost: a glance"]
    R -->|yes| ACC[accepted]
    ACC --> C{is it right?}
    C -->|yes| GOOD["saved the typing"]
    C -->|no| BAD["defect, discovered later<br/>cost: a debugging session"]
    ACC -.->|"acceptance rate<br/>counts both"| M[(metric)]
```

The dotted edge is the problem. The metric is recorded at the moment the two
branches are indistinguishable, and nothing later updates it.

### 7.2 What makes a suggestion safe to accept

{{eq:ratio-decides-acceptance}} gives the principle; four practical proxies follow
from it.

**Length.** A long suggestion saves more, so it tolerates a higher defect
probability. This runs against the instinct to trust short suggestions because they
are easier to read.

**Mechanicalness.** Boilerplate, a repeated pattern, an interface implementation —
the correctness is checkable by inspection rather than by reasoning, so review depth
is cheap and defects are visible.

**Whether a test covers it.** {{ch:as-specialized}}'s verifier argument at the level
of a block: a suggestion in code with good test coverage has an independent check,
so an escaped defect is caught by something other than a debugging session.

**Whether the logic is load-bearing.** A boundary condition, a comparison operator,
an error path — these are where plausible-wrong lives, and they are short, which
compounds with the first proxy.

The composite rule: **accept long mechanical suggestions in tested code; type the
boundary conditions yourself.**

### 7.3 Why reading feels like reviewing and is not

The mechanism behind {{eq:generated-code-is-under-reviewed}} deserves stating
precisely, because "review generated code carefully" is advice everyone accepts and
nobody follows.

Writing code forces a decision per token: you cannot produce the comparison without
choosing its direction. Reading code allows a decision per *block*: you form an
impression of what it does and check the impression against your intent. The second
is faster and usually correct, and it fails exactly on the cases where the code does
something *close to* what you expected.

Which are precisely the cases a language model produces. A model's errors are not
random — they are plausible, because plausibility is what it optimises. So the
failure mode of shallow review and the failure mode of generation are the same
failure mode, viewed from two sides.

This is {{ch:as-failures}}'s correlation result in a small setting: **the reviewer's
blind spot and the generator's error distribution coincide**, and the reviewer here
is a tired human rather than a second model.

### 7.4 The cheap check nobody runs

{{ch:aids-text-to-sql}} found a free rung on the check ladder — the query had
already run, and nobody looked at whether the result was empty. Code completion has
one too.

**Run the tests before accepting the next suggestion.** Not at commit time; at
acceptance time. The cost is whatever the test suite costs, which for a well-factored
project is seconds, and it converts an escaped defect into an announced one at the
moment the context for fixing it is still in the developer's head.

The reason this is not standard is latency: a suggestion arrives in a second and a
test suite takes twenty. But the comparison is not against the suggestion's latency —
it is against the fifteen minutes an escaped defect costs, and
{{sec:9-practical-example}}'s ratio table says which side that lands on.

For projects where the full suite is slow, the same argument applies to whatever
subset is fast.

### 7.5 Where the friction actually comes from

{{eq:visible-half-is-what-is-reported}}'s invisible stages worsen for three
identifiable reasons, and separating them matters because they have different fixes.

**Comprehension debt.** Code you did not write is code whose reasoning you do not
hold. Debugging it later means reconstructing that reasoning, which is work the
author did not do at write time. *Fix: read the suggestion as though you will have to
debug it, because you will.*

**Constraint violation.** A mature codebase has invariants the tool does not know —
this module must not import that one, this function must stay allocation-free, this
error must propagate rather than be logged. A suggestion that violates one is
plausible and wrong in a way only a maintainer sees. *Fix: this is what
{{ch:aise-repo}} is about, and it is the strongest argument for repository context.*

**Scope creep.** A suggestion frequently does more than asked — handles a case that
was not in scope, adds a parameter, generalises. Accepting it enlarges the change,
and the enlargement is reviewed by nobody because it was not the point of the change.
*Fix: reject suggestions that exceed the scope you intended, even when they are
correct.*

The third is the least discussed and the most avoidable.

### 7.6 Reading the productivity trial honestly

{{cite:becker2025devproductivity}} is the most rigorous measurement available and it
is easy to over-read in both directions.

**What it establishes.** With randomisation, on real tasks, in mature repositories:
a $19\%$ slowdown, robust across the twenty explanatory factors the authors examined.
And a $39$-point gap between self-estimate and measurement, which is the finding that
generalises furthest, because it means practitioner testimony is not evidence.

**What it does not establish.** That the effect holds for other developers, other
codebases, or other tool configurations. Sixteen developers is a small sample;
experienced maintainers on repositories they have worked on for five years is one
cell of {{sec:9-practical-example}}'s last table, and the model says it is the cell
with the least assistance available and the most friction to add.

**What to do with it.** Not "these tools do not work" and not "this study is
unrepresentative". The correct inference is that **the effect is setting-dependent
with a sign change inside the plausible range**, which means it has to be measured
per team rather than assumed in either direction — and that the measurement cannot be
a survey.

### 7.7 Why this part is not the pessimistic one

Two chapters of this book have now measured an automation and found the total
effect smaller or worse than the felt effect, and it would be easy to read a pattern
into that which is not there.

The difference in software engineering is that **the verifier is real**. Tests
execute. A compiler rejects malformed programs. A type checker rejects a whole class
of the plausible-wrong errors this chapter is about, mechanically and for free.
{{ch:as-specialized}} found a domain's ceiling set by exactly this, and software has
the strongest verifier of any domain in the book.

That changes the shape of every recommendation. In {{ch:aids-agentic-eda}}, where
exploration had no verifier, the advice was structural and defensive — hold out data,
count the comparisons, do not let unverified output circulate. Here the advice can be
constructive, because **the escape cost that dominates
{{eq:ratio-decides-acceptance}} is a variable a team controls.**

Improve test coverage and $D$ falls, because a defect is caught by the suite rather
than by a debugging session three weeks later. Adopt stricter typing and $(1-p)$
falls for a whole error class. Shorten the feedback loop and the friction on the
invisible stages falls directly.

Each of those moves the ratio, and moving the ratio is what licenses higher
acceptance. So the sequence is the opposite of how these tools are usually adopted:
**the verification investment comes first and the acceptance rate follows from it**,
rather than acceptance rising and verification being asked to catch up.

Which also explains a pattern practitioners report and this chapter's model predicts:
teams with strong test suites and fast pipelines describe these tools far more
favourably than teams without. They are not more optimistic. They are operating at a
different $D/T$, and the same rule gives them a different answer.

## 8. Implementation

Two listings. The first prices acceptance and review depth. The second decomposes a
coding task and reproduces the self-report gap.

```python {tier=A name=acceptance-is-not-correctness}
"""Accepting a suggestion is not the same as having working code.

Code completion is measured by acceptance rate: what share of suggestions a
developer keeps. That is a real number and it is not a quality measure, for the
reason ch:aids-text-to-sql found about queries -- the damaging outcome is the one
that looks fine.

A suggestion can be rejected (costs a glance), accepted and correct (saves the
typing), or accepted and subtly wrong (saves the typing and costs a debugging
session later). The third case is the one acceptance rate cannot see
(eq:acceptance-is-not-correctness).

There is a second effect that makes it worse and that this listing measures
separately: generated code is REVIEWED LESS CAREFULLY than written code, because
reading is cheaper than writing and finished-looking code invites less scrutiny
(eq:generated-code-is-under-reviewed).
"""
import numpy as np

rng = np.random.default_rng(4967)

M = 60000
TYPE_TIME = 42.0        # seconds to write the block by hand
GLANCE = 4.0            # seconds to read and reject a suggestion
REVIEW_FULL = 18.0      # seconds to review a suggestion properly
DEBUG_COST = 900.0      # seconds to find and fix a subtle defect later
P_SUGGEST_OK = 0.72     # a suggestion is correct this often


def run(n_blocks=1, m=M, accept_rate=0.30, review_depth=1.0,
        p_ok=P_SUGGEST_OK, catch_full=0.80, type_time=TYPE_TIME,
        review_full=None):
    """One block of code. `review_depth` scales how thoroughly an accepted
    suggestion is reviewed; 1.0 means as carefully as hand-written code.

    Returns (seconds spent, defects escaped per block).
    """
    offered = np.ones(m, dtype=bool)
    correct = rng.random(m) < p_ok
    # A developer accepts more often when the suggestion looks right, but
    # cannot fully distinguish correct from plausible-wrong.
    look_right = correct | (rng.random(m) < 0.55)
    accepted = look_right & (rng.random(m) < accept_rate / 0.72)

    # Reviewing scales with the size of the suggestion, not with a constant.
    rev = REVIEW_FULL * (type_time / TYPE_TIME) if review_full is None         else review_full
    time = np.zeros(m)
    time[~accepted] = GLANCE + type_time
    time[accepted] = GLANCE + rev * review_depth

    # A wrong-but-accepted suggestion is caught in review with a probability
    # proportional to review depth.
    bad = accepted & ~correct
    caught = bad & (rng.random(m) < catch_full * review_depth)
    time[caught] += type_time          # fix it by hand after catching it
    escaped = bad & ~caught
    time[escaped] += DEBUG_COST * 0.0  # the debug cost lands later, counted apart

    return float(time.mean()), float(escaped.mean()), float(accepted.mean())


print(f"One block of code: {TYPE_TIME:.0f}s to write by hand. A suggestion costs")
print(f"{GLANCE:.0f}s to glance at, {REVIEW_FULL:.0f}s to review properly, and is")
print(f"correct {P_SUGGEST_OK:.0%} of the time. A subtle defect that escapes costs")
print(f"{DEBUG_COST:.0f}s later.")
print()
print(f"{'acceptance rate':>17}{'seconds/block':>15}{'defects escaped':>17}"
      f"{'true cost':>12}")
print("-" * 61)
tab = {}
for a in (0.0, 0.15, 0.30, 0.50, 0.70):
    t, e, acc = run(accept_rate=a)
    tab[a] = (t, e, t + e * DEBUG_COST)
    print(f"{a:>17.0%}{t:>15.1f}{e:>17.3f}{t + e * DEBUG_COST:>12.1f}")

print()
print()
print("The same, with review depth falling as suggestions become routine --")
print("which is what happens, because finished-looking code invites less")
print("scrutiny than a blank line does.")
print()
print(f"{'review depth':>14}{'seconds/block':>15}{'defects escaped':>17}"
      f"{'true cost':>12}")
print("-" * 58)
rd = {}
for d in (1.0, 0.7, 0.45, 0.25, 0.1):
    t, e, acc = run(accept_rate=0.30, review_depth=d)
    rd[d] = (t, e, t + e * DEBUG_COST)
    print(f"{d:>14.0%}{t:>15.1f}{e:>17.3f}{t + e * DEBUG_COST:>12.1f}")

print()
print()
print("Apparent speed against true cost. The apparent saving is what a")
print("developer experiences; the true cost includes the debugging.")
print()
base = run(accept_rate=0.0)[0]
print(f"{'acceptance':>12}{'review depth':>14}{'apparent saving':>17}"
      f"{'true saving':>14}")
print("-" * 57)
ap = {}
for a, d in ((0.30, 1.0), (0.30, 0.45), (0.55, 0.45), (0.70, 0.25)):
    t, e, _ = run(accept_rate=a, review_depth=d)
    true = base - (t + e * DEBUG_COST)
    ap[(a, d)] = (base - t, true)
    print(f"{a:>12.0%}{d:>14.0%}{(base - t) / base:>17.1%}"
          f"{true / base:>14.1%}")

print()
print()
print("The break-even: how good a suggestion must be for shallow review to be")
print("worth it, at a 30% acceptance rate.")
print()
print(f"{'suggestion correct':>20}{'deep review':>14}{'shallow review':>17}"
      f"{'better':>10}")
print("-" * 61)
be = {}
for p in (0.55, 0.72, 0.85, 0.95, 0.99):
    deep = run(accept_rate=0.30, review_depth=1.0, p_ok=p)
    shal = run(accept_rate=0.30, review_depth=0.35, p_ok=p)
    dc = deep[0] + deep[1] * DEBUG_COST
    sc = shal[0] + shal[1] * DEBUG_COST
    be[p] = (dc, sc)
    print(f"{p:>20.0%}{dc:>14.1f}{sc:>17.1f}"
          f"{('shallow' if sc < dc else 'deep'):>10}")

print()
print()
print("And the variable that actually decides it: how much typing a")
print("suggestion saves, against what an escaped defect costs. The ratio is")
print("what matters, so this sweeps the size of the suggestion.")
print()
print(f"{'a suggestion saves':>20}{'defect/save ratio':>19}"
      f"{'best acceptance':>17}{'true cost':>12}")
print("-" * 68)
sz = {}
for tt in (12.0, 42.0, 150.0, 600.0):
    best, bv = None, 1e18
    for a in (0.0, 0.15, 0.30, 0.5, 0.7, 0.9):
        for d in (0.25, 0.45, 0.7, 1.0):
            t, e, _ = run(accept_rate=a, review_depth=d, type_time=tt)
            v = t + e * DEBUG_COST
            if v < bv:
                best, bv = (a, d), v
    sz[tt] = (best, bv)
    print(f"{tt:>20.0f}{DEBUG_COST / tt:>19.0f}{best[0]:>17.0%}{bv:>12.1f}")

print()
print()
print("At each size, the best acceptance rate and the review depth that goes")
print("with it.")
print()
print(f"{'a suggestion saves':>20}{'best acceptance':>17}{'best depth':>13}")
print("-" * 50)
for tt in (12.0, 42.0, 150.0, 600.0):
    (a, d), _ = sz[tt]
    print(f"{tt:>20.0f}{a:>17.0%}{d:>13.0%}")

print(f"""
The first table moves the wrong way. True cost rises from {tab[0.0][2]:.1f}
seconds per block at zero acceptance to {tab[0.7][2]:.1f} at
{0.7:.0%} -- **accepting more suggestions costs more in total**, because the
typing saved is smaller than the debugging added
(eq:acceptance-is-not-correctness).

The seconds-per-block column falls the whole way, from {tab[0.0][0]:.1f} to
{tab[0.7][0]:.1f}. That column is what a developer feels. The true-cost column is
what happens.

The second table adds the effect that makes it worse and that nobody meters.
Holding acceptance at {0.30:.0%} and letting review depth fall from
{1.0:.0%} to {0.1:.0%}, apparent time falls from {rd[1.0][0]:.1f} to
{rd[0.1][0]:.1f} seconds and true cost rises from {rd[1.0][2]:.1f} to
{rd[0.1][2]:.1f}.

**Generated code gets reviewed less carefully than written code**
(eq:generated-code-is-under-reviewed), and the reason is not laziness. Reading is
cheaper than writing, so the same amount of attention goes further and feels like
more; and finished-looking code presents as a completed artefact rather than as a
draft. A blank line demands a decision. A plausible function demands agreement.

The third table is the one to carry, because it is the mechanism behind a real
measurement. At {0.70:.0%} acceptance with {0.25:.0%} review depth, the apparent
saving is {ap[(0.70, 0.25)][0] / base:+.1%} and the true saving is
{ap[(0.70, 0.25)][1] / base:+.1%}.

**The two numbers have opposite signs**, and the developer only has access to the
first one. cite:becker2025devproductivity found experienced developers estimating
they had been made 20% faster while measurement showed them 19% slower -- a
39-point error, by people who had just done the work. This table is what that looks
like block by block.

The break-even table says when shallow review is defensible: at
{0.99:.0%} suggestion correctness and not before. At {0.95:.0%} -- which would be a
remarkable model -- deep review still wins.

And the last two tables give the rule that is actually usable, because acceptance
rate is the wrong thing to tune. What decides it is the RATIO of what a suggestion
saves to what a defect costs.

At a ratio of {DEBUG_COST / 12.0:.0f} -- a small suggestion, an expensive
codebase -- the best acceptance rate is {sz[12.0][0][0]:.0%}. At a ratio of
{DEBUG_COST / 600.0:.0f} -- a large mechanical block -- it is
{sz[600.0][0][0]:.0%}.

**Accept long mechanical suggestions and be sceptical of short subtle ones.** The
saving scales with the size of the suggestion and the risk is roughly constant per
acceptance, so the ratio is what matters and the crossover in this table sits
around a defect costing ten times what the suggestion saves.

That is a rule a developer can apply in the moment, which "your acceptance rate
should be 30%" is not.""")
```

The second listing asks where a coding task's time goes.

```python {tier=A name=writing-is-a-small-share}
"""Where a coding task's time goes, and why speeding up typing does so little.

cite:becker2025devproductivity ran a randomised controlled trial: 16 experienced
open-source developers, 246 tasks, mature repositories they averaged five years
on. Developers forecast AI would make them 24% faster. Afterwards they estimated
it had made them 20% faster. It made them 19% SLOWER.

That is three numbers and the interesting one is the gap between the second and
the third: a 39-point error in self-assessment, by people who had just done the
work.

This listing decomposes a coding task and asks which arrangement of effects
reproduces that pattern (eq:writing-is-a-small-share). The structure is
ch:aids-stack's: a stage that gets automated is a stage that was small, and the
automation adds work elsewhere.
"""
import numpy as np

# Minutes per task, by stage, for an experienced developer on a familiar
# codebase. These are this listing's assumptions, stated so they can be checked.
# (stage, baseline minutes, AI multiplier, is the change VISIBLE to the developer)
STAGES = [
    ("understand the issue",   18.0, 0.95, True),
    ("locate the change",      22.0, 0.80, True),
    ("write the code",         16.0, 0.45, True),
    ("review what was written", 9.0, 1.60, False),
    ("get it actually working", 31.0, 1.35, False),
    ("integrate and land it",  14.0, 1.05, False),
]

BASE = sum(s[1] for s in STAGES)

print("An experienced developer's task on a codebase they know, by stage.")
print()
print(f"{'stage':>26}{'minutes':>10}{'share':>8}{'with AI':>10}{'change':>10}"
      f"{'visible?':>11}")
print("-" * 75)
for name, base, mult, vis in STAGES:
    print(f"{name:>26}{base:>10.0f}{base / BASE:>8.0%}{base * mult:>10.1f}"
          f"{base * (mult - 1):>+10.1f}{('yes' if vis else 'no'):>11}")

total_ai = sum(s[1] * s[2] for s in STAGES)
print("-" * 75)
print(f"{'total':>26}{BASE:>10.0f}{1.0:>8.0%}{total_ai:>10.1f}"
      f"{total_ai - BASE:>+10.1f}")
print()
print(f"   measured effect: {total_ai / BASE - 1:+.0%} on task time")

print()
print()
print("What the developer EXPERIENCES: only the visible stages, which are the")
print("ones where the tool is doing something in front of them.")
print()
vis_base = sum(s[1] for s in STAGES if s[3])
vis_ai = sum(s[1] * s[2] for s in STAGES if s[3])
inv_base = sum(s[1] for s in STAGES if not s[3])
inv_ai = sum(s[1] * s[2] for s in STAGES if not s[3])
print(f"{'':>26}{'baseline':>11}{'with AI':>10}{'change':>10}")
print("-" * 57)
print(f"{'visible stages':>26}{vis_base:>11.0f}{vis_ai:>10.1f}"
      f"{vis_ai / vis_base - 1:>+10.0%}")
print(f"{'invisible stages':>26}{inv_base:>11.0f}{inv_ai:>10.1f}"
      f"{inv_ai / inv_base - 1:>+10.0%}")
print(f"{'all stages':>26}{BASE:>11.0f}{total_ai:>10.1f}"
      f"{total_ai / BASE - 1:>+10.0%}")
print()
print(f"   Self-estimate, if only visible stages register: "
      f"{vis_ai / vis_base - 1:+.0%}")
print(f"   Measured:                                       "
      f"{total_ai / BASE - 1:+.0%}")
print(f"   Gap:                                            "
      f"{abs(vis_ai / vis_base - total_ai / BASE):.0%} points")

print()
print()
print("Amdahl on the stage everyone means by 'AI coding'. Perfect automation")
print("of writing, and nothing else changed:")
print()
write = next(s for s in STAGES if s[0] == "write the code")
print(f"{'writing time':>26}{'total':>10}{'speedup':>10}")
print("-" * 46)
for mult, label in ((1.0, "unchanged"), (0.45, "as measured"),
                    (0.10, "near-perfect"), (0.0, "free")):
    t = BASE - write[1] * (1 - mult)
    print(f"{label:>26}{t:>10.1f}{BASE / t:>10.2f}x")

print()
print()
print("The two effects separated. 'Assistance' is the speedup on the stages AI")
print("helps; 'friction' is the slowdown on the ones it does not.")
print()
print(f"{'friction multiplier':>21}{'total':>10}{'effect':>10}{'verdict':>12}")
print("-" * 53)
fr = {}
for f in (1.0, 1.1, 1.25, 1.35, 1.6):
    t = sum(s[1] * (s[2] if s[3] else 1.0 + (s[2] - 1.0) * (f - 1.0) / 0.35)
            for s in STAGES)
    t = sum(s[1] * s[2] if s[3] else s[1] * f for s in STAGES)
    fr[f] = (t, t / BASE - 1)
    print(f"{f:>21.2f}{t:>10.1f}{t / BASE - 1:>+10.0%}"
          f"{('faster' if t < BASE else 'slower'):>12}")

print()
print()
print("And where the effect flips. The break-even friction, and what it implies")
print("about which developers and codebases benefit.")
print()
lo, hi = 1.0, 2.0
for _ in range(60):
    mid = (lo + hi) / 2
    t = sum(s[1] * s[2] if s[3] else s[1] * mid for s in STAGES)
    if t < BASE:
        lo = mid
    else:
        hi = mid
breakeven = (lo + hi) / 2
print(f"   break-even friction multiplier: {breakeven:.2f}")
print(f"   i.e. the tool pays off if it adds less than "
      f"{(breakeven - 1) * 100:.0f}% to review, debugging and integration.")
print()
print(f"{'setting':>34}{'assistance':>12}{'friction':>10}{'effect':>10}")
print("-" * 66)
SETTINGS = [
    ("expert, mature familiar codebase", 0.45, 1.35),
    ("expert, unfamiliar codebase", 0.45, 1.10),
    ("novice, any codebase", 0.35, 1.05),
    ("greenfield, few constraints", 0.25, 1.00),
]
st = {}
for label, wmult, f in SETTINGS:
    t = 0.0
    for name, base, mult, vis in STAGES:
        if name == "write the code":
            t += base * wmult
        elif vis:
            t += base * mult
        else:
            t += base * f
    st[label] = t / BASE - 1
    print(f"{label:>34}{wmult:>12.2f}{f:>10.2f}{t / BASE - 1:>+10.0%}")

print(f"""
The share column is the first finding and it is ch:aids-stack's exactly. **Writing
the code is {16 / BASE:.0%} of the task.** Understanding, locating, debugging and
integrating are the other {1 - 16 / BASE:.0%}.

So the Amdahl table is unsurprising once the share is known: making writing FREE
gives a {BASE / 94.0:.2f}x speedup on the task. Not free -- free. The stage that
"AI coding" means is the stage that was already small.

The second table is the mechanism this listing was built for.

Visible stages -- the ones where the tool is doing something on screen -- improve
by {vis_ai / vis_base - 1:.0%}. Invisible stages -- reviewing, getting it actually
working, landing it -- worsen by {inv_ai / inv_base - 1:+.0%}. The total is
{total_ai / BASE - 1:+.0%}.

**A developer who registers the visible stages would report being
{abs(vis_ai / vis_base - 1):.0%} faster while being {total_ai / BASE - 1:+.0%}
slower**, a gap of {abs(vis_ai / vis_base - total_ai / BASE) * 100:.0f} points.

cite:becker2025devproductivity measured that gap at 39 points -- self-estimate
-20%, measurement +19% -- and this decomposition reproduces its shape from
plausible per-stage assumptions. Note the honest discrepancy: this listing's
parameters produce {total_ai / BASE - 1:+.0%} where the trial measured +19%, which
means the real friction on the invisible stages was **larger** than assumed here,
not smaller.

The reason the gap exists at all is that the two effects land in different places.
The assistance is concentrated, immediate and observed. The friction is diffuse,
delayed, and indistinguishable from the ordinary difficulty of software. A
developer debugging for forty minutes does not experience twelve of those minutes
as attributable to the suggestion they accepted an hour ago.

**The self-report is not dishonest. It is a faithful report of the visible half.**

The friction table locates the boundary. The break-even multiplier is
{breakeven:.2f}: **the tool pays off if it adds less than
{(breakeven - 1) * 100:.0f}% to review, debugging and integration** and costs time
if it adds more.

Which makes the last table the important one, because it says who is on which side.

An expert on a mature codebase they know well: {st['expert, mature familiar codebase']:+.0%}.
The same expert on unfamiliar code: {st['expert, unfamiliar codebase']:+.0%}. A
novice: {st['novice, any codebase']:+.0%}. Greenfield work with few constraints:
{st['greenfield, few constraints']:+.0%}.

**The effect changes sign across settings**, and it does so for a legible reason:
assistance is worth most where the developer's own writing speed is the constraint,
and friction is worst where the code has many constraints the developer knows and
the tool does not.

cite:becker2025devproductivity studied experienced open-source developers on mature
repositories they averaged five years on -- **the least favourable cell in this
table**, and also the cell where the strongest claims are usually made. That is not
a criticism of the study; it is the reason its result is important and the reason
it does not generalise to every setting.

The practical readings are three.

**Do not infer the effect from how it feels.** The felt quantity and the measured
quantity have different signs in some settings and nobody can tell from inside.

**Expect the benefit where writing is the constraint** -- unfamiliar APIs,
boilerplate, greenfield, novices -- and expect friction where the constraint is
knowing what the code must not break.

**And measure it on your own work**, because the parameters that decide this are
per-team and the sign flips inside the plausible range.""")
```

## 9. Practical Example

The first listing prices one block of code — $42$ seconds to write by hand,
suggestions correct $72\%$ of the time, an escaped defect costing $900$ seconds:

```
  acceptance rate  seconds/block  defects escaped   true cost
-------------------------------------------------------------
               0%           46.0            0.000        46.0
              30%           39.4            0.013        51.0
              70%           30.6            0.030        57.5
```

**Apparent time falls the whole way and true cost rises**
({{eq:acceptance-is-not-correctness}}). Review depth compounds it:

```
  review depth  seconds/block  defects escaped   true cost
----------------------------------------------------------
          100%           39.4            0.013        51.2
           45%           34.7            0.041        71.2
           10%           31.5            0.059        84.7
```

**Generated code gets reviewed less carefully than written code**
({{eq:generated-code-is-under-reviewed}}) — a blank line demands a decision, a
plausible function demands agreement.

The two together:

```
  acceptance  review depth  apparent saving   true saving
---------------------------------------------------------
         30%          100%            14.6%        -11.3%
         55%           45%            45.4%       -105.0%
         70%           25%            66.4%       -170.7%
```

**Opposite signs**, with the developer having access only to the first column.

And the rule that is actually usable, since acceptance rate is the wrong dial:

```
  a suggestion saves  defect/save ratio  best acceptance   true cost
--------------------------------------------------------------------
                  12                 75               0%        16.0
                  42                 21               0%        46.0
                 150                  6              90%       124.8
                 600                  2              90%       263.3
```

**Accept long mechanical suggestions and be sceptical of short subtle ones**
({{eq:ratio-decides-acceptance}}) — the saving scales with the suggestion's size and
the risk per acceptance does not.

The second listing decomposes a task for an experienced developer on familiar code:

```
                     stage   minutes   share   with AI    change   visible?
---------------------------------------------------------------------------
      understand the issue        18     16%      17.1      -0.9        yes
         locate the change        22     20%      17.6      -4.4        yes
            write the code        16     15%       7.2      -8.8        yes
   review what was written         9      8%      14.4      +5.4         no
   get it actually working        31     28%      41.9     +10.9         no
     integrate and land it        14     13%      14.7      +0.7         no
```

**Writing is $15\%$ of the task**, so:

```
              writing time     total   speedup
----------------------------------------------
               as measured     101.2      1.09x
                      free      94.0      1.17x
```

Making writing *free* gives $1.17\times$ ({{eq:writing-is-a-small-share}}).

And the mechanism:

```
                             baseline   with AI    change
---------------------------------------------------------
            visible stages         56      41.9      -25%
          invisible stages         54      71.0      +31%
                all stages        110     112.9       +3%
```

**A developer registering the visible stages reports $25\%$ faster while being
$3\%$ slower** ({{eq:visible-half-is-what-is-reported}}) — a $28$-point gap from
plausible assumptions. {{cite:becker2025devproductivity}} measured $39$ points, which
means the real friction was *larger* than assumed here.

The break-even friction is $1.26$: the tool pays off if it adds less than $26\%$ to
review, debugging and integration. Which decides who benefits:

```
                           setting  assistance  friction    effect
------------------------------------------------------------------
  expert, mature familiar codebase        0.45      1.35       +4%
       expert, unfamiliar codebase        0.45      1.10       -8%
              novice, any codebase        0.35      1.05      -12%
       greenfield, few constraints        0.25      1.00      -16%
```

**The effect changes sign across settings**, and the randomised trial studied the
least favourable cell — which is what makes its result important and what stops it
generalising.

## 10. Production Considerations

Do not report acceptance rate as a quality metric. It is recorded at the moment the
good and bad branches are indistinguishable.

Accept long mechanical suggestions; type the boundary conditions, comparisons and
error paths yourself.

Run the fast tests at acceptance time, not at commit time. It converts an escaped
defect into an announced one while the context is still loaded.

Reject suggestions that exceed the scope you intended, even correct ones — the
excess is reviewed by nobody.

Read a suggestion as though you will debug it later, because you will.

Expect benefit where writing is the constraint and friction where knowing what not to
break is the constraint.

Measure the effect on your own team, on real tasks, with a control. And do not
measure it with a survey — the bias is signed.

## 11. Common Mistakes

**Treating acceptance rate as quality.** It counts correct and plausible-wrong
together.

**Tuning an acceptance rate.** The optimum is a ratio, and it varies within a
working day.

**Trusting short suggestions because they are easy to read.** Short suggestions save
least and carry the subtle logic.

**Letting review depth drift.** It is the larger of the two effects and the less
visible.

**Accepting correct suggestions that do more than asked.** The enlargement escapes
review entirely.

**Surveying developers about the effect.** The bias has a known sign.

**Generalising the trial's slowdown to every setting**, or dismissing it because it
does not.

## 12. Failure Modes

*Plausible-wrong acceptance.* The characteristic failure: right shape, wrong
boundary, found three weeks later.

*Review erosion.* Depth falling with volume, invisibly, as suggestions become
routine.

*Comprehension debt.* Code in the repository that nobody, including its committer,
has reasoned about.

*Silent scope growth.* Changes larger than intended because a suggestion offered
more.

*Constraint violation.* A plausible change that breaks an invariant only a
maintainer knows about.

*Confident testimony.* A team reporting large gains, sincerely, from the visible
half.

## 13. Alternatives

**Suggestion at a coarser grain.** Whole functions or files rather than lines, which
raises the saving per acceptance and moves the ratio favourably — {{ch:aise-swe-agents}}'s
regime.

**Test-first generation.** Write the test, generate against it, and let the verifier
grade the suggestion — the strongest available structure and
{{ch:aise-testing}}'s subject.

**Explanation-required acceptance.** Requiring the developer to state what the
suggestion does before accepting, which forces read depth up at a real cost in flow.

**Suggestions as reference rather than insertion.** Show the code; make the developer
type it. Slower, and it eliminates comprehension debt entirely.

**Restricting completion to tested code paths.** Where a verifier exists, the escape
cost falls by an order of magnitude and the ratio rule permits much higher
acceptance.

## 14. Evaluation

Measure defect escape rate attributable to accepted suggestions, by tagging
suggestion-derived lines and tracking subsequent fixes to them. It is the number
acceptance rate stands in for and it is recoverable from version control.

Measure review depth directly — time between suggestion appearing and acceptance —
and watch it over weeks. The trend is the diagnostic.

Measure task completion time with randomisation, not with a survey.

Report the setting alongside any productivity claim: developer experience, codebase
maturity, familiarity. The sign depends on it.

And measure your own defect cost. It is the denominator in the only rule this
chapter offers.

## 15. Advanced Concepts

**Acceptance-time verification.** Running the relevant tests in the editor before
acceptance, at suggestion latency. The engineering is a test-selection problem and
the payoff follows directly from {{eq:ratio-decides-acceptance}}.
{{maturity:EMERGING}}.

**Suggestion-attributed defect tracking.** Tagging generated lines through version
control so escape rates are measurable rather than estimated.

**Calibrated suggestion confidence.** A completion that signalled when it was
guessing at a boundary condition would let review depth vary with risk instead of
with fatigue. {{maturity:RESEARCH FRONTIER}}.

**Constraint-aware completion.** Feeding architectural invariants into the
suggestion context, which attacks the largest friction source directly and is
{{ch:aise-repo}}'s subject.

## 16. Connection to Previous Chapters

{{ch:aids-stack}}'s Amdahl argument transfers exactly: the automated stage is small
because being checkable made it efficient first.

{{ch:aids-text-to-sql}}'s free-check finding recurs as running the tests at
acceptance time rather than at commit time.

{{ch:as-failures}}'s correlation result appears in miniature: a model's errors are
plausible, and shallow review fails on plausible errors — the generator's error
distribution and the reviewer's blind spot are the same region.

{{ch:ag-termination}}'s habituation explains why review depth drifts rather than
holding.

Ahead: {{ch:aise-repo}} takes up the constraint-knowledge problem that
{{sec:7-internal-mechanics}} identifies as the largest friction source, and
{{ch:aise-swe-agents}} moves to the regime where the suggestion is a whole change.

## 17. Exercises

1. Derive the review-depth condition from
   {{eq:generated-code-is-under-reviewed}} and evaluate it for your own defect cost.

2. Measure your team's defect cost by sampling recent bug fixes and timing the
   investigation.

3. Add acceptance-time testing to the first listing — an escape probability reduced
   by test coverage — and find how much it moves the optimal acceptance rate.

4. Re-parameterise the second listing for a novice on unfamiliar code and check the
   sign.

5. Model scope creep explicitly: accepted suggestions that enlarge the change. How
   much of the friction does it account for?

6. Estimate your own stage shares by timing three tasks, and compute your Amdahl
   bound.

## 18. Interview Questions

1. Your completion tool reports a $35\%$ acceptance rate. What have you learned?

2. Which suggestions should you accept without close review?

3. Why is generated code reviewed less carefully than written code?

4. You make writing code free. How much faster is the task?

5. Developers report a $20\%$ speedup and measurement shows a $19\%$ slowdown. Are
   they lying?

6. Would you expect a junior or a senior engineer to benefit more, and why?

## 19. Research Questions

1. Can acceptance-time verification be made fast enough to run at suggestion
   latency?

2. Can suggestion-derived defects be attributed reliably through version control?

3. Does the productivity effect's sign change hold across the settings the model
   predicts?

4. Can a model signal its own uncertainty about a boundary condition well enough to
   modulate review?

5. How much of the measured friction is comprehension debt, constraint violation and
   scope creep respectively?

## 20. Chapter Summary

Code completion is measured by acceptance rate, which is recorded at the moment the
correct and plausible-wrong branches are indistinguishable. Across the sweep,
apparent time per block fell the whole way while true cost including later debugging
*rose* from $46.0$ to $57.5$ seconds ({{eq:acceptance-is-not-correctness}}).

**Generated code is reviewed less carefully than written code**
({{eq:generated-code-is-under-reviewed}}) — reading permits a decision per block
where writing forces one per token, and a plausible function invites agreement where
a blank line demands a decision. Letting depth fall from full to a tenth cut apparent
time $20\%$ and raised true cost $65\%$. And a model's errors are plausible by
construction, so shallow review's blind spot and the generator's error distribution
are the same region.

Together they produce opposite signs: at high acceptance with shallow review, an
apparent saving of $+66\%$ against a true saving of $-171\%$. The usable rule is not
an acceptance rate but a ratio — **accept long mechanical suggestions and type the
boundary conditions** ({{eq:ratio-decides-acceptance}}), with the crossover around a
defect costing ten times what the suggestion saves.

Decomposing the task, **writing is $15\%$ of it**, so making writing free gives
$1.17\times$ ({{eq:writing-is-a-small-share}}). The stages the tool does not speed up
are stages it can slow down — comprehension debt, constraint violation, scope creep —
and the break-even is a friction multiplier of $1.26$.

Which explains a measured result. {{cite:becker2025devproductivity}} found sixteen
experienced developers forecasting $24\%$ faster, self-estimating $20\%$ faster, and
measuring **$19\%$ slower**. Separating visible from invisible stages reproduces that
shape ({{eq:visible-half-is-what-is-reported}}), and the bias is *signed*: the
self-report is a faithful account of the half the developer can see.

The effect changes sign across settings — $+4\%$ for an expert on mature familiar
code, $-16\%$ on greenfield — and the trial studied the least favourable cell. So the
honest conclusion is neither that the tools do not work nor that the study is
unrepresentative: **it has to be measured per team, and it cannot be measured by
asking.**

## 21. Further Reading

{{cite:becker2025devproductivity}} should be read in full, including its twenty
explanatory factors — it is the most rigorous measurement available and its
self-report gap is the more transferable of its two findings.

{{cite:jimenez2023swebench}} for where this part goes next: whole-issue resolution
rather than line completion, and a verifier that actually executes.

{{cite:chan2024mlebench}} for the scaffolding-matters-as-much-as-the-model result,
which recurs throughout this part.

{{ch:aids-stack}} for the Amdahl framing this chapter transposes, and
{{ch:aids-text-to-sql}} for the free-check argument that recurs as testing at
acceptance time.
