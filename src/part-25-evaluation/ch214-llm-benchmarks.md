---
id: ev-llm-benchmarks
number: 214
part: XXV
tier: full
status: draft
requires: [metric-choice-manufactures-the-finding, discontinuity-hides-progress,
           evaluation-sets-decay-silently, auc-averages-over-thresholds-you-will-not-use]
provides: [contamination-inflates-and-flattens, headroom-sets-benchmark-lifespan,
           aggregate-hides-which-scenario-moved, a-score-needs-a-human-baseline]
citations: [liang2022helm, rein2023gpqa, singh2025leaderboard, hendrycks2020mmlu,
            card2020power]
---

## 1. Learning Objectives

By the end of this chapter you will be able to model benchmark contamination and show that
it compresses the gap between models as well as inflating their scores; compute the number
of items needed to detect a given difference and derive the year a benchmark stops being
able to track generational progress; explain why growing a benchmark does not extend its
life; interpret a raw score against expert and non-expert baselines rather than against
intuition; decompose an aggregate suite score into per-scenario contributions; and explain
why unequal scenario coverage makes a published ranking partly a fact about coverage.

## 2. Why This Matters

A benchmark has a lifespan and the arithmetic of it is unforgiving. Items leak into
training corpora, so a model with true capability **0.40** reports **0.709** after five
years without learning anything. The part that matters is the compression: the gap between
a 0.55 model and a 0.70 model falls from **0.300 to 0.145** over the same period
({{eq:contamination-inflates-and-flattens}}). Score inflation is correctable; **the gap is
the signal**, and it shrinks by exactly the contaminated fraction.

Because the items needed to detect a difference go as one over the gap squared, that fixes
an expiry date. A 4,000-item benchmark tracking one generation of progress needs **2,086**
items at release and **24,361** in year eight — it stops working in **year 2**
({{eq:headroom-sets-benchmark-lifespan}}). Saturation does the same thing independently: at
a frontier of 0.965 a generation's progress needs **26×** the items it needed at 0.50.

Growing the benchmark does not help. It changes what you have, not what you need.

The second half is what the score means once you can measure it. A scenario reporting
**0.39** has closed **16%** of the distance from a skilled non-expert to a domain expert;
one reporting **0.47** has closed **−23%**, being *below* the non-expert baseline
({{eq:a-score-needs-a-human-baseline}}). Without those two columns, 0.39 and 0.47 rank the
wrong way round.

And aggregates hide which scenario moved. Two models with tied aggregate gains produce
**three different answers** across four defensible summaries
({{eq:aggregate-hides-which-scenario-moved}}), and on a suite where models are run on
different subsets — {{cite:liang2022helm}} found coverage at **17.9%** before a deliberate
effort — the reported ranking inverts on the **25%** of scenarios all models share.

## 3. Prerequisites

{{eq:metric-choice-manufactures-the-finding}} and {{eq:discontinuity-hides-progress}} from
{{ch:ev-why-hard}} govern what a benchmark's *scoring function* does to the finding. This
chapter takes the scoring as given and asks what happens to the benchmark itself over time.

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is the private-set
version of the same decay: coverage falls while the reported number does not move. Here the
mechanism is contamination and saturation rather than traffic drift, and the symptom is
identical.

{{eq:auc-averages-over-thresholds-you-will-not-use}} from {{ch:ev-classical-metrics}} is
this chapter's aggregation result one level down: an average over conditions can be correct
and uninformative about the condition you occupy.

{{cite:hendrycks2020mmlu}} is the benchmark this chapter's arithmetic is most often applied
to, and {{cite:card2020power}} supplies the power analysis every sample-size claim here
rests on.

## 4. Intuitive Explanation

Benchmarks feel permanent. A number on a leaderboard looks like a measurement of a
capability, comparable across models and across years. Both parts of that are wrong, and
they are wrong for different reasons.

Start with contamination. A benchmark is published, which means its items are on the public
internet, which means they end up in training corpora. Not deliberately — the corpora are
scraped, and the benchmark is a web page. A few percent of items leak in the first year,
more in the second, and after five years a substantial fraction of any widely used
benchmark is somewhere in the training data of the models being evaluated on it.

The obvious consequence is score inflation, and everybody knows about it. Scores drift up
without capability drifting up, papers report caveats, and readers discount.

The non-obvious consequence is the one that kills the benchmark. When a model gets a
contaminated item right regardless of capability, that item stops distinguishing models.
Every model gets it right. So the *difference* between two models is computed over the
uncontaminated remainder, and it shrinks in exact proportion to the contaminated fraction.

Score inflation is a bias you can subtract. **Gap compression is a loss of signal, and there
is nothing to subtract.**

Now recall how statistics works. To detect a difference of size $d$ between two proportions,
you need roughly one over $d$ squared items. Halve the gap and you need four times as many.
So contamination does not gently degrade a benchmark; it degrades it quadratically.

Run the arithmetic on a 4,000-item benchmark. At release, distinguishing one model
generation from the next needs about 2,086 items — comfortable. By year two it needs 4,450,
and the benchmark has 4,000. By year eight it needs 24,361.

That gives an honest definition, and it is a definition worth having because it is
computable in advance: **a benchmark is finished when the sample size it would need exceeds
the sample size it has.**

Two things about that definition are worth noticing. First, the answer depends on what you
are asking. The same benchmark can still tell a mediocre model from a good one for twenty
years; it stops being able to tell this generation from the last after two. The durable
version is the one that gets quoted and the fragile one is the one that governs whether a
leaderboard means anything.

Second, saturation does the whole thing again by itself, with no contamination at all.
Suppose each model generation closes a fixed fraction of the remaining gap to perfect —
plausible, and roughly what progress looks like. Then as the frontier rises, each
generation's absolute gain shrinks, and the items needed grow quadratically. Going from a
frontier of 0.50 to one of 0.965 multiplies the required sample size by twenty-six. The
benchmark did not change. The models got good, and getting good is what breaks it.

The instinctive remedy is to make the benchmark bigger, and it is worth being explicit that
this does not work. Growing a benchmark from 4,000 items to 16,000 changes what you have. It
does not change what you need, which is set by the gap and the noise. Four times the
labelling budget buys you four years, maybe, against a requirement growing quadratically.

What does work: hold out a private split, which removes contamination outright and is a
one-time cost. Regenerate items periodically, which helps less than a private split and
costs more, because a set regenerated last year has already absorbed a year of leakage. Or
raise the difficulty so the frontier sits low again — which is not a repair. It is a new
benchmark, and it is why the field keeps building them, each harder than the last by
construction.

That is the first half. The second half is about what the number means on the day it is
still valid.

A model scores 39% on a scenario. Is that bad?

You cannot answer that. Not "you need context" — you *cannot answer it*, because 39% is a
raw count with no units. {{cite:rein2023gpqa}} is the standard example of supplying the
units: on those questions, domain experts reach 65% and skilled non-experts with unrestricted
web access and half an hour per question reach 34%. Now 39% has a meaning. It is a little
above what a determined generalist achieves and well below a specialist.

Do that for a suite of scenarios and the ordering changes. In the example here, a scenario
reporting 0.39 has closed 16% of the expert-minus-non-expert distance. A scenario reporting
0.47 — higher — has closed *negative* 23%, because it sits below the non-expert baseline
entirely. The higher number describes the worse situation, and nothing in the raw report
says so.

**A benchmark score is a raw measurement and the human baselines are its units.**

Then aggregation. A suite reports one number over many scenarios, which is convenient and
throws away the thing you needed.

Take two models with the same aggregate gain. One improved two scenarios that were far
behind; the other improved five that were already near their ceilings. By the aggregate they
are tied. By headroom closed, the second is ahead — because points are cheapest, in headroom
terms, exactly where headroom is smallest. By worst remaining gap, the first is ahead. By
"scenarios still below the non-expert baseline," the first leaves zero and the second leaves
one.

Four defensible summaries. Three different answers. Only the first one gets reported.

There is a Goodhart consequence hiding in that. If a team optimises the aggregate, effort
flows to where points are cheapest, which is where the model is already strong. **The suite
rewards polishing**, and no individual decision inside that process looks wrong.

Finally, coverage. Published comparisons are usually between models run on different subsets
of scenarios, because each lab reports what it chose to run. {{cite:liang2022helm}} measured
this and found models were compared on 17.9% of the same scenarios before a deliberate
effort to fix it — and taking that to 96.0% was not a completeness exercise, it was the
precondition for the comparisons being comparisons.

The example here makes the failure concrete. Three models, different subsets, and the
reported ranking is one order; on the 25% of scenarios all three share, it is a different
order, with the strongest model reported worst because it ran on the hardest subset.

{{cite:singh2025leaderboard}} adds the adversarial version. Unequal access to a
leaderboard's data lets one participant fit the evaluation distribution better than another
— they estimate 19.2% and 20.4% of arena data going to two labs against 29.7% for 83
open-weight models combined, with relative gains up to 112% from the extra data. Coverage
differences and access differences are the same failure at different scales: **the score
partly measures the conditions under which it was collected.**

## 5. Formal Explanation

**Contamination.** Let $c(t)$ be the fraction of items present in training data at time $t$,
and $q$ the model's true capability on uncontaminated items. Assuming a contaminated item is
answered correctly regardless,

$$\hat q(t) = q + (1-q)\,c(t),$$

so for two models the observed gap is $\hat q_2 - \hat q_1 = (q_2 - q_1)(1 - c(t))$: the
inflation is model-dependent and the compression is not. The signal available for
discrimination is scaled by $(1 - c)$ uniformly.

**Sample size.** For a two-proportion comparison at power $1-\beta$ and level $\alpha$, the
per-arm requirement is approximately $n \approx z^2 \cdot 2\bar p(1-\bar p)/d^2$ with $z =
z_{1-\alpha/2} + z_{1-\beta}$. Substituting the compressed gap gives $n(t) \propto (1 -
c(t))^{-2}$, so a benchmark's discriminative capacity decays with the square of its
contamination survival.

**Saturation.** If each generation closes a fixed fraction $\gamma$ of remaining headroom,
$q_{k+1} = q_k + \gamma(1 - q_k)$, then $d_k = \gamma(1 - q_k)$ shrinks geometrically while
$\bar p(1-\bar p)$ shrinks only linearly near the ceiling. The ratio grows, so $n$ grows —
the two effects do not cancel.

**Lifespan.** The benchmark is usable for a question while $n(t) \le N$. Since $n$ is
increasing and $N$ is fixed, there is a well-defined $t^\star$; and since $n$ depends on the
question through $d$, $t^\star$ is a function of the comparison being made, not a property
of the benchmark alone.

**Baselines.** A raw score $s$ carries no scale. Given a non-expert floor $f$ and an expert
ceiling $e$, the normalised quantity $(s - f)/(e - f)$ is comparable across scenarios and can
be negative. Ranking by $s$ and ranking by the normalised value are different orderings, and
neither dominates — they answer different questions.

**Aggregation.** For weights $w_i$ and per-scenario deltas $\delta_i$, the aggregate gain is
$\sum_i w_i \delta_i$, a linear functional with a large null space: many distinct $\delta$
vectors map to the same scalar. Every alternative summary — headroom closed, worst remaining
gap, count below floor — is a different functional of the same vector, and they induce
different orderings.

## 6. Mathematical Foundation

Contamination inflates each score and compresses every gap:

$$\hat q(t) = q + (1-q)c(t), \qquad \hat q_2 - \hat q_1 = (q_2 - q_1)\bigl(1 - c(t)\bigr)$$ (eq:contamination-inflates-and-flattens)

At $c = 51.6\%$ (year 5), a true gap of 0.300 is observed as 0.145.

Lifespan as the crossing of a growing requirement and a fixed supply:

$$n(t) \approx \frac{z^2 \cdot 2\bar p(1-\bar p)}{\bigl[d\,(1 - c(t))\bigr]^2}, \qquad t^\star = \min\{t : n(t) > N\}$$ (eq:headroom-sets-benchmark-lifespan)

At $N = 4000$: $t^\star = 2$ for generational progress and $t^\star > 18$ for coarse tiers.
Note $\partial t^\star/\partial N$ is logarithmic while $\partial n/\partial d$ is quadratic —
which is why growing the benchmark loses the race.

A score's units are its human baselines:

$$\tilde s = \frac{s - f}{e - f}, \qquad \tilde s \in (-\infty, 1], \qquad \operatorname{rank}(s) \neq \operatorname{rank}(\tilde s)$$ (eq:a-score-needs-a-human-baseline)

At $s = 0.39$, $f = 0.34$, $e = 0.65$: $\tilde s = 16\%$. At $s = 0.47$, $f = 0.55$,
$e = 0.90$: $\tilde s = -23\%$.

And the aggregate's null space:

$$G(\boldsymbol\delta) = \sum_i w_i \delta_i, \qquad \dim \ker G = |S| - 1$$ (eq:aggregate-hides-which-scenario-moved)

so tied aggregates carry no information about which of $|S|$ scenarios moved, and three
alternative functionals of the same $\boldsymbol\delta$ give three different verdicts.

## 7. Internal Mechanics

Why does contamination happen despite everyone knowing about it? Because the incentives
that produce a good benchmark and the incentives that keep it clean point in opposite
directions. A benchmark is useful in proportion to how widely it is adopted; it is adopted
by being public; being public is exactly what puts it in the crawl. **The property that makes
a benchmark valuable is the property that destroys it**, on a timescale of a few years, and
no amount of care by the benchmark's authors changes that.

Decontamination checks — searching training corpora for benchmark strings — help less than
they appear to. They find exact matches and miss paraphrases, translations, discussions of
the items in forum posts, and derived exercises. And they require access to the training
corpus, which the party most motivated to run the check usually does not have.

The saturation mechanism has a subtlety that explains why benchmarks feel fine and then
suddenly do not. Near the ceiling, the variance term $\bar p(1-\bar p)$ *is* falling, which
helps. But it falls linearly in $(1 - \bar p)$ while the gap falls linearly too and then gets
squared. So the net requirement grows like $1/(1-\bar p)$, slowly at first and then not. A
benchmark at 78% feels usable, at 93% feels marginal, and at 98.5% needs 63 times the items
it needed at 50% — and the transition between "fine" and "useless" occupies about two
generations.

The baseline problem has an organisational rather than a technical origin. Measuring an
expert baseline means paying experts to sit the benchmark, which is expensive and produces
no headline. Measuring a non-expert baseline means paying non-experts to try hard, which is
cheaper and produces a number that makes the benchmark look easier. Neither is rewarded, so
neither is done, and the scores ship without units. {{cite:rein2023gpqa}} is cited so often
partly because doing this is unusual.

The coverage problem is structural in the same way. Running every model on every scenario is
expensive and coordination-heavy; running your model on the scenarios where it does well is
free and produces a better chart. There is no central authority, so the equilibrium is
partial coverage, and partial coverage means published rankings encode subset choice.
{{cite:liang2022helm}}'s contribution was largely to pay for the coordination, and its
17.9% figure is the measurement of what the equilibrium had produced.

Finally, note how these interact. A saturating benchmark is one where models are close
together, which is exactly when subset choice and small coverage differences flip rankings.
So the two halves of this chapter compound: **an old benchmark is both less able to detect a
difference and more sensitive to which items were run**, and the two failures arrive at the
same time.

## 8. Implementation

The first listing computes a benchmark's expiry date.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hc1}
"""A public benchmark has a lifespan, and it is shorter than anyone plans for.

Two things happen to a benchmark after publication. Its items leak into training corpora,
so reported scores rise faster than capability does. And the frontier moves toward the
ceiling, so the difference between the models anyone cares about gets smaller.

Both effects push the same way: **the gap between two models shrinks while the noise stays
put** (eq:contamination-inflates-and-flattens), so the number of items needed to tell them
apart grows until it exceeds the number of items the benchmark has
(eq:headroom-sets-benchmark-lifespan).

This listing computes that crossing point, which is the honest definition of when a
benchmark is finished.
"""
import math

N_ITEMS = 4000
LEAK_RATE = 0.145             # per year; contamination = 1 - exp(-rate * years)
POWER_Z = 2.80                # z(0.80 power) + z(0.05 two-sided), roughly


def contamination(years):
    return 1.0 - math.exp(-LEAK_RATE * years)


def reported(true_q, years):
    """Contaminated items are answered correctly whether or not the model can."""
    c = contamination(years)
    return true_q + (1.0 - true_q) * c


print(f"A {N_ITEMS}-item benchmark. Items leak into training corpora over time.")
print()
print(f"{'years':>7}{'contaminated':>15}", end="")
for q in (0.40, 0.55, 0.70):
    print(f"{('true ' + format(q, '.2f')):>13}", end="")
print(f"{'A vs C gap':>13}")
print("-" * 74)
tab = {}
for y in (0, 1, 2, 3, 5, 8):
    c = contamination(y)
    row = [reported(q, y) for q in (0.40, 0.55, 0.70)]
    tab[y] = (c, row, row[2] - row[0])
    print(f"{y:>7}{c:>15.1%}", end="")
    for v in row:
        print(f"{v:>13.3f}", end="")
    print(f"{row[2] - row[0]:>13.3f}")

print()
print("Every model looks better every year without improving, and the space")
print("between them closes.")

print()
print()
print("What that does to the sample size needed to separate two models.")
print()


def n_needed(p1, p2):
    """Two-proportion comparison, 80% power, alpha 0.05."""
    d = abs(p2 - p1)
    if d < 1e-9:
        return float("inf")
    p = (p1 + p2) / 2.0
    return (POWER_Z ** 2) * 2.0 * p * (1 - p) / (d ** 2)


print(f"{'years':>7}{'reported gap':>15}{'items needed':>15}"
      f"{'benchmark has':>16}{'usable?':>10}")
print("-" * 63)
life = {}
for y in (0, 1, 2, 3, 5, 8, 12, 18):
    c = contamination(y)
    a, b = reported(0.55, y), reported(0.70, y)
    n = n_needed(a, b)
    life[y] = (b - a, n)
    print(f"{y:>7}{b - a:>15.3f}{n:>15.0f}{N_ITEMS:>16}"
          f"{('yes' if n <= N_ITEMS else 'no'):>10}")

dead = min((y for y in (0, 1, 2, 3, 5, 8, 12, 18) if life[y][1] > N_ITEMS),
           default=None)
print()
print("for a gap this coarse the benchmark survives "
      + (f"until year {dead}" if dead else "past year 18"))

print()
print()
print("Saturation does the same thing independently, with no contamination at all.")
print()
print(f"{'frontier score':>16}{'next model':>13}{'true gap':>11}"
      f"{'items needed':>15}{'vs at 0.50':>13}")
print("-" * 68)
STEP = 0.045
sat = {}
base = None
for f in (0.50, 0.65, 0.78, 0.87, 0.93, 0.965, 0.985):
    nxt = min(0.999, f + STEP * (1 - f) / 0.5)
    n = n_needed(f, nxt)
    if base is None:
        base = n
    sat[f] = (nxt, nxt - f, n)
    print(f"{f:>16.3f}{nxt:>13.3f}{nxt - f:>11.4f}{n:>15.0f}{n / base:>12.1f}x")

print()
print("A fixed fraction of remaining headroom is a shrinking absolute gain,")
print("and sample size goes as one over the gap squared.")

print()
print()
print("Both effects together: benchmark lifespan against the frontier it tracks.")
print()
print(f"{'year':>6}{'contamination':>16}{'frontier':>11}{'true step':>12}"
      f"{'observed step':>16}{'items needed':>15}{'usable?':>10}")
print("-" * 86)
front = 0.52
both = {}
for y in range(0, 9):
    c = contamination(y)
    nxt = front + STEP * (1 - front) / 0.5
    obs_a, obs_b = reported(front, y), reported(min(nxt, 0.999), y)
    n = n_needed(obs_a, obs_b)
    both[y] = (c, front, nxt - front, obs_b - obs_a, n)
    print(f"{y:>6}{c:>16.1%}{front:>11.3f}{nxt - front:>12.4f}"
          f"{obs_b - obs_a:>16.4f}{n:>15.0f}"
          f"{('yes' if n <= N_ITEMS else 'no'):>10}")
    front = nxt
dead_gen = min((y for y in both if both[y][4] > N_ITEMS), default=None)
print()
print(f"for one generation of progress it survives until year {dead_gen}")

print()
print()
print("What each remedy buys, at year 5.")
print()
Y = 5
c5 = contamination(Y)
f5 = both[Y][1]
n5 = both[Y][4]
print(f"{'remedy':>32}{'items needed':>15}{'vs nothing':>13}{'cost':>22}")
print("-" * 82)
REMEDIES = [
    ("nothing", n5, "zero"),
    ("grow the benchmark 4x", n5, "4x labelling"),
    ("hold out a private split", n_needed(f5, f5 + both[Y][2]), "one-time"),
    ("regenerate items annually",
     n_needed(reported(f5, 1), reported(f5 + both[Y][2], 1)), "annual labelling"),
    ("raise difficulty (reset headroom)",
     n_needed(0.39, 0.39 + STEP * (1 - 0.39) / 0.5), "a new benchmark"),
]
rem = {}
for name, n, cost in REMEDIES:
    rem[name] = n
    print(f"{name:>32}{n:>15.0f}{n5 / n:>12.1f}x{cost:>22}")

print()
print(f"note: growing the benchmark changes what {N_ITEMS} means, not what is needed.")

print(f"""
The contamination table is the mechanism and the last column is the finding. A model with
true capability {0.40:.2f} reports {tab[5][1][0]:.3f} after five years without having
learned anything, and one at {0.70:.2f} reports {tab[5][1][2]:.3f}. The gap between them
falls from {tab[0][2]:.3f} to {tab[5][2]:.3f}.

That compression is the part that matters. Score inflation on its own is annoying and
correctable -- everyone knows the numbers are optimistic. **The gap shrinking is not
correctable**, because it is the signal (eq:contamination-inflates-and-flattens), and it
shrinks by exactly the contaminated fraction.

The sample-size table converts that into a date. Separating a {0.55:.2f} model from a
{0.70:.2f} model needs {life[0][1]:.0f} items at release and {life[18][1]:.0f} in year 18,
because the required count goes as one over the gap squared. For a gap that coarse, a
{N_ITEMS}-item benchmark survives past year 18.

That is the reassuring version, and it is why benchmarks feel durable. It is also the wrong
question, because nobody is choosing between a {0.55:.2f} model and a {0.70:.2f} one -- the
comparisons that matter are between this generation and the last.

The saturation table shows the same failure arriving without any contamination at all.
Suppose each model generation closes a fixed fraction of the remaining headroom -- a
reasonable model of progress. At a frontier of {0.50:.2f} that is a
{sat[0.50][1]:.4f} gain needing {sat[0.50][2]:.0f} items; at {0.965:.3f} it is
{sat[0.965][1]:.4f} needing {sat[0.965][2]:.0f} --
**{sat[0.965][2] / sat[0.50][2]:.0f} times as many** (eq:headroom-sets-benchmark-lifespan).

The two mechanisms are independent and they compound. The combined table walks a frontier
from {0.52:.2f} upward while contamination accumulates, and the items needed to detect one
generation's progress goes from {both[0][4]:.0f} in year 0 to {both[8][4]:.0f} in year 8.

Against {N_ITEMS} items, the benchmark can do it in years 0 and 1 and cannot from
**year {dead_gen}** onward.

So the useful definition is conditional on the question: **a benchmark is finished when the
sample size it would need exceeds the sample size it has**, and the year that happens is
{dead_gen} for tracking progress and past 18 for sorting coarse tiers. Both numbers are
computable in advance from a leak-rate estimate; the second is the one that gets quoted and
the first is the one that governs whether a leaderboard means anything.

The remedies table is where this becomes a decision rather than an observation, and the
first two rows are the important ones. **Growing the benchmark does not help.** Quadrupling
the item count leaves the required count exactly where it was -- it changes what you have,
not what you need, and it is the intervention teams reach for because it is the one that
feels like effort.

A private held-out split removes contamination entirely and takes the requirement to
{rem['hold out a private split']:.0f} items -- a
{n5 / rem['hold out a private split']:.1f}x improvement for a one-time cost, and the best
return per unit of effort on the list. Annual regeneration reaches only
{rem['regenerate items annually']:.0f} for a *recurring* cost, because a set regenerated
last year has already absorbed a year of leakage.

And the last row is the honest one. Raising difficulty so the frontier sits back at
{0.39:.2f} needs {rem['raise difficulty (reset headroom)']:.0f} items --
{n5 / rem['raise difficulty (reset headroom)']:.0f} times better than anything else -- and
it is not a repair. It is a new benchmark, which is why the field keeps building them, and
why each new one is harder than the last by construction rather than by ambition.

Two cautions. The leak rate here is a parameter and nobody publishes theirs, so the dates
above are arithmetic on an estimate; the *shape* is robust and the year is not. And this
listing measures only whether two models can be *separated*, which is a lower bar than
whether the separation means anything -- ch:ev-llm-benchmarks' second listing takes up what
the score is measuring in the first place.""")
```

## 9. Practical Example

A 4,000-item benchmark leaking into training corpora:

```
  years   contaminated    true 0.40    true 0.55    true 0.70   A vs C gap
--------------------------------------------------------------------------
      0           0.0%        0.400        0.550        0.700        0.300
      2          25.2%        0.551        0.663        0.776        0.224
      3          35.3%        0.612        0.709        0.806        0.194
      5          51.6%        0.709        0.782        0.855        0.145
      8          68.7%        0.812        0.859        0.906        0.094
```

Every model looks better every year without improving, and **the gap closes from 0.300 to
0.094** ({{eq:contamination-inflates-and-flattens}}). Inflation is a bias you can subtract;
the gap is the signal.

```
  years   reported gap   items needed   benchmark has   usable?
---------------------------------------------------------------
      0          0.150            163            4000       yes
      5          0.073            442            4000       yes
     12          0.026           1391            4000       yes
     18          0.011           3456            4000       yes
```

For a gap that coarse the benchmark survives past year 18 — which is the reassuring version,
and the wrong question. Nobody is choosing between a 0.55 model and a 0.70 model.

```
  frontier score   next model   true gap   items needed   vs at 0.50
--------------------------------------------------------------------
           0.500        0.545     0.0450           1932         1.0x
           0.780        0.800     0.0198           6638         3.4x
           0.930        0.936     0.0063          24644        12.8x
           0.985        0.986     0.0013         121481        62.9x
```

Saturation alone, with no contamination: **26× the items at a 0.965 frontier, 63× at 0.985**
({{eq:headroom-sets-benchmark-lifespan}}).

```
  year   contamination   frontier   true step   observed step   items needed   usable?
--------------------------------------------------------------------------------------
     0            0.0%      0.520      0.0432          0.0432           2086       yes
     1           13.5%      0.563      0.0393          0.0340           3127       yes
     2           25.2%      0.603      0.0358          0.0268           4450        no
     5           51.6%      0.700      0.0270          0.0131          10978        no
     8           68.7%      0.774      0.0203          0.0064          24361        no
```

Both effects compounding: **the benchmark stops tracking generational progress in year 2**,
having survived past year 18 for coarse tiers. The lifespan is a property of the question,
not of the benchmark.

```
                          remedy   items needed   vs nothing                  cost
----------------------------------------------------------------------------------
                         nothing          10978         1.0x                  zero
           grow the benchmark 4x          10978         1.0x          4x labelling
        hold out a private split           4406         2.5x              one-time
       regenerate items annually           5369         2.0x      annual labelling
raise difficulty (reset headroom)           1265         8.7x       a new benchmark
```

**Growing the benchmark buys nothing** — it changes what you have, not what you need. A
private held-out split is 2.5× for a one-time cost. And the 8.7× row is not a repair; it is
a new benchmark, which is why the field keeps building them.

The second listing asks what the number means while it is still valid.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/hc2}
"""A benchmark score has no scale until somebody measures a human on the same items.

39% sounds bad. 72% sounds good. Neither statement is meaningful without knowing what a
domain expert scores on the same questions -- and cite:rein2023gpqa is the standard example
of supplying that: experts reach 65%, skilled non-experts with unrestricted web access
reach 34% (eq:a-score-needs-a-human-baseline).

The second problem is aggregation. A suite reports one number over many scenarios, and two
models with the same aggregate gain can have improved on completely different things
(eq:aggregate-hides-which-scenario-moved).

And underneath both, cite:liang2022helm's coverage result: before a deliberate effort,
models were compared on 17.9% of the same scenarios, which means most published comparisons
were between different measurements.
"""
# (scenario, weight, base model score, non-expert baseline, expert baseline)
SCENARIOS = [
    ("factual recall",       0.14, 0.81, 0.42, 0.94),
    ("multi-step reasoning", 0.16, 0.54, 0.31, 0.88),
    ("code generation",      0.13, 0.63, 0.22, 0.91),
    ("summarisation",        0.11, 0.77, 0.61, 0.86),
    ("graduate science",     0.09, 0.39, 0.34, 0.65),
    ("legal reasoning",      0.10, 0.58, 0.36, 0.83),
    ("tool use",             0.14, 0.47, 0.55, 0.90),
    ("long-context recall",  0.13, 0.71, 0.48, 0.92),
]

base = {n: s for n, w, s, ne, ex in SCENARIOS}
weight = {n: w for n, w, s, ne, ex in SCENARIOS}
nonexp = {n: ne for n, w, s, ne, ex in SCENARIOS}
expert = {n: ex for n, w, s, ne, ex in SCENARIOS}
agg = sum(weight[n] * base[n] for n in base)

print(f"An eight-scenario suite. Aggregate score: {agg:.4f}")
print()
print(f"{'scenario':>22}{'weight':>9}{'score':>8}{'non-expert':>13}"
      f"{'expert':>9}{'headroom':>11}{'% closed':>11}")
print("-" * 83)
for n, w, s, ne, ex in SCENARIOS:
    head = ex - s
    closed = (s - ne) / (ex - ne) if ex > ne else 1.0
    print(f"{n:>22}{w:>9.2f}{s:>8.3f}{ne:>13.3f}{ex:>9.3f}"
          f"{head:>11.3f}{closed:>11.0%}")

print()
print("The raw-score ranking and the headroom-closed ranking are different")
print("orderings of the same table.")
print()
by_raw = sorted(base, key=lambda n: -base[n])
by_closed = sorted(base, key=lambda n:
                   -((base[n] - nonexp[n]) / (expert[n] - nonexp[n])))
print(f"{'rank':>6}{'by raw score':>24}{'by headroom closed':>26}")
print("-" * 56)
for i in range(len(SCENARIOS)):
    print(f"{i + 1:>6}{by_raw[i]:>24}{by_closed[i]:>26}")

print()
print()
print("Two models with the same aggregate gain, improving different things.")
print()
MODEL_A = {"tool use": 0.14, "multi-step reasoning": 0.11}
MODEL_B = {"factual recall": 0.10, "summarisation": 0.08,
           "long-context recall": 0.07, "code generation": 0.02,
           "legal reasoning": 0.02}


def gain(deltas):
    return sum(weight[n] * d for n, d in deltas.items())


print(f"{'model':>9}{'scenarios improved':>21}{'aggregate gain':>17}"
      f"{'new aggregate':>16}")
print("-" * 63)
for name, d in (("A", MODEL_A), ("B", MODEL_B)):
    g = gain(d)
    print(f"{name:>9}{len(d):>21}{g:>17.4f}{agg + g:>16.4f}")

print()
print("Same headline movement. Four defensible summaries of the same pair.")
print()


def closed(n, d=0.0):
    s = min(0.999, base[n] + d)
    return (s - nonexp[n]) / (expert[n] - nonexp[n])


print(f"{'summary':>36}{'model A':>12}{'model B':>12}{'says':>10}")
print("-" * 70)
summ = {}
for label, f in (
        ("aggregate gain",
         lambda d: gain(d)),
        ("weighted headroom closed",
         lambda d: sum(weight[n] * (closed(n, d.get(n, 0.0)) - closed(n))
                       for n in base)),
        ("worst remaining gap to expert",
         lambda d: max(expert[n] - min(0.999, base[n] + d.get(n, 0.0))
                       for n in base)),
        ("scenarios below non-expert",
         lambda d: float(sum(1 for n in base
                             if base[n] + d.get(n, 0.0) < nonexp[n]))),
):
    a, b = f(MODEL_A), f(MODEL_B)
    summ[label] = (a, b)
    if abs(a - b) < 1e-3:
        verdict = "tie"
    elif label == "worst remaining gap to expert":
        verdict = "A" if a < b else "B"
    elif label == "scenarios below non-expert":
        verdict = "A" if a < b else "B"
    else:
        verdict = "A" if a > b else "B"
    print(f"{label:>36}{a:>12.4f}{b:>12.4f}{verdict:>10}")

print()
print("Three different answers from four summaries, and only the first is")
print("ever reported.")

print()
print()
print("Where each model's aggregate gain actually came from.")
print()
print(f"{'scenario':>22}{'A contributes':>16}{'share of A':>13}"
      f"{'B contributes':>16}{'share of B':>13}")
print("-" * 80)
ga, gb = gain(MODEL_A), gain(MODEL_B)
for n, w, s, ne, ex in SCENARIOS:
    ca = weight[n] * MODEL_A.get(n, 0.0)
    cb = weight[n] * MODEL_B.get(n, 0.0)
    print(f"{n:>22}{ca:>16.4f}{ca / ga:>13.0%}{cb:>16.4f}{cb / gb:>13.0%}")

top_a = max(MODEL_A, key=lambda n: weight[n] * MODEL_A[n])
top_b = max(MODEL_B, key=lambda n: weight[n] * MODEL_B[n])
sh_a = weight[top_a] * MODEL_A[top_a] / ga
sh_b = weight[top_b] * MODEL_B[top_b] / gb
print()
print(f"largest single contributor: A {sh_a:.0%} ({top_a}), "
      f"B {sh_b:.0%} ({top_b})")

print()
print()
print("And the coverage problem: three models evaluated on different subsets.")
print()
COVER = {
    "model P": ["factual recall", "summarisation", "long-context recall",
                "code generation", "tool use"],
    "model Q": ["multi-step reasoning", "graduate science", "legal reasoning",
                "tool use", "code generation"],
    "model R": ["factual recall", "summarisation", "code generation",
                "multi-step reasoning", "tool use"],
}
SKILL = {"model P": 0.00, "model Q": 0.09, "model R": 0.05}
print(f"{'model':>10}{'true skill offset':>20}{'reported mean':>16}"
      f"{'reported rank':>16}{'scenarios':>12}")
print("-" * 74)
rep = {}
for m, sc in COVER.items():
    uniq = sorted(set(sc))
    rep[m] = sum(min(0.999, base[n] + SKILL[m]) for n in uniq) / len(uniq)
order = sorted(rep, key=lambda m: -rep[m])
for m in COVER:
    print(f"{m:>10}{SKILL[m]:>20.2f}{rep[m]:>16.4f}"
          f"{order.index(m) + 1:>16}{len(set(COVER[m])):>12}")

common = set.intersection(*[set(v) for v in COVER.values()])
print()
print(f"common scenarios across all three: {len(common)} of {len(SCENARIOS)} "
      f"({len(common) / len(SCENARIOS):.1%})")
print()
fair = {m: sum(min(0.999, base[n] + SKILL[m]) for n in common) / len(common)
        for m in COVER}
forder = sorted(fair, key=lambda m: -fair[m])
print(f"{'model':>10}{'on common set':>16}{'fair rank':>12}"
      f"{'reported rank':>16}{'moved':>8}")
print("-" * 62)
for m in COVER:
    print(f"{m:>10}{fair[m]:>16.4f}{forder.index(m) + 1:>12}"
          f"{order.index(m) + 1:>16}"
          f"{order.index(m) - forder.index(m):>8}")

print(f"""
The scenario table is where the chapter's first point lives, in the last two columns.
`graduate science` reports {base['graduate science']:.3f}, which reads as a failure. Against
a non-expert baseline of {nonexp['graduate science']:.3f} and an expert ceiling of
{expert['graduate science']:.3f} it has closed
{(base['graduate science'] - nonexp['graduate science']) / (expert['graduate science'] - nonexp['graduate science']):.0%}
of the available distance (eq:a-score-needs-a-human-baseline).

`tool use` reports {base['tool use']:.3f} -- higher -- and has closed
{(base['tool use'] - nonexp['tool use']) / (expert['tool use'] - nonexp['tool use']):.0%},
because it is *below the non-expert baseline*. A number that looks better is describing a
system that is worse than an untrained human at the task.

Without the two human columns those two rows are 0.39 and 0.47 and the ordering is obvious
and wrong. **A benchmark score is a raw measurement and the baselines are its units**, which
is why cite:rein2023gpqa's decision to measure experts and non-experts under the same
conditions matters more than its headline difficulty.

The two rankings confirm it. Sorting by raw score and sorting by headroom closed produce
different orders of the same eight rows, and a roadmap built from the first is aimed at the
scenarios where the model is already doing well relative to what is achievable.

The model-comparison table is the aggregation problem. Model A improves
{len(MODEL_A)} scenarios and model B improves {len(MODEL_B)}, and the aggregate moves by
{ga:.4f} and {gb:.4f} -- **indistinguishable in the headline**
(eq:aggregate-hides-which-scenario-moved).

The four-summary table is the uncomfortable part. By the aggregate the two models are
tied -- {summ['aggregate gain'][0]:.4f} against {summ['aggregate gain'][1]:.4f}.

Measured as *headroom closed*, B is ahead
({summ['weighted headroom closed'][1]:.4f} against
{summ['weighted headroom closed'][0]:.4f}), because points are cheapest in headroom terms
exactly where headroom is smallest: improving a scenario already near its ceiling closes a
larger *fraction* of what remained.

Measured by the worst remaining gap to expert, A is ahead
({summ['worst remaining gap to expert'][0]:.2f} against
{summ['worst remaining gap to expert'][1]:.2f}), because A attacked what was furthest
behind. And by scenarios still below the non-expert baseline, A leaves
{summ['scenarios below non-expert'][0]:.0f} and B leaves
{summ['scenarios below non-expert'][1]:.0f} -- B never touched `tool use`, so it ships a
suite containing a task the model does worse than an untrained human.

**Four defensible summaries, three different answers, and only the first one is ever
reported.** That is worth sitting with, because the usual reaction to a hidden aggregate is
"report a better aggregate," and this table says there is no better aggregate -- there are
questions, and each summary answers one.

There is a Goodhart consequence hiding in the headroom row that is worth naming. If a team
optimises the aggregate, points are cheapest where the model is already strong, so the
optimisation pushes effort toward scenarios that are nearly finished and away from the ones
that are far behind. **The suite rewards polishing**, and no individual decision inside that
process looks wrong.

The contribution table says something further that the aggregate cannot. Model A's gain is
{sh_a:.0%} concentrated in `{top_a}` and model B's is {sh_b:.0%} in `{top_b}`, spread across
{len(MODEL_B)} scenarios. Report the aggregate alone and a reader assumes broad improvement
in both cases; one of them is broad and the other is two scenarios, and those are different
claims about what the model can now do.

**The decomposition costs nothing** -- it is the same numbers, not summed -- and almost no
report includes it.

The coverage table is the third failure and it is cite:liang2022helm's contribution.
Three models are evaluated on overlapping but different scenario subsets, which is how
public comparison actually works when each lab reports what it chose to run. The reported
ranking is {' > '.join(order)}.

On the {len(common)} scenarios all three actually share -- {len(common) / len(SCENARIOS):.0%}
of the suite -- the ranking is {' > '.join(forder)}.

Model Q reports worst and is best on the shared items; it is the strongest model in the
table and it ran on the hardest subset. The reported ordering is partly a fact about which
scenarios each model was run on. That is
the whole of cite:liang2022helm's coverage argument: taking coverage from
{0.179:.1%} to {0.960:.1%} was not a completeness exercise, it was **the precondition for the
comparisons to be comparisons at all.**

And cite:singh2025leaderboard's audit adds the adversarial version of the same point.
Unequal access to a leaderboard's data lets one participant fit the evaluation distribution
better than another, with relative gains reported up to {1.12:.0%} on the arena
distribution. Coverage differences and access differences are the same failure at different
scales: **the score partly measures the conditions under which it was collected.**

One thing to keep hold of. None of this says benchmarks are useless -- it says a benchmark
number is a measurement that has not yet been given units, a decomposition, or a matched
comparison set. All three are cheap. Two of them are free.""")
```

```
              scenario   weight   score   non-expert   expert   headroom   % closed
-----------------------------------------------------------------------------------
        factual recall     0.14   0.810        0.420    0.940      0.130        75%
  multi-step reasoning     0.16   0.540        0.310    0.880      0.340        40%
      graduate science     0.09   0.390        0.340    0.650      0.260        16%
              tool use     0.14   0.470        0.550    0.900      0.430       -23%
   long-context recall     0.13   0.710        0.480    0.920      0.210        52%
```

`graduate science` at **0.390** has closed **16%**; `tool use` at **0.470** — higher — has
closed **−23%**, being below the non-expert baseline
({{eq:a-score-needs-a-human-baseline}}). **The higher number describes the worse
situation**, and nothing in the raw report says so.

```
    model   scenarios improved   aggregate gain   new aggregate
---------------------------------------------------------------
        A                    2           0.0372          0.6548
        B                    5           0.0365          0.6541

                             summary     model A     model B      says
----------------------------------------------------------------------
                      aggregate gain      0.0372      0.0365       tie
            weighted headroom closed      0.0869      0.0908         B
       worst remaining gap to expert      0.2900      0.4300         A
          scenarios below non-expert      0.0000      1.0000         A
```

**Four defensible summaries, three different answers**
({{eq:aggregate-hides-which-scenario-moved}}), and only the first is ever reported. Note the
Goodhart consequence in row two: points are cheapest in headroom terms where headroom is
smallest, so **optimising the aggregate rewards polishing**.

```
     model   true skill offset   reported mean   reported rank   scenarios
--------------------------------------------------------------------------
   model P                0.00          0.6780               2           5
   model Q                0.09          0.6120               3           5
   model R                0.05          0.6940               1           5

common scenarios across all three: 2 of 8 (25.0%)

     model   on common set   fair rank   reported rank   moved
--------------------------------------------------------------
   model P          0.5500           3               2      -1
   model Q          0.6400           1               3       2
   model R          0.6000           2               1      -1
```

The strongest model reports worst because it ran on the hardest subset. This is
{{cite:liang2022helm}}'s coverage argument: **17.9% → 96.0% was the precondition for the
comparisons to be comparisons**, and {{cite:singh2025leaderboard}}'s **112%** is the
adversarial version of the same point.

## 10. Production Considerations

Estimate your benchmark's expiry date before adopting it. Leak rate times gap size gives a
year, and the year is usually sooner than the benchmark's age.

Hold out a private split from day one. It is the cheapest remedy on the list and it must be
created before publication, not after contamination is suspected.

Stop growing benchmarks to extend their life. Four times the labelling buys a fraction of a
year against a quadratic requirement.

Measure expert and non-expert baselines on your own evaluation scenarios. Two afternoons of
human time gives every score in your report units it currently lacks.

Publish the per-scenario decomposition alongside every aggregate. It is the same numbers,
not summed, and it is what distinguishes a broad improvement from two scenarios.

Report at least two summaries — one aggregate, one worst-case — because they disagree and
the disagreement is information.

Compare models only on the scenarios all of them ran. If that set is small, say how small;
the reported ranking is otherwise partly a ranking of subset choices.

## 11. Common Mistakes

**Discounting contamination as inflation only.** The inflation is subtractable; the gap
compression is not.

**Growing the benchmark to keep it alive.** It changes the supply, not the requirement, and
the requirement grows quadratically.

**Quoting a score without a human baseline.** 39% and 47% rank the wrong way round without
one.

**Reading a tied aggregate as a tie.** Four summaries of the same pair here give three
different answers.

**Comparing across incomplete coverage.** The reported ordering inverts on the shared 25%.

**Assuming a benchmark's lifespan is a property of the benchmark.** It is a property of the
comparison: 18+ years for coarse tiers, 2 years for generational progress.

## 12. Failure Modes

**Leaderboard that stops moving.** Every frontier model reports within a point of every
other, which reads as convergence and is a benchmark that has run out of resolution.

**Roadmap aimed at the raw-score ranking.** Effort goes to the scenarios with the highest
scores, which are the ones nearest their ceilings.

**A capability below the non-expert floor shipped as adequate.** The scenario reported 0.47
and nobody measured what an untrained human scores.

**Aggregate improvement that is one scenario.** The report says the model got better; the
truth is one scenario moved and the reader cannot tell.

**Ranking that inverts on a fair subset.** Two labs each ran the scenarios that suited them,
and the published comparison encodes the choice.

**Decontamination check that finds nothing.** Exact-match search over a corpus you do not
control, reported as evidence of cleanliness.

## 13. Alternatives

**Private held-out evaluation.** Contamination cannot occur if the items were never
published. The strongest remedy, and it forecloses external replication.

**Continuously regenerated items.** Fresh items each cycle from a generator or a live
stream. Removes contamination and introduces the question of whether this cycle's items are
comparable to last cycle's.

**Execution-graded tasks.** {{ch:ev-why-hard}}'s escape, applied at benchmark scale: real
repositories and real test suites, where the answer cannot be memorised as a string.
Contamination still helps and helps less.

**Pairwise preference arenas.** Rank models by head-to-head human preference rather than by
a fixed item set. No saturation ceiling and, per {{cite:singh2025leaderboard}}, a different
set of distortions from unequal data access.

**Capability-specific test suites.** Many small targeted evaluations instead of one score.
Harder to game, harder to summarise, and the only form that survives the aggregation
critique intact.

## 14. Evaluation

Estimate your leak rate by testing whether a model completes benchmark items from a partial
prompt. It is imperfect and it is a number, which beats the assumption of zero.

Compute the items needed to detect your expected generational gain and compare against your
benchmark's size. If it exceeds, the benchmark is already retired and nobody has said so.

Measure expert and non-expert baselines for your top three scenarios and republish every
past score with units.

Publish the per-scenario decomposition for every aggregate you report, and the ordering
under at least one alternative summary.

Compute the shared-scenario subset for any external comparison you rely on, and recompute
the ranking on it.

## 15. Advanced Concepts

The contamination model here assumes a leaked item is answered correctly with probability
one, which overstates the effect for hard items and understates it for easy ones. Real
memorisation is partial and depends on how many times the item appeared, in what form, and
how distinctive it is. The correction shrinks the inflation term and — importantly — leaves
the *compression* result nearly intact, because compression depends on the leaked items
becoming uninformative rather than on them being answered perfectly. An item that every model
gets right 85% of the time discriminates almost as poorly as one every model gets right.
**The result that matters is robust to the assumption that looks most questionable**, which
is worth checking before dismissing the model.

There is an identification problem underneath the whole chapter that no amount of care
resolves. Rising scores are consistent with rising capability, rising contamination, and
rising benchmark-specific optimisation, and a single benchmark cannot separate the three.
The only clean separations available are a private held-out split evaluated at the same time
— which isolates contamination — and a transfer task the model was not optimised for, which
isolates benchmark-specific fitting. Both are cheap, neither is standard, and their absence
is why the field argues about whether progress is real rather than measuring it.

The aggregation result connects to {{ch:ev-classical-metrics}} more tightly than the two
chapters suggest. There, a metric implied a cost ratio nobody had chosen; here, a weighting
vector implies a *scenario* importance ranking nobody has chosen. Suite weights are almost
always set by item counts or by round numbers, which means the aggregate encodes an
importance judgement that arrived by accident. The remedy is the same in both cases — state
the weights as a decision and defend them — and it is skipped for the same reason: it makes
a judgement visible that was more comfortable implicit.

Finally, the interaction between saturation and coverage deserves more attention than it
gets. As a benchmark saturates, models cluster, and clustered models are exactly the regime
where a small coverage difference flips a ranking. So the two failures are not independent
risks to be managed separately — **an old benchmark is simultaneously less able to detect a
difference and more sensitive to which items were run.** Any leaderboard that has been
running for years is in both regimes at once, and its rankings should be read as such.

## 16. Connection to Previous Chapters

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is the private-set
analogue of this chapter's contamination result: coverage falls, the reported number does
not move, and nothing announces it.

{{eq:auc-averages-over-thresholds-you-will-not-use}} from {{ch:ev-classical-metrics}}
reappears as {{eq:aggregate-hides-which-scenario-moved}} — an average over conditions that is
correct and silent about the condition you occupy.

{{eq:metric-choice-manufactures-the-finding}} from {{ch:ev-why-hard}} governs the scoring
function; this chapter governs the item set and its decay, and the two failures compose.

{{cite:hendrycks2020mmlu}}'s calibration finding remains the most durable result attached to
the benchmark most subject to this chapter's arithmetic, which is a coincidence worth
noticing: the part that ages best is the part nobody quotes.

## 17. Exercises

1. Estimate the leak rate for a benchmark you use and compute its expiry date for your
   expected generational gain.

2. Compute the items needed to detect a 2-point difference at your current frontier score.
   How does it change at a frontier 10 points higher?

3. Measure a non-expert baseline on ten items from one of your evaluation scenarios. How
   many of your reported scores change meaning?

4. Decompose your last aggregate improvement by scenario. What share came from the largest
   contributor?

5. Model partial memorisation — a leaked item answered correctly with probability $m < 1$ —
   and show how much of {{sec:15-advanced-concepts}}'s compression result survives.

## 18. Interview Questions

1. Our model scores 72% on this benchmark. Is that good?

2. Why does contamination hurt more than the score inflation suggests?

3. We are doubling our benchmark to keep it useful. Will that work?

4. Two models tie on the aggregate. What do you ask next?

5. Why can two labs' published scores not be compared directly?

6. When should a benchmark be retired, and how would you compute the date?

## 19. Research Questions

1. What are empirical leak rates for widely used benchmarks, and how do they vary with
   publication venue and item format?

2. How much of the observed gap compression on aging benchmarks is contamination versus
   saturation, and can the two be separated observationally?

3. Can a partial-memorisation model be fitted from response patterns without corpus access?

4. How sensitive are published leaderboard rankings to the shared-scenario subset, measured
   across the major public leaderboards?

## 20. Chapter Summary

Benchmarks expire, and the expiry date is computable in advance.

Contamination inflates every score and — the part that matters — compresses every gap by
exactly the contaminated fraction: a true gap of **0.300** reads **0.145** after five years
({{eq:contamination-inflates-and-flattens}}). Since items needed goes as one over the gap
squared, a 4,000-item benchmark that needs **2,086** items at release needs **24,361** by
year eight and **stops tracking generational progress in year 2** — while remaining fine for
coarse comparisons past year 18 ({{eq:headroom-sets-benchmark-lifespan}}). Saturation does
the same independently: **26×** the items at a 0.965 frontier. **Growing the benchmark buys
nothing**; a private held-out split is 2.5× for a one-time cost.

What the score means is a separate problem. A scenario at **0.390** has closed **16%** of
the non-expert-to-expert distance and one at **0.470** has closed **−23%**, being below the
non-expert floor ({{eq:a-score-needs-a-human-baseline}}) — the higher number is the worse
situation. Aggregates hide which scenario moved: four defensible summaries of two tied models
give **three different answers** ({{eq:aggregate-hides-which-scenario-moved}}), and optimising
the aggregate rewards polishing what is already strong.

And rankings partly measure coverage. {{cite:liang2022helm}} found **17.9%** shared coverage
before a deliberate effort; here the reported ranking inverts on the **25%** of scenarios all
three models share, with the strongest model reported worst for running the hardest subset.

The uncomfortable synthesis is that these failures arrive together. An old benchmark is
simultaneously less able to detect a difference and more sensitive to which items were run,
so the leaderboards that have run longest — and therefore command the most trust — are the
ones whose rankings mean least. Trust in a benchmark accumulates on exactly the schedule its
information content decays on.

Carry forward: **compute the expiry date**, and **a score without a human baseline has no
units**.

## 21. Further Reading

- {{cite:liang2022helm}} — the coverage argument, and the measurement of what partial
  coverage had produced.
- {{cite:rein2023gpqa}} — expert and non-expert baselines under matched conditions, which is
  what gives a score its units.
- {{cite:singh2025leaderboard}} — the adversarial version of the coverage problem, with data
  shares and the gain purchasable with them.
- {{cite:hendrycks2020mmlu}} — the benchmark this chapter's arithmetic is most often applied
  to, and a calibration result that outlasted its accuracy figure.
- {{cite:card2020power}} — the power analysis every sample-size claim here rests on, applied
  to standard NLP evaluation practice.
