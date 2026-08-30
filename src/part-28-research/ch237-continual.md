---
id: res-continual
number: 237
part: XXVIII
tier: full
status: draft
requires: [evaluation-sets-decay-silently, rollback-restores-code-not-state,
           test-time-compute-has-a-ceiling-training-does-not,
           reproducibility-is-a-product-over-artefacts]
provides: [plasticity-and-retention-are-one-dial,
           update-cadence-has-an-interior-optimum,
           self-training-improves-only-the-verifiable-fraction,
           collapse-rate-is-set-by-the-real-data-fraction]
citations: [gama2014, kirkpatrick2017ewc, zelikman2022star, wang2023selfinstruct]
---

## 1. Learning Objectives

By the end of this chapter you will be able to show that plasticity and retention are a single
parameter rather than two objectives; compare continual-learning regularisers by what they keep
of each; compute an update cadence's total cost from staleness, update cost and regression risk,
and locate its interior optimum; derive a self-improvement loop's ceiling from the verifiable
fraction of the task; and show that mixing external data sets the *rate* of diversity
contraction rather than a floor.

## 2. Why This Matters

A deployed model faces a moving world ({{cite:gama2014}}), and the obvious response — keep
training it — degrades what it already knew ({{cite:kirkpatrick2017ewc}}).

Those are not two problems to balance. At plasticity 0.10 an update absorbs **0.221** of the new
distribution and loses **0.0028** of prior capability; at 4.00 it absorbs **1.000** and loses
**0.4029**. Both columns move together because they are the same mechanism
({{eq:plasticity-and-retention-are-one-dial}}).

That makes the operational question a cadence question, and cadence has an interior optimum.
Daily updates cost **$40.7 million** a year, every-60-days costs **$828 thousand**, annual costs
**$1.28 million** — the extremes are **49×** and **1.5×** the optimum
({{eq:update-cadence-has-an-interior-optimum}}).

Self-improvement loops have the same shape and a sharper ceiling. Net capability peaks at
**0.5996 at round 3** and falls to **0.4901** by round 8. The ceiling is set by the verifiable
fraction: **0.6400** at $v = 0.20$, **1.0000** at $v = 1.00$
({{eq:self-training-improves-only-the-verifiable-fraction}}).

And mixing in external data sets the contraction *rate*, not a floor: with none, diversity at
round 8 is **0.5596**; with 35%, **0.6890**; only fully external data avoids collapse
({{eq:collapse-rate-is-set-by-the-real-data-fraction}}).

## 3. Prerequisites

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is why the signal most
teams watch — the frozen evaluation set — correlates only **0.34** with the loss that matters.

{{eq:rollback-restores-code-not-state}} from {{ch:ops-deployment}} is the regression cost in the
cadence model: a bad update is not free to undo.

{{eq:test-time-compute-has-a-ceiling-training-does-not}} from {{ch:res-test-time}} is the same
verifier bound arriving through a different mechanism, and the two should be read together.

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is why a
self-improvement loop that has run eight rounds cannot be rewound to round three.

## 4. Intuitive Explanation

Start with the tension that defines the subject, and note that it is usually described wrongly.

The standard framing is "stability versus plasticity" — two desirable properties, in tension,
to be balanced. That framing suggests there is some clever mechanism that gives you both.

Look at the numbers. At plasticity 0.10, a 30-day update absorbs **0.221** of the new
distribution and loses **0.0028** of prior capability. At 0.50: **0.713** absorbed, **0.0243**
lost. At 1.00: 0.918 and 0.0620. At 4.00: **1.000** and **0.4029**.

Both columns move together, monotonically, because **they are the same mechanism**
({{eq:plasticity-and-retention-are-one-dial}}). A model that changes its weights readily in
response to new data changes them readily away from what the old data put there. There is no
setting that is plastic about new information and rigid about old information, because *the model
does not know which is which*. It sees gradients.

The net column — value gained minus value lost — has an interior maximum at plasticity **0.25**,
at **+0.0044**. The extremes are bad for opposite reasons: **+0.0038** at 0.10 because nothing
was learned, **−0.3730** at 4.00 because too much was lost.

The field's regularisers move along this dial rather than removing it. Each keeps some fraction
of absorption and blocks some fraction of forgetting.

No regularisation: **−0.0345**. Replay 5% of old data: **−0.0126**. Replay 25%: **+0.0030**.
An elastic weight penalty ({{cite:kirkpatrick2017ewc}}): **+0.0038**. Adapters with the base
frozen: **+0.0128**.

And retraining from scratch: **+0.0275** — keeping 1.00 of absorption and 0.00 of forgetting,
the best possible pair. **The clean answer exists and it is priced out of the loop**, which is
precisely why the rest of the table exists.

Worth noticing which row has quietly won in practice: adapters with the base frozen, keeping
only 0.05 of the forgetting for 0.58 of the absorption. Its cost is serving complexity — which
{{ch:res-test-time}} priced, and which is affordable exactly because adapters are small.

So the model-side question has an answer: pick a plasticity, use a regulariser, accept the
trade. The operational question is harder and it is the one teams actually get wrong.

**How often should you update?**

Two costs move in opposite directions. Staleness falls as updates get more frequent — the model
tracks a distribution that is moving under it. Update cost rises, and so does regression risk,
because a more frequent update is fitted to less data and carries more of the variance
{{ch:ops-deployment}} charges to any release.

Price it at a 90-day drift half-life. Daily updates: mean staleness 0.0006, regression risk
0.386, total **$40,718,365** a year. Weekly: **$5,210,527**. Every 30 days: **$1,120,586**.
Every 60: **$827,919**. Every 90: $866,315. Every 180: $1,082,262. Annual: **$1,277,519**.

**The cheapest cadence is every 60 days.** Daily updates cost **49×** that; annual updates
**1.5×** ({{eq:update-cadence-has-an-interior-optimum}}).

The two failure directions cost real money and they look nothing alike, which is why teams end
up on the wrong side of the optimum in both directions.

Updating too rarely shows up as a slow, uniform decline that nobody attributes to anything in
particular. No alert fires. The metrics that would show it are on a distribution that no longer
exists.

Updating too often shows up as capability regressions on things that used to work — visible,
alarming, and **blamed on the last change rather than on the cadence.** A team in that state
tightens its release process, which raises the cost per update and moves it further from the
optimum.

Now the part that is genuinely surprising. The optimum moves with the drift rate, and not in the
direction intuition says.

At a 365-day drift half-life the best cadence is every 90 days, costing $418,837. At a 90-day
half-life, every 60 days, $827,919. At a 30-day half-life, every 60 days. And at a **14-day**
half-life the best cadence is **every 365 days** at $1,356,000.

Read that last row again. When the world moves much faster than you can retrain, staleness is
already saturated at every cadence you can afford — so buying updates buys nothing, and you
should buy fewer of them. **Very fast drift is an argument against frequent retraining**, and
the correct response is a different architecture, not a shorter schedule.

Across the range, a fixed 30-day schedule carries a penalty of **1.14× to 2.21×**. A fixed
cadence is right for exactly one drift rate.

Which raises the question of how you would know your drift rate, and here the news is bad.

The quantity you need is loss on the *current* distribution. The only signal that measures it
directly — accuracy on a fresh labelled sample, correlation **0.91** — needs new labels and
arrives **14 days** late.

The signal everyone actually watches is the frozen evaluation set, correlating **0.34**, because
it measures accuracy on a distribution that stopped existing — {{ch:ops-prompt-versioning}}'s
{{eq:evaluation-sets-decay-silently}} arriving as a control-loop failure.

The interesting middle is label-free: input feature drift at **0.58** and output distribution
shift at **0.49**, both available immediately, neither sufficient alone. User-reported failures
correlate **0.72** but arrive 21 days late and on a biased sample.

**A continual system needs a drift estimate more than it needs a better update rule**, and the
drift estimate is much the cheaper thing to build.

That is continual learning from external data. The other half of this chapter is what happens
when the data comes from the model itself.

The loop is simple. Generate answers, keep the ones a verifier accepts, train on those, repeat
({{cite:zelikman2022star}}, {{cite:wang2023selfinstruct}}). It works, and it has two structural
failure modes.

Run it eight rounds. Verifiable competence rises monotonically: 0.5500, 0.6527, 0.7400, 0.8098,
0.8632, 0.9028, 0.9316, 0.9521, 0.9666. The loop genuinely works on the part the verifier can
judge.

Diversity falls monotonically: 1.0000 down to 0.5767, because each round trains on the previous
round's outputs.

Net capability is their product: 0.5500, 0.5824, 0.5985, **0.5996**, 0.5889, 0.5700, 0.5458,
0.5186, **0.4901**.

**It peaks at round 3 and then goes backwards.** The loop gains **+0.0496** and gives back
**0.1094**.

That is the first thing to take away. **A self-improvement loop is not a process that converges;
it is a process with an optimal number of rounds**, and running it longer is not the
conservative choice.

Now the ceiling on the rising factor, which is not 1. Competence converges to
$v + (1-v)p_0$: the verifiable fraction improves and the rest stays exactly where it started,
because it receives no signal at all
({{eq:self-training-improves-only-the-verifiable-fraction}}).

At a verifiable fraction of 0.20 the ceiling is **0.6400**. At 0.45, 0.7525. At 0.72, 0.8740. At
0.90, 0.9550. At 1.00, 1.0000.

That is the same shape {{ch:res-test-time}} found for sampling, and the pair is worth stating
together: **both ways of using a verifier to improve a model are bounded by the verifier, in
different ways.** Sampling is bounded by how well the verifier *ranks*. Self-training is bounded
by how much of the task it can *adjudicate at all*.

The second failure mode is the diversity contraction, and its mechanism matters more than its
numbers. Diversity is multiplied each round by $r + (1-r)\kappa$, where $r$ is the fraction of
genuinely external data and $\kappa < 1$ is what self-generated data retains. That multiplier is
below 1 for every $r$ short of 100%.

**So the external-data fraction sets the rate of contraction, not a floor**
({{eq:collapse-rate-is-set-by-the-real-data-fraction}}). Mixing in real data buys rounds. With
no external data, diversity at round 8 is 0.5596 and net capability at round 20 has fallen to
**0.2047**. With 35%: 0.6890 and 0.3443. With 70%: 0.8438 and 0.5715.

The 100% row is the control, and it is not a self-improvement loop at all — it is ordinary
training on external data, and it is the only row that improves monotonically all the way to
round 20, reaching **0.8737**.

That reframes what the technique is for. It converts compute into capability *for a bounded
number of rounds* when external data is scarce. It is not a way to stop needing external data.

Task by task, the available gain differs enormously. Arithmetic with a checker: verifiable
fraction 0.99, ceiling **0.9955**, gain available **0.4455**. Code with a test suite: 0.86,
0.9370, 0.3870. Factual QA with retrieval: 0.71, 0.8695, 0.3195. Summarisation: 0.38, 0.7210,
0.1710. Open-ended advice: 0.19, 0.6355, 0.0855. Creative writing: 0.08, 0.5860, **0.0360**.

**A spread of 12.4× across the same technique.** Self-improvement is a method for verifiable
tasks and a distribution-narrowing procedure for everything else, and the same code produces
both outcomes.

Finally, the cost of running it. Round 1 delivers +0.0324 at **$1,909** per 0.001 of gain. Round
2, +0.0160 at $3,861. Round 3, **+0.0011 at $56,445**. Rounds 4 and 5 are negative, at the same
$61,912 a round.

**The loop has a stopping rule and it is not "when it stops improving"** — because by the time
the measured metric turns over, diversity has contracted past the point where the earlier rounds
could be reproduced, which is {{ch:ops-versioning}}'s
{{eq:reproducibility-is-a-product-over-artefacts}} with the artefact being the data
distribution itself. The stopping round has to be chosen in advance, from the ceiling and the
mixture ratio, both of which are computable before the first round runs.

## 5. Formal Explanation

**One dial.** Let $\pi$ index how strongly an update moves weights. Absorption
$A(\pi) = 1 - e^{-\pi t/\tau}$ is increasing and concave; forgetting $F(\pi) = \phi \pi^{\beta}$
with $\beta > 1$ is increasing and convex. Net value $S \cdot A(\pi) - F(\pi)$ therefore has a
unique interior maximum for any staleness $S > 0$ — and both terms depend on the *same* $\pi$,
which is the formal content of "one dial".

**Cadence.** Annual cost is
$C(d) = \bar{S}(d)\,V + \frac{365}{d}\left(u + \rho(d)\,g\right)$ where $\bar S$ is mean
staleness (increasing in $d$), $u$ the update cost, $\rho(d)$ regression risk (decreasing in
$d$), $g$ the regression cost. The first term increases in $d$ and the second decreases, so
$C$ has an interior minimum whenever $\bar S$ is not already saturated. When drift is fast
enough that $\bar S(d) \approx \bar S_{\max}$ for all affordable $d$, the first term is flat and
the minimum moves to the largest $d$ — which is the fast-drift row.

**The verifiable fraction.** Partition the task into a verifiable part of measure $v$ and its
complement. Self-training supplies gradient signal only on the first, so
$p_\infty = v \cdot p_v^\star + (1-v)p_0$, and $p_v^\star \le 1$ regardless of rounds. The
ceiling is therefore $v + (1-v)p_0$ at best, independent of the loop's schedule.

**Contraction.** Diversity obeys $c_{t+1} = c_t(r + (1-r)\kappa)$, a geometric sequence with
ratio strictly less than 1 for $r < 1$ and $\kappa < 1$. Hence $c_t \to 0$ for any $r < 1$: the
mixture ratio sets the decay constant, and only $r = 1$ has a non-zero limit.

## 6. Mathematical Foundation

Plasticity and retention share a parameter:

$$\max_\pi \left[S \cdot A(\pi) - F(\pi)\right], \qquad A(0.10) = 0.221,\ F(0.10) = 0.0028; \quad A(4.00) = 1.000,\ F(4.00) = 0.4029$$ (eq:plasticity-and-retention-are-one-dial)

with the net maximised at $\pi = 0.25$.

Cadence trades staleness against update cost and regression risk:

$$C(d) = \bar S(d)\,V + \frac{365}{d}\left(u + \rho(d)g\right), \qquad d^\star = 60 \ \text{days},\ \$827{,}919$$ (eq:update-cadence-has-an-interior-optimum)

against **$40.7M** daily and **$1.28M** annually.

A self-training loop improves only what can be checked:

$$p_\infty = v + (1-v)p_0 = 0.6400 \ (v = 0.20) \ \to \ 1.0000 \ (v = 1.00)$$ (eq:self-training-improves-only-the-verifiable-fraction)

And diversity contracts geometrically:

$$c_t = c_0\left(r + (1-r)\kappa\right)^t \to 0 \ \text{for all } r < 1$$ (eq:collapse-rate-is-set-by-the-real-data-fraction)

## 7. Internal Mechanics

Why can't a mechanism be plastic about new information and rigid about old? Because "old" and
"new" are not properties the model can read off a gradient. Every regulariser in
{{sec:9-practical-example}} works by proxy — replay reintroduces old examples so the gradient
sees both; an elastic penalty estimates which weights mattered before and resists moving them;
adapters keep the old weights literally unchanged and put the new information somewhere else.
Each proxy is imperfect in a specific way, and the imperfection is exactly the absorption it
also blocks.

The adapters row is worth understanding as the limiting case. It gets forgetting almost to zero
by not touching the base at all, which means the new information has to fit in the adapter's
capacity — hence 0.58 absorption rather than 1.00. It converts a plasticity problem into a
capacity problem, and capacity problems have clean engineering answers where plasticity problems
do not.

The cadence result's most useful mechanism is the regression-risk term, because it is the one
people leave out. A less frequent update is fitted to more data, which is less noisy, which
regresses less. Teams that model cadence at all typically model staleness against update cost
and get a monotone answer pointing at "as often as possible" — and then discover the regression
cost empirically, one incident at a time.

The fast-drift inversion has a mechanism worth stating plainly, because it looks like a bug. If
the distribution's half-life is 14 days and your fastest affordable cadence is a week, then the
model is always fitted to a distribution roughly half of which has already moved. Halving the
interval barely changes that, so the staleness term is effectively constant and only the update
costs vary — and the cheapest constant-staleness option is the cheapest cadence.

The self-training ceiling has the cleanest mechanism in the chapter and it generalises. The loop
is a control system whose only feedback is the verifier's signal. Regions the verifier cannot
adjudicate are outside the loop entirely — not poorly served, *not served*. This is why the
ceiling is a partition rather than a rate: $1 - v$ of the task receives literally zero gradient
from the procedure, however many rounds run.

Finally, the contraction and the ceiling interact in a way that makes the peak earlier than
either predicts alone. Competence rises fastest in the early rounds, when precision gains are
largest; diversity falls at a constant rate. So the product peaks where the marginal competence
gain equals the constant fractional diversity loss — which, because the competence curve is
concave, happens early. **The optimal number of rounds is small for structural reasons**, and it
gets smaller as the verifiable fraction falls.

## 8. Implementation

The first listing prices continual updating.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ke1}
"""Retention and plasticity are one dial, and the update cadence has an interior optimum.

A deployed model faces a moving world (cite:gama2014). The obvious response is to keep training
it on new data. The obvious problem is that training on new data degrades what it knew
(cite:kirkpatrick2017ewc).

Those are not two problems to be balanced. They are one quantity read in two directions: any
mechanism that makes the model absorb new information faster makes it discard old information
faster, and the regularisation that protects the old blocks the new
(eq:plasticity-and-retention-are-one-dial).

Which turns the operational question into a cadence question. Update rarely and the model is
stale; update often and it forgets and oscillates. Both costs are monotone in opposite
directions, so there is an interior optimum
(eq:update-cadence-has-an-interior-optimum).
"""
import math

DRIFT_HALF_LIFE = 90.0     # days for half the distribution to move
BASE_ACC = 0.880


def staleness(days):
    """Accuracy lost to a distribution that has moved since the last update."""
    return 0.145 * (1.0 - 0.5 ** (days / DRIFT_HALF_LIFE))


def absorbed(plasticity, days):
    """Share of the new distribution the update actually takes on."""
    return 1.0 - math.exp(-plasticity * days / 12.0)


def forgotten(plasticity):
    """Share of prior capability lost per update, at a given plasticity."""
    return 0.062 * plasticity ** 1.35


print("The dial: one parameter, two effects.")
print()
print(f"{'plasticity':>13}{'new material absorbed':>24}{'prior capability lost':>24}"
      f"{'net after one update':>23}")
print("-" * 84)
DAYS = 30
dial = {}
for p in (0.10, 0.25, 0.50, 1.00, 2.00, 4.00):
    a = absorbed(p, DAYS)
    f = forgotten(p)
    net = staleness(DAYS) * a - f
    dial[p] = (a, f, net)
    print(f"{p:>13.2f}{a:>24.3f}{f:>24.4f}{net:>23.4f}")

BEST_P = max(dial, key=lambda p: dial[p][2])
print()
print(f"best plasticity at a {DAYS}-day cadence: {BEST_P:.2f}, net {dial[BEST_P][2]:+.4f}")
print(f"the extremes: {dial[0.10][2]:+.4f} at 0.10, {dial[4.00][2]:+.4f} at 4.00")

print()
print()
print("Regularisation moves along the dial; it does not remove it.")
print()
METHODS = [
    ("no regularisation",         1.00, 1.00, "--"),
    ("replay 5% old data",        0.94, 0.62, "storage + a pass"),
    ("replay 25% old data",       0.81, 0.31, "4x the storage"),
    ("elastic weight penalty",    0.77, 0.28, "cite:kirkpatrick2017ewc"),
    ("adapters, base frozen",     0.58, 0.05, "serving complexity"),
    ("retrain from scratch",      1.00, 0.00, "the full training cost"),
]
print(f"{'method':>26}{'absorption kept':>18}{'forgetting kept':>18}"
      f"{'net at plasticity 1.0':>24}{'what it costs':>26}")
print("-" * 112)
meth = {}
BASE_A, BASE_F = absorbed(1.0, DAYS), forgotten(1.0)
for name, keep_a, keep_f, cost in METHODS:
    a, f = BASE_A * keep_a, BASE_F * keep_f
    net = staleness(DAYS) * a - f
    meth[name] = (a, f, net)
    print(f"{name:>26}{keep_a:>18.2f}{keep_f:>18.2f}{net:>24.4f}{cost:>26}")

BEST_M = max(meth, key=lambda n: meth[n][2])
print()
print(f"best net: {BEST_M} at {meth[BEST_M][2]:+.4f}")
print(f"against {meth['no regularisation'][2]:+.4f} unregularised")

print()
print()
print("Now cadence, where the two costs are monotone and opposite.")
print()
UPDATE_COST = 42_000.0
REGRESSION_COST = 180_000.0
ACC_VALUE = 18_000_000.0      # dollars per unit of accuracy per year


def regression_risk(days):
    """Smaller, more frequent updates carry noisier data and more regressions."""
    return 0.35 * math.exp(-days / 25.0) + 0.05


print(f"{'days between updates':>22}{'updates / year':>16}{'mean staleness':>17}"
      f"{'regression risk':>18}{'staleness cost':>17}{'update cost':>14}"
      f"{'total':>14}")
print("-" * 118)
cad = {}
for days in (1, 7, 14, 30, 60, 90, 180, 365):
    n = 365.0 / days
    mean_stale = staleness(days) / 2.0
    risk = regression_risk(days)
    stale_cost = mean_stale * ACC_VALUE
    upd_cost = n * (UPDATE_COST + risk * REGRESSION_COST)
    cad[days] = (mean_stale, risk, stale_cost, upd_cost, stale_cost + upd_cost)
    print(f"{days:>22}{n:>16.1f}{mean_stale:>17.4f}{risk:>18.3f}"
          f"{stale_cost:>17,.0f}{upd_cost:>14,.0f}{stale_cost + upd_cost:>14,.0f}")

BEST_D = min(cad, key=lambda d: cad[d][4])
print()
print(f"cheapest cadence: every {BEST_D} days at {cad[BEST_D][4]:,.0f} per year")
print(f"daily updates cost {cad[1][4] / cad[BEST_D][4]:.1f}x that;"
      f" annual ones {cad[365][4] / cad[BEST_D][4]:.1f}x")
print("(eq:update-cadence-has-an-interior-optimum)")

print()
print()
print("And the optimum moves with how fast the world moves.")
print()
print(f"{'drift half-life (days)':>24}{'best cadence (days)':>22}"
      f"{'annual cost':>16}{'cost at a 30-day cadence':>27}{'penalty':>11}")
print("-" * 100)
CANDIDATES = (1, 3, 7, 14, 30, 60, 90, 180, 365)
for half in (14, 30, 90, 180, 365):
    best, best_c, at30 = None, None, None
    for days in CANDIDATES:
        stale = 0.145 * (1.0 - 0.5 ** (days / half)) / 2.0
        n = 365.0 / days
        total = stale * ACC_VALUE + n * (UPDATE_COST
                                         + regression_risk(days) * REGRESSION_COST)
        if days == 30:
            at30 = total
        if best_c is None or total < best_c:
            best, best_c = days, total
    print(f"{half:>24}{best:>22}{best_c:>16,.0f}{at30:>27,.0f}{at30 / best_c:>10.2f}x")

print()
print("A fixed cadence is right for exactly one drift rate.")

print()
print()
print("What you can actually measure, and what it would tell you.")
print()
SIGNALS = [
    ("accuracy on a fresh labelled sample", 0.91, 14, "labels, delayed"),
    ("accuracy on the frozen eval set",     0.34, 0,  "ch:ops-prompt-versioning"),
    ("input feature drift",                 0.58, 0,  "no labels needed"),
    ("output distribution shift",           0.49, 0,  "no labels needed"),
    ("user-reported failures",              0.72, 21, "biased sample"),
    ("disagreement with a held-out model",  0.66, 1,  "needs a second model"),
]
print(f"{'signal':>38}{'correlation with true loss':>29}{'lag (days)':>13}"
      f"{'catch':>26}")
print("-" * 106)
for name, corr, lag, note in SIGNALS:
    print(f"{name:>38}{corr:>29.2f}{lag:>13}{note:>26}")

best_sig = max(SIGNALS, key=lambda s: s[1] / (1 + s[2] / 30.0))
print()
print(f"best signal per unit of lag: {best_sig[0]}")
print(f"the frozen evaluation set correlates {0.34:.2f} -- it is the one everyone watches")

print(f"""
The dial table is the framing this listing exists for. At plasticity {0.10:.2f} an update absorbs
{dial[0.10][0]:.3f} of the new distribution and loses {dial[0.10][1]:.4f} of prior capability.
At {4.00:.2f} it absorbs {dial[4.00][0]:.3f} and loses {dial[4.00][1]:.4f}.

Both columns move together because they are the same mechanism
(eq:plasticity-and-retention-are-one-dial). A model that changes its weights readily in response
to new data changes them readily away from what the old data put there. There is no setting that
is plastic about the new and rigid about the old, because the model does not know which is which.

The net column has an interior maximum at plasticity {BEST_P:.2f} -- and note that the extremes
are bad for opposite reasons: {dial[0.10][2]:+.4f} because nothing was learned,
{dial[4.00][2]:+.4f} because too much was lost.

The regularisation table is what the field has built to move along that dial. Every row keeps
some absorption and blocks some forgetting, and the ratio is the whole story.
`{BEST_M}` reaches {meth[BEST_M][2]:+.4f} against
{meth['no regularisation'][2]:+.4f} unregularised.

`retrain from scratch` is worth reading carefully: it keeps {1.00:.2f} absorption and
{0.00:.2f} forgetting, which is the best possible pair, and the last column says what that
costs. **The clean answer exists and it is priced out of the loop** -- which is why the rest of
the table exists.

`adapters, base frozen` is the row that has quietly won in practice, keeping
{0.05:.2f} of the forgetting for {0.58:.2f} of the absorption, and its cost is the serving
complexity ch:res-test-time priced.

The cadence table is the operational result (eq:update-cadence-has-an-interior-optimum).
Staleness falls as updates get more frequent; update cost and regression risk both rise, because
a more frequent update is fitted to less data and carries more of the variance
ch:ops-deployment charged to releases.

The cheapest cadence is **every {BEST_D} days at {cad[BEST_D][4]:,.0f} per year**. Daily updates
cost **{cad[1][4] / cad[BEST_D][4]:.0f} times** that and annual updates
{cad[365][4] / cad[BEST_D][4]:.1f} times.

Both failure directions are expensive and they look completely different. Updating too rarely
shows up as a slow, uniform decline that nobody attributes to anything. Updating too often shows
up as capability regressions on things that used to work -- **visible, alarming, and blamed on
the last change rather than on the cadence.**

The drift table says the optimum is not a constant, and it does not move the way intuition
says. A fixed 30-day schedule carries a penalty of between 1.14x and 2.21x depending on the
drift rate, and the best cadence at a {14}-day half-life is *slower* than at a {30}-day one.

That is worth pausing on. When the world moves much faster than you can retrain, staleness is
saturated at every cadence you can afford, so buying updates buys nothing and you should buy
fewer of them. **Very fast drift is an argument against frequent retraining**, not for it -- and
the correct response there is a different architecture, not a shorter schedule.

**Nobody measures their drift half-life**, and it is the parameter that sets the schedule -- and,
per the row above, sometimes tells you the schedule is not the lever.

The signals table is why. The quantity you need is loss on the current distribution, and the
only signal that measures it directly -- {0.91:.2f} correlation -- needs fresh labels and arrives
{14} days late. The signal everyone actually watches is the frozen evaluation set, which
correlates **{0.34:.2f}**, because it measures accuracy on a distribution that stopped existing
(ch:ops-prompt-versioning's `evaluation-sets-decay-silently`).

The label-free signals are the interesting middle: input drift at {0.58:.2f} and output shift at
{0.49:.2f}, both available immediately and neither sufficient alone. **A continual system needs a
drift estimate more than it needs a better update rule**, and the drift estimate is the cheaper
thing to build.""")
```

## 9. Practical Example

One dial, two effects:

```
   plasticity   new material absorbed   prior capability lost   net after one update
------------------------------------------------------------------------------------
         0.10                   0.221                  0.0028                 0.0038
         0.25                   0.465                  0.0095                 0.0044
         0.50                   0.713                  0.0243                -0.0030
         1.00                   0.918                  0.0620                -0.0345
         4.00                   1.000                  0.4029                -0.3730
```

**The net peaks at plasticity 0.25** ({{eq:plasticity-and-retention-are-one-dial}}).

```
                    method   absorption kept   forgetting kept   net at plasticity 1.0             what it costs
----------------------------------------------------------------------------------------------------------------
         no regularisation              1.00              1.00                 -0.0345                        --
       replay 25% old data              0.81              0.31                  0.0030            4x the storage
    elastic weight penalty              0.77              0.28                  0.0038   cite:kirkpatrick2017ewc
     adapters, base frozen              0.58              0.05                  0.0128        serving complexity
      retrain from scratch              1.00              0.00                  0.0275    the full training cost
```

**The clean answer exists and is priced out of the loop.**

```
  days between updates  updates / year   mean staleness   regression risk   staleness cost   update cost         total
----------------------------------------------------------------------------------------------------------------------
                     1           365.0           0.0006             0.386           10,012    40,708,353    40,718,365
                     7            52.1           0.0038             0.315           68,492     5,142,035     5,210,527
                    30            12.2           0.0150             0.155          269,221       851,365     1,120,586
                    60             6.1           0.0268             0.082          482,902       345,018       827,919
                    90             4.1           0.0362             0.060          652,500       213,815       866,315
                   365             1.0           0.0681             0.050        1,226,519        51,000     1,277,519
```

**Every 60 days, at $827,919** — daily is 49× that
({{eq:update-cadence-has-an-interior-optimum}}).

```
  drift half-life (days)   best cadence (days)     annual cost   cost at a 30-day cadence    penalty
----------------------------------------------------------------------------------------------------
                      14                   365       1,356,000                  1,860,873      1.37x
                      30                    60       1,323,768                  1,503,865      1.14x
                      90                    60         827,919                  1,120,586      1.35x
                     365                    90         418,837                    923,634      2.21x

                                signal   correlation with true loss   lag (days)                     catch
----------------------------------------------------------------------------------------------------------
  accuracy on a fresh labelled sample                         0.91           14           labels, delayed
      accuracy on the frozen eval set                         0.34            0  ch:ops-prompt-versioning
                  input feature drift                         0.58            0          no labels needed
            user-reported failures                            0.72           21            biased sample
```

**Very fast drift argues against frequent retraining**, and the signal everyone watches
correlates 0.34.

The second listing prices self-improvement.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ke2}
"""A self-improvement loop improves the part it can check, and shrinks the part it can see.

Generate answers, keep the ones a verifier accepts, train on those, repeat (cite:zelikman2022star,
cite:wang2023selfinstruct). It works, and both of its failure modes are structural rather than
incidental.

The first: only the verifiable fraction of the task improves. Whatever the verifier cannot
adjudicate receives no signal at all and stays exactly where it started, so the loop's ceiling is
set by verifiability rather than by the number of rounds
(eq:self-training-improves-only-the-verifiable-fraction).

The second: each round trains on the previous round's outputs, so the training distribution
drifts toward the model's own. Diversity contracts geometrically, at a rate set by the fraction
of genuinely external data mixed back in
(eq:collapse-rate-is-set-by-the-real-data-fraction).
"""
V_FRACTION = 0.72        # share of the task a verifier can adjudicate
P0 = 0.550               # starting competence
SENS, SPEC, RHO = 0.95, 0.90, 0.35    # verifier sensitivity, specificity, shared-error rate
LAMBDA = 0.55            # how much of the accepted-set precision a round absorbs
KAPPA = 0.93             # diversity retained when training on own outputs
ROUNDS = 8


def precision(p):
    """Share of the accepted set that is actually correct."""
    accept_wrong = RHO + (1.0 - RHO) * (1.0 - SPEC)
    return SENS * p / (SENS * p + accept_wrong * (1.0 - p))


def run(real_fraction, rounds=ROUNDS, v=V_FRACTION):
    pv, cov = P0, 1.0
    out = []
    for r in range(rounds + 1):
        p = v * pv + (1.0 - v) * P0
        out.append((r, pv, p, cov, p * cov))
        pv = pv + LAMBDA * (precision(pv) - pv)
        cov = cov * (real_fraction + (1.0 - real_fraction) * KAPPA)
    return out


print("Eight rounds of a self-improvement loop.")
print()
REAL = 0.05
print(f"verifiable fraction {V_FRACTION:.2f}, real data mixed back in {REAL:.0%}")
print()
print(f"{'round':>8}{'verifiable competence':>24}{'overall competence':>21}"
      f"{'diversity retained':>21}{'net capability':>17}")
print("-" * 91)
base = run(REAL)
for r, pv, p, cov, net in base:
    print(f"{r:>8}{pv:>24.4f}{p:>21.4f}{cov:>21.4f}{net:>17.4f}")

PEAK = max(base, key=lambda x: x[4])
print()
print(f"peak net capability at round {PEAK[0]}: {PEAK[4]:.4f}")
print(f"round 0: {base[0][4]:.4f}; round {ROUNDS}: {base[-1][4]:.4f}")
print(f"the loop gains {PEAK[4] - base[0][4]:+.4f}, then gives back"
      f" {PEAK[4] - base[-1][4]:.4f}")

print()
print()
print("The ceiling is verifiability, and no number of rounds moves it.")
print()
print(f"{'verifiable fraction':>21}{'competence ceiling':>21}{'peak net':>12}"
      f"{'peak round':>13}{'gain over round 0':>21}")
print("-" * 88)
ceil = {}
for v in (0.20, 0.45, 0.72, 0.90, 1.00):
    series = run(REAL, rounds=40, v=v)
    top = max(series, key=lambda x: x[4])
    limit = v * 1.0 + (1.0 - v) * P0
    ceil[v] = (limit, top[4], top[0])
    print(f"{v:>21.2f}{limit:>21.4f}{top[4]:>12.4f}{top[0]:>13}"
          f"{top[4] - series[0][4]:>21.4f}")

print()
print(f"at verifiable fraction {0.20:.2f} the ceiling is {ceil[0.20][0]:.4f};"
      f" at {1.00:.2f} it is {ceil[1.00][0]:.4f}")
print("(eq:self-training-improves-only-the-verifiable-fraction)")

print()
print()
print("And diversity contracts at a rate set by the external data fraction.")
print()
print(f"{'real data mixed in':>21}{'diversity at round 8':>23}{'peak net':>12}"
      f"{'peak round':>13}{'net at round 20':>19}")
print("-" * 88)
mix = {}
for r_frac in (0.00, 0.05, 0.15, 0.35, 0.70, 1.00):
    series = run(r_frac, rounds=20)
    top = max(series, key=lambda x: x[4])
    mix[r_frac] = (series[8][3], top[4], top[0], series[20][4])
    print(f"{r_frac:>21.0%}{series[8][3]:>23.4f}{top[4]:>12.4f}{top[0]:>13}"
          f"{series[20][4]:>19.4f}")

print()
print(f"with no external data, diversity at round 8 is {mix[0.00][0]:.4f}"
      f" and net at round 20 is {mix[0.00][3]:.4f}")
print(f"with {0.35:.0%} external data: {mix[0.35][0]:.4f} and {mix[0.35][3]:.4f}")
print("(eq:collapse-rate-is-set-by-the-real-data-fraction)")

print()
print()
print("What that means task by task.")
print()
TASKS = [
    ("arithmetic with a checker",         0.99, "an exact answer"),
    ("code with a test suite",            0.86, "tests are partial"),
    ("factual QA with retrieval",         0.71, "sources may disagree"),
    ("summarisation",                     0.38, "no reference exists"),
    ("open-ended advice",                 0.19, "the criterion is contested"),
    ("creative writing",                  0.08, "quality is a preference"),
]
print(f"{'task':>32}{'verifiable fraction':>22}{'loop ceiling':>15}"
      f"{'gain available':>17}{'why':>30}")
print("-" * 116)
for name, v, why in TASKS:
    limit = v + (1.0 - v) * P0
    print(f"{name:>32}{v:>22.2f}{limit:>15.4f}{limit - P0:>17.4f}{why:>30}")

print()
print(f"the spread of available gain across these tasks is"
      f" {(0.99 + 0.01 * P0 - P0) / (0.08 + 0.92 * P0 - P0):.1f}x")

print()
print()
print("And what the loop costs to run at each round.")
print()
GEN_COST, TRAIN_COST = 0.00038, 61_000.0
ITEMS = 400_000
print(f"{'round':>8}{'items generated':>18}{'accepted':>12}{'generation $':>15}"
      f"{'training $':>14}{'net gain':>12}{'$ per 0.001 gain':>19}")
print("-" * 98)
prev_net = None
total = 0.0
for r, pv, p, cov, net in base[:6]:
    kept = ITEMS * (SENS * pv + (RHO + (1 - RHO) * (1 - SPEC)) * (1 - pv))
    gen = ITEMS * 6 * GEN_COST
    cost = gen + TRAIN_COST
    total += cost
    if prev_net is None:
        print(f"{r:>8}{'--':>18}{'--':>12}{'--':>15}{'--':>14}"
              f"{'--':>12}{'--':>19}")
    else:
        d = net - prev_net
        per = cost / (d / 0.001) if d > 0 else float("inf")
        ps = f"{per:>19,.0f}" if d > 0 else f"{'negative':>19}"
        print(f"{r:>8}{ITEMS:>18,}{kept:>12,.0f}{gen:>15,.0f}"
              f"{TRAIN_COST:>14,.0f}{d:>+12.4f}{ps}")
    prev_net = net

print()
print(f"total for {5} rounds: {total:,.0f}")

print(f"""
The first table is a self-improvement loop behaving exactly as advertised, and then not. Net
capability rises from {base[0][4]:.4f} to a peak of **{PEAK[4]:.4f} at round {PEAK[0]}**, then
falls to {base[-1][4]:.4f} by round {ROUNDS}.

Two separate things are happening in those columns and they are worth reading apart. Verifiable
competence rises monotonically -- the loop genuinely works on the part the verifier can judge.
Diversity falls monotonically, because each round trains on the last round's outputs. The net is
their product, and a rising bounded factor times a falling one has a peak.

**A self-improvement loop is not a process that converges. It is a process with an optimal number
of rounds**, and running it longer is not conservative.

The ceiling table says how high the rising factor can go, and the answer is not "1". Competence
converges to `v + (1-v)p0` -- the verifiable fraction improves, the rest stays exactly where it
started (eq:self-training-improves-only-the-verifiable-fraction). At a verifiable fraction of
{0.20:.2f} the ceiling is {ceil[0.20][0]:.4f}; at {0.90:.2f} it is {ceil[0.90][0]:.4f}.

That is the same structure ch:res-test-time found for sampling and it is worth stating together:
**both ways of using a verifier to improve a model are bounded by the verifier, in different
ways.** Sampling is bounded by how well the verifier ranks; self-training is bounded by how much
of the task it can adjudicate at all.

The mixture table is the second failure mode
(eq:collapse-rate-is-set-by-the-real-data-fraction). With no external data, diversity at round 8
is {mix[0.00][0]:.4f} and net capability at round 20 has fallen to {mix[0.00][3]:.4f}. With
{0.35:.0%} external data, {mix[0.35][0]:.4f} and {mix[0.35][3]:.4f}.

Read the mechanism rather than the rows. Diversity is multiplied each round by
`r + (1-r)*kappa`, which is below 1 for every `r` short of {1.00:.0%}. **The external-data
fraction sets the rate of contraction, not a floor** -- mixing in real data buys rounds, and only
data that is entirely external avoids collapse in the limit
(eq:collapse-rate-is-set-by-the-real-data-fraction).

The {1.00:.0%} row is the control, and it is not a self-improvement loop at all: it is ordinary
training on external data, and it is the only row that improves monotonically to round 20.

That reframes what the technique is for. It is a way to convert compute into capability *for a
bounded number of rounds* when external data is scarce -- not a way to escape needing it. The
right quantity to control is the mixture ratio and the right thing to monitor is diversity
itself, and neither appears on a standard training dashboard.

The task table converts this into a decision. `arithmetic with a checker` has a verifiable
fraction of {0.99:.2f} and a ceiling of {0.99 + 0.01 * P0:.4f}. `creative writing` has
{0.08:.2f} and {0.08 + 0.92 * P0:.4f}. The available gain differs by
**{(0.99 + 0.01 * P0 - P0) / (0.08 + 0.92 * P0 - P0):.1f}x** across the same technique.

**Self-improvement is a technique for verifiable tasks and a distribution-narrowing procedure
for everything else**, and the same code produces both outcomes.

The cost table is the last practical point. The gain per round falls sharply while the cost per
round is constant, so the dollars per unit of improvement rise every round -- and eventually the
gain turns negative while the bill does not.

**The loop has a stopping rule and it is not "when it stops improving"**, because by the time the
measured metric turns over, diversity has already contracted past the point where the earlier
rounds could be reproduced. The stopping rule has to be set in advance from the ceiling and the
mixture ratio, which are both computable before the first round runs.""")
```

```
   round   verifiable competence   overall competence   diversity retained   net capability
-------------------------------------------------------------------------------------------
       0                  0.5500               0.5500               1.0000           0.5500
       2                  0.7400               0.6868               0.8714           0.5985
       3                  0.8098               0.7370               0.8135           0.5996
       5                  0.9028               0.8040               0.7089           0.5700
       8                  0.9666               0.8500               0.5767           0.4901
```

**Peaks at round 3 and goes backwards** — gains +0.0496, gives back 0.1094.

```
  verifiable fraction   competence ceiling    peak net   peak round    gain over round 0
----------------------------------------------------------------------------------------
                 0.20               0.6400      0.5500            0               0.0000
                 0.45               0.7525      0.5566            1               0.0066
                 0.72               0.8740      0.5996            3               0.0496
                 0.90               0.9550      0.6376            3               0.0876
                 1.00               1.0000      0.6587            3               0.1087
```

({{eq:self-training-improves-only-the-verifiable-fraction}})

```
   real data mixed in   diversity at round 8    peak net   peak round    net at round 20
----------------------------------------------------------------------------------------
                   0%                 0.5596      0.5940            2             0.2047
                  15%                 0.6122      0.6132            3             0.2562
                  35%                 0.6890      0.6437            4             0.3443
                  70%                 0.8438      0.7261            6             0.5715
                 100%                 1.0000      0.8737           20             0.8737
```

**External data sets the rate, not a floor**
({{eq:collapse-rate-is-set-by-the-real-data-fraction}}).

```
                            task   verifiable fraction   loop ceiling   gain available                           why
--------------------------------------------------------------------------------------------------------------------
       arithmetic with a checker                  0.99         0.9955           0.4455               an exact answer
          code with a test suite                  0.86         0.9370           0.3870             tests are partial
                   summarisation                  0.38         0.7210           0.1710           no reference exists
                creative writing                  0.08         0.5860           0.0360       quality is a preference

   round   items generated    accepted   generation $    training $    net gain   $ per 0.001 gain
--------------------------------------------------------------------------------------------------
       1           400,000     305,674            912        61,000     +0.0324              1,909
       2           400,000     324,353            912        61,000     +0.0160              3,861
       3           400,000     339,295            912        61,000     +0.0011             56,445
       4           400,000     350,730            912        61,000     -0.0107           negative
```

**A 12.4× spread in available gain**, and the cost per unit rises 30× in three rounds.

## 10. Production Considerations

Measure your drift half-life before choosing a cadence. It is the parameter that sets the
schedule, and sometimes it tells you the schedule is not the lever.

Build a label-free drift signal. Input drift at 0.58 correlation and zero lag beats a frozen
evaluation set at 0.34 for a control loop.

Model regression risk in the cadence decision. Leaving it out gives a monotone answer pointing
at "as often as possible", which is 49× the optimum here.

Prefer adapters with a frozen base where serving allows it. 0.05 of the forgetting for 0.58 of
the absorption, and the serving cost is affordable when adapters are small.

Choose a self-improvement loop's round count in advance, from the verifiable fraction and the
mixture ratio. By the time the metric turns over you cannot get back.

Measure the verifiable fraction of your task before running the loop. Below about 0.45 there is
almost nothing to gain.

Monitor diversity, not just quality, on every self-training round. Quality peaks three rounds
before the composite does.

Keep a snapshot of every round's data mixture. {{eq:reproducibility-is-a-product-over-artefacts}}
applies to distributions, and round 3 cannot be recovered from round 8.

## 11. Common Mistakes

**Treating stability and plasticity as two knobs.** They are one.

**Optimising cadence against staleness and update cost alone.** Omitting regression risk points
at daily.

**Reading fast drift as a reason to update more.** At a 14-day half-life the best cadence is
annual.

**Watching a frozen evaluation set for drift.** Correlation 0.34 with the loss that matters.

**Running a self-improvement loop until it stops improving.** It peaks at round 3 and the peak is
not visible until after it.

**Applying self-training to unverifiable tasks.** Ceiling 0.5860 for creative writing against
0.9955 for checked arithmetic.

**Mixing in real data and assuming collapse is solved.** It sets the rate; every $r < 1$ still
goes to zero.

## 12. Failure Modes

**A model that quietly decays over a year.** No alert, no incident, and the evaluation set says
it is fine.

**A weekly retrain that keeps breaking things.** Regression risk 0.315 per update, blamed on the
last change.

**A self-improvement loop at round 12.** Verifiable competence 0.97, diversity 0.42, and users
noticing the outputs all sound the same.

**A loop applied to summarisation.** 0.1710 of gain available and the same compute bill as
arithmetic's 0.4455.

**A self-trained model that cannot be rolled back to a good round.** The data distribution is not
in version control.

**A cadence chosen once and never revisited.** 1.14× to 2.21× penalty as drift changes.

## 13. Alternatives

**Retrain from scratch on a schedule.** The best absorption/forgetting pair in the table at
+0.0275, and priced out for most teams — but the right answer whenever the training cost is
small relative to the accumulated regression cost.

**Adapters with a frozen base.** +0.0128 with almost no forgetting, and the serving complexity is
the one {{ch:res-test-time}} showed is affordable for small adapters.

**Retrieval instead of retraining.** Keeps weights fixed and updates an index, which sidesteps
the dial entirely; {{ch:res-memory}}'s conclusion arriving here as a continual-learning strategy.

**More external data rather than more rounds.** The 100% row reaches 0.8737 monotonically while
every self-training row peaks and falls.

**Widen the verifier rather than run more rounds.** Moving $v$ from 0.45 to 0.90 moves the
ceiling from 0.7525 to 0.9550; no schedule change does that.

## 14. Evaluation

Estimate your drift half-life from historical accuracy on freshly labelled samples. One number,
and it sets the cadence.

Measure absorption and forgetting separately after each update — new-distribution accuracy and
old-capability retention. Most teams measure one.

Track regression rate against update frequency. The curve is the missing term in every cadence
model this book has seen.

Measure your task's verifiable fraction directly: what share of items can your verifier
adjudicate at all? That is the loop's ceiling, before any round runs.

Instrument diversity per self-training round — output entropy, n-gram coverage, embedding spread
— and plot it beside quality. The gap between the two peaks is the finding.

## 15. Advanced Concepts

The one-dial result is stated for a scalar plasticity, and the interesting research direction is
that it need not be scalar. If you could identify which weights encode which capabilities, you
could be plastic in one subspace and rigid in another — which is what an elastic weight penalty
attempts and what adapters achieve by fiat. The limit of that idea is a model with explicitly
partitioned capability, where updates are routed to the partition they concern, and it is the
same structural idea as {{ch:res-moe}}'s experts with the routing done by *topic of update*
rather than by token. Nothing here evaluates that, and it is the most promising direction the
chapter touches.

The diversity model treats contraction as a scalar multiplier, which understates a specific and
important asymmetry. Self-generated data does not lose diversity uniformly; it loses the tails
first, because tail behaviours are exactly the ones a sampler produces rarely and a verifier
judges least reliably. So the collapse is concentrated on rare cases — which are
disproportionately the ones that matter for safety, fairness and long-tail user satisfaction.
**A metric average will look healthy for several rounds after the tail has gone**, and the
measurements that would show it are the disaggregated ones {{ch:rai-bias}} argued for on
entirely different grounds.

There is an interaction between the two halves of this chapter that neither listing models. A
continual system trained partly on its own deployed outputs is running a self-improvement loop
whether or not anyone designed one — user interactions include model outputs, logged data
includes model-influenced behaviour, and the "external" data fraction is lower than the pipeline
believes. Estimating the *true* $r$ in a production feedback loop is genuinely hard and, on
these numbers, it is the parameter that decides whether the system degrades. It deserves a
measurement and does not usually get one.

Finally, both halves share an assumption worth flagging: that capability is a scalar that can be
gained and lost. It is not, and the aggregate hides the composition. A model can improve on
average while losing a capability entirely, and the cadence and round-count optima computed here
would look different under a per-capability accounting — almost certainly recommending slower
updating and fewer rounds, because the aggregate is the most forgiving possible view of what an
update does.

## 16. Connection to Previous Chapters

{{eq:evaluation-sets-decay-silently}} from {{ch:ops-prompt-versioning}} is why the frozen
evaluation set correlates **0.34** with the loss a continual system needs to control.

{{eq:rollback-restores-code-not-state}} from {{ch:ops-deployment}} is the regression cost that
makes cadence's optimum interior rather than "as often as possible".

{{eq:test-time-compute-has-a-ceiling-training-does-not}} from {{ch:res-test-time}} is the same
verifier bound through a different mechanism: sampling is bounded by how well the verifier
ranks, self-training by how much it can adjudicate.

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is why round 3
cannot be recovered from round 8 — the artefact that was not pinned is the data distribution.

## 17. Exercises

1. Measure absorption and forgetting at three plasticity settings on your own update pipeline.
   Where is your net optimum?

2. Estimate your drift half-life and compute the cadence optimum. How far is it from your current
   schedule?

3. Add a regression-risk term to your cadence model from historical incident data. How much does
   the optimum move?

4. Compute your task's verifiable fraction and the implied self-training ceiling.

5. Run a self-improvement loop for six rounds, tracking quality and diversity separately. Where
   does each peak?

6. Estimate the *true* external-data fraction in your production feedback loop per
   {{sec:15-advanced-concepts}}. Is it what your pipeline assumes?

## 18. Interview Questions

1. Why does training on new data make the model worse at old tasks?

2. How often should we retrain?

3. Our distribution changes weekly. Should we retrain weekly?

4. How do we know the model has drifted?

5. We ran a self-improvement loop for twelve rounds. What would you check?

6. Which tasks is self-training a good idea for, and why?

## 19. Research Questions

1. Can plasticity be made subspace-selective in a way that beats adapters on the
   absorption/forgetting frontier?

2. Does diversity contraction under self-training concentrate on distribution tails, and by how
   much?

3. What is the true external-data fraction in production feedback loops, and how is it estimated?

4. Do per-capability accountings move the cadence and round-count optima, and in which direction?

## 20. Chapter Summary

Continual learning is usually described as balancing two objectives. It is one parameter.

At plasticity 0.10 an update absorbs **0.221** and loses **0.0028**; at 4.00, **1.000** and
**0.4029** ({{eq:plasticity-and-retention-are-one-dial}}). The net peaks at **0.25**, and every
regulariser moves along the dial rather than removing it — replay 25% at **+0.0030**, elastic
penalty **+0.0038**, adapters **+0.0128**, and retraining from scratch **+0.0275**, which is the
best pair available and priced out of the loop.

Cadence has an interior optimum for the same reason. Staleness falls with frequency; update cost
and regression risk rise. **Every 60 days at $827,919**, against **$40.7 million** daily and
**$1.28 million** annually ({{eq:update-cadence-has-an-interior-optimum}}). And the optimum
moves with drift in a direction intuition gets wrong: at a **14-day** half-life the best cadence
is **annual**, because staleness is saturated at every affordable schedule.

Self-improvement loops share the shape and sharpen the ceiling. Net capability peaks at **0.5996
at round 3** and falls to **0.4901** by round 8 — a gain of +0.0496 followed by a give-back of
0.1094. The ceiling is the verifiable fraction: **0.6400** at $v = 0.20$, **1.0000** at
$v = 1.00$ ({{eq:self-training-improves-only-the-verifiable-fraction}}), which is a **12.4×**
spread in available gain from checked arithmetic to creative writing. And external data sets the
*rate* of diversity contraction rather than a floor
({{eq:collapse-rate-is-set-by-the-real-data-fraction}}) — every mixture short of fully external
goes to zero eventually.

What runs through the chapter is that both procedures are control loops whose feedback is
partial, and in both cases the partiality is structural rather than fixable. A continual learner
gets its signal from a distribution it can only observe with a lag; a self-improvement loop gets
its signal from a verifier that covers part of the task. In each case the system optimises what
it can see and quietly gives up what it cannot, and the metric that would show it is the one
nobody has: drift for the first, diversity for the second.

Carry forward: **plasticity and retention are one dial**, and **choose the round count before
running the loop**.

## 21. Further Reading

- {{cite:gama2014}} — concept drift, its forms, and what adapting to it requires.
- {{cite:kirkpatrick2017ewc}} — the elastic weight penalty, and the framing of forgetting as a
  parameter-importance problem.
- {{cite:zelikman2022star}} — bootstrapping on self-generated, verifier-filtered outputs.
- {{cite:wang2023selfinstruct}} — generating instruction data from the model itself, and what
  that does to the distribution.
