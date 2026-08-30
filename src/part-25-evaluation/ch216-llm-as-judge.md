---
id: ev-llm-judge
number: 216
part: XXV
tier: full
status: draft
requires: [agreement-caps-measurable-quality, guideline-defect-is-the-cheapest-disagreement,
           a-score-needs-a-human-baseline, reference-scoring-penalises-valid-answers]
provides: [judge-agreement-is-at-the-human-ceiling, position-advantage-decides-close-pairs,
           self-preference-distorts-the-ranking, optimising-against-a-judge-diverges]
citations: [zheng2023judge, wang2023unfair, rein2023gpqa, card2020power]
---

## 1. Learning Objectives

By the end of this chapter you will be able to read a judge–human agreement figure against
the human–human ceiling rather than against 100%; express position, verbosity and
self-preference biases in quality-equivalent units and compute which comparisons they
decide; design a both-orders protocol and explain why its value is in the verdicts it
refuses; show that a selection loop run against a judge diverges from true quality at a
predictable rate; and compute the human spot-check rate needed to detect a drift of a given
size.

## 2. Why This Matters

{{cite:zheng2023judge}} reported that strong LLM judges reach **over 80% agreement** with
human preferences. That is usually read as "the judge is 80% right." Two humans agree at
**81%** on the same kind of comparison, so the correct reading is **the judge performs like
another annotator** ({{eq:judge-agreement-is-at-the-human-ceiling}}) — a claim about
concordance, not about truth, and one that {{ch:ev-why-hard}}'s ceiling says cannot be
improved on by this validation design.

The same abstract lists position, verbosity and self-enhancement biases, and
{{cite:wang2023unfair}} measured the first: **swapping the order of two candidates made a
weaker model beat a stronger one on 66 of 80 queries**. Expressed in quality-equivalent
units, a first-position advantage of 0.06 means **45%** of candidate pairs are closer than
the bias, and **24%** of verdicts flip on a swap
({{eq:position-advantage-decides-close-pairs}}). Judging both orders and keeping only
verdicts that survive takes accuracy from **74.6% to 88.7%** — and its real value is that it
**declines to decide 36%** of pairs.

Self-preference is structurally worse because it closes the loop. A judge favouring its own
family by **0.055** overturns a ranking whose true top-two gap is **0.013**
({{eq:self-preference-distorts-the-ranking}}), and only *family diversity* in an ensemble
removes it — more samples from the same model do nothing.

And any selection loop run against a judge diverges. Six rounds of picking the best of eight
variants takes the judge's score up **0.1094** and true quality up **0.0219** — **20% of the
reported gain is real** ({{eq:optimising-against-a-judge-diverges}}), and the share falls
every round.

## 3. Prerequisites

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} is what makes the 80%
figure interpretable: agreement with noisy labels is bounded, so a judge validated this way
can reach the human level and no further.

{{eq:guideline-defect-is-the-cheapest-disagreement}} from {{ch:ev-human}} transfers
directly. A judge prompt *is* an annotation guideline, and it is underspecified in the same
places for the same reasons — which makes prompt revision the judge's cheapest improvement
too.

{{eq:a-score-needs-a-human-baseline}} from {{ch:ev-llm-benchmarks}} is why the human–human
number belongs beside the judge–human number in every report.

{{eq:reference-scoring-penalises-valid-answers}} from {{ch:ev-why-hard}} is why judges exist
at all: for open-ended tasks a reference is one draw from the acceptable set, and a judge
evaluates against a learned boundary instead.

{{cite:card2020power}} supplies the spot-check sizing in {{sec:9-practical-example}}.

## 4. Intuitive Explanation

The case for an LLM judge is straightforward and mostly correct. Open-ended outputs cannot
be scored by comparison to a reference — {{ch:ev-why-hard}} showed a single-reference metric
marks 99% of correct summaries wrong. Humans can judge them, and humans are slow and
expensive. A model can judge them at a fraction of a cent each. So use a model.

The question is what you have bought, and the answer starts with reading the headline
number correctly.

{{cite:zheng2023judge}} reports over 80% agreement between strong judges and human
preferences. The instinct is to read that as an accuracy: the judge is right four times in
five, wrong one time in five, and there is a 20% gap to close by building better judges.

Now put the human–human number next to it. Two annotators asked to compare the same pair of
outputs agree about 81% of the time. Not because they are careless — because a meaningful
fraction of pairs are genuinely close, and a genuinely close pair has no fact of the matter
that both annotators can converge on.

So the judge agrees with a human about as often as a human does. **There is no 20% gap to
close.** The judge has reached the level this validation design can measure, and any further
improvement will show up as noise or, worse, as a regression against labels that were the
limiting factor all along.

That is the first and most useful correction, and it changes what you do next. If the judge
is at the human level, stop trying to make it agree more and start asking what *systematic*
errors it makes that a human would not — because those are not visible in an agreement
statistic at all.

There are three, and {{cite:zheng2023judge}} names them: position, verbosity, and
self-enhancement.

**Position** is the easiest to measure and the most embarrassing. Show the judge two
candidates and it prefers the first one, independently of content.
{{cite:wang2023unfair}} demonstrated this by taking 80 queries where ChatGPT judged ChatGPT
better than Vicuna-13B, swapping the presentation order, and getting the opposite verdict on
66 of them.

The useful way to think about this is in quality-equivalent units. If being shown first is
worth 0.06 on a 0-to-1 quality scale, then any pair of candidates whose true quality differs
by less than 0.06 is decided by the order rather than by the content. On a realistic
distribution of candidate pairs, that is 45% of them, and 24% of all verdicts flip when the
order is swapped.

The fix is obvious once stated: judge both orders. What is less obvious is where the value
comes from. Running both orders and keeping only the verdicts that survive the swap takes
accuracy on decided pairs from 74.6% to 88.7%. But it decides only 64% of pairs — it
*refuses* the other 36%.

That refusal is the point. Those are the pairs where the position advantage exceeds the
quality difference, which is to say the pairs where the judge has no information. A
single-order run answers them anyway, confidently, at barely better than chance. Break the
ties with a coin and overall accuracy is 78.2%, barely above single-order. **The gain is not
in the extra judgement, it is in knowing which verdicts to discard.**

**Verbosity** is the same shape with a nastier consequence. Longer output carries a bonus:
in the model here, doubling length is worth 0.085 quality-equivalent, enough to win 74% of
genuine ties and to overturn a 0.10 quality deficit 46% of the time.

The consequence is not the individual wrong verdict. It is what happens when you use the
judge in a loop. Select the best of eight variants each round, five rounds, and output
length goes to 2.29× baseline while the judge's score rises from 0.640 to 0.762 and true
quality rises from 0.640 to 0.660. Most of the reported gain is length.

Nothing in that loop looks wrong from inside it. Each round genuinely improves the judge's
score. Each variant is genuinely selected on merit, as the judge sees merit.

**Self-enhancement** is the third and it has a different structure from the other two.
Position and verbosity are biases toward a *property* of an answer. Self-preference is a
bias toward a *source* — and when the favoured source is the family you are developing, the
evaluation is no longer external to the system it is evaluating.

The magnitude needed to matter is small. In the example here a judge from family A adds
0.055 to family-A candidates, and that moves the genuinely best model out of first place —
because the true gap between the top two candidates is 0.013. And 0.013 is not an unusually
tight race; it is what a race looks like when a team is choosing between models it has
already narrowed down to a shortlist.

The remedy is a judge ensemble, with one important detail. Two judges from the same family
leave the bias exactly where one judge did. Two judges from different families halve it.
**Family diversity removes the bias; judge count does not** — which matters because "ensemble
of judges" is very often implemented as several samples from the same model, which addresses
variance and not bias at all.

Which brings the chapter to the failure that has no protocol fix.

Any selection loop run against a judge optimises toward the judge's decision boundary. That
boundary correlates with quality — that is why the judge agrees with humans 81% of the time
— but it is not quality. Selecting the best of eight variants each round advances both true
quality and the judge-favoured feature, and it advances the feature faster whenever the
feature has more spread among the candidates than quality does. Which it usually will:
verbosity, formatting, confident phrasing, and structural markers all vary more across
candidate variants than genuine correctness does.

Six rounds later, the judge reports an improvement of 0.109 and the real improvement is
0.022. Twenty percent of the reported gain is real, and the share is falling.

This is selection on a noisy proxy, a well-understood failure. What makes it hard here is
that the proxy looks like a measurement. There is no point in the loop where anyone is
gaming anything.

The only instrument that sees it is a human spot-check, and its cost is set by how small a
divergence you refuse to miss. Detecting 0.06 needs 977 human-rated items, 24% of a
4,000-item round. Detecting 0.02 needs the whole set, at which point the judge has bought
nothing at that resolution.

That is the honest summary of what a judge is for. **It converts an evaluation you could not
afford into one you can, at the cost of a blind spot whose size you must independently
measure.** The measurement is the spot-check. Its cost is a dial. Setting it to zero converts
the judge from an instrument into a hypothesis.

## 5. Formal Explanation

**Agreement and its ceiling.** Let $a_{hh}$ be human–human agreement and $a_{jh}$
judge–human agreement on the same population of pairs. Since both are measured against the
same noisy referent, $a_{jh} \le a_{hh} + \epsilon$ with equality when the judge's error is
independent of and comparable to the human's. Observing $a_{jh} \approx a_{hh}$ therefore
identifies "judge is at annotator level," not "judge is $a_{jh}$ accurate" — the latter
would require a referent with no error.

**Biases as quality-equivalent offsets.** Model the judge's decision as choosing the first
candidate when $\Delta q + \beta + \varepsilon > 0$, with $\Delta q$ the true quality
difference, $\beta$ the sum of presentation offsets (position, length), and $\varepsilon
\sim \mathcal{N}(0, \sigma^2)$. Then $\Pr[\text{correct}] = \Phi((\Delta q + \beta)/\sigma)$
when the better candidate carries the offset and $\Phi((\Delta q - \beta)/\sigma)$ when it
does not, so the swap-flip rate is the difference of those, maximised at $\Delta q = 0$ and
decaying as $|\Delta q|$ grows past $\beta$.

**Both-orders protocol.** Running both orders yields agreement with probability $p_A p_B +
(1-p_A)(1-p_B)$, where $p_A, p_B$ are the correctness probabilities under each order.
Conditional on agreement, correctness is $p_A p_B / [p_A p_B + (1-p_A)(1-p_B)]$, which
exceeds $\tfrac12(p_A + p_B)$ whenever $p_A, p_B > \tfrac12$. The protocol is a conditioning
operation, not an averaging one, which is why breaking the ties destroys most of the gain.

**Ensembles.** With judges drawn from families $\{f_1, \dots, f_m\}$ and a self-bonus
$\beta_s$ applied when candidate family matches judge family, the mean bonus for a candidate
of family $g$ is $\beta_s \cdot |\{i : f_i = g\}| / m$. The residual differential bias across
candidates is the spread of that quantity, which is zero when every candidate family is
equally represented among judges and unchanged by replicating one judge.

**Selection divergence.** Let candidate variants have true quality $\sim
\mathcal{N}(\mu_q, \sigma_q^2)$ and judge-favoured feature $\sim \mathcal{N}(\mu_b,
\sigma_b^2)$, independent, with judge score their sum. Selecting the maximum of $N$ on the
judge score advances the two components in proportion to their variances: each round adds
approximately $k_N \sigma_q^2 / \sqrt{\sigma_q^2 + \sigma_b^2}$ to true quality and
$k_N \sigma_b^2 / \sqrt{\cdot}$ to the bias, where $k_N \approx \sqrt{2 \ln N}$. The share
of reported gain that is real is therefore $\sigma_q^2 / (\sigma_q^2 + \sigma_b^2)$,
constant per round and independent of $N$.

## 6. Mathematical Foundation

The agreement figure, read against the right referent:

$$a_{jh} \approx a_{hh} \;\Longrightarrow\; \text{judge} \equiv \text{annotator}, \qquad r_{\max} = \sqrt{\kappa(a_{hh})}$$ (eq:judge-agreement-is-at-the-human-ceiling)

At $a_{hh} = a_{jh} = 0.81$: $\kappa = 0.62$, and the judge is at the ceiling, not 19% below
perfect.

Presentation bias in quality-equivalent units:

$$\Pr[\text{flip on swap}] = \Phi\!\left(\frac{\Delta q + \beta}{\sigma}\right) - \Phi\!\left(\frac{\Delta q - \beta}{\sigma}\right), \qquad \Pr[|\Delta q| < \beta] = 45\%$$ (eq:position-advantage-decides-close-pairs)

At $\beta = 0.06$, $\sigma = 0.13$: a 24% overall flip rate, and the both-orders protocol
decides 64% of pairs at 88.7% accuracy.

Self-preference against the margin it has to beat:

$$\text{ranking preserved} \iff \beta_s < \Delta q_{(1),(2)}, \qquad \beta_s^{\text{ens}} = \beta_s \frac{|\{i : f_i = g\}|}{m}$$ (eq:self-preference-distorts-the-ranking)

At $\beta_s = 0.055$ against a top-two gap of $0.013$: ranking inverted. Ensembles reduce
$\beta_s^{\text{ens}}$ only through family diversity.

And the divergence of a selection loop:

$$\frac{\Delta q_{\text{true}}}{\Delta q_{\text{judged}}} = \frac{\sigma_q^2}{\sigma_q^2 + \sigma_b^2} = 20\%$$ (eq:optimising-against-a-judge-diverges)

independent of the number of variants — more candidates per round accelerates both terms
equally and does not improve the ratio.

## 7. Internal Mechanics

Why does position bias exist at all? Because the judge reads the two candidates in sequence
and the first one establishes the frame against which the second is evaluated. That is not a
quirk of transformers — it is the same anchoring effect that
{{ch:ev-human}}'s presentation table measured in humans, with a comparable magnitude. The
mechanism is shared, which is a reason to expect it to be persistent across model
generations rather than an artefact that better training removes.

Verbosity bias has a more specific origin and it is worth naming because it suggests a
mitigation. A longer answer contains more opportunities to satisfy any criterion the judge
is checking. If the prompt asks whether the answer is complete, thorough, and well-supported,
a longer answer is more likely to visibly satisfy all three — and *visibly* is the operative
word, because the judge scores what it can see rather than what is true. Constraining the
judge to a rubric with an explicit length-independence instruction reduces this measurably,
and it does not eliminate it, because the underlying effect is about evidence density rather
than about instructions.

Self-preference is the least understood of the three. The plausible mechanisms are that a
model assigns higher likelihood to text resembling its own distribution, and that stylistic
conventions shared within a family read as quality markers to a judge from that family.
Both predict the effect scales with stylistic distinctiveness, which suggests it will be
*larger* for models with strong house styles and smaller for models trained on similar data
— and it is measurable directly, by scoring a fixed candidate set with judges from several
families and looking at the spread.

The loop divergence has an important structural property that the model makes explicit and
intuition gets wrong. The share of reported gain that is real is
$\sigma_q^2/(\sigma_q^2 + \sigma_b^2)$ — the ratio of *variances* among candidates, not the
correlation between judge and quality. So a judge that agrees with humans 81% of the time
can still deliver a loop where 80% of the reported gain is bias, if the bias-carrying
features happen to vary more across the candidates being generated. **The agreement statistic
does not bound the divergence**, and teams that validate a judge once and then run it in a
loop have measured a different quantity from the one that governs the loop's behaviour.

Notice also what does *not* help: increasing the number of variants per round. Larger $N$
increases $k_N$, which accelerates both components equally and leaves the ratio unchanged.
Searching harder against a biased objective finds more bias, faster.

## 8. Implementation

The first listing reads the agreement figure and prices the presentation biases.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/he1}
"""A judge agreeing with humans 80% of the time has matched the humans, not the truth.

cite:zheng2023judge reported that strong LLM judges reach over 80% agreement with human
preferences, which is usually read as "the judge is 80% right". It is not that. Two humans
agree at about the same rate, so the judge has reached the level of another annotator, and
ch:ev-why-hard's ceiling says no instrument validated this way can do better
(eq:judge-agreement-is-at-the-human-ceiling).

The same abstract lists position, verbosity and self-enhancement biases, and
cite:wang2023unfair measured the first: swapping the order of two candidates made a weaker
model beat a stronger one on 66 of 80 queries.

This listing computes what an order advantage is worth in quality-equivalent units, and
therefore which comparisons are decided by presentation rather than by content
(eq:position-advantage-decides-close-pairs).
"""
import math

HUMAN_HUMAN = 0.81            # two annotators on the same pair, from ch:ev-human
JUDGE_HUMAN = 0.81            # cite:zheng2023judge, "over 80%"
JUDGE_SELF = 0.88             # same judge, same pair, resampled
CHANCE = 0.50


def kappa(obs):
    return (obs - CHANCE) / (1.0 - CHANCE)


print("What an agreement rate actually says, once there is something to")
print("compare it against.")
print()
print(f"{'comparison':>28}{'agreement':>12}{'kappa':>9}"
      f"{'implied error':>16}{'reading':>26}")
print("-" * 91)
ROWS = [
    ("two humans",              HUMAN_HUMAN, "the ceiling"),
    ("judge vs human",          JUDGE_HUMAN, "at the ceiling"),
    ("judge vs itself",         JUDGE_SELF,  "more self-consistent"),
    ("coin flip",               CHANCE,      "the floor"),
]
agree = {}
for name, a, reading in ROWS:
    e = (1.0 - math.sqrt(max(0.0, 2.0 * a - 1.0))) / 2.0
    agree[name] = (a, kappa(a), e)
    print(f"{name:>28}{a:>12.0%}{kappa(a):>9.2f}{e:>16.3f}{reading:>26}")

print()
print("The judge is not 80% right. It is as close to a human as another")
print("human is, which is the strongest claim this design can support.")

print()
print()
print("Position advantage, in quality-equivalent units.")
print()
POS_ADV = 0.06                # first-shown candidate's bonus, on a 0-1 quality scale
NOISE = 0.13                  # judge decision noise, same units
GAPS = [0.02, 0.05, 0.10, 0.16, 0.25, 0.40]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_pick_better(gap, first_is_better):
    """P(judge picks the genuinely better candidate)."""
    adv = POS_ADV if first_is_better else -POS_ADV
    return phi((gap + adv) / NOISE)


print(f"{'true quality gap':>18}{'better shown first':>21}"
      f"{'better shown second':>22}{'flips on swap':>16}")
print("-" * 77)
flip = {}
for g in GAPS:
    a = p_pick_better(g, True)
    b = p_pick_better(g, False)
    flip[g] = (a, b, a - b)
    print(f"{g:>18.2f}{a:>21.1%}{b:>22.1%}{a - b:>16.1%}")

print()
print(f"a candidate shown first carries a {POS_ADV:.2f} quality-equivalent bonus,")
print("so any pair closer than that is decided by the order")

print()
print()
print("How much of a real comparison set that covers.")
print()
# Realistic distribution of true quality gaps between two candidate models.
GAP_DIST = [(0.02, 0.24), (0.05, 0.21), (0.10, 0.19),
            (0.16, 0.14), (0.25, 0.13), (0.40, 0.09)]
print(f"{'true gap':>10}{'share of pairs':>17}{'decided by order?':>20}"
      f"{'flip rate':>12}{'weighted':>11}")
print("-" * 70)
tot_flip = 0.0
below = 0.0
for g, sh in GAP_DIST:
    f = flip[g][2]
    tot_flip += sh * f
    if g < POS_ADV:
        below += sh
    print(f"{g:>10.2f}{sh:>17.0%}{('yes' if g < POS_ADV else 'no'):>20}"
          f"{f:>12.1%}{sh * f:>11.3f}")
print("-" * 70)
print(f"{'TOTAL':>10}{1.0:>17.0%}{'':>20}{'':>12}{tot_flip:>11.3f}")
print()
print(f"{below:.0%} of pairs are closer than the position advantage;")
print(f"{tot_flip:.0%} of verdicts change when the order is swapped")

print()
print()
print("The fix, and what it costs: judge both orders and keep only the")
print("verdicts that survive.")
print()
print(f"{'protocol':>28}{'judgements':>13}{'decided':>11}"
      f"{'undecided':>12}{'accuracy on decided':>22}")
print("-" * 86)
ACC_SINGLE = sum(sh * (0.5 * (flip[g][0] + flip[g][1])) for g, sh in GAP_DIST)
# Both orders: decided when the two runs agree.
dec, corr = 0.0, 0.0
for g, sh in GAP_DIST:
    a, b = flip[g][0], flip[g][1]
    both_right = a * b
    both_wrong = (1 - a) * (1 - b)
    dec += sh * (both_right + both_wrong)
    corr += sh * both_right
print(f"{'single order':>28}{1:>13}{1.0:>11.0%}{0.0:>12.0%}"
      f"{ACC_SINGLE:>22.1%}")
print(f"{'both orders, agree required':>28}{2:>13}{dec:>11.0%}"
      f"{1 - dec:>12.0%}{corr / dec:>22.1%}")
print(f"{'both orders, tie broken by coin':>28}{2:>13}{1.0:>11.0%}"
      f"{0.0:>12.0%}{(corr + 0.5 * (dec - corr) + 0.5 * (1 - dec)):>22.1%}")

print()
print(f"balancing order raises accuracy on decided pairs from "
      f"{ACC_SINGLE:.1%} to {corr / dec:.1%}")
print(f"and honestly refuses to decide {1 - dec:.0%} of them")

print()
print()
print("Verbosity, the second bias, priced the same way.")
print()
LEN_ADV_PER_50PCT = 0.05      # quality-equivalent bonus per 50% more output
print(f"{'length vs baseline':>20}{'quality-equiv bonus':>22}"
      f"{'win rate at gap 0':>20}{'wins a 0.10 deficit?':>23}")
print("-" * 85)
verb = {}
for mult in (1.0, 1.25, 1.5, 2.0, 3.0):
    bonus = LEN_ADV_PER_50PCT * math.log(mult) / math.log(1.5)
    w0 = phi(bonus / NOISE)
    beats = phi((bonus - 0.10) / NOISE)
    verb[mult] = (bonus, w0, beats)
    print(f"{mult:>19.2f}x{bonus:>22.3f}{w0:>20.1%}{beats:>23.1%}")

print()
print()
print("And what that does over rounds of selecting variants against the judge.")
print()
print(f"{'round':>7}{'length mult':>14}{'judge score':>14}"
      f"{'true quality':>15}{'divergence':>13}")
print("-" * 63)
length = 1.0
true_q = 0.640
drift = {}
for r in range(0, 6):
    bonus = LEN_ADV_PER_50PCT * math.log(length) / math.log(1.5)
    judged = true_q + bonus
    drift[r] = (length, judged, true_q, judged - true_q)
    print(f"{r:>7}{length:>13.2f}x{judged:>14.3f}"
          f"{true_q:>15.3f}{judged - true_q:>13.3f}")
    length *= 1.18
    true_q += 0.004          # real progress, small
print(f"""
The agreement table is the correction most needed and it is the smallest table here. Two
humans agree {HUMAN_HUMAN:.0%} of the time on these pairs; the judge agrees with a human
{JUDGE_HUMAN:.0%} of the time (eq:judge-agreement-is-at-the-human-ceiling).

**Those are the same number**, and cite:zheng2023judge's result should be read as "the judge
performs like another annotator" rather than "the judge is right four times in five." The
second reading is a claim about truth; the data supports only a claim about concordance.

Notice also that the judge is *more consistent with itself* ({JUDGE_SELF:.0%}) than it is
with a human. That gap is where the biases live: a systematic preference is perfectly
self-consistent and reduces agreement with people who do not share it.

The position table converts cite:wang2023unfair's finding into something you can plan with.
An order advantage of {POS_ADV:.2f} on a quality scale means the judge picks the genuinely
better candidate {flip[0.05][0]:.0%} of the time when the better one is shown first, and
{flip[0.05][1]:.0%} of the time when it is shown second, on a pair whose true gap is
{0.05:.2f}.

**That is a {flip[0.05][2]:.0%} swing produced by the order of two items in a prompt**, on
a comparison that has nothing to do with order.

The coverage table says how much of a real comparison set that governs.
{below:.0%} of candidate pairs differ by less than the position advantage, and
{tot_flip:.0%} of all verdicts flip when the order is swapped. cite:wang2023unfair reported
66 of 80 queries flippable, which is a stronger result on a set chosen to demonstrate the
effect; the arithmetic here says a garden-variety comparison set is around
{tot_flip:.0%} exposed.

The protocol table is the fix and it is cheap. Judging both orders and keeping only verdicts
that survive the swap takes accuracy on decided pairs from {ACC_SINGLE:.1%} to
{corr / dec:.1%}, at exactly twice the judging cost.

The important column is the fourth one. **The protocol declines to decide
{1 - dec:.0%} of pairs**, and that refusal is the feature rather than the cost. Those are
the pairs where the position advantage exceeds the quality difference, which is to say the
pairs where the judge has no information -- and a single-order run answers them anyway,
confidently, at slightly better than chance.

The last row shows what happens if you break the ties with a coin: overall accuracy
{(corr + 0.5 * (dec - corr) + 0.5 * (1 - dec)):.1%}, barely above the single-order number.
**The gain is not in the extra judgement, it is in knowing which verdicts to discard.**

The verbosity table prices the second listed bias the same way. Output
{2.0:.1f} times longer carries a {verb[2.0][0]:.3f} quality-equivalent bonus, enough to win
{verb[2.0][1]:.0%} of ties and to overturn a {0.10:.2f} quality deficit
{verb[2.0][2]:.0%} of the time.

And the drift table is why that matters more than it looks. Selecting variants against this
judge for five rounds -- an entirely ordinary development loop -- takes output length to
{drift[5][0]:.2f} times baseline, the judge's score from {drift[0][1]:.3f} to
{drift[5][1]:.3f}, and true quality from {drift[0][2]:.3f} to {drift[5][2]:.3f}.

The measured improvement is {drift[5][1] - drift[0][1]:.3f} and the real one is
{drift[5][2] - drift[0][2]:.3f}. **Most of the reported gain is length.**

Nothing in that loop looks wrong from inside it. Each round genuinely improves the judge's
score, each variant is genuinely selected on merit as the judge sees merit, and the drift is
visible only against a measurement the loop does not contain. Which is
ch:ev-llm-judge's second listing.""")
```

## 9. Practical Example

What an agreement rate says once there is something to compare it against:

```
                  comparison   agreement    kappa   implied error                   reading
-------------------------------------------------------------------------------------------
                  two humans         81%     0.62           0.106               the ceiling
              judge vs human         81%     0.62           0.106            at the ceiling
             judge vs itself         88%     0.76           0.064      more self-consistent
                   coin flip         50%     0.00           0.500                 the floor
```

**The judge is not 80% right — it is as close to a human as another human is**
({{eq:judge-agreement-is-at-the-human-ceiling}}). And it is *more consistent with itself*
than with a human, which is where the biases live: a systematic preference is perfectly
self-consistent.

```
  true quality gap   better shown first   better shown second   flips on swap
-----------------------------------------------------------------------------
              0.02                73.1%                 37.9%           35.2%
              0.05                80.1%                 46.9%           33.2%
              0.10                89.1%                 62.1%           27.0%
              0.25                99.1%                 92.8%            6.3%
              0.40               100.0%                 99.6%            0.4%
```

```
  true gap   share of pairs   decided by order?   flip rate   weighted
----------------------------------------------------------------------
      0.02              24%                 yes       35.2%      0.084
      0.05              21%                 yes       33.2%      0.070
      0.10              19%                  no       27.0%      0.051
      0.25              13%                  no        6.3%      0.008
----------------------------------------------------------------------
     TOTAL             100%                                      0.239
```

**45% of pairs are closer than the position advantage and 24% of verdicts flip on a swap**
({{eq:position-advantage-decides-close-pairs}}) — {{cite:wang2023unfair}}'s 66-of-80 is the
same effect on a set chosen to demonstrate it.

```
                    protocol   judgements    decided   undecided   accuracy on decided
--------------------------------------------------------------------------------------
                single order            1       100%          0%                 74.6%
 both orders, agree required            2        64%         36%                 88.7%
both orders, tie broken by coin            2       100%          0%                 78.2%
```

Both orders takes accuracy from **74.6% to 88.7%** and **refuses 36%** of pairs. Break those
ties with a coin and you get 78.2% — **the gain is in knowing which verdicts to discard**,
not in the extra judgement.

```
  length vs baseline   quality-equiv bonus   win rate at gap 0   wins a 0.10 deficit?
-------------------------------------------------------------------------------------
               1.50x                 0.050               65.0%                  35.0%
               2.00x                 0.085               74.5%                  45.6%
               3.00x                 0.135               85.1%                  60.8%

  round   length mult   judge score   true quality   divergence
---------------------------------------------------------------
      0         1.00x         0.640          0.640        0.000
      3         1.64x         0.713          0.652        0.061
      5         2.29x         0.762          0.660        0.102
```

Five rounds of selection: length **2.29×**, judge score **+0.122**, true quality **+0.020**.
**Most of the reported gain is length**, and nothing in the loop looks wrong from inside it.

The second listing takes up the bias that closes the loop.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/he2}
"""Self-preference is the bias that closes the loop, and a closed loop stops measuring.

cite:zheng2023judge lists self-enhancement alongside position and verbosity, and it is the
one with a different structure. Position and verbosity are biases toward a *property* of an
answer; self-enhancement is a bias toward a *source*, and when the source being favoured is
also the thing being developed, the evaluation stops being external to the system
(eq:self-preference-distorts-the-ranking).

Worse, any selection loop run against a judge optimises toward the judge's boundary rather
than toward quality, and the divergence is invisible from inside the loop
(eq:optimising-against-a-judge-diverges).

This listing measures the ranking distortion, prices judge ensembles, and computes the
human spot-check rate needed to notice the drift before it has been shipped.
"""
import math

SELF_BONUS = 0.055            # quality-equivalent bonus a judge gives its own family
NOISE = 0.13

CANDIDATES = [
    ("model from family A", 0.712, "A"),
    ("model from family B", 0.734, "B"),
    ("model from family C", 0.699, "C"),
    ("model from family A2", 0.721, "A"),
]


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


print(f"A judge from family A scores four candidates. Self-bonus "
      f"{SELF_BONUS:.3f}.")
print()
print(f"{'candidate':>22}{'true quality':>15}{'judged':>10}"
      f"{'true rank':>12}{'judged rank':>14}{'moved':>8}")
print("-" * 81)
judged = {n: q + (SELF_BONUS if f == "A" else 0.0) for n, q, f in CANDIDATES}
truth = {n: q for n, q, f in CANDIDATES}
tr = sorted(truth, key=lambda n: -truth[n])
jr = sorted(judged, key=lambda n: -judged[n])
for n, q, f in CANDIDATES:
    print(f"{n:>22}{q:>15.3f}{judged[n]:>10.3f}"
          f"{tr.index(n) + 1:>12}{jr.index(n) + 1:>14}"
          f"{tr.index(n) - jr.index(n):>8}")

print()
print(f"true winner: {tr[0]}")
print(f"judged winner: {jr[0]}")

print()
print()
print("How large the self-bonus has to be to flip the top of the ranking.")
print()
gap_top = truth[tr[0]] - max(truth[n] for n, q, f in CANDIDATES
                             if f == "A")
print(f"{'self-bonus':>12}{'judged winner':>24}{'correct?':>11}"
       f"{'margin':>10}")
print("-" * 57)
for b in (0.00, 0.01, 0.02, 0.03, 0.055, 0.10):
    j = {n: q + (b if f == "A" else 0.0) for n, q, f in CANDIDATES}
    w = max(j, key=lambda n: j[n])
    srt = sorted(j.values(), reverse=True)
    print(f"{b:>12.3f}{w:>24}{('yes' if w == tr[0] else 'no'):>11}"
          f"{srt[0] - srt[1]:>10.3f}")
print()
print(f"the true top-two gap is {gap_top:.3f}, so any self-bonus above that")
print("decides the comparison")

print()
print()
print("Judge ensembles: the bias averages out if the judges differ in family.")
print()
print(f"{'ensemble':>34}{'families':>11}{'residual bias':>16}"
       f"{'cost':>8}{'winner correct?':>18}")
print("-" * 87)
ENSEMBLES = [
    ("one judge, family A",              ["A"]),
    ("two judges, both family A",        ["A", "A"]),
    ("two judges, families A and B",     ["A", "B"]),
    ("three judges, A, B, C",            ["A", "B", "C"]),
    ("five judges, A, A, B, C, D",       ["A", "A", "B", "C", "D"]),
]
ens = {}
for name, fams in ENSEMBLES:
    sc = {}
    for n, q, f in CANDIDATES:
        bonus = sum(SELF_BONUS for jf in fams if jf == f) / len(fams)
        sc[n] = q + bonus
    resid = max(sc[n] - truth[n] for n in sc) - min(sc[n] - truth[n] for n in sc)
    w = max(sc, key=lambda n: sc[n])
    ens[name] = (len(set(fams)), resid, len(fams), w == tr[0])
    print(f"{name:>34}{len(set(fams)):>11}{resid:>16.4f}"
          f"{len(fams):>8}{('yes' if w == tr[0] else 'no'):>18}")

print()
print("Family diversity is what removes the bias, not judge count.")

print()
print()
print("The closed loop: selecting variants against the judge, round by round.")
print()
VARIANTS = 8
TRUE_SD = 0.004               # spread of true quality among candidate variants
BIAS_SD = 0.008               # spread of the judge-favoured feature among them
print(f"{'round':>7}{'judge score':>14}{'true quality':>15}"
      f"{'divergence':>13}{'share of gain that is real':>29}")
print("-" * 78)
true_q, judge_bias = 0.640, 0.0
loop = {}
for r in range(0, 7):
    js = true_q + judge_bias
    loop[r] = (js, true_q, js - true_q)
    real = (true_q - loop[0][1]) / max(js - loop[0][0], 1e-9) if r else 1.0
    print(f"{r:>7}{js:>14.4f}{true_q:>15.4f}{js - true_q:>13.4f}"
          f"{real:>29.0%}")
    # Selecting the best of VARIANTS on judge score advances both, but the
    # judge-favoured feature has more spread, so it advances more.
    k = math.sqrt(2.0 * math.log(VARIANTS))
    denom = math.sqrt(TRUE_SD ** 2 + BIAS_SD ** 2)
    true_q += k * TRUE_SD ** 2 / denom
    judge_bias += k * BIAS_SD ** 2 / denom

print()
print(f"after {6} rounds: judge says +{loop[6][0] - loop[0][0]:.4f}, "
      f"reality is +{loop[6][1] - loop[0][1]:.4f}")

print()
print()
print("Human spot-checks: how many judged items must be re-rated by a person")
print("to notice a divergence of a given size.")
print()
POWER_Z = 2.80
HUMAN_COST = 3.40
JUDGE_COST = 0.019
ITEMS_PER_ROUND = 4000
print(f"{'divergence to detect':>22}{'items to check':>17}{'share of set':>15}"
      f"{'cost/round':>13}{'vs full human':>16}")
print("-" * 83)
spot = {}
for d in (0.02, 0.04, 0.06, 0.10, 0.15):
    n = (POWER_Z ** 2) * 2.0 * 0.66 * 0.34 / (d ** 2)
    n = min(n, ITEMS_PER_ROUND)
    spot[d] = (n, n / ITEMS_PER_ROUND, n * HUMAN_COST)
    print(f"{d:>22.3f}{n:>17.0f}{n / ITEMS_PER_ROUND:>15.1%}"
          f"{n * HUMAN_COST:>13,.0f}"
          f"{n * HUMAN_COST / (ITEMS_PER_ROUND * HUMAN_COST):>15.1%}")

print()
print()
print("Putting a protocol together, priced per round.")
print()
print(f"{'protocol':>38}{'judge cost':>13}{'human cost':>13}"
      f"{'total':>10}{'drift it can see':>22}")
print("-" * 96)
def detectable(share):
    n = ITEMS_PER_ROUND * share
    if n < 1:
        return None
    return math.sqrt((POWER_Z ** 2) * 2.0 * 0.66 * 0.34 / n)


PROTOCOLS = [
    ("judge only, single order", 1, 0.0),
    ("judge, both orders", 2, 0.0),
    ("3-family ensemble, both orders", 6, 0.0),
    ("ensemble + 5% spot-check", 6, 0.05),
    ("ensemble + 20% spot-check", 6, 0.20),
    ("full human evaluation", 0, 1.00),
]
prot = {}
det = {}
for name, jc, hs in PROTOCOLS:
    j = ITEMS_PER_ROUND * jc * JUDGE_COST
    h = ITEMS_PER_ROUND * hs * HUMAN_COST
    d = detectable(hs)
    prot[name] = j + h
    det[name] = d
    catches = "no drift detection" if d is None else f"drift > {d:.3f}"
    print(f"{name:>38}{j:>13,.0f}{h:>13,.0f}{j + h:>10,.0f}{catches:>22}")

print()
print(f"full human evaluation is "
      f"{prot['full human evaluation'] / prot['ensemble + 5% spot-check']:.1f}x "
      f"the ensemble-plus-spot-check protocol")

print(f"""
The self-preference table is the smallest result and the most awkward one. The judge belongs
to family A, so it adds {SELF_BONUS:.3f} to both family-A candidates -- and that is enough to
move `{tr[0]}`, genuinely the best at {truth[tr[0]]:.3f}, out of first place
(eq:self-preference-distorts-the-ranking).

The true top-two gap is {gap_top:.3f} and the self-bonus is {SELF_BONUS:.3f}. **Any bias
larger than {gap_top:.3f} of a quality point decides the comparison**, because the candidates
a team is actually choosing between are close by construction -- nobody runs an evaluation
to distinguish a good model from a terrible one.

The threshold table makes that precise: the ranking is correct up to a bonus of
{gap_top:.3f} and wrong above it. That number is not a property of the judge, it is a
property of how close your candidates are, and **it gets smaller every year** as models
converge.

The ensemble table gives the fix and names the thing that matters. Two judges from the same
family leave the residual bias exactly where one did; two judges from different families cut
it in half; three families cut it further. **Diversity of family is what removes the bias,
and judge count on its own does nothing**, which matters because "use an ensemble of judges"
is usually implemented as several samples from the same model.

The loop table is the more serious problem, because there is no protocol fix for it inside
the loop. Selecting the best of {VARIANTS} variants each round against the judge advances
both true quality and the judge-favoured feature, and it advances the feature faster because
the feature has more spread among candidates -- {BIAS_SD:.3f} against {TRUE_SD:.3f}.

Six rounds later the judge reports an improvement of
{loop[6][0] - loop[0][0]:.4f} and the real improvement is
{loop[6][1] - loop[0][1]:.4f}. **{(loop[6][1] - loop[0][1]) / (loop[6][0] - loop[0][0]):.0%}
of the reported gain is real** (eq:optimising-against-a-judge-diverges), and the share falls
every round.

This is selection on a noisy proxy, which is a well-understood failure, arriving in a form
where the proxy looks like a measurement. Nothing inside the loop is wrong: each round
genuinely selects the variant the judge scores highest, and each round genuinely improves the
judge's score.

The spot-check table is the only instrument that sees it. Detecting a divergence of
{0.060:.3f} needs {spot[0.06][0]:.0f} human-rated items -- {spot[0.06][1]:.0%} of the round's
evaluation set -- at {spot[0.06][2]:,.0f} per round. Detecting {0.020:.3f} needs the whole
set, which is to say the judge has bought nothing at that resolution.

That is the honest statement of what a judge is for: **it converts an evaluation you could
not afford into one you can, at the cost of a blind spot whose size you must independently
measure.** The measurement is the spot-check, its cost is set by the divergence you are
willing to miss, and skipping it converts the judge from an instrument into a hypothesis.

The protocol table prices the whole arrangement. A three-family ensemble judged in both
orders with a {0.05:.0%} human spot-check costs
{prot['ensemble + 5% spot-check']:,.0f} a round against
{prot['full human evaluation']:,.0f} for full human evaluation --
{prot['full human evaluation'] / prot['ensemble + 5% spot-check']:.1f} times cheaper -- and
it catches position bias, self-preference, and any drift above
{det['ensemble + 5% spot-check']:.3f}.

Raising the spot-check to {0.20:.0%} costs
{prot['ensemble + 20% spot-check'] / prot['ensemble + 5% spot-check']:.1f} times as much and
takes the visible drift down to {det['ensemble + 20% spot-check']:.3f}. That is the dial worth arguing about, and it is the one that
is usually set to zero without a discussion.""")
```

```
             candidate   true quality    judged   true rank   judged rank   moved
---------------------------------------------------------------------------------
   model from family A          0.712     0.767           3             2       1
   model from family B          0.734     0.734           1             3      -2
   model from family C          0.699     0.699           4             4       0
  model from family A2          0.721     0.776           2             1       1
```

A **0.055** self-bonus moves the genuinely best model from first to third
({{eq:self-preference-distorts-the-ranking}}), because the true top-two gap is **0.013** —
which is what a shortlist looks like.

```
                          ensemble   families   residual bias    cost   winner correct?
---------------------------------------------------------------------------------------
               one judge, family A          1          0.0550       1                no
         two judges, both family A          1          0.0550       2                no
      two judges, families A and B          2          0.0275       2               yes
             three judges, A, B, C          3          0.0000       3               yes
        five judges, A, A, B, C, D          4          0.0110       5               yes
```

**Family diversity removes the bias; judge count does nothing** — and "ensemble of judges"
is usually implemented as several samples from one model.

```
  round   judge score   true quality   divergence   share of gain that is real
------------------------------------------------------------------------------
      0        0.6400         0.6400       0.0000                         100%
      3        0.6947         0.6509       0.0438                          20%
      6        0.7494         0.6619       0.0876                          20%
```

Six rounds: judge **+0.1094**, reality **+0.0219** — **20% of the reported gain is real**
({{eq:optimising-against-a-judge-diverges}}), a ratio set by variance among candidates and
*not* by the judge's agreement rate.

```
  divergence to detect   items to check   share of set   cost/round   vs full human
-----------------------------------------------------------------------------------
                 0.020             4000         100.0%       13,600         100.0%
                 0.040             2199          55.0%        7,477          55.0%
                 0.060              977          24.4%        3,323          24.4%
                 0.150              156           3.9%          532           3.9%

                              protocol   judge cost   human cost     total      drift it can see
------------------------------------------------------------------------------------------------
        3-family ensemble, both orders          456            0       456    no drift detection
              ensemble + 5% spot-check          456          680     1,136         drift > 0.133
             ensemble + 20% spot-check          456        2,720     3,176         drift > 0.066
                 full human evaluation            0       13,600    13,600         drift > 0.030
```

**A judge converts an evaluation you cannot afford into one you can, at the cost of a blind
spot whose size you must independently measure.** The spot-check rate is the dial, and it is
usually set to zero without a discussion.

## 10. Production Considerations

Report human–human agreement beside judge–human agreement, always. Without it the judge
figure has no scale, which is {{eq:a-score-needs-a-human-baseline}} applied to the
instrument rather than the model.

Judge both orders and record the undecided pairs as undecided. The refusals are the most
informative output of the protocol.

Build ensembles from different model families, not from repeated samples of one. Repetition
addresses variance; only diversity addresses the bias.

Never use a judge from the family you are developing as the sole judge of your own
candidates. The self-bonus is larger than a shortlist's margins.

Set a spot-check rate deliberately and compute the drift it can see. Zero is a choice, and
it is the choice that converts the judge into an untested hypothesis.

Measure your candidates' spread on judge-favoured features against their spread on quality.
That ratio, not the agreement rate, predicts how fast a selection loop will diverge.

Treat the judge prompt as an annotation guideline and revise it with the same discipline —
{{ch:ev-human}}'s cheapest intervention applies unchanged.

## 11. Common Mistakes

**Reading 80% agreement as 80% accuracy.** The referent is noisy; the ceiling is the
human–human rate.

**Single-order judging.** 24% of verdicts flip on a swap and 45% of pairs are inside the
bias.

**Breaking the ties.** The refused pairs are the ones where the judge has no information;
deciding them anyway recovers almost none of the accuracy.

**Ensembling by resampling one model.** Reduces variance and leaves the family bias exactly
where it was.

**Validating a judge once, then running it in a loop.** The agreement rate does not bound
the loop's divergence.

**Increasing variants per round to improve selection.** It accelerates the bias and the
quality equally, leaving the ratio unchanged.

## 12. Failure Modes

**Length inflation reported as quality improvement.** Five rounds of selection, outputs
2.3× longer, most of the reported gain in the length.

**Home-family judge picks the home-family model.** The self-bonus exceeds the shortlist
margin, and the evaluation confirms the decision that produced it.

**Undecided pairs silently coin-flipped.** The protocol runs both orders, disagreements are
resolved by a tiebreak rule nobody documented, and the accuracy gain evaporates.

**Spot-check too small to see anything.** Five percent of a 4,000-item round detects drift
above 0.133; the drift is 0.09 and the check reports no problem, correctly.

**Judge prompt drift.** The prompt is edited to fix a case, the boundary moves, and every
comparison before and after is on a different instrument —
{{ch:ops-prompt-versioning}}'s ungated prompt, in the evaluation harness.

**Judge and candidate share a training lineage.** The self-preference is present, undetected,
and unmeasurable without a judge from outside the lineage.

## 13. Alternatives

**Human evaluation.** The referent. Roughly 12× the cost of an ensemble-plus-spot-check
protocol here, and it does not escape {{ch:ev-human}}'s own biases.

**Reference-anchored judging.** Give the judge a reference answer and ask about equivalence
rather than preference. Narrows the boundary considerably and reintroduces
{{eq:reference-scoring-penalises-valid-answers}}'s problem.

**Rubric scoring with explicit criteria.** Score each criterion separately and combine.
Reduces verbosity bias, raises cost per judgement, and moves the design problem into the
rubric.

**Execution or verification.** Where the task admits a checkable predicate, skip the judge
entirely — {{ch:ev-why-hard}}'s escape, and still the best option when available.

**Pairwise arena with human voters.** Crowd preference at scale. No judge bias, and it
inherits the data-access distortions {{ch:ev-llm-benchmarks}} discussed.

## 14. Evaluation

Measure your own human–human agreement on the same pairs you judge. Without it you cannot
tell whether your judge is failing or your task is ambiguous.

Measure the position advantage directly: run a sample in both orders and compute the flip
rate. It is one extra pass and it converts a known bias into a number.

Measure the self-bonus by scoring one fixed candidate set with judges from three families
and reporting the spread.

Compute the variance ratio between judge-favoured features and quality among your candidate
variants. That is the loop's divergence rate.

Run the spot-check every round, not once. The divergence accumulates, so a single validation
measures the state before the drift began.

## 15. Advanced Concepts

The independence assumption between judge error and human error is doing a lot of work in
the ceiling argument, and it fails in a specific direction. Judges are trained on human
preference data, so their errors are *correlated* with human errors — they have learned the
same conventions, including the arbitrary ones. That inflates measured judge–human
agreement above what an independent instrument of the same quality would achieve, which
means the true reading is slightly worse than "at the human ceiling." The correction is
unmeasurable without a third, independent referent, which for most tasks does not exist.

The both-orders protocol's refusal rate has an interpretation worth developing. The 36% of
pairs it declines are not evenly distributed — they are concentrated in the close comparisons,
which are exactly the comparisons a shortlist consists of. So the protocol's coverage is
worst precisely where it is most needed, and a team using it to choose between two finalists
may find that most of the evaluation set has no opinion. That is honest, and it is also a
reason to expect judge-based evaluation to be more useful for coarse screening than for
final selection — the opposite of how it is typically deployed.

The divergence result generalises past judges and deserves stating in that form. **Any
selection process run against a proxy diverges from the target at a rate set by the ratio of
variances, not by the proxy's correlation with the target.** That covers reward models, click
metrics, engagement scores, and internal quality dashboards. The recurring mistake is
validating the proxy's *correlation* — which is high — and then running a selection loop
whose behaviour is governed by *variance decomposition*, which nobody measured. It is worth
noticing that this is {{ch:ev-classical-metrics}}'s point about AUC one level up: the
statistic that was validated and the statistic that governs the decision are different
functionals.

Finally, there is an unresolved question about whether judge biases are stable enough to
correct for. If the position advantage were a fixed 0.06, you could debias by subtracting it.
It is not fixed — it varies by task, by prompt, by candidate length, and probably by model
version. That makes the both-orders protocol, which requires no estimate of the bias's
magnitude, strictly more robust than any correction that does. **Protocols that are invariant
to the bias beat corrections that estimate it**, and that principle is worth carrying into
every debiasing decision in this part.

## 16. Connection to Previous Chapters

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} is what makes 80% agreement
interpretable, and {{sec:15-advanced-concepts}} notes the correlated-error correction that
makes the real reading slightly worse.

{{eq:guideline-defect-is-the-cheapest-disagreement}} from {{ch:ev-human}} transfers to the
judge prompt unchanged: it is an annotation guideline, underspecified in the same places, and
revising it is still the cheapest improvement available.

{{eq:a-score-needs-a-human-baseline}} from {{ch:ev-llm-benchmarks}} applies to the judge
itself — a judge score without a human–human baseline beside it has no units.

{{eq:auc-averages-over-thresholds-you-will-not-use}} from {{ch:ev-classical-metrics}} is the
same structural error as {{eq:optimising-against-a-judge-diverges}}: the statistic that was
validated is a different functional from the one that governs the decision.

## 17. Exercises

1. Measure your judge's position advantage by running 200 pairs in both orders. What share
   of your comparison set is closer than it?

2. Implement the both-orders protocol and report the refusal rate. Where in the quality-gap
   distribution do the refusals fall?

3. Score one candidate set with judges from three families. What is the spread, and how does
   it compare to your shortlist's top-two margin?

4. Compute the variance of your candidate variants on length and on measured quality. What
   divergence rate does the ratio predict?

5. Model correlated judge and human errors and recompute the effective ceiling. How much
   worse than "at the human level" is the true reading?

## 18. Interview Questions

1. Our judge agrees with humans 82% of the time. How good is it?

2. Why does judging both orders help, and where does the improvement come from?

3. We use five samples from the same judge model as an ensemble. What does that fix?

4. We have improved our judge score 15% over six months. What do you check?

5. Why can a judge with high human agreement still produce a badly divergent selection loop?

6. How would you set a human spot-check rate?

## 19. Research Questions

1. How correlated are judge errors with human errors, and how much does that inflate measured
   agreement?

2. How stable is the position advantage across tasks, prompts and model versions, and is
   estimating it ever preferable to a both-orders protocol?

3. Does self-preference scale with stylistic distinctiveness, as the likelihood explanation
   predicts?

4. What is the empirical variance ratio between judge-favoured features and true quality
   among candidate variants in real development loops?

## 20. Chapter Summary

An LLM judge is an annotator, and reading it as anything else is where the trouble starts.

{{cite:zheng2023judge}}'s **over 80%** agreement sits against a human–human rate of **81%**:
the judge is **at the annotator level**, and there is no gap to close by improving agreement
({{eq:judge-agreement-is-at-the-human-ceiling}}). What is left are systematic biases that an
agreement statistic cannot see.

Position bias of **0.06** in quality-equivalent units puts **45%** of candidate pairs inside
the bias and flips **24%** of verdicts on a swap
({{eq:position-advantage-decides-close-pairs}}) — {{cite:wang2023unfair}}'s 66-of-80 on a set
built to show it. Judging both orders takes accuracy from **74.6% to 88.7%** and **refuses
36%** of pairs; coin-flipping those refusals recovers almost nothing, so **the value is in
the discards**.

Self-preference of **0.055** overturns a ranking whose true margin is **0.013**
({{eq:self-preference-distorts-the-ranking}}), and only family diversity in an ensemble
removes it — repeated samples of one model do nothing.

And the loop diverges. Six rounds of selecting the best of eight takes the judge's score up
**0.1094** and reality up **0.0219**: **20% of the reported gain is real**
({{eq:optimising-against-a-judge-diverges}}), a ratio set by variance among candidates rather
than by the judge's agreement rate — so validating the judge does not bound it. Only a human
spot-check sees it: **24%** of a round to detect a drift of 0.06.

The thread is that every failure here comes from treating a concordance measurement as a
truth measurement. Agreement says the judge behaves like an annotator; it says nothing about
which systematic errors it makes, and a systematic error is perfectly consistent with high
agreement because it is perfectly consistent with itself. The protocols that work —
both orders, family-diverse ensembles, spot-checks — all share the property of being
invariant to the bias rather than estimating it, which is the right instinct whenever the bias
is real, unmeasured, and moving.

Carry forward: **the judge is an annotator, not an oracle**, and **a selection loop needs a
spot-check, not a validation**.

## 21. Further Reading

- {{cite:zheng2023judge}} — the agreement figure and the three biases, in one abstract; the
  second half is quoted far less than the first.
- {{cite:wang2023unfair}} — position bias measured to the point of absurdity, and a
  calibration proposal for it.
- {{cite:rein2023gpqa}} — the human-baseline discipline that makes any of these figures
  interpretable.
- {{cite:card2020power}} — the power analysis behind the spot-check sizing, and a reminder
  that most reported differences were never detectable.
