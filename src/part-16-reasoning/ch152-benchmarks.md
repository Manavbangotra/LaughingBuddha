---
id: rsn-benchmarks
number: 152
part: XVI
tier: full
status: draft
requires: [paraphrase-consistency, specification-is-the-ceiling,
           lucky-chain-rate]
provides: [score-is-a-draw, form-variance-share, ranking-stability,
           contamination-signature, reworded-benchmark, reliability-gap-measure]
citations: [mirzadeh2024gsmsymbolic, sprague2024tocot, cobbe2021gsm8k,
            lightman2023verify, brown2024monkeys, deepseek2025r1,
            wei2022cot, snell2024testtime]
---

## 1. Learning Objectives

By the end of this chapter you will be able to read a benchmark score as an
estimate with a spread rather than as a measurement; compute how much of that
spread comes from the wording rather than from sampling, and explain why more
trials per item does not reduce it; say how many surface renderings a comparison
between two models needs before its ranking is trustworthy; measure the effect of
contamination without access to anyone's training data; and explain why the
measurement that detects contamination cannot distinguish it from ordinary
overfitting to phrasing — and why that turns out not to matter.

## 2. Why This Matters

{{part:16}} has spent six chapters on techniques whose value was established by
benchmark numbers. Chain-of-thought's gains ({{cite:wei2022cot}},
{{cite:sprague2024tocot}}), test-time scaling ({{cite:brown2024monkeys}},
{{cite:snell2024testtime}}), process supervision ({{cite:lightman2023verify}}),
reasoning-trained models ({{cite:deepseek2025r1}}) — every one of those arguments
rests on scores. This chapter asks what a score is.

The answer is narrower than the way scores are used.
{{cite:mirzadeh2024gsmsymbolic}} generated GSM8K problems from templates, varying
only names and numbers, and found performance varying across the generated
variants. That single observation reframes every leaderboard: the number you see
is one draw from a distribution over renderings, and the width of that
distribution is not published anywhere.

{{sec:9-practical-example}} builds the experiment and finds the consequence is not
a caveat but a coin flip. Two models whose *true* abilities — averaged over every
rendering — differ by $1.2$ points produce a measured gap with a standard deviation
of $1.4$ points, and on $21.5\%$ of renderings the worse model wins. A paper
generating one version of a benchmark and reporting the winner has roughly a one
in five chance of reporting the wrong one, through no error at all.

The fix is cheap: averaging eight renderings gets the ranking right $99.2\%$ of
the time. What is *not* cheap, and is the more important finding, is that running
more trials per item does not help. For a form-sensitive model $96.7\%$ of the
score variance comes from the wording and only $3.3\%$ from sampling, so a
confidence interval computed from item-level sampling is a tight interval on the
wrong quantity.

The second half is contamination, and it produces the most useful measurement in
the chapter and a genuinely negative result. You cannot reliably detect
contamination from familiarity signals — {{sec:9-practical-example}} shows a
detector's precision falling from $100\%$ to $38\%$ as ordinary text starts
resembling benchmark prose, which is precisely the regime real benchmarks live in.
But you can *route around* it: score a reworded copy, and the reworded score
estimates true ability at every contamination level.

And then the negative result. That same gap is produced, to within a rounding
error, by a completely clean model that is merely fitted to the published
phrasing. The test cannot tell them apart. That is the right note for
{{part:16}} to end on, because it is {{ch:rsn-vs-generation}}'s opening question
arriving in its most practical form.

## 3. Prerequisites

You need {{ch:rsn-vs-generation}}'s paraphrase-consistency criterion, which is the
instrument this chapter turns on benchmarks rather than on individual answers, and
its distinction between computing an answer and fitting the surface form.

From {{ch:rsn-tool-assisted}}, the specification-versus-correctness distinction
transfers directly: a benchmark is a specification of what "good" means, and a
score is a measure of performance against that specification rather than against
the thing it stands for.

From {{ch:rsn-supervision}}, the lucky-chain rate is the same kind of object as
contamination — a mechanism by which a correct-looking result is produced without
the capability the result is taken to demonstrate.

Elementary statistics is assumed: variance decomposition, standard error, and why
averaging $k$ independent draws shrinks a standard deviation by $\sqrt{k}$.

## 4. Intuitive Explanation

Suppose you want to know which of two students is better at algebra, and you have
one exam.

You give them the exam. One scores 72 and the other 70. What have you learned?

Very little, and the reason is not sampling error in the usual sense. You could
give each student a hundred attempts at those exact questions and narrow the
measurement to a fraction of a point, and you would still not know which is
better at algebra — because you would have measured their performance on *this
exam*, and an exam is a particular set of questions with particular wordings.
Change the names in the word problems, change the numbers, rephrase the setups,
and the scores move.

That movement is not noise in the "random and averages out" sense. It is real and
systematic: a student who has drilled on problems phrased one way does better on
that phrasing. Which means the exam's specific wording is part of what you
measured, and it is entangled with the thing you wanted to measure.

Here is the part that decides how much this matters. Suppose the true difference
between the two students is small — a point or two, which is what differences
between neighbouring leaderboard entries usually are. And suppose the movement
across rewordings is also a point or two. Then the exam you happened to give is
selecting the winner about as much as the students' ability is.

The fix is obvious once you see it: give several versions of the exam and average.
The interesting question is how many, and the answer in {{sec:9-practical-example}}
is that eight is enough to get a genuine $1.2$-point difference right $99\%$ of the
time, and that this is essentially free for a template-generated benchmark since
you are substituting names and numbers.

The other thing that comes free with several renderings is more valuable than the
average. Compare a model's *spread* across renderings with another model's. A
model whose score barely moves is computing something about the problem; a model
whose score swings several points is partly responding to the phrasing. That
spread is a direct measurement of {{ch:rsn-vs-generation}}'s question, it costs
nothing extra once you have the renderings, and nobody reports it.

Now contamination. The worry is that a model has seen the test set, so its score
reflects recall rather than capability.

The instinct is to detect it — look for the test items in the training data, or
check whether the model finds them unusually unsurprising. Those methods work
when benchmark text is distinctive and fail when it is not, and benchmark text
assembled from textbooks and exam papers is not distinctive at all.

But you do not have to detect it. Memorising an answer to a specific wording does
not help on a different wording of the same problem. So score a reworded copy, and
whatever leaked stops mattering — the reworded score is an estimate of the
capability regardless.

That is the good news. The bad news, and it is the last idea in {{part:16}}, is
that the gap between the original and reworded scores does not tell you *why* it
exists. A model that memorised a quarter of the test set and a clean model that
merely learned the benchmark's characteristic phrasing produce the same gap. The
measurement cannot separate them.

Which turns out to be acceptable, because the two have the same consequence for
you. Both are scores that will not reproduce on your traffic. The distinction
matters for assigning blame and not for deciding what to believe.

## 5. Formal Explanation

Let a benchmark be a set of $N$ items with latent difficulties $d_i$, and let a
*rendering* $v$ supply surface forms. Model the probability that model $m$ gets
item $i$ right under rendering $v$ as:

$$p_{m,i,v} = \sigma\big(a_m - d_i + s_m \, \phi_{i,v}\big)$$ (eq:item-response)

where $a_m$ is the model's content ability, $\phi_{i,v}$ is the rendering's effect
on that item, and $s_m$ is how sensitive the model is to surface form. This is an
item-response model with one extra term, and that term is the whole subject.

The quantity a benchmark is trying to estimate is ability averaged over
renderings:

$$\bar{A}_m = \mathbb{E}_{v,i}\big[\,p_{m,i,v}\,\big]$$ (eq:true-ability)

What it reports is a score on one rendering with $T$ trials per item:

$$\hat{A}_{m,v} = \frac{1}{NT}\sum_{i,t} \mathbb{1}[\text{correct}]$$ (eq:score-is-a-draw)

whose variance decomposes into two terms with completely different behaviour:

$$\operatorname{Var}\big[\hat{A}_{m,v}\big] = \underbrace{\operatorname{Var}_v\big[\mathbb{E}_i\, p_{m,i,v}\big]}_{\text{rendering}} + \underbrace{\frac{\mathbb{E}[p(1-p)]}{NT}}_{\text{sampling}}$$ (eq:variance-decomposition)

The second term falls as $1/NT$. **The first does not fall at all** — it is a
property of the rendering you chose, and more items or more trials on that
rendering leave it untouched. Define the form share:

$$\rho_m = \frac{\operatorname{Var}_v\big[\mathbb{E}_i\, p_{m,i,v}\big]}{\operatorname{Var}\big[\hat{A}_{m,v}\big]}$$ (eq:form-variance-share)

When $\rho_m$ is near 1, a confidence interval computed from item-level sampling
describes a quantity that is almost entirely irrelevant to the comparison you are
making. {{sec:9-practical-example}} measures $\rho = 96.7\%$ for a form-sensitive
model.

For a comparison between two models, what matters is the distribution of the
measured gap. With one rendering:

$$\Pr\big[\hat{A}_{A,v} > \hat{A}_{B,v}\big] = \Phi\!\left(\frac{\bar{A}_A - \bar{A}_B}{\sigma_{\text{gap}}}\right)$$ (eq:ranking-stability)

so the probability of getting the ranking right depends on the true gap *relative
to* $\sigma_{\text{gap}}$, and averaging $k$ renderings replaces
$\sigma_{\text{gap}}$ with $\sigma_{\text{gap}}/\sqrt{k}$. That is the entire
prescription, and it says the number of renderings you need scales with the square
of the resolution you want.

Now contamination. Let a fraction $c$ of items be memorised, so the model answers
them correctly on the *published* rendering regardless of ability, and let a clean
model have a form-specific edge $\beta$ that applies only to the published
rendering. Then:

$$\hat{A}^{\text{pub}} = c + (1-c)\,\bar{A}, \qquad \hat{A}^{\text{new}} = \bar{A}$$ (eq:contamination-inflation)

$$\hat{A}^{\text{pub}}_{\beta} = \mathbb{E}\big[\sigma(a - d + \beta)\big], \qquad \hat{A}^{\text{new}}_{\beta} = \mathbb{E}\big[\sigma(a - d)\big]$$ (eq:form-fitting-inflation)

Both produce a positive original-minus-reworded gap, and for suitable $c$ and
$\beta$ they produce *the same* gap. So:

$$\text{gap} = \hat{A}^{\text{pub}} - \hat{A}^{\text{new}}$$ (eq:contamination-signature)

is an estimate of how much of the score depends on the published surface form, and
it does not identify the mechanism. It is nonetheless the useful quantity, because
$\hat{A}^{\text{new}}$ is an unbiased estimate of $\bar{A}$ under both models.

## 6. Mathematical Foundation

Three consequences of {{eq:variance-decomposition}} that are worth having
explicitly.

**More items do not substitute for more renderings.** Both appear in the sampling
term and neither appears in the rendering term. A benchmark with $10{,}000$ items
in one wording has a tighter sampling interval and exactly the same rendering
variance as one with $500$. This is why benchmark size has grown without
benchmark reliability growing with it.

**The renderings must be *independent* renderings.** If your variants are
generated by a single template with substituted names, they share whatever the
template's phrasing contributes, and $\operatorname{Var}_v$ is measured over a
narrower distribution than the one you care about — which is the distribution of
phrasings your users will produce. Template variance is a lower bound on real
variance, and {{cite:mirzadeh2024gsmsymbolic}}'s result should be read as such.

**The number of renderings needed scales quadratically in resolution.** From
{{eq:ranking-stability}}, distinguishing a gap of $\delta$ at a given confidence
requires $k \propto (\sigma_{\text{gap}}/\delta)^2$. Halving the gap you want to
resolve quadruples the renderings. Leaderboards resolve gaps of tenths of a point
with $k = 1$, which is not a small error of degree.

Now the contamination arithmetic. From {{eq:contamination-inflation}}, the gap
under memorisation is $c(1 - \bar{A})$ — proportional to contamination and larger
for weaker models, since a weak model has more headroom for memorisation to fill.
That has an uncomfortable implication: **contamination inflates weak models more
than strong ones**, so it compresses apparent differences at the top of a
leaderboard and can reorder the middle.

For the identifiability problem, set the two gaps equal:

$$c\,(1 - \bar{A}) \;=\; \mathbb{E}\big[\sigma(a - d + \beta)\big] - \mathbb{E}\big[\sigma(a - d)\big]$$ (eq:gap-is-not-identifiable)

For any $c$ there is a $\beta$ solving this, and vice versa. The gap is a
one-dimensional statistic being asked to identify a two-dimensional situation, so
no amount of care in measuring it helps. Distinguishing the two requires a
different observation — item-level familiarity signals, training-data access, or a
held-out set constructed after the model's cutoff — and the first of those is what
{{sec:9-practical-example}} shows failing in the relevant regime.

## 7. Internal Mechanics

### 7.1 Why rendering variance is not sampling noise

It is tempting to treat the spread across renderings as another source of noise to
be averaged away, and mostly that is the right operational response. But the two
are different in kind, and the difference shows up when you ask what the average
means.

Sampling noise is zero-mean around the quantity of interest. Rendering variance is
zero-mean around $\bar{A}$ *only if you average over the same distribution of
renderings your users produce*. If your benchmark's renderings are drawn from a
narrower or differently-shaped distribution — all written by the same annotators,
all in one register, all with the same numeric ranges — then the average over
renderings is an unbiased estimate of ability *on that distribution* and a biased
estimate of anything else.

So {{eq:true-ability}} is only as meaningful as the rendering distribution behind
it, and "average over renderings" is a prescription for reducing variance rather
than for removing the bias.

### 7.2 The spread is the more informative statistic

Given $k$ renderings you can report a mean and a standard deviation. The mean is
what leaderboards want. The standard deviation is a direct estimate of $s_m$ in
{{eq:item-response}} — how much the surface form enters the computation — and it is
therefore a measurement of the thing {{ch:rsn-vs-generation}} said accuracy could
not measure.

{{sec:9-practical-example}} finds form shares of $38.8\%$ and $96.7\%$ for two
models of nearly equal ability. That is a large, cheap, reportable difference in
robustness, and it predicts which model will hold up on traffic that does not look
like the benchmark.

### 7.3 What contamination does to the shape of a score

Memorisation is not a uniform lift. It applies to the items that leaked, which
are typically the older, more widely-copied ones — so contamination correlates with
item age and with how often a problem appears on the web.

The practical signature is therefore not only an inflated mean but a *changed
difficulty profile*: a contaminated model solves some hard items it should not
while still failing easy ones, so its item-level scores fit
{{eq:item-response}} badly. Residuals from an item-response fit are an
under-used diagnostic, and unlike a familiarity signal they need no access to
training data.

### 7.4 Why a reworded benchmark is not a solved problem

The prescription "score a reworded copy" assumes the rewording preserves
difficulty. It usually does not, exactly. Substituting larger numbers makes
arithmetic harder; rephrasing can remove or add ambiguity; a template can produce
degenerate instances. {{cite:mirzadeh2024gsmsymbolic}}'s templates are careful
about this and it is the main engineering cost of the approach.

The check is that a *clean, form-insensitive* model should score the same on both
renderings. {{sec:9-practical-example}}'s clean model gaps are $-0.1\%$ and
$-0.2\%$, which is what a difficulty-preserving rewording looks like. A systematic
gap on a model you have reason to trust means the rewording changed the task.

### 7.5 The reliability gap

Benchmark accuracy and production reliability differ for reasons this part has
enumerated: benchmarks measure one rendering ({{eq:variance-decomposition}}), score
answers rather than derivations ({{ch:rsn-supervision}}), test a specification
rather than correctness ({{ch:rsn-tool-assisted}}), and may be contaminated
({{eq:contamination-signature}}).

Every one of those pushes the same direction — benchmark scores are optimistic —
and they compound. A system reported at $80\%$ that is measured on one rendering,
scored on final answers, against a partial specification, with unknown
contamination, has an expected production number that no one has estimated and
that nothing in the reported figure bounds.

## 8. Implementation

Two listings. The first builds {{cite:mirzadeh2024gsmsymbolic}}'s experiment and
asks not whether scores move but whether *conclusions* move. The second looks at
what a leaked benchmark looks like from the inside, and at what the obvious
detector can and cannot distinguish.

```python {tier=A name=score-is-a-draw}
"""A benchmark score is a draw from a distribution nobody reports.

cite:mirzadeh2024gsmsymbolic generated GSM8K problems from templates, varying
only names and numbers, and found performance varying across the generated
variants -- which means the single number everyone quotes is one sample from a
spread whose width is not published.

This listing builds that experiment. Problems have a latent difficulty; a model's
chance on an item depends on that difficulty AND on the surface form, because
ch:rsn-vs-generation established that models are partly fitted to surface form
rather than purely to content. Two models are given the same mean ability and
different sensitivities to form (eq:score-is-a-draw).

The question is not whether the scores move. It is whether the CONCLUSIONS move.
"""
import numpy as np

rng = np.random.default_rng(1289)

N_ITEMS = 400            # items in the benchmark
N_VARIANTS = 200         # surface renderings of the same benchmark
N_TRIALS = 40            # attempts per item, to separate model noise from form


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Latent item difficulty, shared by every model and every surface form.
difficulty = rng.normal(size=N_ITEMS)
# Each surface form perturbs each item by an amount specific to that form.
form_effect = rng.normal(size=(N_VARIANTS, N_ITEMS))


class Model:
    """ability: how good it is at the content. sensitivity: how much the surface
    form moves it. Two models can have identical ability and very different
    sensitivity, which is exactly ch:rsn-vs-generation's distinction."""

    def __init__(self, name, ability, sensitivity):
        self.name, self.ability, self.sensitivity = name, ability, sensitivity

    def p(self, variant):
        return sigmoid(self.ability - difficulty
                       + self.sensitivity * form_effect[variant])

    def score(self, variant):
        p = self.p(variant)
        return float(np.mean(rng.random((N_TRIALS, N_ITEMS)) < p))

    def true_ability(self):
        """What the model would score averaged over ALL surface forms -- the
        quantity a benchmark is trying to estimate."""
        p = sigmoid(self.ability - difficulty[None, :]
                    + self.sensitivity * form_effect)
        return float(p.mean())


TARGET_GAP = 0.012        # how much better model A really is

B = Model("model B", 0.55, 2.0)      # form-sensitive
# Solve for A's content ability so that its TRUE ability -- averaged over every
# rendering -- is exactly TARGET_GAP above B's. Fixing the quantity we are
# trying to estimate is what makes the rest of the listing a measurement rather
# than a demonstration.
lo, hi = 0.0, 3.0
for _ in range(40):
    mid = (lo + hi) / 2
    if Model("t", mid, 0.25).true_ability() < B.true_ability() + TARGET_GAP:
        lo = mid
    else:
        hi = mid
A = Model("model A", (lo + hi) / 2, 0.25)   # genuinely better, robust to form

print(f"{N_ITEMS} items, {N_VARIANTS} surface renderings of the same benchmark,")
print(f"{N_TRIALS} attempts per item. The items and their difficulties never")
print("change; only the wording does.")
print()
print(f"{'model':>10}{'true ability':>15}{'form':>10}{'mean score':>13}"
      f"{'sd':>8}{'min':>8}{'max':>8}")
print(f"{'':>10}{'(all forms)':>15}{'sensitivity':>10}{'':>13}{'':>8}"
      f"{'':>8}{'':>8}")
print("-" * 72)

scores = {}
for m in (A, B):
    s = np.array([m.score(v) for v in range(N_VARIANTS)])
    scores[m.name] = s
    print(f"{m.name:>10}{m.true_ability():>15.2%}{m.sensitivity:>10.2f}"
          f"{s.mean():>13.1%}{s.std():>8.1%}{s.min():>8.1%}{s.max():>8.1%}")

sa, sb = scores["model A"], scores["model B"]

print()
print()
print("The comparison a paper reports: one benchmark, one number each.")
print("Which model wins depends on which rendering you happened to publish.")
print()
wins_a = float(np.mean(sa > sb))
gap = sa - sb
print(f"{'quantity':>46}{'value':>10}")
print("-" * 56)
print(f"{'model A ahead on true ability by':>46}"
      f"{A.true_ability() - B.true_ability():>+10.2%}")
print(f"{'variants where model A scores higher':>46}{wins_a:>10.1%}")
print(f"{'mean measured gap (A - B)':>46}{gap.mean():>+10.1%}")
print(f"{'sd of the measured gap':>46}{gap.std():>10.1%}")
print(f"{'largest gap in A''s favour':>46}{gap.max():>+10.1%}")
print(f"{'largest gap in B''s favour':>46}{gap.min():>+10.1%}")

print()
print()
print("How many surface renderings do you need before the ranking is stable?")
print("Average k renderings and ask how often the winner is the right one.")
print()
print(f"{'renderings averaged':>21}{'correct ranking':>18}{'sd of gap':>12}")
print("-" * 51)
stab = {}
for k in (1, 2, 4, 8, 16, 32):
    idx = rng.integers(0, N_VARIANTS, size=(4000, k))
    ga = sa[idx].mean(1)
    gb = sb[idx].mean(1)
    stab[k] = (float(np.mean(ga > gb)), float((ga - gb).std()))
    print(f"{k:>21}{stab[k][0]:>18.1%}{stab[k][1]:>12.1%}")

print()
print()
print("And the part that is usually invisible: how much of the measured spread")
print("is the SURFACE FORM rather than sampling noise in the model?")
print()
print(f"{'model':>10}{'total sd':>12}{'sd from':>12}{'sd from':>12}"
      f"{'form share':>13}")
print(f"{'':>10}{'':>12}{'sampling':>12}{'form':>12}{'of variance':>13}")
print("-" * 59)
decomp = {}
for m in (A, B):
    v = 0
    p = sigmoid(m.ability - difficulty[None, :] + m.sensitivity * form_effect)
    per_form_mean = p.mean(1)
    sd_form = float(per_form_mean.std())
    # Sampling sd of a mean over N_ITEMS x N_TRIALS Bernoulli draws.
    sd_samp = float(np.sqrt((p * (1 - p)).mean() / (N_ITEMS * N_TRIALS)))
    tot = float(np.sqrt(sd_form ** 2 + sd_samp ** 2))
    decomp[m.name] = (tot, sd_samp, sd_form, sd_form ** 2 / tot ** 2)
    print(f"{m.name:>10}{tot:>12.2%}{sd_samp:>12.2%}{sd_form:>12.2%}"
          f"{decomp[m.name][3]:>13.1%}")

print(f"""
The first table contains the result this listing exists for, and it is in the
first two columns rather than in the spread.

Model A's true ability -- its score averaged over every surface rendering, which
is the quantity a benchmark is trying to estimate -- is {A.true_ability():.2%}.
Model B's is {B.true_ability():.2%}. **A is the better model**, by
{A.true_ability() - B.true_ability():.2%}.

Now look at what a single benchmark reports. A scores between {sa.min():.1%} and
{sa.max():.1%} depending on the rendering; B scores between {sb.min():.1%} and
{sb.max():.1%}. Those ranges overlap heavily, and on {1 - wins_a:.1%} of
renderings B comes out ahead.

So a paper that generates one version of a benchmark, runs both models on it, and
reports the winner has roughly a {1 - wins_a:.0%} chance of reporting the wrong
one -- not through any error, and not because the benchmark is bad, but because
the score is a draw and nobody reports the spread it was drawn from
(eq:score-is-a-draw).

The measured gap has mean {gap.mean():+.1%} and standard deviation
{gap.std():.1%}, ranging from {gap.min():+.1%} to {gap.max():+.1%}. A difference
of a couple of points between two models on one benchmark rendering is inside
that noise, and differences of a couple of points are what most leaderboard
positions are made of.

The second table says what it would take to fix, and the answer is cheap.
Averaging {2} renderings gets the ranking right {stab[2][0]:.1%} of the time,
{8} gets {stab[8][0]:.1%}, and {32} gets {stab[32][0]:.1%}. The standard
deviation of the gap falls as 1/sqrt(k), as it must.

Generating thirty-two renderings of a template-based benchmark costs almost
nothing -- it is the substitution of names and numbers -- and it converts a
coin-flip ranking into a reliable one. cite:mirzadeh2024gsmsymbolic's contribution
is exactly this machinery, and the reason to adopt it is not that it makes models
look worse. It is that it makes comparisons mean something.

The third table explains why more attempts per item does NOT fix this, which is
the mistake the confidence intervals in most papers make.

For model B, {decomp['model B'][3]:.1%} of the score variance comes from the
surface form and only the remainder from sampling. Running more trials per item
shrinks the sampling term and leaves the form term untouched, so the confidence
interval gets narrower around a number that is still a draw from a distribution
of width {decomp['model B'][2]:.2%}.

**A confidence interval computed from item-level sampling is an interval on the
wrong quantity.** It tells you how precisely you measured performance on THIS
rendering. It says nothing about how much the rendering itself moved the result,
and for a form-sensitive model that is most of the variance.

Compare the two models' form shares: {decomp['model A'][3]:.1%} for A against
{decomp['model B'][3]:.1%} for B. That difference is measurable, it is a direct
estimate of ch:rsn-vs-generation's "is the surface form entering the
computation", and it comes free with the renderings you generated for the
ranking. **The spread across paraphrases is a better model-quality signal than
the mean across them**, and it is the number that is never reported.""")
```

The second listing turns to contamination.

```python {tier=A name=contamination-signature}
"""What a leaked benchmark looks like from the inside.

Contamination is usually discussed as an integrity problem. It is also a
measurement problem with a specific signature, and this listing looks for that
signature (eq:contamination-signature).

Three models are compared. One is clean. One has memorised a fraction of the test
set -- it answers those items correctly regardless of ability, and that
memorisation does not transfer to a rewording. One is clean but heavily fitted to
the surface form of the benchmark, in ch:rsn-vs-generation's sense.

The last of the three is the reason this listing is not a detector-building
exercise. Two of these models are behaving completely differently and, on the
measurement everyone reaches for, they look the same.
"""
import numpy as np

rng = np.random.default_rng(1381)

N_ITEMS = 3000
N_TRIALS = 12
ABILITY = 0.55


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


difficulty = rng.normal(size=N_ITEMS)
form_orig = rng.normal(size=N_ITEMS)      # the published wording
form_new = rng.normal(size=N_ITEMS)       # an equivalent rewording


def accuracy(ability, fit_bonus, form, memorised=None, published=True):
    """`fit_bonus` is an advantage the model has ON THE PUBLISHED WORDING ONLY --
    features of that specific phrasing that it learned and that do not transfer.
    That is what "fitted to the surface form" means, and it is a different thing
    from being noisy across forms: it is a systematic edge on one rendering."""
    p = sigmoid(ability - difficulty + 0.3 * form
                + (fit_bonus if published else 0.0))
    if memorised is not None and published:
        p = np.where(memorised, 1.0, p)
    hit = rng.random((N_TRIALS, N_ITEMS)) < p
    return float(hit.mean())


def familiarity(memorised, noise):
    """The signal a contamination detector uses: memorised items look more
    'familiar' -- lower perplexity, higher n-gram overlap with training data.
    `noise` is how much ordinary text also looks familiar."""
    return memorised * 1.0 + noise * rng.normal(size=N_ITEMS)


def auroc(s, y, n=200000):
    a = rng.choice(s[y], n)
    b = rng.choice(s[~y], n)
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


CONTAM = 0.25
mem = rng.random(N_ITEMS) < CONTAM

MODELS = [
    ("clean", ABILITY, 0.0, None),
    ("contaminated 25%", ABILITY, 0.0, mem),
    ("clean, form-fitted", ABILITY, 0.75, None),
]

print(f"{N_ITEMS} items, {N_TRIALS} attempts each. All three models have the")
print(f"same underlying ability. The contaminated one has memorised {CONTAM:.0%}")
print("of the test set; the form-fitted one has memorised nothing.")
print()
print(f"{'model':>22}{'published':>12}{'reworded':>11}{'gap':>9}")
print(f"{'':>22}{'benchmark':>12}{'benchmark':>11}{'':>9}")
print("-" * 54)

res = {}
for name, ab, sens, m in MODELS:
    o = accuracy(ab, sens, form_orig, m, published=True)
    # Neither memorisation nor a form-specific edge survives a rewording.
    n = accuracy(ab, sens, form_new, m, published=False)
    res[name] = (o, n, o - n)
    print(f"{name:>22}{o:>12.1%}{n:>11.1%}{o - n:>+9.1%}")

print()
print()
print("The measurement everyone reaches for: a familiarity signal. How well")
print("does it separate memorised items from the rest, as ordinary text gets")
print("more familiar-looking?")
print()
print(f"{'detector noise':>16}{'AUROC on':>12}{'items flagged':>16}"
      f"{'of those,':>13}")
print(f"{'':>16}{'memorised':>12}{'at 10% rate':>16}{'truly leaked':>13}")
print("-" * 57)
det = {}
for noise in (0.3, 0.6, 1.0, 1.6, 2.5):
    s = familiarity(mem, noise)
    a = auroc(s, mem)
    thr = np.quantile(s, 0.90)
    flagged = s >= thr
    prec = float(np.mean(mem[flagged]))
    det[noise] = (a, prec)
    print(f"{noise:>16.1f}{a:>12.2f}{'10.0%':>16}{prec:>13.1%}")

print()
print()
print("The alternative: score the benchmark, then score a reworded copy, and")
print("look at the gap. Sweep how much of the test set is contaminated.")
print()
print(f"{'contamination':>15}{'published':>12}{'reworded':>11}{'gap':>9}"
      f"{'true ability':>15}")
print("-" * 62)
sweep = {}
for c in (0.0, 0.05, 0.10, 0.25, 0.50):
    mm = rng.random(N_ITEMS) < c
    o = accuracy(ABILITY, 0.0, form_orig, mm, published=True)
    n = accuracy(ABILITY, 0.0, form_new, mm, published=False)
    sweep[c] = (o, n, o - n)
    print(f"{c:>15.0%}{o:>12.1%}{n:>11.1%}{o - n:>+9.1%}{n:>15.1%}")

print()
print()
print("Can the gap tell contamination from ordinary form-fitting? Sweep the")
print("form-specific edge on a CLEAN model and compare the gaps it produces.")
print()
print(f"{'form-specific edge':>20}{'published':>12}{'reworded':>11}{'gap':>9}")
print("-" * 52)
fs = {}
for sens in (0.0, 0.25, 0.5, 0.75, 1.2):
    o = accuracy(ABILITY, sens, form_orig, None, published=True)
    n = accuracy(ABILITY, sens, form_new, None, published=False)
    fs[sens] = (o, n, o - n)
    print(f"{sens:>20.2f}{o:>12.1%}{n:>11.1%}{o - n:>+9.1%}")

c_gap = res["contaminated 25%"][2]
f_gap = res["clean, form-fitted"][2]
clean_gap = res["clean"][2]
print(f"""
The first table is the signature. All three models have the same underlying
ability, and on the published benchmark they score {res['clean'][0]:.1%},
{res['contaminated 25%'][0]:.1%} and {res['clean, form-fitted'][0]:.1%}.

On a reworded copy of the same benchmark -- same problems, same difficulties, new
surface -- they score {res['clean'][1]:.1%}, {res['contaminated 25%'][1]:.1%} and
{res['clean, form-fitted'][1]:.1%}.

The clean model's gap is {clean_gap:+.1%}. The contaminated model's is
{c_gap:+.1%}. **That difference is the whole of what a benchmark score was
measuring**, and it appears only because a second rendering existed to compare
against.

The second table is why the usual detector is not enough on its own. A
familiarity signal -- low perplexity on the test item, n-gram overlap with the
training corpus -- separates memorised items well when ordinary text does not look
familiar (AUROC {det[0.3][0]:.2f} at low noise) and poorly when it does
({det[2.5][0]:.2f}). At a 10% flag rate the precision falls from
{det[0.3][1]:.1%} to {det[2.5][1]:.1%}.

The noise level is not a knob you control. It is how much your model's training
data resembles benchmark prose, and for benchmarks assembled from textbooks, exam
papers and web text it resembles it a great deal. Familiarity detectors work best
exactly where contamination is least likely and worst where it is most likely.

The third table is the practical instrument, and its virtue is that it needs no
access to the training data at all. Sweeping contamination from {0:.0%} to
{0.5:.0%}, the published score climbs from {sweep[0.0][0]:.1%} to
{sweep[0.5][0]:.1%} while the reworded score stays at roughly
{sweep[0.5][1]:.1%} -- because memorisation does not survive a rewording. The gap
tracks contamination almost linearly: {sweep[0.05][2]:+.1%}, {sweep[0.1][2]:+.1%},
{sweep[0.25][2]:+.1%}, {sweep[0.5][2]:+.1%}.

And the reworded column is a direct estimate of the model's true ability at every
contamination level, which is the useful part. **You cannot easily detect
contamination, but you can route around it**: report the reworded score and the
question of what leaked stops mattering.

Now the fourth table, which is the reason this listing does not end with a
recommended detector.

A CLEAN model with a form-specific edge of {0.5} produces a gap of
{fs[0.5][2]:+.1%}. The model that memorised {CONTAM:.0%} of the test set produces
{c_gap:+.1%}. Those are the same number to within a rounding error, and the two
models have nothing whatever in common: one has memorised a quarter of the test
set and the other has memorised none of it.

The whole sweep says the same: gaps of {fs[0.25][2]:+.1%}, {fs[0.5][2]:+.1%},
{fs[0.75][2]:+.1%} and {fs[1.2][2]:+.1%} from a clean model, against
{sweep[0.05][2]:+.1%} to {sweep[0.5][2]:+.1%} from contamination. The two ranges
sit on top of each other.

So the original-minus-reworded gap does NOT measure contamination. It measures
how much of the score depends on the specific surface form that was published,
and memorisation is one way to acquire that dependence while ordinary
overfitting to phrasing is another (eq:contamination-signature). The test cannot
separate them, and this listing has no measurement that can.

Which is, on reflection, the right conclusion rather than a limitation, because
the two failures have the same consequence. A score inflated by memorised answers
and a score inflated by surface-form fitting are both scores that will not
reproduce on your traffic. The distinction matters for assigning blame and not
for deciding whether to trust the number.

That gives the operating procedure for reading any reasoning benchmark, and it is
the same one ch:rsn-vs-generation arrived at from a completely different
direction. Do not ask whether the benchmark leaked. Ask what the model scores on
a rendering that did not exist when it was trained -- and treat the difference
between the two numbers as the size of the claim you should discount, whatever
produced it.""")
```

## 9. Practical Example

The first listing has $400$ items, $200$ surface renderings of the same benchmark,
and $40$ attempts per item. The items and their difficulties never change; only
the wording does. Model A's content ability is solved for numerically so that its
*true* ability — averaged over every rendering — is exactly $1.2$ points above model
B's.

```
     model   true ability      form   mean score      sd     min     max
              (all forms)sensitivity                                     
------------------------------------------------------------------------
   model A         59.10%      0.25        59.1%    0.4%   58.0%   60.2%
   model B         57.90%      2.00        57.9%    1.6%   54.0%   62.7%
```

A is the better model. B scores anywhere from $54.0\%$ to $62.7\%$ depending on
which rendering you publish, and A from $58.0\%$ to $60.2\%$.

```
                                      quantity     value
--------------------------------------------------------
              model A ahead on true ability by    +1.20%
          variants where model A scores higher     78.5%
                     mean measured gap (A - B)     +1.1%
                        sd of the measured gap      1.4%
```

On $21.5\%$ of renderings the *worse* model wins. A paper generating one version
of a benchmark and reporting the winner has roughly a one-in-five chance of
reporting the wrong one — through no error, and not because the benchmark is bad,
but because the score is a draw and nobody reports the spread it came from
({{eq:score-is-a-draw}}).

The measured gap has a standard deviation of $1.4$ points against a true gap of
$1.2$. Differences of a couple of points on one benchmark rendering are inside
that noise, and differences of a couple of points are what most leaderboard
positions are made of.

```
  renderings averaged   correct ranking   sd of gap
---------------------------------------------------
                    1             78.8%        1.4%
                    2             87.2%        1.0%
                    8             99.2%        0.5%
                   32            100.0%        0.2%
```

Eight renderings get the ranking right $99.2\%$ of the time. For a
template-generated benchmark that is substitution of names and numbers, and it
converts a near-coin-flip into a reliable comparison. This is what
{{cite:mirzadeh2024gsmsymbolic}}'s machinery is for, and the reason to adopt it is
not that it makes models look worse — it is that it makes comparisons mean
something.

The last table is the one that changes how confidence intervals should be read:

```
     model    total sd     sd from     sd from   form share
                          sampling        form  of variance
-----------------------------------------------------------
   model A       0.45%       0.35%       0.28%        38.8%
   model B       1.62%       0.29%       1.60%        96.7%
```

For model B, $96.7\%$ of the score variance is the rendering and $3.3\%$ is
sampling. Running more trials per item shrinks the second term and leaves the
first untouched, so the interval gets tighter around a number that is still a draw
from a distribution of width $1.60\%$. **A confidence interval computed from
item-level sampling is a precise interval on the wrong quantity**
({{eq:variance-decomposition}}).

And the difference between the two form shares — $38.8\%$ against $96.7\%$ — is a
direct, free estimate of how much the surface form enters each model's
computation. It is {{ch:rsn-vs-generation}}'s question answered with data you
already generated, and it is never reported.

The second listing gives three models the same underlying ability. One is clean;
one has memorised $25\%$ of the test set; one is clean but has a systematic edge on
the published phrasing and no memorisation at all.

```
                 model   published   reworded      gap
                         benchmark  benchmark         
------------------------------------------------------
                 clean       61.4%      61.5%    -0.1%
      contaminated 25%       70.9%      61.5%    +9.5%
    clean, form-fitted       74.8%      61.6%   +13.2%
```

The reworded column recovers the true ability in all three cases. The published
column ranks the *clean* model last.

The obvious detector does not rescue this:

```
  detector noise    AUROC on   items flagged    of those,
                   memorised     at 10% rate truly leaked
---------------------------------------------------------
             0.3        0.99           10.0%       100.0%
             1.0        0.75           10.0%        59.0%
             2.5        0.60           10.0%        38.0%
```

A familiarity signal separates memorised items well when ordinary text does not
look familiar and poorly when it does. The noise level is not a knob you control —
it is how much your training data resembles benchmark prose, and for benchmarks
built from textbooks and exam papers it resembles it a great deal. **Familiarity
detectors work best where contamination is least likely and worst where it is most
likely.**

Sweeping contamination instead:

```
  contamination   published   reworded      gap   true ability
--------------------------------------------------------------
             0%       61.3%      61.4%    -0.2%          61.4%
             5%       63.2%      61.6%    +1.6%          61.6%
            25%       71.2%      61.4%    +9.8%          61.4%
            50%       80.6%      61.2%   +19.4%          61.2%
```

The published score climbs from $61.3\%$ to $80.6\%$ while the reworded score
stays flat, and the reworded score is a direct estimate of true ability at every
level. **You cannot easily detect contamination, but you can route around it.**

And then the result that closes {{part:16}}:

```
  form-specific edge   published   reworded      gap
----------------------------------------------------
                0.00       61.4%      61.6%    -0.2%
                0.25       65.8%      61.1%    +4.7%
                0.50       70.8%      61.2%    +9.6%
                1.20       81.2%      61.6%   +19.6%
```

A clean model with a form-specific edge of $0.5$ produces a gap of $+9.6\%$. The
model that memorised a quarter of the test set produces $+9.5\%$. Those are the
same number to within a rounding error, and the two models have nothing whatever
in common. Across the sweep, the clean model's gaps ($+4.7$ to $+19.6$) sit
directly on top of contamination's ($+1.6$ to $+19.4$).

So the gap does not measure contamination. It measures how much of the score
depends on the published surface form, and memorisation is one route to that
dependence while ordinary overfitting to phrasing is another
({{eq:gap-is-not-identifiable}}).

Which is the right conclusion rather than a limitation, because the two failures
have the same consequence: both are scores that will not reproduce on your
traffic. The distinction matters for assigning blame, not for deciding what to
believe.

## 10. Production Considerations

Never compare two models on one rendering. Generate at least eight, average, and
report the spread. {{sec:9-practical-example}} puts the cost at substitution of
names and numbers and the benefit at moving a $78.8\%$ ranking accuracy to
$99.2\%$.

Report the standard deviation across renderings alongside the mean. It is the
robustness number, it is free once you have the renderings, and it predicts
behaviour on traffic that does not look like your benchmark.

Do not quote a confidence interval computed from item-level sampling unless you
have also measured the form share. For a form-sensitive model that interval
describes $3\%$ of the variance ({{eq:form-variance-share}}).

Build your own evaluation set from your own traffic, dated after your model's
cutoff. This is the only construction that is contamination-proof by
construction, and it is also the only one measuring your distribution.

Score a reworded copy of any public benchmark you rely on, and treat the gap as
the discount on the claim. You do not need to know whether it leaked.

Watch for the reliability gap explicitly. Track the same tasks in benchmark form
and in production form, and record the difference as a standing metric rather than
discovering it during an incident.

## 11. Common Mistakes

**Comparing two models on a single benchmark rendering.** A one-in-five chance of
the wrong ranking at a $1.2$-point true gap, and worse for smaller gaps.

**Adding items instead of renderings.** Both reduce the sampling term and neither
touches the rendering term ({{eq:variance-decomposition}}).

**Quoting a sampling confidence interval as if it bounded the score.** It bounds
performance on that rendering, which is not the quantity of interest.

**Trusting a contamination detector on benchmark-like text.**
{{sec:9-practical-example}} shows precision falling to $38\%$ exactly where the
risk is highest.

**Reading a large original-versus-reworded gap as proof of contamination.** A
clean, form-fitted model produces an identical gap
({{eq:gap-is-not-identifiable}}).

**Assuming a rewording preserves difficulty.** Check it against a model you trust
to be form-insensitive; a systematic gap there means the rewording changed the
task, not the model.

**Treating template variance as the full picture.** Templates share phrasing, so
their spread is a lower bound on the spread over the phrasings users actually
produce.

## 12. Failure Modes

*Leaderboard churn without capability change.* Models reordering between
evaluations because the evaluation was re-rendered, reported as progress or
regression.

*Optimising against a rendering.* Iterating on a benchmark tunes the model toward
that rendering's phrasing, which raises the published score and the form share
together — indistinguishable from improvement until something reworded is tried.

*Contamination compressing the top of a leaderboard.* From
{{sec:6-mathematical-foundation}}, the inflation is $c(1 - \bar{A})$, larger for
weaker models, so contamination narrows apparent gaps and can reorder the middle
of a ranking.

*Silent difficulty drift in generated variants.* A template that produces harder
instances makes every model look worse and gets read as a robustness finding.

*The reliability gap discovered in production.* Benchmark optimism compounds across
rendering, answer-only scoring, partial specification and contamination, and none
of the four is visible in the reported number.

## 13. Alternatives

**Held-out sets created after the cutoff.** Contamination-proof by construction,
and the only fully reliable option. The cost is that they expire, and that the
model you evaluate next year needs a new one.

**Private test sets with submission APIs.** Prevents direct contamination and not
indirect leakage through published examples, and it makes the reworded-copy
measurement impossible for anyone outside the maintainers.

**Adversarial or perturbed benchmarks.** {{cite:mirzadeh2024gsmsymbolic}}'s
approach generalised: measure under perturbation rather than under one rendering.
This chapter's recommendation, and it needs the difficulty-preservation check of
{{sec:7-internal-mechanics}}.

**Production traffic evaluation.** The distribution you actually care about, at the
cost of needing labels and of measuring only what your users already send.

**Human preference comparisons.** Sidesteps rendering variance by comparing
outputs rather than scoring items, and introduces the annotator-agreement problems
{{part:9}} covers. Different failure modes rather than fewer.

## 14. Evaluation

Report mean, standard deviation, and number of renderings. Three numbers, and the
second two are what make the first interpretable.

Report the form share $\rho$ ({{eq:form-variance-share}}). It costs one extra
computation over data you already have and it is the robustness measurement this
part has been asking for since {{ch:rsn-vs-generation}}.

Report a reworded score next to every public-benchmark score, and the gap. That
gap is the discount, whatever caused it.

For any ranking claim, state the resolution: what is the smallest true gap your
number of renderings can distinguish? {{eq:ranking-stability}} makes this a
calculation rather than a judgement.

And validate the rewording itself on a model known to be form-insensitive before
drawing conclusions from anyone else's gap.

## 15. Advanced Concepts

**Item-response residuals as a contamination diagnostic.** Fitting
{{eq:item-response}} and examining which items a model over-performs on gives a
signal that needs no training-data access and is not confounded by form-fitting in
the same way the aggregate gap is — because memorisation is item-specific and
form-fitting is not. {{maturity:EMERGING}} and under-used.

**Rendering distributions as a specification.** {{eq:true-ability}} is only
meaningful relative to a distribution over renderings, which means a benchmark
should specify that distribution as deliberately as it specifies its items.
Almost none do, and this is where the field's measurement practice is weakest.

**Benchmark half-life.** Contamination accumulates as a benchmark ages and is
copied, so a benchmark's usefulness decays on a schedule set by its publicity. That
schedule is estimable from the growth of its web presence, and treating it
quantitatively — as an expiry date rather than a reputational question — would be
an improvement.

**The reliability gap as a measured quantity.** The distance between benchmark
score and production performance is the number practitioners actually need and
nobody publishes. Measuring it systematically across task families is the most
useful open empirical project in {{part:16}}. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:rsn-vs-generation}} opened {{part:16}} by arguing that held-out accuracy
cannot distinguish computing an answer from fitting one, and proposed paraphrase
consistency as the test. This chapter is that test applied to the benchmark
itself, and {{eq:form-variance-share}} is its quantitative form.

{{ch:rsn-cot}}, {{ch:rsn-test-time-compute}}, {{ch:rsn-self-consistency}},
{{ch:rsn-supervision}} and {{ch:rsn-tool-assisted}} all rest on measured
comparisons, and this chapter says how many renderings those comparisons needed to
be trustworthy. Read the effect sizes in this part with
{{eq:ranking-stability}} in mind: the large ones — a $128\times$ budget moving
coverage $9.6\% \to 100\%$, a per-step tool losing at every length — survive it
comfortably, and small ones do not.

{{ch:rsn-tool-assisted}}'s specification-versus-correctness distinction is the same
structure as benchmark-versus-capability, and
{{ch:rsn-supervision}}'s lucky-chain rate is the same structure as contamination:
a correct-looking result produced without the capability it is taken to
demonstrate.

## 17. Exercises

1. In the first listing, set the true gap to $0.3$ points and find how many
   renderings are needed for a $95\%$ correct ranking. Compare with
   {{eq:ranking-stability}}'s quadratic prediction.

2. Make the renderings correlated — draw `form_effect` from a distribution with a
   shared component — and show that the effective number of independent renderings
   is smaller than $k$.

3. Add a third model with high ability and high form sensitivity, and construct a
   rendering on which it beats both others. How hard is it to find?

4. In the second listing, make memorisation partially transfer to the reworded
   copy and measure how quickly the reworded score stops estimating true ability.

5. Implement the item-response residual diagnostic from
   {{sec:15-advanced-concepts}} and test whether it separates the contaminated
   model from the form-fitted one where the aggregate gap cannot.

6. Take a public benchmark, generate ten renderings of a subset by substituting
   names and numbers, and measure the form share for two models you have access
   to.

## 18. Interview Questions

1. A model scores 72 and another 70 on the same benchmark. What can you conclude?

2. Why does running more trials per item not reduce the uncertainty in a benchmark
   comparison?

3. How would you measure contamination without access to anyone's training data?

4. Your reworded benchmark shows a 10-point drop. Name two very different causes
   and say how you would distinguish them.

5. Why does contamination compress the top of a leaderboard rather than the
   bottom?

6. You are asked to certify that a model is better than a competitor's. What
   experiment do you run, and what do you report?

## 19. Research Questions

1. Can the rendering distribution a benchmark implicitly defines be characterised,
   and how far is it from the distribution of real user phrasings?

2. Is there a statistic that identifies contamination separately from form-fitting
   without training-data access? {{eq:gap-is-not-identifiable}} rules out the
   aggregate gap; item-level structure may not be ruled out.

3. How large is the reliability gap in practice, by task family, and what predicts
   it?

4. Does optimising against a benchmark measurably raise a model's form share over
   successive releases — and would that make form share a leading indicator of
   benchmark exhaustion?

5. Can benchmark difficulty be preserved under rewording in a verifiable way,
   rather than checked after the fact against a trusted model?

## 20. Chapter Summary

A benchmark score is a draw from a distribution over renderings
({{eq:score-is-a-draw}}), and the width of that distribution is not published. Two
models whose true abilities differ by $1.2$ points produced a measured gap with a
standard deviation of $1.4$, and the worse model won on $21.5\%$ of renderings.
Averaging eight renderings raised ranking accuracy from $78.8\%$ to $99.2\%$, at
the cost of substituting names and numbers.

More trials per item does not help. Score variance decomposes into a sampling term
that falls as $1/NT$ and a rendering term that does not
({{eq:variance-decomposition}}), and for a form-sensitive model the rendering term
was $96.7\%$ of the total. **A confidence interval from item-level sampling is a
precise interval on the wrong quantity.** The compensation is that the spread
across renderings — free once you have them — is a direct estimate of how much the
surface form enters the computation, and it differed by a factor of two and a half
between two models of nearly equal ability.

On contamination, the useful result is that you do not need to detect it.
Familiarity detectors lose precision from $100\%$ to $38\%$ exactly as benchmark
text starts resembling training text, which is the realistic regime. But a reworded
copy estimates true ability at every contamination level from $0\%$ to $50\%$, so
scoring one routes around the problem entirely.

And the result that closes {{part:16}}: a clean model with a form-specific edge
produced a $+9.6$ point original-versus-reworded gap, and a model that memorised a
quarter of the test set produced $+9.5$. The gap does not identify the mechanism
({{eq:gap-is-not-identifiable}}) — it measures how much of the score depends on the
published wording, and memorisation and phrasing-overfit are two routes to the
same place.

That is acceptable, because both are scores that will not reproduce on your
traffic. Which returns {{part:16}} to where it began: do not ask whether the model
scored well. Ask what it scores on a version of the question that did not exist
when it was trained.

## 21. Further Reading

{{cite:mirzadeh2024gsmsymbolic}} is the paper this chapter is built on, and the
part worth reading closely is the template machinery rather than the headline
result — it is the reusable contribution.

{{cite:sprague2024tocot}} is a model of what careful benchmark aggregation looks
like: a meta-analysis over a hundred papers plus its own evaluations across twenty
datasets and fourteen models, which is what it takes to make a claim about where a
technique helps.

{{cite:cobbe2021gsm8k}} and {{cite:lightman2023verify}} are the benchmarks most of
{{part:16}}'s results are measured on, and worth reading with this chapter's
rendering-variance question in mind.

{{cite:brown2024monkeys}} and {{cite:snell2024testtime}} report effect sizes large
enough to survive everything in this chapter, which is worth noticing: the results
{{part:16}} leans on hardest are the ones with the most headroom above the noise.
