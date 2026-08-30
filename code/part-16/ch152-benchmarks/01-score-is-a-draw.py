# -*- coding: utf-8 -*-
# Extracted from: Chapter 152 — Reasoning Benchmarks and the Reliability Gap
# Source: src/.../ch152-benchmarks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
