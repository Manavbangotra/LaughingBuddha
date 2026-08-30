# -*- coding: utf-8 -*-
# Extracted from: Chapter 152 — Reasoning Benchmarks and the Reliability Gap
# Source: src/.../ch152-benchmarks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
