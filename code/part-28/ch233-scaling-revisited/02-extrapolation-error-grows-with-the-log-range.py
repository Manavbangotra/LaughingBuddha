# -*- coding: utf-8 -*-
# Extracted from: Chapter 233 — Scaling Laws Revisited
# Source: src/.../ch233-scaling-revisited.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""One run, three metrics, three stories -- and a fit used outside the range it was fitted on.

The first listing treated the loss curve as given. This one asks what the curve is made of.

Two failures recur. A metric that requires every token of an answer to be right raises a smooth
per-token improvement to a power, and a power of a smooth curve looks like a threshold. The
discontinuity belongs to the metric, not to the model
(cite:wei2022emergent, cite:schaeffer2023mirage; eq:discontinuity-is-a-property-of-the-metric).

And a power law fitted over two decades and used over six is an extrapolation whose error grows
with the log-range, in a direction that is always optimistic when the true curve has a floor
(eq:extrapolation-error-grows-with-the-log-range).
"""
import math

E_FLOOR, A_RED, GAMMA = 1.69, 2.7e3, 0.155


def true_loss(c):
    return E_FLOOR + A_RED * c ** -GAMMA


def token_acc(c):
    """Smooth per-token accuracy: no thresholds anywhere in it."""
    return 1.0 / (1.0 + math.exp(-(math.log10(c) - 22.0) / 1.6))


BUDGETS = [10 ** x for x in range(18, 28)]

print("One run, scored four ways.")
print()
print(f"{'training FLOPs':>17}{'per-token accuracy':>21}{'loss':>10}"
      f"{'exact match, 5 tokens':>24}{'exact match, 20 tokens':>25}")
print("-" * 97)
series = {"per-token accuracy": [], "loss": [], "em5": [], "em20": []}
for c in BUDGETS:
    p = token_acc(c)
    series["per-token accuracy"].append(p)
    series["loss"].append(true_loss(c))
    series["em5"].append(p ** 5)
    series["em20"].append(p ** 20)
    print(f"{c:>17.0e}{p:>21.4f}{true_loss(c):>10.3f}"
          f"{p ** 5:>24.6f}{p ** 20:>25.8f}")

print()
print("Nothing in the generating process has a threshold in it.")

print()
print()
print("How discontinuous each metric looks.")
print()


def jumpiness(vals):
    """Largest single-step gain as a multiple of the median step."""
    steps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    med = sorted(steps)[len(steps) // 2]
    return max(steps) / med if med > 0 else float("inf")


print(f"{'metric':>26}{'largest step / median step':>30}{'reads as':>22}")
print("-" * 78)
jump = {}
for label in ("loss", "per-token accuracy", "em5", "em20"):
    j = jumpiness(series[label])
    jump[label] = j
    reads = "smooth" if j < 3 else ("kinked" if j < 8 else "emergent")
    print(f"{label:>26}{j:>30.1f}{reads:>22}")

print()
print(f"the same run reads as {'smooth':>0} on loss ({jump['loss']:.1f}) and"
      f" as a threshold on 20-token exact match ({jump['em20']:.1f})")
print(f"a factor of {jump['em20'] / jump['loss']:.0f} in apparent discontinuity")

print()
print()
print("And each metric names a different budget as the moment it 'appeared'.")
print()
print(f"{'metric':>26}{'5% of final':>16}{'50% of final':>16}"
      f"{'decades between them':>24}")
print("-" * 82)
onset = {}
for label in ("loss", "per-token accuracy", "em5", "em20"):
    vals = series[label]
    if label == "loss":
        rng = [(vals[0] - v) / (vals[0] - vals[-1]) for v in vals]
    else:
        rng = [v / vals[-1] for v in vals]
    c5 = next(BUDGETS[i] for i, r in enumerate(rng) if r >= 0.05)
    c50 = next(BUDGETS[i] for i, r in enumerate(rng) if r >= 0.50)
    onset[label] = (c5, c50)
    print(f"{label:>26}{c5:>16.0e}{c50:>16.0e}"
          f"{math.log10(c50 / c5):>24.0f}")

print()
print(f"`loss` starts moving at {onset['loss'][0]:.0e};"
      f" `em20` at {onset['em20'][0]:.0e}")
print(f"a difference of {math.log10(onset['em20'][0] / onset['loss'][0]):.0f}"
      f" orders of magnitude, from the same run")

print()
print()
print("Now the second failure: a fit used outside its range.")
print()
FIT_LO, FIT_HI = 1e19, 1e21


def fit_powerlaw(lo, hi, with_floor):
    """Least-squares slope and intercept in log space over [lo, hi]."""
    xs, ys = [], []
    for k in range(24):
        c = lo * (hi / lo) ** (k / 23)
        v = true_loss(c) - (E_FLOOR if with_floor else 0.0)
        xs.append(math.log10(c))
        ys.append(math.log10(v))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    g = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    return g, my - g * mx


G_NO, I_NO = fit_powerlaw(FIT_LO, FIT_HI, with_floor=False)
G_YES, I_YES = fit_powerlaw(FIT_LO, FIT_HI, with_floor=True)
print(f"fitted over {FIT_LO:.0e} to {FIT_HI:.0e} -- two decades")
print(f"  no floor term: exponent {G_NO:.4f}")
print(f"  floor term:    exponent {G_YES:.4f} (true {-GAMMA:.4f})")

print()
print(f"{'predicted at':>16}{'decades out':>14}{'no-floor fit':>15}"
      f"{'floor fit':>13}{'truth':>10}{'no-floor error':>17}")
print("-" * 85)
err = {}
for c in (1e21, 1e23, 1e25, 1e27, 1e29):
    x = math.log10(c)
    p_no = 10 ** (I_NO + G_NO * x)
    p_yes = E_FLOOR + 10 ** (I_YES + G_YES * x)
    truth = true_loss(c)
    dec = x - math.log10(FIT_HI)
    err[c] = abs(p_no - truth) / truth
    print(f"{c:>16.0e}{dec:>14.0f}{p_no:>15.3f}{p_yes:>13.3f}"
          f"{truth:>10.3f}{err[c]:>16.1%}")

print()
print("A fit that omits the floor predicts loss going to zero, and the error")
print("grows with every decade of extrapolation.")

print()
print()
print("What breaks the extrapolation in practice, beyond the functional form.")
print()
FACTORS = [
    ("benchmark contamination",     "the measured score, not the loss",
     "ch:ev-llm-benchmarks",  0.061, "inflates and flattens"),
    ("repeated training data",      "effective D below nominal D",
     "cite:lee2022dedup", 0.048, "the D term stalls"),
    ("reduced numerical precision", "an effective-parameter penalty",
     "cite:kumar2024precisionscaling", 0.037, "the N term stalls"),
    ("distribution shift at eval",  "a different loss surface",
     "ch:ops-observability", 0.029, "the fit does not apply"),
    ("data exhaustion",             "D cannot be bought at any price",
     "--",                0.055, "the budget stops splitting"),
]
print(f"{'factor':>30}{'what it changes':>36}{'where':>32}{'loss shortfall':>17}")
print("-" * 115)
shortfall = 0.0
for name, what, where, gap, effect in FACTORS:
    shortfall += gap
    print(f"{name:>30}{what:>36}{where:>32}{gap:>17.3f}")
print("-" * 115)
print(f"{'TOTAL':>30}{'':>36}{'':>32}{shortfall:>17.3f}")

PRED = true_loss(1e26)
print()
print(f"a fit predicts {PRED:.3f} at 1e26; these five together leave"
      f" {PRED + shortfall:.3f}")
print(f"which is the loss the curve reaches at {10 ** ((-(PRED + shortfall - E_FLOOR) / A_RED) ** 0 * 0):.0f}"
      if False else
      f"a shortfall of {shortfall / (PRED - E_FLOOR):.1%} of the reducible loss at that budget")

print(f"""
The first table is one training run, scored four ways, with **no threshold anywhere in the
generating process**. Per-token accuracy is a smooth logistic in log-compute; the loss is a
smooth power law with a floor.

Raise that smooth per-token accuracy to the fifth power and you get a curve that spends four
orders of magnitude near zero and then climbs. Raise it to the twentieth -- which is what
"the answer must be exactly right" means for a twenty-token answer -- and it spends six.

The jumpiness table quantifies it. Measured as the largest single step divided by the median
step, `loss` scores {jump['loss']:.1f} and 20-token exact match scores {jump['em20']:.1f} --
**a factor of {jump['em20'] / jump['loss']:.0f} in apparent discontinuity, from the same run**
(eq:discontinuity-is-a-property-of-the-metric).

The onset table is the practical consequence and the reason this matters for planning. Asked
"when did this capability appear", `loss` answers {onset['loss'][0]:.0e} and `em20` answers
{onset['em20'][0]:.0e} -- **{math.log10(onset['em20'][0] / onset['loss'][0]):.0f} orders of
magnitude apart**, on identical data.

That is ch:ev-why-hard' `metric-choice-manufactures-the-finding` and
`discontinuity-hides-progress` arriving in the scaling literature, and it has a specific cost: a
team measuring only exact match sees nothing for four orders of magnitude and concludes the
approach does not work, while the per-token signal was improving throughout. The fix is the
cheap one from that chapter -- keep a continuous metric alongside the binary one -- and it is
worth more here than anywhere else in the book, because the budgets involved are enormous.

The extrapolation section is the second failure and the more expensive one.

Fit a pure power law -- no floor term -- over {FIT_LO:.0e} to {FIT_HI:.0e}, two decades where the
reducible part dominates, and it fits well. Use it at {1e27:.0e}, six decades out, and it is
{err[1e27]:.0%} wrong; at {1e29:.0e}, {err[1e29]:.0%}.

**The error grows with the log-range and it grows in one direction**
(eq:extrapolation-error-grows-with-the-log-range). A fit without a floor term predicts loss
approaching zero, which no model of a stochastic process should predict, and the same fit
*with* a floor term tracks the truth across the whole range.

The distinguishing question is not statistical. It is whether the functional form contains the
irreducible entropy of the data, and a two-decade window where that term is small will not tell
you.

The last table is what breaks the extrapolation for reasons outside the functional form
entirely. Benchmark contamination changes the measured score without changing the model
(ch:ev-llm-benchmarks). Repeated data means effective `D` is below nominal `D`
(cite:lee2022dedup). Reduced precision imposes an effective-parameter penalty
(cite:kumar2024precisionscaling). And data exhaustion means the budget cannot be split the way
the optimum requires, at any price.

Together they leave {shortfall:.3f} of loss on the table at 1e26 -- **{shortfall / (PRED - E_FLOOR):.0%}
of the reducible loss remaining at that budget**. Every one of them is a property of the
pipeline rather than of the scaling relationship, and not one of them appears in the fit.

**A scaling law predicts what a clean run would do**, and the gap between that and what your run
does is the part you control.""")
