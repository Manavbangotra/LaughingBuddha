# -*- coding: utf-8 -*-
# Extracted from: Chapter 180 — Automated Feature Engineering and Model Selection
# Source: src/.../ch180-automl.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Model selection is a search, and the winner's score is a maximum.

ch:aids-agentic-eda found that exploring more comparisons finds more spurious
patterns, and that every correction needs a count the automation does not report.
This listing is that finding at the model stage, where it has a different name and
the same structure.

An AutoML run trains N configurations and reports the best validation score. That
number is a MAXIMUM over N draws, so it is biased upward by an amount that grows
with N (eq:selection-optimism-grows-with-search). The winner is partly good and
partly lucky, and the reported score cannot separate the two.

The parallel to ch:aids-agentic-eda is exact: same denominator problem, same fix.
"""
import numpy as np

rng = np.random.default_rng(4733)

M = 5000                # AutoML runs simulated
TRUE_SPREAD = 0.020     # genuine quality differences among configurations
VAL_NOISE = 0.028       # noise on a validation estimate
BASE = 0.780


def automl(n_configs, m=M, true_spread=TRUE_SPREAD, noise=VAL_NOISE,
           base=BASE, final_holdout=False, holdout_noise=None):
    """Train n_configs, pick the best by validation, and report.

    Returns (reported score, the winner's TRUE quality, optimism, the true
    quality of the best available configuration).
    """
    true = base + rng.normal(0, true_spread, (m, n_configs))
    val = true + rng.normal(0, noise, (m, n_configs))
    pick = val.argmax(1)
    rows = np.arange(m)
    reported = val[rows, pick]
    winner_true = true[rows, pick]
    best_true = true.max(1)
    if final_holdout:
        hn = noise if holdout_noise is None else holdout_noise
        reported = winner_true + rng.normal(0, hn, m)
    return (float(reported.mean()), float(winner_true.mean()),
            float((reported - winner_true).mean()), float(best_true.mean()))


print(f"{M:,} AutoML runs. Configurations differ genuinely by about")
print(f"{TRUE_SPREAD:.3f}; a validation estimate carries {VAL_NOISE:.3f} of noise.")
print("The best validation score is reported.")
print()
print(f"{'configs tried':>15}{'reported':>11}{'winner true':>14}"
      f"{'optimism':>11}{'best available':>16}")
print("-" * 67)
tab = {}
for n in (1, 5, 25, 100, 500, 2000):
    r = automl(n)
    tab[n] = r
    print(f"{n:>15}{r[0]:>11.4f}{r[1]:>14.4f}{r[2]:>11.4f}{r[3]:>16.4f}")

print()
print()
print("What the search actually buys, separated from what it appears to buy.")
print()
print(f"{'configs tried':>15}{'apparent gain':>15}{'real gain':>12}"
      f"{'share real':>13}")
print("-" * 55)
sp = {}
for n in (5, 25, 100, 500, 2000):
    apparent = tab[n][0] - tab[1][0]
    real = tab[n][1] - tab[1][1]
    sp[n] = (apparent, real, real / apparent if apparent else 0)
    print(f"{n:>15}{apparent:>+15.4f}{real:>+12.4f}"
          f"{real / apparent if apparent else 0:>13.1%}")

print()
print()
print("Noise is what converts search into optimism. Holding the search at 100")
print("configurations and varying how noisy the validation estimate is:")
print()
print(f"{'validation noise':>18}{'reported':>11}{'winner true':>14}"
      f"{'optimism':>11}")
print("-" * 54)
nz = {}
for s in (0.004, 0.012, 0.028, 0.060):
    r = automl(100, noise=s)
    nz[s] = r
    print(f"{s:>18.3f}{r[0]:>11.4f}{r[1]:>14.4f}{r[2]:>11.4f}")

print()
print()
print("A final holdout the search never touched. It does not improve the model")
print("-- the same configuration wins -- it corrects the NUMBER.")
print()
print(f"{'configs tried':>15}{'reported, no holdout':>22}"
      f"{'reported, with holdout':>24}{'truth':>9}")
print("-" * 70)
ho = {}
for n in (25, 100, 500, 2000):
    a = automl(n)
    b = automl(n, final_holdout=True)
    ho[n] = (a[0], b[0], a[1])
    print(f"{n:>15}{a[0]:>22.4f}{b[0]:>24.4f}{a[1]:>9.4f}")

print()
print()
print("And the denominator problem, which is ch:aids-agentic-eda's exactly. Two")
print("teams report the same number and did different amounts of searching:")
print()
print(f"{'team':>28}{'configs':>10}{'reported':>11}{'actually':>11}")
print("-" * 60)
for label, n in (("careful, 20 configs", 20), ("exhaustive, 2000 configs", 2000)):
    r = automl(n)
    print(f"{label:>28}{n:>10}{r[0]:>11.4f}{r[1]:>11.4f}")
print()
a20, a2000 = automl(20), automl(2000)
print(f"   Reported scores differ by {a2000[0] - a20[0]:+.4f}.")
print(f"   True quality differs by  {a2000[1] - a20[1]:+.4f}.")
print(f"   Without the config count, the two reports are not comparable.")

print(f"""
The optimism column is the tax on searching, and it grows without bound: 
{tab[1][2]:+.4f} at one configuration and {tab[2000][2]:+.4f} at {2000}
(eq:selection-optimism-grows-with-search).

The second table separates what search buys from what it appears to buy, and the
regularity is striking. Across every search size, **about
{sp[100][2]:.0%} of the apparent gain is real** and the rest is the maximum's
upward bias. Going from one configuration to two thousand appears to buy
{sp[2000][0]:+.4f} and actually buys {sp[2000][1]:+.4f}.

That two-thirds figure is not a universal constant -- it follows from the ratio of
validation noise to genuine configuration spread -- but the SHAPE is general.
Searching harder always buys something, and always reports more than it bought.

The noise table shows what controls the ratio, and contains a second finding worth
separating out. As validation noise rises from {0.004:.3f} to {0.060:.3f}, optimism
goes from {nz[0.004][2]:+.4f} to {nz[0.060][2]:+.4f} -- expected. But the
winner-true column FALLS, from {nz[0.004][1]:.4f} to {nz[0.060][1]:.4f}.

**A noisy validation estimate does not merely inflate the reported score; it
selects a worse configuration**, because the argmax is increasingly driven by which
config got the luckiest split rather than which is best. So the cheapest
improvement to an AutoML pipeline is usually not a bigger search -- it is a less
noisy validation estimate, which improves the model AND the honesty of its score at
the same time.

That is a rare thing in this part: an intervention with no trade-off.

The holdout table is the fix and it is worth being precise about what it does. At
{2000} configurations the search reports {ho[2000][0]:.4f}; a final holdout the
search never touched reports {ho[2000][1]:.4f} against a truth of
{ho[2000][2]:.4f}.

**The holdout does not improve the model.** The same configuration wins either way.
It corrects the number, which is what was wrong.

And the last table is ch:aids-agentic-eda's denominator problem, arriving at the
model stage in a suit. A careful team trying {20} configurations reports
{a20[0]:.4f}; an exhaustive team trying {2000} reports {a2000[0]:.4f}. The reported
gap is {a2000[0] - a20[0]:+.4f} and the true gap is {a2000[1] - a20[1]:+.4f}.

**Without the configuration count, two validation scores are not comparable** --
and an AutoML report gives you the score and rarely the count. That is the same
sentence as the previous chapter's, about a different search, with the same
resolution: a holdout the search never saw needs no denominator.

Which gives this chapter and the last one a single rule.

**Any process that searches and reports its best result must be scored on data the
search did not touch.** Exploration, feature engineering and model selection are
three instances; each one currently reports a maximum and calls it an estimate.""")
