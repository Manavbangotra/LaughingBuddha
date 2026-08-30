# -*- coding: utf-8 -*-
# Extracted from: Chapter 180 — Automated Feature Engineering and Model Selection
# Source: src/.../ch180-automl.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Leakage, which is the one error a strong verifier rewards.

ch:aids-stack found the model stage to have the pipeline's best verifier: a
held-out score is a real number, and a check there was worth more than a check
anywhere else. This listing is about the error class that turns that verifier
around.

A leaking feature carries information about the target that will not exist at
prediction time -- a field populated after the outcome, an identifier correlated
with how rows were collected, an aggregate computed over the full dataset
including the future. It does not make the validation score worse. It makes it
BETTER, and the more it leaks the better it gets (eq:leakage-inverts-the-verifier).

Automated feature engineering searches feature space for whatever raises validation
score. So it is, structurally, a search for leakage.
"""
import numpy as np

rng = np.random.default_rng(4691)

M = 4000                # experiments
N_HONEST = 40           # candidate features with genuine signal available
N_LEAKY = 12            # candidate features that leak
BASE = 0.62             # score with no features
CEILING = 0.88          # the best an honest model can do on this problem
HONEST_DECAY = 0.86     # diminishing returns on honest features
LEAK_DECAY = 0.55       # leaks close the gap to a perfect score fast


def build(n_selected, leak_available=N_LEAKY, m=M, honest=N_HONEST,
          base=BASE, ceiling=CEILING, guard=0.0, greedy=True):
    """Select `n_selected` features and return (validation, deployed, leaks used).

    Honest features move the score toward `ceiling` with diminishing returns --
    that ceiling is what the problem actually permits. Leaking features move the
    VALIDATION score toward a perfect 1.0 and contribute nothing on deployment,
    which is what makes them attractive to a search and useless in production.

    A greedy search ranks candidates by validation lift, so it takes every
    available leak before any honest feature. `guard` is the share of leaking
    features a leakage check rejects before selection.
    """
    surviving_leaks = rng.binomial(leak_available, 1.0 - guard, m)
    if greedy:
        n_leak = np.minimum(surviving_leaks, n_selected)
    else:
        frac = leak_available / (leak_available + honest)
        n_leak = np.minimum(rng.binomial(n_selected, frac, m), surviving_leaks)
    n_honest = np.minimum(n_selected - n_leak, honest)

    deploy = ceiling - (ceiling - base) * (HONEST_DECAY ** n_honest)
    val = deploy + (1.0 - deploy) * (1.0 - LEAK_DECAY ** n_leak)
    return (float(val.mean()), float(deploy.mean()), float(n_leak.mean()))


print(f"{N_HONEST} honest candidate features that move the score toward the")
print(f"problem's real ceiling of {CEILING:.2f}, and {N_LEAKY} leaking ones that")
print("move VALIDATION toward 1.00 and deployment not at all. A greedy search")
print("ranks by validation lift, so it takes the leaks first.")
print()
print(f"{'features selected':>19}{'validation':>12}{'deployed':>11}"
      f"{'gap':>8}{'leaks used':>12}")
print("-" * 62)
tab = {}
for n in (2, 5, 10, 20, 40):
    r = build(n)
    tab[n] = r
    print(f"{n:>19}{r[0]:>12.3f}{r[1]:>11.3f}{r[0] - r[1]:>8.3f}{r[2]:>12.1f}")

print()
print()
print("The same selection made at random rather than greedily -- which is what")
print("a human picking features they can explain does.")
print()
print(f"{'features selected':>19}{'greedy val':>12}{'greedy dep':>12}"
      f"{'random val':>12}{'random dep':>12}")
print("-" * 67)
cmp = {}
for n in (5, 10, 20, 40):
    g = build(n)
    r = build(n, greedy=False)
    cmp[n] = (g, r)
    print(f"{n:>19}{g[0]:>12.3f}{g[1]:>12.3f}{r[0]:>12.3f}{r[1]:>12.3f}")

print()
print()
print("The uncomfortable comparison: greedy search wins on the number that is")
print("reported and loses on the one that matters.")
print()
n = 10
g, r = cmp[n]
print(f"{'at 10 features':>26}{'validation':>13}{'deployed':>11}")
print("-" * 50)
print(f"{'greedy (by val lift)':>26}{g[0]:>13.3f}{g[1]:>11.3f}")
print(f"{'random selection':>26}{r[0]:>13.3f}{r[1]:>11.3f}")
print(f"{'difference':>26}{g[0] - r[0]:>+13.3f}{g[1] - r[1]:>+11.3f}")

print()
print()
print("What a leakage guard buys. `guard` is the share of leaking features a")
print("check rejects before selection.")
print()
print(f"{'guard strength':>16}{'validation':>12}{'deployed':>11}{'gap':>8}"
      f"{'leaks used':>12}")
print("-" * 59)
gd = {}
for g_ in (0.0, 0.5, 0.8, 0.95, 1.0):
    r = build(10, guard=g_)
    gd[g_] = r
    print(f"{g_:>16.0%}{r[0]:>12.3f}{r[1]:>11.3f}{r[0] - r[1]:>8.3f}"
          f"{r[2]:>12.1f}")

print()
print()
print("And the detection problem: validation score alone cannot distinguish a")
print("good model from a leaking one. Two systems, same reported number:")
print()
target = build(10)[0]
# Find the honest-only feature count that matches the leaky model's validation.
best_n, best_d = None, None
for n_ in range(1, N_HONEST + 1):
    v, d, _ = build(n_, guard=1.0)
    if best_n is None or abs(v - target) < abs(best_d - target):
        best_n, best_d = n_, v
clean_v, clean_d, _ = build(best_n, guard=1.0)
leak_v, leak_d, leak_n = build(10)
print(f"{'system':>34}{'validation':>13}{'deployed':>11}")
print("-" * 58)
print(f"{f'{best_n} honest features, no leaks':>34}{clean_v:>13.3f}"
      f"{clean_d:>11.3f}")
print(f"{f'10 features, {leak_n:.0f} of them leaking':>34}{leak_v:>13.3f}"
      f"{leak_d:>11.3f}")
print()
print(f"   Reported validation differs by {abs(clean_v - leak_v):.3f}.")
print(f"   Deployed performance differs by {abs(clean_d - leak_d):.3f}.")

print(f"""
The first table's gap column is the whole problem. At {10} features the model
reports {tab[10][0]:.3f} and deploys at {tab[10][1]:.3f} -- which is the baseline.
It learned nothing and validated perfectly.

Note that the gap NARROWS as more features are selected: {tab[10][0] - tab[10][1]:.3f}
at ten and {tab[40][0] - tab[40][1]:.3f} at forty. That is not the search getting
wiser. It is the leaks running out -- there are only {N_LEAKY} of them, so once
they are all taken the search has to fall back on honest features.

**A leakage problem therefore looks WORSE at small feature counts**, which is the
opposite of the usual intuition that a bigger model is riskier.

The greedy-versus-random comparison is the finding to carry.

At {10} features, ranking by validation lift scores {cmp[10][0][0]:.3f} against
random selection's {cmp[10][1][0]:.3f} -- **the search wins by
{cmp[10][0][0] - cmp[10][1][0]:+.3f} on the number that gets reported.** On
deployed performance it loses by {cmp[10][0][1] - cmp[10][1][1]:+.3f}.

Automated feature engineering ranks candidates by validation lift. Leaking features
have the highest validation lift. **So automated feature engineering is,
structurally, a search for leakage** (eq:leakage-inverts-the-verifier) -- not
because it is badly built, but because it is doing exactly what it was asked.

The guard table is the one that explains why nobody implements the fix. Going from
no leakage guard to a perfect one takes deployed performance from
{gd[0.0][1]:.3f} to {gd[1.0][1]:.3f} -- a real gain of
{gd[1.0][1] - gd[0.0][1]:+.3f} -- and takes the REPORTED score from
{gd[0.0][0]:.3f} to {gd[1.0][0]:.3f}, a loss of {gd[1.0][0] - gd[0.0][0]:+.3f}.

**A leakage guard makes your number worse and your model better.** Every incentive
in a team that reports validation scores points away from installing one, and the
person who installs it has to explain why the metric went down.

The last table gives the one signal available for free. The leaking model reports
{leak_v:.3f}; the best honest model this problem permits reports {clean_v:.3f},
because the problem's ceiling is {CEILING:.2f} and nothing honest exceeds it.

**A validation score above what the problem plausibly permits is itself the
leakage detector.** That requires knowing the ceiling, which requires someone to
have thought about how predictable the outcome actually is -- a judgement, made in
advance, of the kind ch:aids-stack said the ungradeable stages need.

The practical form: before running the search, write down the score that would be
too good. Then treat exceeding it as a finding about the pipeline rather than
about the model.""")
