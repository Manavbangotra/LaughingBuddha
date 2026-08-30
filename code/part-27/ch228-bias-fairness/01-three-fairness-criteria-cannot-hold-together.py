# -*- coding: utf-8 -*-
# Extracted from: Chapter 228 — Bias and Fairness
# Source: src/.../ch228-bias-fairness.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""You can have calibration, equal false-positive rates, or equal false-negative rates. Two.

cite:kleinberg2016tradeoffs proved that three fairness conditions cannot be satisfied
simultaneously except in constrained special cases, and that even approximate satisfaction
requires the data to lie in an approximate version of one of those cases
(eq:three-fairness-criteria-cannot-hold-together).

The special cases are equal base rates or perfect prediction. Neither describes anything
anyone deploys.

So a fairness requirement is a *choice* among criteria rather than a target to hit, and the
size of the compromise is set by how far apart the base rates are
(eq:the-violation-is-proportional-to-base-rate-difference).
"""
import math

# Two groups, different base rates, same underlying score quality.
BASE_A, BASE_B = 0.34, 0.13
SEPARATION = 1.55                 # d-prime: how well the score separates within a group


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rates(threshold, base):
    """TPR, FPR, and the positive rate, for a group with this base rate."""
    tpr = 1.0 - phi(threshold - SEPARATION)
    fpr = 1.0 - phi(threshold)
    ppr = base * tpr + (1 - base) * fpr
    ppv = (base * tpr) / ppr if ppr > 0 else 0.0
    return tpr, fpr, ppr, ppv


print(f"Two groups. Base rates {BASE_A:.0%} and {BASE_B:.0%}, "
      f"identical score quality (d'={SEPARATION:.2f}).")
print()
print(f"{'threshold':>11}{'A: TPR':>9}{'A: FPR':>9}{'A: PPV':>9}"
      f"{'B: TPR':>9}{'B: FPR':>9}{'B: PPV':>9}")
print("-" * 65)
for t in (0.4, 0.8, 1.2, 1.6, 2.0):
    ta, fa, pa, va = rates(t, BASE_A)
    tb, fb, pb, vb = rates(t, BASE_B)
    print(f"{t:>11.2f}{ta:>9.3f}{fa:>9.3f}{va:>9.3f}"
          f"{tb:>9.3f}{fb:>9.3f}{vb:>9.3f}")

print()
print("At a single shared threshold, TPR and FPR match by construction and")
print("PPV does not, because PPV depends on the base rate.")

print()
print()
print("Now enforce each criterion in turn and measure what the other two do.")
print()


IDX = {"tpr": 0, "fpr": 1, "ppr": 2, "ppv": 3}


def solve(base, kind, ref):
    """Threshold for `base` whose `kind` equals `ref`. Monotone bisection."""
    def f(t):
        return rates(t, base)[IDX[kind]]

    lo, hi = -4.0, 8.0
    increasing = f(hi) > f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if (f(mid) < ref) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


T_A = 1.2
ta, fa, pa, va = rates(T_A, BASE_A)
print(f"Group A fixed at threshold {T_A:.2f}: "
      f"TPR {ta:.3f}, FPR {fa:.3f}, PPV {va:.3f}")
print()
print(f"{'criterion enforced on B':>28}{'B threshold':>14}{'B: TPR':>10}"
      f"{'B: FPR':>10}{'B: PPV':>10}{'largest gap':>14}")
print("-" * 86)
res = {}
for label, kind, ref in (("equal true-positive rate", "tpr", ta),
                         ("equal false-positive rate", "fpr", fa),
                         ("equal predictive value", "ppv", va),
                         ("equal selection rate", "ppr", pa)):
    tb_star = solve(BASE_B, kind, ref)
    t2, f2, p2, v2 = rates(tb_star, BASE_B)
    gaps = {"TPR": abs(t2 - ta), "FPR": abs(f2 - fa), "PPV": abs(v2 - va)}
    worst = max(gaps, key=lambda k: gaps[k])
    res[label] = (tb_star, t2, f2, v2, gaps, worst)
    print(f"{label:>28}{tb_star:>14.3f}{t2:>10.3f}{f2:>10.3f}{v2:>10.3f}"
          f"{f'{worst} {gaps[worst]:.3f}':>14}")

print()
print("Every row satisfies one criterion exactly and violates the others.")

print()
print()
print("How the compromise scales with the base-rate gap.")
print()
print(f"{'base rate B':>13}{'gap to A':>11}{'PPV gap at equal FPR':>23}"
      f"{'FPR gap at equal PPV':>23}")
print("-" * 70)
scale = {}
for bb in (0.34, 0.28, 0.21, 0.13, 0.06, 0.02):
    t_eqfpr = T_A
    _, _, _, v_eqfpr = rates(t_eqfpr, bb)
    ppv_gap = abs(v_eqfpr - va)
    t_eqppv = solve(bb, "ppv", va)
    _, f_eqppv, _, _ = rates(t_eqppv, bb)
    fpr_gap = abs(f_eqppv - fa)
    scale[bb] = (BASE_A - bb, ppv_gap, fpr_gap)
    print(f"{bb:>13.0%}{BASE_A - bb:>11.2f}{ppv_gap:>23.3f}{fpr_gap:>23.3f}")

print()
print("At equal base rates both gaps are zero. That is the special case.")

print()
print()
print("What better prediction does, which is the other special case.")
print()
print(f"{'separation d-prime':>20}{'PPV gap at equal FPR':>23}"
      f"{'FPR gap at equal PPV':>23}{'reading':>22}")
print("-" * 88)
sep_tab = {}
for d in (0.6, 1.55, 2.6, 4.0, 6.0):
    globals()["SEPARATION"] = d
    # Hold group A's true-positive rate fixed as separation improves,
    # so the comparison is at the same operating point each time.
    t_d = d - 0.35
    _, _, _, va_d = rates(t_d, BASE_A)
    _, _, _, vb_d = rates(t_d, BASE_B)
    t_eqppv = solve(BASE_B, "ppv", va_d)
    _, fb_d, _, _ = rates(t_eqppv, BASE_B)
    _, fa_d, _, _ = rates(t_d, BASE_A)
    sep_tab[d] = (abs(vb_d - va_d), abs(fb_d - fa_d))
    reading = ("useless" if d < 1 else "typical" if d < 3
               else "excellent" if d < 5 else "near-perfect")
    print(f"{d:>20.2f}{abs(vb_d - va_d):>23.3f}{abs(fb_d - fa_d):>23.3f}"
          f"{reading:>22}")
globals()["SEPARATION"] = 1.55

print()
print("The gaps close as prediction approaches perfect, which is the second")
print("special case and is not available.")

print()
print()
print("So the design question is which criterion the application needs.")
print()
APPS = [
    ("who gets screened for a disease", "equal true-positive rate",
     "a missed case is the harm"),
    ("who is flagged for review",       "equal false-positive rate",
     "a false flag is the harm"),
    ("what a score is told to mean",    "equal predictive value",
     "the number is shown to a decider"),
    ("who receives a scarce resource",  "equal selection rate",
     "allocation is the outcome"),
    ("who is offered credit",           "contested",
     "all three have advocates"),
]
print(f"{'application':>34}{'the criterion it wants':>28}"
      f"{'why':>34}")
print("-" * 96)
for name, crit, why in APPS:
    print(f"{name:>34}{crit:>28}{why:>34}")

print(f"""
The threshold table is the setup and it already contains the problem. At any shared threshold,
the two groups have identical true-positive and false-positive rates -- the score works equally
well in both -- and different predictive values: {va:.3f} against
{rates(T_A, BASE_B)[3]:.3f} at threshold {T_A:.2f}.

Nothing is wrong with the model. **PPV depends on the base rate**, so equal error rates and
equal predictive value cannot both hold when the base rates differ.

The enforcement table is cite:kleinberg2016tradeoffs' result made concrete. Enforcing equal
true-positive rate on group B leaves a
{res['equal true-positive rate'][4]['PPV']:.3f} gap in predictive value. Enforcing equal
predictive value leaves a {res['equal predictive value'][4]['TPR']:.3f} gap in true-positive
rate and a {res['equal predictive value'][4]['FPR']:.3f} gap in false-positive rate. Enforcing
equal selection rate leaves {res['equal selection rate'][4]['PPV']:.3f} in predictive value
(eq:three-fairness-criteria-cannot-hold-together).

Note that the first two rows coincide: with equal within-group score quality, matching
true-positive rates and matching false-positive rates are the same threshold. The tension is
between *either* of those and predictive value, and it is unavoidable.

**Every row satisfies one criterion exactly and violates the others.** There is no threshold
choice that satisfies all three, and the proof is that there is no such threshold rather than
that nobody has found it.

The scaling table says how much of a compromise is being made. At equal base rates
({BASE_A:.0%} and {BASE_A:.0%}) both gaps are zero. At a {BASE_A - 0.02:.2f} gap in base rates
the PPV gap at equal FPR is {scale[0.02][1]:.3f}
(eq:the-violation-is-proportional-to-base-rate-difference).

So **the size of the fairness compromise is a property of the population, not of the model.**
A team that reduces the gap by improving the model is working on the wrong term; a team that
reduces it by changing who is in the population has changed the question.

The separation table is the other special case. As the score approaches perfect prediction
(d-prime {6.0:.1f}), the gaps fall to {sep_tab[6.0][0]:.3f} and {sep_tab[6.0][1]:.3f}. A
typical deployed model sits near {1.55:.2f}, where they are
{sep_tab[1.55][0]:.3f} and {sep_tab[1.55][1]:.3f}.

That is the honest version of "just make the model better": **it does work**, and the
separation required is not one anybody reaches. A d-prime of {4.0:.1f} corresponds to an AUC
above {0.997:.3f}; a typical deployed classifier sits near {1.55:.2f}, which is an AUC around
{0.86:.2f}.

So the second special case is real and unavailable. Between the two -- equal base rates, or a
near-perfect classifier -- a deployed system is in neither, which is exactly what
cite:kleinberg2016tradeoffs' theorem says.

The last table is what to do instead, and it is a product question rather than a statistical
one. The criterion an application needs depends on where the harm falls. If a missed case is
the harm, you want equal true-positive rates. If a false flag is the harm, equal false-positive
rates. If the number is handed to a human decider who will read it as a probability, equal
predictive value -- which is ch:ev-classical-metrics' calibration requirement wearing a fairness
label.

**Pick the criterion the harm structure implies, state it, and report the others as measured
violations.** That is available today, costs nothing but a decision, and is the opposite of the
common practice of reporting whichever criterion the system happens to satisfy.""")
