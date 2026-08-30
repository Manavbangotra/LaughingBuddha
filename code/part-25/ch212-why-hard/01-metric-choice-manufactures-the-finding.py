# -*- coding: utf-8 -*-
# Extracted from: Chapter 212 — Why Evaluating AI Is Hard
# Source: src/.../ch212-why-hard.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The metric is not a lens on the finding. Below a point, it is the finding.

cite:schaeffer2023mirage showed that a large class of reported "emergent abilities"
disappears when the metric is changed from a discontinuous one to a continuous one, and
that the effect can be manufactured on demand in unrelated domains.

This listing reproduces the mechanism from first principles. One smoothly improving
underlying capability, measured six ways, produces six different stories about when the
capability appeared -- and two of them say it never did
(eq:metric-choice-manufactures-the-finding).

The consequence is not philosophical. A team tracking the discontinuous metric sees a
flat line for several generations of work that was in fact improving steadily, and the
flat line is what gets reported to whoever decides whether to continue
(eq:discontinuity-hides-progress).
"""
import math

SCALES = [0.1, 0.3, 1.0, 3.0, 8.0, 20.0, 70.0, 180.0, 400.0]   # billions of parameters
ANSWER_TOKENS = 5           # the task's answer is five tokens; all must be right


def per_token(scale):
    """Underlying capability: smooth, monotone, no discontinuity anywhere."""
    return 1.0 / (1.0 + math.exp(-(math.log10(scale) - 0.55) * 2.35))


print("One underlying capability, improving smoothly with scale. No jumps.")
print(f"The task needs {ANSWER_TOKENS} tokens and all of them must be right.")
print()
print(f"{'params (B)':>12}{'per-token':>12}{'exact match':>14}"
      f"{'token accuracy':>17}{'log-likelihood':>17}")
print("-" * 72)
tab = {}
for s in SCALES:
    p = per_token(s)
    em = p ** ANSWER_TOKENS
    ll = ANSWER_TOKENS * math.log(p)
    tab[s] = (p, em, p, ll)
    print(f"{s:>12.1f}{p:>12.3f}{em:>14.4f}{p:>17.3f}{ll:>17.3f}")

print()
print("Same numbers. The middle column is the one that gets reported.")

print()
print()
print("When did the capability 'appear'? Each metric answers differently.")
print()
THRESH = 0.05               # "it works" is conventionally somewhere around here
print(f"{'metric':>26}{'first scale above 5%':>23}{'value one step earlier':>25}")
print("-" * 74)


def first_above(f, thresh):
    for s in SCALES:
        if f(s) >= thresh:
            return s
    return None


METRICS = [
    ("exact match (all 5 right)", lambda s: per_token(s) ** ANSWER_TOKENS),
    ("exact match, 12 tokens",    lambda s: per_token(s) ** 12),
    ("exact match, 2 tokens",     lambda s: per_token(s) ** 2),
    ("mean token accuracy",       lambda s: per_token(s)),
    ("normalised log-likelihood", lambda s: 1.0 + math.log(per_token(s)) / 6.0),
]
emerge = {}
for name, f in METRICS:
    s = first_above(f, THRESH)
    prev = SCALES[SCALES.index(s) - 1] if s and SCALES.index(s) > 0 else None
    emerge[name] = (s, f(prev) if prev else 0.0)
    print(f"{name:>26}{(str(s) + 'B') if s else 'never':>23}"
          f"{(f(prev) if prev else 0.0):>25.4f}")

print()
print()
print("How abrupt each metric looks: largest ratio between adjacent scales.")
print()
print(f"{'metric':>26}{'largest jump':>15}{'looks like':>28}")
print("-" * 69)
jumps = {}
for name, f in METRICS:
    best = 1.0
    for a, b in zip(SCALES, SCALES[1:]):
        va, vb = f(a), f(b)
        if va > 1e-9:
            best = max(best, vb / va)
    jumps[name] = best
    verdict = ("a phase change" if best > 20 else
               "a sharp gain" if best > 5 else
               "steady progress")
    print(f"{name:>26}{best:>14.1f}x{verdict:>28}")

print()
print()
print("The predictability test: extrapolate each metric one scale step from its")
print("own last two points, and compare against what actually happened.")
print()
print(f"{'known up to':>14}{'next scale':>13}"
      f"{'token acc: pred/act':>22}{'rel err':>10}"
      f"{'exact: pred/act':>20}{'rel err':>10}")
print("-" * 89)


def step(f, i):
    """Linear extrapolation in log-scale from points i-1, i to point i+1."""
    x0, x1, x2 = (math.log10(SCALES[i - 1]), math.log10(SCALES[i]),
                  math.log10(SCALES[i + 1]))
    y0, y1 = f(SCALES[i - 1]), f(SCALES[i])
    return y1 + (y1 - y0) / (x1 - x0) * (x2 - x1), f(SCALES[i + 1])


err_c, err_d = [], []
for i in range(1, len(SCALES) - 1):
    pc, ac = step(per_token, i)
    pd, ad = step(lambda s: per_token(s) ** ANSWER_TOKENS, i)
    rc, rd = abs(pc - ac) / ac, abs(pd - ad) / max(ad, 1e-9)
    err_c.append(rc)
    err_d.append(rd)
    print(f"{SCALES[i]:>13.1f}B{SCALES[i + 1]:>12.1f}B"
          f"{pc:>12.3f}/{ac:<9.3f}{rc:>10.0%}"
          f"{pd:>10.4f}/{ad:<8.4f}{rd:>10.0%}")
print("-" * 89)
mean_c, mean_d = sum(err_c) / len(err_c), sum(err_d) / len(err_d)
print(f"{'MEAN RELATIVE ERROR':>27}{mean_c:>32.0%}{mean_d:>30.0%}")

print()
print()
print("What a team sees, tracking one metric across five generations of work.")
print()
GEN = [(0.1, "gen 1"), (0.3, "gen 2"), (1.0, "gen 3"), (3.0, "gen 4"),
       (8.0, "gen 5")]
print(f"{'':>10}{'exact match':>14}{'reported as':>26}"
      f"{'token accuracy':>17}{'reported as':>22}")
print("-" * 89)
for s, label in GEN:
    p = per_token(s)
    em = p ** ANSWER_TOKENS
    a = "no capability" if em < 0.01 else ("marginal" if em < 0.1 else "works")
    b = ("no capability" if p < 0.15 else
         "clear progress" if p < 0.6 else "works")
    print(f"{label:>10}{em:>14.4f}{a:>26}{p:>17.3f}{b:>22}")

print()
print()
print("And the cost of the wrong choice, in decisions rather than numbers.")
print()
print(f"{'decision':>34}{'under exact match':>20}{'under token accuracy':>23}")
print("-" * 77)
DECISIONS = [
    ("continue funding after gen 3",  "no",  "yes"),
    ("forecast gen 5 from gen 4",     "no",  "yes"),
    ("attribute the gen-5 gain",      "to scale", "to steady progress"),
    ("set a target for gen 6",        "unable",   "extrapolable"),
    ("compare two 1B candidates",     "tied near 0", "separable"),
]
for d, a, b in DECISIONS:
    print(f"{d:>34}{a:>20}{b:>23}")

print(f"""
The first table is the entire mechanism and it fits in three columns. One capability,
improving smoothly -- {tab[0.3][0]:.3f} to {tab[180.0][0]:.3f} per token across the range
-- and an exact-match score that reads {tab[3.0][1]:.4f} at 3B and
{tab[180.0][1]:.4f} at 180B.

Nothing jumped. The exponent did the work: **raising a smooth curve to the fifth power
produces a curve that looks like a threshold** (eq:metric-choice-manufactures-the-finding),
and the threshold's location is a property of the exponent rather than of the model.

The emergence table makes that concrete by varying only the answer length. The same
capability "appears" at {emerge['exact match, 2 tokens'][0]}B if the answer is two tokens
and {emerge["exact match, 12 tokens"][0]}B
if it is twelve. **The task did not change. The formatting of the answer changed**, and with
it the published finding about where the ability emerged.

The abruptness table is what a reader of the chart would conclude. Exact match at 5 tokens
jumps {jumps['exact match (all 5 right)']:.1f}x between adjacent scales, which reads as a
phase change; mean token accuracy jumps {jumps['mean token accuracy']:.1f}x, which reads as
steady progress. Same underlying numbers, two incompatible scientific claims, and the
choice between them was made when somebody decided how to score the answer.

The predictability table is the practical loss and it is larger than the aesthetic one.
Extrapolating one scale step from the previous two points, the continuous metric is off by
**{mean_c:.0%}** on average and the discontinuous one by **{mean_d:.0%}** --
{mean_d / mean_c:.0f} times worse, from the same data
(eq:discontinuity-hides-progress). And the errors are worst exactly where the decision
gets made: at the low end, where the discontinuous metric reads near zero and therefore
carries no gradient to extrapolate along.

That is not a subtlety about charts. It means the discontinuous metric is unusable for
exactly the decisions evaluation exists to support: is this working, is it improving, and
how much more of this do we need?

The generations table says what that looks like inside an organisation. Under exact match,
generations one through four all report `no capability` or `marginal`. Under token
accuracy, generation three already reports `clear progress` and generation four confirms
it. **Four consecutive review cycles of real, measurable, compounding progress reported as
nothing happening** -- and a programme cancelled after generation three on the first metric
would have been continued on the second, with the same model in both rooms.

The decisions table is the summary worth carrying. Under a discontinuous metric you cannot
fund, forecast, attribute, target, or compare. Under a continuous one you can do all five.
That is the case for choosing metrics by their *derivative* rather than by their
interpretability, which is close to the opposite of how metrics are usually chosen.

One caution. Continuous metrics are not automatically better -- a per-token accuracy that
is high while the answer is wrong is measuring something the user does not receive. The
claim here is narrower and it is about *when* to use which: **a discontinuous metric is the
right acceptance test and the wrong progress signal**, and most teams own one metric and
use it for both.""")
