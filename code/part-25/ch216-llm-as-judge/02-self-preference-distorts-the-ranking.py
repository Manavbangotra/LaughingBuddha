# -*- coding: utf-8 -*-
# Extracted from: Chapter 216 — LLM-as-a-Judge
# Source: src/.../ch216-llm-as-judge.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
