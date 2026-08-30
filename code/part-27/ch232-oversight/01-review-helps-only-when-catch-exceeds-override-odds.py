# -*- coding: utf-8 -*-
# Extracted from: Chapter 232 — Human Oversight in Practice
# Source: src/.../ch232-oversight.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A human in the loop is a claim about a team, and teams are measurable.

"A human reviews the output" is the most common answer to every question in the previous four
chapters -- bias, interpretability, privacy, regulation. It is also the least measured.

The claim is that the pair is more accurate than the model alone. That is a testable claim with
a closed form: the reviewer helps only when the rate at which they catch the model's errors
exceeds the rate at which they overturn its correct answers, scaled by the model's own odds of
being right (eq:review-helps-only-when-catch-exceeds-override-odds).

The second half asks what the reviewer is given to review with. An explanation raises the
reviewer's confidence whether or not it reflects the model's actual computation
(cite:turpin2023faithfulness), and confidence and accuracy move at different rates
(eq:an-explanation-raises-confidence-faster-than-accuracy).
"""
# (task, model accuracy, human-alone accuracy, catch rate on model errors,
#  false-override rate on model's correct answers)
TASKS = [
    ("content policy call",        0.91, 0.94, 0.38, 0.02),
    ("loan adjudication",          0.87, 0.79, 0.31, 0.03),
    ("clinical triage note",       0.83, 0.88, 0.46, 0.02),
    ("security review of a diff",  0.71, 0.76, 0.29, 0.05),
    ("fraud alert disposition",    0.94, 0.81, 0.22, 0.04),
]


def team(a_m, c, f):
    """Accuracy of model-proposes / human-disposes."""
    return a_m * (1 - f) + (1 - a_m) * c


print("Model alone, human alone, and the two together.")
print()
print(f"{'task':>30}{'model':>9}{'human':>9}{'team':>9}"
      f"{'vs model':>11}{'vs better member':>19}")
print("-" * 87)
gain_better, beats = 0.0, 0
for name, a_m, a_h, c, f in TASKS:
    t = team(a_m, c, f)
    gm = t - a_m
    gb = t - max(a_m, a_h)
    gain_better += gb
    beats += 1 if gb > 0 else 0
    print(f"{name:>30}{a_m:>9.3f}{a_h:>9.3f}{t:>9.3f}"
          f"{gm * 100:>+10.1f}p{gb * 100:>+18.1f}p")

beats_model = sum(1 for n, am, ah, c, f in TASKS if team(am, c, f) > am)
print()
print(f"the team beats the model alone in {beats_model} of {len(TASKS)} tasks")
print(f"the team beats the better of the two members in {beats} of {len(TASKS)}")
print(f"mean gain over the better member: {gain_better / len(TASKS) * 100:+.2f} points")

print()
print()
print("Why: the reviewer both catches errors and creates them.")
print()
print(f"{'task':>30}{'catch rate':>13}{'false override':>17}"
      f"{'ratio needed':>15}{'ratio actual':>15}{'helps?':>9}")
print("-" * 99)
for name, a_m, a_h, c, f in TASKS:
    need = a_m / (1 - a_m)
    have = c / f
    print(f"{name:>30}{c:>13.2f}{f:>17.2f}{need:>15.1f}{have:>15.1f}"
          f"{('yes' if have > need else 'no'):>9}")

print()
print("A reviewer helps the model only when catch/override exceeds the model's")
print("own odds of being right (eq:review-helps-only-when-catch-exceeds-override-odds).")

print()
print()
print("Which means the bar rises as the model improves.")
print()
C_FIX, F_FIX = 0.35, 0.03
print(f"holding the reviewer fixed at catch={C_FIX:.2f}, override={F_FIX:.2f}")
print()
print(f"{'model accuracy':>18}{'odds of right':>16}{'team accuracy':>16}"
      f"{'change':>11}{'review is':>13}")
print("-" * 74)
for a_m in (0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.99):
    t = team(a_m, C_FIX, F_FIX)
    d = t - a_m
    print(f"{a_m:>18.2f}{a_m / (1 - a_m):>16.1f}{t:>16.4f}"
          f"{d * 100:>+10.2f}p{('helping' if d > 0 else 'harmful'):>13}")

BREAK = (C_FIX / F_FIX) / (1 + C_FIX / F_FIX)
print()
print(f"break-even model accuracy: {BREAK:.3f}")
print("above that, this review process makes the system worse")

print()
print()
print("What the reviewer is given to review with.")
print()
# (what the reviewer sees, false-override rate, catch rate, self-reported confidence)
SHOWN = [
    ("the output alone",                       0.030, 0.29, 0.54),
    ("output + model confidence, calibrated",  0.028, 0.41, 0.63),
    ("output + confidence, overconfident",     0.019, 0.24, 0.71),
    ("output + a faithful explanation",        0.026, 0.47, 0.69),
    ("output + a plausible wrong explanation", 0.014, 0.19, 0.78),
    ("output + the source documents",          0.031, 0.52, 0.66),
]
A_M = 0.88
print(f"{'what the reviewer sees':>42}{'override':>11}{'catch':>9}"
      f"{'team acc':>11}{'confidence':>13}")
print("-" * 86)
shown_acc = {}
for name, f, c, conf in SHOWN:
    t = team(A_M, c, f)
    shown_acc[name] = t
    print(f"{name:>42}{f:>11.3f}{c:>9.2f}{t:>11.4f}{conf:>13.2f}")

best_shown = max(shown_acc, key=lambda n: shown_acc[n])
print()
print(f"best: {best_shown} at {shown_acc[best_shown]:.4f}")

print()
print()
print("Confidence and accuracy do not move together.")
print()
print(f"{'what the reviewer sees':>42}{'confidence':>13}{'team accuracy':>16}"
      f"{'confidence per point':>23}")
print("-" * 94)
BASE_CONF = SHOWN[0][3]
BASE_ACC = shown_acc[SHOWN[0][0]]
for name, f, c, conf in SHOWN:
    d_conf = conf - BASE_CONF
    d_acc = shown_acc[name] - BASE_ACC
    if abs(d_acc) < 1e-9:
        rs = f"{'(baseline)':>23}"
    else:
        rs = f"{d_conf / d_acc:>23.1f}"
    print(f"{name:>42}{conf:>13.2f}{shown_acc[name]:>16.4f}{rs}")

WRONG = "output + a plausible wrong explanation"
WRONG_CONF = [s[3] for s in SHOWN if s[0] == WRONG][0]
print()
print(f"a plausible wrong explanation raises confidence from {BASE_CONF:.2f}"
      f" to {WRONG_CONF:.2f}")
print(f"and moves team accuracy from {BASE_ACC:.4f} to {shown_acc[WRONG]:.4f}")

print()
print()
print("And what pointing the reviewer at the right items is worth.")
print()
print(f"{'routing policy':>36}{'reviewed':>11}{'errors covered':>17}"
      f"{'team accuracy':>16}{'minutes / 1k':>15}")
print("-" * 95)
MIN_PER_ITEM = 1.5
ROUTING = [
    ("review everything",                   1.00, 0.35, 0.030),
    ("review the bottom 20% by confidence",  0.20, 0.61, 0.021),
    ("review the bottom 5% by confidence",   0.05, 0.74, 0.017),
    ("oracle: review exactly the errors", 1 - A_M, 0.35, 0.000),
]
ORACLE = "oracle: review exactly the errors"
ALL = "review everything"
NARROW = "review the bottom 5% by confidence"
routed = {}
for name, share, c, f in ROUTING:
    covered = 1.0 if name == ORACLE else min(1.0, share * 2.6)
    t = team(A_M, c * covered, f * share)
    routed[name] = (share, covered, t, share * MIN_PER_ITEM * 1000)
    print(f"{name:>36}{share:>11.0%}{covered:>17.0%}{t:>16.4f}"
          f"{share * MIN_PER_ITEM * 1000:>15.0f}")

BEST_ROUTE = max((n for n, s, c, f in ROUTING if n != ORACLE),
                 key=lambda n: routed[n][2])
print()
print(f"best practical policy: {BEST_ROUTE} at {routed[BEST_ROUTE][2]:.4f}")
print(f"using {routed[ALL][3] / routed[BEST_ROUTE][3]:.0f}x less reviewer time"
      f" than reviewing everything")
print(f"perfect routing is worth {(routed[ORACLE][2] - A_M) * 100:.1f} points over the model")

print(f"""
The first table is the measurement almost nobody makes. Across {len(TASKS)} tasks, the
model-proposes / human-disposes team beats the model alone in {beats_model} of {len(TASKS)}
cases -- and beats **the better of its two members in only {beats}**, with a mean gain over that
better member of {gain_better / len(TASKS) * 100:+.2f} points.

That is not an argument against human oversight. It is an argument against assuming it. The
usual claim is that adding a reviewer can only help, and the arithmetic says otherwise: a
reviewer both catches errors and creates them, and which effect dominates is an empirical
question with a closed-form answer.

The second table gives the form. Team accuracy is `a_m(1 - f) + (1 - a_m)c`, so the reviewer
improves on the model exactly when `c/f > a_m/(1 - a_m)`
(eq:review-helps-only-when-catch-exceeds-override-odds). The reviewer's catch-to-override ratio
has to beat the model's own odds of being right.

`fraud alert disposition` fails that test. The model is right {0.94:.0%} of the time, so the bar
is {0.94 / 0.06:.1f}, and the reviewer's ratio is {0.22 / 0.04:.1f}. Every hour spent reviewing
those alerts makes the system slightly worse, and it looks exactly like diligence.

The third table is why this gets harder rather than easier. Holding a fixed reviewer at
catch={C_FIX:.2f} and override={F_FIX:.2f}, review helps up to a model accuracy of
**{BREAK:.3f}** and hurts above it. **The better the model gets, the harder it is for a human to
add anything** -- the reviewer's own mistakes are drawn from a pool that grows while the errors
they might catch are drawn from one that shrinks.

That is the same structure as ch:sec-permissions' approval queue reached from the opposite
direction. That chapter showed a low rejection rate trains approval; this one shows why the
rejection rate falls in the first place.

The fourth table is about what the reviewer is handed. `{best_shown}` produces the best team
accuracy at {shown_acc[best_shown]:.4f}, because it is the only row that lets the reviewer check
the claim rather than assess the presentation.

The row that matters most is `{WRONG}`. It drops the catch rate to {0.19:.2f} and the override
rate to {0.014:.3f} -- a reviewer who defers more and catches less -- while raising
self-reported confidence from {BASE_CONF:.2f} to {WRONG_CONF:.2f}.

The fifth table makes the divergence explicit: confidence rises fastest exactly where accuracy
does not (eq:an-explanation-raises-confidence-faster-than-accuracy). **An explanation is a
persuasion artefact before it is an evidence artefact**, which is ch:rai-interpretability's
`stated-reasons-need-not-be-actual-reasons` arriving as a measured effect on a person.

An overconfident confidence score does the same thing more cheaply -- {0.019:.3f} override,
{0.24:.2f} catch, {0.71:.2f} confidence. Which is why cite:guo2017calibration's calibration
result is an oversight requirement rather than a modelling nicety: an uncalibrated score is a
reliance signal pointing the wrong way, and cite:kadavath2022 is the question of whether the
model can supply a better one.

The last table has two findings.

The first is that reviewing everything is not the best use of the reviewer.
`{BEST_ROUTE}` reaches {routed[BEST_ROUTE][2]:.4f} against {routed[ALL][2]:.4f} for reviewing
everything, at {routed[BEST_ROUTE][3]:.0f} reviewer-minutes per thousand items instead of
{routed[ALL][3]:.0f} -- **higher accuracy at
{routed[ALL][3] / routed[BEST_ROUTE][3]:.0f} times less human time**, because concentrating
attention raises the catch rate and shrinks the surface on which the reviewer can introduce
errors.

Narrower is not monotonically better. Routing the bottom {0.05:.0%} covers only
{routed[NARROW][1]:.0%} of the model's errors and lands at {routed[NARROW][2]:.4f} -- below
reviewing everything. **Coverage collapses faster than precision improves**, which is
ch:ev-framework's union result in another costume.

The second finding is the harder one. The oracle row hands the reviewer *exactly* the model's
errors, with no opportunity to overturn a correct answer, and reaches {routed[ORACLE][2]:.4f} --
**{(routed[ORACLE][2] - A_M) * 100:.1f} points over the model**. A reviewer who catches
{C_FIX:.0%} of errors catches {C_FIX:.0%} of errors however well you point them.

**Routing is worth a great deal and it is not the binding constraint.** The catch rate is, and
the fourth table said what moves it: give the reviewer the sources, not the presentation.""")
