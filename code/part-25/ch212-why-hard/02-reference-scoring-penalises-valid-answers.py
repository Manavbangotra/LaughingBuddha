# -*- coding: utf-8 -*-
# Extracted from: Chapter 212 — Why Evaluating AI Is Hard
# Source: src/.../ch212-why-hard.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""For most useful tasks there is no ground truth, only one sample from a set of them.

A classifier has a label. A summariser does not -- it has a space of acceptable summaries,
and whichever one a human wrote down is a draw from that space rather than the truth about
it.

Reference-based scoring compares the model's draw against the reference's draw, so it
penalises every correct answer that happens to be a different one
(eq:reference-scoring-penalises-valid-answers).

And the human judgement that would settle it is itself noisy, which puts a ceiling on how
well any automated metric can correlate with quality no matter how good the metric is
(eq:agreement-caps-measurable-quality).

This listing measures both, prices the standard workarounds, and finds the one case where
the problem genuinely goes away.
"""
import math

# (task, |A| = size of the acceptable-answer space, note)
TASKS = [
    ("classify sentiment",          1.0,   "the label is the answer"),
    ("extract the invoice date",    2.0,   "two valid formats"),
    ("name the capital city",       1.4,   "occasional alias"),
    ("write a SQL query",          24.0,   "many correct queries"),
    ("summarise a paragraph",     180.0,   "many faithful summaries"),
    ("explain a concept",        2400.0,   "many correct explanations"),
    ("draft a reply email",     15000.0,   "many acceptable replies"),
]
TRUE_ACCURACY = 0.78          # share of the model's answers that ARE acceptable

print("Reference-based scoring credits the model only when its draw from the")
print("acceptable set matches the reference's draw.")
print()
print(f"{'task':>26}{'|A|':>10}{'P(match), 1 ref':>18}"
      f"{'measured':>11}{'valid answers scored wrong':>28}")
print("-" * 93)
tab = {}
for name, A, note in TASKS:
    hit = min(1.0, 1.0 / A)
    measured = TRUE_ACCURACY * hit
    tab[name] = (A, hit, measured, 1.0 - hit)
    print(f"{name:>26}{A:>10.1f}{hit:>18.3f}"
          f"{measured:>11.3f}{1.0 - hit:>28.2%}")

print()
print(f"true accuracy is {TRUE_ACCURACY:.0%} in every row. The last column is the")
print("share of the model's CORRECT answers that the metric marks wrong.")

print()
print()
print("Adding references buys coverage, and the curve is unkind.")
print()
print(f"{'references':>12}", end="")
for name, A, note in TASKS[3:]:
    print(f"{name.split()[-1]:>14}", end="")
print()
print("-" * 68)
multi = {}
for R in (1, 3, 5, 10, 25, 100):
    print(f"{R:>12}", end="")
    for name, A, note in TASKS[3:]:
        cov = min(1.0, R / A)
        multi[(R, name)] = cov
        print(f"{cov:>14.1%}", end="")
    print()

print()
print("A hundred references cover a summarisation space and do not touch an")
print("email space. Labelling cost is linear; coverage is not.")

print()
print()
print("What each standard workaround actually replaces |A| with.")
print()
print(f"{'approach':>26}{'what it measures':>34}{'penalty on task 5':>20}")
print("-" * 80)
SUMM_A = 180.0
APPROACHES = [
    ("single reference, exact",  "match to one arbitrary draw",   1 - 1 / SUMM_A),
    ("5 references, exact",      "match to five draws",           1 - 5 / SUMM_A),
    ("n-gram overlap",           "surface form near one draw",    0.42),
    ("embedding similarity",     "semantic distance to one draw", 0.29),
    ("LLM judge",                "the judge's acceptability set", 0.17),
    ("execution / unit tests",   "whether it works",              0.02),
]
appr = {}
for name, what, pen in APPROACHES:
    appr[name] = pen
    print(f"{name:>26}{what:>34}{pen:>20.1%}")

print()
print("Only the last one changes the problem instead of approximating it.")

print()
print()
print("The other ceiling: human labels are noisy, so no metric can correlate")
print("with quality better than the labels do with themselves.")
print()
print(f"{'annotator agreement':>21}{'reliability':>14}{'metric ceiling':>17}"
      f"{'best reported r':>18}{'headroom':>11}")
print("-" * 81)
REPORTED_R = 0.71             # a good automated metric's correlation with human scores
ceil = {}
for obs in (0.95, 0.88, 0.81, 0.74, 0.66):
    chance = 0.50
    kappa = (obs - chance) / (1 - chance)
    rel = kappa                       # treat kappa as the reliability of one label
    c = math.sqrt(max(rel, 0.0))      # attenuation: r_max = sqrt(reliability)
    ceil[obs] = (kappa, c)
    print(f"{obs:>21.0%}{kappa:>14.2f}{c:>17.2f}"
          f"{REPORTED_R:>18.2f}{c - REPORTED_R:>11.2f}")

print()
print("Below 81% raw agreement, a metric correlating at 0.71 is already at the")
print("ceiling -- and improving the metric cannot help.")

print()
print()
print("Putting both together: what a reported score means.")
print()
print(f"{'measurement design':>30}{'reports':>10}{'true':>8}"
      f"{'level usable?':>16}{'ranking usable?':>18}")
print("-" * 82)
DESIGNS = [
    ("single-reference exact match",  TRUE_ACCURACY / SUMM_A, "no",  "within task"),
    ("n-gram overlap",                TRUE_ACCURACY * 0.58,   "no",  "within task"),
    ("LLM judge",                     TRUE_ACCURACY * 0.83,   "approximately", "yes"),
    ("execution",                     TRUE_ACCURACY * 0.98,   "yes", "yes"),
    ("human, 2 annotators",           TRUE_ACCURACY * 0.94,   "yes", "yes"),
]
for name, rep, lvl, rank in DESIGNS:
    print(f"{name:>30}{rep:>10.3f}{TRUE_ACCURACY:>8.2f}{lvl:>16}{rank:>18}")

print()
print()
print("And the case where even the ranking fails: two systems whose answer")
print("spaces differ, which is what happens when one is more verbose.")
print()
print(f"{'system':>12}{'true quality':>15}{'|A| of its outputs':>21}"
      f"{'single-ref score':>18}{'true rank':>11}{'measured rank':>15}")
print("-" * 92)
SYSTEMS = [("terse", 0.72, 95.0), ("verbose", 0.78, 420.0)]
scores = {}
for name, q, A in SYSTEMS:
    scores[name] = (q, A, q / A)
by_true = sorted(SYSTEMS, key=lambda s: -s[1])
by_meas = sorted(SYSTEMS, key=lambda s: -scores[s[0]][2])
for name, q, A in SYSTEMS:
    tr = [s[0] for s in by_true].index(name) + 1
    mr = [s[0] for s in by_meas].index(name) + 1
    print(f"{name:>12}{q:>15.2f}{A:>21.0f}{scores[name][2]:>18.5f}"
          f"{tr:>11}{mr:>15}")
best_true = max(SYSTEMS, key=lambda s: s[1])[0]
best_meas = max(scores, key=lambda k: scores[k][2])
print()
print(f"better system: {best_true}    better score: {best_meas}")
print(f"score ratio: {scores[best_meas][2] / scores[best_true][2]:.1f}x the wrong way")

print(f"""
The first table is the mechanism and it needs one sentence. True accuracy is
{TRUE_ACCURACY:.0%} in every row, and the reported number ranges from
{tab['classify sentiment'][2]:.3f} to {tab['draft a reply email'][2]:.5f}
(eq:reference-scoring-penalises-valid-answers).

The difference between the rows is not model quality. It is **how many correct answers the
task has**, and the metric divides by that number.

Notice which tasks sit at each end. Classification is the one place reference scoring is
exact, and classification is also the task nobody deploys a language model for. The tasks
people actually ship -- summarise, explain, reply -- have acceptable-answer spaces in the
hundreds or thousands, and on those the single-reference metric marks
{tab['summarise a paragraph'][3]:.1%} to {tab['draft a reply email'][3]:.2%} of *correct*
answers wrong.

The multi-reference table is the standard fix and the table shows why it does not scale.
Five references cover {multi[(5, 'summarise a paragraph')]:.1%} of a summarisation space
and {multi[(5, 'draft a reply email')]:.2%} of an email space. A hundred references --
which is a serious annotation programme -- reach
{multi[(100, 'summarise a paragraph')]:.0%} and
{multi[(100, 'draft a reply email')]:.2%} respectively.

**Labelling cost is linear in R and coverage is R over |A|**, so the approach works exactly
when |A| is small, which is exactly when you did not need it.

The workaround table is the honest survey. Overlap and embedding metrics reduce the penalty
from {appr['single reference, exact']:.0%} to {appr['n-gram overlap']:.0%} and
{appr['embedding similarity']:.0%} by giving partial credit -- but read the middle column:
they are measuring *proximity to one arbitrary draw*, not acceptability. A judge does
better at {appr['LLM judge']:.0%} because it evaluates against a learned acceptability
boundary rather than a sample, which is ch:ev-llm-judge's subject and its own set of
problems.

Only execution changes the question. A unit test does not sample the acceptable set --
**it defines it**, collapsing |A| to one equivalence class by construction, which is why
cite:chen2021humaneval's pass@k and cite:jimenez2023swebench's test-graded issues are the
most trustworthy numbers in this book's evaluation chapters.

The lesson generalises past code: **wherever you can state an acceptance predicate instead
of writing an answer, do that.** It is usually possible for more tasks than teams assume,
and it is almost never the first thing tried.

The agreement table is the second ceiling and it is the one that ends arguments about
metric quality. At {0.81:.0%} raw agreement between annotators, kappa is
{ceil[0.81][0]:.2f} and the highest correlation any metric can achieve with the true
quality is {ceil[0.81][1]:.2f} (eq:agreement-caps-measurable-quality). A metric already
correlating at {REPORTED_R:.2f} has {ceil[0.81][1] - REPORTED_R:.2f} of headroom.

Below that agreement level the headroom is negative, which means **the metric is already
performing better than the labels it is being validated against**, and every further
improvement will be measured as a regression.

That is worth stating plainly because it is routinely misdiagnosed. A metric that stops
improving against human labels has either stopped improving or hit the annotation ceiling,
and the two look identical from the metric's side. The way to tell them apart is to measure
annotator agreement, which costs a double-labelled sample and is skipped almost universally.

The design table converts both results into what a score is good for. A single-reference
score of {TRUE_ACCURACY / SUMM_A:.4f} against a true {TRUE_ACCURACY:.2f} carries no usable
*level* -- you cannot tell a stakeholder the system is right {TRUE_ACCURACY / SUMM_A:.1%} of
the time -- but it may still rank two systems correctly on the same task.

Which is the defence usually offered for these metrics, and the last table is where it
fails. Two systems, true quality {SYSTEMS[0][1]:.2f} and {SYSTEMS[1][1]:.2f}, differing in
verbosity so that their answer spaces are {SYSTEMS[0][2]:.0f} and {SYSTEMS[1][2]:.0f}. The
better system scores {scores['verbose'][2] / scores['terse'][2]:.1f} times *lower*.

**Reference-based scoring is order-preserving only when the compared systems have the same
answer-space size**, and nothing about the comparison guarantees that -- in fact any change
that makes a system more expansive violates it. Which means a metric that has ranked
correctly for two years can invert the first time somebody makes the model more helpful.""")
