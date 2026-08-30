# -*- coding: utf-8 -*-
# Extracted from: Chapter 179 — Agentic EDA, Cleaning, and Visualization
# Source: src/.../ch179-agentic-eda.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Cleaning decisions are choices, and choices move the answer.

"Cleaning" sounds like correction -- removing errors, restoring the data to what
it should have been. Most cleaning decisions are not that. They are choices among
defensible options:

  missing values      drop the rows, impute the mean, impute by group, flag
  outliers            keep, winsorise, trim at 1%, trim at 5%
  duplicates          keep first, keep last, merge
  a category with 12  keep, fold into 'other', drop
  timezone at a boundary  the event's local time, or UTC

Each is defensible. Each produces a different dataset, and a different answer to
the same question. The spread across defensible pipelines is a real quantity, and
in many analyses it is larger than the effect being estimated
(eq:cleaning-choices-move-the-answer).

A human making these choices at least knows they chose. An agent produces one
number and a paragraph explaining what it did.
"""
import numpy as np
import itertools

rng = np.random.default_rng(4637)

M = 4000                # datasets
TRUE_EFFECT = 0.30      # the quantity being estimated, in whatever units

# (decision, options, how much each option shifts the estimate)
DECISIONS = [
    ("missing values",  ["drop rows", "impute mean", "impute by group", "flag"],
     [-0.09, +0.06, +0.01, -0.02]),
    ("outliers",        ["keep", "winsorise", "trim 1%", "trim 5%"],
     [+0.11, +0.02, -0.03, -0.12]),
    ("duplicates",      ["keep first", "keep last", "merge"],
     [-0.04, +0.05, +0.00]),
    ("rare categories", ["keep", "fold to other", "drop"],
     [+0.02, -0.01, -0.07]),
    ("timezone",        ["local", "UTC"],
     [+0.03, -0.03]),
]

NOISE = 0.05            # sampling noise on any single estimate


def estimate(choice_idx, m=M, true=TRUE_EFFECT, noise=NOISE):
    """The estimate produced by one specific pipeline."""
    shift = sum(DECISIONS[d][2][c] for d, c in enumerate(choice_idx))
    return true + shift + rng.normal(0, noise, m)


ALL = list(itertools.product(*[range(len(d[1])) for d in DECISIONS]))

print(f"An analysis with {len(DECISIONS)} cleaning decisions and")
print(f"{len(ALL)} defensible combinations. The true effect is {TRUE_EFFECT:.2f};")
print(f"sampling noise on any single estimate is {NOISE:.2f}.")
print()
print(f"{'decision':>18}{'options':>10}{'range of shift':>17}")
print("-" * 46)
for name, opts, shifts in DECISIONS:
    print(f"{name:>18}{len(opts):>10}{max(shifts) - min(shifts):>17.2f}")

print()
print()
print("Every defensible pipeline, run. The spread is the multiverse.")
print()
means = np.array([estimate(c).mean() for c in ALL])
print(f"{'true effect':>26}{TRUE_EFFECT:>10.3f}")
print(f"{'sampling noise (1 sd)':>26}{NOISE:>10.3f}")
print(f"{'lowest defensible estimate':>26}{means.min():>10.3f}")
print(f"{'highest defensible estimate':>26}{means.max():>10.3f}")
print(f"{'spread across pipelines':>26}{means.max() - means.min():>10.3f}")
print()
print(f"   The spread is {(means.max() - means.min()) / NOISE:.1f}x the sampling")
print(f"   noise and {(means.max() - means.min()) / TRUE_EFFECT:.0%} of the effect"
      f" being measured.")

print()
print()
print("What a single reported number could have been. Percentiles across the")
print("defensible pipelines:")
print()
for q in (0, 10, 25, 50, 75, 90, 100):
    print(f"{f'p{q}':>10}{np.percentile(means, q):>12.3f}")

print()
print()
print("Which decisions carry the spread. Range of the estimate as each single")
print("decision varies, with the others held at their first option:")
print()
print(f"{'decision':>18}{'range':>10}{'share of total':>16}")
print("-" * 44)
tot_var = 0.0
per = {}
for d, (name, opts, shifts) in enumerate(DECISIONS):
    base = [0] * len(DECISIONS)
    vals = []
    for c in range(len(opts)):
        base[d] = c
        vals.append(estimate(tuple(base)).mean())
    per[name] = max(vals) - min(vals)
    tot_var += per[name]
for name in per:
    print(f"{name:>18}{per[name]:>10.3f}{per[name] / tot_var:>16.1%}")

print()
print()
print("The sign question, which is what a decision actually turns on. Share of")
print("defensible pipelines that would support each conclusion, for three")
print("possible decision thresholds:")
print()
print(f"{'threshold':>12}{'above':>10}{'below':>10}{'verdict':>22}")
print("-" * 54)
th = {}
for t in (0.20, 0.30, 0.40):
    above = float((means > t).mean())
    th[t] = above
    verdict = ("unanimous" if above > 0.99 or above < 0.01
               else "contested" if 0.2 < above < 0.8 else "leaning")
    print(f"{t:>12.2f}{above:>10.1%}{1 - above:>10.1%}{verdict:>22}")

print()
print()
print("And what reporting the multiverse costs, against reporting one number.")
print()
one = estimate(ALL[0])
print(f"{'reporting style':>26}{'what the reader gets':>34}")
print("-" * 62)
print(f"{'a single pipeline':>26}{f'{one.mean():.3f} +/- {one.std():.3f}':>34}")
print(f"{'the multiverse':>26}"
      f"{f'{means.min():.3f} to {means.max():.3f} across {len(ALL)}':>34}")
print()
print(f"   The single-pipeline interval has width {2 * one.std():.3f}.")
print(f"   The multiverse spread is {means.max() - means.min():.3f}, "
      f"{(means.max() - means.min()) / (2 * one.std()):.1f}x wider.")

print(f"""
The spread is the whole listing. A true effect of {TRUE_EFFECT:.2f} produces
defensible estimates from {means.min():.3f} to {means.max():.3f}.

The low end has **the opposite sign from the truth**, and every pipeline that
produced it is defensible -- drop the missing rows, trim at {5}%, keep the first
duplicate. Nobody did anything wrong.

Against sampling noise of {NOISE:.2f}, the spread is
{(means.max() - means.min()) / NOISE:.1f} times larger
(eq:cleaning-choices-move-the-answer). Against the effect being measured it is
{(means.max() - means.min()) / TRUE_EFFECT:.0%}.

The last comparison is the one to take to an argument. A single pipeline reports
{one.mean():.3f} with an interval of width {2 * one.std():.3f}. The multiverse
spans {means.max() - means.min():.3f}, which is
{(means.max() - means.min()) / (2 * one.std()):.1f} times wider.

**The reported uncertainty describes sampling and omits the analysis**, and the
omitted part is the larger one. That is not a criticism of confidence intervals;
they measure what they claim to measure. It is a statement about what a single
number from a single pipeline can mean.

The decision table makes it concrete. At a threshold of {0.20:.2f},
{th[0.20]:.1%} of defensible pipelines support acting and {1 - th[0.20]:.1%}
support not acting. The analysis does not settle the question; the cleaning
choices do.

The per-decision table says where to look. Outlier handling carries
{per['outliers'] / tot_var:.1%} of the spread and timezone handling
{per['timezone'] / tot_var:.1%}, so the two are not equally worth discussing --
which is useful, because reporting a full multiverse is impractical and reporting
the two decisions that carry {(per['outliers'] + per['missing values']) / tot_var:.0%}
of it is not.

Now the part that concerns automation specifically.

Every one of these choices has to be made, and an agent makes them. It makes them
quickly, it makes them consistently, and it makes them **invisibly** -- the output
is a number and a paragraph saying what was done, which reads as a description of
the data rather than as a decision about it.

A human analyst is not better at choosing. They are, however, aware of having
chosen, and that awareness is what produces the sentence "we also tried it
winsorised and it did not change much" -- or, more importantly, its absence when it
did.

**The risk of automating cleaning is not that the agent chooses badly. It is that
the choice stops being visible as a choice**, and the multiverse collapses to a
point estimate whose uncertainty is understated
{(means.max() - means.min()) / (2 * one.std()):.0f}-fold.

Which suggests the intervention, and it is one automation is unusually good at. A
human cannot run {len(ALL)} pipelines. An agent can run them in the time it takes
to run one, and report the spread instead of the point.

**The same speed that makes exploratory automation dangerous makes multiverse
reporting free.** It is the one place in this part where the agent's advantage
lines up with the methodological need.""")
