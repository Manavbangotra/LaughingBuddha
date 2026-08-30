# -*- coding: utf-8 -*-
# Extracted from: Chapter 177 — The AI-Assisted Data Science Stack
# Source: src/.../ch177-stack.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where the time goes, against where the automation landed.

cite:testini2025dsautomation surveyed how data science automation is evaluated and
found the coverage concentrated on a small subset of goal-oriented activities,
with data management and exploratory work largely ignored.

That is a selection effect with a mechanism. Benchmarks measure what can be
GRADED, and what can be graded is the part with a checkable answer: a query that
returns the right rows, a model that beats a threshold. The activities
practitioners spend most of their time on -- deciding what to ask, finding the
data, cleaning it -- have no reference answer, so they are not benchmarked, so
progress on them is not measured (eq:gradeable-is-not-representative).

This listing prices what that does to any claim of the form "agents now do X% of
data science".
"""
import numpy as np

# Activity shares are this listing's assumptions, stated so they can be
# challenged. The ordering -- data work dominating, modelling small -- is the
# consistent finding of practitioner surveys over two decades.
# (name, share of practitioner time, how gradeable, current automation quality)
ACTIVITIES = [
    ("framing the question",   0.12, 0.05, 0.20),
    ("finding and accessing",  0.15, 0.20, 0.35),
    ("cleaning and shaping",   0.26, 0.35, 0.55),
    ("exploration",            0.17, 0.10, 0.40),
    ("modelling",              0.11, 0.95, 0.75),
    ("validation",             0.08, 0.80, 0.60),
    ("communicating",          0.11, 0.15, 0.50),
]

TOTAL = sum(a[1] for a in ACTIVITIES)
assert abs(TOTAL - 1.0) < 1e-9, TOTAL

print("A data science project's activities, their share of practitioner time,")
print("how gradeable each is, and how well automation currently does it.")
print()
print(f"{'activity':>22}{'time share':>12}{'gradeable':>11}{'automation':>12}")
print("-" * 57)
for name, share, grade, auto in ACTIVITIES:
    print(f"{name:>22}{share:>12.0%}{grade:>11.0%}{auto:>12.0%}")

print()
print()
print("Benchmark attention follows gradeability, not time. Modelling a")
print("benchmark suite that allocates coverage in proportion to how gradeable")
print("an activity is:")
print()
g_total = sum(a[2] for a in ACTIVITIES)
print(f"{'activity':>22}{'time share':>12}{'benchmark share':>17}{'ratio':>9}")
print("-" * 60)
bench = {}
for name, share, grade, auto in ACTIVITIES:
    b = grade / g_total
    bench[name] = b
    print(f"{name:>22}{share:>12.0%}{b:>17.0%}{b / share:>9.2f}")

print()
print()
print("So what does a benchmark-weighted score actually claim? Comparing the")
print("headline number against the one that describes a practitioner's day:")
print()
bench_score = sum(bench[n] * a for n, _, _, a in ACTIVITIES)
time_score = sum(s * a for _, s, _, a in ACTIVITIES)
grade_frac = sum(s * g for _, s, g, _ in ACTIVITIES)
print(f"{'benchmark-weighted automation score':>40}{bench_score:>10.1%}")
print(f"{'time-weighted automation score':>40}{time_score:>10.1%}")
print(f"{'gradeable share of a practitioner day':>40}{grade_frac:>10.1%}")
print()
print(f"   The headline overstates the time-weighted figure by "
      f"{bench_score - time_score:.1f} points,")
print(f"   and only {grade_frac:.0%} of the day is gradeable at all.")

print()
print()
print("Amdahl's law on the analysis pipeline: perfect automation of one")
print("activity, and what it does to total project time.")
print()
print(f"{'activity fully automated':>26}{'time saved':>12}{'speedup':>10}")
print("-" * 48)
amd = {}
for name, share, grade, auto in ACTIVITIES:
    remaining = 1.0 - share
    amd[name] = (share, 1.0 / remaining)
    print(f"{name:>26}{share:>12.0%}{1.0 / remaining:>10.2f}x")

print()
print()
print("And the ceiling: automate everything above a gradeability threshold")
print("perfectly, leave the rest untouched.")
print()
print(f"{'gradeability threshold':>24}{'activities':>12}{'time covered':>14}"
      f"{'speedup':>10}")
print("-" * 61)
ceil = {}
for thr in (0.9, 0.7, 0.3, 0.15, 0.08):
    covered = [a for a in ACTIVITIES if a[2] >= thr]
    share = sum(a[1] for a in covered)
    sp = 1.0 / max(1.0 - share, 1e-9)
    ceil[thr] = (len(covered), share, sp)
    cell = "unbounded" if share > 0.999 else f"{sp:.2f}x"
    print(f"{thr:>24.0%}{len(covered):>12}{share:>14.0%}{cell:>10}")

print()
print()
print("The honest version of the automation figure: current capability applied")
print("to each activity's actual time share, swept over how good automation on")
print("the UNGRADEABLE activities becomes.")
print()
print(f"{'ungradeable automation':>24}{'time-weighted':>15}{'speedup':>10}")
print("-" * 50)
sw = {}
for lift in (0.0, 0.25, 0.50, 0.75, 1.0):
    total_auto = 0.0
    for name, share, grade, auto in ACTIVITIES:
        a = auto if grade >= 0.5 else auto + (1.0 - auto) * lift
        total_auto += share * a
    sw[lift] = (total_auto, 1.0 / max(1.0 - total_auto, 1e-9))
    print(f"{lift:>24.0%}{total_auto:>15.1%}"
          f"{1.0 / max(1.0 - total_auto, 1e-9):>10.2f}x")

print(f"""
The second table is the selection effect, made arithmetic. Modelling gets
{bench['modelling'] / 0.11:.2f} times its share of attention and validation
{bench['validation'] / 0.08:.2f} times, while framing the question gets
{bench['framing the question'] / 0.12:.2f} and exploration
{bench['exploration'] / 0.17:.2f}.

Nothing about that is a conspiracy. **Benchmarks measure what can be graded**, a
model score is a number and a well-posed question is not, so attention follows
gradeability (eq:gradeable-is-not-representative). cite:testini2025dsautomation
found exactly this distribution in the actual literature.

The consequence is in the summary. A benchmark-weighted reading of current
capability gives {bench_score:.1%}; weighting the same capabilities by where the
time actually goes gives {time_score:.1%}. **The headline overstates the
practitioner-relevant figure by {bench_score - time_score:.1f} points** -- and
only {grade_frac:.0%} of a practitioner's day is gradeable at all, so two thirds
of the work is in a region where progress is not being measured in either
direction.

The Amdahl table is the part worth carrying into an argument. Fully automating
modelling -- the single most-benchmarked activity, at
{bench['modelling'] / 0.11:.1f} times its time share of attention -- produces a
{amd['modelling'][1]:.2f}x speedup on the project.

That is not a claim that modelling automation is worthless. It is a claim that
**the size of a capability's benchmark presence tells you nothing about the size
of its effect**, because the benchmark measures a fraction of time that is small
precisely because it was tractable enough to automate first.

Cleaning and shaping, at {0.26:.0%} of the time, would give
{amd['cleaning and shaping'][1]:.2f}x. It gets {bench['cleaning and shaping'] / 0.26:.2f}
times its share of benchmark attention.

The threshold table gives the ceiling for a strategy of automating only what can
be checked. Perfect automation of everything above {0.7:.0%} gradeability -- two
activities -- gives {ceil[0.7][2]:.2f}x. **Every gradeable activity, automated
perfectly, is a {ceil[0.7][2]:.2f}x project speedup**, which is worth having and
is not the transformation the discourse describes.

The last table says where the transformation actually lives. Holding gradeable
automation where it is and lifting the UNGRADEABLE activities from
{0.0:.0%} to {1.0:.0%} of their remaining headroom takes the project from
{sw[0.0][1]:.2f}x to {sw[1.0][1]:.2f}x.

**The whole of the remaining opportunity is in the activities nobody can score.**
Which is an awkward place for it to be, because it means the field's ability to
measure its own progress runs out exactly where the value starts -- and it is why
this part spends more time on oversight and verification than on capability.""")
