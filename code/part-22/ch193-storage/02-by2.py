# -*- coding: utf-8 -*-
# Extracted from: Chapter 193 — Storage: Databases, Vector Stores, and Object Storage
# Source: src/.../ch193-storage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every derived copy of a fact is another chance for the system to contradict itself.

An AI system stores the same fact several times over. The source document sits in
object storage; a chunked copy sits in a vector index; an extracted summary sits in a
database; a cached answer quoting it sits in a cache. Each is derived from the last
and each updates on its own schedule.

When the underlying fact changes, those copies converge at different times. During
the gap the system can retrieve one version and quote another
(eq:derived-copies-multiply-contradiction).

The window in which that is possible is E[max lag] - E[min lag] across the copies.
Since the source itself is never stale, that window is set by the DEEPEST copy --
which makes pipeline depth, not pipeline speed, the parameter that matters.
"""
import math

# Derived representations, in dependency order. Each is rebuilt from the one
# above it. (name, mean lag in minutes after the source changes, lag std dev)
PIPELINE = [
    ("source document",      0.0,   0.0),
    ("extracted text",       4.0,   3.0),
    ("chunk embeddings",    22.0,  26.0),
    ("summary row",         35.0,  40.0),
    ("cached answer",      110.0, 210.0),
]
CHANGE_PER_DAY = 3.2       # how often a given fact changes
QUERIES_PER_DAY = 900.0    # queries touching that fact
MINUTES = 1440.0
GRID = [i * 0.5 for i in range(0, 4801)]      # 0 to 2400 minutes


def cdf(t, mean, sd):
    """P(lag <= t) for a lognormal lag with the given mean and standard deviation.

    Lognormal because a rebuild lag is non-negative and right-skewed: it is
    usually near its typical value and occasionally much longer.
    """
    if mean <= 0.0:
        return 1.0
    if t <= 0.0:
        return 0.0
    if sd <= 0.0:
        return 1.0 if t >= mean else 0.0
    var = math.log(1.0 + (sd * sd) / (mean * mean))
    mu = math.log(mean) - var / 2.0
    return 0.5 * (1.0 + math.erf((math.log(t) - mu) / (math.sqrt(var) * math.sqrt(2.0))))


def disagreement_window(copies):
    """E[max lag] - E[min lag]: the expected minutes, after each change, during
    which at least one copy has caught up and at least one has not."""
    if len(copies) < 2:
        return 0.0
    step = GRID[1] - GRID[0]
    e_max = 0.0
    e_min = 0.0
    for t in GRID:
        all_done = 1.0
        none_done = 1.0
        for _, m, s in copies:
            f = cdf(t, m, s)
            all_done *= f
            none_done *= (1.0 - f)
        e_max += (1.0 - all_done) * step      # E[max] = integral of P(max > t)
        e_min += none_done * step             # E[min] = integral of P(min > t)
    return e_max - e_min


def contradiction_rate(copies):
    """P(a random query lands inside a disagreement window)."""
    w = disagreement_window(copies)
    return min(1.0, w * CHANGE_PER_DAY / MINUTES)


print("A fact, stored five times over. Each copy is rebuilt from the one above.")
print()
print(f"{'representation':>20}{'mean lag':>12}{'std dev':>10}"
      f"{'spread ratio':>15}")
print("-" * 57)
for name, m, sp in PIPELINE:
    ratio = sp / m if m else 0.0
    print(f"{name:>20}{m:>10.0f}m{sp:>9.0f}m{ratio:>15.2f}")

print()
print()
print("Contradiction rate as copies are added, one at a time. The window is")
print("E[slowest copy] - E[fastest copy] after each change to the fact.")
print()
print(f"{'copies kept':>13}{'deepest copy':>20}{'window':>11}"
      f"{'contradiction':>16}{'queries/day':>14}")
print("-" * 74)
rates = {}
for i in range(2, len(PIPELINE) + 1):
    sub = PIPELINE[:i]
    w = disagreement_window(sub)
    r = contradiction_rate(sub)
    rates[i] = (r, w)
    print(f"{i:>13}{sub[-1][0]:>20}{w:>10.0f}m{r:>16.2%}"
          f"{r * QUERIES_PER_DAY:>14.0f}")

print()
print()
print("Now hold the number of copies at five and vary how fast the fact changes.")
print("This is the parameter a product decision moves without noticing.")
print()
print(f"{'changes/day':>13}{'contradiction rate':>21}{'queries/day affected':>22}")
print("-" * 56)
byrate = {}
BASE_WINDOW = disagreement_window(PIPELINE)
for cpd in (0.2, 1.0, 3.2, 12.0, 48.0):
    r = min(1.0, BASE_WINDOW * cpd / MINUTES)
    byrate[cpd] = r
    print(f"{cpd:>13.1f}{r:>21.2%}{r * QUERIES_PER_DAY:>22.0f}")

print()
print()
print("Three ways to attack the window: speed up every stage, tighten the")
print("variance, or synchronise everything onto one schedule.")
print()
print(f"{'strategy':>36}{'worst lag':>12}{'window':>10}{'contradiction':>16}")
print("-" * 74)


def variant(scale_mean, scale_sd):
    return [(n, m * scale_mean, s * scale_sd) for n, m, s in PIPELINE]


OPTIONS = [
    ("as built",                             1.00, 1.00),
    ("all lags halved",                      0.50, 0.50),
    ("all lags quartered",                   0.25, 0.25),
    ("variance halved, means unchanged",     1.00, 0.50),
    ("variance quartered, means unchanged",  1.00, 0.25),
    ("all copies on one 110m schedule",      0.00, 0.00),
]
res = {}
for label, sm, ss in OPTIONS:
    if label.startswith("all copies on one"):
        v = [(n, 0.0 if m == 0 else 110.0, 0.0) for n, m, s in PIPELINE]
    else:
        v = variant(sm, ss)
    r = contradiction_rate(v)
    w = disagreement_window(v)
    worst = max(m for _, m, _ in v)
    res[label] = (r, worst, w)
    print(f"{label:>36}{worst:>10.0f}m{w:>9.0f}m{r:>16.2%}")

print()
print()
print("And the structural alternative: stop deriving. Serve the deep copies from")
print("the shallow ones at query time instead of materialising them.")
print()
print(f"{'design':>32}{'copies':>9}{'contradiction':>16}{'read cost':>12}")
print("-" * 69)
struct = {}
for label, keep, extra in (
        ("materialise all five",        5, 1.00),
        ("materialise four",            4, 1.35),
        ("materialise three",           3, 1.90),
        ("materialise two",             2, 3.10),
        ("source of truth only",        1, 6.40)):
    r = contradiction_rate(PIPELINE[:keep]) if keep >= 2 else 0.0
    struct[keep] = (r, extra)
    print(f"{label:>32}{keep:>9}{r:>16.2%}{extra:>11.2f}x")

print(f"""
The pipeline table looks unremarkable. Every lag is a number somebody chose
deliberately, and the slowest is under two hours. No individual row is alarming.

The second table is what those rows compose into. Two representations disagree for
{rates[2][1]:.0f} minutes after each change, giving a contradiction rate of
{rates[2][0]:.2%}. Five representations disagree for {rates[5][1]:.0f} minutes,
giving {rates[5][0]:.2%} (eq:derived-copies-multiply-contradiction) -- at
{QUERIES_PER_DAY:.0f} queries a day, {rates[5][0] * QUERIES_PER_DAY:.0f} queries
where the system can retrieve one version of a fact and quote another.

Look at where the growth comes from. Adding the cached answer -- ONE copy -- takes
the window from {rates[4][1]:.0f} minutes to {rates[5][1]:.0f}, nearly tripling it,
while adding the summary row before it added only
{rates[4][1] - rates[3][1]:.0f}. The window is the distance from the fastest copy to
the slowest, the source is never stale, and so **the deepest copy sets the window
almost single-handedly.**

**Nothing in any store's monitoring shows this.** Every store is healthy, every write
succeeded, every read returned exactly what it held. The inconsistency exists only
BETWEEN the stores, which is precisely where nobody is looking -- the same gap
ch:sd-architecture identified for semantic failure, relocated to the storage layer.

The change-rate table is why this is a product problem rather than an infrastructure
one. At {0.2:.1f} changes per day the contradiction rate is {byrate[0.2]:.2%}; at
{48.0:.0f} it is {byrate[48.0]:.2%}. A feature that makes documents editable, or a
migration that starts syncing an upstream system hourly, moves this parameter by an
order of magnitude with no storage change at all and no review that would catch it.

The strategy table is where the intuition carried over from ch:sd-async breaks, and
the break is worth being precise about. There, variance did the damage and reducing
the mean barely helped. Here it is the other way round: halving every lag and its
variance takes contradiction from {res['as built'][0]:.2%} to
{res['all lags halved'][0]:.2%}, while halving the variance alone reaches only
{res['variance halved, means unchanged'][0]:.2%} and quartering it
{res['variance quartered, means unchanged'][0]:.2%}.

Synchronising is worth even less. Putting every derived copy on one
{110.0:.0f}-minute schedule gives {res['all copies on one 110m schedule'][0]:.2%} --
almost no improvement -- and the reason is the one thing the two chapters do share.
A queue's wait depends on the spread of service times because every job is compared
against every other. **A staleness window is measured against the source, and the
source is instant.** You cannot synchronise a derived copy to a thing that was never
late; there is no schedule slow enough to close a gap whose other end is zero.

So the lever here is neither speed nor synchronisation. It is DEPTH, and the last
table prices it. Dropping the cached answer -- the single deepest copy -- takes
contradiction from {struct[5][0]:.2%} to {struct[4][0]:.2%}, better than halving
every lag in the pipeline achieved, and it costs {struct[4][1]:.2f} times the read
cost rather than an infrastructure project. Dropping to three copies reaches
{struct[3][0]:.2%} at {struct[3][1]:.2f} times.

**Removing one derived copy beat making the entire pipeline twice as fast.** That is
the chapter's practical result, and it inverts the usual response, which is to keep
every materialisation and buy freshness with engineering.

It is worth naming what a materialised derived copy actually is, because the name
makes the decision obvious. **It is a cache** -- precomputed output kept because
recomputing is expensive -- and every cache over a changing source is a bet that
staleness costs less than recomputation. Here the bet is made implicitly, once, by
whoever built the pipeline, and it is never revisited when the change rate moves,
which eq:cache-threshold-is-an-error-cost-decision says is exactly when it should
be.""")
