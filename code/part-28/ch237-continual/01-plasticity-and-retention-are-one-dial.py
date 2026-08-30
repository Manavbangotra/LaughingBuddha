# -*- coding: utf-8 -*-
# Extracted from: Chapter 237 — Continual, Online, and Self-Improving Systems
# Source: src/.../ch237-continual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Retention and plasticity are one dial, and the update cadence has an interior optimum.

A deployed model faces a moving world (cite:gama2014). The obvious response is to keep training
it on new data. The obvious problem is that training on new data degrades what it knew
(cite:kirkpatrick2017ewc).

Those are not two problems to be balanced. They are one quantity read in two directions: any
mechanism that makes the model absorb new information faster makes it discard old information
faster, and the regularisation that protects the old blocks the new
(eq:plasticity-and-retention-are-one-dial).

Which turns the operational question into a cadence question. Update rarely and the model is
stale; update often and it forgets and oscillates. Both costs are monotone in opposite
directions, so there is an interior optimum
(eq:update-cadence-has-an-interior-optimum).
"""
import math

DRIFT_HALF_LIFE = 90.0     # days for half the distribution to move
BASE_ACC = 0.880


def staleness(days):
    """Accuracy lost to a distribution that has moved since the last update."""
    return 0.145 * (1.0 - 0.5 ** (days / DRIFT_HALF_LIFE))


def absorbed(plasticity, days):
    """Share of the new distribution the update actually takes on."""
    return 1.0 - math.exp(-plasticity * days / 12.0)


def forgotten(plasticity):
    """Share of prior capability lost per update, at a given plasticity."""
    return 0.062 * plasticity ** 1.35


print("The dial: one parameter, two effects.")
print()
print(f"{'plasticity':>13}{'new material absorbed':>24}{'prior capability lost':>24}"
      f"{'net after one update':>23}")
print("-" * 84)
DAYS = 30
dial = {}
for p in (0.10, 0.25, 0.50, 1.00, 2.00, 4.00):
    a = absorbed(p, DAYS)
    f = forgotten(p)
    net = staleness(DAYS) * a - f
    dial[p] = (a, f, net)
    print(f"{p:>13.2f}{a:>24.3f}{f:>24.4f}{net:>23.4f}")

BEST_P = max(dial, key=lambda p: dial[p][2])
print()
print(f"best plasticity at a {DAYS}-day cadence: {BEST_P:.2f}, net {dial[BEST_P][2]:+.4f}")
print(f"the extremes: {dial[0.10][2]:+.4f} at 0.10, {dial[4.00][2]:+.4f} at 4.00")

print()
print()
print("Regularisation moves along the dial; it does not remove it.")
print()
METHODS = [
    ("no regularisation",         1.00, 1.00, "--"),
    ("replay 5% old data",        0.94, 0.62, "storage + a pass"),
    ("replay 25% old data",       0.81, 0.31, "4x the storage"),
    ("elastic weight penalty",    0.77, 0.28, "cite:kirkpatrick2017ewc"),
    ("adapters, base frozen",     0.58, 0.05, "serving complexity"),
    ("retrain from scratch",      1.00, 0.00, "the full training cost"),
]
print(f"{'method':>26}{'absorption kept':>18}{'forgetting kept':>18}"
      f"{'net at plasticity 1.0':>24}{'what it costs':>26}")
print("-" * 112)
meth = {}
BASE_A, BASE_F = absorbed(1.0, DAYS), forgotten(1.0)
for name, keep_a, keep_f, cost in METHODS:
    a, f = BASE_A * keep_a, BASE_F * keep_f
    net = staleness(DAYS) * a - f
    meth[name] = (a, f, net)
    print(f"{name:>26}{keep_a:>18.2f}{keep_f:>18.2f}{net:>24.4f}{cost:>26}")

BEST_M = max(meth, key=lambda n: meth[n][2])
print()
print(f"best net: {BEST_M} at {meth[BEST_M][2]:+.4f}")
print(f"against {meth['no regularisation'][2]:+.4f} unregularised")

print()
print()
print("Now cadence, where the two costs are monotone and opposite.")
print()
UPDATE_COST = 42_000.0
REGRESSION_COST = 180_000.0
ACC_VALUE = 18_000_000.0      # dollars per unit of accuracy per year


def regression_risk(days):
    """Smaller, more frequent updates carry noisier data and more regressions."""
    return 0.35 * math.exp(-days / 25.0) + 0.05


print(f"{'days between updates':>22}{'updates / year':>16}{'mean staleness':>17}"
      f"{'regression risk':>18}{'staleness cost':>17}{'update cost':>14}"
      f"{'total':>14}")
print("-" * 118)
cad = {}
for days in (1, 7, 14, 30, 60, 90, 180, 365):
    n = 365.0 / days
    mean_stale = staleness(days) / 2.0
    risk = regression_risk(days)
    stale_cost = mean_stale * ACC_VALUE
    upd_cost = n * (UPDATE_COST + risk * REGRESSION_COST)
    cad[days] = (mean_stale, risk, stale_cost, upd_cost, stale_cost + upd_cost)
    print(f"{days:>22}{n:>16.1f}{mean_stale:>17.4f}{risk:>18.3f}"
          f"{stale_cost:>17,.0f}{upd_cost:>14,.0f}{stale_cost + upd_cost:>14,.0f}")

BEST_D = min(cad, key=lambda d: cad[d][4])
print()
print(f"cheapest cadence: every {BEST_D} days at {cad[BEST_D][4]:,.0f} per year")
print(f"daily updates cost {cad[1][4] / cad[BEST_D][4]:.1f}x that;"
      f" annual ones {cad[365][4] / cad[BEST_D][4]:.1f}x")
print("(eq:update-cadence-has-an-interior-optimum)")

print()
print()
print("And the optimum moves with how fast the world moves.")
print()
print(f"{'drift half-life (days)':>24}{'best cadence (days)':>22}"
      f"{'annual cost':>16}{'cost at a 30-day cadence':>27}{'penalty':>11}")
print("-" * 100)
CANDIDATES = (1, 3, 7, 14, 30, 60, 90, 180, 365)
for half in (14, 30, 90, 180, 365):
    best, best_c, at30 = None, None, None
    for days in CANDIDATES:
        stale = 0.145 * (1.0 - 0.5 ** (days / half)) / 2.0
        n = 365.0 / days
        total = stale * ACC_VALUE + n * (UPDATE_COST
                                         + regression_risk(days) * REGRESSION_COST)
        if days == 30:
            at30 = total
        if best_c is None or total < best_c:
            best, best_c = days, total
    print(f"{half:>24}{best:>22}{best_c:>16,.0f}{at30:>27,.0f}{at30 / best_c:>10.2f}x")

print()
print("A fixed cadence is right for exactly one drift rate.")

print()
print()
print("What you can actually measure, and what it would tell you.")
print()
SIGNALS = [
    ("accuracy on a fresh labelled sample", 0.91, 14, "labels, delayed"),
    ("accuracy on the frozen eval set",     0.34, 0,  "ch:ops-prompt-versioning"),
    ("input feature drift",                 0.58, 0,  "no labels needed"),
    ("output distribution shift",           0.49, 0,  "no labels needed"),
    ("user-reported failures",              0.72, 21, "biased sample"),
    ("disagreement with a held-out model",  0.66, 1,  "needs a second model"),
]
print(f"{'signal':>38}{'correlation with true loss':>29}{'lag (days)':>13}"
      f"{'catch':>26}")
print("-" * 106)
for name, corr, lag, note in SIGNALS:
    print(f"{name:>38}{corr:>29.2f}{lag:>13}{note:>26}")

best_sig = max(SIGNALS, key=lambda s: s[1] / (1 + s[2] / 30.0))
print()
print(f"best signal per unit of lag: {best_sig[0]}")
print(f"the frozen evaluation set correlates {0.34:.2f} -- it is the one everyone watches")

print(f"""
The dial table is the framing this listing exists for. At plasticity {0.10:.2f} an update absorbs
{dial[0.10][0]:.3f} of the new distribution and loses {dial[0.10][1]:.4f} of prior capability.
At {4.00:.2f} it absorbs {dial[4.00][0]:.3f} and loses {dial[4.00][1]:.4f}.

Both columns move together because they are the same mechanism
(eq:plasticity-and-retention-are-one-dial). A model that changes its weights readily in response
to new data changes them readily away from what the old data put there. There is no setting that
is plastic about the new and rigid about the old, because the model does not know which is which.

The net column has an interior maximum at plasticity {BEST_P:.2f} -- and note that the extremes
are bad for opposite reasons: {dial[0.10][2]:+.4f} because nothing was learned,
{dial[4.00][2]:+.4f} because too much was lost.

The regularisation table is what the field has built to move along that dial. Every row keeps
some absorption and blocks some forgetting, and the ratio is the whole story.
`{BEST_M}` reaches {meth[BEST_M][2]:+.4f} against
{meth['no regularisation'][2]:+.4f} unregularised.

`retrain from scratch` is worth reading carefully: it keeps {1.00:.2f} absorption and
{0.00:.2f} forgetting, which is the best possible pair, and the last column says what that
costs. **The clean answer exists and it is priced out of the loop** -- which is why the rest of
the table exists.

`adapters, base frozen` is the row that has quietly won in practice, keeping
{0.05:.2f} of the forgetting for {0.58:.2f} of the absorption, and its cost is the serving
complexity ch:res-test-time priced.

The cadence table is the operational result (eq:update-cadence-has-an-interior-optimum).
Staleness falls as updates get more frequent; update cost and regression risk both rise, because
a more frequent update is fitted to less data and carries more of the variance
ch:ops-deployment charged to releases.

The cheapest cadence is **every {BEST_D} days at {cad[BEST_D][4]:,.0f} per year**. Daily updates
cost **{cad[1][4] / cad[BEST_D][4]:.0f} times** that and annual updates
{cad[365][4] / cad[BEST_D][4]:.1f} times.

Both failure directions are expensive and they look completely different. Updating too rarely
shows up as a slow, uniform decline that nobody attributes to anything. Updating too often shows
up as capability regressions on things that used to work -- **visible, alarming, and blamed on
the last change rather than on the cadence.**

The drift table says the optimum is not a constant, and it does not move the way intuition
says. A fixed 30-day schedule carries a penalty of between 1.14x and 2.21x depending on the
drift rate, and the best cadence at a {14}-day half-life is *slower* than at a {30}-day one.

That is worth pausing on. When the world moves much faster than you can retrain, staleness is
saturated at every cadence you can afford, so buying updates buys nothing and you should buy
fewer of them. **Very fast drift is an argument against frequent retraining**, not for it -- and
the correct response there is a different architecture, not a shorter schedule.

**Nobody measures their drift half-life**, and it is the parameter that sets the schedule -- and,
per the row above, sometimes tells you the schedule is not the lever.

The signals table is why. The quantity you need is loss on the current distribution, and the
only signal that measures it directly -- {0.91:.2f} correlation -- needs fresh labels and arrives
{14} days late. The signal everyone actually watches is the frozen evaluation set, which
correlates **{0.34:.2f}**, because it measures accuracy on a distribution that stopped existing
(ch:ops-prompt-versioning's `evaluation-sets-decay-silently`).

The label-free signals are the interesting middle: input drift at {0.58:.2f} and output shift at
{0.49:.2f}, both available immediately and neither sufficient alone. **A continual system needs a
drift estimate more than it needs a better update rule**, and the drift estimate is the cheaper
thing to build.""")
