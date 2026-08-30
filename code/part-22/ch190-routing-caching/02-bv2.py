# -*- coding: utf-8 -*-
# Extracted from: Chapter 190 — Model Routing, Caching, and Cost Optimization
# Source: src/.../ch190-routing-caching.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A semantic cache threshold is an error-cost decision, not a hit-rate decision.

ch:sd-architecture found caching survives at 19% under the three properties. What
survives is the utility; what breaks is the guarantee. A semantic cache restores
some of the utility by serving a stored answer when a NEW query is similar enough
to an old one -- and "similar enough" is a threshold somebody has to pick.

Loosening the threshold raises the hit rate and raises the rate of serving an answer
that was right for a different question. This listing finds where the optimum sits,
and what moves it (eq:cache-threshold-is-an-error-cost-decision).
"""
# Query stream: a similarity distribution against the cache, and the probability
# that a stored answer is still CORRECT for the new query at that similarity.
# (similarity band, share of queries, P(stored answer is right for this query))
BANDS = [
    (0.98, 0.06, 0.99),   # near-duplicate
    (0.94, 0.09, 0.96),
    (0.90, 0.12, 0.88),
    (0.86, 0.14, 0.74),
    (0.82, 0.16, 0.55),
    (0.78, 0.17, 0.34),
    (0.74, 0.26, 0.12),   # merely topical
]
C_MISS = 1.00      # cost of generating a fresh answer
C_HIT = 0.02       # cost of serving from cache
A_FRESH = 0.91     # a freshly generated answer is right this often


def evaluate(threshold, error_cost):
    """Serve from cache when similarity >= threshold. Returns
    (hit rate, mean cost, accuracy, total cost including errors)."""
    hits = 0.0
    right = 0.0
    spend = 0.0
    for sim, share, p_right in BANDS:
        if sim >= threshold:
            hits += share
            right += share * p_right
            spend += share * C_HIT
        else:
            right += share * A_FRESH
            spend += share * C_MISS
    total = spend + (1 - right) * error_cost
    return hits, spend, right, total


print("A semantic cache over a query stream. Loosening the similarity threshold")
print("buys hit rate and sells accuracy.")
print()
print(f"{'similarity band':>17}{'share of queries':>18}"
      f"{'stored answer right':>21}")
print("-" * 56)
for sim, share, p in BANDS:
    print(f"{sim:>17.2f}{share:>18.0%}{p:>21.0%}")

print()
print()
print("Sweeping the threshold. A miss costs %.2f and is right %.0f%% of the time;"
      % (C_MISS, A_FRESH * 100))
print("a hit costs %.2f and is right as often as the band allows." % C_HIT)
print()
print(f"{'threshold':>11}{'hit rate':>11}{'spend':>9}{'accuracy':>11}"
      f"{'vs no cache':>14}")
print("-" * 56)
sweep = {}
for t in (1.01, 0.98, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74):
    hits, spend, acc, _ = evaluate(t, 0.0)
    sweep[t] = (hits, spend, acc)
    label = "no cache" if t > 1.0 else "%.2f" % t
    print(f"{label:>11}{hits:>11.0%}{spend:>9.3f}{acc:>11.1%}"
          f"{acc - A_FRESH:>+14.1%}")

print()
print()
print("Now price the errors. The optimum threshold is wherever total cost --")
print("generation plus the cost of a wrong answer -- is lowest. Three domains,")
print("differing ONLY in what a wrong answer costs.")
print()
DOMAINS = [
    ("suggesting a tag",       2.0),
    ("answering a support Q", 20.0),
    ("quoting a price",      200.0),
]
THRESHOLDS = [1.01, 0.98, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74]
print(f"{'domain':>22}{'error cost':>12}" + "".join(f"{t:>8.2f}" if t <= 1.0
                                                    else f"{'none':>8}"
                                                    for t in THRESHOLDS))
print("-" * 98)
best = {}
for name, ec in DOMAINS:
    totals = {}
    for t in THRESHOLDS:
        totals[t] = evaluate(t, ec)[3]
    b = min(totals, key=lambda k: totals[k])
    best[name] = (b, totals[b], sweep[b][0])
    cells = "".join(f"{totals[t]:>8.2f}" for t in THRESHOLDS)
    print(f"{name:>22}{ec:>12.0f}{cells}")

print()
print(f"{'domain':>22}{'best threshold':>16}{'hit rate there':>16}"
      f"{'total cost':>12}")
print("-" * 66)
for name, ec in DOMAINS:
    b, tot, hr = best[name]
    label = "no cache" if b > 1.0 else "%.2f" % b
    print(f"{name:>22}{label:>16}{hr:>16.0%}{tot:>12.2f}")

print()
print()
print("What the hit-rate-maximising choice costs in each domain -- the mistake of")
print("tuning a cache on its own dashboard.")
print()
LOOSE = 0.74
print(f"{'domain':>22}{'best total':>12}{'loose total':>13}{'penalty':>10}"
      f"{'as multiple':>13}")
print("-" * 70)
pen = {}
for name, ec in DOMAINS:
    b, tot, hr = best[name]
    loose = evaluate(LOOSE, ec)[3]
    pen[name] = (loose - tot, loose / tot)
    print(f"{name:>22}{tot:>12.2f}{loose:>13.2f}{loose - tot:>10.2f}"
          f"{loose / tot:>13.2f}x")

print(f"""
The sweep table is the trade with no error pricing at all, and it looks like an
easy win: at threshold {0.74:.2f} the cache serves {sweep[0.74][0]:.0%} of queries
and spend falls from {sweep[1.01][1]:.3f} to {sweep[0.74][1]:.3f} -- a
{(1 - sweep[0.74][1] / sweep[1.01][1]):.0%} reduction. Any cost dashboard would
call that a success.

The accuracy column is what the cost dashboard does not show. At that same
threshold accuracy is {sweep[0.74][2]:.1%}, against {A_FRESH:.1%} with no cache --
a loss of {(A_FRESH - sweep[0.74][2]) * 100:.1f} points. This is
ch:sd-architecture's semantic failure arriving through the cache: every one of
those is a 200 response containing an answer to a question the user did not ask.

There is a second thing in that column worth pausing on, because it runs the other
way. At threshold {0.94:.2f} accuracy is {sweep[0.94][2]:.1%} -- **above** the
{A_FRESH:.1%} of generating every answer fresh. A near-duplicate's stored answer is
right {0.96:.0%} to {0.99:.0%} of the time, which is better than a fresh sample.

So a tight semantic cache is not a quality compromise made for cost reasons. It is
a quality **improvement** that also happens to cost less, and the mechanism is
ch:sd-architecture's nondeterminism turned around: when the same question is asked
twice, re-sampling is a fresh chance to be wrong, and serving the vetted stored
answer avoids it. The cache is acting as a stability layer.

That only holds while the threshold is tight. It is the interior of the curve that
matters, and both ends are worse than the middle.

Pricing the errors is what makes the decision well-posed, and the result is the
finding. **The optimal threshold is not a property of the cache. It is a property of
what a wrong answer costs** (eq:cache-threshold-is-an-error-cost-decision).

Across three domains that differ in nothing but error cost, the optimum moves from
`{best['suggesting a tag'][0] if best['suggesting a tag'][0] <= 1 else 'no cache'}`
to `{best['quoting a price'][0] if best['quoting a price'][0] <= 1 else 'no cache'}`:

  - suggesting a tag (error costs {2.0:.0f}): best threshold
    {best['suggesting a tag'][0]:.2f}, hit rate {best['suggesting a tag'][2]:.0%}
  - answering a support question (error costs {20.0:.0f}): best threshold
    {best['answering a support Q'][0]:.2f}, hit rate
    {best['answering a support Q'][2]:.0%}
  - quoting a price (error costs {200.0:.0f}): best threshold
    {best['quoting a price'][0]:.2f}, hit rate {best['quoting a price'][2]:.0%}

The same cache, the same query stream, the same similarity model -- and hit rates
{best['suggesting a tag'][2]:.0%}, {best['answering a support Q'][2]:.0%} and
{best['quoting a price'][2]:.0%}. A team that ships one threshold across a product
with all three surfaces has it wrong on at least two of them.

The last table prices that mistake. Tuning for hit rate -- taking the loosest
threshold, which any cache dashboard rewards -- costs
{pen['suggesting a tag'][1]:.2f} times optimal in the cheap-error domain,
{pen['answering a support Q'][1]:.2f} times in the middle one, and
{pen['quoting a price'][1]:.2f} times when a wrong answer costs {200.0:.0f}.

**The penalty for over-caching grows with the thing the cache dashboard cannot
see.** That is the same structural failure as ch:sd-architecture's availability
graph, one layer down: an instrument that is accurate about its own quantity and
silent about the one that decides whether the choice was right.

The practical rule this gives is short. Set the threshold per surface, from the
error cost of that surface, and hold the cache team accountable for total cost
rather than hit rate. A cache reporting {sweep[0.74][0]:.0%} hits is not reporting
good news until someone has priced the {(A_FRESH - sweep[0.74][2]) * 100:.1f} points
it gave up to get there.""")
