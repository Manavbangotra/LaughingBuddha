# -*- coding: utf-8 -*-
# Extracted from: Chapter 192 — Retrieval and Agent Architecture at Scale
# Source: src/.../ch192-retrieval-agents.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fan-out converts a good tail into a bad one, and the conversion is arithmetic.

An agent that calls five tools in parallel, or a retrieval layer that queries eight
shards, waits for the SLOWEST of them. That is not the mean latency of a dependency;
it is the maximum of a sample, and the maximum of a sample behaves very differently.

This listing measures how fan-out width turns a per-dependency tail into a
request-level one (eq:fanout-amplifies-the-tail), and what that does to the
p99 a user actually experiences.

The result is why a system composed entirely of fast, reliable components can be
slow and unreliable, with nothing in any component's dashboard looking wrong.
"""
# A dependency whose latency is mostly fast with a thin slow tail.
# (latency in seconds, probability)
DEP = [
    (0.05, 0.60),
    (0.09, 0.25),
    (0.18, 0.10),
    (0.45, 0.037),
    (1.60, 0.013),   # the tail: 1.3% of calls
]
WIDTHS = [1, 2, 3, 5, 8, 12, 20]


def percentile_of_max(width, q):
    """The q-th percentile of the MAXIMUM of `width` independent draws.

    P(max <= t) = P(single <= t)^width, so we walk the support and find the
    smallest t whose cumulative probability raised to `width` reaches q.
    """
    cum = 0.0
    for t, p in DEP:
        cum += p
        if cum ** width >= q:
            return t
    return DEP[-1][0]


def mean_of_max(width):
    """Expected value of the maximum of `width` independent draws."""
    total = 0.0
    prev = 0.0
    cum = 0.0
    for t, p in DEP:
        cum += p
        # P(max == t) = P(all <= t) - P(all <= previous t)
        total += t * (cum ** width - prev)
        prev = cum ** width
    return total


def p_slow(width, threshold):
    """P(at least one of `width` draws exceeds `threshold`)."""
    fast = sum(p for t, p in DEP if t <= threshold)
    return 1.0 - fast ** width


print("One dependency: fast most of the time, with a 1.3% tail at 1.60s.")
print()
print(f"{'latency':>10}{'probability':>14}{'cumulative':>13}")
print("-" * 37)
c = 0.0
for t, p in DEP:
    c += p
    print(f"{t:>9.2f}s{p:>14.1%}{c:>13.1%}")

single_mean = sum(t * p for t, p in DEP)
print()
print(f"mean {single_mean:.3f}s, p50 {percentile_of_max(1, 0.50):.2f}s, "
      f"p95 {percentile_of_max(1, 0.95):.2f}s, "
      f"p99 {percentile_of_max(1, 0.99):.2f}s")

print()
print()
print("Now fan out: call N of them in parallel and wait for all to return.")
print("The request's latency is the MAXIMUM, not the mean.")
print()
print(f"{'fan-out':>9}{'mean':>10}{'p50':>9}{'p95':>9}{'p99':>9}"
      f"{'P(hits tail)':>15}")
print("-" * 61)
tab = {}
for w in WIDTHS:
    m = mean_of_max(w)
    p50 = percentile_of_max(w, 0.50)
    p95 = percentile_of_max(w, 0.95)
    p99 = percentile_of_max(w, 0.99)
    pt = p_slow(w, 0.45)
    tab[w] = (m, p50, p95, p99, pt)
    print(f"{w:>9}{m:>9.3f}s{p50:>8.2f}s{p95:>8.2f}s{p99:>8.2f}s{pt:>15.1%}")

print()
print()
print("The same thing stated as amplification against a single call.")
print()
print(f"{'fan-out':>9}{'mean grows':>13}{'p95 grows':>12}"
      f"{'tail probability':>19}")
print("-" * 53)
for w in WIDTHS:
    m, p50, p95, p99, pt = tab[w]
    print(f"{w:>9}{m / single_mean:>12.2f}x{p95 / percentile_of_max(1, 0.95):>11.2f}x"
          f"{pt:>19.1%}")

print()
print()
print("What it takes to keep a 0.50s budget as fan-out grows: the per-dependency")
print("tail probability you would have to achieve.")
print()
BUDGET_P = 0.95      # we want 95% of requests inside budget
print(f"{'fan-out':>9}{'needed per-dep reliability':>29}"
      f"{'current':>11}{'gap':>10}")
print("-" * 60)
CURRENT_FAST = sum(p for t, p in DEP if t <= 0.45)
need = {}
for w in WIDTHS:
    # need fast**w >= BUDGET_P  ->  fast >= BUDGET_P**(1/w)
    r = BUDGET_P ** (1.0 / w)
    need[w] = r
    print(f"{w:>9}{r:>29.4%}{CURRENT_FAST:>11.2%}"
          f"{(r - CURRENT_FAST):>+10.2%}")

print()
print()
print("And the fix that actually works: hedging. Issue a duplicate request for any")
print("dependency still outstanding at the hedge point, and take the first answer.")
print()
HEDGE_AT = 0.18
print(f"{'fan-out':>9}{'p95 plain':>12}{'p95 hedged':>13}{'extra calls':>14}"
      f"{'improvement':>14}")
print("-" * 63)


def hedged_percentile(width, q, hedge_at):
    """With a hedge, a slow draw is replaced by hedge_at plus a fresh draw, so
    the effective per-call distribution is truncated: anything slower than the
    hedge point becomes hedge_at + (a fresh, usually fast, draw)."""
    eff = []
    slow_mass = 0.0
    for t, p in DEP:
        if t <= hedge_at:
            eff.append((t, p))
        else:
            slow_mass += p
    # The hedged retry lands at hedge_at + a draw from the same distribution.
    for t, p in DEP:
        eff.append((hedge_at + t, slow_mass * p))
    eff.sort()
    cum = 0.0
    for t, p in eff:
        cum += p
        if cum ** width >= q:
            return t
    return eff[-1][0]


for w in WIDTHS:
    plain = tab[w][2]
    hed = hedged_percentile(w, 0.95, HEDGE_AT)
    extra = w * sum(p for t, p in DEP if t > HEDGE_AT)
    print(f"{w:>9}{plain:>11.2f}s{hed:>12.2f}s{extra:>14.2f}"
          f"{(1 - hed / plain):>13.0%}")

print(f"""
A single call to this dependency has a mean of {single_mean:.3f}s and a p95 of
{percentile_of_max(1, 0.95):.2f}s. By any normal standard it is a healthy
dependency: {DEP[0][1]:.0%} of calls return in {DEP[0][0]:.2f}s and only
{DEP[-1][1]:.1%} hit the slow path.

Fan out to {WIDTHS[4]} parallel calls and the picture changes completely. The mean
becomes {tab[8][0]:.3f}s, p95 becomes {tab[8][2]:.2f}s, and the probability that at
least one call lands in the tail rises to {tab[8][4]:.1%}
(eq:fanout-amplifies-the-tail).

At fan-out {WIDTHS[-1]} the tail is hit on {tab[20][4]:.1%} of requests. **The rare
event has become the common case**, and nothing about the dependency changed. Its
own dashboard still shows a {DEP[-1][1]:.1%} slow rate, correctly, and that number
is now describing something that happens to {tab[20][4]:.0%} of user requests.

The mechanism is that P(all fast) is a PRODUCT. Each additional parallel call
multiplies in another chance to be unlucky, so per-request reliability decays
geometrically in the fan-out width -- which is ch:ag-loop's chain
(eq:loop-is-not-a-chain) with the steps running side by side instead of end to
end. Sequential or parallel, the arithmetic is the same; only the latency changes.

The reliability table is the part worth internalising, because it inverts the usual
engineering conversation. To keep {BUDGET_P:.0%} of requests inside a
{0.50:.2f}s budget at fan-out {WIDTHS[-1]}, each dependency needs to stay fast
{need[20]:.2%} of the time. It currently manages {CURRENT_FAST:.2%}.

**That gap is not closeable by optimising the dependency.** Going from
{CURRENT_FAST:.2%} to {need[20]:.2%} means cutting the tail from
{1 - CURRENT_FAST:.2%} to {1 - need[20]:.2%} -- removing four fifths of it -- and
the tail is usually where the real work is -- a cold cache, a large document, a slow
shard. So the answer to "our agent is slow" is rarely "make the tools faster".

Hedging is the lever that does work, and the table shows why. Issuing a duplicate
call for anything still outstanding at {HEDGE_AT:.2f}s cuts p95 at fan-out
{WIDTHS[4]} from {tab[8][2]:.2f}s to {hedged_percentile(8, 0.95, HEDGE_AT):.2f}s --
a {(1 - hedged_percentile(8, 0.95, HEDGE_AT) / tab[8][2]):.0%} improvement -- for
{8 * sum(p for t, p in DEP if t > HEDGE_AT):.2f} extra calls per request, which is
{sum(p for t, p in DEP if t > HEDGE_AT):.0%} more load.

That trade is the chapter's practical result: **a few percent more load buys back
most of the tail amplification**, because a hedge attacks the maximum directly
rather than trying to improve the distribution it is drawn from.

The caveat is that hedging only works when the duplicate is genuinely independent --
a second call to the same overloaded shard hedges nothing. And under ch:sd-async's
cost model, a hedge on an expensive model call is not a few percent of load, it is a
few percent of a large bill, which is why hedging is a retrieval technique far more
often than a generation one.""")
