# -*- coding: utf-8 -*-
# Extracted from: Chapter 201 — Distributed and Disaggregated Inference
# Source: src/.../ch201-distributed.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Cache-aware routing and load balancing want opposite things.

Across a fleet, a request whose prefix is already resident on some node can skip most of
its prefill. So the router should send it there.

Two things stop that being free. Sending requests where their prefix lives concentrates
load, and ch:sd-async established what uneven load does to queueing. And a node's cache
is finite: routing uniformly means every node tries to hold every popular prefix, which
does not fit (eq:affinity-fights-balance).

This listing sweeps between pure load balancing and pure prefix pinning and finds the
optimum is at neither end -- and that it moves with fleet load and prefix length.
"""
NODES = 16
CACHE_TOKENS = 9000.0      # prefix tokens one node can keep resident

# Prefix popularity: a few shared system prompts and documents dominate.
# (label, share of requests, prefix tokens)
PREFIXES = [
    ("shared system prompt", 0.44, 1800),
    ("common document A",    0.14, 6200),
    ("common document B",    0.09, 5400),
    ("team template",        0.07, 2900),
    ("long tail",            0.26, 700),
]
PREFILL_MS_PER_TOK = 0.11
DECODE_MS = 4.18
CATALOGUE = sum(t for _, _, t in PREFIXES)


def replication(affinity):
    """How many nodes hold a given prefix. Uniform routing needs it everywhere;
    full affinity needs it in one place."""
    return NODES - affinity * (NODES - 1)


def resident(affinity):
    """Which prefixes fit on a node.

    At replication r each prefix sits on r of NODES nodes, so one node must hold
    a share r/NODES of the catalogue. Low affinity means high replication means
    every node tries to hold everything -- which does not fit. Popular prefixes
    are kept first.
    """
    budget = CACHE_TOKENS * NODES / replication(affinity)
    keep, used = set(), 0.0
    for label, share, toks in sorted(PREFIXES, key=lambda p: -p[1]):
        if used + toks <= budget:
            keep.add(label)
            used += toks
    return keep


def hit_rate_for(affinity, label, share):
    """P(this request lands on a node that has its prefix resident)."""
    if label not in resident(affinity):
        return 0.0
    r = replication(affinity)
    # Affinity routing targets a holder; the residual routes uniformly and hits
    # with probability r/NODES.
    return affinity + (1.0 - affinity) * (r / NODES)


def load_skew(affinity):
    """Busiest node's load over the mean. Pinning a prefix that carries 44% of
    traffic to a subset of nodes concentrates load on them."""
    return 1.0 + affinity * 2.5


def wait_ms(skew, rho_mean=0.28):
    rho = min(0.985, rho_mean * skew)
    return DECODE_MS * rho / (1.0 - rho)


print("A %d-node fleet, %.0f prefix tokens resident per node." % (NODES,
                                                                  CACHE_TOKENS))
print("The prefix catalogue totals %d tokens." % CATALOGUE)
print()
print(f"{'prefix':>24}{'share':>9}{'tokens':>9}{'prefill ms':>13}")
print("-" * 56)
for label, share, toks in PREFIXES:
    print(f"{label:>24}{share:>9.0%}{toks:>9}{toks * PREFILL_MS_PER_TOK:>13.0f}")
mean_prefill = sum(s * t for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
print()
print(f"mean prefill if nothing is cached: {mean_prefill:.0f} ms")

print()
print()
print("Sweeping affinity from pure load balancing to pure prefix pinning.")
print()
AFF = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
print(f"{'affinity':>10}{'replication':>13}{'hit rate':>11}"
      f"{'prefill saved':>15}{'load skew':>12}{'queue wait':>13}")
print("-" * 74)
sweep = {}
for a in AFF:
    hr = sum(s * hit_rate_for(a, l, s) for l, s, _ in PREFIXES)
    saved = sum(s * hit_rate_for(a, l, s) * t
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    sk = load_skew(a)
    w = wait_ms(sk)
    sweep[a] = (hr, saved, sk, w)
    print(f"{a:>10.1f}{replication(a):>13.1f}{hr:>11.1%}{saved:>14.0f}m"
          f"{sk:>12.2f}{w:>12.1f}m")

print()
print()
print("Total time to first token: prefill not saved, plus queue wait.")
print()
print(f"{'affinity':>10}{'prefill ms':>13}{'queue ms':>11}{'TTFT ms':>11}"
      f"{'vs balanced':>14}")
print("-" * 62)
total = {}
for a in AFF:
    hr, saved, sk, w = sweep[a]
    pf = mean_prefill - saved
    total[a] = pf + w
    print(f"{a:>10.1f}{pf:>13.0f}{w:>11.1f}{pf + w:>11.1f}"
          f"{(pf + w) / total[0.0]:>13.2f}x")

best = min(total, key=lambda k: total[k])
print()
print(f"best affinity: {best:.1f} at {total[best]:.1f} ms TTFT")

print()
print()
print("How the optimum moves with fleet load. A busier fleet cannot afford skew.")
print()


def total_at(a, rho_mean):
    saved = sum(s * hit_rate_for(a, l, s) * t
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    return (mean_prefill - saved) + wait_ms(load_skew(a), rho_mean)


print(f"{'mean utilisation':>18}" + "".join(f"{('a=%.1f' % a):>9}" for a in AFF)
      + f"{'best':>8}")
print("-" * 80)
bestrow = {}
for rho in (0.12, 0.20, 0.28, 0.36, 0.44):
    vals = {a: total_at(a, rho) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    bestrow[rho] = b
    print(f"{rho:>18.0%}" + "".join(f"{vals[a]:>9.0f}" for a in AFF)
          + f"{b:>8.1f}")

print()
print()
print("And with prefix length, which decides how much a hit is worth.")
print()


def total_scaled(a, scale):
    mp = sum(s * t * scale for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    saved = sum(s * hit_rate_for(a, l, s) * t * scale
                for l, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    return mp - saved + wait_ms(load_skew(a))


print(f"{'prefix scale':>14}{'mean prefill':>14}"
      + "".join(f"{('a=%.1f' % a):>9}" for a in AFF) + f"{'best':>8}")
print("-" * 90)
scalebest = {}
for scale in (0.2, 0.5, 1.0, 2.0, 5.0):
    mp = sum(s * t * scale for _, s, t in PREFIXES) * PREFILL_MS_PER_TOK
    vals = {a: total_scaled(a, scale) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    scalebest[scale] = b
    print(f"{scale:>14.1f}{mp:>13.0f}m"
          + "".join(f"{vals[a]:>9.0f}" for a in AFF) + f"{b:>8.1f}")

print()
print()
print("What the cache capacity does: it is why uniform routing cannot simply")
print("cache everything everywhere.")
print()
print(f"{'cache tokens/node':>19}{'catalogue fits at a=0':>24}"
      f"{'best affinity':>16}")
print("-" * 60)
for cap in (2000.0, 4500.0, 9000.0, 17000.0, 40000.0):
    CACHE_TOKENS = cap
    fits = CATALOGUE <= cap
    vals = {a: total_at(a, 0.28) for a in AFF}
    b = min(vals, key=lambda k: vals[k])
    print(f"{cap:>19.0f}{('yes' if fits else 'no'):>24}{b:>16.1f}")
CACHE_TOKENS = 9000.0

print(f"""
The sweep is the conflict in one table. Raising affinity from {0.0:.1f} to {1.0:.1f}
takes replication from {replication(0.0):.0f} nodes per prefix to
{replication(1.0):.0f}, the hit rate from {sweep[0.0][0]:.1%} to {sweep[1.0][0]:.1%},
and prefill saved from {sweep[0.0][1]:.0f}ms to {sweep[1.0][1]:.0f}ms.

It also takes load skew from {sweep[0.0][2]:.2f} to {sweep[1.0][2]:.2f}, and queue wait
from {sweep[0.0][3]:.1f}ms to {sweep[1.0][3]:.1f}ms
(eq:affinity-fights-balance). **Both columns are real and they point in opposite
directions.**

Prefix affinity is not load balancing with a cache benefit attached. It is a trade of
queueing behaviour for prefill, and ch:sd-async established precisely how expensive
queueing gets when load stops being even.

The total column resolves it at **{best:.1f}**, giving {total[best]:.1f}ms against
{total[0.0]:.1f}ms for pure balancing and {total[1.0]:.1f}ms for pure pinning.
**Neither extreme is right.**

The curve is not monotone, and the dip at {0.2:.1f} is worth explaining rather than
smoothing away. At that affinity replication is {replication(0.2):.0f} nodes, which is
low enough that each node must hold {CACHE_TOKENS * NODES / replication(0.2):.0f}
tokens of catalogue and high enough that it still misses on
{1 - sweep[0.2][0]:.0%} of requests. **Partial replication is the worst of both**: it
pays most of the capacity cost of replicating and gets little of the targeting benefit
of pinning. A router configured half-way between the two strategies without
understanding which regime it is in can land here, and the symptom is a hit rate lower
than either pure strategy would give.

The utilisation table shows the optimum is not a constant. At {0.12:.0%} mean
utilisation the best affinity is {bestrow[0.12]:.1f}; at {0.44:.0%} it is
{bestrow[0.44]:.1f}.

The reason is that the queueing term is convex: at low utilisation a hot node absorbs
the extra work and skew costs almost nothing, while near saturation the same skew costs
a great deal. **The correct router behaviour is therefore load-dependent** -- lean on
affinity when there is headroom, fall back to balancing when there is not -- and a
router with a fixed affinity weight is wrong at one end of its own day.

The prefix-length table is the other axis, and it gives a rule that needs no arithmetic:
at {0.2:.1f} times these prefix lengths the best affinity is {scalebest[0.2]:.1f}, and
at {5.0:.1f} times it is {scalebest[5.0]:.1f}. **The longer the shared prefix, the more
skew is worth tolerating.**

A deployment where every request carries a two-thousand-token system prompt should route
aggressively by affinity. One with short, unique prompts should not build the
infrastructure at all.

The capacity table explains why uniform routing cannot simply sidestep this by caching
everything on every node. The catalogue is {CATALOGUE} tokens; a node holding
{CACHE_TOKENS:.0f} cannot keep all of it, so at low affinity the least popular prefixes
are evicted everywhere and their hits are lost. **Cache capacity is what makes
replication expensive**, and it is the term that converts this from a load-balancing
question into a placement one.

Two implementation notes. The hit rate depends on the prefix being *resident*, not on
its having been seen -- so an eviction on the target node silently converts an affinity
hit into a miss, and the router cannot tell without asking. And affinity routing makes
the fleet stateful: a node's value depends on what it holds, so draining one for
maintenance costs more than its capacity share, which ch:inf-kubernetes has to plan
around.""")
