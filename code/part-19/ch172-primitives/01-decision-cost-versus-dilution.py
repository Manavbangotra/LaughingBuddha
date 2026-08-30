# -*- coding: utf-8 -*-
# Extracted from: Chapter 172 — Tools, Resources, and Prompts
# Source: src/.../ch172-primitives.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why there are three primitives instead of one.

cite:mcp2026spec defines Tools, Resources and Prompts, and the split is not by
data type. It is by WHO DECIDES to use the thing:

  Tools      the MODEL decides, by emitting a call
  Resources  the HOST decides, by including content
  Prompts    the USER decides, by invoking a workflow

The common server-design error is to expose everything as a tool, on the grounds
that a tool can do anything. It can, and every tool call is a DECISION -- and
ch:ag-tool-calling measured what decisions cost when the inventory is not
perfectly distinct.

A resource removes the decision by putting the content in context directly. That
is not free either: ch:ag-memory found that everything in context dilutes
everything else. So there is a real trade, and this listing locates it
(eq:decision-cost-versus-dilution).
"""
import numpy as np

rng = np.random.default_rng(4127)

M = 50000
N_ITEMS = 40            # distinct pieces of context a task might need
NEEDED = 5              # how many a given task actually needs
P_SELECT = 0.90         # chance the model picks the right item to fetch
P_NOTICE = 0.55         # chance a wrong fetch is RECOGNISED as wrong
BASE = 0.995            # per-step success with everything present and undiluted


def dilution(loaded):
    """ch:ag-memory's effect: recall of any one item degrades as context grows.
    Modelled as a gentle decay in the per-step success rate."""
    return BASE * (1.0 - 0.0009 * max(loaded - NEEDED, 0))


def run(preload_frac, m=M, n_items=N_ITEMS, needed=NEEDED,
        p_select=P_SELECT, hit_rate=None):
    """`preload_frac` of the inventory is included as resources; the rest must be
    fetched by tool call. A task needs `needed` items. Items that were preloaded
    are simply present; items that were not cost a call that may pick wrong."""
    n_pre = int(round(preload_frac * n_items))
    # Which items a task needs, and whether each happened to be preloaded.
    # Preloading is prioritised by how often an item is needed, which is what a
    # host would actually do.
    if hit_rate is None:
        # Uniform demand: a preloaded item is useful with probability n_pre/n.
        pre_hits = rng.binomial(needed, n_pre / n_items, m)
    else:
        # Skewed demand: the most-needed items are preloaded first, so the
        # preloaded set covers more than its share.
        cover = min(1.0, (n_pre / n_items) ** (1.0 / max(hit_rate, 1e-6)))
        pre_hits = rng.binomial(needed, cover, m)
    to_fetch = needed - pre_hits
    # Each fetch is a decision that can go wrong. The damaging case is not a
    # fetch that fails -- it is one that returns the WRONG item plausibly, which
    # ch:as-specialized's weak-verifier result says often goes unnoticed.
    got = np.zeros(m, dtype=np.int64)
    calls = np.zeros(m, dtype=np.int64)
    poisoned = np.zeros(m, dtype=bool)
    for k in range(needed):
        active = np.flatnonzero(to_fetch > k)
        if not len(active):
            continue
        calls[active] += 1
        ok = rng.random(len(active)) < p_select
        got[active[ok]] += 1
        wrong = active[~ok]
        if len(wrong):
            noticed = rng.random(len(wrong)) < P_NOTICE
            # Noticed: retry, and the error message narrows the choice.
            r = wrong[noticed]
            calls[r] += 1
            again = rng.random(len(r)) < 0.93
            got[r[again]] += 1
            # Unnoticed: the task proceeds on wrong context.
            poisoned[wrong[~noticed]] = True
    have = pre_hits + got
    # The task succeeds if it has everything it needed and the diluted context
    # still supports the reasoning.
    p = dilution(n_pre)
    reason_ok = rng.random(m) < p ** needed
    ok = (have >= needed) & reason_ok & ~poisoned
    return float(ok.mean()), float(calls.mean()), n_pre


print(f"{M:,} tasks. {N_ITEMS} pieces of context exist; each task needs")
print(f"{NEEDED}. A tool fetch is a decision correct {P_SELECT:.0%} of the time;")
print(f"a wrong fetch is noticed {P_NOTICE:.0%} of the time and retried, and")
print("otherwise poisons the task. A preloaded resource needs no decision.")
print()
print(f"{'preloaded':>11}{'items':>8}{'success':>10}{'tool calls':>12}")
print("-" * 41)
tab = {}
for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    r = run(f)
    tab[f] = r
    print(f"{f:>11.0%}{r[2]:>8}{r[0]:>10.1%}{r[1]:>12.2f}")

best = max(tab, key=lambda k: tab[k][0])

print()
print()
print("The two costs separated. 'Decision cost' is what the fetches lose;")
print("'dilution cost' is what the loaded context loses.")
print()
print(f"{'preloaded':>11}{'decision cost':>15}{'dilution cost':>15}"
      f"{'total loss':>12}")
print("-" * 53)
sep = {}
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    n_pre = int(round(f * N_ITEMS))
    # Decision-only: no dilution.
    d_only = run(f)
    p = dilution(n_pre)
    dil = 1.0 - p ** NEEDED
    dec = 1.0 - d_only[0] / max(p ** NEEDED, 1e-9)
    sep[f] = (dec, dil)
    print(f"{f:>11.0%}{dec:>15.1%}{dil:>15.1%}{1 - d_only[0]:>12.1%}")

print()
print()
print("The optimum moves with how well the host can PREDICT what is needed.")
print("A skew of 1 means demand is uniform; higher means a few items dominate.")
print()
print(f"{'demand skew':>13}" + "".join(f"{'pre ' + format(f, '.0%'):>11}"
                                       for f in (0.0, 0.25, 0.5, 1.0))
      + f"{'best':>9}")
print("-" * 66)
sk = {}
for s in (1.0, 1.6, 2.5, 4.0):
    row = [run(f, hit_rate=s)[0] for f in (0.0, 0.25, 0.5, 1.0)]
    names = ["0%", "25%", "50%", "100%"]
    sk[s] = (row, names[int(np.argmax(row))])
    print(f"{s:>13.1f}" + "".join(f"{v:>11.1%}" for v in row)
          + f"{sk[s][1]:>9}")

print()
print()
print("And with selection reliability, which is ch:ag-tool-calling's variable.")
print("A server whose tools are hard to tell apart should preload more.")
print()
print(f"{'selection':>11}" + "".join(f"{'pre ' + format(f, '.0%'):>11}"
                                     for f in (0.0, 0.25, 0.5, 1.0))
      + f"{'best':>9}")
print("-" * 64)
sl = {}
for ps in (0.98, 0.90, 0.75, 0.55):
    row = [run(f, p_select=ps)[0] for f in (0.0, 0.25, 0.5, 1.0)]
    names = ["0%", "25%", "50%", "100%"]
    sl[ps] = (row, names[int(np.argmax(row))])
    print(f"{ps:>11.0%}" + "".join(f"{v:>11.1%}" for v in row)
          + f"{sl[ps][1]:>9}")

print(f"""
The first table has an interior optimum, which is the whole reason there are
three primitives instead of one. Preloading nothing gives {tab[0.0][0]:.1%};
preloading everything gives {tab[1.0][0]:.1%}; the best row is
{best:.0%} at {tab[best][0]:.1%}.

The second table separates the two costs and shows them crossing. Decision cost
falls from {sep[0.0][0]:.1%} to {sep[1.0][0]:.1%} as fetches are eliminated;
dilution cost rises from {sep[0.0][1]:.1%} to {sep[1.0][1]:.1%} as context grows.
**Neither primitive is free, and they are expensive in opposite directions**
(eq:decision-cost-versus-dilution).

Note what the decision cost actually consists of. It is not fetches that fail --
those get retried. It is fetches that return the wrong thing PLAUSIBLY and are not
noticed, which happens {1 - P_NOTICE:.0%} of the time a selection goes wrong. That
is ch:as-specialized's weak-verifier result appearing inside a tool call: **the
damaging failure is the one the model cannot see.**

The third table contains the result that reverses the intuition. As demand
concentrates -- a few items needed by most tasks -- the best preload fraction goes
DOWN, from {sk[1.0][1]} at uniform demand to {sk[4.0][1]} at a skew of {4.0}.

The reason is that skew makes preloading EFFICIENT rather than making it more
necessary. When a quarter of the inventory covers most of the demand, preloading
that quarter captures nearly all of the benefit, and preloading more only adds
dilution. **The better you can predict what is needed, the less you need to
preload** -- because prediction is what lets a small resource set do the work of a
large one.

Which means a host that cannot predict demand at all faces the worst version of
this trade, and its best move is not a middle setting but one of the extremes.

The fourth table is the actionable one, and it is ch:ag-tool-calling's variable.
At {0.98:.0%} selection reliability the best policy is to preload {sl[0.98][1]} --
let the model choose, it is good at it. At {0.55:.0%} the best policy is
{sl[0.55][1]}, and the gap between the extremes at that reliability is
{sl[0.55][0][3] - sl[0.55][0][0]:+.1%}.

**A resource is what you build when the model cannot reliably choose the tool.**
That is the practical content of the primitive distinction: tools put the decision
in the model, resources take it away, and which is right depends on a number
ch:ag-tool-calling told you how to measure.

So the rule is not "prefer resources" or "prefer tools". It is: measure your
selection reliability, measure your demand skew, and let those two numbers pick
the split.""")
