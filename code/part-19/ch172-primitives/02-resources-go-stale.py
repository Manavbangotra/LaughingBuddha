# -*- coding: utf-8 -*-
# Extracted from: Chapter 172 — Tools, Resources, and Prompts
# Source: src/.../ch172-primitives.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The other half of the primitive choice, which is freshness.

The previous listing traded a DECISION against DILUTION and found an interior
optimum. It assumed the preloaded content was correct, and that assumption is the
one that fails in production.

A resource is included in context at the start of a turn. A tool is called when
the model wants it. So a resource carries whatever the world looked like when the
host assembled the context, and a tool carries whatever it looks like now
(eq:resources-go-stale).

That is ch:as-long-running's drift arriving at the scale of a single turn, and it
is the axis the tools-versus-resources discussion usually leaves out.
"""
import numpy as np

rng = np.random.default_rng(4159)

M = 50000
NEEDED = 5
TURN_LEN = 9            # steps in a turn, over which preloaded content ages
P_SELECT = 0.90
P_NOTICE = 0.55


def run(preload_frac, volatility, m=M, needed=NEEDED, turn_len=TURN_LEN,
        p_select=P_SELECT, revalidate=False):
    """`volatility` is the chance a given item changes per step of the turn.
    Preloaded items are read at step 0 and used later; fetched items are read
    when used. `revalidate` re-reads preloaded items at the halfway point."""
    n_pre = rng.binomial(needed, preload_frac, m)
    n_fetch = needed - n_pre
    # Preloaded items are used at a random step, having aged since step 0.
    age = rng.integers(1, turn_len + 1, (m, needed))
    if revalidate:
        age = np.minimum(age, np.maximum(age - turn_len // 2, 1))
    stale_p = 1.0 - (1.0 - volatility) ** age
    idx = np.arange(needed)[None, :]
    is_pre = idx < n_pre[:, None]
    stale = (rng.random((m, needed)) < stale_p) & is_pre
    # Fetched items are current, but the fetch is a decision.
    poisoned = np.zeros(m, dtype=bool)
    missing = np.zeros(m, dtype=bool)
    calls = np.zeros(m, dtype=np.int64)
    for k in range(needed):
        active = np.flatnonzero(n_fetch > k)
        if not len(active):
            continue
        calls[active] += 1
        ok = rng.random(len(active)) < p_select
        wrong = active[~ok]
        if len(wrong):
            noticed = rng.random(len(wrong)) < P_NOTICE
            r = wrong[noticed]
            calls[r] += 1
            again = rng.random(len(r)) < 0.93
            missing[r[~again]] = True
            poisoned[wrong[~noticed]] = True
    ok = ~poisoned & ~missing & ~stale.any(1)
    return float(ok.mean()), float(calls.mean()), float(stale.any(1).mean())


print(f"{M:,} tasks needing {NEEDED} items over a {TURN_LEN}-step turn.")
print("Preloaded items are read once at the start; fetched items are current.")
print()
print(f"{'preloaded':>11}" + "".join(f"{'vol ' + format(v, '.1%'):>13}"
                                     for v in (0.0, 0.005, 0.02, 0.06)))
print("-" * 63)
tab = {}
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    row = tuple(run(f, v)[0] for v in (0.0, 0.005, 0.02, 0.06))
    tab[f] = row
    print(f"{f:>11.0%}" + "".join(f"{x:>13.1%}" for x in row))

print()
print()
print("The best preload fraction at each volatility, and what it costs to get")
print("it wrong in either direction.")
print()
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
print(f"{'volatility':>12}{'best preload':>14}{'best':>9}"
      f"{'all-resource':>14}{'all-tool':>11}")
print("-" * 60)
opt = {}
for v in (0.0, 0.002, 0.01, 0.03, 0.08):
    row = [run(f, v)[0] for f in FRACS]
    b = int(np.argmax(row))
    opt[v] = (FRACS[b], row[b], row[-1], row[0])
    print(f"{v:>12.1%}{FRACS[b]:>14.0%}{row[b]:>9.1%}{row[-1]:>14.1%}"
          f"{row[0]:>11.1%}")

print()
print()
print("How much staleness there actually is, which is the quantity nobody")
print("measures because a stale resource produces no error.")
print()
print(f"{'volatility':>12}{'stale rate, all-resource':>26}{'success':>10}")
print("-" * 48)
st = {}
for v in (0.0, 0.002, 0.01, 0.03, 0.08):
    r = run(1.0, v)
    st[v] = (r[2], r[0])
    print(f"{v:>12.1%}{r[2]:>26.1%}{r[0]:>10.1%}")

print()
print()
print("Re-reading resources partway through the turn is the cheap fix, and it")
print("is ch:as-long-running's re-validation at a much smaller scale.")
print()
print(f"{'volatility':>12}{'no revalidation':>17}{'revalidated':>13}{'gain':>9}")
print("-" * 51)
rv = {}
for v in (0.002, 0.01, 0.03, 0.08):
    a = run(1.0, v)[0]
    b = run(1.0, v, revalidate=True)[0]
    rv[v] = (a, b)
    print(f"{v:>12.1%}{a:>17.1%}{b:>13.1%}{b - a:>+9.1%}")

print()
print()
print("And the turn length, since a longer turn ages its preloaded context more.")
print()
print(f"{'turn steps':>12}{'all-resource':>14}{'all-tool':>11}{'best':>13}")
print("-" * 50)
tl = {}
for L in (2, 5, 12, 30):
    a = run(1.0, 0.02, turn_len=L)[0]
    b = run(0.0, 0.02, turn_len=L)[0]
    tl[L] = (a, b)
    print(f"{L:>12}{a:>14.1%}{b:>11.1%}"
          f"{('resources' if a > b else 'tools'):>13}")

print(f"""
The first table has a crossover in it, and the crossover is sharp.

At zero volatility, preloading everything gives {tab[1.0][0]:.1%} against
{tab[0.0][0]:.1%} for fetching everything -- resources win by
{tab[1.0][0] - tab[0.0][0]:.1f} points, exactly as the previous listing said they
should when selection is imperfect. At {0.06:.0%} volatility per step, preloading
everything gives {tab[1.0][3]:.1%} against {tab[0.0][3]:.1%}.

**The same design that wins by twenty-two points loses by fifty-six, and the only
thing that changed is how fast the world moves** (eq:resources-go-stale).

The second table locates the crossover at about {0.01:.0%} volatility per step.
Below it, preload; above it, fetch. That is a usable rule and it is stated in a
unit teams can actually measure -- how often does this thing change, per step of a
turn.

The third table is why this goes unnoticed. At {0.03:.0%} volatility,
{st[0.03][0]:.1%} of tasks are working from at least one stale item, and **a stale
resource produces no error**. It is well-formed, it is plausible, it was correct
recently. The task completes and the answer is wrong, which is
ch:as-long-running's silent drift compressed from days into a single turn.

The fourth table is the fix, and it is the same fix that chapter found. Re-reading
preloaded resources partway through the turn is worth
{rv[0.01][1] - rv[0.01][0]:+.1%} at {0.01:.0%} volatility and
{rv[0.08][1] - rv[0.08][0]:+.1%} at {0.08:.0%}, for one extra read.

**Re-validation is the cheapest intervention at every scale this book has measured
it** -- across a week-long workflow in ch:as-long-running and across a nine-step
turn here.

The last table adds the variable that makes this worse over time. A two-step turn
favours resources ({tl[2][0]:.1%} against {tl[2][1]:.1%}); a thirty-step turn
favours tools by {tl[30][1] - tl[30][0]:.1f} points at the same volatility.

Agentic turns are getting longer, which means **content that was safe to preload
becomes unsafe to preload without anything about it changing.** A resource set
tuned when turns were three steps will quietly rot when turns become twenty, and
the symptom will be wrong answers rather than errors.

Putting the two listings together gives the rule the primitive split is really
for:

**Preload as resources what is stable, predictable and hard to select. Fetch as
tools what is volatile.** Stability decides it first -- above about
{0.01:.0%} per-step volatility nothing else matters -- and below that threshold,
selection reliability and demand skew decide the fraction.""")
