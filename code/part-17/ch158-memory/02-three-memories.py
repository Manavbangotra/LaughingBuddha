# -*- coding: utf-8 -*-
# Extracted from: Chapter 158 — Agent Memory: Short-Term, Working, and Long-Term
# Source: src/.../ch158-memory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three things are called memory, and they are not substitutes.

The word covers at least three mechanisms that solve different problems:

  the CONTEXT -- the raw history of this run, carried forward verbatim
  the SCRATCHPAD -- derived facts the agent wrote down deliberately
  the STORE -- what survives between runs and is retrieved

Teams reason about "adding memory" as one decision, which produces the two
predictable mistakes: extending the context to fix something a scratchpad would
fix, and adding a retrieval store to fix something the context would fix.

This listing gives an agent three distinct needs and each mechanism in turn, and
measures which mechanism addresses which need (eq:three-memories).
"""
import numpy as np

rng = np.random.default_rng(2213)

N = 60000
K = 14                  # steps in a run
CTX = 6                 # steps of raw history the model attends to reliably
DILUTE = 0.020          # per-entry loss of recall as context fills
P_COMPOSE = 0.88        # combining facts inside one pass
P_LOOKUP = 0.985        # reading one recorded fact


def run(need, context=True, scratchpad=False, store=False, ctx=CTX, k=K):
    """`need` selects which kind of dependency each step has:
      'recent'  -- a fact produced a few steps ago
      'distant' -- a fact produced early in this run
      'derived' -- a value that must be recomposed from three earlier facts
      'prior'   -- a fact established in a PREVIOUS run
    """
    ok = np.ones(N, dtype=bool)
    for i in range(k):
        if need == "recent":
            dist = rng.integers(1, 4, size=N)
        elif need == "distant":
            dist = rng.integers(1, k + 1, size=N)
        else:
            dist = np.full(N, 1)

        if need == "prior":
            # Nothing in this run contains it. Only a store can supply it.
            got = np.full(N, store)
            p = np.where(got, P_LOOKUP, 0.25)
        elif need == "derived":
            if scratchpad:
                # Written down once; every later use is a single lookup.
                p = np.full(N, P_LOOKUP)
            else:
                # Recomposed from three facts inside one forward pass.
                p = np.full(N, P_COMPOSE ** 2)
                if not context:
                    p = p * 0.5
        else:
            in_ctx = context & (dist <= ctx)
            # Recall degrades as the window fills, even inside it.
            recall = np.clip(P_LOOKUP - DILUTE * np.minimum(i, ctx), 0.2, 1.0)
            p = np.where(in_ctx, recall,
                         np.where(store, P_LOOKUP * 0.9, 0.30))
        ok &= rng.random(N) < p
    return float(ok.mean())


NEEDS = ["recent", "distant", "derived", "prior"]
CONFIGS = [
    ("context only", dict(context=True)),
    ("context + scratchpad", dict(context=True, scratchpad=True)),
    ("context + store", dict(context=True, store=True)),
    ("all three", dict(context=True, scratchpad=True, store=True)),
]

print(f"A {K}-step run. The model attends reliably to the last {CTX} steps of raw")
print(f"history, with recall degrading {DILUTE:.1%} per entry as the window")
print("fills. Each row is a kind of dependency the steps have.")
print()
print(f"{'dependency':>13}" + "".join(f"{n:>22}" for n, _ in CONFIGS))
print("-" * 101)
tab = {}
for need in NEEDS:
    row = {}
    for name, cfg in CONFIGS:
        row[name] = run(need, **cfg)
        tab[(need, name)] = row[name]
    print(f"{need:>13}" + "".join(f"{row[n]:>22.1%}" for n, _ in CONFIGS))

print()
print()
print("The wrong fix, priced. For each dependency, what does EXTENDING THE")
print("CONTEXT buy, against the mechanism that actually addresses it?")
print()
print(f"{'dependency':>13}{'ctx 6':>9}{'ctx 14':>9}{'ctx 30':>9}"
      f"{'right fix':>24}{'that fix':>11}")
print("-" * 76)
fixes = {"recent": ("nothing needed", dict(context=True)),
         "distant": ("add a store", dict(context=True, store=True)),
         "derived": ("add a scratchpad", dict(context=True, scratchpad=True)),
         "prior": ("add a store", dict(context=True, store=True))}
ext = {}
for need in NEEDS:
    a = run(need, context=True, ctx=6)
    b = run(need, context=True, ctx=14)
    c = run(need, context=True, ctx=30)
    label, cfg = fixes[need]
    d = run(need, ctx=6, **cfg)
    ext[need] = (a, b, c, d)
    print(f"{need:>13}{a:>9.1%}{b:>9.1%}{c:>9.1%}{label:>24}{d:>11.1%}")

print()
print()
print("What a scratchpad costs and saves on a 'derived' workload, as the number")
print("of times a derived value is reused grows.")
print()
print(f"{'reuses':>8}{'recompute each time':>22}{'write once, read':>19}"
      f"{'gain':>9}")
print("-" * 58)
re_tab = {}
for r in (1, 2, 4, 8, 14):
    a = run("derived", context=True, k=r)
    b = run("derived", context=True, scratchpad=True, k=r)
    re_tab[r] = (a, b)
    print(f"{r:>8}{a:>22.1%}{b:>19.1%}{b - a:>+9.1%}")

print()
print()
print("And how context dilution changes the picture -- the term that makes a")
print("longer window stop helping.")
print()
print(f"{'dilution':>10}{'ctx 6':>10}{'ctx 14':>10}{'ctx 30':>10}{'best':>9}")
print("-" * 49)
dl = {}
for d in (0.0, 0.01, 0.02, 0.04, 0.08):
    DIL_SAVE = DILUTE
    globals()["DILUTE"] = d
    row = [run("distant", context=True, ctx=c) for c in (6, 14, 30)]
    globals()["DILUTE"] = DIL_SAVE
    dl[d] = row
    print(f"{d:>10.0%}" + "".join(f"{v:>10.1%}" for v in row)
          + f"{[6, 14, 30][int(np.argmax(row))]:>9}")

print(f"""
The first table is the argument for refusing the umbrella term, and it is a
diagonal.

A dependency on a RECENT fact is handled by the context and by nothing else:
{tab[('recent', 'context only')]:.1%} with context alone,
{tab[('recent', 'all three')]:.1%} with all three mechanisms. Adding a scratchpad
or a store buys nothing, because the fact is already there.

A DERIVED value -- one that must be recomposed from several earlier facts -- goes
from {tab[('derived', 'context only')]:.1%} to
{tab[('derived', 'context + scratchpad')]:.1%} with a scratchpad, and a store does
nothing for it ({tab[('derived', 'context + store')]:.1%}). The context contains
everything needed; what it does not contain is the RESULT, so every use pays the
composition again.

A PRIOR fact -- established in an earlier run -- is {tab[('prior', 'context only')]:.1%}
without a store and {tab[('prior', 'context + store')]:.1%} with one. No amount of
context or scratchpad reaches it, because it is not in this run.

**Three mechanisms, three needs, one match each.** Calling them all "memory"
guarantees that a team will reach for whichever one it already has, and the table
says that is right one time in three.

The second table prices the wrong fix, and its first row is the result worth
carrying away.

Extending the context from {6} to {14} steps takes the RECENT case from
{ext['recent'][0]:.1%} to {ext['recent'][1]:.1%}. **A longer window made recall of
recent facts worse.** Not neutral -- worse, by
{ext['recent'][0] - ext['recent'][1]:.1%}.

The mechanism is dilution: recall degrades as the window fills, and it degrades
for everything in the window, including the things that were being recalled
perfectly well before. A longer context is not a superset of a shorter one; it is
a different retrieval problem over a larger candidate set, and part:7's attention
arithmetic is why.

That single row explains a very common production experience -- moving to a
longer-context model and finding some behaviours got worse -- and it says the fix
is not more context.

Down the rest of the column: extending the context does nothing at all for
DERIVED ({ext['derived'][0]:.1%} to {ext['derived'][2]:.1%} across a fivefold
window increase) and nothing for PRIOR ({ext['prior'][2]:.1%} at {30} steps of
history). The right fixes, at the ORIGINAL context length, deliver
{ext['derived'][3]:.1%} and {ext['prior'][3]:.1%}.

**The right mechanism at a small context beats the wrong mechanism at a large
one**, in every row, by a wide margin.

The third table says when a scratchpad is worth writing, and the answer is "almost
always, and increasingly".

At one reuse, writing the derived value down instead of recomposing it buys
{re_tab[1][1] - re_tab[1][0]:+.1%}. At fourteen reuses it buys
{re_tab[14][1] - re_tab[14][0]:+.1%}, taking the run from
{re_tab[14][0]:.1%} to {re_tab[14][1]:.1%}.

The reason is the same exponent that has governed this whole part. Recomposing
pays {P_COMPOSE:.0%}-squared every time the value is needed, so the cost is
geometric in reuses. Writing it once converts every subsequent use into a lookup
at {P_LOOKUP:.1%}. **A scratchpad is not a note-taking convenience. It is a way of
removing a repeated composition from the exponent**, which is ch:ag-react's
thought-as-memory observation with a number attached.

The last table is the one that decides whether a longer window helps at all, and
it isolates the term that everything else in this chapter has been working around.

With NO dilution, a longer window is strictly better: {dl[0.0][0]:.1%} at {6}
steps against {dl[0.0][2]:.1%} at {30}. The naive intuition is correct in that
world. At {0.02:.0%} dilution per entry the same comparison is
{dl[0.02][0]:.1%} against {dl[0.02][2]:.1%} -- still better, and both are now
poor. At {0.08:.0%} nothing works at any window size.

So the value of context length is entirely a function of how well recall holds up
across the window, and that is a property of the model rather than of the
architecture. **Context length is a capability you should measure on your model
rather than a resource you should assume**, and the measurement is cheap: place a
fact at varying distances and ask for it back.

The practical summary is four rules, one per row of the first table.

For facts produced a few steps ago: nothing to do, and do not lengthen the window.

For facts produced early in a long run: promote them out of the raw history into a
store, because the window will not hold them and dilution punishes trying.

For values that get recomputed: write them down once. The gain scales with reuse
count and it is the largest single number in this listing.

For anything that must survive the run: it needs a store, and there is no
substitute -- which is where part:12's retrieval material applies directly, with
the previous listing's staleness caveat attached.""")
