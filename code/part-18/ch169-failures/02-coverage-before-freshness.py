# -*- coding: utf-8 -*-
# Extracted from: Chapter 169 — Multi-Agent Failure Modes
# Source: src/.../ch169-failures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where a multi-agent failure gets caught, which decides what it costs.

An error made by agent 3 of 8 does not sit still. Agent 4 reads agent 3's output
and builds on it; agent 5 builds on agent 4. By the time anything checks, the error
is not a wrong sentence -- it is a premise that five agents have written consistent
work around.

Two consequences, and the second is the one that matters:

  cost        repairing an error means redoing everything built on it, so cost
              grows with detection lag
  detectability an error surrounded by consistent downstream work looks RIGHT.
              Detection probability DECAYS with lag (eq:detection-decays-with-lag)

Most systems put their critic at the end, which maximises lag on both counts. This
listing measures placement at a fixed critic budget.
"""
import numpy as np

rng = np.random.default_rng(3877)

M = 60000
K = 8                   # agents in the chain
P_ERR = 0.06            # chance a given agent introduces an error
C0 = 0.88               # a critic's detection rate at zero lag
DECAY = 0.72            # detection multiplier per step of lag
FIX = 0.85              # a caught error is repaired this often


def run(critic_positions, m=M, k=K, p_err=P_ERR, c0=C0, decay=DECAY):
    """Errors appear at agent positions; critics sit after chosen positions. A
    critic catches an error with probability c0 * decay**lag, and repairing costs
    the work done since the error was made."""
    err_at = rng.random((m, k)) < p_err
    first = np.where(err_at.any(1), err_at.argmax(1), -1)
    alive_err = first.copy()          # -1 means no outstanding error
    rework = np.zeros(m, dtype=np.float64)
    caught = np.zeros(m, dtype=bool)
    for j in sorted(critic_positions):
        has = alive_err >= 0
        lag = np.where(has, j - alive_err, 0)
        visible = has & (lag >= 0)
        p = c0 * (decay ** np.clip(lag, 0, None))
        hit = visible & (rng.random(m) < p) & (rng.random(m) < FIX)
        rework[hit] += lag[hit] + 1
        caught |= hit
        alive_err[hit] = -1
    shipped_bad = alive_err >= 0
    n_crit = len(critic_positions)
    return (float((~shipped_bad).mean()), float(rework.mean()),
            float(k + rework.mean() + n_crit), n_crit)


LAST = [K - 1]
MID = [K // 2, K - 1]
EVERY2 = list(range(1, K, 2))
EVERY1 = list(range(K))
EARLY = [0, 1, 2, 3]

print(f"{M:,} runs through {K} agents; each introduces an error with")
print(f"probability {P_ERR:.0%}. A critic detects at {C0:.0%} when the error is")
print(f"fresh, decaying {DECAY:.0%} per step of lag as downstream work makes it")
print("look consistent.")
print()
print(f"{'critic placement':>22}{'critics':>9}{'clean output':>14}"
      f"{'rework':>9}{'total cost':>12}")
print("-" * 66)
plans = [("one at the end", LAST), ("middle + end", MID),
         ("every 2nd agent", EVERY2), ("after every agent", EVERY1),
         ("four, all early", EARLY)]
tab = {}
for name, pos in plans:
    r = run(pos)
    tab[name] = r
    print(f"{name:>22}{r[3]:>9}{r[0]:>14.1%}{r[1]:>9.2f}{r[2]:>12.1f}")

print()
print()
print("The controlled comparison: FOUR critics, placed differently.")
print()
print(f"{'four critics at':>22}{'clean output':>14}{'rework':>9}"
      f"{'mean lag when caught':>23}")
print("-" * 68)
four = {}
FOURS = [("0,1,2,3 (early)", [0, 1, 2, 3]),
         ("1,3,5,7 (spread)", [1, 3, 5, 7]),
         ("4,5,6,7 (late)", [4, 5, 6, 7])]
for name, pos in FOURS:
    r = run(pos)
    four[name] = r
    lag = r[1] / max(r[0] - (1 - P_ERR) ** K, 1e-9)
    print(f"{name:>22}{r[0]:>14.1%}{r[1]:>9.2f}{lag:>23.2f}")

print()
print()
print("How much of this is the decay? Same placements with detection that does")
print("not degrade with lag -- a critic as good on stale errors as fresh ones.")
print()
print(f"{'four critics at':>22}{'with decay':>13}{'no decay':>11}{'loss':>10}")
print("-" * 56)
nd = {}
for name, pos in FOURS:
    a = run(pos)[0]
    b = run(pos, decay=1.0)[0]
    nd[name] = (a, b)
    print(f"{name:>22}{a:>13.1%}{b:>11.1%}{a - b:>+10.1%}")

print()
print()
print("And against chain length, since a longer chain gives an early error more")
print("room to become invisible.")
print()
print(f"{'agents':>8}{'end only':>11}{'spread':>10}{'early':>10}{'best':>10}")
print("-" * 49)
cl = {}
for k in (4, 8, 16, 24):
    end = run([k - 1], k=k)[0]
    spread = run(list(range(1, k, max(1, k // 4))), k=k)[0]
    early = run(list(range(min(4, k))), k=k)[0]
    names = ["end only", "spread", "early"]
    row = (end, spread, early)
    cl[k] = (row, names[int(np.argmax(row))])
    print(f"{k:>8}{end:>11.1%}{spread:>10.1%}{early:>10.1%}{cl[k][1]:>10}")

print(f"""
The first table shows the obvious thing and hides the interesting one. More critics
catch more: one at the end gives {tab['one at the end'][0]:.1%} and one after every
agent gives {tab['after every agent'][0]:.1%}, at a total cost of
{tab['one at the end'][2]:.1f} against {tab['after every agent'][2]:.1f}.

But notice the last row. FOUR critics, all placed early, give
{tab['four, all early'][0]:.1%} -- worse than four critics placed every second agent
at {tab['every 2nd agent'][0]:.1%}, and barely better than TWO critics at the middle
and end.

The controlled table isolates that, and the ordering is not the one the chapter's
premise predicts. Four critics spread across the chain give {four['1,3,5,7 (spread)'][0]:.1%};
four placed late give {four['4,5,6,7 (late)'][0]:.1%}; four placed early give
{four['0,1,2,3 (early)'][0]:.1%}.

**Early placement is the worst of the three**, despite catching errors at the
lowest lag -- mean lag {four['0,1,2,3 (early)'][1] / max(four['0,1,2,3 (early)'][0] - (1 - P_ERR) ** K, 1e-9):.2f}
against {four['4,5,6,7 (late)'][1] / max(four['4,5,6,7 (late)'][0] - (1 - P_ERR) ** K, 1e-9):.2f} -- for a reason
that is obvious once stated and easy to miss when reasoning about freshness alone:
a critic at position 1 cannot catch an error made at position 5. Early critics have
no coverage of the second half of the chain.

So the rule is **coverage first, then freshness** (eq:coverage-before-freshness).
Spread wins because it is the only placement with both.

Freshness is still real and the third table prices it. Turning off the decay -- a
critic as good on a five-step-old error as on a fresh one -- is worth
{nd['4,5,6,7 (late)'][1] - nd['4,5,6,7 (late)'][0]:+.1%} to late placement and
{nd['0,1,2,3 (early)'][1] - nd['0,1,2,3 (early)'][0]:+.1%} to early placement.

**Almost all of the late placement's disadvantage is the decay**
(eq:detection-decays-with-lag), which is the mechanism worth naming: an error
surrounded by consistent downstream work does not look like an error. Five agents
have each written something coherent given the mistaken premise, and a critic
reading the result sees five agreements. It is ch:as-multi-agent's correlation
appearing inside a single run rather than across parallel ones.

The last table says the effect grows with the chain, which is the practical warning.
At {4} agents the placements are within {max(cl[4][0]) - min(cl[4][0]):.1%} of each
other and none of this matters. At {24} agents, one critic at the end gives
{cl[24][0][0]:.1%} and spread critics give {cl[24][0][1]:.1%}.

**The default architecture -- a long chain of agents with a reviewer at the end --
is the worst available placement, and it gets worse the longer the chain.** It is
also, for entirely non-technical reasons, the one almost everyone builds: the
reviewer is added last because that is when someone notices the output is wrong.

Three rules.

**Spread critics across the chain rather than concentrating them.** Coverage is the
binding constraint and freshness is the tiebreak.

**Shorten the chain.** Every result here degrades with length, and ch:as-multi-agent
already found handoff count to be an exponent.

**Do not measure a critic's detection rate on fresh errors** -- the number that
matters is its rate at the lag it will actually see, which is much lower.""")
