# -*- coding: utf-8 -*-
# Extracted from: Chapter 176 — Production MCP and Ecosystem Design
# Source: src/.../ch176-production.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How many servers should a host connect? The part's closing question.

Every chapter in part:19 measured one cost of connecting a server:

  ch:mcp-schemas    schema rent -- tokens spent on every request, linear in tools
  ch:mcp-security   exposure -- more servers, more chances one is hostile
  ch:mcp-architecture  correlation -- a shared server's outage hits every client
  ch:mcp-why        integration -- amortised, and the one that gets cheaper

Against those, capability: a new server can do something the others cannot. That
benefit SATURATES, because the tenth issue-tracker integration adds less than the
first (eq:marginal-server-turns-negative).

Saturating benefit against linear costs has an interior optimum, and this listing
finds it. Nobody computes this; hosts connect servers until something feels wrong.
"""
import numpy as np

rng = np.random.default_rng(4423)

M = 30000
STEPS = 6
TOOLS_PER_SERVER = 9
TOKENS_PER_TOOL = 170
DILUTE = 2.0e-6         # per-token degradation, as in ch:mcp-schemas
BASE = 0.995
P_HOSTILE = 0.02        # ch:mcp-security, set by registry policy
P_OBEY = 0.55
COVER_SCALE = 5.5       # servers at which ~63% of needed capability is covered


def coverage(n_servers, scale=COVER_SCALE):
    """Saturating: each server adds less unique capability than the last."""
    return 1.0 - np.exp(-n_servers / scale)


def run(n_servers, m=M, steps=STEPS, retrieval=None, p_hostile=P_HOSTILE,
        dilute=DILUTE, cover_scale=COVER_SCALE):
    """`retrieval` caps how many tool schemas reach context, per ch:mcp-schemas."""
    tools = n_servers * TOOLS_PER_SERVER
    shown = tools if retrieval is None else min(retrieval, tools)
    tokens = shown * TOKENS_PER_TOOL

    # Capability: does the host have a server that can do what the task needs?
    have = rng.random(m) < coverage(n_servers, cover_scale)
    # Rent: everything in context competes with everything else.
    p_step = BASE * (1.0 - dilute * tokens)
    reason_ok = rng.random(m) < np.clip(p_step, 0.0, 1.0) ** steps
    # Exposure: at least one connected server is hostile, and the model obeys.
    hostile = rng.random(m) < (1.0 - (1.0 - p_hostile) ** n_servers)
    harmed = hostile & (rng.random(m) < P_OBEY)

    ok = have & reason_ok & ~harmed
    return (float(ok.mean()), tokens, float(coverage(n_servers, cover_scale)),
            float(harmed.mean()))


print(f"{M:,} tasks. Each server brings {TOOLS_PER_SERVER} tools at about")
print(f"{TOKENS_PER_TOOL} tokens of schema, adds capability with diminishing")
print(f"returns, and is hostile with probability {P_HOSTILE:.0%}.")
print()
print(f"{'servers':>9}{'coverage':>11}{'schema tokens':>15}{'harm rate':>11}"
      f"{'success':>10}")
print("-" * 56)
tab = {}
for n in (1, 2, 4, 8, 16, 32):
    r = run(n)
    tab[n] = r
    print(f"{n:>9}{r[2]:>11.1%}{r[1]:>15,.0f}{r[3]:>11.2%}{r[0]:>10.1%}")

peak = max(tab, key=lambda k: tab[k][0])

print()
print()
print("The same sweep with ch:mcp-schemas' retrieval layer capping how many")
print("schemas reach the context at 24.")
print()
print(f"{'servers':>9}{'no retrieval':>14}{'retrieval 24':>14}{'gain':>9}")
print("-" * 46)
rt = {}
for n in (1, 2, 4, 8, 16, 32, 64):
    a = run(n)[0]
    b = run(n, retrieval=24)[0]
    rt[n] = (a, b)
    print(f"{n:>9}{a:>14.1%}{b:>14.1%}{b - a:>+9.1%}")

peak_rt = max(rt, key=lambda k: rt[k][1])

print()
print()
print("Which cost binds depends on registry policy, and bh1 showed that is")
print("chosen rather than given. Best server count under each:")
print()
print(f"{'hostile rate':>14}{'best n':>9}{'success there':>15}"
      f"{'success at n=32':>17}")
print("-" * 55)
hp = {}
for p in (0.002, 0.02, 0.06):
    vals = {n: run(n, retrieval=24, p_hostile=p)[0]
            for n in (1, 2, 4, 8, 16, 32, 64)}
    b = max(vals, key=lambda k: vals[k])
    hp[p] = (b, vals[b], vals[32])
    print(f"{p:>14.1%}{b:>9}{vals[b]:>15.1%}{vals[32]:>17.1%}")

print()
print()
print("And with how broad the task distribution is -- a host serving varied")
print("work needs more coverage before saturation sets in.")
print()
print(f"{'coverage scale':>16}{'best n':>9}{'success there':>15}")
print("-" * 40)
cs = {}
for s in (2.0, 5.5, 14.0):
    vals = {n: run(n, retrieval=24, cover_scale=s)[0]
            for n in (1, 2, 4, 8, 16, 32, 64)}
    b = max(vals, key=lambda k: vals[k])
    cs[s] = (b, vals[b])
    print(f"{s:>16.1f}{b:>9}{vals[b]:>15.1%}")

print()
print()
print("Decomposing the marginal server at the optimum: what the next one adds")
print("and what it costs.")
print()
print(f"{'from n to n+1':>15}{'coverage gain':>15}{'added tokens':>14}"
      f"{'added harm':>12}{'net':>9}")
print("-" * 65)
mg = {}
for n in (1, 2, 4, 8, 16, 32):
    a = run(n, retrieval=24)
    b = run(n + 1, retrieval=24)
    mg[n] = (b[2] - a[2], b[1] - a[1], b[3] - a[3], b[0] - a[0])
    print(f"{f'{n} -> {n + 1}':>15}{b[2] - a[2]:>+15.1%}{b[1] - a[1]:>+14,.0f}"
          f"{b[3] - a[3]:>+12.2%}{b[0] - a[0]:>+9.1%}")

print(f"""
The first table has a peak in it, which is the answer to a question hosts do not
usually ask.

Success rises to {tab[peak][0]:.1%} at {peak} servers and falls to
{tab[32][0]:.1%} at {32}. Coverage is still climbing -- {tab[32][2]:.1%} at
thirty-two -- and it climbs into two costs that do not stop: schema rent at
{tab[32][1]:,.0f} tokens, and a {tab[32][3]:.1%} chance that one of the connected
servers is hostile and obeyed.

**Saturating benefit against linear costs has an interior optimum**
(eq:marginal-server-turns-negative), and connecting past it makes the host worse
at everything, not just more expensive.

The second table shows that one of those two costs is removable.
ch:mcp-schemas' retrieval layer, capping the schemas that reach context at
{24}, is worth {rt[8][1] - rt[8][0]:+.1%} at {8} servers and
{rt[64][1] - rt[64][0]:+.1%} at {64} -- and it moves the optimum from
{peak} servers to {peak_rt}.

**Retrieval does not merely improve a large inventory; it changes how many
servers it is rational to connect.** A host without one is choosing between
capability and context on every integration decision, and a host with one is only
paying the security cost.

The third table connects this listing to the previous one. At a
{0.002:.1%} hostile rate the best server count is {hp[0.002][0]}; at
{0.06:.0%} it is {hp[0.06][0]}, and connecting {32} costs
{hp[0.06][1] - hp[0.06][2]:.1f} points against the optimum.

**Registry policy sets how many servers a host can afford to connect.** That is
the strongest argument for the admission work in the previous listing, and it is
not the argument usually made for it: a well-governed registry is not merely
safer, it lets every host that uses it be more capable, because the marginal
server stays positive for longer.

The fourth table says the optimum is also a property of the host's own workload.
Narrow, repetitive work saturates at {cs[2.0][0]} servers; varied work at
{cs[14.0][0]}. So there is no ecosystem-wide right answer, and a host should
compute this from its own task distribution rather than copying a number.

The last table is the optimality condition made visible, and it is the most
useful thing here. Going from {16} to {17} servers adds
{mg[16][0]:+.1%} of coverage and {mg[16][2]:+.2%} of harm exposure, for a net of
{mg[16][3]:+.1%}. Going from {8} to {9} adds {mg[8][0]:+.1%} coverage against
{mg[8][2]:+.2%} harm, for {mg[8][3]:+.1%}.

**The marginal server turns negative when the capability it adds falls below the
exposure it adds**, and because the first shrinks while the second is roughly
constant, that crossing always happens. The only question is where.

Which gives the part's closing instruction, and it is one nobody follows.
Before connecting a server, ask what fraction of your tasks it would newly enable.
If the answer is under about one percent, it is costing more than it brings --
and the honest version of "we support fifty integrations" is that most of them are
making the other forty-nine work slightly worse.""")
