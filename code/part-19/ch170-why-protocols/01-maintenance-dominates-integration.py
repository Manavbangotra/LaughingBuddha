# -*- coding: utf-8 -*-
# Extracted from: Chapter 170 — Why Tool Protocols Exist
# Source: src/.../ch170-why-protocols.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The N x M argument, priced properly.

The case for a tool protocol is always drawn the same way: N hosts and M tool
providers need N x M bespoke integrations, and a protocol turns that into N + M.

That picture is right about the shape and wrong about the economics, because it
compares BUILD costs and the build cost is not what dominates. An integration is
not built once; it is maintained against changes on both sides for as long as it
exists (eq:maintenance-dominates-integration).

This listing prices both regimes over time and finds the break-even.
"""
import numpy as np

# Costs in engineer-days. These are the model's assumptions, stated plainly so
# the conclusions can be checked against your own numbers.
BESPOKE_BUILD = 12.0     # one host <-> one provider, from scratch
BESPOKE_FIX = 3.0        # repairing one integration after a change on either side
PROTO_SERVER = 18.0      # implementing a compliant server: more than one adapter
PROTO_CLIENT = 25.0      # implementing a compliant client: more still
PROTO_FIX = 2.0          # repairing one endpoint after a spec revision
SPEC_LEARN = 8.0         # per team, one-time cost of learning the protocol

HOST_CHANGES = 1.4       # breaking changes per host per year
PROV_CHANGES = 2.1       # breaking changes per provider per year
SPEC_REVS = 0.8          # backwards-incompatible spec revisions per year


def bespoke(n, m, years):
    """Every host-provider pair is its own integration, and every change on
    either side breaks the pairs that touch it."""
    build = n * m * BESPOKE_BUILD
    # A host change breaks that host's m integrations; a provider change breaks
    # that provider's n integrations.
    breaks = years * (n * HOST_CHANGES * m + m * PROV_CHANGES * n)
    return build + breaks * BESPOKE_FIX


def protocol(n, m, years):
    """Each side implements the protocol once. A change on one side is absorbed
    by the protocol rather than propagated; only spec revisions touch everyone."""
    build = n * PROTO_CLIENT + m * PROTO_SERVER + (n + m) * SPEC_LEARN
    revs = years * SPEC_REVS * (n + m)
    return build + revs * PROTO_FIX


print("Integration cost in engineer-days. Bespoke: every host-provider pair is")
print("its own adapter. Protocol: each side implements the protocol once.")
print()
print(f"{'hosts x providers':>19}{'bespoke':>11}{'protocol':>11}"
      f"{'ratio':>9}{'winner':>10}")
print("-" * 60)
Y = 3.0
grid = {}
for n, m in ((2, 3), (3, 8), (5, 20), (10, 60), (20, 200)):
    b = bespoke(n, m, Y)
    p = protocol(n, m, Y)
    grid[(n, m)] = (b, p)
    print(f"{f'{n} x {m}':>19}{b:>11,.0f}{p:>11,.0f}{b / p:>9.1f}"
          f"{('protocol' if p < b else 'bespoke'):>10}")

print()
print()
print("The same ecosystems at year zero -- build cost only, before anything has")
print("had to be maintained.")
print()
print(f"{'hosts x providers':>19}{'bespoke':>11}{'protocol':>11}"
      f"{'ratio':>9}{'winner':>10}")
print("-" * 60)
zero = {}
for n, m in ((2, 3), (3, 8), (5, 20), (10, 60), (20, 200)):
    b = bespoke(n, m, 0.0)
    p = protocol(n, m, 0.0)
    zero[(n, m)] = (b, p)
    print(f"{f'{n} x {m}':>19}{b:>11,.0f}{p:>11,.0f}{b / p:>9.1f}"
          f"{('protocol' if p < b else 'bespoke'):>10}")

print()
print()
print("How the verdict moves with time, for a small ecosystem where the build")
print("cost initially favours bespoke.")
print()
N0, M0 = 3, 6
print(f"{'years':>7}{'bespoke':>11}{'protocol':>11}{'advantage':>12}")
print("-" * 41)
tm = {}
for y in (0.0, 0.5, 1.0, 2.0, 4.0):
    b = bespoke(N0, M0, y)
    p = protocol(N0, M0, y)
    tm[y] = (b, p)
    print(f"{y:>7.1f}{b:>11,.0f}{p:>11,.0f}{b - p:>+12,.0f}")

print()
print()
print("Break-even ecosystem size, as the number of providers at a fixed host")
print("count. The smallest m at which the protocol is cheaper.")
print()
print(f"{'hosts':>7}{'break-even m, year 0':>22}{'year 1':>10}{'year 3':>10}")
print("-" * 49)
be = {}
for n in (1, 2, 3, 5, 10):
    row = []
    for y in (0.0, 1.0, 3.0):
        m = 1
        while m < 10000 and protocol(n, m, y) >= bespoke(n, m, y):
            m += 1
        row.append(m if m < 10000 else None)
    be[n] = row
    cells = "".join(f"{(str(v) if v else 'never'):>{w}}"
                    for v, w in zip(row, (22, 10, 10)))
    print(f"{n:>7}{cells}")

print()
print()
print("What each regime is actually paying for, at 10 x 60 over three years.")
print()
n, m, y = 10, 60, 3.0
b_build = n * m * BESPOKE_BUILD
b_maint = bespoke(n, m, y) - b_build
p_build = n * PROTO_CLIENT + m * PROTO_SERVER + (n + m) * SPEC_LEARN
p_maint = protocol(n, m, y) - p_build
print(f"{'':>12}{'build':>12}{'maintenance':>14}{'maint share':>14}")
print("-" * 52)
print(f"{'bespoke':>12}{b_build:>12,.0f}{b_maint:>14,.0f}"
      f"{b_maint / (b_build + b_maint):>14.0%}")
print(f"{'protocol':>12}{p_build:>12,.0f}{p_maint:>14,.0f}"
      f"{p_maint / (p_build + p_maint):>14.0%}")

print()
print()
print("And who pays. In the bespoke regime someone must own each pair; under a")
print("protocol each party implements once. Cost borne by a single NEW provider")
print("joining an ecosystem that already has n hosts:")
print()
print(f"{'hosts already present':>23}{'bespoke':>11}{'protocol':>11}")
print("-" * 45)
who = {}
for n in (1, 3, 10, 30, 100):
    b = n * BESPOKE_BUILD + Y * n * (HOST_CHANGES + PROV_CHANGES) * BESPOKE_FIX
    p = PROTO_SERVER + SPEC_LEARN + Y * SPEC_REVS * PROTO_FIX
    who[n] = (b, p)
    print(f"{n:>23}{b:>11,.0f}{p:>11,.0f}")

print(f"""
The first table is the argument as it is usually made, and it is correct: at
{20} x {200} the protocol costs {grid[(20, 200)][1] / grid[(20, 200)][0]:.0%} of
bespoke. The second table is the same ecosystems with the maintenance term
removed, and it disagrees at the small end.

At {2} x {3}, building bespoke adapters costs {zero[(2, 3)][0]:,.0f} days against
the protocol's {zero[(2, 3)][1]:,.0f}. At {3} x {8} it is still
{zero[(3, 8)][0]:,.0f} against {zero[(3, 8)][1]:,.0f}. **For a small ecosystem the
protocol loses on build cost**, and it loses for an unsurprising reason: a
compliant server is more work than a single-purpose adapter, and a compliant
client is more work still.

So the N x M argument is not really a build-cost argument, and stating it as one
invites a correct objection from anyone with three integrations to write.

The third table shows what it actually is. The same {N0} x {M0} ecosystem that
favours bespoke by {tm[0.0][1] - tm[0.0][0]:,.0f} days at year zero favours the
protocol by {tm[1.0][0] - tm[1.0][1]:,.0f} days after one year and
{tm[4.0][0] - tm[4.0][1]:,.0f} after four.

The decomposition table says why. Over three years at {10} x {60}, maintenance is
{b_maint / (b_build + b_maint):.0%} of the bespoke regime's cost and
{p_maint / (p_build + p_maint):.0%} of the protocol's.

**The N x M problem is a maintenance problem wearing a build problem's clothes**
(eq:maintenance-dominates-integration). A bespoke adapter has to be repaired
whenever either side changes, and there are N x M of them to repair; a protocol
endpoint is repaired only when the PROTOCOL changes, and there are N + M of them.
The quadratic term is in the maintenance, not the construction.

The break-even table makes this concrete and slightly startling. At year zero a
single host needs an ecosystem the model never reaches before the protocol pays
off. At three years, one host breaks even at {be[1][2]} providers and ten hosts
break even at {be[10][2]}.

**Almost the entire case for a protocol is made by the second year**, which is
also why the case is so hard to make in advance: the costs it avoids have not
happened yet, and the costs it imposes are due immediately.

The last table is the mechanism behind adoption, and it is not about totals at
all. A new provider joining an ecosystem with {100} hosts pays
{who[100][0]:,.0f} days to integrate bespoke and {who[100][1]:,.0f} days to
implement the protocol -- and that {who[100][1]:,.0f} is the SAME number it would
pay to join an ecosystem with one host.

**A protocol turns the marginal joiner's cost from O(N) into O(1)**
(eq:protocol-makes-entry-constant). That is a different claim from the total-cost
claim and a much stronger one, because the marginal joiner is the party deciding
whether to participate. An ecosystem can be below its total-cost break-even and
still adopt a protocol enthusiastically, because every individual decision to
join is made against the O(N) alternative.

Which is the honest summary of why these protocols spread the way they do. Not
because someone computed the ecosystem total -- nobody is in a position to -- but
because each participant faced a constant cost instead of a growing one.""")
