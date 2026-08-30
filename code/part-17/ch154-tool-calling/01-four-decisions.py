# -*- coding: utf-8 -*-
# Extracted from: Chapter 154 — Tool Calling and Tool Design
# Source: src/.../ch154-tool-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Tool-call accuracy is four numbers, and only one of them moves with scale.

cite:schick2023toolformer decomposes tool use into four decisions: which API to
call, when to call it, what arguments to pass, and how to incorporate the result.
Systems report one aggregate "tool use accuracy" that multiplies all four
together, which hides the fact that they behave completely differently as a
system grows (eq:four-decisions).

This listing measures each stage separately against the size of the tool
inventory. Selection is modelled as what it is -- a nearest-neighbour
discrimination in a semantic space, where a request lands near its correct tool
and the model picks the closest one. Everything else is modelled as a per-call
property that has no reason to depend on how many other tools exist.

The prediction going in was that selection degrades with inventory size. It does
not, and the reason it does not is the useful finding: in a space of any real
dimensionality, genuinely distinct tools stay far apart no matter how many you
add. What destroys selection is OVERLAP, and inventories acquire overlap by
growing the way real ones do.
"""
import numpy as np

rng = np.random.default_rng(1597)

D = 24                  # semantic dimensions
N_REQ = 30000
SIGMA = 0.55            # how imprecisely a request locates its tool
P_WHEN = 0.97           # decides correctly WHETHER to call a tool
P_ARGS_BASE = 0.985     # gets one argument right
P_READ = 0.98           # reads the result back correctly


def selection_accuracy(n_tools, cluster=0):
    """Tools live at points in a semantic space; a request is its tool's point
    plus noise; the model picks the nearest tool. `cluster` groups tools into
    families whose descriptions overlap, which is what a real inventory looks
    like once it has grown by accretion."""
    if cluster:
        centres = rng.normal(size=(cluster, D))
        assign = rng.integers(0, cluster, size=n_tools)
        tools = centres[assign] + 0.35 * rng.normal(size=(n_tools, D))
    else:
        tools = rng.normal(size=(n_tools, D))
    tgt = rng.integers(0, n_tools, size=N_REQ)
    req = tools[tgt] + SIGMA * rng.normal(size=(N_REQ, D))
    # Nearest tool by squared distance.
    d = ((req[:, None, :] - tools[None, :, :]) ** 2).sum(2)
    return float(np.mean(d.argmin(1) == tgt))


SIZES = [4, 8, 16, 32, 64, 128]
N_ARGS = 3

print(f"{N_REQ} requests. A request locates its tool to within noise {SIGMA};")
print(f"the model calls the nearest tool. Each call needs {N_ARGS} arguments,")
print(f"each right {P_ARGS_BASE:.1%} of the time. Deciding whether to call at")
print(f"all is {P_WHEN:.0%} accurate and reading the result back {P_READ:.0%}.")
print()
print(f"{'tools':>8}{'select':>10}{'when':>9}{'arguments':>12}{'read':>9}"
      f"{'end to end':>13}")
print("-" * 61)

p_args = P_ARGS_BASE ** N_ARGS
stages = {}
for n in SIZES:
    sel = selection_accuracy(n)
    e2e = sel * P_WHEN * p_args * P_READ
    stages[n] = (sel, P_WHEN, p_args, P_READ, e2e)
    print(f"{n:>8}{sel:>10.1%}{P_WHEN:>9.1%}{p_args:>12.1%}{P_READ:>9.1%}"
          f"{e2e:>13.1%}")

print()
print()
print("Which stage is costing the most? Errors attributable to each, as a share")
print("of all end-to-end failures.")
print()
print(f"{'tools':>8}{'select':>10}{'when':>9}{'arguments':>12}{'read':>9}")
print("-" * 48)
for n in SIZES:
    sel = stages[n][0]
    losses = np.array([1 - sel, 1 - P_WHEN, 1 - p_args, 1 - P_READ])
    share = losses / losses.sum()
    print(f"{n:>8}{share[0]:>10.1%}{share[1]:>9.1%}{share[2]:>12.1%}"
          f"{share[3]:>9.1%}")

print()
print()
print("What happens when the inventory grows by accretion, so tools cluster")
print("into families with overlapping descriptions?")
print()
print(f"{'tools':>8}{'distinct':>12}{'8 families':>13}{'4 families':>13}"
      f"{'2 families':>13}")
print("-" * 59)
clus = {}
for n in (16, 32, 64, 128):
    row = [selection_accuracy(n)]
    for c in (8, 4, 2):
        row.append(selection_accuracy(n, cluster=c))
    clus[n] = row
    print(f"{n:>8}{row[0]:>12.1%}{row[1]:>13.1%}{row[2]:>13.1%}"
          f"{row[3]:>13.1%}")

print()
print()
print("Where does tool COUNT start to matter? Sweep how precisely a request")
print("locates its tool, on a distinct (non-overlapping) inventory.")
print()
print(f"{'noise':>8}" + "".join(f"{str(n) + ' tools':>12}" for n in (8, 32, 128)))
print("-" * 44)
SIG_SAVE = SIGMA
noise_tab = {}
for sg in (0.55, 0.9, 1.3, 1.8, 2.5):
    SIGMA = sg
    row = [selection_accuracy(n) for n in (8, 32, 128)]
    noise_tab[sg] = row
    print(f"{sg:>8.2f}" + "".join(f"{v:>12.1%}" for v in row))
SIGMA = SIG_SAVE

print()
print()
print("Four ways to spend an engineering week, on a realistic inventory:")
print("64 tools that have grown into 2 overlapping families.")
print()
print(f"{'intervention':>40}{'selection':>12}{'end to end':>13}")
print("-" * 65)
base_sel = selection_accuracy(64, cluster=2)
interv = {}


def price(name, sel, na, pa):
    e = sel * P_WHEN * (pa ** na) * P_READ
    interv[name] = (sel, e)
    print(f"{name:>40}{sel:>12.1%}{e:>13.1%}")


price("baseline (64 tools, 2 families)", base_sel, N_ARGS, P_ARGS_BASE)
price("halve the inventory, same overlap", selection_accuracy(32, cluster=2),
      N_ARGS, P_ARGS_BASE)
price("split into 8 families, same count", selection_accuracy(64, cluster=8),
      N_ARGS, P_ARGS_BASE)
price("make all 64 genuinely distinct", selection_accuracy(64), N_ARGS,
      P_ARGS_BASE)
price("drop one required argument", base_sel, N_ARGS - 1, P_ARGS_BASE)
price("constrain args to enums (98.5->99.8%)", base_sel, N_ARGS, 0.998)

s4, s128 = stages[4][0], stages[128][0]
e4, e128 = stages[4][4], stages[128][4]
print(f"""
The first table is the decomposition, and the column that was supposed to move
does not.

Selection is {s4:.1%} at {4} tools and {s128:.1%} at {128}. That was not the
expected result, and the reason for it is worth more than the result I was
looking for: in {D} semantic dimensions, randomly placed points are
overwhelmingly likely to be far apart, so adding tools does not crowd the space.
Distinguishing a request's tool from {127} unrelated alternatives is barely
harder than distinguishing it from three.

**Tool count, on its own, is nearly free.** The advice to "keep the tool list
short" is not wrong for context-window reasons, and it is wrong for the reason it
is usually given.

The second table follows from the first: with selection contributing nothing,
failures are dominated by argument construction at
{(1 - p_args) / ((1 - s4) + (1 - P_WHEN) + (1 - p_args) + (1 - P_READ)):.1%} of
the total, then the when-to-call decision at
{(1 - P_WHEN) / ((1 - s4) + (1 - P_WHEN) + (1 - p_args) + (1 - P_READ)):.1%}, then
reading the result. **Three arguments at {P_ARGS_BASE:.1%} each cost more than
choosing among {128} tools.**

The third table is where selection actually breaks, and it is the mechanism the
first table ruled out being replaced by the one that matters.

Real inventories do not grow by adding unrelated tools. They grow by accretion:
`search_docs`, `search_tickets`, `search_code`, `search_docs_v2`. Forcing {64}
tools into {2} families takes selection from {clus[64][0]:.1%} to
{clus[64][3]:.1%}, and {128} tools into {2} families gives
{clus[128][3]:.1%}.

Compare the two effects at the same inventory size. Quadrupling from {16} to
{64} distinct tools costs {clus[16][0] - clus[64][0]:.1%}. Taking {64} tools from
distinct to two families costs {clus[64][0] - clus[64][3]:.1%}.

**Tool count is not the variable. Tool DISTINCTNESS is**, and the two are
routinely confused because they correlate in practice -- inventories that got
large got large by accretion. The metric to track is not the size of the list but
how separable the descriptions are, which you can measure directly by embedding
them and looking at nearest-neighbour distances.

The fourth table says where the boundary is, because "count does not matter" is
only true while the space is roomy. At noise {0.55} selection is
{noise_tab[0.55][2]:.1%} even at {128} tools. At noise {1.8} it is
{noise_tab[1.8][0]:.1%} at {8} tools and {noise_tab[1.8][2]:.1%} at {128}.

So count matters exactly when the request does NOT locate its tool precisely, and
the two failure modes -- vague requests and overlapping tools -- are the same
failure mode seen from opposite ends. Both are distances in the same space.

The last table prices the interventions on a realistic inventory: {64} tools in
{2} families, which is what an inventory looks like after two years.

Halving the inventory while keeping the overlap buys
{interv['halve the inventory, same overlap'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}
end to end. Splitting the same {64} tools into {8} distinguishable families buys
{interv['split into 8 families, same count'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.
Making all {64} genuinely distinct buys
{interv['make all 64 genuinely distinct'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.

Against those, dropping one required argument buys
{interv['drop one required argument'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}
and constraining the arguments to enumerated values buys
{interv['constrain args to enums (98.5->99.8%)'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.

The ordering is the practical output. **Deduplicating an overlapping inventory is
worth several times what shortening it is, and both are worth more than argument
work once overlap is present** -- but with overlap removed, argument construction
is the whole remaining budget, which is the second table's finding.

One thing this listing does not model, and it is the largest omission. Selection
here is one decision. In an agent the model selects repeatedly, so selection
error compounds over the run in the way ch:rsn-cot's
eq:chain-accuracy-compounds describes: an inventory that costs
{clus[64][3]:.1%} per selection costs {clus[64][3] ** 5:.1%} over five steps
against {clus[64][0] ** 5:.1%} for a distinct one. **The overlap penalty is
raised to the power of the horizon**, which is why it matters far more in an
agent than in a single function call, and why an inventory that is fine for a
chat assistant can be unusable for an agent.""")
