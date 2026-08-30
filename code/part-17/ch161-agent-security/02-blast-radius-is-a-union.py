# -*- coding: utf-8 -*-
# Extracted from: Chapter 161 — Agent Security and Excessive Agency
# Source: src/.../ch161-agent-security.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An agent's blast radius is the union of its tools' capabilities, not the maximum.

Tools are reviewed one at a time. Each one is asked: what is the worst this can do?
And each one, individually, is usually fine -- a search tool reads, an email tool
sends, a file tool writes to a scratch directory.

The dangerous things an agent can do are mostly COMBINATIONS. Read a private
record and send an email: exfiltration, from two tools that are individually
harmless. Read a config and write a file: persistence. Neither capability appears
in either tool's review (eq:blast-radius-is-a-union).

This listing counts how the number of reachable dangerous combinations grows as an
inventory grows, and compares three review policies against it.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(2687)

# Capability tags a tool can carry.
CAPS = ["read_private", "read_public", "write_internal", "write_external",
        "delete", "execute", "spend", "notify"]
NC = len(CAPS)

# Pairs of capabilities that compose into something neither has alone.
DANGEROUS_PAIRS = {
    ("read_private", "write_external"): "exfiltration",
    ("read_private", "notify"): "exfiltration",
    ("execute", "write_internal"): "persistence",
    ("read_public", "execute"): "remote code path",
    ("delete", "write_external"): "destructive publish",
    ("spend", "notify"): "unattended purchase",
    ("read_private", "spend"): "targeted fraud",
    ("execute", "spend"): "automated abuse",
}
SINGLE_DANGEROUS = {"delete", "spend", "execute"}


def make_inventory(n_tools, caps_per_tool=2):
    """Each tool carries a small number of capabilities."""
    inv = []
    for _ in range(n_tools):
        k = max(1, rng.poisson(caps_per_tool - 1) + 1)
        inv.append(set(rng.choice(CAPS, size=min(k, NC), replace=False)))
    return inv


def reachable(inv):
    """Capabilities the agent has at all, and dangerous pairs it can compose."""
    have = set().union(*inv) if inv else set()
    pairs = {v for (a, b), v in DANGEROUS_PAIRS.items()
             if a in have and b in have}
    singles = SINGLE_DANGEROUS & have
    return have, pairs, singles


TRIALS = 400
SIZES = [2, 4, 8, 16, 32]

print(f"{NC} capability tags, {len(DANGEROUS_PAIRS)} dangerous PAIRS that compose")
print(f"from two harmless-looking capabilities, {len(SINGLE_DANGEROUS)} that are")
print("dangerous on their own. Tools carry about two capabilities each.")
print()
print(f"{'tools':>7}{'capabilities held':>19}{'single-tool risks':>19}"
      f"{'composed risks':>17}")
print("-" * 62)
grow = {}
for n in SIZES:
    caps_n = pairs_n = singles_n = 0.0
    for _ in range(TRIALS):
        inv = make_inventory(n)
        h, p, s = reachable(inv)
        caps_n += len(h); pairs_n += len(p); singles_n += len(s)
    grow[n] = (caps_n / TRIALS, singles_n / TRIALS, pairs_n / TRIALS)
    print(f"{n:>7}{grow[n][0]:>19.1f}{grow[n][1]:>19.1f}{grow[n][2]:>17.1f}")

print()
print()
print("What per-tool review sees, against what the agent can actually do.")
print("A per-tool review flags a tool only if it is dangerous BY ITSELF.")
print()
print(f"{'tools':>7}{'tools flagged':>16}{'risks a per-tool':>19}"
      f"{'risks actually':>17}{'missed':>9}")
print(f"{'':>7}{'individually':>16}{'review would find':>19}"
      f"{'reachable':>17}{'':>9}")
print("-" * 68)
miss = {}
for n in SIZES:
    flagged = seen = actual = 0.0
    for _ in range(TRIALS):
        inv = make_inventory(n)
        h, p, s = reachable(inv)
        flagged += sum(1 for t in inv if t & SINGLE_DANGEROUS)
        seen += len(s)
        actual += len(s) + len(p)
    miss[n] = (flagged / TRIALS, seen / TRIALS, actual / TRIALS)
    v = miss[n]
    print(f"{n:>7}{v[0]:>16.1f}{v[1]:>19.1f}{v[2]:>17.1f}"
          f"{v[2] - v[1]:>9.1f}")

print()
print()
print("Three ways to reduce the blast radius of a 16-tool inventory, each")
print("removing the same number of tools.")
print()
REMOVE = 4
print(f"{'policy':>34}{'capabilities':>14}{'composed risks':>17}"
      f"{'total risks':>13}")
print("-" * 78)
pol = {}
for name in ["remove none", "remove 4 at random",
             "remove the 4 most capable tools",
             "remove 4 that break the most pairs"]:
    caps_n = pairs_n = tot = 0.0
    for _ in range(TRIALS):
        inv = make_inventory(16)
        if name == "remove 4 at random":
            keep = list(rng.permutation(len(inv))[REMOVE:])
            inv = [inv[i] for i in keep]
        elif name == "remove the 4 most capable tools":
            order = np.argsort([-len(t) for t in inv])
            inv = [inv[i] for i in order[REMOVE:]]
        elif name == "remove 4 that break the most pairs":
            for _ in range(REMOVE):
                best, bestv = None, -1
                _, cur, _ = reachable(inv)
                for i in range(len(inv)):
                    trial = inv[:i] + inv[i + 1:]
                    _, p2, s2 = reachable(trial)
                    gain = (len(cur) - len(p2))
                    if gain > bestv:
                        best, bestv = i, gain
                inv = inv[:best] + inv[best + 1:]
        h, p, s = reachable(inv)
        caps_n += len(h); pairs_n += len(p); tot += len(p) + len(s)
    pol[name] = (caps_n / TRIALS, pairs_n / TRIALS, tot / TRIALS)
    v = pol[name]
    print(f"{name:>34}{v[0]:>14.1f}{v[1]:>17.1f}{v[2]:>13.1f}")

print()
print()
print("And what splitting one agent into two does. A composed risk counts if")
print("EITHER agent can reach it -- splitting helps only if it breaks the pair.")
print()
print(f"{'arrangement':>34}{'composed risks':>17}{'tools each':>13}")
print("-" * 64)
split = {}
for name in ["one agent, 16 tools", "two agents, random split",
             "two agents, split to break pairs",
             "two agents, disjoint CAPABILITIES"]:
    v = 0.0
    for _ in range(TRIALS):
        inv = make_inventory(16)
        if name == "one agent, 16 tools":
            _, p, _ = reachable(inv)
            v += len(p)
        elif name == "two agents, random split":
            idx = rng.permutation(16)
            a = [inv[i] for i in idx[:8]]
            b = [inv[i] for i in idx[8:]]
            v += len(reachable(a)[1] | reachable(b)[1])
        elif name == "two agents, split to break pairs":
            # Put read_private on one side and everything else on the other.
            a = [t for t in inv if "read_private" in t]
            b = [t for t in inv if "read_private" not in t]
            v += len(reachable(a)[1] | reachable(b)[1])
        else:
            # Partition the CAPABILITIES, not the tools: a reader agent that
            # cannot act, and an actor agent that cannot read anything private.
            READ = {"read_private", "read_public"}
            a = [t & READ for t in inv if t & READ]
            b = [t - READ for t in inv if t - READ]
            a = [t for t in a if t]
            b = [t for t in b if t]
            v += len(reachable(a)[1] | reachable(b)[1])
    split[name] = v / TRIALS
    n_each = 16 if name == "one agent, 16 tools" else 8
    print(f"{name:>34}{split[name]:>17.1f}{n_each:>13}")

print(f"""
The first table is the growth, and the two right-hand columns grow at different
rates for different reasons.

Single-tool risks saturate almost immediately: {grow[2][1]:.1f} at two tools,
{grow[16][1]:.1f} at sixteen, out of {len(SINGLE_DANGEROUS)} possible. Composed
risks go {grow[2][2]:.1f} to {grow[16][2]:.1f} out of {len(DANGEROUS_PAIRS)}.

And the capabilities column explains both: an agent with {16} tools holds
{grow[16][0]:.1f} of the {NC} capability tags. **Past about eight tools it holds
essentially everything**, so it can compose essentially every pair.

That saturation is the structural fact of this listing. An inventory does not need
to be large before it is complete, and completeness is what makes composition
available.

The second table is what a per-tool review sees against what the agent can do. At
{16} tools a per-tool review flags {miss[16][0]:.1f} tools as individually
dangerous and identifies {miss[16][1]:.1f} risks. The agent can actually reach
{miss[16][2]:.1f}.

**It misses {miss[16][2] - miss[16][1]:.1f} of them -- roughly
{(miss[16][2] - miss[16][1]) / miss[16][2]:.0%} of the real risk surface** -- and
it misses them for a reason no amount of care fixes: the risks are not properties
of any tool. Exfiltration is not in the search tool and not in the email tool. It
is in the pair (eq:blast-radius-is-a-union).

Note also that the flagged column keeps growing ({miss[32][0]:.1f} at {32} tools)
while what it finds does not ({miss[32][1]:.1f}). Per-tool review generates more
work and no more coverage as the inventory grows, which is the worst possible
scaling for a manual process.

The third table asks whether you can trim your way out, and the answer is barely.
Removing four of sixteen tools -- a quarter of the inventory -- takes composed risks
from {pol['remove none'][1]:.1f} to {pol['remove 4 at random'][1]:.1f} at random,
and to {pol['remove 4 that break the most pairs'][1]:.1f} with a greedy search that
explicitly targets pair-breaking.

**A quarter of the inventory removed buys about {(pol['remove none'][1] - pol['remove 4 that break the most pairs'][1]) / pol['remove none'][1]:.0%} of the composed risk.** The capabilities column says why:
even after removing four tools the agent still holds
{pol['remove 4 that break the most pairs'][0]:.1f} of {NC} capabilities, because
capabilities are duplicated across tools. You are removing redundancy, not reach.

The fourth table is the intervention that works, and the first two rows are the
trap.

Splitting one agent into two with eight tools each changes composed risks from
{split['one agent, 16 tools']:.1f} to {split['two agents, random split']:.1f}.
Nothing. Each half still holds nearly every capability, so each half can compose
nearly every pair, and a risk reachable by either agent is still reachable.

Splitting deliberately to separate `read_private` gets
{split['two agents, split to break pairs']:.1f} -- a real but small improvement,
because it separates one capability and the tools carrying it also carry others.

Partitioning the CAPABILITIES rather than the tools -- a reader agent that cannot
act and an actor agent that cannot read anything private -- gets
{split['two agents, disjoint CAPABILITIES']:.1f}, a reduction of
{(split['one agent, 16 tools'] - split['two agents, disjoint CAPABILITIES']) / split['one agent, 16 tools']:.0%}.

**Partitioning tools does nothing; partitioning capabilities is the whole
mechanism.** That distinction is the practical output of this listing, and it is
easy to get wrong because "split the agent in two" sounds like it should help and
is usually implemented by dividing the tool list.

Three rules follow.

**Review capability pairs, not tools.** The unit of risk is the pair, the pair
appears in no tool's documentation, and per-tool review scales its cost without
scaling its coverage.

**Expect saturation early.** Eight tools is enough to hold every capability in this
model. Any argument of the form "we only gave it a few tools" should be checked
against the capability union rather than the count.

**If you partition, partition capabilities.** A reader that cannot write and a
writer that cannot read private data have a genuinely smaller joint blast radius.
Two agents with half the tools each have the same one.

And the connection back to the previous listing: the reason this matters is that
injection cannot be reliably prevented -- cite:greshake2023indirect's abstract says
so and the detector sweep confirmed the cost of pretending otherwise -- so the
composed capabilities are what a landed injection gets to use. **The blast radius is not what your
agent does. It is what an attacker who controls your agent for one turn can
reach**, and that is the union of everything on the list.""")
