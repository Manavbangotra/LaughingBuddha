# -*- coding: utf-8 -*-
# Extracted from: Chapter 174 — Authentication, Authorization, and MCP Security
# Source: src/.../ch174-security.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Tool poisoning, which is the vulnerability the measurements actually find.

cite:huang2026mcpthreat modelled MCP with STRIDE and DREAD across five components
and evaluated seven major clients. Its finding: tool poisoning -- malicious
instructions embedded in tool METADATA -- is the most prevalent and impactful
client-side vulnerability, attributed to insufficient static validation and
parameter visibility.

The mechanism is structural rather than a bug. A tool's description is text that
reaches the model, so it is an instruction whether or not it was written as one.
cite:mcp2026spec says as much: tool annotations "should be considered untrusted,
unless obtained from a trusted server."

There is a second problem the first hides. Descriptions are read at DISCOVERY and
tools are invoked later, so a server can be benign when approved and malicious
afterwards (eq:approval-is-a-snapshot). This listing measures both, and compares
the defences cite:huang2026mcpthreat proposes against ch:ag-security's structural
one.
"""
import numpy as np

rng = np.random.default_rng(4327)

M = 60000
CALLS = 40              # tool calls in the period being modelled
P_POISON = 0.03         # share of tools carrying a hostile instruction
P_OBEY = 0.62           # chance the model acts on an injected instruction


def run(defence, m=M, calls=CALLS, p_poison=P_POISON, p_obey=P_OBEY,
        scan_detect=0.55, vis_detect=0.70, vis_attention=0.45,
        partition_cover=0.80):
    """Harmful actions per deployment under one defence.

    none        the description reaches the model unmodified
    scan        static analysis of tool metadata before presentation
    visibility  the user is shown the actual arguments before execution
    partition   ch:ag-security's split: the poisoned tool cannot reach anything
                worth reaching, so obeying it accomplishes nothing
    both        scan + visibility
    all         scan + visibility + partition
    """
    poisoned = rng.random((m, calls)) < p_poison
    fires = poisoned & (rng.random((m, calls)) < p_obey)
    if defence in ("scan", "both", "all"):
        fires &= rng.random((m, calls)) >= scan_detect
    if defence in ("visibility", "both", "all"):
        # A visible argument is only a defence if the user reads it, and
        # ch:ag-termination's habituation says attention is finite.
        caught = (rng.random((m, calls)) < vis_detect) & \
                 (rng.random((m, calls)) < vis_attention)
        fires &= ~caught
    if defence in ("partition", "all"):
        fires &= rng.random((m, calls)) >= partition_cover
    return float(fires.sum(1).mean()), float((fires.sum(1) > 0).mean())


print(f"{M:,} deployments, {CALLS} tool calls each. {P_POISON:.0%} of tools")
print(f"carry a hostile instruction in their metadata; the model acts on one")
print(f"{P_OBEY:.0%} of the time it is present.")
print()
print(f"{'defence':>26}{'harmful actions':>17}{'any harm':>11}{'reduction':>11}")
print("-" * 65)
tab = {}
base = run("none")
for name, label in (("none", "none"), ("scan", "static metadata scan"),
                    ("visibility", "parameter visibility"),
                    ("both", "scan + visibility"),
                    ("partition", "capability partition"),
                    ("all", "all three")):
    r = run(name)
    tab[label] = r
    print(f"{label:>26}{r[0]:>17.3f}{r[1]:>11.1%}"
          f"{1 - r[0] / base[0]:>11.0%}")

print()
print()
print("Parameter visibility depends on the user reading what is shown, and")
print("ch:ag-termination measured what happens to attention under volume.")
print()
print(f"{'user attention':>16}{'harmful actions':>17}{'reduction':>11}")
print("-" * 44)
at = {}
for a in (0.95, 0.60, 0.30, 0.10):
    r = run("visibility", vis_attention=a)
    at[a] = r
    print(f"{a:>16.0%}{r[0]:>17.3f}{1 - r[0] / base[0]:>11.0%}")

print()
print()
print("And how each defence holds as the poisoned share rises, which is what")
print("happens as an ecosystem grows and its registry admits more publishers.")
print()
print(f"{'poisoned tools':>16}{'none':>9}{'scan+vis':>11}{'partition':>12}"
      f"{'all three':>12}")
print("-" * 60)
ps = {}
for p in (0.005, 0.03, 0.10, 0.30):
    row = tuple(run(d, p_poison=p)[0]
                for d in ("none", "both", "partition", "all"))
    ps[p] = row
    print(f"{p:>16.1%}{row[0]:>9.3f}{row[1]:>11.3f}{row[2]:>12.3f}"
          f"{row[3]:>12.3f}")

print()
print()
print("The second problem: a description is read when the tool is APPROVED and")
print("acted on every time it is called. A server can change it in between.")
print()


def rugpull(recheck_every, m=M, calls=CALLS, p_turn=0.004, p_obey=P_OBEY,
            defence_cover=0.55):
    """A benign server turns hostile at some call and stays hostile until a
    re-verification of its metadata catches it. Walk the calls directly rather
    than trying to be clever about the bookkeeping."""
    hostile = np.zeros(m, dtype=bool)
    retired = np.zeros(m, dtype=bool)
    fires = np.zeros(m, dtype=np.int64)
    for t in range(calls):
        # A still-trusted server may turn at this call.
        turning = (~retired) & (~hostile) & (rng.random(m) < p_turn)
        hostile |= turning
        # A re-verification happens before the call is issued.
        if recheck_every and (t % recheck_every == 0):
            caught = hostile & (rng.random(m) < defence_cover)
            retired |= caught
            hostile &= ~caught
        live = hostile & ~retired
        fires += live & (rng.random(m) < p_obey)
    return float(fires.mean()), float((fires > 0).mean())


print(f"{'re-verify every':>17}{'harmful actions':>17}{'any harm':>11}")
print("-" * 45)
rp = {}
for k in (0, 20, 10, 5, 1):
    r = rugpull(k)
    rp[k] = r
    label = "never" if k == 0 else f"{k} calls"
    print(f"{label:>17}{r[0]:>17.3f}{r[1]:>11.1%}")

print(f"""
The first table reproduces ch:ag-security's central result in a new setting, and
the ordering is the one that chapter predicted.

Static metadata scanning -- cite:huang2026mcpthreat's first mitigation layer --
removes {1 - tab['static metadata scan'][0] / base[0]:.0%} of harmful actions.
Parameter visibility removes {1 - tab['parameter visibility'][0] / base[0]:.0%}.
Together they remove {1 - tab['scan + visibility'][0] / base[0]:.0%}.

The capability partition alone removes {1 - tab['capability partition'][0] / base[0]:.0%}
-- **more than both detection defences combined**.

The mechanism is the one ch:ag-security identified. A detector asks "is this
instruction hostile", which is a hard classification with an irreducible error
rate. A partition asks nothing: it arranges that a tool which reads untrusted
content cannot reach anything worth reaching, so obeying an injected instruction
accomplishes nothing. **You cannot classify your way out of a problem you can
structure your way out of.**

That is not an argument for skipping the detectors -- all three together reach
{1 - tab['all three'][0] / base[0]:.0%}, which is much better than any one. It is
an argument about ORDER. Build the partition first, because it is the layer whose
effectiveness does not depend on being right about what an attacker will write.

The second table is why parameter visibility is the weakest of the three, and it
is not a criticism of the mechanism. At {0.95:.0%} user attention it removes
{1 - at[0.95][0] / base[0]:.0%}; at {0.10:.0%} it removes
{1 - at[0.10][0] / base[0]:.0%}.

Showing the user the actual arguments is exactly right, and it inherits
ch:ag-termination's habituation: a user shown forty argument lists per session is
not reading the fortieth. **A defence whose effectiveness is a function of human
attention degrades with the volume it is deployed at**, which is precisely the
regime an agent creates.

The third table matters for ecosystem design rather than for a single deployment.
As the poisoned share rises from {0.005:.1%} to {0.30:.0%}, every defence degrades
proportionally -- none of them has a threshold. So a registry that admits more
publishers moves every connected host along this table at once, and the host
cannot tell that it happened.

That is the argument for cite:hou2025mcp's framing: most of what determines this
number is decided in the server LIFECYCLE -- who may publish, what is verified at
install, what provenance survives -- rather than at the protocol layer, and none
of it is visible from inside a session.

The last table is the problem the whole discussion usually omits. A description is
read when a tool is approved and acted on every time it is called, so approval is
a snapshot of a mutable thing (eq:approval-is-a-snapshot).

Never re-verifying gives {rp[0][0]:.3f} harmful actions and
{rp[0][1]:.1%} of deployments harmed. Re-verifying every call gives
{rp[1][0]:.3f} and {rp[1][1]:.1%}.

**Re-validation is the cheapest intervention here as it was in
ch:as-long-running and ch:mcp-primitives** -- the third time in this book that
re-reading something you already read turns out to be the best available move. The
pattern is consistent enough to state as a rule: **anything approved once and used
many times needs re-approval on a schedule**, and the schedule should be tighter
than feels necessary.""")
