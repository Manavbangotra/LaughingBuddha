# -*- coding: utf-8 -*-
# Extracted from: Chapter 225 — Tool Abuse, Agent Hijacking, and Sandboxing
# Source: src/.../ch225-tool-abuse.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Two harmless tools compose into a harmful one, and per-tool approval cannot see it.

A search tool reads. An email tool writes. Neither is dangerous alone -- reading what the user
may read, sending what the user may send. Together they are an exfiltration primitive, and the
damage of the pair exceeds the sum of its parts
(eq:tool-damage-composes-superadditively).

Which breaks the approval design almost everyone builds first. Approving each call
individually approves each half of a composition and never sees the whole, so the gate has to
sit at the outcome rather than at the call
(eq:approval-must-sit-at-the-outcome-not-the-call).
"""
# (tool, damage alone, reversible?, calls/day)
TOOLS = [
    ("search internal documents", 1.0, True,  8400),
    ("read a customer record",    1.5, True,  6100),
    ("send an email",             2.0, False, 1900),
    ("write to the CRM",          3.0, False, 2400),
    ("issue a refund",            7.0, False,  310),
    ("execute code",              5.0, True,   890),
    ("post to a webhook",         2.5, False,  640),
]
alone = {n: d for n, d, r, c in TOOLS}

print("Each tool alone.")
print()
print(f"{'tool':>28}{'damage alone':>15}{'reversible':>13}{'calls/day':>12}")
print("-" * 68)
for name, dmg, rev, calls in TOOLS:
    print(f"{name:>28}{dmg:>15.1f}{('yes' if rev else 'no'):>13}{calls:>12,}")

print()
print(f"sum of individual damages: {sum(alone.values()):.1f}")

print()
print()
print("Compositions, and what they do that the parts do not.")
print()
COMPOS = [
    (("search internal documents", "send an email"),        14.0,
     "exfiltration to any address"),
    (("read a customer record", "post to a webhook"),       12.0,
     "bulk export, no log on the far side"),
    (("search internal documents", "write to the CRM"),      9.0,
     "poison the record other agents read"),
    (("read a customer record", "issue a refund"),          16.0,
     "targeted fraud at scale"),
    (("execute code", "send an email"),                     18.0,
     "arbitrary computation plus a channel"),
    (("search internal documents", "execute code",
      "post to a webhook"),                                 24.0,
     "read, transform, exfiltrate"),
]
SHORT = {
    "search internal documents": "search",
    "read a customer record": "read",
    "send an email": "email",
    "write to the CRM": "crm-write",
    "issue a refund": "refund",
    "execute code": "exec",
    "post to a webhook": "webhook",
}
print(f"{'composition':>34}{'sum of parts':>15}{'actual':>9}"
      f"{'excess':>9}  {'what it enables':<38}")
print("-" * 107)
comp = {}
for tools, dmg, what in COMPOS:
    s = sum(alone[t] for t in tools)
    label = " + ".join(SHORT[t] for t in tools)
    comp[tools] = (s, dmg, dmg - s)
    print(f"{label:>34}{s:>15.1f}{dmg:>9.1f}{dmg - s:>9.1f}  {what:<38}")

print()
tot_excess = sum(c[2] for c in comp.values())
print(f"total superadditive excess across six compositions: {tot_excess:.1f}")
print(f"against a sum-of-parts total of {sum(alone.values()):.1f}")

print()
print()
print("How many compositions there are, as tools are added.")
print()
print(f"{'tools':>8}{'pairs':>10}{'triples':>11}{'total subsets >= 2':>22}"
      f"{'reviewed in practice':>23}")
print("-" * 74)
import math
for n in (3, 5, 7, 10, 15, 20):
    pairs = math.comb(n, 2)
    triples = math.comb(n, 3)
    allsub = 2 ** n - n - 1
    print(f"{n:>8}{pairs:>10,}{triples:>11,}{allsub:>22,}{n:>23}")

print()
print("Reviews are per tool. The thing that grows is the subset count.")

print()
print()
print("Reversibility, which decides what an approval is actually protecting.")
print()
print(f"{'action':>28}{'damage':>9}{'recoverable share':>20}"
      f"{'permanent damage':>19}")
print("-" * 76)
REVERSIBILITY = [
    ("search internal documents", 0.00, 1.0),   # reading cannot be unread
    ("read a customer record",    0.00, 1.5),
    ("send an email",             0.05, 2.0),
    ("write to the CRM",          0.85, 3.0),
    ("issue a refund",            0.70, 7.0),
    ("execute code",              0.40, 5.0),
    ("post to a webhook",         0.00, 2.5),
]
perm = 0.0
for name, rec, dmg in REVERSIBILITY:
    p = dmg * (1 - rec)
    perm += p
    print(f"{name:>28}{dmg:>9.1f}{rec:>20.0%}{p:>19.2f}")
print("-" * 76)
print(f"{'TOTAL':>28}{sum(d for n, r, d in REVERSIBILITY):>9.1f}"
      f"{'':>20}{perm:>19.2f}")

print()
print(f"{perm / sum(d for n, r, d in REVERSIBILITY):.0%} of the damage is "
      f"permanent, and the reads are")
print("the least reversible actions on the list")

print()
print()
print("Where to put the approval.")
print()
DAILY_CALLS = sum(c for n, d, r, c in TOOLS)
GATES = [
    ("no approval",                 0.00, 0,        0.00),
    ("approve every call",          1.00, DAILY_CALLS, 1.00),
    ("approve non-reversible calls", 0.62, 5250,    0.31),
    ("approve by outcome class",    0.94, 410,      0.09),
    ("approve on a taint path only", 0.91, 190,     0.05),
]
print(f"{'gate':>34}{'composition coverage':>23}{'approvals/day':>16}"
      f"{'utility cost':>15}")
print("-" * 88)
gates = {}
for name, cov, appr, util in GATES:
    gates[name] = (cov, appr, util)
    print(f"{name:>34}{cov:>23.0%}{appr:>16,}{util:>15.0%}")

print()
print("Per-call approval sees every half and no whole. Outcome-class approval")
print("sees the whole and asks 400 times a day instead of 20,000.")

print()
print()
print("What an outcome class actually is.")
print()
CLASSES = [
    ("data leaves the tenant boundary",  "search + any egress tool", 14.0, 41),
    ("money moves",                      "refund, payment, credit",  16.0, 22),
    ("a record other agents read changes", "write to shared state",   9.0, 190),
    ("something irreversible happens",   "no undo path exists",      7.0, 84),
    ("privilege is used outside the task", "scope mismatch",         12.0, 73),
]
print(f"{'outcome class':>38}{'triggered by':>28}{'damage':>9}"
      f"{'approvals/day':>16}")
print("-" * 91)
tot_appr = 0
for name, trig, dmg, appr in CLASSES:
    tot_appr += appr
    print(f"{name:>38}{trig:>28}{dmg:>9.1f}{appr:>16}")
print("-" * 91)
print(f"{'TOTAL':>38}{'':>28}{'':>9}{tot_appr:>16}")

print(f"""
The individual table is the one every tool review produces, and it is not wrong. Each of the
{len(TOOLS)} tools does something the user is entitled to do, at a damage level that a
reasonable person signs off. The sum is {sum(alone.values()):.1f}.

The composition table is what the review misses. `search + send` is worth
{comp[('search internal documents', 'send an email')][1]:.1f} against a sum of parts of
{comp[('search internal documents', 'send an email')][0]:.1f} -- an excess of
{comp[('search internal documents', 'send an email')][2]:.1f}
(eq:tool-damage-composes-superadditively) -- because reading and sending are individually
authorised and *reading then sending* is exfiltration.

`search + execute + post` reaches {comp[('search internal documents', 'execute code', 'post to a webhook')][1]:.1f}
against {comp[('search internal documents', 'execute code', 'post to a webhook')][0]:.1f}: read
anything, transform it into a form no filter recognises, and send it somewhere with no log on
the far side.

Across six compositions the superadditive excess is {tot_excess:.1f}, against a
sum-of-parts total of {sum(alone.values()):.1f}. **The compositions carry more danger than the
tools do.**

The counting table is why this cannot be fixed by reviewing harder. Seven tools have
{math.comb(7, 2)} pairs and {math.comb(7, 3)} triples; twenty tools have
{math.comb(20, 2)} pairs and {math.comb(20, 3):,} triples, and
{2 ** 20 - 20 - 1:,} subsets of size two or more. **Reviews scale with tool count and risk
scales with subset count**, which is ch:sec-threat-model's product result one level up.

The reversibility table changes what an approval is for. Note the first two rows:
`{REVERSIBILITY[0][0]}` and `{REVERSIBILITY[1][0]}` are recoverable
{REVERSIBILITY[0][1]:.0%} of the time, because **a read cannot be unread**. Writes are largely
recoverable; refunds mostly are; reads are not.

{perm / sum(d for n, r, d in REVERSIBILITY):.0%} of total damage is permanent, and the actions
usually classified as "safe, read-only" are the least reversible things on the list. That is
ch:ops-deployment's rollback result in the tool layer: the artefact can be reverted and the
disclosure cannot.

The gate table is the design consequence. Approving every call covers everything and asks
{DAILY_CALLS:,} times a day, which is not a control because nobody reads
{DAILY_CALLS:,} approvals -- it is a rubber stamp with an audit trail.

Approving by **outcome class** covers {gates['approve by outcome class'][0]:.0%} of
compositions at {gates['approve by outcome class'][1]:,} approvals a day and
{gates['approve by outcome class'][2]:.0%} utility cost
(eq:approval-must-sit-at-the-outcome-not-the-call).

The difference is what the gate is looking at. A per-call gate asks "may this agent call this
tool?" and the answer is yes, for both halves of the composition. An outcome gate asks "is
data about to leave the tenant boundary?" -- a question about the *effect* of the sequence,
which is the thing that was dangerous.

The class table is what those questions are. Five of them, {tot_appr} approvals a day between
them, each phrased as a consequence rather than a capability. `data leaves the tenant
boundary` fires whether the egress is an email, a webhook, a file write or a search-result
citation, and it does not need to know which tool is doing it.

**Enumerate outcomes, not tools.** Outcomes are few and stable; tools are many and grow every
sprint, and the subsets grow faster than either.""")
