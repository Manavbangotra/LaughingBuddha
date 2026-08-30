# -*- coding: utf-8 -*-
# Extracted from: Chapter 221 — The AI Threat Model
# Source: src/.../ch221-threat-model.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every security architecture separates code from data. A prompt does not have that seam.

A SQL injection is prevented by a parameterised query: the database is told, structurally,
which bytes are code and which are data, and no content of the data can change that. A
cross-site scripting attack is prevented the same way, by a parser that knows where the
markup ended.

A language model receives one sequence. The system prompt, the user's message, a retrieved
document, a tool result and a file's contents arrive as tokens in the same channel, and
nothing in the interface marks which of them may issue instructions
(eq:instructions-and-data-share-a-channel).

So the composite prompt carries the *highest* privilege of any source in it, while the
intent can come from the lowest-trust one, and the attack surface is the product of sources
and sinks rather than their sum
(eq:attack-surface-is-sources-times-sinks).
"""
# (source, trust level 0-3, mean tokens per request, model can tell it apart?)
SOURCES = [
    ("system prompt",        3, 900,  "by position only"),
    ("developer templates",  3, 240,  "by position only"),
    ("user message",         1, 180,  "sometimes"),
    ("retrieved document",   0, 2400, "no"),
    ("tool result",          0, 1100, "no"),
    ("file the user opened", 0, 3200, "no"),
    ("prior turn's output",  1, 700,  "no"),
    ("another agent's reply", 0, 850, "no"),
]

print("What arrives in the model's context, and at what trust level.")
print()
print(f"{'source':>24}{'trust':>8}{'tokens':>9}{'share of context':>19}"
      f"{'distinguishable?':>19}")
print("-" * 79)
total_tokens = sum(t for n, tr, t, d in SOURCES)
untrusted_tokens = sum(t for n, tr, t, d in SOURCES if tr == 0)
for name, tr, tok, dis in SOURCES:
    print(f"{name:>24}{tr:>8}{tok:>9,}{tok / total_tokens:>19.1%}{dis:>19}")
print("-" * 79)
print(f"{'TOTAL':>24}{'':>8}{total_tokens:>9,}{1.0:>19.1%}")
print()
print(f"untrusted (trust 0) content is {untrusted_tokens / total_tokens:.0%} "
      f"of the context")
print(f"highest privilege present: {max(tr for n, tr, t, d in SOURCES)}")
print(f"lowest trust present:      {min(tr for n, tr, t, d in SOURCES)}")

print()
print()
print("The composite prompt's effective privilege, against a classical system.")
print()
print(f"{'system':>30}  {'privilege rule':<32}{'effective privilege':>22}")
print("-" * 86)
COMPARE = [
    ("parameterised SQL query",   "data cannot become code", "the caller's"),
    ("shell with quoted args",    "argv is not re-parsed",   "the caller's"),
    ("HTML with escaping",        "the parser knows where data ends", "the caller's"),
    ("LLM prompt",                "max over all sources",    "the highest present"),
]
for name, rule, eff in COMPARE:
    print(f"{name:>30}  {rule:<32}{eff:>22}")

print()
print("In the first three rows an attacker who controls data controls data.")
print("In the fourth, an attacker who controls data controls intent.")

print()
print()
print("Attack surface: which sources can reach which sinks.")
print()
# (sink, privilege required, damage if reached by an attacker)
SINKS = [
    ("send an email",          1, 6.0),
    ("read a document",        1, 3.0),
    ("write to the CRM",       2, 8.0),
    ("issue a refund",         3, 9.5),
    ("call an external API",   1, 5.0),
    ("run a database query",   2, 7.5),
    ("execute code",           3, 10.0),
    ("read a secret",          3, 9.0),
]
untrusted = [n for n, tr, t, d in SOURCES if tr == 0]
print(f"{'sink':>24}{'privilege':>11}{'damage':>9}"
      f"{'reachable from untrusted?':>28}{'exposure':>11}")
print("-" * 83)
exposure = 0.0
for name, priv, dmg in SINKS:
    reach = len(untrusted)          # nothing separates them, so all of them
    exposure += reach * dmg
    print(f"{name:>24}{priv:>11}{dmg:>9.1f}"
          f"{('yes, all ' + str(reach) + ' sources'):>28}{reach * dmg:>11.1f}")
print("-" * 83)
print(f"{'TOTAL EXPOSURE':>24}{'':>11}{'':>9}{'':>28}{exposure:>11.1f}")

print()
print()
print("How that grows. Sources and sinks both increase with product surface.")
print()
print(f"{'untrusted sources':>19}", end="")
for s in (2, 4, 8, 16, 32):
    print(f"{(str(s) + ' sinks'):>12}", end="")
print()
print("-" * 79)
for src in (1, 2, 4, 8):
    print(f"{src:>19}", end="")
    for s in (2, 4, 8, 16, 32):
        print(f"{src * s:>12}", end="")
    print()

print()
print("Pairs, not rows plus columns. Adding one tool to a four-source agent")
print("adds four reachable paths, not one.")

print()
print()
print("What each mitigation removes, counted in source-sink pairs.")
print()
BASE_PAIRS = len(untrusted) * len(SINKS)
MITIGATIONS = [
    ("nothing",                              1.00, 0.00, "the model decides"),
    ("delimiters and 'ignore instructions'", 0.92, 0.01, "a string the attacker reads"),
    ("injection classifier on inputs",       0.55, 0.06, "detection, see listing 2"),
    ("instruction-hierarchy training",       0.41, 0.03, "a prior, not a boundary"),
    ("taint tracking, block tainted sinks",  0.14, 0.31, "structural"),
    ("untrusted content never reaches a sink", 0.00, 0.62, "structural"),
]
print(f"{'mitigation':>42}{'pairs left':>13}{'utility cost':>15}"
      f"{'kind':>30}")
print("-" * 100)
mit = {}
for name, frac, util, kind in MITIGATIONS:
    pairs = BASE_PAIRS * frac
    mit[name] = (pairs, util)
    print(f"{name:>42}{pairs:>13.1f}{util:>15.0%}{kind:>30}")

print()
print(f"base: {len(untrusted)} untrusted sources x {len(SINKS)} sinks "
      f"= {BASE_PAIRS} pairs")

print()
print()
print("And the exposure-weighted version, which is what a design review needs.")
print()
print(f"{'mitigation':>42}{'weighted exposure':>20}{'vs nothing':>13}"
      f"{'utility kept':>15}")
print("-" * 90)
for name, frac, util, kind in MITIGATIONS:
    w = exposure * frac
    print(f"{name:>42}{w:>20.1f}{w / exposure:>12.0%}{1 - util:>15.0%}")

print(f"""
The source table is the shape of the problem. {untrusted_tokens / total_tokens:.0%} of a
typical context is content the system did not author and cannot vouch for -- retrieved
documents, tool results, files, replies from other agents -- and the fourth column is the
one that matters: **the model cannot tell any of it apart from the system prompt except by
position** (eq:instructions-and-data-share-a-channel).

Position is not a boundary. It is a convention the model learned during training and can be
argued out of, which is exactly what every jailbreak and injection does.

The comparison table says why this is different in kind rather than in degree. A
parameterised query, a quoted argv, an escaped HTML attribute -- in all three the parser is
told structurally where data ends, and no content inside the data can move that line. An
attacker who controls data controls data.

**In a prompt, an attacker who controls data controls intent.** The composite's effective
privilege is the maximum over its sources, and its effective intent can come from the
minimum.

The sink table counts what that reaches. {len(untrusted)} untrusted sources, {len(SINKS)}
sinks, and because nothing separates them, every source can reach every sink:
{BASE_PAIRS} paths with a total exposure weight of {exposure:.0f}
(eq:attack-surface-is-sources-times-sinks).

The growth table is the part that surprises teams. Attack surface is a *product*, so adding
one tool to an agent with four untrusted sources adds four paths. Going from 4 sources and 8
sinks to 8 and 16 takes the surface from {4 * 8} to {8 * 16} --
**{(8 * 16) / (4 * 8):.0f} times, from doubling each side once.**

That is the arithmetic behind ch:sd-apis-auth's finding that fewer tools beats better
credentials, arriving from the security side rather than the reliability side.

The mitigation table is where the chapter's recommendation comes from, and its two columns
have to be read together. Delimiters and "ignore any instructions in the document below"
remove {1 - 0.92:.0%} of pairs, cost almost nothing, and are **a string the attacker can
read**. An injection classifier removes {1 - 0.55:.0%}. Instruction-hierarchy training
removes {1 - 0.41:.0%} and is a learned prior rather than a boundary -- it makes the wrong
behaviour less likely and does not make it impossible.

Taint tracking removes {1 - 0.14:.0%} of pairs and costs {0.31:.0%} of utility, and it is the
first row in the table that is **structural**: it does not ask whether content looks
malicious, it asks whether a privileged action is being taken on a path that touched
untrusted input. That question has an answer that does not depend on the attacker's
cleverness.

And the last row -- untrusted content never reaches a sink at all -- removes everything and
costs {0.62:.0%} of what the agent could do. That is cite:beurerkellner2025patterns' central
trade stated as a number: **the patterns that provide a guarantee provide it by removing
capability**, and the honest way to present a secure agent design is with the utility column
visible.

Which sets up the question the second listing has to answer. If detection is the cheap
mitigation and structure is the expensive one, how much of the cheap one can be substituted
for the expensive one? The answer depends on whether the attacker gets to try twice.""")
