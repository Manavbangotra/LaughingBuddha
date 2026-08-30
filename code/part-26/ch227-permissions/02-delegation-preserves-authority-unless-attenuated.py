# -*- coding: utf-8 -*-
# Extracted from: Chapter 227 — Permission Systems, Approval Flows, and Governance
# Source: src/.../ch227-permissions.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Delegation preserves authority by default, so a chain is as strong as its strongest link.

When a user calls an agent, which calls a sub-agent, which calls a tool server, each hop
passes some authority along. The default in every system that does not think about it is to
pass *all* of it, because attenuating requires a decision about what to remove and nobody made
one (eq:delegation-preserves-authority-unless-attenuated).

That inverts the usual security intuition. A chain of trust is as weak as its weakest link; a
chain of *authority* is as strong as its most privileged member, and the privilege flows
downhill to whatever is at the end.

The second half is what an audit needs to reconstruct the decision, and the answer is the
whole principal chain rather than the acting identity
(eq:audit-completeness-requires-the-principal-chain).
"""
# (hop, authority it holds alone, does it attenuate by default?)
HOPS = [
    ("the user",            0.08, "-"),
    ("the orchestrator",    0.94, "no"),
    ("a planning sub-agent", 0.94, "no"),
    ("a retrieval sub-agent", 0.94, "no"),
    ("an MCP tool server",  0.61, "no"),
    ("the backend API",     1.00, "-"),
]

print("Authority along a call chain, with and without attenuation.")
print()
print(f"{'hop':>24}{'own authority':>16}{'preserved':>12}{'attenuated':>13}"
      f"{'least-privilege':>18}")
print("-" * 83)
preserved, atten, lp = 0.08, 0.08, 0.08
rows = []
for name, own, att in HOPS:
    preserved = max(preserved, own) if name != "the user" else own
    atten = atten * 0.72 if name != "the user" else own
    lp = 0.08
    rows.append((name, own, preserved, atten, lp))
    print(f"{name:>24}{own:>16.2f}{preserved:>12.2f}{atten:>13.3f}"
          f"{lp:>18.2f}")

print()
print(f"the user holds {HOPS[0][1]:.2f} and the request executes with "
      f"{preserved:.2f}")
print(f"attenuating 28% a hop would end at {atten:.3f}")

print()
print()
print("What each hop can do that the previous one could not.")
print()
GAINS = [
    ("the user -> orchestrator",     0.08, 0.94, "the service account"),
    ("orchestrator -> planner",      0.94, 0.94, "nothing, and nothing removed"),
    ("planner -> retriever",         0.94, 0.94, "nothing, and nothing removed"),
    ("retriever -> tool server",     0.94, 0.61, "narrower, by accident"),
    ("tool server -> backend",       0.61, 1.00, "the server's own credential"),
]
print(f"{'transition':>30}{'before':>10}{'after':>9}{'delta':>9}"
      f"{'what changed':>32}")
print("-" * 90)
for name, a, b, why in GAINS:
    print(f"{name:>30}{a:>10.2f}{b:>9.2f}{b - a:>+9.2f}{why:>32}")

print()
print("Only one transition narrows authority, and it does so because the")
print("tool server happens to have less, not because anything attenuated.")

print()
print()
print("Permission models, and what each can express about an agent.")
print()
MODELS = [
    ("RBAC on the service account", "the agent's role", 0.11, 0.0, "no"),
    ("RBAC on the user",            "the user's role",  0.44, 1.0, "no"),
    ("ABAC with request attributes", "user + resource + action", 0.71, 2.5, "partly"),
    ("delegated token (on-behalf-of)", "user, via the agent", 0.88, 3.5, "yes"),
    ("capability per task",          "this task's resources", 0.97, 5.0, "yes"),
]
print(f"{'model':>34}{'principal':>28}{'expressiveness':>16}"
      f"{'effort':>9}{'attenuates?':>13}")
print("-" * 100)
mod = {}
for name, prin, expr, eff, att in MODELS:
    mod[name] = (expr, eff, att)
    print(f"{name:>34}{prin:>28}{expr:>16.2f}{eff:>9.1f}{att:>13}")

print()
print("Only the last two carry the user's identity to the far end, and only")
print("they can narrow on the way.")

print()
print()
print("Audit reconstruction: what a record must contain to answer 'why'.")
print()
FIELDS = [
    ("acting identity",            "who called the API",        0.98, 0.11),
    ("originating user",           "who asked",                 0.62, 0.31),
    ("the full principal chain",   "every hop, in order",       0.19, 0.44),
    ("the task the chain served",  "what it was for",           0.14, 0.29),
    ("the content that triggered it", "which input, verbatim",  0.09, 0.51),
    ("the authority actually used", "which scope was exercised", 0.07, 0.38),
]
print(f"{'field':>32}{'what it answers':>28}{'recorded in practice':>23}"
      f"{'share of questions it settles':>32}")
print("-" * 115)
cum_miss = 1.0
for name, what, recorded, settles in FIELDS:
    cum_miss *= (1 - settles)
    print(f"{name:>32}{what:>28}{recorded:>23.0%}{settles:>32.0%}")

print()
print(f"if all six were recorded, {1 - cum_miss:.0%} of audit questions are")
print("answerable; in practice the top two carry most deployments")

top2 = 1 - (1 - FIELDS[0][3]) * (1 - FIELDS[1][3])
print(f"acting identity plus originating user alone: {top2:.0%}")

print()
print()
print("What it costs to record the missing four.")
print()
ADD = [
    ("propagate a chain header",       0.44, 0.5),
    ("attach the task id at entry",    0.29, 0.3),
    ("record the triggering content",  0.51, 2.0),
    ("record the scope exercised",     0.38, 1.2),
]
print(f"{'addition':>34}{'settles':>11}{'effort':>9}{'per effort':>13}"
      f"{'cumulative answerable':>24}")
print("-" * 92)
cum = top2
for name, settles, eff in ADD:
    cum = 1 - (1 - cum) * (1 - settles)
    print(f"{name:>34}{settles:>11.0%}{eff:>9.1f}{settles / eff:>13.3f}"
          f"{cum:>24.0%}")

print()
print(f"four additions, {sum(e for n, s, e in ADD):.1f} units of effort, "
      f"{cum:.0%} answerable")

print()
print()
print("And the governance question underneath: what a policy can be about.")
print()
LEVELS = [
    ("a tool",            "may the agent call it",     "static",  "no"),
    ("a resource",        "may this record be touched", "static", "partly"),
    ("a principal chain", "did the user authorise this", "dynamic", "yes"),
    ("an outcome",        "is money moving",           "dynamic", "yes"),
    ("a task",            "is this within the ask",    "dynamic", "yes"),
]
print(f"{'policy is about':>22}{'the question it asks':>32}{'evaluation':>13}"
      f"{'survives a new tool?':>23}")
print("-" * 90)
for name, q, ev, surv in LEVELS:
    print(f"{name:>22}{q:>32}{ev:>13}{surv:>23}")

print(f"""
The chain table is the default and it is worth staring at. The user holds
{HOPS[0][1]:.2f} of the available authority. The request executes with {preserved:.2f},
because the orchestrator runs as a service account and **nothing between the entry point and
the backend removes anything** (eq:delegation-preserves-authority-unless-attenuated).

An explicit attenuation of {1 - 0.72:.0%} a hop would end at {atten:.3f}. Least privilege
would end at {HOPS[0][1]:.2f}. Neither is what happens by default, because attenuating requires
somebody to decide what to remove at each hop and the code that forwards a request has no
opinion.

The transition table shows where authority enters. `{GAINS[0][0]}` is
{GAINS[0][2] - GAINS[0][1]:+.2f} -- the service account -- and every hop after it is
{GAINS[1][2] - GAINS[1][1]:+.2f}. **The only narrowing on the list happens by accident**,
because the tool server happens to hold less, and it is undone at the next hop when the server
uses its own credential.

That is the inversion worth naming. A chain of *trust* is as weak as its weakest link. A chain
of *authority* is as strong as its most privileged member, and privilege flows downhill.

The models table is what can be expressed. `RBAC on the service account` scores
{mod['RBAC on the service account'][0]:.2f} on expressiveness because its principal is the
agent -- so every request from every user looks identical to the policy engine, and the policy
cannot mention the user at all.

`{MODELS[3][0]}` scores {mod[MODELS[3][0]][0]:.2f} and `{MODELS[4][0]}` scores
{mod[MODELS[4][0]][0]:.2f}. **Only the last two carry the user's identity to the far end, and
only they can narrow on the way** -- which is the same delegation result ch:sd-apis-auth
reached from the API side and ch:sec-tool-abuse from the authority side.

The audit table is the governance half. Six fields; the acting identity is recorded
{FIELDS[0][2]:.0%} of the time and the full principal chain {FIELDS[2][2]:.0%}.

Together the top two settle {top2:.0%} of audit questions
(eq:audit-completeness-requires-the-principal-chain), and all six settle
{1 - cum_miss:.0%}. The gap is the questions that begin "why did it do that" rather than
"who did it", and those are the ones an incident actually asks.

The addition table prices closing it. Propagating a chain header settles
{ADD[0][1]:.0%} for {ADD[0][2]:.1f} units of effort -- the best ratio on the list -- and four
additions totalling {sum(e for n, s, e in ADD):.1f} units take answerability to {cum:.0%}.

Note what `record the triggering content` is: it is ch:ops-agent-tracing's payload field,
already argued for on triage grounds and already argued *against* in ch:sec-data-leakage's leak
accounting. **The same field is the most valuable audit record and the largest leak source**,
and the resolution is the same one that chapter reached -- record it, redact at emit -- rather
than a choice between the two.

The last table is the governance point. A policy about a *tool* is static and does not survive
the next integration; a policy about an *outcome* or a *task* is dynamic and does. That is
ch:sec-tool-abuse's approval result in policy form, and it is the reason a permission system
built around a tool list has to be rewritten every time the product grows and one built around
outcomes does not.""")
