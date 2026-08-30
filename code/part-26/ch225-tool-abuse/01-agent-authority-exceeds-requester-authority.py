# -*- coding: utf-8 -*-
# Extracted from: Chapter 225 — Tool Abuse, Agent Hijacking, and Sandboxing
# Source: src/.../ch225-tool-abuse.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""An agent is a confused deputy by construction: it holds authority its requester lacks.

A service account exists because the agent must act for many users. So it is provisioned with
the union of what any of them might need, and every request runs with that union while the
requester is entitled to a slice of it
(eq:agent-authority-exceeds-requester-authority).

That is the classical confused-deputy problem, and the classical fix -- pass the requester's
authority rather than the deputy's -- is ch:sd-apis-auth's delegation result.

Sandboxing is the other reflex, and it addresses a different axis. A sandbox bounds what
*code* can reach. The agent's credentials are carried inside the sandbox, so a sandbox with an
unscoped credential in its environment has moved the boundary and not the authority
(eq:a-sandbox-without-scoped-credentials-moves-nothing).
"""
# (resource class, records, share of users entitled, damage per record)
RESOURCES = [
    ("the requester's own records",      120,     1.000, 1.0),
    ("their team's records",           4_100,     0.180, 1.2),
    ("their tenant's records",       240_000,     0.041, 1.6),
    ("all tenants' records",      31_000_000,     0.000, 4.0),
    ("internal config and secrets",      840,     0.002, 9.0),
    ("billing and payments",          62_000,     0.008, 7.5),
]

print("What the service credential reaches, against what the requester may.")
print()
print(f"{'resource class':>32}{'records':>14}{'entitled users':>17}"
      f"{'agent can reach?':>19}{'excess records':>17}")
print("-" * 99)
total, excess = 0.0, 0.0
res = {}
for name, rec, ent, dmg in RESOURCES:
    total += rec
    ex = rec * (1.0 - ent)
    excess += ex
    res[name] = (rec, ent, ex, dmg)
    print(f"{name:>32}{rec:>14,}{ent:>17.1%}{'yes':>19}{ex:>17,.0f}")
print("-" * 99)
print(f"{'TOTAL':>32}{total:>14,.0f}{'':>17}{'':>19}{excess:>17,.0f}")

print()
print(f"excess authority ratio: {excess / total:.4f} of reachable records are")
print("outside the median requester's entitlement")

print()
print()
print("Damage-weighted, which is the version that matters.")
print()
w_total = sum(r * d for n, r, e, d in RESOURCES)
w_excess = sum(r * (1 - e) * d for n, r, e, d in RESOURCES)
print(f"{'resource class':>32}{'weighted total':>17}{'weighted excess':>18}"
      f"{'share of excess':>18}")
print("-" * 87)
for name, rec, ent, dmg in RESOURCES:
    we = rec * (1 - ent) * dmg
    print(f"{name:>32}{rec * dmg:>17,.0f}{we:>18,.0f}{we / w_excess:>18.1%}")
print("-" * 87)
print(f"{'TOTAL':>32}{w_total:>17,.0f}{w_excess:>18,.0f}{1.0:>18.1%}")

print()
print()
print("Scoping the credential, which is the fix that addresses authority.")
print()
SCOPES = [
    ("one service account for everything",  1.00, 0.00),
    ("per-tenant service accounts",         0.0078, 0.05),
    ("per-user delegated token",            0.0000, 0.22),
    ("per-request capability, task-scoped", 0.0000, 0.34),
]
print(f"{'credential model':>38}{'excess reachable':>19}{'weighted excess':>18}"
      f"{'engineering cost':>19}")
print("-" * 94)
scope = {}
for name, frac, cost in SCOPES:
    scope[name] = (excess * frac, w_excess * frac, cost)
    print(f"{name:>38}{excess * frac:>19,.0f}{w_excess * frac:>18,.0f}"
          f"{cost:>19.0%}")

print()
print("A per-tenant account removes 99% of the excess and leaves the")
print("cross-tenant path closed but the in-tenant one open.")

print()
print()
print("Now the sandbox, which addresses a different axis entirely.")
print()
ESCAPES = [
    ("filesystem outside the workdir",  0.31, 0.4, "yes"),
    ("outbound network",                0.58, 0.6, "yes"),
    ("spawning processes",              0.22, 0.5, "yes"),
    ("reading the environment",         0.91, 0.2, "yes"),
    ("using an injected credential",    1.00, 0.0, "no"),
    ("calling a tool the agent has",    1.00, 0.0, "no"),
]
print(f"{'capability':>34}{'available by default':>23}{'cost to remove':>17}"
      f"{'sandbox removes it?':>21}")
print("-" * 95)
for name, avail, cost, removes in ESCAPES:
    print(f"{name:>34}{avail:>23.0%}{cost:>17.1f}{removes:>21}")

print()
print("The last two rows are what an agent hijack actually uses, and a")
print("sandbox does not touch either.")

print()
print()
print("Combining the two axes. Both are needed and they are not substitutes.")
print()
print(f"{'posture':>44}{'code escape':>14}{'weighted excess authority':>28}"
      f"{'residual':>11}")
print("-" * 97)
POSTURES = [
    ("no sandbox, one service account",     0.58, 1.00),
    ("sandboxed, one service account",      0.04, 1.00),
    ("no sandbox, per-user delegation",     0.58, 0.00),
    ("sandboxed, per-user delegation",      0.04, 0.00),
    ("sandboxed, per-request capability",   0.04, 0.00),
]
for name, esc, auth in POSTURES:
    wa = w_excess * auth
    residual = esc * 1000.0 + wa * 1e-3
    print(f"{name:>44}{esc:>14.0%}{wa:>28,.0f}{residual:>11,.0f}")

print()
print("Sandboxing alone changes the first column. Scoping alone changes the")
print("second. Only doing both changes the residual.")

print()
print()
print("Where the credential physically is, which decides whether scoping works.")
print()
PLACES = [
    ("environment variable in the sandbox", "the agent reads it",  "no"),
    ("mounted secret file",                 "the agent reads it",  "no"),
    ("injected into the tool call by a broker", "never in context", "yes"),
    ("held by a proxy that signs requests", "never in context",    "yes"),
    ("short-lived token minted per request", "in context, briefly", "partly"),
]
print(f"{'where the credential lives':>42}{'agent access':>22}"
      f"{'survives a hijack?':>21}")
print("-" * 85)
for name, access, survives in PLACES:
    print(f"{name:>42}{access:>22}{survives:>21}")

print(f"""
The reach table is the confused-deputy problem counted. The service credential reaches
{total:,.0f} records; the median requester is entitled to a small fraction of them, so
**{excess / total:.2%} of reachable records are outside the requester's entitlement**
(eq:agent-authority-exceeds-requester-authority).

That is not a misconfiguration. It is what a service account is: the union of what any user
might need, held constantly, because the alternative is provisioning per request and nobody
built that.

The damage-weighted version sharpens it. `{RESOURCES[3][0]}` is
{res[RESOURCES[3][0]][2] * res[RESOURCES[3][0]][3] / w_excess:.0%} of weighted excess by
itself, and `{RESOURCES[5][0]}` is
{res[RESOURCES[5][0]][2] * res[RESOURCES[5][0]][3] / w_excess:.1%}. **The excess is
concentrated in exactly the resources a service account is provisioned for and a user never
touches.**

The scoping table is ch:sd-apis-auth's delegation result priced. Per-tenant service accounts
take weighted excess from {w_excess:,.0f} to
{scope['per-tenant service accounts'][1]:,.0f} for {0.05:.0%} of engineering cost -- the
cross-tenant path closed, the in-tenant one open. Per-user delegation takes it to zero for
{0.22:.0%}.

Now the sandbox, and this is where the chapter's second result lives. The escape table lists
what sandboxing removes: filesystem access outside the working directory, outbound network,
process spawning, environment reading. All real, all worth removing.

Read the last two rows. **Using an injected credential** and **calling a tool the agent
already has** are available at {1.00:.0%} and a sandbox removes neither
(eq:a-sandbox-without-scoped-credentials-moves-nothing), because both are things the agent is
*supposed* to do. The sandbox is a boundary around the code; the credential and the tool
handles are inside it.

That is the confusion worth removing. A sandbox is the right control for *untrusted code* --
generated Python, a downloaded package, a user-supplied script. It is not a control for an
*untrusted instruction* reaching a trusted tool, which is what a hijack is.

The posture table makes the non-substitution explicit. Sandboxing alone takes code escape from
{0.58:.0%} to {0.04:.0%} and leaves weighted excess authority at {w_excess:,.0f}. Delegation
alone reverses that. **Only doing both moves the residual**, and each is usually owned by a
different team with a different threat model.

The last table is the implementation detail that decides whether scoping is real. A credential
in an environment variable or a mounted file is readable by the agent -- so a hijacked agent
reads it and uses it outside every control you built, including the sandbox it is sitting in.
A credential injected into the tool call by a broker, or held by a signing proxy, is never in
the agent's context at all.

**Scoping a credential the agent can read is scoping a credential the attacker also has.**
The broker pattern is the one that survives, it is roughly a week of work, and it is the
difference between a control and a configuration.""")
