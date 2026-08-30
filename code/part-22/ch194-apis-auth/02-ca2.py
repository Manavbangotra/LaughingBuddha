# -*- coding: utf-8 -*-
# Extracted from: Chapter 194 — APIs, Authentication, Authorization, and Rate Limiting
# Source: src/.../ch194-apis-auth.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""When an agent acts for a user, the permission check moves to the wrong boundary.

A user calls an API. The API runs an agent. The agent calls tools. The tools touch
data. Somewhere in that chain a decision gets made about what is allowed, and WHERE
it gets made determines how much authority the user effectively gains.

The convenient implementation gives the agent a service account -- one identity with
enough permission to serve any user. Every request then runs with the union of every
user's rights, and the only thing preventing one user's request from reaching another
user's data is the model choosing not to
(eq:delegation-moves-the-check).

This listing measures the over-permission each delegation design produces, against
what it costs to build.
"""
# Tools the agent can call, and what each needs.
# (tool, scope required, share of requests using it, blast radius if misused)
TOOLS = [
    ("search_documents",   "docs:read",      0.91,  40.0),
    ("read_document",      "docs:read",      0.74,  40.0),
    ("list_users",         "dir:read",       0.12, 120.0),
    ("send_message",       "msg:write",      0.09, 260.0),
    ("update_record",      "crm:write",      0.06, 900.0),
    ("run_report",         "analytics:read", 0.04, 310.0),
    ("delete_document",    "docs:delete",    0.01, 1400.0),
]

# What a typical individual user is actually entitled to.
# Share of users holding each scope.
ENTITLED = {
    "docs:read":      0.98,
    "dir:read":       0.34,
    "msg:write":      0.55,
    "crm:write":      0.12,
    "analytics:read": 0.09,
    "docs:delete":    0.04,
}

DESIGNS = [
    # (name, effective scope per request, engineering cost, audit fidelity)
    ("service account",        "union",     1.0, 0.15),
    ("service account + row filter", "union-filtered", 2.4, 0.35),
    ("token passthrough",      "user",      4.1, 0.95),
    ("per-tool exchanged token", "user-tool", 6.8, 1.00),
]


def over_permission(design):
    """Expected share of a request's tool calls the agent could make that the
    invoking user could not make directly, weighted by blast radius."""
    total = 0.0
    excess = 0.0
    for tool, scope, freq, blast in TOOLS:
        held = ENTITLED[scope]
        total += freq * blast
        if design == "union":
            # The agent holds every scope regardless of the user.
            excess += freq * blast * (1.0 - held)
        elif design == "union-filtered":
            # Row filtering catches data the user cannot see, but only where a
            # row-level owner exists. Writes and directory calls have no owner
            # column to filter on.
            filterable = 0.6 if scope.endswith(":read") else 0.0
            excess += freq * blast * (1.0 - held) * (1.0 - filterable)
        elif design == "user":
            # The user's own token is presented; nothing is gained.
            excess += 0.0
        elif design == "user-tool":
            excess += 0.0
    return excess, total


print("Seven tools an agent can call, with what each requires and what a misuse")
print("would reach.")
print()
print(f"{'tool':>20}{'scope':>18}{'used in':>10}{'blast radius':>15}"
      f"{'users holding':>15}")
print("-" * 78)
for tool, scope, freq, blast in TOOLS:
    print(f"{tool:>20}{scope:>18}{freq:>10.0%}{blast:>15.0f}"
          f"{ENTITLED[scope]:>15.0%}")

print()
print()
print("A service account needs the union of every scope, because it must be able")
print("to serve any user. That is what every request then runs with.")
print()
union = sorted(set(s for _, s, _, _ in TOOLS))
print(f"scopes in the union:  {len(union)}")
print(f"scopes a median user holds: "
      f"{sum(1 for s in union if ENTITLED[s] >= 0.5)}")
print()
print(f"{'scope':>18}{'in union':>11}{'users holding':>16}"
      f"{'granted to':>14}")
print("-" * 59)
for s in union:
    print(f"{s:>18}{'yes':>11}{ENTITLED[s]:>16.0%}{1.0:>13.0%}")

print()
print()
print("Over-permission by design: the share of blast radius the agent can reach")
print("that the invoking user could not reach directly.")
print()
print(f"{'design':>30}{'over-permission':>18}{'eng cost':>11}"
      f"{'audit fidelity':>17}")
print("-" * 76)
res = {}
for name, mode, cost, audit in DESIGNS:
    ex, tot = over_permission(mode)
    res[name] = (ex / tot, cost, audit, ex)
    print(f"{name:>30}{ex / tot:>18.1%}{cost:>10.1f}x{audit:>17.0%}")

print()
print()
print("Where the over-permission sits, under a service account. Ranked by")
print("exposure, which is frequency times blast radius times the share of users")
print("who could NOT do it themselves.")
print()
print(f"{'tool':>20}{'used in':>10}{'blast':>9}{'lacking':>10}"
      f"{'exposure':>12}{'share of total':>17}")
print("-" * 78)
ex_total = over_permission("union")[0]
rows = []
for tool, scope, freq, blast in TOOLS:
    e = freq * blast * (1.0 - ENTITLED[scope])
    rows.append((tool, freq, blast, 1.0 - ENTITLED[scope], e))
rows.sort(key=lambda r: -r[4])
for tool, freq, blast, lack, e in rows:
    print(f"{tool:>20}{freq:>10.0%}{blast:>9.0f}{lack:>10.0%}"
          f"{e:>12.1f}{e / ex_total:>17.0%}")

print()
print()
print("The cheap partial fix: remove the two highest-exposure tools from the")
print("agent's set and require a human to invoke them directly.")
print()
drop = [rows[0][0], rows[1][0]]
kept = [t for t in TOOLS if t[0] not in drop]
kept_ex = sum(f * b * (1.0 - ENTITLED[s]) for _, s, f, b in
              [(t[0], t[1], t[2], t[3]) for t in kept])
kept_tot = sum(f * b for _, _, f, b in
               [(t[0], t[1], t[2], t[3]) for t in kept])
print(f"removed: {drop[0]}, {drop[1]}")
print()
print(f"{'configuration':>34}{'over-permission':>18}{'requests affected':>20}")
print("-" * 72)
affected = sum(t[2] for t in TOOLS if t[0] in drop)
print(f"{'service account, all seven tools':>34}"
      f"{res['service account'][0]:>18.1%}{0.0:>20.0%}")
print(f"{'service account, five tools':>34}{kept_ex / kept_tot:>18.1%}"
      f"{affected:>20.0%}")
print(f"{'token passthrough, all seven':>34}"
      f"{res['token passthrough'][0]:>18.1%}{0.0:>20.0%}")

print(f"""
The union table is the whole problem stated once. Serving any user requires
{len(union)} scopes, and a median user holds
{sum(1 for s in union if ENTITLED[s] >= 0.5)} of them. Every request therefore
executes with authority that most of the people making requests do not have.

Under a service account, **{res['service account'][0]:.1%} of the blast radius the
agent can reach is unreachable by the user who invoked it**
(eq:delegation-moves-the-check). The only thing standing between a request and that
authority is the model deciding not to use it -- which ch:ag-security established is
not a security control, because a model's decision is exactly what prompt injection
targets.

Row-level filtering is the usual mitigation and it recovers less than it appears to.
It drops over-permission from {res['service account'][0]:.1%} to
{res['service account + row filter'][0]:.1%}, because filtering works on reads with
an owner column and does not work on writes, on directory lookups, or on anything
whose authorisation is not expressible as a row predicate. **The tools it cannot
protect are the ones with the largest blast radius**, which is the opposite of the
distribution you want a partial mitigation to have.

Token passthrough takes it to {res['token passthrough'][0]:.1%} -- the agent can do
exactly what the user can do -- at {res['token passthrough'][1]:.1f} times the
engineering cost and with audit fidelity rising from
{res['service account'][2]:.0%} to {res['token passthrough'][2]:.0%}. That second
column matters more than it looks: under a service account, the audit log records
that the service account read a document, which is true and useless.

The exposure ranking is where this becomes actionable, because the distribution is
not flat. `{rows[0][0]}` accounts for {rows[0][4] / ex_total:.0%} of all
over-permission and is used in {rows[0][1]:.0%} of requests;
`{rows[1][0]}` accounts for {rows[1][4] / ex_total:.0%}. Together those two are
{(rows[0][4] + rows[1][4]) / ex_total:.0%} of the exposure and appear in
{affected:.0%} of traffic.

So the cheap intervention is available before the expensive one. Removing those two
tools from the agent's set -- requiring a human to invoke them directly -- takes
over-permission from {res['service account'][0]:.1%} to {kept_ex / kept_tot:.1%}
while affecting {affected:.0%} of requests -- **closing
{(res['service account'][0] - kept_ex / kept_tot) / res['service account'][0]:.0%} of
the gap to full passthrough by editing a list**, against
{res['token passthrough'][1]:.1f} times the engineering cost for the remainder.

**The agent's effective authority is a design variable, and the cheapest way to
reduce it is to give the agent fewer tools rather than better credentials.** That is
worth stating plainly because the instinct runs the other way: teams add tools to
make the agent more capable, and reach for a delegation rewrite only once the
security review objects.

The general form is ch:mcp-security's boundary question arriving as an API design
decision. An authorisation check is only meaningful at a boundary where the identity
of the principal is still known, and every hop that replaces a user identity with a
service identity erases the information the check needed. **Authority is not lost
gradually across a call chain; it is lost at exactly one hop**, and which hop that is
is a choice.""")
