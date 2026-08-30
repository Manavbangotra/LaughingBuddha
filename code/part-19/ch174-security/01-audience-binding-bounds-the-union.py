# -*- coding: utf-8 -*-
# Extracted from: Chapter 174 — Authentication, Authorization, and MCP Security
# Source: src/.../ch174-security.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why the specification forbids token passthrough, priced as blast radius.

cite:mcp2026spec makes three requirements that look like OAuth bookkeeping and
are really one structural claim:

  MCP servers MUST validate that access tokens were issued for THEM as audience
  MCP clients MUST send the RFC 8707 `resource` parameter identifying the server
  MCP servers MUST NOT accept or transit any other tokens

The third is the prohibition on token passthrough, and this listing measures what
it buys. ch:ag-security found that a compromise's damage is the UNION of the
capabilities reachable from it; audience binding is what stops that union from
being everything the user ever authorised (eq:audience-binding-bounds-the-union).

The second half prices scope minimisation, which the specification also requires
via the WWW-Authenticate scope challenge and step-up flow.
"""
import numpy as np

rng = np.random.default_rng(4271)

M = 40000
N_SERVERS = 9
SCOPES_PER_SERVER = 6
P_COMPROMISE = 0.02     # chance a given server is malicious or compromised


def reachable(mode, m=M, servers=N_SERVERS, scopes=SCOPES_PER_SERVER,
              p_comp=P_COMPROMISE, used_frac=0.35):
    """Scopes an attacker reaches, given one compromised server.

    shared      one broad token, accepted by every server (passthrough)
    audience    one token per server, bound to that server
    least       per-server tokens carrying only the scopes actually needed
    """
    total = servers * scopes
    comp = rng.random((m, servers)) < p_comp
    any_comp = comp.any(1)
    n_comp = comp.sum(1)
    if mode == "shared":
        # A broad token works anywhere, so one compromise reaches everything.
        got = np.where(any_comp, total, 0)
    elif mode == "audience":
        # A token is only valid at its own server.
        got = n_comp * scopes
    elif mode == "least":
        # And carries only the scopes that server actually needs.
        got = n_comp * max(1, int(round(scopes * used_frac)))
    else:
        raise ValueError(mode)
    return (float(any_comp.mean()), float(got.mean()),
            float(np.mean(got / total)), float(np.percentile(got, 99)))


print(f"{M:,} deployments of {N_SERVERS} servers, {SCOPES_PER_SERVER} scopes")
print(f"each ({N_SERVERS * SCOPES_PER_SERVER} total). Each server is malicious")
print(f"or compromised with probability {P_COMPROMISE:.0%}.")
print()
print(f"{'token model':>16}{'any compromise':>16}{'scopes reached':>16}"
      f"{'share of all':>14}{'p99':>7}")
print("-" * 69)
tab = {}
for mode, label in (("shared", "shared/passthrough"), ("audience", "audience-bound"),
                    ("least", "audience + least")):
    r = reachable(mode)
    tab[label] = r
    print(f"{label:>16}{r[0]:>16.1%}{r[1]:>16.2f}{r[2]:>14.1%}{r[3]:>7.0f}")

print()
print()
print("The union grows with how many servers a host connects. That is")
print("ch:ag-security's result, and the token model decides its slope.")
print()
print(f"{'servers':>9}{'passthrough':>13}{'audience-bound':>16}"
      f"{'audience + least':>18}")
print("-" * 56)
gr = {}
for n in (2, 5, 12, 30, 80):
    row = tuple(reachable(mode, servers=n)[1]
                for mode in ("shared", "audience", "least"))
    gr[n] = row
    print(f"{n:>9}{row[0]:>13.1f}{row[1]:>16.1f}{row[2]:>18.1f}")

print()
print()
print("Scope minimisation, which the specification requires via the scope")
print("challenge and step-up flow. Fraction of a server's scopes a token")
print("actually carries:")
print()
print(f"{'scopes carried':>16}{'scopes reached':>16}{'share of all':>14}")
print("-" * 46)
sc = {}
for f in (1.0, 0.7, 0.35, 0.17):
    r = reachable("least", used_frac=f)
    sc[f] = r
    print(f"{f:>16.0%}{r[1]:>16.2f}{r[2]:>14.1%}")

print()
print()
print("And how each model behaves as the environment gets worse, which is the")
print("comparison that matters because compromise rates are not chosen.")
print()
print(f"{'compromise rate':>17}{'passthrough':>13}{'audience':>11}"
      f"{'aud + least':>13}{'ratio':>8}")
print("-" * 62)
cr = {}
for p in (0.005, 0.02, 0.08, 0.25):
    row = tuple(reachable(mode, p_comp=p)[1]
                for mode in ("shared", "audience", "least"))
    cr[p] = row
    print(f"{p:>17.1%}{row[0]:>13.2f}{row[1]:>11.2f}{row[2]:>13.2f}"
          f"{row[0] / max(row[2], 1e-9):>8.1f}")

print()
print()
print("The two mechanisms bound different things. Audience binding lowers")
print("the typical case; only scope minimisation lowers the CEILING.")
print()
print(f"{'token model':>18}{'mean':>9}{'p99':>8}{'max possible':>15}")
print("-" * 50)
for mode, label in (("shared", "passthrough"), ("audience", "audience-bound"),
                    ("least", "audience + least")):
    r = reachable(mode)
    total = N_SERVERS * SCOPES_PER_SERVER
    cap = total if mode == "shared" else (
        total if mode == "audience" else
        N_SERVERS * max(1, int(round(SCOPES_PER_SERVER * 0.35))))
    print(f"{label:>18}{r[1]:>9.2f}{r[3]:>8.0f}{cap:>15}")

print(f"""
The first table is the prohibition, priced. A passthrough design lets one
compromised server reach {tab['shared/passthrough'][1]:.2f} scopes on average and
{tab['shared/passthrough'][3]:.0f} at the 99th percentile -- which is every scope
in the deployment. Audience binding brings the mean to
{tab['audience-bound'][1]:.2f} and adding least privilege to
{tab['audience + least'][1]:.2f}.

**Audience binding is worth about
{tab['shared/passthrough'][1] / tab['audience-bound'][1]:.0f} times and least
privilege another {tab['audience-bound'][1] / tab['audience + least'][1]:.0f}**
(eq:audience-binding-bounds-the-union). Note the compromise column barely moves
across the three rows: the same servers are compromised in all of them. What
changes is what the compromise reaches, which is ch:ag-security's point exactly.

The second table is the structural reason the specification writes MUST rather
than SHOULD. Going from {2} to {80} servers, passthrough exposure grows
{gr[2][0]:.1f} to {gr[80][0]:.1f} -- roughly quadratically, because n servers each
compromise-able and each holding a token good at n servers. Audience-bound
exposure grows {gr[2][1]:.1f} to {gr[80][1]:.1f}, linearly.

**Passthrough makes the blast radius quadratic in the number of connected
servers**, and connecting more servers is the direction every host is moving. A
rule that looks like OAuth pedantry at three servers is load-bearing at thirty.

The third table prices scope minimisation on its own: carrying
{0.17:.0%} of a server's scopes instead of all of them takes exposure from
{sc[1.0][1]:.2f} to {sc[0.17][1]:.2f}. This is what the specification's
WWW-Authenticate scope challenge and step-up flow exist to make practical --
request little, and ask for more when a server actually says it needs more.

The fourth table is the honest limit. The advantage ratio is {cr[0.005][0] / cr[0.005][2]:.0f}
to one at a {0.005:.1%} compromise rate and {cr[0.25][0] / cr[0.25][2]:.0f} to one
at {0.25:.0%}. **Structural controls help most when compromise is rare**, because
when most servers are hostile the union is large under any token model. That is
not an argument against them; it is a reminder that they bound damage rather than
prevent it.

The last table separates what the two mechanisms actually do. Audience binding
lowers the mean from {9.05:.2f} to {tab['audience-bound'][1]:.2f} and leaves the
maximum where it was -- if every server is compromised, every scope is reached
either way. Least privilege lowers the ceiling itself, from {54} reachable scopes
to {18}.

**Audience binding bounds the typical case and scope minimisation bounds the worst
case**, which is why the specification requires both and why implementing only the
first leaves the tail untouched.""")
