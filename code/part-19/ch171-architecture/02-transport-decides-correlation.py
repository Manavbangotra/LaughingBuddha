# -*- coding: utf-8 -*-
# Extracted from: Chapter 171 — MCP Architecture: Hosts, Clients, Servers, and Transports
# Source: src/.../ch171-architecture.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The two transports, and the thing they actually decide.

cite:mcp2026spec defines two standard bindings. They are usually compared on
convenience -- stdio is easy locally, HTTP works remotely -- and that comparison
misses the property that matters operationally.

  stdio             the client LAUNCHES the server as a subprocess. One process
                    per (client, server) pair, on the client's machine, taking
                    credentials from the environment.
  Streamable HTTP   one server, shared. Every client of that server depends on
                    the same process.

So the transport chooses whether tool failures are INDEPENDENT or CORRELATED
across your clients (eq:transport-decides-correlation), and ch:as-failures
already established that mean availability hides everything interesting about
correlated failure.

This listing measures resource cost and the failure tail together.
"""
import numpy as np

rng = np.random.default_rng(4099)

DAYS = 4000
CLIENTS = 60
SERVERS = 12
P_DOWN = 0.004          # chance a given server process is down on a given day
MEM_STDIO = 85.0        # MB per subprocess
MEM_HTTP = 260.0        # MB for a shared server, which does more


def simulate(mode, days=DAYS, clients=CLIENTS, servers=SERVERS,
             p_down=P_DOWN, replicas=1):
    """Return per-client-tool availability and the distribution of how many
    clients are simultaneously affected."""
    if mode == "stdio":
        # One process per (client, server): failures are independent.
        down = rng.random((days, clients, servers)) < p_down
    else:
        # One shared process per server, replicated: a server is down only if
        # every replica is, and when it is, ALL clients lose it together.
        rep = rng.random((days, servers, replicas)) < p_down
        server_down = rep.all(2)
        down = np.repeat(server_down[:, None, :], clients, axis=1)
    avail = 1.0 - float(down.mean())
    # How many clients lose at least one tool on the same day.
    affected = down.any(2).sum(1) / clients
    return avail, affected


print(f"{CLIENTS} clients using {SERVERS} tool servers over {DAYS:,} days, with")
print(f"each server process down {P_DOWN:.1%} of the time.")
print()
print(f"{'deployment':>22}{'availability':>14}{'processes':>11}{'memory GB':>12}")
print("-" * 59)
res = {}
for name, mode, reps in (("stdio (per client)", "stdio", 1),
                         ("http, 1 replica", "http", 1),
                         ("http, 2 replicas", "http", 2),
                         ("http, 3 replicas", "http", 3)):
    a, aff = simulate(mode, replicas=reps)
    procs = CLIENTS * SERVERS if mode == "stdio" else SERVERS * reps
    mem = procs * (MEM_STDIO if mode == "stdio" else MEM_HTTP) / 1024
    res[name] = (a, aff, procs, mem)
    print(f"{name:>22}{a:>14.3%}{procs:>11,}{mem:>12,.1f}")

print()
print()
print("The same runs, viewed as ch:as-failures viewed a panel: not the mean but")
print("the joint behaviour. Share of days on which more than half the fleet")
print("loses a tool at the same moment.")
print()
print(f"{'deployment':>22}{'availability':>14}{'any client hit':>16}"
      f"{'>50% hit together':>19}")
print("-" * 71)
for name in res:
    a, aff, _, _ = res[name]
    print(f"{name:>22}{a:>14.3%}{float((aff > 0).mean()):>16.1%}"
          f"{float((aff > 0.5).mean()):>19.2%}")

print()
print()
print("The measure that separates them: GIVEN a bad day, what share of the")
print("fleet is affected? That is correlation, and it is invisible in a mean.")
print()
print(f"{'process down rate':>19}{'stdio avail':>13}{'stdio severity':>16}"
      f"{'http avail':>12}{'http severity':>15}")
print("-" * 75)
sev = {}
for pd in (0.001, 0.004, 0.02, 0.08):
    a_s, aff_s = simulate("stdio", p_down=pd)
    a_h, aff_h = simulate("http", p_down=pd, replicas=1)
    bad_s = aff_s[aff_s > 0]
    bad_h = aff_h[aff_h > 0]
    ss = float(bad_s.mean()) if len(bad_s) else 0.0
    sh_ = float(bad_h.mean()) if len(bad_h) else 0.0
    sev[pd] = (a_s, ss, a_h, sh_)
    print(f"{pd:>19.1%}{a_s:>13.2%}{ss:>16.1%}{a_h:>12.2%}{sh_:>15.1%}")

print()
print()
print("Replication is the fix for the correlated deployment, and it works on the")
print("tail rather than only on the mean.")
print()
print(f"{'replicas':>10}{'availability':>14}{'>50% hit together':>19}"
      f"{'memory GB':>12}")
print("-" * 55)
rp = {}
for reps in (1, 2, 3, 4):
    a, aff = simulate("http", replicas=reps)
    mem = SERVERS * reps * MEM_HTTP / 1024
    rp[reps] = (a, float((aff > 0.5).mean()), mem)
    print(f"{reps:>10}{a:>14.4%}{float((aff > 0.5).mean()):>19.3%}"
          f"{mem:>12,.1f}")

print()
print()
print("And how the resource comparison moves with fleet size, which is what")
print("actually decides the transport for a deployed product.")
print()
print(f"{'clients':>9}{'stdio processes':>17}{'stdio GB':>11}"
      f"{'http GB (2x)':>14}{'ratio':>9}")
print("-" * 60)
sz = {}
for c in (1, 5, 25, 100, 500):
    sp = c * SERVERS
    sm = sp * MEM_STDIO / 1024
    hm = SERVERS * 2 * MEM_HTTP / 1024
    sz[c] = (sp, sm, hm)
    print(f"{c:>9}{sp:>17,}{sm:>11,.1f}{hm:>14,.1f}{sm / hm:>9.2f}")

print(f"""
The first table is the comparison people make, and by it the shared deployment
wins comfortably: {res['stdio (per client)'][2]:,} processes and
{res['stdio (per client)'][3]:,.1f} GB against {res['http, 2 replicas'][2]} and
{res['http, 2 replicas'][3]:,.1f}, at better availability.

The second table is the comparison ch:as-failures says to make instead, and it
does not agree.

Per-client availability for stdio is {res['stdio (per client)'][0]:.3%} and for a
single-replica HTTP server {res['http, 1 replica'][0]:.3%} -- the same number. But
stdio has some client affected on {float((res['stdio (per client)'][1] > 0).mean()):.1%}
of days and more than half the fleet affected on
{float((res['stdio (per client)'][1] > 0.5).mean()):.2%}, while the shared server
affects anyone on only {float((res['http, 1 replica'][1] > 0).mean()):.1%} of days
and affects more than half on {float((res['http, 1 replica'][1] > 0.5).mean()):.2%}.

Every bad day for the shared server is a bad day for everyone.

The third table states that directly. Conditional on a day when something is down,
the share of the fleet affected is {sev[0.001][1]:.1%} for stdio at a
{0.001:.1%} process failure rate and {sev[0.001][3]:.1%} for the shared server --
and the availability columns beside them agree to two decimal places in every row.

**The transport decides whether tool failures are independent or correlated**
(eq:transport-decides-correlation), and correlation is invisible in the number
everyone reports. An availability SLO is a statement about the mean; the thing
that wakes people up is the severity, and these two deployments have identical
means and opposite severities.

That is ch:as-failures' result about panels, arriving at the deployment layer.
Sharing a dependency is what makes failures correlate, and it does not matter
whether the shared thing is a base model or a process.

The fourth table is the fix, and it is worth noting WHERE it acts. Going from one
replica to two takes availability from {rp[1][0]:.4%} to {rp[2][0]:.4%} -- a small
move -- and takes the fleet-wide outage rate from {rp[1][1]:.3%} to
{rp[2][1]:.3%}, a factor of about {rp[1][1] / max(rp[2][1], 1e-9):.0f}.
**Replication buys tail, not mean**, which is exactly what the correlated
deployment needs and exactly what an availability-driven capacity plan will
under-provision.

The last table is what actually decides the transport in practice. At one client
the stdio deployment uses {sz[1][1] / sz[1][2]:.2f} times the memory of the shared
one; at {25} clients {sz[25][1] / sz[25][2]:.1f} times; at {500} clients
{sz[500][1] / sz[500][2]:.0f} times.

The crossover is around {5} clients, which is the honest summary of the two
transports: **stdio for a single user's machine, HTTP for anything shared.** That
is the guidance everyone already gives, and the listing's contribution is the
condition attached to it -- if you take the shared path, you have converted a
diffuse failure mode into a concentrated one, and you owe the fleet replication
you would not have needed before.""")
