# -*- coding: utf-8 -*-
# Extracted from: Chapter 171 — MCP Architecture: Hosts, Clients, Servers, and Transports
# Source: src/.../ch171-architecture.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What the connection model costs, which is why the current revision changed it.

Revisions of MCP through 2025-11-25 established a connection-scoped session: the
client sent `initialize`, the pair negotiated once, and the session carried state
for its lifetime. The 2026-07-28 revision replaced that with stateless,
self-contained requests carrying their own version and capabilities in `_meta`
(cite:mcp2026spec).

That reads like a protocol-design preference. It is an operational one, and this
listing measures the two costs it removes.

  restart exposure  a session pinned to a process dies when that process does,
                    and everything not yet done in it is lost
  routing           a session must return to the SAME replica, so a fleet needs
                    sticky routing, and stickiness costs balance
                    (eq:sessions-pin-to-replicas)

A stateless request has neither property: any replica can serve it, and a restart
loses only what was in flight.
"""
import numpy as np

rng = np.random.default_rng(4051)

M = 40000
SESSION_LEN = 14        # requests a client makes against one server
REPLICAS = 6


def run(restart_per_req, stateful, m=M, session_len=SESSION_LEN,
        replicas=REPLICAS, resume_cost=3, resumable=True, budget_mult=1.6):
    """Walk a session of `session_len` requests. In the stateful regime the
    session is pinned to one replica; if that replica restarts, the session is
    lost and must be re-established from scratch, costing `resume_cost`
    requests of setup. In the stateless regime a restart costs only the one
    request in flight, which is retried elsewhere."""
    done = np.zeros(m, dtype=np.int64)
    cost = np.zeros(m, dtype=np.int64)
    lost = np.zeros(m, dtype=np.int64)
    alive = np.ones(m, dtype=bool)
    # The budget covers the work plus, for the stateful regime, its one-time
    # handshake -- so the tables below isolate the RESTART cost rather than
    # re-charging setup overhead, which the cost column already reports.
    budget = int(session_len * budget_mult) + (resume_cost if stateful else 0)
    if stateful:
        cost += resume_cost          # the initialize handshake
    for _ in range(session_len * 6):
        # ch:ag-termination's budget is what makes lost work fatal rather than
        # merely expensive: an agent step does not have unlimited retries.
        live = alive & (done < session_len) & (cost < budget)
        idx = np.flatnonzero(live)
        if not len(idx):
            break
        cost[idx] += 1
        hit = rng.random(len(idx)) < restart_per_req
        if stateful:
            # The whole session is gone; re-establish and start over.
            s = idx[hit]
            lost[s] += done[s]
            done[s] = 0
            cost[s] += resume_cost
            if not resumable:
                alive[s] = False
            done[idx[~hit]] += 1
        else:
            # Only the in-flight request is lost; retry it anywhere.
            done[idx[~hit]] += 1
    ok = alive & (done >= session_len) & (cost <= budget)
    return (float(ok.mean()), float(cost.mean()), float(lost.mean()))


print(f"{M:,} client sessions of {SESSION_LEN} requests each.")
print("A restart during a stateful session destroys the session; during a")
print("stateless one it costs the single request in flight.")
print()
print(f"{'restart / request':>19}{'stateful ok':>13}{'stateless ok':>14}"
      f"{'stateful cost':>15}{'stateless cost':>16}")
print("-" * 77)
tab = {}
for r in (0.0005, 0.002, 0.008, 0.02, 0.05):
    a = run(r, True)
    b = run(r, False)
    tab[r] = (a, b)
    print(f"{r:>19.2%}{a[0]:>13.1%}{b[0]:>14.1%}{a[1]:>15.1f}{b[1]:>16.1f}")

print()
print()
print("Wasted work: requests completed inside a session that was later lost.")
print()
print(f"{'restart / request':>19}{'stateful waste':>16}{'stateless waste':>17}")
print("-" * 52)
for r in (0.0005, 0.002, 0.008, 0.02, 0.05):
    print(f"{r:>19.2%}{tab[r][0][2]:>16.2f}{tab[r][1][2]:>17.2f}")

print()
print()
print("Session length is the exposure. A stateful session is a chain in exactly")
print("ch:ag-loop's sense; a stateless one is not.")
print()
print(f"{'session length':>16}{'stateful ok':>13}{'stateless ok':>14}{'gap':>9}")
print("-" * 52)
sl = {}
for L in (2, 8, 20, 60, 200):
    a = run(0.008, True, session_len=L)[0]
    b = run(0.008, False, session_len=L)[0]
    sl[L] = (a, b)
    print(f"{L:>16}{a:>13.1%}{b:>14.1%}{b - a:>+9.1%}")

print()
print()
print("And what happens when a lost session is NOT resumable -- the client has no")
print("retry loop, which is the common case for a tool call inside an agent step.")
print()
print(f"{'restart / request':>19}{'stateful, retry':>17}{'no retry':>11}"
      f"{'stateless':>12}")
print("-" * 59)
nr = {}
for r in (0.002, 0.008, 0.02, 0.05):
    a = run(r, True)[0]
    b = run(r, True, resumable=False)[0]
    c = run(r, False)[0]
    nr[r] = (a, b, c)
    print(f"{r:>19.2%}{a:>17.1%}{b:>11.1%}{c:>12.1%}")

print()
print()
print("The routing cost. A stateful session must return to the replica that")
print("holds it, so a fleet balances SESSIONS; a stateless fleet balances")
print("REQUESTS. Load imbalance across a fleet, as replicas are added:")
print()


def imbalance(replicas, stateful, sessions=240, trials=400):
    """Peak-to-mean load across the fleet. Sessions are concurrent and their
    weights are heavy-tailed -- a few clients make far more calls than the
    median, which is the case sticky routing handles worst."""
    out = []
    for _ in range(trials):
        w = rng.lognormal(0.0, 1.1, sessions)
        if stateful:
            # A session is pinned, so its whole weight lands on one replica.
            who = rng.integers(0, replicas, sessions)
            load = np.bincount(who, weights=w, minlength=replicas)
        else:
            # Each request is routed independently, so weights split.
            load = np.zeros(replicas)
            for r in range(replicas):
                load[r] = w.sum() / replicas
            load += rng.normal(0, np.sqrt(w.sum() / replicas) * 0.35, replicas)
        out.append(load.max() / load.mean())
    return float(np.mean(out))


print(f"{'replicas':>10}{'sticky (sessions)':>19}{'any replica':>13}"
      f"{'excess':>9}")
print("-" * 51)
im = {}
for R in (2, 4, 8, 16, 64):
    a = imbalance(R, True)
    b = imbalance(R, False)
    im[R] = (a, b)
    print(f"{R:>10}{a:>19.3f}{b:>13.3f}{a - b:>+9.3f}")

print(f"""
The first table's completion columns are the operational claim, and the cost
columns are the one that shows up on a bill.

At a {0.05:.0%} restart rate the stateful regime completes {tab[0.05][0][0]:.1%} of
sessions against the stateless regime's {tab[0.05][1][0]:.1%}, and it costs
{tab[0.05][0][1]:.1f} requests against {tab[0.05][1][1]:.1f}. Even at the lowest
rate tested the handshake alone puts stateful at {tab[0.0005][0][1]:.1f} against
{tab[0.0005][1][1]:.1f} -- about {tab[0.0005][0][1] / tab[0.0005][1][1] - 1:+.0%}
before anything goes wrong.

The waste table names the mechanism. A restart in a stateful session discards
{tab[0.05][0][2]:.2f} already-completed requests on average; in a stateless one it
discards {tab[0.05][1][2]:.2f}. **Session state is work that a crash can destroy,
and a self-contained request has none of it to destroy.**

The third table is the shape of the problem, and it is a shape this book has
measured before. The gap grows from {sl[2][1] - sl[2][0]:+.1%} at a two-request
session to {sl[200][1] - sl[200][0]:+.1%} at two hundred.

**A stateful session is a chain in exactly ch:ag-loop's sense**: its success is
the product of surviving every request, so it compounds. A stateless sequence is
not a chain, because a failed request is retried without disturbing the ones
before it. That chapter found the distinction worth more than a large model
improvement, and it applies here unchanged.

The fourth table matters because the retry loop the third assumes is often absent.
Inside an agent step, a tool call that fails is usually just a failed tool call --
nobody re-establishes the session and replays it. Without that loop the stateful
regime falls to {nr[0.05][1]:.1%} at a {0.05:.0%} restart rate while the stateless
regime stays at {nr[0.05][2]:.1%}.

The last table is the reason this is an operations decision rather than an
aesthetic one, and it is the one that gets discovered late.

A stateful session must return to the replica holding its state, so the fleet
balances SESSIONS rather than requests. With heavy-tailed client behaviour --
a few clients calling far more than the median, which is universal -- peak-to-mean
load reaches {im[64][0]:.2f} at {64} replicas against {im[64][1]:.2f} for
request-level routing.

**Sticky routing gets worse as you add replicas** (eq:sessions-pin-to-replicas).
With {2} replicas the excess is {im[2][0] - im[2][1]:.3f}; with {64} it is
{im[64][0] - im[64][1]:.3f}. Each replica added divides the sessions more finely,
and finer division of a heavy-tailed quantity is less even, not more. The usual
response to an overloaded fleet -- add capacity -- therefore works less well than
it should, and the diagnosis is unpleasant to reach because every replica looks
correctly configured.

That is the case for the design cite:mcp2026spec adopted. Not that stateless is
tidier, but that a self-contained request has no state to lose on restart, no
handshake to amortise, no chain to compound, and no affinity to route.""")
