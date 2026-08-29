---
id: mcp-architecture
number: 171
part: XIX
tier: full
status: draft
requires: [connectivity-is-the-real-quantity, loop-is-not-a-chain,
           agent-errors-correlate, replay-needs-idempotence]
provides: [sessions-pin-to-replicas, stateless-removes-the-chain,
           transport-decides-correlation, severity-hides-in-the-mean,
           replication-buys-tail]
citations: [mcp2026spec, hou2025mcp, cemri2025mast, greshake2023indirect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to name MCP's four architectural
roles and say precisely what each is responsible for; explain what changed when
the protocol moved from connection-scoped sessions to self-contained requests,
and price both costs that change removed; explain why sticky routing degrades as
a fleet grows; choose between the two standard transports on the property that
actually differs; and say why two deployments with identical availability can
have opposite failure severity.

## 2. Why This Matters

{{ch:mcp-why}} established that a protocol's value depends on every compliant
client reaching every compliant server. This chapter is about what the protocol
actually is, and its most instructive feature is something it *stopped* doing.

Revisions through `2025-11-25` established a connection-scoped session: the client
sent `initialize`, the pair negotiated once, and state persisted for the
connection's lifetime. The current revision, `2026-07-28`, replaced that with
**stateless, self-contained requests** carrying their own protocol version and
capabilities in a `_meta` field, plus per-request capability negotiation and a
mandatory `server/discover` RPC ({{cite:mcp2026spec}}).

That reads like a design preference. {{sec:9-practical-example}} prices it as an
operational one, and finds two distinct costs removed. A stateful session is a
chain in {{ch:ag-loop}}'s exact sense — completion falls from $100\%$ to $59.8\%$
at a $5\%$ restart rate, and the gap grows from $+1.5$ points at a two-request
session to $+61.4$ at two hundred ({{eq:stateless-removes-the-chain}}). And a
session pinned to a replica must return to it, so the fleet balances *sessions*
rather than requests: peak-to-mean load reaches $4.64$ at sixty-four replicas
against $1.31$ ({{eq:sessions-pin-to-replicas}}). **Sticky routing gets worse as
you add replicas**, which is the opposite of what adding capacity is supposed to
do.

The second half is about the two transports, and it finds the usual comparison —
stdio is convenient locally, HTTP works remotely — missing the property that
matters. stdio runs one server process per client; HTTP shares one. So **the
transport decides whether tool failures are independent or correlated**
({{eq:transport-decides-correlation}}), and {{ch:as-failures}} already established
that correlation is invisible in a mean. Two deployments measured at
$99.60\%$ and $99.59\%$ availability had conditional severities of $5.0\%$ and
$100.0\%$.

## 3. Prerequisites

{{ch:mcp-why}} for the ecosystem argument and for the versioning policy this
chapter's per-request negotiation implements.

{{ch:ag-loop}}'s {{eq:loop-is-not-a-chain}}, which is the distinction the
stateless change turns on — read in reverse here, since a session *is* a chain and
that is the problem.

{{ch:as-failures}}'s {{eq:agent-errors-correlate}}, whose deployment-layer
instance is this chapter's second listing.

{{ch:as-state-machines}}'s {{eq:replay-needs-idempotence}}, since retrying a
self-contained request against a different replica is a replay.

## 4. Intuitive Explanation

MCP names four things, and the naming is worth getting right because two of them
are routinely conflated.

A **host** is the LLM application — the IDE, the chat interface, the agent
runtime. It is what the user interacts with and what decides which tools to offer
the model.

A **client** is a connector *inside* the host, one per server it talks to. A host
with six tool servers has six clients. This is the piece people forget exists,
and it matters because the client is where per-server state, credentials and
version negotiation live.

A **server** provides context and capabilities. It is usually not the tool itself
but an adapter in front of one: a wrapper around an issue tracker's API rather
than the tracker.

The **model** is not a protocol participant at all. It sees tool descriptions the
host chose to show it and emits calls the host executes. Nothing in the protocol
reaches the model directly, which is a fact {{ch:mcp-security}} depends on.

Now the change that makes this chapter interesting.

Originally a connection was a conversation. The client opened it, said hello, the
two agreed on a version and a capability set, and everything afterwards happened
inside that agreement. That is how most protocols work and it is comfortable to
implement.

It has two problems, and both are operational rather than theoretical.

The first is that the session accumulates state, and state can be destroyed. If
the server process restarts halfway through — a deploy, a preemption, a crash —
everything the session established is gone. The client must reconnect,
re-negotiate, and redo whatever the session had accomplished.
{{sec:9-practical-example}} finds this discarding up to $4.14$ completed requests
per session.

That makes a session behave exactly like {{ch:ag-loop}}'s chain: success is the
product of surviving every step, so it compounds, so long sessions are much worse
than short ones. A self-contained request has no such property, because a failed
request is retried without disturbing the ones before it.

The second problem is routing, and it is the one that surprises people. If the
session's state lives in one process, every subsequent request must reach *that*
process. Behind a load balancer that means sticky routing, and stickiness means
the balancer distributes sessions rather than requests.

Sessions are heavy-tailed — a few clients make far more calls than the median —
so distributing sessions distributes a heavy-tailed quantity into a small number
of bins. That is much less even than distributing the individual requests, and it
gets *less* even as you add replicas, because each replica added divides the
sessions more finely. {{sec:9-practical-example}} measures peak-to-mean load
climbing from $1.09$ at two replicas to $4.64$ at sixty-four.

Self-contained requests remove both. Any replica can serve any request; a restart
costs only what was in flight.

Then the transports, of which there are exactly two.

**stdio**: the client launches the server as a subprocess and they exchange
newline-delimited JSON-RPC over its standard streams. Credentials come from the
environment — the spec explicitly says stdio implementations should *not* use its
OAuth flow.

**Streamable HTTP**: each message is an HTTP POST to a single endpoint, and the
reply is either a JSON object or a request-scoped SSE stream.

The usual comparison is convenience, and the real difference is that stdio gives
each client its own process while HTTP shares one. That is a decision about
failure correlation, and {{sec:9-practical-example}} shows the two shapes having
identical means and opposite tails.

## 5. Formal Explanation

Let a client make $n$ requests against a server, with a per-request probability
$q$ that the serving process restarts.

**Stateful.** A restart destroys the session, so the client must re-establish
(cost $c$) and redo everything. Session completion within a budget $B$ requires
surviving a chain:

$$\Pr[\text{complete}] \approx \sum_{k \ge 0} \Pr[k \text{ restarts}] \cdot \mathbb{1}\big[n + k(n\bar{p} + c) \le B\big], \qquad \Pr[\text{no restart}] = (1-q)^n$$ (eq:stateless-removes-the-chain)

The leading term is $(1-q)^n$ — **exponential in the session length**, which is
{{eq:loop-is-not-a-chain}}'s chain read from the other side.

**Stateless.** A restart costs the single in-flight request, retried elsewhere.
The expected cost is $n/(1-q)$ and completion within $B = \beta n$ concentrates
for $\beta > 1/(1-q)$:

$$\Pr[\text{complete}] \longrightarrow 1 \quad\text{independent of } n$$

**The exponent in $n$ disappears entirely**, which is why the gap in
{{sec:9-practical-example}} grows with session length rather than staying fixed.

Now routing. Let $R$ replicas serve $S$ sessions whose request counts $W_i$ are
heavy-tailed with mean $\mu$ and variance $\sigma^2$. Under sticky routing each
replica's load is a sum of roughly $S/R$ session weights:

$$\text{Var}[L_r^{\text{sticky}}] = \frac{S}{R}\sigma^2, \qquad \mathbb{E}[L_r] = \frac{S\mu}{R}$$

so the coefficient of variation is $\sigma\sqrt{R}/(\mu\sqrt{S})$ — **increasing in
$R$**. Under request-level routing each replica sums $S\mu/R$ individual requests
of unit weight:

$$\frac{\text{CV}^{\text{sticky}}}{\text{CV}^{\text{request}}} = \frac{\sigma}{\mu}\cdot\sqrt{\frac{\mu S/R}{S/R}} \;=\; \frac{\sigma}{\sqrt{\mu}}$$ (eq:sessions-pin-to-replicas)

independent of $R$ in ratio but growing in absolute terms, and the peak-to-mean
statistic that matters for provisioning grows as $\sqrt{R\log R}$. **Adding
replicas increases sticky imbalance.**

Now transports. Let $p$ be the probability a server process is unavailable, with
$C$ clients and $M$ servers. Under stdio there are $CM$ independent processes:

$$\mathbb{E}[\text{share of fleet affected} \mid \text{any}] = \frac{1 - (1-p)^{M}}{1 - (1-p)^{CM}} \;\xrightarrow[p \to 0]{}\; \frac{1}{C}$$

Under a shared server, one process serves all $C$ clients, so:

$$\mathbb{E}[\text{share affected} \mid \text{any}] = 1 \quad\text{always}$$ (eq:transport-decides-correlation)

Mean availability is $1 - p$ in *both* cases to first order. The distributions
differ entirely and the mean cannot see it:

$$\mathbb{E}[A]_{\text{stdio}} = \mathbb{E}[A]_{\text{shared}}, \qquad \text{Var}[A_{\text{fleet}}]_{\text{shared}} \gg \text{Var}[A_{\text{fleet}}]_{\text{stdio}}$$ (eq:severity-hides-in-the-mean)

With $R$ replicas the shared server is down only when all are:

$$p_{\text{eff}} = p^R \quad\Longrightarrow\quad \Pr[\text{fleet outage}] = p^R$$ (eq:replication-buys-tail)

which moves the *tail* exponentially while moving the mean by $O(p)$. **Replication
is a tail intervention priced by an availability metric that cannot see tails.**

## 6. Mathematical Foundation

Three extractions.

**Statelessness removes an exponent rather than a constant.** From
{{eq:stateless-removes-the-chain}}, the stateful regime's leading term is
$(1-q)^n$ and the stateless regime's is $O(1)$ in $n$. That is the same structural
difference {{ch:ag-planning}} exploited with checkpoints, and it means the benefit
is unbounded in session length rather than a fixed percentage.

**Sticky imbalance grows as $\sqrt{R \log R}$.** {{eq:sessions-pin-to-replicas}}
predicts the counterintuitive direction, and it is worth carrying because the
usual remedy for an overloaded fleet — add replicas — makes the imbalance worse
while the *mean* utilisation falls. A team watching mean utilisation sees capacity
being added successfully while individual replicas keep saturating.

**Mean availability is blind by construction.** {{eq:severity-hides-in-the-mean}}
says the two deployments are indistinguishable on the metric almost everyone
reports. The measurement that separates them is *conditional severity* — given a
bad interval, what share of the fleet was affected — and it costs nothing to
compute from data you already have.

## 7. Internal Mechanics

### 7.1 The four roles, and what carries what

```mermaid {#fig:mcp-roles caption="MCP's participants. The model is not a protocol participant: it sees what the host chooses to show it, and the host executes what it emits."}
flowchart LR
    U[user] --> H[host: the LLM application]
    H --> LM[model]
    LM -. tool call .-> H
    H --> C1[client 1]
    H --> C2[client 2]
    H --> C3[client 3]
    C1 --> S1[server: issue tracker]
    C2 --> S2[server: filesystem]
    C3 --> S3[server: search]
    S1 --> X1[(tracker API)]
    S3 --> X3[(the web)]
```

One client per server, all inside the host. The host is the only component that
sees everything, which makes it the only place a capability-union check like
{{ch:ag-security}}'s can be performed — a point {{ch:mcp-security}} builds on.

### 7.2 The message model, which is narrower than it was

{{cite:mcp2026spec}} states a constraint worth quoting in effect: a binding must
deliver client-sent *requests* and *notifications* to the server, and server-sent
*responses* and *notifications* to the client — and **no other message direction
exists.** Servers do not initiate JSON-RPC requests; clients do not send
responses.

That is a real narrowing. Earlier revisions let servers initiate requests, which
is what made server-driven sampling and root enumeration possible in the core
protocol. Those moved out, and the sole core client feature is now **Elicitation**
— a server asking the user, through the client, for more information.

The narrowing is what permits the stateless model. A server that can initiate
requests needs a channel back to a specific client, which is a session by another
name.

### 7.3 Server primitives, and who each is for

Three, and the distinction is about *who decides to use them*:

**Tools** — functions the *model* executes. Model-controlled.

**Resources** — context and data for the user or the model. Application-controlled:
the host decides what to include.

**Prompts** — templated messages and workflows for *users*. User-controlled,
typically surfaced as slash commands or menu items.

{{ch:mcp-primitives}} takes this up properly. The point here is that the split is
by controller rather than by data type, and getting it wrong is the most common
server-design error.

### 7.4 The two transports in detail

**stdio.** Newline-delimited JSON-RPC over the standard streams of a
client-launched subprocess. Cancellation is a `notifications/cancelled`
notification. Credentials come from the environment — {{cite:mcp2026spec}} says
stdio implementations SHOULD NOT use the OAuth flow.

**Streamable HTTP.** Each message is an HTTP POST to a single MCP endpoint; the
reply is a JSON object or a request-scoped SSE stream. Cancellation is the client
closing the response stream. Request metadata is mirrored from `_meta` into HTTP
headers — including `MCP-Protocol-Version` — so intermediaries can route and
inspect without parsing the body, **with the body remaining the source of truth**.

That last clause is a security-relevant detail rather than a convenience: a
mismatch between header and body must be rejected, or an intermediary and a server
can be made to disagree about what a request says.

Custom transports are permitted and must preserve the JSON-RPC format, the message
patterns, and the per-request metadata model. Ones running over a reliable
bidirectional byte stream — Unix sockets, TCP — SHOULD reuse the stdio framing,
which is just newline-delimited JSON-RPC with process-lifecycle rules attached.

### 7.5 Extensions, and why they are separate

Beyond the core, {{cite:mcp2026spec}} defines opt-in extensions negotiated by both
sides. Three are named: **Tasks** (asynchronous long-running operations with
polling, mid-flight input and durable handles), **Skills over MCP**, and **MCP
Apps** (inline interactive UI).

Tasks is the one this book has already argued for. {{ch:as-long-running}} found
that long-horizon work needs durable handles and that wall-clock delay is a
first-class cost; an extension providing exactly that, outside the core, is the
right factoring — because a protocol that made every server implement durable
execution would have a much higher $c_s$ in {{ch:mcp-why}}'s arithmetic, for a
capability most servers do not need.

### 7.6 What per-request negotiation costs

Nothing here is free, and the honest accounting has three entries.

**Every request carries its metadata**, so there is a per-request byte cost where a
session amortised it once. At typical tool-call sizes this is small, and it is the
price of the routing property.

**A server cannot cache per-session work**, because there is no session to attach
it to. Servers that need continuity must externalise it — which is
{{ch:as-state-machines}}'s durable-state chapter, and is better done explicitly
than implicitly in process memory.

**Retrying a request against a different replica is a replay**, so
{{eq:replay-needs-idempotence}} applies in full. The protocol makes retries easy
and does not make them safe; that remains a property of the tool.

### 7.7 Choosing a transport

The measured guidance is unremarkable and its condition is not.

Use **stdio** for a single user's machine: the resource comparison favours it below
about five clients, isolation is free, and there is no authorization surface at
all because credentials come from the environment.

Use **Streamable HTTP** for anything shared: past a handful of clients the resource
ratio becomes overwhelming — $16\times$ at a hundred clients,
$82\times$ at five hundred.

The condition is that taking the shared path converts a diffuse failure mode into
a concentrated one. **You owe the fleet replication you did not previously need**,
and {{eq:replication-buys-tail}} says to size it against the tail rather than
against an availability target, because the availability target will be met at one
replica.

### 7.8 What `server/discover` is for

The current revision makes `server/discover` a **mandatory RPC** — every server
must implement it — while making the *call* optional: a client may send any
request directly and handle a version error if one comes back
({{cite:mcp2026spec}}).

That asymmetry is deliberate and worth understanding, because it is
{{ch:mcp-why}}'s negotiation result implemented at the level of a single round
trip. A client that wants to choose a version up front asks once and receives the
server's supported versions, capabilities and identity together. A client that
would rather assume and correct sends its request and, on a mismatch, receives an
`UnsupportedProtocolVersionError` **listing what the server does support** — which
is enough to retry into the overlap.

Both paths reach the same place, and the choice between them is a latency
decision. The discovery call costs a round trip always; the optimistic path costs
nothing when the guess is right and two round trips when it is wrong. For a client
that talks to a stable set of servers, guessing wins; for one that talks to
strangers, discovery does.

Mandating the *implementation* while leaving the *call* optional is what makes
that choice available to the client rather than forced by the server, and it is a
small piece of protocol design worth stealing: **make the capability universal and
its use discretionary**, so the party with the information decides.

It also gives clients somewhere to learn a server's identity before trusting its
tool descriptions, which {{ch:mcp-security}} needs and which the older handshake
folded into a step that had already committed to the connection.

## 8. Implementation

Two listings. The first prices the connection model. The second measures what the
transport choice actually decides.

```python {tier=A name=stateless-removes-the-chain}
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
```

The second listing asks what the transports really differ on.

```python {tier=A name=transport-decides-correlation}
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
```

## 9. Practical Example

The first listing walks client sessions of fourteen requests against a budget:

```
  restart / request  stateful ok  stateless ok  stateful cost  stateless cost
-----------------------------------------------------------------------------
              0.05%        99.5%        100.0%           17.1            14.0
              0.80%        92.7%        100.0%           17.8            14.1
              5.00%        59.8%        100.0%           21.0            14.7
```

Even at the lowest rate the handshake puts the stateful regime $22\%$ above
baseline before anything goes wrong. Wasted work — requests completed inside a
session later lost — reaches $4.14$ per session against $0.00$.

Against session length:

```
  session length  stateful ok  stateless ok      gap
----------------------------------------------------
               2        98.5%        100.0%    +1.5%
              20        91.2%        100.0%    +8.8%
             200        38.6%        100.0%   +61.4%
```

**A stateful session is a chain in {{ch:ag-loop}}'s exact sense**
({{eq:stateless-removes-the-chain}}) — success is the product of surviving every
request. A stateless sequence is not, because a failed request is retried without
disturbing its predecessors.

And when the client has no retry loop, which is the normal case for a tool call
inside an agent step:

```
  restart / request  stateful, retry   no retry   stateless
-----------------------------------------------------------
              0.80%            93.0%      89.4%      100.0%
              5.00%            59.9%      48.5%      100.0%
```

The routing cost:

```
  replicas  sticky (sessions)  any replica   excess
---------------------------------------------------
         2              1.094        1.013   +0.080
         8              1.537        1.069   +0.468
        64              4.640        1.315   +3.325
```

**Sticky routing gets worse as you add replicas**
({{eq:sessions-pin-to-replicas}}) — each replica divides a heavy-tailed quantity
more finely, and finer division of a heavy tail is less even. The usual response to
an overloaded fleet therefore works less well than it should, and every replica
looks correctly configured.

The second listing runs sixty clients against twelve servers:

```
            deployment  availability  processes   memory GB
-----------------------------------------------------------
    stdio (per client)       99.597%        720        59.8
       http, 1 replica       99.621%         12         3.0
      http, 2 replicas      100.000%         24         6.1
```

By that table the shared deployment wins comfortably. The joint view does not
agree:

```
            deployment  availability  any client hit  >50% hit together
-----------------------------------------------------------------------
    stdio (per client)       99.597%           94.2%              0.00%
       http, 1 replica       99.621%            4.4%              4.42%
```

Identical availability. stdio has *someone* affected almost every day and more than
half the fleet affected never; the shared server has a bad day rarely and every bad
day is a bad day for everyone.

Stated directly:

```
  process down rate  stdio avail  stdio severity  http avail  http severity
---------------------------------------------------------------------------
               0.1%       99.90%            2.3%      99.90%         100.0%
               2.0%       98.01%           21.4%      97.97%         100.0%
               8.0%       91.98%           63.4%      92.16%         100.0%
```

**Availability agrees to two decimal places in every row and severity does not**
({{eq:severity-hides-in-the-mean}}). This is {{ch:as-failures}}'s panel result at
the deployment layer: sharing a dependency is what makes failures correlate, and it
does not matter whether the shared thing is a base model or a process.

Replication, and where it acts:

```
  replicas  availability  >50% hit together   memory GB
-------------------------------------------------------
         1      99.5979%             4.750%         3.0
         2      99.9979%             0.025%         6.1
         3     100.0000%             0.000%         9.1
```

A small move in the mean and a factor of about a hundred in the fleet-wide outage
rate. **Replication buys tail, not mean** ({{eq:replication-buys-tail}}) — which is
what a correlated deployment needs and what an availability-driven capacity plan
will under-provision.

And the resource comparison that decides it in practice:

```
  clients  stdio processes   stdio GB  http GB (2x)    ratio
------------------------------------------------------------
        1               12        1.0           6.1     0.16
        5               60        5.0           6.1     0.82
      100            1,200       99.6           6.1    16.35
      500            6,000      498.0           6.1    81.73
```

## 10. Production Considerations

Prefer self-contained requests wherever the protocol offers the choice, and
externalise anything a server genuinely must remember rather than holding it in
process memory.

If you must keep sessions, keep them *short*. The exposure is exponential in
length, so splitting one long session into several short ones is worth more than
making restarts rarer.

Do not add replicas to fix a sticky-routing imbalance. It makes the imbalance
worse; the fix is request-level routing or externalised state.

Report **conditional severity** alongside availability: given a bad interval, what
share of the fleet was affected. It costs nothing and it is the only number that
distinguishes the two deployment shapes.

Size replication against the fleet-outage tail, not against an availability
target. The target is met at one replica.

Choose stdio below roughly five clients and Streamable HTTP above, and when you
take the shared path, budget the replication that path now requires.

Reject requests whose mirrored HTTP headers disagree with the body. The body is
the source of truth and a disagreement is an attack surface.

And treat every retry as a replay — {{eq:replay-needs-idempotence}} does not go
away because the protocol made retrying easy.

## 11. Common Mistakes

**Conflating host and client.** There is one client per server, inside the host,
and it is where version and credential state live.

**Assuming the model is a protocol participant.** It is not; the host mediates
everything, which is what makes host-side policy possible.

**Long sessions.** The exposure is exponential in length.

**Adding replicas to fix sticky imbalance.** Wrong direction.

**Reporting availability alone.** It cannot distinguish a diffuse failure mode from
a concentrated one.

**Treating stdio and HTTP as a convenience choice.** It is a failure-correlation
choice with a convenience side effect.

**Trusting mirrored headers over the body.** The spec says which one wins.

## 12. Failure Modes

*Session loss on deploy.* The most common operational failure of the stateful
model, and it looks like a client bug from the server's side.

*Fleet-wide tool outage.* The concentrated failure mode a shared server
introduces — rare, total, and invisible to an availability SLO.

*Replica saturation under a balanced mean.* {{eq:sessions-pin-to-replicas}}'s
signature: aggregate utilisation looks fine and individual replicas keep falling
over.

*Duplicate effects from transparent retries.* The protocol makes retrying easy;
{{ch:as-state-machines}}'s hazard follows.

*Header/body disagreement.* An intermediary and a server acting on different
versions of the same request.

*Untrusted content over a trusted transport.* A compliant server returning
{{cite:greshake2023indirect}}'s payload — the transport is authenticated and the
content is not.

## 13. Alternatives

**Session affinity with externalised state.** Keep the session concept but store
its state in a shared store, so any replica can serve it. Most of the benefit at
the cost of a store, and it is what the stateless design achieves without the
concept.

**One server process per tenant.** Restores isolation inside an HTTP deployment,
at stdio's resource profile and HTTP's reachability.

**A local proxy fronting remote servers.** The host talks stdio to a local process
that speaks HTTP outward — a common shape that gets stdio's simplicity in the host
and HTTP's economics upstream.

**Direct API integration.** {{ch:mcp-why}}'s bespoke option, still correct for
short-lived integrations.

## 14. Evaluation

Measure your restart rate per request, not per day. It is the parameter both
listings turn on and it is usually unknown.

Measure conditional severity. Availability plus severity is two numbers and
describes the system; availability alone describes neither shape.

Measure peak-to-mean replica load, not mean utilisation, and track it *as you add
replicas* — the trend is the diagnostic.

Test a rolling deploy under load and count sessions lost. This is the failure the
stateless design removes and the only way to know what it is worth to you.

And audit which of your tools are safe to retry, since the transport now makes
retrying the default.

## 15. Advanced Concepts

**Partial-state sessions.** Externalising only the expensive parts of session state
while keeping requests self-contained — the practical middle ground, with no
established pattern. {{maturity:EMERGING}}.

**Severity-aware SLOs.** Service objectives stated over the joint distribution
rather than the mean, which {{eq:severity-hides-in-the-mean}} argues is the only
form that distinguishes deployment shapes. Rare in practice.

**Transport-level provenance.** Carrying {{ch:ag-security}}'s provenance question
in `_meta` so downstream components can reason about where content came from.
{{maturity:RESEARCH FRONTIER}}.

**Formal analysis of the message-pattern restriction.** The current revision's
"no other message direction exists" is strong enough to admit real analysis of
what a server can and cannot cause. Nobody has done it.

## 16. Connection to Previous Chapters

{{ch:ag-loop}}'s {{eq:loop-is-not-a-chain}} is this chapter's central tool, used in
reverse: a session *is* a chain, and removing it removes an exponent.

{{ch:as-failures}}'s correlation result reappears at the deployment layer, with the
shared process playing the role the shared base model played there.

{{ch:as-state-machines}}'s idempotence requirement becomes load-bearing, because
the stateless model makes cross-replica retries routine.

{{ch:as-long-running}}'s durable handles turn up as the Tasks extension, factored
out of the core for the reason {{ch:mcp-why}}'s arithmetic predicts.

Ahead: {{ch:mcp-primitives}} takes up the three server primitives and the
controller distinction sketched in {{sec:7-internal-mechanics}};
{{ch:mcp-security}} takes up the authorization binding and the trust boundary the
host is uniquely placed to enforce.

## 17. Exercises

1. Derive the $\sqrt{R\log R}$ growth in sticky peak-to-mean load and check it
   against the first listing's last table.

2. Add externalised session state to the first listing — a session that survives a
   restart at a lookup cost. How much of the stateless benefit does it recover?

3. Measure the byte overhead of per-request `_meta` against a realistic tool-call
   payload. At what payload size does it stop being negligible?

4. Implement the one-process-per-tenant deployment in the second listing and
   locate its resource and severity position between the two extremes.

5. Compute the replication factor needed to hold fleet-wide outage below one day
   per year at your measured process failure rate.

6. Model a rolling deploy explicitly — replicas restarting in sequence rather than
   independently — and compare with the independent model.

## 18. Interview Questions

1. Name MCP's participants and say which one the model is.

2. Why did the protocol move from sessions to self-contained requests?

3. Your fleet is imbalanced under sticky routing. You add replicas. What happens?

4. Two deployments both report $99.6\%$ availability. What would you ask next?

5. When would you choose stdio over Streamable HTTP?

6. What does the stateless model make easy that you now have to make safe?

## 19. Research Questions

1. What is the right form for an SLO over the joint availability distribution?

2. Can session state be partially externalised with a general pattern rather than
   per-server engineering?

3. What can be proven about server influence given the current revision's
   message-direction restriction?

4. Could provenance be carried at the transport layer usefully, and what would
   consume it?

5. How much of the observed reliability difference between MCP deployments is
   attributable to transport choice rather than to server quality?

## 20. Chapter Summary

MCP names hosts, clients and servers — one client per server, all inside the host
— and pointedly does not name the model, which sees only what the host shows it.

The current revision replaced connection-scoped sessions with **stateless,
self-contained requests** carrying their own version and capabilities, and removed
server-initiated requests from the core ({{cite:mcp2026spec}}). That change is
operational, and it removes two costs.

A stateful session is a chain in {{ch:ag-loop}}'s exact sense: completion fell from
$100\%$ to $59.8\%$ at a $5\%$ restart rate, discarding up to $4.14$ completed
requests per session, with the gap growing from $+1.5$ points at two requests to
$+61.4$ at two hundred ({{eq:stateless-removes-the-chain}}). **Statelessness
removes an exponent, not a constant.**

And a pinned session must return to its replica, so the fleet balances sessions
rather than requests. Peak-to-mean load rose from $1.09$ at two replicas to $4.64$
at sixty-four ({{eq:sessions-pin-to-replicas}}) — **sticky routing gets worse as
you add replicas**, so the usual remedy for an overloaded fleet is the wrong one.

On transports, stdio gives each client its own process and Streamable HTTP shares
one, so **the transport decides whether failures are independent or correlated**
({{eq:transport-decides-correlation}}). Two deployments measured $99.60\%$ and
$99.59\%$ availability with conditional severities of $5.0\%$ and $100.0\%$:
**availability cannot see the difference** ({{eq:severity-hides-in-the-mean}}).
Replication fixes it, moving fleet-wide outage from $4.750\%$ to $0.025\%$ while
moving the mean barely at all — **replication buys tail**
({{eq:replication-buys-tail}}), which is exactly what an availability target will
under-provision.

The resource crossover sits near five clients. Below it, stdio; above it, HTTP —
and above it you owe the fleet replication that the diffuse failure mode did not
require.

## 21. Further Reading

{{cite:mcp2026spec}} is the primary source, and its transports and versioning pages
repay direct reading — particularly the statement that no message direction exists
beyond client-request and server-response, which is what makes the stateless model
possible.

{{ch:ag-loop}} for the chain distinction this chapter applies in reverse, and
{{ch:as-failures}} for the correlation result the second listing instantiates.

{{cite:hou2025mcp}} for the deployment-phase threats this chapter's architecture
creates surface for, and {{cite:cemri2025mast}} for the system-design failure
category these operational modes belong to.

{{ch:as-state-machines}} for the idempotence requirement that the stateless model
makes routine rather than occasional.
