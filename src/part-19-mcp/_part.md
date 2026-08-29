---
id: part-19-intro
status: final
---

## What this part is for

{{part:17}} gave an agent tools. {{part:18}} built systems out of agents. Neither
asked where the tools come from, and that turns out to decide more than either.

**The hazard here is that a protocol looks like plumbing.** It has a
specification, the specification has RFCs, and the natural response is to reach
for an SDK and stop reading. This part is written against the wire format instead,
because a library hides exactly the three things that were worth measuring: the
version window, the statelessness, and what a tool description carries.

> **A warning about currency.** The Model Context Protocol's current revision is
> **2026-07-28**, and it is architecturally different from the one most written
> material describes. Requests are now **stateless and self-contained**; version
> and capability negotiation happen **per request** rather than in an `initialize`
> handshake; `server/discover` is a mandatory RPC; and **servers no longer
> initiate JSON-RPC requests**, which narrowed the core client feature set to
> Elicitation alone. Anything written against the handshake model is describing a
> different protocol. Every claim in this part was checked against the
> specification on 2026-08-29.

## The organising idea

**A protocol is a mechanism for making the marginal integration cost constant.**
Everything else in this part follows from what that costs and what it does not
cover.

```text
   CHAPTER                   THE DECISION IT OWNS       WHAT DECIDES IT
   ───────────────────────   ────────────────────────   ─────────────────────────
   170 why protocols         adopt or hand-roll         how long integrations live
   171 architecture          transport and connection   failure correlation
   172 primitives            tool, resource, or prompt  volatility, then selection
   173 schemas and budgets   what reaches the context   inventory size
   174 security              what a token may reach     registry policy
   175 building              what your server owes      error information
   176 production            how many servers           marginal coverage
```

The through-line: **almost nothing in this part is decided by the protocol.** The
protocol fixes the wire; the decisions are about volatility, correlation, context
budget, blast radius, and admission policy — every one of which is a measurement
about your situation rather than a clause in a specification.

**And a second through-line, which by now is the book's.** Four separate chapters
here found a structural control beating a vigilant one:

| Chapter | The vigilant thing | What beat it |
|---|---|---|
| {{ch:mcp-primitives}} | fetching fresh every time | re-reading on a schedule |
| {{ch:mcp-schemas}} | showing more tools for recall | showing fewer |
| {{ch:mcp-security}} | scanning tool metadata | partitioning capability |
| {{ch:mcp-production}} | staffing a review queue | requiring a signed identity |

That is one claim four times: **a control priced per unit of design scales, and a
control priced per unit of volume does not**, because the volume is set by someone
else. {{part:17}} and {{part:18}} said the same thing about agents; this part says
it about ecosystems.

## Ten things worth knowing before you start

**The $N \times M$ argument is a maintenance argument, not a build argument.** On
build cost alone, a small ecosystem *loses*: $72$ engineer-days bespoke against
$144$ for a protocol at two hosts and three providers. Over three years at ten by
sixty, maintenance is $72\%$ of the bespoke cost and $15\%$ of the protocol's. The
right question is how long the integrations will live, not how many there are.

**What actually drives adoption is the marginal joiner.** A provider entering an
ecosystem of a hundred hosts pays $4{,}350$ days bespoke and $31$ with a protocol —
the same $31$ as joining an ecosystem of one. Entry cost goes from $O(N)$ to
$O(1)$, and that is the calculation each participant actually performs.

**Connectivity is governed by a threshold, not a rate.** A lagging ecosystem with
four-revision support windows reached $73.2\%$ interoperability against a current
one with single-revision windows at $52.8\%$. And negotiation is what makes a
window reachable: an eight-revision window that cannot be advertised loses to a
three-revision window that can.

**A stateful session is a chain and a stateless request is not.** Completion fell
from $100\%$ to $59.8\%$ at a $5\%$ restart rate, with the gap growing from $+1.5$
points at two requests to $+61.4$ at two hundred. And sticky routing gets *worse*
as you add replicas — peak-to-mean load rose from $1.09$ at two to $4.77$ at
sixty-four.

**The transport decides whether failures correlate.** Two deployments measured
$99.60\%$ and $99.59\%$ availability, with conditional severities of $5.0\%$ and
$100.0\%$. Availability cannot see the difference; the number that can is
conditional severity, and nobody reports it.

**Freshness decides the primitive before anything else does.** Above about $1\%$
change per step, preloaded resources collapse — $100\%$ to $22.5\%$ — while fetched
tools do not move. Below it, weak selection pushes toward resources and
concentrated demand pushes toward a *smaller* preloaded set, not a larger one.

**Tool count is nearly free for selection and ruinous for context.** From eight to
two thousand tools, selection loss moved $9.0\% \to 13.5\%$ and dilution loss moved
$3.9\% \to 99.9\%$. Retrieval starts winning between sixteen and sixty-four tools —
one or two servers, not a marketplace.

**More context is not monotonically better.** Success peaked near $24{,}000$ tokens
and fell to $9.7\%$ at $160{,}000$, because component benefits saturate and dilution
does not. The optimal split also moves with the budget, so a configuration tuned
under scarcity is wrong for a large window.

**Token passthrough is quadratic in connected servers.** From two servers to
eighty, passthrough exposure grew $0.5 \to 384.9$ against audience binding's
$0.2 \to 9.6$. That is why the specification says MUST rather than SHOULD — and
only *scope* minimisation lowers the ceiling, from $54$ reachable scopes to $18$.

**The marginal server turns negative earlier than anyone expects.** Success peaked
at eight connected servers without a retrieval layer and sixteen with one; the
seventeenth added $0.9\%$ of coverage against $0.84\%$ of exposure. A server that
would newly enable under about one percent of your tasks costs more than it brings.

## What this part deliberately does not cover

**SDK usage.** {{ch:mcp-building}} implements the protocol directly, on the view
that you should know what a library decides on your behalf before letting it. For
production work, use the SDK.

**Every RFC in the authorization chain.** {{ch:mcp-security}} covers the
requirements that bound blast radius — audience binding, resource indicators, scope
step-up, `iss` validation — and treats PKCE, client registration and metadata
discovery as conventional OAuth to be got right with a library.

**Alternative protocols.** There are others, and the arithmetic in
{{ch:mcp-why}} applies to any of them. This part uses MCP because it has a current,
public, checkable specification.

**Prompt-level tool design.** {{ch:ag-tool-calling}}'s, and it survives the
protocol entirely: a protocol will happily carry a tool whose name is ambiguous and
whose errors say nothing.

## How to read it

{{ch:mcp-why}} and {{ch:mcp-architecture}} are the foundation, and the second is
the one to read carefully if you have used MCP before — the connection model
changed.

{{ch:mcp-primitives}} and {{ch:mcp-schemas}} are one argument about context split
in two: what to put in it, and how much. Read them together, because the resource
question and the schema question compete for the same budget.

{{ch:mcp-security}} and {{ch:mcp-production}} are also one argument. The first
finds that every defence scales against parameters it cannot observe; the second
finds where those parameters are set. Reading the first alone leaves you defending
against a rate somebody else chose.

{{ch:mcp-building}} can be read at any point and is the fastest way to make the
rest concrete — it is a complete server and client in two listings.

> **One thing to notice on a second reading**: {{ch:mcp-why}} finds a support
> window worth little without negotiation, {{ch:mcp-schemas}} finds a large
> inventory worth little without retrieval, {{ch:mcp-security}} finds audience
> binding worth little without scope minimisation, and {{ch:mcp-production}} finds
> a registry policy worth little without provenance. **All four are the same
> shape**: a capability and the mechanism that makes it reachable are one design,
> and shipping either alone wastes most of it.
