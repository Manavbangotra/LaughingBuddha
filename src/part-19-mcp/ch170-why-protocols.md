---
id: mcp-why
number: 170
part: XIX
tier: full
status: draft
requires: [distinctness-not-count, error-message-as-selector,
           specialization-is-affordance-building]
provides: [maintenance-dominates-integration, protocol-makes-entry-constant,
           connectivity-is-the-real-quantity, support-window-beats-upgrade-pressure,
           negotiation-unlocks-the-window, breaking-means-breaking]
citations: [mcp2026spec, qin2023toolllm, patil2023gorilla, li2023apibank,
            schick2023toolformer, hou2025mcp]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the $N \times M$ argument
correctly — which is not as a build-cost argument — and identify the term that
actually drives it; compute a break-even ecosystem size from your own numbers;
explain why protocols get adopted before they are economically break-even;
define an ecosystem's connectivity and name the three policies that determine it;
and say why multi-version support and version negotiation are one mechanism
rather than two.

## 2. Why This Matters

{{part:17}} treated tools as things an agent has. {{part:18}} treated them as
things a system is built around. Neither asked where they come from, and the
answer determines what an agent can do more than any architecture in the previous
two parts.

The standard argument for a tool protocol is that $N$ hosts and $M$ tool providers
need $N \times M$ bespoke integrations, which a protocol reduces to $N + M$. That
argument is right about the shape and, as usually stated, wrong about the
economics — {{sec:9-practical-example}} finds a small ecosystem where the protocol
*loses* on build cost, $72$ engineer-days against $144$.

The reason it wins anyway is the term the standard argument leaves out. An
integration is not built once; it is repaired every time either side changes, and
there are $N \times M$ of them to repair. Over three years at ten hosts and sixty
providers, maintenance is $72\%$ of the bespoke regime's cost and $15\%$ of the
protocol's ({{eq:maintenance-dominates-integration}}).

That reframing matters because it explains the timing. The same ecosystem that
favours bespoke by $39$ days at year zero favours the protocol by $659$ days at
year four. **Almost the entire case is made by the second year** — which is also
why it is so hard to make in advance.

And there is a stronger claim available that has nothing to do with totals. A new
provider joining an ecosystem of a hundred hosts pays $4{,}350$ days to integrate
bespoke and $31$ days to implement a protocol — the same $31$ it would pay to join
an ecosystem of one. **A protocol turns the marginal joiner's cost from $O(N)$
into $O(1)$** ({{eq:protocol-makes-entry-constant}}), and the marginal joiner is
the party actually deciding.

The second half of the chapter asks what a protocol must get right to deliver any
of this, and finds the answer is versioning policy — where the implementer's
lever beats the ecosystem's, and where a support window is worth almost nothing
without negotiation to reach it.

## 3. Prerequisites

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}} and
{{eq:error-message-as-selector}} — this part is about where tools come from, and
that chapter is about what makes them usable once they arrive.

{{ch:as-specialized}}'s {{eq:specialization-is-affordance-building}}, since a
protocol is an affordance-distribution mechanism: it is how a verifier, an undo,
or a structured observation built by one team reaches another.

Some familiarity with JSON-RPC and with HTTP is assumed from
{{ch:mcp-architecture}} onward, though this chapter needs neither.

## 4. Intuitive Explanation

Before there was a protocol, connecting a model to a tool meant writing an
adapter. Your assistant needed to read the issue tracker, so someone wrote code
that knew both your assistant's tool-calling format and the tracker's API.

That works. It works for the second integration too. The trouble starts when you
notice that the person building a different assistant wrote their own adapter to
the same tracker, and the tracker's own team wrote a third for a third assistant,
and none of the three can be shared because each is shaped to one host's
conventions.

Now the tracker ships a breaking API change. Three adapters break. Nobody who
maintains them works for the tracker, so they break silently, and each is repaired
separately by someone reading the same changelog.

That is the $N \times M$ problem, and the usual way of drawing it — a grid of
lines between hosts and providers — makes it look like a construction problem.
{{sec:9-practical-example}} says it is not. Building $N \times M$ adapters is
expensive but finite; *maintaining* them is a recurring cost proportional to the
same product, and after two years it dwarfs the construction.

A protocol changes what the changelog affects. The tracker publishes one server;
each host implements one client. When the tracker's API changes, one server is
repaired by the people who made the change. When a host changes its internals,
nothing outside it notices.

The build cost of that is genuinely higher per endpoint — a compliant server is
more work than a purpose-built adapter, and {{sec:9-practical-example}} finds a
three-by-six ecosystem where the protocol starts behind. It catches up in about
six months.

But the argument that actually moves people is different and simpler. If you are
the tracker deciding whether to support AI assistants, the bespoke world asks you
to write an integration per host — and the cost of that grows as the ecosystem
you want to join gets more attractive. **The better the ecosystem, the more it
costs to enter it.** With a protocol you implement once, and your cost is the same
whether there is one host or a thousand.

Which is why protocols spread before anyone computes an ecosystem total. Nobody
is in a position to compute one. Every individual decision is made against an
$O(N)$ alternative, and every individual decision comes out the same way.

Then the harder half. A protocol only delivers $N + M$ if every compliant client
can actually talk to every compliant server, and that fails whenever the two are
at incompatible revisions. Ecosystems manage this by pressuring implementers to
stay current, which is slow and adversarial and never finishes.
{{sec:9-practical-example}} finds that pressure is the weaker lever: a lagging
ecosystem where implementations accept four revisions each connects better than a
current one where they accept only one.

## 5. Formal Explanation

Let there be $n$ hosts and $m$ providers, with per-integration build cost $b$,
per-repair cost $f$, and breaking-change rates $\lambda_h$, $\lambda_p$ per year.
Over $T$ years the bespoke regime costs:

$$C_{\text{bespoke}} = nmb + T\,nm(\lambda_h + \lambda_p)\,f$$ (eq:maintenance-dominates-integration)

Both terms are $O(nm)$, but the second grows without bound in $T$. Its share is:

$$\frac{T(\lambda_h+\lambda_p)f}{b + T(\lambda_h+\lambda_p)f} \;\longrightarrow\; 1$$

so for any ecosystem observed long enough, **the integration cost is a maintenance
cost.** The protocol regime replaces the product with a sum:

$$C_{\text{proto}} = n(c_c + s) + m(c_s + s) + T\lambda_{\text{spec}}(n+m)f_p$$

with $c_c, c_s$ the client and server implementation costs, $s$ the per-team cost
of learning the spec, and $\lambda_{\text{spec}}$ the rate of breaking spec
revisions. Setting these equal gives a break-even that depends on $T$, which is
the point: **at $T=0$ the protocol can lose, and the break-even $m$ falls sharply
as $T$ grows.**

Now the marginal joiner. A provider entering an ecosystem of $n$ hosts pays:

$$C_{\text{join}}^{\text{bespoke}} = n\big(b + T(\lambda_h+\lambda_p)f\big) = O(n), \qquad C_{\text{join}}^{\text{proto}} = c_s + s + T\lambda_{\text{spec}}f_p = O(1)$$ (eq:protocol-makes-entry-constant)

**The protocol's effect on the individual decision is asymptotically stronger than
its effect on the total**, and the individual decision is the one that gets made.

Now connectivity. Let revisions arrive at rate $\rho$ per year, each
implementation accept a window of $w$ consecutive revisions, and adoption lag be
distributed with mean $\ell$. Writing $t_h, t_s$ for the newest revision each side
accepts, a pair interoperates iff their accepted intervals intersect:

$$\text{connected} \iff \max(t_h - w + 1,\; t_s - w + 1) \le \min(t_h,\; t_s)$$

which simplifies to $|t_h - t_s| \le w - 1$. Since $t_h - t_s$ has spread $O(\rho
\ell)$:

$$\Pr[\text{connected}] \approx \Pr\big[|\Delta| \le w-1\big], \qquad \Delta \sim \text{spread } \rho\ell$$ (eq:connectivity-is-the-real-quantity)

Three levers appear and they are not equal. $w$ enters as a *hard threshold* the
difference must fall under; $\rho$ and $\ell$ enter only through the *scale* of
that difference. Widening $w$ past the spread takes connectivity to one
regardless of $\rho\ell$; shrinking $\ell$ never does, because the spread is
never zero:

$$\lim_{w \to \infty}\Pr[\text{connected}] = 1 \quad\text{for any }\rho,\ell; \qquad \lim_{\ell \to 0}\Pr[\text{connected}] < 1 \quad\text{for } w = 1$$ (eq:support-window-beats-upgrade-pressure)

Finally, negotiation. Without it, a client offers only its newest revision $t_h$
and the server accepts iff $t_h$ falls in the server's window — an *asymmetric*
condition that uses the client's window not at all:

$$\Pr[\text{connected}]_{\text{no-neg}} = \Pr[t_s - w + 1 \le t_h \le t_s]$$

which is at most half the two-sided condition and does not improve with the
client's $w$. So:

$$\frac{\partial}{\partial w}\Big(\Pr_{\text{neg}} - \Pr_{\text{no-neg}}\Big) > 0$$ (eq:negotiation-unlocks-the-window)

**Negotiation's value increases with the window, because negotiation is what makes
the window reachable.** The two are one mechanism.

One more policy is implicit in $\rho$. If version numbers increment on every
release, $\rho$ is the release rate; if they increment only on breaking change,
$\rho$ is the breaking-change rate, which is much smaller:

$$\rho_{\text{breaking}} \ll \rho_{\text{release}}$$ (eq:breaking-means-breaking)

and this costs nothing but discipline. {{cite:mcp2026spec}} makes exactly this
choice: version strings are dates, and the date advances only when backwards
compatibility breaks.

## 6. Mathematical Foundation

Three extractions.

**The break-even is a function of time, not of size.** Teams debate whether their
ecosystem is "big enough for a protocol", which is the wrong question because the
break-even $m$ collapses as $T$ grows —
{{sec:9-practical-example}} finds it going from *never* at year zero to three
providers at year three for a single host. The right question is how long the
integrations will live.

**Window is a threshold and lag is a scale.** The asymmetry in
{{eq:support-window-beats-upgrade-pressure}} is the whole reason the cross-table
comes out as it does. A threshold parameter can saturate a probability; a scale
parameter can only shrink it toward a floor. That distinction generalises well
past protocols.

**Negotiation halves a condition.** Without negotiation the compatibility test is
one-sided — the client's flexibility is unused. That is why
{{sec:9-practical-example}} finds an eight-revision window without negotiation
losing to a three-revision window with it. A capability you cannot advertise is
not a capability.

## 7. Internal Mechanics

### 7.1 What a protocol standardises, and what it cannot

```mermaid {#fig:protocol-layers caption="What a tool protocol fixes and what it leaves to the parties. The layers it standardises are exactly the ones that were being rewritten per pair."}
flowchart TD
    A["transport and framing<br/>(standardised)"] --> B["message shape: JSON-RPC<br/>(standardised)"]
    B --> C["discovery: what tools exist<br/>(standardised)"]
    C --> D["schemas: argument types<br/>(standardised shape, free content)"]
    D --> E["semantics: what the tool MEANS<br/>(NOT standardised)"]
    E --> F["quality: does it work well<br/>(NOT standardised)"]
```

The top four layers are what every bespoke adapter re-implemented, which is why
standardising them removes the quadratic term. The bottom two are not addressed by
any protocol and are where {{ch:ag-tool-calling}}'s findings still apply in full:
a protocol will happily carry a tool whose name is ambiguous, whose arguments are
under-constrained, and whose errors say nothing.

**A protocol removes the integration cost and leaves the design cost**, which is
worth saying plainly because the two get conflated in adoption arguments.

### 7.2 Why the tool inventory got large enough to need this

{{cite:qin2023toolllm}} built over sixteen thousand real APIs across forty-nine
categories, and {{cite:li2023apibank}} covered 2,138 APIs across a thousand
domains. Those numbers are the reason a protocol became necessary rather than
merely tidy: at a handful of tools, bespoke integration is obviously fine, and the
regime those datasets describe is not that one.

{{cite:patil2023gorilla}} supplies the other half of the argument. It found that
pairing generation with a *document retriever* both reduced API hallucination and
let the system track documentation changes without retraining — which is to say
that API knowledge has a freshness property, and baked-in knowledge goes stale.
**A discovery mechanism is not a convenience; it is the alternative to being
wrong about signatures.**

### 7.3 The three versioning policies, concretely

{{sec:9-practical-example}}'s three levers map onto three decisions a protocol
makes, and {{cite:mcp2026spec}} makes all three in the direction the measurements
favour:

**Increment only on breaking change.** MCP versions are dates, and the date moves
only when backwards compatibility breaks
({{eq:breaking-means-breaking}}). This shifts an ecosystem several columns left in
the first table at zero cost.

**Let implementations support several revisions at once.** The spec says clients
and servers MAY support multiple protocol versions simultaneously — the window
lever, which the measurements make the dominant one.

**Make the window reachable.** An unsupported version returns an
`UnsupportedProtocolVersionError` *listing the versions the server does support*,
so the client can retry into the overlap rather than merely failing. That is
{{eq:negotiation-unlocks-the-window}}, and pairing it with the previous choice is
what makes either worth having.

### 7.4 Per-request negotiation and why it changes the picture

The 2026-07-28 revision of MCP declares the protocol version *per request*, in a
`_meta` field, rather than once per connection at handshake time
({{cite:mcp2026spec}}). On Streamable HTTP the same value is mirrored into an
`MCP-Protocol-Version` header so intermediaries can route without parsing the
body, with the body remaining the source of truth.

This is a bigger change than it sounds. Under a connection-scoped handshake, a
version mismatch fails the whole session, and a client that supports several
revisions must decide which to offer before it knows anything. Per-request
declaration means the negotiation in
{{eq:negotiation-unlocks-the-window}} happens at the granularity where the
information exists, and a client may hold a mixed population of servers without
tracking session state for each.

It also means a server accepts or rejects each request independently, which is why
the same revision could make requests *stateless and self-contained* — a change
{{ch:mcp-architecture}} takes up in full.

### 7.5 The registry problem the protocol does not solve

A protocol says how to talk to a server you have. It does not say how you found
it, whether its author is who they claim, or whether its tool descriptions are
honest.

{{cite:hou2025mcp}} makes this concrete by decomposing the server lifecycle into
four phases — creation, deployment, operation, maintenance — across sixteen
activities, and finding threats distributed across all of them. Several of the
named ones, including installer spoofing, occur **before any protocol message is
exchanged**, which is precisely why protocol-level defences do not address them.

So the $N + M$ arithmetic assumes a trust relationship it does not supply.
{{ch:mcp-security}} and {{ch:mcp-production}} are about that gap, and it is the
main respect in which "we adopted a protocol" is an incomplete answer.

### 7.6 Why this is the Language Server Protocol argument

{{cite:mcp2026spec}} states outright that MCP takes inspiration from the Language
Server Protocol, and the analogy is exact enough to be useful as a prediction.

LSP faced the same product: $N$ editors, $M$ languages, and a quadratic number of
plugins each maintained by whoever cared enough. It resolved the same way, on the
same economics — and the observable consequences are worth expecting here. Editor
support for obscure languages became normal, because the marginal cost of adding
one fell to $O(1)$. Language tooling quality became decoupled from editor
popularity. And a long tail appeared that had previously been uneconomic.

The prediction for tool ecosystems is the same: **the visible effect of a protocol
is not that existing integrations get cheaper, but that integrations nobody would
have built start existing.** That is a tail-coverage argument, and
{{ch:ag-what-is-an-agent}} already established that the tail is where the value is.

### 7.7 When not to adopt a protocol

The measurements support a real exception rather than a rhetorical one.

If $T$ is genuinely short — a prototype, a one-off migration, an integration you
expect to delete — the maintenance term never accrues and
{{eq:maintenance-dominates-integration}} does not bite.
{{sec:9-practical-example}}'s year-zero table is that case, and it favours bespoke
at small $n \times m$.

If you control both sides and neither changes independently, the $\lambda$ terms
are yours to schedule, which removes most of the recurring cost.

And if you need something the protocol does not express, complying with it is
overhead on top of the bespoke work rather than a replacement for it.

What does *not* justify skipping it: "we only have three integrations". Three
integrations that live three years are past break-even in this model, and the
count is the wrong variable.

## 8. Implementation

Two listings. The first prices the $N \times M$ argument over time and finds the
break-even. The second measures what a protocol must get right to deliver it.

```python {tier=A name=maintenance-dominates-integration}
"""The N x M argument, priced properly.

The case for a tool protocol is always drawn the same way: N hosts and M tool
providers need N x M bespoke integrations, and a protocol turns that into N + M.

That picture is right about the shape and wrong about the economics, because it
compares BUILD costs and the build cost is not what dominates. An integration is
not built once; it is maintained against changes on both sides for as long as it
exists (eq:maintenance-dominates-integration).

This listing prices both regimes over time and finds the break-even.
"""
import numpy as np

# Costs in engineer-days. These are the model's assumptions, stated plainly so
# the conclusions can be checked against your own numbers.
BESPOKE_BUILD = 12.0     # one host <-> one provider, from scratch
BESPOKE_FIX = 3.0        # repairing one integration after a change on either side
PROTO_SERVER = 18.0      # implementing a compliant server: more than one adapter
PROTO_CLIENT = 25.0      # implementing a compliant client: more still
PROTO_FIX = 2.0          # repairing one endpoint after a spec revision
SPEC_LEARN = 8.0         # per team, one-time cost of learning the protocol

HOST_CHANGES = 1.4       # breaking changes per host per year
PROV_CHANGES = 2.1       # breaking changes per provider per year
SPEC_REVS = 0.8          # backwards-incompatible spec revisions per year


def bespoke(n, m, years):
    """Every host-provider pair is its own integration, and every change on
    either side breaks the pairs that touch it."""
    build = n * m * BESPOKE_BUILD
    # A host change breaks that host's m integrations; a provider change breaks
    # that provider's n integrations.
    breaks = years * (n * HOST_CHANGES * m + m * PROV_CHANGES * n)
    return build + breaks * BESPOKE_FIX


def protocol(n, m, years):
    """Each side implements the protocol once. A change on one side is absorbed
    by the protocol rather than propagated; only spec revisions touch everyone."""
    build = n * PROTO_CLIENT + m * PROTO_SERVER + (n + m) * SPEC_LEARN
    revs = years * SPEC_REVS * (n + m)
    return build + revs * PROTO_FIX


print("Integration cost in engineer-days. Bespoke: every host-provider pair is")
print("its own adapter. Protocol: each side implements the protocol once.")
print()
print(f"{'hosts x providers':>19}{'bespoke':>11}{'protocol':>11}"
      f"{'ratio':>9}{'winner':>10}")
print("-" * 60)
Y = 3.0
grid = {}
for n, m in ((2, 3), (3, 8), (5, 20), (10, 60), (20, 200)):
    b = bespoke(n, m, Y)
    p = protocol(n, m, Y)
    grid[(n, m)] = (b, p)
    print(f"{f'{n} x {m}':>19}{b:>11,.0f}{p:>11,.0f}{b / p:>9.1f}"
          f"{('protocol' if p < b else 'bespoke'):>10}")

print()
print()
print("The same ecosystems at year zero -- build cost only, before anything has")
print("had to be maintained.")
print()
print(f"{'hosts x providers':>19}{'bespoke':>11}{'protocol':>11}"
      f"{'ratio':>9}{'winner':>10}")
print("-" * 60)
zero = {}
for n, m in ((2, 3), (3, 8), (5, 20), (10, 60), (20, 200)):
    b = bespoke(n, m, 0.0)
    p = protocol(n, m, 0.0)
    zero[(n, m)] = (b, p)
    print(f"{f'{n} x {m}':>19}{b:>11,.0f}{p:>11,.0f}{b / p:>9.1f}"
          f"{('protocol' if p < b else 'bespoke'):>10}")

print()
print()
print("How the verdict moves with time, for a small ecosystem where the build")
print("cost initially favours bespoke.")
print()
N0, M0 = 3, 6
print(f"{'years':>7}{'bespoke':>11}{'protocol':>11}{'advantage':>12}")
print("-" * 41)
tm = {}
for y in (0.0, 0.5, 1.0, 2.0, 4.0):
    b = bespoke(N0, M0, y)
    p = protocol(N0, M0, y)
    tm[y] = (b, p)
    print(f"{y:>7.1f}{b:>11,.0f}{p:>11,.0f}{b - p:>+12,.0f}")

print()
print()
print("Break-even ecosystem size, as the number of providers at a fixed host")
print("count. The smallest m at which the protocol is cheaper.")
print()
print(f"{'hosts':>7}{'break-even m, year 0':>22}{'year 1':>10}{'year 3':>10}")
print("-" * 49)
be = {}
for n in (1, 2, 3, 5, 10):
    row = []
    for y in (0.0, 1.0, 3.0):
        m = 1
        while m < 10000 and protocol(n, m, y) >= bespoke(n, m, y):
            m += 1
        row.append(m if m < 10000 else None)
    be[n] = row
    cells = "".join(f"{(str(v) if v else 'never'):>{w}}"
                    for v, w in zip(row, (22, 10, 10)))
    print(f"{n:>7}{cells}")

print()
print()
print("What each regime is actually paying for, at 10 x 60 over three years.")
print()
n, m, y = 10, 60, 3.0
b_build = n * m * BESPOKE_BUILD
b_maint = bespoke(n, m, y) - b_build
p_build = n * PROTO_CLIENT + m * PROTO_SERVER + (n + m) * SPEC_LEARN
p_maint = protocol(n, m, y) - p_build
print(f"{'':>12}{'build':>12}{'maintenance':>14}{'maint share':>14}")
print("-" * 52)
print(f"{'bespoke':>12}{b_build:>12,.0f}{b_maint:>14,.0f}"
      f"{b_maint / (b_build + b_maint):>14.0%}")
print(f"{'protocol':>12}{p_build:>12,.0f}{p_maint:>14,.0f}"
      f"{p_maint / (p_build + p_maint):>14.0%}")

print()
print()
print("And who pays. In the bespoke regime someone must own each pair; under a")
print("protocol each party implements once. Cost borne by a single NEW provider")
print("joining an ecosystem that already has n hosts:")
print()
print(f"{'hosts already present':>23}{'bespoke':>11}{'protocol':>11}")
print("-" * 45)
who = {}
for n in (1, 3, 10, 30, 100):
    b = n * BESPOKE_BUILD + Y * n * (HOST_CHANGES + PROV_CHANGES) * BESPOKE_FIX
    p = PROTO_SERVER + SPEC_LEARN + Y * SPEC_REVS * PROTO_FIX
    who[n] = (b, p)
    print(f"{n:>23}{b:>11,.0f}{p:>11,.0f}")

print(f"""
The first table is the argument as it is usually made, and it is correct: at
{20} x {200} the protocol costs {grid[(20, 200)][1] / grid[(20, 200)][0]:.0%} of
bespoke. The second table is the same ecosystems with the maintenance term
removed, and it disagrees at the small end.

At {2} x {3}, building bespoke adapters costs {zero[(2, 3)][0]:,.0f} days against
the protocol's {zero[(2, 3)][1]:,.0f}. At {3} x {8} it is still
{zero[(3, 8)][0]:,.0f} against {zero[(3, 8)][1]:,.0f}. **For a small ecosystem the
protocol loses on build cost**, and it loses for an unsurprising reason: a
compliant server is more work than a single-purpose adapter, and a compliant
client is more work still.

So the N x M argument is not really a build-cost argument, and stating it as one
invites a correct objection from anyone with three integrations to write.

The third table shows what it actually is. The same {N0} x {M0} ecosystem that
favours bespoke by {tm[0.0][1] - tm[0.0][0]:,.0f} days at year zero favours the
protocol by {tm[1.0][0] - tm[1.0][1]:,.0f} days after one year and
{tm[4.0][0] - tm[4.0][1]:,.0f} after four.

The decomposition table says why. Over three years at {10} x {60}, maintenance is
{b_maint / (b_build + b_maint):.0%} of the bespoke regime's cost and
{p_maint / (p_build + p_maint):.0%} of the protocol's.

**The N x M problem is a maintenance problem wearing a build problem's clothes**
(eq:maintenance-dominates-integration). A bespoke adapter has to be repaired
whenever either side changes, and there are N x M of them to repair; a protocol
endpoint is repaired only when the PROTOCOL changes, and there are N + M of them.
The quadratic term is in the maintenance, not the construction.

The break-even table makes this concrete and slightly startling. At year zero a
single host needs an ecosystem the model never reaches before the protocol pays
off. At three years, one host breaks even at {be[1][2]} providers and ten hosts
break even at {be[10][2]}.

**Almost the entire case for a protocol is made by the second year**, which is
also why the case is so hard to make in advance: the costs it avoids have not
happened yet, and the costs it imposes are due immediately.

The last table is the mechanism behind adoption, and it is not about totals at
all. A new provider joining an ecosystem with {100} hosts pays
{who[100][0]:,.0f} days to integrate bespoke and {who[100][1]:,.0f} days to
implement the protocol -- and that {who[100][1]:,.0f} is the SAME number it would
pay to join an ecosystem with one host.

**A protocol turns the marginal joiner's cost from O(N) into O(1)**
(eq:protocol-makes-entry-constant). That is a different claim from the total-cost
claim and a much stronger one, because the marginal joiner is the party deciding
whether to participate. An ecosystem can be below its total-cost break-even and
still adopt a protocol enthusiastically, because every individual decision to
join is made against the O(N) alternative.

Which is the honest summary of why these protocols spread the way they do. Not
because someone computed the ecosystem total -- nobody is in a position to -- but
because each participant faced a constant cost instead of a growing one.""")
```

The second listing asks what makes the protocol actually work.

```python {tier=A name=connectivity-is-the-real-quantity}
"""What a protocol has to get right to actually deliver N + M.

The previous listing assumed the protocol works: every compliant client talks to
every compliant server. That assumption is doing a great deal of work, and it is
false whenever the two sides are at incompatible revisions.

An ecosystem's real connectivity is the fraction of (host, server) pairs that can
actually interoperate (eq:connectivity-is-the-real-quantity). Three policy
choices decide it:

  revision rate   how often the spec makes a backwards-incompatible change
  support window  how many revisions each implementation accepts at once
  upgrade lag     how far behind the current revision implementations sit

MCP's own choices are informative: version strings increment ONLY on breaking
change, implementations MAY support several versions simultaneously, and an
unsupported version returns an error listing what IS supported so the caller can
retry (cite:mcp2026spec). This listing measures what each of those is worth.
"""
import numpy as np

rng = np.random.default_rng(4001)

M = 40000
N_HOSTS = 40
N_SERVERS = 400


def connectivity(rev_rate, window, lag_mean, years=3.0, m=M,
                 negotiate=True, hosts=N_HOSTS, servers=N_SERVERS):
    """Revisions arrive at rev_rate per year. Each implementation targets a
    revision it adopted `lag` years ago and accepts `window` consecutive
    revisions. A pair interoperates if their accepted sets intersect."""
    n_rev = max(1, int(round(rev_rate * years)))
    # Each side's newest supported revision, as an index into the revision list.
    h_lag = rng.exponential(lag_mean, hosts)
    s_lag = rng.exponential(lag_mean, servers)
    h_top = np.clip(n_rev - np.round(h_lag * rev_rate), 0, n_rev).astype(int)
    s_top = np.clip(n_rev - np.round(s_lag * rev_rate), 0, n_rev).astype(int)
    h_bot = np.maximum(h_top - (window - 1), 0)
    s_bot = np.maximum(s_top - (window - 1), 0)
    # Broadcast every host against every server.
    lo = np.maximum(h_bot[:, None], s_bot[None, :])
    hi = np.minimum(h_top[:, None], s_top[None, :])
    overlap = hi >= lo
    if not negotiate:
        # Without negotiation a pair must agree on one specific revision: the
        # client offers its newest and the server takes it or fails.
        overlap = (h_top[:, None] >= s_bot[None, :]) & \
                  (h_top[:, None] <= s_top[None, :])
    return float(overlap.mean())


print(f"{N_HOSTS} hosts and {N_SERVERS} servers over 3 years. A pair connects")
print("when their supported revision ranges intersect.")
print()
print(f"{'support window':>16}{'rev 0.5/yr':>13}{'rev 1/yr':>11}"
      f"{'rev 2/yr':>11}{'rev 4/yr':>11}")
print("-" * 62)
tab = {}
for w in (1, 2, 3, 5, 8):
    row = tuple(connectivity(r, w, 0.6) for r in (0.5, 1.0, 2.0, 4.0))
    tab[w] = row
    print(f"{w:>16}" + "".join(f"{v:>{c}.1%}" for v, c in
                               zip(row, (13, 11, 11, 11))))

print()
print()
print("Upgrade lag is the variable ecosystems try to fix, by pressuring")
print("implementers to keep current. Revision rate 2/yr, window 2:")
print()
print(f"{'mean upgrade lag':>18}{'connectivity':>14}")
print("-" * 32)
lg = {}
for L in (0.1, 0.3, 0.6, 1.2, 2.4):
    v = connectivity(2.0, 2, L)
    lg[L] = v
    print(f"{L:>18.1f}{v:>14.1%}")

print()
print()
print("Widening the window is the other lever, and costs the implementer rather")
print("than the ecosystem. Same revision rate, lag held at 1.2 years:")
print()
print(f"{'support window':>16}{'connectivity':>14}{'gain':>9}")
print("-" * 39)
wd = {}
prev = None
for w in (1, 2, 3, 4, 6, 8):
    v = connectivity(2.0, w, 1.2)
    wd[w] = v
    g = "--" if prev is None else f"{v - prev:+.1%}"
    print(f"{w:>16}{v:>14.1%}{g:>9}")
    prev = v

print()
print()
print("The two levers against each other, at a fixed connectivity target.")
print("Each cell is connectivity; the ecosystem chooses a row, the implementer")
print("chooses a column.")
print()
print(f"{'mean lag':>10}" + "".join(f"{'window ' + str(w):>13}"
                                    for w in (1, 2, 4, 8)))
print("-" * 62)
mx = {}
for L in (0.2, 0.6, 1.2, 2.4):
    row = tuple(connectivity(2.0, w, L) for w in (1, 2, 4, 8))
    mx[L] = row
    print(f"{L:>10.1f}" + "".join(f"{v:>13.1%}" for v in row))

print()
print()
print("And what negotiation is worth. Without it, a client offers one revision")
print("and the server accepts or fails; with it, the pair finds any revision")
print("they share.")
print()
print(f"{'support window':>16}{'no negotiation':>16}{'negotiation':>13}"
      f"{'gain':>9}")
print("-" * 54)
ng = {}
for w in (1, 2, 3, 5, 8):
    a = connectivity(2.0, w, 1.2, negotiate=False)
    b = connectivity(2.0, w, 1.2, negotiate=True)
    ng[w] = (a, b)
    print(f"{w:>16}{a:>16.1%}{b:>13.1%}{b - a:>+9.1%}")

print(f"""
The first table is the ecosystem's health as a function of two things it can
choose, and the choice that matters is the one implementers make rather than the
one the spec authors make.

At {2} revisions a year, a support window of {1} connects {tab[1][2]:.1%} of pairs
and a window of {8} connects {tab[8][2]:.1%}. Across the whole table, moving down
a column is worth far more than moving left along a row.

The second and third tables put the two levers side by side. Upgrade lag -- the
variable ecosystems actually try to manage, through deprecation notices and
pressure to stay current -- moves connectivity from {lg[0.1]:.1%} at a lag of
{0.1} years to {lg[2.4]:.1%} at {2.4}. Widening the window from {1} to {8} moves
it from {wd[1]:.1%} to {wd[8]:.1%} at a FIXED lag of {1.2} years.

The cross-table settles it. **A slow ecosystem with wide windows beats a diligent
one with narrow windows**: mean lag {2.4} with window {4} reaches {mx[2.4][2]:.1%},
against {mx[0.2][0]:.1%} for mean lag {0.2} with window {1}. And window {8}
reaches {mx[2.4][3]:.1%} at every lag in the table
(eq:support-window-beats-upgrade-pressure).

That is worth dwelling on because the effort usually goes the other way. Chasing
upgrade lag means persuading hundreds of independent implementers to do work on
your schedule, which is slow, adversarial, and never finishes. Widening a window
means one implementer accepting a few extra revisions, which is a local decision
with a local cost.

Note also that lag has a floor -- {lg[1.2]:.1%} at {1.2} years and {lg[2.4]:.1%} at
{2.4} -- because once everyone is far behind, they are far behind TOGETHER. A
uniformly stale ecosystem is more connected than a half-upgraded one, which is an
uncomfortable thing to know about deprecation campaigns.

The last table is the one that explains a design choice rather than just scoring
it. Negotiation -- the pair searching for any revision they share, rather than the
client offering one and the server accepting or failing -- is worth
{ng[1][1] - ng[1][0]:+.1%} at a window of {1} and {ng[8][1] - ng[8][0]:+.1%} at a
window of {8}.

**The value of negotiation grows with the window, because negotiation is what
makes a window reachable** (eq:negotiation-unlocks-the-window). A server that
supports eight revisions and cannot say so is a server that supports one:
{ng[8][0]:.1%} without negotiation, against {ng[3][1]:.1%} for a THREE-revision
window that can negotiate.

So multi-version support and version negotiation are not two independent good
ideas. They are one mechanism, and implementing either alone wastes most of it.
That is why cite:mcp2026spec pairs them: implementations MAY support several
revisions at once, and an unsupported version returns an error LISTING what is
supported, so the caller can retry into the overlap.

The third choice in that specification is the first table's columns -- version
strings increment only on backwards-incompatible change, so the revision rate in
this model counts breaking changes rather than releases. **Making the version
number mean "breaking" rather than "new" moves an ecosystem several columns to the
left**, and it costs nothing but discipline.""")
```

## 9. Practical Example

The first listing prices both regimes in engineer-days over three years:

```
  hosts x providers    bespoke   protocol    ratio    winner
------------------------------------------------------------
              2 x 3        261        168      1.6  protocol
             5 x 20      4,350        805      5.4  protocol
           20 x 200    174,000      6,916     25.2  protocol
```

The same ecosystems with the maintenance term removed disagree at the small end:

```
              2 x 3         72        144      0.5   bespoke
              3 x 8        288        307      0.9   bespoke
             5 x 20      1,200        685      1.8  protocol
```

**For a small ecosystem the protocol loses on build cost** — a compliant server is
more work than a purpose-built adapter. Stating the $N \times M$ case as a build
argument invites a correct objection from anyone with three integrations to write.

Over time, for a three-by-six ecosystem:

```
  years    bespoke   protocol   advantage
-----------------------------------------
    0.0        216        255         -39
    1.0        405        269        +136
    4.0        972        313        +659
```

And the decomposition at ten by sixty over three years:

```
                   build   maintenance   maint share
----------------------------------------------------
     bespoke       7,200        18,900           72%
    protocol       1,890           336           15%
```

**The $N \times M$ problem is a maintenance problem wearing a build problem's
clothes** ({{eq:maintenance-dominates-integration}}) — the quadratic term is in
the repairs, not the construction.

Break-even ecosystem size:

```
  hosts  break-even m, year 0    year 1    year 3
-------------------------------------------------
      1                 never     never         3
      3                    10         3         2
     10                     4         2         1
```

Almost the entire case is made by the second year, which is why it is hard to make
in advance.

And the mechanism behind adoption:

```
  hosts already present    bespoke   protocol
---------------------------------------------
                      1         44         31
                     10        435         31
                    100      4,350         31
```

**A protocol turns the marginal joiner's cost from $O(N)$ into $O(1)$**
({{eq:protocol-makes-entry-constant}}). Under the bespoke regime, the better the
ecosystem, the more it costs to enter.

The second listing measures whether the protocol delivers. Connectivity — the
fraction of host-server pairs that can actually interoperate:

```
  support window   rev 0.5/yr   rev 1/yr   rev 2/yr   rev 4/yr
--------------------------------------------------------------
               1        65.9%      41.7%      28.2%      16.0%
               3       100.0%      98.7%      88.8%      63.8%
               8       100.0%     100.0%     100.0%      95.3%
```

Against the two levers directly:

```
  mean lag     window 1     window 2     window 4     window 8
--------------------------------------------------------------
       0.2        52.8%        93.9%       100.0%       100.0%
       1.2        16.7%        45.0%        78.5%       100.0%
       2.4        21.8%        37.4%        73.2%       100.0%
```

**A lagging ecosystem with wide windows beats a current one with narrow windows**
— lag $2.4$ with window $4$ reaches $73.2\%$ against lag $0.2$ with window $1$ at
$52.8\%$ ({{eq:support-window-beats-upgrade-pressure}}). Window $8$ reaches
$100\%$ at every lag, because a window is a threshold and lag is only a scale.

Note also the floor: $43.1\%$ at lag $1.2$ and $37.7\%$ at $2.4$. Once everyone is
far behind, they are far behind *together* — a uniformly stale ecosystem is more
connected than a half-upgraded one, which is uncomfortable news for deprecation
campaigns.

And what negotiation is worth:

```
  support window  no negotiation  negotiation     gain
------------------------------------------------------
               1           18.5%        21.8%    +3.2%
               3           40.1%        68.6%   +28.5%
               8           57.8%       100.0%   +42.2%
```

**Negotiation's value grows with the window, because negotiation is what makes the
window reachable** ({{eq:negotiation-unlocks-the-window}}). An eight-revision
window that cannot be advertised loses to a three-revision window that can. The
two are one mechanism, and {{cite:mcp2026spec}} pairs them for this reason.

## 10. Production Considerations

Argue for a protocol on maintenance, not on build cost. The build-cost version of
the argument is refutable and the maintenance version is not.

Estimate $T$ before estimating $n \times m$. The break-even is a function of how
long the integrations live, and teams routinely underestimate that by years.

If you are a provider, compute your entry cost both ways. The $O(N)$ versus
$O(1)$ comparison is usually decisive on its own.

Increment your version only on breaking change. It is free and it moves you
several columns in the first connectivity table.

Support a window of revisions rather than one, and make sure you can *advertise*
the window — an unadvertised window is nearly wasted.

Prefer widening your own window to campaigning for others to upgrade. One is a
local decision; the other is a permanent negotiation with strangers.

And do not confuse adopting a protocol with solving trust.
{{cite:hou2025mcp}}'s threats mostly land before the first message.

## 11. Common Mistakes

**Arguing $N \times M$ as a build-cost saving.** It is not one at small scale, and
the objection is correct.

**Asking whether the ecosystem is big enough.** The break-even depends on $T$ far
more than on size.

**Versioning on release rather than on breaking change.** It inflates $\rho$ for
no benefit.

**Supporting one revision.** The threshold lever left unpulled.

**Supporting many revisions silently.** Without negotiation, most of that work is
unreachable.

**Running deprecation campaigns to fix connectivity.** The weaker lever, and
adversarial besides.

**Assuming compliance implies trust.** The protocol says how to talk, not to whom.

## 12. Failure Modes

*Silent adapter rot.* A bespoke integration broken by an upstream change nobody
told its maintainer about — the failure the protocol exists to prevent.

*Version islands.* Subsets of the ecosystem that interoperate internally and not
across, which is what the connectivity table's low cells look like in practice.

*Unadvertised capability.* A server supporting five revisions that behaves like a
server supporting one.

*Protocol as false assurance.* Compliance mistaken for safety;
{{ch:mcp-security}}'s subject.

*Semantic mismatch under a compliant wire format.* Both sides speak the protocol
correctly and disagree about what a tool means — the layer no protocol covers.

## 13. Alternatives

**Bespoke adapters.** Correct for short-lived integrations and for the case where
you control both sides.

**A shared client library rather than a protocol.** Removes duplicated work
without requiring providers to change, at the cost of only helping hosts that
adopt your library — an $N$-side-only solution.

**Code generation from API specifications.** OpenAPI-to-tool generators address
the same quadratic and produce tools that fail
{{ch:ag-tool-calling}}'s design criteria, since an API surface is not a tool
surface.

**A hosted integration platform.** Someone else operates the $N \times M$ grid.
This is a real answer that trades the maintenance cost for a dependency and a
per-call fee.

**Fine-tuning on API documentation.** {{cite:patil2023gorilla}}'s finding is that
this goes stale and retrieval does not, so it is the weakest option here.

## 14. Evaluation

Measure your own $b$, $f$, $\lambda_h$ and $\lambda_p$ from your issue tracker
before trusting this chapter's assumed values. The conclusions are robust to the
ratios, not to the specific numbers.

Measure connectivity directly if you operate an ecosystem: what fraction of
(client, server) pairs can actually complete a call today. Almost nobody publishes
this and it is the quantity that matters.

Track the distribution of adopted revisions, not the mean. The floor effect in
{{sec:9-practical-example}} means a bimodal ecosystem behaves very differently
from a uniformly lagging one at the same average.

Count integration repairs per quarter. It is the direct measurement of the term
that decides the whole argument.

## 15. Advanced Concepts

**Connectivity as a published ecosystem metric.** Registries could compute and
publish the fraction of interoperable pairs, which would make version policy
debates empirical. Nothing does this. {{maturity:EMERGING}}.

**Automatic window widening.** A server that retains handlers for older revisions
mechanically, rather than by maintained code paths, would make $w$ cheap enough
that {{eq:support-window-beats-upgrade-pressure}} could be pushed to its limit.

**Semantic compatibility checking.** The layer no protocol standardises: whether
two implementations mean the same thing by a tool. This is a specification
problem, not a wire-format problem. {{maturity:RESEARCH FRONTIER}}.

**Measuring the tail effect.** {{sec:7-internal-mechanics}}'s LSP prediction —
that the visible consequence is integrations nobody would have built — is testable
against registry growth data and has not been tested.

## 16. Connection to Previous Chapters

{{ch:ag-tool-calling}}'s design criteria survive the protocol entirely: a
protocol standardises the wire and leaves the naming, the schemas and the error
messages exactly as consequential as that chapter found them.

{{ch:as-specialized}}'s affordance argument acquires a distribution mechanism —
a verifier or a structured observation built once can now reach every host, which
is the strongest practical case for the protocol that this chapter does not price.

{{ch:ag-what-is-an-agent}}'s tail-mass framing returns in
{{sec:7-internal-mechanics}}: the protocol's visible effect is the long tail of
integrations that become economic.

Ahead: {{ch:mcp-architecture}} takes up the per-request, stateless design this
chapter's negotiation result motivates; {{ch:mcp-security}} takes up the trust the
$N + M$ arithmetic silently assumes.

## 17. Exercises

1. Substitute your own $b$, $f$, $\lambda_h$, $\lambda_p$ into the first listing
   and recompute your break-even $T$.

2. Add a term for the cost of a *broken* integration — downtime, not just repair —
   and see how much it moves the break-even.

3. Derive the connectivity expression for a non-contiguous support window (an
   implementation supporting revisions 3 and 7 but not 5). Does the threshold
   argument survive?

4. Model a bimodal adoption distribution and compare with the exponential one at
   the same mean.

5. Implement asymmetric negotiation — only the server advertises — and find how
   much of the two-sided benefit it recovers.

6. Extend the first listing so a fraction of providers refuse to adopt the
   protocol. At what refusal rate does the hybrid ecosystem lose to pure bespoke?

## 18. Interview Questions

1. State the $N \times M$ argument and then state what is wrong with the usual
   version of it.

2. Your ecosystem has three hosts and six providers. Should you adopt a protocol?

3. Why do protocols get adopted before they are break-even in total cost?

4. Your ecosystem's connectivity is $40\%$. What are your two levers, and which is
   stronger?

5. When should a protocol's version number change?

6. A server supports eight revisions and connectivity did not improve. What is
   missing?

## 19. Research Questions

1. Can ecosystem connectivity be measured and published in a way that changes
   version-policy behaviour?

2. Can support windows be widened mechanically rather than by maintained code?

3. Is there a tractable notion of semantic compatibility above the wire format?

4. Does the LSP tail prediction hold quantitatively for tool ecosystems?

5. How do the economics change when one party operates both a host and a
   registry?

## 20. Chapter Summary

The $N \times M$ argument for a tool protocol is right about the shape and, as
usually stated, wrong about the economics. On build cost alone a small ecosystem
favours bespoke adapters — $72$ engineer-days against $144$ at two hosts and three
providers.

What decides it is maintenance. Over three years at ten by sixty, repairs are
$72\%$ of the bespoke regime's cost and $15\%$ of the protocol's
({{eq:maintenance-dominates-integration}}): **the quadratic term is in the
repairs, not the construction.** A three-by-six ecosystem behind by $39$ days at
year zero is ahead by $659$ at year four, and the break-even collapses from
*never* to two or three providers between year zero and year three. **The right
question is how long the integrations will live, not how many there are.**

But the claim that explains adoption is about the marginal joiner. A provider
entering an ecosystem of a hundred hosts pays $4{,}350$ days bespoke and $31$ with
a protocol — the same $31$ as joining an ecosystem of one. **A protocol turns
entry cost from $O(N)$ into $O(1)$** ({{eq:protocol-makes-entry-constant}}), and
that is the calculation each participant actually performs.

Delivering $N + M$ then depends on versioning policy. Connectivity is governed by
a *threshold* — the support window — against a *scale* — revision rate times
adoption lag — and thresholds saturate where scales only shrink toward a floor
({{eq:support-window-beats-upgrade-pressure}}). A lagging ecosystem with
four-revision windows reached $73.2\%$ against a current one with single-revision
windows at $52.8\%$, and an eight-revision window reached $100\%$ at every lag.

Negotiation is what makes a window reachable, and its value grows with the window:
$+3.2$ points at window one, $+42.2$ at window eight
({{eq:negotiation-unlocks-the-window}}). An eight-revision window that cannot be
advertised loses to a three-revision window that can. **Multi-version support and
negotiation are one mechanism**, which is why {{cite:mcp2026spec}} adopts both —
along with the third free choice, incrementing the version only on breaking change
({{eq:breaking-means-breaking}}).

## 21. Further Reading

{{cite:mcp2026spec}} is the primary source for the rest of this part, and its
versioning page is worth reading directly against
{{sec:9-practical-example}}'s second listing — the three policies it adopts are
the three levers measured there.

{{cite:qin2023toolllm}} and {{cite:li2023apibank}} for the inventory scale that
made a protocol necessary rather than tidy, and
{{cite:patil2023gorilla}} for why discovery must happen at runtime rather than at
training time.

{{cite:hou2025mcp}} for the lifecycle and threat surface the $N + M$ arithmetic
assumes away, which {{ch:mcp-security}} takes up.

{{ch:ag-tool-calling}} for everything a protocol does not standardise, which is
most of what makes a tool good.
