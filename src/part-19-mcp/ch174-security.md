---
id: mcp-security
number: 174
part: XIX
tier: full
status: draft
requires: [blast-radius-is-a-union, contain-do-not-detect,
           gate-on-consequence, schemas-are-rent,
           revalidation-is-cheapest]
provides: [audience-binding-bounds-the-union, passthrough-is-quadratic,
           scope-minimisation-lowers-the-ceiling, structure-beats-detection-again,
           approval-is-a-snapshot, lifecycle-decides-the-rate]
citations: [mcp2026spec, hou2025mcp, huang2026mcpthreat, gaire2025mcpsok,
            greshake2023indirect, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what the specification's
authorization requirements are actually for, and why token passthrough is
forbidden rather than discouraged; show that passthrough makes blast radius
quadratic in connected servers while audience binding makes it linear; distinguish
what audience binding bounds from what scope minimisation bounds; rank the
defences against tool poisoning by measured effect; and explain why approving a
tool is approving a snapshot of something mutable.

## 2. Why This Matters

MCP's authorization section reads like OAuth bookkeeping — nine RFCs, a discovery
dance, a table of `iss` validation cases. Underneath it are three requirements
that are one structural claim ({{cite:mcp2026spec}}):

- servers **MUST** validate that access tokens were issued for them as audience;
- clients **MUST** send the RFC 8707 `resource` parameter naming the server,
  *regardless of whether the authorization server supports it*;
- servers **MUST NOT accept or transit any other tokens.**

That last clause forbids token passthrough, and {{sec:9-practical-example}} prices
it. Under passthrough, one compromised server reaches $9.08$ of $54$ scopes on
average and *all $54$* at the 99th percentile. Audience-bound tokens bring that to
$1.08$; adding least privilege to $0.36$
({{eq:audience-binding-bounds-the-union}}).

The structural reason for MUST rather than SHOULD is in the scaling. From two
servers to eighty, passthrough exposure grows $0.5 \to 384.9$ — roughly
quadratically, because $n$ compromisable servers each hold a token good at $n$
servers. Audience-bound exposure grows $0.2 \to 9.6$, linearly
({{eq:passthrough-is-quadratic}}). **A rule that looks like pedantry at three
servers is load-bearing at thirty**, and every host is adding servers.

The second half is the attack that actually shows up.
{{cite:huang2026mcpthreat}} modelled MCP across five components and evaluated
seven major clients, finding **tool poisoning — hostile instructions embedded in
tool metadata — the most prevalent and impactful client-side vulnerability.** That
is structural: a description is text that reaches the model, so it is an
instruction whether written as one or not.

{{sec:9-practical-example}} compares the defences and reproduces
{{ch:ag-security}}'s ordering exactly. Static metadata scanning removes $56\%$ of
harmful actions, parameter visibility $31\%$, both together $69\%$ — and a
capability partition alone removes $80\%$
({{eq:structure-beats-detection-again}}).

## 3. Prerequisites

{{ch:ag-security}}'s {{eq:blast-radius-is-a-union}} — the damage from a compromise
is the union of what is reachable from it, which is exactly what audience binding
bounds — together with its {{eq:contain-do-not-detect}}, whose ordering this
chapter reproduces in a new setting.

{{ch:ag-termination}}'s {{eq:gate-on-consequence}} and its habituation result,
which is why the human-attention defence is the weakest one measured here.

{{ch:mcp-schemas}}'s {{eq:schemas-are-rent}}: tool descriptions are text in the
context, and this chapter is about the other thing that makes them dangerous.

{{ch:as-long-running}}'s {{eq:revalidation-is-cheapest}}, which appears here for
the third time in this book.

## 4. Intuitive Explanation

Start with what the specification says about who holds what.

An MCP server is an OAuth 2.1 **resource server**; the client is an OAuth 2.1
**client**; the authorization server issues tokens and may or may not be
co-hosted. Servers must publish RFC 9728 Protected Resource Metadata, so a client
that gets a `401` can discover where to authenticate from the
`WWW-Authenticate` header. That is all conventional.

The interesting requirement is the `resource` parameter. When the client asks for
a token, it must say *which server the token is for*, using that server's
canonical URI — and it must do so whether or not the authorization server does
anything with the parameter. The server, in turn, must check that tokens presented
to it were issued for it.

The effect is that a token becomes useless anywhere except where it was meant to
be used.

Why does that matter enough to be a MUST? Consider the alternative, which is what
people build when nobody stops them. The user authenticates once; the host gets a
token with a generous scope; every server accepts it. It is simpler, the login
flow happens once, and everything works.

Then one of those servers is compromised — or was hostile from the start, having
been installed from a registry. It now holds a token that every other server will
accept. {{ch:ag-security}} called this the capability union, and here the union is
*everything the user ever authorised*, not just what that server needed.

{{sec:9-practical-example}} measures the difference and finds the scaling worse
than the mean suggests. With audience binding, a compromised server exposes its own
scopes: exposure is linear in the number of servers. With passthrough, each of $n$
compromisable servers holds a key to all $n$, so exposure is quadratic. At three
servers the two designs look similar. At thirty they do not.

Scope minimisation is the second half, and it bounds something different. Audience
binding lowers the *typical* damage and leaves the ceiling untouched — if every
server is compromised, everything is reached either way. Requesting only the scopes
actually needed lowers the ceiling itself. The specification supports this with the
scope challenge: a server responds `403` with `error="insufficient_scope"` and the
scopes it needs, and the client re-authorises for the *union* of what it had and
what was asked. Request little; escalate when told to.

Now the other attack, which no amount of OAuth prevents.

A tool's description is text. It goes into the model's context. The model reads it
as guidance about what the tool does — which is what it is for — and there is no
mechanism separating "description of a tool" from "instruction to the model,"
because both are just text in the same window.

So a hostile server writes a description that says something like *ignore prior
instructions about confirming destructive operations*. {{cite:huang2026mcpthreat}}
found this to be the dominant client-side vulnerability across seven clients, and
attributed it to insufficient static validation and parameter visibility.
{{cite:mcp2026spec}} concedes the problem in its own security principles: tool
annotations "should be considered untrusted, unless obtained from a trusted
server."

Three defences are available and {{sec:9-practical-example}} ranks them. Scanning
the metadata for hostile content is a classifier, with a classifier's error rate.
Showing the user the actual arguments before execution works, and depends entirely
on the user reading them — which {{ch:ag-termination}} showed collapses under
volume. Partitioning capabilities so that a tool exposed to untrusted content
cannot reach anything worth reaching asks no question at all, and beats both
detection defences combined.

And one more thing, which the discussion usually omits entirely. The description
is read when the tool is *approved*. The tool is invoked for months afterwards.
Nothing requires the server to keep serving the description it was approved on.

## 5. Formal Explanation

Let a host connect $n$ servers, each offering $\sigma$ scopes, each independently
compromised with probability $p$. Write $R$ for the scopes an attacker reaches.

**Passthrough.** A token accepted anywhere means one compromise reaches every
scope:

$$\mathbb{E}[R_{\text{pass}}] = n\sigma\big(1 - (1-p)^n\big) \approx n^2\sigma p \quad\text{for small } p$$ (eq:passthrough-is-quadratic)

**Audience-bound.** Each compromised server exposes only its own:

$$\mathbb{E}[R_{\text{aud}}] = n\sigma p$$

The ratio is $\big(1-(1-p)^n\big)/p \approx n$. **Passthrough is quadratic in
connected servers and audience binding is linear**, so the penalty for the
convenient design grows exactly as hosts connect more servers.

**Least privilege.** With a token carrying only the fraction $\mu$ of scopes
actually used:

$$\mathbb{E}[R_{\text{least}}] = n\sigma\mu p$$ (eq:audience-binding-bounds-the-union)

The two mechanisms act on different statistics. Conditioning on the worst case —
every server compromised — gives:

$$\max R_{\text{pass}} = \max R_{\text{aud}} = n\sigma, \qquad \max R_{\text{least}} = n\sigma\mu$$ (eq:scope-minimisation-lowers-the-ceiling)

**Audience binding bounds the mean; only scope minimisation bounds the maximum.**
This is why the specification requires both, and why implementing only the first
leaves the tail exactly where it was.

Now poisoning. Let a fraction $\pi$ of tools carry a hostile instruction, the model
act on one with probability $\omega$, and $c$ calls occur. Undefended:

$$\mathbb{E}[\text{harm}] = c\,\pi\,\omega$$

A **detector** with detection rate $d$ multiplies by $(1-d)$. **Parameter
visibility** with display fidelity $\phi$ and user attention $\alpha$ multiplies by
$(1 - \phi\alpha)$ — a product, so it is bounded above by the weaker factor, and
{{ch:ag-termination}} says $\alpha$ falls with volume. A **partition** with
coverage $\gamma$ multiplies by $(1-\gamma)$ and asks no classification question at
all:

$$\mathbb{E}[\text{harm}]_{\text{part}} = c\pi\omega(1-\gamma) \quad\text{with } \gamma \text{ a design property, not an accuracy}$$ (eq:structure-beats-detection-again)

All three are multiplicative, so they compose — but $\gamma$ is chosen and $d$,
$\alpha$ are estimated, which is the difference that matters when the attacker is
adaptive.

Finally, mutability. Approval happens once at time $0$; the tool is called $c$
times. If a server turns hostile with hazard $\lambda$ per call and re-verification
occurs every $\kappa$ calls with detection $\rho$:

$$\mathbb{E}[\text{exposed calls}] \approx \lambda c \cdot \frac{\kappa}{2\rho}$$ (eq:approval-is-a-snapshot)

**Linear in the re-verification interval**, so exposure is a scheduling choice.
Note $\lambda$ itself is not: it is set by who may publish and what is verified at
install:

$$\lambda = \lambda(\text{registry policy}, \text{provenance}, \text{install-time verification})$$ (eq:lifecycle-decides-the-rate)

which is {{cite:hou2025mcp}}'s point — most of the threat surface is in the server
lifecycle, before any protocol message exists.

## 6. Mathematical Foundation

Three extractions.

**The quadratic is the whole argument.** From
{{eq:passthrough-is-quadratic}}, the passthrough penalty is a factor of $n$. Any
argument of the form "we only have a few servers, passthrough is fine" is an
argument about today's $n$, and $n$ only goes up. This is why the specification
uses MUST: the design is not *slightly* worse, it is worse by a growing factor.

**Mean and maximum need different controls.**
{{eq:scope-minimisation-lowers-the-ceiling}} says audience binding and scope
minimisation are not substitutes and not redundant — one shapes the distribution,
the other truncates it. A deployment with perfect audience binding and generous
scopes has a fine average and an unchanged catastrophe.

**Chosen coverage beats estimated accuracy.**
{{eq:structure-beats-detection-again}}'s $\gamma$ is a property of your
architecture that you can verify by inspection; $d$ and $\alpha$ are estimates
against a distribution an adversary controls. Both appear the same way in the
arithmetic, which is exactly why the arithmetic understates the difference: an
attacker can move $d$ and cannot move $\gamma$.

## 7. Internal Mechanics

### 7.1 The authorization flow, and the two checks that matter

```mermaid {#fig:mcp-authz caption="The MCP authorization flow. The resource parameter and the audience check are the two steps that bound blast radius; everything else is discovery."}
flowchart TD
    C[client] -->|request, no token| S[MCP server]
    S -->|401 + WWW-Authenticate<br/>resource_metadata, scope| C
    C -->|fetch RFC 9728 metadata| S
    C -->|discover AS metadata| A[authorization server]
    C -->|authorize + PKCE + resource=URI| A
    A -->|code + iss| C
    C -->|validate iss RFC 9207| C
    C -->|token request + resource=URI| A
    A -->|access token, audience = server| C
    C -->|Bearer token| S
    S -->|validate audience| S
```

Most of that diagram is discovery and can be got right by using a library. Two
steps cannot be delegated:

**The `resource` parameter**, sent in both the authorization and token requests,
naming the server's canonical URI — with no fragment, and most-specific available.
Clients must send it whether or not the authorization server supports it.

**The audience check** at the server, which is the enforcement half. A client that
sends `resource` diligently and a server that does not check audience buys
nothing.

And the `iss` validation from RFC 9207, which prevents the mix-up attack: the
client records the expected issuer alongside its PKCE verifier and compares on the
callback, with **no URI normalisation** before comparison — no case folding, no
default-port elision, no trailing-slash adjustment. That prohibition exists because
normalisation is where comparison bugs live.

### 7.2 Step-up, and why it makes least privilege practical

Least privilege fails in practice when it means predicting every scope a session
might need. The specification's scope challenge removes the prediction.

A client starts with the scopes in the `WWW-Authenticate` challenge, or with
`scopes_supported` from the resource metadata — which the spec says should be *the
minimal set for basic functionality*. When an operation needs more, the server
returns `403` with `error="insufficient_scope"` and the scopes required. The client
re-authorises for the **union** of what it had and what was challenged, so earlier
permissions are not lost.

Two details worth carrying. Servers should emit *all* scopes needed for an
operation in one challenge — incremental challenges force multiple round trips
through a browser for one action, which is how least privilege acquires its
reputation for being unusable. And clients must treat the challenged scopes as
authoritative and must not assume any set relationship with `scopes_supported`.

### 7.3 stdio has no authorization, deliberately

{{cite:mcp2026spec}} says implementations using stdio **SHOULD NOT** follow the
authorization specification, and should retrieve credentials from the environment
instead.

That is correct rather than a gap. A stdio server is a subprocess the client
launched on the user's own machine; it already runs with the user's privileges, and
an OAuth flow would add ceremony without adding a boundary. The security properties
of stdio come from process isolation and from {{ch:mcp-architecture}}'s
one-process-per-client shape, not from tokens.

The consequence for design: **moving a server from stdio to HTTP creates an
authorization surface that did not previously exist**, and it is a common migration.
A team that developed locally against stdio has not yet made any of the decisions
in this chapter.

### 7.4 Why detection loses to structure, again

{{sec:9-practical-example}} reproduces {{ch:ag-security}}'s ordering, and it is
worth being precise about why rather than treating it as a slogan.

A detector answers "is this hostile?" That is a classification over text an
adversary writes, so its error rate is a function of adversarial effort. Every
improvement to the detector is met by a change in the input.

A partition answers nothing. It arranges that the tool which reads untrusted
content holds no capability worth abusing — {{ch:ag-security}}'s reader/actor split.
An injected instruction is obeyed and accomplishes nothing, because obedience does
not confer capability.

The measurement is $80\%$ for the partition alone against $69\%$ for both detectors
together, and the arithmetic *understates* it, because $\gamma$ is fixed by your
architecture while $d$ moves when someone tries.

None of which argues for skipping detectors: all three together reached $94\%$. It
argues about **order**. Build the partition first, because it is the only layer
whose effectiveness does not depend on predicting what an attacker will write.

### 7.5 Parameter visibility and the attention it spends

{{cite:huang2026mcpthreat}} names parameter visibility as a mitigation layer, and
it is right — showing the user the actual arguments before execution catches things
no scanner will.

It also spends a resource {{ch:ag-termination}} measured. At $95\%$ user attention
it removes $66\%$ of harm; at $10\%$, $7\%$. An agent making forty tool calls a
session is not being watched at the fortieth with the care of the first.

The reconciliation is {{eq:gate-on-consequence}} and
{{ch:as-long-running}}'s placement result: show arguments **before consequential
operations** rather than before every operation. Spending scarce attention where it
changes an outcome was worth an eightfold review budget there, and the same logic
applies here.

### 7.6 The lifecycle is where the rate is set

{{cite:hou2025mcp}} decomposes the server lifecycle into four phases — creation,
deployment, operation, maintenance — across sixteen activities, and finds threats
distributed across all of them, including installer spoofing and tool poisoning
introduced at creation.

The reason that framing matters is {{eq:lifecycle-decides-the-rate}}. Every defence
in {{sec:9-practical-example}} operates on $\pi$, the poisoned fraction, as a
given. But $\pi$ is a policy outcome: who may publish to the registry, what
provenance is attached, what is verified at install time.

**A host cannot see $\pi$ from inside a session**, and a registry that loosens its
admission policy moves every connected host along the table at once, silently. That
is an ecosystem-design problem, and {{ch:mcp-production}} takes it up.

### 7.7 Approval is a snapshot

The problem the discussion usually omits: a tool description is read at approval
and acted on at every call.

Nothing requires a server to keep serving the description it was approved on. A
server can be benign through review, adoption and a hundred uses, then change its
description — the "rug pull". Approval was a snapshot of a mutable thing
({{eq:approval-is-a-snapshot}}).

{{sec:9-practical-example}} finds never re-verifying giving $1.913$ harmful actions
against $0.074$ when metadata is re-verified every call. **Re-validation is the
cheapest intervention here as it was in {{ch:as-long-running}} and
{{ch:mcp-primitives}}** — the third independent appearance in this book, at scales
from a week-long workflow to a single turn to a metadata check.

The rule generalises past MCP: **anything approved once and used many times needs
re-approval on a schedule**, and the schedule should be tighter than feels
necessary. Practically, hash the tool definitions at approval, compare on every
listing, and treat a change as requiring re-approval rather than as a version bump.

## 8. Implementation

Two listings. The first prices the specification's token requirements as blast
radius. The second ranks the defences against tool poisoning.

```python {tier=A name=audience-binding-bounds-the-union}
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
```

The second listing takes up the attack that authorization does not prevent.

```python {tier=A name=structure-beats-detection-again}
"""Tool poisoning, which is the vulnerability the measurements actually find.

cite:huang2026mcpthreat modelled MCP with STRIDE and DREAD across five components
and evaluated seven major clients. Its finding: tool poisoning -- malicious
instructions embedded in tool METADATA -- is the most prevalent and impactful
client-side vulnerability, attributed to insufficient static validation and
parameter visibility.

The mechanism is structural rather than a bug. A tool's description is text that
reaches the model, so it is an instruction whether or not it was written as one.
cite:mcp2026spec says as much: tool annotations "should be considered untrusted,
unless obtained from a trusted server."

There is a second problem the first hides. Descriptions are read at DISCOVERY and
tools are invoked later, so a server can be benign when approved and malicious
afterwards (eq:approval-is-a-snapshot). This listing measures both, and compares
the defences cite:huang2026mcpthreat proposes against ch:ag-security's structural
one.
"""
import numpy as np

rng = np.random.default_rng(4327)

M = 60000
CALLS = 40              # tool calls in the period being modelled
P_POISON = 0.03         # share of tools carrying a hostile instruction
P_OBEY = 0.62           # chance the model acts on an injected instruction


def run(defence, m=M, calls=CALLS, p_poison=P_POISON, p_obey=P_OBEY,
        scan_detect=0.55, vis_detect=0.70, vis_attention=0.45,
        partition_cover=0.80):
    """Harmful actions per deployment under one defence.

    none        the description reaches the model unmodified
    scan        static analysis of tool metadata before presentation
    visibility  the user is shown the actual arguments before execution
    partition   ch:ag-security's split: the poisoned tool cannot reach anything
                worth reaching, so obeying it accomplishes nothing
    both        scan + visibility
    all         scan + visibility + partition
    """
    poisoned = rng.random((m, calls)) < p_poison
    fires = poisoned & (rng.random((m, calls)) < p_obey)
    if defence in ("scan", "both", "all"):
        fires &= rng.random((m, calls)) >= scan_detect
    if defence in ("visibility", "both", "all"):
        # A visible argument is only a defence if the user reads it, and
        # ch:ag-termination's habituation says attention is finite.
        caught = (rng.random((m, calls)) < vis_detect) & \
                 (rng.random((m, calls)) < vis_attention)
        fires &= ~caught
    if defence in ("partition", "all"):
        fires &= rng.random((m, calls)) >= partition_cover
    return float(fires.sum(1).mean()), float((fires.sum(1) > 0).mean())


print(f"{M:,} deployments, {CALLS} tool calls each. {P_POISON:.0%} of tools")
print(f"carry a hostile instruction in their metadata; the model acts on one")
print(f"{P_OBEY:.0%} of the time it is present.")
print()
print(f"{'defence':>26}{'harmful actions':>17}{'any harm':>11}{'reduction':>11}")
print("-" * 65)
tab = {}
base = run("none")
for name, label in (("none", "none"), ("scan", "static metadata scan"),
                    ("visibility", "parameter visibility"),
                    ("both", "scan + visibility"),
                    ("partition", "capability partition"),
                    ("all", "all three")):
    r = run(name)
    tab[label] = r
    print(f"{label:>26}{r[0]:>17.3f}{r[1]:>11.1%}"
          f"{1 - r[0] / base[0]:>11.0%}")

print()
print()
print("Parameter visibility depends on the user reading what is shown, and")
print("ch:ag-termination measured what happens to attention under volume.")
print()
print(f"{'user attention':>16}{'harmful actions':>17}{'reduction':>11}")
print("-" * 44)
at = {}
for a in (0.95, 0.60, 0.30, 0.10):
    r = run("visibility", vis_attention=a)
    at[a] = r
    print(f"{a:>16.0%}{r[0]:>17.3f}{1 - r[0] / base[0]:>11.0%}")

print()
print()
print("And how each defence holds as the poisoned share rises, which is what")
print("happens as an ecosystem grows and its registry admits more publishers.")
print()
print(f"{'poisoned tools':>16}{'none':>9}{'scan+vis':>11}{'partition':>12}"
      f"{'all three':>12}")
print("-" * 60)
ps = {}
for p in (0.005, 0.03, 0.10, 0.30):
    row = tuple(run(d, p_poison=p)[0]
                for d in ("none", "both", "partition", "all"))
    ps[p] = row
    print(f"{p:>16.1%}{row[0]:>9.3f}{row[1]:>11.3f}{row[2]:>12.3f}"
          f"{row[3]:>12.3f}")

print()
print()
print("The second problem: a description is read when the tool is APPROVED and")
print("acted on every time it is called. A server can change it in between.")
print()


def rugpull(recheck_every, m=M, calls=CALLS, p_turn=0.004, p_obey=P_OBEY,
            defence_cover=0.55):
    """A benign server turns hostile at some call and stays hostile until a
    re-verification of its metadata catches it. Walk the calls directly rather
    than trying to be clever about the bookkeeping."""
    hostile = np.zeros(m, dtype=bool)
    retired = np.zeros(m, dtype=bool)
    fires = np.zeros(m, dtype=np.int64)
    for t in range(calls):
        # A still-trusted server may turn at this call.
        turning = (~retired) & (~hostile) & (rng.random(m) < p_turn)
        hostile |= turning
        # A re-verification happens before the call is issued.
        if recheck_every and (t % recheck_every == 0):
            caught = hostile & (rng.random(m) < defence_cover)
            retired |= caught
            hostile &= ~caught
        live = hostile & ~retired
        fires += live & (rng.random(m) < p_obey)
    return float(fires.mean()), float((fires > 0).mean())


print(f"{'re-verify every':>17}{'harmful actions':>17}{'any harm':>11}")
print("-" * 45)
rp = {}
for k in (0, 20, 10, 5, 1):
    r = rugpull(k)
    rp[k] = r
    label = "never" if k == 0 else f"{k} calls"
    print(f"{label:>17}{r[0]:>17.3f}{r[1]:>11.1%}")

print(f"""
The first table reproduces ch:ag-security's central result in a new setting, and
the ordering is the one that chapter predicted.

Static metadata scanning -- cite:huang2026mcpthreat's first mitigation layer --
removes {1 - tab['static metadata scan'][0] / base[0]:.0%} of harmful actions.
Parameter visibility removes {1 - tab['parameter visibility'][0] / base[0]:.0%}.
Together they remove {1 - tab['scan + visibility'][0] / base[0]:.0%}.

The capability partition alone removes {1 - tab['capability partition'][0] / base[0]:.0%}
-- **more than both detection defences combined**.

The mechanism is the one ch:ag-security identified. A detector asks "is this
instruction hostile", which is a hard classification with an irreducible error
rate. A partition asks nothing: it arranges that a tool which reads untrusted
content cannot reach anything worth reaching, so obeying an injected instruction
accomplishes nothing. **You cannot classify your way out of a problem you can
structure your way out of.**

That is not an argument for skipping the detectors -- all three together reach
{1 - tab['all three'][0] / base[0]:.0%}, which is much better than any one. It is
an argument about ORDER. Build the partition first, because it is the layer whose
effectiveness does not depend on being right about what an attacker will write.

The second table is why parameter visibility is the weakest of the three, and it
is not a criticism of the mechanism. At {0.95:.0%} user attention it removes
{1 - at[0.95][0] / base[0]:.0%}; at {0.10:.0%} it removes
{1 - at[0.10][0] / base[0]:.0%}.

Showing the user the actual arguments is exactly right, and it inherits
ch:ag-termination's habituation: a user shown forty argument lists per session is
not reading the fortieth. **A defence whose effectiveness is a function of human
attention degrades with the volume it is deployed at**, which is precisely the
regime an agent creates.

The third table matters for ecosystem design rather than for a single deployment.
As the poisoned share rises from {0.005:.1%} to {0.30:.0%}, every defence degrades
proportionally -- none of them has a threshold. So a registry that admits more
publishers moves every connected host along this table at once, and the host
cannot tell that it happened.

That is the argument for cite:hou2025mcp's framing: most of what determines this
number is decided in the server LIFECYCLE -- who may publish, what is verified at
install, what provenance survives -- rather than at the protocol layer, and none
of it is visible from inside a session.

The last table is the problem the whole discussion usually omits. A description is
read when a tool is approved and acted on every time it is called, so approval is
a snapshot of a mutable thing (eq:approval-is-a-snapshot).

Never re-verifying gives {rp[0][0]:.3f} harmful actions and
{rp[0][1]:.1%} of deployments harmed. Re-verifying every call gives
{rp[1][0]:.3f} and {rp[1][1]:.1%}.

**Re-validation is the cheapest intervention here as it was in
ch:as-long-running and ch:mcp-primitives** -- the third time in this book that
re-reading something you already read turns out to be the best available move. The
pattern is consistent enough to state as a rule: **anything approved once and used
many times needs re-approval on a schedule**, and the schedule should be tighter
than feels necessary.""")
```

## 9. Practical Example

The first listing runs deployments of nine servers with six scopes each, at a
$2\%$ compromise rate:

```
     token model  any compromise  scopes reached  share of all    p99
---------------------------------------------------------------------
shared/passthrough           16.8%            9.08         16.8%     54
  audience-bound           16.5%            1.08          2.0%     12
audience + least           16.7%            0.36          0.7%      4
```

The compromise column barely moves — the same servers are hostile in all three
rows. What changes is what the compromise *reaches*
({{eq:audience-binding-bounds-the-union}}): audience binding is worth about $8\times$
and least privilege another $3\times$.

Scaling with connected servers:

```
  servers  passthrough  audience-bound  audience + least
--------------------------------------------------------
        2          0.5             0.2               0.1
       12         15.6             1.4               0.5
       80        384.9             9.6               3.2
```

**Passthrough makes blast radius quadratic in connected servers and audience
binding makes it linear** ({{eq:passthrough-is-quadratic}}). That is why the
specification writes MUST: a rule that looks like pedantry at three servers is
load-bearing at thirty, and every host is adding servers.

The honest limit:

```
  compromise rate  passthrough   audience  aud + least   ratio
--------------------------------------------------------------
             0.5%         2.37       0.27         0.09    26.1
            25.0%        49.95      13.52         4.50    11.1
```

**Structural controls help most when compromise is rare** — they bound damage
rather than prevent it.

And what each mechanism bounds:

```
       token model     mean     p99   max possible
--------------------------------------------------
       passthrough     9.05      54             54
    audience-bound     1.08      12             54
  audience + least     0.36       4             18
```

**Audience binding bounds the typical case; only scope minimisation lowers the
ceiling** ({{eq:scope-minimisation-lowers-the-ceiling}}). Implementing the first
alone leaves the tail exactly where it was.

The second listing runs forty tool calls with $3\%$ of tools poisoned:

```
                   defence  harmful actions   any harm  reduction
-----------------------------------------------------------------
                      none            0.741      52.5%         0%
      static metadata scan            0.330      28.3%        56%
      parameter visibility            0.510      40.1%        31%
         scan + visibility            0.231      20.5%        69%
      capability partition            0.148      13.8%        80%
                 all three            0.045       4.4%        94%
```

**The capability partition alone beats both detection defences combined**
({{eq:structure-beats-detection-again}}). A detector asks "is this hostile", which
is a classification an adversary optimises against; a partition asks nothing and
arranges that obeying accomplishes nothing.

Parameter visibility against attention:

```
  user attention  harmful actions  reduction
--------------------------------------------
             95%            0.250        66%
             30%            0.590        21%
             10%            0.694         7%
```

{{ch:ag-termination}}'s habituation, in a setting that generates exactly the volume
that destroys attention.

And the mutability problem:

```
  re-verify every  harmful actions   any harm
---------------------------------------------
            never            1.913      14.6%
         10 calls            0.866      13.3%
          1 calls            0.074       4.9%
```

**Approval is a snapshot of a mutable thing** ({{eq:approval-is-a-snapshot}}), and
re-validation is the cheapest fix here as everywhere else in this book.

## 10. Production Considerations

Send the `resource` parameter on both authorization and token requests, using the
server's canonical URI, whether or not your authorization server uses it.

Validate token audience at the server. A diligent client and a lax server buys
nothing.

Never accept or forward a token issued for something else. This is the
specification's MUST NOT, and {{eq:passthrough-is-quadratic}} is why.

Request minimal scopes and use the step-up flow, emitting all scopes an operation
needs in a single challenge rather than incrementally.

Validate `iss` per RFC 9207 with no URI normalisation before comparison.

Partition capabilities before building detectors. Structure is the layer whose
effectiveness does not depend on guessing the attack.

Show arguments before *consequential* operations, not before all of them — spend
attention where it changes an outcome.

Hash tool definitions at approval and compare on every listing. Treat a change as
requiring re-approval.

And remember that moving from stdio to HTTP creates an authorization surface that
did not exist during local development.

## 11. Common Mistakes

**Token passthrough because it is simpler.** Quadratic in connected servers.

**Audience binding without scope minimisation.** Fixes the mean, leaves the
catastrophe.

**Normalising URIs before `iss` comparison.** The spec forbids it because that is
where the bugs are.

**Challenging scopes incrementally.** Multiple browser round trips per action.

**Building a scanner before a partition.** Wrong order; the scanner's rate is set
by the adversary.

**Confirming every operation.** Spends the attention the consequential ones need.

**Treating approval as permanent.** It is a snapshot.

**Assuming compliance implies safety.** The specification says outright it "cannot
enforce these security principles at the protocol level."

## 12. Failure Modes

*Confused deputy via passthrough.* One hostile server using a token that every
other server honours.

*Tool poisoning.* {{cite:huang2026mcpthreat}}'s dominant client-side
vulnerability, riding the metadata channel every server has.

*Rug pull.* Benign at approval, hostile afterwards, with nothing re-checking.

*Approval fatigue.* {{ch:ag-termination}}'s habituation, at agent volumes.

*Installer spoofing.* {{cite:hou2025mcp}}'s pre-protocol threat, invisible to
everything in this chapter.

*Persisted injection.* {{cite:greshake2023indirect}}'s vector reaching durable
state, per {{ch:as-state-machines}}.

## 13. Alternatives

**Fewer servers.** {{eq:passthrough-is-quadratic}} and
{{eq:blast-radius-is-a-union}} both scale in $n$, so the cheapest control is connecting
less.

**Reader/actor split.** {{ch:ag-security}}'s partition, which
{{sec:9-practical-example}} finds the strongest single defence.

**A trusted internal registry.** Sets $\pi$ and $\lambda$ by policy rather than
hoping, which {{eq:lifecycle-decides-the-rate}} says is where they are actually
determined.

**Host-side description rewriting.** Normalise tool text to a template before
presenting it, discarding anything not needed for selection — at the cost of the
distinguishing detail {{ch:mcp-schemas}} says selection needs.

**Sandboxed server execution.** Contains a compromised server's local effects, and
does nothing about what its tokens reach.

## 14. Evaluation

Audit every token your host holds: which server it is bound to, which scopes it
carries, and whether any server would accept a token issued for another. The last
question is the passthrough test.

Compute your capability union per server, as {{ch:ag-security}} describes, and
multiply by connected server count to get your position on the second table.

Measure detector rates against *adaptive* inputs, not a fixed corpus. A rate
measured on known samples is an upper bound that an adversary removes.

Measure user attention at your real call volume, not in a usability study of the
first three confirmations.

Diff tool definitions across listings and count changes. That is your $\lambda$,
and nothing else reveals it.

And test the migration from stdio to HTTP explicitly, since it introduces every
decision here at once.

## 15. Advanced Concepts

**Capability attenuation in the token itself.** Macaroons and similar constructions
let a holder narrow a token before passing it on, which would make delegation safe
where passthrough is not. Not in the specification.
{{maturity:EXPERIMENTAL}}.

**Signed tool definitions.** If descriptions were signed and pinned at approval, the
rug pull becomes detectable without re-fetch comparison, and
{{eq:approval-is-a-snapshot}} closes structurally.
{{maturity:EMERGING}}.

**Provenance-carrying context.** Marking each span of context with its origin so a
model — or a wrapper around it — can weight instructions by source. The general
answer to {{cite:greshake2023indirect}} and still open.
{{maturity:RESEARCH FRONTIER}}.

**Measuring $\pi$ across a real registry.** {{cite:hou2025mcp}} and
{{cite:gaire2025mcpsok}} taxonomise threats; the poisoned fraction in a live
ecosystem is not published and is the parameter every defence scales against.

## 16. Connection to Previous Chapters

{{ch:ag-security}}'s {{eq:blast-radius-is-a-union}} is what audience binding bounds, and
its structure-over-detection finding is reproduced here in a setting that chapter
did not consider.

{{ch:ag-termination}}'s habituation determines the ceiling on parameter
visibility, and its consequence gate is the fix.

{{ch:as-long-running}}'s re-validation appears for the third time as the cheapest
available intervention, now against metadata mutation.

{{ch:mcp-schemas}}'s rent argument gains a second reason to prefer showing fewer
tools: fewer descriptions is less instruction channel.

{{ch:mcp-architecture}}'s stdio-versus-HTTP choice turns out to be an
authorization-surface choice as well as a correlation one.

Ahead: {{ch:mcp-building}} implements a server and client with these requirements
in place; {{ch:mcp-production}} takes up the registry and lifecycle policy that
{{eq:lifecycle-decides-the-rate}} says sets the parameters everything here scales
against.

## 17. Exercises

1. Derive {{eq:passthrough-is-quadratic}} and compute the crossover server count
   at which passthrough exposure exceeds audience-bound by $10\times$.

2. Add partial passthrough — some servers accept foreign tokens, some do not — and
   find how much non-compliance destroys the guarantee.

3. Model an adaptive attacker whose evasion raises with detector strength, and
   re-run the defence comparison.

4. Implement signed, pinned tool definitions in the second listing and compare
   against re-verification.

5. Combine consequence-gated visibility with the partition and check whether the
   effects are independent.

6. Estimate your own $\lambda$ by diffing a real server's tool definitions over
   time.

## 18. Interview Questions

1. Why does the specification forbid token passthrough rather than discourage it?

2. What does the `resource` parameter do, and why send it when the authorization
   server ignores it?

3. You implemented audience binding. What is still unbounded?

4. Rank the defences against tool poisoning and justify the order.

5. Why is parameter visibility weaker in an agent than in a normal application?

6. A server was reviewed and approved six months ago. What have you assumed?

## 19. Research Questions

1. Would token attenuation make safe delegation possible where passthrough is
   forbidden?

2. Can tool definitions be signed and pinned without breaking legitimate updates?

3. What is the poisoned fraction in a live MCP registry?

4. Can provenance be carried through context in a form a model actually weights?

5. How much does detector performance degrade under adaptive pressure in this
   specific setting?

## 20. Chapter Summary

MCP's authorization requirements reduce to one structural claim: a token must be
useless anywhere but where it was meant to be used. Servers **MUST** validate
audience, clients **MUST** send the RFC 8707 `resource` parameter regardless of
support, and servers **MUST NOT accept or transit any other tokens**
({{cite:mcp2026spec}}).

{{sec:9-practical-example}} prices it. Under passthrough one compromised server
reaches $9.08$ of $54$ scopes on average and all $54$ at p99; audience binding
gives $1.08$ and least privilege $0.36$
({{eq:audience-binding-bounds-the-union}}). The reason for MUST is the scaling:
from two to eighty servers, passthrough exposure grows $0.5 \to 384.9$ against
audience binding's $0.2 \to 9.6$ — **quadratic against linear**
({{eq:passthrough-is-quadratic}}).

The two controls bound different things. Audience binding lowers the mean and
leaves the maximum at $54$; scope minimisation lowers the ceiling to $18$
({{eq:scope-minimisation-lowers-the-ceiling}}). Implementing the first alone leaves
the catastrophe untouched.

Authorization does not address the attack that dominates in practice.
{{cite:huang2026mcpthreat}} found **tool poisoning the most prevalent and impactful
client-side vulnerability** across seven clients — structural, because a
description is text reaching the model and therefore an instruction.

Ranked: static scanning $-56\%$, parameter visibility $-31\%$, both $-69\%$, and
**a capability partition alone $-80\%$**
({{eq:structure-beats-detection-again}}). {{ch:ag-security}}'s ordering, reproduced.
Detection asks a question an adversary optimises against; structure asks nothing.
Parameter visibility further inherits {{ch:ag-termination}}'s habituation, falling
from $-66\%$ at high attention to $-7\%$ at low.

And approval is a snapshot: never re-verifying metadata gave $1.913$ harmful
actions against $0.074$ when re-verified every call
({{eq:approval-is-a-snapshot}}) — **the third time in this book that re-reading
something already read is the cheapest available intervention.**

Finally, every rate above is set outside the session. {{cite:hou2025mcp}}'s
lifecycle — who may publish, what provenance, what install-time verification —
determines $\pi$ and $\lambda$ ({{eq:lifecycle-decides-the-rate}}), and a host
cannot see either from where it stands.

## 21. Further Reading

{{cite:mcp2026spec}}'s authorization section is the primary source and repays
direct reading — particularly the token-handling requirements and the `iss`
validation table, whose prohibition on URI normalisation is the kind of detail that
exists because of an incident.

{{cite:huang2026mcpthreat}} for the client evaluation behind
{{sec:9-practical-example}}'s second listing, and
{{cite:hou2025mcp}} for the lifecycle framing that says most of the threat surface
precedes the protocol.

{{cite:gaire2025mcpsok}} for the separation of adversarial threats from
non-adversarial safety failures, which is the right frame for reading
{{cite:cemri2025mast}}'s multi-agent modes alongside this chapter's.

{{ch:ag-security}} for the capability union and the structure-over-detection
result this chapter reproduces, and {{cite:greshake2023indirect}} for the injection
vector underneath all of it.
