# Part XIX research notes — MCP and Tool Ecosystems

Research pass 2026-08-29. Everything below was verified against a primary source
on that date; nothing here is from memory.

## The most important finding of this pass

**The current MCP revision is `2026-07-28`, and it is architecturally different
from the revision most written material describes.**

Verified at `modelcontextprotocol.io/specification/versioning` and
`/specification/2026-07-28`. The changes that matter for teaching:

| | 2025-06-18 and earlier | 2026-07-28 |
|---|---|---|
| Connection model | stateful connections | **stateless, self-contained requests** |
| Negotiation | `initialize` handshake, connection-scoped | **per-request**, via `_meta.io.modelcontextprotocol/protocolVersion` |
| Discovery | implied by handshake | **mandatory `server/discover` RPC** (calling it is optional) |
| Message directions | servers could initiate JSON-RPC requests | **servers do not initiate requests; clients do not send responses** |
| Client features | Sampling, Roots, Elicitation | **Elicitation only** (rest moved to extensions) |
| Utilities | Config, Progress, Cancellation, Errors, Logging | Config, Progress, Cancellation, Errors (**no Logging**) |
| Transports | stdio, HTTP+SSE, Streamable HTTP | **stdio, Streamable HTTP** |

Anything written against the handshake model describes a different protocol.
Backward compatibility is defined for `2025-11-25` and earlier.

Version strings are `YYYY-MM-DD` and increment **only on backwards-incompatible
change**. Unsupported versions produce `UnsupportedProtocolVersionError` listing
supported versions; the client retries. Clients and servers MAY support several
versions at once.

## Architecture facts (verified from the spec)

- Base: **JSON-RPC 2.0**, UTF-8. Explicitly inspired by the Language Server
  Protocol.
- Roles: **Hosts** (LLM applications that initiate connections), **Clients**
  (connectors within the host), **Servers** (services providing context and
  capabilities).
- Server features: **Resources** (context/data), **Prompts** (templated messages
  and workflows for users), **Tools** (functions the model executes).
- Client features: **Elicitation** (server-initiated requests for more
  information from users).
- Extensions (opt-in, negotiated): **Tasks** (async long-running operations with
  polling, mid-flight input, durable handles), **Skills over MCP**, **MCP Apps**
  (inline interactive UI).

### Transports

A transport is a **binding**: it defines framing, delivery, request metadata,
cancellation and termination. Protocol semantics are identical on every binding.

1. **stdio** — newline-delimited JSON-RPC over the standard streams of a
   client-launched subprocess. Cancellation: `notifications/cancelled`.
2. **Streamable HTTP** — each message is an HTTP POST to a single MCP endpoint;
   replies arrive as a JSON object or a request-scoped SSE stream. Cancellation:
   the client closes the response stream. Mirrors `_meta` into HTTP headers
   (including `MCP-Protocol-Version`) so intermediaries can route without parsing
   the body — **the body remains the source of truth**.

Custom transports MUST preserve JSON-RPC format, message patterns and the
per-request metadata model. Custom transports over a reliable bidirectional byte
stream SHOULD reuse stdio framing.

### Authorization (verified from `/basic/authorization`)

Authorization is **OPTIONAL**; for HTTP transports it SHOULD follow the spec, and
**stdio implementations SHOULD NOT** — they take credentials from the environment.

- Based on **OAuth 2.1** (draft-ietf-oauth-v2-1-13), plus RFC 6750, RFC 8414,
  RFC 7591, **RFC 8707**, **RFC 9728**, **RFC 9207**, OIDC Discovery.
- Roles: MCP server = OAuth 2.1 **resource server**; MCP client = OAuth 2.1
  **client**; authorization server may be co-hosted or separate.
- MCP servers **MUST** implement RFC 9728 Protected Resource Metadata; clients
  **MUST** use it for AS discovery.
- Clients **MUST** implement RFC 8707 resource indicators — `resource` in *both*
  authorization and token requests, identifying the MCP server by canonical URI,
  **regardless of whether the AS supports it**.
- Servers **MUST** validate that tokens were issued for them as audience.
  **"MCP servers MUST NOT accept or transit any other tokens."** — this is the
  explicit prohibition on token passthrough.
- Clients **MUST NOT** send tokens other than those issued by the server's AS.
- **Client ID Metadata Documents** (HTTPS URL as `client_id`) are the SHOULD;
  **Dynamic Client Registration (RFC 7591) is deprecated**, retained for
  backwards compatibility.
- RFC 9207 `iss` validation is required before transmitting the code, with a
  four-row table keyed on `authorization_response_iss_parameter_supported`. No
  URI normalisation before comparison.
- Scope: 401 carries `WWW-Authenticate` with `resource_metadata` and `scope`;
  403 + `error="insufficient_scope"` for step-up. Clients compute the **union**
  of previously requested and challenged scopes. Servers SHOULD emit all needed
  scopes in a single challenge rather than incrementally.
- Errors: 401 unauthorized/invalid token, 403 insufficient scope, 400 malformed.

### Security posture stated by the spec itself

Four principles: user consent and control; data privacy; **tool safety**; LLM
sampling controls. The spec states outright that **tool annotations "should be
considered untrusted, unless obtained from a trusted server"**, and that MCP
"cannot enforce these security principles at the protocol level".

That last sentence is the honest framing for the security chapter: the protocol
defines a trust boundary and then declines to police it.

## Papers verified this pass

**Tool ecosystems at scale**

- `qin2023toolllm` — ToolLLM (2307.16789, 31 Jul 2023, 19 authors). ToolBench over
  **16,000+ real REST APIs across 49 categories**; ToolEval; ToolLLaMA comparable
  to ChatGPT, generalises to unseen APIs. *The paper that made the
  large-inventory regime measurable.*
- `patil2023gorilla` — Gorilla (2305.15334, 24 May 2023, 4 authors). APIBench over
  HuggingFace/TorchHub/TensorHub; surpasses GPT-4 on writing API calls;
  **retriever coupling reduces hallucination and tracks documentation changes
  without retraining**. *The argument for discovery-at-runtime rather than
  baked-in knowledge.*
- `li2023apibank` — API-Bank (2304.08244, 14 Apr 2023, 9 authors). **73 tools,
  314 dialogues, 753 calls**; training set of 1,888 dialogues from 2,138 APIs
  across 1,000 domains; Lynx beats baseline by 26+ points. *Decomposes tool use
  into stages — the per-stage view this book keeps arguing for.*

**MCP security**

- `hou2025mcp` — MCP Landscape/Threats (2503.23278, 30 Mar 2025, 4 authors).
  **Four lifecycle phases** (creation, deployment, operation, maintenance) in
  **16 activities**; **16 threat scenarios** across **4 attacker categories**;
  tool poisoning and installer spoofing named. *Reframes MCP security as a
  supply-chain problem — most threats land before any protocol message.*
- `huang2026mcpthreat` — MCP Threat Modeling / Tool Poisoning (2603.22489,
  23 Mar 2026, 4 authors). STRIDE+DREAD over **five components**; **seven MCP
  clients evaluated**; **tool poisoning is the most prevalent and impactful
  client-side vulnerability**, from insufficient static validation and parameter
  visibility. Four-layer mitigation.
- `gaire2025mcpsok` — SoK MCP (2512.08290, 9 Dec 2025, v2 13 Dec, 6 authors).
  Separates **adversarial threats from safety failures**; organised around the
  three primitives; defences from cryptographic verification to runtime intent
  validation.

**Reference**

- `mcp2026spec` — the specification itself, revision 2026-07-28.

## Rejected during verification

- **`arXiv:2603.07473`** ("Give Them an Inch and They Will Take a Mile:
  Understanding and Measuring Caller Identity Confusion in MCP-Based AI
  Systems") — **withdrawn by its authors** for "flaws in experimental methodology
  and unresolved ethical issues in data collection". Its claim — that most MCP
  servers rely on persistent authorization state and re-invoke without
  re-authentication regardless of caller — is interesting and **must not be
  cited**, and no number from it may be used.

This is the second withdrawn-paper catch in the project and the reason the
verification rule exists: the abstract read as a normal result.

## Chapter plan

| ch | id | measurement to build |
|---|---|---|
| 170 | `mcp-why` | N×M integration cost vs N+M; the protocol's break-even in integration count |
| 171 | `mcp-architecture` | stateless vs stateful request handling under failure/restart |
| 172 | `mcp-primitives` | tool vs resource vs prompt — what each costs in context and control |
| 173 | `mcp-schemas` | discovery over large inventories; context budget vs retrieval |
| 174 | `mcp-security` | tool poisoning and the confused deputy; audience binding |
| 175 | `mcp-building` | a working server and client, tier-A |
| 176 | `mcp-production` | registry economics, versioning, and ecosystem failure |

Carry-through from Part XVIII: `distinctness-not-count` (ch154),
`verifier-sets-the-ceiling` (ch168), `gate-on-consequence` (ch160),
`agent-errors-correlate` (ch169). The tool-poisoning material connects directly
to `greshake2023indirect` and `ch:ag-security`.
