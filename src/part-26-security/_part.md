---
id: part-26-intro
status: draft
---

## What this part is for

{{part:25}} asked whether the system works. This part asks what happens when somebody wants it
not to.

**AI security is different from software security in exactly one respect, and everything else
follows from it: there is no parser.**

A parameterised SQL query tells the database, structurally, which bytes are code and which are
data, and nothing inside the data can move that line. A model receives one sequence. In a
typical agent context **79% of the tokens are content the system did not author**, and the
model cannot distinguish any of it from the system prompt except by position — which is a
learned prior, not a boundary.

So the oldest guarantee in application security inverts. In SQL, in a shell, in HTML, an
attacker who controls data controls data. **In a prompt, an attacker who controls data controls
intent.**

> **The rule adopted for this part: a control that can be searched around is a cost-raiser, not
> a boundary.** Every chapter separates the two, prices each, and reports the utility a real
> boundary costs. There are no free controls here, and a design presented without its utility
> column will be reversed the first time it blocks something real.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| Share of an agent context that is untrusted | **$79\%$** | {{ch:sec-threat-model}} |
| Attack surface at 4 sources × 8 sinks, then 8 × 16 | **32** then **128** | {{ch:sec-threat-model}} |
| Detection stack success at 1 attempt, then 100 | **$12.33\%$** → **$100\%$** | {{ch:sec-threat-model}} |
| Capability stack success at 1 attempt, then 1000 | **$0.2436\%$** → **$0.2436\%$** | {{ch:sec-threat-model}} |
| Sessions one poisoned document reaches in 90 days | **141,372** | {{ch:sec-prompt-injection}} |
| Cost per compromise, indirect against direct | **\$0.0006** vs **\$0.0040** | {{ch:sec-prompt-injection}} |
| Output scanning against leaks, then hijacks | **$88\%$** then **$4\%$** | {{ch:sec-prompt-injection}} |
| Jailbreak success, plain English against base64 | **0.06** against **0.53** | {{ch:sec-jailbreaks}} |
| Uncovered capability domains, once the lag is steady | **5.8**, constant | {{ch:sec-jailbreaks}} |
| Guardrail alarms that are wrong at a 0.3% base rate | **$94\%$** | {{ch:sec-jailbreaks}} |
| Legitimate users refused per prevented harm | **41** | {{ch:sec-jailbreaks}} |
| Extraction of a single-occurrence secret, 1.5B → 400B | **0.082** → **0.333** | {{ch:sec-data-leakage}} |
| Memorisation's share of leaked records | **$2.9\%$** | {{ch:sec-data-leakage}} |
| Cache cross-tenant exposure, measured against reported | **433 : 1** | {{ch:sec-data-leakage}} |
| Reachable records outside the requester's entitlement | **$99.96\%$** | {{ch:sec-tool-abuse}} |
| `search + email` damage against the sum of its parts | **14.0** against **3.0** | {{ch:sec-tool-abuse}} |
| Damage that is permanent, not recoverable | **$57\%$** | {{ch:sec-tool-abuse}} |
| Cost to poison 1% of a dataset, any size | **\$6,000** | {{ch:sec-poisoning}} |
| Targeted backdoor against broad degradation | **\$1** against **\$36,000** | {{ch:sec-poisoning}} |
| Composite supply-chain trust over eight links | **0.5570** | {{ch:sec-poisoning}} |
| Approval time per item at a per-call gate | **0.7 seconds** | {{ch:sec-permissions}} |
| Authority the user holds, against what executes | **0.08** against **1.00** | {{ch:sec-permissions}} |

## The organising idea

**Every chapter finds a control aimed at a unit that is not the unit of danger.**

{{part:22}}'s instrument was silent, {{part:23}}'s pointed at the wrong quantity,
{{part:24}}'s reported without bounding, {{part:25}}'s was precise about something adjacent.
This part's controls are correct, well-implemented, and measuring the wrong noun.

```text
   CHAPTER                  THE CONTROL BOUNDS          THE DANGER IS IN
   ──────────────────────   ─────────────────────────   ────────────────────────
   221 threat model         the probability             the set of reachable sinks
   222 prompt injection     the request channel         a document written last year
   223 jailbreaks           one prompt                  a domain nobody covered
   224 data leakage         the weights                 the cache, the log, the index
   225 tool abuse           the code, and the call      the credential, and the sequence
   226 poisoning            the artefact's identity     the artefact's contents
   227 permissions          the acting identity         the principal chain
```

Read the right column downward. Not one of those is exotic, and not one is what the
corresponding control was built for — because each control was imported from a threat model
where the noun in the left column *was* the danger. Sandboxes came from untrusted code.
Signatures came from substitution attacks. Rate limits came from attackers who send requests.
**All of them are correct about the world they came from**, and the world changed underneath
them.

## The three through-lines

**First: detection bounds a probability, capability bounds a set — and only one survives
repetition.**

This is the part's load-bearing result and it recurs in every chapter:

| Chapter | The detector | Its asymptote | The capability limit |
|---|---|---|---|
| {{ch:sec-threat-model}} | four stacked filters | $\to 100\%$ at 100 tries | $0.2436\%$, flat |
| {{ch:sec-prompt-injection}} | injection classifier | searched around | tool allow-list |
| {{ch:sec-jailbreaks}} | guardrail at 58% | $0.0\%$ at 30 tries | refuse the capability |
| {{ch:sec-poisoning}} | six poison detectors | miss $14.6\%$ of targeted | re-host the corpus |
| {{ch:sec-permissions}} | a human reviewer | $0.7$ s per item | attenuate at the hop |

**A blocked attempt is information**, so any control the attacker can query converges to
failure at a rate set by their attempt budget — and
{{cite:zou2023universal}}'s transferability means that budget is not on your platform.

**Second: the effective fixes are subtractions, and they never look like security work.**

| Chapter | The expensive control | The cheap subtraction |
|---|---|---|
| {{ch:sec-prompt-injection}} | an output scanner | delete the session token from the prompt |
| {{ch:sec-jailbreaks}} | a better guardrail | do not connect the capability |
| {{ch:sec-data-leakage}} | DP-SGD at 22 units | key the cache by tenant, 0.4 units |
| {{ch:sec-tool-abuse}} | approving 20,640 calls | remove a tool |
| {{ch:sec-poisoning}} | six detection methods | re-host the corpus |

In every row the subtraction wins by an order of magnitude or more, costs days rather than
quarters, and appears on no security roadmap — because a config change to a cache does not look
like a control, and a classifier does.

**Third: every one of these problems is a product decision wearing a security costume.**

Whether the system acts at all. What is in the prompt. How many tools are connected. Whether
the cache is shared. Whether the corpus is yours. What outcome classes exist. Each is decided
by a product manager, months before a security review, and each bounds a risk no security
control can reach afterwards.

**The largest single move available in this part — proposal-only execution, taking blast radius
from 100 to 3 — is not a security control at all.** It is a decision about what the product
does.

## What this part does not settle

**Correlated failure is assumed away everywhere.** Between detection layers, between
supply-chain links sharing an upstream, between annotators and the judges trained on them. The
corrections all run against the optimistic reading and none is measured.

**No empirical attempt-budget data exists.** The whole detection-versus-capability argument
turns on how many tries an attacker gets, and this book found no published distribution for
real attacks against LLM-integrated products.

**The cost ratios are never stated.** {{ch:sec-jailbreaks}}'s guardrail threshold moves 40×
across a harm-to-refusal ratio nobody writes down, which is
{{eq:f1-asserts-a-cost-ratio}} arriving where users pay for the assumption.

**And the OWASP-style practitioner taxonomies could not be cited.** They are the standard
industry reference and they are not arXiv preprints, so under this book's verification rule
they are absent. The threat structure here is derived from first principles and from the
verified papers, and says so wherever it matters.

## How to read this part

{{ch:sec-threat-model}} is load-bearing for the other six. Its detection-versus-capability
result is used in every subsequent chapter and is the single most transferable idea in the
part.

If you are shipping an agent this quarter: {{ch:sec-tool-abuse}} and {{ch:sec-permissions}} are
the two to read together, because they reach the same recommendation — gate outcomes, not calls
— from composition and from human capacity independently, and the agreement is the strongest
signal in the part.

If you are running a retrieval system: {{ch:sec-prompt-injection}} and {{ch:sec-data-leakage}}
between them contain four fixes costing under a week — key the cache by tenant, redact at emit,
filter before ranking, trim the prompt — and on the numbers here those four beat everything
else in the part combined.
