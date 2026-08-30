---
id: sec-tool-abuse
number: 225
part: XXVI
tier: full
status: draft
requires: [attack-surface-is-sources-times-sinks, only-capability-limits-bound-the-damage,
           delegation-moves-the-check, rollback-restores-code-not-state]
provides: [agent-authority-exceeds-requester-authority, a-sandbox-without-scoped-credentials-moves-nothing,
           tool-damage-composes-superadditively, approval-must-sit-at-the-outcome-not-the-call]
citations: [debenedetti2024agentdojo, beurerkellner2025patterns, cemri2025mast, hou2025mcp]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute an agent's excess authority relative
to its requester and identify where it concentrates; distinguish the axis a sandbox bounds
from the axis a scoped credential bounds, and explain why neither substitutes; explain why a
credential the agent can read is a credential the attacker has; show that tool damage composes
superadditively and that subset count grows faster than review capacity; classify actions by
reversibility and find where permanent damage actually sits; and design an approval gate that
sits at the outcome rather than at the call.

## 2. Why This Matters

An agent is a confused deputy by construction. It runs with a service credential provisioned
for the union of what any user might need, while each requester is entitled to a slice. In the
system modelled here, **99.96%** of reachable records lie outside the median requester's
entitlement ({{eq:agent-authority-exceeds-requester-authority}}), and damage-weighted, **99.3%**
of the excess is the cross-tenant class alone.

Sandboxing is the usual reflex and it addresses a different axis. A sandbox removes filesystem
access, outbound network, process spawning and environment reads. It does not remove **using
an injected credential** or **calling a tool the agent has** — both available at 100%
({{eq:a-sandbox-without-scoped-credentials-moves-nothing}}) — because both are things the agent
is supposed to do, from inside the box.

The second half is composition. `search + email` is worth **14.0** against a sum of parts of
**3.0**; `search + exec + webhook` is **24.0** against **8.5**
({{eq:tool-damage-composes-superadditively}}). Across six compositions the superadditive excess
is **58.0** against a sum-of-parts total of **22.0** — the compositions carry more danger than
the tools.

And reviews cannot keep up: 20 tools have **190** pairs, **1,140** triples and **1,048,555**
subsets, reviewed one at a time. Approving every call asks **20,640** times a day; approving by
outcome class covers **94%** of compositions at **410**
({{eq:approval-must-sit-at-the-outcome-not-the-call}}).

## 3. Prerequisites

{{eq:attack-surface-is-sources-times-sinks}} from {{ch:sec-threat-model}} counted the paths;
this chapter prices what is at the end of them and finds the pricing is not additive.

{{eq:only-capability-limits-bound-the-damage}} from the same chapter is the control that works,
and {{sec:9-practical-example}} shows there are two independent capability axes — code reach
and credential scope — that teams routinely conflate.

{{eq:delegation-moves-the-check}} from {{ch:sd-apis-auth}} is the fix for the authority axis,
and here it is priced against the alternatives.

{{eq:rollback-restores-code-not-state}} from {{ch:ops-deployment}} returns as the reversibility
result: reads cannot be unread, and reads are the actions usually classified as safe.

{{cite:hou2025mcp}}'s threat analysis of the tool-integration layer is the background for what
"a tool the agent has" means in a protocol that lets tools be added at runtime.

## 4. Intuitive Explanation

There is a very old security bug called the confused deputy. A program with more authority
than its caller performs an action on the caller's behalf, using its own authority rather than
theirs, and the caller ends up doing something they were not entitled to do.

An AI agent is a confused deputy by design. It cannot ask each user for their credentials for
every backend, so it is given a service account, and the service account is provisioned for the
union of everything any user might need. Every request then runs with that union.

Count it. The service credential reaches 31 million records. The median requester is entitled
to their own records, some of their team's, a fraction of their tenant's — a small slice.
99.96% of what the agent can reach is outside the requester's entitlement.

Damage-weighted it is starker: 99.3% of the weighted excess is the "all tenants' records"
class, which no individual user is ever entitled to and which the service account reaches
because it serves all of them.

That is not a misconfiguration to be found in an audit. It is the definition of a service
account.

The fix for this axis is {{ch:sd-apis-auth}}'s delegation result: pass the requester's
authority rather than the deputy's. Per-tenant service accounts take the weighted excess from
125 million to 974 thousand for about 5% of engineering effort — the cross-tenant path closed
and the in-tenant one open. Per-user delegated tokens take it to zero for about 22%.

Now the other reflex, which is sandboxing, and this is where the chapter's sharpest point sits.

A sandbox is an excellent control. It removes filesystem access outside a working directory,
outbound network, process spawning, and environment reads. If your agent generates and runs
code, you want one.

Look at what it does not remove. **Using an injected credential**: available 100% of the time,
sandbox removes it — no. **Calling a tool the agent has**: 100%, sandbox removes it — no.

Both of those are things the agent is *supposed* to do. The credential is in the sandbox
because the agent needs it. The tool handles are in the sandbox because the agent needs them. A
sandbox is a boundary around the code, and the authority is inside the boundary.

Which means: **a sandbox with an unscoped credential in its environment has moved the boundary
and not the authority.** An agent hijacked by a prompt injection does not need to escape the
sandbox. It needs to call `issue_refund`, which is right there.

That is the confusion worth removing from most architecture discussions. A sandbox is the
right control for *untrusted code*. It is not a control for an *untrusted instruction reaching
a trusted tool*, which is what agent hijacking is.

The two axes are not substitutes, and the posture table makes it explicit: sandboxing alone
moves code escape from 58% to 4% and leaves excess authority untouched; delegation alone
reverses that. Only doing both moves the residual — and each is usually owned by a different
team with a different threat model.

There is one implementation detail that decides whether scoping is real at all. Where does the
credential physically live?

If it is an environment variable, or a mounted secret file, the agent can read it. A hijacked
agent reads it and uses it *outside* every control you built — outside the tool allow-list,
outside the approval flow, outside the sandbox it is sitting in, because the credential works
anywhere.

If it is injected into the tool call by a broker, or held by a proxy that signs outbound
requests, it is never in the agent's context. **Scoping a credential the agent can read is
scoping a credential the attacker also has.**

That is the second half of the first listing. The second listing is about a different failure
and it is the one that survives all of the above.

Take seven tools. Search internal documents. Read a customer record. Send an email. Write to
the CRM. Issue a refund. Execute code. Post to a webhook.

Review them individually and each is fine. Search reads what the user may read. Email sends
what the user may send. The sum of individual damages is 22.

Now compose them. Search plus email is exfiltration: read anything the agent can reach, send it
anywhere. Individually 1.0 and 2.0, sum of parts 3.0. As a composition, 14.0.

Read plus webhook is bulk export with no log on the far side: 4.0 as a sum, 12.0 as a
composition. Search plus execute plus webhook is read-transform-exfiltrate: 8.5 as a sum, 24.0
as a composition.

Across six compositions the superadditive excess is 58.0 against a sum-of-parts total of 22.0.
**The compositions carry more danger than the tools do**, and a per-tool review sees none of
it.

This cannot be fixed by reviewing harder, and the counting table says why. Seven tools have 21
pairs and 35 triples. Twenty tools have 190 pairs, 1,140 triples, and 1,048,555 subsets of size
two or more. Reviews scale with tool count; risk scales with subset count.

That is {{ch:sec-threat-model}}'s product result one level up, and it is worse — there the
surface grew as sources times sinks, here it grows exponentially in tool count.

Before the design conclusion, one more input: reversibility.

Classify the seven actions by how much of their damage can be undone. Writing to the CRM: 85%
recoverable. Issuing a refund: 70%. Executing code: 40%. Sending an email: 5%.

Searching internal documents: 0%. Reading a customer record: 0%. Posting to a webhook: 0%.

**A read cannot be unread.** The actions usually classified as "safe, read-only, no approval
needed" are the least reversible things on the list, and 57% of total damage is permanent.
That is {{ch:ops-deployment}}'s rollback result arriving in the tool layer: the artefact can be
reverted, the disclosure cannot.

So where does the approval go?

Approving every call covers everything and asks 20,640 times a day. Nobody reads 20,640
approvals. That is not a control; it is a rubber stamp with an audit trail, and the audit trail
is the only part that survives.

Approving by **outcome class** covers 94% of compositions at 410 approvals a day and 9%
utility cost.

The difference is what the gate is looking at. A per-call gate asks "may this agent call this
tool?" and the answer is yes — for both halves of the composition, separately, correctly. An
outcome gate asks "is data about to leave the tenant boundary?", which is a question about the
effect of a sequence.

Five outcome classes cover it here. Data leaves the tenant boundary. Money moves. A record
other agents read changes. Something irreversible happens. Privilege is used outside the task.
410 approvals a day between them.

**Enumerate outcomes, not tools.** Outcomes are few and stable. Tools are many and grow every
sprint, and the subsets grow faster than either.

## 5. Formal Explanation

**Excess authority.** Let the agent's credential reach resource classes $j$ with $r_j$ records
and damage $d_j$, and let requester entitlement be $e_j \in [0,1]$. Excess is $\sum_j r_j(1 -
e_j)$ and weighted excess is $\sum_j r_j (1-e_j) d_j$. Since a service account is provisioned
as $\bigcup_u \text{needs}(u)$ and any single requester needs a small subset, the ratio
approaches one as the tenant count grows. **The excess is a function of multi-tenancy, not of
carelessness.**

**Two axes.** Let $\sigma$ index code-reach controls (sandboxing) and $\alpha$ index
authority-scope controls (delegation). Residual risk is $R(\sigma, \alpha)$ with
$\partial R/\partial\sigma$ acting on code escape and $\partial R/\partial\alpha$ on excess
authority. Because the two terms are additive rather than multiplicative in the damage, neither
control reduces the other's term — they are complements, not substitutes.

**Credential locality.** A credential is *reachable* if it appears in any context the model
sees. Reachable credentials are usable outside every downstream control, so a scoping control
on a reachable credential bounds only the paths that route through the control, and the
attacker does not have to.

**Superadditivity.** For a tool set $T$, define $D(S)$ for $S \subseteq T$. Individual review
establishes $D(\{t\})$ for each $t$. Composition is superadditive when $D(S) > \sum_{t\in S}
D(\{t\})$, which holds whenever tools supply complementary primitives — a reader and a writer,
a computer and a channel. Since $|\{S : |S| \ge 2\}| = 2^{|T|} - |T| - 1$ and review capacity
is $O(|T|)$, the fraction reviewed goes to zero exponentially.

**Approval placement.** A per-call gate evaluates a predicate on $(\text{agent}, t)$. A
composition is a sequence, and no predicate on individual elements can express a property of
the sequence unless state is carried. An outcome gate evaluates a predicate on the *effect* —
data crossing a boundary, funds moving, state becoming irreversible — which is a property of
the sequence's result and is therefore expressible at a single point.

## 6. Mathematical Foundation

Excess authority as a property of multi-tenancy:

$$X = \sum_j r_j (1 - e_j), \qquad X_w = \sum_j r_j (1-e_j) d_j, \qquad \frac{X}{\sum_j r_j} = 99.96\%$$ (eq:agent-authority-exceeds-requester-authority)

with **99.3%** of $X_w$ in the cross-tenant class.

The two axes, and what a sandbox does not reach:

$$R(\sigma,\alpha) = c(\sigma) + a(\alpha), \qquad \frac{\partial c}{\partial \alpha} = 0, \qquad \frac{\partial a}{\partial \sigma} = 0$$ (eq:a-sandbox-without-scoped-credentials-moves-nothing)

Sandboxing takes $c$ from 58% to 4% and leaves $a$ at 124,841,115.

Superadditive composition:

$$D(S) > \sum_{t \in S} D(\{t\}), \qquad \sum_S \left[D(S) - \textstyle\sum_t D(\{t\})\right] = 58.0 \ \text{vs}\ \textstyle\sum_t D(\{t\}) = 22.0$$ (eq:tool-damage-composes-superadditively)

with $2^{20} - 21 = 1{,}048{,}555$ subsets against $O(20)$ reviews.

And where the gate must sit:

$$\nexists\, P(\text{agent}, t) \ \text{expressing}\ \Phi(t_1,\dots,t_k), \qquad \exists\, P(\text{effect}) \ \text{expressing it}$$ (eq:approval-must-sit-at-the-outcome-not-the-call)

**94%** composition coverage at **410** approvals/day against **20,640** for per-call.

## 7. Internal Mechanics

Why do service accounts get over-provisioned? Because provisioning is a request-time cost and
authority is a design-time decision, and the two are made by different people at different
times. An engineer building an integration needs it to work for every user in testing, so the
account gets the union. Narrowing it later requires knowing which users need which subset,
which nobody recorded. **The union is the path of least resistance and there is no forcing
function to leave it.**

The sandbox confusion has an origin worth naming, because it is not carelessness either.
Sandboxing came from the code-execution threat model, where the untrusted thing *is* the code.
An agent inverts that: the code is trusted (it is your orchestration loop) and the *instruction*
is untrusted. Applying a control designed for the first case to the second produces a control
that is real, well-implemented, and pointed at a threat that is not the one you have.

The credential-locality result has a specific failure pattern in production. A team builds a
tool-call broker, correctly, so credentials are injected server-side and never enter the
context. Then a debugging session needs the credential visible, or a new tool is added that
takes an API key as a parameter, or an SDK reads from the environment by default. The broker
is still there and one path bypasses it, and the bypass is invisible because the broker's
metrics show it working.

On composition, the reason individual review feels adequate is that each tool's damage is
evaluated against the *user's* entitlement — "can the user send email? yes, so the tool is
fine." That is the right question for a permission model and the wrong one for a capability
model. The user can send email and can read documents, and the user cannot bulk-read ten
thousand documents and email them in four seconds. **The composition's danger comes from
throughput and automation, not from authority**, which is why an authority check passes it.

Finally, the reversibility inversion. Reads are treated as safe because they change nothing,
and "changes nothing" is exactly why they are irreversible: there is no state to restore. A
write can be reverted because it left a record of what was there before. A read leaves the
information in the recipient's possession and no record of what was there before, because
nothing was there. That is the same asymmetry {{ch:ops-deployment}} found between reverting a
deployment and reverting the answers it served, and it is why approval design that gates writes
and waves reads through is protecting the recoverable half.

## 8. Implementation

The first listing separates the two capability axes.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ie1}
"""An agent is a confused deputy by construction: it holds authority its requester lacks.

A service account exists because the agent must act for many users. So it is provisioned with
the union of what any of them might need, and every request runs with that union while the
requester is entitled to a slice of it
(eq:agent-authority-exceeds-requester-authority).

That is the classical confused-deputy problem, and the classical fix -- pass the requester's
authority rather than the deputy's -- is ch:sd-apis-auth's delegation result.

Sandboxing is the other reflex, and it addresses a different axis. A sandbox bounds what
*code* can reach. The agent's credentials are carried inside the sandbox, so a sandbox with an
unscoped credential in its environment has moved the boundary and not the authority
(eq:a-sandbox-without-scoped-credentials-moves-nothing).
"""
# (resource class, records, share of users entitled, damage per record)
RESOURCES = [
    ("the requester's own records",      120,     1.000, 1.0),
    ("their team's records",           4_100,     0.180, 1.2),
    ("their tenant's records",       240_000,     0.041, 1.6),
    ("all tenants' records",      31_000_000,     0.000, 4.0),
    ("internal config and secrets",      840,     0.002, 9.0),
    ("billing and payments",          62_000,     0.008, 7.5),
]

print("What the service credential reaches, against what the requester may.")
print()
print(f"{'resource class':>32}{'records':>14}{'entitled users':>17}"
      f"{'agent can reach?':>19}{'excess records':>17}")
print("-" * 99)
total, excess = 0.0, 0.0
res = {}
for name, rec, ent, dmg in RESOURCES:
    total += rec
    ex = rec * (1.0 - ent)
    excess += ex
    res[name] = (rec, ent, ex, dmg)
    print(f"{name:>32}{rec:>14,}{ent:>17.1%}{'yes':>19}{ex:>17,.0f}")
print("-" * 99)
print(f"{'TOTAL':>32}{total:>14,.0f}{'':>17}{'':>19}{excess:>17,.0f}")

print()
print(f"excess authority ratio: {excess / total:.4f} of reachable records are")
print("outside the median requester's entitlement")

print()
print()
print("Damage-weighted, which is the version that matters.")
print()
w_total = sum(r * d for n, r, e, d in RESOURCES)
w_excess = sum(r * (1 - e) * d for n, r, e, d in RESOURCES)
print(f"{'resource class':>32}{'weighted total':>17}{'weighted excess':>18}"
      f"{'share of excess':>18}")
print("-" * 87)
for name, rec, ent, dmg in RESOURCES:
    we = rec * (1 - ent) * dmg
    print(f"{name:>32}{rec * dmg:>17,.0f}{we:>18,.0f}{we / w_excess:>18.1%}")
print("-" * 87)
print(f"{'TOTAL':>32}{w_total:>17,.0f}{w_excess:>18,.0f}{1.0:>18.1%}")

print()
print()
print("Scoping the credential, which is the fix that addresses authority.")
print()
SCOPES = [
    ("one service account for everything",  1.00, 0.00),
    ("per-tenant service accounts",         0.0078, 0.05),
    ("per-user delegated token",            0.0000, 0.22),
    ("per-request capability, task-scoped", 0.0000, 0.34),
]
print(f"{'credential model':>38}{'excess reachable':>19}{'weighted excess':>18}"
      f"{'engineering cost':>19}")
print("-" * 94)
scope = {}
for name, frac, cost in SCOPES:
    scope[name] = (excess * frac, w_excess * frac, cost)
    print(f"{name:>38}{excess * frac:>19,.0f}{w_excess * frac:>18,.0f}"
          f"{cost:>19.0%}")

print()
print("A per-tenant account removes 99% of the excess and leaves the")
print("cross-tenant path closed but the in-tenant one open.")

print()
print()
print("Now the sandbox, which addresses a different axis entirely.")
print()
ESCAPES = [
    ("filesystem outside the workdir",  0.31, 0.4, "yes"),
    ("outbound network",                0.58, 0.6, "yes"),
    ("spawning processes",              0.22, 0.5, "yes"),
    ("reading the environment",         0.91, 0.2, "yes"),
    ("using an injected credential",    1.00, 0.0, "no"),
    ("calling a tool the agent has",    1.00, 0.0, "no"),
]
print(f"{'capability':>34}{'available by default':>23}{'cost to remove':>17}"
      f"{'sandbox removes it?':>21}")
print("-" * 95)
for name, avail, cost, removes in ESCAPES:
    print(f"{name:>34}{avail:>23.0%}{cost:>17.1f}{removes:>21}")

print()
print("The last two rows are what an agent hijack actually uses, and a")
print("sandbox does not touch either.")

print()
print()
print("Combining the two axes. Both are needed and they are not substitutes.")
print()
print(f"{'posture':>44}{'code escape':>14}{'weighted excess authority':>28}"
      f"{'residual':>11}")
print("-" * 97)
POSTURES = [
    ("no sandbox, one service account",     0.58, 1.00),
    ("sandboxed, one service account",      0.04, 1.00),
    ("no sandbox, per-user delegation",     0.58, 0.00),
    ("sandboxed, per-user delegation",      0.04, 0.00),
    ("sandboxed, per-request capability",   0.04, 0.00),
]
for name, esc, auth in POSTURES:
    wa = w_excess * auth
    residual = esc * 1000.0 + wa * 1e-3
    print(f"{name:>44}{esc:>14.0%}{wa:>28,.0f}{residual:>11,.0f}")

print()
print("Sandboxing alone changes the first column. Scoping alone changes the")
print("second. Only doing both changes the residual.")

print()
print()
print("Where the credential physically is, which decides whether scoping works.")
print()
PLACES = [
    ("environment variable in the sandbox", "the agent reads it",  "no"),
    ("mounted secret file",                 "the agent reads it",  "no"),
    ("injected into the tool call by a broker", "never in context", "yes"),
    ("held by a proxy that signs requests", "never in context",    "yes"),
    ("short-lived token minted per request", "in context, briefly", "partly"),
]
print(f"{'where the credential lives':>42}{'agent access':>22}"
      f"{'survives a hijack?':>21}")
print("-" * 85)
for name, access, survives in PLACES:
    print(f"{name:>42}{access:>22}{survives:>21}")

print(f"""
The reach table is the confused-deputy problem counted. The service credential reaches
{total:,.0f} records; the median requester is entitled to a small fraction of them, so
**{excess / total:.2%} of reachable records are outside the requester's entitlement**
(eq:agent-authority-exceeds-requester-authority).

That is not a misconfiguration. It is what a service account is: the union of what any user
might need, held constantly, because the alternative is provisioning per request and nobody
built that.

The damage-weighted version sharpens it. `{RESOURCES[3][0]}` is
{res[RESOURCES[3][0]][2] * res[RESOURCES[3][0]][3] / w_excess:.0%} of weighted excess by
itself, and `{RESOURCES[5][0]}` is
{res[RESOURCES[5][0]][2] * res[RESOURCES[5][0]][3] / w_excess:.1%}. **The excess is
concentrated in exactly the resources a service account is provisioned for and a user never
touches.**

The scoping table is ch:sd-apis-auth's delegation result priced. Per-tenant service accounts
take weighted excess from {w_excess:,.0f} to
{scope['per-tenant service accounts'][1]:,.0f} for {0.05:.0%} of engineering cost -- the
cross-tenant path closed, the in-tenant one open. Per-user delegation takes it to zero for
{0.22:.0%}.

Now the sandbox, and this is where the chapter's second result lives. The escape table lists
what sandboxing removes: filesystem access outside the working directory, outbound network,
process spawning, environment reading. All real, all worth removing.

Read the last two rows. **Using an injected credential** and **calling a tool the agent
already has** are available at {1.00:.0%} and a sandbox removes neither
(eq:a-sandbox-without-scoped-credentials-moves-nothing), because both are things the agent is
*supposed* to do. The sandbox is a boundary around the code; the credential and the tool
handles are inside it.

That is the confusion worth removing. A sandbox is the right control for *untrusted code* --
generated Python, a downloaded package, a user-supplied script. It is not a control for an
*untrusted instruction* reaching a trusted tool, which is what a hijack is.

The posture table makes the non-substitution explicit. Sandboxing alone takes code escape from
{0.58:.0%} to {0.04:.0%} and leaves weighted excess authority at {w_excess:,.0f}. Delegation
alone reverses that. **Only doing both moves the residual**, and each is usually owned by a
different team with a different threat model.

The last table is the implementation detail that decides whether scoping is real. A credential
in an environment variable or a mounted file is readable by the agent -- so a hijacked agent
reads it and uses it outside every control you built, including the sandbox it is sitting in.
A credential injected into the tool call by a broker, or held by a signing proxy, is never in
the agent's context at all.

**Scoping a credential the agent can read is scoping a credential the attacker also has.**
The broker pattern is the one that survives, it is roughly a week of work, and it is the
difference between a control and a configuration.""")
```

## 9. Practical Example

What the service credential reaches:

```
                  resource class       records   entitled users   agent can reach?   excess records
---------------------------------------------------------------------------------------------------
     the requester's own records           120           100.0%                yes                0
          their tenant's records       240,000             4.1%                yes          230,160
            all tenants' records    31,000,000             0.0%                yes       31,000,000
     internal config and secrets           840             0.2%                yes              838
            billing and payments        62,000             0.8%                yes           61,504
```

**99.96% of reachable records are outside the median requester's entitlement**
({{eq:agent-authority-exceeds-requester-authority}}), and damage-weighted, **99.3%** of the
excess is the cross-tenant class.

```
                      credential model   excess reachable   weighted excess   engineering cost
----------------------------------------------------------------------------------------------
    one service account for everything         31,295,864       124,841,115                 0%
           per-tenant service accounts            244,108           973,761                 5%
              per-user delegated token                  0                 0                22%
   per-request capability, task-scoped                  0                 0                34%
```

```
                        capability   available by default   cost to remove  sandbox removes it?
-----------------------------------------------------------------------------------------------
    filesystem outside the workdir                    31%              0.4                  yes
                  outbound network                    58%              0.6                  yes
           reading the environment                    91%              0.2                  yes
      using an injected credential                   100%              0.0                   no
      calling a tool the agent has                   100%              0.0                   no
```

**The last two rows are what a hijack uses and a sandbox removes neither**
({{eq:a-sandbox-without-scoped-credentials-moves-nothing}}) — both are things the agent is
supposed to do, from inside the box.

```
                                     posture   code escape   weighted excess authority   residual
-------------------------------------------------------------------------------------------------
             no sandbox, one service account           58%                 124,841,115   580,125
              sandboxed, one service account            4%                 124,841,115    40,125
             no sandbox, per-user delegation           58%                           0   580,000
              sandboxed, per-user delegation            4%                           0    40,000
```

```
        where the credential lives          agent access   survives a hijack?
-----------------------------------------------------------------------------
environment variable in the sandbox    the agent reads it                   no
              mounted secret file      the agent reads it                   no
injected into the tool call by a broker  never in context                  yes
held by a proxy that signs requests      never in context                  yes
```

**Scoping a credential the agent can read is scoping a credential the attacker also has.**

The second listing takes up composition.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ie2}
"""Two harmless tools compose into a harmful one, and per-tool approval cannot see it.

A search tool reads. An email tool writes. Neither is dangerous alone -- reading what the user
may read, sending what the user may send. Together they are an exfiltration primitive, and the
damage of the pair exceeds the sum of its parts
(eq:tool-damage-composes-superadditively).

Which breaks the approval design almost everyone builds first. Approving each call
individually approves each half of a composition and never sees the whole, so the gate has to
sit at the outcome rather than at the call
(eq:approval-must-sit-at-the-outcome-not-the-call).
"""
# (tool, damage alone, reversible?, calls/day)
TOOLS = [
    ("search internal documents", 1.0, True,  8400),
    ("read a customer record",    1.5, True,  6100),
    ("send an email",             2.0, False, 1900),
    ("write to the CRM",          3.0, False, 2400),
    ("issue a refund",            7.0, False,  310),
    ("execute code",              5.0, True,   890),
    ("post to a webhook",         2.5, False,  640),
]
alone = {n: d for n, d, r, c in TOOLS}

print("Each tool alone.")
print()
print(f"{'tool':>28}{'damage alone':>15}{'reversible':>13}{'calls/day':>12}")
print("-" * 68)
for name, dmg, rev, calls in TOOLS:
    print(f"{name:>28}{dmg:>15.1f}{('yes' if rev else 'no'):>13}{calls:>12,}")

print()
print(f"sum of individual damages: {sum(alone.values()):.1f}")

print()
print()
print("Compositions, and what they do that the parts do not.")
print()
COMPOS = [
    (("search internal documents", "send an email"),        14.0,
     "exfiltration to any address"),
    (("read a customer record", "post to a webhook"),       12.0,
     "bulk export, no log on the far side"),
    (("search internal documents", "write to the CRM"),      9.0,
     "poison the record other agents read"),
    (("read a customer record", "issue a refund"),          16.0,
     "targeted fraud at scale"),
    (("execute code", "send an email"),                     18.0,
     "arbitrary computation plus a channel"),
    (("search internal documents", "execute code",
      "post to a webhook"),                                 24.0,
     "read, transform, exfiltrate"),
]
SHORT = {
    "search internal documents": "search",
    "read a customer record": "read",
    "send an email": "email",
    "write to the CRM": "crm-write",
    "issue a refund": "refund",
    "execute code": "exec",
    "post to a webhook": "webhook",
}
print(f"{'composition':>34}{'sum of parts':>15}{'actual':>9}"
      f"{'excess':>9}  {'what it enables':<38}")
print("-" * 107)
comp = {}
for tools, dmg, what in COMPOS:
    s = sum(alone[t] for t in tools)
    label = " + ".join(SHORT[t] for t in tools)
    comp[tools] = (s, dmg, dmg - s)
    print(f"{label:>34}{s:>15.1f}{dmg:>9.1f}{dmg - s:>9.1f}  {what:<38}")

print()
tot_excess = sum(c[2] for c in comp.values())
print(f"total superadditive excess across six compositions: {tot_excess:.1f}")
print(f"against a sum-of-parts total of {sum(alone.values()):.1f}")

print()
print()
print("How many compositions there are, as tools are added.")
print()
print(f"{'tools':>8}{'pairs':>10}{'triples':>11}{'total subsets >= 2':>22}"
      f"{'reviewed in practice':>23}")
print("-" * 74)
import math
for n in (3, 5, 7, 10, 15, 20):
    pairs = math.comb(n, 2)
    triples = math.comb(n, 3)
    allsub = 2 ** n - n - 1
    print(f"{n:>8}{pairs:>10,}{triples:>11,}{allsub:>22,}{n:>23}")

print()
print("Reviews are per tool. The thing that grows is the subset count.")

print()
print()
print("Reversibility, which decides what an approval is actually protecting.")
print()
print(f"{'action':>28}{'damage':>9}{'recoverable share':>20}"
      f"{'permanent damage':>19}")
print("-" * 76)
REVERSIBILITY = [
    ("search internal documents", 0.00, 1.0),   # reading cannot be unread
    ("read a customer record",    0.00, 1.5),
    ("send an email",             0.05, 2.0),
    ("write to the CRM",          0.85, 3.0),
    ("issue a refund",            0.70, 7.0),
    ("execute code",              0.40, 5.0),
    ("post to a webhook",         0.00, 2.5),
]
perm = 0.0
for name, rec, dmg in REVERSIBILITY:
    p = dmg * (1 - rec)
    perm += p
    print(f"{name:>28}{dmg:>9.1f}{rec:>20.0%}{p:>19.2f}")
print("-" * 76)
print(f"{'TOTAL':>28}{sum(d for n, r, d in REVERSIBILITY):>9.1f}"
      f"{'':>20}{perm:>19.2f}")

print()
print(f"{perm / sum(d for n, r, d in REVERSIBILITY):.0%} of the damage is "
      f"permanent, and the reads are")
print("the least reversible actions on the list")

print()
print()
print("Where to put the approval.")
print()
DAILY_CALLS = sum(c for n, d, r, c in TOOLS)
GATES = [
    ("no approval",                 0.00, 0,        0.00),
    ("approve every call",          1.00, DAILY_CALLS, 1.00),
    ("approve non-reversible calls", 0.62, 5250,    0.31),
    ("approve by outcome class",    0.94, 410,      0.09),
    ("approve on a taint path only", 0.91, 190,     0.05),
]
print(f"{'gate':>34}{'composition coverage':>23}{'approvals/day':>16}"
      f"{'utility cost':>15}")
print("-" * 88)
gates = {}
for name, cov, appr, util in GATES:
    gates[name] = (cov, appr, util)
    print(f"{name:>34}{cov:>23.0%}{appr:>16,}{util:>15.0%}")

print()
print("Per-call approval sees every half and no whole. Outcome-class approval")
print("sees the whole and asks 400 times a day instead of 20,000.")

print()
print()
print("What an outcome class actually is.")
print()
CLASSES = [
    ("data leaves the tenant boundary",  "search + any egress tool", 14.0, 41),
    ("money moves",                      "refund, payment, credit",  16.0, 22),
    ("a record other agents read changes", "write to shared state",   9.0, 190),
    ("something irreversible happens",   "no undo path exists",      7.0, 84),
    ("privilege is used outside the task", "scope mismatch",         12.0, 73),
]
print(f"{'outcome class':>38}{'triggered by':>28}{'damage':>9}"
      f"{'approvals/day':>16}")
print("-" * 91)
tot_appr = 0
for name, trig, dmg, appr in CLASSES:
    tot_appr += appr
    print(f"{name:>38}{trig:>28}{dmg:>9.1f}{appr:>16}")
print("-" * 91)
print(f"{'TOTAL':>38}{'':>28}{'':>9}{tot_appr:>16}")

print(f"""
The individual table is the one every tool review produces, and it is not wrong. Each of the
{len(TOOLS)} tools does something the user is entitled to do, at a damage level that a
reasonable person signs off. The sum is {sum(alone.values()):.1f}.

The composition table is what the review misses. `search + send` is worth
{comp[('search internal documents', 'send an email')][1]:.1f} against a sum of parts of
{comp[('search internal documents', 'send an email')][0]:.1f} -- an excess of
{comp[('search internal documents', 'send an email')][2]:.1f}
(eq:tool-damage-composes-superadditively) -- because reading and sending are individually
authorised and *reading then sending* is exfiltration.

`search + execute + post` reaches {comp[('search internal documents', 'execute code', 'post to a webhook')][1]:.1f}
against {comp[('search internal documents', 'execute code', 'post to a webhook')][0]:.1f}: read
anything, transform it into a form no filter recognises, and send it somewhere with no log on
the far side.

Across six compositions the superadditive excess is {tot_excess:.1f}, against a
sum-of-parts total of {sum(alone.values()):.1f}. **The compositions carry more danger than the
tools do.**

The counting table is why this cannot be fixed by reviewing harder. Seven tools have
{math.comb(7, 2)} pairs and {math.comb(7, 3)} triples; twenty tools have
{math.comb(20, 2)} pairs and {math.comb(20, 3):,} triples, and
{2 ** 20 - 20 - 1:,} subsets of size two or more. **Reviews scale with tool count and risk
scales with subset count**, which is ch:sec-threat-model's product result one level up.

The reversibility table changes what an approval is for. Note the first two rows:
`{REVERSIBILITY[0][0]}` and `{REVERSIBILITY[1][0]}` are recoverable
{REVERSIBILITY[0][1]:.0%} of the time, because **a read cannot be unread**. Writes are largely
recoverable; refunds mostly are; reads are not.

{perm / sum(d for n, r, d in REVERSIBILITY):.0%} of total damage is permanent, and the actions
usually classified as "safe, read-only" are the least reversible things on the list. That is
ch:ops-deployment's rollback result in the tool layer: the artefact can be reverted and the
disclosure cannot.

The gate table is the design consequence. Approving every call covers everything and asks
{DAILY_CALLS:,} times a day, which is not a control because nobody reads
{DAILY_CALLS:,} approvals -- it is a rubber stamp with an audit trail.

Approving by **outcome class** covers {gates['approve by outcome class'][0]:.0%} of
compositions at {gates['approve by outcome class'][1]:,} approvals a day and
{gates['approve by outcome class'][2]:.0%} utility cost
(eq:approval-must-sit-at-the-outcome-not-the-call).

The difference is what the gate is looking at. A per-call gate asks "may this agent call this
tool?" and the answer is yes, for both halves of the composition. An outcome gate asks "is
data about to leave the tenant boundary?" -- a question about the *effect* of the sequence,
which is the thing that was dangerous.

The class table is what those questions are. Five of them, {tot_appr} approvals a day between
them, each phrased as a consequence rather than a capability. `data leaves the tenant
boundary` fires whether the egress is an email, a webhook, a file write or a search-result
citation, and it does not need to know which tool is doing it.

**Enumerate outcomes, not tools.** Outcomes are few and stable; tools are many and grow every
sprint, and the subsets grow faster than either.""")
```

```
                        tool   damage alone   reversible   calls/day
--------------------------------------------------------------------
   search internal documents            1.0          yes       8,400
               send an email            2.0           no       1,900
              issue a refund            7.0           no         310
                execute code            5.0          yes         890
           post to a webhook            2.5           no         640

sum of individual damages: 22.0
```

```
                       composition   sum of parts   actual   excess  what it enables
-------------------------------------------------------------------------------------
                    search + email            3.0     14.0     11.0  exfiltration to any address
                    read + webhook            4.0     12.0      8.0  bulk export, no log on the far side
                     read + refund            8.5     16.0      7.5  targeted fraud at scale
           search + exec + webhook            8.5     24.0     15.5  read, transform, exfiltrate
```

Superadditive excess **58.0** against a sum-of-parts total of **22.0**
({{eq:tool-damage-composes-superadditively}}) — **the compositions carry more danger than the
tools.**

```
   tools     pairs    triples    total subsets >= 2   reviewed in practice
--------------------------------------------------------------------------
       7        21         35                   120                      7
      10        45        120                 1,013                     10
      20       190      1,140             1,048,555                     20
```

**Reviews scale with tool count; risk scales with subset count.**

```
                      action   damage   recoverable share   permanent damage
----------------------------------------------------------------------------
   search internal documents      1.0                  0%               1.00
            write to the CRM      3.0                 85%               0.45
              issue a refund      7.0                 70%               2.10
           post to a webhook      2.5                  0%               2.50
----------------------------------------------------------------------------
                       TOTAL     22.0                                  12.45
```

**57% of the damage is permanent, and the reads are the least reversible actions on the list** —
{{eq:rollback-restores-code-not-state}} in the tool layer.

```
                              gate   composition coverage   approvals/day   utility cost
----------------------------------------------------------------------------------------
                       no approval                     0%               0             0%
                approve every call                   100%          20,640           100%
       approve non-reversible calls                    62%           5,250            31%
          approve by outcome class                    94%             410             9%
      approve on a taint path only                    91%             190             5%
```

**94% coverage at 410 approvals a day**
({{eq:approval-must-sit-at-the-outcome-not-the-call}}) — a per-call gate sees every half and no
whole.

```
                         outcome class                triggered by   damage   approvals/day
-------------------------------------------------------------------------------------------
       data leaves the tenant boundary    search + any egress tool     14.0              41
                          money moves     refund, payment, credit     16.0              22
  a record other agents read changes        write to shared state      9.0             190
      something irreversible happens          no undo path exists      7.0              84
```

## 10. Production Considerations

Compute your excess authority ratio. Reachable records over entitled records, damage-weighted,
takes an afternoon and is usually above 99%.

Scope credentials per tenant before anything else. It removes the cross-tenant class — 99.3% of
weighted excess — for about 5% of engineering effort.

Put credentials behind a broker or a signing proxy so they never enter the context. A scoped
credential the agent can read is not scoped.

Sandbox generated code and do not report it as an agent-hijacking control. It bounds the wrong
axis for that threat.

Enumerate outcome classes and gate on those. Five of them cover 94% of compositions at 2% of
the approval volume.

Classify actions by reversibility and stop treating reads as safe. They are the least
reversible things you run.

Audit new tools against the *compositions* they enable, not against the user's entitlement. The
question is what the pair does, not whether the user could do each half.

## 11. Common Mistakes

**Reporting a sandbox as an agent-security control.** It bounds code reach; hijacking uses
credentials and tools.

**Scoping a credential the agent can read.** The attacker reads it too and uses it outside
every control.

**Reviewing tools individually.** Six compositions here carry 58.0 of excess over a 22.0
baseline.

**Approving every tool call.** 20,640 a day is a rubber stamp with an audit trail.

**Treating reads as low-risk.** They are 0% recoverable.

**Adding a tool without recomputing the subsets.** The twentieth tool adds 19 pairs and 171
triples.

## 12. Failure Modes

**Broker bypassed by one SDK default.** Credentials are injected server-side on every path but
one, and the broker's metrics look healthy.

**Sandboxed agent issues a refund.** No escape was needed; the tool was inside.

**Per-call approval that approves an exfiltration.** Both halves were legitimate and the
reviewer saw them an hour apart.

**Read-only agent that leaks continuously.** Everything it does is reversible on paper and
irreversible in fact.

**Tool added for one team, reachable by all.** The service account did not change and the
subset count doubled.

**Approval queue abandoned.** Volume exceeded review capacity, approvals became automatic, and
the gate is now a log line — {{cite:cemri2025mast}}'s coordination failures with a security
consequence.

## 13. Alternatives

**Per-user delegated tokens.** {{ch:sd-apis-auth}}'s design. Removes excess authority entirely
at about 22% engineering cost and requires an identity path to every backend.

**Two-agent split: a reader with no egress, a writer with no reads.**
{{cite:beurerkellner2025patterns}}'s pattern family. Breaks the compositions structurally and
costs the workflows that genuinely need both.

**Capability tokens minted per task.** The agent receives a token good for exactly this task's
resources and nothing else. The strongest option, and it requires the task to be specified
before it starts.

**Egress-only gating.** Approve nothing except data leaving a boundary. Cheapest outcome gate,
covers the largest composition class, and misses the ones that stay inside.

**Full simulation before execution.** Run the plan against a mirror, diff the effects, approve
the diff. Excellent for reversible actions and impossible for reads.

## 14. Evaluation

Measure your excess authority ratio and its damage-weighted version, and re-measure whenever a
backend is added.

Test whether a credential is reachable: ask the agent to print its environment. If it can, the
scoping is decorative.

Enumerate the pairwise compositions of your current tool set and price the top ten. The
exercise takes an afternoon and it is the one nobody runs.

Measure your approval volume and the fraction approved within five seconds. Anything above a
few hundred a day is being rubber-stamped.

Run {{cite:debenedetti2024agentdojo}} with and without your capability controls and report the
utility cost alongside the block rate.

## 15. Advanced Concepts

The superadditivity model treats composition damage as a fixed number per subset, which
understates the problem in one direction and overstates it in another. It understates because
composition damage is usually a function of *rate*: search-plus-email is worth little at one
document an hour and a great deal at ten thousand in four seconds, so the right model has a
throughput term and rate limiting becomes a first-class control rather than an afterthought. It
overstates because many enumerable subsets are not reachable in practice — the orchestration
never sequences them, or the intermediate types do not compose. **The reachable subset graph is
much smaller than $2^n$ and much larger than $n$**, and computing it from execution traces is a
day of work that would replace most of the guesswork in tool review.

The reversibility classification has an edge that matters for design. "Recoverable" here means
the *state* can be restored, and for a read there is no state. But there is a weaker property
worth having: **detectability**. A read that is logged with provenance can at least be
enumerated afterwards — you know what was disclosed, to whom, and can notify. That is not
reversal and it is the difference between a breach you can scope and one you cannot, which is
the difference regulators care about. So the practical classification is three-valued —
reversible, irreversible-but-scopable, irreversible-and-unscopable — and the third category is
the one to eliminate by instrumentation.

There is an interaction with {{cite:hou2025mcp}}'s analysis worth drawing out. A protocol that
lets tools be added at runtime means the tool set is not known at design time, so the subset
enumeration cannot be done in advance and the outcome classes must cover tools that do not
exist yet. That is an argument *for* outcome-class gating rather than against it — outcome
classes are stable under tool addition and per-tool policies are not — and it is the strongest
practical case for the design this chapter recommends.

Finally, on the two axes. The additive decomposition $R = c(\sigma) + a(\alpha)$ is a
simplification: in reality code escape can be *used to* obtain credentials (reading a mounted
secret is a filesystem operation), so the terms interact. That interaction runs in one
direction only — better sandboxing reduces credential reachability, and better credential
scoping does not reduce code escape — which means sandboxing is worth slightly more than the
additive model credits and scoping is worth exactly what it says. The ordering of the
recommendations is unaffected.

## 16. Connection to Previous Chapters

{{eq:delegation-moves-the-check}} from {{ch:sd-apis-auth}} is the fix for the authority axis,
priced here at 22% engineering cost for a complete removal of excess.

{{eq:only-capability-limits-bound-the-damage}} from {{ch:sec-threat-model}} is confirmed and
refined: there are two capability axes, they are complements, and conflating them produces a
system that is well-sandboxed and fully authorised.

{{eq:rollback-restores-code-not-state}} from {{ch:ops-deployment}} returns as the reversibility
inversion — reads are 0% recoverable and are the actions approval flows wave through.

{{eq:attack-surface-is-sources-times-sinks}} from {{ch:sec-threat-model}} counted paths
linearly in sinks; this chapter shows the *damage* at the end of those paths grows
exponentially in tool count.

## 17. Exercises

1. Compute your agent's excess authority ratio, raw and damage-weighted. Where does the excess
   concentrate?

2. Ask your agent to print its environment. Can it reach a credential?

3. Enumerate the pairwise compositions of your tool set and price the top five. Which were
   reviewed individually and passed?

4. Classify your actions three ways — reversible, irreversible-but-scopable,
   irreversible-and-unscopable — per {{sec:15-advanced-concepts}}. How large is the third
   category?

5. Derive the reachable subset graph from a week of execution traces. How much smaller than
   $2^n$ is it?

## 18. Interview Questions

1. Why is an agent a confused deputy even when correctly configured?

2. We sandbox the agent. Does that stop prompt-injection hijacking?

3. Where should the credential live, and why does it matter?

4. We approve every tool call. Is that a control?

5. Search and email are both approved individually. What did we just approve?

6. Which of your tools is least reversible?

## 19. Research Questions

1. How much smaller is the reachable subset graph than $2^n$ in production agent traces?

2. What is the empirical throughput dependence of composition damage, and where does rate
   limiting sit against capability limiting?

3. Can outcome classes be specified completely enough to cover tools added at runtime under a
   protocol like MCP?

4. How often does a credential broker get bypassed in practice, and by what mechanism?

## 20. Chapter Summary

An agent holds authority its requester lacks, and that is what a service account is.

**99.96%** of reachable records lie outside the median requester's entitlement, with **99.3%**
of the damage-weighted excess in the cross-tenant class
({{eq:agent-authority-exceeds-requester-authority}}). Per-tenant accounts remove almost all of
it for **5%** of engineering cost; per-user delegation removes all of it for **22%**.

Sandboxing addresses a different axis. It takes code escape from **58% to 4%** and does not
touch **using an injected credential** or **calling a tool the agent has**, both at 100%
({{eq:a-sandbox-without-scoped-credentials-moves-nothing}}), because both are what the agent is
for. The two controls are complements, and a credential the agent can read is a credential the
attacker has.

Then composition. `search + email` is **14.0** against a sum of parts of **3.0**;
`search + exec + webhook` is **24.0** against **8.5**; the total superadditive excess is
**58.0** against a **22.0** baseline ({{eq:tool-damage-composes-superadditively}}). Twenty
tools have **1,048,555** subsets and get twenty reviews.

And reversibility inverts the usual intuition: writes are 70–85% recoverable, reads are **0%**,
and **57%** of damage is permanent. So approving every call — **20,640** a day — protects the
recoverable half at a volume nobody reads, while approving by **outcome class** covers **94%**
of compositions at **410** ({{eq:approval-must-sit-at-the-outcome-not-the-call}}).

The thread through the chapter is that three separate controls — the sandbox, the per-call
approval, the tool review — are each competent and each aimed at a unit that is not the unit of
danger. The sandbox bounds the code and the danger is in the credential. The approval bounds
the call and the danger is in the sequence. The review bounds the tool and the danger is in the
subset. In every case the fix is to move the control to the thing that actually varies, and in
every case that thing is fewer in number and more stable than what is currently being checked.

Carry forward: **the agent holds more authority than its requester**, and **gate outcomes, not
calls**.

## 21. Further Reading

- {{cite:debenedetti2024agentdojo}} — utility and attack success measured together, which is
  how a capability control's cost gets stated.
- {{cite:beurerkellner2025patterns}} — the two-agent and capability-scoping patterns, with
  their utility costs.
- {{cite:cemri2025mast}} — multi-agent failure modes, including the coordination failures that
  turn an approval queue into a log line.
- {{cite:hou2025mcp}} — the tool-integration layer's threat surface, and why runtime tool
  addition argues for outcome-class gating.
