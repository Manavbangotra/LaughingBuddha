---
id: sd-apis-auth
number: 194
part: XXII
tier: full
status: draft
requires: [three-properties-break-the-stack, access-shape-decides-the-store,
           variance-not-mean-drives-wait, semantic-failure-has-no-instrument]
provides: [count-limits-cannot-bound-cost, cost-limits-need-a-reservation,
           delegation-moves-the-check, fewer-tools-beats-better-credentials]
citations: [mcp2026spec, hou2025mcp, qin2023toolllm, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to show why a request-count rate limit
cannot simultaneously bound cost and treat users fairly when request costs are
heterogeneous, and compute both horns of that dilemma; design a cost-based limiter
and handle the fact that a request's cost is unknown when it arrives; identify the
hop in a call chain at which a user's identity is replaced by a service identity, and
quantify the authority that hop confers; rank an agent's tools by exposure rather
than by risk in isolation; and choose between reducing an agent's tool set and
rebuilding its delegation model, with a number attached to each.

## 2. Why This Matters

Rate limiting and authorization are the two places where an AI system's API surface
meets the properties {{ch:sd-architecture}} established, and both break in ways their
conventional designs cannot express.

Rate limiting assumes a request is a unit of work. When one request costs **106×**
another — measured within a single service in {{sec:9-practical-example}} — a
request-count limiter has one number to set and that spread must fit inside it.
Setting it to bound worst-case spend throttles **52%** of all traffic, including
light users consuming **1.5%** of the ceiling the limit exists to protect. Setting it
fairly lets one tenant reach **4.89×** the cost ceiling with no violation recorded,
because no violation occurred ({{eq:count-limits-cannot-bound-cost}}).

Authorization breaks differently. When a user calls an API that runs an agent that
calls tools, something decides what is allowed — and the convenient implementation
gives the agent a service account holding the union of every user's rights. Under
that design **50.8%** of the blast radius the agent can reach is unreachable by the
user who invoked it ({{eq:delegation-moves-the-check}}), and the only thing
preventing access is the model choosing not to.

Both results have the same practical shape: the cheap fix is available before the
expensive one, and it is not the fix teams reach for first.

## 3. Prerequisites

You need {{ch:sd-architecture}}'s three properties
({{eq:three-properties-break-the-stack}}) — particularly *expensive*, since cost
heterogeneity is what breaks the limiter.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} supplies the same underlying
distribution from the queueing side; a limiter and a queue are two responses to one
phenomenon.

{{ch:ag-security}} and {{ch:mcp-security}} establish that a model's decision not to
use a capability is not a security control. This chapter quantifies what that means
when the capability is granted by an API gateway.

{{eq:access-shape-decides-the-store}} from {{ch:sd-storage}} is a useful contrast: the
same "measure the ratio, not the absolute" move appears in both chapters.

## 4. Intuitive Explanation

Every rate limiter answers two questions at once, and normally they have the same
answer. *Is this user taking more than their share?* and *is this user about to cost
us more than we can afford?* When every request costs the same, counting requests
answers both.

Now let one request cost a hundred times another. The two questions come apart, and
the limiter still only has one number.

Suppose you set that number to protect the budget. You have to assume the worst — that
every request will be an expensive one — so the limit becomes very small. Now a user
sending cheap requests, who could send hundreds before costing anything, gets stopped
after a handful. They are throttled by an arithmetic that has nothing to do with what
they are doing.

Suppose instead you set it to what a typical user needs. The typical user is happy.
And the one user whose requests are enormous now has permission to send a full
allowance of enormous requests, which is several times what the plan was priced at.
Your limiter is working perfectly and your bill is wrong.

There is no third number. The quantity being limited is not the quantity being
protected, and no setting reconciles them. What reconciles them is limiting the right
quantity — spend a budget of cost per minute rather than a count of requests per
minute, and charge each request what it actually costs.

That fix has a wrinkle worth understanding, because it is the reason people avoid it.
You do not know what a request costs until it finishes. Generation cost depends on
output length, which {{ch:sd-async}} established is unknown at admission. So a cost
limiter has to guess on the way in and settle up on the way out: reserve a pessimistic
amount, then refund the difference.

The authorization half has a similar structure — a decision made at the wrong place.

Think about what happens when a user asks an agent to do something. The user
authenticates to your API. Fine. Then the agent needs to search documents, and read
one, and maybe update a record. Which identity makes those calls?

The easy answer is: the agent's own. Give the agent a service account with enough
permission to do its job for anybody, and let it serve everybody. This works
immediately, and it means every single request runs with the combined permissions of
your entire user base.

The user who could not have read that document directly is now sending requests to
something that can. Nothing stops the agent from doing it except the agent deciding
not to — and an agent's decisions are exactly what an attacker manipulates.

The principled fix is to carry the user's identity all the way down, so the agent can
do precisely what the user could do and nothing more. It works, and it is expensive:
token exchange, per-tool credentials, and a set of hard questions about background
work that happens when the user is not there.

But before that, there is a much cheaper move. The over-permission is not spread
evenly across the agent's tools — it concentrates almost entirely in a few, and those
few are usually rare. Take the two worst tools away from the agent and hand them back
to a human, and you close nearly half the gap by editing a list.

## 5. Formal Explanation

Let users be indexed by $u$ with request rate $\nu_u$ and per-request cost drawn from
a distribution with mean $\mu_u$ and standard deviation $s_u$. A request-count limiter
admits $R$ requests per unit time per tenant.

The worst-case spend under such a limiter is $R\max_u(\mu_u + z s_u)$ for a chosen
tail multiplier $z$. Bounding it below a ceiling $B$ requires

$$ R \;\le\; \frac{B}{\max_u(\mu_u + z s_u)} $$ (eq:count-limits-cannot-bound-cost)

which is set by the most expensive request anyone can send. The throttling this
imposes on user $u$ is $\max(0, 1 - R/\nu_u)$, and since $R$ was computed from a
quantity unrelated to $u$'s behaviour, users with small $\mu_u$ are throttled by an
amount that does not depend on their cost at all.

Conversely, choosing $R$ to satisfy a typical user — $R \ge \nu_u$ for most $u$ —
gives worst-case spend $R\max_u(\mu_u + z s_u)$, which exceeds $B$ by the ratio of the
heterogeneity to the typical cost. **The two requirements are satisfied by disjoint
ranges of $R$ whenever heterogeneity exceeds the ratio of the ceiling to typical
demand**, and that is the general statement of the dilemma.

A cost-based limiter admits requests while cumulative charged cost in a window stays
below $B$. Worst-case spend is $B$ by construction, and throttling on user $u$ is
$\max(0, 1 - B/(\nu_u\mu_u))$ — a function of what $u$ actually costs.

The complication is that cost is unknown at admission. Writing $\hat{c}$ for a
reservation and $c$ for realised cost, a limiter must hold $\hat{c}$ against the
budget on entry and release $\hat{c} - c$ on exit. Over-reserving throttles
unnecessarily; under-reserving admits overspend. With $\hat{c}$ set at the $q$-th
percentile of the cost distribution, expected over-reservation is

$$ \mathbb{E}[\hat{c} - c] \;=\; F^{-1}(q) - \mathbb{E}[c] $$ (eq:cost-limits-need-a-reservation)

so the reservation policy is itself a percentile choice with the same shape as every
other tail decision in this part.

For authorization, let the agent have tools $t$ with required scope $\sigma(t)$, usage
frequency $f_t$, and blast radius $\beta_t$. Let $h_{\sigma}$ be the share of users
holding scope $\sigma$. Under a service account the agent holds
$\bigcup_t \sigma(t)$ on every request, so the expected blast radius reachable by the
agent but not by the invoking user is

$$ E \;=\; \sum_t f_t\,\beta_t\,(1 - h_{\sigma(t)}), \qquad \text{over-permission} = \frac{E}{\sum_t f_t \beta_t} $$ (eq:delegation-moves-the-check)

Under token passthrough the agent holds exactly the user's scopes, so $E = 0$.

## 6. Mathematical Foundation

{{eq:delegation-moves-the-check}} has a structure that decides what to do about it,
and it is not the structure the security conversation usually assumes.

The per-tool exposure $e_t = f_t\beta_t(1 - h_{\sigma(t)})$ is a product of three
terms that are **negatively correlated in practice**. Tools with large blast radius
are used rarely and held by few users; tools used constantly have small blast radius
and near-universal entitlement. So $e_t$ is dominated by a small number of tools in
the middle — rare enough that few users hold the scope, common enough to appear in
real traffic, and destructive enough to matter.

That concentration is what makes tool removal effective. Removing the top-$k$ tools by
$e_t$ reduces over-permission by

$$ \frac{\sum_{t \in \text{top-}k} e_t}{\sum_t e_t} \quad \text{while affecting} \quad \sum_{t \in \text{top-}k} f_t \;\text{of traffic} $$ (eq:fewer-tools-beats-better-credentials)

and because $e_t$ and $f_t$ are negatively correlated, the numerator is large while
the traffic cost is small. {{sec:9-practical-example}} measures **65%** of exposure in
two tools appearing in **7%** of requests.

Row-level filtering, the usual partial mitigation, has the opposite profile. It
applies where authorization is expressible as a predicate over rows — reads with an
owner column — and not to writes, directory lookups, or anything whose permission
model is not row-shaped. Writing $\phi_t$ for the filterable share,

$$ E_{\text{filtered}} \;=\; \sum_t f_t\,\beta_t\,(1 - h_{\sigma(t)})(1 - \phi_t) $$

and since $\phi_t \approx 0$ exactly for the high-$\beta_t$ write tools, filtering
removes the small terms and leaves the large ones. **The tools it cannot protect are
the ones that mattered**, which is why it moves over-permission from **50.8%** only to
**43.6%**.

## 7. Internal Mechanics

**Why the reservation must be per-request, not per-window.** A window-level budget
check admits a request when the window has room, which is the moment before an
expensive generation begins. By the time the cost is known the money is spent. The
reservation in {{eq:cost-limits-need-a-reservation}} has to be taken at admission and
held for the request's lifetime, which makes the limiter stateful in a way a
request-counter is not — and that state has to survive a process restart, or a deploy
becomes a budget reset.

**Bursts and the window length.** A cost budget expressed per minute and one
expressed per hour are not the same control even when they permit identical average
spend. The shorter window bounds instantaneous damage and rejects legitimate bursts;
the longer one absorbs bursts and permits an hour of runaway before it reacts. Since
the agent workload in the listing is bursty by nature -- a document arrives, and
thirty expensive calls follow within a minute -- the honest design is two budgets at
different timescales, with the short one sized to protect capacity and the long one
sized to protect the bill. They fail differently and they are both needed, which is
an argument teams routinely lose to a preference for a single number.

**The cost model is a moving part.** A cost limiter needs a price per token, and that
price changes when the model changes. Every tenant's effective rate limit then changes
with no configuration edit and no deploy anyone would connect to the resulting
complaints. Pinning the limiter's cost model separately from the serving model, and
versioning it, is the difference between a limiter and a mystery.

**Where identity is actually lost.** {{cite:mcp2026spec}}'s stateless request model
makes each call self-contained, which means the identity travelling with a call is
whatever the caller put in it — there is no connection-scoped session carrying the
original principal implicitly. {{cite:hou2025mcp}} and the surrounding work document
this as the central authorization question for tool protocols. The practical
consequence is that **identity loss happens at exactly one hop and it is a coding
decision**, not an emergent property of the architecture.

**Background work is the hard case for passthrough.** An agent that continues working
after the user's session ends has no user token to present. The honest designs are
either to refuse background work, or to mint a narrowly-scoped delegated credential at
session time with an explicit expiry and an explicit tool allowlist — which is the
per-tool exchanged token row in {{sec:9-practical-example}}, and the reason it costs
**6.8×**.

**Tool count is an authorization variable.** {{cite:qin2023toolllm}}'s work on large
tool collections treats tool count as a capability question. Under
{{eq:delegation-moves-the-check}} it is also a permission question: each added tool
widens the service account's union, and the widening is permanent while the tool's
usefulness may not be.

**Correlated misuse.** {{cite:cemri2025mast}}'s failure taxonomy indicates that agent
failures cluster; an agent misled into one inappropriate tool call is likely to make
several. Over-permission should therefore be read as a *joint* exposure rather than a
per-call one, which makes the concentration in {{eq:fewer-tools-beats-better-credentials}}
more actionable rather than less.

## 8. Implementation

The first listing measures both horns of the rate-limiting dilemma and prices the
cost-based alternative.

```python {tier=A name=bz1}
"""A request-count rate limit cannot bound cost when requests are not equivalent.

Rate limiting assumes a request is a unit of work. ch:sd-architecture found that
assumption broken -- one request can cost forty times another -- and a limiter built
on the broken assumption fails in a specific way.

To bound worst-case spend with a request limit you must set it for the most expensive
request, which starves everyone sending cheap ones. To be fair to typical users you
must set it for the average, which admits a runaway
(eq:count-limits-cannot-bound-cost).

This listing measures both horns and prices the alternative.
"""
# A user population. (label, share of users, requests/min they want,
#                     mean cost per request, cost spread)
USERS = [
    ("light interactive",   0.62,  4.0,   1.0,  0.4),
    ("heavy interactive",   0.24, 11.0,   2.8,  1.6),
    ("document workload",   0.11,  6.0,  14.0, 11.0),
    ("agent workload",      0.03, 25.0,  38.0, 34.0),
]
COST_CEILING = 260.0    # cost units per minute the system can afford per tenant


def cost_wanted(u):
    return u[2] * u[3]


def peak_cost(u):
    """What this user costs when their requests land at the expensive end."""
    return u[2] * (u[3] + 2.0 * u[4])


print("Four user profiles. The cost of a request varies within a profile as well")
print("as between them.")
print()
print(f"{'profile':>20}{'share':>8}{'req/min':>10}{'cost/req':>11}"
      f"{'spread':>9}{'cost/min':>11}{'peak':>9}")
print("-" * 78)
for u in USERS:
    print(f"{u[0]:>20}{u[1]:>8.0%}{u[2]:>10.1f}{u[3]:>11.1f}{u[4]:>9.1f}"
          f"{cost_wanted(u):>11.1f}{peak_cost(u):>9.1f}")

worst_unit = max(u[3] + 2.0 * u[4] for u in USERS)
print()
print(f"most expensive single request: {worst_unit:.1f} cost units")
print(f"cheapest typical request:      {USERS[0][3]:.1f} cost units")
print(f"heterogeneity:                 {worst_unit / USERS[0][3]:.0f}x")

print()
print()
print("A request-count limit set to bound worst-case cost at %.0f per minute."
      % COST_CEILING)
print("The limit must assume every request is the most expensive kind.")
print()
SAFE_RPM = COST_CEILING / worst_unit
print(f"limit = {COST_CEILING:.0f} / {worst_unit:.1f} = {SAFE_RPM:.1f} requests/min")
print()
print(f"{'profile':>20}{'wants req/min':>16}{'allowed':>10}{'served':>10}"
      f"{'throttled':>12}")
print("-" * 68)
starve = {}
for u in USERS:
    served = min(u[2], SAFE_RPM)
    thr = 1.0 - served / u[2]
    starve[u[0]] = thr
    print(f"{u[0]:>20}{u[2]:>16.1f}{SAFE_RPM:>10.1f}{served:>10.1f}"
          f"{thr:>12.0%}")

weighted_starve = sum(u[1] * starve[u[0]] for u in USERS)
print()
print(f"population-weighted throttling: {weighted_starve:.0%} of requested traffic")

print()
print()
print("Now a request-count limit set fairly -- at what a typical user needs.")
print("Cost is no longer bounded.")
print()
FAIR_RPM = 12.0
print(f"limit = {FAIR_RPM:.0f} requests/min (covers the 86% of users who are")
print("interactive)")
print()
print(f"{'profile':>20}{'allowed req/min':>18}{'typical cost':>15}"
      f"{'peak cost':>12}{'vs ceiling':>13}")
print("-" * 78)
over = {}
for u in USERS:
    allowed = min(u[2], FAIR_RPM)
    typ = allowed * u[3]
    pk = allowed * (u[3] + 2.0 * u[4])
    over[u[0]] = pk / COST_CEILING
    print(f"{u[0]:>20}{allowed:>18.1f}{typ:>15.1f}{pk:>12.1f}"
          f"{pk / COST_CEILING:>12.2f}x")

print()
print()
print("The two horns, side by side. Neither request-count limit does both jobs.")
print()
print(f"{'limit':>26}{'throttling':>13}{'worst-case cost':>18}"
      f"{'bounds cost':>14}{'fair':>7}")
print("-" * 78)
for label, rpm in (("safe (%.1f req/min)" % SAFE_RPM, SAFE_RPM),
                   ("fair (%.0f req/min)" % FAIR_RPM, FAIR_RPM)):
    thr = sum(u[1] * (1.0 - min(u[2], rpm) / u[2]) for u in USERS)
    wc = max(min(u[2], rpm) * (u[3] + 2.0 * u[4]) for u in USERS)
    print(f"{label:>26}{thr:>13.0%}{wc:>18.1f}"
          f"{('yes' if wc <= COST_CEILING else 'no'):>14}"
          f"{('yes' if thr < 0.10 else 'no'):>7}")

print()
print()
print("A cost-based limiter instead: spend a budget of %.0f cost units per minute,"
      % COST_CEILING)
print("charging each request what it actually costs.")
print()
print(f"{'profile':>20}{'cost/min wanted':>18}{'budget':>10}{'served':>10}"
      f"{'throttled':>12}")
print("-" * 70)
cost_starve = {}
for u in USERS:
    want = cost_wanted(u)
    served_cost = min(want, COST_CEILING)
    thr = 1.0 - served_cost / want
    cost_starve[u[0]] = thr
    print(f"{u[0]:>20}{want:>18.1f}{COST_CEILING:>10.1f}{served_cost:>10.1f}"
          f"{thr:>12.0%}")

cost_weighted = sum(u[1] * cost_starve[u[0]] for u in USERS)
cost_worst = COST_CEILING
print()
print(f"population-weighted throttling: {cost_weighted:.0%}")
print(f"worst-case cost per tenant:     {cost_worst:.0f} (bounded by construction)")

print()
print()
print("All three, on the two questions a limiter has to answer.")
print()
print(f"{'limiter':>26}{'throttling':>13}{'worst-case cost':>18}"
      f"{'bounds cost':>14}{'fair':>7}")
print("-" * 78)
rows = [
    ("count, safe", weighted_starve,
     max(min(u[2], SAFE_RPM) * (u[3] + 2.0 * u[4]) for u in USERS)),
    ("count, fair", sum(u[1] * (1.0 - min(u[2], FAIR_RPM) / u[2]) for u in USERS),
     max(min(u[2], FAIR_RPM) * (u[3] + 2.0 * u[4]) for u in USERS)),
    ("cost-based", cost_weighted, cost_worst),
]
for label, thr, wc in rows:
    print(f"{label:>26}{thr:>13.0%}{wc:>18.1f}"
          f"{('yes' if wc <= COST_CEILING + 0.5 else 'no'):>14}"
          f"{('yes' if thr < 0.10 else 'no'):>7}")

print(f"""
The heterogeneity number is what breaks the usual design. The most expensive request
in this population costs {worst_unit:.1f} units and the cheapest typical one costs
{USERS[0][3]:.1f} -- a spread of **{worst_unit / USERS[0][3]:.0f} times** within a
single service. A request-count limiter has one number to set and that spread has to
fit inside it.

Set it safely and the arithmetic is brutal. Bounding spend at {COST_CEILING:.0f}
units a minute requires assuming every request is the most expensive kind, which
gives {SAFE_RPM:.1f} requests per minute. That throttles the light interactive
users -- who want {USERS[0][2]:.0f} requests a minute and cost
{cost_wanted(USERS[0]):.1f} units doing it -- by {starve['light interactive']:.0%},
and throttles {weighted_starve:.0%} of all requested traffic.

Look at what that means for the light users specifically. They are throttled
{starve['light interactive']:.0%} while consuming {cost_wanted(USERS[0]):.1f} units a
minute against a {COST_CEILING:.0f} unit budget -- **{cost_wanted(USERS[0]) / COST_CEILING:.1%}
of the ceiling the limit exists to protect.** The limit that stopped them was computed
from a request none of them will ever send.

Set it fairly and the cost bound evaporates. At {FAIR_RPM:.0f} requests a minute the
interactive users are served properly, and the agent workload -- {USERS[3][1]:.0%} of
users -- reaches {over['agent workload']:.2f} times the cost ceiling at peak
(eq:count-limits-cannot-bound-cost). One tenant on one plan can spend several times
what the plan was priced for, and the limiter records no violation because no
violation occurred.

That is the shape of the failure: **a request-count limiter is either unfair or
unbounded, and which one you get depends on a number set once by whoever configured
it.** There is no value that does both jobs, because the quantity being limited is
not the quantity being protected.

The cost-based limiter closes it by construction. Charging each request what it
actually costs and spending a budget of {COST_CEILING:.0f} units a minute throttles
{cost_weighted:.0%} of traffic -- against {weighted_starve:.0%} for the safe count
limit -- while bounding worst-case spend at exactly {cost_worst:.0f}, because the
bound IS the mechanism rather than an inference from one.

The light users are no longer throttled at all: at {cost_wanted(USERS[0]):.1f} units
a minute they are nowhere near a {COST_CEILING:.0f} budget, and their consumption is
priced correctly rather than assumed to be worst-case.

Two implementation notes follow from that, and both are awkward. The first is that
the cost of a request is not known when it arrives -- generation cost depends on
output length, which ch:sd-async established is unknown at admission. So a cost
limiter must either reserve a pessimistic amount and refund the difference, or admit
the request and charge afterwards, running slightly over budget by design.

The second is that a cost limiter needs a cost model, and a cost model is a thing that
drifts. When the model changes, the price per token changes, and every tenant's
effective rate limit changes with it -- silently, with no configuration edit and no
deployment anyone would connect to the resulting complaints.""")
```

## 9. Practical Example

Four user profiles within one service:

```
             profile   share   req/min   cost/req   spread   cost/min     peak
------------------------------------------------------------------------------
   light interactive     62%       4.0        1.0      0.4        4.0      7.2
   heavy interactive     24%      11.0        2.8      1.6       30.8     66.0
   document workload     11%       6.0       14.0     11.0       84.0    216.0
      agent workload      3%      25.0       38.0     34.0      950.0   2650.0
```

The most expensive single request costs **106.0** units; the cheapest typical one
costs **1.0**. That is **106×** heterogeneity inside one service, and a
request-count limiter has one number for it.

Setting the limit to bound spend at 260 units/minute requires assuming every request
is the most expensive kind, giving **2.5** requests/minute:

```
             profile   wants req/min   allowed    served   throttled
--------------------------------------------------------------------
   light interactive             4.0       2.5       2.5         39%
   heavy interactive            11.0       2.5       2.5         78%
   document workload             6.0       2.5       2.5         59%
      agent workload            25.0       2.5       2.5         90%
```

**52%** of all requested traffic is throttled. The light users are throttled **39%**
while consuming 4.0 units a minute against a 260-unit budget — **1.5%** of the
ceiling the limit exists to protect. The limit that stopped them was computed from a
request none of them will ever send.

Setting it fairly at 12 requests/minute instead:

```
             profile   allowed req/min   typical cost   peak cost   vs ceiling
------------------------------------------------------------------------------
   light interactive               4.0            4.0         7.2        0.03x
   heavy interactive              11.0           30.8        66.0        0.25x
   document workload               6.0           84.0       216.0        0.83x
      agent workload              12.0          456.0      1272.0        4.89x
```

The agent workload — **3%** of users — reaches **4.89×** the cost ceiling at peak
({{eq:count-limits-cannot-bound-cost}}), and the limiter records no violation because
none occurred.

All three designs together:

```
                   limiter   throttling   worst-case cost   bounds cost   fair
------------------------------------------------------------------------------
               count, safe          52%             260.0           yes     no
               count, fair           2%            1272.0            no    yes
                cost-based           2%             260.0           yes    yes
```

**A request-count limiter is either unfair or unbounded**, and which one you get
depends on a number set once by whoever configured it. The cost-based limiter
achieves **2%** throttling — matching the fair count limit — while bounding spend at
exactly **260**, because the bound is the mechanism rather than an inference from one.

```mermaid {#fig:limiter caption="A count limiter has one setting and two jobs. Bounding cost forces it to the worst-case request; being fair forces it to typical demand. The ranges are disjoint whenever heterogeneity is large."}
flowchart TD
  A["one request-count limit"] --> B["set for worst-case request<br/>2.5 req/min"]
  A --> C["set for typical user<br/>12 req/min"]
  B --> D["cost bounded<br/>52% throttled"]
  C --> E["2% throttled<br/>4.89x ceiling"]
  F["cost-based limit<br/>260 units/min"] --> G["cost bounded<br/>2% throttled"]
```

The second listing turns to delegated authority.

```python {tier=A name=ca2}
"""When an agent acts for a user, the permission check moves to the wrong boundary.

A user calls an API. The API runs an agent. The agent calls tools. The tools touch
data. Somewhere in that chain a decision gets made about what is allowed, and WHERE
it gets made determines how much authority the user effectively gains.

The convenient implementation gives the agent a service account -- one identity with
enough permission to serve any user. Every request then runs with the union of every
user's rights, and the only thing preventing one user's request from reaching another
user's data is the model choosing not to
(eq:delegation-moves-the-check).

This listing measures the over-permission each delegation design produces, against
what it costs to build.
"""
# Tools the agent can call, and what each needs.
# (tool, scope required, share of requests using it, blast radius if misused)
TOOLS = [
    ("search_documents",   "docs:read",      0.91,  40.0),
    ("read_document",      "docs:read",      0.74,  40.0),
    ("list_users",         "dir:read",       0.12, 120.0),
    ("send_message",       "msg:write",      0.09, 260.0),
    ("update_record",      "crm:write",      0.06, 900.0),
    ("run_report",         "analytics:read", 0.04, 310.0),
    ("delete_document",    "docs:delete",    0.01, 1400.0),
]

# What a typical individual user is actually entitled to.
# Share of users holding each scope.
ENTITLED = {
    "docs:read":      0.98,
    "dir:read":       0.34,
    "msg:write":      0.55,
    "crm:write":      0.12,
    "analytics:read": 0.09,
    "docs:delete":    0.04,
}

DESIGNS = [
    # (name, effective scope per request, engineering cost, audit fidelity)
    ("service account",        "union",     1.0, 0.15),
    ("service account + row filter", "union-filtered", 2.4, 0.35),
    ("token passthrough",      "user",      4.1, 0.95),
    ("per-tool exchanged token", "user-tool", 6.8, 1.00),
]


def over_permission(design):
    """Expected share of a request's tool calls the agent could make that the
    invoking user could not make directly, weighted by blast radius."""
    total = 0.0
    excess = 0.0
    for tool, scope, freq, blast in TOOLS:
        held = ENTITLED[scope]
        total += freq * blast
        if design == "union":
            # The agent holds every scope regardless of the user.
            excess += freq * blast * (1.0 - held)
        elif design == "union-filtered":
            # Row filtering catches data the user cannot see, but only where a
            # row-level owner exists. Writes and directory calls have no owner
            # column to filter on.
            filterable = 0.6 if scope.endswith(":read") else 0.0
            excess += freq * blast * (1.0 - held) * (1.0 - filterable)
        elif design == "user":
            # The user's own token is presented; nothing is gained.
            excess += 0.0
        elif design == "user-tool":
            excess += 0.0
    return excess, total


print("Seven tools an agent can call, with what each requires and what a misuse")
print("would reach.")
print()
print(f"{'tool':>20}{'scope':>18}{'used in':>10}{'blast radius':>15}"
      f"{'users holding':>15}")
print("-" * 78)
for tool, scope, freq, blast in TOOLS:
    print(f"{tool:>20}{scope:>18}{freq:>10.0%}{blast:>15.0f}"
          f"{ENTITLED[scope]:>15.0%}")

print()
print()
print("A service account needs the union of every scope, because it must be able")
print("to serve any user. That is what every request then runs with.")
print()
union = sorted(set(s for _, s, _, _ in TOOLS))
print(f"scopes in the union:  {len(union)}")
print(f"scopes a median user holds: "
      f"{sum(1 for s in union if ENTITLED[s] >= 0.5)}")
print()
print(f"{'scope':>18}{'in union':>11}{'users holding':>16}"
      f"{'granted to':>14}")
print("-" * 59)
for s in union:
    print(f"{s:>18}{'yes':>11}{ENTITLED[s]:>16.0%}{1.0:>13.0%}")

print()
print()
print("Over-permission by design: the share of blast radius the agent can reach")
print("that the invoking user could not reach directly.")
print()
print(f"{'design':>30}{'over-permission':>18}{'eng cost':>11}"
      f"{'audit fidelity':>17}")
print("-" * 76)
res = {}
for name, mode, cost, audit in DESIGNS:
    ex, tot = over_permission(mode)
    res[name] = (ex / tot, cost, audit, ex)
    print(f"{name:>30}{ex / tot:>18.1%}{cost:>10.1f}x{audit:>17.0%}")

print()
print()
print("Where the over-permission sits, under a service account. Ranked by")
print("exposure, which is frequency times blast radius times the share of users")
print("who could NOT do it themselves.")
print()
print(f"{'tool':>20}{'used in':>10}{'blast':>9}{'lacking':>10}"
      f"{'exposure':>12}{'share of total':>17}")
print("-" * 78)
ex_total = over_permission("union")[0]
rows = []
for tool, scope, freq, blast in TOOLS:
    e = freq * blast * (1.0 - ENTITLED[scope])
    rows.append((tool, freq, blast, 1.0 - ENTITLED[scope], e))
rows.sort(key=lambda r: -r[4])
for tool, freq, blast, lack, e in rows:
    print(f"{tool:>20}{freq:>10.0%}{blast:>9.0f}{lack:>10.0%}"
          f"{e:>12.1f}{e / ex_total:>17.0%}")

print()
print()
print("The cheap partial fix: remove the two highest-exposure tools from the")
print("agent's set and require a human to invoke them directly.")
print()
drop = [rows[0][0], rows[1][0]]
kept = [t for t in TOOLS if t[0] not in drop]
kept_ex = sum(f * b * (1.0 - ENTITLED[s]) for _, s, f, b in
              [(t[0], t[1], t[2], t[3]) for t in kept])
kept_tot = sum(f * b for _, _, f, b in
               [(t[0], t[1], t[2], t[3]) for t in kept])
print(f"removed: {drop[0]}, {drop[1]}")
print()
print(f"{'configuration':>34}{'over-permission':>18}{'requests affected':>20}")
print("-" * 72)
affected = sum(t[2] for t in TOOLS if t[0] in drop)
print(f"{'service account, all seven tools':>34}"
      f"{res['service account'][0]:>18.1%}{0.0:>20.0%}")
print(f"{'service account, five tools':>34}{kept_ex / kept_tot:>18.1%}"
      f"{affected:>20.0%}")
print(f"{'token passthrough, all seven':>34}"
      f"{res['token passthrough'][0]:>18.1%}{0.0:>20.0%}")

print(f"""
The union table is the whole problem stated once. Serving any user requires
{len(union)} scopes, and a median user holds
{sum(1 for s in union if ENTITLED[s] >= 0.5)} of them. Every request therefore
executes with authority that most of the people making requests do not have.

Under a service account, **{res['service account'][0]:.1%} of the blast radius the
agent can reach is unreachable by the user who invoked it**
(eq:delegation-moves-the-check). The only thing standing between a request and that
authority is the model deciding not to use it -- which ch:ag-security established is
not a security control, because a model's decision is exactly what prompt injection
targets.

Row-level filtering is the usual mitigation and it recovers less than it appears to.
It drops over-permission from {res['service account'][0]:.1%} to
{res['service account + row filter'][0]:.1%}, because filtering works on reads with
an owner column and does not work on writes, on directory lookups, or on anything
whose authorisation is not expressible as a row predicate. **The tools it cannot
protect are the ones with the largest blast radius**, which is the opposite of the
distribution you want a partial mitigation to have.

Token passthrough takes it to {res['token passthrough'][0]:.1%} -- the agent can do
exactly what the user can do -- at {res['token passthrough'][1]:.1f} times the
engineering cost and with audit fidelity rising from
{res['service account'][2]:.0%} to {res['token passthrough'][2]:.0%}. That second
column matters more than it looks: under a service account, the audit log records
that the service account read a document, which is true and useless.

The exposure ranking is where this becomes actionable, because the distribution is
not flat. `{rows[0][0]}` accounts for {rows[0][4] / ex_total:.0%} of all
over-permission and is used in {rows[0][1]:.0%} of requests;
`{rows[1][0]}` accounts for {rows[1][4] / ex_total:.0%}. Together those two are
{(rows[0][4] + rows[1][4]) / ex_total:.0%} of the exposure and appear in
{affected:.0%} of traffic.

So the cheap intervention is available before the expensive one. Removing those two
tools from the agent's set -- requiring a human to invoke them directly -- takes
over-permission from {res['service account'][0]:.1%} to {kept_ex / kept_tot:.1%}
while affecting {affected:.0%} of requests -- **closing
{(res['service account'][0] - kept_ex / kept_tot) / res['service account'][0]:.0%} of
the gap to full passthrough by editing a list**, against
{res['token passthrough'][1]:.1f} times the engineering cost for the remainder.

**The agent's effective authority is a design variable, and the cheapest way to
reduce it is to give the agent fewer tools rather than better credentials.** That is
worth stating plainly because the instinct runs the other way: teams add tools to
make the agent more capable, and reach for a delegation rewrite only once the
security review objects.

The general form is ch:mcp-security's boundary question arriving as an API design
decision. An authorisation check is only meaningful at a boundary where the identity
of the principal is still known, and every hop that replaces a user identity with a
service identity erases the information the check needed. **Authority is not lost
gradually across a call chain; it is lost at exactly one hop**, and which hop that is
is a choice.""")
```

Seven tools, with what each requires and what a misuse would reach:

```
                tool             scope   used in   blast radius  users holding
------------------------------------------------------------------------------
    search_documents         docs:read       91%             40            98%
       read_document         docs:read       74%             40            98%
          list_users          dir:read       12%            120            34%
        send_message         msg:write        9%            260            55%
       update_record         crm:write        6%            900            12%
          run_report    analytics:read        4%            310             9%
     delete_document       docs:delete        1%           1400             4%
```

Serving any user requires **6** scopes; a median user holds **2**. Every request
under a service account therefore executes with authority most requesters do not have.

```
                        design   over-permission   eng cost   audit fidelity
----------------------------------------------------------------------------
               service account             50.8%       1.0x              15%
  service account + row filter             43.6%       2.4x              35%
             token passthrough              0.0%       4.1x              95%
      per-tool exchanged token              0.0%       6.8x             100%
```

Under a service account, **50.8%** of reachable blast radius is unreachable by the
invoking user ({{eq:delegation-moves-the-check}}). Row-level filtering — the usual
mitigation — reaches only **43.6%**, because it works on reads with an owner column
and not on writes, directory lookups, or anything not expressible as a row predicate.
**The tools it cannot protect are the ones with the largest blast radius.**

Note the audit column. Under a service account, audit fidelity is **15%**: the log
records that *the service account* read a document, which is true and useless.

Where the exposure actually sits:

```
                tool   used in    blast   lacking    exposure   share of total
------------------------------------------------------------------------------
       update_record        6%      900       88%        47.5              51%
     delete_document        1%     1400       96%        13.4              14%
          run_report        4%      310       91%        11.3              12%
        send_message        9%      260       45%        10.5              11%
          list_users       12%      120       66%         9.5              10%
    search_documents       91%       40        2%         0.7               1%
       read_document       74%       40        2%         0.6               1%
```

`update_record` alone is **51%** of all over-permission while appearing in **6%** of
requests. The two most-used tools together are **2%**.

```
                     configuration   over-permission   requests affected
------------------------------------------------------------------------
  service account, all seven tools             50.8%                  0%
       service account, five tools             28.1%                  7%
      token passthrough, all seven              0.0%                  0%
```

Removing the two highest-exposure tools takes over-permission from **50.8%** to
**28.1%** while affecting **7%** of requests — closing **45%** of the gap to full
passthrough by editing a list ({{eq:fewer-tools-beats-better-credentials}}), against
**4.1×** engineering cost for the remainder.

## 10. Production Considerations

Limit on cost, and reserve pessimistically at admission. The reservation percentile is
a real decision with the shape of every other tail choice in this part; the p90 of the
cost distribution is a reasonable default.

Version the limiter's cost model separately from the serving model, and treat a change
to it as a change to every tenant's limit — because it is.

Keep a request-count limit as well, set generously. It is not a cost control but it is
a good abuse control, and the two failure modes are different.

Publish the cost unit to customers. A limit expressed in requests is legible and
wrong; a limit expressed in cost units is correct and needs documentation. Systems
that hide the unit generate support load proportional to their heterogeneity.

Rank tools by $f_t\beta_t(1 - h_\sigma)$ before designing a delegation model. It is a
spreadsheet, it takes an hour, and it usually identifies a change that closes half the
gap for no engineering cost.

Carry the user identity to the tool boundary wherever the request is user-initiated,
and treat background work as a separate, explicitly-scoped case rather than the reason
not to.

Log the invoking principal, not the executing one. Audit fidelity of **15%** means the
security team cannot answer the question they will be asked first.

## 11. Common Mistakes

**Rate limiting on request count.** Cannot bound cost when costs are heterogeneous;
no setting does both jobs.

**Charging cost after the fact without reserving.** Admits the overspend you were
trying to prevent.

**Treating row-level filtering as a delegation model.** It protects the tools that
did not need protecting.

**Adding tools without re-examining the service account's union.** Each tool widens
authority permanently.

**Relying on the model to decline.** {{ch:ag-security}} established this is not a
control; {{eq:delegation-moves-the-check}} quantifies what it is standing in for.

**Auditing the service account.** Records something true and useless.

## 12. Failure Modes

**Silent limit shift on model change.** The cost model changes, every tenant's
effective limit changes, and no configuration was edited.

**Reservation leak.** Reservations not released on error accumulate until a tenant is
throttled to zero with no traffic.

**Confused deputy via agent.** A user induces the agent to use a scope the user does
not hold; every log entry shows the service account acting normally.

**Background-work scope creep.** A credential minted for background work outlives its
purpose and becomes a second service account.

**Throttling inversion.** Under a count limit, cheap high-volume users are throttled
while an expensive user is admitted — the failure is invisible because both are
within policy.

## 13. Alternatives

**Token-bucket on tokens rather than cost.** Simpler than a full cost model and
captures most of the heterogeneity, since token count is the dominant cost term.
Loses accuracy where tools or retrieval dominate.

**Per-tenant hard spend caps with hard cutoff.** Bounds cost absolutely and produces a
worse experience than throttling; appropriate as a backstop beneath a limiter, not as
the limiter.

**Admission by predicted cost.** Use a classifier to estimate cost before serving,
avoiding the reservation. Trades reservation over-throttling for prediction error, and
{{ch:sd-async}}'s length-prediction discussion applies unchanged.

**Capability-scoped agents.** Run several narrow agents with disjoint tool sets rather
than one broad agent, so the union is never formed. Attacks
{{eq:delegation-moves-the-check}} structurally and costs routing complexity.

**Human-in-the-loop for high-exposure tools.** The intervention
{{eq:fewer-tools-beats-better-credentials}} prices — cheapest available, and correct
whenever the affected traffic share is small.

## 14. Evaluation

Measure throttling by user segment, not in aggregate. An aggregate throttling rate
hides the inversion where cheap users are the ones being stopped.

Report worst-case realised spend per tenant per window against the ceiling. This is
the number a count limiter cannot bound and the one a cost limiter bounds by
construction.

Measure reservation accuracy — the distribution of $\hat{c} - c$ — and tune the
percentile against observed over-throttling.

Compute over-permission from real entitlement data rather than assumed. The
$h_\sigma$ values dominate {{eq:delegation-moves-the-check}} and are the input most
often guessed.

Audit whether the invoking principal appears in tool-call logs. This is a yes/no
question with a large security consequence and it takes one query to answer.

## 15. Advanced Concepts

The two halves of this chapter interact in a way worth noting. A cost-based limiter
needs to attribute cost to a principal, and under a service account there is only one
principal. **A system that has lost user identity at the agent boundary cannot
rate-limit per user either** — the same hop that broke authorization broke
attribution, and teams usually discover the second consequence months after accepting
the first.

The over-permission model treats each tool call independently, which understates the
risk in a specific way. An agent that can both `search_documents` and `send_message`
can exfiltrate: read something the user could not see, then transmit it somewhere the
user could not reach. Neither tool is individually the problem; the *composition* is.
Formally, exposure should be computed over reachable pairs rather than over tools,
which raises the count from $|T|$ to $O(|T|^2)$ and changes which removals are
effective. {{sec:19-research-questions}} takes this up.

A second interaction concerns what happens when a limiter throttles an agent
mid-task. A rejected request in a conventional API is a rejected request; a rejected
tool call halfway through an agent's plan leaves partial work, and by
{{ch:as-state-machines}}'s reasoning that partial work may not be safely resumable.
So a limiter that throttles at the tool boundary is making a correctness decision
disguised as a capacity one. The cleaner design admits or rejects a whole *task* by
reserving its estimated total cost up front, which is harder -- the estimate spans an
unknown number of steps -- but keeps rejection at a boundary where nothing has
happened yet. This is the same admission-versus-scheduling distinction
{{ch:sd-async}} reached from the queueing side, and it lands in the same place: the
lever worth having is at admission.

The reservation policy in {{eq:cost-limits-need-a-reservation}} interacts with
{{ch:sd-async}}'s queueing result. Reserving at the p90 of cost means holding budget
that is usually not spent, which reduces effective concurrency — so an aggressive
reservation policy is also a capacity reduction, and the two should be tuned together
rather than separately by different teams.

## 16. Connection to Previous Chapters

{{eq:three-properties-break-the-stack}} from {{ch:sd-architecture}} listed the
techniques that break under cost heterogeneity. Rate limiting was not in that list and
should have been; this chapter supplies the missing row.

{{eq:variance-not-mean-drives-wait}} from {{ch:sd-async}} and
{{eq:count-limits-cannot-bound-cost}} are two consequences of one distribution — one
in the queue, one at the gate.

{{eq:semantic-failure-has-no-instrument}} appears again in the audit column: a log
that faithfully records the wrong principal is accurate and useless.

{{ch:ag-security}} and {{ch:mcp-security}} established that model compliance is not a
control. {{eq:delegation-moves-the-check}} measures how much is being asked of it.

## 17. Exercises

1. Derive the heterogeneity threshold above which no request-count limit satisfies
   both a given ceiling and a given fairness target.

2. Extend the first listing with a reservation policy at the p75, p90, and p99 of
   cost. Plot over-throttling against overspend.

3. Compute over-permission for a tool set you work with, using real entitlement rates.
   Which two tools dominate?

4. Modify the second listing to compute exposure over tool *pairs* rather than single
   tools. Does the ranking change?

5. Design the background-work credential for an agent you know: scope, expiry, and
   what revokes it.

## 18. Interview Questions

1. Our rate limit is 100 requests per minute and our bill is three times forecast.
   Explain how both can be true.

2. Why can a cost-based limiter not simply charge after the request completes?

3. An agent uses a service account. What is the security property you have given up,
   and what is standing in for it?

4. Row-level filtering was added and over-permission barely moved. Why?

5. You have one sprint. Would you spend it on token passthrough or on reducing the
   tool set, and what number would you look at to decide?

## 19. Research Questions

1. How much does pair-wise composition raise measured over-permission over the
   per-tool figure, on real tool sets?

2. Can per-request cost be predicted accurately enough at admission to replace
   reservation, and what does the prediction error cost relative to over-reservation?

3. Is there a delegation design with passthrough's security properties and materially
   less than **4.1×** the engineering cost?

4. How should exposure be weighted when tool failures are correlated
   ({{cite:cemri2025mast}}) rather than independent?

## 20. Chapter Summary

A request-count rate limiter has one setting and two jobs. With **106×** cost
heterogeneity inside one service, bounding spend requires a 2.5 requests/minute limit
that throttles **52%** of traffic — including light users consuming **1.5%** of the
ceiling — while a fair 12 requests/minute limit lets one tenant reach **4.89×** the
ceiling with no violation recorded ({{eq:count-limits-cannot-bound-cost}}).

A cost-based limiter achieves **2%** throttling with spend bounded at exactly the
ceiling, because the bound is the mechanism. Its complication is that cost is unknown
at admission, requiring a reservation and refund
({{eq:cost-limits-need-a-reservation}}).

When an agent runs under a service account, **50.8%** of the blast radius it can reach
is unreachable by the invoking user ({{eq:delegation-moves-the-check}}), with audit
fidelity of **15%**. Row-level filtering reaches only **43.6%**, because it protects
reads and not the high-blast-radius writes.

Exposure concentrates: `update_record` is **51%** of it while appearing in **6%** of
requests. Removing the two worst tools takes over-permission to **28.1%** — **45%** of
the way to full passthrough — by editing a list, against **4.1×** engineering cost for
the rest ({{eq:fewer-tools-beats-better-credentials}}).

Both halves also share a diagnostic. In each case the system was configured with a
number that felt like the right control -- requests per minute, a service account
scope list -- and in each case the number was a proxy for the thing that actually
mattered. The proxy was chosen because it was easy to measure and easy to enforce,
and it kept working until the underlying quantities stopped being proportional to
it. Heterogeneity is what breaks proportionality, and heterogeneity is exactly what
a model-backed workload introduces.

Carry forward: **limit the quantity you are protecting**, and **give the agent fewer
tools before giving it better credentials**.

## 21. Further Reading

- {{cite:mcp2026spec}} — stateless request model; why identity must travel explicitly.
- {{cite:hou2025mcp}} — tool-protocol authorization, and where the principal is lost.
- {{cite:qin2023toolllm}} — large tool collections; tool count as a design variable.
- {{cite:cemri2025mast}} — correlated agent failures, which make exposure joint rather
  than per-call.
