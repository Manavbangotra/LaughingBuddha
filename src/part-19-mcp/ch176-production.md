---
id: mcp-production
number: 176
part: XIX
tier: full
status: draft
requires: [lifecycle-decides-the-rate, retrieval-crossover-is-small,
           tail-mass-decides, habituation]
provides: [admission-is-a-router, review-does-not-scale,
           marginal-server-turns-negative, retrieval-moves-the-optimum,
           registry-policy-sets-host-capability]
citations: [hou2025mcp, huang2026mcpthreat, gaire2025mcpsok, mcp2026spec,
            qin2023toolllm]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why registry admission
policy is the same decision as {{ch:ag-what-is-an-agent}}'s router-versus-agent
choice; show why human review of submissions does not scale and what does; compute
the point at which the marginal connected server makes a host worse; explain how a
retrieval layer changes that number rather than merely improving it; and state the
connection between registry governance and host capability that is usually left
out of the security argument.

## 2. Why This Matters

{{ch:mcp-security}} ended on an uncomfortable observation: every defence it
measured operates on the poisoned fraction $\pi$ and the hostile-turn rate
$\lambda$ as *given*, and neither is observable from inside a session. They are
outcomes of registry policy ({{eq:lifecycle-decides-the-rate}}), which is
{{cite:hou2025mcp}}'s point about the server lifecycle.

So this chapter asks the two questions that policy actually decides.

**What should a registry admit?** {{sec:9-practical-example}} finds the answer
has a familiar shape. A strict registry admits fewer bad servers *and fewer good
ones* — {{ch:ag-what-is-an-agent}}'s router-versus-agent trade with publishers in
place of requests ({{eq:admission-is-a-router}}). Signing requirements blocked
about $40$ bad servers and turned away $186$ good ones.

More usefully, it finds that the two obvious mechanisms behave completely
differently under growth. A signing requirement holds the poisoned share at
$2.00\%$ whether there are $150$ submissions or $15{,}000$. Human review takes it
from $2.74\%$ at $150$ to $5.92\%$ at $15{,}000$ — back to almost exactly the
open-registry rate. **Review does not scale** ({{eq:review-does-not-scale}}),
because capacity is fixed while submissions are not, and adding reviewers runs
into {{ch:ag-termination}}'s habituation rather than out of it.

**How many servers should a host connect?** This is the part's closing question
and nobody computes it. Capability saturates while schema rent and hostile-server
exposure do not, so there is an interior optimum
({{eq:marginal-server-turns-negative}}): success peaked at eight servers and fell
from $59.4\%$ to $38.5\%$ by thirty-two.

Two things move that number. A retrieval layer moves the optimum from eight
servers to sixteen and is worth $+39.2$ points at sixty-four
({{eq:retrieval-moves-the-optimum}}). And registry quality moves it too — from
sixteen servers at a $6\%$ hostile rate to thirty-two at $0.2\%$. **A
well-governed registry does not merely make hosts safer; it makes them more
capable** ({{eq:registry-policy-sets-host-capability}}), which is not the argument
usually made for governance.

## 3. Prerequisites

{{ch:mcp-security}}'s {{eq:lifecycle-decides-the-rate}}, which is the handoff into
this chapter.

{{ch:mcp-schemas}}'s {{eq:retrieval-crossover-is-small}} — the retrieval layer
that turns out to change the answer here.

{{ch:ag-what-is-an-agent}}'s {{eq:tail-mass-decides}}, applied to publishers rather
than requests.

{{ch:ag-termination}}'s {{eq:habituation}}, which appears here in a review queue
rather than an approval dialogue.

## 4. Intuitive Explanation

An ecosystem needs somewhere to find servers. Call it a registry, and the first
question is who may publish to it.

The permissive answer: anyone. That is how package registries work, and it is why
they have the supply-chain problems they have. The restrictive answer: only
reviewed and verified publishers. That is safer and it is also how a registry ends
up with two hundred entries while the problem space has ten thousand.

{{ch:ag-what-is-an-agent}} met this shape already, deciding between a router and an
agent. A router handles enumerated cases safely and fails on anything unanticipated;
an agent covers everything and is harder to guarantee. The deciding number was tail
mass — how much of the distribution lies outside what you enumerated.

Registry admission is the same decision with publishers on the input. A strict
policy is a router over publishers: it admits the head — established vendors,
signed identities, organisations with review budgets — and turns away the tail. And
the tail of the publisher distribution is exactly where the unusual integrations
live: the one server for a niche internal system, the hobbyist adapter for
equipment nobody else supports.

{{sec:9-practical-example}} prices that at roughly fourteen good servers turned
away per bad one blocked. Whether that is a good trade depends entirely on whether
the rejected servers were replaceable, which is a judgement about your ecosystem
rather than a fact about registries.

But there is a sharper finding underneath, and it is about *which* strict policy.

Two mechanisms are available. **Verification of identity** — you must sign your
submission with a verified publisher identity — costs the publisher effort, once,
and costs the registry nothing per submission. **Human review** — someone reads
each submission — costs the registry attention per submission.

Under growth these diverge completely. Signing is a filter applied to every
submission regardless of volume. Review has a fixed capacity; when submissions
exceed it, the excess is admitted unreviewed, and the registry's stated policy
stops describing what actually happens. {{sec:9-practical-example}} shows review's
effectiveness decaying back toward the open-registry rate as submissions grow,
while signing holds flat.

And staffing the review does not fix it, because
{{ch:ag-termination}}'s habituation applies to a queue exactly as it applied to an
approval dialogue: a reviewer working through hundreds of submissions is not
reading the last one the way they read the first.

Now the host's side of the same ecosystem.

A host connects servers. Each one brings capability, and each brings costs this
part has already measured: schema rent on every request
({{ch:mcp-schemas}}), a chance of being hostile ({{ch:mcp-security}}), a shared
failure mode ({{ch:mcp-architecture}}).

The capability *saturates* — the tenth issue-tracker integration adds less than the
first, because the first already covered issue tracking. The costs do not saturate;
they are roughly linear in server count.

Saturating benefit against linear cost has a maximum, and past it every additional
server makes the host worse at everything, not merely more expensive.
{{sec:9-practical-example}} locates it, and the decomposition at the optimum is the
clearest way to see the condition: going from sixteen servers to seventeen adds
$0.9\%$ of task coverage and $0.84\%$ of harm exposure. The next server is not
worth having, and the one after that is worse.

## 5. Formal Explanation

**Admission.** Let $S$ submissions arrive, a fraction $\beta$ of them bad. A
policy admits with probabilities $(a_g, a_b)$ for good and bad submissions. The
admitted poisoned share and coverage are:

$$\pi = \frac{\beta a_b}{\beta a_b + (1-\beta)a_g}, \qquad c = a_g$$ (eq:admission-is-a-router)

Every policy is a point on a curve trading $\pi$ against $c$, which is
{{eq:tail-mass-decides}}'s structure exactly: the deciding quantity is how much
value lies in the submissions a strict $a_g$ excludes.

**Review capacity.** A structural filter has $a_b$ independent of $S$. Human review
with capacity $K$ reviews only $\min(K, S)$ submissions, so:

$$a_b^{\text{review}} = 1 - \frac{\min(K,S)}{S}\,\kappa(\ell), \qquad \ell = \frac{\min(K,S)}{R}$$ (eq:review-does-not-scale)

with $R$ reviewers and $\kappa$ the habituation curve from
{{eq:habituation}}. Two terms degrade together as $S$ grows: the *fraction*
reviewed falls as $K/S$, and the catch rate on those reviewed falls as load rises.
Hence:

$$\lim_{S \to \infty} a_b^{\text{review}} = 1$$

**Review converges to no policy at all.** Adding reviewers raises $K$ linearly and
lowers $\ell$, but $\kappa$ is concave, so the return on staffing is sublinear
while the submission rate is exogenous.

**The marginal server.** Let a host connect $n$ servers, each with $t$ tools at
$\tau$ tokens. Capability coverage saturates and costs are linear:

$$S(n) = \underbrace{\big(1 - e^{-n/\sigma}\big)}_{\text{coverage}}\cdot\underbrace{\big(p_r(1 - \delta\, n t \tau)\big)^{k}}_{\text{rent}}\cdot\underbrace{\big(1 - \omega(1-(1-h)^n)\big)}_{\text{exposure}}$$ (eq:marginal-server-turns-negative)

Taking logs, the coverage term contributes $+e^{-n/\sigma}/\sigma$ to the
derivative — **exponentially decaying** — while the rent and exposure terms
contribute roughly constant negatives. So $\partial_n \log S$ crosses zero exactly
once, at:

$$n^* \approx \sigma \log\!\left(\frac{1}{\sigma\,(k\delta t\tau + \omega h)}\right)$$

Two consequences. A **retrieval layer** caps the rent term at $r$ shown schemas
rather than $nt$, removing $k\delta t\tau$ from the denominator:

$$n^*_{\text{retrieval}} > n^*, \qquad \lim_{n\to\infty} S_{\text{retrieval}}(n) > 0$$ (eq:retrieval-moves-the-optimum)

**Retrieval changes where the optimum is, not just how high it is.** And the
hostile rate $h$ — a registry-policy outcome — appears in the same denominator:

$$\frac{\partial n^*}{\partial h} < 0$$ (eq:registry-policy-sets-host-capability)

**A better registry raises the number of servers a host can profitably connect**,
which is the link between the two halves of this chapter.

## 6. Mathematical Foundation

Three extractions.

**Capacity-bounded filters have an asymptote and structural filters do not.**
{{eq:review-does-not-scale}}'s limit is the whole argument: any policy whose cost
is per-submission-reviewed rather than per-submission-imposed converges to
permissiveness as the ecosystem succeeds. This generalises well past registries —
it is the same reason {{ch:as-long-running}} found automated re-validation scaling
where human gates did not.

**The optimum is logarithmic in the cost coefficients.** From
{{eq:marginal-server-turns-negative}}, $n^*$ depends on $\log(1/\text{cost})$, so
halving the rent or the hostile rate moves the optimum by a constant, not a factor.
That means improvements compound gently and no single fix produces a dramatic
change in how many servers are worth connecting — but it also means the optimum is
robust to getting the parameters somewhat wrong.

**Coverage saturation is the only term with a shape.** Rent and exposure are
essentially linear; the entire structure of the problem is in $\sigma$, the number
of servers at which capability saturates. A host that knows its own $\sigma$ — from
its task distribution — knows everything it needs, and $\sigma$ is measurable from
a trace.

## 7. Internal Mechanics

### 7.1 Admission policies, ordered by what they cost whom

```mermaid {#fig:admission caption="Admission mechanisms differ by who pays and whether the cost scales with submissions. The first three are per-submission on the publisher; the fourth is per-submission on the registry."}
flowchart TD
    P[submission] --> A{identity verified?}
    A -->|no| R1[rejected]
    A -->|yes| B{signed artefact?}
    B -->|no| R2[rejected]
    B -->|yes| C{automated scan passes?}
    C -->|no| R3[rejected]
    C -->|yes| D{human review capacity?}
    D -->|queue full| ADM1[admitted unreviewed]
    D -->|reviewed| E{approved?}
    E -->|no| R4[rejected]
    E -->|yes| ADM2[admitted]
```

The branch that matters is `queue full`. Every registry has one, whether or not it
is drawn, and it is where the stated policy and the actual policy diverge.

**Identity verification** costs the publisher a one-time setup and the registry
nothing per submission. It deters the low-effort bad actor and, per
{{sec:9-practical-example}}, some low-effort good ones.

**Artefact signing** costs the publisher a build-time step and gives the registry
the provenance {{cite:hou2025mcp}} says is missing — it is what makes
"this server has not changed since review" checkable.

**Automated scanning** — {{cite:huang2026mcpthreat}}'s static metadata analysis —
costs compute per submission, which scales. Its detection rate is a classifier's,
with all of {{ch:ag-security}}'s caveats, but it does not degrade with volume.

**Human review** is the only one whose cost is registry attention, and the only
one with the asymptote.

### 7.2 What a registry should publish

Registries publish downloads and stars. The numbers that matter to
{{eq:marginal-server-turns-negative}} are different and mostly absent:

**Provenance** — who published this, verified how, and does the artefact match.

**The tool audit** — {{ch:mcp-building}}'s annotations, aggregated: how many tools,
how many are writes, how many are irreversible. A host deciding whether to connect
a server should see its consequence profile before installing it.

**Schema token count.** {{ch:mcp-schemas}}'s rent, which is a fixed property of the
server and is currently discovered only after connecting.

**Definition history.** Whether the tool descriptions have changed, and when —
which turns {{ch:mcp-security}}'s {{eq:approval-is-a-snapshot}} from a per-host
problem into an ecosystem-level signal.

None of these are hard. All of them are computable from the server's own
`tools/list`, and a registry is the natural place to compute them once for
everyone.

### 7.3 Deprecation and the feature lifecycle

{{cite:mcp2026spec}} defines a deprecation policy worth noting as a model: a
deprecated feature stays in the specification for at least twelve months — or
ninety days under an expedited exception — documents a migration path, and only
then becomes eligible for removal.

That is the discipline {{ch:mcp-why}}'s connectivity result argues for. A twelve
month floor means implementations have time to widen their windows before anything
disappears, and the published registry of deprecated features means the deadline is
discoverable rather than folklore.

The failure mode it prevents is the one that produces version islands: a change
that is announced, adopted quickly by the fast movers, and leaves the slow half of
the ecosystem unable to talk to the fast half.

### 7.4 Operating servers: first-party, third-party, self-hosted

Three deployment relationships, with different parameters in this chapter's model.

**First-party** — the service operator publishes and runs the server. Lowest $h$,
because the operator's incentives are aligned and their identity is known. Highest
correlation, because everyone uses the same instance
({{ch:mcp-architecture}}).

**Third-party** — someone else wraps a service. Higher $h$, and the wrapper can
lag the underlying API, which is {{ch:as-long-running}}'s drift arriving as a
supply-chain property.

**Self-hosted** — you run the server yourself, from published source. Lowest
correlation and lowest $h$ for a given artefact, at the cost of operating it, and
it converts {{ch:mcp-why}}'s maintenance term back into something you pay.

The practical rule that falls out: **self-host the servers that hold consequential
capability, and use hosted ones for read-only breadth.** That aligns the operating
cost with the exposure, and it is the deployment expression of
{{ch:ag-security}}'s partition.

### 7.5 Why this is the same finding as the rest of the book

{{sec:9-practical-example}}'s review result is the fourth appearance of one
pattern, and by now it is worth naming as a general claim rather than a per-chapter
observation.

{{ch:ag-security}} found containment beating detection. {{ch:as-long-running}}
found placed gates beating frequent ones, and automated re-validation scaling where
human gates did not. {{ch:mcp-security}} found a capability partition beating both
detection defences combined. And here, a structural admission filter holds flat
where human review decays to nothing.

The common structure: **a control whose cost is paid once per unit of *design*
scales, and a control whose cost is paid once per unit of *volume* does not** —
because volume is set by someone else. Every one of those four findings is that
sentence in a different setting.

It is also why the recommendations across this book converge on structure rather
than vigilance, which can read as a stylistic preference and is not one. It is what
the measurements keep saying.

### 7.6 Computing your own optimum

{{eq:marginal-server-turns-negative}} needs three numbers, all measurable:

**$\sigma$, the coverage scale.** From a trace: what fraction of tasks could be
completed with your $k$ most-used servers, swept over $k$. Fit the saturation.

**Your schema token count.** {{ch:mcp-schemas}}'s number, which is a sum over
connected servers and takes a minute.

**$h$, the hostile rate.** Not measurable directly, so use the registry's policy as
a proxy and be conservative. A signed-and-scanned registry justifies a much lower
figure than an open one, and {{sec:9-practical-example}} shows the optimum moving
by a factor of two across that range.

Then connect servers in descending order of marginal coverage until the marginal
gain falls below the marginal exposure — which
{{sec:9-practical-example}} puts at around one percent of tasks newly enabled.

**A server that would newly enable under one percent of your tasks is costing more
than it brings.** That is the operational form of this chapter, and it is a
sentence most hosts would fail to satisfy for most of their integrations.

### 7.7 What the ecosystem argument does not settle

Two things this chapter's model leaves out, stated plainly because the numbers
above are otherwise easy to over-read.

**Servers are not interchangeable draws.** The coverage term treats each server as
adding a random slice of capability, which is wrong in a way that favours the
conclusion. Real inventories are heavily overlapping — six servers that all wrap
issue trackers add almost nothing to each other — and occasionally a single server
is the only route to something essential. So the real coverage curve is lumpier
than an exponential, and a host should sequence by *measured* marginal coverage
rather than trusting a fitted $\sigma$.

**Exposure is not uniform across servers.** The model gives every connected server
the same hostile probability, which ignores that a first-party server from the
operator of the service it wraps is a very different risk from an anonymous
wrapper. In practice the right move is not to reduce the count uniformly but to
apply {{sec:7-internal-mechanics}}'s deployment rule: the exposure term should be
weighted by what each server can actually reach, which is
{{ch:ag-security}}'s union rather than a count.

Both corrections push the same way — they say the *count* is a proxy for what
actually matters, which is overlapping capability and reachable consequence. A host
that connects twenty read-only servers from verified publishers is in a much better
position than the model's $n = 20$ row suggests, and one that connects five
anonymous servers holding write capability is in a much worse position than its
$n = 5$ row.

The number worth carrying is therefore not "connect eight servers". It is the
*condition*: **stop when the next server's marginal coverage falls below its
marginal reachable consequence**, and measure both rather than assuming them.

## 8. Implementation

Two listings. The first prices registry admission policy. The second finds the
number of servers a host should connect.

```python {tier=A name=admission-is-a-router}
"""Registry admission policy, which sets the parameters every defence scales on.

ch:mcp-security found that the poisoned fraction and the rate servers turn
hostile are not properties a host can observe or control. They are outcomes of
registry policy: who may publish, what is verified, what provenance survives
(cite:hou2025mcp).

So the interesting question is not "how do I defend against a bad server" but
"what admission policy should a registry have", and that turns out to be a
familiar shape. A strict registry admits fewer bad servers AND fewer good ones,
which is ch:ag-what-is-an-agent's router-versus-agent trade with publishers in
place of requests (eq:admission-is-a-router).

Review capacity is the binding constraint, and a review queue habituates exactly
as ch:ag-termination measured.
"""
import numpy as np

rng = np.random.default_rng(4391)

M = 4000                # ecosystems simulated
SUBMISSIONS = 900       # servers submitted over the period
P_BAD = 0.06            # share of submissions that are hostile or negligent
REVIEWERS = 3
REVIEWS_PER_DAY = 14
DAYS = 30
CATCH_0 = 0.88          # a fresh reviewer's detection rate
HALF = 90               # reviews after which attention has halved


def catch_rate(load):
    """ch:ag-termination's habituation, per reviewer over the period."""
    return CATCH_0 / (1.0 + load / HALF)


def run(policy, m=M, submissions=SUBMISSIONS, p_bad=P_BAD,
        reviewers=REVIEWERS, signed_deters=0.75):
    """Returns (admitted, poisoned share, coverage, reviewer load).

    open        anything published is listed
    signed      publishers must sign with a verified identity, which deters
                some bad actors and turns some away who cannot be bothered
    reviewed    every submission is human-reviewed, capped by capacity
    signed+rev  both
    """
    capacity = reviewers * REVIEWS_PER_DAY * DAYS
    bad = rng.random((m, submissions)) < p_bad
    admitted = np.ones((m, submissions), dtype=bool)
    load = 0.0

    if policy in ("signed", "signed+rev"):
        # Signing deters bad actors more than good ones, but deters some good
        # ones too -- a hobbyist who will not set up an identity.
        admitted &= ~(bad & (rng.random((m, submissions)) < signed_deters))
        admitted &= ~(~bad & (rng.random((m, submissions)) < 0.22))

    if policy in ("reviewed", "signed+rev"):
        # Only `capacity` submissions can be reviewed; the rest queue and are
        # admitted unreviewed, which is what actually happens.
        pending = admitted.sum(1)
        reviewed_frac = np.minimum(capacity / np.maximum(pending, 1), 1.0)
        load = float(np.minimum(pending, capacity).mean()) / reviewers
        cr = catch_rate(load)
        got_reviewed = rng.random((m, submissions)) < reviewed_frac[:, None]
        caught = bad & admitted & got_reviewed & (rng.random((m, submissions)) < cr)
        admitted &= ~caught

    n_adm = admitted.sum(1)
    n_bad = (admitted & bad).sum(1)
    n_good_total = (~bad).sum(1)
    n_good_adm = (admitted & ~bad).sum(1)
    return (float(n_adm.mean()),
            float(np.mean(n_bad / np.maximum(n_adm, 1))),
            float(np.mean(n_good_adm / np.maximum(n_good_total, 1))),
            load)


POLICIES = [("open", "open"), ("signed", "signed identity"),
            ("reviewed", "human review"), ("signed+rev", "signed + review")]

print(f"{SUBMISSIONS} servers submitted, {P_BAD:.0%} of them hostile or")
print(f"negligent. {REVIEWERS} reviewers at {REVIEWS_PER_DAY}/day for {DAYS}")
print(f"days = {REVIEWERS * REVIEWS_PER_DAY * DAYS} reviews of capacity.")
print()
print(f"{'policy':>18}{'admitted':>11}{'poisoned share':>16}"
      f"{'good coverage':>15}")
print("-" * 60)
tab = {}
for key, label in POLICIES:
    r = run(key)
    tab[label] = r
    print(f"{label:>18}{r[0]:>11.0f}{r[1]:>16.2%}{r[2]:>15.1%}")

print()
print()
print("The trade, stated as ch:ag-what-is-an-agent stated it: a strict policy")
print("keeps bad servers out and keeps good ones out too.")
print()
print(f"{'policy':>18}{'bad admitted':>14}{'good REJECTED':>16}"
      f"{'ratio':>8}")
print("-" * 56)
for key, label in POLICIES:
    r = tab[label]
    bad_adm = r[0] * r[1]
    good_rej = SUBMISSIONS * (1 - P_BAD) * (1 - r[2])
    print(f"{label:>18}{bad_adm:>14.1f}{good_rej:>16.1f}"
          f"{good_rej / max(bad_adm, 1e-9):>8.1f}")

print()
print()
print("Review capacity is the constraint, and adding reviewers runs into")
print("ch:ag-termination's habituation rather than scaling cleanly.")
print()
print(f"{'reviewers':>11}{'poisoned share':>16}{'load/reviewer':>15}"
      f"{'catch rate':>12}")
print("-" * 54)
rv = {}
for n in (1, 3, 8, 20, 50):
    r = run("reviewed", reviewers=n)
    rv[n] = r
    print(f"{n:>11}{r[1]:>16.2%}{r[3]:>15.0f}{catch_rate(r[3]):>12.1%}")

print()
print()
print("And how each policy holds as submissions grow, which is what success")
print("does to a registry.")
print()
print(f"{'submissions':>13}{'open':>9}{'signed':>10}{'reviewed':>11}"
      f"{'signed+rev':>13}")
print("-" * 56)
sc = {}
for n in (150, 900, 4000, 15000):
    row = tuple(run(key, submissions=n)[1] for key, _ in POLICIES)
    sc[n] = row
    print(f"{n:>13}" + "".join(f"{v:>{w}.2%}" for v, w in
                               zip(row, (9, 10, 11, 13))))

print(f"""
The first table looks like a case for human review and the last table withdraws
it.

At {SUBMISSIONS} submissions, review takes the poisoned share from
{tab['open'][1]:.2%} to {tab['human review'][1]:.2%}. Signing takes it to
{tab['signed identity'][1]:.2%}, and the two together to
{tab['signed + review'][1]:.2%}.

So signing is already the stronger of the two here. The scale table says why, and
it is the more important result. As submissions grow from {150} to {15000},
review's poisoned share goes {sc[150][2]:.2%} to {sc[15000][2]:.2%} -- back to
almost exactly the open-registry rate -- while signing holds at
{sc[15000][1]:.2%} throughout.

**A per-submission structural filter is scale-invariant and human review is not**
(eq:review-does-not-scale). Review has fixed capacity; submissions do not. Past
the point where the queue exceeds capacity, the marginal submission is admitted
unreviewed, and the policy's stated strictness stops describing what happens.

The reviewer table shows that adding capacity does not rescue it. Going from
{1} to {50} reviewers takes the poisoned share from {rv[1][1]:.2%} to
{rv[50][1]:.2%} -- real, and a factor of {rv[1][1] / rv[50][1]:.1f} for fifty
times the people. The catch-rate column is why: at {1} reviewer the load is
{rv[1][3]:.0f} reviews and the catch rate {catch_rate(rv[1][3]):.1%}.

That is ch:ag-termination's habituation in a queue rather than an approval
dialogue, and it is the third setting in this book where the same curve decides
the answer. **A review process whose volume is set by someone else's submission
rate cannot be staffed out of its problem.**

The second table is the cost, and it is the one registries under-report. Signing
blocks {tab['signed identity'][0] * tab['signed identity'][1]:.0f} bad servers and
rejects {SUBMISSIONS * (1 - P_BAD) * (1 - tab['signed identity'][2]):.0f} good
ones -- about {SUBMISSIONS * (1 - P_BAD) * (1 - tab['signed identity'][2]) / max(tab['signed identity'][0] * tab['signed identity'][1], 1e-9):.0f}
good servers turned away per bad one blocked.

Whether that is a good trade depends on something outside this listing: whether
the rejected servers are replaceable. **This is ch:ag-what-is-an-agent's
router-versus-agent decision with publishers in place of requests**
(eq:admission-is-a-router). A strict registry is a router: it handles the head of
the publisher distribution safely and turns the tail away. An open one is an
agent: it covers everything and admits what it cannot check.

And as there, the answer is set by tail mass. If the servers deterred by an
identity requirement are hobbyist duplicates of things already listed, the
coverage loss is nominal. If they are the only integration to some niche system,
the registry has traded away exactly the long tail that made an ecosystem worth
having.

The practical reading for a registry operator: **prefer filters that cost the
publisher effort over filters that cost you attention**, because the first scales
with submissions and the second does not -- and then measure the coverage you are
losing, which is the number nobody publishes.""")
```

The second listing asks how many servers to connect.

```python {tier=A name=marginal-server-turns-negative}
"""How many servers should a host connect? The part's closing question.

Every chapter in part:19 measured one cost of connecting a server:

  ch:mcp-schemas    schema rent -- tokens spent on every request, linear in tools
  ch:mcp-security   exposure -- more servers, more chances one is hostile
  ch:mcp-architecture  correlation -- a shared server's outage hits every client
  ch:mcp-why        integration -- amortised, and the one that gets cheaper

Against those, capability: a new server can do something the others cannot. That
benefit SATURATES, because the tenth issue-tracker integration adds less than the
first (eq:marginal-server-turns-negative).

Saturating benefit against linear costs has an interior optimum, and this listing
finds it. Nobody computes this; hosts connect servers until something feels wrong.
"""
import numpy as np

rng = np.random.default_rng(4423)

M = 30000
STEPS = 6
TOOLS_PER_SERVER = 9
TOKENS_PER_TOOL = 170
DILUTE = 2.0e-6         # per-token degradation, as in ch:mcp-schemas
BASE = 0.995
P_HOSTILE = 0.02        # ch:mcp-security, set by registry policy
P_OBEY = 0.55
COVER_SCALE = 5.5       # servers at which ~63% of needed capability is covered


def coverage(n_servers, scale=COVER_SCALE):
    """Saturating: each server adds less unique capability than the last."""
    return 1.0 - np.exp(-n_servers / scale)


def run(n_servers, m=M, steps=STEPS, retrieval=None, p_hostile=P_HOSTILE,
        dilute=DILUTE, cover_scale=COVER_SCALE):
    """`retrieval` caps how many tool schemas reach context, per ch:mcp-schemas."""
    tools = n_servers * TOOLS_PER_SERVER
    shown = tools if retrieval is None else min(retrieval, tools)
    tokens = shown * TOKENS_PER_TOOL

    # Capability: does the host have a server that can do what the task needs?
    have = rng.random(m) < coverage(n_servers, cover_scale)
    # Rent: everything in context competes with everything else.
    p_step = BASE * (1.0 - dilute * tokens)
    reason_ok = rng.random(m) < np.clip(p_step, 0.0, 1.0) ** steps
    # Exposure: at least one connected server is hostile, and the model obeys.
    hostile = rng.random(m) < (1.0 - (1.0 - p_hostile) ** n_servers)
    harmed = hostile & (rng.random(m) < P_OBEY)

    ok = have & reason_ok & ~harmed
    return (float(ok.mean()), tokens, float(coverage(n_servers, cover_scale)),
            float(harmed.mean()))


print(f"{M:,} tasks. Each server brings {TOOLS_PER_SERVER} tools at about")
print(f"{TOKENS_PER_TOOL} tokens of schema, adds capability with diminishing")
print(f"returns, and is hostile with probability {P_HOSTILE:.0%}.")
print()
print(f"{'servers':>9}{'coverage':>11}{'schema tokens':>15}{'harm rate':>11}"
      f"{'success':>10}")
print("-" * 56)
tab = {}
for n in (1, 2, 4, 8, 16, 32):
    r = run(n)
    tab[n] = r
    print(f"{n:>9}{r[2]:>11.1%}{r[1]:>15,.0f}{r[3]:>11.2%}{r[0]:>10.1%}")

peak = max(tab, key=lambda k: tab[k][0])

print()
print()
print("The same sweep with ch:mcp-schemas' retrieval layer capping how many")
print("schemas reach the context at 24.")
print()
print(f"{'servers':>9}{'no retrieval':>14}{'retrieval 24':>14}{'gain':>9}")
print("-" * 46)
rt = {}
for n in (1, 2, 4, 8, 16, 32, 64):
    a = run(n)[0]
    b = run(n, retrieval=24)[0]
    rt[n] = (a, b)
    print(f"{n:>9}{a:>14.1%}{b:>14.1%}{b - a:>+9.1%}")

peak_rt = max(rt, key=lambda k: rt[k][1])

print()
print()
print("Which cost binds depends on registry policy, and bh1 showed that is")
print("chosen rather than given. Best server count under each:")
print()
print(f"{'hostile rate':>14}{'best n':>9}{'success there':>15}"
      f"{'success at n=32':>17}")
print("-" * 55)
hp = {}
for p in (0.002, 0.02, 0.06):
    vals = {n: run(n, retrieval=24, p_hostile=p)[0]
            for n in (1, 2, 4, 8, 16, 32, 64)}
    b = max(vals, key=lambda k: vals[k])
    hp[p] = (b, vals[b], vals[32])
    print(f"{p:>14.1%}{b:>9}{vals[b]:>15.1%}{vals[32]:>17.1%}")

print()
print()
print("And with how broad the task distribution is -- a host serving varied")
print("work needs more coverage before saturation sets in.")
print()
print(f"{'coverage scale':>16}{'best n':>9}{'success there':>15}")
print("-" * 40)
cs = {}
for s in (2.0, 5.5, 14.0):
    vals = {n: run(n, retrieval=24, cover_scale=s)[0]
            for n in (1, 2, 4, 8, 16, 32, 64)}
    b = max(vals, key=lambda k: vals[k])
    cs[s] = (b, vals[b])
    print(f"{s:>16.1f}{b:>9}{vals[b]:>15.1%}")

print()
print()
print("Decomposing the marginal server at the optimum: what the next one adds")
print("and what it costs.")
print()
print(f"{'from n to n+1':>15}{'coverage gain':>15}{'added tokens':>14}"
      f"{'added harm':>12}{'net':>9}")
print("-" * 65)
mg = {}
for n in (1, 2, 4, 8, 16, 32):
    a = run(n, retrieval=24)
    b = run(n + 1, retrieval=24)
    mg[n] = (b[2] - a[2], b[1] - a[1], b[3] - a[3], b[0] - a[0])
    print(f"{f'{n} -> {n + 1}':>15}{b[2] - a[2]:>+15.1%}{b[1] - a[1]:>+14,.0f}"
          f"{b[3] - a[3]:>+12.2%}{b[0] - a[0]:>+9.1%}")

print(f"""
The first table has a peak in it, which is the answer to a question hosts do not
usually ask.

Success rises to {tab[peak][0]:.1%} at {peak} servers and falls to
{tab[32][0]:.1%} at {32}. Coverage is still climbing -- {tab[32][2]:.1%} at
thirty-two -- and it climbs into two costs that do not stop: schema rent at
{tab[32][1]:,.0f} tokens, and a {tab[32][3]:.1%} chance that one of the connected
servers is hostile and obeyed.

**Saturating benefit against linear costs has an interior optimum**
(eq:marginal-server-turns-negative), and connecting past it makes the host worse
at everything, not just more expensive.

The second table shows that one of those two costs is removable.
ch:mcp-schemas' retrieval layer, capping the schemas that reach context at
{24}, is worth {rt[8][1] - rt[8][0]:+.1%} at {8} servers and
{rt[64][1] - rt[64][0]:+.1%} at {64} -- and it moves the optimum from
{peak} servers to {peak_rt}.

**Retrieval does not merely improve a large inventory; it changes how many
servers it is rational to connect.** A host without one is choosing between
capability and context on every integration decision, and a host with one is only
paying the security cost.

The third table connects this listing to the previous one. At a
{0.002:.1%} hostile rate the best server count is {hp[0.002][0]}; at
{0.06:.0%} it is {hp[0.06][0]}, and connecting {32} costs
{hp[0.06][1] - hp[0.06][2]:.1f} points against the optimum.

**Registry policy sets how many servers a host can afford to connect.** That is
the strongest argument for the admission work in the previous listing, and it is
not the argument usually made for it: a well-governed registry is not merely
safer, it lets every host that uses it be more capable, because the marginal
server stays positive for longer.

The fourth table says the optimum is also a property of the host's own workload.
Narrow, repetitive work saturates at {cs[2.0][0]} servers; varied work at
{cs[14.0][0]}. So there is no ecosystem-wide right answer, and a host should
compute this from its own task distribution rather than copying a number.

The last table is the optimality condition made visible, and it is the most
useful thing here. Going from {16} to {17} servers adds
{mg[16][0]:+.1%} of coverage and {mg[16][2]:+.2%} of harm exposure, for a net of
{mg[16][3]:+.1%}. Going from {8} to {9} adds {mg[8][0]:+.1%} coverage against
{mg[8][2]:+.2%} harm, for {mg[8][3]:+.1%}.

**The marginal server turns negative when the capability it adds falls below the
exposure it adds**, and because the first shrinks while the second is roughly
constant, that crossing always happens. The only question is where.

Which gives the part's closing instruction, and it is one nobody follows.
Before connecting a server, ask what fraction of your tasks it would newly enable.
If the answer is under about one percent, it is costing more than it brings --
and the honest version of "we support fifty integrations" is that most of them are
making the other forty-nine work slightly worse.""")
```

## 9. Practical Example

The first listing runs $900$ submissions, $6\%$ bad, against three reviewers:

```
            policy   admitted  poisoned share  good coverage
------------------------------------------------------------
              open        900           6.00%         100.0%
   signed identity        673           2.02%          78.0%
      human review        889           4.84%         100.0%
   signed + review        670           1.52%          78.0%
```

Signing is already stronger than review. The reason is in the scale table:

```
  submissions     open    signed   reviewed   signed+rev
--------------------------------------------------------
          150    5.97%     2.02%      2.74%        0.77%
          900    5.97%     2.00%      4.83%        1.50%
        15000    6.01%     2.00%      5.92%        1.97%
```

**Review's effectiveness decays back to the open-registry rate as the ecosystem
grows; signing holds flat** ({{eq:review-does-not-scale}}). Capacity is fixed and
submissions are not, so past the crossover the marginal submission is admitted
unreviewed and the stated policy stops describing what happens.

Staffing does not fix it:

```
  reviewers  poisoned share  load/reviewer  catch rate
------------------------------------------------------
          1           5.59%            420       15.5%
          8           3.72%            112       39.1%
         50           1.69%             18       73.3%
```

Fifty times the reviewers for a factor of $3.3$, because
{{eq:habituation}} applies to a queue exactly as it applied to an approval
dialogue.

And the cost nobody reports:

```
            policy  bad admitted   good REJECTED   ratio
--------------------------------------------------------
   signed identity          13.6           186.4    13.7
```

About fourteen good servers turned away per bad one blocked
({{eq:admission-is-a-router}}). Whether that is a good trade depends on whether
the rejected servers were replaceable — {{ch:ag-what-is-an-agent}}'s tail-mass
question, asked about publishers.

The second listing sweeps how many servers a host connects:

```
  servers   coverage  schema tokens  harm rate   success
--------------------------------------------------------
        1      16.6%          1,530      1.05%     15.7%
        4      51.7%          6,120      4.36%     44.4%
        8      76.6%         12,240      8.22%     59.4%
       16      94.5%         24,480     15.28%     57.7%
       32      99.7%         48,960     26.21%     38.5%
```

Coverage is still climbing at thirty-two and success is falling.
**Saturating benefit against linear costs has an interior optimum**
({{eq:marginal-server-turns-negative}}).

With {{ch:mcp-schemas}}'s retrieval layer capping schemas at $24$:

```
  servers  no retrieval  retrieval 24     gain
----------------------------------------------
        8         58.5%         64.8%    +6.3%
       16         57.5%         74.1%   +16.6%
       64         15.8%         55.0%   +39.2%
```

**Retrieval moves the optimum from eight servers to sixteen**
({{eq:retrieval-moves-the-optimum}}) — it changes how many servers it is rational
to connect, not just how well a large inventory performs.

Registry policy moves it too:

```
  hostile rate   best n  success there  success at n=32
-------------------------------------------------------
          0.2%       32          88.8%            88.8%
          6.0%       16          57.0%            48.8%
```

**A well-governed registry makes hosts more capable, not just safer**
({{eq:registry-policy-sets-host-capability}}) — which is the argument for the first
listing's work that is usually left out.

And the optimality condition, visible:

```
  from n to n+1  coverage gain  added tokens  added harm      net
-----------------------------------------------------------------
         2 -> 3         +11.6%        +1,020      +1.06%    +9.5%
         8 -> 9          +3.9%            +0      +1.10%    +2.6%
       16 -> 17          +0.9%            +0      +0.84%    -0.4%
```

**The marginal server turns negative when the capability it adds falls below the
exposure it adds**, and since the first shrinks while the second is roughly
constant, that crossing always happens.

## 10. Production Considerations

For a registry: prefer filters whose cost falls on the publisher over filters whose
cost falls on your review queue. The first scales with submissions; the second
converges to no policy at all.

Publish provenance, the aggregated tool audit, the schema token count, and
definition history. All are computable once for everyone and none are published
today.

Measure the coverage you are losing to strictness. It is the other half of
{{eq:admission-is-a-router}} and nobody reports it.

Adopt a deprecation floor. {{cite:mcp2026spec}}'s twelve-month policy is a good
model and it is what {{ch:mcp-why}}'s connectivity result asks for.

For a host: compute $\sigma$ from your own traces, connect servers in descending
marginal coverage, and stop when the marginal server would newly enable under about
one percent of tasks.

Build the retrieval layer before connecting the ninth server, not after the
thirtieth — it moves the optimum rather than merely raising it.

Self-host the servers holding consequential capability; use hosted ones for
read-only breadth.

And treat the registry you depend on as a parameter of your own system, because
{{eq:registry-policy-sets-host-capability}} says it is one.

## 11. Common Mistakes

**Staffing a review queue to fix admission quality.** Sublinear returns against an
exogenous submission rate.

**Reporting a strict policy that the queue does not implement.** The `queue full`
branch is where stated and actual policy diverge.

**Not measuring coverage loss.** Strictness has a cost that no registry publishes.

**Connecting servers until something feels wrong.** The optimum is computable and
is passed well before anything feels wrong.

**Adding a retrieval layer late.** It changes the optimum, so adding it late means
having been at the wrong optimum meanwhile.

**Treating registry choice as procurement.** It sets a parameter in your own
capability calculation.

**Assuming more integrations is a better product.** Past $n^*$, each one makes the
others work slightly worse.

## 12. Failure Modes

*Silent policy drift.* A registry whose review queue overflowed and now admits
most submissions unreviewed, while its documentation still describes review.

*Tail collapse.* A strict registry that blocked the niche integrations that made
the ecosystem worth adopting.

*Integration bloat.* A host past $n^*$, with a marketing page listing fifty
integrations and a success rate lower than at eight.

*Version islands.* A deprecation without a migration floor, splitting the
ecosystem.

*Supply-chain lag.* A third-party wrapper trailing the API it wraps, producing
{{ch:as-long-running}}'s drift as a dependency property.

## 13. Alternatives

**A curated allowlist per organisation.** Sets $h$ to nearly zero for that
organisation at the cost of maintaining the list — usually the right answer inside
an enterprise, and it is a router with a known and accepted tail loss.

**Reputation rather than admission.** Admit everything, rank by observed behaviour.
Moves the filter from publication time to usage time, and needs a feedback signal
nobody currently collects.

**Attestation of running artefacts.** Verify at connection time that the server is
the reviewed build, which addresses {{eq:approval-is-a-snapshot}} at the ecosystem
level rather than per host.

**No registry.** Configuration by explicit URL, which is where most production
deployments actually are and which is an allowlist by another name.

## 14. Evaluation

For a registry: report the admitted poisoned share *and* the good-submission
rejection rate. One without the other describes half a policy.

Report the fraction of submissions actually reviewed against the fraction the
policy claims. The gap is the real policy.

For a host: measure $\sigma$ by ablating servers and re-running a task sample. It
is the only parameter in {{eq:marginal-server-turns-negative}} with any shape.

Track schema tokens and connected-server count as operational metrics.

Ablate individual servers periodically and check whether removing one costs
anything. Past $n^*$, several will not.

And re-compute the optimum when you add a retrieval layer or change registries,
since both move it.

## 15. Advanced Concepts

**Reputation-weighted admission.** Combining a low-cost structural filter with
usage-derived reputation would get scale-invariance and tail coverage together, and
needs a behaviour signal registries do not collect. {{maturity:EMERGING}}.

**Attested builds.** Verifying at connection that the running server matches the
reviewed artefact closes the rug-pull gap ecosystem-wide rather than per host.
{{maturity:EMERGING}}.

**Registry-published capability profiles.** Enough metadata for a host to compute
marginal coverage *before* connecting, which is what
{{eq:marginal-server-turns-negative}} needs and what nothing supplies.

**Measuring $\pi$ and $\sigma$ in the wild.** Both parameters govern every result
in this chapter and neither has a published estimate for any real MCP ecosystem.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-what-is-an-agent}}'s router-versus-agent trade turns out to describe
registry admission as well as request handling, with publishers on the input and
tail mass deciding.

{{ch:ag-termination}}'s habituation appears in a review queue — its fourth setting
in this book, and the one where the consequences are ecosystem-wide.

{{ch:mcp-schemas}}'s retrieval layer changes the optimum here rather than merely
improving throughput, which is a stronger claim than that chapter made for it.

{{ch:mcp-security}}'s handoff — that $\pi$ and $\lambda$ are policy outcomes — is
answered here, and answered with the observation that governance is a capability
question as much as a safety one.

{{ch:ag-security}}, {{ch:as-long-running}} and {{ch:mcp-security}} together with
this chapter make one claim four times: **controls priced per unit of design
scale, and controls priced per unit of volume do not.**

Ahead: {{part:20}} leaves the protocol layer for the systems that consume it.

## 17. Exercises

1. Derive $n^*$ from {{eq:marginal-server-turns-negative}} and evaluate it for your
   own measured $\sigma$ and schema token count.

2. Add reputation to the first listing — admit everything, but weight by observed
   behaviour — and compare against signing on both poisoned share and coverage.

3. Model a review queue with a backlog that carries between periods. Does the
   asymptote change?

4. Make coverage heterogeneous across tasks and check whether a single $n^*$ still
   exists.

5. Add {{ch:mcp-architecture}}'s correlation term to the second listing and see
   how much it moves the optimum.

6. Estimate the coverage loss from a signing requirement for a real registry, using
   its publisher distribution.

## 18. Interview Questions

1. Why does human review of registry submissions stop working as an ecosystem
   grows?

2. Your registry blocks fourteen good servers per bad one. Is that a good policy?

3. How many MCP servers should a host connect?

4. Why does adding a retrieval layer change that number rather than just improving
   things?

5. What does registry governance have to do with host capability?

6. You support fifty integrations. What would you check?

## 19. Research Questions

1. What is $\pi$ in a real MCP registry, and how does it vary with admission
   policy?

2. Can reputation signals be collected without a central observer of usage?

3. Would attested builds be adopted, given they constrain legitimate updates?

4. What metadata would let a host compute marginal coverage before connecting?

5. Does the coverage-scale parameter $\sigma$ vary enough across hosts to make a
   published optimum meaningless?

## 20. Chapter Summary

{{ch:mcp-security}} left two parameters unexplained: the poisoned fraction and the
hostile-turn rate, neither observable from inside a session. Both are registry
policy outcomes, and this chapter asks what that policy should be.

**Admission is a router-versus-agent decision** ({{eq:admission-is-a-router}}) with
publishers on the input: a signing requirement blocked about $40$ bad servers and
turned away $186$ good ones, roughly fourteen good per bad. Whether that is right
depends on {{ch:ag-what-is-an-agent}}'s tail-mass question.

Which strict policy matters more than how strict. A signing requirement held the
poisoned share at $2.00\%$ from $150$ submissions to $15{,}000$; human review went
from $2.74\%$ to $5.92\%$ — back to the open-registry rate.
**Review does not scale** ({{eq:review-does-not-scale}}), because capacity is fixed
while submissions are not, and fifty times the reviewers bought a factor of $3.3$
against {{eq:habituation}}.

For hosts, capability saturates while schema rent and hostile exposure do not, so
there is an interior optimum ({{eq:marginal-server-turns-negative}}): success
peaked at eight servers and fell from $59.4\%$ to $38.5\%$ by thirty-two. At the
optimum the condition is visible — the seventeenth server adds $0.9\%$ coverage
against $0.84\%$ exposure. **A server that would newly enable under about one
percent of your tasks costs more than it brings.**

Two things move that number. {{ch:mcp-schemas}}'s retrieval layer moved it from
eight to sixteen and was worth $+39.2$ points at sixty-four
({{eq:retrieval-moves-the-optimum}}). And registry quality moved it from sixteen at
a $6\%$ hostile rate to thirty-two at $0.2\%$ — **a well-governed registry makes
hosts more capable, not merely safer** ({{eq:registry-policy-sets-host-capability}}).

And a pattern this chapter completes. {{ch:ag-security}}'s containment over
detection, {{ch:as-long-running}}'s automated re-validation over human gates,
{{ch:mcp-security}}'s partition over scanners, and now a structural filter over a
review queue: **controls priced per unit of design scale, and controls priced per
unit of volume do not**, because volume is set by someone else.

## 21. Further Reading

{{cite:hou2025mcp}} for the four-phase lifecycle this chapter operationalises, and
its finding that most threats land before any protocol message — which is the
argument for treating admission as the primary control.

{{cite:huang2026mcpthreat}} for the static-scanning layer that belongs in an
admission pipeline, and {{cite:gaire2025mcpsok}} for the separation of adversarial
from non-adversarial failures that a registry policy has to address differently.

{{cite:mcp2026spec}}'s versioning and feature-lifecycle pages for the deprecation
discipline {{sec:7-internal-mechanics}} recommends.

{{cite:qin2023toolllm}} for the inventory scale at which
{{eq:marginal-server-turns-negative}} stops being a curiosity and becomes the
binding design constraint.
