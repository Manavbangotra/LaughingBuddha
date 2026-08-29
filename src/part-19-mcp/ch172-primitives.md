---
id: mcp-primitives
number: 172
part: XIX
tier: full
status: draft
requires: [distinctness-not-count, transport-decides-correlation,
           revalidation-is-cheapest, retry-needs-a-verifier]
provides: [decision-cost-versus-dilution, resources-go-stale,
           prediction-reduces-preloading, resource-when-selection-is-weak,
           primitive-is-a-controller-choice]
citations: [mcp2026spec, qin2023toolllm, li2023apibank, greshake2023indirect,
            huang2026mcpthreat]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what actually distinguishes
MCP's three primitives — which is not the kind of data they carry; explain why a
tool call is a decision and what decisions cost; locate the interior optimum
between decision cost and context dilution; explain why better demand prediction
means *less* preloading rather than more; and apply the freshness threshold that
decides the primitive choice before any other consideration does.

## 2. Why This Matters

{{cite:mcp2026spec}} gives servers three things to offer: **Tools**, **Resources**
and **Prompts**. Server authors routinely expose everything as a tool, because a
tool can do anything, and the result is a server that works less well than it
should for a reason that never appears in a log.

The split is not by data type. It is by **who decides to use the thing**
({{eq:primitive-is-a-controller-choice}}): a tool is invoked by the *model*, a
resource is included by the *host*, a prompt is invoked by the *user*.

That makes the tools-versus-resources choice a question about where a decision
lives — and {{ch:ag-tool-calling}} already measured what decisions cost. Every
tool call is a selection, selections are imperfect, and the damaging case is not a
call that fails but one that plausibly returns the wrong thing and is not noticed.
{{sec:9-practical-example}} finds that costing $21.9$ points when everything is
fetched.

Resources remove the decision and charge for it differently: everything in context
dilutes everything else, which {{ch:ag-memory}} measured directly. So the two
primitives are expensive in opposite directions and there is an interior optimum
({{eq:decision-cost-versus-dilution}}).

Two results from locating it are worth the chapter on their own. **The better a
host can predict what is needed, the *less* it should preload** — the optimum
moves from $100\%$ to $25\%$ as demand concentrates
({{eq:prediction-reduces-preloading}}), because prediction makes a small resource
set do a large one's work. And **the worse the model is at choosing, the more you
should choose for it** — the optimum moves from $0\%$ to $100\%$ as selection
reliability falls from $98\%$ to $55\%$
({{eq:resource-when-selection-is-weak}}).

Then the second listing adds the axis that overrides both. A resource is read once
when the host assembles context; a tool is read when it is used. So resources go
stale ({{eq:resources-go-stale}}), and above roughly $1\%$ change per step the
question stops being interesting: preloading everything falls from $100\%$ to
$22.5\%$ while fetching everything does not move.

## 3. Prerequisites

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}} — selection reliability is
one of the two numbers that decides the split.

{{ch:ag-memory}}'s dilution result, which is the cost resources pay.

{{ch:as-long-running}}'s {{eq:revalidation-is-cheapest}}, whose single-turn
version this chapter measures.

{{ch:as-specialized}}'s {{eq:retry-needs-a-verifier}}, which explains why a wrong
fetch is worse than a failed one.

{{ch:mcp-architecture}} for the roles, since "the host decides" is only meaningful
once host, client and server are distinct.

## 4. Intuitive Explanation

A server author wants to expose a project's documentation to an agent. There are
three ways, and the protocol's three primitives correspond exactly to them.

**A tool**: `search_docs(query)`. The model decides when to call it and with what.

**A resource**: the documentation is addressable content the host can include, and
the host decides whether this turn needs it.

**A prompt**: a templated workflow — "review this file against the style guide" —
that the user invokes, typically from a menu or a slash command.

Notice that the *data* is the same in all three. What differs is who initiates.
That is the distinction, and it is why "expose everything as tools" is a design
error rather than a shortcut: it moves every decision to the model, including the
ones the host or the user could have made correctly and for free.

Because a tool call is a decision, and decisions have an error rate.
{{ch:ag-tool-calling}} found selection robust to inventory *size* and fragile to
inventory *overlap*, and any real server has overlap — `search_docs`,
`get_readme`, `list_files` and `read_file` are four ways to get at the same
content.

The failure that costs is subtle. A call that errors gets retried, and
{{ch:ag-tool-calling}} showed a good error message makes that retry cheap. The
expensive case is a call that *succeeds and returns the wrong thing*: the model
asked for the wrong document, got a real document, and has no way to tell.
{{sec:9-practical-example}} models that at $45\%$ of selection errors and finds it
dominating the decision cost.

A resource removes the decision entirely. The host includes the content; there is
nothing to get wrong.

But context is not free. {{ch:ag-memory}} found that extending a context window
made recall of recent facts *worse* — everything present competes with everything
else. So preloading is a real cost, paid on every turn, whether or not the content
was needed.

Two costs pulling opposite ways means an optimum in the middle, and where it sits
depends on two things you can measure.

The first is how well the host can predict what a turn will need — and the
direction here is backwards from intuition. When demand is concentrated,
preloading gets *more efficient*, so a smaller preloaded set captures most of the
benefit and anything beyond it is pure dilution. **Better prediction means less
preloading**, not more.

The second is how good the model is at choosing, which is
{{ch:ag-tool-calling}}'s number. If it selects well, let it — resources only add
dilution. If it selects badly, decide for it.

And then there is the consideration that overrides both, which the
tools-versus-resources discussion usually omits: **a resource is a snapshot.** The
host read it when it assembled the turn; the model uses it eight steps later. If
the underlying thing changed in between, the model is reasoning about a world that
no longer exists, and nothing errors. That is {{ch:as-long-running}}'s silent
drift compressed from days into one turn, and {{sec:9-practical-example}} finds it
deciding the whole question above about one percent change per step.

## 5. Formal Explanation

Write the three primitives as a map from capability to *initiator*:

$$	ext{Tool} \mapsto 	ext{model}, \qquad 	ext{Resource} \mapsto 	ext{host}, \qquad 	ext{Prompt} \mapsto 	ext{user}$$ (eq:primitive-is-a-controller-choice)

Each initiator has a different information set and a different error rate. The
model knows what its reasoning currently requires and selects imperfectly; the
host knows the task and the history and cannot see the reasoning; the user knows
the intent exactly and must act deliberately. **Choosing a primitive is choosing
which of those three does the deciding**, and the rest of this section prices that
choice for the two automatic cases.

Let a task need $k$ items drawn from an inventory of $n$, with a fraction $\phi$
preloaded as resources and the rest fetched as tools.

**Decision cost.** Each fetch selects correctly with probability $s$. A wrong
selection is noticed with probability $\nu$ and retried; otherwise the task
proceeds on wrong content and fails. Per fetch:

$$p_{\text{fetch}} = s + (1-s)\nu s' , \qquad p_{\text{poison}} = (1-s)(1-\nu)$$

Over the $k(1-\phi)$ expected fetches, the surviving probability is
$\big(1 - p_{\text{poison}}\big)^{k(1-\phi)}$ — **decreasing in $(1-\phi)$**.

**Dilution cost.** With $r = \phi n$ items in context, per-step success degrades:

$$p_{\text{step}}(r) = p_0\big(1 - \delta \max(r - k, 0)\big)$$

so the reasoning survives with $p_{\text{step}}(r)^k$ — **decreasing in $\phi$**.
Total success is the product:

$$S(\phi) = \big(1 - (1-s)(1-\nu)\big)^{k(1-\phi)} \cdot p_{\text{step}}(\phi n)^{k}$$ (eq:decision-cost-versus-dilution)

The first factor increases in $\phi$ and the second decreases, so $\log S$ is a
sum of a linear increasing term and a concave decreasing one, giving an interior
maximum whenever both are active.

**Prediction.** If demand is skewed with concentration $\kappa$, the coverage of
the top $\phi n$ items is $\phi^{1/\kappa}$ rather than $\phi$. The benefit term
becomes $\big(1-(1-s)(1-\nu)\big)^{k(1-\phi^{1/\kappa})}$ while the dilution term
is unchanged. Differentiating:

$$\frac{\partial \phi^*}{\partial \kappa} < 0$$ (eq:prediction-reduces-preloading)

**Concentration lowers the optimal preload fraction**, because it raises the
benefit's *derivative near zero* without changing the cost — so the marginal
resource stops paying sooner.

**Selection.** Differentiating $S$ with respect to $\phi$ and noting the benefit
term scales as $(1-s)(1-\nu)$:

$$\frac{\partial \phi^*}{\partial s} < 0$$ (eq:resource-when-selection-is-weak)

**A resource is what you build when the model cannot reliably choose the tool.**

**Freshness.** A preloaded item read at step $0$ and used at step $a$, with
per-step change probability $v$, is stale with probability $1 - (1-v)^a$. Over $k$
preloaded items used at mean step $\bar{a}$:

$$S_{\text{fresh}}(\phi) = S(\phi)\cdot\big((1-v)^{\bar{a}}\big)^{k\phi}$$ (eq:resources-go-stale)

This factor is $1$ at $v=0$ and falls exponentially in $v\bar{a}k\phi$. Since a
fetched item has no such factor, there is a threshold volatility above which
$\phi^* = 0$ regardless of $s$ and $\kappa$:

$$v^* \approx \frac{-\log\big(1-(1-s)(1-\nu)\big)}{\bar{a}}$$

**Freshness decides first**, and the other two considerations only operate below
the threshold.

## 6. Mathematical Foundation

Three extractions.

**The optimum is interior only when both costs are active.** From
{{eq:decision-cost-versus-dilution}}: if selection is perfect the benefit term
vanishes and $\phi^* = 0$; if context is free the cost term vanishes and
$\phi^* = 1$. Real systems sit between, which is why "always prefer resources" and
"always prefer tools" are both wrong and both defensible from a corner case.

**Prediction acts on the benefit's curvature, not its level.**
{{eq:prediction-reduces-preloading}} is counterintuitive because concentration
feels like it should justify more preloading. It justifies *more effective*
preloading, which is a reason to do less of it — the top decile already carries
the value.

**The freshness factor is exponential in $\bar{a}$, and $\bar{a}$ is growing.**
The mean step at which preloaded content is consumed rises with turn length, and
agentic turns are getting longer. So $v^*$ falls over time: **content that was
safe to preload becomes unsafe without changing at all.**

## 7. Internal Mechanics

### 7.1 The controller distinction, drawn properly

```mermaid {#fig:primitive-control caption="The three primitives differ by who initiates, not by what they carry. The same documentation can be all three."}
flowchart TD
    U[user] -->|invokes| P[Prompt: templated workflow]
    H[host] -->|includes| R[Resource: addressed content]
    LM[model] -->|calls| T[Tool: a function]
    P --> CTX[the turn's context]
    R --> CTX
    T --> CTX
    CTX --> LM
```

Reading this as "tools do things, resources are data" is the common
simplification and it breaks immediately: a read-only `get_document` tool is a
tool, and it is data. The distinction survives because it is about *initiation*.

The practical test for a server author: **could the host know it needs this before
the model asks?** If yes, it is a resource. If the answer depends on the model's
reasoning, it is a tool. If a human would reach for it deliberately, it is a
prompt.

### 7.2 Prompts, which are the least used and the most underrated

Prompts get the least attention of the three and they resolve a problem the other
two cannot.

A prompt is a *tested path*. The user invokes it, so its invocation is not a model
decision and cannot be selected wrongly; its content is authored, so it can encode
the phrasing, the ordering and the constraints that were found to work.

That makes a prompt {{ch:ag-what-is-an-agent}}'s router applied to intent: it
handles the head of the distribution with no decision cost at all, and it does not
attempt the tail. For a workflow performed daily by the same team, that is a very
good trade, and it is available without any of {{sec:9-practical-example}}'s
arithmetic.

The reason prompts are underused is that they require someone to notice a workflow
is repeated, which is an observation about usage rather than about code.

### 7.3 Resource design: addressing and subscription

A resource is identified by a URI, which makes two things possible that a tool
result does not.

**Stable reference.** The same URI names the same thing across turns, so a host
can cache, diff and re-read it — which is what
{{sec:9-practical-example}}'s revalidation depends on.

**Selective inclusion.** The host can enumerate what is available cheaply and
include only what this turn plausibly needs, which is the mechanism
{{eq:prediction-reduces-preloading}} rewards.

The design consequence is that resource URIs should be *stable and meaningful*.
A URI that encodes a query rather than an identity — `search?q=...` — cannot be
cached, diffed or re-validated, and is a tool wearing a resource's clothing.

### 7.4 Why a wrong fetch is worse than a failed one

{{sec:9-practical-example}} models $45\%$ of selection errors going unnoticed, and
that assumption carries most of the decision cost. It is
{{ch:as-specialized}}'s finding applied inside a single call.

A failed fetch is a verified event: something returned an error, the model knows,
{{ch:ag-tool-calling}}'s error-message result applies, and the retry is cheap. A
fetch that returns the wrong document returns a *well-formed, plausible* document.
There is no verifier. The model proceeds.

Which gives a concrete server-design instruction that costs nothing: **make it
possible to tell whether the returned thing is the requested thing.** Echo the
identity in the response — the document's title, its URI, its version — so that a
mis-selection becomes detectable rather than silent. This is the cheapest
available reduction in $(1-\nu)$, and almost no server does it.

### 7.5 Tool descriptions are an instruction channel

{{cite:huang2026mcpthreat}} evaluated seven MCP clients and found **tool
poisoning — malicious instructions embedded in tool metadata — the most prevalent
and impactful client-side vulnerability**, attributed to insufficient static
validation and parameter visibility.

That is a security finding and it is also a design finding, because it follows
from something true of every server: a tool's description is text that reaches the
model, so it is an instruction whether or not it was written as one. The same
applies to resource contents, which is {{cite:greshake2023indirect}}'s vector.

{{cite:mcp2026spec}} says as much in its own security principles: tool annotations
"should be considered untrusted, unless obtained from a trusted server."
{{ch:mcp-security}} takes this up properly; the point here is that the primitive
choice determines *how much* untrusted text reaches the model by default, and
resources — included without a decision — are the largest such channel.

### 7.6 The combined rule

Putting both listings together:

**Above about $1\%$ per-step volatility, use tools.** Freshness dominates and
nothing else matters.

**Below it, set the preload fraction from selection reliability and demand skew.**
Weak selection pushes toward resources; concentrated demand pushes toward a small,
well-chosen resource set rather than a large one.

**Revalidate whatever you do preload**, at least once mid-turn. It was worth
$+28.7$ points at high volatility for one extra read, and it is the same
intervention {{ch:as-long-running}} found cheapest at a scale a thousand times
larger.

**And use prompts wherever a workflow repeats**, because a user-initiated path has
no decision cost at all.

### 7.7 What this means for a server author

The instruction that falls out is unusual: **a good server exposes the same
underlying capability through more than one primitive**, and lets the host choose.

Documentation as a resource for hosts that can predict the need, as a search tool
for hosts that cannot, and as a prompt for the review workflow that uses it every
day. That looks like duplication and it is not — it is offering the same content
at three different points on {{eq:decision-cost-versus-dilution}}'s curve, so that
a host with a different selection reliability or a different turn length can sit
where it should.

Servers that expose one primitive force every host to the same operating point,
and {{sec:9-practical-example}} shows that point being wrong by fifty-six points
in the worst case.

### 7.8 The failure each initiator is prone to

{{eq:primitive-is-a-controller-choice}} says the three primitives differ by who
decides, and each decider fails differently. Naming the three failures is more
useful than naming the three primitives, because it is what tells you which one
you are looking at in an incident.

**The model decides wrongly** — it selects a plausible wrong tool, gets a
well-formed answer to the wrong question, and continues. This is the tool failure,
and {{sec:9-practical-example}} finds it to be the larger part of decision cost.
Its signature is a confidently wrong answer where the trace shows a successful
call.

**The host decides wrongly** — it includes content the turn did not need
(dilution), omits content it did (a fetch that now has to happen anyway), or
includes content that has since changed (staleness). The resource failures are all
failures of prediction, and their common signature is that nothing in the trace
looks wrong at all, because a resource is not an event.

**The user decides wrongly** — invoking a prompt for a situation it was not
written for. This is the least damaging of the three because it is the most
visible: the person who made the choice is present, sees the result, and can tell
that the workflow did not fit.

That visibility ordering is worth carrying, because it runs opposite to the
autonomy ordering. The most automatic decider fails most silently, and the least
automatic fails most obviously. So moving a capability from prompt to resource to
tool buys convenience and sells detectability — which is
{{ch:as-specialized}}'s verifier argument again, arriving as a statement about
interface design rather than about domains.

It also suggests where to put effort when something is going wrong and you cannot
tell what. Resource failures are the ones with no trace, so a system whose errors
are invisible should suspect its resources first.

## 8. Implementation

Two listings. The first trades decision cost against dilution. The second adds
freshness, which turns out to decide the question first.

```python {tier=A name=decision-cost-versus-dilution}
"""Why there are three primitives instead of one.

cite:mcp2026spec defines Tools, Resources and Prompts, and the split is not by
data type. It is by WHO DECIDES to use the thing:

  Tools      the MODEL decides, by emitting a call
  Resources  the HOST decides, by including content
  Prompts    the USER decides, by invoking a workflow

The common server-design error is to expose everything as a tool, on the grounds
that a tool can do anything. It can, and every tool call is a DECISION -- and
ch:ag-tool-calling measured what decisions cost when the inventory is not
perfectly distinct.

A resource removes the decision by putting the content in context directly. That
is not free either: ch:ag-memory found that everything in context dilutes
everything else. So there is a real trade, and this listing locates it
(eq:decision-cost-versus-dilution).
"""
import numpy as np

rng = np.random.default_rng(4127)

M = 50000
N_ITEMS = 40            # distinct pieces of context a task might need
NEEDED = 5              # how many a given task actually needs
P_SELECT = 0.90         # chance the model picks the right item to fetch
P_NOTICE = 0.55         # chance a wrong fetch is RECOGNISED as wrong
BASE = 0.995            # per-step success with everything present and undiluted


def dilution(loaded):
    """ch:ag-memory's effect: recall of any one item degrades as context grows.
    Modelled as a gentle decay in the per-step success rate."""
    return BASE * (1.0 - 0.0009 * max(loaded - NEEDED, 0))


def run(preload_frac, m=M, n_items=N_ITEMS, needed=NEEDED,
        p_select=P_SELECT, hit_rate=None):
    """`preload_frac` of the inventory is included as resources; the rest must be
    fetched by tool call. A task needs `needed` items. Items that were preloaded
    are simply present; items that were not cost a call that may pick wrong."""
    n_pre = int(round(preload_frac * n_items))
    # Which items a task needs, and whether each happened to be preloaded.
    # Preloading is prioritised by how often an item is needed, which is what a
    # host would actually do.
    if hit_rate is None:
        # Uniform demand: a preloaded item is useful with probability n_pre/n.
        pre_hits = rng.binomial(needed, n_pre / n_items, m)
    else:
        # Skewed demand: the most-needed items are preloaded first, so the
        # preloaded set covers more than its share.
        cover = min(1.0, (n_pre / n_items) ** (1.0 / max(hit_rate, 1e-6)))
        pre_hits = rng.binomial(needed, cover, m)
    to_fetch = needed - pre_hits
    # Each fetch is a decision that can go wrong. The damaging case is not a
    # fetch that fails -- it is one that returns the WRONG item plausibly, which
    # ch:as-specialized's weak-verifier result says often goes unnoticed.
    got = np.zeros(m, dtype=np.int64)
    calls = np.zeros(m, dtype=np.int64)
    poisoned = np.zeros(m, dtype=bool)
    for k in range(needed):
        active = np.flatnonzero(to_fetch > k)
        if not len(active):
            continue
        calls[active] += 1
        ok = rng.random(len(active)) < p_select
        got[active[ok]] += 1
        wrong = active[~ok]
        if len(wrong):
            noticed = rng.random(len(wrong)) < P_NOTICE
            # Noticed: retry, and the error message narrows the choice.
            r = wrong[noticed]
            calls[r] += 1
            again = rng.random(len(r)) < 0.93
            got[r[again]] += 1
            # Unnoticed: the task proceeds on wrong context.
            poisoned[wrong[~noticed]] = True
    have = pre_hits + got
    # The task succeeds if it has everything it needed and the diluted context
    # still supports the reasoning.
    p = dilution(n_pre)
    reason_ok = rng.random(m) < p ** needed
    ok = (have >= needed) & reason_ok & ~poisoned
    return float(ok.mean()), float(calls.mean()), n_pre


print(f"{M:,} tasks. {N_ITEMS} pieces of context exist; each task needs")
print(f"{NEEDED}. A tool fetch is a decision correct {P_SELECT:.0%} of the time;")
print(f"a wrong fetch is noticed {P_NOTICE:.0%} of the time and retried, and")
print("otherwise poisons the task. A preloaded resource needs no decision.")
print()
print(f"{'preloaded':>11}{'items':>8}{'success':>10}{'tool calls':>12}")
print("-" * 41)
tab = {}
for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    r = run(f)
    tab[f] = r
    print(f"{f:>11.0%}{r[2]:>8}{r[0]:>10.1%}{r[1]:>12.2f}")

best = max(tab, key=lambda k: tab[k][0])

print()
print()
print("The two costs separated. 'Decision cost' is what the fetches lose;")
print("'dilution cost' is what the loaded context loses.")
print()
print(f"{'preloaded':>11}{'decision cost':>15}{'dilution cost':>15}"
      f"{'total loss':>12}")
print("-" * 53)
sep = {}
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    n_pre = int(round(f * N_ITEMS))
    # Decision-only: no dilution.
    d_only = run(f)
    p = dilution(n_pre)
    dil = 1.0 - p ** NEEDED
    dec = 1.0 - d_only[0] / max(p ** NEEDED, 1e-9)
    sep[f] = (dec, dil)
    print(f"{f:>11.0%}{dec:>15.1%}{dil:>15.1%}{1 - d_only[0]:>12.1%}")

print()
print()
print("The optimum moves with how well the host can PREDICT what is needed.")
print("A skew of 1 means demand is uniform; higher means a few items dominate.")
print()
print(f"{'demand skew':>13}" + "".join(f"{'pre ' + format(f, '.0%'):>11}"
                                       for f in (0.0, 0.25, 0.5, 1.0))
      + f"{'best':>9}")
print("-" * 66)
sk = {}
for s in (1.0, 1.6, 2.5, 4.0):
    row = [run(f, hit_rate=s)[0] for f in (0.0, 0.25, 0.5, 1.0)]
    names = ["0%", "25%", "50%", "100%"]
    sk[s] = (row, names[int(np.argmax(row))])
    print(f"{s:>13.1f}" + "".join(f"{v:>11.1%}" for v in row)
          + f"{sk[s][1]:>9}")

print()
print()
print("And with selection reliability, which is ch:ag-tool-calling's variable.")
print("A server whose tools are hard to tell apart should preload more.")
print()
print(f"{'selection':>11}" + "".join(f"{'pre ' + format(f, '.0%'):>11}"
                                     for f in (0.0, 0.25, 0.5, 1.0))
      + f"{'best':>9}")
print("-" * 64)
sl = {}
for ps in (0.98, 0.90, 0.75, 0.55):
    row = [run(f, p_select=ps)[0] for f in (0.0, 0.25, 0.5, 1.0)]
    names = ["0%", "25%", "50%", "100%"]
    sl[ps] = (row, names[int(np.argmax(row))])
    print(f"{ps:>11.0%}" + "".join(f"{v:>11.1%}" for v in row)
          + f"{sl[ps][1]:>9}")

print(f"""
The first table has an interior optimum, which is the whole reason there are
three primitives instead of one. Preloading nothing gives {tab[0.0][0]:.1%};
preloading everything gives {tab[1.0][0]:.1%}; the best row is
{best:.0%} at {tab[best][0]:.1%}.

The second table separates the two costs and shows them crossing. Decision cost
falls from {sep[0.0][0]:.1%} to {sep[1.0][0]:.1%} as fetches are eliminated;
dilution cost rises from {sep[0.0][1]:.1%} to {sep[1.0][1]:.1%} as context grows.
**Neither primitive is free, and they are expensive in opposite directions**
(eq:decision-cost-versus-dilution).

Note what the decision cost actually consists of. It is not fetches that fail --
those get retried. It is fetches that return the wrong thing PLAUSIBLY and are not
noticed, which happens {1 - P_NOTICE:.0%} of the time a selection goes wrong. That
is ch:as-specialized's weak-verifier result appearing inside a tool call: **the
damaging failure is the one the model cannot see.**

The third table contains the result that reverses the intuition. As demand
concentrates -- a few items needed by most tasks -- the best preload fraction goes
DOWN, from {sk[1.0][1]} at uniform demand to {sk[4.0][1]} at a skew of {4.0}.

The reason is that skew makes preloading EFFICIENT rather than making it more
necessary. When a quarter of the inventory covers most of the demand, preloading
that quarter captures nearly all of the benefit, and preloading more only adds
dilution. **The better you can predict what is needed, the less you need to
preload** -- because prediction is what lets a small resource set do the work of a
large one.

Which means a host that cannot predict demand at all faces the worst version of
this trade, and its best move is not a middle setting but one of the extremes.

The fourth table is the actionable one, and it is ch:ag-tool-calling's variable.
At {0.98:.0%} selection reliability the best policy is to preload {sl[0.98][1]} --
let the model choose, it is good at it. At {0.55:.0%} the best policy is
{sl[0.55][1]}, and the gap between the extremes at that reliability is
{sl[0.55][0][3] - sl[0.55][0][0]:+.1%}.

**A resource is what you build when the model cannot reliably choose the tool.**
That is the practical content of the primitive distinction: tools put the decision
in the model, resources take it away, and which is right depends on a number
ch:ag-tool-calling told you how to measure.

So the rule is not "prefer resources" or "prefer tools". It is: measure your
selection reliability, measure your demand skew, and let those two numbers pick
the split.""")
```

The second listing asks what preloading costs in freshness.

```python {tier=A name=resources-go-stale}
"""The other half of the primitive choice, which is freshness.

The previous listing traded a DECISION against DILUTION and found an interior
optimum. It assumed the preloaded content was correct, and that assumption is the
one that fails in production.

A resource is included in context at the start of a turn. A tool is called when
the model wants it. So a resource carries whatever the world looked like when the
host assembled the context, and a tool carries whatever it looks like now
(eq:resources-go-stale).

That is ch:as-long-running's drift arriving at the scale of a single turn, and it
is the axis the tools-versus-resources discussion usually leaves out.
"""
import numpy as np

rng = np.random.default_rng(4159)

M = 50000
NEEDED = 5
TURN_LEN = 9            # steps in a turn, over which preloaded content ages
P_SELECT = 0.90
P_NOTICE = 0.55


def run(preload_frac, volatility, m=M, needed=NEEDED, turn_len=TURN_LEN,
        p_select=P_SELECT, revalidate=False):
    """`volatility` is the chance a given item changes per step of the turn.
    Preloaded items are read at step 0 and used later; fetched items are read
    when used. `revalidate` re-reads preloaded items at the halfway point."""
    n_pre = rng.binomial(needed, preload_frac, m)
    n_fetch = needed - n_pre
    # Preloaded items are used at a random step, having aged since step 0.
    age = rng.integers(1, turn_len + 1, (m, needed))
    if revalidate:
        age = np.minimum(age, np.maximum(age - turn_len // 2, 1))
    stale_p = 1.0 - (1.0 - volatility) ** age
    idx = np.arange(needed)[None, :]
    is_pre = idx < n_pre[:, None]
    stale = (rng.random((m, needed)) < stale_p) & is_pre
    # Fetched items are current, but the fetch is a decision.
    poisoned = np.zeros(m, dtype=bool)
    missing = np.zeros(m, dtype=bool)
    calls = np.zeros(m, dtype=np.int64)
    for k in range(needed):
        active = np.flatnonzero(n_fetch > k)
        if not len(active):
            continue
        calls[active] += 1
        ok = rng.random(len(active)) < p_select
        wrong = active[~ok]
        if len(wrong):
            noticed = rng.random(len(wrong)) < P_NOTICE
            r = wrong[noticed]
            calls[r] += 1
            again = rng.random(len(r)) < 0.93
            missing[r[~again]] = True
            poisoned[wrong[~noticed]] = True
    ok = ~poisoned & ~missing & ~stale.any(1)
    return float(ok.mean()), float(calls.mean()), float(stale.any(1).mean())


print(f"{M:,} tasks needing {NEEDED} items over a {TURN_LEN}-step turn.")
print("Preloaded items are read once at the start; fetched items are current.")
print()
print(f"{'preloaded':>11}" + "".join(f"{'vol ' + format(v, '.1%'):>13}"
                                     for v in (0.0, 0.005, 0.02, 0.06)))
print("-" * 63)
tab = {}
for f in (0.0, 0.25, 0.5, 0.75, 1.0):
    row = tuple(run(f, v)[0] for v in (0.0, 0.005, 0.02, 0.06))
    tab[f] = row
    print(f"{f:>11.0%}" + "".join(f"{x:>13.1%}" for x in row))

print()
print()
print("The best preload fraction at each volatility, and what it costs to get")
print("it wrong in either direction.")
print()
FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
print(f"{'volatility':>12}{'best preload':>14}{'best':>9}"
      f"{'all-resource':>14}{'all-tool':>11}")
print("-" * 60)
opt = {}
for v in (0.0, 0.002, 0.01, 0.03, 0.08):
    row = [run(f, v)[0] for f in FRACS]
    b = int(np.argmax(row))
    opt[v] = (FRACS[b], row[b], row[-1], row[0])
    print(f"{v:>12.1%}{FRACS[b]:>14.0%}{row[b]:>9.1%}{row[-1]:>14.1%}"
          f"{row[0]:>11.1%}")

print()
print()
print("How much staleness there actually is, which is the quantity nobody")
print("measures because a stale resource produces no error.")
print()
print(f"{'volatility':>12}{'stale rate, all-resource':>26}{'success':>10}")
print("-" * 48)
st = {}
for v in (0.0, 0.002, 0.01, 0.03, 0.08):
    r = run(1.0, v)
    st[v] = (r[2], r[0])
    print(f"{v:>12.1%}{r[2]:>26.1%}{r[0]:>10.1%}")

print()
print()
print("Re-reading resources partway through the turn is the cheap fix, and it")
print("is ch:as-long-running's re-validation at a much smaller scale.")
print()
print(f"{'volatility':>12}{'no revalidation':>17}{'revalidated':>13}{'gain':>9}")
print("-" * 51)
rv = {}
for v in (0.002, 0.01, 0.03, 0.08):
    a = run(1.0, v)[0]
    b = run(1.0, v, revalidate=True)[0]
    rv[v] = (a, b)
    print(f"{v:>12.1%}{a:>17.1%}{b:>13.1%}{b - a:>+9.1%}")

print()
print()
print("And the turn length, since a longer turn ages its preloaded context more.")
print()
print(f"{'turn steps':>12}{'all-resource':>14}{'all-tool':>11}{'best':>13}")
print("-" * 50)
tl = {}
for L in (2, 5, 12, 30):
    a = run(1.0, 0.02, turn_len=L)[0]
    b = run(0.0, 0.02, turn_len=L)[0]
    tl[L] = (a, b)
    print(f"{L:>12}{a:>14.1%}{b:>11.1%}"
          f"{('resources' if a > b else 'tools'):>13}")

print(f"""
The first table has a crossover in it, and the crossover is sharp.

At zero volatility, preloading everything gives {tab[1.0][0]:.1%} against
{tab[0.0][0]:.1%} for fetching everything -- resources win by
{tab[1.0][0] - tab[0.0][0]:.1f} points, exactly as the previous listing said they
should when selection is imperfect. At {0.06:.0%} volatility per step, preloading
everything gives {tab[1.0][3]:.1%} against {tab[0.0][3]:.1%}.

**The same design that wins by twenty-two points loses by fifty-six, and the only
thing that changed is how fast the world moves** (eq:resources-go-stale).

The second table locates the crossover at about {0.01:.0%} volatility per step.
Below it, preload; above it, fetch. That is a usable rule and it is stated in a
unit teams can actually measure -- how often does this thing change, per step of a
turn.

The third table is why this goes unnoticed. At {0.03:.0%} volatility,
{st[0.03][0]:.1%} of tasks are working from at least one stale item, and **a stale
resource produces no error**. It is well-formed, it is plausible, it was correct
recently. The task completes and the answer is wrong, which is
ch:as-long-running's silent drift compressed from days into a single turn.

The fourth table is the fix, and it is the same fix that chapter found. Re-reading
preloaded resources partway through the turn is worth
{rv[0.01][1] - rv[0.01][0]:+.1%} at {0.01:.0%} volatility and
{rv[0.08][1] - rv[0.08][0]:+.1%} at {0.08:.0%}, for one extra read.

**Re-validation is the cheapest intervention at every scale this book has measured
it** -- across a week-long workflow in ch:as-long-running and across a nine-step
turn here.

The last table adds the variable that makes this worse over time. A two-step turn
favours resources ({tl[2][0]:.1%} against {tl[2][1]:.1%}); a thirty-step turn
favours tools by {tl[30][1] - tl[30][0]:.1f} points at the same volatility.

Agentic turns are getting longer, which means **content that was safe to preload
becomes unsafe to preload without anything about it changing.** A resource set
tuned when turns were three steps will quietly rot when turns become twenty, and
the symptom will be wrong answers rather than errors.

Putting the two listings together gives the rule the primitive split is really
for:

**Preload as resources what is stable, predictable and hard to select. Fetch as
tools what is volatile.** Stability decides it first -- above about
{0.01:.0%} per-step volatility nothing else matters -- and below that threshold,
selection reliability and demand skew decide the fraction.""")
```

## 9. Practical Example

The first listing runs tasks needing five of forty context items, with tool
selection correct $90\%$ of the time and wrong selections noticed $55\%$ of the
time:

```
  preloaded   items   success  tool calls
-----------------------------------------
         0%       0     75.9%        6.42
        25%      10     79.3%        4.80
        50%      20     80.4%        3.20
       100%      40     83.1%        0.00
```

The two costs, separated:

```
  preloaded  decision cost  dilution cost  total loss
-----------------------------------------------------
         0%          21.9%           2.5%       23.8%
        50%          11.5%           8.9%       19.4%
       100%          -0.1%          16.9%       16.8%
```

**The curves cross** ({{eq:decision-cost-versus-dilution}}) — decision cost falls
from $21.9\%$ to nothing as fetches are eliminated, dilution rises from $2.5\%$ to
$16.9\%$ as context grows. Most of the decision cost is *unnoticed wrong fetches*
rather than failed ones: {{ch:as-specialized}}'s weak-verifier result inside a
single call.

As demand concentrates:

```
  demand skew     pre 0%    pre 25%    pre 50%   pre 100%     best
------------------------------------------------------------------
          1.0      75.9%      79.2%      80.4%      83.1%     100%
          2.5      75.7%      85.8%      86.3%      82.8%      50%
          4.0      75.8%      88.7%      87.6%      82.9%      25%
```

**The better you can predict what is needed, the less you should preload**
({{eq:prediction-reduces-preloading}}) — concentration makes preloading efficient,
so a small set captures the benefit and the rest is pure dilution.

And against selection reliability:

```
  selection     pre 0%    pre 25%    pre 50%   pre 100%     best
----------------------------------------------------------------
        98%      92.8%      92.1%      88.9%      83.1%       0%
        90%      75.9%      79.3%      80.4%      83.1%     100%
        55%      28.8%      38.6%      51.1%      82.8%     100%
```

**A resource is what you build when the model cannot reliably choose the tool**
({{eq:resource-when-selection-is-weak}}) — a $54$-point swing at $55\%$ selection.

The second listing adds freshness over a nine-step turn:

```
  preloaded     vol 0.0%     vol 0.5%     vol 2.0%     vol 6.0%
---------------------------------------------------------------
         0%        77.8%        77.8%        77.6%        78.4%
        50%        88.5%        82.8%        69.0%        43.3%
       100%       100.0%        88.6%        60.5%        22.5%
```

**The same design that wins by $22.2$ points loses by $55.9$**, and the only thing
that changed is how fast the world moves ({{eq:resources-go-stale}}). Note the
all-tool row barely moves — fetched content is current by construction.

The crossover:

```
  volatility  best preload     best  all-resource   all-tool
------------------------------------------------------------
        0.0%          100%   100.0%        100.0%      77.8%
        0.2%          100%    95.0%         95.0%      77.7%
        1.0%            0%    78.2%         77.8%      78.2%
        8.0%            0%    78.2%         14.0%      78.2%
```

About $1\%$ change per step. Below it, preload; above it, fetch.

Why it goes unnoticed:

```
  volatility  stale rate, all-resource   success
------------------------------------------------
        1.0%                     22.2%     77.8%
        3.0%                     52.4%     47.6%
        8.0%                     86.2%     13.8%
```

At $3\%$ volatility more than half of tasks work from a stale item, and **a stale
resource produces no error** — it is well-formed, plausible, and was correct
recently.

Revalidation:

```
  volatility  no revalidation  revalidated     gain
---------------------------------------------------
        1.0%            77.8%        90.2%   +12.4%
        8.0%            13.9%        42.6%   +28.7%
```

One extra read. **Re-validation is the cheapest intervention at every scale this
book has measured it**, across a week-long workflow in {{ch:as-long-running}} and
across a nine-step turn here.

And turn length:

```
  turn steps  all-resource   all-tool         best
--------------------------------------------------
           2         86.0%      77.3%    resources
          12         52.4%      77.8%        tools
          30         22.8%      77.7%        tools
```

Agentic turns are getting longer, so **content that was safe to preload becomes
unsafe without anything about it changing.**

## 10. Production Considerations

Classify each capability by who can know it is needed: the host (resource), the
model (tool), or the user (prompt). That question, asked per capability, is the
entire design.

Measure per-step volatility for anything you preload. Above roughly $1\%$, make it
a tool and stop reasoning about the other variables.

Measure your selection reliability, as {{ch:ag-tool-calling}} describes. It and
demand skew set the preload fraction below the volatility threshold.

Echo identity in every fetch response — title, URI, version — so a mis-selection
is detectable. It is the cheapest reduction in unnoticed wrong fetches available
and almost nothing does it.

Revalidate preloaded resources at least once mid-turn.

Give resources stable, identity-bearing URIs. A URI encoding a query is a tool.

Expose important capabilities through more than one primitive so hosts can pick
their own operating point.

Add prompts for repeated workflows — a user-initiated path has no decision cost.

And treat tool descriptions and resource contents as untrusted text reaching the
model, per {{cite:mcp2026spec}}'s own guidance.

## 11. Common Mistakes

**Exposing everything as tools.** Moves every decision to the model, including
free ones.

**Reading the split as data versus actions.** It is about who initiates.

**Preloading more because demand is concentrated.** Backwards — concentration
means a smaller set suffices.

**Preloading volatile content.** Above the threshold nothing else matters.

**Returning content without identity.** Makes mis-selection silent.

**Query-shaped resource URIs.** Uncacheable, undiffable, unrevalidatable.

**Ignoring prompts.** The one primitive with no decision cost at all.

**Tuning a resource set once.** Turn lengths grow and the tuning silently expires.

## 12. Failure Modes

*Silent stale context.* The characteristic resource failure — plausible, recently
correct, wrong, and errorless.

*Unnoticed wrong fetch.* The characteristic tool failure, and the larger half of
decision cost.

*Dilution from generous preloading.* {{ch:ag-memory}}'s effect, invisible because
the content is all technically relevant.

*Tool poisoning.* {{cite:huang2026mcpthreat}}'s most prevalent client-side
vulnerability, riding the metadata channel every server has.

*Operating-point mismatch.* A server exposing one primitive forcing every host to
the same point on the curve.

## 13. Alternatives

**Retrieval instead of either.** Index the inventory and include the top-$k$ by
relevance — this is {{cite:qin2023toolllm}}'s regime, and it is a resource strategy
with a learned host-side predictor rather than a third option.

**Lazy resources.** Include a manifest of what is available, let the model request
expansion. A hybrid whose first hop is a cheap decision over a distinct inventory.

**Caching with invalidation.** Preload but subscribe to changes, which converts the
freshness problem into a delivery problem — the right answer when the underlying
system supports it.

**Prompt-only servers.** For a workflow that never varies, exposing nothing but
prompts eliminates decision cost entirely, at
{{eq:graph-surrenders-the-tail}}'s price.

## 14. Evaluation

Measure per-step volatility for each resource. It is one number per item and it
decides the primitive.

Measure your unnoticed-wrong-fetch rate by injecting mis-selections and counting
how many the model proceeds on. This is the term that dominates decision cost and
nobody measures it.

Measure demand skew from real turns: what fraction of context requests fall on the
top decile of items.

Report success at several preload fractions rather than shipping one. The curve is
flat near the optimum and steep away from it.

And re-measure when turn length changes, since $v^*$ moves with it.

## 15. Advanced Concepts

**Host-side demand prediction as a learned component.** The coverage function in
{{eq:prediction-reduces-preloading}} is learnable from turn histories, which would
make the preload set adaptive rather than configured. {{maturity:EMERGING}}.

**Volatility-aware resource protocols.** A server could publish a change rate
alongside each resource, letting hosts apply the threshold automatically. Nothing
does this and it is a small extension.

**Subscription semantics.** Push-based invalidation would collapse the freshness
axis entirely, at the cost of the stateless property
{{ch:mcp-architecture}} was written to defend.

**Measuring the instruction channel.** {{cite:huang2026mcpthreat}}'s finding
invites a quantitative question nobody has answered: how much of a typical turn's
context is text authored by parties the user never chose?
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-tool-calling}}'s selection reliability turns out to be one of two numbers
that set the primitive split, and its error-message finding explains why *failed*
fetches are cheap and wrong ones are not.

{{ch:ag-memory}}'s dilution is the price of the resource side, and this chapter is
where that cost gets weighed against something rather than merely noted.

{{ch:as-long-running}}'s re-validation reappears at a scale a thousand times
smaller with the same verdict, which is about as strong a robustness check as this
book offers.

{{ch:as-specialized}}'s weak-verifier result explains the shape of decision cost:
the damaging failure is the undetectable one.

Ahead: {{ch:mcp-schemas}} takes up what tool descriptions cost in context and how
discovery works over a large inventory; {{ch:mcp-security}} takes up the
instruction channel {{sec:7-internal-mechanics}} names here.

## 17. Exercises

1. Derive $v^*$ from {{eq:resources-go-stale}} and check it against the second
   listing's crossover.

2. Add identity echo to the first listing — a fraction of wrong fetches become
   noticed. How much is it worth per point of $\nu$?

3. Implement the lazy-resource hybrid: a manifest plus expansion. Where does it
   sit on the curve?

4. Model subscription-based invalidation and find the volatility at which its
   complexity pays.

5. Make demand skew time-varying and test whether a fixed preload set degrades.

6. Combine both listings into one model and check whether the freshness threshold
   really does dominate, or whether very weak selection can overcome it.

## 18. Interview Questions

1. What distinguishes a tool from a resource?

2. When is a wrong tool call worse than a failed one?

3. Your demand is highly concentrated. Should you preload more or less?

4. Your model selects tools correctly $60\%$ of the time. What follows?

5. Your agent returns confidently wrong answers with no errors logged and all
   context preloaded. What do you check?

6. Why would a server expose the same capability three different ways?

## 19. Research Questions

1. Can host-side demand prediction be learned well enough to make preload sets
   adaptive?

2. Should servers publish per-resource volatility, and would hosts use it?

3. What is the right subscription semantics for resources given a stateless
   transport?

4. How much of a typical agent turn's context is authored by third parties?

5. Does the freshness threshold hold across domains, or is $\bar{a}$ too
   task-specific?

## 20. Chapter Summary

MCP's three primitives differ by **who decides to use them** — model for tools,
host for resources, user for prompts
({{eq:primitive-is-a-controller-choice}}) — not by what they carry. Exposing
everything as a tool moves every decision to the model, including the free ones.

A tool call is a decision, and decisions cost. Fetching everything lost $21.9$
points, most of it from selections that returned the wrong thing *plausibly* and
were not noticed rather than from calls that failed. A resource removes the
decision and pays in dilution instead — $16.9$ points at full preloading — so the
two costs cross and there is an interior optimum
({{eq:decision-cost-versus-dilution}}).

Two results locate it. **Better prediction means less preloading**: the optimum
moved from $100\%$ to $25\%$ as demand concentrated
({{eq:prediction-reduces-preloading}}), because concentration makes a small set
sufficient. And **a resource is what you build when the model cannot reliably
choose the tool**: the optimum moved from $0\%$ to $100\%$ as selection fell from
$98\%$ to $55\%$ ({{eq:resource-when-selection-is-weak}}).

Freshness overrides both. A resource is read when the host assembles the turn and
used many steps later, so it goes stale ({{eq:resources-go-stale}}) — preloading
everything fell from $100\%$ to $22.5\%$ as volatility rose to $6\%$ per step,
while fetching everything did not move. **The crossover is about $1\%$ per step**,
and above it nothing else matters.

At $3\%$ volatility, more than half of tasks worked from a stale item, and **a
stale resource produces no error.** Re-reading once mid-turn recovered $+28.7$
points — the same intervention {{ch:as-long-running}} found cheapest across a
week-long workflow, here across nine steps.

And a warning with a date on it: turn length enters the freshness factor
exponentially, and turns are getting longer. **Content that was safe to preload
becomes unsafe without changing at all.**

## 21. Further Reading

{{cite:mcp2026spec}} for the primitive definitions and for its own statement that
tool annotations should be treated as untrusted.

{{cite:huang2026mcpthreat}} for tool poisoning as the dominant client-side
vulnerability, which is the security face of
{{sec:7-internal-mechanics}}'s instruction-channel observation, and
{{cite:greshake2023indirect}} for the same vector through resource content.

{{cite:qin2023toolllm}} and {{cite:li2023apibank}} for the inventory scale at
which host-side selection becomes retrieval, which
{{ch:mcp-schemas}} takes up next.

{{ch:ag-tool-calling}} for selection reliability and {{ch:ag-memory}} for
dilution — the two numbers this chapter's optimum is built from.
