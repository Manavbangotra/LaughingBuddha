---
id: mcp-schemas
number: 173
part: XIX
tier: full
status: draft
requires: [distinctness-not-count, decision-cost-versus-dilution,
           enumerate-before-remove, connectivity-is-the-real-quantity]
provides: [schemas-are-rent, verbosity-is-multiplicative,
           retrieval-trades-context-for-recall, retrieval-crossover-is-small,
           context-is-a-budget, allocation-depends-on-budget]
citations: [mcp2026spec, qin2023toolllm, patil2023gorilla, li2023apibank,
            schick2023toolformer]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish what a tool inventory
costs in *selection* from what it costs in *context*, and say why
{{ch:ag-tool-calling}}'s "count is nearly free" result is compatible with a large
inventory being ruinous; compute the token rent of a schema set; explain why tool
description length is the highest-leverage thing a server author controls; locate
the inventory size at which retrieval starts winning; and allocate a context
budget deliberately rather than by subtraction.

## 2. Why This Matters

{{ch:ag-tool-calling}} measured selection at $100\%$ with four tools and $100\%$
with $128$, and concluded that tool *count* is nearly free while tool *overlap* is
expensive. That result is correct and it is about the **decision**.

This chapter is about the **bill**, which behaves entirely differently. A tool's
schema — name, description, argument types — is text in the model's context, paid
for on every request whether or not the tool is called
({{eq:schemas-are-rent}}). {{sec:9-practical-example}} finds selection loss moving
from $9.0\%$ to $13.5\%$ between eight tools and two thousand, while dilution loss
moves from $3.9\%$ to $99.9\%$.

**Tool count is nearly free for selection and ruinous for context.** The two
findings do not conflict; a team that read the first as permission to connect
every available server will discover the second.

One variable in that bill is entirely under a server author's control and costs
nothing to change. The same $512$ tools cost $33{,}792$ tokens described tersely
and $253{,}440$ written as reference documentation, for $58.0\%$ success against
$1.4\%$ ({{eq:verbosity-is-multiplicative}}). A verbose server charges every host
that connects to it, on every request, forever.

The escape is retrieval — index the inventory, show only what this request
plausibly needs, which is the standard answer at
{{cite:qin2023toolllm}}'s sixteen-thousand-API scale. It converts a context
problem into a recall problem ({{eq:retrieval-trades-context-for-recall}}), and
the crossover is smaller than people expect: retrieval starts winning between
sixteen and sixty-four tools ({{eq:retrieval-crossover-is-small}}).

The chapter closes on the budget the schemas are competing for. Schemas,
preloaded resources and conversation history all draw on the same context, all
saturate, and all dilute — so the split is a real decision
({{eq:context-is-a-budget}}) that almost nobody makes. The allocation that arrives
on its own, history-heavy, is $20.6$ points off the best.

## 3. Prerequisites

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}}, which this chapter
qualifies rather than contradicts, and its {{eq:enumerate-before-remove}} — since
enumerated arguments cost tokens as well as buying reliability.

{{ch:mcp-primitives}}'s {{eq:decision-cost-versus-dilution}}, whose dilution term
is the same one charged here.

{{ch:ag-memory}}'s dilution result, which is the underlying effect.

{{ch:mcp-why}}'s connectivity framing, since discovery is what makes a large
inventory reachable at all.

## 4. Intuitive Explanation

Connect a host to six MCP servers and look at what reaches the model.

Before the user's message, before any reasoning, the context contains a
description of every tool every one of those servers offers. A filesystem server
brings a dozen; an issue tracker brings twenty; a cloud provider's server can
bring hundreds. Each is a name, a sentence or two of description, and a JSON
Schema naming and typing each argument.

That is not a small amount of text, and it is present on *every single request*.
The model pays for it when it uses a tool and when it does not.

Now recall what {{ch:ag-tool-calling}} found: adding tools barely hurt selection.
Quadrupling a distinct inventory cost nothing measurable. It is easy to read that
as "connect everything, the model will cope."

The model will cope with *choosing*. What it will not cope with is the context.
{{ch:ag-memory}} found that extending a context window made recall of recent facts
*worse*, because everything present competes with everything else. Tool schemas
are present, so tool schemas compete — with the user's actual question, with the
retrieved documents, with the conversation so far.

{{sec:9-practical-example}} separates the two effects and they diverge sharply.
Selection degrades gently and logarithmically. Dilution degrades linearly in
tokens and there are a lot of tokens.

Which brings the design question to somewhere unexpected: the single most
consequential thing a server author does is decide how long the descriptions are.
Terse or documentation-length is a factor of seven in tokens, and the author
writing them has no visibility into how many other servers are connected
alongside.

The way out at scale is the one every large system reaches: don't show everything.
Index the tools, and for each request retrieve the handful most likely to be
relevant. {{cite:qin2023toolllm}} worked with sixteen thousand APIs, and nobody
puts sixteen thousand schemas anywhere.

Retrieval costs something specific. It can fail to surface the tool the task
needed, and when it does, the tool may as well not exist. So retrieval turns a
context problem into a recall problem, and recall becomes the number to measure.

The surprise in the measurement is how *early* this pays. Retrieval starts winning
somewhere between sixteen and sixty-four tools — which is one or two servers, not
a marketplace. Hosts that connect a dozen servers with no retrieval layer are past
the crossover and degrade in a way nobody attributes to the tool list, because the
tool list is not what appears to have changed.

And then the larger question the schemas are a part of. Context is a budget with
three claimants: tool schemas, preloaded resources, and conversation history. Each
helps with diminishing returns; each dilutes the others. There is a best split, and
almost nobody chooses one — schemas take what the connected servers need, resources
take what someone configured once, and history takes the rest by simply
continuing.

## 5. Formal Explanation

Let an inventory of $n$ tools have mean argument count $a$, per-tool description
cost $d$, per-argument cost $\alpha$, and a verbosity multiplier $\nu$. The token
rent is:

$$T(n) = n\,\nu\,(d + \alpha a)$$ (eq:schemas-are-rent)

**Linear in $n$ and linear in $\nu$**, and paid per request. Selection, by
contrast, degrades with the log of the candidate count — distinct points in a
high-dimensional space stay separated:

$$p_{\text{sel}}(n) = p_0^{\,1 + \log_2 n / c}$$

Task success over $k$ steps combines both:

$$S(n) = p_{\text{sel}}(n)^{k}\cdot\big(p_r(1 - \delta T(n))\big)^{k}$$

Differentiating, the selection term contributes $O(k\log n)$ to $-\log S$ and the
dilution term contributes $O(k\,\delta\,n\nu(d+\alpha a))$ — **linear beats
logarithmic**, so for any fixed $\delta > 0$ there is an $n$ past which dilution
dominates, regardless of how good selection is.

Since $\nu$ multiplies the dominant term:

$$\frac{\partial \log S}{\partial \nu} = -k\,\delta\,n(d+\alpha a)\Big/\big(1 - \delta T\big)$$ (eq:verbosity-is-multiplicative)

**Verbosity is as expensive as tool count and free to change.**

Now retrieval. Show $r \ll n$ schemas selected by an index with recall $\varrho$ —
the probability the needed tool is among them. Success becomes:

$$S_{\text{ret}}(n, r) = \varrho(r)\cdot p_{\text{sel}}(r)^{k}\cdot\big(p_r(1-\delta T(r))\big)^{k}$$ (eq:retrieval-trades-context-for-recall)

The dilution term now depends on $r$ rather than $n$, so it is bounded however
large the inventory grows. The cost is the new leading factor $\varrho(r)$, which
is a property of the index rather than of the model.

Note that $\varrho$ is increasing in $r$ while the dilution term is decreasing, so
there is an optimal $r$ — and because $\varrho$ saturates quickly for a decent
index while dilution does not, that optimum is *small*.

Setting $S_{\text{ret}}(n, r^*) = S(n)$ and solving for the crossover:

$$n^* \approx r^* + \frac{-\log \varrho(r^*)}{k\,\delta\,\nu(d+\alpha a)}$$ (eq:retrieval-crossover-is-small)

The second term is the recall penalty divided by the per-tool dilution cost. With
realistic values both are small, so **$n^*$ is tens, not thousands.**

Finally the budget. Let total context $B$ be split $B = \sum_i \beta_i B$ among
components $i$, each with saturating benefit and shared dilution:

$$S(\boldsymbol{\beta}) = \prod_i q_i(\beta_i B)^{w_i}\cdot\big(p_r(1-\delta B\textstyle\sum_i \beta_i)\big)^{k}, \qquad q_i(t) = f_i + (1-f_i)\big(1 - e^{-t/s_i}\big)$$ (eq:context-is-a-budget)

Each $\log q_i$ is concave increasing and the dilution term is linear decreasing,
so $\log S$ is concave on the simplex and has a unique interior maximum. Two
consequences:

$$\frac{\partial \beta_i^*}{\partial B} \ne 0$$ (eq:allocation-depends-on-budget)

— **the optimal split moves with the budget**, so a configuration tuned for a
small window is wrong for a large one — and, since the benefit saturates while the
cost does not, $S$ is non-monotone in $B$ itself: **there is a best amount of
context, and it is finite.**

## 6. Mathematical Foundation

Three extractions.

**Linear beats logarithmic, eventually and then suddenly.** The reason
{{ch:ag-tool-calling}}'s result and this one coexist is a growth-rate difference.
At small $n$ the log term is visible and the linear term is negligible; the
crossover is not gradual, because a linear function overtaking a logarithm does so
decisively. That is why teams experience tool-list bloat as a cliff rather than a
slope.

**The optimal retrieval width is small because recall saturates and dilution does
not.** From {{eq:retrieval-trades-context-for-recall}}, $\varrho(r)$ is concave and
approaching one while $T(r)$ is linear. The marginal tool shown adds shrinking
recall and constant cost, so the optimum sits where a decent index has most of its
recall — typically under a dozen.

**Context has a finite optimum.** {{eq:context-is-a-budget}}'s benefit terms
saturate and its cost term does not, so more context is not monotonically better.
This is a stronger claim than "long contexts degrade" — it says there is a point
past which adding *relevant* material makes things worse, and
{{sec:9-practical-example}} locates it.

## 7. Internal Mechanics

### 7.1 What a schema actually costs

```mermaid {#fig:schema-rent caption="Every connected server's schemas are present on every request. The rent is paid whether or not any tool is called."}
flowchart TD
    S1[server A: 12 tools] --> CTX[request context]
    S2[server B: 20 tools] --> CTX
    S3[server C: 140 tools] --> CTX
    R[preloaded resources] --> CTX
    H[conversation history] --> CTX
    Q[the user's question] --> CTX
    CTX --> LM[model]
    LM --> ANS[one tool call, or none]
```

The asymmetry the diagram is drawn to show: everything on the left is paid for
every time, and the output uses at most one of them.

A rough accounting per tool: a dozen tokens of name and one-line description, plus
thirty-odd per documented argument with its type and constraints, at four or five
arguments typical. That is roughly $170$ tokens per tool — so a hundred-tool
inventory is a $17{,}000$-token standing charge.

### 7.2 The argument-count tension with {{ch:ag-tool-calling}}

{{eq:enumerate-before-remove}} found that enumerating an argument's valid values
beat removing the argument entirely — constraints buy reliability.

Enumerations also cost tokens, and this chapter says tokens are not free. The two
results are reconciled by scale rather than by compromise: an enumeration on a
tool that is *shown* is worth its cost, because it prevents a selection error on a
tool the model is actually considering.

Which is an argument for retrieval rather than for terseness. **Show fewer tools,
described more completely**, rather than more tools described sparsely. The first
keeps {{ch:ag-tool-calling}}'s reliability and pays {{eq:schemas-are-rent}} only on
what is shown; the second sacrifices reliability to pay rent on tools that will
not be used.

### 7.3 Writing a description that earns its tokens

Given {{eq:verbosity-is-multiplicative}}, the question is what belongs in a
description and what does not.

**Belongs**: what distinguishes this tool from the others in the inventory — which
is {{eq:distinctness-not-count}}'s variable, and the only thing selection actually
needs. Argument constraints that prevent malformed calls. The one non-obvious
precondition, if there is one.

**Does not belong**: prose about what the underlying service is. Examples, unless
the call shape is genuinely unguessable. Error semantics — those belong in the
error message, where {{ch:ag-tool-calling}} found them worth far more anyway.
Anything a reader would call background.

The test: **would removing this sentence make the model more likely to choose the
wrong tool?** If not, it is rent with no return.

### 7.4 Discovery, and what the protocol supplies

{{cite:mcp2026spec}} gives a mandatory `server/discover` RPC returning a server's
supported versions, capabilities and identity, and tool listing per server. What it
does not give — deliberately — is cross-server retrieval or ranking.

That is host-side work, and it is the layer this chapter says most hosts are
missing. The protocol makes the inventory *enumerable*; deciding which slice to
show the model is not a protocol concern, because it depends on the request, the
conversation, and the host's own budget.

{{cite:patil2023gorilla}} is the relevant prior result: pairing generation with a
document retriever both reduced API hallucination and let the system track
documentation changes without retraining. The same mechanism, applied to schemas,
is what keeps a large inventory usable.

### 7.5 Retrieval failure is invisible

{{eq:retrieval-trades-context-for-recall}}'s $\varrho$ term deserves its own
warning, because its failure mode is unlike the others in this chapter.

When retrieval misses, the model does not see a missing tool. It sees an inventory
that does not contain what it needs, and it does the reasonable thing: it uses
something else, or it says it cannot. Neither looks like a retrieval failure. The
trace shows a model that chose poorly or gave up.

So a host with a retrieval layer needs to measure recall *directly* — by checking,
on tasks whose correct tool is known, whether that tool was among those shown.
Nothing else surfaces it, and the symptom is easy to misattribute to the model.

### 7.6 The budget nobody allocates

The three claimants on context arrive from three different places and none of them
negotiates.

**Schemas** are set by which servers a user connected, which is a configuration
decision usually made for reasons of capability rather than budget.

**Resources** are set by the host's preloading policy, which
{{ch:mcp-primitives}} showed should depend on volatility and selection reliability
and in practice is a fixed list.

**History** grows by itself. It requires no decision at all, which is why the
history-heavy allocation is the one that arrives on its own in every long session
— and {{sec:9-practical-example}} finds it the worst of the four defaults tested.

The instruction is to make the split explicit: name the line items, give each a
cap, and enforce the caps. The surface is flat near its optimum so precision is not
required — an even three-way split was within $1.8$ points of best — but the
defaults that arise by accident are much further off.

### 7.7 Why bigger windows do not fix this

{{eq:allocation-depends-on-budget}} carries a warning worth stating for the case
teams actually encounter.

When a host moves to a model with a much larger window, the natural response is to
relax every cap. {{sec:9-practical-example}} finds success peaking around
$24{,}000$ tokens and falling to $9.7\%$ at $160{,}000$ — because the benefit of
each component saturates while the dilution cost does not.

And the optimal split *shifts* as the budget grows: at $4{,}000$ tokens the best
allocation spends nothing on history, and by $24{,}000$ it spends over a quarter.
A configuration tuned under scarcity under-weights history and over-weights
schemas, so a team that upgrades the window and changes nothing else keeps an
allocation chosen for a different regime — and then concludes the larger window did
not help.

**It did help. It was filled with the wrong proportions.**

### 7.8 The rent is paid by someone who cannot see it

There is a structural problem behind {{eq:schemas-are-rent}} that no amount of
careful authoring fixes, and it is worth naming because it predicts how tool
ecosystems degrade.

The server author decides the description length. The host pays for it. And the
two parties have no channel between them: the author cannot see how many other
servers are connected alongside, what the host's context budget is, or how much of
it their tools are consuming. A server that is perfectly reasonable alone — a
hundred and seventy tokens per tool, forty tools, seven thousand tokens — is
perfectly reasonable right up until six of them are connected together.

That is a tragedy-of-the-commons in the exact technical sense: the cost of each
server's verbosity is borne by a shared resource, and no participant sees the
aggregate. Each author's locally correct decision to document their tools well
produces a collectively bad outcome, and nobody involved has done anything wrong.

Which means the fix cannot be exhortation to server authors. It has to be
host-side, and there are only two mechanisms available. **Retrieval** caps the
rent at whatever the host chooses to show, regardless of how verbose the inventory
is — which is a second, independent argument for it beyond
{{eq:retrieval-crossover-is-small}}. And **truncation** — the host rewriting or
trimming descriptions before presentation — works but discards exactly the
distinguishing detail {{eq:distinctness-not-count}} says selection depends on.

Retrieval is clearly the better of the two, because it reduces the *number* of
schemas rather than the *quality* of each. That distinction is the practical
content of {{sec:7-internal-mechanics}}'s earlier rule: show fewer tools,
described completely.

It also suggests what a protocol extension would be for. A host that could
advertise its budget, and servers that could return schemas at a requested detail
level, would close the information gap that creates the commons problem in the
first place. Nothing in {{cite:mcp2026spec}} does this today.

## 8. Implementation

Two listings. The first prices a tool inventory and locates the retrieval
crossover. The second allocates the context budget the schemas compete in.

```python {tier=A name=schemas-are-rent}
"""What a tool inventory costs before any tool is called.

Every tool a host offers is described in the model's context: a name, a
description, a JSON Schema for its arguments. That description is paid for on
EVERY request, whether or not the tool is used.

ch:ag-tool-calling found selection robust to inventory size and fragile to
inventory overlap, and measured selection at 100% with 128 distinct tools. That
result is about the DECISION. This listing is about the BILL, which behaves
differently: a schema is tokens, tokens are context, and ch:ag-memory found
context dilutes (eq:schemas-are-rent).

cite:qin2023toolllm worked with 16,000+ APIs. Nobody puts 16,000 schemas in a
context window, so something has to choose, and the something is retrieval.
"""
import numpy as np

rng = np.random.default_rng(4201)

M = 40000
TOK_NAME = 12           # tokens for a name and one-line description
TOK_ARG = 34            # tokens per documented argument
ARGS_MEAN = 4.5
CTX = 16000             # tokens the host is willing to spend on tool schemas
BASE = 0.995


def schema_tokens(n_tools, args_mean=ARGS_MEAN, verbose=1.0):
    """Tokens consumed by n_tools of schema. `verbose` scales description length."""
    return n_tools * (TOK_NAME * verbose + TOK_ARG * args_mean * verbose)


def dilution(tokens):
    """ch:ag-memory's effect, as a per-step multiplier on reasoning quality."""
    return BASE * (1.0 - 2.2e-6 * tokens)


def run(n_tools, m=M, steps=5, args_mean=ARGS_MEAN, verbose=1.0,
        retrieved=None, recall=0.94, p_select_base=0.985):
    """`retrieved` is how many tools are actually put in context; None means all.
    Retrieval may miss the needed tool, at rate 1 - recall."""
    shown = n_tools if retrieved is None else min(retrieved, n_tools)
    toks = schema_tokens(shown, args_mean, verbose)
    # Selection degrades gently with how many candidates are visible, per
    # ch:ag-tool-calling: it is overlap that hurts, and overlap grows with count.
    p_sel = p_select_base ** (1.0 + np.log2(max(shown, 1)) / 12.0)
    # Retrieval can simply fail to surface the right tool.
    present = np.ones(m, dtype=bool) if retrieved is None else \
        (rng.random(m) < recall)
    ok_sel = (rng.random((m, steps)) < p_sel).all(1)
    p_reason = dilution(toks)
    ok_reason = rng.random(m) < p_reason ** steps
    ok = present & ok_sel & ok_reason
    return float(ok.mean()), toks, float(p_sel)


print(f"A schema costs about {TOK_NAME} tokens of name and description plus")
print(f"{TOK_ARG} per argument, at {ARGS_MEAN} arguments on average. Every")
print("request pays for every tool offered.")
print()
print(f"{'tools':>8}{'schema tokens':>15}{'select/step':>13}{'success':>10}")
print("-" * 46)
tab = {}
for n in (8, 32, 128, 512, 2048):
    r = run(n)
    tab[n] = r
    print(f"{n:>8}{r[1]:>15,.0f}{r[2]:>13.2%}{r[0]:>10.1%}")

print()
print()
print("Where the loss comes from. 'Selection' is the decision degrading with")
print("candidate count; 'dilution' is the schema text crowding the context.")
print()
print(f"{'tools':>8}{'selection loss':>16}{'dilution loss':>15}{'total':>9}")
print("-" * 48)
sep = {}
for n in (8, 128, 512, 2048):
    toks = schema_tokens(n)
    p_sel = run(n)[2]
    sel = 1 - p_sel ** 5
    dil = 1 - dilution(toks) ** 5
    sep[n] = (sel, dil)
    print(f"{n:>8}{sel:>16.1%}{dil:>15.1%}{1 - tab.get(n, run(n))[0]:>9.1%}")

print()
print()
print("Verbosity is the variable a server author controls directly, and it is")
print("multiplicative in the token cost. 512 tools:")
print()
print(f"{'description length':>20}{'schema tokens':>15}{'success':>10}")
print("-" * 45)
vb = {}
for v, label in ((0.4, "terse"), (1.0, "typical"), (1.8, "generous"),
                 (3.0, "documentation")):
    r = run(512, verbose=v)
    vb[label] = r
    print(f"{label:>20}{r[1]:>15,.0f}{r[0]:>10.1%}")

print()
print()
print("Retrieval: show only the k most relevant schemas. The cost is that")
print("retrieval sometimes fails to surface the tool the task needs.")
print()
print(f"{'shown of 2048':>15}{'schema tokens':>15}{'recall 99%':>12}"
      f"{'recall 94%':>12}{'recall 85%':>12}")
print("-" * 66)
rt = {}
for k in (8, 16, 32, 64, 2048):
    row = tuple(run(2048, retrieved=(None if k == 2048 else k), recall=rc)[0]
                for rc in (0.99, 0.94, 0.85))
    rt[k] = (schema_tokens(min(k, 2048)), row)
    label = "all" if k == 2048 else str(k)
    print(f"{label:>15}{schema_tokens(min(k, 2048)):>15,.0f}"
          + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("And the inventory size at which retrieval starts winning, which is the")
print("number a host actually needs.")
print()
print(f"{'inventory':>11}{'show all':>11}{'retrieve 24':>14}{'better':>11}")
print("-" * 47)
xo = {}
for n in (16, 64, 256, 1024, 4096):
    a = run(n)[0]
    b = run(n, retrieved=24)[0]
    xo[n] = (a, b)
    print(f"{n:>11}{a:>11.1%}{b:>14.1%}"
          f"{('retrieve' if b > a else 'show all'):>11}")

print(f"""
The first table looks like ch:ag-tool-calling's result reversed, and the second
shows it is not.

Selection loss barely moves: {sep[8][0]:.1%} at {8} tools and {sep[2048][0]:.1%} at
{2048}. That is that chapter's finding intact -- **the DECISION is nearly free in
inventory size**, because distinct tools stay distinguishable.

Dilution loss goes {sep[8][1]:.1%} to {sep[2048][1]:.1%}. That is the bill, and it
is a different quantity entirely (eq:schemas-are-rent). A schema is text in the
context window; it is paid for on every request whether or not the tool is used;
and ch:ag-memory found that everything present competes with everything else.

So the two chapters do not disagree. **Tool count is nearly free for selection and
ruinous for context**, and a team that read the first result as permission to
connect every server available will discover the second.

The verbosity table is the part a server author controls unilaterally. The same
{512} tools cost {vb['terse'][1]:,.0f} tokens described tersely and
{vb['documentation'][1]:,.0f} written as documentation, for
{vb['terse'][0]:.1%} against {vb['documentation'][0]:.1%}.

**Description length is multiplicative in the rent**, and it is the one variable
in this listing that costs nothing to change. A server whose descriptions read
like reference documentation is charging every host that connects to it, on every
request, forever.

The retrieval table is the way out, and it is the standard answer at
cite:qin2023toolllm's scale: index the inventory, show only the schemas this
request plausibly needs. Showing {8} of {2048} gives {rt[8][1][1]:.1%} against
{rt[2048][1][1]:.1%} for showing everything.

Note the direction within that table: showing FEWER is better.
{rt[8][1][1]:.1%} at {8} shown against {rt[64][1][1]:.1%} at {64}, even though a
larger set is more likely to contain the right tool. The marginal recall is worth
less than the marginal dilution, which is the same shape ch:mcp-primitives found
for resources.

Retrieval is not free either -- the recall columns are the cost. At {0.85:.0%}
recall the whole scheme gives {rt[8][1][2]:.1%}, because the tool the task needed
was sometimes not shown. **Retrieval converts a context problem into a recall
problem**, and the recall is now the thing to measure.

The last table gives the number a host actually needs: retrieval starts winning
between {16} and {64} tools. Below that, show everything and do not build an
index. Above it, the index is not an optimisation -- at {1024} tools, showing
everything gives {xo[1024][0]:.1%}.

Which is a much smaller crossover than people expect, and it is why hosts that
connect a dozen servers without a retrieval layer degrade in a way nobody
attributes to the tool list.""")
```

The second listing asks how to divide the context.

```python {tier=A name=context-is-a-budget}
"""Allocating a context budget, which almost nobody does deliberately.

Three things compete for the same context, and each is added by a different part
of the system:

  schemas    tool descriptions, added by whichever servers are connected
  resources  ch:mcp-primitives' preloaded content, added by the host
  history    the conversation so far, added by simply continuing

Each helps, with diminishing returns, and each dilutes the others -- so the total
is capped and the SPLIT is a real decision (eq:context-is-a-budget).

In practice nobody makes it. Schemas take whatever the connected servers happen
to need, resources take what the host was configured to preload, and history
takes the rest. This listing asks what that costs against an allocation chosen on
purpose.
"""
import numpy as np

rng = np.random.default_rng(4241)

M = 30000
STEPS = 6
BUDGET = 24000          # tokens available for schemas + resources + history
BASE = 0.995
DILUTE = 2.0e-6         # per-token degradation of reasoning

# Each component's benefit saturates: the first tokens matter far more than the
# last. `scale` is the token count at which roughly 63% of the benefit is had.
COMPONENTS = {
    "schemas":   dict(scale=3000.0, weight=0.34, floor=0.30),
    "resources": dict(scale=6000.0, weight=0.40, floor=0.35),
    "history":   dict(scale=4000.0, weight=0.26, floor=0.55),
}


def component_quality(name, tokens):
    """Saturating benefit: floor with nothing, approaching 1 with plenty."""
    c = COMPONENTS[name]
    return c["floor"] + (1.0 - c["floor"]) * (1.0 - np.exp(-tokens / c["scale"]))


def run(alloc, m=M, steps=STEPS, budget=BUDGET, dilute=DILUTE):
    """`alloc` maps component -> fraction of budget. Success needs every
    component to do its job, against a reasoning quality that degrades with the
    total tokens present."""
    total = sum(alloc.values()) * budget
    q = 1.0
    for name, frac in alloc.items():
        q *= component_quality(name, frac * budget) ** (
            COMPONENTS[name]["weight"] * 3.0)
    p_step = BASE * (1.0 - dilute * total)
    ok = rng.random(m) < (q * p_step ** steps)
    return float(ok.mean()), total


def sweep_best(budget=BUDGET, grid=11, m=20000):
    """Search the simplex for the best split."""
    best, arg = -1.0, None
    for i in range(grid + 1):
        for j in range(grid + 1 - i):
            k = grid - i - j
            a = {"schemas": i / grid, "resources": j / grid,
                 "history": k / grid}
            v = run(a, m=m, budget=budget)[0]
            if v > best:
                best, arg = v, a
    return arg, best


print(f"{M:,} tasks. {BUDGET:,} tokens to divide between tool schemas,")
print("preloaded resources, and conversation history. Each helps with")
print("diminishing returns; all three dilute.")
print()
print(f"{'allocation':>34}{'schemas':>9}{'resources':>11}{'history':>9}"
      f"{'success':>10}")
print("-" * 73)
PLANS = [
    ("even thirds", {"schemas": 1 / 3, "resources": 1 / 3, "history": 1 / 3}),
    ("schema-heavy (many servers)", {"schemas": 0.6, "resources": 0.25,
                                     "history": 0.15}),
    ("resource-heavy (preload all)", {"schemas": 0.15, "resources": 0.65,
                                      "history": 0.20}),
    ("history-heavy (long chat)", {"schemas": 0.15, "resources": 0.15,
                                   "history": 0.70}),
]
tab = {}
for name, a in PLANS:
    r = run(a)
    tab[name] = r
    print(f"{name:>34}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}{r[0]:>10.1%}")

best_alloc, best_v = sweep_best()
print(f"{'best split found':>34}{best_alloc['schemas']:>9.0%}"
      f"{best_alloc['resources']:>11.0%}{best_alloc['history']:>9.0%}"
      f"{run(best_alloc)[0]:>10.1%}")

print()
print()
print("The cost of allocating by accident. Gap between the best split and each")
print("of the plausible defaults:")
print()
bv = run(best_alloc)[0]
print(f"{'allocation':>34}{'success':>10}{'gap to best':>13}")
print("-" * 57)
for name, _ in PLANS:
    print(f"{name:>34}{tab[name][0]:>10.1%}{tab[name][0] - bv:>+13.1%}")

print()
print()
print("More context is not monotonically better: the dilution term applies")
print("to every token, including the useful ones. Sweeping the total spend,")
print("each row allocated at its own best split:")
print()
print(f"{'tokens spent':>14}{'schemas':>9}{'resources':>11}{'history':>9}"
      f"{'success':>10}")
print("-" * 53)
us = {}
for B in (4000, 12000, 24000, 48000, 96000, 160000):
    a, _ = sweep_best(budget=B)
    v = run(a, budget=B)[0]
    us[B] = (a, v)
    print(f"{B:>14,}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}{v:>10.1%}")
peak = max(us, key=lambda b: us[b][1])

print()
print()
print("The split itself moves with the budget, which is what teams get wrong")
print("when they upgrade to a larger window and change nothing else.")
print()
print(f"{'tokens spent':>14}{'schemas':>9}{'resources':>11}{'history':>9}")
print("-" * 43)
for B in (4000, 12000, 24000, 48000):
    a = us[B][0]
    print(f"{B:>14,}{a['schemas']:>9.0%}{a['resources']:>11.0%}"
          f"{a['history']:>9.0%}")

print(f"""
The first table prices four allocations that all arise by accident, and the
spread between them is the finding.

The best split found is {best_alloc['schemas']:.0%} schemas,
{best_alloc['resources']:.0%} resources, {best_alloc['history']:.0%} history, at
{bv:.1%}. An even three-way split gives {tab['even thirds'][0]:.1%}, which is
respectable -- the surface is flat near its peak. The specific bad defaults are
not respectable: a schema-heavy host that connected many servers loses
{bv - tab['schema-heavy (many servers)'][0]:.1f} points, and a long conversation
that has crowded out everything else loses
{bv - tab['history-heavy (long chat)'][0]:.1f}.

**Nobody chooses these allocations. They are what is left over**
(eq:context-is-a-budget): schemas take what the connected servers need, resources
take what someone configured, and history takes the rest by simply continuing.
The history-heavy row is the one that arrives on its own, in every long session,
without any decision being made.

The second table is the result worth arguing with. Success peaks at
{peak:,} tokens and falls to {us[160000][1]:.1%} at {160000:,}.

**More context is not monotonically better**, because the dilution term applies to
every token including the useful ones. The benefit of each component saturates --
the first few thousand tokens of tool schema carry nearly all of the tool
capability -- while the cost does not saturate at all. Past the peak, additional
context is subtracting.

That is worth stating carefully, because it is easy to over-read. It does not say
large windows are useless; it says filling them is not free, and the filling is
usually automatic. A host that grew its context four times over and put four times
as much in it has moved right along this curve, not up.

The last table is the practical trap. At {4000:,} tokens the best split spends
{us[4000][0]['history']:.0%} on history -- nothing at all -- and at {24000:,} it
spends {us[24000][0]['history']:.0%}.

**The right split depends on the budget**, so a configuration tuned for a small
window is wrong for a large one in a specific direction: it under-weights history
and over-weights schemas. Teams that upgrade to a larger window and change nothing
else keep an allocation that was chosen for scarcity, and then conclude the larger
window did not help.

The general instruction is the one this listing was built to make concrete.
**Allocate the context deliberately, as a budget with named line items**, rather
than letting three subsystems each take what they want. The line items are
knowable, the curve is flat near its peak so precision is not required, and the
defaults that arise by accident are the ones furthest from it.""")
```

## 9. Practical Example

The first listing prices inventories at roughly $170$ tokens per tool:

```
   tools  schema tokens  select/step   success
----------------------------------------------
       8          1,320       98.13%     87.7%
     128         21,120       97.64%     68.2%
     512         84,480       97.39%     30.5%
    2048        337,920       97.14%      0.1%
```

Decomposed:

```
   tools  selection loss  dilution loss    total
------------------------------------------------
       8            9.0%           3.9%    12.3%
     128           11.3%          23.1%    31.8%
    2048           13.5%          99.9%    99.9%
```

**Selection loss barely moves and dilution loss explodes**
({{eq:schemas-are-rent}}). {{ch:ag-tool-calling}}'s finding is intact — the
*decision* is nearly free in inventory size — and it is a different quantity from
the *bill*.

Verbosity, at $512$ tools:

```
  description length  schema tokens   success
---------------------------------------------
               terse         33,792     58.0%
             typical         84,480     30.8%
       documentation        253,440      1.4%
```

**Description length is multiplicative in the rent**
({{eq:verbosity-is-multiplicative}}) and it is the one variable here that costs
nothing to change. A server whose descriptions read like reference documentation
charges every host that connects to it, on every request.

Retrieval over a $2{,}048$-tool inventory:

```
  shown of 2048  schema tokens  recall 99%  recall 94%  recall 85%
------------------------------------------------------------------
              8          1,320       86.8%       82.3%       73.8%
             32          5,280       82.0%       77.7%       70.3%
             64         10,560       76.8%       73.0%       65.9%
            all        337,920        0.1%        0.1%        0.1%
```

Note that showing *fewer* is better — $86.8\%$ at eight against $76.8\%$ at
sixty-four — because marginal recall is worth less than marginal dilution. And the
recall columns are the price: **retrieval converts a context problem into a recall
problem** ({{eq:retrieval-trades-context-for-recall}}), so recall becomes the thing
to measure.

The crossover:

```
  inventory   show all   retrieve 24     better
-----------------------------------------------
         16      85.6%         80.4%   show all
         64      77.5%         79.1%   retrieve
       1024       8.3%         79.1%   retrieve
```

**Retrieval starts winning between sixteen and sixty-four tools**
({{eq:retrieval-crossover-is-small}}) — one or two servers, not a marketplace.

The second listing divides $24{,}000$ tokens three ways:

```
                        allocation  schemas  resources  history   success
-------------------------------------------------------------------------
                       even thirds      33%        33%      33%     52.4%
       schema-heavy (many servers)      60%        25%      15%     44.4%
      resource-heavy (preload all)      15%        65%      20%     47.8%
         history-heavy (long chat)      15%        15%      70%     33.6%
                  best split found      27%        45%      27%     53.8%
```

An even split is within $1.8$ points of best — the surface is flat near its peak.
The *specific* accidental defaults are not: schema-heavy loses $9.8$ points and
history-heavy loses $20.6$. **The history-heavy allocation is the one that arrives
on its own**, in every long session, with no decision made
({{eq:context-is-a-budget}}).

Sweeping the total spend, each row at its own best split:

```
  tokens spent  schemas  resources  history   success
-----------------------------------------------------
         4,000      45%        55%       0%     17.2%
        24,000      27%        45%      27%     53.6%
        96,000      27%        45%      27%     27.6%
       160,000      27%        45%      27%      9.7%
```

**More context is not monotonically better.** The benefit of each component
saturates and the dilution cost does not, so past the peak additional context is
subtracting.

And the split moves with the budget:

```
  tokens spent  schemas  resources  history
-------------------------------------------
         4,000      45%        55%       0%
        12,000      36%        45%      18%
        24,000      27%        45%      27%
```

At $4{,}000$ tokens the best allocation spends *nothing* on history; by $24{,}000$
it spends over a quarter ({{eq:allocation-depends-on-budget}}). A configuration
tuned under scarcity is wrong for a large window in a specific direction.

## 10. Production Considerations

Count your schema tokens. It is a number every host can compute in a minute and
almost none reports, and it is the standing charge on every request.

Write descriptions to distinguish, not to explain. The test is whether removing a
sentence would make a wrong choice more likely.

Build retrieval past roughly thirty tools. The crossover is much earlier than
intuition suggests, and above it the index is not an optimisation.

Show few tools, described completely, rather than many described sparsely — this
keeps {{ch:ag-tool-calling}}'s reliability and pays rent only on what is shown.

Measure retrieval recall directly against tasks with known correct tools. Its
failure looks like a model failure and nothing else surfaces it.

Name your context line items and cap each. Schemas, resources, history — and
enforce the caps rather than letting each subsystem take what it wants.

Re-tune the allocation when the window changes. The split that was right under
scarcity under-weights history.

And do not fill a larger window just because it is there.

## 11. Common Mistakes

**Reading "tool count is free" as unconditional.** It is free for selection and
expensive for context.

**Writing tool descriptions as documentation.** A factor of seven in rent, paid by
everyone forever.

**Connecting every available server.** Past the crossover well before it feels
like it.

**Deferring retrieval until the inventory is large.** It starts paying at tens.

**Showing more tools to improve recall.** Marginal recall loses to marginal
dilution.

**Not measuring retrieval recall.** Its failures are attributed to the model.

**Letting history take the remainder.** The worst default and the one that needs
no decision.

**Relaxing every cap on a larger window.** Success is non-monotone in context.

## 12. Failure Modes

*Standing-charge degradation.* Every request slightly worse because of tools none
of them used, with nothing in the trace pointing at the tool list.

*Retrieval miss read as model failure.* The needed tool was never shown; the trace
shows a poor choice.

*History crowd-out.* A long session slowly losing access to its own tools and
resources, one turn at a time.

*Post-upgrade disappointment.* A larger window filled at the old proportions and
judged not to have helped.

*Verbose-server contagion.* One server's documentation-length descriptions
degrading every host that connects it, invisibly to that server's author.

## 13. Alternatives

**Fewer servers.** The simplest response, and correct more often than it is tried.

**Tool grouping behind a dispatcher.** One tool with a mode argument instead of
twenty tools — reduces rent and risks {{eq:distinctness-not-count}}'s overlap
problem, so it must be measured rather than assumed.

**Progressive disclosure.** Show a compact list, let the model request full
schemas for candidates. A two-hop version of retrieval with a cheap first hop.

**Fine-tuning on the inventory.** {{cite:patil2023gorilla}} found this goes stale
where retrieval does not, so it addresses rent at the cost of freshness.

**Per-task tool sets.** If the host knows the task type, it can present a
hand-curated set — a router over inventories, with {{ch:as-graph}}'s tail caveat.

## 14. Evaluation

Report schema tokens as a first-class metric alongside latency and cost.

Measure the marginal effect of each connected server: success with and without it,
on tasks that do not use it. That is the direct measurement of rent.

Measure retrieval recall on tasks with known correct tools, and report it
separately from task success.

Sweep the retrieval width rather than picking one. The optimum is small and the
curve is asymmetric.

Report the context split — what fraction went to schemas, resources and history —
per request. Almost nobody logs this and it is the input to every decision here.

And test at your real window size, since {{eq:allocation-depends-on-budget}} says
the answer moves.

## 15. Advanced Concepts

**Learned retrieval width.** Choosing $r$ per request from the query's specificity
rather than fixing it, which {{eq:retrieval-trades-context-for-recall}}'s structure
supports and nothing implements. {{maturity:EMERGING}}.

**Schema compression.** Shared type definitions across a server's tools, or a
compact wire form expanded only for candidates — a direct attack on
{{eq:schemas-are-rent}}'s constant.

**Budget negotiation in the protocol.** A host could advertise a token budget and
servers could return schemas at an appropriate detail level. Nothing in
{{cite:mcp2026spec}} supports this and it is a natural extension.

**Measuring $\delta$ per model.** The dilution coefficient is the parameter every
result here scales with, and it is model-specific and rarely published.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-tool-calling}}'s {{eq:distinctness-not-count}} is qualified rather than
overturned: count is free for the decision and not for the context, and the two
statements have different growth rates.

{{ch:ag-memory}}'s dilution is the mechanism behind every result here, and this is
where its coefficient becomes load-bearing.

{{ch:mcp-primitives}}'s decision-versus-dilution trade is the same curve viewed
from the resource side; this chapter adds the third claimant and makes it a budget.

{{cite:patil2023gorilla}}'s retrieval-over-documentation result, cited in
{{ch:mcp-why}} as the argument for runtime discovery, is the mechanism that makes
a large inventory survivable.

Ahead: {{ch:mcp-security}} takes up what else those schemas carry, since a
description is text that reaches the model and
{{cite:huang2026mcpthreat}} found that channel to be the dominant client-side
vulnerability.

## 17. Exercises

1. Compute the schema token count for your own connected servers. Where does it
   put you in the first table?

2. Rewrite one server's tool descriptions to the distinguish-only standard and
   measure the token reduction.

3. Derive the optimal retrieval width from
   {{eq:retrieval-trades-context-for-recall}} given a recall curve, and check it
   against the listing.

4. Implement progressive disclosure — compact list then full schema on request —
   and price the extra hop.

5. Estimate $\delta$ for a model you use, by measuring task success against
   injected irrelevant context.

6. Add a fourth claimant, the system prompt, to the second listing. How much does
   it displace?

## 18. Interview Questions

1. {{ch:ag-tool-calling}} says tool count is nearly free. Why is a
   two-thousand-tool inventory unusable?

2. What is the highest-leverage thing a server author controls?

3. At what inventory size would you build tool retrieval?

4. Would you show eight tools or sixty-four? Why?

5. Your retrieval layer has $85\%$ recall. What does that look like in a trace?

6. You moved to a window eight times larger and quality fell. What happened?

## 19. Research Questions

1. Can retrieval width be chosen per request from query specificity?

2. What is the right compact wire form for schemas, and what does expansion cost?

3. Should the protocol carry a host token budget, and would servers honour it?

4. How does the dilution coefficient vary across current models, and is it
   published anywhere?

5. Does tool grouping behind a dispatcher trade rent for overlap at a favourable
   rate, or an unfavourable one?

## 20. Chapter Summary

A tool's schema is text in the context, paid for on every request whether the tool
is used or not ({{eq:schemas-are-rent}}). Across eight to two thousand tools,
selection loss moved $9.0\% \to 13.5\%$ and dilution loss moved
$3.9\% \to 99.9\%$. **Tool count is nearly free for selection and ruinous for
context** — {{ch:ag-tool-calling}}'s finding intact, and a different quantity.

Verbosity multiplies that rent: $512$ tools cost $33{,}792$ tokens terse and
$253{,}440$ as documentation, for $58.0\%$ against $1.4\%$
({{eq:verbosity-is-multiplicative}}). It is the one variable a server author
controls unilaterally and it costs nothing to change.

Retrieval is the escape and it converts a context problem into a recall problem
({{eq:retrieval-trades-context-for-recall}}). Showing eight of two thousand gave
$82.3\%$ against $0.1\%$ for showing all — and showing *fewer* beat showing more,
because marginal recall loses to marginal dilution. **The crossover is between
sixteen and sixty-four tools** ({{eq:retrieval-crossover-is-small}}), which is one
or two servers rather than a marketplace.

Schemas compete with preloaded resources and conversation history for one budget,
all saturating, all diluting ({{eq:context-is-a-budget}}). An even split was
within $1.8$ points of optimal, but the accidental defaults were not:
schema-heavy lost $9.8$ points and **history-heavy — the allocation that arrives
on its own — lost $20.6$.**

And two warnings about size. Success peaked near $24{,}000$ tokens and fell to
$9.7\%$ at $160{,}000$: **more context is not monotonically better**, because
benefits saturate and dilution does not. The optimal split also moves with the
budget ({{eq:allocation-depends-on-budget}}) — at $4{,}000$ tokens the best
allocation spends nothing on history — so a configuration tuned under scarcity is
wrong for a large window, and a team that upgrades and changes nothing else will
conclude the window did not help.

## 21. Further Reading

{{cite:qin2023toolllm}} and {{cite:li2023apibank}} for inventories at the scale
that makes retrieval unavoidable, and {{cite:patil2023gorilla}} for the
retrieval-over-documentation result this chapter's escape route depends on.

{{cite:schick2023toolformer}} for the tool-selection competence that
{{eq:distinctness-not-count}} measures and this chapter charges rent on.

{{cite:mcp2026spec}} for `server/discover` and for what the protocol deliberately
leaves to the host, which is exactly the retrieval layer
{{sec:7-internal-mechanics}} argues most hosts are missing.

{{ch:ag-tool-calling}} and {{ch:ag-memory}} for the two results this chapter sits
between, and {{ch:mcp-primitives}} for the other claimant on the same budget.
