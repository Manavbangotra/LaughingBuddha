---
id: ag-tool-calling
number: 154
part: XVII
tier: full
status: draft
requires: [control-location, translation-versus-execution,
           boundary-crossing-cost]
provides: [four-decisions, distinctness-not-count, signature-cost,
           optional-is-not-free, error-message-as-selector,
           enumerate-before-remove]
citations: [schick2023toolformer, gao2023pal, yao2023react, liu2024agentbench,
            zhou2024webarena, greshake2023indirect, sprague2024tocot]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose tool-call accuracy into
four independently measurable stages and say which one your system is losing to;
explain why tool *count* is nearly free and tool *overlap* is expensive, and
measure the difference on your own inventory; predict the effect of a signature
change before making it; state why an optional argument is not optional from the
model's point of view; and explain why a tool's error message is the highest-return
line of code in an agent system.

## 2. Why This Matters

{{ch:ag-what-is-an-agent}} treated an agent's per-step accuracy as a single
number and showed that everything depends on it: task success is $p^k$, the
architecture crossover moves with $p$, and the cost distribution's tail is
governed by how often steps fail. This chapter is about where $p$ comes from.

The answer is not "the model". {{cite:schick2023toolformer}} decomposes tool use
into four decisions — which API to call, when to call it, what arguments to pass,
and how to incorporate the result — and {{sec:9-practical-example}} measures them
separately. Most of what moves them is in the tool inventory and the tool
signatures, which are things you control, rather than in the weights, which are
not.

Two of this chapter's findings contradicted the hypothesis they were written to
test, and both change what you would work on.

The first is that **tool count is nearly free**. The standard advice to keep the
tool list short is given as though selection gets harder with more options.
{{sec:9-practical-example}} finds selection at $100.0\%$ with four tools and
$100.0\%$ with $128$, because in a semantic space of any real dimensionality,
genuinely distinct things stay far apart. What destroys selection is *overlap*:
the same $128$ tools forced into two families score $67.1\%$. Count and
distinctness correlate in practice — inventories that got large got large by
accretion — and the advice attaches to the wrong one.

The second is that **the error message matters more than the signature**.
{{sec:9-practical-example}} measures three retries against an opaque error
recovering $0.9$ points and the same three retries against a message that names
the bad field and lists valid values recovering $16.1$. A bad signature with good
errors beats a good signature with bad errors, *and makes fewer calls doing it*.

Both findings are about the tool rather than the model, which is the chapter's
point. And both compound: {{ch:ag-what-is-an-agent}}'s $p^k$ means a per-call
difference of a few points is a task difference of tens.
{{sec:9-practical-example}} measures task success across five tool calls at
$43.94\%$ for the worst design and $99.99\%$ for the best.

## 3. Prerequisites

You need {{ch:ag-what-is-an-agent}}'s framing — that per-step accuracy compounds
over a run — because every number in this chapter gets raised to the power of the
horizon.

From {{ch:rsn-tool-assisted}} you need the translation-versus-execution
distinction and the boundary-crossing cost. That chapter established *whether* to
use a tool; this one is about making the call succeed once you have decided to.

From {{part:8}}, constrained decoding: the strongest intervention in this chapter
is making an invalid call unrepresentable rather than unlikely, and that is the
mechanism.

## 4. Intuitive Explanation

A tool call is four decisions in a row, and they fail for different reasons.

**Which tool.** The model has a request and a list of tools, and has to pick one.
This is a matching problem: the request describes a need, each tool describes a
capability, and the model finds the closest match.

**Whether to call at all.** Some requests need a tool and some do not, and calling
one unnecessarily is its own failure — it wastes a step and injects an irrelevant
observation into the context.

**What arguments.** Having chosen, the model must fill in the parameters. This is
where most of the failures are once the first decision is working, and it is the
most controllable.

**What the result means.** The tool returns something, and the model has to read
it correctly and decide what to do next.

Now, the matching problem. The intuition that more tools means harder matching
comes from thinking about it as a list you scan. But it is not a scan — it is a
nearest-neighbour lookup in a space with many dimensions, and high-dimensional
spaces are extremely roomy. Two hundred randomly chosen capabilities are all far
apart from each other. Adding the two hundred and first does not crowd anything.

What *does* crowd is similarity. `search_docs`, `search_tickets`,
`search_code`, `search_docs_v2` are four points close together, and a request that
lands anywhere near them could match any of the four. That is a hard problem, and
it does not get easier by having fewer *other* tools elsewhere in the space.

This distinction matters because the two have different remedies. If count were
the problem, you would remove tools — which removes capability. If overlap is the
problem, you merge or differentiate the overlapping ones, which does not.

Arguments are simpler and more mechanical. Every argument the model has to compose
is a chance to compose it wrong, and the chances multiply. Two things follow.

Constraining an argument helps more than removing it. A field with five allowed
values is nearly free to get right; a free-text field is not. So enumerate before
you delete — you keep the capability and lose the failure.

And an optional argument is not free. From the model's point of view "optional"
does not mean "ignore this"; it means "decide whether to supply this", which is a
decision with two branches and a chance of getting the value wrong on one of them.
Six optional parameters are six such decisions.

Finally, the thing that is almost never designed: what the tool says when it
refuses.

An agent that gets a call wrong will try again. If the error said `error`, the
second attempt is a fresh draw from the same distribution — it succeeds at the
same rate the first one did, so retrying buys almost nothing. If the error said
*which field was wrong and what values are acceptable*, the second attempt is
conditioned on that information and is a completely different draw.

That is the same coverage-versus-selection distinction from
{{ch:rsn-test-time-compute}}, arriving in the humblest possible place. An
informative error message is a selector, it costs one string, and it is the
cheapest one in this book.

## 5. Formal Explanation

Following {{cite:schick2023toolformer}}, decompose a call into four stages with
independent success probabilities:

$$p_{\text{call}} = p_{\text{select}} \cdot p_{\text{when}} \cdot p_{\text{args}} \cdot p_{\text{read}}$$ (eq:four-decisions)

The value of the decomposition is that the four have different dependencies. Only
$p_{\text{select}}$ is a function of the inventory; $p_{\text{args}}$ is a
function of the signature; $p_{\text{read}}$ is a function of the response format.
An aggregate "tool use accuracy" is a product of four numbers with four different
remedies.

Model selection as nearest-neighbour retrieval. Tools occupy points
$t_1, \ldots, t_n$ in a $D$-dimensional semantic space, and a request for tool $i$
arrives at $t_i + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.
Selection succeeds when:

$$\|\varepsilon\| < \tfrac{1}{2}\min_{j \neq i}\|t_i - t_j\|$$ (eq:distinctness-not-count)

approximately. The governing quantity is the *nearest-neighbour distance*, not
$n$. For points drawn uniformly in $D$ dimensions that distance shrinks like
$n^{-1/D}$, which for $D = 24$ means doubling the inventory shrinks it by $3\%$ —
negligible. But if tools are drawn from $c$ clusters, the nearest-neighbour
distance is set by the *within-cluster* spread and does not depend on $n$ at all,
only on $c$ and the cluster radius.

**So count enters through a $n^{-1/D}$ term that is nearly flat, and overlap
enters through a term that does not involve $n$.** That is the formal version of
this chapter's first finding.

For arguments, let $r$ be the number of required arguments, $o$ the number of
optional ones supplied with probability $s$, and $p_a$ the per-argument success
rate:

$$p_{\text{args}} = p_a^{\,r} \cdot \big(1 - s(1 - p_a)\big)^{o}$$ (eq:signature-cost)

The second factor is the cost of optionality. An optional argument contributes
$1 - s(1 - p_a)$ rather than $1$, so it is free only when $s = 0$ or $p_a = 1$.
With $s = 0.5$, an optional argument costs half of what a required one costs —
which is much closer to "required" than to "free", and is why
{{eq:optional-is-not-free}} deserves its own name:

$$\text{cost of an optional argument} = \tfrac{s}{1} \times \text{cost of a required one}$$ (eq:optional-is-not-free)

Now retries. Let $p_0$ be first-call success and $\phi$ the probability that a
failed call's error message lets the model locate the fault. After $m$ retries:

$$p_m = 1 - (1 - p_0)\,(1 - \phi)^{m}$$ (eq:error-message-as-selector)

and the expected number of calls is $\sum_{j=0}^{m}(1 - p_j)$ plus one. Note the
shape: $\phi$ enters as a base raised to the retry count, exactly as
{{eq:chain-accuracy-compounds}} does, so improving the error message has a
*geometric* return in retries while improving $p_0$ has a linear one.

That is the formal reason {{sec:9-practical-example}}'s error-message effect is
larger than any signature effect: they enter the same equation in different
places.

## 6. Mathematical Foundation

Three consequences worth extracting.

**Enumerate before you remove.** From {{eq:signature-cost}}, dropping one required
argument multiplies $p_{\text{args}}$ by $1/p_a$; enumerating all $r$ of them
replaces $p_a^r$ with $p_e^r$. The second wins whenever:

$$\left(\frac{p_e}{p_a}\right)^{r} > \frac{1}{p_a} \quad\Longleftrightarrow\quad r \ln\frac{p_e}{p_a} > \ln\frac{1}{p_a}$$ (eq:enumerate-before-remove)

which holds for $r \ge 2$ at any realistic $p_e$ and $p_a$. Since removing an
argument removes capability and constraining one does not, the ordering is
convenient as well as correct.

**The horizon is an exponent on everything.** A per-call improvement $\delta$
becomes a task improvement of roughly $k\delta$ near $p = 1$, and much more when
$p$ is lower. {{sec:9-practical-example}} measures $84.84\%$ against $99.99\%$
per call becoming $43.94\%$ against $99.99\%$ across five calls. **Tool design is
an agent concern rather than an API concern**, and nothing in an API's own metrics
would reveal it.

**Retries and coverage are the same object.** {{eq:error-message-as-selector}} with
$\phi = 0$ reduces to $p_m = p_0$ — retrying an opaque failure is sampling without
a selector, which {{ch:rsn-test-time-compute}} showed produces coverage you cannot
cash in. With $\phi > 0$ the retry is *conditioned*, which is what a selector
does. An error message is a selector implemented as a string.

That correspondence also bounds it. $\phi$ is the probability that the message
identifies the fault, so it is capped by how much the tool actually knows about
why it refused. A validation layer that checks fields in order and reports the
first failure has a higher $\phi$ than one that returns a boolean, and neither
requires a model.

## 7. Internal Mechanics

### 7.1 What a tool description is for

A tool description does two jobs that pull in different directions. It has to make
the tool *findable* — close to the requests it should serve — and *distinguishable*
— far from the tools it should not be confused with. Teams write descriptions for
the first and are surprised by failures caused by the second.

The measurable form: embed your tool descriptions and compute each one's nearest
neighbour distance. Two tools whose descriptions are close are two tools the model
will confuse, and the fix is to write into the descriptions what makes them
*different*, which is usually a clause about when *not* to use each one.

### 7.2 Why "when to call" is a separate failure

Deciding whether a tool is needed at all is a different computation from deciding
which one. It fails in two directions — calling unnecessarily, and failing to call
when needed — and the two have different costs.

An unnecessary call wastes a step and injects an irrelevant observation, which
degrades the next decision. A missing call produces a confidently wrong answer
from parametric knowledge, which is worse and harder to detect. Systems that tune
this decision usually tune it toward calling more, which trades the second failure
for the first, and that is generally right — but it should be a deliberate
choice, because it changes the cost profile.

### 7.3 The response format is part of the tool

$p_{\text{read}}$ in {{eq:four-decisions}} is a property of what the tool returns.
A response that is long, nested, inconsistently shaped, or contains fields the
agent does not need is harder to read correctly *and* occupies context that the
rest of the run must carry ({{part:15}}'s growing KV cache).

The design rule that follows: return the smallest thing that answers the question,
in a consistent shape, with the field the agent needs at the top. This is not
aesthetics — it is $p_{\text{read}}$ and it is context cost, and both compound
over a run.

### 7.4 Where constrained decoding fits

{{part:8}}'s constrained decoding is the strongest intervention available here,
because it changes an argument from *unlikely to be wrong* to *impossible to be
wrong*. A JSON schema with enumerated values, enforced at the sampler, sets the
structural part of $p_{\text{args}}$ to 1.

What it cannot fix is semantic correctness: the model can emit a perfectly valid
call with the wrong values. So constrained decoding removes a class of failures
entirely and leaves another class untouched, and knowing which of your failures
are which is worth measuring before you invest in either.

### 7.5 Tools as an attack surface

Every tool is a capability that a successful prompt injection can reach.
{{cite:greshake2023indirect}} showed that retrieved content can control which
APIs an application calls, so the tool inventory is also the blast radius, and
{{ch:ag-security}} treats it that way.

Two design consequences belong here rather than there, because they are signature
decisions. An argument that can name an arbitrary resource is more dangerous than
one drawn from an enumeration — the same change that raises $p_{\text{args}}$
narrows the attack surface. And a tool that does one thing has a smaller blast
radius than one with a `mode` parameter, so the same accretion that hurts
selection also concentrates risk.

## 8. Implementation

Two listings. The first decomposes tool-call failure into the four stages and
measures each against inventory size and against overlap. The second measures what
a signature and an error message are worth.

```python {tier=A name=four-decisions}
"""Tool-call accuracy is four numbers, and only one of them moves with scale.

cite:schick2023toolformer decomposes tool use into four decisions: which API to
call, when to call it, what arguments to pass, and how to incorporate the result.
Systems report one aggregate "tool use accuracy" that multiplies all four
together, which hides the fact that they behave completely differently as a
system grows (eq:four-decisions).

This listing measures each stage separately against the size of the tool
inventory. Selection is modelled as what it is -- a nearest-neighbour
discrimination in a semantic space, where a request lands near its correct tool
and the model picks the closest one. Everything else is modelled as a per-call
property that has no reason to depend on how many other tools exist.

The prediction going in was that selection degrades with inventory size. It does
not, and the reason it does not is the useful finding: in a space of any real
dimensionality, genuinely distinct tools stay far apart no matter how many you
add. What destroys selection is OVERLAP, and inventories acquire overlap by
growing the way real ones do.
"""
import numpy as np

rng = np.random.default_rng(1597)

D = 24                  # semantic dimensions
N_REQ = 30000
SIGMA = 0.55            # how imprecisely a request locates its tool
P_WHEN = 0.97           # decides correctly WHETHER to call a tool
P_ARGS_BASE = 0.985     # gets one argument right
P_READ = 0.98           # reads the result back correctly


def selection_accuracy(n_tools, cluster=0):
    """Tools live at points in a semantic space; a request is its tool's point
    plus noise; the model picks the nearest tool. `cluster` groups tools into
    families whose descriptions overlap, which is what a real inventory looks
    like once it has grown by accretion."""
    if cluster:
        centres = rng.normal(size=(cluster, D))
        assign = rng.integers(0, cluster, size=n_tools)
        tools = centres[assign] + 0.35 * rng.normal(size=(n_tools, D))
    else:
        tools = rng.normal(size=(n_tools, D))
    tgt = rng.integers(0, n_tools, size=N_REQ)
    req = tools[tgt] + SIGMA * rng.normal(size=(N_REQ, D))
    # Nearest tool by squared distance.
    d = ((req[:, None, :] - tools[None, :, :]) ** 2).sum(2)
    return float(np.mean(d.argmin(1) == tgt))


SIZES = [4, 8, 16, 32, 64, 128]
N_ARGS = 3

print(f"{N_REQ} requests. A request locates its tool to within noise {SIGMA};")
print(f"the model calls the nearest tool. Each call needs {N_ARGS} arguments,")
print(f"each right {P_ARGS_BASE:.1%} of the time. Deciding whether to call at")
print(f"all is {P_WHEN:.0%} accurate and reading the result back {P_READ:.0%}.")
print()
print(f"{'tools':>8}{'select':>10}{'when':>9}{'arguments':>12}{'read':>9}"
      f"{'end to end':>13}")
print("-" * 61)

p_args = P_ARGS_BASE ** N_ARGS
stages = {}
for n in SIZES:
    sel = selection_accuracy(n)
    e2e = sel * P_WHEN * p_args * P_READ
    stages[n] = (sel, P_WHEN, p_args, P_READ, e2e)
    print(f"{n:>8}{sel:>10.1%}{P_WHEN:>9.1%}{p_args:>12.1%}{P_READ:>9.1%}"
          f"{e2e:>13.1%}")

print()
print()
print("Which stage is costing the most? Errors attributable to each, as a share")
print("of all end-to-end failures.")
print()
print(f"{'tools':>8}{'select':>10}{'when':>9}{'arguments':>12}{'read':>9}")
print("-" * 48)
for n in SIZES:
    sel = stages[n][0]
    losses = np.array([1 - sel, 1 - P_WHEN, 1 - p_args, 1 - P_READ])
    share = losses / losses.sum()
    print(f"{n:>8}{share[0]:>10.1%}{share[1]:>9.1%}{share[2]:>12.1%}"
          f"{share[3]:>9.1%}")

print()
print()
print("What happens when the inventory grows by accretion, so tools cluster")
print("into families with overlapping descriptions?")
print()
print(f"{'tools':>8}{'distinct':>12}{'8 families':>13}{'4 families':>13}"
      f"{'2 families':>13}")
print("-" * 59)
clus = {}
for n in (16, 32, 64, 128):
    row = [selection_accuracy(n)]
    for c in (8, 4, 2):
        row.append(selection_accuracy(n, cluster=c))
    clus[n] = row
    print(f"{n:>8}{row[0]:>12.1%}{row[1]:>13.1%}{row[2]:>13.1%}"
          f"{row[3]:>13.1%}")

print()
print()
print("Where does tool COUNT start to matter? Sweep how precisely a request")
print("locates its tool, on a distinct (non-overlapping) inventory.")
print()
print(f"{'noise':>8}" + "".join(f"{str(n) + ' tools':>12}" for n in (8, 32, 128)))
print("-" * 44)
SIG_SAVE = SIGMA
noise_tab = {}
for sg in (0.55, 0.9, 1.3, 1.8, 2.5):
    SIGMA = sg
    row = [selection_accuracy(n) for n in (8, 32, 128)]
    noise_tab[sg] = row
    print(f"{sg:>8.2f}" + "".join(f"{v:>12.1%}" for v in row))
SIGMA = SIG_SAVE

print()
print()
print("Four ways to spend an engineering week, on a realistic inventory:")
print("64 tools that have grown into 2 overlapping families.")
print()
print(f"{'intervention':>40}{'selection':>12}{'end to end':>13}")
print("-" * 65)
base_sel = selection_accuracy(64, cluster=2)
interv = {}


def price(name, sel, na, pa):
    e = sel * P_WHEN * (pa ** na) * P_READ
    interv[name] = (sel, e)
    print(f"{name:>40}{sel:>12.1%}{e:>13.1%}")


price("baseline (64 tools, 2 families)", base_sel, N_ARGS, P_ARGS_BASE)
price("halve the inventory, same overlap", selection_accuracy(32, cluster=2),
      N_ARGS, P_ARGS_BASE)
price("split into 8 families, same count", selection_accuracy(64, cluster=8),
      N_ARGS, P_ARGS_BASE)
price("make all 64 genuinely distinct", selection_accuracy(64), N_ARGS,
      P_ARGS_BASE)
price("drop one required argument", base_sel, N_ARGS - 1, P_ARGS_BASE)
price("constrain args to enums (98.5->99.8%)", base_sel, N_ARGS, 0.998)

s4, s128 = stages[4][0], stages[128][0]
e4, e128 = stages[4][4], stages[128][4]
print(f"""
The first table is the decomposition, and the column that was supposed to move
does not.

Selection is {s4:.1%} at {4} tools and {s128:.1%} at {128}. That was not the
expected result, and the reason for it is worth more than the result I was
looking for: in {D} semantic dimensions, randomly placed points are
overwhelmingly likely to be far apart, so adding tools does not crowd the space.
Distinguishing a request's tool from {127} unrelated alternatives is barely
harder than distinguishing it from three.

**Tool count, on its own, is nearly free.** The advice to "keep the tool list
short" is not wrong for context-window reasons, and it is wrong for the reason it
is usually given.

The second table follows from the first: with selection contributing nothing,
failures are dominated by argument construction at
{(1 - p_args) / ((1 - s4) + (1 - P_WHEN) + (1 - p_args) + (1 - P_READ)):.1%} of
the total, then the when-to-call decision at
{(1 - P_WHEN) / ((1 - s4) + (1 - P_WHEN) + (1 - p_args) + (1 - P_READ)):.1%}, then
reading the result. **Three arguments at {P_ARGS_BASE:.1%} each cost more than
choosing among {128} tools.**

The third table is where selection actually breaks, and it is the mechanism the
first table ruled out being replaced by the one that matters.

Real inventories do not grow by adding unrelated tools. They grow by accretion:
`search_docs`, `search_tickets`, `search_code`, `search_docs_v2`. Forcing {64}
tools into {2} families takes selection from {clus[64][0]:.1%} to
{clus[64][3]:.1%}, and {128} tools into {2} families gives
{clus[128][3]:.1%}.

Compare the two effects at the same inventory size. Quadrupling from {16} to
{64} distinct tools costs {clus[16][0] - clus[64][0]:.1%}. Taking {64} tools from
distinct to two families costs {clus[64][0] - clus[64][3]:.1%}.

**Tool count is not the variable. Tool DISTINCTNESS is**, and the two are
routinely confused because they correlate in practice -- inventories that got
large got large by accretion. The metric to track is not the size of the list but
how separable the descriptions are, which you can measure directly by embedding
them and looking at nearest-neighbour distances.

The fourth table says where the boundary is, because "count does not matter" is
only true while the space is roomy. At noise {0.55} selection is
{noise_tab[0.55][2]:.1%} even at {128} tools. At noise {1.8} it is
{noise_tab[1.8][0]:.1%} at {8} tools and {noise_tab[1.8][2]:.1%} at {128}.

So count matters exactly when the request does NOT locate its tool precisely, and
the two failure modes -- vague requests and overlapping tools -- are the same
failure mode seen from opposite ends. Both are distances in the same space.

The last table prices the interventions on a realistic inventory: {64} tools in
{2} families, which is what an inventory looks like after two years.

Halving the inventory while keeping the overlap buys
{interv['halve the inventory, same overlap'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}
end to end. Splitting the same {64} tools into {8} distinguishable families buys
{interv['split into 8 families, same count'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.
Making all {64} genuinely distinct buys
{interv['make all 64 genuinely distinct'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.

Against those, dropping one required argument buys
{interv['drop one required argument'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}
and constraining the arguments to enumerated values buys
{interv['constrain args to enums (98.5->99.8%)'][1] - interv['baseline (64 tools, 2 families)'][1]:+.1%}.

The ordering is the practical output. **Deduplicating an overlapping inventory is
worth several times what shortening it is, and both are worth more than argument
work once overlap is present** -- but with overlap removed, argument construction
is the whole remaining budget, which is the second table's finding.

One thing this listing does not model, and it is the largest omission. Selection
here is one decision. In an agent the model selects repeatedly, so selection
error compounds over the run in the way ch:rsn-cot's
eq:chain-accuracy-compounds describes: an inventory that costs
{clus[64][3]:.1%} per selection costs {clus[64][3] ** 5:.1%} over five steps
against {clus[64][0] ** 5:.1%} for a distinct one. **The overlap penalty is
raised to the power of the horizon**, which is why it matters far more in an
agent than in a single function call, and why an inventory that is fine for a
chat assistant can be unusable for an agent.""")
```

The second listing holds the inventory fixed and varies the tool itself.

```python {tier=A name=signature-and-errors}
"""Tool design: the signature, and the error message.

The previous listing found argument construction to be the whole remaining
failure budget once tool overlap is removed. This one asks what a tool's DESIGN
does to that budget (eq:signature-cost), and then measures the design decision
that is almost never treated as one: what the tool says when it fails.

An agent that gets a call wrong will try again. Whether the second attempt is
better than the first depends entirely on whether the error told it anything, and
that is a property of the tool rather than of the model.
"""
import numpy as np

rng = np.random.default_rng(1663)

N = 60000
P_FREE = 0.965          # an argument the model must compose freely
P_ENUM = 0.995          # an argument chosen from an enumerated set
MAX_RETRIES = 3


def first_call(n_req, n_opt, p_req, p_opt=None):
    """Probability the call is well formed. Required arguments must all be
    right; optional ones only matter when the model chooses to supply them,
    which it does half the time."""
    p_opt = p_req if p_opt is None else p_opt
    supplied = rng.random((N, n_opt)) < 0.5 if n_opt else np.zeros((N, 0), bool)
    ok_req = (rng.random((N, n_req)) < p_req).all(1) if n_req else np.ones(N, bool)
    ok_opt = ((rng.random((N, n_opt)) < p_opt) | ~supplied).all(1) \
        if n_opt else np.ones(N, bool)
    return ok_req & ok_opt


print(f"A well-formed call needs every required argument right. A free-text")
print(f"argument is right {P_FREE:.1%} of the time; an enumerated one")
print(f"{P_ENUM:.1%}. Optional arguments are supplied half the time and can")
print("only hurt when they are.")
print()
print(f"{'signature':>34}{'first call':>13}{'vs 3 free':>12}")
print("-" * 59)
SIGS = [
    ("3 required, free text", 3, 0, P_FREE),
    ("2 required, free text", 2, 0, P_FREE),
    ("1 required, free text", 1, 0, P_FREE),
    ("3 required, enumerated", 3, 0, P_ENUM),
    ("2 required + 2 optional, free", 2, 2, P_FREE),
    ("2 required + 6 optional, free", 2, 6, P_FREE),
    ("2 required + 6 optional, enum", 2, 6, P_ENUM),
]
sig = {}
base = None
for name, nr, no, p in SIGS:
    v = float(first_call(nr, no, p).mean())
    if base is None:
        base = v
    sig[name] = v
    print(f"{name:>34}{v:>13.1%}{v - base:>+12.1%}")

print()
print()
print("Now let the agent retry. A tool's error message either identifies the")
print("problem, hints at it, or says nothing. Success after up to")
print(f"{MAX_RETRIES} retries, on the '2 required + 6 optional, free' signature:")
print()
QUALITY = [("opaque ('error')", 0.02),
           ("generic ('invalid request')", 0.20),
           ("names the bad field", 0.62),
           ("names it and lists valid values", 0.93)]
print(f"{'error message':>34}" + "".join(f"{'try ' + str(k):>10}"
                                         for k in range(MAX_RETRIES + 1)))
print("-" * 74)
p0 = sig["2 required + 6 optional, free"]
retry = {}
for name, fix in QUALITY:
    cum, p = [p0], p0
    for _ in range(MAX_RETRIES):
        # A retry succeeds if the message let the model locate the fault.
        p = p + (1 - p) * fix
        cum.append(p)
    retry[name] = cum
    print(f"{name:>34}" + "".join(f"{v:>10.1%}" for v in cum))

print()
print()
print("Two budgets for the same end-to-end target. Which is cheaper: a better")
print("signature, or a better error message?")
print()
print(f"{'design':>44}{'calls made':>13}{'success':>11}")
print("-" * 67)


def cost_and_success(p_first, fix, tries=MAX_RETRIES):
    p, calls, succ = p_first, 1.0, p_first
    for _ in range(tries):
        calls += (1 - p)          # a retry happens only if the last call failed
        p = p + (1 - p) * fix
    return calls, p


plans = [
    ("6 optional args, opaque errors", sig["2 required + 6 optional, free"],
     0.02),
    ("6 optional args, good errors", sig["2 required + 6 optional, free"],
     0.93),
    ("no optional args, opaque errors", sig["2 required, free text"], 0.02),
    ("no optional args, good errors", sig["2 required, free text"], 0.93),
    ("enumerated args, opaque errors", sig["3 required, enumerated"], 0.02),
]
plan = {}
for name, pf, fx in plans:
    c, s = cost_and_success(pf, fx)
    plan[name] = (c, s)
    print(f"{name:>44}{c:>13.2f}{s:>10.2%}")

print()
print()
print("And what it costs across an agent run. Five tool calls per task, each")
print("with retries, under the same designs.")
print()
print(f"{'design':>44}{'task success':>14}{'calls':>9}")
print("-" * 67)
for name, pf, fx in plans:
    c, s = cost_and_success(pf, fx)
    print(f"{name:>44}{s ** 5:>14.2%}{5 * c:>9.1f}")

opa = retry["opaque ('error')"]
good = retry["names it and lists valid values"]
print(f"""
The first table is the signature, and the ordering is not subtle.

Going from three required free-text arguments to one buys
{sig['1 required, free text'] - sig['3 required, free text']:+.1%}. Keeping three
but enumerating them buys
{sig['3 required, enumerated'] - sig['3 required, free text']:+.1%} -- more,
because {P_ENUM:.1%} per argument cubed still beats {P_FREE:.1%} per argument
once.

**Constraining an argument's range is worth more than removing it**, which is
convenient, because removing arguments removes capability and constraining them
does not.

The optional-argument rows are the ones worth staring at. Adding six optional
arguments to a two-required signature costs
{sig['2 required + 6 optional, free'] - sig['2 required, free text']:.1%}, even
though the model supplies each of them only half the time and is never obliged to
supply any. An optional argument is not free. It is a coin flip on whether the
model creates an opportunity to be wrong, and six of them are six coin flips.

Enumerating those same six recovers most of it:
{sig['2 required + 6 optional, enum'] - sig['2 required + 6 optional, free']:+.1%}.

The second table is the design decision that is almost never treated as one.

With an opaque error message, three retries take success from {opa[0]:.1%} to
{opa[3]:.1%} -- the retries are independent draws from the same distribution and
buy {opa[3] - opa[0]:.1%}. With a message that names the bad field and lists the
valid values, the same three retries take it to {good[3]:.1%}.

**The error message is worth {good[3] - opa[3]:.1%}, which is more than any
signature change in the first table.** And it costs nothing at inference time: it
is a string the tool already had to construct in order to reject the call.

The reason the effect is so large is worth stating precisely. A retry after an
opaque failure is a fresh sample from the same distribution, so it succeeds at
the original rate -- ch:rsn-test-time-compute's coverage, with no selector. A
retry after an informative failure is CONDITIONED on the fault, so it is a
different and much better distribution. **An error message is the cheapest
selector in this book**, and it is the only one that requires no model at all.

The third table prices designs against each other, and it contains the practical
recommendation.

Six optional arguments with opaque errors reaches {plan['6 optional args, opaque errors'][1]:.2%}
in {plan['6 optional args, opaque errors'][0]:.2f} calls. The same signature with
good errors reaches {plan['6 optional args, good errors'][1]:.2%} in
{plan['6 optional args, good errors'][0]:.2f}. Removing the optional arguments
and keeping opaque errors reaches {plan['no optional args, opaque errors'][1]:.2%}.

So a bad signature with good errors beats a good signature with bad errors, by
{plan['6 optional args, good errors'][1] - plan['no optional args, opaque errors'][1]:.1%}
-- and it does so while making FEWER calls, because most of its retries succeed
on the first attempt after the fault is named.

The last table is the same comparison across an agent run of five tool calls, and
it is where the numbers stop being incremental. Task success goes
{plan['6 optional args, opaque errors'][1] ** 5:.1%} for the worst design against
{plan['no optional args, good errors'][1] ** 5:.1%} for the best. **Per-call
differences of a few points become task differences of tens of points**, because
five calls compound (ch:rsn-cot's eq:chain-accuracy-compounds, again).

That exponent is the reason tool design is an agent concern rather than an API
concern. A tool whose call succeeds {sig['2 required + 6 optional, free']:.0%}
of the time is a perfectly reasonable API and a bad agent tool, and nothing about
the API's own metrics would tell you.

Three rules follow, in order of what they buy.

Enumerate every argument that can be enumerated, and make the enumeration part of
the schema so constrained decoding (part:8) can enforce it. This removes the
failure rather than reducing it.

Make every error message name the field and list the acceptable values. It is the
highest-return line of code in an agent system and it is usually written by
whoever was in a hurry.

And treat every optional argument as a required decision. If the model has to
decide whether to supply it, it is not optional from the model's point of view --
it is a required decision with two branches, and it costs like one.""")
```

## 9. Practical Example

The first listing places tools at points in a $24$-dimensional semantic space,
puts each request near its correct tool with noise $0.55$, and has the model call
the nearest one.

```
   tools    select     when   arguments     read   end to end
-------------------------------------------------------------
       4    100.0%    97.0%       95.6%    98.0%        90.8%
      32    100.0%    97.0%       95.6%    98.0%        90.8%
     128    100.0%    97.0%       95.6%    98.0%        90.8%
```

The column that was supposed to move does not. Selection is $100.0\%$ at four
tools and $100.0\%$ at $128$, because in $24$ dimensions randomly placed points
are overwhelmingly likely to be far apart ({{eq:distinctness-not-count}}).
**Tool count, on its own, is nearly free.**

With selection contributing nothing, the failure budget is dominated by argument
construction at $47.0\%$ of all failures, then the when-to-call decision at
$31.8\%$, then reading the result. Three arguments at $98.5\%$ each cost more than
choosing among $128$ tools.

Here is where selection actually breaks:

```
   tools    distinct   8 families   4 families   2 families
-----------------------------------------------------------
      16      100.0%        96.5%        94.0%        87.9%
      64      100.0%        89.8%        83.9%        76.9%
     128      100.0%        85.3%        77.8%        67.1%
```

Real inventories grow by accretion — `search_docs`, `search_tickets`,
`search_code`, `search_docs_v2` — and forcing $64$ tools into two families takes
selection from $100.0\%$ to $76.9\%$. Compare the two effects: quadrupling from
$16$ to $64$ *distinct* tools costs nothing; taking $64$ tools from distinct to
two families costs $23.1$ points.

**Tool count is not the variable. Tool distinctness is**, and the two are confused
because they correlate in practice.

Count does start to matter when requests locate their tools imprecisely:

```
   noise     8 tools    32 tools   128 tools
--------------------------------------------
    0.55      100.0%      100.0%      100.0%
    1.30       94.7%       89.0%       75.2%
    2.50       65.3%       43.1%       23.1%
```

So vague requests and overlapping tools are the same failure seen from opposite
ends — both are distances in the same space — and a large inventory is safe only
while requests are specific.

Pricing interventions on a realistic inventory ($64$ tools, two families):

```
                            intervention   selection   end to end
-----------------------------------------------------------------
         baseline (64 tools, 2 families)       76.3%        69.3%
       halve the inventory, same overlap       85.5%        77.6%
       split into 8 families, same count       89.3%        81.1%
          make all 64 genuinely distinct      100.0%        90.8%
              drop one required argument       76.3%        70.3%
   constrain args to enums (98.5->99.8%)       76.3%        72.1%
```

Deduplicating is worth $+11.8$ points and making the inventory fully distinct
$+21.5$, against $+1.0$ for dropping an argument and $+2.8$ for enumerating them.
**With overlap present, inventory work dominates signature work** — and with
overlap removed, the second table says signature work is the entire remaining
budget.

The second listing fixes the inventory and varies the tool:

```
                         signature   first call   vs 3 free
-----------------------------------------------------------
             3 required, free text        89.9%       +0.0%
             1 required, free text        96.5%       +6.6%
            3 required, enumerated        98.5%       +8.6%
     2 required + 6 optional, free        83.9%       -6.0%
     2 required + 6 optional, enum        97.5%       +7.6%
```

Enumerating three arguments beats removing two of them ($+8.6$ against $+6.6$),
which is {{eq:enumerate-before-remove}} and is convenient because removing
arguments removes capability.

And six optional arguments cost $9.3$ points against a two-required baseline, even
though the model supplies each only half the time and is never obliged to.
**An optional argument is a required decision with two branches**
({{eq:optional-is-not-free}}).

Then the design decision that is almost never treated as one:

```
                     error message     try 0     try 1     try 2     try 3
--------------------------------------------------------------------------
                  opaque ('error')     83.9%     84.2%     84.5%     84.8%
       generic ('invalid request')     83.9%     87.1%     89.7%     91.8%
               names the bad field     83.9%     93.9%     97.7%     99.1%
   names it and lists valid values     83.9%     98.9%     99.9%    100.0%
```

Three retries against an opaque error buy $0.9$ points. The same three retries
against a message naming the field and listing valid values buy $16.1$. **The
error message is worth more than any signature change in the first table**, and it
costs nothing at inference time.

The reason is {{eq:error-message-as-selector}}: a retry after an opaque failure is
a fresh sample from the same distribution — coverage with no selector. A retry
after an informative failure is *conditioned* on the fault.

```
                                      design   calls made   success
-------------------------------------------------------------------
              6 optional args, opaque errors         1.47    84.84%
                6 optional args, good errors         1.17    99.99%
             no optional args, opaque errors         1.20    93.57%
               no optional args, good errors         1.07   100.00%
```

A bad signature with good errors beats a good signature with bad errors by $6.4$
points *while making fewer calls*, because most of its retries succeed
immediately once the fault is named.

Across five tool calls per task, the same designs give $43.94\%$ and $99.97\%$.
Per-call differences of a few points become task differences of tens
({{eq:chain-accuracy-compounds}}).

## 10. Production Considerations

Measure the four stages separately. Log which tool was chosen, whether a call was
made, whether the call was well formed, and whether the result was used correctly.
The aggregate number tells you nothing about where to work.

Embed your tool descriptions and compute nearest-neighbour distances. Tools whose
descriptions are close are tools the model will confuse, and the number is
available today from data you already have.

Write descriptions that say when *not* to use each tool. Findability is what teams
optimise; distinguishability is what fails.

Enumerate every argument that can be enumerated, and enforce it with a schema at
the sampler ({{part:8}}). This removes a failure class rather than reducing it.

Audit optional arguments. Each one is a decision with a cost
({{eq:optional-is-not-free}}); a tool with eight optional parameters is usually
several tools that were merged for the implementer's convenience.

Make every error message name the field and list acceptable values. It is the
highest-return change in this chapter and it is usually written by whoever was in
a hurry.

Return the smallest useful response in a consistent shape. It raises
$p_{\text{read}}$ and lowers context cost, and both compound over a run.

## 11. Common Mistakes

**Shortening the tool list to fix selection.** {{sec:9-practical-example}} finds
count nearly free and overlap expensive. Halving an overlapping inventory bought
$+8.3$ points; de-overlapping it bought $+21.5$.

**Treating optional arguments as free.** Six of them cost $9.3$ points.

**Retrying against an opaque error.** Three retries bought $0.9$ points; the same
retries against an informative error bought $16.1$.

**Reporting one tool-use accuracy number.** It is a product of four factors with
four different remedies ({{eq:four-decisions}}).

**Adding a `mode` parameter instead of a second tool.** It concentrates blast
radius, it makes the description less distinguishable, and it converts a selection
decision the model is good at into an argument decision it is worse at.

**Optimising the API's metrics.** A tool that succeeds $84\%$ per call is a
reasonable API and a bad agent tool, at $44\%$ over five calls.

## 12. Failure Modes

*Silent tool confusion.* The model calls a plausible neighbour of the right tool,
gets a plausible result, and proceeds. Nothing errors. This is the dominant
failure of an accreted inventory and it is invisible without per-call logging of
the chosen tool.

*Retry storms.* An opaque error plus an aggressive retry policy produces repeated
identical failing calls — {{eq:error-message-as-selector}} with $\phi \approx 0$ —
which is also one of {{ch:ag-loop}}'s non-productive cycles.

*Context poisoning by verbose responses.* Large tool outputs crowd the context,
degrade later decisions, and raise cost. The tool looks fine in isolation.

*Schema drift.* A tool's signature changes and the agent's $p_{\text{args}}$ drops
without any model change. Tool-call accuracy is a property of a model *against an
interface*.

*Over-calling.* A system tuned to call tools whenever plausible injects irrelevant
observations, which is cheap per instance and expensive over a horizon.

## 13. Alternatives

**One program instead of many calls.** {{cite:gao2023pal}} and
{{ch:rsn-tool-assisted}}: where the steps do not depend on intermediate
observations, emit one program and pay the boundary cost once.

**Constrained decoding.** {{part:8}}: make invalid calls unrepresentable. Strictly
better than validating after the fact where it applies.

**Tool retrieval.** For very large inventories, retrieve a candidate subset before
selecting. {{sec:9-practical-example}} says this is unnecessary for distinct tools
and it does help when the inventory is genuinely huge or when context cost binds —
and it introduces a retrieval failure ahead of the selection failure.

**Trained tool use.** {{cite:schick2023toolformer}}: train the four decisions
rather than prompting them. Raises $p$ at a fixed cost, and must be redone when
the interface changes.

**Fewer, larger tools.** Merging tools reduces selection difficulty and increases
argument difficulty and blast radius. {{sec:9-practical-example}} says the trade is
usually bad, since selection among distinct tools is nearly free and arguments are
not.

## 14. Evaluation

Report the four stage accuracies, not their product. And report them per tool, not
just in aggregate — one badly described tool can dominate the selection error.

Measure nearest-neighbour distance across your tool descriptions and track it as
the inventory grows. It is the leading indicator of the failure
{{sec:9-practical-example}} measures.

Measure $\phi$ — the fraction of failed calls whose retry succeeds — per tool. It
is the error-message quality metric, it is computable from logs you already have,
and it is the number {{eq:error-message-as-selector}} turns on.

Evaluate at the horizon you deploy at. A per-call benchmark understates the effect
of every intervention in this chapter by the exponent $k$.

And evaluate against a changed schema. Tool-call accuracy is a property of a model
against an interface; a system that has never been measured after an interface
change has not been measured.

## 15. Advanced Concepts

**Description optimisation as a retrieval problem.** Finding descriptions that
maximise separation while preserving findability is a well-posed optimisation over
embeddings, and it is almost never done. The objective is the minimum pairwise
distance in {{eq:distinctness-not-count}}, subject to each description remaining
close to the requests it should serve. {{maturity:EMERGING}}.

**Error messages as a learned interface.** $\phi$ is a property of the message
*and* of the model reading it. Optimising messages for the specific model that
consumes them is possible, measurable, and unusual — the message that helps a
human debug is not necessarily the one that helps a model retry.

**The over-calling/under-calling trade.** $p_{\text{when}}$ has two error
directions with different costs, and the operating point should be chosen from the
cost ratio rather than from a default. Almost no system exposes it as a knob.

**Tool composition.** If tools could be composed — one call producing a value that
another consumes, expressed declaratively rather than through the model's context
— the boundary-crossing count in {{ch:rsn-tool-assisted}} would fall sharply. This
is what {{cite:gao2023pal}}'s program-generation does implicitly, and doing it
explicitly at the tool layer is {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-what-is-an-agent}}'s per-step accuracy is {{eq:four-decisions}}, and this
chapter says which of its four factors to work on. The $p^k$ compounding is why
the answers matter more in an agent than in a single call.

{{ch:rsn-tool-assisted}} established when to use a tool at all, and its
boundary-crossing result is why {{sec:13-alternatives}} recommends one program
over many calls where the steps do not interleave.

{{ch:rsn-test-time-compute}}'s coverage/selection decomposition reappears in
{{eq:error-message-as-selector}}: a retry with no information is coverage, and an
error message is the selector.

{{part:8}}'s constrained decoding is the strongest intervention available, and
{{part:12}}'s retrieval is the fallback for inventories large enough that context
cost binds.

Ahead: {{ch:ag-loop}} takes the per-call accuracy this chapter decomposed and
puts it in a loop; {{ch:ag-security}} takes up the observation that the tool
inventory is also the blast radius.

## 17. Exercises

1. Vary $D$ in the first listing from 24 down to 4 and re-run the count sweep. At
   what dimensionality does tool count start to matter, and what does that say
   about {{eq:distinctness-not-count}}?

2. Add a "retrieve top-8 tools, then select" stage and measure it against direct
   selection at 128 tools in 2 families. Where does it help, and what does it cost?

3. Derive {{eq:enumerate-before-remove}} and find the $r$ at which enumerating
   overtakes removing, for $p_a = 0.95$ and $p_e = 0.995$.

4. In the second listing, make the error message quality depend on which argument
   was wrong. Which argument is most worth good messages for?

5. Model over-calling explicitly: add a cost for an unnecessary call and find the
   optimal $p_{\text{when}}$ operating point as a function of that cost.

6. Take your own tool inventory, embed the descriptions, and compute the
   nearest-neighbour distance distribution. Identify the three most confusable
   pairs.

## 18. Interview Questions

1. Your agent calls the wrong tool 15% of the time. What do you check first?

2. Why is adding the 50th tool cheaper than you would expect, and what is
   expensive instead?

3. Should you remove an argument or constrain it? Why?

4. Why is an optional argument not free?

5. Your retry rate is high and your retry success rate is barely above your
   first-call rate. What does that tell you?

6. A tool succeeds on 90% of calls. What does that imply about a five-step task,
   and what does it imply about the tool's API quality?

## 19. Research Questions

1. Can tool descriptions be optimised jointly for findability and separation, and
   how much does that buy over hand-written descriptions?

2. Is $\phi$ — retry-success given an error message — predictable from the message
   alone, and can messages be generated to maximise it for a specific model?

3. What determines the effective dimensionality of a model's tool-matching space,
   and does it vary enough between models to change the count-versus-overlap
   conclusion?

4. Can the over-calling/under-calling operating point be set adaptively per
   request rather than globally?

5. Does constrained decoding's removal of structural failures shift the remaining
   failures in a way that makes semantic errors harder to detect?

## 20. Chapter Summary

Tool-call accuracy is four numbers multiplied together
({{eq:four-decisions}}), and they have four different remedies. Reporting the
product hides which one you are losing to.

**Tool count is nearly free.** Selection was $100.0\%$ at four tools and $100.0\%$
at $128$, because distinct points in a high-dimensional space stay far apart
({{eq:distinctness-not-count}}). What destroys selection is overlap: the same $64$
tools in two families scored $76.9\%$, and $128$ in two families $67.1\%$.
Quadrupling a distinct inventory cost nothing; de-distinguishing one cost $23$
points. **Distinctness, not count**, and count only begins to matter when requests
themselves are vague.

With overlap removed, argument construction is the whole remaining budget — three
arguments at $98.5\%$ each cost more than choosing among $128$ tools. Enumerating
arguments beats removing them ({{eq:enumerate-before-remove}}), which is
convenient because removing them removes capability. And optional arguments are
not free: six cost $9.3$ points, because "optional" means "decide whether to
supply", which is a decision with two branches
({{eq:optional-is-not-free}}).

**The error message is worth more than any of it.** Three retries against an
opaque error bought $0.9$ points; against a message naming the field and listing
valid values, $16.1$. A bad signature with good errors beat a good signature with
bad errors by $6.4$ points while making fewer calls. The reason is
{{eq:error-message-as-selector}}: an uninformative retry is coverage with no
selector, and an informative one is conditioned. **An error message is the
cheapest selector in this book.**

All of it compounds. Across five tool calls the same designs gave $43.94\%$ and
$99.97\%$ task success, which is why tool design is an agent concern rather than
an API one — and why nothing in an API's own metrics would tell you.

## 21. Further Reading

{{cite:schick2023toolformer}} for the four-decision decomposition this chapter is
built on, and for the demonstration that all four can be trained rather than
prompted.

{{cite:gao2023pal}} for the alternative that avoids most of these failures by
avoiding most of the calls, and {{ch:rsn-tool-assisted}} for when that applies.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for what tool-using
agents actually achieve on long-horizon tasks, which is the calibration for how
much the numbers in this chapter matter.

{{cite:greshake2023indirect}} because the tool inventory is also the attack
surface, and {{ch:ag-security}} follows it up.
