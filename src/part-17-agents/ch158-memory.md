---
id: ag-memory
number: 158
part: XVII
tier: full
status: draft
requires: [checkpoints-cap-the-exponent, thought-buys-composition,
           control-location]
provides: [three-memories, memory-turns-over, curation-scales-with-size,
           staleness-is-invisible, dilution-penalty, scratchpad-removes-an-exponent]
citations: [shinn2023reflexion, yao2023react, liu2024agentbench,
            zhou2024webarena, greshake2023indirect, huang2024selfcorrect]
---

## 1. Learning Objectives

By the end of this chapter you will be able to name the three distinct mechanisms
that get called "memory" and say which failure each one addresses; explain why
extending the context window can make recall *worse*; compute what a scratchpad
buys as a function of how often a derived value is reused; state why the value of
curating a memory store scales with its size while the harm of not curating it
stays invisible in accuracy; and choose a retrieval ranking that degrades safely
when you have no freshness signal.

## 2. Why This Matters

"Add memory" is a single sentence covering at least three mechanisms that solve
different problems, and treating it as one decision guarantees that a team reaches
for whichever mechanism it already has. {{sec:9-practical-example}} measures a
clean diagonal: a dependency on a recent fact is handled by the context and by
nothing else; a value that must be recomposed goes from $2.7\%$ to $81.2\%$ with a
scratchpad and gains nothing from a store; a fact from a previous run is $0\%$
without a store and $81.1\%$ with one. **Three needs, three mechanisms, one match
each.**

The chapter's sharpest single result is a row in the second table. Extending the
context window from six steps to fourteen took recall of *recent* facts from
$21.0\%$ to $10.5\%$ — not neutral, worse. Recall degrades as the window fills, and
it degrades for everything in the window including what was being recalled
perfectly well before. **A longer context is not a superset of a shorter one**, and
that row explains a common production experience: moving to a longer-context model
and finding some behaviours got worse.

The second listing was written to show that memory turns over — that accumulating
entries eventually becomes net harmful — and it does not. Accuracy rose
monotonically from $56.0\%$ at zero entries to $67.8\%$ at three thousand with no
curation at all. What the measurement *does* support is different and more
actionable: at fifty entries the curation policies differed by $2.3$ points and at
three thousand by $24.2$. **The value of curation scales with size**, so a policy
correctly judged unnecessary when the store was small becomes the largest lever
once it is not.

And there is a failure hiding under that rising accuracy. By three thousand
entries, $34.1\%$ of queries were answered from an entry that was true when written
and is not now. The agent has no signal that this is happening — it is not saying
"I do not know", it is answering confidently from something that matched. That
share is invisible in the accuracy column, which is going up.

## 3. Prerequisites

You need {{ch:ag-react}}'s observation that a thought stays in the context and can
turn a deep step into a shallow one later — this chapter puts a number on it.

From {{ch:ag-planning}}, the checkpoint: a plan's verified state is exactly the
kind of thing that belongs in a scratchpad, and the two mechanisms are the same
idea at different granularities.

From {{part:7}}, attention over a growing context, which is the mechanism behind
the dilution penalty. From {{part:12}}, retrieval — the store in this chapter *is*
a retrieval system and everything there applies, with a staleness caveat this
chapter adds.

And {{ch:rsn-self-consistency}}'s correlation result, because a memory the agent
wrote is a source the agent will trust.

## 4. Intuitive Explanation

Three different things get called memory. Separating them is most of the work.

**The context** is the raw history of this run — every action taken and every
observation received, carried forward verbatim. It is not something you build; it
is what the loop already does. Its property is that everything in it is available
without any retrieval step, and its limit is that the model attends to it
imperfectly and increasingly imperfectly as it fills.

**The scratchpad** is what the agent writes down deliberately: a derived value, a
conclusion, a note about what failed. The distinction from the context is that the
context contains the *inputs* to a computation and the scratchpad contains its
*result*. That difference is small to describe and enormous in effect.

**The store** is what survives between runs and is retrieved when relevant. This is
the only one of the three that reaches outside the current run, and it is the only
one that is a database problem rather than a prompting problem.

Now the needs, which map onto them one-to-one.

If a step needs a fact from three steps ago, the context has it. Adding a
scratchpad or a store changes nothing, because the fact is already present.

If a step needs a value that must be *combined* from several earlier facts — the
account from step two, the amount from the first message, the currency implied by
step four — the context has all the ingredients and none of the answer. The model
redoes the combination every time it needs it, and the combination is the
unreliable part. Writing the result down once converts every later use into a
lookup.

If a step needs something established in a previous run, nothing in this run
contains it. No amount of context reaches it.

So the three mechanisms are not a hierarchy where more is better. They are three
answers to three questions, and using the wrong one costs you most of the benefit.

The counterintuitive part concerns the context, and it is worth stating carefully
because the industry has spent several years making windows longer.

You would expect a longer window to be strictly better: everything the short window
held, plus more. It is not, because recall from a context is not lookup from an
array. The model has to *find* the relevant part, and adding irrelevant material
makes finding harder — for the new material and for the old. {{sec:9-practical-example}}
measures recall of recent facts getting worse when the window is extended, which
means a longer window is a different retrieval problem rather than a bigger
container.

That reframes what to do about it. If a fact matters, do not rely on it being in
the window; move it somewhere addressable. That is what a scratchpad and a store
both are, and it is why they beat window extension in every row of the
measurement.

The last idea concerns what happens to a store over time. Facts go stale. An entry
that was true when written is retrieved because it *matches*, and matching is not
freshness. The agent gets a confident answer from an authoritative-looking source
that happens to be out of date.

The instinct is to prune aggressively. The measurement says otherwise: a bigger
store keeps being better, because the benefit of having the right entry outweighs
the harm of sometimes surfacing an old one. What changes with size is not the sign
of the effect but the *value of curation* — and the amount of quietly wrong output,
which does not show up in an accuracy number that is going up.

## 5. Formal Explanation

Partition the facts a step can depend on by where they live:

$$\text{dep} \in \{\underbrace{\text{recent}}_{\text{context}},\ \underbrace{\text{derived}}_{\text{scratchpad}},\ \underbrace{\text{distant}}_{\text{context or store}},\ \underbrace{\text{prior}}_{\text{store only}}\}$$ (eq:three-memories)

Each mechanism serves a subset, and the subsets barely overlap. That is the
formal content of "not substitutes" and it is what {{sec:9-practical-example}}'s
diagonal measures.

For the context, model recall of an item at distance $d$ in a window of size $c$:

$$\Pr[\text{recall}] = \mathbb{1}[d \le c] \cdot \big(p_\ell - \lambda \min(t, c)\big)$$ (eq:dilution-penalty)

where $t$ is how full the window is and $\lambda$ is the per-entry dilution rate.
The indicator improves with $c$ and the parenthesis *degrades* with $c$. So
$\partial \Pr / \partial c$ has two terms of opposite sign, and for items already
inside the window only the negative one applies:

$$\frac{\partial}{\partial c}\Pr[\text{recall} \mid d \le c] = -\lambda < 0$$ (eq:longer-window-hurts-recent)

**Extending the window strictly degrades recall of everything already in it.** The
extension pays off only for items it newly admits, and whether the net is positive
depends on $\lambda$ against the share of dependencies at the new distances.

For the scratchpad, let a derived value require combining $m$ facts, succeeding at
$p_c$ per combination, and be needed $r$ times in a run:

$$S_{\text{recompute}} = \big(p_c^{\,m-1}\big)^{r}, \qquad S_{\text{written}} = p_\ell^{\,r}$$ (eq:scratchpad-removes-an-exponent)

Both are geometric in $r$, and the ratio is $\left(p_\ell / p_c^{m-1}\right)^{r}$ —
so the scratchpad's advantage grows geometrically in reuse count. **Writing a
derived value down removes a composition from the exponent**, which is
{{ch:ag-react}}'s thought-as-memory observation with the arithmetic attached.

For the store, model retrieval as ranking. Entries score
$s_i = \mathbb{1}[\text{topic match}] + \varepsilon_i$, the top $k$ enter context,
and the agent uses the highest-scoring match. With $n$ entries of which a fraction
$\sigma$ are stale, the three outcomes are:

$$\Pr[\text{no match}] \downarrow n, \qquad \Pr[\text{fresh on top}] \uparrow n, \qquad \Pr[\text{stale on top}] \uparrow n$$ (eq:memory-turns-over)

All three move with $n$, and the first falling is what makes the net positive. The
third rising is what makes the harm invisible: it is not a decline in accuracy, it
is a growing share of *correct-looking* answers sourced from stale entries.

Curation enters as a term in the ranking rather than as a deletion policy. Adding
$-\gamma \cdot \text{age}$ or $-\gamma \cdot \mathbb{1}[\text{stale}]$ to the score
shifts mass from the third outcome to the second, and its value scales with the
number of competing entries:

$$\Delta_{\text{curation}}(n) \propto \Pr[\text{stale on top} \mid n] \;\uparrow n$$ (eq:curation-scales-with-size)

which is why the measured policy gap goes from $2.3$ points at fifty entries to
$24.2$ at three thousand.

## 6. Mathematical Foundation

Three consequences.

**The window-extension decision has a computable threshold.** From
{{eq:dilution-penalty}}, extending from $c$ to $c'$ helps iff the probability mass
of dependencies at distances in $(c, c']$ exceeds the dilution loss
$\lambda(c' - c)$ applied to everything already inside. That is two numbers you can
measure — a distance histogram over your dependencies, and a needle-in-a-haystack
recall curve for your model — and almost nobody computes it before doubling a
window.

**The scratchpad's break-even is at one reuse.** From
{{eq:scratchpad-removes-an-exponent}}, writing wins whenever
$p_\ell > p_c^{m-1}$, which holds for any $m \ge 2$ at realistic reliabilities and
does not depend on $r$ at all. $r$ governs the *size* of the win, not its sign.
{{sec:9-practical-example}} measures $+21.1$ points at one reuse and $+77.9$ at
fourteen. **There is essentially no case where recomputing a derived value beats
recording it**, which makes this the cheapest unambiguous win in {{part:17}}.

**Staleness, not size, is the trigger for curation.** From
{{eq:curation-scales-with-size}}, the policy gap is proportional to the stale-on-top
probability, which is a product of the stale fraction and the number of competing
entries. Teams observe the second and act on it; the first is what actually moves.
{{sec:9-practical-example}}'s staleness sweep at fixed size shows no-curation
falling from $86.1\%$ to $46.8\%$ as the stale share goes $0 \to 75\%$ while a
freshness check holds $86.9\% \to 74.1\%$.

One thing the model deliberately omits, and it matters for
{{sec:12-failure-modes}}. Entries here go stale by the passage of time. In a real
agent, entries also go stale because the agent *wrote something wrong* — a failed
inference recorded as a fact. Those entries are indistinguishable from correct ones
by recency and by any freshness check keyed to time, and they inherit
{{ch:rsn-self-consistency}}'s correlation: the agent trusts them because it wrote
them.

## 7. Internal Mechanics

### 7.1 The three mechanisms, side by side

```mermaid {#fig:three-memories caption="What each mechanism holds and when it is read. The context is written by the loop; the scratchpad by the agent; the store by a previous run."}
flowchart LR
    A[step] --> B[context: raw history]
    A --> C[scratchpad: derived results]
    D[(store: prior runs)] --> A
    B -. attended, diluted .-> A
    C -. looked up .-> A
    A --> C
    A --> D
```

The arrows into the step are the read paths and they have different reliabilities:
attention over a diluting window, a lookup of something explicitly recorded, and a
ranked retrieval over a large candidate set.

### 7.2 Why a scratchpad entry is not a context entry

Both are text in the prompt, so the distinction can look cosmetic. It is not, and
the difference is *addressability*. A context entry is one item among many
competing for attention; a scratchpad entry is written in a known place with a
known name, so retrieving it is closer to a lookup than to a search.

That is why {{eq:scratchpad-removes-an-exponent}} uses $p_\ell$ rather than the
diluted recall of {{eq:dilution-penalty}}. Structure the scratchpad — a named
section, a stable key per fact — and it holds. Let it become free-form prose in the
history and it degrades into context.

### 7.3 What belongs in a scratchpad

Three kinds of thing, in decreasing order of return.

**Derived values**, per {{eq:scratchpad-removes-an-exponent}}. Anything the agent
computed from several inputs and will need again.

**Verified state**, which is {{ch:ag-planning}}'s checkpoint. "Segment two
completed; the record exists with id 4471" is both a restore point and a fact later
steps can read without re-deriving it. The two mechanisms are the same idea at
different granularities and should share a representation.

**Failed approaches**, which is {{cite:shinn2023reflexion}}'s episodic memory and
{{ch:ag-loop}}'s deduplication in a more general form: a record of what was tried
changes the context after a failure, which is
{{eq:context-change-breaks-loops}}'s requirement.

### 7.4 The store is a retrieval system with one extra problem

Everything in {{part:12}} applies: chunking, embedding, hybrid search, reranking.
The extra problem is that entries were written by the agent at a point in time, and
{{eq:memory-turns-over}} says the share surfaced stale grows with the store.

Two design responses. Attach a *validity* signal at write time — a timestamp, a
source, an expiry where one is knowable — and use it in ranking rather than as a
deletion criterion, since {{sec:9-practical-example}} says deletion is the wrong
instinct. And where no validity signal exists, rank by recency, which recovered a
useful fraction of the benefit for free.

### 7.5 Memory as an attack surface

{{cite:greshake2023indirect}}'s indirect injection has a specific consequence here:
anything an agent writes to a store from content it read is content an attacker can
place there. A store is persistent, shared across runs, and trusted because it is
"our own memory".

That makes memory writes a privileged operation. Content derived from untrusted
input should be marked as such and should not be retrievable as a fact, which is
{{ch:ag-security}}'s subject and one of the few places where the design response is
a schema change rather than a policy.

## 8. Implementation

Two listings. The first measures what happens to a store as it grows, and what
curation is worth at each size. The second gives an agent three distinct needs and
each mechanism in turn.

```python {tier=A name=memory-turns-over}
"""More memory is not better memory.

cite:shinn2023reflexion keeps reflective text in episodic memory so later attempts
can read what went wrong. It works, and it is the technique that made agent memory
a standard component. The question this listing asks is what happens when that
memory keeps growing, which is what happens to every deployed system
(eq:memory-turns-over).

The mechanism to watch is not capacity. It is RANKING. An agent retrieves the
entries that look most relevant, and as memory grows the number of entries
competing for those slots grows with it. Some of those entries are stale -- facts
that were true when written and are not now. A stale entry that outranks a fresh
one does not merely fail to help; it supplies a confident wrong answer.

So the interesting variable is the probability that the top-ranked matching entry
is the stale one, and that rises with memory size.
"""
import numpy as np

rng = np.random.default_rng(2141)

N = 40000
TOPICS = 200
K_RETRIEVE = 5            # entries pulled into context per query
RANK_NOISE = 0.45         # how imprecisely relevance is judged
P_USE_FRESH = 0.95        # answers correctly from a fresh matching entry
P_MISLED = 0.80           # follows a stale matching entry into a wrong answer
P_BASE = 0.55             # answers from parametric knowledge, no memory hit
STALE_RATE = 0.35         # share of accumulated entries that have gone stale


def run(n_entries, policy="none", stale_rate=STALE_RATE, k=K_RETRIEVE):
    """Memory holds n_entries. Each has a topic and may be stale. A query on
    topic t retrieves the k highest-scoring entries; score is a topic match plus
    ranking noise. The agent uses the highest-scoring matching entry it sees."""
    if n_entries == 0:
        return float(np.mean(rng.random(N) < P_BASE)), 0.0

    ent_topic = rng.integers(0, TOPICS, size=n_entries)
    ent_age = rng.random(n_entries)                     # 0 newest, 1 oldest
    ent_stale = rng.random(n_entries) < stale_rate * ent_age * 2
    ent_stale &= rng.random(n_entries) < 1.0

    q_topic = rng.integers(0, TOPICS, size=N)
    correct = np.zeros(N, dtype=bool)
    for i in range(N):
        base = (ent_topic == q_topic[i]).astype(float)
        score = base + RANK_NOISE * rng.normal(size=n_entries)
        if policy == "recency":
            score -= 0.5 * ent_age                      # prefer newer entries
        elif policy == "verified":
            score -= 1.5 * ent_stale                    # a freshness check
        top = np.argpartition(-score, min(k, n_entries) - 1)[:k]
        match = top[base[top] > 0]
        if len(match) == 0:
            correct[i] = rng.random() < P_BASE
        else:
            best = match[np.argmax(score[match])]
            if ent_stale[best]:
                correct[i] = rng.random() > P_MISLED
            else:
                correct[i] = rng.random() < P_USE_FRESH
    return float(correct.mean()), float(n_entries)


SIZES = [0, 50, 200, 800, 3000]
N = 4000        # per-query loop is slow; smaller sample, still tight enough

print(f"{TOPICS} topics, {K_RETRIEVE} entries retrieved per query, ranking noise")
print(f"{RANK_NOISE}. An entry that has gone stale misleads the agent")
print(f"{P_MISLED:.0%} of the time; with no matching entry the agent answers")
print(f"from parametric knowledge at {P_BASE:.0%}.")
print()
print(f"{'entries':>9}{'no eviction':>14}{'prefer recent':>16}"
      f"{'freshness check':>18}")
print("-" * 57)
tab = {}
for m in SIZES:
    a = run(m)[0]
    b = run(m, "recency")[0]
    c = run(m, "verified")[0]
    tab[m] = (a, b, c)
    print(f"{m:>9}{a:>14.1%}{b:>16.1%}{c:>18.1%}")

print()
print()
print("Where does the loss come from? Split the outcome by what the retrieval")
print("put in front of the agent, with no eviction.")
print()
print(f"{'entries':>9}{'no match':>11}{'fresh match':>14}{'stale on top':>15}")
print("-" * 49)
split = {}
for m in (50, 200, 800, 3000):
    ent_topic = rng.integers(0, TOPICS, size=m)
    ent_age = rng.random(m)
    ent_stale = rng.random(m) < STALE_RATE * ent_age * 2
    q_topic = rng.integers(0, TOPICS, size=N)
    none_c = fresh_c = stale_c = 0
    for i in range(N):
        base = (ent_topic == q_topic[i]).astype(float)
        score = base + RANK_NOISE * rng.normal(size=m)
        top = np.argpartition(-score, min(K_RETRIEVE, m) - 1)[:K_RETRIEVE]
        match = top[base[top] > 0]
        if len(match) == 0:
            none_c += 1
        elif ent_stale[match[np.argmax(score[match])]]:
            stale_c += 1
        else:
            fresh_c += 1
    split[m] = (none_c / N, fresh_c / N, stale_c / N)
    print(f"{m:>9}{split[m][0]:>11.1%}{split[m][1]:>14.1%}"
          f"{split[m][2]:>15.1%}")

print()
print()
print("How much staleness can memory tolerate? Sweep it at a fixed 800 entries.")
print()
print(f"{'stale share':>13}{'no eviction':>14}{'prefer recent':>16}"
      f"{'freshness check':>18}")
print("-" * 61)
st = {}
for s in (0.0, 0.10, 0.25, 0.50, 0.75):
    a = run(800, stale_rate=s)[0]
    b = run(800, "recency", stale_rate=s)[0]
    c = run(800, "verified", stale_rate=s)[0]
    st[s] = (a, b, c)
    print(f"{s:>13.0%}{a:>14.1%}{b:>16.1%}{c:>18.1%}")

print()
print()
print("And what retrieving more entries does, which is the other knob teams")
print("reach for. 800 entries, no eviction.")
print()
print(f"{'retrieved k':>13}{'accuracy':>11}{'context entries':>18}")
print("-" * 42)
kk = {}
for k in (1, 3, 5, 10, 20):
    kk[k] = run(800, k=k)[0]
    print(f"{k:>13}{kk[k]:>11.1%}{k:>18}")

print(f"""
The first table refutes the thing this listing was written to show, so start
there.

The outline predicted a non-monotone curve: memory helping up to a point and then
degrading as stale entries crowded the retrieval. It does not. With no eviction at
all, accuracy rises from {tab[0][0]:.1%} at zero entries to {tab[3000][0]:.1%} at
{3000} -- monotonically, at every size swept. **Accumulating memory without
curating it does not become net harmful**, and the common advice to prune
aggressively is not supported by this measurement.

What IS supported is in the columns rather than the rows. At {50} entries the
three policies differ by {max(tab[50]) - min(tab[50]):.1%}; at {3000} they differ
by {max(tab[3000]) - min(tab[3000]):.1%}. **The value of curation scales with
memory size**, from negligible to decisive, which means a policy that was
correctly judged unnecessary when the store was small becomes the largest
available lever once it is not.

The second table shows why both things are true at once, and it is the useful
decomposition.

As memory grows, the share of queries with NO matching entry collapses from
{split[50][0]:.1%} to {split[3000][0]:.1%} -- that is the benefit, and it is
large. The share answered from a FRESH match rises from {split[50][1]:.1%} to
{split[3000][1]:.1%}.

And the share where a STALE entry outranked everything rises from
{split[50][2]:.1%} to {split[3000][2]:.1%}.

Both grow. The good grows faster, so the net is positive -- but by {3000} entries,
**a third of all queries are being answered from an entry that was true when it
was written and is not now**, and the agent has no signal that this is happening.
It is not answering "I do not know"; it is answering confidently from a source
that looks authoritative because it matched.

That is the failure this chapter is about, and note that it is invisible in the
accuracy column. A team watching accuracy rise as its memory grows has no
indication that a third of its hits are stale, and the first symptom will be a
class of confidently wrong answers that correlates with how long the system has
been running.

The third table says how much staleness the design tolerates. At {800} entries
with nothing stale, all three policies land near {st[0.0][0]:.1%} -- curation buys
nothing when there is nothing to curate. At {0.75:.0%} stale, no eviction gives
{st[0.75][0]:.1%} and a freshness check gives {st[0.75][2]:.1%}.

**Curation is not a memory-size intervention. It is a staleness intervention**,
and the two get confused because staleness accumulates with size. The right
trigger for investing in it is a measurement of how fast your facts go stale, not
of how many you have.

Note also that preferring recent entries -- the cheap heuristic, requiring no
freshness signal -- recovers a meaningful part of the gap: {st[0.5][1]:.1%} against
no-eviction's {st[0.5][0]:.1%} and a real check's {st[0.5][2]:.1%} at
{0.5:.0%} stale. Recency is a proxy for freshness and it is available when nothing
else is.

The last table is the other knob teams reach for, and it is the weakest of the
three. Retrieving {20} entries instead of {5} buys
{kk[20] - kk[5]:+.1%} while quadrupling the context those entries occupy. Compare
that with the freshness check's {tab[800][2] - tab[800][0]:+.1%} at the same store
size and no additional context at all.

**Retrieving more is a way to spend context to partially compensate for ranking
you do not trust**, and it is dominated by fixing the ranking. It also runs
directly into part:15's cost model, since every retrieved entry is context that
every subsequent step of the run must carry.

So the practical ordering, which is not the one the chapter set out to argue:

Do not prune to keep memory small. The measurement says size helps.

Do track staleness, because it is the variable that decides everything and it is
not visible in accuracy.

Attach a freshness signal to entries and use it in ranking. It was worth
{tab[3000][2] - tab[3000][0]:+.1%} at {3000} entries here, more than any other
intervention measured.

And if you have no freshness signal, rank by recency, which recovers a useful
fraction of it for free.""")
```

The second listing separates the mechanisms.

```python {tier=A name=three-memories}
"""Three things are called memory, and they are not substitutes.

The word covers at least three mechanisms that solve different problems:

  the CONTEXT -- the raw history of this run, carried forward verbatim
  the SCRATCHPAD -- derived facts the agent wrote down deliberately
  the STORE -- what survives between runs and is retrieved

Teams reason about "adding memory" as one decision, which produces the two
predictable mistakes: extending the context to fix something a scratchpad would
fix, and adding a retrieval store to fix something the context would fix.

This listing gives an agent three distinct needs and each mechanism in turn, and
measures which mechanism addresses which need (eq:three-memories).
"""
import numpy as np

rng = np.random.default_rng(2213)

N = 60000
K = 14                  # steps in a run
CTX = 6                 # steps of raw history the model attends to reliably
DILUTE = 0.020          # per-entry loss of recall as context fills
P_COMPOSE = 0.88        # combining facts inside one pass
P_LOOKUP = 0.985        # reading one recorded fact


def run(need, context=True, scratchpad=False, store=False, ctx=CTX, k=K):
    """`need` selects which kind of dependency each step has:
      'recent'  -- a fact produced a few steps ago
      'distant' -- a fact produced early in this run
      'derived' -- a value that must be recomposed from three earlier facts
      'prior'   -- a fact established in a PREVIOUS run
    """
    ok = np.ones(N, dtype=bool)
    for i in range(k):
        if need == "recent":
            dist = rng.integers(1, 4, size=N)
        elif need == "distant":
            dist = rng.integers(1, k + 1, size=N)
        else:
            dist = np.full(N, 1)

        if need == "prior":
            # Nothing in this run contains it. Only a store can supply it.
            got = np.full(N, store)
            p = np.where(got, P_LOOKUP, 0.25)
        elif need == "derived":
            if scratchpad:
                # Written down once; every later use is a single lookup.
                p = np.full(N, P_LOOKUP)
            else:
                # Recomposed from three facts inside one forward pass.
                p = np.full(N, P_COMPOSE ** 2)
                if not context:
                    p = p * 0.5
        else:
            in_ctx = context & (dist <= ctx)
            # Recall degrades as the window fills, even inside it.
            recall = np.clip(P_LOOKUP - DILUTE * np.minimum(i, ctx), 0.2, 1.0)
            p = np.where(in_ctx, recall,
                         np.where(store, P_LOOKUP * 0.9, 0.30))
        ok &= rng.random(N) < p
    return float(ok.mean())


NEEDS = ["recent", "distant", "derived", "prior"]
CONFIGS = [
    ("context only", dict(context=True)),
    ("context + scratchpad", dict(context=True, scratchpad=True)),
    ("context + store", dict(context=True, store=True)),
    ("all three", dict(context=True, scratchpad=True, store=True)),
]

print(f"A {K}-step run. The model attends reliably to the last {CTX} steps of raw")
print(f"history, with recall degrading {DILUTE:.1%} per entry as the window")
print("fills. Each row is a kind of dependency the steps have.")
print()
print(f"{'dependency':>13}" + "".join(f"{n:>22}" for n, _ in CONFIGS))
print("-" * 101)
tab = {}
for need in NEEDS:
    row = {}
    for name, cfg in CONFIGS:
        row[name] = run(need, **cfg)
        tab[(need, name)] = row[name]
    print(f"{need:>13}" + "".join(f"{row[n]:>22.1%}" for n, _ in CONFIGS))

print()
print()
print("The wrong fix, priced. For each dependency, what does EXTENDING THE")
print("CONTEXT buy, against the mechanism that actually addresses it?")
print()
print(f"{'dependency':>13}{'ctx 6':>9}{'ctx 14':>9}{'ctx 30':>9}"
      f"{'right fix':>24}{'that fix':>11}")
print("-" * 76)
fixes = {"recent": ("nothing needed", dict(context=True)),
         "distant": ("add a store", dict(context=True, store=True)),
         "derived": ("add a scratchpad", dict(context=True, scratchpad=True)),
         "prior": ("add a store", dict(context=True, store=True))}
ext = {}
for need in NEEDS:
    a = run(need, context=True, ctx=6)
    b = run(need, context=True, ctx=14)
    c = run(need, context=True, ctx=30)
    label, cfg = fixes[need]
    d = run(need, ctx=6, **cfg)
    ext[need] = (a, b, c, d)
    print(f"{need:>13}{a:>9.1%}{b:>9.1%}{c:>9.1%}{label:>24}{d:>11.1%}")

print()
print()
print("What a scratchpad costs and saves on a 'derived' workload, as the number")
print("of times a derived value is reused grows.")
print()
print(f"{'reuses':>8}{'recompute each time':>22}{'write once, read':>19}"
      f"{'gain':>9}")
print("-" * 58)
re_tab = {}
for r in (1, 2, 4, 8, 14):
    a = run("derived", context=True, k=r)
    b = run("derived", context=True, scratchpad=True, k=r)
    re_tab[r] = (a, b)
    print(f"{r:>8}{a:>22.1%}{b:>19.1%}{b - a:>+9.1%}")

print()
print()
print("And how context dilution changes the picture -- the term that makes a")
print("longer window stop helping.")
print()
print(f"{'dilution':>10}{'ctx 6':>10}{'ctx 14':>10}{'ctx 30':>10}{'best':>9}")
print("-" * 49)
dl = {}
for d in (0.0, 0.01, 0.02, 0.04, 0.08):
    DIL_SAVE = DILUTE
    globals()["DILUTE"] = d
    row = [run("distant", context=True, ctx=c) for c in (6, 14, 30)]
    globals()["DILUTE"] = DIL_SAVE
    dl[d] = row
    print(f"{d:>10.0%}" + "".join(f"{v:>10.1%}" for v in row)
          + f"{[6, 14, 30][int(np.argmax(row))]:>9}")

print(f"""
The first table is the argument for refusing the umbrella term, and it is a
diagonal.

A dependency on a RECENT fact is handled by the context and by nothing else:
{tab[('recent', 'context only')]:.1%} with context alone,
{tab[('recent', 'all three')]:.1%} with all three mechanisms. Adding a scratchpad
or a store buys nothing, because the fact is already there.

A DERIVED value -- one that must be recomposed from several earlier facts -- goes
from {tab[('derived', 'context only')]:.1%} to
{tab[('derived', 'context + scratchpad')]:.1%} with a scratchpad, and a store does
nothing for it ({tab[('derived', 'context + store')]:.1%}). The context contains
everything needed; what it does not contain is the RESULT, so every use pays the
composition again.

A PRIOR fact -- established in an earlier run -- is {tab[('prior', 'context only')]:.1%}
without a store and {tab[('prior', 'context + store')]:.1%} with one. No amount of
context or scratchpad reaches it, because it is not in this run.

**Three mechanisms, three needs, one match each.** Calling them all "memory"
guarantees that a team will reach for whichever one it already has, and the table
says that is right one time in three.

The second table prices the wrong fix, and its first row is the result worth
carrying away.

Extending the context from {6} to {14} steps takes the RECENT case from
{ext['recent'][0]:.1%} to {ext['recent'][1]:.1%}. **A longer window made recall of
recent facts worse.** Not neutral -- worse, by
{ext['recent'][0] - ext['recent'][1]:.1%}.

The mechanism is dilution: recall degrades as the window fills, and it degrades
for everything in the window, including the things that were being recalled
perfectly well before. A longer context is not a superset of a shorter one; it is
a different retrieval problem over a larger candidate set, and part:7's attention
arithmetic is why.

That single row explains a very common production experience -- moving to a
longer-context model and finding some behaviours got worse -- and it says the fix
is not more context.

Down the rest of the column: extending the context does nothing at all for
DERIVED ({ext['derived'][0]:.1%} to {ext['derived'][2]:.1%} across a fivefold
window increase) and nothing for PRIOR ({ext['prior'][2]:.1%} at {30} steps of
history). The right fixes, at the ORIGINAL context length, deliver
{ext['derived'][3]:.1%} and {ext['prior'][3]:.1%}.

**The right mechanism at a small context beats the wrong mechanism at a large
one**, in every row, by a wide margin.

The third table says when a scratchpad is worth writing, and the answer is "almost
always, and increasingly".

At one reuse, writing the derived value down instead of recomposing it buys
{re_tab[1][1] - re_tab[1][0]:+.1%}. At fourteen reuses it buys
{re_tab[14][1] - re_tab[14][0]:+.1%}, taking the run from
{re_tab[14][0]:.1%} to {re_tab[14][1]:.1%}.

The reason is the same exponent that has governed this whole part. Recomposing
pays {P_COMPOSE:.0%}-squared every time the value is needed, so the cost is
geometric in reuses. Writing it once converts every subsequent use into a lookup
at {P_LOOKUP:.1%}. **A scratchpad is not a note-taking convenience. It is a way of
removing a repeated composition from the exponent**, which is ch:ag-react's
thought-as-memory observation with a number attached.

The last table is the one that decides whether a longer window helps at all, and
it isolates the term that everything else in this chapter has been working around.

With NO dilution, a longer window is strictly better: {dl[0.0][0]:.1%} at {6}
steps against {dl[0.0][2]:.1%} at {30}. The naive intuition is correct in that
world. At {0.02:.0%} dilution per entry the same comparison is
{dl[0.02][0]:.1%} against {dl[0.02][2]:.1%} -- still better, and both are now
poor. At {0.08:.0%} nothing works at any window size.

So the value of context length is entirely a function of how well recall holds up
across the window, and that is a property of the model rather than of the
architecture. **Context length is a capability you should measure on your model
rather than a resource you should assume**, and the measurement is cheap: place a
fact at varying distances and ask for it back.

The practical summary is four rules, one per row of the first table.

For facts produced a few steps ago: nothing to do, and do not lengthen the window.

For facts produced early in a long run: promote them out of the raw history into a
store, because the window will not hold them and dilution punishes trying.

For values that get recomputed: write them down once. The gain scales with reuse
count and it is the largest single number in this listing.

For anything that must survive the run: it needs a store, and there is no
substitute -- which is where part:12's retrieval material applies directly, with
the previous listing's staleness caveat attached.""")
```

## 9. Practical Example

The first listing gives an agent a memory store over $200$ topics, retrieving five
entries per query, where an entry that has gone stale misleads it $80\%$ of the
time.

```
  entries   no eviction   prefer recent   freshness check
---------------------------------------------------------
        0         56.0%           55.2%             54.5%
       50         58.1%           59.6%             60.4%
      200         58.7%           60.2%             68.8%
      800         66.5%           69.0%             81.8%
     3000         67.8%           75.6%             92.0%
```

This refutes what the listing was written to show. There is no non-monotone curve:
with no curation at all, accuracy rises at every size swept. **Accumulating memory
without curating it does not become net harmful**, and the standard advice to prune
aggressively is not supported here.

What *is* supported is in the columns. At fifty entries the three policies differ
by $2.3$ points; at three thousand by $24.2$. **The value of curation scales with
size** ({{eq:curation-scales-with-size}}).

The decomposition shows why both are true:

```
  entries   no match   fresh match   stale on top
-------------------------------------------------
       50      81.0%         13.2%           5.8%
      200      55.3%         28.6%          16.1%
      800      20.5%         52.0%          27.5%
     3000       3.5%         62.4%          34.1%
```

"No match" collapses from $81.0\%$ to $3.5\%$ — that is the benefit. "Stale on top"
rises from $5.8\%$ to $34.1\%$. Both grow; the good grows faster.

But by three thousand entries **a third of all queries are answered from an entry
that was true when written and is not now**, and the agent has no signal. It is not
saying "I do not know"; it is answering confidently from something that matched.
That share is invisible in an accuracy column that is going up, and the first
symptom will be a class of confidently wrong answers correlating with how long the
system has been running.

Sweeping staleness at fixed size isolates the real trigger:

```
  stale share   no eviction   prefer recent   freshness check
-------------------------------------------------------------
           0%         86.1%           84.3%             86.9%
          25%         71.5%           75.3%             83.4%
          75%         46.8%           55.1%             74.1%
```

At zero stale, curation buys nothing. **Curation is a staleness intervention, not a
size intervention**, and the two get confused because staleness accumulates with
size. Note that ranking by recency — available when no freshness signal exists —
recovers a useful part of the gap.

Retrieving more entries, the other knob teams reach for, is the weakest:

```
  retrieved k   accuracy   context entries
------------------------------------------
            1      60.9%                 1
            5      65.3%                 5
           20      68.1%                20
```

Quadrupling $k$ buys $+2.8$ points and quadruples the context those entries occupy,
against the freshness check's $+15.3$ at the same store size and no extra context.

The second listing separates the mechanisms, and the result is a diagonal:

```
   dependency    context only   ctx + scratchpad    ctx + store    all three
----------------------------------------------------------------------------
       recent           20.8%              21.2%          20.8%        21.1%
      distant            0.0%               0.0%          19.6%        19.3%
      derived            2.7%              81.2%           2.8%        80.8%
        prior            0.0%               0.0%          81.1%        81.0%
```

A recent fact is handled by the context and nothing else. A derived value goes
$2.7\% \to 81.2\%$ with a scratchpad and gains nothing from a store. A prior fact
is $0\%$ without a store and $81.1\%$ with one. **Three needs, three mechanisms,
one match each** ({{eq:three-memories}}) — so calling them all "memory" means a
team reaches for whichever it has, and that is right one time in three.

The wrong fix, priced:

```
   dependency    ctx 6   ctx 14   ctx 30               right fix   that fix
----------------------------------------------------------------------------
       recent    21.0%    10.5%    10.4%          nothing needed      20.9%
      distant     0.0%    10.4%    10.5%             add a store      19.4%
      derived     2.8%     2.7%     2.7%        add a scratchpad      80.9%
        prior     0.0%     0.0%     0.0%             add a store      80.7%
```

The first row is the result to carry away. **Extending the window from six to
fourteen made recall of recent facts worse**, $21.0\% \to 10.5\%$, because dilution
applies to everything in the window including what was working
({{eq:longer-window-hurts-recent}}). And extending does nothing at all for derived
or prior, where the right mechanism at the *original* window delivers $80.9\%$ and
$80.7\%$.

**The right mechanism at a small context beats the wrong mechanism at a large one**
in every row.

The scratchpad's value against reuse count:

```
  reuses   recompute each time   write once, read     gain
----------------------------------------------------------
       1                 77.4%              98.6%   +21.1%
       4                 36.3%              94.2%   +57.9%
      14                  2.8%              80.7%   +77.9%
```

$+21.1$ points at a single reuse, $+77.9$ at fourteen
({{eq:scratchpad-removes-an-exponent}}). The break-even is below one reuse, so this
is the cheapest unambiguous win in the part.

And what decides whether a longer window helps at all:

```
  dilution     ctx 6    ctx 14    ctx 30     best
-------------------------------------------------
        0%      0.1%     80.8%     81.2%       30
        2%      0.0%     10.5%     10.6%       30
        8%      0.0%      0.0%      0.0%        6
```

With no dilution the naive intuition holds and longer is better. **Context length
is a capability to measure on your model, not a resource to assume**, and the
measurement is cheap: place a fact at varying distances and ask for it back.

## 10. Production Considerations

Classify your dependencies before adding anything. Sample traces and label each
step's need as recent, distant, derived, or prior. That histogram picks the
mechanism, and {{sec:9-practical-example}} says picking by habit is right a third
of the time.

Write derived values to a structured scratchpad with stable keys. Break-even is
below one reuse and the gain is geometric after that.

Do not extend the context to fix a memory problem. Measure your model's recall
against distance first, and note that extending degrades what was already working.

Do not prune a store to keep it small. Size helps; staleness hurts. Prune for
staleness or, better, rank for it.

Attach a validity signal at write time and use it in ranking rather than for
deletion. Where none exists, rank by recency.

Instrument the *stale-on-top* share, not just accuracy. It was $34.1\%$ at three
thousand entries while accuracy was rising, and it is the metric that would have
warned you.

Treat memory writes derived from untrusted content as privileged
({{cite:greshake2023indirect}}). A store is persistent, shared, and trusted because
it is yours.

## 11. Common Mistakes

**Treating "memory" as one decision.** Three mechanisms, three needs, one match
each ({{eq:three-memories}}).

**Extending the context window to improve recall.** It degrades recall of
everything already inside ({{eq:longer-window-hurts-recent}}); measured at
$-10.5$ points here.

**Recomputing derived values instead of recording them.** Break-even is below one
reuse.

**Pruning a store to keep it small.** Accuracy rose monotonically with size; the
problem is staleness, not volume.

**Letting the scratchpad become prose in the history.** Without stable keys it
degrades into context and loses the addressability that made it work.

**Judging memory by accuracy alone.** The stale-on-top share grew to a third while
accuracy improved.

**Retrieving more entries to compensate for bad ranking.** $+2.8$ points for
$4\times$ the context, against $+15.3$ for fixing the ranking.

## 12. Failure Modes

*Confident staleness.* An out-of-date entry outranks a current one and the agent
answers from it. No error, no uncertainty, and the rate grows with store size.

*Self-poisoning.* The agent records a wrong inference as a fact, retrieves it
later, and trusts it because it wrote it — {{ch:rsn-self-consistency}}'s
correlation, persisted. Recency and time-based freshness checks do not touch this.

*Context dilution after a window upgrade.* Behaviours that worked degrade, and the
change that caused it looks like a strict improvement.

*Injected memory.* Content read from an untrusted source is written to the store
and retrieved later as trusted fact ({{cite:greshake2023indirect}}).

*Scratchpad drift.* Keys change between runs or between prompt versions, so lookups
silently miss and the agent recomputes — the win disappears with no error.

## 13. Alternatives

**No store at all.** For single-run tasks with no cross-run knowledge, the context
and a scratchpad cover everything, and a store adds a staleness problem for no
benefit.

**External state instead of memory.** A database the agent queries as a tool is a
store with a schema, validity semantics, and someone else's freshness guarantees.
Where the domain has one, use it.

**{{part:12}}'s full retrieval stack.** The store *is* retrieval; hybrid search,
reranking and chunking all apply directly.

**Summarisation instead of extension.** Compress the history rather than lengthening
the window — this trades dilution for information loss, and it is worth measuring
against {{eq:dilution-penalty}} rather than assuming.

**Structured state objects.** Replace free-text memory with typed fields the agent
reads and writes. Loses generality, removes the ranking problem entirely, and is the
right answer whenever the state is enumerable.

## 14. Evaluation

Measure recall against distance for your model — a fact placed $d$ steps back,
asked for later. That curve is {{eq:dilution-penalty}} and it decides every context
question.

Report the retrieval outcome split — no match, fresh on top, stale on top — not
just accuracy. The third column is the one that fails silently.

Measure the stale fraction of your store directly, by sampling entries and checking
them. It is the trigger for curation and it is not inferable from size.

Measure scratchpad hit rate: how often a lookup for a recorded key succeeds. A
falling hit rate is key drift and it is invisible in end-to-end accuracy.

And evaluate at the store size you will have in a year, not the one you have now.
The policy gap grew tenfold between fifty and three thousand entries.

## 15. Advanced Concepts

**Write-time validity estimation.** The strongest curation signal is an expiry
attached when the entry is written, and for many fact types it is derivable — a
price, a schedule, a headcount all have known volatility. Estimating it
automatically is tractable and rarely done. {{maturity:EMERGING}}.

**Contradiction detection at write time.** A new entry that contradicts an existing
one is the strongest possible staleness signal, and it converts the problem from
ranking to reconciliation. It needs a notion of contradiction over free text, which
is the hard part.

**Provenance as a first-class field.** Recording *why* an entry is believed —
observed, derived, or read from an untrusted source — addresses both the
self-poisoning and the injection failure modes with one schema change, and it makes
the trust decision explicit rather than implicit.

**Memory as process supervision.** {{ch:ag-planning}} noted that verified
checkpoints are free process labels; a scratchpad of derived-and-later-confirmed
values is the same thing. An agent that records and verifies is producing
{{ch:rsn-supervision}}'s step-level data as a by-product.
{{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-react}}'s thought-as-memory observation is
{{eq:scratchpad-removes-an-exponent}} with a number: a thought that records a
composition converts every later use into a lookup.

{{ch:ag-planning}}'s checkpoint and this chapter's scratchpad are the same
mechanism at different granularities, and they should share a representation — a
verified state is both a restore point and a recorded fact.

{{ch:ag-loop}}'s deduplication is a scratchpad of failed actions, and
{{eq:context-change-breaks-loops}}'s requirement — that the context after a failure
differ from the context before — is what recording failures satisfies.

{{part:12}}'s retrieval is the store, and {{part:7}}'s attention is why
{{eq:dilution-penalty}} exists.

Ahead: {{ch:ag-recovery}} uses the record of failed approaches;
{{ch:ag-termination}} uses the scratchpad as the structure a budget hangs from; and
{{ch:ag-security}} takes up memory writes as a privileged operation.

## 17. Exercises

1. Compute the window-extension threshold from
   {{eq:dilution-penalty}} for your own dependency-distance histogram and a
   measured $\lambda$. Should you extend?

2. In the second listing, make the scratchpad keys drift between steps with some
   probability and find the drift rate at which the scratchpad stops beating
   recomputation.

3. Add self-poisoning to the first listing: entries written from a failed inference
   are wrong from birth. Show that recency ranking does not help and say what
   would.

4. Sweep `RANK_NOISE` in the first listing. At what ranking quality does the
   stale-on-top share stop growing with store size?

5. Implement contradiction detection — a new entry on a topic marks older ones
   suspect — and measure it against the freshness check.

6. Take a real agent trace and classify every step's dependency into the four
   categories. What does the histogram say you should build?

## 18. Interview Questions

1. Name three things called "memory" and the failure each one fixes.

2. Why can a longer context window make an agent worse?

3. When is it worth writing a derived value to a scratchpad?

4. Your memory store keeps growing and accuracy keeps improving. What are you not
   measuring?

5. Should you prune an agent's memory? What decides it?

6. Why is a memory write from retrieved content a security decision?

## 19. Research Questions

1. Can entry validity be estimated at write time from the fact's type, and how
   much of the freshness check's benefit does that recover?

2. Does contradiction detection over free-text memory outperform recency ranking,
   and at what cost?

3. How much of a real store's staleness is time-based versus self-poisoning, and
   does any signal separate them?

4. Is there a summarisation policy that beats window extension under
   {{eq:dilution-penalty}}, and how does its information loss compare with
   dilution?

5. Do checkpoint and scratchpad records make usable process-supervision data
   ({{ch:rsn-supervision}}), given that the agent generated and verified them?

## 20. Chapter Summary

Three mechanisms are called memory and they are not substitutes
({{eq:three-memories}}). {{sec:9-practical-example}} measures a diagonal: recent
facts are the context's job ($20.8\%$ with context alone, $21.1\%$ with everything);
derived values are the scratchpad's ($2.7\% \to 81.2\%$, unchanged by a store); and
prior-run facts are the store's ($0\% \to 81.1\%$).

**Extending the context window made recall of recent facts worse** — $21.0\%$ to
$10.5\%$ going from six steps to fourteen — because dilution applies to everything
inside ({{eq:longer-window-hurts-recent}}). The right mechanism at a small context
beat the wrong mechanism at a large one in every row.

A scratchpad removes a composition from the exponent
({{eq:scratchpad-removes-an-exponent}}): $+21.1$ points at one reuse, $+77.9$ at
fourteen, with break-even below one reuse. It is the cheapest unambiguous win in
{{part:17}}.

The store listing refuted its own hypothesis. Accuracy rose monotonically with
store size, $56.0\% \to 67.8\%$, with no curation — so the instinct to prune is
wrong. What scales with size is the *value of curation*: a $2.3$-point policy gap
at fifty entries and $24.2$ at three thousand ({{eq:curation-scales-with-size}}).
And curation is a staleness intervention rather than a size one, since at zero
staleness all policies tie.

The failure to watch is invisible in accuracy. At three thousand entries, $34.1\%$
of queries were answered from an entry that was true when written and is not now
({{eq:memory-turns-over}}) — confidently, from a source that matched — while the
accuracy column went up.

## 21. Further Reading

{{cite:shinn2023reflexion}} for episodic memory of failures, which is the
scratchpad's third use and the one with a published result behind it.

{{part:12}} in full, because the store is a retrieval system and everything there
applies — with this chapter's staleness term added.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for long-horizon
environments where the dependency-distance histogram is genuinely spread out, which
is when these distinctions start to bind.

{{cite:greshake2023indirect}} for why memory writes are privileged, taken up in
{{ch:ag-security}}.
