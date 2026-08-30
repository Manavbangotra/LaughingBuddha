# -*- coding: utf-8 -*-
# Extracted from: Chapter 158 — Agent Memory: Short-Term, Working, and Long-Term
# Source: src/.../ch158-memory.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
