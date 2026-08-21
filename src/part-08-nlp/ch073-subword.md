---
id: nlp-subword
number: 73
part: VIII
tier: full
status: draft
requires: [nlp-preprocessing, tf-embeddings, math-probability, math-random-vars,
           py-fundamentals]
provides: [subword-tokenization, byte-pair-encoding, merge-table, wordpiece,
           unigram-tokenizer, sentencepiece, byte-level-bpe, viterbi-segmentation,
           subword-regularization, special-tokens, vocabulary-size-choice,
           compression-ratio]
citations: [gage1994, sennrich2016, schuster2012, wu2016, kudo2018sentencepiece,
            kudo2018subword, radford2019, petrov2023]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement byte pair encoding — both the training loop that learns a merge
   table and the encoder that applies it — from scratch.
2. State WordPiece's merge criterion exactly and show that it is a pointwise
   mutual information score, not a frequency count.
3. Derive the unigram tokenizer's objective and its Viterbi decoding, and explain
   what having a probability model buys that BPE does not have.
4. Explain what SentencePiece changes, and why it is orthogonal to the choice of
   merge algorithm.
5. Explain why byte-level BPE makes the unknown token structurally impossible.
6. Measure compression ratio against merge count and choose a vocabulary size
   from the curve rather than from convention.
7. State honestly what these algorithms do and do not know about language.

## 2. Why This Matters

**This is the algorithm that runs before every forward pass in the rest of the
book.** {{ch:nlp-preprocessing}} established the problem; this chapter is the
solution the entire field converged on. GPT, Llama, T5, BERT, and every
production model in {{part:10}} tokenize with one of the three algorithms here,
or a variant of one.

**It is the most consequential heuristic in machine learning.**
{{cite:gage1994}} is a 1994 article in a trade magazine about making files
smaller. It contains no linguistics, no statistics beyond counting, and no
learning. {{cite:sennrich2016}} applied it to text unchanged, and it is now the
input layer of models trained on tens of thousands of GPUs.

**The differences between the three algorithms are small and the differences in
how they are described are enormous.** Most secondary sources state WordPiece's
merge criterion incorrectly — as frequency, when it is a likelihood ratio — and
most conflate SentencePiece with unigram, when SentencePiece implements both BPE
and unigram and its actual contribution is neither. Getting this right is worth
a chapter because these are the details that determine behaviour.

**Vocabulary size is a design parameter you will have to choose**, and it is
bought with embedding parameters and sold for sequence length
({{eq:tokenizer-tradeoff}}). The curve that governs it is measurable in a few
seconds, and almost nobody measures it.

## 3. Prerequisites

{{ch:nlp-preprocessing}} for the tradeoff this chapter resolves, for fertility,
and for the normalisation stage that runs before any of these algorithms.
{{ch:tf-embeddings}} for why $|V|$ is a parameter-count decision.
{{ch:math-probability}} for the likelihood objective in
{{sec:6-mathematical-foundation}}, and {{ch:math-random-vars}} for the expected
value that the unigram model's EM step maximises. {{ch:py-fundamentals}} for
the dictionaries and string operations the implementation uses.

## 4. Intuitive Explanation

You have a corpus and a budget: you may keep $|V|$ strings, and every piece of
text must be expressible as a sequence of them. Which strings do you keep?

Start with every character. Nothing is unrepresentable, and everything is
maximally long. Now spend your budget buying shortcuts: each new vocabulary
entry is a string of characters that, from now on, counts as one token.

**The only question is which shortcut to buy next**, and the three algorithms
give three different answers.

**BPE buys the most common adjacent pair.** Look through the corpus, find which
two neighbouring units occur together most often, and glue them into one. Repeat.
That is the entire algorithm. It is greedy, it is unambiguous, and it has no
model of anything — it is exactly the file compressor from 1994.

**WordPiece buys the pair that is most surprising.** Instead of asking "which
pair is most common", ask "which pair occurs together far more often than its
two halves would predict independently". `th` is very common but `t` and `h` are
both very common too, so their co-occurrence is unremarkable. A pair that is
frequent *and* whose parts are individually rare is evidence of a real unit.
That ratio is the criterion.

**Unigram buys nothing — it sells.** Start with a large candidate vocabulary and
repeatedly throw away the entries whose removal hurts a probability model
least. It is pruning rather than merging, and the model it prunes against gives
it something the other two lack: a probability for every possible segmentation,
so the "best" split is a computed answer rather than whatever the greedy rule
happened to do.

> NOTE: SentencePiece is not a fourth algorithm. It is the decision to feed the
> raw Unicode string in — with spaces treated as an ordinary character, usually
> written `▁` — instead of splitting on whitespace first. It implements BPE and
> unigram; what it changes is the input, and that change is what makes the whole
> pipeline lossless and language-independent.

**The mental model:** a tokenizer is a codebook fitted by compression, and the
merge rule is a policy for spending a fixed codebook budget. Where the model
breaks down: a compressor is judged only by output length, but a tokenizer is
also choosing which units the model can reason about atomically. Nothing in any
of these three objectives knows that.

## 5. Formal Explanation

### 5.1 Byte pair encoding

**Training.** Given a corpus represented as a multiset of words, each written as
a sequence of symbols, and a target number of merges $M$:

1. Initialise the symbol inventory to the set of characters appearing in the
   corpus.
2. Count every adjacent symbol pair across the corpus, weighted by word
   frequency.
3. Let $(a,b)$ be the most frequent pair. Add $ab$ to the vocabulary and record
   the merge.
4. Replace every occurrence of the adjacent pair $(a,b)$ with the single symbol
   $ab$.
5. Repeat from 2 until $M$ merges have been made.

The output is an **ordered list of merges**. Order is essential: the merges must
be applied at encoding time in exactly the order they were learned, because
later merges are defined over symbols that earlier merges created.

**Encoding.** Split the input into characters and apply the learned merges in
order, each one exhaustively, until none applies.

$$
\tau_{\text{BPE}}(w) = \text{apply}(m_M, \cdots \text{apply}(m_2,
 \text{apply}(m_1, \text{chars}(w))))
$$ (eq:bpe-encode)

This is deterministic, and it is not optimal in any sense — {{sec:13-alternatives}}
returns to the fact that a shorter valid segmentation using the same vocabulary
frequently exists.

### 5.2 WordPiece

The algorithm is BPE with a different step 3. Instead of the most frequent pair,
WordPiece selects the pair maximising the increase in the training-data
likelihood under a unigram model ({{cite:schuster2012}}, described practically
in {{cite:wu2016}}). For a corpus of $N$ symbol tokens, the unigram log
likelihood is $\sum_x c(x)\log p(x)$ with $p(x) = c(x)/N$, and merging $(a,b)$
changes it by an amount monotone in

$$
\text{score}(a,b) = \frac{c(ab)}{c(a)\,c(b)}
$$ (eq:wordpiece-score)

where $c(\cdot)$ is a corpus count.

**This is a pointwise mutual information criterion, not a frequency one.**
Multiplying numerator and denominator by $N$:

$$
\log \big(N\cdot\text{score}(a,b)\big)
 = \log \frac{p(ab)}{p(a)p(b)} = \text{PMI}(a,b)
$$ (eq:wordpiece-pmi)

so WordPiece merges the pair with the highest pointwise mutual information, and
BPE merges the pair with the highest joint count. They differ exactly by the
denominator, and the denominator is what suppresses pairs that are frequent only
because their parts are frequent.

WordPiece also marks continuation rather than word start: BERT's vocabulary
writes `playing` as `play`, `##ing`, where `##` means "this piece does not begin
a word".

### 5.3 The unigram language model tokenizer

{{cite:kudo2018subword}} inverts the construction. Assume a unigram model in
which a segmentation's probability is the product of its pieces' probabilities:

$$
P(\vec{x}) = \prod_{i=1}^{k} p(x_i), \qquad \sum_{x \in V} p(x) = 1
$$ (eq:unigram-model)

for a segmentation $\vec{x} = (x_1,\dots,x_k)$ of a string. The best segmentation
is

$$
\vec{x}^* = \argmax_{\vec{x}\,\in\,S(w)} \prod_i p(x_i)
$$ (eq:unigram-viterbi)

over $S(w)$, the set of all segmentations of $w$ using $V$. That set is
exponentially large, but the objective factorises over positions, so
{{sec:7-internal-mechanics}} computes it exactly with dynamic programming in
$O(|w|^2)$.

Training proceeds by EM and pruning:

1. Seed $V$ with a large set of candidate substrings.
2. **E step**: compute the expected count of each piece under the current $p$.
3. **M step**: set $p(x)$ proportional to its expected count.
4. **Prune**: for each piece, compute the loss in total corpus likelihood if it
   were removed, and drop the bottom fraction.
5. Repeat until $|V|$ reaches the target.

Characters are never pruned, which is what keeps the vocabulary complete.

**What the probability model buys:** a distribution over segmentations rather
than one segmentation. Sampling from it at training time is **subword
regularization** — a genuine data augmentation, because the model sees the same
text segmented differently across epochs and cannot over-fit to one arbitrary
split.

### 5.4 SentencePiece

{{cite:kudo2018sentencepiece}} is an implementation decision, orthogonal to all
three algorithms above. Every earlier subword implementation assumed a
whitespace pre-tokenizer had already run, which has two consequences:

- It is **wrong for languages without whitespace word boundaries** — Japanese,
  Chinese, Thai — where "split on spaces first" is not a simplification but an
  absence of behaviour.
- It makes **detokenization ambiguous** everywhere else, because the number of
  spaces between two words was discarded and must be guessed back.

SentencePiece treats the input as a raw Unicode stream and encodes whitespace as
an ordinary symbol, conventionally `▁` (U+2581). Then

$$
\text{detokenize}(\vec{x}) = \text{replace}(\text{concat}(\vec{x}),\ \text{▁} \to \text{space})
$$ (eq:sentencepiece-detok)

is exact, with no heuristics and no language-specific rules.

### 5.5 Byte-level BPE

{{cite:radford2019}} closes the vocabulary completely: run BPE over **bytes**
rather than Unicode characters. The base alphabet is exactly the 256 byte values,
so every possible input — any script, any emoji, any corrupt encoding — is
representable.

$$
|V| = 256 + M + |\text{specials}|
$$ (eq:byte-bpe-vocab)

**There is no unknown token and there cannot be one.** This is the property that
made byte-level BPE the default for generative models: a model that may be
prompted with anything cannot afford an input it is unable to represent.

The cost is fertility on non-Latin scripts, where one character is two to four
bytes before any merge applies — which is the mechanism behind
{{cite:petrov2023}}'s measured disparities.

## 6. Mathematical Foundation

### 6.1 BPE by hand, completely

Corpus (word: frequency):

$$
\{\ \texttt{low}: 5,\quad \texttt{lower}: 2,\quad \texttt{newest}: 6,\quad
   \texttt{widest}: 3\ \}
$$

Initial symbols are characters, with a word-boundary marker `_` appended to each
word. Pair counts, weighted by word frequency:

- `(e,s)`: appears in `newest` (6) and `widest` (3) → **9**
- `(s,t)`: `newest` (6) + `widest` (3) → **9**
- `(l,o)`: `low` (5) + `lower` (2) → 7
- `(o,w)`: `low` (5) + `lower` (2) → 7
- `(t,_)`: `newest` (6) + `widest` (3) → 9
- `(w,e)`: `newest` (6) + `lower` (2) → 8

Three pairs tie at 9. Ties are broken by a fixed rule — first encountered — which
is worth noticing: **the vocabulary depends on an arbitrary tie-break.** Take
`(e,s)`.

**Merge 1: `e`+`s` → `es`.** Now `newest` = `n e w es t _`, `widest` = `w i d es
t _`.

Recount: `(es,t)` = 6 + 3 = **9**, still the maximum.

**Merge 2: `es`+`t` → `est`.** Now `newest` = `n e w est _`.

Recount: `(est,_)` = 9.

**Merge 3: `est`+`_` → `est_`.** The suffix and the word boundary are now one
token.

Recount: `(l,o)` = 7, `(o,w)` = 7, `(w,e)` = 6 + 2 = 8 → take `(w,e)`.

**Merge 4: `w`+`e` → `we`.**

After four merges the vocabulary is the characters plus
$\{\texttt{es}, \texttt{est}, \texttt{est\_}, \texttt{we}\}$, and `newest`
encodes as `n we est_` — three tokens instead of seven.

**Note what happened at merge 3.** The algorithm glued a suffix to a word
boundary, producing a token that means "…est at end of word". That is a
morphologically sensible unit, and it arrived from pure frequency counting with
no linguistic input. **This is why BPE looks like it learns morphology, and why
that impression is misleading** — the same procedure will just as happily produce
a token spanning the end of one morpheme and the start of the next, whenever the
counts point that way.

### 6.2 Why WordPiece's denominator matters

Take a corpus where $c(\texttt{t}) = 10{,}000$, $c(\texttt{h}) = 8{,}000$,
$c(\texttt{th}) = 5{,}000$, and separately $c(\texttt{q}) = 200$,
$c(\texttt{u}) = 3{,}000$, $c(\texttt{qu}) = 195$.

BPE's criterion:

$$
c(\texttt{th}) = 5000 \;>\; c(\texttt{qu}) = 195 \implies \text{merge } \texttt{th}
$$

WordPiece's criterion {{eq:wordpiece-score}}:

$$
\frac{5000}{10000 \times 8000} = 6.25\times10^{-5},
\qquad
\frac{195}{200\times 3000} = 3.25\times 10^{-4}
$$

$$
\implies \text{merge } \texttt{qu},\ \text{by a factor of } 5.2
$$ (eq:qu-vs-th)

**WordPiece prefers `qu`, and it is right to.** In English `q` is almost always
followed by `u`, so `qu` is a genuine unit — knowing you have a `q` tells you
almost everything about the next symbol. `th` is common because `t` and `h` are
both common. The PMI criterion detects the difference and the frequency
criterion cannot.

### 6.3 Viterbi segmentation is exact and cheap

Let $w$ have length $n$ and let $\text{best}(i)$ be the log probability of the
best segmentation of the prefix $w_{1:i}$. Then

$$
\text{best}(i) = \max_{j < i,\ w_{j+1:i}\,\in\,V}
 \Big[\text{best}(j) + \log p(w_{j+1:i})\Big],
 \qquad \text{best}(0) = 0
$$ (eq:viterbi-recurrence)

Each of the $n$ positions considers at most $n$ predecessors, so the exact
maximum over an exponentially large set costs $O(n^2)$ — and in practice $O(nL)$
with $L$ the longest vocabulary entry.

$\square$

**This is the concrete difference between unigram and BPE.** BPE's output is
whatever its greedy merge order produced. Unigram's output is the
provably-highest-probability segmentation under an explicit model. Whether that
matters for downstream quality is an empirical question and the answer is
"slightly, sometimes" — but it means unigram can *sample* alternatives and BPE
cannot.

## 7. Internal Mechanics

```mermaid {#fig:bpe-training caption="BPE training and encoding. Training produces an ordered merge list; encoding replays it. The order is load-bearing — merge k may operate on symbols that merge j<k created, so applying the list out of order produces a different, silently wrong segmentation."}
graph TD
  subgraph TRAIN["fit, once"]
    A["corpus → word frequencies"] --> B["split every word<br/>into characters"]
    B --> C["count adjacent pairs<br/>weighted by word frequency"]
    C --> D{"budget<br/>exhausted?"}
    D -- no --> E["merge the argmax pair<br/>append to merge list"]
    E --> C
    D -- yes --> F["merge list m₁…m_M<br/>+ vocabulary"]
  end
  subgraph ENCODE["serve, every request"]
    G["word → characters"] --> H["apply m₁, then m₂, …<br/>in learned order"]
    H --> I["token strings → IDs"]
  end
  F -.->|frozen artefact| H
  style F fill:#dfe,stroke:#5a5
```

**The training loop's cost.** A naive implementation recounts every pair after
every merge, which is $O(M \cdot |C|)$ for corpus size $|C|$ and $M$ merges — and
with $M = 50{,}000$ on a large corpus that is prohibitive. Real implementations
keep an index from each pair to the words containing it, and after a merge update
only the affected words. The from-scratch implementation in
{{sec:8-implementation}} does the naive version because it is the one you can
read; the optimisation is an exercise.

**Word-frequency representation.** Notice that training operates on *unique words
with counts*, not on the raw token stream. A corpus of a billion tokens may have
only a few million distinct words, so this collapses the work by two or three
orders of magnitude — and it is why the word-boundary marker matters: without it,
merges could cross word boundaries and the collapse would be invalid.

**Encoding cost.** Applying $M$ merges to a word of length $n$ naively costs
$O(Mn)$. Production encoders instead use a priority queue over merge ranks, which
gives $O(n\log n)$ per word, and cache results per unique word — which is very
effective, because word frequencies are Zipfian.

## 8. Implementation

A complete BPE trainer and encoder. This is the whole algorithm; there is nothing
omitted for brevity.

```python {tier=A name=bpe-from-scratch}
"""Byte pair encoding: train a merge table, then apply it. Complete."""
from collections import Counter

CORPUS = """
the quick brown fox jumps over the lazy dog the quick brown fox
lowest lower low newest newer new widest wider wide
the lowest of the low and the newest of the new
tokenization tokenizer tokenized tokenizing token tokens
"""


def word_freqs(text):
    """Training operates on unique words with counts, not the raw stream."""
    return Counter(text.split())


def pair_counts(splits, freqs):
    """Count adjacent symbol pairs, weighted by the frequency of their word."""
    pairs = Counter()
    for word, symbols in splits.items():
        f = freqs[word]
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += f
    return pairs


def merge_pair(splits, pair):
    """Replace every adjacent occurrence of `pair` with the joined symbol."""
    a, b = pair
    joined = a + b
    out = {}
    for word, symbols in splits.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(joined)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        out[word] = merged
    return out


def train_bpe(text, n_merges):
    """Returns the ordered merge list and the resulting vocabulary."""
    freqs = word_freqs(text)
    # '_' marks the end of a word, so merges cannot cross word boundaries and
    # a suffix at word-end is distinguishable from the same letters inside.
    splits = {w: list(w) + ["_"] for w in freqs}
    vocab = {s for symbols in splits.values() for s in symbols}
    merges = []
    for _ in range(n_merges):
        pairs = pair_counts(splits, freqs)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # deterministic tie-break
        if pairs[best] < 2:
            break
        merges.append(best)
        vocab.add(best[0] + best[1])
        splits = merge_pair(splits, best)
    return merges, vocab


def encode(word, merges):
    """Replay the merges in the order they were learned. Order is load-bearing."""
    symbols = list(word) + ["_"]
    for a, b in merges:
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


merges, vocab = train_bpe(CORPUS, n_merges=40)

print(f"vocabulary: {len(vocab)} symbols after {len(merges)} merges")
print(f"first ten merges: {[a + b for a, b in merges[:10]]}\n")

for w in ["token", "tokenizer", "tokenizers", "unbelievable", "lowest"]:
    pieces = encode(w, merges)
    print(f"{w:<14} -> {' '.join(pieces):<40} ({len(pieces)} tokens)")

# The critical property: a word never seen in training is still representable,
# because the character symbols are always in the vocabulary.
assert "unbelievable" not in CORPUS
assert "".join(encode("unbelievable", merges)).rstrip("_") == "unbelievable"
print("\nunseen word encodes losslessly — no UNK is possible")
```

`tokenizers` was never in the training corpus, and it still encodes — reusing the
pieces learned from `tokenizer` and `tokens`. That is the open-vocabulary
property, and it is the entire reason this replaced word-level segmentation.

Now the vocabulary-size curve, which is the measurement that should precede
choosing $|V|$:

```python {tier=A name=merges-vs-compression}
"""How much compression does each additional merge buy?"""
from collections import Counter

CORPUS = """
the quick brown fox jumps over the lazy dog the quick brown fox
lowest lower low newest newer new widest wider wide
the lowest of the low and the newest of the new
tokenization tokenizer tokenized tokenizing token tokens
statistical machine learning learns statistics from machines
"""


# A minimal trainer, re-declared so this listing stands alone.
def _wf(t):
    return Counter(t.split())


def _pairs(sp, fr):
    p = Counter()
    for w, s in sp.items():
        for a, b in zip(s, s[1:]):
            p[(a, b)] += fr[w]
    return p


def _merge(sp, pair):
    a, b = pair
    out = {}
    for w, s in sp.items():
        m, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                m.append(a + b); i += 2
            else:
                m.append(s[i]); i += 1
        out[w] = m
    return out


def total_tokens(splits, freqs):
    return sum(len(s) * freqs[w] for w, s in splits.items())


freqs = _wf(CORPUS)
splits = {w: list(w) + ["_"] for w in freqs}
base = total_tokens(splits, freqs)
n_words = sum(freqs.values())
n_chars = sum(len(w) * f for w, f in freqs.items())

print(f"{'merges':>7} {'vocab':>7} {'tokens':>8} {'fertility':>10} "
      f"{'chars/token':>12} {'marginal':>9}")
prev = base
for step in range(0, 61):
    if step:
        p = _pairs(splits, freqs)
        if not p:
            break
        best = max(p, key=lambda k: (p[k], k))
        splits = _merge(splits, best)
    if step % 10 == 0:
        t = total_tokens(splits, freqs)
        marginal = (prev - t) / 10 if step else 0.0
        print(f"{step:>7} {len(set(s for v in splits.values() for s in v)):>7} "
              f"{t:>8} {t / n_words:>10.2f} {n_chars / t:>12.2f} "
              f"{marginal:>9.1f}")
        prev = t

print("\nThe marginal column is the point: the first merges buy a great deal "
      "and later ones buy progressively less, which is the curve that should "
      "decide the vocabulary size.")
```

The `chars/token` column is the compression ratio, and it is the number that
converts to money — more characters per token means fewer tokens per request.

Finally, WordPiece's criterion contrasted with BPE's on the same corpus, which
makes {{eq:qu-vs-th}} concrete:

```python {tier=A name=wordpiece-vs-bpe-criterion}
"""The same pair counts, ranked by frequency and by the WordPiece score."""
from collections import Counter

TEXT = ("the theory that the queen requires a quiet quarter is quite the "
        "thing that these theorists think through thoroughly, and the "
        "quantity of quotations they require is quite the quandary")

symbols = Counter(TEXT.replace(" ", ""))
pairs = Counter()
for word in TEXT.split():
    for a, b in zip(word, word[1:]):
        pairs[(a, b)] += 1

rows = []
for (a, b), c_ab in pairs.items():
    if c_ab < 2:
        continue
    bpe_score = c_ab
    wp_score = c_ab / (symbols[a] * symbols[b])      # equation (eq:wordpiece-score)
    rows.append((a + b, c_ab, symbols[a], symbols[b], bpe_score, wp_score))

print("Top 6 by BPE's criterion (raw joint count):")
for p, c, ca, cb, _, wp in sorted(rows, key=lambda r: -r[4])[:6]:
    print(f"  {p:<4} count={c:<4} c(a)={ca:<4} c(b)={cb:<4} wp={wp:.2e}")

print("\nTop 6 by WordPiece's criterion (pointwise mutual information):")
for p, c, ca, cb, _, wp in sorted(rows, key=lambda r: -r[5])[:6]:
    print(f"  {p:<4} count={c:<4} c(a)={ca:<4} c(b)={cb:<4} wp={wp:.2e}")

qu = next((r for r in rows if r[0] == "qu"), None)
th = next((r for r in rows if r[0] == "th"), None)
if qu and th:
    print(f"\nqu: count {qu[1]}, wp score {qu[5]:.2e}")
    print(f"th: count {th[1]}, wp score {th[5]:.2e}")
    print(f"BPE prefers 'th' by count ({th[1]} > {qu[1]}); "
          f"WordPiece prefers 'qu' by {qu[5] / th[5]:.1f}x")
```

The two rankings disagree, and the disagreement is systematic: BPE's list is
dominated by pairs of common letters, WordPiece's by pairs that genuinely
predict each other.

## 9. Practical Example

A team is fitting a tokenizer for a code assistant. The obvious move is to copy a
general-purpose configuration — 32,000 merges, fitted on web text — because that
is what everything else uses. The measurement that shows why this is wrong takes
under a minute.

Code has a different symbol distribution from prose: four-space indents, `_` and
`.` inside identifiers, `()` and `[]` at high frequency, and camelCase and
snake_case compounds that a prose-fitted vocabulary has never seen as units.

```python {tier=A name=domain-fitted-vocabulary}
"""Fitting on the wrong domain has a measurable, and large, cost."""
from collections import Counter

PROSE = """
the report describes how the team measured the effect of the change on the
users of the service and the results were consistent with the earlier study
which described a similar effect in a different population of users
"""

CODE = """
def get_user_by_id(self, user_id): return self._db.get_user(user_id)
def get_user_by_name(self, user_name): return self._db.get_user_by(user_name)
def set_user_name(self, user_id, user_name): self._db.set_user(user_id, user_name)
class UserRepository: def __init__(self, db_conn): self._db = db_conn
"""


def train(text, n_merges):
    freqs = Counter(text.split())
    splits = {w: list(w) + ["_"] for w in freqs}
    merges = []
    for _ in range(n_merges):
        p = Counter()
        for w, s in splits.items():
            for a, b in zip(s, s[1:]):
                p[(a, b)] += freqs[w]
        if not p or max(p.values()) < 2:
            break
        best = max(p, key=lambda k: (p[k], k))
        merges.append(best)
        a, b = best
        new = {}
        for w, s in splits.items():
            m, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    m.append(a + b); i += 2
                else:
                    m.append(s[i]); i += 1
            new[w] = m
        splits = new
    return merges


def apply_merges(text, merges):
    total = 0
    for word in text.split():
        s = list(word) + ["_"]
        for a, b in merges:
            m, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    m.append(a + b); i += 2
                else:
                    m.append(s[i]); i += 1
            s = m
        total += len(s)
    return total


prose_merges = train(PROSE, 40)
code_merges = train(CODE, 40)
words = len(CODE.split())

matched = apply_merges(CODE, code_merges)
mismatched = apply_merges(CODE, prose_merges)

print(f"code text: {words} words")
print(f"  tokenized with a code-fitted vocabulary:  {matched:>4} tokens "
      f"(fertility {matched / words:.2f})")
print(f"  tokenized with a prose-fitted vocabulary: {mismatched:>4} tokens "
      f"(fertility {mismatched / words:.2f})")
print(f"  penalty for the mismatch: {mismatched / matched:.2f}x")
print("\nThe penalty is paid on every request, forever, and it is invisible "
      "unless it is measured before the vocabulary is frozen.")
```

The mismatch penalty is a permanent multiplier on serving cost and a permanent
subtraction from usable context. It is also unfixable after pretraining, which is
what makes this a decision worth ten minutes of measurement.

> PRODUCTION TIP: Fit a candidate vocabulary on a sample of your *actual traffic*
> and compare its fertility against the vocabulary you were going to adopt. If
> the ratio is above roughly 1.2, the domain mismatch is costing more than most
> inference optimisations will ever save.

## 10. Production Considerations

**The merge list is a frozen artefact and must be versioned as one.** It is not
configuration and it is not derivable — it is the output of a fit over a corpus
that may no longer exist. Store it with the model weights, hash it, and refuse to
load a mismatched pair ({{ch:mle-registry}}).

**Encoding throughput matters at ingestion scale.** A naive $O(Mn)$ encoder is
fine for chat and disastrous for indexing a document corpus ({{part:12}}).
Production encoders are compiled and use a merge-rank priority queue; cache
encoded results keyed on the unique word.

**Reserve special-token IDs at fit time.** Allocate a block of unused IDs before
freezing. Adding `[TOOL_CALL]` later without reserved space means either
retraining the embedding matrix or appending IDs whose embeddings were never
trained.

**Vocabulary size is a serving-cost decision, not only a quality one.** Larger
$|V|$ costs $2|V|d$ parameters and a wider softmax at every decoding step
({{ch:llm-decoding}}), and buys lower fertility on every request. At high volume
the fertility saving usually wins; at small scale the parameter cost usually
does.

**Metrics to log:** compression ratio (characters per token) on live traffic,
byte-fallback rate, and encode latency at p99. The first two drift as traffic
changes; the third does not drift but does regress on library upgrades.

## 11. Common Mistakes

**Beginners:**

*Believing BPE learns morphology.* It learns frequency. Morpheme-like units are a
frequent by-product, not a goal, and counting on them is unwise —
{{sec:6-mathematical-foundation}} showed a suffix emerging and also showed why
nothing guarantees it.

*Applying merges in the wrong order, or as a set.* The merge list is ordered and
later merges consume symbols earlier ones created. Applying them as an unordered
set produces different, silently wrong output.

*Assuming two tokenizers with the same vocabulary size are comparable.* Fitted on
different corpora they have entirely different behaviour on your text; only
fertility on your traffic compares them.

**Experienced practitioners:**

*Fitting the vocabulary on cleaned data and serving raw data.* The fit sees a
distribution the traffic does not match, and fertility on production is worse
than the fit reported.

*Ignoring the tie-break.* {{sec:6-mathematical-foundation}}'s worked example had a
three-way tie at the first merge. Different implementations break ties
differently, so "BPE with 32k merges" does not uniquely determine a tokenizer —
which matters when reproducing a published result.

*Conflating SentencePiece with unigram.* SentencePiece implements both BPE and
unigram; its contribution is the raw-Unicode input convention. Saying "we used
SentencePiece" does not state which merge algorithm is running.

*Reusing a tokenizer across model families.* Token counts differ by tens of
percent, so any budget or context calculation carried across is wrong.

## 12. Failure Modes

**Domain mismatch.** The vocabulary was fitted on a distribution the traffic does
not match, so fertility is permanently elevated. *Symptom:* cost per request
higher than a comparable system, with no quality difference. *Detection:*
{{sec:9-practical-example}}'s measurement. *Fix:* only at pretraining time, or by
vocabulary adaptation ({{ch:nlp-preprocessing}} §15).

**Greedy suboptimality.** BPE's output is not the shortest segmentation available
from its own vocabulary. A word may encode in five pieces when four exist,
because the greedy order committed early. *Symptom:* invisible, absent a
comparison against a Viterbi decode over the same vocabulary. *Consequence:* a
few percent of unnecessary fertility.

**Frozen vocabulary versus drifting traffic.** New product names, new libraries,
new slang all arrive after the fit. They tokenize into fragments, and the
byte-fallback rate creeps up over months. *Detection:* trend the compression
ratio on live traffic, not on a fixed benchmark.

**Number and identifier fragmentation.** BPE assigns whole tokens to frequent
numbers and splits rare ones arbitrarily; the same happens to identifiers in
code. *Symptom:* arithmetic and identifier-copying errors that depend on the
specific string. *Mitigation:* some tokenizers force digits to split
individually, which trades fertility for consistency — a deliberate choice
against the compression objective.

**Whitespace-boundary ambiguity in non-SentencePiece pipelines.** Detokenization
guesses spacing back, and the guess is wrong for code, for CJK text, and for
anything with meaningful runs of whitespace. *Symptom:* generated code with
mangled indentation and round-trip tests that fail on whitespace only.

## 13. Alternatives

{#tbl:subword-algorithms caption="The three merge criteria and what each optimises. All three produce complete vocabularies when combined with a character or byte base alphabet; they differ in what they consider a good unit and in whether a segmentation has a probability."}

| Algorithm | Criterion | Direction | Segmentation | Probabilistic |
|---|---|---|---|---|
| BPE {{cite:sennrich2016}} | joint count $c(ab)$ | grow by merging | greedy, deterministic | no |
| WordPiece {{cite:schuster2012}} | PMI $c(ab)/c(a)c(b)$ | grow by merging | greedy longest-match | implicitly |
| Unigram {{cite:kudo2018subword}} | likelihood loss on removal | shrink by pruning | Viterbi, exact | **yes** |
| Byte-level BPE {{cite:radford2019}} | joint count, over bytes | grow by merging | greedy | no |
| Word-level | frequency cutoff | — | trivial | no |
| Character / byte | none | — | trivial | no |

**Which to choose.** For a generative model that may be prompted with anything,
byte-level BPE — completeness is not negotiable. For an encoder over a known
domain, WordPiece and unigram are both reasonable and the difference is small.
For a multilingual model, unigram with SentencePiece is the common choice, partly
because subword regularization helps most where data is thinnest.

**What they all approximate versus compute exactly.** BPE and WordPiece compute
*a* segmentation. Unigram computes *the* most probable segmentation under its
model {{eq:unigram-viterbi}}, which is exact — but the model itself is a unigram
approximation, so exactness of the decode does not mean correctness of the
result. Being precise about which layer is exact is worth the sentence.

## 14. Evaluation

**Is the implementation correct?**

1. **Round trip.** `decode(encode(s)) == s` for every $s$ in a corpus including
   emoji, CJK text, and code. For byte-level BPE this must hold for arbitrary
   bytes, including invalid UTF-8.
2. **Determinism across runs and platforms**, including the tie-break.
3. **Merge order is respected** — a test that shuffles the merge list and asserts
   the output changes is a genuine regression test.
4. **Completeness** — no input in a held-out sample produces `UNK`.

**Is this vocabulary good for this traffic?**

1. **Compression ratio** (characters per token) on a real traffic sample. This is
   the primary number.
2. **Fertility** ({{eq:fertility}}) per language and per content type.
3. **The marginal-merge curve** from `merges-vs-compression`, which is what makes
   $|V|$ a measured decision rather than a copied one.
4. **Downstream task performance with the tokenizer as the only varied
   component** — expensive, definitive, and the only one that answers the real
   question.

A tokenizer that wins on 1-3 and loses on 4 should lose. In practice 1-3 are run
continuously and 4 is run once, before the vocabulary is frozen forever.

## 15. Advanced Concepts

**Subword regularization.** {{maturity:ESTABLISHED}} Sampling from
{{eq:unigram-model}}'s distribution over segmentations during training, so the
model sees the same string split differently across epochs
({{cite:kudo2018subword}}). It is a real augmentation and it helps most in
low-resource settings. BPE-dropout achieves a similar effect by randomly skipping
merges.

**Vocabulary trimming and expansion after pretraining.** {{maturity:EMERGING}}
Removing unused rows to shrink a deployed model, or adding domain tokens with
embeddings initialised from the average of their current pieces. The second is
one of the few post-hoc interventions available for a frozen tokenizer.

**Tokenizer transfer between models.** {{maturity:EMERGING}} Re-embedding a
model's vocabulary into another's so weights can be reused across tokenizers.
Motivated by model merging and by adapting English-fitted models to other
languages.

**Learned or differentiable segmentation.** {{maturity:RESEARCH FRONTIER}}
Training the segmentation jointly with the model. The obstacle is that
segmentation is discrete, and differentiable relaxations have not yet been made
to pay for themselves at scale.

**Multi-token prediction and tokenizer-aware decoding.** {{maturity:EMERGING}}
Once you accept that tokens are a compression artefact, "predict the next token"
is an arbitrary unit of prediction — and predicting several at once is a natural
question. {{part:16}} returns to this.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-preprocessing}} set up {{eq:tokenizer-tradeoff}}, and
this chapter is its resolution: subword vocabularies achieve completeness *and*
low fertility, which is why word-level and character-level segmentation both
disappeared. {{ch:tf-embeddings}} priced $|V|$ in parameters, which is one side
of the vocabulary-size decision. {{ch:math-probability}} supplies the likelihood
that {{eq:unigram-model}} maximises, and {{ch:math-random-vars}} the expected
counts the EM step uses. {{ch:ds-feature-eng}}'s point that features are fitted
artefacts requiring the same versioning discipline as models applies here in its
strongest form.

**Forwards.** {{ch:llm-anatomy}} takes the token IDs this chapter produces and
follows them to logits. {{ch:llm-decoding}} shows the softmax over $|V|$ at every
step, which is the other side of the vocabulary-size cost. {{ch:rag-chunking}}
faces the same cut-the-text problem at document granularity, with the same
absence of a principled objective. {{part:15}} counts the embedding matrix as a
memory term when quantising for local execution.

## 17. Exercises

**Beginner**

1. Run BPE by hand for three merges on the corpus `{aaab: 3, aab: 2}` and give
   the vocabulary and the merge list.
2. Explain in two sentences why the merge list is ordered rather than a set.
3. Given $c(a)=100$, $c(b)=50$, $c(ab)=40$ and $c(x)=1000$, $c(y)=900$,
   $c(xy)=300$, say which pair BPE merges and which WordPiece merges, with the
   arithmetic.

**Intermediate**

4. Prove that a byte-level BPE vocabulary is complete for every possible input
   byte string, and state where the argument would fail for a character-level
   base alphabet.
5. Using {{eq:viterbi-recurrence}}, segment `abcabc` under
   $V=\{a,b,c,ab,bc,abc\}$ with $p = (0.1, 0.1, 0.1, 0.25, 0.2, 0.25)$. Show
   the DP table.
6. Explain why training on unique words with counts, rather than the token
   stream, is valid — and give a case where it would not be.

**Advanced**

7. Construct a corpus and a vocabulary where BPE's greedy segmentation is
   strictly longer than the Viterbi segmentation over the same vocabulary.
   Quantify the gap.
8. WordPiece's criterion is derived as the likelihood gain from a merge. Carry
   out that derivation and identify the approximation that turns it into
   {{eq:wordpiece-score}}.
9. Argue for or against forcing digits to tokenize individually, using the
   fertility cost and the arithmetic-consistency benefit.

**Implementation**

10. Replace the naive recount in `bpe-from-scratch` with an incremental index
    from pairs to affected words. Measure the speedup as a function of merge
    count and confirm the output is byte-identical.
11. Implement the unigram tokenizer: seed a candidate vocabulary from all
    substrings up to length 6, run EM, prune to a target size, and decode with
    the Viterbi recurrence {{eq:viterbi-recurrence}}. Compare its fertility
    against BPE at the same vocabulary size.
12. Implement subword regularization by sampling segmentations from the unigram
    model, and show that the same string yields different splits across draws.
13. Extend `bpe-from-scratch` to byte level: operate on `bytes` rather than
    `str`, and verify with a property test that arbitrary byte strings round
    trip — including sequences that are not valid UTF-8.

**Reasoning**

14. A colleague proposes doubling the vocabulary to halve fertility. Explain why
    the second half of that sentence is wrong, using {{eq:zipf-coverage}}.
15. Explain why unigram can do subword regularization and BPE cannot, in terms of
    what each algorithm's output actually is.

## 18. Interview Questions

**Beginner**

1. What problem does subword tokenization solve that word-level does not?
2. Walk through BPE training on a three-word corpus.
3. What is the difference between a merge list and a vocabulary?

**Intermediate**

4. How does WordPiece's merge criterion differ from BPE's, and why does the
   difference matter?
5. What exactly does SentencePiece contribute? Is it an algorithm?
6. Why can byte-level BPE never emit an unknown token?

**Senior**

7. You are choosing a vocabulary size for a model that will serve mostly code.
   What do you measure, and what do you trade off?
8. A model has a tokenizer fitted on 2019 web text and now serves 2026 traffic.
   What degrades, how would you detect it, and what can you actually do?
9. When would you choose unigram over BPE? Be specific about the evidence.

**Systems**

10. Design a tokenization service for a corpus-ingestion pipeline handling a
    billion documents. Address throughput, caching, and versioning.
11. Two models in one product have different tokenizers. How do you build a
    cost-estimation API that is correct for both?

## 19. Research Questions

**Is there a segmentation objective aligned with downstream loss?** Every
algorithm here optimises compression or likelihood of the *text*, not the
performance of the *model*. What would it take to optimise the thing we care
about, and why has nobody made it work? Begin from {{cite:kudo2018subword}}'s
objective and ask what it is a proxy for.

**How much does the greedy gap cost?** Exercise 7 constructs a case where BPE's
segmentation is longer than necessary. Measure it at scale: what fraction of
fertility on a real corpus is pure greedy suboptimality, recoverable by Viterbi
decoding over the identical vocabulary? The experiment is a weekend's work and
the answer does not appear to be published.

**Can vocabulary allocation be made fair across languages?**
{{cite:petrov2023}} measured the disparity. Formulate the allocation as a
constrained optimisation — minimise the worst-case fertility subject to a
vocabulary budget — and ask what the English cost of the fair solution is.

**Does the tokenizer's arbitrariness matter at scale?** Larger models appear more
robust to tokenization pathologies. Is that a real trend or a measurement
artefact, and if real, does it mean the problem solves itself?

## 20. Chapter Summary

Subword tokenization resolves the vocabulary-versus-sequence-length tradeoff by
letting corpus statistics decide the units: frequent strings stay whole, rare
ones decompose into pieces, and because the base alphabet is always present, no
input is unrepresentable.

**BPE** merges the most frequent adjacent pair, repeatedly, producing an ordered
merge list that must be replayed in order at encoding time. It is
{{cite:gage1994}}'s file-compression algorithm applied unchanged to text by
{{cite:sennrich2016}}, and it has no model of language.

**WordPiece** merges the pair maximising $c(ab)/c(a)c(b)$, which
{{eq:wordpiece-pmi}} shows is pointwise mutual information. The denominator is
the whole difference from BPE, and it is what makes `qu` beat `th`.

**Unigram** goes the other way — start large, prune by likelihood loss — and is
the only one of the three with an explicit probability model. That model gives an
exact Viterbi decode {{eq:viterbi-recurrence}} and a distribution over
segmentations, which BPE cannot provide and which subword regularization
requires.

**SentencePiece** is orthogonal to all three: it feeds in raw Unicode with
whitespace as an ordinary symbol, which makes segmentation lossless, reversible,
and correct for languages without whitespace word boundaries.

**Byte-level BPE** makes completeness structural — 256 base symbols means the
unknown token cannot exist — at the cost of fertility on non-Latin scripts.

The practical consequences are measurable and mostly ignored. Vocabulary size
should come from the marginal-compression curve rather than from convention.
Domain mismatch between the fitting corpus and the traffic is a permanent
multiplier on cost. And the merge list is a frozen artefact that must be
versioned with the model, because it is not derivable from anything.

## 21. Further Reading

{{cite:sennrich2016}} is three pages of algorithm and should be read in full. The
striking thing on a careful reading is how little is there: the method section is
one figure and one paragraph, and the paper's contribution is the observation
that an existing compression algorithm solves an open-vocabulary problem nobody
had connected it to.

{{cite:kudo2018subword}} is the most technically interesting paper in this
chapter, because it is the only one that starts from an objective. Sections 3.1
and 3.2 give the model and the EM procedure compactly. Read it after implementing
BPE, when the contrast will land.

{{cite:kudo2018sentencepiece}} is a demo paper and reads like one, but §2 is the
clearest statement anywhere of why whitespace pre-tokenization is a bug rather
than a convenience.

{{cite:schuster2012}} is the WordPiece origin and is behind a paywall; §3.1 of
{{cite:wu2016}} is the accessible description and is what most implementations
actually follow. Read the latter and cite the former.

{{cite:gage1994}} is worth finding for one reason: reading a 1994 data-compression
article and recognising the input layer of a modern language model in it,
unchanged, is the most efficient available cure for the belief that tokenizers
know something about language.

**Where to go next:** {{ch:nlp-static-embeddings}} takes the units this chapter
produces and asks what vector each should get — the first learned representation
in the part, and the origin of the geometry that {{part:11}} builds
infrastructure around.
