---
id: nlp-preprocessing
number: 72
part: VIII
tier: full
status: draft
requires: [tf-embeddings, tf-architectures, ds-cleaning, py-fundamentals,
           ml-metrics]
provides: [tokenization, token, vocabulary, out-of-vocabulary, unicode-normalization,
           tokenizer-fertility, character-tokenization, word-tokenization,
           byte-tokenization, text-preprocessing-pipeline, lemmatization-stemming]
citations: [petrov2023, rust2021, sennrich2016, radford2019, kudo2018sentencepiece,
            conneau2020xlmr]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the tokenization problem formally as a choice of segmentation function
   and explain why it has no correct answer.
2. Derive the three-way tradeoff between vocabulary size, sequence length, and
   the out-of-vocabulary rate, and show that the three cannot be optimised
   independently.
3. Compute a tokenizer's **fertility** and use it to compare segmentations across
   languages and content types.
4. Explain why Unicode normalisation is a correctness requirement rather than a
   stylistic preference, and name the case where NFC and NFKC give different
   model behaviour.
5. Explain why lowercasing, stopword removal and stemming disappeared from the
   NLP pipeline, and identify the two situations where they are still correct.
6. Quantify what tokenization costs a non-English user in money, latency, and
   usable context.
7. Choose a segmentation granularity from measured cost rather than from
   convention.

## 2. Why This Matters

**Every model in the remaining twenty-one parts of this book begins with the
operation this chapter describes.** Before an embedding is looked up
({{ch:tf-embeddings}}), before a single attention score is computed, a string
becomes a list of integers. That function is not learned end to end with the
rest of the model — it is fitted once, frozen, and then depended on by
everything downstream.

**It is the last remaining hand-designed component of a modern language model,
and it is the one people think about least.** The architecture is learned. The
weights are learned. The segmentation is a greedy heuristic fitted to a corpus
somebody chose, and then it is immutable for the model's entire life.

**It has a price, in currency.** {{cite:petrov2023}} measured the same content
translated into different languages and found tokenized lengths differing by up
to a factor of fifteen. Since API pricing, latency, and context limits are all
denominated in tokens, an identical request costs a Burmese speaker an
order of magnitude more than an English speaker. That is not a rounding error
and it is not a policy decision anyone made — it falls out of which corpus the
merges were fitted on.

**It causes model failures that look like reasoning failures.** A model that
cannot compare `9.11` and `9.9` correctly may be failing at arithmetic, or its
tokenizer may have split those strings into pieces that destroy the place-value
structure. These have different fixes, and telling them apart requires knowing
what the tokenizer did. {{sec:12-failure-modes}} lists the family.

## 3. Prerequisites

{{ch:tf-embeddings}} for the embedding matrix that consumes token IDs, and for
why vocabulary size is a parameter-count decision as well as a linguistic one.
{{ch:tf-architectures}} for what a "sequence" means to a transformer.
{{ch:ds-cleaning}} for the general principle that preprocessing decisions are
modelling decisions. {{ch:py-fundamentals}} for Python's string and bytes types,
which this chapter's failure modes depend on. {{ch:ml-metrics}} for the habit of
measuring a design choice rather than arguing about it.

## 4. Intuitive Explanation

A model consumes vectors, and a vector is looked up by an integer. So somewhere
between a user typing a sentence and a matrix multiplication happening, the
sentence must become a list of integers. The question is where to cut.

Three answers are obvious and all three are bad.

**Cut at every character.** The vocabulary is tiny — a hundred symbols or so —
and no input is ever unrepresentable. But the sequence becomes five times longer
than the word count, and since attention costs $O(T^2)$ ({{ch:tf-complexity}}),
five times longer is twenty-five times more expensive. Worse, the model must
spend its early layers rediscovering that `c`,`a`,`t` is a unit, which is
capacity spent on something a segmenter could have handled for free.

**Cut at every word.** Sequences are short and each unit is meaningful. But the
vocabulary is unbounded: English has hundreds of thousands of word forms, new
ones appear constantly, and any fixed vocabulary will meet words it has never
seen. Those become a single `UNK` placeholder, which means the model literally
cannot read them — and the words most likely to be unknown are proper nouns and
technical terms, exactly the ones carrying the information.

**Cut at every byte.** Nothing is unrepresentable, ever, and the vocabulary is
exactly 256. But sequences get even longer than characters for non-ASCII text,
because a single character may be four bytes.

> NOTE: The three failures are the same failure seen from three sides. Short
> sequences require a large vocabulary; a bounded vocabulary requires small
> units; small units make long sequences. You are choosing a point on a curve,
> not finding a right answer.

The resolution — the subject of {{ch:nlp-subword}} — is to let the data choose
the units: keep frequent words whole, break rare words into pieces, and let a
statistic decide which is which. This chapter establishes what that choice is
being made *against*.

**The mental model to carry:** the tokenizer is a lossy compressor whose codebook
was fitted to one corpus, and every property of the model downstream — cost,
context length, arithmetic ability, multilingual fairness — inherits from how
well that corpus matched the text you actually send. Where the model breaks
down: the tokenizer is not *only* a compressor, because the units it produces
are the units the model can reason about atomically. A compressor that hides
structure the model needs is worse than a less efficient one that exposes it.

## 5. Formal Explanation

### 5.1 The segmentation function

Let $\Sigma$ be an alphabet (Unicode code points, or bytes) and $\Sigma^*$ the
set of finite strings over it. A **tokenizer** is a pair $(V, \tau)$ where
$V \subset \Sigma^*$ is a finite **vocabulary** of strings and

$$
\tau : \Sigma^* \to V^*
$$ (eq:tokenizer-map)

is a segmentation function mapping a string to a sequence of vocabulary items.

Two properties matter and are independent:

**Losslessness.** $\tau$ is lossless if there is a $\tau^{-1}$ with
$\tau^{-1}(\tau(s)) = s$ for all $s$. Concatenating the pieces must recover the
input exactly — including whitespace, which is where naive implementations fail.

**Completeness.** $\tau$ is complete if it is defined on all of $\Sigma^*$. A
word-level tokenizer is incomplete: it must map unseen words to a designated
$\texttt{UNK} \in V$, and that map is not injective, so losslessness fails too.

A byte-level vocabulary containing all 256 byte values is complete and lossless
by construction ({{cite:radford2019}}). This is not a small advantage — it is
the difference between "this input may be unrepresentable" and "no input is".

### 5.2 The three quantities in tension

For a corpus $C$ and a tokenizer $(V,\tau)$, define:

- **Vocabulary size** $|V|$.
- **Sequence length** $T(s) = |\tau(s)|$, the number of tokens for a string $s$.
- **OOV rate**, the fraction of corpus tokens mapped to `UNK`.

The embedding and unembedding matrices cost $2|V|d$ parameters
({{ch:tf-embeddings}}), so $|V|$ is bought with memory. Attention costs
$O(T^2)$, so $T$ is bought with compute. And the OOV rate is bought with
capability, since an `UNK` is information destroyed before the model sees it.

$$
\text{cost} \;\propto\; \underbrace{2|V|d}_{\text{parameters}}
 \;+\; \underbrace{c_1 T + c_2 T^2}_{\text{compute per sequence}}
 \;+\; \underbrace{\lambda \cdot \text{OOV}}_{\text{unpriced}}
$$ (eq:tokenizer-tradeoff)

The third term has no natural units, which is exactly why the field converged on
making it identically zero — a complete vocabulary removes the term rather than
trading against it.

### 5.3 Fertility

The standard measure for comparing tokenizers:

$$
\text{fertility}(\tau, C) = \frac{\sum_{s \in C} |\tau(s)|}{\sum_{s\in C} w(s)}
$$ (eq:fertility)

where $w(s)$ is the number of whitespace-delimited words in $s$. Fertility is
tokens per word: 1.0 means every word is one token, 3.0 means the average word
is cut into three pieces.

**Fertility is the number that translates into money.** {{cite:petrov2023}}
measured it across languages for production tokenizers and found ratios up to
15 between the best- and worst-served languages. {{cite:rust2021}} held the
model and data fixed, varied only the tokenizer, and attributed a measurable
share of the monolingual-versus-multilingual performance gap to fertility alone
— which establishes that this is a capability effect and not only a cost one.

### 5.4 What Unicode requires

A "character" is not well defined. The string `café` can be:

- **NFC**: `c`,`a`,`f`,`é` — four code points, `é` = U+00E9.
- **NFD**: `c`,`a`,`f`,`e`,`◌́` — five code points, `e` followed by a combining
  acute accent U+0301.

These are different byte sequences that render identically. If training data was
NFC and a user sends NFD, the tokenizer sees a string it has never encountered,
and the model's behaviour on it is undefined in the ordinary sense of the word.

The four normalisation forms:

{#tbl:unicode-forms caption="Unicode normalisation forms. NFC and NFD are lossless round trips of each other; NFKC and NFKD are not reversible and destroy distinctions that may matter."}

| Form | Composition | Compatibility folding | Reversible |
|---|---|---|---|
| NFC | composed | no | yes |
| NFD | decomposed | no | yes |
| NFKC | composed | yes | **no** |
| NFKD | decomposed | yes | **no** |

Compatibility folding maps `ﬁ` (the ligature, U+FB01) to `fi`, the full-width
`Ａ` to `A`, and the superscript `²` to `2`. That last one is the case to
remember: **under NFKC, `x²` and `x2` become the same string**, which is
sometimes what a search index wants and never what a mathematics-aware model
wants.

## 6. Mathematical Foundation

### 6.1 Why vocabulary size has diminishing returns

Word frequencies follow an approximate power law. Under Zipf's law, the word of
rank $r$ has frequency proportional to $r^{-\alpha}$ with $\alpha \approx 1$, so
the probability mass covered by the top $|V|$ words is

$$
\text{coverage}(|V|) = \frac{\sum_{r=1}^{|V|} r^{-\alpha}}
                            {\sum_{r=1}^{R} r^{-\alpha}}
 \;\approx\; \frac{\ln |V|}{\ln R}
$$ (eq:zipf-coverage)

for $\alpha = 1$ and total vocabulary $R$, using
$\sum_{r=1}^{n} r^{-1} \approx \ln n + \gamma$.

**Coverage grows with the logarithm of vocabulary size.** Doubling $|V|$ adds a
constant amount of coverage, while costing a constant *fraction* more parameters
— so the marginal return per parameter falls hyperbolically.

Worked numerically, with $R = 10^6$ distinct word forms:

$$
\text{coverage}(10^4) \approx \frac{\ln 10^4}{\ln 10^6} = \frac{9.21}{13.82}
 = 0.667, \qquad
\text{coverage}(10^5) \approx \frac{11.51}{13.82} = 0.833
$$

Ten times the vocabulary buys 17 percentage points of coverage — and leaves 17%
of tokens still unknown. **This is the quantitative reason word-level
tokenization was abandoned rather than scaled.** There is no vocabulary size at
which the OOV problem goes away, because the tail is infinite: proper nouns,
numbers, typos, and compounds are generated, not drawn from a fixed set.

### 6.2 The sequence-length cost of small units

Let $f$ be fertility. A document of $W$ words becomes $T = fW$ tokens, and the
per-layer attention cost is quadratic ({{ch:tf-complexity}}):

$$
\text{FLOPs}_{\text{attn}} \propto T^2 = f^2 W^2
$$ (eq:fertility-quadratic)

**Attention cost is quadratic in fertility.** A tokenizer with $f = 2.0$ on some
language does not cost twice as much as one with $f=1.0$ on the attention term;
it costs four times as much. The parameter term $2NT$ is only linear in $f$, so
which term dominates depends on context length — but at long context the
penalty compounds.

Combining with {{eq:zipf-coverage}}: shrinking the vocabulary to save embedding
parameters raises fertility, which raises attention cost quadratically. The two
levers push against each other and neither is free.

### 6.3 A worked fertility calculation

Take the string `unbelievability` (1 word, 15 characters) under three schemes:

- Character: 15 tokens, $f = 15$.
- Word, in-vocabulary: 1 token, $f = 1$.
- Word, out-of-vocabulary: 1 token (`UNK`), $f = 1$ — but with all information
  destroyed, which fertility does not measure.
- Subword: `un`,`bel`,`iev`,`abil`,`ity` — 5 tokens, $f = 5$.

**Fertility alone does not rank these**, because the OOV case has the best
fertility and the worst behaviour. Fertility is only comparable between complete
tokenizers, which is one more reason the field standardised on completeness.

## 7. Internal Mechanics

The pipeline that turns a string into tensor input has four stages, and each is
a place where information is destroyed:

```mermaid {#fig:tokenizer-pipeline caption="The four stages between a user's string and the model's first matrix multiplication. Stages 1 and 2 are lossy by design; stages 3 and 4 are lossless bookkeeping. Everything downstream sees only the output of stage 4."}
graph LR
  A["raw string<br/>bytes from the wire"] --> B["1 · normalise<br/>NFC / NFKC, casing"]
  B --> C["2 · segment<br/>string → subword strings"]
  C --> D["3 · map to IDs<br/>vocabulary lookup"]
  D --> E["4 · assemble<br/>specials, truncation, padding"]
  E --> F["tensor of int64<br/>shape (B, T)"]
  style B fill:#fde,stroke:#c69
  style C fill:#fde,stroke:#c69
```

**Stage 1, normalisation**, applies a Unicode form and any casing decision. It is
irreversible whenever it folds distinctions, and it must be identical at training
and inference time or the model sees inputs from a distribution it never saw.

**Stage 2, segmentation**, is the algorithm {{ch:nlp-subword}} covers. Note that
it operates on the *normalised* string, so a normalisation change silently
invalidates a fitted vocabulary.

**Stage 3, ID mapping**, is a dictionary lookup. The only subtlety is that the
mapping is fixed at fit time: adding a token later shifts nothing, but any ID
already assigned can never be reused, which is why vocabularies are padded to a
round number with reserved slots.

**Stage 4, assembly**, adds special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`),
truncates to the context limit, and pads a batch to a common length with an
attention mask marking the padding ({{ch:tf-masking-kv}}).

> WARNING: Truncation at stage 4 is silent by default in most libraries. A
> document longer than the context limit is cut, and the model answers
> confidently about a document whose second half it never received. Log the
> truncation rate; it belongs on a dashboard, not in a debug print.

## 8. Implementation

The three baseline segmentation schemes, and the measurement that compares them.
There is nothing learned here yet — that is {{ch:nlp-subword}} — but the
measurement harness built here is what the next chapter's tokenizer is judged
against.

```python {tier=A name=baseline-tokenizers}
"""The three naive segmentations, and the fertility measurement that ranks them."""
import unicodedata

SAMPLES = {
    "english":  "The quick brown fox jumps over the lazy dog near the riverbank.",
    "german":   "Die Donaudampfschifffahrtsgesellschaft veroeffentlichte gestern "
                "ihren Geschaeftsbericht.",
    "code":     "def fit(self, X, y): return self._solve(X.T @ X, X.T @ y)",
    "numbers":  "Revenue rose from 1234567 to 2345678 between 2023 and 2024.",
}


def char_tokenize(s):
    """One token per Unicode code point. Complete, lossless, maximally fertile."""
    return list(s)


def word_tokenize(s):
    """Whitespace segmentation. Incomplete: any unseen word becomes UNK."""
    return s.split()


def byte_tokenize(s):
    """One token per UTF-8 byte. Complete and lossless with a 256-item vocabulary."""
    return [bytes([b]) for b in s.encode("utf-8")]


def fertility(tokens, text):
    """Tokens per whitespace-delimited word — equation (eq:fertility)."""
    words = max(len(text.split()), 1)
    return len(tokens) / words


print(f"{'sample':<10} {'words':>6} {'chars':>6} {'bytes':>6} "
      f"{'f_char':>7} {'f_byte':>7}")
for name, text in SAMPLES.items():
    w = len(word_tokenize(text))
    c = len(char_tokenize(text))
    b = len(byte_tokenize(text))
    print(f"{name:<10} {w:>6} {c:>6} {b:>6} "
          f"{fertility(char_tokenize(text), text):>7.2f} "
          f"{fertility(byte_tokenize(text), text):>7.2f}")

# The vocabulary each scheme needs to cover just these four samples.
print()
print(f"character vocabulary: {len(set(''.join(SAMPLES.values())))} distinct")
print(f"word vocabulary:      "
      f"{len(set(w for t in SAMPLES.values() for w in t.split()))} distinct")
print(f"byte vocabulary:      256 by construction, always")
```

The character and word rows are the two ends of {{eq:tokenizer-tradeoff}}:
character segmentation needs a vocabulary of a few dozen and a sequence five
times longer than the word count; word segmentation needs one entry per distinct
form and produces the shortest sequence available.

Now the normalisation problem, which is the one that produces silent bugs rather
than obvious ones:

```python {tier=A name=unicode-normalisation}
"""Two strings that render identically and tokenize differently."""
import unicodedata

composed = "café"            # e-acute as one code point
decomposed = "café"          # 'e' + combining acute

print(f"visually equal:      {composed == decomposed}")
print(f"code points:         {len(composed)} vs {len(decomposed)}")
print(f"utf-8 bytes:         {len(composed.encode())} vs "
      f"{len(decomposed.encode())}")
print(f"equal after NFC:     "
      f"{unicodedata.normalize('NFC', composed) == unicodedata.normalize('NFC', decomposed)}")

# Compatibility folding destroys distinctions that NFC preserves.
pairs = [("x²", "x2"), ("ﬁre", "fire"), ("ＡBC", "ABC")]
print()
print(f"{'input':<10} {'NFC':<10} {'NFKC':<10} {'NFKC collides':>14}")
for a, b in pairs:
    nfc = unicodedata.normalize("NFC", a)
    nfkc = unicodedata.normalize("NFKC", a)
    print(f"{a!r:<10} {nfc!r:<10} {nfkc!r:<10} {str(nfkc == b):>14}")

assert unicodedata.normalize("NFKC", "x²") == "x2"
assert unicodedata.normalize("NFC", "x²") != "x2"
print("\nNFKC merges the superscript with the digit; NFC does not.")
```

The assertion at the end is the whole point: **choosing NFKC decides that `x²`
and `x2` are the same input.** For a search index that is usually right. For a
model expected to do algebra it is a capability removed before training started.

## 9. Practical Example

A support-ticket classifier is being extended from English to eight languages.
The model is unchanged; only the data is new. Accuracy on the new languages is
substantially worse, and the team's first hypothesis is that the training data is
too small.

The diagnostic that costs ten minutes and should be run first is a fertility
measurement, because if fertility is high the model is seeing a *harder problem*
on those languages regardless of data volume — more tokens per unit of meaning,
so more positions to attend over, more of the context window consumed, and a
representation that has to reassemble words the English tokenizer would have
handed over whole.

```python {tier=A name=fertility-diagnostic}
"""Fertility across scripts, with the cost consequences spelled out."""

# A rough stand-in for a Latin-fitted subword vocabulary: common English
# fragments are single tokens, everything else falls back to bytes.
LATIN_MERGES = {
    "the", "ing", "ed", "er", "tion", "ly", "an", "re", "in", "on", "at",
    "es", "is", "it", "or", "en", "of", "to", "and", "for", "with", "sup",
    "port", "ticket", "account", "pay", "ment", "please", "help",
}

DOCUMENTS = {
    "english":   "please help with my payment account support ticket",
    "german":    "bitte helfen sie mir mit meinem zahlungskonto",
    "turkish":   "lutfen odeme hesabimla ilgili bana yardim edin",
    "greek":     "παρακαλώ "
                 "βοηθήστε με",
    "japanese":  "支払いアカウントを"
                 "手伝ってください",
}


def latin_fitted_tokenize(text):
    """Greedy longest-match over the merge set; bytes for anything unmatched."""
    out, i = [], 0
    while i < len(text):
        for length in range(min(8, len(text) - i), 0, -1):
            piece = text[i:i + length]
            if piece in LATIN_MERGES:
                out.append(piece)
                i += length
                break
        else:
            # No merge applies: fall back to the UTF-8 bytes of one character.
            out.extend(bytes([b]) for b in text[i].encode("utf-8"))
            i += 1
    return out


PRICE_PER_1K = 0.003   # a representative input price, in dollars
baseline = None

print(f"{'language':<10} {'words':>6} {'tokens':>7} {'fertility':>10} "
      f"{'vs english':>11} {'$/1M docs':>11}")
for lang, text in DOCUMENTS.items():
    toks = latin_fitted_tokenize(text)
    words = len(text.split())
    f = len(toks) / words
    if baseline is None:
        baseline = f
    cost = len(toks) / 1000 * PRICE_PER_1K * 1_000_000
    print(f"{lang:<10} {words:>6} {len(toks):>7} {f:>10.2f} "
          f"{f / baseline:>10.1f}x {cost:>11,.0f}")

print("\nSame request, same information, different bill — and the ratio is a "
      "property of the merge table, not of the languages.")
```

The output makes the argument that a paragraph cannot: the non-Latin rows are
not slightly worse, they are multiples worse, and the multipliers are the same
kind of ratio {{cite:petrov2023}} measured on production tokenizers.

Read the Japanese row carefully, because it is misleading in an instructive way.
Its fertility of 51 is arithmetically correct and comparatively meaningless —
Japanese is written without spaces, so the denominator of {{eq:fertility}} is 1
for the entire sentence. **Fertility is only comparable between languages that
share a word-delimiting convention.** For the rest, compare bytes per token or
characters per token instead. That the standard metric quietly breaks on a
quarter of the world's text is the same failure as `.split()` from
{{sec:11-common-mistakes}}, one level up.

The team's data-volume hypothesis may still be true, but fertility is a second,
independent problem that more data will not fix.

## 10. Production Considerations

**Version the tokenizer with the model, always.** A model and its tokenizer are
one artefact. Serving a model with a tokenizer fitted separately — even one
differing by a single merge — produces silently wrong IDs and outputs that look
like degraded quality rather than a bug. Pin the tokenizer hash in the model
registry ({{ch:mle-registry}}) and refuse to load a mismatch.

**Normalise identically at train and serve time.** This is the same
train/serve-skew problem {{ch:mle-pipelines}} describes for features, and it has
the same fix: one code path, called from both places, never reimplemented.

**Metrics worth logging per request:** token count in, token count out,
truncation events, and the fraction of input that fell back to byte-level
tokens. The last one is the early-warning signal — a rising byte-fallback rate
means the traffic distribution is drifting away from the corpus the vocabulary
was fitted on, which shows up as cost before it shows up as quality.

**Budget for fertility when quoting costs across markets.** A per-request cost
model built on English measurements understates non-English traffic by whatever
the fertility ratio is. This is a forecasting error, not a modelling one, and it
is entirely predictable in advance.

**Tokenize once and cache.** Tokenization is cheap relative to a forward pass but
not free, and in a retrieval system ({{part:12}}) the same documents are
tokenized repeatedly. Store token IDs alongside the text.

## 11. Common Mistakes

**Beginners:**

*Treating `len(text)` as a cost estimate.* Cost is tokens, and the ratio between
characters and tokens varies by a factor of ten across the inputs a real system
sees. Measure, do not estimate.

*Stripping punctuation and lowercasing "to clean the text".* Both destroy
information a contextual model uses. `US` and `us` are different words;
`Apple's` and `Apples` are different constructions. See
{{sec:13-alternatives}} for the two cases where this is still right.

*Assuming whitespace splitting is universal.* Japanese, Chinese, and Thai are
written without spaces between words. A pipeline that begins with `.split()`
does not have a bug on those languages so much as it has no behaviour at all.

**Experienced practitioners:**

*Changing the normalisation form during a data-pipeline refactor.* This
invalidates a fitted vocabulary silently — no error is raised, tokenization
simply becomes slightly wrong in a way that shows up as a small unexplained
quality regression weeks later.

*Adding special tokens after fitting without reserving the IDs.* Every ID after
the insertion point shifts, and the embedding matrix rows no longer correspond to
the tokens they were trained for.

*Benchmarking multilingual quality without controlling for fertility.* This is
{{cite:rust2021}}'s finding: some of what looks like a modelling gap is a
tokenization gap, and the two have different fixes. Comparing models without
holding fertility fixed measures both at once.

*Trusting a token counter from a different model family.* Tokenizers are not
interchangeable and the counts differ by tens of percent. A budget computed with
one and spent on another will be wrong.

## 12. Failure Modes

**Number splitting.** A tokenizer fitted by frequency assigns single tokens to
common number strings and splits rare ones arbitrarily. `2024` may be one token
while `2027` is `20`+`27`, and `9.11` may split as `9`+`.`+`11` while `9.9` is
`9`+`.`+`9`. Arithmetic and comparison over inconsistently-segmented digits are
genuinely harder than over consistently-segmented ones. *Symptom:* arithmetic
errors that depend on the specific numbers rather than on the operation.
*Detection:* tokenize the failing inputs and look at the pieces before concluding
anything about reasoning.

**Whitespace attached to words.** Most modern tokenizers encode a leading space
as part of the token, so `" the"` and `"the"` are different IDs. A prompt ending
in a trailing space therefore puts the model in a state where the natural
continuation token — the one carrying its own leading space — is now wrong.
*Symptom:* markedly worse completions from a prompt that differs only by a
trailing space. *Detection:* strip trailing whitespace and re-run.

**Byte fallback explosion.** Text outside the fitted distribution — an unusual
script, heavy emoji, base64 — falls back to byte tokens, and one emoji becomes
four. *Symptom:* a short-looking input exceeding the context limit. *Detection:*
the byte-fallback rate metric from {{sec:10-production-considerations}}.

**Code indentation.** Python's semantics live in leading whitespace. A tokenizer
that normalises whitespace, or that segments four spaces inconsistently from
eight, damages the structure the model must reproduce. *Symptom:* generated code
with subtly wrong indentation.

**Normalisation mismatch between client and server.** Different platforms
normalise differently — macOS filesystems use NFD, most web input arrives NFC.
The same user-visible string arrives as different byte sequences from different
clients. *Symptom:* cache misses and quality differences correlated with client
platform rather than with content.

**Silent truncation.** Covered in {{sec:7-internal-mechanics}} and repeated here
because it is the most consequential: the model answers about a document whose
tail it never saw, with no signal that anything was dropped.

## 13. Alternatives

{#tbl:segmentation-alternatives caption="Segmentation granularities and what each trades away. Only the last row is used for new models; the others are here because they are still correct for specific jobs, and because subword tokenization is only comprehensible as the resolution of their tradeoffs."}

| Scheme | Vocabulary | Fertility | Complete | Trades away |
|---|---|---|---|---|
| Character | ~100 | ~5 | yes | sequence length, early-layer capacity |
| Byte | 256 | ~5-20 | yes | sequence length, worst for non-Latin scripts |
| Word | unbounded | 1.0 | **no** | completeness — the OOV problem is unfixable |
| Word + stemming | smaller | 1.0 | no | inflection, tense, number — irreversibly |
| Morphological | ~50k | ~1.5 | no | needs a hand-built analyser per language |
| Subword | 30k-256k | 1.2-2.5 | yes (byte-level) | nothing structural; the merges are arbitrary |

**Where the classical pipeline is still correct.** Two cases, and only two:

1. **Sparse lexical retrieval.** BM25 and its relatives ({{ch:emb-hybrid}}) match
   on exact terms, so stemming genuinely increases recall — `running` should
   match `run`. The information stemming destroys is information BM25 could not
   use anyway.
2. **Interpretable count-based features.** When the deliverable is a
   coefficient a human will read ({{ch:ds-feature-eng}}), a stopword-free
   stemmed bag of words produces features that mean something. The model is
   worse and the explanation is better, which is sometimes the correct trade.

Everywhere else, the pipeline exists in tutorials because it existed in 2012, and
it is subtraction from the model's input.

## 14. Evaluation

Separate two questions, as {{ch:ml-metrics}} insists.

**Is the tokenizer implementation correct?** Three properties, all testable
without a model:

1. **Round trip:** `decode(encode(s)) == s` for a corpus including emoji,
   combining characters, and mixed scripts. Any failure here is a bug, not a
   tradeoff.
2. **Determinism:** the same string yields the same IDs across processes and
   library versions.
3. **Vocabulary coverage:** no input in a held-out sample produces `UNK`.

**Is the tokenizer well suited to this traffic?** Four measurements:

1. **Fertility** ({{eq:fertility}}) per language and per content type, against
   real traffic rather than a benchmark corpus.
2. **Byte-fallback rate** — the fraction of tokens that are raw bytes.
3. **Compression ratio** — bytes of input per token. Higher is cheaper, and it
   is the number that converts directly to cost.
4. **Downstream task performance with the tokenizer as the only varied
   component**, which is {{cite:rust2021}}'s design and the only measurement that
   answers the question that actually matters.

The first three are cheap and correlate imperfectly with the fourth. Run them
continuously; run the fourth when choosing.

## 15. Advanced Concepts

**Tokenizer-free models.** {{maturity:EXPERIMENTAL}} Architectures that consume
raw bytes and learn their own pooling remove the hand-designed component
entirely. The cost is sequence length, which is attacked with hierarchical or
pooled early layers. The idea is periodically revisited and has not displaced
subword tokenization in any production system.

**Vocabulary adaptation and tokenizer transplants.** {{maturity:EMERGING}} Given
a pretrained model, replace its tokenizer with one fitted to a target language
and initialise the new embedding rows from the old ones. This directly attacks
{{cite:rust2021}}'s finding and {{cite:conneau2020xlmr}}'s capacity dilution, and
it is one of the few interventions that improves a fixed model's non-English
behaviour without retraining it.

**Learned segmentation as part of the model.** {{maturity:RESEARCH FRONTIER}}
Making $\tau$ differentiable so segmentation is trained jointly with the model.
The obstacle is that segmentation is discrete and the relaxations that make it
differentiable are expensive.

**Multilingual vocabulary allocation.** {{maturity:ESTABLISHED}} When fitting one
vocabulary over many languages, the sampling distribution over languages controls
who gets whole words and who gets bytes. {{cite:conneau2020xlmr}} documents the
resulting tradeoff at fixed capacity: transfer helps low-resource languages and
hurts high-resource ones. The allocation is a policy decision made by an
engineer, and it is worth recognising it as one.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-embeddings}} established that the embedding matrix costs
$|V|d$ parameters and is often tied to the unembedding — which is why
{{eq:tokenizer-tradeoff}}'s first term is $2|V|d$ and why vocabulary size is a
model-architecture decision. {{ch:tf-complexity}} gave the $O(T^2)$ scaling that
makes fertility expensive rather than merely inelegant. {{ch:ds-cleaning}}
argued that preprocessing decisions are modelling decisions; this chapter is the
strongest instance of that claim in the book, because the decision is frozen for
the model's whole life. {{ch:mle-pipelines}} gave train/serve skew, which
reappears here as normalisation mismatch.

**Forwards.** {{ch:nlp-subword}} is the resolution of {{eq:tokenizer-tradeoff}}
and gives the three algorithms actually used. {{ch:llm-anatomy}} picks the
pipeline up at stage 4 and follows the IDs to logits. {{ch:rag-chunking}} makes
the same cut-the-text decision at document scale, with the same absence of a
correct answer. {{part:15}} returns to vocabulary size as a memory-footprint
term when models are quantised for local execution.

## 17. Exercises

**Beginner**

1. Compute by hand the fertility of a character tokenizer on `"machine
   learning"` and explain why the number exceeds the character count divided by
   the word count only if you count the space.
2. Give a string for which NFC and NFKC differ, and one for which they agree.
3. Explain in two sentences why a 500,000-word vocabulary still has an OOV
   problem.

**Intermediate**

4. Using {{eq:zipf-coverage}}, find the vocabulary size needed for 95% coverage
   with $R = 10^6$. Comment on whether the answer is practical.
5. A tokenizer has fertility 1.3 on English and 3.9 on Finnish. By what factor
   does the attention term of the cost differ? By what factor does the
   parameter term differ? Explain why the two answers differ.
6. Find two inputs to a tokenizer of your choice that render identically and
   produce different ID sequences.

**Advanced**

7. Show that no complete tokenizer can have fertility below 1.0 on a corpus with
   more distinct word forms than $|V|$, and state the bound exactly.
8. Argue for or against: compatibility normalisation (NFKC) should never be used
   for a generative model. Support the position with a specific failure.

**Implementation**

9. Extend `baseline-tokenizers` with a fourth scheme that keeps the 1,000 most
   frequent words whole and falls back to bytes. Measure fertility and
   byte-fallback rate on the sample corpus, and plot fertility against
   vocabulary size for 100, 1,000 and 10,000 kept words.
10. Write a round-trip test that generates random strings including combining
    characters, emoji with modifiers, and mixed scripts, and asserts
    `decode(encode(s)) == s` for a byte tokenizer. Then break it deliberately by
    inserting NFKC normalisation, and record which inputs fail.
11. Build the truncation monitor from
    {{sec:10-production-considerations}}: given a stream of documents and a
    context limit, report the truncation rate and the distribution of discarded
    tail lengths.

**Reasoning**

12. A colleague proposes lowercasing all input to halve the effective vocabulary.
    Give the strongest argument for the proposal, then the argument that defeats
    it, and say what measurement would settle it.
13. Explain why fertility is a fair comparison between two complete tokenizers
    and an unfair one between a complete and an incomplete tokenizer.

## 18. Interview Questions

**Beginner**

1. What is a token, and why do models not use words?
2. What is the out-of-vocabulary problem and what does `UNK` cost you?
3. Why is `len(text)` a poor proxy for API cost?

**Intermediate**

4. Walk through what happens to a string between the API boundary and the first
   matrix multiplication.
5. Why is Unicode normalisation a correctness issue? Give a concrete failure.
6. A model does arithmetic badly. How would you determine whether the tokenizer
   is implicated?

**Senior**

7. You are launching in eight new languages. What tokenization work happens
   before launch, and what do you measure afterwards?
8. Your inference bill rose 30% with flat request volume. Tokenization is a
   candidate cause — how do you confirm or eliminate it?
9. Argue both sides of replacing a pretrained model's tokenizer with a
   language-specific one.

**Systems**

10. Design the tokenization layer for a multi-tenant API serving several model
    families. Address versioning, caching, and the metrics you expose.
11. How do you guarantee that training and serving tokenization can never
    diverge? Not "be careful" — a mechanism.

## 19. Research Questions

**Is there a segmentation objective better than compression?** BPE's criterion is
frequency and unigram's is likelihood ({{ch:nlp-subword}}); neither optimises for
anything the downstream model cares about. What would an objective look like that
targeted downstream loss, and why has nobody made one work at scale? Start from
{{cite:kudo2018sentencepiece}} and ask what it does *not* optimise.

**How much capability is lost to tokenization?** {{cite:rust2021}} isolated the
tokenizer and found a measurable effect on one axis. Design the equivalent
experiment for arithmetic: hold the model fixed, vary only digit segmentation,
and measure. Predict the result first.

**Can the fairness gap be closed without retraining?** {{cite:petrov2023}}
documents the disparity; vocabulary adaptation is the obvious lever. What
fraction of the gap does it close, and what does it cost in English performance?

**Why has byte-level modelling not won?** The completeness argument is
overwhelming and the sequence-length argument is the only thing standing against
it. With attention costs falling ({{ch:tf-efficient}}), does that argument still
hold at current context lengths? This is a runnable experiment, not a literature
question.

## 20. Chapter Summary

Tokenization is the map from strings to integers, and it is the last
hand-designed component in an otherwise learned pipeline. It is fitted once and
then frozen for the model's entire life, which makes every one of its properties
inherited rather than chosen by anyone downstream.

The problem has no correct answer because three quantities are in tension:
vocabulary size costs embedding parameters, sequence length costs quadratic
attention, and the out-of-vocabulary rate costs capability outright. Word-level
segmentation optimises the second and fails the third — and
{{eq:zipf-coverage}} shows it cannot be fixed by scaling, because coverage grows
only with the logarithm of vocabulary size. Character and byte segmentation fix
completeness and pay in sequence length, which {{eq:fertility-quadratic}} shows
is a quadratic penalty on the attention term.

**Fertility — tokens per word — is the number that makes the tradeoff
concrete**, and it converts directly into money, latency, and usable context.
{{cite:petrov2023}} measured ratios up to fifteen across languages for the same
content, and {{cite:rust2021}} showed the effect is not only financial: holding
the model fixed and varying only the tokenizer moves downstream accuracy.

Unicode normalisation is a correctness requirement, not a style choice. NFC and
NFD are different byte sequences for identical-looking text; NFKC additionally
folds `x²` into `x2` and cannot be undone. The pipeline's four stages — normalise,
segment, map, assemble — must be byte-identical at training and serving time, and
the failure when they are not is silent.

The classical preprocessing pipeline of lowercasing, stopword removal, and
stemming survives in exactly two places: sparse lexical retrieval, where the
information it destroys was unusable anyway, and interpretable count-based
features, where a worse model buys a better explanation. Everywhere else it is
subtraction.

## 21. Further Reading

{{cite:petrov2023}} is the paper to read first and it is short. Its contribution
is a measurement, not a method, and the measurement reframes tokenization from an
implementation detail into a distributional question with a price attached. Read
the tables rather than the discussion.

{{cite:rust2021}} is the controlled experiment: same model, same data, different
tokenizer. It is the paper to cite when someone claims the tokenizer does not
matter much, and its methodology — isolate one component, hold everything else
fixed — is worth more than its specific result.

{{cite:sennrich2016}} and {{cite:kudo2018sentencepiece}} are the subject of
{{ch:nlp-subword}} and are better read after it, but §2 of the SentencePiece
paper is the clearest short statement of why pre-tokenization by whitespace is a
bug rather than a simplification.

The Unicode Standard Annex #15 on normalisation forms is the primary source for
{{tbl:unicode-forms}}. It is a specification rather than a paper, and the useful
part is the table of compatibility mappings — reading it once cures the belief
that "normalise the text" is a single well-defined operation.

**Where to go next:** {{ch:nlp-subword}} resolves the tradeoff this chapter set
up, with the three algorithms every production tokenizer actually uses.
