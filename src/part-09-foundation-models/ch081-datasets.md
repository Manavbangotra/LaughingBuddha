---
id: fm-datasets
number: 81
part: IX
tier: full
status: draft
requires: [fm-pretraining, fm-what-they-are, nlp-preprocessing, ds-cleaning,
           ds-leakage, mle-splits, math-probability]
provides: [corpus-construction, quality-filtering, near-duplicate-detection,
           minhash, shingling, benchmark-contamination, data-mixture, mixture-weights,
           memorisation, provenance-of-data, perplexity-filtering]
citations: [gao2020pile, lee2022dedup, gunasekar2023, touvron2023llama,
            brown2020, radford2019, hoffmann2022chinchilla, bommasani2021]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Describe the five stages of a pretraining corpus pipeline and say what each
   one removes.
2. Implement MinHash with locality-sensitive hashing and explain why exact
   deduplication is insufficient.
3. Derive the LSH collision probability and choose bands and rows to hit a
   target similarity threshold.
4. Explain what {{cite:lee2022dedup}} established about duplication,
   memorisation, and train/test overlap.
5. Detect benchmark contamination with $n$-gram overlap and state the limits of
   that method.
6. Explain what a data mixture is and why its weights are a modelling decision
   nobody publishes.
7. Assess a claim about a model's capability given what you can and cannot know
   about its training data.

## 2. Why This Matters

**This is the input to the most expensive computation in the book**, and
{{ch:fm-what-they-are}} showed nothing downstream can repair it: adaptation
carries roughly $10^{-8}$ of pretraining's information
{{eq:adaptation-information-ratio}}. A defect here is permanent.

**Deduplication is not housekeeping — it is a quality lever.**
{{cite:lee2022dedup}} showed standard corpora are full of near-duplicates, and
that removing them produces models that memorise less at equal or better
quality. Two benefits from one cheap pipeline stage is unusual; the usual case
is a tradeoff.

**Contamination undermines evaluation everywhere else in this book.** The same
paper found train/test overlap inflating reported results. Every benchmark
number in {{part:10}} through {{part:28}} inherits that caveat, and the only
defences are evaluating on data created after the training cutoff or on data you
made yourself.

**And this chapter has a hole in it that cannot be filled.** Nobody outside the
labs knows what is in the frontier models' training data. That is not a gap in
my research; it is the state of the field, and pretending otherwise would be
worse than saying it. {{cite:gao2020pile}} exists precisely because someone
decided that was intolerable.

## 3. Prerequisites

{{ch:fm-pretraining}} for the run that consumes this corpus, and for the sampler
whose position is part of the training state. {{ch:fm-what-they-are}} for why
pretraining data determines what adaptation cannot fix.
{{ch:nlp-preprocessing}} for normalisation and tokenization, which happen after
this pipeline. {{ch:ds-cleaning}} for the general discipline of deciding what to
throw away. {{ch:ds-leakage}} for leakage, which reappears here as
contamination at corpus scale. {{ch:mle-splits}} for honest evaluation.
{{ch:math-probability}} for the collision probabilities in
{{sec:6-mathematical-foundation}}.

## 4. Intuitive Explanation

You need a trillion tokens. The web has them, and almost all of it is
unsuitable: navigation boilerplate, spam, machine-translated filler, adult
content, and enormous quantities of near-identical text — the same news article
republished across four hundred sites, the same licence header in a million
repositories.

The pipeline is five stages, and each throws away more than you expect.

**Source.** Crawl, or take an existing crawl, plus curated collections — books,
code, papers, reference works.

**Filter.** Drop what is not text you want to model. Language identification,
length limits, symbol ratios, and a quality signal. The last one is the
interesting one: a common trick is to train a cheap classifier to distinguish
your crawl from a corpus you consider high quality, and keep what it thinks
looks like the good side.

**Deduplicate.** The most valuable stage, and the least intuitive. Exact
duplicates are easy and rare; *near*-duplicates are pervasive and are what
actually matter. Two pages differing in a timestamp and an advertisement are the
same document for training purposes.

**Decontaminate.** Remove anything that appears in the benchmarks you intend to
evaluate on. Otherwise you measure memorisation and call it capability.

**Mix.** Decide how much of each source to include, and how many times. This is
the stage with the most influence on the resulting model and the least published
evidence.

> NOTE: The stages are not independent and their order matters. Deduplicating
> before filtering wastes work on documents you will drop; filtering before
> language identification means your quality classifier sees languages it was
> never fitted on. Pipeline order is a design decision with a cost, and it is
> usually inherited rather than chosen.

**Why near-duplicates hurt.** A document appearing a hundred times is seen a
hundred times by the model, which is a hundred gradient steps toward
memorising it. It consumes budget that unique text would have used, and it
teaches the model to reproduce that text verbatim — which is a privacy problem,
a copyright problem, and a quality problem at once.

**The mental model:** corpus construction is a filtering problem where every
stage trades recall for precision, and the quantity being maximised is *unique,
in-distribution, uncontaminated tokens per dollar.* Where it breaks down:
"quality" has no definition independent of the model you are about to train, so
the quality filter encodes somebody's judgement about what good text looks like,
and that judgement propagates into everything the model does.

## 5. Formal Explanation

### 5.1 The pipeline as a sequence of filters

Let $\Data_0$ be the raw corpus. Each stage is a map $\Data_{i} = F_i(\Data_{i-1})$
with a **yield** $y_i = |\Data_i| / |\Data_{i-1}|$:

$$
\Data_{\text{final}} = F_{\text{mix}}\circ F_{\text{decon}}\circ F_{\text{dedup}}
 \circ F_{\text{filter}}\circ F_{\text{lang}}(\Data_0)
$$ (eq:corpus-pipeline)

with total yield $y = \prod_i y_i$. Reported end-to-end yields from raw crawl to
trainable tokens are on the order of a few per cent, so **the corpus you train
on is a small and heavily selected subset of what you started with** — and every
selection is a modelling decision.

### 5.2 Near-duplicate detection: shingles and Jaccard

Represent a document $d$ by its set of $k$-shingles — all contiguous $k$-token
subsequences:

$$
S_k(d) = \{\,d_{i:i+k}\ :\ 1 \le i \le |d| - k + 1\,\}
$$ (eq:shingles)

Similarity is Jaccard:

$$
J(d_1,d_2) = \frac{|S_k(d_1)\cap S_k(d_2)|}{|S_k(d_1)\cup S_k(d_2)|}
$$ (eq:jaccard)

Computing this for all pairs is $O(n^2)$ and impossible at $n = 10^9$
documents. MinHash plus LSH reduces it to approximately linear.

### 5.3 MinHash

For a random permutation $\pi$ of the shingle universe, define
$h_\pi(d) = \min_{s\in S_k(d)} \pi(s)$. The defining property:

$$
\Prob\big[h_\pi(d_1) = h_\pi(d_2)\big] = J(d_1,d_2)
$$ (eq:minhash-property)

**The probability that two documents' minimum hashes agree is exactly their
Jaccard similarity.** Proof in {{sec:6-mathematical-foundation}}.

Using $m$ independent hash functions gives a signature
$\vec{\sigma}(d)\in\mathbb{Z}^m$, and

$$
\hat{J} = \frac{1}{m}\sum_{j=1}^{m}\Ind\big[\sigma_j(d_1) = \sigma_j(d_2)\big]
$$ (eq:minhash-estimator)

is an unbiased estimator of $J$ with variance $J(1-J)/m$ — so accuracy is
controlled by signature length, independent of document length.

### 5.4 Locality-sensitive hashing

Split the $m$-element signature into $b$ bands of $r$ rows each, $m = br$. Two
documents are **candidates** if they agree on all rows of at least one band. For
similarity $s$:

$$
\Prob[\text{candidate}] = 1 - \big(1 - s^{\,r}\big)^{b}
$$ (eq:lsh-probability)

This is an S-curve in $s$ with a threshold near

$$
s^* \approx \left(\frac{1}{b}\right)^{1/r}
$$ (eq:lsh-threshold)

**$b$ and $r$ are the tuning knobs**, and they trade false positives against
false negatives. Larger $r$ sharpens the curve; larger $b$ moves the threshold
down. {{sec:8-implementation}} measures the curve rather than trusting the
approximation.

### 5.5 Contamination

Let $\mathcal{B}$ be a benchmark's test set. A training document $d$ is
contaminated if it shares a sufficiently long $n$-gram with any test item:

$$
\text{contaminated}(d) \iff \exists\, t \in \mathcal{B}:\
 G_n(d)\cap G_n(t) \neq \emptyset
$$ (eq:contamination)

for $n$-gram sets $G_n$. Typical $n$ is 8 to 13 tokens.

> WARNING: This detects *verbatim* overlap only. A paraphrase of a test item, a
> translation of it, or a discussion of its answer is not detected and is
> arguably worse — the model learns the answer without the string. **Reported
> contamination rates are lower bounds**, and any statement of the form "we
> decontaminated against benchmark X" means only that exact $n$-gram matches
> were removed.

### 5.6 The data mixture

A corpus is a weighted combination of sources $\Data^{(1)},\dots,\Data^{(K)}$
with sampling weights $w_k$ and epoch counts $e_k$:

$$
\Data = \bigcup_{k=1}^{K} \big(\Data^{(k)}\big)^{\times e_k},
\qquad
w_k = \frac{e_k\,|\Data^{(k)}|}{\sum_j e_j\,|\Data^{(j)}|}
$$ (eq:data-mixture)

The weights are a modelling decision of the first order — how much code, how
much multilingual text, how much of any domain — and **they are essentially
unpublished for frontier models.** {{cite:touvron2023llama}} is unusually
specific and is the exception rather than the norm.

Note $e_k > 1$ means deliberately repeating a source, which sits in direct
tension with {{cite:lee2022dedup}}'s deduplication result. Both practices are
common. The reconciliation is that repeating a small high-quality source is not
the same as failing to remove accidental web duplication, but the boundary is
not well characterised.

## 6. Mathematical Foundation

### 6.1 Why MinHash works

Let $U = S_k(d_1)\cup S_k(d_2)$ and $I = S_k(d_1)\cap S_k(d_2)$. Under a
uniformly random permutation $\pi$, consider the element of $U$ receiving the
smallest value — every element of $U$ is equally likely to be that element.

$h_\pi(d_1) = h_\pi(d_2)$ if and only if that minimising element lies in $I$:
if it is in $I$ it is present in both sets and is the minimum of both; if it is
in $U\setminus I$ it belongs to only one, which then has a smaller minimum than
the other.

$$
\Prob\big[h_\pi(d_1) = h_\pi(d_2)\big] = \frac{|I|}{|U|} = J(d_1,d_2)
$$

$\square$

**The estimator's cost is independent of document length.** Signatures are $m$
integers whatever the documents contain, which is what makes the method usable
on a corpus of billions.

### 6.2 The LSH S-curve

Within one band of $r$ rows, two documents agree on all rows with probability
$s^r$ under {{eq:minhash-property}} and independence across hashes. They fail to
match in a band with probability $1 - s^r$, and fail in all $b$ bands with
probability $(1-s^r)^b$, giving {{eq:lsh-probability}}.

$\square$

Worked, with $m = 128$, $b = 32$, $r = 4$:

$$
s = 0.5:\ 1 - (1-0.5^4)^{32} = 1 - 0.9375^{32} = 0.87
$$

$$
s = 0.3:\ 1 - (1-0.3^4)^{32} = 1 - 0.9919^{32} = 0.23
$$

$$
s = 0.8:\ 1 - (1-0.8^4)^{32} = 1 - 0.5904^{32} \approx 1.000
$$

The threshold estimate {{eq:lsh-threshold}} gives
$(1/32)^{1/4} = 0.42$, which matches where the curve turns. **Documents 80%
similar are caught essentially always and documents 30% similar are mostly
passed**, which is the behaviour wanted: catch republished articles, keep two
independent documents on the same topic.

### 6.3 Why duplication costs more than its share of the budget

Suppose a fraction $\rho$ of a corpus is duplicated $c$ times on average. Unique
content is

$$
\text{unique fraction} = (1-\rho) + \frac{\rho}{c}
$$ (eq:unique-fraction)

With $\rho = 0.3$ and $c = 10$: $0.7 + 0.03 = 0.73$ — so 27% of the training
budget buys nothing new.

But the loss is worse than the budget. The duplicated documents receive $c$
times the gradient signal of unique ones, so the model's effective objective is
reweighted toward them:

$$
\Loss_{\text{effective}} = \sum_d \frac{c_d}{\sum_{d'} c_{d'}} \Loss(d)
$$ (eq:duplication-reweighting)

$\square$

**Duplication silently changes the objective**, upweighting whatever happens to
be republished most — boilerplate, licences, popular articles. That is the
mechanism behind {{cite:lee2022dedup}}'s memorisation finding, and it is why
deduplication improves quality rather than merely saving compute.

## 7. Internal Mechanics

```mermaid {#fig:corpus-pipeline caption="The corpus pipeline. Yields compound, so a few per cent of the raw crawl survives to training. The deduplication and decontamination stages are the two that change model behaviour rather than merely reducing volume."}
graph LR
  A["raw crawl<br/>~10^15 bytes"] --> B["extract text<br/>strip boilerplate"]
  B --> C["language ID<br/>+ length, symbol ratios"]
  C --> D["quality filter<br/>classifier or perplexity"]
  D --> E["EXACT dedup<br/>hash the document"]
  E --> F["NEAR dedup<br/>MinHash + LSH"]
  F --> G["decontaminate<br/>n-gram vs benchmarks"]
  G --> H["mix + weight<br/>per-source epochs"]
  H --> I["tokenize, shard<br/>ch:nlp-subword"]
  style F fill:#dfe,stroke:#5a5
  style G fill:#fde,stroke:#c69
```

**Where the cost is.** Near-duplicate detection over $10^9$ documents is the
expensive stage and the one that must be distributed. MinHash signatures are
computed independently per document — embarrassingly parallel — and the LSH
banding is a shuffle-and-group, which is why the whole thing maps naturally onto
MapReduce-style infrastructure.

**Quality filtering by perplexity.** A common approach: train a small language
model on a reference corpus you trust, score every candidate document, and keep
those in a middle perplexity band. Very high perplexity is gibberish; very low
perplexity is boilerplate and repetition. **Both tails are discarded**, which
surprises people who expect "keep the most predictable text".

**Decontamination is per-benchmark and therefore incomplete.** You can only
decontaminate against benchmarks that exist when you build the corpus. A
benchmark published after your training cutoff is clean by construction, which
is precisely why {{part:25}} recommends evaluating on post-cutoff data.

**What the labs publish.** {{cite:gao2020pile}} lists 22 sources with sizes and
weights. {{cite:touvron2023llama}}'s §2 gives proportions. {{cite:brown2020}}
gives a table of five sources with weights and epoch counts. Beyond that, for
current frontier models, the answer is that the composition is not public — and
{{cite:bommasani2021}}'s homogenisation argument makes that a systemic issue
rather than a commercial curiosity.

## 8. Implementation

MinHash and LSH from scratch, measured against exact Jaccard.

```python {tier=A name=minhash-lsh}
"""Near-duplicate detection: MinHash signatures, LSH banding, measured."""
import hashlib
from collections import defaultdict

import numpy as np

rng = np.random.default_rng(0)
SHINGLE_K, NUM_HASHES = 5, 128
BANDS, ROWS = 32, 4                    # BANDS * ROWS must equal NUM_HASHES
assert BANDS * ROWS == NUM_HASHES

MERSENNE = (1 << 61) - 1
coeffs = rng.integers(1, MERSENNE, size=(NUM_HASHES, 2))


def shingles(text, k=SHINGLE_K):
    """Equation (eq:shingles), over whitespace tokens."""
    toks = text.split()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def base_hash(s):
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


def signature(text):
    """m independent min-hashes — equation (eq:minhash-property)."""
    sh = np.array([base_hash(s) for s in shingles(text)], dtype=np.uint64)
    if len(sh) == 0:
        return np.zeros(NUM_HASHES, dtype=np.uint64)
    a, b = coeffs[:, 0][:, None], coeffs[:, 1][:, None]
    hashed = (a * sh[None, :].astype(object) + b) % MERSENNE
    return np.array([min(row) for row in hashed], dtype=object)


def exact_jaccard(t1, t2):
    s1, s2 = shingles(t1), shingles(t2)
    return len(s1 & s2) / len(s1 | s2)


def estimated_jaccard(sig1, sig2):
    """Equation (eq:minhash-estimator)."""
    return float(np.mean([x == y for x, y in zip(sig1, sig2)]))


BASE = ("the quarterly report shows revenue growth across all regions with "
        "particular strength in the enterprise segment and continued expansion "
        "of the subscription business during the period under review")


def perturb(text, fraction):
    """Replace a fraction of tokens — a stand-in for editorial variation."""
    toks = text.split()
    n = int(len(toks) * fraction)
    idx = rng.choice(len(toks), size=n, replace=False)
    for i in idx:
        toks[i] = f"tok{rng.integers(0, 999)}"
    return " ".join(toks)


print("MinHash estimate against exact Jaccard\n")
print(f"{'perturbation':>13} {'exact J':>9} {'estimated':>11} {'error':>8}")
sig_base = signature(BASE)
for frac in (0.0, 0.05, 0.15, 0.30, 0.50, 0.80):
    variant = perturb(BASE, frac)
    ex = exact_jaccard(BASE, variant)
    es = estimated_jaccard(sig_base, signature(variant))
    print(f"{frac:>12.0%} {ex:>9.3f} {es:>11.3f} {abs(ex - es):>8.3f}")

print(f"\nestimator standard error at J=0.5, m={NUM_HASHES}: "
      f"{(0.5 * 0.5 / NUM_HASHES) ** 0.5:.4f}")


def lsh_buckets(signatures):
    """Band the signatures; documents sharing any band are candidates."""
    buckets = defaultdict(list)
    for doc_id, sig in signatures.items():
        for band in range(BANDS):
            key = (band, tuple(sig[band * ROWS:(band + 1) * ROWS]))
            buckets[key].append(doc_id)
    pairs = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(tuple(sorted((members[i], members[j]))))
    return pairs


# Build a corpus with a known duplicate structure.
docs, truth = {}, {}
docs["orig"] = BASE
for name, frac in [("edit-0.5%", 0.005), ("edit-1%", 0.01), ("edit-2%", 0.02),
                   ("edit-5%", 0.05), ("edit-10%", 0.10), ("edit-20%", 0.20),
                   ("edit-40%", 0.40)]:
    docs[name] = perturb(BASE, frac)
    truth[name] = exact_jaccard(BASE, docs[name])
docs["unrelated"] = " ".join(f"q{i % 200}" for i in range(400))
truth["unrelated"] = exact_jaccard(BASE, docs["unrelated"])

sigs = {k: signature(v) for k, v in docs.items()}
candidates = lsh_buckets(sigs)

print(f"\nLSH with b={BANDS}, r={ROWS} "
      f"(threshold ~ (1/b)^(1/r) = {(1 / BANDS) ** (1 / ROWS):.2f})\n")
print(f"{'document':>11} {'exact J':>9} {'flagged':>9} {'predicted P':>13}")
for name in ("edit-0.5%", "edit-1%", "edit-2%", "edit-5%", "edit-10%",
             "edit-20%", "edit-40%", "unrelated"):
    s = truth[name]
    flagged = ("orig", name) in candidates or (name, "orig") in candidates
    predicted = 1 - (1 - s ** ROWS) ** BANDS      # equation (eq:lsh-probability)
    print(f"{name:>11} {s:>9.3f} {str(flagged):>9} {predicted:>13.3f}")

print("\nThe S-curve does the work: high-similarity documents are caught almost "
      "always, low-similarity ones almost never, and the transition is sharp "
      "enough to separate a republished article from an independent one on the "
      "same topic.")
```

Now the duplication cost from {{eq:duplication-reweighting}}, which is the
argument that deduplication is a quality lever rather than a saving:

```python {tier=A name=duplication-cost}
"""What duplication does to the training budget and to the objective."""
import numpy as np

rng = np.random.default_rng(1)

N_UNIQUE = 10_000
DUP_FRACTION = 0.30          # share of documents that are duplicated
DUP_COPIES = 10              # how many times each appears

# Build a corpus: most documents appear once, a subset appears many times.
n_dup_docs = int(N_UNIQUE * DUP_FRACTION)
counts = np.ones(N_UNIQUE, dtype=int)
counts[:n_dup_docs] = DUP_COPIES

total_docs = counts.sum()
unique_fraction = N_UNIQUE / total_docs
print(f"{N_UNIQUE:,} unique documents, {n_dup_docs:,} of them duplicated "
      f"{DUP_COPIES}x")
print(f"corpus as stored : {total_docs:,} documents")
print(f"distinct docs / stored docs : {unique_fraction:.3f}")
print(f"reads spent re-reading      : {1 - unique_fraction:.1%} of the budget\n")

# Equation (eq:duplication-reweighting): the effective objective weights.
weights = counts / counts.sum()
dup_share = weights[:n_dup_docs].sum()
print(f"duplicated documents are {n_dup_docs / N_UNIQUE:.0%} of unique content")
print(f"but receive {dup_share:.0%} of the gradient signal")
print(f"over-representation factor: {dup_share / (n_dup_docs / N_UNIQUE):.1f}x\n")

# Memorisation proxy: exposures determine how strongly a document is imprinted.
print(f"{'document class':<22} {'exposures':>10} {'relative imprint':>18}")
print(f"{'unique':<22} {1:>10} {1.0:>18.1f}x")
print(f"{'duplicated':<22} {DUP_COPIES:>10} {float(DUP_COPIES):>18.1f}x")

# What deduplication actually frees. It does not create data.
freed = total_docs - N_UNIQUE
print(f"At a fixed budget of {total_docs:,} document-reads:")
print(f"  with duplicates : {N_UNIQUE:,} distinct documents seen, "
      f"{freed:,} reads re-reading")
print(f"  after dedup     : the same {N_UNIQUE:,} seen in {N_UNIQUE:,} reads, "
      f"{freed:,} freed")
print(f"  those freed reads buy new content only if the corpus HAS more unique "
      f"documents.\n  Deduplication frees budget; it does not create data.")

print("""
Two effects, and the second is the one that matters. Deduplication frees budget
(the first effect, and it is merely a saving). It also removes an unintended
reweighting of the objective — equation (eq:duplication-reweighting) — that was
pushing the model toward whatever the web happens to republish most:
boilerplate, licences, syndicated articles. That is why removing duplicates
improves quality rather than just costing less.""")
```

And contamination detection, with its limits made visible:

```python {tier=A name=contamination-check}
"""n-gram contamination detection, and what it cannot see."""

N_GRAM = 8

BENCHMARK = [
    "what is the capital city of the republic of france in europe",
    "compute the derivative of the function f of x equals x squared plus three",
    "who wrote the novel one hundred years of solitude in nineteen sixty seven",
]

TRAINING_DOCS = {
    "clean-1": "the weather in paris is generally mild during the spring months",
    "clean-2": "derivatives measure how quickly a function changes at a point",
    "verbatim": ("a quiz question asks what is the capital city of the republic "
                 "of france in europe and the answer is paris"),
    "paraphrase": ("france is a european republic whose capital city is paris, "
                   "a fact commonly tested in geography examinations"),
    "answer-only": ("the answer to the famous solitude authorship question is "
                    "gabriel garcia marquez who published it in 1967"),
}


def ngrams(text, n=N_GRAM):
    toks = text.split()
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


bench_ngrams = set()
for item in BENCHMARK:
    bench_ngrams |= ngrams(item)

print(f"benchmark: {len(BENCHMARK)} items, {len(bench_ngrams)} distinct "
      f"{N_GRAM}-grams\n")
print(f"{'document':<14} {'overlap':>9} {'flagged':>9}  note")
for name, doc in TRAINING_DOCS.items():
    overlap = len(ngrams(doc) & bench_ngrams)
    flagged = overlap > 0
    note = ""
    if name == "paraphrase" and not flagged:
        note = "<- CONTAMINATED but invisible to n-grams"
    if name == "answer-only" and not flagged:
        note = "<- teaches the answer, no string overlap"
    print(f"{name:<14} {overlap:>9} {str(flagged):>9}  {note}")

detected = sum(1 for n, d in TRAINING_DOCS.items() if ngrams(d) & bench_ngrams)
truly_contaminated = 3          # verbatim, paraphrase, answer-only
print(f"\ndetected {detected} of {truly_contaminated} genuinely contaminated "
      f"documents -> recall {detected / truly_contaminated:.0%}")

# Sensitivity to n: shorter n-grams catch more and produce false positives.
print(f"\n{'n':>4} {'flagged docs':>13} {'note':<40}")
for n in (5, 8, 13, 20):
    bench_n = set()
    for item in BENCHMARK:
        bench_n |= ngrams(item, n)
    hits = sum(1 for d in TRAINING_DOCS.values() if ngrams(d, n) & bench_n)
    note = ("catches more, risks false positives" if n <= 5
            else "misses paraphrase and restatement")
    print(f"{n:>4} {hits:>13} {note:<40}")

print("""
Equation (eq:contamination) detects verbatim overlap and nothing else. The
paraphrase and the bare answer are contamination by any reasonable definition
and neither leaves an n-gram trace, so every published contamination rate is a
LOWER BOUND. "We decontaminated against benchmark X" means exact n-gram matches
were removed — which is worth doing and is not the same as a clean evaluation.""")
```

## 9. Practical Example

A team is assembling a domain corpus for continued pretraining on ten years of
internal engineering documents — design docs, incident reports, code review
threads, wiki pages. The instinct is that internal data is clean because it is
not the web. It is not clean; it is differently dirty, and the specific
pathologies are predictable.

```python {tier=A name=domain-corpus-audit}
"""Auditing a domain corpus before committing it to a training run."""
import numpy as np

rng = np.random.default_rng(4)

# A synthetic stand-in with the pathologies real internal corpora have.
TEMPLATE_HEADER = "confidential internal document do not distribute externally"
INCIDENT_BOILER = "incident severity impact detection mitigation follow up actions"

docs = []
# 1. Genuine unique content.
for i in range(600):
    docs.append(("unique", f"design note {i} " + " ".join(
        f"w{rng.integers(0, 400)}" for _ in range(60))))
# 2. Template-heavy documents: mostly identical boilerplate.
for i in range(300):
    docs.append(("templated", f"{TEMPLATE_HEADER} {INCIDENT_BOILER} "
                 + " ".join(f"w{rng.integers(0, 40)}" for _ in range(12))))
# 3. Machine-generated logs: high volume, near-zero information.
for i in range(400):
    docs.append(("generated", "timestamp service latency status code "
                 + " ".join(str(rng.integers(0, 9)) for _ in range(40))))
# 4. Verbatim copies (a doc pasted into several threads).
for i in range(120):
    docs.append(("duplicate", docs[i % 50][1]))

labels = [d[0] for d in docs]
texts = [d[1] for d in docs]
print(f"corpus: {len(docs):,} documents\n")


def token_entropy(text):
    """Low entropy signals templated or generated text."""
    toks = text.split()
    _, counts = np.unique(toks, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def type_token_ratio(text):
    toks = text.split()
    return len(set(toks)) / max(len(toks), 1)


print(f"{'class':<12} {'n':>5} {'mean entropy':>14} {'mean TTR':>10} "
      f"{'verdict':<22}")
for cls in ("unique", "templated", "generated", "duplicate"):
    sel = [t for t, l in zip(texts, labels) if l == cls]
    ent = np.mean([token_entropy(t) for t in sel])
    ttr = np.mean([type_token_ratio(t) for t in sel])
    verdict = "keep" if ent > 3.5 and ttr > 0.7 else "filter or downweight"
    print(f"{cls:<12} {len(sel):>5} {ent:>14.3f} {ttr:>10.3f} {verdict:<22}")

# Exact duplicate detection is trivial and worth doing first.
seen, exact_dups = set(), 0
for t in texts:
    if t in seen:
        exact_dups += 1
    seen.add(t)
print(f"\nexact duplicates: {exact_dups} ({exact_dups / len(texts):.1%})")

# What survives, and what the corpus is actually worth.
keep = [t for t, l in zip(texts, labels) if l in ("unique",)]
unique_keep = set(keep)
raw_tokens = sum(len(t.split()) for t in texts)
kept_tokens = sum(len(t.split()) for t in unique_keep)
print(f"raw tokens        : {raw_tokens:,}")
print(f"tokens after audit: {kept_tokens:,}  "
      f"(yield {kept_tokens / raw_tokens:.1%})")

print(f"""
The yield is the number to take to the planning meeting. A corpus advertised as
{raw_tokens:,} tokens is worth {kept_tokens:,} for training purposes, and the
difference is templates, machine-generated logs, and copies — none of which a
document count reveals. Equation (eq:adaptation-information-ratio) says
continued pretraining needs on the order of 10^10 tokens to move what a model
knows; measure the real yield before assuming you have them.""")
```

**The general point:** every corpus is advertised in raw size and is worth its
unique, in-distribution token count. Those differ by a large factor, and the
factor is measurable in an afternoon.

> PRODUCTION TIP: Run the audit before the planning meeting, not after the
> training run. The most common outcome is discovering the corpus is an order of
> magnitude smaller than advertised, which changes the decision from "continued
> pretraining" to "retrieval" — and that is a much cheaper thing to learn early.

## 10. Production Considerations

**Version the corpus, not just the model.** Record the source snapshot, the
pipeline version, the filter thresholds, and the mixture weights. A model is not
reproducible without them, and {{ch:mle-reproducibility}}'s discipline applies at
a scale where re-running the pipeline is itself expensive.

**Deduplication must be distributed and is the pipeline's cost centre.** MinHash
signatures parallelise trivially; the LSH grouping is a shuffle. Budget for it
explicitly rather than discovering it.

**Decontaminate against every benchmark you will report**, and record which.
Then say so in the writeup, because a contamination claim without a list of
benchmarks is not a claim.

**Keep a held-out slice from the same pipeline.** {{ch:fm-pretraining}} needs it
to detect pipeline faults, and it must pass through identical processing or the
comparison measures the pipeline rather than the model.

**Licensing and provenance are engineering constraints, not legal footnotes.**
Which sources are permitted, whether removal requests must be honoured, and
whether a source can be redistributed all constrain the pipeline's design.
{{part:27}} treats this properly; the point here is that it is decided at corpus
construction time and is expensive to revisit.

**What to log:** per-stage yields, the duplicate-cluster size distribution,
per-source token counts before and after each stage, and contamination hits per
benchmark. Yields are the health metric — a stage whose yield changes between
runs indicates the input distribution moved.

## 11. Common Mistakes

**Beginners:**

*Deduplicating exactly and calling it done.* Exact duplicates are the easy,
rare case. Near-duplicates are pervasive and are what
{{cite:lee2022dedup}} measured.

*Assuming more data is better.* Duplicated and low-quality data actively harms
via {{eq:duplication-reweighting}}. The quantity to maximise is unique
in-distribution tokens, not bytes.

*Trusting a corpus's advertised size.* `domain-corpus-audit` shows the gap
between raw and usable, and it is usually large.

**Experienced practitioners:**

*Treating contamination detection as sound.* It is a lower bound
({{sec:5-formal-explanation}}). Paraphrases and discussions of answers are
invisible to $n$-grams and are real contamination.

*Filtering only the high-perplexity tail.* Both tails should go — very low
perplexity is boilerplate. Keeping the most predictable text is the opposite of
what you want.

*Getting the pipeline order wrong.* Deduplicating before filtering wastes work;
quality-filtering before language ID applies a classifier outside its fitted
distribution.

*Ignoring that the quality filter encodes a judgement.* A classifier trained to
recognise "high quality" as resembling a reference corpus imports that corpus's
demographics, registers, and topics into the model. That is not a neutral
operation, and {{part:27}} is where it comes back.

## 12. Failure Modes

**Contamination discovered after publication.** Benchmark results are inflated
and the model appears more capable than it is. *Detection:* re-run
{{eq:contamination}} against the released test sets; better, evaluate on
post-cutoff data. *This is the most consequential failure in this chapter
because it corrupts evidence rather than models.*

**Memorisation of duplicated content.** The model reproduces training text
verbatim — a privacy exposure and a copyright exposure at once. *Detection:*
prompt with prefixes of known training documents and measure verbatim
continuation. *Cause:* {{eq:duplication-reweighting}}.

**Quality filter removing a demographic or register.** A classifier fitted to
resemble a reference corpus systematically drops dialects, non-standard
orthography, and low-resource languages. *Symptom:* poor model performance on
populations that were filtered out of the training data. *Detection:* measure
per-group yield through the filter, not only aggregate yield.

**Mixture weights that nobody remembers choosing.** Weights inherited from a
previous project, applied to a different corpus. *Symptom:* a model unexpectedly
strong or weak in a domain. *Detection:* the per-source token log.

**Pipeline drift between runs.** A source's format changes and the extractor
silently produces degraded text. *Symptom:* a per-stage yield moving between
runs. *This is why yields are logged.*

**Over-filtering.** Aggressive thresholds produce a small, homogeneous, and very
clean corpus, and a model that is narrow. The failure is invisible in the
pipeline metrics — every yield looks fine, because each stage is doing exactly
what it was told.

## 13. Alternatives

{#tbl:dedup-methods caption="Ways to find duplicate content, by what they catch and what they cost. Only the first is cheap enough to skip thinking about; the rest trade recall against compute, and the last catches what none of the others can."}

| Method | Catches | Cost | Misses |
|---|---|---|---|
| Exact hash | byte-identical | $O(n)$ | anything edited at all |
| Suffix array | long exact substrings | $O(n\log n)$, high memory | reordered or reworded text |
| MinHash + LSH | high Jaccard | $O(n)$ approx | paraphrase, translation |
| SimHash | high cosine on features | $O(n)$ | same as MinHash |
| Embedding + ANN | semantic near-duplicates | encoder pass per doc | nothing much — but costs a forward pass each |

**What genuinely differs.** The first four are all lexical: they compare surface
strings under different similarity measures, and none detects a document that
says the same thing in different words. Embedding-based detection
({{part:11}}) does, at the cost of an encoder forward pass over the whole
corpus — which at $10^9$ documents is comparable to a training run, and is why
it is used for curated subsets rather than raw crawls.

**On quality filtering**, the alternatives are a classifier against a reference
corpus, perplexity banding from a small model, and hand-written heuristics
(symbol ratios, line lengths, stopword presence). All three are in production
use. The heuristics are the most auditable and the least effective; the
classifier is the most effective and the least auditable, which is exactly the
tradeoff {{part:27}} cares about.

## 14. Evaluation

**Is the pipeline correct?**

1. **Per-stage yields** within expected bands, and stable between runs.
2. **Deduplication recall and precision** against a labelled set of known
   duplicates — the `minhash-lsh` listing is that measurement in miniature.
3. **Round-trip integrity**: text extracted from a source can be traced back to
   it, which is what makes removal requests actionable.
4. **Contamination hits per benchmark**, reported as a count, not a boolean.

**Is the corpus any good?** This can only be answered by training, which is the
uncomfortable part of the chapter: a corpus's quality is defined by the model it
produces, and the feedback loop costs a training run. The practical response is
proxies — train small models on candidate mixtures and compare, which is what
{{cite:gunasekar2023}} effectively did and is the strongest available method.

**What can you actually know about someone else's corpus?** For an open corpus,
its documented composition. For a frontier model, essentially nothing — and any
claim about *why* such a model behaves a certain way that depends on its
training data is speculation. Say so when you write it.

## 15. Advanced Concepts

**Synthetic and curated data.** {{maturity:EMERGING}}
{{cite:gunasekar2023}} generates and curates for pedagogical quality and beats
scaling-law extrapolations at small size. The contamination concerns it attracted
are part of the result and belong next to it.

**Data-constrained scaling.** {{maturity:EMERGING}} What to do when unique
tokens run out before compute does — repeating data a few epochs degrades
gracefully up to a point, and beyond it the returns go negative. This is the
direct tension with {{cite:lee2022dedup}} noted in {{sec:5-formal-explanation}}.

**Learned data selection.** {{maturity:RESEARCH FRONTIER}} Selecting training
documents by their measured effect on downstream loss rather than by a quality
proxy. The obstacle is that measuring the effect requires training.

**Machine unlearning.** {{maturity:RESEARCH FRONTIER}} Removing a document's
influence after training, without retraining. Motivated directly by removal
requests. Current methods do not offer the guarantee the use case needs.

**Model collapse from synthetic data.** {{maturity:EMERGING}} As web text
increasingly contains model output, training on the web means training partly on
predecessors. Whether this degrades models over generations is actively studied
and not settled.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-pretraining}} is the consumer of this pipeline, and its
sampler draws from the output of {{fig:corpus-pipeline}}.
{{ch:fm-what-they-are}}'s {{eq:adaptation-information-ratio}} is why defects here
are permanent. {{ch:ds-leakage}} introduced leakage on a dataset; contamination
is the same failure at corpus scale, with the difference that the test set is
public and the training set is not. {{ch:mle-splits}} required honest
evaluation, which {{eq:contamination}} is the corpus-scale defence of.
{{ch:nlp-subword}} is the stage after this one, and
{{ch:nlp-preprocessing}}'s normalisation must be identical here and at serving.

**Forwards.** {{ch:fm-scaling-laws}} assumes a token budget that this chapter
determines the real size of. {{ch:emb-ann}} generalises the LSH of
{{eq:lsh-probability}} into approximate nearest-neighbour search — the same
technique doing a different job. {{part:25}} inherits the contamination caveat
for every benchmark it discusses, and {{part:27}} takes up the filter's encoded
judgements as a fairness question.

## 17. Exercises

**Beginner**

1. Compute the Jaccard similarity of the 3-shingle sets of "the cat sat on the
   mat" and "the cat sat on a mat".
2. Why is exact deduplication insufficient? Give a concrete pair.
3. A corpus has 30% of documents duplicated 5 times. What fraction of the
   training budget buys new content?

**Intermediate**

4. Using {{eq:lsh-probability}}, compute the candidate probability at $s=0.6$
   for $b=20, r=6$ and for $b=50, r=3$. Which is more permissive, and why?
5. Choose $b$ and $r$ for $m=100$ giving a threshold near 0.7, and verify with
   {{eq:lsh-threshold}}.
6. Explain why both the high and low perplexity tails are filtered.

**Advanced**

7. Prove {{eq:minhash-property}} and identify where the argument needs $\pi$ to
   be a uniformly random permutation.
8. Derive {{eq:duplication-reweighting}} and explain why the effect is not
   captured by counting the wasted budget alone.
9. Design a contamination detector that catches paraphrase. State its false
   positive risk and why that risk may be unacceptable.

**Implementation**

10. Extend `minhash-lsh` to measure precision and recall of the LSH candidate
    set against exact Jaccard at a threshold of 0.7, over a corpus of 1,000
    synthetic documents. Sweep $b$ and $r$ and plot the operating curve.
11. Implement suffix-array-based exact long-substring deduplication and compare
    what it catches against MinHash on the same corpus.
12. Implement perplexity filtering: train a small model on a reference corpus,
    score candidates, and show that both tails contain text you would not want.
13. Build the per-stage yield dashboard from
    {{sec:10-production-considerations}} over a synthetic pipeline, and inject a
    format change in one source to show which yield moves.

**Reasoning**

14. {{cite:lee2022dedup}} says remove duplicates; {{eq:data-mixture}}'s $e_k>1$
    says deliberately repeat good sources. Reconcile these, and state what
    experiment would locate the boundary.
15. A model scores highly on a benchmark published before its training cutoff.
    Enumerate the explanations and say what evidence would distinguish them.

## 18. Interview Questions

**Beginner**

1. What are the stages of a pretraining data pipeline?
2. What is a near-duplicate and why does it matter?
3. What is benchmark contamination?

**Intermediate**

4. Explain MinHash. Why is the estimator's cost independent of document length?
5. How do you tune LSH bands and rows, and what do they trade?
6. Why does deduplication improve quality rather than only save compute?

**Senior**

7. You have a domain corpus and want to continue pretraining. Walk through the
   audit before you commit.
8. How would you decide the mixture weights for a code-and-prose corpus?
9. A published model scores well on a benchmark. What would make you doubt it,
   and what would resolve the doubt?

**Systems**

10. Design deduplication over $10^9$ documents. Address parallelism, memory, and
    the shuffle.
11. How do you make a training corpus reproducible and auditable, including
    honouring a removal request two years later?

## 19. Research Questions

**Where is the boundary between harmful duplication and beneficial repetition?**
{{cite:lee2022dedup}} says remove duplicates; standard practice repeats curated
sources. Both cannot be unconditionally right. Measure downstream quality as a
function of epoch count separately for high- and low-quality sources, and locate
the crossover.

**How much contamination is undetected?** Every published rate uses $n$-gram
overlap and is a lower bound. Build a paraphrase-aware detector, run it on an
open corpus against public benchmarks, and report the ratio between semantic and
lexical contamination. If it is large, a great deal of the evaluation literature
needs re-reading.

**What does the quality filter actually select for?** Train two models on the
same crawl, one filtered by a quality classifier and one by heuristics alone,
and measure per-dialect and per-register performance. The hypothesis that
quality filtering imports the reference corpus's demographics is testable and
largely untested.

**Does model-generated text in the corpus degrade successive generations?**
Web text increasingly contains model output. Simulate the loop at small scale
over several generations with realistic mixing ratios, and measure whether
degradation appears at ratios that resemble the actual web.

## 20. Chapter Summary

A pretraining corpus is built by a pipeline of filters
{{eq:corpus-pipeline}} — source, filter, deduplicate, decontaminate, mix —
whose yields compound to a few per cent of the raw crawl. Every stage is a
modelling decision, and {{eq:adaptation-information-ratio}} means none of them
can be corrected later.

**Near-duplicate detection is the pipeline's most valuable stage.** Exact
matching is easy and rare; the pervasive case is documents differing in a
timestamp or an advertisement. MinHash makes similarity estimable in space
independent of document length, because the probability that two documents'
minimum hashes agree is exactly their Jaccard similarity
{{eq:minhash-property}}. LSH banding turns the all-pairs problem into an
approximately linear one with a tunable S-curve {{eq:lsh-probability}}, sharp
enough to catch a republished article and pass two independent documents on the
same topic.

**Deduplication improves quality, not just cost.** The budget argument
{{eq:unique-fraction}} is the smaller half. The larger half is
{{eq:duplication-reweighting}}: duplicated documents receive proportionally more
gradient, so duplication silently reweights the objective toward whatever the
web republishes most. That is the mechanism behind
{{cite:lee2022dedup}}'s memorisation result.

**Contamination detection is a lower bound and should always be described as
one.** {{eq:contamination}} finds verbatim $n$-gram overlap; a paraphrase, a
translation, or a discussion of the answer is contamination that leaves no
lexical trace, as `contamination-check` demonstrates. Every benchmark number in
the rest of this book inherits that caveat, and the durable defences are
evaluating on post-cutoff data or on data you created.

**The mixture weights are the most consequential unpublished numbers in the
field.** {{eq:data-mixture}} makes them explicit; {{cite:gao2020pile}} and
{{cite:touvron2023llama}} are the notable exceptions to their being secret. For
frontier models, what is in the training data is not knowable from outside — and
any explanation of such a model's behaviour that depends on its corpus is
speculation, which should be stated as such.

## 21. Further Reading

{{cite:lee2022dedup}} is the essential paper here and it is short. Read §3 for
the methods and §5 for the train/test overlap finding, which is the part with
consequences outside this chapter. It is the rare paper whose result makes both
your models better and your evaluations more honest.

{{cite:gao2020pile}} is best read as a table: §2's list of 22 sources with sizes
and weights is the document, and the significance is that it exists at all.
Compare its specificity against any frontier model's description of its data.

{{cite:touvron2023llama}}'s §2 is two pages and is the most specific public
description of a modern pretraining mixture. Read it alongside
{{cite:brown2020}}'s Table 2.2 and notice that both are far more detailed than
anything published since.

{{cite:gunasekar2023}} for the data-quality argument. Read it sceptically — the
contamination concerns are real and the benchmark is narrow — while taking
seriously that the scaling laws contain no data-quality term at all.

**Where to go next:** {{ch:fm-scaling-laws}} takes the corpus this chapter
builds and the run {{ch:fm-pretraining}} describes, and answers the question
both of them assumed: how big a model, on how many tokens.
