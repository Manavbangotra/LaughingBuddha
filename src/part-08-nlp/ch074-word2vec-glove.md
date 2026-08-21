---
id: nlp-static-embeddings
number: 74
part: VIII
tier: full
status: draft
requires: [nlp-subword, tf-embeddings, dl-losses, dl-optimizers, math-covariance,
           math-probability, ml-pca, math-eigen, dl-autoencoders]
provides: [static-embedding, distributional-hypothesis, skip-gram, cbow,
           negative-sampling, glove, cooccurrence-matrix, pmi, word-analogy,
           embedding-evaluation, fasttext, context-window]
citations: [mikolov2013efficient, mikolov2013distributed, pennington2014, reimers2019,
            levy2014, levy2015, bojanowski2017, peters2018]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the distributional hypothesis and explain precisely which part of
   meaning it captures and which it cannot.
2. Derive the skip-gram objective and show why its full-softmax form is
   computationally impossible at realistic vocabulary sizes.
3. Derive negative sampling as a replacement objective, and state exactly what
   problem it is solving and what it gives up.
4. Derive GloVe's weighted least-squares objective from a requirement on ratios
   of co-occurrence probabilities.
5. Explain {{cite:levy2014}}'s result that skip-gram implicitly factorises a
   shifted PMI matrix, and what it implies about the count-versus-predict debate.
6. Evaluate embeddings intrinsically and extrinsically, and state why the two
   correlate poorly.
7. Decide when a static embedding is still the correct engineering choice.

## 2. Why This Matters

**This is where a vector acquires meaning for the first time in this book.**
{{ch:tf-embeddings}} described the embedding matrix as a lookup table trained by
whatever loss sits at the end of the network. This chapter is about training that
table directly, with an objective whose only goal is that the geometry be useful
— and it is the origin of every claim about "semantic space" made later.

**The techniques here are the direct ancestors of modern retrieval.** Negative
sampling ({{cite:mikolov2013distributed}}) is the contrastive objective that
trains today's embedding models ({{part:11}}). The cosine-similarity-in-a-learned-
space idea is what a vector database indexes. Skip a chapter of history and you
lose the derivation of the objective that {{part:12}} depends on.

**They are still deployed, and the reason is arithmetic.** A 300-dimensional
GloVe lookup is a memory access. A BERT encoding is 110M parameters of compute
per sequence — five or six orders of magnitude more work. At high throughput on
short texts that difference decides the architecture, and
{{cite:reimers2019}} reports the uncomfortable detail that averaged GloVe vectors
*beat* out-of-the-box BERT sentence vectors, which makes the cheap baseline a
real baseline rather than a courtesy.

**This chapter also contains the field's cleanest cautionary tale.**
{{cite:levy2014}} proved that the celebrated neural method was factorising a
matrix that count-based methods had been building for twenty years, and
{{cite:levy2015}} showed that most of the reported improvement was
hyperparameters. Both findings arrived after the method was universally adopted.
The pattern recurs in {{ch:nlp-bert}} and again in {{part:9}}, and recognising it
is more valuable than any individual result here.

## 3. Prerequisites

{{ch:nlp-subword}} for the units being embedded. {{ch:tf-embeddings}} for the
embedding matrix and the two-matrices structure that reappears here as input and
output vectors. {{ch:dl-losses}} for cross-entropy and the logistic loss.
{{ch:dl-optimizers}} for SGD. {{ch:math-probability}} for conditional
probability, {{ch:math-covariance}} for the second-order statistics a
co-occurrence matrix holds, and {{ch:ml-pca}} with {{ch:math-eigen}} for the
factorisation view in {{sec:6-mathematical-foundation}}. {{ch:dl-autoencoders}}
for representation learning as a goal in itself.

## 4. Intuitive Explanation

Here is a sentence with a word you do not know:

> The bartender poured me a glass of *wug* and I drank it slowly.

You now know a great deal about *wug*. It is a liquid, it is drinkable, it is
served in glasses by bartenders, and it is probably alcoholic. You learned this
from the company the word keeps — and that is the entire idea.

**The distributional hypothesis:** words appearing in similar contexts have
similar meanings. It is not a theory of meaning; it is an observation that
meaning and distribution are correlated strongly enough to be useful.

Turning that into vectors requires one more step. Give every word a vector, and
train those vectors so that a word's vector predicts its neighbours' vectors.
Words with similar neighbours end up with similar vectors, because they are being
pulled toward the same places by the same forces.

**Two ways to set up the prediction.** Predict the context from the word
(**skip-gram**), or predict the word from its context (**CBOW**). Skip-gram makes
more training examples per sentence and works better on rare words, which is why
it won.

**Two ways to count.** Use each local window as a training example
(word2vec), or first build the whole co-occurrence table and fit vectors to it
(GloVe). For a decade these looked like rival philosophies. {{cite:levy2014}}
showed they are the same thing computed differently.

> NOTE: The famous result — `king - man + woman ≈ queen` — is real and it is
> the most over-quoted finding in NLP. The standard evaluation protocol excludes
> the three input words from the candidate answers. Without that exclusion the
> nearest vector is frequently just `king` again, because the arithmetic moves
> you a short distance in a very high-dimensional space. State the result with
> its protocol or do not state it.

**The mental model:** each word is a point positioned by the average company it
keeps, and direction in that space encodes relationships that are consistent
across many word pairs. Where it breaks down: one word gets exactly one point, so
*bank* is placed at a compromise between riverbanks and finance — a location that
is correct for neither. That single limitation is what {{ch:nlp-contextual}} is
the answer to.

## 5. Formal Explanation

### 5.1 Setup

Let $V$ be the vocabulary. Every word $w$ gets **two** vectors: an input vector
$\vec{v}_w \in \R^d$ used when $w$ is the centre word, and an output vector
$\vec{u}_w \in \R^d$ used when $w$ is a context word. Two matrices, exactly as in
{{ch:tf-embeddings}}, and for the same reason — a word's role as predictor and as
prediction are different jobs.

A **context window** of size $m$ around position $t$ is the set of positions
$\{t-m,\dots,t-1,t+1,\dots,t+m\}$.

### 5.2 Skip-gram

Maximise the probability of the context given the centre word, over the corpus:

$$
\Loss = -\frac{1}{T}\sum_{t=1}^{T}\ \sum_{\substack{-m \le j \le m \\ j \ne 0}}
 \log P(w_{t+j} \given w_t)
$$ (eq:skipgram-objective)

with the softmax parameterisation

$$
P(o \given c) = \frac{\exp(\vec{u}_o\T\vec{v}_c)}
                     {\sum_{w \in V}\exp(\vec{u}_w\T\vec{v}_c)}
$$ (eq:skipgram-softmax)

**The denominator is the problem.** It sums over the entire vocabulary, for every
training example. With $|V| = 10^6$ and $10^{10}$ context pairs in the corpus,
the normalisation alone is $10^{16}$ dot products — which is why
{{eq:skipgram-softmax}} is a definition and never an implementation.

### 5.3 Negative sampling

{{cite:mikolov2013distributed}} replaces the multi-class problem with a binary
one. Instead of "which of a million words is the context", ask "is this pair a
real co-occurrence or a fabricated one":

$$
\Loss_{\text{neg}} = -\log\sigma(\vec{u}_o\T\vec{v}_c)
 - \sum_{k=1}^{K}\E_{w_k \sim P_n}\big[\log\sigma(-\vec{u}_{w_k}\T\vec{v}_c)\big]
$$ (eq:negative-sampling)

where $\sigma$ is the logistic function and $P_n$ is a noise distribution over
words. $K$ is typically 5-20 for small corpora and 2-5 for large ones.

**The cost changes from $O(|V|)$ to $O(K)$ per example** — six orders of
magnitude at realistic vocabulary sizes, and that is the whole reason the method
was usable.

The noise distribution is the unigram distribution raised to the $3/4$ power:

$$
P_n(w) \propto f(w)^{3/4}
$$ (eq:noise-distribution)

The exponent is empirical, not derived. It flattens the distribution so that
rare words appear as negatives more often than their frequency warrants — with
$f = 0.9$ and $f = 0.09$, the raw ratio of 10 becomes $10^{3/4} \approx 5.6$.

> RESEARCH NOTE: The $3/4$ has never been given a principled justification. It
> was tuned, it worked, and it was copied into every subsequent implementation
> including several that are not word2vec. It is worth knowing which constants
> in a field are derived and which are inherited.

### 5.4 CBOW

Reverse the conditioning: average the context vectors and predict the centre.

$$
\hat{\vec{v}} = \frac{1}{2m}\sum_{\substack{-m\le j\le m\\ j\ne 0}} \vec{v}_{w_{t+j}},
\qquad
\Loss = -\log P(w_t \given \hat{\vec{v}})
$$ (eq:cbow)

CBOW is faster — one prediction per position instead of $2m$ — and worse on rare
words, because averaging the context smooths away exactly the signal a rare word
needs.

### 5.5 GloVe

{{cite:pennington2014}} starts from a different question: what should a good
embedding satisfy? Let $X_{ij}$ be the number of times word $j$ appears in the
context of word $i$, and $P_{ij} = X_{ij}/X_i$.

The insight is that **ratios** of co-occurrence probabilities carry the meaning.
For $i = \texttt{ice}$, $j = \texttt{steam}$:

{#tbl:glove-ratios caption="Co-occurrence probability ratios from the GloVe paper's motivating example. The ratio, not the individual probability, is what discriminates: solid and gas separate cleanly while water and fashion — one relevant to both, one to neither — both sit near 1."}

| probe word $k$ | $P(k\given\texttt{ice})$ | $P(k\given\texttt{steam})$ | ratio |
|---|---|---|---|
| solid | large | small | $\gg 1$ |
| gas | small | large | $\ll 1$ |
| water | large | large | $\approx 1$ |
| fashion | small | small | $\approx 1$ |

Requiring a function $F$ of the vectors to reproduce these ratios,

$$
F(\vec{w}_i, \vec{w}_j, \tilde{\vec{w}}_k) = \frac{P_{ik}}{P_{jk}}
$$ (eq:glove-requirement)

and demanding that $F$ depend only on the difference $\vec{w}_i - \vec{w}_j$, be
a scalar function of that difference and $\tilde{\vec{w}}_k$, and turn addition
into multiplication, forces $F = \exp$ and yields

$$
\vec{w}_i\T\tilde{\vec{w}}_k + b_i + \tilde{b}_k = \log X_{ik}
$$ (eq:glove-log-bilinear)

The derivation is completed in {{sec:6-mathematical-foundation}}. The training
objective is weighted least squares on this identity:

$$
\Loss_{\text{GloVe}} = \sum_{i,j=1}^{|V|} f(X_{ij})
 \big(\vec{w}_i\T\tilde{\vec{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2
$$ (eq:glove-objective)

with the weighting function

$$
f(x) = \begin{cases}
 (x/x_{\max})^{\alpha} & x < x_{\max}\\
 1 & \text{otherwise}
\end{cases}
\qquad x_{\max}=100,\ \alpha = 3/4
$$ (eq:glove-weighting)

**The weighting function is doing two jobs.** It stops very frequent
co-occurrences from dominating the loss, and — because $f(0) = 0$ — it excludes
the zero entries entirely, which is essential since the matrix is over 99%
zeros. Note $3/4$ again, and again empirically chosen.

## 6. Mathematical Foundation

### 6.1 The gradient of negative sampling

Differentiate {{eq:negative-sampling}} with respect to the centre vector:

$$
\frac{\partial \Loss_{\text{neg}}}{\partial \vec{v}_c}
 = \big(\sigma(\vec{u}_o\T\vec{v}_c) - 1\big)\vec{u}_o
 + \sum_{k=1}^{K}\sigma(\vec{u}_{w_k}\T\vec{v}_c)\,\vec{u}_{w_k}
$$ (eq:negsampling-gradient)

using $\frac{\dd}{\dd z}\log\sigma(z) = 1-\sigma(z)$ and
$\frac{\dd}{\dd z}\log\sigma(-z) = -\sigma(z)$.

Read the two terms. The first pulls $\vec{v}_c$ **toward** $\vec{u}_o$, with
strength $1 - \sigma(\vec{u}_o\T\vec{v}_c)$ — large when the model is wrong,
vanishing when it is already confident. The second pushes $\vec{v}_c$ **away
from** each negative, with strength $\sigma(\vec{u}_{w_k}\T\vec{v}_c)$ — large
only when a negative is wrongly scored high.

**This is exactly the structure of a contrastive loss**: attract the positive,
repel sampled negatives, weight both by how wrong you currently are. The
InfoNCE objectives of {{part:11}} are this equation with a different
normalisation.

### 6.2 Completing the GloVe derivation

From {{eq:glove-requirement}}, require $F$ to depend on the difference of the
first two arguments and to be a scalar function of the resulting vector against
$\tilde{\vec{w}}_k$:

$$
F\big((\vec{w}_i - \vec{w}_j)\T\tilde{\vec{w}}_k\big) = \frac{P_{ik}}{P_{jk}}
$$ (eq:glove-step1)

The left side turns a difference of arguments into a ratio of outputs, so $F$
must be a homomorphism from $(\R,+)$ to $(\R_{>0},\times)$ — that is,
$F = \exp$:

$$
\exp\big(\vec{w}_i\T\tilde{\vec{w}}_k - \vec{w}_j\T\tilde{\vec{w}}_k\big)
 = \frac{P_{ik}}{P_{jk}}
 \implies \vec{w}_i\T\tilde{\vec{w}}_k = \log P_{ik} = \log X_{ik} - \log X_i
$$ (eq:glove-step2)

The term $\log X_i$ depends on $i$ alone, so absorb it into a bias $b_i$; add
$\tilde{b}_k$ to restore the symmetry that $X$ has and the equation does not.
This gives {{eq:glove-log-bilinear}}.

$\square$

**GloVe is a log-bilinear model fitted by weighted least squares to the log of
the co-occurrence matrix.** Nothing in the derivation is neural.

### 6.3 Skip-gram is implicit matrix factorisation

{{cite:levy2014}}'s result. Consider the negative-sampling objective for a single
$(w,c)$ pair, aggregated over the corpus. Let $\#(w,c)$ be the pair count,
$\#(w)$ and $\#(c)$ the marginals, $D$ the total. Writing
$x = \vec{u}_c\T\vec{v}_w$, the aggregate objective contributed by this pair is

$$
\ell(x) = \#(w,c)\log\sigma(x)
 + K\,\frac{\#(w)\#(c)}{D}\log\sigma(-x)
$$ (eq:levy-objective)

Set $\ell'(x) = 0$:

$$
\#(w,c)\big(1-\sigma(x)\big) = K\frac{\#(w)\#(c)}{D}\sigma(x)
$$

$$
\implies \frac{\sigma(x)}{1-\sigma(x)} = e^{x}
 = \frac{\#(w,c)\,D}{K\,\#(w)\#(c)}
$$

$$
\implies \vec{u}_c\T\vec{v}_w = \log\frac{\#(w,c)\,D}{\#(w)\#(c)} - \log K
 = \text{PMI}(w,c) - \log K
$$ (eq:sgns-is-pmi)

$\square$

**Skip-gram with negative sampling is factorising the PMI matrix shifted by
$\log K$.** No approximation was made — this is the exact optimum of its own
objective when the dimension is unconstrained.

The consequences are worth stating plainly:

1. **The count/predict distinction is not real.** Both methods factorise a
   function of the same co-occurrence statistics.
2. **The number of negatives $K$ is a shift constant**, not a mysterious
   regularizer.
3. **A truncated SVD of the shifted PMI matrix is a legitimate alternative
   algorithm**, and {{cite:levy2014}} shows it is competitive.
4. GloVe fits $\log X_{ij}$; skip-gram fits $\text{PMI} - \log K$. Both are
   log-transformed functions of $X$, differing in normalisation and weighting.

### 6.4 A worked PMI calculation

Corpus of $D = 1000$ context pairs. `doctor` occurs 50 times, `patient` 40
times, `the` 300 times. Pair counts: `(doctor, patient)` = 12,
`(doctor, the)` = 20.

$$
\text{PMI}(\texttt{doctor},\texttt{patient})
 = \log\frac{12 \times 1000}{50\times 40} = \log 6 = 1.79
$$

$$
\text{PMI}(\texttt{doctor},\texttt{the}) = \log\frac{20\times 1000}{50 \times 300}
 = \log 1.33 = 0.29
$$

**`doctor` co-occurs with `the` more often in raw count and with `patient` far
more in PMI.** The raw count is dominated by the marginal frequency of `the`;
PMI divides it out. This is the same denominator that separated WordPiece from
BPE in {{eq:wordpiece-pmi}} — the identical statistical idea, applied one level
up. The `pmi-svd-baseline` listing reproduces this reversal on a real
co-occurrence matrix rather than invented counts.

With $K = 5$ negatives, skip-gram's target inner product is
$1.79 - \log 5 = 0.18$ for the informative pair and $0.29 - 1.61 = -1.32$ for
the uninformative one — the shift by $\log K$ pushes the uninformative pair
below zero, which is what makes it repel rather than attract.

## 7. Internal Mechanics

```mermaid {#fig:skipgram-mechanics caption="One skip-gram training step with negative sampling. The centre word's input vector is pulled toward the true context word's output vector and pushed away from K sampled negatives. Only K+2 of the |V| vectors are touched, which is why the update is O(Kd) rather than O(|V|d)."}
graph TD
  A["corpus stream<br/>…the doctor examined the patient…"] --> B["centre word: 'examined'<br/>context: doctor, the, the, patient"]
  B --> C["positive pair<br/>(examined, patient)"]
  B --> D["sample K negatives<br/>from f(w)^0.75"]
  C --> E["v_examined · u_patient<br/>→ σ → pull together"]
  D --> F["v_examined · u_neg_k<br/>→ σ → push apart"]
  E --> G["SGD update<br/>K+2 vectors touched"]
  F --> G
  style G fill:#dfe,stroke:#5a5
```

**Two matrices, one kept.** Training learns $\mat{V}$ (input) and $\mat{U}$
(output), both $|V|\times d$. At the end $\mat{U}$ is discarded and $\mat{V}$ is
"the embeddings" — or their sum is used, which sometimes works slightly better.
There is no principled reason for either choice, which is worth knowing.

**Subsampling frequent words.** Before training, discard word $w$ with
probability

$$
p_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}
$$ (eq:subsampling)

with $t\approx 10^{-5}$. This removes most occurrences of `the`, `of`, and `and`,
which speeds up training and improves quality — and it has a side effect worth
noticing: deleting a word *widens* the effective window across the gap it leaves,
so subsampling silently increases context range.

**Memory.** The co-occurrence matrix GloVe needs is $|V|^2$ in principle. With
$|V| = 400{,}000$ that is $1.6\times 10^{11}$ entries — impossible dense, and
about 99.9% zero. Only nonzeros are stored, and {{eq:glove-weighting}}'s $f(0)=0$
means the zeros are not merely skipped for efficiency but genuinely excluded from
the objective. word2vec never materialises the matrix at all, which is its main
engineering advantage.

## 8. Implementation

Skip-gram with negative sampling, complete, in numpy.

```python {tier=A name=skipgram-negative-sampling}
"""Skip-gram with negative sampling from scratch. Equation (eq:negative-sampling)."""
import numpy as np
from collections import Counter

SENTENCES = [
    "the doctor examined the patient in the clinic",
    "the nurse treated the patient in the clinic",
    "the doctor prescribed medicine for the patient",
    "the nurse prescribed medicine for the patient",
    "the patient visited the clinic to see the doctor",
    "the patient visited the clinic to see the nurse",
    "the engineer debugged the program in the terminal",
    "the programmer debugged the program in the terminal",
    "the engineer deployed the server for the program",
    "the programmer deployed the server for the program",
    "the program crashed so the engineer read the logs",
    "the program crashed so the programmer read the logs",
    "the chef prepared the dish in the kitchen",
    "the baker prepared the dish in the kitchen",
    "the chef seasoned the dish with the spices",
    "the baker seasoned the dish with the spices",
    "the dish burned so the chef opened the window",
    "the dish burned so the baker opened the window",
]
# Note what this corpus does NOT contain: 'doctor' and 'nurse' never occur in
# the same sentence, and neither do 'engineer'/'programmer' or 'chef'/'baker'.
# Any similarity the model finds between them is second-order — inferred from
# shared contexts alone, which is the distributional hypothesis under test.
CORPUS = ((" ".join(SENTENCES) + " ") * 40).split()

WINDOW, DIM, K, EPOCHS, LR = 2, 32, 5, 5, 0.05
rng = np.random.default_rng(0)

counts = Counter(CORPUS)
vocab = sorted(counts)
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Noise distribution: unigram^(3/4) — equation (eq:noise-distribution).
freqs = np.array([counts[w] for w in vocab], dtype=float)
noise = freqs ** 0.75
noise /= noise.sum()

# Training pairs from the sliding window.
ids = [idx[w] for w in CORPUS]
pairs = np.array([(ids[t], ids[t + j])
                  for t in range(len(ids))
                  for j in range(-WINDOW, WINDOW + 1)
                  if j != 0 and 0 <= t + j < len(ids)])
print(f"{len(vocab)} word types, {len(CORPUS):,} tokens, {len(pairs):,} training pairs")

Vin = rng.normal(0, 0.1, (V, DIM))     # input (centre) vectors
Uout = np.zeros((V, DIM))              # output (context) vectors


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


for epoch in range(EPOCHS):
    order = rng.permutation(len(pairs))
    total = 0.0
    for c, o in pairs[order]:
        negs = rng.choice(V, size=K, p=noise)
        v = Vin[c]

        pos = sigmoid(Uout[o] @ v)
        neg = sigmoid(Uout[negs] @ v)
        total += -np.log(pos + 1e-10) - np.log(1 - neg + 1e-10).sum()

        # Gradient of equation (eq:negsampling-gradient): attract the positive,
        # repel the K negatives, each weighted by how wrong the model is now.
        grad_v = (pos - 1.0) * Uout[o] + neg @ Uout[negs]
        Uout[o] -= LR * (pos - 1.0) * v
        for i, n in enumerate(negs):
            Uout[n] -= LR * neg[i] * v
        Vin[c] -= LR * grad_v
    print(f"epoch {epoch + 1}: mean loss {total / len(pairs):.4f}")

E = Vin / (np.linalg.norm(Vin, axis=1, keepdims=True) + 1e-10)


def neighbours(word, k=3):
    sims = E @ E[idx[word]]
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 3)) for i in order if vocab[i] != word][:k]


print()
for w in ["doctor", "engineer", "chef", "patient", "program"]:
    print(f"{w:<11} -> {neighbours(w)}")

med = {"doctor", "nurse", "patient", "clinic", "medicine"}
tech = {"engineer", "programmer", "program", "server", "terminal", "logs"}


def mean_sim(a, b):
    return float(np.mean([E[idx[x]] @ E[idx[y]] for x in a for y in b if x != y]))


print(f"\nwithin medical:   {mean_sim(med, med):+.3f}")
print(f"within technical: {mean_sim(tech, tech):+.3f}")
print(f"across the two:   {mean_sim(med, tech):+.3f}")

# doctor and nurse NEVER co-occur in this corpus. Any similarity is inferred.
assert not any("doctor" in s and "nurse" in s for s in SENTENCES)
print("\n'doctor' and 'nurse' share no sentence, yet are near neighbours — "
      "the similarity is second-order, from shared contexts alone.")
```

**The assertion at the end is the point of the listing.** `doctor` and `nurse`
never appear in the same sentence, so no amount of counting their
co-occurrences could relate them — their co-occurrence count is exactly zero.
They end up as near neighbours because they occupy the same *slots*: both are
followed by `examined` or `treated`, both precede `the patient`. That is
second-order similarity, and it is what the distributional hypothesis actually
buys. The within-versus-across similarity figures show the same thing
aggregated over three domains.

Now the same corpus through the count-based route, to check
{{eq:sgns-is-pmi}} empirically:

```python {tier=A name=pmi-svd-baseline}
"""Build the co-occurrence matrix, shift its PMI, factorise with SVD."""
import numpy as np

SENTENCES = [
    "the doctor examined the patient in the clinic",
    "the nurse treated the patient in the clinic",
    "the doctor prescribed medicine for the patient",
    "the nurse prescribed medicine for the patient",
    "the patient visited the clinic to see the doctor",
    "the patient visited the clinic to see the nurse",
    "the engineer debugged the program in the terminal",
    "the programmer debugged the program in the terminal",
    "the engineer deployed the server for the program",
    "the programmer deployed the server for the program",
    "the program crashed so the engineer read the logs",
    "the program crashed so the programmer read the logs",
    "the chef prepared the dish in the kitchen",
    "the baker prepared the dish in the kitchen",
    "the chef seasoned the dish with the spices",
    "the baker seasoned the dish with the spices",
    "the dish burned so the chef opened the window",
    "the dish burned so the baker opened the window",
]
# Note what this corpus does NOT contain: 'doctor' and 'nurse' never occur in
# the same sentence, and neither do 'engineer'/'programmer' or 'chef'/'baker'.
# Any similarity the model finds between them is second-order — inferred from
# shared contexts alone, which is the distributional hypothesis under test.
CORPUS = ((" ".join(SENTENCES) + " ") * 40).split()

WINDOW, DIM, K = 2, 32, 5
vocab = sorted(set(CORPUS))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Co-occurrence counts X_ij.
X = np.zeros((V, V))
ids = [idx[w] for w in CORPUS]
for t, c in enumerate(ids):
    for j in range(-WINDOW, WINDOW + 1):
        if j != 0 and 0 <= t + j < len(ids):
            X[c, ids[t + j]] += 1

D = X.sum()
row, col = X.sum(1, keepdims=True), X.sum(0, keepdims=True)

with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((X * D) / (row * col))
pmi[~np.isfinite(pmi)] = 0.0

# Shifted positive PMI — equation (eq:sgns-is-pmi) says SGNS targets PMI - log K.
sppmi = np.maximum(pmi - np.log(K), 0.0)

U, S, _ = np.linalg.svd(sppmi)
E = U[:, :DIM] * np.sqrt(S[:DIM])
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-10


def neighbours(word, k=3):
    sims = E @ E[idx[word]]
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 3)) for i in order if vocab[i] != word][:k]


print("Nearest neighbours from a truncated SVD of the shifted PMI matrix:")
for w in ["doctor", "engineer", "chef", "patient"]:
    print(f"  {w:<11} -> {neighbours(w)}")

print()
print(f"{'pair':<22} {'raw count':>10} {'PMI':>8}")
for a, b in [("doctor", "patient"), ("doctor", "the")]:
    print(f"{a + ' / ' + b:<22} {X[idx[a], idx[b]]:>10.0f} "
          f"{pmi[idx[a], idx[b]]:>+8.3f}")

print("\nRaw counts rank 'the' far above 'patient'; PMI divides out the marginal "
      "frequency and reverses the ranking. No gradient descent produced these "
      "vectors — only counting and an SVD.")
```

**Two programs, no shared code, and the same domain structure recovered.** One
ran stochastic gradient descent over sampled pairs; the other counted and took
an SVD. The neighbour lists are not identical — they are two finite-sample
estimates of the same underlying matrix — but `engineer` finds `programmer` and
`chef` finds `baker` either way. That is {{cite:levy2014}}'s theorem showing up
as an experimental result rather than an algebraic one.

Finally, GloVe's objective on the same matrix:

```python {tier=A name=glove-fit}
"""GloVe: weighted least squares on log co-occurrence. Equation (eq:glove-objective)."""
import numpy as np

SENTENCES = [
    "the doctor examined the patient in the clinic",
    "the nurse treated the patient in the clinic",
    "the doctor prescribed medicine for the patient",
    "the nurse prescribed medicine for the patient",
    "the patient visited the clinic to see the doctor",
    "the patient visited the clinic to see the nurse",
    "the engineer debugged the program in the terminal",
    "the programmer debugged the program in the terminal",
    "the engineer deployed the server for the program",
    "the programmer deployed the server for the program",
    "the program crashed so the engineer read the logs",
    "the program crashed so the programmer read the logs",
    "the chef prepared the dish in the kitchen",
    "the baker prepared the dish in the kitchen",
    "the chef seasoned the dish with the spices",
    "the baker seasoned the dish with the spices",
    "the dish burned so the chef opened the window",
    "the dish burned so the baker opened the window",
]
# Note what this corpus does NOT contain: 'doctor' and 'nurse' never occur in
# the same sentence, and neither do 'engineer'/'programmer' or 'chef'/'baker'.
# Any similarity the model finds between them is second-order — inferred from
# shared contexts alone, which is the distributional hypothesis under test.
CORPUS = ((" ".join(SENTENCES) + " ") * 40).split()

WINDOW, DIM, XMAX, ALPHA, EPOCHS, LR = 2, 32, 100.0, 0.75, 300, 0.05
rng = np.random.default_rng(0)

vocab = sorted(set(CORPUS))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
ids = [idx[w] for w in CORPUS]

X = np.zeros((V, V))
for t, c in enumerate(ids):
    for j in range(-WINDOW, WINDOW + 1):
        if j != 0 and 0 <= t + j < len(ids):
            X[c, ids[t + j]] += 1.0 / abs(j)      # GloVe weights by distance

nz_i, nz_j = np.nonzero(X)
nz_x = X[nz_i, nz_j]
print(f"co-occurrence matrix: {V}x{V} = {V * V} cells, "
      f"{len(nz_x)} nonzero ({100 * len(nz_x) / V ** 2:.1f}% dense)")

# f(x) from equation (eq:glove-weighting): caps frequent pairs, and f(0)=0
# excludes the zeros entirely — which is why only nonzeros are iterated.
w = np.minimum((nz_x / XMAX) ** ALPHA, 1.0)
logx = np.log(nz_x)

W = rng.normal(0, 0.1, (V, DIM))
Wt = rng.normal(0, 0.1, (V, DIM))
b, bt = np.zeros(V), np.zeros(V)

# AdaGrad, as in the GloVe paper: the per-parameter step size matters here
# because word frequencies span orders of magnitude, and plain SGD at a rate
# large enough for rare words diverges on frequent ones.
aW, aWt = np.ones_like(W), np.ones_like(Wt)
ab, abt = np.ones(V), np.ones(V)

for epoch in range(EPOCHS):
    pred = np.einsum("ij,ij->i", W[nz_i], Wt[nz_j]) + b[nz_i] + bt[nz_j]
    diff = pred - logx
    loss = float((w * diff ** 2).sum())

    g = (2 * w * diff)[:, None]
    dW, dWt = np.zeros_like(W), np.zeros_like(Wt)
    db, dbt = np.zeros(V), np.zeros(V)
    np.add.at(dW, nz_i, g * Wt[nz_j])
    np.add.at(dWt, nz_j, g * W[nz_i])
    np.add.at(db, nz_i, 2 * w * diff)
    np.add.at(dbt, nz_j, 2 * w * diff)

    aW += dW ** 2; aWt += dWt ** 2; ab += db ** 2; abt += dbt ** 2
    W -= LR * dW / np.sqrt(aW)
    Wt -= LR * dWt / np.sqrt(aWt)
    b -= LR * db / np.sqrt(ab)
    bt -= LR * dbt / np.sqrt(abt)

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:>4}: weighted squared error {loss:9.2f}")

E = W + Wt                       # the standard choice: sum the two matrices
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-10

print()
for word in ["doctor", "engineer", "chef"]:
    sims = E @ E[idx[word]]
    order = [i for i in np.argsort(-sims) if vocab[i] != word][:3]
    print(f"{word:<11} -> {[(vocab[i], round(float(sims[i]), 3)) for i in order]}")

print("\nSame corpus, same neighbourhoods, a least-squares fit to log counts "
      "instead of sampled gradient steps.")
```

Note that `the` appears in `doctor`'s neighbour list at a high similarity. This
is not a bug and it is worth pausing on: a word that occurs in every context has
high co-occurrence with everything, so it lands near the centre of the space and
is close to everything. It is the same frequency pathology that
{{sec:12-failure-modes}} lists, and the same one that reappears at sentence
level as anisotropy in {{ch:nlp-similarity}}. On a real corpus, subsampling
{{eq:subsampling}} removes most of these occurrences before training and the
effect largely disappears.

## 9. Practical Example

A product-search team must decide how to represent 40 million short queries per
day for a downstream intent classifier. Latency budget: 5 ms end to end for the
whole feature pipeline. The obvious modern answer is a transformer encoder; the
question is whether the obvious answer is affordable.

The honest comparison is not accuracy against accuracy. It is accuracy per
millisecond and per dollar, at the volume in question.

```python {tier=A name=static-vs-contextual-cost}
"""The cost side of the static-versus-contextual decision, made explicit."""

QUERIES_PER_DAY = 40_000_000
LATENCY_BUDGET_MS = 5.0

# Averaged static vectors: one lookup and one add per token.
STATIC = dict(dim=300, params=0, flops_per_token=300, ms_per_query=0.02)

# A small transformer encoder: 2·N FLOPs per token (see ch:tf-complexity).
BERT_BASE = dict(dim=768, params=110e6, flops_per_token=2 * 110e6, ms_per_query=8.0)
MINILM = dict(dim=384, params=22e6, flops_per_token=2 * 22e6, ms_per_query=1.6)

TOKENS = 8          # a short query
GPU_COST_PER_HOUR = 2.0

print(f"{'model':<12} {'dim':>5} {'MFLOPs/query':>13} {'ms':>7} "
      f"{'fits 5ms':>9} {'GPU-hours/day':>14}")
for name, m in [("static-avg", STATIC), ("MiniLM", MINILM), ("BERT-base", BERT_BASE)]:
    mflops = m["flops_per_token"] * TOKENS / 1e6
    gpu_hours = QUERIES_PER_DAY * m["ms_per_query"] / 1000 / 3600
    print(f"{name:<12} {m['dim']:>5} {mflops:>13,.1f} {m['ms_per_query']:>7.2f} "
          f"{str(m['ms_per_query'] < LATENCY_BUDGET_MS):>9} {gpu_hours:>14,.0f}")

ratio = BERT_BASE["flops_per_token"] / STATIC["flops_per_token"]
print(f"\nBERT-base does {ratio:,.0f}x the arithmetic per token of a lookup-and-add.")
print(f"At {QUERIES_PER_DAY:,} queries/day the difference between MiniLM and "
      f"BERT-base alone is "
      f"${QUERIES_PER_DAY * (BERT_BASE['ms_per_query'] - MINILM['ms_per_query']) / 1000 / 3600 * GPU_COST_PER_HOUR:,.0f}"
      f"/day of GPU time.")
print("\nStatic embeddings are not the best representation. They are sometimes "
      "the only one that fits the budget — and that is an engineering answer, "
      "not a quality claim.")
```

The decision this produces is usually: static vectors for the high-volume
first-pass filter, a small encoder for the cases that survive it. That is the
same cascade shape as retrieve-then-rerank in {{ch:emb-reranking}}, arrived at
from cost rather than from quality.

> PRODUCTION TIP: Before adopting a transformer encoder for a high-volume feature,
> measure the averaged-static-vector baseline. It takes an afternoon, it is
> occasionally within a point or two of the expensive option, and knowing the
> gap is what makes the expensive option a decision rather than a default.

## 10. Production Considerations

**Storage and memory.** A 400k-word vocabulary at 300 dimensions in float32 is
480 MB — too large for a per-request memory budget and fine as a shared read-only
mapping. Truncating to the top 100k words typically loses very little, because
of the same Zipfian coverage argument as {{eq:zipf-coverage}}.

**Quantisation is unusually effective here.** Static vectors tolerate int8
quantisation with negligible retrieval-quality loss, giving a 4x memory
reduction. {{part:15}} covers the general technique; embeddings are the easiest
case because nothing downstream is differentiating through them.

**Vocabulary drift.** A frozen static vocabulary meets new words continuously and
has no representation for them at all — the OOV problem returns in full, since
these are word vectors, not subword ones. {{cite:bojanowski2017}}'s fastText is
the fix: compose an unseen word's vector from its character n-grams.

**They are a data artefact and inherit their corpus.** Vectors trained on a news
corpus encode that corpus's associations, including its biases, and those
associations become features in every downstream model. {{part:27}} treats this
as a measurable property rather than a caveat.

**What to log:** OOV rate on live traffic, and the fraction of queries whose
vector is the zero vector (all tokens unknown) — the second is a silent failure
that produces confident garbage downstream.

## 11. Common Mistakes

**Beginners:**

*Quoting the analogy result without its protocol.* The evaluation excludes the
three input words from the candidate set. Without that exclusion the result
frequently does not hold, and reporting it as unconditional is repeating a claim
you have not checked.

*Using cosine similarity on unnormalised vectors and calling it cosine
similarity.* Normalise, or use a dot product and say so.

*Expecting a single vector to disambiguate.* One word type, one vector, one
compromise position between all its senses. This is not a training failure; it is
the model class.

**Experienced practitioners:**

*Comparing embedding methods without equalising hyperparameters.* This is
{{cite:levy2015}}'s finding exactly: window size, negative count, subsampling
threshold, and dimensionality dominate the differences between methods. A
comparison that varies the method and the tuning measures both.

*Treating the analogy benchmark as an evaluation.* It correlates weakly with
downstream task performance. {{sec:14-evaluation}} separates the two kinds of
measurement.

*Forgetting that subsampling widens the window.* Deleting frequent words changes
the effective context radius, so the window parameter does not mean what it says
after subsampling is enabled.

*Assuming more dimensions is better.* Quality typically plateaus between 200 and
400 dimensions for static vectors, and the extra dimensions cost memory and
retrieval time for no gain.

## 12. Failure Modes

**Polysemy collapse.** A word with multiple senses receives one vector at the
frequency-weighted compromise of its senses. *Symptom:* nearest neighbours mixing
unrelated domains — `bank` next to both `river` and `loan`. *Detection:* inspect
neighbourhoods for words you know are ambiguous. *Fix:* contextual embeddings
({{ch:nlp-contextual}}); there is no fix within this model class.

**Antonyms as near neighbours.** `hot` and `cold` appear in nearly identical
contexts, so the distributional hypothesis places them close together.
*Symptom:* a sentiment or logic feature built on similarity treats opposites as
equivalent. *This is not a bug in the training*; it is the hypothesis being
exactly correct and the hypothesis being insufficient.

**Frequency bias in the geometry.** Rare words get vectors from few updates and
end up with smaller norms and less reliable directions. *Symptom:* rare words
either dominate or never appear in nearest-neighbour lists depending on
normalisation. *Detection:* plot vector norm against word frequency.

**Corpus bias becoming a model feature.** Occupational and demographic
associations present in the corpus are encoded in the geometry and inherited by
every downstream model. *Detection:* direct measurement of association strength
along known axes ({{part:27}}).

**Silent OOV.** An unknown word is mapped to a zero vector or skipped, and an
averaged sentence vector quietly becomes an average over fewer words — or, if
every word is unknown, the zero vector, which has cosine similarity 0 with
everything and produces confident nonsense downstream.

## 13. Alternatives

{#tbl:static-embedding-alternatives caption="Ways to get a vector per word, and what each trades. The first three fit essentially the same statistics with different objectives, which is levy2014's point; the last two change what a vector is a function of."}

| Method | Fits | Handles OOV | Contextual | Trades away |
|---|---|---|---|---|
| Skip-gram + negatives | PMI $-\log K$ | no | no | needs the corpus streamed |
| CBOW | same statistics | no | no | rare-word quality |
| GloVe | $\log X_{ij}$ | no | no | must materialise co-occurrences |
| PMI + truncated SVD | shifted PPMI | no | no | memory; exact, not stochastic |
| fastText | subword-composed | **yes** | no | larger model, slower lookup |
| Contextual (ELMo/BERT) | context-dependent | yes | **yes** | 10⁴-10⁶x the compute |

**Which compute the same function and which do not.** The first four are all
factorising a log-transformed function of the same co-occurrence matrix — they
differ in weighting and normalisation, and {{cite:levy2015}} showed the resulting
quality differences are smaller than the hyperparameter differences. fastText
changes the input representation. Contextual embeddings change what the vector is
a function of, and are therefore a different model class, not a better fit of the
same one.

## 14. Evaluation

**Intrinsic evaluation** asks whether the geometry looks right.

- **Word similarity**: correlate cosine similarity against human judgement
  datasets, reporting Spearman's $\rho$.
- **Analogy**: `a:b :: c:?` solved as $\argmax_x \cos(\vec{x}, \vec{b}-\vec{a}+\vec{c})$
  with $a,b,c$ excluded from the candidates. **Report the exclusion or the number
  is not interpretable.**
- **Categorisation**: cluster the vectors and compare against known categories.

**Extrinsic evaluation** asks whether the vectors help a task you care about:
freeze them, train a downstream classifier, and measure the task metric.

**The two correlate poorly, and this is the important fact in this section.**
{{cite:levy2015}} demonstrated that intrinsic rankings shift substantially under
hyperparameter changes that barely move downstream performance. Intrinsic
measures are cheap diagnostics; extrinsic measures are evidence.

The methodological rule this part inherits, stated once and applied throughout:
**equalise the budget before comparing methods.** Same corpus, same window, same
dimensionality, same number of updates. Otherwise you are measuring the tuning.

## 15. Advanced Concepts

**Subword-composed static vectors.** {{maturity:ESTABLISHED}}
{{cite:bojanowski2017}} represents a word as the sum of its character n-gram
vectors, giving open-vocabulary static embeddings. Still the correct choice for
morphologically rich languages and for any setting with a long tail of unseen
tokens.

**Post-processing for isotropy.** {{maturity:EMERGING}} Trained embeddings
occupy a narrow cone; removing the top principal components spreads them out and
measurably improves similarity tasks at near-zero cost. The same pathology
reappears for sentence embeddings in {{ch:nlp-similarity}}, which is where it is
treated properly.

**Cross-lingual alignment.** {{maturity:ESTABLISHED}} Two independently trained
embedding spaces can be aligned with a single orthogonal transform learned from a
small bilingual dictionary — evidence that the geometry induced by the
distributional hypothesis is substantially language-independent.

**Compositionality limits.** {{maturity:RESEARCH FRONTIER}} Averaging word
vectors discards word order entirely, so `dog bites man` and `man bites dog` are
identical. Every attempt to fix this within the static framework either
reintroduces a sequence model or fails, which is a reasonable way to understand
why the field moved to contextual representations rather than patching this one.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-embeddings}} introduced the embedding matrix as a
by-product of training; here it is the objective, and the input/output split
reappears as $\mat{V}$ and $\mat{U}$. {{ch:dl-losses}} supplied the logistic loss
that {{eq:negative-sampling}} uses. {{ch:ml-pca}} and {{ch:math-eigen}} supplied
the truncated SVD that {{sec:8-implementation}} uses to reproduce word2vec's
neighbourhoods by an entirely different route. {{ch:nlp-subword}}'s PMI criterion
for WordPiece is the identical statistic that {{eq:sgns-is-pmi}} shows skip-gram
targeting, one level up in the hierarchy.

**Forwards.** {{ch:nlp-contextual}} removes this chapter's central limitation —
one vector per type — and is best read as the direct answer to
{{sec:12-failure-modes}}'s first entry. {{ch:emb-what-they-are}} generalises
these vectors into infrastructure, and {{ch:emb-similarity}} formalises the
geometry. The contrastive structure of {{eq:negsampling-gradient}} is the
objective that trains modern retrieval encoders in {{ch:emb-models}}.
{{part:27}} treats the corpus bias these vectors encode as a measurable property.

## 17. Exercises

**Beginner**

1. State the distributional hypothesis in one sentence, then give a pair of words
   it predicts to be similar that are in fact opposites.
2. Given $|V| = 10^5$, $d = 300$, how many parameters does a skip-gram model
   have? Remember there are two matrices.
3. Why does subsampling frequent words improve embedding quality rather than only
   speed?

**Intermediate**

4. Compute $\text{PMI}$ for a pair with $\#(w,c)=30$, $\#(w)=200$, $\#(c)=150$,
   $D=10^4$, and give skip-gram's target inner product at $K=10$.
5. Explain why {{eq:glove-weighting}} sets $f(0)=0$ and what breaks without it.
6. Derive {{eq:negsampling-gradient}} for $\vec{u}_o$ rather than $\vec{v}_c$,
   and say why the update touches only $K+2$ vectors.

**Advanced**

7. Reproduce {{eq:sgns-is-pmi}} in full, stating every assumption. Where does
   the derivation require the embedding dimension to be unconstrained, and what
   changes when it is not?
8. GloVe fits $\log X_{ij}$ and skip-gram fits $\text{PMI}-\log K$. Express one
   as a function of the other and identify precisely where they differ.
9. Argue whether the $3/4$ exponents in {{eq:noise-distribution}} and
   {{eq:glove-weighting}} are related or coincidental, and design an experiment
   that would distinguish the two possibilities.

**Implementation**

10. Add CBOW to `skipgram-negative-sampling` and compare rare-word neighbourhood
    quality against skip-gram at an equal number of updates.
11. Implement the analogy evaluation with and without excluding the three input
    words, and report how often the answer changes on a set of analogies you
    construct.
12. Take the `pmi-svd-baseline` embeddings and the `skipgram-negative-sampling`
    embeddings and measure the correlation between their similarity matrices.
    {{cite:levy2014}} predicts it should be high — quantify it.
13. Implement subsampling {{eq:subsampling}} and measure the effect on the
    effective window size, defined as the mean distance between surviving
    context pairs.

**Reasoning**

14. A colleague reports that method A beats method B on a word-similarity
    benchmark. What must you know before believing this tells you anything about
    a downstream task?
15. Explain why antonyms being near neighbours is evidence that the
    distributional hypothesis is working, not failing.

## 18. Interview Questions

**Beginner**

1. What is a word embedding and what does the distributional hypothesis claim?
2. What is negative sampling and what problem does it solve?
3. Why do word2vec and GloVe give a word one vector regardless of context?

**Intermediate**

4. Derive the skip-gram objective and explain why the softmax is intractable.
5. What does GloVe fit, and what is its weighting function for?
6. What did {{cite:levy2014}} prove and why does it matter?

**Senior**

7. When would you deploy static embeddings today over a transformer encoder? Be
   specific about the constraint that decides it.
8. Your intrinsic benchmark improved and your downstream metric did not. What is
   going on, and what do you do?
9. How do you evaluate embeddings for a task you have not built yet?

**Systems**

10. Design the serving layer for static embeddings at 40M lookups per day.
    Address memory, quantisation, OOV, and updates.
11. Your embeddings encode a demographic association that is showing up in
    production decisions. Walk through detection, measurement, and remediation.

## 19. Research Questions

**How much of any embedding advance is hyperparameters?** {{cite:levy2015}}
answered this for 2015. Repeat the study design for a modern embedding model:
equalise data, budget, and tuning across two published methods and see what
survives. The methodology transfers exactly; the result is not published.

**Is there a principled noise distribution?** {{eq:noise-distribution}}'s $3/4$ is
tuned and universally copied. Derive the optimal $P_n$ for a stated objective and
measure the gap against the folk constant. A negative result would be worth
publishing.

**What exactly does the analogy structure require?** Linear analogy relations
hold for some relation types and fail for others. Characterise which relations
are linearly encodable as a function of their distributional signature, rather
than cataloguing successes.

**Where is the static/contextual crossover today?** {{sec:9-practical-example}}
computes the cost side. Run the quality side: at what task difficulty and volume
does the averaged-static baseline stop being competitive with a small distilled
encoder? The frontier has moved as distillation improved and nobody has re-drawn
it.

## 20. Chapter Summary

The distributional hypothesis — words in similar contexts have similar meanings —
converts a question about meaning into a question about co-occurrence statistics,
and every method in this chapter fits vectors to some transformation of those
statistics.

**Skip-gram** maximises the probability of context words given the centre word.
Its softmax over the vocabulary {{eq:skipgram-softmax}} is intractable, so
**negative sampling** {{eq:negative-sampling}} replaces it with a binary
discrimination against $K$ sampled words, reducing the cost per example from
$O(|V|)$ to $O(K)$. The gradient {{eq:negsampling-gradient}} attracts the
positive and repels the negatives in proportion to how wrong the model currently
is — the contrastive structure that trains modern retrieval encoders.

**GloVe** derives a log-bilinear objective {{eq:glove-log-bilinear}} from the
requirement that ratios of co-occurrence probabilities be recoverable, and fits
it by weighted least squares over the nonzero entries of the co-occurrence
matrix.

**{{cite:levy2014}} proved the two are the same thing.** Skip-gram with negative
sampling is exactly factorising the PMI matrix shifted by $\log K$
{{eq:sgns-is-pmi}}, which collapses the count-versus-predict distinction and
makes a truncated SVD a legitimate alternative algorithm.
{{cite:levy2015}} then showed most reported differences between methods were
hyperparameters — the first of three instances in this book of a methodological
failure worth recognising by name.

The model class has one structural limit: one vector per word type, which places
every ambiguous word at a compromise between its senses and places antonyms next
to each other. No amount of data or dimensionality fixes it, and
{{ch:nlp-contextual}} is the response.

Static embeddings remain deployed because the arithmetic is decisive: a lookup
and an add against 220 million FLOPs. When the volume is large and the text is
short, the cheap representation is sometimes the only one that fits — which is an
engineering conclusion, not a claim about quality.

## 21. Further Reading

{{cite:mikolov2013distributed}} is the paper to read, not
{{cite:mikolov2013efficient}}. The first introduced the architectures; the second
introduced negative sampling, subsampling, and the tricks that made it work, and
those are what survived. Sections 2.2 and 2.3 are the technical core and are two
pages.

{{cite:pennington2014}} is worth reading for its derivation, which is unusual in
this literature for starting from a stated requirement and deriving a form rather
than proposing one and evaluating it. Section 3 is the argument.

{{cite:levy2014}} is the most important paper in this chapter and it is short.
The derivation reproduced in {{sec:6-mathematical-foundation}} is essentially the
whole paper, and it is the cleanest example available of a neural method turning
out to have a closed form.

{{cite:levy2015}} is the methodological companion, and it is the one to give a
colleague who is about to compare two methods with different tuning. Its lesson
outlives its subject matter.

{{cite:bojanowski2017}} for fastText, which is the only method here still worth
reaching for on a genuinely new problem in a morphologically rich language.

{{cite:peters2018}} is the beginning of the next chapter and reads best
immediately after this one, when the one-vector-per-type limitation is fresh.

**Where to go next:** {{ch:nlp-contextual}} makes the vector a function of the
sentence rather than of the type, which is the single change that ended this
chapter's era.
