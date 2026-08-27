---
id: emb-similarity
number: 100
part: XI
tier: full
status: draft
requires: [emb-what-they-are, math-norms, math-matrices, ml-metrics,
           nlp-similarity]
provides: [metric-equivalence, magnitude-bias, relative-contrast,
           distance-concentration, intrinsic-dimension, hubness,
           mips-reduction, score-incomparability, metric-choice]
citations: [beyer1999nn, radovanovic2010hubs, malkov2020hnsw, guo2020scann,
            karpukhin2020dpr, gao2021simcse, jegou2011pq]
---

## 1. Learning Objectives

By the end of this chapter you will be able to prove when cosine, inner product,
and Euclidean distance induce the same ranking and demonstrate how badly they
diverge when they do not; state {{cite:beyer1999nn}}'s concentration theorem
with its actual hypotheses and explain why vector search works anyway; measure
relative contrast and hubness on any corpus; explain why maximum inner product
search is not a nearest-neighbour problem and what indexes do about it; and say
precisely what a similarity score of 0.82 tells you.

## 2. Why This Matters

Every decision in the rest of {{part:11}} is made *in a metric space*, and the
metric is chosen once, early, usually without discussion, and then baked into an
index that costs a full corpus re-embed to change.

The choice looks trivial — three functions that all measure "closeness" — and it
is not. On normalised vectors the three are the same function up to a monotone
transform. On un-normalised vectors, inner product and Euclidean distance agree
on the single best result **essentially never**, and {{sec:9-practical-example}}
measures exactly that. Which of the two your index computes is therefore not a
detail; it is the difference between two different retrieval systems.

The second reason is that this chapter contains the book's honest treatment of
the curse of dimensionality. The result is real, the theorem is correct, and the
conclusion people draw from it — that high-dimensional nearest-neighbour search
is hopeless — does not follow. Getting that argument right is what licenses the
next three chapters.

{{maturity:ESTABLISHED}} The geometry here is settled mathematics, most of it
from before 2011. What is unsettled is only how to *estimate* the quantities it
identifies on a real corpus.

## 3. Prerequisites

{{ch:emb-what-they-are}} for the embedding-as-dot-product framing and for
normalisation; {{ch:math-norms}} for norms, inner products, and the Cauchy–Schwarz
inequality; {{ch:math-matrices}} for the effect of a linear map on distances;
{{ch:ml-metrics}} for rank correlation; {{ch:nlp-similarity}} for cosine
similarity as first introduced.

## 4. Intuitive Explanation

Three ways to ask "are these two vectors close?"

**Inner product** $a\T b$ asks: how much do they point the same way, *scaled by
how long they are*. A long vector has a large inner product with almost
everything.

**Cosine** asks: how much do they point the same way, *ignoring length*. It is
the inner product of the two directions.

**Euclidean distance** asks: how far apart are the two points. It cares about
both direction and length — but with the opposite sign on length from the inner
product. Making a vector longer moves it *away* from a fixed query, while making
it longer *increases* its inner product with that query.

That last observation is the whole story of {{sec:5-formal-explanation}}. Inner
product rewards magnitude and Euclidean distance punishes it, so when magnitudes
vary the two are not merely different — they are pulling in opposite directions.
Cosine sits between them by refusing to look at magnitude at all.

Put every vector on the unit sphere and the distinction evaporates: all lengths
are 1, so there is nothing for the three to disagree about, and they become the
same ranking exactly. **This is why normalisation is not a preference but a
decision about which of three different retrieval systems you are building.**

### The second idea: contrast, not distance

The intuition that fails in high dimension is not "things are far apart". It is
that the *spread* of distances is informative.

In two dimensions, your nearest neighbour is much closer than your farthest, and
the ratio between them is large. As dimension grows for generic data, all
pairwise distances converge toward each other — everything is roughly the same
distance from everything — and "nearest" stops carrying information, because the
nearest and the farthest point are nearly equidistant.

The useful quantity is therefore not distance but **relative contrast**: how much
closer is the nearest than the farthest? When that ratio approaches 1, retrieval
is meaningless regardless of how good your index is.

And the crucial fact, which {{sec:9-practical-example}} demonstrates: **relative
contrast collapses with the data's intrinsic dimension, not with the number of
coordinates you store it in.** A 512-dimensional embedding of data that really
lives on an 8-dimensional manifold behaves like 8-dimensional data. That is why
768-dimensional retrieval works, and it is the entire resolution of the apparent
contradiction between {{cite:beyer1999nn}} and every deployed vector database.

## 5. Formal Explanation

### 5.1 The three scorers

For $a, b \in \R^k$:

$$ \text{ip}(a,b) = a\T b, \qquad \cos(a,b) = \frac{a\T b}{\lVert a\rVert\,\lVert b\rVert}, \qquad \text{d}_2(a,b) = \lVert a - b \rVert_2 $$ (eq:three-scorers)

They are linked by one identity, obtained by expanding the square:

$$ \lVert a - b \rVert_2^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a\T b $$ (eq:l2-ip-identity)

### 5.2 When the three agree

Fix a query $q$ and rank documents $d$. **Ranking is invariant under any strictly
monotone transform of the score**, so two scorers induce the same order whenever
one is a monotone function of the other with $q$ held fixed.

From {{eq:l2-ip-identity}}, with $\lVert q \rVert$ constant across the ranking:

$$ \lVert q - d \rVert_2^2 = \underbrace{\lVert q \rVert^2}_{\text{constant}} + \lVert d \rVert^2 - 2\,q\T d $$ (eq:l2-ranking)

Distance decreases as $q\T d$ increases *only if* $\lVert d \rVert^2$ is also
constant. So:

$$ \lVert d \rVert = c \;\;\forall d \quad\Longrightarrow\quad \text{rank}_{\text{ip}} = \text{rank}_{\cos} = \text{rank}_{-\text{d}_2} $$ (eq:rank-equivalence)

and on the unit sphere the identity specialises to the form worth memorising:

$$ \lVert a - b\rVert_2^2 = 2 - 2\cos(a,b) $$ (eq:sphere-identity)

> **MATH NOTE:** {{eq:rank-equivalence}} requires only that *documents* share a
> common norm. The query's norm is irrelevant, since it is constant across the
> ranking. This is why some systems normalise the corpus and not the queries and
> get away with it — for ranking. Any *threshold* on the raw score then depends
> on the query's norm, which is a different bug.

### 5.3 When they disagree

Drop the constant-norm assumption and {{eq:l2-ranking}}'s $\lVert d\rVert^2$
term becomes a per-document additive penalty. Comparing two documents:

$$ \text{ip}(q,d_1) > \text{ip}(q,d_2) \;\;\text{but}\;\; \text{d}_2(q,d_1) > \text{d}_2(q,d_2) \iff 0 < 2\,q\T(d_1 - d_2) < \lVert d_1\rVert^2 - \lVert d_2\rVert^2 $$ (eq:disagreement-condition)

which is a non-empty region whenever norms differ. Concretely, **inner product
is biased toward long vectors and Euclidean distance against them**:

$$ \E_{q}\big[\text{ip}(q,d)\big] \propto \lVert d \rVert, \qquad \E_q\big[\text{d}_2(q,d)^2\big] = \lVert d\rVert^2 + \text{const} $$ (eq:magnitude-bias)

for $q$ isotropic. The two biases have opposite sign, which is why
{{sec:9-practical-example}} finds their top-1 results agreeing 0% of the time.

> **WARNING:** This is not a corner case. Un-normalised embeddings routinely
> have norms varying by 3–5×, and encoder norm correlates with document length
> and token frequency — so an inner-product index over un-normalised vectors
> systematically prefers long documents, and a Euclidean one systematically
> prefers short ones. Neither preference was intended by anyone.

### 5.4 Inner product is not a metric

$\text{d}_2$ is a metric: non-negative, zero only on identity, symmetric, and
triangle-inequality-satisfying. Inner product satisfies none of the first three
usefully and has no triangle inequality at all — indeed $\text{ip}(a,a) =
\lVert a\rVert^2$, so a vector is not even maximally similar to itself among all
candidates.

This is an engineering problem, not a pedantic one. **Graph and tree indexes
assume a metric.** HNSW's greedy descent ({{cite:malkov2020hnsw}}) is justified
by the neighbourhood structure a metric induces; without a triangle inequality
the guarantee that a locally best neighbour leads toward a globally near one is
gone. Maximum inner product search (MIPS) is therefore a genuinely different
problem from nearest-neighbour search, and it has three standard treatments:

**Normalise**, and MIPS becomes NN by {{eq:rank-equivalence}}. Free when the
model was trained with normalised similarities, which is most of them.

**Augment.** Append a coordinate that absorbs the magnitude. With
$M = \max_d \lVert d\rVert$:

$$ \tilde{d} = \Big[\,d \;,\; \sqrt{M^2 - \lVert d\rVert^2}\,\Big], \qquad \tilde{q} = [\,q\;,\;0\,] $$ (eq:mips-augmentation)

Then $\lVert \tilde d\rVert = M$ for every document — constant — and
$\tilde q\T \tilde d = q\T d$. The reduction is exact: MIPS over $d$ is nearest
neighbour over $\tilde d$. It costs one dimension and a corpus-wide maximum,
and it degrades when norms are very heterogeneous, since the padded coordinate
then dominates the geometry.

**Solve MIPS directly**, which is what {{cite:guo2020scann}} does — and its
insight is that the *quantization objective* must change too, because the error
that matters for an inner product is the component parallel to the datapoint.
{{ch:emb-ann}} develops this.

## 6. Mathematical Foundation

### 6.1 Relative contrast and the concentration theorem

Define, for a query $q$ over a corpus $\mathcal{D}$:

$$ \text{RC}(q) = \frac{\max_{d \in \mathcal{D}} \text{d}_2(q,d)}{\min_{d \in \mathcal{D}} \text{d}_2(q,d)}, \qquad \text{contrast}(q) = \frac{D_{\max} - D_{\min}}{D_{\min}} = \text{RC}(q) - 1 $$ (eq:relative-contrast)

{{cite:beyer1999nn}}'s theorem: if the data and query are drawn i.i.d. and the
distance distribution satisfies

$$ \lim_{k \to \infty} \frac{\Var\!\big[\lVert X_k \rVert\big]}{\big(\E\big[\lVert X_k \rVert\big]\big)^2} = 0 $$ (eq:beyer-condition)

then for any $\epsilon > 0$,

$$ \Prob\big[\,D_{\max} \leq (1+\epsilon) D_{\min}\,\big] \;\longrightarrow\; 1 \quad \text{as } k \to \infty $$ (eq:concentration-limit)

— the nearest and farthest points become indistinguishable, and the
nearest-neighbour query is "unstable".

**Read {{eq:beyer-condition}} carefully, because it is where the escape is.** The
hypothesis is about the distribution of the *distance*, not about the number of
coordinates. It holds when coordinates contribute independent, comparable
variance — i.i.d. dimensions — because then $\lVert X_k\rVert$ concentrates by
the law of large numbers at rate $1/\sqrt{k}$.

It fails, and {{eq:concentration-limit}} does not apply, when the data lies on a
low-dimensional manifold embedded in $\R^k$. The coordinates are then heavily
dependent, the effective number of independent contributions is the *intrinsic*
dimension $k_{\text{int}} \ll k$, and the concentration rate is
$1/\sqrt{k_{\text{int}}}$:

$$ \frac{\sqrt{\Var[\lVert X\rVert]}}{\E[\lVert X\rVert]} \;\sim\; \frac{1}{\sqrt{k_{\text{int}}}} \quad\text{rather than}\quad \frac{1}{\sqrt{k}} $$ (eq:intrinsic-concentration)

**This is the resolution, and it is worth stating as a slogan: concentration
tracks intrinsic dimension, not stored dimension.** Learned embeddings are
low-rank by construction — a 768-dimensional output of a network whose useful
variation is far smaller — so they sit in the regime where the theorem's
hypothesis fails. {{sec:9-practical-example}} demonstrates this directly, by
holding intrinsic dimension fixed at 8 while sweeping the ambient dimension from
2 to 512 and showing the contrast flat.

> **RESEARCH NOTE:** What is *not* resolved is estimating $k_{\text{int}}$
> reliably. Maximum-likelihood, two-nearest-neighbour, and correlation-dimension
> estimators routinely disagree by a factor of two on real embedding corpora,
> and there is no accepted way to convert an estimate into a prediction of
> retrieval quality. The qualitative argument above is solid; the quantitative
> version is an open problem.

### 6.2 Hubness

Concentration is not the only high-dimensional pathology, and the second one is
more directly visible in production. Define the **$k$-occurrence** of a point:

$$ N_k(x) = \big|\{\,y \in \mathcal{D} : x \in \text{kNN}(y)\,\}\big| $$ (eq:k-occurrence)

By construction $\E[N_k] = k$ regardless of dimension. But
{{cite:radovanovic2010hubs}} showed that the *distribution* of $N_k$ becomes
severely right-skewed as dimension grows:

$$ \text{skew}\big(N_k\big) \;\longrightarrow\; \text{large} \quad \text{as } k \to \infty \text{ for i.i.d. data} $$ (eq:hubness-skew)

A few points — **hubs** — appear in almost everyone's neighbour list, and a large
fraction of points appear in nobody's and are unretrievable by any query. The
mechanism is proximity to the data mean: in high dimension, points slightly
closer to the centroid than average are closer to *everything* than average.

This is the formal version of a complaint every retrieval team eventually files:
*the same three documents keep coming back for unrelated queries.* It is not a
bug in the index and not a bug in the ranking; it is the geometry.

And, as with concentration, {{sec:9-practical-example}} shows it tracks intrinsic
dimension: at ambient 512 the i.i.d. corpus has one point returned for over a
thousand queries while a third of the corpus is never returned at all, and the
intrinsic-8 corpus at the same ambient dimension shows essentially none of it.

### 6.3 The score has no absolute meaning

Restating {{ch:emb-what-they-are}}'s point with the geometry available. A cosine
of 0.82 is uninterpretable without knowing the corpus's mean pairwise cosine,
because the quantity that carries information is the *margin over background*:

$$ \text{margin}(q,d) = \cos(q,d) - \E_{d' \sim \mathcal{D}}\big[\cos(q,d')\big] $$ (eq:score-margin)

An anisotropic space with background 0.80 makes 0.82 nearly meaningless; an
isotropic one with background 0.00 makes it a strong match. Since the background
differs by model, by corpus, and even by query, **raw-score thresholds do not
transfer** — not across models, not across corpora, and not reliably across time
as a corpus grows.

The operational alternatives, in increasing order of cost: rank-based cutoffs
(top-$k$), per-query normalisation against the retrieved distribution's own mean
and spread, and calibration against labelled relevance judgements. Only the last
gives a number that means something, and it must be redone whenever the model
changes.

## 7. Internal Mechanics

```mermaid {#fig:metric-decision caption="The metric decision, and where each branch leads. The question is never 'which metric is better' but 'do my document norms carry information I want to rank by' — and for almost all trained embedding models the answer is no."}
flowchart TD
    A["document vectors"] --> B{"do norms carry<br/>ranking information?"}
    B -->|"no (usual case)"| C["L2-normalise at write time"]
    C --> D["cosine = inner product = -L2<br/>(eq:rank-equivalence)<br/>any index works"]
    B -->|"yes"| E{"index supports<br/>native MIPS?"}
    E -->|"yes"| F["use it<br/>(ScaNN-style)"]
    E -->|"no"| G["augment: eq:mips-augmentation<br/>exact reduction, one extra dim"]
    G --> H["degrades if norms<br/>very heterogeneous"]
```

### 7.1 What normalisation does to the space

Normalisation is a projection onto the unit sphere, and it is **not** an
isometry — it changes distances, and can change rankings, which is exactly the
point of {{eq:rank-equivalence}}. What it preserves is direction, and what it
discards is magnitude.

The question {{fig:metric-decision}} poses is therefore whether magnitude carries
information. Three cases:

- **Contrastively trained models** ({{cite:gao2021simcse}},
  {{cite:karpukhin2020dpr}}): trained *with* normalisation in the loss, so the
  magnitude is not optimised for anything. Normalise; there is nothing there.
- **By-product embeddings**: magnitude correlates with token frequency and
  sequence length, which is information about the document but not about
  relevance. Normalise; it is the wrong information.
- **Deliberately un-normalised scorers** — recommendation models where the norm
  encodes item popularity or a learned prior. Here magnitude is signal, MIPS is
  the right problem, and {{eq:mips-augmentation}} or a native MIPS index applies.

The third case is real but rare in text retrieval, which is why "normalise" is
the correct default and why so many systems get away with never thinking about it.

### 7.2 Why the equivalence is exact and not approximate

Worth emphasising because it is often stated as an approximation. On the unit
sphere, {{eq:sphere-identity}} makes squared distance an *exactly* affine,
strictly decreasing function of cosine. Not approximately. Not asymptotically.
The rank correlation between the three scorers is exactly $1$, and
{{sec:9-practical-example}} prints it as `+1.0000` rather than `0.9997` —
a floating-point-exact result, since all three reduce to comparing the same
dot products.

## 8. Implementation

```python {tier=A name=metric-equivalence}
"""Cosine, inner product, and Euclidean: identical, then wildly different.

Two corpora over the same 2,000 directions. In the first, every document has
norm 1. In the second, norms are lognormal -- a realistic spread for
un-normalised encoder outputs, where norm tracks length and token frequency.

We rank all documents for 300 queries under all three scorers and measure
agreement, both at the top (which document wins) and over the full ranking
(Kendall tau).
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr

rng = np.random.default_rng(5)
N_Q, N_D, DIM = 300, 2000, 64

directions = rng.normal(size=(N_D, DIM))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
norms = rng.lognormal(mean=0.0, sigma=0.6, size=(N_D, 1))

queries = rng.normal(size=(N_Q, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)


def score_all(Q, D):
    """Return the three score matrices of eq:three-scorers, queries x documents."""
    ip = Q @ D.T
    cos = ip / (np.linalg.norm(Q, axis=1, keepdims=True)
                * np.linalg.norm(D, axis=1)[None, :])
    # -||q-d||^2 via eq:l2-ip-identity; negated so that larger is better.
    neg_l2 = -(np.sum(Q ** 2, axis=1)[:, None]
               + np.sum(D ** 2, axis=1)[None, :] - 2.0 * ip)
    return ip, cos, neg_l2


def top1_agreement(a, b):
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def mean_tau(a, b, n=40):
    """Kendall tau over the FULL ranking, averaged over n queries."""
    return float(np.mean([kendalltau(a[i], b[i]).statistic for i in range(n)]))


for label, docs in [("documents normalised (all norms = 1)", directions),
                    ("documents NOT normalised (lognormal norms)",
                     directions * norms)]:
    ip, cos, neg_l2 = score_all(queries, docs)
    doc_norm = np.linalg.norm(docs, axis=1)

    print(f"\n{label}")
    print(f"  {'pair':<22}{'top-1 agree':>13}{'Kendall tau':>14}")
    for name, a, b in [("inner prod vs cosine", ip, cos),
                       ("inner prod vs -L2", ip, neg_l2),
                       ("cosine vs -L2", cos, neg_l2)]:
        print(f"  {name:<22}{top1_agreement(a, b):>13.4f}{mean_tau(a, b):>+14.4f}")

    # Who does the inner product actually return?
    winners = np.argmax(ip, axis=1)
    times_returned = np.bincount(winners, minlength=N_D)
    rho = spearmanr(doc_norm, times_returned).statistic
    print(f"  spearman(||d||, times returned by inner product) = {rho:+.4f}")
    print(f"  mean ||d|| of inner-product winners = {doc_norm[winners].mean():.3f}"
          f"   (corpus mean {doc_norm.mean():.3f})")

print("""
The first block is eq:rank-equivalence. Not approximately equal -- Kendall tau is
exactly +1.0000 and top-1 agreement exactly 1.0000, because with every norm equal
all three scorers are strictly monotone functions of the same dot product.

The second block is what happens when that assumption is dropped, and it is
worse than "somewhat different". Inner product and negative-L2 pick the same best
document essentially NEVER. That is not noise; it is eq:magnitude-bias. Inner
product rewards long vectors and L2 penalises them, so on a corpus whose only
difference is norm, the two scorers rank in nearly opposite directions.

Note the Kendall tau of inner product against cosine stays high while its top-1
agreement collapses. The bulk ordering is broadly preserved and the TOP of the
ranking is not -- which is the only part anyone retrieves. Rank correlation over
a full list is the wrong diagnostic for a retrieval system.

The last two lines are the practical damage. Inner-product winners have a mean
norm several times the corpus mean: the index has quietly become a popularity
ranker over whatever property drove the norm -- in a real encoder, document
length and token frequency.""")
```

```python {tier=A name=high-dimensional-pathologies}
"""Concentration and hubness track INTRINSIC dimension, not stored dimension.

Two regimes at each ambient dimension, same corpus size:

  iid gaussian  -- every coordinate contributes independent variance, so
                   intrinsic dimension = ambient dimension. This is exactly the
                   setting of Beyer's hypothesis (eq:beyer-condition).
  intrinsic 8   -- an 8-dimensional Gaussian pushed through a random linear map
                   into the ambient space. The coordinates are heavily
                   dependent; intrinsic dimension stays 8 however wide we store it.

Measured: relative contrast (eq:relative-contrast) and the k-occurrence
distribution (eq:k-occurrence, eq:hubness-skew).
"""
import numpy as np
from scipy.stats import skew

rng = np.random.default_rng(3)
N, K, LATENT = 5000, 10, 8
N_QUERY = 200
AMBIENT = [2, 8, 32, 128, 512]


def pairwise_sq(A, B):
    """Squared Euclidean distances via eq:l2-ip-identity, clipped for stability."""
    return np.maximum(np.sum(A ** 2, axis=1)[:, None]
                      + np.sum(B ** 2, axis=1)[None, :] - 2.0 * A @ B.T, 0.0)


def contrast(X):
    """Mean of (Dmax - Dmin)/Dmin over held-out queries -- eq:relative-contrast."""
    q = X[rng.choice(len(X), N_QUERY, replace=False)]
    D = np.sqrt(pairwise_sq(q, X))
    # Each query is a member of X, so its own zero distance must be removed.
    np.put_along_axis(D, np.argmin(D, axis=1)[:, None], np.inf, axis=1)
    d_min = D.min(axis=1)
    d_max = np.where(np.isinf(D), -np.inf, D).max(axis=1)
    return float(np.mean((d_max - d_min) / d_min))


def hubness(X):
    """Skew of the k-occurrence distribution, its max, and the unreachable share."""
    D = pairwise_sq(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, K, axis=1)[:, :K]
    counts = np.bincount(nn.ravel(), minlength=len(X))
    return float(skew(counts)), int(counts.max()), float(np.mean(counts == 0))


latent = rng.normal(size=(N, LATENT))
summary = {}

print(f"{'ambient':>8} {'regime':<15}{'contrast':>10}{'k-occ skew':>12}"
      f"{'max k-occ':>11}{'never (%)':>11}")
print("-" * 68)
for dim in AMBIENT:
    corpora = {
        "iid gaussian": rng.normal(size=(N, dim)),
        "intrinsic 8": latent @ (rng.normal(size=(LATENT, dim)) / np.sqrt(LATENT)),
    }
    for name, X in corpora.items():
        c = contrast(X)
        s, mx, never = hubness(X)
        summary[(dim, name)] = (c, s, mx, never)
        print(f"{dim:>8} {name:<15}{c:>10.3f}{s:>12.3f}{mx:>11d}{100 * never:>11.1f}")

c_lo = summary[(AMBIENT[0], "iid gaussian")][0]
c_hi, s_hi, mx_hi, nv_hi = summary[(AMBIENT[-1], "iid gaussian")]
c_int, s_int, _, nv_int = summary[(AMBIENT[-1], "intrinsic 8")]

print(f"""
Read the CONTRAST column down the iid rows. It collapses from {c_lo:.0f} at
ambient {AMBIENT[0]} to {c_hi:.2f} at ambient {AMBIENT[-1]} -- that is
eq:concentration-limit happening, and at the bottom the nearest and farthest
points differ by a fraction of the nearest distance. Beyer's
theorem is not an abstraction; this is it.

Now read the intrinsic-8 rows. Same corpus size, same ambient dimensions, same
distance function -- and the contrast is flat and healthy at every width, still
{c_int:.2f} at ambient {AMBIENT[-1]}. The data is 8-dimensional and behaves
8-dimensionally whether we store it in 32
coordinates or 512. That is eq:intrinsic-concentration, and it is why a
768-dimensional embedding index works.

The hubness columns tell the same story in a form you can see in production. In
the iid regime at ambient {AMBIENT[-1]}, the k-occurrence skew is {s_hi:.1f}, one
point is returned for {mx_hi} queries when the expected count is {K}, and
{100 * nv_hi:.0f}% of the corpus is returned for NOTHING -- unreachable by any
query, no matter how the index is built. In the intrinsic-8 regime at the same
width the skew is {s_int:.2f} and {100 * nv_int:.1f}% is unreachable.

Both pathologies are real, both are consequences of eq:beyer-condition holding,
and both are governed by intrinsic dimension. The practical reading: if your
retrieval degrades as you scale the corpus, measure contrast and k-occurrence
skew before you touch the index -- they distinguish "the geometry is bad" from
"the index is losing recall", and those need opposite fixes.""")
```

## 9. Practical Example

Two listings, two results, and they answer the two questions this chapter exists
to settle.

**The metric question.** With documents normalised, the three scorers give
Kendall tau of exactly $+1.0000$ and top-1 agreement of exactly $1.0000$. This is
{{eq:rank-equivalence}}, and the exactness is the point: on the unit sphere,
choosing a metric is not a choice.

Un-normalise and inner product and negative-L2 select the same top document
**0.00%** of the time — not rarely, never — because {{eq:magnitude-bias}} makes
their preferences opposite. Inner product against cosine keeps a high Kendall tau
(0.815) while its top-1 agreement falls to 4.3%, which is a lesson in its own
right: **rank correlation over a full ranked list is the wrong diagnostic for a
retrieval system**, because it is dominated by the tail nobody retrieves.

Then the damage: the mean norm of the inner product's winners is 4.46 against a
corpus mean of 1.20 — a factor of 3.7. An inner-product index over un-normalised vectors has
silently become a ranker over whatever property drove the norm.

**The dimension question.** The second listing sweeps ambient dimension from 2 to
512 in two regimes. In the i.i.d. regime, relative contrast falls from roughly
295 to 0.22 — {{eq:concentration-limit}} in action, and at that point "nearest"
means almost nothing. In the intrinsic-8 regime, the contrast stays between about
5.6 and 11 across every ambient dimension, with no downward trend.

Hubness is the more striking half. At ambient 512 the i.i.d. corpus has a
$k$-occurrence skew above 10, a single point returned for 664 queries against an
expectation of 10, and **34% of the corpus unreachable by any query
whatsoever**. The intrinsic-8 corpus at the same width shows a skew of 0.19 and
1.6% unreachable — and, notably, that 1.6% does not grow with ambient dimension
either.

> **IMPORTANT:** Both diagnostics are cheap and neither requires labels. If
> retrieval quality degrades as a corpus grows, measure relative contrast and
> $k$-occurrence skew before touching the index. They separate "the embedding
> geometry is bad" from "the index is losing recall", and those two diagnoses
> have opposite remedies — re-train or re-embed in the first case, retune
> {{ch:emb-ann}}'s parameters in the second.

## 10. Production Considerations

**Normalise at write time, once.** Storing normalised vectors makes
{{eq:rank-equivalence}} hold by construction and eliminates the class of bug
where the ingestion path normalises and the query path does not. It also makes
the index's configured metric irrelevant, which is one fewer thing to get wrong
during a migration.

**Store the metric with the index.** A metric change is not a config change; it
is a different retrieval system. Systems that let you switch metrics on an
existing index are offering you a footgun.

**Measure the corpus's background similarity at build time and store it.** It is
one pass, it gives you {{eq:score-margin}}'s denominator, and it is the only
thing that makes a score interpretable later. Recompute on rebuild — it drifts as
the corpus grows.

**Watch $k$-occurrence in production.** Log which documents are returned. A
handful of documents dominating the returns is hubness, and the fixes are
geometric — centring per side ({{ch:emb-what-they-are}}), a better embedding
model, or a mutual-proximity correction — not index tuning.

**Float precision matters at scale.** Cosines between near-duplicates in float16
collide, and the ordering among the top few results becomes arbitrary. Score in
float32 even when storing in float16, which is what mature indexes do by default
and immature ones do not.

## 11. Common Mistakes

**Assuming the index's metric matches the model's training objective.** A model
trained with cosine similarity, served through an L2 index over un-normalised
vectors, is a different system from the one that was evaluated. This is a
top-three source of "the model was better in the notebook".

**Thresholding raw scores.** {{eq:score-margin}}: without the background there is
no information in the number.

**Comparing scores across models or across corpora.** Same reason.

**Citing the curse of dimensionality to justify reducing dimensions.** The
theorem is about intrinsic dimension ({{eq:intrinsic-concentration}}); projecting
768 coordinates down to 128 does not reduce it and will not fix concentration.
It reduces cost, which is a good reason on its own.

**Using Kendall tau or Spearman to validate a retrieval change.** The listing
shows a case with $\tau = 0.82$ and 4% top-1 agreement. Evaluate at $k$.

**Treating hubness as a ranking bug.** It is geometry, it will not be fixed by
re-ranking, and it is diagnosable in one pass with {{eq:k-occurrence}}.

## 12. Failure Modes

**Silent metric mismatch.** Ingestion normalises, query does not, or vice versa.
Results remain plausible and get consistently worse. Detect by asserting unit
norm on both paths at the API boundary.

**Norm drift on a growing corpus.** If documents are ingested un-normalised and
the length distribution shifts — a new source of long documents — an
inner-product index's effective ranking shifts with it, with no code change.

**Hub domination.** A handful of documents in every result set. Grows worse as
the corpus grows, because {{eq:hubness-skew}}'s skew increases with $N$ as well
as with dimension.

**Unreachable documents.** The other tail of {{eq:k-occurrence}}: documents that
are in no query's neighbour list at all. They are indexed, they are searchable in
principle, and they will never be returned. This is invisible to every recall
metric computed over queries you have.

**Concentration on a badly trained embedding.** If a model collapses
({{ch:emb-what-they-are}}), contrast goes to zero and every query returns an
arbitrary $k$. The index is working perfectly.

**Augmentation blow-up.** {{eq:mips-augmentation}} with one outlier norm makes
$M$ huge, so the padded coordinate dominates every distance and the geometry is
destroyed. Clip norms before augmenting.

## 13. Alternatives

**Mahalanobis / learned metrics.** Rank under $ (a-b)\T \mat{M} (a-b) $ for a
learned positive-definite $\mat{M}$. Equivalent to a linear map followed by
Euclidean distance, so in practice it is folded into the encoder — which is what
the final projection layer of an embedding model already is.

**Manhattan and other $\ell_p$.** Occasionally better for sparse or count data;
for dense learned embeddings the differences are small and the tooling support
is worse.

**Hamming distance over binary codes.** Extremely fast, badly lossy, and useful
as a first stage in a cascade. {{ch:emb-ann}}.

**Jaccard / set overlap.** The right metric when the representation is a set
rather than a vector, which is what learned sparse retrieval produces
({{ch:emb-hybrid}}).

**Learned similarity.** Give up on a fixed function and score with a model —
which is the cross-encoder ({{ch:emb-reranking}}), and cannot be indexed.

## 14. Evaluation

**Metric agreement.** Before shipping, confirm your index's metric and your
model's training objective induce the same ranking, by scoring a sample both
ways and reporting top-$k$ agreement — not rank correlation.

**Relative contrast.** {{eq:relative-contrast}} on a sample of queries. Low
contrast is a model problem and no index will fix it.

**$k$-occurrence distribution.** {{eq:k-occurrence}}: report the skew, the max,
and the fraction of the corpus with $N_k = 0$. That last number is the one
nobody measures and it is directly interpretable — it is the share of your
corpus that cannot be retrieved.

**Background similarity.** The mean pairwise cosine, which calibrates every score
you will ever look at.

**What none of these measure.** Whether the documents retrieved are *relevant*.
All four are geometry diagnostics: they tell you whether the space can support
retrieval, not whether it retrieves the right things. {{ch:emb-models}} handles
the second question and it needs labels.

## 15. Advanced Concepts

**Concentration is about the query too.** {{eq:beyer-condition}} assumes the
query is drawn from the same distribution as the data. A query from a different
distribution — which is what an out-of-domain query is — sits outside the
corpus's manifold, and its distances to every document concentrate faster than an
in-domain query's. This is a geometric account of why out-of-domain retrieval
degrades, and it predicts that the degradation shows up as *falling contrast*
before it shows up as falling relevance.

**Hubness and concentration are distinct.** They are correlated in i.i.d. data
and separable in general. Concentration is about the *spread* of distances from
one query; hubness is about the *asymmetry* of the neighbour relation across
queries. A space can have healthy contrast and severe hubness — it happens when
density is very uneven — and the fixes differ.

**Mutual proximity as a hubness correction.** Replace the raw similarity with the
probability that each point is in the other's neighbourhood, estimated from the
empirical distance distributions. It symmetrises the neighbour relation and
demonstrably reduces hub domination, at the cost of a corpus-wide statistic per
point and a query-time correction — which is why it is rare in production
despite working.

**The augmentation trick generalises.** {{eq:mips-augmentation}} is one instance
of a pattern: convert a non-metric ranking problem into a metric one by adding
coordinates that absorb the non-metric part. The same idea handles bias terms and
per-document priors, and it is why "my scorer has an extra term" is usually not a
reason to abandon a standard index.

**Why quantization interacts with the metric.** {{cite:jegou2011pq}}'s error
analysis assumes Euclidean geometry; {{cite:guo2020scann}}'s point is that under
an inner product the error's *direction* matters and not only its magnitude. So
the metric choice made in this chapter propagates all the way into how
{{ch:emb-ann}} compresses vectors — one more reason it is not a late-binding
detail.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}} defined the embedding by its dot product; this chapter
works out what that dot product does. {{ch:math-norms}}'s Cauchy–Schwarz
inequality is what bounds cosine to $[-1,1]$, and {{eq:l2-ip-identity}} is the
polarisation identity from the same chapter. {{ch:nlp-similarity}} used cosine
without justifying it; {{eq:rank-equivalence}} is the justification and its
condition. {{ch:ml-metrics}}'s rank correlations appear here as a *cautionary*
tool rather than a recommended one. {{ch:ml-knn-nb}}'s treatment of
high-dimensional volume is the geometric intuition behind
{{eq:concentration-limit}}; this chapter supplies the theorem and, more
importantly, its hypotheses.

## 17. Exercises

1. Prove {{eq:rank-equivalence}} from {{eq:l2-ip-identity}}, stating exactly
   where the constant-norm assumption is used.
2. Verify {{eq:mips-augmentation}} is exact: show $\lVert\tilde d\rVert = M$ for
   all $d$ and $\tilde q\T\tilde d = q\T d$.
3. Construct a two-document, one-query example in $\R^2$ where inner product and
   Euclidean distance disagree, using {{eq:disagreement-condition}}.
4. In `metric-equivalence`, replace the lognormal norms with a distribution
   whose spread you can dial. At what spread does top-1 agreement between inner
   product and cosine first fall below 50%?
5. In `high-dimensional-pathologies`, sweep `LATENT` over $\{2, 8, 32, 128\}$ at
   fixed ambient 512. Confirm that contrast tracks `LATENT` and not the ambient
   dimension, and find the value at which the two regimes coincide.
6. Add a regime in which the intrinsic-8 data is given a large constant offset.
   Predict the effect on hubness first, then measure it, and explain the result
   in terms of proximity to the data mean.
7. Compute the fraction of the corpus with $N_k = 0$ for a real embedding set
   you have access to. Is it larger than you expected?
8. A colleague reports that a new embedding model has Spearman correlation 0.95
   with the old one's scores and concludes the change is safe. Write the
   objection.

## 18. Interview Questions

1. When do cosine and inner product give the same ranking?
2. Why does normalising change results at all, if it only rescales?
3. Your index uses L2 and your model was trained with cosine. Is that a problem?
4. What is the curse of dimensionality, precisely, and why does 768-dimensional
   search work anyway?
5. What is relative contrast and what does a low value tell you?
6. Some documents are returned for every query. Diagnose.
7. Is inner product a metric? Why does the answer matter for HNSW?
8. What does a cosine of 0.85 tell you?
9. How would you check that a corpus has documents that can never be retrieved?
10. Would reducing from 768 to 128 dimensions help with the curse of
    dimensionality?

## 19. Research Questions

1. Is there an intrinsic-dimension estimator stable enough that its value
   *predicts* retrieval degradation, rather than merely correlating with it?
2. Can hubness be designed out at training time — a regulariser on the
   $k$-occurrence distribution — rather than corrected at query time?
3. {{eq:score-margin}} makes scores interpretable per corpus. Is there a training
   objective that makes them interpretable *across* corpora without a
   calibration set?
4. Concentration analyses assume a fixed corpus. What is the right theory for a
   corpus that grows, where the nearest-neighbour distance shrinks as
   $N^{-1/k_{\text{int}}}$ while the query distribution is static?
5. Is there a principled account of *which* directions in an embedding space
   carry relevance information, so that a metric could weight them — a learned
   Mahalanobis metric that is not just another linear layer?

## 20. Chapter Summary

Cosine, inner product, and negative Euclidean distance induce **exactly** the
same ranking when documents share a common norm ({{eq:rank-equivalence}}), and
that is the only condition required. Off the sphere they diverge violently:
inner product rewards magnitude and Euclidean distance punishes it
({{eq:magnitude-bias}}), so their top results agree essentially never, and an
inner-product index over un-normalised vectors is a covert ranker over document
length. Normalise at write time and the whole question disappears.

Inner product is not a metric, so maximum inner product search is a different
problem from nearest-neighbour search. The exact reduction
({{eq:mips-augmentation}}) costs one dimension; the alternative is a native MIPS
index, which changes how quantization must work ({{ch:emb-ann}}).

{{cite:beyer1999nn}}'s concentration theorem is correct and its usual conclusion
is not. The hypothesis {{eq:beyer-condition}} is about the distance
distribution's relative variance, which is governed by **intrinsic** dimension
({{eq:intrinsic-concentration}}). Learned embeddings are low-rank, so they sit
outside the theorem's regime — demonstrated by holding intrinsic dimension at 8
while sweeping ambient dimension to 512 and watching relative contrast stay
between 5.6 and 11 while the i.i.d. control falls from 295 to 0.22.

Hubness ({{eq:k-occurrence}}) is the second pathology and the one visible in
production: a few documents returned for everything, and — in the i.i.d. control
at 512 dimensions — 34% of the corpus returned for nothing at all, against 1.6%
in the intrinsic-8 corpus of identical width. It tracks
intrinsic dimension too.

Finally, the score is a rank, not a measurement. Only the margin over the
corpus's background ({{eq:score-margin}}) carries information, which is why
thresholds do not transfer across models, corpora, or time.

## 21. Further Reading

{{cite:beyer1999nn}} for the concentration theorem — read the hypotheses in
Section 3 rather than the abstract, since the hypotheses are the whole content.
{{cite:radovanovic2010hubs}} for hubness, still the definitive treatment;
Sections 3 and 5 give the mechanism and the connection to the data mean.
{{cite:guo2020scann}} for why the metric choice reaches into quantization.
{{cite:malkov2020hnsw}} for what a graph index assumes about the metric.
{{cite:jegou2011pq}} for the Euclidean error analysis that
{{cite:guo2020scann}} argues against.
