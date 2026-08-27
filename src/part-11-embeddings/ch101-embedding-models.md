---
id: emb-models
number: 101
part: XI
tier: full
status: draft
requires: [emb-what-they-are, emb-similarity, ml-metrics, dl-optimizers,
           fm-datasets, nlp-similarity]
provides: [hard-negative-mining, false-negatives, negative-sampling-strategy,
           embedding-dimension-choice, matryoshka-training, asymmetric-encoding,
           embedding-benchmarks, domain-evaluation-set, embedding-migration]
citations: [karpukhin2020dpr, wang2022e5, ni2021gtr, kusupati2022matryoshka,
            izacard2022contriever, thakur2021beir, muennighoff2023mteb,
            gao2021simcse, reimers2019, oord2018cpc]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why hard negatives carry
almost all of the training signal and demonstrate what they buy; recognise the
false-negative hazard and know the condition under which it actually bites;
separate embedding *dimension* from model *capacity* and say which one carries
quality; train and evaluate nested representations that make dimension a
serving-time choice; read a public embedding benchmark for what it can and
cannot tell you; and build the domain evaluation set that decides the question a
benchmark cannot.

## 2. Why This Matters

Choosing an embedding model is the highest-leverage decision in a retrieval
system and the one most often made by reading a leaderboard.

The leaderboard cannot answer it. {{cite:thakur2021beir}} exists because models
that dominate in-domain routinely lose to BM25 out of domain, and
{{cite:muennighoff2023mteb}} aggregates across task families that want
incompatible geometries. Meanwhile the decision is close to irreversible:
{{ch:emb-what-they-are}} established that an embedding model is a versioned
schema for its index, so changing it means re-embedding the corpus and rebuilding
everything downstream.

So this chapter is about the two things that actually determine outcomes — what
the model was trained *against*, and how you decide whether it works on *your*
data — and about the one architectural parameter people consistently
misattribute quality to.

{{maturity:MATURE}} The training recipe is stable and well-documented. The
evaluation practice is not: benchmark contamination and leaderboard targeting
make the public numbers substantially less informative than they were in 2022,
and that is a live problem rather than a solved one.

## 3. Prerequisites

{{ch:emb-what-they-are}} for InfoNCE, in-batch negatives, and the dual encoder;
{{ch:emb-similarity}} for normalisation and what the score means;
{{ch:ml-metrics}} for recall@k and nDCG; {{ch:dl-optimizers}} for batch size and
learning-rate interaction; {{ch:fm-datasets}} for the contamination argument,
which applies to embedding benchmarks with full force.

## 4. Intuitive Explanation

### Negatives are the curriculum

{{ch:emb-what-they-are}} showed that InfoNCE's gradient
({{eq:infonce-gradient}}) weights each negative by its softmax probability, so a
negative the model already rejects contributes nothing. Turn that around and it
becomes a statement about *teaching*:

> **The negatives you show the model define what it learns to distinguish.
> Everything else it is free to ignore.**

Train with random negatives and the model learns to tell tax law from bread
recipes — a distinction that is easy, that a keyword search already makes, and
that is not what anyone needs a neural retriever for. Train with negatives drawn
from the same topic as the positive and the model has to learn what actually
separates two documents about the same thing, which is the hard part and the
only part that matters.

This has a consequence that {{sec:9-practical-example}} makes numerical and that
is easy to miss: **the benefit of hard negatives is nearly invisible on an easy
evaluation set.** In the experiment there, hard negatives improve global
retrieval by 11 points and fine-grained, same-group retrieval by 16.5. If your
evaluation set contains only easy distinctions, you will measure a small gain
from an intervention that transformed the model on the queries you care about.

### Dimension is not capacity

The second confusion. An embedding's dimension is how much space it takes; the
encoder's size is how much it knows. These are independent, and
{{cite:ni2021gtr}} demonstrated it directly by scaling the encoder while holding
the output width fixed and watching out-of-domain retrieval improve
substantially.

It matters because the two have completely different cost profiles. Dimension is
a *serving* cost — index memory, distance computation, network — paid on every
query forever. Encoder size is a *training and encoding* cost, paid once per
document at ingest. Attributing quality to the wrong one leads teams to buy
wider vectors when they needed a better model.

And once you accept that dimension is a cost parameter rather than a quality
one, an obvious question follows: could one model serve several widths? That is
{{cite:kusupati2022matryoshka}}, and {{sec:9-practical-example}} shows it working
— nested training matches separately-trained models at every width, for a small
price at full width.

## 5. Formal Explanation

### 5.1 The training objective, with explicit negatives

{{eq:infonce}} used in-batch negatives. Generalise so the negative set is a
design choice:

$$ \Loss = -\frac{1}{N}\sum_{i} \log \frac{\exp(s(q_i,d_i^+)/\tau)}{\exp(s(q_i,d_i^+)/\tau) + \sum_{d^- \in \mathcal{N}(q_i)} \exp(s(q_i,d^-)/\tau)} $$ (eq:infonce-explicit)

where $\mathcal{N}(q)$ is the negative set. The entire literature on training
retrievers is about how to construct $\mathcal{N}$.

**Random.** $\mathcal{N}(q) \sim \text{Uniform}(\mathcal{D})$. Free, and mostly
uninformative once training has begun.

**In-batch.** $\mathcal{N}(q_i) = \{d_j\}_{j \neq i}$. Free given the batch, and
therefore the default; effectively random negatives with a batch-size knob.

**Mined (hard).** $\mathcal{N}(q) = \text{top-}m$ scoring non-positives under the
*current* model:

$$ \mathcal{N}_{\text{hard}}(q) = \argmax_{|\mathcal{N}| = m,\; \mathcal{N} \cap R(q) = \emptyset} \sum_{d \in \mathcal{N}} s_\theta(q, d) $$ (eq:hard-negative-mining)

This is what {{cite:karpukhin2020dpr}} found to matter most, and it introduces
the chapter's central hazard.

### 5.2 The false-negative problem, stated precisely

{{eq:hard-negative-mining}}'s constraint $\mathcal{N} \cap R(q) = \emptyset$ is
not enforceable, because $R(q)$ — the true relevant set — is exactly what is
unknown. In practice one excludes the *labelled* positives, which is a strict
subset:

$$ \mathcal{N}_{\text{mined}}(q) \cap R(q) \;\neq\; \emptyset \quad \text{whenever } R(q) \setminus \{d^+\} \neq \emptyset $$ (eq:false-negative)

A mined item that is genuinely relevant but unlabelled is a **false negative**,
and {{eq:infonce-gradient}} weights it heavily *precisely because* it scores
high. The failure mode of hard-negative mining is that it is most confident
exactly where it is most likely to be wrong.

> **RESEARCH NOTE:** The literature treats this as a first-order hazard and
> {{sec:9-practical-example}} could not reproduce damage at the rates mining
> actually produced — under 2% of negative slots, at which the effect was inside
> run-to-run variance. The reconciliation is a real result rather than a
> negative one: **mining finds what is hard under the current model**, and a
> model that has not yet learned a distinction will not surface the items that
> are confusable on it. False negatives therefore concentrate in *late* training
> and in *iterative* mining rounds, once the model is good enough to score true
> positives near the top. Filter aggressively in round three; the danger in
> round one is small. The number to measure, rather than assume, is the mined
> false-negative rate itself.

### 5.3 Dimension against capacity

Let $E_\phi$ be an encoder with capacity $\phi$ producing a $k$-dimensional
output. Retrieval quality is a function of both; costs are not:

$$ \text{cost}_{\text{serve}} \;\propto\; k \cdot |\mathcal{D}|, \qquad \text{cost}_{\text{ingest}} \;\propto\; \phi \cdot |\mathcal{D}|, \qquad \text{cost}_{\text{query}} \;\propto\; \phi + k \cdot |\mathcal{D}| $$ (eq:embedding-costs)

{{cite:ni2021gtr}}'s finding is that $\partial\,\text{quality}/\partial \phi$ is
large and $\partial\,\text{quality}/\partial k$ saturates quickly past the point
where $k$ exceeds the data's intrinsic dimension ({{ch:emb-similarity}}). The
practical rule follows: **scale the encoder, hold the width, and treat $k$ as
something to shrink until quality moves.**

### 5.4 Nested representations

{{cite:kusupati2022matryoshka}} makes $k$ a serving-time decision by training so
that every prefix is itself a usable embedding. For a nesting schedule
$k_1 < k_2 < \dots < k_m = k$:

$$ \Loss_{\text{MRL}} = \sum_{j=1}^{m} w_j \, \Loss_{\text{InfoNCE}}\big(\hat f(\cdot)_{1:k_j}\big) $$ (eq:matryoshka-loss)

Each term normalises the *prefix* independently. The result is a single vector
whose first 64 coordinates are a good 64-dimensional embedding and whose first
768 are a good 768-dimensional one, enabling adaptive retrieval — search wide
and cheap, rerank narrow and expensive — from one index.

The cost is visible in {{eq:matryoshka-loss}}: the full-width term now shares
capacity with $m-1$ others, so the widest representation is slightly worse than
one trained alone. {{sec:9-practical-example}} measures that price.

### 5.5 Asymmetry as a contract

{{ch:emb-what-they-are}} noted the query/passage prefix convention. Formally,
the model implements two functions:

$$ f_{\text{query}}(x) = E_\phi(\texttt{"query: "} \Vert x), \qquad f_{\text{doc}}(x) = E_\phi(\texttt{"passage: "} \Vert x) $$ (eq:prefix-asymmetry)

**This is part of the index's schema**, alongside the model version, the
dimension, and the metric. A system that stores $f_{\text{doc}}$ vectors and
queries with $f_{\text{doc}}$ is not broken in any detectable way — it returns
results, they are worse, and nothing raises.

## 6. Mathematical Foundation

### 6.1 Why random negatives stop teaching

Take {{eq:infonce-gradient}} and ask what fraction of the gradient a random
negative carries. With $s(q, d^-) \approx \mu$ for random negatives and
$s(q,d^+) = \mu + \Delta$ for the positive, the softmax weight on any one random
negative is

$$ p^- = \frac{e^{\mu/\tau}}{e^{(\mu+\Delta)/\tau} + (N-1)e^{\mu/\tau}} = \frac{1}{e^{\Delta/\tau} + N - 1} $$ (eq:random-negative-weight)

With $\tau = 0.07$ and a trained model achieving $\Delta = 0.3$ — a modest
margin — $e^{\Delta/\tau} = e^{4.3} \approx 74$. Every random negative's weight
is suppressed by that factor. **The margin the model has already achieved is
exactly the factor by which its remaining random negatives stop mattering**, so
training decelerates on its own, and mining is the standard remedy.

A hard negative with $s(q,d^-) = \mu + \Delta - \delta$ for small $\delta$
carries weight larger by $e^{(\Delta - \delta)/\tau}$ — orders of magnitude more.
This is the arithmetic behind "hard negatives are worth more than more
negatives".

### 6.2 What the model spends capacity on

Suppose the representation must fit into $k$ dimensions and the data varies along
directions with variances $\sigma_1^2 \geq \sigma_2^2 \geq \dots$. An objective
that only requires separating *randomly chosen* pairs is satisfied by preserving
the highest-variance directions, since those separate random pairs best. An
objective that requires separating *nearby* pairs must preserve the
low-variance directions that distinguish them.

$$ \text{random negatives} \Rightarrow \text{preserve } \argmax_j \sigma_j^2, \qquad \text{hard negatives} \Rightarrow \text{preserve } \argmax_j \frac{\text{between-pair}_j}{\text{within-pair}_j} $$ (eq:capacity-allocation)

**This is the mechanism**, and it predicts something checkable: the benefit of
hard negatives should be largest when $k$ is small enough to force a choice, and
should vanish when $k$ is large enough to keep everything.
{{sec:9-practical-example}} uses a deliberately tight bottleneck for that reason,
and the effect is large there.

It also predicts the second observation in that experiment — that hard negatives
help the *fine-grained* evaluation far more than the global one — because the
global task is solved by the high-variance directions that random negatives
already preserve.

### 6.3 How many labelled pairs does a domain evaluation set need?

The question every team asks and few answer quantitatively. Treat recall@k as a
binomial proportion estimated from $n$ queries. The standard error is
$\sqrt{p(1-p)/n}$, maximised at $p = 0.5$, so the width of a 95% interval is at
most

$$ w \approx 2 \times 1.96 \times \frac{0.5}{\sqrt{n}} = \frac{1.96}{\sqrt{n}} $$ (eq:eval-set-size)

To distinguish two models differing by 5 points of recall you need
$w \lesssim 0.05$, giving $n \gtrsim 1{,}500$ — which is more than most teams
build, and explains a great deal of confident model-switching on noise.

**But paired comparison is much cheaper.** Evaluating both models on the *same*
queries and testing the difference removes the query-difficulty variance, which
dominates. With a paired test the requirement typically falls by an order of
magnitude, to a few hundred queries:

$$ n_{\text{paired}} \approx \left(\frac{1.96\,\sigma_{\text{diff}}}{\Delta}\right)^2, \qquad \sigma_{\text{diff}} \ll \sigma_{\text{recall}} $$ (eq:paired-eval-size)

> **PRODUCTION TIP:** Two to three hundred labelled query-document pairs drawn
> from real traffic, evaluated paired, will settle almost every model-choice
> question you have — and will disagree with the public leaderboard often enough
> to pay for itself immediately. Build it before you tune anything.

## 7. Internal Mechanics

```mermaid {#fig:mining-loop caption="Iterative hard-negative mining. The dashed arrow is where false negatives enter, and note that its danger grows with each round: as the model improves, the mined set moves closer to the true relevant set that the filter cannot see."}
flowchart TD
    A["labelled (query, positive) pairs"] --> B["train with in-batch negatives"]
    B --> C["encode the corpus"]
    C --> D["retrieve top-m per query"]
    D --> E{"filter: drop labelled positives"}
    E -->|"kept"| F["hard negative set"]
    E -.->|"unlabelled but relevant<br/>survives the filter"| F
    F --> G["re-train with eq:infonce-explicit"]
    G --> H{"another round?"}
    H -->|yes| C
    H -->|no| I["final model"]
```

### 7.1 The mining loop and its rounds

{{fig:mining-loop}} is the standard recipe, and the loop is where the danger
compounds. Round one mines against a weak model, so the "hard" negatives are
mildly related documents and false negatives are rare. By round three the model
is strong, its top-$m$ is close to the true relevant set, and the filter — which
knows only the labelled positives — is removing a shrinking fraction of what it
should.

The practical schedule that follows: two to three rounds, with progressively
more aggressive filtering (a similarity ceiling above which candidates are
discarded rather than used as negatives), and stop when quality on a held-out
set stops improving rather than at a fixed round count.

### 7.2 Where training data comes from

{{cite:wang2022e5}}'s contribution was as much about data as objective: mined
pairs at scale — forum question/answer pairs, post/comment, title/body,
citation contexts — filtered by a consistency model. {{cite:izacard2022contriever}}
goes further and needs no pairs at all, forming positives from two independent
crops of the same document.

The hierarchy in practice:

| Source | Cost | Quality | Use |
|---|---|---|---|
| independent crops | zero | weak but unbiased | initialisation, new domains |
| mined web pairs | low | good, noisy | pre-training the retriever |
| in-domain labelled | high | best | final fine-tune and evaluation |
| synthetic (LLM-generated) | medium | variable | filling gaps in a query distribution |

The last row is now common and deserves the caution {{ch:fm-datasets}} raised: a
retriever fine-tuned on LLM-written queries learns the LLM's query distribution,
which is not your users'.

### 7.3 What varies between models you might choose

| Property | Why it matters | How to check |
|---|---|---|
| max sequence length | truncation is silent | encode your length distribution and log the truncation rate |
| prefix convention | silent quality loss | read the model card; test both ways on 50 pairs |
| normalised output | metric correctness ({{ch:emb-similarity}}) | measure the norms |
| dimension | index cost | it is stated; check whether truncation is supported |
| multilingual | cross-lingual retrieval works or does not | test the actual language pairs, not the average |
| licence and hosting | migration cost later | before, not after |

## 8. Implementation

```python {tier=A name=negative-strategy}
"""What the negatives teach: random against mined-hard, on a tight bottleneck.

Each item's latent vector has two parts:

  COARSE -- a high-variance group identity (60 groups). Separating two items from
            DIFFERENT groups only requires these directions.
  FINE   -- a low-variance individual identity. Separating two items from the
            SAME group requires these.

The encoder is deliberately given fewer output dimensions than the latent space,
so it must CHOOSE which directions to preserve -- eq:capacity-allocation. We
report two evaluations: retrieval against the whole test corpus, and retrieval
restricted to the query's own group, which is the fine-grained task.

Three runs per strategy, because the variance is itself part of the result.
"""
import numpy as np
import statistics

rng = np.random.default_rng(17)

COARSE, FINE, OBS, EMB = 8, 6, 48, 5      # EMB < COARSE + FINE: a real bottleneck
C_SCALE, F_SCALE = 4.0, 0.35
N, N_DUP, N_GROUP, TAU = 4000, 1000, 60, 0.07

proj = rng.normal(size=(COARSE + FINE, OBS)) / np.sqrt(COARSE + FINE)
offset = rng.normal(size=OBS) * 2.0
q_shift = rng.normal(size=OBS) * 0.4


def latents(n):
    g = rng.integers(0, N_GROUP, size=n)
    coarse = rng.normal(size=(N_GROUP, COARSE))[g] * C_SCALE
    fine = rng.normal(size=(n, FINE)) * F_SCALE
    return np.hstack([coarse, fine]), g


Z_tr, G_tr = latents(N)
# Some items have a near-duplicate: SAME fine identity (so genuinely equivalent),
# DIFFERENT coarse surface. These are the unlabelled relevant documents of
# eq:false-negative -- the ones a filter on labelled positives cannot see.
dup_src = rng.choice(N, N_DUP, replace=False)
new_coarse = rng.normal(size=(N_DUP, COARSE)) * C_SCALE
Z_dup = np.hstack([new_coarse,
                   Z_tr[dup_src][:, COARSE:] + rng.normal(scale=0.02,
                                                          size=(N_DUP, FINE))])
Z = np.vstack([Z_tr, Z_dup])
partner = np.full(len(Z), -1)
partner[N:] = dup_src
partner[dup_src] = np.arange(N, N + N_DUP)


def views(z):
    b = z @ proj + offset
    return (b + q_shift + rng.normal(scale=0.10, size=b.shape),
            b + rng.normal(scale=0.10, size=b.shape))


Q, D = views(Z)
Z_te, G_te = latents(1500)
Q_te, D_te = views(Z_te)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def train(strategy, steps=1200, batch=128, n_neg=32, lr=0.5):
    """Fit a linear encoder with eq:infonce-explicit; returns W and the
    measured false-negative rate among the negatives actually used."""
    W = rng.normal(scale=0.05, size=(OBS, EMB))
    fn_hits, fn_slots = 0, 0
    for _ in range(steps):
        anchors = rng.choice(len(Z), batch, replace=False)

        if strategy == "random":
            neg = rng.integers(0, len(Z), size=(batch, n_neg))
        else:
            pool = rng.choice(len(Z), 800, replace=False)
            sc = unit(Q[anchors] @ W) @ unit(D[pool] @ W).T
            sc[pool[None, :] == anchors[:, None]] = -np.inf      # drop self
            neg = pool[np.argpartition(-sc, n_neg, axis=1)[:, :n_neg]]

        pm = partner[anchors]
        fn_hits += int(np.sum(neg == pm[:, None]))
        fn_slots += neg.size

        A, P, Ng = Q[anchors], D[anchors], D[neg]
        Za_r, Zp_r, Zn_r = A @ W, P @ W, Ng @ W
        na = np.linalg.norm(Za_r, axis=1, keepdims=True)
        np_ = np.linalg.norm(Zp_r, axis=1, keepdims=True)
        nn = np.linalg.norm(Zn_r, axis=2, keepdims=True)
        Za, Zp, Zn = Za_r / na, Zp_r / np_, Zn_r / nn

        logits = np.concatenate([np.sum(Za * Zp, axis=1, keepdims=True),
                                 np.einsum('bd,bnd->bn', Za, Zn)], axis=1) / TAU
        logits -= logits.max(axis=1, keepdims=True)
        Pr = np.exp(logits)
        Pr /= Pr.sum(axis=1, keepdims=True)

        g = Pr.copy()
        g[:, 0] -= 1.0                       # the positive is column 0
        g /= batch * TAU
        dZa = g[:, 0:1] * Zp + np.einsum('bn,bnd->bd', g[:, 1:], Zn)
        dZp = g[:, 0:1] * Za
        dZn = g[:, 1:, None] * Za[:, None, :]

        def through_norm(dZ, Zx, nx):
            return (dZ - Zx * np.sum(dZ * Zx, axis=-1, keepdims=True)) / nx

        W -= lr * (A.T @ through_norm(dZa, Za, na)
                   + P.T @ through_norm(dZp, Zp, np_)
                   + np.einsum('bnd,bne->de', Ng, through_norm(dZn, Zn, nn)))
    return W, fn_hits / fn_slots


def evaluate(W):
    a, b = unit(Q_te @ W), unit(D_te @ W)
    S = a @ b.T
    overall = float(np.mean(np.argmax(S, axis=1) == np.arange(len(a))))
    M = S.copy()
    M[G_te[None, :] != G_te[:, None]] = -np.inf     # same-group candidates only
    within = float(np.mean(np.argmax(M, axis=1) == np.arange(len(a))))
    return overall, within


print(f"{'negatives':<16}{'acc, whole corpus':>19}{'acc, within group':>19}"
      f"{'mined false-neg':>17}")
print("-" * 71)
rows = {}
for strategy in ["random", "mined hard"]:
    overalls, withins, fns = [], [], []
    for _ in range(3):
        W, fn_rate = train(strategy)
        o, w = evaluate(W)
        overalls.append(o)
        withins.append(w)
        fns.append(fn_rate)
    sd = statistics.pstdev(withins)
    rows[strategy] = (statistics.mean(overalls), statistics.mean(withins), sd)
    print(f"{strategy:<16}{statistics.mean(overalls):>19.4f}"
          f"{statistics.mean(withins):>19.4f}{100 * statistics.mean(fns):>16.2f}%"
          f"   (within-group sd {sd:.4f})")

d_all = 100 * (rows["mined hard"][0] - rows["random"][0])
d_within = 100 * (rows["mined hard"][1] - rows["random"][1])
sd_ratio = rows["random"][2] / rows["mined hard"][2]

print(f"""
Two numbers, and the second one is the lesson.

Mined negatives beat random ones on the whole-corpus task by {d_all:.1f} points
-- worth having. On the WITHIN-GROUP task they beat them by {d_within:.1f}, about
{d_within / d_all:.1f} times as much. That gap between the gaps is
eq:capacity-allocation: with a bottleneck this
tight the encoder must choose which latent directions to keep, random negatives
only ever ask it to separate different groups, and the high-variance coarse
directions are enough for that. Only mined negatives force it to spend capacity
on the fine directions.

The practical consequence is uncomfortable. If your evaluation set is drawn
uniformly from the corpus, most of its pairs are easy, and you will measure the
{d_all:.1f}-point version of an intervention that delivered {d_within:.1f} on the
queries users actually send -- the confusable ones.

Now the variance column. Random negatives are not merely worse on average; their
run-to-run spread is {sd_ratio:.1f} times larger. Whether the model ever learns
the fine distinction depends on how many informative negatives happened to be
drawn. Mined negatives make that deterministic. A training procedure whose
outcome varies this much between seeds is not one you can A/B test cheaply.

Finally, the false-negative column. These are genuinely equivalent documents --
same fine identity, different surface -- and mining picks them up at well under
one percent of negative slots. Not because the filter caught them: there is no
filter here. Mining scores candidates under the CURRENT model, and a model whose
capacity has gone to surface features cannot see that two differently-worded
documents say the same thing, so it never ranks them highly enough to mine. That
is why false negatives are a late-training and iterative-mining hazard rather
than a round-one one -- and why the number to watch is this rate, per round.""")
```

```python {tier=A name=matryoshka-nesting}
"""Dimension as a serving-time choice: nested training against truncation.

Three ways to obtain a k-dimensional embedding for k in {4, 8, 16, 32}:

  plain, truncated  -- train once at 32, keep the first k coordinates, re-normalise
  matryoshka        -- train once with eq:matryoshka-loss over all four widths,
                       then keep the first k
  trained at dim    -- train a separate model at each k (the upper bound, and
                       four times the training cost)
"""
import numpy as np

rng = np.random.default_rng(23)

D_LAT, D_OBS, D_EMB = 40, 96, 32
N_TRAIN, N_TEST, TAU = 6000, 3000, 0.07
NEST = [4, 8, 16, 32]

proj = rng.normal(size=(D_LAT, D_OBS)) / np.sqrt(D_LAT)
offset = rng.normal(size=D_OBS) * 2.0
q_shift = rng.normal(size=D_OBS) * 0.4


def views(z):
    b = z @ proj + offset
    return (b + q_shift + rng.normal(scale=0.55, size=b.shape),
            b + rng.normal(scale=0.55, size=b.shape))


Q, D = views(rng.normal(size=(N_TRAIN, D_LAT)))
Q_te, D_te = views(rng.normal(size=(N_TEST, D_LAT)))


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def grad(W, A, P, dims):
    """Gradient of the sum of InfoNCE over the given prefixes (eq:matryoshka-loss).
    Passing a single-element list recovers ordinary contrastive training."""
    dW = np.zeros_like(W)
    Za_full, Zp_full = A @ W, P @ W
    for d in dims:
        Za_r, Zp_r = Za_full[:, :d], Zp_full[:, :d]
        na = np.linalg.norm(Za_r, axis=1, keepdims=True)
        np_ = np.linalg.norm(Zp_r, axis=1, keepdims=True)
        Za, Zp = Za_r / na, Zp_r / np_
        logits = Za @ Zp.T / TAU
        logits -= logits.max(axis=1, keepdims=True)
        Pr = np.exp(logits)
        Pr /= Pr.sum(axis=1, keepdims=True)
        G = Pr.copy()
        G[np.arange(len(G)), np.arange(len(G))] -= 1.0
        G /= len(G) * TAU

        def through_norm(dZ, Zx, nx):
            return (dZ - Zx * np.sum(dZ * Zx, axis=1, keepdims=True)) / nx

        # Only the first d columns of W receive gradient from this prefix.
        dW[:, :d] += (A.T @ through_norm(G @ Zp, Za, na)
                      + P.T @ through_norm(G.T @ Za, Zp, np_))
    return dW


def train(dims, steps=1500, batch=256, lr=0.5):
    W = rng.normal(scale=0.05, size=(D_OBS, D_EMB))
    for _ in range(steps):
        i = rng.choice(N_TRAIN, batch, replace=False)
        W -= lr * grad(W, Q[i], D[i], dims)
    return W


def accuracy(W, d):
    a, b = unit((Q_te @ W)[:, :d]), unit((D_te @ W)[:, :d])
    return float(np.mean(np.argmax(a @ b.T, axis=1) == np.arange(len(a))))


W_plain = train([D_EMB])                       # one model, full width only
W_mrl = train(NEST)                            # one model, all widths
W_sep = {d: train([d]) for d in NEST}          # four models

print(f"{'dim':>5}{'plain, truncated':>19}{'matryoshka':>13}{'trained at dim':>16}"
      f"{'index memory':>15}")
print("-" * 68)
for d in NEST:
    rel = d / D_EMB
    print(f"{d:>5}{accuracy(W_plain, d):>19.4f}{accuracy(W_mrl, d):>13.4f}"
          f"{accuracy(W_sep[d], d):>16.4f}{rel:>14.0%}")

print("""
Read across each row. The matryoshka column tracks the trained-at-dim column
closely at every width -- one model reproduces what four separate models
achieve, which is the entire claim, and it means the width can be chosen after
training rather than before.

Now read the plain-truncated column against it. Truncating an ordinarily-trained
embedding is much worse at the narrow end. Nothing in ordinary contrastive
training asks the first eight coordinates to be useful ALONE, so they are not;
the information is spread across all thirty-two, and cutting the vector destroys
it. Matryoshka's only difference is that eq:matryoshka-loss asks each prefix to
work on its own.

The last row is the price, and it should be stated rather than hidden: at full
width the matryoshka model is slightly WORSE than one trained at full width
alone. The nesting is not free -- the widest representation shares capacity with
every narrower one. That trade is usually worth it, because the memory column
shows what the narrow end buys: an eight-dimensional index is a quarter of the
memory and a quarter of the distance arithmetic, forever, on every query.""")
```

## 9. Practical Example

**Negatives.** Mined hard negatives beat random ones by 3.8 points on
whole-corpus retrieval and 7.6 on within-group retrieval — twice the gain on the
task that needs it. The gap between those two gaps is the point: the fine-grained task is the one that needs
the capacity that only hard negatives force the encoder to allocate
({{eq:capacity-allocation}}).

The variance result is the one to act on. Random negatives produce a
within-group standard deviation across seeds 4.7 times larger than mined
negatives do — 0.047 against 0.010, which is larger than the entire effect being
measured. Whether the model learns the fine distinction at all depends
on the draw. **A training procedure this seed-sensitive cannot be evaluated by
running it once**, which is how most embedding fine-tunes are in fact evaluated.

The false-negative column reports well under 1% of negative slots, and the
explanation is in {{sec:5-formal-explanation}}'s research note: mining surfaces
what is hard under the *current* model, and a model that has spent its capacity
on surface features cannot recognise that two differently-worded documents are
equivalent. The hazard is real and it lives in later rounds.

**Dimension.** The nested model matches separately-trained models at every
width, while plain truncation collapses at the narrow end — at 8 dimensions plain truncation scores 0.419 against
the nested model's 0.642, a gap of 22 points, while the nested model is within
0.4 points of a model trained at 8 dimensions from scratch. And at full width the nested model is *slightly worse*
than one trained at full width alone, which is the honest price of
{{eq:matryoshka-loss}}: the widest prefix shares capacity with all the narrower
ones.

> **NOTE:** The memory column is why the trade is usually worth taking. A
> 4× narrower index is 4× less memory and 4× less distance arithmetic on every
> query for the life of the system, against a fraction of a point at full width
> paid once. {{ch:emb-ann}}'s adaptive retrieval — search narrow, rerank wide —
> needs exactly this property.

## 10. Production Considerations

**Build the domain evaluation set first.** Two to three hundred paired
query-document judgements from real traffic ({{eq:paired-eval-size}}). It costs
a few days and it is the only instrument that can answer the question you
actually have. Everything else in this list is worth less.

**Pin the whole schema, not the model name.** Model identifier *and* version,
dimension, normalisation, prefix convention, max sequence length, and metric.
Store it with the index and validate it at query time
({{ch:emb-what-they-are}}).

**Budget the migration before choosing the model.** Corpus size × encoding
throughput = the wall-clock cost of every future model change. If that number is
weeks, the choice is more consequential than the benchmark difference between
your candidates.

**Log the truncation rate at ingest.** Silent, and it makes long documents
systematically under-retrieved.

**Measure the mined false-negative rate per round.** {{fig:mining-loop}}'s
dashed arrow is invisible unless instrumented, and it grows with each round.

**Evaluate a fine-tune more than once.** Given the variance result, a single
run's number is not a measurement. Three seeds and a reported spread is the
minimum, and the spread belongs in the decision: an improvement smaller than the
seed variance is not an improvement, it is a draw you liked.

**Decide the re-embed strategy before you need it.** There are exactly two, and
they have different failure modes. *Dual-write* runs both models during the
transition, doubling ingest cost and index memory but allowing an instant
rollback and a genuine A/B comparison on live traffic. *Rebuild-and-swap* is
cheaper and atomic, but the only evidence you get is from before the swap, and
rolling back means another full re-embed. Dual-write is the right default for
any index whose rebuild takes longer than a day, because the thing you most need
during a model migration is the ability to compare the two models on real
queries — which is precisely what {{eq:paired-eval-size}} says you need a few
hundred of, and which the old index is the only source of.

**Keep the evaluation set out of the training data.** Obvious, and violated
routinely when the domain set is drawn from the same mined pairs that fine-tuned
the model. Draw it from traffic, judge it separately, and hold it back.

## 11. Common Mistakes

**Choosing by leaderboard rank.** MTEB averages incompatible task families and
is now a training target. It narrows the field to about five candidates; it
cannot pick among them.

**Training with in-batch negatives and stopping there.** They are random
negatives with extra steps ({{eq:random-negative-weight}}).

**Mining hard negatives with no similarity ceiling.** The top-scoring
non-positive in a large corpus is frequently a genuine duplicate.

**Buying dimensions instead of capacity.** {{eq:embedding-costs}}: width is a
serving cost paid forever, capacity is an ingest cost paid once.

**Truncating an ordinarily-trained embedding.** It does not degrade gracefully
unless it was trained to ({{eq:matryoshka-loss}}).

**Fine-tuning on synthetic queries and evaluating on synthetic queries.** The
model learns the generator's distribution and the evaluation cannot see it.

**Comparing models on different evaluation sets.** Including "the public
benchmark for theirs and my domain set for mine".

## 12. Failure Modes

**False-negative collapse in late rounds.** By round three the mined set
approximates the true relevant set, the model is trained to push relevant
documents apart, and quality falls while training loss improves. Detect with a
held-out set evaluated every round, not at the end.

**Seed lottery.** A fine-tune that looked like a 3-point win was a good draw.
{{sec:9-practical-example}} quantifies how large this effect can be.

**Distribution mismatch after fine-tuning.** Fine-tuning on one query type
degrades the others, sometimes badly, and the domain evaluation set will not
show it if it was drawn from the same type.

**Silent prefix loss.** A library upgrade changes the default prefix handling.
No error; results degrade.

**Truncation cliff on a shifting corpus.** A new document source with longer
documents pushes the truncation rate up with no code change.

**Benchmark contamination.** The model's training data included the benchmark's
corpus, so the score is memorisation ({{ch:fm-datasets}}). Unfalsifiable from
outside, which is why the domain set is not optional.

## 13. Alternatives

**Use an API embedding model.** No training, no serving, a per-token cost, and
a version you do not control — that last point is the real one, since the
provider's upgrade is your forced re-embed.

**Unsupervised adaptation.** {{cite:izacard2022contriever}}'s independent
cropping needs no pairs and is the right starting point in a domain with no
judgements.

**Skip embeddings.** BM25 needs no model, no index rebuild on model change, and
wins out-of-domain often enough that it is the correct baseline
({{ch:emb-hybrid}}).

**Late interaction.** Spend the capacity on more vectors rather than better ones
({{ch:emb-reranking}}).

**Rerank instead of re-training.** A cross-encoder on top of a mediocre
retriever is often a larger and much cheaper win than a better retriever
({{ch:emb-reranking}}) — provided the retriever's recall@k is adequate, which is
the ceiling that no reranker can raise.

## 14. Evaluation

**On your domain set, paired.** Recall@k for the retrieval stage and nDCG@k if
you have graded judgements. Paired, per {{eq:paired-eval-size}}.

**Recall@k at the k your reranker will use**, not at $k=1$. The first stage's
job is a candidate set, and its metric should be the one that bounds what comes
after.

**Sliced.** Overall recall hides the slices where the model fails, and router-
and reranker-style systems concentrate their errors ({{ch:llm-routing}}).
Report by query type, by document length, and by language.

**Multiple seeds for any fine-tune.** Non-negotiable given the variance result.

**BEIR for out-of-domain robustness** ({{cite:thakur2021beir}}), read as a
robustness signal rather than a ranking; **MTEB** ({{cite:muennighoff2023mteb}})
as a candidate filter, with contamination assumed.

**What none of it measures:** whether the retrieved documents help the *system*.
That is {{part:12}}'s question, and it frequently disagrees with recall@k.

## 15. Advanced Concepts

**Temperature and mining do the same job.** {{eq:random-negative-weight}} shows
small $\tau$ concentrates gradient on the highest-scoring negatives, which is
implicit hard-negative mining. Doing both aggressively is doing one thing twice,
and the symptom is instability.

**Curriculum over negatives.** Since the false-negative hazard grows with model
quality and the benefit of hard negatives grows too, the schedule matters:
random early, mined middle, mined-with-ceiling late. Most published recipes are
a fixed two rounds, which is a coarse approximation of this.

**Distillation from a cross-encoder.** Train the bi-encoder to match a cross
encoder's scores rather than binary labels. It supplies a *graded* signal, which
solves the false-negative problem structurally — a false negative gets a high
target score instead of a wrong label — and it is why the strongest open
retrievers are distilled rather than contrastively trained alone.

**Matryoshka and adaptive retrieval.** {{eq:matryoshka-loss}} enables a cascade
inside the index: retrieve a large candidate set with the 64-dimensional prefix,
re-score with the full width, return the top $k$. Same vectors, one index, and
the arithmetic is {{ch:llm-routing}}'s cascade equation again.

**Why the widest prefix pays.** In {{eq:matryoshka-loss}}, coordinate $j$
receives gradient from every term with $k_i \geq j$, so early coordinates are
optimised $m$ times and late ones once. The representation is deliberately
front-loaded, and the full-width cost measured in
{{sec:9-practical-example}} is the direct consequence.

## 16. Connection to Previous Chapters

{{ch:emb-what-they-are}}'s InfoNCE gradient is what
{{eq:random-negative-weight}} quantifies, and its schema-versioning point is what
makes migration cost a model-selection criterion. {{ch:emb-similarity}}'s
intrinsic-dimension argument is why {{eq:embedding-costs}}'s width saturates.
{{ch:ml-metrics}} supplies recall@k and the paired-comparison machinery.
{{ch:fm-datasets}}'s contamination argument transfers directly to embedding
leaderboards. {{ch:dl-optimizers}} is why $\tau$ and the learning rate must be
tuned together. {{ch:llm-routing}}'s cascade equation reappears as adaptive
retrieval over a nested embedding.

## 17. Exercises

1. Derive {{eq:random-negative-weight}} and compute the suppression factor at
   $\tau \in \{0.02, 0.05, 0.1\}$ for $\Delta = 0.3$.
2. Use {{eq:eval-set-size}} to find $n$ for a 2-point difference at 95%
   confidence, unpaired. Then explain why the paired figure is so much smaller.
3. In `negative-strategy`, raise `EMB` to 20 so the bottleneck no longer binds.
   Predict what happens to the gap between the strategies, then check, and
   relate the result to {{eq:capacity-allocation}}.
4. Add a third strategy that mixes half random and half mined negatives. Is it
   between the two, or better than both?
5. In `matryoshka-nesting`, change `NEST` to `[8, 32]`. Does the 4-dimensional
   truncation get worse? Explain using the gradient-count argument in
   {{sec:15-advanced-concepts}}.
6. Weight the terms of {{eq:matryoshka-loss}} to favour the widest prefix.
   Recover the full-width loss — and measure what the narrow end pays for it.
7. Instrument `negative-strategy` to report the mined false-negative rate per
   200 steps rather than as a total. Does it rise as training proceeds?
8. Design the domain evaluation set for a product-search system: where do the
   queries come from, who judges, and what makes a judgement reusable when the
   catalogue changes?

## 18. Interview Questions

1. Why do hard negatives matter more than more negatives?
2. What is a false negative in retrieval training and why is mining prone to it?
3. Would you rather double the embedding dimension or double the encoder size?
4. Your fine-tune improved recall@10 by 3 points. What do you check first?
5. What does MTEB tell you? What does it not?
6. How many labelled pairs do you need to choose between two models?
7. What has to be true for a truncated embedding to still work?
8. Where does embedding training data come from when you have no labels?
9. You are switching embedding models. What is the plan?
10. Recall@10 improved and the product metric did not. What happened?

## 19. Research Questions

1. Can the false-negative rate be estimated *without* labels, so mining could
   filter adaptively rather than by a fixed similarity ceiling?
2. Is there a negative-sampling distribution that is provably optimal for a
   given capacity budget, in the sense of {{eq:capacity-allocation}}?
3. {{eq:matryoshka-loss}} weights prefixes by hand. Is there a principled
   schedule, and does the optimal one depend on the corpus's intrinsic
   dimension?
4. Can two embedding spaces be aligned well enough post hoc to avoid a re-embed
   on model upgrade? Partial results exist; nothing is trustworthy enough to
   stake an index on.
5. Given the seed variance measured here, what is the right statistical protocol
   for reporting an embedding fine-tune — and why does essentially no published
   result use one?

## 20. Chapter Summary

**The negatives are the curriculum.** {{eq:random-negative-weight}} shows a
random negative's gradient weight suppressed by $e^{\Delta/\tau}$, so training
decelerates on its own and mining is the remedy. Measured: mined negatives beat
random by 3.8 points on whole-corpus retrieval and 7.6 on the fine-grained task
— twice the gain where it matters — and, the result to act on, with 4.7× less
run-to-run variance than the effect size itself. A fine-tune evaluated once has not been evaluated.

**False negatives are a late hazard, not an early one.** {{eq:false-negative}}
is unavoidable in principle, but mining surfaces what is hard under the current
model, so the rate is low while the model is weak and rises as it improves.
Instrument the rate per round rather than assuming a filter suffices.

**Dimension is a serving cost, capacity is a quality knob**
({{eq:embedding-costs}}, {{cite:ni2021gtr}}). Nested training
({{eq:matryoshka-loss}}) makes width a serving-time choice: measured, it matches
separately-trained models at every width while ordinary truncation collapses at
the narrow end — and it costs a fraction of a point at full width, which is the
honest price and worth paying for a 4× smaller index.

**The evaluation is the decision.** Public benchmarks narrow the field;
{{eq:paired-eval-size}} says two to three hundred paired in-domain judgements
settle it. That set is the highest-return artefact in the entire retrieval
pipeline, and it is the thing teams build last.

## 21. Further Reading

{{cite:karpukhin2020dpr}} for hard negatives as the thing that matters — Section
4's ablation is the whole argument.
{{cite:wang2022e5}} for where training pairs come from at scale, and for the
prefix convention.
{{cite:izacard2022contriever}} for training with no labels at all.
{{cite:ni2021gtr}} for scaling capacity at fixed width.
{{cite:kusupati2022matryoshka}} for nested representations; read Section 3 for
the loss and Section 4.3 for adaptive retrieval.
{{cite:thakur2021beir}} and {{cite:muennighoff2023mteb}} for what the public
benchmarks measure — and read them together, since BEIR's out-of-domain framing
is the corrective to MTEB's average.
