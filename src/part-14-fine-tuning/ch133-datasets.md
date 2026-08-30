---
id: ft-datasets
number: 133
part: XIV
tier: full
status: draft
requires: [ft-sft, ft-when, fm-instruction-tuning, mle-splits]
provides: [coverage-over-count, stratified-selection, skill-taxonomy,
           macro-versus-aggregate, group-aware-splitting, decontamination-overshoot,
           provenance-key]
citations: [zhou2023lima, lee2022dedup, gunasekar2023, wei2022flan, ouyang2022]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why "we need more data" is
usually the wrong diagnosis for a heavy-tailed task, and what to do instead;
allocate a fixed example budget across skills rather than by natural frequency,
and state honestly what that trades away; recognise when an evaluation metric has
inherited the training selection's bias and cannot see the problem; split a
dataset by **provenance group** rather than at random; and explain why
threshold-based decontamination fails in **two directions at once**.

## 2. Why This Matters

{{ch:ft-sft}} covered how to train on a dataset. This chapter is about the dataset
being wrong, which is the more common failure and the harder one to see.

Two measurements make the case, and both are experiments rather than assertions.

**First, allocation beats collection.** {{sec:9-practical-example}} trains a real
classifier over forty skills with a Zipf frequency profile. At a budget of 1,000
examples, random selection scores **0.486** and stratified selection scores
**0.639** — the same budget, the same pool, the same training code, **31% better
from selection alone.** And **1,000 stratified examples beat 30,000 random ones**
(0.639 against 0.630). Thirty times the data, and it loses.

**But it is a trade, and the honest version says so.** On a test set drawn the way
production traffic arrives, the ranking **reverses**: random scores **0.729** at
1,000 where stratified scores 0.649. Stratification buys the tail by selling the
head. **Treating it as strictly better is how curated datasets end up worse than
what they replaced.**

**Second, the metric can inherit the bias.** If training data is sampled from
production traffic and the eval set is too, the evaluation agrees with the
training selection about what matters, and **cannot see what that selection
missed.**

**Third — and this is the chapter's sharpest result — a random split lies.** With
clustered data at a 50% duplication rate, a random split reports **0.811** where a
group-aware split reports **0.768**. Nothing about the model changed between those
numbers.

**And deduplication by similarity threshold does not fix it.** It *overshoots*:
from **+0.031** before to **−0.031** after, while **50% of the surviving test set
is still leaked.** Push the threshold harder and **92% of the test set is
discarded and the leaked share rises to 77%.**

{{maturity:ESTABLISHED}} Group-aware splitting, deduplication.
{{maturity:MATURE}} Curation over collection. {{maturity:EMERGING}} Treating the
skill taxonomy as a design artefact rather than a post-hoc analysis.

## 3. Prerequisites

{{ch:ft-sft}} for what the training loop does with these examples;
{{ch:ft-when}} for whether to fine-tune at all; {{ch:fm-instruction-tuning}} for
instruction formats and template diversity, which this chapter assumes rather
than repeats; {{ch:mle-splits}} for the splitting machinery this chapter shows is
usually applied wrongly.

**This chapter is about *selection and splitting*.** Generating data is
{{ch:ft-synthetic}}; preference data is {{ch:ft-preference}}.

## 4. Intuitive Explanation

### "More data" is the wrong diagnosis for a heavy tail

Real task distributions are heavy-tailed: a handful of request types dominate and
a long tail of rare ones carries most of the *variety*. Sample more from that
distribution and you get **more of what you already had**.

{{sec:9-practical-example}} makes it concrete. Random selection at a budget of
1,000 leaves the rarest skill with **3 examples**; at 30,000 it reaches **133**.
The tail grows at the tail's rate, which is far slower than the budget, while the
head saturated long ago and each additional head example buys nothing.

**Stratified selection spends the budget on the axis that matters.** At 1,000 the
rarest skill has **25** examples rather than 3, and macro accuracy is **0.639**
against **0.486**.

> **This is what "curate rather than collect" means concretely.** Not that quality
> is a virtue — that a fixed budget allocated across skills buys capability the
> same budget allocated by frequency does not.

### The trade, stated honestly

Stratification is not free, and the chapter would be dishonest without this.
Deliberately under-sampling the head costs head accuracy. On the
natural-distribution test set, random wins at **every** budget in the table:
0.729 against 0.649 at 1,000, and 0.798 against 0.760 at the top.

**Both rankings are correct.** They disagree because they ask different questions:
*how well do you serve a typical request* versus *how well do you serve every kind
of request*. **The selection strategy is the answer to that question, whether or
not anyone made the choice deliberately.**

### Why nobody notices

Here is the part that makes this hard to catch in practice.

If training data is sampled from production traffic, and the eval set is sampled
from production traffic, then **the eval set is dominated by the same head the
training set is.** The evaluation cannot see what the selection missed, because it
made the same selection.

The model looks fine. It fails on exactly the requests that were rare in both —
and rare is not the same property as unimportant.

**Report macro-averaged metrics alongside aggregate ones**, or the tail is
structurally invisible.

### Random splits lie about clustered data

Now the second half, which is a different failure with a similar shape.

Fine-tuning data is rarely independent examples. It arrives in **clusters**: the
same question asked three ways, one template instantiated per customer, a document
and its summary, a ticket and its resolution.

Split that pool at random and cluster members land on both sides. The model has
**seen the answer**, and the held-out score measures recall while reporting
generalisation.

{{sec:9-practical-example}} measures the gap: **+0.042** at a 50% duplication
rate, against a control of **−0.005** when there are no clusters at all.

**This is the most common way a fine-tuning result turns out to be fictional**, and
it is invisible from the inside — healthy loss curve, good held-out score, and a
number that is about something other than what it says.

### Why the standard remedy fails in both directions

The usual response is to drop test examples too similar to a training example. It
fails twice.

**It misses what leaks.** Contamination is about a shared *answer*, not surface
distance. The same question from a different angle, a different document about the
same fact — far apart in input space, identical in what they give away. At a
threshold that discards a third of the test set, **50% of the survivors are still
leaked.**

**And it worsens with force.** Push the threshold to discard **92%** and the
leaked share **rises to 77%**, because an aggressive distance filter
preferentially removes surface duplicates and preferentially *keeps* semantic
ones.

**Meanwhile the reported score overshoots the truth** — from **+0.031** to
**−0.031** — because removing test examples near training examples does not remove
a *random* sample: it removes the ones the model finds easy.

> **Two biases in opposite directions, partly cancelling.** That is worse than
> either alone, because the cancellation is accidental and the result lands
> nowhere in particular.

### The fix is upstream and it is cheap

Split by **group**, using the provenance you had before the examples became
vectors: source document, customer, template, URL, ticket.

**That information is free at collection time and largely unrecoverable
afterwards.** Which is the whole practical lesson: the dataset decision that
matters most costs nothing and has to be made first.

## 5. Formal Explanation

### 5.1 Coverage under a heavy tail

For skill frequencies $p_1 \ge \dots \ge p_K$ and budget $N$, random selection
gives skill $k$ about $N p_k$ examples. If $m$ examples are needed to learn a
skill, the budget for full coverage is

$$ N_{\text{cover}} = \frac{m}{p_K} $$ (eq:coverage-saturates)

For Zipf $p_k \propto k^{-s}$, $p_K \approx K^{-s}/H$, so
$N_{\text{cover}} \propto m K^{s}$ — **the budget must grow with the tail's
depth, while the useful information is already saturated in the head.**

Stratified selection instead sets $n_k = \min(N/K, |{\rm pool}_k|)$, reaching
coverage at $N \approx mK$ — **a factor $K^{s-1}$ smaller.**

### 5.2 Macro accuracy is bounded by the tail

$$ \mathcal{A}_{\text{macro}} = \frac{1}{K}\sum_k a_k, \qquad \mathcal{A}_{\text{agg}} = \sum_k p_k a_k $$ (eq:macro-versus-aggregate)

with $a_k$ increasing in $n_k$. Under random selection $n_k \propto N p_k$, so

$$ \frac{\partial \mathcal{A}_{\text{macro}}}{\partial N} \propto \frac{1}{K}\sum_k p_k a_k'(n_k) $$

which is dominated by the **small** $p_k$ terms where $a_k'$ is largest — and
those grow slowest. **{{eq:macro-versus-aggregate}} is why the two metrics rank
selection strategies in opposite directions**, and why reporting only the second
hides the problem.

### 5.3 The metric inherits the selection

If the eval set is drawn from the same distribution as the training pool,

$$ \mathbb{E}[\mathcal{A}_{\text{eval}}] = \sum_k p_k a_k = \mathcal{A}_{\text{agg}} $$ (eq:metric-inherits-bias)

so the evaluation *weights each skill by exactly the frequency that caused it to be
under-trained*. **{{eq:metric-inherits-bias}} is a structural blindness, not a
sampling error** — more eval data does not fix it.

### 5.4 Leakage inflates

Let $\lambda$ be the fraction of test examples sharing a cluster with a training
example, $a_{\text{mem}}$ the accuracy on those (near 1 for a model with capacity
to memorise), and $a_{\text{gen}}$ true generalisation. Then

$$ \mathcal{A}_{\text{random}} = \lambda\, a_{\text{mem}} + (1-\lambda)\, a_{\text{gen}}, \qquad \mathcal{A}_{\text{group}} = a_{\text{gen}} $$

$$ \text{inflation} = \lambda\,(a_{\text{mem}} - a_{\text{gen}}) $$ (eq:leakage-inflates)

**{{eq:leakage-inflates}} is linear in the duplication rate and in the
memorisation gap**, which is why it grows with model capacity — a larger model
memorises more, so the *same dataset flaw* produces a *larger* lie.

### 5.5 Why distance thresholds miss

Decontamination keeps test example $i$ when $d(x_i, \mathcal{D}_{\text{train}}) >
\tau$. That is a statement about **inputs**. Contamination is a statement about
**answers**:

$$ \text{leaked}(i) \iff \exists j \in \mathcal{D}_{\text{train}} : g(i) = g(j) $$

and the two coincide only when cluster membership implies input proximity:

$$ g(i) = g(j) \;\not\Rightarrow\; d(x_i, x_j) < \tau $$ (eq:distance-misses-semantics)

**{{eq:distance-misses-semantics}} has no threshold that fixes it**, because the
relation being tested is not the relation being measured.

### 5.6 And the filter is not neutral

The surviving set is conditioned on being far from training data:

$$ \mathbb{E}\!\left[a \mid d > \tau\right] < \mathbb{E}[a] $$ (eq:decontamination-overshoot)

since $d$ correlates negatively with difficulty. **So filtering biases the score
down while leakage biases it up**, and {{eq:decontamination-overshoot}} plus
{{eq:leakage-inflates}} give a number with **two errors of unknown relative
size**.

### 5.7 The only clean estimator

$$ \text{split by } g, \quad \mathcal{D}_{\text{train}} \cap_g \mathcal{D}_{\text{test}} = \varnothing $$ (eq:group-split)

**{{eq:group-split}} is unbiased and requires no threshold, no embedding, and no
judgement** — only that $g$ was recorded.

> **IMPORTANT:** $g$ must be the *provenance* key, not a cluster id computed from
> the data afterwards. Clustering post hoc reintroduces
> {{eq:distance-misses-semantics}}: it can only group what looks similar, which is
> the thing that was never the problem.

## 6. Mathematical Foundation

### 6.1 The budget comparison, worked

With $K = 40$, $s = 1.1$, $m = 50$: $H = \sum k^{-1.1} \approx 3.9$, so
$p_{40} \approx 40^{-1.1}/3.9 \approx 0.0044$ and

$$ N_{\text{cover}}^{\text{random}} = \frac{50}{0.0044} \approx 11{,}400, \qquad N_{\text{cover}}^{\text{strat}} = 50 \times 40 = 2{,}000 $$

The measurement agrees: stratified reaches full coverage at 3,000 and random
reaches it between 10,000 and 30,000 — a **factor of roughly 5**, which is
$K^{s-1} = 40^{0.1} \approx 1.45$ times the ratio of head to tail frequency.

### 6.2 Why leakage grows with capacity

From {{eq:leakage-inflates}}, $\partial(\text{inflation})/\partial a_{\text{mem}}
= \lambda > 0$. A model with more capacity has $a_{\text{mem}} \to 1$ while
$a_{\text{gen}}$ is bounded by the task, so

$$ \lim_{\text{capacity} \to \infty} \text{inflation} = \lambda\,(1 - a_{\text{gen}}) $$ (eq:leakage-ceiling)

**{{eq:leakage-ceiling}} means the same dataset gives a larger overstatement to a
larger model** — so a leaked benchmark makes scaling look better than it is, in
addition to making every individual result wrong.

### 6.3 The overshoot is not a bug in the threshold

Write the filtered estimate as

$$ \hat{a}(\tau) = \mathbb{E}\!\left[a \mid d > \tau\right] = \underbrace{\lambda(\tau) a_{\text{mem}} + (1 - \lambda(\tau)) a_{\text{gen}}(\tau)}_{\text{leak up}},\quad a_{\text{gen}}(\tau) \searrow \tau $$

Both terms move as $\tau$ rises: $\lambda(\tau)$ falls (good) and
$a_{\text{gen}}(\tau)$ falls (bad). **There is no reason for
$\hat{a}(\tau) = a_{\text{gen}}$ at any $\tau$**, and the measured crossing
between $\tau = 0$ and $\tau = 0.3$ is a coincidence of this dataset, not a
calibration point.

> **MATH NOTE:** This is why "we decontaminated at threshold $\tau$" is not a
> validity claim. It states that a procedure was run, not that the resulting
> number estimates anything. {{eq:group-split}} is the claim worth making.

## 7. Internal Mechanics

```mermaid {#fig:dataset-pipeline caption="The two decisions that determine whether a fine-tuning number means anything, both made before any training. Allocation decides what the model can do; the provenance key decides whether you can measure it. Both are free at collection time and expensive or impossible to add later."}
flowchart TB
    SRC["raw source<br/>(traffic, tickets, docs)"] --> KEY{{"record the<br/>PROVENANCE KEY"}}
    KEY --> TAX{{"write the<br/>SKILL TAXONOMY"}}
    TAX --> ALLOC["allocate budget<br/>across skills"]
    ALLOC -->|"stratified"| TRAIN["training set"]
    KEY --> SPLIT["split by group<br/>eq:group-split"]
    SPLIT --> TRAIN
    SPLIT --> TEST["held-out set"]
    TEST --> M1["aggregate metric<br/>(serves the head)"]
    TEST --> M2["macro metric<br/>(sees the tail)"]
    KEY -.->|"if skipped: no valid<br/>split is recoverable"| X["unknown bias"]
```

### 7.1 The order that matters

1. **Record provenance at collection.** Free now, impossible later.
2. **Write the skill taxonomy** before sampling — stratification requires knowing
   the strata, and the reason this is skipped is not difficulty but that nobody
   writes it down.
3. **Allocate deliberately** between head and tail. That allocation *is* the
   product decision.
4. **Split by group.**
5. **Report macro and aggregate**, always both.

### 7.2 Deduplication still has a job

Nothing above says deduplication is useless — {{cite:lee2022dedup}} showed
duplicate *training* data wastes budget and degrades models. That is a different
problem from the one in this chapter, and it has a different fix:

| Problem | Where | Fix |
|---|---|---|
| duplicates inside training | train | dedup ({{cite:lee2022dedup}}) |
| duplicates across the split | train/test | group split |
| tail under-representation | selection | stratify |
| eval blind to the tail | metric | macro-average |

**Confusing rows one and two is the error this chapter is about.** Deduplicating
the training set is worth doing and does not make a random split valid.

### 7.3 What a taxonomy costs

Usually a day, and it is the highest-return day in the project. It does not need
to be right — it needs to *exist*, so allocation and macro-averaging have an axis
to work on. A taxonomy that is 70% correct still surfaces the tail; no taxonomy
surfaces nothing.

### 7.4 When curation cannot help

Stratification is capped by what the pool contains. If a skill is genuinely
**absent**, no selection strategy recovers it and the fix is collection —
{{ch:ft-synthetic}}'s subject when collection is impractical.

**The rarest-skill count is how you tell those cases apart**, and it is the first
thing to check on a fine-tune that underperforms: 3 examples means allocate, 0
means collect.

## 8. Implementation

```python {tier=A name=coverage-not-count}
"""Coverage, not count. And the metric that hides the difference.

"We need more data" is the default diagnosis for a fine-tune that underperforms,
and it is usually wrong, because real task distributions are heavy-tailed. Drawing
more examples from a Zipf distribution buys mostly more of what you already had
(eq:coverage-saturates).

This listing does not simulate that claim -- it trains a real classifier on real
sampled data and measures it. A pool of examples spans forty latent skills with a
Zipf frequency profile. Two selection strategies get the SAME budget: draw at
random, or stratify across skills. Then both are evaluated two ways, because the
choice of metric turns out to decide whether the problem is visible at all.
"""
import numpy as np

rng = np.random.default_rng(163)

D, K = 24, 40
POOL = 60000
CLASS_SPREAD, WITHIN = 3.0, 1.0

MU = rng.normal(size=(K, D))
MU = CLASS_SPREAD * MU / np.linalg.norm(MU, axis=1, keepdims=True)

freq = 1.0 / (1 + np.arange(K)) ** 1.1
freq = freq / freq.sum()


def draw(labels):
    return MU[labels] + WITHIN * rng.normal(size=(len(labels), D))


pool_y = rng.choice(K, size=POOL, p=freq)
pool_x = draw(pool_y)

# Two test sets, because they answer different questions.
nat_y = rng.choice(K, size=8000, p=freq)          # the natural distribution
nat_x = draw(nat_y)
bal_y = np.repeat(np.arange(K), 200)              # every skill weighted equally
bal_x = draw(bal_y)


def train(X, y, steps=400, lr=0.5):
    W = np.zeros((D, K))
    b = np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(steps):
        z = X @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / len(X)
        W -= lr * (X.T @ g + 1e-4 * W)
        b -= lr * g.sum(axis=0)
    return W, b


def acc(model, X, y):
    W, b = model
    return float(((X @ W + b).argmax(axis=1) == y).mean())


def select_random(n):
    idx = rng.permutation(POOL)[:n]
    return pool_x[idx], pool_y[idx]


def select_stratified(n):
    """Take an equal share of every skill -- capped by what the pool contains,
    which is the honest constraint. You cannot stratify what you do not have."""
    per = max(1, n // K)
    keep = []
    for k in range(K):
        have = np.flatnonzero(pool_y == k)
        keep.append(rng.permutation(have)[:per])
    idx = np.concatenate(keep)
    return pool_x[idx], pool_y[idx]


def counts(y, m=50):
    """Coverage at a threshold that reflects what a skill needs to be learned,
    and the count for the rarest skill -- which is what bounds macro accuracy."""
    c = np.bincount(y, minlength=K)
    return float((c >= m).mean()), int(c.min())


print(f"Pool of {POOL:,} examples over {K} skills, Zipf-distributed. The most "
      f"common\nskill is {freq[0]/freq[-1]:.0f}x more frequent than the rarest.\n")
print(f"{'budget':>8}{'selection':>13}{'skills':>10}{'rarest':>7}"
      f"{'accuracy on':>14}{'accuracy on':>14}")
print(f"{'':>8}{'':>13}{'>= 50 ex':>10}{'ex':>7}{'natural dist':>14}"
      f"{'every skill':>14}")
print("-" * 65)

results = {}
for n in (500, 1000, 3000, 10000, 30000):
    for name, sel in (("random", select_random), ("stratified",
                                                  select_stratified)):
        X, y = sel(n)
        m = train(X, y)
        cov, rare = counts(y)
        r = (cov, acc(m, nat_x, nat_y), acc(m, bal_x, bal_y), len(y), rare)
        results[(n, name)] = r
        print(f"{len(y):>8,}{name:>13}{cov:>10.0%}{rare:>7,}"
              f"{r[1]:>14.3f}{r[2]:>14.3f}")
    print()

s1k = results[(1000, "stratified")]
r1k = results[(1000, "random")]
r30k = results[(30000, "random")]
print(f"""
Start with the two rows at a budget of 1,000, which is a controlled comparison:
same number of examples, same pool, same training code, different selection.
Random scores {r1k[2]:.3f} on the metric that weights every skill equally.
Stratified scores {s1k[2]:.3f} -- {s1k[2]/r1k[2]-1:.0%} better from selection
alone, with no additional data collected.

Then the comparison that makes the point about budgets. Stratified at
{s1k[3]:,} examples scores {s1k[2]:.3f}; random at {r30k[3]:,} examples scores
{r30k[2]:.3f}. Thirty times the data, and it still loses. That is what "curate
rather than collect" means concretely, and it is an argument about allocation
rather than about quality as a virtue.

The rarest-skill column is the mechanism. Random sampling from a heavy-tailed
distribution grows every skill's count in proportion to its frequency, so the
rarest skill has {r1k[4]} examples at a budget of 1,000 and {r30k[4]} at 30,000.
Macro accuracy is bounded by the worst-covered skills, so it improves at the rate
the TAIL grows, and the tail grows far
more slowly than the budget (eq:coverage-saturates). The head, meanwhile, is
already saturated and each additional head example buys nothing.

Now the column that makes this hard to see in practice, and the honest cost of
stratification.

On the NATURAL distribution -- a test set drawn the same way production traffic
arrives -- the ranking REVERSES. Random at 1,000 scores {r1k[1]:.3f} against
stratified's {s1k[1]:.3f}, and random at 30,000 reaches {r30k[1]:.3f}. Random
selection wins that metric at every budget in the table.

Both rankings are correct, and they disagree because the metrics ask different
questions. Stratification deliberately under-samples the head, so it gives up
accuracy on common requests to buy accuracy on rare ones. This is a TRADE, not a
free improvement, and treating it as strictly better is how curated datasets end
up worse than the ones they replaced.

What is not a trade is the visibility problem. If you sample training data from
production traffic and sample your eval set from production traffic, the eval
inherits the training selection's bias and cannot see what that selection missed
(eq:metric-inherits-bias). The model looks fine, and it fails on exactly the
requests that were rare in both -- which are frequently the requests that matter
most, since rare and unimportant are not the same property.

So three things follow for building a fine-tuning dataset. Report macro-averaged
metrics alongside aggregate ones, because the aggregate cannot see the tail.
Decide the skill taxonomy BEFORE sampling, since stratification requires knowing
what the strata are, and the reason this is skipped is not difficulty but that
nobody writes the taxonomy down. And choose the head/tail balance deliberately,
because the selection strategy IS that choice whether or not it was made
consciously.

One honest limit on this experiment. Stratification is capped here by what the
pool contains, so it works because the rare skills are present but under-sampled.
If a skill is genuinely absent, no selection strategy recovers it and the fix is
collection rather than curation. The rarest-skill column is how you tell those
two cases apart, and it is the first thing to check on a fine-tune that
underperforms.""")
```

Allocation decides what the model learns. The second listing is about whether you
can measure it.

```python {tier=A name=contamination-and-splits}
"""Contamination: why a random split lies, and why deduplication does not fix it.

Fine-tuning data is rarely a set of independent examples. It arrives in clusters
-- the same question asked three ways, a template instantiated per customer, a
document and its summary. Split that pool at random and members of one cluster
land on both sides, so the held-out score measures memorisation and reports it as
generalisation (eq:leakage-inflates).

The standard response is deduplication against a similarity threshold. This
listing measures how much of the problem that removes, and finds two failures at
once: what the threshold catches is not what leaks, and what it discards is not a
random sample of the test set.
"""
import numpy as np

rng = np.random.default_rng(167)

D, NF = 8, 400             # input dim, random-feature width (enough to memorise)
N_CLUST = 1500
REPS, REPS2 = 20, 8


def make_pool(dup_rate, far_share=0.5):
    """Clusters share an ANSWER. Some members are surface-similar to each other
    (a paraphrase), some are surface-DIFFERENT but still share the answer (the
    same question posed from another angle). Both are contamination."""
    centres = rng.normal(size=(N_CLUST, D))
    w = rng.normal(size=D)
    y_c = (np.sin(1.2 * centres @ w / np.sqrt(D)) + 0.5 * centres[:, 0]
           > 0).astype(int)

    xs, ys, gs = [], [], []
    for g in range(N_CLUST):
        n_extra = rng.integers(1, 4) if rng.random() < dup_rate else 0
        for j in range(1 + n_extra):
            eps = 0.05 if (j == 0 or rng.random() > far_share) else 0.9
            xs.append(centres[g] + eps * rng.normal(size=D))
            ys.append(y_c[g]); gs.append(g)
    return np.array(xs), np.array(ys), np.array(gs)


W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def fit_predict(Xtr, ytr, Xte, lam=3e-5):
    """Low ridge: the model CAN memorise, which is the point."""
    P = feat(Xtr)
    A = P.T @ P + lam * len(Xtr) * np.eye(NF)
    c = np.linalg.solve(A, P.T @ (2.0 * ytr - 1))
    return (feat(Xte) @ c > 0).astype(int)


def split_random(n, frac=0.3):
    idx = rng.permutation(n)
    k = int(frac * n)
    return idx[k:], idx[:k]


def split_group(g, frac=0.3):
    groups = np.unique(g)
    gp = rng.permutation(groups)
    held = set(gp[:int(frac * len(groups))].tolist())
    mask = np.array([x in held for x in g])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


print("A pool of clustered examples: members of a cluster share the answer.\n")
print(f"{'duplication':>12}{'random split':>15}{'group split':>14}"
      f"{'inflation':>12}")
print(f"{'rate':>12}{'reports':>15}{'reports':>14}{'':>12}")
print("-" * 53)

table = {}
for dr in (0.0, 0.1, 0.25, 0.5, 0.9):
    ar, ag = [], []
    for _ in range(REPS):
        X, y, g = make_pool(dr)
        tr, te = split_random(len(y))
        ar.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).mean())
        tr, te = split_group(g)
        ag.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).mean())
    a_rand, a_grp = float(np.mean(ar)), float(np.mean(ag))
    table[dr] = (a_rand, a_grp)
    print(f"{dr:>12.0%}{a_rand:>15.3f}{a_grp:>14.3f}{a_rand - a_grp:>+12.3f}")

print("\n\nDoes decontamination by distance threshold recover the truth?\n")
truth = table[0.5][1]
print(f"{'threshold':>10}{'% of test':>12}{'reports':>10}{'error vs':>13}"
      f"{'share of kept':>15}")
print(f"{'':>10}{'discarded':>12}{'':>10}{'group split':>13}"
      f"{'still leaked':>15}")
print("-" * 60)

TAUS = (0.0, 0.3, 0.6, 1.0, 2.0)
agg = {t: [[], [], []] for t in TAUS}
for _ in range(REPS2):
    X, y, g = make_pool(0.5)
    tr, te = split_random(len(y))
    d2 = ((X[te] ** 2).sum(1)[:, None] + (X[tr] ** 2).sum(1)[None, :]
          - 2.0 * X[te] @ X[tr].T)
    d = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    leaked = np.isin(g[te], g[tr])          # ground truth: shares a cluster
    for tau in TAUS:
        keep = d > tau
        if keep.sum() < 40:
            continue
        a = (fit_predict(X[tr], y[tr], X[te][keep]) == y[te][keep]).mean()
        agg[tau][0].append(1 - keep.mean())
        agg[tau][1].append(a)
        agg[tau][2].append(leaked[keep].mean())

rows = {}
for tau in TAUS:
    rm, ac, rs = (float(np.mean(v)) for v in agg[tau])
    rows[tau] = (rm, ac, rs)
    print(f"{tau:>10.1f}{rm:>12.1%}{ac:>10.3f}{ac - truth:>+13.3f}{rs:>15.1%}")

t0, t3, t20 = rows[0.0], rows[0.3], rows[2.0]
print(f"""
The first table is the size of the problem, and its top row is the control: with
no duplication the two splits agree to {abs(table[0.0][0]-table[0.0][1]):.3f}, as
they must, because with singleton clusters the two procedures are the same
procedure.

Add clustering and they separate, monotonically. At a 50% duplication rate the
random split reports {table[0.5][0]:.3f} where the group split reports
{table[0.5][1]:.3f}, an inflation of {table[0.5][0]-table[0.5][1]:+.3f}; at 90% it
is {table[0.9][0]-table[0.9][1]:+.3f}.

Nothing about the model changed between those two numbers. The same features, the
same training procedure, the same pool -- a different assignment of clusters to
sides. The random split measures how well the model recalls answers it was shown
and reports it as how well the model answers new questions
(eq:leakage-inflates).

This is the most common way a fine-tuning result turns out to be fictional, and
it is invisible from the inside: the loss curve looks healthy, the held-out score
looks good, and the number is simply about something other than what it claims.

The second table is the part that decides what to do about it, and it fails in
two directions at once.

Read the last column first. At a threshold of 0.3, {t3[0]:.0%} of the test set has
been discarded and {t3[2]:.0%} of what remains STILL shares a cluster with a
training example. The threshold removed the surface-similar duplicates and left
the rest, because half of each cluster's members were built to be surface-
DIFFERENT while sharing the answer -- the same question from another angle, a
different document about the same fact. Those are contamination by every meaning
that matters, and they sit far away in input space, so no threshold on surface
distance will find them (eq:distance-misses-semantics).

The bottom row settles it. Push the threshold to 2.0 and {t20[0]:.0%} of the test
set is thrown away -- and the leaked share of what remains goes UP, to
{t20[2]:.0%}. That is not a plateau, it is the wrong direction. An aggressive
distance filter preferentially removes the surface-similar contamination and
preferentially KEEPS the semantic contamination, so the harder you scrub, the
more concentrated in real leakage the surviving test set becomes.

Now read the error column, which is the failure people do not anticipate.
Decontamination does not converge on the truth from above; it swings PAST it.
Undecontaminated, the split over-reports by {t0[1]-truth:+.3f}. After thresholding
it reports {t3[1]-truth:+.3f} against the group split, while half the remaining
test set is still leaked.

Both effects are present at once and they partly cancel, which is worse than
either alone, because the cancellation is accidental. Removing test examples that
are near training examples does not remove a random sample of the test set: it
removes the ones the model finds easy, so what survives is harder than the task
is. The number you get is a leaked score on an unrepresentative subset, and there
is no reason for it to land anywhere in particular.

So the fix is not a better threshold. It is to split by GROUP, using the
provenance you had before the examples became vectors: the document they came
from, the customer, the template, the source URL, the ticket. That information is
free at collection time and largely unrecoverable afterwards, which is the whole
practical lesson of this chapter.

Write down the group key when you build the dataset. If you did not, the honest
options are to reconstruct provenance or to report that your held-out numbers
carry an unknown bias -- and the second is far more common than the literature
would suggest.""")
```

## 9. Practical Example

**Allocation beats collection.** A pool of 60,000 examples over 40 Zipf-distributed
skills, head **58×** the rarest. At a budget of 1,000: random scores **0.486**
macro, stratified **0.639** — **31% better from selection alone**, same budget,
same pool, same code.

**And 1,000 stratified beats 30,000 random: 0.639 against 0.630.** Thirty times
the data, and it loses.

**The mechanism is the rarest-skill count**, not a vague notion of quality. Random
leaves the rarest skill with **3** examples at 1,000 and **133** at 30,000;
stratified gives it **25** at 1,000. Macro accuracy is bounded by the
worst-covered skills, so it improves at the tail's rate —
{{eq:coverage-saturates}} — while the head saturated long ago.

**But this is a trade and not a free win.** On the natural-distribution test set,
random wins at every budget: **0.729 against 0.649** at 1,000, **0.798 against
0.760** at the top. Stratification sells head accuracy to buy tail accuracy.
{{eq:macro-versus-aggregate}} is why both rankings are correct.

> **IMPORTANT:** {{eq:metric-inherits-bias}} is the part that makes this invisible.
> Sample training data from traffic, sample eval from traffic, and the evaluation
> weights each skill by exactly the frequency that under-trained it. **More eval
> data does not fix a structural blindness.**

**A random split lies about clustered data.** Control first: at 0% duplication the
two splits agree to **0.005**, as they must. At 50%: random reports **0.811**,
group reports **0.768**, inflation **+0.042**; at 90%, **+0.041**.

**Nothing about the model changed** between those numbers — same features, same
training, same pool, different assignment of clusters to sides.
{{eq:leakage-inflates}}.

**Threshold decontamination fails in both directions.** At $\tau = 0.3$: **33.9%**
of the test set discarded, and **50.1%** of the survivors still share a cluster
with a training example. At $\tau = 2.0$: **92.2%** discarded and the leaked share
**rises to 77.0%** — the wrong direction, because an aggressive distance filter
preferentially removes *surface* duplicates and keeps *semantic* ones
({{eq:distance-misses-semantics}}).

**Meanwhile the reported score overshoots**: **+0.031** before filtering,
**−0.031** after ({{eq:decontamination-overshoot}}). **Two biases in opposite
directions, partly cancelling by accident** — which is worse than either alone,
because the result lands nowhere in particular and looks precise.

**{{eq:group-split}} is the only clean estimator here**, and it needs nothing but
a provenance key that was free to record and cannot be reconstructed.

## 10. Production Considerations

**Record the provenance key at collection.** Document, customer, template, URL,
ticket. This is the cheapest high-value decision in the chapter.

**Write the skill taxonomy before sampling.** It does not need to be right; it
needs to exist.

**Report macro and aggregate metrics together, always.** One of them cannot see
the tail and the other cannot see the typical case.

**Split by group, not at random, and not by post-hoc clustering.**

**State the head/tail allocation as a decision**, with whoever owns the product in
the room. It is not a data-engineering detail.

**Check the rarest-skill count** before concluding you need more data.

**Do not report "decontaminated at threshold τ" as a validity claim.** It says a
procedure ran.

**Deduplicate the training set anyway** ({{cite:lee2022dedup}}) — different
problem, still worth doing.

## 11. Common Mistakes

**Diagnosing "we need more data"** without checking the rarest-skill count.

**Evaluating on a sample from the same distribution the training data came from**,
and concluding the model is fine.

**Reporting only aggregate accuracy.**

**Splitting at random** on clustered data.

**Clustering post hoc to build groups** — that re-imports
{{eq:distance-misses-semantics}}.

**Believing decontamination converges on the truth.**

**Treating stratification as strictly better** rather than as a trade.

**Skipping the taxonomy because it feels unscientific**, then having no axis on
which to allocate or report.

## 12. Failure Modes

**Great offline scores, poor production behaviour on rare requests.** Cause:
{{eq:metric-inherits-bias}}. Fix: macro metrics and a taxonomy.

**Held-out score that does not survive contact with new data.** Cause:
{{eq:leakage-inflates}}. Fix: {{eq:group-split}}.

**Curated dataset that made the product worse.** Cause: stratification traded away
head accuracy that mattered. Fix: allocate deliberately, measure both metrics.

**Scores that fall after "cleaning" the eval set.** Cause:
{{eq:decontamination-overshoot}}, not a real regression.

**Larger model shows a bigger benchmark gain than it deserves.** Cause:
{{eq:leakage-ceiling}}.

**More data does not help at all.** Cause: the missing skills are absent, not
under-sampled. Fix: collect or generate ({{ch:ft-synthetic}}).

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| collect more, sample randomly | budget, tail coverage | the tail genuinely does not matter |
| stratified selection | head accuracy | every kind of request matters |
| importance weighting in the loss | optimisation stability | you cannot re-select |
| oversampling the tail | overfitting rare examples | very small tail counts |
| synthetic tail data ({{ch:ft-synthetic}}) | distribution fidelity | the skill is absent |
| group split | test-set size | always, if $g$ exists |
| threshold decontamination | unknown bias | only when $g$ is unrecoverable |

**The last row is a salvage operation, not a method.** If provenance was not
recorded, threshold decontamination is what you do while reporting that the number
carries an unknown bias — which is a great deal more common than the literature
suggests.

## 14. Evaluation

**Report macro and aggregate, with the taxonomy.** A macro number without the
strata is uninterpretable.

**State the split policy explicitly**, and the group key.

**Report the rarest-skill count** in the training set — it bounds the macro score
and no other number reveals it.

**Report duplication rate** if you know it, and the group-split delta if you can
compute both.

**Never report a decontaminated score as if it were a group-split score.** They
estimate different things and one of them estimates nothing in particular.

## 15. Advanced Concepts

**Leakage scales with capacity.** {{maturity:MATURE}} {{eq:leakage-ceiling}} means
the same flawed dataset flatters larger models more — so a contaminated benchmark
distorts scaling conclusions on top of individual ones.

**Taxonomy as a product artefact.** {{maturity:EMERGING}} The skill list is the
one document that makes allocation, macro-averaging, and gap analysis possible at
once. It is usually treated as analysis output; it is more useful as design input.

**Quality filtering as selection.** {{maturity:MATURE}}
{{cite:gunasekar2023}} showed aggressive quality filtering can beat scale
outright. Note this chapter's caution: filtering is a *selection* operation and
inherits every property above, including the ability to remove the tail while
looking like an improvement — which is {{ch:ft-synthetic}}'s central hazard.

**A thousand examples, revisited.** {{maturity:MATURE}}
{{cite:zhou2023lima}}'s result is often read as "you need very little data". This
chapter suggests the sharper reading: **you need very little data *per skill*, and
the number of skills is what you were never counting.**

**Provenance as an infrastructure decision.** {{maturity:EMERGING}}
{{eq:group-split}} needs a key that only the ingestion path can supply.
{{part:20}}'s pipelines are where this is won or lost, long before anyone opens a
training script.

**And the deduplication result has a limit worth naming.**
{{maturity:EMERGING}} Exact and near-duplicate removal is measurable and worth
doing, but the duplicates that damage a fine-tune most are *semantic*: fifty
examples teaching the same lesson in different words look diverse to every metric
here and are one example to the model. **Diversity measured on the surface form
overstates diversity of instruction**, which is the mechanism behind a dataset that
passes every check and still produces a model with one behaviour.

## 16. Connection to Previous Chapters

{{ch:fm-instruction-tuning}} established that quality dominates quantity; this
chapter supplies the *mechanism* — coverage, not quality as a virtue — and the
trade that claim conceals. {{ch:ft-sft}}'s training loop consumes whatever
selection produces, and cannot detect a bad one. {{ch:mle-splits}}'s machinery is
correct and routinely misapplied, which {{eq:group-split}} corrects.
{{ch:ft-when}}'s decision depends on a *reliable* measurement of the gap, so a
leaked evaluation corrupts the decision to fine-tune at all.
Forward: {{ch:ft-synthetic}} generates what curation cannot find and inherits
every hazard here; {{ch:ft-preference}} needs the same provenance discipline for
preference pairs; {{part:25}} owns evaluation infrastructure.

## 17. Exercises

1. Compute $N_{\text{cover}}$ from {{eq:coverage-saturates}} for $K=100$,
   $s=1.2$, $m=30$, and compare with the stratified budget.
2. Derive {{eq:leakage-inflates}} and compute the inflation at $\lambda = 0.3$,
   $a_{\text{mem}} = 0.98$, $a_{\text{gen}} = 0.70$.
3. In `coverage-not-count`, change the Zipf exponent to 0.7 and 1.5. How does the
   stratification advantage move, and why?
4. Add a third strategy to the same listing: stratify by skill but weight
   proportionally to $\sqrt{p_k}$. Where does it land on both metrics?
5. In `contamination-and-splits`, set `far_share=0.0` so all duplicates are
   surface-similar. Does threshold decontamination work now? What does that tell
   you about when it is safe?
6. Explain, using {{eq:decontamination-overshoot}}, why a decontaminated score can
   be *below* the true generalisation score.
7. Your dataset has no provenance key. Describe two ways to reconstruct one and
   what each would miss.
8. Write a skill taxonomy for a task you work on. How many strata, and what is the
   rarest one's count in your current data?

## 18. Interview Questions

1. Your fine-tune underperforms. What do you check before asking for more data?
2. Why can 1,000 curated examples beat 30,000 random ones?
3. What does stratified selection cost?
4. Why can an eval set drawn from production traffic fail to detect a tail
   problem?
5. What is the difference between deduplicating training data and splitting by
   group?
6. Why does a random split inflate scores on clustered data, and by how much?
7. Why does the inflation grow with model size?
8. Does decontamination by similarity threshold converge on the true score?
9. What is a provenance key, and when must you record it?
10. When is curation the wrong answer?

## 19. Research Questions

1. {{eq:leakage-ceiling}} predicts that leakage flatters larger models more. How
   much of reported scaling on public benchmarks is this effect?
2. Can a semantic contamination detector — grouping by *answer* rather than by
   input distance — be made reliable enough to replace {{eq:group-split}} when
   provenance is lost?
3. {{eq:decontamination-overshoot}} makes the filtered estimator biased in an
   unknown direction. Is there a correction using the $d$-distribution that
   recovers an unbiased estimate?
4. How should budget be allocated when skills have different *values* as well as
   different frequencies? {{eq:macro-versus-aggregate}} assumes uniform value.
5. Does the coverage argument hold for capabilities that compose, where a skill's
   examples partly teach its neighbours?

## 20. Chapter Summary

**"More data" is the wrong diagnosis for a heavy tail.** At a budget of 1,000,
stratified selection scores **0.639** against random's **0.486** — 31% from
selection alone — and **1,000 stratified beats 30,000 random** (0.639 vs 0.630).
The mechanism is the rarest-skill count (**3 → 25** at equal budget), because
{{eq:coverage-saturates}} makes coverage cost grow with the tail's depth while the
head saturated long ago.

**And it is a trade.** On a natural-distribution test set the ranking reverses at
every budget (**0.729 vs 0.649** at 1,000). {{eq:macro-versus-aggregate}} says
both rankings are correct; **the selection strategy is a product decision about
whom to serve**, made deliberately or by default.

**{{eq:metric-inherits-bias}} is why nobody notices.** An eval set drawn from the
same traffic weights each skill by the frequency that under-trained it. **More
eval data cannot fix a structural blindness — only macro-averaging and a
taxonomy can.**

**A random split lies about clustered data.** Control **−0.005** at no
duplication; **+0.042** at 50%, **+0.041** at 90%
({{eq:leakage-inflates}}). Nothing about the model changed. **And
{{eq:leakage-ceiling}} means the same flaw flatters bigger models more.**

**Threshold decontamination fails in both directions at once.** It misses what
leaks — **50.1%** of survivors still contaminated after discarding a third of the
test set, **rising to 77.0%** when 92.2% is discarded, because distance filters
remove surface duplicates and keep semantic ones
({{eq:distance-misses-semantics}}). And it **overshoots the truth**, **+0.031 →
−0.031** ({{eq:decontamination-overshoot}}), because what it removes is the easy
examples. **Two accidental biases partly cancelling is worse than one**, because
the result looks precise and means nothing.

**{{eq:group-split}} is the clean estimator, and it needs only a key that was free
to record.** Which is the lesson worth carrying: **the two decisions that
determine whether a fine-tune works and whether you can tell — the skill taxonomy
and the provenance key — both cost nothing, both must be made before any data is
collected, and neither can be added afterwards.**

## 21. Further Reading

{{cite:zhou2023lima}} for the small-data result, read as *little data per skill*
rather than *little data*, which is the reading this chapter's coverage argument
supports.
{{cite:lee2022dedup}} for deduplication done properly, and as the clearest
statement of the problem this chapter distinguishes from leakage.
{{cite:gunasekar2023}} for quality filtering beating scale — and note that
filtering is a selection operation subject to every hazard in
{{sec:5-formal-explanation}}.
{{cite:wei2022flan}} for task mixtures, which is the taxonomy idea at pretraining
scale before anyone called it that.
{{cite:ouyang2022}} for what a deliberately constructed demonstration set looks
like, including the parts about annotator selection that the model-focused
retellings omit.
