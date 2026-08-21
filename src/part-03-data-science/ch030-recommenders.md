---
id: ds-recsys
number: 30
part: III
tier: focused
status: reviewed
requires: [ds-timeseries, math-eigen]
provides: [collaborative-filtering, matrix-factorisation, cold-start,
           implicit-feedback, feedback-loop]
citations: [koren2009]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain the recommendation problem and why it is not standard supervised
   learning.
2. Distinguish content-based from collaborative approaches and state each one's
   failure mode.
3. Implement matrix factorisation and connect it to the SVD.
4. Explain why the SVD cannot be applied directly to a sparse ratings matrix.
5. Handle implicit feedback and explain why absence is ambiguous.
6. Address the cold-start problem.
7. Explain feedback loops and why offline evaluation systematically misleads.
8. Choose ranking metrics appropriate to a recommendation task.

## 2. Why This Matters

Recommendation is where several threads of this book meet, and it is the last
chapter of Part III because it violates the assumptions of everything preceding
it in an instructive way.

**It is not standard supervised learning.** There is no fixed input-output
mapping to learn; there is a sparse matrix of interactions with 99%+ of entries
missing, and the missingness is not random — you see what was shown to you.

**It is where representation learning starts.** {{cite:koren2009}} showed that
representing users and items as vectors in a shared low-dimensional space and
predicting an interaction as their inner product beats hand-engineered
similarity. That embedding-plus-inner-product formulation is the direct
ancestor of the two-tower retrieval architectures of {{part:11}} and, in a
sense, of how attention scores are computed ({{ch:tf-scaled-dot-product}}).

**It has a feedback loop.** The model determines what users see, which
determines the next training set. The system learns from its own past behaviour
rather than from preferences, and offline evaluation on logged data is therefore
biased in a direction that flatters the incumbent model.

That last point generalises well beyond recommenders. Any deployed model whose
outputs influence future data has it, which by {{part:17}} includes most agentic
systems.

## 3. Prerequisites

{{ch:math-eigen}} for the SVD and low-rank approximation — this chapter is
largely an application of it. {{ch:ds-timeseries}} for the i.i.d. violation
theme. {{ch:ds-collection}} for selection bias, which is what a feedback loop
produces.

## 4. Intuitive Explanation

### 4.1 The problem shape

You have a matrix of users by items, mostly empty:

```text
              item1  item2  item3  item4  item5
    user1       5      ?      3      ?      ?
    user2       ?      4      ?      ?      2
    user3       4      ?      ?      5      ?
    user4       ?      ?      2      ?      ?
```

The task is to fill in the blanks — or, more usefully, to rank the blanks for
each user.

The density is the striking part. A large platform might have $10^{8}$ users and
$10^{6}$ items, of which perhaps $10^{10}$ cells are observed out of $10^{14}$ —
0.01%. Almost everything is missing.

### 4.2 Two families

**Content-based** uses item attributes. If you liked action films, here are
other action films. It handles new items well, needs no other users, and is
limited to what the attributes capture — it will never discover that people who
like a particular obscure film also like a documentary from a different genre.

**{{term:collaborative-filtering}}** uses the interaction patterns alone. People
who agreed with you before will agree with you again. It needs no item
attributes and discovers relationships no attribute encodes; it fails completely
for new users and items.

The interesting insight from {{cite:koren2009}} is that collaborative filtering
works *better* despite knowing nothing about the items, because the interaction
matrix encodes similarity judgements that no feasible set of attributes would
capture.

### 4.3 Latent factors

The core idea: explain a large sparse matrix with a small number of hidden
dimensions.

Suppose ratings are driven by a handful of unobserved factors — how much a film
leans toward action, how recent it is, how demanding it is. Each item has a
position on each factor; each user has a preference weight on each factor. The
predicted rating is their inner product:

$$
\hat{r}_{ui} = \vec{p}_u \cdot \vec{q}_i
$$ (eq:mf-basic)

Two vectors of length $k \approx 50$ replace the whole row and column. For
$10^{6}$ users and $10^{5}$ items, that is $5.5 \times 10^{7}$ parameters
instead of $10^{11}$ cells — and, crucially, it *generalises*, because a user's
vector is estimated from all their ratings jointly.

Nobody labels the factors. They emerge from fitting, and they are usually only
loosely interpretable — which is the first appearance in this book of learned
representations replacing engineered ones.

### 4.4 Why you cannot just use the SVD

{{ch:math-eigen}} showed that the truncated SVD gives the provably best low-rank
approximation. It is natural to reach for it here and it does not work.

The SVD is defined for a **complete** matrix. A ratings matrix is 99% missing,
and the two obvious repairs both fail:

- **Fill missing with zero.** That asserts every unseen item was rated zero,
  which is a strong and false claim. The factorisation then spends its capacity
  reproducing those zeros.
- **Fill with the mean.** Better, and it still biases the result toward the
  mean everywhere and destroys the sparsity that makes computation feasible.

The correct approach is to fit only on the observed entries — minimising
reconstruction error over the cells you actually have, ignoring the rest. That
is no longer an eigenvalue problem with a closed-form answer; it is an
optimisation solved by gradient descent or alternating least squares
({{ch:math-optimization}}).

## 5. Formal Explanation

### 5.1 Matrix factorisation with biases

The model that {{cite:koren2009}} describes adds bias terms, which matter more
than they look:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \vec{p}_u \cdot \vec{q}_i
$$ (eq:mf-with-bias)

with $\mu$ the global mean, $b_u$ a per-user bias (some users rate everything
highly), and $b_i$ a per-item bias (some items are simply better).

The biases capture the large, uninteresting part of the variation, leaving the
factors to model genuine interaction — *this user's affinity for this kind of
item*. A bias-only model is a strong baseline and frequently captures most of
the achievable accuracy.

The objective, fitted over observed entries only:

$$
\min_{p, q, b} \sum_{(u,i) \in \mathcal{K}}
  \big(r_{ui} - \hat{r}_{ui}\big)^{2}
  + \lambda\big(\|\vec{p}_u\|^{2} + \|\vec{q}_i\|^{2} + b_u^{2} + b_i^{2}\big)
$$ (eq:mf-objective)

where $\mathcal{K}$ is the set of observed pairs. The regularisation is
essential: a user with three ratings would otherwise get a factor vector fitted
exactly to those three points.

### 5.2 Implicit feedback

Explicit ratings are rare. Most systems observe clicks, plays and purchases —
{{term:implicit-feedback}}.

Two properties change everything:

**There are no negatives.** A user not clicking an item may mean dislike, or may
mean they never saw it. Absence conflates preference with exposure.

**Values are confidence, not preference.** Playing a song fifty times is
stronger evidence than playing it once, but it is not "fifty times better".

The standard treatment introduces a binary preference and a confidence weight:

$$
p_{ui} = \begin{cases} 1 & r_{ui} > 0 \\ 0 & \text{otherwise}\end{cases},
\qquad
c_{ui} = 1 + \alpha r_{ui}
$$ (eq:implicit-confidence)

and minimises a *confidence-weighted* error over **all** pairs, including the
unobserved ones — which are treated as weak negatives with confidence 1:

$$
\min \sum_{u, i} c_{ui}\big(p_{ui} - \vec{p}_u\cdot\vec{q}_i\big)^{2}
  + \lambda(\cdots)
$$ (eq:implicit-objective)

Summing over all pairs rather than observed ones is what makes this different,
and it is computationally feasible only because of an algebraic trick that
exploits the structure of the all-pairs term.

### 5.3 Cold start

{{term:cold-start}} is the absence of interaction data. Three cases, three
answers:

{#tbl:cold-start caption="Cold-start cases and the standard responses. Every production recommender needs an explicit answer for each."}

| Case | Response |
|---|---|
| New user | popularity, onboarding questions, contextual signals, demographics |
| New item | content-based on attributes; deliberate exploration |
| New system | content-based or rules until interactions accumulate |

The general answer is a **hybrid**: content-based when interaction data is
absent, collaborative once it exists, blended in between. A pure collaborative
system has nothing to say about a new item, and new items are a permanent
condition rather than a transient one.

### 5.4 Feedback loops

This is the structural problem, and it is more serious than the others.

```mermaid {#fig:feedback-loop caption="The recommender feedback loop. The model determines exposure, exposure determines the interactions logged, and those interactions train the next model. The data is not a sample of preferences; it is a sample of what the previous model chose to show."}
graph LR
  M[model] --> R[recommendations]
  R --> E[what users are exposed to]
  E --> I[observed interactions]
  I --> M
  style E fill:#fde68a,stroke:#ca8a04
```

Three consequences follow.

**Popularity amplification.** Popular items are recommended, therefore seen,
therefore interacted with, therefore more popular. The rich get richer for
reasons unrelated to quality.

**Offline evaluation is biased.** Logged interactions only cover items the old
model showed. A new model recommending something the old one never showed gets
no credit, because there is no logged interaction to score against — so offline
metrics systematically favour models similar to the incumbent.

**The data is not a preference sample.** It is a sample of *what was shown*,
filtered by what users then chose. That is selection bias
({{ch:ds-collection}}) with the selection performed by your own system.

> IMPORTANT: The practical consequence is that offline metrics and online
> results routinely disagree, and when they do, the online result is right.
> Recommenders must be evaluated by A/B test ({{ch:ds-experiments}}). Offline
> evaluation is useful for eliminating clearly bad candidates, not for choosing
> between good ones. Logging the *propensity* — the probability the old model
> had of showing each item — permits importance-weighted offline estimates that
> partly correct for this.

### 5.5 Ranking metrics

Recommendation is ranking, not rating prediction, so RMSE on held-out ratings is
the wrong measure. What matters is the quality of the top few.

{#tbl:ranking-metrics caption="Ranking metrics for recommendation. Position-aware metrics matter because users see only the top of the list."}

| Metric | Measures |
|---|---|
| Precision@k | fraction of the top $k$ that are relevant |
| Recall@k | fraction of relevant items appearing in the top $k$ |
| MAP@k | precision averaged over relevant positions |
| NDCG@k | gain discounted by position; rewards ranking relevant items higher |
| MRR | reciprocal rank of the first relevant item |
| Coverage | fraction of the catalogue ever recommended |
| Diversity | dissimilarity within a recommendation list |

The last two are not accuracy metrics and are frequently the ones that matter. A
recommender maximising accuracy converges on recommending the same popular items
to everyone, which is accurate and useless — a failure that no accuracy metric
detects.

## 6. Mathematical Foundation

### 6.1 Why fitting only observed entries changes the problem

The SVD solves

$$
\min_{\rank(\mat{B}) \le k} \|\mat{A} - \mat{B}\|_{F}^{2}
$$

over *all* entries, and {{ch:math-eigen}} gave the closed-form answer via
Eckart-Young.

Matrix factorisation solves the masked version:

$$
\min_{\mat{P}, \mat{Q}} \sum_{(u,i) \in \mathcal{K}}
  \big(A_{ui} - \vec{p}_u\cdot\vec{q}_i\big)^{2}
$$ (eq:masked-objective)

The mask destroys the structure the SVD relies on. The problem is **non-convex**
in $(\mat{P}, \mat{Q})$ jointly — a product of two unknowns — so there is no
closed form and no guarantee of a global optimum.

It is, however, convex in each factor *given the other*, which is what makes
alternating least squares work: fix $\mat{Q}$ and solve for $\mat{P}$ in closed
form, then swap, and repeat. Each step is a ridge regression and each strictly
decreases the objective.

### 6.2 The gradient updates

For stochastic gradient descent on {{eq:mf-objective}}, with error
$e_{ui} = r_{ui} - \hat{r}_{ui}$:

$$
\vec{p}_u \leftarrow \vec{p}_u + \eta\big(e_{ui}\,\vec{q}_i - \lambda\vec{p}_u\big)
$$ (eq:mf-update-p)

$$
\vec{q}_i \leftarrow \vec{q}_i + \eta\big(e_{ui}\,\vec{p}_u - \lambda\vec{q}_i\big)
$$ (eq:mf-update-q)

with the biases updated analogously. Each factor moves toward the other, scaled
by the error — the symmetry reflecting the symmetry of the inner product.

The regularisation term shrinks both toward zero, which is
{{ch:math-optimization}}'s Gaussian prior applied per user and per item, and it
is what prevents a user with three ratings from acquiring a confident factor
vector.

### 6.3 Why popularity amplification is self-reinforcing

Let $s_i(t)$ be item $i$'s share of exposures at time $t$, and suppose the
probability of interaction given exposure is $q_i$. Interactions accrue in
proportion to $s_i q_i$, and if the next round's exposure share is proportional
to accumulated interactions:

$$
s_i(t+1) \propto s_i(t)\,q_i
$$ (eq:popularity-dynamics)

This is multiplicative growth. Iterating, $s_i(t) \propto s_i(0)\,q_i^{t}$, so
the share of the item with the largest $q_i$ tends to 1 exponentially — and the
initial condition $s_i(0)$ matters permanently, because an item that was never
shown early never accumulates the interactions that would get it shown later.

The consequence is that **an item's long-run exposure depends on its initial
exposure, not only on its quality**. Deliberate exploration — showing items the
model does not favour, at some cost in short-term accuracy — is the standard
correction, and it is the same explore-exploit trade that recurs in
{{ch:ag-planning}}.

### 6.4 NDCG

Discounted cumulative gain at $k$:

$$
\text{DCG@}k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}
$$ (eq:dcg)

normalised by the ideal ordering:

$$
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$ (eq:ndcg)

Two design choices are worth noting. The **exponential gain** $2^{\text{rel}}-1$
makes a highly relevant item worth disproportionately more than a marginally
relevant one. The **logarithmic discount** means position 1 is worth about 1.6
times position 3, reflecting that users attend to the top of a list — a
plausible model of attention rather than a derived truth.

## 7. Implementation

```python {tier=A name=matrix-factorisation}
"""Matrix factorisation from scratch, and why the SVD cannot be used directly.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- a synthetic ratings matrix with known latent structure -----------------
n_users, n_items, k_true = 800, 400, 6
P_true = rng.normal(0, 0.7, (n_users, k_true))
Q_true = rng.normal(0, 0.7, (n_items, k_true))
bu_true = rng.normal(0, 0.4, n_users)
bi_true = rng.normal(0, 0.5, n_items)
MU = 3.6
full = MU + bu_true[:, None] + bi_true[None, :] + P_true @ Q_true.T
full = np.clip(full + rng.normal(0, 0.25, full.shape), 1, 5)

# Observe only ~4% of cells, as a real system would.
density = 0.04
mask = rng.random(full.shape) < density
rows, cols = np.where(mask)
ratings = full[rows, cols]
print(f"matrix {n_users}x{n_items} = {full.size:,} cells")
print(f"observed: {mask.sum():,} ({mask.mean():.1%})")

# hold out 20% of the OBSERVED entries
perm = rng.permutation(len(ratings))
cut = int(0.8 * len(ratings))
tr, te = perm[:cut], perm[cut:]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- baselines ---------------------------------------------------------------
print("\n" + "=" * 72)
print("baselines first")
print("=" * 72)
mu = ratings[tr].mean()
print(f"global mean only          RMSE {rmse(mu, ratings[te]):.4f}")

# bias-only model, fitted by alternating averages
bu = np.zeros(n_users)
bi = np.zeros(n_items)
for _ in range(15):
    resid = ratings[tr] - mu - bi[cols[tr]]
    for u in range(n_users):
        m = rows[tr] == u
        bu[u] = resid[m].sum() / (m.sum() + 8) if m.any() else 0.0
    resid = ratings[tr] - mu - bu[rows[tr]]
    for i in range(n_items):
        m = cols[tr] == i
        bi[i] = resid[m].sum() / (m.sum() + 8) if m.any() else 0.0
bias_pred = np.clip(mu + bu[rows[te]] + bi[cols[te]], 1, 5)
print(f"biases only (eq. 30.2)    RMSE {rmse(bias_pred, ratings[te]):.4f}")

# --- eq. 30.5/30.6: matrix factorisation by SGD -----------------------------
print("\n" + "=" * 72)
print("matrix factorisation (eqs. 30.5-30.6)")
print("=" * 72)


def fit_mf(k=10, epochs=40, lr=0.012, lam=0.06):
    P = rng.normal(0, 0.05, (n_users, k))
    Q = rng.normal(0, 0.05, (n_items, k))
    bu_ = np.zeros(n_users)
    bi_ = np.zeros(n_items)
    order = tr.copy()
    for ep in range(epochs):
        rng.shuffle(order)
        for idx in order:
            u, i, r = rows[idx], cols[idx], ratings[idx]
            pred = mu + bu_[u] + bi_[i] + P[u] @ Q[i]
            e = r - pred
            bu_[u] += lr * (e - lam * bu_[u])
            bi_[i] += lr * (e - lam * bi_[i])
            pu = P[u].copy()
            P[u] += lr * (e * Q[i] - lam * P[u])
            Q[i] += lr * (e * pu - lam * Q[i])
    return P, Q, bu_, bi_


print(f"{'k':>4} {'train RMSE':>12} {'test RMSE':>11}")
for k in (2, 6, 12, 30):
    P, Q, bu_, bi_ = fit_mf(k=k)
    def predict(idx):
        return np.clip(mu + bu_[rows[idx]] + bi_[cols[idx]]
                       + np.sum(P[rows[idx]] * Q[cols[idx]], axis=1), 1, 5)
    print(f"{k:>4} {rmse(predict(tr), ratings[tr]):>12.4f} "
          f"{rmse(predict(te), ratings[te]):>11.4f}")
print(f"\ntrue latent dimension is {k_true}. Beyond it, train error keeps")
print("falling and test error does not — the regularisation of eq. 30.3 is")
print("what stops it diverging entirely.")

# --- section 4.4: why the SVD cannot be used directly -----------------------
print("\n" + "=" * 72)
print("why not just take the SVD? (section 4.4)")
print("=" * 72)

k = 6
filled_zero = np.zeros_like(full)
filled_zero[rows[tr], cols[tr]] = ratings[tr]
filled_mean = np.full_like(full, mu)
filled_mean[rows[tr], cols[tr]] = ratings[tr]

for label, M in (("fill missing with 0", filled_zero),
                 ("fill missing with the mean", filled_mean)):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    approx = (U[:, :k] * s[:k]) @ Vt[:k]
    pred = np.clip(approx[rows[te], cols[te]], 1, 5)
    print(f"  {label:<28} test RMSE {rmse(pred, ratings[te]):.4f}")

P, Q, bu_, bi_ = fit_mf(k=6)
mf_pred = np.clip(mu + bu_[rows[te]] + bi_[cols[te]]
                  + np.sum(P[rows[te]] * Q[cols[te]], axis=1), 1, 5)
print(f"  {'MF on observed entries only':<28} test RMSE "
      f"{rmse(mf_pred, ratings[te]):.4f}")
print("\nFilling makes the SVD applicable and wrong: it spends its capacity")
print("reproducing invented values. Fitting the observed cells only is a")
print("different, non-convex problem (eq. 30.7) with no closed form.")
```

## 8. Practical Example

Ranking evaluation and the feedback loop — the two things that make offline
recommender numbers untrustworthy.

```python {tier=A name=ranking-and-feedback}
"""Ranking metrics, and a simulation of the popularity feedback loop.
"""
import numpy as np

rng = np.random.default_rng(3)

# --- ranking metrics ---------------------------------------------------------
def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(rels, k):
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / ideal if ideal > 0 else 0.0


def precision_at_k(rels, k):
    return float(np.mean(np.asarray(rels)[:k] > 0))


print("=" * 72)
print("ranking metrics: position matters (eqs. 30.10-30.11)")
print("=" * 72)
lists = {
    "relevant first":  [3, 2, 1, 0, 0],
    "relevant last":   [0, 0, 1, 2, 3],
    "one great at #1": [3, 0, 0, 0, 0],
    "three mediocre":  [1, 1, 1, 0, 0],
}
print(f"{'ranking':<20} {'P@5':>7} {'DCG@5':>9} {'NDCG@5':>9}")
for label, rels in lists.items():
    print(f"{label:<20} {precision_at_k(rels,5):>7.2f} "
          f"{dcg_at_k(rels,5):>9.3f} {ndcg_at_k(rels,5):>9.3f}")
print("\nThe first two lists contain identical items and have identical")
print("precision. NDCG separates them, because users read from the top.")

# --- eq. 30.9: the popularity feedback loop ---------------------------------
print("\n" + "=" * 72)
print("the feedback loop: exposure determines the next training set")
print("=" * 72)

n_items = 200
true_quality = rng.beta(2, 5, n_items)          # intrinsic click propensity
true_quality /= true_quality.sum()


def simulate(rounds=40, users_per_round=2000, explore=0.0, seed=1):
    r = np.random.default_rng(seed)
    interactions = np.ones(n_items)             # weak uniform prior
    exposure_history = []
    for _ in range(rounds):
        popularity = interactions / interactions.sum()
        # epsilon-greedy: mostly recommend by popularity, sometimes explore
        probs = (1 - explore) * popularity + explore / n_items
        shown = r.choice(n_items, size=users_per_round, p=probs)
        clicked = r.random(users_per_round) < true_quality[shown] * 8
        np.add.at(interactions, shown[clicked], 1)
        exposure_history.append(np.bincount(shown, minlength=n_items))
    return interactions, np.array(exposure_history)


print(f"{'exploration':>12} {'top-10 exposure share':>23} "
      f"{'items never shown':>19} {'quality-exposure corr':>23}")
for explore in (0.0, 0.05, 0.20):
    inter, hist = simulate(explore=explore)
    final = hist[-1] / hist[-1].sum()
    top10 = np.sort(final)[-10:].sum()
    never = int((hist.sum(axis=0) == 0).sum())
    corr = np.corrcoef(true_quality, final)[0, 1]
    print(f"{explore:>12.0%} {top10:>23.1%} {never:>19} {corr:>23.3f}")

print("\nWith no exploration, ten items out of 200 take most of the exposure")
print("and many are never shown at all — so their quality is never learned.")
print("Exploration costs short-term clicks and buys a catalogue the system")
print("actually knows something about (eq. 30.9).")

# --- offline evaluation is biased toward the incumbent ----------------------
print("\n" + "=" * 72)
print("why offline evaluation flatters the model that generated the logs")
print("=" * 72)

# The logging policy favours the first 50 items; the "new" model favours others.
logging_pref = np.zeros(n_items)
logging_pref[:50] = 1.0
logging_probs = (logging_pref + 0.05) / (logging_pref + 0.05).sum()

n_log = 40_000
shown = rng.choice(n_items, size=n_log, p=logging_probs)
clicked = rng.random(n_log) < true_quality[shown] * 8
logged = {"item": shown, "click": clicked}

# Two candidate models: one mimics the logging policy, one ranks by true quality.
incumbent_scores = logging_probs
challenger_scores = true_quality

def offline_ctr(scores, top_n=40):
    """Estimate CTR from logs by restricting to this model's top items —
    the standard naive offline evaluation."""
    top = set(np.argsort(-scores)[:top_n].tolist())
    m = np.isin(logged["item"], list(top))
    return logged["click"][m].mean() if m.sum() > 30 else float("nan"), int(m.sum())


inc_ctr, inc_n = offline_ctr(incumbent_scores)
cha_ctr, cha_n = offline_ctr(challenger_scores)

# The truth: expected CTR if each model's top items were actually shown.
inc_true = true_quality[np.argsort(-incumbent_scores)[:40]].mean() * 8
cha_true = true_quality[np.argsort(-challenger_scores)[:40]].mean() * 8

print(f"{'model':<14} {'offline CTR':>13} {'logged rows':>13} "
      f"{'true CTR if deployed':>22}")
print(f"{'incumbent':<14} {inc_ctr:>13.4f} {inc_n:>13,} {inc_true:>22.4f}")
print(f"{'challenger':<14} {cha_ctr:>13.4f} {cha_n:>13,} {cha_true:>22.4f}")

print(f"\nThe challenger is genuinely better ({cha_true:.3f} vs {inc_true:.3f}")
print(f"true CTR) but is evaluated on only {cha_n:,} logged rows — the few")
print("times its preferred items happened to be shown. Offline evaluation")
print("has far less evidence about the model that differs from the logger,")
print("which is exactly the model you are trying to assess.")
print("\nThis is why recommenders are decided by A/B test (Chapter 26), and")
print("why logging the propensity of each impression is worth the effort.")
```

## 9. Common Mistakes

**Applying the SVD to a filled ratings matrix.** Fit the observed entries only.

**Omitting bias terms.** They capture most of the variance and are nearly free.

**No regularisation.** Users with few ratings get confident, wrong factors.

**Treating unobserved as negative in explicit-feedback settings.** Absence is
not dislike.

**Ignoring exposure in implicit feedback.** Not clicking may mean never seeing.

**No cold-start plan.** New items are a permanent condition.

**Optimising RMSE for a ranking task.** Users see the top of a list.

**Trusting offline metrics to choose between good models.** They favour the
incumbent; A/B test.

**Not measuring coverage or diversity.** An accurate recommender that shows
everyone the same ten items is a failure no accuracy metric detects.

**Never exploring.** Exposure share becomes self-reinforcing and quality is
never learned for most of the catalogue.

## 10. Connection to Previous Chapters

{{ch:math-eigen}} supplied the SVD and low-rank approximation that
{{sec:4-intuitive-explanation}} adapts and {{sec:6-mathematical-foundation}}
explains cannot be used directly. {{ch:math-optimization}} supplied the gradient
descent and the regularisation-as-prior argument behind
{{eq:mf-objective}}. {{ch:ds-collection}} supplied selection bias, which the
feedback loop generates internally. {{ch:ds-experiments}} supplied the A/B test
that offline evaluation cannot replace. {{ch:ds-timeseries}} established the
i.i.d.-violation theme this chapter completes.

Beyond Part III: {{part:11}} generalises {{eq:mf-basic}} into learned embeddings
and two-tower retrieval, where the inner product of a user vector and an item
vector becomes the inner product of a query vector and a document vector —
which is also, structurally, how attention scores a query against a key
({{ch:tf-scaled-dot-product}}). {{ch:rai-bias}} takes up the fairness
consequences of the amplification dynamics of {{eq:popularity-dynamics}}.
{{cite:koren2009}} is the reference account.

## 11. Exercises

**Beginner**

1. Why is a ratings matrix mostly empty, and what does that rule out?
2. Distinguish content-based from collaborative filtering, with a failure mode
   for each.
3. What do the bias terms in {{eq:mf-with-bias}} capture?
4. Name the three cold-start cases and a response to each.
5. Why is RMSE the wrong metric for a recommender?

**Intermediate**

6. Explain why the SVD cannot be applied directly, and why filling with the mean
   is not a fix.
7. In implicit feedback, why is a non-click ambiguous? Give two interpretations.
8. Compute NDCG@3 for relevance lists $[3,1,0]$ and $[0,1,3]$.
9. Explain the feedback loop and why it biases offline evaluation.
10. Why can a recommender maximising accuracy be a commercial failure?
11. Using {{eq:popularity-dynamics}}, explain why initial exposure has a
    permanent effect.

**Advanced**

12. Show that {{eq:masked-objective}} is non-convex jointly and convex in each
    factor separately, and explain why that licenses ALS.
13. Derive the SGD updates {{eq:mf-update-p}} and {{eq:mf-update-q}} from
    {{eq:mf-objective}}.
14. Explain why {{eq:implicit-objective}} sums over all pairs and what makes it
    computationally feasible.
15. Design an offline evaluation using logged propensities that partly corrects
    for exposure bias. State the assumption it requires.
16. Relate {{eq:mf-basic}} to the two-tower retrieval architecture and to the
    query-key inner product of attention. What is the same and what differs?

**Implementation**

17. Implement ALS and compare its convergence against SGD on the same data.
18. Add implicit-feedback confidence weighting and evaluate with ranking metrics
    rather than RMSE.
19. Implement NDCG@k, MAP@k and coverage, and evaluate a popularity baseline
    against your factorisation on all three.
20. Extend the feedback simulation with a bandit exploration policy and measure
    the trade-off between short-term clicks and long-run catalogue coverage.

**Reasoning**

21. Exploration costs short-term engagement. How would you justify it to someone
    accountable for this quarter's numbers?
22. Offline and online results disagree. Why is the online result right, and
    what is offline evaluation still good for?

## 12. Chapter Summary

Recommendation is a sparse-matrix completion and ranking problem, not standard
supervised learning: over 99% of entries are missing, and the missingness is
determined by what the system chose to show.

Collaborative filtering uses interaction patterns alone and outperforms
content-based methods despite knowing nothing about the items, because the
interaction matrix encodes similarity judgements no feasible attribute set
captures. Its weakness is cold start, which is permanent rather than transient.

Matrix factorisation represents users and items as vectors in a shared
low-dimensional space and predicts an interaction as their inner product, with
bias terms capturing the large uninteresting variation. This is the first
appearance of learned representations replacing engineered features, and it is
the direct ancestor of two-tower retrieval.

The SVD cannot be applied directly, because it requires a complete matrix and
filling the gaps asserts values that were never observed. Fitting only the
observed entries gives a non-convex problem with no closed form, solved by
gradient descent or by alternating least squares — which works because the
objective is convex in each factor given the other.

Implicit feedback has no negatives and its values are confidence rather than
preference. The standard treatment separates a binary preference from a
confidence weight and sums over all pairs, treating unobserved as weak negatives.

The feedback loop is the structural problem: the model determines exposure,
exposure determines the logged interactions, and those train the next model.
This amplifies popularity multiplicatively, makes initial exposure permanently
consequential, and biases offline evaluation toward the incumbent — which is why
recommenders must be decided by online experiment, and why deliberate
exploration is worth its short-term cost.
