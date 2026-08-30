# -*- coding: utf-8 -*-
# Extracted from: Chapter 133 — Dataset Creation for Fine-Tuning
# Source: src/.../ch133-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
