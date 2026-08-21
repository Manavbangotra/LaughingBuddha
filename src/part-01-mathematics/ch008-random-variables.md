---
id: math-random-vars
number: 8
part: I
tier: focused
status: reviewed
requires: [math-probability]
provides: [random-variable, probability-distribution, expectation,
           gaussian-distribution, bernoulli-distribution,
           categorical-distribution]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define a random variable as a function on the sample space, and explain why
   that definition is not pedantry.
2. Distinguish probability mass functions from densities, and explain why a
   density may exceed 1.
3. Compute expectations for discrete and continuous random variables.
4. Use the linearity of expectation, including in cases where independence
   fails.
5. Recognise the Bernoulli, categorical, uniform, and Gaussian distributions and
   state where each appears in machine learning.
6. Explain what the law of the unconscious statistician says and why it saves
   work.
7. Explain why a model's output layer is a categorical distribution and what
   that implies for its loss.

## 2. Why This Matters

{{ch:math-probability}} dealt with events — things that either happen or do not.
Most quantities of interest are not like that. A model's loss is a number. A
token's probability is a number. The time a request takes is a number. To reason
about these you need random *variables*, and the machinery of distributions and
expectations that comes with them.

Three specific reasons this chapter earns its place.

**Every model output is a distribution.** A classifier's output layer produces a
categorical distribution over classes; a language model produces one over the
vocabulary, at every position, for every token generated. The softmax
({{ch:math-functions}}) exists to construct exactly this object, and the
cross-entropy loss exists to score it.

**Every loss is an expectation.** Training minimises the expected loss over the
data distribution, approximated by an average over a sample. The gap between the
expectation you want and the average you can compute is what
{{ch:math-inference}} is about, and it is the origin of overfitting.

**Expectation is linear, and that is used constantly, often without comment.**
$\E[X + Y] = \E[X] + \E[Y]$ holds whether or not $X$ and $Y$ are independent.
This unremarkable-looking fact is what makes minibatch gradients unbiased
estimates of the full gradient ({{ch:math-optimization}}), which is the entire
justification for stochastic gradient descent.

## 3. Prerequisites

{{ch:math-probability}} for probability, conditioning and independence.
{{ch:math-functions}} for the exponential, used in the Gaussian density.
{{ch:math-notation}} for summation and indicator functions.

## 4. Intuitive Explanation

### 4.1 A random variable is a function, not a variable

The name is unhelpful. A {{term:random-variable}} is neither random nor a
variable in the algebraic sense — it is a **function** that assigns a number to
each outcome in the sample space.

Roll two dice. The sample space is the 36 ordered pairs. Define $X$ = the sum.
Then $X$ is a function: $X((3,4)) = 7$. The randomness lives in *which outcome
occurs*; $X$ itself is a perfectly deterministic rule.

This is worth taking seriously rather than filing away, because it explains why
you can do arithmetic with random variables at all. If $X$ and $Y$ are both
functions on the same sample space, then $X + Y$ is just another function, and
so is $X^{2}$ or $\log X$. The algebra of random variables is the algebra of
functions, which is why expressions like $\E[X + Y]$ make sense before you know
anything about dependence.

By convention, capital letters denote the random variable and lowercase its
realised value: $\Prob(X = x)$ reads "the probability that the random variable
$X$ takes the value $x$".

### 4.2 Mass versus density

Discrete and continuous random variables need different machinery, and the
difference trips people up.

For a **discrete** variable, the probability mass function gives the probability
of each value directly: $p(x) = \Prob(X = x)$, and these sum to 1.

For a **continuous** variable, the probability of any single exact value is
**zero**. The probability that a measurement is exactly 1.7 metres — not
1.70001, exactly 1.7 — is zero. What is meaningful is the probability of an
interval, obtained by integrating a **density**.

The consequence that surprises people: **a density can exceed 1.** A uniform
distribution on $[0, 0.1]$ has density 10 everywhere on that interval, because
$10 \times 0.1 = 1$. A density is probability *per unit length*, not
probability, and only its integral is constrained.

> IMPORTANT: This distinction matters in practice. A model that outputs a
> density — a continuous-valued generative model, or a mixture-density network —
> can report log-likelihoods that are *positive*, which looks like a bug and is
> not. Densities are unbounded above; only probabilities are bounded by 1.

### 4.3 Expectation is a weighted average

The {{term:expectation}} of a random variable is the average value it takes,
weighting each possibility by how likely it is:

$$
\E[X] = \sum_{x} x\,p(x)
$$ (eq:expectation-discrete)

For a fair die, $\E[X] = \frac{1}{6}(1+2+3+4+5+6) = 3.5$.

Note that 3.5 is not a value the die can show. An expectation is a summary of
the distribution, not a prediction of any particular outcome, and treating it as
a prediction is a real source of error — the expected number of retries before
success might be 2.3, but no individual request retries 2.3 times.

### 4.4 The four distributions you need first

{#tbl:distributions caption="The distributions that account for most of what this book needs, and where each appears."}

| Distribution | Models | Parameters | Where it appears |
|---|---|---|---|
| Bernoulli | one yes/no trial | $p$ | binary classification output |
| Categorical | one draw from $k$ options | $p_1, \ldots, p_k$ | every language model output |
| Uniform | equal chance over a range | $a, b$ | initialisation, random sampling |
| Gaussian | sums of many small effects | $\mu, \sigma^{2}$ | noise, initialisation, priors |

The {{term:bernoulli-distribution}} and {{term:categorical-distribution}} are the
output formats of classification. The {{term:gaussian-distribution}} is
everywhere else, and the reason it is everywhere is the central limit theorem
({{ch:math-inference}}) rather than any preference for elegance.

## 5. Formal Explanation

### 5.1 Definitions

A random variable is a function $X: \Omega \to \R$. It is **discrete** if its
range is countable, **continuous** if it has a density.

For discrete $X$, the **probability mass function** is

$$
p(x) = \Prob(X = x), \qquad p(x) \ge 0, \qquad \sum_{x} p(x) = 1
$$ (eq:pmf)

For continuous $X$, the **probability density function** satisfies

$$
\Prob(a \le X \le b) = \int_{a}^{b} p(x)\,\dd x,
\qquad p(x) \ge 0, \qquad \int_{-\infty}^{\infty} p(x)\,\dd x = 1
$$ (eq:pdf)

The **cumulative distribution function**, defined for both cases, is

$$
F(x) = \Prob(X \le x)
$$ (eq:cdf)

It is non-decreasing, tends to 0 at $-\infty$ and 1 at $+\infty$, and for
continuous variables $F'(x) = p(x)$.

### 5.2 Expectation

$$
\E[X] = \sum_{x} x\,p(x)
\qquad\text{(discrete)}, \qquad
\E[X] = \int_{-\infty}^{\infty} x\,p(x)\,\dd x
\qquad\text{(continuous)}
$$ (eq:expectation)

The expectation exists when the sum or integral converges absolutely. It need
not: the Cauchy distribution has no mean, which is not a pathology invented for
textbooks — heavy-tailed quantities in real systems, such as request latencies,
can have such badly behaved sample means that averaging them is misleading.

**Law of the unconscious statistician.** To compute the expectation of a
function of $X$, you do not need the distribution of $g(X)$:

$$
\E[g(X)] = \sum_{x} g(x)\,p(x)
$$ (eq:lotus)

This saves a great deal of work. Computing $\E[X^{2}]$ does not require deriving
the distribution of $X^{2}$; you weight $x^{2}$ by $p(x)$ and sum. Nearly every
expectation in machine learning is computed this way.

### 5.3 Linearity, and what it does not require

For any random variables $X, Y$ on the same space and any constants $a, b$:

$$
\E[aX + bY] = a\,\E[X] + b\,\E[Y]
$$ (eq:linearity-expectation)

**This holds whether or not $X$ and $Y$ are independent.** That is the crucial
point, and it makes expectation unusually well behaved — the corresponding
statement for variance is false without independence
({{ch:math-covariance}}).

Linearity follows directly from the linearity of summation
({{eq:sum-linearity}} in {{ch:math-notation}}), which is why that identity was
worth stating there.

For independent $X$ and $Y$ only:

$$
\E[XY] = \E[X]\,\E[Y]
\qquad\text{(requires independence)}
$$ (eq:expectation-product)

> IMPORTANT: The asymmetry between {{eq:linearity-expectation}} and
> {{eq:expectation-product}} is the single most useful thing to remember about
> expectations. Sums are always well behaved; products are not. When a
> derivation seems to need independence, check whether it actually does — often
> it is only using linearity and the assumption is unnecessary.

### 5.4 The distributions

**Bernoulli.** $X \in \{0, 1\}$ with $\Prob(X = 1) = p$:

$$
p(x) = p^{x}(1-p)^{1-x}, \qquad \E[X] = p, \qquad \Var(X) = p(1-p)
$$ (eq:bernoulli)

Writing the mass function as $p^{x}(1-p)^{1-x}$ is a small trick worth
recognising: the exponents act as switches, selecting $p$ when $x = 1$ and
$1-p$ when $x = 0$. Taking its log gives $x\log p + (1-x)\log(1-p)$, which is
exactly the binary cross-entropy of {{ch:math-functions}} — the loss *is* the
negative log of this mass function.

**Categorical.** $X \in \{1, \ldots, k\}$ with $\Prob(X = i) = p_i$, where
$p_i \ge 0$ and $\sum_i p_i = 1$. The parameter vector lives on the probability
simplex, which is exactly the range of the softmax. A language model's output at
each position is a categorical distribution over the vocabulary, typically with
$k$ in the tens or hundreds of thousands.

**Uniform.** Continuous on $[a, b]$:

$$
p(x) = \frac{1}{b-a} \text{ for } x \in [a,b],
\qquad \E[X] = \frac{a+b}{2},
\qquad \Var(X) = \frac{(b-a)^{2}}{12}
$$ (eq:uniform)

**Gaussian.** With mean $\mu$ and variance $\sigma^{2}$:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\exp\!\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)
$$ (eq:gaussian)

Symmetric about $\mu$, with about 68% of mass within one standard deviation,
95% within two, and 99.7% within three. The **standard normal** has $\mu = 0$,
$\sigma = 1$, and any Gaussian can be standardised via $Z = (X - \mu)/\sigma$
({{ch:math-covariance}}).

> MATH NOTE: The Gaussian's dominance is not aesthetic. The central limit
> theorem ({{ch:math-inference}}) says that sums of many independent
> contributions tend toward it regardless of their individual distributions —
> so anything that is the accumulation of many small effects is approximately
> Gaussian. Measurement noise, initialisation schemes, and the sampling
> distribution of a mean all qualify. Where a quantity is *not* such an
> accumulation — a distribution of file sizes, or request latencies — assuming
> normality is a mistake, and usually an expensive one.

### 5.5 Joint, marginal and conditional distributions

For two random variables, the **joint** distribution $p(x, y)$ gives the
probability of each combination. **Marginals** are recovered by summing out:

$$
p(x) = \sum_{y} p(x, y), \qquad p(y) = \sum_{x} p(x, y)
$$ (eq:marginal)

and the **conditional** is

$$
p(y \given x) = \frac{p(x, y)}{p(x)}
$$ (eq:conditional-dist)

Every supervised model estimates $p(y \given \vec{x})$ — that is what supervised
learning *is*, stated in one line. A generative model estimates the joint
$p(\vec{x}, y)$, or sometimes $p(\vec{x})$ alone, which is a strictly harder
problem and is why generative models need far more data
({{ch:ml-what-it-is}}).

## 6. Mathematical Foundation

### 6.1 Deriving the Bernoulli mean and variance

Directly from {{eq:expectation-discrete}} with two outcomes:

$$
\E[X] = 0 \cdot (1-p) + 1 \cdot p = p
$$ (eq:bernoulli-mean)

For the variance, use $\Var(X) = \E[X^{2}] - \E[X]^{2}$ (derived in
{{ch:math-covariance}}) and note that $X^{2} = X$ when $X \in \{0,1\}$, so
$\E[X^{2}] = p$:

$$
\Var(X) = p - p^{2} = p(1-p)
$$ (eq:bernoulli-variance)

This is maximised at $p = 0.5$, where it equals 0.25, and vanishes at $p = 0$ or
$p = 1$. That makes sense: a coin that always lands the same way has no
variability. It also has a practical consequence — a classifier is most
uncertain, and most informative to label, when its output is near 0.5, which is
the basis of uncertainty sampling in active learning.

### 6.2 Why linearity does not need independence

The proof is short and worth seeing, because the result is used so often.

$$
\E[X + Y] = \sum_{x}\sum_{y}(x + y)\,p(x, y)
$$

Split the sum:

$$
= \sum_{x}\sum_{y} x\,p(x,y) + \sum_{x}\sum_{y} y\,p(x,y)
$$

In the first term, $x$ does not depend on $y$, so it factors out of the inner
sum, and $\sum_{y}p(x,y) = p(x)$ by {{eq:marginal}}. Similarly for the second:

$$
= \sum_{x} x\,p(x) + \sum_{y} y\,p(y) = \E[X] + \E[Y]
$$

Nowhere did the argument require $p(x,y) = p(x)p(y)$. Only marginalisation was
used, and marginalisation always works.

> MATH NOTE: This is exactly why minibatch gradient descent is justified. The
> full gradient is $\nabla\Loss = \frac{1}{N}\sum_i \nabla\ell_i$, and the
> minibatch gradient averages a random subset. By linearity, the expectation of
> the minibatch gradient equals the full gradient — it is *unbiased* — even
> though individual examples are certainly not independent of one another in any
> useful sense. The estimate is noisy but not systematically wrong, and that is
> the whole argument for SGD ({{ch:math-optimization}}).

### 6.3 A worked example: expected loss

Suppose a binary classifier outputs probability $\hat{p}$ for class 1, and the
true label is Bernoulli with parameter $q$. What is the expected cross-entropy
loss?

The loss for a given label is $\ell(y) = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$.
Applying {{eq:lotus}}:

$$
\E[\ell] = q\big(-\log\hat{p}\big) + (1-q)\big(-\log(1-\hat{p})\big)
$$ (eq:expected-ce)

Take $q = 0.7$ and compare two predictions.

With $\hat{p} = 0.7$ (well calibrated):

$$
\E[\ell] = 0.7(-\log 0.7) + 0.3(-\log 0.3) = 0.7(0.3567) + 0.3(1.2040) = 0.6108
$$

With $\hat{p} = 0.9$ (overconfident):

$$
\E[\ell] = 0.7(-\log 0.9) + 0.3(-\log 0.1) = 0.7(0.1054) + 0.3(2.3026) = 0.7645
$$

The calibrated prediction has lower expected loss. This is not a coincidence:
{{eq:expected-ce}} is minimised exactly at $\hat{p} = q$, which is the sense in
which cross-entropy is a **proper scoring rule** — it rewards honest reporting
of your actual belief, and there is no way to game it by shading your
predictions. Exercise 12 asks you to prove it.

The minimum value, $0.6108$, is the entropy of the true distribution. No model,
however good, can beat it. Any irreducible uncertainty in the labels puts a
floor under the achievable loss — which is worth remembering the next time a
training loss plateaus above zero and it looks like a failure.

## 7. Implementation

```python {tier=A name=random-variables}
"""Random variables, distributions, and expectation — verified by simulation.

Every analytic formula in the chapter is checked against a large sample.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 500_000

# --- a random variable is a function on the sample space --------------------
d1, d2 = rng.integers(1, 7, N), rng.integers(1, 7, N)
X = d1 + d2                                  # X: Omega -> R, the sum

print(f"E[X] simulated : {X.mean():.4f}")
print(f"E[X] analytic  : {7.0}   (3.5 + 3.5 by linearity)")
values, counts = np.unique(X, return_counts=True)
pmf = counts / N
print(f"pmf sums to    : {pmf.sum():.6f}")
print(f"E[X] from pmf  : {(values * pmf).sum():.4f}")

# --- eq. 8.7: linearity of expectation does NOT need independence -----------
# Deliberately dependent: Y is a function of X.
Y = X ** 2
lhs = (3 * X + 2 * Y).mean()
rhs = 3 * X.mean() + 2 * Y.mean()
print(f"\nE[3X + 2Y] = {lhs:.3f}, 3E[X] + 2E[Y] = {rhs:.3f}  -> equal")
assert np.isclose(lhs, rhs, rtol=1e-9)

# But E[XY] = E[X]E[Y] fails badly for dependent variables (eq. 8.8).
print(f"E[XY]      = {(X * Y).mean():.2f}")
print(f"E[X]E[Y]   = {X.mean() * Y.mean():.2f}   <- not equal; X and Y depend")

ind_a, ind_b = rng.normal(size=N), rng.normal(size=N)
print(f"independent: E[AB] = {(ind_a*ind_b).mean():+.4f}, "
      f"E[A]E[B] = {ind_a.mean()*ind_b.mean():+.4f}   -> equal")

# --- eq. 8.9: Bernoulli mean and variance -----------------------------------
print(f"\n{'p':>5} {'E[X] sim':>10} {'E[X] = p':>10} {'Var sim':>10} "
      f"{'Var = p(1-p)':>13}")
for p in (0.1, 0.3, 0.5, 0.9):
    s = (rng.random(N) < p).astype(float)
    print(f"{p:>5} {s.mean():>10.4f} {p:>10.4f} {s.var():>10.4f} "
          f"{p*(1-p):>13.4f}")
print("variance peaks at p = 0.5 — maximum uncertainty")

# --- densities can exceed 1 -------------------------------------------------
# Uniform on [0, 0.1] has density 10 everywhere on that interval.
u = rng.uniform(0, 0.1, N)
hist, edges = np.histogram(u, bins=10, range=(0, 0.1), density=True)
print(f"\nuniform[0, 0.1] density estimate: {hist.mean():.2f}  <- above 1")
print(f"but it integrates to {(hist * np.diff(edges)).sum():.4f}")
print("A density is probability PER UNIT, not probability.")

# --- Gaussian: the 68-95-99.7 rule ------------------------------------------
g = rng.normal(0.0, 1.0, N)
for k in (1, 2, 3):
    print(f"within {k} sd: {np.mean(np.abs(g) < k):.4f}")

# --- eq. 8.14: the law of the unconscious statistician ----------------------
# E[g(X)] computed from p(x) directly, without deriving the law of g(X).
values, counts = np.unique(d1, return_counts=True)
p_x = counts / N
for name, g in (("X^2", lambda v: v**2), ("log X", np.log),
                ("1/X", lambda v: 1.0 / v)):
    lotus = (g(values) * p_x).sum()
    direct = g(d1).mean()
    print(f"\nE[{name:<5}] via LOTUS: {lotus:.5f} | direct average: {direct:.5f}")
    assert np.isclose(lotus, direct, rtol=1e-3)

# --- eq. 8.19: cross-entropy is a proper scoring rule -----------------------
q = 0.7                                      # true probability of class 1


def expected_ce(p_hat, q=q):
    return -(q * np.log(p_hat) + (1 - q) * np.log(1 - p_hat))


print(f"\ntrue class-1 probability q = {q}")
print(f"{'prediction':>11} {'expected loss':>15}")
for p_hat in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
    marker = "  <- minimum, at p_hat = q" if np.isclose(p_hat, q) else ""
    print(f"{p_hat:>11.2f} {expected_ce(p_hat):>15.4f}{marker}")

grid = np.linspace(0.01, 0.99, 9999)
best = grid[np.argmin(expected_ce(grid))]
print(f"\nnumerical minimiser: {best:.4f}  (true q = {q})")
assert abs(best - q) < 0.001

entropy = expected_ce(q)
print(f"minimum achievable loss = entropy of the labels = {entropy:.4f}")
print("No model can do better. Irreducible label noise floors the loss.")

# --- the minibatch gradient is unbiased (section 6.2) -----------------------
per_example_grads = rng.normal(loc=2.0, scale=5.0, size=10_000)
full_gradient = per_example_grads.mean()
batch_means = [rng.choice(per_example_grads, size=32, replace=False).mean()
               for _ in range(3000)]
print(f"\nfull-data gradient        : {full_gradient:.4f}")
print(f"mean of 3000 minibatches  : {np.mean(batch_means):.4f}  <- unbiased")
print(f"sd of a single minibatch  : {np.std(batch_means):.4f}   <- noisy")
print("Noisy but not systematically wrong. That is the entire case for SGD.")
```

## 8. Practical Example

A language model's output layer is a categorical distribution, and every
decoding strategy is a way of turning that distribution into a token. Seeing the
distribution explicitly makes the decoding choices in {{ch:llm-decoding}} much
less mysterious.

```python {tier=A name=categorical-output}
"""The output of a language model is a categorical distribution.

Sampling strategies are transformations of that distribution before drawing
from it. Here the effects are measured rather than described.
"""
import numpy as np

rng = np.random.default_rng(7)

vocab = ["the", "a", "cat", "dog", "runs", "sleeps", "quantum", "purple"]
logits = np.array([3.2, 2.8, 1.5, 1.2, 0.4, 0.1, -2.0, -3.5])


def softmax(z, temperature=1.0):
    z = z / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


probs = softmax(logits)
print(f"{'token':<10} {'logit':>7} {'p':>8}")
for t, lg, p in zip(vocab, logits, probs):
    print(f"{t:<10} {lg:>7.1f} {p:>8.4f}")
print(f"{'sum':<10} {'':>7} {probs.sum():>8.4f}  <- a categorical distribution")


def entropy(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


# Temperature reshapes the distribution before sampling (Chapter 90).
print(f"\n{'temperature':>12} {'entropy':>9} {'max p':>8} "
      f"{'effective choices':>18}")
for tau in (0.2, 0.5, 1.0, 1.5, 3.0):
    p = softmax(logits, tau)
    h = entropy(p)
    print(f"{tau:>12.1f} {h:>9.4f} {p.max():>8.4f} {np.exp(h):>18.2f}")
print("exp(entropy) is the 'perplexity' — roughly how many tokens are")
print("genuinely in play. Low temperature narrows it toward 1.")

# Top-k truncation: keep k tokens, renormalise. This is conditioning (Ch. 7)
# on the event 'the token is one of these k'.
def top_k(p, k):
    out = np.zeros_like(p)
    idx = np.argsort(-p)[:k]
    out[idx] = p[idx]
    return out / out.sum()


print(f"\n{'k':>4} {'kept':>34} {'P(nonsense) removed':>21}")
nonsense = {"quantum", "purple"}
nonsense_idx = [vocab.index(w) for w in nonsense]
for k in (1, 2, 4, 8):
    p = top_k(probs, k)
    kept = ", ".join(vocab[i] for i in np.argsort(-probs)[:k])
    removed = probs[nonsense_idx].sum() - p[nonsense_idx].sum()
    print(f"{k:>4} {kept[:34]:>34} {removed:>21.5f}")

# Empirical draws converge to the distribution — the law of large numbers
# (Chapter 10), and the reason a sampled model is consistent in aggregate.
draws = rng.choice(len(vocab), size=200_000, p=probs)
emp = np.bincount(draws, minlength=len(vocab)) / len(draws)
print(f"\n{'token':<10} {'analytic p':>11} {'empirical':>11} {'abs diff':>10}")
for i, t in enumerate(vocab):
    print(f"{t:<10} {probs[i]:>11.4f} {emp[i]:>11.4f} "
          f"{abs(probs[i]-emp[i]):>10.5f}")
assert np.allclose(probs, emp, atol=0.01)
```

## 9. Common Mistakes

**Treating a density as a probability.** Densities can exceed 1. Only integrals
of densities are probabilities, and only probabilities are bounded.

**Assuming $\E[g(X)] = g(\E[X])$.** False in general. For convex $g$, Jensen's
inequality gives $\E[g(X)] \ge g(\E[X])$. Concretely,
$\E[1/X] \neq 1/\E[X]$, which is why averaging rates and averaging times give
different answers.

**Requiring independence for linearity of expectation.** It is not needed.
Requiring it unnecessarily blocks perfectly valid arguments.

**Forgetting that $\E[XY] = \E[X]\E[Y]$ *does* require independence.** The
converse of the previous mistake, and equally common.

**Assuming everything is Gaussian.** Latencies, file sizes, wealth, word
frequencies and node degrees are not. Applying Gaussian reasoning to a
heavy-tailed quantity underestimates extreme events, sometimes by orders of
magnitude.

**Reporting the mean of a heavy-tailed sample.** For distributions with infinite
or very large variance, the sample mean is unstable and can be dominated by a
single observation. Report a median and a high quantile instead.

**Confusing the expectation with a typical value.** For a skewed distribution the
mean can be far from anything you would ever observe.

**Forgetting that irreducible noise floors the loss.** A training loss plateauing
above zero is often correct behaviour, not a bug. Compare against the label
entropy before concluding the model is underfitting.

## 10. Connection to Previous Chapters

{{ch:math-probability}} supplied probability, conditioning and independence;
this chapter attaches numbers to outcomes and defines expectation over them. The
conditional distribution {{eq:conditional-dist}} is {{eq:conditional}} for random
variables, and {{eq:marginal}} is the law of total probability.
{{ch:math-functions}} supplied the exponential in {{eq:gaussian}}, and its
Bernoulli log-likelihood is exactly the binary cross-entropy derived there.
{{ch:math-notation}}'s linearity of summation is what proves
{{eq:linearity-expectation}}.

Forward: {{ch:math-covariance}} defines variance and covariance as expectations
of specific functions, and shows that variance — unlike expectation — needs
independence to add. {{ch:math-inference}} treats sample statistics as random
variables with distributions of their own, which is the key idea of statistical
inference.

Beyond Part I: {{ch:ml-metrics}} scores predicted distributions;
{{ch:llm-decoding}} manipulates the categorical distribution of
{{sec:8-practical-example}}; {{ch:math-optimization}} minimises an expected loss;
and {{ch:dl-initialization}} chooses distributions for initial weights on the
basis of their variance.

## 11. Exercises

**Beginner**

1. A fair four-sided die is rolled. Write the pmf of the outcome and compute its
   expectation.
2. $X$ is Bernoulli with $p = 0.3$. Give $\E[X]$ and $\Var(X)$.
3. For $X \sim \mathcal{N}(10, 4)$, what is the standard deviation, and roughly
   what fraction of draws lie in $[6, 14]$?
4. A uniform distribution on $[0, 0.2]$ has what density? Explain why it exceeds
   1.
5. Give the expectation of the sum of two fair dice, using linearity rather than
   enumerating 36 outcomes.

**Intermediate**

6. Prove $\E[aX + b] = a\E[X] + b$ from {{eq:expectation-discrete}}.
7. $X$ takes values 1, 2, 3 with probabilities 0.2, 0.5, 0.3. Compute $\E[X]$,
   $\E[X^{2}]$, and $\E[1/X]$ using {{eq:lotus}}.
8. Show that $\E[1/X] \neq 1/\E[X]$ for the variable in Exercise 7, and explain
   the direction of the discrepancy using Jensen's inequality.
9. Verify {{eq:expected-ce}} for $q = 0.7$ and $\hat{p} = 0.5$, and confirm it
   exceeds the value at $\hat{p} = 0.7$.
10. A language model assigns probabilities $[0.5, 0.3, 0.15, 0.05]$ to four
    tokens. Compute the entropy and $\exp(\text{entropy})$, and interpret the
    second number.

**Advanced**

11. Prove {{eq:linearity-expectation}} for continuous random variables.
12. Prove that {{eq:expected-ce}} is minimised at $\hat{p} = q$ by
    differentiating with respect to $\hat{p}$. This establishes that
    cross-entropy is a proper scoring rule.
13. Show that for a Bernoulli variable, $\E[X^{n}] = p$ for every $n \ge 1$.
    What does that imply about all its moments?
14. Derive the mean and variance of the uniform distribution on $[a, b]$ by
    integration.
15. Give a distribution with a finite mean but infinite variance, and explain
    what goes wrong when you try to construct a confidence interval for its
    mean.

**Implementation**

16. Simulate the sum of $n$ independent uniform variables for
    $n \in \{1, 2, 5, 30\}$ and plot the histograms. Describe what happens as
    $n$ grows, and name the theorem.
17. Verify {{eq:lotus}} numerically for three different functions $g$ on a
    discrete distribution of your choice.
18. Implement top-p (nucleus) sampling and compare its effective vocabulary size
    against top-k across a range of temperatures.
19. Estimate $\E[1/X]$ for $X$ uniform on $[0.01, 1]$ by simulation, and compare
    with $1/\E[X]$. Explain the gap.

**Reasoning**

20. Why is a model's output layer a categorical distribution rather than a
    single predicted class? What is gained, and what does it cost?
21. Your training loss plateaus at 0.31 and refuses to fall further. Give two
    fundamentally different explanations and describe how you would tell them
    apart.

## 12. Chapter Summary

A random variable is a function from the sample space to the reals; the
randomness is in which outcome occurs, not in the function. Because random
variables are functions, arithmetic on them is just arithmetic on functions,
which is what makes expressions like $\E[X + Y]$ meaningful before any
assumption about dependence.

Discrete variables have probability mass functions summing to 1; continuous ones
have densities integrating to 1. A density is probability per unit and may
exceed 1, which is why continuous log-likelihoods can be positive.

Expectation is the probability-weighted average, and the law of the unconscious
statistician lets you compute $\E[g(X)]$ from $p(x)$ without deriving the
distribution of $g(X)$.

Expectation is linear, and linearity does *not* require independence — a fact
that justifies minibatch gradients being unbiased estimates of the full gradient,
and hence justifies SGD. By contrast $\E[XY] = \E[X]\E[Y]$ *does* require
independence. Sums are always well behaved; products are not.

Four distributions cover most of what this book needs: Bernoulli for binary
outcomes, categorical for multi-class outputs including every language model
token, uniform for sampling and initialisation, and Gaussian for anything that
accumulates many small independent effects. The Gaussian's ubiquity follows from
the central limit theorem, not from convenience, and assuming it for
heavy-tailed quantities is a real and costly error.

Cross-entropy is a proper scoring rule: expected loss is minimised exactly when
the predicted probability equals the true one, so honest reporting is optimal.
Its minimum value is the entropy of the labels, which is an irreducible floor no
model can beat.
