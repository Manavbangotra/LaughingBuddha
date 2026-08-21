---
id: math-functions
number: 2
part: I
tier: focused
status: reviewed
requires: [math-notation]
provides: [function-term, function-composition, monotonic-function, logarithm,
           exponential-function, logistic-function]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what makes a rule a function, and identify rules that are not.
2. Compose functions, and explain why $f \circ g$ applies $g$ first.
3. Manipulate exponents and logarithms fluently using the identity laws.
4. Explain the three properties that make logarithms indispensable in machine
   learning: they turn products into sums, compress scale, and preserve
   ordering.
5. Derive the log-sum-exp trick and explain the numerical failure it prevents.
6. Define the logistic function, derive its derivative, and explain what
   saturation is and why it stops learning.
7. Explain what a monotonic transformation preserves, and why that lets you
   optimise a log-likelihood instead of a likelihood.

## 2. Why This Matters

A surprisingly small family of functions does almost all the work in machine
learning. Exponentials and logarithms are two of them, and the logistic curve —
which is built from both — is the third.

They are not there for decoration. Every probability a model produces passes
through an exponential; every loss you minimise passes through a logarithm.
When a model outputs `nan` during training, the cause is very often an
exponential that overflowed or a logarithm of zero. When gradients vanish and a
network stops learning, the cause is very often a logistic function that
saturated. These are not exotic failures — they are the two most common ways
training breaks, and both are properties of the functions in this chapter.

There is also a conceptual reason. Composition is the idea that makes deep
learning *deep*: a neural network is nothing but a long chain of simple
functions applied in sequence. Understanding composition here is what makes the
chain rule in {{ch:math-derivatives}} feel inevitable rather than arbitrary, and
backpropagation in {{ch:dl-backprop}} feel like bookkeeping rather than magic.

## 3. Prerequisites

{{ch:math-notation}}: function signatures, set and interval notation, and
summation. Secondary-school algebra — rearranging equations, and what a power
means.

## 4. Intuitive Explanation

### 4.1 A function is a reliable machine

A {{term:function-term}} is a machine with one slot and one output, whose
defining virtue is *reliability*: put the same thing in twice and you get the
same thing out twice.

That is the entire content of the formal definition, and it is worth dwelling on
because it is the part people skip. The rule "$y$ is a number whose square is
$x$" is **not** a function of $x$, because for $x = 9$ it offers both $3$ and
$-3$ and refuses to choose. The rule "$y = \sqrt{x}$, the non-negative root" *is*
a function, because the phrase "non-negative" resolves the ambiguity.

This matters practically. A model is a function from inputs to outputs. If the
same input can produce different outputs, you do not have a function — you have
something with hidden state or randomness, and debugging it is a categorically
different problem. Much of the discipline in {{ch:mle-reproducibility}} is about
forcing systems to behave like functions.

### 4.2 Composition, and why it reads backwards

Composition is chaining: take the output of one machine and feed it into the
next.

$$
(f \circ g)(x) = f(g(x))
$$ (eq:composition)

The notation reads right to left, which is a genuine and persistent nuisance.
$f \circ g$ means *$g$ first, then $f$*. The reason is that it mirrors the way
the expanded form $f(g(x))$ nests: $x$ goes into the innermost bracket, and you
work outwards.

A three-layer neural network is literally

$$
\text{model} = f_3 \circ f_2 \circ f_1
$$ (eq:network-composition)

Data enters at $f_1$ and leaves from $f_3$. Everything difficult about training
deep networks follows from this structure: because the layers are composed
rather than added, a change in $f_1$ affects everything downstream, and credit
for the final error must be traced backwards through the chain. That tracing is
the chain rule ({{ch:math-derivatives}}), and its systematic application is
backpropagation ({{ch:dl-backprop}}).

### 4.3 Exponentials grow at a rate equal to their size

An exponential function is what you get when a quantity's growth rate is
proportional to how much of it there already is. Compound interest, population
growth, and the number of parameters in frontier models over the last decade all
have this character.

The specific base $e \approx 2.71828$ is chosen not because it is natural in any
everyday sense but because it makes calculus clean: $e^x$ is the unique function
that is its own derivative. Every other exponential is $e^x$ in disguise, since
$b^x = e^{x \ln b}$.

The practically important fact about exponentials is how violently they grow.
$e^{10} \approx 22{,}026$. $e^{100} \approx 2.7 \times 10^{43}$. $e^{800}$
overflows a 64-bit float. This is why raw scores are exponentiated with care —
{{sec:6-mathematical-foundation}} shows how.

### 4.4 Logarithms are the undo button, and three other things

The {{term:logarithm}} inverts exponentiation: $\log_b(x)$ answers "what power
of $b$ gives $x$?" That is the definition, but it undersells why logarithms are
everywhere in this book. They do three separate jobs:

**They turn multiplication into addition.** $\log(ab) = \log a + \log b$. This
is not a convenience; it is what makes likelihoods computable. The probability
of a thousand independent observations is a product of a thousand numbers each
less than one, which underflows to exactly zero in floating point. Its logarithm
is a sum of a thousand negative numbers, which is perfectly well behaved.

**They compress scale.** The range from $10^{-9}$ to $10^{9}$ — eighteen orders
of magnitude — becomes $-20.7$ to $20.7$ under a natural log. Quantities that
span orders of magnitude are usually best reasoned about, plotted, and searched
over on a log scale, which is why learning rates are tuned as $10^{-3}$ versus
$10^{-4}$ rather than as $0.001$ versus $0.0001$.

**They preserve ordering.** $\log$ is strictly increasing, so if $a > b$ then
$\log a > \log b$. This is the property that lets you maximise a log-likelihood
instead of a likelihood and get the same answer — a substitution used so
routinely that it is rarely remarked on, and it is valid only because of
monotonicity.

## 5. Formal Explanation

### 5.1 Functions, domain, codomain, range

A function $f: X \to Y$ assigns to each $x \in X$ exactly one $y \in Y$. The set
$X$ is the **domain**, $Y$ the **codomain**, and

$$
\operatorname{range}(f) = \{f(x) : x \in X\} \subseteq Y
$$ (eq:range)

is the **range** — the values actually attained. The distinction matters: the
logistic function has codomain $\R$ if you declare it so, but its range is the
open interval $(0, 1)$.

A function is **injective** (one-to-one) if distinct inputs give distinct
outputs, **surjective** (onto) if its range is all of $Y$, and **bijective** if
both. Only bijections have inverses, which is why $\exp: \R \to (0,\infty)$ has
the inverse $\log$, but $x \mapsto x^2$ on all of $\R$ does not.

### 5.2 Composition

For $g: X \to Y$ and $f: Y \to Z$, the composition $f \circ g: X \to Z$ is
defined by {{eq:composition}}. Composition is **associative** —
$(f \circ g) \circ h = f \circ (g \circ h)$ — but **not commutative**:
$f \circ g \neq g \circ f$ in general.

A quick counterexample worth keeping: with $f(x) = x^2$ and $g(x) = x + 1$,
$(f \circ g)(2) = f(3) = 9$ while $(g \circ f)(2) = g(4) = 5$.

Associativity is what lets a deep network be grouped into blocks arbitrarily —
you may treat layers 1–4 as one function or four, and it makes no difference to
the result. Non-commutativity is why layer order matters.

### 5.3 Exponents

For $b > 0$:

$$
b^{m}b^{n} = b^{m+n}, \qquad
\frac{b^{m}}{b^{n}} = b^{m-n}, \qquad
(b^{m})^{n} = b^{mn}
$$ (eq:exponent-laws)

$$
b^{0} = 1, \qquad b^{-n} = \frac{1}{b^{n}}, \qquad b^{1/n} = \sqrt[n]{b}
$$ (eq:exponent-special)

The {{term:exponential-function}} is $\exp(x) = e^{x}$, with domain $\R$ and
range $(0, \infty)$. It is characterised by

$$
\frac{\dd}{\dd x}e^{x} = e^{x}, \qquad e^{0} = 1
$$ (eq:exp-derivative)

and it is *strictly positive everywhere* — a fact used constantly, since it is
why exponentiating a score guarantees a positive number that can be normalised
into a probability.

### 5.4 Logarithms

For $b > 0$, $b \neq 1$, and $x > 0$:

$$
y = \log_{b}(x) \iff b^{y} = x
$$ (eq:log-def)

The identity laws follow directly from {{eq:exponent-laws}}:

$$
\log(xy) = \log x + \log y, \qquad
\log\!\left(\frac{x}{y}\right) = \log x - \log y, \qquad
\log(x^{n}) = n\log x
$$ (eq:log-laws)

$$
\log_{b}(x) = \frac{\log_{c}(x)}{\log_{c}(b)}, \qquad
\log(1) = 0, \qquad
\log_{b}(b) = 1
$$ (eq:log-change-base)

> IMPORTANT: This book writes $\log$ for the **natural** logarithm, base $e$,
> throughout — as most machine learning literature does. Information-theoretic
> quantities are therefore in *nats* rather than bits unless a passage says
> otherwise. The conversion is one nat $= 1/\ln 2 \approx 1.4427$ bits. Where a
> result is conventionally quoted in bits, the book converts explicitly.

The domain restriction $x > 0$ is not a technicality. $\log(0) = -\infty$ and
$\log$ of a negative number is undefined over the reals, and both cases arise in
practice the instant a model assigns probability zero to something that happened.

### 5.5 Monotonicity, and why it licenses a substitution

A function is **monotonically increasing** if $x \le y \implies f(x) \le f(y)$,
and **strictly** so if the second inequality is strict. A
{{term:monotonic-function}} preserves ordering, and that has an immediate
consequence:

$$
\argmax_{x} f(x) = \argmax_{x} g(f(x)) \quad \text{for strictly increasing } g
$$ (eq:argmax-invariance)

The *location* of the maximum is unchanged; the *value* is not. Since $\log$ is
strictly increasing on $(0, \infty)$,

$$
\argmax_{\theta} \; p(\Data \given \theta)
  = \argmax_{\theta} \; \log p(\Data \given \theta)
$$ (eq:mle-log)

Maximum likelihood estimation is always performed on the log-likelihood, for the
numerical reasons of {{sec:4-intuitive-explanation}}, and {{eq:mle-log}} is the
licence to do so. It is worth noticing how much rests on a one-line property.

### 5.6 The logistic function

The {{term:logistic-function}}, also called the sigmoid, is

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$ (eq:sigmoid)

with domain $\R$ and range $(0, 1)$ — open at both ends. It is strictly
increasing, and it maps $0 \mapsto 0.5$, large positive inputs toward 1, and
large negative inputs toward 0.

Its useful identities:

$$
\sigma(-x) = 1 - \sigma(x), \qquad
\sigma^{-1}(p) = \log\!\left(\frac{p}{1-p}\right), \qquad
\sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big)
$$ (eq:sigmoid-identities)

The inverse is called the **logit** function, and it is the origin of the term
{{term:logit}} used throughout Parts VII onward: a logit is a value on the scale
where the logistic function is the way back to a probability. The middle
identity says that a logit is the log-odds — the log of the ratio of the
probability of an event to the probability of its complement.

## 6. Mathematical Foundation

### 6.1 Deriving the logistic derivative

{{eq:sigmoid-identities}}'s third identity is unusually elegant and is used in
{{ch:dl-backprop}}, so it is worth deriving rather than quoting.

Write $\sigma(x) = (1 + e^{-x})^{-1}$ and apply the chain rule (taken on faith
here; derived in {{ch:math-derivatives}}):

$$
\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot \frac{\dd}{\dd x}(1 + e^{-x})
$$

The inner derivative is $-e^{-x}$, so the two minus signs cancel:

$$
\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^{2}}
$$ (eq:sigmoid-deriv-raw)

Now split the fraction deliberately:

$$
\sigma'(x)
  = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}}
  = \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}}
$$

The remaining factor is $1 - \sigma(x)$, since

$$
1 - \frac{1}{1+e^{-x}} = \frac{(1+e^{-x}) - 1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}}
$$

giving

$$
\sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big)
$$ (eq:sigmoid-derivative)

> MATH NOTE: The practical significance of {{eq:sigmoid-derivative}} is that the
> derivative is expressible in terms of the output alone. A backward pass can
> reuse the value it already computed going forward, with no need to keep the
> input. That is a real memory saving, and it is one reason the sigmoid was
> popular long after better activations existed.

### 6.2 Saturation, and why it kills learning

{{eq:sigmoid-derivative}} has a maximum at $x = 0$, where
$\sigma = 0.5$ and $\sigma' = 0.25$. Away from the origin it collapses fast:

{#tbl:sigmoid-saturation caption="The logistic function saturates quickly. By an input of ±10 the derivative has fallen by four orders of magnitude from its peak."}

| $x$ | $\sigma(x)$ | $\sigma'(x)$ |
|---|---|---|
| 0 | 0.5000 | 0.2500 |
| 2 | 0.8808 | 0.1050 |
| 5 | 0.9933 | 0.0066 |
| 10 | 0.99995 | 0.0000454 |
| 20 | 0.9999999979 | $2.1 \times 10^{-9}$ |

Now recall {{eq:network-composition}}: a deep network is a composition, and by
the chain rule its gradient is a *product* of per-layer derivatives. Stack ten
sigmoid layers, each contributing at most $0.25$, and the gradient reaching the
first layer is at most $0.25^{10} \approx 10^{-6}$ of the signal — and that is
the *best* case, at the peak of the derivative. In the saturated regime it is
astronomically smaller.

This is the vanishing-gradient problem. It is not a subtle phenomenon or a
tuning issue; it is arithmetic, visible directly in
{{tbl:sigmoid-saturation}}. It is why ReLU displaced the sigmoid for hidden
layers ({{ch:dl-activations}}), why residual connections exist
({{ch:tf-ffn-residual}}), and — in a different guise — why attention scores are
scaled by $1/\sqrt{d_k}$ ({{ch:tf-scaled-dot-product}}). The same failure recurs
throughout the book in different costumes, and this table is where you meet it
first.

### 6.3 The log-sum-exp trick

Here is a concrete numerical failure and its standard fix.

You need $\log \sum_{i} e^{z_i}$ — a quantity that appears in the denominator of
every softmax, and hence in every classification loss. With $z = [1000, 1001,
1002]$, computing $e^{1000}$ directly overflows: the largest 64-bit float is
about $1.8 \times 10^{308}$, and $e^{710}$ already exceeds it. The naive
computation returns `inf`, and every downstream number becomes `nan`.

The fix rests on an exact identity. For any constant $c$:

$$
\log\sum_{i} e^{z_i}
  = \log\sum_{i} e^{z_i - c}e^{c}
  = \log\left(e^{c}\sum_{i} e^{z_i - c}\right)
  = c + \log\sum_{i} e^{z_i - c}
$$ (eq:logsumexp)

This holds for *every* $c$, so choose $c = \max_i z_i$. Then the largest
exponent is exactly $e^{0} = 1$, nothing can overflow, and the smallest terms
underflow harmlessly to zero — contributing nothing they would not have
contributed anyway.

For $z = [1000, 1001, 1002]$: take $c = 1002$, so the shifted values are
$[-2, -1, 0]$, whose exponentials are $[0.135, 0.368, 1.0]$ summing to $1.503$.
Then $\log(1.503) = 0.4076$, and the answer is $1002.4076$.

> PRODUCTION TIP: Never write `np.log(np.sum(np.exp(z)))`. Use
> `scipy.special.logsumexp`, or subtract the max yourself. The same reasoning
> underlies the max-subtraction inside every stable softmax implementation,
> including the one in {{ch:tf-scaled-dot-product}}. This is the single most
> common source of `nan` in hand-written model code.

### 6.4 Why exponentiate to get probabilities

{{ch:math-notation}} showed the softmax without justifying it. The reason to
exponentiate rather than, say, divide by the sum of raw scores is now available:

- Scores may be negative; probabilities may not. $\exp$ maps $\R \to (0,\infty)$,
  guaranteeing positivity.
- Dividing raw scores by their sum fails whenever the sum is zero or negative,
  and produces negative "probabilities" for negative scores.
- Exponentiating makes the result depend only on score *differences*, since
  $e^{z_i - c}/\sum_j e^{z_j - c}$ is independent of $c$. Adding a constant to
  every score leaves the distribution unchanged — a desirable invariance, and
  precisely what {{eq:logsumexp}} exploits.

## 7. Implementation

```python {tier=A name=functions-and-logs}
"""Exponentials, logarithms, the logistic curve, and the log-sum-exp trick.

Every identity stated in the chapter is checked numerically here rather than
taken on trust.
"""
import numpy as np

# --- eq. 2.4 / 2.6: exponent and logarithm laws -----------------------------
x, y, n = 7.0, 3.0, 4
assert np.isclose(np.exp(x) * np.exp(y), np.exp(x + y))
assert np.isclose(np.log(x * y), np.log(x) + np.log(y))
assert np.isclose(np.log(x / y), np.log(x) - np.log(y))
assert np.isclose(np.log(x ** n), n * np.log(x))
print("exponent and logarithm laws verified")

# --- logarithms compress scale ----------------------------------------------
values = np.array([1e-9, 1e-3, 1.0, 1e3, 1e9])
print("\nvalues :", values)
print("log10  :", np.log10(values), " <- 18 orders of magnitude become 18 units")


# --- the logistic function and its derivative -------------------------------
def sigmoid(x):
    """Numerically stable logistic function.

    The naive form 1/(1 + exp(-x)) overflows for large negative x, because
    exp(-x) becomes inf. For x < 0 the algebraically identical form
    exp(x)/(1 + exp(x)) keeps every exponent negative and cannot overflow.
    """
    out = np.empty_like(x, dtype=float)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def sigmoid_derivative(x):
    """eq. 2.16 — expressible in terms of the output alone."""
    s = sigmoid(x)
    return s * (1.0 - s)


xs = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
print(f"\n{'x':>6} {'sigma(x)':>14} {'sigma-prime(x)':>16}")
for xi, s, d in zip(xs, sigmoid(xs), sigmoid_derivative(xs)):
    print(f"{xi:>6.1f} {s:>14.10f} {d:>16.3e}")

# eq. 2.14: the symmetry identity, and the peak derivative of 1/4
assert np.allclose(sigmoid(-xs), 1.0 - sigmoid(xs))
assert np.isclose(sigmoid_derivative(np.array([0.0]))[0], 0.25)

# The naive form really does overflow where the stable one does not.
with np.errstate(over="ignore"):
    naive = 1.0 / (1.0 + np.exp(-np.array([-800.0])))
print(f"\nnaive sigmoid(-800) = {naive[0]}  (overflow in exp(800))")
print(f"stable sigmoid(-800) = {sigmoid(np.array([-800.0]))[0]}")

# --- vanishing gradients are just multiplication -----------------------------
print("\nGradient reaching layer 1 through a stack of sigmoids, best case:")
for depth in (1, 5, 10, 20):
    print(f"  depth {depth:>2}: {0.25 ** depth:.3e}")


# --- eq. 2.18: the log-sum-exp trick ----------------------------------------
def logsumexp(z):
    c = np.max(z)
    return c + np.log(np.sum(np.exp(z - c)))


z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive_lse = np.log(np.sum(np.exp(z)))
print(f"\nnaive  log-sum-exp: {naive_lse}")
print(f"stable log-sum-exp: {logsumexp(z):.4f}")
assert np.isclose(logsumexp(z), 1002.4076, atol=1e-4)

# It agrees with the naive form wherever the naive form works at all.
small = np.array([1.0, 2.0, 3.0])
assert np.isclose(logsumexp(small), np.log(np.sum(np.exp(small))))
print("stable and naive agree on inputs the naive form can handle")

# --- eq. 2.12: monotonicity preserves the argmax ----------------------------
rng = np.random.default_rng(0)
likelihoods = rng.random(8) + 0.01
assert likelihoods.argmax() == np.log(likelihoods).argmax()
print("\nargmax of a likelihood == argmax of its log (eq. 2.13)")
print(f"  max likelihood {likelihoods.max():.4f} at index {likelihoods.argmax()}")
print(f"  max log-lik   {np.log(likelihoods).max():.4f} at index "
      f"{np.log(likelihoods).argmax()}  <- same index, different value")

# --- why products of probabilities need logs --------------------------------
probs = rng.uniform(0.3, 0.9, size=2000)
print(f"\nproduct of 2000 probabilities : {np.prod(probs)}  <- underflowed to 0")
print(f"sum of their logs             : {np.sum(np.log(probs)):.2f}  <- fine")
```

## 8. Practical Example

Binary classification is where all three functions of this chapter meet.

A model produces a single real-valued score $z$ for an input — unbounded, of
either sign. Three steps convert that into a trained classifier:

**Score to probability.** Apply the logistic function:
$\hat{p} = \sigma(z) \in (0,1)$. The score is now interpretable as the model's
probability that the label is 1.

**Probability to loss.** Use binary cross-entropy, for a true label
$y \in \{0, 1\}$:

$$
\ell(\hat{p}, y) = -\big[y\log\hat{p} + (1-y)\log(1-\hat{p})\big]
$$ (eq:bce)

The indicator-like structure means only one term survives per example: when
$y = 1$ the loss is $-\log\hat{p}$, and when $y = 0$ it is $-\log(1-\hat{p})$.
In both cases it is the negative log of the probability assigned to what
actually happened — exactly {{eq:nll}} from {{ch:math-notation}}, specialised to
two classes.

**Loss to gradient.** Combining {{eq:bce}} with {{eq:sigmoid-derivative}}
produces a result of unusual simplicity, derived in {{ch:ml-logistic}}:

$$
\frac{\partial \ell}{\partial z} = \hat{p} - y
$$ (eq:bce-gradient)

The gradient with respect to the score is just the prediction error. The
$\sigma(1-\sigma)$ factor from the sigmoid derivative cancels exactly against a
matching factor from the logarithm in the loss.

> IMPORTANT: That cancellation is why the pairing of the logistic function with
> cross-entropy loss is not arbitrary. Pair a sigmoid with squared error instead
> and the $\sigma'$ factor survives, so a confidently wrong prediction — deep in
> saturation, where $\sigma' \approx 0$ — produces almost no gradient and the
> model cannot correct itself. This is why classification uses cross-entropy and
> not mean squared error, and it is a decision made for us by
> {{eq:sigmoid-derivative}}.

> PRODUCTION TIP: In practice, never compute $\sigma(z)$ and then take its
> logarithm. Libraries provide a fused operation —
> `torch.nn.BCEWithLogitsLoss`, or `scipy.special.log_expit` — which applies the
> log-sum-exp trick internally and stays stable for scores of any magnitude.
> Computing the two steps separately is a common and hard-to-diagnose source of
> `nan` in training logs.

## 9. Common Mistakes

**Reading $f \circ g$ left to right.** It applies $g$ first. Almost everyone
gets this wrong at least once.

**Taking the log of zero.** $\log(0) = -\infty$, and it propagates. Any
probability entering a logarithm should be clipped away from zero, or produced
by a fused log-probability operation that never materialises the probability.

**Computing `exp` before `log`.** The log-sum-exp trick exists precisely to
avoid this. Reach for the fused library function.

**Assuming $\log$ means base 10.** In this book, and in most machine learning
literature, it is base $e$. In some engineering fields it is base 10, and in
information theory it is often base 2. Check.

**Believing $\log(x + y) = \log x + \log y$.** It does not. The identity is for
*products*, not sums. This is the single most common algebra error in the
subject, and there is no simple expansion for the log of a sum — which is, in a
sense, exactly why log-sum-exp needs a trick at all.

**Pairing a sigmoid with squared error.** See {{sec:8-practical-example}}. It
trains, badly, for reasons that are invisible unless you have looked at the
gradient.

**Forgetting that the range is open.** $\sigma(x)$ never equals 0 or 1. Code
that tests `if p == 1.0` will not fire, and code that assumes a probability can
reach its bounds will be surprised by the floating-point value that rounds there.

## 10. Connection to Previous Chapters

{{ch:math-notation}} supplied the function signatures and summation notation
this chapter builds on, and showed {{eq:nll}} without justifying its logarithm.
{{sec:4-intuitive-explanation}} of this chapter supplies the justification.

Forward: {{ch:math-derivatives}} derives the chain rule that
{{sec:6-mathematical-foundation}} used on faith. {{ch:math-probability}} uses
$\log$ to make products of probabilities tractable, and
{{ch:math-random-vars}} uses $\exp$ in the Gaussian density.
{{ch:math-optimization}} builds cross-entropy from {{eq:bce}}.
{{ch:dl-activations}} takes up saturation and the alternatives to the sigmoid,
and {{ch:ml-logistic}} derives {{eq:bce-gradient}} in full. The
{{term:logistic-function}}'s inverse gives the name to the
{{term:logit}}s that Parts VII onward are full of.

The composition of {{eq:network-composition}} is the structural idea that makes
{{part:6}} possible at all.

## 11. Exercises

**Beginner**

1. For $f(x) = 2x + 1$ and $g(x) = x^2$, compute $(f \circ g)(3)$ and
   $(g \circ f)(3)$. Confirm they differ.
2. Simplify $\log(8) - \log(2)$ and $\log(2^{10})$ without a calculator.
3. Compute $\sigma(0)$, $\sigma(2)$ and $\sigma(-2)$ to four decimal places, and
   verify the symmetry identity of {{eq:sigmoid-identities}}.
4. Convert the probability $0.8$ to a logit, then convert it back.
5. State the domain and range of $\exp$, $\log$ and $\sigma$.

**Intermediate**

6. Show that $b^{x} = e^{x\ln b}$, and use it to express $2^{x}$ in terms of
   $\exp$.
7. Explain why $\log$ has domain $(0, \infty)$ rather than $[0, \infty)$, and
   what practical consequence this has for a model that assigns probability zero
   to an observed event.
8. Verify {{eq:logsumexp}} algebraically, then evaluate
   $\log(e^{500} + e^{501})$ by hand using it.
9. A model outputs the score $z = -6$. What probability does it imply, and what
   is $\sigma'(-6)$? What does the second number tell you about how quickly this
   prediction can be corrected?
10. Prove {{eq:argmax-invariance}} for a strictly increasing $g$. Where exactly
    does the proof use strictness rather than mere monotonicity?

**Advanced**

11. Derive the logit function $\sigma^{-1}(p) = \log(p/(1-p))$ from
    {{eq:sigmoid}} by solving for $x$.
12. Prove that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ by a different route than
    {{sec:6-mathematical-foundation}}: write $\sigma(x) = \frac{1}{2}(1 +
    \tanh(x/2))$ and differentiate.
13. The softmax with two classes reduces to a sigmoid. Show this explicitly,
    and identify what plays the role of $z$.
14. Show that $\log$ is the *only* continuous function (up to a constant factor)
    satisfying $f(xy) = f(x) + f(y)$ for all positive $x, y$. What does this say
    about why logarithms appear wherever independence does?

**Implementation**

15. Implement `logsumexp` yourself and test it against `scipy.special.logsumexp`
    on inputs spanning $[-1000, 1000]$. Then find an input where a naive
    implementation returns `nan` and yours does not.
16. Implement binary cross-entropy two ways — from probabilities, and fused from
    logits — and find a score magnitude at which the first produces `inf` and
    the second does not.
17. Plot $\sigma$ and $\sigma'$ on $[-10, 10]$. Mark the region where
    $\sigma' < 0.01$ and state what fraction of the plotted range it occupies.

**Reasoning**

18. Learning rates are conventionally searched over a logarithmic grid
    ($10^{-2}, 10^{-3}, 10^{-4}$) rather than a linear one. Justify this using
    the scale-compression property.
19. Cross-entropy penalises a confident error far more than an uncertain one.
    Is that desirable? Construct a scenario where it is, and one where it makes
    a model fragile.

## 12. Chapter Summary

A function assigns exactly one output to each input; that uniqueness is the
whole definition, and it is what makes a model debuggable. Composition chains
functions and reads right to left, and a deep network is nothing but a long
composition — which is why the chain rule matters so much later.

The exponential function is strictly positive with range $(0,\infty)$, is its own
derivative, and grows fast enough to overflow floating point at inputs above
about 710. The logarithm inverts it, and earns its ubiquity through three
separate properties: it turns products into sums, which makes likelihoods
computable; it compresses orders of magnitude, which is why hyperparameters are
searched on log scales; and it preserves ordering, which licenses maximising a
log-likelihood in place of a likelihood.

The logistic function maps $\R$ into the open interval $(0,1)$, so an unbounded
score becomes a probability. Its derivative $\sigma(1-\sigma)$ peaks at $0.25$
and collapses toward zero away from the origin. Because a deep network's
gradient is a product of such factors, that collapse compounds — which is the
vanishing-gradient problem, and it is arithmetic rather than mystery.

The log-sum-exp trick — subtract the maximum before exponentiating — makes
softmax and cross-entropy computable at any score magnitude. It is an exact
identity, not an approximation, and forgetting it is the most common cause of
`nan` in hand-written training code.

Pairing the logistic function with cross-entropy makes the gradient with respect
to the score exactly $\hat{p} - y$, because the sigmoid's derivative cancels
against the logarithm's. That cancellation is why classification uses
cross-entropy rather than squared error.

{{cite:deisenroth2020}} covers this material more briefly; readers wanting more
practice with the algebra will find it there.
