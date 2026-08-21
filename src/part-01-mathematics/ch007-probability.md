---
id: math-probability
number: 7
part: I
tier: focused
status: reviewed
requires: [math-notation, math-functions]
provides: [probability-term, sample-space, conditional-probability,
           statistical-independence, bayes-theorem, prior-term, posterior,
           likelihood-term]
citations: [deisenroth2020]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the three axioms of probability and derive basic consequences from
   them.
2. Identify the sample space of a problem, and explain why getting it wrong is
   the most common source of probability errors.
3. Compute conditional probabilities and apply the chain rule of probability.
4. Test whether two events are independent, and distinguish independence from
   mutual exclusivity.
5. State and apply Bayes' theorem, and correctly identify prior, likelihood,
   evidence and posterior.
6. Explain the base-rate fallacy and work a medical-testing example correctly.
7. Explain why a likelihood is not a probability distribution over parameters.
8. Recognise where each of these appears in machine learning: classifiers,
   language models, and evaluation metrics.

## 2. Why This Matters

Machine learning models do not output answers. They output probability
distributions, and the answer is whatever you decide to do with the
distribution.

A classifier does not say "cat"; it assigns 0.87 to cat and 0.13 to dog, and
something downstream picks the argmax. A language model does not produce a
sentence; it produces a distribution over the next token, repeatedly
({{ch:llm-decoding}}). A retrieval system does not find the right document; it
ranks by estimated relevance. Uncertainty is not an unfortunate residue in these
systems — it is the output format.

Conditional probability in particular is the shape of nearly everything here.
Every supervised model estimates $p(y \given \vec{x})$: the distribution over
outputs given an input. A language model estimates
$p(\text{next token} \given \text{context})$. Once you see that, a great deal of
apparently disparate machinery turns out to be the same machinery.

And Bayes' theorem is where most people's intuition fails hardest. The
base-rate fallacy in {{sec:6-mathematical-foundation}} is not a curiosity for
statistics classes; it is the reason a model with 99% accuracy on a rare
condition can be nearly useless in deployment, and the reason precision and
recall exist as separate metrics ({{ch:ml-metrics}}).

## 3. Prerequisites

{{ch:math-notation}} for set notation, summation, and indicator functions.
{{ch:math-functions}} for logarithms — needed because probabilities are almost
always handled in log space.

## 4. Intuitive Explanation

### 4.1 What a probability is

A {{term:probability-term}} is a number between 0 and 1 measuring how strongly
an outcome is expected. That much is uncontroversial. What it *means* is not,
and there are two long-standing interpretations:

**Frequentist.** A probability is the long-run frequency of an outcome over
repeated trials. "This coin lands heads with probability 0.5" means that over
many flips, about half come up heads.

**Bayesian.** A probability is a degree of belief. "There is a 30% chance it
rains tomorrow" cannot be a long-run frequency — tomorrow happens once — but it
is a perfectly meaningful statement about confidence given the evidence.

This book uses both, because machine learning uses both, and it says which is
meant wherever the distinction affects the argument. Cross-validation estimates
are frequentist; a language model's token distribution is closer to a Bayesian
degree of belief; and the argument in {{ch:math-inference}} about what a
confidence interval means turns entirely on the difference.

The arithmetic, importantly, is identical under both readings. The axioms in
{{sec:5-formal-explanation}} do not care which philosophy you hold.

### 4.2 The sample space is where errors begin

Before computing any probability, you must be clear about the set of possible
outcomes — the {{term:sample-space}}. This sounds like a formality. It is where
most mistakes actually happen.

The classic illustration: a family has two children, and you learn that at least
one is a girl. What is the probability both are girls?

The intuitive answer is 1/2 — the other child is a girl or a boy. It is wrong.
The sample space for two children, in birth order, is
$\{GG, GB, BG, BB\}$, each equally likely. Learning "at least one is a girl"
eliminates only $BB$, leaving three equally likely outcomes, of which one is
$GG$. The answer is **1/3**.

The intuitive answer implicitly used the sample space $\{GG, GB\}$ — treating
"the other child" as a well-defined single child, when the information given
does not pick one out. The arithmetic was never the problem; the enumeration
was.

> IMPORTANT: When a probability answer feels wrong, write out the sample space
> explicitly before doubting the arithmetic. In machine learning this discipline
> shows up as being precise about *what is being conditioned on*: the
> probability of a token given the preceding context is not the same as the
> probability of that token overall, and confusing the two is the same class of
> error.

### 4.3 Conditioning is restricting and renormalising

{{term:conditional-probability}} — the probability of $A$ given that $B$ has
happened — is not a new kind of quantity. It is ordinary probability computed in
a shrunken world.

$$
\Prob(A \given B) = \frac{\Prob(A \cap B)}{\Prob(B)}
$$ (eq:conditional)

Discard every outcome inconsistent with $B$, then renormalise over what is left
so the probabilities sum to 1 again. The denominator is doing exactly the
renormalisation.

This picture makes the two-children puzzle obvious. Conditioning on "at least
one girl" discards $BB$, leaving three outcomes; renormalising gives each
probability 1/3; and $GG$ is one of them.

It is also worth noticing that {{eq:conditional}} is precisely the structure of
a softmax ({{ch:math-functions}}): restrict to a set, divide by the total over
that set. A language model's next-token distribution is a conditional
probability computed by exactly this pattern.

### 4.4 Bayes' theorem reverses the question

You usually know one conditional and want the other.

A test for a disease is characterised by $\Prob(\text{positive} \given
\text{disease})$ — how often it fires when the disease is present. That is what
the manufacturer measures. But a patient who tests positive wants
$\Prob(\text{disease} \given \text{positive})$, which is a different number, and
sometimes a *wildly* different number.

{{term:bayes-theorem}} converts one into the other:

$$
\Prob(H \given E) = \frac{\Prob(E \given H)\,\Prob(H)}{\Prob(E)}
$$ (eq:bayes)

The four pieces have names worth learning, because they recur throughout
machine learning:

- $\Prob(H)$ — the **prior**: belief before seeing the evidence.
- $\Prob(E \given H)$ — the **likelihood**: how well the hypothesis explains the
  evidence.
- $\Prob(E)$ — the **evidence** or marginal likelihood: how likely the evidence
  is overall.
- $\Prob(H \given E)$ — the **posterior**: belief after updating.

The theorem's content, in one sentence: *the posterior is proportional to the
likelihood times the prior.* Evidence does not replace your prior belief; it
reweights it. When the prior is extreme — a disease affecting one person in ten
thousand — even strong evidence may leave the posterior small, and that is the
base-rate fallacy in a sentence.

## 5. Formal Explanation

### 5.1 The axioms

Let $\Omega$ be the sample space and let events be subsets of $\Omega$. A
probability measure $\Prob$ satisfies:

$$
\Prob(A) \ge 0 \quad\text{for every event } A
$$ (eq:axiom-nonneg)

$$
\Prob(\Omega) = 1
$$ (eq:axiom-total)

$$
\Prob\!\left(\bigcup_i A_i\right) = \sum_i \Prob(A_i)
\quad\text{for pairwise disjoint } A_i
$$ (eq:axiom-additivity)

Everything else follows. Some immediate consequences:

$$
\Prob(A^{c}) = 1 - \Prob(A), \qquad
\Prob(\varnothing) = 0, \qquad
A \subseteq B \implies \Prob(A) \le \Prob(B)
$$ (eq:axiom-consequences)

$$
\Prob(A \cup B) = \Prob(A) + \Prob(B) - \Prob(A \cap B)
$$ (eq:inclusion-exclusion)

{{eq:inclusion-exclusion}} is inclusion-exclusion: the intersection is
subtracted because adding both probabilities counts it twice. It reduces to
{{eq:axiom-additivity}} when the events are disjoint.

### 5.2 Conditional probability and the chain rule

{{eq:conditional}} defines conditioning, for $\Prob(B) > 0$. Rearranging gives
the **product rule**:

$$
\Prob(A \cap B) = \Prob(A \given B)\,\Prob(B) = \Prob(B \given A)\,\Prob(A)
$$ (eq:product-rule)

Extending to many events gives the **chain rule of probability**:

$$
\Prob(A_1 \cap \cdots \cap A_n)
  = \Prob(A_1)\prod_{i=2}^{n}\Prob\!\left(A_i \given A_1 \cap \cdots \cap A_{i-1}\right)
$$ (eq:chain-rule-prob)

{{eq:chain-rule-prob}} is not a technicality — it is the entire mathematical
basis of autoregressive language modelling. The probability of a sequence of
tokens factorises as the product of each token's probability given all previous
ones:

$$
p(t_1, \ldots, t_n) = \prod_{i=1}^{n} p(t_i \given t_1, \ldots, t_{i-1})
$$ (eq:autoregressive)

A language model is a machine that estimates one factor of {{eq:autoregressive}}
({{ch:llm-next-token}}). Everything about causal masking
({{ch:tf-masking-kv}}) exists to enforce that each factor conditions only on
what precedes it.

> MATH NOTE: Taking logs of {{eq:autoregressive}} turns the product into a sum,
> $\log p(t_{1:n}) = \sum_i \log p(t_i \given t_{<i})$, which is both
> numerically necessary — a product of thousands of probabilities underflows —
> and the reason model quality is reported as *average* log-probability per
> token. Perplexity is the exponential of the negative of that average.

### 5.3 Independence

Events $A$ and $B$ are {{term:statistical-independence}} when

$$
\Prob(A \cap B) = \Prob(A)\,\Prob(B)
$$ (eq:independence)

equivalently $\Prob(A \given B) = \Prob(A)$: knowing $B$ tells you nothing about
$A$.

> WARNING: Independence and mutual exclusivity are opposites, not synonyms, and
> conflating them is extremely common. Mutually exclusive events *cannot* both
> occur, so learning one happened tells you the other did not — which is
> maximal dependence. Formally, if $A$ and $B$ are disjoint with positive
> probability, then $\Prob(A \cap B) = 0 \neq \Prob(A)\Prob(B)$, so they are
> necessarily *dependent*.

**Conditional independence** — $A$ and $B$ independent given $C$ — is written

$$
\Prob(A \cap B \given C) = \Prob(A \given C)\,\Prob(B \given C)
$$ (eq:conditional-independence)

This is the assumption behind naive Bayes ({{ch:ml-knn-nb}}): features are
assumed conditionally independent given the class. The assumption is nearly
always false, and the classifier often works anyway — a discrepancy that chapter
takes up.

### 5.4 Marginalisation and total probability

To eliminate a variable you do not care about, sum over all its values. For a
partition $\{B_1, \ldots, B_k\}$ of the sample space:

$$
\Prob(A) = \sum_{i=1}^{k}\Prob(A \cap B_i) = \sum_{i=1}^{k}\Prob(A \given B_i)\,\Prob(B_i)
$$ (eq:total-probability)

This is the **law of total probability**, and it is how the denominator of
{{eq:bayes}} is usually computed: the evidence $\Prob(E)$ is rarely known
directly but can always be assembled from the likelihoods and priors.

### 5.5 Bayes' theorem

Combining {{eq:conditional}} with {{eq:product-rule}}:

$$
\Prob(H \given E) = \frac{\Prob(E \given H)\,\Prob(H)}{\Prob(E)}
$$

and expanding the denominator with {{eq:total-probability}} over hypotheses
$H_1, \ldots, H_k$:

$$
\Prob(H_j \given E) = \frac{\Prob(E \given H_j)\,\Prob(H_j)}
                            {\sum_{i=1}^{k}\Prob(E \given H_i)\,\Prob(H_i)}
$$ (eq:bayes-expanded)

Since the denominator does not depend on which hypothesis you are evaluating, it
is often dropped:

$$
\Prob(H \given E) \propto \Prob(E \given H)\,\Prob(H)
$$ (eq:bayes-proportional)

*Posterior is proportional to likelihood times prior.* When you only need to
compare hypotheses — as when taking an argmax — the normalising constant is
irrelevant, which saves computing it.

### 5.6 The likelihood is not a distribution over hypotheses

$\Prob(E \given H)$ read as a function of $H$, with $E$ fixed at what was
actually observed, is called the {{term:likelihood-term}}, written $L(H)$.

It is not a probability distribution over $H$. It does not sum or integrate to 1
over hypotheses, and there is no reason it should — it is a slice through a
function in the wrong direction.

This matters because the two most common estimation strategies differ precisely
in whether they respect it. **Maximum likelihood** picks
$\argmax_{H} L(H)$, ignoring the prior. **Maximum a posteriori** picks
$\argmax_{H} L(H)\Prob(H)$, including it. Regularisation, as
{{ch:math-optimization}} shows, is exactly a prior in disguise: an $L_2$ penalty
is a Gaussian prior on the weights, and an $L_1$ penalty is a Laplace prior.

## 6. Mathematical Foundation

### 6.1 The base-rate fallacy, worked properly

This is the most important calculation in the chapter.

A disease affects 1 in 1,000 people. A test has 99% sensitivity — it correctly
identifies 99% of people who have the disease — and 95% specificity, correctly
clearing 95% of people who do not. You test positive. What is the probability
you have the disease?

Most people, including most doctors when surveyed, answer around 95% or 99%. The
correct answer is about **1.9%**.

Set it up. Let $D$ be having the disease and $+$ a positive test.

$$
\Prob(D) = 0.001, \qquad
\Prob(+ \given D) = 0.99, \qquad
\Prob(+ \given D^{c}) = 0.05
$$

The false-positive rate is $1 - 0.95 = 0.05$. Compute the evidence with
{{eq:total-probability}}:

$$
\Prob(+) = (0.99)(0.001) + (0.05)(0.999) = 0.00099 + 0.04995 = 0.05094
$$ (eq:evidence-computed)

Then Bayes:

$$
\Prob(D \given +) = \frac{(0.99)(0.001)}{0.05094} = \frac{0.00099}{0.05094} \approx 0.0194
$$ (eq:posterior-computed)

About 1.9%.

The reason is visible directly in {{eq:evidence-computed}}. Out of 100,000
people, 100 have the disease and 99 of them test positive. But 99,900 do not
have it, and 5% of those — 4,995 people — also test positive. The false
positives outnumber the true positives fifty to one, purely because the healthy
group is a thousand times larger. The test is *good*; the base rate is what
dominates.

{#tbl:base-rate caption="The base-rate calculation as a contingency table over 100,000 people. The bottom-left cell is what intuition ignores."}

| | Test positive | Test negative | Total |
|---|---|---|---|
| **Has disease** | 99 | 1 | 100 |
| **No disease** | 4,995 | 94,905 | 99,900 |
| **Total** | 5,094 | 94,906 | 100,000 |

$\Prob(D \given +) = 99 / 5{,}094 \approx 1.94\%$.

> IMPORTANT: This is not a puzzle about medicine. It is why a fraud detector
> with 99% accuracy on a problem where 0.1% of transactions are fraudulent will
> drown its operators in false positives, and it is why accuracy is a nearly
> useless metric for imbalanced problems. Precision — which is exactly
> $\Prob(D \given +)$ — is the quantity that matters, and
> {{ch:ml-metrics}} and {{ch:ds-leakage}} return to this at length.

### 6.2 Updating twice

Bayesian updating composes. Suppose you take a second, independent test and it
is also positive. The posterior from the first test becomes the prior for the
second:

$$
\Prob(D) = 0.0194 \;\longrightarrow\;
\Prob(D \given ++) = \frac{(0.99)(0.0194)}{(0.99)(0.0194) + (0.05)(0.9806)} \approx 0.281
$$ (eq:second-update)

Two positives: 28%. A third would give about 89%. Evidence accumulates
multiplicatively, and this sequential structure — posterior becomes prior — is
the entire mechanic of Bayesian inference.

It also explains a practical rule: repeat testing is far more informative than
improving a single test, when the base rate is low.

### 6.3 Odds make Bayes easy

Bayes' theorem is more tractable in odds form. Define the odds of a hypothesis
as $\text{odds}(H) = \Prob(H)/\Prob(H^{c})$. Then

$$
\underbrace{\frac{\Prob(H \given E)}{\Prob(H^{c} \given E)}}_{\text{posterior odds}}
= \underbrace{\frac{\Prob(E \given H)}{\Prob(E \given H^{c})}}_{\text{likelihood ratio}}
\times
\underbrace{\frac{\Prob(H)}{\Prob(H^{c})}}_{\text{prior odds}}
$$ (eq:bayes-odds)

The evidence term cancels entirely, because it appears in both numerator and
denominator.

For the disease example: prior odds $= 0.001/0.999 \approx 1/999$. Likelihood
ratio $= 0.99/0.05 = 19.8$. Posterior odds $= 19.8/999 \approx 0.0198$, which
converts back to a probability of $0.0198/(1 + 0.0198) \approx 0.0194$ ✓.

The likelihood ratio of 19.8 is the honest measure of how much the test tells
you: it multiplies your odds by about twenty. Starting from odds of 1 in 999,
twenty-fold is not enough.

> MATH NOTE: Taking logs of {{eq:bayes-odds}} turns it into addition:
> log-posterior-odds = log-likelihood-ratio + log-prior-odds. Evidence adds up
> on the log-odds scale. This is precisely the scale a logistic regression works
> on — its output before the sigmoid is a log-odds, which is why the sigmoid's
> inverse is called the logit ({{ch:math-functions}}) — and it is why stacking
> independent evidence in a linear model is a defensible thing to do
> ({{ch:ml-logistic}}).

## 7. Implementation

```python {tier=A name=probability-and-bayes}
"""Probability axioms, conditioning, independence, and Bayes — all checked
numerically against simulation rather than asserted.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- the two-children puzzle, by enumeration and by simulation --------------
space = ["GG", "GB", "BG", "BB"]
at_least_one_girl = [s for s in space if "G" in s]
both_girls = [s for s in at_least_one_girl if s == "GG"]
print(f"enumeration : P(both girls | at least one girl) = "
      f"{len(both_girls)}/{len(at_least_one_girl)} = "
      f"{len(both_girls)/len(at_least_one_girl):.4f}")

trials = rng.integers(0, 2, size=(400_000, 2))       # 0 = boy, 1 = girl
has_girl = trials.sum(axis=1) >= 1
both = trials.sum(axis=1) == 2
print(f"simulation  : {both.sum() / has_girl.sum():.4f}   <- 1/3, not 1/2")
assert abs(both.sum() / has_girl.sum() - 1/3) < 0.01

# --- independence vs mutual exclusivity -------------------------------------
# Two fair dice. A = "first is 6", B = "second is 6" -> independent.
# C = "sum is 2",  D = "sum is 12"  -> mutually exclusive, hence DEPENDENT.
d1 = rng.integers(1, 7, size=300_000)
d2 = rng.integers(1, 7, size=300_000)
A, B = d1 == 6, d2 == 6
C, D = (d1 + d2) == 2, (d1 + d2) == 12

print(f"\nP(A)P(B) = {A.mean() * B.mean():.5f}  vs  P(A and B) = "
      f"{(A & B).mean():.5f}   -> independent")
print(f"P(C)P(D) = {C.mean() * D.mean():.7f}  vs  P(C and D) = "
      f"{(C & D).mean():.7f}   -> mutually exclusive means DEPENDENT")

# --- eq. 7.13: Bayes, and the base-rate fallacy -----------------------------
def bayes(prior, sensitivity, false_positive_rate):
    """P(D | +) from the prior, P(+|D), and P(+|not D)."""
    evidence = sensitivity * prior + false_positive_rate * (1 - prior)
    return sensitivity * prior / evidence


prior, sens, fpr = 0.001, 0.99, 0.05
post = bayes(prior, sens, fpr)
print(f"\nP(disease) = {prior}, sensitivity = {sens}, "
      f"false-positive rate = {fpr}")
print(f"P(disease | positive) = {post:.4f}  ({post:.2%})")
assert abs(post - 0.0194) < 0.0005

# The contingency table of table 7.1, from a simulated population.
N = 1_000_000
sick = rng.random(N) < prior
positive = np.where(sick, rng.random(N) < sens, rng.random(N) < fpr)
print(f"\nsimulated population of {N:,}:")
print(f"  sick and positive     : {int((sick & positive).sum()):>7,}")
print(f"  healthy and positive  : {int((~sick & positive).sum()):>7,}")
print(f"  P(sick | positive)    : {(sick & positive).sum()/positive.sum():.4f}")

# --- eq. 7.15: sequential updating ------------------------------------------
print("\nrepeated independent positive tests:")
p = prior
for k in range(1, 6):
    p = bayes(p, sens, fpr)
    print(f"  after {k} positive test(s): {p:.4f}  ({p:.1%})")

# --- eq. 7.16: the odds form -------------------------------------------------
prior_odds = prior / (1 - prior)
lr = sens / fpr
posterior_odds = lr * prior_odds
print(f"\nprior odds      : {prior_odds:.6f}  (about 1 in {1/prior_odds:.0f})")
print(f"likelihood ratio: {lr:.2f}   <- how much one test multiplies your odds")
print(f"posterior odds  : {posterior_odds:.6f}")
print(f"back to probability: {posterior_odds/(1+posterior_odds):.4f}  <- matches")
assert np.isclose(posterior_odds / (1 + posterior_odds), post)

# --- how the answer depends on the base rate --------------------------------
print(f"\n{'base rate':>12} {'P(disease | positive)':>22}")
for br in (0.0001, 0.001, 0.01, 0.1, 0.5):
    print(f"{br:>12.4f} {bayes(br, sens, fpr):>21.1%}")
print("The test never changed. Only the prior did.")

# --- eq. 7.9: the chain rule, and why language models use logs --------------
seq_probs = rng.uniform(0.05, 0.6, size=1500)       # per-token probabilities
print(f"\nproduct of 1500 token probabilities: {np.prod(seq_probs)}  <- underflow")
total_logp = np.sum(np.log(seq_probs))
print(f"sum of their logs                  : {total_logp:.2f}")
print(f"average log-prob per token         : {total_logp/len(seq_probs):.4f}")
print(f"perplexity = exp(-avg log-prob)    : "
      f"{np.exp(-total_logp/len(seq_probs)):.2f}")
```

## 8. Practical Example

Spam filtering is the canonical application of Bayes' theorem, and building a
small one makes the abstract pieces concrete.

```python {tier=A name=naive-bayes-spam}
"""A naive Bayes spam filter, from Bayes' theorem and nothing else.

'Naive' names the conditional-independence assumption of eq. 7.11: words are
treated as independent given the class. That is plainly false — 'free' and
'money' co-occur — and the classifier works well anyway. Chapter 35 examines why.
"""
import numpy as np
from collections import Counter

train = [
    ("win free money now claim prize", "spam"),
    ("free money click here now", "spam"),
    ("claim your free prize today", "spam"),
    ("urgent win cash prize claim", "spam"),
    ("meeting moved to tuesday morning", "ham"),
    ("please review the attached report", "ham"),
    ("lunch tomorrow at the usual place", "ham"),
    ("the report is attached for review", "ham"),
    ("can we move the meeting to friday", "ham"),
]

classes = ["spam", "ham"]
docs = {c: [t for t, lab in train if lab == c] for c in classes}
vocab = sorted({w for t, _ in train for w in t.split()})

# Priors: P(class), estimated as the class frequency.
priors = {c: len(docs[c]) / len(train) for c in classes}

# Likelihoods: P(word | class), with add-one (Laplace) smoothing so that a word
# unseen in a class gets a small probability rather than zero. Without it, one
# unseen word would zero the entire product — the classic naive Bayes failure.
counts = {c: Counter(w for t in docs[c] for w in t.split()) for c in classes}
totals = {c: sum(counts[c].values()) for c in classes}


def log_likelihood(word, c):
    return np.log((counts[c][word] + 1) / (totals[c] + len(vocab)))


def classify(text):
    """Return log-posteriors, using eq. 7.14 in log space."""
    words = [w for w in text.split() if w in vocab]
    scores = {}
    for c in classes:
        scores[c] = np.log(priors[c]) + sum(log_likelihood(w, c) for w in words)
    # Normalise the log-scores into probabilities (the log-sum-exp of Ch. 2).
    mx = max(scores.values())
    exp = {c: np.exp(scores[c] - mx) for c in classes}
    z = sum(exp.values())
    return {c: exp[c] / z for c in classes}


print(f"priors: {priors}\n")
for msg in ["free money claim now",
            "the meeting is moved to tuesday",
            "please review the free report",
            "win prize"]:
    p = classify(msg)
    verdict = max(p, key=p.get)
    print(f"{msg:<38} -> {verdict:<5} (spam {p['spam']:.3f})")

# The most discriminative words are those with the largest likelihood ratio —
# exactly the quantity in the odds form of Bayes (eq. 7.16).
print("\nmost spam-indicative words by likelihood ratio (eq. 7.16):")
ratios = {w: log_likelihood(w, "spam") - log_likelihood(w, "ham") for w in vocab}
for w, r in sorted(ratios.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {w:<10} log-likelihood-ratio {r:+.3f}  (x{np.exp(r):.1f} odds)")
```

Three things in that code are worth naming, because they generalise well beyond
spam.

**Smoothing is not optional.** A word never seen in a class would give
probability zero, and one zero factor annihilates the entire product. Add-one
smoothing is the crudest fix; the general principle — never assign exactly zero
probability to something merely because you have not observed it — recurs
throughout machine learning.

**Everything happens in log space.** Multiplying dozens of small probabilities
underflows. This is {{ch:math-functions}}'s argument, applied.

**The likelihood ratio identifies the informative features.** The odds form of
Bayes' theorem tells you not just how to classify but *which evidence carried the
decision*, which is the beginning of interpretability.

## 9. Common Mistakes

**Confusing $\Prob(A \given B)$ with $\Prob(B \given A)$.** The prosecutor's
fallacy, the base-rate fallacy, and most misreadings of medical test results are
all this one error.

**Ignoring the base rate.** A test's accuracy tells you almost nothing about the
probability of the condition given a positive result. You need the prior.

**Treating mutually exclusive events as independent.** They are maximally
dependent. See the warning in {{sec:5-formal-explanation}}.

**Getting the sample space wrong.** Enumerate it explicitly whenever an answer
feels surprising.

**Interpreting a p-value as $\Prob(H_0 \given \text{data})$.** It is
$\Prob(\text{data or more extreme} \given H_0)$ — the conditional runs the other
way. {{ch:math-inference}} treats this properly.

**Assigning zero probability to unobserved events.** One zero factor destroys a
product. Smooth.

**Multiplying probabilities in linear space.** Underflow. Work in logs.

**Assuming independence because it is convenient.** Naive Bayes does this
knowingly and states it in its name. Doing it unknowingly, for example by
treating correlated validation folds as independent samples, produces confidence
intervals that are far too narrow ({{ch:mle-splits}}).

## 10. Connection to Previous Chapters

{{ch:math-notation}} supplied set notation — events are sets, and
{{eq:inclusion-exclusion}} is a statement about them — and the indicator
function, whose expectation {{ch:math-random-vars}} will show is a probability.
{{ch:math-functions}} supplied the logarithm, without which
{{eq:autoregressive}} is uncomputable, and the log-sum-exp trick used in
{{sec:8-practical-example}}.

Forward: {{ch:math-random-vars}} replaces events with random variables and
introduces distributions and expectation. {{ch:math-covariance}} measures the
relationships between them, giving a quantitative version of the dependence
discussed here. {{ch:math-inference}} asks what can be concluded from a finite
sample, and confronts the $\Prob(A \given B)$ versus $\Prob(B \given A)$
confusion again in the guise of p-values.

Beyond Part I: {{ch:ml-knn-nb}} builds the classifier of
{{sec:8-practical-example}} properly; {{ch:ml-metrics}} shows that precision is
$\Prob(\text{positive class} \given \text{predicted positive})$ and therefore
base-rate dependent; {{ch:llm-next-token}} is {{eq:autoregressive}}; and
{{ch:math-optimization}} shows that regularisation is a prior.

## 11. Exercises

**Beginner**

1. A fair six-sided die is rolled. Give $\Prob(\text{even})$,
   $\Prob(> 4)$, and $\Prob(\text{even or} > 4)$.
2. Two fair coins are flipped. Write out the sample space and give
   $\Prob(\text{exactly one head})$.
3. If $\Prob(A) = 0.3$ and $\Prob(B) = 0.5$ and they are independent, compute
   $\Prob(A \cap B)$ and $\Prob(A \cup B)$.
4. A bag holds 3 red and 7 blue balls. Two are drawn without replacement. What
   is the probability both are red?
5. State Bayes' theorem and name all four terms.

**Intermediate**

6. Rework the base-rate example with a base rate of 1 in 100 instead of 1 in
   1,000. How much does the posterior change?
7. In the same example, which improves the posterior more: raising sensitivity
   from 99% to 99.9%, or raising specificity from 95% to 99%? Compute both.
8. Show that if $A$ and $B$ are independent, so are $A$ and $B^{c}$.
9. Give two events that are conditionally independent given a third but not
   independent unconditionally.
10. Explain why a likelihood does not integrate to 1 over hypotheses, and give a
    concrete example.
11. Using {{eq:bayes-odds}}, compute how many consecutive positive tests are
    needed to push the posterior above 95% in the running example.

**Advanced**

12. Derive {{eq:bayes-odds}} from {{eq:bayes}}, showing where the evidence term
    cancels.
13. Prove the chain rule {{eq:chain-rule-prob}} by induction from
    {{eq:product-rule}}.
14. Simpson's paradox: construct a dataset where a treatment appears beneficial
    within every subgroup but harmful overall. Explain it in terms of
    conditioning.
15. Show that the naive Bayes decision rule is linear in the log-space of word
    counts, and relate this to logistic regression.

**Implementation**

16. Write `bayes_update(prior, likelihood_ratio)` operating in odds space, and
    use it to reproduce the sequential updates in {{sec:6-mathematical-foundation}}.
17. Simulate the two-children puzzle for both phrasings — "at least one is a
    girl" and "the elder is a girl" — and explain why the answers differ.
18. Extend the spam filter to report, for each classification, the three words
    that contributed most to the decision.
19. Empirically show that removing the add-one smoothing from the spam filter
    causes a catastrophic failure, and characterise exactly when.

**Reasoning**

20. A model achieves 99.9% accuracy detecting a condition present in 0.05% of
    cases. Is it useful? What would you need to know to decide?
21. Language models are trained on {{eq:autoregressive}}, which conditions each
    token only on what precedes it. What does that factorisation make easy, and
    what does it make hard?

## 12. Chapter Summary

Probability is a number in $[0,1]$ measuring expectation of an outcome, readable
either as a long-run frequency or as a degree of belief. The arithmetic is
identical under both readings; the interpretation matters when saying what a
result means.

The three axioms — non-negativity, total mass 1, and additivity over disjoint
events — generate everything else, including inclusion-exclusion and the
complement rule. Most probability errors are not arithmetic errors but sample
space errors, so enumerate explicitly when an answer surprises you.

Conditioning restricts the sample space and renormalises. The product rule
extends to the chain rule, which factorises a joint probability into a product
of conditionals — and that factorisation is exactly the definition of an
autoregressive language model.

Independence means $\Prob(A \cap B) = \Prob(A)\Prob(B)$: one event tells you
nothing about the other. Mutually exclusive events are maximally *dependent*,
not independent.

Bayes' theorem reverses a conditional: the posterior is proportional to the
likelihood times the prior. When the prior is extreme, strong evidence can leave
the posterior small — the base-rate fallacy — which is why a 99%-accurate test
for a 1-in-1,000 condition yields only a 1.9% posterior after a positive result.
In odds form the evidence term cancels, and evidence simply multiplies the odds
by the likelihood ratio; in log-odds form it adds.

The likelihood, read as a function of the hypothesis, is not a probability
distribution over hypotheses. Maximum likelihood ignores the prior; maximum a
posteriori includes it, and regularisation is precisely a prior in disguise.
