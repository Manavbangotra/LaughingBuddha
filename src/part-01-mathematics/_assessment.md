---
id: part-01-assessment
status: final
---

## How to use this

Four sections, in increasing order of difficulty and decreasing order of
structure.

The **knowledge check** verifies you can state and apply what Part I covered.
The **practical assignment** is the real test: a project requiring most of the
part at once. The **advanced challenge** has no worked route and is meant to
take a while. The **interview preparation** is calibrated against what is
actually asked, from junior to senior level.

Do the practical assignment. It is worth more than the other three combined,
for the same reason implementing backpropagation is worth more than reading
about it.

---

## Knowledge Check

Twenty questions. Answer without looking anything up; then check the chapter
noted in brackets.

1. What does the typography of a symbol tell you before you read the equation,
   and what are the five categories? [{{ch:math-notation}}]

2. Evaluate $\sum_{i=3}^{1} i^{2}$ and $\prod_{i=3}^{1} i$. Why are the answers
   what they are? [{{ch:math-notation}}]

3. State the difference between $\max_x f(x)$ and $\argmax_x f(x)$, and say
   which a classifier returns. [{{ch:math-notation}}]

4. Give three distinct reasons logarithms are used pervasively in machine
   learning. [{{ch:math-functions}}]

5. What is the maximum value of $\sigma'(x)$, and where does it occur? What does
   this imply for a stack of ten sigmoid layers? [{{ch:math-functions}}]

6. State the log-sum-exp trick and the numerical failure it prevents.
   [{{ch:math-functions}}]

7. What three things does a dot product measure, and what does it *fail* to
   distinguish? [{{ch:math-vectors}}]

8. Two vectors in $\R^{1000}$ are drawn at random. What angle between them
   should you expect, and why does this matter for embeddings?
   [{{ch:math-vectors}}]

9. An $m \times n$ matrix represents a map between which spaces? State the shape
   rule for a product. [{{ch:math-matrices}}]

10. Why does a neural network need nonlinearities between its linear layers?
    Answer in one sentence using composition. [{{ch:math-matrices}}]

11. Give the four norm axioms, and say which one the squared $L_2$ norm
    violates. [{{ch:math-norms}}]

12. Why does $L_1$ regularisation produce exactly-zero coefficients while $L_2$
    does not? Answer geometrically. [{{ch:math-norms}}]

13. State the SVD and the geometric meaning of each of its three factors. Why
    does it exist for every matrix when diagonalisation does not?
    [{{ch:math-eigen}}]

14. What does the Eckart-Young theorem guarantee, and why is that guarantee
    surprising? [{{ch:math-eigen}}]

15. A disease affects 1 in 1,000. A test has 99% sensitivity and 95%
    specificity. You test positive. Roughly what is the probability you have the
    disease, and why is it so far from 99%? [{{ch:math-probability}}]

16. Does $\E[X + Y] = \E[X] + \E[Y]$ require independence? Does
    $\Var(X + Y) = \Var(X) + \Var(Y)$? Explain the asymmetry.
    [{{ch:math-random-vars}}, {{ch:math-covariance}}]

17. Construct two variables with correlation exactly zero that are perfectly
    dependent. [{{ch:math-covariance}}]

18. What is $\Var(\vec{q}\T\vec{k})$ for independent unit-variance
    $d$-dimensional vectors, and what two pieces of deep-learning practice does
    it explain? [{{ch:math-covariance}}]

19. State precisely what a p-value is. Then state precisely what it is not.
    [{{ch:math-inference}}]

20. Why is the gradient the direction of steepest ascent? Give the one-line
    argument. [{{ch:math-derivatives}}]

**Bonus.** For $f(x) = ax^{2}/2$, derive the maximum stable learning rate for
gradient descent. What is the multidimensional generalisation?
[{{ch:math-optimization}}]

---

## Practical Assignment

### Build a linear model library from scratch

No scikit-learn, no PyTorch. NumPy only. The point is that every component
should be something you derived in Part I.

**What to build**

A small library exposing two estimators — `LinearRegression` and
`LogisticRegression` — each supporting $L_1$ and $L_2$ regularisation, trained
by gradient descent, with proper uncertainty reporting.

```text
linmodels/
├── linmodels/
│   ├── __init__.py
│   ├── preprocessing.py    # StandardScaler with fit/transform separation
│   ├── linear.py           # LinearRegression
│   ├── logistic.py         # LogisticRegression
│   ├── optim.py            # gradient descent, momentum, SGD, schedules
│   ├── metrics.py          # losses, accuracy, R², confidence intervals
│   └── diagnostics.py      # condition number, gradient checking
├── tests/
└── demo.ipynb
```

**Requirements**

*Preprocessing.* A scaler with separate `fit` and `transform`, fitted on
training data only. Write a test that fails if statistics leak from the test
set. [{{ch:math-covariance}}, {{ch:ds-leakage}}]

*Models.* Both estimators expose `fit`, `predict`, and `predict_proba` where
applicable. Losses are mean squared error and cross-entropy; the second must be
derived from maximum likelihood, not copied. [{{ch:math-optimization}}]

*Gradients.* Derive every gradient by hand and document the derivation in a
docstring. Then verify each against central differences to a tolerance of
$10^{-6}$. A model whose gradient check fails does not count as working.
[{{ch:math-derivatives}}]

*Numerical stability.* The logistic function must not overflow for inputs of
$\pm 1000$. Cross-entropy must not produce `nan` for confident predictions. Use
the techniques of {{ch:math-functions}}.

*Optimisation.* Implement plain gradient descent, momentum, and minibatch SGD,
plus at least one learning-rate schedule satisfying the Robbins-Monro
conditions. [{{ch:math-optimization}}]

*Regularisation.* $L_2$ by adding its gradient; $L_1$ by soft-thresholding,
because a plain subgradient step will not produce exact zeros. Never regularise
the intercept. [{{ch:math-norms}}]

*Diagnostics.* Report the condition number of the feature covariance matrix and
warn above $10^{4}$. Include a `gradient_check` utility.
[{{ch:math-eigen}}]

*Uncertainty.* Every reported metric carries a confidence interval. An accuracy
without one is an incomplete result. [{{ch:math-inference}}]

**Experiments to run and write up**

1. Fit on standardised and unstandardised features at several learning rates.
   Show the divergence, and connect it to the condition number and the
   $2/\lambda_{\max}$ threshold.
2. Sweep $\lambda$ for both penalties on data with 5 informative and 55
   irrelevant features. Plot the number of exact zeros against $\lambda$ and
   confirm that only $L_1$ produces them.
3. Compare batch sizes 1, 32, and full-batch at a fixed budget of *gradient
   evaluations*, not steps. Report which reaches the lower loss.
4. Compare two models on a shared test set using both an unpaired and a paired
   test. Report how much data the unpaired test would need to reach the same
   conclusion.
5. Reproduce the selection-bias experiment: 50 identically-good configurations,
   pick the best on validation, measure the optimism on fresh data.

**Acceptance criteria**

- Gradient checks pass for every model and every regularisation setting.
- No `nan` at extreme inputs; include the tests that prove it.
- Coefficients recovered on synthetic data within the confidence intervals you
  report.
- $L_1$ produces exact zeros; $L_2$ does not.
- A written analysis of each experiment above, explaining *why* using Part I.

---

## Advanced Challenge

### Implement PCA from scratch, and use it to prove Eckart-Young empirically

No route is given. Everything needed is in {{ch:math-eigen}},
{{ch:math-covariance}} and {{ch:math-norms}}.

**Part A — Two routes to the same answer.** Implement PCA twice: once by
eigendecomposing the covariance matrix, once by taking the SVD of the centred
data matrix. Prove they agree, both algebraically and numerically, and explain
the relationship between the eigenvalues of one and the singular values of the
other. Then determine, empirically, which is more numerically accurate on
badly-conditioned data, and explain why.

**Part B — Test the optimality claim.** Eckart-Young says the truncated SVD is
the best rank-$k$ approximation. Try to beat it. Implement at least three
alternative rank-$k$ approximations — random projection, greedy column
selection, and gradient descent directly on the factors — and measure the
Frobenius error of each against the SVD's. Confirm none wins, and explain what
would have to be true about the norm for one of them to.

**Part C — Denoising.** Construct a matrix that is genuinely rank-$r$ plus
Gaussian noise. Show that for a range of $k$, the rank-$k$ truncation is *closer
to the clean signal* than the noisy observations are. Characterise how the
optimal $k$ depends on the noise level and on $r$. This is the honest
justification for LoRA ({{ch:ft-lora}}), and you are testing when it holds.

**Part D — When it fails.** Construct a matrix whose spectrum decays so slowly
that truncation is worthless, and one where a single outlier row dominates the
top singular vector. In each case explain the failure and propose a diagnostic
that would have caught it before you truncated.

**Deliverable.** A written report with reproducible code, plots, and — most
importantly — the explanation of *why* in each part. A correct implementation
with no analysis is a partial answer.

---

## Interview Preparation

### Junior

1. What is a dot product and what does it measure?
2. What does it mean for a matrix to be singular?
3. Explain the difference between variance and standard deviation.
4. What is a gradient? Which way does it point?
5. Why do we take the log of probabilities?
6. What is overfitting, in terms of bias and variance?
7. Why standardise features before training?

### Mid-level

8. Why does attention divide by $\sqrt{d_k}$? Derive it.
9. Explain why $L_1$ regularisation produces sparse models and $L_2$ does not.
10. What is the condition number and why does it matter for training?
11. A model gets 94% on 1,000 test examples. What is your uncertainty in that
    number? How much data to halve it?
12. Explain the chain rule and its relationship to backpropagation.
13. Why is SGD preferred over full-batch gradient descent?
14. What is the difference between correlation and independence?
15. Explain what a p-value means, and one thing people commonly get wrong about
    it.

### Senior

16. Why is reverse-mode differentiation the right choice for training neural
    networks? What does it cost?
17. Explain the connection between PCA, LoRA, and low-rank approximation.
18. In high-dimensional loss landscapes, why are saddle points a bigger problem
    than local minima?
19. Derive the maximum stable learning rate for a quadratic, and explain what
    that implies for learning-rate schedules in practice.
20. Show that $L_2$ regularisation is equivalent to a Gaussian prior. What does
    the regularisation strength correspond to?
21. Your team runs 200 hyperparameter configurations and reports the best
    validation score. What is wrong with that number, and how large is the
    effect?
22. Two models differ by 0.4 percentage points on a shared benchmark. Walk
    through how you would determine whether the difference is real.

### Systems and judgement

23. A training run produces `nan` after 400 steps. Enumerate the mechanisms from
    Part I that could cause it, and the order in which you would check them.
24. A colleague proposes screening 500 features by their correlation with the
    target and keeping the top 50. What is wrong with this, and what would you
    do instead?
25. Your model's loss plateaus well above zero. Give three distinct explanations
    and an experiment that distinguishes them.

---

## Before moving on

You are ready for {{part:2}} when you can, without reference:

- Read an unfamiliar equation and identify the type and shape of every symbol.
- Multiply two matrices and state the output shape before computing.
- Explain what a dot product measures and why cosine similarity normalises it.
- Apply Bayes' theorem to a base-rate problem and get the counter-intuitive
  answer right.
- State what a p-value is without saying anything false.
- Compute a gradient by hand for a small composed function and verify it
  numerically.
- Explain why gradient descent works and the two distinct ways it fails.

If several of these are shaky, the specific chapters are named in the knowledge
check above. It is worth going back; everything from {{part:4}} onward assumes
this material without further comment.
