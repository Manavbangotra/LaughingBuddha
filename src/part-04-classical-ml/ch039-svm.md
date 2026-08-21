---
id: ml-svm
number: 39
part: IV
tier: focused
status: reviewed
requires: [ml-logistic, ml-metrics, math-vectors, math-optimization]
provides: [svm, maximum-margin, hinge-loss, support-vectors, kernel-trick,
           rbf-kernel, soft-margin, dual-formulation, representer-theorem]
citations: [cortes1995, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive the maximum-margin hyperplane and explain why margin is a sensible
   objective.
2. Derive the soft-margin formulation and interpret $C$.
3. Explain hinge loss and compare it against log loss.
4. Derive the dual and explain why it is what makes kernels possible.
5. Explain the kernel trick precisely, without hand-waving.
6. Choose between linear, polynomial and RBF kernels, and tune $\gamma$.
7. Explain why SVMs lost to gradient boosting and neural networks, and where
   they remain the right choice.

## 2. Why This Matters

Support vector machines were the dominant classifier from the mid-1990s to
roughly 2012, and they are worth a chapter now for reasons that have outlived
their dominance.

**The kernel trick is one of the genuinely beautiful ideas in the subject.** You
can work in an infinite-dimensional feature space without ever computing a
coordinate in it, because the algorithm only ever needs inner products. Seeing
exactly how that works — and it is four lines of algebra
({{sec:6-mathematical-foundation}}) — permanently changes how you think about
what a feature space *is*.

**Margin maximisation is a different answer to overfitting.** Everything else in
this part controls complexity by shrinking coefficients, limiting depth, or
averaging. The SVM instead asks for the boundary that stays as far as possible
from the data, and derives everything from that geometric requirement. It is
worth having more than one idea about what regularisation means.

**Attention is a kernel-shaped computation.** Scaled dot-product attention
({{ch:tf-scaled-dot-product}}) computes a similarity between every query and
every key, normalises it, and takes a weighted combination of values. That is
structurally the same operation as {{eq:kernel-prediction}}, with a learned
similarity instead of a fixed one, and the same $O(N^{2})$ cost for the same
reason.

## 3. Prerequisites

{{ch:math-vectors}} for hyperplanes, norms and projections — the derivation is
mostly geometry. {{ch:math-optimization}} for constrained optimisation and
Lagrange multipliers. {{ch:ml-logistic}} for the loss comparison.
{{ch:ml-metrics}} for the validation this chapter's two-parameter grid needs.

## 4. Intuitive Explanation

### 4.1 The widest street

Many hyperplanes separate two linearly separable classes. Which is best?

The SVM's answer: the one furthest from both classes. Equivalently, imagine
widening a street around the boundary until it touches points on either side —
the best boundary is the centre line of the widest such street.

```text
       ●   ●                        ●   ●
    ●     ●        ╱             ●     ●     ╎    ╎    ╎
       ●        ╱                   ●        ╎    ╎    ╎
              ╱                              ╎    ╎    ╎
           ╱      ▲   ▲                 ╎    ╎    ╎   ▲   ▲
        ╱      ▲      ▲              ╎  ╎    ╎    ╎ ▲    ▲
     ╱      ▲                              ╎    ╎ ▲
                                     └──margin──┘
   a separating line                the WIDEST street
   (one of infinitely many)         and its centre line
```

The intuition for why width is worth maximising: a wide margin means the
boundary can be perturbed without changing any classification, so the solution
is stable under sampling variation. That is variance reduction in the sense of
{{ch:ml-metrics}}, arrived at geometrically rather than statistically.

### 4.2 Support vectors

Only the points touching the street determine the boundary. These are the
**support vectors**. Every other point could be moved — or deleted — without
changing the solution at all.

This is a strong and unusual property. Logistic regression's coefficients depend
on every observation; an SVM's depend on a handful. It makes the model compact
(you need only store the support vectors), and it makes it *sensitive*: the
solution is determined by the points nearest the boundary, which are exactly the
ambiguous, often mislabelled ones.

### 4.3 Soft margin

Real data is not separable, and demanding a perfect split would fail or produce
an absurdly narrow margin around one outlier. The **soft margin** permits
violations and charges for them, with $C$ setting the price:

- **Large $C$** — violations are expensive. Narrow margin, few errors, more
  overfitting. As $C \to \infty$ you recover the hard margin.
- **Small $C$** — violations are cheap. Wide margin, more training errors, more
  regularisation.

$C$ is inversely a regularisation strength, exactly like scikit-learn's `C` in
logistic regression {{cite:pedregosa2011}}, and for the same reason: it
multiplies the loss rather than the penalty.

### 4.4 The kernel trick

XOR is not linearly separable in two dimensions. Map it to three by adding
$x_1 x_2$ and it is. That is basis expansion, and {{ch:ml-linear-regression}}
already used it.

The problem is cost. A degree-$d$ polynomial expansion of $D$ features has
$O(D^{d})$ terms; a Gaussian expansion has infinitely many. You cannot write the
coordinates down.

The trick: **the SVM's dual formulation only ever needs inner products between
pairs of points.** If some function $K(\vec{x}, \vec{z})$ equals
$\phi(\vec{x})\T\phi(\vec{z})$ for a feature map $\phi$, you can substitute $K$
everywhere and never compute $\phi$ at all.

The RBF kernel $K(\vec{x},\vec{z}) = \exp(-\gamma\|\vec{x}-\vec{z}\|^{2})$
corresponds to an **infinite-dimensional** feature map, and evaluating it costs
one subtraction, one norm and one exponential.
{{sec:6-mathematical-foundation}} shows the expansion explicitly, and
{{sec:7-implementation}} verifies the identity numerically.

## 5. Formal Explanation

### 5.1 The hard-margin problem

With labels $y_i \in \{-1, +1\}$ — the convention differs from
{{ch:ml-logistic}}'s $\{0,1\}$, and it is what makes the algebra clean — a
hyperplane is $\vec{w}\T\vec{x} + b = 0$. The distance from $\vec{x}_i$ to it is
$|\vec{w}\T\vec{x}_i + b| / \|\vec{w}\|$.

Scaling $(\vec{w}, b)$ by any constant describes the same hyperplane, so fix the
scale by requiring $\min_i y_i(\vec{w}\T\vec{x}_i + b) = 1$. The margin is then
$2/\|\vec{w}\|$, and maximising it means minimising $\|\vec{w}\|$:

$$
\min_{\vec{w}, b} \tfrac{1}{2}\|\vec{w}\|^{2}
\quad \text{subject to} \quad
y_i(\vec{w}\T\vec{x}_i + b) \ge 1 \;\; \forall i
$$ (eq:hard-margin)

A convex quadratic programme with linear constraints: a unique global optimum,
and no local minima to worry about.

### 5.2 Soft margin and hinge loss

Introduce slack $\xi_i \ge 0$:

$$
\min_{\vec{w},b,\vecgreek{\xi}} \tfrac{1}{2}\|\vec{w}\|^{2}
  + C\sum_{i=1}^{N}\xi_i
\quad \text{s.t.} \quad
y_i(\vec{w}\T\vec{x}_i+b) \ge 1 - \xi_i, \;\; \xi_i \ge 0
$$ (eq:soft-margin)

The optimal slack is $\xi_i = \max(0, 1 - y_i(\vec{w}\T\vec{x}_i+b))$, so
{{eq:soft-margin}} is equivalent to unconstrained minimisation of

$$
\frac{1}{N}\sum_{i=1}^{N}\max\big(0, 1 - y_i f(\vec{x}_i)\big)
  + \frac{\lambda}{2}\|\vec{w}\|^{2},
\qquad \lambda = \frac{1}{CN}
$$ (eq:hinge-objective)

**This is the single most clarifying fact about SVMs**: an SVM is
$\ell_2$-regularised empirical risk minimisation with the **hinge loss**. It sits
in the same framework as everything else in Part IV, differing only in the loss.

{#tbl:hinge-vs-log caption="Hinge loss against log loss. The zero-loss region is what produces support vectors; the lack of a probabilistic interpretation is what costs calibration."}

| | Hinge | Log loss |
|---|---|---|
| Formula | $\max(0, 1-yf)$ | $\log(1+e^{-yf})$ |
| Zero loss when | $yf \ge 1$ | never |
| Gradient for confident correct | exactly 0 | small but non-zero |
| Sparse solution | yes — support vectors | no |
| Probabilistic reading | none | Bernoulli likelihood |
| Calibrated output | no | yes |

The row that matters is the first. Hinge loss is exactly zero once a point is
correctly classified beyond the margin, so such points contribute nothing to the
gradient and nothing to the solution. That is where sparsity comes from, and it
is why the SVM has support vectors and logistic regression does not.

### 5.3 The dual

Introducing Lagrange multipliers $\alpha_i \ge 0$ and eliminating $\vec{w}$ and
$b$ gives the dual:

$$
\max_{\vecgreek{\alpha}} \sum_{i=1}^{N}\alpha_i
  - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \,\vec{x}_i\T\vec{x}_j
\quad \text{s.t.} \quad
0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0
$$ (eq:svm-dual)

with the solution recovered as $\vec{w} = \sum_i \alpha_i y_i \vec{x}_i$.

Two facts about {{eq:svm-dual}} carry the rest of the chapter.

**The data appears only as inner products $\vec{x}_i\T\vec{x}_j$.** Nowhere else.
That is the opening the kernel trick walks through.

**Most $\alpha_i$ are zero.** By complementary slackness, $\alpha_i > 0$ only for
points on or inside the margin — the support vectors.

### 5.4 Kernels

Replace every inner product by $K(\vec{x}_i, \vec{x}_j)$. Prediction becomes

$$
f(\vec{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\vec{x}_i, \vec{x}) + b
$$ (eq:kernel-prediction)

a weighted sum of similarities to the support vectors. Compare
{{ch:ml-knn-nb}}: this is k-NN with learned weights and a learned neighbourhood
size, and compare {{ch:tf-scaled-dot-product}}: it is attention with a fixed
similarity function.

{#tbl:kernels caption="Common kernels. RBF is the default for a reason; the others are for specific structure or for speed."}

| Kernel | $K(\vec{x},\vec{z})$ | Feature space | Use when |
|---|---|---|---|
| Linear | $\vec{x}\T\vec{z}$ | the original | $D$ large, $N$ large, text |
| Polynomial | $(\gamma\vec{x}\T\vec{z}+r)^{d}$ | degree-$d$ monomials | interactions of known order |
| RBF | $\exp(-\gamma\|\vec{x}-\vec{z}\|^{2})$ | infinite-dimensional | the default |
| Sigmoid | $\tanh(\gamma\vec{x}\T\vec{z}+r)$ | — | rarely; not always valid |

**$\gamma$ is the most important hyperparameter after $C$.** It sets the width of
the RBF bump. Large $\gamma$ means each support vector influences only its
immediate neighbourhood, giving a wiggly boundary that overfits; small $\gamma$
means broad influence and an almost linear boundary. $C$ and $\gamma$ interact
strongly, so they must be tuned **jointly** on a two-dimensional logarithmic
grid.

> WARNING: RBF kernels require standardised features, for the same reason k-NN
> does ({{ch:ml-knn-nb}}): $\|\vec{x}-\vec{z}\|^{2}$ is dominated by whichever
> feature has the largest scale. This is not optional, and the failure is silent.

### 5.5 Why SVMs fell out of favour

Three reasons, and they are worth knowing because they are about scaling rather
than about the idea being wrong.

**Cost.** Training is roughly $O(N^{2})$ to $O(N^{3})$ and the kernel matrix is
$O(N^{2})$ in memory. At $N = 10^{6}$ that matrix is eight terabytes. Gradient
boosting and SGD are near-linear.

**No probabilities.** Hinge loss has no likelihood interpretation, so the output
is an uncalibrated score. Platt scaling can be fitted afterwards, but it needs
an internal cross-validation and is a patch on the output rather than a property
of the model.

**Learned beats fixed.** A kernel is a *fixed* similarity chosen in advance. Deep
networks learn the representation from the data, and on images and text that
turned out to be worth far more than any hand-chosen kernel.

Where they remain the right choice: small to medium datasets ($N \lesssim
10^{5}$), high-dimensional problems where $D > N$ (the margin argument is
dimension-independent, which is a genuine and underappreciated strength), and
text classification with a linear kernel, where linear SVMs are still
competitive and extremely fast.

## 6. Mathematical Foundation

### 6.1 Why $2/\|\vec{w}\|$ is the margin

Take two points $\vec{x}_+$ and $\vec{x}_-$ on the two margin boundaries:
$\vec{w}\T\vec{x}_+ + b = 1$ and $\vec{w}\T\vec{x}_- + b = -1$. Subtracting,

$$
\vec{w}\T(\vec{x}_+ - \vec{x}_-) = 2
$$

The margin is the component of $\vec{x}_+ - \vec{x}_-$ along the unit normal
$\vec{w}/\|\vec{w}\|$:

$$
\text{margin} = \frac{\vec{w}\T(\vec{x}_+-\vec{x}_-)}{\|\vec{w}\|}
 = \frac{2}{\|\vec{w}\|}
$$ (eq:margin-width)

So maximising the margin is minimising $\|\vec{w}\|$, and
$\tfrac{1}{2}\|\vec{w}\|^{2}$ is used instead because it is differentiable and
convex with the same minimiser.

Notice what {{eq:margin-width}} implies: the SVM's $\ell_2$ penalty is not a
regularisation term bolted on, as it is in {{ch:ml-linear-regression}}. It *is*
the objective. The margin interpretation and the ridge penalty are the same
thing seen from two directions, which is a satisfying and non-obvious identity.

### 6.2 Deriving the dual

The Lagrangian of {{eq:hard-margin}}, with $\alpha_i \ge 0$:

$$
\Like(\vec{w},b,\vecgreek{\alpha}) = \tfrac{1}{2}\|\vec{w}\|^{2}
 - \sum_i \alpha_i\big[y_i(\vec{w}\T\vec{x}_i+b) - 1\big]
$$

Stationarity in $\vec{w}$ and $b$:

$$
\frac{\partial \Like}{\partial \vec{w}} = \vec{w} - \sum_i \alpha_i y_i \vec{x}_i = 0
 \;\Rightarrow\; \vec{w} = \sum_i \alpha_i y_i \vec{x}_i
$$ (eq:w-from-alpha)

$$
\frac{\partial \Like}{\partial b} = -\sum_i \alpha_i y_i = 0
 \;\Rightarrow\; \sum_i \alpha_i y_i = 0
$$

Substituting {{eq:w-from-alpha}} back and simplifying gives
{{eq:svm-dual}}. The essential step is that $\|\vec{w}\|^{2} = \sum_{i,j}
\alpha_i\alpha_j y_i y_j \vec{x}_i\T\vec{x}_j$ — the data enters only through
pairwise inner products, and it does so because $\vec{w}$ is a *linear
combination of the training points*.

That last observation is the **representer theorem** in its simplest form: for
any objective consisting of a loss on the training predictions plus a penalty
increasing in $\|\vec{w}\|$, the optimal $\vec{w}$ lies in the span of the
training data. Directions orthogonal to every training point contribute nothing
to any prediction and only add to the penalty, so they are set to zero. This is
what guarantees the kernel substitution is not merely convenient but *complete*
— no solution is lost by working in the dual.

Complementary slackness gives the sparsity: $\alpha_i\big[y_i f(\vec{x}_i) -
1\big] = 0$, so either $\alpha_i = 0$ or the point sits exactly on the margin.

### 6.3 The RBF kernel's infinite feature map

Take $\gamma = 1/2$ and scalar inputs for clarity. Then

$$
K(x,z) = e^{-\frac{1}{2}(x-z)^{2}}
 = e^{-\frac{x^{2}}{2}} e^{-\frac{z^{2}}{2}} e^{xz}
$$

Expand the last factor as its Taylor series:

$$
e^{xz} = \sum_{k=0}^{\infty} \frac{(xz)^{k}}{k!}
$$

so

$$
K(x,z) = \sum_{k=0}^{\infty}
  \left(e^{-\frac{x^{2}}{2}}\frac{x^{k}}{\sqrt{k!}}\right)
  \left(e^{-\frac{z^{2}}{2}}\frac{z^{k}}{\sqrt{k!}}\right)
 = \phi(x)\T\phi(z)
$$ (eq:rbf-expansion)

with the explicit feature map

$$
\phi(x) = e^{-\frac{x^{2}}{2}}
  \left(1, \; x, \; \frac{x^{2}}{\sqrt{2!}}, \;
        \frac{x^{3}}{\sqrt{3!}}, \; \dots\right)
$$ (eq:rbf-feature-map)

An **infinite-dimensional** vector containing every polynomial power, damped by
a Gaussian envelope. Computing it is impossible; computing its inner product
with another such vector costs one exponential.

That is the kernel trick, in full, with no hand-waving:
{{sec:7-implementation}} truncates {{eq:rbf-feature-map}} at $k = 30$ and
confirms the inner product matches $K$ to nine decimal places.

A function is a valid kernel exactly when it is symmetric and positive
semi-definite — Mercer's condition — which guarantees such a $\phi$ exists. In
practice one uses kernels already known to satisfy it.

### 6.4 Hinge loss and the gradient

$$
\ell_{\text{hinge}}(y, f) = \max(0, 1-yf),
\qquad
\frac{\partial \ell}{\partial f} =
\begin{cases}
-y & \text{if } yf < 1\\
0 & \text{if } yf > 1
\end{cases}
$$ (eq:hinge-gradient)

undefined at exactly $yf = 1$, where any subgradient in $[-y, 0]$ will do.

Compare the log-loss gradient $\sigma(-yf)\cdot(-y)$ from {{ch:ml-logistic}},
which is never exactly zero. That single difference produces both of the SVM's
distinguishing behaviours: sparsity (confident correct points drop out of the
problem entirely) and the absence of calibration (the model has no reason to
distinguish "just barely beyond the margin" from "enormously far beyond it", so
its scores carry no probabilistic information).

Subgradient descent on {{eq:hinge-objective}} is a perfectly good way to fit a
*linear* SVM, and is what large-scale implementations use. It cannot be
kernelised directly, which is why the dual matters.

## 7. Implementation

```python {tier=A name=svm-from-scratch}
"""SVM from scratch: margin geometry, hinge loss, the dual, and the kernel
trick verified numerically.
"""
import math

import numpy as np

rng = np.random.default_rng(0)

# --- section 6.3: the kernel trick is an identity, not an analogy -----------
print("=" * 72)
print("the RBF kernel IS an inner product in an infinite space (eq. 39.11)")
print("=" * 72)


def rbf_scalar(x, z, gamma=0.5):
    return float(np.exp(-gamma * (x - z) ** 2))


def rbf_feature_map(x, k_max=30):
    """The explicit map of eq. 39.12, truncated at k_max. For gamma = 1/2.

    math.factorial, not a numpy product: 30! is about 2.7e32 and overflows
    int64 silently, which turns the sqrt into a nan.
    """
    k = np.arange(k_max + 1)
    fact = np.array([float(math.factorial(int(i))) for i in k])
    return np.exp(-x ** 2 / 2) * x.astype(float) ** k / np.sqrt(fact) \
        if isinstance(x, np.ndarray) else \
        np.exp(-x ** 2 / 2) * np.power(float(x), k) / np.sqrt(fact)


print(f"{'x':>7} {'z':>7} {'K(x,z) directly':>18} "
      f"{'phi(x).phi(z), 30 terms':>26} {'difference':>13}")
for x, z in ((0.5, 0.7), (1.0, -1.0), (0.0, 2.0), (1.5, 1.6), (-2.0, 2.5)):
    direct = rbf_scalar(x, z)
    viafeat = float(rbf_feature_map(x) @ rbf_feature_map(z))
    print(f"{x:>7.1f} {z:>7.1f} {direct:>18.12f} {viafeat:>26.12f} "
          f"{abs(direct - viafeat):>13.2e}")

print("\nThe two columns agree to twelve decimal places. The left one costs")
print("one exponential; the right one required truncating an INFINITE vector")
print("at 30 terms and taking a dot product. That is the entire trick: the")
print("algorithm never needs phi, only phi(x).phi(z), and K computes that")
print("directly.")

# --- section 5.1: the margin, and which points determine it -----------------
print("\n" + "=" * 72)
print("the maximum-margin hyperplane and its support vectors")
print("=" * 72)


def fit_linear_svm(X, y, C=1.0, n_iter=8000, lr0=0.5):
    """Subgradient descent on eq. 39.3, the unconstrained hinge form.

    y must be in {-1, +1}. lambda = 1/(C*N) as in eq. 39.3.
    """
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    lam = 1.0 / (C * n)
    for t in range(1, n_iter + 1):
        lr = lr0 / (1 + 0.001 * t)
        margins = y * (X @ w + b)
        viol = margins < 1                       # eq. 39.14: zero gradient
        gw = lam * w - (X[viol].T @ y[viol]) / n
        gb = -y[viol].sum() / n
        w -= lr * gw
        b -= lr * gb
    return w, b


# a well-separated problem, so the geometry is unambiguous
n = 200
Xa = rng.normal([-2.5, -1.5], 0.7, (n // 2, 2))
Xb = rng.normal([2.5, 1.8], 0.7, (n // 2, 2))
X = np.vstack([Xa, Xb])
y = np.r_[-np.ones(n // 2), np.ones(n // 2)]

w, b = fit_linear_svm(X, y, C=100.0)
margins = y * (X @ w + b)
print(f"||w||          = {np.linalg.norm(w):.4f}")
print(f"margin 2/||w|| = {2 / np.linalg.norm(w):.4f}   (eq. 39.7)")
print(f"training accuracy = {np.mean(np.sign(X @ w + b) == y):.4f}")
print("\nthe constraint of eq. 39.1 is y_i * f(x_i) >= 1. The margins,")
print("sorted, closest points first:")
srt = np.sort(margins)
print("  five smallest :", " ".join(f"{v:6.3f}" for v in srt[:5]))
print("  five largest  :", " ".join(f"{v:6.3f}" for v in srt[-5:]))
print(f"  median        : {np.median(margins):6.3f}")
print(f"  points within 20% of the closest: "
      f"{int((margins < 1.2 * srt[0]).sum())} of {n}")
print("\nA few points sit at the edge of the street and the rest are several")
print("times further out. Section 6.4 says every point with y*f > 1")
print("contributes EXACTLY zero gradient, so the solution cannot depend on")
print("the far ones at all. Listing 2 makes that exact rather than visual,")
print("by solving the dual and reading off which alpha_i are nonzero.")

# --- section 5.2: what C does -----------------------------------------------
print("\n" + "=" * 72)
print("C prices margin violations (eq. 39.2)")
print("=" * 72)
# a small, well-separated sample with three mislabelled points planted deep
# inside the opposite class — the situation where C genuinely matters
# a small, OVERLAPPING sample with five mislabelled points planted deep
# inside the opposite class. C matters most when the classes overlap and the
# labels are noisy — which is when the boundary's exact position is contested.
n = 60
ctr = np.array([0.8, 0.5])
Xo = np.vstack([rng.normal(-ctr, 1.0, (n // 2, 2)),
                rng.normal(ctr, 1.0, (n // 2, 2))])
yo = np.r_[-np.ones(n // 2), np.ones(n // 2)]
Xo[:5] = rng.normal(ctr * 1.3, 0.25, (5, 2))
yo[:5] = -1.0                      # planted mislabels

Xte = np.vstack([rng.normal(-ctr, 1.0, (4000, 2)),
                 rng.normal(ctr, 1.0, (4000, 2))])
yte = np.r_[-np.ones(4000), np.ones(4000)]

print(f"{'C':>10} {'||w||':>9} {'margin':>9} {'train acc':>11} "
      f"{'test acc':>10} {'# violating':>13}")
for C in (0.003, 0.03, 0.3, 3.0, 100.0):
    wc, bc = fit_linear_svm(Xo, yo, C=C)
    m = yo * (Xo @ wc + bc)
    print(f"{C:>10} {np.linalg.norm(wc):>9.4f} "
          f"{2 / max(np.linalg.norm(wc), 1e-9):>9.4f} "
          f"{np.mean(np.sign(Xo @ wc + bc) == yo):>11.4f} "
          f"{np.mean(np.sign(Xte @ wc + bc) == yte):>10.4f} "
          f"{int((m < 1).sum()):>13}")
print("\nFive of the sixty training points are mislabelled and sit deep")
print("inside the wrong class. Small C treats them as violations to be")
print("tolerated and keeps a wide margin; large C tries harder to classify")
print("them and pulls the boundary towards them.")
print("\nRead the margin and violation columns: they move by an order of")
print("magnitude and monotonically, which is exactly what eq. 39.2 says C")
print("controls. At C = 0.003 violations are so cheap that the model")
print("essentially gives up, keeping a margin of 21 and a training accuracy")
print("of 0.65.")
print("\nNow read the test-accuracy column, which is the honest part: past")
print("C = 0.03 it barely moves. C is a real lever on the SHAPE of the")
print("solution and, on this data, a small one on its accuracy. That is")
print("worth knowing before you spend a grid search on it — the failure to")
print("watch for is C far too SMALL, which is visible immediately in the")
print("training accuracy, rather than C somewhat too large.")
print("\nNote the direction: larger C means LESS regularisation, because C")
print("multiplies the loss rather than the penalty (eq. 39.2). This is the")
print("same inverted convention as scikit-learn's logistic regression, and")
print("it catches people out constantly.")

# --- section 6.4: hinge vs log loss -----------------------------------------
print("\n" + "=" * 72)
print("hinge vs log loss (table 39.1)")
print("=" * 72)
print(f"{'y*f':>7} {'hinge loss':>12} {'hinge grad':>12} "
      f"{'log loss':>10} {'log grad':>10}")
for m in (-2.0, -0.5, 0.0, 0.5, 0.999, 1.0, 1.5, 3.0, 10.0):
    hl = max(0.0, 1 - m)
    hg = -1.0 if m < 1 else 0.0
    ll = float(np.logaddexp(0, -m))
    lg = -1.0 / (1 + np.exp(m))
    print(f"{m:>7.3f} {hl:>12.4f} {hg:>12.4f} {ll:>10.4f} {lg:>10.6f}")

print("\nAt y*f = 3 the hinge gradient is EXACTLY zero and the log-loss")
print("gradient is -0.0474. That exact zero is where support vectors come")
print("from: a confidently correct point leaves the optimisation problem")
print("entirely. It is also where the lack of calibration comes from — the")
print("model has no reason to prefer y*f = 3 to y*f = 10, so its scores")
print("carry no probabilistic information.")
```

```python {tier=A name=kernel-svm}
"""A kernel SVM by SMO-style dual optimisation, and the C/gamma grid.
"""
import numpy as np

rng = np.random.default_rng(7)


def kernel_matrix(A, B, kind="rbf", gamma=1.0, degree=3, coef0=1.0):
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kind == "rbf":
        sq = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
              - 2 * A @ B.T)
        return np.exp(-gamma * np.maximum(sq, 0.0))
    raise ValueError(kind)


class KernelSVM:
    """Simplified SMO on the dual of eq. 39.4.

    Pairs of multipliers are optimised at a time, because the equality
    constraint sum(alpha_i y_i) = 0 means a single alpha cannot move alone.
    """

    def __init__(self, C=1.0, kind="rbf", gamma=1.0, degree=3,
                 max_passes=12, tol=1e-3, seed=0):
        self.C, self.kind, self.gamma, self.degree = C, kind, gamma, degree
        self.max_passes, self.tol, self.seed = max_passes, tol, seed

    def fit(self, X, y):
        rs = np.random.default_rng(self.seed)
        n = len(y)
        K = kernel_matrix(X, X, self.kind, self.gamma, self.degree)
        a = np.zeros(n)
        b = 0.0
        passes = 0
        while passes < self.max_passes:
            changed = 0
            f = (a * y) @ K + b
            for i in range(n):
                Ei = f[i] - y[i]
                if ((y[i] * Ei < -self.tol and a[i] < self.C)
                        or (y[i] * Ei > self.tol and a[i] > 0)):
                    j = int(rs.integers(0, n - 1))
                    j = j + (j >= i)
                    Ej = f[j] - y[j]
                    ai_old, aj_old = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0.0, aj_old - ai_old), \
                               min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0.0, ai_old + aj_old - self.C), \
                               min(self.C, ai_old + aj_old)
                    if H - L < 1e-12:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= -1e-12:
                        continue
                    a[j] = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj_old) < 1e-9:
                        continue
                    a[i] = ai_old + y[i] * y[j] * (aj_old - a[j])
                    b1 = (b - Ei - y[i] * (a[i] - ai_old) * K[i, i]
                          - y[j] * (a[j] - aj_old) * K[i, j])
                    b2 = (b - Ej - y[i] * (a[i] - ai_old) * K[i, j]
                          - y[j] * (a[j] - aj_old) * K[j, j])
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    f = (a * y) @ K + b
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            if changed == 0:
                break
        self.a, self.b, self.X, self.y = a, b, X, y
        self.sv = a > 1e-8
        return self

    def decision(self, Z):
        K = kernel_matrix(Z, self.X[self.sv], self.kind, self.gamma,
                          self.degree)
        return K @ (self.a[self.sv] * self.y[self.sv]) + self.b

    def predict(self, Z):
        return np.sign(self.decision(Z))


def make_rings(n):
    r = rng.uniform(0, 3, n)
    th = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return X, np.where(r > 1.6, 1.0, -1.0)


def make_xor(n):
    X = rng.normal(size=(n, 2))
    return X, np.where((X[:, 0] > 0) ^ (X[:, 1] > 0), 1.0, -1.0)


def make_linear(n):
    X = rng.normal(size=(n, 2))
    return X, np.where(X[:, 0] + X[:, 1] > 0, 1.0, -1.0)


# --- the kernel decides what the boundary can be ----------------------------
print("=" * 72)
print("kernel choice is inductive bias (table 39.2)")
print("=" * 72)
datasets = {"linear boundary": make_linear, "concentric rings": make_rings,
            "XOR": make_xor}
print(f"{'dataset':<20} {'linear':>9} {'poly d=2':>10} {'poly d=3':>10} "
      f"{'RBF':>8}")
for name, gen in datasets.items():
    Xtr, ytr = gen(300)
    Xte, yte = gen(2000)
    row = []
    for kind, kw in (("linear", {}), ("poly", dict(degree=2, gamma=0.5)),
                     ("poly", dict(degree=3, gamma=0.5)),
                     ("rbf", dict(gamma=0.5))):
        m = KernelSVM(C=1.0, kind=kind, **kw).fit(Xtr, ytr)
        row.append((m.predict(Xte) == yte).mean())
    print(f"{name:<20} {row[0]:>9.4f} {row[1]:>10.4f} {row[2]:>10.4f} "
          f"{row[3]:>8.4f}")

print("\nThe linear kernel solves the linear problem and nothing else — it is")
print("at 0.63-0.65 on rings and XOR, barely above chance.")
print("\nThe degree-2 polynomial is the best model on BOTH nonlinear")
print("problems, and that is not luck: its feature space contains exactly")
print("the monomials x0^2, x1^2 and x0*x1. A ring boundary is")
print("x0^2 + x1^2 = r^2 and XOR's is x0*x1 = 0, so both are LINEAR in that")
print("space. When you know the form of the boundary, the matching kernel")
print("beats the general-purpose one.")
print("\nRBF is close behind on everything without being told anything,")
print("which is why it is the default: it assumes only smoothness. Choosing")
print("a kernel is choosing an assumption, exactly as in Chapter 31 — and")
print("the RBF's assumption is the weakest one available.")

# --- sparsity: how many points actually matter ------------------------------
print("\n" + "=" * 72)
print("sparsity: the model is stored as support vectors (section 5.3)")
print("=" * 72)
Xtr, ytr = make_rings(400)
Xte, yte = make_rings(3000)
print(f"{'C':>8} {'support vectors':>17} {'fraction':>10} {'test acc':>10}")
for C in (0.1, 1.0, 10.0, 100.0):
    m = KernelSVM(C=C, kind="rbf", gamma=0.5).fit(Xtr, ytr)
    print(f"{C:>8} {int(m.sv.sum()):>17} {m.sv.mean():>10.3f} "
          f"{(m.predict(Xte) == yte).mean():>10.4f}")
print("\nSmaller C means a wider margin, so MORE points fall on or inside it")
print("and more become support vectors. The stored model is exactly those")
print("points — which is also why prediction cost grows with them.")

# --- section 5.4: C and gamma must be tuned jointly -------------------------
print("\n" + "=" * 72)
print("C and gamma interact: tune them on a 2-D log grid")
print("=" * 72)
Xtr, ytr = make_rings(400)
Xva, yva = make_rings(1000)
Xte, yte = make_rings(3000)

gammas = [0.01, 0.1, 1.0, 10.0, 100.0]
Cs = [0.1, 1.0, 10.0, 100.0]
print(f"{'':>8}" + "".join(f"{'g=' + str(g):>10}" for g in gammas))
best = (None, -1.0)
for C in Cs:
    row = []
    for g in gammas:
        m = KernelSVM(C=C, kind="rbf", gamma=g).fit(Xtr, ytr)
        acc = (m.predict(Xva) == yva).mean()
        row.append(acc)
        if acc > best[1]:
            best = ((C, g), acc)
    print(f"C={C:<6}" + "".join(f"{a:>10.4f}" for a in row))

(C_star, g_star), _ = best
final = KernelSVM(C=C_star, kind="rbf", gamma=g_star).fit(Xtr, ytr)
print(f"\nbest on validation: C={C_star}, gamma={g_star}")
print(f"test accuracy      : {(final.predict(Xte) == yte).mean():.4f}")
print(f"support vectors    : {int(final.sv.sum())} of {len(ytr)}")

print("\nThe grid is not separable into two one-dimensional searches: the")
print("best gamma at C=0.1 is not the best gamma at C=100. Both control")
print("effective complexity — gamma through the width of each bump, C")
print("through how hard the fit is pushed — so a large gamma can be rescued")
print("by a small C and vice versa. Always search the plane.")

# --- gamma alone: from almost-linear to memorising --------------------------
print("\n" + "=" * 72)
print("gamma is the RBF bandwidth: too large and it memorises")
print("=" * 72)
print(f"{'gamma':>9} {'train acc':>11} {'test acc':>10} {'SVs':>6} "
      f"{'behaviour':<28}")
for g in (0.001, 0.01, 0.1, 1.0, 10.0, 200.0):
    m = KernelSVM(C=10.0, kind="rbf", gamma=g).fit(Xtr, ytr)
    tr = (m.predict(Xtr) == ytr).mean()
    te = (m.predict(Xte) == yte).mean()
    note = ("under-fitting: one broad bump" if g <= 0.01
            else "memorising" if tr - te > 0.05 else "")
    print(f"{g:>9} {tr:>11.4f} {te:>10.4f} {int(m.sv.sum()):>6} {note:<28}")
print("\nBoth ends fail, in opposite ways. At gamma = 0.001 every point is")
print("effectively at distance zero from every other, the kernel is nearly")
print("constant, and almost every point becomes a support vector — the")
print("model is one broad bump and cannot separate anything.")
print("\nAt gamma = 200 each support vector influences only a tiny ball")
print("around itself, so the model approaches a lookup table: training")
print("accuracy 1.0000 and a test accuracy 11 points below its peak, with")
print("almost every point needed as a support vector because nothing")
print("generalises to its neighbours.")
print("\nOne knob, the whole bias-variance trade — the same curve as k in")
print("k-NN and depth in a tree (Chapter 34). Note that the support-vector")
print("count is a useful diagnostic in itself: when nearly every point is a")
print("support vector, the model has stopped generalising.")
```

## 8. Practical Example

```python {tier=A name=svm-in-practice}
"""When an SVM is the right choice, and what it costs — against the rest of
Part IV, using scikit-learn so the comparison is fair.
"""
import time

import numpy as np

rng = np.random.default_rng(19)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def roc_auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int((y == 1).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * ((y != 1).sum())))


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    # --- 1. the regime where SVMs still win: D > N ---------------------------
    print("=" * 72)
    print("1. high dimension, few samples — the SVM's remaining home ground")
    print("=" * 72)
    print("The margin argument of section 6.1 does not mention the dimension,")
    print("which is why SVMs degrade gracefully when D exceeds N.\n")
    print(f"{'N':>6} {'D':>6} {'logistic':>10} {'linear SVM':>12} "
          f"{'RBF SVM':>9} {'boosting':>10}")
    for N, D in ((40, 500), (80, 500), (200, 500), (1000, 500)):
        w_true = np.zeros(D)
        w_true[rng.choice(D, 15, replace=False)] = rng.normal(0, 1.5, 15)
        Xall = rng.normal(size=(N + 3000, D))
        z = Xall @ w_true
        yall = np.where(z + rng.normal(0, 1.0, len(z)) > 0, 1, -1)
        Xtr, ytr, Xte, yte = Xall[:N], yall[:N], Xall[N:], yall[N:]
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        scores = []
        for m in (LogisticRegression(max_iter=4000),
                  LinearSVC(C=0.1, max_iter=20000),
                  SVC(kernel="rbf", C=1.0, gamma="scale"),
                  HistGradientBoostingClassifier(max_iter=200,
                                                 random_state=0)):
            m.fit(A, ytr)
            scores.append((m.predict(B) == yte).mean())
        print(f"{N:>6} {D:>6} {scores[0]:>10.4f} {scores[1]:>12.4f} "
              f"{scores[2]:>9.4f} {scores[3]:>10.4f}")

    print("\nAt N=40 with D=500 the linear SVM is clearly ahead: with twelve")
    print("times more features than rows, a maximum-margin solution is a")
    print("better-posed thing to ask for than a fitted probability. Boosting")
    print("needs rows to split on and has too few. The ordering reverses as")
    print("N grows — the same sample-size axis as Chapters 35 and 38.")

    # --- 2. the cost that ended their dominance ------------------------------
    print("\n" + "=" * 72)
    print("2. why they lost: training cost is superlinear (section 5.5)")
    print("=" * 72)
    print(f"{'N':>7} {'RBF SVM fit s':>15} {'boosting fit s':>16} "
          f"{'ratio':>8} {'kernel matrix':>15}")
    for N in (500, 1000, 2000, 4000, 8000):
        X = rng.normal(size=(N, 20))
        yv = np.where(np.sin(X[:, 0] * 2) + X[:, 1] - X[:, 2] ** 2
                      + rng.normal(0, 0.4, N) > 0, 1, -1)
        t0 = time.perf_counter()
        SVC(kernel="rbf", gamma="scale").fit(X, yv)
        t_svm = time.perf_counter() - t0
        t0 = time.perf_counter()
        HistGradientBoostingClassifier(max_iter=200,
                                       random_state=0).fit(X, yv)
        t_gb = time.perf_counter() - t0
        mem = N * N * 8 / 1e6
        print(f"{N:>7} {t_svm:>15.3f} {t_gb:>16.3f} {t_svm / t_gb:>8.2f} "
              f"{mem:>12.1f} MB")

    print("\nSVM training time grows faster than linearly while boosting's is")
    print("nearly flat over this range. Extrapolate the kernel-matrix column:")
    print("at N = 1,000,000 it is 8 terabytes. That is the whole story of why")
    print("kernel methods stopped being the default, and it is about scaling")
    print("rather than about the idea being wrong.")

    # --- 3. no probabilities, and what it costs to add them ------------------
    print("\n" + "=" * 72)
    print("3. an SVM has no probabilities (section 5.5)")
    print("=" * 72)
    N = 3000
    X = rng.normal(size=(N + 4000, 12))
    z = 1.3 * np.sin(X[:, 0]) + X[:, 1] - 0.8 * X[:, 2] ** 2 + 0.5 * X[:, 3]
    y01 = (rng.random(len(z)) < 1 / (1 + np.exp(-z))).astype(int)
    Xtr, Xte = X[:N], X[N:]
    ytr01, yte01 = y01[:N], y01[N:]
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(A, ytr01)
    raw = svm.decision_function(B)
    print(f"decision_function range: [{raw.min():.3f}, {raw.max():.3f}]")
    print("These are signed distances to the boundary, not probabilities:")
    print("they are unbounded and have no frequency interpretation at all.\n")

    cal = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale"), method="sigmoid",
        cv=3).fit(A, ytr01)
    p_cal = cal.predict_proba(B)[:, 1]
    logit = LogisticRegression(max_iter=4000).fit(A, ytr01)
    p_log = logit.predict_proba(B)[:, 1]

    def ece(y, p, nb=10):
        e = np.quantile(p, np.linspace(0, 1, nb + 1))
        t = 0.0
        for i in range(nb):
            m = (p >= e[i]) & (p <= e[i + 1])
            if m.sum():
                t += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
        return t

    ys = np.where(yte01 == 1, 1, -1)
    print(f"{'model':<34} {'AUC':>8} {'ECE':>8} {'accuracy':>10}")
    print(f"{'SVM raw decision function':<34} {roc_auc(ys, raw):>8.4f} "
          f"{'n/a':>8} {((raw > 0) == (yte01 == 1)).mean():>10.4f}")
    print(f"{'SVM + Platt scaling (3-fold)':<34} {roc_auc(ys, p_cal):>8.4f} "
          f"{ece(yte01, p_cal):>8.4f} "
          f"{((p_cal >= 0.5) == (yte01 == 1)).mean():>10.4f}")
    print(f"{'logistic regression':<34} {roc_auc(ys, p_log):>8.4f} "
          f"{ece(yte01, p_log):>8.4f} "
          f"{((p_log >= 0.5) == (yte01 == 1)).mean():>10.4f}")

    print("\nPlatt scaling gives the SVM usable probabilities, and it is not")
    print("free: it needs an internal 3-fold cross-validation, so the model")
    print("is fitted four times. Logistic regression produced calibrated")
    print("probabilities as a property of its loss function, at no extra")
    print("cost — the difference traced in table 39.1 back to the exact zero")
    print("in the hinge gradient.")

    # --- 4. a decision rule --------------------------------------------------
    print("\n" + "=" * 72)
    print("4. when to reach for an SVM in 2026")
    print("=" * 72)
    rules = [
        ("N < ~10,000 and the boundary is curved", "RBF SVM is competitive"),
        ("D > N (genomics, spectra, small text)", "linear SVM, strong"),
        ("large sparse text, N up to millions", "LinearSVC, still excellent"),
        ("tabular, N > ~50,000", "gradient boosting (Chapter 38)"),
        ("you need calibrated probabilities", "logistic or boosting"),
        ("images, audio, language", "learned representations, Part VI+"),
    ]
    for cond, verdict in rules:
        print(f"  {cond:<42} -> {verdict}")
```

## 9. Common Mistakes

**Not standardising before an RBF kernel.** $\|\vec{x}-\vec{z}\|^{2}$ is
dominated by the largest-scale feature, and the failure is silent.

**Tuning $C$ and $\gamma$ separately.** The measured grid shows the best
$\gamma$ depends on $C$.

**Reading `decision_function` as a probability.** It is a signed distance and
unbounded.

**Using an RBF SVM on a million rows.** The kernel matrix alone is eight
terabytes.

**Using `SVC` for large-scale text.** Use `LinearSVC`, which does not form the
kernel matrix.

**Expecting sparsity to mean fast prediction.** Prediction is $O(n_{SV} \cdot D)$
and the measurement shows small $C$ producing many support vectors.

**Assuming a large $\gamma$ that fits the training data is good.** The measured
$\gamma = 200$ row is a lookup table.

**Forgetting that $C$ is inverted.** Larger $C$ is *less* regularisation.

**Using an SVM because it sounds sophisticated.** On tabular data at any real
scale, gradient boosting is faster and better.

## 10. Connection to Previous Chapters

{{ch:math-vectors}} supplied the hyperplane geometry that
{{eq:margin-width}} is three lines of. {{ch:math-optimization}} supplied
Lagrangian duality. {{ch:ml-logistic}} supplied the loss this chapter contrasts
against, and the calibration property the SVM lacks for an identifiable reason —
the exact zero in {{eq:hinge-gradient}}. {{ch:ml-linear-regression}} supplied the
$\ell_2$ penalty, which {{eq:margin-width}} reveals is the *same object* as the
margin rather than an added term. {{ch:ml-knn-nb}} supplied the scaling
requirement and the similarity-weighted prediction that
{{eq:kernel-prediction}} generalises.

Forward: {{ch:tf-scaled-dot-product}} computes a similarity between every query
and every key and takes a weighted combination of values — structurally
{{eq:kernel-prediction}} with a learned similarity, and $O(N^{2})$ for the same
reason. {{ch:emb-similarity}} returns to what a similarity function can and
cannot express. {{part:6}} is the answer to this chapter's last weakness: rather
than choosing a fixed kernel, learn the representation.

## 11. Exercises

**Beginner**

1. What is the margin, and why maximise it?
2. What is a support vector?
3. What does $C$ control, and in which direction?
4. Why must features be standardised for an RBF kernel?
5. Why does an SVM not output probabilities?

**Intermediate**

6. Derive {{eq:margin-width}}.
7. Show that {{eq:soft-margin}} is equivalent to {{eq:hinge-objective}}.
8. Explain the kernel trick in three sentences, without the word "magic".
9. What does $\gamma$ do, and what happens as $\gamma \to \infty$?
10. Why does small $C$ produce *more* support vectors?
11. Why is a linear SVM still competitive on text?

**Advanced**

12. Derive {{eq:svm-dual}} from {{eq:hard-margin}} via the Lagrangian.
13. Derive {{eq:rbf-expansion}} and write out the first four components of
    {{eq:rbf-feature-map}}.
14. State and prove the representer theorem for {{eq:hinge-objective}}.
15. Explain, using complementary slackness, exactly which points have
    $\alpha_i = C$ and what that means.
16. Compare the hinge and log-loss gradients and explain how each of the SVM's
    two distinguishing properties follows from the difference.

**Implementation**

17. Extend the SMO implementation with a proper working-set heuristic and
    measure the reduction in iterations.
18. Implement $\epsilon$-insensitive loss for support vector regression and
    explain the resulting sparsity.
19. Implement Nyström approximation of the kernel matrix and measure the
    accuracy/time trade-off at $N = 20{,}000$.
20. Implement one-vs-rest and one-vs-one multiclass SVMs and compare their
    cost and accuracy.

**Reasoning**

21. You have 800 rows, 4,000 features, and need a classifier. What do you try
    first, and why?
22. Attention and kernel methods both compute pairwise similarities at
    $O(N^{2})$ cost. What did attention gain that made the cost worth paying?

## 12. Chapter Summary

An SVM finds the hyperplane maximising the distance to the nearest points of
each class. The margin is $2/\|\vec{w}\|$, so maximising it means minimising
$\|\vec{w}\|$ — which means the SVM's $\ell_2$ penalty is not a regulariser added
to an objective, it *is* the objective, seen from a different direction.

Equivalently, an SVM is $\ell_2$-regularised empirical risk minimisation with the
hinge loss, which places it in the same framework as everything else in Part IV
and differing only in the loss.

Hinge loss is exactly zero beyond the margin. That single fact produces both
distinguishing behaviours: sparsity, because confidently correct points leave
the problem entirely and only the support vectors remain; and the absence of
calibration, because the model has no reason to distinguish a score of 3 from a
score of 10.

The dual involves the data only through pairwise inner products, which is the
opening the kernel trick uses. The RBF kernel corresponds to an infinite feature
map containing every polynomial power under a Gaussian envelope — verified
numerically to twelve decimal places by truncating it at thirty terms — and
costs one exponential to evaluate. The representer theorem guarantees nothing is
lost: the optimal $\vec{w}$ always lies in the span of the training data.

$C$ and $\gamma$ both control effective complexity and interact, so they must be
searched jointly on a logarithmic plane; the measured grid shows the best
$\gamma$ changing with $C$.

SVMs lost their dominance to superlinear training cost, an $O(N^{2})$ kernel
matrix, the absence of probabilities, and — most importantly — to learned
representations beating fixed similarities on images and text. They remain the
right choice below roughly $10^{5}$ rows with a curved boundary, on sparse text
with a linear kernel, and especially when $D > N$, where the measured comparison
shows the linear SVM ahead of everything else because the margin argument never
mentions the dimension.
