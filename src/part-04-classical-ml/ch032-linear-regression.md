---
id: ml-linear-regression
number: 32
part: IV
tier: focused
status: reviewed
requires: [ml-what-it-is, math-vectors, math-matrices, math-eigen, math-optimization]
provides: [least-squares, normal-equations, ridge-regression, lasso, multicollinearity,
           gauss-markov, heteroscedasticity]
citations: [tibshirani1996, hoerl1970, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive the least-squares solution as an orthogonal projection.
2. Explain why the normal equations are the wrong way to compute it, in terms of
   the condition number.
3. State the Gauss–Markov assumptions and what each one buys.
4. Diagnose multicollinearity and explain what it does and does not break.
5. Derive ridge regression and explain its bias-variance trade in closed form.
6. Explain why the lasso produces exact zeros and ridge does not.
7. Interpret a coefficient correctly, including what standardisation changes.
8. Recognise heteroscedasticity and non-linearity from residual plots.

## 2. Why This Matters

Linear regression is the model whose every property can be derived rather than
asserted, and that makes it the reference against which everything else is
understood.

Three reasons it earns a full chapter rather than a paragraph.

**It is the only model where the answer is a formula.** Fitting is one matrix
operation, not an optimisation loop, and that means every question — how the
coefficients depend on the data, what happens as features become correlated,
what regularisation does — can be answered exactly. Every intuition you build
here transfers to models where the same questions can only be answered
empirically.

**It is a neural network with no hidden layer.** {{part:6}} builds networks by
stacking these and inserting nonlinearities. Backpropagation through a
single-layer network reduces exactly to the gradient derived in
{{sec:6-mathematical-foundation}}. If the linear case is not clear, the deep
case will be a black box.

**It is still the right answer surprisingly often.** For a small dataset, a
mostly-linear relationship and a requirement to explain the model to someone, a
regularised linear model beats gradient boosting on every criterion including
accuracy. Reaching for a complex model first is the most common avoidable error
in applied work.

## 3. Prerequisites

{{ch:math-vectors}} for the projection argument, which *is* the derivation.
{{ch:math-matrices}} for matrix calculus. {{ch:math-eigen}} for the condition
number and the SVD, both of which explain the numerical behaviour.
{{ch:math-optimization}} for gradient descent as the alternative to the closed
form. {{ch:ds-feature-eng}} for scaling, which regularisation makes mandatory.

## 4. Intuitive Explanation

### 4.1 The best line as a shadow

You have $N$ observations and want to predict $y$ from features $\vec{x}$ with a
weighted sum. Write the target as a single vector $\vec{y} \in \R^{N}$ — one
coordinate per observation — and each feature as another vector in the same
space. The set of predictions reachable by *any* choice of weights is the span
of those feature vectors: a plane (technically a subspace) sitting inside
$\R^{N}$.

$\vec{y}$ almost certainly does not lie in that plane. The best you can do is
find the point in the plane closest to it.

```text
              y  ●
                 │╲
                 │ ╲  residual  e = y − ŷ
                 │  ╲                    (perpendicular to the plane)
     ────────────┴───●─────────────────
                    ŷ = Xβ̂
        the column space of X — everything the model can express
```

That closest point is the **orthogonal projection**, and "orthogonal" is not
decoration: the residual is perpendicular to every feature. That single fact
generates the normal equations, the zero-mean residual, and the guarantee from
{{ch:py-visualization}} that residuals are uncorrelated with fitted values.

### 4.2 What the coefficients mean

$\beta_j$ is the change in the prediction for a one-unit increase in $x_j$
**holding all other features fixed**. That last clause is where nearly every
misinterpretation lives.

It means the coefficient is not the relationship between $x_j$ and $y$; it is
the relationship *after removing what the other features already explain*. Add a
correlated feature and the coefficient changes, sometimes flipping sign. Two
consequences follow, and both are chapter-length topics in their own right:
coefficients are not comparable across models with different feature sets
({{sec:5-formal-explanation}}), and "holding all else fixed" is a statement
about the model, not about any intervention you could actually perform
({{ch:ds-causation}}).

### 4.3 Why regularisation

Adding features always reduces training error — a new feature enlarges the
column space, so the projection can only get closer. It frequently increases
test error, because some of what it fits is noise.

Regularisation adds a penalty on the size of the coefficients. The fit is now a
compromise between explaining the data and staying small, and the compromise
turns out to be exactly the bias-variance trade of {{ch:ml-metrics}}, visible
here in closed form rather than empirically.

Two penalties, two different behaviours:

- **Ridge** penalises $\sum \beta_j^{2}$. Shrinks everything, zeroes nothing,
  and handles correlated features gracefully by splitting the credit between
  them.
- **Lasso** penalises $\sum |\beta_j|$. Drives coefficients to exactly zero,
  which selects features, and picks arbitrarily among correlated ones.

The geometric reason for the difference is in {{sec:6-mathematical-foundation}}
and is worth understanding rather than memorising.

## 5. Formal Explanation

### 5.1 The model and the estimator

With $\mat{X} \in \R^{N \times (D+1)}$ (a column of ones for the intercept),

$$
\vec{y} = \mat{X}\vecgreek{\beta} + \vecgreek{\epsilon}, \qquad
\E[\vecgreek{\epsilon}] = \vec{0}, \quad
\Cov[\vecgreek{\epsilon}] = \sigma^{2}\mat{I}
$$ (eq:linear-model)

Least squares minimises

$$
\Loss(\vecgreek{\beta}) = \|\vec{y} - \mat{X}\vecgreek{\beta}\|_2^{2}
$$ (eq:ols-objective)

Setting the gradient to zero gives the **normal equations**:

$$
\mat{X}\T\mat{X}\,\hat{\vecgreek{\beta}} = \mat{X}\T\vec{y}
\quad\Longrightarrow\quad
\hat{\vecgreek{\beta}} = (\mat{X}\T\mat{X})^{-1}\mat{X}\T\vec{y}
$$ (eq:normal-equations)

when $\mat{X}\T\mat{X}$ is invertible, which requires the columns of $\mat{X}$ to
be linearly independent — no feature exactly a combination of the others, and
$N \ge D+1$.

### 5.2 The Gauss–Markov assumptions

Each assumption buys something specific, and knowing which is which tells you
what a violation actually costs.

{#tbl:gauss-markov caption="The Gauss–Markov assumptions, what each one buys, and what a violation costs. Note that only the first threatens the coefficients themselves."}

| Assumption | Buys | Violation costs |
|---|---|---|
| Linearity in parameters | unbiasedness | biased coefficients — the serious one |
| Exogeneity $\E[\epsilon \mid \mat{X}] = 0$ | unbiasedness | biased coefficients |
| Homoscedasticity | minimum variance | wrong standard errors, valid point estimates |
| No autocorrelation | minimum variance | wrong standard errors |
| No perfect collinearity | uniqueness | no unique solution at all |

**Gauss–Markov theorem.** Under {{eq:linear-model}}, $\hat{\vecgreek{\beta}}$ is
the Best Linear Unbiased Estimator — minimum variance among all *unbiased linear*
estimators.

> IMPORTANT: The two qualifications are the whole point. Ridge regression is a
> *biased* estimator and routinely has lower total error than OLS, which is not
> a contradiction — it simply leaves the class the theorem quantifies over.
> "BLUE" is a much weaker guarantee than it sounds, and treating it as a reason
> not to regularise is a common error.

Normality of errors is **not** required for Gauss–Markov. It is required only
for exact $t$ and $F$ inference in small samples; in large samples the CLT of
{{ch:math-inference}} does that job.

### 5.3 Multicollinearity

When features are highly correlated, $\mat{X}\T\mat{X}$ is near-singular. Using
the SVD $\mat{X} = \mat{U}\mat{\Sigma}\mat{V}\T$ from {{ch:math-eigen}}, the
coefficient covariance is

$$
\Cov[\hat{\vecgreek{\beta}}] = \sigma^{2}(\mat{X}\T\mat{X})^{-1}
  = \sigma^{2}\mat{V}\mat{\Sigma}^{-2}\mat{V}\T
$$ (eq:coef-covariance)

A small singular value $\sigma_k$ contributes $1/\sigma_k^{2}$ — so a direction
in feature space with little variation produces an enormous coefficient
variance. This is the same condition-number story as {{ch:math-eigen}},
appearing now as a statistical rather than numerical pathology.

The standard diagnostic is the **variance inflation factor**

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^{2}}
$$ (eq:vif)

where $R_j^{2}$ is from regressing $x_j$ on all other features. VIF above 5 to
10 conventionally signals a problem.

> NOTE: Multicollinearity inflates coefficient variance and does **not** hurt
> predictive accuracy. If you only need predictions, correlated features are
> harmless. If you need to interpret a coefficient, they are fatal. This is
> Breiman's two cultures ({{ch:ml-what-it-is}}) producing genuinely different
> advice from the same diagnostic.

### 5.4 Ridge and lasso

Ridge {{cite:hoerl1970}} adds an $\ell_2$ penalty:

$$
\hat{\vecgreek{\beta}}_{\text{ridge}}
  = \argmin_{\vecgreek{\beta}} \|\vec{y} - \mat{X}\vecgreek{\beta}\|_2^{2}
  + \lambda\|\vecgreek{\beta}\|_2^{2}
  = (\mat{X}\T\mat{X} + \lambda\mat{I})^{-1}\mat{X}\T\vec{y}
$$ (eq:ridge)

The $\lambda\mat{I}$ makes the matrix invertible even when $\mat{X}\T\mat{X}$ is
singular — ridge is defined when OLS is not, including when $D > N$.

Lasso {{cite:tibshirani1996}} adds an $\ell_1$ penalty:

$$
\hat{\vecgreek{\beta}}_{\text{lasso}}
  = \argmin_{\vecgreek{\beta}} \|\vec{y} - \mat{X}\vecgreek{\beta}\|_2^{2}
  + \lambda\|\vecgreek{\beta}\|_1
$$ (eq:lasso)

which has no closed form — the absolute value is not differentiable at zero —
and is solved by coordinate descent or LARS.

> WARNING: Both penalties are scale-dependent: a feature measured in metres gets
> a coefficient a thousand times larger than the same feature in kilometres, and
> so is penalised a million times more heavily by ridge. **Always standardise
> before regularising.** The intercept is left unpenalised, since shrinking it
> towards zero would mean assuming the mean response is near zero.

### 5.5 Linear in parameters, not in inputs

The Gauss–Markov table says "linearity in parameters", and the qualifier is the
most useful thing in it. Nothing in {{eq:normal-equations}} requires the *inputs*
to enter linearly — only that the prediction is a linear combination of
coefficients. So

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^{2} + \beta_3 \log x
        + \beta_4 x_1 x_2
$$ (eq:basis-expansion)

is still least squares. Replace the design matrix's columns with any fixed
functions of the inputs — powers, logs, splines, interactions, indicator
variables for categories — and every result in this chapter continues to hold
unchanged, because the model is linear in the only thing being estimated.

This is a much larger hypothesis space than "straight lines", and it is the same
lever {{ch:ml-what-it-is}} used to solve XOR by adding one product term. Three
consequences worth holding on to:

**The functional-form problem is a feature-engineering problem.** The residual
diagnostics in {{sec:8-practical-example}} detect a wrong form; adding the right
basis functions is how you fix it, and {{ch:ds-feature-eng}} is the catalogue.

**Interactions must be added explicitly.** A linear model cannot discover that
two features matter only together. Trees ({{ch:ml-trees}}) find interactions
automatically, which is a large part of why they dominate on tabular data — the
linear model can represent any interaction you name and none you do not.

**Basis expansion is where overfitting arrives.** A degree-15 polynomial on 20
points fits perfectly and predicts nothing. Regularisation and basis expansion
are therefore used together, and the combination — a rich basis with a penalty
choosing how much of it to use — is the shape of most well-behaved statistical
models, including the splines in generalised additive models.

### 5.6 Choosing $\lambda$, and the elastic net

$\lambda$ is a hyperparameter: it cannot be estimated from the training loss,
because the training loss is monotonically minimised at $\lambda = 0$. It is
chosen by cross-validation, exactly as {{ch:ds-leakage}} prescribed — fit on
$k-1$ folds, score on the held-out fold, repeat over a grid, and take the
$\lambda$ with the best mean validation score.

Two practical points that libraries assume you know:

**Use a logarithmic grid.** The interesting range of $\lambda$ spans orders of
magnitude and the behaviour is smooth in $\log\lambda$; a linear grid wastes
almost all its points.

**Standardise inside the fold, never before splitting.** The mean and standard
deviation used for scaling are estimated quantities, and computing them on the
whole dataset leaks the validation fold into the training fit. This is the
preprocessing-leakage mechanism from {{ch:ds-leakage}}, and it is easy to commit
here because the scaling feels like a data property rather than a fitted one.

The **elastic net** combines both penalties:

$$
\Loss + \lambda\big[\alpha\|\vecgreek{\beta}\|_1
      + (1-\alpha)\|\vecgreek{\beta}\|_2^{2}\big]
$$ (eq:elastic-net)

with $\alpha$ interpolating between ridge ($\alpha = 0$) and lasso
($\alpha = 1$). Its purpose is precisely the failure mode measured at the end of
{{sec:7-implementation}}: on correlated features the lasso picks one arbitrarily,
while the elastic net keeps the group together, because the $\ell_2$ component
makes the objective strictly convex and removes the tie the lasso was breaking
at random. When features come in correlated blocks — repeated measurements, one-
hot categories, rolling windows of the same series — this is usually what you
want.

## 6. Mathematical Foundation

### 6.1 Least squares as projection

Rather than differentiating, use geometry. We want $\hat{\vec{y}} \in
\mathrm{col}(\mat{X})$ minimising $\|\vec{y} - \hat{\vec{y}}\|$. The minimiser is
the orthogonal projection, characterised by the residual being orthogonal to the
subspace:

$$
\mat{X}\T(\vec{y} - \mat{X}\hat{\vecgreek{\beta}}) = \vec{0}
$$ (eq:orthogonality)

which rearranges directly to {{eq:normal-equations}}. No calculus needed.

The **hat matrix** projects onto the column space:

$$
\mat{H} = \mat{X}(\mat{X}\T\mat{X})^{-1}\mat{X}\T, \qquad
\hat{\vec{y}} = \mat{H}\vec{y}
$$ (eq:hat-matrix)

$\mat{H}$ is symmetric and idempotent ($\mat{H}^{2} = \mat{H}$) — projecting
twice is projecting once — with eigenvalues 0 and 1 and $\tr(\mat{H}) = D+1$,
the number of parameters. Its diagonal entries $h_{ii}$ are the **leverages**:
how much observation $i$ influences its own fitted value. High leverage is how
a single point can drag the entire fit, which is why {{ch:py-visualization}} insisted on
plotting the data.

Two corollaries fall straight out of {{eq:orthogonality}}, both used in
{{ch:py-visualization}}: with an intercept column, residuals sum to exactly zero, and
residuals are exactly uncorrelated with fitted values. Neither is evidence the
model is correct — they are algebraic identities that hold for any data
whatsoever.

### 6.2 Why not to use the normal equations

Forming $\mat{X}\T\mat{X}$ **squares the condition number**:

$$
\kappa(\mat{X}\T\mat{X}) = \kappa(\mat{X})^{2}
$$ (eq:condition-squaring)

If $\mat{X}$ has $\kappa = 10^{8}$ — unremarkable for real feature matrices with
different units — then $\kappa(\mat{X}\T\mat{X}) = 10^{16}$, which exhausts
double precision entirely. The result is not merely inaccurate; it can be
arbitrary.

The fix is to never form the product. QR factorisation $\mat{X} = \mat{Q}\mat{R}$
gives $\mat{R}\hat{\vecgreek{\beta}} = \mat{Q}\T\vec{y}$, solved by back
substitution at $\kappa(\mat{X})$. SVD is more robust still and handles rank
deficiency by returning the minimum-norm solution. `np.linalg.lstsq` uses SVD;
scikit-learn's `LinearRegression` calls it {{cite:pedregosa2011}}.

> PRODUCTION TIP: Never write `inv(X.T @ X) @ X.T @ y`. It appears in a great
> deal of teaching code and is wrong for every dataset where it matters.
> `np.linalg.lstsq(X, y, rcond=None)` is shorter, faster and numerically sound.

### 6.3 Ridge in the SVD basis

Substituting $\mat{X} = \mat{U}\mat{\Sigma}\mat{V}\T$ into {{eq:ridge}}:

$$
\hat{\vec{y}}_{\text{ridge}}
  = \sum_{k=1}^{r} \vec{u}_k \frac{\sigma_k^{2}}{\sigma_k^{2} + \lambda}
    \,\vec{u}_k\T\vec{y}
$$ (eq:ridge-svd)

Compare OLS, which is the same sum with every shrinkage factor equal to 1. Ridge
multiplies the $k$-th principal direction by $\sigma_k^{2}/(\sigma_k^{2} +
\lambda)$: directions with large variance ($\sigma_k^{2} \gg \lambda$) pass
almost untouched, and directions with little variance are shrunk hard.

This is the cleanest available statement of what regularisation *does*. It is
not a blanket dampening — it suppresses precisely the directions where the data
carries least information and where {{eq:coef-covariance}} says the variance is
worst. The **effective degrees of freedom**

$$
\mathrm{df}(\lambda) = \sum_{k} \frac{\sigma_k^{2}}{\sigma_k^{2} + \lambda}
$$ (eq:effective-df)

falls smoothly from $r$ at $\lambda = 0$ towards 0, making "model complexity" a
continuous quantity rather than a count of parameters.

### 6.4 Why lasso zeroes and ridge does not

Both penalties have a constrained form: minimise the residual sum of squares
subject to $\|\vecgreek{\beta}\|_2^{2} \le t$ (ridge) or $\|\vecgreek{\beta}\|_1
\le t$ (lasso). The solution is where the elliptical contours of the residual
sum of squares first touch the constraint region.

```text
      ridge: a disc                    lasso: a diamond
        β₂                                  β₂
        │    ╱‾‾╲  contours                 │    ╱‾‾╲
        │  ╱  ╭─╮ ╲                         │  ╱  ╱╲  ╲
        │ │  ╱   ╲ │                        │ │  ╱  ╲ │
     ───┼─│─●─────┼┼──── β₁              ───┼─●───────┼──── β₁
        │ │ ╲   ╱ │                         │  ╲  ╱   ← touches at a CORNER,
        │  ╲ ╰─╯ ╱                          │   ╲╱       where β₂ = 0 exactly
        │    ╲__╱                           │
      touch is generically                corners are where the
      away from the axes                  probability mass is
```

The $\ell_1$ ball has corners **on the axes**, and a corner is exactly a point
where some coefficients are zero. Because a corner is a single point that many
contour orientations touch, the probability of touching at one is positive — so
zeros occur generically, not by accident. The $\ell_2$ ball is smooth: the
tangency is at a generic point, where no coordinate is exactly zero.

For the orthonormal case ($\mat{X}\T\mat{X} = \mat{I}$) both have closed forms
that make the difference stark:

$$
\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1+\lambda},
\qquad
\hat{\beta}_j^{\text{lasso}} = \sign(\hat{\beta}_j^{\text{OLS}})
   \left(|\hat{\beta}_j^{\text{OLS}}| - \tfrac{\lambda}{2}\right)_{+}
$$ (eq:shrinkage-closed-form)

Ridge scales proportionally — a large coefficient loses more in absolute terms
and never reaches zero. Lasso subtracts a constant and clips at zero — the
**soft-thresholding** operator, which small coefficients cannot survive.

## 7. Implementation

```python {tier=A name=ols-from-scratch}
"""Least squares from the projection argument, and why not to invert.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 6.1: fit by projection -----------------------------------------
n, d = 200, 4
X_raw = rng.normal(size=(n, d))
beta_true = np.array([3.0, -1.5, 0.0, 2.0])
y = 5.0 + X_raw @ beta_true + rng.normal(0, 1.0, n)
X = np.column_stack([np.ones(n), X_raw])           # intercept column


def fit_ols_lstsq(X, y):
    """SVD-based least squares — the way you should actually do it."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_ols_qr(X, y):
    """QR factorisation — the projection made explicit."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)


def fit_ols_normal(X, y):
    """The normal equations. Correct in exact arithmetic, fragile in floats."""
    return np.linalg.inv(X.T @ X) @ X.T @ y


beta_hat = fit_ols_lstsq(X, y)
print("true       :", np.round(np.r_[5.0, beta_true], 4))
print("lstsq      :", np.round(beta_hat, 4))
print("QR         :", np.round(fit_ols_qr(X, y), 4))
print("normal eqns:", np.round(fit_ols_normal(X, y), 4))

# --- the orthogonality property (eq. 32.7) ----------------------------------
resid = y - X @ beta_hat
print(f"\nX^T e (should be all zero): {np.abs(X.T @ resid).max():.2e}")
print(f"sum of residuals          : {resid.sum():.2e}")
print(f"corr(residual, fitted)    : "
      f"{np.corrcoef(resid, X @ beta_hat)[0, 1]:.2e}")
print("These are algebraic identities, not evidence the model is right.")

# --- the hat matrix (eq. 32.8) ----------------------------------------------
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"\nH idempotent? max|H@H - H| = {np.abs(H @ H - H).max():.2e}")
print(f"trace(H) = {np.trace(H):.4f}   (should equal d+1 = {d + 1})")
lev = np.diag(H)
print(f"leverages: mean {lev.mean():.4f} (= (d+1)/n = {(d + 1) / n:.4f}), "
      f"max {lev.max():.4f}")

# --- section 6.2: what conditioning does to the normal equations ------------
print("\n" + "=" * 72)
print("why NOT to form X^T X  (eq. 32.10)")
print("=" * 72)
print(f"{'kappa(X)':>12} {'kappa(X^T X)':>16} {'lstsq err':>13} "
      f"{'QR err':>13} {'normal-eq err':>15}")

for exponent in (2, 5, 8, 10):
    # build a design matrix with a controlled condition number via its SVD
    m, k = 300, 6
    U, _ = np.linalg.qr(rng.normal(size=(m, k)))
    V, _ = np.linalg.qr(rng.normal(size=(k, k)))
    s = np.logspace(0, -exponent, k)
    Xc = U @ np.diag(s) @ V.T
    b = rng.normal(size=k)
    yc = Xc @ b                                     # noiseless: exact answer known

    kx = np.linalg.cond(Xc)
    kxx = np.linalg.cond(Xc.T @ Xc)

    def err(fn):
        try:
            return np.linalg.norm(fn(Xc, yc) - b) / np.linalg.norm(b)
        except np.linalg.LinAlgError:
            return np.inf

    print(f"{kx:>12.2e} {kxx:>16.2e} {err(fit_ols_lstsq):>13.2e} "
          f"{err(fit_ols_qr):>13.2e} {err(fit_ols_normal):>15.2e}")

print("\nSquaring the condition number costs you half your digits. At")
print("kappa(X) = 1e8 the normal equations have no correct digits left,")
print("while lstsq and QR are still usable.")

# --- section 5.3: multicollinearity inflates variance, not error ------------
print("\n" + "=" * 72)
print("multicollinearity: coefficients explode, predictions do not")
print("=" * 72)


print(f"{'corr(x1,x2)':>12} {'VIF':>9} {'sd(beta1) over 300 fits':>26} "
      f"{'test RMSE':>11}")
for rho in (0.0, 0.9, 0.99, 0.999):
    coefs, rmses = [], []
    for trial in range(300):
        z1 = rng.normal(size=400)
        z2 = rho * z1 + np.sqrt(max(1e-12, 1 - rho ** 2)) * rng.normal(size=400)
        Xm = np.column_stack([z1, z2])
        ym = 2.0 * z1 + 3.0 * z2 + rng.normal(0, 1.0, 400)
        A = np.column_stack([np.ones(300), Xm[:300]])
        bhat = np.linalg.lstsq(A, ym[:300], rcond=None)[0]
        coefs.append(bhat[1])
        B = np.column_stack([np.ones(100), Xm[300:]])
        rmses.append(np.sqrt(np.mean((B @ bhat - ym[300:]) ** 2)))
    v = 1.0 / (1 - rho ** 2) if rho < 1 else np.inf
    print(f"{rho:>12.3f} {v:>9.1f} {np.std(coefs):>26.4f} "
          f"{np.mean(rmses):>11.4f}")

print("\nThe standard deviation of the estimated coefficient grows without")
print("bound as the features align, while out-of-sample RMSE barely moves.")
print("Collinearity is an interpretation problem, not a prediction problem.")
```

```python {tier=A name=ridge-lasso}
"""Ridge and lasso: shrinkage in the SVD basis, and why one produces zeros.
"""
import numpy as np

rng = np.random.default_rng(1)


def standardise(Xtr, Xte):
    """Regularisation is scale-dependent, so this is mandatory, not optional."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, y, lam):
    """Closed form (eq. 32.13), intercept handled by centring rather than
    by penalising it."""
    ybar = y.mean()
    A = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ (y - ybar))
    return ybar, beta


def lasso_fit(X, y, lam, n_iter=2000, tol=1e-9):
    """Coordinate descent with soft thresholding (eq. 32.17).

    Cycle over coordinates; for each, the univariate solution is the OLS
    residual correlation passed through the soft-threshold operator.
    """
    n, d = X.shape
    ybar = y.mean()
    yc = y - ybar
    beta = np.zeros(d)
    col_ss = (X ** 2).sum(0)
    r = yc - X @ beta
    for _ in range(n_iter):
        delta_max = 0.0
        for j in range(d):
            if col_ss[j] < 1e-12:
                continue
            rho = X[:, j] @ (r + X[:, j] * beta[j])
            new = np.sign(rho) * max(abs(rho) - lam / 2, 0.0) / col_ss[j]
            delta = new - beta[j]
            if delta != 0.0:
                r -= X[:, j] * delta
                beta[j] = new
                delta_max = max(delta_max, abs(delta))
        if delta_max < tol:
            break
    return ybar, beta


# --- section 6.3: ridge as per-direction shrinkage --------------------------
print("=" * 72)
print("ridge shrinks each SVD direction by sigma^2/(sigma^2 + lambda)")
print("=" * 72)

n, d = 300, 8
U, _ = np.linalg.qr(rng.normal(size=(n, d)))
V, _ = np.linalg.qr(rng.normal(size=(d, d)))
svals = np.logspace(0.5, -2.0, d)          # a deliberately ill-conditioned X
X = U @ np.diag(svals) @ V.T
beta_true = rng.normal(size=d)
y = X @ beta_true + rng.normal(0, 0.02, n)

print(f"{'sigma_k':>10} " + " ".join(f"{'l=' + str(l):>10}"
                                     for l in (0.0, 0.001, 0.01, 0.1)))
for k, s in enumerate(svals):
    row = [f"{s ** 2 / (s ** 2 + lam):>10.4f}" if lam > 0 else f"{1.0:>10.4f}"
           for lam in (0.0, 0.001, 0.01, 0.1)]
    print(f"{s:>10.4f} " + " ".join(row))

print("\nHigh-variance directions pass through almost untouched; low-variance")
print("directions — exactly the ones with the worst coefficient variance —")
print("are suppressed. Effective degrees of freedom (eq. 32.15):")
for lam in (0.0, 0.001, 0.01, 0.1, 1.0):
    df = np.sum(svals ** 2 / (svals ** 2 + lam)) if lam > 0 else float(d)
    print(f"  lambda = {lam:<7} df = {df:.3f}  (of {d} parameters)")

# --- section 6.4: soft thresholding vs proportional shrinkage ---------------
print("\n" + "=" * 72)
print("what the two penalties do to a coefficient (eq. 32.17, orthonormal X)")
print("=" * 72)
lam = 1.0
print(f"{'OLS beta':>10} {'ridge':>10} {'lasso':>10}")
for b in (0.05, 0.2, 0.49, 0.51, 1.0, 4.0):
    print(f"{b:>10.2f} {b / (1 + lam):>10.4f} "
          f"{np.sign(b) * max(abs(b) - lam / 2, 0.0):>10.4f}")
print("\nRidge scales; lasso subtracts a constant and clips. Everything below")
print("lambda/2 = 0.50 is set to exactly zero by lasso and merely halved by")
print("ridge. That is feature selection as a side effect of the penalty.")

# --- the penalty is a PRIOR: each wins when its prior is right --------------
def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def bake_off(beta_true, label, n=80, n_test=800):
    d = len(beta_true)
    Xall = rng.normal(size=(n + n_test, d))
    yall = Xall @ beta_true + rng.normal(0, 1.0, n + n_test)
    Xtr, ytr = Xall[:n], yall[:n]
    Xte, yte = Xall[n:], yall[n:]
    Xtr, Xte = standardise(Xtr, Xte)

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    bo = np.linalg.lstsq(np.column_stack([np.ones(n), Xtr]), ytr, rcond=None)[0]
    print(f"{'model':<22} {'test RMSE':>10} {'nonzero':>9} {'true kept':>11}")
    print(f"{'OLS':<22} "
          f"{rmse(np.column_stack([np.ones(len(Xte)), Xte]) @ bo, yte):>10.4f} "
          f"{np.sum(np.abs(bo[1:]) > 1e-6):>9d} {'-':>11}")

    for lam in (5.0, 10.0, 20.0, 40.0):
        ic, br = ridge_fit(Xtr, ytr, lam)
        print(f"{'ridge lambda=' + str(lam):<22} "
              f"{rmse(ic + Xte @ br, yte):>10.4f} "
              f"{np.sum(np.abs(br) > 1e-6):>9d} {'-':>11}")

    n_true = int(np.sum(beta_true != 0))
    for lam in (3.0, 10.0, 40.0):
        ic, bl = lasso_fit(Xtr, ytr, lam)
        hits = int(np.sum((np.abs(bl) > 1e-6) & (beta_true != 0)))
        print(f"{'lasso lambda=' + str(lam):<22} "
              f"{rmse(ic + Xte @ bl, yte):>10.4f} "
              f"{np.sum(np.abs(bl) > 1e-6):>9d} {f'{hits}/{n_true}':>11}")


d = 60
sparse = np.zeros(d)
sparse[rng.choice(d, 5, replace=False)] = rng.normal(0, 3.0, 5)
bake_off(sparse, "SPARSE truth: 5 strong features hidden among 60")

dense = rng.normal(0, 0.35, d)      # every feature contributes a little
bake_off(dense, "DENSE truth: all 60 features contribute weakly")

print("\nA penalty is a prior, and it pays off only when the prior is right.")
print("On the sparse problem the lasso more than halves the error and keeps")
print("all five real features while discarding 52 of the 55 fakes, whereas")
print("NO ridge setting meaningfully beats plain OLS — shrinking five")
print("genuinely large coefficients is nearly pure bias with little variance")
print("to buy back. On the dense problem the ordering reverses: ridge is")
print("best, and the lasso stays competitive only by keeping most of the")
print("features, because every zero it sets is a real effect deleted.")
print("'Which regulariser?' is the question 'what do you believe about the")
print("coefficients?' in disguise.")

# --- correlated features: the case where they differ most -------------------
print("\n" + "=" * 72)
print("correlated features: ridge splits credit, lasso picks one")
print("=" * 72)
z = rng.normal(size=(400, 1))
Xc = np.hstack([z + rng.normal(0, 0.01, (400, 1)) for _ in range(3)])
Xc = np.hstack([Xc, rng.normal(size=(400, 2))])
yc = 6.0 * z[:, 0] + rng.normal(0, 0.5, 400)
Xc_s, _ = standardise(Xc, Xc)

_, br = ridge_fit(Xc_s, yc, 1.0)
_, bl = lasso_fit(Xc_s, yc, 20.0)
print("three near-identical copies of one signal, plus two noise features\n")
print(f"{'feature':<12} {'ridge':>10} {'lasso':>10}")
for j in range(5):
    tag = f"copy {j+1}" if j < 3 else f"noise {j-2}"
    print(f"{tag:<12} {br[j]:>10.4f} {bl[j]:>10.4f}")
print(f"{'sum of 3':<12} {br[:3].sum():>10.4f} {bl[:3].sum():>10.4f}")
print("\nRidge divides the signal roughly equally between the copies; lasso")
print("concentrates it. The totals are similar, so predictions agree — but")
print("the lasso's choice among identical features is arbitrary, and would")
print("change with a slightly different sample. Never read a lasso's")
print("selection among correlated features as a finding.")
```

## 8. Practical Example

```python {tier=A name=regression-workflow}
"""A complete regression workflow: fit, diagnose, regularise, interpret.

Synthetic housing data with a deliberately planted nonlinearity, planted
heteroscedasticity, and a pair of collinear features — the three things
residual plots are supposed to catch.
"""
import numpy as np

rng = np.random.default_rng(7)
n = 900

area = rng.gamma(9.0, 12.0, n)                       # sq metres
bedrooms = np.clip(np.round(area / 45 + rng.normal(0, 0.6, n)), 1, 6)
area_sqft = area * 10.7639                           # collinear by construction
age = rng.uniform(0, 90, n)
dist_centre = rng.gamma(2.0, 3.0, n)                 # km

# the truth: LOG price is linear; price itself is not, and noise scales with it
log_price = (11.0
             + 0.0060 * area
             + 0.030 * bedrooms
             - 0.0035 * age
             - 0.045 * dist_centre
             + rng.normal(0, 0.18, n))
price = np.exp(log_price)

names = ["area", "bedrooms", "area_sqft", "age", "dist_centre"]
X = np.column_stack([area, bedrooms, area_sqft, age, dist_centre])
cut = 700
Xtr, Xte, ytr, yte = X[:cut], X[cut:], price[:cut], price[cut:]


def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def r2(pred, y):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


# --- step 1: fit naively and look at the diagnostics ------------------------
b = fit(Xtr, ytr)
fitted = predict(b, Xtr)
resid = ytr - fitted
print("=" * 72)
print("step 1 — fit on the raw target")
print("=" * 72)
print(f"train R2 {r2(fitted, ytr):.4f}   test R2 {r2(predict(b, Xte), yte):.4f}")

# nonlinearity: are residuals systematically signed across the fitted range?
# heteroscedasticity: does residual spread grow with the fitted value?
# Both are read off the same binning of the residuals by fitted value.
order = np.argsort(fitted)
bins = np.array_split(resid[order], 6)
print("\nresiduals binned by fitted value — the only plot that matters:")
print(f"{'sextile':>8} {'mean resid':>14} {'std resid':>14}")
for i, t in enumerate(bins):
    print(f"{i + 1:>8} {t.mean():>14,.0f} {t.std():>14,.0f}")

signs = "".join("+" if t.mean() > 0 else "-" for t in bins)
print(f"\nsign pattern of the mean residual: {signs}")
print("Runs of one sign, not an alternating jumble. The model under-predicts")
print("at both ends of its own range and over-predicts in the middle — the")
print("signature of fitting a straight line to a convex relationship. A")
print("correctly specified model cannot be systematically wrong in one")
print("region of its own predictions, so this is a specification error.")

spread_ratio = bins[-1].std() / bins[0].std()
print(f"\nresidual spread, highest bin / lowest bin = {spread_ratio:.2f}   "
      f"({'heteroscedastic' if spread_ratio > 2 else 'roughly constant'})")
print("Constant-variance errors were assumed by Gauss-Markov (section 5.2).")
print("Violating it does not bias the coefficients — it invalidates every")
print("standard error and confidence interval computed from this fit.")

# --- step 2: collinearity ---------------------------------------------------
print("\n" + "=" * 72)
print("step 2 — variance inflation factors (eq. 32.12)")
print("=" * 72)


def vif(Xf, names):
    for j in range(Xf.shape[1]):
        others = np.column_stack([np.ones(len(Xf)), np.delete(Xf, j, axis=1)])
        pred = others @ np.linalg.lstsq(others, Xf[:, j], rcond=None)[0]
        ss_res = np.sum((Xf[:, j] - pred) ** 2)
        ss_tot = np.sum((Xf[:, j] - Xf[:, j].mean()) ** 2)
        rsq = 1 - ss_res / ss_tot
        v = np.inf if rsq > 1 - 1e-12 else 1 / (1 - rsq)
        flag = "  <-- collinear" if v > 10 else ""
        print(f"  {names[j]:<14} R2 {rsq:>8.5f}   VIF {v:>12,.1f}{flag}")


vif(Xtr, names)
print("\narea and area_sqft are the same measurement in different units.")
print("Coefficients on both are meaningless; predictions are unaffected.")

keep = [0, 1, 3, 4]
b2 = fit(Xtr[:, keep], ytr)
print(f"\nafter dropping area_sqft: test R2 "
      f"{r2(predict(b2, Xte[:, keep]), yte):.4f} "
      f"(was {r2(predict(b, Xte), yte):.4f}) — essentially unchanged")
print(f"coefficient on area: {b[1]:>12,.1f} with the duplicate present")
print(f"                     {b2[1]:>12,.1f} with it removed")

# --- step 3: fix the functional form ----------------------------------------
print("\n" + "=" * 72)
print("step 3 — model log(price) instead, as the diagnostics suggested")
print("=" * 72)
b3 = fit(Xtr[:, keep], np.log(ytr))
log_fit = predict(b3, Xtr[:, keep])
log_res = np.log(ytr) - log_fit
lo = np.array_split(log_res[np.argsort(log_fit)], 6)
print("mean residual by sextile:", " ".join(f"{t.mean():>8.4f}" for t in lo))
print("std  residual by sextile:", " ".join(f"{t.std():>8.4f}" for t in lo))
print("sign pattern:            ",
      "".join("+" if t.mean() > 0 else "-" for t in lo))
print(f"spread ratio high/low = {lo[-1].std() / lo[0].std():.2f}")

pred_price = np.exp(predict(b3, Xte[:, keep]))
print(f"\ntest R2 on the price scale: {r2(pred_price, yte):.4f} "
      f"(linear model gave {r2(predict(b2, Xte[:, keep]), yte):.4f})")
print("Both diagnostics are now flat, and accuracy improved. The residual")
print("plot told us the functional form was wrong before any tuning.")

# --- step 4: interpret ------------------------------------------------------
print("\n" + "=" * 72)
print("step 4 — interpreting the coefficients")
print("=" * 72)
kept_names = [names[j] for j in keep]
truth = {"area": 0.0060, "bedrooms": 0.030, "age": -0.0035,
         "dist_centre": -0.045}
print(f"{'feature':<14} {'coef':>10} {'true':>10} "
      f"{'interpretation (log model)':<40}")
for j, nm in enumerate(kept_names):
    pct = (np.exp(b3[j + 1]) - 1) * 100
    print(f"{nm:<14} {b3[j + 1]:>10.5f} {truth[nm]:>10.5f} "
          f"{'+1 unit -> ' + format(pct, '+.2f') + '% in price':<40}")

# standardised coefficients answer a DIFFERENT question
sd = Xtr[:, keep].std(0)
print(f"\n{'feature':<14} {'raw coef':>10} {'per-SD effect':>15}")
for j, nm in enumerate(kept_names):
    print(f"{nm:<14} {b3[j + 1]:>10.5f} {b3[j + 1] * sd[j]:>15.5f}")
print("\nRaw coefficients answer 'what does one more unit do'. Standardised")
print("ones answer 'which feature moves the prediction most across its")
print("observed range'. They rank features differently and neither is the")
print("importance of a feature in any causal sense (Chapter 30).")
```

## 9. Common Mistakes

**`inv(X.T @ X) @ X.T @ y`.** Squares the condition number. Use `lstsq`.

**Regularising without standardising.** The penalty is scale-dependent, so the
result depends on your choice of units.

**Penalising the intercept.** Shrinking it towards zero asserts that the mean
response is near zero.

**Reading a coefficient as a causal effect.** "Holding all else fixed" is a
statement about the model's arithmetic, not about an intervention
({{ch:ds-causation}}).

**Comparing raw coefficients across features with different units.** Compare
per-standard-deviation effects, and even then only as a description.

**Treating high $R^2$ as validation.** It rises whenever a feature is added,
never falls, and is computed on the training data.

**Dropping a collinear feature you need.** If you only want predictions,
collinearity is harmless; dropping the feature loses nothing but gains nothing
either.

**Reading lasso's selection among correlated features as a finding.** The choice
is arbitrary and unstable across samples.

**Ignoring residual plots.** Both the functional-form error and the
heteroscedasticity in {{sec:8-practical-example}} are invisible in $R^2$ and
obvious in two lines of diagnostic output.

## 10. Connection to Previous Chapters

{{ch:math-vectors}} supplied the projection that {{eq:orthogonality}} *is*.
{{ch:math-eigen}} supplied the SVD and the condition number, which reappear as
{{eq:ridge-svd}} and {{eq:condition-squaring}} — the same mathematics explaining
both the numerical fragility and the statistical variance.
{{ch:math-optimization}} supplied gradient descent, which is how the lasso is
solved. {{ch:py-visualization}} claimed residuals are uncorrelated with fitted values by
construction; {{eq:orthogonality}} is the proof.

Forward: {{ch:ml-logistic}} keeps this representation and changes only the loss.
{{ch:ml-metrics}} formalises the bias-variance trade that {{eq:ridge-svd}} shows
in closed form. {{ch:dl-neural-networks}} stacks these layers. The $\ell_2$ penalty here is
weight decay in {{ch:dl-regularization}} — the same equation with a different
name.

## 11. Exercises

**Beginner**

1. Derive {{eq:normal-equations}} from {{eq:orthogonality}} in two lines.
2. Why must $\mat{X}$ have full column rank?
3. What does the intercept represent when features are not centred?
4. Give the interpretation of $\beta_j$ including the qualifying clause.
5. Why does adding a feature never increase training error?

**Intermediate**

6. Explain why {{eq:condition-squaring}} makes the normal equations unusable at
   $\kappa(\mat{X}) = 10^{8}$.
7. Compute the VIF for two features with correlation 0.95.
8. Using {{eq:ridge-svd}}, say what happens to a direction with $\sigma_k^{2} =
   0.01$ at $\lambda = 1$.
9. State Gauss–Markov, and explain why ridge does not contradict it.
10. Why is normality of errors not required for Gauss–Markov?
11. From {{eq:shrinkage-closed-form}}, find the OLS coefficient below which
    lasso sets it to zero.

**Advanced**

12. Prove $\mat{H}$ is idempotent and that $\tr(\mat{H})$ equals the number of
    parameters.
13. Derive {{eq:ridge-svd}} from {{eq:ridge}} using the SVD.
14. Show that ridge with $\lambda$ is equivalent to OLS on data augmented with
    $\sqrt{\lambda}\mat{I}$ rows and zero targets. What does this say about how
    ridge should be implemented?
15. Explain effective degrees of freedom {{eq:effective-df}} and why it is not
    an integer.
16. Derive the soft-thresholding operator from the $\ell_1$ subgradient
    condition.

**Implementation**

17. Implement ridge via the augmented-data trick of exercise 14 and confirm it
    matches {{eq:ridge}} numerically.
18. Extend the coordinate-descent lasso to the elastic net and show it keeps
    correlated features together.
19. Implement leverage-based outlier detection and demonstrate a single point
    changing the fit.
20. Reproduce the multicollinearity experiment and plot coefficient standard
    deviation against VIF.

**Reasoning**

21. When is a linear model preferable to gradient boosting even at equal
    accuracy?
22. A colleague reports $R^2 = 0.97$ on a 40-row dataset with 35 features. What
    do you say?

## 12. Chapter Summary

Least squares is an orthogonal projection of the target onto the column space of
the features. The normal equations follow immediately from the residual being
perpendicular to every feature, and so do the algebraic identities — residuals
summing to zero, residuals uncorrelated with fitted values — that are sometimes
mistaken for evidence of a good fit.

Do not compute it by inverting $\mat{X}\T\mat{X}$: forming that product squares
the condition number, and the measured error in
{{sec:7-implementation}} shows the normal equations losing every correct digit
at $\kappa(\mat{X}) = 10^{8}$ where `lstsq` and QR remain sound.

Gauss–Markov makes OLS best among *unbiased linear* estimators. Both
qualifications matter: ridge is biased and often better.

Multicollinearity inflates coefficient variance without harming prediction. It
is fatal for interpretation and irrelevant for forecasting — the same diagnostic
yielding opposite advice depending on which of Breiman's two cultures you are
in.

Ridge shrinks each SVD direction by $\sigma_k^{2}/(\sigma_k^{2}+\lambda)$,
suppressing exactly the low-information directions where coefficient variance is
worst, and makes the problem solvable when $D > N$. Lasso soft-thresholds,
producing exact zeros because the $\ell_1$ ball has corners on the axes. Ridge
splits credit among correlated features; lasso picks one arbitrarily, and that
choice should never be read as a finding.

Both penalties are scale-dependent, so standardise first and leave the intercept
unpenalised.
