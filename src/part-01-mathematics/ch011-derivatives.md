---
id: math-derivatives
number: 11
part: I
tier: focused
status: reviewed
requires: [math-functions, math-vectors]
provides: [derivative, partial-derivative, gradient, directional-derivative,
           chain-rule, jacobian, hessian]
citations: [deisenroth2020, goodfellow2016]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define the derivative as a limit and interpret it as a slope and as a rate of
   change.
2. Differentiate the handful of functions this book actually uses, including the
   logistic function.
3. Compute partial derivatives and assemble them into a gradient.
4. Explain why the gradient points in the direction of steepest ascent, and
   prove it.
5. Apply the chain rule to compositions, including the multivariable form.
6. Construct a computational graph and propagate derivatives backwards through
   it by hand.
7. Build a Jacobian and a Hessian and say what each is for.
8. Explain why reverse-mode differentiation is the right choice for machine
   learning and forward mode is not.

## 2. Why This Matters

Training a neural network means computing a gradient and taking a step against
it, several million times. Everything else — architectures, optimisers,
schedules, initialisation — is machinery around that one loop.

The gradient is what makes learning possible. A model has millions or billions
of parameters, and the question "how should I change each one to reduce the
loss?" would be hopeless if it had to be answered by trial. The gradient answers
it for every parameter simultaneously, at a cost comparable to one forward pass.
That efficiency is not obvious and is not free: it comes from reverse-mode
differentiation, and understanding why it is efficient is the point of
{{sec:6-mathematical-foundation}}.

This chapter is also where backpropagation stops being intimidating. Once you
have seen the chain rule applied to a computational graph by hand,
{{ch:dl-backprop}} is bookkeeping rather than revelation — a systematic way of
doing what you will do manually here.

> IMPORTANT: This and {{ch:math-optimization}} are the two hardest chapters in
> Part I. If you have not seen calculus before, expect to spend real time here.
> The payoff is that {{part:6}} — the whole of deep learning — becomes
> mechanical rather than mysterious.

## 3. Prerequisites

{{ch:math-functions}} for functions, composition, exponentials and logarithms.
{{ch:math-vectors}} for vectors and the dot product — the gradient is a vector
and the directional derivative is a dot product.
{{ch:math-matrices}} for the Jacobian and Hessian.

No prior calculus is assumed.

## 4. Intuitive Explanation

### 4.1 The derivative is a slope

The {{term:derivative}} of a function at a point answers: *if I nudge the input
slightly, how much does the output change, and in which direction?*

Take $f(x) = x^{2}$ at $x = 3$, where $f(3) = 9$. Nudge the input by a small
$h$:

{#tbl:derivative-limit caption="Approaching the derivative of x² at x = 3. The difference quotient converges to 6."}

| $h$ | $f(3+h)$ | $\dfrac{f(3+h) - f(3)}{h}$ |
|---|---|---|
| 1 | 16 | 7 |
| 0.1 | 9.61 | 6.1 |
| 0.01 | 9.0601 | 6.01 |
| 0.001 | 9.006001 | 6.001 |
| $\to 0$ | | $\to 6$ |

The derivative at $x = 3$ is 6: near that point, increasing $x$ by a tiny amount
increases $f$ by about six times as much.

Two readings, both used constantly. **Geometrically**, the derivative is the
slope of the tangent line. **Physically**, it is a rate of change. In machine
learning the second is usually the more useful: $\partial\Loss/\partial w$ is
"how fast the loss changes as this weight changes", which tells you immediately
which way to move it.

### 4.2 Partial derivatives handle many inputs

Real functions have many inputs — a loss depends on every parameter in the
model. A {{term:partial-derivative}} isolates one:

$$
\frac{\partial f}{\partial x_i}
$$

means "differentiate with respect to $x_i$, treating every other input as a
constant". That is the whole rule, and it makes multivariable differentiation no
harder than the single-variable kind: you do the same thing, several times.

For $f(x, y) = x^{2}y + 3y$:

- $\partial f/\partial x = 2xy$ (treat $y$ as a constant; the $3y$ term
  disappears because it does not involve $x$)
- $\partial f/\partial y = x^{2} + 3$ (treat $x$ as a constant)

### 4.3 The gradient assembles them

Collect all the partial derivatives into a vector and you have the
{{term:gradient}}:

$$
\nabla f = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]\T
$$ (eq:gradient-def)

The gradient has the same shape as the input, which is worth internalising: if
your model has 7 billion parameters, its gradient is a 7-billion-dimensional
vector, one number per parameter, each saying how the loss responds to that
parameter.

And the gradient has a property that makes the whole enterprise work: **it points
in the direction of steepest increase.** Moving against it decreases the function
fastest. That is not a definition — it is a theorem, proved in
{{sec:6-mathematical-foundation}}, and it is the entire justification for
gradient descent.

### 4.4 The chain rule is credit assignment

The {{term:chain-rule}} differentiates a composition. If $y = f(u)$ and
$u = g(x)$:

$$
\frac{\dd y}{\dd x} = \frac{\dd y}{\dd u}\cdot\frac{\dd u}{\dd x}
$$ (eq:chain-rule)

Read it as a rate calculation. If $y$ changes 3× as fast as $u$, and $u$ changes
2× as fast as $x$, then $y$ changes 6× as fast as $x$. Rates multiply along a
chain.

This is what makes deep networks trainable. A network is a composition
({{ch:math-functions}}), so the derivative of the loss with respect to an early
weight is a product of derivatives, one per intervening layer. That product is
the credit assignment: it says how much this particular weight, buried thirty
layers back, contributed to the final error.

It also explains the pathologies. Each factor either amplifies or shrinks the
signal. Multiply thirty factors that are each around 0.25 — as saturated
sigmoids are ({{ch:math-functions}}) — and the gradient reaching the first layer
is around $10^{-18}$. That is the vanishing gradient problem, and it is the
chain rule doing exactly what it should.

## 5. Formal Explanation

### 5.1 The derivative

$$
f'(x) = \lim_{h \to 0}\frac{f(x+h) - f(x)}{h}
$$ (eq:derivative-def)

when the limit exists. A function is **differentiable** at $x$ if it does.
Differentiability implies continuity, but not conversely: $\lvert x\rvert$ is
continuous at 0 and not differentiable there, because the limit differs from the
left and the right. That corner is exactly the non-smoothness that makes $L_1$
regularisation produce sparsity ({{ch:math-norms}}).

The rules this book needs:

{#tbl:derivative-rules caption="Differentiation rules used in this book. The last four are the ones that appear in nearly every gradient derivation."}

| Function | Derivative | Note |
|---|---|---|
| $c$ | $0$ | |
| $x^{n}$ | $nx^{n-1}$ | power rule |
| $e^{x}$ | $e^{x}$ | its own derivative |
| $\log x$ | $1/x$ | for $x > 0$ |
| $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ | derived in {{ch:math-functions}} |
| $\max(0, x)$ | $0$ or $1$ | undefined at 0; ReLU |
| $f + g$ | $f' + g'$ | sum rule |
| $fg$ | $f'g + fg'$ | product rule |
| $f/g$ | $(f'g - fg')/g^{2}$ | quotient rule |
| $f(g(x))$ | $f'(g(x))\,g'(x)$ | chain rule |

> NOTE: ReLU is not differentiable at exactly 0, which sounds like a problem and
> is not. Frameworks define the derivative there as 0 or 1 by convention, and
> the input landing on exactly 0.0 has probability zero in floating point.
> Non-differentiability at isolated points is a non-issue in practice; it is
> non-differentiability on a *set the optimiser visits often* that matters.

### 5.2 Partial derivatives and the gradient

$$
\frac{\partial f}{\partial x_i}
  = \lim_{h\to 0}\frac{f(x_1,\ldots,x_i+h,\ldots,x_n) - f(\vec{x})}{h}
$$ (eq:partial-def)

The gradient collects them, as in {{eq:gradient-def}}, and always has the same
shape as $\vec{x}$.

The **{{term:directional-derivative}}** measures the rate of change along an
arbitrary unit direction $\vec{u}$:

$$
D_{\vec{u}}f(\vec{x}) = \nabla f(\vec{x})\T\vec{u}
$$ (eq:directional-derivative)

It is a dot product with the gradient — which is the fact that proves the
steepest-ascent property in {{sec:6-mathematical-foundation}}.

### 5.3 The multivariable chain rule

When a function depends on $x$ through several intermediate variables, the
contributions **add**:

$$
\frac{\dd f}{\dd x} = \sum_{i}\frac{\partial f}{\partial u_i}\cdot\frac{\dd u_i}{\dd x}
$$ (eq:multivariable-chain)

Sum over every path from $x$ to $f$, multiplying along each path. This is the
form backpropagation implements: a node with several consumers accumulates
gradient from all of them, which is why gradients are *accumulated* rather than
assigned in every autograd framework.

### 5.4 Jacobian and Hessian

For a vector-valued $f: \R^{n} \to \R^{m}$, the {{term:jacobian}} stacks the
gradients of each output:

$$
\mat{J}_{ij} = \frac{\partial f_i}{\partial x_j}, \qquad \mat{J} \in \R^{m \times n}
$$ (eq:jacobian)

The chain rule in this setting is matrix multiplication:

$$
\mat{J}_{f \circ g} = \mat{J}_{f}\,\mat{J}_{g}
$$ (eq:jacobian-chain)

{{eq:jacobian-chain}} is the whole of backpropagation stated in one line. A
network is a composition, so the Jacobian of the whole is a product of
per-layer Jacobians. Everything else is about computing that product efficiently
without ever forming the matrices.

For a scalar function, the {{term:hessian}} holds the second derivatives:

$$
\mat{H}_{ij} = \frac{\partial^{2} f}{\partial x_i \partial x_j}
$$ (eq:hessian)

It is symmetric when $f$ is twice continuously differentiable, and it describes
curvature. Its eigenvalues classify stationary points, and its condition number
determines how badly gradient descent zigzags — both taken up in
{{ch:math-optimization}}.

> PRODUCTION TIP: The Hessian of a model with $P$ parameters has $P^{2}$
> entries. For a 7-billion-parameter model that is $4.9 \times 10^{19}$ numbers
> — utterly out of reach. This is why second-order optimisation methods are not
> used directly in deep learning, and why the methods that do exploit curvature
> (Adam, and quasi-Newton methods) approximate it with a diagonal or a low-rank
> summary ({{ch:dl-optimizers}}).

## 6. Mathematical Foundation

### 6.1 Why the gradient is the steepest direction

This is the theorem the entire field rests on, and the proof is three lines.

The rate of change in a unit direction $\vec{u}$ is, by
{{eq:directional-derivative}} and the geometric form of the dot product
({{ch:math-vectors}}):

$$
D_{\vec{u}}f = \nabla f\T\vec{u} = \norm{\nabla f}\,\norm{\vec{u}}\cos\theta
  = \norm{\nabla f}\cos\theta
$$ (eq:steepest-proof)

since $\norm{\vec{u}} = 1$. The only thing that varies with the choice of
direction is $\cos\theta$, which is maximised at $\theta = 0$ — that is,
when $\vec{u}$ points along $\nabla f$.

So:

- **Steepest ascent** is along $+\nabla f$, at rate $\norm{\nabla f}$.
- **Steepest descent** is along $-\nabla f$, at rate $-\norm{\nabla f}$.
- Moving **perpendicular** to the gradient ($\theta = 90°$) changes $f$ not at
  all — you are travelling along a contour line.

That is why gradient descent steps against the gradient. Not by convention or
convenience: it is provably the locally fastest direction of decrease.

### 6.2 Backpropagation on a computational graph, by hand

Doing this once by hand is worth more than reading three descriptions of it.

Consider

$$
f(x, y) = (x + y)\cdot\max(0, x)
$$ (eq:graph-example)

evaluated at $x = 2$, $y = -5$. Break it into elementary operations:

$$
a = x + y, \qquad b = \max(0, x), \qquad f = a \cdot b
$$

```mermaid {#fig:computational-graph caption="A computational graph. The forward pass evaluates left to right; the backward pass propagates ∂f/∂· right to left, multiplying by each local derivative and summing where paths rejoin."}
graph LR
  X["x = 2"] --> A["a = x + y<br/>= −3"]
  Y["y = −5"] --> A
  X --> B["b = max(0, x)<br/>= 2"]
  A --> F["f = a · b<br/>= −6"]
  B --> F
```

**Forward pass.** $a = 2 + (-5) = -3$; $b = \max(0, 2) = 2$;
$f = (-3)(2) = -6$.

**Backward pass.** Start at the output with $\partial f/\partial f = 1$ and work
backwards, multiplying by each local derivative.

At the multiplication node, the local derivatives are
$\partial f/\partial a = b = 2$ and $\partial f/\partial b = a = -3$:

$$
\frac{\partial f}{\partial a} = 2, \qquad \frac{\partial f}{\partial b} = -3
$$

At the addition node, $\partial a/\partial x = 1$ and $\partial a/\partial y = 1$
— addition passes gradient through unchanged:

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a}\cdot\frac{\partial a}{\partial y} = 2 \cdot 1 = 2
$$

At the ReLU node, $\partial b/\partial x = 1$ since $x = 2 > 0$:

$$
\left(\frac{\partial f}{\partial x}\right)_{\text{via } b} = (-3)(1) = -3
$$

Now the crucial step. $x$ feeds *two* paths — into $a$ and into $b$ — so by
{{eq:multivariable-chain}} the contributions **add**:

$$
\frac{\partial f}{\partial x}
  = \underbrace{2 \cdot 1}_{\text{via } a} + \underbrace{(-3) \cdot 1}_{\text{via } b}
  = 2 - 3 = -1
$$ (eq:graph-backward)

So $\nabla f = [-1, 2]\T$. Verify against the analytic form: for $x > 0$,
$f = (x+y)x = x^{2} + xy$, giving $\partial f/\partial x = 2x + y = 4 - 5 = -1$
✓ and $\partial f/\partial y = x = 2$ ✓.

Three patterns generalise from this and are worth memorising, because they are
what an autograd engine actually implements:

- **Addition** distributes the gradient unchanged to both inputs.
- **Multiplication** sends each input the gradient times the *other* input's
  value — which is why the forward values must be kept for the backward pass,
  and hence why training uses far more memory than inference.
- **Branching** (one value used twice) sums the incoming gradients.

### 6.3 Why reverse mode, and not forward

There are two ways to apply the chain rule through a graph, and choosing
correctly is the difference between deep learning working and not.

**Forward mode** computes $\partial(\text{everything})/\partial x_i$ for one
input $x_i$ in a single sweep. To get the full gradient of a function with $n$
inputs, you need $n$ sweeps.

**Reverse mode** computes $\partial f/\partial(\text{everything})$ for one output
$f$ in a single sweep. To get the full gradient of a scalar function, you need
**one** sweep.

Neural network training has exactly the shape reverse mode is good at: billions
of inputs (the parameters), one scalar output (the loss).

{#tbl:autodiff-modes caption="Cost of computing a full gradient, in units of one forward pass. For neural network training, n is in the billions and m is 1."}

| | $f: \R^{n} \to \R^{m}$ | Neural network ($m = 1$) |
|---|---|---|
| Forward mode | $O(n)$ sweeps | $O(n)$ — hopeless |
| Reverse mode | $O(m)$ sweeps | $O(1)$ — one sweep |

For a 7-billion-parameter model, forward mode would need 7 billion forward
passes per gradient. Reverse mode needs roughly two — one forward, one backward.

The price is memory: reverse mode must retain the intermediate values from the
forward pass in order to compute local derivatives on the way back. That is
precisely why training needs several times more memory than inference, and why
gradient checkpointing — recomputing intermediates instead of storing them —
trades compute for memory ({{ch:dl-backprop}}).

### 6.4 A worked gradient: logistic regression

Putting the pieces together on a real loss. For one example with input
$\vec{x}$, label $y \in \{0,1\}$, weights $\vec{w}$:

$$
z = \vec{w}\T\vec{x}, \qquad
\hat{p} = \sigma(z), \qquad
\ell = -\big[y\log\hat{p} + (1-y)\log(1-\hat{p})\big]
$$ (eq:logreg-forward)

Differentiate the loss with respect to $\hat{p}$:

$$
\frac{\partial \ell}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}}
  = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}
$$ (eq:dl-dp)

The sigmoid derivative, from {{ch:math-functions}}:

$$
\frac{\partial \hat{p}}{\partial z} = \hat{p}(1-\hat{p})
$$ (eq:dp-dz)

Chain them, and the denominators cancel exactly:

$$
\frac{\partial \ell}{\partial z}
  = \frac{\hat{p}-y}{\hat{p}(1-\hat{p})}\cdot\hat{p}(1-\hat{p})
  = \hat{p} - y
$$ (eq:dl-dz)

Finally $\partial z/\partial \vec{w} = \vec{x}$, so

$$
\nabla_{\vec{w}}\,\ell = (\hat{p} - y)\,\vec{x}
$$ (eq:logreg-gradient)

The gradient is the prediction error times the input. It is hard to imagine a
simpler result, and the simplicity is not luck — it is the cancellation in
{{eq:dl-dz}}, which happens because the logistic function and cross-entropy are
matched to each other ({{ch:math-functions}}).

The same cancellation occurs for softmax with categorical cross-entropy, giving
$\hat{\vec{p}} - \vec{y}$ ({{ch:ml-logistic}}). Any time you see a gradient of
the form "prediction minus target", this is why.

## 7. Implementation

```python {tier=A name=derivatives-and-gradients}
"""Derivatives, gradients, the chain rule, and backpropagation by hand.

Every analytic derivative is checked against a numerical one.
"""
import numpy as np

# --- eq. 11.1: the derivative as a limit ------------------------------------
def f(x):
    return x ** 2


print(f"{'h':>10} {'difference quotient':>22}")
for h in (1.0, 0.1, 0.01, 0.001, 1e-6):
    print(f"{h:>10} {(f(3 + h) - f(3)) / h:>22.9f}")
print("converging to 6 = 2x at x = 3\n")


def numerical_gradient(fn, x, h=1e-6):
    """Central-difference gradient. More accurate than the forward difference:
    its error is O(h^2) rather than O(h)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = h
        grad.flat[i] = (fn(x + step) - fn(x - step)) / (2 * h)
    return grad


# --- partial derivatives and the gradient (eq. 11.3) ------------------------
def g(v):
    x, y = v
    return x**2 * y + 3 * y


def g_grad(v):
    x, y = v
    return np.array([2 * x * y, x**2 + 3])


point = np.array([2.0, 5.0])
print(f"analytic  grad g at {point}: {g_grad(point)}")
print(f"numerical grad g at {point}: {np.round(numerical_gradient(g, point), 6)}")
assert np.allclose(g_grad(point), numerical_gradient(g, point), atol=1e-5)

# --- section 6.1: the gradient really is the steepest direction -------------
grad = g_grad(point)
unit_grad = grad / np.linalg.norm(grad)
print(f"\nrate of increase in 2000 random unit directions vs the gradient:")
best_rate, best_dir = -np.inf, None
rng = np.random.default_rng(0)
for _ in range(2000):
    u = rng.normal(size=2)
    u /= np.linalg.norm(u)
    rate = grad @ u                              # eq. 11.5
    if rate > best_rate:
        best_rate, best_dir = rate, u
print(f"  best random direction : rate {best_rate:.6f}")
print(f"  the gradient direction: rate {grad @ unit_grad:.6f}  <- larger")
print(f"  ||grad||              : {np.linalg.norm(grad):.6f}  <- eq. 11.10")
assert grad @ unit_grad >= best_rate - 1e-9

# Perpendicular to the gradient, nothing changes — a contour line.
perp = np.array([-unit_grad[1], unit_grad[0]])
print(f"  perpendicular direction: rate {grad @ perp:+.2e}  <- zero")

# --- section 6.2: backpropagation by hand -----------------------------------
print("\n" + "=" * 62)
print("backpropagation on f(x, y) = (x + y) * max(0, x)  at x=2, y=-5")
print("=" * 62)

x, y = 2.0, -5.0

# forward
a = x + y
b = max(0.0, x)
out = a * b
print(f"forward : a = {a}, b = {b}, f = {out}")

# backward
d_out = 1.0
d_a = d_out * b                  # multiplication: gradient x the OTHER input
d_b = d_out * a
d_x_via_a = d_a * 1.0            # addition passes gradient through
d_y = d_a * 1.0
d_x_via_b = d_b * (1.0 if x > 0 else 0.0)    # relu gates it
d_x = d_x_via_a + d_x_via_b      # eq. 11.13: branching SUMS

print(f"backward: df/da = {d_a}, df/db = {d_b}")
print(f"          df/dx via a = {d_x_via_a}, via b = {d_x_via_b}, "
      f"total = {d_x}")
print(f"          df/dy = {d_y}")


def f_xy(v):
    return (v[0] + v[1]) * max(0.0, v[0])


num = numerical_gradient(f_xy, np.array([x, y]))
print(f"\nhand-computed : [{d_x}, {d_y}]")
print(f"numerical     : {np.round(num, 6)}")
assert np.allclose([d_x, d_y], num, atol=1e-5)

# --- eq. 11.19: the logistic-regression gradient ----------------------------
print("\n" + "=" * 62)
print("logistic regression gradient: prediction error times input")
print("=" * 62)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss(w, xi, yi):
    p = sigmoid(w @ xi)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(yi * np.log(p) + (1 - yi) * np.log(1 - p))


rng = np.random.default_rng(1)
w = rng.normal(size=5)
xi = rng.normal(size=5)
yi = 1.0

p_hat = sigmoid(w @ xi)
analytic = (p_hat - yi) * xi                    # eq. 11.19
numeric = numerical_gradient(lambda v: loss(v, xi, yi), w)

print(f"p_hat = {p_hat:.6f}, y = {yi}")
print(f"analytic (p_hat - y) * x : {np.round(analytic, 6)}")
print(f"numerical                : {np.round(numeric, 6)}")
print(f"max abs difference       : {np.abs(analytic - numeric).max():.2e}")
assert np.allclose(analytic, numeric, atol=1e-6)

# --- the chain rule as multiplication, and why gradients vanish -------------
print("\n" + "=" * 62)
print("the chain rule multiplies — which is why depth is dangerous")
print("=" * 62)
print(f"{'depth':>7} {'sigmoid (x0.25)':>18} {'relu (x1.0)':>14} "
      f"{'slightly >1 (x1.1)':>20}")
for depth in (1, 10, 30, 60):
    print(f"{depth:>7} {0.25**depth:>18.3e} {1.0**depth:>14.3e} "
          f"{1.1**depth:>20.3e}")
print("\nBelow 1 the product vanishes; above 1 it explodes. Keeping the")
print("per-layer factor near 1 is the whole job of initialisation and")
print("normalisation (Part VI).")

# --- eq. 11.7: the Jacobian, and eq. 11.8: chain rule as matmul -------------
def h1(v):
    return np.array([v[0] ** 2, v[0] * v[1], np.sin(v[1])])


def jac_h1(v):
    return np.array([[2 * v[0], 0.0],
                     [v[1], v[0]],
                     [0.0, np.cos(v[1])]])


pt = np.array([1.5, 0.7])
J_analytic = jac_h1(pt)
J_numeric = np.stack([numerical_gradient(lambda v: h1(v)[i], pt) for i in range(3)])
print(f"\nJacobian is {J_analytic.shape} for f: R^2 -> R^3")
assert np.allclose(J_analytic, J_numeric, atol=1e-5)
print("analytic and numerical Jacobians agree")

# eq. 11.8: the Jacobian of a composition is the product of the Jacobians.
A = rng.normal(size=(4, 3))
B = rng.normal(size=(3, 2))
composed = lambda v: A @ (B @ v)
v0 = rng.normal(size=2)
J_comp = np.stack([numerical_gradient(lambda v: composed(v)[i], v0)
                   for i in range(4)])
assert np.allclose(J_comp, A @ B, atol=1e-5)
print(f"J_(f o g) == J_f J_g : {np.allclose(J_comp, A @ B, atol=1e-5)}")
```

## 8. Practical Example

A tiny reverse-mode autodiff engine, in about sixty lines, makes concrete what
PyTorch does. Writing one is the fastest route to never being confused by
autograd again.

```python {tier=A name=micro-autograd}
"""A minimal reverse-mode automatic differentiation engine.

Each Value node stores its data, its parents, and a closure that knows how to
push gradient to those parents. Calling backward() topologically sorts the
graph and runs those closures in reverse — which is exactly what PyTorch does,
minus the tensors, kernels and device management.
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(parents)
        self._op = op

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # Addition passes the gradient through unchanged, to BOTH parents.
            # The += is essential: a node used twice must accumulate.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Each parent receives the gradient times the OTHER parent's value,
            # which is why forward values must be retained.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad      # eq. 2.16

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """Topologically sort, then run each node's backward closure in reverse."""
        order, visited = [], set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    build(parent)
                order.append(node)

        build(self)
        self.grad = 1.0                    # d(self)/d(self) = 1
        for node in reversed(order):
            node._backward()


# --- reproduce the hand computation of section 6.2 --------------------------
x = Value(2.0)
y = Value(-5.0)
f = (x + y) * x.relu()
f.backward()
print("f(x, y) = (x + y) * relu(x)  at x=2, y=-5")
print(f"  f      = {f.data}")
print(f"  df/dx  = {x.grad}   (hand-computed: -1)")
print(f"  df/dy  = {y.grad}   (hand-computed:  2)")
assert x.grad == -1.0 and y.grad == 2.0

# --- a full logistic-regression step, differentiated automatically ----------
print("\nlogistic regression on 3 features, gradient by autodiff:")
w = [Value(0.5), Value(-0.3), Value(0.8)]
xs = [1.2, -0.7, 0.4]
y_true = 1.0

z = w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2]
p = z.sigmoid()
# Binary cross-entropy for y = 1 reduces to -log(p).
loss = -p.log()
loss.backward()

p_val = p.data
print(f"  z = {z.data:.6f}, p = {p_val:.6f}, loss = {loss.data:.6f}")
print(f"  autodiff gradient : {[round(wi.grad, 6) for wi in w]}")
analytic = [(p_val - y_true) * xi for xi in xs]      # eq. 11.19
print(f"  analytic (p-y)*x  : {[round(a, 6) for a in analytic]}")
for wi, a in zip(w, analytic):
    assert abs(wi.grad - a) < 1e-9
print("  they agree exactly.")

# --- branching: a node used twice accumulates gradient ----------------------
a = Value(3.0)
out = a * a + a          # d/da (a^2 + a) = 2a + 1 = 7
out.backward()
print(f"\nd/da (a*a + a) at a=3 : {a.grad}  (expected 7)")
print("The += in each _backward is what makes this correct — a node with")
print("several consumers must SUM its incoming gradients (eq. 11.6).")
assert a.grad == 7.0
```

## 9. Common Mistakes

**Forgetting the chain rule on a composition.** Differentiating $\sin(x^{2})$ as
$\cos(x^{2})$ omits the inner $2x$.

**Assigning instead of accumulating gradient at a branch.** A value used twice
must sum both contributions. Using `=` instead of `+=` in an autodiff engine
produces silently wrong gradients — the demonstration at the end of
{{sec:8-practical-example}} would give 6 instead of 7.

**Confusing $\partial f/\partial x$ with $\dd f/\dd x$.** The partial holds other
variables fixed; the total accounts for their dependence on $x$ too.

**Getting the gradient's shape wrong.** $\nabla f$ has the same shape as the
input. If it does not, something is transposed.

**Using a step size $h$ that is too small in numerical differentiation.** Below
about $10^{-8}$, floating-point cancellation dominates and accuracy gets
*worse*. Central differences at $h \approx 10^{-6}$ are near optimal for double
precision.

**Believing ReLU's kink is a real problem.** It is not, in practice. Hitting
exactly zero has probability zero, and frameworks pick a subgradient by
convention.

**Expecting forward-mode differentiation to be usable for training.** It costs
one sweep per parameter. Reverse mode costs one sweep total.

**Forgetting that the backward pass needs the forward values.** This is why
training memory is several times inference memory, and it is the constraint
gradient checkpointing exists to relax.

## 10. Connection to Previous Chapters

{{ch:math-functions}} supplied composition — which the chain rule
differentiates — and the derivative of the logistic function that
{{eq:dl-dz}} depends on. {{ch:math-vectors}} supplied the dot product, without
which {{eq:steepest-proof}} could not be proved, and the geometric form of the
dot product is doing the real work in that proof.
{{ch:math-matrices}} supplied the matrix product that {{eq:jacobian-chain}}
turns the chain rule into.

Forward: {{ch:math-optimization}} uses the gradient to actually minimise
something, and uses the Hessian to explain when that goes badly.

Beyond Part I: {{ch:dl-backprop}} is {{sec:6-mathematical-foundation}} applied
systematically to a whole network — the engine in
{{sec:8-practical-example}} is a working miniature of it.
{{ch:dl-activations}} chooses activation functions partly by their derivatives,
and {{ch:dl-initialization}} chooses initial weights to keep the per-layer
factor in the chain-rule product near 1. {{ch:tf-scaled-dot-product}} derives
the backward pass through attention using exactly these rules.

## 11. Exercises

**Beginner**

1. Differentiate $f(x) = 3x^{2} + 5x - 2$ and evaluate at $x = 2$.
2. Differentiate $f(x) = e^{2x}$ using the chain rule.
3. For $f(x, y) = x^{3}y^{2}$, compute both partial derivatives.
4. Give $\nabla f$ for $f(x,y,z) = x + 2y + 3z$. What does it tell you about the
   function?
5. Using {{eq:derivative-def}}, estimate the derivative of $\log x$ at $x = 2$
   numerically and compare with $1/2$.

**Intermediate**

6. Differentiate $f(x) = \log(1 + e^{x})$ — the softplus — and show its
   derivative is the logistic function.
7. Verify $\sigma'(x) = \sigma(x)(1-\sigma(x))$ numerically at $x = 1$.
8. For $f(x,y) = x^{2}y + y^{3}$, compute the gradient at $(1, 2)$ and the
   directional derivative along $[1,1]\T/\sqrt{2}$.
9. Work {{sec:6-mathematical-foundation}}'s backpropagation example again with
   $x = -1$, $y = 3$. What changes, and why?
10. Compute the Jacobian of $f(x,y) = [x + y,\; xy,\; x/y]\T$.
11. Derive {{eq:logreg-gradient}} yourself, showing the cancellation explicitly.

**Advanced**

12. Prove that the gradient points in the direction of steepest ascent, being
    explicit about which property of the dot product you use.
13. Show that the Hessian is symmetric when the second partials are continuous
    (Clairaut's theorem — state the condition precisely).
14. Derive the gradient of softmax cross-entropy and show it equals
    $\hat{\vec{p}} - \vec{y}$.
15. Explain, with a cost analysis, why reverse mode is $O(1)$ sweeps and forward
    mode $O(n)$ for a scalar-output function. When would forward mode be
    preferable?
16. Derive the backward pass for $f(\mat{A}, \mat{B}) = \mat{A}\mat{B}$ with
    respect to both matrices, given an incoming gradient $\mat{G}$.

**Implementation**

17. Extend the autodiff engine with `tanh`, `exp`, and division, and verify each
    against numerical differentiation.
18. Add a `zero_grad()` method and use the engine to run 100 steps of gradient
    descent on a two-parameter quadratic.
19. Write a gradient checker comparing analytic against central-difference
    gradients for an arbitrary function, and use it to find a deliberately
    introduced bug in a hand-written gradient.
20. Investigate empirically how numerical-differentiation error depends on $h$,
    sweeping from $10^{-1}$ to $10^{-14}$. Explain the U-shape.

**Reasoning**

21. Training uses several times the memory of inference. Explain why, in terms
    of the backward pass.
22. A network trains for a while and then the loss becomes `nan`. List three
    mechanisms from this chapter and {{ch:math-functions}} that could cause it,
    and how you would distinguish them.

## 12. Chapter Summary

A derivative is a limit of difference quotients, readable as the slope of a
tangent or as a rate of change. Partial derivatives differentiate with respect
to one input, holding the rest fixed; the gradient collects them into a vector
with the same shape as the input.

The gradient points in the direction of steepest increase, and this is a theorem
rather than a definition: the directional derivative is $\nabla f\T\vec{u} =
\norm{\nabla f}\cos\theta$, which is maximised when $\vec{u}$ aligns with the
gradient. Moving perpendicular to the gradient changes nothing. Gradient descent
steps against the gradient because that is provably the locally fastest descent.

The chain rule multiplies derivatives along a composition and sums over paths
where they rejoin. Because a deep network is a composition, its gradient is a
product of per-layer factors — which is why the chain rule is simultaneously
what makes training possible and the cause of vanishing and exploding gradients.

Backpropagation is the chain rule applied systematically over a computational
graph. Three patterns cover most of it: addition distributes gradient unchanged,
multiplication sends each input the gradient times the other input's value, and
a branching node sums the gradients from its consumers.

Reverse-mode differentiation computes the full gradient of a scalar function in
one backward sweep, whatever the number of parameters. Forward mode would need
one sweep per parameter. That asymmetry is why training billion-parameter models
is possible, and its price — retaining forward activations for the backward pass
— is why training needs far more memory than inference.

The Jacobian generalises the gradient to vector-valued functions, and the chain
rule becomes matrix multiplication. The Hessian holds second derivatives and
describes curvature; it is far too large to form for real models, which is why
practical optimisers approximate it rather than compute it.
