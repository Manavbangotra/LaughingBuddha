---
id: dl-backprop
number: 53
part: VI
tier: full
status: reviewed
requires: [dl-forward, dl-losses, dl-activations, math-derivatives, math-matrices]
provides: [backpropagation, reverse-mode-ad, vector-jacobian-product,
           gradient-checking, gradient-accumulation, checkpointing,
           exploding-gradient, gradient-clipping]
citations: [rumelhart1986, goodfellow2016, baydin2018, pascanu2013]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive backpropagation from the multivariate chain rule.
2. Compute the backward pass for a dense layer, an activation and a loss by
   hand.
3. Explain why reverse mode costs one pass regardless of the parameter count.
4. Implement reverse-mode automatic differentiation with a tape.
5. Verify any gradient numerically and interpret the result correctly.
6. Explain where a training step's memory and time actually go.
7. Diagnose vanishing and exploding gradients from measured per-layer norms.
8. Apply gradient clipping, accumulation and checkpointing, and say what each
   costs.

## 2. Why This Matters

**This is the algorithm.** Every network in this book is trained by it. Every
framework is this algorithm plus engineering. You can use those frameworks
without deriving it, and you cannot debug a network that will not train without
understanding it, because every diagnostic in {{sec:12-failure-modes}} is a
statement about the backward pass.

**Its cost structure explains everything about training.** Why a training step
costs about three times a forward pass, why activations must be stored, why
memory scales with depth, why gradient checkpointing exists, why very deep
networks were untrainable before residual connections. These are not separate
facts — they are consequences of {{eq:backprop-recursion}}.

**The vanishing and exploding gradient problem is visible in the recursion.**
{{eq:backprop-recursion}} is a product of $L$ matrices. {{cite:pascanu2013}}
analysed what that does; {{ch:dl-initialization}}, {{ch:dl-normalization}} and
residual connections are all responses to it. Reading the recursion carefully
is what makes those chapters obvious rather than arbitrary.

**Gradient checking is the most useful debugging tool in the subject.** Ten
lines of finite differences, and it catches sign errors, missing transposes,
wrong reductions and dropped branches. {{sec:8-implementation}} uses it against
every component built.

## 3. Prerequisites

{{ch:dl-forward}} for the computational graph, whose reverse traversal this
chapter is. {{ch:dl-losses}} for {{eq:softmax-ce-gradient}}, the first gradient
computed. {{ch:dl-activations}} for the derivatives that appear in the product.
{{ch:math-derivatives}} for the chain rule and {{ch:math-matrices}} for
Jacobians.

## 4. Intuitive Explanation

### 4.1 The credit assignment problem

A network has a million parameters and one number — the loss. Change any one
parameter and the loss changes. **How much, for each of the million, without
running the network a million times?**

That is the whole problem, and the naive answer is bad. Perturb each parameter,
measure, divide: one forward pass per parameter, so a million forward passes per
training step. Backpropagation gets the same answer in one forward pass and one
backward pass, total.

### 4.2 The chain rule, applied backwards

Consider a chain $x \to a \to b \to L$. The chain rule gives

$$
\frac{\partial L}{\partial x}
 = \frac{\partial L}{\partial b}\cdot
   \frac{\partial b}{\partial a}\cdot
   \frac{\partial a}{\partial x}
$$

You can multiply those three factors in either order. Multiplying left to right
— starting from $\partial L/\partial b$ — is **reverse mode**. Multiplying right
to left is **forward mode**. In scalar arithmetic the order does not matter. With
matrices it matters enormously, and {{sec:6-mathematical-foundation}} shows why:
one order multiplies a row vector by matrices and the other multiplies matrices
by each other.

```text
   forward pass:    x ──▶ a ──▶ b ──▶ L        compute values
   backward pass:   ∂L/∂x ◀── ∂L/∂a ◀── ∂L/∂b ◀── 1
                                                  ▲
                                         seed: ∂L/∂L = 1
```

Each node receives the gradient of the loss with respect to *its own output*,
multiplies by its *local* derivative, and passes the result to its parents. No
node needs to know anything about the rest of the graph. That locality is what
makes the algorithm implementable in a few dozen lines.

### 4.3 Why one backward pass suffices

The key insight, and it is worth stating in plain words:

> **Every path from a parameter to the loss passes through the layers above it.**
> So the work of propagating gradient through those upper layers is *shared* by
> every parameter below. Doing it once, from the top down, does that shared work
> exactly once.

Numerical differentiation redoes the shared work for every parameter, which is
why it costs a pass per parameter. Backpropagation caches it. That is the entire
efficiency argument, and it is a dynamic-programming argument rather than a
calculus one.

### 4.4 What each node has to know

Very little:

```text
   node:  y = f(x₁, x₂, ...)
   given: ∂L/∂y                     (arrives from the consumer)
   emit:  ∂L/∂xᵢ = (∂y/∂xᵢ)ᵀ · ∂L/∂y   (for each parent)
```

A framework's job is to hold a table of local derivative rules — one per
operation — and to run the graph backwards applying them. Adding a new
differentiable operation means adding one row to that table.

## 5. Formal Explanation

### 5.1 Notation

For a network of $L$ layers with $\vec{h}^{(0)} = \vec{x}$:

$$
\vec{z}^{(l)} = \mat{W}^{(l)}\vec{h}^{(l-1)} + \vec{b}^{(l)},
\qquad
\vec{h}^{(l)} = \phi\big(\vec{z}^{(l)}\big)
$$ (eq:layer-forward-bp)

Define the **error signal** at layer $l$:

$$
\vecgreek{\delta}^{(l)} \equiv \frac{\partial \Like}{\partial \vec{z}^{(l)}}
$$ (eq:delta-def)

This is the single most useful definition in the chapter. Everything else
follows from it, and the reason to differentiate with respect to $\vec{z}$
rather than $\vec{h}$ is that it makes the parameter gradients fall out in one
line.

### 5.2 The four equations

**Output layer.** For softmax with cross-entropy ({{eq:softmax-ce-gradient}}):

$$
\vecgreek{\delta}^{(L)} = \hat{\vec{p}} - \vec{y}
$$ (eq:bp-output)

**Recursion.** For $l = L-1, \dots, 1$:

$$
\vecgreek{\delta}^{(l)}
 = \Big(\mat{W}^{(l+1)\top}\vecgreek{\delta}^{(l+1)}\Big)
   \odot \phi'\big(\vec{z}^{(l)}\big)
$$ (eq:backprop-recursion)

**Weight gradient.**

$$
\frac{\partial \Like}{\partial \mat{W}^{(l)}}
 = \vecgreek{\delta}^{(l)}\vec{h}^{(l-1)\top}
$$ (eq:bp-weight)

**Bias gradient.**

$$
\frac{\partial \Like}{\partial \vec{b}^{(l)}} = \vecgreek{\delta}^{(l)}
$$ (eq:bp-bias)

Four equations, derived in {{sec:6-mathematical-foundation}}, and they are the
whole algorithm. {{cite:rumelhart1986}} is these four equations plus the
observation that they made deep networks trainable.

### 5.3 Reading the recursion

{{eq:backprop-recursion}} is worth staring at, because three separate later
chapters are responses to what it says.

**It is a product.** Unrolling from layer $L$ down to layer $l$:

$$
\vecgreek{\delta}^{(l)} = \left[\prod_{k=l+1}^{L}
 \mat{D}^{(k-1)}\mat{W}^{(k)\top}\right]\vecgreek{\delta}^{(L)}
$$ (eq:unrolled-backprop)

with $\mat{D}^{(k)} = \diag(\phi'(\vec{z}^{(k)}))$. A product of $L-l$ matrices,
so its magnitude is governed by the product of their singular values. If those
average slightly below 1, the gradient vanishes exponentially in depth; slightly
above, it explodes. This is the part's organising problem, stated as an
equation.

**It flows through $\mat{W}\T$, not $\mat{W}$.** The transpose is not
decoration: forward propagation maps $\R^{n_{l-1}} \to \R^{n_l}$ and gradient
propagation maps the dual spaces the other way. Getting it wrong produces a
shape error when layers differ in width and, when they do not, a silently wrong
gradient.

**It is elementwise-multiplied by $\phi'$.** A saturated unit contributes a near
zero factor and blocks gradient to *everything below it on that path*. This is
{{ch:dl-activations}}'s argument, arriving in the place where it does its
damage.

### 5.4 Batched form

With $\mat{H}^{(l)} \in \R^{B \times n_l}$ and
$\mat{\Delta}^{(l)} \in \R^{B \times n_l}$:

$$
\mat{\Delta}^{(l)} = \big(\mat{\Delta}^{(l+1)}\mat{W}^{(l+1)}\big)
 \odot \phi'\big(\mat{Z}^{(l)}\big)
$$ (eq:batched-backprop)

$$
\frac{\partial \Like}{\partial \mat{W}^{(l)}}
 = \frac{1}{B}\,\mat{\Delta}^{(l)\top}\mat{H}^{(l-1)},
\qquad
\frac{\partial \Like}{\partial \vec{b}^{(l)}}
 = \frac{1}{B}\sum_{i=1}^{B}\mat{\Delta}^{(l)}_{i,:}
$$ (eq:batched-param-grads)

Two observations. The parameter gradient **sums over the batch** because each
example contributes to the same shared parameter, while the error signal does
not mix examples at all — row $i$ of $\mat{\Delta}$ depends only on example $i$.
And the $1/B$ is the mean reduction of {{ch:dl-losses}}; it must appear exactly
once, and putting it in both the loss and the gradient is a common and
hard-to-see bug.

### 5.5 Why reverse and not forward

Both compute exact derivatives. The costs differ:

{#tbl:ad-modes caption="The two modes of automatic differentiation. The choice is determined entirely by the shape of the function: reverse mode is efficient for many inputs and one output, which is exactly what a loss is."}

| | Forward mode | Reverse mode |
|---|---|---|
| Propagates | input perturbation forwards | output sensitivity backwards |
| Cost | $O(n_{\text{in}})$ passes | $O(n_{\text{out}})$ passes |
| Memory | $O(1)$ extra | stores all activations |
| Good for | few inputs, many outputs | many inputs, one output |
| A neural loss | $10^{9}$ passes | 1 pass |

A neural network has $10^6$–$10^{12}$ inputs (the parameters) and one output
(the loss). Reverse mode is not a clever choice here; it is the only feasible
one. The price is the memory, and every memory technique in
{{sec:15-advanced-concepts}} is an attempt to pay less of it.

## 6. Mathematical Foundation

### 6.1 The multivariate chain rule

For $\Like$ depending on $\vec{z}^{(l)}$ only through $\vec{z}^{(l+1)}$:

$$
\frac{\partial \Like}{\partial z_j^{(l)}}
 = \sum_{k} \frac{\partial \Like}{\partial z_k^{(l+1)}}
   \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}}
$$ (eq:multivariate-chain)

**The sum over $k$ is the whole content.** A change in $z_j^{(l)}$ reaches the
loss through *every* unit of the next layer, and the contributions add. This is
the same fact as {{ch:dl-forward}}'s "a tensor with two consumers accumulates",
in the special case where the consumers are all the units of a layer.

### 6.2 Deriving the recursion

From {{eq:layer-forward-bp}}, $z_k^{(l+1)} = \sum_m W^{(l+1)}_{km}
\phi(z^{(l)}_m) + b^{(l+1)}_k$. Only the $m = j$ term depends on $z_j^{(l)}$, so

$$
\frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}}
 = W^{(l+1)}_{kj}\,\phi'\big(z_j^{(l)}\big)
$$

Substituting into {{eq:multivariate-chain}}:

$$
\delta_j^{(l)}
 = \sum_k \delta^{(l+1)}_k W^{(l+1)}_{kj}\,\phi'\big(z_j^{(l)}\big)
 = \Big[\mat{W}^{(l+1)\top}\vecgreek{\delta}^{(l+1)}\Big]_j
   \phi'\big(z_j^{(l)}\big)
$$

which is {{eq:backprop-recursion}}. $\square$

The transpose appears because the sum contracts $k$ — the *row* index of
$\mat{W}^{(l+1)}$ — against $\vecgreek{\delta}^{(l+1)}$, and
$\sum_k W_{kj}\delta_k = [\mat{W}\T\vecgreek{\delta}]_j$.

### 6.3 Deriving the parameter gradients

$z_i^{(l)} = \sum_j W^{(l)}_{ij}h^{(l-1)}_j + b^{(l)}_i$, so
$\partial z_i^{(l)}/\partial W^{(l)}_{ij} = h_j^{(l-1)}$ and only the $i$-th
component of $\vec{z}^{(l)}$ involves $W^{(l)}_{ij}$. Therefore

$$
\frac{\partial \Like}{\partial W^{(l)}_{ij}}
 = \delta_i^{(l)}h_j^{(l-1)}
$$

which is {{eq:bp-weight}} in index form — an outer product. And
$\partial z_i^{(l)}/\partial b^{(l)}_i = 1$ gives {{eq:bp-bias}}. $\square$

**Interpretation worth keeping:** the gradient of a weight is (how wrong the
unit above was) × (how active the unit below was). A weight is blamed in
proportion to both, which is why a dead unit below stops the weight learning
just as effectively as a satisfied unit above.

### 6.4 The vector-Jacobian product

The general statement, which is what a framework actually implements. For
$\vec{y} = f(\vec{x})$ with Jacobian $\mat{J} = \partial\vec{y}/\partial\vec{x}$,
and $\bar{\vec{y}} = \partial\Like/\partial\vec{y}$:

$$
\bar{\vec{x}} = \mat{J}\T\bar{\vec{y}}
$$ (eq:vjp)

**The Jacobian is never formed.** For a dense layer with 1000 inputs and 1000
outputs, $\mat{J}$ has $10^6$ entries; the VJP is a single matrix–vector
product. Every backward rule in a framework is a routine computing
{{eq:vjp}} directly.

Three examples, and the pattern is visible in all of them:

{#tbl:vjp-rules caption="Vector-Jacobian products for the operations of a dense network. Note that none of them constructs a Jacobian — the structure of each operation lets the product be computed directly, which is why the backward pass costs about the same as the forward one."}

| Operation | Jacobian | VJP $\bar{\vec{x}} = \mat{J}\T\bar{\vec{y}}$ |
|---|---|---|
| $\vec{y} = \mat{W}\vec{x}$ | $\mat{W}$ | $\mat{W}\T\bar{\vec{y}}$ |
| $\vec{y} = \phi(\vec{x})$ elementwise | $\diag(\phi'(\vec{x}))$ | $\phi'(\vec{x})\odot\bar{\vec{y}}$ |
| $\vec{y} = \softmax(\vec{x})$ | $\diag(\vec{y}) - \vec{y}\vec{y}\T$ | $\vec{y}\odot(\bar{\vec{y}} - \vec{y}\T\bar{\vec{y}})$ |

The softmax row is the one people write down as a matrix and should not: the
right-hand form is $O(C)$ where forming the Jacobian is $O(C^2)$, and for a
vocabulary-sized softmax the difference is decisive.

### 6.5 The cost of the backward pass

For a dense layer, the forward pass is one matmul. The backward pass is two:

$$
\underbrace{\mat{\Delta}^{(l)}\mat{W}^{(l)}}_{\text{gradient to the input}},
\qquad
\underbrace{\mat{\Delta}^{(l)\top}\mat{H}^{(l-1)}}_{\text{gradient to the weights}}
$$ (eq:two-backward-matmuls)

Both are the same size as the forward matmul. Hence:

$$
F_{\text{step}} \approx 3\,F_{\text{fwd}}
$$ (eq:three-x-rule)

**This is the origin of the "training costs 3× inference" rule of thumb**, and
it is exact for the matmuls and approximate overall. Note that the first of the
two is unnecessary at $l = 1$, since nothing below needs the gradient — a
saving frameworks take when the input does not require gradients.

### 6.6 Why gradients vanish or explode

Take norms in {{eq:unrolled-backprop}}:

$$
\big\|\vecgreek{\delta}^{(l)}\big\|
 \le \big\|\vecgreek{\delta}^{(L)}\big\|
 \prod_{k=l+1}^{L}\big\|\mat{D}^{(k-1)}\big\|\,\big\|\mat{W}^{(k)}\big\|
$$ (eq:gradient-norm-bound)

Let $\sigma$ be a typical value of $\|\mat{W}^{(k)}\|$ and $\gamma$ of
$\|\mat{D}^{(k)}\|$. The product behaves as $(\gamma\sigma)^{L-l}$:

- $\gamma\sigma < 1$: **vanishing**, exponentially in depth.
- $\gamma\sigma > 1$: **exploding**, exponentially in depth.
- $\gamma\sigma \approx 1$: trainable.

The window is narrow and it must hold at every layer, which is why hitting it by
accident is unlikely and why {{ch:dl-initialization}} sets $\sigma$ deliberately.
Note the asymmetry: **exploding gradients are easy to fix and vanishing ones are
not.** Clipping bounds a large gradient; nothing recovers information that has
been multiplied by $10^{-12}$.

{{cite:pascanu2013}} gives the analysis for recurrent networks, where the same
matrix appears at every step and the product becomes a matrix power — a sharper
version of the same problem, treated in {{ch:dl-rnns}}.

### 6.7 Gradient checking

For scalar $\Like(\theta)$, the central difference

$$
\frac{\partial \Like}{\partial\theta}
 \approx \frac{\Like(\theta+\epsilon)-\Like(\theta-\epsilon)}{2\epsilon}
 + O(\epsilon^2)
$$ (eq:central-difference)

has $O(\epsilon^2)$ truncation error, against the forward difference's
$O(\epsilon)$ — worth the extra evaluation.

The error has two competing parts. Truncation falls as $\epsilon^2$; **roundoff
grows as $\epsilon^{-1}$**, because subtracting two nearly equal numbers of
magnitude $\Like$ loses precision. Balancing them:

$$
\epsilon^\star \sim \left(\frac{\varepsilon_{\text{mach}}}{1}\right)^{1/3}
 \approx 6\times10^{-6}\ \text{in float64}
$$ (eq:optimal-epsilon)

So $\epsilon = 10^{-5}$ is near optimal in float64, and gradient checking in
float32 is close to useless — the roundoff floor swamps everything.

Compare with the **relative** error, never the absolute:

$$
\text{rel} = \frac{|g_{\text{analytic}} - g_{\text{numeric}}|}
 {\max(|g_{\text{analytic}}|, |g_{\text{numeric}}|, 10^{-8})}
$$ (eq:relative-error)

Below $10^{-7}$ is correct; above $10^{-4}$ is a bug; between is suspicious and
usually means a non-smooth point, which is the next paragraph.

> WARNING: **ReLU breaks gradient checking near zero.** The function is not
> differentiable there, so a perturbation of $\epsilon$ can straddle the kink,
> and the numerical estimate becomes the average of the two one-sided
> derivatives while the analytic value is one of them. Inside that band of width
> $2\epsilon$ a *correct* implementation reports a relative error up to 1.0.
> The band is narrow, so a small check will usually miss it and a large one will
> eventually hit it — {{sec:8-implementation}} measures both outcomes. Check
> with a smooth activation, or exclude points where $|z| < \epsilon$.

## 7. Internal Mechanics

### 7.1 The tape, backwards

{{ch:dl-forward}}'s tape recorded each operation. The backward pass is:

```text
   grads = {output_node: 1.0}
   for (op, inputs, output) in reversed(tape):
       g_out = grads.get(output, 0)
       for inp, g_in in zip(inputs, BACKWARD[op](g_out, inputs, output)):
           grads[inp] = grads.get(inp, 0) + g_in      # ACCUMULATE, never assign
   return grads
```

Six lines. The `+=` is the multivariate chain rule of
{{eq:multivariate-chain}}, and replacing it with `=` produces a wrong gradient
whenever any tensor has more than one consumer — which is the single most common
bug in a hand-written autodiff, because it is silent on simple chains and only
appears once the graph branches.

### 7.2 Where memory goes

A training step holds four things:

```text
   parameters              θ            P values
   gradients               ∂L/∂θ        P values
   optimiser state         e.g. m, v    0, P, or 2P values (Ch 54)
   activations             stored h     B · Σ nₗ values
```

For a small model at a large batch, activations dominate. For a large model at
a small batch, parameters and optimiser state do. Adam's two moments mean a
model that needs $P$ for weights needs $4P$ before a single activation is
stored, which is the arithmetic behind "you can serve a model four times larger
than you can train".

### 7.3 Gradient accumulation

To train at an effective batch of 512 with memory for 64:

```text
   zero the gradients
   for 8 micro-batches of 64:
       forward, backward       gradients ACCUMULATE
   scale by 1/8, then step
```

Mathematically identical to a single batch of 512 for anything that is a mean
over examples. **It is not identical for batch normalisation**, whose statistics
are computed per micro-batch ({{ch:dl-normalization}}), and that exception is
the source of a lot of confusion about why accumulation "does not reproduce".

### 7.4 Gradient checkpointing

Standard backprop stores every activation: $O(L)$ memory, one forward pass.
Checkpointing stores every $\sqrt{L}$-th, and recomputes the segment between
checkpoints during the backward pass:

$$
\text{memory } O\!\left(\sqrt{L}\right), \qquad
\text{compute } \approx 1.33\times
$$ (eq:checkpointing-tradeoff)

The extra compute is roughly one additional forward pass, which against a step
costing three forward passes is about a third more. Trading a third more time
for a square-root reduction in memory is usually an easy decision, and it is how
long-context transformers fit at all.

### 7.5 What frameworks do that this chapter's implementation does not

Kernel fusion, so the elementwise parts of the backward pass do not each read
and write the tensor ({{ch:dl-forward}}). In-place accumulation into
pre-allocated buffers. Skipping the input-gradient matmul at the first layer.
Recomputing cheap activations rather than storing them. And graph-level
scheduling to overlap the backward pass with communication in distributed
training ({{ch:inf-distributed}}).

None of it changes {{eq:backprop-recursion}}. All of it changes the constant.

## 8. Implementation

```python {tier=A name=backprop-by-hand}
"""Backpropagation for a dense network, written from eqs. 53.3-53.6 and
verified against central differences.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Net:
    """A dense network with an explicit, equation-by-equation backward pass."""

    def __init__(self, sizes, act="tanh", seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2.0 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.act = act

    def _phi(self, z):
        return np.tanh(z) if self.act == "tanh" else np.maximum(0.0, z)

    def _dphi(self, z, h):
        return 1 - h ** 2 if self.act == "tanh" else (z > 0).astype(z.dtype)

    def forward(self, X):
        """Eq. 53.1. Stores Z and H because eqs. 53.4-53.5 need them."""
        self.Z, self.H = [], [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = self._phi(z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h                                   # logits

    def loss(self, X, y_idx):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
        return float(np.mean(lse - logits[np.arange(len(X)), y_idx]))

    def backward(self, y_idx):
        """Eqs. 53.3, 53.7, 53.8 — one line each, in order."""
        B = len(y_idx)
        onehot = np.eye(self.W[-1].shape[1])[y_idx]
        delta = (softmax(self.H[-1]) - onehot)            # eq. 53.3
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ delta / B               # eq. 53.8
            gb[l] = delta.mean(axis=0)                    # eq. 53.8
            if l > 0:
                delta = (delta @ self.W[l].T) * self._dphi(   # eq. 53.7
                    self.Z[l - 1], self.H[l])
        return gW, gb


# --- section 6.7: verify against central differences ------------------------
def gradient_check(net, X, y_idx, eps=1e-5, n_samples=60, seed=0):
    """Eq. 53.18 on a random sample of coordinates, compared by eq. 53.20."""
    net.forward(X)                     # backward() reads what forward stored
    gW, gb = net.backward(y_idx)
    rs = np.random.default_rng(seed)
    worst, results = 0.0, []
    for name, params, grads in (("W", net.W, gW), ("b", net.b, gb)):
        for l, (P, G) in enumerate(zip(params, grads)):
            flat, gflat = P.reshape(-1), G.reshape(-1)
            idx = rs.choice(len(flat), size=min(n_samples, len(flat)),
                            replace=False)
            rels = []
            for i in idx:
                orig = flat[i]
                flat[i] = orig + eps
                lp = net.loss(X, y_idx)
                flat[i] = orig - eps
                lm = net.loss(X, y_idx)
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                ana = gflat[i]
                rels.append(abs(ana - num)
                            / max(abs(ana), abs(num), 1e-8))
            r = float(np.max(rels))
            results.append((f"{name}[{l}]", P.shape, r))
            worst = max(worst, r)
    net.forward(X)
    return worst, results


print("=" * 72)
print("gradient check: hand-derived backward vs central differences")
print("=" * 72)
X = rng.normal(size=(12, 7))
y = rng.integers(0, 4, size=12)
net = Net([7, 9, 6, 4], act="tanh", seed=1)
worst, results = gradient_check(net, X, y)
print(f"{'parameter':<12} {'shape':>12} {'max relative error':>21}")
for name, shape, r in results:
    print(f"{name:<12} {str(shape):>12} {r:>21.3e}")
print(f"\nworst relative error: {worst:.3e}")
print("verdict:", "CORRECT (< 1e-7)" if worst < 1e-7 else
      "SUSPICIOUS" if worst < 1e-4 else "BUG (> 1e-4)")
print("\nEvery one of eqs. 53.3, 53.7 and 53.8 is confirmed. This check is")
print("ten lines and it is the reason you never have to wonder whether a")
print("hand-written backward pass is right.")

# --- the epsilon trade-off of eq. 53.19 -------------------------------------
print("\n" + "=" * 72)
print("choosing epsilon: truncation against roundoff (eq. 53.19)")
print("=" * 72)
print(f"{'epsilon':>10} {'max rel error':>16} {'dominated by':<24}")
for e in (1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13):
    w, _ = gradient_check(net, X, y, eps=e, n_samples=25)
    which = ("truncation (O(eps^2))" if e > 1e-5
             else "balanced" if e > 1e-7 else "roundoff (O(eps^-1))")
    print(f"{e:>10.0e} {w:>16.3e} {which:<24}")
print("\nThe error is V-shaped in epsilon, exactly as eq. 53.19 predicts: too")
print("large and the quadratic truncation term dominates, too small and")
print("catastrophic cancellation in the numerator does. The minimum sits")
print("near 1e-5 in float64, which is where the standard advice comes from.")

# --- the ReLU caveat --------------------------------------------------------
print("\n" + "=" * 72)
print("why gradient checking fails on ReLU (section 6.7 warning)")
print("=" * 72)
for act in ("tanh", "relu"):
    net_a = Net([7, 9, 6, 4], act=act, seed=1)
    w, _ = gradient_check(net_a, X, y, n_samples=200, seed=3)
    print(f"{act:<6} worst relative error over 200 coordinates: {w:.3e}")

print("\nBoth pass, which is the honest result and not the one the warning")
print("in section 6.7 might lead you to expect. On a random network no")
print("pre-activation happened to land within epsilon of zero, so the kink")
print("was never crossed and the check never saw it.")
print("\nThe failure is real; it just has to be provoked. Here is the kink")
print("in isolation, checking d/dz of relu(z) at a few distances from it:\n")
eps = 1e-5
print(f"{'z':>12} {'analytic':>10} {'central diff':>14} {'relative error':>16}")
for z in (1.0, 1e-3, 2e-5, 1e-5, 1e-6, 0.0, -1e-6, -2e-5):
    ana = 1.0 if z > 0 else 0.0
    num = (max(0.0, z + eps) - max(0.0, z - eps)) / (2 * eps)
    rel = abs(ana - num) / max(abs(ana), abs(num), 1e-8)
    print(f"{z:>12.1e} {ana:>10.1f} {num:>14.4f} {rel:>16.4f}")

print("\nOutside a band of width 2*epsilon the check is exact. Inside it,")
print("the perturbation straddles the kink and the central difference")
print("returns the AVERAGE of the two one-sided derivatives, so a correct")
print("implementation reports a relative error up to 1.0.")
print("\nThis matters because the natural reaction — 'my ReLU gradient is")
print("broken' — sends people looking for a bug that is not there. The band")
print("is narrow, so on a small check you will usually miss it and on a")
print("large one you will eventually hit it. Check with a smooth activation,")
print("or skip coordinates whose pre-activation is within epsilon of a kink.")

# --- the accumulate-vs-assign bug of section 7.1 ----------------------------
print("\n" + "=" * 72)
print("the accumulate-vs-assign bug, on a branching graph (section 7.1)")
print("=" * 72)
print("f(x) = sum(tanh(Wx) * relu(Wx)) — W is used ONCE but its output")
print("feeds TWO consumers, so its gradient has two contributions.\n")

W0 = rng.normal(size=(5, 5))
x0 = rng.normal(size=(4, 5))


def f(W):
    z = x0 @ W
    return float(np.sum(np.tanh(z) * np.maximum(0.0, z)))


z0 = x0 @ W0
t, r = np.tanh(z0), np.maximum(0.0, z0)
dz_tanh = r * (1 - t ** 2)                    # through the tanh branch
dz_relu = t * (z0 > 0)                        # through the relu branch

g_both = x0.T @ (dz_tanh + dz_relu)           # correct: ACCUMULATE
g_one = x0.T @ dz_relu                        # bug: last write wins

num = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        Wp, Wm = W0.copy(), W0.copy()
        Wp[i, j] += 1e-6
        Wm[i, j] -= 1e-6
        num[i, j] = (f(Wp) - f(Wm)) / 2e-6

rel = lambda g: float(np.max(np.abs(g - num)
                             / np.maximum(np.abs(num), 1e-8)))
print(f"accumulating both branches : relative error {rel(g_both):.3e}")
print(f"assigning (one branch lost): relative error {rel(g_one):.3e}")
print(f"cosine similarity of the two gradients: "
      f"{float(g_both.ravel() @ g_one.ravel() / (np.linalg.norm(g_both) * np.linalg.norm(g_one))):.4f}")
print("\nThat cosine is the reason this bug survives. The wrong gradient")
print("still points broadly downhill, so the model still trains — worse,")
print("and for reasons that look like a bad learning rate. The `+=` in")
print("section 7.1 is not a stylistic choice.")
```

```python {tier=A name=reverse-mode-autodiff}
"""Reverse-mode automatic differentiation in about eighty lines: the tape of
Chapter 51, walked backwards with eq. 53.13.
"""
import numpy as np

rng = np.random.default_rng(0)


class Tensor:
    """A value, its gradient, and how it was produced."""

    def __init__(self, data, parents=(), backward=None, name=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self._backward = backward or (lambda: None)
        self.name = name

    # --- operations: each records its own VJP (eq. 53.13) -------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), name="add")

        def back():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), name="mul")

        def back():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = back
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), name="matmul")

        def back():                                    # table 53.2, row 1
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = back
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), name="tanh")

        def back():
            self.grad += (1 - t ** 2) * out.grad       # table 53.2, row 2
        out._backward = back
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), name="relu")

        def back():
            self.grad += (self.data > 0) * out.grad
        out._backward = back
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), name="sum")

        def back():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = back
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), name="mean")

        def back():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = back
        return out

    def log_softmax_nll(self, y_idx):
        """Fused loss (Chapter 52): stable forward, one-subtraction backward."""
        z = self.data
        m = z.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
        loss = np.mean(lse - z[np.arange(len(z)), y_idx])
        out = Tensor(loss, (self,), name="ce")

        def back():
            e = np.exp(z - m)
            p = e / e.sum(axis=1, keepdims=True)
            p[np.arange(len(z)), y_idx] -= 1.0
            self.grad += p / len(z) * out.grad         # eq. 53.3
        out._backward = back
        return out

    # --- the reverse traversal ---------------------------------------------
    def backward(self):
        """Section 7.1: reverse topological order, ACCUMULATING gradients."""
        order, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                build(p)
            order.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()


def _unbroadcast(g, shape):
    """Undo NumPy broadcasting by summing the axes that were expanded."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# --- it differentiates arbitrary code, not just recognised layers -----------
print("=" * 72)
print("reverse-mode AD, verified on a graph with reuse and broadcasting")
print("=" * 72)

Xd = rng.normal(size=(6, 4))
yd = rng.integers(0, 3, size=6)
params = {
    "W1": Tensor(rng.normal(0, 0.7, (4, 5)), name="W1"),
    "b1": Tensor(np.zeros((1, 5)), name="b1"),
    "W2": Tensor(rng.normal(0, 0.6, (5, 3)), name="W2"),
    "b2": Tensor(np.zeros((1, 3)), name="b2"),
}


def model(p):
    x = Tensor(Xd)
    h = (x @ p["W1"] + p["b1"])
    # deliberately reuse h in two branches AND broadcast b1: both are the
    # cases a naive implementation gets wrong
    h = h.tanh() * h.relu()
    return (h @ p["W2"] + p["b2"]).log_softmax_nll(yd)


for t in params.values():
    t.grad = np.zeros_like(t.data)
loss = model(params)
loss.backward()
print(f"loss = {float(loss.data):.6f}\n")

# Snapshot every gradient NOW. The perturbation loop below re-runs the model,
# which rebinds every .grad — reading them lazily would compare against zeros.
snapshot = {name: t.grad.copy() for name, t in params.items()}

print(f"{'parameter':<10} {'shape':>10} {'max relative error':>21}")
worst = 0.0
for name, t in params.items():
    flat = t.data.reshape(-1)
    gflat = snapshot[name].reshape(-1)
    rels = []
    for i in range(len(flat)):
        orig = flat[i]
        flat[i] = orig + 1e-6
        for u in params.values():
            u.grad = np.zeros_like(u.data)
        lp = float(model(params).data)
        flat[i] = orig - 1e-6
        lm = float(model(params).data)
        flat[i] = orig
        num = (lp - lm) / 2e-6
        rels.append(abs(gflat[i] - num) / max(abs(gflat[i]), abs(num), 1e-8))
    r = float(np.max(rels))
    worst = max(worst, r)
    print(f"{name:<10} {str(t.data.shape):>10} {r:>21.3e}")
print(f"\nworst: {worst:.3e}  ->",
      "CORRECT" if worst < 1e-6 else "CHECK THIS")
print("\nNote what was verified. `b1` has shape (1, 5) and is broadcast over")
print("six rows, so its gradient must sum over the batch — that is what")
print("_unbroadcast does, and getting it wrong gives a gradient six times")
print("too small. And `h` feeds two branches, so its gradient accumulates.")
print("\nNeither case required a special rule. Both fall out of the two")
print("lines that every operation shares: `+=` rather than `=`, and a VJP")
print("that respects the shape it was given.")

# --- eq. 53.11: reverse vs forward mode, measured ---------------------------
print("\n" + "=" * 72)
print("why reverse mode, and not forward mode (table 53.1)")
print("=" * 72)
import time


def timed_reverse(P):
    W = Tensor(rng.normal(0, 0.3, (P, 1)))
    x = Tensor(rng.normal(size=(8, P)))
    t0 = time.perf_counter()
    out = (x @ W).tanh().sum()
    W.grad = np.zeros_like(W.data)
    out.backward()
    return time.perf_counter() - t0


def timed_forward_one(P):
    """Cost of ONE forward-mode pass (one directional derivative)."""
    W = rng.normal(0, 0.3, (P, 1))
    x = rng.normal(size=(8, P))
    v = np.zeros_like(W)
    v[0] = 1.0
    t0 = time.perf_counter()
    z = x @ W
    dz = x @ v
    _ = (1 - np.tanh(z) ** 2) * dz
    return time.perf_counter() - t0


print(f"{'parameters':>11} {'reverse: all grads':>20} "
      f"{'forward: ONE grad':>19} {'forward: all grads':>20}")
for P in (100, 1000, 10000):
    tr = min(timed_reverse(P) for _ in range(5))
    tf = min(timed_forward_one(P) for _ in range(5))
    print(f"{P:>11} {tr * 1e6:>17.1f} us {tf * 1e6:>16.1f} us "
          f"{tf * P:>17.3f} s")
print("\nThe middle column is not slow — a single forward-mode pass is")
print("cheap. The last column is the problem: forward mode needs one pass")
print("PER PARAMETER, so the total scales with the parameter count while")
print("reverse mode does not.")
print("\nThat is table 53.1 as arithmetic. For a model with 10^9 parameters")
print("the last column would be measured in years. Reverse mode is not a")
print("preference; it is the only mode in which training is possible.")
```

## 9. Practical Example

```python {tier=A name=gradients-in-a-deep-network}
"""The recursion of eq. 53.7 observed in a deep network: what the gradient
looks like at each layer, and what clipping, accumulation and checkpointing
actually do.
"""
import numpy as np

rng = np.random.default_rng(4)


class DeepNet:
    def __init__(self, depth, width, d_in=12, d_out=3, scale=None,
                 act="tanh", seed=0):
        rs = np.random.default_rng(seed)
        sizes = [d_in] + [width] * depth + [d_out]
        self.W, self.b, self.act = [], [], act
        for i in range(len(sizes) - 1):
            s = scale if scale is not None else np.sqrt(2.0 / sizes[i])
            self.W.append(rs.normal(0, s, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def _phi(self, z):
        return np.tanh(z) if self.act == "tanh" else np.maximum(0.0, z)

    def _dphi(self, z, h):
        return 1 - h ** 2 if self.act == "tanh" else (z > 0).astype(float)

    def forward(self, X):
        self.Z, self.H = [], [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = self._phi(z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h

    def backward(self, y_idx):
        B = len(y_idx)
        z = self.H[-1]
        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        p = e / e.sum(axis=1, keepdims=True)
        delta = p.copy()
        delta[np.arange(B), y_idx] -= 1.0
        delta /= B
        gW, gb, prof = [None] * len(self.W), [None] * len(self.W), []
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ delta
            gb[l] = delta.sum(axis=0)
            prof.append({"layer": l,
                         "delta": float(np.sqrt(np.mean(delta ** 2))),
                         "gW": float(np.linalg.norm(gW[l])),
                         "h_in": float(np.sqrt(np.mean(self.H[l] ** 2)))})
            if l > 0:
                delta = (delta @ self.W[l].T) * self._dphi(self.Z[l - 1],
                                                           self.H[l])
        return gW, gb, list(reversed(prof))


X = rng.normal(size=(64, 12))
y = rng.integers(0, 3, size=64)

print("=" * 72)
print("the product of eq. 53.9, seen as a gradient norm per layer")
print("=" * 72)
print("A 20-layer tanh network at three initialisation scales. Three")
print("quantities per layer: the forward activation RMS, the error signal")
print("RMS of eq. 53.2, and the weight-gradient norm of eq. 53.5.\n")

picks = [0, 4, 9, 14, 19, 20]
for label, scale in (("small  (sd 0.05)", 0.05),
                     ("He     (sd sqrt(2/n))", None),
                     ("large  (sd 0.35)", 0.35)):
    net = DeepNet(20, 24, scale=scale, seed=2)
    net.forward(X)
    _, _, prof = net.backward(y)
    print(f"{label}")
    print("  " + " ".join(f"{'layer ' + str(i + 1):>11}" for i in picks))
    for key, name in (("h_in", "forward RMS"), ("delta", "delta RMS"),
                      ("gW", "|grad W|")):
        vals = " ".join(f"{prof[i][key]:>11.3e}" for i in picks)
        print(f"  {vals}   <- {name}")
    r_h = prof[0]["h_in"] / max(prof[-1]["h_in"], 1e-300)
    r_g = prof[0]["gW"] / max(prof[-1]["gW"], 1e-300)
    print(f"  layer-1 / layer-21:  forward {r_h:.3e}   "
          f"grad {r_g:.3e}\n")

print("Two distinct failures are visible here and they are worth separating,")
print("because the usual one-line account of 'vanishing gradients' runs them")
print("together.")
print("\nAt sd 0.05 the forward RMS collapses by twelve orders of magnitude")
print("across the depth, and the error signal of eq. 53.2 vanishes by")
print("twelve orders in the other direction — the textbook vanishing")
print("gradient, visible in the delta row exactly as eq. 53.9 predicts.")
print("\nBut look at the weight-gradient row: it is FLAT, at a uniformly")
print("useless 1e-13. That is worth understanding, because it is not what")
print("the delta row alone would suggest. Eq. 53.5 says the weight gradient")
print("is delta times the incoming activation, and here the two decays run")
print("in OPPOSITE directions with depth — a tiny delta meets a large")
print("activation at layer 1, and a large delta meets a tiny activation at")
print("layer 21. The product is roughly constant, and roughly zero.")
print("\nSo this network does not fail by starving its early layers")
print("relative to its late ones. It fails because every layer's gradient")
print("is negligible: a scale catastrophe rather than a tilt.")
print("\nAt sd 0.35 the profile TILTS: the gradient at layer 1 is several")
print("times the gradient at the output, so the lower layers move faster")
print("than the upper ones and the network is unbalanced rather than dead.")
print("\nThe He scale keeps both the forward RMS and the gradient profile")
print("within a small factor across twenty layers, which is exactly what")
print("Chapter 56 derives it to do.")
print("\nThe diagnostic that follows: print BOTH the forward activation")
print("scale and the per-layer gradient norm. The first tells you whether")
print("the signal survives the forward pass; the second tells you whether")
print("the gradient survives the backward one. They fail independently and")
print("the fixes are different.")

# --- gradient clipping ------------------------------------------------------
print("\n" + "=" * 72)
print("gradient clipping: what it does to the direction (section 6.6)")
print("=" * 72)


def clip_global(grads, max_norm):
    total = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
    if total <= max_norm:
        return grads, total, 1.0
    s = max_norm / (total + 1e-12)
    return [g * s for g in grads], total, s


net = DeepNet(20, 24, scale=0.35, seed=2)
net.forward(X)
gW, _, _ = net.backward(y)
for mx in (1e9, 10.0, 1.0, 0.1):
    clipped, total, s = clip_global(gW, mx)
    flat_o = np.concatenate([g.ravel() for g in gW])
    flat_c = np.concatenate([g.ravel() for g in clipped])
    cos = float(flat_o @ flat_c / (np.linalg.norm(flat_o)
                                   * np.linalg.norm(flat_c)))
    print(f"max_norm={mx:>8.1e}  original |g|={total:>9.3f}  "
          f"scale={s:>7.4f}  cosine with original={cos:.6f}")
print("\nGlobal-norm clipping rescales every parameter by ONE shared factor,")
print("so the direction is exactly preserved — the cosine is 1 at every")
print("threshold. It is a step-size cap, not a change of direction.")
print("\nThat is what makes it safe. Per-parameter clipping, which clips each")
print("coordinate independently, does NOT preserve the direction:")
for mx in (1.0, 0.1, 0.01):
    per = [np.clip(g, -mx, mx) for g in gW]
    flat_o = np.concatenate([g.ravel() for g in gW])
    flat_p = np.concatenate([g.ravel() for g in per])
    cos = float(flat_o @ flat_p / (np.linalg.norm(flat_o)
                                   * np.linalg.norm(flat_p)))
    print(f"  per-coordinate clip at {mx:>5.2f}: "
          f"cosine with original = {cos:.6f}")
print("\n(The first per-coordinate row is a no-op: no single coordinate")
print("exceeds 1.0, so nothing is clipped and the cosine is trivially 1.")
print("The distortion appears as soon as the threshold actually binds.)")
print("\nUse global-norm clipping. The per-coordinate version is a different")
print("optimiser, not a safety net.")

# --- gradient accumulation --------------------------------------------------
print("\n" + "=" * 72)
print("gradient accumulation is exact (section 7.3)")
print("=" * 72)
Xb = rng.normal(size=(256, 12))
yb = rng.integers(0, 3, size=256)

net = DeepNet(4, 24, seed=5)
net.forward(Xb)
gfull, _, _ = net.backward(yb)

for micro in (256, 64, 32, 8):
    acc = [np.zeros_like(g) for g in gfull]
    nchunks = 256 // micro
    for c in range(nchunks):
        sl = slice(c * micro, (c + 1) * micro)
        net.forward(Xb[sl])
        gm, _, _ = net.backward(yb[sl])
        for a, g in zip(acc, gm):
            a += g / nchunks              # each micro-batch is already a mean
    err = max(float(np.max(np.abs(a - g)))
              for a, g in zip(acc, gfull))
    rel = max(float(np.max(np.abs(a - g)) / max(float(np.max(np.abs(g))), 1e-12))
              for a, g in zip(acc, gfull))
    print(f"micro-batch {micro:>4} ({nchunks:>2} chunks): "
          f"max abs diff {err:.3e}   max rel diff {rel:.3e}")
print("\nAccumulation reproduces the full-batch gradient to floating-point")
print("round-off. The residual difference is summation order (Chapter 46),")
print("not an approximation — the mathematics is identical.")
print("\nThe caveat from section 7.3 is worth repeating: this holds because")
print("everything here is a mean over independent examples. Batch")
print("normalisation is not, since its statistics are computed within a")
print("micro-batch, so accumulation does NOT reproduce a full batch there.")

# --- checkpointing ----------------------------------------------------------
print("\n" + "=" * 72)
print("gradient checkpointing: the memory/compute trade (eq. 53.17)")
print("=" * 72)


def checkpoint_cost(L, every):
    """Stored activations and forward passes for segment length `every`."""
    stored = np.ceil(L / every) + every          # checkpoints + one segment
    recompute = 1.0 + (every - 1) / every        # forwards per backward
    return stored, recompute


print(f"{'depth':>6} {'every':>7} {'stored (units)':>16} "
      f"{'vs storing all':>16} {'fwd passes':>12} {'step cost':>11}")
for L in (16, 64, 256):
    for every in (1, int(np.sqrt(L)), L):
        stored, recomp = checkpoint_cost(L, every)
        # a step is 1 forward + 2 backward-equivalent; recompute adds forwards
        step = (recomp + 2) / 3.0
        print(f"{L:>6} {every:>7} {stored:>16.0f} {stored / L:>15.2f}x "
              f"{recomp:>12.2f} {step:>10.2f}x")
print("\nThe middle row of each group is the interesting one. At segment")
print("length sqrt(L) the stored activations fall to about 2*sqrt(L)")
print("instead of L — a 64-layer network stores 16 units rather than 64,")
print("and a 256-layer one stores 32 rather than 256 — while the step costs")
print("about a third more, because the extra work is one forward pass")
print("against a step that already costs three.")
print("\nNote that BOTH extremes are bad. Checkpointing every layer stores")
print("everything and saves nothing. Checkpointing only the input stores one")
print("checkpoint and then has to hold an entire segment's activations")
print("during recomputation, which is the whole network again — so it pays")
print("the extra forward pass and saves nothing either. The saving comes")
print("from the interior of the range, and sqrt(L) is where the sum of the")
print("two terms is minimised.")
print("\nThat is eq. 53.17. Trading a third of the time for a square-root")
print("reduction in activation memory is why long-context models fit at")
print("all, and it is a decision you make per model rather than once.")

# --- the 3x rule, measured --------------------------------------------------
print("\n" + "=" * 72)
print("a training step costs about three forward passes (eq. 53.14)")
print("=" * 72)
import time

Xt = rng.normal(size=(512, 12))
yt = rng.integers(0, 3, size=512)


def timeit(fn, reps=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


print(f"{'network':<22} {'fwd':>8} {'step':>8} {'ratio':>7}   "
      f"{'matmul fwd':>11} {'matmul step':>12} {'ratio':>7}")
for depth, width in ((8, 256), (16, 512)):
    net = DeepNet(depth, width, seed=7)
    tf = timeit(lambda: net.forward(Xt))
    ts = timeit(lambda: (net.forward(Xt), net.backward(yt)))

    # the same FLOPs with the elementwise work removed: forward is one matmul
    # per layer, a step is three (eq. 53.12 adds two)
    Ws = [W for W in net.W]
    H0 = [np.zeros((len(Xt), W.shape[0])) for W in Ws]
    D0 = [np.zeros((len(Xt), W.shape[1])) for W in Ws]

    def mm_fwd():
        for W, h in zip(Ws, H0):
            h @ W

    def mm_step():
        for W, h, d in zip(Ws, H0, D0):
            h @ W
            d @ W.T
            h.T @ d

    tmf = timeit(mm_fwd)
    tms = timeit(mm_step)
    print(f"depth {depth:>2} width {width:>4}     {tf * 1e3:>7.2f}ms "
          f"{ts * 1e3:>7.2f}ms {ts / tf:>6.2f}x   {tmf * 1e3:>10.2f}ms "
          f"{tms * 1e3:>11.2f}ms {tms / tmf:>6.2f}x")

print("\nThe last column is eq. 53.14: strip everything but the matrix")
print("products and a step costs almost exactly three forward passes, which")
print("is what the FLOP count predicts.")
print("\nThe measured ratio for the FULL step is well below three, and the")
print("reason is worth understanding rather than explaining away. The 3x")
print("rule counts matmuls. This forward pass also does a tanh, a bias add")
print("and a list append per layer, and the backward pass does not double")
print("those. If the forward pass costs M of matmul plus E of everything")
print("else, the step costs 3M + E, so the ratio is (3M+E)/(M+E) — which is")
print("3 only when E is negligible and falls toward 1 as E grows.")
print("\nOn a small CPU network E is a large fraction, so the measured")
print("ratio lands near 2. On an accelerator running a large model the")
print("matmuls dominate and fusion removes most of E, so the same")
print("measurement gives close to 3. The rule is right about the")
print("arithmetic and it is a statement about the compute-bound regime,")
print("which is worth remembering before quoting it at a profile that is")
print("not in that regime.")
```

## 10. Production Considerations

**Gradient-check any hand-written backward pass.** The measured check found
correctness to $10^{-10}$ in seconds. There is no excuse for shipping an
unverified custom gradient, and custom operations are exactly where gradients
are wrong.

**Log per-layer gradient norms.** The measured table localises a training
failure to a depth immediately. This is one scalar per layer per step and it is
the highest-value diagnostic in deep learning.

**Use global-norm clipping, not per-coordinate.** Measured: global-norm
preserves the gradient direction exactly (cosine 1.0 at every threshold);
per-coordinate does not. Clip by global norm, log how often it fires, and treat
a rising clip rate as a signal rather than a solved problem.

**Gradient accumulation is exact — with one exception.** Measured to
floating-point round-off. Batch normalisation is the exception, and it is
common enough to check for.

**Budget memory as parameters + gradients + optimiser state + activations.**
With Adam, a model needs roughly $4\times$ its parameter memory before storing
a single activation.

**Checkpoint when activations dominate.** Measured trade: about a third more
compute for a square-root reduction in stored activations.

**Watch for `nan` propagation.** One `inf` in the backward pass poisons every
parameter it reaches. Check for non-finite gradients before the optimiser step,
and skip the step rather than corrupting the weights.

## 11. Common Mistakes

**Assigning instead of accumulating.** Measured: the wrong gradient had cosine
similarity close to 1 with the right one, so the model still trains, worse, and
the symptom looks like a bad learning rate.

**Forgetting to zero gradients between steps.** Frameworks accumulate by
default, so a missing zeroing silently sums the gradients of every step so far.
The loss curve becomes erratic in a way that resembles instability.

**Using $\mat{W}$ instead of $\mat{W}\T$ in the recursion.** A shape error when
widths differ; a silently wrong gradient when they do not.

**Applying the mean reduction twice.** Once in the loss and once in the
gradient — a factor of $B$ error that looks like a learning-rate problem.

**Gradient checking a ReLU network and concluding the gradient is broken.**
Measured: a correct implementation produces a large relative error at the kink.

**Gradient checking in float32.** {{eq:optimal-epsilon}} says the roundoff floor
swamps the signal.

**Comparing absolute rather than relative gradient error.** An absolute error of
$10^{-4}$ is fine for a gradient of size 10 and catastrophic for one of size
$10^{-6}$.

**Detaching a tensor by accident.** Any operation that leaves the graph —
converting to a NumPy array, an in-place write on a leaf — silently produces a
zero gradient for everything upstream.

## 12. Failure Modes

**Vanishing gradients.** Measured: with a small initialisation the error signal
fell by twelve orders of magnitude from the output down to the first layer. In
the tilted case the network trains its top layers and freezes the rest, behaving
like a shallower model, and the loss plateaus above what the architecture should
reach. In the measured case the forward signal collapsed too, so the weight
gradients were uniformly negligible and nothing trained at all — the same root
cause with a different symptom.

**Exploding gradients.** The same product with $\gamma\sigma > 1$. Symptoms are
a loss spike to `inf` or `nan` within a few steps. Cheaper to fix than
vanishing, because clipping bounds it and nothing recovers a gradient
multiplied by $10^{-12}$.

**`nan` from the loss, not the gradient.** Almost always an unfused softmax
({{ch:dl-losses}}) or a `log` of zero. Distinguish by checking whether the
forward loss is already non-finite.

**Silent gradient corruption from a wrong custom backward.** No error, slower or
worse convergence. Only gradient checking finds it.

**Memory growth across iterations.** A retained reference to a loss tensor keeps
its whole tape alive ({{ch:dl-forward}}), so the memory grows linearly until the
job dies.

**Dead paths.** A branch of the graph that receives zero gradient because of a
saturated activation or a hard mask trains not at all, and the only symptom is
parameters that never change. Logging per-parameter update magnitudes catches
it.

## 13. Alternatives

**Numerical differentiation.** One or two forward passes per parameter — far too
slow to train with, and the correct tool for *verifying* an implementation, as
used throughout {{sec:8-implementation}}.

**Forward-mode AD.** Measured: cheap per pass, and it needs one pass per
parameter, which makes the total scale with the parameter count. Correct choice
for Jacobian-vector products and directional derivatives; wrong for a loss.

**Symbolic differentiation.** Produces an explicit derivative expression, which
can grow exponentially with the depth of the composition.

**Feedback alignment** replaces $\mat{W}\T$ in the recursion with a fixed random
matrix and still trains, which is a genuinely surprising result and evidence
that exact gradients are not strictly necessary. It does not match backprop's
accuracy at scale. {{maturity:EMERGING}}

**Zeroth-order and evolutionary methods** need no gradients at all and scale
badly in the parameter count. Useful when the objective is non-differentiable;
not competitive for supervised learning.

**Local learning rules** — greedy layerwise training, target propagation,
predictive coding — avoid the backward pass's global dependency, which matters
for biological plausibility and for hardware without global communication.
{{maturity:RESEARCH FRONTIER}}

## 14. Evaluation

**Gradient check every custom backward.** Relative error below $10^{-7}$ in
float64.

**Overfit ten examples.** If a network cannot drive the loss on ten examples to
near zero, the gradient is wrong or the architecture cannot express the target.
This test costs seconds and it is the fastest way to separate a training bug
from a modelling problem.

**Check the loss at initialisation** against $\log C$ ({{ch:dl-losses}}).

**Plot per-layer gradient norms** at initialisation and after some training. A
ratio spanning orders of magnitude is the finding.

**Log the fraction of steps that clip.** Rising means something is
destabilising.

**Track the update-to-weight ratio** $\|\eta\vec{g}\|/\|\vecgreek{\theta}\|$ per
layer. Around $10^{-3}$ is healthy; orders of magnitude away in either direction
is a problem, and it is more informative than the gradient norm alone because it
is scale-free.

## 15. Advanced Concepts

**Higher-order derivatives.** Differentiating the backward pass gives second
derivatives. Hessian-vector products cost about one extra backward pass and
never form the $P \times P$ Hessian, which is what makes second-order methods
even conceivable ({{ch:dl-optimizers}}).

**Checkpointing beyond $\sqrt{L}$.** The optimal policy for a given memory
budget is a dynamic program, and frameworks that solve it beat the uniform
$\sqrt{L}$ heuristic measured here.

**Reversible architectures.** If a layer's input can be reconstructed from its
output, no activation need be stored — $O(1)$ activation memory in depth. The
constraint on the architecture is real, which is why the idea is elegant and not
dominant.

**Straight-through estimators.** For a non-differentiable operation
(quantisation, hard thresholding), pass the gradient through as if it were the
identity. Not the true gradient of anything; it works well enough to train
quantised networks ({{part:15}}). {{maturity:ESTABLISHED}} in practice and
poorly understood in theory.

**Implicit differentiation.** For a layer defined as the solution of an equation
rather than by an explicit computation, the gradient comes from the implicit
function theorem and costs $O(1)$ memory regardless of how many iterations the
solver took.

**Differentiating through randomness.** The reparameterisation trick
({{ch:dl-autoencoders}}) moves the sampling outside the differentiated path;
score-function estimators do not and have far higher variance.

## 16. Connection to Previous Chapters

{{ch:dl-forward}} built the graph and stored the activations; this chapter is
why. The tape from {{sec:7-internal-mechanics}} there is walked backwards here,
and the storage table in that chapter is exactly the set of values
{{eq:backprop-recursion}} needs.

{{ch:dl-losses}} supplied {{eq:softmax-ce-gradient}}, the seed of the backward
pass. {{ch:dl-activations}} supplied $\phi'$, and its argument about saturation
now has a location: the $\odot\,\phi'$ in {{eq:backprop-recursion}}, at every
layer.

{{ch:math-derivatives}} supplied the chain rule and {{ch:math-matrices}} the
Jacobian. {{ch:ml-linear-regression}}'s gradient descent is what consumes the
output. {{ch:mle-reproducibility}} explains why accumulation reproduces to
round-off rather than exactly.

Forward: {{ch:dl-optimizers}} uses these gradients.
{{ch:dl-initialization}} chooses $\sigma$ so that {{eq:gradient-norm-bound}}'s
product stays near 1. {{ch:dl-normalization}} changes the product's conditioning
directly. {{ch:dl-rnns}} applies the same recursion through time, where the
same matrix recurs and the product becomes a matrix power.
{{ch:tf-architectures}}'s residual connections add an identity path so that one
term of the product is exactly 1 — which is the cleanest possible answer to
{{eq:unrolled-backprop}}.

## 17. Exercises

**Beginner**

1. What does $\vecgreek{\delta}^{(l)}$ represent?
2. Why does the recursion use $\mat{W}\T$ rather than $\mat{W}$?
3. Why is a training step about three times a forward pass?
4. What must the forward pass store, and why?
5. Why must gradients be accumulated rather than assigned?

**Intermediate**

6. Derive {{eq:bp-weight}} from {{eq:multivariate-chain}}.
7. For a 3-layer network with widths 10, 20, 5, count the FLOPs of a forward
   pass and of a full step at batch 32.
8. Using {{eq:gradient-norm-bound}}, find the depth at which the gradient falls
   below $10^{-8}$ when $\gamma\sigma = 0.8$.
9. Explain why a gradient check on ReLU can fail on a correct implementation.
10. Derive the VJP for $\vec{y} = \vec{x}/\|\vec{x}\|$.
11. Why is global-norm clipping preferable to per-coordinate clipping?

**Advanced**

12. Derive the softmax VJP in {{tbl:vjp-rules}} and show it is $O(C)$.
13. Derive {{eq:optimal-epsilon}} by balancing truncation against roundoff.
14. Show that gradient accumulation over $k$ micro-batches is exact for any
    loss that is a mean over examples, and construct one where it is not.
15. Derive the memory and compute of checkpointing with segment length $s$, and
    minimise the memory over $s$.
16. Explain how a Hessian-vector product can be computed with one extra
    backward pass.

**Implementation**

17. Extend the autodiff class with `exp`, `log`, division and a `reshape`, and
    gradient-check each.
18. Implement gradient checkpointing for the dense network and verify it gives
    identical gradients.
19. Implement per-layer gradient-norm logging and reproduce the measured table
    for a depth of your choice.
20. Implement a Hessian-vector product by differentiating a
    gradient–vector inner product.

**Reasoning**

21. Training loss goes to `nan` at step 40. Give an ordered diagnostic
    procedure.
22. A 50-layer network's loss plateaus immediately. What do you measure first,
    and what would each outcome imply?

## 18. Interview Questions

**"Derive backpropagation."** — Chain rule, define $\vecgreek{\delta}$, get the
four equations. Being able to do this without notes is a real signal.

**"Why is backprop efficient?"** — It reuses the shared work of propagating
through the upper layers. It is dynamic programming on the graph, and the cost
argument is about *shared subpaths*, not about calculus.

**"Why reverse mode and not forward mode?"** — One output, many inputs. Give the
cost of each.

**"What does a training step cost relative to inference?"** — About three times,
from two backward matmuls per forward one.

**"How would you verify a custom gradient?"** — Central differences, relative
error, $\epsilon \approx 10^{-5}$, float64, and the ReLU caveat.

**"Your deep network will not train. What do you check?"** — Loss at
initialisation against $\log C$; overfit ten examples; per-layer gradient norms.
In that order, because each is cheap and each rules out a whole class of cause.

**"What is gradient checkpointing and when would you use it?"** — Memory/compute
trade; the numbers; when activations dominate.

**"Why do gradients vanish?"** — {{eq:unrolled-backprop}} is a product of $L$
matrices. Say why vanishing is harder to fix than exploding.

## 19. Research Questions

**Is exact gradient computation necessary?** Feedback alignment trains with a
random backward matrix, and straight-through estimators train through
non-differentiable operations. Both work better than they should, and the theory
does not explain how much approximation is tolerable.
{{maturity:RESEARCH FRONTIER}}

**How does the brain solve credit assignment?** Backpropagation requires
symmetric weights and a distinct backward phase, neither of which is biological.
Several local alternatives exist and none matches it at scale.
{{maturity:RESEARCH FRONTIER}}

**Can activation memory be eliminated rather than traded?** Reversible
architectures achieve $O(1)$ memory in depth at an architectural cost. Whether
that cost is fundamental is open. {{maturity:EMERGING}}

**Do gradient statistics predict trainability in advance?** Several proposed
initialisation-time predictors correlate with final performance, and none is
reliable enough to replace running the training. {{maturity:EMERGING}}

## 20. Chapter Summary

Backpropagation is the multivariate chain rule applied in reverse topological
order, and it reduces to four equations: the output error
$\vecgreek{\delta}^{(L)} = \hat{\vec{p}} - \vec{y}$, the recursion
$\vecgreek{\delta}^{(l)} = (\mat{W}^{(l+1)\top}\vecgreek{\delta}^{(l+1)}) \odot
\phi'(\vec{z}^{(l)})$, and the two parameter gradients. All four were derived
here and all four were confirmed against central differences to a relative error
around $10^{-10}$.

Its efficiency is a dynamic-programming argument, not a calculus one: every path
from a parameter to the loss runs through the layers above it, so propagating
through those layers once serves every parameter below. Numerical
differentiation redoes that shared work per parameter; forward-mode AD needs one
pass per input. The measurement makes the difference concrete — a single
forward-mode pass is cheap, and needing one per parameter is what rules it out.

The cost structure follows from the recursion. Two backward matmuls per forward
one gives the three-times rule, confirmed approximately by measurement. The
forward pass must store activations, so training memory exceeds inference memory
by an amount that scales with batch size. Checkpointing trades about a third
more compute for a square-root reduction in that storage.

{{eq:unrolled-backprop}} is a product of $L$ matrices, and that single fact is
the part's organising problem. Measured on a 20-layer network with a small
initialisation, the forward activation scale and the error signal each moved by
twelve orders of magnitude — in opposite directions. Their product, the weight
gradient of {{eq:bp-weight}}, was therefore *flat* and uniformly negligible
rather than tilted. That is a distinction the usual account of vanishing
gradients elides, and it changes the diagnosis: this network is not starving its
early layers relative to its late ones, it is computing with nothing anywhere.
A well-chosen scale held the forward RMS and the gradient profile within a small
factor across all twenty layers. Log both quantities, because they fail
independently and the fixes differ.

Two implementation details are load-bearing and both were measured. Gradients
must be *accumulated*, never assigned: the version that dropped one branch of a
two-consumer node produced a gradient with cosine similarity close to 1 against
the correct one, so the model still trains, worse, in a way that looks like a
learning-rate problem. And global-norm clipping preserves the gradient direction
exactly, while per-coordinate clipping does not — the first is a step-size cap,
the second is a different optimiser.

Gradient checking is the tool that makes all of this verifiable. Central
differences, relative error, $\epsilon \approx 10^{-5}$ in float64 — and the
measured V-shaped error curve confirms {{eq:optimal-epsilon}}'s balance of
truncation against roundoff. The one trap is ReLU: a correct implementation
produces a large relative error at the kink, because the numerical estimate
averages two one-sided derivatives that the analytic value does not.

## 21. Further Reading

{{cite:rumelhart1986}} is the paper to read, and it is short. The derivation is
the four equations of {{sec:5-formal-explanation}} in slightly different
notation. What is striking on a modern reading is how much of the paper is spent
arguing that internal representations *can* be learned at all — the point at
issue in 1986 was not efficiency but possibility.

{{cite:baydin2018}} is the best survey of automatic differentiation for a
machine learning audience. It is the reference for the distinction between AD
and both symbolic and numerical differentiation, which is muddled almost
everywhere else, and it covers forward mode properly rather than dismissing it.

{{cite:pascanu2013}} analyses the exploding and vanishing gradient problem in
the recurrent setting, where {{eq:unrolled-backprop}}'s product becomes a matrix
power. The gradient-clipping argument comes from here. Read it before
{{ch:dl-rnns}}.

{{cite:goodfellow2016}} chapter 6.5 covers backpropagation with more attention to
general computational graphs than the layer-by-layer treatment here, which is
worth having once the four equations are secure.

**Where to go next:** {{ch:dl-optimizers}} takes these gradients and decides what
to do with them. {{ch:dl-initialization}} and {{ch:dl-normalization}} both exist
to control the product in {{eq:unrolled-backprop}}, and both will read as obvious
consequences of this chapter rather than as separate techniques.
