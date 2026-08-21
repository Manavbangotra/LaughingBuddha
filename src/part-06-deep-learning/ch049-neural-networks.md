---
id: dl-neural-networks
number: 49
part: VI
tier: full
status: reviewed
requires: [ml-logistic, ml-what-it-is, math-derivatives, math-matrices]
provides: [perceptron, neuron-unit, hidden-layer, universal-approximation,
           depth-vs-width, multilayer-perceptron, representation-hierarchy]
citations: [rosenblatt1958, cybenko1989, rumelhart1986, hinton2006, krizhevsky2012]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Implement the perceptron learning rule and state its convergence guarantee.
2. Explain precisely why a perceptron cannot represent XOR, and why that is a
   statement about its hypothesis space rather than its learning rule.
3. Explain why stacking linear layers without a nonlinearity gains nothing.
4. State the universal approximation theorem and, more importantly, what it
   does not say.
5. Explain the depth-versus-width trade-off and why practice chose depth.
6. Write the forward pass of a multi-layer perceptron in matrix form and count
   its parameters and FLOPs.
7. Describe what hidden layers learn and why the representation is
   hierarchical.
8. Explain why the field abandoned neural networks twice and what changed each
   time.

## 2. Why This Matters

**Everything after this chapter is this chapter with more layers.** A
transformer block is affine maps and nonlinearities arranged in a particular
order; a convolutional layer is an affine map with shared weights; an LSTM cell
is affine maps and nonlinearities with a loop. If the two-line unit —
"multiply, add, squash" — and its composition are clear here, the rest of the
book is variations. If they are not, every later architecture is a diagram to
memorise.

**The XOR story is the clearest lesson in the book about hypothesis spaces.**
{{ch:ml-what-it-is}} argued that choosing a model is choosing an assumption. The
perceptron is the case where that choice was made implicitly, the limitation
was proved, and an entire research field stopped for fifteen years as a result.
The technical content is three lines; the methodological content is that
**capability questions and training questions are different questions**, and
confusing them is expensive.

**This is where "deep learning is just curve fitting" gets its answer.** It is
curve fitting, in the sense that {{eq:mlp-forward}} is a parameterised function
fitted by minimising a loss. What the dismissal misses is that the *function
class* is one whose intermediate values are themselves useful — a hierarchy of
representations that transfers to other tasks. That property, not the
approximation power, is why the field looks the way it does, and it is the
thread running to {{part:9}}.

## 3. Prerequisites

{{ch:ml-logistic}} for the sigmoid, the linear predictor and the cross-entropy
loss — a logistic regression is exactly a one-unit network, and this chapter
adds layers to it. {{ch:ml-what-it-is}} for the hypothesis-space framing that
the XOR argument depends on. {{ch:math-matrices}} for matrix multiplication and
shapes. {{ch:math-derivatives}} for the chain rule, used informally here and
derived properly in {{ch:dl-backprop}}.

## 4. Intuitive Explanation

### 4.1 One unit

The unit is two operations and nothing else:

```text
        x₁ ──w₁──┐
        x₂ ──w₂──┤
        x₃ ──w₃──┼──▶ Σ ──▶  z = w·x + b  ──▶ φ(z) ──▶ output
             ...  │
        xₙ ──wₙ──┘        ▲                    ▲
                          │                    │
                     affine map          nonlinearity
```

Take a weighted sum of the inputs, add a bias, pass the result through a
nonlinear function. That is a **unit**, and a layer is many units computing in
parallel from the same inputs.

You have already met one. A logistic regression is a single unit with
$\phi = \sigma$; {{ch:ml-logistic}}'s entire chapter is the analysis of one
neuron. What this part adds is the observation that the *output* of such a unit
can be the *input* to another.

### 4.2 The perceptron and its rule

{{cite:rosenblatt1958}} used a hard threshold rather than a sigmoid, and gave a
learning rule of remarkable simplicity: when you get an example wrong, nudge the
weights towards getting it right.

$$
\vec{w} \leftarrow \vec{w} + \eta\,(y - \hat{y})\,\vec{x}
$$ (eq:perceptron-rule)

If the prediction was too low, add a multiple of the input; if too high,
subtract. That is the whole algorithm, and it comes with a theorem: **if the
data is linearly separable, this converges in a finite number of updates**,
with a bound depending only on the margin and the data's scale — not on the
number of examples or the dimension.

That guarantee generated enormous excitement, and the conditional clause is the
whole story.

### 4.3 Why XOR broke it

Four points. Two features. The label is 1 when exactly one input is 1.

```text
        x₂
         │
       1 ●───────○            ● = class 1
         │       │            ○ = class 0
         │       │
       0 ○───────●
         └───────────  x₁
         0       1

   any straight line leaves one point on the wrong side
```

A perceptron computes $\Ind[w_1x_1 + w_2x_2 + b > 0]$ — a half-plane. XOR's
positive class is the two off-diagonal corners, and no half-plane contains
exactly those two. {{sec:6-mathematical-foundation}} proves it in four
inequalities.

The crucial point, and the one usually lost: **this is not a failure of the
learning rule.** The perceptron rule is optimal for what it is. XOR is simply
not in the hypothesis space, so no amount of training, data, or cleverness in
the optimiser will find it. It is precisely the underfitting of
{{ch:ml-metrics}} — a property of the choice, not of the fit.

The published demonstration of this limitation in 1969 is widely credited with
ending the first wave of neural network research. The rest of this part is what
happened when people came back.

### 4.4 The fix, and the trap

Add a hidden layer. Now the network can build intermediate features — one unit
detecting "at least one input is on", another "both are on" — and combine them.
{{sec:7-internal-mechanics}} constructs exactly those weights by hand.

But there is a trap that must be understood before anything else works. Stack
two *linear* layers:

$$
\mat{W}_2(\mat{W}_1\vec{x} + \vec{b}_1) + \vec{b}_2
= (\mat{W}_2\mat{W}_1)\vec{x} + (\mat{W}_2\vec{b}_1 + \vec{b}_2)
$$ (eq:linear-collapse)

The composition is *another affine map*. A hundred linear layers have exactly
the representational power of one, and all you have gained is a slower way to
parameterise the same function. **The nonlinearity is what makes depth mean
anything**, and this is the single most important sentence in the chapter.

### 4.5 What the hidden layers do

The useful mental model is that each layer transforms the representation into
one where the next layer's job is easier — and the final layer is always a
linear classifier.

```text
   raw pixels  ─▶  edges  ─▶  corners, textures  ─▶  object parts  ─▶  linear
   (not linearly                                                        readout
    separable)                                     (linearly separable)
```

That progression is observed in trained vision networks, not merely asserted.
It is why intermediate activations transfer to other tasks: the network has
learned features, and features are reusable in a way that a decision boundary
is not. Everything in {{part:9}} rests on it.

## 5. Formal Explanation

### 5.1 The multi-layer perceptron

An MLP with $L$ layers computes

$$
\vec{h}^{(0)} = \vec{x}, \qquad
\vec{h}^{(l)} = \phi^{(l)}\!\big(\mat{W}^{(l)}\vec{h}^{(l-1)} + \vec{b}^{(l)}\big),
\qquad
\hat{\vec{y}} = \vec{h}^{(L)}
$$ (eq:mlp-forward)

with $\mat{W}^{(l)} \in \R^{n_l \times n_{l-1}}$ and $\vec{b}^{(l)} \in
\R^{n_l}$. The activation $\phi^{(l)}$ is applied elementwise, and the final
layer's activation is chosen to match the loss — identity for regression,
softmax for classification ({{ch:dl-losses}}).

For a batch of $B$ examples, stack them as rows of $\mat{X} \in \R^{B \times
n_0}$ and the layer becomes one matrix multiply:

$$
\mat{H}^{(l)} = \phi\big(\mat{H}^{(l-1)}\mat{W}^{(l)\top} + \vec{1}\vec{b}^{(l)\top}\big)
$$ (eq:mlp-batched)

This is not a notational nicety. It is why deep learning runs on GPUs at all:
the entire forward pass is a sequence of dense matrix multiplications, which is
the operation modern hardware is built to do.

### 5.2 Parameter and FLOP counts

Layer $l$ has $n_l n_{l-1}$ weights and $n_l$ biases, so

$$
P = \sum_{l=1}^{L} n_l\,(n_{l-1} + 1)
$$ (eq:mlp-params)

and the forward pass for a batch of $B$ costs approximately

$$
F_{\text{fwd}} \approx 2B\sum_{l=1}^{L} n_l n_{l-1}
$$ (eq:mlp-flops)

counting a multiply-accumulate as two operations. The backward pass costs
roughly twice the forward, so a training step is about $3\times$ the forward
cost — a rule of thumb worth remembering and derived in {{ch:dl-backprop}}.

Note what {{eq:mlp-params}} implies: parameters scale with the *product* of
adjacent widths. Doubling every width quadruples the parameters. This is why
fully connected layers on images are hopeless — a $224\times224\times3$ image
into a 1,000-unit layer is 150 million parameters in the first layer alone,
and it is the arithmetic that motivates {{ch:dl-cnns}}.

### 5.3 Universal approximation

**Theorem** ({{cite:cybenko1989}}). Let $\phi$ be a continuous sigmoidal
function. Then finite sums of the form

$$
g(\vec{x}) = \sum_{i=1}^{N}\alpha_i\,\phi\big(\vec{w}_i\T\vec{x} + b_i\big)
$$ (eq:universal-approx)

are dense in $C(I_n)$, the continuous functions on the unit hypercube. That is:
for any continuous $f$ and any $\epsilon > 0$, there exist $N$, $\alpha_i$,
$\vec{w}_i$, $b_i$ with $\sup_{\vec{x}} |g(\vec{x}) - f(\vec{x})| < \epsilon$.

One hidden layer suffices. This is the answer to the question Minsky and Papert
posed.

> IMPORTANT: What the theorem does **not** say, and all four omissions matter.
> It does not bound $N$ — the required width may be exponential in the input
> dimension. It does not say the parameters can be *found*, since it is an
> existence result with no algorithm. It says nothing about behaviour off the
> compact set, so extrapolation is unaddressed. And it says nothing about
> generalisation: approximating $f$ on the training points is not approximating
> $f$.
>
> Universal approximation is therefore reassurance that the hypothesis space is
> not the bottleneck. It is not an argument that networks work, and citing it
> as one is a common error.

### 5.4 Depth versus width

If one layer suffices in principle, why is the field called *deep* learning?

Because the required width can be astronomically large. Depth-separation
results exhibit function families that a depth-$k$ network represents with
polynomially many units and a depth-$(k-1)$ network requires exponentially many
units to approximate.

The intuition is compositional. A deep network can build a feature once and
reuse it in many downstream combinations; a shallow one must re-derive it for
every combination. A function that is naturally written as $f_4 \circ f_3 \circ
f_2 \circ f_1$ costs four layers, and flattening it costs a product rather than
a sum.

{{sec:9-practical-example}} measures this: at a fixed parameter budget, a deep
narrow network beats a shallow wide one on a compositionally structured target,
and the gap grows with the depth of the target's structure.

### 5.5 Why the field stopped twice

Worth knowing, because both stops were caused by something that looks like a
fundamental limitation and turned out to be an engineering problem.

**First winter (roughly 1969–1986).** The perceptron's linearity was
demonstrated to be fatal, and no method existed for training hidden layers. The
thaw was {{cite:rumelhart1986}} popularising backpropagation, which made
multi-layer training practical.

**Second winter (roughly 1995–2006).** Deep networks were trainable in
principle and did not work: gradients vanished, initialisation was poor,
compute and data were scarce, and support vector machines
({{ch:ml-svm}}) had better theory and better results. The thaw was
{{cite:hinton2006}}'s layer-wise pre-training, which showed depth was
achievable, and then {{cite:krizhevsky2012}} demonstrating on ImageNet that
with enough data, GPUs and a better activation, the plain supervised approach
won outright.

The pattern in both cases: a genuine limitation was correctly identified, the
inference "therefore this direction is a dead end" was wrong, and the fix was
methodological rather than conceptual. That is worth holding in mind when
reading confident claims about the limits of current methods.

## 6. Mathematical Foundation

### 6.1 The perceptron convergence theorem

Assume the data is linearly separable with margin $\gamma > 0$: there exists a
unit vector $\vec{w}^{*}$ with $y_i(\vec{w}^{*\top}\vec{x}_i) \ge \gamma$ for
all $i$, with labels $y_i \in \{-1, +1\}$. Assume also $\|\vec{x}_i\| \le R$.

Consider the perceptron rule with $\eta = 1$, updating only on mistakes:
$\vec{w} \leftarrow \vec{w} + y_i\vec{x}_i$.

**Lower bound on the projection.** Each update increases the alignment with
$\vec{w}^{*}$ by at least $\gamma$:

$$
\vec{w}_{k+1}\T\vec{w}^{*} = \vec{w}_k\T\vec{w}^{*} + y_i\vec{x}_i\T\vec{w}^{*}
 \ge \vec{w}_k\T\vec{w}^{*} + \gamma
$$

so after $k$ updates, $\vec{w}_k\T\vec{w}^{*} \ge k\gamma$.

**Upper bound on the norm.** An update happens only on a mistake, which means
$y_i \vec{w}_k\T\vec{x}_i \le 0$, so

$$
\|\vec{w}_{k+1}\|^{2}
 = \|\vec{w}_k\|^{2} + 2y_i\vec{w}_k\T\vec{x}_i + \|\vec{x}_i\|^{2}
 \le \|\vec{w}_k\|^{2} + R^{2}
$$

giving $\|\vec{w}_k\|^{2} \le kR^{2}$.

**Combine.** By Cauchy–Schwarz, $k\gamma \le \vec{w}_k\T\vec{w}^{*} \le
\|\vec{w}_k\| \le \sqrt{k}R$, hence

$$
k \le \frac{R^{2}}{\gamma^{2}}
$$ (eq:perceptron-convergence)

**The number of mistakes is bounded by $(R/\gamma)^{2}$ — independent of the
number of examples and of the dimension.** It depends only on how well
separated the classes are relative to their scale.

This is a genuinely beautiful result and it is worth noticing what it
guarantees: convergence *if a separator exists*. It says nothing whatever about
the case where one does not, and that case is XOR.

### 6.2 XOR is not in the hypothesis space

Suppose weights $w_1, w_2$ and bias $b$ compute XOR with a threshold at zero.
The four constraints are:

$$
\begin{aligned}
(0,0) \to 0: &\quad b \le 0\\
(1,0) \to 1: &\quad w_1 + b > 0\\
(0,1) \to 1: &\quad w_2 + b > 0\\
(1,1) \to 0: &\quad w_1 + w_2 + b \le 0
\end{aligned}
$$

Add the second and third: $w_1 + w_2 + 2b > 0$. From the fourth,
$w_1 + w_2 \le -b$. Substituting, $-b + 2b > 0$, so $b > 0$ — contradicting the
first constraint.

No such $(w_1, w_2, b)$ exists. The proof takes four lines and no assumptions
about the learning procedure, which is exactly the point: **the limitation is
representational.**

### 6.3 Why linear composition collapses

{{eq:linear-collapse}} generalises immediately. For $L$ affine layers with no
nonlinearity,

$$
\vec{h}^{(L)}
 = \mat{W}^{(L)}\cdots\mat{W}^{(1)}\vec{x}
 + \sum_{l=1}^{L}\Big(\prod_{j=l+1}^{L}\mat{W}^{(j)}\Big)\vec{b}^{(l)}
 = \tilde{\mat{W}}\vec{x} + \tilde{\vec{b}}
$$ (eq:deep-linear-collapse)

which is affine. The composition of affine maps is affine, always.

There is a subtlety worth stating, because it is genuinely interesting. The
*function class* is unchanged, but the *parameterisation* is not, and the
optimisation dynamics differ: gradient descent on a deep linear network does
not follow the same trajectory as gradient descent on a single layer, and the
implicit bias it induces is an active research topic. Deep linear networks are
therefore a useful theoretical model precisely because their expressivity is
trivial while their optimisation is not.

For practical purposes: no nonlinearity, no depth.

### 6.4 A constructive sketch of universal approximation

Cybenko's proof is non-constructive, using the Hahn–Banach theorem and the Riesz
representation theorem. A constructive intuition in one dimension is short
enough to give, and it makes the theorem feel less magical.

Take $\phi$ to be a steep sigmoid, so $\phi(a(x - t))$ approximates a step at
$t$ as $a$ grows. Then

$$
\phi\big(a(x-t_1)\big) - \phi\big(a(x-t_2)\big)
$$

approximates a **bump**: approximately 1 on $[t_1, t_2]$ and approximately 0
outside. Two hidden units make one bump.

Any continuous function on a compact interval is uniformly continuous, so it
can be approximated to within $\epsilon$ by a piecewise-constant function on a
fine enough partition. Build one bump per partition cell, scale each by the
function's value there, and sum. That is a one-hidden-layer network, and it
approximates $f$ to within $\epsilon$.

The cost is immediately visible: the number of bumps grows as the partition is
refined, and in $n$ dimensions a grid partition needs $O(k^{n})$ cells. **The
theorem is true and the construction is exponentially wasteful**, which is
precisely the gap depth exists to close.

## 7. Internal Mechanics

### 7.1 What a layer is in memory

A dense layer is a weight matrix, a bias vector, and — during training — the
saved input:

```text
   layer l:
     W    : (n_l, n_{l-1})  float32   parameters, persistent
     b    : (n_l,)          float32   parameters, persistent
     ─────────────────────────────────────────────────────────
     h_in : (B, n_{l-1})    float32   SAVED for the backward pass
     z    : (B, n_l)        float32   saved if phi' needs it
     h_out: (B, n_l)        float32   passed to the next layer
```

The row worth noticing is the third. The parameters are a fixed cost; the saved
activations scale with **batch size**, and they are usually the dominant memory
consumer in training. A network with 10 million parameters occupies 40 MB of
weights and can easily use several gigabytes of activation memory at a batch
size of 256. {{sec:8-implementation}} measures the ratio, and
{{ch:dl-backprop}} explains why they must be saved at all.

### 7.2 The forward pass, operation by operation

For each layer:

1. **GEMM.** $\mat{Z} = \mat{H}\mat{W}\T$, a general matrix multiply. This is
   where essentially all the FLOPs are, and it is the operation that determines
   whether a model is fast.
2. **Bias add.** Broadcast $\vec{b}$ across the batch dimension. Memory-bound,
   negligible FLOPs, and usually fused into the GEMM by the library.
3. **Activation.** Elementwise $\phi$. Also memory-bound, also usually fused.

The distinction between *compute-bound* and *memory-bound* operations runs
through the rest of this book. A GEMM does $O(Bn_ln_{l-1})$ work on
$O(Bn_l + n_ln_{l-1})$ data, so its arithmetic intensity grows with size and it
can saturate the hardware's arithmetic units. An elementwise activation does
$O(Bn_l)$ work on $O(Bn_l)$ data — one operation per element loaded — so it is
limited by memory bandwidth no matter how fast the arithmetic is. This is why
kernel fusion matters, and why {{ch:tf-efficient}}'s FlashAttention is a memory
argument rather than a FLOP argument.

### 7.3 Constructing an XOR network by hand

The clearest way to believe a hidden layer changes the hypothesis space is to
write the weights down. Two hidden units with a step activation:

$$
h_1 = \Ind[x_1 + x_2 - 0.5 > 0] \quad(\text{OR}), \qquad
h_2 = \Ind[x_1 + x_2 - 1.5 > 0] \quad(\text{AND})
$$

and an output unit computing $h_1 - h_2$:

{#tbl:xor-hand caption="XOR solved by hand with two hidden units. The hidden layer re-represents the input in coordinates where the problem is linearly separable — which is what every hidden layer does."}

| $x_1$ | $x_2$ | $h_1$ (OR) | $h_2$ (AND) | $h_1 - h_2$ | XOR |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 1 | 1 |
| 0 | 1 | 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 | 0 |

Look at the $(h_1, h_2)$ column pair. The four inputs map to $(0,0)$, $(1,0)$,
$(1,0)$, $(1,1)$ — three distinct points, and the two positive cases now
coincide. In that space the classes *are* linearly separable, by the line
$h_1 - h_2 = 0.5$.

**That is the whole idea of a hidden layer**, and it is worth stating in the
general form: a hidden layer learns a change of coordinates in which the next
layer's job is linear. Every architecture in this book is an opinion about what
kind of coordinate change is useful.

### 7.4 Initialisation cannot be zero

If every weight starts at zero — or at any identical value — every unit in a
layer computes the same thing, receives the same gradient, and updates
identically. They remain identical forever. The layer has the capacity of one
unit no matter how wide it is.

This is **symmetry breaking**, and it is the reason weights are initialised
randomly. Biases can safely be zero, because the weights already break the
symmetry. {{ch:dl-initialization}} takes up the question of what the random
*scale* should be, which turns out to matter enormously; {{sec:8-implementation}}
here just measures the failure.

## 8. Implementation

```python {tier=A name=perceptron-and-xor}
"""The perceptron, its convergence bound, and the limitation that stopped
the field for fifteen years.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the rule (eq. 49.1) ----------------------------------------------------
def perceptron_fit(X, y, max_epochs=1000):
    """y in {-1, +1}. Updates only on mistakes. Returns (w, b, n_updates)."""
    n, d = X.shape
    w, b, updates = np.zeros(d), 0.0, 0
    for epoch in range(max_epochs):
        errors = 0
        for i in range(n):
            if y[i] * (X[i] @ w + b) <= 0:          # a mistake
                w += y[i] * X[i]
                b += y[i]
                updates += 1
                errors += 1
        if errors == 0:
            return w, b, updates, epoch + 1
    return w, b, updates, max_epochs


def margin_under(X, y, w):
    """Margin achieved by a KNOWN unit-norm separator w.

    A random search for the best separator fails badly in high dimensions —
    a random unit vector in 20-D is essentially never a good separator — so
    the honest thing is to use the w the data was generated from. That gives
    a valid lower bound on the true margin, hence a valid UPPER bound on
    (R/gamma)^2, which is what eq. 49.7 needs.
    """
    return float(np.min(y * (X @ w)))


# --- section 6.1: the convergence bound, checked ----------------------------
print("=" * 72)
print("the perceptron convergence bound (eq. 49.7)")
print("=" * 72)
print("The theorem says mistakes <= (R/gamma)^2, independent of n and d.\n")
print(f"{'n':>6} {'d':>4} {'R':>7} {'gamma':>8} {'(R/gamma)^2':>13} "
      f"{'actual updates':>16}")
for n, d, sep in ((50, 2, 1.5), (500, 2, 1.5), (50, 20, 1.5),
                  (500, 20, 1.5), (500, 2, 0.4)):
    # a linearly separable problem with a controlled gap
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(n, d))
    proj = X @ w_true
    keep = np.abs(proj) > sep / 2                     # carve out a margin
    X, proj = X[keep], proj[keep]
    y = np.sign(proj)
    R = float(np.max(np.linalg.norm(X, axis=1)))
    gamma = margin_under(X, y, w_true)
    _, _, upd, _ = perceptron_fit(X, y)
    bound = (R / gamma) ** 2 if gamma > 0 else np.inf
    print(f"{len(y):>6} {d:>4} {R:>7.3f} {gamma:>8.4f} {bound:>13.1f} "
          f"{upd:>16}")

print("\nThe actual number of updates stays far below the bound in every row,")
print("and — the point of the theorem — it does not grow with n. Going from")
print("14 examples to several hundred, or from 2 dimensions to 20, barely")
print("moves it.")
print("\nThe last row is the one that does move the bound: halving the")
print("enforced gap shrinks gamma and the bound grows as 1/gamma^2, exactly")
print("as eq. 49.7 says. Difficulty for a perceptron is measured by how close")
print("the classes come, not by how much data there is.")

# --- section 6.2: XOR is not representable ----------------------------------
print("\n" + "=" * 72)
print("XOR: the failure is REPRESENTATIONAL, not a training failure")
print("=" * 72)
X_xor = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y_xor = np.array([-1., 1., 1., -1.])

w, b, upd, epochs = perceptron_fit(X_xor, y_xor, max_epochs=2000)
pred = np.where(X_xor @ w + b > 0, 1.0, -1.0)     # break the tie at zero
print(f"after {epochs:,} epochs and {upd:,} updates the rule has still not")
print("converged — it cycles forever, because there is nothing to converge")
print("to:")
print(f"  weights {np.round(w, 3)}, bias {b:.3f}")
print(f"  accuracy {np.mean(pred == y_xor):.2f}  (chance is 0.50)")
print("\nThe weights returned to exactly zero: the updates cancel over a")
print("cycle. Section 6.1's guarantee assumed separability, and without it")
print("the theorem says nothing at all — not that convergence is slow, but")
print("that there is no fixed point.")

# exhaustive search over a fine grid of ALL possible perceptrons
print("\nsearching every perceptron on a grid, to rule out bad luck:")
best_acc, n_tried = 0.0, 0
grid = np.linspace(-4, 4, 81)
for w1 in grid:
    for w2 in grid:
        for bb in np.linspace(-4, 4, 41):
            n_tried += 1
            acc = np.mean(np.sign(X_xor @ np.array([w1, w2]) + bb) == y_xor)
            best_acc = max(best_acc, acc)
print(f"  {n_tried:,} weight settings tried")
print(f"  best accuracy achievable by ANY perceptron: {best_acc:.2f}")
print("\nNo perceptron reaches 1.00, and the four-line proof in section 6.2")
print("says none ever will. This is the underfitting of Chapter 34: a")
print("property of the hypothesis space, not of the optimiser.")

# --- section 7.3: two hidden units are enough -------------------------------
print("\n" + "=" * 72)
print("a hidden layer changes the hypothesis space (table 49.1)")
print("=" * 72)
W1 = np.array([[1.0, 1.0],        # h1 = OR
               [1.0, 1.0]])       # h2 = AND
b1 = np.array([-0.5, -1.5])
W2 = np.array([[1.0, -1.0]])      # output = h1 - h2
b2 = np.array([-0.5])


def step(z):
    return (z > 0).astype(float)


H = step(X_xor @ W1.T + b1)
out = step(H @ W2.T + b2).ravel()
print(f"{'x1':>4} {'x2':>4} {'h1(OR)':>8} {'h2(AND)':>9} {'output':>8} "
      f"{'XOR':>5}")
for i in range(4):
    print(f"{X_xor[i,0]:>4.0f} {X_xor[i,1]:>4.0f} {H[i,0]:>8.0f} "
          f"{H[i,1]:>9.0f} {out[i]:>8.0f} {(y_xor[i] > 0) * 1:>5}")
print(f"\naccuracy: {np.mean((out > 0.5) == (y_xor > 0)):.2f}")

print("\nLook at the (h1, h2) columns. The four inputs map to three distinct")
print("points and the two POSITIVE cases now coincide at (1, 0). In those")
print("coordinates the classes are linearly separable, and the output unit")
print("is an ordinary perceptron. The hidden layer did not add power to the")
print("classifier; it changed the coordinates the classifier works in.")

# --- section 6.3: without a nonlinearity, depth is free of charge -----------
print("\n" + "=" * 72)
print("no nonlinearity, no depth (eq. 49.9)")
print("=" * 72)
rs = np.random.default_rng(4)
x = rs.normal(size=(6, 5))
Ws = [rs.normal(size=(7, 5)) * 0.5, rs.normal(size=(9, 7)) * 0.5,
      rs.normal(size=(3, 9)) * 0.5]
bs = [rs.normal(size=7) * 0.1, rs.normal(size=9) * 0.1, rs.normal(size=3) * 0.1]

h = x
for W, b in zip(Ws, bs):
    h = h @ W.T + b                              # NO activation
deep_linear = h

W_eq = Ws[2] @ Ws[1] @ Ws[0]
b_eq = Ws[2] @ Ws[1] @ bs[0] + Ws[2] @ bs[1] + bs[2]
single = x @ W_eq.T + b_eq

print(f"3 linear layers (5 -> 7 -> 9 -> 3), {sum(W.size + b.size for W, b in zip(Ws, bs))} parameters")
print(f"equivalent single layer (5 -> 3), {W_eq.size + b_eq.size} parameters")
print(f"max |difference| in outputs: {np.abs(deep_linear - single).max():.2e}")
print("\nIdentical to machine precision. The three-layer network is an")
print("elaborate parameterisation of a 5->3 affine map and represents")
print("nothing a single layer cannot.")

# ...and with a nonlinearity it is not
h = x
for W, b in zip(Ws, bs):
    h = np.tanh(h @ W.T + b)
print(f"\nwith tanh between the layers, max |difference| from the single")
print(f"equivalent layer: {np.abs(h - single).max():.4f}  (no longer affine)")

# --- section 7.4: symmetry breaking -----------------------------------------
print("\n" + "=" * 72)
print("why weights cannot be initialised to a constant (section 7.4)")
print("=" * 72)


def train_mlp(X, y, hidden=8, epochs=600, lr=0.5, init="random", seed=0):
    """A minimal two-layer MLP with tanh and squared error, trained by
    explicit gradients. Backpropagation is derived properly in Chapter 53;
    this is the two-layer case written out by hand."""
    rs = np.random.default_rng(seed)
    d = X.shape[1]
    if init == "zeros":
        W1, W2 = np.zeros((hidden, d)), np.zeros((1, hidden))
    elif init == "constant":
        W1, W2 = np.full((hidden, d), 0.5), np.full((1, hidden), 0.5)
    else:
        W1 = rs.normal(0, 0.8, (hidden, d))
        W2 = rs.normal(0, 0.8, (1, hidden))
    b1, b2 = np.zeros(hidden), np.zeros(1)
    for _ in range(epochs):
        z1 = X @ W1.T + b1
        h1 = np.tanh(z1)
        out = (h1 @ W2.T + b2).ravel()
        err = out - y
        gW2 = (err[:, None] * h1).mean(0, keepdims=True)
        gb2 = err.mean(keepdims=True)
        dh = err[:, None] * W2
        dz = dh * (1 - h1 ** 2)
        gW1 = dz.T @ X / len(y)
        gb1 = dz.mean(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    z1 = X @ W1.T + b1
    h1 = np.tanh(z1)
    out = (h1 @ W2.T + b2).ravel()
    return float(np.mean((out - y) ** 2)), W1, h1


y01 = (y_xor > 0).astype(float)
print(f"{'initialisation':<16} {'final MSE':>11} {'distinct hidden units':>24}")
for init in ("zeros", "constant", "random"):
    mse, W1, h1 = train_mlp(X_xor, y01, init=init)
    n_distinct = len(np.unique(np.round(W1, 6), axis=0))
    print(f"{init:<16} {mse:>11.6f} {n_distinct:>24}")

print("\nWith identical initial weights every hidden unit computes the same")
print("function, receives the same gradient, and stays identical forever —")
print("an 8-unit layer with the capacity of one unit. Random initialisation")
print("breaks the symmetry, and only then does the network solve XOR.")
print("\nBiases can safely start at zero: the weights already break the tie.")
```

## 9. Practical Example

```python {tier=A name=depth-vs-width}
"""Depth versus width at a fixed parameter budget, and what hidden layers
learn.
"""
import numpy as np

rng = np.random.default_rng(7)


# --- a minimal MLP trained with explicit gradients --------------------------
class MLP:
    """Fully connected, tanh hidden, linear output, squared error.

    Written out by hand rather than with a framework so the parameter count
    and the gradient are both visible. Chapter 53 replaces this with
    reverse-mode autodiff.
    """

    def __init__(self, sizes, seed=0, scale=None):
        rs = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            s = scale if scale is not None else np.sqrt(1.0 / fan_in)
            self.W.append(rs.normal(0, s, (sizes[i + 1], sizes[i])))
            self.b.append(np.zeros(sizes[i + 1]))

    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def forward(self, X):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            h = z if i == len(self.W) - 1 else np.tanh(z)
            acts.append(h)
        return h.ravel() if h.shape[1] == 1 else h, acts

    def fit(self, X, y, epochs=4000, lr=0.05, batch=128, seed=0):
        rs = np.random.default_rng(seed)
        n = len(y)
        for ep in range(epochs):
            idx = rs.integers(0, n, min(batch, n))
            xb, yb = X[idx], y[idx]
            out, acts = self.forward(xb)
            grad = (2.0 * (out - yb) / len(yb))[:, None]
            for i in range(len(self.W) - 1, -1, -1):
                h_in = acts[i]
                gW = grad.T @ h_in
                gb = grad.sum(0)
                if i > 0:
                    grad = (grad @ self.W[i]) * (1 - acts[i] ** 2)
                self.W[i] -= lr * gW
                self.b[i] -= lr * gb
        return self

    def mse(self, X, y):
        out, _ = self.forward(X)
        return float(np.mean((out - y) ** 2))


# --- a target with genuine compositional structure --------------------------
def compositional_target(X, depth):
    """f_depth o ... o f_1, each stage a fixed smooth map of the previous.

    A target built by composition should favour a model that composes. A
    target that is a simple sum should not, and both are measured below.
    """
    h = X.copy()
    for k in range(depth):
        h = np.tanh(1.7 * h @ ROT[k].T + SHIFT[k])
    return h[:, 0] * 2.0


D_IN = 4
ROT = [np.linalg.qr(np.random.default_rng(100 + k).normal(size=(D_IN, D_IN)))[0]
       for k in range(6)]
SHIFT = [np.random.default_rng(200 + k).normal(0, 0.4, D_IN) for k in range(6)]


def additive_target(X, depth=None):
    """A target with no compositional structure: a sum of one-dimensional
    functions. Depth should buy nothing here."""
    return (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1]
            - np.abs(X[:, 2]) + 0.5 * X[:, 3] ** 2)


print("=" * 72)
print("depth vs width at a MATCHED parameter budget (section 5.4)")
print("=" * 72)
Xtr = rng.uniform(-2, 2, (3000, D_IN))
Xva = rng.uniform(-2, 2, (1500, D_IN))
Xte = rng.uniform(-2, 2, (4000, D_IN))

LRS = (0.2, 0.05, 0.01, 0.003, 0.001)


def best_fit(sizes, Xtr, ytr, Xva, yva, seed=1):
    """Each architecture gets its own learning rate, selected on VALIDATION
    data. Without this the comparison is unfair: a wide layer produces
    larger gradients and diverges at a rate a narrow one is happy with —
    which is itself a preview of Chapter 56."""
    best = (None, np.inf, None)
    for lr in LRS:
        # rates that are too large diverge to inf/nan; that is information,
        # not an error, so the overflow warnings are suppressed and the
        # non-finite result simply loses the selection
        with np.errstate(over="ignore", invalid="ignore"):
            m = MLP(sizes, seed=seed).fit(Xtr, ytr, epochs=6000, lr=lr, seed=2)
            v = m.mse(Xva, yva)
        if np.isfinite(v) and v < best[1]:
            best = (m, v, lr)
    return best

# architectures chosen to have comparable parameter counts
# widths chosen so the parameter counts genuinely match — a comparison at
# unmatched budgets would be measuring size, not shape
archs = [
    ("1 hidden x 433", [D_IN, 433, 1]),
    ("2 hidden x 48", [D_IN, 48, 48, 1]),
    ("3 hidden x 34", [D_IN, 34, 34, 34, 1]),
    ("5 hidden x 24", [D_IN, 24, 24, 24, 24, 24, 1]),
]

for target_name, target_fn, depth in (
        ("additive (no composition)", additive_target, None),
        ("composition of 4 stages", compositional_target, 4)):
    ytr = (target_fn(Xtr, depth) if depth else target_fn(Xtr)) \
        + rng.normal(0, 0.02, len(Xtr))
    yva = target_fn(Xva, depth) if depth else target_fn(Xva)
    yte = target_fn(Xte, depth) if depth else target_fn(Xte)
    var = float(np.var(yte))
    print(f"\n{target_name}   (target variance {var:.4f})")
    print(f"{'architecture':<18} {'params':>8} {'best lr':>9} "
          f"{'test MSE':>11} {'fraction unexplained':>22}")
    for name, sizes in archs:
        m, _, lr = best_fit(sizes, Xtr, ytr, Xva, yva)
        mse = m.mse(Xte, yte)
        print(f"{name:<18} {m.n_params:>8} {lr:>9} {mse:>11.5f} "
              f"{mse / var:>22.4f}")

print("\nNote the learning-rate column before the accuracy column: the wide")
print("shallow network needs a substantially smaller step than the narrow")
print("deep ones, and diverges outright at rates they are happy with. That")
print("is a preview of Chapter 56 — the scale of a layer's gradient depends")
print("on its width — and it is why this comparison tunes the rate per")
print("architecture rather than fixing one.")
print("\nTwo effects are visible and they need separating.")
print("\nFIRST: going from one hidden layer to two helps on BOTH targets, and")
print("substantially. A single wide layer is simply harder to optimise, and")
print("that is an optimisation fact rather than a representational one — the")
print("theorem in section 5.3 says the wide layer COULD express either")
print("target.")
print("\nSECOND, and this is the depth-separation signature: past two layers")
print("the two targets diverge. On the ADDITIVE target the error flattens —")
print("three and five layers are no better than two, because there is")
print("nothing left to compose. On the COMPOSED target it keeps falling all")
print("the way to five layers, ending roughly thirty times better than the")
print("single wide layer.")
print("\nThat is the mechanism made concrete: a deep network can build a")
print("feature once and reuse it downstream, and a shallow one must")
print("re-derive it for every combination. Depth pays in proportion to how")
print("much compositional structure the target actually has.")
print("\nNote the honest limit of this experiment. It shows depth helping on")
print("a target that was DEFINED by composition. It does not show that real")
print("problems are compositional — that is an empirical bet the field has")
print("made and largely won, not a theorem.")

# --- what the hidden units actually learn -----------------------------------
print("\n" + "=" * 72)
print("what a hidden layer learns (section 4.5)")
print("=" * 72)
print("A two-dimensional problem, so the learned features can be read.\n")


def two_ring(n, rs):
    r = rs.uniform(0, 3, n)
    th = rs.uniform(0, 2 * np.pi, n)
    X = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return X, (r > 1.6).astype(float)


rs = np.random.default_rng(11)
Xr, yr = two_ring(2000, rs)
Xr_te, yr_te = two_ring(3000, rs)

net = MLP([2, 6, 1], seed=3).fit(Xr, yr, epochs=8000, lr=0.1, seed=4)
out, acts = net.forward(Xr_te)
acc = float(np.mean((out > 0.5) == (yr_te > 0.5)))
print(f"test accuracy on concentric rings: {acc:.4f}")

# each hidden unit is a half-plane in the INPUT space; the output layer is
# a linear readout of those half-planes
h = acts[1]
print(f"\n{'hidden unit':>12} {'weight vector':>22} {'output weight':>14} "
      f"{'corr with radius':>18}")
radius = np.linalg.norm(Xr_te, axis=1)
for j in range(6):
    w = net.W[0][j]
    print(f"{j:>12} {str(np.round(w, 2)):>22} {net.W[1][0, j]:>14.3f} "
          f"{np.corrcoef(h[:, j], radius)[0, 1]:>18.3f}")

print("\nEach hidden unit is still a linear boundary in the INPUT space —")
print("that has not changed. What changed is that the output layer sees six")
print("of them at once, and a weighted combination of half-planes can carve")
print("out a ring that no single half-plane can.")
print("\nNotice the last column: no individual unit correlates strongly with")
print("the radius, which is the quantity that actually determines the label.")
print("The representation is DISTRIBUTED — the information is in the pattern")
print("across units, not in any one of them. That is the general case, and it")
print("is why interpreting individual neurons is so difficult (Chapter 229).")

# --- parameter and FLOP arithmetic (eqs. 49.3, 49.4) ------------------------
print("\n" + "=" * 72)
print("why fully connected layers do not scale to images (eq. 49.3)")
print("=" * 72)
print(f"{'input':<22} {'first layer':>14} {'parameters':>16} "
      f"{'GFLOPs @ B=32':>15}")
for label, n_in, n_out in (("4 features", 4, 96),
                           ("784 (28x28 grey)", 784, 1000),
                           ("150,528 (224x224x3)", 224 * 224 * 3, 1000)):
    params = n_out * (n_in + 1)
    flops = 2 * 32 * n_out * n_in / 1e9
    print(f"{label:<22} {n_out:>14,} {params:>16,} {flops:>15.3f}")

print("\nA single fully connected layer on a modest image is 150 million")
print("parameters — more than most complete modern vision models — and it")
print("has learned nothing about images in the process: it treats a pixel and")
print("its neighbour as unrelated coordinates. Chapter 59's convolution is")
print("the response, and it is an argument about ARITHMETIC as much as about")
print("inductive bias.")
```

## 10. Production Considerations

**Parameter count is not the cost that matters.** {{eq:mlp-params}} gives the
memory for weights; {{sec:7-internal-mechanics}} showed activations usually
dominate during training, scaling with batch size rather than model size. For
inference the reverse holds — no activations are retained — which is why a model
that trains only on an 80 GB accelerator may serve comfortably on a much
smaller one.

**Precision.** Weights and activations are commonly stored in `bfloat16` or
`float16` and accumulated in `float32`. The reason is
{{eq:summation-error}} from {{ch:mle-reproducibility}}: a GEMM sums many
products, and accumulating that sum in 16-bit loses precision rapidly. `bfloat16`
is generally preferred over `float16` because it keeps `float32`'s exponent
range, so it overflows far less readily at the cost of mantissa bits.
{{part:15}} treats this properly.

**Batch size interacts with everything.** It sets activation memory, the
gradient's variance ({{ch:dl-optimizers}}), and hardware utilisation. Small
batches underuse the accelerator because the GEMM is too small to saturate it;
large batches need a larger learning rate and often warmup
({{ch:dl-lr-schedules}}).

**Determinism.** {{ch:mle-reproducibility}} measured a random forest losing
bitwise reproducibility at four threads. Neural network training on a GPU is
worse: non-deterministic reduction order and algorithm selection mean two runs
with identical seeds routinely differ. Frameworks expose deterministic modes at
a real throughput cost. Measure your run-to-run variance before believing any
improvement.

**The last layer's activation must match the loss.** Softmax paired with
cross-entropy is implemented as one fused operation for numerical reasons
derived in {{ch:dl-losses}}; applying softmax and then a separate log is a
common and genuinely damaging bug.

## 11. Common Mistakes

**Stacking linear layers and expecting depth.** {{eq:deep-linear-collapse}}: the
composition is affine, and the measurement shows agreement to machine
precision.

**Initialising all weights to the same value.** Every unit stays identical; the
measured layer has the capacity of one unit.

**Citing universal approximation as evidence that networks work.** It bounds
neither width, trainability, nor generalisation.

**Using a fully connected first layer on images.** 150 million parameters, and
it has assumed nothing about images.

**Forgetting the output activation.** A sigmoid on a regression output caps
predictions at 1; a missing softmax makes cross-entropy meaningless.

**Comparing architectures at matched layer counts rather than matched
parameters.** Then you are measuring size, not shape.

**Assuming more layers is better.** Beyond a point, without the machinery of
chapters 56–58, deeper networks train *worse* — and
{{cite:he2016resnet}} showed the degradation is in *training* error, so it is an
optimisation failure and not overfitting.

**Reading meaning into a single hidden unit.** The measured representation is
distributed; no unit tracked the radius that determined the label.

## 12. Failure Modes

Distinct from mistakes: these occur when the method is applied correctly.

**Dead network from bad initialisation scale.** Weights too small make
activations and gradients vanish through depth; too large and they saturate or
explode. Neither is a coding error, and {{ch:dl-initialization}} is the fix.

**Underfitting that looks like a bug.** A network with insufficient capacity or
an inappropriate architecture produces a stubbornly flat loss curve that is
easily mistaken for a broken gradient. The diagnostic is to try to overfit a
handful of examples deliberately: a correct implementation with enough capacity
can always drive the loss on twenty examples to near zero, and failure to do so
localises the problem to the implementation rather than the data.

**Plateau at the mean.** A regression network that outputs approximately the
target mean for every input has found the loss-minimising constant and stopped.
Usually the learning rate is too low, the inputs are unscaled, or the output
activation is squashing the range.

**Symmetric solutions in wide layers.** Even with random initialisation, units
can converge to near-duplicates, so effective capacity is lower than the
parameter count suggests. Measurable as the rank of the activation matrix, and
one of the things dropout and weight decay discourage.

**Silent extrapolation.** {{sec:5-formal-explanation}} noted universal
approximation says nothing off the compact set. A network given inputs outside
its training range produces confident nonsense, exactly as
{{ch:ml-trees}} measured for trees — and, as there, nothing in the output
signals it.

## 13. Alternatives

**Linear and logistic models** ({{part:4}}). Fewer parameters, convex
optimisation, interpretable coefficients. Correct whenever the relationship is
close to linear or data is scarce, and the honest first thing to try.

**Gradient boosting** ({{ch:ml-boosting}}). On tabular data this usually beats
an MLP, and the measured comparison in {{ch:ml-boosting}} shows why: trees are
robust to uninformative features and fit irregular functions, both of which
tabular data supplies. **Do not reach for an MLP on a spreadsheet by default.**

**Kernel methods** ({{ch:ml-svm}}). An infinite-dimensional feature space with
convex optimisation. Superseded by learned representations at scale, and
competitive when $N$ is small.

**Gaussian processes.** Calibrated uncertainty and $O(N^{3})$ cost. Where
uncertainty matters more than scale.

The honest summary: a neural network is the right choice when the input is
high-dimensional and structured — images, audio, text — and there is enough
data for a learned representation to beat a designed one. On low-dimensional
tabular data with a few thousand rows it is usually the wrong tool, and saying
so is not modesty.

## 14. Evaluation

Everything in {{ch:ml-metrics}} applies unchanged, plus three things specific to
networks.

**Overfit a tiny subset first.** Before any real training run, verify the
network can drive the loss to near zero on twenty examples. This separates
implementation bugs from learning problems, and it is the single most valuable
diagnostic in deep learning.

**Watch the training curve, not only the final number.** A loss that plateaus
immediately, oscillates, or diverges each indicates a different problem, and
the shape is diagnostic in a way the endpoint is not.

**Report the run-to-run spread.** {{ch:mle-reproducibility}} measured a
single-seed comparison reporting a confident difference that a paired
fifteen-seed comparison declined to resolve. Network training is noisier than
the forests measured there, so the discipline matters more.

For this chapter's material specifically, two additional checks: measure the
**rank of the hidden activation matrix** to detect units that have collapsed to
duplicates, and compare against a **linear baseline** — if the network does not
beat logistic regression, the nonlinearity is not earning its cost.

## 15. Advanced Concepts

**Depth separation.** The formal versions of {{sec:5-formal-explanation}}'s
claim exhibit specific function families — typically built from compositions of
sawtooth or oscillatory maps — requiring exponentially many units at depth
$k-1$ and polynomially many at depth $k$. The constructions are somewhat
artificial, which is the honest caveat: they prove depth *can* be exponentially
more efficient, not that it is on natural problems.

**The lottery ticket hypothesis.** A dense randomly-initialised network is
conjectured to contain a sparse subnetwork which, trained in isolation from the
same initialisation, matches the full network's accuracy. If true it suggests
overparameterisation's role is to supply many candidate subnetworks rather than
to provide capacity. {{maturity:EMERGING}} — reproduced in many settings and
sensitive to the pruning and rewinding procedure.

**Neural tangent kernel.** In the infinite-width limit with appropriate
scaling, gradient descent on a network behaves like kernel regression with a
fixed kernel determined by the architecture and initialisation. This makes the
dynamics analysable, and it is also a limitation: the regime it describes is one
in which **no feature learning occurs**, since the kernel does not change. Real
finite networks do learn features, so the NTK explains the wrong thing about
them — informative precisely where it breaks.

**Mixture of experts.** Route each input to a small subset of many parallel
subnetworks, so parameter count and per-token compute decouple. The dominant way
of scaling parameters in 2026 language models, and covered in {{part:10}}.

**Weight tying and equivariance.** Convolution ties weights across space and
recurrence ties them across time; both are instances of imposing a symmetry on
the hypothesis space. The general framework — designing architectures from the
symmetry group of the data — is the geometric deep learning programme, named
here and not developed.

## 16. Connection to Previous Chapters

{{ch:ml-logistic}} is this chapter's single-unit case: its sigmoid is an
activation, its linear predictor is the affine map, and
{{eq:logreg-gradient}} is the last layer's gradient in the network derived in
{{ch:dl-backprop}}. {{ch:ml-what-it-is}} supplied the hypothesis-space framing
that {{sec:6-mathematical-foundation}}'s XOR proof depends on, and the
measurement there that adding an interaction feature let a linear model solve
XOR is exactly what the hidden layer does automatically.
{{ch:ml-linear-regression}} supplied the basis-expansion idea; a hidden layer is
a *learned* basis expansion rather than a chosen one, which is the whole
difference. {{ch:ml-metrics}} supplied underfitting as a property of the
hypothesis space. {{ch:math-matrices}} supplied the shapes in
{{eq:mlp-batched}}.

Forward: {{ch:dl-activations}} examines the choice of $\phi$ and why sigmoid was
abandoned. {{ch:dl-forward}} formalises the computational graph implicit in
{{eq:mlp-forward}}. {{ch:dl-backprop}} derives the gradients this chapter
computed by hand for two layers. {{ch:dl-cnns}} replaces the dense layer whose
parameter count {{sec:9-practical-example}} measured as prohibitive.
{{ch:tf-scaled-dot-product}} is built from exactly these components arranged
differently.

## 17. Exercises

**Beginner**

1. Write the perceptron update rule and explain each factor.
2. Why can a perceptron not represent XOR? Answer in one sentence about
   hypothesis spaces.
3. What happens if you stack ten linear layers with no activation?
4. Why must weights be initialised randomly but biases need not be?
5. Count the parameters in a $10 \to 64 \to 64 \to 3$ MLP.

**Intermediate**

6. Using {{eq:perceptron-convergence}}, bound the mistakes for $R=5$,
   $\gamma=0.5$.
7. State the universal approximation theorem and three things it does not say.
8. Explain, using {{tbl:xor-hand}}, what the hidden layer did to the input
   space.
9. Using {{eq:mlp-flops}}, compute the forward FLOPs of a
   $784 \to 512 \to 512 \to 10$ network at batch 64.
10. Why is an elementwise activation memory-bound while a GEMM is
    compute-bound?
11. Give a case where an MLP is the wrong choice and say what you would use.

**Advanced**

12. Prove the perceptron convergence theorem, stating where separability is
    used.
13. Prove that XOR is not representable, and generalise the argument to parity
    on $n$ inputs.
14. Derive {{eq:deep-linear-collapse}} including the bias term.
15. Give the bump construction of {{sec:6-mathematical-foundation}} in two
    dimensions and count the units needed for a grid of resolution $k$.
16. Explain why the neural tangent kernel regime involves no feature learning,
    and why that limits what it can explain.

**Implementation**

17. Extend the perceptron to the pocket algorithm for non-separable data and
    compare against logistic regression.
18. Implement a three-layer MLP with explicit gradients and verify them against
    finite differences.
19. Measure the effective rank of the hidden activation matrix during training
    and see whether units collapse.
20. Reproduce the depth-versus-width experiment with a target composed of a
    different number of stages, and check whether the optimal depth tracks it.

**Reasoning**

21. A colleague proposes a 12-layer MLP for a 5,000-row tabular dataset. What
    do you say?
22. The field abandoned neural networks twice on the basis of correct
    observations. What does that suggest about current claims of fundamental
    limitations?

## 18. Interview Questions

**"Why do we need activation functions?"** — The expected answer is that
without them the composition collapses to a single affine map
({{eq:deep-linear-collapse}}). A stronger answer adds that the function class
collapses while the *parameterisation* and hence the optimisation dynamics do
not, which is why deep linear networks remain a useful theoretical object.

**"Can a neural network learn XOR? How?"** — Yes, with a hidden layer, and the
strong answer draws {{tbl:xor-hand}}: the hidden layer maps the four inputs to
three points in which the classes are linearly separable.

**"What does the universal approximation theorem tell us?"** — The trap is
stopping at "one hidden layer can approximate anything". The distinguishing
answer is the four caveats — no bound on width, no algorithm, nothing off the
compact set, nothing about generalisation.

**"Why deep rather than wide?"** — Compositional reuse, plus the honest
addition that the depth-separation constructions are somewhat artificial and
the real argument is empirical.

**"How many parameters does this architecture have, and how much memory will it
need to train?"** — A practical question that catches people out. Parameters
from {{eq:mlp-params}}; then the multiplier for optimiser state (Adam stores two
extra copies), then activation memory scaling with batch size — which usually
dominates.

**"Your network is not learning. Walk me through your debugging."** — Overfit
twenty examples first; check the loss curve's shape; check input scaling; check
the output activation matches the loss; check gradient norms per layer; compare
against a linear baseline. Naming the overfit-a-tiny-subset test first is what
signals experience.

## 19. Research Questions

**Why does depth help on natural data?** The depth-separation theorems exhibit
functions where it must, and they do not establish that real problems have that
structure. The empirical evidence is overwhelming and the explanation is not
settled. {{maturity:RESEARCH FRONTIER}}

**What is the right notion of capacity?** Parameter counting fails —
{{cite:zhang2017rethinking}} showed networks fitting random labels — and no
replacement predicts generalisation reliably across architectures.
{{maturity:RESEARCH FRONTIER}}

**Is overparameterisation necessary?** The lottery ticket hypothesis suggests
the trained function may be representable far more compactly than the trained
network, which would make overparameterisation a property of the search rather
than of the solution. {{maturity:EMERGING}}

**What do individual units represent?** The measured representation here was
distributed, with no unit tracking the label-determining quantity. Whether
useful monosemantic units can be recovered by a change of basis — the
superposition hypothesis and sparse autoencoders — is active work.
{{maturity:EMERGING}}, and {{ch:rai-interpretability}} returns to it.

## 20. Chapter Summary

A unit is an affine map followed by a nonlinearity; a layer is many units in
parallel; a network is layers composed. A logistic regression is the one-unit
case, so this part begins exactly where {{ch:ml-logistic}} ended.

The perceptron {{cite:rosenblatt1958}} came with a convergence guarantee that
is genuinely strong — the number of mistakes is bounded by $(R/\gamma)^{2}$,
independent of the number of examples and of the dimension, as the measurement
confirms. It is also conditional on linear separability, and XOR is the
counterexample. The four-line proof in {{sec:6-mathematical-foundation}} and the
exhaustive grid search in {{sec:8-implementation}} both show that no perceptron
achieves better than 75% on XOR: **the limitation is representational, not a
training failure.**

Stacking linear layers gains nothing. The measured three-layer linear network
agrees with its equivalent single layer to machine precision, and
{{eq:deep-linear-collapse}} says it must. The nonlinearity is what makes depth
mean anything.

A hidden layer solves XOR by changing coordinates. The hand-built network maps
four inputs to three points in which the two positive cases coincide and the
classes are linearly separable — and that is the general account of what every
hidden layer does.

Universal approximation {{cite:cybenko1989}} says one hidden layer suffices in
principle, and says nothing about the required width, whether the parameters can
be found, behaviour outside the training region, or generalisation. It is
reassurance that the hypothesis space is not the bottleneck, and it is not an
argument that networks work.

Depth is preferred in practice because compositional structure can be built once
and reused. The measurement shows deeper networks winning at matched parameter
count on a target defined by composition and gaining little on an additive one —
which is the mechanism, and also the honest limit of what such an experiment can
establish.

Weights must be initialised randomly: with identical initial values every unit
in a layer computes the same function, receives the same gradient, and remains
identical, so a wide layer has the capacity of one unit.

Finally, the representation is distributed. In the measured ring problem no
individual hidden unit correlated with the radius that determined the label; the
information lives in the pattern across units. That is the general case, and it
is why interpreting single neurons is hard.

## 21. Further Reading

**The primary sources**, in the order they matter for this chapter:

{{cite:rosenblatt1958}} for the perceptron and its rule. Worth reading for the
framing as much as the algorithm — it is explicit about the model being a
hypothesis about biological learning, which the field later dropped.

{{cite:cybenko1989}} for universal approximation. Short, and the proof
technique (Hahn–Banach plus Riesz representation) is instructive even if you
skip the details.

{{cite:rumelhart1986}} for backpropagation's popularisation. Four pages, and
the clarity is striking given how much followed from it. Read it before
{{ch:dl-backprop}}.

{{cite:hinton2006}} for the layer-wise pre-training that ended the second
winter. Historically important and methodologically superseded, which makes it a
good example of a paper whose contribution was to prove something possible.

{{cite:krizhevsky2012}} for the demonstration that ended the argument. Notice
how much of it is engineering — data augmentation, GPU implementation, dropout,
ReLU — and how little is architectural novelty.

**Where to go next in this book:** {{ch:dl-activations}} immediately, since the
choice of $\phi$ is the first thing this chapter left open. Readers who want the
optimisation story before the architecture story can read
{{ch:dl-backprop}} directly after this chapter; it depends on
{{ch:dl-forward}} only for notation.
