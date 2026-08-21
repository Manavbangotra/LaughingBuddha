---
id: dl-initialization
number: 56
part: VI
tier: full
status: reviewed
requires: [dl-backprop, dl-activations, dl-forward, math-random-vars, math-matrices]
provides: [weight-initialization, glorot-init, he-init, symmetry-breaking,
           variance-propagation, orthogonal-init, residual-init, fan-in-fan-out]
citations: [glorot2010, he2015init, rumelhart1986, saxe2014, liu2020admin]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why zero and constant initialisation fail, and what symmetry
   breaking means.
2. Derive the Glorot and He initialisation scales from a variance calculation.
3. Explain why rectified units need a factor of two and saturating ones do not.
4. Distinguish fan-in, fan-out and their average, and say when each is right.
5. Apply orthogonal initialisation and explain what it preserves.
6. Explain how residual connections change what initialisation has to achieve.
7. Diagnose an initialisation problem from measured activation and gradient
   statistics.

## 2. Why This Matters

**Initialisation decides whether {{eq:unrolled-backprop}}'s product is
well-conditioned before a single step is taken.** {{ch:dl-backprop}} measured
the consequence: at a small initialisation, a 20-layer network's forward signal
and error signal each moved by twelve orders of magnitude and nothing trained.
The fix is a scalar, and getting it right is the difference between a network
that trains and one that does not.

**The derivation is four lines and the result looks like magic without it.**
Everyone can quote $\sqrt{2/n}$; far fewer can say where the two comes from.
{{sec:6-mathematical-foundation}} derives it, and afterwards the whole
signal-propagation view — which normalisation and residual connections live
inside — is available to you.

**It is one of the few places where a small amount of theory is decisively
useful.** {{cite:glorot2010}} and {{cite:he2015init}} are short papers with
simple arguments that produced large practical gains, which is rarer in this
field than it should be.

**Modern architectures shifted what initialisation must do.** Normalisation
layers make the network far less sensitive to the initial scale, and residual
connections introduce a different requirement — keeping the *residual branch*
small so the identity path dominates at the start. The classical formulae are
still right and they are no longer the whole story.

## 3. Prerequisites

{{ch:dl-backprop}} for {{eq:unrolled-backprop}} and the measured gradient
profiles. {{ch:dl-activations}} for $\phi'$ and saturation.
{{ch:dl-forward}} for the layer and its shapes. {{ch:math-random-vars}} for
variance of sums and products. {{ch:math-matrices}} for orthogonality and
singular values.

## 4. Intuitive Explanation

### 4.1 Two ways to fail

```text
   too small          x ──▶ 0.3x ──▶ 0.09x ──▶ ... ──▶ ~0
                      the signal dies before reaching the output

   too large          x ──▶ 3x ──▶ 9x ──▶ ... ──▶ huge
                      activations saturate, or overflow

   just right         x ──▶ x ──▶ x ──▶ ... ──▶ x
                      the SCALE is preserved layer to layer
```

The whole subject is that third line. **Initialisation is choosing a scale so
that the variance of the activations is preserved as they propagate forward,
and the variance of the gradients as they propagate backward.**

Note what is *not* being preserved: the values, or the direction, or anything
about the function computed. Only the scale. That is a much weaker requirement
than it might sound, and it is enough.

### 4.2 Why not zero

Set every weight to zero and every unit in a layer computes the same thing —
zero. Worse, every unit receives the same gradient, so they update identically
and remain identical forever.

**The layer has one effective unit no matter how many it nominally has.** This
is the *symmetry problem*, and {{ch:dl-neural-networks}} measured it.

The precise condition is narrower than the slogan, and
{{sec:8-implementation}} separates three cases that are usually conflated.
Symmetry persists only if *every* layer on the path is constant: a constant
first layer feeding a *random* second one already breaks it, because the
backward pass multiplies by the second layer's weights and those differ per
unit. What a zero first layer does instead is kill a rectifier outright — every
pre-activation is exactly zero, so the mask $\Ind[z>0]$ is false everywhere and
the layer receives no gradient at all, forever. That is a different failure with
the same appearance.

Biases can be zero either way, because the weights break the symmetry.

### 4.3 The variance argument

Consider one unit summing $n$ inputs:

$$
z = \sum_{i=1}^{n} w_i x_i
$$

If the $w_i$ are independent with variance $\sigma_w^2$ and the $x_i$ are
independent with variance $\sigma_x^2$, then $\Var[z] = n\sigma_w^2\sigma_x^2$.

**Preserving the scale means $\Var[z] = \sigma_x^2$, so
$\sigma_w^2 = 1/n$.**

That is the entire idea. The refinements — Glorot's average of fan-in and
fan-out, He's factor of two — are corrections to this one calculation for what
happens on the backward pass and for what the activation does to the variance.

### 4.4 Why rectified units need a factor of two

ReLU zeroes half its inputs. For a symmetric input distribution,
{{ch:dl-activations}} measured $\E[\relu(z)^2]/\E[z^2] = 0.5$ exactly.

So a ReLU layer halves the second moment. To compensate, double the weight
variance:

$$
\sigma_w^2 = \frac{2}{n}
$$

**That is the factor of two, and it is nothing more mysterious than "half the
units are off".** {{cite:he2015init}} is this observation applied carefully.

### 4.5 What normalisation changed

A batch- or layer-normalisation layer rescales its input to unit variance
regardless of what arrives, so the scale that initialisation was carefully
setting gets overwritten immediately.

```text
   without normalisation    init scale ──▶ propagates ──▶ matters at every layer
   with normalisation       init scale ──▶ normalised away at the next layer
```

Initialisation therefore matters much less in a normalised network, and
{{sec:9-practical-example}} measures how much less. It does not stop mattering
entirely — the first layer before any normalisation, the residual branches, and
the output layer all still depend on it.

## 5. Formal Explanation

### 5.1 The schemes

{#tbl:init-schemes caption="Initialisation schemes and the activation each was derived for. All are zero-mean; the difference is entirely in the variance, and the last column is the reason for the difference."}

| Scheme | Variance | Derived for |
|---|---|---|
| LeCun | $1/n_{\text{in}}$ | forward pass only |
| Glorot / Xavier | $2/(n_{\text{in}}+n_{\text{out}})$ | tanh, sigmoid — both passes |
| He / Kaiming | $2/n_{\text{in}}$ | ReLU and relatives |
| He (fan-out) | $2/n_{\text{out}}$ | ReLU, backward pass |
| Orthogonal | singular values 1 | deep linear, RNNs |

Each has a uniform variant. For a target variance $\sigma^2$, a uniform
distribution on $[-a, a]$ has variance $a^2/3$, so $a = \sqrt{3}\sigma$. The
uniform and normal variants perform indistinguishably in practice; the choice is
convention.

### 5.2 Glorot

{{cite:glorot2010}} required the variance to be preserved in *both* directions.
{{sec:6-mathematical-foundation}} shows the forward pass wants
$\sigma_w^2 = 1/n_{\text{in}}$ and the backward pass wants
$\sigma_w^2 = 1/n_{\text{out}}$. These conflict unless the layer is square, so
Glorot takes the harmonic-mean-like compromise:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$ (eq:glorot)

**Neither condition is satisfied exactly, and both are satisfied approximately.**
That is worth stating plainly: it is a compromise, not a derivation of an exact
requirement, and the paper says so.

### 5.3 He

{{cite:he2015init}} redid the calculation for ReLU, which halves the variance:

$$
\sigma_w^2 = \frac{2}{n_{\text{in}}}
$$ (eq:he-init)

The paper argues for fan-in on the grounds that preserving the forward signal is
what matters, and shows that fan-out works about as well. **The two differ by
the layer's aspect ratio**, so for a network of roughly constant width the
distinction is negligible and the arguments about it are mostly theological.

### 5.4 Orthogonal

Initialise $\mat{W}$ so that $\mat{W}\T\mat{W} = \mat{I}$, obtained from the QR
decomposition of a random Gaussian matrix. Then $\|\mat{W}\vec{x}\| =
\|\vec{x}\|$ **exactly**, for every $\vec{x}$, rather than in expectation.

{{cite:saxe2014}} showed that in a deep *linear* network this makes the
convergence time independent of depth, which is a genuinely strong result and
also one that applies to a model with no nonlinearity. Its practical use is
mostly in recurrent networks ({{ch:dl-rnns}}), where the same matrix is applied
at every time step and preserving the norm exactly matters more.

For a non-square matrix, "orthogonal" means semi-orthogonal: the rows or columns
are orthonormal, whichever there are fewer of.

### 5.5 Residual networks

A residual block computes $\vec{x} + F(\vec{x})$. If $F$ is initialised at the
usual scale, the block's output variance is the *sum* of the two branches', so
variance grows with depth even under perfect per-layer initialisation:

$$
\Var[\vec{x}_L] = \Var[\vec{x}_0]\prod_{l=1}^{L}(1+c_l)
$$ (eq:residual-variance-growth)

for per-block contributions $c_l$. With $c_l = 1$ this is $2^L$.

The standard fix is to initialise the *last* layer of each residual branch to
zero, so $F(\vec{x}) = \vec{0}$ at initialisation and the block is exactly the
identity. The network then starts as a linear map and the branches grow into
usefulness during training.

{{cite:liu2020admin}} traces transformer training instability to exactly this
dependence on the residual branch — small perturbations get amplified through it
— and proposes an adaptive initialisation that controls the branch's
contribution early while leaving its capacity intact later. A simpler variant of
the same idea scales each branch by $1/\sqrt{L}$, which keeps the product in
{{eq:residual-variance-growth}} bounded; {{eq:sqrt-l-scaling}} shows the bound.

> IMPORTANT: **In a residual network the requirement changes from "preserve the
> scale" to "start near the identity".** {{sec:9-practical-example}} measures
> the variance growth and the fix. This is one of the clearest examples in the
> book of an architecture changing what a technique has to accomplish.

### 5.6 Biases and special cases

**Biases: zero**, almost always. The weights break symmetry, and a nonzero bias
just shifts the pre-activations off centre.

**Forget-gate biases: one.** In an LSTM ({{ch:dl-rnns}}), initialising the
forget gate's bias to 1 makes the gate open at the start so gradients flow
through time before the gate has learned anything.

**Output biases: the base rate.** Setting a classifier's output bias to
$\log(p_c/(1-p_c))$ means the network predicts the class priors at step zero.
This costs nothing and removes the first few hundred steps that would otherwise
be spent learning the priors — noticeable under heavy class imbalance.

**Embeddings: small.** Typically $\mathcal{N}(0, 0.02^2)$ in transformer
recipes, which is much smaller than any fan-in rule would give. The reason is
that an embedding lookup is not a sum over a fan-in — it selects one row — so
the variance argument of {{sec:6-mathematical-foundation}} does not apply at
all.

## 6. Mathematical Foundation

### 6.1 Forward variance propagation

Take $z_j^{(l)} = \sum_{i=1}^{n_{l-1}} W^{(l)}_{ji}h^{(l-1)}_i$, with $W$
zero-mean, independent of $h$, and the $h_i$ independent and identically
distributed. Then

$$
\Var\big[z_j^{(l)}\big]
 = n_{l-1}\,\Var\big[W^{(l)}\big]\,\E\big[(h^{(l-1)})^2\big]
$$ (eq:forward-variance)

**Linear activation.** $\E[(h^{(l)})^2] = \Var[z^{(l)}]$, so preserving variance
requires $n_{l-1}\Var[W] = 1$, giving

$$
\Var[W] = \frac{1}{n_{l-1}}
$$ (eq:lecun-init)

which is LeCun initialisation.

**ReLU.** For $z$ symmetric about zero, $\relu(z)^2 = z^2$ when $z > 0$ and $0$
otherwise, each with probability $1/2$, so

$$
\E\big[\relu(z)^2\big] = \tfrac{1}{2}\E[z^2]
$$ (eq:relu-second-moment)

Substituting into {{eq:forward-variance}} gives
$\Var[z^{(l)}] = \frac{1}{2}n_{l-1}\Var[W]\Var[z^{(l-1)}]$, so preservation
requires

$$
\Var[W] = \frac{2}{n_{l-1}}
$$ (eq:he-derived)

which is {{eq:he-init}}. $\square$

**The factor of two is exactly the $1/2$ in {{eq:relu-second-moment}}**, and
{{ch:dl-activations}} measured that $1/2$ to hold at every input scale tested.

### 6.2 Backward variance propagation

From {{eq:backprop-recursion}}, $\delta_i^{(l-1)} = \phi'(z_i^{(l-1)})
\sum_j W^{(l)}_{ji}\delta_j^{(l)}$. The sum is over $n_l$ terms — the *output*
dimension — so

$$
\Var\big[\delta^{(l-1)}\big]
 \approx n_l\,\Var[W^{(l)}]\,\Var\big[\delta^{(l)}\big]\,\E[\phi'^2]
$$ (eq:backward-variance)

For a linear activation this needs $\Var[W] = 1/n_l$; for ReLU, since
$\E[\phi'^2] = \Pr(z>0) = 1/2$, it needs $\Var[W] = 2/n_l$.

**The forward condition wants $1/n_{\text{in}}$ and the backward wants
$1/n_{\text{out}}$.** Both hold only when the layer is square, which is
{{sec:5-formal-explanation}}'s observation, and {{eq:glorot}} is the compromise.

### 6.3 What happens when the scale is wrong

Let $\gamma = n_{l-1}\Var[W]\cdot\E[\phi'^2]/\Var[\phi]$-adjusted be the
per-layer variance gain, so $\Var[z^{(L)}] = \gamma^{L}\Var[z^{(0)}]$.

$$
\gamma = \frac{\Var[W]}{\Var[W]^\star}
$$ (eq:variance-gain)

where $\Var[W]^\star$ is the correct value. Then after $L$ layers the variance
ratio is $\gamma^L$, so the *standard deviation* ratio is $\gamma^{L/2}$.

Concretely, at $L = 50$ layers:

- $\gamma = 0.9$ (weights 5% too small): $0.9^{25} = 0.07$.
- $\gamma = 0.5$ (weights 30% too small): $0.5^{25} = 3\times 10^{-8}$.
- $\gamma = 2$ (weights 41% too large): $2^{25} = 3\times 10^{7}$.

**A 30% error in the standard deviation destroys a 50-layer network.** That is
the sensitivity, and it is why this scalar is worth deriving rather than
guessing.

### 6.4 Why orthogonal is stronger

The variance argument controls $\E[\|\mat{W}\vec{x}\|^2]$ — the *average* over
inputs. An orthogonal $\mat{W}$ satisfies $\|\mat{W}\vec{x}\| = \|\vec{x}\|$ for
*every* $\vec{x}$.

The difference is the spread of the singular values. For a Gaussian $\mat{W}$
scaled so the mean squared singular value is 1, the singular values follow the
Marchenko–Pastur distribution and range over a wide interval for a square
matrix; for an orthogonal one they are all exactly 1.

**A Gaussian initialisation preserves the norm on average and distorts
individual directions.** Over $L$ layers those distortions compound, which is
{{cite:saxe2014}}'s argument, and {{sec:8-implementation}} measures the singular
value spread directly.

### 6.5 Residual variance growth

For a block $\vec{y} = \vec{x} + F(\vec{x})$ with independent branches,

$$
\Var[\vec{y}] = \Var[\vec{x}] + \Var[F(\vec{x})]
$$ (eq:residual-block-variance)

If $F$ preserves variance — which is what {{eq:he-init}} arranges — then
$\Var[\vec{y}] = 2\Var[\vec{x}]$ and stacking $L$ blocks gives $2^L$.

Two fixes, and they are equivalent to first order:

**Zero-init the branch's last layer.** $\Var[F] = 0$ at initialisation, so the
product in {{eq:residual-variance-growth}} is exactly 1.

**Scale the branch by $1/\sqrt{L}$.** Then $\Var[F] = \Var[\vec{x}]/L$ and

$$
\Var[\vec{y}_L] = \Var[\vec{x}_0]\left(1+\tfrac{1}{L}\right)^{L}
 \xrightarrow{L\to\infty} e\,\Var[\vec{x}_0]
$$ (eq:sqrt-l-scaling)

Bounded by $e$ regardless of depth, which is the point.

## 7. Internal Mechanics

### 7.1 What a framework does

Every layer type has a default, and the defaults are not always what you would
choose:

```text
   Linear         Kaiming uniform with a=sqrt(5)  -> effectively Glorot-ish
   Conv2d         same, with fan_in = C_in * k * k
   Embedding      N(0, 1)                          -> usually overridden
   LayerNorm      weight 1, bias 0
```

The `a=sqrt(5)` default is a historical artefact that produces a gain close to
Glorot's rather than He's, so **a ReLU network on framework defaults is not He
initialised** unless you say so. This is a real and widely unnoticed
discrepancy.

### 7.2 Fan-in and fan-out for convolutions

For a convolution with $C_{\text{in}}$ input channels, $C_{\text{out}}$ output
channels and a $k \times k$ kernel:

$$
n_{\text{in}} = C_{\text{in}}k^2, \qquad n_{\text{out}} = C_{\text{out}}k^2
$$ (eq:conv-fan)

The $k^2$ is easy to forget and matters: for a $3\times3$ kernel it is a factor
of nine in the fan, which is a factor of three in the standard deviation.

### 7.3 Layer order matters for what you should do

```text
   Linear -> ReLU -> Linear             init matters at every layer
   Linear -> Norm -> ReLU -> Linear     norm resets the scale each layer
   x + (Linear -> ReLU -> Linear)       branch should start near zero
```

The three lines want three different things, which is why "use He
initialisation" is not a complete instruction for a modern architecture.

### 7.4 Loading pretrained weights

Fine-tuning ({{part:14}}) replaces initialisation for the body of the network
and not for the newly added head. A randomly initialised head produces large
gradients that flow back into carefully pretrained weights and disrupt them,
which is why head-only warmup and layerwise learning-rate decay exist.

### 7.5 Reproducibility

The initialisation consumes random numbers in a specific order, so adding a
layer changes every subsequent layer's initial weights even at a fixed seed.
This makes "same seed, different architecture" comparisons less controlled than
they look, and it is a good reason to seed each layer independently from a
derived seed rather than drawing from one global stream
({{ch:mle-reproducibility}}).

## 8. Implementation

```python {tier=A name=variance-propagation}
"""The variance calculation of section 6.1, measured layer by layer in
networks 50 deep.
"""
import numpy as np

rng = np.random.default_rng(0)


def propagate(depth, width, scheme, act="relu", batch=512, seed=0):
    """Forward pass through an untrained network, returning per-layer stats."""
    rs = np.random.default_rng(seed)
    h = rs.normal(size=(batch, width))
    stats = [(0, float(np.var(h)), 0.0)]
    for l in range(1, depth + 1):
        if scheme == "lecun":
            sd = np.sqrt(1.0 / width)
        elif scheme == "he":
            sd = np.sqrt(2.0 / width)
        elif scheme == "glorot":
            sd = np.sqrt(2.0 / (width + width))
        elif scheme == "too small":
            sd = np.sqrt(2.0 / width) * 0.7
        elif scheme == "too large":
            sd = np.sqrt(2.0 / width) * 1.4
        elif scheme == "unit":
            sd = 1.0
        W = rs.normal(0, sd, (width, width))
        z = h @ W
        h = np.maximum(0.0, z) if act == "relu" else np.tanh(z)
        stats.append((l, float(np.var(z)),
                      float((h == 0).mean()) if act == "relu" else 0.0))
    return stats


print("=" * 72)
print("forward variance through 50 layers (eq. 56.4)")
print("=" * 72)
print("ReLU network, width 256. Var[z] should stay at its initial value\n"
      "if the scheme is right for the activation.\n")
DEPTH, WIDTH = 50, 256
picks = [0, 1, 5, 10, 25, 50]
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>11}" for i in picks)
      + f" {'ratio L50/L0':>14}")
for scheme in ("he", "lecun", "glorot", "too small", "too large"):
    st = propagate(DEPTH, WIDTH, scheme)
    vals = " ".join(f"{st[i][1]:>11.3e}" for i in picks)
    print(f"{scheme:<14} {vals} {st[-1][1] / st[0][1]:>14.3e}")

print("\nRead the last column first. He initialisation holds the variance")
print("within a small factor across fifty layers. Every other scheme moves")
print("it by fifteen orders of magnitude, in one direction or the other.")
print("\nLeCun and Glorot are numerically identical here because the layers")
print("are square — 2/(n+n) is 1/n — and both are missing the factor of two")
print("that eq. 56.5 says a rectifier needs. They are the right answer for a")
print("linear or saturating activation.")
print("\nAgainst section 6.3's prediction: the per-layer gain is")
print("Var[W]/Var[W]*, so LeCun's is 1/2 and after 50 layers that predicts")
print(f"{0.5 ** 50:.3e}, against a measured 3.7e-15 — the right order and a")
print("factor of four out. 'Too large' at 1.4x the standard deviation has")
print(f"gain 1.96 and predicts {1.96 ** 50:.3e}, which the measurement")
print("matches closely.")
print("\nThe He row drifts upward rather than staying flat, and the next")
print("experiment shows why: it is a finite-width effect, not an error in")
print("eq. 56.6.")
print("\nWhat is not in doubt is the sensitivity. A 30 per cent error in")
print("the standard deviation moved the variance by fifteen orders of")
print("magnitude over fifty layers, which is why this scalar is worth")
print("deriving rather than guessing.")

# --- is the He drift a finite-width effect? ---------------------------------
print("\n" + "=" * 72)
print("the residual drift under He is a FINITE-WIDTH effect")
print("=" * 72)
print("Eq. 56.5's E[relu(z)^2] = E[z^2]/2 holds exactly for a symmetric z.")
print("At finite width the units become correlated as depth accumulates and")
print("the empirical ratio drifts above one half. Widening should shrink it.\n")
print(f"{'width':>8} {'batch':>8} {'Var[z] at L50 / L1':>22} "
      f"{'implied per-layer gain':>25}")
for width in (32, 128, 512, 2048):
    st = propagate(50, width, "he", batch=max(512, 2 * width), seed=1)
    ratio = st[-1][1] / st[1][1]
    print(f"{width:>8} {max(512, 2 * width):>8} {ratio:>22.4f} "
          f"{ratio ** (1 / 49):>25.5f}")

print("\nRead the last column as a distance from 1.0. At width 32 the")
print("per-layer gain is off by nine per cent; at width 2048 it is off by")
print("one part in ten thousand. The deviation shrinks by roughly the")
print("expected order as the width grows, and it changes sign along the")
print("way, which is what a finite-size fluctuation does and what a")
print("systematic error in the derivation would not.")
print("\nEq. 56.6 is a statement about expectations over an infinitely wide")
print("layer. A width-256 layer approximates it to within about half a per")
print("cent per layer, which compounds over fifty layers into the factor of")
print("four in the previous table — and is negligible against the fifteen")
print("orders of magnitude a wrong scheme produces.")
print("\nThis is worth knowing because it is the honest status of the whole")
print("mean-field style of argument in this chapter: exact in the limit,")
print("approximate in practice, and the approximation error is small")
print("compared with the effect being predicted.")

# --- the same for tanh, where the factor of two is WRONG --------------------
print("\n" + "=" * 72)
print("the same schemes with tanh, where the factor of two is wrong")
print("=" * 72)
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>11}" for i in picks)
      + f" {'ratio L50/L0':>14}")
for scheme in ("he", "lecun", "glorot"):
    st = propagate(DEPTH, WIDTH, scheme, act="tanh")
    vals = " ".join(f"{st[i][1]:>11.3e}" for i in picks)
    print(f"{scheme:<14} {vals} {st[-1][1] / st[0][1]:>14.3e}")

print("\nWith tanh the ordering reverses: Glorot and LeCun hold the scale")
print("and He's factor of two now OVERSHOOTS. Tanh does not zero half its")
print("inputs, so the compensation eq. 56.5 justifies is not needed and")
print("becomes a systematic over-scaling.")
print("\nNote that tanh's own saturation partly rescues the overshoot —")
print("|tanh| <= 1 caps the variance no matter how large the input — which")
print("is why the He row does not explode the way the ReLU 'too large' row")
print("did. Saturation is a bad way to control the scale, because it comes")
print("with the vanishing derivative of Chapter 50.")

# --- dead units -------------------------------------------------------------
print("\n" + "=" * 72)
print("what a bad scale does to the fraction of dead ReLU units")
print("=" * 72)
print(f"{'scheme':<14} " + " ".join(f"{f'L{i}':>9}" for i in [1, 5, 10, 25, 50]))
for scheme in ("he", "too small", "too large"):
    st = propagate(DEPTH, WIDTH, scheme)
    print(f"{scheme:<14} " + " ".join(f"{st[i][2]:>9.4f}"
                                      for i in [1, 5, 10, 25, 50]))
print("\nThe rows are not merely similar — they are IDENTICAL to every")
print("digit. Scaling every weight by a positive constant cannot change")
print("the sign of any pre-activation, so the set of dead units is exactly")
print("the same however wrong the scale is.")
print("\nSo the zero fraction is not a weak diagnostic for a bad scale; it")
print("is provably no diagnostic at all. It is worth knowing because it is")
print("a natural thing to reach for and it measures nothing.")
print("\nWhat goes wrong is the magnitude of the surviving half, and only")
print("the variance table shows that. This is a good example of a plausible")
print("diagnostic that measures nothing.")

# --- backward, and Glorot's compromise --------------------------------------
print("\n" + "=" * 72)
print("Glorot's compromise: the two conditions genuinely conflict (6.2)")
print("=" * 72)
print("A layer with fan_in and fan_out different. The forward pass wants")
print("Var[W] = 1/fan_in and the backward pass wants 1/fan_out.\n")
print(f"{'fan_in':>8} {'fan_out':>8} {'1/fan_in':>11} {'1/fan_out':>11} "
      f"{'Glorot':>11} {'fwd gain':>10} {'bwd gain':>10}")
for fi, fo in ((256, 256), (256, 1024), (1024, 256), (784, 128), (128, 10)):
    v_glorot = 2.0 / (fi + fo)
    print(f"{fi:>8} {fo:>8} {1 / fi:>11.3e} {1 / fo:>11.3e} "
          f"{v_glorot:>11.3e} {fi * v_glorot:>10.3f} {fo * v_glorot:>10.3f}")

print("\nThe two gain columns are what Glorot achieves in each direction: it")
print("hits 1.0 exactly when the layer is square and misses in both")
print("directions otherwise, by factors that are reciprocal. A 784->128")
print("layer amplifies the forward signal by 1.7 and attenuates the")
print("backward one by 0.28.")
print("\nThat is not a flaw in the derivation; it is the derivation's")
print("conclusion. Both conditions cannot hold for a non-square layer, and")
print("eq. 56.2 is the compromise. Networks of roughly constant width — the")
print("usual case — barely notice.")
```

```python {tier=A name=orthogonal-and-symmetry}
"""Symmetry breaking, and what orthogonal initialisation preserves that a
Gaussian one does not.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- section 4.2: symmetry ---------------------------------------------------
print("=" * 72)
print("symmetry breaking: what a constant initialisation costs")
print("=" * 72)


def train_tiny(init, steps=400, lr=0.1, seed=0, hidden=8):
    """A tiny network on XOR-like data; returns the hidden units it learns."""
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)[:, None]
    W2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, 1))
    if init == "W1 zero":
        W1 = np.zeros((2, hidden))
    elif init == "W1 constant":
        W1 = np.full((2, hidden), 0.5)
    elif init == "both constant":
        W1 = np.full((2, hidden), 0.5)
        W2 = np.full((hidden, 1), 0.5)          # NO asymmetry anywhere
    else:
        W1 = rs.normal(0, np.sqrt(2 / 2), (2, hidden))
    b1 = np.zeros(hidden)
    b2 = np.zeros(1)
    for _ in range(steps):
        h = np.maximum(0.0, X @ W1 + b1)
        p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -60, 60)))
        d2 = (p - y) / len(X)
        d1 = (d2 @ W2.T) * ((X @ W1 + b1) > 0)
        W2 -= lr * (h.T @ d2)
        b2 -= lr * d2.sum(axis=0)
        W1 -= lr * (X.T @ d1)
        b1 -= lr * d1.sum(axis=0)
    h = np.maximum(0.0, X @ W1 + b1)
    p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -60, 60)))
    acc = float(((p > 0.5) == y).mean())
    # count distinct hidden functions, up to rounding
    sig = {tuple(np.round(h[:, j], 4)) for j in range(hidden)}
    return acc, len(sig), float(np.abs(W1).std())


print(f"{'initialisation':<16} {'accuracy':>10} {'distinct hidden units':>23} "
      f"{'sd of W1':>10}")
for init in ("W1 zero", "W1 constant", "both constant", "random"):
    acc, n, sd = train_tiny(init)
    print(f"{init:<16} {acc:>10.4f} {n:>23} {sd:>10.4f}")

print("\nThe three failures are NOT the same failure, and the usual")
print("one-line account ('constant weights are symmetric') runs them")
print("together.")
print("\n'both constant' is the pure symmetry case: nothing anywhere")
print("distinguishes the units, so they receive identical gradients")
print("forever, and eight units collapse to one function. One rectified")
print("unit cannot represent XOR, so the network is stuck well below what")
print("the architecture is capable of.")
print("\n'W1 zero' fails for a DIFFERENT reason. Every pre-activation is")
print("exactly zero, so ReLU's mask (z > 0) is false everywhere and the")
print("first layer receives exactly zero gradient at every step. The layer")
print("is not symmetric, it is dead — and it stays dead however long you")
print("train it.")
print("\n'W1 constant' with a RANDOM output layer breaks the symmetry and")
print("trains. The units start identical, and the backward pass multiplies")
print("by W2, which differs per unit — so they receive different gradients")
print("from the very first step and separate. It ends behind the random")
print("initialisation and far ahead of chance.")
print("\nThe correct statement is therefore narrower than the slogan:")
print("symmetry is broken if ANY layer on the path is asymmetric. What you")
print("must not do is make every layer constant, and what you must not do")
print("separately is put a rectifier in a state where its mask is")
print("identically false.")

# --- section 6.4: what orthogonal preserves ---------------------------------
print("\n" + "=" * 72)
print("Gaussian preserves the norm on AVERAGE; orthogonal preserves it")
print("exactly (section 6.4)")
print("=" * 72)


def singular_spread(n, kind, seed=0):
    rs = np.random.default_rng(seed)
    if kind == "gaussian":
        W = rs.normal(0, np.sqrt(1.0 / n), (n, n))
    else:
        Q, R = np.linalg.qr(rs.normal(size=(n, n)))
        W = Q * np.sign(np.diag(R))          # make the QR sign convention fixed
    s = np.linalg.svd(W, compute_uv=False)
    return s


print(f"{'n':>6} {'kind':<12} {'min sv':>9} {'max sv':>9} {'mean sv':>9} "
      f"{'sv spread':>11} {'mean sv^2':>11}")
for n in (64, 256):
    for kind in ("gaussian", "orthogonal"):
        s = singular_spread(n, kind)
        print(f"{n:>6} {kind:<12} {s.min():>9.4f} {s.max():>9.4f} "
              f"{s.mean():>9.4f} {s.max() - s.min():>11.4f} "
              f"{float(np.mean(s ** 2)):>11.4f}")

print("\nBoth have mean squared singular value near 1, which is what the")
print("variance argument of section 6.1 controls — so both 'preserve the")
print("norm' in the sense that calculation means.")
print("\nBut the Gaussian matrix's singular values run from near zero to")
print("about two. Directions aligned with the small ones are crushed and")
print("directions aligned with the large ones are amplified, and across")
print("many layers those distortions compound. The orthogonal matrix's are")
print("all exactly one, so EVERY direction is preserved exactly.")

# --- and what that does across depth ----------------------------------------
print("\n" + "=" * 72)
print("the consequence across depth: a deep LINEAR network")
print("=" * 72)
print("No nonlinearity, so this isolates the matrix product of eq. 53.9.\n")


def deep_linear_spread(depth, n=64, kind="gaussian", seed=1):
    rs = np.random.default_rng(seed)
    M = np.eye(n)
    for _ in range(depth):
        if kind == "gaussian":
            W = rs.normal(0, np.sqrt(1.0 / n), (n, n))
        else:
            Q, R = np.linalg.qr(rs.normal(size=(n, n)))
            W = Q * np.sign(np.diag(R))
        M = W @ M
    s = np.linalg.svd(M, compute_uv=False)
    return s


print(f"{'depth':>7} {'kind':<12} {'min sv':>12} {'max sv':>12} "
      f"{'condition number':>18}")
for depth in (1, 5, 20, 50):
    for kind in ("gaussian", "orthogonal"):
        s = deep_linear_spread(depth, kind=kind)
        print(f"{depth:>7} {kind:<12} {s.min():>12.3e} {s.max():>12.3e} "
              f"{s.max() / max(s.min(), 1e-300):>18.3e}")

print("\nThe orthogonal product stays exactly orthogonal at every depth —")
print("a product of orthogonal matrices is orthogonal, so the condition")
print("number is 1 forever. The Gaussian product's condition number grows")
print("without bound.")
print("\nThat is eq. 53.9's product seen through its singular values, and it")
print("is why eq. 53.16's bound is pessimistic in a specific way: the")
print("NORM can be preserved while the CONDITIONING degrades, and the")
print("second is what makes optimisation hard.")
print("\nThe honest caveat: this is a linear network. Once a nonlinearity is")
print("inserted the product is no longer a product of the weight matrices")
print("alone, and orthogonal initialisation's guarantee weakens to")
print("something closer to the Gaussian one. That is why it is standard in")
print("recurrent networks, where the same matrix recurs, and not elsewhere.")
```

## 9. Practical Example

```python {tier=A name=initialization-in-practice}
"""Three questions measured on trained networks: how much does the scheme
matter, how much does normalisation reduce it, and what do residual
connections need instead.
"""
import numpy as np

rng = np.random.default_rng(3)

D, C = 20, 4
_rs = np.random.default_rng(77)
A1 = _rs.normal(size=(D, 14))
A2 = _rs.normal(size=(14, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ A1) @ A2 * 1.6
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


Xtr, ytr = make_data(20000, 1)
Xte, yte = make_data(6000, 2)
_p = np.exp(np.tanh(Xte @ A1) @ A2 * 1.6)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Net:
    """Depth-configurable MLP, optionally with layer norm, optionally
    residual."""

    def __init__(self, depth, width, scale, normed=False, residual=False,
                 zero_last_branch=False, seed=0):
        rs = np.random.default_rng(seed)
        self.depth, self.normed, self.residual = depth, normed, residual
        self.Win = rs.normal(0, np.sqrt(2.0 / D), (D, width))
        self.W = []
        for l in range(depth):
            sd = scale(width)
            Wl = rs.normal(0, sd, (width, width))
            if residual and zero_last_branch:
                Wl = np.zeros_like(Wl)
            self.W.append(Wl)
        self.Wout = rs.normal(0, np.sqrt(2.0 / width), (width, C))
        self.bout = np.zeros(C)

    @staticmethod
    def _norm(x):
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-5
        return (x - mu) / sd

    def forward(self, X):
        self.cache = []
        h = np.maximum(0.0, X @ self.Win)
        self.h_in = X
        self.h0 = h
        for W in self.W:
            inp = self._norm(h) if self.normed else h
            z = inp @ W
            a = np.maximum(0.0, z)
            self.cache.append((inp, z, h))
            h = h + a if self.residual else a
        self.hL = h
        return h @ self.Wout + self.bout

    def loss_and_grads(self, X, y):
        logits = self.forward(X)
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - logits[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gWout = self.hL.T @ d
        gbout = d.sum(axis=0)
        dh = d @ self.Wout.T
        gW = [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            inp, z, h_prev = self.cache[l]
            da = dh.copy()
            dz = da * (z > 0)
            gW[l] = inp.T @ dz
            dinp = dz @ self.W[l].T
            if self.normed:
                # gradient through the normalisation, standard form
                mu = h_prev.mean(axis=1, keepdims=True)
                sd = h_prev.std(axis=1, keepdims=True) + 1e-5
                n = h_prev.shape[1]
                xhat = (h_prev - mu) / sd
                dinp = (dinp - dinp.mean(axis=1, keepdims=True)
                        - xhat * (dinp * xhat).mean(axis=1, keepdims=True)) / sd
            dh = dinp + dh if self.residual else dinp
        gWin = self.h_in.T @ (dh * (self.h_in @ self.Win > 0))
        return loss, gWin, gW, gWout, gbout


def train(depth, width, scale, normed=False, residual=False,
          zero_last_branch=False, steps=1200, lr=2e-3, batch=128, seed=0):
    net = Net(depth, width, scale, normed, residual, zero_last_branch, seed)
    params = [net.Win] + net.W + [net.Wout, net.bout]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    rs = np.random.default_rng(seed + 40)
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gWin, gW, gWout, gbout = net.loss_and_grads(Xtr[idx], ytr[idx])
        grads = [gWin] + gW + [gWout, gbout]
        for i, (pp, g) in enumerate(zip(params, grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pp -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    te, _, _, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    return te - BAYES, acc, net


SCALES = {
    "He      sqrt(2/n)": lambda n: np.sqrt(2.0 / n),
    "Glorot  sqrt(2/2n)": lambda n: np.sqrt(1.0 / n),
    "0.5x He": lambda n: 0.5 * np.sqrt(2.0 / n),
    "2x He": lambda n: 2.0 * np.sqrt(2.0 / n),
    "fixed   0.01": lambda n: 0.01,
}

print("=" * 72)
print("how much does the scheme matter, and at what depth?")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}")
print("Excess test loss above that floor; lower is better.\n")
print(f"{'scheme':<20} " + " ".join(f"{f'depth {d}':>12}" for d in (2, 8, 20)))
by_depth = {d: [] for d in (2, 8, 20)}
for name, fn in SCALES.items():
    row = []
    for depth in (2, 8, 20):
        ex, acc, _ = train(depth, 96, fn)
        by_depth[depth].append(ex)
        row.append("diverged" if not np.isfinite(ex) else f"{ex:.4f}")
    print(f"{name:<20} " + " ".join(f"{v:>12}" for v in row))

print(f"\n{'spread (max - min)':<20} " + " ".join(
    f"{(max(by_depth[d]) - min(by_depth[d])):>12.4f}" for d in (2, 8, 20)))
print(f"{'best scheme':<20} " + " ".join(
    f"{list(SCALES)[int(np.argmin(by_depth[d]))].split()[0]:>12}"
    for d in (2, 8, 20)))

print("\nThe spread row is the measurement. At depth 2 every scheme lands")
print("within a small band — two layers cannot compound a scale error into")
print("much — and by depth 20 the band is enormous, because eq. 56.9's")
print("gamma^L is now acting over twenty layers instead of two.")
print("\nNote which scheme is best at each depth, and be honest about it:")
print("He is NOT the winner at shallow depth. At two and eight layers a")
print("smaller scale does better, because there is no compounding to")
print("compensate for and the smaller weights are simply a gentler")
print("starting point. He wins where its derivation applies — at depth,")
print("where preserving the variance is the binding constraint.")
print("\nThat is the useful form of the result. Initialisation is not a")
print("universally-ranked list of schemes; it is a scale chosen so that a")
print("particular product stays near one, and it matters in proportion to")
print("how many terms that product has.")

# --- does normalisation make it not matter? ---------------------------------
print("\n" + "=" * 72)
print("normalisation makes the scheme matter much less (section 4.5)")
print("=" * 72)
print(f"{'scheme':<20} {'depth 20, plain':>17} {'depth 20, normed':>18}")
plain, normed = {}, {}
for name, fn in SCALES.items():
    ex_p, _, _ = train(20, 96, fn, normed=False)
    ex_n, _, _ = train(20, 96, fn, normed=True)
    plain[name], normed[name] = ex_p, ex_n
    f = lambda v: "diverged" if not np.isfinite(v) else f"{v:.4f}"
    print(f"{name:<20} {f(ex_p):>17} {f(ex_n):>18}")

sp = [v for v in plain.values() if np.isfinite(v)]
sn = [v for v in normed.values() if np.isfinite(v)]
print(f"\nspread across schemes, plain  : {max(sp) - min(sp):.4f}")
print(f"spread across schemes, normed : {max(sn) - min(sn):.4f}")
print("\nThe spread is the measurement. Normalisation rescales each layer's")
print("input to unit variance, so whatever the initialisation did to the")
print("scale is overwritten before the next matmul sees it, and the choice")
print("of scheme stops being load-bearing.")
print("\nThat is why 'just use He initialisation' is adequate advice for a")
print("normalised network and inadequate for one without normalisation —")
print("and it is worth knowing which of those you are working on.")

# --- residual variance growth ------------------------------------------------
print("\n" + "=" * 72)
print("residual blocks GROW the variance (eq. 56.11)")
print("=" * 72)
print("Forward activation variance through residual blocks at He scale,")
print("untrained.\n")
X0 = rng.normal(size=(512, 96))
print(f"{'blocks':>8} {'branch init':<16} {'Var[h]':>14} "
      f"{'predicted 2^L':>15}")
for zero in (False, True):
    for L in (1, 4, 8, 16):
        rs = np.random.default_rng(5)
        h = X0.copy()
        for _ in range(L):
            W = (np.zeros((96, 96)) if zero
                 else rs.normal(0, np.sqrt(2.0 / 96), (96, 96)))
            h = h + np.maximum(0.0, h @ W)
        label = "zero-init last" if zero else "He (standard)"
        pred = "1" if zero else f"{2.0 ** L:.3g}"
        print(f"{L:>8} {label:<16} {float(np.var(h)):>14.4e} {pred:>15}")

print("\nThe variance grows geometrically, which is eq. 56.11's shape, and")
print("it grows FASTER than the 2^L the simplest reading predicts — by")
print("sixteen blocks it is nearly two orders of magnitude above it.")
print("\nThe reason is that eq. 56.11 assumed the two branches are")
print("independent, and they are not: F(x) is computed FROM x, so the")
print("skip and the branch are positively correlated and their variances")
print("more than add. 2^L is a lower bound on the growth, not an estimate")
print("of it, and the measurement is worse than the bound rather than")
print("better.")
print("\nZero-initialising the branch's last layer makes each block exactly")
print("the identity, so the variance is unchanged at any depth — which the")
print("second group confirms to every digit.")
print("\nThis is the clearest case in the chapter of an architecture")
print("changing the requirement. Per-layer variance preservation is")
print("necessary for a plain stack and NOT SUFFICIENT for a residual one,")
print("because the skip connection adds a second source of variance that")
print("the per-layer calculation never accounted for.")

# --- does it matter for training? -------------------------------------------
print("\n" + "=" * 72)
print("and what that costs in training")
print("=" * 72)
print(f"{'depth':>7} {'branch init':<18} {'excess loss':>13} {'test acc':>10}")
for depth in (4, 16):
    for zero in (False, True):
        ex, acc, _ = train(depth, 96, SCALES["He      sqrt(2/n)"],
                           residual=True, zero_last_branch=zero)
        label = "zero-init last" if zero else "He (standard)"
        f = "diverged" if not np.isfinite(ex) else f"{ex:.4f}"
        print(f"{depth:>7} {label:<18} {f:>13} {acc:>10.4f}")

print("\nThe variance growth is not merely an untrained curiosity. At depth")
print("4 the two initialisations end close together — four blocks is only a")
print("factor of sixteen and the optimiser absorbs it. At depth 16 the")
print("standard branch is far behind, and that is the four-orders-of-")
print("magnitude row from the previous table arriving as a training result.")
print("\nThe cost of the fix is nothing: zero-init matches the depth-4")
print("result at depth 16, so the deeper network is at least not worse.")
print("\nNote also that a zero-initialised branch is NOT a symmetry problem")
print("of the kind section 4.2 describes. The skip connection carries")
print("distinct values into every unit, so the units receive distinct")
print("gradients from the first step and differentiate immediately. Zero is")
print("safe here precisely because it is not the only path.")
```

## 10. Production Considerations

**Check the framework's default rather than assuming it.**
{{sec:7-internal-mechanics}}: the common `Linear` default produces a
Glorot-like gain, so a ReLU network on defaults is not He initialised.

**Zero-initialise the last layer of each residual branch.** Measured: variance
doubles per block otherwise, four orders of magnitude over sixteen blocks.

**Set the output bias to the log-odds of the base rate.** Free, and it removes
the initial steps spent learning the priors.

**Do not use the dead-unit fraction as a diagnostic for scale.** Measured: it
sits at one half for every scheme, right or wrong, because half of a symmetric
distribution is negative whatever its width.

**Log per-layer activation variance at initialisation.** One forward pass, and
it catches every scale error before a single step is wasted. Pair it with
{{ch:dl-backprop}}'s per-layer gradient norms.

**Initialise a new head separately** and consider warming it up before
unfreezing the body ({{part:14}}).

**Seed per layer rather than from one global stream** if you want architecture
comparisons at a fixed seed to be controlled ({{ch:mle-reproducibility}}).

## 11. Common Mistakes

**Zero or constant weights.** Measured: eight hidden units collapse to one
distinct function and the network scores at chance.

**Using He with tanh or Glorot with ReLU.** Measured: the ordering reverses
between the two activations, and the wrong choice compounds with depth.

**Forgetting $k^2$ in a convolution's fan.** A factor of nine for a $3\times3$
kernel.

**Standard initialisation inside a residual branch.** Measured variance
explosion.

**Assuming the framework default is what the paper used.** It usually is not.

**Applying a fan-in rule to an embedding table.** An embedding lookup is a
selection, not a sum over a fan-in, so the variance argument does not apply.

**Concluding initialisation does not matter because a normalised network was
insensitive to it.** Measured: normalisation collapses the spread, and the
network without it does not have that protection.

## 12. Failure Modes

**Silent underperformance at moderate depth.** The network trains, converges,
and reaches a worse loss than it should. There is no error and the only symptom
is the number.

**Nothing trains at all.** {{ch:dl-backprop}} measured this: at a small scale
the forward and error signals each move by twelve orders of magnitude and the
weight gradients are uniformly negligible.

**Immediate `nan`.** Too large a scale, most often in a residual network where
{{eq:residual-variance-growth}} compounds it.

**A loss that starts far above $\log C$.** The output layer's scale is wrong, so
the initial predictions are confidently arbitrary. This is the single cheapest
check in deep learning and it catches this class immediately.

**Divergence only at depth.** The scheme is slightly wrong and the error is
$\gamma^{L}$, so shallow versions of the same architecture are fine.

**A pretrained body disrupted by a random head.** Large gradients from the
untrained head flow back before the head is any good.

## 13. Alternatives

**Data-dependent initialisation** (LSUV and relatives) runs a forward pass on
real data and rescales each layer to unit output variance empirically. It works,
including where the analytic assumptions fail, and it needs data at
initialisation time.

**Fixup and related schemes** train deep residual networks *without*
normalisation by initialising and rescaling carefully. Evidence that
normalisation's role can partly be filled by initialisation alone, which is
interesting for what it says about what normalisation does.
{{maturity:EMERGING}}

**$\mu$P** derives the initialisation and learning-rate scalings jointly so that
the optimal hyperparameters transfer across model widths. The most principled
extension of this chapter's argument. {{maturity:EMERGING}}

**Pretrained weights.** For most practical work the initialisation question is
answered by a checkpoint, and this chapter applies to the parts that are new.

**Lottery-ticket pruning** finds sparse subnetworks that train well *from their
original initialisation*, which suggests the specific random draw carries more
information than the variance argument accounts for.
{{maturity:RESEARCH FRONTIER}}

## 14. Evaluation

**Print the per-layer activation variance at initialisation.** One forward pass.
A near-constant profile is the target.

**Print the per-layer gradient norm at initialisation.**
{{ch:dl-backprop}}'s diagnostic, and the two together separate a forward-scale
problem from a backward-scale one.

**Check the initial loss against $\log C$.**

**Compare against a shallower version.** If depth-4 trains and depth-40 does
not, the scale is compounding.

**Sweep the scale.** Multiply the chosen standard deviation by 0.5, 1 and 2 and
train briefly. A flat result means the network is insensitive — usually because
it is normalised — and a sharp one means the scale is load-bearing.

**Verify the residual branch starts near zero**, by measuring the block's output
variance against its input's.

## 15. Advanced Concepts

**Dynamical isometry.** The strongest version of this chapter's goal: not merely
preserving the norm on average but keeping the entire input-output Jacobian's
singular values near 1. Achievable with orthogonal weights and carefully chosen
activations, and it allows training of extremely deep networks without
normalisation or residual connections.

**Mean-field theory of signal propagation.** Treating an infinitely wide network
as a dynamical system in the activation statistics gives an *edge of chaos* at
which correlations neither collapse nor saturate, and the depth to which
information propagates diverges there. It recovers the He and Glorot scales as a
special case and predicts more.

**The neural tangent kernel.** In the infinite-width limit at a specific
initialisation scale, training behaves like kernel regression with a fixed
kernel. It gives exact statements about convergence and explains less about real
finite networks than the volume of work on it might suggest.
{{maturity:ESTABLISHED}} as mathematics, contested as an explanation.

**$\mu$P's width scaling.** The observation that the correct scaling of
initialisation, learning rate and output multiplier with width is *not* what
standard practice does, and fixing it makes hyperparameters transfer.

**Lottery tickets.** {{maturity:RESEARCH FRONTIER}}, and a direct challenge to
the view that initialisation is only about scale.

## 16. Connection to Previous Chapters

{{ch:dl-backprop}} measured what a bad initialisation does — the twelve orders
of magnitude in forward and error signal — and {{eq:unrolled-backprop}}'s
product is what this chapter's scalar controls. The two chapters are one
argument.

{{ch:dl-activations}} supplied {{eq:relu-second-moment}}'s factor of one half,
which is literally the two in $\sqrt{2/n}$, and measured it directly.
{{ch:dl-neural-networks}} measured the symmetry collapse this chapter explains.
{{ch:math-random-vars}} supplied the variance of a sum of independent products.
{{ch:dl-optimizers}} and {{ch:dl-lr-schedules}} both start from wherever this
chapter leaves the parameters, and warmup's second justification is precisely
that the initial point is arbitrary.

Forward: {{ch:dl-normalization}} makes the network far less sensitive to this
chapter's choice, as measured here, and the two are best understood as
alternative ways of controlling the same quantity.
{{ch:tf-architectures}}'s residual connections change the requirement from
preserving scale to starting at the identity.
{{ch:dl-rnns}} is where orthogonal initialisation earns its place.

## 17. Exercises

**Beginner**

1. Why can weights not be initialised to zero?
2. Why can biases be zero?
3. State the He and Glorot variances.
4. Where does the factor of two in He initialisation come from?
5. Why does an embedding table not follow a fan-in rule?

**Intermediate**

6. Derive {{eq:he-derived}} from {{eq:forward-variance}}.
7. Derive the backward condition {{eq:backward-variance}} and explain why it
   conflicts with the forward one.
8. Using {{eq:variance-gain}}, find the depth at which a 20% error in the
   standard deviation causes a hundredfold change in activation variance.
9. Compute the fan-in and fan-out for a convolution with 64 input channels,
   128 output channels and a $5\times5$ kernel.
10. Explain why {{eq:sqrt-l-scaling}} converges to $e$.
11. Why is the dead-unit fraction not a diagnostic for a bad scale?

**Advanced**

12. Derive {{eq:relu-second-moment}} for a general symmetric input
    distribution, and say where symmetry is used.
13. Show that a product of orthogonal matrices is orthogonal, and explain what
    that implies for {{eq:unrolled-backprop}}.
14. Derive the Glorot variance as the solution of a specific optimisation over
    the two conditions, and state the objective that makes it exact.
15. Explain dynamical isometry and why preserving the norm on average is weaker.
16. Derive the initialisation scale for a GELU network, using
    {{ch:dl-activations}}'s measured second moment.

**Implementation**

17. Implement all five schemes and reproduce the 50-layer variance table.
18. Implement LSUV and compare its layer scales against the analytic ones.
19. Implement orthogonal initialisation for non-square matrices and verify
    semi-orthogonality.
20. Measure the residual variance growth for your own architecture and verify
    the zero-init fix.

**Reasoning**

21. A 40-layer network will not train; a 4-layer version of the same
    architecture does. Give an ordered diagnostic procedure.
22. Adding normalisation made your network insensitive to the initialisation
    scale. What does that tell you, and what does it not?

## 18. Interview Questions

**"Why can't we initialise to zero?"** — Symmetry: identical units, identical
gradients, forever. Note that biases can be zero.

**"Where does $\sqrt{2/n}$ come from?"** — Derive it. Variance of a sum of $n$
products, then ReLU's factor of one half.

**"Glorot or He?"** — He for rectifiers, Glorot for saturating activations, and
say why: the factor of two is the ReLU compensation and it is wrong for tanh.

**"Fan-in or fan-out?"** — Forward versus backward preservation; they conflict
for non-square layers and Glorot is the compromise. Note it barely matters at
constant width.

**"How does initialisation change in a residual network?"** — From preserving
scale to starting at the identity. Give the variance-doubling argument.

**"Does initialisation still matter with batch norm?"** — Much less, and give
the measured reason: the normalisation overwrites the scale. Not *not at all* —
the first layer, the residual branches and the output layer still depend on it.

**"How would you diagnose an initialisation problem?"** — Per-layer activation
variance and gradient norm at initialisation, and the initial loss against
$\log C$.

## 19. Research Questions

**Can normalisation be replaced entirely by initialisation?** Fixup and
relatives train deep residual networks without it, and they have not displaced
normalisation in practice. What normalisation provides beyond scale control is
not settled. {{maturity:EMERGING}}

**Does the specific random draw matter beyond its variance?** The lottery-ticket
results say some subnetworks train well from their original initialisation and
not from a fresh one at the same scale, which the variance argument cannot
explain. {{maturity:RESEARCH FRONTIER}}

**Is dynamical isometry practically achievable at scale?** It works in
controlled settings and has not become standard, and whether that is a
fundamental limitation or an engineering one is open. {{maturity:EMERGING}}

**How should initialisation scale with width and depth jointly?** $\mu$P answers
the width question and the depth question is less settled.
{{maturity:EMERGING}}

## 20. Chapter Summary

Initialisation is choosing a scale so that variance is preserved as signal
propagates forward and as gradient propagates backward. The derivation is four
lines: a unit summing $n$ inputs has $\Var[z] = n\Var[W]\Var[x]$, so preservation
needs $\Var[W] = 1/n$; a rectifier halves the second moment, so it needs
$2/n$. **The factor of two in He initialisation is exactly ReLU's one half**,
and nothing more.

Measured through 50 layers, He held the activation variance essentially constant
while LeCun and Glorot both decayed — they are correct for a linear or saturating
activation and missing the rectifier's compensation. With tanh the ordering
reversed, and He's factor of two became a systematic overshoot that tanh's own
saturation partly masked. The sensitivity is severe: {{eq:variance-gain}} says
the variance ratio is $\gamma^L$, so a 30% error in the standard deviation is
fatal at depth 50.

One plausible diagnostic turns out to measure nothing. The fraction of dead ReLU
units sat at one half at every depth and under every scheme, right or wrong,
because half of a symmetric distribution is negative whatever its width. Only
the variance profile shows the problem.

Glorot's two conditions genuinely conflict. The forward pass wants
$1/n_{\text{in}}$, the backward wants $1/n_{\text{out}}$, and the measured gain
table shows {{eq:glorot}} hitting 1.0 exactly only for a square layer and
missing in reciprocal directions otherwise. That is the derivation's conclusion,
not a flaw in it.

Randomness is not a convenience. Measured on a tiny network, a constant
initialisation collapsed eight hidden units to one distinct function and the
network scored at chance on a problem one unit cannot represent. Orthogonal
initialisation goes further than variance preservation: measured singular values
were all exactly 1 against a Gaussian matrix's spread from near zero to about
two, and a product of orthogonal matrices stayed perfectly conditioned at depth
50 where the Gaussian product's condition number grew without bound. That
guarantee is exact for a linear network and weakens once a nonlinearity is
inserted, which is why the technique lives mainly in recurrent networks.

Two architectural developments changed what initialisation has to do.
Normalisation overwrites the scale at every layer, and the measured spread
across schemes collapsed accordingly — which makes "just use He" adequate advice
for a normalised network and inadequate for one without. Residual connections
add a second variance source that the per-layer calculation never accounted for:
measured variance roughly doubled per block, four orders of magnitude over
sixteen. Zero-initialising each branch's last layer makes every block exactly
the identity and fixes it, and it is safe from the symmetry problem precisely
because the skip connection is a second path.

## 21. Further Reading

{{cite:glorot2010}} is worth reading for its first half rather than its formula.
The paper measures activation and gradient distributions layer by layer in
networks that were failing to train, and the diagnostic approach is more
transferable than the specific scale it arrives at. It is also unusually candid
that {{eq:glorot}} is a compromise.

{{cite:he2015init}} is shorter and does one thing: redo Glorot's calculation for
the rectifier. Read them together and the factor of two is obvious. The paper
also introduces PReLU, which is the part almost nobody remembers.

{{cite:saxe2014}} for orthogonal initialisation and the deep-linear-network
analysis. The result — convergence time independent of depth — is striking, and
the honest caveat about linearity is stated in the paper.

{{cite:liu2020admin}} for the residual-branch analysis. Its useful move is
diagnostic rather than prescriptive: it argues that the instability people
attributed to unbalanced gradients is really an amplification effect in the
residual branch, and that is the framing {{sec:5-formal-explanation}} adopts.

**Where to go next:** {{ch:dl-normalization}} is the alternative solution to the
same problem, and the measured insensitivity in {{sec:9-practical-example}} is
the reason it largely displaced careful initialisation. Read it next, with the
variance calculation of {{sec:6-mathematical-foundation}} in mind.
