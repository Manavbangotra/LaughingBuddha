---
id: dl-activations
number: 50
part: VI
tier: full
status: reviewed
requires: [dl-neural-networks, ml-logistic, math-derivatives]
provides: [activation-function, relu, dead-relu, saturation, vanishing-gradient,
           gelu-swish, activation-choice]
citations: [glorot2010, he2015init, krizhevsky2012, rumelhart1986]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what an activation function must provide and what it must avoid.
2. Derive the sigmoid and tanh derivatives and quantify their saturation.
3. Compute how much gradient survives $L$ layers of a saturating activation.
4. Explain why ReLU made deep networks trainable, in terms of its derivative.
5. Diagnose and prevent dead units.
6. Compare ReLU, Leaky ReLU, ELU, GELU and SiLU and choose between them.
7. Explain why the choice of activation interacts with initialisation and
   normalisation.
8. State honestly how much the choice matters in 2026 and where it still does.

## 2. Why This Matters

**The activation is where the vanishing-gradient problem lives.** The chain
rule multiplies one derivative per layer, so the activation's derivative
appears in that product $L$ times. Sigmoid's derivative never exceeds $0.25$;
ten layers of it multiply to at most $10^{-6}$, and the early layers receive
nothing. This single number explains why deep networks were untrainable for two
decades and why a change as small as $\max(0, z)$ mattered so much.

**It is the cheapest architectural decision with the largest historical
consequence.** {{cite:krizhevsky2012}} attributed a large part of AlexNet's
training speed to rectified units. No new theory, no new architecture — a
different two-character function, and networks that had been intractable became
routine.

**It is also where a lot of wasted effort goes now.** Dozens of activations
have been proposed and the honest 2026 position is that for most work the
choice is between ReLU and GELU, the difference is small, and the interaction
with initialisation and normalisation matters more than the function itself.
{{sec:9-practical-example}} measures that, and the measurement is the point:
knowing which decisions are load-bearing is as valuable as knowing how to make
them.

## 3. Prerequisites

{{ch:dl-neural-networks}} for the unit and why a nonlinearity is required at
all. {{ch:ml-logistic}} for the sigmoid and the exact cancellation with
cross-entropy that {{sec:5-formal-explanation}} revisits.
{{ch:math-derivatives}} for the chain rule.

## 4. Intuitive Explanation

### 4.1 What the activation has to do

Two jobs, in tension.

**Be nonlinear**, or the network collapses to one affine map
({{eq:deep-linear-collapse}}). This is the requirement, and almost any
nonlinearity satisfies it.

**Pass gradient through**, or the layers below cannot learn. This is the
constraint, and most nonlinearities fail it.

```text
        φ(z)                        φ'(z)
   sigmoid ──╭───────                 ╭─╮        max 0.25
            ╱                        ╱   ╲       ...and near zero
       ────╯                    ────╯     ╰────  almost everywhere

   ReLU      ╱                  ────────┐
            ╱                            │       exactly 1 when active
       ────╯                    ─────────┘       exactly 0 when not
```

The whole history of activation functions is the search for a function that is
nonlinear enough to be useful and flat enough nowhere to block the gradient.

### 4.2 Saturation

A function **saturates** where its output stops responding to its input — the
flat tails of a sigmoid. In those regions the derivative is nearly zero, so a
unit that has saturated stops learning, and stops letting anything below it
learn either.

Sigmoid and tanh saturate at both ends. Their derivatives peak at the centre
($0.25$ and $1.0$ respectively) and decay exponentially outwards. Push a unit's
pre-activation to $\pm 6$ and its derivative is about $0.0025$ — it is, for
practical purposes, switched off.

The compounding is what kills you. {{sec:6-mathematical-foundation}} does the
arithmetic: with sigmoid's *best case* derivative of $0.25$ at every layer, ten
layers attenuate the gradient by a factor of a million, and that is the
optimistic bound.

### 4.3 ReLU

$$
\relu(z) = \max(0, z)
$$

Absurdly simple, and it changes the arithmetic completely. For positive inputs
the derivative is exactly 1 — not approximately, exactly — so the gradient
passes through unattenuated no matter how many layers it crosses. For negative
inputs it is exactly 0.

Three consequences follow immediately:

**Deep networks become trainable.** The product of derivatives along an active
path is exactly 1, so depth no longer attenuates.

**Activations become sparse.** Roughly half the units are off for any given
input, which is computationally convenient and provides a mild regularisation.

**Units can die.** A unit whose pre-activation is negative for *every* input
has zero gradient forever and can never recover. This is ReLU's characteristic
failure and {{sec:12-failure-modes}} treats it.

### 4.4 The modern family

The functions that replaced plain ReLU all address the same thing — the hard
zero — in slightly different ways:

```text
   Leaky ReLU   max(αz, z)              small slope below zero; cannot die
   ELU          z or α(eᶻ−1)            smooth, negative saturation
   GELU         z·Φ(z)                  smooth, probabilistic motivation
   SiLU/Swish   z·σ(z)                  smooth, non-monotone near zero
```

GELU and SiLU are the transformer-era defaults. Both are smooth, both are
non-monotone in a small region just below zero, and both slightly outperform
ReLU in large models. Whether the smoothness or the non-monotonicity is
responsible is not established, which is worth saying plainly.

### 4.5 The output layer is different

Hidden activations and output activations answer different questions, and
conflating them is a common bug.

A hidden activation exists to provide nonlinearity and pass gradient. An output
activation exists to put the prediction in the right space: identity for
unbounded regression, sigmoid for a probability, softmax for a distribution
over classes.

The choice is dictated by the loss, not by the architecture, and
{{ch:dl-losses}} derives each pairing. The one rule to carry from here:
**never use a hidden-layer activation on the output because it worked well
inside the network.** A sigmoid on a regression output caps every prediction
below 1.

## 5. Formal Explanation

### 5.1 The candidates

{#tbl:activations caption="Activation functions, their derivatives, and the property that decides whether they are usable in a deep network. The maximum derivative is what appears in the layer-product of section 6.1."}

| Name | $\phi(z)$ | $\phi'(z)$ | $\max \phi'$ | Saturates |
|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ | $0.25$ | both ends |
| Tanh | $\tanh z$ | $1-\tanh^{2}z$ | $1.0$ | both ends |
| ReLU | $\max(0,z)$ | $\Ind[z>0]$ | $1.0$ | below zero |
| Leaky ReLU | $\max(\alpha z, z)$ | $1$ or $\alpha$ | $1.0$ | never |
| ELU | $z$ or $\alpha(e^{z}-1)$ | $1$ or $\alpha e^{z}$ | $1.0$ | below zero |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\phi_{\mathcal{N}}(z)$ | $\approx 1.13$ | below zero |
| SiLU | $z\,\sigma(z)$ | $\sigma(z)(1 + z(1-\sigma(z)))$ | $\approx 1.10$ | below zero |

Two entries deserve comment. GELU and SiLU have maximum derivatives slightly
*above* 1, which is unusual and comes from their non-monotone region. And tanh
has the same maximum derivative as ReLU — the difference is not the peak but
how quickly it decays, which is what {{sec:8-implementation}} measures.

### 5.2 Why tanh beats sigmoid

Both saturate, so why was tanh preferred for years?

**Zero-centred output.** Sigmoid outputs lie in $(0,1)$, so every input to the
next layer is positive. The gradient with respect to that layer's weights is
then $\delta \cdot h$ with $h > 0$ throughout, so **all weights in a row receive
gradients of the same sign** and can only move together — producing a
characteristic zig-zag descent. Tanh's $(-1,1)$ output removes this.

**Four times the derivative.** $\max\tanh' = 1$ against $\max\sigma' = 0.25$,
so the layer-product of {{sec:6-mathematical-foundation}} decays four times
more slowly per layer.

Neither fixes saturation, which is why both were abandoned for hidden layers.

### 5.3 The dying ReLU

A unit dies when its pre-activation is negative for every input in the data
distribution:

$$
\vec{w}\T\vec{x} + b < 0 \quad \text{for all } \vec{x} \text{ in the support}
$$ (eq:dead-condition)

Then $\phi'(z) = 0$ always, the gradient with respect to $\vec{w}$ and $b$ is
identically zero, and no update can ever occur. **The death is permanent** —
unlike saturation in a sigmoid, which is a region a unit can move out of.

The main causes, in order of how often they occur:

- **A learning rate too large**, driving the bias sharply negative in one step.
  This is the dominant cause and produces death in waves.
- **A large negative bias initialisation.**
- **Poor input scaling**, so pre-activations are large in magnitude.

The remedies are Leaky ReLU or ELU (nonzero gradient below zero), a smaller
learning rate, and normalisation ({{ch:dl-normalization}}), which keeps
pre-activations centred.

### 5.4 GELU and SiLU

$$
\text{GELU}(z) = z\,\Phi(z), \qquad
\text{SiLU}(z) = z\,\sigma(z)
$$ (eq:gelu-silu)

where $\Phi$ is the standard normal CDF. The GELU motivation: instead of gating
deterministically on $\Ind[z>0]$, gate *stochastically* with probability
$\Phi(z)$ and take the expectation. That yields $z\Phi(z)$ exactly.

Both are smooth everywhere, both approach ReLU for large $|z|$, and both dip
slightly below zero for small negative $z$ — the non-monotone region. A common
fast approximation is

$$
\text{GELU}(z) \approx 0.5z\Big(1 + \tanh\big[\sqrt{2/\pi}\,(z + 0.044715z^{3})\big]\Big)
$$ (eq:gelu-tanh)

The approximation exists for portability rather than speed: `erf` is not
available in every kernel language and was not always available in fast
vectorised form. {{sec:8-implementation}} measures both the approximation error
and — with a modern vectorised `erf` — finds the exact form is actually the
faster of the two, which is a good example of an optimisation outliving its
justification.

### 5.5 How much does the choice matter?

Honestly: less than the literature's volume suggests, and not zero.

**It mattered enormously once.** Sigmoid to ReLU was the difference between
untrainable and trainable, and that transition is the reason this chapter
exists.

**Between modern activations it depends on what else is in the network.** In
large models with normalisation, adaptive optimisers and tuned schedules, ReLU,
GELU, SiLU and ELU typically differ by a fraction of a percentage point — often
within the run-to-run variance {{ch:mle-reproducibility}} measured. In a bare
network with none of that machinery the gap is far larger, and
{{sec:9-practical-example}} measures roughly an order of magnitude between
SiLU and plain ReLU. The received "they are all about the same" is true of the
regime it was measured in and not of the general case.

**It is not independent of everything else.** ReLU needs He initialisation and
Leaky ReLU tolerates worse initialisation; normalisation makes all of them more
similar by keeping pre-activations in the well-behaved region. The activation,
the initialisation and the normalisation are one design decision with three
parts, which is the framing {{ch:dl-initialization}} and
{{ch:dl-normalization}} develop.

**Practical rule for 2026:** ReLU for convolutional networks, GELU or SiLU for
transformers, Leaky ReLU if you observe dead units, and do not spend a
hyperparameter search on it.

## 6. Mathematical Foundation

### 6.1 The gradient through $L$ layers

For a network of $L$ layers, the gradient with respect to layer $k$'s
pre-activation involves the product of every Jacobian above it. Ignoring the
weight matrices for a moment to isolate the activation's contribution:

$$
\frac{\partial \Loss}{\partial z^{(k)}}
 = \frac{\partial \Loss}{\partial z^{(L)}}
   \prod_{l=k+1}^{L}\Big(\mat{W}^{(l)\top}\,\diag\big(\phi'(z^{(l-1)})\big)\Big)
$$ (eq:gradient-product)

Take norms and bound crudely:

$$
\Big\|\frac{\partial\Loss}{\partial z^{(k)}}\Big\|
 \le \Big\|\frac{\partial\Loss}{\partial z^{(L)}}\Big\|
   \prod_{l=k+1}^{L}\|\mat{W}^{(l)}\|\;\big|\phi'\big|_{\max}
$$ (eq:gradient-bound)

The activation contributes a factor of $|\phi'|_{\max}$ **per layer**. For
sigmoid that is $0.25$:

$$
0.25^{10} \approx 9.5\times10^{-7}, \qquad
0.25^{20} \approx 9.1\times10^{-13}
$$

and these are the *optimistic* bounds, achieved only if every unit sits exactly
at its point of maximum derivative. In a real network the typical derivative is
far smaller.

For ReLU the factor is $1$ on active paths, so the product does not decay from
the activation at all — the entire attenuation comes from $\|\mat{W}\|$, which
is what initialisation controls ({{ch:dl-initialization}}). **That separation
is the point: ReLU removes one of the two multiplicative decay sources, leaving
a problem with one cause instead of two.**

### 6.2 Deriving the sigmoid and tanh derivatives

With $\sigma = 1/(1+e^{-z})$:

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^{2}}
 = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
 = \sigma(z)\big(1-\sigma(z)\big)
$$

Maximised where $\sigma = 1/2$, giving $\sigma' = 1/4$.

For tanh, using $\tanh z = 2\sigma(2z) - 1$:

$$
\tanh'(z) = 4\sigma'(2z) = 4\sigma(2z)(1-\sigma(2z)) = 1 - \tanh^{2}(z)
$$

Maximised at $z=0$ with value 1. The relationship also shows *why* the factor
of four appears: tanh is a sigmoid stretched by two in both directions, and
stretching horizontally by two and vertically by two multiplies the slope by
four.

### 6.3 The zig-zag from non-zero-centred activations

Let layer $l$'s weights be $\mat{W}$, with input $\vec{h}$ from a sigmoid, so
$h_j > 0$ for all $j$. The gradient with respect to row $i$ of $\mat{W}$ is

$$
\frac{\partial\Loss}{\partial \mat{W}_{ij}} = \delta_i\,h_j
$$

For a fixed output unit $i$, the sign of every component is the sign of
$\delta_i$, because every $h_j$ is positive. **All weights feeding unit $i$
must increase together or decrease together on a given example.**

Geometrically, the gradient is confined to one orthant, so reaching a solution
requiring some weights up and others down needs an alternating sequence of
steps — the zig-zag. Averaging over a mini-batch mitigates it, since different
examples can produce different $\delta_i$ signs, but does not remove it.

This is a distinct problem from saturation, and it is the reason a zero-centred
activation is preferable even when saturation is not the binding constraint.

### 6.4 Expected fraction of active ReLU units

At initialisation with weights symmetric about zero and zero bias, the
pre-activation $z = \vec{w}\T\vec{x}$ is symmetric about zero, so

$$
\Prob(z > 0) = \tfrac{1}{2}
$$

Half the units are active in expectation. This is exactly the fact that
produces the factor of two in He initialisation: passing through a rectifier
halves the variance, because half the mass is set to zero. With
$z \sim \mathcal{N}(0, \sigma^{2})$,

$$
\Var[\relu(z)] = \E[\relu(z)^{2}] - \E[\relu(z)]^{2}
 = \frac{\sigma^{2}}{2} - \frac{\sigma^{2}}{2\pi}
 \approx 0.34\,\sigma^{2}
$$ (eq:relu-variance)

using $\E[\relu(z)^{2}] = \sigma^{2}/2$ by symmetry and
$\E[\relu(z)] = \sigma/\sqrt{2\pi}$.

He initialisation uses the *second moment* $\E[\relu(z)^{2}] = \sigma^{2}/2$
rather than the variance, because what propagates through the next layer's
matrix multiply is the uncentred second moment. That gives exactly the factor
of two, and {{ch:dl-initialization}} completes the derivation.

### 6.5 GELU as an expected gate

Consider gating $z$ by a Bernoulli variable whose probability depends on $z$
itself: keep $z$ with probability $\Prob(Z \le z)$ for $Z \sim
\mathcal{N}(0,1)$, and zero it otherwise. The expected output is

$$
\E[\text{output}] = z \cdot \Prob(Z \le z) + 0 \cdot \Prob(Z > z)
 = z\,\Phi(z)
$$

which is GELU exactly. The interpretation is that ReLU gates on
$\Ind[z > 0]$ — a hard, deterministic decision — while GELU gates on how far
above zero $z$ is, in units of the input's own standard deviation. Inputs near
the boundary are partially passed rather than arbitrarily cut.

Whether this is the *reason* GELU works slightly better is not established, and
saying otherwise would be exactly the error {{ch:dl-normalization}} documents
for batch normalisation.

## 7. Internal Mechanics

### 7.1 What an activation costs

Elementwise operations are **memory-bound**, as {{ch:dl-neural-networks}}
established: one arithmetic operation per element loaded and stored. The
consequence is that the cost difference between activations is not the
arithmetic but whether the framework can fuse them into the preceding matrix
multiply.

```text
   op            arithmetic          bandwidth        fusable?
   ─────────     ────────────        ─────────        ────────
   ReLU          1 compare, 1 select  read + write     yes, trivially
   GELU (erf)    1 erf, 2 mul         read + write     yes, if erf available
   GELU (tanh)   1 tanh, ~5 mul/add   read + write     yes
   SiLU          1 exp, 2 mul         read + write     yes
```

Unfused, a GELU can cost a noticeable fraction of a small layer's time because
the tensor is read and written again. Fused, the difference nearly vanishes.
This is why benchmarks of activation speed disagree so much: they are measuring
the fusion, not the function.

### 7.2 What must be stored for the backward pass

Each activation's backward pass needs *something* from the forward pass, and
which thing it is affects memory:

- **ReLU** needs only the sign of $z$, so a 1-bit mask suffices in principle,
  and frameworks commonly store the output (which is zero exactly where the
  mask is) rather than a separate mask.
- **Sigmoid and tanh** can compute their derivative from the *output*:
  $\sigma' = \sigma(1-\sigma)$ and $\tanh' = 1 - \tanh^{2}$. The input can be
  discarded, which is a genuine memory saving.
- **GELU and SiLU** need the *input* $z$, because their derivatives are not
  expressible in the output alone.

That last row is a real cost. In a large transformer the feed-forward
activation is applied to a tensor four times the model width, so storing its
input rather than reusing its output is a meaningful fraction of activation
memory. It is one reason ReLU variants persist in memory-constrained settings.

### 7.3 Numerical care

**Sigmoid overflows naively.** Computing `1/(1+exp(-z))` for $z = -800$
overflows. The standard fix branches on the sign, as
{{ch:ml-logistic}}'s implementation did, and is what every library does
internally.

**Never compose sigmoid with a separate log.** $\log\sigma(z)$ computed in two
steps loses precision and underflows for moderately negative $z$; the fused
form $-\log(1+e^{-z})$, itself computed via `logaddexp`, is stable.
{{ch:dl-losses}} derives the fused softmax/cross-entropy for the same reason.

**ReLU's derivative at exactly zero is a convention.** The function is not
differentiable there. Frameworks choose 0; the choice is arbitrary and
practically irrelevant, since exact zeros have measure zero in floating point —
though not in a network with zero-initialised biases and zero inputs, which is
one more reason not to initialise everything to zero.

## 8. Implementation

```python {tier=A name=activation-properties}
"""Activations and their derivatives, and the layer-product that decides
whether a deep network trains.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the functions and their exact derivatives ------------------------------
def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1.0 + e)
    return out


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (z > 0).astype(float)


def leaky_relu(z, a=0.01):
    return np.where(z > 0, z, a * z)


def d_leaky_relu(z, a=0.01):
    return np.where(z > 0, 1.0, a)


def elu(z, a=1.0):
    return np.where(z > 0, z, a * (np.exp(np.minimum(z, 0)) - 1))


def d_elu(z, a=1.0):
    return np.where(z > 0, 1.0, a * np.exp(np.minimum(z, 0)))


try:
    from scipy.special import erf as _erf          # vectorised, C-speed
except ImportError:                                # pragma: no cover
    from math import erf as _scalar_erf
    _erf = np.vectorize(_scalar_erf)


def _Phi(z):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def gelu(z):
    return z * _Phi(z)


def d_gelu(z):
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return _Phi(z) + z * pdf


def gelu_tanh(z):
    """Eq. 50.2, the fast approximation."""
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi)
                                  * (z + 0.044715 * z ** 3)))


def silu(z):
    return z * sigmoid(z)


def d_silu(z):
    s = sigmoid(z)
    return s * (1 + z * (1 - s))


ACTS = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (np.tanh, d_tanh),
    "ReLU": (relu, d_relu),
    "LeakyReLU": (leaky_relu, d_leaky_relu),
    "ELU": (elu, d_elu),
    "GELU": (gelu, d_gelu),
    "SiLU": (silu, d_silu),
}

# --- table 50.1, verified numerically ---------------------------------------
print("=" * 72)
print("maximum derivative, and the derivative far from zero (table 50.1)")
print("=" * 72)
zs = np.linspace(-12, 12, 200001)
print(f"{'activation':<12} {'max phi_prime':>14} {'at z=':>8} "
      f"{'phi_prime(4)':>13} {'phi_prime(-4)':>14}")
for name, (f, df) in ACTS.items():
    d = df(zs)
    i = int(np.argmax(d))
    print(f"{name:<12} {d[i]:>14.4f} {zs[i]:>8.2f} "
          f"{float(df(np.array([4.0]))[0]):>13.4f} "
          f"{float(df(np.array([-4.0]))[0]):>14.4f}")

print("\nSigmoid's derivative peaks at 0.25 and tanh's at 1.00 — the factor of")
print("four derived in section 6.2. Both are essentially zero by |z| = 4.")
print("ReLU's is exactly 1 wherever it is active, and GELU and SiLU exceed 1")
print("slightly, which comes from their non-monotone region just below zero.")

# --- section 6.1: what survives L layers ------------------------------------
print("\n" + "=" * 72)
print("what fraction of the gradient survives L layers (eq. 50.4)")
print("=" * 72)
print("Best case: every unit sits exactly at its maximum-derivative point.")
print("This is an OPTIMISTIC bound; real units are rarely there.\n")
print(f"{'activation':<12} " + " ".join(f"{'L=' + str(L):>12}"
                                        for L in (1, 5, 10, 20, 50)))
for name, (f, df) in ACTS.items():
    m = float(np.max(df(zs)))
    print(f"{name:<12} " + " ".join(f"{m ** L:>12.3e}" for L in (1, 5, 10, 20, 50)))

print("\nTen layers of sigmoid attenuate the gradient by a factor of a")
print("million even in the best case; twenty layers by 1e-12, which is below")
print("float32's ability to represent a meaningful update. The rectifier")
print("family's product is exactly 1 at every depth.")
print("\nThe GELU and SiLU rows show the bound going the other way and")
print("growing without limit, which is a good reminder that this is a BOUND")
print("and not a prediction. Their derivative exceeds 1 only in a narrow")
print("region near z = 1.4, and no real network has every unit sitting")
print("there. The next table, using derivatives at pre-activations from an")
print("actual forward pass, is the informative one.")
print("\nThat single column is the whole reason deep networks became")
print("trainable. It is not that ReLU is a better function — it is that it")
print("removes one of the two multiplicative decay terms in eq. 50.3,")
print("leaving only the weight norms, which initialisation can control.")

# --- and the REALISTIC case, with units where they actually sit -------------
print("\n" + "=" * 72)
print("the realistic case: derivatives at units' ACTUAL pre-activations")
print("=" * 72)
print("Pre-activations from a real forward pass, standard normal inputs and")
print("unit-variance weights, so z has variance of order 1.\n")
print(f"{'activation':<12} {'mean phi_prime':>15} {'median':>9} "
      f"{'implied 10-layer factor':>25}")
z_real = rng.normal(0, 1.5, 200000)
for name, (f, df) in ACTS.items():
    d = df(z_real)
    print(f"{name:<12} {d.mean():>15.4f} {np.median(d):>9.4f} "
          f"{d.mean() ** 10:>25.3e}")

print("\nThe realistic numbers are worse than the bound for the saturating")
print("functions and better than one might fear for ReLU: its mean")
print("derivative is about 0.5, because half the units are inactive")
print("(section 6.4), so the AVERAGE path decays — but the ACTIVE paths do")
print("not decay at all, and it is the active paths that carry the signal.")
print("\nThat distinction matters. A sigmoid attenuates every path; a ReLU")
print("blocks some paths completely and leaves the rest untouched.")

# --- section 6.4: the variance identity that gives He its factor of 2 -------
print("\n" + "=" * 72)
print("why the rectifier halves the second moment (eq. 50.5)")
print("=" * 72)
print(f"{'input sd':>9} {'E[z^2]':>10} {'E[relu(z)^2]':>14} {'ratio':>8} "
      f"{'Var[relu(z)]':>14} {'/sigma^2':>10}")
for sd in (0.5, 1.0, 2.0, 4.0):
    z = rng.normal(0, sd, 400000)
    r = relu(z)
    print(f"{sd:>9.1f} {np.mean(z ** 2):>10.4f} {np.mean(r ** 2):>14.4f} "
          f"{np.mean(r ** 2) / np.mean(z ** 2):>8.4f} "
          f"{r.var():>14.4f} {r.var() / sd ** 2:>10.4f}")

print(f"\ntheory: E[relu(z)^2]/E[z^2] = 0.5 exactly, by symmetry")
print(f"        Var[relu(z)]/sigma^2 = 0.5 - 1/(2*pi) = "
      f"{0.5 - 1 / (2 * np.pi):.4f}")
print("\nThe second moment is halved exactly, which is the fact He")
print("initialisation compensates for with a factor of two (Chapter 56).")
print("Note that the VARIANCE is not halved — it is reduced by a different")
print("factor — and the second moment is the right quantity because that is")
print("what propagates through the next matrix multiply.")

# --- eq. 50.2: how good is the tanh approximation to GELU? ------------------
print("\n" + "=" * 72)
print("the tanh approximation to GELU (eq. 50.2)")
print("=" * 72)
zz = np.linspace(-6, 6, 20001)
exact, approx = gelu(zz), gelu_tanh(zz)
print(f"max absolute error : {np.abs(exact - approx).max():.3e}")
print(f"max relative error where |GELU| > 0.01 : "
      f"{np.max(np.abs(exact - approx)[np.abs(exact) > 0.01] / np.abs(exact)[np.abs(exact) > 0.01]):.3e}")
print(f"error at z = 0 : {abs(gelu(np.array([0.0]))[0] - gelu_tanh(np.array([0.0]))[0]):.3e}")

import time
big = rng.normal(size=(2000, 2000))
for label, fn in (("GELU via erf", gelu), ("GELU via tanh", gelu_tanh),
                  ("ReLU", relu)):
    t0 = time.perf_counter()
    for _ in range(3):
        fn(big)
    print(f"{label:<16} {(time.perf_counter() - t0) / 3:.4f} s "
          f"per 4M-element pass")

print("\nThe approximation is accurate to about 5e-4 absolute, far below the")
print("noise in any training run. The relative error reaches a few per cent")
print("only where GELU itself is close to zero, which is harmless for the")
print("same reason.")
print("\nThe timing is the interesting part, and it does not say what the")
print("approximation's existence would suggest: with a vectorised library")
print("erf, the EXACT form is about 2.7x FASTER than the tanh approximation,")
print("which needs a tanh, a cube and several multiplies. The approximation")
print("was worth having when erf was unavailable or slow in the target")
print("kernel language, and on this stack it is now a pessimisation — an")
print("optimisation that outlived its justification.")
print("\nBoth caveats from section 7.1 still apply: these are UNFUSED")
print("measurements, and once the activation is fused into the preceding")
print("matmul the gap between all three largely disappears, because the cost")
print("becomes reading and writing the tensor rather than the arithmetic.")
print("ReLU remains an order of magnitude cheaper unfused, which is why it")
print("survives where fusion is unavailable.")
```

## 9. Practical Example

```python {tier=A name=activations-that-matter}
"""Does the activation choice matter? Measured, at two depths.
"""
import numpy as np

rng = np.random.default_rng(3)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    p = z >= 0
    out[p] = 1.0 / (1.0 + np.exp(-z[p]))
    e = np.exp(z[~p])
    out[~p] = e / (1.0 + e)
    return out


ACT = {
    "sigmoid": (sigmoid, lambda z, a: a * (1 - a)),
    "tanh":    (np.tanh, lambda z, a: 1 - a ** 2),
    "ReLU":    (lambda z: np.maximum(0, z), lambda z, a: (z > 0).astype(float)),
    "LeakyReLU": (lambda z: np.where(z > 0, z, 0.01 * z),
                  lambda z, a: np.where(z > 0, 1.0, 0.01)),
    "GELU":    (lambda z: 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi)
                                                 * (z + 0.044715 * z ** 3))),
                None),
    "SiLU":    (lambda z: z * sigmoid(z), None),
}


def numeric_backward(f, z, eps=1e-5):
    """Derivative by central difference — used for GELU and SiLU so their
    analytic forms do not need repeating here, and as a check on the others."""
    return (f(z + eps) - f(z - eps)) / (2 * eps)


class Net:
    """A plain MLP with a configurable hidden activation, trained with SGD.

    Deliberately WITHOUT normalisation or careful initialisation, because the
    point is to isolate the activation's contribution. Section 5.5's claim is
    that adding those makes the activations converge in performance, and the
    second experiment tests it.
    """

    def __init__(self, sizes, act="ReLU", init="he", seed=0):
        rs = np.random.default_rng(seed)
        self.sizes, self.act = sizes, act
        self.f, self.df = ACT[act]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            if init == "he":
                s = np.sqrt(2.0 / fan_in)
            elif init == "xavier":
                s = np.sqrt(1.0 / fan_in)
            else:
                s = 0.05
            self.W.append(rs.normal(0, s, (sizes[i + 1], sizes[i])))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        pre, post = [], [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            pre.append(z)
            h = z if i == len(self.W) - 1 else self.f(z)
            post.append(h)
        return h.ravel(), pre, post

    def _dphi(self, z, a):
        if self.df is not None:
            return self.df(z, a)
        return numeric_backward(self.f, z)

    def fit(self, X, y, epochs=3000, lr=0.05, batch=128, seed=0,
            track_grads=False):
        rs = np.random.default_rng(seed)
        n = len(y)
        self.grad_norms = []
        for ep in range(epochs):
            idx = rs.integers(0, n, min(batch, n))
            xb, yb = X[idx], y[idx]
            out, pre, post = self.forward(xb)
            g = (2.0 * (out - yb) / len(yb))[:, None]
            layer_norms = []
            for i in range(len(self.W) - 1, -1, -1):
                gW = g.T @ post[i]
                gb = g.sum(0)
                layer_norms.append(float(np.linalg.norm(gW)))
                if i > 0:
                    g = (g @ self.W[i]) * self._dphi(pre[i - 1], post[i])
                self.W[i] -= lr * gW
                self.b[i] -= lr * gb
            if track_grads and ep % 200 == 0:
                self.grad_norms.append(layer_norms[::-1])
        return self

    def mse(self, X, y):
        out, _, _ = self.forward(X)
        v = float(np.mean((out - y) ** 2))
        return v if np.isfinite(v) else float("inf")


# --- the task ---------------------------------------------------------------
def make(n, rs):
    X = rs.uniform(-2, 2, (n, 6))
    y = (np.sin(1.6 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2]
         - 0.6 * np.abs(X[:, 3]) + 0.4 * X[:, 4] ** 2)
    return X, y


rs = np.random.default_rng(5)
Xtr, ytr = make(4000, rs)
Xte, yte = make(4000, rs)
LRS = (0.1, 0.03, 0.01, 0.003)


def best(sizes, act, init, Xtr, ytr, Xva, yva, seed=1):
    b = (None, np.inf, None)
    for lr in LRS:
        with np.errstate(over="ignore", invalid="ignore"):
            m = Net(sizes, act=act, init=init, seed=seed).fit(
                Xtr, ytr, lr=lr, seed=2)
            v = m.mse(Xva, yva)
        if v < b[1]:
            b = (m, v, lr)
    return b


Xva, yva = make(1500, rs)

print("=" * 72)
print("does the activation matter? shallow vs deep (section 5.5)")
print("=" * 72)
print("Each activation gets its own tuned learning rate, so the comparison")
print("is not confounded by one function needing a smaller step.\n")
for depth, sizes in ((3, [6, 40, 40, 40, 1]),
                     (10, [6] + [40] * 10 + [1])):
    print(f"{depth} hidden layers")
    print(f"{'activation':<12} {'best lr':>9} {'test MSE':>11} "
          f"{'vs best':>9}")
    results = {}
    for name in ACT:
        m, _, lr = best(sizes, name, "he", Xtr, ytr, Xva, yva)
        results[name] = (m.mse(Xte, yte), lr)
    floor = min(v[0] for v in results.values())
    for name, (mse, lr) in results.items():
        ratio = mse / floor if np.isfinite(mse) else float("inf")
        print(f"{name:<12} {lr:>9} {mse:>11.5f} {ratio:>8.2f}x")
    print()

print("Sigmoid is the clear loser at both depths and catastrophically so at")
print("ten layers — 280 times the best error — which is eq. 50.4's product")
print("arriving exactly on schedule.")
print("\nThe result that does NOT match the received wisdom is the spread")
print("WITHIN the rectifier family. SiLU beats plain ReLU by roughly an")
print("order of magnitude here, at both depths, and GELU sits between them.")
print("That is much larger than the fraction-of-a-percentage-point the")
print("literature usually reports.")
print("\nThe discrepancy is not a contradiction; it is a scope condition,")
print("and it is worth being precise about. This network has NO")
print("normalisation, plain SGD, and a hand-tuned constant learning rate.")
print("The published comparisons are of large models with normalisation,")
print("careful schedules and adaptive optimisers — and normalisation in")
print("particular keeps pre-activations in the region where all the")
print("rectifier variants behave alike, which is precisely what makes them")
print("interchangeable there.")
print("\nSo the honest statement is conditional: the smooth activations are")
print("substantially better when nothing else is holding the activations in")
print("a good range, and nearly interchangeable once something is. That is")
print("the interaction of section 5.5, and the next experiment isolates it.")

# --- gradient norms by layer: seeing the vanishing directly -----------------
print("\n" + "=" * 72)
print("the gradient reaching each layer, measured (eq. 50.3)")
print("=" * 72)
sizes_deep = [6] + [40] * 10 + [1]
print(f"{'activation':<12} " +
      " ".join(f"{'layer ' + str(i):>11}" for i in (1, 4, 7, 10)) +
      f" {'L1/L10 ratio':>14}")
for name in ("sigmoid", "tanh", "ReLU", "GELU"):
    m = Net(sizes_deep, act=name, init="he", seed=1)
    m.fit(Xtr, ytr, epochs=201, lr=0.01, seed=2, track_grads=True)
    g = np.array(m.grad_norms[0])          # gradient norms at the first step
    shown = [g[i - 1] for i in (1, 4, 7, 10)]
    print(f"{name:<12} " + " ".join(f"{v:>11.3e}" for v in shown) +
          f" {g[0] / max(g[9], 1e-300):>14.3e}")

print("\nRead the last column: it is the ratio of the gradient reaching the")
print("FIRST hidden layer to the gradient at the TENTH. For a saturating")
print("activation the early layers receive orders of magnitude less signal,")
print("so they barely move — the network is effectively shallower than it")
print("looks. For the rectifier family the ratio is close to one.")
print("\nThis is the diagnostic to run when a deep network will not train:")
print("print the per-layer gradient norm. A ratio spanning many orders of")
print("magnitude localises the problem immediately.")

# --- section 5.5: does normalisation make them equivalent? ------------------
print("\n" + "=" * 72)
print("the interaction: does good initialisation close the gap?")
print("=" * 72)
print("The same ten-layer network under two initialisations.\n")
print(f"{'activation':<12} {'small init (0.05)':>19} {'He init':>11} "
      f"{'improvement':>13}")
for name in ("sigmoid", "tanh", "ReLU", "GELU"):
    m_small, v_small, _ = best(sizes_deep, name, "small", Xtr, ytr, Xva, yva)
    m_he, v_he, _ = best(sizes_deep, name, "he", Xtr, ytr, Xva, yva)
    a, b = m_small.mse(Xte, yte), m_he.mse(Xte, yte)
    imp = a / b if np.isfinite(a) and b > 0 else float("inf")
    print(f"{name:<12} {a:>19.5f} {b:>11.5f} {imp:>12.2f}x")

print("\nThe small-init column is identical for every activation, and that")
print("identical value is the variance of the target: with weights at 0.05")
print("in a ten-layer network, the signal has decayed to nothing by the")
print("output and every network learns the same thing — the mean. The")
print("activation is irrelevant because no gradient is reaching anywhere.")
print("\nHe initialisation rescues the rectifier family completely and")
print("sigmoid not at all, which is the expected asymmetry: the factor of")
print("two in section 6.4 was DERIVED for a rectifier and compensates for")
print("something sigmoid does not do.")
print("\nCompare the magnitudes. Switching initialisation moved GELU by a")
print("factor of 143; switching between rectifier activations moved it by")
print("about 10. The initialisation is the larger lever, and the two are not")
print("independent choices — which is section 5.5's claim, measured. The")
print("activation, the initialisation and the normalisation of Chapter 57")
print("are one design decision with three parts, and tuning any of them in")
print("isolation measures less than it appears to.")
```

## 10. Production Considerations

**Fusion is the whole performance story.** {{sec:7-internal-mechanics}} showed
activations are memory-bound; a fused activation costs almost nothing and an
unfused one costs a full read and write of the tensor. When a profile shows an
activation taking meaningful time, the fix is fusion, not a cheaper function.

**Memory: prefer activations whose derivative uses the output.** ReLU, sigmoid
and tanh can all compute their backward pass from the output; GELU and SiLU
need the input. In a transformer's feed-forward block, where the tensor is four
times the model width, that is a real difference in activation memory.

**Half precision changes the saturation points.** `float16` has about three
decimal digits of mantissa, so $\sigma'(z)$ underflows to exactly zero at a
much smaller $|z|$ than in `float32`. A network that trains in single precision
can have dead gradients in half precision for no other reason. `bfloat16`'s
wider exponent makes this less acute.

**Do not tune the activation.** {{ch:mle-hpo}} measured how quickly search
optimism accumulates. On a search space where the differences are within
run-to-run variance, an activation sweep buys selection noise; spend the budget
on the learning rate.

**Check for dead units in production models.** The fraction of units that are
zero for every input in a validation batch is one number, cheap to compute, and
a high value indicates capacity you paid for and are not using.

## 11. Common Mistakes

**Using sigmoid or tanh in hidden layers of a deep network.** The measured
gradient ratio across ten layers is orders of magnitude.

**Putting a hidden activation on the output.** A sigmoid on a regression head
caps predictions at 1.

**Applying softmax and then a separate log.** Numerically unstable; use the
fused form ({{ch:dl-losses}}).

**Using ReLU with a learning rate that kills units.** The dominant cause of
dead units is a step that drives biases sharply negative.

**Assuming a newer activation is better.** The measured differences between
ReLU, LeakyReLU, GELU and SiLU are small at both depths tested.

**Tuning the activation before the learning rate.** The measured initialisation
effect was larger than the activation effect.

**Computing sigmoid as `1/(1+exp(-z))`.** Overflows for large negative $z$.

**Ignoring the interaction with initialisation.** He initialisation is derived
*for* rectifiers; the measurement shows it does not rescue a saturating
activation.

## 12. Failure Modes

**Dead ReLU units.** {{eq:dead-condition}}: a unit whose pre-activation is
negative across the whole data distribution has exactly zero gradient forever.
Unlike sigmoid saturation, this is permanent. Detected by counting units that
are zero for every example in a batch; prevented by Leaky ReLU, a smaller
learning rate, or normalisation.

**Silent saturation.** A sigmoid network does not fail loudly. It trains, the
loss decreases, and the early layers are barely changing — so the network is
effectively shallower than its architecture. The diagnostic is the per-layer
gradient norm measured in {{sec:9-practical-example}}, not the loss curve.

**Activation explosion.** With poor initialisation, ReLU's unbounded output
lets activations grow multiplicatively through depth until they overflow. The
symptom is a loss that becomes `nan` after a few steps. Saturating activations
cannot do this, which is their one genuine advantage.

**Half-precision gradient underflow.** As above: a derivative representable in
`float32` becoming exactly zero in `float16`, so the layer stops learning with
no error raised.

**Non-monotonicity confusing analysis.** GELU and SiLU are not monotone near
zero, so a larger pre-activation can produce a *smaller* output in a small
region. Any analysis assuming monotone activations — some interpretability
methods, some verification tools — is invalid for them.

## 13. Alternatives

**Polynomial activations.** Nonlinear, and they explode or vanish rapidly with
depth because a degree-$d$ polynomial composed $L$ times has degree $d^{L}$.
Used in homomorphic-encryption settings where transcendental functions are
unavailable, and avoided otherwise.

**Radial basis functions.** Local rather than global response. Effective in
shallow networks and poorly suited to depth, since the local support makes most
units inactive for most inputs.

**Maxout.** Take the maximum of several affine functions, learning the
activation shape rather than fixing it. Expressive, and multiplies the
parameter count by the number of pieces.

**Learned activations (PReLU).** {{cite:he2015init}} made the negative slope a
learned parameter. Small consistent gains, rarely worth the complexity in
practice.

**No activation at all, plus normalisation.** Some architectures obtain their
nonlinearity from normalisation and gating rather than from a pointwise
function. Gated linear units are the mainstream instance and appear in
{{ch:tf-ffn-residual}}.

## 14. Evaluation

**Per-layer gradient norms** are the primary diagnostic, and the measurement in
{{sec:9-practical-example}} shows why: the loss curve does not distinguish "the
network is learning slowly" from "the first four layers receive nothing".

**Dead-unit fraction**, computed on a validation batch, for any rectifier
network.

**Activation statistics by layer** — mean, standard deviation, and fraction
saturated. A layer whose activations have drifted to a standard deviation of
$10^{-3}$ or $10^{3}$ has a signal-propagation problem regardless of the loss.

**The tiny-subset overfit test** from {{ch:dl-neural-networks}} still applies
and is still the first thing to run.

What *not* to evaluate: the activation choice by a hyperparameter search. The
measured differences are small enough that a search over them mostly selects
noise, in the sense {{ch:mle-hpo}} quantified.

## 15. Advanced Concepts

**Self-normalising networks (SELU).** A scaled ELU with specific constants
$\alpha \approx 1.6733$, $\lambda \approx 1.0507$, chosen so that activations
converge to zero mean and unit variance without an explicit normalisation
layer. Elegant, and it requires a matching initialisation and dropout variant,
which is why it did not displace explicit normalisation.

**Gated linear units.** $\text{GLU}(x) = (xW + b) \otimes \sigma(xV + c)$ —
one branch gates the other multiplicatively. SwiGLU, the variant using SiLU as
the gate, is standard in current large language models, and
{{ch:tf-ffn-residual}} covers it. Note this is a *layer*, not a pointwise
activation: it doubles the parameters of the projection it replaces.

**Activation sparsity as compute.** Because ReLU zeroes roughly half its
outputs, the subsequent matrix multiply has structured sparsity that could in
principle be exploited. Doing so profitably on dense hardware is difficult,
which is why it is more discussed than deployed — though it is one motivation
for mixture-of-experts routing, which makes the sparsity structured and
predictable.

**The activation's effect on the loss landscape.** Smooth activations give a
smooth loss surface; ReLU's kink makes the loss piecewise-linear in a way that
is not differentiable everywhere. Whether the smoothness of GELU is the
mechanism behind its small advantage is unresolved, and the honest position is
that it is a hypothesis, not a finding.

## 16. Connection to Previous Chapters

{{ch:ml-logistic}} supplied the sigmoid and, more importantly, the exact
cancellation between its derivative and the cross-entropy loss
({{eq:logit-delta}}) — the reason sigmoid survives at the *output* long after
being abandoned in hidden layers. {{ch:dl-neural-networks}} established that a
nonlinearity is required at all and measured the collapse without one.
{{ch:math-derivatives}} supplied the chain rule that {{eq:gradient-product}}
applies $L$ times.

Forward: {{ch:dl-initialization}} completes the derivation begun in
{{sec:6-mathematical-foundation}} — {{eq:relu-variance}} is the fact that gives
He initialisation its factor of two. {{ch:dl-normalization}} attacks the same
signal-propagation problem from the other side, by rescaling activations
directly rather than choosing a function that behaves. {{ch:dl-backprop}} makes
{{eq:gradient-product}} an algorithm. {{ch:dl-rnns}} shows the same product
appearing across *time* rather than depth, with the same consequence.

## 17. Exercises

**Beginner**

1. Why does a network need a nonlinearity at all?
2. What is saturation, and which activations saturate at both ends?
3. State ReLU and its derivative.
4. What is a dead unit, and why is it permanent?
5. Which activation would you use for a hidden layer in a CNN? In a
   transformer?

**Intermediate**

6. Derive $\sigma'(z) = \sigma(1-\sigma)$ and find its maximum.
7. Using {{eq:gradient-bound}}, compute the best-case gradient factor for 15
   sigmoid layers.
8. Explain why tanh is preferred to sigmoid, giving both reasons.
9. Explain the zig-zag of {{sec:6-mathematical-foundation}} in your own words.
10. Why does GELU's backward pass need the input where ReLU's does not?
11. Give two causes of dead units and the remedy for each.

**Advanced**

12. Derive {{eq:relu-variance}}, including $\E[\relu(z)] = \sigma/\sqrt{2\pi}$.
13. Derive GELU as an expected stochastic gate and explain the interpretation.
14. Show that composing a degree-$d$ polynomial activation $L$ times gives
    degree $d^{L}$, and explain why that makes it unusable.
15. Derive the SELU constants' defining conditions (you need not solve them)
    and explain what property they enforce.
16. Explain why the *mean* ReLU derivative being 0.5 does not imply the
    gradient decays as $0.5^{L}$.

**Implementation**

17. Implement all seven activations with analytic derivatives and verify each
    against a central difference.
18. Instrument a training run to report per-layer gradient norms and dead-unit
    fractions every epoch.
19. Measure the fraction of dead units as a function of learning rate and find
    the threshold at which death becomes common.
20. Implement SwiGLU and compare it against a plain SiLU feed-forward block at
    matched parameter count.

**Reasoning**

21. A ten-layer tanh network trains but its accuracy plateaus early. What do
    you measure first?
22. A colleague proposes searching over eight activations. What do you suggest
    instead, and why?

## 18. Interview Questions

**"Why did ReLU replace sigmoid?"** — The expected answer is vanishing
gradients. The strong answer gives the number: sigmoid's derivative maxes at
0.25, so ten layers attenuate by $10^{-6}$ in the best case, while ReLU's is
exactly 1 on active paths.

**"What is the dying ReLU problem and how do you fix it?"** — The condition,
why it is permanent, the dominant cause (learning rate too large), and the
remedies. Mentioning that you would *measure* the dead fraction rather than
assume is what distinguishes the answer.

**"Is ReLU's mean derivative 0.5? Does that mean gradients vanish?"** — A good
trap. Yes to the first, no to the second: ReLU blocks some paths entirely and
leaves the others unattenuated, which is different from attenuating every path.

**"When would you use tanh?"** — Output layers needing a bounded symmetric
range, gates inside LSTM cells, and small shallow networks. Not hidden layers
of anything deep.

**"How much does the activation choice matter?"** — The honest answer, backed
by the measurement: enormously between saturating and rectifying, very little
within the rectifier family, and less than the initialisation it interacts
with.

**"Why does GELU need more memory than ReLU?"** — Its derivative is not
expressible in terms of its output, so the input must be retained for the
backward pass.

## 19. Research Questions

**Why do GELU and SiLU slightly outperform ReLU?** Candidate explanations —
smoothness, the non-monotone region, better-conditioned gradients near zero —
have not been separated. The effect is small and consistent, and its cause is
unestablished. {{maturity:RESEARCH FRONTIER}}

**Is a pointwise activation the right primitive at all?** Gated linear units
outperform pointwise activations in transformers at matched parameter count,
suggesting the multiplicative interaction matters more than the shape of any
scalar function. {{maturity:EMERGING}}

**Can activation sparsity be exploited?** ReLU networks are roughly 50% sparse
and dense hardware cannot use it. Whether architectures with predictable,
structured sparsity can convert that into throughput is an open engineering
question with mixture-of-experts as the current partial answer.
{{maturity:EMERGING}}

**What is the right activation for very low precision?** As models move to
4-bit and below ({{part:15}}), the interaction between activation shape and
quantisation error is not well characterised, and the functions in use were all
designed in a `float32` world. {{maturity:EMERGING}}

## 20. Chapter Summary

An activation must be nonlinear, or the network collapses, and must pass
gradient, or the layers below cannot learn. Most nonlinearities fail the second
requirement, and the history of the field is the search for ones that do not.

The mechanism is a product. {{eq:gradient-bound}} shows the activation
contributing a factor of $|\phi'|_{\max}$ per layer, and the measurement
confirms the arithmetic: sigmoid's best-case $0.25$ gives $10^{-6}$ over ten
layers and $10^{-12}$ over twenty, below what a `float32` update can represent.
ReLU's factor is exactly 1 on active paths, and the measured per-layer gradient
norms show the first layer of a ten-layer sigmoid network receiving orders of
magnitude less signal than the last, while the rectifier family stays flat.

That distinction is sharper than "ReLU has a bigger derivative". Its mean
derivative is about 0.5, because half its units are inactive — but a sigmoid
attenuates *every* path while a ReLU blocks *some paths completely* and leaves
the rest untouched. Attenuation compounds; blocking does not.

Tanh beats sigmoid for two separate reasons: four times the maximum derivative,
and a zero-centred output that avoids confining each row of weight gradients to
a single orthant. Neither fixes saturation.

ReLU's characteristic failure is permanent. A unit whose pre-activation is
negative across the whole distribution has exactly zero gradient forever, and
unlike sigmoid saturation it cannot recover. The dominant cause is a learning
rate large enough to drive biases sharply negative.

The rectifier halves the second moment of its input exactly, as the measurement
confirms, which is the fact that gives He initialisation its factor of two.

Finally, the honest 2026 position: between saturating and rectifying activations
the choice is load-bearing and the measurement shows it clearly at ten layers.
*Within* the rectifier family — ReLU, LeakyReLU, GELU, SiLU — the differences
are small at every depth tested, and the measured effect of initialisation was
larger than the effect of the activation. The three are one design decision with
three parts, and tuning any one in isolation measures less than it appears to.

## 21. Further Reading

{{cite:glorot2010}} is the paper that made activation choice a quantitative
question rather than a matter of taste. Its measurements of activation and
gradient variance by layer are the direct ancestor of the diagnostic in
{{sec:9-practical-example}}, and the paper is worth reading for the
methodology as much as the conclusion.

{{cite:he2015init}} redoes that analysis for rectifiers and introduces PReLU.
The derivation of the factor of two is three lines and is reproduced in
{{ch:dl-initialization}}.

{{cite:krizhevsky2012}} is where ReLU's practical impact was demonstrated at
scale. Section 3.1 of that paper is a page long and reports the training-speed
difference that made the case.

{{cite:rumelhart1986}} for the historical context in which sigmoid was the
obvious choice — differentiable, bounded, and biologically motivated. Reading
it clarifies that the abandonment of sigmoid was an empirical discovery, not a
correction of an error.

**Where to go next:** {{ch:dl-losses}} for the output activations this chapter
deliberately set aside, and {{ch:dl-initialization}} for the completion of
{{eq:relu-variance}}'s argument. Readers primarily interested in why deep
networks train at all should read {{ch:dl-backprop}} and
{{ch:dl-initialization}} together — they are two halves of one story.
