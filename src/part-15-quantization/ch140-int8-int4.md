---
id: q-int8-int4
number: 140
part: XV
tier: full
status: draft
requires: [q-theory, q-formats, tf-ffn-residual]
provides: [emergent-outliers, difficulty-migration, activation-aware-importance,
           incoherence-processing, compensate-not-round, calibration-is-a-parameter,
           reduction-axis-constraint]
citations: [dettmers2022int8, xiao2023smoothquant, lin2023awq, frantar2023gptq,
            tseng2024quipsharp, dettmers2023spqr, egiazarian2024aqlm]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what emergent outliers are
and why they appear only above a scale threshold; explain the **four distinct
answers** to them and what each assumes; say why the method that performs worst in
a numerical comparison is the one production uses, and what constraint decides
that; implement error-compensated rounding and explain why it deliberately chooses
worse weights; and recognise the calibration set as **a hyperparameter that can
make things worse**.

## 2. Why This Matters

{{ch:q-theory}} showed that an outlier's damage is a statement about the step
size, and that the remedy is therefore local. **This chapter is the four local
remedies**, and they are genuinely different ideas.

{{sec:9-practical-example}} puts all four on one layer. **Six outlier activation
channels out of 256 raise the naive 8-bit output error from 0.0166 to 0.0581 — a
factor of 3.5.** Then:

```text
   method                              8 bits    4 bits
   ─────────────────────────────       ──────    ──────
   naive: one scale each               0.0581    0.3524
   per-channel activation scales       0.0109    0.2011
   mixed precision for outliers        0.0043    0.0785
   SmoothQuant migration               0.0186    0.3283
   AWQ salient-channel scaling         0.0191    0.3344
   Hadamard rotation                   0.0148    0.2690
```

**The two most-cited methods are the two worst performers here**, and that needs
an explanation rather than a shrug. It is the chapter's most useful lesson: **an
INT8 matmul requires one scale for the whole reduction, so a per-channel
activation scale — the best-performing fix — cannot be folded into the kernel that
makes INT8 worth using.** SmoothQuant exists to move that scale onto the weights,
where it becomes a compile-time constant.

Then the second idea, which is about **rounding rather than scaling**.
{{cite:frantar2023gptq}} rounds each weight to a value that is *not* the nearest
one, in order to cancel errors already made. Measured: GPTQ is **worse on weight
error** (0.2985 against 0.1696 at 4 bits) and **better on output error** (0.1273
against 0.1725) — **1.36× better on the metric that matters, by being worse on the
one that does not.**

**And the warning that comes with it.** With the true input covariance, GPTQ
reaches **0.1141**. With an isotropic calibration — right size, wrong shape —
**0.1603**, which is **worse than round-to-nearest with no calibration at all
(0.1578)**.

> **Mis-calibrated error compensation is worse than none**, and nothing in the
> output of a quantization run says so.

{{maturity:ESTABLISHED}} INT8, GPTQ, AWQ. {{maturity:MATURE}} Rotation-based
methods. {{maturity:EMERGING}} Sub-3-bit representations.

## 3. Prerequisites

{{ch:q-theory}} for {{eq:outlier-inflates-the-step}} and
{{eq:effective-levels}}, which say what an outlier costs; {{ch:q-formats}} for
{{eq:scale-group-condition}}; {{ch:tf-ffn-residual}} for where in a transformer
these tensors live.

## 4. Intuitive Explanation

### The phenomenon, and why it emerges

{{cite:dettmers2022int8}} found that above a scale threshold, a small number of
**activation** feature dimensions carry values far larger than the rest — and that
they are *systematic*, appearing in the same dimensions across inputs and
dominating the model's predictions.

**Two properties make them the central problem of this chapter.** They are in the
activations, not the weights, so they are data-dependent and cannot be inspected
from a checkpoint alone. And they are **emergent**: a method validated on a 1B
model can fail outright on a 70B one, which is why quantization results are scale-
qualified in a way most ML results are not.

By {{ch:q-theory}}'s {{eq:effective-levels}}, a 16× outlier leaves 4-bit
neighbours **fewer than one effective level**. The bits are spent on a range
nothing occupies.

### Four answers, and they are not variations

**Isolate them** ({{cite:dettmers2022int8}}). Keep the outlier dimensions in
16-bit and quantize the rest. Best measured error (**0.0043** at 8 bits) — it
removes the problem rather than solving it. Costs a second precision path in the
kernel.

**Migrate them** ({{cite:xiao2023smoothquant}}). Divide activations by a
per-channel constant and multiply the corresponding weight *rows* by the same
constant. **The function is exactly unchanged**; the activation tensor the
quantizer sees no longer has outliers, and the weight tensor, which had range to
spare, absorbs them.

**Protect them** ({{cite:lin2023awq}}). Find the salient channels from
**activation** statistics rather than weight magnitudes, and scale them so they are
represented more finely. The insight that transfers: **a small weight multiplying
a large activation matters more than a large weight multiplying a small one**, and
only the data can tell you which is which.

**Destroy the basis** ({{cite:tseng2024quipsharp}}). Rotate with a Hadamard
transform so no coordinate is special. An orthogonal mixing spreads each channel's
energy across all of them, **turning a spiky distribution into an approximately
Gaussian one** — which is exactly what a uniform quantizer handles well.

### Why the best method is not the one used

Per-channel activation scales score **0.0109** at 8 bits against SmoothQuant's
**0.0186**. So why is SmoothQuant the one in production?

**Because an INT8 matrix multiply computes a dot product along the reduction
axis, and one scale factor must apply to that entire dot product.** A per-channel
activation scale varies *along* the reduction axis. It is expressible in numpy and
not in the kernel that makes INT8 fast.

**That is precisely what SmoothQuant fixes.** After migration, the surviving
per-channel factor lives on the **weights**, where it is folded in once at
quantization time rather than applied at runtime inside the reduction.

> **So the ranking by error is the wrong ranking.** These methods are chosen on
> kernel cost, and reading the error column alone inverts the answer.

### Rounding is a choice

Every quantizer so far rounded each weight to its nearest representable value.
**That minimises weight error — and weight error is not what you want.**

The weights are not independent in their effect: an error in one can be partly
cancelled by deliberately mis-rounding another. {{cite:frantar2023gptq}} turns
that into an algorithm — quantize in order, and after each weight push its
rounding error into the ones not yet quantized, **weighted by the input covariance
so the compensation is aimed where the data has variance.**

Measured at 4 bits: weight error **0.2985** against RTN's 0.1696 (worse), output
error **0.1273** against 0.1725 (better by 1.36×).

**GPTQ chooses worse weights to get a better function.**

### And the calibration set is where it gets its information

Which makes the calibration data load-bearing:

```text
   calibration                       output error   vs correct
   ──────────────────────────────    ────────────   ──────────
   the true input covariance               0.1141        1.00×
   noisy copy of the real data             0.1297        1.14×
   isotropic (wrong shape)                 0.1603        1.40×
   64 samples of the real data             0.2167        1.90×
   RTN, no calibration at all              0.1578        1.38×
```

**Noise is nearly free** — the method needs the *shape* of the input distribution
and tolerates a lot of noise in it, which matches the folklore that a few hundred
calibration sequences suffice.

**The wrong shape is not free, and it is worse than doing nothing.** Generic web
text for a model serving code, English for a model serving another language, short
sequences for a long-context deployment: **these are not slightly worse
calibration sets. They can be actively harmful ones**, and the quantization run
reports success either way.

## 5. Formal Explanation

### 5.1 The outlier condition

For an activation tensor with typical magnitude $m$ and outlier magnitude $M$ in a
scale group, {{ch:q-theory}}'s {{eq:effective-levels}} gives

$$ n_{\text{eff}} = \frac{m}{M}\,(2^{b}-2) $$

**Quantization is viable when $n_{\text{eff}} \gtrsim 4$**, so the condition on the
outlier ratio is

$$ \frac{M}{m} \lesssim \frac{2^{b}-2}{4} $$ (eq:outlier-budget)

At $b = 8$ that permits a ratio of about 60; at $b = 4$, about 3.5.
**{{eq:outlier-budget}} is why INT8 tolerated outliers for years and INT4 did
not**, and why the phenomenon became urgent exactly when 4-bit did.

### 5.2 Migration is an exact reparameterisation

For a diagonal $S = \text{diag}(s_1, \dots, s_k)$,

$$ XW = (XS^{-1})(SW) $$ (eq:difficulty-migration)

**{{eq:difficulty-migration}} changes nothing about the function.** It changes only
what the quantizer sees: $XS^{-1}$ has its outlier channels divided down and $SW$
has the corresponding weight rows scaled up.

{{cite:xiao2023smoothquant}} chooses

$$ s_j = \frac{(\max_i |X_{ij}|)^{\alpha}}{(\max_i |W_{ji}|)^{1-\alpha}} $$ (eq:smoothquant-scale)

with $\alpha$ balancing how much difficulty is moved. **The search over $\alpha$ is
part of the method**, not a refinement of it.

### 5.3 Importance is a property of the data

For the layer output $Y = XW$, perturbing $W_{ji}$ by $\delta$ changes $Y$ by
$\delta \cdot X_{:,j}$. So

$$ \frac{\partial \|Y\|}{\partial \delta_{ji}} \;\propto\; \|X_{:,j}\| $$ (eq:output-error-is-the-target)

**{{eq:output-error-is-the-target}} is {{cite:lin2023awq}}'s whole argument**: a
weight's importance is set by the *activation* it multiplies, not by its own
magnitude. Weight-magnitude pruning and weight-magnitude quantization both make
the same category error.

### 5.4 Rotation makes every coordinate ordinary

For orthogonal $Q$,

$$ XW = (XQ)(Q^{\top}W), \qquad \|XQ\|_F = \|X\|_F $$ (eq:incoherence-processing)

so the transform is free in the Frobenius sense, and it redistributes the energy.
For a random-ish orthogonal $Q$, each output coordinate is a sum of $k$ input
coordinates, so by the central limit theorem

$$ \max_j |(XQ)_{ij}| \;\approx\; \sigma\sqrt{2\log k} \quad\text{rather than}\quad M $$ (eq:rotation-flattens)

**{{eq:rotation-flattens}} replaces the outlier ratio $M/m$ with
$\sqrt{2\log k}$**, which for $k = 4096$ is about 4.1 — comfortably inside
{{eq:outlier-budget}} even at 4 bits. The cost is a transform on the critical path,
which is why the Hadamard construction matters: it is $O(k\log k)$ and needs no
multiplies.

### 5.5 Error compensation

Minimise the output error rather than the weight error:

$$ \min_{Q} \big\| XW - XQ \big\|_F^2 = \min_{Q} \text{tr}\big[(W-Q)^{\top} H (W-Q)\big], \qquad H = X^{\top}X $$ (eq:compensate-not-round)

**{{eq:compensate-not-round}} is a different objective from $\min\|W - Q\|$**, and
round-to-nearest solves the second. {{cite:frantar2023gptq}} solves the first
greedily: quantize row $i$, then update the remaining rows by

$$ W_{i+1:} \;\mathrel{-}= \frac{w_i - q_i}{[U]_{ii}} \, U_{i,\,i+1:}, \qquad U = \text{chol}\big(H^{-1}\big)^{\top} $$ (eq:gptq-update)

**The $H^{-1}$ weighting is what aims the compensation.** Directions the data
occupies heavily get small corrections because their effect is large; directions
it barely occupies get large ones, because there the correction is nearly free.

> **IMPORTANT:** {{eq:gptq-update}} is only as good as $H$. An $H$ estimated from
> the wrong distribution aims the compensation at directions the real data does
> not use — spending rounding freedom to fix errors that do not matter while
> creating ones that do. **That is why mis-calibration is worse than no
> compensation**, and it is a failure with no symptom at quantization time.

### 5.6 The reduction-axis constraint

An INT8 GEMM computes

$$ Y_{ij} = s_X s_W \sum_{k} \hat{X}_{ik}\hat{W}_{kj} $$ (eq:reduction-axis-constraint)

with $\hat{X}, \hat{W}$ integer. **A scale that varies with $k$ cannot be factored
out of the sum.** Per-*row* scales on $X$ (varying with $i$) and per-*column*
scales on $W$ (varying with $j$) both factor out; per-channel scales on $X$
(varying with $k$) do not.

**{{eq:reduction-axis-constraint}} is why the best-performing fix is unusable and
{{eq:difficulty-migration}} exists.**

## 6. Mathematical Foundation

### 6.1 The outlier budget, checked

From {{eq:outlier-budget}} at $b = 8$: $M/m \lesssim 63$. The measured layer has
$M/m = 24$ — inside the budget — and naive 8-bit still degraded **3.5×**, from
0.0166 to 0.0581.

**The discrepancy is instructive.** {{eq:outlier-budget}} asks whether the
ordinary values are *representable*; the measured error asks how *accurately*.
Being inside the budget means quantization does not collapse, not that it is free.
At $b = 4$ the budget is $M/m \lesssim 3.5$ and the measurement is at 24 — **seven
times over**, which is why the 4-bit naive column reads 0.3524.

### 6.2 Why migration cannot go too far

Pushing all the difficulty onto the weights makes them the problem. With
$\alpha = 1$, {{eq:smoothquant-scale}} flattens the activations completely and
inflates the weight range by exactly $M/m$. The total error is roughly

$$ \varepsilon_{\text{tot}}^2 \;\propto\; \Big(\frac{M_X^{1-\alpha}}{\cdot}\Big)^2 + \big(M_W^{\alpha}\big)^2 $$

which has an interior minimum. **That is why $\alpha$ is searched and why
$\alpha \approx 0.5$ is a common default** — it is the balance point of a product,
not a tuning accident.

### 6.3 Why GPTQ loses on weight error by exactly as much as it must

Round-to-nearest achieves $\|W - Q\|$ within $\Delta/2$ per element, which is the
minimum. Any $Q$ that improves {{eq:compensate-not-round}} must move at least one
weight further than $\Delta/2$, so

$$ \|W - Q_{\text{GPTQ}}\| \;\ge\; \|W - Q_{\text{RTN}}\| $$ (eq:weight-error-floor)

**with equality only if the compensation does nothing.** So the measured 0.2985
against 0.1696 is not a defect — **it is evidence the method is doing something**,
and a GPTQ implementation whose weight error matches RTN's is a broken one.

> **MATH NOTE:** {{eq:gptq-update}} propagates error *forward* through an
> ordering, so the last rows have no one left to compensate into and carry the
> accumulated residual. Group-wise application resets the scale but not the
> ordering, which is why the measured 3-bit group-64 result (**0.3221**) is
> *worse* than tensor-wide GPTQ (**0.2994**): at 3 bits there is not enough
> headroom for the compensation to work within a short group. **These techniques
> interact, and "apply both" is a default rather than a theorem.**

## 7. Internal Mechanics

```mermaid {#fig:four-answers caption="One problem and four remedies. Outliers violate the outlier budget (eq:outlier-budget) by inflating a shared scale. Isolation removes them, migration moves them to a tensor with room (eq:difficulty-migration), salient-channel scaling protects them using activation statistics (eq:output-error-is-the-target), and rotation destroys the basis in which they are special (eq:rotation-flattens). The choice between them is made on kernel cost — specifically the reduction-axis constraint (eq:reduction-axis-constraint) — not on error."}
flowchart TB
    O["emergent outlier channels<br/>in the ACTIVATIONS"] --> V{{"violates<br/>eq:outlier-budget"}}
    V --> A["isolate: keep them in fp16<br/>cost: two precision paths"]
    V --> B["migrate: X S^-1 times S W<br/>cost: a calibrated alpha"]
    V --> C["protect: scale salient channels<br/>cost: a calibration set"]
    V --> D["rotate: X Q times Q^T W<br/>cost: a transform on the path"]
    K{{"eq:reduction-axis-constraint:<br/>one scale per dot product"}} -->|"rules out<br/>per-channel X scales"| B
    R["and separately:<br/>round to compensate,<br/>not to the nearest"] --> G["eq:compensate-not-round"]
    G -->|"needs H = X'X"| CAL[("calibration set:<br/>wrong shape is<br/>worse than none")]
```

### 7.1 The methods, by what they cost

| Method | Where it acts | Runtime cost | Needs calibration |
|---|---|---|---|
| per-channel weight scales | weights | none | no |
| per-token activation scales | activations | cheap (row-wise) | no |
| per-channel activation scales | activations | **not expressible in an INT8 GEMM** | no |
| mixed precision | activations | second kernel path | statistics only |
| SmoothQuant | both, offline | none after folding | yes ($\alpha$) |
| AWQ | weights, offline | none after folding | **yes, critically** |
| Hadamard rotation | both | $O(k\log k)$ per call | no |
| GPTQ | weights, offline | none | **yes, critically** |

**Read the last two columns together.** Methods with no runtime cost pay for it
with a calibration dependency, and that dependency is where the failures live.

### 7.2 What actually ships

Weight-only INT4 with group scales, quantized by GPTQ or AWQ, is the dominant
local-inference configuration — and {{sec:9-practical-example}}'s weight-only row
explains why: leaving activations in full precision **sidesteps the outlier
problem entirely** (0.0101 at 8 bits, 0.1825 at 4).

**The cost is that you never get INT8 tensor cores**, which is invisible at batch
1 and decisive at batch 64. {{ch:q-throughput-latency}} prices it.

W8A8 with SmoothQuant is the dominant *server* configuration, for the mirror
reason: it accepts calibration complexity to get integer arithmetic on both
operands.

### 7.3 Choosing, in order

1. **Are you quantizing activations at all?** If not, most of this chapter is
   background and group size is your lever.
2. **Measure $M/m$** on real activations and check {{eq:outlier-budget}}.
3. **Can your kernel express the scale you want?**
   {{eq:reduction-axis-constraint}} eliminates the best option first.
4. **Do you have representative calibration data?** If not, prefer methods that
   need none — the wrong data is worse than none.
5. **Then choose**, and validate on the deployment distribution rather than on
   perplexity over generic text.

## 8. Implementation

```python {tier=A name=four-answers-to-outliers}
"""Four answers to one problem, measured against each other.

cite:dettmers2022int8 identified the phenomenon that breaks naive INT8 above a
scale threshold: a small number of activation channels carry values far larger
than the rest, and any scale factor shared with them is forced to cover a range
the ordinary values never use.

Four responses followed, and they are genuinely different ideas rather than
variations on one:

  keep outliers in higher precision   cite:dettmers2022int8
  migrate the difficulty to weights   cite:xiao2023smoothquant
  protect the salient channels        cite:lin2023awq
  rotate so no channel dominates      cite:tseng2024quipsharp

This listing implements all four on the same layer and measures the OUTPUT error,
which is what matters, rather than the weight error, which is not
(eq:output-error-is-the-target).
"""
import numpy as np

rng = np.random.default_rng(251)

D_IN, D_OUT, N = 256, 256, 4096
OUTLIER_COLS = 6
OUTLIER_SCALE = 24.0


def quantize(A, bits, axis=None):
    """Symmetric integer quantization. axis=None shares one scale over the whole
    tensor; axis=0 gives each column its own."""
    qmax = 2 ** (bits - 1) - 1
    s = (np.max(np.abs(A)) if axis is None
         else np.max(np.abs(A), axis=axis, keepdims=True)) / qmax
    s = np.maximum(s, 1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


W = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
X = rng.normal(size=(N, D_IN))
hot = rng.choice(D_IN, size=OUTLIER_COLS, replace=False)
X[:, hot] *= OUTLIER_SCALE                      # the emergent outlier features
REF = X @ W


def err(Y):
    return float(np.linalg.norm(Y - REF) / np.linalg.norm(REF))


def hadamard(n):
    """A Hadamard matrix of size n (a power of two), normalised to be
    orthogonal. Multiplying by it mixes every coordinate into every other."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def baseline_fp(bits):
    """Weights quantized, activations left alone -- weight-only quantization."""
    return err(X @ quantize(W, bits))


def naive(bits):
    """Both quantized, one scale each. This is the configuration that breaks."""
    return err(quantize(X, bits) @ quantize(W, bits))


def per_channel(bits):
    """A scale per activation channel. The outlier channel gets its own."""
    return err(quantize(X, bits, axis=0) @ quantize(W, bits, axis=0))


def mixed_precision(bits, k=OUTLIER_COLS):
    """cite:dettmers2022int8: keep the k largest-magnitude channels in full
    precision and quantize the rest."""
    mag = np.max(np.abs(X), axis=0)
    keep = np.argsort(-mag)[:k]
    mask = np.zeros(D_IN, bool); mask[keep] = True
    Y = X[:, mask] @ W[mask]
    Xq = quantize(X[:, ~mask], bits)
    Y = Y + Xq @ quantize(W[~mask], bits)
    return err(Y)


def smoothquant(bits):
    """cite:xiao2023smoothquant: divide activations by a per-channel factor and
    multiply the corresponding weight rows by it. Exactly equivalent as a
    function; only what the quantizer sees changes. alpha balances how much
    difficulty is moved, and is searched here as the paper searches it."""
    a = np.maximum(np.max(np.abs(X), axis=0), 1e-12)
    w = np.maximum(np.max(np.abs(W), axis=1), 1e-12)
    best = None
    for alpha in np.linspace(0.1, 0.9, 9):
        s = np.maximum((a ** alpha) / (w ** (1 - alpha)), 1e-12)
        s = s / np.mean(s)
        e = err(quantize(X / s, bits) @ quantize(W * s[:, None], bits))
        if best is None or e < best[0]:
            best = (e, alpha)
    return best[0]


def awq(bits):
    """cite:lin2023awq: importance comes from ACTIVATION statistics, and the
    per-channel scale is s = mean|X_j|^alpha with alpha SEARCHED. The search is
    part of the method, not a refinement of it."""
    imp = np.mean(np.abs(X), axis=0)
    imp = np.maximum(imp / np.mean(imp), 1e-12)
    best = None
    for alpha in np.linspace(0.0, 1.0, 11):
        s = np.maximum(imp ** alpha, 1e-12)
        e = err(quantize(X / s, bits) @ quantize(W * s[:, None], bits))
        if best is None or e < best[0]:
            best = (e, alpha)
    return best[0]


def rotated(bits):
    """cite:tseng2024quipsharp: rotate into a basis where the energy is spread
    over all coordinates, quantize there, and rotate back. Orthogonal, so the
    function is unchanged."""
    H = hadamard(D_IN)
    return err((quantize(X @ H, bits) @ quantize(H.T @ W, bits)))


print(f"A {D_IN}x{D_OUT} layer with {OUTLIER_COLS} activation channels "
      f"{OUTLIER_SCALE:.0f}x larger than the rest.")
print("Relative error of the layer OUTPUT. Weights and activations both "
      "quantized\nexcept where noted.")
print()
print(f"{'method':>34}" + "".join(f"{str(b) + ' bits':>12}" for b in (8, 6, 4)))
print("-" * 70)

METHODS = [
    ("weight-only (activations in fp)", baseline_fp),
    ("naive: one scale each", naive),
    ("per-channel activation scales", per_channel),
    ("mixed precision for outliers", mixed_precision),
    ("SmoothQuant migration", smoothquant),
    ("AWQ salient-channel scaling", awq),
    ("Hadamard rotation", rotated),
]
res = {}
for name, fn in METHODS:
    vals = [fn(b) for b in (8, 6, 4)]
    res[name] = vals
    print(f"{name:>34}" + "".join(f"{v:>12.4f}" for v in vals))

print()
print()
print("How much of the damage is the outliers? Same methods, no outliers.")
print()
print(f"{'method':>34}" + "".join(f"{str(b) + ' bits':>12}" for b in (8, 6, 4)))
print("-" * 70)
X_HOT = X.copy()
X[:, hot] /= OUTLIER_SCALE
REF = X @ W
clean = {}
for name, fn in METHODS:
    vals = [fn(b) for b in (8, 6, 4)]
    clean[name] = vals
    print(f"{name:>34}" + "".join(f"{v:>12.4f}" for v in vals))
X = X_HOT
REF = X @ W

nv, pc = res["naive: one scale each"], res["per-channel activation scales"]
mp, sq = res["mixed precision for outliers"], res["SmoothQuant migration"]
aw, ro = res["AWQ salient-channel scaling"], res["Hadamard rotation"]
wo = res["weight-only (activations in fp)"]
cn = clean["naive: one scale each"]
print(f"""
The naive row is the problem. With both tensors quantized against a single scale
each, the 8-bit output error is {nv[0]:.4f}; the identical configuration without
outliers gives {cn[0]:.4f}. Six channels out of {D_IN} cost a factor of
{nv[0]/cn[0]:.1f}.

The weight-only row is why weight-only quantization dominates local inference.
Leaving activations in full precision sidesteps the problem entirely --
{wo[0]:.4f} at 8 bits, {wo[2]:.4f} at 4 -- because the outliers are an ACTIVATION
phenomenon and a method that never quantizes activations never meets them. It also
never gets INT8 tensor cores, which is the trade ch:q-throughput-latency prices.

Now the four responses. All of them recover most of the loss, and the ranking is
not the one the literature's chronology suggests.

Per-channel activation scales reach {pc[0]:.4f} at 8 bits against naive's
{nv[0]:.4f} -- a factor of {nv[0]/pc[0]:.1f}, from the simplest possible change.
Mixed precision, keeping the {OUTLIER_COLS} largest channels in full precision,
reaches {mp[0]:.4f}, the best number in the table. The Hadamard rotation reaches
{ro[0]:.4f}. SmoothQuant reaches {sq[0]:.4f} and AWQ {aw[0]:.4f}, both with their
scaling exponent searched as the papers search it.

So the two most-cited methods are the two WORST performers here, and that requires
an explanation rather than a shrug.

The explanation is that this listing measures error and the methods were designed
under a constraint it does not model. An INT8 matrix multiply computes a dot
product along the reduction axis, and a single scale factor must apply to that
whole dot product -- so a per-channel activation scale, which varies ALONG the
reduction axis, cannot be folded into an INT8 GEMM. It is expressible in numpy and
not in the kernel that makes INT8 worth using.

That is precisely what SmoothQuant exists to fix. Dividing the activations by a
per-channel constant and multiplying the corresponding weight ROWS by the same
constant leaves the function identical, moves the difficulty into a tensor that
had range to spare, and -- the part that matters -- the surviving per-channel
factor now lives on the weights, where it is a compile-time constant folded in
once rather than a runtime scale varying inside the reduction.

AWQ's contribution is orthogonal to that and survives this listing intact: the
importance of a weight channel is computed from ACTIVATION statistics rather than
from weight magnitudes. A small weight multiplying a large activation matters more
than a large weight multiplying a small one, and only the data can say which is
which (eq:output-error-is-the-target). Note what that implies about the
calibration set -- it is not a formality, it is where the method gets its
information, and it is almost never reported.

The Hadamard row deserves its own note because it attacks the problem from the
furthest away. Rather than protecting the outlier coordinates it changes the basis
so that no coordinate is an outlier: an orthogonal mixing spreads each channel's
energy across all of them, turning a spiky distribution into an approximately
Gaussian one, which is exactly what a uniform quantizer handles well. It costs a
transform on the critical path, which is the trade cite:tseng2024quipsharp makes.

Read the second table against the first and the framing settles. Without outliers,
every method lands within a small factor of every other and naive is competitive
at {cn[0]:.4f}. The entire difference between these techniques is what they do
about a handful of channels.

Which is the chapter's organising claim. These are not four quantization
algorithms. They are four answers to what to do when ch:q-formats's
eq:scale-group-condition is violated by a few coordinates -- isolate them, move
them, protect them, or destroy the basis in which they are special. And the choice
between them is made on kernel cost rather than on the error column, which is why
reading the error column alone gives the wrong ranking.""")
```

The first listing is about *scaling*. The second is about *rounding*, which is a
separate lever that composes with it.

```python {tier=A name=compensate-not-round}
"""Rounding is a choice, and round-to-nearest is the wrong one.

Every quantizer so far has rounded each weight to its nearest representable
value, independently. That is optimal if you want each WEIGHT to be close to its
original. It is not what you want.

What you want is the layer's OUTPUT to be close, and that is a different problem
because the weights are not independent in their effect: an error in one weight
can be partly cancelled by deliberately mis-rounding another
(eq:compensate-not-round).

cite:frantar2023gptq turns that observation into an algorithm. Quantize the
weights in order; after each one, push its rounding error into the weights not yet
quantized, weighted by the input covariance so the compensation is aimed at the
directions the data actually occupies. This listing implements it and measures
what it is worth against plain rounding at the same bit-width.
"""
import numpy as np

rng = np.random.default_rng(257)

D_IN, D_OUT, N = 256, 128, 3000


def rtn(W, bits, group=0):
    """Round to nearest, per group of input dimensions."""
    qmax = 2 ** (bits - 1) - 1
    Q = np.empty_like(W)
    g = group if group > 0 else W.shape[0]
    for a in range(0, W.shape[0], g):
        blk = W[a:a + g]
        s = np.maximum(np.max(np.abs(blk)) / qmax, 1e-12)
        Q[a:a + g] = np.clip(np.round(blk / s), -qmax, qmax) * s
    return Q


def gptq(W, H, bits, group=0, damp=0.01):
    """cite:frantar2023gptq. Quantize input dimensions in order; after each,
    subtract the induced error from the remaining ones through the inverse
    Hessian, so the compensation is aimed where the data has variance."""
    d = W.shape[0]
    qmax = 2 ** (bits - 1) - 1
    Hd = H + damp * np.mean(np.diag(H)) * np.eye(d)
    U = np.linalg.cholesky(np.linalg.inv(Hd)).T      # upper triangular
    Wk = W.copy()
    Q = np.empty_like(W)
    g = group if group > 0 else d
    for a in range(0, d, g):
        blk = Wk[a:a + g]
        s = np.maximum(np.max(np.abs(blk)) / qmax, 1e-12)
        for i in range(a, min(a + g, d)):
            w = Wk[i]
            q = np.clip(np.round(w / s), -qmax, qmax) * s
            Q[i] = q
            e = (w - q) / U[i, i]
            if i + 1 < d:
                Wk[i + 1:] -= np.outer(U[i, i + 1:], e)
    return Q


# A layer whose input covariance is far from isotropic, as real activations are.
A = rng.normal(size=(D_IN, D_IN))
COV = A @ A.T / D_IN + 0.05 * np.eye(D_IN)
L = np.linalg.cholesky(COV)
X = rng.normal(size=(N, D_IN)) @ L.T
hot = rng.choice(D_IN, size=5, replace=False)
X[:, hot] *= 10.0
H = X.T @ X / N

W = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
REF = X @ W


def out_err(Q):
    return float(np.linalg.norm(X @ Q - REF) / np.linalg.norm(REF))


def wgt_err(Q):
    return float(np.linalg.norm(Q - W) / np.linalg.norm(W))


print(f"A {D_IN}x{D_OUT} layer, correlated non-isotropic inputs with 5 outlier")
print("channels. Both metrics reported, because they rank the methods "
      "differently.")
print()
print(f"{'bits':>6}{'group':>8}" + f"{'WEIGHT error':>26}"
      + f"{'OUTPUT error':>28}")
print(f"{'':>6}{'':>8}{'RTN':>12}{'GPTQ':>14}{'RTN':>14}{'GPTQ':>14}")
print("-" * 68)

rows = {}
for bits in (8, 4, 3):
    for group in (0, 64):
        qr, qg = rtn(W, bits, group), gptq(W, H, bits, group)
        r = (wgt_err(qr), wgt_err(qg), out_err(qr), out_err(qg))
        rows[(bits, group)] = r
        lbl = "tensor" if group == 0 else str(group)
        print(f"{bits:>6}{lbl:>8}{r[0]:>12.4f}{r[1]:>14.4f}"
              f"{r[2]:>14.4f}{r[3]:>14.4f}")

print()
print()
print("Does the compensation depend on the calibration data being right?")
print()
print(f"{'calibration set':>28}{'output error':>15}{'vs correct':>13}")
print("-" * 56)

Xw = rng.normal(size=(N, D_IN))                        # isotropic: wrong shape
Xs = X[:64]                                            # too few samples
Xn = X + 0.5 * np.std(X) * rng.normal(size=X.shape)     # noisy, right shape

CAL = [
    ("the true input covariance", out_err(gptq(W, H, 4, 64))),
    ("isotropic (wrong shape)", out_err(gptq(W, Xw.T @ Xw / N, 4, 64))),
    ("64 samples of the real data",
     out_err(gptq(W, Xs.T @ Xs / len(Xs), 4, 64))),
    ("noisy copy of the real data", out_err(gptq(W, Xn.T @ Xn / N, 4, 64))),
    ("RTN, no calibration at all", out_err(rtn(W, 4, 64))),
]
correct = CAL[0][1]
cal = dict(CAL)
for name, v in CAL:
    print(f"{name:>28}{v:>15.4f}{v / correct:>12.2f}x")

t8, t4, t3 = rows[(8, 0)], rows[(4, 0)], rows[(3, 0)]
g4 = rows[(4, 64)]
print(f"""
Read the two metric blocks against each other, because the disagreement is the
whole point.

On WEIGHT error, GPTQ is worse than round-to-nearest at every setting:
{t4[1]:.4f} against {t4[0]:.4f} at 4 bits. That is not a bug and not a close
call. Round-to-nearest MINIMISES weight error by construction, so nothing can
beat it there, and any method that beats it on something else must be worse here.

On OUTPUT error, which is what the layer is for, the ranking reverses:
{t4[3]:.4f} against RTN's {t4[2]:.4f} at 4 bits, better by {t4[2]/t4[3]:.2f}x. At
3 bits, {t3[3]:.4f} against {t3[2]:.4f}. GPTQ is deliberately choosing worse
weights in order to get a better function (eq:compensate-not-round).

The mechanism is easy to describe and easy to misremember. After rounding weight
i downward, the layer's output is slightly too small in a particular direction.
The weights not yet quantized can be nudged upward to put it back -- and the
inverse Hessian says how much of the nudge each should absorb, as a function of
how much variance the input data has in each direction. Weights multiplying
high-variance inputs get small corrections because their effect is large; weights
multiplying directions the data barely occupies get large ones, because there they
are nearly free.

Which makes the calibration data load-bearing rather than a formality, and the
second table shows how much.

With the true input covariance, GPTQ reaches {correct:.4f}. With a noisy copy of
the real data -- substantial noise added, correlation structure preserved --
{cal['noisy copy of the real data']:.4f}, only {cal['noisy copy of the real data']/correct:.2f}x
worse. So the method needs the SHAPE of the input distribution and tolerates a
great deal of noise in it, which matches the practical folklore that a few hundred
calibration sequences suffice.

Now the two failure rows, which are the reason to run this experiment. With an
isotropic covariance -- right size, wrong shape -- GPTQ reaches
{cal['isotropic (wrong shape)']:.4f}. With only 64 samples of the real data,
{cal['64 samples of the real data']:.4f}. And plain round-to-nearest, with no
calibration at all, reaches {cal['RTN, no calibration at all']:.4f}.

Read those three together. **Mis-calibrated GPTQ is worse than no GPTQ.** The
compensation is aimed at directions the data does not occupy, so it is spending
its rounding freedom to fix errors that do not matter while creating ones that
do. An algorithm that improves on RTN by {t4[2]/t4[3]:.2f}x with correct
calibration is {cal['isotropic (wrong shape)']/cal['RTN, no calibration at all']:.2f}x
worse than RTN with the wrong kind.

That is the practical warning the method comes with and rarely carries. A
calibration set drawn from the wrong distribution -- generic web text for a model
serving code, English for a model serving another language, short sequences for a
long-context deployment -- is not a slightly worse calibration set. It can be an
actively harmful one, and nothing in the output of the quantization step says so.

Compare the group column while you are here. Group 64 improves both methods at 8
and 4 bits, and the gains compose: GPTQ at group 64 reaches {g4[3]:.4f} against
tensor-wide GPTQ's {t4[3]:.4f} and group-64 RTN's {rows[(4, 64)][2]:.4f}. Error
compensation and finer scales solve different parts of the problem -- one chooses
better rounding directions, the other reduces the step those roundings work with
-- so applying both is not redundant.

At 3 bits the composition breaks down: GPTQ at group 64 gives
{rows[(3, 64)][3]:.4f} against tensor-wide GPTQ's {t3[3]:.4f}. Resetting the scale
every 64 weights leaves less room for the error propagation to work with, and at
3 bits there is not enough headroom to spare. Worth noting as a reminder that
these techniques interact, and that "apply both" is a default rather than a
theorem.

The general lesson outlasts the algorithm. Every quantizer before this one asked
"what is the closest representable value"; this one asks "what assignment of
representable values minimises the error in the thing I care about". Those
questions have different answers, and the second is the one that was always
meant. The cost is that it needs to know what you care about -- which means
calibration data, which means a hyperparameter that is rarely reported and
occasionally decisive.""")
```

## 9. Practical Example

**Six activation channels out of 256, at 24× magnitude, cost a factor of 3.5** at
8 bits: naive error **0.0581** against **0.0166** without them. At 4 bits,
**0.3524** — {{eq:outlier-budget}} puts the 4-bit budget at $M/m \lesssim 3.5$ and
the measurement is at **24**, seven times over.

**The four remedies, at 8 bits:** isolation **0.0043**, per-channel scales
**0.0109**, Hadamard rotation **0.0148**, SmoothQuant **0.0186**, AWQ **0.0191**.

> **IMPORTANT:** The two most-cited methods rank last, and the explanation is
> {{eq:reduction-axis-constraint}}: **a per-channel activation scale varies along
> the reduction axis and cannot be factored out of an INT8 dot product.** It is
> expressible in numpy and not in the kernel that makes INT8 worth using.
> {{eq:difficulty-migration}} exists to move that scale onto the weights, where it
> folds in once. **Ranking these methods by error inverts the answer.**

**AWQ's contribution survives the ranking**: importance from *activation*
statistics ({{eq:output-error-is-the-target}}), because a small weight multiplying
a large activation matters more than the reverse. **Which makes the calibration
set where the method gets its information.**

**Weight-only quantization sidesteps all of it** — **0.0101** at 8 bits and
**0.1825** at 4, because outliers are an activation phenomenon. It also never gets
INT8 tensor cores.

**Then rounding, which is a separate lever.** GPTQ scores **0.2985** on weight
error against RTN's **0.1696** — worse, necessarily, by
{{eq:weight-error-floor}} — and **0.1273** on output error against **0.1725**,
better by **1.36×**. **It chooses worse weights to get a better function**
({{eq:compensate-not-round}}), and an implementation whose weight error matches
RTN's is broken.

**And the calibration warning.** True covariance **0.1141**; a noisy copy
**0.1297** (1.14×, nearly free); isotropic **0.1603**; 64 samples **0.2167**. And
**RTN with no calibration at all: 0.1578.**

**Mis-calibrated GPTQ is worse than no GPTQ.** A method that improves on RTN by
1.36× with the right data is **1.02× worse** than RTN with the wrong kind — and
nothing in the quantization run reports it.

**The gains compose, with a limit.** GPTQ at group 64 reaches **0.1141** against
tensor-wide GPTQ's **0.1273** and group-64 RTN's **0.1578**. But at 3 bits, GPTQ
at group 64 gives **0.3221** against tensor-wide GPTQ's **0.2994** — resetting the
scale every 64 rows leaves too little headroom for the compensation. **"Apply
both" is a default, not a theorem.**

## 10. Production Considerations

**Measure $M/m$ on real activations** before choosing anything
({{eq:outlier-budget}}).

**Check {{eq:reduction-axis-constraint}} against your kernel** before selecting a
scaling scheme — it eliminates the best-scoring option first.

**Draw the calibration set from the deployment distribution.** Wrong-shape data is
worse than none.

**Report the calibration set** with any GPTQ or AWQ result. It is a
hyperparameter.

**Prefer weight-only** unless you need integer activations for throughput, and
know which regime you are in ({{ch:q-throughput-latency}}).

**Validate on the deployment distribution**, not on perplexity over generic text —
that is the same failure as the calibration one, one step later.

**Re-validate at scale.** Outliers are emergent; a recipe proven at 7B is not
proven at 70B.

## 11. Common Mistakes

**Choosing a method by its error column** rather than by kernel cost.

**Assuming a per-channel activation scale is implementable.**

**Using a generic calibration set** for a specialised deployment.

**Judging a weight importance by its magnitude** rather than by the activation it
multiplies.

**Expecting GPTQ to improve weight error** — it cannot, by
{{eq:weight-error-floor}}.

**Assuming group-wise and compensation always compose.**

**Validating a quantization recipe at a smaller scale** than deployment.

**Treating outliers as a weight problem.** They are an activation phenomenon.

## 12. Failure Modes

**INT8 works at 7B and collapses at 70B.** Cause: emergent outliers
({{cite:dettmers2022int8}}).

**Quantized model fine on benchmarks, poor on the real workload.** Cause: the
calibration set and the evaluation share a distribution the deployment does not.

**GPTQ performs worse than RTN.** Cause: mis-calibration. Check $H$ before
blaming the algorithm.

**4-bit fails where 8-bit was fine, on the same model.** Cause:
{{eq:outlier-budget}} — the budget shrank by 16× and the outliers did not.

**SmoothQuant helps activations and ruins weights.** Cause: $\alpha$ too high.

**Rotation costs more latency than the quantization saves.** Cause: the transform
is on the critical path and the deployment is compute-bound.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| weight-only INT4 + groups | no INT8 arithmetic | local inference; the default |
| W8A8 + SmoothQuant | calibration | server throughput |
| mixed precision | kernel complexity | when a second path is affordable |
| Hadamard / QuIP# | runtime transform | extreme compression |
| SpQR ({{cite:dettmers2023spqr}}) | sparse storage format | near-lossless 3-4 bit |
| AQLM ({{cite:egiazarian2024aqlm}}) | decode speed | 2-3 bits, memory-bound |
| a smaller model at higher precision | capability | frequently the honest answer |

**The last two rows are in tension and worth naming.**
{{cite:egiazarian2024aqlm}}'s sub-3-bit formats trade decode speed for memory,
which is the opposite of what {{ch:q-gguf}} shows local inference wants — and the
last row is the comparison {{cite:dettmers2023case4bit}} actually settled.

## 14. Evaluation

**Report the calibration set** — source, size, and sequence length.

**Report $M/m$** for the layers you quantized.

**Report weight error and output error separately.** They rank methods
differently and only one matters.

**Report the group size and the scale precision.**

**Evaluate on the deployment distribution**, and say what it was.

**Report the scale at which the recipe was validated.**

## 15. Advanced Concepts

**Outliers as an architectural artefact.** {{maturity:EMERGING}} The emergent
features of {{cite:dettmers2022int8}} appear in specific dimensions and persist
across inputs, which suggests they are doing a job — attention sinks and
normalisation escape valves are both candidate explanations. **If so, they are
fixable at training time**, and some recent architectures do exactly that.

**Rotation as preprocessing for everything.** {{maturity:MATURE}}
{{eq:incoherence-processing}} is free in the Frobenius sense and composes with any
quantizer, so it is increasingly applied as a preprocessing step rather than as
part of a specific method.

**Calibration as a distribution-shift problem.** {{maturity:EMERGING}}
The GPTQ/AWQ failure mode is exactly {{ch:ft-datasets}}'s selection bias in a new
setting: the method is optimised for the distribution it was shown, and the
mismatch is invisible from inside. **The remedy is the same one — measure on
something the procedure did not choose.**

**Sub-3-bit changes the question.** {{maturity:EMERGING}}
{{cite:egiazarian2024aqlm}} and {{cite:tseng2024quipsharp}} abandon
scalar quantization for codebooks and lattices, at which point
{{eq:outlier-budget}} no longer applies — the representation is no longer a grid.
**The binding constraint moves from accuracy to decode speed.**

**Quantizing the calibration decision itself.** {{maturity:RESEARCH FRONTIER}}
Nothing currently reports how sensitive a quantized model is to its calibration
set. A cheap sensitivity estimate — quantize twice with disjoint calibration data
and compare — would turn an invisible risk into a number, and it costs one extra
run.

**And a note on what the accuracy numbers are measured over.**
{{maturity:EMERGING}} Quantisation is evaluated on aggregate benchmark scores,
which are averages over many items, and the loss it causes is concentrated on the
items where the model was least confident to begin with. An aggregate that moves
half a point can hide a much larger change on the tail — the long-context, the
rare entity, the unusual format. **The measurement that would show it is
disaggregated**, and it is the same argument {{ch:rai-bias}} makes for a different
reason.

## 16. Connection to Previous Chapters

{{ch:q-theory}}'s {{eq:outlier-inflates-the-step}} and {{eq:effective-levels}} are
what {{eq:outlier-budget}} formalises, and its finding that outliers inflate the
per-layer error without changing its accumulation is why all four remedies here
are local.
{{ch:q-formats}}'s {{eq:scale-group-condition}} is the inequality every method in
this chapter is trying to satisfy, and {{eq:reduction-axis-constraint}} is the
engineering reason some ways of satisfying it are unavailable.
{{ch:ft-datasets}}'s selection-bias argument is the calibration failure in a
different domain.
Forward: {{ch:q-gguf}} applies weight-only quantization where bandwidth is the
whole story; {{ch:q-activation-kv}} handles the activations this chapter mostly
avoided; {{ch:q-throughput-latency}} prices the INT8-tensor-core trade.

## 17. Exercises

1. Derive {{eq:outlier-budget}} from {{eq:effective-levels}} and compute the
   permitted $M/m$ at 8, 4 and 3 bits.
2. Verify {{eq:difficulty-migration}} algebraically and state what breaks if $S$
   is not diagonal.
3. From {{eq:rotation-flattens}}, compute the post-rotation outlier ratio for
   $k = 1024$ and $k = 8192$. Why does rotation get *better* with width?
4. Prove {{eq:weight-error-floor}} and explain what it implies about validating a
   GPTQ implementation.
5. In `four-answers-to-outliers`, raise the outlier scale to 64. Which methods
   still work, and does {{eq:outlier-budget}} predict it?
6. In `compensate-not-round`, use a calibration set drawn from a *shifted*
   distribution rather than an isotropic one. Is the damage similar?
7. Explain why per-token activation scales are implementable in an INT8 GEMM and
   per-channel ones are not, using {{eq:reduction-axis-constraint}}.
8. Estimate the cost of a Hadamard transform per token for a 4096-wide model, and
   compare against the matmul it precedes.

## 18. Interview Questions

1. What are emergent outliers, and why do they appear only above a scale?
2. Name four responses to them and what each assumes.
3. Per-channel activation scaling scores best. Why is it not used?
4. What does SmoothQuant actually change about the model?
5. How does AWQ decide which channels are salient, and why does that matter?
6. Why does a Hadamard rotation help, and what does it cost?
7. GPTQ has worse weight error than round-to-nearest. Is that a bug?
8. When is GPTQ worse than doing nothing?
9. Why does weight-only quantization avoid the outlier problem entirely?
10. Your INT8 recipe worked at 7B and fails at 70B. What happened?

## 19. Research Questions

1. Are emergent outliers necessary, or an artefact of normalisation and attention
   design? What does an architecture trained to avoid them cost?
2. {{eq:rotation-flattens}} improves with width. Does that mean quantization gets
   *easier* as models get wider, and is that visible in practice?
3. How sensitive is a quantized model to its calibration set, measured as
   disagreement between two disjoint calibrations? Is that a useful risk metric?
4. {{eq:gptq-update}} propagates error along an ordering. Does the ordering
   matter, and is there a better one than input-dimension order?
5. Below 3 bits the grid disappears and {{eq:outlier-budget}} stops applying. What
   replaces it as the viability condition for codebook methods?

## 20. Chapter Summary

**Emergent outliers are an activation phenomenon that appears above a scale
threshold**, and {{eq:outlier-budget}} says why they became urgent exactly when
4-bit did: the tolerable ratio falls from about 60 at 8 bits to about 3.5 at 4.
Measured, six channels at 24× cost **3.5×** at 8 bits and took 4-bit naive
quantization to **0.3524**.

**Four remedies, all local, all different.** Isolation (**0.0043**), per-channel
scales (**0.0109**), rotation (**0.0148**), migration (**0.0186**), salient-channel
scaling (**0.0191**).

**And the ranking by error is the wrong ranking.**
{{eq:reduction-axis-constraint}}: one scale must serve an entire dot product, so a
per-channel *activation* scale cannot be folded into an INT8 GEMM.
{{eq:difficulty-migration}} exists to move that scale onto the weights, where it
becomes a compile-time constant — **which is why the second-worst performer is the
one production uses.**

**Rounding is a separate lever, and round-to-nearest is the wrong choice.**
{{eq:compensate-not-round}} minimises *output* error, and GPTQ solves it greedily
via {{eq:gptq-update}}: **weight error 0.2985 against RTN's 0.1696 (worse,
necessarily, by {{eq:weight-error-floor}}) and output error 0.1273 against 0.1725
(better by 1.36×).** It chooses worse weights to get a better function.

**Which makes the calibration set a hyperparameter with teeth.** Noise is nearly
free (**1.14×**), but an isotropic calibration gives **0.1603** against
round-to-nearest's **0.1578** — **mis-calibrated compensation is worse than no
compensation**, and nothing in the quantization run says so.

**And the levers compose only up to a point**: GPTQ plus group-64 reaches
**0.1141** at 4 bits, and at 3 bits the same combination is *worse* than
tensor-wide GPTQ because a short group leaves no headroom for compensation.

Which leaves the chapter's practical shape: **measure the outlier ratio, check
what your kernel can express, and treat the calibration set as part of the
specification** — because the first decides whether quantization is viable, the
second decides which remedy is available, and the third decides whether the remedy
helps or hurts.

## 21. Further Reading

{{cite:dettmers2022int8}} for the phenomenon, and specifically for its
scale-dependence, which is what makes quantization results non-transferable
between model sizes.
{{cite:xiao2023smoothquant}} read against {{eq:reduction-axis-constraint}}: the
paper's contribution looks like an accuracy technique and is really a kernel
feasibility one.
{{cite:lin2023awq}} for activation-derived importance, which is the idea in this
chapter most likely to outlive the specific methods.
{{cite:frantar2023gptq}} for error compensation, and note how little of the paper
is about the calibration set that {{sec:9-practical-example}} shows can invert the
result.
{{cite:tseng2024quipsharp}} and {{cite:dettmers2023spqr}} for the two directions
beyond scalar grids — change the basis, or accept a sparse exception list.
