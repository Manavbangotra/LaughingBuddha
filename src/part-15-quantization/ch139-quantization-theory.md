---
id: q-theory
number: 139
part: XV
tier: full
status: draft
requires: [q-formats, dl-normalization, math-random-vars]
provides: [errors-add-in-quadrature, quantization-noise-variance,
           group-size-dominates, outlier-inflates-the-step,
           fragility-grows-with-training, checkpoint-not-architecture]
citations: [kumar2024precisionscaling, dettmers2022int8, dettmers2023case4bit,
            xiao2023smoothquant, frantar2023gptq]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why quantization error does
not compound catastrophically through depth, and what assumption that relies on;
state what an outlier actually does, in terms of the step size rather than in
terms of its own value; show that **group size matters more than bit-width**, and
price the trade; and explain why a quantization recipe is a claim about a
**checkpoint** rather than about an architecture.

## 2. Why This Matters

{{ch:q-formats}} established what a format can represent. This chapter asks the
question that has to be answered before any of it is useful: **an 80-layer model
with a 1% perturbation at every layer should be destroyed, and it is not. Why?**

{{sec:9-practical-example}} measures the growth directly. Output error grows as
depth to the power **0.52** at 8 bits, **0.50** at 6, and **0.49** at 4 — all
close to one half.

**That is the signature of independent perturbations adding in quadrature.** $L$
layers each contributing error $\epsilon$ produce output error $\epsilon\sqrt{L}$,
not $\epsilon L$ and certainly not $\epsilon$ compounding multiplicatively. At 80
layers, square-root growth multiplies the per-layer error by **nine** — which is
the entire reason a quantized deep model still works.

**And it identifies exactly what would break it: correlation.** Scaling a few
weight columns up — so one shared scale must cover a range it otherwise would not
— raises the 8-bit error at depth 16 from **4.23e-02** to **6.94e-01**, a factor
of **16.4**, with **no change to the bit-width**. Ten per cent of columns costs
**27.9×**.

**Compare that to what bits buy**: 8 to 6 on the clean network costs **4.0×**.
**The distribution matters more than the bit-width, by a wide margin**, and that
is not how the choice is usually framed.

**Then the parameter that actually controls it.** At 8 bits, outliers cost
**5.8×** with a whole-tensor scale and only **2.2×** with groups of 32. **Shrink
the group and the damage is contained.**

**And finally the result that dates every quantization recipe.** Training the same
model longer and quantizing identically at each checkpoint, relative damage grows
from **+3.8%** to **+4008%** and absolute damage by a factor of **31**. **A
better-trained model is a more fragile one**, which inverts the usual intuition
and is exactly what {{cite:kumar2024precisionscaling}} finds at pretraining scale.

{{maturity:ESTABLISHED}} Error accumulation, outlier mechanics.
{{maturity:MATURE}} Group size as the dominant parameter.
{{maturity:EMERGING}} Fragility as a checkpoint property.

## 3. Prerequisites

{{ch:q-formats}} for {{eq:scale-group-condition}}, which this chapter turns into a
quantitative statement, and for the step-size vocabulary;
{{ch:dl-normalization}} for why activation scales are what they are;
{{ch:math-random-vars}} for variance addition, which is the whole argument in
{{sec:5-formal-explanation}}.

## 4. Intuitive Explanation

### Why deep networks survive at all

The worry is easy to state. Each quantized layer perturbs its output. That
perturbation is the input to the next layer, which perturbs it again. Over 80
layers, surely the error explodes.

**It does not, and the reason is that the errors do not point the same way.**

Each weight's rounding error is determined by where that weight happened to fall
between two representable values — an essentially arbitrary quantity, uncorrelated
between weights, between layers, and with the data flowing through. **Nothing in
the computation aligns them.**

Independent perturbations add in **quadrature**:

$$ \text{total} = \sqrt{\epsilon_1^2 + \dots + \epsilon_L^2} = \epsilon\sqrt{L} $$

not $\epsilon L$. {{sec:9-practical-example}} measures the exponent at **0.52,
0.50, 0.49** across three bit-widths.

> **At 80 layers that is a factor of nine rather than eighty.** Against a
> per-layer error that is already around 1%, nine is survivable and eighty is
> not. **The whole practice rests on that square root.**

### Which tells you what to be afraid of

If independence is doing the work, then **anything that correlates the errors is
the threat** — and the bit-width is not it.

{{sec:9-practical-example}} constructs the correlation directly. Scaling up a
small number of weight columns forces the shared scale factor to cover a wider
range, which **coarsens the step for every weight sharing that scale**:

```text
   outlier columns   scale   8-bit error   vs clean
   ───────────────   ─────   ───────────   ────────
              none       —      4.23e-02       1.0×
                2%      4×      1.14e-01       2.7×
                2%     16×      6.94e-01      16.4×
               10%     16×      1.18e+00      27.9×
```

**No bit-width changed in that table.** For comparison, dropping from 8 bits to 6
on the clean network costs **4.0×**.

### An outlier is a statement about the step size

This is the sentence to remember, and it is not how outliers are usually
described.

An outlier does not do damage by being large. It does damage by being **in the
same scale group as everything else**. The scale is set by the group's maximum, so
one huge value raises the step for all its neighbours — and they, not the outlier,
are where most of the error appears.

**Which immediately gives the fix: make the group smaller.** The outlier still
ruins its own group and every other group is untouched.

Measured at 8 bits: outliers cost **5.8×** with a whole-tensor scale and **2.2×**
with groups of 32.

### Group size is worth more than a bit

```text
   group size    4-bit, clean    4-bit, outliers
   ──────────    ────────────    ───────────────
   whole tensor        0.1813             0.4307
           256         0.1261             0.3450
            64         0.1074             0.2359
            32         0.0970             0.1732
```

Tensor-wide to groups of 64 is worth **1.7×** on clean weights. **A bit is worth
about 2×.** So finer grouping buys roughly a bit — and a 16-bit scale per 64
weights costs **a quarter of a bit**, four times cheaper than the bit it replaces.

> **Which is why every practical 4-bit format has a group size in its name, and
> why "4-bit" alone is an incomplete specification.** The number people quote is
> the one that matters less.

### The result that gives recipes an expiry date

Now the finding that inverts an intuition.

Train the same model for progressively longer and quantize each checkpoint
identically:

```text
   steps    test loss   ‖W‖    after 4-bit    damage
   ─────    ─────────   ────   ───────────   ───────
      50       0.2075   8.88        0.2154     +3.8%
     800       0.0030  13.26        0.0591   +1840%
    9600       0.0060  50.98        0.2473   +4008%
```

**The longer-trained model is hurt far more by an identical operation** — 31×
more in absolute terms, so it is not an artefact of a shrinking denominator.

The weight-norm column names the mechanism: training moves weights away from
their small initialisation, so the tensor's dynamic range grows and **the same
number of levels has more ground to cover.**

{{cite:kumar2024precisionscaling}} establishes this properly at pretraining scale
and finds it strong enough that **past some number of tokens, additional
pretraining makes the post-quantization model worse.**

> **The practical consequence is uncomfortable. A quantization recipe is validated
> on a CHECKPOINT, not on an architecture.** "We use 4-bit for this model" is a
> claim with an expiry date.

## 5. Formal Explanation

### 5.1 Quantization as additive noise

For a uniform quantizer with step $\Delta$, rounding error is approximately
uniform on $[-\Delta/2, \Delta/2]$:

$$ \mathbb{E}[\varepsilon] = 0, \qquad \text{Var}[\varepsilon] = \frac{\Delta^2}{12} $$ (eq:quantization-noise-variance)

For $b$ bits with a group maximum $M$, $\Delta = 2M/(2^{b}-2)$, so

$$ \sigma_{\varepsilon} \approx \frac{M}{\sqrt{3}\,(2^{b}-2)} $$ (eq:step-from-bits)

**{{eq:step-from-bits}} has $M$ and $b$ entering differently**: halving $M$ (a
finer group) and adding a bit have the same effect on $\sigma_\varepsilon$, which
is the quantitative form of the previous section's claim.

### 5.2 Errors add in quadrature

Let layer $\ell$'s quantization perturb its output by $\delta_\ell$, and let the
network be locally linear with Jacobian $J_\ell$ from layer $\ell$ to the output.
Then

$$ \delta_{\text{out}} = \sum_{\ell=1}^{L} J_\ell\, \delta_\ell $$

If the $\delta_\ell$ are **independent and zero-mean**, the cross terms vanish in
expectation:

$$ \mathbb{E}\|\delta_{\text{out}}\|^2 = \sum_{\ell} \|J_\ell\|^2\,\mathbb{E}\|\delta_\ell\|^2 \;\approx\; L\,\|J\|^2 \sigma^2 $$

$$ \Rightarrow \quad \|\delta_{\text{out}}\| \propto \sqrt{L} $$ (eq:errors-add-in-quadrature)

**{{eq:errors-add-in-quadrature}} is the measured exponent 0.5.** Had the errors
been perfectly correlated, the cross terms would survive and the sum would be
$L\|J\|\sigma$ — exponent 1.

### 5.3 What an outlier does

Let a group contain values with typical magnitude $m$ and one outlier of magnitude
$M \gg m$. The step is set by $M$:

$$ \Delta = \frac{2M}{2^{b}-2}, \qquad \text{so} \qquad \frac{\sigma_\varepsilon}{m} \;\propto\; \frac{M}{m} $$ (eq:outlier-inflates-the-step)

**{{eq:outlier-inflates-the-step}} is the whole mechanism.** The relative error of
the *ordinary* values scales with the ratio $M/m$ — the outlier's own error is
irrelevant, because there is one of it and thousands of them.

Equivalently, in terms of **effective levels**:

$$ n_{\text{eff}} = \frac{2m}{\Delta} = \frac{m}{M}\,(2^{b}-2) $$ (eq:effective-levels)

At $M/m = 16$, a 4-bit quantizer has **fewer than one effective level** for the
ordinary weights. **The bits are spent representing a range nothing occupies.**

### 5.4 Group size, priced

With group size $g$ and a $c$-bit scale, the storage cost is

$$ b_{\text{eff}} = b + \frac{c}{g} \quad \text{bits per weight} $$ (eq:group-size-cost)

and the benefit is that $M$ becomes the maximum over $g$ values rather than over
the whole tensor. For an outlier rate $\rho$, the fraction of groups containing an
outlier is $1 - (1-\rho)^{g} \approx \rho g$ for small $\rho g$:

$$ \mathbb{E}[\text{damage}] \;\approx\; \rho g \cdot \frac{M}{m} + (1 - \rho g) \cdot 1 $$ (eq:group-size-dominates)

**{{eq:group-size-dominates}} is linear in $g$ and the cost in
{{eq:group-size-cost}} is $1/g$** — which is why the optimum is at small $g$ and
why the returns flatten: at $g = 32$ with $c = 16$, the overhead is half a bit and
almost no group contains two outliers.

### 5.5 Why longer training increases fragility

Two mechanisms, and they compound.

**Range growth.** Training increases $\|W\|$, so $M$ grows and
{{eq:step-from-bits}} makes $\Delta$ grow with it. Measured: $\|W\|$ from **8.88
to 50.98**.

**Sharper optima.** Write the loss increase from a perturbation $\delta$ as
{{ch:ft-lora}}'s quadratic:

$$ \Delta\mathcal{L} \approx \tfrac{1}{2}\,\delta^{\top} H \delta $$

A converged model sits at a point with larger curvature in the directions that
matter, so the same $\|\delta\|$ costs more:

$$ \frac{\partial\,\Delta\mathcal{L}}{\partial\,\text{training}} \;>\; 0 \quad \text{through both } \|\delta\| \text{ and } H $$ (eq:fragility-grows-with-training)

> **IMPORTANT:** {{eq:fragility-grows-with-training}} means quantization
> robustness is a property of the **weights you have**, not of the architecture
> that produced them. Two checkpoints of one model, differing only in training
> duration, can need different recipes — and nothing in a config file records
> that.

## 6. Mathematical Foundation

### 6.1 The exponent, derived and checked

From {{eq:errors-add-in-quadrature}}, $\log \|\delta_{\text{out}}\| = \tfrac12 \log
L + \text{const}$, so a log-log fit should give slope $\tfrac12$. Measured:
**0.52, 0.50, 0.49** at 8, 6 and 4 bits. **Three independent confirmations across
an order of magnitude in per-layer error.**

The exponent drifts upward in the severe-outlier rows (to **0.80**), and that is a
**ceiling effect** rather than a change in accumulation: a relative error near 1
means the output is uncorrelated with the reference, and a saturating quantity
cannot keep following a power law.

### 6.2 Effective levels, worked

At $b = 4$, $2^b - 2 = 14$ levels across $[-M, M]$. For a Gaussian with
$\sigma = m$ and no outlier, $M \approx 4m$, so

$$ n_{\text{eff}} = \frac{14}{4} \approx 3.5 $$

Introduce a 16× outlier and $M \approx 16m$:

$$ n_{\text{eff}} = \frac{14}{16} \approx 0.9 $$

**Below one.** The ordinary weights are being rounded to $\{-\Delta, 0, +\Delta\}$
where $\Delta$ exceeds their entire range — which is why the measured damage
(16.4×) is so much larger than the outlier's own share of the tensor (2%).

### 6.3 Why the returns to grouping flatten

From {{eq:group-size-dominates}}, halving $g$ halves the fraction of contaminated
groups until $\rho g \ll 1$, after which further halving buys only the
$\sqrt{2\log g}$ shrinkage of a Gaussian maximum:

$$ \mathbb{E}[\max_{i \le g} |x_i|] \approx \sigma\sqrt{2\log g} $$ (eq:gaussian-max)

**{{eq:gaussian-max}} is the clean-weight benefit**, and it is logarithmic — which
is exactly the measured pattern: **0.1813 → 0.1074** from tensor to 64 (a real
gain) and **0.1074 → 0.0970** from 64 to 32 (much smaller).

> **MATH NOTE:** {{eq:quantization-noise-variance}}'s uniform-error assumption
> fails when the step is comparable to the signal — at 2–3 bits the error is not
> small, not uniform, and not independent of the value. **The quadrature argument
> is a large-$n$-levels approximation**, which is one reason sub-3-bit methods
> ({{cite:dettmers2023case4bit}}'s boundary) need entirely different analysis.

## 7. Internal Mechanics

```mermaid {#fig:error-accum caption="Why quantization survives depth, and what breaks it. Independent per-layer errors accumulate as the square root of depth (eq:errors-add-in-quadrature), which is the difference between a factor of nine and a factor of eighty at 80 layers. Outliers do not change that accumulation — they inflate the per-layer error itself by raising the step size for every weight sharing their scale (eq:outlier-inflates-the-step), which is why the remedy is local."}
flowchart TB
    B["b bits"] --> STEP["step size<br/>eq:step-from-bits"]
    M["group max M"] --> STEP
    G["group size g"] -->|"smaller g,<br/>smaller M"| M
    OUT["outlier in the group"] -->|"raises M<br/>for everyone"| M
    STEP --> PL["per-layer error"]
    PL -->|"independent:<br/>x sqrt(L)"| OK["survivable at depth"]
    PL -->|"correlated:<br/>x L"| BAD["not survivable"]
    TRAIN["longer training"] -->|"grows ||W||,<br/>sharpens the optimum"| PL
```

### 7.1 The three levers, ranked by measured effect

| Lever | Measured effect | Cost |
|---|---|---|
| remove or isolate outliers | up to **27.9×** | kernel complexity |
| group size (tensor → 32) | **5.1×** with outliers | $c/g$ bits per weight |
| bit-width (8 → 6) | **4.0×** | 2 bits per weight |
| depth | $\sqrt{L}$ only | not a lever |

**The ordering is the chapter's practical content**, and it is close to the
reverse of where attention usually goes.

### 7.2 What to measure before choosing a recipe

1. **The outlier ratio $M/m$ per tensor.** {{eq:effective-levels}} turns it
   directly into how many levels you actually have.
2. **The group size the format will use**, and whether the scale is 16 or 32 bits.
3. **The checkpoint's weight norms**, compared against any earlier checkpoint the
   recipe was validated on ({{eq:fragility-grows-with-training}}).
4. **Layer-by-layer sensitivity**, because
   {{eq:errors-add-in-quadrature}}'s $\|J_\ell\|$ is not uniform — some layers
   contribute far more than others, and uniform bit allocation ignores that.

### 7.3 Why this chapter does not settle on a bit-width

{{cite:dettmers2023case4bit}} ran 35,000 experiments and found 4 bits almost
universally optimal for total model bits against zero-shot accuracy. **That
result is about the model-size/precision trade at fixed memory**, and this chapter
is about the error mechanics underneath it.

The two fit together: 4 bits is where {{eq:effective-levels}} still leaves enough
resolution *given* a sensible group size and outlier handling — which is why the
same paper's caveats are about block size and data type, the two things this
chapter identifies as dominant.

### 7.4 The one place the square root does not save you

{{eq:errors-add-in-quadrature}} is reassuring about depth and says nothing about
width, and the distinction is worth drawing because it is where the reassurance
runs out.

Within a single matmul, each output element is a sum of $K$ products, and the
quantization errors in those products are also independent — so they too add in
quadrature, giving an error proportional to $\sqrt{K}\,\sigma_\varepsilon$
against a signal proportional to $\sqrt{K}\,\sigma_w\sigma_x$. **The two square
roots cancel**, and the relative error of a matmul output is independent of the
reduction width. That is a second piece of good news and it comes from the same
argument.

**What does not cancel is anything the model computes as a difference.** A
residual stream accumulates contributions that partly oppose each other, so the
signal can be small while the errors still add in quadrature — and the relative
error there is amplified by exactly the cancellation the architecture relies on.
This is the reason attention logits and normalisation statistics are more
sensitive than their magnitudes suggest, and it is why
{{ch:q-activation-kv}} treats them separately rather than as more tensors.

**The general rule: quantization is safe wherever the computation adds and
dangerous wherever it subtracts.** Depth is addition, so depth is safe. Residuals
and logit differences are subtraction, so they are not.

## 8. Implementation

```python {tier=A name=errors-through-depth}
"""Does quantization error compound through depth, or cancel?

The obvious worry about quantizing a deep network is that each layer adds error
to the layer below it, so a 1% perturbation at every layer of an 80-layer model
should arrive at the output as something enormous. If that were true, quantization
would not work at all, and it plainly does.

The reason it does not is worth measuring rather than asserting. Rounding errors
in different layers are approximately INDEPENDENT, and independent perturbations
add in quadrature rather than linearly -- so the output error grows like the
square root of depth, not like depth (eq:errors-add-in-quadrature).

This listing measures the growth rate directly, and then breaks the independence
on purpose to show what the assumption is worth.
"""
import numpy as np

rng = np.random.default_rng(239)

D, N = 128, 2048
MAX_L = 32


def quantize(W, bits, group=None):
    """Symmetric integer quantization with a scale per group of rows. group=None
    means one scale for the whole tensor."""
    qmax = 2 ** (bits - 1) - 1
    if group is None:
        s = np.max(np.abs(W)) / qmax
    else:
        g = np.max(np.abs(W.reshape(-1, group)), axis=1, keepdims=True) / qmax
        s = np.repeat(g, group, axis=1).reshape(W.shape)
    s = np.where(s == 0, 1e-12, s)
    return np.clip(np.round(W / s), -qmax, qmax) * s


def make_net(L, outlier_frac=0.0, outlier_scale=1.0):
    Ws = []
    for _ in range(L):
        W = rng.normal(size=(D, D)) / np.sqrt(D)
        if outlier_frac > 0:
            k = max(1, int(outlier_frac * D))
            cols = rng.choice(D, size=k, replace=False)
            W[:, cols] *= outlier_scale
        Ws.append(W)
    return Ws


def forward(Ws, X, bits=None, upto=None):
    h = X
    for W in Ws[:upto]:
        Q = W if bits is None else quantize(W, bits)
        h = np.tanh(h @ Q)
    return h


X = rng.normal(size=(N, D))
BITS = (8, 6, 4)

print(f"{D}-wide tanh network. Relative output error against depth, and the")
print("growth exponent p in error ~ depth^p fitted over the last half.")
print()
print(f"{'depth':>7}" + "".join(f"{str(b) + '-bit':>12}" for b in BITS))
print("-" * 43)

DEPTHS = (1, 2, 4, 8, 16, 32)
curves = {b: [] for b in BITS}
nets = {b: make_net(MAX_L) for b in BITS}
for L in DEPTHS:
    row = []
    for b in BITS:
        Ws = nets[b]
        ref = forward(Ws, X, upto=L)
        got = forward(Ws, X, bits=b, upto=L)
        e = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        curves[b].append(e)
        row.append(e)
    print(f"{L:>7}" + "".join(f"{v:>12.4e}" for v in row))


def exponent(depths, errs):
    """Fit error ~ depth^p on the log-log tail."""
    d = np.log(np.array(depths[2:], float))
    e = np.log(np.array(errs[2:], float))
    return float(np.polyfit(d, e, 1)[0])


print()
print(f"{'':>7}" + "".join(f"{'p = ' + f'{exponent(DEPTHS, curves[b]):.2f}':>12}"
                            for b in BITS))

print()
print()
print("Now break the independence: a few weight columns scaled up, so one")
print("shared scale factor must cover a much wider range.")
print()
print(f"{'outlier':>9}{'outlier':>9}{'8-bit err':>12}{'vs clean':>10}"
      f"{'6-bit err':>12}{'vs clean':>10}{'exponent':>10}")
print(f"{'columns':>9}{'scale':>9}{'at depth 16':>12}{'':>10}"
      f"{'at depth 16':>12}{'':>10}{'p':>10}")
print("-" * 73)

out_rows = {}
DD = (1, 2, 4, 8, 16)
for frac, sc in ((0.0, 1.0), (0.02, 4.0), (0.02, 16.0), (0.10, 16.0)):
    Ws = make_net(16, frac, sc)
    e8, e6 = [], []
    for L in DD:
        ref = forward(Ws, X, upto=L)
        e8.append(float(np.linalg.norm(forward(Ws, X, 8, L) - ref)
                        / np.linalg.norm(ref)))
        e6.append(float(np.linalg.norm(forward(Ws, X, 6, L) - ref)
                        / np.linalg.norm(ref)))
    p8 = exponent(DD, e8)
    out_rows[(frac, sc)] = (e8[-1], e6[-1], p8)
    b = out_rows[(0.0, 1.0)]
    print(f"{frac:>9.0%}{sc:>9.0f}{e8[-1]:>12.4e}{e8[-1]/b[0]:>10.1f}x"
          f"{e6[-1]:>12.4e}{e6[-1]/b[1]:>10.1f}x{p8:>10.2f}")

p8 = exponent(DEPTHS, curves[8])
p4 = exponent(DEPTHS, curves[4])
base = out_rows[(0.0, 1.0)]
mild = out_rows[(0.02, 4.0)]
bad = out_rows[(0.02, 16.0)]
worst = out_rows[(0.10, 16.0)]
print(f"""
The first table answers the question the chapter exists for, and the answer is in
the fitted exponents rather than in the errors themselves.

Error grows as depth to the power {p8:.2f} at 8 bits and {p4:.2f} at 4 bits.
Both are close to one half. That is the signature of INDEPENDENT perturbations
adding in quadrature: L layers each contributing an error of size e produce an
output error of about e times the square root of L, not e times L and certainly
not e compounding multiplicatively (eq:errors-add-in-quadrature).

The difference matters enormously at the depths real models have. At 80 layers,
linear compounding would multiply the per-layer error by 80 and multiplicative
compounding would be far worse still; square-root growth multiplies it by about
nine. That factor of nine, against a per-layer error that is already small, is
the entire reason a quantized 80-layer model still works.

It is worth being precise about why the errors are independent. Each weight's
rounding error is determined by where that weight happens to fall between two
representable values, which is essentially arbitrary and uncorrelated between
weights, between layers, and with the data flowing through. Nothing in the
computation aligns them, so they do not reinforce.

Which immediately identifies what would break the argument: anything that makes
the errors CORRELATED. The second table constructs exactly that.

Scaling up a small number of weight columns forces the shared scale factor to
cover a range it did not have to cover before. The step size grows for every
weight in the tensor, including the ordinary ones, so the errors are no longer
small independent perturbations -- they are a systematic coarsening driven by a
handful of entries.

At 2% of columns scaled by 4, the 8-bit error at depth 16 rises from
{base[0]:.2e} to {mild[0]:.2e} -- {mild[0]/base[0]:.1f}x. At a scale of 16,
{bad[0]:.2e}, which is {bad[0]/base[0]:.1f}x. With 10% of columns scaled by 16,
{worst[0]:.2e}: {worst[0]/base[0]:.1f}x the clean network, with no change to the
bit-width at all.

Put that beside what bit-width buys. Going from 8 bits to 6 on the CLEAN network
costs {base[1]/base[0]:.1f}x -- two whole bits, for a factor of four. Adding 2% of
columns at 16x scale, at 8 bits throughout, costs {bad[0]/base[0]:.1f}x.

So the distribution matters more than the bit-width, by a wide margin, and that
is not how the choice is usually framed. Teams argue about 4-bit against 8-bit
and accept whatever outlier structure the checkpoint happens to have, when the
second factor is the larger one.

One honest note on the exponent column. It sits at {base[2]:.2f} on the clean
network and drifts up to {worst[2]:.2f} in the worst outlier row. That drift is
not a change in how errors accumulate -- it is a ceiling effect. A relative error
near 1 means the output has become uncorrelated with the reference, and a
saturating quantity cannot keep following a power law. Where the errors are still
small, the exponent stays near one half.

Which leaves the practical reading. Outliers do not break the way errors
accumulate; they inflate the per-layer error that then accumulates in the same
square-root fashion. The quadrature argument survives and the constant in front of
it does not.

That distinction is what makes the problem tractable. If outliers changed the
accumulation structure, deep models would be unquantizable and the only remedy
would be fewer layers. Because they inflate a local quantity instead, the remedy
is local -- give the outliers their own scale, or their own precision, or rotate
the basis so no coordinate dominates -- and cite:dettmers2022int8,
cite:xiao2023smoothquant, cite:lin2023awq and cite:tseng2024quipsharp are four
different ways of doing exactly that.""")
```

The first listing shows what governs damage at the network level. The second asks
what governs the per-layer error it accumulates.

```python {tier=A name=what-makes-a-model-fragile}
"""What makes a model fragile to quantization? Two answers, and one is a surprise.

The previous listing showed that quantization damage is governed by the per-layer
error rather than by depth. This one asks what governs the per-layer error, and
finds two things.

The first is how finely the scale factors are shared -- the group size, which is
the parameter ch:q-formats identified as more consequential than the bit-width
everyone quotes (eq:group-size-dominates).

The second is not about storage at all. cite:kumar2024precisionscaling reports
that post-training quantization damage INCREASES with the amount of pretraining
data, so a better-trained model is a MORE fragile one. That inverts the usual
intuition, and it is testable directly: train the same model for different lengths
and quantize each checkpoint identically.
"""
import numpy as np

rng = np.random.default_rng(241)


def quantize(W, bits, group=0):
    """Symmetric integer quantization. `group` is how many consecutive weights
    share one scale factor; 0 means the whole tensor shares one."""
    qmax = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    if group <= 0 or group >= flat.size:
        s = np.max(np.abs(flat)) / qmax
        s = max(s, 1e-12)
        return (np.clip(np.round(flat / s), -qmax, qmax) * s).reshape(W.shape)
    pad = (-flat.size) % group
    f = np.concatenate([flat, np.zeros(pad)])
    g = f.reshape(-1, group)
    s = np.maximum(np.max(np.abs(g), axis=1, keepdims=True) / qmax, 1e-12)
    q = (np.clip(np.round(g / s), -qmax, qmax) * s).reshape(-1)
    return q[:flat.size].reshape(W.shape)


D = 384
W = rng.normal(size=(D, D)) / np.sqrt(D)
W_OUT = W.copy()
cols = rng.choice(D, size=max(1, D // 50), replace=False)
W_OUT[:, cols] *= 16.0                      # 2% of columns, 16x larger

GROUPS = (0, 1024, 256, 64, 32)
LABEL = {0: "whole tensor", 1024: "1024", 256: "256", 64: "64", 32: "32"}
BITS = (8, 6, 4, 3)


def rel(A, B):
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


print(f"A {D}x{D} weight matrix. Relative error against group size and bits.")
print("Left block is a clean Gaussian; right block has 2% of columns 16x larger.")
print()
print(f"{'group':>14}" + "".join(f"{str(b) + 'b':>10}" for b in BITS)
      + f"{'':>4}" + "".join(f"{str(b) + 'b':>10}" for b in BITS))
print(f"{'size':>14}" + f"{'CLEAN':>40}" + f"{'':>4}" + f"{'WITH OUTLIERS':>40}")
print("-" * 88)

tab = {}
for g in GROUPS:
    c = [rel(quantize(W, b, g), W) for b in BITS]
    o = [rel(quantize(W_OUT, b, g), W_OUT) for b in BITS]
    tab[g] = (c, o)
    print(f"{LABEL[g]:>14}" + "".join(f"{v:>10.4f}" for v in c) + f"{'':>4}"
          + "".join(f"{v:>10.4f}" for v in o))

print()
print()
print("Now the second question: does training a model LONGER make it more")
print("fragile? Same architecture, same quantization, different checkpoints.")
print()

DI, H, DO_, N = 24, 64, 6, 2000
Wq = rng.normal(size=(DI, DO_))
Xtr = rng.normal(size=(N, DI))
Ytr = (np.tanh(Xtr @ Wq) + 0.4 * Xtr[:, :DO_]
       + 0.10 * rng.normal(size=(N, DO_)))
Xte = rng.normal(size=(2500, DI))
Yte = np.tanh(Xte @ Wq) + 0.4 * Xte[:, :DO_]


def init():
    return [rng.normal(size=(DI, H)) / np.sqrt(DI), np.zeros(H),
            rng.normal(size=(H, DO_)) / np.sqrt(H), np.zeros(DO_)]


def fwd(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def mse(p, X, Y):
    return float(((fwd(p, X)[1] - Y) ** 2).mean())


def step(p, m, v, t, lr=0.01):
    h, o = fwd(p, Xtr)
    d = 2 * (o - Ytr) / N
    dh = d @ p[2].T * (1 - h ** 2)
    g = [Xtr.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]
    for i in range(4):
        m[i] = 0.9 * m[i] + 0.1 * g[i]
        v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
        p[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (
            np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)


print(f"{'steps':>8}{'train':>10}{'test':>10}{'||W||':>9}"
      f"{'test after':>12}{'damage':>10}{'damage':>10}")
print(f"{'':>8}{'loss':>10}{'loss':>10}{'':>9}{'4-bit':>12}{'absolute':>10}"
      f"{'relative':>10}")
print("-" * 69)

p = init()
m = [np.zeros_like(w) for w in p]
v = [np.zeros_like(w) for w in p]
rows = {}
t = 0
for target in (50, 200, 800, 3200, 9600):
    while t < target:
        t += 1
        step(p, m, v, t)
    tr, te = mse(p, Xtr, Ytr), mse(p, Xte, Yte)
    nrm = float(np.sqrt(sum((w ** 2).sum() for w in p)))
    pq = [quantize(w, 4, 64) if w.ndim == 2 else w.copy() for w in p]
    teq = mse(pq, Xte, Yte)
    rows[target] = (tr, te, nrm, teq, teq - te, teq / te - 1)
    print(f"{target:>8}{tr:>10.4f}{te:>10.4f}{nrm:>9.2f}{teq:>12.4f}"
          f"{teq - te:>+10.4f}{teq / te - 1:>+10.1%}")

c, o = tab[0]
c64, o64 = tab[64]
c32, o32 = tab[32]
early, late = rows[50], rows[9600]
print(f"""
The first table makes the group-size point concrete, and it is best read by
comparing what a finer group buys against what a bit buys.

On the clean matrix at 4 bits, a whole-tensor scale gives {c[2]:.4f} and a group
of 64 gives {c64[2]:.4f} -- better by {c[2]/c64[2]:.1f}x, from one extra number
per 64 weights. What does a bit buy on the same matrix? Eight bits gives
{c[0]:.4f} against six bits' {c[1]:.4f}: a factor of {c[1]/c[0]:.1f} for two
bits, so roughly {(c[1]/c[0]) ** 0.5:.1f}x per bit.

So on clean weights, going from a tensor-wide scale to groups of 64 is worth
about a bit of width, and it costs a fraction of one -- a 16-bit scale per 64
weights is a quarter of a bit per weight, four times cheaper than the bit it
replaces.

The outlier block is where the argument becomes decisive rather than merely
favourable. At 8 bits the outliers cost {o[0]/c[0]:.1f}x at whole-tensor scale
({o[0]:.4f} against {c[0]:.4f}) and only {o32[0]/c32[0]:.1f}x at groups of 32
({o32[0]:.4f} against {c32[0]:.4f}).

That is the mechanism stated precisely. An outlier does damage by forcing a
shared scale factor to cover a range the other weights never use, which coarsens
the step for every weight sharing that scale. Shrink the group and the damage is
CONTAINED: the outlier still ruins its own group of 32, and every other group is
untouched (eq:group-size-dominates).

So the cost of an outlier is proportional to how many weights share its scale.
That is why every practical 4-bit format has a group size somewhere in its name,
and why quoting "4-bit" without it is an incomplete specification -- the number
people quote is the one that matters less.

The second table inverts an intuition, and it does so sharply.

The model is trained for progressively longer and quantized identically at each
checkpoint. Training works: test loss falls from {early[1]:.4f} at 50 steps to
{rows[800][1]:.4f} at 800. The question is what the same 4-bit quantization costs
at each point.

Absolute damage rises from {early[4]:+.4f} to {late[4]:+.4f} -- a factor of
{late[4]/early[4]:.0f} -- and relative damage from {early[5]:+.1%} to
{late[5]:+.1%}. The longer-trained model is hurt far more by an identical
operation. Both accountings agree, which matters: the relative figure alone could
be an artefact of a shrinking denominator, and the absolute figure rules that out.

The weight-norm column names the mechanism: {early[2]:.1f} at 50 steps rising to
{late[2]:.1f} at {max(rows)}. Training moves weights away from their small random
initialisation, so the tensor's dynamic range grows and a fixed number of levels
has more ground to cover. The same 4 bits are being asked to span a wider
interval, so each step is coarser.

Two honest caveats, because this is a small experiment standing in for a large
result. The weight growth here is unchecked -- there is no weight decay -- and
regularisation would slow it, which is a genuine partial mitigation that real
training runs already apply. And the later checkpoints are overfitting: test loss
bottoms at {rows[800][1]:.4f} around step 800 and rises after, so some of what the
last rows measure is a model that was already getting worse.

Neither caveat removes the effect. cite:kumar2024precisionscaling establishes it
properly at pretraining scale, without either confound, and finds it strong enough
that past some number of tokens additional pretraining makes the post-quantization
model WORSE. This listing shows the direction and the mechanism; the paper shows
the crossover.

The practical consequence is the uncomfortable one. A quantization recipe is
validated on a CHECKPOINT, not on an architecture. The same model family trained
longer may not survive the same recipe, so "we use 4-bit for this model" is a
claim with an expiry date, and the only way to know is to measure again when the
checkpoint changes.""")
```

## 9. Practical Example

**Errors add in quadrature.** Output error grows as depth$^{0.52}$ at 8 bits,
depth$^{0.50}$ at 6, depth$^{0.49}$ at 4 —
{{eq:errors-add-in-quadrature}} confirmed three times across an order of magnitude
in per-layer error. **At 80 layers that is a factor of nine, not eighty**, and it
is why quantized deep models work at all.

**Which identifies the threat as correlation, not bit-width.** Scaling 2% of
weight columns by 16× raises the 8-bit error at depth 16 from **4.23e-02 to
6.94e-01 — 16.4×**; 10% of columns costs **27.9×**. Dropping from 8 bits to 6 on
the clean network costs **4.0×**.

> **IMPORTANT:** The exponent stays near 0.5 through the mild outlier rows and
> drifts to 0.80 in the severe ones, which is a **ceiling effect** — a relative
> error near 1 means the output is uncorrelated and cannot keep following a power
> law. **Outliers inflate the per-layer error; they do not change how it
> accumulates.** That is what keeps the remedy local.

**An outlier is a statement about the step size.**
{{eq:outlier-inflates-the-step}}: the group's scale is set by its maximum, so one
16× value gives the ordinary weights **fewer than one effective level** at 4 bits
({{eq:effective-levels}}: 14/16 ≈ 0.9). **The damage lands on the thousands of
ordinary weights, not on the outlier.**

**So group size is the control.** At 8 bits, outliers cost **5.8×** with a
whole-tensor scale and **2.2×** with groups of 32. At 4 bits clean: **0.1813**
tensor-wide, **0.1074** at 64, **0.0970** at 32.

**Tensor to 64 is worth ~1.7×, about a bit — and costs a quarter of one**
({{eq:group-size-cost}} with a 16-bit scale). **Which is why "4-bit" without a
group size is an incomplete specification.** And the flattening from 64 to 32 is
{{eq:gaussian-max}}'s logarithm.

**Finally, fragility grows with training.** Identical 4-bit quantization at
successive checkpoints: relative damage **+3.8% → +1840% → +4008%**, absolute
damage **+0.0079 → +0.2412** — a factor of **31**, so it is not a denominator
artefact. Weight norm grew **8.88 → 50.98**
({{eq:fragility-grows-with-training}}, both mechanisms visible).

**Two honest caveats.** There is no weight decay here, so the norm growth is
unchecked and real training would slow it. And the later checkpoints are
overfitting — test loss bottoms at **0.0030** around step 800 and rises after.
**Neither removes the effect**; {{cite:kumar2024precisionscaling}} establishes it
at pretraining scale without either confound, and finds a crossover this
experiment is far too small to reach.

## 10. Production Considerations

**Measure $M/m$ per tensor** before choosing a bit-width.
{{eq:effective-levels}} converts it into what you actually have.

**Always specify the group size** alongside the bit-width. Half a specification is
not a specification.

**Prefer smaller groups over more bits** when the trade is available — it is
usually four times cheaper for the same effect.

**Re-validate the recipe when the checkpoint changes**, even for the same
architecture ({{eq:fragility-grows-with-training}}).

**Do not budget error linearly in depth.** {{eq:errors-add-in-quadrature}} — a
$\sqrt{L}$ budget is the right one and a linear budget is needlessly
conservative.

**Watch for saturation when interpreting error metrics.** A relative error near 1
means uncorrelated, and ratios between saturated numbers are meaningless.

**Measure per-layer sensitivity** rather than allocating bits uniformly.

## 11. Common Mistakes

**Assuming errors compound multiplicatively**, and concluding deep models cannot
be quantized.

**Comparing bit-widths while ignoring group size.**

**Treating an outlier as a problem for the outlier**, rather than for its
neighbours.

**Reusing a quantization recipe across checkpoints** of the same architecture.

**Reading ratios between saturated error values.**

**Allocating bits uniformly across layers** when $\|J_\ell\|$ is not uniform.

**Applying the uniform-noise model at 2–3 bits**, where it does not hold.

## 12. Failure Modes

**A method validated on a small model fails on a large one.** Cause: outliers
emerge with scale ({{cite:dettmers2022int8}}).

**A recipe that worked last quarter fails on the new checkpoint.** Cause:
{{eq:fragility-grows-with-training}}.

**4-bit works for one layer type and not another.** Cause: different $M/m$ per
tensor — measure it.

**Quality collapses between 4 and 3 bits rather than degrading.** Cause:
{{eq:effective-levels}} crossing one, plus the uniform-noise model breaking down.

**Error metrics stop distinguishing configurations.** Cause: saturation.

**Deeper model quantizes better than expected.** Cause: correct —
{{eq:errors-add-in-quadrature}} is sublinear, and intuition is usually pessimistic
here.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| more bits | memory, bandwidth | the blunt lever |
| smaller groups | $c/g$ bits, kernel complexity | almost always first |
| outlier isolation | a second precision path | {{cite:dettmers2022int8}} |
| difficulty migration | calibration | {{cite:xiao2023smoothquant}} |
| error compensation | calibration | {{cite:frantar2023gptq}} |
| per-layer bit allocation | tooling | when sensitivity varies a lot |
| quantization-aware training | a training run | when post-hoc is not enough |

**The last row is the escape hatch and it is expensive.**
{{eq:fragility-grows-with-training}} suggests why it works: training *with* the
quantization present finds an optimum that is flat in the directions rounding
moves, instead of finding a sharp one and then perturbing it.

## 14. Evaluation

**Report group size, scale precision, and bit-width together.**

**Report $M/m$** for the tensors that matter.

**Report the checkpoint**, not just the model family.

**Report per-layer error as well as end-to-end**, since
{{eq:errors-add-in-quadrature}} makes them relate non-obviously.

**Never compare saturated error values.**

## 15. Advanced Concepts

**Independence is an assumption, not a theorem.** {{maturity:MATURE}}
{{eq:errors-add-in-quadrature}} holds because nothing aligns the errors. Anything
that does — a systematic rounding bias, a shared calibration artefact, quantizing
a residual stream and its skip connection with the same scale — restores the
linear term.

**Per-layer sensitivity is measurable and rarely measured.**
{{maturity:EMERGING}} $\|J_\ell\|$ varies substantially across a transformer, so
uniform bit allocation is leaving quality on the table. The k-quant families in
{{ch:q-gguf}} do this by heuristic rather than by measurement.

**Quantization-aware training as flatness-seeking.** {{maturity:MATURE}}
Reading {{eq:fragility-grows-with-training}} backwards: if a sharp optimum is
fragile, training under quantization noise selects a flat one. That connects QAT
to the flat-minima literature, which is not how it is usually presented.

**The uniform-noise model's boundary.** {{maturity:EMERGING}}
{{eq:quantization-noise-variance}} needs many levels. Below about 4 bits the
error is comparable to the signal and correlates with it, which is why
{{cite:egiazarian2024aqlm}} and {{cite:tseng2024quipsharp}} change the
representation rather than refine the rounding.

**Fragility as a training-time decision.** {{maturity:RESEARCH FRONTIER}}
{{cite:kumar2024precisionscaling}} implies that a model *intended* for
quantization should be trained differently — weight decay, precision schedule,
possibly a shorter run. Almost nobody plans this way, and the arithmetic says
they should.

## 16. Connection to Previous Chapters

{{ch:q-formats}}'s {{eq:scale-group-condition}} is this chapter's
{{eq:outlier-inflates-the-step}} made quantitative, and its
{{eq:scale-factor-as-exponent}} is priced by {{eq:group-size-cost}}.
{{ch:ft-lora}}'s {{eq:forgetting-quadratic}} reappears in
{{eq:fragility-grows-with-training}} — the same quadratic, applied to a
perturbation that comes from rounding rather than from an optimiser.
{{ch:dl-normalization}} explains why activation magnitudes are what they are, which
is where {{ch:q-activation-kv}}'s outliers come from.
Forward: {{ch:q-int8-int4}} is four responses to
{{eq:outlier-inflates-the-step}}; {{ch:q-gguf}} allocates bits per tensor using
this chapter's sensitivity argument; {{ch:q-memory-math}} converts
{{eq:group-size-cost}} into bytes.

## 17. Exercises

1. Derive {{eq:quantization-noise-variance}} for a uniform quantizer and state
   where the assumption fails.
2. Using {{eq:effective-levels}}, compute $n_{\text{eff}}$ at 4 bits for
   $M/m = 4, 8, 16, 32$. At which ratio does 4-bit stop being meaningful?
3. Show that halving the group size and adding one bit have the same effect on
   {{eq:step-from-bits}} when the group maximum halves.
4. From {{eq:group-size-cost}}, find the group size at which a 16-bit scale costs
   half a bit per weight, and compare against the measured benefit.
5. In `errors-through-depth`, make the rounding errors correlated by adding a
   constant bias to every quantized weight. What exponent do you measure?
6. In `what-makes-a-model-fragile`, add weight decay to the training loop. How
   much of the fragility growth does it remove?
7. Derive {{eq:gaussian-max}} and use it to predict the clean-weight benefit of
   going from group 64 to group 32.
8. For a model you have: measure $M/m$ per weight tensor and identify which tensor
   would fail first at 4 bits.

## 18. Interview Questions

1. Why does quantization error not explode through an 80-layer model?
2. What assumption is that answer relying on, and what would break it?
3. What does an outlier actually do?
4. Which matters more, one more bit or half the group size? Justify.
5. Why is "4-bit" an incomplete specification?
6. Your recipe worked on last quarter's checkpoint and fails on this one. What
   changed?
7. Why might a longer-trained model be harder to quantize?
8. When does the uniform-noise model of quantization error stop applying?
9. How would you decide bit allocation across layers?
10. Why is quantization-aware training related to flat minima?

## 19. Research Questions

1. {{eq:errors-add-in-quadrature}} assumes independence. How well does it hold
   across real transformer layers, and which architectural choices correlate the
   errors?
2. {{eq:fragility-grows-with-training}} has two mechanisms — norm growth and
   curvature. Which dominates at scale, and does weight decay separate them?
3. Per-layer sensitivity is measurable. How much quality does optimal bit
   allocation recover over uniform, at equal average bits?
4. {{cite:kumar2024precisionscaling}} finds a crossover in pretraining tokens. Is
   there a training-time intervention that moves it, and what does it cost?
5. Below 4 bits the noise model fails. Is there a tractable replacement that
   predicts which weights will be damaged, rather than only how much?

## 20. Chapter Summary

**Quantization error grows as the square root of depth**, not linearly: measured
exponents **0.52, 0.50, 0.49** at 8, 6 and 4 bits
({{eq:errors-add-in-quadrature}}). Rounding errors are independent because nothing
in the computation aligns them, so they add in quadrature — **a factor of nine at
80 layers rather than eighty**, which is the entire reason the practice works.

**Which makes correlation the threat, and the bit-width a side issue.** Scaling
2% of weight columns by 16× cost **16.4×** at fixed 8 bits; 10% cost **27.9×**;
dropping two bits cost **4.0×**. **The distribution matters more than the width.**

**An outlier is a statement about the step size.**
{{eq:outlier-inflates-the-step}}: it raises the group's maximum, so it coarsens
the step for its *neighbours*, and {{eq:effective-levels}} shows a 16× outlier
leaves 4-bit weights with **fewer than one effective level**. The damage lands on
the thousands of ordinary weights.

**So group size is the control, and it is cheap.** Outliers cost **5.8×**
tensor-wide and **2.2×** at groups of 32. Tensor to 64 is worth about a bit and
costs a quarter of one ({{eq:group-size-cost}}), with the flattening beyond that
being {{eq:gaussian-max}}'s logarithm. **"4-bit" without a group size is half a
specification, and it is the less important half.**

**And the exponent stayed near 0.5 through it all.** Outliers inflate the
per-layer error without changing how it accumulates — which is why the remedy is
**local**, and why {{ch:q-int8-int4}}'s four methods can all work by acting on a
handful of coordinates.

**Finally, fragility grows with training.** Relative damage **+3.8% → +4008%**
and absolute damage **31×** across checkpoints of one model under identical
quantization, with weight norm growing **8.88 → 50.98**
({{eq:fragility-grows-with-training}}). **A better-trained model is a more
fragile one**, and {{cite:kumar2024precisionscaling}} finds the effect strong
enough at scale that additional pretraining eventually makes the quantized model
worse.

Which leaves the sentence to carry into the rest of the part: **a quantization
recipe is validated on a checkpoint, not on an architecture** — and nothing in a
config file records the difference.

## 21. Further Reading

{{cite:kumar2024precisionscaling}} is the most important recent paper in this part
and the least absorbed; read it for the claim that post-training quantization
damage grows with pretraining data, which turns robustness into a per-checkpoint
measurement.
{{cite:dettmers2022int8}} for outliers as an emergent, scale-dependent phenomenon
— this chapter's {{eq:outlier-inflates-the-step}} is what they do once they exist.
{{cite:dettmers2023case4bit}} for the model-size/precision trade at fixed memory,
and note that its caveats are about block size and data type, which is exactly
what {{eq:group-size-dominates}} predicts.
{{cite:xiao2023smoothquant}} and {{cite:frantar2023gptq}} as the two responses
developed in {{ch:q-int8-int4}}, read here for what they are responding to.
