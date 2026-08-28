---
id: mm-cv-fundamentals
number: 118
part: XIII
tier: full
status: draft
requires: [dl-cnns, dl-backprop, math-matrices, math-derivatives]
provides: [image-as-tensor, weight-sharing, equivariance-versus-invariance,
           receptive-field-arithmetic, effective-receptive-field,
           resolution-budget, feature-map-cost, jump-localisation]
citations: [lecun1998, krizhevsky2012, simonyan2015vgg, russakovsky2015ilsvrc,
            ronneberger2015unet, liu2022convnext]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what an image is as a tensor
and what a dense layer assumes about it that is false; define convolution as
**weight sharing plus locality** and derive the parameter count that follows;
distinguish **equivariance** from **invariance** and say which operation supplies
which; compute a receptive field from a layer specification and — the part that
matters — explain why the **effective** receptive field is smaller by a factor
that grows with depth, and demonstrate it; and use "resolution is the budget" as
the frame that every later chapter in this part spends.

## 2. Why This Matters

A reader in 2026 will build almost nothing in this chapter. They will call a
vision-language model, and it will work.

**And then it will fail in ways that only make sense from here.** It will not read
the small print on a scanned invoice. It will miss the one distant object that
mattered. It will describe a chart's trend correctly and get its values wrong. It
will find an object and be vague about where it is.

Every one of those is a *resolution* failure, and resolution is the subject of
this chapter. A vision tower is a stack of the operations described here; when it
downsamples an image to a grid of patch embeddings, it makes a decision about what
can still be seen, and no amount of language model behind it recovers what was
discarded.

The specific idea worth carrying out of this chapter — and it is measured in
{{sec:9-practical-example}} rather than asserted — is that **a deep network sees
much less than its specification claims.** The standard receptive-field formula
counts every input pixel that *can* influence an output. Weight that by how much
influence actually arrives and the window shrinks by a factor that grows as
$\sqrt{\text{depth}}$: at ten layers the effective field is about half the
theoretical one, at twenty layers about a third, which is a **ninth of the area**.
The rule of thumb everyone applies — make the receptive field at least as large as
the object — is applied to the wrong number.

{{maturity:ESTABLISHED}} Everything in this chapter is settled and thirty years
old. It is here because it is the vocabulary the rest of the part needs.

## 3. Prerequisites

{{ch:dl-cnns}} for convolution as a layer — this chapter revisits it as a
*prior* rather than as an operation; {{ch:dl-backprop}} for how the gradient
reaches a shared weight; {{ch:math-matrices}} for shapes; and
{{ch:math-derivatives}} for the chain rule that the receptive-field argument is
about.

## 4. Intuitive Explanation

### An image is a tensor with structure a dense layer cannot use

An RGB image is an array of shape $(H, W, 3)$ — height, width, channels. Flatten
it and you have a vector of $3HW$ numbers, which is what a dense layer wants.

**The flattening is where the damage is done.** Two facts about images survive in
the array and die in the vector:

1. **Neighbouring pixels are related.** Pixel $(i,j)$ and pixel $(i,j+1)$ are
   almost always similar and almost always part of the same thing.
2. **Content is translation-invariant.** A cat three pixels to the left is still
   a cat.

A dense layer knows neither. To it, position 47 and position 48 are two unrelated
coordinates, and a shape learned at one location teaches it nothing about the same
shape elsewhere. {{sec:9-practical-example}} trains one and shows exactly this:
perfect on centred shapes, **0.346 on the same shapes moved** — where chance is
0.333.

### Convolution is a claim about the world

A convolution makes two assumptions and gets two properties in return:

> **Locality.** A feature can be detected from a small neighbourhood. So each
> output looks at a $k \times k$ window rather than the whole image.
>
> **Stationarity.** A feature worth detecting in one place is worth detecting in
> every place. So the *same* weights are used at every position.

Stationarity — weight sharing — is the load-bearing half, and it is worth being
precise about what it buys. It is not primarily that there are fewer parameters.
It is that **evidence gathered anywhere updates the same weights.** A vertical
edge seen in the corner improves the vertical-edge detector used in the centre.
The dense model has to learn the same thing once per location; the convolution
learns it once.

That is what an architectural prior *is*: a statement about the world, paid for
once in design rather than repeatedly in training examples.
{{sec:9-practical-example}} makes the distinction concrete — given training data
covering every position, the dense model recovers most of the gap, so the prior
was buying **sample efficiency**, not capability.

### Equivariance and invariance are not the same word

This distinction confuses people permanently, and it is one sentence.

> **Equivariant:** shift the input, the output shifts too. *Convolution is
> equivariant.*
>
> **Invariant:** shift the input, the output does not change. *Pooling makes
> things invariant.*

Convolution preserves *where*. Pooling discards it. A classifier wants invariance
— "there is a cat" regardless of position. A detector or a segmenter wants
equivariance — it needs the *where*. **The entire architecture of this part is
about when to discard position and how to get it back**, which is why
{{ch:mm-segmentation}}'s U-Net exists and why {{ch:mm-detection}}'s detectors keep
a spatial grid.

### The receptive field, and the trap in it

A unit deep in the stack sees a bounded window of the input. Stack ten $3\times3$
convolutions and each output sees a $21 \times 21$ input window.

**That number is correct and it is the wrong number to plan with.** It counts
every pixel that *can* influence the output. Ask instead how much influence
actually reaches each pixel and the answer is a Gaussian bump, because influence
reaching a far pixel must survive one specific path through every layer, and there
are vastly more paths to the centre than to the edge.

So the *effective* field is smaller, and the gap grows with depth:

```text
   10 layers of 3x3:   theory 21 px    effective 10.3 px    ratio 0.49
   20 layers of 3x3:   theory 41 px    effective 14.6 px    ratio 0.36
```

Theory grows **linearly** in depth; effective grows like its **square root**. The
deeper the backbone, the more the formula overstates. And this is not a claim
about trained weights — {{sec:9-practical-example}}'s simulation gives every layer
uniform weights, so the concentration comes from *stacking*, not from training.

## 5. Formal Explanation

### 5.1 The dense layer's cost, and its real problem

An image $X \in \mathbb{R}^{H \times W \times C}$ into a dense layer with $m$
units:

$$ \#\text{params} = HWC \cdot m + m $$ (eq:dense-parameter-count)

At $224 \times 224 \times 3$ into 1000 units that is 150 million parameters for
one layer. Bad, and **not the objection**. The objection is structural: the layer
is a function of a permutation-arbitrary vector, so

$$ f_{\text{dense}}(X) \quad\text{and}\quad f_{\text{dense}}(T_\delta X) \quad\text{are unrelated for a shift } T_\delta $$ (eq:dense-no-prior)

Nothing in the parameterisation ties them. Any relationship must be learned
separately for every $\delta$, from examples.

### 5.2 Convolution

For input $X$, kernel $K$ of size $k \times k$, output channel $o$:

$$ Y[i,j,o] = b_o + \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}\sum_{c} K[u,v,c,o]\, X[i+u,\, j+v,\, c] $$ (eq:convolution)

(Deep learning's "convolution" is cross-correlation — no kernel flip. The
distinction is immaterial because $K$ is learned.)

The parameter count is the point:

$$ \#\text{params} = k^2 C C_{\text{out}} + C_{\text{out}} \quad\text{— independent of } H \text{ and } W $$ (eq:conv-parameter-count)

**Independent of image size.** {{eq:dense-parameter-count}} scales with pixels;
{{eq:conv-parameter-count}} does not.

### 5.3 Equivariance, proved

Let $T_\delta$ shift by $\delta$. For the convolution operator $\Phi$:

$$ \Phi(T_\delta X) = T_\delta(\Phi X) $$ (eq:translation-equivariance)

which follows immediately from {{eq:convolution}}: the sum defining $Y[i+\delta,j]$
on the shifted input is termwise identical to the one defining $Y[i,j]$ on the
original. **This is the property weight sharing buys**, and it holds exactly,
before any training.

Global pooling then converts equivariance to invariance:

$$ \text{maxpool}\big(\Phi(T_\delta X)\big) = \text{maxpool}\big(T_\delta(\Phi X)\big) = \text{maxpool}(\Phi X) $$ (eq:pooling-invariance)

since a max over all positions does not care which position won. **Two operations,
two different jobs**, and conflating them is why people are surprised that a
segmentation network cannot use global pooling.

> **Two caveats, both real.** {{eq:translation-equivariance}} holds exactly only
> for integer shifts and ignoring borders; and *strided* convolution is
> equivariant only to shifts that are multiples of the stride, which is a known
> source of aliasing. Neither changes the argument; both matter when you are
> debugging a model that is oddly sensitive to one-pixel crops.

### 5.4 Receptive field arithmetic

Track two quantities through the stack: the field $r$ and the **jump** $\jmath$
(input pixels per output step). Initialise $r = 1$, $\jmath = 1$, and for each
layer with kernel $k$, stride $s$, dilation $d$:

$$ r \leftarrow r + \big((k-1)\,d\big)\,\jmath, \qquad \jmath \leftarrow \jmath \cdot s $$ (eq:receptive-field)

Three consequences worth reading off directly:

- **Depth grows $r$ linearly.** $n$ layers of $3\times3$ at stride 1 give
  $r = 2n+1$.
- **Stride grows $r$ geometrically** — and grows $\jmath$ with it, coarsening the
  output grid. Field bought with resolution.
- **Dilation grows $r$ geometrically at $\jmath = 1$.** Same field, dense output,
  which is why dilated convolutions live in segmentation.

### 5.5 The effective receptive field

Weight each input pixel by the influence that reaches it. Under
{{eq:convolution}}, influence spreads over $k$ positions per layer; composing $n$
such spreads is an $n$-fold convolution of a bounded kernel with itself, so by the
central limit theorem the profile tends to a Gaussian with

$$ \sigma^2 \propto n \quad\Longrightarrow\quad r_{\text{eff}} \propto \sqrt{n}, \qquad r_{\text{theory}} \propto n $$ (eq:effective-receptive-field)

and therefore

$$ \frac{r_{\text{eff}}}{r_{\text{theory}}} \propto \frac{1}{\sqrt{n}} $$ (eq:erf-depth-scaling)

**{{eq:erf-depth-scaling}} is the chapter's quantitative result.** It says the
overstatement is not a fixed factor to memorise — it is a function of depth, and
{{sec:9-practical-example}} confirms the exponent by doubling depth and watching
the effective field grow by $\sqrt{2}$.

### 5.6 Resolution as the budget

A feature map costs

$$ \text{memory} \propto H \cdot W \cdot C, \qquad \text{convolution FLOPs} \propto H \cdot W \cdot k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} $$ (eq:feature-map-cost)

Both are linear in *area*, so halving each spatial dimension quarters the cost.
That is why every architecture downsamples early, and why every architecture in
this part is fighting the same fight:

$$ \text{context wants downsampling} \;\;\bot\;\; \text{localisation wants resolution} $$ (eq:resolution-tension)

{{eq:resolution-tension}} is the part's through-line.
{{ch:mm-segmentation}}'s skip connections, {{ch:mm-vit}}'s patch size, and
{{ch:mm-vlms}}'s visual token budget are three answers to it.

## 6. Mathematical Foundation

### 6.1 Receptive field, worked by hand

VGG-style: two $3\times3$ convolutions then a $2\times2$ stride-2 pool, repeated.
Start $r=1$, $\jmath=1$.

| layer | $k$ | $s$ | $r$ after | $\jmath$ after |
|---|---|---|---|---|
| conv3 | 3 | 1 | $1 + 2(1) = 3$ | 1 |
| conv3 | 3 | 1 | $3 + 2(1) = 5$ | 1 |
| pool2 | 2 | 2 | $5 + 1(1) = 6$ | 2 |
| conv3 | 3 | 1 | $6 + 2(2) = 10$ | 2 |
| conv3 | 3 | 1 | $10 + 2(2) = 14$ | 2 |
| pool2 | 2 | 2 | $14 + 1(2) = 16$ | 4 |

After two blocks: $r = 16$, $\jmath = 4$. Continue to four blocks and
{{sec:9-practical-example}} reports $r = 76$, $\jmath = 16$ — so one unit at that
depth summarises a $76\times76$ window, and consecutive units are 16 input pixels
apart.

**Read the jump as the localisation error.** At $\jmath = 16$ the network cannot
distinguish positions finer than 16 pixels without help, which is precisely the
help {{ch:mm-detection}} and {{ch:mm-segmentation}} spend their architectures
providing.

### 6.2 Why the effective field is Gaussian

Model each layer as spreading influence uniformly over its $k$ taps: a discrete
uniform kernel $u_k$ with variance $\sigma_1^2 = (k^2-1)/12$. Composing $n$
layers convolves $u_k$ with itself $n$ times, and variances add:

$$ \sigma_n^2 = n\,\frac{k^2 - 1}{12} $$ (eq:variance-adds)

For $k=3$: $\sigma_n^2 = 2n/3$. Take the effective width as $\pm 2\sigma$:

$$ r_{\text{eff}} = 4\sigma_n = 4\sqrt{2n/3} $$ (eq:erf-worked)

At $n = 10$: $4\sqrt{6.67} = 10.3$ pixels — and the simulation measures **10.3**.
At $n = 20$: $4\sqrt{13.3} = 14.6$ — measured **14.6**. Against theoretical fields
of 21 and 41.

**The closed form and the simulation agree to three significant figures**, which
is the useful outcome: {{eq:erf-worked}} can be used directly, with no simulation,
to check whether a backbone can see an object.

> **MATH NOTE:** The uniform-spreading assumption is doing real work and it is the
> *conservative* choice. Trained kernels are not uniform, and a kernel that
> concentrates weight centrally makes the effective field *smaller* still. So
> {{eq:erf-worked}} is an upper bound on what a trained network sees, and the
> direction of the error is the safe one for the design rule that follows from
> it.

### 6.3 The design rule

To detect an object of size $D$ pixels, the layer making the decision needs

$$ r_{\text{eff}} \gtrsim D \quad\Longrightarrow\quad 4\sqrt{\tfrac{(k^2-1)n}{12}} \gtrsim D \quad\Longrightarrow\quad n \gtrsim \frac{12 D^2}{16(k^2-1)} $$ (eq:depth-for-object)

For $k=3$, $D=100$: $n \gtrsim 938$ layers. **Which is absurd, and that is the
point** — it is why nobody builds detectors from stride-1 $3\times3$ stacks. The
practical architectures buy field through stride and dilation, which
{{eq:receptive-field}} grows *geometrically* rather than linearly, at the cost of
{{eq:resolution-tension}}.

## 7. Internal Mechanics

```mermaid {#fig:cv-stack caption="What a convolutional stage does to shape and to what a unit can see. Spatial dimensions shrink and channels grow, so cost per stage stays roughly flat (eq:feature-map-cost) while the receptive field grows. The dashed annotation is the jump: how far apart two neighbouring units' windows are, which is the localisation resolution available at that depth."}
flowchart LR
    I["image<br/>224 x 224 x 3"] -->|"conv 3x3<br/>r=3, jump=1"| S1["112 x 112 x 64<br/>after stride 2"]
    S1 -->|"conv x2 + stride"| S2["56 x 56 x 128<br/>r~30, jump=4"]
    S2 -->|"conv x2 + stride"| S3["28 x 28 x 256<br/>r~90, jump=8"]
    S3 -->|"conv x2 + stride"| S4["14 x 14 x 512<br/>r~200, jump=16"]
    S4 -.->|"EFFECTIVE r ~ 70,<br/>not 200"| S4
    S4 -->|"global pool:<br/>discards position"| V["512-vector"]
    S3 -.->|"kept for detection<br/>and segmentation"| K["spatial features"]
```

### 7.1 The standard stage, and why it is shaped that way

Every convolutional backbone is the same loop: convolve, activate, normalise,
downsample; halve the spatial dimensions and double the channels. The doubling is
not decoration — it keeps {{eq:feature-map-cost}} roughly constant per stage,
since area falls $4\times$ and channels rise $2\times$.

**What is being traded is explicit:** spatial resolution is exchanged for channel
capacity and receptive field. By the last stage the network knows a great deal
about *what* and very little about *where*.

### 7.2 Padding, stride, and the arithmetic that bites

Output size for input $n$, kernel $k$, padding $p$, stride $s$:

$$ n_{\text{out}} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 $$

`same` padding ($p = (k-1)/2$ at $s=1$) preserves size and is the default for a
reason: without it, a 20-layer stack loses 40 pixels, and it loses them from the
*edges*, which is where the objects you missed usually were.

### 7.3 What pooling is really for

Three jobs, and they are often conflated:

| Job | What does it | Note |
|---|---|---|
| reduce cost | stride or pool | the main reason in practice |
| grow receptive field | stride | geometric, per {{eq:receptive-field}} |
| discard position | **global** pooling only | local pooling gives *tolerance*, not invariance |

**Local max-pooling does not give invariance.** It gives a small amount of shift
tolerance. Only a global pool over the whole map gives
{{eq:pooling-invariance}}'s exact invariance, and only for the classification head
that sits after it.

## 8. Implementation

```python {tier=A name=receptive-field-arithmetic}
"""Receptive field: the number every backbone user computes, and computes wrong.

A unit deep in a convolutional network sees a bounded window of the input, and
eq:receptive-field gives its size. That formula is exact and it is also
misleading, because it counts every input pixel that CAN influence the output,
not every pixel that DOES.

This listing computes both. The theoretical field is the support of the
influence: which pixels are reachable at all. The EFFECTIVE field weights each
pixel by how much influence actually reaches it, which is what decides whether a
feature has really seen an object (eq:effective-receptive-field).
"""
import numpy as np

# A stack is a list of (kernel, stride, dilation) triples.
STACKS = {
    "10x conv3, stride 1": [(3, 1, 1)] * 10,
    "20x conv3, stride 1": [(3, 1, 1)] * 20,
    "VGG-ish: 2 conv + pool, x4": [(3, 1, 1), (3, 1, 1), (2, 2, 1)] * 4,
    "5x conv3, stride 2": [(3, 2, 1)] * 5,
    "dilated 1,2,4,8,16": [(3, 1, d) for d in (1, 2, 4, 8, 16)],
}


def theoretical_rf(stack):
    """eq:receptive-field, accumulated forward: r <- r + (k_eff - 1) * jump."""
    r, jump = 1, 1
    for k, s, d in stack:
        r += ((k - 1) * d) * jump
        jump *= s
    return r, jump


def influence_profile(stack, width=1024):
    """Propagate INFLUENCE, not just support.

    Start with a single unit of influence at one output position and push it
    backwards through the stack. Each layer spreads a unit's influence uniformly
    over the k input positions it read. The support of the result is the
    theoretical field; its shape is what the effective field measures.

    Uniform spreading is the honest choice here: it assumes every weight
    contributes equally, so any concentration in the result comes from the
    STRUCTURE of the stack rather than from an assumption about the weights.
    """
    infl = np.zeros(width)
    infl[width // 2] = 1.0
    for k, s, d in reversed(stack):
        # Upsample by the stride: one output position came from every s-th input.
        if s > 1:
            up = np.zeros(width)
            centre = width // 2
            idx = centre + (np.arange(width) - centre) * s
            keep = (idx >= 0) & (idx < width)
            up[idx[keep]] = infl[keep]
            infl = up
        # Spread over the k dilated taps this layer read.
        spread = np.zeros(width)
        offs = (np.arange(k) - (k - 1) / 2) * d
        for o in offs:
            spread += np.roll(infl, int(round(o))) / k
        infl = spread
    return infl


def effective_rf(infl):
    """Two standard deviations of the influence distribution, in pixels --
    the window that carries about 95% of the influence (eq:erf-worked)."""
    x = np.arange(len(infl)) - len(infl) // 2
    p = infl / infl.sum()
    var = float((p * x ** 2).sum() - (p * x).sum() ** 2)
    return 4.0 * np.sqrt(var)          # +/- 2 sigma


print(f"{'stack':<28}{'theory':>9}{'stride':>8}{'effective':>11}"
      f"{'eff/theory':>12}{'area ratio':>12}")
print("-" * 80)
for name, stack in STACKS.items():
    rf, jump = theoretical_rf(stack)
    infl = influence_profile(stack)
    erf = effective_rf(infl)
    print(f"{name:<28}{rf:>9}{jump:>8}{erf:>11.1f}{erf / rf:>12.2f}"
          f"{(erf / rf) ** 2:>12.2f}")

r10 = effective_rf(influence_profile(STACKS["10x conv3, stride 1"]))
r20 = effective_rf(influence_profile(STACKS["20x conv3, stride 1"]))
print(f"""
The theory column is eq:receptive-field and it is correct: those pixels CAN
influence the output. The effective column is the window that carries most of the
influence, and the two disagree -- but not uniformly, and the pattern is the
result.

Compare the two plain stacks. Doubling the depth from 10 to 20 layers doubles the
theoretical field, 21 to 41, exactly as the formula says. The effective field
grows from {r10:.1f} to {r20:.1f}, a factor of {r20 / r10:.2f} -- and the square
root of 2 is {2 ** 0.5:.2f}. Theoretical field grows LINEARLY in depth and
effective field grows like its SQUARE ROOT, so the ratio decays as 1/sqrt(depth):
0.49 at ten layers, 0.36 at twenty.

The mechanism is the central limit theorem. Influence reaching a distant input
pixel has to survive one particular path through every layer, and there are
vastly more paths to the centre than to the edge, so the influence profile of a
deep stack approaches a Gaussian regardless of what the individual layers look
like. Nothing about the weights is assumed here -- every layer in this simulation
spreads influence uniformly -- so the concentration is a property of STACKING, not
of training.

Now read the bottom two rows, which is where the framing "effective field is
about half" breaks. Both reach 0.96: with only five layers there has not been
enough compounding to concentrate anything. So the shrinkage is not a fact about
large receptive fields, it is a fact about DEEP ones, and a shallow stack with
aggressive stride or dilation genuinely sees what the formula says it sees.

The engineering consequence is the reason this listing exists. The standard rule
of thumb -- make the receptive field at least as large as the object you want to
detect -- gets applied to the theory column and belongs on the effective one. For
a deep backbone that is a factor of two or three in width, and a factor of four
to nine in AREA, so an object comfortably inside the theoretical field can be
classified by features that never took in its edges.

The last two rows also price the two ways of buying field cheaply. Striding grows
the field fast and grows the JUMP with it, so the output grid gets coarse -- the
resolution-versus-context tension ch:mm-segmentation spends a whole architecture
resolving. Dilation reaches the same 63 pixels at stride 1, leaving the output
dense. Same field, same depth, and one of them can still tell you where things
are.""")
```

The second listing asks what the convolutional prior is actually buying, by
training a model without it.

```python {tier=A name=why-not-fully-connected}
"""Why a fully connected layer is the wrong prior for an image.

The usual reason given is parameter count, and it is true and it is the less
interesting half. The real objection is that a dense layer has no notion that
translating an image leaves its content unchanged: pixel 47 and pixel 48 are
unrelated coordinates to it, so a shape it learned in one place teaches it
nothing about the same shape three pixels over (eq:translation-equivariance).

This listing trains both models from scratch on the same task and separates the
two objections. Both see the same data, both are trained the same way, and the
test set contains translations the training set did not.
"""
import numpy as np

rng = np.random.default_rng(3)

H = W = 16
K = 5                        # shape template size
N_CLASS = 3
N_TRAIN, N_TEST = 3000, 1500
HID = 24
EPOCHS, LR = 24, 0.08

# Three 5x5 shapes: a cross, a corner, a bar.
TEMPLATES = np.zeros((N_CLASS, K, K))
TEMPLATES[0, 2, :] = 1; TEMPLATES[0, :, 2] = 1                 # cross
TEMPLATES[1, 0, :] = 1; TEMPLATES[1, :, 0] = 1                 # corner
TEMPLATES[2, 2, :] = 1                                         # bar


def make(n, centred):
    """centred=True places every shape in the middle; False places it anywhere."""
    X = np.zeros((n, H, W))
    y = rng.integers(0, N_CLASS, size=n)
    for i in range(n):
        if centred:
            r = c = (H - K) // 2
        else:
            r, c = rng.integers(0, H - K + 1, size=2)
        X[i, r:r + K, c:c + K] = TEMPLATES[y[i]]
    X += 0.08 * rng.normal(size=X.shape)
    return X, y


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return loss, g / len(y)


class Dense:
    """Flatten the image, then two dense layers. Every pixel is its own
    coordinate and nothing ties neighbours together (eq:dense-no-prior)."""

    def __init__(self):
        self.W1 = rng.normal(scale=np.sqrt(2 / (H * W)), size=(H * W, HID))
        self.b1 = np.zeros(HID)
        self.W2 = rng.normal(scale=np.sqrt(2 / HID), size=(HID, N_CLASS))
        self.b2 = np.zeros(N_CLASS)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, X):
        self.f = X.reshape(len(X), -1)
        self.h = np.maximum(self.f @ self.W1 + self.b1, 0)
        return self.h @ self.W2 + self.b2

    def backward(self, g, lr):
        gW2, gb2 = self.h.T @ g, g.sum(axis=0)
        gh = (g @ self.W2.T) * (self.h > 0)
        gW1, gb1 = self.f.T @ gh, gh.sum(axis=0)
        for p, gp in ((self.W1, gW1), (self.b1, gb1), (self.W2, gW2), (self.b2, gb2)):
            p -= lr * gp


class Conv:
    """One bank of shared KxK filters applied at every position, then a global
    max over positions. Weight sharing IS the translation prior: the same filter
    is asked the same question everywhere, and the max discards where."""

    def __init__(self):
        self.F = rng.normal(scale=np.sqrt(2 / (K * K)), size=(K * K, HID))
        self.bf = np.zeros(HID)
        self.W2 = rng.normal(scale=np.sqrt(2 / HID), size=(HID, N_CLASS))
        self.b2 = np.zeros(N_CLASS)

    def n_params(self):
        return self.F.size + self.bf.size + self.W2.size + self.b2.size

    @staticmethod
    def patches(X):
        n, P = len(X), H - K + 1
        out = np.empty((n, P * P, K * K))
        for i in range(P):
            for j in range(P):
                out[:, i * P + j] = X[:, i:i + K, j:j + K].reshape(n, -1)
        return out

    def forward(self, X):
        self.p = self.patches(X)                       # (n, pos, K*K)
        self.a = np.maximum(self.p @ self.F + self.bf, 0)   # (n, pos, HID)
        self.arg = self.a.argmax(axis=1)               # global max pool
        self.h = np.take_along_axis(self.a, self.arg[:, None, :], axis=1)[:, 0]
        return self.h @ self.W2 + self.b2

    def backward(self, g, lr):
        gW2, gb2 = self.h.T @ g, g.sum(axis=0)
        gh = (g @ self.W2.T) * (self.h > 0)
        # Gradient flows only through the winning position for each channel.
        gF = np.zeros_like(self.F)
        gbf = gh.sum(axis=0)
        n = len(gh)
        for c in range(HID):
            win = self.p[np.arange(n), self.arg[:, c]]          # (n, K*K)
            gF[:, c] = win.T @ gh[:, c]
        for p, gp in ((self.F, gF), (self.bf, gbf), (self.W2, gW2), (self.b2, gb2)):
            p -= lr * gp


def train_eval(model, Xtr, ytr, Xte, yte):
    n = len(Xtr)
    for ep in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, 64):
            b = order[s:s + 64]
            logits = model.forward(Xtr[b])
            _, g = softmax_ce(logits, ytr[b])
            model.backward(g, LR)
    return float((model.forward(Xte).argmax(axis=1) == yte).mean())


Xc, yc = make(N_TRAIN, centred=True)         # training: shapes always centred
Xa, ya = make(N_TRAIN, centred=False)        # training: shapes anywhere
Xt, yt = make(N_TEST, centred=False)         # test: always anywhere
Xtc, ytc = make(N_TEST, centred=True)

print(f"{H}x{W} images, {N_CLASS} shapes, {HID} hidden units in both models\n")
print(f"{'model':<10}{'params':>9}{'train centred ->':>19}{'':>3}"
      f"{'train anywhere ->':>19}")
print(f"{'':<10}{'':>9}{'test centred':>12}{'test anywhere':>15}"
      f"{'test anywhere':>18}")
print("-" * 66)

for name, cls in (("dense", Dense), ("conv", Conv)):
    m1 = cls(); cen = train_eval(m1, Xc, yc, Xtc, ytc)
    shift = float((m1.forward(Xt).argmax(axis=1) == yt).mean())
    m3 = cls(); anyw = train_eval(m3, Xa, ya, Xt, yt)
    print(f"{name:<10}{cls().n_params():>9,}{cen:>12.3f}{shift:>15.3f}"
          f"{anyw:>18.3f}")

print(f"""
Read the middle two columns together, because the pair is the argument. Trained
on centred shapes and tested on centred shapes, both models solve the task -- the
dense network is perfectly capable of learning three shapes at a fixed location.
Tested on the SAME shapes moved elsewhere in the frame, the dense model collapses
to 0.346 against a chance rate of 0.333, while the convolutional one holds 0.970.

That gap is not a capacity problem and it is not an optimisation problem. The
dense model learned the task it was shown, completely. It simply has no way to
know that the task it was shown and the task it is being tested on are the same
task, because to a dense layer pixel 47 and pixel 48 are unrelated coordinates.
Every translated copy of a shape is, to that architecture, a new shape.

The convolutional model gets that for free from weight sharing
(eq:translation-equivariance). The same filter is applied at every position, so
evidence about a shape gathered anywhere updates the same weights, and the global
max discards WHERE the evidence was found. Note that the prior is doing two jobs
here: sharing makes learning at one position transfer to all of them, and pooling
makes the answer invariant to which position won.

The last column is the control that makes the argument honest. Given training
data that already covers every position, the dense model recovers to 0.893 -- so
the prior is not supplying capability the dense model lacks, it is supplying
SAMPLE EFFICIENCY the dense model would otherwise have to buy with data. That is
what an architectural prior is: a statement about the world, paid for once in
design instead of repeatedly in examples.

And only now is the parameter count worth mentioning. The dense model uses about
{Dense().n_params() / Conv().n_params():.0f} times the parameters to be worse,
and the ratio grows with image area -- eq:dense-parameter-count scales with the
number of pixels and eq:conv-parameter-count does not scale with image size at
all. The parameter argument is the one usually given for convolutions. It is the
weaker of the two.""")
```

## 9. Practical Example

**The receptive field is not what the formula says.** Ten stacked $3\times3$
convolutions have a theoretical field of 21 pixels and an effective field of
**10.3**. Twenty layers: theoretical 41, effective **14.6**.

Read those as a pair. **Depth doubled, theoretical field doubled, effective field
grew by 1.42** — and $\sqrt{2} = 1.41$. That is {{eq:erf-depth-scaling}} confirmed
by its exponent rather than by a single ratio, and the closed form
{{eq:erf-worked}} predicts 10.3 and 14.6 exactly.

The mechanism is {{eq:variance-adds}}: influence reaching a far pixel must survive
one specific path through every layer, and there are vastly more paths to the
centre. **The simulation gives every layer uniform weights**, so the concentration
is a property of stacking, not of training — and a trained kernel that
concentrates weight centrally makes it *worse*.

> **IMPORTANT:** The two shallow rows are where the convenient framing breaks.
> Five layers with stride, and five with dilation, both reach a ratio of **0.96**
> — they see essentially all of their theoretical field. So the shrinkage is not a
> fact about large receptive fields; it is a fact about **deep** ones. Quoting
> "effective field is about half" is as wrong as quoting the formula. Use
> {{eq:erf-worked}}.

The engineering consequence: a backbone with a 200-pixel theoretical field is
reliably seeing something nearer 70, so an object 150 pixels across is being
classified by features that never took in its edges — a factor of two or three in
width and **four to nine in area**.

The bottom two rows also price the two ways of buying field. Striding reaches 63
pixels and drags the jump to 32; dilation reaches the same 63 at jump 1. **Same
field, same depth, and only one of them can still tell you where things are** —
which is why dilated convolutions live in segmentation and rarely in
classification.

**What the convolutional prior actually buys.** Both models learn centred shapes
perfectly. Moved to positions never seen in training, the dense model scores
**0.346 against a chance rate of 0.333** — it has learned nothing transferable —
while the convolutional model holds **0.970**.

That is not capacity and it is not optimisation. **The dense model learned the
task it was shown, completely.** It has no way to know that the shifted task is
the same task, because {{eq:dense-no-prior}} leaves nothing connecting them.

**And the last column is what makes the argument honest.** Trained on data that
already covers every position, the dense model recovers to **0.893**. So the prior
is not supplying capability — it is supplying *sample efficiency*. An
architectural prior is a statement about the world paid for once in design instead
of repeatedly in examples, and this is what that sentence means quantitatively.

Only then is the parameter count worth mentioning: **9× fewer parameters** in the
model that does better, with the ratio growing as image area grows. The parameter
argument is the one usually given for convolutions, and it is the weaker of the
two.

## 10. Production Considerations

**Compute the effective receptive field, not the theoretical one**
({{eq:erf-worked}}), and check it against the smallest object you care about. It
is four lines and it explains most "why does it miss small things" reports.

**Know your jump.** {{eq:receptive-field}}'s $\jmath$ is the finest localisation
available at that layer. If you need finer, you need features from an earlier
stage ({{ch:mm-detection}}) or a decoder ({{ch:mm-segmentation}}).

**Do not resize away the signal.** Downsampling to $224\times224$ is the single
most common silent failure in a vision pipeline: 8-point text and distant objects
are gone before the model runs. Log input resolution and the resize factor.

**Use `same` padding by default.** Losing pixels from the edges loses them
precisely where missed detections concentrate.

**Match preprocessing to the pretrained backbone exactly** — normalisation
constants, channel order, interpolation. Mismatches degrade quietly rather than
failing.

**Prefer dilation over stride when you need field and resolution**
({{eq:resolution-tension}}).

**Beware strided equivariance.** {{eq:translation-equivariance}} holds only for
shifts that are multiples of the stride, so a one-pixel crop can change a
prediction. If that matters, test it explicitly.

## 11. Common Mistakes

**Planning with the theoretical receptive field.** The chapter's headline.

**Confusing equivariance with invariance**, then wondering why a segmentation head
after global pooling cannot localise.

**Assuming local max-pooling gives translation invariance.** It gives tolerance;
only {{eq:pooling-invariance}}'s global pool gives invariance.

**Citing parameter count as the reason for convolutions.** It is the weaker
argument, and {{sec:9-practical-example}} shows the stronger one.

**Resizing to the backbone's training resolution without checking what is lost.**

**Forgetting that stride multiplies into the jump for every subsequent layer.**

**Treating the border as unimportant.**

## 12. Failure Modes

**Small-object blindness.** Symptom: recall falls off a cliff below some object
size. Cause: {{eq:erf-worked}} at the decision layer, or the input resize.
Diagnose by plotting recall against object pixel area.

**Localisation quantised to the jump.** Symptom: predicted boxes or masks snap to
a grid. Cause: predicting from too deep a stage.

**One-pixel-shift instability.** Symptom: predictions change under a trivial crop.
Cause: strided aliasing breaking {{eq:translation-equivariance}}.

**Edge misses.** Symptom: objects near the frame border are missed
disproportionately. Cause: padding, or the effective field being truncated there.

**Preprocessing mismatch.** Symptom: a pretrained backbone underperforms its
published numbers by a few points, consistently. Cause: normalisation or channel
order, and nothing errors.

**Resolution regression.** Symptom: accuracy drops after an "optimisation" that
reduced input size. The cost model in {{eq:feature-map-cost}} made it attractive
and nobody measured what was lost.

## 13. Alternatives

| Alternative to convolution | What it gives up | When it wins |
|---|---|---|
| dense/MLP on pixels | translation prior, scaling | never for raw images; fine after a backbone |
| MLP-Mixer style | locality prior | large data, where priors matter less |
| self-attention ({{ch:mm-vit}}) | locality and equivariance | large data; global context from layer one |
| dilated convolution | nothing — same params | when you need field at full resolution |
| depthwise separable | some capacity per FLOP | mobile and latency-bound settings |

**The third row is the whole of {{ch:mm-vit}}.** A transformer discards the
convolutional prior and must learn from data what convolution assumes — which is
a good trade at large scale and a bad one at small, and the crossover is
measurable rather than a matter of taste.

## 14. Evaluation

**Report accuracy against object size**, not just aggregate accuracy. The
aggregate hides {{eq:erf-worked}}'s consequence entirely.

**Test translation robustness explicitly**: shift the evaluation set by a few
pixels and re-score. A large drop indicates strided aliasing.

**Report input resolution with every number.** A vision result without its
resolution is not reproducible.

**Measure the effective receptive field** of a candidate backbone before adopting
it, and compare it against your object-size distribution.

**Separate localisation error from classification error.** They come from
different depths and have different fixes.

## 15. Advanced Concepts

**Equivariance to more than translation.** {{maturity:EMERGING}} Group-equivariant
convolutions extend {{eq:translation-equivariance}} to rotations and reflections,
which matters where orientation is arbitrary — microscopy, astronomy, aerial
imagery — and matters little for photographs of a world with gravity in it.

**Anti-aliased downsampling.** {{maturity:MATURE}} Strided operations violate the
sampling theorem, which is the mechanism behind one-pixel-shift instability.
Blurring before subsampling restores much of the equivariance for a small cost,
and is under-used.

**The convolution/attention continuum.** {{maturity:MATURE}} A convolution is
attention with a fixed, local, content-independent pattern. Seeing them as ends of
one axis makes {{ch:mm-vit}}'s trade a question about how much of the pattern to
learn, and {{cite:liu2022convnext}} showed much of the apparent gap between them
was training recipe rather than architecture.

**The effective receptive field as a design instrument.**
{{maturity:ESTABLISHED}} {{eq:erf-worked}} inverts to
{{eq:depth-for-object}}: given an object size, it says what the stack must do.
Almost nobody uses it that way, and it is four lines of arithmetic.

**Resolution is the budget, everywhere.** {{maturity:ESTABLISHED}}
{{eq:resolution-tension}} recurs in every chapter of this part with different
units — feature-map pixels here, patch count in {{ch:mm-vit}}, visual tokens in
{{ch:mm-vlms}}, frames in {{ch:mm-video-audio}}. Same trade, four currencies.

## 16. Connection to Previous Chapters

{{ch:dl-cnns}} introduced convolution as an operation; this chapter recasts
it as a *prior* and measures what the prior buys.
{{ch:dl-backprop}}'s chain rule is what {{eq:effective-receptive-field}}'s
influence propagation is tracing. {{ch:math-derivatives}}'s composition argument
is the same one. Forward: {{ch:mm-classification}} is what happens when you stack
this deeply enough to hit the degradation problem;
{{ch:mm-detection}} and {{ch:mm-segmentation}} are two ways of recovering the
*where* that {{eq:pooling-invariance}} discards; {{ch:mm-vit}} discards the prior
entirely; and {{ch:mm-vlms}}'s visual token budget is
{{eq:resolution-tension}} with a price attached.

## 17. Exercises

1. Derive {{eq:translation-equivariance}} from {{eq:convolution}} and state
   exactly where the border breaks it.
2. Compute the receptive field and jump of a ResNet-50 stem plus stage 1 using
   {{eq:receptive-field}}. Then compute the effective field with
   {{eq:erf-worked}}.
3. In `receptive-field-arithmetic`, add a 40-layer stack. Does the ratio follow
   $1/\sqrt{n}$?
4. Modify the same listing so each layer spreads influence with a centre-weighted
   rather than uniform kernel. Which direction does the effective field move, and
   why does that make {{eq:erf-worked}} a bound?
5. In `why-not-fully-connected`, remove the global max pool and use the mean
   instead. What happens to the shifted-test column, and what does that tell you
   about {{eq:pooling-invariance}}?
6. Add a second convolutional layer to the same listing. Does the shifted-test
   accuracy improve, and is that a fair comparison at equal parameters?
7. Use {{eq:depth-for-object}} to compute the depth needed to see a 64-pixel
   object with $3\times3$ stride-1 convolutions. Then find a stack with stride
   that achieves it in under 20 layers.
8. Take a pretrained backbone you use. Measure its effective receptive field at
   the layer you predict from, and compare against your smallest labelled object.

## 18. Interview Questions

1. Why not use a fully connected network on images? Give the strong reason, not
   the parameter-count one.
2. Define equivariance and invariance and say which operation gives which.
3. Compute the receptive field of five stacked $3\times3$ convolutions with a
   stride-2 pool after each pair.
4. Why is the effective receptive field smaller than the theoretical one, and by
   how much?
5. What is the jump, and what does it limit?
6. When would you use dilation instead of stride?
7. Your detector misses small objects. Give three candidate causes and the
   measurement that separates them.
8. Why do channels double when spatial dimensions halve?
9. What does weight sharing buy that fewer parameters does not?
10. Your model's prediction changes when you crop one pixel. Explain.

## 19. Research Questions

1. {{eq:erf-worked}} assumes uniform spreading. What is the effective field of a
   *trained* network, and how much does it vary across channels and across
   training runs?
2. Anti-aliased downsampling restores equivariance at a cost. Is there a
   principled way to choose the filter given a target robustness?
3. {{eq:depth-for-object}} says stride is necessary. Is there an architecture that
   grows effective field linearly in depth rather than as its square root?
4. Group equivariance helps where orientation is arbitrary. Can the right group be
   *learned* from data rather than assumed?
5. {{cite:liu2022convnext}} attributed much of the transformer advantage to
   training recipe. What is the residual architectural difference once recipe is
   fully controlled, and does it depend on data scale?

## 20. Chapter Summary

An image is a tensor with two properties a dense layer cannot use: **neighbouring
pixels are related, and content is translation-invariant.** Flattening destroys
both, and {{eq:dense-no-prior}} means every shifted copy of a pattern is, to a
dense layer, a new pattern.

**Convolution is a claim about the world** — locality plus stationarity — and
weight sharing is the load-bearing half. Measured: trained on centred shapes and
tested on shifted ones, a dense model scores **0.346 against a chance rate of
0.333** while a convolutional model holds **0.970**. Given training data covering
every position the dense model recovers to **0.893**, which shows what the prior
actually supplies: **sample efficiency, not capability.** The 9× parameter saving
is real and is the weaker argument.

**Equivariance and invariance are different.** Convolution preserves *where*
({{eq:translation-equivariance}}); global pooling discards it
({{eq:pooling-invariance}}). Classification wants the second; detection and
segmentation want the first, which is why the rest of this part is largely about
recovering position after it was thrown away.

**And a deep network sees far less than its specification claims.**
{{eq:receptive-field}} counts what *can* influence an output; weight by influence
actually delivered and the field grows as $\sqrt{\text{depth}}$ while the formula
grows linearly ({{eq:erf-depth-scaling}}). Measured: 21 → 10.3 pixels at ten
layers, 41 → 14.6 at twenty, a ratio of **1.42** against $\sqrt{2} = 1.41$, and
matching the closed form {{eq:erf-worked}} to three significant figures. Shallow
stacks with stride or dilation show no such gap, so **the shrinkage is a property
of depth, not of field size**.

That makes {{eq:erf-worked}} a design instrument: it inverts to
{{eq:depth-for-object}}, which says what a stack must do to see an object of a
given size — and explains why real architectures buy field with stride and
dilation, paying {{eq:resolution-tension}}'s price.

**Resolution is the budget.** Every architecture in this part spends it
differently, and every one of them is answering the same question: what is the
unit of visual information, and how many can you afford?

## 21. Further Reading

{{cite:lecun1998}} for the original argument, which is still the clearest
statement of why weight sharing rather than parameter count is the point.
{{cite:krizhevsky2012}} for the moment the argument became decisive, and
{{cite:simonyan2015vgg}} for the controlled depth sweep that made $3\times3$
stacks the default.
{{cite:russakovsky2015ilsvrc}} for what the benchmark actually measured — worth
reading before assuming "ImageNet accuracy" means general visual competence.
{{cite:ronneberger2015unet}} for the cleanest architectural response to
{{eq:resolution-tension}}, developed properly in {{ch:mm-segmentation}}.
{{cite:liu2022convnext}} for the ablation that separates architecture from
training recipe, and as the counterweight to the idea that convolution has been
superseded.
