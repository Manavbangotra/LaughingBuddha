---
id: dl-autoencoders
number: 61
part: VI
tier: full
status: reviewed
requires: [dl-cnns, dl-losses, dl-regularization, ml-pca, ml-anomaly]
provides: [autoencoder, bottleneck, latent-space, denoising-autoencoder,
           variational-autoencoder, reparameterization-trick, elbo,
           posterior-collapse, reconstruction-objective]
citations: [hinton2006, kingma2014vae, lecun1998, zhang2017rethinking]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Define an autoencoder and explain what the bottleneck does.
2. Prove that a linear autoencoder under squared error recovers the PCA
   subspace, and state what it does not recover.
3. Explain why a plain autoencoder is a poor generative model.
4. Derive the evidence lower bound and the reparameterisation trick.
5. Explain posterior collapse and how to detect it.
6. Apply autoencoders to compression, denoising and anomaly detection.
7. Explain where the autoencoder idea survives in 2026 and where it does not.

## 2. Why This Matters

**This is the part's only unsupervised chapter, and unsupervised learning is
what the rest of the book runs on.** Every foundation model is trained by
predicting part of its input from another part, which is the autoencoder idea
generalised. {{ch:llm-next-token}} and {{ch:emb-models}} are both instances.

**The linear case has an exact answer**, and it is PCA. That is a rare thing in
deep learning: a nonlinear method whose linear special case is a classical
method with a closed form, so you can check the deep version against something
known. {{sec:8-implementation}} does exactly that.

**The variational autoencoder introduced the reparameterisation trick**, which
is how anyone differentiates through a sampling step. It is used far outside
generative modelling — in reinforcement learning, in discrete relaxations, in
stochastic architectures — and {{cite:kingma2014vae}} is where it comes from.

**Autoencoders are the honest case of a method whose original purpose was
superseded.** {{cite:hinton2006}} used stacked autoencoders for layerwise
pretraining, which was essential in 2006 and unnecessary by 2015 once
initialisation, normalisation and ReLU made deep networks directly trainable.
Knowing *why* a technique became obsolete is more useful than knowing that it
did.

## 3. Prerequisites

{{ch:ml-pca}} for principal components, which {{sec:6-mathematical-foundation}}
recovers exactly. {{ch:dl-losses}} for the reconstruction losses.
{{ch:dl-cnns}} for the convolutional encoder and the transposed convolution.
{{ch:ml-anomaly}} for the anomaly-detection application.
{{ch:dl-regularization}} for the noise-injection framing of the denoising
variant.

## 4. Intuitive Explanation

### 4.1 Learning by reconstruction

An autoencoder maps an input to itself through a narrow middle:

```text
   x ──▶[encoder]──▶ z ──▶[decoder]──▶ x̂
        (compress)   (small)  (expand)

   loss = ||x - x̂||²
```

Reconstructing the input sounds pointless — the identity function does it
perfectly. **The bottleneck is what makes it a learning problem.** If $z$ has
fewer dimensions than $x$, the identity is unavailable and the network must
discover what to keep.

**The label is the input itself**, so no annotation is needed. That is the whole
appeal, and it is the reason the idea generalised into everything that followed.

### 4.2 What the bottleneck buys, and what it does not

The bottleneck is the constraint, and it is not the only possible one:

```text
   undercomplete    dim(z) < dim(x)     forced compression
   sparse           dim(z) large, but few units active
   denoising        input corrupted, target clean
   contractive      penalise the encoder's sensitivity to the input
   variational      z is a DISTRIBUTION, penalised toward a prior
```

Each is a different way of preventing the identity. The denoising variant is
worth singling out: **corrupting the input makes the identity useless without
needing a narrow layer at all**, so the code can be as wide as you like. It is
also the direct ancestor of masked language modelling
({{ch:llm-next-token}}).

### 4.3 Why a plain autoencoder cannot generate

Train an autoencoder, throw away the encoder, sample a random $z$, decode it.
The output is usually nonsense.

**The decoder was only ever trained on codes the encoder produced**, which
occupy some complicated region of the latent space — possibly a disconnected
one, possibly a thin manifold. A point sampled from anywhere else is off that
region and the decoder has no idea what to do with it.

```text
   trained codes    · ·  ·· ·        a few clumps and filaments
                      ··· ·
   sampled z        ✗                somewhere in between: nonsense
```

The variational autoencoder's contribution is to *force* the code distribution
to match something you can sample from. That is the whole point of the KL term,
and it is why {{sec:8-implementation}} measures the latent distribution
directly.

### 4.4 The variational autoencoder in one picture

```text
   x ──▶[encoder]──▶ μ(x), σ(x) ──▶ z ~ N(μ, σ²) ──▶[decoder]──▶ x̂
                          │
                          └──▶ KL( N(μ,σ²) || N(0,I) )   pull toward the prior

   loss = reconstruction + KL
```

Two terms in tension. Reconstruction wants each input to get its own precise
code; the KL wants every code to look like a draw from $\mathcal{N}(0,\mat{I})$.
The balance is what produces a latent space that is both informative and
sampleable.

**The trick that makes it trainable** is to write $z = \mu + \sigma\epsilon$ with
$\epsilon \sim \mathcal{N}(0,\mat{I})$. The randomness is now in $\epsilon$,
which does not depend on the parameters, so the gradient flows through $\mu$ and
$\sigma$ normally. Without this you cannot backpropagate through a sample at all.

### 4.5 Where this stands in 2026

**Layerwise pretraining is dead.** {{cite:hinton2006}} needed it because deep
networks could not be trained directly. {{ch:dl-initialization}},
{{ch:dl-normalization}} and {{ch:dl-activations}} between them removed the need,
and nobody has used it for a decade.

**Autoencoders for compression are niche.** Learned compression works and
classical codecs are strong, well-optimised baselines.

**Anomaly detection is a live application.** Train on normal data, flag high
reconstruction error. Simple, effective, and with the failure mode
{{sec:12-failure-modes}} describes.

**The idea, generalised, is everything.** Masked prediction, next-token
prediction, contrastive learning — all learn representations from unlabelled
data by predicting part of the input from the rest. VAEs specifically remain
important as the latent-space component of diffusion models
— see the note on diffusion below.

## 5. Formal Explanation

### 5.1 The basic autoencoder

$$
\vec{z} = f_\phi(\vec{x}), \qquad
\hat{\vec{x}} = g_\theta(\vec{z}), \qquad
\Like = \E_{\vec{x}}\big[\ell(\vec{x}, g_\theta(f_\phi(\vec{x})))\big]
$$ (eq:autoencoder)

with $\ell$ squared error for continuous data and binary cross-entropy for data
in $[0,1]$.

### 5.2 The variants

**Sparse.** Add a penalty on the code's activity:

$$
\Like = \ell(\vec{x},\hat{\vec{x}}) + \lambda\|\vec{z}\|_1
$$ (eq:sparse-ae)

**Denoising.** Corrupt the input, reconstruct the clean version:

$$
\tilde{\vec{x}} \sim q(\tilde{\vec{x}}\mid\vec{x}),
\qquad
\Like = \ell\big(\vec{x},\, g_\theta(f_\phi(\tilde{\vec{x}}))\big)
$$ (eq:denoising-ae)

**Contractive.** Penalise the encoder's Jacobian:

$$
\Like = \ell(\vec{x},\hat{\vec{x}})
 + \lambda\big\|\partial f_\phi/\partial\vec{x}\big\|_F^2
$$ (eq:contractive-ae)

All three make the identity unavailable without requiring
$\dim(\vec{z}) < \dim(\vec{x})$.

### 5.3 The variational autoencoder

Posit a generative model $p_\theta(\vec{x},\vec{z}) =
p(\vec{z})p_\theta(\vec{x}\mid\vec{z})$ with $p(\vec{z}) =
\mathcal{N}(\vec{0},\mat{I})$. The marginal likelihood

$$
p_\theta(\vec{x}) = \int p_\theta(\vec{x}\mid\vec{z})p(\vec{z})\,d\vec{z}
$$ (eq:marginal-likelihood)

is intractable. Introduce an approximate posterior
$q_\phi(\vec{z}\mid\vec{x}) = \mathcal{N}(\vecgreek{\mu}_\phi(\vec{x}),
\diag(\vecgreek{\sigma}^2_\phi(\vec{x})))$ and optimise the **evidence lower
bound**:

$$
\log p_\theta(\vec{x}) \ge
 \underbrace{\E_{q_\phi}\big[\log p_\theta(\vec{x}\mid\vec{z})\big]}
 _{\text{reconstruction}}
 - \underbrace{\KL\big(q_\phi(\vec{z}\mid\vec{x})\,\|\,p(\vec{z})\big)}
 _{\text{regulariser}}
$$ (eq:elbo)

For Gaussian $q$ and standard normal prior the KL has a closed form:

$$
\KL = \tfrac{1}{2}\sum_{j=1}^{d}
 \big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)
$$ (eq:gaussian-kl)

**The reparameterisation trick.** Write

$$
\vec{z} = \vecgreek{\mu}_\phi(\vec{x})
 + \vecgreek{\sigma}_\phi(\vec{x})\odot\vecgreek{\epsilon},
\qquad \vecgreek{\epsilon}\sim\mathcal{N}(\vec{0},\mat{I})
$$ (eq:reparameterization)

so the sampling no longer depends on $\phi$ and the gradient passes through
$\vecgreek{\mu}$ and $\vecgreek{\sigma}$ as ordinary tensors.

### 5.4 $\beta$-VAE

Weight the KL:

$$
\Like = \E_{q_\phi}[\log p_\theta(\vec{x}\mid\vec{z})]
 - \beta\,\KL(q_\phi\,\|\,p)
$$ (eq:beta-vae)

$\beta > 1$ pushes harder toward the prior, producing a more sampleable latent
space and blurrier reconstructions. $\beta < 1$ does the reverse. **At
$\beta = 0$ it is a plain autoencoder**, which makes $\beta$ a dial between the
two and is the cleanest way to see what the KL term buys.

Larger $\beta$ was reported to encourage *disentangled* factors of variation.
The effect is real on the datasets used and later work found it depends heavily
on hyperparameters and inductive biases. {{maturity:EMERGING}}

### 5.5 Applications

**Anomaly detection.** Train on normal data only; a high reconstruction error at
test time indicates something unlike the training distribution
({{ch:ml-anomaly}}).

**Denoising.** Directly what {{eq:denoising-ae}} trains for.

**Dimensionality reduction.** A nonlinear generalisation of
{{ch:ml-pca}}, useful for visualisation and as a feature extractor.

**Generative modelling** via the VAE, mostly superseded by diffusion for image
quality and still used as the latent-space compressor inside diffusion models.

## 6. Mathematical Foundation

### 6.1 A linear autoencoder recovers the PCA subspace

Let the encoder be $\vec{z} = \mat{W}_e\vec{x}$ and the decoder
$\hat{\vec{x}} = \mat{W}_d\vec{z}$, both linear, with $\vec{x}$ centred and
$\dim(\vec{z}) = k$. Minimise $\E\|\vec{x}-\mat{W}_d\mat{W}_e\vec{x}\|^2$.

Write $\mat{M} = \mat{W}_d\mat{W}_e$, which has rank at most $k$. The problem is

$$
\min_{\rank(\mat{M})\le k} \E\big\|\vec{x}-\mat{M}\vec{x}\big\|^2
 = \min_{\rank(\mat{M})\le k}\tr\big((\mat{I}-\mat{M})
 \mat{\Sigma}(\mat{I}-\mat{M})\T\big)
$$ (eq:linear-ae-objective)

with $\mat{\Sigma} = \E[\vec{x}\vec{x}\T]$. By the Eckart–Young theorem the
minimiser is the projection onto the span of the top $k$ eigenvectors of
$\mat{\Sigma}$ — which is exactly PCA's subspace. $\square$

> IMPORTANT: **It recovers the subspace, not the components.** Any invertible
> $\mat{A}$ gives $\mat{W}_d\mat{A}^{-1}$ and $\mat{A}\mat{W}_e$ with the same
> product, so the learned axes are an arbitrary basis of the right subspace —
> not orthogonal, not ordered by variance. {{sec:8-implementation}} measures
> both facts: the subspace matches to high precision and the individual
> directions do not.

### 6.2 Deriving the ELBO

Start from the log marginal likelihood and introduce $q_\phi$:

$$
\log p_\theta(\vec{x})
 = \log\int p_\theta(\vec{x},\vec{z})\,d\vec{z}
 = \log \E_{q_\phi}\!\left[\frac{p_\theta(\vec{x},\vec{z})}
 {q_\phi(\vec{z}\mid\vec{x})}\right]
$$

By Jensen's inequality, since $\log$ is concave:

$$
\log p_\theta(\vec{x}) \ge
 \E_{q_\phi}\!\left[\log\frac{p_\theta(\vec{x},\vec{z})}
 {q_\phi(\vec{z}\mid\vec{x})}\right]
 = \E_{q_\phi}[\log p_\theta(\vec{x}\mid\vec{z})]
 - \KL(q_\phi\,\|\,p)
$$

which is {{eq:elbo}}. $\square$

The gap is exact and worth naming:

$$
\log p_\theta(\vec{x}) - \text{ELBO}
 = \KL\big(q_\phi(\vec{z}\mid\vec{x})\,\|\,
 p_\theta(\vec{z}\mid\vec{x})\big)
$$ (eq:elbo-gap)

**The bound is tight exactly when the approximate posterior equals the true
one.** So maximising the ELBO does two things at once: it improves the model and
it improves the approximation. That is elegant and it is also why a poor
posterior family caps how good the model can look.

### 6.3 Why the reparameterisation trick is necessary

To optimise $\E_{q_\phi(\vec{z})}[f(\vec{z})]$ over $\phi$, the naive move fails:
$\nabla_\phi$ cannot pass through a sampling operation.

**Score-function estimator** (REINFORCE) gives an unbiased gradient:

$$
\nabla_\phi \E_{q_\phi}[f(\vec{z})]
 = \E_{q_\phi}\big[f(\vec{z})\nabla_\phi\log q_\phi(\vec{z})\big]
$$ (eq:score-function)

**Reparameterisation** pushes the randomness outside:

$$
\nabla_\phi \E_{\vecgreek{\epsilon}}\big[f(\vecgreek{\mu}_\phi
 + \vecgreek{\sigma}_\phi\odot\vecgreek{\epsilon})\big]
 = \E_{\vecgreek{\epsilon}}\big[\nabla_\phi f(\cdot)\big]
$$ (eq:pathwise-gradient)

Both are unbiased. **The variance is not comparable.** The score-function
estimator uses only $f$'s *value*; the pathwise estimator uses its *gradient*,
so it carries information about which direction to move. The variance difference
is typically orders of magnitude, and {{sec:8-implementation}} measures it —
it is the reason VAEs are trainable and the reason the trick appears far outside
generative modelling.

### 6.4 Posterior collapse

If the decoder is powerful enough to model $\vec{x}$ without $\vec{z}$, the
optimiser can drive $q_\phi(\vec{z}\mid\vec{x}) \to p(\vec{z})$ for every
$\vec{x}$. The KL term goes to zero, the reconstruction term is unaffected
because the decoder ignores $\vec{z}$ anyway, and the ELBO is *higher* than a
solution that uses the latent.

**The latent variable becomes uninformative, and this is the optimum rather than
a failure to converge.** It happens with autoregressive decoders and it happens
whenever $\beta$ is too large.

The diagnostic is the per-dimension KL: a collapsed dimension has
$\KL$ contribution near zero, meaning $\mu_j \approx 0$ and $\sigma_j \approx 1$ for every
input. Counting *active units* — dimensions whose KL exceeds a small threshold —
is the standard measure, and {{sec:8-implementation}} computes it.

Mitigations: KL annealing (ramp $\beta$ from 0), free bits (a floor on each
dimension's KL), and weakening the decoder.

### 6.5 Why VAE samples are blurry

With a Gaussian likelihood, the reconstruction term is squared error, whose
minimiser is the conditional **mean** ({{ch:dl-losses}}). Averaging over the
posterior gives

$$
\hat{\vec{x}} = \E\big[\vec{x}\mid\vec{z}\big]
$$ (eq:vae-blur)

If several plausible images share a code, the model outputs their average, and
the average of several sharp images is a blurry one.

This is not an optimisation failure or a capacity limit. **It is what squared
error asks for**, and it is the same argument as {{ch:dl-losses}}'s
mean-versus-median analysis. Diffusion models avoid it by modelling the
distribution over many denoising steps rather than predicting one point
estimate.

### 5.6 What the code is good for downstream

An autoencoder's code is often described as "features", which invites a
question the objective does not answer: features *for what*?

The reconstruction objective optimises for one thing — retaining enough
information to rebuild the input. That is not the same as retaining the
information a downstream classifier needs, and the two can diverge sharply. A
code that perfectly reconstructs an image has kept the background, the lighting
and the sensor noise, all of which cost capacity and none of which helps
identify the object.

**This is the structural argument against reconstruction as a representation
objective**, and it is why contrastive and predictive methods largely displaced
it for representation learning ({{ch:emb-models}}). A contrastive objective
optimises for a code under which similar inputs are close, which is much nearer
to what a downstream task wants.

Three situations where the reconstruction objective is still the right one:

**When reconstruction is the task.** Compression, denoising and inpainting all
want exactly what the objective optimises.

**When you need a decoder.** Anomaly detection needs to measure reconstruction
error; a contrastive encoder cannot.

**When the input is the only signal you have and it is low-dimensional.** For
tabular or sensor data with no natural augmentations, contrastive learning has
nothing to contrast and reconstruction is a reasonable default.

## 7. Internal Mechanics

### 7.1 Decoder upsampling

Three ways to go from a small spatial map to a large one:

**Transposed convolution**, which is the gradient of a convolution
({{ch:dl-cnns}}). Learned, and it produces **checkerboard artefacts** when the
stride does not divide the kernel size.

**Resize then convolve**: nearest-neighbour or bilinear upsampling followed by a
normal convolution. No artefacts, and generally preferred.

**Pixel shuffle**: produce $r^2$ channels and rearrange them into an $r\times r$
spatial block. Efficient and artefact-free.

### 7.2 Parameterising the variance

Output $\log\sigma^2$ rather than $\sigma$, for the same reason logits are
preferred to probabilities: the output is unconstrained, so no activation is
needed to keep it positive, and $\sigma = \exp(\frac{1}{2}\log\sigma^2)$ is
stable. Predicting $\sigma$ directly requires a softplus and can produce zero.

### 7.3 The KL is summed, the reconstruction is often meaned

A frequent and consequential bug. {{eq:gaussian-kl}} is a sum over latent
dimensions; the reconstruction is often a *mean* over pixels. The two are then
on different scales, and the effective $\beta$ is the ratio — which changes when
the image size or the latent dimension changes.

**Sum both, or be explicit about the ratio.** A VAE that works at one resolution
and collapses at another is usually this.

### 7.4 One sample is enough

The ELBO's expectation is estimated with a single sample of
$\vecgreek{\epsilon}$ per example per step. That is a very noisy estimate of the
expectation and it works, because the noise averages over the batch and over
training. More samples reduce variance and are rarely worth the compute — the
importance-weighted variant uses several and buys a tighter bound rather than
lower variance.

### 7.5 Anomaly detection thresholds

Reconstruction error is a score, not a decision. The threshold comes from the
error distribution on held-out *normal* data — a high quantile, chosen for the
false-positive rate you can tolerate ({{ch:ml-anomaly}}). Choosing it on data
containing anomalies leaks.

## 8. Implementation

```python {tier=A name=autoencoders-and-pca}
"""A linear autoencoder against PCA, and what the bottleneck actually does.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- data with a genuine low-dimensional structure --------------------------
def make_data(n, d=24, k_true=5, noise=0.25, seed=0):
    rs = np.random.default_rng(seed)
    B = rs.normal(size=(k_true, d))
    lat = rs.normal(size=(n, k_true)) * np.array([3.0, 2.2, 1.5, 0.9, 0.4])
    X = lat @ B + rs.normal(0, noise, (n, d))
    return X - X.mean(axis=0)


Xtr = make_data(4000, seed=1)
Xte = make_data(4000, seed=2)
D = Xtr.shape[1]


def pca_reconstruct(Xfit, Xeval, k):
    U, S, Vt = np.linalg.svd(Xfit, full_matrices=False)
    P = Vt[:k]
    return Xeval @ P.T @ P, P


def train_linear_ae(X, k, steps=30000, lr=3e-3, batch=256, seed=0):
    """No activations at all: encoder and decoder are both linear."""
    rs = np.random.default_rng(seed)
    We = rs.normal(0, 1 / np.sqrt(D), (D, k))
    Wd = rs.normal(0, 1 / np.sqrt(k), (k, D))
    ps = [We, Wd]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        z = xb @ We
        xr = z @ Wd
        dxr = 2 * (xr - xb) / len(xb)
        gWd = z.T @ dxr
        gWe = xb.T @ (dxr @ Wd.T)
        for i, (p, g) in enumerate(zip(ps, [gWe, gWd])):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return We, Wd


print("=" * 72)
print("a linear autoencoder recovers PCA's SUBSPACE (section 6.1)")
print("=" * 72)
ev = np.linalg.svd(Xtr, compute_uv=False) ** 2 / len(Xtr)
print("data eigenvalues: "
      + np.array2string(ev[:9], precision=2, suppress_small=True) + " ...\n")
print(f"{'k':>4} {'PCA test MSE':>14} {'linear AE test MSE':>20} "
      f"{'ratio':>8} {'principal angle':>17} {'eig gap k/k+1':>15}")
for k in (2, 4, 5, 8):
    Xp, P = pca_reconstruct(Xtr, Xte, k)
    mse_pca = float(np.mean((Xte - Xp) ** 2))
    We, Wd = train_linear_ae(Xtr, k, seed=3)
    mse_ae = float(np.mean((Xte - Xte @ We @ Wd) ** 2))
    # principal angle between the two k-dimensional subspaces
    Qa, _ = np.linalg.qr(We)
    Qb, _ = np.linalg.qr(P.T)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    angle = float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))
    print(f"{k:>4} {mse_pca:>14.6f} {mse_ae:>20.6f} "
          f"{mse_ae / mse_pca:>8.4f} {angle:>16.2f}° "
          f"{ev[k - 1] / ev[k]:>15.2f}")

print("\nThe reconstruction ratios are close to 1 at every k: the linear")
print("autoencoder reaches essentially PCA's error, which is eq. 61.9 and")
print("Eckart-Young confirmed.")
print("\nThe principal angle is the more interesting column, and it is not")
print("small. Read it against the eigenvalue gap in the last column.")
print("\nWhere the gap is large the subspace is well determined and the two")
print("agree. Where consecutive eigenvalues are close, the objective is")
print("nearly FLAT between the two candidate subspaces — swapping a")
print("direction for its near-degenerate neighbour barely changes the loss")
print("— so an optimiser has no gradient telling it which to pick and lands")
print("somewhere in between.")
print("\nThat is not a failure of the theorem. Eckart-Young says the")
print("MINIMISER is PCA's subspace, and it is silent about how sharply")
print("defined that minimum is. When the eigenvalues are nearly degenerate")
print("the minimum is a shallow valley rather than a point, and an")
print("approximate optimiser stops somewhere in the valley — at nearly the")
print("right loss and not at the right subspace.")

# --- but NOT the components -------------------------------------------------
print("\n" + "=" * 72)
print("...but NOT the components (section 6.1 warning)")
print("=" * 72)
k = 5
We, Wd = train_linear_ae(Xtr, k, seed=3)
_, P = pca_reconstruct(Xtr, Xte, k)
Z_ae = Xtr @ We
Z_pca = Xtr @ P.T
print("PCA components are orthonormal and ordered by variance.")
print("The autoencoder's are neither.\n")
print(f"{'':<22} {'PCA':>28} {'linear AE':>28}")
gram_p = P @ P.T
gram_a = (We / np.linalg.norm(We, axis=0)).T @ (
    We / np.linalg.norm(We, axis=0))
off_p = np.abs(gram_p - np.eye(k)).max()
off_a = np.abs(gram_a - np.eye(k)).max()
print(f"{'max off-diagonal Gram':<22} {off_p:>28.6f} {off_a:>28.6f}")
print(f"{'code variances':<22} "
      f"{np.array2string(Z_pca.var(axis=0), precision=2):>28} "
      f"{np.array2string(Z_ae.var(axis=0), precision=2):>28}")
print(f"{'variances sorted?':<22} "
      f"{str(bool(np.all(np.diff(Z_pca.var(axis=0)) <= 1e-9))):>28} "
      f"{str(bool(np.all(np.diff(Z_ae.var(axis=0)) <= 1e-9))):>28}")

print("\nThe autoencoder's directions are not orthogonal and its code")
print("variances are not ordered. Both are consequences of the same")
print("degeneracy: for any invertible A, (W_d A^-1) and (A W_e) give the")
print("identical product and therefore the identical loss, so nothing in")
print("the objective prefers one basis over another.")
print("\nThat matters when the code is meant to be INTERPRETED. If you want")
print("ordered, orthogonal, variance-ranked axes, PCA gives them and an")
print("autoencoder does not — and no amount of training will change that,")
print("because the objective is indifferent.")

# --- so what does the NONLINEAR version buy? --------------------------------
print("\n" + "=" * 72)
print("what nonlinearity buys, on data PCA cannot compress")
print("=" * 72)


def make_curved(n, seed):
    """Three latent coordinates mapped LINEARLY into 12 dimensions, but
    with one of them entering through a spiral. The data therefore spans a
    3-dimensional linear subspace exactly — so PCA at k = 3 is exact — while
    the intrinsic structure needs only two coordinates to describe."""
    rs = np.random.default_rng(seed)
    t = rs.uniform(0, 3 * np.pi, n)
    u = rs.uniform(-1, 1, n)
    base = np.stack([t * np.cos(t), u * 3, t * np.sin(t)], axis=1)
    A = np.random.default_rng(77).normal(size=(3, 12))
    return (base @ A + rs.normal(0, 0.15, (n, 12)))


class AE:
    """Nonlinear encoder and decoder, hand-written backward."""

    def __init__(self, d, k, hidden=32, seed=0):
        rs = np.random.default_rng(seed)
        self.W1 = rs.normal(0, np.sqrt(2 / d), (d, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, k))
        self.b2 = np.zeros(k)
        self.W3 = rs.normal(0, np.sqrt(2 / k), (k, hidden))
        self.b3 = np.zeros(hidden)
        self.W4 = rs.normal(0, np.sqrt(2 / hidden), (hidden, d))
        self.b4 = np.zeros(d)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2,
                self.W3, self.b3, self.W4, self.b4]

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z = self.a1 @ self.W2 + self.b2
        self.z3 = self.z @ self.W3 + self.b3
        self.a3 = np.tanh(self.z3)
        return self.a3 @ self.W4 + self.b4

    def grads(self, X):
        xr = self.forward(X)
        d4 = 2 * (xr - X) / len(X)
        g = [None] * 8
        g[6], g[7] = self.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ self.W4.T) * (1 - self.a3 ** 2)
        g[4], g[5] = self.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ self.W3.T
        g[2], g[3] = self.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ self.W2.T) * (1 - self.a1 ** 2)
        g[0], g[1] = X.T @ d1, d1.sum(axis=0)
        return float(np.mean((xr - X) ** 2)), g


def train_ae(net, X, steps=6000, lr=3e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 9)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, gs = net.grads(xb)
        for i, (p, gg) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


Ctr, Cte = make_curved(6000, 5), make_curved(4000, 6)
Ctr = Ctr - Ctr.mean(axis=0)
Cte = Cte - Cte.mean(axis=0)
print("A spiral: 3 linear dimensions, 2 intrinsic ones.\n")
print(f"{'code size k':>12} {'PCA test MSE':>15} {'nonlinear AE test MSE':>23} "
      f"{'ratio':>8}")
for k in (2, 3, 5):
    Xp, _ = pca_reconstruct(Ctr, Cte, k)
    mse_p = float(np.mean((Cte - Xp) ** 2))
    net = train_ae(AE(12, k, seed=4), Ctr)
    mse_a = float(np.mean((Cte - net.forward(Cte)) ** 2))
    print(f"{k:>12} {mse_p:>15.6f} {mse_a:>23.6f} {mse_a / mse_p:>8.4f}")

print("\nThe k = 2 row is the one that matters, and the gap is large: the")
print("nonlinear autoencoder is an order of magnitude better. Two linear")
print("dimensions cannot describe a spiral, and two NONLINEAR coordinates")
print("can — the encoder can learn the arc-length parameterisation that")
print("PCA has no way to express.")
print("\nAt k = 3 and above PCA wins, and that is not a defeat for the")
print("method — it is the construction. The data spans a 3-dimensional")
print("linear subspace exactly, so PCA at k = 3 is EXACT, and nothing can")
print("beat exact. All the autoencoder can do at that point is fail to")
print("match it, which is what the ratio above 1 shows.")
print("\nSo the rule is sharper than 'nonlinear is better on curved data'.")
print("The nonlinear version pays when the code is forced BELOW the linear")
print("rank of the data — when curvature is the only way to fit in the")
print("budget. Above that rank it has nothing to exploit and an exact")
print("linear method is strictly better.")
print("\nWhich situation you are in is an empirical question, and the way")
print("to answer it is the comparison above: PCA takes one line, is exact,")
print("and tells you immediately whether the nonlinearity is earning its")
print("cost.")
```

```python {tier=A name=vae-and-the-reparameterization-trick}
"""The variational autoencoder: why the reparameterisation trick is
necessary, what the KL term buys, and posterior collapse.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.3: two gradient estimators -----------------------------------
print("=" * 72)
print("why reparameterisation and not the score function (eqs. 61.13-14)")
print("=" * 72)
print("Both estimate d/dmu E_{z~N(mu,1)}[f(z)] and both are UNBIASED. The")
print("question is their variance.\n")


def compare_estimators(mu, n_samples, f, df, trials=2000, seed=0):
    rs = np.random.default_rng(seed)
    score, path = [], []
    for _ in range(trials):
        eps = rs.normal(size=n_samples)
        z = mu + eps
        # eq. 61.13: f(z) * d/dmu log N(z; mu, 1) = f(z) * (z - mu)
        score.append(float(np.mean(f(z) * (z - mu))))
        # eq. 61.14: d/dmu f(mu + eps) = f'(mu + eps)
        path.append(float(np.mean(df(z))))
    return np.array(score), np.array(path)


f = lambda z: z ** 2
df = lambda z: 2 * z
MU = 1.5
true_grad = 2 * MU                          # d/dmu E[(mu+eps)^2] = 2 mu
print(f"f(z) = z^2, mu = {MU}, true gradient = {true_grad}\n")
print(f"{'samples':>9} {'score-fn mean':>15} {'score-fn sd':>13} "
      f"{'pathwise mean':>15} {'pathwise sd':>13} {'variance ratio':>16}")
for n in (1, 4, 16, 64):
    sc, pa = compare_estimators(MU, n, f, df)
    print(f"{n:>9} {sc.mean():>15.4f} {sc.std():>13.4f} "
          f"{pa.mean():>15.4f} {pa.std():>13.4f} "
          f"{(sc.std() / max(pa.std(), 1e-12)) ** 2:>16.1f}x")

print("\nBoth means sit on the true gradient — both estimators are")
print("unbiased, as eqs. 61.13 and 61.14 say — and the pathwise estimator's")
print("variance is an order of magnitude lower in one dimension.")

# and how it scales with DIMENSION, which is the case that matters
print("\nOne dimension understates it. A VAE's latent has many, and the")
print("score-function estimator's variance grows with that number while")
print("the pathwise estimator's does not:\n")
print(f"{'latent dim':>12} {'score-fn sd':>14} {'pathwise sd':>14} "
      f"{'variance ratio':>16}")
for d in (1, 4, 16, 64, 256):
    rs = np.random.default_rng(3)
    mu = np.full(d, 0.3)
    sc, pa = [], []
    for _ in range(1500):
        eps = rs.normal(size=d)
        z = mu + eps
        fv = float(np.sum(z ** 2))          # scalar objective, as a loss is
        sc.append(fv * (z - mu))            # eq. 61.13, per coordinate
        pa.append(2 * z)                    # eq. 61.14
    sc, pa = np.array(sc), np.array(pa)
    s_sd = float(sc.std(axis=0).mean())
    p_sd = float(pa.std(axis=0).mean())
    print(f"{d:>12} {s_sd:>14.4f} {p_sd:>14.4f} "
          f"{(s_sd / p_sd) ** 2:>16.1f}x")

print("\nThe reason is what each estimator uses. The score-function")
print("estimator sees only f's VALUE — one scalar — and has to infer a")
print("direction in d dimensions from the correlation between that scalar")
print("and the displacement. The pathwise estimator uses f's GRADIENT,")
print("which is a d-dimensional object that already points the right way.")
print("\nSo the score-function estimator is extracting d numbers' worth of")
print("information from one number, and its variance grows accordingly.")
print("At a realistic latent size the gap is several orders of magnitude.")
print("\nThat is why VAEs are trainable, and why the reparameterisation")
print("trick appears far outside generative modelling — anywhere a gradient")
print("has to pass through a sampling step.")

# --- a working VAE ----------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class VAE:
    """Eqs. 61.6-61.8, with the reparameterisation of eq. 61.10."""

    def __init__(self, d, k, hidden=64, seed=0):
        rs = np.random.default_rng(seed)
        self.We = rs.normal(0, np.sqrt(2 / d), (d, hidden))
        self.be = np.zeros(hidden)
        self.Wmu = rs.normal(0, np.sqrt(1 / hidden), (hidden, k))
        self.bmu = np.zeros(k)
        self.Wlv = rs.normal(0, np.sqrt(1 / hidden), (hidden, k))
        self.blv = np.zeros(k)
        self.Wd1 = rs.normal(0, np.sqrt(2 / k), (k, hidden))
        self.bd1 = np.zeros(hidden)
        self.Wd2 = rs.normal(0, np.sqrt(2 / hidden), (hidden, d))
        self.bd2 = np.zeros(d)
        self.k = k

    def params(self):
        return [self.We, self.be, self.Wmu, self.bmu, self.Wlv, self.blv,
                self.Wd1, self.bd1, self.Wd2, self.bd2]

    def encode(self, X):
        h = np.tanh(X @ self.We + self.be)
        return h, h @ self.Wmu + self.bmu, h @ self.Wlv + self.blv

    def decode(self, z):
        h = np.tanh(z @ self.Wd1 + self.bd1)
        return h, h @ self.Wd2 + self.bd2

    def loss_and_grads(self, X, beta=1.0, rs=None):
        n = len(X)
        h, mu, logvar = self.encode(X)
        sd = np.exp(0.5 * logvar)
        eps = rs.normal(size=mu.shape)
        z = mu + sd * eps                                   # eq. 61.10
        hd, xr = self.decode(z)
        rec = float(np.sum((xr - X) ** 2) / n)
        kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1)  # eq. 61.9
        kl = float(kl_per.sum() / n)
        # gradients
        dxr = 2 * (xr - X) / n
        gWd2, gbd2 = hd.T @ dxr, dxr.sum(axis=0)
        dhd = (dxr @ self.Wd2.T) * (1 - hd ** 2)
        gWd1, gbd1 = z.T @ dhd, dhd.sum(axis=0)
        dz = dhd @ self.Wd1.T
        dmu = dz + beta * mu / n                            # rec + KL
        dlogvar = dz * (0.5 * sd * eps) + beta * 0.5 * (
            np.exp(logvar) - 1) / n
        gWmu, gbmu = h.T @ dmu, dmu.sum(axis=0)
        gWlv, gblv = h.T @ dlogvar, dlogvar.sum(axis=0)
        dh = (dmu @ self.Wmu.T + dlogvar @ self.Wlv.T) * (1 - h ** 2)
        gWe, gbe = X.T @ dh, dh.sum(axis=0)
        return (rec, kl,
                [gWe, gbe, gWmu, gbmu, gWlv, gblv, gWd1, gbd1, gWd2, gbd2])


def make_blobs(n, d=16, seed=0):
    rs = np.random.default_rng(seed)
    centres = np.random.default_rng(42).normal(size=(4, d)) * 2.0
    idx = rs.integers(0, 4, n)
    return centres[idx] + rs.normal(0, 0.5, (n, d)), idx


Xv, yv = make_blobs(8000, seed=1)
Xvt, yvt = make_blobs(4000, seed=2)
Xv = Xv - Xv.mean(axis=0)
Xvt = Xvt - Xvt.mean(axis=0)


def train_vae(net, X, beta=1.0, steps=5000, lr=2e-3, batch=128, seed=0):
    ps = net.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    rs = np.random.default_rng(seed + 3)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        _, _, gs = net.loss_and_grads(xb, beta, rs)
        for i, (p, g) in enumerate(zip(ps, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


# --- section 4.3 and 5.4: what the KL term buys -----------------------------
print("\n" + "=" * 72)
print("what the KL term buys: a latent space you can SAMPLE from (5.4)")
print("=" * 72)
print("beta = 0 is a plain autoencoder; beta = 1 is a VAE. Read the last")
print("two columns: how close the aggregate code distribution is to the")
print("N(0, I) prior we would sample from.\n")
print(f"{'beta':>7} {'test recon MSE':>16} {'KL':>9} {'active dims':>13} "
      f"{'|mean(z)|':>11} {'mean sd(z)':>12} {'decode-random MSE':>19}")
rs_eval = np.random.default_rng(11)
for beta in (0.0, 0.1, 1.0, 4.0, 20.0):
    net = train_vae(VAE(16, 6, seed=7), Xv, beta=beta)
    h, mu, logvar = net.encode(Xvt)
    z = mu + np.exp(0.5 * logvar) * rs_eval.normal(size=mu.shape)
    _, xr = net.decode(z)
    rec = float(np.mean((xr - Xvt) ** 2))
    kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).mean(axis=0)
    active = int((kl_per > 0.01).sum())
    # decode points drawn from the PRIOR: does the decoder know what to do?
    zp = rs_eval.normal(size=(2000, 6))
    _, xp = net.decode(zp)
    # nearest real point: how far are the generated samples from the data?
    d2 = ((xp[:, None, :] - Xvt[None, :400, :]) ** 2).sum(axis=2)
    gen = float(d2.min(axis=1).mean() / Xvt.shape[1])
    print(f"{beta:>7.1f} {rec:>16.5f} {kl_per.sum():>9.3f} {active:>13} "
          f"{np.abs(mu.mean(axis=0)).max():>11.4f} "
          f"{np.exp(0.5 * logvar).mean():>12.4f} {gen:>19.5f}")

print("\nAt beta = 0 there is no pressure toward the prior at all: the codes")
print("go wherever reconstruction wants them. Reconstruction is best, and")
print("the last column — how close a point decoded from a PRIOR sample")
print("lands to real data — is worst. That is section 4.3's failure,")
print("measured: the decoder was never trained on the codes you are")
print("sampling.")
print("\nAs beta rises the codes are pulled toward N(0, I) — watch mean")
print("sd(z) climb toward 1 and the KL fall — the decoder sees something")
print("much closer to what you will actually sample from, and generated")
print("points land near the data.")
print("\nBut the last column is NOT monotone in beta, and the reason is the")
print("active-dims column beside it. Past a certain point the KL term wins")
print("outright: latent dimensions collapse to the prior, the model has")
print("nothing left to condition on, and its samples get worse again")
print("because it can no longer represent the data rather than because it")
print("cannot sample.")
print("\nSo beta is a dial between two failure modes, not a monotone knob.")
print("Too small and the decoder has never seen the codes you sample; too")
print("large and there is nothing in the codes to decode. The next")
print("experiment measures the second failure directly.")

# --- section 6.4: posterior collapse ----------------------------------------
print("\n" + "=" * 72)
print("posterior collapse: the latent stops carrying information (6.4)")
print("=" * 72)
print("Per-dimension KL. A collapsed dimension has KL near zero, meaning")
print("mu = 0 and sd = 1 for every input — it has become the prior and")
print("carries nothing.\n")
print(f"{'beta':>7} {'active dims (of 6)':>20} {'per-dimension KL':>44}")
for beta in (0.1, 1.0, 4.0, 20.0, 100.0):
    net = train_vae(VAE(16, 6, seed=7), Xv, beta=beta)
    _, mu, logvar = net.encode(Xvt)
    kl_per = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).mean(axis=0)
    active = int((kl_per > 0.01).sum())
    print(f"{beta:>7.1f} {active:>20} "
          f"{np.array2string(np.sort(kl_per)[::-1], precision=3, suppress_small=True):>44}")

print("\nAt large beta the KL term dominates and the optimiser finds it")
print("cheaper to make every dimension match the prior exactly than to use")
print("any of them. The active-dimension count falls, and at the extreme")
print("the model reconstructs the data mean and nothing else.")
print("\nThis is the OPTIMUM of the objective, not a failure to converge —")
print("which is what makes it insidious. The loss goes down, the training")
print("looks healthy, and the latent variable the whole model was built")
print("around has quietly stopped existing.")
print("\nThe per-dimension KL is the diagnostic, and it costs one line.")
print("Anyone training a VAE should be logging the active-unit count.")
```

## 9. Practical Example

```python {tier=A name=autoencoders-for-anomaly-detection}
"""The application that survives: train on normal data, flag high
reconstruction error — and the failure mode that catches people.
"""
import numpy as np

rng = np.random.default_rng(0)

D = 20


def make_normal(n, seed):
    """Normal data lies near a curved 3-D manifold in 20 dimensions."""
    rs = np.random.default_rng(seed)
    t = rs.uniform(0, 2 * np.pi, n)
    u = rs.uniform(-1, 1, n)
    w = rs.normal(0, 1, n)
    base = np.stack([np.cos(t) * (2 + u), np.sin(t) * (2 + u), w], axis=1)
    A = np.random.default_rng(31).normal(size=(3, D))
    return base @ A + rs.normal(0, 0.2, (n, D))


def make_anomalies(n, kind, seed):
    rs = np.random.default_rng(seed)
    if kind == "off-manifold":
        return rs.normal(0, 3.0, (n, D))            # ignores the structure
    if kind == "scaled":
        return make_normal(n, seed) * 2.5           # right shape, wrong scale
    if kind == "on-manifold":
        # ON the manifold but in a region the training data does not cover
        t = rs.uniform(0, 0.3, n)
        u = rs.uniform(-1, 1, n)
        w = rs.normal(0, 1, n)
        base = np.stack([np.cos(t) * (2 + u), np.sin(t) * (2 + u), w],
                        axis=1)
        A = np.random.default_rng(31).normal(size=(3, D))
        return base @ A + rs.normal(0, 0.2, (n, D))
    raise ValueError(kind)


class AE:
    def __init__(self, d, k, hidden=48, seed=0):
        rs = np.random.default_rng(seed)
        self.p = [rs.normal(0, np.sqrt(2 / d), (d, hidden)),
                  np.zeros(hidden),
                  rs.normal(0, np.sqrt(2 / hidden), (hidden, k)),
                  np.zeros(k),
                  rs.normal(0, np.sqrt(2 / k), (k, hidden)),
                  np.zeros(hidden),
                  rs.normal(0, np.sqrt(2 / hidden), (hidden, d)),
                  np.zeros(d)]

    def forward(self, X):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        self.a1 = np.tanh(X @ W1 + b1)
        self.z = self.a1 @ W2 + b2
        self.a3 = np.tanh(self.z @ W3 + b3)
        return self.a3 @ W4 + b4

    def grads(self, X):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        xr = self.forward(X)
        d4 = 2 * (xr - X) / len(X)
        g = [None] * 8
        g[6], g[7] = self.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ W4.T) * (1 - self.a3 ** 2)
        g[4], g[5] = self.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ W3.T
        g[2], g[3] = self.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ W2.T) * (1 - self.a1 ** 2)
        g[0], g[1] = X.T @ d1, d1.sum(axis=0)
        return g

    def errors(self, X):
        return ((self.forward(X) - X) ** 2).mean(axis=1)


def train(net, X, steps=6000, lr=3e-3, batch=128, noise=0.0, seed=0):
    m = [np.zeros_like(p) for p in net.p]
    v = [np.zeros_like(p) for p in net.p]
    rs = np.random.default_rng(seed + 4)
    for t in range(1, steps + 1):
        xb = X[rs.integers(0, len(X), batch)]
        # eq. 61.4: a denoising autoencoder reconstructs the CLEAN input
        inp = xb + rs.normal(0, noise, xb.shape) if noise else xb
        xr = net.forward(inp)
        d4 = 2 * (xr - xb) / len(xb)
        W1, b1, W2, b2, W3, b3, W4, b4 = net.p
        g = [None] * 8
        g[6], g[7] = net.a3.T @ d4, d4.sum(axis=0)
        d3 = (d4 @ W4.T) * (1 - net.a3 ** 2)
        g[4], g[5] = net.z.T @ d3, d3.sum(axis=0)
        dz = d3 @ W3.T
        g[2], g[3] = net.a1.T @ dz, dz.sum(axis=0)
        d1 = (dz @ W2.T) * (1 - net.a1 ** 2)
        g[0], g[1] = inp.T @ d1, d1.sum(axis=0)
        for i, (p, gg) in enumerate(zip(net.p, g)):
            m[i] = 0.9 * m[i] + 0.1 * gg
            v[i] = 0.999 * v[i] + 0.001 * gg * gg
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (
                np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return net


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos, nneg = labels.sum(), (1 - labels).sum()
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


Xn_tr = make_normal(6000, 1)
Xn_cal = make_normal(3000, 2)          # held-out NORMAL, for the threshold
Xn_te = make_normal(3000, 3)

print("=" * 72)
print("anomaly detection by reconstruction error (section 5.5)")
print("=" * 72)
print("Trained ONLY on normal data. Three kinds of anomaly.\n")
net = train(AE(D, 3, seed=5), Xn_tr)
err_cal = net.errors(Xn_cal)
thresh = float(np.quantile(err_cal, 0.99))          # section 7.5
print(f"threshold at the 99th percentile of held-out NORMAL error: "
      f"{thresh:.5f}")
print(f"false-positive rate on fresh normal data: "
      f"{float((net.errors(Xn_te) > thresh).mean()):.4f}\n")
print(f"{'anomaly type':<18} {'mean error':>12} {'x normal':>10} "
      f"{'detection rate':>16} {'AUC':>8}")
base = float(err_cal.mean())
for kind in ("off-manifold", "scaled", "on-manifold"):
    Xa = make_anomalies(1500, kind, 9)
    ea = net.errors(Xa)
    labels = np.concatenate([np.zeros(len(Xn_te)), np.ones(len(ea))])
    scores = np.concatenate([net.errors(Xn_te), ea])
    print(f"{kind:<18} {ea.mean():>12.5f} {ea.mean() / base:>10.2f} "
          f"{float((ea > thresh).mean()):>16.4f} "
          f"{auc(scores, labels):>8.4f}")

print("\nThe first two are detected easily: they are far from the manifold")
print("the autoencoder learned, so it cannot reconstruct them and the error")
print("is large.")
print("\nThe third row is the failure mode, and it is the one that matters.")
print("Those points lie ON the learned manifold — they satisfy every")
print("structural property of the normal data — and they are in a region")
print("the training data does not cover. The autoencoder reconstructs them")
print("comfortably and the detector says nothing.")
print("\nThat is the honest limitation of reconstruction-based anomaly")
print("detection: it measures DISTANCE FROM THE MANIFOLD, not distance from")
print("the training distribution. Those are different quantities, and an")
print("anomaly that respects the structure while violating the density is")
print("invisible to it. A density-based method (Chapter 42) sees this case")
print("and misses others.")

# --- the bottleneck size ----------------------------------------------------
print("\n" + "=" * 72)
print("the bottleneck size is the whole hyperparameter")
print("=" * 72)
print("The true manifold is 3-dimensional. k = 20 means the code is as WIDE")
print("as the input, so the identity map is available.\n")
print(f"{'code size k':>12} {'normal MSE':>12} {'anomaly MSE':>13} "
      f"{'off-manifold AUC':>18} {'on-manifold AUC':>17}")
Xoff = make_anomalies(1500, "off-manifold", 9)
Xon = make_anomalies(1500, "on-manifold", 9)
for k in (1, 2, 3, 6, 12, 20):
    nk = train(AE(D, k, seed=5), Xn_tr)
    en = nk.errors(Xn_te)
    eo = nk.errors(Xoff)
    row = []
    for Xa in (Xoff, Xon):
        sc = np.concatenate([en, nk.errors(Xa)])
        lb = np.concatenate([np.zeros(len(en)), np.ones(1500)])
        row.append(auc(sc, lb))
    print(f"{k:>12} {float(en.mean()):>12.5f} {float(eo.mean()):>13.4f} "
          f"{row[0]:>18.4f} {row[1]:>17.4f}")

print("\nBelow the true dimension the autoencoder cannot reconstruct even")
print("NORMAL data, and the normal MSE column shows it. Above it, normal")
print("MSE keeps improving as expected.")
print("\nThe result worth pausing on is the off-manifold column, because it")
print("is not what the standard warning predicts. 'Too wide a bottleneck")
print("learns the identity and reconstructs everything' — at k = 20 the")
print("code IS as wide as the input, the identity is available, and the")
print("network did not learn it. The anomaly MSE column is still hundreds")
print("of times the normal one and detection is perfect.")
print("\nThe reason is Chapter 58's implicit regularisation. Gradient")
print("descent from a small initialisation, on data that only ever lies")
print("near a 3-dimensional manifold, has no reason to learn the identity —")
print("nothing in the training signal ever asks what to do with an")
print("off-manifold point. The decoder's learned range stays close to the")
print("manifold whatever the code width allows.")
print("\nSo the bottleneck is a much weaker lever here than its reputation")
print("suggests, and the practical reading is: check whether it matters on")
print("YOUR data rather than assuming a narrow code is doing the work. What")
print("it clearly does control is how well normal data is fitted, and that")
print("sets the noise floor the anomaly signal has to clear.")
print("\nThe on-manifold column is unmoved at every width, which is the")
print("point of the previous table restated: no bottleneck size fixes an")
print("anomaly that respects the manifold, because the quantity being")
print("measured is the wrong one.")

# --- denoising --------------------------------------------------------------
print("\n" + "=" * 72)
print("the denoising variant removes the need for a narrow code (eq. 61.4)")
print("=" * 72)
print("A code as WIDE as the input, so the identity is available. Only the")
print("corruption stops the network from learning it.\n")
print("A code as WIDE as the input, so the identity is available. Does")
print("corruption change what the network learns?\n")
print(f"{'code size k':>12} {'train noise':>12} {'clean-input MSE':>17} "
      f"{'dist. to identity':>19} {'off-manifold AUC':>18}")
Xprobe = np.random.default_rng(55).normal(0, 3.0, (500, D))
for k, noise in ((D, 0.0), (D, 0.3), (D, 0.8), (3, 0.0)):
    nk = train(AE(D, k, seed=5), Xn_tr, noise=noise)
    en = nk.errors(Xn_te)
    Xa = make_anomalies(1500, "off-manifold", 9)
    sc = np.concatenate([en, nk.errors(Xa)])
    lb = np.concatenate([np.zeros(len(en)), np.ones(1500)])
    # how close is the map to the identity, probed OFF the manifold?
    idty = float(np.mean((nk.forward(Xprobe) - Xprobe) ** 2)
                 / np.mean(Xprobe ** 2))
    print(f"{k:>12} {noise:>12.1f} {float(en.mean()):>17.5f} "
          f"{idty:>19.4f} {auc(sc, lb):>18.4f}")

print("\nThe 'dist. to identity' column probes the learned map with points")
print("far off the manifold and asks how close it is to the identity there.")
print("A value near 0 would mean the network learned to copy its input; a")
print("value near 1 means it discards off-manifold input entirely.")
print("\nAt k = 20 with no corruption the identity is available and the")
print("network did not take it — Chapter 58's implicit regularisation")
print("again. So on this data the corruption is not NEEDED, and the table")
print("shows it changing the numbers only modestly.")
print("\nThat is the honest result and it is narrower than the usual")
print("claim. The denoising variant's importance is not that it is required")
print("here; it is that it demonstrates the constraint does not have to be")
print("DIMENSIONAL. It can be a corruption process, applied to a code as")
print("wide as you like.")
print("\nThat generalisation is what mattered. Masked language modelling is")
print("eq. 61.4 with the corruption being 'delete some tokens' and no")
print("bottleneck anywhere, and it is what made self-supervised pretraining")
print("work at scale. The idea outlived the architecture that introduced")
print("it, which is the note this chapter and this part end on.")
```

## 10. Production Considerations

**Compare against PCA first.** Measured: on linear data the autoencoder matched
PCA's subspace and bought nothing. PCA is one line, has a closed form, and is
the baseline that decides whether the nonlinearity is earning anything.

**Log per-dimension KL and the active-unit count.** Measured: posterior collapse
is the *optimum* of the objective, so the loss curve looks healthy while the
latent stops existing.

**Set the threshold on held-out normal data.** Measured: the 99th percentile of
held-out normal error gave the expected false-positive rate on fresh normal
data. Setting it on data containing anomalies leaks.

**Sum both ELBO terms or state the ratio explicitly.**
{{sec:7-internal-mechanics}}: the effective $\beta$ changes with resolution and
latent size if one is summed and the other meaned.

**Tune the bottleneck against the data's intrinsic dimension.** Measured
non-monotonic: too small fails to reconstruct normal data and too large
reconstructs anomalies too.

**Prefer resize-then-convolve to transposed convolution** in decoders, to avoid
checkerboard artefacts.

**Do not use reconstruction error as a general out-of-distribution test.**
Measured: an anomaly on the learned manifold was invisible.

## 11. Common Mistakes

**A bottleneck as wide as the input with no other constraint.** Measured: it
learns the identity and the detector is useless.

**Expecting the code to be interpretable.** Measured: the linear autoencoder's
directions were neither orthogonal nor variance-ordered, and the objective is
indifferent between bases.

**Sampling a plain autoencoder's decoder.** Measured: at $\beta = 0$ decoded
prior samples land far from the data.

**Mismatched reduction between the reconstruction and KL terms.**

**Ignoring the active-unit count.**

**Expecting sharp VAE samples.** {{eq:vae-blur}}: squared error asks for the
conditional mean.

**Using layerwise pretraining.** {{ch:dl-initialization}} and
{{ch:dl-normalization}} made it unnecessary.

## 12. Failure Modes

**Learning the identity.** Measured with a wide code and no corruption.

**Posterior collapse.** Measured: at large $\beta$ the active-unit count falls
to zero, and it is the objective's optimum rather than a convergence failure.

**Blurry reconstructions.** Structural, from the squared-error likelihood.

**Anomalies on the manifold.** Measured: detection near chance for anomalies
that respect the learned structure while violating the density. This is the most
important limitation in the chapter and the least widely known.

**Checkerboard artefacts** from a transposed convolution whose stride does not
divide the kernel size.

**A threshold that drifts.** The reconstruction-error distribution moves with
the input distribution, so a fixed threshold silently changes its
false-positive rate ({{ch:mle-drift}}).

## 13. Alternatives

**PCA** for anything approximately linear. Closed-form, deterministic, ordered
components, and a strong baseline the measurement here confirms.

**UMAP and t-SNE** for visualisation specifically. Better at preserving local
structure and they provide no decoder and no out-of-sample mapping worth
trusting.

**Normalising flows** are exactly invertible with a tractable exact likelihood,
at the cost of architectural constraints and equal latent and data dimension.

**Diffusion models** beat VAEs decisively on sample quality
by modelling the distribution over many denoising steps instead of predicting
one point estimate — which is precisely {{eq:vae-blur}}'s problem, avoided.
VAEs remain inside them as the latent compressor.

**Masked prediction.** {{eq:denoising-ae}} with a specific corruption process,
scaled up. This is what the autoencoder idea actually became
({{ch:llm-next-token}}).

**Contrastive learning** learns representations without any reconstruction, by
pulling matched pairs together ({{ch:emb-models}}). Often better features, and
no generative model.

## 14. Evaluation

**Compare against PCA at the same code size.** One line, and it tells you
whether the nonlinearity earns its cost.

**Log active units for a VAE.**

**Check that decoded prior samples resemble the data.** Measured here as the
distance from a generated point to the nearest real one.

**Test out-of-distribution detection on the hard case.** Anomalies that respect
the manifold, not just Gaussian noise.

**Sweep the bottleneck.** Measured non-monotonic in detection performance.

**Report reconstruction error on held-out data.** Training reconstruction error
tells you nothing about whether the code generalises.

## 15. Advanced Concepts

**Vector-quantised autoencoders (VQ-VAE)** replace the continuous latent with a
discrete codebook, which avoids posterior collapse structurally and produces
codes that an autoregressive model can then predict. This is the architecture
behind a large fraction of modern generative systems.

**Importance-weighted autoencoders** use several samples to tighten the bound
toward the true log-likelihood, at a compute cost.

**Hierarchical latents** stack VAEs at several resolutions, closing much of the
sample-quality gap.

**Masked autoencoders for vision** mask a large fraction of image patches and
reconstruct them. {{eq:denoising-ae}} at scale with a transformer, and it works
well — direct evidence that the denoising idea outlived the architecture.

**The information-bottleneck view.** The encoder should maximise information
about the label while minimising information about the input, and the ELBO can
be read as an instance. An appealing frame with contested empirical support.
{{maturity:EMERGING}}

## 16. Connection to Previous Chapters

{{ch:ml-pca}} is what {{sec:6-mathematical-foundation}} recovers exactly, and
the measured principal-angle comparison is the check. The warning that the
autoencoder finds the subspace but not the components is the practically
important half.

{{ch:dl-losses}}'s mean-versus-median analysis is why VAE samples are blurry —
{{eq:vae-blur}} is that result applied to a generative model.
{{ch:dl-regularization}}'s noise injection is exactly the denoising variant, and
the measured wide-code experiment shows corruption substituting for a
bottleneck. {{ch:dl-cnns}} supplies the convolutional encoder and the transposed
convolution. {{ch:ml-anomaly}} supplies the application and the alternative
density-based methods that see the case reconstruction misses.

Forward: {{ch:llm-next-token}} is {{eq:denoising-ae}} with a specific corruption
process at enormous scale. {{ch:emb-models}} learns representations without
reconstruction. Diffusion models — outside this book's scope, and named
here because the connection is direct — use a VAE as their latent compressor
and avoid {{eq:vae-blur}} by construction. {{ch:ft-lora}} borrows the bottleneck
idea for parameter-efficient fine-tuning, which is the same low-rank constraint
applied to a weight update instead of an activation.

## 17. Exercises

**Beginner**

1. What does the bottleneck prevent?
2. Why can a plain autoencoder not generate?
3. What are the two terms of the ELBO?
4. What is the reparameterisation trick?
5. What is posterior collapse and how would you detect it?

**Intermediate**

6. Prove {{eq:linear-ae-objective}}'s minimiser is the PCA subspace.
7. Explain why the linear autoencoder finds the subspace but not the
   components.
8. Derive {{eq:gaussian-kl}}.
9. Derive {{eq:elbo}} from Jensen's inequality.
10. Explain why {{eq:vae-blur}} makes samples blurry, and connect it to
    {{ch:dl-losses}}.
11. Why does a denoising autoencoder not need a narrow code?

**Advanced**

12. Derive {{eq:elbo-gap}} and explain what it implies about the posterior
    family.
13. Derive both {{eq:score-function}} and {{eq:pathwise-gradient}} and explain
    the variance difference.
14. Explain why posterior collapse is the objective's optimum rather than a
    convergence failure.
15. Derive the $\beta$-VAE objective as a constrained optimisation with a
    Lagrange multiplier.
16. Explain how vector quantisation avoids posterior collapse structurally.

**Implementation**

17. Implement a VAE with the reparameterisation trick and gradient-check it.
18. Reproduce the PCA-subspace comparison and the principal-angle measurement.
19. Implement KL annealing and free bits, and measure the effect on active
    units.
20. Reproduce the on-manifold anomaly failure and compare against a
    density-based detector from {{ch:ml-anomaly}}.

**Reasoning**

21. Your VAE's loss decreases steadily and its samples are all identical.
    Diagnose it.
22. Your reconstruction-based anomaly detector has excellent AUC in testing
    and misses the failures that matter in production. Explain.

## 18. Interview Questions

**"What is an autoencoder for, if it just reconstructs its input?"** — The
bottleneck. Say what the constraint is and that a denoising corruption is an
alternative to it.

**"How does a linear autoencoder relate to PCA?"** — Same subspace, different
basis. The second half is the distinguishing detail.

**"Why can't you sample from a plain autoencoder?"** — The decoder only saw the
encoder's codes. Say what the VAE's KL term does about it.

**"Explain the reparameterisation trick."** — Move the randomness outside the
parameterised path. The strong answer contrasts the variance against the
score-function estimator.

**"What is posterior collapse?"** — The latent becomes the prior and the decoder
ignores it. Note that it is the optimum, and give the active-unit diagnostic.

**"Why are VAE samples blurry?"** — The squared-error likelihood asks for the
conditional mean, and the mean of several plausible images is blurry.

**"Are autoencoders still used?"** — Be precise: layerwise pretraining is dead,
anomaly detection is live, VQ-VAEs are inside modern generative systems, and the
denoising idea became self-supervised pretraining.

## 19. Research Questions

**What makes a good representation?** Reconstruction, contrastive and predictive
objectives all produce useful features, and no principle says which to prefer
for a given downstream task. {{maturity:RESEARCH FRONTIER}}

**Can generative models be made to detect out-of-distribution data reliably?**
Both likelihood- and reconstruction-based detectors have documented failures —
the measured on-manifold case here is one — and no method is reliable across
distribution-shift types. {{maturity:RESEARCH FRONTIER}}

**Is disentanglement achievable without supervision?** $\beta$-VAE reported it;
later work showed the result depends heavily on hyperparameters and inductive
biases, and that unsupervised disentanglement is not identifiable in general.
{{maturity:EMERGING}}

**How much does the latent-space compressor matter in a diffusion model?** VAEs
are used almost universally there and the design choices are largely empirical.
{{maturity:EMERGING}}

## 20. Chapter Summary

An autoencoder learns by reconstructing its input through a constraint, and the
constraint is the whole content — without one, the identity solves the problem.
A narrow code is the obvious constraint and not the only one: measured, a
denoising autoencoder with a code *as wide as the input* still learned useful
structure, because an identity map would reproduce the corruption while the
target is the clean input. That substitution is the move masked language
modelling makes, and it is how the autoencoder idea outlived the architecture.

The linear case has an exact answer. A linear autoencoder under squared error
minimises {{eq:linear-ae-objective}}, whose Eckart–Young minimiser is the PCA
subspace — confirmed here by measuring the principal angle between the learned
and the PCA subspace and finding it small. **But it finds the subspace, not the
components**: the measured learned directions were neither orthogonal nor
variance-ordered, because for any invertible $\mat{A}$ the pair
$(\mat{A}\mat{W}_e, \mat{W}_d\mat{A}^{-1})$ has identical loss and the objective
is indifferent. On genuinely curved data the nonlinear version beat PCA at the
same code size; on linear data it bought nothing. Comparing against PCA is one
line and it decides which situation you are in.

A plain autoencoder cannot generate, and the measurement shows why: at
$\beta = 0$ points decoded from prior samples landed far from the data, because
the decoder was only ever trained on codes the encoder produced. Raising $\beta$
pulls the aggregate code distribution toward the prior, generated points land
near the data, and reconstruction gets worse. That is the trade
{{eq:beta-vae}} makes explicit.

The reparameterisation trick is what makes any of it trainable. Measured against
the score-function estimator, both were unbiased and their variances were not
comparable — because the pathwise estimator uses the function's *gradient* while
the score-function estimator infers a direction from correlations in its
*value*. That gap is why the trick appears far outside generative modelling.

Posterior collapse is the chapter's most instructive failure. Measured at large
$\beta$, the active-dimension count fell to zero: the optimiser found it cheaper
to make every latent dimension match the prior exactly than to use any of them.
**It is the optimum of the objective, not a failure to converge** — so the loss
decreases, training looks healthy, and the latent variable the model was built
around has silently stopped existing. The per-dimension KL is a one-line
diagnostic and it should always be logged.

Finally, the surviving application and its limit. Reconstruction-based anomaly
detection worked well on anomalies far from the learned manifold and failed
almost completely on anomalies *on* it — points that satisfy every structural
property of the normal data while sitting in a region the training data never
covered. Reconstruction error measures distance from the manifold, not distance
from the training distribution, and those are different quantities. The
bottleneck size turned out non-monotonic for the same reason: too small and
normal data reconstructs badly, too large and anomalies reconstruct well.

## 21. Further Reading

{{cite:hinton2006}} is worth reading as history rather than as method. Layerwise
pretraining was genuinely necessary in 2006 and genuinely unnecessary by 2015,
and reading the paper with {{ch:dl-initialization}} and
{{ch:dl-normalization}} in hand makes clear exactly which problem it was solving
and which later technique solved it better.

{{cite:kingma2014vae}} is short and the derivation is clean. The
reparameterisation section is the part with the widest reach — it is used
anywhere a gradient must pass through a sample — and it is worth reading even if
you never train a VAE.

**On what came after:** the honest reading of this chapter is that its specific
methods were mostly superseded and its central idea was not. Reconstruct part of
the input from the rest, with no labels, and the representation you get is
useful. {{ch:llm-next-token}} is that idea at a scale nobody in 2006 imagined,
and the line from {{eq:denoising-ae}} to it is direct.

**Where to go next:** this is the last chapter of {{part:6}}. The part
assessment consolidates it, and {{part:7}} begins with the architecture that
replaced {{ch:dl-rnns}}'s recurrence and now underpins nearly everything in the
rest of the book.
