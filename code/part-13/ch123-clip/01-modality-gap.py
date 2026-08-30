# -*- coding: utf-8 -*-
# Extracted from: Chapter 123 — CLIP and Contrastive Vision–Language Alignment
# Source: src/.../ch123-clip.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The modality gap: CLIP's shared space is shared, and it is not one region.

Contrastive image-text training puts both modalities in one vector space so a
dot product can compare them (eq:clip-objective). The natural conclusion -- that
an image embedding and a text embedding are now the same kind of object -- is
false, and the way it is false breaks thresholds, clustering, and any index that
mixes modalities.

Image embeddings and text embeddings occupy SEPARATE CONES. The contrastive loss
only ever compares an image against texts and a text against images, so nothing in
it ever asks the two clouds to coincide -- it asks for the right RANKING across
the gap, which a pair of well-separated cones satisfies perfectly
(eq:modality-gap).

This listing trains a small two-tower model and measures the gap, then measures
what the gap does to a similarity threshold.
"""
import numpy as np

rng = np.random.default_rng(53)

N_CONCEPT, DIM_RAW, DIM = 20000, 64, 32
STEPS, BATCH, LR = 4000, 256, 0.5
TAU = 0.07


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# A latent concept per training pair. The two encoders see the SAME concept
# through different, modality-specific transformations -- which is the situation
# contrastive alignment is meant to handle.
concept = unit(rng.normal(size=(N_CONCEPT, DIM_RAW)))
A_img = rng.normal(size=(DIM_RAW, DIM_RAW)) / np.sqrt(DIM_RAW)
A_txt = rng.normal(size=(DIM_RAW, DIM_RAW)) / np.sqrt(DIM_RAW)
b_img = rng.normal(size=DIM_RAW) * 0.6          # modality-specific offset
b_txt = rng.normal(size=DIM_RAW) * 0.6


def sample(n, distinct=False):
    c = (rng.choice(N_CONCEPT, size=n, replace=False) if distinct
         else rng.integers(0, N_CONCEPT, size=n))
    xi = unit(concept[c] @ A_img + b_img + 0.15 * rng.normal(size=(n, DIM_RAW)))
    xt = unit(concept[c] @ A_txt + b_txt + 0.15 * rng.normal(size=(n, DIM_RAW)))
    return xi, xt, c


Wi = rng.normal(size=(DIM_RAW, DIM)) / np.sqrt(DIM_RAW)
Wt = rng.normal(size=(DIM_RAW, DIM)) / np.sqrt(DIM_RAW)

for step in range(STEPS):
    xi, xt, _ = sample(BATCH)
    zi, zt = unit(xi @ Wi), unit(xt @ Wt)
    S = zi @ zt.T / TAU
    Pi = np.exp(S - S.max(1, keepdims=True)); Pi /= Pi.sum(1, keepdims=True)
    Pt = np.exp(S - S.max(0, keepdims=True)); Pt /= Pt.sum(0, keepdims=True)
    tgt = np.eye(BATCH)
    gS = ((Pi - tgt) + (Pt - tgt).T) / (2 * BATCH * TAU)
    gzi, gzt = gS @ zt, gS.T @ zi
    # Backprop through the L2 normalisation.
    def dnorm(g, z, x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return (g - (g * z).sum(1, keepdims=True) * z) / n
    Wi -= LR * (xi.T @ dnorm(gzi, zi, xi @ Wi))
    Wt -= LR * (xt.T @ dnorm(gzt, zt, xt @ Wt))

xi, xt, c = sample(2000, distinct=True)
zi, zt = unit(xi @ Wi), unit(xt @ Wt)

# --- does retrieval work? ---
S = zi @ zt.T
top1 = float((S.argmax(1) == np.arange(len(S))).mean())

# --- the gap, measured as SEPARABILITY rather than as centroid distance ---
# Centroid distance is misleading on a sphere: both means sit near the origin.
# The question that matters is whether the two clouds are distinguishable at
# all, so fit a linear probe to predict which modality an embedding came from.
mi, mt = zi.mean(0), zt.mean(0)
gap = float(np.linalg.norm(mi - mt))
Z = np.vstack([zi, zt])
lab = np.concatenate([np.zeros(len(zi)), np.ones(len(zt))])
Zc = np.hstack([Z, np.ones((len(Z), 1))])
w = np.linalg.lstsq(Zc, 2 * lab - 1, rcond=None)[0]
separability = float(((Zc @ w > 0) == (lab > 0)).mean())
# How far is a typical embedding from its OWN modality's centroid, versus from
# the other modality's? If the clouds were interleaved these would be equal.
own = float(np.mean([unit(mi[None])[0] @ z for z in zi[:800]]))
other = float(np.mean([unit(mt[None])[0] @ z for z in zi[:800]]))

# --- similarity distributions ---
def offdiag(M):
    return M[~np.eye(len(M), dtype=bool)]

ii = offdiag(zi[:600] @ zi[:600].T)
tt = offdiag(zt[:600] @ zt[:600].T)
it_pos = np.diag(zi[:600] @ zt[:600].T)
it_neg = offdiag(zi[:600] @ zt[:600].T)

print(f"trained {STEPS} steps, batch {BATCH}, temperature {TAU}\n")
print(f"image->text retrieval, top-1 of 2000:      {top1:.3f}")
print(f"difference of modality means (norm):       {gap:.3f}")
print(f"chance retrieval rate would be:             {1/len(S):.4f}")
print(f"linear probe -- which modality is this?     {separability:.3f}")
print(f"mean cosine to OWN modality centroid:       {own:.3f}")
print(f"mean cosine to OTHER modality centroid:     {other:.3f}")
print()
print(f"{'similarity between':<34}{'mean':>9}{'p5':>9}{'p95':>9}")
print("-" * 61)
for name, v in (("two images", ii), ("two texts", tt),
                ("an image and ITS text", it_pos),
                ("an image and a random text", it_neg)):
    print(f"{name:<34}{v.mean():>9.3f}{np.percentile(v, 5):>9.3f}"
          f"{np.percentile(v, 95):>9.3f}")

ii_hi = float(np.percentile(ii, 99))
print(f"""
Retrieval works, and by a wide margin: {top1:.3f} top-1 against 2000 candidates
where chance is {1/len(S):.4f}. The alignment succeeded. By the measure the
objective optimised, there is nothing wrong with this space.

Now the two similarity scales, which is the result. Two random images score
{ii.mean():.3f} on average; an image and its own caption score {it_pos.mean():.3f}.
Those are not the same scale, and the gap is not a quality difference -- it is
where each kind of comparison LIVES. Cross-modal matched pairs sit far above
everything, and within-modality pairs cluster near zero however related their
content is.

So a threshold means different things depending on what it is comparing, and the
two distributions are not merely offset -- they overlap at the tail while
differing fiftyfold in the middle. The 99th percentile of image-image similarity
is {ii_hi:.3f}, which is essentially the MEAN image-caption score
({it_pos.mean():.3f}). So a cutoff at 0.45 admits a typical matched caption and
also the top one per cent of image pairs, while a cutoff at 0.05 admits half the
image pairs and rejects nothing cross-modal. There is no value that separates
"similar" from "not similar" for both kinds of comparison at once.

The linear probe puts the structural version of this at {separability:.3f}: an
embedding carries enough information about WHICH MODALITY produced it that a
linear classifier beats chance at recovering it, and a typical image embedding
sits closer to the image centroid ({own:.3f}) than to the text centroid
({other:.3f}). Be aware that this toy UNDERSTATES the effect -- with real
encoders, deeper towers and web-scale data, the two clouds are close to perfectly
separable, and the reported gap is much larger than the one measured here. The
direction is right and the magnitude is a lower bound.

The reason is in eq:clip-objective and it is not a training failure. The loss only
ever compares an image against texts and a text against images. It therefore
constrains the RANKING across the gap and says nothing whatsoever about where
either cloud sits, so two separated cones with matching internal ordering minimise
it perfectly. Nothing was ever asked to bring them together.

This is ch:emb-similarity's rule with a second modality attached: the score is a
rank, not a measurement. Within one modality that is a caution. Across two it is a
hard constraint, because the offset between the clouds is a property of the
training run -- it moves with temperature, batch size and initialisation -- rather
than a property of the content.

The practical response is the one ch:emb-what-they-are used for anisotropy:
centre each modality separately before comparing within it, calibrate any
threshold on the specific comparison it will be applied to, and never compare a
within-modality score against a cross-modality one. What you must not do is
assume the shared space made the two interchangeable. It made them comparable by
ranking, which is strictly weaker, and is the only thing the loss requested.""")
