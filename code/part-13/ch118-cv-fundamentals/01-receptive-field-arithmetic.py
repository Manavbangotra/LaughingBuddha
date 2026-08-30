# -*- coding: utf-8 -*-
# Extracted from: Chapter 118 — Computer Vision Fundamentals
# Source: src/.../ch118-cv-fundamentals.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
