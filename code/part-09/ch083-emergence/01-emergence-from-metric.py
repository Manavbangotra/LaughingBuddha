# Extracted from: Chapter 83 — Emergent Capabilities and What Emergence Means
# Source: src/.../ch083-emergence.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""One smoothly improving model family, two metrics, two different stories."""
import numpy as np

# Model scale, log-spaced over four orders of magnitude.
scales = np.logspace(7, 11, 40)          # 10M to 100B parameters

# Per-step competence improves SMOOTHLY and monotonically with log scale.
# Nothing here is discontinuous. This is the ground truth.
p = 1 / (1 + np.exp(-(np.log10(scales) - 9.0) * 1.6))

K = 12                                    # sub-steps the task requires
exact = p ** K                            # equation (eq:exact-match-composition)

print("A single smooth p(s), scored two ways\n")
print(f"{'params':>10} {'per-step p':>12} {'exact match p^12':>18}")
for i in range(0, len(scales), 6):
    print(f"{scales[i]:>10.1e} {p[i]:>12.4f} {exact[i]:>18.6f}")


def sharpness(y, x):
    """Fraction of the total rise that happens in the steepest 10% of x."""
    y = (y - y.min()) / (y.max() - y.min())
    n = max(1, len(x) // 10)
    gains = [y[i + n] - y[i] for i in range(len(y) - n)]
    return max(gains)


print(f"\nlargest rise within any 10% window of log-scale:")
print(f"  per-step accuracy : {sharpness(p, scales):.3f}")
print(f"  exact match       : {sharpness(exact, scales):.3f}")

# Equation (eq:transition-width): how wide is the transition in p?
for k in (1, 6, 12, 50):
    lo, hi = 0.1 ** (1 / k), 0.9 ** (1 / k)
    print(f"k={k:>3}: exact match goes 0.1 -> 0.9 as p goes "
          f"{lo:.4f} -> {hi:.4f}  (width {hi - lo:.4f})")

# Equation (eq:sampling-transition): what a sparse scale sweep sees.
print(f"\nWhat a study with only a few model sizes observes (k={K}):")
lo, hi = 0.1 ** (1 / K), 0.9 ** (1 / K)
width = hi - lo
for m in (3, 4, 6, 10, 40):
    idx = np.linspace(0, len(scales) - 1, m).astype(int)
    in_window = int(((p[idx] > lo) & (p[idx] < hi)).sum())
    prob = 1 - (1 - width) ** m
    print(f"  {m:>3} model sizes: {in_window} land inside the transition "
          f"(predicted chance of >=1: {prob:.0%})")

print("""
Read the last block carefully. With three or four model sizes — which is what
most emergence studies have — it is likely that NO sampled model lands in the
transition window. The plot then shows a flat line at chance followed by a
jump, and the jump is an artefact of sampling a continuous function sparsely
with a metric that compresses its transition into a few per cent of the range.

The model in this listing improves perfectly smoothly. There is no emergence
anywhere in it. Everything sharp on the right-hand plot was put there by the
exponent.""")
