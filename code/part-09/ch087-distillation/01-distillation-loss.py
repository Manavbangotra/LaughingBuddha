# Extracted from: Chapter 87 — Distillation and Model Specialization
# Source: src/.../ch087-distillation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Knowledge distillation: soft targets, temperature, and the T^2 factor."""
import numpy as np

TEACHER_LOGITS = np.array([5.0, 2.0, 1.5, -3.0])
CLASSES = ["dog", "wolf", "cat", "aeroplane"]


def softmax(z, T=1.0):
    """Equation (eq:temperature-softmax)."""
    s = z / T
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


print("The same teacher, at different temperatures\n")
print(f"{'T':>5} " + " ".join(f"{c:>11}" for c in CLASSES) + f"{'entropy':>10}")
for T in (0.5, 1.0, 2.0, 4.0, 8.0):
    p = softmax(TEACHER_LOGITS, T)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    print(f"{T:>5.1f} " + " ".join(f"{v:>11.4f}" for v in p) + f"{ent:>10.4f}")

print("""
At T=1 the last two classes hold about 3% of the mass between them and
contribute almost nothing to a gradient. At T=4 they hold 27%, and the fact
that 'cat' is much closer to 'wolf' than 'aeroplane' is becomes a signal the
student can learn from. That relational structure is what a hard label 'dog'
cannot express.""")

# The T^2 correction of section 6.1, measured.
rng = np.random.default_rng(0)
student_logits = TEACHER_LOGITS + rng.normal(0, 1.5, size=4)


def kl_grad_magnitude(zt, zs, T, with_correction):
    """Gradient of the soft term w.r.t. student logits — eq:distillation-gradient."""
    p, q = softmax(zt, T), softmax(zs, T)
    grad = (q - p) / T
    if with_correction:
        grad = grad * T ** 2
    return float(np.abs(grad).sum())


print(f"\n{'T':>6} {'|grad| without T^2':>21} {'ratio to previous':>19} "
      f"{'|grad| with T^2':>17}")
prev = None
for T in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    without = kl_grad_magnitude(TEACHER_LOGITS, student_logits, T, False)
    with_c = kl_grad_magnitude(TEACHER_LOGITS, student_logits, T, True)
    ratio = f"{prev / without:.2f}x" if prev else "-"
    print(f"{T:>6.0f} {without:>21.3e} {ratio:>19} {with_c:>17.4f}")
    prev = without

print("""
The ratio column is the derivation of section 6.1 as a measurement. Each
DOUBLING of T should shrink the uncorrected gradient by T^2 = 4x once the
high-temperature expansion (eq:high-temperature-expansion) is valid — and the
column converges to exactly 4.00 by T=32. At low T it is smaller, because the
expansion assumes logit gaps are small relative to T and at T=1 they are not.

That is why the T^2 factor is not cosmetic. Without it, alpha silently means
something different at every temperature, and the last column shows the
correction doing its job: the magnitude stays in the same range across six
doublings of T.""")


def distillation_loss(zt, zs, y, T, alpha):
    """Equation (eq:distillation-loss)."""
    pt, ps = softmax(zt, T), softmax(zs, T)
    soft = float((pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12))).sum())
    hard = float(-np.log(softmax(zs, 1.0)[y] + 1e-12))
    return alpha * T ** 2 * soft + (1 - alpha) * hard


print(f"\n{'alpha':>7} {'T':>5} {'total loss':>12}")
for alpha in (0.0, 0.5, 0.9, 1.0):
    for T in (1.0, 4.0):
        print(f"{alpha:>7.1f} {T:>5.1f} "
              f"{distillation_loss(TEACHER_LOGITS, student_logits, 0, T, alpha):>12.4f}")
