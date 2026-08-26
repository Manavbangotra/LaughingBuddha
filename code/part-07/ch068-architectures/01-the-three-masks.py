# -*- coding: utf-8 -*-
# Extracted from: Chapter 68 — Encoder, Decoder, and Encoder–Decoder Transformers
# Source: src/.../ch068-architectures.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Three architectures, one boolean matrix of difference."""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_mask(T, kind, prefix=0):
    """Eq. 68.2."""
    if kind == "encoder":
        return np.ones((T, T), dtype=bool)
    if kind == "decoder":
        return np.tril(np.ones((T, T), dtype=bool))
    if kind == "prefix":
        m = np.tril(np.ones((T, T), dtype=bool))
        m[:, :prefix] = True
        return m
    raise ValueError(kind)


print("=" * 72)
print("the three architectures differ in one boolean matrix (eq. 68.2)")
print("=" * 72)
T = 8
for kind, kw in (("encoder", {}), ("decoder", {}), ("prefix", {"prefix": 3})):
    m = make_mask(T, kind, **kw)
    print(f"\n{kind}" + (f" (P = {kw['prefix']})" if kw else "") + ":")
    for i in range(T):
        print("   " + " ".join("#" if m[i, j] else "." for j in range(T)))
    print(f"   positions attended, per query: "
          f"{m.sum(1).tolist()}")

print("\nEverything else — the blocks, the embeddings, the positions, the")
print("parameters — is identical. Section 6.4 makes that precise: the mask")
print("enters only as an additive term before the softmax, so the SAME")
print("weights can be run under any of these.")

# --- section 6.5: the causal mask supplies position -------------------------
print("\n" + "=" * 72)
print("the causal mask carries positional information (section 6.5)")
print("=" * 72)
print("Under a causal mask, query i attends over exactly i+1 positions. The")
print("softmax normalises over that count, so a model can read position off")
print("the attention denominator with NO positional encoding.\n")
d = 32
X = rng.normal(size=(T, d))
Wq = rng.normal(0, 1 / np.sqrt(d), (d, d))
Wk = rng.normal(0, 1 / np.sqrt(d), (d, d))
S = (X @ Wq) @ (X @ Wk).T / np.sqrt(d)

print(f"{'query i':>9} {'keys visible':>14} {'max attention weight':>22} "
      f"{'entropy':>9}")
for kind in ("encoder", "decoder"):
    m = make_mask(T, kind)
    A = softmax(np.where(m, S, -1e9))
    print(f"  {kind}:")
    for i in (0, 1, 3, 7):
        ent = float(-(A[i][m[i]] * np.log(A[i][m[i]] + 1e-12)).sum())
        print(f"{i:>9} {int(m[i].sum()):>14} {A[i].max():>22.4f} "
              f"{ent:>9.4f}")

print("\nUnder the causal mask the entropy grows with the position, because")
print("more keys are visible. That quantity is a monotone function of i and")
print("the model can use it.")
print("\nThat is why decoder-only transformers with no positional encoding")
print("work at all, and it means the causal mask is doing double duty:")
print("enforcing eq. 68.6's factorisation AND breaking the permutation")
print("symmetry of eq. 65.1. An ENCODER has neither, which is why")
print("bidirectional models depend on their positional scheme absolutely.")

# --- section 7.2: the mask value -------------------------------------------
print("\n" + "=" * 72)
print("the mask value, and the bug it causes (section 7.2)")
print("=" * 72)
scores = np.array([2.0, 1.0, 3.0, 0.5])
mask = np.array([True, True, False, False])
print(f"scores {scores}, mask {mask}\n")
print(f"{'mask value':>14} {'weight on the MASKED entries':>32} "
      f"{'leaked?':>9}")
for mv in (-10.0, -50.0, -1e4, -1e9):
    A = softmax(np.where(mask, scores, mv))
    leak = float(A[~mask].sum())
    print(f"{mv:>14.0e} {leak:>32.3e} {('YES' if leak > 1e-12 else 'no'):>9}")

print("\nA mask value that is large but not large enough leaks attention")
print("across the boundary. At -10 the leak is percent-scale: the model can")
print("see the future, slightly, and it will use it. Training looks fine and")
print("generation is subtly wrong.")

print("\nAnd the other failure — a row that is ENTIRELY masked:")
full_mask = np.zeros(4, dtype=bool)
for mv in (-1e9, -np.inf):
    with np.errstate(invalid="ignore"):
        A = softmax(np.where(full_mask, scores, mv))
    print(f"  mask value {mv:>10}: result = {A}, "
          f"finite = {bool(np.all(np.isfinite(A)))}")

print("\nWith a finite mask value a fully-masked row gives a uniform")
print("distribution — wrong, and silent. With -inf it gives nan, which at")
print("least announces itself. Fully-masked rows occur whenever a sequence")
print("is all padding, so this is not hypothetical.")
print("\nIn fp16 the situation is worse: the largest representable magnitude")
print(f"is 65504, so a mask value of -1e9 becomes -inf on conversion and")
print("the nan appears without anyone having written -inf.")

# --- the objectives ---------------------------------------------------------
print("\n" + "=" * 72)
print("supervised positions per forward pass (eq. 68.9)")
print("=" * 72)
print(f"{'objective':<26} {'supervised / T':>16} {'FLOPs':>10} "
      f"{'signal per FLOP':>18}")
for name, frac in (("masked LM (15%)", 0.15), ("masked LM (30%)", 0.30),
                   ("next-token", 1.00), ("prefix-LM (50% suffix)", 0.50)):
    print(f"{name:<26} {frac:>16.2f} {'same':>10} {frac:>18.2f}")

print("\nThe forward and backward passes cost the same regardless of how")
print("many positions carry a loss term — the compute is O(L d^2 T) either")
print("way. So the last column is the ratio that matters, and next-token")
print("prediction extracts about seven times the supervision per unit of")
print("compute that BERT-style masking does.")
print("\nWhy not just mask more? Because masking removes the context the")
print("other predictions depend on. At 100% masking there is no context at")
print("all and the task is unlearnable, so there is a genuine optimum well")
print("below 1 — which is exactly the constraint next-token prediction does")
print("not have.")
