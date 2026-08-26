# -*- coding: utf-8 -*-
# Extracted from: Chapter 65 — Positional Encoding, RoPE, and ALiBi
# Source: src/.../ch065-positional.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Length extrapolation: why RoPE fails past its training length, why ALiBi
does not, and what the scaling recipes do (eqs. 65.10-65.12).
"""
import numpy as np

rng = np.random.default_rng(1)


def rope_tables(T, dk, base=10000.0, pos_scale=1.0):
    theta = base ** (-np.arange(0, dk, 2) / dk)
    m = np.arange(T)[:, None] / pos_scale
    ang = m * theta[None, :]
    return np.cos(ang), np.sin(ang), theta


# --- section 6.3: the frequency spectrum ------------------------------------
print("=" * 72)
print("RoPE's wavelengths span four orders of magnitude (eq. 65.10)")
print("=" * 72)
dk = 128
_, _, theta = rope_tables(1, dk)
wl = 2 * np.pi / theta
print(f"head dimension {dk}, so {dk // 2} frequency pairs\n")
print(f"{'pair j':>8} {'theta_j':>12} {'wavelength':>14} "
      f"{'cycles at T=4096':>19}")
for j in (0, 8, 16, 32, 48, 63):
    print(f"{j:>8} {theta[j]:>12.3e} {wl[j]:>14.1f} {4096 / wl[j]:>19.2f}")

full = int((wl < 4096).sum())
print(f"\npairs completing at least one full cycle within T = 4096: "
      f"{full} of {dk // 2}")
print(f"pairs that never complete one: {dk // 2 - full}")

print("\nThat last number is where extrapolation breaks. A pair whose")
print("wavelength exceeds the training length has only ever been seen on")
print("part of its cycle, so the model has no calibration for the angles it")
print("will encounter at longer positions. The short-wavelength pairs have")
print("been fully exercised and are fine.")
print("\nThat asymmetry is the whole basis of NTK-aware scaling: stretch the")
print("long-wavelength pairs, which are undertrained, and leave the")
print("short-wavelength ones, which are not.")

# --- section 6.4: what each scaling recipe does -----------------------------
print("\n" + "=" * 72)
print("what the scaling recipes do to the wavelengths (eqs. 65.11-65.12)")
print("=" * 72)
s = 8.0                                    # extend 4k -> 32k
print(f"extending by a factor of s = {s:g}\n")
_, _, th_base = rope_tables(1, dk, base=10000.0)
_, _, th_pi = rope_tables(1, dk, base=10000.0, pos_scale=s)
b_ntk = 10000.0 * s ** (dk / (dk - 2))
_, _, th_ntk = rope_tables(1, dk, base=b_ntk)

print(f"NTK-aware base: 10000 -> {b_ntk:.0f}  (eq. 65.12)\n")
print(f"{'pair j':>8} {'base wavelength':>17} {'interp. stretch':>17} "
      f"{'NTK stretch':>14}")
for j in (0, 8, 16, 32, 48, 63):
    wl0 = 2 * np.pi / th_base[j]
    st_pi = (2 * np.pi / (th_pi[j] / s)) / wl0 if False else s
    st_ntk = (2 * np.pi / th_ntk[j]) / wl0
    print(f"{j:>8} {wl0:>17.1f} {s:>17.2f}x {st_ntk:>13.2f}x")

print("\nPosition interpolation stretches EVERY wavelength by s, including")
print("the shortest. Two adjacent tokens, which were separated by an angle")
print(f"of theta_0, are now separated by theta_0 / {s:g} — so the model's")
print("ability to tell token m from token m+1 degrades by exactly the scale")
print("factor. That is section 6.4, and it is why interpolation needs a")
print("fine-tune and hurts most on tasks needing precise local order.")
print("\nNTK-aware scaling stretches the longest wavelength by about s and")
print("the shortest by about 1. Local resolution is preserved and only the")
print("undertrained long-range pairs are moved, which is why it often works")
print("with no fine-tuning at all.")

# --- what happens to the score at unseen distances --------------------------
print("\n" + "=" * 72)
print("what the attention score does past the training length")
print("=" * 72)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


T_TRAIN, T_TEST = 512, 4096
q = rng.normal(size=(200, dk)) / np.sqrt(dk) ** 0.5
k = rng.normal(size=(200, dk)) / np.sqrt(dk) ** 0.5

print("Mean |score| as a function of offset, for random q and k. Inside the")
print("training range the model has calibrated against these magnitudes;")
print("outside it has not.\n")
print(f"{'offset':>9} {'in training range?':>20} {'mean |score|':>14} "
      f"{'sd of score':>13}")
cos, sin, _ = rope_tables(T_TEST + 1, dk)
for off in (1, 16, 128, 512, 1024, 4096):
    qm = apply_rope(q, cos[0:1], sin[0:1])
    kn = apply_rope(k, cos[off:off + 1], sin[off:off + 1])
    sc = (qm * kn).sum(-1) / np.sqrt(dk)
    print(f"{off:>9} {str(off <= T_TRAIN):>20} {float(np.abs(sc).mean()):>14.4f} "
          f"{float(sc.std()):>13.4f}")

print("\nFor RANDOM q and k the score statistics barely change with offset,")
print("which is worth stating plainly because it shows what the")
print("extrapolation problem is NOT. It is not that the scores blow up.")
print("\nThe problem is that the ROTATION ANGLES at large offsets are ones")
print("the trained q and k directions were never optimised against. A")
print("trained model has learned specific q-k geometries that produce")
print("useful scores at the offsets it saw, and those geometries have no")
print("reason to produce useful scores at angles outside that range.")
print("Random vectors cannot show this, because they have no learned")
print("geometry to lose — which is why the honest measurement of")
print("extrapolation needs a trained model, and section 9 uses one.")

# --- ALiBi's behaviour ------------------------------------------------------
print("\n" + "=" * 72)
print("ALiBi's decay is defined at any distance (eq. 65.13)")
print("=" * 72)
h = 8
slopes = 2.0 ** (-8.0 * np.arange(1, h + 1) / h)
print(f"{h} heads, slopes 2^(-8h'/h)\n")
print(f"{'head':>5} {'slope':>12} {'effective window 1/m':>22} " +
      " ".join(f"{f'penalty@{d_}':>13}" for d_ in (10, 100, 1000)))
for i, m in enumerate(slopes):
    pen = [f"{np.exp(-m * d_):.2e}" for d_ in (10, 100, 1000)]
    print(f"{i:>5} {m:>12.5f} {1 / m:>22.1f} " +
          " ".join(f"{p:>13}" for p in pen))

print("\nThe 'penalty' columns are the multiplicative factor eq. 65.13")
print("applies to the attention weight at that distance. Head 0 is")
print("effectively blind past a few positions; head 7 still sees a")
print("thousand.")
print("\nNothing in this table refers to the training length, which is")
print("exactly why ALiBi extrapolates: a distance of 8000 gets penalty")
print("exp(-8000m) whether or not the model has ever seen one.")
print("\nAnd that is also its limitation. The heads' scales are FIXED by the")
print("slope schedule, so a relationship at distance 5000 can only be")
print("learned by the two or three heads whose slopes permit it. RoPE lets")
print("every head attend at any distance and pays for it with the")
print("extrapolation problem above. Neither is free.")
