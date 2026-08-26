# -*- coding: utf-8 -*-
# Extracted from: Chapter 71 — Efficient Attention: FlashAttention, GQA/MQA, Sparse and Linear Variants
# Source: src/.../ch071-efficient-attention.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Choosing from the cost accounting rather than from a list."""
import numpy as np


class Model:
    def __init__(self, name, N, L, d, h, g, dk, dff):
        self.name, self.N, self.L, self.d = name, N, L, d
        self.h, self.g, self.dk, self.dff = h, g, dk, dff


M7 = Model("7B", 7e9, 32, 4096, 32, 8, 128, 11008)
M70 = Model("70B", 7e10, 80, 8192, 64, 8, 128, 28672)


def costs(m, B, T, b=2, flash=True, window=None):
    """Every row of table 70.1, for one configuration."""
    Teff = min(window, T) if window else T
    return {
        "param FLOPs": 2 * m.N * B * T,
        "attn FLOPs": 4 * m.L * T * Teff * m.d * B,
        "attn memory GB": (0.0 if flash
                           else b * B * m.L * m.h * T * Teff / 1e9),
        "act memory GB": b * B * T * m.L * (10 * m.d + m.dff) / 1e9,
        "opt state GB": 16 * m.N / 1e9,
        "KV cache GB": 2 * b * m.L * m.g * m.dk * Teff * B / 1e9,
    }


print("=" * 72)
print("which term is binding? (table 70.1, instantiated)")
print("=" * 72)
for m in (M7, M70):
    print(f"\n{m.name}, training, B=4, bf16:")
    print(f"  {'T':>7} {'attn mem (no flash)':>21} {'attn mem (flash)':>18} "
          f"{'act mem':>10} {'opt state':>11} {'BINDING':>18}")
    for T in (2048, 8192, 32768):
        c_no = costs(m, 4, T, flash=False)
        c_fl = costs(m, 4, T, flash=True)
        tot = {k: v for k, v in c_fl.items() if "GB" in k}
        binding = max(tot, key=tot.get)
        print(f"  {T:>7} {c_no['attn memory GB']:>20,.0f}G "
              f"{c_fl['attn memory GB']:>17,.0f}G "
              f"{c_fl['act memory GB']:>9,.0f}G "
              f"{c_fl['opt state GB']:>10,.0f}G {binding:>18}")

print("\nWithout FlashAttention the attention matrix dominates everything at")
print("every length past a couple of thousand tokens — by orders of")
print("magnitude, into the terabytes. With it, that term is zero and the")
print("binding constraint becomes something else entirely.")
print("\nThat is why the decision table in section 5.6 has FlashAttention as")
print("an unconditional first row: until it is applied, no other")
print("optimisation is addressing the actual bottleneck.")

# --- what a window buys, and costs ------------------------------------------
print("\n" + "=" * 72)
print("what a sliding window buys, with FlashAttention already applied")
print("=" * 72)
for m in (M70,):
    print(f"{m.name}, serving, B=32, bf16:\n")
    print(f"  {'T':>7} {'window':>9} {'attn TFLOPs':>13} {'KV cache':>11} "
          f"{'effective context (Lw)':>24}")
    for T in (32768, 131072):
        for w in (None, 4096, 1024):
            c = costs(m, 32, T, window=w)
            eff = "unbounded" if w is None else f"{m.L * w:,}"
            flag = "" if w is None or m.L * w >= T else "  << T!"
            print(f"  {T:>7,} {str(w or 'full'):>9} "
                  f"{c['attn FLOPs'] / 1e12:>12,.0f}T "
                  f"{c['KV cache GB']:>10,.0f}G {eff:>24}{flag}")

print("\nThe last column is eq. 71.10 and it is the check people skip. A")
print("1024-token window on an 80-layer model caps the effective context at")
print("81,920 tokens — fine at 32k and NOT fine at 131k, where positions")
print("provably cannot interact.")
print("\nThat is decidable before training, from two integers, and it should")
print("be the first thing computed when a window is proposed.")

# --- the alternative people forget ------------------------------------------
print("\n" + "=" * 72)
print("the option that is usually skipped: retrieve instead")
print("=" * 72)
print("Attending over 128k tokens against retrieving the relevant 4k and")
print("attending over those fully.\n")
m = M70
print(f"{'approach':<32} {'prefill TFLOPs':>16} {'KV cache/user':>15} "
      f"{'path length':>13}")
for label, T, w in (("full attention, 128k", 131072, None),
                    ("window 4k, 128k context", 131072, 4096),
                    ("retrieve 4k, full attention", 4096, None)):
    c = costs(m, 1, T, window=w)
    pl = "1" if w is None else f"{int(np.ceil(T / w))}"
    print(f"{label:<32} "
          f"{(c['param FLOPs'] + c['attn FLOPs']) / 1e12:>15,.0f}T "
          f"{c['KV cache GB']:>14,.1f}G {pl:>13}")

print("\nRetrieval is cheaper than either attention variant by a wide")
print("margin, and it keeps a path length of 1 over the tokens it does")
print("attend to. What it costs is a retrieval system and the risk of")
print("retrieving the wrong 4k (Part XII).")
print("\nThat trade is an engineering decision rather than an architectural")
print("one, which is exactly why it gets left out of architecture papers —")
print("and why it is frequently the right answer anyway.")

# --- honest accounting of what is deployed ----------------------------------
print("\n" + "=" * 72)
print("what is actually deployed (section 7.5)")
print("=" * 72)
TECHNIQUES = [
    ("FlashAttention", "exact", "attn memory", "universal"),
    ("GQA", "small quality cost", "KV cache", "universal since 2023"),
    ("KV quantisation", "small quality cost", "KV cache", "common"),
    ("Sliding window", "long-range pairs", "attn FLOPs + cache",
     "used, interleaved"),
    ("SSM / hybrid", "quality, currently", "everything", "a few models"),
    ("Linear attention", "retrieval ability", "everything", "not alone"),
    ("Learned sparsity", "complexity", "attn FLOPs", "not used"),
]
print(f"{'technique':<20} {'trades':<22} {'attacks':<20} {'deployed':<22}")
for t, tr, at, dep in TECHNIQUES:
    print(f"{t:<20} {tr:<22} {at:<20} {dep:<22}")

print("\nRead the first column against the last. The techniques that are")
print("universally deployed are the ones near the top — the ones that trade")
print("nothing or almost nothing — and the ones with the largest published")
print("literature are near the bottom.")
print("\nThat is not conservatism. FlashAttention removed most of the")
print("PRESSURE that the approximate methods were built to relieve: once")
print("exact attention became cheap enough in practice, an approximation")
print("has to justify its quality cost against a much better baseline than")
print("the one it was benchmarked against.")
print("\nThe general lesson is worth more than the specific table. When an")
print("expensive operation gets a better implementation, every approximation")
print("of it has to be re-evaluated — and most of them do not survive.")
