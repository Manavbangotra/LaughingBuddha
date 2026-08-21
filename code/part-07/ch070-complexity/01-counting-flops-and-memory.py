# Extracted from: Chapter 70 — Computational and Memory Complexity of Attention
# Source: src/.../ch070-complexity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Every cost in a transformer, counted (eqs. 70.1-70.9)."""
import numpy as np


class Config:
    def __init__(self, name, V, L, d, h, d_ff=None, g=None, b=2):
        self.name, self.V, self.L, self.d, self.h = name, V, L, d, h
        self.d_ff = d_ff or 4 * d
        self.g = g if g is not None else h
        self.dk = d // h
        self.b = b

    def params(self):
        emb = 2 * self.V * self.d
        blocks = self.L * (4 * self.d ** 2 + 2 * self.d * self.d_ff)
        return emb + blocks, emb, blocks

    def fwd_flops_per_token(self, T):
        """Eq. 70.2."""
        N, emb, blocks = self.params()
        param_flops = 2 * (blocks + self.V * self.d)   # blocks + unembedding
        attn_flops = 4 * self.L * T * self.d
        return param_flops + attn_flops, param_flops, attn_flops

    def act_bytes_per_token(self, T):
        """Eq. 70.6-70.7, alpha = 10."""
        per_layer = self.b * (10 * self.d + self.d_ff + self.h * T)
        return self.L * per_layer

    def kv_bytes_per_token(self):
        return 2 * self.b * self.L * self.g * self.dk


MODELS = [
    Config("GPT-2 small", 50257, 12, 768, 12),
    Config("1.3B", 50257, 24, 2048, 16),
    Config("7B (GQA 8)", 32000, 32, 4096, 32, d_ff=11008, g=8),
    Config("70B (GQA 8)", 128000, 80, 8192, 64, d_ff=28672, g=8),
]

print("=" * 72)
print("parameters (eq. 70.1)")
print("=" * 72)
print(f"{'model':<14} {'total':>12} {'embeddings':>12} {'blocks':>12} "
      f"{'embed %':>9} {'12Ld^2 approx':>15} {'error':>8}")
for c in MODELS:
    N, emb, blk = c.params()
    approx = 12 * c.L * c.d ** 2
    print(f"{c.name:<14} {N / 1e9:>11.2f}B {emb / 1e9:>11.3f}B "
          f"{blk / 1e9:>11.2f}B {emb / N:>9.1%} {approx / 1e9:>14.2f}B "
          f"{abs(approx - N) / N:>7.1%}")

print("\nThe 12Ld^2 approximation is good for large models and poor for")
print("small ones, and the embed-% column says why — it drops the")
print("embeddings, which are a third of GPT-2 small and one per cent of a")
print("70B model. That is Chapter 66's crossover appearing in the FLOP")
print("accounting.")
print("\nNote also that the 7B and 70B rows use a gated feed-forward block,")
print("so d_ff is about 8d/3 across three matrices rather than 4d across")
print("two — the 12Ld^2 shorthand still lands close because eq. 67.6 was")
print("chosen to keep the parameter count matched.")

# --- section 6.3: where the quadratic term takes over -----------------------
print("\n" + "=" * 72)
print("where attention's FLOPs overtake everything else (eq. 70.10)")
print("=" * 72)
print(f"{'model':<14} {'6d (predicted)':>16} " +
      " ".join(f"{f'T={T}':>14}" for T in (2048, 8192, 32768, 131072)))
print(f"{'':<14} {'':>16} " +
      " ".join(f"{'attn % of FLOPs':>14}" for _ in range(4)))
for c in MODELS:
    row = []
    for T in (2048, 8192, 32768, 131072):
        tot, par, att = c.fwd_flops_per_token(T)
        row.append(att / tot)
    print(f"{c.name:<14} {6 * c.d:>16,} " +
          " ".join(f"{x:>14.1%}" for x in row))

print("\nEq. 70.10 says the crossover — where attention is half the FLOPs —")
print("is at T = 6d. The columns confirm it: attention is a minority of the")
print("arithmetic at every context below that and a majority above.")
print("\nFor a 70B model that threshold is about 49,000 tokens. So at any")
print("ordinary context length a transformer is dominated by matrix")
print("multiplications against WEIGHTS, not by attention — which is the")
print("opposite of the usual framing.")

# --- but the memory crossover is much earlier -------------------------------
print("\n" + "=" * 72)
print("...but attention's MEMORY overtakes much earlier (eq. 70.11)")
print("=" * 72)
print(f"{'model':<14} {'14d/h (predicted)':>19} " +
      " ".join(f"{f'T={T}':>14} " for T in (2048, 8192, 32768)))
print(f"{'':<14} {'':>19} " +
      " ".join(f"{'attn % of act':>15}" for _ in range(3)))
for c in MODELS:
    row = []
    for T in (2048, 8192, 32768):
        per_layer_other = c.b * (10 * c.d + c.d_ff)
        per_layer_attn = c.b * c.h * T
        row.append(per_layer_attn / (per_layer_other + per_layer_attn))
    print(f"{c.name:<14} {14 * c.d // c.h:>19,} " +
          " ".join(f"{x:>15.1%}" for x in row))

print("\nEq. 70.11 puts this crossover at 14d/h, which is roughly FOURTEEN")
print("TIMES EARLIER than the FLOP crossover for a typical head count.")
print("\nThat gap is the entire reason 'attention is quadratic' is")
print("confusing. Memory goes quadratic around two thousand tokens; FLOPs")
print("only around fifty thousand. In between — which is where most models")
print("operate — attention is a MEMORY problem and not a compute problem.")
print("\nAnd that is precisely why FlashAttention's title says IO-awareness")
print("and says nothing about arithmetic: it removes the memory term and")
print("performs the same FLOPs.")

# --- the absolute numbers ---------------------------------------------------
print("\n" + "=" * 72)
print("the attention matrix, in absolute terms (section 4.4)")
print("=" * 72)
print(f"{'model':<14} {'batch':>6} " +
      " ".join(f"{f'T={T}':>13}" for T in (2048, 8192, 32768)))
for c in MODELS[2:]:
    for B in (1, 8):
        row = [c.b * B * c.L * c.h * T * T / 1e9
               for T in (2048, 8192, 32768)]
        print(f"{c.name:<14} {B:>6} " +
              " ".join(f"{x:>12,.0f}G" for x in row))

print("\nThose are the attention matrices alone, in gigabytes, if they are")
print("materialised. At a 32k context the numbers are in the hundreds of")
print("terabytes — not a memory-pressure problem, an impossibility.")
print("\nFlashAttention makes them zero by never writing the matrix out.")
print("That is the single largest term in the training-memory accounting")
print("above about two thousand tokens, and removing it is what made long")
print("contexts feasible at all.")
