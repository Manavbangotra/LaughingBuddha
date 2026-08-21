# Extracted from: Chapter 66 — Token Embeddings and the Unembedding Matrix
# Source: src/.../ch066-embeddings.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The embedding table, weight tying, and the parameter arithmetic that
decides it.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 4.2: a lookup, not a multiply ----------------------------------
print("=" * 72)
print("an embedding is an indexed read, not a matrix product")
print("=" * 72)
import time

V, d, n = 50000, 768, 4096
E = rng.normal(0, 0.02, (V, d))
ids = rng.integers(0, V, n)

t0 = time.perf_counter()
out_lookup = E[ids]
t_lookup = time.perf_counter() - t0

oh = np.zeros((n, V), dtype=np.float32)
oh[np.arange(n), ids] = 1.0
t0 = time.perf_counter()
out_matmul = oh @ E.astype(np.float32)
t_matmul = time.perf_counter() - t0

print(f"vocabulary {V:,}, width {d}, {n:,} tokens\n")
print(f"  indexed read      {t_lookup * 1e3:>9.3f} ms")
print(f"  one-hot @ E       {t_matmul * 1e3:>9.3f} ms   "
      f"({t_matmul / t_lookup:.0f}x slower)")
print(f"  results agree     max |diff| = "
      f"{np.abs(out_lookup - out_matmul).max():.3e}")
print(f"  wasted multiplies {2 * n * V * d / 1e9:.1f} GFLOP to read "
      f"{n * d * 8 / 1e6:.1f} MB")

print("\nSame answer, and the matmul does billions of operations to produce")
print("what a gather produces with none. Every framework implements the")
print("gather; the one-hot formulation is a notational device.")

# --- section 6.3: the gradient is sparse ------------------------------------
print("\n" + "=" * 72)
print("the embedding gradient is sparse (eq. 66.7)")
print("=" * 72)
print(f"{'batch tokens':>14} {'distinct rows':>15} {'fraction of V':>15} "
      f"{'zero rows':>12}")
for n_ in (256, 1024, 4096, 16384, 65536):
    b = rng.integers(0, V, n_)
    distinct = len(np.unique(b))
    print(f"{n_:>14,} {distinct:>15,} {distinct / V:>15.4f} "
          f"{V - distinct:>12,}")

print("\nEven a batch of 65,536 tokens leaves most of a 50,000-row table")
print("untouched, and a realistic batch touches a few per cent. So the")
print("embedding gradient is mostly zeros, by construction.")

# --- what that does to Adam -------------------------------------------------
print("\n" + "=" * 72)
print("what sparsity does to Adam (section 6.3)")
print("=" * 72)
print("Adam's v decays by beta_2 EVERY step, whether or not the parameter")
print("got a gradient. Track one row that appears every k steps.\n")


def adam_step_sizes(period, steps=4000, b1=0.9, b2=0.999, lr=1e-3, g=1.0):
    """Return the update magnitude on the steps where the row appears."""
    m = v = 0.0
    sizes = []
    for t in range(1, steps + 1):
        grad = g if (t % period == 0) else 0.0
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad * grad
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        if grad != 0.0 and t > steps // 2:
            sizes.append(lr * abs(mh) / (np.sqrt(vh) + 1e-8))
    return float(np.mean(sizes)) if sizes else float("nan")


print(f"{'appears every':>15} {'appearances/1000 steps':>24} "
      f"{'mean |update| when seen':>26} {'vs every-step':>14}")
base = adam_step_sizes(1)
for period in (1, 10, 100, 500, 1000):
    s = adam_step_sizes(period)
    print(f"{period:>15} {1000 / period:>24.1f} {s:>26.3e} "
          f"{s / base:>13.1f}x")

print("\nThe effect is real and it is smaller than the mechanism might")
print("suggest: a row seen every thousandth step takes an update about two")
print("and a half times larger than one seen every step, from the identical")
print("gradient. Between appearances v decays toward zero and Adam divides")
print("by its square root, but the bias correction 1/(1-b2^t) partly")
print("compensates, which is why the factor is not the hundreds the naive")
print("reading would predict.")
print("\nThe direction is what matters: rare tokens take systematically")
print("BIGGER steps than common ones, which is the opposite of what anyone")
print("would choose, and it is eq. 66.7's sparsity meeting Chapter 54's")
print("dense optimiser.")
print("\nA SPARSE optimiser implementation, which updates only the touched")
print("rows, does not decay v for untouched ones and removes the effect")
print("entirely. It is a correctness fix disguised as an efficiency")
print("optimisation, and frameworks disagree about the default.")

# --- section 5.2: the parameter accounting ----------------------------------
print("\n" + "=" * 72)
print("where the parameters are (eqs. 66.3-66.5)")
print("=" * 72)
MODELS = [
    ("GPT-2 small",  50257, 12, 768),
    ("GPT-2 XL",     50257, 48, 1600),
    ("7B-class",    32000, 32, 4096),
    ("70B-class",  128000, 80, 8192),
]
print(f"{'model':<14} {'V':>8} {'L':>4} {'d':>6} {'embed M':>9} "
      f"{'blocks M':>10} {'embed %':>9} {'tying saves':>12}")
for name, V_, L_, d_ in MODELS:
    emb = 2 * V_ * d_
    blk = 12 * L_ * d_ * d_
    print(f"{name:<14} {V_:>8,} {L_:>4} {d_:>6} {emb / 1e6:>9.1f} "
          f"{blk / 1e6:>10.1f} {emb / (emb + blk):>8.1%} "
          f"{(V_ * d_) / (emb + blk):>11.1%}")

print("\nRead the last two columns together. At GPT-2-small scale the")
print("embeddings are nearly HALF the model and tying saves a quarter of")
print("all parameters; at 70B scale they are three per cent and tying saves")
print("under two.")
print("\nThat is eq. 66.9, and it is the whole explanation for why small")
print("models tie and large ones mostly do not. The BENEFIT falls by more")
print("than an order of magnitude across that range; the CONSTRAINT — one")
print("matrix serving two roles at two different residual-stream scales —")
print("does not get any milder.")

# --- section 6.4: the unembedding is expensive ------------------------------
print("\n" + "=" * 72)
print("the unembedding costs more than a transformer block (eq. 66.10)")
print("=" * 72)
print(f"{'model':<14} {'unembed GFLOP/tok':>19} {'per block':>12} "
      f"{'unembed = N blocks':>20} {'of L':>6}")
for name, V_, L_, d_ in MODELS:
    un = 2 * d_ * V_
    blk = 24 * d_ * d_
    print(f"{name:<14} {un / 1e9:>19.4f} {blk / 1e9:>12.4f} "
          f"{un / blk:>19.1f} {L_:>6}")

print(f"\nEq. 66.10 says the crossover is at V = 12d:")
for name, V_, L_, d_ in MODELS:
    print(f"  {name:<14} 12d = {12 * d_:>7,}   actual V = {V_:>7,}   "
          f"{'ABOVE' if V_ > 12 * d_ else 'below'}")

print("\nThree of the four are above the crossover, so their output")
print("projection costs more arithmetic than a transformer block — for")
print("GPT-2 small it is worth about five and a half of its twelve.")
print("\nThe 7B row is below it, and the reason is instructive: that")
print("configuration pairs a comparatively small 32,000-token vocabulary")
print("with a wide d = 4096, so 12d exceeds V. Modern models with 128k")
print("vocabularies are back above the line even at large d, which is the")
print("70B row.")
print("\nSo the rule is not 'always above'; it is that the crossover sits")
print("at V = 12d and vocabularies have been growing faster than widths.")
print("\nThat is why the vocabulary size is a compute decision and not only")
print("a tokenisation one, and why large-scale training shards the logits")
print("and computes the loss without ever gathering them.")
