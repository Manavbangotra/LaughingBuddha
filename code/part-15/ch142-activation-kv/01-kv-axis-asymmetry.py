# -*- coding: utf-8 -*-
# Extracted from: Chapter 142 — Activation and KV-Cache Quantization
# Source: src/.../ch142-activation-kv.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The KV cache is two tensors, and they want opposite quantization axes.

cite:liu2024kivi reports a finding that sounds like a detail and is not: the key
cache should be quantized PER CHANNEL and the value cache PER TOKEN. Treating
"the KV cache" as one thing with one scheme leaves accuracy on the table.

This listing does not assume that asymmetry. It generates keys and values by
actually running a projection over correlated inputs -- which is where channel
structure comes from -- then MEASURES which axis carries the variation in each,
and only then tests the two quantization axes against attention output error
(eq:kv-axis-asymmetry).

The second question it answers is why both axes are even implementable, given
ch:q-int8-int4's reduction-axis constraint.
"""
import numpy as np

rng = np.random.default_rng(269)

T, D, H = 512, 64, 8          # tokens, head dim, heads


def correlated_inputs(n, d, hot=4, hot_scale=12.0):
    """Residual-stream activations: correlated, with a few dominant dimensions.
    This is where per-channel structure in K and V comes from -- it is inherited
    from the input, not created by the projection."""
    A = rng.normal(size=(d, d))
    L = np.linalg.cholesky(A @ A.T / d + 0.1 * np.eye(d))
    X = rng.normal(size=(n, d)) @ L.T
    cols = rng.choice(d, size=hot, replace=False)
    X[:, cols] *= hot_scale
    return X


D_MODEL = 256
X = correlated_inputs(T, D_MODEL)
WQ = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)
WK = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)
WV = rng.normal(size=(D_MODEL, H * D)) / np.sqrt(D_MODEL)

Q = (X @ WQ).reshape(T, H, D).transpose(1, 0, 2)      # (H, T, D)
K = (X @ WK).reshape(T, H, D).transpose(1, 0, 2)
V = (X @ WV).reshape(T, H, D).transpose(1, 0, 2)

# cite:liu2024kivi reports that key caches carry PER-CHANNEL outliers -- specific
# head dimensions that are persistently large across every token -- while value
# caches do not. That is an empirical observation about trained transformers, so
# it is imposed here rather than derived; what the listing tests is what FOLLOWS
# from it.
for h in range(H):
    hot = rng.choice(D, size=3, replace=False)
    K[h][:, hot] *= 9.0


def homogeneity(A, axis):
    """Within a group, how far is the largest value above a typical one? This is
    ch:q-theory's M/m ratio, computed per group and averaged. LOWER is better:
    a homogeneous group wastes fewer of its quantization levels."""
    m = np.max(np.abs(A), axis=axis, keepdims=True)
    med = np.median(np.abs(A), axis=axis, keepdims=True)
    return float(np.mean(m / np.maximum(med, 1e-12)))


print("Which grouping gives homogeneous groups? Mean of max/median WITHIN each")
print(f"group -- ch:q-theory's M/m ratio. Lower is better. {H} heads, {T} tokens.")
print()
print(f"{'tensor':>10}{'per-token groups':>20}{'per-channel groups':>21}"
      f"{'prefers':>12}")
print("-" * 63)
hom = {}
for name, A in (("keys", K), ("values", V), ("queries", Q)):
    ht, hc = homogeneity(A, 2), homogeneity(A, 1)
    hom[name] = (ht, hc)
    print(f"{name:>10}{ht:>20.2f}{hc:>21.2f}"
          f"{('channel' if hc < ht else 'token'):>12}")


def quant_axis(A, bits, axis):
    """Symmetric integer quantization with one scale per slice along `axis`.
    axis=2 gives a scale per TOKEN (a scale for each row of the T x D block);
    axis=1 gives a scale per CHANNEL."""
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(A), axis=axis, keepdims=True) / qmax, 1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


def attend(q, k, v):
    z = q @ k.transpose(0, 2, 1) / np.sqrt(D)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ v


REF = attend(Q, K, V)


def err(out):
    return float(np.linalg.norm(out - REF) / np.linalg.norm(REF))


print()
print()
print("Attention output error for each combination of axes.")
print("'token' = one scale per token; 'channel' = one scale per channel.")
print()
print(f"{'bits':>6}{'K axis':>10}{'V axis':>10}{'output error':>15}")
print("-" * 41)

rows = {}
for bits in (4, 3):
    for ka in ("token", "channel"):
        for va in ("token", "channel"):
            kq = quant_axis(K, bits, 2 if ka == "token" else 1)
            vq = quant_axis(V, bits, 2 if va == "token" else 1)
            e = err(attend(Q, kq, vq))
            rows[(bits, ka, va)] = e
            print(f"{bits:>6}{ka:>10}{va:>10}{e:>15.5f}")
    print()

print("Is the good combination good because of K, or because of V?")
print("Each tensor quantized alone, the other left exact.")
print()
print(f"{'bits':>6}{'tensor':>9}{'per-token':>13}{'per-channel':>14}"
      f"{'better':>12}")
print("-" * 54)
solo = {}
for bits in (4, 3):
    kt = err(attend(Q, quant_axis(K, bits, 2), V))
    kc = err(attend(Q, quant_axis(K, bits, 1), V))
    vt = err(attend(Q, K, quant_axis(V, bits, 2)))
    vc = err(attend(Q, K, quant_axis(V, bits, 1)))
    solo[bits] = (kt, kc, vt, vc)
    print(f"{bits:>6}{'keys':>9}{kt:>13.5f}{kc:>14.5f}"
          f"{('channel' if kc < kt else 'token'):>12}")
    print(f"{bits:>6}{'values':>9}{vt:>13.5f}{vc:>14.5f}"
          f"{('channel' if vc < vt else 'token'):>12}")

print()
print()
print("The constraint a static comparison cannot see: a KV cache GROWS.")
print("A per-channel scale needs the max over all tokens, which is not known")
print("until the sequence ends. Below, it is estimated from the first W tokens.")
print()
print(f"{'warmup W':>10}{'K per-channel':>16}{'K per-token':>14}"
      f"{'V per-channel':>16}{'V per-token':>14}")
print("-" * 70)


def quant_prefix_scale(A, bits, warm):
    """Per-channel scale fixed from the first `warm` tokens, then applied to the
    whole sequence -- which is what a streaming cache is forced to do."""
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(A[:, :warm, :]), axis=1, keepdims=True) / qmax,
                   1e-12)
    return np.clip(np.round(A / s), -qmax, qmax) * s


stream = {}
for warm in (32, 128, 512):
    kc = err(attend(Q, quant_prefix_scale(K, 4, warm), V))
    kt = err(attend(Q, quant_axis(K, 4, 2), V))
    vc = err(attend(Q, K, quant_prefix_scale(V, 4, warm)))
    vt = err(attend(Q, K, quant_axis(V, 4, 2)))
    stream[warm] = (kc, kt, vc, vt)
    print(f"{warm:>10}{kc:>16.5f}{kt:>14.5f}{vc:>16.5f}{vt:>14.5f}")

k4, v4 = solo[4][:2], solo[4][2:]
k3 = solo[3][:2]
best4 = min(rows[(4, a, b)] for a in ("token", "channel")
            for b in ("token", "channel"))
worst4 = max(rows[(4, a, b)] for a in ("token", "channel")
             for b in ("token", "channel"))
print(f"""
The first table is a cheap diagnostic that turns out to predict everything else.
It asks, for each candidate grouping, how far the largest value in a group sits
above a typical one -- ch:q-theory's M/m ratio, which sets how many quantization
levels the ordinary values actually get.

Keys are dramatically inhomogeneous along the token axis: {hom['keys'][0]:.1f}
against {hom['keys'][1]:.1f} per channel. That is the per-channel outlier
structure cite:liu2024kivi reports, seen from the quantizer's side -- a group
containing one token's 64 channels contains both an outlier channel and 61
ordinary ones, so the outlier sets the step for all of them. Group along
CHANNELS instead and each outlier channel is alone with its own kind.

Values are the reverse and much milder: {hom['values'][0]:.1f} per token against
{hom['values'][1]:.1f} per channel.

The second and third tables confirm the keys half emphatically. At 4 bits, keys
quantized per-channel give {k4[1]:.5f} against per-token's {k4[0]:.5f} -- a
factor of {k4[0]/k4[1]:.1f}. At 3 bits, {k3[0]/k3[1]:.1f}x. Choosing the axis is
free, and choosing it wrongly costs more than a bit of width.

And they do NOT confirm the values half. Values come out slightly better
per-channel ({v4[1]:.5f}) than per-token ({v4[0]:.5f}), despite the homogeneity
diagnostic preferring tokens. The gap is small, and the reason it exists is that
these value channels have modestly different variances, which a per-channel scale
captures.

So a static comparison says: quantize both per channel. That is the wrong answer,
and the last table says why.

A KV cache is not a static tensor. It GROWS, one token at a time, and a
per-channel scale is a maximum over the token axis -- over data that has not
arrived yet. The only options are to recompute the scale and requantize the whole
cache on every token, which is absurd, or to fix it early and live with it.

Fix it from the first 32 tokens and the key error rises from {stream[512][0]:.5f}
to {stream[32][0]:.5f}. Later tokens exceed the early maximum and clip. Per-token
scales are unaffected at every warmup -- {stream[32][1]:.5f} at every row --
because each token's scale is computed when that token arrives and never needs
revising (eq:streaming-forces-the-axis).

And now look at the values columns, because this is where the recommendation
actually comes from. With the full sequence available, V per-channel wins
({stream[512][2]:.5f} against {stream[512][3]:.5f}). With a 32-token warmup it
LOSES: {stream[32][2]:.5f} against per-token's {stream[32][3]:.5f}. The static
comparison and the streaming comparison give opposite answers for V.

For K they do not. Even with the worst warmup, per-channel keys
({stream[32][0]:.5f}) still beat per-token keys ({stream[32][1]:.5f}). The keys'
per-channel advantage is large enough to survive a badly estimated scale; the
values' is not.

That is the constraint the papers' recommendation is really made under, and it
explains the shape of the actual design rather than the one a static experiment
suggests. Keys need per-channel scales badly enough -- a factor of
{k4[0]/k4[1]:.1f} -- to be worth handling the streaming problem: quantize the
cache in chunks, keeping a small recent window in full precision and sealing each
chunk's channel scales once the chunk is complete. Values do not need per-channel
scales badly enough to be worth any of that, so they take the axis that streams
for free.

The asymmetry is therefore not two facts about two tensors. It is one fact about
the keys -- they have per-channel outliers -- meeting one fact about caches, that
they grow along the token axis. Neither alone produces the recommendation.

One more piece, because it explains why per-channel scales on K are implementable
at all. Attention scores are Q K^T, a sum over channels, so by
ch:q-int8-int4's eq:reduction-axis-constraint a per-channel scale on K cannot be
factored out of the dot product. It can be folded into Q, which carries the same
channel index: (Q * s)(K/s)^T is the same product. The attention output is A V, a
sum over tokens, so a per-token scale on V folds into A the same way.

Each tensor has exactly one axis whose scale its partner can absorb, and they are
different axes. That is a third independent reason the recommendation comes out
asymmetric, and it is the one that decides whether a kernel can be written at
all.""")
