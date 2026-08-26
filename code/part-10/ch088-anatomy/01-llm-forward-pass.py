# -*- coding: utf-8 -*-
# Extracted from: Chapter 88 — Anatomy of an LLM: From Tokens to Logits
# Source: src/.../ch088-anatomy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A full LLM forward pass in numpy, with every intermediate shape checked."""
import numpy as np

rng = np.random.default_rng(0)

# A model small enough to inspect, with a real model's structure.
V, D, L, H, D_FF = 512, 64, 4, 4, 176      # vocab, width, layers, heads, ffn
D_HEAD = D // H
assert D % H == 0

params = {
    "embed": rng.normal(0, 0.02, (V, D)),
    "final_norm_gain": np.ones(D),
    "blocks": [],
}
for _ in range(L):
    params["blocks"].append({
        "n1_gain": np.ones(D), "n2_gain": np.ones(D),
        "wq": rng.normal(0, 0.02, (D, D)), "wk": rng.normal(0, 0.02, (D, D)),
        "wv": rng.normal(0, 0.02, (D, D)), "wo": rng.normal(0, 0.02, (D, D)),
        "w_gate": rng.normal(0, 0.02, (D, D_FF)),
        "w_up": rng.normal(0, 0.02, (D, D_FF)),
        "w_down": rng.normal(0, 0.02, (D_FF, D)),
    })


def rmsnorm(x, gain, eps=1e-6):
    """ch:dl-normalization — no mean subtraction, just scale."""
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * gain


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x):
    return x / (1 + np.exp(-x))


def attention(x, p, trace):
    T = x.shape[0]
    q = (x @ p["wq"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)   # (H, T, dh)
    k = (x @ p["wk"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)
    v = (x @ p["wv"]).reshape(T, H, D_HEAD).transpose(1, 0, 2)
    trace["q"] = q.shape

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)          # (H, T, T)
    trace["scores"] = scores.shape
    # Causal mask, ch:tf-masking-kv: position t may not see t' > t.
    mask = np.triu(np.full((T, T), -np.inf), 1)
    weights = softmax(scores + mask)
    assert np.allclose(weights.sum(-1), 1.0), "attention rows must be distributions"
    assert np.allclose(np.triu(weights, 1), 0.0), "no attention to future positions"

    out = (weights @ v).transpose(1, 0, 2).reshape(T, D)          # (T, D)
    return out @ p["wo"]


def ffn(x, p):
    """Gated FFN — three matrices, not two (shazeer2020glu)."""
    return (silu(x @ p["w_gate"]) * (x @ p["w_up"])) @ p["w_down"]


def forward(token_ids, verbose=True):
    T = len(token_ids)
    trace = {}
    h = params["embed"][token_ids]                                # (T, D)
    if verbose:
        print(f"{'stage':<28} {'shape':>14}")
        print(f"{'token ids':<28} {str((T,)):>14}")
        print(f"{'after embedding':<28} {str(h.shape):>14}")
    assert h.shape == (T, D)

    for i, p in enumerate(params["blocks"]):
        h = h + attention(rmsnorm(h, p["n1_gain"]), p, trace)     # eq:attn-sublayer
        assert h.shape == (T, D), "attention sublayer must preserve shape"
        h = h + ffn(rmsnorm(h, p["n2_gain"]), p)                  # eq:ffn-sublayer
        assert h.shape == (T, D), "ffn sublayer must preserve shape"
        if verbose and i == 0:
            print(f"{'  qkv per head':<28} {str(trace['q']):>14}")
            print(f"{'  attention scores':<28} {str(trace['scores']):>14}")
            print(f"{'after block 1':<28} {str(h.shape):>14}")

    h = rmsnorm(h, params["final_norm_gain"])                     # eq:final-norm
    if verbose:
        print(f"{'after block ' + str(L):<28} {str(h.shape):>14}")
        print(f"{'after final norm':<28} {str(h.shape):>14}")

    last = h[-1]                                                  # (D,)
    logits = last @ params["embed"].T                             # weight tying
    if verbose:
        print(f"{'last position only':<28} {str(last.shape):>14}")
        print(f"{'logits':<28} {str(logits.shape):>14}")
    assert logits.shape == (V,)
    return logits


tokens = rng.integers(0, V, size=12)
logits = forward(tokens)
probs = softmax(logits)

print(f"\nlogit range        : [{logits.min():+.3f}, {logits.max():+.3f}]")
print(f"probabilities sum  : {probs.sum():.6f}")
print(f"top-5 token ids    : {np.argsort(-probs)[:5].tolist()}")
print(f"entropy            : {-(probs * np.log(probs + 1e-12)).sum():.4f} nats")
print(f"uniform entropy    : {np.log(V):.4f} nats  (untrained model, so close)")

# The shape invariance of eq:residual-stream-sum, verified across lengths.
print(f"\n{'sequence length':>16} {'logits shape':>14}")
for T in (1, 5, 40):
    lg = forward(rng.integers(0, V, size=T), verbose=False)
    print(f"{T:>16} {str(lg.shape):>14}")
print("\nThe logit vector is (|V|,) whatever the input length — the model maps "
      "any sequence to one distribution over the next token, which is "
      "equation (eq:llm-as-function).")
