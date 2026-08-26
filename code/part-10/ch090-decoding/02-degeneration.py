# -*- coding: utf-8 -*-
# Extracted from: Chapter 90 — Decoding: Softmax, Temperature, Top-k, Top-p, and Beam Search
# Source: src/.../ch090-decoding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Maximising likelihood produces repetitive text. Sampling does not."""
import numpy as np

rng = np.random.default_rng(1)
V, CONTEXT = 60, 6


def make_model():
    """A toy autoregressive model with the ONE property that matters here:
    a token becomes more likely the more it has recently appeared, which is
    equation (eq:repetition-feedback)."""
    base = rng.normal(size=(V, V))          # base bigram preferences

    def logits(history):
        z = base[history[-1]].copy()
        recent = history[-CONTEXT:]
        for tok in recent:
            z[tok] += 1.1                   # self-reinforcement
        return z
    return logits


model = make_model()


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def generate(strategy, n=140, T=1.0, p=0.9, seed=0):
    g = np.random.default_rng(seed)
    hist = [int(g.integers(V))]
    for _ in range(n):
        z = model(hist)
        if strategy == "greedy":
            nxt = int(z.argmax())
        elif strategy == "sample":
            pr = softmax(z / T)
            nxt = int(g.choice(V, p=pr))
        elif strategy == "nucleus":
            pr = softmax(z / T)
            order = np.argsort(-pr)
            cum = np.cumsum(pr[order])
            keep = order[:int(np.searchsorted(cum, p) + 1)]
            q = pr[keep] / pr[keep].sum()
            nxt = int(g.choice(keep, p=q))
        hist.append(nxt)
    return hist[1:]


def repetition_rate(seq, n=3):
    """Fraction of n-grams that are repeats."""
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def distinct_ratio(seq):
    return len(set(seq)) / len(seq)


def mean_surprise(seq):
    """Equation (eq:human-surprise-gap): -log p of each chosen token."""
    hist, total = [seq[0]], 0.0
    for tok in seq[1:]:
        pr = softmax(model(hist))
        total += -np.log(pr[tok] + 1e-12)
        hist.append(tok)
    return total / (len(seq) - 1)


print(f"{'strategy':<22} {'distinct':>10} {'3-gram repeat':>15} "
      f"{'mean surprise':>15}")
results = {}
for label, kwargs in [("greedy", dict(strategy="greedy")),
                      ("sample T=0.7", dict(strategy="sample", T=0.7)),
                      ("sample T=1.0", dict(strategy="sample", T=1.0)),
                      ("nucleus T=1.0 p=0.9", dict(strategy="nucleus", T=1.0, p=0.9))]:
    seq = generate(**kwargs)
    r = (distinct_ratio(seq), repetition_rate(seq), mean_surprise(seq))
    results[label] = r
    print(f"{label:<22} {r[0]:>10.3f} {r[1]:>15.3f} {r[2]:>15.4f}")

g = results["greedy"]
s = results["sample T=1.0"]
print(f"\ngreedy repeats {g[1]:.1%} of its 3-grams; sampling at T=1 repeats "
      f"{s[1]:.1%}")
print(f"greedy's mean surprise is {g[2]:.3f} nats, sampling's is {s[2]:.3f}")
assert g[1] > s[1], "greedy must repeat more than sampling"
assert g[2] < s[2], "greedy must produce lower-surprise (higher-likelihood) text"

# Show the loop forming.
seq = generate(strategy="greedy", n=60)
print(f"\ngreedy output (token ids): {seq[:40]}")
tail = seq[-20:]
print(f"last 20 tokens          : {tail}")
print(f"distinct tokens in tail : {len(set(tail))} of 20")

print("""
This is holtzman2020's result in miniature. Greedy decoding produces text that
is MORE probable under the model — lower mean surprise, by construction, since
it takes the argmax every step — and it is degenerate: it falls into a loop and
cannot leave, because leaving requires choosing a lower-probability token.

Sampling produces higher-surprise, less probable text that does not degenerate.
The fix for degeneration is stochasticity, not a better search, and equation
(eq:repetition-feedback) is why: a maximiser is structurally unable to escape a
positive feedback loop.""")
