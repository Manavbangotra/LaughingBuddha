# -*- coding: utf-8 -*-
# Extracted from: Chapter 80 — Pretraining and Self-Supervised Objectives
# Source: src/.../ch080-pretraining.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A run with no causal structure available plateaus at the unigram entropy."""
import math
from collections import Counter

import torch
import torch.nn as nn

torch.manual_seed(0)

# Same corpus and model as the previous listing, deliberately kept independent.
SUBJ = ["the doctor", "the engineer", "the chef", "the pilot"]
VERB = {"the doctor": "examined", "the engineer": "debugged",
        "the chef": "seasoned", "the pilot": "landed"}
OBJ = {"examined": "the patient", "debugged": "the service",
       "seasoned": "the dish", "landed": "the aircraft"}
TEXT = " . ".join(f"{s} {VERB[s]} {OBJ[VERB[s]]}" for s in SUBJ * 60).split()

vocab = sorted(set(TEXT))
idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
data = torch.tensor([idx[w] for w in TEXT])
counts = Counter(TEXT)
total = sum(counts.values())
unigram_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
T, D = 8, 64


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, 4, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.nf = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x, shuffle_labels=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), 1)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h),
                         attn_mask=mask, need_weights=False)
        h = h + a
        h = h + self.ff(self.n2(h))
        return self.nf(h) @ self.tok.weight.T


def run(scramble, steps=400):
    torch.manual_seed(0)
    m = M()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    last = None
    for _ in range(steps):
        starts = torch.randint(0, len(data) - T - 1, (16,))
        x = torch.stack([data[s:s + T] for s in starts])
        y = torch.stack([data[s + 1:s + T + 1] for s in starts])
        if scramble:
            # Break the link between context and target: the model can still
            # learn the marginal token distribution and nothing more.
            y = y[torch.randperm(len(y))]
        loss = nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    return last


healthy = run(scramble=False)
broken = run(scramble=True)

def verdict(loss):
    """Report the GAP to H(X), not a binary — a broken run lands ON it, and
    may sit a hair either side because the batch marginal is not exactly the
    corpus marginal. What identifies it is the distance, not the sign."""
    gap = unigram_entropy - loss
    if gap > 0.5:
        return f"{gap:+.4f} below H(X) — using context"
    if abs(gap) <= 0.5:
        return f"{gap:+.4f} from H(X) — AT the unigram floor"
    return f"{gap:+.4f} — worse than frequencies alone"


print(f"unigram entropy H(X)          : {unigram_entropy:.4f}")
print(f"healthy run, final loss       : {healthy:.4f}   {verdict(healthy)}")
print(f"scrambled labels, final loss  : {broken:.4f}   {verdict(broken)}")

assert unigram_entropy - healthy > 0.5, "healthy run must clear H(X) decisively"
assert abs(unigram_entropy - broken) <= 0.5, "scrambled run must sit at H(X)"
print("""
The broken run does not crash, does not spike, and produces a loss curve that
falls convincingly — it simply stops at the unigram entropy, because token
frequencies are all the signal left. Watching the loss GO DOWN tells you
nothing. Watching where it stops tells you everything.""")
