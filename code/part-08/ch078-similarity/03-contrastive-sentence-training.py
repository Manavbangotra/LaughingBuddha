# -*- coding: utf-8 -*-
# Extracted from: Chapter 78 — Semantic Similarity and Sentence Embeddings
# Source: src/.../ch078-similarity.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""In-batch contrastive training — equation (eq:infonce) — and hard negatives."""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
D, N_TOPICS, DIM, TAU = 64, 8, 32, 0.07

# Synthetic sentence features: each topic has a centre, a "sentence" is that
# centre plus noise, and a positive pair is two sentences from one topic.
centres = torch.randn(N_TOPICS, D)
centres = centres / centres.norm(dim=1, keepdim=True)


def sample(topics, noise=0.12):
    return centres[topics] + noise * torch.randn(len(topics), D)


encoder = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, DIM))
opt = torch.optim.Adam(encoder.parameters(), lr=1e-2)


def embed(X):
    Z = encoder(X)
    return Z / (Z.norm(dim=1, keepdim=True) + 1e-9)


def infonce(a, b, tau=TAU):
    """Positives on the diagonal, every other column an in-batch negative."""
    S = embed(a) @ embed(b).T / tau
    return nn.functional.cross_entropy(S, torch.arange(len(a))), S


topics = torch.arange(N_TOPICS)
print(f"random-guess loss for {N_TOPICS} in-batch candidates = "
      f"log {N_TOPICS} = {math.log(N_TOPICS):.4f}")
for step in range(1, 401):
    loss, _ = infonce(sample(topics), sample(topics))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in (1, 100, 200, 300, 400):
        print(f"step {step:>4}: InfoNCE loss {loss.item():.4f}")

# Did the space separate by topic?
with torch.no_grad():
    labels = topics.repeat_interleave(6)
    Z = embed(sample(labels))
    S = Z @ Z.T
    same = labels[:, None] == labels[None, :]
    off = ~torch.eye(len(labels), dtype=bool)
    within = S[same & off].mean().item()
    between = S[~same].mean().item()

print(f"\nmean similarity, same topic:      {within:+.3f}")
print(f"mean similarity, different topic: {between:+.3f}")
print(f"separation:                       {within - between:+.3f}")

# Where does the gradient come from? An easy batch versus a hard one.
with torch.no_grad():
    easy = infonce(sample(topics), sample(topics))                     # 8 topics
    zeros = torch.zeros(N_TOPICS, dtype=torch.long)
    hard = infonce(sample(zeros), sample(zeros))                       # all one topic

print(f"\n{'batch':<28} {'loss':>8} {'P(correct)':>12} {'gradient signal':>17}")
for name, (l, S) in [("random negatives (easy)", easy),
                     ("same-topic negatives (hard)", hard)]:
    pc = S.softmax(-1).diag().mean().item()
    print(f"{name:<28} {l.item():>8.4f} {pc:>12.3f} {1 - pc:>17.3f}")

print("\nAfter training, random negatives are perfectly separated: the loss is "
      "~0 and so is the gradient, so those batches teach nothing further. The "
      "hard batch is back at the random-guess baseline — all the remaining "
      "learning is there, which is why hard-negative mining is the main quality "
      "lever in an embedding model.")
