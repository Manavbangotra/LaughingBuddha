# Extracted from: Chapter 80 — Pretraining and Self-Supervised Objectives
# Source: src/.../ch080-pretraining.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A causal LM pretraining loop, with the two baselines that diagnose it."""
import math
from collections import Counter

import torch
import torch.nn as nn

torch.manual_seed(0)

# A tiny structured corpus: sentences with real conditional structure, so a
# context-using model can beat the unigram baseline and a broken one cannot.
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

# --- the two baselines of section 6.1 ---------------------------------------
init_loss = math.log(V)
counts = Counter(TEXT)
total = sum(counts.values())
unigram_entropy = -sum((c / total) * math.log(c / total) for c in counts.values())

print(f"vocabulary {V} types, corpus {len(TEXT):,} tokens")
print(f"  loss at initialisation  log|V|      = {init_loss:.4f}")
print(f"  unigram entropy         H(X)        = {unigram_entropy:.4f}")
print(f"  a working run must fall BELOW the unigram entropy.\n")

T, D, HEADS = 8, 64, 4


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, causal=True):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        mask = None
        if causal:
            # The causal mask is what makes this a language model rather than
            # a lookup: without it, position t can read token t.
            mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf")), 1)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h),
                         attn_mask=mask, need_weights=False)
        h = h + a
        h = h + self.ff(self.n2(h))
        return h @ self.tok.weight.T          # weight tying, ch:tf-embeddings


def batches(bs=16):
    starts = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[s:s + T] for s in starts])
    y = torch.stack([data[s + 1:s + T + 1] for s in starts])
    return x, y


def train(causal, steps=400, accum=2, lr=3e-3, warmup=40):
    torch.manual_seed(0)
    model = TinyCausalLM()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(1, steps + 1):
        # Warmup then cosine, equation (eq:pretraining-schedule).
        if step < warmup:
            scale = step / warmup
        else:
            prog = (step - warmup) / (steps - warmup)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr * scale

        opt.zero_grad()
        total_loss = 0.0
        for _ in range(accum):                      # equation (eq:gradient-accumulation)
            x, y = batches()
            loss = nn.functional.cross_entropy(
                model(x, causal).reshape(-1, V), y.reshape(-1))
            (loss / accum).backward()               # the division matters
            total_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(total_loss)
    return model, history


model, hist = train(causal=True)
print(f"{'step':>6} {'loss':>9} {'perplexity':>12} {'vs unigram':>12}")
for s in (1, 50, 100, 200, 400):
    L = hist[s - 1]
    verdict = "learning context" if L < unigram_entropy else "frequencies only"
    print(f"{s:>6} {L:>9.4f} {math.exp(L):>12.2f} {verdict:>18}")

assert abs(hist[0] - init_loss) < 1.5, "step-1 loss should start near log|V|"
assert hist[-1] < unigram_entropy, "a working run must beat the unigram entropy"
print(f"\nfinal loss {hist[-1]:.4f} < unigram entropy {unigram_entropy:.4f} "
      f"-> the model is using context, not just frequencies")
