# Extracted from: Chapter 76 — BERT, RoBERTa, and Masked Language Modeling
# Source: src/.../ch076-bert.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A tiny masked language model. The loss must start near log|V| and fall."""
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# A structured toy language: each sentence is SUBJ VERB OBJ, drawn from three
# domains whose words never mix. Reconstructing a masked token therefore
# requires reading the rest of the sentence — which is the point of the task.
DOMAINS = {
    "med": (["doctor", "nurse", "surgeon"], ["examined", "treated", "admitted"],
            ["patient", "child", "athlete"]),
    "tech": (["engineer", "developer", "architect"], ["deployed", "debugged", "refactored"],
             ["service", "pipeline", "database"]),
    "food": (["chef", "baker", "cook"], ["seasoned", "baked", "plated"],
             ["dish", "bread", "dessert"]),
}
SENTENCES = [[s, v, o] for subj, vb, ob in DOMAINS.values()
             for s in subj for v in vb for o in ob]

words = sorted({w for s in SENTENCES for w in s})
VOCAB = ["[MASK]"] + words
idx = {w: i for i, w in enumerate(VOCAB)}
V, D, T = len(VOCAB), 32, 3
MASK_ID, IGNORE = 0, -100

X = torch.tensor([[idx[w] for w in s] for s in SENTENCES])
print(f"{len(SENTENCES)} sentences, {V} vocabulary items")
print(f"uniform-prediction loss = log|V| = {math.log(V):.4f}   "
      f"<- equation (eq:mlm-baseline-loss)")


class TinyEncoder(nn.Module):
    """A one-layer bidirectional transformer encoder with a tied MLM head."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        a, _ = self.attn(h, h, h, need_weights=False)     # no mask: bidirectional
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h @ self.tok.weight.T                      # weight tying


def corrupt(x, gen):
    """Mask exactly one position per sentence — 1/3, since T = 3."""
    labels = torch.full_like(x, IGNORE)
    pos = torch.randint(0, T, (x.shape[0],), generator=gen)
    rows = torch.arange(x.shape[0])
    labels[rows, pos] = x[rows, pos]
    out = x.clone()
    draw = torch.rand(x.shape[0], generator=gen)
    out[rows[draw < 0.8], pos[draw < 0.8]] = MASK_ID
    rand = (draw >= 0.8) & (draw < 0.9)
    out[rows[rand], pos[rand]] = torch.randint(1, V, (int(rand.sum()),), generator=gen)
    return out, labels


gen = torch.Generator().manual_seed(0)
model = TinyEncoder()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(1, 601):
    inp, lab = corrupt(X, gen)
    logits = model(inp)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V), lab.reshape(-1), ignore_index=IGNORE)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 150 == 0 or step == 1:
        print(f"step {step:>4}: MLM loss {loss.item():.4f}")

# Does the model use both sides of the mask?
model.eval()
with torch.no_grad():
    probes = [["[MASK]", "examined", "patient"],
              ["engineer", "[MASK]", "pipeline"],
              ["chef", "seasoned", "[MASK]"]]
    print()
    for p in probes:
        x = torch.tensor([[idx[w] for w in p]])
        out = model(x)[0, p.index("[MASK]")]
        top = torch.topk(out.softmax(-1), 3)
        preds = [(VOCAB[i], round(float(v), 3)) for v, i in zip(top.values, top.indices)]
        print(f"{' '.join(p):<34} -> {preds}")

print("\nThe first probe masks position 0 and is only solvable by reading to "
      "the RIGHT — which a causal model could not do.")
