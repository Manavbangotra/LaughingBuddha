# Extracted from: Chapter 75 — Contextual Embeddings and the Encoder Revolution
# Source: src/.../ch075-contextual.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Train a tiny biLM and measure self-similarity: static is 1.0 by construction."""
import torch
import torch.nn as nn

torch.manual_seed(0)

SENTENCES = [
    "i sat on the river bank and watched the water flow",
    "we walked along the river bank until the sun set",
    "the boat drifted past the muddy river bank at dusk",
    "the bank approved the mortgage after a credit check",
    "she visited the bank to deposit the monthly cheque",
    "the bank raised the interest rate on every account",
    "the water in the river was cold and very clear",
    "the river carried the boat past the town at dusk",
    "he opened an account and made a deposit by cheque",
    "the credit union raised the rate on the account",
]
# 'bank' occurs in two disjoint senses: three riverside sentences, three
# financial ones. Nothing labels them; the model never sees a sense inventory.

tokens = sorted({w for s in SENTENCES for w in s.split()})
idx = {w: i for i, w in enumerate(tokens)}
V, D, H = len(tokens), 32, 48


class BiLM(nn.Module):
    """Two independent directional LMs sharing token embeddings — as in ELMo."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.fwd = nn.LSTM(D, H, batch_first=True)
        self.bwd = nn.LSTM(D, H, batch_first=True)
        self.out_f = nn.Linear(H, V)
        self.out_b = nn.Linear(H, V)

    def forward(self, x):
        e = self.emb(x)
        hf, _ = self.fwd(e)
        hb, _ = self.bwd(torch.flip(e, [1]))
        hb = torch.flip(hb, [1])
        return e, hf, hb

    def loss(self, x):
        e, hf, hb = self(x)
        ce = nn.functional.cross_entropy
        # forward predicts token t+1, backward predicts token t-1
        lf = ce(self.out_f(hf[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        lb = ce(self.out_b(hb[:, 1:]).reshape(-1, V), x[:, :-1].reshape(-1))
        return lf + lb


batch = [[idx[w] for w in s.split()] for s in SENTENCES]
width = min(len(b) for b in batch)
X = torch.tensor([b[:width] for b in batch])

model = BiLM()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for step in range(400):
    opt.zero_grad()
    loss = model.loss(X)
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 399:
        print(f"step {step:>4}: biLM loss {loss.item():.4f}")

# Read out the contextual vector for each occurrence of a word.
model.eval()
with torch.no_grad():
    e, hf, hb = model(X)
    ctx = torch.cat([hf, hb], dim=-1)      # concatenate the two directions


def occurrences(word):
    out = []
    for r, s in enumerate(SENTENCES):
        for c, w in enumerate(s.split()[:width]):
            if w == word:
                out.append((r, c))
    return out


def self_similarity(word):
    occ = occurrences(word)
    if len(occ) < 2:
        return None, len(occ)
    vs = torch.stack([ctx[r, c] for r, c in occ])
    vs = vs / vs.norm(dim=1, keepdim=True)
    sims = vs @ vs.T
    n = len(occ)
    off = (sims.sum() - sims.diag().sum()) / (n * (n - 1))
    return float(off), n


print()
print(f"{'word':<10} {'occurrences':>12} {'contextual':>12} {'static':>8}")
for w in ["bank", "river", "the", "account", "dusk"]:
    s, n = self_similarity(w)
    if s is not None:
        print(f"{w:<10} {n:>12} {s:>12.3f} {1.0:>8.3f}")

# Now split 'bank' by which sense the sentence carries and compare
# within-sense against across-sense similarity.
river_rows, money_rows = {0, 1, 2, 6, 7}, {3, 4, 5, 8, 9}
occ = occurrences("bank")
vs = {r: ctx[r, c] / ctx[r, c].norm() for r, c in occ}


def mean_pair(rows_a, rows_b):
    vals = [float(vs[a] @ vs[b]) for a in rows_a for b in rows_b
            if a in vs and b in vs and a != b]
    return sum(vals) / len(vals) if vals else float("nan")


rv = [r for r, _ in occ if r in river_rows]
mv = [r for r, _ in occ if r in money_rows]
print(f"\n'bank' within the riverside sentences: {mean_pair(rv, rv):+.3f}")
print(f"'bank' within the financial sentences: {mean_pair(mv, mv):+.3f}")
print(f"'bank' across the two senses:          {mean_pair(rv, mv):+.3f}")
print("\nA static embedding reports 1.000 for all three, because it cannot "
      "represent the distinction — equation (eq:static-self-similarity).")
