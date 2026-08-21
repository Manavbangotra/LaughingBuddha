# Extracted from: Chapter 73 — Subword Tokenization: BPE, WordPiece, and SentencePiece
# Source: src/.../ch073-subword.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The same pair counts, ranked by frequency and by the WordPiece score."""
from collections import Counter

TEXT = ("the theory that the queen requires a quiet quarter is quite the "
        "thing that these theorists think through thoroughly, and the "
        "quantity of quotations they require is quite the quandary")

symbols = Counter(TEXT.replace(" ", ""))
pairs = Counter()
for word in TEXT.split():
    for a, b in zip(word, word[1:]):
        pairs[(a, b)] += 1

rows = []
for (a, b), c_ab in pairs.items():
    if c_ab < 2:
        continue
    bpe_score = c_ab
    wp_score = c_ab / (symbols[a] * symbols[b])      # equation (eq:wordpiece-score)
    rows.append((a + b, c_ab, symbols[a], symbols[b], bpe_score, wp_score))

print("Top 6 by BPE's criterion (raw joint count):")
for p, c, ca, cb, _, wp in sorted(rows, key=lambda r: -r[4])[:6]:
    print(f"  {p:<4} count={c:<4} c(a)={ca:<4} c(b)={cb:<4} wp={wp:.2e}")

print("\nTop 6 by WordPiece's criterion (pointwise mutual information):")
for p, c, ca, cb, _, wp in sorted(rows, key=lambda r: -r[5])[:6]:
    print(f"  {p:<4} count={c:<4} c(a)={ca:<4} c(b)={cb:<4} wp={wp:.2e}")

qu = next((r for r in rows if r[0] == "qu"), None)
th = next((r for r in rows if r[0] == "th"), None)
if qu and th:
    print(f"\nqu: count {qu[1]}, wp score {qu[5]:.2e}")
    print(f"th: count {th[1]}, wp score {th[5]:.2e}")
    print(f"BPE prefers 'th' by count ({th[1]} > {qu[1]}); "
          f"WordPiece prefers 'qu' by {qu[5] / th[5]:.1f}x")
