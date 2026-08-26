# Extracted from: Chapter 89 — Next-Token Prediction and Cross-Entropy Loss
# Source: src/.../ch089-next-token.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why longer sequences are always less probable, and what to do about it."""
import numpy as np

rng = np.random.default_rng(2)

# Per-token probabilities for two candidate answers: a short mediocre one and
# a long good one.
short = np.array([0.55, 0.48, 0.60, 0.51])                       # 4 tokens
long_good = np.array([0.82, 0.79, 0.85, 0.88, 0.81, 0.86,
                      0.83, 0.90, 0.84, 0.87, 0.85, 0.88])       # 12 tokens

print(f"{'candidate':<16} {'tokens':>7} {'mean p':>9} {'log P':>10} "
      f"{'P':>12}")
for name, p in [("short, mediocre", short), ("long, good", long_good)]:
    lp = float(np.log(p).sum())
    print(f"{name:<16} {len(p):>7} {p.mean():>9.3f} {lp:>10.3f} "
          f"{np.exp(lp):>12.3e}")

print(f"\nThe long answer has better tokens at every position and is "
      f"{np.exp(np.log(short).sum()) / np.exp(np.log(long_good).sum()):.0f}x "
      f"MORE probable in the short one's favour.")
print("Equation (eq:log-sequence-probability): every term is negative, so "
      "length is a penalty regardless of quality.\n")

# Length normalisation, equation (eq:length-normalised-score).
print(f"{'alpha':>7} {'short score':>13} {'long score':>12} {'winner':>16}")
for alpha in (0.0, 0.5, 0.7, 1.0):
    s = np.log(short).sum() / (len(short) ** alpha)
    l = np.log(long_good).sum() / (len(long_good) ** alpha)
    print(f"{alpha:>7.1f} {s:>13.4f} {l:>12.4f} "
          f"{('long' if l > s else 'short'):>16}")

# How strong is the bias in general?
print(f"\n{'length':>8} {'log P at mean p=0.8':>22} {'P':>12}")
for n in (1, 5, 10, 50, 200):
    lp = n * np.log(0.8)
    print(f"{n:>8} {lp:>22.2f} {np.exp(lp):>12.2e}")

print("""
A 200-token answer in which the model is 80% confident at every single step has
a sequence probability of about 1e-19. That number is not a quality judgement
and cannot be compared against a 5-token answer's.

Length normalisation with alpha near 1 — mean token log-probability — makes the
comparison sane, and it is a heuristic with no principled basis. There is no
correct alpha, which is worth knowing before building a ranking on it.""")
