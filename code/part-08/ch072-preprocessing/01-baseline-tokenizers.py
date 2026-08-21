# Extracted from: Chapter 72 — Text Preprocessing and the Tokenization Problem
# Source: src/.../ch072-preprocessing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The three naive segmentations, and the fertility measurement that ranks them."""
import unicodedata

SAMPLES = {
    "english":  "The quick brown fox jumps over the lazy dog near the riverbank.",
    "german":   "Die Donaudampfschifffahrtsgesellschaft veroeffentlichte gestern "
                "ihren Geschaeftsbericht.",
    "code":     "def fit(self, X, y): return self._solve(X.T @ X, X.T @ y)",
    "numbers":  "Revenue rose from 1234567 to 2345678 between 2023 and 2024.",
}


def char_tokenize(s):
    """One token per Unicode code point. Complete, lossless, maximally fertile."""
    return list(s)


def word_tokenize(s):
    """Whitespace segmentation. Incomplete: any unseen word becomes UNK."""
    return s.split()


def byte_tokenize(s):
    """One token per UTF-8 byte. Complete and lossless with a 256-item vocabulary."""
    return [bytes([b]) for b in s.encode("utf-8")]


def fertility(tokens, text):
    """Tokens per whitespace-delimited word — equation (eq:fertility)."""
    words = max(len(text.split()), 1)
    return len(tokens) / words


print(f"{'sample':<10} {'words':>6} {'chars':>6} {'bytes':>6} "
      f"{'f_char':>7} {'f_byte':>7}")
for name, text in SAMPLES.items():
    w = len(word_tokenize(text))
    c = len(char_tokenize(text))
    b = len(byte_tokenize(text))
    print(f"{name:<10} {w:>6} {c:>6} {b:>6} "
          f"{fertility(char_tokenize(text), text):>7.2f} "
          f"{fertility(byte_tokenize(text), text):>7.2f}")

# The vocabulary each scheme needs to cover just these four samples.
print()
print(f"character vocabulary: {len(set(''.join(SAMPLES.values())))} distinct")
print(f"word vocabulary:      "
      f"{len(set(w for t in SAMPLES.values() for w in t.split()))} distinct")
print(f"byte vocabulary:      256 by construction, always")
