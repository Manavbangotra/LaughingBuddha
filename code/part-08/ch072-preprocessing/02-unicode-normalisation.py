# -*- coding: utf-8 -*-
# Extracted from: Chapter 72 — Text Preprocessing and the Tokenization Problem
# Source: src/.../ch072-preprocessing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Two strings that render identically and tokenize differently."""
import unicodedata

composed = "café"            # e-acute as one code point
decomposed = "café"          # 'e' + combining acute

print(f"visually equal:      {composed == decomposed}")
print(f"code points:         {len(composed)} vs {len(decomposed)}")
print(f"utf-8 bytes:         {len(composed.encode())} vs "
      f"{len(decomposed.encode())}")
print(f"equal after NFC:     "
      f"{unicodedata.normalize('NFC', composed) == unicodedata.normalize('NFC', decomposed)}")

# Compatibility folding destroys distinctions that NFC preserves.
pairs = [("x²", "x2"), ("ﬁre", "fire"), ("ＡBC", "ABC")]
print()
print(f"{'input':<10} {'NFC':<10} {'NFKC':<10} {'NFKC collides':>14}")
for a, b in pairs:
    nfc = unicodedata.normalize("NFC", a)
    nfkc = unicodedata.normalize("NFKC", a)
    print(f"{a!r:<10} {nfc!r:<10} {nfkc!r:<10} {str(nfkc == b):>14}")

assert unicodedata.normalize("NFKC", "x²") == "x2"
assert unicodedata.normalize("NFC", "x²") != "x2"
print("\nNFKC merges the superscript with the digit; NFC does not.")
