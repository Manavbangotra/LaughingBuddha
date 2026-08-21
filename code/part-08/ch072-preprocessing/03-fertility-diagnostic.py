# Extracted from: Chapter 72 — Text Preprocessing and the Tokenization Problem
# Source: src/.../ch072-preprocessing.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fertility across scripts, with the cost consequences spelled out."""

# A rough stand-in for a Latin-fitted subword vocabulary: common English
# fragments are single tokens, everything else falls back to bytes.
LATIN_MERGES = {
    "the", "ing", "ed", "er", "tion", "ly", "an", "re", "in", "on", "at",
    "es", "is", "it", "or", "en", "of", "to", "and", "for", "with", "sup",
    "port", "ticket", "account", "pay", "ment", "please", "help",
}

DOCUMENTS = {
    "english":   "please help with my payment account support ticket",
    "german":    "bitte helfen sie mir mit meinem zahlungskonto",
    "turkish":   "lutfen odeme hesabimla ilgili bana yardim edin",
    "greek":     "παρακαλώ "
                 "βοηθήστε με",
    "japanese":  "支払いアカウントを"
                 "手伝ってください",
}


def latin_fitted_tokenize(text):
    """Greedy longest-match over the merge set; bytes for anything unmatched."""
    out, i = [], 0
    while i < len(text):
        for length in range(min(8, len(text) - i), 0, -1):
            piece = text[i:i + length]
            if piece in LATIN_MERGES:
                out.append(piece)
                i += length
                break
        else:
            # No merge applies: fall back to the UTF-8 bytes of one character.
            out.extend(bytes([b]) for b in text[i].encode("utf-8"))
            i += 1
    return out


PRICE_PER_1K = 0.003   # a representative input price, in dollars
baseline = None

print(f"{'language':<10} {'words':>6} {'tokens':>7} {'fertility':>10} "
      f"{'vs english':>11} {'$/1M docs':>11}")
for lang, text in DOCUMENTS.items():
    toks = latin_fitted_tokenize(text)
    words = len(text.split())
    f = len(toks) / words
    if baseline is None:
        baseline = f
    cost = len(toks) / 1000 * PRICE_PER_1K * 1_000_000
    print(f"{lang:<10} {words:>6} {len(toks):>7} {f:>10.2f} "
          f"{f / baseline:>10.1f}x {cost:>11,.0f}")

print("\nSame request, same information, different bill — and the ratio is a "
      "property of the merge table, not of the languages.")
