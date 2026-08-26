# -*- coding: utf-8 -*-
# Extracted from: Chapter 73 — Subword Tokenization: BPE, WordPiece, and SentencePiece
# Source: src/.../ch073-subword.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Fitting on the wrong domain has a measurable, and large, cost."""
from collections import Counter

PROSE = """
the report describes how the team measured the effect of the change on the
users of the service and the results were consistent with the earlier study
which described a similar effect in a different population of users
"""

CODE = """
def get_user_by_id(self, user_id): return self._db.get_user(user_id)
def get_user_by_name(self, user_name): return self._db.get_user_by(user_name)
def set_user_name(self, user_id, user_name): self._db.set_user(user_id, user_name)
class UserRepository: def __init__(self, db_conn): self._db = db_conn
"""


def train(text, n_merges):
    freqs = Counter(text.split())
    splits = {w: list(w) + ["_"] for w in freqs}
    merges = []
    for _ in range(n_merges):
        p = Counter()
        for w, s in splits.items():
            for a, b in zip(s, s[1:]):
                p[(a, b)] += freqs[w]
        if not p or max(p.values()) < 2:
            break
        best = max(p, key=lambda k: (p[k], k))
        merges.append(best)
        a, b = best
        new = {}
        for w, s in splits.items():
            m, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    m.append(a + b); i += 2
                else:
                    m.append(s[i]); i += 1
            new[w] = m
        splits = new
    return merges


def apply_merges(text, merges):
    total = 0
    for word in text.split():
        s = list(word) + ["_"]
        for a, b in merges:
            m, i = [], 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    m.append(a + b); i += 2
                else:
                    m.append(s[i]); i += 1
            s = m
        total += len(s)
    return total


prose_merges = train(PROSE, 40)
code_merges = train(CODE, 40)
words = len(CODE.split())

matched = apply_merges(CODE, code_merges)
mismatched = apply_merges(CODE, prose_merges)

print(f"code text: {words} words")
print(f"  tokenized with a code-fitted vocabulary:  {matched:>4} tokens "
      f"(fertility {matched / words:.2f})")
print(f"  tokenized with a prose-fitted vocabulary: {mismatched:>4} tokens "
      f"(fertility {mismatched / words:.2f})")
print(f"  penalty for the mismatch: {mismatched / matched:.2f}x")
print("\nThe penalty is paid on every request, forever, and it is invisible "
      "unless it is measured before the vocabulary is frozen.")
