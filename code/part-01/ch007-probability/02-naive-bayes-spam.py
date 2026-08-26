# -*- coding: utf-8 -*-
# Extracted from: Chapter 7 — Probability, Conditional Probability, and Bayes' Theorem
# Source: src/.../ch007-probability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A naive Bayes spam filter, from Bayes' theorem and nothing else.

'Naive' names the conditional-independence assumption of eq. 7.11: words are
treated as independent given the class. That is plainly false — 'free' and
'money' co-occur — and the classifier works well anyway. Chapter 35 examines why.
"""
import numpy as np
from collections import Counter

train = [
    ("win free money now claim prize", "spam"),
    ("free money click here now", "spam"),
    ("claim your free prize today", "spam"),
    ("urgent win cash prize claim", "spam"),
    ("meeting moved to tuesday morning", "ham"),
    ("please review the attached report", "ham"),
    ("lunch tomorrow at the usual place", "ham"),
    ("the report is attached for review", "ham"),
    ("can we move the meeting to friday", "ham"),
]

classes = ["spam", "ham"]
docs = {c: [t for t, lab in train if lab == c] for c in classes}
vocab = sorted({w for t, _ in train for w in t.split()})

# Priors: P(class), estimated as the class frequency.
priors = {c: len(docs[c]) / len(train) for c in classes}

# Likelihoods: P(word | class), with add-one (Laplace) smoothing so that a word
# unseen in a class gets a small probability rather than zero. Without it, one
# unseen word would zero the entire product — the classic naive Bayes failure.
counts = {c: Counter(w for t in docs[c] for w in t.split()) for c in classes}
totals = {c: sum(counts[c].values()) for c in classes}


def log_likelihood(word, c):
    return np.log((counts[c][word] + 1) / (totals[c] + len(vocab)))


def classify(text):
    """Return log-posteriors, using eq. 7.14 in log space."""
    words = [w for w in text.split() if w in vocab]
    scores = {}
    for c in classes:
        scores[c] = np.log(priors[c]) + sum(log_likelihood(w, c) for w in words)
    # Normalise the log-scores into probabilities (the log-sum-exp of Ch. 2).
    mx = max(scores.values())
    exp = {c: np.exp(scores[c] - mx) for c in classes}
    z = sum(exp.values())
    return {c: exp[c] / z for c in classes}


print(f"priors: {priors}\n")
for msg in ["free money claim now",
            "the meeting is moved to tuesday",
            "please review the free report",
            "win prize"]:
    p = classify(msg)
    verdict = max(p, key=p.get)
    print(f"{msg:<38} -> {verdict:<5} (spam {p['spam']:.3f})")

# The most discriminative words are those with the largest likelihood ratio —
# exactly the quantity in the odds form of Bayes (eq. 7.16).
print("\nmost spam-indicative words by likelihood ratio (eq. 7.16):")
ratios = {w: log_likelihood(w, "spam") - log_likelihood(w, "ham") for w in vocab}
for w, r in sorted(ratios.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {w:<10} log-likelihood-ratio {r:+.3f}  (x{np.exp(r):.1f} odds)")
