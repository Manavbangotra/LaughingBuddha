# Extracted from: Chapter 81 — Pretraining Dataset Construction and Curation
# Source: src/.../ch081-datasets.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Auditing a domain corpus before committing it to a training run."""
import numpy as np

rng = np.random.default_rng(4)

# A synthetic stand-in with the pathologies real internal corpora have.
TEMPLATE_HEADER = "confidential internal document do not distribute externally"
INCIDENT_BOILER = "incident severity impact detection mitigation follow up actions"

docs = []
# 1. Genuine unique content.
for i in range(600):
    docs.append(("unique", f"design note {i} " + " ".join(
        f"w{rng.integers(0, 400)}" for _ in range(60))))
# 2. Template-heavy documents: mostly identical boilerplate.
for i in range(300):
    docs.append(("templated", f"{TEMPLATE_HEADER} {INCIDENT_BOILER} "
                 + " ".join(f"w{rng.integers(0, 40)}" for _ in range(12))))
# 3. Machine-generated logs: high volume, near-zero information.
for i in range(400):
    docs.append(("generated", "timestamp service latency status code "
                 + " ".join(str(rng.integers(0, 9)) for _ in range(40))))
# 4. Verbatim copies (a doc pasted into several threads).
for i in range(120):
    docs.append(("duplicate", docs[i % 50][1]))

labels = [d[0] for d in docs]
texts = [d[1] for d in docs]
print(f"corpus: {len(docs):,} documents\n")


def token_entropy(text):
    """Low entropy signals templated or generated text."""
    toks = text.split()
    _, counts = np.unique(toks, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def type_token_ratio(text):
    toks = text.split()
    return len(set(toks)) / max(len(toks), 1)


print(f"{'class':<12} {'n':>5} {'mean entropy':>14} {'mean TTR':>10} "
      f"{'verdict':<22}")
for cls in ("unique", "templated", "generated", "duplicate"):
    sel = [t for t, l in zip(texts, labels) if l == cls]
    ent = np.mean([token_entropy(t) for t in sel])
    ttr = np.mean([type_token_ratio(t) for t in sel])
    verdict = "keep" if ent > 3.5 and ttr > 0.7 else "filter or downweight"
    print(f"{cls:<12} {len(sel):>5} {ent:>14.3f} {ttr:>10.3f} {verdict:<22}")

# Exact duplicate detection is trivial and worth doing first.
seen, exact_dups = set(), 0
for t in texts:
    if t in seen:
        exact_dups += 1
    seen.add(t)
print(f"\nexact duplicates: {exact_dups} ({exact_dups / len(texts):.1%})")

# What survives, and what the corpus is actually worth.
keep = [t for t, l in zip(texts, labels) if l in ("unique",)]
unique_keep = set(keep)
raw_tokens = sum(len(t.split()) for t in texts)
kept_tokens = sum(len(t.split()) for t in unique_keep)
print(f"raw tokens        : {raw_tokens:,}")
print(f"tokens after audit: {kept_tokens:,}  "
      f"(yield {kept_tokens / raw_tokens:.1%})")

print(f"""
The yield is the number to take to the planning meeting. A corpus advertised as
{raw_tokens:,} tokens is worth {kept_tokens:,} for training purposes, and the
difference is templates, machine-generated logs, and copies — none of which a
document count reveals. Equation (eq:adaptation-information-ratio) says
continued pretraining needs on the order of 10^10 tokens to move what a model
knows; measure the real yield before assuming you have them.""")
