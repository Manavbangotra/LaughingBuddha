# Extracted from: Chapter 77 — Classification, Named Entity Recognition, and Information Extraction
# Source: src/.../ch077-extraction.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The same predictions, scored two ways. Equation (eq:entity-f1)."""

GOLD = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-DATE", "I-DATE"]
TOKENS = ["Jane", "Smith", "visited", "Paris", "in", "March", "2024"]

# A single dropped final token on the PER entity — the most common NER error.
PRED = ["B-PER", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE"]


def bio_to_spans(tags):
    spans, start, typ = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-") or tag == "O" or \
                (tag.startswith("I-") and tag[2:] != typ):
            if typ is not None:
                spans.append((start, i, typ))
                start, typ = None, None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
    return set(spans)


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# Token level, over non-O gold tokens (the usual convention).
tp_tok = sum(1 for g, p in zip(GOLD, PRED) if g == p and g != "O")
n_pred_tok = sum(1 for p in PRED if p != "O")
n_gold_tok = sum(1 for g in GOLD if g != "O")

# Entity level: a span counts only on exact start, end, and type.
gold_spans, pred_spans = bio_to_spans(GOLD), bio_to_spans(PRED)
tp_ent = len(gold_spans & pred_spans)

print(f"{'':13} {'precision':>10} {'recall':>8} {'F1':>8}")
for name, (p, r, f) in [
        ("token level", prf(tp_tok, n_pred_tok, n_gold_tok)),
        ("entity level", prf(tp_ent, len(pred_spans), len(gold_spans)))]:
    print(f"{name:<13} {p:>10.3f} {r:>8.3f} {f:>8.3f}")

print(f"\ntoken accuracy over all positions: "
      f"{sum(g == p for g, p in zip(GOLD, PRED)) / len(GOLD):.3f}")
print(f"gold spans: {sorted(gold_spans)}")
print(f"pred spans: {sorted(pred_spans)}")
print(f"exactly matched: {sorted(gold_spans & pred_spans)}")

# Now the systematic case from equation (eq:token-recall): drop the last token
# of every entity, for entities of length L.
print(f"\n{'L':>3} {'token recall':>14} {'entity recall':>15}")
for L in [2, 3, 4, 8]:
    print(f"{L:>3} {1 - 1 / L:>14.3f} {0.0:>15.3f}")
print("\nEquation (eq:token-recall) against (eq:entity-recall): the token metric "
      "rises toward 1.0 as entities get longer while the entity metric stays "
      "at zero. Reporting the first for a span task reports a different task.")
