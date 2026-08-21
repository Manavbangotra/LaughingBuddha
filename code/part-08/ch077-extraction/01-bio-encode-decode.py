# Extracted from: Chapter 77 — Classification, Named Entity Recognition, and Information Extraction
# Source: src/.../ch077-extraction.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""BIO tagging: encode spans to tags, decode tags to spans, repair what leaks."""

TOKENS = ["Jane", "Smith", "visited", "Paris", "in", "March", "2024",
          "with", "Mary", "Jones", "and", "Ann", "Lee"]
# Adjacent same-type entities are the reason B- exists at all.
GOLD_SPANS = [(0, 2, "PER"), (3, 4, "LOC"), (5, 7, "DATE"),
              (8, 10, "PER"), (11, 13, "PER")]


def spans_to_bio(spans, n):
    tags = ["O"] * n
    for start, end, typ in spans:
        tags[start] = f"B-{typ}"
        for i in range(start + 1, end):
            tags[i] = f"I-{typ}"
    return tags


def bio_to_spans(tags):
    """Decode. Any I- without a matching open entity is dropped as ill-formed."""
    spans, start, typ = [], None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or tag == "O" or \
                (tag.startswith("I-") and tag[2:] != typ):
            if typ is not None:
                spans.append((start, i, typ))
                start, typ = None, None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
        elif tag.startswith("I-") and typ is None:
            pass                      # ill-formed: I- with nothing open
    return spans


gold_tags = spans_to_bio(GOLD_SPANS, len(TOKENS))
print(f"{'token':<10} {'gold tag':<10}")
for tok, tag in zip(TOKENS, gold_tags):
    print(f"{tok:<10} {tag:<10}")

assert bio_to_spans(gold_tags) == GOLD_SPANS
print("\nround trip: spans -> BIO -> spans is exact")

# Two adjacent PER entities: without B-, 'Mary Jones and Ann Lee' would decode
# as one entity. Check that they stay separate.
decoded = bio_to_spans(gold_tags)
print(f"entities decoded: {len(decoded)} (gold {len(GOLD_SPANS)})")

# What an independent per-token classifier can emit.
ILLEGAL = ["O", "I-PER", "O", "B-LOC", "I-DATE", "B-DATE", "I-DATE",
           "O", "B-PER", "I-LOC", "O", "I-PER", "I-PER"]


def find_illegal(tags):
    bad = []
    prev = "O"
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            typ = tag[2:]
            if not (prev == f"B-{typ}" or prev == f"I-{typ}"):
                bad.append((i, prev, tag))
        prev = tag
    return bad


print(f"\nill-formed sequence: {' '.join(ILLEGAL)}")
for i, prev, tag in find_illegal(ILLEGAL):
    print(f"  position {i}: '{tag}' after '{prev}' — continuation of nothing")
print(f"decoded anyway (repair drops them): {bio_to_spans(ILLEGAL)}")
print("\nRepair is a patch. The CRF makes these sequences unreachable instead.")
