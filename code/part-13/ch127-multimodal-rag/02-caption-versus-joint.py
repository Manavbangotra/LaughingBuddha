# -*- coding: utf-8 -*-
# Extracted from: Chapter 127 — Multimodal Embeddings and Multimodal Retrieval
# Source: src/.../ch127-multimodal-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Caption it, or embed it? Two ways to make an image searchable.

  CAPTION-THEN-EMBED   describe the image in words, embed the words. The index
                       is text, so ch:emb-hybrid's lexical machinery works, the
                       entry is human-readable, and everything the captioner did
                       not mention is GONE (eq:caption-coverage).
  JOINT EMBEDDING      put the image straight into a shared space
                       (ch:mm-clip). Nothing is selected away, and everything is
                       diluted together into one vector (eq:attribute-dilution).

The trade is not "lossy versus lossless". Both lose. They lose DIFFERENT things,
and this listing measures which -- by asking, for each image attribute, whether a
query about that attribute finds the image.
"""
import numpy as np

rng = np.random.default_rng(107)

DIM = 64
N_ATTR = 900               # attribute vocabulary
N_IMG = 4000
ATTR_PER_IMG = 8
K = 25                     # retrieval depth
N_QUERY = 1800


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


attr = unit(rng.normal(size=(N_ATTR, DIM)))
img_attrs = np.array([rng.choice(N_ATTR, ATTR_PER_IMG, replace=False)
                      for _ in range(N_IMG)])


def build(coverage):
    """Joint embedding holds all ATTR_PER_IMG attributes; the caption holds a
    `coverage` fraction of them, chosen at random."""
    joint = np.zeros((N_IMG, DIM))
    cap = np.zeros((N_IMG, DIM))
    n_cap = max(int(round(coverage * ATTR_PER_IMG)), 1)
    captioned = np.zeros((N_IMG, ATTR_PER_IMG), dtype=bool)
    for i in range(N_IMG):
        joint[i] = attr[img_attrs[i]].sum(0)
        pick = rng.choice(ATTR_PER_IMG, n_cap, replace=False)
        captioned[i, pick] = True
        cap[i] = attr[img_attrs[i][pick]].sum(0)
    joint = unit(joint + 0.25 * rng.normal(size=joint.shape))
    cap = unit(cap + 0.25 * rng.normal(size=cap.shape))
    return joint, cap, captioned


def recall(bank_score, target_attr, was_captioned):
    """Recall@K for queries about a given attribute, split by whether the
    captioner happened to mention it."""
    hits = {True: [0, 0], False: [0, 0]}
    for t in range(len(target_attr)):
        j = target_attr[t]
        rel = np.where((img_attrs == j).any(axis=1))[0]
        if len(rel) == 0:
            continue
        q = unit(attr[j] + 0.25 * rng.normal(size=DIM))
        top = np.argpartition(-(bank_score @ q), K)[:K]
        for i in rel:
            slot = bool(was_captioned[i, list(img_attrs[i]).index(j)])
            hits[slot][0] += int(i in top)
            hits[slot][1] += 1
    return {k: (v[0] / v[1] if v[1] else 0.0) for k, v in hits.items()}


COVERAGES = (0.25, 0.5, 0.75, 1.0)
targets = rng.integers(0, N_ATTR, size=N_QUERY)

_per = N_IMG * ATTR_PER_IMG / N_ATTR
print(f"{N_IMG} images, {ATTR_PER_IMG} attributes each, "
      f"{N_ATTR} attribute types, depth {K}.")
print(f"About {_per:.0f} images share any attribute, and a query names one "
      f"attribute out of eight present, so absolute recall is low by "
      f"construction -- read the RATIOS between columns, not the levels.")
print()
print(f"{'caption covers':>15}{'':>3}{'CAPTION index':>26}{'':>3}"
      f"{'JOINT embedding':>26}{'':>3}{'hybrid':>9}")
print(f"{'':>15}{'':>3}{'mentioned':>13}{'omitted':>13}{'':>3}"
      f"{'mentioned':>13}{'omitted':>13}{'':>3}{'overall':>9}")
print("-" * 100)

rows = {}
for cov in COVERAGES:
    joint, cap, captioned = build(cov)
    rc = recall(cap, targets, captioned)
    rj = recall(joint, targets, captioned)
    # Hybrid: an item is found if EITHER index finds it (eq:multimodal-union).
    hyb_num = hyb_den = 0
    for t in range(0, len(targets), 3):
        j = targets[t]
        rel = np.where((img_attrs == j).any(axis=1))[0]
        if len(rel) == 0:
            continue
        q = unit(attr[j] + 0.25 * rng.normal(size=DIM))
        tc = set(np.argpartition(-(cap @ q), K)[:K].tolist())
        tj = set(np.argpartition(-(joint @ q), K)[:K].tolist())
        both = tc | tj
        hyb_num += sum(int(i in both) for i in rel)
        hyb_den += len(rel)
    hyb = hyb_num / max(hyb_den, 1)
    rows[cov] = (rc[True], rc[False], rj[True], rj[False], hyb)
    print(f"{cov:>15.2f}{'':>3}{rc[True]:>13.3f}{rc[False]:>13.3f}{'':>3}"
          f"{rj[True]:>13.3f}{rj[False]:>13.3f}{'':>3}{hyb:>9.3f}")

lo, hi = rows[0.25], rows[1.0]
print(f"""
Read the two "omitted" columns against each other, because that is the whole
comparison. When the captioner did not mention an attribute, the caption index
finds the image {lo[1]:.3f} of the time at 25% coverage -- essentially never,
because the information is not in the index at all (eq:caption-coverage). The
joint embedding finds the same images {lo[3]:.3f} of the time, because nothing
was selected away: every attribute is in the vector, just diluted.

Now the "mentioned" columns, which is the other half and the one that explains
why captions persist. When the captioner DID mention the attribute, the caption
index finds it {lo[0]:.3f} against the joint embedding's {lo[2]:.3f}. The caption
is BETTER on what it covers, and the reason is dilution
(eq:attribute-dilution): a caption mentioning two attributes puts each at half
weight, while a joint embedding carrying eight puts each at an eighth. Selecting
fewer things makes the survivors louder.

So the two failure modes are opposite. A caption index is precise about a subset
and blind outside it; a joint embedding is uniformly mediocre about everything.
Neither is "lossy" in the same sense -- one loses by omission and the other by
superposition.

Follow the coverage sweep and the caption index improves on omitted attributes
for a trivial reason (there are fewer of them) while its mentioned-attribute
score DROPS, from {lo[0]:.3f} to {hi[0]:.3f}. At full coverage the caption has
become the joint embedding -- {hi[0]:.3f} against {hi[2]:.3f} -- which is
eq:coverage-tradeoff arriving exactly: a captioner that describes everything has
recreated the dilution it was avoiding. There is no coverage setting at which
captions dominate on both columns.

The hybrid column is the practical answer and it beats both at every coverage.
That is not surprising once the failure modes are stated -- they are
complementary, so a union recovers the caption's precision on mentioned
attributes and the joint embedding's coverage on omitted ones
(eq:multimodal-union). It is ch:emb-hybrid's argument with modality standing in
for retriever.

And the reason to prefer captions goes beyond this table, exactly as
ch:mm-ocr's did. A caption index is TEXT: it supports lexical search for an exact
product code, it is human-readable when someone audits a result, it can be
diffed, and it can be re-embedded when the model changes without re-encoding the
images. A joint embedding supports none of that and must be rebuilt in full on
every model upgrade (eq:reindex-asymmetry). Build both.""")
