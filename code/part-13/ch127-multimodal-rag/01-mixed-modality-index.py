# -*- coding: utf-8 -*-
# Extracted from: Chapter 127 — Multimodal Embeddings and Multimodal Retrieval
# Source: src/.../ch127-multimodal-rag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A mixed-modality index, and what the modality gap does to it.

ch:mm-clip measured the modality gap and warned that no single similarity
threshold means the same thing across modalities. This listing shows the
operational consequence, which is worse than a threshold problem: put images and
texts in ONE index, query it, and the ranking is decided partly by modality rather
than by relevance (eq:modality-bias-in-ranking).

Nothing here is a bug in the retriever. Within-modality similarities live on a
different scale from cross-modality ones, so sorting one merged list by score is
sorting quantities that are not commensurable -- and the errors are systematic by
modality rather than random.

The fix is the one ch:emb-what-they-are used for anisotropy, and it is three
lines.
"""
import numpy as np

rng = np.random.default_rng(101)

DIM = 48
N_CONCEPT = 500
N_IMG = N_TXT = 3000
K = 20                       # retrieval depth
N_QUERY = 1500


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


# A shared space with a genuine modality offset: both modalities encode the same
# concepts, and each sits in its own cone (ch:mm-clip, eq:modality-gap).
concept = unit(rng.normal(size=(N_CONCEPT, DIM)))
off_img = unit(rng.normal(size=DIM)) * 0.55
off_txt = unit(rng.normal(size=DIM)) * 0.55

img_c = rng.integers(0, N_CONCEPT, size=N_IMG)
txt_c = rng.integers(0, N_CONCEPT, size=N_TXT)
IMG = unit(concept[img_c] + off_img + 0.20 * rng.normal(size=(N_IMG, DIM)))
TXT = unit(concept[txt_c] + off_txt + 0.20 * rng.normal(size=(N_TXT, DIM)))

BANK = np.vstack([IMG, TXT])
IS_TXT = np.concatenate([np.zeros(N_IMG, bool), np.ones(N_TXT, bool)])
BANK_C = np.concatenate([img_c, txt_c])


def centred_bank():
    """Per-modality centring: subtract each modality's own mean, renormalise.
    This removes the shared offset that is a property of the training run rather
    than of the content (eq:per-modality-centring)."""
    b = BANK.copy()
    b[~IS_TXT] -= BANK[~IS_TXT].mean(0)
    b[IS_TXT] -= BANK[IS_TXT].mean(0)
    return unit(b)


BANK_CENT = centred_bank()


def evaluate(bank, centre_query):
    """Query with TEXT. Relevant items are those sharing the query's concept,
    and they exist in both modalities."""
    rec_i = rec_t = 0.0
    share_txt = 0.0
    n = 0
    for _ in range(N_QUERY):
        c = int(rng.integers(0, N_CONCEPT))
        rel = np.where(BANK_C == c)[0]
        if len(rel) < 2 or not (IS_TXT[rel].any() and (~IS_TXT[rel]).any()):
            continue
        q = unit(concept[c] + off_txt + 0.20 * rng.normal(size=DIM))
        if centre_query:
            q = unit(q - BANK[IS_TXT].mean(0))
        top = np.argpartition(-(bank @ q), K)[:K]
        rel_i, rel_t = rel[~IS_TXT[rel]], rel[IS_TXT[rel]]
        rec_i += np.isin(rel_i, top).mean()
        rec_t += np.isin(rel_t, top).mean()
        share_txt += IS_TXT[top].mean()
        n += 1
    return rec_i / n, rec_t / n, share_txt / n


print(f"index: {N_IMG} images + {N_TXT} texts in one shared space; "
      f"text queries; depth {K}\n")
print(f"{'setup':<34}{'recall: images':>16}{'recall: texts':>15}"
      f"{'% of results that are text':>28}")
print("-" * 93)

raw = evaluate(BANK, centre_query=False)
print(f"{'raw shared space':<34}{raw[0]:>16.3f}{raw[1]:>15.3f}{raw[2]:>28.1%}")

cen = evaluate(BANK_CENT, centre_query=True)
print(f"{'per-modality centred':<34}{cen[0]:>16.3f}{cen[1]:>15.3f}{cen[2]:>28.1%}")

print(f"""
The last column is the finding. Half the index is images and half is text, and
the relevant set for every query contains both -- so a retriever that ranked
purely by relevance would return roughly half of each. The raw shared space
returns {raw[2]:.1%} text.

That is not the retriever preferring text because text is more relevant. It is
eq:modality-bias-in-ranking: a text query is compared against text items with
one similarity distribution and against image items with another, and the merged
list is sorted as though the two scales were the same. The modality with the
higher-scoring distribution wins slots regardless of content.

Read the recall columns for what that costs. Image recall is {raw[0]:.3f} against
text recall's {raw[1]:.3f} -- the same concepts, equally present, retrieved at
very different rates purely because of which modality they are stored in. A user
searching this index would conclude the image collection is poor. It is not; it
is being outbid.

Per-modality centring removes the shared offset, and the second row is what
happens: the text share moves to {cen[2]:.1%} and image recall rises from
{raw[0]:.3f} to {cen[0]:.3f}. The fix is subtracting each modality's own mean and
renormalising -- three lines, no retraining, no model change.

Note that text recall FALLS, from {raw[1]:.3f} to {cen[1]:.3f}, and that is
correct rather than a regression: the depth is fixed at {K}, so text had been
occupying slots it did not earn. The two recalls end up equal, which is what
"modality is no longer part of the ranking" looks like.

Note WHY it works, because the reason is the same one ch:emb-what-they-are gave
for anisotropy. The offset between the two clouds is a property of the training
run rather than of any item's content: it moves with temperature, batch size and
initialisation. Subtracting a per-modality mean removes a component that carries
no information about which item is relevant, and removing a constant direction
cannot lose content that varies between items (eq:centring-is-safe).

The operational rule is narrow and worth stating plainly. If your index contains
one modality, raw similarities are fine. If it contains two and you rank them in
one list, centre per modality first -- or rank within each modality separately and
merge by rank rather than by score, which is ch:emb-hybrid's fusion argument
applied across modalities instead of across retrievers.""")
