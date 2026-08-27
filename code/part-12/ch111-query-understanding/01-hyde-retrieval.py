# -*- coding: utf-8 -*-
# Extracted from: Chapter 111 — Query Understanding: Rewriting, Expansion, and Multi-Query
# Source: src/.../ch111-query-understanding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Searching with a hypothetical ANSWER instead of the question -- and what the
claim that its factual accuracy is irrelevant actually depends on.

Embeddings carry a topic component, a FACT component (the passage's specific
claims), and a FORM component (register, length, declarative vs interrogative)
-- eq:topic-form-decomposition. Documents share a form; questions do not, so
eq:asymmetric-score's form term works against a raw query.

Four retrieval keys over the same corpus:
  raw question         -- question form, right topic, no facts
  hypothetical, right  -- document form, right topic, right facts
  hypothetical, WRONG  -- document form, right topic, WRONG facts
  the true answer      -- an upper bound, unavailable at query time

HyDE's claim is that the third key works nearly as well as the second. That is
not a free-standing fact: it is a claim about how much of the embedding's mass
sits in the FACT component. So we sweep that weight rather than fixing it, and
report where the claim holds and where it fails.
"""
import numpy as np

rng = np.random.default_rng(41)

N_DOC, N_QUERY, K = 5000, 400, 10
D_TOPIC, D_FACT, D_FORM = 24, 12, 8
W_TOPIC, W_FORM = 1.0, 0.75
FACT_WEIGHTS = [0.10, 0.20, 0.35, 0.55, 0.80]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


form_doc = unit(rng.normal(size=D_FORM))            # long, declarative
form_question = unit(rng.normal(size=D_FORM))       # short, interrogative

doc_topic = unit(rng.normal(size=(N_DOC, D_TOPIC)))
doc_fact = unit(rng.normal(size=(N_DOC, D_FACT)))
doc_form = unit(np.tile(form_doc, (N_DOC, 1))
                + rng.normal(scale=0.12, size=(N_DOC, D_FORM)))

# Fixed query draws, shared across every fact weight, so the sweep isolates w.
CASES = []
for _ in range(N_QUERY):
    i = int(rng.integers(0, N_DOC))
    CASES.append((
        i,
        unit(doc_topic[i] + rng.normal(scale=0.30, size=D_TOPIC)),   # query topic
        unit(doc_fact[i] + rng.normal(scale=0.45, size=D_FACT)),     # facts guessed right
        unit(rng.normal(size=D_FACT)),                               # facts wrong
        unit(rng.normal(size=D_FACT)) * 0.15,                        # question: no facts
    ))


def block(topic, fact, form, w_fact):
    return unit(np.concatenate([W_TOPIC * topic, w_fact * fact,
                                W_FORM * form], axis=-1))


print(f"{'fact weight':>12}{'raw question':>15}{'hypo (right)':>15}"
      f"{'hypo (WRONG)':>15}{'wrong vs right':>17}")
print("-" * 74)

for w in FACT_WEIGHTS:
    docs = block(doc_topic, doc_fact, doc_form, w)
    hits = {"raw": 0, "right": 0, "wrong": 0}
    for i, q_topic, f_right, f_wrong, f_none in CASES:
        keys = {
            "raw":   block(q_topic, f_none,  form_question, w),
            "right": block(q_topic, f_right, form_doc,      w),
            "wrong": block(q_topic, f_wrong, form_doc,      w),
        }
        for name, key in keys.items():
            if i in np.argpartition(-(docs @ key), K)[:K]:
                hits[name] += 1
    r_raw, r_right, r_wrong = (hits[k] / N_QUERY for k in ("raw", "right", "wrong"))
    print(f"{w:>12.2f}{r_raw:>15.3f}{r_right:>15.3f}{r_wrong:>15.3f}"
          f"{(r_wrong - r_right) * 100:>+16.1f}pp")

print("""
Read the raw-question column first: it is the worst key at every fact weight, and
it is the one every system uses by default. eq:asymmetric-score says why -- its
form component points somewhere documents do not live, so the form term works
AGAINST it however well the topic matches.

Now read the last column, which is HyDE's actual claim under test. At LOW fact
weight the wrong hypothetical performs close to the right one: retrieval is
carried by topic and form, and the invented facts barely register. At HIGH fact
weight the gap opens and the wrong hypothetical degrades sharply -- at the top of
the sweep it can fall below the raw question, because a confidently wrong fact
vector points AWAY from the target while an absent one merely fails to help.

So "the hypothetical does not need to be correct" is not a free-standing fact
about HyDE. It is a claim about where the embedding puts its mass, and it holds
in the regime where passage embeddings actually sit -- topic and register
dominating, specific claims contributing little. That is the regime described in
ch:emb-what-they-are, and it is why the technique works in practice.

But it also says exactly when to be careful: for a corpus where documents are
distinguished mainly by their SPECIFIC CLAIMS rather than their subject -- a
price list, a specification table, a set of near-identical policy variants -- the
fact component carries the discriminative signal, and a hypothetical with
invented specifics will retrieve confidently and wrongly.

Two practical consequences hold across the whole sweep. You cannot evaluate HyDE
by reading the hypotheticals: at low fact weight a factually wrong key is a good
key, so rejecting the technique because the text is wrong rejects it for the
wrong reason. And the rewriting model does not need to be a strong one -- it
needs to produce document-SHAPED text about the right topic, which is far cheaper
(eq:query-transform-cost).""")
