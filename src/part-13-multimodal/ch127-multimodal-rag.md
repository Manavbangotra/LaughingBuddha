---
id: mm-multimodal-rag
number: 127
part: XIII
tier: full
status: draft
requires: [mm-clip, mm-vlms, mm-ocr, emb-what-they-are, emb-hybrid,
           rag-structured, rag-indexing]
provides: [mixed-modality-index, modality-bias-in-ranking, per-modality-centring,
           caption-versus-joint, attribute-dilution, modality-rank-fusion,
           multimodal-hybrid-retrieval]
citations: [radford2021clip, faysse2025colpali, khattab2020colbert,
            zhai2023siglip, kim2022donut, cormack2009rrf]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why putting two modalities
in one index makes **modality**, not relevance, part of the ranking — and measure
the bias; apply **per-modality centring** and say why removing a constant
direction cannot lose content; state the opposite failure modes of
caption-then-embed and joint embedding, and why neither dominates at any caption
coverage; explain why the two combine so well; and place
{{cite:faysse2025colpali}}'s visual retrieval against both, including what it
gives up.

## 2. Why This Matters

{{ch:rag-structured}} left an unanswered question: once your corpus contains
pages that are pictures, how do you *search* it? This chapter is that question,
and it has a trap in it that {{ch:mm-clip}} set up.

**Put images and text in one index and the ranking is decided partly by
modality.** {{sec:9-practical-example}} builds an index that is half images and
half text, where every query's relevant set contains both, and measures what comes
back: **83.4% text**. Image recall is **0.149** against text's **0.418** — the
same concepts, equally present, retrieved at very different rates purely because
of how they are stored.

**That is not a retriever bug and it is not a content difference.** It is
{{ch:mm-clip}}'s modality gap arriving in a ranked list: a text query is compared
against text with one similarity distribution and against images with another, and
sorting the merged list treats the two scales as one.

**And the fix is three lines.** Per-modality centring moves the text share to
**49.6%** and more than doubles image recall to **0.356** — with no retraining and
no model change.

The second half is the architectural choice: caption the image and index the words,
or embed the image directly. Neither wins.
{{sec:9-practical-example}} measures **opposite** failure modes — a caption index
finds a mentioned attribute at **0.135** and an unmentioned one at **0.006**,
while a joint embedding is uniformly around **0.06** for both. One loses by
omission; the other by superposition.

{{maturity:ESTABLISHED}} Joint-space retrieval. {{maturity:EMERGING}} Visual
document retrieval ({{cite:faysse2025colpali}}), which skips the text layer
entirely.

## 3. Prerequisites

{{ch:mm-clip}} for the shared space and the modality gap — this chapter is that
finding's operational consequence; {{ch:emb-what-they-are}} for the anisotropy
correction being reused here; {{ch:emb-hybrid}} for fusion, which returns as
fusion *across modalities*; {{ch:mm-vlms}} for captioning; {{ch:mm-ocr}} for the
artefact argument; {{ch:rag-indexing}} for the index this all lives in.

## 4. Intuitive Explanation

### One index, two scales

You have a corpus of product photos and product descriptions. Both go into the
shared space; both go into one vector index; a user types a query.

**The results will be mostly text**, and not because the text is better. Text
queries and text items are both text, so they sit in the same cone and score
highly against one another. An image of the same product sits in the other cone
and scores lower — on *every* query, regardless of what it depicts.

**The retriever is sorting by a number that means different things for different
rows.** {{sec:9-practical-example}} measures the consequence at 83.4% text in a
50/50 index, and the user's conclusion — "the image search is bad" — is wrong. The
images are being outbid.

### Why centring works, and why it is safe

Subtract each modality's mean from its own items, renormalise.

The reason it is safe is worth being precise about. **The offset between the two
clouds carries no information about which item is relevant.** It is the same
vector for every image, so it cannot distinguish one image from another, and
{{ch:mm-clip}} showed it moves with temperature, batch size and seed — it is a
property of the training run, not of the content.

**Removing a constant direction cannot remove content that varies between items.**
That is the same argument {{ch:emb-what-they-are}} made for anisotropy, and it is
why this fix is nearly free.

### Two ways to make an image searchable

> **Caption it.** Describe the image in words, index the words. The index is
> *text*, so lexical search works, a human can read an entry, and it can be
> re-embedded without touching the images. Everything the captioner did not
> mention is **gone**.
>
> **Embed it.** Put the image directly in the shared space. Nothing is selected
> away, and everything is **diluted** into one vector together.

The instinct is to call the caption "lossy" and the embedding "lossless". Both
lose; they lose different things, and {{sec:9-practical-example}} measures which.

**The caption is better on what it covers** — 0.135 against 0.061 — because a
caption mentioning two things gives each half the vector, while an embedding
carrying eight gives each an eighth. **Selecting fewer things makes the survivors
louder.**

**And it is blind outside its coverage** — 0.006, essentially never.

**There is no coverage setting that fixes both.** Raising coverage helps omitted
attributes only by making fewer of them, and it *lowers* the mentioned score, from
0.135 to 0.064 — a captioner that describes everything has recreated the dilution
it was avoiding.

### Which is why you build both

The two failure modes are complementary, so a union recovers each one's strength.
{{sec:9-practical-example}}'s hybrid beats both at every coverage.

**And the reason to keep captions goes beyond the table**, exactly as
{{ch:mm-ocr}}'s did. A caption index is text: it supports lexical search for an
exact product code ({{ch:emb-hybrid}}), it is human-readable when someone audits a
result, and it can be re-embedded when the model changes **without re-encoding the
images**. A joint embedding must be rebuilt in full on every model upgrade.

### Or skip the text entirely

{{cite:faysse2025colpali}}'s argument, met in {{ch:rag-structured}}: for scanned
or layout-heavy documents, embed the *page image* with a vision-language model and
match with late interaction ({{cite:khattab2020colbert}}). No OCR, no captioning,
no layout parsing, and none of their failures.

**What it costs is storage and the text layer.** Late interaction is many vectors
per page ({{ch:emb-reranking}}), and you get no words — so no lexical search, no
citation of a text span, and no auditability.

## 5. Formal Explanation

### 5.1 Ranking across modalities

For query $q$ and items $x$ of modality $m(x)$, a single ranked list sorts by
$s(q, x)$. But the score distributions differ by modality:

$$ s(q, x) \mid m(x) = \text{text} \;\sim\; \mathcal{D}_T, \qquad s(q, x) \mid m(x) = \text{image} \;\sim\; \mathcal{D}_I, \qquad \mathbb{E}[\mathcal{D}_T] > \mathbb{E}[\mathcal{D}_I] $$ (eq:modality-score-shift)

so for a fixed relevance level, an item's rank depends on its modality:

$$ \Prob[x \in \text{top-}k] = f\big(\text{relevance}(x),\; m(x)\big) \quad \text{— } m \text{ should not be an argument} $$ (eq:modality-bias-in-ranking)

**{{eq:modality-bias-in-ranking}} is the defect.** Modality is a nuisance variable
that has entered the ranking function.

### 5.2 Per-modality centring

$$ \tilde{x} = \frac{x - \mu_{m(x)}}{\|x - \mu_{m(x)}\|}, \qquad \mu_m = \frac{1}{|X_m|}\sum_{x \in X_m} x $$ (eq:per-modality-centring)

Since $\mu_m$ is constant within a modality, it contributes nothing to
*within-modality* discrimination:

$$ \langle x_1 - \mu_m,\; x_2 - \mu_m\rangle \text{ preserves relative structure of } X_m $$ (eq:centring-is-safe)

and it removes the between-modality offset that {{ch:mm-clip}}'s
{{eq:modality-gap}} showed is unconstrained by the training objective. **Removing
a component with zero variance across items cannot remove information about
items.**

### 5.3 Caption coverage

Let an image have attribute set $A$, $|A| = K$, and let the caption mention
$C \subseteq A$ with $|C| = cK$. Then

$$ \Prob[\text{query about } a \text{ retrieves the image}] = \begin{cases} r_{\text{cap}}(|C|) & a \in C \\ \approx 0 & a \notin C \end{cases} $$ (eq:caption-coverage)

**The second branch is a hard zero, not a small number.** The attribute is not in
the index.

### 5.4 Attribute dilution

For a vector formed by summing $n$ attribute directions and normalising, each
contributes weight $\approx 1/\sqrt{n}$, so the similarity to a single-attribute
query is

$$ s(q_a, v) \approx \frac{1}{\sqrt{n}} \quad \Longrightarrow \quad r(n) \text{ decreasing in } n $$ (eq:attribute-dilution)

which is {{ch:rag-chunking}}'s {{eq:chunk-dilution}} in a new setting. It gives the
caption's advantage on covered attributes — $n = cK$ against $K$ — and it gives
the cost of raising coverage:

$$ \frac{r_{\text{cap}}(cK)}{r_{\text{joint}}(K)} \approx \frac{1}{\sqrt{c}} \quad\longrightarrow\quad 1 \text{ as } c \to 1 $$ (eq:coverage-tradeoff)

**{{eq:coverage-tradeoff}} says the trade cannot be tuned away.** At $c \to 1$ the
caption *is* the joint embedding — measured 0.064 against 0.063 — so full coverage
buys nothing and partial coverage buys precision at the price of blindness.

### 5.5 Why the hybrid wins

The two indexes fail on **disjoint** sets: the caption fails on $A \setminus C$
and the joint embedding fails uniformly-but-mildly everywhere. A union at depth
$k$ gives

$$ \text{recall}_{\cup} \approx 1 - \big(1 - r_{\text{cap}}\big)\big(1 - r_{\text{joint}}\big) $$ (eq:multimodal-union)

which exceeds both whenever the failures are not perfectly correlated — and here
they are close to independent by construction, since one is determined by the
captioner's choices and the other by geometry. This is {{ch:emb-hybrid}}'s
{{eq:retriever-overlap}} argument with modality replacing retriever.

### 5.6 The artefact asymmetry, again

$$ \text{caption index: rebuild on model change} = \text{re-embed text}, \qquad \text{joint index} = \text{re-encode every image} $$ (eq:reindex-asymmetry)

A caption is durable in a way an embedding is not: the words survive every model
upgrade, and {{ch:emb-what-they-are}}'s versioned-schema rule means the vectors do
not. **The same argument as {{ch:mm-ocr}}'s {{eq:different-artefacts}}**, one layer
up.

## 6. Mathematical Foundation

### 6.1 The bias, and what centring recovers

Measured, at depth 20 in a 50/50 index:

| | image recall | text recall | share of results that are text |
|---|---|---|---|
| raw | **0.149** | 0.418 | **83.4%** |
| centred | **0.356** | 0.358 | **49.6%** |

Three things to read off. **The text share moves to essentially exactly the
index's composition (49.6% against a true 50%)** — the bias is not reduced, it is
removed. **The two recalls equalise** (0.356 against 0.358), which is what
{{eq:modality-bias-in-ranking}} predicts once $m$ stops being an argument. And
image recall **more than doubles**.

Note text recall *falls*, 0.418 → 0.358. That is correct behaviour: text was
occupying slots it had not earned, and the total retrieved is fixed at $k$.

### 6.2 The dilution advantage, worked

From {{eq:attribute-dilution}}, at $K = 8$ and $c = 0.25$ the caption carries $n =
2$ attributes:

$$ \frac{s_{\text{cap}}}{s_{\text{joint}}} = \frac{1/\sqrt{2}}{1/\sqrt{8}} = 2.0 $$ (eq:dilution-advantage)

against a measured recall ratio of $0.135 / 0.061 = 2.2$. The prediction is for
*similarity* and the measurement is *recall@k*, which is a monotone but non-linear
function of it, so agreement to 10% is as much as this comparison supports.

> **MATH NOTE:** {{eq:dilution-advantage}} predicts the caption's advantage grows
> as coverage falls, without bound — at $c \to 0$ a one-attribute caption would be
> a perfect retriever for that one attribute. The table shows that too (0.135 at
> $c=0.25$, falling monotonically). What it does not show, and what stops this
> being an argument for one-word captions, is the *omitted* column: the advantage
> is bought entirely by making $|A \setminus C|$ larger. There is no free
> concentration.

### 6.3 The union, checked

At $c = 0.75$: $r_{\text{cap}} \approx 0.076$ (mentioned) and $0.006$ (omitted),
$r_{\text{joint}} \approx 0.063$ throughout. Averaging over the 75/25 split of
mentioned to omitted:

$$ \bar{r}_{\text{cap}} = 0.75(0.076) + 0.25(0.006) = 0.059 $$

and {{eq:multimodal-union}} predicts $1 - (1-0.059)(1-0.063) = 0.118$ against a
measured **0.110**. Close, and the small shortfall is the correlation the equation
assumes away — both indexes are more likely to find an image whose attribute
happens to be rare.

## 7. Internal Mechanics

```mermaid {#fig:multimodal-index caption="Three routes from an image to a searchable index, and what each leaves behind. The caption route produces text — lexically searchable, auditable, cheap to re-embed (eq:reindex-asymmetry) — and discards whatever was not described. The joint route keeps everything and dilutes it. The visual route skips words entirely and pays in storage."}
flowchart TB
    IM["image / page"] --> CAP["caption it<br/>(ch:mm-vlms)"]
    IM --> JNT["embed into shared space<br/>(ch:mm-clip)"]
    IM --> VIS["embed page patches<br/>(cite:faysse2025colpali)"]
    CAP --> TXT[("text index<br/>lexical + dense")]
    JNT --> VEC[("one vector per image")]
    VIS --> MV[("many vectors per page")]
    TXT --> FUSE["fuse by RANK<br/>(ch:emb-hybrid)"]
    VEC --> FUSE
    MV --> FUSE
    VEC -.->|"needs per-modality<br/>centring first"| FUSE
    FUSE --> R["results"]
```

### 7.1 The centring recipe

1. Compute $\mu_m$ per modality **over the corpus**, not per query.
2. Subtract and renormalise every item at index time.
3. **Centre the query with the mean of the modality it is written in.** A text
   query gets $\mu_T$.
4. Recompute $\mu_m$ when the corpus composition shifts materially.

Step 3 is the one that gets missed, and skipping it half-applies the correction.

**The alternative that avoids the issue entirely**: retrieve within each modality
separately and fuse by *rank* ({{cite:cormack2009rrf}}). Rank fusion never compares
the two score scales, so {{eq:modality-score-shift}} cannot bite — and it is the
right choice when the modalities' score distributions are very different or
unstable.

### 7.2 What to caption for

A captioner writes what is *salient*, and retrieval needs what is *distinctive*.
Those differ, and the gap is where {{eq:caption-coverage}}'s hard zero lives.

Practical consequences:

- **Prompt the captioner for the attributes you will search by**, not for a nice
  description. "List the visible text, brand marks, colours, and object counts"
  beats "describe this image".
- **Extract text separately.** OCR'd text is exact and searchable lexically;
  a caption's paraphrase of it is neither.
- **Store the caption**, not just its embedding. It is the artefact
  ({{eq:reindex-asymmetry}}).

### 7.3 When visual retrieval is the answer

{{cite:faysse2025colpali}} wins where the pipeline's *parsing* is the failure —
scanned pages, dense tables, forms, handwriting — because it removes the stage
that was losing information rather than improving it.

**It is a poor default for photographs**, where OCR was never involved and a
caption plus a joint embedding is much cheaper. And it inherits
{{ch:emb-reranking}}'s late-interaction storage cost, which is many vectors per
page rather than one.

There is a subtler reason it works so well on documents, and it is worth naming
because it connects the two halves of this chapter. A page is an unusually
*dense* item — dozens of distinct facts in one image — so
{{eq:attribute-dilution}} punishes single-vector representations of it severely,
and a caption of a page is hopeless because no summary covers a page's searchable
content. Late interaction sidesteps both by refusing to summarise: every patch
keeps its own vector, so nothing is averaged and nothing is selected away. **The
technique wins on documents not because pages are visually hard but because they
are informationally dense**, which is precisely the regime where both of the
cheaper representations fail.

## 8. Implementation

```python {tier=A name=mixed-modality-index}
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
```

The first listing fixes the index. The second asks what should go into it.

```python {tier=A name=caption-versus-joint}
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
```

## 9. Practical Example

**Modality enters the ranking.** In an index that is half images and half text,
where every query's relevant set contains both, a raw shared space returns
**83.4% text**. Image recall **0.149**, text recall **0.418** — the same concepts,
equally present.

**A user would conclude the image search is bad. It is being outbid.**
{{eq:modality-bias-in-ranking}}: modality is a nuisance variable that has entered
the ranking function, because {{eq:modality-score-shift}} puts the two comparisons
on different scales and one sorted list treats them as one.

**Per-modality centring removes it rather than reducing it.** The text share moves
to **49.6%** — essentially the index's true composition — and image recall
**more than doubles to 0.356**.

> **IMPORTANT:** Text recall *falls*, 0.418 → 0.358, and that is the correct
> behaviour rather than a regression. Depth is fixed at 20, so text had been
> occupying slots it did not earn. **The two recalls end up equal — 0.356 against
> 0.358 — which is what "modality is no longer part of the ranking" looks like.**
> And {{eq:centring-is-safe}} is why it costs nothing: the offset is constant
> within a modality, so removing it cannot remove anything that distinguishes one
> item from another.

**Caption and joint embedding fail in opposite directions.** On an attribute the
captioner mentioned, the caption index scores **0.135** against the joint
embedding's **0.061**. On one it omitted: **0.006** against **0.062**.

**The caption's advantage is dilution, not quality.** {{eq:dilution-advantage}}
predicts a factor of 2.0 from carrying two attributes instead of eight; measured
2.2. **Selecting fewer things makes the survivors louder** — and buys that
entirely by making the omitted set larger.

**And the trade cannot be tuned away.** Raising coverage lowers the mentioned
score monotonically, **0.135 → 0.064**, until at full coverage the caption *is* the
joint embedding — 0.064 against 0.063, exactly {{eq:coverage-tradeoff}}. **A
captioner that describes everything has recreated the dilution it was avoiding.**

**So build both.** The hybrid beats both at every coverage, because the failures
are near-independent by construction — one determined by the captioner's choices,
the other by geometry. {{eq:multimodal-union}} predicted 0.118 at 75% coverage
against a measured 0.110, the shortfall being the correlation it assumes away.

**And the caption's real advantage is not in this table.** It is text: lexically
searchable for an exact code, human-readable under audit, and re-embeddable on a
model upgrade **without re-encoding a single image** ({{eq:reindex-asymmetry}}).

## 10. Production Considerations

**Centre per modality before ranking a mixed index**, and remember to centre the
query too — that step is the one that gets missed.

**Or fuse by rank** ({{cite:cormack2009rrf}}), which never compares the two score
scales at all and is the safer default when the distributions are unstable.

**Recompute modality means when the corpus composition shifts.**

**Prompt captioners for searchable attributes**, not for pleasant descriptions.
Salience and distinctiveness are different targets.

**Extract text separately from captioning it.** OCR'd text is exact and lexically
searchable; a paraphrase is neither.

**Store the caption, not only its embedding.**
{{eq:reindex-asymmetry}} is the whole argument.

**Never mix embeddings from two model versions in one index** —
{{ch:emb-what-they-are}}'s rule, and the modality gap gives it a second reason,
since the gap itself moves between runs.

**Use visual retrieval where parsing is the failure**, not as a general default.
Storage is many vectors per page.

## 11. Common Mistakes

**One ranked list over two modalities with raw scores.**

**Centring the corpus and forgetting the query.**

**Concluding the image collection is poor** when it is being outbid.

**Treating a joint embedding as lossless.**

**Captioning for description rather than for search.**

**Discarding captions once embeddings exist.**

**Using one similarity threshold across modalities** — {{ch:mm-clip}} again.

## 12. Failure Modes

**Modality-skewed results.** Symptom: one modality dominates every result page.
Cause: {{eq:modality-bias-in-ranking}}. Detect by logging the modality mix of
results against the index's composition — a one-line check that almost nobody
runs.

**Invisible attributes.** Symptom: images are never found by a property that is
plainly visible in them. Cause: {{eq:caption-coverage}}'s hard zero.

**Diluted specifics.** Symptom: a distinctive detail in a busy image is
unretrievable. Cause: {{eq:attribute-dilution}}.

**Version skew.** Symptom: retrieval degrades after a model upgrade with no code
change. Cause: mixed-version vectors, and a moved modality gap.

**Threshold nonsense.** Symptom: a relevance cutoff behaves differently by
modality.

**Storage surprise from visual retrieval.** Symptom: index size an order of
magnitude above forecast. Cause: late interaction's many vectors per page.

## 13. Alternatives

| Approach | Trades away | When it wins |
|---|---|---|
| joint embedding only | lexical search, auditability, cheap reindex | photos, semantic queries |
| caption only | anything undescribed | when captions are curated and complete |
| caption + joint (hybrid) | one more index | almost always — measured best at every coverage |
| rank fusion instead of centring | score-level control | unstable or very different score distributions |
| visual page retrieval ({{cite:faysse2025colpali}}) | text layer, storage | scanned or layout-heavy documents |
| OCR text + lexical ({{ch:mm-ocr}}) | semantic matching | exact codes, identifiers, quotes |

**The bottom two rows are complements, not competitors**, and a document system
usually wants both — {{ch:mm-ocr}}'s hybrid conclusion arriving at the retrieval
layer.

## 14. Evaluation

**Report the modality mix of results** alongside recall. It is the diagnostic for
{{eq:modality-bias-in-ranking}} and it costs nothing.

**Evaluate recall per modality**, never pooled. A pooled number hid a 0.149
against 0.418 split.

**Evaluate captioned and uncaptioned attributes separately**, since
{{eq:caption-coverage}} makes them different populations.

**Measure caption coverage directly**: sample images, list their searchable
attributes by hand, and check how many the captioner mentioned. This is the number
that decides the architecture and it is rarely known.

**Report index size and reindex cost** with any retrieval comparison —
{{eq:reindex-asymmetry}} is a real operational difference.

**Test cross-modal and within-modal queries separately.**

## 15. Advanced Concepts

**Late interaction across modalities.** {{maturity:EMERGING}}
{{cite:faysse2025colpali}} applies {{cite:khattab2020colbert}}'s many-vectors-per-item
idea to page images, which sidesteps {{eq:attribute-dilution}} entirely — nothing
is summed into one vector, so nothing is diluted. It pays in storage, which is the
same trade {{ch:emb-reranking}} priced.

**Centring as a general nuisance-removal.** {{maturity:MATURE}}
{{eq:per-modality-centring}} is one instance of a pattern: identify a direction
that is constant within a group and carries no discriminative information, and
subtract it. Anisotropy correction, per-document contextual augmentation, and this
are the same operation on different groupings.

**Learned gap closure.** {{maturity:EMERGING}} Since
{{eq:modality-gap}} is unconstrained by the objective, one can train a small map
that aligns the clouds after the fact. Whether it helps depends on whether
anything downstream uses absolute scores — for pure ranking within a modality it
changes nothing.

**Structured multimodal retrieval.** {{maturity:EMERGING}} An image's *metadata* —
capture time, location, product ID — is structured data, and
{{ch:rag-structured}} says query it rather than embedding it. The strongest
multimodal systems are usually filter-then-rank, not embed-everything.

**Caption generation as index construction.** {{maturity:EMERGING}} If a caption
is an index entry rather than a description, it should be *generated for
retrieval*: enumerate attributes, include extracted text verbatim, and avoid
prose. Almost nobody prompts for this, and it is the cheapest available
improvement to a multimodal index.

**The unresolved question is what a shared embedding space actually shares.**
{{maturity:EXPERIMENTAL}} Retrieval across modalities assumes an image and a
paragraph can be near each other in a way that predicts usefulness, and the
evidence for that is strongest exactly where the pairing was supervised — captions
and their images. For a diagram and the prose that explains it, the pairing is
looser and the retrieval quality falls accordingly. **The modality gap is not a
constant to be corrected but a function of how the pair was learned**, and no
number in this chapter separates the two.

## 16. Connection to Previous Chapters

{{ch:mm-clip}}'s {{eq:modality-gap}} is what
{{eq:modality-bias-in-ranking}} operationalises, and its warning about thresholds
becomes a ranking defect here. {{ch:emb-what-they-are}}'s anisotropy correction is
{{eq:per-modality-centring}} on a different grouping.
{{ch:rag-chunking}}'s {{eq:chunk-dilution}} is {{eq:attribute-dilution}} with
attributes instead of sentences. {{ch:emb-hybrid}}'s fusion argument becomes
{{eq:multimodal-union}} with modality replacing retriever, and
{{ch:mm-ocr}}'s {{eq:different-artefacts}} becomes
{{eq:reindex-asymmetry}}. {{ch:rag-structured}} supplies the reminder that metadata
should be filtered rather than embedded. Forward:
{{ch:mm-video-audio}} adds a time axis to everything here, and
{{ch:ev-rag}} is where retrieval evaluation is treated properly.

## 17. Exercises

1. Derive {{eq:centring-is-safe}} and state what centring *would* destroy if the
   offset varied between items.
2. In `mixed-modality-index`, change the index to 90% images and 10% text. Does
   the raw bias persist, and does centring still land on the true composition?
3. In the same listing, centre the corpus but not the query. How much of the
   correction survives?
4. Implement rank fusion instead of centring and compare. Which is more robust to
   changing the modality offsets?
5. Derive {{eq:coverage-tradeoff}} and use it to predict the caption's mentioned
   score at $c = 0.125$. Check it.
6. In `caption-versus-joint`, make the captioner cover the *most distinctive*
   attributes rather than random ones. How much does the hybrid gain shrink?
7. Use {{eq:multimodal-union}} to predict the hybrid at $c = 0.25$ and account for
   the gap against the measurement.
8. Take a multimodal index you have. Log the modality mix of the top 20 for a
   hundred queries and compare it with the index's composition.

## 18. Interview Questions

1. What goes wrong when you put images and text in one vector index?
2. Why is that not a retriever bug?
3. What is per-modality centring and why is it safe?
4. Why must you centre the query as well?
5. Caption-then-embed or joint embedding — which loses more?
6. Why is a caption *better* on the attributes it mentions?
7. Is there a caption coverage that dominates? Justify.
8. Why keep captions once you have embeddings?
9. When would you use visual page retrieval, and what does it cost?
10. Your image results are never returned for text queries. Diagnose.

## 19. Research Questions

1. {{eq:per-modality-centring}} removes a mean. Is a full whitening per modality
   better, and does it damage within-modality structure?
2. {{eq:multimodal-union}} assumes independent failures. How correlated are
   caption and joint-embedding failures in practice, and what does that do to the
   hybrid's value?
3. Captioners write what is salient. Can a captioner be trained against a
   *retrieval* objective, so coverage tracks what queries ask for?
4. Late interaction avoids {{eq:attribute-dilution}} at a storage cost. Where is
   the crossover as a function of attributes per image?
5. The modality gap moves between training runs. Is there a normalisation under
   which multimodal embeddings from different runs are comparable?

## 20. Chapter Summary

**Put two modalities in one index and modality becomes part of the ranking.**
Measured in a 50/50 index where every relevant set contains both: **83.4% of
results were text**, with image recall **0.149** against text's **0.418**. Not a
retriever bug and not a content difference —
{{eq:modality-score-shift}} puts the two comparisons on different scales and one
sorted list treats them as one.

**Per-modality centring removes the bias rather than reducing it.** The text share
lands at **49.6%**, the two recalls equalise at **0.356 and 0.358**, and image
recall more than doubles. Text recall falls, correctly, because it had been
occupying slots it did not earn. And {{eq:centring-is-safe}} is why it is free: a
direction that is constant within a modality carries no information about which
item is relevant.

**Caption-then-embed and joint embedding fail in opposite directions.** On a
mentioned attribute the caption scores **0.135** against **0.061**; on an omitted
one, **0.006** against **0.062**. **One loses by omission, the other by
superposition** — and the caption's advantage is {{eq:attribute-dilution}} rather
than quality, predicted at 2.0× and measured at 2.2×.

**The trade cannot be tuned away.** Raising coverage lowers the mentioned score
monotonically until, at full coverage, the caption *is* the joint embedding —
0.064 against 0.063. {{eq:coverage-tradeoff}}: **a captioner that describes
everything has recreated the dilution it was avoiding**, and there is no setting
that dominates on both columns.

**So build both.** The failures are near-independent — one determined by the
captioner's choices, the other by geometry — so {{eq:multimodal-union}} makes the
hybrid beat both at every coverage, predicted at 0.118 and measured at 0.110.

**And the caption's decisive advantage is not retrieval accuracy at all.** It is
text: lexically searchable for an exact identifier, readable by a human auditing a
result, and re-embeddable on a model upgrade **without re-encoding a single
image** ({{eq:reindex-asymmetry}}) — the same artefact argument
{{ch:mm-ocr}} made about the text layer, one layer up.

The chapter's operating checklist is short: **log the modality mix of your
results**, centre before ranking or fuse by rank, measure your captioner's actual
coverage, and keep the words.

## 21. Further Reading

{{cite:radford2021clip}} for the shared space this chapter operates in, and
{{cite:zhai2023siglip}} for the loss most current towers use.
{{cite:faysse2025colpali}} for visual document retrieval, with
{{cite:khattab2020colbert}} for the late-interaction machinery that lets it avoid
{{eq:attribute-dilution}} — and for the storage bill that comes with it.
{{cite:kim2022donut}} for the OCR-free route into the same problem.
{{cite:cormack2009rrf}} for rank fusion, which is the alternative to centring and
often the safer one.
Within the book: {{ch:emb-hybrid}} for fusion done properly,
{{ch:emb-what-they-are}} for why the correction is safe, and
{{ch:rag-structured}} for the reminder that metadata should be queried rather than
embedded.
