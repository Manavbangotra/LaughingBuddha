---
id: part-08-intro
status: final
---

## What this part is for

{{part:7}} built the architecture. This part is about the six years in which the
field worked out what to feed it, what to train it on, and how to get a usable
representation out of it — and then watched most of its conclusions get
superseded by decoder-only models.

That last clause is why this part is easy to write badly. BERT is seven years
old. The reader already knows how the story ends. A part that presents the
encoder era as a museum wastes thirty thousand words on models nobody will
deploy.

**The encoder era's answers are still in production. They are just doing
different jobs than the ones they were built for.**

> Subword tokenization is the input layer of every model in this book. Masked
> language modelling lost pretraining and won retrieval. Named-entity
> recognition became structured extraction from an LLM. The bi-encoder, invented
> to make sentence similarity tractable, is the retrieval architecture that
> {{part:11}} and {{part:12}} are built on.

Read this part as infrastructure, not as history. Three of the load-bearing
components of the RAG systems in {{part:12}} are designed here, and two of them —
the tokenizer and the bi-encoder — run inside every generative system in the
remaining twenty parts.

## The build order

```text
   THE INPUT                     THE REPRESENTATION           THE OUTPUT
   ─────────────────────         ────────────────────────     ──────────────
   72 the tokenization           74 static embeddings         77 classification,
      problem: why there            word2vec, GloVe, and         NER, extraction
      is no right answer            what they cannot do
   73 BPE, WordPiece,            75 contextual embeddings     78 sentence
      SentencePiece, and            ELMo, and the transfer        embeddings,
      byte-level BPE                recipe before BERT           bi-encoders,
                                 76 BERT, RoBERTa, and MLM       reranking
                                    what the budget hid
```

Chapters 72–73 are the input layer: the one hand-designed component left in an
otherwise learned pipeline, and the algorithm every production tokenizer
actually runs. Chapters 74–76 are the representation, in the order the field
discovered it — one vector per word, then one vector per occurrence, then
bidirectional pretraining. Chapters 77–78 are the two output shapes that
survived: extracting structure, and comparing meaning.

**Chapter 78 is the one to read if you read only one.** It is where
{{part:11}} and {{part:12}} begin.

## Three things worth knowing before you start

**The tokenizer is fitted once and frozen for the model's entire life.** It is
not learned end-to-end with anything, it has no objective connected to
downstream loss, and every property of the model above it — cost, context
length, arithmetic ability, multilingual fairness — inherits from a compression
heuristic somebody ran over a corpus that may no longer exist.
{{ch:nlp-preprocessing}} makes this concrete and {{ch:nlp-subword}} gives the
algorithm, which is a 1994 file-compression trick applied to text unchanged.

**Three separate times, a reported advance turned out to be a training budget.**
{{cite:levy2015}} found it for word embeddings, {{cite:liu2019roberta}} for
encoder pretraining, and {{ch:fm-scaling-laws}} finds it again for foundation
models. Each time, equalising the budget across the compared systems dissolved
most of the reported difference. The disposition this should produce — *ask what
was held fixed before believing a comparison* — is the most transferable thing
in these seven chapters, and it is worth more than any of the architectures.

**"Similar" is not one relation, and choosing which one you mean is a training
decision.** A model fitted on paraphrase pairs will rank a contradiction of your
query above its answer, because a contradiction is lexically and topically
almost identical to what it contradicts. {{ch:nlp-similarity}} treats this as a
design parameter rather than a caveat, and it is the failure that most often
makes a retrieval system feel subtly, unfixably wrong.

## What is genuinely unsettled

**Whether subword tokenization has any linguistic justification.** BPE merges
the most frequent adjacent pair. That criterion has no relationship to
morphology, and the units it produces routinely cut across morpheme boundaries.
It sometimes yields morpheme-like pieces, and that is a consequence of frequency
statistics rather than a design goal. Twelve years of attempts to improve it
linguistically have not displaced it, which is itself evidence about how much
the structure it ignores is worth.

**Why bidirectional pretraining lost and whether it should have.** The arguments
against masked language modelling are countable — one-seventh the supervision
per unit of compute, a `[MASK]` token that never appears at inference, and no
ability to generate. {{cite:clark2020electra}} fixed the first two and remains a
minority choice. Nobody has run the large-scale bidirectional experiment with a
modern compute budget, and the reasons are partly sociological.

**Whether the anisotropy everyone reports actually matters.**
{{ch:nlp-similarity}} measures it, finds it severe, removes it — and the
retrieval does not improve. Anisotropy makes similarity *scores* uninformative;
it does not necessarily make *rankings* wrong. The folk version of this advice
is stated more confidently than the evidence supports, and the chapter says so.

## A note on {{ch:nlp-similarity}}

The last chapter of this part carries more forward weight than any other chapter
in the book so far. The bi-encoder/cross-encoder split, the retrieve-then-rerank
cascade, the recall ceiling, contrastive training with hard negatives — all four
are defined there and all four are assumed, without re-derivation, throughout
{{part:11}}, {{part:12}}, and the capstone.

Its cost derivation is also the cleanest instance in the book of an architecture
falling out of arithmetic rather than taste. {{eq:seven-orders}} is a ratio of
about $10^7$ between a forward pass and a dot product, and every retrieval
system in production has the shape it has because of that number.

## What you should be able to do at the end

Implement a BPE tokenizer from scratch and choose its vocabulary size from a
measured compression curve rather than from convention. Derive skip-gram's
objective, its negative-sampling replacement, and the PMI factorisation that
shows the two traditions were computing the same thing. Explain what masked
language modelling costs and what it buys, in numbers. Encode and decode BIO
tags, and say why entity-level F1 and token-level F1 disagree so violently.
Derive the bi-encoder cost advantage and the recall ceiling it implies.

And, throughout: **state what was held fixed before believing a comparison.**
Three chapters of this part exist because somebody eventually did.
