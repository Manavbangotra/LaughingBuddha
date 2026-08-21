# Part VIII — Natural Language Processing: research notes

Research pass run 2026-08-20, before writing. Full tier: 21 sections per
chapter, 4,200-word floor, seven chapters. Twenty-seven new bibliography
entries, every one verified against an arXiv abstract page, an ACL Anthology
record, or a publisher listing on the date above.

## The problem this part has to solve

Part VIII is the part most likely to be written badly, and the failure mode is
specific: **writing it as history.** BERT is seven years old. The reader knows
that decoder-only models won. A chapter that presents the encoder era as a
sequence of superseded systems teaches nothing and wastes 30,000 words.

The material is not history, and the argument for that is concrete:

- **Subword tokenization is the input layer of every model in the book.** GPT,
  Llama, and Claude all tokenize with a byte-level BPE or unigram variant.
  {{cite:sennrich2016}} is not background for Part VIII; it is the thing that
  runs before every forward pass in Parts IX through XXVIII.
- **The bi-encoder is the retrieval architecture of Parts XI and XII.**
  {{cite:reimers2019}} is where it is invented. RAG is not comprehensible
  without it.
- **Cross-encoder reranking is still BERT-shaped.** Not a legacy system — the
  best reranker for a given latency budget is usually still a fine-tuned
  encoder.
- **Span extraction is the grounding pattern.** {{cite:rajpurkar2016}}'s "the
  answer is a span of the source" is the constraint that makes a RAG citation
  checkable.

**The organising claim for the part: the encoder era solved representation, and
every one of its answers is still in production doing a different job than the
one it was built for.** MLM lost the pretraining race and won retrieval. NER
became structured extraction. Static embeddings lost the accuracy argument and
kept the latency one. Each chapter should end by naming where its subject
currently lives, not where it used to.

## What changes at this tier for this material

Parts VI and VII derive things. Part VIII mostly cannot — there is no
derivation of BPE, because BPE is a greedy heuristic with no objective
function. The sections have to be earned differently:

- **§6 Mathematical Foundation.** The genuine mathematical content is: the
  softmax-over-vocabulary cost and its negative-sampling approximation
  ({{cite:mikolov2013distributed}}), the PMI factorisation result
  ({{cite:levy2014}}), GloVe's weighted least-squares objective
  ({{cite:pennington2014}}), the unigram tokenizer's likelihood and its Viterbi
  segmentation ({{cite:kudo2018subword}}), and the MLM objective's
  15%-of-positions sample inefficiency. That is enough for seven chapters if it
  is distributed honestly and not padded. Where a method has no derivation, say
  so — that is itself the finding.
- **§7 Internal Mechanics.** Strong here: the merge table and how encoding
  actually applies it, the co-occurrence matrix's memory cost, the
  `[MASK]`-token train/test mismatch and the 80/10/10 patch, BIO decoding and
  its illegal transitions.
- **§12 Failure Modes.** Unusually rich, and mostly about tokenization:
  number splitting, code indentation, non-Latin fertility
  ({{cite:petrov2023}}), the trailing-whitespace class of bugs, and the
  anisotropy of raw BERT sentence vectors.
- **§19 Research Questions.** Genuinely open ones exist here — see below.

The padding risk is highest in chapters 76 and 77, where the temptation is to
enumerate model variants. Rule for this part: **a model variant earns a
paragraph only if it isolated a variable.** ALBERT earns one because
sentence-order prediction diagnosed NSP. ELECTRA earns one because
replaced-token detection isolated the 15% inefficiency. A variant that is just
"BERT but bigger" earns a table row.

## The genuinely live questions

### 1. Does BPE have any linguistic justification?

No, and most tutorials imply otherwise.

{{cite:gage1994}} is a data-compression article in *The C Users Journal* — a
trade magazine — about shrinking files by replacing frequent byte pairs.
{{cite:sennrich2016}} applied it to text unchanged. The merge criterion is
"most frequent adjacent pair", which has no relationship to morphology, and the
resulting units routinely cut across morpheme boundaries.

The honest position: it is a compression heuristic that works well enough that
twelve years of attempts to improve it linguistically have not displaced it.
{{cite:kudo2018subword}}'s unigram model is the one mainstream alternative with
an actual probabilistic objective, and it is competitive but not decisively
better — which is itself evidence that the linguistic structure BPE ignores is
not worth much to a model with enough capacity.

**What not to claim:** that BPE "learns morphology". It sometimes produces
morpheme-like units, and this is a consequence of frequency statistics, not a
design goal.

### 2. What did RoBERTa actually establish?

{{cite:liu2019roberta}} changed no architecture. It trained BERT longer, on
more data, with bigger batches, dropped next-sentence prediction, and made
masking dynamic — and reached state of the art. The paper's own framing is that
BERT was "significantly undertrained".

The uncomfortable implication, which should be stated plainly: **a large share
of the 2019-2020 encoder literature was measuring training budget while
reporting architecture.** This is the same finding as {{cite:levy2015}} for
static embeddings four years earlier, and the same finding as
{{cite:hoffmann2022chinchilla}} for pretraining three years later. Three
independent instances of one methodological failure is a pattern worth naming
explicitly in the text — it is the most transferable thing in this part.

Note RoBERTa was never published at a venue. Cite it as an arXiv preprint;
the entry says so.

### 3. Why did bidirectional MLM lose pretraining and win retrieval?

Settled enough to state, subtle enough to get wrong.

MLM's disadvantage for pretraining is sample efficiency and task alignment: it
supervises 15% of positions per sequence where causal LM supervises 100%, and
it cannot generate, so the generative capability had to come from somewhere
else. {{cite:clark2020electra}} attacked precisely the first of these.

MLM's advantage for representation is that the objective is bidirectional, so a
token's representation can depend on what follows it. For encoding a fixed
document into one vector, this is exactly what is wanted — and it is why
embedding and reranking models are still encoder-shaped
({{cite:reimers2019}}) even though every generative model is not.

**What not to claim:** that decoder models cannot produce good embeddings. They
can and increasingly do. The claim is narrower: at equal size and latency, the
bidirectional objective is better suited to the encoding job, and the market
still reflects that.

### 4. Are static embeddings obsolete?

No, and the argument has to be made with numbers rather than asserted.

A 300-dimensional GloVe lookup is a memory access. A BERT-base encoding is
~110M parameters of compute per sequence. For a classifier over short texts at
high throughput, averaged static vectors with a linear model remain a genuinely
reasonable baseline — and {{cite:reimers2019}} reports that *out-of-the-box*
BERT sentence vectors are worse than averaged GloVe, which is the single most
useful fact in this part for calibrating expectations.

{{cite:bojanowski2017}} keeps fastText relevant for morphologically rich and
low-resource languages, where a subword-composed vector for an unseen word
beats no vector at all.

**The chapter's job** is to give the reader the cost/quality frontier, not to
declare a winner.

### 5. Is the count-versus-predict distinction real?

No. {{cite:levy2014}} proved skip-gram with negative sampling implicitly
factorises a shifted PMI matrix, and {{cite:levy2015}} showed the remaining
performance differences were hyperparameters. This is the cleanest available
example of a neural method turning out to have a closed-form interpretation,
and it should be presented as such rather than as trivia.

## Per-chapter findings

### 72 — Text Preprocessing and the Tokenization Problem

The chapter is really about **why the classical preprocessing pipeline
disappeared**. Lowercasing, stopword removal, and stemming all destroy
information that a contextual model uses; they made sense when the model was a
bag-of-words count vector and stopped making sense the moment the model could
condition on order.

Concrete material: Unicode normalisation forms and why NFC-vs-NFKC is a
correctness issue not a style issue; the fact that "the same string" can be two
byte sequences; whitespace and the trailing-space class of bugs.

The cost argument lands here: {{cite:petrov2023}} measured up to 15x
differences in tokenized length for the same content across languages, with
direct consequences for price, latency, and usable context.
{{cite:rust2021}} isolated the tokenizer as a variable and attributed a
measurable share of the multilingual performance gap to it alone.

Tier A code: a Unicode normalisation demonstration; a fertility measurement
comparing character, word, and byte-level segmentation over inline multilingual
strings.

### 73 — Subword Tokenization: BPE, WordPiece, SentencePiece

The technical core of the part. Three algorithms, one job, three different
merge criteria:

- **BPE** ({{cite:gage1994}}, {{cite:sennrich2016}}): merge the most *frequent*
  pair. No objective.
- **WordPiece** ({{cite:schuster2012}}, described practically in
  {{cite:wu2016}}): merge the pair that most increases the training-data
  *likelihood* under a unigram model — that is, maximise
  `count(xy) / (count(x)·count(y))`, which is a pointwise-mutual-information
  criterion, not a frequency one. This distinction is what most tutorials get
  wrong.
- **Unigram** ({{cite:kudo2018subword}}): start large, prune by likelihood loss,
  keep a distribution over segmentations, decode with Viterbi.

{{cite:kudo2018sentencepiece}} is orthogonal to all three — it is the
implementation decision to treat raw Unicode as the input and whitespace as an
ordinary symbol, which is what makes segmentation lossless and reversible.

Byte-level BPE ({{cite:radford2019}}, already in the bibliography) closes the
vocabulary properly: 256 byte values as the base alphabet means no unknown
token is possible, ever, for any input.

Tier A code: a complete BPE trainer and encoder from scratch (~60 lines), then
a measurement of merges-versus-compression, then a demonstration that the same
string segments differently under a different merge budget.

### 74 — Static Word Embeddings: Word2Vec and GloVe

Skip-gram, CBOW, negative sampling, and GloVe's weighted least-squares
objective, all derivable and all worth deriving. Then {{cite:levy2014}} to
collapse the apparent distinction, and {{cite:levy2015}} to explain why the
benchmark differences were smaller than reported.

The analogy result (`king - man + woman ≈ queen`) needs careful handling: it is
real, it is the most over-quoted result in NLP, and the standard evaluation
protocol excludes the three input words from the answer candidates — without
which the nearest neighbour is frequently just `king`. State this.

Tier A code: skip-gram with negative sampling trained on an inline corpus in
numpy, showing the loss falling and the nearest neighbours becoming sensible;
a co-occurrence matrix and a small GloVe fit; the PMI-SVD baseline that
{{cite:levy2014}} predicts should be comparable.

### 75 — Contextual Embeddings and the Encoder Revolution

The chapter's argument: **one vector per word type is a modelling error**, and
{{cite:peters2018}} is the fix. "Bank" is the standard example and is fine, but
the stronger demonstration is quantitative — the same word type in different
contexts should have low cosine similarity under a contextual model and
similarity exactly 1.0 under a static one.

{{cite:howard2018}} belongs here: it established fine-tune-the-whole-model
before BERT and with a recurrent architecture, which separates the transfer
recipe from the transformer. Feature-based versus fine-tuning transfer is the
axis this chapter sets up and Part XIV revisits.

ELMo's layer-mixing result — different depths carry different linguistic
information, and a learned combination beats any single layer — is the origin of
the modern practice of choosing which layer to read features from.

### 76 — BERT, RoBERTa, and Masked Language Modeling

MLM as an objective, derived: masking rate, the `[MASK]` train/test mismatch,
the 80/10/10 patch and what it actually buys, and the sample-efficiency
accounting that {{cite:clark2020electra}} attacks.

NSP is the interesting failure. {{cite:liu2019roberta}} dropped it with no
loss; {{cite:lan2020albert}} diagnosed why — the task is too easy, solvable
from topic overlap alone — and replaced it with sentence-order prediction.
Together they make a complete story about auxiliary objectives that a list of
model variants would not.

{{cite:sanh2019}} is why encoder models are still economically deployable.
{{cite:conneau2020xlmr}} is the multilingual capacity dilution result.
{{cite:wang2019glue}} is the benchmark this era optimised against, and its
saturation within about a year is the canonical benchmark-saturation example —
which forward-references Part XXV.

Tier A code: the masking function with the 80/10/10 split, exact and
verifiable; a tiny MLM training loop on a toy vocabulary in torch showing the
loss falling below the uniform-prediction baseline. Tier B: loading a
pretrained checkpoint, clearly marked as not executed.

### 77 — Classification, NER, and Information Extraction

Sequence labelling as a task type. BIO encoding, the illegal-transition problem
(`I-PER` following `O`), and why {{cite:lample2016}}'s CRF layer beats
independent per-token classification — structured prediction when the label
space has hard constraints.

{{cite:tjongkimsang2003}} matters for one specific reason worth a full
subsection: **entity-level F1, not token-level.** Getting three tokens of a
four-token entity right scores zero. The unit of evaluation is a modelling
decision, and this is the book's cleanest example of it.

The forward connection is the point of the chapter: this task is now usually
performed by prompting an LLM for structured output. The chapter should be
honest that the encoder approach still wins on cost and latency at volume, and
that the LLM approach wins on schema flexibility and zero-shot coverage.

Tier A code: BIO encode/decode with constraint repair; entity-level F1
implemented from scratch and contrasted with token-level F1 on the same
predictions, showing the two numbers differ substantially.

### 78 — Semantic Similarity and Sentence Embeddings

The bi-encoder/cross-encoder split, which is the architectural payload of the
whole part for Parts XI and XII.

Pooling: CLS versus mean, and why mean pooling usually wins on an unfine-tuned
model. The anisotropy problem — raw BERT embeddings occupy a narrow cone, so
cosine similarities are compressed into a high, uninformative range.
{{cite:reimers2019}}'s finding that out-of-the-box BERT vectors underperform
averaged GloVe is the concrete consequence.

Then the cost accounting that makes the architecture obvious: comparing n
sentences pairwise costs O(n²) cross-encoder forward passes and O(n) bi-encoder
encodings plus O(n²) dot products, where the dot product is nanoseconds and the
forward pass is milliseconds. Retrieve-then-rerank falls straight out of this.

{{cite:muennighoff2023mteb}} for model selection, with the honest framing: no
model dominates, the ranking depends on the task, and the leaderboard is
increasingly optimised against.

Tier A code: mean versus CLS pooling on random-init weights to show the pooling
mechanics; an anisotropy measurement (mean pairwise cosine of random vectors
versus of a cone-distributed set); the O(n²)-versus-O(n) cost model computed
explicitly.

## Cross-part bookkeeping

**Backwards** — the anchors that already exist: `tf-embeddings` (ch066) for the
embedding matrix and weight tying, `tf-architectures` (ch068) for the
encoder/decoder distinction, `dl-losses` (ch052) for cross-entropy,
`ml-metrics` (ch034) for precision/recall/F1, `math-norms` (ch005) for cosine
similarity, `dl-autoencoders` (ch061) for representation learning,
`dl-rnns` (ch060) for ELMo's BiLSTM.

**Forwards** — what Part VIII must set up and not spend: tokenization economics
(Part X), the bi-encoder (Part XI), retrieve-then-rerank (Part XI and XII),
benchmark saturation (Part XXV), full-fine-tuning as the assumed default that
Part XIV then challenges.

**Do not write in this part:** decoding strategies (Part X), attention internals
(done in Part VII), vector index structures (Part XI), or anything about
generative evaluation (Part XXV).
