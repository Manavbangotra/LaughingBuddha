---
id: part-08-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about an hour and tells you what to
re-read. The assignment — build a tokenizer and evaluate it honestly — is the
piece of work this part was written for, and it is deliberately a *measurement*
project rather than a modelling one. The challenge is open-ended. The interview
section is what to rehearse.

No answers are provided. Every question is answerable from the chapters, and
looking it up is the exercise.

## Knowledge check

**The tokenization problem**

1. State the three quantities in tension in {{eq:tokenizer-tradeoff}} and say
   what each is bought with.
2. Using {{eq:zipf-coverage}}, explain why the out-of-vocabulary problem cannot
   be solved by making the vocabulary bigger. Give the coverage at
   $|V|=10^4$ and $10^5$ for $R=10^6$.
3. Define fertility. Name the class of languages for which it is not a
   meaningful comparison, and say why.
4. Why is Unicode normalisation a correctness issue rather than a style choice?
   Give the input on which NFC and NFKC disagree in a way that matters to a
   model doing algebra.
5. A model compares `9.11` and `9.9` incorrectly. Describe the two-minute check
   that determines whether the tokenizer is implicated.

**Subword algorithms**

6. Run BPE by hand for three merges on `{low: 5, lower: 2, newest: 6, widest: 3}`
   and state where the tie-break changed the vocabulary.
7. State WordPiece's merge criterion and show it is a pointwise mutual
   information score. Why does it prefer `qu` to `th`?
8. What does SentencePiece contribute? It is not an algorithm — say what it
   actually is.
9. Why can byte-level BPE never emit an unknown token, and what does that
   property cost?
10. Unigram can do subword regularization and BPE cannot. Explain the difference
    in terms of what each algorithm's output *is*.

**Static embeddings**

11. Derive the skip-gram objective and say precisely why
    {{eq:skipgram-softmax}} is a definition rather than an implementation.
12. Write down the negative-sampling gradient {{eq:negsampling-gradient}} and
    read its two terms aloud in words.
13. State {{cite:levy2014}}'s result. What does it imply about the
    count-versus-predict distinction, and about the role of $K$?
14. What did {{cite:levy2015}} find, and what does it demand of anyone comparing
    two embedding methods?
15. Chapter 74 trained on a corpus where `doctor` and `nurse` never co-occur,
    and they came out as near neighbours. What is that demonstrating?

**Contextual representations**

16. State the domain of {{eq:static-embedding}} and of
    {{eq:contextual-embedding}}, and explain why no amount of capacity lets the
    first approximate the second.
17. Derive {{eq:polysemy-floor}}. What does it say the irreducible error is, and
    which words have a small one?
18. Prove that ELMo's learned layer mixture is weakly better than any single
    layer. What is the paper's *empirical* claim, as distinct from this?
19. Distinguish feature-based from fine-tuning transfer. Name the variable that
    usually decides between them.
20. What did {{cite:howard2018}} establish that BERT is normally credited with?

**Masked language modelling**

21. Using {{eq:dependency-closure}}, explain why the predicted token must be
    removed from the *input* rather than masked inside the attention pattern.
22. State the three branches of {{eq:bert-corruption}} and what each one buys.
    Which of the three is most often described incorrectly?
23. Compute MLM's supervision per unit of compute relative to a causal
    objective. Which chapter's technique recovers that ratio?
24. Next-sentence prediction was removed with no loss. Give RoBERTa's evidence
    and ALBERT's explanation, and state the general lesson about auxiliary
    tasks.
25. What is the first number to check in an MLM training run, and what does it
    tell you when it is wrong?

**Extraction**

26. Why does BIO need `B-` in addition to `I-`? Give a sentence that breaks
    without it.
27. Gold `B-PER I-PER O B-ORG I-ORG I-ORG`, predicted
    `B-PER I-PER O B-ORG I-ORG O`. Compute both token-level and entity-level
    P/R/F1.
28. Using {{eq:token-recall}} and {{eq:entity-recall}}, explain how a system can
    score 0.875 on one metric and 0.000 on the other with the same predictions.
29. Write down the CRF score {{eq:crf-score}} and say why the sum over
    transitions runs to $T+1$ rather than $T$. What breaks if you forget?
30. An entity recogniser has recall 0.9. What is the ceiling on a binary
    relation extractor built on top of it, before it makes any error of its own?

**Similarity and retrieval**

31. Prove that the bi-encoder's hypothesis class is a strict subset of the
    cross-encoder's.
32. Derive {{eq:cost-ratio}} and evaluate it. Why is the answer the reason every
    retrieval system has the shape it has?
33. State {{eq:recall-ceiling}}. What does it say about which metric to tune the
    first stage on, and what is the common error?
34. Why do out-of-the-box BERT sentence vectors lose to averaged GloVe? Give both
    reasons.
35. Define anisotropy and give the reference value it should be compared
    against. Chapter 78 removed it and retrieval did not improve — what does
    that establish?
36. Why do in-batch negatives stop teaching anything once a model is trained,
    and what replaces them?

## Practical assignment

**Build a subword tokenizer from scratch and evaluate it honestly.** Not a
model — a tokenizer, plus the measurement harness that decides whether it is any
good. The point of the assignment is that almost nobody does the second half,
and the second half is where the decisions are.

### Part A — the tokenizer

Implement, in one file, with no tokenization library:

1. **A BPE trainer** producing an ordered merge list, operating on unique words
   with counts, with a deterministic tie-break you have chosen and documented.
2. **An encoder** that replays the merges in order, with a per-word cache.
3. **A byte-level variant** whose base alphabet is the 256 byte values.
4. **A unigram tokenizer**: seed from all substrings up to length 6, run EM,
   prune by likelihood loss to a target size, decode with the Viterbi recurrence
   {{eq:viterbi-recurrence}}.

**Acceptance criteria.** `decode(encode(s)) == s` for every string in a corpus
that includes emoji with modifiers, combining characters, CJK text, code with
significant indentation, and — for the byte-level variant — at least one byte
sequence that is not valid UTF-8. The BPE and unigram tokenizers must be fitted
to the same target vocabulary size from the same corpus.

### Part B — the measurements

For each tokenizer, on a held-out corpus you did not fit on:

5. **Compression ratio** in characters per token, and **fertility**
   {{eq:fertility}} per language and per content type. Report fertility only
   where it is meaningful, and say where it is not.
6. **The marginal-merge curve**: compression against merge count, from 0 to your
   target. Identify the knee and state the vocabulary size you would choose from
   it, with the parameter cost $2|V|d$ at $d = 768$ attached.
7. **Byte-fallback rate** per content type.
8. **The greedy gap**: Viterbi-decode BPE's *own* vocabulary and report what
   fraction of fertility is pure greedy suboptimality. Chapter 73 flags this as
   apparently unpublished — so measure it and see.

**Acceptance criterion.** Each measurement is a number with a stated corpus and
a stated method, not a plot without axes.

### Part C — the pathologies

9. **Digits.** Tokenize the integers 1–10,000 and report how many are a single
   token, the distribution of pieces per number, and whether the segmentation is
   consistent across magnitudes.
10. **Whitespace.** Demonstrate the trailing-space failure: find a prompt whose
    tokenization changes when a trailing space is added, and explain the
    consequence for a generative model.
11. **Domain mismatch.** Fit on prose, evaluate on code; fit on code, evaluate
    on prose. Report the penalty ratio both ways and say which direction is
    worse.
12. **Non-Latin cost.** Take one paragraph, translate it into five languages
    including at least two non-Latin scripts, and produce the table of tokens,
    cost, and share of a 128k context window consumed. Compare your ratios with
    the range {{cite:petrov2023}} reports.

### Part D — the decision

13. Write one page recommending a vocabulary size and algorithm for a stated
    use case of your choosing — name the traffic mix, the volume, and the
    latency budget. Justify it from your own numbers in Parts B and C, and state
    explicitly which measurement would change your mind.

**The last clause is the point.** A recommendation that no measurement could
overturn is a preference, and this part has spent three chapters on why those
are hard to tell apart from findings.

## Advanced challenge

Pick one. Each is a real experiment with an outcome nobody has handed you.

**Measure what tokenization costs arithmetic.** {{ch:nlp-preprocessing}} asserts
that inconsistent digit segmentation makes arithmetic harder. Test it: hold a
small model and its training data fixed, vary *only* the digit segmentation
(byte-level, frequency-fitted, forced single-digit), and measure accuracy on
addition and comparison by operand magnitude. Predict the result before you run
it, and record the prediction.

**Equalise the budget on an encoder claim.** Take one architectural claim from
the 2019–2020 encoder literature, reimplement both sides at genuinely matched
training budget and data, and report what survives. {{cite:liu2019roberta}} did
this once and the field is still absorbing it. A negative result here is worth
more than a positive one.

**Close the fairness gap without retraining.** {{cite:petrov2023}} documents the
disparity and vocabulary adaptation is the obvious lever. Take a pretrained
model, replace its tokenizer with one fitted to a target language, initialise
the new embedding rows from the old, and measure both the gap closed and the
English performance lost.

**Find out whether ELECTRA should have won.** Implement replaced-token detection
and MLM at matched compute on the same data, and compare downstream quality per
FLOP. Then form a view on whether its minority status is a technical fact or an
ecosystem accident, and say which of your measurements supports that view.

**Separate anisotropy from the bottleneck.** {{ch:nlp-similarity}} found that
fixing the geometry did not fix the retrieval. Take a real embedding model and a
real corpus, and decompose the bi-encoder's gap to a cross-encoder into the part
attributable to the cone and the part attributable to the fixed-size summary.
The standard advice assumes the first term dominates; the chapter's experiment
suggests it does not.

## Interview preparation

**The six derivations to do without notes.**

1. Skip-gram's objective, why its softmax is intractable, and negative sampling
   as the replacement — {{eq:skipgram-objective}} to {{eq:negative-sampling}}.
2. The negative-sampling gradient, read as attract-and-repel
   {{eq:negsampling-gradient}}.
3. Skip-gram as implicit PMI factorisation {{eq:sgns-is-pmi}}.
4. Why bidirectional conditioning leaks without masking
   {{eq:dependency-closure}}.
5. The CRF forward recurrence {{eq:forward-recurrence}}, and why it is exact.
6. The bi-encoder cost ratio {{eq:cost-ratio}} and the recall ceiling
   {{eq:recall-ceiling}}.

**The six numbers.**

- **15%** of positions produce MLM's learning signal — a factor of about
  **6.7** against a causal objective, at identical compute.
- **12%** of pretraining tokens are `[MASK]`; **0%** at inference.
- **Up to 15x** difference in tokenized length for the same content across
  languages.
- **~10⁷** — the ratio of an encoder forward pass to a dot product, and the
  reason retrieval is a cascade.
- **$2|V|d$** parameters bought with vocabulary size; **$O(T^2)$** compute
  bought with fertility.
- **$1/\sqrt{d}$** — the spread of cosine similarity between random directions,
  and the reference anisotropy must be measured against.

**The six things people get wrong, and the correction.**

- *"BPE learns morphology."* It learns frequency. Morpheme-like units are a
  by-product of counting, not a goal.
- *"WordPiece merges the most frequent pair."* It merges the highest-PMI pair.
  The denominator is the entire difference from BPE.
- *"SentencePiece is a tokenization algorithm."* It is an input convention —
  raw Unicode, whitespace as an ordinary symbol. It implements BPE *and*
  unigram.
- *"BERT's `[CLS]` is a sentence embedding."* Nothing in the MLM objective
  trains it to summarise. Mean pooling is the better untuned default, and both
  lose to a model trained for the job.
- *"Word2Vec and GloVe are rival paradigms."* {{cite:levy2014}}: both factorise
  a log-transformed function of the same co-occurrence matrix.
- *"Fix anisotropy to fix retrieval."* Anisotropy compresses scores, not
  necessarily rankings. Measure before assuming it is your problem.

**The debugging order for an extraction or retrieval system that underperforms.**

1. **Subword–label alignment.** Off by one presents as a bad model, not as an
   error.
2. **Pooling mask.** The same sentence must give the identical vector in a batch
   of 1 and a batch of 64.
3. **Encoder/index version match.** Vectors from two models are not comparable
   and nothing will tell you.
4. **Recall@k of the retriever**, measured with known-answer queries. This is
   the most underdiagnosed failure in RAG.
5. **Entity-level, not token-level, F1** — if the two disagree sharply the
   errors are boundaries, which want better annotation rather than more data.
6. **Fertility on live traffic** against the fitting corpus. A rising
   byte-fallback rate is drift showing up as cost before it shows up as quality.

Steps 1, 2 and 4 are specific to this part, and each is a bug that produces a
plausible-looking system with no error message anywhere.

**The one disposition to carry forward.** Three times in seven chapters, a
reported advance dissolved when somebody equalised the training budget —
{{cite:levy2015}} for embeddings, {{cite:liu2019roberta}} for encoders, and
{{ch:fm-scaling-laws}} for foundation models. Each was discovered *after*
universal adoption, by a replication study rather than by a new method. The habit
worth taking into {{part:9}} is not skepticism about results; it is the specific
question **what was held fixed?** — asked before the result is believed rather
than after it is deployed.
