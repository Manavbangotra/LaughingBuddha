---
id: part-13-assessment
status: final
---

## How to use this

Four sections. The knowledge check takes about two hours and tells you what to
re-read. The assignment builds a document-understanding system end to end, and —
as in {{part:12}} — the deliverable is the **measurement table and a decision
memo**, because every architectural choice in this part is settled by a number you
either measured or did not. The challenge is open-ended. The interview section is
what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**Fundamentals and resolution**

1. Derive {{eq:translation-equivariance}} from {{eq:convolution}} and state where
   the border breaks it.
2. Compute the receptive field and jump of four VGG-style blocks using
   {{eq:receptive-field}}. Then compute the *effective* field with
   {{eq:erf-worked}}, and say which number the standard design rule should use.
3. Why does the ratio of effective to theoretical receptive field fall as
   $1/\sqrt{n}$? Why is the simulation's uniform-weight assumption the
   conservative choice?
4. {{ch:mm-cv-fundamentals}} measured a dense model at 0.346 on shifted shapes and
   0.893 when trained on all positions. What exactly does the convolutional prior
   supply, and what does it not?
5. State {{eq:resolution-tension}} and name the answer each of chapters 121, 122
   and 126 gives to it.

**Classification and the residual connection**

6. State the degradation problem and explain why {{eq:identity-embedding}} makes
   it surprising. Why is it not overfitting?
7. The measured gradient ratio at depth 32 was 1.8. Why does that make
   "residual connections fix vanishing gradients" an incomplete explanation?
8. Explain {{eq:identity-is-default}} and why a plain stack failed to learn the
   identity even at depth 2 with an adaptive optimiser.
9. Derive {{eq:residual-variance}} and explain why normalisation is structural
   rather than a training aid.
10. At depths 2 and 4 the plain network *beat* the residual one. Explain, and say
    what that implies about adding skips to a shallow model.

**Detection and segmentation**

11. Define {{eq:iou}} and explain why it is a poor regression loss.
12. State {{eq:crowd-ambiguity}} and explain why NMS's assumption is definitionally
    false at a neighbour IoU of 0.36.
13. The best NMS threshold did not move with density; the penalty for missing it
    grew 4.4×. Why is that the more dangerous shape?
14. Prove {{eq:junk-is-free}} and state the condition under which appending a
    prediction *can* reduce AP.
15. Two detectors ranked 0.9557/0.7209 at mAP@0.5 and 0.4180/0.5948 at
    mAP@[.5:.95]. What does that tell you about reporting a single mAP?
16. Explain {{eq:one-to-one-removes-nms}}. What is DETR's actual contribution?
17. Why did overall IoU read 0.881 while boundary IoU read 0.498? Derive the
    dilution ({{eq:boundary-dilution}}) and say what to report instead.
18. Why is a structure thinner than the stride unrecoverable at any depth?
19. Derive {{eq:ce-imbalance}}'s gradient ratio at a 0.3% foreground fraction, and
    explain why the optimiser is succeeding rather than failing.
20. Inverse-frequency weighting measured *worse* than plain cross-entropy at 3%
    foreground. Explain via {{eq:weighting-overshoot}}.

**Transformers, CLIP, and the vision tower**

21. Compute {{eq:patch-compression}} for $p=16$, $C=3$, $d=768$ and explain why
    that configuration is not a coincidence.
22. Attention was 9.3% of FLOPs at 224 px and 68.1% at 1024. What does that do to
    the claim "attention is quadratic"?
23. Using {{eq:halving-patch}}, state the three cost factors when patch size
    halves, and which is unchanged.
24. A convnet scored 0.495, 0.528, 0.503, 0.514 across a 64-fold data increase on
    a positional task. Why is more data not the answer?
25. State {{eq:no-within-modality-term}} and derive why the modality gap is
    unconstrained rather than a training failure.
26. Image-image similarity had a 99th percentile of 0.444 and matched image-text a
    mean of 0.454. What follows for thresholds?
27. Prompt selection was worth 0.021 and prompt ensembling 0.240. Explain the
    $1/\sqrt{T}$ mechanism, and why a shared shift that carries no class
    information still changes predictions.

**Documents, VLMs, retrieval, video**

28. Convert a 99% character error rate into field accuracy for a 40-character
    field, and invert {{eq:required-cer}} for 99% field accuracy.
29. Which OCR errors does a format check catch, and what fraction? What catches
    the rest?
30. Derive {{eq:crossover-inverse-length}} and explain why the same document can
    want two architectures.
31. Reading-order association scored exactly 0.500 on a two-column form. Show why
    the number is exactly one half and independent of the field count.
32. Why is a nearly-correct table worse than a failed one
    ({{eq:table-misalignment-silent}})?
33. A chart value read to 0.0046 gave a difference at 0.089 and a ratio at 0.0066.
    Explain both, and say which questions to refuse.
34. Derive {{eq:legible-times-attended}} and explain why the optimum moved from
    256 to 1024 tokens between content types.
35. A fixed-query connector recovered 0.213 at *four* facts. Why is that a uniform
    tax rather than a capacity cliff, and what does it explain about BLIP-2 versus
    LLaVA?
36. A mixed index returned 83.4% text from a 50/50 corpus. Explain
    {{eq:modality-bias-in-ranking}} and why centring is safe.
37. Caption indexes scored 0.135 on mentioned attributes and 0.006 on omitted
    ones. Why does raising coverage not fix it ({{eq:coverage-tradeoff}})?
38. Derive {{eq:temporal-redundancy}} and compute the ceiling for a 10-minute video
    with $\tau = 4$ s.
39. A 0.5-second event was caught 0.069, 0.140, 0.259, 0.538 as frames doubled.
    Why is that curve not converging, and why does $\tau$ not appear?
40. Compute {{eq:required-video-budget}} for reading a 20-pixel timestamp at a
    1-second event in a 5-minute video. What architecture does the answer imply?

## Practical assignment: a document-understanding system and its decision memo

Build a system that answers questions over a corpus of at least 500 **real**
document pages that you did not generate — scanned invoices, regulatory filings,
academic papers with figures, product manuals. It must include scans or
photographs, not only born-digital PDFs, because the failures in
{{ch:mm-ocr}} and {{ch:mm-layout}} do not occur in clean text.

**Required measurements, before any architecture is chosen.**

1. **Character error rate on your own pages**, by hand-labelling twenty of them.
   Then convert it to field accuracy with
   {{eq:field-error-amplification}} using your actual field lengths.
2. **The smallest feature you must read**, in pixels at your capture resolution,
   and the token count {{eq:tokens-for-feature}} implies.
3. **Caption coverage**: sample thirty images or figures, list the attributes a
   user might search by, and count how many a captioner mentions.
4. **Layout census**: what fraction of your pages are single-column,
   multi-column, forms, or tables? This decides whether
   {{eq:reading-order-loses-2d}} applies to you.

**Required components.**

5. **A parse pipeline producing a text layer**, with per-stage loss measured
   ({{eq:pipeline-composition}}) and the text retained as an artefact.
6. **A layout-aware or 2D-position-preserving path** for whatever fraction of your
   corpus the census says needs it, evaluated on **key–value association**
   separately from value extraction.
7. **A table extractor with arithmetic validation** — per-cell and per-row
   accuracy ({{eq:table-row-accuracy}}), plus a check that columns which should
   sum, sum.
8. **A VLM path** at two token budgets, evaluated against the pipeline on the same
   questions, with cost per page recorded.
9. **A retrieval index** containing both text and page images, with the modality
   mix of results logged, and **per-modality centring implemented and ablated**
   ({{eq:per-modality-centring}}).
10. **Silent-error instrumentation**: for numeric fields, measure the rate of
    corruptions that still parse ({{eq:silent-numeric-error}}), and implement one
    redundancy check that catches them.

**The deliverable is two artefacts.** A table with a row per configuration and
columns for accuracy by question type, cost per page, and latency — with
field-level accuracy reported rather than CER, and boundary/structural accuracy
reported separately from aggregate. And a **two-page decision memo** stating, with
numbers: which architecture you chose for which fraction of the corpus, what
{{eq:ocr-crossover}} said, and which measurement would change your mind.

## Advanced challenge

Pick one.

**Find your resolution floor.** Measure the effective receptive field
({{eq:erf-worked}}) of the vision tower you use, and its patch compression ratio
({{eq:patch-compression}}). Predict the smallest text it can read, then test the
prediction by rendering the same page at descending resolutions until accuracy
collapses. Report the gap between prediction and measurement, and explain it.

**Break your own detector or segmenter.** Measure your scene-density distribution,
then find the crowding level at which your NMS threshold starts deleting real
objects ({{eq:crowd-ambiguity}}) — or the object size at which your segmenter's
boundary IoU falls below half its region IoU. Predict the point first from the
equations, then verify. **A well-argued prediction that misses is a complete
answer if you explain the miss.**

**Quantify the modality gap in a system you use.** Take a real CLIP or SigLIP
model and your own data. Measure the linear separability of the two modalities,
the image-image and image-text similarity distributions, and the modality mix of
your top-20 results. Then implement per-modality centring and re-measure all
three. Report what improved, what did not, and whether any threshold in your
system was being applied across both.

**Price the video question properly.** For a video task you care about, measure
$d$ (the shortest event you must not miss) and $s$ (the smallest thing you must
see), compute $B^*$ from {{eq:required-video-budget}}, and compare it against what
uniform sampling at your budget actually provides. If $B^*$ exceeds your budget,
design and evaluate the temporal cascade — including measuring the cheap stage's
**recall**, which is the ceiling.

## Interview preparation

**"Why can't the model read the small text?"** Patch compression at the tokeniser,
or the input resize. Both are upstream of anything a prompt reaches. A strong
answer computes $\rho = p^2C/d$ and asks what resolution the page was sent at.

**"Is attention the bottleneck in a ViT?"** At benchmark resolution, no — it is
under a tenth of the FLOPs. At document resolution, yes — over two thirds. The
question is incomplete without a resolution.

**"Should we use a transformer or a convnet?"** A data-scale question, not a
correctness one, and it inverts on tasks the convolutional prior forbids. A strong
answer asks how much labelled data exists and whether the task depends on absolute
position.

**"Our detector misses things in busy scenes."** NMS assuming overlap means
duplication. Ask for the neighbour-IoU distribution, and note that the threshold's
*penalty* grows with density even when its optimum does not.

**"Our segmentation IoU is 0.88 and users complain."** Report boundary IoU. The
aggregate is dominated by interior pixels that were never in doubt.

**"Our OCR is 99% accurate."** In what units? Convert to field accuracy for your
field lengths. Then ask which errors survive a format check.

**"Should we drop OCR and use a VLM?"** For which fields? Short structured fields
favour the pipeline; long free-text ones favour the model. And the pipeline leaves
a text layer that is indexable, citable, and cheap to re-embed — usually the
deciding factor.

**"Our image search returns mostly text results."** The modality gap in a ranked
list. Centre per modality, or fuse by rank. A strong answer notices this is a
scale problem rather than a relevance problem.

**"How many frames should we sample?"** Duration divided by the shortest event you
must not miss. If nobody knows that number, that is the work to do first — and for
sustained-evidence tasks the answer is far fewer than people expect.

**"Can the VLM read this chart and tell me the difference between the bars?"**
It can read the values well and the difference badly, because differences of close
quantities are ill-conditioned. Extract the series to a table and compute.
