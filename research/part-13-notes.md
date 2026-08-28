# Part XIII — Multimodal AI: research notes

Research pass run 2026-08-28, before writing. Full tier: 21 sections per chapter,
4,200-word floor, eleven chapters. Twenty-four new bibliography entries, each
verified against an arXiv abstract page on the date above. 219 entries total,
none unverified.

## What this part is, and what it is not

{{part:12}} ended by admitting that a great deal of enterprise knowledge is not
text: it is tables inside PDFs and pages that are pictures. This part is the
machinery for that, and it has a hazard the previous parts did not.

**The hazard is that this part is a history lesson pretending to be a
curriculum.** A reader in 2026 will use a VLM for almost everything in chapters
118–121, and it would be easy to write eleven chapters of archaeology. It would
also be wrong, because the VLM's failures are exactly the failures of the
components it absorbed — resolution, reading order, small objects, class
imbalance — and a reader who has never seen those components cannot debug them.

> **The rule adopted for this part: teach the older architecture only where its
> failure mode survives into the modern one.** Concretely — convolution is taught
> because receptive field still explains what a vision tower can and cannot see;
> anchors are taught briefly and NMS at length, because NMS failure on crowded
> scenes is still visible in production; U-Net is taught because the
> resolution-versus-context tension is permanent; and the R-CNN lineage gets one
> section rather than one chapter.

## The organising idea

**Every architecture in this part is a different answer to one question: what is
the unit of visual information, and how many of them can you afford?**

A convolution says the unit is a local patch and you can afford as many as the
image has pixels, because weights are shared. A ViT says the unit is a 16×16
patch and you can afford $O(n^2)$ attention over a few hundred of them. A VLM
says the unit is a visual token entering a language model's context, and you can
afford a few hundred to a few thousand — which is the entire reason document
understanding was hard until dynamic resolution
({{cite:wang2024qwen2vl}}).

```text
   WHAT IS THE UNIT?           WHAT IT COSTS              WHAT IT CANNOT SEE
   ─────────────────────       ──────────────────         ───────────────────
   118 a pixel neighbourhood   O(1) per pixel, shared     beyond its receptive
   119 a feature at a scale    depth, and gradients           field
   120 a box                   NMS, anchors              overlapping instances
   121 a pixel, again          resolution vs context     thin structures
   122 a 16x16 patch           O(n^2) attention          sub-patch detail
   123 a whole image           one vector per image      anything compositional
   126 a visual TOKEN          context window            what got downsampled
   128 a frame, mostly         redundant frames          long-range time
```

The through-line to state in {{ch:mm-cv-fundamentals}} and return to in
{{ch:mm-video-audio}}: **resolution is the budget, and every architecture in this
part spends it differently.** Chapter 126's visual-token count is chapter 118's
receptive field with a different name and a price attached.

## The genuinely live questions

### 1. Is classical computer vision obsolete for a practitioner?

Mostly yes for *building*, emphatically no for *debugging*. The honest position:
a general VLM now beats a task-specific model on most tasks a reader will meet,
and the tasks where it does not — small objects at high resolution, precise
masks, real-time embedded detection, tight latency budgets — are exactly the
tasks where the classical architecture is still shipped.

**State the crossover rather than a preference**, and note that the crossover
moves. {{ch:mm-detection}} and {{ch:mm-segmentation}} should each end with the
conditions under which the specialist model is still correct.

### 2. Does CLIP's shared space actually let you compare images to text?

This is the most consequential misunderstanding in multimodal retrieval and it
deserves a measured demonstration rather than a caveat. Image embeddings and text
embeddings occupy **separate cones** in the shared space — the modality gap — so
a naive cross-modal threshold is meaningless and image-image similarity is not
comparable to image-text similarity. Contrastive training aligns *rankings*
across modalities, not positions.

{{ch:mm-clip}} should measure the gap directly and {{ch:mm-multimodal-rag}} should
show what it does to a mixed-modality index. This is {{ch:emb-similarity}}'s
"the score is a rank, not a measurement" arriving with a second modality.

### 3. Should documents be parsed or looked at?

{{ch:rag-structured}} already made the ColPali argument. Here the question is
sharper because both options are fully available: OCR-then-layout-then-text
({{cite:huang2022layoutlmv3}}) against OCR-free end-to-end
({{cite:kim2022donut}}) against a general VLM reading the page
({{cite:wang2024qwen2vl}}).

**The deciding variable is not accuracy, it is what you need out.** A pipeline
that emits text gives you something to index, cite, and diff; a VLM that answers
questions gives you an answer and no artefact. {{ch:mm-ocr}} should say this
plainly, because benchmark tables do not capture it.

### 4. Why is resolution the recurring villain?

Because every architecture in the part reduces it early and cannot recover what
it dropped. A 224×224 vision tower cannot read 8-point text no matter how large
the language model behind it is, and this single fact explains most surprising
VLM failures — dense documents, small chart labels, distant objects.

**{{ch:mm-vlms}} should measure the token-count/accuracy curve** rather than
asserting it, and {{cite:wang2024qwen2vl}}'s dynamic resolution is the fix to
explain.

### 5. Is video a modality or a cost problem?

Mostly a cost problem, and {{cite:tong2022videomae}}'s 90–95% masking ratio is
the quantitative statement of why: adjacent frames carry almost no independent
information. **The design question in video is never "how do I model time", it is
"which frames do I keep"** — and the honest answer for most production systems is
"remarkably few".

## Per-chapter findings

### 118 — Computer Vision Fundamentals

Not a CNN chapter. The content is: an image as a tensor, why a fully connected
layer is the wrong prior (parameter count *and* translation sensitivity), the
convolution as weight sharing plus locality, receptive field arithmetic, and the
equivariance/invariance distinction that pooling trades between.

**Listing:** receptive field growth measured against depth, dilation, and
stride — the number that predicts what a backbone can see, and the one readers
compute wrong.

### 119 — Image Classification and the ResNet Lineage

The degradation problem is the point, and it is *not* vanishing gradients — a
deeper plain net is worse on TRAINING error, which no optimisation-difficulty
story alone explains. The residual connection makes the identity the default
rather than something to learn.

**Listing:** plain versus residual at increasing depth, showing degradation
appear in training loss, then gradient norms by layer to show what the skip
changes. Include {{cite:liu2022convnext}}'s finding that much of the
transformer-versus-convnet gap was training recipe.

### 120 — Object Detection

Content: the set-prediction problem, IoU, anchors briefly, NMS at length, mAP
computed properly, and the one-stage/two-stage/set-prediction trichotomy.

**The thing to demonstrate rather than assert:** NMS is a hand-designed piece of
the *loss surface* smuggled into inference, and it fails predictably on crowded
scenes. That failure is why {{cite:carion2020detr}} exists, and it is measurable
in twenty lines.

**Listing:** NMS threshold swept against crowd density, with mAP implemented from
scratch so the reader sees what the metric actually rewards.

### 121 — Segmentation and SAM

Content: semantic/instance/panoptic, the U-Net resolution-versus-context tension
({{cite:ronneberger2015unet}}), why cross-entropy fails under class imbalance and
what Dice does about it, and promptable segmentation
({{cite:kirillov2023sam}}, {{cite:ravi2024sam2}}).

**Listing:** skip connections ablated, measuring boundary accuracy separately
from region accuracy — because the skip's whole contribution is at boundaries and
a mean-IoU number hides it.

### 122 — Vision Transformers

Content: patches as tokens, why position embeddings are needed at all, the
inductive-bias-versus-data trade, and the quadratic cost that sets patch size.

**The measurement that matters:** the CNN/ViT sample-efficiency crossover.
{{cite:dosovitskiy2021vit}}'s actual claim is conditional on data scale, and it
is routinely quoted unconditionally.

### 123 — CLIP

{{cite:radford2021clip}} and {{cite:zhai2023siglip}}. Content: the contrastive
objective over pairs, zero-shot classification as retrieval against prompts,
prompt sensitivity, and the modality gap per live question 2.
{{cite:oquab2023dinov2}} is the counterweight — language supervision is not
required for strong features and is not best for dense tasks.

**Listing:** train a tiny two-tower model, measure the modality gap, and show
what happens to a cross-modal threshold.

### 124 — OCR and Document AI

The pipeline, its error propagation, and reading order. Per live question 3.
{{cite:huang2022layoutlmv3}}, {{cite:kim2022donut}},
{{cite:mathew2021docvqa}}.

**Listing:** OCR character error rate propagated to downstream extraction
accuracy — the amplification factor is the number nobody computes, and it decides
whether to improve the OCR or replace the pipeline.

### 125 — Layout, Tables, and Charts

2D position as a first-class input, table structure recovery (the hardest and
least glamorous problem in the part), and chart reading where perception and
arithmetic cannot be separated ({{cite:masry2022chartqa}}).

**Listing:** 1D reading-order position versus 2D coordinates on a form-like
task.

### 126 — Vision-Language Models

{{cite:alayrac2022flamingo}}, {{cite:li2023blip2}}, {{cite:liu2023llava}},
{{cite:wang2024qwen2vl}}. Content: the frozen-tower-plus-connector architecture,
connector designs and what they cost, the visual token budget, and resolution.

**Listing:** visual token count against both accuracy and cost, showing the
interior optimum and the point where a fixed grid stops being able to represent
the content at all.

### 127 — Multimodal Embeddings and Retrieval

The retrieval question, building on {{ch:rag-structured}}. Caption-then-embed
versus joint embedding, the modality gap in a mixed index, and
{{cite:faysse2025colpali}}.

**Listing:** the modality gap's effect on a mixed-modality index, and the cheap
fix (per-modality centring), which is {{ch:emb-what-they-are}}'s anisotropy
correction again.

### 128 — Video, Audio, and Spatial Reasoning

Per live question 5. {{cite:tong2022videomae}} for redundancy,
{{cite:radford2022whisper}} for audio, {{cite:ravi2024sam2}} for temporal
propagation.

**Listing:** frame sampling rate against task accuracy and token cost, showing
how flat the accuracy curve is and how steep the cost curve is.

## Cross-part bookkeeping

- **Do not** re-teach attention, transformers, or training dynamics —
  {{part:06}} and {{part:07}} own them. ViT is a *tokenisation* chapter here.
- Contrastive learning's theory is {{ch:emb-what-they-are}}; this part applies it
  across modalities and does not rederive InfoNCE.
- Multimodal *retrieval* is here; multimodal RAG *architecture* was
  {{ch:rag-structured}}, and the two must not overlap.
- Generation of images is not in this part and is not in this book's scope beyond
  a pointer; the part is about perception and grounding.
- Terminology collision check before writing: `patch`, `token`, `resolution`,
  `feature map`, `embedding`, `mask` — `token` and `embedding` certainly collide
  with Parts VII and XI and must be disambiguated on first use.
- Reuse, do not restate: {{eq:chunk-dilution}}, {{eq:recall-ceiling}},
  {{eq:max-distractor}}, {{eq:table-recoverability}}, {{eq:u-shape}}.
