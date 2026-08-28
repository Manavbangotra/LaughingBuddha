---
id: part-13-intro
status: final
---

## What this part is for

{{part:12}} ended by admitting that much of what an organisation knows is not
text: it is tables inside PDFs and pages that are pictures. This part is the
machinery for that, and it carries a hazard the previous parts did not.

**The hazard is that this part is a history lesson pretending to be a
curriculum.** A reader in 2026 will call a vision-language model for almost
everything in {{ch:mm-classification}} through {{ch:mm-layout}}, and it will work.
Eleven chapters of archaeology would be easy to write and useless to read.

It would also be wrong, because **a VLM's failures are the failures of the
components it absorbed**. It cannot read small print because of
{{ch:mm-vit}}'s {{eq:patch-compression}}. It miscounts because of
{{ch:mm-clip}}'s caption supervision. It gets chart differences wrong because of
{{ch:mm-layout}}'s {{eq:derived-quantity-amplification}}. A reader who has never
seen those components cannot debug them.

> **The rule adopted for this part: teach the older architecture only where its
> failure mode survives into the modern one.** Convolution is taught because
> receptive field still explains what a vision tower can see; anchors get a
> paragraph and NMS gets a section, because NMS's crowd failure is still visible in
> production; the R-CNN lineage gets one table rather than one chapter.

## The organising idea

**Every architecture in this part is a different answer to one question: what is
the unit of visual information, and how many of them can you afford?**

```text
   WHAT IS THE UNIT?           WHAT IT COSTS              WHAT IT CANNOT SEE
   ─────────────────────       ──────────────────         ───────────────────
   118 a pixel neighbourhood   O(1) per pixel, shared     beyond its effective
   119 a feature at a scale    depth, and trainability        receptive field
   120 a box                   NMS, anchors              overlapping instances
   121 a pixel, again          resolution vs context     thin structures
   122 a 16x16 patch           O(n^2) attention          sub-patch detail
   123 a whole image           one vector per image      anything compositional
   126 a visual TOKEN          context window            what got downsampled
   128 a frame, mostly         redundant frames          brief events
```

The through-line, stated in {{ch:mm-cv-fundamentals}} and returned to in
{{ch:mm-video-audio}}: **resolution is the budget, and every architecture here
spends it differently.** Chapter 126's visual-token count is chapter 118's
receptive field with a price attached, and chapter 128 spends the same budget in
two dimensions at once.

## Six things worth knowing before you start

**A deep network sees far less than its specification says.**
{{ch:mm-cv-fundamentals}} measures the effective receptive field growing as
$\sqrt{\text{depth}}$ while the standard formula grows linearly — 21 pixels of
theory against 10.3 of effect at ten layers. The rule everyone applies is applied
to the wrong number, and the gap widens the deeper the backbone.

**A prior is a constraint: it buys sample efficiency inside its scope and costs
everything outside it.** {{ch:mm-vit}} measures both halves — a convnet at 1.000
with 100 examples where a transformer is at chance, and a convnet stuck at chance
forever on a task about *position*, because {{eq:pooling-invariance}} discarded the
information. More data cannot help what the architecture cannot express.

**The aggregate metric hides the failure that matters.**
{{ch:mm-segmentation}} measures overall IoU at 0.881 and boundary IoU at 0.498 on
the same masks; {{ch:mm-detection}} finds mAP barely punishing duplicates or junk.
**In both cases the number everyone reports is dominated by the part of the
problem that was never hard.**

**CLIP's shared space is shared and it is not one region.** {{ch:mm-clip}} shows
the loss contains no term comparing an image to an image, so the modality gap is
unconstrained rather than a defect — and {{ch:mm-multimodal-rag}} measures what
that does to a mixed index: 83.4% of results from one modality in a 50/50 corpus,
fixed by three lines of centring.

**Perception errors are amplified by the arithmetic you do next.**
{{ch:mm-layout}} finds a chart value read to 0.0046 relative error and a
*difference* of two close values at 0.089 — a factor of 19 from the same reading,
because differences are ill-conditioned and ratios are not. **Ask what the
arithmetic does to the error before blaming the model.**

**And almost every recommendation here is a measurement you have not taken.**
Effective receptive field, orphan-boundary share, NMS threshold against your scene
density, character error rate on *your* documents, caption coverage, the shortest
event you must not miss. Each decides an architecture and each takes an afternoon.

## What this part does not cover

Attention and transformer mechanics are {{part:07}}; {{ch:mm-vit}} is a
*tokenisation* chapter and rederives nothing. Contrastive learning's theory is
{{ch:emb-what-they-are}}. Multimodal RAG *architecture* was
{{ch:rag-structured}}; this part covers the retrieval question and deliberately
does not repeat it. Image generation is out of scope entirely — this part is about
perception and grounding. Fine-tuning a vision tower is {{part:14}}.

## How the chapters build

{{ch:mm-cv-fundamentals}} is load-bearing and short on novelty: read it for
{{eq:erf-worked}} and {{eq:resolution-tension}}, which every later chapter spends.
{{ch:mm-classification}} is worth reading even if you will never train a CNN,
because the residual connection is the idea that generalised and the convolution
is the one that did not.

{{ch:mm-detection}} and {{ch:mm-segmentation}} are the two chapters most likely to
be skipped and most likely to explain a production failure — NMS on crowds, and
boundary error hidden by mean IoU. {{ch:mm-vit}} and {{ch:mm-clip}} are the
foundation of everything after, and {{ch:mm-clip}}'s modality gap is the single
finding most likely to save you a week.

{{ch:mm-ocr}} through {{ch:mm-vlms}} are the document stack, which is what most
real multimodal systems actually are. {{ch:mm-vlms}} is the chapter to read first
if you are building today and last if you are learning properly, because most of
what it explains happened earlier.

{{ch:mm-multimodal-rag}} and {{ch:mm-video-audio}} apply all of it, and both end
in the same place: **measure the property of your data that decides the
architecture, before choosing the architecture.**
