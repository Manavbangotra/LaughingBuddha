# Part VI — Deep Learning: research notes

Research pass run 2026-08-14, before writing. **First full-tier part**: 21
sections per chapter, 4,200-word floor, thirteen chapters.

## What changes at full tier

The focused template asks what something is and how to use it. The full
template adds four sections that change the character of the writing:

- **Internal Mechanics** (§7) — what the implementation actually does,
  including the parts libraries hide. For this part that means memory layout,
  the backward pass's storage cost, and where the FLOPs go.
- **Production Considerations** (§10) — numerics, throughput, memory,
  determinism. This is where Part V's discipline gets applied to models that
  cost real money to train.
- **Failure Modes** (§12) — distinct from Common Mistakes. Mistakes are things
  a person does wrong; failure modes are things the method does wrong when
  correctly applied.
- **Research Questions** (§19) — genuinely open problems, labelled honestly.

The risk at this tier is padding. The rule for this part: **every section
either derives something, measures something, or states a decision rule.** If a
section would only restate the chapter, it should be short.

## The organising idea

Part IV built models by choosing a hypothesis space and fitting it. Part VI
builds the hypothesis space itself, out of composable differentiable pieces,
and the whole part is about the consequences of that one change.

The through-line: **a deep network is a composition, and composition is what
makes it both powerful and fragile.** Universal approximation says the class is
rich enough ({{cite:cybenko1989}}); the chain rule says gradients multiply
through the layers, and multiplying many numbers together is numerically
treacherous. Almost everything in chapters 55-58 — initialisation, schedules,
normalisation, residual connections — exists to keep that product well-behaved.
Framing the part that way makes nine apparently unrelated tricks into one
problem with nine attacks on it.

## The genuinely live questions

### 1. Why does batch normalisation work?

Not settled, and the settled-sounding answer is wrong.

{{cite:ioffe2015}} introduced BatchNorm and attributed its benefit to reducing
**internal covariate shift** — the changing distribution of each layer's inputs
as the layers below it update. That explanation is in the paper's title and in
every tutorial since.

{{cite:santurkar2018}} tested it directly and found it does not hold: injecting
deliberate distributional noise *after* BatchNorm — restoring the covariate
shift it was supposed to remove — leaves the training speed-up intact. Their
proposed mechanism is that BatchNorm smooths the loss landscape, improving the
Lipschitz constants of the loss and its gradients, which permits larger stable
learning rates.

**Position for the book:** teach the mechanism as {{maturity:ESTABLISHED}}
(BatchNorm works, reliably), the internal-covariate-shift explanation as
**superseded**, and the smoothing explanation as {{maturity:MATURE}} but not
the final word. This is a good example for {{ch:dl-normalization}} of a
technique whose *effect* is certain and whose *reason* was wrong for three
years — worth teaching as an epistemic lesson, not only a technical one.

### 2. Do we still need normalisation at all?

2026 answer: yes in practice, and the theoretical necessity is now in question.

- **Pre-norm with RMSNorm** {{cite:zhang2019rmsnorm}} is the standard in every
  major open-weight LLM since 2023 — LLaMA, Mistral, Gemma, Qwen, DeepSeek.
  RMSNorm drops the mean-centring of LayerNorm {{cite:ba2016layernorm}},
  which costs little in quality and saves meaningful compute.
- **Dynamic Tanh (DyT, 2025)** replaces the statistical normalisation entirely
  with a scaled `tanh`, reportedly matching across several architectures. No
  production LLM had adopted it as of mid-2026.

**Position:** RMSNorm + pre-norm is {{maturity:ESTABLISHED}} as the default;
normalisation-free approaches are {{maturity:EMERGING}}. The durable content is
*what normalisation does to the gradient scale*, which survives whichever
mechanism wins.

### 3. Are CNNs obsolete?

No, and the honest answer is more useful than either extreme.

- Vision transformers win at scale with large pretraining.
- CNNs win in the limited-data regime without large-scale pretraining, and in
  latency-constrained real-time systems.
- **Hybrids** — a convolutional stem feeding a transformer encoder — are
  reported to give the best accuracy/data-efficiency/cost balance, and are the
  common 2026 choice.
- ConvNeXt (2022) showed that much of the ViT advantage was training recipe
  rather than architecture: a pure CNN modernised with the same recipe matches
  ViTs on ImageNet.

**Position for {{ch:dl-cnns}}:** the *inductive bias* — locality, weight
sharing, translation equivariance — is the durable content and is exactly the
Part IV framing (an architecture is an assumption). The
architecture-versus-recipe confound from ConvNeXt is the honest caveat worth
teaching, because it generalises: **many architectural comparisons are recipe
comparisons in disguise.**

### 4. Are RNNs still worth teaching?

Yes, and the reason changed.

LSTMs {{cite:hochreiter1997}} are legacy for language — transformers replaced
them — but the *recurrent idea* is alive: state space models (S4, Mamba, and
the 2026 successors) revive linear-time sequential state with a different
parameterisation, and hybrid SSM/attention stacks are shipping. The 2026
literature describes the post-transformer question as genuinely open for the
first time in seven years.

**Position for {{ch:dl-rnns}}:** teach the recurrence, the vanishing-gradient
analysis {{cite:bengio1994}}, and the gating solution, then state plainly that
LSTMs are {{maturity:MATURE}} and largely superseded for language while the
sequential-state idea is {{maturity:EMERGING}} again via SSMs. The
vanishing-gradient derivation is the part that transfers.

### 5. Why do overparameterised networks generalise?

Open. {{cite:zhang2017rethinking}} showed that standard architectures can fit
*random labels* perfectly, which means their capacity is effectively unbounded
and classical uniform-convergence bounds — including
{{eq:generalisation-bound}} from {{ch:ml-what-it-is}} — cannot explain why they
generalise on real labels. Explicit regularisation helps but is not what makes
generalisation possible.

**Position:** {{maturity:RESEARCH FRONTIER}}, and this is the honest payoff of
the promise made in {{ch:ml-metrics}} that the classical U-shape stops
describing what happens.

## Structural decisions

**Chapter 53 (backpropagation) is the load-bearing chapter** and should be the
longest. Derive it from the chain rule by hand on a concrete two-layer network,
implement reverse-mode autodiff from scratch with a tape, and verify against
numerical gradients. Everything after it can then be honest about what it costs
in memory and time, because the reader has seen where the stored activations
come from.

**Chapters 49-52 build the forward pass; 53 is the hinge; 54-58 are all about
keeping the gradient well-conditioned.** The part introduction should say so
explicitly, because otherwise chapters 55-58 read as a grab-bag.

**Chapter 56 (initialisation) derives Glorot and He** rather than quoting the
formulae. The variance-propagation argument is four lines and it is what makes
the factor of 2 for ReLU obvious rather than magic.

**Chapter 59 (CNNs) must carry the ConvNeXt caveat** and the honest 2026
position from question 3.

**Chapter 61 (autoencoders) connects to {{ch:ml-pca}}** — a linear autoencoder
with squared error recovers the PCA subspace — and is the natural place for the
representation-learning framing that {{part:11}} needs.

## References checked

All verified 2026-08-14 against Crossref, the publisher's own page, the arXiv
API, or the official proceedings site.

| Key | What | Checked against |
|---|---|---|
| `rosenblatt1958` | The perceptron, Psych. Review 65, 386-408 | Crossref 10.1037/h0042519 |
| `rumelhart1986` | Learning representations by back-propagating errors, Nature 323, 533-536 | Crossref 10.1038/323533a0 |
| `cybenko1989` | Approximation by superpositions of a sigmoidal function, MCSS 2, 303-314 | Crossref 10.1007/BF02551274 |
| `bengio1994` | Learning long-term dependencies is difficult, IEEE TNN 5, 157-166 | Crossref 10.1109/72.279181 |
| `hochreiter1997` | Long Short-Term Memory, Neural Computation 9, 1735-1780 | Crossref 10.1162/neco.1997.9.8.1735 |
| `lecun1998` | Gradient-based learning applied to document recognition, Proc. IEEE 86, 2278-2324 | Crossref 10.1109/5.726791 |
| `hinton2006` | A fast learning algorithm for deep belief nets, Neural Comp. 18, 1527-1554 | Crossref 10.1162/neco.2006.18.7.1527 |
| `glorot2010` | Understanding the difficulty of training deep feedforward nets, AISTATS/PMLR 9, 249-256 | proceedings.mlr.press/v9/glorot10a.html |
| `krizhevsky2012` | ImageNet classification with deep CNNs, CACM 60(6), 84-90 | Crossref 10.1145/3065386 |
| `srivastava2014` | Dropout, JMLR 15, 1929-1958 | jmlr.org/papers/v15/srivastava14a.html |
| `kingma2015adam` | Adam, arXiv 1412.6980 | arXiv API, v1 2014-12-22 |
| `ioffe2015` | Batch Normalization, arXiv 1502.03167 | arXiv API, v1 2015-02-11 |
| `he2015init` | Delving Deep into Rectifiers, ICCV 2015, 1026-1034 | Crossref 10.1109/ICCV.2015.123 |
| `he2016resnet` | Deep Residual Learning, CVPR 2016, 770-778 | Crossref 10.1109/CVPR.2016.90 |
| `ba2016layernorm` | Layer Normalization, arXiv 1607.06450 | arXiv API, v1 2016-07-21 |
| `zhang2017rethinking` | Rethinking generalization, arXiv 1611.03530 | arXiv API, v1 2016-11-10 |
| `santurkar2018` | How Does Batch Normalization Help Optimization?, arXiv 1805.11604 | arXiv API, v1 2018-05-29 |
| `loshchilov2019adamw` | Decoupled Weight Decay Regularization, arXiv 1711.05101 | arXiv API, v1 2017-11-14 |
| `zhang2019rmsnorm` | Root Mean Square Layer Normalization, arXiv 1910.07467 | arXiv API, v1 2019-10-16 |
| `kingma2014vae` | Auto-Encoding Variational Bayes, arXiv 1312.6114 | arXiv API, v1 2013-12-20 |

Notes on what could not be confirmed and is therefore recorded as it stands:

- Several of these were published at conferences (ICLR, NeurIPS, ICML) whose
  proceedings carry no DOI. Where the arXiv record has no `journal_ref`, the
  entry cites the arXiv identifier and the conference is named only where the
  arXiv page itself states it. **No conference/year was added from memory.**
- `krizhevsky2012` is verified against the 2017 *Communications of the ACM*
  reprint, which is the version with a DOI; the original is NIPS 2012. The
  bibliography records the CACM venue because that is what was verified.
- Nair & Hinton (2010), the usual ReLU citation, could not be verified from an
  authoritative machine-readable source in this pass, so **ReLU is discussed
  without a citation** rather than with an unverified one. Glorot et al. (2011)
  on deep sparse rectifiers is in the same position.

## Deliberately omitted

- **Distributed and mixed-precision training.** Named in Production
  Considerations; {{part:23}} owns them.
- **Neural architecture search.** {{part:20}}.
- **Attention and transformers.** {{part:7}}, deliberately — this part ends at
  the point where recurrence's limitations motivate them.
- **Graph neural networks, diffusion, GANs.** Not in this book's TOC; named
  once each where relevant and not developed.
- **Second-order optimisation** beyond noting why it is unaffordable.
- **The neural tangent kernel and mean-field theory.** Named in
  {{ch:dl-initialization}}'s research section; not developed.

## Chapter-level notes

**Ch 49** must do the XOR/Minsky-Papert story properly and connect it to
{{ch:ml-what-it-is}}'s hypothesis-space framing — a perceptron is a linear
model and the entire history follows from that.

**Ch 50** should measure the vanishing-derivative problem directly rather than
asserting it: sigmoid's derivative peaks at 0.25, so a ten-layer sigmoid
network multiplies at most $0.25^{10} \approx 10^{-6}$.

**Ch 51** introduces the computational graph as the data structure, which
Ch 53 then differentiates. The two should be written together.

**Ch 52** connects back to {{ch:ml-logistic}}: cross-entropy is the negative
log-likelihood, and the softmax/cross-entropy gradient simplification is the
same cancellation as {{eq:logit-delta}}.

**Ch 53** derives backprop by hand, implements a tape, and verifies with
finite differences. The measurement to make: memory grows linearly in depth
because activations must be stored, and that is why gradient checkpointing
exists.

**Ch 54** must measure where Adam beats SGD and where it does not, rather than
asserting either. AdamW's decoupling {{cite:loshchilov2019adamw}} is a small
change with a clean derivation.

**Ch 56** derives Glorot and He from variance propagation, and measures
activation variance across depth under each.

**Ch 57** carries the internal-covariate-shift correction from question 1
above, and should reproduce the Santurkar experiment in miniature if feasible.

**Ch 58** carries {{cite:zhang2017rethinking}}: fit random labels and show the
network succeed, which is the honest end of the classical story.

**Ch 60** carries the vanishing-gradient derivation and the honest 2026
position on SSMs from question 4.
