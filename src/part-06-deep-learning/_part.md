---
id: part-06-intro
status: final
---

## What this part is for

Part IV chose a hypothesis space and fitted it. This part **builds the
hypothesis space**, out of small differentiable pieces composed into a deep
stack, and everything that follows is a consequence of that one change.

The composition buys enormous expressive power. {{cite:cybenko1989}} proved
that a single hidden layer of sufficient width can approximate any continuous
function arbitrarily well, so the class is not merely bigger than a linear
model's — it is universal. That result closed a question that had been open
since {{cite:rosenblatt1958}}'s perceptron was shown unable to represent XOR.

The composition also creates one problem, and it is the same problem
everywhere:

> **The chain rule turns a deep network's gradient into a long product, and
> long products of numbers are numerically treacherous.** Slightly less than
> one, multiplied fifty times, is $10^{-9}$. Slightly more than one is
> $10^{9}$.

That single sentence organises nine chapters. Activations, initialisation,
learning-rate schedules, normalisation and residual connections are not five
unrelated tricks — they are five attacks on the conditioning of one product.
Reading them that way is the difference between memorising a recipe and being
able to debug a network that will not train.

```text
   FORWARD                    THE HINGE                 KEEPING IT STABLE
   ───────────────────        ─────────────────         ─────────────────────
   49 perceptron              53 backpropagation        54 optimisers
   50 activations                                       55 schedules
   51 forward / graphs         everything before        56 initialisation
   52 losses                   builds the graph;        57 normalisation
                               everything after         58 regularisation
                               keeps its gradient
                               well-conditioned        ARCHITECTURES
                                                        59 convolutional
                                                        60 recurrent
                                                        61 autoencoders
```

## The three chapters that carry the most weight

**{{ch:dl-backprop}}** is the load-bearing chapter of the book so far.
Backpropagation {{cite:rumelhart1986}} is derived by hand on a concrete
network, then implemented as reverse-mode automatic differentiation with a
tape, then verified against numerical gradients. Every framework in the rest of
the book is this algorithm with better engineering, and the reason to build it
once is that afterwards you can reason about what training *costs* — where the
memory goes, why activations must be stored, why depth is expensive.

**{{ch:dl-initialization}}** derives {{cite:glorot2010}} and
{{cite:he2015init}} from a four-line variance calculation rather than quoting
the formulae. Once you have done it, the factor of two for rectified units is
obvious rather than magic, and the whole signal-propagation framing that
normalisation and residual connections live inside becomes available.

**{{ch:dl-normalization}}** carries the part's most useful epistemic lesson.
{{cite:ioffe2015}} introduced batch normalisation and attributed its success to
reducing *internal covariate shift* — an explanation repeated in the paper's
title and in nearly every tutorial since. {{cite:santurkar2018}} tested it
directly and found it does not hold. The technique works; the stated reason was
wrong for three years. **A method working is not evidence that the explanation
for why it works is correct**, and this part will say so where it applies.

## What is genuinely unsettled

Three things, labelled honestly where they arise:

**Why overparameterised networks generalise.**
{{cite:zhang2017rethinking}} showed that standard architectures trained with
standard methods can fit *randomly labelled* data perfectly. Their capacity is
therefore sufficient to memorise the training set outright, and the
uniform-convergence bound of {{eq:generalisation-bound}} cannot explain why
they nonetheless generalise on real labels. Regularisation helps and is not
what makes generalisation possible. {{maturity:RESEARCH FRONTIER}}, and it is
the honest payoff of the warning in {{ch:ml-metrics}} that the classical
U-shape stops describing what happens.

**Whether normalisation is necessary at all.** RMSNorm with pre-normalisation
{{cite:zhang2019rmsnorm}} is the {{maturity:ESTABLISHED}} default in every
major open-weight language model since 2023. Normalisation-free approaches that
replace the statistics with a fixed nonlinearity are {{maturity:EMERGING}} and,
as of 2026, unused in production models.

**What comes after the transformer.** {{ch:dl-rnns}} ends by noting that the
recurrent idea — a linear-time sequential state — has returned via state space
models, and that the post-transformer question is genuinely open for the first
time in seven years. The vanishing-gradient analysis of {{cite:bengio1994}} is
what transfers; the LSTM is largely legacy for language.

## Two things worth saying up front

**An architecture is an assumption**, exactly as in Part IV. A convolution
assumes locality and translation equivariance {{cite:lecun1998}}; a recurrence
assumes that a fixed-size state can summarise the past. Choosing an
architecture is choosing a prior, and the reason convolutions beat fully
connected layers on images is not that they are more powerful — they are
strictly less powerful — but that they are less powerful *in the right way*.

**Most of what people call "deep learning" is engineering, not theory.** The
components of {{cite:krizhevsky2012}} were largely known beforehand; what was
new was the combination of scale, hardware and a dataset large enough for the
capacity to pay. The chapters that follow are unusually full of measurements
for that reason: on this material, what works is an empirical question and the
literature has been wrong about *why* more than once.

## What this part deliberately does not cover

Attention and transformers, which are {{part:7}} — this part ends exactly where
recurrence's limitations motivate them. Distributed and mixed-precision
training, which is {{part:23}}, though numerics appear in every Production
Considerations section. Neural architecture search, which is {{part:20}}.
Graph networks, GANs and diffusion, which are named once each where relevant.
Second-order optimisation beyond an explanation of why it is unaffordable. The
neural tangent kernel, named in {{ch:dl-initialization}}'s research section and
not developed.

## What you should be able to do at the end

Derive backpropagation from the chain rule and implement reverse-mode autodiff
with a tape. Say where a training run's memory goes and why gradient
checkpointing helps. Derive the Glorot and He initialisation scales, and
predict what happens to activation variance under each. Explain what
normalisation does to the gradient, and state what is and is not known about
why it helps. Diagnose a network that will not train from its gradient norms.
Choose between SGD and Adam with a reason. Explain the convolution's inductive
bias, and say when it is the wrong one. Derive why gradients vanish through a
recurrence, and explain how gating and residual connections solve the same
problem in two settings.

The assignment at the end asks for a network built from scratch — autodiff
included — trained to a target accuracy, with the diagnostics to prove each
component earns its place.

## A note on the chapters themselves

This is the book's first **full-tier** part: twenty-one sections per chapter
rather than twelve. Four of the added sections change the character of the
writing, and are worth knowing about in advance.

**Internal Mechanics** describes what an implementation actually does,
including what libraries hide. **Production Considerations** covers numerics,
throughput and memory — Part V's discipline applied to models that cost real
money to train. **Failure Modes** is distinct from Common Mistakes: a mistake
is something a person does wrong, a failure mode is something the method does
wrong when applied correctly. **Research Questions** names open problems and
labels them as open.

The chapters are roughly twice the length of Part V's. The rule they were
written to is that every section must derive something, measure something, or
state a decision rule.
