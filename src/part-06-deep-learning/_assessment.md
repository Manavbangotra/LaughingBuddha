---
id: part-06-assessment
status: final
---

## How to use this

Four sections, in increasing order of commitment. The knowledge check should
take an hour and tells you whether to re-read anything. The assignment is the
substantial piece of work and it is the one that will teach you the most. The
challenge is optional and open-ended. The interview preparation is what to
rehearse if you are being examined on this material.

Answers to the knowledge check are not provided, and that is deliberate: every
question is answerable from the chapters, and looking the answer up is the
exercise.

## Knowledge check

**The gradient product**

1. {{eq:unrolled-backprop}} is a product of $L$ matrices. State the three
   distinct techniques in this part that address it, and say what each one does
   to the product.

2. Chapter 53 measured a 20-layer network at a small initialisation whose
   forward signal and error signal each moved by twelve orders of magnitude,
   while the *weight gradient* profile was flat. Explain why the two decays did
   not show up as a tilted weight-gradient profile.

3. Why is the vanishing-gradient problem worse in a recurrent network than in a
   feedforward network of the same depth? Your answer must not simply be
   "because sequences are long".

4. Exploding gradients are easier to fix than vanishing ones. Give the reason
   in one sentence.

**Activations, losses and the output layer**

5. Chapter 50 measured $\E[\relu(z)^2]/\E[z^2] = 0.5$ exactly at every input
   scale. Name two places in the rest of the part where that number appears.

6. Derive $\nabla_{\vec{z}}\ell = \hat{\vec{p}} - \vec{y}$ for softmax with
   cross-entropy, and say what cancels.

7. A model assigns probability 0.001 to the correct class. Give the gradient
   magnitude under sigmoid-with-cross-entropy and under sigmoid-with-squared-
   error, and say which regime this is.

8. Both unfused cross-entropy implementations failed at a logit magnitude of
   800 in Chapter 52's measurement, for *different* reasons. Give both.

**Optimisation**

9. Chapter 54 measured Adam's step *halving* on a gradient spike a thousand
   times the normal size. Explain why it shrank rather than grew.

10. Raising momentum from 0.9 to 0.99 at a fixed learning rate is equivalent to
    what other change? Give the factor.

11. Why is $\lambda$ not transferable between Adam-with-$\ell_2$ and AdamW?

12. {{eq:sgd-convergence}} has two terms that pull in opposite directions.
    State what each wants from $\eta$ and what follows.

**Initialisation and normalisation**

13. Where does the factor of two in $\sqrt{2/n}$ come from? Answer in one
    sentence, then say why it is wrong for tanh.

14. Chapter 56 measured the dead-unit fraction as *identical to every digit*
    across right and wrong initialisation scales. Explain why, and say what
    that implies about using it as a diagnostic.

15. Chapter 57 measured the gradient as exactly orthogonal to $\mat{W}$ for a
    layer feeding a normalisation. State the consequence for $\|\mat{W}\|$ over
    training, and the consequence for the effective learning rate.

16. What did {{cite:santurkar2018}} do, and what did it establish about
    {{cite:ioffe2015}}'s stated explanation?

17. RMSNorm and LayerNorm were measured to agree on zero-mean data with a
    discrepancy shrinking as $1/\sqrt{d}$, and to disagree by a constant amount
    on offset data. What does RMSNorm's success in practice therefore imply
    about trained networks?

**Regularisation and generalisation**

18. {{cite:zhang2017rethinking}}'s result is a reductio. State the premise, the
    contradiction, and the conclusion.

19. Chapter 58 measured the double-descent peak at roughly one parameter per
    training example. What is that point called, and what happens on either
    side of it?

20. Chapter 58's measured comparison across data sizes had a headline result
    that was not about any regulariser. What was it?

**Architectures**

21. A convolution is strictly *less* expressive than a fully connected layer.
    Explain why that is the reason it wins on images, not a cost of using it.

22. Chapter 59 measured a fixed random permutation of the pixels as invisible
    to a dense network and damaging to a convolution. What does the size of
    that damage measure?

23. Distinguish equivariance from invariance and say where each belongs in a
    classification network.

24. {{eq:residual-expansion}}'s first term is the identity. Say why that
    single fact solves the degradation problem.

25. Chapter 59 constructed a deeper network that represents a shallower one
    exactly. What does that construction prove about the degradation problem?

26. Give the work and the critical-path length for a recurrence and for
    attention, and say which one decided the outcome.

27. The LSTM's carry Jacobian is $\diag(\vec{f}_t)$. Name the two things that
    are *absent* from it and why each matters.

28. Why is a positive forget-gate bias standard? Frame your answer as a
    chicken-and-egg problem.

**Autoencoders**

29. A linear autoencoder recovers PCA's subspace but not its components.
    Explain the degeneracy that causes this.

30. Explain posterior collapse, and say why calling it a "training failure" is
    wrong.

31. Chapter 61 measured reconstruction-based anomaly detection failing almost
    completely on one class of anomaly. Which one, and what does that tell you
    about what reconstruction error actually measures?

32. Why does the denoising variant not need a bottleneck, and what modern
    technique is the same move?

## Practical assignment

**Build a deep network from scratch, with automatic differentiation, and
demonstrate that each component earns its place.**

No frameworks. NumPy only. This is the assignment the part was written for and
it is worth several days.

### Part A — the engine

Implement reverse-mode automatic differentiation with a tape, following
{{ch:dl-backprop}}. Required operations: matrix multiply, add with
broadcasting, elementwise multiply, ReLU, tanh, sigmoid, exp, log, sum, mean,
reshape, and a fused softmax–cross-entropy.

**Acceptance criterion.** Every operation passes a central-difference gradient
check at relative error below $10^{-7}$ in float64, on inputs that include at
least one tensor with two consumers and at least one broadcast. Report the
worst relative error per operation in a table.

If your two-consumer test passes with `=` instead of `+=`, your test is not
exercising the case. Fix the test.

### Part B — the network

Build a configurable multi-layer network on top of Part A, with:

- a choice of activation (ReLU, tanh, GELU);
- a choice of initialisation (LeCun, Glorot, He, orthogonal, and a
  deliberately-wrong scale);
- optional layer normalisation, placed either before or after the activation;
- optional residual connections, with the branch's last layer zero-initialised
  or not;
- dropout, weight decay, and a configurable learning-rate schedule.

**Acceptance criterion.** The network reaches near-zero loss on ten examples
(the overfit-a-batch test from {{ch:dl-backprop}}), and its loss at
initialisation is within 5% of $\log C$.

### Part C — the demonstrations

Pick a dataset with genuine structure — the shape-classification generator in
{{ch:dl-cnns}}'s listings is adequate, and a real one is better. For each
component below, produce a measurement that shows what it does *at a depth
where it matters* and, where relevant, at a depth where it does not.

1. **Initialisation.** Per-layer activation variance and gradient norm at
   initialisation, for at least four schemes, at depths 2 and 20. Reproduce
   the widening spread.

2. **Normalisation.** The same four initialisation schemes with and without
   normalisation, at depth 20. Report the *spread*, not the best value.

3. **Residual connections.** Gradient norm at each layer, plain against
   residual, at three depths. Confirm the ratio stays near 1 for the residual
   stack.

4. **Optimiser.** SGD, momentum and Adam, each with its own learning-rate
   search. Report the best rate for each and the shape of the whole grid.

5. **Schedule.** Constant against cosine at a budget long enough for the
   constant schedule to reach its noise floor. If it has not, your budget is
   too short and the comparison is measuring the wrong thing.

6. **Regularisation.** Your chosen recipe at two training-set sizes differing
   by at least a factor of twenty. Report the train/test gap, not only the
   test loss.

### Part D — the report

Write it up. For each component, state:

- what you expected, with the equation you expected it from;
- what you measured;
- whether they agree, and if not, what you did about it.

**The third bullet is the point of the assignment.** Several of these
measurements will disagree with the textbook expectation — most of the
corrections in this part's chapters came from exactly that. A report in which
everything agreed is a report whose experiments were not sharp enough to
disagree.

**Acceptance criterion.** Every claim in the report is backed by a number your
code produced, and every number appears alongside the equation that predicted
it.

## Advanced challenge

Pick one. Each is a genuine open-ended piece of work.

**Reproduce Santurkar et al. properly.** {{ch:dl-normalization}}'s
reproduction is small-scale and its result should be read as gesturing at the
shape of the argument. Do it at a scale where the answer is trustworthy: a real
dataset, a real convolutional network, and an injection magnitude chosen
carefully enough that you can distinguish "covariate shift restored" from
"signal destroyed". Report the injection scale at which the two regimes
separate, and be prepared for the answer to be that you cannot cleanly separate
them.

**Find the edge of stability.** There is evidence that training operates with
the largest Hessian eigenvalue hovering at $2/\eta$ rather than safely below
it — so the learning rate selects a curvature rather than the reverse. Measure
the largest Hessian eigenvalue during training (power iteration on
Hessian-vector products, which cost one extra backward pass each) and plot it
against $2/\eta$. This is a real research finding and reproducing it is
achievable on a laptop.

**Implement a parallel scan.** {{ch:dl-rnns}} argues that a *linear* recurrence
is associative and can therefore be computed in $O(\log T)$ depth. Implement
it, verify it against the sequential version to floating point, measure the
wall-clock difference at several sequence lengths, and then implement a
gated linear recurrence and train it on the copy task. You will have built the
core of a state space model.

**Measure the effective receptive field properly.** {{ch:dl-cnns}}'s
measurement uses random weights. Do it on a *trained* network: backpropagate
from a single output unit to the input and measure where the gradient actually
is. Compare against the theoretical field and against the random-weight
version, and see whether training concentrates or spreads the influence.

**Break your own anomaly detector.** {{ch:dl-autoencoders}} measured
reconstruction-based detection failing on anomalies that lie on the learned
manifold. Take a detector you have built, construct the on-manifold failure
case for it deliberately, and then build a detector that catches both that and
the off-manifold case. Report what it costs.

## Interview preparation

**The eight derivations to be able to do without notes.** Each has appeared in
an interview, and each is a five-minute whiteboard exercise.

1. Backpropagation's four equations from the chain rule
   ({{ch:dl-backprop}}).
2. The softmax cross-entropy gradient, including the cancellation
   ({{ch:dl-losses}}).
3. He initialisation from the variance of a sum
   ({{ch:dl-initialization}}).
4. Adam's bias correction and why the uncorrected step is too *large*
   ({{ch:dl-optimizers}}).
5. Why squared error on a sigmoid saturates ({{ch:dl-losses}}).
6. The receptive field of a convolutional stack ({{ch:dl-cnns}}).
7. The LSTM carry path and what is absent from it ({{ch:dl-rnns}}).
8. The ELBO from Jensen's inequality ({{ch:dl-autoencoders}}).

**The five numbers to have ready.**

- A training step costs about **three** forward passes, and why: two backward
  matmuls per forward one.
- Adam needs **16 bytes per parameter** in fp32, so a 7B model needs 112 GB
  before a single activation.
- Sigmoid's maximum derivative is **0.25**, so ten layers attenuate by
  $10^{-6}$.
- Gradient checkpointing: **$\sqrt{L}$ memory for about a third more compute.**
- Cosine spends exactly **half** its budget above half the peak rate.

**The seven diagnostics, in the order you would run them.**

1. Loss at initialisation against $\log C$. One forward pass.
2. Overfit ten examples. Seconds.
3. Per-layer activation variance at initialisation.
4. Per-layer gradient norm at initialisation.
5. Update-to-weight ratio per layer during training (target $\approx 10^{-3}$).
6. Gradient clip rate.
7. A gradient check on anything hand-written.

Being able to give this list in order is a strong signal, because it is the
list of someone who has debugged a network rather than read about one.

**The four "it depends" questions and what the answer depends on.**

- *Adam or SGD?* Architecture and tuning budget. AdamW for transformers, SGD
  with momentum for convolutional vision if you can tune it. Say the
  generalisation-gap claim is setting-specific.
- *BatchNorm or LayerNorm?* Batch size and inference pattern.
- *Convolutions or transformers?* Data scale. Say that modernised
  convolutional networks closed much of the gap with the transformer's training
  recipe.
- *How much regularisation?* The train/test gap, and whether more data is
  available — which the measurement says beats every technique.

**The three places to say "the field was wrong about this."** Being able to name
them, precisely and without overclaiming, is worth more than any single
derivation.

- **Internal covariate shift.** {{cite:ioffe2015}}'s explanation for batch
  normalisation was tested by {{cite:santurkar2018}} and did not survive. The
  technique is established; the explanation is not.
- **Adam's convergence proof.** The original was flawed; the proposed fix
  (AMSGrad) is not used and Adam is.
- **Weight decay in adaptive optimisers.** {{cite:loshchilov2019adamw}} showed
  that what everyone was writing was not what they thought they were writing,
  and that it had been true for years.

The pattern is worth naming out loud in an interview: **in this field, a method
working is not evidence that the stated reason for it working is correct.** That
is the single most useful disposition this part can leave you with.
