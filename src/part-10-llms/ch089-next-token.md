---
id: llm-next-token
number: 89
part: X
tier: full
status: draft
requires: [llm-anatomy, fm-pretraining, dl-losses, ml-metrics, math-probability,
           math-random-vars]
provides: [logit-semantics, calibration, reliability-diagram, expected-calibration-error,
           confidence-versus-accuracy, entropy-as-uncertainty, perplexity-revisited,
           token-probability, sequence-probability]
citations: [radford2019, brown2020, hoffmann2022chinchilla, ji2023survey,
            ouyang2022, holtzman2020, guo2017calibration, kadavath2022]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what a logit is and what its softmax means, precisely enough to say
   what it does *not* mean.
2. Define calibration and compute expected calibration error from scratch.
3. Build a reliability diagram and read a model's over- or under-confidence off
   it.
4. Explain why cross-entropy training encourages calibration and why alignment
   degrades it.
5. Compute sequence probability and explain why it favours short sequences.
6. Distinguish aleatoric from epistemic uncertainty in a language model's output
   and say which the softmax can express.
7. Use entropy as a runtime uncertainty signal, and state its limits.

## 2. Why This Matters

**Every number this part manipulates comes from here.** Temperature scales
logits, top-p truncates the distribution, hallucination detection thresholds
confidence, and routing decides difficulty. All four assume the probabilities
mean something. This chapter asks whether they do.

**The answer is: partially, and less after alignment.** A pretrained model's
next-token probabilities are reasonably calibrated — a token predicted at 0.7
occurs about 70% of the time — because that is precisely what cross-entropy
minimisation optimises for. **RLHF systematically degrades this**, because
{{eq:rlhf-objective}} rewards preferred outputs rather than accurate
probabilities. The most useful model is the least calibrated one, which is an
uncomfortable and load-bearing fact.

**Confidence is the only uncertainty signal available at inference.** There is
no separate "am I sure" output. If you want to know whether to trust a
generation, whether to route to a larger model, or whether to abstain, the
probabilities are what you have — and knowing how much they are worth decides
how much machinery you build on them.

**And it sets up hallucination properly.** {{ch:llm-hallucination}} is much
easier to reason about once you can distinguish a model that is confidently
wrong from one that is uncertain and unlucky. Those have different causes and
different fixes, and the difference is measurable here.

## 3. Prerequisites

{{ch:llm-anatomy}} for the logit vector and {{eq:next-token-distribution}}.
{{ch:fm-pretraining}} for cross-entropy training and perplexity.
{{ch:dl-losses}} for cross-entropy as a proper scoring rule.
{{ch:ml-metrics}} for the general discipline of interpreting a model's outputs.
{{ch:math-probability}} and {{ch:math-random-vars}} for expectation and
entropy.

## 4. Intuitive Explanation

The model hands you 128,000 numbers. Softmax them and they sum to one. It is
tempting to read the result as the model's belief, and mostly wrong to do so
without qualification.

**What the number actually is.** During training the model was penalised by
$-\log P(\text{actual next token})$ ({{ch:fm-pretraining}}). To minimise that
over a corpus, the best strategy is to output the true conditional frequency of
each token given the context. So a well-trained model's 0.7 means *in contexts
like this one, this token followed about 70% of the time in the training
distribution.*

That is a claim about the corpus, not about the world, and the gap between them
is where a great deal of trouble lives.

**Calibration is the testable version.** Collect every token the model predicted
at roughly 0.7 confidence; check how often it was right. If the answer is 70%,
the model is calibrated in that bin. Do it for every bin and you have a
reliability diagram — the single most informative plot about a model's
probabilities.

> NOTE: Calibration is not accuracy. A model that outputs 0.5 for everything and
> is right half the time is perfectly calibrated and useless. A model that is
> right 95% of the time but always says 0.99 is highly accurate and badly
> calibrated. **They are independent axes and both matter**, for different
> decisions.

**Two kinds of uncertainty, and the softmax expresses one.** When the next token
is genuinely unpredictable — the continuation of "My favourite colour is" —
uncertainty is *aleatoric*, inherent in the task, and a high-entropy
distribution is the correct answer. When the model does not know something —
the birth year of an obscure person — uncertainty is *epistemic*, and the right
answer is "I don't know", which is not a token. **The softmax cannot
distinguish these**, and it will express epistemic ignorance as a confident
guess whenever the corpus contained confident guesses.

**Alignment makes it worse, predictably.** RLHF optimises for human preference,
and humans prefer confident answers. Nothing in {{eq:rlhf-objective}} rewards
saying 0.6 when you mean 0.6. The result is a model that is more useful and less
honest about its uncertainty than the base model it came from.

**The mental model:** the probability is a well-calibrated estimate of a corpus
frequency, degraded by alignment, conflating two kinds of uncertainty. Where it
breaks down: for anything you would call a *fact*, the corpus frequency and the
truth can differ arbitrarily, and the model has no way to tell you which it is
reporting.

## 5. Formal Explanation

### 5.1 What the softmax optimises toward

Training minimises {{eq:clm-loss}}. For a fixed context $c$, the expected loss
over the true conditional distribution $p^*(\cdot\given c)$ is

$$
\E_{x\sim p^*}\big[-\log q(x)\big]
 = H\big(p^*\big) + \KL\big(p^*\,\|\,q\big)
$$ (eq:cross-entropy-decomposition)

which is minimised uniquely at $q = p^*$.

$\square$

**Cross-entropy is a strictly proper scoring rule**: the loss-minimising
prediction is the true probability, not a sharpened or hedged version of it.
That is the formal basis for expecting calibration, and it is why a *pretrained*
model is roughly calibrated without anyone asking it to be.

### 5.2 Calibration

A model is **perfectly calibrated** if, for every confidence level $p$,

$$
\Prob\big[\hat{y}\ \text{correct} \,\big|\, \hat{p} = p\big] = p
$$ (eq:perfect-calibration)

Partition predictions into $M$ bins by confidence. With $B_m$ the set of
predictions in bin $m$:

$$
\text{acc}(B_m) = \frac{1}{|B_m|}\sum_{i\in B_m}\Ind[\hat{y}_i = y_i],
\qquad
\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i\in B_m}\hat{p}_i
$$ (eq:bin-accuracy)

**Expected calibration error** is the weighted average gap:

$$
\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}
 \Big|\text{acc}(B_m) - \text{conf}(B_m)\Big|
$$ (eq:ece)

> WARNING: ECE depends on the binning. Too few bins hides miscalibration; too
> many leaves bins with too little data to estimate accuracy. Report the bin
> count with the number, and prefer the reliability diagram — which shows the
> *shape* of the miscalibration — over the scalar, which does not distinguish
> systematic overconfidence from noise.

### 5.3 Sequence probability

The probability of a generated sequence factorises
({{eq:autoregressive-factorisation}}):

$$
P(\vec{y}\given x) = \prod_{t=1}^{n} P(y_t\given x, y_{<t})
$$ (eq:sequence-probability)

In logs:

$$
\log P(\vec{y}\given x) = \sum_{t=1}^{n}\log P(y_t\given x, y_{<t})
$$ (eq:log-sequence-probability)

**Every term is negative, so longer sequences are less probable — always.** A
20-token answer cannot be more probable than a good 5-token one, whatever their
relative quality. This is the length bias that makes beam search prefer short
outputs ({{ch:llm-decoding}}) and that makes raw sequence probability a poor
quality score.

The standard correction is length normalisation:

$$
s(\vec{y}) = \frac{1}{n^{\alpha}}\log P(\vec{y}\given x)
$$ (eq:length-normalised-score)

with $\alpha \approx 1$ giving mean token log-probability. It is a heuristic
with no principled justification, and $\alpha$ is tuned.

### 5.4 Entropy as an uncertainty signal

$$
H\big(P(\cdot\given x)\big) = -\sum_{v\in V} P(v\given x)\log P(v\given x)
$$ (eq:token-entropy)

Bounded by $\log|V|$ (uniform) and 0 (deterministic). Entropy is a *distribution*
property and confidence is a *top-token* property, and they differ: a
distribution with 0.5 on one token and 0.5 on another has the same top
confidence as one with 0.5 on one and 0.001 on each of five hundred others, and
very different entropy.

**Entropy is the better runtime signal** because it accounts for the whole
distribution, and it is what {{ch:llm-routing}} thresholds on.

### 5.5 Why alignment degrades calibration

Pretraining optimises {{eq:cross-entropy-decomposition}}, whose optimum is the
true distribution. Alignment optimises {{eq:rlhf-objective}}, whose optimum is
{{eq:rlhf-optimal-policy}} — the reference reweighted by exponentiated reward.

$$
\pi^*(y\given x) \propto \pi_{\text{ref}}(y\given x)\,e^{r(x,y)/\beta}
$$

**Nothing in this expression preserves calibration.** The reweighting sharpens
the distribution toward high-reward outputs, and since human raters prefer
confident, decisive answers, the reward model rewards exactly the outputs whose
probabilities overstate their reliability. Aligned models are measurably more
overconfident than their base models, and it follows from the objective rather
than from any implementation defect.

## 6. Mathematical Foundation

### 6.1 Why cross-entropy is a proper scoring rule

Suppose the model reports $q$ while the truth is $p^*$. Expected loss:

$$
L(q) = -\sum_v p^*(v)\log q(v)
$$

Minimise subject to $\sum_v q(v) = 1$ with a multiplier $\lambda$:

$$
\frac{\partial}{\partial q(v)}\Big[L(q) + \lambda\big(\textstyle\sum_v q(v)-1\big)\Big]
 = -\frac{p^*(v)}{q(v)} + \lambda = 0
\implies q(v) = \frac{p^*(v)}{\lambda}
$$

Normalisation forces $\lambda = 1$, so $q = p^*$.

$\square$

**Reporting anything other than your true belief is penalised.** A model that
sharpened its outputs to look confident would incur *higher* loss, which is why
calibration emerges from pretraining without being an objective.

### 6.2 Why the softmax cannot express epistemic uncertainty

The distribution lives on the simplex over $V$. Every point on it asserts a
distribution over *tokens*. There is no point on the simplex meaning "I have no
information", because the closest candidate — the uniform distribution — asserts
that every token is equally likely, which is a strong and usually false claim.

Formally, the model has one output channel and two things to communicate:

$$
\text{aleatoric: } H\big(p^*(\cdot\given c)\big),
\qquad
\text{epistemic: } \KL\big(p^*(\cdot\given c)\,\|\,q(\cdot\given c)\big)
$$ (eq:two-uncertainties)

From {{eq:cross-entropy-decomposition}}, the loss is their sum, and **the model
observes only the sum.** It cannot report the second term because it does not
have access to it — knowing $\KL(p^*\|q)$ would require knowing $p^*$, which is
exactly what it lacks.

$\square$

This is not a limitation of the softmax parameterisation; it is a limitation of
having one distribution and no held-out reference. Every method that estimates
epistemic uncertainty for these models — ensembling, sampling variance, asking
the model directly ({{cite:kadavath2022}}) — works by manufacturing a second
opinion to compare against.

### 6.3 A worked calibration calculation

Ten predictions with confidences and outcomes:

| conf | 0.95 | 0.92 | 0.88 | 0.71 | 0.68 | 0.65 | 0.42 | 0.38 | 0.35 | 0.31 |
|---|---|---|---|---|---|---|---|---|---|---|
| correct | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |

Three bins.

**Bin [0.8, 1.0]:** confidences 0.95, 0.92, 0.88; two correct.
$\text{conf} = 0.917$, $\text{acc} = 0.667$, gap $0.250$, weight $3/10$.

**Bin [0.6, 0.8):** 0.71, 0.68, 0.65; two correct.
$\text{conf} = 0.680$, $\text{acc} = 0.667$, gap $0.013$, weight $3/10$.

**Bin [0.0, 0.6):** 0.42, 0.38, 0.35, 0.31; one correct.
$\text{conf} = 0.365$, $\text{acc} = 0.250$, gap $0.115$, weight $4/10$.

$$
\text{ECE} = 0.3(0.250) + 0.3(0.013) + 0.4(0.115) = 0.125
$$

**Every gap has the same sign — confidence exceeds accuracy in all three bins.**
That systematic direction is overconfidence, and it is what the scalar ECE
hides: a model with the same ECE from gaps of alternating sign is noisy rather
than biased, and the two need different corrections.

## 7. Internal Mechanics

```mermaid {#fig:calibration-pipeline caption="Where calibration is created and destroyed. Pretraining's objective has calibration as its optimum; every stage after it optimises something else, and each degrades the probabilities in a predictable direction."}
graph LR
  A["pretraining<br/>cross-entropy"] -->|"optimum IS<br/>the true distribution"| B["base model<br/>roughly calibrated"]
  B --> C["instruction tuning<br/>ch:fm-instruction-tuning"]
  C -->|"narrows the<br/>output distribution"| D["sharper"]
  D --> E["RLHF / DPO<br/>eq:rlhf-optimal-policy"]
  E -->|"rewards confident<br/>answers"| F["overconfident"]
  F --> G["temperature at serving<br/>ch:llm-decoding"]
  G -->|"rescales again"| H["what the user sees"]
  style B fill:#dfe,stroke:#5a5
  style F fill:#fde,stroke:#c69
```

**Temperature scaling as a post-hoc fix.** {{cite:guo2017calibration}}'s method:
fit a single scalar $T$ on a held-out set to minimise negative log likelihood,
then divide logits by it at inference. One parameter, no retraining, and it
substantially reduces ECE for classifiers. It cannot fix miscalibration that
varies with input — a single scalar applies the same correction everywhere.

**Where confidence is read from.** For a single token, the top softmax
probability. For a *sequence*, there is no agreed definition: mean token
probability, minimum token probability, and length-normalised sequence
probability all get used and they disagree. The minimum is the most useful for
detecting a specific unreliable step; the mean is the most stable.

**Why entropy and top-token confidence disagree, and when it matters.** Consider
two distributions over a 50,000-token vocabulary: one places 0.4 on a single
token and spreads 0.6 over three others; the other places 0.4 on a single token
and spreads 0.6 over five thousand. Top-token confidence is 0.4 for both. The
entropies differ by more than a factor of three, and so does what you should do
about them — the first is a genuine choice between a handful of continuations,
the second is a model with no idea. **Any system thresholding on top-token
confidence treats those cases identically**, which is the practical argument for
{{eq:token-entropy}} over the top probability as a runtime signal.

**Verbalised confidence.** Asking the model to state its confidence in words
produces a number that is a *generation*, not a readout of
{{eq:next-token-distribution}}. It is subject to all the same pressures as any
other output, including alignment's preference for confidence. It correlates
with correctness — {{cite:kadavath2022}} shows large models have some access to
their own reliability — and it is not the same measurement as the softmax
probability, and the two should not be conflated.

**Why the base model is the better-calibrated artefact.** This has a practical
consequence that surprises people: if you need calibrated probabilities — for
routing, for abstention, for active learning — the aligned chat model is the
wrong tool, and the base model's logits are better. Access to base-model logits
is therefore a real requirement rather than a research nicety.

**What a served API actually gives you.** Most hosted endpoints return either no
probabilities at all, or the top-$k$ log-probabilities for some small $k$. That
is enough for top-token confidence and *not* enough for
{{eq:token-entropy}}, which sums over the whole vocabulary. The usual workaround
is to compute entropy over the returned top-$k$ and renormalise, which
systematically **underestimates** entropy — the discarded tail is exactly the
mass that would have raised it. The error is largest precisely in the
high-uncertainty cases the signal exists to detect, so the approximation fails
where it matters most. Self-hosting, or an endpoint that returns full
distributions, is the only way around it.

**Sampling temperature at serving time changes the numbers you read back.** If
the endpoint applies temperature before returning log-probabilities, the values
are $\log\softmax(\vec{z}/T)$ rather than $\log\softmax(\vec{z})$, and by
{{eq:scale-is-temperature}} that is a different distribution entirely. Any
calibration fitted against those numbers is fitted against the serving
configuration as well as the model, and it silently becomes wrong when someone
changes a default. Read the API documentation on this specific point; the
behaviour differs between providers and is rarely prominent.

## 8. Implementation

Calibration measured from scratch, with the reliability diagram computed rather
than plotted.

```python {tier=A name=calibration-measurement}
"""Expected calibration error and a reliability diagram, from scratch."""
import numpy as np

rng = np.random.default_rng(0)
N, N_CLASSES = 6000, 20


# Draw the ground truth ONCE. Every row below is the same data and the same
# outcomes, differing only in how sharply the model reports its beliefs — which
# is what makes the accuracy column a controlled comparison rather than four
# independent experiments.
TRUE_LOGITS = rng.normal(size=(N, N_CLASSES))
_p = np.exp(TRUE_LOGITS - TRUE_LOGITS.max(1, keepdims=True))
P_TRUE = _p / _p.sum(1, keepdims=True)
Y = np.array([rng.choice(N_CLASSES, p=row) for row in P_TRUE])


def make_predictions(sharpening):
    """The same model, reporting its beliefs at a given sharpness.

    sharpening = 1.0 -> calibrated; > 1 -> overconfident; < 1 -> underconfident.
    """
    z = TRUE_LOGITS * sharpening
    rep = np.exp(z - z.max(1, keepdims=True))
    rep /= rep.sum(1, keepdims=True)
    return rep, Y


def ece_and_bins(probs, y, n_bins=10):
    """Equations (eq:bin-accuracy) and (eq:ece)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.sum() / len(conf)
        ece += w * abs(acc - cf)
        rows.append((lo, hi, int(m.sum()), cf, acc, acc - cf))
    return ece, rows


print("Reliability diagram for a CALIBRATED model\n")
probs, y = make_predictions(1.0)
ece, rows = ece_and_bins(probs, y)
print(f"{'bin':>12} {'n':>6} {'confidence':>12} {'accuracy':>10} {'gap':>8}")
for lo, hi, n, cf, acc, gap in rows:
    bar = "#" * int(abs(gap) * 100)
    print(f"[{lo:.1f},{hi:.1f}] {n:>6} {cf:>12.3f} {acc:>10.3f} {gap:>+8.3f} {bar}")
print(f"\nECE = {ece:.4f}  (overall accuracy {(probs.argmax(1) == y).mean():.3f})")

# Calibration and accuracy are independent axes.
print(f"\n{'model':<26} {'accuracy':>10} {'mean conf':>11} {'ECE':>8} "
      f"{'verdict':<18}")
for label, sharp in [("underconfident", 0.6), ("calibrated", 1.0),
                     ("overconfident", 1.6), ("very overconfident", 2.5)]:
    p, yy = make_predictions(sharp)
    e, _ = ece_and_bins(p, yy)
    acc = float((p.argmax(1) == yy).mean())
    mc = float(p.max(1).mean())
    verdict = ("well calibrated" if e < 0.03
               else ("overconfident" if mc > acc else "underconfident"))
    print(f"{label:<26} {acc:>10.3f} {mc:>11.3f} {e:>8.4f} {verdict:<18}")

print("""
ACCURACY IS IDENTICAL IN ALL FOUR ROWS, to the last digit. Sharpening a
distribution is a monotone transformation of the logits, so it cannot change
which token is the argmax — only how confident the model claims to be about it.

That is equation (eq:perfect-calibration)'s point made concretely: calibration
and accuracy are independent axes. A model can be made to look confident
without being made to be right, and the four rows here are literally the same
model on the same data.""")

# The binning sensitivity of the warning in section 5.2, at two sample sizes.
p_full, y_full = make_predictions(1.6)
print(f"\n{'bins':>6} {'ECE (n=' + str(N) + ')':>18} {'ECE (n=200)':>14} "
      f"{'empty bins at n=200':>21}")
for nb in (5, 10, 20, 50, 100):
    e_full, _ = ece_and_bins(p_full, y_full, n_bins=nb)
    e_small, rows_small = ece_and_bins(p_full[:200], y_full[:200], n_bins=nb)
    print(f"{nb:>6} {e_full:>18.4f} {e_small:>14.4f} "
          f"{nb - len(rows_small):>21}")

print("""
With 6,000 samples the estimate is stable across bin counts — which is worth
knowing, because it means the usual warning is conditional rather than general.
With 200 samples it is not: bins empty out, the surviving ones are estimated
from a handful of points, and the number wanders.

So report the bin count AND the sample size, and prefer the reliability diagram,
which shows whether the gaps share a sign — systematic overconfidence — or
alternate, which is noise. The scalar cannot distinguish those and the shape
can.""")
```

Now the post-hoc fix, and its limits:

```python {tier=A name=temperature-scaling}
"""Temperature scaling: one parameter, fitted on held-out data."""
import numpy as np

rng = np.random.default_rng(1)
N, K = 8000, 20

# A model that is systematically overconfident by a constant factor — the
# situation temperature scaling is designed for.
true_logits = rng.normal(size=(N, K))
p_true = np.exp(true_logits - true_logits.max(1, keepdims=True))
p_true /= p_true.sum(1, keepdims=True)
y = np.array([rng.choice(K, p=row) for row in p_true])
model_logits = true_logits * 1.8            # the miscalibration


def softmax_T(z, T):
    s = z / T
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def nll(z, y, T):
    p = softmax_T(z, T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def ece(z, y, T, n_bins=15):
    p = softmax_T(z, T)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return tot


split = N // 2
val_z, val_y = model_logits[:split], y[:split]
test_z, test_y = model_logits[split:], y[split:]

# Fit T on the validation half by minimising NLL — one scalar parameter.
grid = np.linspace(0.5, 4.0, 200)
losses = [nll(val_z, val_y, T) for T in grid]
T_hat = float(grid[int(np.argmin(losses))])

print(f"fitted temperature: {T_hat:.3f}  (the miscalibration was a 1.8x "
      f"logit scale, so the correct T is 1.8)")
print(f"\n{'':<22} {'NLL':>9} {'ECE':>9} {'accuracy':>10}")
for label, T in [("uncorrected (T=1)", 1.0), (f"scaled (T={T_hat:.2f})", T_hat)]:
    print(f"{label:<22} {nll(test_z, test_y, T):>9.4f} "
          f"{ece(test_z, test_y, T):>9.4f} "
          f"{(softmax_T(test_z, T).argmax(1) == test_y).mean():>10.4f}")

print("\nAccuracy is unchanged — dividing every logit by a constant cannot "
      "reorder them. Only the probabilities moved.")

# Where a single scalar cannot help: input-dependent miscalibration.
easy = rng.normal(size=(N // 2, K))
hard = rng.normal(size=(N // 2, K)) * 0.4          # genuinely more uncertain
mixed_true = np.vstack([easy, hard])
p_m = np.exp(mixed_true - mixed_true.max(1, keepdims=True))
p_m /= p_m.sum(1, keepdims=True)
y_m = np.array([rng.choice(K, p=row) for row in p_m])
# Overconfident on the hard half only.
mixed_logits = np.vstack([easy, hard * 3.0])

grid_losses = [nll(mixed_logits, y_m, T) for T in grid]
T_mixed = float(grid[int(np.argmin(grid_losses))])
print(f"\ninput-dependent miscalibration: best single T = {T_mixed:.3f}")
print(f"{'subset':<14} {'ECE at T=1':>12} {'ECE at fitted T':>17}")
for name, sl in [("easy half", slice(0, N // 2)), ("hard half", slice(N // 2, N))]:
    print(f"{name:<14} {ece(mixed_logits[sl], y_m[sl], 1.0):>12.4f} "
          f"{ece(mixed_logits[sl], y_m[sl], T_mixed):>17.4f}")

print("""
One scalar applies the same correction everywhere. When the miscalibration
differs across inputs, fitting T trades one subset's calibration against the
other's — the fitted value is a compromise that is wrong for both. That is the
limit of the method, and it is why calibration should be reported per slice
rather than in aggregate.""")
```

And the length effect that makes raw sequence probability a poor score:

```python {tier=A name=sequence-probability-length-bias}
"""Why longer sequences are always less probable, and what to do about it."""
import numpy as np

rng = np.random.default_rng(2)

# Per-token probabilities for two candidate answers: a short mediocre one and
# a long good one.
short = np.array([0.55, 0.48, 0.60, 0.51])                       # 4 tokens
long_good = np.array([0.82, 0.79, 0.85, 0.88, 0.81, 0.86,
                      0.83, 0.90, 0.84, 0.87, 0.85, 0.88])       # 12 tokens

print(f"{'candidate':<16} {'tokens':>7} {'mean p':>9} {'log P':>10} "
      f"{'P':>12}")
for name, p in [("short, mediocre", short), ("long, good", long_good)]:
    lp = float(np.log(p).sum())
    print(f"{name:<16} {len(p):>7} {p.mean():>9.3f} {lp:>10.3f} "
          f"{np.exp(lp):>12.3e}")

print(f"\nThe long answer has better tokens at every position and is "
      f"{np.exp(np.log(short).sum()) / np.exp(np.log(long_good).sum()):.0f}x "
      f"MORE probable in the short one's favour.")
print("Equation (eq:log-sequence-probability): every term is negative, so "
      "length is a penalty regardless of quality.\n")

# Length normalisation, equation (eq:length-normalised-score).
print(f"{'alpha':>7} {'short score':>13} {'long score':>12} {'winner':>16}")
for alpha in (0.0, 0.5, 0.7, 1.0):
    s = np.log(short).sum() / (len(short) ** alpha)
    l = np.log(long_good).sum() / (len(long_good) ** alpha)
    print(f"{alpha:>7.1f} {s:>13.4f} {l:>12.4f} "
          f"{('long' if l > s else 'short'):>16}")

# How strong is the bias in general?
print(f"\n{'length':>8} {'log P at mean p=0.8':>22} {'P':>12}")
for n in (1, 5, 10, 50, 200):
    lp = n * np.log(0.8)
    print(f"{n:>8} {lp:>22.2f} {np.exp(lp):>12.2e}")

print("""
A 200-token answer in which the model is 80% confident at every single step has
a sequence probability of about 1e-19. That number is not a quality judgement
and cannot be compared against a 5-token answer's.

Length normalisation with alpha near 1 — mean token log-probability — makes the
comparison sane, and it is a heuristic with no principled basis. There is no
correct alpha, which is worth knowing before building a ranking on it.""")
```

## 9. Practical Example

A team wants to route: send easy questions to a small model and hard ones to a
large one ({{ch:llm-routing}}). The obvious signal is the small model's
confidence — send it onward when unsure. Whether that works depends entirely on
calibration, and the aligned model they are using is not calibrated.

```python {tier=A name=confidence-as-a-routing-signal}
"""Can confidence decide when to escalate? Only if it is calibrated."""
import numpy as np

rng = np.random.default_rng(4)
N, K = 5000, 20

# Ground truth difficulty: some questions the small model can answer, some not.
difficulty = rng.random(N)
small_correct = rng.random(N) > difficulty          # harder -> less likely right
LARGE_ACC = 0.93
large_correct = rng.random(N) < LARGE_ACC

COST_SMALL, COST_LARGE = 1.0, 12.0


def confidence(sharpening):
    """The small model's reported confidence, at a given miscalibration."""
    # True confidence tracks difficulty; sharpening distorts the reported value.
    true_conf = np.clip(1 - difficulty + rng.normal(0, 0.08, N), 0.02, 0.98)
    logit = np.log(true_conf / (1 - true_conf))
    return 1 / (1 + np.exp(-logit * sharpening))


def evaluate(conf, threshold):
    escalate = conf < threshold
    correct = np.where(escalate, large_correct, small_correct)
    cost = np.where(escalate, COST_SMALL + COST_LARGE, COST_SMALL)
    return float(correct.mean()), float(cost.mean()), float(escalate.mean())


print(f"small model alone : accuracy {small_correct.mean():.3f}, "
      f"cost {COST_SMALL:.1f}")
print(f"large model alone : accuracy {large_correct.mean():.3f}, "
      f"cost {COST_LARGE:.1f}\n")

for label, sharp in [("calibrated", 1.0), ("overconfident (aligned)", 2.6)]:
    conf = confidence(sharp)
    print(f"--- {label} confidence ---")
    print(f"{'threshold':>10} {'escalated':>11} {'accuracy':>10} {'cost':>8} "
          f"{'acc per cost':>13}")
    best = None
    for thr in (0.0, 0.3, 0.5, 0.7, 0.85, 1.0):
        acc, cost, esc = evaluate(conf, thr)
        eff = acc / cost
        if best is None or eff > best[1]:
            best = (thr, eff, acc, cost, esc)
        print(f"{thr:>10.2f} {esc:>11.1%} {acc:>10.3f} {cost:>8.2f} "
              f"{eff:>13.4f}")
    print(f"  best efficiency at threshold {best[0]:.2f}: "
          f"accuracy {best[2]:.3f} at cost {best[3]:.2f}\n")

# The measurement that decides whether the signal is usable at all.
for label, sharp in [("calibrated", 1.0), ("overconfident", 2.6)]:
    conf = confidence(sharp)
    # Does confidence actually separate correct from incorrect?
    auc = float(np.mean([
        (conf[i] > conf[j]) for i in rng.choice(np.flatnonzero(small_correct), 2000)
        for j in rng.choice(np.flatnonzero(~small_correct), 1)]))
    print(f"{label:<16} mean conf when right {conf[small_correct].mean():.3f}, "
          f"when wrong {conf[~small_correct].mean():.3f}, "
          f"separation AUC {auc:.3f}")

print("""
The separation AUC is the number that matters and it barely moves: sharpening a
distribution is monotone, so it preserves the ORDERING of confidences and
therefore the ranking quality of the signal.

What miscalibration destroys is the meaning of the THRESHOLD. On the calibrated
model, 0.7 means roughly 70% and a threshold can be chosen from a target error
rate. On the overconfident model, 0.7 means something else entirely, and the
threshold has to be found empirically and refitted whenever the model changes.

So confidence remains usable for routing after alignment — as a rank, not as a
probability. Anything that needs the number to mean what it says (abstention at
a stated error rate, expected-value calculations, cost-sensitive decisions)
needs calibration first.""")
```

> PRODUCTION TIP: If you need probabilities that mean what they say, fit
> temperature scaling on a held-out slice of your own traffic and re-fit it
> whenever the model version changes. It is one parameter, it costs minutes, and
> almost nobody does it.

## 10. Production Considerations

**Report calibration per slice.** `temperature-scaling` shows a single scalar
cannot fix input-dependent miscalibration, and aggregate ECE hides exactly the
case that matters — a model well calibrated on common inputs and badly
calibrated on rare ones.

**Re-fit temperature on model change.** It is fitted to a specific checkpoint's
logit scale, which moves with every fine-tune.

**Prefer entropy to top-token confidence as a runtime signal.**
{{eq:token-entropy}} uses the whole distribution; top-token confidence discards
it.

**Use base-model logits where calibration matters.** The aligned model is the
more useful generator and the worse probability estimator, and both facts follow
from {{eq:rlhf-optimal-policy}}.

**Never compare raw sequence probabilities across lengths.**
{{eq:log-sequence-probability}} makes length a penalty. Normalise, and record
the $\alpha$ you used.

**What to monitor:** ECE on a labelled sample, mean top-token confidence, mean
entropy, and the confidence distribution's shape. A confidence distribution
collapsing toward 1.0 over time is a drifting model or a changed serving
temperature, and it is visible long before accuracy moves.

## 11. Common Mistakes

**Beginners:**

*Reading the softmax as a truth probability.* It estimates a corpus frequency
under the training distribution, which can differ arbitrarily from fact.

*Confusing calibration with accuracy.* `calibration-measurement` shows accuracy
identical across four models with wildly different ECE.

*Comparing sequence probabilities of different lengths.* Always favours the
shorter one.

**Experienced practitioners:**

*Reporting ECE without the bin count.* It is binning-dependent, and the
diagram carries the information the scalar loses.

*Assuming an aligned model is calibrated.* It is systematically less calibrated
than its base, by {{eq:rlhf-optimal-policy}}.

*Treating verbalised confidence as a probability readout.* It is a generation,
subject to the same pressures as any other output.

*Aggregating calibration across heterogeneous traffic.* The compromise
temperature is wrong for every subset — `temperature-scaling` shows the fitted
value making the easy half's calibration seven times worse while improving the
hard half's, which is a trade nobody chose.

*Fitting calibration against an endpoint that applies temperature.* You are then
calibrating the model and the serving configuration jointly, and the result
silently breaks when a default changes.

## 12. Failure Modes

**Confident fabrication.** High probability, wrong answer, epistemic uncertainty
expressed as certainty. *Detection:* not from the softmax alone — it requires an
external check or a manufactured second opinion. *This is the mechanism behind
most of {{ch:llm-hallucination}}.*

**Threshold drift after a model update.** A confidence threshold tuned on one
checkpoint silently means something else on the next. *Detection:* monitor the
confidence distribution, not just accuracy.

**Length bias in ranking.** Best-of-$n$ or beam search selecting short outputs.
*Detection:* correlate selected-output length against candidate length.

**Aggregate calibration masking a bad slice.** *Detection:* per-slice ECE.

**Temperature double-application.** Serving applies temperature and an upstream
component already scaled the logits. *Symptom:* sampling behaviour that does not
match the configured temperature. *Cause:* {{eq:scale-is-temperature}} —
anything that scales logits is a temperature change.

## 13. Alternatives

{#tbl:uncertainty-methods caption="Ways to estimate uncertainty from a language model. Only the first is free; the rest manufacture a second opinion, which is what equation (eq:two-uncertainties) says is necessary to separate the two kinds."}

| Method | Cost | Captures | Limitation |
|---|---|---|---|
| Top-token probability | free | aleatoric | conflates the two |
| Entropy of the distribution | free | aleatoric, better | same conflation |
| Temperature scaling | one held-out fit | fixes global bias | not input-dependent |
| Sampling variance | $n$ generations | some epistemic | expensive, sampler-dependent |
| Ensembles | $n$ models | epistemic | $n$ times everything |
| Verbalised confidence | one generation | correlates | a generation, not a readout |
| External verification | a retrieval or tool call | factuality | only where checkable |

**What genuinely differs.** The first two read the existing distribution and
cannot separate {{eq:two-uncertainties}}'s terms. Everything below them
constructs a comparison — a second sample, a second model, a second channel —
which is the only way to estimate the second term, since it requires information
the single distribution does not contain.

## 14. Evaluation

**Is the model calibrated?** Reliability diagram first, ECE second, per slice
always. Report the bin count.

**Is confidence usable as a signal?** Two different questions:

1. **As a rank** — does it separate correct from incorrect? Measured by AUC, and
   preserved under monotone miscalibration, as
   `confidence-as-a-routing-signal` shows.
2. **As a probability** — does 0.7 mean 70%? Requires calibration, and is what
   abstention and expected-value decisions need.

**A model can pass the first and fail the second**, which is the common case
after alignment, and knowing which one your system depends on determines whether
you need to do anything about it.

## 15. Advanced Concepts

**Temperature scaling and its relatives.** {{maturity:ESTABLISHED}}
{{cite:guo2017calibration}}'s single-parameter fit, plus vector and matrix
scaling for more capacity at the cost of more held-out data.

**Self-evaluation.** {{maturity:EMERGING}} {{cite:kadavath2022}} finds large
models have non-trivial access to whether their own answers are correct, when
asked in a specific format. This is a second channel and is therefore not
subject to {{eq:two-uncertainties}}'s impossibility.

**Semantic entropy.** {{maturity:EMERGING}} Clustering multiple samples by
meaning and computing entropy over clusters rather than token sequences —
distinguishing "the model is unsure what to say" from "the model is unsure what
is true".

**Conformal prediction.** {{maturity:EMERGING}} Producing prediction sets with a
distribution-free coverage guarantee, which sidesteps calibration entirely by
targeting coverage rather than probability accuracy. The trade is worth stating:
you give up a per-item probability and receive, in exchange, a set that provably
contains the right answer at your chosen rate — a guarantee no amount of
temperature fitting provides, at the cost of the set sometimes being large
enough to be useless. For a system that must abstain at a stated error rate,
that is frequently the better bargain, and it is the one method in
{{tbl:uncertainty-methods}} whose guarantee survives distribution shift by
construction rather than by assumption.

**Calibration under distribution shift.** {{maturity:RESEARCH FRONTIER}}
Temperature fitted on one distribution does not transfer to another, and
production traffic drifts. Whether calibration can be maintained without
continuous relabelling is open.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:llm-anatomy}}'s {{eq:next-token-distribution}} is what this
chapter interprets, and {{eq:scale-is-temperature}} is why anything touching
logit magnitude is a calibration change. {{ch:fm-pretraining}}'s cross-entropy
is why base models are calibrated at all
({{eq:cross-entropy-decomposition}}). {{ch:fm-rlhf}}'s
{{eq:rlhf-optimal-policy}} is why aligned models are not.
{{ch:ml-metrics}}'s separation of "is it correct" from "is it well behaved" is
this chapter's accuracy/calibration distinction.

**Forwards.** {{ch:llm-decoding}} manipulates the distribution this chapter
interprets. {{ch:llm-hallucination}} depends on distinguishing confident error
from unlucky sampling. {{ch:llm-routing}} thresholds on the signal evaluated
here. {{part:25}} builds calibration into evaluation practice, and
{{ch:rag-failures}} uses groundedness as the external check that
{{tbl:uncertainty-methods}}'s last row describes.

## 17. Exercises

**Beginner**

1. A model predicts a token at 0.9 and is wrong. Is it miscalibrated? Explain.
2. Compute the entropy of $(0.7, 0.2, 0.1)$ and of $(0.7, 0.15, 0.15)$.
3. Why is a 30-token answer always less probable than a 5-token one?

**Intermediate**

4. Compute ECE for: confidences $(0.9, 0.8, 0.7, 0.6)$ with outcomes
   $(✓,✗,✓,✓)$, using two bins.
5. Explain why sharpening a distribution leaves accuracy unchanged.
6. Using {{eq:length-normalised-score}}, find the $\alpha$ at which a 6-token
   answer with mean $p=0.5$ ties a 3-token answer with mean $p=0.7$.

**Advanced**

7. Prove cross-entropy is a strictly proper scoring rule.
8. Derive {{eq:two-uncertainties}} and explain why a single distribution cannot
   report the second term.
9. Explain why {{eq:rlhf-optimal-policy}} degrades calibration, and predict the
   direction and which inputs are worst affected.

**Implementation**

10. Extend `calibration-measurement` with adaptive binning (equal-count rather
    than equal-width) and compare the ECE estimates.
11. Implement vector scaling — a per-class temperature — and compare against
    scalar temperature on the input-dependent case.
12. Implement semantic entropy: sample $n$ generations, cluster by exact match,
    and compute entropy over clusters. Compare against token entropy.
13. Reproduce the routing analysis with a genuinely uncalibrated signal and find
    the threshold empirically. Show it moving when the sharpening changes.

**Reasoning**

14. Your system abstains when confidence < 0.8 and the error rate among answered
    questions is 12%, not the 20% you expected. What happened?
15. Explain why the base model is the better tool for routing and the worse tool
    for answering.

## 18. Interview Questions

**Beginner**

1. What does a token probability of 0.7 mean?
2. What is calibration and how does it differ from accuracy?
3. Why can't you compare sequence probabilities of different lengths?

**Intermediate**

4. How would you measure calibration? What are ECE's weaknesses?
5. Why are base models better calibrated than aligned ones?
6. What is the difference between aleatoric and epistemic uncertainty here?

**Senior**

7. You want to abstain at a 5% error rate. Walk through what you need.
8. Confidence-based routing stopped working after a model upgrade. Diagnose.
9. When is verbalised confidence useful and when is it misleading?

**Systems**

10. Design calibration monitoring for a production LLM service.
11. How would you maintain a confidence threshold across model versions?

## 19. Research Questions

**How much calibration does alignment actually cost?** The direction follows
from {{eq:rlhf-optimal-policy}}; the magnitude is not systematically measured.
Compare base and aligned checkpoints of the same models on identical
calibration sets, as a function of $\beta$.

**Can calibration be preserved during alignment?** Adding a calibration term to
the alignment objective is the obvious attempt. What does it cost in preference
win-rate, and is the trade favourable?

**Does self-evaluation survive alignment?** {{cite:kadavath2022}} measured a
second channel's access to correctness. Alignment pressures that channel too —
raters prefer confident self-assessments. Measure whether it degrades in the
same direction as the softmax.

**Is there a usable epistemic signal that is free?** Everything in
{{tbl:uncertainty-methods}} below the first two rows costs extra compute. Whether
some function of a single forward pass's internals — attention entropy, residual
norm, layer disagreement — carries epistemic information is testable and largely
untested.

## 20. Chapter Summary

A logit vector softmaxed is a distribution over the next token, and what it
estimates is **the conditional frequency of that token in the training
distribution** — a claim about a corpus, not about the world.

**Pretraining produces calibration for free**, because cross-entropy is a
strictly proper scoring rule {{eq:cross-entropy-decomposition}}: the
loss-minimising report is the true probability, and sharpening to look confident
is penalised. **Alignment destroys it, also for free.**
{{eq:rlhf-optimal-policy}} reweights toward high-reward outputs and human raters
prefer confident answers, so the most useful model is systematically the least
calibrated one.

**Calibration and accuracy are independent axes.**
`calibration-measurement` shows four models with identical accuracy and ECE
ranging over an order of magnitude — sharpening a distribution is monotone, so
it cannot change the argmax. That has a practical corollary: **miscalibrated
confidence is still usable as a rank and not as a probability.** Routing by
confidence survives alignment; abstaining at a stated error rate does not.

**The softmax cannot express epistemic uncertainty.**
{{eq:two-uncertainties}} splits the loss into the task's inherent
unpredictability and the model's own error, and the model observes only their
sum — knowing the second term would require knowing the true distribution, which
is exactly what it lacks. Every method that estimates it works by manufacturing
a second opinion: another sample, another model, or another channel.

**Sequence probability is not a quality score.** Every term of
{{eq:log-sequence-probability}} is negative, so a longer answer is always less
probable regardless of how much better it is — a 200-token answer at 80%
per-token confidence scores $10^{-19}$. Length normalisation makes the
comparison sane and is an unprincipled heuristic with a tuned exponent.

The practical residue: fit temperature scaling per slice on your own traffic,
re-fit it on every model change, prefer entropy to top-token confidence, use
base-model logits when the number needs to mean something, and never compare raw
sequence probabilities across lengths.

## 21. Further Reading

{{cite:guo2017calibration}} is the reference for the problem and for temperature
scaling. It predates LLMs and is about image classifiers, which is useful — the
phenomenon is a property of modern neural networks trained with cross-entropy,
not something specific to language.

{{cite:kadavath2022}} for self-evaluation, and for the framing that a model's
access to its own reliability is a separate channel from its output
distribution. Read §3 and §4.

{{cite:holtzman2020}} belongs to the next chapter but its central observation is
a calibration observation: the highest-probability sequence is not the best one,
which means likelihood and quality come apart in a specific measurable way.

{{cite:ji2023survey}} for where this chapter leads. Its intrinsic/extrinsic
split maps onto {{eq:two-uncertainties}} more closely than either literature
usually acknowledges.

**Where to go next:** {{ch:llm-decoding}} takes the distribution this chapter
interpreted and turns it into text — where temperature, top-k and top-p live,
and where {{cite:holtzman2020}} shows that maximising likelihood produces text
nobody wants.
