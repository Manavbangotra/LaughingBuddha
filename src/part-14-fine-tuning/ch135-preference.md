---
id: ft-preference
number: 135
part: XIV
tier: full
status: draft
requires: [ft-synthetic, ft-datasets, fm-rlhf, fm-dpo]
provides: [annotator-agreement, agreement-caps-measurement, noise-is-a-tax,
           comparison-cancels-the-bar, bar-drift, rubric-as-leverage,
           preference-budget-order]
citations: [ethayarajh2024kto, rafailov2023, ouyang2022, zhou2023lima,
            wang2023selfinstruct]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state why annotator agreement
bounds what you can **measure** even when it does not bound what you can
**learn**; recognise the instrumentation failure that makes a working reward model
look broken; choose between binary and pairwise feedback from a property of your
annotators rather than from a paper; explain the exact cancellation that makes
comparisons immune to bar drift; and order a preference-data budget so the
cheapest intervention comes first.

## 2. Why This Matters

{{part:9}} derived Bradley–Terry, RLHF and DPO. **This chapter is about the data
those objectives consume**, and about two numbers nobody reports.

**The first is annotator agreement.** {{sec:9-practical-example}} trains a reward
model on preferences from a low-discrimination annotator: two annotators agree
**56.9%** of the time, and the reward model is **71.3%** accurate against latent
truth — while scoring **58.9%** against a held-out annotator label.

**The model is far better than its own evaluation says.** Not because the
evaluation is badly built, but because held-out labels come from the same noisy
process, so a *perfect* model would still disagree with them at the annotator's
error rate.

**And that cuts both ways.** A reward-model accuracy of 59% might mean nearly
optimal on an ambiguous task, or badly broken on a clean one. **The number alone
distinguishes nothing, and the agreement rate is almost never published beside
it.**

**The scaling result was not what the first table sets you up to expect.** Going
500 → 32,000 pairs takes the noisy annotator from **0.567 to 0.762** against
truth, close to the clean annotator's **0.786**. **Unbiased annotator noise is a
tax on data efficiency, not a ceiling on capability** — roughly 4× the budget for
the same result.

**Put the two together and you get the failure that matters.** You *can* buy your
way past noisy labels. You can *never measure* that you did. A team watching a
reward model refuse to score above the high fifties concludes the approach is not
working and stops — while the model is at 71% on what they care about.

**The second number is bar drift**, and it decides binary versus pairwise. At zero
drift binary wins (**0.758** against **0.744**) on twice the judgements per pound.
At high drift it **falls off a cliff to 0.636**, while pairwise is **exactly flat
at every drift level** — not robustness, cancellation.

{{maturity:ESTABLISHED}} Preference data collection. {{maturity:MATURE}} Binary
feedback objectives. {{maturity:EMERGING}} Treating agreement and bar spread as
the first measurements rather than afterthoughts.

## 3. Prerequisites

{{ch:fm-rlhf}} and {{ch:fm-dpo}} own the objectives — Bradley–Terry, the KL
penalty, the implicit reward. **This chapter does not restate them.**
{{ch:ft-datasets}} for provenance and group splits, which preference pairs need
identically; {{ch:ft-synthetic}} for the systematic-versus-random distinction,
which bar drift is an instance of.

> **NOTE:** *Alignment* is overloaded. {{part:9}} uses it for making a model
> behave as intended; {{ch:emb-what-they-are}} uses it geometrically. **This
> chapter means the first.**

## 4. Intuitive Explanation

### Agreement bounds the ruler, not just the model

Everyone expects noisy labels to limit what a model learns. The under-appreciated
half is that **they limit what you can measure by exactly the same amount.**

If two annotators agree 57% of the time, then a model predicting the *truth*
perfectly still disagrees with a randomly-chosen held-out label 43% of the time.
**The measurement ceiling is the agreement rate, not 100%.**

So the shortfall between "accuracy against truth" and "accuracy against labels" is
**+0.124** at low agreement and **+0.008** at high agreement. Clean labels make
the measurement honest; noisy labels make it pessimistic by an amount nobody
quantifies.

### But noise is a tax, not a wall

Here the measurement contradicted the natural expectation. More data *does* climb
past the annotator: **0.567 → 0.762** across a 64× budget increase, ending near
the clean annotator's 0.786.

**The reason is that this noise is unbiased.** An annotator who is merely
*uncertain* is wrong in both directions with roughly equal probability, and enough
independent draws average that out. Compare where the two reach the same accuracy:
the clean annotator hits **0.773 at 8,000 pairs**, and the noisy one has not
matched it at 32,000. **Past 4× the budget, for the same result.**

> **The combination is the actionable part, and neither half implies it alone.**
> You can buy your way past noisy preference labels with volume. You can never
> measure that you did — so the project gets cancelled for a reason that is an
> artefact of its own instrumentation.

### Binary or pairwise: ask about your annotators

{{cite:ethayarajh2024kto}}'s practical argument is strong: production emits
thumbs-up and thumbs-down for free, in volumes no comparison campaign matches. The
counter-argument — a comparison carries more information — is also true.

**Neither settles it, because the real comparison is at equal *annotation
budget*.** A pairwise judgement requires reading two items; a binary one, one. So
**binary starts with twice the judgements for the money.**

What decides it is that annotators do not share a bar. Asked "is this good?", each
applies their own threshold — and **real annotation is routed**: one person takes
the coding queries, another the writing queries, so *which* bar an item meets is a
function of *what the item is*.

**A comparison is immune.** The annotator's bar appears on both sides and
subtracts out. **A rating is not**: the label is quality-minus-bar, and nothing in
the data identifies the bar.

### And it is a cliff, not a slope

The measured shape matters as much as the direction:

```text
   bar drift    binary       pairwise
   ─────────    ──────       ────────
        0.0      0.758          0.744     binary wins on volume
        0.3      0.748          0.751     level
        0.6      0.754          0.741     level
        1.0      0.636          0.749     binary collapses
```

**Binary is fine — better, even — up to a threshold, and then much worse.** So
"how much do our annotators disagree about where the bar is" is not a
nice-to-know: it is the variable that decides which data to buy, and it has a
**sharp** answer.

**And the damage is not confined to the topic boundary.** At high drift, *within*-
topic accuracy is **0.636** against across-topic **0.624** — barely different. The
routed bars corrupt the learned score badly enough that ordering fails inside a
topic too, because one smooth function is being fitted to labels that four
different questions produced.

### A hypothesis that failed, and why it is worth reporting

The expectation going in was that within-topic comparisons would leave the
*offsets between* topics unidentified — that a reward model trained only on
same-topic pairs would rank correctly inside each topic and arbitrarily between
them, so a share of topic-spanning comparisons would be load-bearing.

**It made no difference**: 0.735 with 30% spanning pairs against 0.734 without, at
identical sample size.

**The reason is more useful than the hypothesis was.** A reward model is a smooth
function of content, not a lookup table with a free parameter per topic. Topics
overlap in feature space, so an ordering learned in one region constrains the
function in neighbouring regions, and the cross-topic offsets are identified
*implicitly*.

> **With a condition to check rather than assume.** That identification relies on
> topics not being genuinely disjoint in the reward model's representation.
> Separate specialist heads, or topics with no shared features, would restore the
> original worry.

## 5. Formal Explanation

### 5.1 The measurement ceiling

Let $y^{*}$ be the latent preference and $\tilde{y}$ an annotator's label, with
$P(\tilde{y} = y^{*}) = p$. For a model $\hat{y}$ with true accuracy
$a = P(\hat{y} = y^{*})$, the *measured* accuracy against a single held-out label
is

$$ \mathcal{A}_{\text{obs}} = a p + (1-a)(1-p) $$ (eq:agreement-caps-measurement)

At $a = 1$ this is $p$ — **the ceiling is the annotator's accuracy.** Two
annotators agree at rate $\alpha = p^2 + (1-p)^2$, so $p$ is recoverable from a
double-labelled sample:

$$ p = \tfrac{1}{2}\left(1 + \sqrt{2\alpha - 1}\right) $$ (eq:p-from-agreement)

**{{eq:p-from-agreement}} is the whole reason to double-label**: it converts an
uninterpretable score into an interpretable one. Inverting
{{eq:agreement-caps-measurement}} gives the true accuracy:

$$ a = \frac{\mathcal{A}_{\text{obs}} + p - 1}{2p - 1} $$ (eq:deconvolved-accuracy)

At $\alpha = 0.569$, $p \approx 0.686$, and $\mathcal{A}_{\text{obs}} = 0.589$
gives $a \approx 0.735$ — close to the measured 0.713.

### 5.2 Why the noise is a tax rather than a wall

Under Bradley–Terry noise the label is correct with probability
$\sigma(\beta \Delta q)$, which is **unbiased**: $\mathbb{E}[\tilde{y}] $ is a
monotone function of $\Delta q$, so the Bayes-optimal ranking under the noisy
distribution is the *true* ranking. Therefore

$$ \lim_{n \to \infty} \hat{a}(n) = a^{*} \quad \text{for any } \beta > 0 $$ (eq:noise-is-a-tax)

with the approach rate set by the label entropy: halving the residual costs
roughly $4\times$ the data, and the constant in front is $\beta$.

**{{eq:noise-is-a-tax}} is why "our annotators are unreliable" is a budget
statement, not a capability statement** — provided the unreliability is
*uncertainty* rather than *bias*.

### 5.3 Why comparisons cancel the bar

An annotator with bar $b$ rating item $x$:

$$ \tilde{y}_{\text{bin}}(x) = \mathbb{1}\big[q(x) - b + \varepsilon > 0\big] $$

Comparing $x_a$ and $x_b$:

$$ \tilde{y}_{\text{pair}}(x_a, x_b) = \mathbb{1}\big[(q(x_a) - b) - (q(x_b) - b) + \varepsilon > 0\big] = \mathbb{1}\big[q(x_a) - q(x_b) + \varepsilon > 0\big] $$ (eq:comparison-cancels-the-bar)

**{{eq:comparison-cancels-the-bar}} is exact, not approximate** — which is why the
pairwise columns are flat rather than merely robust. $b$ does not appear.

### 5.4 Why bar drift is systematic, not noise

If annotator assignment were random, $b$ would be independent of $x$ and

$$ \mathbb{E}_b\big[\sigma(\beta(q(x) - b))\big] = g(q(x)) $$

a **monotone** function of $q$ — so ranking survives. Random assignment makes
drift harmless.

**Routed assignment breaks that.** With $b = b_{t(x)}$ for topic $t(x)$:

$$ \mathbb{E}[\tilde{y} \mid x] = \sigma\big(\beta(q(x) - b_{t(x)})\big) $$ (eq:bar-drift)

which is **not** a monotone function of $q(x)$ alone. **{{eq:bar-drift}} is
{{ch:ft-synthetic}}'s systematic error in a new setting**: a consistent,
content-correlated bias that more data confirms rather than dilutes.

> **IMPORTANT:** This is the sharpest practical corollary in the chapter.
> **Randomly assigned annotators with different bars are safe. Routed annotators
> with different bars are not** — and routing is the default, because work gets
> assigned by expertise.

### 5.5 The budget comparison

With cost $c$ per item read, a budget $B$ buys $B/c$ ratings or $B/2c$
comparisons. Equal-quality requires

$$ \mathcal{A}_{\text{bin}}(2N) \;\gtrless\; \mathcal{A}_{\text{pair}}(N) $$ (eq:budget-crossover)

which the measurement resolves as: **binary wins below the drift threshold and
loses above it**, with the crossover between drift 0.6 and 1.0 in these units.

## 6. Mathematical Foundation

### 6.1 The deconvolution, worked both ways

At high discrimination, $\alpha = 0.910 \Rightarrow p = 0.951$, and
$\mathcal{A}_{\text{obs}} = 0.758$ gives
$a = (0.758 + 0.951 - 1)/(2 \times 0.951 - 1) = 0.786$ — against a measured 0.766.

**The correction is worth 0.028 at high agreement and 0.146 at low agreement.**
That asymmetry is the point: {{eq:deconvolved-accuracy}} matters most exactly when
the raw number looks worst, which is when teams give up.

### 6.2 Why the collapse is a cliff

Ranking survives {{eq:bar-drift}} while the bar differences are small relative to
within-topic quality spread:

$$ \max_{t,t'} |b_t - b_{t'}| \lesssim \text{sd}\big(q \mid t\big) \;\Longrightarrow\; \text{ordering mostly preserved} $$ (eq:drift-threshold)

Above that, cross-topic comparisons invert wholesale rather than degrading
gradually. **{{eq:drift-threshold}} explains the shape**: a monotone-ish distortion
until the offsets exceed the signal, then a regime change.

**And it explains why within-topic accuracy falls too.** A single smooth model
fitted to four differently-offset labellings cannot satisfy them simultaneously,
so it compromises everywhere rather than only at the boundaries.

### 6.3 Implicit identification of offsets

The failed hypothesis, stated properly. Let the score be $s_\theta(x)$ with
$\theta$ in a smooth class. Within-topic constraints give
$s(x_a) > s(x_b)$ for $t(x_a) = t(x_b)$. **If topic supports overlap** — that is,
if $\exists\, x$ with non-negligible density under two topics' feature
neighbourhoods — then continuity of $s_\theta$ propagates the ordering across the
boundary, and no free per-topic constant exists to be unidentified.

$$ \text{offsets identified} \iff \text{topics not separated in } \text{supp}(\phi) $$ (eq:implicit-identification)

> **MATH NOTE:** {{eq:implicit-identification}} is why the spanning column made no
> difference here, and precisely what to check before generalising it. A
> mixture-of-experts reward model, or topics in genuinely disjoint feature
> regions, breaks the antecedent — and then spanning comparisons *are*
> load-bearing. **The experiment refutes the hypothesis in this regime and does
> not refute it in general.**

## 7. Internal Mechanics

```mermaid {#fig:pref-pipeline caption="A preference-data pipeline with the two measurements that should come first, and usually come last. Agreement determines whether any reward-model score is interpretable (eq:agreement-caps-measurement); bar spread determines whether free binary feedback is an asset or a liability (eq:bar-drift). Both are afternoon-sized measurements that change what you buy."}
flowchart TB
    RUB["rubric + worked examples<br/>THE highest-leverage artefact"] --> POOL["annotator pool"]
    POOL --> CAL{{"calibration set,<br/>double-labelled"}}
    CAL -->|"agreement rate"| M1["is any RM score<br/>interpretable?"]
    CAL -->|"bar spread"| M2["binary or pairwise?"]
    M2 -->|"tight"| BIN["binary: free, 2x volume"]
    M2 -->|"wide"| PAIR["pairwise: bar cancels<br/>eq:comparison-cancels-the-bar"]
    BIN --> RM["reward model / DPO"]
    PAIR --> RM
    M1 -->|"deconvolve<br/>eq:deconvolved-accuracy"| REPORT["report BOTH numbers"]
    RM --> REPORT
```

### 7.1 The order to spend in

1. **Write the rubric.** It moves bar spread, which decides everything downstream,
   and costs less than any quantity of labels.
2. **Double-label a calibration set.** Agreement (for
   {{eq:deconvolved-accuracy}}) and per-annotator positive rates (for
   {{eq:bar-drift}}) come from the same afternoon.
3. **Decide the format** from the bar spread, not from a paper.
4. **Then scale**, expecting {{eq:noise-is-a-tax}}'s rate.
5. **Report agreement beside every reward-model number.**

### 7.2 If you are stuck with routed binary feedback

Production thumbs cannot be retro-fitted into comparisons — there is nothing to
compare. What *can* be done:

| Remedy | Cost | Removes |
|---|---|---|
| shared calibration set per annotator | one afternoon | most of {{eq:bar-drift}} |
| per-annotator offset correction | trivial once measured | the identifiable part |
| randomise assignment where possible | scheduling friction | **all** of it |
| commission a small pairwise set | real | anchors the scale |

**Randomising assignment is the strongest and least used**, because
{{sec:5-formal-explanation}} shows random assignment reduces drift to a harmless
monotone distortion. It costs expertise-matching, which is often a real loss — but
it should be a decision rather than an accident.

### 7.3 What neither format fixes

Both inherit the deeper problem: **"better" is not a property of a response
alone.** It depends on who asked and why. A rubric addresses it; neither an
objective nor a data format does. {{cite:ouyang2022}}'s annotator-selection
sections are the least-retold and most useful part of that paper for this reason.

## 8. Implementation

```python {tier=A name=agreement-caps-measurement}
"""Annotator agreement is the ceiling, and it is also the measuring stick.

part:09 derived the Bradley-Terry model and the DPO objective. This listing is
about the data those objectives consume, and about a number that is almost never
reported alongside a reward-model result: how often two annotators labelling the
same pair give the same answer.

That number matters twice, and the second time is the one people miss. It bounds
what the reward model can LEARN, which is expected. It also bounds what any
evaluation can MEASURE, because the held-out labels come from the same noisy
process (eq:agreement-caps-measurement) -- so a perfect reward model scores the
agreement rate, not 100%.
"""
import numpy as np

rng = np.random.default_rng(191)

D, NF = 10, 400
N_ITEMS = 20000

W_Q = rng.normal(size=D)


def quality(X):
    """The latent quality a preference is 'really' about."""
    return np.tanh(X @ W_Q / np.sqrt(D)) + 0.35 * X[:, 0]


W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def annotate(qa, qb, beta):
    """A noisy annotator. beta is discrimination: high beta means the annotator
    reliably picks the better item, low beta means close calls are coin flips.
    This is the Bradley-Terry likelihood used as a NOISE model rather than as a
    training objective."""
    p = 1.0 / (1.0 + np.exp(-beta * (qa - qb)))
    return (rng.random(len(p)) < p).astype(int)


def train_rm(Xa, Xb, y, steps=500, lr=1.0, lam=1e-3):
    """Fit a scoring function under the Bradley-Terry loss on pair differences."""
    Pa, Pb = feat(Xa), feat(Xb)
    Dm = Pa - Pb
    w = np.zeros(NF)
    for _ in range(steps):
        z = Dm @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = Dm.T @ (p - y) / len(y) + lam * w
        w -= lr * g
    return w


def make_pairs(n):
    Xa = rng.normal(size=(n, D)); Xb = rng.normal(size=(n, D))
    return Xa, Xb, quality(Xa), quality(Xb)


BETAS = (0.8, 1.5, 3.0, 8.0)
N_TRAIN = 8000

Xta, Xtb, qta, qtb = make_pairs(6000)
truth_te = (qta > qtb).astype(int)

print(f"{N_TRAIN:,} training pairs, {len(truth_te):,} test pairs. The reward "
      f"model never\nsees the latent quality -- only the annotator's choices.\n")
print(f"{'annotator':>10}{'two annotators':>17}{'RM accuracy':>14}"
      f"{'RM accuracy':>14}{'apparent':>11}")
print(f"{'beta':>10}{'agree':>17}{'vs TRUTH':>14}{'vs LABELS':>14}"
      f"{'shortfall':>11}")
print("-" * 66)

rows = {}
for beta in BETAS:
    Xa, Xb, qa, qb = make_pairs(N_TRAIN)
    y = annotate(qa, qb, beta)
    w = train_rm(Xa, Xb, y)

    pred = ((feat(Xta) - feat(Xtb)) @ w > 0).astype(int)
    lab1 = annotate(qta, qtb, beta)
    lab2 = annotate(qta, qtb, beta)
    agree = float((lab1 == lab2).mean())
    a_truth = float((pred == truth_te).mean())
    a_label = float((pred == lab1).mean())
    rows[beta] = (agree, a_truth, a_label)
    print(f"{beta:>10.1f}{agree:>17.3f}{a_truth:>14.3f}{a_label:>14.3f}"
          f"{a_truth - a_label:>+11.3f}")

print("\n\nDoes more data climb past the annotator?\n")
print(f"{'pairs':>9}" + "".join(f"{'beta=' + str(b):>14}" for b in BETAS))
print(f"{'':>9}" + "".join(f"{'(vs truth)':>14}" for b in BETAS))
print("-" * 65)
scale = {}
for n in (500, 2000, 8000, 32000):
    line = []
    for beta in BETAS:
        Xa, Xb, qa, qb = make_pairs(n)
        w = train_rm(Xa, Xb, annotate(qa, qb, beta))
        acc = float((((feat(Xta) - feat(Xtb)) @ w > 0).astype(int)
                     == truth_te).mean())
        line.append(acc)
        scale[(n, beta)] = acc
    print(f"{n:>9,}" + "".join(f"{v:>14.3f}" for v in line))

lo, hi = rows[BETAS[0]], rows[BETAS[-1]]
b0, b8 = BETAS[0], BETAS[-1]
print(f"""
The first table contains the sentence this listing exists for. Look at the
low-discrimination row: two independent annotators agree {lo[0]:.1%} of the time,
the reward model is {lo[1]:.1%} accurate against the LATENT TRUTH, and it scores
{lo[2]:.1%} when measured against a held-out annotator label.

The reward model is substantially better than its own evaluation says it is. Not
because the evaluation is badly built, but because the held-out labels come from
the same noisy process as the training labels, so a model predicting the truth
perfectly would still disagree with them at the annotator's error rate
(eq:agreement-caps-measurement).

That is a reporting failure that runs in both directions. A reward-model accuracy
of {lo[2]:.0%} sounds poor and might mean the model is nearly as good as the task
permits. The same {lo[2]:.0%} on a task where annotators agree {hi[0]:.0%} of the
time would mean the model is badly broken. The number alone distinguishes
nothing, and the agreement rate is almost never published beside it.

Watch the shortfall column shrink as discrimination rises: {lo[1]-lo[2]:+.3f} at
beta={b0}, {hi[1]-hi[2]:+.3f} at beta={b8:.0f}. Clean labels make the measurement
honest. Noisy labels make it pessimistic by exactly the amount nobody quantifies.

The second table answers the obvious next question, and the answer is not the one
the first table sets you up to expect.

More data does climb past the annotator. In the beta={b0} column, 500 to 32,000
pairs moves accuracy against truth from {scale[(500, b0)]:.3f} to
{scale[(32000, b0)]:.3f}. The clean beta={b8:.0f} column ends at
{scale[(32000, b8)]:.3f}. At 64x the data, the noisy annotator has closed most of
a gap that looked structural.

So annotator noise of this kind is a TAX ON DATA EFFICIENCY, not a ceiling on
capability. The reason is that the noise is unbiased -- an annotator who is
merely uncertain is wrong in both directions with roughly equal probability, and
enough independent draws average that out (eq:noise-is-a-tax). Compare where
beta={b0} and beta={b8:.0f} reach the same accuracy: the clean annotator hits
{scale[(8000, b8)]:.3f} at 8,000 pairs, and the noisy one has not matched it at
32,000. Somewhere past 4x the annotation budget, for the same result.

Now put the two tables together, because the combination is the actionable part
and neither half implies it alone.

You CAN buy your way past noisy preference labels with volume. You can NEVER
measure that you did, because the measurement is capped by the same noise at
{lo[2]:.0%} no matter how good the model becomes. A team in this position sees a
reward model that refuses to score above the high fifties, concludes the approach
is not working, and stops -- while the model is at {lo[1]:.0%} against the thing
they actually care about.

That is a failure of instrumentation producing a wrong decision, and it is
invisible without double-labelling.

Which gives the ordering for a preference-data budget, roughly the reverse of how
these projects usually run. Double-label a small sample FIRST and compute
agreement. If agreement is low, decide deliberately between two different
projects: fix the task definition, rubric, or annotator pool to raise the ceiling
on measurement, or accept the ceiling and buy the volume the second table says
you need. Both are legitimate; conflating them is not.

And report the agreement rate beside every reward-model number, because without it
the number is not interpretable by anyone, including the team that produced it."""
)
```

The first listing is about whether your number means anything. The second is about
which kind of judgement to buy.

```python {tier=A name=binary-versus-pairwise}
"""Pairwise or binary? It depends on one measurable property of your annotators.

cite:ethayarajh2024kto makes the practical case for binary feedback: production
produces thumbs-up and thumbs-down for free, in volumes no comparison campaign
can match. The usual counter-argument is that a comparison carries more
information per judgement.

Both are true, and the comparison people actually face is at equal ANNOTATION
BUDGET rather than equal example count -- a pairwise judgement requires reading
two items, a binary one requires reading one. So binary starts with twice the
judgements for the money.

What decides it is a property of the annotators. Asked "is this good?", each
person applies their own bar, and real annotation is ROUTED: one person takes the
coding queries, another the writing queries, so which bar an item is measured
against is a function of what the item is about. A COMPARISON is immune -- the
annotator's bar appears on both sides and cancels (eq:comparison-cancels-the-bar).
A rating is not.

This listing also tests a hypothesis that turned out to be wrong, and reports it,
because the reason it is wrong is worth knowing.
"""
import numpy as np

rng = np.random.default_rng(199)

D, NF, C = 10, 400, 4             # dims, random features, topic areas
BETA = 5.0

W_Q = rng.normal(size=D)
U_TOPIC = rng.normal(size=(C, D))
W_RF = rng.normal(size=(D, NF)) * 0.9
B_RF = rng.uniform(0, 2 * np.pi, NF)


def quality(X):
    return np.tanh(X @ W_Q / np.sqrt(D)) + 0.35 * X[:, 0]


def topic(X):
    return (X @ U_TOPIC.T).argmax(axis=1)


def feat(X):
    return np.cos(X @ W_RF + B_RF)


def noisy(delta):
    return (rng.random(len(delta)) < 1.0 / (1.0 + np.exp(-BETA * delta)))


def logistic_fit(P, y, steps=600, lr=1.0, lam=1e-3):
    w = np.zeros(P.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(P @ w)))
        w -= lr * (P.T @ (p - y) / len(y) + lam * w)
    return w


def draw(n):
    X = rng.normal(size=(n, D))
    return X, quality(X), topic(X)


def train_binary(n_items, bars):
    """One reading per judgement. The rating is quality against THIS topic's
    annotator's bar, and nothing in the data identifies that bar."""
    X, q, t = draw(n_items)
    return logistic_fit(feat(X), noisy(q - bars[t]).astype(float))


def build_pairs(n_pairs, cross):
    """Exactly n_pairs, of which `cross` are between different topics. Both
    columns must train on the SAME number of comparisons or the comparison is
    about sample size rather than about spanning."""
    n_cross = int(n_pairs * cross)
    n_within = n_pairs - n_cross
    Xa, qa, ta = draw(n_pairs * 12)
    Xb, qb, tb = draw(n_pairs * 12)
    same = ta == tb
    iw = np.flatnonzero(same)[:n_within]
    ic = np.flatnonzero(~same)[:n_cross]
    i = np.concatenate([iw, ic])
    assert len(i) == n_pairs, (len(i), n_pairs)
    return Xa[i], Xb[i], qa[i], qb[i]


def train_pairwise(n_pairs, bars, cross):
    """Two readings per judgement. An annotator's bar cancels in a comparison,
    so the bars never enter the labels at all."""
    Xa, Xb, qa, qb = build_pairs(n_pairs, cross)
    return logistic_fit(feat(Xa) - feat(Xb), noisy(qa - qb).astype(float))


# Test sets: pairs drawn WITHIN one topic, and pairs drawn ACROSS topics.
def test_pairs(n, want_cross):
    Xa, qa, ta = draw(n * 4)
    Xb, qb, tb = draw(n * 4)
    m = (ta != tb) if want_cross else (ta == tb)
    i = np.flatnonzero(m)[:n]
    return feat(Xa[i]), feat(Xb[i]), (qa[i] > qb[i]).astype(int)


FA_W, FB_W, Y_W = test_pairs(3000, False)
FA_C, FB_C, Y_C = test_pairs(3000, True)


def acc(w):
    return (float((((FA_W - FB_W) @ w > 0).astype(int) == Y_W).mean()),
            float((((FA_C - FB_C) @ w > 0).astype(int) == Y_C).mean()))


BUDGET = 16000
DRIFTS = (0.0, 0.3, 0.6, 1.0)

print(f"{C} topic areas, one annotator each, {BUDGET:,} item-readings of budget.")
print(f"That buys {BUDGET:,} binary ratings or {BUDGET // 2:,} comparisons.\n")
print(f"{'annotator':>10}" + f"{'binary ratings':>22}"
      + f"{'pairwise, within only':>25}" + f"{'pairwise, 30% span':>22}")
print(f"{'bar drift':>10}" + "".join(f"{'within':>11}{'ACROSS':>11}"
                                     for _ in range(3)))
print("-" * 76)

rows = {}
for drift in DRIFTS:
    bars = drift * rng.normal(size=C)
    b = acc(train_binary(BUDGET, bars))
    p0 = acc(train_pairwise(BUDGET // 2, bars, cross=0.0))
    p3 = acc(train_pairwise(BUDGET // 2, bars, cross=0.30))
    rows[drift] = (b, p0, p3)
    print(f"{drift:>10.1f}" + "".join(f"{v[0]:>11.3f}{v[1]:>11.3f}"
                                      for v in (b, p0, p3)))

z, d6, hi = rows[0.0], rows[0.6], rows[1.0]
bz, pz, qz = z
b6, p6, q6 = d6
bh, ph, qh = hi
print(f"""
Start with the drift-zero row, where every topic's annotator happens to share a
bar. Binary wins on both test sets: {bz[0]:.3f} and {bz[1]:.3f} against
pairwise's {pz[0]:.3f} and {pz[1]:.3f}. That is the information-per-cost argument
working as advertised -- a rating costs half the reading of a comparison, so the
same budget buys twice as many, and twice as many noisy absolute judgements beats
half as many clean relative ones.

Now go down the binary column. At drift {0.3} and {0.6} it is still competitive:
{b6[0]:.3f} and {b6[1]:.3f}. Then at drift {1.0} it falls off a cliff, to
{bh[0]:.3f} and {bh[1]:.3f}.

The pairwise columns do not move at all, at any drift. That is not robustness in
the statistical sense -- it is exact cancellation. When an annotator ranks two
items their bar appears on both sides of the comparison and subtracts out, so the
bars never enter the labels (eq:comparison-cancels-the-bar). The pairwise numbers
are flat because the quantity being varied cannot reach them.

Two things are worth taking from the shape of the binary column rather than just
its endpoints.

It is a CLIFF, not a slope. Binary feedback is fine, and better than pairwise, up
to a threshold -- and then it is much worse. That means "how much do our
annotators disagree about where the bar is" is not a nice-to-know: it is the
variable that decides which kind of data to buy, and it has a sharp answer rather
than a gradual one.

And the damage is not confined to the topic boundary. At drift {1.0} the WITHIN-
topic accuracy is {bh[0]:.3f}, barely better than the across-topic {bh[1]:.3f}.
The routed bars do not merely misalign the topics relative to each other; they
corrupt the learned score badly enough that ordering fails inside a topic too,
because the model is fitting one smooth function to labels that four different
questions produced.

Now the hypothesis that failed, which was the reason for the third column.

The expectation was that within-topic comparisons would leave the offsets BETWEEN
topics unidentified -- that a reward model trained only on same-topic pairs would
rank correctly inside each topic and arbitrarily between them, and that a share of
topic-spanning comparisons would be needed to tie the scale together. The third
column adds 30% spanning pairs at identical sample size to test it.

It made no difference: {qh[1]:.3f} against within-only's {ph[1]:.3f} at the
highest drift, and the columns are within noise of each other at every row.

The reason is worth more than the hypothesis was. A reward model is a smooth
function of content, not a lookup table with a free parameter per topic. Topics
overlap in feature space, and an ordering learned in one region constrains the
function in neighbouring regions, so the cross-topic offsets are identified
implicitly even though nothing in the data compares across topics directly.

Which comes with a condition to check rather than assume. That identification
relies on the topics not being genuinely disjoint in whatever representation the
reward model uses. If they were -- separate specialist heads, or topics so
different that no shared feature relates them -- the original worry would apply
and spanning comparisons would be load-bearing. In this setup they were not, and
saying so is more useful than quietly deleting the column.

So the practical rules, in order.

Measure the bar spread before choosing a data format. Have several annotators
rate one shared calibration set and compare their positive rates. That measurement
costs an afternoon and determines whether the free production feedback is an asset
or a liability.

If the spread is tight, take the binary data and the volume that comes with it.
If it is wide, either fix it -- a rubric, worked examples and a calibration round
move the bar spread far more cheaply than either kind of label -- or commission
comparisons, which are immune by construction.

And note the ordering that implies: the rubric is the highest-leverage artefact in
a preference-data project, because it moves the term that decides everything else.
The choice between a KTO-style objective and a DPO-style one is downstream of it.""")
```

## 9. Practical Example

**Agreement bounds the ruler.** At an annotator agreement of **56.9%**, the reward
model is **71.3%** accurate against latent truth and measures **58.9%** against
held-out labels — a shortfall of **+0.124**. At **91.0%** agreement the shortfall
is **+0.008**.

**{{eq:agreement-caps-measurement}} says a perfect model scores $p$, not 1**, and
{{eq:deconvolved-accuracy}} recovers the truth: 0.589 observed at $p = 0.686$
implies $a \approx 0.735$ against a measured 0.713. **The correction is worth
0.146 at low agreement and 0.028 at high** — largest exactly when the raw number
looks worst.

> **IMPORTANT:** A reward-model accuracy of 59% might mean near-optimal on an
> ambiguous task or broken on a clean one. **Without the agreement rate the number
> is uninterpretable**, and it is almost never published.

**Noise is a tax, not a wall.** 500 → 32,000 pairs moves the noisy annotator
**0.567 → 0.762**, against the clean annotator's **0.786**.
{{eq:noise-is-a-tax}}: unbiased noise averages out, at roughly 4× the budget for
the same result — the clean annotator reached **0.773 at 8,000** pairs and the
noisy one had not matched it at 32,000.

**Together these produce the failure that matters.** You can buy past noisy labels;
**you cannot measure that you did.** A team sees a reward model stuck in the high
fifties, concludes the approach fails, and stops — while it sits at 71% on the
thing they care about. **A wrong decision caused by instrumentation, invisible
without double-labelling.**

**Binary versus pairwise is decided by bar drift.** At zero drift binary wins
**0.758 / 0.753** against **0.744 / 0.747** — twice the judgements per pound. At
drift 1.0 binary collapses to **0.636 / 0.624** while pairwise holds **0.749 /
0.735.**

**Pairwise is flat at every drift because {{eq:comparison-cancels-the-bar}} is
exact** — the bar appears on both sides and subtracts. That is cancellation, not
robustness.

**And it is a cliff.** Binary is competitive at drift 0.3 and 0.6, then falls.
{{eq:drift-threshold}} explains the shape, and **the damage is not confined to the
boundary**: within-topic accuracy falls to 0.636 against across-topic 0.624,
because one smooth model cannot satisfy four differently-offset labellings.

**A hypothesis that failed.** Adding 30% topic-spanning comparisons at identical
sample size changed nothing: **0.735 against 0.734.** Cross-topic offsets are
identified *implicitly* through the score's smoothness
({{eq:implicit-identification}}) — topics overlap in feature space, so orderings
propagate. **Reported rather than deleted, because the condition it depends on is
checkable and would fail for a mixture-of-experts reward model.**

## 10. Production Considerations

**Write the rubric first.** It moves bar spread, which decides the format, and
costs less than any quantity of labels.

**Double-label a calibration set before committing a budget.** Agreement and
per-annotator positive rates come from the same afternoon.

**Report agreement beside every reward-model number**, and deconvolve with
{{eq:deconvolved-accuracy}} before concluding a model is inadequate.

**Randomise annotator assignment where you can afford to** — it converts
{{eq:bar-drift}} into a harmless monotone distortion.

**If assignment must be routed, measure per-annotator offsets and correct.**

**Budget for {{eq:noise-is-a-tax}}'s rate** — 4× the data for the same result at
low agreement is a plan, not a surprise.

**Apply {{ch:ft-datasets}}'s group splits to preference pairs.** Two paraphrases
of one prompt across the split leak exactly as before.

## 11. Common Mistakes

**Reporting reward-model accuracy without the agreement rate.**

**Concluding a preference project has failed** from a score that is capped by
{{eq:agreement-caps-measurement}}.

**Treating annotator disagreement as a reason not to proceed** — it is a budget
multiplier if it is uncertainty.

**Choosing binary or pairwise from a paper** rather than from measured bar spread.

**Assuming annotator differences average out.** They do under random assignment
and not under routing.

**Retro-fitting comparisons from production thumbs.** There is nothing to compare.

**Skipping the rubric** because it feels like process rather than engineering.

**Splitting preference pairs at random.**

## 12. Failure Modes

**Reward model stuck at a low score whatever you do.** Cause:
{{eq:agreement-caps-measurement}}. Fix: deconvolve, then decide.

**Model good in evaluation, policy drifts oddly after optimisation.** Cause:
{{eq:bar-drift}} systematically favouring one topic.

**Adding preference data barely helps.** Cause: {{eq:noise-is-a-tax}}'s rate —
expected, and a reason to fix the rubric instead.

**Binary feedback that worked at small scale degrades as the annotator pool
grows.** Cause: more annotators, more bar spread, and routing.

**Within-topic quality also degrades.** Cause: the compromise fit of
{{eq:drift-threshold}}, not a separate problem.

**Two annotators disagree and both are right.** Cause: the task genuinely
under-specifies "better". Only the rubric addresses it.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| pairwise comparisons | 2× reading cost | bar spread is wide |
| binary feedback ({{cite:ethayarajh2024kto}}) | bar sensitivity | spread is tight; volume is free |
| scalar ratings (1–5) | worse bar sensitivity | rarely worth it |
| rankings of $k$ items | annotator fatigue | when $k$ candidates already exist |
| AI feedback | inherits {{ch:ft-synthetic}} entirely | scale beats fidelity |
| more rubric, less data | project time | almost always first |

**The last row is the recommendation.** Every other row is downstream of bar
spread and agreement, and the rubric is the only intervention that moves both.

**And note what AI feedback imports.** {{ch:ft-synthetic}}'s
{{eq:self-eval-agreement}} applies directly: a model judging preferences shares
the generator's misconceptions and certifies them. Preference data has **no
execution oracle**, so the one domain where synthetic data is cleanly checkable is
precisely the one this chapter cannot use.

## 14. Evaluation

**Report the agreement rate.** Always, next to the accuracy.

**Report the deconvolved accuracy** ({{eq:deconvolved-accuracy}}) alongside the
raw one.

**Report the annotator count and assignment policy** — routed or random changes
the interpretation entirely.

**Report per-annotator positive rates** on a shared calibration set.

**Split preference pairs by group** per {{ch:ft-datasets}}.

**Evaluate the policy, not only the reward model.** A reward model good on
average can be systematically wrong in one region, which is
{{eq:bar-drift}}'s signature and invisible in an aggregate score.

## 15. Advanced Concepts

**Deconvolution should be standard reporting.** {{maturity:EMERGING}}
{{eq:deconvolved-accuracy}} is elementary and almost never applied, and it changes
conclusions most where teams are most likely to abandon a working approach.

**Bar drift as a fairness problem.** {{maturity:EMERGING}} {{eq:bar-drift}} says
routed annotation encodes each annotator's standards into the model *for their
topic*. When routing follows demographic or linguistic lines rather than technical
ones, that is a mechanism for systematically different treatment — with the same
mathematics and much higher stakes.

**Online versus offline is downstream of this.** {{maturity:MATURE}}
{{ch:fm-dpo}} covers the distinction; note that fresh on-policy data does not fix
either problem in this chapter, because both are properties of the *annotation*
rather than of the sampling distribution.

**AI feedback has no oracle.** {{maturity:EXPERIMENTAL}} It is the cheapest
scaling path and the one {{ch:ft-synthetic}}'s argument most directly warns
against, with no execution check available to break the agreement.

**Rubrics as the actual research object.** {{maturity:RESEARCH FRONTIER}}
Everything here says the rubric dominates, and essentially all published effort
goes to objectives instead. **What makes a rubric reduce bar spread is not
systematically studied, and it is the highest-leverage unstudied thing in the
pipeline.**

**The deepest limitation is that a preference is not a measurement.**
{{maturity:EMERGING}} Every method here optimises agreement with a comparison
someone made between two outputs, and that comparison is influenced by length,
formatting and confidence as much as by correctness — which is the same
correlation {{ch:ev-llm-judge}} finds in automated judges, arriving here as the
training signal rather than the evaluation. **Optimising against a biased
comparator moves the model toward the bias**, and no amount of algorithmic care in
the optimiser corrects a systematically tilted preference set.

## 16. Connection to Previous Chapters

{{ch:fm-rlhf}} and {{ch:fm-dpo}} supply the objectives this chapter feeds.
{{ch:ft-synthetic}}'s systematic-versus-random distinction is exactly
{{eq:bar-drift}} versus unbiased annotator noise — **the same dichotomy, one
chapter later, deciding a different question.** {{ch:ft-datasets}}'s provenance
discipline applies unchanged to preference pairs, and its
{{eq:metric-inherits-bias}} is the sibling of
{{eq:agreement-caps-measurement}}: one is an evaluation blind to what selection
missed, the other an evaluation capped by the noise it shares.
Forward: {{ch:ft-training-config}} measures what optimising against these
preferences costs elsewhere in the model; {{part:25}} owns evaluation, and
{{eq:deconvolved-accuracy}} belongs in its reporting standards.

## 17. Exercises

1. Derive {{eq:p-from-agreement}} from $\alpha = p^2 + (1-p)^2$ and compute $p$
   for $\alpha = 0.65, 0.80, 0.95$.
2. A reward model scores 0.64 where annotators agree 0.72. What is its accuracy
   against truth?
3. Prove {{eq:comparison-cancels-the-bar}} and state one assumption that makes it
   fail in practice.
4. Show that random annotator assignment makes bar drift a monotone transform of
   $q$, and explain why ranking survives.
5. In `binary-versus-pairwise`, set the number of topics to 1. What happens to the
   binary column and why?
6. In the same listing, randomise annotator assignment while keeping the bar
   spread. Confirm the prediction of {{sec:5-formal-explanation}}.
7. Using {{eq:noise-is-a-tax}}, estimate the budget needed to reach a clean
   annotator's 8,000-pair accuracy at $\alpha = 0.60$.
8. Design the calibration set you would use to measure both agreement and bar
   spread in one pass. How many items and how many annotators?

## 18. Interview Questions

1. Your reward model scores 58% on held-out preferences. Is that bad?
2. What does annotator agreement bound — learning, measurement, or both?
3. Why is unbiased annotator noise a budget problem rather than a ceiling?
4. When is binary feedback better than pairwise, and when much worse?
5. Why exactly are comparisons immune to annotator threshold differences?
6. Why does randomly assigning annotators make their differences harmless?
7. What is the single highest-leverage artefact in a preference-data project?
8. Why can't production thumbs-up data be converted into comparisons?
9. Why does bar drift hurt within-topic accuracy and not just across topics?
10. What does AI feedback inherit from the synthetic-data chapter?

## 19. Research Questions

1. {{eq:deconvolved-accuracy}} is elementary and unused. How many published
   reward-model comparisons change ranking once deconvolved?
2. What rubric properties actually reduce bar spread? The question is answerable
   experimentally and appears not to have been asked systematically.
3. {{eq:implicit-identification}} held here because topics overlapped in feature
   space. Does it hold for real reward models, and does it fail for
   mixture-of-experts architectures?
4. {{eq:drift-threshold}} predicts a regime change rather than a gradient. Where
   is the threshold for real annotation pools, in units of within-topic quality
   spread?
5. Routed annotation encodes per-annotator standards per topic. How much of
   observed model behaviour variation across domains is this rather than
   capability?

## 20. Chapter Summary

**Annotator agreement bounds the ruler, not only the model.** At 56.9% agreement a
reward model was **71.3%** accurate against truth and measured **58.9%** —
{{eq:agreement-caps-measurement}} means a *perfect* model scores $p$, not 1, and
{{eq:deconvolved-accuracy}} recovers 0.735 from the observed 0.589. **The
correction is worth 0.146 at low agreement and 0.028 at high**, which is exactly
backwards from when people bother to apply it.

**And unbiased noise is a tax, not a wall.** 500 → 32,000 pairs took the noisy
annotator **0.567 → 0.762** against a clean **0.786** — roughly 4× the budget for
the same result ({{eq:noise-is-a-tax}}).

**The combination is the failure that matters: you can buy your way past noisy
labels and you can never measure that you did**, so the project is abandoned on an
artefact of its own instrumentation.

**Binary versus pairwise is decided by bar drift, and it is a cliff.** Binary won
at zero drift (**0.758** against **0.744**, twice the judgements per pound), stayed
level at 0.3 and 0.6, then collapsed to **0.636** — while pairwise held **0.749**
at every level, because {{eq:comparison-cancels-the-bar}} is exact rather than
approximate.

**The deciding property is routing, not disagreement.** Randomly assigned
annotators with different bars produce a harmless monotone distortion; **routed
annotators produce {{eq:bar-drift}}, which is {{ch:ft-synthetic}}'s systematic
error in a new setting** — content-correlated, confirmed by more data rather than
diluted. And it damages within-topic accuracy too, because one smooth model cannot
satisfy four differently-offset labellings.

**One hypothesis failed and is reported.** Topic-spanning comparisons made no
difference (**0.735 against 0.734**), because {{eq:implicit-identification}}
identifies cross-topic offsets through smoothness. **The condition is checkable
and would fail for a disjoint or expert-partitioned reward model**, which is why
the negative result is worth more than the guess was.

**Which yields an ordering that is the reverse of how these projects run: rubric,
then calibration set, then format, then volume.** The rubric moves bar spread and
agreement — the two terms everything else is downstream of — **and it is the
cheapest thing in the pipeline and the least studied.**

## 21. Further Reading

{{cite:ethayarajh2024kto}} for the binary-feedback case, read alongside this
chapter's measurement: its practical advantage is real and conditional on a
property of your annotators that the paper cannot know.
{{cite:rafailov2023}} for the objective, whose derivation {{ch:fm-dpo}} owns; note
how little of the difficulty in a real preference project lives in the loss.
{{cite:ouyang2022}} for the annotator-selection and agreement discussion, which is
the most useful and least retold part of that paper.
{{cite:wang2023selfinstruct}} for AI-generated feedback, and
{{ch:ft-synthetic}} for why preference data is the worst case for it.
{{cite:zhou2023lima}} as the reminder that a small, carefully specified dataset
frequently beats a large ambiguous one — which is this chapter's rubric argument
arriving from another direction.
