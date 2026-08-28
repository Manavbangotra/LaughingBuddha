---
id: rsn-vs-generation
number: 146
part: XVI
tier: full
status: draft
requires: [fm-what-they-are, ft-datasets, mle-splits]
provides: [accuracy-does-not-separate, invariance-criterion,
           criteria-without-labels, paraphrase-consistency,
           extrapolation-diversity, reliability-gap]
citations: [mirzadeh2024gsmsymbolic, sprague2024tocot, turpin2023faithfulness,
            wei2022cot, cobbe2021gsm8k]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state why held-out accuracy cannot
distinguish computing an answer from fitting one; name three perturbations that
can, and what each tests; run two of them **without ground truth**, on production
traffic; explain why ensemble disagreement is a weaker uncertainty signal than it
appears; and read a reasoning benchmark score as a point estimate on a curve
rather than as a property of a model.

## 2. Why This Matters

{{part:15}} made a model run. This part is about making it *think*, and it opens
with the distinction everything else depends on.

**"Reasoning" gets applied to any system that answers problems that look like they
need reasoning.** That is not a definition, and {{sec:9-practical-example}} shows
it does not distinguish a system that **computes** the answer from one that has
learned what answers to problems of this shape usually look like.

Two systems, same task, same training data. On held-out problems drawn the same
way: **100.0% and 100.0%.** A benchmark cannot tell them apart, and neither can
any amount of held-out data drawn from the same distribution.

**Three perturbations separate them completely, and none changes what the problem
means.** Doubling the numbers: **0.1% against 100%.** Adding one irrelevant
clause: **0.2%.** A template never seen: **0.5%.**

> **That is {{cite:mirzadeh2024gsmsymbolic}}'s experiment**, which reports the
> same shape on real models — performance declining when only the numbers change,
> and drops of **up to 65%** from a single irrelevant sentence.

**And the degradation is graceful, which is why it goes unnoticed.** Mean relative
error runs **0.001 → 0.812** as the numbers double: not nonsense, but answers of
roughly the right size and the wrong value.

**Then the criteria you can actually deploy.** Paraphrase consistency needs no
ground truth — the same problem, phrased twice — and agreement falls **100.0% →
1.1%** as the distribution shifts. **It measures whether the surface form is
entering the computation**, which is exactly the question.

**And a caution about the criterion people reach for first.** Ensemble spread
separated in- from out-of-distribution with an **AUROC of 1.00** here, for a
reason that will not transfer: random-feature models extrapolate *independently*.
**Models trained the same way on the same data are wrong in correlated ways, so
they agree while being wrong.**

{{maturity:MATURE}} Perturbation-based evaluation. {{maturity:EMERGING}}
Label-free consistency checks in production. {{maturity:RESEARCH FRONTIER}}
Compositional generalisation as a measurable property.

## 3. Prerequisites

{{ch:fm-what-they-are}} for what next-token prediction is and is not;
{{ch:ft-datasets}} for the selection-bias argument this chapter is a special case
of; {{ch:mle-splits}} for why a held-out set drawn the same way answers a narrower
question than people take it to.

> **NOTE:** *Reasoning* is overloaded. {{part:11}} uses it for combining retrieved
> documents. **This part means multi-step inference toward an answer**, and the
> retrieval sense is never intended.

## 4. Intuitive Explanation

### The two systems, and why a benchmark cannot separate them

Build a task with a known structure — word problems generated from templates,
each template a small arithmetic rule. Then build two systems:

- one that **maps the problem to its rule and evaluates it**;
- one **fitted to the surface form** — which words are present, and the numbers —
  which is what a learned model sees.

```text
   test condition                        learned    rel err   reasoner
   ───────────────────────────────────   ───────    ───────   ────────
   held-out, same templates and range     100.0%      0.001     100.0%
   numbers 2x larger                        0.1%      0.812     100.0%
   numbers 10x larger                       0.1%      1.368     100.0%
   one irrelevant clause added              0.2%      5.542     100.0%
   templates never seen in training         0.5%      2.005     100.0%
```

**The first row is the one that appears in a paper.** Every other row separates
them, and **none of the perturbations changes the answer.**

### What each perturbation tests

**Changing the numbers** tests whether the system evaluates a rule or interpolates
between remembered cases. A rule is indifferent to magnitude; an interpolation has
a region it was fitted on and nothing outside it.

**Adding an irrelevant clause** tests whether the system knows which parts of the
surface matter. A rule mentions the quantities it uses; a fitted model has every
feature in its input and no way to know which are load-bearing.

**A novel template** tests whether the system can handle structure it has not
seen — and it is the one the executing system also fails, honestly, because nobody
gave it the rule.

### The degradation is graceful, which is the problem

Relative error runs **0.001 → 0.812 → 1.368** across the perturbations. The fitted
model does not start producing nonsense: it produces answers of roughly the right
size and the wrong value.

**A wrong answer that looks wrong gets caught. A wrong answer that looks
reasonable does not.**

And the sweep shows there is no threshold:

```text
   number range    exact    rel error    gap from baseline
   ────────────    ─────    ─────────    ─────────────────
   2–20            100.0%       0.001                 0.0%
   3–30             44.1%       0.064                55.9%
   4–40             15.3%       0.529                84.7%
   10–100            0.5%       1.533                99.5%
```

> **Any single benchmark score is a point estimate on a curve whose slope nobody
> reports.**

### Two criteria that need no ground truth

The perturbations above need labels — you must know the right answer to the
perturbed problem. **Two criteria do not**, which makes them runnable against
production traffic.

**Paraphrase consistency.** Ask the same problem two ways. In distribution the
fitted model's two phrasings agree **100.0%** of the time; at double the number
range, **1.1%**, with mean disagreement **37.15**.

The executing system agrees with itself **100%** at every range — not because it
is better, but because **a system that maps to a rule and evaluates it has nothing
left that could depend on the phrasing.**

> **Paraphrase consistency is not a proxy for correctness. It is a proxy for
> whether the surface form is entering the computation**, which is the question
> this chapter is about — and disagreement is evidence regardless of which answer
> was right.

**Ensemble disagreement**, with a large caveat. Spread rose **2,354×** at double
the range, separating in- from out-of-distribution perfectly (**AUROC 1.00**).

**That should be read with suspicion.** Random-feature models extrapolate
*independently*, so outside the fitted region they fly apart. Members of a real
ensemble — same architecture, same data, same inductive biases — **extrapolate
similarly**, so they agree while being wrong.

> **Ensemble disagreement measures diversity of extrapolation, not distance from
> the training distribution.** Those coincide only when the members are genuinely
> diverse, and sampling one model repeatedly at temperature is the weakest version
> of all — every sample shares every parameter.

### And the criterion that matters most is the hardest

**Compositional generalisation** — a rule built from operations the system has
seen, in a combination it has not. The fitted model scores **0.4%**.

**The executing system scores 0% too**, because nobody gave it the rule. It cannot
induce structure from examples.

> **Neither system is doing what "reasoning" is usually meant to name.**
> Constructing one that composes known operations into unseen combinations is the
> actual open problem, not a measurement problem — and this chapter is about being
> able to tell whether you have.

## 5. Formal Explanation

### 5.1 Why a held-out set answers a narrower question

For a distribution $\mathcal{D}$ and systems $f$ (computes) and $g$ (fitted),
held-out accuracy estimates

$$ \mathbb{E}_{x \sim \mathcal{D}}\big[\mathbb{1}[f(x) = y(x)]\big] $$

**which is a property of the pair (system, distribution)**, not of the system. If
$g$ was fitted on $\mathcal{D}$ and $f$ is correct everywhere,

$$ \mathbb{E}_{\mathcal{D}}[f] = \mathbb{E}_{\mathcal{D}}[g] \quad\text{while}\quad \mathbb{E}_{\mathcal{D}'}[f] \gg \mathbb{E}_{\mathcal{D}'}[g] $$ (eq:accuracy-does-not-separate)

for any $\mathcal{D}' \ne \mathcal{D}$. **{{eq:accuracy-does-not-separate}} is why
no amount of held-out data helps**: more samples from $\mathcal{D}$ estimate the
same quantity more precisely.

### 5.2 The invariance criterion

Let $T$ be a transformation with $y(T x) = y(x)$ — it changes the problem's
surface and not its answer. Then a correct system satisfies

$$ f(Tx) = f(x) \quad \forall x, \forall T \in \mathcal{T} $$ (eq:invariance-criterion)

**{{eq:invariance-criterion}} is testable without knowing $y$**, which is the
useful part: you need only that $T$ preserves the answer, not that you can compute
it.

Measured violation is the reliability gap:

$$ \Delta(T) = \mathbb{E}_{\mathcal{D}}[f] - \mathbb{E}_{T\mathcal{D}}[f] $$ (eq:reliability-gap)

### 5.3 Two label-free criteria

**Paraphrase consistency.** For two surface realisations $s_1, s_2$ of one
problem:

$$ C = P\big(f(s_1) = f(s_2)\big) $$ (eq:criteria-without-labels)

**{{eq:criteria-without-labels}} needs no $y$ at all.** And it is a *lower bound*
on error: $f(s_1) \ne f(s_2)$ implies at least one is wrong, so

$$ \text{error rate} \;\ge\; \tfrac{1}{2}(1 - C) $$ (eq:consistency-bounds-error)

**{{eq:consistency-bounds-error}} converts a free measurement into a guaranteed
statement about accuracy**, which is unusual and worth using.

### 5.4 Ensemble disagreement and what it measures

For ensemble members $f_1 \dots f_M$:

$$ U(x) = \text{sd}_i\big[f_i(x)\big] $$

This detects out-of-distribution inputs only when the members' extrapolations are
*independent*. Writing $f_i(x) = \bar{f}(x) + \epsilon_i(x)$ with correlation
$\rho$ between members:

$$ \mathbb{E}[U^2] = \sigma^2(1 - \rho) $$ (eq:extrapolation-diversity)

**{{eq:extrapolation-diversity}} vanishes as $\rho \to 1$.** Random features give
$\rho \approx 0$ outside the fitted region, hence the measured AUROC of 1.00.
**Identically-trained neural networks give $\rho$ close to 1**, and repeated
samples from one model give $\rho = 1$ in every respect except the sampling noise.

> **IMPORTANT:** So "the model was confident and wrong" is not a calibration
> failure to be fixed by better training. It is
> {{eq:extrapolation-diversity}} with $\rho \approx 1$ — **the disagreement you
> would need does not exist**, because there is only one function.

### 5.5 Composition

Let $\mathcal{P}$ be a set of primitive operations seen in training and
$\mathcal{C}(\mathcal{P})$ the compositions of them. A system generalises
compositionally if

$$ \text{accuracy on } \mathcal{C}(\mathcal{P}) \setminus \text{train} \;\approx\; \text{accuracy on train} $$

**Measured, the fitted system scores 0.4%.** And the executing system scores 0%
because $\mathcal{C}$ was never given to it — **neither system induces rules**,
which is the honest boundary of what this chapter can demonstrate.

## 6. Mathematical Foundation

### 6.1 Why the gap grows smoothly

The fitted model minimises error on $\mathcal{D}$, so its error off-distribution
grows with the distance the input moves in feature space. For a smooth fitted
function,

$$ |g(x') - y(x')| \;\lesssim\; L \cdot \|x' - \text{supp}(\mathcal{D})\| $$

with $L$ the fitted function's Lipschitz constant. **That is linear in the
distance**, which matches the measured relative error rising 0.001 → 0.064 → 0.529
→ 1.533 as the range multiplier grows.

**There is no threshold**, so a benchmark reports one point on a line and the
slope is the property you wanted.

### 6.2 What consistency buys, exactly

From {{eq:consistency-bounds-error}}: measured consistency of **1.1%** at double
the range implies an error rate of at least **49.5%**, without knowing a single
correct answer.

In distribution, consistency of **100.0%** implies nothing at all — a system can
be consistently wrong. **The bound is one-sided**, and that asymmetry is exactly
right for a production monitor: it can prove a problem and never prove its
absence.

### 6.3 Where the irrelevant clause does its damage

Adding a distractor changes the input by $\delta$ in the irrelevant coordinates.
A correct system has $\partial f/\partial x_{\text{irrelevant}} = 0$ by
construction. The fitted system has

$$ \frac{\partial g}{\partial x_{\text{irr}}} \ne 0 \quad \text{whenever the coordinate varied at all in training} $$

**and it varied, because nothing told the fitting procedure it was irrelevant.**
Measured relative error under one added clause: **5.542**, the largest in the
table — larger than a tenfold change in the numbers.

> **MATH NOTE:** That ordering is worth pausing on. A distractor that carries no
> information damaged the fitted system **more** than moving the inputs an order
> of magnitude. Irrelevance is not a small perturbation in feature space; it is a
> move into a region the training distribution never occupied at all, because in
> training that coordinate was always zero.

## 7. Internal Mechanics

```mermaid {#fig:two-systems caption="Two systems that agree on a benchmark and differ everywhere else. Held-out accuracy estimates a property of the pair (system, distribution), so it cannot separate them (eq:accuracy-does-not-separate). Three perturbations can, and two of the resulting criteria need no ground truth — which is what makes them runnable against production traffic (eq:criteria-without-labels)."}
flowchart TB
    P["a problem"] --> A["compute: map to a rule,<br/>evaluate the rule"]
    P --> B["fit: features of the<br/>surface form to answers"]
    A --> S{{"same score on<br/>held-out data"}}
    B --> S
    S -->|"eq:accuracy-does-not-separate"| X["a benchmark<br/>cannot separate them"]
    T1["change the numbers"] -->|"needs labels"| SEP["they separate"]
    T2["add an irrelevant clause"] -->|"needs labels"| SEP
    T3["paraphrase the problem"] -->|"NO labels needed"| SEP
    T4["ensemble disagreement"] -->|"NO labels; weak if<br/>eq:extrapolation-diversity<br/>has rho near 1"| SEP
```

### 7.1 The tests, ordered by what they cost

| Test | Needs | Detects | Deployable |
|---|---|---|---|
| held-out accuracy | labels | nothing about this | — |
| number perturbation | labels + generator | interpolation limits | offline |
| irrelevant clause | labels + generator | surface sensitivity | offline |
| **paraphrase consistency** | **one extra call** | surface sensitivity | **on live traffic** |
| ensemble spread | several diverse models | distribution shift, weakly | on live traffic |
| composition | a hand-built set | the thing people mean | offline, rarely |

**The bolded row is the one to add first**, because it costs a second call and
gives {{eq:consistency-bounds-error}}'s guarantee.

### 7.2 Building a perturbation set

The offline tests need a **generator**, not a dataset — you must be able to
produce the same problem in many surface forms:

1. **Template the problems** so semantics and surface vary independently.
2. **Vary the numbers** across a range wider than any plausible training
   distribution.
3. **Add distractors** that are grammatical, plausible, and unused by the answer.
4. **Report the variance across surface forms**, not the mean over one set.
5. **Keep the generator private**, because a published generator becomes training
   data ({{ch:ft-datasets}}).

### 7.3 What this chapter is not claiming

**It is not claiming real models are the fitted system.** It shows that accuracy
cannot distinguish the two and exhibits measurements that can.
{{cite:mirzadeh2024gsmsymbolic}} runs those measurements on real models, and the
answer is *partly, and it varies* — which is a more useful thing to know than
either extreme.

**And it is not claiming the executing system is a model of good reasoning.** It
fails composition too, for a different reason: it cannot induce a rule it was not
given. **The interesting system is neither of these**, and knowing how to
recognise it is the point.

### 7.4 Why this distinction is not philosophical

It would be easy to read this chapter as an argument about what the word
"reasoning" ought to mean, and that is not what it is for. **Every claim here is
a prediction about behaviour**, and the reason to care is that the two systems
fail differently in ways that determine how you deploy them.

**A system that computes has a domain and fails outside it, visibly.** Given a
problem whose rule it does not have, it produces nothing — which is a failure mode
you can build around with a fallback path and an alert.

**A system fitted to surface form has no domain boundary at all.** It produces an
answer for every input, and the answer's quality degrades smoothly with distance
from the training distribution ({{sec:6-mathematical-foundation}}'s Lipschitz
bound). There is no input for which it declines, and no signal in the output that
distinguishes an answer it computed from one it extrapolated.

**That is the operational content of the distinction, and it is why the
measurement matters more than the terminology.** Two systems with identical
benchmark scores need different production architectures: one needs a fallback
for out-of-domain inputs, the other needs a monitor for silent degradation, and
choosing the wrong one leaves the actual failure mode unhandled.

The rest of this part is largely about the second architecture — because the
systems in question are the fitted kind, and every technique that follows is a way
of adding a boundary that the model does not have on its own.

## 8. Implementation

```python {tier=A name=accuracy-does-not-separate}
"""Two systems that score the same on a benchmark and are not the same thing.

The word "reasoning" gets applied to any system that produces correct answers to
problems that look like they need reasoning. That is not a definition, and it does
not distinguish a system that COMPUTES the answer from one that has learned what
answers to problems of this shape usually look like.

This listing builds both, on the same task, with the same training data. One
executes the arithmetic the problem describes. The other is fitted to a surface
representation of the problem -- the words present, and the numbers -- exactly as
a learned model sees it.

Then it looks for a measurement that tells them apart, because held-out accuracy
does not (eq:accuracy-does-not-separate).
"""
import numpy as np

rng = np.random.default_rng(281)

# Each template is a word pattern plus an arithmetic rule over three numbers.
VOCAB = ["has", "gives", "buys", "each", "times", "more", "left", "total",
         "boxes", "apples", "friends", "shop", "then", "how", "many", "cost",
         "sold", "bought", "spare", "remaining", "altogether", "twice"]
V = len(VOCAB)

TEMPLATES = [
    (["has", "buys", "each", "total"],        lambda a, b, c: a + b * c),
    (["has", "gives", "left"],                lambda a, b, c: a - b - c),
    (["boxes", "each", "apples", "total"],    lambda a, b, c: a * b + c),
    (["shop", "sold", "remaining"],           lambda a, b, c: a - b * c),
    (["friends", "each", "altogether"],       lambda a, b, c: a * b - c),
    (["bought", "twice", "more", "total"],    lambda a, b, c: a + 2 * b + c),
]
DISTRACTOR = ["spare", "cost", "how", "many"]      # words that carry no rule


def encode(t_idx, nums, distract=False):
    """The surface form a learned model sees: which words are present, and the
    numbers. Nothing tells it which arithmetic rule applies."""
    words, _ = TEMPLATES[t_idx], None
    x = np.zeros(V + 4)
    for w in TEMPLATES[t_idx][0]:
        x[VOCAB.index(w)] = 1.0
    if distract:
        for w in rng.choice(DISTRACTOR, size=2, replace=False):
            x[VOCAB.index(w)] = 1.0
        x[V + 3] = float(rng.integers(2, 30))       # an irrelevant quantity
    x[V:V + 3] = nums
    return x


def make(n, t_pool, lo, hi, distract=False):
    X, Y = [], []
    for _ in range(n):
        t = int(rng.choice(t_pool))
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(t, nums, distract))
        Y.append(TEMPLATES[t][1](*nums))
    return np.array(X), np.array(Y, float)


SEEN = [0, 1, 2, 3]           # templates present in training
UNSEEN = [4, 5]               # held out entirely
LO, HI = 2, 20

Xtr, Ytr = make(20000, SEEN, LO, HI)

NF = 900
W = rng.normal(size=(Xtr.shape[1], NF)) * 0.6
B = rng.uniform(0, 2 * np.pi, NF)
MU, SD = Xtr.mean(0), Xtr.std(0) + 1e-9


def feat(X):
    return np.cos(((X - MU) / SD) @ W + B)


coef = np.linalg.solve(feat(Xtr).T @ feat(Xtr) + 1e-3 * np.eye(NF),
                       feat(Xtr).T @ Ytr)


def learned(X):
    return feat(X) @ coef


def reasoner(t_idx, nums):
    """Executes the rule the problem describes. It cannot be wrong about
    arithmetic, and it cannot answer a template it does not know."""
    return TEMPLATES[t_idx][1](*nums)


def evaluate(t_pool, lo, hi, distract=False, n=4000, tol=0.5):
    X, Y = [], []
    ok_r = 0
    for _ in range(n):
        t = int(rng.choice(t_pool))
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(t, nums, distract))
        Y.append(TEMPLATES[t][1](*nums))
        ok_r += 1 if t in SEEN + UNSEEN else 0
    X, Y = np.array(X), np.array(Y, float)
    pred = learned(X)
    acc_l = float(np.mean(np.abs(pred - Y) <= tol))
    rel_l = float(np.mean(np.abs(pred - Y) / np.maximum(np.abs(Y), 1.0)))
    return acc_l, ok_r / n, rel_l


print("Two systems, same task, same training data. Exact-match accuracy.")
print("The 'reasoner' executes the arithmetic; the 'learned' model is fitted to")
print("the surface form -- words present, and the numbers.")
print()
print(f"{'test condition':>38}{'learned':>10}{'learned':>11}{'reasoner':>11}")
print(f"{'':>38}{'exact':>10}{'rel error':>11}{'exact':>11}")
print("-" * 70)

CASES = [
    ("held-out, same templates and range", SEEN, LO, HI, False),
    ("numbers 2x larger", SEEN, HI, 2 * HI, False),
    ("numbers 10x larger", SEEN, 10 * LO, 10 * HI, False),
    ("one irrelevant clause added", SEEN, LO, HI, True),
    ("templates never seen in training", UNSEEN, LO, HI, False),
]
res = {}
for name, pool, lo, hi, dis in CASES:
    al, ar, rl = evaluate(pool, lo, hi, dis)
    res[name] = (al, ar, rl)
    print(f"{name:>38}{al:>10.1%}{rl:>11.3f}{ar:>11.1%}")

print()
print()
print("How large does the perturbation have to be? Sweeping the number range.")
print()
print(f"{'number range':>16}{'exact':>11}{'rel error':>12}{'gap from':>11}")
print(f"{'':>16}{'':>11}{'':>12}{'baseline':>11}")
print("-" * 51)
base = res["held-out, same templates and range"][0]
sweep, sweep_rel = {}, {}
for mult in (1.0, 1.5, 2.0, 3.0, 5.0):
    lo, hi = int(LO * mult), int(HI * mult)
    a, _, r = evaluate(SEEN, lo, hi)
    sweep[mult], sweep_rel[mult] = a, r
    print(f"{f'{lo}-{hi}':>16}{a:>11.1%}{r:>12.3f}{base - a:>10.1%}")

h = res["held-out, same templates and range"]
n2 = res["numbers 2x larger"]
irr = res["one irrelevant clause added"]
uns = res["templates never seen in training"]
print(f"""
The first row is the one that would appear in a paper. On held-out problems drawn
from the same distribution as training, the learned model scores {h[0]:.1%} and
the reasoner scores {h[1]:.1%}. **A benchmark cannot tell them apart**
(eq:accuracy-does-not-separate), and neither can any amount of held-out data
drawn the same way.

Every other row separates them completely, and none of the perturbations changes
what the problem MEANS.

Doubling the size of the numbers takes the learned model to {n2[0]:.1%} while the
reasoner is unaffected at {n2[1]:.1%}. Nothing about the arithmetic changed.

But read the relative-error column beside it, because exact match overstates the
collapse. In distribution the learned model's mean relative error is
{h[2]:.3f}; at double the range it is {n2[2]:.3f}. It has not started producing
nonsense -- it is producing answers of roughly the right SIZE and the wrong VALUE,
which is a much more familiar failure than a random one and much harder to notice.

That distinction matters for what this demonstrates. The learned model has
captured the right shape of function and cannot evaluate it outside the region it
was fitted on. It interpolated, which is what fitting does, and outside the
training range there is nothing to interpolate between.

Adding one irrelevant clause -- two extra words and a number the rule never uses
-- takes it to {irr[0]:.1%}. The words are in the vocabulary and the number is in
the feature vector, so they move the prediction. A system that executed the rule
would ignore them because the rule does not mention them; a system fitted to
surface form has no way to know which parts of the surface matter.

That is precisely cite:mirzadeh2024gsmsymbolic's experiment, and it reports the
same shape of result on real models: performance declining when only the numbers
change, and drops of up to 65% from a single irrelevant sentence. This listing
makes the mechanism visible rather than inferred, because here we know exactly
what each system is doing.

The last row is the one that matters most and is easiest to miss. On templates
never seen in training, the learned model scores {uns[0]:.1%} and the reasoner
scores {uns[1]:.1%} -- but the reasoner's score is only that because it was GIVEN
the rule. It cannot induce a rule it has not been told, and a system that could
would be doing something neither of these does.

Which is the honest statement of what this listing shows and does not show. It
demonstrates that accuracy on held-out data cannot distinguish computing from
fitting, and it exhibits three perturbations that can. It does not show that real
models are the learned system -- that is cite:mirzadeh2024gsmsymbolic's job, and
their answer is "partly, and it varies."

The sweep makes the last point quantitative. The degradation is not a cliff at
some boundary; it is smooth in how far the test distribution moves. At
{1.5}x the range the learned model has already lost {base - sweep[1.5]:.1%}, and
at {5.0}x, {base - sweep[5.0]:.1%}. There is no threshold at which the model
stops working and starts working -- which means **any single benchmark score is a
point estimate on a curve whose slope nobody reports.**

So the practical content is a measurement protocol rather than a definition.
Whatever you mean by reasoning, a system that has it should be INVARIANT to
changes that do not change the answer. Generate your evaluation from templates so
you can vary the surface independently of the semantics, report the variance and
not just the mean, and treat a large gap between in-distribution and perturbed
accuracy as the finding rather than as noise.""")
```

The first listing shows the perturbations that separate the two systems, all of
which need ground truth. The second looks for criteria that do not.

```python {tier=A name=criteria-without-labels}
"""Three candidate tests for reasoning, and which of them you can run.

The previous listing showed that held-out accuracy cannot distinguish a system
that computes an answer from one fitted to the surface form of the problem, and
that perturbing the problem can. That is useful and it requires ground truth: you
have to know the right answer to the perturbed problem to score it.

This listing looks for criteria that do NOT require ground truth, because those
are the ones you can run on a deployed system against real traffic
(eq:criteria-without-labels).

Three candidates: does the system give the same answer to the same problem phrased
two ways; does its confidence rise when it is out of its depth; and does it
compose operations it has seen into combinations it has not.
"""
import numpy as np

rng = np.random.default_rng(283)

VOCAB = ["has", "gains", "buys", "each", "acquires", "obtains", "total", "sum",
         "boxes", "crates", "apples", "pears", "shop", "store", "sold", "left",
         "friends", "colleagues", "altogether", "combined", "spare", "extra"]
V = len(VOCAB)

# Each RULE has two phrasings. A system that computes gives the same answer to
# both; a system fitted to surface form has no reason to.
RULES = [
    ("a + b*c",   lambda a, b, c: a + b * c,
     [["has", "buys", "each", "total"], ["gains", "acquires", "each", "sum"]]),
    ("a*b + c",   lambda a, b, c: a * b + c,
     [["boxes", "each", "apples", "total"],
      ["crates", "each", "pears", "sum"]]),
    ("a - b*c",   lambda a, b, c: a - b * c,
     [["shop", "sold", "left"], ["store", "sold", "left", "spare"]]),
    ("a*b - c",   lambda a, b, c: a * b - c,
     [["friends", "each", "altogether"],
      ["colleagues", "each", "combined"]]),
]
# Held out entirely: a composition of operations that appear in the seen rules.
NOVEL = ("a*b + a", lambda a, b, c: a * b + a,
         [["boxes", "each", "gains", "total"], ["crates", "each", "obtains",
                                                "sum"]])


def encode(words, nums):
    x = np.zeros(V + 3)
    for w in words:
        x[VOCAB.index(w)] = 1.0
    x[V:] = nums
    return x


LO, HI = 2, 20


def sample(rules, n, phrase=None, lo=LO, hi=HI):
    X, Y, R = [], [], []
    for _ in range(n):
        r = rules[int(rng.integers(len(rules)))]
        p = r[2][phrase if phrase is not None else int(rng.integers(2))]
        nums = rng.integers(lo, hi, size=3).astype(float)
        X.append(encode(p, nums)); Y.append(r[1](*nums)); R.append(nums)
    return np.array(X), np.array(Y, float), np.array(R)


Xtr, Ytr, _ = sample(RULES, 24000)

NF, ENS = 700, 8
MU, SD = Xtr.mean(0), Xtr.std(0) + 1e-9
models = []
for e in range(ENS):
    W = rng.normal(size=(Xtr.shape[1], NF)) * 0.6
    B = rng.uniform(0, 2 * np.pi, NF)
    F = np.cos(((Xtr - MU) / SD) @ W + B)
    c = np.linalg.solve(F.T @ F + 1e-3 * np.eye(NF), F.T @ Ytr)
    models.append((W, B, c))


def predict(X):
    """Ensemble mean and spread. The spread is the model's own uncertainty,
    computed without any labels."""
    P = np.stack([np.cos(((X - MU) / SD) @ W + B) @ c for W, B, c in models])
    return P.mean(0), P.std(0)


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float((ranks[pos].mean() - (pos.sum() - 1) / 2) / neg.sum())


print("TEST 1 -- consistency under reformulation. The same problem, phrased two")
print("ways. Needs no ground truth: you only need two phrasings.")
print()
print(f"{'condition':>30}{'agree within 0.5':>19}{'mean disagreement':>20}")
print("-" * 69)

consist = {}
for name, lo, hi in (("in-distribution numbers", LO, HI),
                     ("numbers 2x larger", HI, 2 * HI),
                     ("numbers 5x larger", 5 * LO, 5 * HI)):
    agree, gaps = [], []
    for _ in range(3000):
        r = RULES[int(rng.integers(len(RULES)))]
        nums = rng.integers(lo, hi, size=3).astype(float)
        p0, _ = predict(encode(r[2][0], nums)[None])
        p1, _ = predict(encode(r[2][1], nums)[None])
        agree.append(abs(p0[0] - p1[0]) <= 0.5)
        gaps.append(abs(p0[0] - p1[0]))
    consist[name] = (float(np.mean(agree)), float(np.mean(gaps)))
    print(f"{name:>30}{consist[name][0]:>19.1%}{consist[name][1]:>20.2f}")

print()
print("  (the executing system agrees 100% by construction: both phrasings")
print("   invoke the same rule, so it cannot disagree with itself)")

print()
print()
print("TEST 2 -- does confidence rise when the system is out of its depth?")
print("Ensemble spread as an uncertainty signal. Also needs no ground truth.")
print()
print(f"{'condition':>30}{'mean spread':>14}{'vs baseline':>13}"
      f"{'AUROC vs':>11}")
print(f"{'':>30}{'':>14}{'':>13}{'baseline':>11}")
print("-" * 68)

Xb, Yb, _ = sample(RULES, 3000)
_, sb = predict(Xb)
base_spread = float(sb.mean())
unc = {}
for name, rules, lo, hi in (("in-distribution", RULES, LO, HI),
                            ("numbers 2x larger", RULES, HI, 2 * HI),
                            ("numbers 5x larger", RULES, 5 * LO, 5 * HI),
                            ("novel composition", [NOVEL], LO, HI)):
    X, Y, _ = sample(rules, 3000, lo=lo, hi=hi)
    _, sp = predict(X)
    a = auroc(np.concatenate([sb, sp]),
              np.concatenate([np.zeros(len(sb)), np.ones(len(sp))]))
    unc[name] = (float(sp.mean()), float(sp.mean()) / base_spread, a)
    print(f"{name:>30}{unc[name][0]:>14.2f}{unc[name][1]:>12.1f}x"
          f"{a:>11.2f}")

print()
print()
print("TEST 3 -- composition. A rule built from operations the system has seen,")
print("in a combination it has not. Needs ground truth, unlike the first two.")
print()
print(f"{'system':>30}{'exact match':>14}{'rel error':>12}")
print("-" * 56)
Xn, Yn, _ = sample([NOVEL], 3000)
pn, _ = predict(Xn)
comp_l = (float(np.mean(np.abs(pn - Yn) <= 0.5)),
          float(np.mean(np.abs(pn - Yn) / np.maximum(np.abs(Yn), 1.0))))
print(f"{'learned':>30}{comp_l[0]:>14.1%}{comp_l[1]:>12.3f}")
print(f"{'executing, rule not supplied':>30}{0.0:>14.1%}{'--':>12}")

c_in = consist["in-distribution numbers"]
c_2x = consist["numbers 2x larger"]
print(f"""
Test 1 is the one worth having, because it needs nothing you would not already
have: the same question asked twice, differently.

In distribution the two phrasings agree {c_in[0]:.1%} of the time, mean
disagreement {c_in[1]:.2f}. At double the number range they agree
{c_2x[0]:.1%}, mean disagreement {c_2x[1]:.2f}.

The executing system agrees with itself 100% of the time at every range, and not
because it is better -- because a system that maps the problem to a rule and then
evaluates the rule has nothing left that could depend on the phrasing. **Self-
consistency across paraphrase is not a proxy for correctness; it is a proxy for
whether the surface form is entering the computation** (eq:criteria-without-labels).

That makes it deployable in a way accuracy is not. You do not need to know the
right answer, or to have a benchmark, or to construct perturbations that preserve
semantics -- you need one paraphrase, and disagreement is evidence regardless of
which answer was right.

Test 2 comes out far stronger than expected, and the reason it does is the
interesting part.

Ensemble spread rises by {unc['numbers 2x larger'][1]:.0f}x at double the number
range and {unc['novel composition'][1]:.0f}x on a novel composition, separating
out-of-distribution from in-distribution examples with an AUROC of
{unc['numbers 2x larger'][2]:.2f} -- perfect. On this system, ensemble
disagreement is a flawless detector of the conditions that make it wrong.

That result should be read with suspicion rather than enthusiasm, because it
depends on a property of the model class rather than on anything general. Each
ensemble member is a random-feature model, and random features EXTRAPOLATE
DIFFERENTLY: outside the fitted region there is nothing constraining them to
agree, so they fly apart. The disagreement is enormous because the extrapolations
are independent.

Real models do not have that property. Members of an ensemble trained the same way
on the same data, with the same architecture and the same inductive biases,
extrapolate SIMILARLY -- they are wrong in correlated ways, so they agree while
being wrong. **Ensemble disagreement measures diversity of extrapolation, not
distance from the training distribution**, and those two coincide only when the
members are genuinely diverse.

Which is the general form of the caveat, and it applies well beyond this listing.
An uncertainty estimate built from agreement is only as good as the independence
of the things agreeing. Sampling one model several times at a nonzero temperature
gives the weakest version of this, because every sample shares every parameter --
and it is also the version most commonly used.

Test 3 is the criterion that would be most convincing and is the hardest to
apply. On a rule composed of operations the system has seen -- multiply, then add
a quantity already present -- the fitted model scores {comp_l[0]:.1%} with a
relative error of {comp_l[1]:.3f}.

The executing system scores 0%, and this is the honest part of the listing: it
scores 0% because nobody gave it the rule. It cannot induce a rule from examples,
which the fitted model at least attempts. **Neither system is doing the thing the
word "reasoning" is usually meant to name**, and constructing a system that
composes known operations into unseen combinations is the actual open problem
rather than a measurement problem.

So the practical output is a hierarchy of tests ordered by what they cost.

Paraphrase consistency costs one extra call and no labels, and it directly
measures whether the surface form is entering the answer. Run it on production
traffic.

Ensemble spread costs several models and worked perfectly here for a reason that
will not transfer. Run it where the members are genuinely diverse -- different
architectures, different data, different seeds all the way down -- and treat
agreement among near-identical models as almost no evidence at all.

Compositional generalisation costs a hand-built evaluation set and is the only one
that measures the thing people actually mean. Run it when you are choosing between
systems, and expect the answer to be uncomfortable.

None of the three is what a benchmark reports, and a benchmark score with none of
them beside it is a measurement of performance on one distribution, which was the
previous listing's finding stated a second way.""")
```

## 9. Practical Example

**A benchmark cannot separate them.** On held-out problems drawn the same way,
**100.0% and 100.0%** ({{eq:accuracy-does-not-separate}}) — and more held-out data
estimates the same quantity more precisely rather than answering a different
question.

**Three perturbations separate them completely**, none changing the answer:
numbers doubled **0.1% / 100%**, one irrelevant clause **0.2% / 100%**, unseen
template **0.5% / 100%**.

**And the degradation is graceful, which is why it is missed.** Relative error
**0.001 → 0.812 → 1.368**: answers of roughly the right size and the wrong value.
The sweep gives **100.0% → 44.1% → 15.3% → 0.5%** with no threshold —
**a benchmark score is a point on a line whose slope is the property you wanted.**

> **IMPORTANT:** The largest damage came from **one irrelevant clause** (relative
> error **5.542**), exceeding a tenfold change in the numbers. **Irrelevance is
> not a small perturbation** — it moves the input into a region the training
> distribution never occupied, because that coordinate was always zero.

**Paraphrase consistency needs no labels and works.** Agreement **100.0% → 1.1%**
as the distribution shifts, mean disagreement **37.15**. The executing system
agrees with itself **100%** at every range, because **a system that maps to a rule
and evaluates it has nothing left that could depend on phrasing.**

**And {{eq:consistency-bounds-error}} converts it into a guarantee**: 1.1%
consistency implies **at least 49.5% error**, without knowing a single correct
answer. **One-sided** — it can prove a problem and never prove its absence, which
is exactly right for a monitor.

**Ensemble spread separated perfectly — AUROC 1.00 — for a reason that will not
transfer.** Spread rose **2,354×** at double the range, because random-feature
models extrapolate *independently*. {{eq:extrapolation-diversity}}: the signal
vanishes as member correlation approaches 1, **and identically-trained networks
are correlated.** Repeated samples from one model are the degenerate case.

**Composition defeated both.** The fitted model scored **0.4%**; the executing
system scored **0%**, because nobody gave it the rule. **Neither is doing what the
word is usually meant to name.**

## 10. Production Considerations

**Add a paraphrase consistency check** to any deployed reasoning system. One extra
call, no labels, and {{eq:consistency-bounds-error}}'s bound.

**Report perturbation variance, not just the mean.** The spread is the finding.

**Keep your perturbation generator private.** A published one becomes training
data.

**Do not trust confidence from a single model**, or from repeated samples of one
({{eq:extrapolation-diversity}}).

**Test with distractors specifically**, since they did the most damage here.

**Treat a large in-distribution/perturbed gap as a result**, not as noise to be
averaged away.

**State the distribution any accuracy number was measured on**, because
{{eq:accuracy-does-not-separate}} makes it half the claim.

## 11. Common Mistakes

**Treating held-out accuracy as a property of the system.**

**Adding more held-out data** to resolve a question it cannot answer.

**Reading a single benchmark score** as a capability.

**Trusting a model's stated confidence**, or the agreement of samples from one
model.

**Testing only number perturbations** and not distractors, which are worse.

**Publishing a perturbation generator**, thereby destroying it.

**Concluding from a perturbation result that the model "cannot reason"** — the
measurement supports a narrower claim, and stating it precisely is the chapter's
point.

## 12. Failure Modes

**Excellent benchmark score, poor production behaviour.** Cause:
{{eq:accuracy-does-not-separate}} — the benchmark and production are different
distributions.

**Performance drops when a form field is added.** Cause: distractor sensitivity.

**A system that was fine last quarter degrades with no change.** Cause: the input
distribution moved; {{eq:reliability-gap}} was always there and is now being
sampled.

**Confidence scores fail to flag errors.** Cause:
{{eq:extrapolation-diversity}} with $\rho \approx 1$.

**Paraphrase check fires constantly.** Cause: correct — either the system is
surface-sensitive, or the paraphrases are not semantically equivalent, and both
are worth knowing.

**Benchmark score improves and users notice nothing.** Cause: the improvement was
on the surface forms in the benchmark.

## 13. Alternatives

| Alternative | Cost | Detects |
|---|---|---|
| more held-out data | cheap | nothing new |
| template-generated perturbations | a generator | interpolation limits |
| paraphrase consistency | one extra call | surface sensitivity, live |
| diverse ensembles | several models | distribution shift, if diverse |
| a symbolic checker ({{ch:rsn-tool-assisted}}) | integration work | actual correctness |
| human review of a sample | expensive | everything, slowly |

**The symbolic-checker row is where this part ends up**, and it is worth
signposting now: **the only test in this table that measures correctness rather
than a proxy for it is the one that does not use the model at all.**

## 14. Evaluation

**Report the generator, not just the dataset**, so a reader can regenerate.

**Report accuracy under each perturbation**, and the variance across surface
forms.

**Report paraphrase consistency**, which costs almost nothing.

**State whether any uncertainty estimate came from diverse models** or from one.

**Never report a single reasoning score without the distribution it was measured
on.**

## 15. Advanced Concepts

**Consistency as a one-sided proof.** {{maturity:EMERGING}}
{{eq:consistency-bounds-error}} is elementary and almost never used. It turns a
free measurement into a lower bound on error rate, and it works on any system
including ones you cannot inspect.

**Correlated extrapolation is the deep problem with confidence.**
{{maturity:MATURE}} {{eq:extrapolation-diversity}} explains why ensembling helps
less than the theory suggests and why sampling one model helps not at all.
**Genuine diversity — different data, architectures and objectives — is expensive
and is what the estimate actually requires.**

**Perturbation robustness as a training objective.**
{{maturity:RESEARCH FRONTIER}} If invariance is what distinguishes the systems,
it can be trained for directly — augmenting with paraphrases and distractors and
penalising disagreement. **That optimises the criterion rather than the score**,
and how far it gets is an open question.

**The distractor result generalises.** {{maturity:EMERGING}} An irrelevant input
does more damage than a large change in a relevant one, because it moves the input
outside the training support entirely. **Real deployments add irrelevant context
constantly** — extra form fields, retrieved documents, conversation history — and
this is a reason to expect that to hurt.

**Composition is the actual frontier.** {{maturity:RESEARCH FRONTIER}} Neither
system here composes. {{cite:sprague2024tocot}} and
{{cite:mirzadeh2024gsmsymbolic}} both point at it from different directions, and
nothing in this part solves it — what the rest of the part does is buy reliability
in domains where an external check exists.

## 16. Connection to Previous Chapters

{{ch:ft-datasets}}'s {{eq:metric-inherits-bias}} is this chapter's
{{eq:accuracy-does-not-separate}} in a different setting: there the evaluation
inherited the training *selection*, here it inherits the training
*distribution*, and in both cases more evaluation data does not help.
{{ch:ft-synthetic}}'s {{eq:self-eval-agreement}} is
{{eq:extrapolation-diversity}} with $\rho = 1$ — a model grading itself is an
ensemble of one.
{{ch:fm-what-they-are}} established what next-token prediction optimises, which is
why a plausible chain and a correct derivation are not the same object.
Forward: {{ch:rsn-cot}} asks what intermediate tokens do;
{{ch:rsn-tool-assisted}} is where an external check replaces all of these
proxies; and {{ch:rsn-benchmarks}} turns {{eq:reliability-gap}} into an
evaluation protocol.

## 17. Exercises

1. Explain {{eq:accuracy-does-not-separate}} and say why collecting ten times more
   held-out data does not help.
2. Derive {{eq:consistency-bounds-error}} and compute the implied error bound for
   a measured consistency of 80%.
3. From {{eq:extrapolation-diversity}}, compute the ensemble spread when member
   correlation is 0.0, 0.9 and 0.99. What does that say about sampling one model?
4. In `accuracy-does-not-separate`, add a fourth perturbation: reorder the clauses
   without changing the rule. Does it separate the systems?
5. In the same listing, train on a wider number range and re-run the sweep. Does
   the gap move or disappear?
6. In `criteria-without-labels`, make the ensemble members share features and
   differ only in the ridge penalty. What happens to the AUROC, and does
   {{eq:extrapolation-diversity}} predict it?
7. Design a paraphrase generator for a task you work on. What makes two phrasings
   genuinely equivalent, and how would you check?
8. For a deployed system: measure paraphrase consistency on a day of real traffic
   and report the implied error bound.

## 18. Interview Questions

1. Why can a benchmark not distinguish reasoning from pattern matching?
2. Name three perturbations that can, and what each tests.
3. Which of them can you run without ground truth?
4. What does paraphrase consistency actually measure?
5. Why is it a one-sided test?
6. Why is ensemble disagreement a weaker signal than it looks?
7. Why is sampling one model repeatedly not an ensemble?
8. Why did an irrelevant clause do more damage than a tenfold change in the
   numbers?
9. Your benchmark score improved and users noticed nothing. What happened?
10. What would convince you a system reasons?

## 19. Research Questions

1. {{eq:consistency-bounds-error}} gives a free lower bound on error. How tight is
   it on real systems, and can it be sharpened with more paraphrases?
2. {{eq:extrapolation-diversity}} says ensembles need genuine diversity. What is
   the cheapest source of it that still gives a usable uncertainty estimate?
3. Distractors did more damage than magnitude shifts. Does that ordering hold on
   real models, and does it explain the degradation from long retrieved contexts?
4. Can invariance be trained directly, and does a system trained for it transfer
   the invariance to perturbations it did not see?
5. Neither system composes. What is the minimal architectural addition that would,
   and what does it cost on the tasks that currently work?

## 20. Chapter Summary

**Held-out accuracy cannot distinguish computing an answer from fitting one.** Two
systems, same task, same data: **100.0% and 100.0%**
({{eq:accuracy-does-not-separate}}), and more held-out data estimates the same
quantity rather than answering a different question.

**Three perturbations separate them completely and none changes the answer** —
numbers doubled (**0.1%**), one irrelevant clause (**0.2%**), an unseen template
(**0.5%**), against **100%** throughout for the executing system. That is
{{cite:mirzadeh2024gsmsymbolic}}'s design, and their real-model results have the
same shape.

**The degradation is graceful, which is why it goes unnoticed**: relative error
**0.001 → 0.812**, answers of the right size and the wrong value. **And there is no
threshold** — the sweep runs 100.0% → 44.1% → 15.3% → 0.5%, so **a benchmark score
is a point on a line whose slope is what you wanted to know.**

**The largest damage came from the irrelevant clause**, exceeding a tenfold change
in the numbers, because irrelevance moves the input outside the training support
entirely rather than to its edge.

**Two criteria need no ground truth.** Paraphrase consistency fell **100.0% →
1.1%** while the executing system stayed at 100% by construction, and
{{eq:consistency-bounds-error}} turns that into a guarantee: **1.1% consistency
implies at least 49.5% error, with no correct answers required.** It is one-sided
— it proves problems and never their absence — which is exactly what a monitor
should be.

**Ensemble disagreement separated perfectly here and will not elsewhere.** AUROC
**1.00**, because random features extrapolate independently;
{{eq:extrapolation-diversity}} shows the signal vanishing as members correlate,
**and identically-trained models are correlated.** A model grading itself is an
ensemble of one.

**And composition defeated both systems** — 0.4% and 0%. **Neither is doing what
the word is usually meant to name**, which is the honest boundary of what this
chapter demonstrates, and the reason the rest of the part is about buying
reliability from **external checks** rather than from better generation.

## 21. Further Reading

{{cite:mirzadeh2024gsmsymbolic}} for this chapter's experiment on real models,
and for the number worth carrying: up to 65% from a single irrelevant sentence.
{{cite:sprague2024tocot}} for the meta-analysis that bounds where reasoning
techniques help at all, which {{ch:rsn-cot}} develops.
{{cite:turpin2023faithfulness}} for the related result that a chain of thought can
justify an answer it did not cause — the same gap between appearance and mechanism,
one level down.
{{cite:cobbe2021gsm8k}} for the benchmark that defined the field's measure of
reasoning, read with {{eq:accuracy-does-not-separate}} in mind.
{{cite:wei2022cot}} as the paper that made this a live question, and which
{{ch:rsn-cot}} takes apart.
