---
id: llm-hallucination
number: 96
part: X
tier: full
status: draft
requires: [llm-next-token, llm-decoding, llm-function-calling, fm-pretraining,
           fm-rlhf, nlp-extraction, ml-metrics]
provides: [hallucination-taxonomy, intrinsic-hallucination, extrinsic-hallucination,
           groundedness, faithfulness-metric, abstention, citation-checking,
           objective-truthfulness-gap, confabulation]
citations: [ji2023survey, kadavath2022, ouyang2022, holtzman2020, liu2023lost,
            brown2020, lee2022dedup, guo2017calibration]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State why hallucination follows from the training objective rather than from
   a defect.
2. Distinguish intrinsic from extrinsic hallucination and say why the
   distinction decides the mitigation.
3. Explain why alignment increases confident fabrication, from
   {{eq:rlhf-optimal-policy}}.
4. Implement a groundedness check and state precisely what it does and does not
   catch.
5. Build an abstention policy and compute the coverage it costs.
6. Rank mitigations by which failure class each addresses.
7. Evaluate a hallucination claim, including what the measurement cannot see.

## 2. Why This Matters

**This is the failure mode that decides whether an LLM system can be deployed
at all**, and it is the one most often addressed with the wrong tool. Teams
measure one kind of hallucination while mitigating another, which is why
mitigation efforts so frequently disappoint.

**It is not a bug that was introduced.** {{eq:clm-loss}} contains no term for
truthfulness — the objective rewards plausible continuation, and a model that
said "I don't know" where its corpus said something confident would score
*worse*. Hallucination is the objective working as specified, and treating it as
a defect to be patched misunderstands where it comes from.

**Alignment makes it worse in a predictable direction.**
{{ch:llm-next-token}} showed {{eq:rlhf-optimal-policy}} degrading calibration
because raters prefer confident answers. The same mechanism produces confident
fabrication: a hedged correct answer loses to a decisive wrong one in preference
data, and the reward model learns it.

**And the useful distinction is old and under-used.**
{{cite:ji2023survey}}'s intrinsic/extrinsic split is operational rather than
taxonomic: the two have different detection methods, different fixes, and
different costs, and conflating them is the single most common error in this
area.

## 3. Prerequisites

{{ch:llm-next-token}} for calibration and {{eq:two-uncertainties}} — the softmax
cannot express epistemic uncertainty, which is most of the problem.
{{ch:fm-pretraining}} for the objective. {{ch:fm-rlhf}} for
{{eq:rlhf-optimal-policy}} and the length/confidence biases.
{{ch:llm-decoding}} for temperature's role. {{ch:nlp-extraction}} for span
grounding, which returns here as the strongest cheap defence.
{{ch:ml-metrics}} for precision/recall, which abstention trades between.

## 4. Intuitive Explanation

The model states, confidently and in fluent prose, that a paper you have never
heard of was published in *Nature* in 2019 by three authors whose names sound
right. None of it is true.

**Nothing went wrong.** The model was trained to continue text plausibly, and
that is a plausible continuation. Nowhere in {{eq:clm-loss}} is there a term
distinguishing "true" from "typical" — the loss is computed against what the
corpus said, and the corpus is full of confident claims.

> NOTE: "Hallucination" is a poor name, and "confabulation" is closer: the model
> is not perceiving something absent, it is *generating a plausible completion
> in the absence of grounds*, which is exactly what it was optimised to do. The
> name matters because "hallucination" suggests a malfunction with a fix, and
> confabulation suggests a property with a mitigation. The second framing
> produces better engineering.

**Two kinds, and the difference decides everything.**

**Intrinsic** hallucination contradicts a source you supplied. The document says
the meeting is on Tuesday; the summary says Thursday. **This is checkable**
against the source, cheaply and mechanically.

**Extrinsic** hallucination asserts something the source does not address. The
document says nothing about attendance; the summary says forty people came.
**This is not checkable against the source** — you need external knowledge, or
you must decide that anything unsupported is disallowed.

**Why the distinction matters practically:** retrieval fixes extrinsic
hallucination by supplying grounds; it does nothing for intrinsic, where grounds
were already present and were contradicted. Citation checking fixes intrinsic;
it does nothing for extrinsic claims that cite nothing. **A team adding
retrieval to fix a contradiction problem has bought the wrong mitigation**, and
this happens constantly.

**And a third thing that is not hallucination at all.** A model that is simply
*wrong* — it computed something incorrectly, misread a number — is making an
error, not confabulating. The distinction matters because errors respond to
better models and confabulation responds to better grounding.

**The mental model:** the model produces the most plausible continuation given
its context, and when the context does not determine the answer it produces a
plausible one anyway. Where it breaks down: "plausible" is relative to the
training distribution, so hallucination concentrates exactly where the corpus was
confident and the world is uncertain — obscure entities, recent events, specific
numbers.

## 5. Formal Explanation

### 5.1 Why the objective permits it

Training minimises {{eq:clm-loss}}, whose optimum is the true conditional
distribution of the *corpus*:

$$
q^*(\cdot\given c) = p_{\text{corpus}}(\cdot\given c)
$$ (eq:corpus-optimum)

For a factual question, $p_{\text{corpus}}$ places mass on whatever the corpus
said. If the corpus contained confident statements — as text overwhelmingly does
— the model reproduces confidence. **There is no term anywhere rewarding a
correct expression of ignorance**, and by {{eq:cross-entropy-decomposition}}'s
properness, hedging when the corpus did not hedge *increases* the loss.

$\square$

### 5.2 The taxonomy

For a generated statement $s$ and a provided source $D$:

$$
\begin{aligned}
&\text{intrinsic:} && s \text{ contradicts } D\\
&\text{extrinsic:} && s \text{ is unsupported by } D \text{ and does not contradict it}\\
&\text{grounded:} && s \text{ is entailed by } D
\end{aligned}
$$ (eq:hallucination-taxonomy)

{#tbl:hallucination-types caption="The two hallucination types and their mitigations. The rows require different machinery, and applying one row's fix to the other row's problem is the most common error in this area."}

| | Intrinsic | Extrinsic |
|---|---|---|
| Relation to source | contradicts | unsupported |
| Detectable from source alone | **yes** | **no** |
| Fixed by retrieval | no | yes |
| Fixed by citation checking | yes | partly |
| Fixed by abstention | partly | yes |
| Typical cause | attention/position failure | no grounds available |

### 5.3 Groundedness

For a claim $s$ decomposed into atomic assertions $a_1,\dots,a_m$:

$$
\text{groundedness}(s, D)
 = \frac{1}{m}\sum_{i=1}^{m}\Ind\big[a_i \text{ is entailed by } D\big]
$$ (eq:groundedness)

**Decomposition into atomic claims is the hard part**, and it is where
groundedness metrics differ. A sentence containing four assertions of which
three are supported is 0.75 grounded and, for most purposes, wrong.

### 5.4 Why alignment increases confident fabrication

From {{eq:rlhf-optimal-policy}}, the aligned policy reweights by exponentiated
reward. Let $r$ be the learned reward and consider two responses to a question
the model cannot answer:

$$
y_{\text{hedge}} = \text{"I'm not certain, but possibly X"},
\qquad
y_{\text{confident}} = \text{"X."}
$$

Human raters, comparing these without knowing the truth, prefer the second —
it is more useful *if correct*, and its incorrectness is not visible at
comparison time. So $r(y_{\text{confident}}) > r(y_{\text{hedge}})$ and

$$
\frac{\pi^*(y_{\text{confident}})}{\pi^*(y_{\text{hedge}})}
 = \frac{\pi_{\text{ref}}(y_{\text{confident}})}{\pi_{\text{ref}}(y_{\text{hedge}})}
 \exp\!\Big(\frac{r(y_{\text{confident}}) - r(y_{\text{hedge}})}{\beta}\Big)
 > \frac{\pi_{\text{ref}}(y_{\text{confident}})}{\pi_{\text{ref}}(y_{\text{hedge}})}
$$ (eq:alignment-confidence-shift)

$\square$

**Alignment systematically shifts mass from hedged to confident responses**, and
it does so because the preference data cannot distinguish confident-and-right
from confident-and-wrong. This is not a flaw in any particular RLHF
implementation; it follows from comparing responses without ground truth.

### 5.5 Abstention

An abstention policy answers when a confidence signal exceeds a threshold:

$$
\text{answer if } \kappa(x) > \tau,
\qquad
\text{coverage} = \Prob[\kappa > \tau],
\qquad
\text{risk} = \Prob[\text{wrong} \given \kappa > \tau]
$$ (eq:abstention)

**Raising $\tau$ lowers risk and lowers coverage**, and the achievable
(coverage, risk) pairs form a curve. The useful question is not "what threshold"
but **"what risk can the downstream process tolerate"**, which is a product
decision that then determines $\tau$.

> IMPORTANT: {{ch:llm-next-token}} showed confidence survives alignment as a
> *rank* and not as a *probability*. So the abstention curve can be traced
> empirically and the threshold cannot be computed from a target error rate
> without calibration. That is the practical consequence of
> {{eq:rlhf-optimal-policy}} for this chapter.

## 6. Mathematical Foundation

### 6.1 The risk–coverage tradeoff

Let $\kappa$ be a confidence signal and suppose it ranks correct answers above
incorrect ones with AUC $A$. Sorting by $\kappa$ and answering the top fraction
$c$:

$$
\text{risk}(c) = \frac{1}{c}\int_0^{c}\big(1 - \text{acc}(u)\big)\,\dd u
$$ (eq:risk-coverage)

where $\text{acc}(u)$ is accuracy at rank quantile $u$.

$\square$

**With a perfectly ranking signal ($A = 1$), risk is zero until coverage reaches
the base accuracy** — you answer exactly the questions you get right. With a
useless signal ($A = 0.5$), risk is constant at the base error rate for every
coverage. **The area between those two curves is what a confidence signal is
worth**, and it is measurable.

### 6.2 Why groundedness is not accuracy

A grounded statement is entailed by the source. If the source is wrong, a
grounded statement is wrong:

$$
\text{grounded}(s, D) \wedge \neg\text{true}(D) \implies \neg\text{true}(s)
$$ (eq:grounded-not-true)

$\square$

**Groundedness measures faithfulness to a source, not truth.** For a RAG system
that is usually what you want — the system's job is to report what the documents
say — but it means groundedness metrics cannot detect a wrong document, and a
system with 100% groundedness on a corpus of errors is 100% wrong and reports
perfect scores.

### 6.3 Fabrication concentrates where the corpus is thin

Let an entity appear $n$ times in pretraining. The model's estimate of facts
about it has variance decreasing in $n$; below some $n^*$ the model has seen too
little to distinguish this entity from similar ones, and
{{eq:corpus-optimum}} places mass on the *typical* pattern for entities of that
kind.

$$
P(\text{fabrication}) \text{ decreasing in } n
$$ (eq:fabrication-vs-frequency)

**This predicts the observed pattern precisely**: hallucination is rare for
famous entities and common for obscure ones, and the transition is gradual. It
also predicts that fabrications are *plausible* rather than random — the model
fills in the typical pattern, which is why invented citations have real-looking
author names and real-looking journals.

### 6.4 A worked abstention calculation

A system answers 10,000 questions at 82% accuracy. A confidence signal ranks
with AUC 0.78. The downstream process requires ≤5% error.

At full coverage, risk is 18% — far too high. Sorting by confidence and
answering the top $c$:

$$
c = 1.00 \Rightarrow \text{risk } 0.18,\quad
c = 0.60 \Rightarrow \text{risk } \approx 0.09,\quad
c \approx 0.40 \Rightarrow \text{risk } \approx 0.05
$$

**To reach 5% error the system answers roughly 40% of questions.** The other 60%
must be escalated, deferred, or refused — and whether that is acceptable is a
product question, not a modelling one.

`abstention-curve` computes this properly across signal qualities, and the
result is worth stating separately: **holding accuracy fixed at 82% and varying
only the confidence signal moves usable coverage from 0% to 85%.** Improving the
signal raises coverage at fixed risk, and it is frequently a better investment
than improving the model.

## 7. Internal Mechanics

```mermaid {#fig:hallucination-pipeline caption="Where each mitigation acts. Retrieval supplies grounds, constraints fix format, grounding checks catch contradiction, and abstention catches the rest — and each addresses a different column of tbl:hallucination-types."}
graph TD
  A["question"] --> B{"grounds<br/>available?"}
  B -- no --> C["EXTRINSIC risk<br/>model fills the gap"]
  B -- yes --> D["retrieval supplies D<br/>ch:part-12"]
  D --> E["generate conditioned on D"]
  C --> E
  E --> F["decompose into claims<br/>eq:groundedness"]
  F --> G{"each claim<br/>entailed by D?"}
  G -- no --> H["INTRINSIC — flag or regenerate"]
  G -- yes --> I{"confidence<br/>above threshold?"}
  I -- no --> J["abstain / escalate"]
  I -- yes --> K["answer"]
  style C fill:#fde,stroke:#c69
  style H fill:#fde,stroke:#c69
  style K fill:#dfe,stroke:#5a5
```

**Temperature is a weak lever and is reached for first.** Lowering $T$ makes
generation more deterministic, which reduces the *variance* of fabrication and
not its *rate* — at $T=0$ the model still produces its most likely continuation,
which for an unanswerable question is still a fabrication. **Temperature helps
when the fabrication was a sampling accident and not when it was the mode**, and
{{eq:corpus-optimum}} says it is usually the mode.

**Why "say I don't know if you're unsure" partially works.** It shifts
probability toward abstention phrasings that exist in the model's distribution,
which is {{eq:continuation-mixture}}'s re-weighting again. It cannot create
knowledge of *when* to be unsure, so it produces both correct abstentions and
spurious ones — and the ratio is not controllable from the prompt.

**Citation requirements work by making claims checkable.** Requiring every claim
to cite a source span does not prevent fabrication; it makes fabrication
*detectable*, because the cited span can be verified to exist and to contain the
claim. **That shifts the problem from generation to verification**, which is a
much easier problem — and it is {{ch:nlp-extraction}}'s span-grounding argument
in its most useful form.

**Self-consistency detects some fabrication.** Sampling several answers and
checking agreement works because fabrications are less stable than knowledge:
the model that knows a fact reproduces it, and the model filling a gap fills it
differently each time. It costs $n\times$ and it fails exactly where
{{eq:self-consistency-condition}} said it would — on systematic error, where the
model confabulates the *same* wrong thing consistently.

**Why fabricated citations are the canonical example.** They combine every
mechanism in this chapter. Citations are highly structured, so
{{eq:corpus-optimum}} has a very strong pattern to reproduce — author names,
year, venue, volume. They concern obscure entities, so
{{eq:fabrication-vs-frequency}} puts the model in its fill-in-the-pattern
regime. And they are *checkable*, which is why they became the public face of
the problem while equally-fabricated prose passed unnoticed. **The lesson is not
that models are especially bad at citations; it is that citations are especially
easy to catch.**

**Sycophancy is hallucination with a different trigger.** When a user's question
contains a false premise — "why did the 2019 study find X?" — the aligned model
frequently accepts the premise and elaborates, because
{{eq:alignment-confidence-shift}} rewards agreement as it rewards confidence.
Raters comparing a response that corrects the user against one that helps with
the stated task tend to prefer the second. **This is the same mechanism producing
a different symptom**, and it is invisible to any evaluation whose questions all
contain true premises — which is nearly all of them.

**The measurement problem underneath all of this.** Detecting hallucination
requires knowing the truth, which is precisely what is unavailable at inference
time. Every technique here is a proxy: agreement across samples, entailment
against a source, calibrated confidence. **None observes truth**, and a system's
reported hallucination rate is always a rate against a proxy.

That has an uncomfortable consequence for the whole chapter. A mitigation can be
measured only against the proxy it was designed to satisfy, so improvements are
partly definitional: a system optimised for groundedness becomes more grounded,
and whether it becomes more truthful is a separate question
{{eq:grounded-not-true}} says the metric cannot answer. **The only escape is an
external source of truth**, which means labelled data, a verifiable tool, or a
human — and each of those is expensive in exactly the way that made the model
attractive in the first place.

## 8. Implementation

Groundedness checking, with its limits made explicit.

```python {tier=A name=groundedness-check}
"""Checking claims against a source, and what the check cannot see."""

SOURCE = (
    "The Q3 board meeting is scheduled for Tuesday 14 November at 09:00 in "
    "the Bristol office. Attendance is mandatory for all directors. The "
    "agenda covers the budget review and the hiring plan."
)

CANDIDATES = {
    "Tuesday 14 November": ("grounded", "appears verbatim"),
    "09:00": ("grounded", "appears verbatim"),
    "Bristol office": ("grounded", "appears verbatim"),
    "Thursday 16 November": ("INTRINSIC", "contradicts the stated date"),
    "the London office": ("INTRINSIC", "contradicts the stated location"),
    "forty people attended": ("EXTRINSIC", "source says nothing about numbers"),
    "the CFO will present": ("EXTRINSIC", "source names no presenter"),
    "attendance is mandatory": ("grounded", "appears in substance"),
}


def substring_grounded(claim, source):
    """The cheapest possible check: does the claim's text appear in the source?
    This is ch:nlp-extraction's span-grounding idea at its simplest."""
    return claim.lower() in source.lower()


print(f"{'claim':<26} {'in source':>10} {'truth':<12} note")
for claim, (label, note) in CANDIDATES.items():
    found = substring_grounded(claim, SOURCE)
    print(f"{claim:<26} {str(found):>10} {label:<12} {note}")

detected = sum(1 for c, (l, _) in CANDIDATES.items()
               if not substring_grounded(c, SOURCE) and l != "grounded")
total_bad = sum(1 for _, (l, _) in CANDIDATES.items() if l != "grounded")
false_alarms = sum(1 for c, (l, _) in CANDIDATES.items()
                   if not substring_grounded(c, SOURCE) and l == "grounded")

print(f"\nsubstring check: flagged {detected}/{total_bad} bad claims, "
      f"{false_alarms} false alarms on good ones")
print("""
The substring check catches every fabricated claim here and also flags a
grounded one — 'attendance is mandatory' is supported in substance and not
verbatim. That is the method's shape: high recall on fabrication, poor
precision, because paraphrase is indistinguishable from invention to a string
match.

It is still worth doing. It costs a string search, it catches the fabrications
that matter most (invented specifics — names, dates, numbers), and its false
alarms are cheap to route to a stronger check.""")


# Equation (eq:groundedness): decomposition is where the difficulty lives.
SUMMARIES = {
    "faithful": [
        "The Q3 board meeting is on Tuesday 14 November.",
        "It will be held in the Bristol office.",
        "Attendance is mandatory for directors.",
    ],
    "one intrinsic error": [
        "The Q3 board meeting is on Thursday 16 November.",
        "It will be held in the Bristol office.",
        "Attendance is mandatory for directors.",
    ],
    "one extrinsic addition": [
        "The Q3 board meeting is on Tuesday 14 November.",
        "It will be held in the Bristol office.",
        "Roughly forty directors are expected to attend.",
    ],
}

# A stand-in entailment oracle. In production this is a model or a human;
# the point of the listing is what the SCORE does, not how entailment is judged.
ENTAILED = {
    "The Q3 board meeting is on Tuesday 14 November.": True,
    "It will be held in the Bristol office.": True,
    "Attendance is mandatory for directors.": True,
    "The Q3 board meeting is on Thursday 16 November.": False,
    "Roughly forty directors are expected to attend.": False,
}

print(f"\n{'summary':<24} {'claims':>7} {'grounded':>9} {'score':>7}")
for name, claims in SUMMARIES.items():
    ok = sum(ENTAILED[c] for c in claims)
    print(f"{name:<24} {len(claims):>7} {ok:>9} {ok / len(claims):>7.3f}")

print("""
Both faulty summaries score 0.667 — equation (eq:groundedness) cannot tell an
intrinsic contradiction from an extrinsic addition, and they need completely
different fixes. A groundedness score is a useful aggregate and a poor
diagnostic; the per-claim labels are what actually direct the work.""")
```

Now the abstention curve, which is the deployable mitigation:

```python {tier=A name=abstention-curve}
"""Risk against coverage. Equation (eq:risk-coverage), measured."""
import math

import numpy as np

rng = np.random.default_rng(0)
N = 20_000
BASE_ACCURACY = 0.82


def make_system(auc):
    """Answers with a given accuracy, plus a confidence signal of a given
    ranking quality. AUC 0.5 is useless, 1.0 is perfect."""
    correct = rng.random(N) < BASE_ACCURACY
    # Separation chosen to hit the target AUC for two Gaussians.
    sep = math.sqrt(2) * _probit(auc)
    conf = rng.normal(np.where(correct, sep, 0.0), 1.0)
    return correct, conf


def _probit(p):
    """Inverse normal CDF by bisection — no scipy dependency."""
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1 + math.erf(mid / math.sqrt(2)))
        lo, hi = (mid, hi) if cdf < p else (lo, mid)
    return (lo + hi) / 2


def risk_at_coverage(correct, conf, coverage):
    """Answer the top `coverage` fraction by confidence; report the error rate."""
    k = max(int(coverage * len(conf)), 1)
    idx = np.argsort(-conf)[:k]
    return float(1 - correct[idx].mean())


print(f"base accuracy {BASE_ACCURACY:.0%}, so risk at full coverage is "
      f"{1 - BASE_ACCURACY:.0%}\n")
print(f"{'coverage':>10} " + " ".join(f"{'AUC ' + str(a):>10}"
                                       for a in (0.5, 0.7, 0.8, 0.9, 0.99)))
systems = {a: make_system(a) for a in (0.5, 0.7, 0.8, 0.9, 0.99)}
for cov in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
    row = " ".join(f"{risk_at_coverage(*systems[a], cov):>10.3f}"
                   for a in (0.5, 0.7, 0.8, 0.9, 0.99))
    print(f"{cov:>10.0%} {row}")

print("""
The AUC=0.5 column is flat: a useless confidence signal means abstaining buys
nothing, because the questions you decline are no worse than the ones you keep.
Every other column falls with coverage, and the rate it falls at IS the value of
the confidence signal.""")

# The product question: what coverage does a risk budget buy?
TARGET_RISK = 0.05
print(f"\nrisk budget {TARGET_RISK:.0%} — what coverage is achievable?\n")
print(f"{'AUC':>6} {'max coverage':>14} {'questions answered':>20}")
for a in (0.5, 0.7, 0.8, 0.9, 0.99):
    correct, conf = systems[a]
    best = 0.0
    for cov in np.linspace(0.02, 1.0, 99):
        if risk_at_coverage(correct, conf, cov) <= TARGET_RISK:
            best = cov
    print(f"{a:>6.2f} {best:>13.0%} {int(best * N):>20,}")

print("""
This is the table to take to a product discussion, and the AUC column is the
one to read. A useless signal answers NOTHING within a 5% budget — every
question it would keep is as likely to be wrong as one it would decline. At AUC
0.8 the system answers 44% of questions; at 0.9 it answers 72%.

Note what that implies about where to invest. The model's accuracy is 82% in
every row — only the confidence signal changed, and it moved usable coverage
from 0% to 85%. Improving the SIGNAL raises coverage at fixed risk, and it is
frequently cheaper than improving accuracy: a well-calibrated 82% model is worth
far more here than a poorly-calibrated 86% one.""")
```

And the frequency effect, which predicts where fabrication concentrates:

```python {tier=A name=fabrication-and-frequency}
"""Fabrication concentrates where the corpus was thin. Eq (eq:fabrication-vs-frequency)."""
import numpy as np

rng = np.random.default_rng(3)

# Entity frequency in a corpus is heavy-tailed (ch:nlp-preprocessing's Zipf).
N_ENTITIES = 6000
ranks = np.arange(1, N_ENTITIES + 1)
frequency = 1e7 / ranks ** 1.1


def fabrication_rate(n):
    """Decreasing in corpus frequency: below n*, the model fills in the
    typical pattern for entities of this kind rather than recalling."""
    n_star = 800.0
    return float(1 / (1 + (n / n_star) ** 0.8))


print(f"{'rank':>8} {'corpus mentions':>17} {'P(fabricate)':>14} "
      f"{'entity class'}")
CLASSES = [(1, "the most-discussed entities"), (10, ""), (100, ""),
           (500, ""), (2000, ""), (5000, "long tail")]
for r, note in CLASSES:
    n = frequency[r - 1]
    print(f"{r:>8,} {n:>17,.0f} {fabrication_rate(n):>14.3f}  {note}")

# What that means for an evaluation set's composition.
print(f"\n{'evaluation set':<32} {'mean P(fabricate)':>19}")
SETS = {
    "famous entities only (top 100)":    ranks[:100],
    "uniform over all entities":         ranks,
    "sampled by corpus frequency":       None,
    "long tail only (rank > 3000)":      ranks[3000:],
}
for name, sel in SETS.items():
    if sel is None:
        p = frequency / frequency.sum()
        sel = rng.choice(ranks, size=4000, p=p)
    rates = [fabrication_rate(frequency[r - 1]) for r in sel]
    print(f"{name:<32} {np.mean(rates):>19.3f}")

print("""
The same model has a fabrication rate varying by more than an order of magnitude
depending on which entities you evaluate it on — and 'sampled by corpus
frequency' looks excellent because it is dominated by entities the model has
seen thousands of times.

That is the measurement trap. A benchmark built from prominent entities
understates fabrication on exactly the queries where users encounter it, because
users ask about the things THEY care about, not the things the corpus discussed
most. Evaluation sets must be sampled from the traffic distribution, not from a
convenient one.""")
```

## 9. Practical Example

A team's document assistant fabricates. They plan to add retrieval. Whether that
helps depends entirely on which kind of fabrication they have, and the
measurement takes an afternoon.

```python {tier=A name=mitigation-selection}
"""Which mitigation? It depends on the failure mix, which must be measured."""

# A hundred sampled bad outputs, classified by hand.
OBSERVED = {
    "intrinsic (contradicts the document)": 34,
    "extrinsic (unsupported addition)":     41,
    "wrong tool/argument value":             9,
    "malformed output":                      6,
    "correct but unhelpful":                10,
}

# What each mitigation addresses, and roughly how much of it.
MITIGATIONS = {
    "retrieval (add grounds)": {
        "extrinsic (unsupported addition)": 0.75},
    "citation requirement + span check": {
        "intrinsic (contradicts the document)": 0.70,
        "extrinsic (unsupported addition)": 0.40},
    "constrained decoding": {
        "malformed output": 1.00,
        "wrong tool/argument value": 0.35},
    "lower temperature": {
        "intrinsic (contradicts the document)": 0.15,
        "extrinsic (unsupported addition)": 0.10},
    "abstention at a confidence threshold": {
        "intrinsic (contradicts the document)": 0.30,
        "extrinsic (unsupported addition)": 0.45,
        "wrong tool/argument value": 0.30},
}

total = sum(OBSERVED.values())
print(f"{total} sampled failures\n")
print(f"{'failure class':<40} {'count':>7} {'share':>8}")
for k, v in sorted(OBSERVED.items(), key=lambda kv: -kv[1]):
    print(f"{k:<40} {v:>7} {v / total:>8.0%}")

print(f"\n{'mitigation':<38} {'failures removed':>18} {'share':>8}")
ranked = []
for name, effects in MITIGATIONS.items():
    removed = sum(OBSERVED.get(k, 0) * frac for k, frac in effects.items())
    ranked.append((name, removed))
for name, removed in sorted(ranked, key=lambda x: -x[1]):
    print(f"{name:<38} {removed:>18.1f} {removed / total:>8.0%}")

best = max(ranked, key=lambda x: x[1])
print(f"\nsingle best intervention: {best[0]} ({best[1] / total:.0%})")

# Combining, without double-counting.
print(f"\n{'stacked':<52} {'cumulative removed':>19}")
remaining = dict(OBSERVED)
cumulative = 0.0
for name, _ in sorted(ranked, key=lambda x: -x[1])[:3]:
    removed = 0.0
    for k, frac in MITIGATIONS[name].items():
        take = remaining.get(k, 0) * frac
        remaining[k] = remaining.get(k, 0) - take
        removed += take
    cumulative += removed
    print(f"{'+ ' + name:<52} {cumulative / total:>18.0%}")

print("""
Retrieval is the single best intervention HERE because extrinsic failures are
the largest class — and that was a measurement, not an assumption. On a failure
mix dominated by intrinsic contradiction it would be near the bottom of the
table, because retrieval supplies grounds and intrinsic failures already had
them.

The general rule: classify a hundred failures before choosing a mitigation. It
costs an afternoon and it is the difference between the top row of this table
and the bottom one.""")
```

> PRODUCTION TIP: Sample a hundred failures and classify them by
> {{eq:hallucination-taxonomy}} before building anything. Teams routinely add
> retrieval to fix contradiction problems, which is buying the wrong mitigation
> at considerable expense.

## 10. Production Considerations

**Classify failures before mitigating.** {{tbl:hallucination-types}} — the two
types need different machinery.

**Require citations for any claim that can be sourced.** It converts generation
into verification, which is the easier problem.

**Trace the abstention curve on your own data.** {{eq:risk-coverage}} — and note
that improving the confidence signal frequently beats improving accuracy.

**Sample evaluation sets from traffic, not from prominence.**
`fabrication-and-frequency` shows an order-of-magnitude difference between
evaluation-set compositions for the same model.

**Do not rely on temperature.** It reduces fabrication variance, not rate
({{sec:7-internal-mechanics}}).

**Remember groundedness is not truth.** {{eq:grounded-not-true}} — a system
faithfully reporting wrong documents scores perfectly.

**What to monitor:** abstention rate, groundedness on a sampled set, citation
coverage (fraction of claims with a verifiable source), and the failure-class
mix over time. The mix shifting is the signal that a mitigation has stopped
matching the problem.

## 11. Common Mistakes

**Beginners:**

*Treating hallucination as a bug.* {{eq:corpus-optimum}} — the objective
contains no truthfulness term.

*Lowering temperature to fix it.* Helps with sampling accidents, not with the
mode.

*Believing "say I don't know" instructions work reliably.* They re-weight
existing behaviour and cannot create knowledge of when to be unsure.

**Experienced practitioners:**

*Adding retrieval for an intrinsic problem.* The grounds were already there.

*Reporting groundedness as accuracy.* {{eq:grounded-not-true}}.

*Evaluating on prominent entities.* `fabrication-and-frequency` shows the
measurement trap.

*Setting an abstention threshold from a target error rate on an aligned model.*
Confidence survives alignment as a rank, not a probability
({{ch:llm-next-token}}) — the curve must be traced empirically.

*Expecting self-consistency to catch systematic fabrication.*
{{eq:self-consistency-condition}} — it detects instability, and a consistent
confabulation is stable.

*Evaluating only on questions with true premises.* Sycophancy is invisible to
such a set, and it is the same mechanism as confident fabrication
({{sec:7-internal-mechanics}}).

## 12. Failure Modes

**Confident fabrication.** High confidence, wrong content. *Cause:*
{{eq:alignment-confidence-shift}}. *Detection:* requires ground truth or a
source.

**Plausible invented specifics.** Real-looking names, dates and citations.
*Cause:* {{eq:fabrication-vs-frequency}} — the model fills the typical pattern.
*Detection:* verify specifics against a source; they are the easiest class to
check and the most damaging when missed.

**Faithful reporting of a wrong source.** *Detection:* not by any groundedness
metric ({{eq:grounded-not-true}}). Requires source quality control.

**Abstention miscalibration after a model update.** *Symptom:* abstention rate
moving with no threshold change. *Detection:* monitor the rate directly.

**Evaluation-set optimism.** *Cause:* prominence-weighted sampling.
*Detection:* compare fabrication rate on head and tail entities.

**Sycophantic agreement.** The model adopting a false premise in the user's
question. *Cause:* alignment rewards agreement. *Detection:* evaluation with
deliberately false premises, which almost nobody runs.

## 13. Alternatives

{#tbl:hallucination-mitigations caption="Mitigations by which failure class each addresses and what it costs. No row addresses everything, which is why the failure mix must be measured before choosing."}

| Mitigation | Addresses | Cost | Guarantee |
|---|---|---|---|
| Retrieval | extrinsic | index + prefill | none |
| Citation + span verification | intrinsic, some extrinsic | verification pass | detection |
| Constrained decoding | malformed only | negligible | structural |
| Abstention | both, by declining | coverage | risk bound |
| Self-consistency | unstable fabrication | $n\times$ | none |
| Lower temperature | sampling accidents | none | none |
| Fine-tuning on refusals | teaches the behaviour | training | none |
| Human review | everything | throughput | as good as the reviewer |

**What genuinely differs.** Only abstention and constrained decoding provide
anything like a guarantee, and both do so by *declining to produce* rather than
by producing better output. Everything else shifts a probability. **That is the
honest summary of the field**: there is no mitigation that makes a model
truthful, only mitigations that supply grounds, make claims checkable, or refuse.

## 14. Evaluation

**Measuring hallucination.** Three things must be specified or the number is
uninterpretable:

1. **Which type** — {{eq:hallucination-taxonomy}}, since the rates differ and
   the mitigations differ.
2. **Against what source** — groundedness is relative to $D$, and
   {{eq:grounded-not-true}} means a wrong $D$ gives flattering scores.
3. **On what distribution of entities** — `fabrication-and-frequency` shows an
   order of magnitude riding on this choice.

**Measuring a mitigation.** Before and after, on the *same* classified failure
sample, reporting per-class deltas. An aggregate improvement can hide a
mitigation that helped one class and hurt another.

**The limit.** Every technique here measures against a proxy — a source, an
agreement, a confidence. **None observes truth**, and a reported hallucination
rate is always a rate against a proxy. Saying which proxy is the minimum
honesty requirement, and {{part:25}} builds this into a discipline.

## 15. Advanced Concepts

**Semantic entropy.** {{maturity:EMERGING}} Clustering sampled generations by
meaning and computing entropy over clusters, separating "unsure how to phrase
it" from "unsure what is true". A genuine improvement on token-level entropy for
this purpose.

**Self-evaluation.** {{maturity:EMERGING}} {{cite:kadavath2022}} — asking the
model whether its answer is correct provides a second channel, which
{{eq:two-uncertainties}} says is necessary to estimate epistemic uncertainty at
all.

**Attribution and citation verification.** {{maturity:ESTABLISHED}} Requiring
span-level citations and verifying them mechanically. The most reliable
available mitigation for sourceable claims, and it is
{{ch:nlp-extraction}}'s span extraction doing the work.

**Retrieval-augmented verification.** {{maturity:EMERGING}} Generating first,
then retrieving evidence for each claim and checking entailment. Costs a second
pass and catches extrinsic fabrication that generation-time retrieval missed.

**Training for calibrated refusal.** {{maturity:EMERGING}} Including "I don't
know" as a *rewarded* response in alignment data, directly countering
{{eq:alignment-confidence-shift}}. It works and it trades away helpfulness on
questions the model could have answered, which is why it is applied cautiously —
and note the trade is not symmetric, because a spurious refusal is visible to
the user while a confident fabrication is not, so the *perceived* cost of
refusal exceeds its actual cost.

**Knowledge cutoffs and temporal fabrication.** {{maturity:ESTABLISHED}} A model
asked about events after its training cutoff has no grounds by construction, so
{{eq:corpus-optimum}} guarantees fabrication rather than merely permitting it.
This is the one case where the failure is perfectly predictable in advance from
the question's date, and it is therefore the easiest to route around — which is
why time-sensitive queries are the canonical retrieval trigger in
{{part:12}}.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-pretraining}}'s objective is {{eq:corpus-optimum}} and
contains no truthfulness term. {{ch:fm-rlhf}}'s {{eq:rlhf-optimal-policy}}
becomes {{eq:alignment-confidence-shift}}. {{ch:llm-next-token}}'s
{{eq:two-uncertainties}} is why the model cannot report its own ignorance, and
its rank-not-probability finding is why abstention thresholds must be traced
empirically. {{ch:nlp-extraction}}'s span grounding is the strongest cheap
defence. {{ch:llm-structured-output}} removes the structural class entirely, and
{{ch:llm-function-calling}}'s argument hallucination is this chapter's problem
applied to actions. {{ch:fm-datasets}}'s corpus composition is
{{eq:fabrication-vs-frequency}}.

**Forwards.** {{part:12}} is retrieval as the systematic answer to extrinsic
hallucination, and {{ch:rag-failures}} catalogues where it does not work.
{{part:25}} builds hallucination evaluation into a discipline.
{{part:27}} takes up the consequences when confident fabrication reaches
decisions about people.

## 17. Exercises

**Beginner**

1. Why does the training objective permit hallucination?
2. Classify: a summary states a figure the document does not mention. Which
   type, and which mitigation?
3. Why does lowering temperature help only partially?

**Intermediate**

4. Using {{eq:groundedness}}, score a four-claim summary with one unsupported
   claim, and say what the score fails to distinguish.
5. Explain {{eq:alignment-confidence-shift}} in terms of what raters can and
   cannot see.
6. A system is 78% accurate with an AUC-0.85 confidence signal. Estimate the
   coverage achievable at 10% risk.

**Advanced**

7. Derive {{eq:risk-coverage}} and describe the curves for $A=0.5$ and $A=1$.
8. Prove {{eq:grounded-not-true}} and state what it implies for RAG evaluation.
9. Explain why {{eq:fabrication-vs-frequency}} predicts fabrications are
   plausible rather than random.

**Implementation**

10. Extend `groundedness-check` with an entailment model rather than a substring
    match, and compare precision and recall on paraphrased claims.
11. Implement semantic entropy: sample $n$ answers, cluster by equivalence, and
    compare its detection against token entropy.
12. Build the abstention curve on a real labelled set and find the threshold for
    a stated risk budget.
13. Implement the failure-classification pipeline from
    `mitigation-selection` and run it on a sample of your own system's errors.

**Reasoning**

14. Your groundedness score is 0.98 and users report wrong answers. Give three
    explanations.
15. Explain why no mitigation makes a model truthful, and what the available
    ones actually do.

## 18. Interview Questions

**Beginner**

1. What is hallucination and why does it happen?
2. What is the difference between intrinsic and extrinsic?
3. Does lowering temperature fix it?

**Intermediate**

4. Why does alignment increase confident fabrication?
5. What does a groundedness score measure, and what does it miss?
6. How would you build an abstention policy?

**Senior**

7. Your system fabricates. Walk through choosing a mitigation.
8. How would you evaluate hallucination honestly?
9. Why is improving the confidence signal sometimes better than improving
   accuracy?

**Systems**

10. Design citation verification for a document assistant.
11. What would you monitor to detect a rising hallucination rate before users
    report it?

## 19. Research Questions

**Can calibrated refusal be trained without a helpfulness cost?**
{{eq:alignment-confidence-shift}} says preference data pushes toward confidence.
Including rewarded refusals counters it. Measure the frontier: refusal accuracy
against helpfulness loss, as a function of how refusals are rewarded.

**Does self-evaluation survive alignment?** {{cite:kadavath2022}} measured a
second channel. Raters also prefer confident self-assessments, so the same
pressure applies. Whether it degrades in the same direction is testable and
would determine how much weight that channel can bear.

**Is fabrication predictable from corpus frequency?**
{{eq:fabrication-vs-frequency}} is a plausible model.
Test it directly on an open corpus where entity frequencies are countable — if
it holds quantitatively, fabrication risk becomes estimable in advance for a
given query.

**What is the best claim-decomposition for {{eq:groundedness}}?** The metric is
sensitive to how a generation is split into atomic claims, and different
implementations disagree. Characterise the sensitivity, because it currently
makes groundedness numbers incomparable across papers.

## 20. Chapter Summary

Hallucination follows from the objective. {{eq:clm-loss}} optimises toward the
corpus's conditional distribution {{eq:corpus-optimum}}, which contains no term
for truth — and by cross-entropy's properness, hedging where the corpus was
confident *increases* the loss. **The model is doing what it was built to do**,
which is why "confabulation" is the better word and why the framing matters for
engineering.

**The taxonomy decides the mitigation.**
{{eq:hallucination-taxonomy}}: intrinsic hallucination contradicts a supplied
source and is checkable against it; extrinsic asserts what the source does not
address and is not. **Retrieval fixes extrinsic and does nothing for
intrinsic**, where grounds were already present — and adding retrieval to fix a
contradiction problem is the most common misapplied mitigation in the field.
`mitigation-selection` shows the ranking inverting entirely with the failure
mix, which is why a hundred classified failures must come before any building.

**Alignment increases confident fabrication, provably.**
{{eq:alignment-confidence-shift}}: raters comparing a hedge against a confident
answer cannot see which is correct, so they prefer the confident one, and
{{eq:rlhf-optimal-policy}} shifts mass accordingly. This is not an
implementation flaw — it follows from comparing responses without ground truth.

**Groundedness is faithfulness, not truth** {{eq:grounded-not-true}}. A system
reporting wrong documents perfectly scores perfectly. And
{{eq:fabrication-vs-frequency}} predicts both where fabrication concentrates —
the long tail — and why it is *plausible* rather than random, since the model
fills in the typical pattern for entities of that kind. That creates a
measurement trap `fabrication-and-frequency` quantifies: the same model's
fabrication rate varies by an order of magnitude with the evaluation set's
entity composition.

Finally the honest summary of the field. Only abstention and constrained
decoding offer anything resembling a guarantee, and both work **by declining to
produce**. Everything else shifts a probability. There is no mitigation that
makes a model truthful — only ones that supply grounds, make claims checkable,
or refuse.

## 21. Further Reading

{{cite:ji2023survey}} for the taxonomy, which is the chapter's organising
contribution. Read §2 and §3; the survey's breadth across generation tasks is
useful precisely because it shows the intrinsic/extrinsic split holding outside
the LLM setting where it was later popularised.

{{cite:kadavath2022}} for self-evaluation, and read it as a paper about a
*second channel* rather than about accuracy — that framing is what makes it
relevant to {{eq:two-uncertainties}}.

{{cite:guo2017calibration}} from {{ch:llm-next-token}} is worth revisiting here:
abstention depends entirely on a confidence signal, and its quality is a
calibration question before it is a hallucination question.

{{cite:lee2022dedup}} for the corpus side of
{{eq:fabrication-vs-frequency}} — memorisation and fabrication are two ends of
the same frequency axis, which is a connection the two literatures rarely make.

**Where to go next:** {{ch:llm-long-context}} takes up a related failure that is
frequently misdiagnosed as hallucination — a model that has the evidence in its
context and does not use it.
