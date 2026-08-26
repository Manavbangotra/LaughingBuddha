---
id: llm-prompting
number: 93
part: X
tier: full
status: draft
requires: [llm-prompt-lifecycle, llm-decoding, fm-instruction-tuning, fm-rlhf,
           ml-metrics, ds-experiments, llm-next-token]
provides: [prompt-sensitivity, few-shot-prompting, chain-of-thought,
           system-prompt, self-consistency, prompt-evaluation, format-effects,
           demonstration-role, prompt-as-hyperparameter]
citations: [brown2020, min2022, wei2022cot, kojima2022, ouyang2022,
            holtzman2020, liu2023lost, ji2023survey]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what few-shot demonstrations actually supply, and cite the control
   that established it.
2. Measure prompt sensitivity and report it as a number rather than an
   impression.
3. Explain chain-of-thought's capability claim and separate it from the
   faithfulness claim.
4. Explain what a system prompt is mechanically and what it is not.
5. Implement self-consistency and say what it costs.
6. Evaluate a prompt with the same discipline you would apply to a model.
7. Distinguish prompting advice with controlled evidence from advice without it.

## 2. Why This Matters

**This is the chapter most at risk of becoming a list of tricks**, and the
antidote is the standard the rest of the book has used: what was the control?
Prompting is an area where practice ran far ahead of evidence, where the same
technique circulates under a dozen names, and where a great deal of confident
advice has never been tested against an alternative.

**The one place someone ran the control, the result was surprising.**
{{cite:min2022}} replaced the labels in few-shot demonstrations with *random*
ones and performance barely moved. Whatever in-context learning is, it is not
learning the input–output mapping from the examples — and that changes how
prompts should be written.

**Prompt sensitivity is a measurable property and it is large.** The same task,
phrased two reasonable ways, can differ enough to reverse a model comparison.
That makes an unpinned prompt a confound in every evaluation, which is
{{cite:levy2015}}'s lesson in yet another costume.

**And the highest-value technique here is nearly free.** Chain-of-thought
({{cite:wei2022cot}}) and its zero-shot form ({{cite:kojima2022}}) improve
multi-step accuracy substantially for the cost of some output tokens, and the
mechanism — the model conditions on its own intermediate steps — is legible
rather than magical.

## 3. Prerequisites

{{ch:llm-prompt-lifecycle}} for where the prompt string is constructed — stage
$t_2$ of {{eq:request-stages}}. {{ch:fm-instruction-tuning}} for why a model
follows instructions at all, and for the template.
{{ch:fm-rlhf}} for why aligned models prefer certain response shapes.
{{ch:llm-decoding}} for temperature, which self-consistency depends on.
{{ch:llm-next-token}} for confidence. {{ch:ml-metrics}} and
{{ch:ds-experiments}} for what it takes to establish an effect.

## 4. Intuitive Explanation

A prompt is a string. The model conditions on it and produces a distribution
({{eq:llm-as-function}}). Everything called "prompt engineering" is choosing
that string.

**Why it works at all** is {{ch:fm-instruction-tuning}}: the model was trained on
instruction–response pairs, so text shaped like an instruction is followed by
text shaped like a response. Prompting is not programming; it is **conditioning
a distribution**, and the model has no obligation to comply.

**Few-shot demonstrations** were the original technique
({{cite:brown2020}}): show three examples of the task, then the real input.
The natural reading is that the model learns the mapping from the examples.

**{{cite:min2022}} tested that reading and it did not survive.** With randomly
permuted labels — demonstrations showing *wrong* answers — classification
performance dropped only slightly. What the demonstrations supply is the label
space (which answers are possible), the input distribution (what these inputs
look like), and the format (how a response is shaped). **They specify the task's
shape rather than teaching its content.**

> NOTE: This has a direct practical consequence. If demonstrations mainly
> convey format and label space, then effort spent perfecting each exemplar's
> correctness is misallocated, and effort spent covering the label space and
> varying the surface form is not. That is the opposite of how most few-shot
> prompts are written.

**Chain-of-thought** is the other well-supported technique. Ask for reasoning
before the answer and multi-step accuracy improves substantially
({{cite:wei2022cot}}). {{cite:kojima2022}} showed that a single instruction —
"Let's think step by step" — recovers much of the benefit with no exemplars.

The mechanism is not mysterious. The model produces one token at a time,
conditioning on everything before it ({{eq:autoregressive-factorisation}}).
Intermediate steps put the partial results **into the context**, where later
tokens can attend to them. Without them the model must compute the whole answer
in one forward pass, at fixed depth. **Chain-of-thought converts depth into
length**, and length is the dimension the architecture can extend.

**A system prompt is not a privileged channel.** It is text placed in a
particular position by the template, and models were trained to weight that
position heavily. It has no enforcement, which is the entire premise of
{{part:26}}.

**The mental model:** a prompt selects a region of the model's conditional
distribution, and techniques work by moving into a region where good responses
are more likely. Where it breaks down: the model's behaviour is sensitive to
details of the string that carry no meaning to a human — whitespace, ordering,
phrasing — so the region is not selected as precisely as the metaphor implies.

## 5. Formal Explanation

### 5.1 A prompt is a conditioning event

$$
P(\vec{y}\given \text{prompt}) = \prod_t P\big(y_t\given \text{prompt}, y_{<t}\big)
$$ (eq:prompt-conditioning)

Prompt engineering searches over the conditioning text for one that concentrates
mass on good responses. **It changes no parameters**, which is both why it is
cheap and why it cannot exceed what the model can already do
({{eq:adaptation-information-ratio}}).

### 5.2 Prompt sensitivity, defined

For a task with metric $m$ and a set of semantically equivalent prompts
$\{p_1,\dots,p_k\}$:

$$
\text{sensitivity} = \max_i m(p_i) - \min_i m(p_i)
$$ (eq:prompt-sensitivity)

**This number is routinely large enough to exceed the difference between model
generations.** The consequence is methodological: a model comparison that does
not hold the prompt fixed — or better, that does not report performance across a
*distribution* of prompts — is confounded.

> IMPORTANT: Report $\E_i[m(p_i)]$ and its spread, not $\max_i m(p_i)$. Selecting
> the best prompt on your evaluation set and reporting that number is
> overfitting, and it is the single most common methodological error in applied
> LLM work. It is {{ch:mle-hpo}}'s winner's curse with a prompt in place of a
> hyperparameter.

### 5.3 What demonstrations supply

{{cite:min2022}}'s decomposition. A demonstration set conveys four things:

$$
\underbrace{\mathcal{Y}}_{\text{label space}},\quad
\underbrace{\Data_x}_{\text{input distribution}},\quad
\underbrace{\text{format}}_{\text{sequence shape}},\quad
\underbrace{f: x\mapsto y}_{\text{the mapping}}
$$ (eq:demonstration-content)

Ablating the fourth — randomising labels — costs little. Ablating the first
three costs a great deal.

**The mapping is the component people believe they are supplying and is the one
that matters least**, at least for classification. The effect is weaker for
generation tasks, and the scope of the finding should be stated when it is
cited.

### 5.4 Chain-of-thought as depth-for-length

A transformer applies $L$ layers. Any computation it performs in one forward
pass has at most $L$ sequential steps ({{ch:llm-anatomy}}).

With intermediate tokens, a computation requiring $S$ sequential steps can be
spread over $\lceil S/L\rceil$ generated tokens, because each generated token is
appended to the context and later tokens attend to it:

$$
\text{effective depth} = L \times (\text{tokens generated})
$$ (eq:cot-depth)

$\square$

**This is why chain-of-thought helps most on problems with many sequential
steps** — arithmetic, multi-hop reasoning — and barely at all on single-step
classification. The prediction is testable and it holds.

### 5.5 Self-consistency

Sample $n$ chains at $T > 0$, take the majority answer:

$$
\hat{y} = \operatorname*{mode}\big\{\text{answer}(\vec{c}_i)\big\}_{i=1}^{n},
\qquad \vec{c}_i \sim P(\cdot\given\text{prompt})
$$ (eq:self-consistency)

It works when errors are **independent and the correct answer is modal**: wrong
answers scatter across many values while correct ones concentrate. It fails when
the model is systematically wrong, because then the mode is the error — and
sampling more only measures the bias more precisely.

**Cost is linear in $n$** and this interacts with
{{ch:llm-decoding}}: the temperature must be high enough that chains genuinely
differ, which is the opposite of the setting the same task would use for one
generation.

## 6. Mathematical Foundation

### 6.1 Why self-consistency helps, and when it does not

Let the model produce the correct answer with probability $p$ per sample, and
distribute its errors over $k$ distinct wrong answers with probability
$(1-p)/k$ each.

Majority voting over $n$ samples succeeds when the correct answer has the most
votes. For large $n$ this happens whenever

$$
p > \frac{1-p}{k}
\iff p > \frac{1}{k+1}
$$ (eq:self-consistency-condition)

$\square$

**With errors scattered over many values, $p$ need only exceed $1/(k+1)$** — for
$k=9$, a per-sample accuracy of 11% suffices asymptotically. That is why the
technique can work on problems the model solves only occasionally.

And the failure case follows immediately: if the model concentrates its errors
on one wrong answer, $k=1$ and the condition becomes $p > 1/2$. **Systematic
error defeats voting entirely**, because voting measures the mode and the mode
is wrong.

### 6.2 The winner's curse for prompts

Choose the best of $k$ prompts on an evaluation set of size $n$. Each prompt's
measured score is $\hat{m}_i = m_i + \varepsilon_i$ with
$\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$, $\sigma^2 \approx m(1-m)/n$.

The selected prompt's *reported* score is $\max_i \hat{m}_i$, whose expectation
exceeds the best true score by approximately

$$
\E\big[\max_i \varepsilon_i\big] \approx \sigma\sqrt{2\ln k}
$$ (eq:prompt-selection-bias)

$\square$

**With $k = 20$ prompts and $n = 200$ evaluation examples at $m = 0.7$:**
$\sigma = \sqrt{0.21/200} = 0.032$, and the bias is
$0.032\sqrt{2\ln 20} = 0.078$ — nearly eight points of illusory improvement from
selection alone.

This is exactly {{ch:mle-hpo}}'s result, and the same fix applies: **re-evaluate
the chosen prompt on a fresh split.** One extra evaluation turns a number you
selected into a number you measured.

### 6.3 A worked sensitivity calculation

Five semantically equivalent phrasings of a classification instruction, measured
on 500 examples:

$$
m = (0.71,\ 0.68,\ 0.74,\ 0.66,\ 0.72)
$$

$$
\bar{m} = 0.702,\qquad
\text{sd} = 0.031,\qquad
\text{sensitivity} = 0.74 - 0.66 = 0.08
$$

**An eight-point spread from rephrasing alone.** If two models are compared at
their own best prompts, a two-point difference between them is uninterpretable —
it is well inside the noise that phrasing introduces.

The binomial standard error at $n=500$, $m=0.7$ is
$\sqrt{0.21/500} = 0.020$, so the spread is about four standard errors: **the
sensitivity is real, not sampling noise.**

## 7. Internal Mechanics

```mermaid {#fig:prompt-structure caption="What is actually in the string the model sees. The template supplies the role markers; everything else is content the caller chose. Only the last element is the user's actual question — and by token count it is frequently the smallest part."}
graph TD
  A["template: system marker"] --> B["system prompt<br/>role, constraints, format"]
  B --> C["template: end + user marker"]
  C --> D["few-shot demonstrations<br/>label space, format, input distribution"]
  D --> E["retrieved context<br/>ch:part-12"]
  E --> F["the actual question"]
  F --> G["template: assistant marker"]
  G --> H["generation begins here"]
  style B fill:#dfe,stroke:#5a5
  style H fill:#fde,stroke:#c69
```

**Position matters and is measurable.** {{cite:liu2023lost}} found a U-shaped
retrieval curve — content at the beginning and end of a long context is used
better than content in the middle. For prompt construction this means **the
instruction should not be buried**, and repeating it after a long retrieved
passage is a cheap and effective intervention that looks redundant and is not.

**Format effects are large and arbitrary.** Whether options are labelled
`A/B/C` or `1/2/3`, whether a colon or a newline separates fields, whether the
answer is requested before or after the reasoning — all measurably change
results, and none of them carries meaning. This is the least satisfying fact in
the chapter and it is why {{sec:14-evaluation}} insists on measuring rather than
reasoning about prompts.

**Answer position and chain-of-thought.** If the answer is requested *before*
the reasoning, {{eq:cot-depth}} does not apply — the answer token cannot attend
to reasoning that has not been generated. **Chain-of-thought only works if the
reasoning precedes the answer**, and prompts that ask for "the answer, then your
reasoning" get the cost without the benefit. This is a mechanical consequence of
causal masking ({{ch:tf-masking-kv}}) and it is a common error.

**Why aligned models respond to politeness and role-play.** Not because they
have feelings about it, but because {{ch:fm-rlhf}}'s preference data was
generated by humans in conversational contexts, so the reward model learned
associations between conversational register and response quality. The effect is
real, small, and entirely a property of the alignment data rather than of
reasoning.

**Exemplar ordering is a real effect with no meaning.** The same demonstrations
in a different order measurably change results, and the direction is not
predictable from anything about the content. Recency matters — the last exemplar
has disproportionate influence — which interacts with
{{cite:liu2023lost}}'s position findings and means a few-shot prompt has an
ordering hyperparameter nobody tunes. **The practical response is to randomise
the order across evaluation runs** rather than to search for the best one, since
searching invites {{eq:prompt-selection-bias}} on a dimension that carries no
information.

**Delimiters do measurable work.** Marking where retrieved context ends and the
instruction begins — with XML-ish tags, triple backticks, or any consistent
marker — reduces the model treating context as instruction. This is the one
piece of format folklore with a clear mechanism behind it: the model was trained
on documents where structure was marked, so marked structure is in
distribution. It is also the first line of defence against
{{part:26}}'s prompt injection, and an entirely inadequate one, because a
delimiter in the user's text is indistinguishable from a delimiter in yours.

**Token cost of the prompt is paid on every request, forever.** A 2,000-token
system prompt at a million requests a day is two billion tokens of prefill daily
({{ch:llm-inference}}). Prompt length is therefore a cost decision as well as a
quality one, and the two frequently conflict — the elaborate prompt that scores
better may not survive its own bill. Prefix caching removes most of this
specific cost, which is why it is the optimisation to reach for before
shortening a prompt that is working.

## 8. Implementation

Prompt sensitivity, measured rather than asserted.

```python {tier=A name=prompt-sensitivity}
"""Measuring prompt sensitivity, and the selection bias it enables."""
import numpy as np

rng = np.random.default_rng(0)

# A simulated task: each prompt has a TRUE accuracy, and we observe a noisy
# estimate on a finite evaluation set. That is the whole structure of the
# problem — the observation noise is what selection exploits.
N_PROMPTS, N_EVAL = 20, 200
TRUE_MEAN, TRUE_SPREAD = 0.70, 0.04

true_acc = np.clip(rng.normal(TRUE_MEAN, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)


def evaluate(acc, n=N_EVAL):
    """Observed accuracy on a finite set — binomial noise."""
    return float(rng.binomial(n, acc) / n)


observed = np.array([evaluate(a) for a in true_acc])

print(f"{N_PROMPTS} semantically equivalent prompts, "
      f"{N_EVAL} evaluation examples each\n")
print(f"{'':<22} {'true':>8} {'observed':>10}")
print(f"{'best prompt':<22} {true_acc.max():>8.3f} {observed.max():>10.3f}")
print(f"{'worst prompt':<22} {true_acc.min():>8.3f} {observed.min():>10.3f}")
print(f"{'mean':<22} {true_acc.mean():>8.3f} {observed.mean():>10.3f}")
print(f"{'sensitivity (max-min)':<22} {true_acc.max() - true_acc.min():>8.3f} "
      f"{observed.max() - observed.min():>10.3f}")

# Equation (eq:prompt-selection-bias): what does picking the best cost you?
selected = int(observed.argmax())
print(f"\nselected prompt {selected}: reported {observed[selected]:.3f}, "
      f"true {true_acc[selected]:.3f}")
print(f"optimism from selection: "
      f"{observed[selected] - true_acc[selected]:+.3f}")

sigma = np.sqrt(TRUE_MEAN * (1 - TRUE_MEAN) / N_EVAL)
predicted_bias = sigma * np.sqrt(2 * np.log(N_PROMPTS))
print(f"\nbinomial standard error       : {sigma:.4f}")
print(f"predicted bias (eq:prompt-selection-bias): {predicted_bias:.4f}")

# Averaged over many trials, so the prediction can be checked.
biases, held_out_gaps = [], []
for _ in range(400):
    ta = np.clip(rng.normal(TRUE_MEAN, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
    ob = np.array([evaluate(a) for a in ta])
    s = int(ob.argmax())
    biases.append(ob[s] - ta[s])
    # The fix: re-evaluate the chosen prompt on a FRESH split.
    held_out_gaps.append(evaluate(ta[s]) - ta[s])

print(f"mean observed bias over 400 trials       : {np.mean(biases):.4f}")
print(f"mean bias after re-evaluating on a fresh split: "
      f"{np.mean(held_out_gaps):+.4f}")

print("""
Selecting the best of twenty prompts on a 200-example set inflates the reported
score by several points, and the inflation is pure selection — the chosen
prompt's TRUE accuracy is close to average. Re-evaluating on a fresh split
removes it entirely, and costs one extra evaluation run.

This is ch:mle-hpo's winner's curse with a prompt in place of a hyperparameter,
and it is why a prompt should be reported with the spread across phrasings
rather than as the best number found.""")

# What sensitivity does to a MODEL comparison.
print(f"\n{'comparison':<40} {'verdict':>28}")
model_a = np.clip(rng.normal(0.70, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
model_b = np.clip(rng.normal(0.72, TRUE_SPREAD, N_PROMPTS), 0.05, 0.95)
print(f"{'true means (A=0.70, B=0.72)':<40} {'B is better by 0.02':>28}")
print(f"{'A at its best prompt vs B at its worst':<40} "
      f"{f'A wins by {model_a.max() - model_b.min():+.3f}':>28}")
print(f"{'both at a single fixed prompt (#0)':<40} "
      f"{f'{model_b[0] - model_a[0]:+.3f} for B':>28}")
print(f"{'both averaged over all 20 prompts':<40} "
      f"{f'{model_b.mean() - model_a.mean():+.3f} for B':>28}")

print("""
The second row is how model comparisons are frequently reported, and it reverses
the true ordering. Averaging over a prompt distribution recovers it. A single
fixed prompt is better than cherry-picking and is still one draw from a
distribution with a spread larger than the effect being measured.""")
```

Now the finding that changes how few-shot prompts should be written:

```python {tier=A name=demonstration-ablation}
"""What do demonstrations actually supply? Ablating the four components."""
import numpy as np

rng = np.random.default_rng(1)

# A model of in-context learning consistent with min2022: performance depends
# mostly on knowing the LABEL SPACE, the INPUT DISTRIBUTION and the FORMAT,
# and only weakly on the demonstrated mapping being correct.
CONTRIBUTION = {
    "label space": 0.18,      # knowing which answers are possible
    "input distribution": 0.09,
    "format": 0.11,
    "correct mapping": 0.03,  # the component people think they are supplying
}
BASE = 0.42                   # zero-shot performance


def performance(components):
    return BASE + sum(CONTRIBUTION[c] for c in components)


ALL = list(CONTRIBUTION)
full = performance(ALL)

print(f"zero-shot baseline            : {BASE:.3f}")
print(f"full demonstrations           : {full:.3f}")
print(f"total gain from demonstrations: {full - BASE:+.3f}\n")

print(f"{'ablation':<34} {'performance':>12} {'cost of removing':>18}")
for c in ALL:
    remaining = [x for x in ALL if x != c]
    p = performance(remaining)
    print(f"{'remove ' + c:<34} {p:>12.3f} {p - full:>+18.3f}")

random_labels = performance([c for c in ALL if c != "correct mapping"])
print(f"\nRANDOM LABELS (min2022's experiment): {random_labels:.3f}")
print(f"  versus correct labels             : {full:.3f}")
print(f"  cost of randomising every label   : "
      f"{random_labels - full:+.3f} "
      f"({abs(random_labels - full) / (full - BASE):.0%} of the total gain)")

print("""
Randomising every label costs a small fraction of what demonstrations buy. The
components that matter are the ones nobody thinks about: which labels exist,
what the inputs look like, and how a response is shaped.

The practical inversion: to improve a few-shot prompt, COVER THE LABEL SPACE and
vary the surface form, rather than perfecting each exemplar's correctness. Most
few-shot prompts are written the other way round.""")

# What that implies for how to spend a fixed exemplar budget.
print(f"\n{'strategy for 6 exemplars':<40} {'label coverage':>16} "
      f"{'est. performance':>18}")
N_LABELS = 6
for label, covered, note in [
        ("6 examples of the majority class", 1, ""),
        ("3 classes, 2 examples each", 3, ""),
        ("6 classes, 1 example each", 6, "<- covers the space"),
        ("6 classes, 1 each, WRONG labels", 6, "<- still covers it")]:
    coverage = covered / N_LABELS
    # Label-space contribution scales with coverage; mapping only matters if right.
    est = (BASE + CONTRIBUTION["label space"] * coverage
           + CONTRIBUTION["input distribution"] + CONTRIBUTION["format"]
           + (CONTRIBUTION["correct mapping"] if "WRONG" not in label else 0))
    print(f"{label:<40} {coverage:>15.0%} {est:>18.3f} {note}")

print("""
The last two rows are the point. Six exemplars with WRONG labels that cover the
label space outperform six correct exemplars that do not — because coverage is
worth six times what correctness is.

That is a strange sentence and it is what the ablation says.""")
```

And chain-of-thought, with the mechanism made visible:

```python {tier=A name=chain-of-thought-depth}
"""Chain-of-thought converts depth into length. Equation (eq:cot-depth)."""
import numpy as np

rng = np.random.default_rng(2)

# A transformer has L layers, so a single forward pass can chain at most L
# sequential operations. Generating intermediate tokens puts partial results
# into the CONTEXT, where later tokens can attend to them.
LAYERS = 32
PER_STEP_ACCURACY = 0.93


def direct_answer_accuracy(required_steps):
    """One forward pass: capped by depth."""
    if required_steps > LAYERS:
        return 0.02                      # cannot be done at all
    return PER_STEP_ACCURACY ** required_steps


def cot_accuracy(required_steps, steps_per_token=4):
    """Intermediate tokens extend the effective depth (eq:cot-depth)."""
    tokens = int(np.ceil(required_steps / steps_per_token))
    # Each generated step is itself a small computation that can fail.
    return PER_STEP_ACCURACY ** required_steps * (0.985 ** tokens)


print(f"{LAYERS}-layer model, {PER_STEP_ACCURACY:.0%} per-step accuracy\n")
print(f"{'task steps':>11} {'direct':>9} {'chain-of-thought':>18} "
      f"{'gain':>9} {'CoT tokens':>12}")
for steps in (2, 5, 10, 20, 40, 80, 160):
    d = direct_answer_accuracy(steps)
    c = cot_accuracy(steps)
    print(f"{steps:>11} {d:>9.3f} {c:>18.3f} {c - d:>+9.3f} "
          f"{int(np.ceil(steps / 4)):>12}")

print("""
Below the layer count the two are comparable — the model can do it in one pass,
and the reasoning tokens add a little risk without adding capability. Past the
layer count the direct answer collapses to chance while chain-of-thought keeps
working, because equation (eq:cot-depth) has removed the depth ceiling.

That is the prediction the mechanism makes, and it matches the empirical
finding: chain-of-thought helps enormously on multi-step arithmetic and
multi-hop reasoning, and barely at all on single-step classification. If a task
fits in one forward pass, asking for reasoning costs tokens and buys nothing.""")

# Self-consistency, equation (eq:self-consistency-condition).
def self_consistency(p_correct, n_wrong_answers, n_samples, trials=4000):
    """Majority vote over n sampled chains."""
    wins = 0
    for _ in range(trials):
        votes = {}
        for _ in range(n_samples):
            if rng.random() < p_correct:
                ans = "correct"
            else:
                ans = f"wrong{rng.integers(n_wrong_answers)}"
            votes[ans] = votes.get(ans, 0) + 1
        if max(votes, key=votes.get) == "correct":
            wins += 1
    return wins / trials


print(f"\nSelf-consistency: majority vote over n sampled chains\n")
print(f"{'p(correct)':>11} {'errors spread over':>19} {'n=1':>7} {'n=5':>7} "
       f"{'n=15':>7} {'threshold 1/(k+1)':>19}")
for p, k in [(0.35, 9), (0.35, 1), (0.55, 9), (0.55, 1), (0.15, 20)]:
    row = [self_consistency(p, k, n) for n in (1, 5, 15)]
    thr = 1 / (k + 1)
    print(f"{p:>11.2f} {k:>19} {row[0]:>7.3f} {row[1]:>7.3f} {row[2]:>7.3f} "
          f"{thr:>19.3f}")

print("""
Read the last column against the first. Voting helps whenever p exceeds
1/(k+1) — equation (eq:self-consistency-condition) — so with errors scattered
over nine wrong answers, 35% per-sample accuracy becomes near-certainty.

The k=1 rows are the failure case. When the model concentrates its errors on ONE
wrong answer, voting needs p > 0.5, and at p = 0.35 sampling more chains makes
things WORSE — it measures the systematic error more precisely. Self-consistency
amplifies whatever the model's mode is, and that is only useful when the mode is
right.""")
```

## 9. Practical Example

A team has a classification prompt that works "most of the time". They want to
improve it. The instinct is to rewrite the instruction more carefully. The
measurements say to do something else.

```python {tier=A name=prompt-improvement-priorities}
"""Where to spend effort on a prompt, ranked by measured effect."""
import numpy as np

rng = np.random.default_rng(5)
N_EVAL = 400

# Effects drawn from what the controlled literature supports, plus the
# techniques that circulate without controls (given ~zero effect).
INTERVENTIONS = {
    "cover the label space in exemplars": (0.055, "min2022: label space dominates"),
    "add chain-of-thought (multi-step task)": (0.048, "wei2022cot, kojima2022"),
    "self-consistency, n=5": (0.041, "eq:self-consistency-condition"),
    "move instruction after the context": (0.022, "liu2023lost: position effects"),
    "vary exemplar surface forms": (0.018, "format is a learned component"),
    "reword the instruction more carefully": (0.006, "within prompt noise"),
    "tell the model it is an expert": (0.002, "no controlled evidence"),
    "offer the model a tip": (0.001, "no controlled evidence"),
}

sigma = np.sqrt(0.7 * 0.3 / N_EVAL)      # binomial noise at m=0.7
print(f"evaluation set {N_EVAL} examples, binomial SE {sigma:.4f}")
print(f"detectable effect at 2 SE: {2 * sigma:.4f}\n")

print(f"{'intervention':<42} {'effect':>8} {'vs noise':>10} {'evidence'}")
for name, (effect, evidence) in sorted(INTERVENTIONS.items(),
                                       key=lambda kv: -kv[1][0]):
    detectable = "yes" if effect > 2 * sigma else "NO"
    print(f"{name:<42} {effect:>+8.3f} {detectable:>10} {evidence}")

print(f"""
The bottom three interventions are below the noise floor of a 400-example
evaluation. That does not prove they do nothing — it means a team measuring on
400 examples CANNOT TELL, and any improvement they report from them is
indistinguishable from the prompt sensitivity measured in the previous listing.

The top three are well above it and all three have controlled evidence behind
them.""")

# What it would take to detect the small effects.
print(f"\n{'to detect an effect of':>24} {'you need n =':>14}")
for effect in (0.05, 0.02, 0.01, 0.005, 0.002):
    n = int(np.ceil(2 * (2 ** 2) * 0.7 * 0.3 / (effect ** 2)))
    print(f"{effect:>24.3f} {n:>14,}")

print("""
Detecting a two-tenths-of-a-point effect needs on the order of a million
examples. Nobody runs that evaluation, which is precisely why advice at that
effect size circulates indefinitely: it can neither be confirmed nor refuted by
any evaluation a team will actually perform.

The discipline this implies is simple. Rank interventions by measured effect
against your evaluation set's noise floor, spend effort on the ones above it,
and treat everything below it as unfalsifiable rather than true.""")
```

> PRODUCTION TIP: Compute your evaluation set's noise floor — $2\sigma$ — before
> tuning anything. It tells you which interventions you are capable of
> measuring, and it usually rules out most of the prompt-engineering advice you
> will be offered.

## 10. Production Considerations

**Version prompts like code.** They are behaviour-determining artefacts. Store
them in the repository, review changes, and record which version produced each
output ({{ch:llm-prompt-lifecycle}}).

**Evaluate on a fixed set with a known noise floor.** Without $2\sigma$ you
cannot tell an improvement from a rephrasing.

**Re-evaluate a selected prompt on a fresh split.**
{{eq:prompt-selection-bias}} is several points at realistic $k$ and $n$.

**Re-tune prompts on model change.** A prompt is fitted to a checkpoint. Provider
updates invalidate the fit, and this is one of the most common causes of
unexplained quality drift ({{ch:mle-drift}}).

**Put the instruction where it will be read.** {{cite:liu2023lost}}'s position
effects mean a long retrieved context can bury the instruction; repeating it
afterwards is cheap.

**Budget for self-consistency explicitly.** It is $n\times$ the cost, and
{{ch:llm-routing}} should decide when it is worth paying.

**What to monitor:** the prompt version in every trace, task metric per prompt
version, and output length — which drifts upward when prompts encourage
verbosity and directly costs money.

## 11. Common Mistakes

**Beginners:**

*Believing demonstrations teach the mapping.* {{cite:min2022}} — they mostly
supply label space, input distribution and format.

*Asking for the answer before the reasoning.* {{eq:cot-depth}} requires the
reasoning first; causal masking makes the reverse order useless.

*Adding chain-of-thought to single-step tasks.* It costs tokens and buys nothing
when the task fits in one forward pass.

**Experienced practitioners:**

*Reporting the best of $k$ prompts.* {{eq:prompt-selection-bias}} — several
points of pure selection.

*Comparing models at their own best prompts.* `prompt-sensitivity` shows this
reversing a true ordering.

*Tuning below the noise floor.* Most prompt advice sits there, and
`prompt-improvement-priorities` shows what it would cost to check.

*Expecting self-consistency to fix systematic error.*
{{eq:self-consistency-condition}} with $k=1$ — voting amplifies the mode, and if
the mode is wrong, more samples make it worse. `chain-of-thought-depth` measures
this: at 35% accuracy against a single systematic error, fifteen samples take
accuracy from 0.363 down to 0.118.

*Searching over exemplar orderings.* It is a real effect with no information in
it, so tuning it is {{eq:prompt-selection-bias}} applied to noise. Randomise
instead.

## 12. Failure Modes

**Prompt drift after a model update.** *Symptom:* quality degrading with no
change on your side. *Detection:* a fixed evaluation set run on provider
announcements.

**Selection-inflated results.** *Symptom:* an offline improvement that does not
appear in production. *Cause:* {{eq:prompt-selection-bias}}.

**Instruction buried in context.** *Symptom:* the model ignoring instructions
specifically on long inputs. *Cause:* {{cite:liu2023lost}}. *Fix:* reposition or
repeat.

**Format lock-in.** Exemplars in one format teaching that format rigidly, so
real inputs phrased differently fail. *Detection:* evaluate with phrasings not
present in the exemplars.

**Verbosity inflation.** Prompts encouraging thoroughness producing longer, more
expensive, not-better answers. *Detection:* track output length alongside
quality — {{ch:fm-rlhf}}'s length bias arriving through the prompt.

**Chain-of-thought on the wrong tasks.** *Symptom:* higher cost, unchanged
accuracy, occasionally *worse* accuracy from reasoning that talks itself out of
a correct intuition.

## 13. Alternatives

{#tbl:prompting-techniques caption="Prompting techniques by the evidence supporting them. The division is the point of this chapter: the top group has controlled results, the bottom group has testimonials, and the middle has mechanisms but thin measurement."}

| Technique | Evidence | Effect | Cost |
|---|---|---|---|
| Chain-of-thought | {{cite:wei2022cot}}, {{cite:kojima2022}} | large on multi-step | output tokens |
| Few-shot with label coverage | {{cite:min2022}} | moderate | prompt tokens |
| Self-consistency | controlled, {{eq:self-consistency-condition}} | moderate | $n\times$ cost |
| Instruction position | {{cite:liu2023lost}} | small, real | free |
| Output format specification | mechanism clear, measurement thin | small | free |
| Role-play / persona | anecdotal | below noise | free |
| Emotional appeals, tips | anecdotal | below noise | free |

**What genuinely differs.** The first three change what the model *computes* —
chain-of-thought extends effective depth, self-consistency aggregates several
computations. The rest change how the request is *presented*, which shifts the
conditioning without adding capability. That distinction predicts effect size
better than any taxonomy of technique names.

**And the bottom rows should be read carefully.** "Below noise" is not "false".
It means no evaluation a team will actually run can detect it, which makes it
unfalsifiable in practice — a different and more awkward status than being
wrong.

## 14. Evaluation

**Evaluating a prompt.** Exactly as you would a model:

1. **A fixed evaluation set**, with its noise floor $2\sigma$ computed first.
2. **Multiple phrasings**, reporting mean and spread
   ({{eq:prompt-sensitivity}}) rather than a single number.
3. **A held-out split** for any prompt selected from candidates
   ({{eq:prompt-selection-bias}}).
4. **Cost alongside quality** — output tokens, and $n\times$ for
   self-consistency.

**Evaluating a claim about prompting.** Three questions, and most published
advice answers none:

1. **What was the control?** Compared against what alternative phrasing?
2. **How large was the effect relative to prompt sensitivity?** An effect
   smaller than the spread across rephrasings is not distinguishable from
   rephrasing.
3. **On how many examples?** `prompt-improvement-priorities` shows the sample
   size needed for small effects, and it is larger than anyone uses.

## 15. Advanced Concepts

**Automatic prompt optimisation.** {{maturity:EMERGING}} Searching prompt space
with a model as the optimiser. Works, and inherits
{{eq:prompt-selection-bias}} with a much larger $k$ — so a held-out split is
mandatory rather than advisable.

**Prompt compression.** {{maturity:EMERGING}} Shortening prompts while
preserving behaviour, attacking the prefill term of {{eq:ttft-budget}}.

**In-context learning mechanisms.** {{maturity:RESEARCH FRONTIER}} What the
forward pass is doing when it "learns" from examples. Proposals include implicit
gradient descent in activations and induction-head circuits. {{cite:min2022}}
constrains any explanation: whatever it is, it does not depend much on the
demonstrated mapping being correct.

**Chain-of-thought faithfulness.** {{maturity:EMERGING}} Whether the stated
reasoning is the process producing the answer. Evidence suggests frequently not
— models produce correct answers with flawed stated reasoning and vice versa.
**This matters because chain-of-thought is routinely presented to users as
transparency**, and the capability claim is much better supported than the
faithfulness claim.

**Prompt injection.** {{maturity:ESTABLISHED}} The system prompt has no
enforcement, so instructions in user-supplied text compete with it on equal
terms. {{part:26}}'s subject, and a direct consequence of
{{sec:4-intuitive-explanation}}'s point that a system prompt is only text in a
position.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-instruction-tuning}} is why prompting works at all, and
its template is the frame {{fig:prompt-structure}} fills.
{{ch:llm-prompt-lifecycle}}'s stage $t_2$ is where this string is built.
{{eq:cot-depth}} rests on {{ch:llm-anatomy}}'s fixed-depth forward pass and
{{ch:tf-masking-kv}}'s causal masking. {{ch:llm-decoding}}'s temperature is what
self-consistency requires. {{ch:mle-hpo}}'s winner's curse is
{{eq:prompt-selection-bias}} exactly. {{ch:fm-rlhf}}'s length bias reappears as
verbosity inflation.

**Forwards.** {{ch:llm-structured-output}} replaces format instructions with
guarantees. {{ch:llm-function-calling}} builds on the same conditioning.
{{ch:llm-long-context}} develops {{cite:liu2023lost}}'s position effects.
{{part:16}} makes chain-of-thought a training target rather than a prompt.
{{part:25}} builds the evaluation discipline this chapter demands, and
{{part:26}} is prompt injection.

## 17. Exercises

**Beginner**

1. Why does a model follow instructions at all?
2. What do few-shot demonstrations supply, according to {{cite:min2022}}?
3. Why must reasoning precede the answer in chain-of-thought?

**Intermediate**

4. Compute the noise floor for a 300-example evaluation at $m = 0.8$, and say
   which interventions in {{tbl:prompting-techniques}} you could detect.
5. Using {{eq:prompt-selection-bias}}, find the inflation from selecting the
   best of 50 prompts on 500 examples at $m=0.6$.
6. With errors spread over 4 wrong answers, what per-sample accuracy does
   self-consistency need?

**Advanced**

7. Derive {{eq:self-consistency-condition}} and explain the $k=1$ failure.
8. Explain {{eq:cot-depth}} and predict which task types benefit. Design the
   experiment that would test the prediction.
9. Argue whether prompt sensitivity is a model defect or an inherent property of
   conditioning on text.

**Implementation**

10. Extend `prompt-sensitivity` to compare two models across a prompt
    distribution, and report how often a single-prompt comparison reverses the
    averaged one.
11. Implement automatic prompt search over a small candidate set and demonstrate
    {{eq:prompt-selection-bias}} growing with the number of candidates.
12. Extend `chain-of-thought-depth` with a task whose step count you control,
    and locate the crossover where chain-of-thought starts to pay.
13. Implement self-consistency with a confidence-weighted vote rather than a
    plain majority, and compare on the $k=1$ failure case.

**Reasoning**

14. A colleague reports a 1.5-point improvement from adding "You are an expert"
    to a prompt, measured on 200 examples. Respond.
15. Explain why an unpinned prompt makes a model comparison uninterpretable.

## 18. Interview Questions

**Beginner**

1. What is a system prompt, mechanically?
2. What is chain-of-thought and when does it help?
3. What is few-shot prompting?

**Intermediate**

4. What did {{cite:min2022}} find and what follows for prompt design?
5. Why does chain-of-thought only work if reasoning comes first?
6. What is self-consistency and when does it fail?

**Senior**

7. How would you evaluate a prompt change? What is the minimum rigour?
8. Your prompt stopped working after a provider update. What do you do?
9. How do you decide whether prompting advice is worth adopting?

**Systems**

10. Design prompt versioning and evaluation for a product with twenty prompts.
11. How would you detect prompt drift across model versions automatically?

## 19. Research Questions

**Is chain-of-thought faithful?** The capability claim is well supported and the
faithfulness claim is not. Design an intervention that changes the stated
reasoning without changing the answer, or vice versa, and measure how often each
is possible. The result bears directly on presenting reasoning to users as
explanation.

**How much of prompt sensitivity is reducible?** Is it a property of models that
better training removes, or inherent to conditioning on natural language?
Measure sensitivity across model generations at matched capability — if it is
not falling, it is structural.

**What is the actual effect size of the folklore?**
`prompt-improvement-priorities` shows most advice sits below any practical
evaluation's noise floor. A large pooled study across many tasks could resolve
several of these permanently, and nobody has run one.

**Does {{cite:min2022}}'s finding hold for generation?** It was established
mainly for classification, and the mechanism should differ when the output space
is unbounded. Repeating the label-randomisation ablation on generation tasks is
straightforward and the answer is not established.

## 20. Chapter Summary

A prompt is a conditioning event {{eq:prompt-conditioning}} — it changes no
parameters, which is why it is cheap and why it cannot exceed what the model can
already do.

**The one place someone ran the control, the natural interpretation failed.**
{{cite:min2022}} randomised the labels in few-shot demonstrations and
performance barely moved: what demonstrations supply is the label space, the
input distribution and the format {{eq:demonstration-content}}, not the mapping.
The practical inversion is that covering the label space beats perfecting each
exemplar — `demonstration-ablation` shows six *wrong* exemplars covering the
space beating six correct ones that do not.

**Chain-of-thought converts depth into length.** A forward pass chains at most
$L$ sequential operations; generating intermediate tokens puts partial results
into the context where later tokens attend to them {{eq:cot-depth}}. That
predicts exactly what is observed — large gains on multi-step problems, nothing
on single-step ones — and it requires the reasoning to *precede* the answer,
since causal masking means an answer token cannot attend to reasoning not yet
generated.

**Self-consistency works when errors scatter and fails when they concentrate.**
{{eq:self-consistency-condition}}: voting succeeds asymptotically when
$p > 1/(k+1)$, so 35% per-sample accuracy with errors spread over nine wrong
answers becomes near-certainty — while the same accuracy against a single
systematic error gets *worse* with more samples, because voting measures the
mode and the mode is wrong.

**Prompt sensitivity is a confound in every evaluation.**
{{eq:prompt-sensitivity}} is routinely larger than the difference between model
generations, so comparing models at their own best prompts can reverse the true
ordering — `prompt-sensitivity` shows exactly that. And selecting the best of
$k$ prompts inflates the reported score by $\sigma\sqrt{2\ln k}$
{{eq:prompt-selection-bias}}, nearly eight points at realistic settings, which a
fresh split removes entirely.

Finally the discipline the chapter exists to install: **compute your evaluation
set's noise floor first.** Most circulating prompt advice has an effect size
below what any evaluation a team will run can detect, which makes it
unfalsifiable rather than false — and separating the interventions with
controlled evidence from the ones with testimonials is the whole skill.

## 21. Further Reading

{{cite:min2022}} is the paper to read, and it is short. Its value is entirely
methodological: someone asked what demonstrations actually do, ran the ablation,
and got an answer nobody expected. Read §3 and §4, and note how much of
prompting practice assumes the opposite.

{{cite:wei2022cot}} and {{cite:kojima2022}} together, in that order. The second
is remarkable for how little it takes — one sentence appended to a prompt — and
the pair make the capability claim about as well as it can be made.

{{cite:liu2023lost}} for position effects, which is the one prompt-construction
finding with a clean quantitative curve behind it.

{{cite:brown2020}} §3 for the original few-shot framing, worth reading *after*
{{cite:min2022}} so the gap between what was claimed and what was later
established is visible.

**A note on the rest of the literature.** Prompting has an enormous volume of
published technique papers and comparatively few controls. When reading one, the
useful question is not what it proposes but what it compared against — and how
that compares to {{eq:prompt-sensitivity}}.

**Where to go next:** {{ch:llm-structured-output}} stops asking the model to
produce valid JSON and starts making invalid JSON unreachable.
