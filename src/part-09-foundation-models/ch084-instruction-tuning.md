---
id: fm-instruction-tuning
number: 84
part: IX
tier: full
status: draft
requires: [fm-pretraining, fm-what-they-are, nlp-contextual, dl-losses,
           mle-splits, fm-datasets]
provides: [instruction-tuning, instruction-format, task-mixture, template-diversity,
           held-out-task-generalisation, loss-masking, chat-template,
           demonstration-data, capability-interface-distinction]
citations: [wei2022flan, ouyang2022, brown2020, howard2018, touvron2023llama,
            hinton2015, lee2022dedup, gunasekar2023]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why a base model does not answer questions, in terms of its training
   objective rather than as a curiosity.
2. State the instruction-tuning objective and explain how loss masking makes it
   differ from pretraining.
3. Explain what {{cite:wei2022flan}} established, and why held-out *task type*
   generalisation is the result that matters.
4. Implement a chat template and explain why the format is a contract rather
   than a convention.
5. Explain why instruction data quality dominates quantity, and cite what that
   claim rests on.
6. Distinguish capability from interface, and identify which stage of the
   pipeline supplies each.
7. Design an instruction dataset and evaluate it on the right axis.

## 2. Why This Matters

**This stage is why models are usable at all.**
{{ch:fm-what-they-are}} computed the ratio: instruction tuning is roughly one
part in forty thousand of pretraining compute {{eq:stage-compute-ratio}} and
produces most of the difference a user perceives between a base model and a
product. Almost nothing else in machine learning has that leverage.

**It is the cleanest demonstration in the book that capability and interface are
separate things.** The base model can already do the tasks —
{{cite:brown2020}}'s few-shot results prove that. What it cannot do is
*recognise a request as a request*. Instruction tuning teaches only the second,
and the second is what was missing.

**And the practical guidance is unusually actionable.** Unlike pretraining,
which almost nobody does, instruction tuning is within reach of a small team:
thousands of examples rather than trillions of tokens, GPU-hours rather than
GPU-months. Most teams that fine-tune a model are doing this, whether or not
they call it that — and most of them make the same three mistakes, which
{{sec:11-common-mistakes}} enumerates.

**The format is a contract with sharp edges.** A chat template mismatch between
training and serving produces a model that is subtly, unfixably worse, with no
error anywhere. It is the train/serve skew of {{ch:mle-pipelines}} in a form
that catches people who have never made that mistake before.

## 3. Prerequisites

{{ch:fm-pretraining}} for the base model this stage starts from and for the
causal objective it reuses. {{ch:fm-what-they-are}} for the pipeline and for
{{eq:adaptation-information-ratio}}, which bounds what this stage can achieve.
{{ch:nlp-contextual}} for fine-tuning versus feature-based transfer, and for
catastrophic forgetting. {{ch:dl-losses}} for cross-entropy and masked loss
terms. {{ch:mle-splits}} for held-out evaluation, which acquires a new meaning
here. {{ch:fm-datasets}} for data quality and contamination, both of which
recur at this smaller scale.

## 4. Intuitive Explanation

Ask a base model "What is the capital of France?" and it may answer. It may also
produce:

> What is the capital of Germany? What is the capital of Italy?

or

> — Geography Quiz, Chapter 3. Answers on page 47.

**Neither is a failure.** Both are high-probability continuations of that string
in web text, and the model was trained to produce high-probability continuations.
It has the knowledge; it does not have the behaviour.

The fix is embarrassingly simple. Collect examples of instructions paired with
good responses, and continue training with the same next-token objective on
those pairs. The model learns that text of this shape is followed by text of
that shape.

**What makes it work is not obvious in advance.** You might expect it to teach
the specific tasks in your dataset. {{cite:wei2022flan}}'s finding is stronger:
train on a mixture of many task *types* and performance improves on task types
held out entirely. The model is not learning the tasks — it is learning that an
instruction should be followed, and that transfers.

> NOTE: This is the capability/interface distinction, and it is the single most
> useful idea in the chapter. Instruction tuning does not add knowledge —
> {{eq:adaptation-information-ratio}} says it cannot, at eight orders of
> magnitude below pretraining. It changes which of the model's existing
> behaviours are likely. That is why it is so cheap and so effective at once.

**Two details do most of the work in practice.**

**Loss masking.** During pretraining every token contributes to the loss. Here
you usually want the loss only on the *response*, because you are teaching the
model to produce responses, not to produce instructions. Getting this wrong is a
common and quiet bug.

**Template diversity.** If every training example says "Question: ... Answer:
...", the model learns that exact format rather than the general behaviour.
Varying the phrasing is what makes the learned behaviour robust to how a user
actually writes.

**The mental model:** instruction tuning is a small, sharply targeted
redistribution of probability mass — away from continuing text and toward
answering it. Where it breaks down: because it is small, it can be undone or
overwhelmed. A sufficiently unusual prompt reaches past it to the base model
underneath, which is the mechanism behind much of {{part:26}}.

## 5. Formal Explanation

### 5.1 The objective

An instruction dataset is a set of pairs $(\vec{x}, \vec{y})$ — instruction and
response. The objective is the causal loss of {{ch:fm-pretraining}}, restricted
to the response tokens:

$$
\Loss_{\text{IT}} = -\sum_{(\vec{x},\vec{y})\in\Data}\ \sum_{t=1}^{|\vec{y}|}
 \log P_\theta\big(y_t \given \vec{x},\ y_{<t}\big)
$$ (eq:instruction-tuning-loss)

**Compare with {{eq:clm-loss}}.** The functional form is identical; the sum runs
over response tokens only. The instruction is conditioned on and not predicted.

### 5.2 Loss masking

In implementation, instruction and response are concatenated into one sequence
and a mask selects which positions contribute:

$$
\Loss = -\frac{\sum_{t} m_t \log P_\theta(z_t \given z_{<t})}{\sum_t m_t},
\qquad
m_t = \Ind[\,t \in \text{response}\,]
$$ (eq:loss-masking)

**Whether to mask is a design decision, not a formality.** Masking the
instruction focuses all capacity on generating responses. Not masking — training
on the full sequence — additionally teaches the distribution of instructions,
which is useless if users write the instructions and mildly useful if the model
must ever generate them.

The common failure is masking incorrectly: an off-by-one that includes the last
instruction token or excludes the first response token. Neither raises an error
and both degrade quality slightly, which is the worst kind of bug.

### 5.3 The template as a contract

Models are trained with a specific serialisation — special tokens delimiting
roles, a fixed ordering, a fixed system-message position:

$$
z = \texttt{[SYS]}\,s\,\texttt{[/SYS]}\ \texttt{[INST]}\,x\,\texttt{[/INST]}\ y
$$ (eq:chat-template)

> WARNING: The template used at inference must match the template used in
> training **exactly** — the same special tokens, the same whitespace, the same
> ordering. A mismatch does not error. The model sees a string from a
> distribution it was not tuned on, and behaves like a partially-instruction-tuned
> model: mostly fine, occasionally reverting to continuation. Teams lose days to
> this, and the symptom is "the model got worse after we changed the serving
> code".

### 5.4 What {{cite:wei2022flan}} established

The experimental design is the contribution. Group tasks into *clusters* by type
— sentiment, natural language inference, summarisation, and so on. Train on all
clusters but one. Evaluate on the held-out cluster.

$$
\text{train on } \bigcup_{c \neq c^*} \Data_c,
\qquad
\text{evaluate on } \Data_{c^*}
$$ (eq:held-out-cluster)

**Performance on the held-out cluster improves.** Since no example of that task
type was seen, the model cannot have learned the task; it learned something
transferable about following instructions.

Two secondary findings matter:

- **More task clusters help**, with diminishing returns — the diversity of task
  *types* matters more than the number of examples.
- **The effect requires scale.** At small model sizes instruction tuning can
  *hurt* held-out performance, presumably because capacity spent on format is
  capacity taken from capability. This is a genuine interaction and it is why
  the technique arrived when it did.

### 5.5 Where the data comes from

Four sources, with different economics:

{#tbl:instruction-data-sources caption="Sources of instruction data. The first is the highest quality and least scalable; the third dominates open practice and inherits the teacher's flaws along with its strengths."}

| Source | Cost | Quality | Scale | Main risk |
|---|---|---|---|---|
| Human-written demonstrations | very high | highest | thousands | cost, annotator drift |
| Converted existing datasets | low | mixed | millions | templated, unnatural phrasing |
| Distilled from a stronger model | low | good | millions | inherits teacher errors; licensing |
| Self-generated with filtering | very low | variable | unbounded | drift, narrowing |

{{cite:ouyang2022}} used human demonstrations for its first stage.
{{cite:wei2022flan}} converted existing datasets with templates. Most open
instruction-tuned models use the third, which is {{cite:hinton2015}}'s
distillation applied to behaviour rather than logits — and which
{{ch:fm-distillation}} treats properly.

## 6. Mathematical Foundation

### 6.1 Why the format is learnable from few examples

Instruction tuning changes behaviour with $10^3$–$10^5$ examples, against
pretraining's $10^{12}$ tokens. {{eq:adaptation-information-ratio}} says this
cannot install capability. Why does it change behaviour so reliably?

Consider the model's distribution over continuations of a prompt $\vec{x}$.
Pretraining leaves it as $P_{\text{base}}(\cdot\given\vec{x})$, a mixture over
the ways such text continues in the corpus:

$$
P_{\text{base}}(\vec{y}\given\vec{x})
 = \sum_{m} \pi_m\, P_m(\vec{y}\given\vec{x})
$$ (eq:continuation-mixture)

where $m$ indexes modes — answering, continuing with more questions, quoting a
page footer. **All the modes already exist**; what instruction tuning changes is
$\pi$.

Re-weighting a mixture requires far less information than learning its
components. Formally, if the answering mode $P_{m^*}$ is already well modelled,
moving $\pi_{m^*}$ from 0.1 to 0.9 is a low-dimensional change — which is why it
is achievable with a dataset eight orders of magnitude smaller than pretraining.

$\square$

**This also predicts the failure mode.** If a mode is not present in the base —
a capability the model does not have — no re-weighting produces it, which is
{{eq:adaptation-information-ratio}} again from a different direction.

### 6.2 The effect of loss masking on gradient allocation

Let a sequence have $n_x$ instruction tokens and $n_y$ response tokens. Without
masking, the fraction of gradient signal spent on modelling instructions is

$$
\frac{n_x}{n_x + n_y}
$$ (eq:unmasked-gradient-share)

For a typical instruction dataset with $n_x \approx n_y$, that is about half.
**Masking doubles the effective learning rate on the thing you care about**, at
no cost, which is why it is standard.

The exception is when instructions are highly templated and responses are long:
then $n_x/(n_x+n_y)$ is small and masking matters less. Measuring the ratio in
your own data tells you how much the decision is worth.

### 6.3 Why template diversity is necessary

Suppose all training examples use template $T$. The model learns
$P(\vec{y}\given T(\vec{x}))$. At inference a user writes $T'(\vec{x})$, and
the model's behaviour depends on how close $T'$ is to $T$ under its
representation.

Training on $k$ distinct templates $\{T_1,\dots,T_k\}$ instead teaches

$$
P\big(\vec{y}\given T_i(\vec{x})\big) \approx P\big(\vec{y}\given T_j(\vec{x})\big)
 \quad \forall i,j
$$ (eq:template-invariance)

which is an *invariance* rather than a mapping. The model learns that the
response should not depend on the phrasing — and an invariance learned from $k$
examples generalises to a $(k{+}1)$th phrasing in a way a single mapping does
not.

$\square$

This is the same argument as data augmentation in {{ch:dl-regularization}}:
present the same content under many surface forms and the model learns the
content rather than the form.

### 6.4 A worked accounting

A dataset of 50,000 instruction pairs, average 60 instruction tokens and 180
response tokens, on a 7B model:

$$
\text{tokens} = 50{,}000 \times 240 = 1.2\times10^{7}
$$

$$
C = 6ND = 6\times 7\times10^9\times 1.2\times10^7 = 5.0\times10^{17}\ \text{FLOPs}
$$

Against pretraining at $8.4\times10^{22}$ ({{ch:fm-what-they-are}}), this is
$6\times10^{-6}$ — six parts per million. On one accelerator at $10^{15}$ FLOPs
and 45% utilisation, about **19 minutes.**

**Masked, the useful fraction is $180/240 = 75\%$**, so 25% of even that tiny
budget would have gone to modelling instructions.

## 7. Internal Mechanics

```mermaid {#fig:instruction-tuning caption="Instruction tuning. The objective and the model are unchanged from pretraining; what changes is the data and the loss mask. The instruction is conditioned on, the response is predicted, and the boundary between them is where the common bugs live."}
graph TD
  A["base model<br/>from ch:fm-pretraining"] --> B["serialise with the chat template<br/>eq:chat-template"]
  C["instruction pairs<br/>(x, y)"] --> B
  B --> D["tokenize the FULL sequence"]
  D --> E["build the loss mask<br/>m_t = 1 on response tokens only"]
  E --> F["causal LM loss, masked<br/>eq:loss-masking"]
  F --> G["instruction-tuned model"]
  G -.->|"template must match<br/>EXACTLY at serving"| H["inference"]
  style E fill:#fde,stroke:#c69
  style H fill:#dfe,stroke:#5a5
```

**Why the mask boundary is error-prone.** The template inserts special tokens
between instruction and response, and whether those tokens belong to the
instruction or the response is a judgement. The response's *first* token must be
predicted — that is where the model decides to start answering — so the mask
must begin at or before it. Excluding it teaches the model everything about
continuing an answer and nothing about starting one.

**Catastrophic forgetting is real at this scale but small.** Training on
$10^7$ tokens after $10^{12}$ moves the weights very little, so wholesale
forgetting is not the usual outcome. The observed effect is narrower: the model
becomes worse at *continuation-style* tasks, because that behaviour is precisely
what was down-weighted. Whether that is forgetting or the intended effect is a
matter of what you wanted.

**Mixing in pretraining data.** {{cite:ouyang2022}} mixes pretraining gradients
into the alignment stage to limit capability regression. The same trick applies
here, and it is cheap: a small fraction of pretraining batches interleaved with
instruction batches measurably reduces the regression at negligible cost.

**Multi-turn conversations** are the same objective with a mask that is 1 on
every assistant turn and 0 on every user turn — one sequence, several masked
regions. This is where mask bugs multiply, because there are now many
boundaries rather than one.

## 8. Implementation

The masking machinery, which is where the bugs are, verified rather than
assumed.

```python {tier=A name=instruction-masking}
"""Chat template serialisation and loss masking, with the boundaries checked."""

IGNORE = -100          # the conventional "do not compute loss here" label

SPECIAL = {"<|system|>": 0, "<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}
WORDS = ["you", "are", "helpful", "what", "is", "the", "capital", "of",
         "france", "paris", "italy", "rome", "please", "explain", "briefly"]
VOCAB = list(SPECIAL) + WORDS
idx = {w: i for i, w in enumerate(VOCAB)}


def tokenize(text):
    return [idx[w] for w in text.split()]


def build_example(system, turns):
    """Serialise a conversation and build the loss mask in one pass.

    turns is a list of (user, assistant) pairs. The mask is 1 on assistant
    tokens INCLUDING the token that begins the response — that is where the
    model decides to start answering — and 0 everywhere else.
    """
    tokens, labels = [], []

    def emit(chunk, supervised):
        for t in chunk:
            tokens.append(t)
            labels.append(t if supervised else IGNORE)

    emit([idx["<|system|>"]], False)
    emit(tokenize(system), False)
    emit([idx["<|end|>"]], False)

    for user, assistant in turns:
        emit([idx["<|user|>"]], False)
        emit(tokenize(user), False)
        emit([idx["<|end|>"]], False)
        # The assistant marker is NOT supervised — the template supplies it —
        # but everything from the first content token onward is.
        emit([idx["<|assistant|>"]], False)
        emit(tokenize(assistant), True)
        emit([idx["<|end|>"]], True)      # the model must learn to stop

    return tokens, labels


tokens, labels = build_example(
    "you are helpful",
    [("what is the capital of france", "paris"),
     ("what is the capital of italy", "rome")])

print(f"{'pos':>4} {'token':<15} {'label':<15} supervised")
for i, (t, l) in enumerate(zip(tokens, labels)):
    lab = "IGNORE" if l == IGNORE else VOCAB[l]
    print(f"{i:>4} {VOCAB[t]:<15} {lab:<15} {'yes' if l != IGNORE else ''}")

n_sup = sum(1 for l in labels if l != IGNORE)
print(f"\nsupervised {n_sup} of {len(labels)} positions "
      f"({n_sup / len(labels):.0%})")

# The checks that catch the two classic bugs.
first_response = tokens.index(idx["<|assistant|>"]) + 1
assert labels[first_response] != IGNORE, \
    "the first response token MUST be supervised — it is where answering starts"
assert labels[tokens.index(idx["<|user|>"]) + 1] == IGNORE, \
    "user content must NOT be supervised"

# Every assistant turn must be supervised, not just the first — the multi-turn
# bug is a mask that stops after the first response.
assistant_starts = [i for i, t in enumerate(tokens) if t == idx["<|assistant|>"]]
print(f"assistant turns: {len(assistant_starts)}")
for start in assistant_starts:
    assert labels[start + 1] != IGNORE, f"turn at {start} is not supervised"
print("all assistant turns supervised — the multi-turn mask bug would fail here")

# Equation (eq:unmasked-gradient-share): what masking is worth on this data.
n_instruction = sum(1 for l in labels if l == IGNORE)
print(f"\nwithout masking, {n_instruction / len(labels):.0%} of the gradient "
      f"would model INSTRUCTIONS rather than responses")
print(f"masking raises the useful share from "
      f"{n_sup / len(labels):.0%} to 100%")
print("(This toy has one-word answers, so the ratio is extreme. Real "
      "instruction data runs nearer half — measure it on your own data with "
      "equation eq:unmasked-gradient-share before deciding how much masking "
      "is worth.)")
```

Now the template-mismatch failure, which is the one that costs days:

```python {tier=A name=template-mismatch}
"""What a serving/training template mismatch actually does."""

TRAIN_TEMPLATE = "<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

VARIANTS = {
    "exact match":        "<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
    "missing newline":    "<|user|>{instruction}<|end|>\n<|assistant|>\n",
    "trailing space":     "<|user|>\n{instruction}<|end|>\n<|assistant|> ",
    "different marker":   "<|human|>\n{instruction}<|end|>\n<|assistant|>\n",
    "no special tokens":  "User: {instruction}\nAssistant: ",
    "system prepended":   "<|system|>\n<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
}

instruction = "what is the capital of france"
reference = TRAIN_TEMPLATE.format(instruction=instruction)


def char_diff(a, b):
    """Where the strings first diverge, and by how much."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    return first, abs(len(a) - len(b)) + sum(1 for i in range(first, n)
                                             if a[i] != b[i])


print(f"training template: {TRAIN_TEMPLATE!r}\n")
print(f"{'variant':<20} {'identical':>10} {'first diff':>11} {'chars differ':>13}")
for name, tmpl in VARIANTS.items():
    served = tmpl.format(instruction=instruction)
    same = served == reference
    pos, n = char_diff(reference, served)
    print(f"{name:<20} {str(same):>10} {(pos if not same else '-'):>11} "
          f"{(n if not same else 0):>13}")

print("""
Every row except the first is a different string from the one the model was
tuned on, and none of them raises an error anywhere. The model receives a
prompt from a distribution it was not trained on and behaves like a partially
instruction-tuned model: mostly fine, occasionally reverting to continuation.

Note how small some of the differences are. A missing newline is one character.
The reason this costs teams days is that the symptom — 'quality dropped after
we refactored the serving code' — points at everything except a whitespace
change in a template string.

The defence is mechanical, not vigilance: serialise with ONE function, import
it in both the training and the serving path, and assert on a golden string in
CI.""")

# The mechanical defence, demonstrated.
GOLDEN = "<|user|>\nwhat is the capital of france<|end|>\n<|assistant|>\n"


def serialise(instruction):
    """The single source of truth. Both paths must call this."""
    return TRAIN_TEMPLATE.format(instruction=instruction)


assert serialise(instruction) == GOLDEN, "template drifted from the golden string"
print(f"golden-string check passed: {serialise(instruction)!r}")
```

And the finding that makes instruction tuning worth doing, simulated:

```python {tier=A name=held-out-task-generalisation}
"""Held-out task clusters: does instruction tuning teach tasks, or following?"""
import numpy as np

rng = np.random.default_rng(0)

CLUSTERS = ["sentiment", "nli", "summarisation", "qa", "translation",
            "classification", "reasoning", "extraction"]
D = 32


def cluster_vector(name):
    r = np.random.default_rng(abs(hash(name)) % (2 ** 31))
    v = r.normal(size=D)
    return v / np.linalg.norm(v)


# Two components of task performance:
#   - a task-specific part, only learnable from that cluster's examples
#   - a shared "follow the instruction" part, learnable from ANY cluster
task_dirs = {c: cluster_vector(c) for c in CLUSTERS}
follow_dir = np.ones(D) / np.sqrt(D)


def train(seen_clusters):
    """Returns learned task-specific strength per cluster, and follow strength."""
    task_strength = {c: (1.0 if c in seen_clusters else 0.0) for c in CLUSTERS}
    # Instruction-following accrues with the NUMBER OF DISTINCT clusters seen,
    # with diminishing returns — this is wei2022flan's secondary finding.
    follow = 1 - np.exp(-len(seen_clusters) / 3.0)
    return task_strength, follow


def performance(task_strength, follow, cluster):
    base = 0.25                                   # chance level
    return base + 0.45 * task_strength[cluster] + 0.30 * follow


print("Held-out cluster evaluation (eq:held-out-cluster)\n")
print(f"{'clusters trained on':>20} {'held-out perf':>15} {'seen-task perf':>16}")
held_out = "reasoning"
pool = [c for c in CLUSTERS if c != held_out]
for k in range(0, len(pool) + 1):
    seen = pool[:k]
    ts, fol = train(seen)
    held = performance(ts, fol, held_out)
    seen_perf = (np.mean([performance(ts, fol, c) for c in seen])
                 if seen else float("nan"))
    print(f"{k:>20} {held:>15.3f} {seen_perf:>16.3f}")

ts0, f0 = train([])
ts_all, f_all = train(pool)
print(f"\nheld-out '{held_out}' with no instruction tuning : "
      f"{performance(ts0, f0, held_out):.3f}")
print(f"held-out '{held_out}' after 7 other clusters      : "
      f"{performance(ts_all, f_all, held_out):.3f}")
print(f"improvement on a task type never seen             : "
      f"{performance(ts_all, f_all, held_out) - performance(ts0, f0, held_out):+.3f}")

assert performance(ts_all, f_all, held_out) > performance(ts0, f0, held_out)

print("""
The held-out column rises even though not one example of that task type was in
the training data. Nothing task-specific was learned for it — the model's
task_strength for 'reasoning' is zero throughout. What improved is the shared
instruction-following component, which accrues from cluster DIVERSITY and
saturates.

That is wei2022flan's result in miniature, and it is the reason instruction
tuning is a general capability unlock rather than a way of teaching specific
tasks: the thing being learned is 'a request should be answered', and that
transfers to requests you never trained on.""")
```

## 9. Practical Example

A team has a base model and 3,000 hand-written examples of their support
workflow. They want to know whether to spend another month collecting 30,000
more, or to spend it on template diversity and cleaning the 3,000 they have.

The evidence says the second, and the reason is worth understanding rather than
taking on authority.

```python {tier=A name=quality-versus-quantity}
"""Instruction data: does quality or quantity move the needle?"""
import numpy as np

rng = np.random.default_rng(3)


def simulate(n_examples, n_templates, quality, noise_floor=0.02):
    """A stand-in for downstream instruction-following quality.

    Three inputs, deliberately not symmetric:
      - n_examples: strong diminishing returns (log-shaped)
      - n_templates: teaches the INVARIANCE of eq:template-invariance
      - quality: bounds what can be learned at all — a ceiling, not a term
    """
    from_count = 0.30 * (1 - np.exp(-n_examples / 2000))
    from_templates = 0.25 * (1 - np.exp(-n_templates / 6))
    ceiling = quality
    raw = 0.25 + from_count + from_templates
    return min(raw, ceiling) - rng.normal(0, noise_floor)


print(f"{'option':<38} {'examples':>9} {'templates':>10} {'quality':>8} "
      f"{'result':>8}")
options = [
    ("as-is", 3_000, 2, 0.72),
    ("10x more examples, same templates", 30_000, 2, 0.72),
    ("same examples, 12 templates", 3_000, 12, 0.72),
    ("same examples, clean to high quality", 3_000, 2, 0.88),
    ("clean + diversify (one month)", 3_000, 12, 0.88),
    ("10x examples AND clean + diversify", 30_000, 12, 0.88),
]
results = {}
for name, n, k, q in options:
    r = simulate(n, k, q)
    results[name] = r
    print(f"{name:<38} {n:>9,} {k:>10} {q:>8.2f} {r:>8.3f}")

base = results["as-is"]
print(f"\n{'intervention':<38} {'gain over as-is':>17}")
for name, r in results.items():
    if name != "as-is":
        print(f"{name:<38} {r - base:>+17.3f}")

print("""
Three readings, and the third is the one that decides the month.

Diversifying templates on the data already collected beats collecting ten times
as much data with the same two templates. Example count has diminishing returns
because the model is re-weighting a mixture (eq:continuation-mixture), and
re-weighting does not need many samples; template diversity teaches an
INVARIANCE (eq:template-invariance), which is a different kind of thing.

But cleaning ALONE barely moves anything — less than the extra data does. That
is not a contradiction, it is what a ceiling means. At two templates the
configuration is nowhere near 0.72, so raising the cap to 0.88 has nothing to
bind on. Quality is not a term you add, it is a limit you eventually hit.

Hence the ordering. Diversify first, because it raises the achieved score; then
clean, because cleaning is what lets the raised score keep going. Doing them in
the other order looks like quality work that did not pay, and teams conclude
from that experience that data quality does not matter.""")
```

> PRODUCTION TIP: Before collecting more instruction data, measure the ceiling
> you are already hitting. Train on 25%, 50%, and 100% of what you have and plot
> the curve. If it has flattened, more of the same data will not help, and the
> curve costs three short training runs to produce.

## 10. Production Considerations

**Version the template with the model.** It is part of the artefact, not part of
the serving code. Store it in the model repository, load it from there in both
paths, and assert on a golden string in CI — the mechanical defence from
`template-mismatch`.

**Hold out task types, not just examples.** A random split measures whether you
have learned your training distribution. Holding out a whole task cluster
measures whether you have learned to follow instructions, which is what you
actually want to know ({{eq:held-out-cluster}}).

**Decontaminate the instruction set against your evaluations.**
{{ch:fm-datasets}}'s contamination problem applies at this scale too, and it is
easier to create accidentally: instruction data distilled from a stronger model
frequently contains benchmark items, because the teacher was asked about them.

**Mix in pretraining data to limit regression.** A small fraction of pretraining
batches interleaved with instruction batches measurably reduces capability loss
on continuation-style tasks, at negligible cost.

**Keep a general held-out set that the instruction data never touches.**
Capability regression is invisible in instruction-following metrics by
construction — this is {{ch:nlp-contextual}}'s catastrophic-forgetting warning
in its cheapest form.

**What to monitor:** instruction-following rate on a fixed eval set, refusal
rate, average response length (which drifts upward alarmingly easily), and the
general held-out score. Length drift is the early warning that the data has a
verbosity bias.

## 11. Common Mistakes

**Beginners:**

*Computing loss on the instruction.* {{eq:loss-masking}} exists for a reason and
{{eq:unmasked-gradient-share}} quantifies it — typically half the gradient
wasted.

*Using one template.* The model learns the mapping instead of the invariance
{{eq:template-invariance}}, and is then brittle to how users actually phrase
things.

*Expecting instruction tuning to add knowledge.*
{{eq:adaptation-information-ratio}} again. It re-weights
{{eq:continuation-mixture}}; it does not add modes.

**Experienced practitioners:**

*Template mismatch between training and serving.* The single most expensive
mistake in this chapter, and it produces no error. Serialise with one shared
function.

*Splitting randomly instead of by task type.* A random split cannot detect that
you have taught tasks rather than following.

*Ignoring length bias.* Instruction data with long responses teaches verbosity,
which is then measured as quality by human raters who prefer longer answers —
a bias that compounds into the preference data of {{ch:fm-rlhf}}.

*Getting the multi-turn mask wrong.* Masking only the first assistant turn is a
silent bug that costs most of a multi-turn dataset's value. The
`instruction-masking` listing asserts against exactly this.

*Distilling from a stronger model without checking licensing or contamination.*
Both are real constraints and both are usually discovered late.

## 12. Failure Modes

**Template mismatch.** Serving format differs from training format. *Symptom:*
degraded quality with no error, often after unrelated refactoring. *Detection:*
golden-string assertion in CI.

**Capability regression.** The model becomes worse at things the instruction
data did not cover. *Symptom:* invisible in instruction metrics by construction.
*Detection:* the untouched general held-out set. *Mitigation:* pretraining
data mixed in.

**Verbosity drift.** Responses grow longer over training. *Symptom:* rising mean
response length, and raters preferring the model for reasons unrelated to
correctness. *Detection:* track length as a first-class metric.

**Format overfitting.** Excellent on the training template, poor on rephrasings.
*Detection:* evaluate with templates never seen in training.

**Mode collapse onto a response shape.** Every answer becomes a bulleted list,
or every answer opens with the same clause. *Cause:* a dominant pattern in the
instruction data. *Detection:* n-gram statistics over generated openings.

**Refusal over-generalisation.** Safety-motivated examples teach refusal of
benign requests that share surface features. *This becomes a central problem in
{{ch:fm-rlhf}}*, where it is optimised for rather than merely demonstrated.

## 13. Alternatives

{#tbl:behaviour-shaping caption="Ways to make a base model behave usefully, by cost and by what they change. The first two change nothing about the model and are the correct first attempt; the last changes the most and is the subject of the next two chapters."}

| Method | Changes | Cost | Ceiling |
|---|---|---|---|
| Few-shot prompting | nothing | inference only | limited by context and consistency |
| System prompt | nothing | inference only | fragile to user input |
| Instruction tuning | weights, slightly | GPU-hours | format and behaviour, not preference |
| Preference optimisation | weights | GPU-hours + preference data | which of several good answers |
| Full RLHF | weights | GPU-days + pipeline | same, with online sampling |

**Which compute the same function.** The first two supply the task at inference
and are undone by a new prompt; instruction tuning makes the behaviour a
property of the weights. The last two do something instruction tuning cannot:
they optimise *which* of several acceptable responses is preferred, which
demonstrations cannot express because a demonstration shows one answer rather
than a ranking. That gap is why {{ch:fm-rlhf}} exists.

**Few-shot prompting is the baseline to beat**, and it is frequently competitive
for narrow tasks. It costs context on every request, which at scale is a real
price ({{ch:tf-complexity}}), but it requires no training and no artefact to
version.

## 14. Evaluation

**Is the implementation correct?**

1. **Mask boundaries** — the first response token supervised, instruction tokens
   not, every assistant turn covered. The `instruction-masking` assertions are
   this test.
2. **Template round-trip** — the golden-string check, in CI, in both paths.
3. **Loss magnitude** — should start below pretraining's converged loss, since
   the model already models this text well. A loss starting at $\log|V|$ means
   the checkpoint did not load.

**Is the model any good?**

1. **Held-out task-type performance** {{eq:held-out-cluster}} — the measurement
   that distinguishes following from memorising.
2. **Template robustness** — the same evaluation under unseen phrasings.
3. **General capability on an untouched set** — for regression.
4. **Length and refusal statistics** — for drift.

**On human evaluation.** Instruction-following quality is ultimately judged by
people, and people prefer longer, more confident answers independent of
correctness. Any human evaluation of this stage must control for length, or it
measures verbosity. {{part:25}} treats this properly and it starts here.

## 15. Advanced Concepts

**Self-instruct and synthetic instruction data.** {{maturity:ESTABLISHED}}
Generate instructions and responses with a model, filter, and train on the
result. Cheap and effective, with a drift risk: each generation narrows toward
the generator's distribution.

**Instruction backtranslation.** {{maturity:EMERGING}} Take unlabelled documents
and generate plausible instructions that would elicit them, giving instruction
data from a corpus rather than from annotators.

**Task-mixture optimisation.** {{maturity:EMERGING}} Choosing cluster weights to
maximise held-out generalisation rather than sampling uniformly. Diversity is
known to matter; the optimal weighting is not.

**Long-context instruction tuning.** {{maturity:EMERGING}} Instruction data is
short, so a model tuned on it may follow instructions poorly at long context —
a distribution mismatch between tuning and deployment that is easy to overlook.

**System-prompt training.** {{maturity:ESTABLISHED}} Training with varied system
prompts so the model learns to condition on them, rather than treating them as
ordinary text. This is what makes a system prompt reliable enough to build on.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:fm-pretraining}} supplies the base model and the objective —
{{eq:instruction-tuning-loss}} is {{eq:clm-loss}} with a mask.
{{ch:fm-what-they-are}}'s {{eq:adaptation-information-ratio}} bounds what this
stage can do, and {{eq:continuation-mixture}} explains why a bounded
intervention is nonetheless so effective. {{ch:nlp-contextual}} supplied
fine-tuning and catastrophic forgetting, and {{cite:howard2018}}'s recipe is
this one at smaller scale. {{ch:dl-regularization}}'s augmentation argument is
{{eq:template-invariance}}. {{ch:mle-pipelines}}'s train/serve skew is the
template mismatch.

**Forwards.** {{ch:fm-rlhf}} adds what demonstrations cannot express — a
preference between acceptable answers — and {{ch:fm-dpo}} simplifies it.
{{ch:fm-distillation}} is where most instruction data actually comes from.
{{ch:llm-prompting}} is the inference-time alternative this stage competes with.
{{part:14}} makes this stage parameter-efficient, and {{part:26}} is about
prompts that reach past it to the base model underneath.

## 17. Exercises

**Beginner**

1. Give three plausible base-model continuations of "Explain photosynthesis."
   and say why each is reasonable.
2. Why is the loss masked to the response? What is lost by not masking?
3. What is a chat template and why must it match at serving?

**Intermediate**

4. Using {{eq:unmasked-gradient-share}}, compute the wasted gradient fraction for
   instructions of 200 tokens and responses of 50.
5. Explain {{eq:held-out-cluster}} and why held-out *task type* is a stronger
   test than held-out examples.
6. Compute the instruction-tuning compute for 20,000 examples averaging 300
   tokens on a 13B model, and express it as a fraction of a 2T-token pretraining
   run.

**Advanced**

7. Derive {{eq:continuation-mixture}}'s implication: why can a small dataset
   re-weight modes but not create them? Relate it to
   {{eq:adaptation-information-ratio}}.
8. Design an experiment separating "learned the task" from "learned to follow
   instructions", and say what result would distinguish them.
9. Argue whether instruction tuning should mix in pretraining data, using the
   regression risk and the compute cost.

**Implementation**

10. Extend `instruction-masking` to handle a conversation where the assistant
    turn contains a tool call that should not be supervised, and assert the mask
    is correct.
11. Implement the ablation from {{sec:10-production-considerations}}: train on
    25%, 50% and 100% of a dataset and plot the curve to find the ceiling.
12. Implement template diversity: train with 1, 3 and 10 templates and evaluate
    on a template never seen. Show the invariance appearing.
13. Build the golden-string CI check as a real test, and demonstrate it failing
    when a single whitespace character changes.

**Reasoning**

14. A team reports that instruction tuning made their model worse at code
    completion. Explain what probably happened and how to confirm it.
15. Explain why demonstrations cannot express a preference between two good
    answers, and what that implies for the next chapter.

## 18. Interview Questions

**Beginner**

1. Why does a base model not answer questions?
2. What is instruction tuning and what does it change?
3. What is loss masking?

**Intermediate**

4. What did FLAN establish, and why is held-out task-type evaluation the right
   test?
5. Why does template diversity matter?
6. How much compute does instruction tuning take relative to pretraining?

**Senior**

7. You have 3,000 examples and a month. More data or better data? Justify it.
8. Your model degraded after a serving refactor. Walk through the diagnosis.
9. When would you use few-shot prompting instead of instruction tuning?

**Systems**

10. Design the instruction-tuning pipeline: data versioning, template handling,
    evaluation, and the CI checks that prevent the failures in this chapter.
11. How do you detect capability regression from a stage whose own metrics
    cannot see it?

## 19. Research Questions

**What is the optimal task mixture?** Diversity helps with diminishing returns
{{cite:wei2022flan}}, but the weighting is chosen by convention. Optimise
cluster weights against held-out generalisation directly and report how much is
available over uniform sampling.

**How much of instruction tuning is format?** Train on instruction data with the
responses replaced by generic acknowledgements — teaching only that a request is
followed by a response — and measure how much of the improvement survives. If
much of it does, the field is over-investing in response quality at this stage.

**Where does synthetic instruction data stop helping?** Self-instruct narrows
toward the generator. Measure diversity and downstream quality across several
generations of self-generated data to locate the point where narrowing dominates.

**Does long-context instruction-following require long-context instruction
data?** Instruction sets are short and deployments are not. Measure
instruction-following as a function of context length for models tuned on short
data — the mismatch is plausible, easy to test, and largely unmeasured.

## 20. Chapter Summary

A base model does not answer questions because it was trained to continue text,
and in web text a question is frequently followed by another question.
Instruction tuning fixes this with the *same objective* as pretraining
{{eq:instruction-tuning-loss}}, applied to instruction–response pairs with the
loss masked to the response {{eq:loss-masking}}.

**The stage is tiny and its effect is enormous.** A 50,000-example dataset is
about six parts per million of pretraining compute — roughly nineteen minutes on
one accelerator — and it produces most of the difference between a base model
and something a person would call usable. {{eq:continuation-mixture}} explains
why: the answering behaviour already exists as one mode among many, and
re-weighting a mixture requires vastly less information than learning its
components. It also predicts the limit — no re-weighting creates a mode that is
not there, which is {{eq:adaptation-information-ratio}} from another direction.

**{{cite:wei2022flan}}'s result is about generalisation across task types, not
tasks.** Training on many clusters and evaluating on a held-out cluster
{{eq:held-out-cluster}} shows improvement on a task type never seen, so what is
learned is "a request should be answered" rather than any specific task. Cluster
diversity matters more than example count, with diminishing returns.

**Two implementation details carry most of the practical risk.** Loss masking
determines whether half the gradient models instructions rather than responses
{{eq:unmasked-gradient-share}}, and the multi-turn version of the bug silently
discards most of a dataset's value. And the chat template is a *contract*: a
one-character difference between the training and serving serialisation produces
a model that is quietly worse with no error anywhere, which is why the defence
must be a shared function and a golden-string assertion rather than care.

Finally, the axis this stage cannot reach. A demonstration shows *one* good
answer; it cannot express that one acceptable answer is better than another.
That is a preference, it requires comparisons rather than demonstrations, and it
is what {{ch:fm-rlhf}} is for.

## 21. Further Reading

{{cite:wei2022flan}} is the paper, and §2 is the experimental design — the
cluster-holdout construction is the contribution, more than any number in the
results. Read it asking how you would have designed the experiment, and notice
that the obvious design (held-out examples) would not have shown anything
interesting.

{{cite:ouyang2022}}'s §3.1 covers the demonstration-collection stage and is the
best published description of what human instruction data actually looks like
and what it costs. The rest of the paper belongs to {{ch:fm-rlhf}}.

{{cite:touvron2023llama}} and its successors document instruction-tuning recipes
with unusual specificity for models you can actually download, which makes them
better reading than a paper you cannot reproduce.

{{cite:gunasekar2023}}, from {{ch:fm-datasets}}, is worth revisiting here: its
argument that curated data beats volume is the same claim
`quality-versus-quantity` makes at instruction scale, and the two together are
the strongest case in the part for spending effort on data rather than on
quantity.

**Where to go next:** {{ch:fm-rlhf}} takes up what demonstrations cannot teach —
which of several acceptable answers a person actually prefers — and the
machinery that turns comparisons into a training signal.
