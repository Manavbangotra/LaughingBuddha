---
id: fm-what-they-are
number: 79
part: IX
tier: full
status: draft
requires: [nlp-bert, nlp-contextual, tf-architectures, nlp-similarity,
           ml-what-it-is, mle-registry]
provides: [foundation-model, pretrain-adapt-paradigm, homogenisation,
           in-context-learning, base-model, alignment-pipeline, capability-overhang,
           model-provenance, proxy-objective]
citations: [bommasani2021, brown2020, ouyang2022, touvron2023llama,
            devlin2019bert, howard2018, radford2019, hoffmann2022chinchilla]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State what distinguishes a foundation model from a large pretrained model,
   and say whether the distinction is technical or economic.
2. Explain the pretrain-then-adapt paradigm and identify which stage each of
   the next eight chapters addresses.
3. Explain in-context learning as a consequence of the training pipeline rather
   than as a designed feature.
4. State the homogenisation argument precisely, and give a concrete way a
   defect propagates from a base model to everything built on it.
5. Explain why every stage after pretraining exists — each as a correction to a
   specific deficiency of the objective before it.
6. Assess a claim about a foundation model given what kind of evidence supports
   it, and say what a lab reporting on its own model can and cannot establish.
7. Decide whether to use, adapt, or train a model, from the cost structure.

## 2. Why This Matters

**This part is where the models you actually use come from.** Everything in
{{part:10}} onward — prompting, tool use, RAG, agents — assumes a model produced
by the pipeline this chapter introduces. Understanding the pipeline is what
makes those later behaviours predictable rather than surprising.

**The single most useful idea in this part is that every alignment stage is a
patch.** Pretraining optimises next-token prediction on web text, which is not
what anyone wants — it is what can be optimised at scale. Instruction tuning
exists because such a model does not recognise a request as a request. RLHF
exists because "be helpful and harmless" cannot be written as a loss. Each stage
corrects something the previous objective did not contain, and knowing which
patch produced which behaviour is how you debug a model that misbehaves.

**The evidence base here is weaker than anywhere else in this book, and that is
itself the lesson.** Several of this part's primary sources have no peer-reviewed
venue, and the most important numbers — what is in the training data, what the
frontier recipes are — are unpublished. This chapter establishes the habit the
rest of the part depends on: **say what kind of evidence a claim rests on, in
the sentence that makes the claim.**

**And the economics decide more architecture than the capabilities do.** Almost
nobody trains a foundation model. The decision that actually gets made is use,
adapt, or distil, and {{sec:9-practical-example}} does that arithmetic.

## 3. Prerequisites

{{ch:nlp-bert}} for pretraining and fine-tuning as separate stages, and for the
budget-equalisation lesson that this part needs repeatedly.
{{ch:nlp-contextual}} for transfer learning and {{cite:howard2018}}'s recipe.
{{ch:tf-architectures}} for the decoder-only stack these models are.
{{ch:nlp-similarity}} for contrastive training, which returns as preference
learning. {{ch:ml-what-it-is}} for the distinction between an objective and a
goal. {{ch:mle-registry}} for versioning artefacts, which becomes model
provenance here.

## 4. Intuitive Explanation

Until about 2018, building an NLP system meant collecting labelled data for your
task and training a model on it. A sentiment classifier saw sentiment labels. A
translation system saw translation pairs. The model was built for the job.

The foundation-model paradigm inverts this. Train one very large model on an
enormous amount of unlabelled text, with an objective that has nothing to do
with any particular task — predict the next token — and then *adapt* it. The
adaptation might be fine-tuning, or it might be nothing more than describing the
task in the prompt.

**The strange part is that the second approach produces better task performance
than the first, on tasks the model was never trained for.** That needs
explaining, and the explanation is the one Part VIII established for
representation: predicting the next token well requires modelling a great deal
about the world, because the next token frequently depends on syntax, on facts,
on arithmetic, on the intent of the person writing. The objective is a proxy,
and the proxy is demanding enough that satisfying it produces general
capability.

> NOTE: This is the autoencoder argument from {{ch:dl-autoencoders}} and the
> ELMo argument from {{ch:nlp-contextual}} at a different scale — set up a task
> whose solution requires a good internal model, then use the internal model. It
> is the same trick three times, and the third time it produced something
> qualitatively different.

**Then the trouble starts.** A model trained to continue text is a *text
continuer*. Ask it a question and a plausible continuation is another question —
because in web text, questions cluster. It has the knowledge and not the
behaviour.

So the pipeline grew stages. **Instruction tuning** teaches it that a request
should be answered. **Alignment** teaches it which answers people prefer.
**Distillation** makes the result cheap enough to serve. Each stage exists
because the stage before it optimised something that was not quite the goal.

**The mental model to carry:** a foundation model is a general capability
produced by a proxy objective, wrapped in successive layers of correction that
translate capability into behaviour. Where it breaks down: the layers are thin.
The corrections are much smaller interventions than pretraining, they are
applied to a model whose knowledge they cannot substantially change, and a good
deal of what looks like reasoning failure is a base model showing through a thin
patch.

## 5. Formal Explanation

### 5.1 The definition, and what it commits to

{{cite:bommasani2021}} defines a foundation model as one "trained on broad data
at scale" and "adaptable to a wide range of downstream tasks". Three clauses,
each doing work:

- **Broad data** — not task data, and not domain data. The corpus is not chosen
  for what the model will do.
- **At scale** — deliberately unquantified, and the definition is weaker for it.
- **Adaptable** — one model serves many tasks, and adaptation is cheap relative
  to training.

> IMPORTANT: The term was coined by an institute that also builds such models,
> in a report with 128 authors and no peer review. That does not make it wrong —
> the category is real and the word is useful — but a term of art introduced by
> interested parties deserves the same scrutiny as any other claim. What the
> definition is *not* is a technical threshold: there is no parameter count, no
> corpus size, and no capability test that decides membership.

Formally, contrast the two paradigms. Task-specific training solves

$$
\theta^*_{\text{task}} = \argmin_{\theta}\ \E_{(x,y)\sim\Data_{\text{task}}}
 \big[\Loss_{\text{task}}(f_\theta(x), y)\big]
$$ (eq:task-specific-training)

for each task separately, requiring $\Data_{\text{task}}$ to exist. The
foundation-model paradigm solves one general problem

$$
\theta^*_{\text{base}} = \argmin_{\theta}\ \E_{x\sim\Data_{\text{broad}}}
 \big[\Loss_{\text{proxy}}(f_\theta, x)\big]
$$ (eq:foundation-pretraining)

and then obtains task behaviour by an adaptation $A$ that is cheap:

$$
f_{\text{task}} = A\big(f_{\theta^*_{\text{base}}},\ c_{\text{task}}\big),
\qquad \text{cost}(A) \ll \text{cost}(\theta^*_{\text{base}})
$$ (eq:adaptation)

**The economics are the paradigm.** {{eq:foundation-pretraining}} is paid once
by someone with a data centre; {{eq:adaptation}} is paid by everyone else. When
$c_{\text{task}}$ is a prompt, the cost of adaptation is *zero training runs*,
which is what changed who can build things.

### 5.2 The adaptation ladder

Adaptations, ordered by cost:

{#tbl:adaptation-ladder caption="Ways to turn a base model into task behaviour, from cheapest to most expensive. The first three change no weights at all — which is the practical content of the foundation-model paradigm, and the reason most teams never train anything."}

| Adaptation | Weights changed | Data needed | Cost | Where treated |
|---|---|---|---|---|
| Zero-shot prompt | none | none | one inference | {{ch:llm-prompting}} |
| Few-shot prompt | none | a handful of examples | one inference | {{ch:llm-prompting}} |
| Retrieval augmentation | none | a corpus | inference + index | {{part:12}} |
| Parameter-efficient tuning | ~0.1–1% | hundreds to thousands | GPU-hours | {{part:14}} |
| Full fine-tuning | all | thousands+ | GPU-days | {{part:14}} |
| Continued pretraining | all | billions of tokens | GPU-months | {{sec:15-advanced-concepts}} |
| Pretraining from scratch | all | trillions of tokens | $10^6$–$10^8$ | {{ch:fm-pretraining}} |

### 5.3 In-context learning

{{cite:brown2020}}'s central observation: for sufficiently large models,
supplying $k$ input–output examples in the prompt improves task performance,
with no gradient update:

$$
P\big(y \given x,\ (x_1,y_1),\dots,(x_k,y_k);\ \theta\big)
$$ (eq:in-context-learning)

The model's weights are identical before and after. **Whatever learning means
here, it happens inside a forward pass and is discarded when the context is.**

The mechanism is not settled and {{ch:llm-prompting}} treats it properly. What
is settled and belongs here is the *reason it exists*: with no task-specific
training stage, the only place left to put the task is the input. In-context
learning is not a feature somebody designed. It is what a general next-token
predictor does when you condition it on a pattern, and the field discovered it
rather than building it.

### 5.4 The pipeline, and why each stage exists

$$
\underbrace{\text{pretrain}}_{\text{capability}}
\ \to\ \underbrace{\text{instruction-tune}}_{\text{format}}
\ \to\ \underbrace{\text{align}}_{\text{preference}}
\ \to\ \underbrace{\text{distil}}_{\text{cost}}
$$ (eq:foundation-pipeline)

Each arrow is a correction to a deficiency of the objective to its left:

| Stage | Corrects | Chapter |
|---|---|---|
| Pretraining | nothing — this is the proxy | {{ch:fm-pretraining}} |
| Instruction tuning | a text continuer does not answer requests | {{ch:fm-instruction-tuning}} |
| Alignment | "helpful and harmless" is not a loss function | {{ch:fm-rlhf}}, {{ch:fm-dpo}} |
| Distillation | the result is too expensive to serve | {{ch:fm-distillation}} |

**A base model is the output of stage one only**, and it behaves very
differently from anything you have used through a chat interface. Knowing that
distinction is what makes the later chapters legible.

### 5.5 Homogenisation

{{cite:bommasani2021}}'s most durable argument. When many applications are built
on one base:

$$
\text{defect}(\text{base}) \implies
 \text{defect}\big(A_1(\text{base})\big),\ \dots,\ \text{defect}\big(A_n(\text{base})\big)
$$ (eq:homogenisation)

for any adaptation $A_i$ that does not specifically remove it. A bias, a factual
error, a security weakness, or a capability gap in the base is inherited by
everything above it — and since $\text{cost}(A_i) \ll \text{cost}(\text{base})$,
the adaptations are far too weak to correct anything fundamental.

**This is a structural argument, not a speculative one.** It is the same
reasoning as a shared dependency in software supply chains, and it has the same
consequence: the blast radius of a defect is the whole ecosystem above it, and
correlated failure is the default rather than the exception.

## 6. Mathematical Foundation

### 6.1 Why the proxy objective produces general capability

The pretraining objective is
$\Loss = -\E[\log P(x_t \given x_{<t})]$, whose optimum is the true conditional
distribution of the data. Decompose the cross-entropy:

$$
\E\big[-\log P_\theta(x_t\given x_{<t})\big]
 = \underbrace{H(X_t \given X_{<t})}_{\text{irreducible}}
 + \underbrace{\KL\big(P^* \,\|\, P_\theta\big)}_{\text{what training reduces}}
$$ (eq:pretraining-decomposition)

$\square$

The first term is the entropy of the text itself and no model reduces it. **All
progress is the second term**, and driving it to zero requires matching the true
conditional distribution over *every* context in the corpus.

That is the argument in one line: the corpus contains arithmetic, so matching
the distribution requires arithmetic; it contains code, so it requires syntax;
it contains dialogue, so it requires modelling intent. **Generality is not a
side effect of scale — it is a requirement of the objective, and scale is what
makes satisfying it possible.**

The honest limit of this argument: it says the *optimum* has these properties.
It says nothing about how far along that path a given model is, and the gap
between "would need to model arithmetic" and "does model arithmetic" is where
most disappointment with these systems lives.

### 6.2 Why adaptation cannot fix what pretraining lacks

Let adaptation update parameters by $\Delta\theta$ using $n_A$ examples, against
pretraining's $n_P$ tokens. For the adapted model to acquire a capability
absent from the base, $\Delta\theta$ must carry the information for it.

A rough but load-bearing accounting: fine-tuning on $n_A$ examples supplies at
most $O(n_A)$ label-bits of new information, whereas the base absorbed $O(n_P)$
token-bits. With $n_P \approx 10^{12}$ and $n_A \approx 10^4$:

$$
\frac{n_A}{n_P} \approx 10^{-8}
$$ (eq:adaptation-information-ratio)

**Adaptation is eight orders of magnitude smaller than pretraining**, so it can
select, emphasise, and reformat what the base already represents, and it cannot
install a capability the base lacks.

$\square$

This is the formal version of a rule practitioners learn the hard way:
**fine-tuning teaches format and style reliably, and facts and skills poorly.**
{{part:14}} returns to it with measurements, and it is the reason
{{part:12}}'s retrieval exists — supplying knowledge through the *context* rather
than the weights sidesteps {{eq:adaptation-information-ratio}} entirely.

### 6.3 Where the pipeline's compute goes

Approximate compute, in FLOPs, using $6ND$ from {{ch:tf-complexity}} for a 7B
model:

$$
C_{\text{pretrain}} = 6ND = 6 \times 7\times10^9 \times 2\times10^{12}
 \approx 8.4\times10^{22}
$$

$$
C_{\text{instruct}} \approx 6 \times 7\times10^9 \times 5\times10^7
 \approx 2.1\times10^{18},
\qquad
\frac{C_{\text{instruct}}}{C_{\text{pretrain}}} \approx 2.5\times10^{-5}
$$ (eq:stage-compute-ratio)

**Instruction tuning is about one part in forty thousand of the pretraining
compute** and is responsible for most of the difference between a base model and
something a person would call usable. That ratio is the strongest single piece
of evidence that these stages are teaching *behaviour* rather than capability —
and it is consistent with {{eq:adaptation-information-ratio}}.

## 7. Internal Mechanics

```mermaid {#fig:foundation-pipeline caption="The foundation-model pipeline. Pretraining is the only stage that builds capability, and it consumes essentially all of the compute; the stages to its right shape behaviour and cost. A base model is the output of stage one alone, and behaves very unlike a chat model."}
graph TD
  A["broad corpus<br/>10^12+ tokens"] --> B["PRETRAIN<br/>next-token prediction<br/>~100% of compute"]
  B --> C["base model<br/>capability, no behaviour"]
  C --> D["INSTRUCTION TUNE<br/>demonstrations<br/>~0.0025% of compute"]
  D --> E["instruct model<br/>answers requests"]
  E --> F["ALIGN<br/>preference data<br/>RLHF or DPO"]
  F --> G["aligned model<br/>preferred answers"]
  G --> H["DISTIL / QUANTISE<br/>optional"]
  H --> I["served model"]
  C -.->|"few-shot prompt<br/>no weights changed"| I
  style B fill:#fde,stroke:#c69
  style C fill:#dfe,stroke:#5a5
```

**The dotted path matters.** A base model with a few-shot prompt reaches the
serving stage with no training whatsoever, which is {{cite:brown2020}}'s result
and the reason the paradigm changed who can build things.

**What a base model actually does.** Prompted with "What is the capital of
France?", a base model may answer, or continue with more questions, or produce a
quiz-page footer — all high-probability continuations in web text. It is not
failing; it is doing exactly what it was trained to do. Every chapter after this
one is about closing the distance between that and a usable system.

**Where the artefacts live.** Each stage produces a checkpoint that must be
versioned, and provenance is the practical form of {{eq:homogenisation}}: to
know whether a defect affects you, you must know which base you are on and which
adaptations were applied. {{ch:mle-registry}}'s discipline applies, with the
complication that the base is usually someone else's artefact whose composition
you cannot inspect.

## 8. Implementation

The adaptation ladder is a cost decision before it is a technical one, and the
arithmetic is short enough to do exactly.

```python {tier=A name=adaptation-cost-ladder}
"""What each rung of the adaptation ladder costs, in compute and dollars."""

N = 7e9                     # parameters
D_PRETRAIN = 2e12           # pretraining tokens
FLOPS_PER_TOKEN_TRAIN = 6 * N          # the 6ND rule, ch:tf-complexity
FLOPS_PER_TOKEN_INFER = 2 * N

GPU_FLOPS = 1e15            # a realistic sustained rate, not peak
GPU_COST_PER_HOUR = 2.50
MFU = 0.45                  # ch:tf-complexity: 40-55% is the achievable band


def hours(flops):
    return flops / (GPU_FLOPS * MFU) / 3600


def dollars(flops, n_gpus=1):
    return hours(flops) * GPU_COST_PER_HOUR * n_gpus


RUNGS = [
    ("zero-shot prompt",        0,        "none"),
    ("few-shot prompt",         0,        "none"),
    ("parameter-efficient tune", 6 * N * 1e7 * 0.01, "~1% of weights"),
    ("full fine-tune",          6 * N * 1e7,        "all weights"),
    ("continued pretraining",   6 * N * 5e10,       "all weights"),
    ("pretrain from scratch",   6 * N * D_PRETRAIN, "all weights"),
]

print(f"{'adaptation':<24} {'train FLOPs':>13} {'GPU-hours':>12} "
      f"{'cost (1 GPU)':>14} {'vs pretrain':>13}")
base = 6 * N * D_PRETRAIN
for name, flops, _ in RUNGS:
    if flops == 0:
        print(f"{name:<24} {'0':>13} {'0':>12} {'$0':>14} {'—':>13}")
        continue
    print(f"{name:<24} {flops:>13.2e} {hours(flops):>12,.1f} "
          f"${dollars(flops):>13,.0f} {flops / base:>12.1e}x")

print(f"\nPretraining on one GPU would take {hours(base) / 24 / 365:,.1f} years "
      f"— which is why it is done on thousands at once.")

# Equation (eq:stage-compute-ratio): instruction tuning against pretraining.
instruct = 6 * N * 5e7
print(f"instruction tuning / pretraining = {instruct / base:.1e}")
print("That fraction produces most of the difference between a base model and "
      "one a person would call usable — which is evidence it teaches behaviour, "
      "not capability.")

# The decision nobody does the arithmetic for: at what request volume does
# serving your own adapted model beat paying per token?
API_PER_1K_TOKENS = 0.002
SELF_HOST_GPU_HOUR = 2.50
TOKENS_PER_REQUEST = 800

throughput = GPU_FLOPS * MFU / FLOPS_PER_TOKEN_INFER      # tokens/second
self_host_per_token = SELF_HOST_GPU_HOUR / 3600 / throughput
api_per_token = API_PER_1K_TOKENS / 1000

print(f"\nself-hosted throughput: {throughput:,.0f} tokens/s")
print(f"self-hosted: ${self_host_per_token:.3e}/token   "
      f"API: ${api_per_token:.3e}/token")
if self_host_per_token < api_per_token:
    daily = 24 * 3600 * throughput / TOKENS_PER_REQUEST
    print(f"Self-hosting is cheaper per token, but only if the GPU is BUSY: "
          f"it must serve ~{daily:,.0f} requests/day to stay saturated.")
    print("An idle GPU costs full price. Utilisation, not unit cost, is the "
          "variable that decides this.")
```

The last block is the calculation that actually gets made in practice, and its
conclusion is usually counter-intuitive: self-hosting wins on unit cost and
loses on utilisation, so the deciding variable is whether your traffic can keep
the hardware busy.

Now the homogenisation argument, which is easier to believe as a simulation than
as prose:

```python {tier=A name=homogenisation}
"""How a defect in one base propagates to everything adapted from it."""
import numpy as np

rng = np.random.default_rng(0)

N_APPS, N_PROBES = 200, 2000
BASE_DEFECT_RATE = 0.04          # 4% of probes trigger a base-model defect
ADAPT_FIX_RATE = 0.25            # adaptation happens to fix a quarter of them
ADAPT_OWN_DEFECT = 0.01          # and introduces its own, independently


def simulate(n_bases):
    """N_APPS applications spread over n_bases distinct base models."""
    base_of = rng.integers(0, n_bases, N_APPS)
    base_defects = rng.random((n_bases, N_PROBES)) < BASE_DEFECT_RATE

    fails = np.zeros((N_APPS, N_PROBES), dtype=bool)
    for a in range(N_APPS):
        inherited = base_defects[base_of[a]] & (rng.random(N_PROBES) > ADAPT_FIX_RATE)
        own = rng.random(N_PROBES) < ADAPT_OWN_DEFECT
        fails[a] = inherited | own

    per_app = fails.mean(1).mean()
    # The quantity that matters for systemic risk: given one app fails on a
    # probe, how much of the ecosystem fails on that same probe?
    hit = fails.sum(0)
    correlated = float(hit[hit > 0].mean() / N_APPS)
    worst = float(hit.max() / N_APPS)
    return per_app, correlated, worst


print(f"{N_APPS} applications, {N_PROBES} probes, "
      f"base defect rate {BASE_DEFECT_RATE:.0%}\n")
print(f"{'distinct bases':>15} {'per-app failure':>17} "
      f"{'mean co-failure':>17} {'worst probe':>13}")
for n_bases in (1, 2, 5, 20, 200):
    per_app, corr, worst = simulate(n_bases)
    print(f"{n_bases:>15} {per_app:>17.3f} {corr:>17.1%} {worst:>13.1%}")

print("""
Read the first column against the last.

Per-app failure barely moves: about 4% however many bases exist, because each
application's reliability is dominated by its own adaptation. An audit that
samples one application cannot distinguish these worlds at all.

The worst-probe column is where homogenisation lives. On one shared base, the
single worst input takes down ~82% of the ecosystem simultaneously; spread over
two hundred bases, the same per-application reliability caps the worst input at
~13%. Mean co-failure moves much less, because it averages over the many probes
that trip only one or two applications — the tail is the risk, not the mean.

That is equation (eq:homogenisation) as a measurement: homogenisation does not
make any individual system worse, it makes the whole system fail TOGETHER, and
only a tail statistic can see it.""")
```

## 9. Practical Example

A team is building a customer-support assistant. Someone asks whether they
should fine-tune a model on their support history. It is the default suggestion
and it is usually wrong, and {{eq:adaptation-information-ratio}} says why: the
support history contains *knowledge*, and fine-tuning is a poor way to install
knowledge.

The decision has four candidates, and it is worth settling on evidence.

```python {tier=A name=use-adapt-or-train}
"""Choosing an adaptation: what each option can actually deliver."""

REQUIREMENTS = {
    "answer from our current docs":      "knowledge",
    "keep up when docs change weekly":   "knowledge-freshness",
    "reply in our house voice":          "style",
    "always emit our JSON ticket schema": "format",
    "handle our product's jargon":       "vocabulary",
    "cost under $0.01 per conversation": "cost",
}

# What each adaptation is actually good at. Grounded in
# eq:adaptation-information-ratio: adaptation reshapes, retrieval supplies.
CAPABILITY = {
    "prompting only":      {"style": 0.6, "format": 0.7, "vocabulary": 0.4,
                            "knowledge": 0.1, "knowledge-freshness": 0.1, "cost": 0.9},
    "prompt + retrieval":  {"style": 0.6, "format": 0.7, "vocabulary": 0.8,
                            "knowledge": 0.9, "knowledge-freshness": 0.95, "cost": 0.7},
    "fine-tune":           {"style": 0.95, "format": 0.95, "vocabulary": 0.8,
                            "knowledge": 0.3, "knowledge-freshness": 0.0, "cost": 0.8},
    "fine-tune + retrieval": {"style": 0.95, "format": 0.95, "vocabulary": 0.85,
                              "knowledge": 0.9, "knowledge-freshness": 0.95, "cost": 0.6},
}
THRESHOLD = 0.5      # a requirement counts as met above this

SETUP_COST = {"prompting only": 0, "prompt + retrieval": 15_000,
              "fine-tune": 40_000, "fine-tune + retrieval": 55_000}

print(f"{'requirement':<36} {'need':<20}")
for req, need in REQUIREMENTS.items():
    print(f"  {req:<34} {need:<20}")

print(f"\n{'approach':<24} {'worst requirement':<24} {'score':>7} {'setup':>10}")
rows = []
for approach, caps in CAPABILITY.items():
    scores = {need: caps[need] for need in REQUIREMENTS.values()}
    weakest = min(scores, key=scores.get)
    rows.append((approach, weakest, scores[weakest], SETUP_COST[approach]))
    print(f"{approach:<24} {weakest:<24} {scores[weakest]:>7.2f} "
          f"${SETUP_COST[approach]:>9,}")

viable = [r for r in rows if r[2] >= THRESHOLD]
best = min(viable, key=lambda r: r[3]) if viable else None
print(f"\nA system is only as good as its weakest requirement, so the column "
      f"to read is the minimum, not the average.")
if best:
    print(f"Cheapest option clearing every requirement: {best[0]} "
          f"(${best[3]:,} setup)")
else:
    print("No option clears every requirement — the requirements need cutting.")

print("""
Note what the table says about fine-tuning alone: it scores highest on style
and format and fails outright on knowledge freshness, because weights are
frozen at training time and the docs change weekly. No amount of fine-tuning
fixes that — it is a property of where the information lives, which is
equation (eq:adaptation-information-ratio) showing up as a product decision.""")
```

**The conclusion generalises past this example.** Fine-tuning is for *how the
model behaves*; retrieval is for *what it knows*. Teams reach for fine-tuning
when they mean retrieval because fine-tuning sounds like the more serious
intervention, and it is the more expensive way to not solve the problem.

> PRODUCTION TIP: Before any fine-tuning project, write down which requirement
> it is meant to satisfy. If the answer is a fact the model should know rather
> than a behaviour it should have, the project is misaimed and {{part:12}} is
> the correct chapter.

## 10. Production Considerations

**Provenance is the practical form of homogenisation.** Record which base model,
which version, and which adaptations produced every artefact you serve. When a
defect is announced in a base — and they are announced — the question "does this
affect us?" must be answerable in minutes, and it is only answerable if the
lineage was recorded at build time ({{ch:mle-registry}}).

**Base models are updated underneath you.** A hosted model behind a stable name
may change, and behaviour changes with it. Pin versions where the provider
allows it, keep a regression set, and re-run it on every provider announcement.
This is {{ch:mle-drift}}'s problem with the additional insult that the drift is
someone else's deployment.

**Model choice is a supply-chain decision.** Availability, licence terms, data
residency, and the provider's continued existence are all inputs, and none of
them appear on a benchmark table.

**Budget for the whole pipeline, not the training run.** Evaluation, red-teaming,
and monitoring recur; training does not. Teams routinely budget for the visible
one-off and not the invisible recurring one.

**What to monitor:** cost per conversation, latency at p95, refusal rate,
requests hitting the context limit, and a fixed regression set run on a
schedule. The regression set is the one that catches a base-model change.

## 11. Common Mistakes

**Beginners:**

*Believing a base model is a chat model.* They behave very differently. A base
model continues text; it does not answer. Most surprise at "raw" model behaviour
is this distinction.

*Fine-tuning to add knowledge.* {{eq:adaptation-information-ratio}}: adaptation
carries roughly $10^{-8}$ of pretraining's information. It reshapes; it does not
install. Use retrieval.

*Treating parameter count as capability.* Training tokens, data quality, and
alignment all move quality more per unit than parameters do past a point —
{{cite:hoffmann2022chinchilla}} and {{cite:touvron2023llama}} in
{{ch:fm-scaling-laws}}.

**Experienced practitioners:**

*Comparing models without holding the prompt fixed.* Prompt sensitivity is large
enough to reverse a model ranking, so an unpinned prompt makes the comparison
meaningless. This is {{cite:levy2015}}'s lesson from {{part:8}} in a new costume.

*Trusting a lab's evaluation of its own model.* Not because labs are dishonest,
but because the selection of what to report is itself an unmeasured degree of
freedom. Reproduce anything decision-relevant on your own data.

*Ignoring contamination.* {{cite:lee2022dedup}}'s train/test overlap finding
means a benchmark number may reflect memorisation. Any benchmark whose test set
predates the model's training cutoff deserves suspicion.

*Assuming the pipeline stages are independent.* Alignment can degrade capability
({{ch:fm-rlhf}}'s alignment tax), and distillation inherits everything upstream.
The stages interact, and evaluating one in isolation is optimistic.

## 12. Failure Modes

**Base-model drift.** A hosted model changes and your prompts silently perform
differently. *Symptom:* quality moving with no deployment on your side.
*Detection:* a scheduled regression set. *This is the most common production
surprise with hosted models and the least often instrumented.*

**Inherited defects.** A bias or weakness in the base appears in your product,
and no adaptation you can afford removes it {{eq:homogenisation}}. *Detection:*
evaluate the base directly, not only your adapted system.

**Correlated ecosystem failure.** Everyone built on the same base fails at the
same time on the same input. *Symptom:* invisible in per-application metrics —
the `homogenisation` listing shows precisely this. *Mitigation:* base diversity
for genuinely critical paths, which is expensive and rarely done.

**Capability overhang.** The base can do something the adaptation did not
anticipate — including things you would rather it did not — and a sufficiently
unusual prompt surfaces it. This is the mechanism underneath much of
{{part:26}}: the capability was always there and alignment is a thin layer over
it.

**Contamination-inflated expectations.** The model scores well on a benchmark it
effectively memorised, and production performance does not match. *Detection:*
evaluate on data created after the training cutoff, or on your own.

**Cost surprise at scale.** Per-request costs that are trivial in a pilot are
material at production volume, and the pilot did not include retries, long
contexts, or the retrieval passages. *Detection:* the arithmetic in
`adaptation-cost-ladder`, run before launch rather than after the first invoice.

## 13. Alternatives

{#tbl:model-sourcing caption="Where a model can come from, and what each choice trades. The rows differ far more in operational and legal exposure than in capability, which is why the decision is rarely made on benchmarks."}

| Source | Control | Cost shape | Data exposure | Main risk |
|---|---|---|---|---|
| Hosted API, frontier | none | per token | leaves your boundary | drift, dependency |
| Hosted API, small | none | per token, lower | leaves your boundary | capability ceiling |
| Open weights, self-hosted | full | per GPU-hour | stays inside | utilisation, ops burden |
| Open weights, fine-tuned | full | GPU-hour + setup | stays inside | staleness, maintenance |
| Trained from scratch | total | $10^6$–$10^8$ | total | almost never justified |

**What genuinely differs versus what is merely cheaper.** The first four rows
are the same paradigm at different points on a cost/control curve — all consume
a base someone pretrained. The last is a different activity, and the honest
guidance is that it is justified for approximately no one outside a handful of
labs and a few genuinely unusual domains where no public base has seen the data.

**Continued pretraining is the underused middle.** For a domain with billions of
tokens of in-house text — clinical notes, legal filings, a large proprietary
codebase — continuing pretraining on a public base is far cheaper than training
from scratch and, unlike fine-tuning, does supply enough tokens to move what the
model knows. It sits between rows four and five and is skipped more often than
its economics warrant.

## 14. Evaluation

**Is the adaptation doing what I think?** Ablate it. Compare base + prompt
against base + retrieval against fine-tuned, on the *same* held-out set with the
*same* prompt. Teams routinely deploy a fine-tune without ever measuring against
the prompt-only baseline, and the baseline is sometimes competitive.

**Is this model right for this task?** Task-specific evaluation on your own data.
Public benchmarks shortlist; they do not decide, for the contamination reason in
{{sec:12-failure-modes}} and the prompt-sensitivity reason in
{{sec:11-common-mistakes}}.

**Is the system stable over time?** A fixed regression set on a schedule, with
alerts on movement. For a hosted model this is the only instrument that detects
a provider-side change.

**What kind of evidence is this?** Ask it of every claim in this part. A result
from a lab about its own model, in a paper with no peer review, on a benchmark
that may be contaminated, is still evidence — it is just weaker than the same
result reproduced independently, and the difference should be visible in how you
write it down. {{part:25}} builds this into a discipline.

## 15. Advanced Concepts

**Continued pretraining.** {{maturity:ESTABLISHED}} Extending pretraining on
domain text, at $10^{10}$–$10^{11}$ tokens. Enough information to move knowledge,
unlike fine-tuning; requires care against catastrophic forgetting.

**Model merging.** {{maturity:EMERGING}} Combining the weights of models
fine-tuned from a common base, by averaging or task arithmetic. It works better
than it has any right to, and why is not well understood.

**Mixture of experts as a scaling axis.** {{maturity:ESTABLISHED}} Growing
parameters while holding active parameters per token fixed, which breaks the
$2N$ identity from {{ch:tf-complexity}} and changes what "model size" means.
{{ch:res-moe}} treats it properly.

**Synthetic pretraining data.** {{maturity:EMERGING}}
{{cite:gunasekar2023}} trains on generated, curated text and reaches results the
scaling curves do not predict at that size — evidence that data quality is a
term the laws omit. The contamination concerns it attracted are part of the
result and should be reported with it.

**Capability forecasting.** {{maturity:RESEARCH FRONTIER}} Predicting what a
model will be able to do before training it. {{ch:fm-scaling-laws}} predicts
*loss* reliably and capability poorly, and {{ch:fm-emergence}} is about why that
gap is contested.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-bert}} established pretrain-then-fine-tune and the
budget-equalisation lesson this part needs repeatedly; {{cite:howard2018}} in
{{ch:nlp-contextual}} established the transfer recipe before the architecture
that carried it. {{ch:tf-architectures}} is the decoder-only stack these models
are, and {{ch:tf-complexity}}'s $6ND$ is what makes
{{eq:stage-compute-ratio}} computable. {{ch:dl-autoencoders}} supplied the
proxy-objective pattern that {{eq:pretraining-decomposition}} formalises.
{{ch:mle-registry}} supplied the versioning discipline that becomes provenance.

**Forwards.** {{ch:fm-pretraining}} is stage one in detail;
{{ch:fm-datasets}} is what goes into it; {{ch:fm-scaling-laws}} says how big to
make it; {{ch:fm-emergence}} asks what scaling produces;
{{ch:fm-instruction-tuning}}, {{ch:fm-rlhf}}, and {{ch:fm-dpo}} are the
correction stages; {{ch:fm-distillation}} makes it affordable. {{part:10}}
takes the finished model and follows a prompt through it. {{part:12}} is the
answer to {{eq:adaptation-information-ratio}}, and {{part:14}} measures the
adaptation ladder properly.

## 17. Exercises

**Beginner**

1. Give three ways a base model might respond to "What is the capital of
   France?" and explain why each is a reasonable continuation.
2. Name the deficiency each pipeline stage in {{eq:foundation-pipeline}}
   corrects.
3. Why is in-context learning called learning when no weights change?

**Intermediate**

4. Using {{eq:stage-compute-ratio}}, compute the instruction-tuning fraction for
   a 70B model on $3\times10^{12}$ pretraining tokens and $10^8$ instruction
   tokens.
5. State the homogenisation argument formally and give a concrete defect that
   would propagate, plus one that would not.
6. A team wants the model to know their internal API. Argue against fine-tuning
   using {{eq:adaptation-information-ratio}}, then say what you would do.

**Advanced**

7. {{eq:pretraining-decomposition}} says all progress is the KL term. Explain
   why this does not imply a low-loss model is generally capable, and identify
   the step where the inference fails.
8. Construct a task on which few-shot prompting must beat fine-tuning at equal
   cost, and one where the reverse must hold. State what distinguishes them.
9. Critique {{cite:bommasani2021}}'s definition. Propose a sharper one and give
   a case your version excludes that theirs admits.

**Implementation**

10. Extend `adaptation-cost-ladder` with a break-even analysis between a hosted
    API and self-hosting as a function of daily request volume and GPU
    utilisation. Plot the crossover.
11. Extend `homogenisation` so adaptations vary in strength, and find the
    adaptation fix-rate at which the correlated-failure advantage of base
    diversity disappears.
12. Build a provenance record: a small schema capturing base model, version,
    adaptation type, data hash and evaluation results, plus a query answering
    "which of our deployed systems use base X?"

**Reasoning**

13. Foundation models are claimed to democratise AI by removing the need to
    train. Give the strongest case for, the strongest case against, and say what
    evidence would settle it.
14. Explain why {{eq:adaptation-information-ratio}} predicts retrieval would
    become important, before retrieval is introduced in {{part:12}}.

## 18. Interview Questions

**Beginner**

1. What is a foundation model, and how does it differ from a task-specific one?
2. What is the difference between a base model and an instruction-tuned model?
3. What is in-context learning?

**Intermediate**

4. Walk through the stages from raw corpus to a served chat model, and say what
   each stage corrects.
5. When would you fine-tune, and when would you use retrieval?
6. What is homogenisation and why does it matter for risk?

**Senior**

7. A team proposes fine-tuning on support tickets so the model knows their
   product. Respond.
8. How do you decide between a hosted API and self-hosted open weights?
9. Your provider updated their model and quality changed. How do you detect,
   quantify, and respond?

**Systems**

10. Design provenance tracking for an organisation with fifty LLM-backed
    features across three base models.
11. How would you evaluate a new base model for adoption? What would you not
    rely on?

## 19. Research Questions

**How much of a model's capability is attributable to data versus scale?**
{{cite:gunasekar2023}} suggests the data term is large and the scaling laws omit
it. Design the controlled experiment — same architecture and compute, corpora
differing only in curation — and report what fraction of the gap curation
explains.

**Can the information bound of {{eq:adaptation-information-ratio}} be made
rigorous?** The argument here is a plausibility estimate, not a theorem. What is
the actual channel capacity of fine-tuning, and does it predict the observed
knowledge-versus-style asymmetry quantitatively?

**Does base diversity buy real robustness?** The `homogenisation` listing shows
correlated failure falling with diversity under an independence assumption that
is certainly false — bases share architectures, data sources, and methods.
Measure the true correlation between base models on adversarial probes. If it is
high, diversity buys much less than the simulation suggests.

**What is the actual half-life of a deployed prompt?** Prompts are tuned against
a model version and models change. Measure prompt performance decay across
provider updates. Nobody has published this and every production team is exposed
to it.

## 20. Chapter Summary

A foundation model is trained on broad data at scale with an objective unrelated
to any specific task, and then adapted. The definition {{cite:bommasani2021}}
gives is not a technical threshold — it is a description of an economic
arrangement, in which {{eq:foundation-pretraining}} is paid once by an
organisation with a data centre and {{eq:adaptation}} is paid by everyone else.

The proxy objective produces general capability for a reason that is derivable
rather than mysterious: {{eq:pretraining-decomposition}} shows all progress is
the KL term against the true conditional distribution, and matching that
distribution over a corpus containing arithmetic, code, and dialogue requires
modelling all three. The argument describes the optimum, not any actual model,
and the gap between the two is where most disappointment lives.

**Adaptation cannot install what pretraining lacks.**
{{eq:adaptation-information-ratio}} puts fine-tuning about eight orders of
magnitude below pretraining in information supplied, which is the formal version
of the practitioner's rule: fine-tuning teaches format and style reliably, facts
and skills poorly. This single inequality is why {{part:12}}'s retrieval exists
— it supplies knowledge through the context rather than the weights.

**Each pipeline stage is a patch for the previous objective's deficiency.**
Instruction tuning exists because a text continuer does not answer requests;
alignment exists because "be helpful" is not a loss function; distillation
exists because the result is too expensive. And {{eq:stage-compute-ratio}} shows
instruction tuning is roughly one part in forty thousand of pretraining compute
while producing most of the difference a user perceives — strong evidence these
stages teach behaviour rather than capability.

**Homogenisation is a structural risk, not a speculative one.** When everything
is adapted from a few bases, defects propagate {{eq:homogenisation}} and
adaptations are far too weak to correct them. The `homogenisation` listing shows
the consequence that per-application metrics cannot see: shared bases do not
make any one system less reliable, they make the whole ecosystem fail together.

Finally, the evidentiary habit this part requires: many of its primary sources
have no peer review, several are labs reporting on their own products, and the
most important facts — what is in the training data — are unpublished. Say what
kind of evidence a claim rests on, in the sentence that makes it.

## 21. Further Reading

{{cite:bommasani2021}} is 200 pages and nobody should read all of it. Read §1
for the definition and §1.3 on homogenisation, which is the argument that has
aged best. Read it aware that it is a report from an institute that also builds
these models, and notice how much of it is a research agenda rather than a
result.

{{cite:brown2020}} is the paper that established the paradigm. Read §1 and §3.1,
and note that the striking result is not the benchmark scores — it is that a
model with frozen weights improved from examples in its input, which nobody had
designed and nobody could immediately explain.

{{cite:ouyang2022}} is the other pivotal paper and belongs to
{{ch:fm-rlhf}}, but its abstract is worth reading now for one number: a 1.3B
aligned model preferred to a 175B base. Carry that into the rest of the part.

{{cite:touvron2023llama}} is the clearest short statement of the
inference-aware correction to scaling laws, and the reason open weights exist as
a category. §3 is the training setup.

**Where to go next:** {{ch:fm-pretraining}} opens the box on stage one — what a
pretraining run actually is, as an engineering process rather than an equation.
