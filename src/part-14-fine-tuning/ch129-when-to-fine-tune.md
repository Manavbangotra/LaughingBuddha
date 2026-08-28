---
id: ft-when
number: 129
part: XIV
tier: full
status: draft
requires: [fm-what-they-are, fm-instruction-tuning, llm-prompting, rag-why,
           llm-routing]
provides: [adaptation-ladder, adaptation-tco, churn-penalty,
           format-versus-facts, fine-tuning-decision-rule, adapter-as-delta]
citations: [zhou2023lima, hu2021lora, biderman2024loralearnsless,
            wang2023selfinstruct, lewis2020rag, ouyang2022]
---

## 1. Learning Objectives

By the end of this chapter you will be able to place prompting, few-shot,
retrieval and fine-tuning on one **adaptation ladder** and say what each rung
buys; compute the total cost of ownership of a fine-tune including the term
everyone omits, and find the break-even query volume; show that **churn — not
volume — usually decides the answer**, and by how much; demonstrate that
fine-tuning teaches **format** reliably and **facts** not at all, from a single
training run; and state the decision rule this part will spend eight more chapters
qualifying.

## 2. Why This Matters

Fine-tuning is the most over-prescribed intervention in applied AI. It is what
people reach for when a model is not doing what they want, and it is usually not
the right tool — prompting is cheaper, retrieval is the correct mechanism for
facts, and the underlying problem is frequently that nobody has written down what
"what they want" means.

**The literature cannot help you here, because papers fine-tune by
construction.** Nobody publishes *we tried prompting and it was fine*. So the
published evidence is systematically biased toward the intervention this chapter
exists to talk you out of, most of the time.

Two measurements make the case. {{sec:9-practical-example}} prices the ladder and
finds the break-even at **61,111 queries a year** when requirements never change —
and **1,944,667** when they change weekly, a factor of **32** driven by a variable
with nothing to do with machine learning. **The decision is usually argued on
per-query cost, where fine-tuning wins, and usually decided by churn, where it
often loses.**

The second measurement separates what fine-tuning can teach from what it cannot,
on the same training run. A format rule transferred to unseen inputs at **1.000**.
A fact about a key the model had not seen sat at **0.058** against a chance rate
of **0.040** — flat, at every training budget.

**Not "worse". Not "needs more data". Flat at chance, permanently**, because there
is no rule to generalise. That single table is the argument for {{part:12}}
existing, and for the decision rule this chapter ends on.

{{maturity:ESTABLISHED}} All four rungs. What has changed recently is the *cost*
of the top rung ({{cite:hu2021lora}}), which moves the crossover without changing
its shape.

## 3. Prerequisites

{{ch:fm-what-they-are}} for {{eq:adaptation-information-ratio}} — the reason
adaptation cannot install much that is new; {{ch:fm-instruction-tuning}} for what
the SFT stage does; {{ch:llm-prompting}} for the rung below;
{{ch:rag-why}} for the rung beside it, and for the parametric/non-parametric
decision this chapter re-enters from the weights' side;
{{ch:llm-routing}} for the cost-modelling habit.

## 4. Intuitive Explanation

### The ladder

Four ways to change what a model does, in increasing order of commitment:

| Rung | Changes | Reversible in | Per-query cost |
|---|---|---|---|
| **prompt** | instructions | seconds | tokens, every call |
| **few-shot** | + examples | seconds | more tokens, every call |
| **retrieve** | + fetched context | minutes | tokens + a retrieval |
| **fine-tune** | the weights | a training run | **nothing** |

**Read the last two columns together, because they are the whole trade.** The
first three rungs are paid per query, forever. The fourth is paid once — and then
paid again every time the requirement moves.

### The term nobody budgets

When people compare fine-tuning against prompting, they compare **token cost per
query**, and fine-tuning wins by roughly a factor of ten. That comparison is
correct and it is incomplete, because it prices the wrong axis.

**A prompt change costs an edit and a test run. A weight change costs a
release** — retraining, re-evaluating, and establishing that the new model did not
break behaviour that nobody was testing for. The retraining is the cheap half.

So the real comparison has two variables:

$$ \text{fine-tuning wins} \iff \text{volume is high} \;\textbf{and}\; \text{requirements are stable} $$

and {{sec:9-practical-example}} shows the second condition moving the crossover by
32×. **Before running a training job, count how many times last year someone
changed what the system was supposed to do.**

### What fine-tuning actually teaches

The other half of the decision is about capability rather than cost, and it has a
sharper answer than "fine-tuning helps a bit".

> **A format is a rule.** "Answer in JSON", "always cite", "refuse this
> category" — these depend on the *kind* of request, not its content. Show the
> model each kind a few times and it has the rule, and the rule applies to every
> future input.
>
> **A fact is a lookup.** "Customer 8812's plan is Enterprise." There is no rule
> connecting one customer to the next, so learning four hundred of them tells the
> model nothing about the four hundred and first.

{{sec:9-practical-example}} measures both on one run: format transfers to unseen
inputs at **1.000**, and facts about unseen keys sit at **chance** at every
budget. **Adding data does not move the second column**, because the thing being
asked for does not exist in the data.

**That is why {{cite:zhou2023lima}} could align a model with a thousand
examples** — instruction tuning is mostly format, and format is cheap. And it is
why every new fact needs another training run, which is the churn term that
decides the economics.

### The rule this part will qualify

$$ \textbf{Fine-tune behaviour. Retrieve facts.} $$

Eight chapters follow that qualify it — LoRA changes the cost
({{ch:ft-lora}}), preference optimisation changes what "behaviour" can mean
({{ch:ft-preference}}), and merging changes what you do with the result
({{ch:ft-merging}}). **None of them overturn it.**

## 5. Formal Explanation

### 5.1 The ladder, formally

Each rung is a different place to put the information that adapts the model:

$$ \underbrace{f_\theta(\text{prompt} \Vert x)}_{\text{in the context}} \qquad\text{versus}\qquad \underbrace{f_{\theta + \Delta}(x)}_{\text{in the weights}} $$ (eq:adaptation-ladder)

**{{eq:adaptation-ladder}} is the entire distinction.** Context is per-query,
mutable, and inspectable; weights are amortised, fixed, and opaque. Everything
else follows.

Note what the right-hand side is: a **delta**, $\Delta$. That object has structure
this part will spend chapters on — a rank ({{ch:ft-lora}}), an algebra
({{ch:ft-merging}}), and a size that determines how much of $\theta$ it overwrites
({{ch:ft-training-config}}).

### 5.2 Total cost of ownership

$$ C(\text{mode}) = \underbrace{c_q(\text{mode}) \cdot V}_{\text{per query}} + \underbrace{F(\text{mode}) + n_{\text{changes}} \cdot r(\text{mode})}_{\text{per change}} $$ (eq:adaptation-tco)

with $V$ the annual query volume and $r$ the rework cost per requirement change.
The asymmetry that decides everything:

$$ c_q(\text{fine-tune}) \ll c_q(\text{prompt}), \qquad r(\text{fine-tune}) \ggg r(\text{prompt}) $$ (eq:cost-asymmetry)

### 5.3 The break-even volume

Setting {{eq:adaptation-tco}} equal for fine-tuning and the best alternative:

$$ V^{*} = \frac{F_{\text{ft}} + n_{\text{ch}}\,r_{\text{ft}} - n_{\text{ch}}\,r_{\text{alt}}}{c_q(\text{alt}) - c_q(\text{ft})} $$ (eq:breakeven-volume)

**$V^*$ is linear in the change count**, with slope $r_{\text{ft}}/\Delta c_q$.
Since $r_{\text{ft}}$ is a training run *plus an evaluation rebuild* and
$\Delta c_q$ is a fraction of a token price, that slope is steep:

$$ \frac{\partial V^{*}}{\partial n_{\text{ch}}} = \frac{r_{\text{ft}} - r_{\text{alt}}}{c_q(\text{alt}) - c_q(\text{ft})} $$ (eq:churn-penalty)

{{sec:9-practical-example}} measures $V^*$ going **61,111 → 1,944,667** across 0
to 52 changes a year.

### 5.4 Why format generalises

Let the target decompose into a component determined by the request *type* $t$ and
one determined by the *key* $k$:

$$ y = \big(g(t),\; h(k)\big) $$ (eq:target-decomposition)

For the first component, training covers all $|T|$ types with $|T|$ small, so the
learned $\hat g$ is defined everywhere the model will be used:

$$ \Prob[\hat{g}(t) = g(t)] \to 1 \text{ for all } t, \text{ including inputs with unseen } k $$ (eq:format-generalises)

**Generalisation here is not an achievement — it is the absence of a
generalisation problem.** The function's domain was covered.

### 5.5 Why facts do not

For the second component, $h$ is arbitrary: knowing $h$ on a training set $K$
constrains it nowhere else.

$$ I\big(h(k_{\text{new}});\, \{h(k)\}_{k \in K}\big) = 0 \quad \text{for } k_{\text{new}} \notin K $$ (eq:facts-do-not-generalise)

so

$$ \Prob[\hat{h}(k_{\text{new}}) = h(k_{\text{new}})] = \tfrac{1}{|V|} \quad \text{regardless of } |K| $$ (eq:fact-chance-floor)

**{{eq:fact-chance-floor}} has no data-size term.** Measured: 0.058 against a
0.040 chance rate, unchanged across a 256-fold increase in training.

This is {{ch:fm-what-they-are}}'s {{eq:adaptation-information-ratio}} made
specific. Adaptation carries little information; what it carries is best spent on
a rule that applies everywhere, not on a table that applies once each.

### 5.6 The decision rule

Combining {{eq:breakeven-volume}} with the capability split:

$$ \text{fine-tune} \iff \underbrace{V > V^{*}(n_{\text{ch}})}_{\text{economics}} \;\wedge\; \underbrace{\text{the requirement is a rule}}_{\text{capability}} $$ (eq:fine-tuning-decision)

**Both conditions, not either.** A high-volume stable system whose requirement is
a table of facts should retrieve; a rule-shaped requirement at low volume should
be a prompt.

## 6. Mathematical Foundation

### 6.1 The break-even, worked

With per-query costs of 1.150 (prompt) and 0.250 (fine-tuned), a fine-tuning run
plus evaluation at 55,000, and 60% rework per change:

$$ V^{*}(0) = \frac{55{,}000}{1.150 - 0.250} = 61{,}111 $$ (eq:breakeven-worked)

matching the measurement exactly. At 52 changes:

$$ V^{*}(52) = \frac{55{,}000 + 52(0.6)(55{,}000) - 52(400)}{0.900} = 1{,}944{,}667 $$

**A factor of 32**, and the numerator is dominated by the rework term.

> **MATH NOTE:** The rework fraction of 0.6 is the parameter doing the work here,
> and it deserves scrutiny rather than acceptance. It is not the cost of the
> training job — that is usually the small part. It is the cost of rebuilding and
> rerunning the evaluation that establishes the new weights did not regress
> something nobody was testing for. Teams that have no such evaluation have $r$
> near zero *and* no idea whether their fine-tune broke anything, which is a worse
> position than the one this model prices.

### 6.2 Why the format column is flat at 1.000

With $|T| = 6$ request types and $|F| = 4$ formats, a training set of $n$ examples
covers every type with probability

$$ \Prob[\text{all types seen}] = 1 - \sum_{i} (-1)^{i+1}\binom{|T|}{i}\left(1 - \tfrac{i}{|T|}\right)^{n} \approx 1 \text{ for } n \gtrsim 30 $$ (eq:coupon-coverage)

**At the smallest budget in the sweep the model has seen every type hundreds of
times.** So format accuracy is 1.000 not because the model generalised well but
because there was nothing left to generalise to — the coupon collector finished
long ago.

**That is the honest reading, and it is the useful one**: format is cheap
precisely because its domain is small and enumerable. A "format" with thousands of
distinct cases is not a format, it is a table, and it will behave like the second
column.

### 6.3 The fact column's floor

At $|V| = 25$ values, chance is 0.040. Measured on unseen keys: 0.029, 0.045,
0.041, 0.034, 0.058 across the sweep — scattered around chance with no trend.

Meanwhile *seen* keys go 0.079 → 0.473, which is memorisation behaving as
memorisation does: slow, capacity-limited, and requiring repetition.

$$ \frac{\partial\,\text{acc}_{\text{seen}}}{\partial n} > 0, \qquad \frac{\partial\,\text{acc}_{\text{unseen}}}{\partial n} = 0 $$ (eq:memorise-not-generalise)

**{{eq:memorise-not-generalise}} is the shape to recognise.** If your fine-tune's
held-out accuracy is flat while training accuracy climbs, you are not
under-trained and you do not need more data — **you are asking the model to learn
something that has no structure to learn.**

## 7. Internal Mechanics

```mermaid {#fig:adaptation-ladder caption="The ladder, with the decision variables on the edges. Moving down costs more to change and less to run (eq:cost-asymmetry). The horizontal split is the one that matters more: rules go left into the weights, facts go right into the context, and eq:fine-tuning-decision requires both conditions before the bottom-left box is correct."}
flowchart TB
    Q["a model that is not<br/>doing what you want"] --> W{"is the requirement a<br/>RULE or a FACT?"}
    W -->|"fact"| R["retrieve<br/>(Part XII)"]
    W -->|"rule"| V{"volume x stability<br/>past break-even?<br/>(eq:breakeven-volume)"}
    V -->|"no"| P["prompt, then few-shot"]
    V -->|"yes"| F["fine-tune"]
    P -.->|"cheap to change,<br/>paid every call"| P
    F -.->|"free to run,<br/>paid every change"| F
    R -.->|"new facts need<br/>no training"| R
```

### 7.1 What each rung is genuinely for

| Rung | Genuinely good at | Genuinely bad at |
|---|---|---|
| prompt | behaviour, tone, constraints, iteration speed | consistency under load, token cost |
| few-shot | format demonstration, edge cases | context budget, and it degrades with count |
| retrieve | facts, freshness, attribution, access control | changing *how* the model behaves |
| fine-tune | format, style, refusal policy, output schema | facts, freshness, anything that changes weekly |

**The two middle rows are complements, not alternatives.** Retrieval cannot make a
model answer in your house style and fine-tuning cannot tell it today's inventory.
A system needing both should use both, and most do.

### 7.2 Things fine-tuning is blamed for and does not fix

- **"The model doesn't know our products."** A fact problem. Retrieve.
- **"It hallucinates."** Fine-tuning on correct answers teaches the *shape* of
  confident answers, which can make this worse ({{ch:llm-hallucination}}).
- **"It's too slow."** Fine-tuning a smaller model can genuinely help — but
  {{ch:llm-routing}}'s cascade usually helps more, sooner.
- **"The output format is inconsistent."** This one it does fix, and well. It is
  also fixable with constrained decoding ({{ch:llm-structured-output}}) at no
  training cost.

**Three of those four are not fine-tuning problems**, and the fourth has a cheaper
answer. That ratio is roughly what the ratio of requests looks like in practice.

### 7.3 What to do before deciding

1. **Write the evaluation first.** If you cannot measure the deficiency, you
   cannot tell whether any rung fixed it — and you will pay $r_{\text{ft}}$ every
   change without knowing what you bought.
2. **Try the prompt properly.** Not one attempt: a structured pass with
   {{ch:llm-prompting}}'s techniques, measured on the evaluation.
3. **Classify the requirement.** Rule or fact ({{eq:target-decomposition}}). This
   takes ten minutes and settles most cases.
4. **Count last year's requirement changes.** That is $n_{\text{ch}}$, and it
   moves the answer more than anything else you will measure.

### 7.4 The conversation this chapter is really about

The fine-tuning request rarely arrives as a technical proposal. It arrives as
*"the model isn't good enough — can we train it on our data?"*, which bundles a
diagnosis and a remedy into one sentence and skips the measurement between them.

The productive response is not to argue about fine-tuning. It is to separate that
sentence into its three claims and check each: **what specifically is it getting
wrong, is that a rule or a fact, and how do we know it is not already fixable in
the prompt?** In practice the first question is the one that stalls, because "not
good enough" has usually not been written down as anything a measurement could
confirm or refute.

That is a technical objection rather than a procedural one. Without the deficiency
stated precisely enough to measure, {{eq:target-decomposition}} cannot be applied,
{{eq:adaptation-tco}} has no denominator, and the fine-tune — if it happens — will
be judged by whether it *feels* better, which is a standard no amount of training
can reliably meet. **The evaluation is not a gate in front of the work; it is what
makes the work decidable.**

## 8. Implementation

```python {tier=A name=adaptation-ladder-cost}
"""The adaptation ladder, priced -- including the term everyone leaves out.

Four ways to make a model do what you want, in increasing order of commitment:

  PROMPT      write instructions into every request. Free to change, and you pay
              for those tokens on every call forever.
  FEW-SHOT    add examples to every request. Better behaviour, more tokens.
  RETRIEVE    fetch relevant context per query (Part XII). Handles facts, adds a
              retrieval call.
  FINE-TUNE   move the weights. No per-request overhead at all, one large
              up-front cost -- and a MAINTENANCE cost nobody budgets for
              (eq:adaptation-tco).

The first three are priced per query and the fourth is priced per CHANGE, so the
comparison depends on two numbers: how many queries you serve, and how often the
requirement moves. This listing computes the crossover in both.
"""
import numpy as np

# Token prices in arbitrary but consistent units.
P_IN = 1.0                     # cost per 1k input tokens
BASE_TOKENS = 250              # the actual question
PROMPT_TOKENS = 900            # instructions repeated every call
FEWSHOT_TOKENS = 2600          # instructions + eight examples
RAG_TOKENS = 2200              # instructions + retrieved chunks
RAG_RETRIEVAL = 0.15           # the retrieval call itself, per query

TRAIN_COST = 40_000.0          # one fine-tuning run
EVAL_COST = 15_000.0           # building and running the eval that says it worked
CHANGE_REWORK = 0.6            # fraction of train+eval repaid on each change


def per_query(mode):
    if mode == "prompt":
        return (BASE_TOKENS + PROMPT_TOKENS) / 1000 * P_IN
    if mode == "few-shot":
        return (BASE_TOKENS + FEWSHOT_TOKENS) / 1000 * P_IN
    if mode == "retrieve":
        return (BASE_TOKENS + RAG_TOKENS) / 1000 * P_IN + RAG_RETRIEVAL
    return BASE_TOKENS / 1000 * P_IN          # fine-tuned: no overhead


def total(mode, queries, changes):
    """eq:adaptation-tco: per-query cost times volume, plus per-change cost
    times churn. Only fine-tuning has a meaningful second term."""
    variable = per_query(mode) * queries
    if mode == "fine-tune":
        fixed = TRAIN_COST + EVAL_COST + changes * CHANGE_REWORK * (TRAIN_COST + EVAL_COST)
    else:
        # Changing a prompt is not free either -- someone edits and re-tests it.
        fixed = changes * 400.0
    return variable + fixed


MODES = ("prompt", "few-shot", "retrieve", "fine-tune")
VOLUMES = (10_000, 100_000, 1_000_000, 10_000_000)
CHANGES = (0, 2, 12, 52)

print("cost per query, before any fixed cost:")
for m in MODES:
    print(f"   {m:<12}{per_query(m):>8.3f}")
print(f"\none fine-tuning run costs {TRAIN_COST + EVAL_COST:,.0f} "
      f"(training + the eval that proves it worked)")
print(f"each requirement change repays {CHANGE_REWORK:.0%} of that\n")

for changes in CHANGES:
    label = {0: "requirements never change", 2: "2 changes/year",
             12: "monthly changes", 52: "weekly changes"}[changes]
    print(f"--- {label} ---")
    print(f"{'queries/year':>14}" + "".join(f"{m:>14}" for m in MODES)
          + f"{'winner':>12}")
    for v in VOLUMES:
        costs = {m: total(m, v, changes) for m in MODES}
        win = min(costs, key=costs.get)
        print(f"{v:>14,}" + "".join(f"{costs[m]:>14,.0f}" for m in MODES)
              + f"{win:>12}")
    print()

# Where does fine-tuning start winning, as a function of churn?
# (eq:breakeven-volume, found by bisection rather than the formula so the two
# can be checked against each other.)
print(f"{'changes/year':>14}{'break-even query volume':>26}")
print("-" * 41)
be = {}
for changes in (0, 1, 4, 12, 26, 52):
    lo, hi = 1.0, 1e12
    for _ in range(80):
        mid = (lo + hi) / 2
        best_other = min(total(m, mid, changes) for m in MODES if m != "fine-tune")
        if total("fine-tune", mid, changes) < best_other:
            hi = mid
        else:
            lo = mid
    be[changes] = hi
    print(f"{changes:>14}{hi:>26,.0f}")

print(f"""
The per-query table at the top is where the intuition for fine-tuning comes from,
and it is correct as far as it goes: a fine-tuned model carries no instructions,
no examples and no retrieved context, so its marginal cost is the question alone.
Against few-shot prompting that is a factor of about ten per call, and at high
volume a factor of ten is the whole argument.

Read down the volume rows in the first block and that argument holds. With
requirements that never change, fine-tuning overtakes the alternatives somewhere
in the hundreds of thousands of queries and wins decisively above that. If you
serve a stable, high-volume workload, the case is not close.

Now read across the blocks, because the churn dimension is the one that is
usually missing from the decision. Every requirement change repays a large
fraction of the training and evaluation cost -- not because retraining is
technically hard, but because the EVAL has to be rebuilt and rerun to establish
that the new model did not break the previous behaviour. A prompt change costs an
edit and a test run. A weight change costs a release.

The break-even table makes that concrete. With no churn the crossover sits at
{be[0]:,.0f} queries a year -- a modest system clears it. At weekly changes it is
{be[52]:,.0f}, a factor of {be[52]/be[0]:.0f} higher, and for many real systems
that is past any volume they will ever see. Same technique, same model, same cost
per token, and the recommendation inverts on a variable that has nothing to do
with machine learning.

Which is the practical point of this listing. The fine-tuning decision is usually
argued on per-query cost, where fine-tuning wins, and it is usually DECIDED by
churn, where it often loses. Before running a training job, write down how many
times last year someone changed what the system was supposed to do. That number,
not the token price, is the one that decides.

Two things this model deliberately understates, both in fine-tuning's disfavour.
It does not price the risk that a fine-tune regresses behaviour nobody was
testing for (ch:ft-training-config), and it does not price the fact that a
prompt can be rolled back in seconds while a model cannot. Include those and the
break-even volumes rise further.""")
```

The first listing answers *whether it is worth it*. The second answers *what it
can teach*, which decides the question before cost does.

```python {tier=A name=format-versus-facts}
"""Fine-tuning teaches format reliably and facts unreliably. The same run shows both.

ch:fm-what-they-are argued that adaptation carries far less information than
pretraining, so a fine-tune cannot install much that is new. This listing makes
the distinction sharper than "less": it separates what fine-tuning teaches into
two kinds and shows they behave completely differently on the SAME training run.

  FORMAT   a rule that applies to every input -- answer in JSON, always cite,
           refuse this category. Depends on the request TYPE, not its content, so
           one example teaches it for all inputs (eq:format-generalises).
  FACTS    a mapping from a specific key to a specific value. There is no rule to
           learn; each pair must be memorised, and memorising one says nothing
           about the next (eq:facts-do-not-generalise).

A tiny model is trained on inputs carrying both, and evaluated on keys it saw and
keys it did not.
"""
import numpy as np

rng = np.random.default_rng(131)

N_KEY, N_TYPE = 400, 6
N_FORMAT, N_VALUE = 4, 25
D_EMB, D_HID = 24, 64
EPOCHS, LR = 60, 0.08

# The two targets. FORMAT depends only on the request type; VALUE depends only
# on the key, and is an arbitrary lookup with no structure to generalise from.
format_of_type = rng.integers(0, N_FORMAT, size=N_TYPE)
value_of_key = rng.integers(0, N_VALUE, size=N_KEY)

# Keys the model will be trained on, and keys held out entirely.
perm = rng.permutation(N_KEY)
SEEN, UNSEEN = perm[:N_KEY // 2], perm[N_KEY // 2:]


def make_batch(keys, n):
    k = rng.choice(keys, size=n)
    t = rng.integers(0, N_TYPE, size=n)
    return k, t, format_of_type[t], value_of_key[k]


class Model:
    """Key embedding + type embedding -> hidden -> two heads. The only way to
    get VALUE right is to store it in the key embedding; FORMAT can be read off
    the type embedding alone."""

    def __init__(self):
        self.Ek = rng.normal(scale=0.3, size=(N_KEY, D_EMB))
        self.Et = rng.normal(scale=0.3, size=(N_TYPE, D_EMB))
        self.W1 = rng.normal(scale=np.sqrt(2 / (2 * D_EMB)), size=(2 * D_EMB, D_HID))
        self.b1 = np.zeros(D_HID)
        self.Wf = rng.normal(scale=np.sqrt(2 / D_HID), size=(D_HID, N_FORMAT))
        self.bf = np.zeros(N_FORMAT)
        self.Wv = rng.normal(scale=np.sqrt(2 / D_HID), size=(D_HID, N_VALUE))
        self.bv = np.zeros(N_VALUE)

    def forward(self, k, t):
        self.k, self.t = k, t
        self.x = np.hstack([self.Ek[k], self.Et[t]])
        self.h = np.maximum(self.x @ self.W1 + self.b1, 0)
        return self.h @ self.Wf + self.bf, self.h @ self.Wv + self.bv

    def step(self, gf, gv, lr):
        gWf, gbf = self.h.T @ gf, gf.sum(0)
        gWv, gbv = self.h.T @ gv, gv.sum(0)
        gh = (gf @ self.Wf.T + gv @ self.Wv.T) * (self.h > 0)
        gW1, gb1 = self.x.T @ gh, gh.sum(0)
        gx = gh @ self.W1.T
        np.add.at(self.Ek, self.k, -lr * gx[:, :D_EMB])
        np.add.at(self.Et, self.t, -lr * gx[:, D_EMB:])
        for p, g in ((self.W1, gW1), (self.b1, gb1), (self.Wf, gWf),
                     (self.bf, gbf), (self.Wv, gWv), (self.bv, gbv)):
            p -= lr * g


def ce(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return g / len(y)


def evaluate(m, keys):
    k, t, yf, yv = make_batch(keys, 4000)
    lf, lv = m.forward(k, t)
    return float((lf.argmax(1) == yf).mean()), float((lv.argmax(1) == yv).mean())


print(f"{N_KEY // 2} keys seen in training, {N_KEY // 2} never seen.")
print(f"FORMAT is a function of the request type ({N_TYPE} types, "
      f"{N_FORMAT} formats).")
print(f"FACT is a function of the key ({N_KEY} keys, {N_VALUE} values, "
      f"chance = {1/N_VALUE:.3f}).\n")
print(f"{'examples per seen key':>23}{'':>3}{'FORMAT':>18}{'':>4}{'FACT':>18}")
print(f"{'':>23}{'':>3}{'seen':>9}{'unseen':>9}{'':>4}{'seen':>9}{'unseen':>9}")
print("-" * 76)

rows = {}
for per_key in (1, 4, 16, 64, 256):
    m = Model()
    n_batch = max(per_key * len(SEEN) // EPOCHS, 32)
    for _ in range(EPOCHS):
        for _ in range(max(n_batch // 128, 1)):
            k, t, yf, yv = make_batch(SEEN, 128)
            lf, lv = m.forward(k, t)
            m.step(ce(lf, yf), ce(lv, yv), LR)
    fs, vs = evaluate(m, SEEN)
    fu, vu = evaluate(m, UNSEEN)
    rows[per_key] = (fs, fu, vs, vu)
    print(f"{per_key:>23}{'':>3}{fs:>9.3f}{fu:>9.3f}{'':>4}{vs:>9.3f}{vu:>9.3f}")

lo, hi = rows[1], rows[256]
print(f"""
Read the two FORMAT columns first, and read them together. They are
approximately equal at every training budget -- {hi[0]:.3f} on keys the model
trained on and {hi[1]:.3f} on keys it has never seen. The format rule transferred
completely to inputs that were not in the training set, because there was a rule
to learn: format depends on the request type, and the types were all covered
(eq:format-generalises).

That is what makes format cheap. Once the model has seen each request type a few
times, it has the rule, and the rule applies to every future input regardless of
content. This is why instruction tuning works on a thousand examples
(cite:zhou2023lima) and why "always answer in this shape" is the thing
fine-tuning is genuinely good at.

Now the FACT columns, which behave in the opposite way. On seen keys, accuracy
climbs with repetition -- {lo[2]:.3f} at one example per key up to {hi[2]:.3f} at
256 -- which is memorisation working as memorisation does. On UNSEEN keys it sits
at {hi[3]:.3f} against a chance rate of {1/N_VALUE:.3f} and does not move at any
budget.

Not "worse". Not "needs more data". Flat at chance, permanently, because there is
nothing to generalise (eq:facts-do-not-generalise). The mapping from key to value
is arbitrary by construction -- as facts about your customers, your inventory or
your policies are arbitrary with respect to each other -- so learning four hundred
of them tells the model nothing whatsoever about the four hundred and first.

Those two behaviours from one training run are the whole argument for the
architecture of the previous two parts. If a capability is a RULE, fine-tuning
installs it cheaply and it generalises. If it is a FACT, fine-tuning memorises the
ones you showed it, generalises to none, and every new fact requires another
training run -- which is exactly the churn term that this chapter's first listing
shows decides the economics.

Retrieval has the complementary shape: it is poor at changing behaviour and
excellent at supplying a fact that was never in the weights, including one created
this morning. So the two are not competitors to be ranked. They address the two
halves of this table, and a system that needs both should use both -- fine-tune
the format, retrieve the facts.""")
```

## 9. Practical Example

**Per-query cost says fine-tune.** A fine-tuned model carries no instructions, no
examples and no retrieved context: **0.250 against few-shot's 2.850**, a factor of
about eleven. At high volume that is the whole argument, and with stable
requirements it holds — fine-tuning wins from the hundreds of thousands of
queries upward.

**Churn says otherwise, and by a lot.** The break-even volume goes **61,111** with
no requirement changes to **1,944,667** at weekly changes — a factor of **32**.
{{eq:breakeven-worked}} reproduces the first exactly, and
{{eq:churn-penalty}} explains the slope: each change repays a training run *plus
an evaluation rebuild*, divided by a fraction of a token price.

> **IMPORTANT:** The rework cost is not the training job — that is the cheap half.
> It is rebuilding and rerunning the evaluation that establishes the new weights
> did not regress behaviour nobody was testing for. **A prompt change costs an
> edit and a test run; a weight change costs a release.** And a team with no such
> evaluation has a low rework cost *and* no idea whether their fine-tune broke
> anything — a worse position than the one this model prices.

**So the decision is usually argued on the axis where fine-tuning wins and decided
on the axis where it often loses.** Before a training job, count last year's
requirement changes.

**And the capability question settles most cases before cost is reached.** On one
training run: **format transferred to unseen inputs at 1.000** at every budget,
while **facts about unseen keys sat at 0.058 against a chance rate of 0.040** —
scattered around chance with no trend across a 256-fold increase in training data.

**Meanwhile facts about *seen* keys climbed 0.079 → 0.473.**
{{eq:memorise-not-generalise}}: training accuracy rising while held-out accuracy
stays flat. **That shape means you are not under-trained and do not need more
data — you are asking the model to learn something with no structure to learn.**

The honest reading of the format column, which {{eq:coupon-coverage}} supplies: it
is 1.000 not because the model generalised impressively but because **its domain
was small and fully covered**. Which is exactly the point — format is cheap
because it is enumerable. **A "format" with thousands of distinct cases is not a
format, it is a table, and it will behave like the second column.**

## 10. Production Considerations

**Write the evaluation before choosing a rung.** Without it you cannot tell what
any intervention bought, and you will pay the rework cost forever without knowing.

**Classify the requirement as rule or fact** ({{eq:target-decomposition}}). Ten
minutes, and it settles most cases.

**Count requirement changes over the last year.** That is $n_{\text{ch}}$, and
{{eq:churn-penalty}} makes it the dominant term.

**Try the prompt properly first** — a structured pass, measured, not one attempt.

**Do not fine-tune facts.** {{eq:fact-chance-floor}} has no data-size term.

**If you fine-tune, keep the training data and the eval under version control
together.** The next change needs both, and the eval is the expensive one.

**Prefer constrained decoding for output schemas** ({{ch:llm-structured-output}}) —
same result, no training, instant rollback.

**Budget the rollback.** A prompt reverts in seconds; weights require a
deployment. That asymmetry belongs in the decision, not in the incident review.

## 11. Common Mistakes

**Comparing on per-query cost alone**, which is the axis fine-tuning wins.

**Fine-tuning to install facts.** The most common and most expensive error in this
part.

**Concluding "we need more data" from flat held-out accuracy.**
{{eq:memorise-not-generalise}} says otherwise.

**Fine-tuning before trying a real prompting pass.**

**Having no evaluation, then fine-tuning.** You have bought something unmeasurable
and a recurring bill.

**Treating retrieval and fine-tuning as competitors** rather than as the two
halves of {{eq:target-decomposition}}.

**Forgetting that a fine-tune is a delta with side effects** — what it overwrites
is {{ch:ft-training-config}}'s subject and is rarely measured.

## 12. Failure Modes

**The fact fine-tune.** Symptom: excellent on the training set, chance on anything
new, and each new fact needs a training run. Cause:
{{eq:facts-do-not-generalise}}.

**Churn bankruptcy.** Symptom: a fine-tuning programme that never catches up with
requirement changes. Cause: {{eq:churn-penalty}}, unmeasured.

**Silent regression.** Symptom: the fine-tuned model is better at the target task
and worse at something nobody tested. {{ch:ft-training-config}}.

**Evaluation-free adaptation.** Symptom: nobody can say whether the fine-tune
helped. The most common state of affairs.

**Prompt abandonment.** Symptom: a fine-tune deployed because prompting "didn't
work", where prompting was tried once, unsystematically.

**Format inflation.** Symptom: a "format" fine-tune that keeps needing retraining
because the format has hundreds of special cases — it was a table.

## 13. Alternatives

| Instead of fine-tuning | When it wins |
|---|---|
| better prompting ({{ch:llm-prompting}}) | almost always, first |
| constrained decoding ({{ch:llm-structured-output}}) | output schemas — same result, no training |
| retrieval ({{ch:rag-why}}) | anything that is a fact, or changes |
| a cascade ({{ch:llm-routing}}) | cost and latency, sooner than a fine-tune would |
| few-shot with a longer context | small volumes, fast iteration |
| LoRA rather than full fine-tuning ({{ch:ft-lora}}) | when you do fine-tune — cheaper, and forgets less |

**The last row is this part's rest**, and it changes the cost of the top rung
without changing the shape of the decision:
{{eq:fine-tuning-decision}}'s second condition is unaffected by how cheaply you can
train.

## 14. Evaluation

**Measure the deficiency before intervening**, and use the same measurement after.

**Evaluate held-out and training performance separately**, and watch for
{{eq:memorise-not-generalise}}'s divergence — it is the diagnostic that says stop.

**Evaluate general capability, not only the target task.** A fine-tune's cost
includes what it broke.

**Report the total cost of ownership**, not the training cost
({{eq:adaptation-tco}}).

**Re-run the comparison after each requirement change.** The break-even moves, and
it moves against fine-tuning.

## 15. Advanced Concepts

**The ladder is a spectrum, not four rungs.** {{maturity:MATURE}} Prefix tuning
({{ch:ft-qlora-peft}}) learns a *continuous prompt*, sitting between rungs one and
four, which shows that "in the context" and "in the weights" is a gradient rather
than a dichotomy.

**A fine-tune is a delta with algebraic structure.**
{{maturity:EMERGING}} {{eq:adaptation-ladder}}'s $\Delta$ can be added, scaled and
subtracted ({{ch:ft-merging}}), which makes "adapt the model" closer to an editing
operation than a training one.

**LoRA changes the cost, not the decision.** {{maturity:MATURE}}
{{cite:hu2021lora}} lowers $F_{\text{ft}}$ and $r_{\text{ft}}$, which moves
$V^{*}$ down — and {{cite:biderman2024loralearnsless}} shows it also lowers what
you get. **Both terms move**, which is why {{ch:ft-lora}} is a chapter rather than
a footnote.

**Synthetic data lowers the data barrier and raises a different one.**
{{maturity:EMERGING}} {{cite:wang2023selfinstruct}} makes instruction data a
compute problem, which removes the usual reason not to fine-tune — and introduces
the diversity problem {{ch:ft-synthetic}} measures.

**The evaluation is the durable asset.** {{maturity:ESTABLISHED}} Models are
replaced, prompts are rewritten, adapters are retrained. **The evaluation set
outlives all of them**, and it is the artefact that makes any of these decisions
answerable — the same argument {{ch:mm-ocr}} made about the text layer, applied to
process.

## 16. Connection to Previous Chapters

{{ch:fm-what-they-are}}'s {{eq:adaptation-information-ratio}} is what
{{eq:facts-do-not-generalise}} makes concrete: adaptation carries little
information, and what it carries is best spent on a rule rather than a table.
{{ch:fm-instruction-tuning}} is the format half of this chapter's split, done at
scale. {{ch:rag-why}}'s parametric/non-parametric decision is the same decision
approached from the other side — that chapter asked where knowledge should live,
this one asks whether to move it. {{ch:llm-routing}}'s cost modelling is
{{eq:adaptation-tco}}'s method. Forward: {{ch:ft-sft}} does the format half
properly, {{ch:ft-lora}} changes $F_{\text{ft}}$, and
{{ch:ft-training-config}} prices what {{eq:adaptation-ladder}}'s $\Delta$ destroys.

## 17. Exercises

1. Derive {{eq:breakeven-volume}} from {{eq:adaptation-tco}} and verify
   {{eq:breakeven-worked}}'s 61,111 by hand.
2. In `adaptation-ladder-cost`, set `CHANGE_REWORK = 0.1` (a team with excellent
   automated evaluation). How far does the break-even move at 52 changes?
3. Add a fifth mode: a LoRA fine-tune at one fifth the training cost. At what
   churn does it beat prompting?
4. Prove {{eq:fact-chance-floor}} and state the assumption about $h$ that it
   depends on.
5. In `format-versus-facts`, make the value a *function* of the key's index
   (say, $k \bmod 25$) rather than arbitrary. What happens to the unseen column,
   and what does that tell you about the rule/fact distinction?
6. Increase `N_TYPE` to 500 in the same listing. Does format still generalise, and
   what does {{eq:coupon-coverage}} predict?
7. Using {{eq:fine-tuning-decision}}, classify five real requirements from a
   system you know. How many pass both conditions?
8. Estimate $r_{\text{ft}}$ for your own team by timing one full retrain-and-
   revalidate cycle. Compare with the 0.6 fraction used here.

## 18. Interview Questions

1. When should you fine-tune, and when should you not?
2. Why is per-query cost the wrong axis to decide on?
3. What does fine-tuning teach reliably, and what does it not?
4. Your fine-tune has high training accuracy and flat held-out accuracy. What is
   the diagnosis, and does more data help?
5. Why did LIMA work with a thousand examples?
6. A stakeholder wants the model to "know our product catalogue". What do you
   build?
7. What is the maintenance cost of a fine-tune, and what dominates it?
8. Does LoRA change the fine-tuning decision? Which parts?
9. What would you do before running any training job?
10. Why are retrieval and fine-tuning not competitors?

## 19. Research Questions

1. {{eq:adaptation-tco}}'s rework fraction is estimated rather than measured. What
   does a real distribution of $r_{\text{ft}}$ look like across teams, and what
   predicts it?
2. {{eq:target-decomposition}} splits requirements cleanly into rules and facts.
   Real requirements are mixtures — is there a cheap classifier that estimates the
   mixture from a specification?
3. {{eq:fact-chance-floor}} assumes an arbitrary mapping. Real facts have partial
   structure (customers in a segment behave alike). How much structure is needed
   before fine-tuning beats retrieval?
4. Prefix tuning sits between context and weights. Is there a principled account of
   where a given capability *should* live on that spectrum?
5. The evaluation is the durable asset. Can evaluation sets be made to survive
   requirement changes, or does churn invalidate them at the same rate?

## 20. Chapter Summary

**Fine-tuning is the most over-prescribed intervention in applied AI**, and the
literature cannot correct that because papers fine-tune by construction.

**The economics are usually argued on the wrong axis.** Per-query cost favours
fine-tuning by roughly eleven to one, and that is real — but
{{eq:adaptation-tco}} has a second term, and {{eq:churn-penalty}} makes it
dominant. Measured: break-even at **61,111** queries a year with stable
requirements and **1,944,667** at weekly changes, a factor of **32**. The rework
is not the training job; it is rebuilding the evaluation that proves nothing
regressed. **A prompt change costs a test run; a weight change costs a release.**

**And the capability question settles most cases before cost is reached.** From
one training run: format transferred to unseen inputs at **1.000**, and facts
about unseen keys sat at **chance** — 0.058 against 0.040 — with **no trend across
a 256-fold data increase**. {{eq:fact-chance-floor}} contains no data-size term
because there is nothing to generalise.

**The diagnostic shape to recognise** is {{eq:memorise-not-generalise}}: training
accuracy climbing while held-out accuracy stays flat. That does not mean
under-trained and it does not mean more data. **It means the thing being asked for
has no structure to learn.**

**And the reason format is cheap is less flattering than it looks.**
{{eq:coupon-coverage}}: its domain is small and was fully covered, so there was
nothing left to generalise to. Which gives the useful warning — **a format with
thousands of special cases is a table**, and it will behave like the fact column.

So the rule, which the next eight chapters qualify and none overturn:

$$ \textbf{Fine-tune behaviour. Retrieve facts.} $$

with {{eq:fine-tuning-decision}}'s two conditions — **volume times stability past
break-even, and a requirement that is a rule** — both required, not either.

## 21. Further Reading

{{cite:zhou2023lima}} for the thousand-example result, and read its
interpretation rather than its number: the claim is that pretraining already
supplied the knowledge and alignment selects a format.
{{cite:ouyang2022}} for what instruction tuning at scale actually changed, which
is the format half of this chapter's split.
{{cite:lewis2020rag}} for the other half, and {{ch:rag-why}} for the decision
approached from the knowledge side rather than the weights side.
{{cite:hu2021lora}} for what lowered the cost of the top rung, and
{{cite:biderman2024loralearnsless}} immediately after, because it prices what the
discount costs — the pair is the honest version of "just use LoRA".
{{cite:wang2023selfinstruct}} for why the data barrier fell, developed in
{{ch:ft-synthetic}} along with the problem it created.
