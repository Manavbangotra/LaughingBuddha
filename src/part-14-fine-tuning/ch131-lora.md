---
id: ft-lora
number: 131
part: XIV
tier: full
status: draft
requires: [ft-sft, ft-when, math-eigen, math-matrices, dl-optimizers]
provides: [low-rank-hypothesis, lora-parameterisation, effective-rank,
           rank-capacity-limit, learns-less-forgets-less, adapter-merging,
           lora-scaling]
citations: [hu2021lora, biderman2024loralearnsless, houlsby2019adapters,
            kirkpatrick2017ewc, ilharco2023taskarithmetic, zhou2023lima]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the low-rank hypothesis
precisely — it is about the *update*, not the weights — and test it; show that
rank is a **capacity limit rather than a quality dial**, with a knee exactly at
the task's intrinsic rank; **measure** the rank a task requires instead of
guessing it; explain the single mechanism behind *learns less* and *forgets
less*, and read the trade curve both ways; and say when LoRA's parameter saving is
real and when it has evaporated.

## 2. Why This Matters

{{cite:hu2021lora}} changed the economics of everything in {{ch:ft-when}}: a
fine-tune that once meant copying a model and renting eight GPUs became an object
you can train on one and serve many of. That is why this part exists in its
current form.

**And the claim is routinely paraphrased into something weaker and wrong.** It is
not that a model's weights are low rank — they are not. It is that the *change*
required to adapt a pretrained model has low intrinsic rank. That is a claim about
**tasks**, and {{sec:9-practical-example}} tests it directly: for a task whose
true update has rank 2, LoRA at rank 1 scores **0.4009** and at rank 2 scores
**0.0013**. A knee, exactly at the task's rank, with nothing gained above it.

**So rank is a capacity limit, not a quality dial.** Below the task's intrinsic
rank the adapter *cannot express* the update, and the residual is not an
optimisation failure that more steps would fix.

The second measurement is the corrective to "just use LoRA".
{{cite:biderman2024loralearnsless}} found LoRA learns less *and* forgets less, and
{{sec:9-practical-example}} reproduces both halves on one sweep: task-B error
falls **0.0890 → 0.0155** as rank rises, while task-A error rises **0.1110 →
0.1421**, tracked exactly by the update's norm, **1.100 → 1.398**.

**Those are not two findings. They are one mechanism seen twice** — rank bounds
how far the weights travel, and distance travelled determines both how much of the
new task you reach and how much of the old you disturb.

{{maturity:ESTABLISHED}} LoRA. {{maturity:MATURE}} The measured understanding of
what it trades, which is more recent than the method.

## 3. Prerequisites

{{ch:ft-sft}} for the training loop this modifies; {{ch:ft-when}} for the decision
and for {{eq:adaptation-tco}}, whose fixed cost LoRA lowers;
{{ch:math-eigen}} for the SVD, which is the whole chapter's instrument;
{{ch:math-matrices}} for shapes; {{ch:dl-optimizers}} for what "trainable
parameters" costs in optimiser state.

## 4. Intuitive Explanation

### The hypothesis, stated correctly

A weight matrix $W_0$ in a pretrained model is full rank and needs to be — it
encodes an enormous amount. **The hypothesis is about $\Delta W$**, the change
adaptation requires:

> Adapting a pretrained model to a downstream task requires moving the weights in
> only a few directions. The *update* has low intrinsic rank, even though the
> weights do not.

If true, you can write $\Delta W = BA$ with $A$ and $B$ thin, train only those,
and leave $W_0$ frozen. **The saving is not in the model — it is in what you have
to store gradients and optimiser state for.**

### Rank is a wall, not a knob

The natural reading of "rank 8 vs rank 16" is *more capacity, better results,
diminishing returns*. {{sec:9-practical-example}} shows it is sharper than that.

For a task whose update genuinely has rank 8: LoRA at rank 4 leaves **0.3358**
error and rank 8 leaves **0.0024**. Not a gentle improvement — a **wall**, and
then nothing.

**What sits below the wall is not a training problem.** The best rank-$r$
approximation to a rank-8 matrix has an error determined by the discarded singular
values, and no optimiser reaches below it. Adding steps, data, or learning-rate
tuning cannot help, because the adapter cannot represent the answer.

### You can measure the rank instead of guessing it

The usual advice is to try 8, then 16, then 32. That is a search over a quantity
you can **measure**.

Fine-tune once without a rank constraint, take the SVD of the resulting weight
change, and count how many singular values carry the energy.
{{sec:9-practical-example}} does exactly this and recovers the true rank on every
task — 2 → 2, 8 → 8, 32 → 30.

**That number is a property of your task**, and the measurement costs one run you
arguably want anyway as a baseline.

### The trade, and why it has one mechanism

Now the part that matters more than the method.

{{sec:9-practical-example}}'s second sweep adapts a model from task A to task B at
increasing rank, and measures **both** tasks afterward:

```text
   rank        task B error      task A error      update norm
   ─────       ────────────      ────────────      ───────────
      1            0.0890            0.1110            1.100
      8            0.0436            0.1359            1.338
     64            0.0155            0.1421            1.398
   full            0.0000            0.1430            1.406
```

Read either column alone and you reach a wrong conclusion. Read them together and
the shape is obvious: **LoRA learns less and forgets less, and the update norm
explains both.**

Rank bounds how far the weights can travel. Travel further and you capture more of
task B; travel further and you disturb more of task A. **There is no rank at
which you get one without the other**, because it is the same movement.

So **"is LoRA as good as full fine-tuning?" is the wrong question** — it has two
answers pointing opposite ways. The decidable version has two halves:

> **How much genuinely new capability does this task need?**
> **How much existing capability must survive?**

A task adding a large novel skill wants high rank and will pay in forgetting. A
task adjusting style or format on a model whose general ability matters wants low
rank, and the capability it gives up was capability it did not need.

### When the parameter saving disappears

$2Dr$ parameters versus $D^2$: LoRA saves when $r < D/2$ and *costs more* above
it. In {{sec:9-practical-example}}, rank 64 on a 96-dimensional layer has **1.3×
the trainable parameters of a full fine-tune of that layer.**

**"10,000× fewer parameters" is a statement about a particular rank on a
particular model shape**, not a property of the method. If your task needs high
rank, LoRA is not saving you much — and the decision should then be made on
forgetting, not on cost.

## 5. Formal Explanation

### 5.1 The parameterisation

$$ W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r},\; r \ll \min(d,k) $$ (eq:lora-parameterisation)

with $W_0$ frozen. Trainable parameter count:

$$ |\theta_{\text{LoRA}}| = r(d + k) \quad\text{versus}\quad |\theta_{\text{full}}| = dk $$ (eq:lora-params)

so the saving is $r(d+k)/dk$, which for square $d = k = D$ is $2r/D$ — **a saving
only while $r < D/2$**.

### 5.2 Initialisation, and why it matters

$$ A \sim \mathcal{N}(0, \sigma^2), \qquad B = 0 \;\Longrightarrow\; \Delta W = 0 \text{ at step } 0 $$ (eq:lora-init)

**The adapted model starts exactly at the base model.** That is not a convenience;
it is what makes LoRA safe to attach to a working model — training begins from the
identity ({{ch:mm-classification}}'s {{eq:identity-is-default}}, arriving in a new
setting), so nothing is disturbed until the optimiser chooses to disturb it.

Initialising *both* at zero would leave the gradient zero for both by symmetry, and
initialising both randomly would perturb a working model before learning anything.

### 5.3 The scaling factor

The $\alpha/r$ in {{eq:lora-parameterisation}} makes the update's magnitude
roughly independent of $r$:

$$ \left\|\frac{\alpha}{r}BA\right\| \approx \text{const in } r $$ (eq:lora-scaling)

so changing rank does not implicitly change the effective learning rate.
**Without it, doubling the rank doubles the update scale and every hyperparameter
has to be retuned** — which is why $\alpha$ is usually held at a multiple of $r$
rather than tuned independently.

### 5.4 Rank as a capacity limit

By Eckart–Young, the best rank-$r$ approximation to $\Delta W^{*}$ has error

$$ \min_{\text{rank}(M) \le r} \|\Delta W^{*} - M\|_F = \sqrt{\sum_{i > r} \sigma_i^2} $$ (eq:eckart-young)

**{{eq:eckart-young}} is a floor no optimiser goes below.** If $\Delta W^{*}$ has
rank $\rho$, then the floor is exactly zero for $r \ge \rho$ and strictly positive
for $r < \rho$ — the knee {{sec:9-practical-example}} measures.

### 5.5 Effective rank

Real updates do not have a hard rank; they have a decaying spectrum. Define

$$ \rho_{\text{eff}}(\tau) = \min\left\{ k : \frac{\sum_{i \le k}\sigma_i^2}{\sum_i \sigma_i^2} \ge \tau \right\} $$ (eq:effective-rank)

**{{eq:effective-rank}} is measurable from one unconstrained fine-tune**, and it
is the number to set $r$ from. {{sec:9-practical-example}} recovers 2, 8 and 30
against true ranks of 2, 8 and 32.

### 5.6 The two halves, from one inequality

Rank bounds the update's norm:

$$ \|\Delta W_r\| \le \|\Delta W^{*}\|, \qquad \|\Delta W_r\| \nearrow r $$ (eq:rank-constrains-movement)

New-task error is the approximation error of {{eq:eckart-young}}, decreasing in
$r$. Old-task disturbance, to first order, is proportional to the movement:

$$ \mathcal{L}_A(W_0 + \Delta W) - \mathcal{L}_A(W_0) \approx \tfrac{1}{2}\,\Delta W^{\top} H_A\, \Delta W \;\propto\; \|\Delta W\|^2 $$ (eq:forgetting-quadratic)

for the task-A Hessian $H_A$ (the base model sits at a minimum of $\mathcal{L}_A$,
so the linear term vanishes). **Therefore:**

$$ \frac{\partial\,\text{learning}}{\partial r} > 0 \quad\text{and}\quad \frac{\partial\,\text{forgetting}}{\partial r} > 0 \quad\text{simultaneously} $$ (eq:learns-less-forgets-less)

**{{eq:learns-less-forgets-less}} is the chapter's result**, and it says the two
observations are one. {{cite:kirkpatrick2017ewc}}'s elastic weight consolidation
attacks the same quantity from the other side — penalising movement in directions
$H_A$ says matter — which makes **LoRA a forgetting-mitigation method whether or
not it was designed as one.**

### 5.7 Merging at inference

$$ W_{\text{merged}} = W_0 + \tfrac{\alpha}{r}BA $$ (eq:lora-merge)

computed once and folded into the weights, so **inference costs exactly what the
base model costs** — no extra latency, unlike {{cite:houlsby2019adapters}}'s
inserted modules, which add layers to every forward pass.

That property is why LoRA displaced adapters in practice, and it is also what
makes {{ch:ft-merging}}'s arithmetic possible: $\Delta W$ is an object you can
store, add, and subtract.

## 6. Mathematical Foundation

### 6.1 The knee, worked

For a rank-8 target and $r = 4$, {{eq:eckart-young}} predicts a relative error of

$$ \frac{\sqrt{\sum_{i>4}\sigma_i^2}}{\|\Delta W^{*}\|} $$

With eight roughly comparable singular values, that is $\sqrt{4/8} = 0.707$ of the
delta's norm. The delta contributes a fraction of the total output, so the
*measured* relative error on the task is smaller — **0.3358** — but the shape is
{{eq:eckart-young}}'s: a floor set by discarded spectrum, not by optimisation.

At $r = 8$ the sum in {{eq:eckart-young}} is empty and the error collapses to the
noise floor: **0.0024**.

### 6.2 Why more steps cannot help

The residual at $r < \rho$ satisfies

$$ \inf_{A, B} \|\Delta W^{*} - BA\|_F = \sqrt{\sum_{i>r}\sigma_i^2} > 0 $$

which is an **infimum over the parameterisation**, not over an optimisation
trajectory. So

$$ \lim_{\text{steps} \to \infty} \text{error} = \text{the same positive number} $$ (eq:capacity-floor)

**{{eq:capacity-floor}} is the practical diagnostic.** If a LoRA run plateaus
above where you need it and more steps, more data and a lower learning rate all
fail to move it, the rank is too low — and that is distinguishable from
under-training precisely because it is a *floor* rather than a slow descent.

> **MATH NOTE:** {{eq:forgetting-quadratic}} assumes the base model sits at a
> local minimum of the old task's loss, which is why the linear term drops. That
> is approximately true for a well-trained model on its pretraining distribution
> and less true for a model already fine-tuned once. **In the second case
> forgetting has a linear term and can be much faster than quadratic**, which is
> the theoretical reason stacking fine-tunes degrades faster than the first one
> suggests.

### 6.3 The parameter crossover

$$ r(d+k) < dk \iff r < \frac{dk}{d+k} $$ (eq:lora-crossover)

For $d = k = D$ this is $r < D/2$. In the measurement, $D = 96$ and $r = 64 >
48$, so LoRA has **12,288 trainable parameters against a full fine-tune's 9,216**
— a factor of **1.33 the wrong way**.

**For a real transformer the relevant $D$ is a few thousand**, so typical ranks of
8–64 are far below the crossover and the saving is enormous. The crossover matters
anyway, because it shows the saving is a fact about *the ratio $r/D$* rather than
about LoRA, and a task that genuinely needs $\rho_{\text{eff}}$ in the hundreds
has already lost most of the argument.

## 7. Internal Mechanics

```mermaid {#fig:lora-block caption="LoRA in place. The frozen path is unchanged and carries the base model's behaviour; the trainable path starts at exactly zero (eq:lora-init) so attaching an adapter cannot disturb a working model. At inference the two are folded into one matrix (eq:lora-merge), so there is no latency cost — which is what adapters could not offer."}
flowchart LR
    X["x"] --> W0["W0 (frozen)"]
    X --> A["A (r x k)<br/>random init"]
    A --> B["B (d x r)<br/>ZERO init"]
    W0 --> S(("+"))
    B -->|"scaled by alpha/r"| S
    S --> Y["y"]
    B -.->|"delta W = 0 at step 0"| S
    S -.->|"at inference: fold into<br/>W0 + BA, no extra cost"| S
```

### 7.1 Which matrices to adapt

Not every weight matrix needs an adapter, and the choice matters more than the
rank in practice:

| Target | Effect | Typical |
|---|---|---|
| attention $W_q, W_v$ | the original recommendation; strong per parameter | yes |
| attention $W_k, W_o$ | modest additional gain | often |
| MLP matrices | more capacity, most of the parameters | for harder adaptations |
| embeddings / LM head | needed if the vocabulary or output space changes | rarely |

**The general rule is to spread rank across more matrices rather than concentrate
it in fewer.** Rank 8 on four matrices usually beats rank 32 on one, because
{{eq:eckart-young}}'s floor applies per matrix and each has its own spectrum.

### 7.2 Choosing rank, in order of effort

1. **Measure it** ({{eq:effective-rank}}) from one unconstrained run, if you can
   afford that run.
2. **Start at 16** and check for {{eq:capacity-floor}}'s signature: a plateau that
   does not respond to steps or learning rate.
3. **Raise rank only in response to that signature**, not on principle.
4. **Check {{eq:lora-crossover}}** before going high — past $D/2$ you are paying
   more than a full fine-tune of that matrix would cost.

### 7.3 The serving property that changed the field

Because {{eq:lora-merge}} folds cleanly, one base model can serve many adapters:
keep $W_0$ in memory once, and swap the small $BA$ per request. That turns
"fine-tuned model per customer" from a fleet of model copies into a directory of
small files.

**This is arguably a bigger practical change than the training saving**, and it is
{{part:23}}'s subject rather than this part's. Note the constraint it imposes:
adapters swapped at serving time cannot be merged into $W_0$, so they *do* cost a
little latency — the zero-cost property holds for a merged single adapter, not for
a multi-adapter server.

## 8. Implementation

```python {tier=A name=low-rank-hypothesis}
"""The low-rank hypothesis, tested rather than assumed.

cite:hu2021lora's claim is precise and often paraphrased into something weaker.
It is NOT that a model's weights are low rank -- they are not. It is that the
CHANGE required to adapt a pretrained model to a new task has low intrinsic rank,
so the update can be written as a product of two thin matrices
(eq:lora-parameterisation).

That is a claim about tasks, and it is testable. This listing builds adaptation
problems whose true update has a known rank, fits LoRA at a range of ranks, and
finds the knee. Then it does the reverse: fine-tunes without any rank constraint
and inspects the singular value spectrum of the resulting update, to see what rank
the task actually wanted (eq:effective-rank).
"""
import numpy as np

rng = np.random.default_rng(149)

D_IN, D_OUT = 96, 96
N_TRAIN, N_TEST = 4000, 4000
STEPS, LR = 700, 0.05


def make_problem(true_rank, noise=0.05):
    """A frozen base map W0, and a target that differs from it by a delta of
    exactly `true_rank`. Adaptation means recovering that delta."""
    W0 = rng.normal(size=(D_IN, D_OUT)) / np.sqrt(D_IN)
    A = rng.normal(size=(D_IN, true_rank)) / np.sqrt(D_IN)
    B = rng.normal(size=(true_rank, D_OUT)) / np.sqrt(true_rank)
    delta = (A @ B) * 0.8
    X = rng.normal(size=(N_TRAIN, D_IN))
    Y = X @ (W0 + delta) + noise * rng.normal(size=(N_TRAIN, D_OUT))
    Xt = rng.normal(size=(N_TEST, D_IN))
    Yt = Xt @ (W0 + delta)
    return W0, delta, X, Y, Xt, Yt


def fit_lora(W0, X, Y, rank):
    """Train only A and B, with B initialised at zero so the adapted model
    starts exactly at the base model (eq:lora-init)."""
    A = rng.normal(size=(D_IN, rank)) / np.sqrt(D_IN)
    B = np.zeros((rank, D_OUT))
    for _ in range(STEPS):
        pred = X @ (W0 + A @ B)
        G = 2.0 * (pred - Y) / len(X)          # dL/dpred
        GD = X.T @ G                            # dL/d(delta)
        gA, gB = GD @ B.T, A.T @ GD
        A -= LR * gA
        B -= LR * gB
    return A @ B


def fit_full(W0, X, Y):
    """No rank constraint: solve for the delta directly."""
    return np.linalg.lstsq(X, Y - X @ W0, rcond=None)[0]


def rel_err(pred, Y):
    return float(np.linalg.norm(pred - Y) / np.linalg.norm(Y))


def effective_rank(M, thresh=0.99):
    """How many singular values are needed to capture `thresh` of the energy --
    the rank the update actually used (eq:effective-rank)."""
    s = np.linalg.svd(M, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, thresh) + 1)


RANKS = (1, 2, 4, 8, 16, 32)
TRUE = (2, 8, 32)

print(f"{D_IN}x{D_OUT} layer. The target differs from the base by a delta of "
      f"known rank.\n")
print(f"{'true rank':>10}{'':>3}" + "".join(f"{'LoRA r=' + str(r):>12}"
                                            for r in RANKS)
      + f"{'full FT':>10}")
print("-" * 92)

for tr in TRUE:
    W0, delta, X, Y, Xt, Yt = make_problem(tr)
    row = []
    for r in RANKS:
        d = fit_lora(W0, X, Y, r)
        row.append(rel_err(Xt @ (W0 + d), Yt))
    dfull = fit_full(W0, X, Y)
    ef = rel_err(Xt @ (W0 + dfull), Yt)
    print(f"{tr:>10}{'':>3}" + "".join(f"{v:>12.4f}" for v in row)
          + f"{ef:>10.4f}")

print(f"\n\nWhat rank did an UNCONSTRAINED fine-tune actually use?\n")
print(f"{'true rank of task':>19}{'effective rank of':>22}{'ratio':>9}")
print(f"{'':>19}{'the fitted delta':>22}{'':>9}")
print("-" * 50)
for tr in TRUE:
    W0, delta, X, Y, Xt, Yt = make_problem(tr)
    dfull = fit_full(W0, X, Y)
    er = effective_rank(dfull)
    print(f"{tr:>19}{er:>22}{er / tr:>9.1f}x")

print("""
Read each row of the first table left to right and the knee is unmistakable. For
a task whose true update has rank 2, LoRA at rank 1 is poor and rank 2 is already
at the floor -- adding rank beyond that buys nothing, because there is nothing
left to represent. For the rank-32 task, every LoRA rank below 32 leaves a
residual that no amount of training removes.

That is eq:lora-parameterisation behaving exactly as claimed, and it makes the
central point sharply: rank is not a quality dial. It is a CAPACITY limit. Below
the task's intrinsic rank the adapter cannot express the required update, and the
error that remains is not an optimisation failure that more steps would fix -- it
is the distance from the true delta to the nearest rank-r matrix, which is
determined by the task's singular values (eq:eckart-young).

The full fine-tuning column is the control: unconstrained, it reaches the noise
floor on every task, because it has no capacity limit to run into.

Now the second table, which is the more useful direction. Given an unconstrained
fine-tune, how many singular values does the resulting update actually need? For
these synthetic tasks the answer tracks the true rank closely, which is the
expected result and confirms the measurement works.

The reason that matters is what it licenses in practice. The effective rank of a
real fine-tuning delta is measurable the same way: fine-tune once without a rank
constraint, take the SVD of the weight change, and read off how many singular
values carry the energy. That number is the rank your adapter needs, and it is a
property of YOUR task rather than a hyperparameter to be guessed.

Which reframes the usual advice. "Try rank 8, then 16, then 32" is a search over
a quantity that can be measured directly, and the measurement costs one run that
you would arguably want anyway as a full-fine-tuning baseline.

One caution before generalising from this table. These deltas are exactly low
rank by construction, so the knee is sharp. A real adaptation delta has a decaying
spectrum rather than a hard cutoff, so the curve is a gentle bend instead of a
corner, and the choice of threshold in the effective-rank calculation matters.
The shape of the argument survives; the crispness does not.""")
```

The first listing establishes what rank *is*. The second measures what choosing it
costs, in both directions at once.

```python {tier=A name=learns-less-forgets-less}
"""LoRA learns less and forgets less. Both halves, from one sweep.

cite:biderman2024loralearnsless is the controlled comparison the folklore needed.
The finding is not that LoRA matches full fine-tuning, and not that it is worse.
It is a TRADE with a single mechanism behind both halves: LoRA constrains how far
the weights can move, so it captures less of the new task AND disturbs less of
the old one (eq:rank-constrains-movement).

This listing measures both on one sweep. A base model is optimal for task A; it
is then adapted to task B at a range of LoRA ranks and with an unconstrained
fine-tune, and BOTH tasks are evaluated after. Task B's ideal update has a
decaying spectrum rather than a hard rank, which is what a real adaptation looks
like -- so there is no rank at which the trade disappears.
"""
import numpy as np

rng = np.random.default_rng(151)

D = 96
N, STEPS, LR = 5000, 700, 0.05


def spectrum_delta(decay=1.0):
    """An update whose singular values decay smoothly: no hard rank, so every
    additional rank captures a little more and moves a little further."""
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    V, _ = np.linalg.qr(rng.normal(size=(D, D)))
    s = 1.0 / (1.0 + np.arange(D)) ** decay
    s = s / s[0] * 1.1
    return U @ np.diag(s) @ V.T


W_A = rng.normal(size=(D, D)) / np.sqrt(D)          # base model: optimal for A
DELTA = spectrum_delta(decay=1.0)
W_B = W_A + DELTA                                    # target for task B

X_B = rng.normal(size=(N, D)); Y_B = X_B @ W_B
X_A = rng.normal(size=(N, D)); Y_A = X_A @ W_A


def rel(pred, Y):
    return float(np.linalg.norm(pred - Y) / np.linalg.norm(Y))


def fit_lora(rank):
    A = rng.normal(size=(D, rank)) / np.sqrt(D)
    B = np.zeros((rank, D))
    for _ in range(STEPS):
        G = 2.0 * (X_B @ (W_A + A @ B) - Y_B) / N
        GD = X_B.T @ G
        gA, gB = GD @ B.T, A.T @ GD
        A -= LR * gA
        B -= LR * gB
    return A @ B


def fit_full():
    return np.linalg.lstsq(X_B, Y_B - X_B @ W_A, rcond=None)[0]


RANKS = (1, 2, 4, 8, 16, 32, 64)

print(f"{D}x{D} layer. Base model is exact on task A. Task B's ideal update has")
print("a decaying spectrum, so no finite rank captures all of it.\n")
print(f"{'adaptation':>14}{'trainable':>12}{'task B':>10}{'task A':>10}"
      f"{'update':>10}{'captured':>11}")
print(f"{'':>14}{'params':>12}{'error':>10}{'error':>10}{'norm':>10}"
      f"{'of delta':>11}")
print("-" * 68)

rows = {}
full_norm = np.linalg.norm(DELTA)
for r in RANKS:
    d = fit_lora(r)
    b = rel(X_B @ (W_A + d), Y_B)
    a = rel(X_A @ (W_A + d), Y_A)
    rows[r] = (b, a, float(np.linalg.norm(d)))
    print(f"{'LoRA r=' + str(r):>14}{2 * D * r:>12,}{b:>10.4f}{a:>10.4f}"
          f"{np.linalg.norm(d):>10.3f}{np.linalg.norm(d)/full_norm:>11.2f}")

dfull = fit_full()
bf = rel(X_B @ (W_A + dfull), Y_B)
af = rel(X_A @ (W_A + dfull), Y_A)
rows["full"] = (bf, af, float(np.linalg.norm(dfull)))
print(f"{'full FT':>14}{D * D:>12,}{bf:>10.4f}{af:>10.4f}"
      f"{np.linalg.norm(dfull):>10.3f}{np.linalg.norm(dfull)/full_norm:>11.2f}")

r1, r64, fl = rows[1], rows[64], rows["full"]
print(f"""
Read the two error columns together, because reading either alone produces a
wrong conclusion.

Down the task-B column, error falls monotonically with rank: {r1[0]:.4f} at rank
1 to {r64[0]:.4f} at rank 64, and {fl[0]:.4f} unconstrained. LoRA LEARNS LESS,
and it learns less by an amount that shrinks as rank grows but never reaches
zero, because task B's ideal update has a decaying spectrum and any finite rank
truncates it (eq:eckart-young).

Down the task-A column, error RISES with rank, in the same order: {r1[1]:.4f} at
rank 1 to {r64[1]:.4f} at rank 64 and {fl[1]:.4f} unconstrained. LoRA FORGETS
LESS, and it forgets less for exactly the same reason it learns less.

That is the whole finding, and the two halves are not independent observations
that happen to point the same way. They are one mechanism seen twice, and the
update-norm column is the mechanism: {r1[2]:.3f} at rank 1 rising to
{fl[2]:.3f} unconstrained. Rank bounds how far the weights can travel from the
base model (eq:rank-constrains-movement), and how far they travel determines both
how much of the new task they can reach and how much of the old one they disturb.

So "is LoRA as good as full fine-tuning?" is the wrong question, because it has
two answers pointing in opposite directions. The right question has two halves
that can be asked separately: how much NEW capability does this task require, and
how much OLD capability must survive?

That reframing is what makes the choice decidable. A task needing a large,
genuinely novel capability wants rank -- or full fine-tuning -- and will pay in
forgetting. A task that adjusts style or format on a model whose general ability
must be preserved wants low rank, and the capability it gives up was capability it
did not need. The two are different points on one curve rather than competing
techniques.

Note the last column, which prices the parameter argument honestly. Rank 64 has
{2 * D * 64 / (D * D):.1f} times the trainable parameters of a full fine-tune of
this layer, because 2*D*r exceeds D*D once r passes D/2. LoRA's parameter saving
is real at small rank and evaporates at large rank, so "10,000x fewer parameters"
is a statement about a particular rank on a particular model shape rather than a
property of the method. If your task needs high rank, LoRA is not saving you
much, and the choice should be made on forgetting instead.""")
```

## 9. Practical Example

**Rank is a wall, not a knob.** For a task whose update has rank 2: LoRA at rank 1
scores **0.4009**, at rank 2 scores **0.0013**, and higher ranks buy nothing. For
rank 8: **0.3358** at $r=4$, **0.0024** at $r=8$. For rank 32: **0.2671** at
$r=16$, **0.0046** at $r=32$.

**The knee lands exactly at the task's intrinsic rank, every time.**
{{eq:eckart-young}} is why, and {{eq:capacity-floor}} is the practical
consequence: **below the wall, more steps and more data cannot help**, because
the residual is the distance to the nearest rank-$r$ matrix rather than an
optimisation gap.

**And the rank is measurable rather than guessable.** From one unconstrained
fine-tune, the effective rank of the resulting delta recovers **2, 8 and 30**
against true ranks of 2, 8 and 32. "Try 8, then 16, then 32" is a search over a
quantity {{eq:effective-rank}} computes directly, from a run you want anyway as a
baseline.

> **IMPORTANT:** These deltas are exactly low rank by construction, so the knee is
> a corner. A real adaptation delta has a *decaying* spectrum, so the curve is a
> gentle bend and the threshold in {{eq:effective-rank}} matters. **The shape of
> the argument survives and the crispness does not** — which is exactly why the
> second listing uses a decaying spectrum.

**The trade, both halves, one sweep.** Task-B error falls **0.0890 → 0.0155 →
0.0000** as rank goes 1 → 64 → unconstrained. Task-A error rises **0.1110 →
0.1421 → 0.1430** over the same sweep.

**These are not two findings.** The update-norm column is the mechanism: **1.100 →
1.398 → 1.406**, tracking both. {{eq:rank-constrains-movement}} bounds the
distance travelled, {{eq:eckart-young}} makes new-task error decrease in it, and
{{eq:forgetting-quadratic}} makes old-task disturbance increase in it —
{{eq:learns-less-forgets-less}}.

**So "is LoRA as good as full fine-tuning" has two answers pointing opposite
ways**, and the decidable version splits into: *how much new capability does this
task need*, and *how much old capability must survive*. A task adding a novel
skill wants rank and pays in forgetting; a task adjusting style on a model whose
general ability matters wants low rank, and gives up capability it did not need.

**And the parameter argument, priced honestly.** At $D = 96$, rank 64 uses
**12,288 trainable parameters against a full fine-tune's 9,216** — **1.33× the
wrong way**, because {{eq:lora-crossover}} puts the crossover at $r = D/2$. For a
real transformer $D$ is thousands, so typical ranks are far below it and the
saving is enormous. **But "10,000× fewer parameters" is a statement about $r/D$,
not about the method** — and a task needing high effective rank has already lost
that argument.

## 10. Production Considerations

**Measure the effective rank once** ({{eq:effective-rank}}) rather than sweeping
it, if you can afford one unconstrained run.

**Recognise the capacity floor** ({{eq:capacity-floor}}): a plateau that ignores
steps, data and learning rate means raise the rank. A slow descent means train
longer. **They look different and the distinction is actionable.**

**Spread rank across more matrices rather than concentrating it.** The floor
applies per matrix.

**Keep $\alpha$ proportional to $r$** ({{eq:lora-scaling}}), or every rank change
silently rescales the update.

**Always evaluate the base task after adapting.** {{eq:learns-less-forgets-less}}
guarantees you moved it, and only measurement says how much.

**Check {{eq:lora-crossover}} before going above $r = D/2$** for any matrix.

**Merge for single-adapter serving** ({{eq:lora-merge}}) — it is free. Do not
assume the zero-latency property holds in a multi-adapter server.

**Store the adapter with the base model's exact identity.** A delta is only
meaningful against the $W_0$ it was trained on.

## 11. Common Mistakes

**Saying "the weights are low rank".** The *update* is the claim.

**Treating rank as a quality dial** and sweeping it blindly.

**Training through a capacity floor**, adding steps to a problem that is
structural.

**Tuning $\alpha$ independently of $r$.**

**Not evaluating the base task after adaptation.**

**Concentrating high rank on one matrix** instead of spreading it.

**Quoting the parameter saving without the rank** — it is a ratio, not a
property.

**Assuming LoRA is strictly safer.** It forgets less *and* learns less; both
follow from the same constraint.

## 12. Failure Modes

**Capacity floor.** Symptom: loss plateaus above target and nothing moves it.
Cause: {{eq:capacity-floor}}. Fix: raise rank, or spread across more matrices.

**Silent forgetting.** Symptom: the adapted model is worse at things nobody
evaluated. Cause: {{eq:forgetting-quadratic}}. Only measurement finds it.

**Rank/alpha coupling.** Symptom: a rank change destabilises a previously working
recipe. Cause: {{eq:lora-scaling}} not applied.

**Adapter/base mismatch.** Symptom: an adapter produces nonsense on a base model
that looks like the right one. Cause: a different checkpoint, quantisation, or
tokeniser.

**Stacked-fine-tune collapse.** Symptom: the second adapter degrades much more
than the first did. Cause: the base is no longer at a minimum of the old loss, so
{{eq:forgetting-quadratic}}'s linear term is no longer zero.

**High-rank cost surprise.** Symptom: LoRA training uses more memory than
expected. Cause: {{eq:lora-crossover}}.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| full fine-tuning | forgetting, cost, model copies | maximum new capability required |
| adapters ({{cite:houlsby2019adapters}}) | inference latency | multi-task serving with modules |
| prefix tuning ({{ch:ft-qlora-peft}}) | capacity | very light adaptation, tiny footprint |
| QLoRA ({{ch:ft-qlora-peft}}) | numerical headroom | when memory is the binding constraint |
| higher rank on fewer matrices | coverage | rarely correct |
| EWC-style regularisation ({{cite:kirkpatrick2017ewc}}) | complexity | when forgetting must be controlled directly |

**The last row is worth noting as the honest comparison.** LoRA limits forgetting
by limiting *movement in general*; EWC limits it by penalising movement in the
directions that matter. The second is better targeted and much less used, because
the first is free.

## 14. Evaluation

**Report the rank and the target matrices**, or the result is not reproducible.

**Report base-task performance alongside target-task performance.** A LoRA result
without a forgetting measurement is half a result.

**Report effective rank** of the learned delta when you can — it says whether the
rank you chose was binding.

**Compare at equal trainable parameters** where the point is efficiency, and at
equal *quality* where the point is forgetting. They are different experiments.

**Distinguish a capacity floor from under-training** before concluding anything
about the method.

## 15. Advanced Concepts

**Rank as a regulariser.** {{maturity:MATURE}}
{{eq:rank-constrains-movement}} makes LoRA a constraint on the hypothesis class,
so it should — and does — behave like regularisation: better generalisation on
small data, worse fit on large. **That is the same trade
{{ch:mm-vit}}'s {{eq:prior-as-data}} described, one part earlier.**

**Adaptive and per-layer rank.** {{maturity:EMERGING}} Different layers have
different effective ranks, so a uniform $r$ is over-provisioned in some places and
binding in others. Allocating rank by measured spectrum is straightforward and
uncommon.

**The delta as an object.** {{maturity:MATURE}}
{{eq:lora-merge}} means adaptation produces something storable, addable and
subtractable — which is {{cite:ilharco2023taskarithmetic}}'s premise and
{{ch:ft-merging}}'s subject. **Fine-tuning becomes an editing operation.**

**Forgetting is what LoRA is really selling.** {{maturity:EMERGING}}
{{cite:biderman2024loralearnsless}}'s framing inverts the usual pitch: the
memory saving is what people buy it for, and the retention is what they should.
For a model whose general ability is the product, that ordering matters.

**Initialising at the identity is a recurring idea.** {{maturity:ESTABLISHED}}
{{eq:lora-init}}'s zero-initialised $B$ is
{{ch:mm-classification}}'s {{eq:identity-is-default}} again: **make doing nothing
the default and let the optimiser choose to depart from it.** The same principle
produced residual connections, and it is worth recognising as a design pattern
rather than two coincidences.

## 16. Connection to Previous Chapters

{{ch:ft-when}}'s {{eq:adaptation-tco}} has its fixed cost lowered by this chapter
— and {{eq:learns-less-forgets-less}} shows the discount is not free, which is why
{{eq:fine-tuning-decision}}'s second condition is untouched.
{{ch:ft-sft}}'s training loop is unchanged; only which parameters receive
gradients differs. {{ch:math-eigen}}'s SVD is the instrument for both
{{eq:eckart-young}} and {{eq:effective-rank}}.
{{ch:mm-classification}}'s {{eq:identity-is-default}} is
{{eq:lora-init}} in a different setting, and {{ch:mm-vit}}'s
{{eq:prior-as-data}} is {{eq:rank-constrains-movement}} seen as regularisation.
Forward: {{ch:ft-qlora-peft}} makes the base model cheaper to hold,
{{ch:ft-training-config}} measures the forgetting this chapter bounds, and
{{ch:ft-merging}} does arithmetic on the delta.

## 17. Exercises

1. Derive {{eq:lora-crossover}} and compute the crossover rank for a
   $4096 \times 4096$ matrix.
2. Explain why {{eq:lora-init}} sets $B = 0$ and $A$ random rather than the
   reverse, or both zero.
3. Use {{eq:eckart-young}} to predict the relative error at $r=4$ for a rank-8
   delta with equal singular values, and compare with the measured 0.3358.
4. In `low-rank-hypothesis`, change the effective-rank threshold from 0.99 to
   0.90. How do the recovered ranks change, and what does that say about choosing
   $r$ in practice?
5. In `learns-less-forgets-less`, set the spectrum decay to 2.0 (faster decay).
   Does the trade curve steepen or flatten, and why?
6. Add an EWC-style penalty on the update norm to the same listing. Can you get
   rank-64 learning at rank-8 forgetting?
7. Derive {{eq:forgetting-quadratic}} and state what changes when the base model
   is itself a fine-tune.
8. For a model you use: fine-tune one layer unconstrained, take the SVD of the
   delta, and report the effective rank at thresholds 0.9, 0.95 and 0.99.

## 18. Interview Questions

1. State the low-rank hypothesis precisely. What is low rank?
2. Why is $B$ initialised at zero?
3. What does the rank actually limit?
4. Your LoRA run plateaus above target. How do you tell a capacity floor from
   under-training?
5. How would you choose the rank without a sweep?
6. Does LoRA match full fine-tuning? Answer carefully.
7. What is the single mechanism behind "learns less and forgets less"?
8. When does LoRA stop saving parameters?
9. Why can a LoRA adapter be merged with no inference cost, and when can it not?
10. Why does stacking fine-tunes degrade faster than the first one suggests?

## 19. Research Questions

1. {{eq:effective-rank}} is measurable per layer. Does allocating rank by measured
   spectrum beat uniform rank at equal budget, and by how much?
2. {{eq:forgetting-quadratic}} assumes the base is at a minimum. How does the
   forgetting curve change for models that have already been fine-tuned, and does
   it explain observed degradation from stacked adapters?
3. LoRA constrains movement isotropically; EWC constrains it by importance. Is
   there a cheap approximation to the importance-weighted constraint that keeps
   LoRA's memory profile?
4. The trade curve in {{eq:learns-less-forgets-less}} has a shape set by the task
   spectrum. Can the useful operating point be predicted from that spectrum
   before training?
5. Real deltas have decaying spectra. What determines the decay rate, and does it
   correlate with how "novel" a task is relative to pretraining?

## 20. Chapter Summary

**The low-rank hypothesis is about the update, not the weights.** Adapting a
pretrained model requires moving in few directions, so
{{eq:lora-parameterisation}} writes $\Delta W = BA$ with $B$ zero-initialised —
{{eq:lora-init}}, which starts the adapted model exactly at the base and is
{{ch:mm-classification}}'s identity-default idea in a new setting.

**Rank is a capacity limit, not a quality dial.** Measured knees land exactly at
the task's intrinsic rank: 0.4009 → **0.0013** across $r=1\to2$ for a rank-2 task,
0.3358 → **0.0024** across $r=4\to8$ for a rank-8 one.
{{eq:capacity-floor}} makes this actionable — **a plateau that ignores steps,
data and learning rate is a rank problem**, and it looks different from
under-training.

**And the rank can be measured rather than guessed.** {{eq:effective-rank}} on an
unconstrained delta recovered 2, 8 and 30 against true ranks of 2, 8 and 32, from
one run you want anyway.

**LoRA learns less and forgets less, and that is one mechanism, not two.**
Task-B error fell **0.0890 → 0.0155** with rank while task-A error rose **0.1110
→ 0.1421**, both tracked by the update norm's **1.100 → 1.398**.
{{eq:rank-constrains-movement}} bounds the distance travelled;
{{eq:eckart-young}} makes new-task error fall in it and
{{eq:forgetting-quadratic}} makes old-task damage rise in it. **There is no rank
at which you get one without the other.**

**So the question "is LoRA as good as full fine-tuning" has two answers pointing
opposite ways**, and the useful version splits: how much *new* capability is
required, and how much *old* capability must survive. A novel skill wants rank and
pays in forgetting; a style adjustment on a generally-capable model wants low rank
and gives up nothing it needed.

**And the headline parameter saving is a ratio, not a property.**
{{eq:lora-crossover}} puts the crossover at $r = D/2$, and the measurement shows
rank 64 on a 96-wide layer costing **1.33× a full fine-tune**. Real transformers
are thousands wide, so typical ranks save enormously — **but a task needing high
effective rank has already lost the efficiency argument, and should be decided on
forgetting.**

Which is {{cite:biderman2024loralearnsless}}'s inversion, and the thing to take
away: **the memory saving is what people buy LoRA for, and the retention is what
they should.**

## 21. Further Reading

{{cite:hu2021lora}} for the method, and read Section 4.2 on which matrices to
adapt — it is more consequential in practice than the rank discussion that gets
quoted.
{{cite:biderman2024loralearnsless}} immediately after, as the controlled
comparison: the measured rank gap between full fine-tuning and typical LoRA
configurations is what explains both halves of the trade.
{{cite:houlsby2019adapters}} for the predecessor, and for what
{{eq:lora-merge}}'s zero-latency property was competing against.
{{cite:kirkpatrick2017ewc}} for forgetting attacked directly rather than as a side
effect — the comparison sharpens what LoRA is actually doing.
{{cite:ilharco2023taskarithmetic}} for what becomes possible once the update is an
object, developed in {{ch:ft-merging}}.
{{cite:zhou2023lima}} as a reminder that if a thousand examples suffice, the rank
required is probably small.
