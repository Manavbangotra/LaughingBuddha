---
id: ft-merging
number: 137
part: XIV
tier: full
status: draft
requires: [ft-training-config, ft-lora, ft-qlora-peft, dl-initialization]
provides: [linear-mode-connectivity, shared-base-precondition, task-arithmetic,
           conflict-governs-merging, merge-scale-coefficient, soup-versus-merge,
           distillation-as-transfer]
citations: [wortsman2022modelsoups, ilharco2023taskarithmetic, yadav2023ties,
            hinton2015, hu2021lora]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state merging's precondition
operationally and recognise the three ways it is violated; distinguish a **model
soup** from a **task merge**, which share a name and almost nothing else; predict
merge quality from a measurable property of the tasks before merging; explain why
no combining rule fixes genuinely conflicting tasks; and place distillation
correctly as the technique for the case merging cannot reach.

## 2. Why This Matters

{{ch:ft-training-config}} ended with an escape hatch: if the exchange rate is
unacceptable, keep two models. **This chapter is about whether you can put them
back together**, and it is the part of the fine-tuning literature most surrounded
by folklore.

{{cite:wortsman2022modelsoups}} reported that averaging fine-tuned models produces
something better than any of them, which has no right to work — a network's
function is unchanged by permuting hidden units, so two independently trained
networks put the same feature in different coordinates and averaging puts it
nowhere.

**It works because of a precondition that gets dropped in retelling.**
{{sec:9-practical-example}} measures both sides: fine-tunes from a **shared base**
merge to a joint loss of **0.1997**; models trained from **independent
initialisations** merge to **0.3869** — **1.9× worse** on the same tasks,
architecture and budget.

**And the sharper diagnostic is where each optimum sits.** The shared-base optimum
is at $\alpha = 0.5$, genuinely combining. The independent optimum is at
$\alpha = 0.875$ — essentially an endpoint. **That is what a failed merge looks
like: not an error, an optimum that quietly degenerates to "do not merge".**

**Then the number that makes merging worth a chapter.** The shared-base merge
reaches **0.1997** against a model *trained jointly on both tasks* at **0.1851** —
**within 8%**, for a weighted sum over two checkpoints you already had.

**And the limit, from a controlled sweep.** With tasks constructed from closely
related (ρ = 0.9) to directly opposed (ρ = −0.9), merge quality runs **0.0515 →
0.4208** — an **eightfold** degradation — while the specialists stay flat at
**0.0155 → 0.0156.**

**Merge quality is governed by whether the tasks agree, and by nothing else you
control.**

{{maturity:MATURE}} Weight averaging, task arithmetic.
{{maturity:EMERGING}} Merging as a routine deployment step.
{{maturity:ESTABLISHED}} Distillation.

## 3. Prerequisites

{{ch:ft-training-config}} for why you might have two models rather than one;
{{ch:ft-lora}} for the delta as an object ({{eq:lora-merge}}) — this chapter is
that idea taken seriously; {{ch:ft-qlora-peft}} for {{eq:qlora-forward}}'s base
identity, which becomes a merging constraint; {{ch:dl-initialization}} for why two runs of the same architecture
land in different places at all.

## 4. Intuitive Explanation

### The precondition, and why it is the whole story

Averaging two neural networks should be catastrophic. Permutation symmetry means
two independently trained networks encode the same features in different
coordinates, so coordinate-wise averaging averages unrelated things.

**Fine-tuning from a shared base never leaves the neighbourhood where the
coordinates mean the same thing.** The two fine-tunes stay in one low-loss basin,
and the segment between them stays inside it.

{{sec:9-practical-example}} measures the difference at **1.9×**, and the shape of
the failure is the useful part: the independent merge's best point sits at an
endpoint. **The optimiser's answer to "how should I combine these?" is "don't".**

> **So state the precondition operationally**: merging works between models that
> **share an ancestor** and **have not travelled far from it.**

### Three violations that look innocuous

- **Same architecture, different pretraining runs.** Not a shared base. This is
  the most common one, because the models look identical in every metadata field
  that gets checked.
- **A model fine-tuned twice in sequence.** It has travelled further than the
  merge assumes; its delta is no longer small.
- **A quantised copy.** {{ch:ft-qlora-peft}}'s {{eq:qlora-forward}} showed an
  adapter is fitted to the *quantised* base. A merge across a precision change is
  a merge across different bases.

**Essentially every merge failure reported in practice is one of these**, rather
than a subtlety of the merging algorithm.

### Soup and merge are different techniques with one name

This distinction is routinely blurred and it matters:

| | **Model soup** | **Task merge** |
|---|---|---|
| ingredients | same task, different hyperparameters | different tasks |
| what averaging does | cancels independent noise | combines skills |
| result vs ingredients | **better than each** | worse than each on its own task |
| why | variance reduction | one model instead of two |

**{{cite:wortsman2022modelsoups}}'s headline result is the first column.** The
measurement here is the second: the merged model is worse on task A than model A
and worse on task B than model B. **Merging did not produce a model better at
everything. It produced one model instead of two, at a measured cost on each.**

Only the first is free. Quoting the first to justify the second is the single most
common error in this area.

### What the merge is actually worth

The right comparison is not against the specialists — it is against **training one
model on both tasks**, which is what a merge substitutes for.

**Merged: 0.1997. Jointly trained: 0.1851. Within 8%**, for arithmetic that takes
milliseconds on checkpoints you already have.

**That is the argument for merging, and it is an argument about cost rather than
quality.** If you were going to run the multi-task training anyway, run it. If you
have two fine-tunes and a serving budget for one model, the merge recovers most of
what you would have got.

### Conflict is the limit, and it is measurable in advance

{{sec:9-practical-example}}'s second sweep constructs task pairs from aligned to
opposed:

```text
   relatedness   sign conflict   specialists   merged
   ───────────   ─────────────   ───────────   ──────
        +0.9          18.0%          0.0155    0.0515
        +0.5          19.4%          0.0160    0.1362
         0.0          26.8%          0.0160    0.2419
        -0.5          30.2%          0.0154    0.3521
        -0.9          26.2%          0.0156    0.4208
```

**The specialists column is flat.** Each task is equally learnable regardless of
the other. **The merged column degrades eightfold.**

The reason is a limit rather than a technique. **Where two tasks require the same
parameter to move in opposite directions, no combining rule satisfies both,
because no single value satisfies both.** Averaging splits the difference and
implements neither; electing a sign implements one and abandons the other.

### The diagnostic, with its limitation

The sign-conflict rate is a cheap proxy — an elementwise comparison of two tensors
you already have. It **does** rise with opposition (18.0% → 30.2%).

**And it is loose**: it peaks at ρ = −0.5 rather than at −0.9, because counting
disagreements per parameter ignores **how much** each task disagrees, and a few
large conflicts hurt more than many small ones.

> **Use it as a screen, not as a prediction.** A low conflict rate is good news you
> can act on; a high one says look closer, not "abandon".

### The negative result on TIES

{{cite:yadav2023ties}} adds trimming and sign election to fix exactly the
mechanisms above. **In this experiment it loses to a plain average at every
conflict level**, including where it should help most: **0.5320 against 0.4208**
at ρ = −0.9, both with a tuned scale coefficient.

**That is a statement about the regime, not the method.** Even at maximum
opposition the disagreement rate here is 30%, and TIES's setting has far more to
work with — eight or more task vectors from a large model, where deltas are sparse
and pairwise conflicts accumulate. **Two dense deltas from a small network do not
present enough conflict for a conflict-resolution mechanism to recover the signal
that trimming discards.**

**What transfers is the diagnosis rather than the remedy.**

## 5. Formal Explanation

### 5.1 Linear mode connectivity

For fine-tunes $\theta_A, \theta_B$ from a common $\theta_0$, define the barrier

$$ B = \max_{\alpha \in [0,1]} \mathcal{L}\big((1-\alpha)\theta_A + \alpha\theta_B\big) - \max\big(\mathcal{L}(\theta_A), \mathcal{L}(\theta_B)\big) $$ (eq:linear-mode-connectivity)

**{{eq:linear-mode-connectivity}} is small when the fine-tunes share a basin and
large when they do not.** The shared basin is what a shared ancestor buys: with
$\|\theta_A - \theta_0\|$ and $\|\theta_B - \theta_0\|$ both small, the segment
cannot leave a neighbourhood of $\theta_0$.

### 5.2 Why permutation symmetry breaks independent merges

For a hidden layer with permutation $P$,

$$ f_{\theta}(x) = f_{P\theta}(x) \quad \forall x $$ (eq:permutation-symmetry)

so the loss landscape has $H!$ equivalent copies of every minimum. Two independent
runs land in *different* copies with probability $\to 1$, and

$$ \tfrac{1}{2}\big(\theta + P\theta\big) \ne \text{any } P'\theta $$

**{{eq:permutation-symmetry}} is why averaging independent networks fails and why
sharing a base fixes it** — fine-tuning does not permute anything.

### 5.3 Task arithmetic

Writing $\tau_i = \theta_i - \theta_0$ for the task vector
({{cite:ilharco2023taskarithmetic}}):

$$ \theta_{\text{merged}} = \theta_0 + \lambda \sum_i \tau_i $$ (eq:task-arithmetic)

**{{eq:task-arithmetic}}'s $\lambda$ is a real hyperparameter, not a detail.**
Plain averaging is $\lambda = 1/n$; TIES's disjoint mean divides by the number of
*agreeing* tasks, which is smaller, so it produces a larger update at the same
nominal setting. Comparing merge methods at a fixed $\lambda$ compares scale
tuning rather than methods — which is why the measurement tunes $\lambda$ per
method.

### 5.4 Conflict bounds every method

Decompose each parameter's task vectors into agreeing and conflicting components.
For two tasks at parameter $j$:

$$ \text{agree}_j \iff \operatorname{sign}(\tau_{1j}) = \operatorname{sign}(\tau_{2j}) $$

On agreeing parameters any sensible rule moves in the shared direction. On
conflicting parameters, **any** single value $v$ satisfies

$$ |v - \tau_{1j}| + |v - \tau_{2j}| \ge |\tau_{1j} - \tau_{2j}| $$ (eq:conflict-governs-merging)

with equality for any $v$ between them. **{{eq:conflict-governs-merging}} is a
floor no algorithm goes below**: the total displacement from what the two tasks
wanted is at least their separation, and the merging rule only chooses how to
distribute it.

Averaging distributes evenly; election gives one task everything and the other
nothing. **Neither creates a value that satisfies both, because none exists.**

### 5.5 What the merge is worth, formally

$$ \mathcal{L}_{\text{joint}} \;\le\; \mathcal{L}_{\text{merge}} \;\le\; \mathcal{L}_{\text{base}} $$ (eq:merge-bounds)

with the measurement giving **0.1851 ≤ 0.1997 ≤ 0.8002.** The useful statistic is
where in that interval the merge lands:

$$ \text{merge efficiency} = \frac{\mathcal{L}_{\text{base}} - \mathcal{L}_{\text{merge}}}{\mathcal{L}_{\text{base}} - \mathcal{L}_{\text{joint}}} = \frac{0.8002 - 0.1997}{0.8002 - 0.1851} = 97.6\% $$ (eq:merge-efficiency)

**{{eq:merge-efficiency}} is the number to report**, because it prices the merge
against its actual alternative rather than against the specialists.

### 5.6 Distillation is the other direction

Where merging combines weights, distillation ({{cite:hinton2015}}) combines
*behaviour*:

$$ \mathcal{L}_{\text{distil}} = \mathrm{KL}\big(p_{\text{teacher}}(\cdot \mid x)\,\|\,p_{\text{student}}(\cdot \mid x)\big) $$ (eq:distillation)

**{{eq:distillation}} has no shared-base precondition** — the student need not
share an architecture, a size, or an ancestor. That is exactly the case merging
cannot reach, and the cost is that it requires *training* and *data*, where a
merge requires neither.

> **IMPORTANT:** For combining tasks, distillation also **has no conflict floor**
> in the sense of {{eq:conflict-governs-merging}}, because the student can allocate
> *different parameters* to the two behaviours instead of reconciling one. When the
> sign-conflict screen says the tasks are opposed, **distillation from both
> teachers is the technique that is not blocked.**

## 6. Mathematical Foundation

### 6.1 Why the failed merge's optimum sits at an endpoint

If the barrier {{eq:linear-mode-connectivity}} is large, the joint loss along the
segment is roughly

$$ \mathcal{L}(\alpha) \approx (1-\alpha)\mathcal{L}_A + \alpha\mathcal{L}_B + B\cdot\alpha(1-\alpha) $$

Minimising: the interior stationary point exists only if $B$ is small relative to
the endpoint difference. For large $B$ the minimum is **at a boundary** — which is
the measured $\alpha = 0.875$ against the shared base's $\alpha = 0.5$.

**So $\alpha^{*}$ is itself a diagnostic**: an interior optimum means the merge is
combining; a boundary optimum means it is selecting.

### 6.2 Why soups improve and merges do not

For $n$ models fine-tuned on the *same* task with independent noise $\epsilon_i$
around a common optimum:

$$ \operatorname{Var}\Big[\tfrac{1}{n}\sum_i \epsilon_i\Big] = \frac{\sigma^2}{n} $$ (eq:soup-variance)

**{{eq:soup-variance}} is ordinary variance reduction**, and it is why a soup beats
its ingredients. For *different* tasks the $\tau_i$ are not noise around a shared
optimum — they are signal pointing to different optima, and averaging signal is
{{eq:conflict-governs-merging}}, not {{eq:soup-variance}}.

**One name, two entirely different mathematical situations.**

### 6.3 The conflict floor, quantified

For two tasks with conflicting parameters at rate $c$ and mean separation $s$ on
those parameters, the merged model's expected displacement from each task's
optimum is at least

$$ \mathbb{E}\|\theta_{\text{merge}} - \theta_i\| \;\gtrsim\; \sqrt{c}\;\frac{s}{2} $$

so, by {{ch:ft-lora}}'s {{eq:forgetting-quadratic}}, the loss penalty grows as
$c\,s^2/4$ — **linear in the conflict rate and quadratic in how strongly the tasks
disagree.**

> **MATH NOTE:** That quadratic dependence on $s$ is why the sign-conflict *rate*
> is a loose proxy: it measures $c$ and ignores $s$. A merge with few but severe
> conflicts scores well on the screen and badly in reality, which is exactly the
> non-monotonicity in the measurement — the peak conflict rate at $\rho = -0.5$
> against the worst merge at $\rho = -0.9$.

## 7. Internal Mechanics

```mermaid {#fig:merge-decision caption="The decision procedure. Two screens run before any merging algorithm is chosen: a shared ancestor (without which the optimum degenerates to an endpoint) and task agreement (which bounds what any rule can achieve, per eq:conflict-governs-merging). Only after both pass does the choice of merging method matter, and it matters less than either screen."}
flowchart TB
    START["two fine-tuned models"] --> Q1{{"same base checkpoint,<br/>same precision,<br/>first fine-tune?"}}
    Q1 -->|no| SEP["keep separate,<br/>or distil from both"]
    Q1 -->|yes| Q2{{"sign agreement<br/>between the deltas?"}}
    Q2 -->|low| CONF["tasks are in tension:<br/>joint training, or<br/>decide which loses"]
    Q2 -->|high| MERGE["merge"]
    MERGE --> LAM["tune the scale<br/>coefficient lambda"]
    LAM --> CHK{{"is the optimum<br/>in the interior?"}}
    CHK -->|"at an endpoint"| SEP
    CHK -->|"interior"| DONE["ship one model"]
```

### 7.1 The methods, and how much they matter

| Method | Rule | When it is the right choice |
|---|---|---|
| uniform average | $\theta_0 + \frac{1}{n}\sum\tau_i$ | the default; hard to beat at small $n$ |
| task arithmetic | $\theta_0 + \lambda\sum\tau_i$ | same, with $\lambda$ tuned |
| weighted | per-task $\lambda_i$ | when tasks differ in importance |
| TIES | trim, elect, disjoint mean | many sparse deltas, high conflict |
| negation | $\theta_0 - \lambda\tau$ | removing a behaviour |

**The last row is the most under-used consequence of
{{eq:task-arithmetic}}.** If a delta is an object, subtracting it is meaningful:
fine-tune deliberately *toward* an unwanted behaviour, then negate the resulting
task vector. It is a strange and effective technique, and it inherits every
precondition in this chapter.

### 7.2 The screens, in order

1. **Shared ancestor?** Same checkpoint, same precision, first fine-tune.
2. **Sign agreement?** One elementwise comparison.
3. **Tune $\lambda$.** It is a real hyperparameter and the methods are not
   comparable without it.
4. **Is $\alpha^{*}$ interior?** A boundary optimum means the merge failed.
5. **Report {{eq:merge-efficiency}}**, not the loss against the specialists.

### 7.3 When to distil instead

Distillation is the answer when a screen fails:

- **No shared ancestor** — different families, different pretraining, a
  proprietary teacher.
- **Genuine task conflict** — the student can allocate separate capacity where a
  merge must reconcile.
- **A size change is wanted** — the student can be smaller, which no merge
  achieves.

**The cost is a training run and data**, which is exactly what merging avoids. So
the two are complements: merge when the screens pass because it is free; distil
when they fail because nothing else works.

## 8. Implementation

```python {tier=A name=merging-needs-a-shared-base}
"""Merging works, and the precondition is not optional.

cite:wortsman2022modelsoups reported that averaging the weights of several
fine-tuned models produces a model better than any of them, which sounds like it
should not work. Averaging two neural networks is normally catastrophic: the same
function can be represented by permuting hidden units, so two independently
trained networks put the same feature in different places and their average puts
it nowhere.

The reason soups work is a precondition that gets dropped in retelling: the models
were fine-tuned FROM A SHARED BASE. That keeps them in one low-loss basin, where
the segment between them stays low-loss too (eq:linear-mode-connectivity).

This listing measures the barrier both ways -- shared base and independent
initialisation -- because the difference between them is the whole reason merging
is a technique rather than folklore.
"""
import numpy as np

rng = np.random.default_rng(223)

D, H, DO = 14, 48, 6
N = 2500


def init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def grad(p, X, Y):
    h, o = forward(p, X)
    d = 2 * (o - Y) / len(X)
    dh = d @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def fit(p, X, Y, steps, lr=0.01):
    p = [w.copy() for w in p]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    for t in range(steps):
        g = grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p


def lerp(pa, pb, a):
    return [(1 - a) * x + a * y for x, y in zip(pa, pb)]


W_PRE = rng.normal(size=(D, DO)) / np.sqrt(D)
W_A = rng.normal(size=(D, DO)) / np.sqrt(D)
W_B = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.12):
    X = rng.normal(size=(n, D))
    return X, np.tanh(X @ W) + 0.3 * X[:, :DO] + noise * rng.normal(size=(n, DO))


Xp, Yp = task(W_PRE, N)
Xa, Ya = task(W_A, N); Xa_t, Ya_t = task(W_A, 1500)
Xb, Yb = task(W_B, N); Xb_t, Yb_t = task(W_B, 1500)

# Shared base: one pretrained model, two fine-tunes.
BASE = fit(init(), Xp, Yp, 3000)
SA = fit(BASE, Xa, Ya, 900)
SB = fit(BASE, Xb, Yb, 900)

# Independent: two models trained on the same two tasks from scratch.
IA = fit(init(), Xa, Ya, 3000)
IB = fit(init(), Xb, Yb, 3000)

# The thing a merge is a cheap substitute for: one model trained on both tasks.
XJ = np.concatenate([Xa, Xb]); YJ = np.concatenate([Ya, Yb])
JOINT = fit(BASE, XJ, YJ, 1800)

ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

print(f"Interpolating between two fine-tunes. alpha=0 is model A, 1 is model B.\n")
print(f"{'alpha':>7}{'SHARED BASE':>34}{'INDEPENDENT INIT':>36}")
print(f"{'':>7}{'A loss':>11}{'B loss':>11}{'joint':>11}{'':>2}"
      f"{'A loss':>11}{'B loss':>11}{'joint':>11}")
print("-" * 77)

sh, ind = [], []
for a in ALPHAS:
    ps, pi = lerp(SA, SB, a), lerp(IA, IB, a)
    sv = (mse(ps, Xa_t, Ya_t), mse(ps, Xb_t, Yb_t))
    iv = (mse(pi, Xa_t, Ya_t), mse(pi, Xb_t, Yb_t))
    sh.append(sv); ind.append(iv)
    print(f"{a:>7.3f}{sv[0]:>11.4f}{sv[1]:>11.4f}"
          f"{(sv[0]+sv[1])/2:>11.4f}{'':>2}"
          f"{iv[0]:>11.4f}{iv[1]:>11.4f}{(iv[0]+iv[1])/2:>11.4f}")

joint = lambda v: (v[0] + v[1]) / 2
m_sh = min(sh, key=joint)
m_ind = min(ind, key=joint)
A_SH = ALPHAS[sh.index(m_sh)]
A_IND = ALPHAS[ind.index(m_ind)]
MID = len(ALPHAS) // 2
J_MULTI = (mse(JOINT, Xa_t, Ya_t), mse(JOINT, Xb_t, Yb_t))
J_BASE = (mse(BASE, Xa_t, Ya_t), mse(BASE, Xb_t, Yb_t))

print("")
print(f"{'one model that does both tasks':>34}{'A loss':>11}{'B loss':>11}"
      f"{'joint':>11}")
print("-" * 67)
for name, v in (("base model, no fine-tuning", J_BASE),
                ("merged, shared base", m_sh),
                ("merged, independent init", m_ind),
                ("trained jointly on both tasks", J_MULTI)):
    print(f"{name:>34}{v[0]:>11.4f}{v[1]:>11.4f}{joint(v):>11.4f}")
print(f"""
The alpha sweep says merging is possible at all. The summary table says what it
is worth.

Start with the sweep. Both columns move smoothly from one specialist to the other
-- there is no wall in the middle, which is already worth noting, because
averaging two neural networks has no right to work. A network's function is
unchanged by permuting its hidden units, so two independently trained networks
that compute similar things generally put those things in different coordinates,
and averaging coordinate-wise then averages unrelated features.

But compare the joint columns at the true midpoint, which is what "average the
two models" actually means. With a shared base, alpha=0.5 gives
{joint(sh[MID]):.4f}. With independent initialisation, {joint(ind[MID]):.4f} --
{joint(ind[MID])/joint(sh[MID]):.1f}x worse, on the same tasks, the same
architecture and the same budget.

Then look at where each column's BEST point sits, which is the sharper diagnostic.
The shared-base optimum is at alpha={A_SH}, in the interior, where the merge is
genuinely combining two models. The independent optimum is at alpha={A_IND},
essentially at an endpoint -- the best thing you can do with those two models is
to pick one and use it. That is what a failed merge looks like from the outside:
not an error, just an optimum that quietly degenerates to "do not merge".

That gap is the precondition, measured. Fine-tuning from a shared base never
leaves the region where the coordinates mean the same thing, so the segment
between two fine-tunes stays inside one low-loss basin
(eq:linear-mode-connectivity). Independent runs land in different basins that
happen to compute similar functions, and the straight line between them leaves
both.

So the precondition is operational rather than theoretical: merging works between
models that share an ancestor and have not travelled far from it. It is not a
general model-combination technique, and essentially every merge failure reported
in practice is a violation of that sentence rather than a subtlety of the merging
algorithm.

Three violations worth naming because they look innocuous. Two models of the same
architecture from different pretraining runs do not share a base. A model
fine-tuned twice in sequence has travelled further than the merge assumes. And a
quantised copy is not the same base as the original, which is
ch:ft-qlora-peft's adapter constraint arriving here in a new form.

Now the summary table, which prices the technique honestly.

The base model before any fine-tuning has a joint loss of {joint(J_BASE):.4f} --
the do-nothing baseline. The shared-base merge reaches {joint(m_sh):.4f}.
Training one model on both tasks together, which is what a merge is a cheap
substitute for, reaches {joint(J_MULTI):.4f}.

That is the number that makes merging worth a chapter. The merge is within
{joint(m_sh)/joint(J_MULTI)-1:.0%} of an actual multi-task training run, and it
cost a weighted sum over two checkpoints you already had. Not an approximation
that gets you most of the way with caveats -- {joint(m_sh)/joint(J_MULTI)-1:.0%},
for milliseconds of arithmetic.

Read the same numbers the other way and the limit is equally clear. The merged
model is worse on task A than model A and worse on task B than model B. Merging
did not produce a model better at everything; it produced ONE model instead of
two, at a measured cost on each.

That distinction matters because it is routinely blurred.
cite:wortsman2022modelsoups's headline -- a soup that beats every ingredient --
comes from averaging models fine-tuned on the SAME task with different
hyperparameters, where averaging cancels independent noise. Merging models
fine-tuned on DIFFERENT tasks combines skills and pays for it. Two techniques, one
name, and only the first is free.""")
```

The first listing establishes when merging is possible. The second establishes
what bounds it.

```python {tier=A name=conflict-governs-merging}
"""What decides whether a merge works: how much the tasks disagree.

cite:ilharco2023taskarithmetic makes the fine-tuning delta an object you can add
and subtract. cite:yadav2023ties observes that adding several of them works less
well than it should, and names two mechanisms -- REDUNDANCY (most entries carry
no task information but dilute the average) and SIGN CONFLICT (where two tasks
want opposite movements, averaging produces something neither wanted). Its
remedy trims small entries, elects a sign per parameter, and averages only the
entries agreeing with it.

That is usually presented as a better averaging rule. This listing tests it as
what it actually is: a rule whose payoff is a function of how much your tasks
conflict. Task pairs are constructed with a controlled relationship, from nearly
aligned to directly opposed, and at each level the sign-disagreement rate and the
merge quality are both measured (eq:conflict-governs-merging).
"""
# --- setup, identical to the previous listing ------------------------
import numpy as np

rng = np.random.default_rng(223)

D, H, DO = 14, 48, 6
N = 2500


def init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def grad(p, X, Y):
    h, o = forward(p, X)
    d = 2 * (o - Y) / len(X)
    dh = d @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), h.T @ d, d.sum(0)]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def fit(p, X, Y, steps, lr=0.01):
    p = [w.copy() for w in p]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    for t in range(steps):
        g = grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p


def lerp(pa, pb, a):
    return [(1 - a) * x + a * y for x, y in zip(pa, pb)]


W_PRE = rng.normal(size=(D, DO)) / np.sqrt(D)
W_A = rng.normal(size=(D, DO)) / np.sqrt(D)
W_B = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.12):
    X = rng.normal(size=(n, D))
    return X, np.tanh(X @ W) + 0.3 * X[:, :DO] + noise * rng.normal(size=(n, DO))


Xp, Yp = task(W_PRE, N)
Xa, Ya = task(W_A, N); Xa_t, Ya_t = task(W_A, 1500)
Xb, Yb = task(W_B, N); Xb_t, Yb_t = task(W_B, 1500)
# --- end of shared setup ---------------------------------------------

BASE = fit(init(), Xp, Yp, 3000)
W_1 = rng.normal(size=(D, DO)) / np.sqrt(D)
W_PERP = rng.normal(size=(D, DO)) / np.sqrt(D)


def paired_tasks(rho):
    """Task 2's target is correlated with task 1's at `rho`: +1 is the same
    task, 0 unrelated, -1 directly opposed."""
    W2 = rho * W_1 + np.sqrt(max(0.0, 1 - rho ** 2)) * W_PERP
    t1 = task(W_1, N) + task(W_1, 1200)
    t2 = task(W2, N) + task(W2, 1200)
    return t1, t2


def combine(deltas, do_trim, do_elect, keep=0.30):
    out = []
    for i in range(4):
        S = np.stack([d[i] for d in deltas])
        if do_trim:
            T = S.copy()
            for t in range(len(S)):
                a = np.abs(T[t])
                if a.size:
                    T[t] = np.where(a >= np.quantile(a, 1 - keep), T[t], 0.0)
            S = T
        if do_elect:
            sign = np.sign((S * (S != 0)).sum(0) + 1e-12)
            ok = (np.sign(S) == sign[None]) & (S != 0)
            n = ok.sum(0)
            C = np.where(n > 0, (S * ok).sum(0) / np.maximum(n, 1), 0.0)
        else:
            C = S.mean(0)
        out.append(C)
    return out


LAMS = (0.3, 0.5, 0.7, 0.85, 1.0, 1.2)


def best(deltas, tests, do_trim, do_elect):
    """Every merge method carries a scale coefficient. Comparing methods at a
    fixed one compares scale tuning rather than the methods."""
    C = combine(deltas, do_trim, do_elect)
    out = None
    for lam in LAMS:
        p = [b + lam * c for b, c in zip(BASE, C)]
        L = float(np.mean([mse(p, Xt, Yt) for Xt, Yt in tests]))
        if out is None or L < out:
            out = L
    return out


def disagreement(deltas):
    """Share of significant entries where the two tasks want opposite signs."""
    tot, bad = 0, 0
    for i in range(4):
        S = np.stack([d[i] for d in deltas])
        a = np.abs(S)
        if not a.size:
            continue
        big = (a >= 0.1 * a.max()).all(0)
        tot += big.sum()
        bad += ((np.sign(S[0]) != np.sign(S[1])) & big).sum()
    return bad / max(tot, 1)


RHOS = (0.9, 0.5, 0.0, -0.5, -0.9)
print(f"Two tasks fine-tuned from one base, merged. `rho` is how related the "
      f"two\ntasks are: +1 identical, 0 unrelated, -1 opposed.\n")
print(f"{'rho':>6}{'sign':>9}{'specialists':>13}{'average of':>13}"
      f"{'TIES':>9}{'better':>10}")
print(f"{'':>6}{'conflict':>9}{'on own task':>13}{'deltas':>13}{'':>9}"
      f"{'method':>10}")
print("-" * 60)

rows = {}
for rho in RHOS:
    (X1, Y1, X1t, Y1t), (X2, Y2, X2t, Y2t) = paired_tasks(rho)
    p1 = fit(BASE, X1, Y1, 900)
    p2 = fit(BASE, X2, Y2, 900)
    deltas = [[w - b for w, b in zip(p, BASE)] for p in (p1, p2)]
    tests = [(X1t, Y1t), (X2t, Y2t)]
    own = float(np.mean([mse(p1, X1t, Y1t), mse(p2, X2t, Y2t)]))
    avg = best(deltas, tests, False, False)
    ties = best(deltas, tests, True, True)
    cf = disagreement(deltas)
    rows[rho] = (cf, own, avg, ties)
    print(f"{rho:>6.1f}{cf:>9.1%}{own:>13.4f}{avg:>13.4f}{ties:>9.4f}"
          f"{('TIES' if ties < avg else 'average'):>10}")

hi, lo = rows[0.9], rows[-0.9]
mid = rows[0.0]
cf_max = max(rows[r][0] for r in RHOS)
print(f"""
Read the two right-hand columns down the page. The specialists column is the
control, and it is flat: {hi[1]:.4f} at rho=0.9 and {lo[1]:.4f} at rho=-0.9. Each
task is equally learnable regardless of what the other task is.

The merge column is not flat at all. Averaging two deltas from closely related
tasks reaches {hi[2]:.4f} -- close to the specialists, so the merge is nearly
free. Averaging two deltas from opposed tasks reaches {lo[2]:.4f}, an eightfold
degradation, on tasks that are individually just as easy.

That is the result, and it is a limit rather than a technique. Merge quality is
governed by how much the tasks AGREE about which way the weights should move.
Where two tasks require the same parameter to move in opposite directions, no
combining rule satisfies both, because no single value satisfies both. Averaging
splits the difference and implements neither; electing a sign implements one and
abandons the other. Both are honest answers to an impossible request
(eq:conflict-governs-merging).

The practical consequence is a question to ask before merging rather than after:
are these tasks compatible? A merge between two tasks that pull the same
direction is close to free and scales to many tasks. A merge between tasks in
genuine tension has a floor set by the tension, and no amount of merging-method
sophistication lowers it.

The sign-conflict column is the cheap proxy for that, and it is worth reporting
with its limitation. It does rise as the tasks become opposed -- {hi[0]:.1%} at
rho=0.9 against {cf_max:.1%} at its peak -- so it carries real signal for the
price of an elementwise comparison of two tensors you already have. But it is
loose: it peaks at rho=-0.5 rather than at rho=-0.9, because measuring
disagreement per-parameter ignores how MUCH each task disagrees, and a few large
conflicts hurt more than many small ones. Use it as a screen, not as a prediction.

Now the part that did not come out as expected, and is reported rather than tuned
away. TIES loses to a plain average at every conflict level here, including at
rho={-0.9} where it should have the most to fix: {lo[3]:.4f} against
{lo[2]:.4f}. Both were given a tuned scale coefficient, so this is not a
calibration artefact.

The reason is visible in the conflict column. Even at maximum opposition the
disagreement rate is only {cf_max:.1%}, and cite:yadav2023ties's setting has far
more to work with -- eight or more task vectors from a large model, where deltas
are sparse and pairwise conflicts accumulate across many pairs. Two dense deltas
from a small network do not present enough conflict for a conflict-resolution
mechanism to recover the signal that trimming discards.

TIES is a real improvement in the regime it was proposed for, and this experiment
is not in that regime. What transfers is the DIAGNOSIS rather than the remedy:
task conflict is the quantity that decides merging, it is measurable in advance,
and it bounds what any merging method can achieve.

So the rule to take away. Measure agreement between the deltas you intend to
merge, and treat a low value as information about the TASKS rather than as a
problem to be solved by a better merging rule. When tasks are in genuine tension
the honest options are to keep separate models, to train one model on both tasks
together, or to decide which task is served worse -- and the last of those is a
product decision that a merging algorithm should not be making silently.""")
```

## 9. Practical Example

**The precondition, measured.** Fine-tunes from a shared base merge to a joint
loss of **0.1997** at $\alpha = 0.5$; independently initialised models merge to
**0.3869** at $\alpha = 0.5$ — **1.9× worse**, same tasks, architecture and
budget.

**And the optimum's location is the sharper signal.** Shared base: interior,
$\alpha = 0.5$. Independent: $\alpha = 0.875$, essentially an endpoint —
**the best thing to do with those two models is pick one.**
{{eq:permutation-symmetry}} explains why, and
{{sec:6-mathematical-foundation}} explains why a large barrier pushes the optimum
to a boundary.

**What the merge is worth.** Base model, no fine-tuning: **0.8002.** Merged:
**0.1997.** Trained jointly on both tasks: **0.1851.** That is
{{eq:merge-efficiency}} = **97.6%** of the way to a full multi-task training run,
for a weighted sum over checkpoints you already had.

> **IMPORTANT:** The merged model is still worse on task A than model A and worse
> on task B than model B. **Merging bought one model instead of two, at a cost on
> each** — not a model better at everything.
> {{cite:wortsman2022modelsoups}}'s stronger claim is about **soups**: the same
> task, different hyperparameters, where {{eq:soup-variance}} makes averaging pure
> variance reduction. **Two techniques, one name, and only the first is free.**

**Conflict is the limit.** Across tasks constructed from ρ = +0.9 to −0.9, the
specialists stay flat (**0.0155 → 0.0156**) while the merge degrades **0.0515 →
0.4208** — eightfold, on tasks individually just as easy.
{{eq:conflict-governs-merging}} is the floor: where two tasks want opposite
movements, no value satisfies both, and the rule only chooses how to distribute
the failure.

**The sign-conflict screen works, loosely.** It rises with opposition (**18.0% →
30.2%**) and it is **not monotone** — peaking at ρ = −0.5 while the worst merge is
at ρ = −0.9 — because it counts conflicts and ignores their size, and the loss
penalty is **quadratic** in size. **Use it as a screen, not a prediction.**

**And a negative result, reported rather than tuned away.** TIES lost to a plain
average at every conflict level here, **0.5320 against 0.4208** at maximum
opposition, both with tuned $\lambda$. Even at ρ = −0.9 the disagreement rate is
only ~30%, where {{cite:yadav2023ties}}'s setting has eight or more sparse task
vectors and accumulating pairwise conflict. **The method is right for its regime
and this is not that regime; what transfers is the diagnosis, not the remedy.**

## 10. Production Considerations

**Verify the shared ancestor** — checkpoint hash, precision, and whether either
model has been fine-tuned before. This catches most failures.

**Run the sign-agreement screen** before choosing a method. It is one elementwise
comparison.

**Tune $\lambda$.** Method comparisons without it are scale-tuning comparisons.

**Check that $\alpha^{*}$ is interior.** A boundary optimum means the merge
failed, whatever the loss says.

**Report {{eq:merge-efficiency}}**, against joint training rather than against the
specialists.

**Never merge across a quantisation change** ({{eq:qlora-forward}}).

**Prefer a uniform average as the default.** At small $n$ it is hard to beat and
it has no hyperparameters beyond $\lambda$.

**Reach for distillation when a screen fails**, not for a more sophisticated
merging rule.

## 11. Common Mistakes

**Quoting the soup result to justify a task merge.**

**Merging models from different pretraining runs** because the architecture
matches.

**Merging a QLoRA adapter into an unquantised base.**

**Comparing merge methods at a fixed $\lambda$.**

**Treating a boundary optimum as a successful merge** with an unlucky weight.

**Reaching for TIES on two dense deltas**, where it has nothing to fix.

**Expecting a merge to beat the specialists on their own tasks.**

**Responding to high conflict with a better algorithm** rather than with joint
training or separate models.

## 12. Failure Modes

**Merged model is worse than either input everywhere.** Cause: no shared base
({{eq:permutation-symmetry}}), or severe conflict
({{eq:conflict-governs-merging}}).

**Merge works for two tasks and collapses at five.** Cause: pairwise conflicts
accumulate; this is TIES's actual regime.

**Best $\lambda$ is at the edge of the grid.** Cause: the combining rule's implicit
scale differs from what you assumed — extend the grid before concluding anything.

**Merge succeeded offline, model behaves oddly in production.** Cause: the two
tasks were evaluated separately and the merge was never tested on inputs that
trigger both.

**A previously reliable merge recipe stops working.** Cause: one input model is
now a second-generation fine-tune.

**Sign-agreement screen said fine, merge was poor.** Cause: few but severe
conflicts — the screen measures $c$ and the penalty scales with $c s^2$.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| uniform average | quality on each task | screens pass; the default |
| task arithmetic with tuned $\lambda$ | a small sweep | same, slightly better |
| TIES ({{cite:yadav2023ties}}) | complexity | many sparse deltas, real conflict |
| joint multi-task training | a training run | the quality ceiling |
| separate models | serving cost | genuine conflict, both tasks matter |
| adapter switching ({{ch:ft-lora}}) | serving complexity | many tasks, one base |
| distillation ({{cite:hinton2015}}) | training run, data | no shared base, or conflict |

**The adapter row deserves emphasis.** If both fine-tunes are LoRA adapters over
one base, you do not have to choose: {{ch:ft-lora}}'s
{{eq:lora-merge}} lets you merge *or* swap per request. **Merging is a decision you
can defer**, which is a strong argument for keeping adaptations as adapters even
when you expect to merge them.

## 14. Evaluation

**Report the base checkpoint identity and precision** for every input model.

**Report the sign-agreement rate** between deltas.

**Report $\lambda$ and $\alpha^{*}$**, and say whether the optimum was interior.

**Report {{eq:merge-efficiency}}** against joint training.

**Evaluate on inputs that exercise both tasks**, not only on each task separately
— a merge can pass two separate evaluations and fail where they meet.

**Compare merge methods only at their own tuned $\lambda$.**

## 15. Advanced Concepts

**Permutation alignment before merging.** {{maturity:EMERGING}}
{{eq:permutation-symmetry}} is the obstacle for independent models, and it is in
principle removable: find the permutation aligning one network to the other, then
merge. It works, it is expensive, and it turns "merging needs a shared base" from
a hard constraint into an economic one.

**Negation as a capability edit.** {{maturity:EMERGING}}
{{eq:task-arithmetic}} with a negative coefficient removes a behaviour by training
toward it and subtracting. It is the clearest demonstration that a delta is an
object, and it inherits every precondition here.

**Conflict as a task-similarity metric.** {{maturity:EMERGING}} The sign-agreement
rate is a cheap, model-derived measure of how related two tasks are — computable
without any semantic judgement. **Its use as a general task-similarity metric is
mostly unexplored**, and this chapter's caution applies: it measures conflict count
and not conflict severity.

**Merging as an alternative to multi-task scheduling.** {{maturity:MATURE}}
{{eq:merge-efficiency}} at 97.6% says that for compatible tasks, merging removes
the need to balance task mixtures during training — which is a large practical
simplification for the case where it applies.

**Distillation as the general combiner.** {{maturity:ESTABLISHED}}
{{eq:distillation}} has no shared-base precondition and no conflict floor, because
the student allocates capacity rather than reconciling values. **It is the
technique that is never blocked, and it is never free.**

## 16. Connection to Previous Chapters

{{ch:ft-lora}}'s {{eq:lora-merge}} is this chapter's premise — the delta as a
storable, addable object — and its adapter-swapping property is why merging can be
deferred. {{ch:ft-qlora-peft}}'s {{eq:qlora-forward}} becomes a merging
precondition: an adapter belongs to its base's precision.
{{ch:ft-training-config}} produced the situation this chapter addresses, and its
{{eq:forgetting-quadratic}} supplies the quadratic penalty in
{{sec:6-mathematical-foundation}}'s conflict bound.
{{ch:ft-datasets}} and {{ch:ft-synthetic}} determine whether the two fine-tunes
were worth having in the first place.
Forward: {{part:15}} quantises the merged result, {{part:23}} serves it, and
{{part:25}} evaluates it — including on the inputs where both tasks meet, which is
where merges fail.

## 17. Exercises

1. Explain {{eq:permutation-symmetry}} and estimate how many equivalent minima a
   64-unit hidden layer has.
2. Derive the condition under which
   {{sec:6-mathematical-foundation}}'s interpolation minimum is interior, and
   relate it to the measured $\alpha^{*} = 0.5$ against $0.875$.
3. Compute {{eq:merge-efficiency}} for a merge at 0.25 with base 0.90 and joint
   training at 0.20.
4. Prove {{eq:conflict-governs-merging}} and state what it implies about *any*
   merging algorithm.
5. In `conflict-governs-merging`, add a third task and see whether pairwise
   conflicts accumulate as predicted.
6. In the same listing, replace the sign-conflict measure with one weighted by
   conflict magnitude. Is it monotone in $\rho$?
7. Implement negation: fine-tune toward an unwanted behaviour and subtract the
   task vector. What happens to the base capability?
8. For two adapters you have: compute the sign-agreement rate and predict whether
   merging will work, then check.

## 18. Interview Questions

1. Why should averaging two neural networks fail, and why does it sometimes work?
2. State merging's precondition operationally.
3. Name three ways the precondition is violated that look innocuous.
4. What is the difference between a model soup and a task merge?
5. Your merge's best interpolation weight is 0.95. What does that tell you?
6. What bounds the quality of *any* merging algorithm?
7. Why is the sign-conflict rate a loose predictor?
8. What should a merge be compared against, and why not the specialists?
9. When is distillation the right answer instead?
10. Why can merging be deferred if your fine-tunes are LoRA adapters?

## 19. Research Questions

1. Permutation alignment makes the shared-base precondition economic rather than
   absolute. What is the actual cost/benefit at production model scales?
2. {{sec:6-mathematical-foundation}}'s bound predicts a $c\,s^2$ penalty. Does a
   severity-weighted conflict measure predict merge quality monotonically where
   the count-based one does not?
3. {{eq:merge-efficiency}} was 97.6% for two compatible tasks. How does it decay
   with the number of merged tasks, and is the decay predictable from pairwise
   conflict?
4. Task negation edits behaviour without retraining. What are its limits, and can
   the edited capability be recovered?
5. Distillation has no conflict floor because the student allocates capacity. Is
   there a merging rule that allocates capacity — expanding the model slightly
   rather than reconciling values?

## 20. Chapter Summary

**Merging works because of a precondition, and the precondition is the whole
story.** Shared-base fine-tunes merged to **0.1997**; independently initialised
models to **0.3869** — **1.9× worse** — and the failed merge's optimum sat at
$\alpha = 0.875$, essentially an endpoint. **A failed merge is not an error; it is
an optimum that degenerates to "do not merge".**
{{eq:permutation-symmetry}} is why, and the three violations to check are a
different pretraining run, a second-generation fine-tune, and a precision change.

**Soup and merge share a name and no mathematics.**
{{eq:soup-variance}} makes averaging same-task models pure variance reduction, so
a soup beats its ingredients. Averaging *different-task* models combines signal,
not noise, and the merged model was worse on each task than that task's
specialist. **It bought one model instead of two.**

**And it is worth buying.** Merged **0.1997** against joint training's **0.1851**
and a do-nothing baseline of **0.8002** — {{eq:merge-efficiency}} = **97.6%** of a
full multi-task run, for milliseconds of arithmetic on checkpoints you already
had.

**Conflict is the limit and it is not fixable.** Merge quality ran **0.0515 →
0.4208** as tasks went from aligned to opposed, while the specialists stayed flat
at **0.0155 → 0.0156.** {{eq:conflict-governs-merging}} is a floor no algorithm
goes below: where two tasks want opposite movements, no single value satisfies
both, and the rule only distributes the failure.

**The sign-agreement screen is cheap and loose** — rising 18.0% → 30.2% with
opposition but peaking at ρ = −0.5 rather than −0.9, because the penalty scales
with conflict *severity* squared and the screen counts *occurrences*.

**And TIES lost to a plain average here, at every conflict level**, which is a
statement about the regime rather than the method: two dense deltas do not supply
enough conflict to repay what trimming discards. **What transfers is the
diagnosis, not the remedy.**

Which leaves the clean decision procedure: **shared ancestor, then sign agreement,
then tune $\lambda$, then check the optimum is interior.** If a screen fails,
**distillation** is the technique that is not blocked — it has no shared-base
requirement and no conflict floor, because the student allocates capacity rather
than reconciling values. **It is never blocked and never free**, which is exactly
the complement merging needs.

## 21. Further Reading

{{cite:wortsman2022modelsoups}} for the result that made this a technique, read
carefully for what its models had in common — the shared base and the shared task
are both doing work, and only the first survives into task merging.
{{cite:ilharco2023taskarithmetic}} for the delta as an algebraic object, and for
negation, which is the most surprising consequence and the least used.
{{cite:yadav2023ties}} for the two interference mechanisms; read the setup
carefully, because the regime — many sparse task vectors — is what makes the
remedy pay, and this chapter's measurement is outside it.
{{cite:hinton2015}} for distillation, positioned here as the general combiner
rather than only as a compression technique.
{{cite:hu2021lora}} for why keeping adaptations as adapters lets you postpone
every decision in this chapter.
