---
id: ft-training-config
number: 136
part: XIV
tier: full
status: draft
requires: [ft-lora, ft-sft, ft-preference, dl-optimizers]
provides: [capability-exchange-rate, distance-is-the-lever, rehearsal,
           weight-anchoring, joint-stopping-criterion, forgetting-measurement,
           mitigation-hierarchy]
citations: [kirkpatrick2017ewc, biderman2024loralearnsless, hu2021lora,
            zhou2023lima, ouyang2022]
---

## 1. Learning Objectives

By the end of this chapter you will be able to measure the **exchange rate**
between new capability and old capability at every point in a fine-tuning run;
explain why the standard stopping rule takes a bad deal without being asked; show
that learning rate and epoch count are **mostly one lever**, and what that lever
actually sets; rank rehearsal, anchoring and configuration by measured effect
rather than by folklore; and order the interventions so the largest one comes
first.

## 2. Why This Matters

Forgetting is usually a section in a fine-tuning chapter. It is the **spine** of
this one, because {{ch:ft-lora}}'s {{eq:learns-less-forgets-less}} showed that
every fine-tune trades one capability for another — and almost nobody measures the
rate.

{{sec:9-practical-example}} measures it. Fine-tuning a converged model on a new
task, the marginal exchange rate — task-B capability gained per unit of task-A
capability spent — runs **19.0 → 5.5 → 3.0 → 1.9 → 1.0 → 0.3 → 0.1** across a
single short run.

**Early steps are nearly free. Late steps are nearly pure damage.** By step 12,
task B has gained **59.1%** of everything it will ever gain, for **14.6%** of the
task-A capability it will eventually destroy. The remaining 41% of task B costs
the other **85%** of task A.

**The standard stopping rule — lowest validation loss on the target task — takes
that second deal without being asked**, because it has no information about the
thing being traded away.

Then the configuration question, and the answer is a negative result.
{{sec:9-practical-example}} runs three learning rates spanning an order of
magnitude with step counts matched so each covers the same ground. At a distance
of 0.6 the task-A damage is **10.8%, 12.4%, 10.7%.** At 1.5 it is **63.4%, 70.2%,
76.3%.**

**How you travelled barely mattered. How far you travelled determined almost
everything.** Learning rate and epochs are mostly two ways of setting one
quantity.

**And the mitigations are real but small.** At 75% of task B, the plain run costs
**36.0%** of task A; lower LR **30.2%**, rehearsal at 20% **33.8%**, rehearsal at
equal parts **27.9%**, anchoring **31.1%**. The best is a **1.29×** improvement.

**Against a stopping decision worth roughly 4×.** That ordering is the chapter.

{{maturity:ESTABLISHED}} Early stopping, rehearsal.
{{maturity:MATURE}} Forgetting as a measured quantity.
{{maturity:EMERGING}} Treating the exchange rate as the object being chosen.

## 3. Prerequisites

{{ch:ft-lora}} supplies {{eq:forgetting-quadratic}}, which this chapter's shape
follows from, and {{eq:rank-constrains-movement}}, which is the structural version
of what anchoring does continuously. {{ch:ft-sft}} for the training loop;
{{ch:ft-preference}} because preference optimisation is a fine-tune and every
result here applies to it; {{ch:dl-optimizers}} for what a step actually does.

## 4. Intuitive Explanation

### There is an exchange rate, and it collapses

A fine-tune buys new capability with old capability. That is not a metaphor —
{{sec:9-practical-example}} prices it per step:

```text
   step   B gained   A lost   marginal exchange
   ────   ────────   ──────   ─────────────────
      3      17.5%     1.0%            19.0
      9      47.3%     8.5%             3.0
     12      59.1%    14.6%             1.9
     18      76.7%    30.2%             1.0
     30      94.3%    65.1%             0.3
     57      99.9%   100.0%             0.2
```

**At the start each unit of task A buys nineteen units of task B. By the end it
buys a fifth of one.**

### Why the shape is what it is

It follows from the two quantities, not from this setup.

{{ch:ft-lora}}'s {{eq:forgetting-quadratic}} makes task-A loss **quadratic** in
distance travelled from its minimum — so at small distances it is nearly flat, and
the first steps cost almost nothing. Task-B loss starts far from its own minimum
and is **concave** in the same distance: fast at first, slower as it approaches.

**Quadratic against concave gives exactly this curve**, and it means the shape is
general rather than incidental.

### The stopping rule is choosing for you

The usual criterion stops at the target task's lowest validation loss. In the
measurement, that is **step 57, having given up 100% of task A.**

Nothing is wrong with that checkpoint *if task A does not matter*. The problem is
that **the decision was never made.** The target task's validation curve selected
the trade silently, and it is the one curve that carries no information about what
is being given up.

> **The fix is almost free.** The base-capability evaluation you need already
> exists — it is whatever told you the base model was good enough to start from.
> Run it at every checkpoint. It costs a fraction of the training budget and
> converts an invisible exchange into a visible one.

### Learning rate and epochs are one lever

The standard advice is: lower the learning rate, train fewer epochs, forget less.
{{sec:9-practical-example}} tests it by matching on **distance travelled** instead
of on steps.

Three learning rates across an order of magnitude agree closely at matched
distance — **10.8%, 12.4%, 10.7%** at 0.6 — and diverge only modestly at large
distance.

**So the advice works if you also stop after the same number of steps**, because
then you have travelled less. Run the lower learning rate to convergence — as
anyone chasing target-task quality eventually does — and you arrive at nearly the
same place by a slower route.

### What actually moves the curve, and by how much

Compared at matched task-B gain, so these are genuine trade-off comparisons:

```text
   strategy                       A lost at 75% of B
   ────────────────────────────   ──────────────────
   plain fine-tune                       36.0%
   lower LR, more steps                  30.2%
   rehearsal: 20% task-A data            33.8%
   rehearsal: equal parts                27.9%
   anchor to base weights                31.1%
```

**Everything helps. The best is 1.29× better than doing nothing.**

Now put that beside the stopping decision: **accepting 59% of task B instead of
95% took the damage from 65% to 15%** — a factor of roughly four.

> **The hierarchy: decide where to stop, then mitigate, then configure. The usual
> ordering of effort is the exact reverse.**

### Why rehearsal is the one to reach for

It is the only intervention here that changes what a given distance **costs**
rather than how much of it you cover: the gradient acquires a term pulling toward
task A's minimum, so the trajectory curves around the damage instead of walking
through it.

**And teams skip it for the wrong reason.** You do not need the original
pretraining corpus. Any data exercising the capability you want to keep will do —
a few thousand general instruction-following examples, held out from the
fine-tuning task. **The 20% row is one line of a data loader.**

## 5. Formal Explanation

### 5.1 The exchange rate

Let $d$ be distance travelled from the base weights, $G(d)$ the fraction of
target-task capability gained and $L(d)$ the fraction of base capability lost.
Define

$$ R(d) = \frac{dG/dd}{dL/dd} $$ (eq:capability-exchange-rate)

**{{eq:capability-exchange-rate}} is what a fine-tuning run is actually
spending**, and it is monotone decreasing whenever $G$ is concave and $L$ is
convex — which is the generic case.

### 5.2 Why $R$ collapses

From {{ch:ft-lora}}'s {{eq:forgetting-quadratic}}:

$$ L(d) \approx \tfrac{1}{2} d^{\top} H_A d \;\propto\; d^2 \quad\Rightarrow\quad \frac{dL}{dd} \propto d $$

and target-task loss decreasing toward its own minimum gives

$$ G(d) = 1 - e^{-d/\tau} \ \text{(roughly)} \quad\Rightarrow\quad \frac{dG}{dd} \propto e^{-d/\tau} $$

so

$$ R(d) \;\propto\; \frac{e^{-d/\tau}}{d} \;\xrightarrow[d \to 0]{} \infty, \quad \xrightarrow[d \to \infty]{} 0 $$ (eq:rate-collapses)

**{{eq:rate-collapses}} is the measured 19.0 → 0.1.** The divergence at $d \to 0$
is why the first steps look free, and it is also why "just train a little" is
better advice than it sounds.

### 5.3 The stopping criterion, stated properly

The usual rule is

$$ d^{*}_{\text{naive}} = \arg\min_d \mathcal{L}_B(d) $$

which ignores $\mathcal{L}_A$ entirely. The rule that makes the trade explicit is

$$ d^{*} = \arg\min_d \Big[ \mathcal{L}_B(d) + \omega\,\mathcal{L}_A(d) \Big] $$ (eq:joint-stopping)

**{{eq:joint-stopping}}'s $\omega$ is a product decision, not a hyperparameter.**
The point is not that some particular $\omega$ is right — it is that
$d^{*}_{\text{naive}}$ corresponds to $\omega = 0$, and nobody chose that.

Equivalently, stop when the exchange rate crosses a threshold you set:

$$ \text{stop when } R(d) < R_{\min} $$ (eq:stop-on-rate)

### 5.4 Distance is the lever

For an optimiser with step size $\eta$ over $T$ steps, the distance travelled is
approximately

$$ d \approx \eta \sum_{t=1}^{T} \|\hat{g}_t\| $$ (eq:distance-is-the-lever)

**{{eq:distance-is-the-lever}} has $\eta$ and $T$ entering as a product.** Halving
$\eta$ and doubling $T$ leaves $d$ roughly unchanged — and with it $L$ — which is
exactly what the measurement shows.

The residual divergence at large $d$ (63.4% vs 76.3%) is a **path-efficiency**
effect: a smaller step follows the gradient field more closely and acquires
slightly more $G$ per unit $d$.

### 5.5 Rehearsal changes the cost of distance

Training on a mixture $\mathcal{L} = \mathcal{L}_B + \rho\,\mathcal{L}_A$ gives

$$ \nabla \mathcal{L} = \nabla \mathcal{L}_B + \rho\, \nabla\mathcal{L}_A $$ (eq:rehearsal-gradient)

Since $\nabla\mathcal{L}_A = H_A d$ near the base minimum, the second term is a
**restoring force proportional to displacement** — a spring toward the base. The
trajectory therefore moves through directions $H_A$ prices cheaply.

**That is the key structural difference**: {{eq:rehearsal-gradient}} is
*direction-aware*, because $H_A$ appears.

### 5.6 Anchoring is direction-blind

$$ \nabla \mathcal{L} = \nabla \mathcal{L}_B + \gamma\, d $$ (eq:anchor-penalty)

**{{eq:anchor-penalty}} penalises all directions equally** — an isotropic version
of {{eq:rehearsal-gradient}} with $H_A$ replaced by $\gamma I$. That is precisely
the approximation {{cite:kirkpatrick2017ewc}}'s Fisher weighting exists to remove.

> **IMPORTANT:** So the theory predicts a strict ordering — EWC $\ge$ rehearsal
> $\ge$ anchoring — and the measurement gives rehearsal **27.9%** against
> anchoring's **31.1%**, a small gap in the predicted direction. **The theory is
> right about the ordering and wrong about the magnitude**, which is why nobody
> implements EWC: rehearsal captures most of the available benefit for none of the
> machinery.

## 6. Mathematical Foundation

### 6.1 Where the rate crosses one

From {{eq:rate-collapses}}, $R(d) = 1$ at $d$ satisfying $e^{-d/\tau} = cd$. In the
measurement the crossing is at **step 18**, where $G = 76.7\%$ and $L = 30.2\%$.

**That is a useful landmark**: past the crossing, every additional unit of target
capability costs more than a unit of base capability, in whatever units you chose.

### 6.2 The two levers, sized

Stopping moves $L$ from **65.1%** (step 30, $G=94.3\%$) to **14.6%** (step 12,
$G=59.1\%$) — a factor of **4.5**, at the cost of 35 points of $G$.

Mitigation moves $L$ from **36.0%** to **27.9%** at fixed $G = 75\%$ — a factor of
**1.29**, at no cost in $G$.

$$ \frac{\text{stopping lever}}{\text{mitigation lever}} \approx \frac{4.5}{1.29} \approx 3.5 $$ (eq:lever-ratio)

**{{eq:lever-ratio}} is the chapter's practical claim.** They are not
alternatives — mitigation is free and stopping is not — but the effort should be
allocated in that ratio, and it is usually allocated in the inverse.

### 6.3 Why epochs are a bad unit

Rewriting {{eq:distance-is-the-lever}} per epoch with $N/B$ steps per epoch:

$$ d \approx \eta\,\frac{N}{B}\, E \cdot \overline{\|\hat g\|} $$ (eq:epochs-and-distance)

**{{eq:epochs-and-distance}} shows "train for 3 epochs" is a statement about
$\eta N E / B$** — so the same epoch count means different distances on different
dataset sizes and batch sizes. That is why the advice does not transfer between
projects, and why {{cite:zhou2023lima}}'s small datasets need different epoch
counts than large ones for reasons that have nothing to do with overfitting.

> **MATH NOTE:** {{eq:forgetting-quadratic}} assumes the base sits at a minimum of
> $\mathcal{L}_A$. For a model already fine-tuned once, the linear term does not
> vanish and $L(d)$ acquires a component **linear** in $d$ — so
> {{eq:rate-collapses}} starts lower and never has the nearly-free region. **The
> second fine-tune has no cheap phase**, which is the formal version of why
> stacking adaptations degrades faster than the first one suggests.

## 7. Internal Mechanics

```mermaid {#fig:exchange caption="A fine-tuning run as a purchase. Distance travelled is the currency: it buys target capability along a concave curve and spends base capability along a quadratic one, so the exchange rate (eq:capability-exchange-rate) falls monotonically. Configuration sets how far you go; rehearsal and anchoring change what a given distance costs; the stopping rule decides how much you buy, and by default it is chosen by a curve that cannot see the price."}
flowchart LR
    D["distance travelled<br/>d = eta x steps x |g|"] --> G["target capability<br/>G(d): concave"]
    D --> L["base capability lost<br/>L(d): quadratic"]
    G --> R{{"exchange rate<br/>R = dG/dL"}}
    L --> R
    CFG["learning rate,<br/>epochs, batch"] -->|"sets HOW FAR"| D
    MIT["rehearsal,<br/>anchoring"] -->|"changes the PRICE<br/>of distance"| L
    R -->|"stop when R < R_min"| STOP["checkpoint"]
    NAIVE["min validation loss<br/>on the target task"] -.->|"omega = 0:<br/>chooses silently"| STOP
```

### 7.1 The measurement to add

Every fine-tuning run should log, at each checkpoint:

| Quantity | Cost | Why |
|---|---|---|
| target-task validation loss | already logged | $G$ |
| **base-capability eval** | a few minutes | $L$ — the missing half |
| $\|\theta - \theta_0\|$ | microseconds | $d$, which predicts $L$ |
| marginal exchange rate | arithmetic | {{eq:capability-exchange-rate}} |

**The second row is the whole intervention.** It is a benchmark you already ran
once, and running it per checkpoint turns a hidden trade into a visible one.

### 7.2 Choosing $\omega$

{{eq:joint-stopping}}'s weight is a product question with a real answer:

- **Base capability is the product**, fine-tune is a refinement → large $\omega$,
  stop early where the rate is favourable.
- **Base model is scaffolding** for a narrow deployment → $\omega \approx 0$,
  train to convergence and ignore the column.
- **Both matter** → set $R_{\min}$ and stop on {{eq:stop-on-rate}}.

**All three are legitimate. Picking by default is not.**

### 7.3 Overfitting is a different failure with a similar look

Overfitting and forgetting both show as "the model got worse", and they are
distinguishable:

| Symptom | Overfitting | Forgetting |
|---|---|---|
| target-task **train** loss | keeps falling | keeps falling |
| target-task **val** loss | rises | falls or flat |
| base capability | usually unaffected | falls |
| fix | fewer steps, more data, regularisation | fewer steps, rehearsal |

**They share a fix and not a cause**, which is why "just train less" appears to
work for both and explains neither. {{cite:zhou2023lima}}'s small-data regime
makes overfitting the binding constraint; large fine-tuning sets make forgetting
the binding one.

### 7.4 Why this chapter's advice differs from what tooling encourages

Every training framework exposes learning rate, batch size, epochs and a
scheduler. Almost none exposes a base-capability metric, a distance readout, or a
stopping rule that can see both tasks. The defaults therefore push effort toward
the smallest of the three levers, and the tooling is not wrong so much as
incomplete — it was built for the pretraining case, where there is no old
capability to protect because there is no old model.

That mismatch explains a pattern worth recognising. A team fine-tunes, the model
regresses somewhere nobody checked, and the response is a hyperparameter sweep,
because hyperparameters are what the interface offers. The sweep produces a small
improvement, consistent with {{eq:lever-ratio}}, and the underlying trade is never
made visible.

**The three additions that change this are all cheap**: log the base benchmark per
checkpoint, log $\|\theta - \theta_0\|$, and write down $\omega$ before the
run. None requires a new method, a new library, or a larger budget. What they
require is treating the base model's existing capability as a quantity the run is
spending rather than as a property it inherits — and that reframing is the whole
of the chapter's practical content.

### 7.5 A worked stopping decision

Suppose a support-assistant fine-tune on a model whose general reasoning is also
part of the product. The base benchmark scores 71%; the target task starts at 40%
and reaches 88% at convergence, by which point the base benchmark reads 58%.

Under {{eq:stop-on-rate}} with $R_{\min} = 1$ — refuse any step that costs more
base capability than it buys target capability — the run stops where the marginal
curves cross, which in the measurement was around 77% of the available target gain
for 30% of the available damage. Translated: roughly 77% of the way from 40% to
88% is 77%, and 30% of the way from 71% to 58% is 67%.

**So the choice is 88% and 58%, or 77% and 67%.** Eleven points of the new task
against nine points of the old one. Neither answer is obviously right, and that is
the point: it is a decision about which capability the product sells, and it is
answerable in a meeting once the numbers exist.

**What is not defensible is arriving at 88/58 without knowing 77/67 was
available**, which is what the default stopping rule guarantees.

## 8. Implementation

```python {tier=A name=capability-exchange-rate}
"""The exchange rate between new capability and old capability.

Every fine-tuning run has a stopping rule, and in practice it is the target
task's validation loss. That rule optimises one of the two things a fine-tune
changes and is silent about the other.

This listing tracks both. A network is trained to convergence on task A, then
fine-tuned on task B, and after every few steps BOTH are measured -- along with
the distance travelled from the base weights, which ch:ft-lora's
eq:forgetting-quadratic says should govern the damage.

The column to watch is the last one: how much task-B capability the run is buying
per unit of task-A capability it is spending, at each point along the way
(eq:capability-exchange-rate).
"""
import numpy as np

rng = np.random.default_rng(211)

D, H, DO = 16, 64, 8
N = 3000


def mlp_init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def loss_grad(p, X, Y):
    h, o = forward(p, X)
    d = (o - Y) / len(X)
    g3, g4 = h.T @ (2 * d), (2 * d).sum(0)
    dh = (2 * d) @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), g3, g4]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def train(p, X, Y, steps, lr, snap=None, Xa=None, Ya=None, base=None):
    """Plain Adam, full batch, so the trajectory is deterministic and the
    distance travelled is a clean quantity to reason about."""
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    hist = []
    for t in range(steps + 1):
        if snap and t % snap == 0:
            dist = np.sqrt(sum(((a - b) ** 2).sum()
                               for a, b in zip(p, base))) if base else 0.0
            hist.append((t, mse(p, X, Y), mse(p, Xa, Ya), dist))
        g = loss_grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p, hist


WA = rng.normal(size=(D, DO)) / np.sqrt(D)
WB = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.15):
    """Observation noise gives the base model a nonzero loss floor, so
    'degradation' is measured against something meaningful."""
    X = rng.normal(size=(n, D))
    return X, (np.tanh(X @ W) + 0.3 * X[:, :DO]
               + noise * rng.normal(size=(n, DO)))


Xa, Ya = task(WA, N)
Xb, Yb = task(WB, N)
Xa_te, Ya_te = task(WA, 2000)
Xb_te, Yb_te = task(WB, 2000)

base, _ = train(mlp_init(), Xa, Ya, 4000, 0.01)
base = [w.copy() for w in base]
A0, B0 = mse(base, Xa_te, Ya_te), mse(base, Xb_te, Yb_te)
print(f"Base model trained on task A: A loss {A0:.4f}, B loss {B0:.4f}.\n")

p = [w.copy() for w in base]
_, hist = train(p, Xb, Yb, 57, 0.0015, snap=3,
                Xa=Xa_te, Ya=Ya_te, base=base)

B_BEST = min(r[1] for r in hist)
A_WORST = max(r[2] for r in hist)


def gained(lb):
    return (B0 - lb) / (B0 - B_BEST)


def lost(la):
    return (la - A0) / (A_WORST - A0)


print(f"{'step':>6}{'B loss':>9}{'A loss':>9}{'B gained':>10}{'A lost':>9}"
      f"{'||delta||':>11}{'marginal':>11}")
print(f"{'':>6}{'':>9}{'':>9}{'':>10}{'':>9}{'':>11}{'exchange':>11}")
print("-" * 65)
prev = None
for t, lb, la, dist in hist:
    g, l = gained(lb), lost(la)
    rate = ""
    if prev is not None:
        dl = l - prev[1]
        rate = f"{(g - prev[0]) / dl:>11.1f}" if dl > 1e-9 else f"{'--':>11}"
    print(f"{t:>6}{lb:>9.4f}{la:>9.4f}{g:>10.1%}{l:>9.1%}{dist:>11.3f}{rate}")
    prev = (g, l)

b_only = min(hist, key=lambda r: r[1])
cheap = [r for r in hist if lost(r[2]) <= 0.15]
best_cheap = max(cheap, key=lambda r: gained(r[1]))
print(f"""
Read the last column down the page, because it is the whole listing.

Early in the run each unit of task-A capability spent buys several units of
task-B capability. Late in the run it buys a fraction of one. The exchange rate
is not constant, it does not degrade gently, and it crosses 1.0 well before the
target task's loss curve gives any sign of stopping.

That shape follows from the two quantities rather than from this setup. Task A's
loss is quadratic in the distance travelled from its minimum
(ch:ft-lora's eq:forgetting-quadratic), so at small distances it is nearly flat --
the first steps cost almost nothing. Task B's loss starts far from its own minimum
and falls fast and then slower, concave in the same distance. Quadratic against
concave gives exactly this: early steps nearly free, late steps nearly pure
damage (eq:capability-exchange-rate).

The numbers make the trade concrete. By step {best_cheap[0]}, task B has gained
{gained(best_cheap[1]):.0%} of everything it will ever gain, for
{lost(best_cheap[2]):.0%} of the task-A capability it will eventually destroy. The
remaining {1-gained(best_cheap[1]):.0%} of task B costs the other
{1-lost(best_cheap[2]):.0%} of task A.

That is not a marginal call. It is most of the benefit for a small fraction of the
cost, followed by a small benefit for most of the cost, and the usual stopping
rule takes the second deal without being asked.

The standard criterion -- lowest validation loss on the target task -- stops at
step {b_only[0]}, having given up {lost(b_only[2]):.0%} of task A. Nothing about
that checkpoint is wrong if task A does not matter. The problem is that the
decision was never made: the target task's validation curve chose the trade
silently, and it has no information about the thing being traded away.

Which makes the fix cheap and specific. The base-capability evaluation you need
already exists -- it is whatever told you the base model was good enough to start
from. Run it at every checkpoint. It costs a small fraction of the training budget
and converts an invisible exchange into a visible one, with an exchange rate you
can read off directly.

Then choose deliberately. If the base capability is the product and the fine-tune
is a refinement, stop early where the rate is favourable. If the base model is
scaffolding for a narrow deployment and general ability is irrelevant, train to
convergence and ignore the column. Both are legitimate. Picking by default is not,
and picking by default is the current standard.

One thread left for the next listing. Both effects track the distance column, and
if forgetting is a function of HOW FAR the weights moved rather than of how they
got there, then learning rate and epoch count are not independent levers at all --
they are two ways of setting one quantity, and most training-configuration advice
is about that quantity without saying so.""")
```

The first listing prices the trade. The second asks which knobs change it.

```python {tier=A name=distance-is-the-lever}
"""Learning rate and epochs set one quantity. Here is what actually moves the curve.

The standard advice for reducing catastrophic forgetting is to lower the learning
rate and train for fewer epochs. The previous listing suggested why that advice
behaves oddly in practice: both effects tracked the DISTANCE travelled from the
base weights, and learning rate and epoch count are two ways of setting one
distance.

This listing tests that directly, then asks the question that follows -- if
configuration mostly sets how far you go rather than what going that far costs,
what changes the cost?
"""
# --- setup, identical to the previous listing -----------------------
import numpy as np

rng = np.random.default_rng(211)

D, H, DO = 16, 64, 8
N = 3000


def mlp_init():
    return [rng.normal(size=(D, H)) / np.sqrt(D), np.zeros(H),
            rng.normal(size=(H, DO)) / np.sqrt(H), np.zeros(DO)]


def forward(p, X):
    h = np.tanh(X @ p[0] + p[1])
    return h, h @ p[2] + p[3]


def loss_grad(p, X, Y):
    h, o = forward(p, X)
    d = (o - Y) / len(X)
    g3, g4 = h.T @ (2 * d), (2 * d).sum(0)
    dh = (2 * d) @ p[2].T * (1 - h ** 2)
    return [X.T @ dh, dh.sum(0), g3, g4]


def mse(p, X, Y):
    return float(((forward(p, X)[1] - Y) ** 2).mean())


def train(p, X, Y, steps, lr, snap=None, Xa=None, Ya=None, base=None):
    """Plain Adam, full batch, so the trajectory is deterministic and the
    distance travelled is a clean quantity to reason about."""
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    hist = []
    for t in range(steps + 1):
        if snap and t % snap == 0:
            dist = np.sqrt(sum(((a - b) ** 2).sum()
                               for a, b in zip(p, base))) if base else 0.0
            hist.append((t, mse(p, X, Y), mse(p, Xa, Ya), dist))
        g = loss_grad(p, X, Y)
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return p, hist


WA = rng.normal(size=(D, DO)) / np.sqrt(D)
WB = rng.normal(size=(D, DO)) / np.sqrt(D)


def task(W, n, noise=0.15):
    """Observation noise gives the base model a nonzero loss floor, so
    'degradation' is measured against something meaningful."""
    X = rng.normal(size=(n, D))
    return X, (np.tanh(X @ W) + 0.3 * X[:, :DO]
               + noise * rng.normal(size=(n, DO)))


Xa, Ya = task(WA, N)
Xb, Yb = task(WB, N)
Xa_te, Ya_te = task(WA, 2000)
Xb_te, Yb_te = task(WB, 2000)

base, _ = train(mlp_init(), Xa, Ya, 4000, 0.01)
base = [w.copy() for w in base]
A0, B0 = mse(base, Xa_te, Ya_te), mse(base, Xb_te, Yb_te)
# --- end of shared setup --------------------------------------------

base = [w.copy() for w in base]
A0, B0 = mse(base, Xa_te, Ya_te), mse(base, Xb_te, Yb_te)


def run(lr, steps, rehearse=0.0, anchor=0.0, snap=2):
    """Fine-tune on B. `rehearse` mixes task-A examples back into training;
    `anchor` penalises distance from the base weights."""
    p = [w.copy() for w in base]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    out = []
    n_a = int(len(Xb) * rehearse)
    Xm = np.concatenate([Xb, Xa[:n_a]]) if n_a else Xb
    Ym = np.concatenate([Yb, Ya[:n_a]]) if n_a else Yb
    for t in range(steps + 1):
        if t % snap == 0:
            d = np.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(p, base)))
            out.append((d, mse(p, Xb_te, Yb_te), mse(p, Xa_te, Ya_te)))
        g = loss_grad(p, Xm, Ym)
        for i in range(4):
            if anchor:
                g[i] = g[i] + anchor * (p[i] - base[i])
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** (t + 1))) / (
                np.sqrt(v[i] / (1 - 0.999 ** (t + 1))) + 1e-8)
    return out


REF = run(0.0015, 300)
B_BEST = min(r[1] for r in REF)
A_WORST = max(r[2] for r in REF)


def gained(lb):
    return (B0 - lb) / (B0 - B_BEST)


def lost(la):
    return (la - A0) / (A_WORST - A0)


print("Does HOW you travel matter, or only how far?\n")
CONFIGS = [(0.0004, 320, 2), (0.0015, 100, 1), (0.0050, 32, 1)]
TARGETS = (0.3, 0.6, 0.9, 1.2, 1.5)
print(f"{'distance':>9}" + "".join(f"{'lr=' + str(lr):>22}"
                                   for lr, _, _ in CONFIGS))
print(f"{'travelled':>9}" + "".join(f"{'B gained':>11}{'A lost':>11}"
                                    for _ in CONFIGS))
print("-" * 75)

traj = {lr: run(lr, st, snap=sn) for lr, st, sn in CONFIGS}
spread, dmg = [], []
for tgt in TARGETS:
    cells = []
    for lr, _, _ in CONFIGS:
        r = min(traj[lr], key=lambda x: abs(x[0] - tgt))
        cells.append((gained(r[1]), lost(r[2])))
    spread.append(max(c[1] for c in cells) - min(c[1] for c in cells))
    dmg.append([c[1] for c in cells])
    print(f"{tgt:>9.1f}" + "".join(f"{g:>11.1%}{l:>11.1%}" for g, l in cells))

print("\n\nWhat DOES move the curve? Compared at matched task-B gain.\n")
STRATS = [("plain fine-tune", dict()),
          ("lower LR, more steps", dict(lr=0.0004, steps=400)),
          ("rehearsal: 20% task-A data", dict(rehearse=0.20)),
          ("rehearsal: equal parts A and B", dict(rehearse=1.00)),
          ("anchor to base weights", dict(anchor=3.0))]

print(f"{'strategy':>32}{'A lost at':>12}{'A lost at':>12}{'A lost at':>12}")
print(f"{'':>32}{'50% of B':>12}{'75% of B':>12}{'90% of B':>12}")
print("-" * 68)
res = {}
for name, kw in STRATS:
    lr = kw.pop("lr", 0.0015)
    st = kw.pop("steps", 300)
    tr = run(lr, st, **kw)
    row = []
    for want in (0.50, 0.75, 0.90):
        hit = [r for r in tr if gained(r[1]) >= want]
        row.append(lost(hit[0][2]) if hit else float("nan"))
    res[name] = row
    print(f"{name:>32}" + "".join(f"{'--':>12}" if np.isnan(v)
                                  else f"{v:>12.1%}" for v in row))

pl = res["plain fine-tune"]
lo = res["lower LR, more steps"]
r2 = res["rehearsal: 20% task-A data"]
r1 = res["rehearsal: equal parts A and B"]
an = res["anchor to base weights"]
print(f"""
The first table is the negative result, and it is the more useful half.

Three learning rates spanning an order of magnitude, with step counts chosen so
each covers the same ground. At short distances the three columns agree closely:
at 0.6 the task-A damage is {tuple(f'{v:.1%}' for v in dmg[1])}. The spread widens
with distance -- at 1.5 it is {tuple(f'{v:.1%}' for v in dmg[-1])} -- so the
fastest configuration is somewhat worse per unit of ground covered, which is a
real effect and a modest one.

The headline is the agreement, not the residual. Across an order of magnitude in
learning rate, distance travelled predicts damage far better than configuration
does. Learning rate and epoch count are not two independent levers on forgetting:
they are mostly two ways of setting one quantity, and the quantity is distance
from the base weights (eq:distance-is-the-lever).

That explains why the standard advice behaves inconsistently in practice. "Lower
the learning rate to forget less" works if you also stop after the same number of
steps, because then you have travelled less. Run the lower learning rate to
convergence, as anyone chasing target-task quality eventually does, and you arrive
at nearly the same place by a slower route.

The second table compares at matched task-B gain rather than matched distance, so
these are genuine trade-off comparisons. Every intervention helps, and the sizes
are the story.

At 75% of task B: the plain run costs {pl[1]:.1%} of task A. Lowering the learning
rate and training longer costs {lo[1]:.1%} -- the small path-efficiency effect the
first table predicted. Rehearsal at 20% costs {r2[1]:.1%}, at equal parts
{r1[1]:.1%}. Anchoring to the base weights costs {an[1]:.1%}.

The best of them improves on the plain run by a factor of
{pl[1]/min(lo[1], r2[1], r1[1], an[1]):.2f}. That is worth having and it is not
the big lever, and saying so is the point of running the comparison rather than
recommending a favourite.

Put it next to the previous listing to see what the big lever is. Stopping early
took 59% of task B for 15% of task A, where training to the target task's best
validation loss took essentially all of task B for essentially all of task A. The
stopping decision moves the damage by roughly a factor of four. Every mitigation
in this table moves it by about {pl[1]/min(lo[1], r2[1], r1[1], an[1]):.1f}.

So the hierarchy is: decide where to stop, then mitigate, then configure -- and
the usual ordering of effort is the exact reverse.

Two notes on the mitigations, since they are the part people can act on today.

Rehearsal is the one that changes what a given distance COSTS rather than how much
of it you cover, because the gradient now contains a term pulling toward task A's
minimum. It is also the cheapest to try, and teams skip it for the wrong reason:
you do not need the original pretraining corpus. Any data exercising the
capability you want to keep will do -- a few thousand general
instruction-following examples, held out from the fine-tuning task. The 20% row is
one line of a data loader.

Anchoring performs comparably here while treating every direction as equally
expensive, which rehearsal does not -- rehearsal implicitly knows which directions
task A cares about. That gap is exactly what cite:kirkpatrick2017ewc's Fisher
weighting exists to close, and it is why an importance-weighted penalty should in
principle beat both. It is also why nobody uses one: rehearsal gets most of the
benefit for none of the machinery.""")
```

## 9. Practical Example

**The exchange rate collapses.** Marginal task-B capability per unit of task-A
capability, along one run: **19.0 → 5.5 → 3.0 → 1.9 → 1.0 → 0.3 → 0.1.**

**By step 12, task B has gained 59.1% of everything it will gain, for 14.6% of the
task A it will eventually destroy.** The remaining 41% of B costs the other 85% of
A. {{eq:rate-collapses}} explains the shape: quadratic damage
({{eq:forgetting-quadratic}}) against concave gain.

**The standard stopping rule takes the second deal without being asked.** Lowest
target-task validation loss lands at step 57, having given up **100%** of task A.
{{eq:joint-stopping}} makes that $\omega = 0$ — a choice nobody made.

> **IMPORTANT:** The fix costs almost nothing. **The base-capability evaluation
> already exists**; run it per checkpoint. The exchange rate then reads straight
> off the log.

**Learning rate and epochs are mostly one lever.** Three learning rates across an
order of magnitude, matched on distance: task-A damage **10.8% / 12.4% / 10.7%**
at distance 0.6, and **63.4% / 70.2% / 76.3%** at 1.5.

**How you travelled barely mattered; how far you travelled determined almost
everything** ({{eq:distance-is-the-lever}}, where $\eta$ and $T$ enter as a
product). The modest divergence at large distance is path efficiency: a smaller
step acquires slightly more capability per unit of ground.

**So "lower the learning rate to forget less" works only if you also stop at the
same step count.** Run it to convergence and you arrive at nearly the same place
by a slower route.

**The mitigations are real and small.** At 75% of task B: plain **36.0%**, lower
LR **30.2%**, rehearsal 20% **33.8%**, rehearsal equal parts **27.9%**, anchoring
**31.1%** — best improvement **1.29×**.

**Rehearsal wins narrowly and for the right reason.**
{{eq:rehearsal-gradient}} is direction-aware because $H_A$ appears in it;
{{eq:anchor-penalty}} prices every direction alike. **The measured ordering matches
the theory and the gap is small**, which is exactly why {{cite:kirkpatrick2017ewc}}'s
Fisher weighting is admired and unused.

**And the levers are not the same size.** Stopping moved the damage **4.5×**;
mitigation moved it **1.29×** ({{eq:lever-ratio}}). **Decide where to stop, then
mitigate, then configure — the usual ordering of effort is the exact reverse.**

## 10. Production Considerations

**Log base capability at every checkpoint.** One line, and it is the missing half
of every fine-tuning run.

**Log $\|\theta - \theta_0\|$ too.** It predicts damage and costs nothing.

**Set $\omega$ or $R_{\min}$ explicitly** before the run, with whoever owns the
product.

**Add rehearsal by default.** Any held-out data exercising the capability you want
to keep; you do not need the pretraining corpus.

**Do not sweep learning rate to reduce forgetting.** {{eq:distance-is-the-lever}}
— sweep it for optimisation stability and set the *stopping point* for
forgetting.

**Report epochs with dataset size and batch size**, or the number means nothing
({{eq:epochs-and-distance}}).

**Expect no cheap phase on a second fine-tune.** The linear term is back.

**Distinguish overfitting from forgetting** before applying a fix that happens to
work for both.

## 11. Common Mistakes

**Stopping on target-task validation loss alone.**

**Never measuring base capability**, then being surprised in production.

**Tuning learning rate as a forgetting remedy.**

**Quoting epoch counts** as if they transferred between projects.

**Treating rehearsal as requiring the original pretraining data.**

**Assuming a second fine-tune behaves like the first.**

**Confusing overfitting with forgetting** because both respond to fewer steps.

**Spending the effort budget on configuration** when the stopping decision is
3.5× larger.

## 12. Failure Modes

**Model excellent at the new task, worse at everything else.** Cause:
$\omega = 0$ stopping. Fix: {{eq:joint-stopping}}.

**Lowering the learning rate did not help.** Cause:
{{eq:distance-is-the-lever}} — you ran it longer.

**Same recipe, different dataset, very different damage.** Cause:
{{eq:epochs-and-distance}}.

**Second fine-tune degrades far more than the first.** Cause: the base is no
longer at a minimum, so $L$ has a linear term.

**Rehearsal did nothing.** Cause: the rehearsal data does not exercise the
capability that is being lost — it must be measured, not assumed.

**Anchoring destabilised training.** Cause: {{eq:anchor-penalty}} interacts with
adaptive optimisers, which rescale the penalty gradient.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| early stopping on {{eq:stop-on-rate}} | target-task quality | the largest lever; almost always |
| rehearsal | data pipeline work | free; do it by default |
| anchoring / L2-to-base | direction-blindness | when rehearsal data is unavailable |
| EWC ({{cite:kirkpatrick2017ewc}}) | Fisher estimation | theoretically best, rarely worth it |
| LoRA ({{ch:ft-lora}}) | capacity | structural version of the same constraint |
| separate models | serving cost | when the trade is unacceptable |
| lower learning rate | nothing much | not a forgetting remedy |

**The last row is the point of the chapter**, and the second-to-last is the honest
escape hatch: if the exchange rate is bad everywhere, the answer may be **two
models**, and {{ch:ft-merging}} is about whether you can put them back together.

## 14. Evaluation

**Report base capability before and after.** A fine-tuning result without it is
half a result.

**Report the distance travelled**, which makes runs comparable across
configurations in a way epoch counts do not.

**Report the exchange rate at the chosen checkpoint**, and the $\omega$ or
$R_{\min}$ that selected it.

**Report dataset size and batch size with epoch counts.**

**Compare mitigations at matched target-task quality**, never at matched steps —
matched steps compares distances.

## 15. Advanced Concepts

**The exchange rate as the object being chosen.** {{maturity:EMERGING}}
{{eq:capability-exchange-rate}} reframes fine-tuning as a purchase with a price
that rises. Almost no tooling exposes it, and it is computable from data every run
already produces.

**Rehearsal is implicit EWC.** {{maturity:MATURE}}
{{eq:rehearsal-gradient}} contains $H_A$ by construction, so it approximates
Fisher weighting without estimating anything. **That is why the elegant method
lost to the crude one.**

**Structural versus continuous constraints.** {{maturity:MATURE}}
{{ch:ft-lora}} limits movement by restricting the parameterisation;
{{eq:anchor-penalty}} limits it by pricing. They compose, and the combination is
under-explored relative to how obvious it is.

**No cheap phase on repeat.** {{maturity:EMERGING}} The vanishing of
{{eq:forgetting-quadratic}}'s linear term is special to a model at its minimum.
**Sequential adaptation therefore has qualitatively different economics**, and
most practical pipelines are sequential.

**Curriculum as distance allocation.** {{maturity:RESEARCH FRONTIER}}
If damage is governed by distance and direction, ordering the training data to
travel through cheap directions first is a well-posed problem that is essentially
unstudied outside continual learning.

## 16. Connection to Previous Chapters

{{ch:ft-lora}} is this chapter's foundation twice over:
{{eq:forgetting-quadratic}} produces the exchange rate's shape, and
{{eq:learns-less-forgets-less}} is the same trade seen through rank rather than
distance — **LoRA sets a maximum distance structurally, this chapter chooses one
dynamically.**
{{ch:ft-sft}} and {{ch:ft-preference}} both produce runs this chapter measures;
preference optimisation is a fine-tune and forgets like one.
{{ch:ft-datasets}} supplies the rehearsal data, and its group-splitting discipline
applies to it.
{{ch:dl-optimizers}} explains why {{eq:anchor-penalty}} misbehaves under adaptive
methods.
Forward: {{ch:ft-merging}} is what to do when the exchange rate is unacceptable
and you need two models after all.

## 17. Exercises

1. Derive {{eq:rate-collapses}} from a quadratic $L$ and exponential $G$, and find
   where $R = 1$.
2. Using {{eq:distance-is-the-lever}}, predict the damage at half the learning
   rate and twice the steps. Verify against `distance-is-the-lever`.
3. Compute {{eq:epochs-and-distance}} for two projects with the same epoch count
   and a 10× dataset-size difference. What travels further?
4. In `capability-exchange-rate`, change the base model's convergence (train it
   for 1,000 steps instead of 4,000). Does the cheap phase survive, and why?
5. Add a second sequential fine-tune to the same listing and compare the exchange
   rate with the first. Explain via the linear term.
6. Implement {{eq:stop-on-rate}} with $R_{\min} = 1$ and report where it stops.
7. Show that {{eq:rehearsal-gradient}} and {{eq:anchor-penalty}} coincide when
   $H_A = \gamma I$, and say what that assumption means physically.
8. For a model you fine-tune: log base capability per checkpoint for one run and
   report the exchange rate curve.

## 18. Interview Questions

1. What does a fine-tune spend, and what does it buy?
2. Why are early steps nearly free?
3. What is wrong with stopping on target-task validation loss?
4. Does lowering the learning rate reduce forgetting? Answer carefully.
5. Why is "3 epochs" not portable advice?
6. Why does rehearsal beat an L2 penalty to the base weights?
7. Why is EWC theoretically better and practically unused?
8. How do you tell overfitting from forgetting?
9. Why does a second fine-tune have no cheap phase?
10. You have a fixed effort budget. Where do you spend it, and why?

## 19. Research Questions

1. {{eq:capability-exchange-rate}} is computable per run. What do real exchange
   rate curves look like across model scales, and is the crossing point
   predictable?
2. {{eq:lever-ratio}} was 3.5 here. Does the ratio between the stopping lever and
   the mitigation lever hold at scale?
3. Rehearsal approximates Fisher weighting implicitly. How close is the
   approximation, and where does explicit EWC still win?
4. Sequential fine-tuning loses the quadratic's flat region. Can a model be
   "re-centred" — returned to a minimum of the composite objective — cheaply
   enough to restore it?
5. If damage depends on direction as well as distance, can training data be
   ordered to travel through cheap directions first?

## 20. Chapter Summary

**A fine-tune buys new capability with old capability, and the exchange rate
collapses**: measured at **19.0 → 5.5 → 3.0 → 1.9 → 1.0 → 0.3 → 0.1** along one
run. By step 12, task B had gained **59.1%** of its total for **14.6%** of task A;
the remaining 41% cost the other 85%. {{eq:rate-collapses}} makes the shape
general — quadratic damage against concave gain.

**The standard stopping rule takes the expensive half without being asked.**
Lowest target-task validation loss landed at **100% of task A given up**, which
{{eq:joint-stopping}} identifies as $\omega = 0$ — a choice nobody made, made by
the one curve that cannot see the price. **The fix is to run the base-capability
evaluation you already have, per checkpoint.**

**Learning rate and epochs are mostly one lever.** Across an order of magnitude,
matched on distance, damage agreed to within a point or two at short distances
(**10.8 / 12.4 / 10.7**) and diverged modestly at long ones (**63.4 / 70.2 /
76.3**). {{eq:distance-is-the-lever}} has $\eta$ and $T$ entering as a product, so
"lower the learning rate to forget less" only works if you also stop sooner.

**The mitigations are real and small**: 36.0% → **27.9%** at matched capability, a
**1.29×** improvement, with rehearsal narrowly ahead of anchoring because
{{eq:rehearsal-gradient}} carries $H_A$ and {{eq:anchor-penalty}} does not — the
ordering {{cite:kirkpatrick2017ewc}} predicts, at a magnitude that explains why
nobody implements it.

**And the levers are not the same size.** Stopping moved the damage **4.5×**
against mitigation's **1.29×** ({{eq:lever-ratio}}). **Decide where to stop, then
mitigate, then configure — and the usual ordering of effort is the exact
reverse.**

Which leaves one honest escape. If the exchange rate is unacceptable everywhere —
if the task genuinely needs movement the base capability cannot afford — **the
answer is two models**, and whether they can be put back together is
{{ch:ft-merging}}'s subject.

## 21. Further Reading

{{cite:kirkpatrick2017ewc}} for forgetting attacked directly, and read
{{eq:rehearsal-gradient}} against it: the paper's Fisher matrix is what rehearsal
supplies implicitly, which is the most useful thing to know about both.
{{cite:biderman2024loralearnsless}} for the same trade measured through rank
rather than distance — the two accounts are the same phenomenon in different
coordinates.
{{cite:hu2021lora}} for the structural version of the constraint this chapter
applies dynamically.
{{cite:zhou2023lima}} for the small-data regime where overfitting rather than
forgetting binds, which is the case the standard advice was written for.
{{cite:ouyang2022}} for a production fine-tuning pipeline described in enough
detail to see where the base capability was and was not measured.
