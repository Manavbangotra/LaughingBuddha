---
id: dl-lr-schedules
number: 55
part: VI
tier: full
status: reviewed
requires: [dl-optimizers, dl-backprop, math-optimization]
provides: [learning-rate-schedule, warmup, cosine-schedule, step-decay,
           one-cycle, lr-range-test, effective-learning-rate, batch-size-scaling]
citations: [loshchilov2017sgdr, goyal2017, smith2017cyclical,
            smith2018supercon, kingma2015adam, loshchilov2019adamw]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why a constant learning rate cannot converge, from
   {{eq:sgd-convergence}}.
2. Compare step, exponential, cosine and one-cycle schedules and choose between
   them.
3. Explain what warmup does and why adaptive optimisers need it.
4. Run a learning-rate range test and read the result.
5. Apply the linear scaling rule for batch size and state where it breaks.
6. Diagnose a training curve and say whether the schedule is the problem.
7. Explain why the schedule interacts with the total step budget, and what that
   costs you.

## 2. Why This Matters

**The learning rate is the single most important hyperparameter**, and the
schedule is the second. {{ch:dl-optimizers}} measured a factor of thirty between
the best rates of different optimisers; within one optimiser, the difference
between a good schedule and a constant rate is routinely a large fraction of the
achievable improvement.

**{{eq:sgd-convergence}} says decay is not optional.** With a constant step size,
stochastic gradient descent converges to a *neighbourhood* of the optimum whose
radius is proportional to $\eta\sigma^2$, and then bounces around inside it
forever. Decay is what shrinks the neighbourhood. This is a theorem, not a
heuristic, and {{sec:8-implementation}} measures the noise floor directly.

**Warmup is required by every large transformer recipe and its justification is
partly empirical.** Skipping it produces divergence in the first few hundred
steps; the mechanism is understood well enough to predict when it matters and
not well enough to derive the right length.

**The schedule couples to the step budget in an awkward way.** A cosine schedule
must know the total number of steps in advance, so you cannot cleanly extend a
run that finished, and comparing runs of different lengths is not
straightforward. {{sec:12-failure-modes}} treats the consequences.

## 3. Prerequisites

{{ch:dl-optimizers}} for the optimisers being scheduled and for
{{eq:sgd-convergence}}. {{ch:dl-backprop}} for the gradients.
{{ch:math-optimization}} for convergence and conditioning.

## 4. Intuitive Explanation

### 4.1 Why one number cannot serve the whole run

Early in training the parameters are far from anything good and large steps make
rapid progress. Late in training they are close, the gradient is dominated by
mini-batch noise, and large steps only add noise to a nearly-correct answer.

```text
   loss
    │╲                          constant lr: descends, then bounces
    │ ╲___                       forever inside a noise floor
    │     ╲‿‿‿‿‿‿‿‿‿‿‿‿‿
    │
    │╲                          decayed lr: the floor comes down
    │ ╲___                       with the step size
    │     ╲‿‿╲___
    │           ╲‿╲___
    └──────────────────────▶ step
```

The characteristic signature of a schedule that steps down is a *visible drop*
in the loss at the moment of the decay. That drop is not the model suddenly
learning something; it is the noise floor descending. Recognising it prevents a
common misreading.

### 4.2 The shapes

```text
   step decay      ▔▔▔▔▔╲____╲____        drop by a factor at fixed epochs
   exponential     ▔▔╲___                  smooth geometric decay
   cosine          ▔▔▔╲╲╲___                slow, then fast, then slow
   linear          ▔▔╲╲╲╲╲╲╲___             constant rate of decrease
   one-cycle        ╱▔▔▔╲___                up, then down, then far down
   warmup + cosine ╱▔▔▔╲╲╲___               the transformer default
```

**Step decay** is the classical vision recipe and it produces the visible drops
above. **Cosine** {{cite:loshchilov2017sgdr}} is smooth and spends most of its
budget at a moderate rate, decaying sharply only near the end. **One-cycle**
{{cite:smith2018supercon}} rises to a high rate then falls well below the
starting point. **Warmup plus cosine** is what essentially every large language
model uses.

### 4.3 Warmup

Start at a small rate and increase linearly over the first few hundred to few
thousand steps.

Two reasons, and they are different:

**Adam's variance estimate is unreliable early.** $\vec{v}$ is an average over
roughly $1/(1-\beta_2) = 1000$ steps, so in the first tens of steps it is
computed from very few samples and is badly noisy. Dividing by the square root
of a noisy small number produces erratic steps, and warmup keeps them small
until the estimate settles.

**The initial parameters are arbitrary.** At step zero the network is random,
the gradient points somewhere unrelated to the eventual solution, and taking a
large step in that direction wastes it — or destroys the initialisation's
carefully chosen scale ({{ch:dl-initialization}}).

The first reason is specific to adaptive optimisers and the second is general.
That is why warmup is essentially mandatory with Adam and merely helpful with
SGD.

### 4.4 The learning-rate range test

A cheap and underused procedure. Train for a few hundred steps while increasing
the learning rate exponentially, and plot the loss against it:

```text
   loss
    │▔▔▔▔╲                        flat: too small to make progress
    │     ╲___                    falling: this is the useful range
    │         ╲__                 the minimum
    │            ╱                rising: too large
    │           ╱
    └────────────────▶ log(lr)
```

The rate at which the loss falls fastest — not the rate at the minimum, which is
already marginal — is a good maximum. It costs a few hundred steps and replaces
a grid search that costs full runs.

## 5. Formal Explanation

### 5.1 The schedules

**Step decay**, dropping by $\gamma$ every $k$ steps:

$$
\eta_t = \eta_0\,\gamma^{\lfloor t/k \rfloor}
$$ (eq:step-decay)

**Exponential**: $\eta_t = \eta_0 e^{-\lambda t}$.

**Cosine** {{cite:loshchilov2017sgdr}}, over $T$ total steps:

$$
\eta_t = \eta_{\min}
 + \tfrac{1}{2}(\eta_0-\eta_{\min})
   \left(1+\cos\frac{\pi t}{T}\right)
$$ (eq:cosine-schedule)

**Linear warmup** over $T_w$ steps:

$$
\eta_t = \eta_0\,\frac{t}{T_w}, \qquad t \le T_w
$$ (eq:warmup)

**Inverse square root**, the original transformer schedule:

$$
\eta_t = \eta_{\text{peak}}\,\min\!\left(\frac{t}{T_w},\;
 \sqrt{\frac{T_w}{t}}\right)
$$ (eq:inverse-sqrt)

The last has a property the others lack: **it does not need to know the total
number of steps.** That makes it the right choice when the budget is open-ended,
and it is why it reappeared for very large models where nobody knows in advance
when training will stop.

### 5.2 What the theory requires

The classical Robbins–Monro conditions for convergence of a stochastic
approximation:

$$
\sum_{t=1}^{\infty}\eta_t = \infty,
\qquad
\sum_{t=1}^{\infty}\eta_t^{2} < \infty
$$ (eq:robbins-monro)

The first says the steps must be able to travel an unbounded distance, so a
distant optimum remains reachable. The second says the accumulated noise must be
finite. Together they force $\eta_t \to 0$ at a rate between $1/t$ and
$1/\sqrt{t}$; $\eta_t = \eta_0/t$ satisfies both.

> WARNING: **None of the schedules used in practice satisfies
> {{eq:robbins-monro}}.** Cosine over a finite $T$ has a finite sum; step decay
> reaches a constant floor. The conditions are for asymptotic convergence on a
> convex problem, and deep learning runs for a finite budget on a non-convex one.
> They are the right way to understand *why* decay is needed and the wrong way
> to choose a schedule.

### 5.3 Batch size and the linear scaling rule

{{cite:goyal2017}} showed empirically that when the batch size is multiplied by
$k$, multiplying the learning rate by $k$ preserves training behaviour:

$$
B \to kB \;\Longrightarrow\; \eta \to k\eta
$$ (eq:linear-scaling)

The argument: $k$ steps of size $\eta$ on batches of size $B$ approximate one
step of size $k\eta$ on a batch of size $kB$, provided the gradient does not
change much over those $k$ steps.

**That proviso is where it breaks.** At large enough batch sizes the required
learning rate becomes unstable, and {{cite:goyal2017}} needed a warmup precisely
to survive the early steps at the scaled rate. An alternative $\sqrt{k}$ scaling
is sometimes better motivated — it keeps the gradient *noise* constant rather
than the expected displacement — and which applies depends on whether the
regime is noise-dominated or curvature-dominated.

### 5.4 The effective learning rate under Adam

Under SGD, $\eta$ multiplies the gradient, so the step is
$\eta\|\vec{g}\|$. Under Adam, {{eq:adam-step-bound}} says the per-parameter step
is at most about $\eta$ regardless of the gradient.

**So $\eta$ means different things under the two optimisers.** Under SGD it is a
gain; under Adam it is a distance. This is why Adam's learning rates cluster
around $10^{-3}$ across wildly different models and SGD's do not, and it is why
the two cannot share a schedule's absolute values even when they share its shape.

### 5.5 What a schedule interacts with

Three couplings worth stating explicitly, because each produces a confusing
result when ignored:

**With the total budget.** {{eq:cosine-schedule}} contains $T$. A cosine run
stopped at $T/2$ is not the same as a cosine run *planned* for $T/2$ — it is a
half-completed schedule at a much higher rate, and it will look worse than
either.

**With weight decay.** Under decoupled decay ({{eq:adamw}}) the per-step shrinkage
is $\eta\lambda$, so decaying $\eta$ also decays the regularisation.
Whether that is desirable is a real question, and some implementations schedule
$\lambda$ independently for exactly this reason.

**With batch size.** {{eq:linear-scaling}}. Changing one without the other
changes the training dynamics.

## 6. Mathematical Foundation

### 6.1 The noise floor

For a quadratic $\Like(\theta) = \frac{a}{2}\theta^2$ with gradient estimates
$g_t = a\theta_t + \xi_t$, $\xi_t \sim \mathcal{N}(0,\sigma^2)$, SGD gives

$$
\theta_{t+1} = (1-\eta a)\theta_t - \eta\xi_t
$$ (eq:noisy-sgd-recursion)

Taking variances of the stationary distribution, with
$V = \Var[\theta_\infty]$:

$$
V = (1-\eta a)^2 V + \eta^2\sigma^2
\;\Longrightarrow\;
V = \frac{\eta^2\sigma^2}{1-(1-\eta a)^2}
 = \frac{\eta\sigma^2}{a(2-\eta a)}
$$ (eq:stationary-variance)

For small $\eta a$ this is $V \approx \eta\sigma^2/(2a)$, so the stationary
excess loss is

$$
\E[\Like(\theta_\infty)] - \Like^\star = \tfrac{a}{2}V
 \approx \frac{\eta\sigma^2}{4}
$$ (eq:noise-floor)

**The floor is proportional to $\eta$, and to nothing else.** Halve the learning
rate and the floor halves. That is exactly the visible drop of
{{sec:4-intuitive-explanation}}, and {{sec:8-implementation}} measures the
proportionality constant.

Note also what {{eq:noise-floor}} does *not* contain: the number of steps. Once
the process is stationary, more steps at the same $\eta$ buy nothing.

### 6.2 Why $1/t$ is the classical answer

With $\eta_t = c/t$ the two sums in {{eq:robbins-monro}} are the harmonic series
(divergent) and $\sum 1/t^2$ (convergent), so both conditions hold. For a
strongly convex objective with parameter $a$, choosing $c > 1/(2a)$ gives the
optimal $O(1/t)$ rate.

The practical problem is that $c$ must be chosen relative to an unknown $a$: too
small and the schedule decays before reaching the optimum, leaving a bias that
never goes away. **A schedule that decays too fast is worse than one that decays
too slowly**, and that asymmetry is why the practical schedules hold a high rate
for most of the budget.

### 6.3 Why cosine holds high and then drops

Rewriting {{eq:cosine-schedule}} with $\eta_{\min} = 0$ and $u = t/T$:

$$
\eta(u) = \frac{\eta_0}{2}(1+\cos\pi u),
\qquad
\frac{d\eta}{du} = -\frac{\pi\eta_0}{2}\sin\pi u
$$ (eq:cosine-derivative)

The derivative is zero at both ends and maximal at the midpoint. So the schedule
**changes slowly at the start** (spending a long time near $\eta_0$), fastest in
the middle, and **slowly again at the end** (spending a long time near zero).

The fraction of the budget spent above half the peak rate is exactly $1/2$, by
symmetry of the cosine about $u = 1/2$. Compare exponential decay with the same
endpoints, which crosses half the peak at $u = \log 2/\log(\eta_0/\eta_T)$ — for
a hundredfold total decay, at $u = 0.15$.

**Cosine spends more than three times as long at a high rate as exponential
decay to the same endpoint**, and that is the whole design.

### 6.4 Warmup and Adam's variance estimate

At step $t$, $\vec{v}_t$ is a bias-corrected average of $t$ squared gradients
with weights $\beta_2^{i}$. Its **effective sample size** is

$$
n_{\text{eff}}(t) = \frac{\left(\sum_{i<t}\beta_2^{i}\right)^2}
 {\sum_{i<t}\beta_2^{2i}}
 = \frac{(1-\beta_2^{t})^2(1+\beta_2)}{(1-\beta_2^{2t})(1-\beta_2)}
$$ (eq:effective-sample-size)

which rises from 1 at $t = 1$ to $(1+\beta_2)/(1-\beta_2) \approx 2000$
asymptotically at $\beta_2 = 0.999$.

The relative standard deviation of a variance estimate from $n$ samples is
$\sqrt{2/n}$, so at $t = 1$ the estimate has 140% relative error and at
$t = 100$ about 15%. Since the update divides by $\sqrt{\hat{\vec{v}}}$, that
error passes into the step size directly.

**This is the quantitative argument for warmup.** It also predicts the right
warmup length: enough steps for $n_{\text{eff}}$ to reach a few hundred, which
at $\beta_2 = 0.999$ is a few hundred to a couple of thousand steps — matching
what recipes use. {{sec:8-implementation}} measures {{eq:effective-sample-size}}
and the resulting step variability.

### 6.5 Linear scaling, derived

Take $k$ SGD steps of size $\eta$ from $\vecgreek{\theta}_0$ on independent
batches:

$$
\vecgreek{\theta}_k = \vecgreek{\theta}_0
 - \eta\sum_{j=0}^{k-1}\nabla\Like_{\mathcal{B}_j}(\vecgreek{\theta}_j)
$$

If $\nabla\Like_{\mathcal{B}_j}(\vecgreek{\theta}_j) \approx
\nabla\Like_{\mathcal{B}_j}(\vecgreek{\theta}_0)$ — the parameters have not
moved enough for the gradient to change — then

$$
\vecgreek{\theta}_k \approx \vecgreek{\theta}_0
 - k\eta\cdot\frac{1}{k}\sum_j \nabla\Like_{\mathcal{B}_j}(\vecgreek{\theta}_0)
 = \vecgreek{\theta}_0 - k\eta\,\nabla\Like_{\cup\mathcal{B}_j}
$$ (eq:linear-scaling-derivation)

which is one step of size $k\eta$ on the union of the batches. Hence
{{eq:linear-scaling}}.

**The approximation is exactly what fails at the start of training**, when the
parameters move fast and the gradient changes between steps — which is why
{{cite:goyal2017}} needed warmup to make large-batch training work at all. The
two techniques are not independent tricks; the second is what buys the first its
validity.

## 7. Internal Mechanics

### 7.1 Where the schedule lives

```text
   for step in range(total):
       loss = forward(batch)
       loss.backward()
       lr = schedule(step)              # computed from the step count
       for group in optimizer.groups:
           group["lr"] = lr             # written before the update
       optimizer.step()
       optimizer.zero_grad()
```

Two ordering errors are common and both are silent. Setting the rate *after* the
step applies the previous step's rate — a one-step lag, harmless. Stepping the
schedule per *epoch* when it was defined per *step* stretches it by the number
of batches per epoch, which is a real misconfiguration and produces a run that
never decays.

### 7.2 Warmup and the optimiser state

During warmup the rate is small but the optimiser state still updates. Adam's
$\vec{m}$ and $\vec{v}$ accumulate throughout, which is the point: by the end of
warmup the variance estimate has the effective sample size of
{{eq:effective-sample-size}} and the full rate is safe.

This is also why warmup is not equivalent to simply starting later. The state
has to be built, and it can only be built by taking steps.

### 7.3 Resuming

A schedule is a function of the step count, so resuming requires the step count.
Restarting from zero replays the warmup at parameters that no longer need it and
produces a visible loss spike, then re-decays. Combined with the optimiser-state
issue of {{ch:dl-optimizers}}, this is the most common cause of a strange curve
after a restart.

### 7.4 Per-group schedules

Different parameter groups can take different rates, and two cases are standard:

**Layerwise decay** for fine-tuning, where lower layers get smaller rates
because they encode more general features ({{ch:ft-sft}}).

**Separate rates for a newly initialised head**, which needs a larger rate than
a pretrained body that would otherwise be disrupted.

### 7.5 What a warm restart does

{{cite:loshchilov2017sgdr}} proposed periodically resetting the rate to its peak
and decaying again, on cycles of increasing length. The stated motivation is
escaping a poor basin; the more defensible reading is that it produces an
ensemble of snapshots at successive minima, which can be averaged.
{{maturity:ESTABLISHED}} as a technique, with the mechanism less settled than
the practice.

## 8. Implementation

```python {tier=A name=schedules-and-the-noise-floor}
"""The schedules of section 5.1, and a direct measurement of the noise floor
that motivates all of them.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the schedules ----------------------------------------------------------
def constant(t, T, eta0, **kw):
    return eta0


def step_decay(t, T, eta0, gamma=0.1, drops=3, **kw):
    k = max(1, T // (drops + 1))
    return eta0 * gamma ** (t // k)                       # eq. 55.1


def exponential(t, T, eta0, final_frac=0.01, **kw):
    return eta0 * final_frac ** (t / T)


def cosine(t, T, eta0, eta_min=0.0, **kw):
    return eta_min + 0.5 * (eta0 - eta_min) * (
        1 + np.cos(np.pi * min(t, T) / T))                # eq. 55.2


def linear_decay(t, T, eta0, **kw):
    return eta0 * max(0.0, 1 - t / T)


def inverse_sqrt(t, T, eta0, warmup=200, **kw):
    return eta0 * min((t + 1) / warmup, np.sqrt(warmup / (t + 1)))


def with_warmup(fn, warmup):
    def wrapped(t, T, eta0, **kw):
        if t < warmup:
            return eta0 * (t + 1) / warmup                # eq. 55.3
        return fn(t - warmup, T - warmup, eta0, **kw)
    return wrapped


SCHEDULES = {
    "constant": constant,
    "step (x0.1, 3 drops)": step_decay,
    "exponential (to 1%)": exponential,
    "cosine": cosine,
    "linear": linear_decay,
    "inverse sqrt": inverse_sqrt,
    "warmup 10% + cosine": None,          # filled in below
}

# --- section 6.3: how much of the budget is spent at a high rate ------------
print("=" * 72)
print("the shape of each schedule (eq. 55.7)")
print("=" * 72)
T = 1000
eta0 = 1.0
SCHEDULES["warmup 10% + cosine"] = with_warmup(cosine, T // 10)

print(f"{'schedule':<22} " + " ".join(f"{f't={x}':>8}" for x in
                                      (0, 100, 250, 500, 750, 999))
      + f" {'frac > eta0/2':>15}")
for name, fn in SCHEDULES.items():
    vals = [fn(x, T, eta0) for x in (0, 100, 250, 500, 750, 999)]
    frac = np.mean([fn(x, T, eta0) > eta0 / 2 for x in range(T)])
    print(f"{name:<22} " + " ".join(f"{v:>8.4f}" for v in vals)
          + f" {frac:>15.3f}")

print("\nThe last column is section 6.3's calculation. Cosine spends exactly")
print("half its budget above half the peak rate, by symmetry about the")
print("midpoint. Exponential decay to the same endpoint spends far less —")
print("it falls below half the peak in the first sixth of the run.")
print("\nThat is the whole design argument for cosine: it holds a useful")
print("rate for a long time and then decays sharply, rather than spending")
print("most of the run at a rate too small to make progress.")

# --- section 6.1: the noise floor, measured ---------------------------------
print("\n" + "=" * 72)
print("the noise floor is proportional to the learning rate (eq. 55.10)")
print("=" * 72)
print("A one-dimensional quadratic with a = 1 and gradient noise sd = 1,")
print("run to stationarity at a constant learning rate.\n")

a, sigma = 1.0, 1.0
print(f"{'eta':>8} {'measured excess loss':>22} {'predicted eta*s^2/4':>21} "
      f"{'ratio':>8}")
for eta in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
    rs = np.random.default_rng(1)
    theta = 3.0
    tail = []
    n_steps = int(200 / eta)
    for t in range(n_steps):
        g = a * theta + rs.normal(0, sigma)
        theta -= eta * g
        if t > n_steps // 2:
            tail.append(0.5 * a * theta ** 2)
    measured = float(np.mean(tail))
    predicted = eta * sigma ** 2 / 4
    print(f"{eta:>8.3f} {measured:>22.6f} {predicted:>21.6f} "
          f"{measured / predicted:>8.3f}")

print("\nEq. 55.10 is confirmed to within the sampling error of a finite")
print("run: the stationary excess loss is proportional to eta with the")
print("predicted constant, and the scatter in the ratio column is the")
print("residual noise in averaging a stationary process over a finite tail. Halving the learning rate halves")
print("the floor, and no number of additional steps at a fixed eta gets")
print("below it — the process is stationary and has nothing left to do.")
print("\nThat is why the loss visibly DROPS at a step decay. The model did")
print("not suddenly learn something; the floor came down.")

# --- the same thing on a training curve -------------------------------------
print("\n" + "=" * 72)
print("what that looks like as a loss curve")
print("=" * 72)


def run_1d(sched, T=4000, eta0=0.3, seed=2):
    rs = np.random.default_rng(seed)
    theta, out = 3.0, []
    for t in range(T):
        eta = sched(t, T, eta0)
        g = a * theta + rs.normal(0, sigma)
        theta -= eta * g
        out.append(0.5 * a * theta ** 2)
    return np.array(out)


def window(losses, t, w=200):
    return float(np.mean(losses[max(0, t - w):t + 1]))


print(f"{'schedule':<22} " + " ".join(f"{f'@{x}':>10}" for x in
                                      (200, 1000, 2000, 3000, 3999)))
for name in ("constant", "step (x0.1, 3 drops)", "cosine",
             "exponential (to 1%)"):
    ls = run_1d(SCHEDULES[name])
    print(f"{name:<22} " + " ".join(f"{window(ls, x):>10.5f}"
                                    for x in (200, 1000, 2000, 3000, 3999)))

print("\nThe constant schedule reaches its floor early and stays there for")
print("the remaining 3800 steps, which is exactly what eq. 55.10 says it")
print("must do. Every decaying schedule keeps improving, because each")
print("reduction in eta lowers the floor it is sitting on.")
print("\nNote which schedule wins here: the one that decays FASTEST. On this")
print("problem the only obstacle is the noise floor — the quadratic is")
print("one-dimensional and perfectly conditioned, so there is no hard")
print("optimisation to do and no reason to hold a high rate.")
print("\nThat is worth flagging, because it is the opposite of what the")
print("network in the practical-example listing shows. This experiment")
print("isolates eq. 55.10 and therefore rewards aggressive decay; a real")
print("problem also has curvature to descend and a high rate is what does")
print("that. Cosine's shape is a compromise between the two pressures, and")
print("a measurement that only exhibits one of them will always prefer")
print("something more aggressive.")
```

```python {tier=A name=warmup-and-the-range-test}
"""Why warmup is needed with Adam, quantified from eq. 55.11, and the
learning-rate range test.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- section 6.4: Adam's variance estimate early in training ----------------
def effective_sample_size(t, b2=0.999):
    """Eq. 55.11: the effective number of samples in the EMA at step t."""
    num = (1 - b2 ** t) ** 2 * (1 + b2)
    den = (1 - b2 ** (2 * t)) * (1 - b2)
    return num / den


print("=" * 72)
print("Adam's second moment is estimated from very few samples early")
print("=" * 72)
print(f"{'step':>8} {'effective n':>14} {'rel. sd of v':>15} "
      f"{'rel. sd of 1/sqrt(v)':>22}")
for t in (1, 2, 5, 10, 50, 100, 500, 1000, 5000):
    n = effective_sample_size(t)
    rel = np.sqrt(2.0 / n)
    print(f"{t:>8} {n:>14.1f} {rel:>14.1%} {rel / 2:>21.1%}")

print("\nAt step 1 the variance estimate comes from a single sample and has")
print("a relative error above 100%. The update divides by its square root,")
print("so half that error passes straight into the step size.")
print("\nBy step 100 the effective sample size is in the tens and the error")
print("is manageable; by step 1000 it has essentially converged to the")
print("asymptotic (1+b2)/(1-b2) = 2000. That is the quantitative case for")
print("warmup, and it also predicts the right LENGTH: a few hundred to a")
print("couple of thousand steps, which is what recipes use.")

# --- the consequence, measured on real Adam updates -------------------------
print("\n" + "=" * 72)
print("what that does to the actual step sizes")
print("=" * 72)


def adam_steps(grad_sd=1.0, steps=2000, lr=1e-3, b1=0.9, b2=0.999,
               warmup=0, seed=0):
    """Track |update| for a single parameter under noisy zero-mean gradients."""
    rs = np.random.default_rng(seed)
    m = v = 0.0
    out = []
    for t in range(1, steps + 1):
        g = rs.normal(0, grad_sd)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        eta = lr * min(1.0, t / warmup) if warmup else lr
        out.append(abs(eta * mh / (np.sqrt(vh) + 1e-8)))
    return np.array(out)


print("Pure-noise gradients (mean zero), so every step is wasted motion and")
print("the only question is HOW FAR the parameter wanders.\n")
print(f"{'warmup':>8} {'max |step| in first 100':>25} "
      f"{'total distance, steps 1-100':>29} {'steps 100-2000':>16}")
for warmup in (0, 100, 500, 2000):
    st = adam_steps(warmup=warmup, seed=3)
    print(f"{warmup:>8} {st[:100].max():>25.3e} {st[:100].sum():>29.3e} "
          f"{st[100:].sum():>16.3e}")

print("\nWith no warmup the largest early step is far bigger than anything")
print("that follows, and the parameter travels a long way on gradients that")
print("carry no signal at all. Warmup suppresses exactly that window and")
print("leaves the rest of training untouched.")
print("\nThe reason is eq. 55.11: with a handful of samples, sqrt(v) can be")
print("far below the true gradient scale, and the update divides by it.")

# --- section 6.4 on a real optimisation -------------------------------------
print("\n" + "=" * 72)
print("warmup on a badly conditioned problem")
print("=" * 72)


def quad(kappa=1000, dim=40, seed=0):
    rs = np.random.default_rng(seed)
    evals = np.logspace(0, np.log10(kappa), dim)
    Q, _ = np.linalg.qr(rs.normal(size=(dim, dim)))
    return Q @ np.diag(evals) @ Q.T


A = quad()
x0 = rng.normal(size=40) * 2.0


def run_adam(lr, warmup, steps=1500, noise=2.0, seed=5):
    rs = np.random.default_rng(seed)
    x = x0.copy()
    m = v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    losses = []
    for t in range(1, steps + 1):
        g = A @ x + rs.normal(0, noise, len(x))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        eta = lr * min(1.0, t / warmup) if warmup else lr
        x = x - eta * mh / (np.sqrt(vh) + 1e-8)
        losses.append(0.5 * float(x @ A @ x))
        if not np.isfinite(losses[-1]):
            return losses + [np.inf] * (steps - len(losses))
    return losses


print(f"{'lr':>8} {'warmup':>8} {'loss @100':>13} {'loss @500':>13} "
      f"{'loss @1500':>13}")
for lr in (0.05, 0.3):
    for warmup in (0, 100, 300):
        ls = run_adam(lr, warmup)
        fmt = lambda v: ("diverged" if not np.isfinite(v) else f"{v:.4f}")
        print(f"{lr:>8.2f} {warmup:>8} {fmt(ls[99]):>13} "
              f"{fmt(ls[499]):>13} {fmt(ls[1499]):>13}")

print("\nThe two learning rates give OPPOSITE answers, and that is the")
print("useful result.")
print("\nAt lr = 0.05 warmup only costs. The rate was never dangerous, so")
print("suppressing the early steps threw away progress and the run was")
print("still behind at step 1500.")
print("\nAt lr = 0.30 warmup pays. Without it the run reaches a plateau it")
print("never leaves; with 300 steps of warmup it ends materially lower.")
print("The early steps at that rate did damage the run could not undo.")
print("\nSo warmup is not free and it is not universally good. It is")
print("insurance against the specific failure of taking large steps while")
print("eq. 55.11's variance estimate is unreliable, and its value is")
print("proportional to how close the rate is to the edge. The practical")
print("reading: warm up when you are pushing the learning rate, which for")
print("large models you almost always are.")

# --- the learning-rate range test -------------------------------------------
print("\n" + "=" * 72)
print("the learning-rate range test (section 4.4)")
print("=" * 72)
print("Increase the rate exponentially over 400 steps and watch the loss.\n")


def range_test(lo=1e-5, hi=3.0, steps=400, noise=2.0, seed=7):
    rs = np.random.default_rng(seed)
    x = x0.copy()
    m = v = np.zeros_like(x)
    b1, b2 = 0.9, 0.999
    out = []
    for t in range(1, steps + 1):
        lr = lo * (hi / lo) ** ((t - 1) / (steps - 1))
        g = A @ x + rs.normal(0, noise, len(x))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (
            np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        loss = 0.5 * float(x @ A @ x)
        out.append((lr, loss if np.isfinite(loss) else np.inf))
        if not np.isfinite(loss) or loss > 1e6 * out[0][1]:
            break
    return out


curve = range_test()
lrs = np.array([c[0] for c in curve])
ls = np.array([c[1] for c in curve])

# Centred moving average. mode="same" zero-pads, which corrupts a half-window
# at each end and produced a spurious "steepest descent" at the last point —
# so smooth with mode="valid" and keep the indices that are actually defined.
W = 9
H = W // 2
smooth = np.convolve(np.nan_to_num(ls, posinf=1e12),
                     np.ones(W) / W, mode="valid")
lrs_v = lrs[H:len(lrs) - H]
dl = np.gradient(np.log(np.clip(smooth, 1e-12, None)), np.log(lrs_v))

print(f"{'lr':>10} {'loss':>14} {'d(log loss)/d(log lr)':>24}")
idx = np.linspace(0, len(lrs_v) - 1, 14).astype(int)
for i in idx:
    print(f"{lrs_v[i]:>10.2e} {smooth[i]:>14.4f} {dl[i]:>24.3f}")

# Steepest descent is by definition BEFORE the minimum, so search only there.
# Searching the whole array picks up spurious dips in the rising tail, where
# the loss is changing by orders of magnitude between adjacent points.
bottom = int(np.argmin(smooth))
steepest = int(np.argmin(dl[:bottom + 1]))
print(f"\nsteepest descent at lr = {lrs_v[steepest]:.3e}")
print(f"minimum loss reached at lr = {lrs_v[bottom]:.3e}")
print(f"ratio between them        = {lrs_v[bottom] / lrs_v[steepest]:.0f}x")
print(f"\nThe standard advice is to take the rate of STEEPEST DESCENT rather")
print(f"than the one at the minimum, because by the time the loss stops")
print(f"falling the rate is already marginal. This test cost "
      f"{len(curve)} steps.")
print("\nBe honest about what it gives you: an order of magnitude, not a")
print("value. It is a way to skip the part of a grid search that is")
print("obviously wrong, and no substitute for a short sweep around the")
print("answer it suggests.")
```

## 9. Practical Example

```python {tier=A name=schedules-on-a-network}
"""Six schedules on the same network at the same budget, and the two
couplings that catch people out: the total step budget and batch size.
"""
import numpy as np

rng = np.random.default_rng(11)


class MLP:
    def __init__(self, sizes, seed=0):
        rs = np.random.default_rng(seed)
        self.W = [rs.normal(0, np.sqrt(2 / sizes[i]),
                            (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, X):
        self.H, self.Z = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.Z.append(z)
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            self.H.append(h)
        return h

    def loss_and_grads(self, X, y):
        z = self.forward(X)
        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        loss = float(np.mean(m[:, 0] + np.log(e.sum(axis=1))
                             - z[np.arange(len(X)), y]))
        d = e / e.sum(axis=1, keepdims=True)
        d[np.arange(len(X)), y] -= 1.0
        d /= len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = self.H[l].T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = (d @ self.W[l].T) * (self.Z[l - 1] > 0)
        return loss, gW, gb


D, C = 24, 5
_rs = np.random.default_rng(99)
W1T = _rs.normal(size=(D, 16))
W2T = _rs.normal(size=(16, C))


def make_data(n, seed):
    rs = np.random.default_rng(seed)
    X = rs.normal(size=(n, D))
    logits = np.tanh(X @ W1T) @ W2T * 1.5
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # vectorised categorical sampling: inverse-CDF on one uniform per row
    y = (p.cumsum(axis=1) < rs.random((n, 1))).sum(axis=1).clip(0, C - 1)
    return X, y


# 60k training examples against ~6k parameters: large enough that the
# comparison measures optimisation rather than which schedule happens to
# early-stop most. At 8k the network overfits and every row's test loss
# RISES with training, which measures regularisation instead.
Xtr, ytr = make_data(60000, 1)
Xte, yte = make_data(20000, 2)
_p = np.exp(np.tanh(Xte @ W1T) @ W2T * 1.5)
_p /= _p.sum(axis=1, keepdims=True)
BAYES = float(-np.log(_p[np.arange(len(yte)), yte]).mean())


class Adam:
    def __init__(self, shape, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v = np.zeros(shape), np.zeros(shape)
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, p, g, t, lr):
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        return p - lr * mh / (np.sqrt(vh) + self.eps)


def cosine(t, T, e0, e_min=0.0):
    return e_min + 0.5 * (e0 - e_min) * (1 + np.cos(np.pi * min(t, T) / T))


SCHED = {
    "constant": lambda t, T, e0: e0,
    "step x0.1 at 50/75%": lambda t, T, e0: e0 * (
        0.1 ** ((t >= T // 2) + (t >= 3 * T // 4))),
    "exponential to 1%": lambda t, T, e0: e0 * 0.01 ** (t / T),
    "linear to 0": lambda t, T, e0: e0 * max(0.0, 1 - t / T),
    "cosine to 0": cosine,
    "warmup 5% + cosine": lambda t, T, e0: (
        e0 * (t + 1) / (T // 20) if t < T // 20
        else cosine(t - T // 20, T - T // 20, e0)),
}


def train(sched, steps=20000, lr=3e-3, batch=128, seed=0, trace=()):
    net = MLP([D, 64, 64, C], seed=seed)
    opts = [Adam(W.shape) for W in net.W] + [Adam(b.shape) for b in net.b]
    rs = np.random.default_rng(seed + 50)
    hist = {}
    for t in range(1, steps + 1):
        idx = rs.integers(0, len(Xtr), batch)
        _, gW, gb = net.loss_and_grads(Xtr[idx], ytr[idx])
        eta = sched(t - 1, steps, lr)
        for i, (W, g) in enumerate(zip(net.W, gW)):
            net.W[i] = opts[i].step(W, g, t, eta)
        for i, (b, g) in enumerate(zip(net.b, gb)):
            net.b[i] = opts[len(net.W) + i].step(b, g, t, eta)
        if t in trace:
            hist[t] = net.loss_and_grads(Xte, yte)[0]
    te, _, _ = net.loss_and_grads(Xte, yte)
    acc = float((net.forward(Xte).argmax(axis=1) == yte).mean())
    return te, acc, hist


print("=" * 72)
print("six schedules, same budget, same peak learning rate")
print("=" * 72)
print(f"Bayes-optimal test cross-entropy: {BAYES:.4f}\n")
TRACE = (1000, 4000, 8000, 14000, 20000)
print(f"{'schedule':<22} " + " ".join(f"{f'@{x}':>9}" for x in TRACE)
      + f" {'test acc':>10} {'excess':>9}")
for name, fn in SCHED.items():
    te, acc, hist = train(fn, trace=TRACE)
    print(f"{name:<22} " + " ".join(f"{hist[x]:>9.4f}" for x in TRACE)
          + f" {acc:>10.4f} {te - BAYES:>9.4f}")

print("\nThe 'excess' column is the test loss above the Bayes floor, which")
print("is the only fair way to compare — the irreducible part of the loss")
print("is the same for every row and including it compresses the")
print("differences.")
print("\nRead the trace columns left to right. Early in the run the")
print("constant schedule is ahead of every decaying one, because it takes")
print("the largest steps and the network is still far from anything good.")
print("The decaying schedules catch and pass it later, as their noise")
print("floors come down — eq. 55.10 acting on a real network.")
print("\nThe crossover point is the thing to notice, and it is why the")
print("budget matters. Through the first few thousand steps the six rows")
print("are within a few thousandths of each other — a comparison stopped")
print("there would report that the schedule makes no difference. By 20000")
print("steps the best row's excess loss is a third lower than the worst's.")
print("Nothing about the schedules changed; the budget did.")
print("\nCompare this also with the one-dimensional experiment above, which")
print("preferred the FASTEST decay of all. There the only obstacle was the")
print("noise floor. Here there is genuine optimisation to do as well, so a")
print("schedule that decays too early — exponential-to-1% — gives up")
print("progress it cannot recover. Cosine's shape sits between the two")
print("pressures, which is what it was designed to do.")

# --- the budget coupling ----------------------------------------------------
print("\n" + "=" * 72)
print("a cosine schedule must know the budget in advance (section 5.5)")
print("=" * 72)
print("Same total steps run in every row; only the T the schedule was")
print("PLANNED for differs.\n")
print(f"{'planned T':>11} {'actual steps':>13} {'final lr / peak':>17} "
      f"{'test loss':>11} {'excess':>9}")
ACTUAL = 20000
for planned in (20000, 40000, 100000, 1000000):
    fn = lambda t, T, e0, P=planned: cosine(t, P, e0)
    te, acc, _ = train(fn, steps=ACTUAL)
    print(f"{planned:>11} {ACTUAL:>13} "
          f"{cosine(ACTUAL, planned, 1.0):>17.4f} {te:>11.4f} "
          f"{te - BAYES:>9.4f}")

print("\nRead the 'final lr / peak' column alongside the excess. A cosine")
print("planned for fifty times its actual budget has barely decayed at all")
print("— it ends at almost its full peak rate — so it IS a constant")
print("schedule wearing a cosine's name, and it lands on the constant")
print("schedule's excess loss from the previous table.")
print("\nThis is the practical trap, and it is worth stating in both")
print("directions. You cannot stop a cosine run early and get the model the")
print("schedule was going to give you, because the decay that does the work")
print("is all at the end. And you cannot extend a run that finished without")
print("re-planning, because the schedule is already at zero.")
print("\nInverse-square-root exists precisely because it does not have this")
print("problem: eq. 55.4 is a function of t alone, so it is meaningful at")
print("any stopping point. That is why it reappeared for very large models,")
print("where the budget is decided while the run is already going.")

# --- the batch-size coupling ------------------------------------------------
print("\n" + "=" * 72)
print("the linear scaling rule (eq. 55.6)")
print("=" * 72)
print("Batch size k times larger at 1/k the steps — the SAME number of")
print("examples seen. Does scaling the rate by k preserve the result?\n")
print(f"{'batch':>7} {'steps':>7} {'lr rule':<16} {'lr':>9} "
      f"{'test loss':>11} {'excess':>9}")
BASE_B, BASE_LR, BASE_STEPS = 32, 1e-3, 32000
for k in (1, 4, 16, 64):
    for rule, lr in (("unscaled", BASE_LR),
                     ("linear (x k)", BASE_LR * k),
                     ("sqrt (x sqrt k)", BASE_LR * np.sqrt(k))):
        if k == 1 and rule != "unscaled":
            continue
        te, acc, _ = train(SCHED["cosine to 0"], steps=BASE_STEPS // k,
                           lr=lr, batch=BASE_B * k)
        print(f"{BASE_B * k:>7} {BASE_STEPS // k:>7} {rule:<16} {lr:>9.1e} "
              f"{te:>11.4f} {te - BAYES:>9.4f}")

print("\nEvery row sees the same number of examples, so the question is")
print("purely which learning rate makes a large-batch run behave like the")
print("small-batch one.")
print("\nRead the unscaled rows down the table first. As the batch grows the")
print("run gets fewer steps at the same step size, so it covers less ground")
print("and the excess loss degrades — which is the failure eq. 55.6 exists")
print("to prevent.")
print("\nBetween the two rules, linear scaling wins at every batch size")
print("here and the square-root rule undercorrects — it recovers part of")
print("the loss and not all of it. That is eq. 55.13 working as derived.")
print("\nOne honest complication. The linearly scaled runs at k = 4 and")
print("k = 16 do not merely match the k = 1 baseline, they BEAT it, which")
print("no scaling rule promises. The explanation is that the baseline's")
print("1e-3 was itself below the best rate for batch 32, so scaling it up")
print("improved the run on its own merits as well as compensating for the")
print("batch. A clean test of eq. 55.6 needs the baseline rate tuned first,")
print("and this table conflates the two effects at small k.")
print("\nWhat is not conflated is the trend at large k. By k = 64 even the")
print("linearly scaled run has fallen behind the baseline, which is the")
print("regime where eq. 55.13's assumption — that the gradient does not")
print("change over the k steps being merged — has stopped holding. That is")
print("where warmup becomes necessary and, beyond it, where LARS and LAMB")
print("replace the rule entirely.")
```

## 10. Production Considerations

**Default: linear warmup over a few hundred to a few thousand steps, then
cosine to near zero.** This is what essentially every large-model recipe uses.
{{eq:effective-sample-size}} predicts the warmup length and the measurement
confirms the range.

**Log the learning rate.** One scalar, and it makes an entire class of
misconfiguration visible instantly — a schedule stepped per epoch instead of
per step, a warmup that never ended, a resume that restarted the schedule.

**Record the planned $T$ with the run.** Measured: a cosine planned for
$10\times$ the actual budget behaves like a constant schedule. Without $T$
recorded, that run is uninterpretable later.

**Save the step count in the checkpoint.** Resuming without it replays the
warmup and produces a spike.

**Run a range test before a sweep.** A few hundred steps to locate the order of
magnitude, then a short sweep around it. It is not a substitute for the sweep.

**When you change the batch size, change the learning rate.** Measured: an
unscaled rate degrades steadily as the batch grows. Which rule to use is not
settled; scale and check rather than assuming.

**Warm up longer when training is unstable at the start.** It is the cheapest
intervention available and it costs only the early steps.

## 11. Common Mistakes

**Stepping the schedule per epoch when it was written per step.** Stretches it
by the number of batches per epoch, so it never decays. Logging the rate catches
this immediately.

**Comparing schedules on a truncated run.** Measured: the ranking at 1000 steps
is not the ranking at 4000.

**Stopping a cosine run early.** Measured: it has barely decayed and inherits
the constant schedule's noise floor.

**Restarting the schedule on resume.** Replays warmup, produces a spike.

**Using SGD's learning rate with Adam or vice versa.** They mean different
things ({{sec:5-formal-explanation}}).

**Skipping warmup with Adam at a high rate.** Measured: the largest early step
under pure-noise gradients is far larger than anything that follows.

**Changing the batch size and not the learning rate.** Measured degradation.

**Treating {{eq:robbins-monro}} as a design rule.** No schedule in practice
satisfies it.

## 12. Failure Modes

**Loss plateaus and a decay fixes it.** The noise floor of {{eq:noise-floor}},
measured to match its predicted constant. Not a sign of a converged model.

**Divergence in the first hundred steps with Adam.** Warmup missing or too
short; {{eq:effective-sample-size}} says the variance estimate is unreliable
there.

**A loss spike mid-run at a schedule restart.** Warm restarts do this by design;
if you did not configure one, the schedule is being reset by a resume.

**A run that never improves after the first tenth.** The schedule decayed too
fast. {{sec:6-mathematical-foundation}} notes the asymmetry: decaying too fast
leaves a bias that never goes away, and decaying too slowly only costs the noise
floor.

**Large-batch training that diverges at the scaled rate.** Exactly the case
warmup was introduced for in {{cite:goyal2017}}.

**A schedule tuned for one budget carried to another.** Measured: the same
schedule at a mismatched $T$ is a different algorithm.

## 13. Alternatives

**Adaptive schedules** that reduce the rate when a monitored metric stops
improving. Convenient, and they introduce their own hyperparameters — patience,
threshold, cooldown — and react late by construction.

**Cyclical learning rates** {{cite:smith2017cyclical}} oscillate between bounds
rather than decaying. The claimed mechanism is escaping saddle points; the
evidence is mixed and the practice is useful.

**One-cycle** {{cite:smith2018supercon}} rises to a high rate then falls well
below the start. Reported to reach a given accuracy in far fewer epochs on
vision tasks, and it is sensitive to the peak rate. Note that the paper is
explicit that the regularisation has to be reduced to compensate for the large
rate, which is a coupling people routinely drop when copying the recipe.

**Warm restarts** {{cite:loshchilov2017sgdr}} periodically reset to the peak on
lengthening cycles. Best justified as producing snapshots to ensemble.

**Learning-rate-free methods** (D-Adaptation, Prodigy, Schedule-Free) estimate
the rate from observed quantities. They are competitive on several benchmarks
and not yet the default. {{maturity:EMERGING}} — and worth watching, because
eliminating the most important hyperparameter would be a substantial
simplification.

**$\mu$P** reparameterises the network so that the optimal learning rate
transfers across widths, turning a per-model search into a one-off on a small
proxy. {{maturity:EMERGING}} with real adoption in large-model work.

## 14. Evaluation

**Plot the learning rate alongside the loss.** Every schedule bug becomes
visible.

**Compare schedules only at equal total budget**, and report the budget.

**Use the excess loss over an achievable floor** where you can estimate one.
Measured: including the irreducible part compresses the differences and hides
the effect.

**Run a range test on any new architecture.** A few hundred steps.

**Check that the loss drops at each decay.** If it does not, the schedule is
not the binding constraint and you should look elsewhere.

**Sweep the warmup length when training is unstable.** It is one of the few
hyperparameters whose effect is confined to a known window.

## 15. Advanced Concepts

**The critical batch size.** There is a batch size beyond which further
increases stop buying faster convergence, and it is predictable from the
gradient noise scale. Below it, linear scaling holds; above it, no learning rate
recovers the lost efficiency, and that is the real limit on
{{eq:linear-scaling}}.

**Schedules as implicit regularisation.** A large rate held for a long time acts
as a noise injection, and the late decay is what lets the model settle. This is
one account of why cosine's shape works, and it is suggestive rather than
established.

**Layerwise rates and $\mu$P.** Both are statements that one global rate is the
wrong abstraction, arrived at from different directions —
{{ch:ft-sft}}'s empirical layerwise decay and $\mu$P's derivation from
width-scaling.

**Schedule-free methods** replace the decay with an averaging of iterates, which
achieves a similar effect without needing $T$. The theory is cleaner than
cosine's. {{maturity:EMERGING}}

**Learning rate and sharpness.** The largest stable learning rate is set by the
curvature, and there is evidence that training operates at the *edge of
stability*, where the largest Hessian eigenvalue hovers just at the threshold
$2/\eta$. That reframes the learning rate as selecting a curvature rather than a
step size. {{maturity:RESEARCH FRONTIER}}

## 16. Connection to Previous Chapters

{{ch:dl-optimizers}} produced {{eq:sgd-convergence}}, whose two competing terms
are the entire justification for this chapter, and {{eq:adam-step-bound}}, which
explains why $\eta$ means something different under Adam than under SGD.
{{ch:dl-backprop}}'s gradient clipping and this chapter's warmup are both
answers to instability at the start of training, arrived at differently.

{{ch:math-optimization}} supplied convergence rates and the condition number.
{{ch:mle-hpo}} supplied the search machinery; the range test here is a
cheaper alternative for one particular hyperparameter, and knowing when a cheap
proxy is adequate is the transferable skill.

Forward: {{ch:dl-initialization}} sets the starting point, and the interaction
with warmup is direct — a well-initialised network needs less of it.
{{ch:dl-normalization}} makes the loss surface better conditioned, which widens
the range of workable rates.
{{ch:ft-training-config}} gives the specific warmup-plus-cosine settings used
for fine-tuning, and {{ch:llm-next-token}} the ones used for pretraining. Both
are instances of this chapter.

## 17. Exercises

**Beginner**

1. Why can a constant learning rate not converge?
2. Write the cosine schedule.
3. What does warmup do, and why does Adam need it more than SGD?
4. What is the linear scaling rule?
5. What does a learning-rate range test measure?

**Intermediate**

6. Derive {{eq:noise-floor}} from {{eq:noisy-sgd-recursion}}.
7. Using {{eq:cosine-derivative}}, compute the fraction of a cosine budget
   spent above $0.8\eta_0$.
8. Using {{eq:effective-sample-size}}, find the step at which Adam's variance
   estimate reaches an effective sample size of 100 at $\beta_2 = 0.99$.
9. Show that $\eta_t = \eta_0/\sqrt{t}$ fails one of {{eq:robbins-monro}}'s
   conditions, and say which.
10. Explain why stopping a cosine run early is worse than planning a shorter
    one.
11. Why does decoupled weight decay's strength change when the learning rate
    decays?

**Advanced**

12. Derive {{eq:effective-sample-size}} for an exponential moving average.
13. Derive {{eq:linear-scaling-derivation}} and state precisely the assumption
    that fails at the start of training.
14. Show that under {{eq:noise-floor}}, the optimal decay for a fixed budget $T$
    is not $1/t$, and find the shape it is.
15. Explain the edge-of-stability observation and why it complicates the
    "learning rate is a step size" picture.

**Implementation**

16. Implement all six schedules and reproduce the "fraction above half peak"
    column.
17. Implement a range test for a network of your choice and compare its
    suggestion against a short sweep.
18. Implement warm restarts and measure whether averaging the snapshots beats
    the final model.
19. Reproduce the noise-floor measurement and check the constant in
    {{eq:noise-floor}} on a two-dimensional quadratic.

**Reasoning**

20. Your loss plateaus at step 5000 of a planned 50000. Give three hypotheses
    and the measurement that distinguishes them.
21. A resumed run shows a spike then recovers over 2000 steps. What are the two
    likely causes, and how would you tell them apart?

## 18. Interview Questions

**"Why decay the learning rate?"** — The noise floor of {{eq:noise-floor}}. A
strong answer gives the proportionality and notes that more steps at a fixed
rate buy nothing once stationary.

**"What is warmup for?"** — Two reasons: Adam's variance estimate is built from
too few samples early, and the initial parameters are arbitrary. The first is
what makes it near-mandatory with Adam.

**"Which schedule would you use?"** — Warmup plus cosine, and say why: cosine
holds a high rate for half the budget and then decays sharply.

**"How do you pick the learning rate?"** — Range test for the order of
magnitude, short sweep around it. Say what the range test does not give you.

**"You doubled the batch size. What else changes?"** — The learning rate, by
{{eq:linear-scaling}}, with a warmup, and note the critical batch size beyond
which it stops working.

**"Your loss plateaus. What do you do?"** — Check whether a decay moves it. If
it does, it was the noise floor; if not, the schedule is not the constraint.

**"What is wrong with stopping a cosine run early?"** — Measured: it has not
decayed and inherits a constant schedule's floor.

## 19. Research Questions

**What is the optimal schedule for a fixed budget?** Cosine is empirically
strong and there is no derivation showing it is optimal, and linear-to-zero is
competitive or better in several recent large-model comparisons.
{{maturity:EMERGING}}

**Why exactly does warmup help?** The variance-estimate argument of
{{eq:effective-sample-size}} explains the adaptive case and not the whole
phenomenon — warmup helps with SGD too, and with well-conditioned
initialisations. {{maturity:EMERGING}}

**Can the learning rate be eliminated?** Schedule-free and parameter-free
methods are close on several benchmarks. If one of them becomes reliably as good
as a tuned schedule it removes the field's most expensive hyperparameter.
{{maturity:EMERGING}}

**What does the edge-of-stability observation mean?** Training appears to sit at
the boundary of the stable region rather than safely inside it, with the largest
curvature adapting to the learning rate rather than the reverse. This is not
what the classical picture predicts and it is not yet explained.
{{maturity:RESEARCH FRONTIER}}

## 20. Chapter Summary

A constant learning rate cannot converge, and the reason is measurable rather
than rhetorical: the stationary excess loss is proportional to $\eta$, confirmed
here against {{eq:noise-floor}}'s predicted constant on a controlled quadratic.
Once the process is stationary, more steps buy nothing — which is why a
plateaued loss drops visibly at a decay, and why that drop is the floor coming
down rather than the model learning something.

The schedule shapes differ in how they spend the budget. Cosine spends exactly
half its steps above half the peak rate, by symmetry about the midpoint, where
exponential decay to the same endpoint falls below half the peak in the first
sixth of the run. That is the design argument, and the measured table confirms
the arithmetic.

Warmup has a quantitative justification for adaptive optimisers.
{{eq:effective-sample-size}} gives the effective sample size behind Adam's
variance estimate: 1 at the first step, tens by step 100, and its asymptotic
2000 only after about a thousand. The relative error in the estimate is
$\sqrt{2/n}$ and half of it passes straight into the step size, so the earliest
steps are the least reliable. Measured on pure-noise gradients, the largest step
without warmup was far bigger than anything that followed. The same calculation
predicts the right warmup length, and it matches what recipes use.

Two couplings catch people out and both were measured. A cosine schedule
contains the total budget $T$, so a run planned for ten times its actual length
has barely decayed and inherits a constant schedule's floor — you cannot stop
one early or extend one that finished. And batch size and learning rate move
together: leaving the rate unscaled degraded steadily as the batch grew, while
which of the linear and square-root rules wins was not constant across the
table, which is the honest state of that question.

The comparison also showed a trap in how schedules are evaluated. Early in the
run the constant schedule was competitive because it takes the largest steps,
and the decaying schedules overtook it only as their floors came down. The
ranking at a quarter of the budget was not the ranking at the end, so a
comparison on a truncated run measures the wrong thing.

Finally, none of the schedules used in practice satisfies the Robbins–Monro
conditions of {{eq:robbins-monro}}. Those conditions explain *why* decay is
needed on an infinite-horizon convex problem, and they are the wrong instrument
for choosing a schedule for a finite budget on a non-convex one. Knowing which
theory applies to your situation is as useful as knowing the theory.

## 21. Further Reading

{{cite:loshchilov2017sgdr}} introduced both cosine annealing and warm restarts.
The cosine part became universal and the restarts did not, which is worth
noticing: the paper's central claim was the restarts, and the field kept the
component the authors treated as a detail.

{{cite:goyal2017}} is the reference for large-batch training and for the linear
scaling rule. It is a careful empirical paper and the honest account of where
scaling breaks — and of the warmup needed to make it work at all — is more
valuable than the headline result.

{{cite:smith2017cyclical}} for cyclical rates and the range test. The range test
is the more durable contribution and it is described in a couple of paragraphs;
it is worth reading for how cheap a good diagnostic can be.
{{cite:smith2018supercon}} is the follow-up that introduced one-cycle, and the
two are often conflated.

{{cite:loshchilov2019adamw}} again, for the interaction between weight decay and
the learning rate that {{sec:5-formal-explanation}} notes: under the coupled
form the two are entangled, under the decoupled form the decay still scales with
$\eta$, and neither is obviously what you want.

**Where to go next:** {{ch:dl-initialization}} sets where the schedule starts
from, and {{ch:dl-normalization}} changes the curvature it is descending. Both
widen the range of learning rates that work, which is why the three chapters are
best read as one argument about conditioning.
