---
id: fm-scaling-laws
number: 82
part: IX
tier: full
status: draft
requires: [fm-pretraining, fm-datasets, tf-complexity, math-optimization,
           math-functions, ml-metrics, dl-lr-schedules]
provides: [scaling-law, power-law-fit, compute-optimal, chinchilla-optimal,
           irreducible-loss, inference-aware-scaling, overtraining,
           data-constrained-scaling, budget-allocation, extrapolation-risk]
citations: [kaplan2020scaling, hoffmann2022chinchilla, touvron2023llama,
            brown2020, gunasekar2023, lee2022dedup, wei2022emergent,
            schaeffer2023]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. State the scaling-law functional form and explain what each of its three
   terms represents.
2. Fit a power law to measured loss data and recover its exponents.
3. Derive the compute-optimal allocation under $C = 6ND$ and obtain the
   Chinchilla ratio from first principles.
4. Explain precisely what {{cite:kaplan2020scaling}} got wrong and why
   {{cite:hoffmann2022chinchilla}} reached a different answer.
5. Derive the inference-aware correction and compute the deployment-optimal
   model size for a given expected request volume.
6. State what the scaling laws do not contain, and why data quality is the
   largest omission.
7. Distinguish what scaling laws predict reliably from what they do not.

## 2. Why This Matters

**This is the chapter that decides how the money is spent.**
{{ch:fm-pretraining}} showed the token budget must be fixed before step one
because the schedule depends on it. This chapter is how that number is chosen,
and getting it wrong wastes a training run.

**It is also the field's best-documented case of a confident, universally-adopted
result being wrong.** {{cite:kaplan2020scaling}} concluded that model size should
grow much faster than data. The field followed it for two years and built models
that were, in retrospect, badly undertrained.
{{cite:hoffmann2022chinchilla}} redid the analysis and found parameters and data
should scale *equally*. The methodological difference between them is small,
specific, and worth understanding exactly — it is the third instance in this book
of the pattern {{cite:levy2015}} and {{cite:liu2019roberta}} established.

**The compute-optimal answer is not the answer you want.** Chinchilla minimises
training loss for a training budget. A deployed model is paid for on every
request forever, which makes the right model smaller and trained longer than
Chinchilla says — the argument {{cite:touvron2023llama}} acted on, and the
reason the open-weight ecosystem looks the way it does.

**And the laws predict loss, not capability.** That gap is the whole of
{{ch:fm-emergence}}, and it is why a scaling law can be accurate and still not
tell you whether the model will do the thing you need.

## 3. Prerequisites

{{ch:fm-pretraining}} for the run being budgeted and for $6ND$ in context.
{{ch:fm-datasets}} for what a "token" is worth, which the laws assume is
constant. {{ch:tf-complexity}} for the $C=6ND$ identity that the whole chapter
rests on. {{ch:math-optimization}} for constrained optimisation with Lagrange
multipliers. {{ch:math-functions}} for logarithms and power laws — a scaling law
is a straight line on a log-log plot and the algebra is easier if that is
obvious. {{ch:ml-metrics}} for fitting and evaluating.
{{ch:dl-lr-schedules}} for why the schedule interacts with the budget, which is
where Kaplan's analysis went wrong.

## 4. Intuitive Explanation

Train a series of models of increasing size on increasing amounts of data,
record the final loss of each, and plot loss against compute on log-log axes.
You get a straight line — over many orders of magnitude, with remarkably little
scatter.

**That is the entire empirical content, and it is surprising.** There is no
theoretical reason a system this complicated should be so predictable. The
practical consequence is enormous: you can train small models cheaply, fit the
line, and extrapolate to predict the loss of a model you have not trained and
cannot afford to train twice.

A straight line on log-log axes is a power law:

$$
L \approx \frac{A}{C^{\alpha}}
$$

Loss falls as compute rises, with diminishing returns — each factor of ten in
compute buys a fixed reduction in loss, and the reductions get harder to notice.

**Then the interesting question.** Compute is roughly $6ND$
({{ch:tf-complexity}}): parameters times tokens. Given a fixed budget, you can
spend it on a big model trained briefly, or a small model trained long. Which?

{{cite:kaplan2020scaling}} said: mostly on parameters. Grow the model fast, the
data slowly.

{{cite:hoffmann2022chinchilla}} said: equally. Double the model, double the data.

**They cannot both be right, and the difference matters by a factor of several.**
A 175B model trained on 300B tokens ({{cite:brown2020}}) is Kaplan-shaped. Under
Chinchilla, that compute buys a ~70B model on ~1.4T tokens, which would have
been substantially better.

The cause of the disagreement is small and specific: **Kaplan used a fixed
learning-rate schedule across runs of different lengths.**
{{ch:dl-lr-schedules}} says the cosine must decay to its minimum exactly at the
end of training. If it does not, the short runs are unfairly penalised, which
makes training longer look worse than it is, which makes parameters look like
the better investment. Chinchilla tuned the schedule per run and the conclusion
inverted.

> NOTE: This is {{cite:levy2015}} and {{cite:liu2019roberta}} for the third
> time. A published comparison did not equalise a nuisance variable, the
> conclusion was wrong, and the field adopted it for years before anyone redid
> the experiment. The variable differs each time; the failure does not.

**And then a third correction.** Chinchilla minimises loss per unit of
*training* compute. But a model is trained once and served forever. If you will
serve a billion requests, a smaller model trained past the compute-optimal point
is cheaper overall, even though its training was "wasteful". That is why
{{cite:touvron2023llama}} trained 7B models on a trillion tokens — a $D/N$ ratio
of 140 against Chinchilla's 20.

**The mental model:** a scaling law is a fitted empirical curve that predicts
loss from compute, plus an allocation rule for splitting compute between model
and data. Where it breaks down: the curve is fitted under assumptions — a fixed
data distribution, a tuned schedule, a particular architecture — and it says
nothing about capability, only about loss.

## 5. Formal Explanation

### 5.1 The functional form

{{cite:hoffmann2022chinchilla}} fits

$$
L(N, D) = \underbrace{E}_{\text{irreducible}}
 + \underbrace{\frac{A}{N^{\alpha}}}_{\text{model capacity}}
 + \underbrace{\frac{B}{D^{\beta}}}_{\text{data}}
$$ (eq:chinchilla-form)

with $N$ parameters and $D$ training tokens. The three terms:

- **$E$** is the entropy of the text itself — the irreducible term from
  {{eq:pretraining-decomposition}}. No model beats it.
- **$A/N^\alpha$** is the cost of finite capacity: a model too small to
  represent the distribution.
- **$B/D^\beta$** is the cost of finite data: a model that has not seen enough
  to estimate what it could represent.

The reported exponents are $\alpha \approx 0.34$ and $\beta \approx 0.28$ —
**close to each other**, which is exactly why the compute-optimal allocation
splits compute roughly evenly. That near-equality is the whole result.

### 5.2 Compute-optimal allocation

Given $C = 6ND$ fixed, minimise {{eq:chinchilla-form}}. Substituting
$D = C/(6N)$:

$$
L(N) = E + \frac{A}{N^{\alpha}} + B\left(\frac{6N}{C}\right)^{\beta}
$$ (eq:loss-single-variable)

Setting $\dd L/\dd N = 0$:

$$
-\frac{\alpha A}{N^{\alpha+1}} + \frac{\beta B\,6^{\beta}N^{\beta-1}}{C^{\beta}} = 0
$$

$$
\implies N^{\alpha+\beta} = \frac{\alpha A\,C^{\beta}}{\beta B\, 6^{\beta}}
\implies N^* \propto C^{\frac{\beta}{\alpha+\beta}}
$$ (eq:n-optimal)

and correspondingly

$$
D^* = \frac{C}{6N^*} \propto C^{\frac{\alpha}{\alpha+\beta}}
$$ (eq:d-optimal)

$\square$

**With $\alpha \approx \beta$, both exponents are $\approx 1/2$**: parameters and
tokens should each scale as the square root of compute, so doubling compute means
multiplying both by $\sqrt{2}$. That is Chinchilla's headline.

The empirical ratio at the fitted constants:

$$
\frac{D^*}{N^*} \approx 20\ \text{tokens per parameter}
$$ (eq:chinchilla-ratio-2)

### 5.3 What Kaplan found instead

{{cite:kaplan2020scaling}} reported $N^* \propto C^{0.73}$ and
$D^* \propto C^{0.27}$ — parameters growing nearly three times as fast as data.

The difference traces to the learning-rate schedule. Kaplan used one cosine
schedule length across runs of differing duration, so a run stopped early was
stopped at a point where its learning rate had not decayed. From
{{ch:dl-lr-schedules}}, a model evaluated mid-decay is materially worse than the
same model with a schedule tuned to end there.

**The effect is systematic in one direction.** It penalises long-data runs,
making data look less valuable than it is, which biases the allocation toward
parameters. Chinchilla fitted a separate, correctly-terminated schedule for each
run and the exponents changed from $(0.73, 0.27)$ to approximately
$(0.5, 0.5)$.

> IMPORTANT: Both papers are careful, both fit hundreds of models, and both
> report tight confidence intervals. Statistical rigour did not protect against
> a systematic error in the experimental design — which is the reason
> {{sec:14-evaluation}} insists on asking what was held fixed before asking how
> tight the fit is.

### 5.4 The inference-aware correction

Chinchilla optimises training cost alone. Total lifetime cost adds inference:

$$
C_{\text{total}} = \underbrace{6ND}_{\text{training, once}}
 + \underbrace{2N R}_{\text{inference}}
$$ (eq:lifetime-cost-2)

where $R$ is the expected number of tokens generated over the model's life and
$2N$ is per-token inference cost ({{ch:tf-complexity}}).

Minimising {{eq:lifetime-cost-2}} subject to a target loss $L(N,D) = L_0$ gives
a smaller $N$ and larger $D$ than {{eq:n-optimal}}, because the inference term
depends on $N$ alone and grows without bound in $R$.

**The practical rule:** the more you will serve, the smaller and longer-trained
the model should be. {{sec:8-implementation}} computes the crossover, and for
realistic serving volumes it lands far past $D/N = 20$ — which is exactly what
{{cite:touvron2023llama}} did at $D/N \approx 140$.

### 5.5 What the laws do not contain

{{eq:chinchilla-form}} has three variables: $N$, $D$, and fitted constants. It
does not contain:

- **Data quality.** Every token counts once. {{cite:gunasekar2023}} is the direct
  challenge; {{cite:lee2022dedup}} shows duplicated tokens are worth less than
  unique ones, so even $D$ is not well defined without a deduplication policy.
- **Data distribution.** A law fitted on web text does not transfer to a
  different mixture, and the constants are mixture-specific.
- **Architecture.** The fit is for a family. Mixture-of-experts models break the
  $2N$-per-token identity entirely ({{ch:res-moe}}).
- **Capability.** The law predicts loss. Whether a given loss corresponds to
  being able to write correct code is a separate, much harder question — the
  subject of {{ch:fm-emergence}}.

## 6. Mathematical Foundation

### 6.1 Fitting a power law

A pure power law $L = A C^{-\alpha}$ is linear in logs:

$$
\log L = \log A - \alpha \log C
$$ (eq:log-linear)

so ordinary least squares on $(\log C, \log L)$ recovers $\alpha$ as the slope.

**The irreducible term breaks this.** With $L = E + AC^{-\alpha}$, the log-log
plot bends toward horizontal as $L \to E$, and fitting a straight line through
the bent region underestimates $\alpha$. The fix is to fit all three parameters
jointly by nonlinear least squares — or, if $E$ is known, to fit
$\log(L - E)$ against $\log C$, which is straight again.

$\square$

**This is the most common practical error in scaling-law work**, and it biases
in a predictable direction: including near-converged points in a straight-line
fit makes scaling look worse than it is.

### 6.2 The allocation, with a Lagrange multiplier

Minimise {{eq:chinchilla-form}} subject to $6ND = C$. The Lagrangian is

$$
\mathcal{J} = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
 + \lambda\big(6ND - C\big)
$$ (eq:scaling-lagrangian)

Stationarity:

$$
\frac{\partial\mathcal{J}}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}}
 + 6\lambda D = 0,
\qquad
\frac{\partial\mathcal{J}}{\partial D} = -\frac{\beta B}{D^{\beta+1}}
 + 6\lambda N = 0
$$

Dividing the first by the second eliminates $\lambda$:

$$
\frac{\alpha A / N^{\alpha+1}}{\beta B / D^{\beta+1}} = \frac{D}{N}
\implies
\frac{\alpha A}{N^{\alpha}} = \frac{\beta B}{D^{\beta}}
$$ (eq:balanced-marginals)

$\square$

**The optimum equalises the two terms weighted by their exponents.** That is the
economically intuitive statement: spend until the marginal loss reduction per
FLOP is equal across the two ways of spending. With $\alpha\approx\beta$ it
reduces to equalising the terms themselves, which is why the allocation is
balanced.

### 6.3 The inference-aware optimum

Fix a target loss $L_0$. From {{eq:chinchilla-form}}, the achievable $(N,D)$
pairs satisfy

$$
\frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} = L_0 - E \equiv \ell
$$ (eq:iso-loss)

which defines an iso-loss curve. Minimise {{eq:lifetime-cost-2}} along it.
Solving {{eq:iso-loss}} for $D$:

$$
D = \left(\frac{B}{\ell - A N^{-\alpha}}\right)^{1/\beta}
$$ (eq:d-of-n)

and substituting into $C_{\text{total}} = 6ND + 2NR$ gives a
one-dimensional minimisation in $N$ — done numerically in
{{sec:8-implementation}}, since the derivative has no clean closed form.

**The qualitative result is robust to the constants.** As $R\to\infty$ the
$2NR$ term dominates, so the optimum drives $N$ down to the smallest value for
which {{eq:d-of-n}} has a solution — that is, until $A N^{-\alpha} = \ell$ and no
amount of data can reach the target. **There is a hard floor on model size for a
given loss**, and the inference-aware optimum sits just above it for
high-volume deployments.

### 6.4 A worked allocation

Take $\alpha = 0.34$, $\beta = 0.28$, and a budget of
$C = 10^{22}$ FLOPs.

From {{eq:n-optimal}}, $N^* \propto C^{\beta/(\alpha+\beta)}
= C^{0.28/0.62} = C^{0.452}$ and $D^*\propto C^{0.548}$.

Using the Chinchilla ratio directly, $C = 6ND$ with $D = 20N$:

$$
10^{22} = 6N(20N) = 120N^2
\implies N = \sqrt{\frac{10^{22}}{120}} = 9.1\times10^{9}
$$

$$
D = 20N = 1.8\times10^{11}
$$

**A 9B model on 180B tokens.** Compare against a Kaplan-style allocation at the
same budget, which would put roughly three times as much into parameters and
correspondingly less into data — a ~30B model on ~55B tokens, which
{{eq:chinchilla-form}} predicts is worse at identical cost.

## 7. Internal Mechanics

```mermaid {#fig:scaling-decisions caption="How a compute budget becomes a run plan. The three corrections on the right are applied in order, and each moves the answer toward a smaller model trained on more tokens than the one before it."}
graph TD
  A["compute budget C<br/>(cluster x days)"] --> B["fit L(N,D) on small runs<br/>eq:chinchilla-form"]
  B --> C["compute-optimal split<br/>eq:n-optimal, D/N ~ 20"]
  C --> D["correction 1: inference<br/>expected request volume R"]
  D --> E["correction 2: data<br/>are there enough UNIQUE tokens?"]
  E --> F["correction 3: serving<br/>does N fit the memory budget?"]
  F --> G["final N, D<br/>and the schedule length"]
  style C fill:#fde,stroke:#c69
  style G fill:#dfe,stroke:#5a5
```

**Why fitting requires small runs, plural.** A single model gives one point and
no slope. The method is a *sweep*: train a grid of model sizes on a grid of token
counts, all with correctly-terminated schedules, and fit
{{eq:chinchilla-form}} to the resulting surface. The sweep costs a few per cent
of the final run and is what makes the final run predictable.

**Extrapolation risk.** The fit is reliable over the range it was fitted on and
degrades outside it. Extrapolating two orders of magnitude beyond the largest
fitted point is standard practice and is an act of faith — the whole history in
{{sec:5-formal-explanation}} is what happens when the faith is misplaced.

**Where data runs out.** {{eq:d-optimal}} says tokens should grow with compute,
and {{ch:fm-datasets}} says unique high-quality tokens are finite. At large
enough budgets the compute-optimal $D$ exceeds the available corpus, at which
point the choice is repeating data, accepting a smaller $D$, or generating data
— which is the data-constrained regime in {{sec:15-advanced-concepts}}.

## 8. Implementation

Fitting the law, recovering the exponents, and confirming the allocation falls
out of the fit rather than being assumed.

```python {tier=A name=fitting-scaling-laws}
"""Fit L(N,D) = E + A/N^a + B/D^b and recover the compute-optimal allocation."""
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

rng = np.random.default_rng(0)

# Ground truth we will try to recover from noisy "measurements".
TRUE = dict(E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28)


def loss_true(N, D):
    return (TRUE["E"] + TRUE["A"] / N ** TRUE["alpha"]
            + TRUE["B"] / D ** TRUE["beta"])


# A sweep: a grid of model sizes and token counts, as a real fit would use.
Ns = np.array([3e7, 1e8, 3e8, 1e9, 3e9, 1e10])
Ds = np.array([1e9, 3e9, 1e10, 3e10, 1e11, 3e11])
grid_N, grid_D = np.meshgrid(Ns, Ds, indexing="ij")
obs = loss_true(grid_N, grid_D) * (1 + 0.01 * rng.normal(size=grid_N.shape))

print(f"sweep: {len(Ns)} model sizes x {len(Ds)} token counts "
      f"= {obs.size} runs, 1% observation noise\n")


def model(X, E, A, alpha, B, beta):
    N, D = X
    return E + A / N ** alpha + B / D ** beta


popt, _ = curve_fit(
    model, (grid_N.ravel(), grid_D.ravel()), obs.ravel(),
    p0=[1.0, 100.0, 0.3, 100.0, 0.3],
    bounds=([0, 0, 0.05, 0, 0.05], [10, 1e5, 1.0, 1e5, 1.0]), maxfev=200_000)

names = ["E", "A", "alpha", "B", "beta"]
print(f"{'parameter':>10} {'true':>12} {'recovered':>12} {'error':>10}")
for name, fitted in zip(names, popt):
    truth = TRUE[name]
    print(f"{name:>10} {truth:>12.4f} {fitted:>12.4f} "
          f"{abs(fitted - truth) / truth:>9.1%}")

E_f, A_f, a_f, B_f, b_f = popt

# --- the allocation, from the fitted exponents (eq:n-optimal) ---------------
print(f"\nexponent-implied scaling: N* ~ C^{b_f / (a_f + b_f):.3f}, "
      f"D* ~ C^{a_f / (a_f + b_f):.3f}")
print(f"(Chinchilla reports ~0.5 and ~0.5; Kaplan reported 0.73 and 0.27)")


def optimal_split(C):
    """Minimise the fitted loss along the C = 6ND constraint."""
    def nl(log_n):
        N = np.exp(log_n)
        D = C / (6 * N)
        return E_f + A_f / N ** a_f + B_f / D ** b_f
    r = minimize_scalar(nl, bounds=(np.log(1e6), np.log(1e13)), method="bounded")
    N = float(np.exp(r.x))
    return N, C / (6 * N), float(r.fun)


print(f"\n{'budget C':>12} {'N*':>12} {'D*':>14} {'D*/N*':>8} {'loss':>8}")
for C in (1e19, 1e20, 1e21, 1e22, 1e23, 1e24):
    N, D, L = optimal_split(C)
    print(f"{C:>12.0e} {N / 1e9:>11.2f}B {D / 1e9:>13.0f}B {D / N:>8.1f} "
          f"{L:>8.4f}")

print(f"""
Nothing here assumed a ratio: the sweep was fitted and the allocation fell out
of the exponents. Two things to read off.

First, both scaling exponents are near 1/2 ({b_f / (a_f + b_f):.2f} and """
      f"""{a_f / (a_f + b_f):.2f}), which is Chinchilla's headline — parameters
and tokens grow together, not parameters three times faster.

Second, D*/N* is NOT constant. It drifts upward with budget, because alpha and
beta are close but not equal: D*/N* scales as C^{(a_f - b_f) / (a_f + b_f):.3f}.
The famous "20 tokens per parameter" is the value at the scale Chinchilla
itself was trained at, not a law. Quoting it at a budget three orders of
magnitude away is an extrapolation, and this column is what it extrapolates
to.""")

# --- the fitting trap of section 6.1 ---------------------------------------
# Hold D fixed and vary N, so the exponent being recovered is alpha itself.
D_fixed = 3e11
N_sweep = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
L_sweep = loss_true(N_sweep, D_fixed)

# Wrong: regress log L on log N, ignoring the irreducible floor.
naive_slope, _ = np.polyfit(np.log(N_sweep), np.log(L_sweep), 1)

# Right: subtract the floor first, then the relationship is a true power law.
floor = E_f + B_f / D_fixed ** b_f          # everything not attributable to N
corrected_slope, _ = np.polyfit(np.log(N_sweep), np.log(L_sweep - floor), 1)

print(f"\nrecovering alpha (true value {TRUE['alpha']:.3f}) at fixed D:")
print(f"  naive     log L      vs log N : {-naive_slope:.4f}  "
      f"({-naive_slope / TRUE['alpha'] - 1:+.0%} error)")
print(f"  corrected log(L-floor) vs log N: {-corrected_slope:.4f}  "
      f"({-corrected_slope / TRUE['alpha'] - 1:+.0%} error)")
print("The naive fit is badly biased DOWNWARD, because most of L is a floor "
      "that does not respond to N at all. Subtract the floor first.")
```

Now the correction that changes the answer in practice:

```python {tier=A name=inference-aware-scaling}
"""Training compute is paid once; inference is paid forever. Where is the optimum?"""
import numpy as np
from scipy.optimize import minimize_scalar

E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def tokens_for_loss(N, target):
    """Invert eq:chinchilla-form for D — equation (eq:d-of-n)."""
    residual = target - E - A / N ** ALPHA
    if residual <= 0:
        return np.inf              # this N cannot reach the target at any D
    return (B / residual) ** (1 / BETA)


TARGET_LOSS = 2.10

# The hard floor from section 6.3: the smallest N that can reach the target.
n_floor = (A / (TARGET_LOSS - E)) ** (1 / ALPHA)
print(f"target loss {TARGET_LOSS}")
print(f"model-size floor (no D suffices below this): {n_floor / 1e9:.2f}B\n")

print(f"{'serving R (tokens)':>20} {'N*':>10} {'D*':>12} {'D*/N*':>8} "
      f"{'train FLOPs':>13} {'infer FLOPs':>13}")
for R in (0, 1e11, 1e12, 1e13, 1e14, 1e15):
    def total_cost(log_n):
        N = np.exp(log_n)
        D = tokens_for_loss(N, TARGET_LOSS)
        if not np.isfinite(D):
            return 1e30
        return 6 * N * D + 2 * N * R

    res = minimize_scalar(total_cost,
                          bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                          method="bounded")
    N = float(np.exp(res.x))
    D = tokens_for_loss(N, TARGET_LOSS)
    print(f"{R:>20.0e} {N / 1e9:>9.2f}B {D / 1e9:>11.0f}B {D / N:>8.0f} "
          f"{6 * N * D:>13.2e} {2 * N * R:>13.2e}")

print("""
At R = 0 the answer is the compute-optimal one: minimise training cost alone.
As serving volume grows the optimum slides toward a SMALLER model trained on
MORE tokens, because the inference term scales with N and not with D. By
R = 10^14 generated tokens the ratio is far past Chinchilla's 20 — which is the
regime real deployments are in, and the argument LLaMA acted on.

Note the floor: no amount of data reaches the target below a certain model
size, so the optimum approaches that floor from above and stops.""")
```

## 9. Practical Example

A team has a fixed cluster for six weeks and must commit to a model size. They
also know, roughly, that the model will serve about 50 million requests a month
for two years. Those two facts determine the answer together, and neither alone.

```python {tier=A name=run-budget-decision}
"""Turning a cluster booking and a traffic forecast into N and D."""
import numpy as np
from scipy.optimize import minimize_scalar

DEVICES, DEVICE_FLOPS, MFU = 512, 1e15, 0.45
WEEKS = 6
REQUESTS_PER_MONTH = 50e6
TOKENS_PER_REQUEST = 600           # prompt + completion
MONTHS_DEPLOYED = 24
UNIQUE_TOKENS = 1.5e12             # from the corpus audit, ch:fm-datasets

E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28

C_train = DEVICES * DEVICE_FLOPS * MFU * WEEKS * 7 * 86_400
R = REQUESTS_PER_MONTH * TOKENS_PER_REQUEST * MONTHS_DEPLOYED

print(f"training budget : {C_train:.3e} FLOPs ({WEEKS} weeks x {DEVICES} devices)")
print(f"serving forecast: {R:.3e} tokens over {MONTHS_DEPLOYED} months")
print(f"unique tokens   : {UNIQUE_TOKENS:.2e} available\n")


def loss(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


def solve(C, cap_D=None):
    """Best (N, D) on the C = 6ND constraint, optionally capping D."""
    def objective(log_n):
        N = np.exp(log_n)
        D = C / (6 * N)
        if cap_D is not None and D > cap_D:
            return 1e30                      # infeasible: not enough tokens
        return loss(N, D)
    lo = np.log(C / (6 * cap_D)) if cap_D else np.log(1e7)
    r = minimize_scalar(objective, bounds=(lo, np.log(1e12)), method="bounded")
    N = float(np.exp(r.x))
    return N, C / (6 * N)


# --- allocation 1: minimise training loss alone (compute-optimal) -----------
n_co, d_co = solve(C_train)

# --- allocation 2: minimise LIFETIME FLOPs at the loss the first achieves ---
# This is eq:lifetime-cost-2 with a target loss, and it is the honest
# comparison: same quality, different total cost.
target = loss(n_co, d_co)
n_floor = (A / (target - E)) ** (1 / ALPHA)


def tokens_for_loss(N, L0):
    residual = L0 - E - A / N ** ALPHA
    return np.inf if residual <= 0 else (B / residual) ** (1 / BETA)


def lifetime(log_n):
    N = np.exp(log_n)
    D = tokens_for_loss(N, target)
    return 1e30 if not np.isfinite(D) else 6 * N * D + 2 * N * R


r = minimize_scalar(lifetime, bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                    method="bounded")
n_ia = float(np.exp(r.x))
d_ia = tokens_for_loss(n_ia, target)

print(f"both allocations reach loss {target:.4f}; model-size floor for that "
      f"loss is {n_floor / 1e9:.2f}B\n")
print(f"{'allocation':<20} {'N':>9} {'D':>11} {'D/N':>7} "
      f"{'train FLOPs':>12} {'infer FLOPs':>12} {'lifetime':>12}")
for label, N, D in [("compute-optimal", n_co, d_co),
                    ("inference-aware", n_ia, d_ia)]:
    tr, inf = 6 * N * D, 2 * N * R
    print(f"{label:<20} {N / 1e9:>8.2f}B {D / 1e9:>10.0f}B {D / N:>7.0f} "
          f"{tr:>12.2e} {inf:>12.2e} {tr + inf:>12.2e}")

saving = (6 * n_co * d_co + 2 * n_co * R) - (6 * n_ia * d_ia + 2 * n_ia * R)
base = 6 * n_co * d_co + 2 * n_co * R
print(f"\nlifetime saving from the inference-aware choice: {saving:.2e} FLOPs "
      f"({saving / base:.1%})")
print(f"inference is {2 * n_co * R / base:.1%} of lifetime cost at this traffic "
      f"— which is why the correction is small HERE.")

# When does the correction actually matter? Sweep the traffic forecast.
print(f"\n{'requests/month':>16} {'infer share':>12} {'N*':>9} {'D*/N*':>7} "
      f"{'saving':>8}")
for rpm in (5e6, 5e7, 5e8, 5e9, 5e10):
    R_i = rpm * TOKENS_PER_REQUEST * MONTHS_DEPLOYED

    def lifetime_i(log_n):
        N = np.exp(log_n)
        D = tokens_for_loss(N, target)
        return 1e30 if not np.isfinite(D) else 6 * N * D + 2 * N * R_i

    ri = minimize_scalar(lifetime_i, bounds=(np.log(n_floor * 1.001), np.log(1e12)),
                         method="bounded")
    N_i = float(np.exp(ri.x))
    D_i = tokens_for_loss(N_i, target)
    base_i = 6 * n_co * d_co + 2 * n_co * R_i
    save_i = base_i - (6 * N_i * D_i + 2 * N_i * R_i)
    print(f"{rpm:>16.0e} {2 * n_co * R_i / base_i:>11.1%} {N_i / 1e9:>8.1f}B "
          f"{D_i / N_i:>7.0f} {save_i / base_i:>7.1%}")

print("\nThe correction is worth having only once inference is a material "
      "share of lifetime cost. Below that it is a rounding error, and above it "
      "it dominates — which is why LLaMA's regime and a research run's regime "
      "give genuinely different answers.")

# --- correction 2 from fig:scaling-decisions: is there enough unique data? --
print()
for label, D in [("compute-optimal", d_co), ("inference-aware", d_ia)]:
    epochs = D / UNIQUE_TOKENS
    verdict = ("fits in unique data" if epochs <= 1
               else f"needs {epochs:.1f} epochs of repetition")
    print(f"{label:<20} D = {D / 1e12:>6.2f}T -> {verdict}")

n_cap, d_cap = solve(C_train, cap_D=UNIQUE_TOKENS)
print(f"\nre-solved with D capped at the unique-token supply:")
print(f"  N = {n_cap / 1e9:.2f}B, D = {d_cap / 1e12:.2f}T, "
      f"loss = {loss(n_cap, d_cap):.4f} "
      f"(vs {target:.4f} uncapped — a penalty of "
      f"{loss(n_cap, d_cap) - target:+.4f})")

print("""
Three inputs decided this and only one is a fact about machine learning: the
compute budget, the traffic forecast, and the size of the deduplicated corpus.

Note which one actually bound here. At this traffic level the inference-aware
correction moved N by under 10%, while the data cap forced a 2.4x change in
model size and cost real loss. The chapter's headline correction was the least
important of the three for THIS deployment — and the only way to know that was
to compute all three.""")
```

> PRODUCTION TIP: Do the data check before committing. The most common way this
> planning fails is choosing a $D$ the corpus cannot supply, discovering it a
> week in, and either repeating data unplanned or shortening the run — both of
> which invalidate the schedule that {{ch:fm-pretraining}} said must be fixed in
> advance.

## 10. Production Considerations

**Fit your own law before a large run.** The published constants are for someone
else's data mixture and architecture. A sweep costing a few per cent of the
final run gives constants that apply to your setup, and the sweep is also the
only way to catch a pipeline problem before it is expensive.

**Terminate every sweep run's schedule correctly.** This is the specific error
that made {{cite:kaplan2020scaling}} wrong, and it is easy to repeat: a sweep
where all runs share a schedule length produces biased exponents.

**Record the mixture with the law.** A fitted law is only valid for the data
distribution it was fitted on. Change the mixture and refit.

**The token budget must clear the data audit.** {{ch:fm-datasets}}'s yield
measurement determines whether $D^*$ is achievable with unique tokens. Plan the
epochs deliberately rather than discovering them.

**Report the extrapolation range.** When you quote a predicted loss, say what
range the fit covers and how far beyond it you are predicting. A prediction two
orders of magnitude out is a different kind of claim from an interpolation.

**What to monitor during the run:** actual loss against the predicted curve. A
sustained divergence is information — usually a data problem — and it is
available early enough to act on.

## 11. Common Mistakes

**Beginners:**

*Fitting a straight line through the bend.* {{sec:6-mathematical-foundation}}:
with an irreducible term the log-log plot is not straight, and including
near-converged points underestimates the exponent.

*Applying Chinchilla's constants to a different setup.* The exponents are
roughly transferable; the constants $A$, $B$, $E$ are not.

*Treating $D/N = 20$ as a law.* It is the compute-optimal ratio for one fitted
setup, and it is the wrong target for anything that will be served at volume.

**Experienced practitioners:**

*Optimising training cost when inference dominates.*
{{eq:lifetime-cost-2}} is the objective a deployed model should be planned
against, and for realistic $R$ it moves the answer a long way.

*Sharing one learning-rate schedule across a sweep.* The Kaplan error. Every run
needs a schedule that terminates correctly at its own length.

*Assuming tokens are fungible.* {{cite:lee2022dedup}} shows duplicated tokens
contribute less, so $D$ measured before deduplication overstates the effective
budget.

*Extrapolating capability from loss.* The law predicts loss.
{{ch:fm-emergence}} is about why the step from loss to capability is not
straightforward, and {{cite:schaeffer2023}} is about why it can look
discontinuous even when it is not.

## 12. Failure Modes

**Biased exponents from a mis-specified sweep.** *Symptom:* an allocation that
disagrees with published ratios by a large factor. *Cause:* usually schedule
termination, sometimes fitting through the bend. *Detection:* refit with
correctly-terminated schedules on a subset and compare.

**Extrapolation failure.** The fit predicts a loss the large run does not
achieve. *Symptom:* actual loss diverging from predicted, sustained.
*Detection:* plot the prediction and the actual on the same axes from step one.

**Data exhaustion mid-run.** $D^*$ exceeds the unique corpus and the pipeline
starts repeating without anyone deciding to. *Detection:* the sampler position
from {{ch:fm-pretraining}}, plus an explicit epoch counter.

**Optimising the wrong objective.** A compute-optimal model that is too
expensive to serve. *Symptom:* a model that trained beautifully and cannot be
deployed within the latency or cost budget. *This is a planning failure and it
is fully preventable with {{eq:lifetime-cost-2}}.*

**Mixture change invalidating the law.** The corpus is improved mid-project and
the fitted constants no longer apply. *Symptom:* predictions systematically off
in one direction after a data change.

**Believing a tight confidence interval means a correct answer.** Both Kaplan
and Chinchilla reported tight intervals and reached different conclusions. The
interval measures scatter around a fit, not the validity of the design.

## 13. Alternatives

{#tbl:scaling-approaches caption="Ways to decide model size and token count. The first is what most teams actually do; the last two are what the chapter argues for, in that order."}

| Method | Inputs | Cost | Fails when |
|---|---|---|---|
| Copy a published config | none | free | your mixture or goal differs |
| Published exponents + your budget | $C$ | free | constants do not transfer |
| Fit your own law on a sweep | sweep runs | ~2–5% of the run | extrapolating too far |
| Fit + inference-aware objective | sweep, traffic forecast | same sweep | the forecast is wrong |
| Empirical search at target scale | many large runs | prohibitive | always, at frontier scale |

**What differs versus what is cheaper.** The first four all estimate the same
underlying surface with increasing fidelity. The last actually measures it and
is unaffordable, which is why the entire field runs on extrapolation — a fact
worth sitting with, since it means the central planning tool of a
multi-billion-dollar industry is a fitted curve evaluated outside its range.

**Mixture-of-experts changes the question rather than answering it.** Scaling
laws for MoE have different exponents and a different cost identity, since
active parameters and total parameters diverge ({{ch:res-moe}}).

## 14. Evaluation

**Is the fit sound?**

1. **Held-out runs.** Fit on a subset of the sweep, predict the rest, and report
   the error. A fit that has not been tested out of sample is a description, not
   a prediction.
2. **Residual structure.** Residuals should be unstructured. Systematic
   curvature in the residuals against $\log C$ means the functional form is
   wrong for your range.
3. **Schedule audit.** Confirm every sweep run's schedule terminated at its own
   end. This is the Kaplan check.
4. **Sensitivity.** Refit with the largest points removed and see whether the
   exponents move. If they do, the extrapolation is fragile.

**Is the allocation right?** It is right for an objective, and the objective is a
choice. Compute-optimal is right if the only cost is training. For anything
served, {{eq:lifetime-cost-2}} is the objective and the traffic forecast is an
input — so the evaluation question becomes "is the forecast defensible", which
is a product question rather than a machine-learning one.

**And the standing question.** Before believing any scaling comparison: what was
held fixed? For Kaplan versus Chinchilla the answer was the learning-rate
schedule, and it inverted the conclusion.

## 15. Advanced Concepts

**Data-constrained scaling.** {{maturity:EMERGING}} When unique tokens run out
before compute does, repeating data degrades gracefully for a few epochs and
then sharply. This is the direct interaction with {{ch:fm-datasets}}'s
deduplication result and is where large runs increasingly sit.

**Scaling laws for downstream tasks.** {{maturity:RESEARCH FRONTIER}} Predicting
benchmark performance rather than loss. Much less reliable, and
{{cite:wei2022emergent}} versus {{cite:schaeffer2023}} is partly an argument
about whether it is possible in principle.

**Mixture-of-experts scaling.** {{maturity:EMERGING}} Total and active
parameters scale differently, so both the loss surface and the $6ND$ identity
change form ({{ch:res-moe}}).

**Learning-rate and batch-size scaling.** {{maturity:ESTABLISHED}} The optimal
learning rate and batch size are themselves functions of scale, and getting them
wrong biases a sweep — which is precisely how the Kaplan/Chinchilla disagreement
arose.

**Quality-adjusted token counts.** {{maturity:RESEARCH FRONTIER}} Defining an
effective $D$ that weights tokens by quality or novelty, which would give
{{eq:chinchilla-form}} the term it is missing. Nobody has a definition that both
works and is measurable in advance.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:tf-complexity}}'s $C = 6ND$ is the constraint every
optimisation in this chapter is subject to, and its $2N$-per-token inference
cost is the second term of {{eq:lifetime-cost-2}}. {{ch:fm-pretraining}}'s
insistence that the token budget be fixed in advance is what makes this chapter
necessary, and its warning about schedule termination is the error that made
{{cite:kaplan2020scaling}} wrong. {{ch:fm-datasets}} determines whether $D^*$ is
achievable and shows that tokens are not fungible.
{{ch:math-optimization}} supplies the Lagrangian in
{{eq:scaling-lagrangian}}. {{eq:pretraining-decomposition}} from
{{ch:fm-what-they-are}} is the irreducible term $E$.

**Forwards.** {{ch:fm-emergence}} asks what the loss predicted here corresponds
to in capability, which is where the laws stop being useful.
{{ch:res-scaling}} revisits scaling with the frontier's current questions.
{{part:15}} makes the inference term of {{eq:lifetime-cost-2}} smaller by
quantisation, which shifts the optimum again. {{part:23}} builds the serving
systems whose cost this chapter's second term estimates.

## 17. Exercises

**Beginner**

1. A model has loss 2.5 and the irreducible term is 1.7. What fraction of the
   remaining loss could in principle be removed?
2. Using $C = 6ND$ and $D = 20N$, find $N$ for $C = 10^{23}$.
3. Why does a scaling law appear as a straight line on log-log axes?

**Intermediate**

4. Derive {{eq:n-optimal}} from {{eq:loss-single-variable}}, showing every step.
5. With $\alpha = 0.34$, $\beta = 0.28$, compute the exponents in
   {{eq:n-optimal}} and {{eq:d-optimal}} and compare against Kaplan's
   $(0.73, 0.27)$.
6. Explain why a shared learning-rate schedule across a sweep biases the
   allocation toward parameters rather than data.

**Advanced**

7. Derive {{eq:balanced-marginals}} and give its economic interpretation in one
   sentence.
8. Show that as $R\to\infty$ in {{eq:lifetime-cost-2}} the optimal $N$
   approaches the floor $(A/\ell)^{1/\alpha}$, and interpret that floor.
9. {{eq:chinchilla-form}} has no data-quality term. Propose one, state how you
   would measure it before training, and identify why that is hard.

**Implementation**

10. Extend `fitting-scaling-laws` with a held-out evaluation: fit on the four
    smallest model sizes and predict the largest. Report the prediction error
    and comment on extrapolation risk.
11. Reproduce the Kaplan bias: simulate a sweep where all runs share a schedule
    length, penalise the under-decayed runs, refit, and show the exponents move
    toward Kaplan's.
12. Add the data constraint to `run-budget-decision`: cap $D$ at the unique
    token count and re-optimise, reporting the loss penalty from the cap.
13. Implement quality-weighted tokens — an effective $D$ discounting duplicates
    by the factor from {{eq:duplication-reweighting}} — and show how the optimal
    allocation shifts.

**Reasoning**

14. Both Kaplan and Chinchilla reported tight confidence intervals and reached
    different conclusions. Explain how, and say what this implies about reading
    error bars.
15. A colleague proposes copying a frontier lab's published model configuration.
    Give the strongest argument for and the strongest against.

## 18. Interview Questions

**Beginner**

1. What is a scaling law and what does it predict?
2. What is the Chinchilla ratio?
3. Why does loss have an irreducible component?

**Intermediate**

4. Derive the compute-optimal allocation from $C = 6ND$.
5. What did Chinchilla change relative to Kaplan, and why did the answer change?
6. Why do production models often train past the compute-optimal point?

**Senior**

7. You have a cluster for six weeks. Walk through choosing $N$ and $D$.
8. When would you fit your own scaling law rather than use published exponents?
9. What do scaling laws not tell you, and what do you do about that?

**Systems**

10. Design the sweep that would let you plan a large run. How many runs, at what
    sizes, and what would you check before trusting the fit?
11. How do you monitor a long run against its predicted loss curve, and what
    would make you stop it?

## 19. Research Questions

**What is the correct data-quality term?** {{eq:chinchilla-form}} treats every
token as equal, {{cite:lee2022dedup}} shows duplicates are worth less, and
{{cite:gunasekar2023}} suggests curation is worth a great deal. Propose an
effective-token definition, fit the law with it, and see whether the constants
become more transferable across mixtures — which would be the test that it is
the right term.

**How far can a fit be trusted?** Every frontier run extrapolates well beyond
its sweep. Quantify it: fit on a range, predict beyond it, and characterise
prediction error as a function of extrapolation distance. This is measurable at
small scale and would put a number on an act of faith.

**Where exactly does repeated data stop helping?** The data-constrained regime
is increasingly the normal one. Measure the loss penalty as a function of epoch
count, separately for high- and low-quality sources, and connect it to
{{eq:duplication-reweighting}}.

**Do loss-optimal and capability-optimal allocations coincide?** The laws
optimise loss. If a different $(N, D)$ split at the same compute produced the
same loss but better downstream capability, the entire planning framework would
be optimising the wrong thing. Nobody has looked carefully.

## 20. Chapter Summary

Scaling laws fit loss as a function of parameters and tokens
{{eq:chinchilla-form}}: an irreducible term $E$ equal to the entropy of the
text, plus a capacity term $A/N^\alpha$ and a data term $B/D^\beta$. The
empirical content is that the fit is remarkably good over many orders of
magnitude, which makes a run's outcome predictable from cheap small runs.

**The allocation follows from the exponents.** Minimising
{{eq:chinchilla-form}} subject to $C = 6ND$ gives
$N^*\propto C^{\beta/(\alpha+\beta)}$ and $D^*\propto C^{\alpha/(\alpha+\beta)}$
{{eq:n-optimal}}, and because $\alpha\approx\beta$ both exponents are near
$1/2$ — parameters and tokens scale together, at roughly 20 tokens per
parameter.

**The field got this wrong for two years, and the cause was one nuisance
variable.** {{cite:kaplan2020scaling}} shared a learning-rate schedule across
runs of different lengths, which penalised the long-data runs and biased the
allocation toward parameters, giving $(0.73, 0.27)$ instead of
$(0.5, 0.5)$. Both papers fitted hundreds of models and reported tight
intervals. **Statistical rigour did not protect against a design error**, which
is the third time this book has met that pattern.

**Compute-optimal is not deployment-optimal.** {{eq:lifetime-cost-2}} adds the
inference term $2NR$, which depends on $N$ alone and grows with serving volume.
Minimising it drives the optimum toward smaller models trained on more tokens,
approaching a hard floor $(A/\ell)^{1/\alpha}$ below which no amount of data
reaches the target loss. For realistic volumes the optimum sits far past
$D/N = 20$, which is what {{cite:touvron2023llama}} acted on at $D/N \approx
140$ — and the deciding input is a traffic forecast, not a fact about machine
learning.

**The laws omit data quality entirely.** Every token counts once in
{{eq:chinchilla-form}}, while {{cite:lee2022dedup}} shows duplicates are worth
less and {{cite:gunasekar2023}} suggests curation can beat the curve at fixed
size. And the laws predict *loss*, not capability — the step from one to the
other is {{ch:fm-emergence}}'s subject and is much less settled than the
smoothness of these curves suggests.

## 21. Further Reading

{{cite:hoffmann2022chinchilla}} is the paper to read, and §3 is the argument.
Its three estimation approaches reaching the same answer is the part that makes
it convincing — a single method could be mis-specified, three agreeing is
harder to dismiss. Read §3.1's IsoFLOP analysis carefully; it is the clearest
presentation of the allocation problem in the literature.

{{cite:kaplan2020scaling}} should be read *after* Chinchilla, as a case study.
It is careful, thorough, and wrong in a way that is invisible without knowing
where to look. Read §3 asking what was held fixed across the runs, and note how
little in the paper's presentation would alert a reader to the issue.

{{cite:touvron2023llama}}'s §1 states the inference-aware argument in a
paragraph and is the clearest short version of {{eq:lifetime-cost-2}}'s
practical consequence.

{{cite:gunasekar2023}} for the missing data-quality term. It is the most direct
published challenge to the completeness of {{eq:chinchilla-form}}, and it should
be read with its contamination caveats attached rather than as a settled result.

**Where to go next:** {{ch:fm-emergence}} takes the loss these laws predict so
well and asks the question they cannot answer — what the model can actually
*do*, and whether capability arrives smoothly or suddenly.
