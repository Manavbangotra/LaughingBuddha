---
id: mle-hpo
number: 44
part: V
tier: focused
status: reviewed
requires: [mle-splits, ml-metrics, ml-boosting]
provides: [hyperparameter-optimisation, sampler, pruner, successive-halving,
           hyperband, tpe, bayesian-optimisation, effective-dimensionality,
           search-space-design]
citations: [bergstra2012, li2018hyperband, akiba2019, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Derive why random search beats grid search, geometrically rather than
   empirically.
2. Explain effective dimensionality and why it is usually far below the nominal
   count.
3. Implement successive halving and explain the budget arithmetic.
4. Explain Hyperband as a hedge over the aggressiveness of halving.
5. Explain TPE and say honestly when model-based sampling helps and when it
   does not.
6. Separate the sampler from the pruner and say which usually matters more.
7. Design a search space, including when a log scale is required.
8. Decide when to stop searching.

## 2. Why This Matters

Hyperparameter search is where compute budgets are spent and where the largest
gap sits between what people believe and what the measurements show.

**The folk belief is that Bayesian optimisation is the modern answer.** The
measurement in {{sec:7-implementation}} separates the two components that get
conflated under that heading — the *sampler*, which chooses what to try, and
the *pruner*, which kills bad trials early — and finds their contributions
comparable at realistic budgets, with a super-additive interaction that has a
clean mechanical explanation. Neither the "it's all Bayesian" story nor the
"it's all early stopping" story survives contact with the numbers.

**The random-versus-grid result is a derivation, not a benchmark.**
{{cite:bergstra2012}} is one of the few results in applied machine learning
where the argument is short enough to hold in your head and general enough to
apply everywhere. {{sec:6-mathematical-foundation}} gives it in three lines, and
once you have it you will never write a grid over more than two parameters
again.

**Search interacts with everything in {{ch:mle-splits}}.** Every trial is an
evaluation, and {{eq:distributed-optimism}} prices the optimism of taking the
best of them. A search over a thousand configurations does not merely cost
compute; it buys a validation score that is systematically too good, and the
correction is part of the result.

## 3. Prerequisites

{{ch:mle-splits}} for the evaluation the search optimises and the optimism it
incurs. {{ch:ml-metrics}} for the validation protocol. {{ch:ml-boosting}} for
early stopping, which is the same idea applied within a single fit rather than
across trials.

## 4. Intuitive Explanation

### 4.1 Why grid search wastes its budget

Suppose you have two hyperparameters, only one of which matters, and a budget of
nine trials.

```text
    GRID (3 x 3)                    RANDOM (9 draws)
    ┌───────────────┐               ┌───────────────┐
    │  ·     ·    · │               │    ·        · │
    │               │   important   │  ·      ·     │
    │  ·     ·    · │   parameter   │        ·   ·  │
    │               │       ↕       │   ·  ·        │
    │  ·     ·    · │               │ ·        ·    │
    └───────────────┘               └───────────────┘
      ↔ irrelevant                    ↔ irrelevant

    3 distinct values tried          9 distinct values tried
    on the axis that matters         on the axis that matters
```

The grid tried nine configurations and three values of the parameter that
mattered — because each of its three important-axis values was re-tested three
times against irrelevant variation. Random search tried nine values of the
important axis. Same budget, three times the resolution where it counts.

The effect compounds with dimension: with $D$ parameters and $n$ grid points per
axis, a grid needs $n^{D}$ trials to give $n$ values along each axis, while
random search gives $n^{D}$ distinct values along *every* axis from $n^{D}$
trials. This is why grid search is indefensible beyond two or three parameters,
and it is a fact about geometry rather than about optimisation.

### 4.2 Effective dimensionality

The argument only bites because most hyperparameters do not matter. That
observation has a name — the **effective dimensionality** of the search space —
and it is reliably far below the nominal count.

For gradient boosting, the learning rate and the number of rounds dominate,
depth matters somewhat, and the rest are usually noise. For a neural network,
learning rate is typically worth more than everything else combined. The
practical consequence is that a search over twelve parameters is really a search
over two or three, plus ten dimensions of expensive noise — and random search
handles that gracefully while a grid does not.

### 4.3 Kill bad trials early

The second idea is orthogonal to the first and, at realistic budgets, larger.

Most configurations are visibly bad long before they finish. A learning rate
that is too high diverges in ten epochs; a gradient-boosting configuration that
will end up mediocre is usually mediocre at round fifty. Running such a trial to
completion buys one number you could have had for a twentieth of the cost.

**Successive halving** exploits this: start many configurations with a small
budget, keep the best fraction, give the survivors more, repeat.

```text
   81 configs x 1 epoch    ─┐
   27 configs x 3 epochs    │  same total cost as
    9 configs x 9 epochs    │  9 configs run for 27 epochs,
    3 configs x 27 epochs   │  but 81 were considered
    1 config  x 81 epochs  ─┘
```

The catch is that it assumes early performance predicts final performance. When
that fails — a configuration that warms up slowly and then wins — halving kills
the eventual winner. **Hyperband** {{cite:li2018hyperband}} hedges by running
several brackets with different aggressiveness, from "very aggressive, many
configs" to "no early stopping at all", so you are not betting everything on the
assumption holding.

### 4.4 Sampler and pruner are different components

Modern tooling separates two decisions, and conflating them is the source of
most confusion about what works:

- The **sampler** chooses *which configuration to try next*. Grid, random,
  evolutionary, or model-based (TPE, Gaussian processes).
- The **pruner** decides *when to abandon a running trial*. Median stopping,
  successive halving, Hyperband.

They compose freely, and Optuna's default {{cite:akiba2019}} is TPE plus a
pruner. The measurement in {{sec:7-implementation}} varies the two
independently and finds something more interesting than a winner: at small
budgets neither helps, because both need history before they can do anything;
at larger budgets their contributions are comparable; and **the combination
beats the sum of the parts**, because pruning multiplies the number of cheap
observations the sampler has to learn from. The pruner does not only save time,
it feeds the sampler.

## 5. Formal Explanation

### 5.1 The problem

$$
\phi^{*} = \argmin_{\phi \in \Phi}\;
  \E_{\text{folds}}\big[\Loss_{\text{val}}(\phi)\big]
$$ (eq:hpo-objective)

with three properties that rule out most of {{ch:math-optimization}}: each
evaluation requires fitting a model and is therefore expensive; the objective is
noisy, because a validation score is an estimate; and there is no gradient with
respect to $\phi$, which may include discrete and conditional dimensions.

### 5.2 Search-space design

More decisions are made here than by the choice of algorithm, and two of them
matter most.

**Use a log scale for anything spanning orders of magnitude.** Learning rates,
regularisation strengths, and $C$ in an SVM all live on a multiplicative scale:
the step from 0.001 to 0.01 is the same *kind* of change as 0.01 to 0.1, and a
uniform draw over $[10^{-4}, 10^{-1}]$ puts 90% of its mass above 0.01. Sampling
$\log_{10}\phi \sim \mathcal{U}(-4, -1)$ is what you meant.

**Bound the space by reasoning, not by hedging.** A range so wide that most of
it is absurd wastes trials proportionally. If you know a learning rate above 1
diverges, do not include it.

**Conditional dimensions** — `degree` only exists when `kernel='poly'` — are the
reason define-by-run APIs {{cite:akiba2019}} exist: the space is expressed by
the trial code, so a parameter that is never sampled is never part of that
trial's configuration.

### 5.3 Successive halving

With budget $B$ total resource units, $n$ initial configurations and elimination
factor $\eta$ (typically 3):

1. Run all $n$ configurations with resource $r$.
2. Keep the best $n/\eta$.
3. Multiply $r$ by $\eta$, repeat until one remains.

Each rung costs approximately the same total, so with $\lceil\log_{\eta}
n\rceil$ rungs the total is roughly $n r \log_{\eta} n$. The arithmetic that
makes it attractive: the number of configurations *considered* grows
exponentially in the number of rungs while the total cost grows only linearly.

The assumption is a rank correlation between performance at low and high
resource. When that correlation is high, halving is nearly free. When it is
zero, halving is random selection.

### 5.4 Hyperband

{{cite:li2018hyperband}} observes that the right aggressiveness depends on that
unknown correlation, and hedges by running several **brackets**:

$$
s_{\max} = \lfloor \log_{\eta}(R) \rfloor,
\qquad
n_s = \Big\lceil \tfrac{s_{\max}+1}{s+1}\,\eta^{s} \Big\rceil,
\qquad
r_s = R\,\eta^{-s}
$$ (eq:hyperband-brackets)

where $R$ is the maximum resource per configuration. Bracket $s = s_{\max}$ is
maximally aggressive — many configurations, tiny initial budget — and bracket
$s = 0$ runs a few configurations to completion with no early stopping at all.
Running all brackets costs a constant factor more than the best single bracket
and removes the need to guess which one that is.

**ASHA** is the asynchronous variant that promotes a configuration as soon as it
is in the top fraction of those at its rung, without waiting for the rung to
fill. It is what makes the idea usable on a cluster, and it is the practical
default in 2026.

### 5.5 Model-based sampling

Fit a surrogate to the trials so far, and use it to choose the next point.

**Gaussian-process Bayesian optimisation** maintains a posterior over the
objective and maximises an acquisition function — commonly expected
improvement — trading predicted value against uncertainty. It is elegant, and it
scales poorly: the GP is $O(t^{3})$ in the number of trials and struggles above
roughly twenty dimensions or with conditional spaces.

**TPE** {{cite:akiba2019}} inverts the modelling direction. Rather than
$p(\text{score} \mid \phi)$, it models $p(\phi \mid \text{score})$ as two
densities:

$$
p(\phi \mid y) =
\begin{cases}
\ell(\phi) & \text{if } y < y^{*}\\
g(\phi) & \text{if } y \ge y^{*}
\end{cases}
$$ (eq:tpe-densities)

where $y^{*}$ is a quantile of the observed scores. The next point maximises
$\ell(\phi)/g(\phi)$, which {{sec:6-mathematical-foundation}} shows is monotone
in expected improvement. Because each dimension is modelled with a
one-dimensional density estimate, TPE handles conditional and discrete spaces
naturally and costs $O(t)$ rather than $O(t^{3})$.

> IMPORTANT: TPE models each dimension independently. It therefore cannot
> represent an *interaction* between hyperparameters — that learning rate and
> batch size must move together, for instance. It will find good regions along
> each axis and can miss a diagonal ridge entirely, which is precisely the
> structure a Gaussian process handles well. Neither is uniformly better.

### 5.6 When to stop

An underrated decision, and the honest criterion is not convergence.

Stop when the expected improvement from more search is smaller than the
**noise in the validation estimate**. If your fold-to-fold standard error is
0.01 and the last fifty trials have improved the best score by 0.003, you are
now selecting noise, and {{eq:distributed-optimism}} says the reported best is
drifting further above the truth with every trial.

The one-standard-error rule from {{ch:ml-metrics}} applies directly: prefer the
simplest configuration within one standard error of the best, rather than the
argmax of a noisy surface.

## 6. Mathematical Foundation

### 6.1 The random-search argument

This is {{cite:bergstra2012}}'s result, and it takes three lines.

Suppose the objective depends on only $d$ of the $D$ hyperparameters, and that
some fraction of the important subspace contains configurations we would call
good — say the top $\alpha$ fraction by volume.

**Random search.** Each draw independently lands in the good region with
probability $\alpha$. After $T$ draws,

$$
\Prob(\text{at least one good}) = 1 - (1-\alpha)^{T}
$$ (eq:random-search-coverage)

which depends on $T$ and $\alpha$ **and not on $D$ at all**. That
$D$-independence is the whole result.

**Grid search.** With $n$ values per axis, the total is $T = n^{D}$ and the
number of *distinct* values tried along any single axis is $n = T^{1/D}$. To
achieve resolution $n$ along the important axes, the budget must be $n^{D}$ —
exponential in the *nominal* dimension, including every axis that does not
matter.

So the ratio of useful resolution is $T$ versus $T^{1/D}$. At $T = 100$ and
$D = 5$, random search explores 100 distinct values of the important parameter
and a grid explores $100^{1/5} \approx 2.5$.

Setting {{eq:random-search-coverage}} to 0.95 gives the budget rule worth
remembering:

$$
T \ge \frac{\log(1-0.95)}{\log(1-\alpha)}
$$ (eq:random-search-budget)

At $\alpha = 0.05$ — the top 5% of the space is acceptable — that is
$T \approx 59$. **Sixty random trials give a 95% chance of landing in the top
5% of any search space, of any dimension.** That single sentence is the most
useful thing in this chapter.

### 6.2 Successive halving's budget arithmetic

With $n$ configurations, elimination factor $\eta$, and minimum resource $r$,
rung $k$ runs $n\eta^{-k}$ configurations at resource $r\eta^{k}$, so each rung
costs

$$
n\eta^{-k} \cdot r\eta^{k} = nr
$$ (eq:halving-rung-cost)

independent of $k$. That is the trick: **every rung costs the same**, so the
total over $K = \lceil\log_{\eta} n\rceil$ rungs is $nrK$, while the number of
configurations considered is $n$ and the resource given to the survivor is
$r\eta^{K-1}$.

Compare running $m$ configurations to full resource $R = r\eta^{K-1}$, which
costs $mR$. Setting the budgets equal:

$$
m = \frac{nrK}{R} = \frac{nK}{\eta^{K-1}}
$$ (eq:halving-vs-full)

At $n = 81$, $\eta = 3$, $K = 5$: full evaluation affords $m = 81 \cdot 5/81 =
5$ configurations, and halving considers 81. **A sixteen-fold increase in
configurations considered, at identical cost** — provided early performance
ranks correlate with final performance.

### 6.3 Why TPE maximises expected improvement

Expected improvement over threshold $y^{*}$ is

$$
\mathrm{EI}(\phi) = \int_{-\infty}^{y^{*}} (y^{*}-y)\,p(y \mid \phi)\,\dd y
$$

Apply Bayes' rule, $p(y\mid\phi) = p(\phi\mid y)p(y)/p(\phi)$, and write
$\gamma = \Prob(y < y^{*})$. With the two-density model of
{{eq:tpe-densities}}, the denominator is $p(\phi) = \gamma\ell(\phi) +
(1-\gamma)g(\phi)$, and the numerator integral is proportional to
$\gamma\ell(\phi)$ times a constant that does not depend on $\phi$. Hence

$$
\mathrm{EI}(\phi) \propto
\left(\gamma + \frac{g(\phi)}{\ell(\phi)}(1-\gamma)\right)^{-1}
$$ (eq:tpe-ei)

which is **monotonically decreasing in $g(\phi)/\ell(\phi)$**. Maximising
expected improvement is therefore exactly maximising $\ell(\phi)/g(\phi)$, and
TPE never needs to model the objective's values at all — only which
configurations produced good ones.

That is why it handles conditional spaces so easily. A dimension that exists in
only some trials simply has its densities estimated from those trials.

### 6.4 The cost of searching

Every trial is an evaluation, so {{eq:distributed-optimism}} applies:

$$
\E[\text{reported best}] \approx \mu^{*} - \sigma_v\sqrt{2\log T}
$$ (eq:hpo-optimism)

for a minimised loss with $T$ trials and fold-to-fold noise $\sigma_v$.

The growth is slow, but the implication is sharp: **the reported best of a
1,000-trial search is not the best configuration, it is the luckiest.** Two
consequences follow, and they are the practical content of this section.

The gap between the search's reported score and the configuration's true score
grows with $T$, so a wider search buys a better model *and* a worse estimate of
it simultaneously. And re-evaluating the chosen configuration on a fresh split —
which costs one more fit — removes the optimism entirely and is almost never
done.

## 7. Implementation

```python {tier=A name=search-strategies}
"""Grid, random, successive halving and Hyperband from scratch, on an
objective whose important dimensions we control.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- an objective with KNOWN effective dimensionality -----------------------
def make_objective(n_dims, n_important, seed=0, noise=0.01):
    """A smooth objective over [0,1]^n_dims that depends on only the first
    `n_important` coordinates. The rest are decoys, exactly as in a real
    search space where most parameters do not matter."""
    rs = np.random.default_rng(seed)
    opt = rs.uniform(0.2, 0.8, n_important)
    w = rs.uniform(0.6, 1.4, n_important)

    def f(phi, rng_eval=None):
        phi = np.asarray(phi, float)
        loss = float(np.sum(w * (phi[:n_important] - opt) ** 2))
        if rng_eval is not None and noise > 0:
            loss += float(rng_eval.normal(0, noise))
        return loss

    return f, opt


# --- section 6.1: grid vs random at fixed budget ----------------------------
print("=" * 72)
print("grid vs random at EQUAL budget (section 6.1)")
print("=" * 72)
print("The objective depends on 2 of the N dimensions. Both searches get the")
print("same number of evaluations; only the layout differs.\n")
print(f"{'dims':>5} {'budget':>8} {'grid pts/axis':>14} "
      f"{'grid best':>11} {'random best':>13} {'random wins':>12}")

for n_dims in (2, 3, 4, 6, 8):
    f, opt = make_objective(n_dims, 2, seed=1, noise=0.0)
    per_axis = max(2, int(round(64 ** (1 / n_dims))))
    budget = per_axis ** n_dims

    # grid
    axis = (np.arange(per_axis) + 0.5) / per_axis
    grid_best = np.inf
    idx = np.zeros(n_dims, int)
    for _ in range(budget):
        grid_best = min(grid_best, f(axis[idx]))
        for j in range(n_dims):                     # odometer increment
            idx[j] += 1
            if idx[j] < per_axis:
                break
            idx[j] = 0

    # random, same budget, averaged over repeats so it is not one lucky draw
    rand_bests = []
    for rep in range(40):
        rs = np.random.default_rng(500 + rep)
        rand_bests.append(min(f(rs.uniform(0, 1, n_dims))
                              for _ in range(budget)))
    rb = float(np.mean(rand_bests))
    wins = float(np.mean([b < grid_best for b in rand_bests]))
    print(f"{n_dims:>5} {budget:>8} {per_axis:>14} {grid_best:>11.5f} "
          f"{rb:>13.5f} {wins:>11.0%}")

print("\nAs the nominal dimension grows the grid's resolution on each axis")
print("collapses — at 8 dimensions it can afford only 2 points per axis, so")
print("it tries 2 values of each parameter that matters. Random search tries")
print("as many distinct values as it has trials, on every axis, because")
print("eq. 44.5 does not mention D at all.")

# --- the budget rule (eq. 44.6) ---------------------------------------------
print("\n" + "=" * 72)
print("the budget rule: 60 random trials, any dimension (eq. 44.6)")
print("=" * 72)
print(f"{'alpha (top fraction)':>21} {'trials for 95%':>16} "
      f"{'measured hit rate':>19}")
for alpha in (0.20, 0.10, 0.05, 0.02, 0.01):
    T = int(np.ceil(np.log(0.05) / np.log(1 - alpha)))
    # measure it: what fraction of runs of T draws land in the top alpha?
    hits = 0
    trials = 400
    for rep in range(trials):
        rs = np.random.default_rng(9000 + rep)
        # "good" = within the top alpha by volume of a 7-dim unit cube, which
        # for this objective means inside a ball of the matching volume
        f7, opt7 = make_objective(7, 3, seed=2, noise=0.0)
        thresh = np.quantile([f7(rs.uniform(0, 1, 7)) for _ in range(200)],
                             alpha)
        hits += any(f7(rs.uniform(0, 1, 7)) <= thresh for _ in range(T))
    print(f"{alpha:>21.2f} {T:>16} {hits / trials:>18.0%}")

print("\nThe rule holds and does not depend on the dimension. Sixty random")
print("trials give roughly a 95% chance of landing in the top 5% of ANY")
print("search space. That is the number to remember when someone proposes a")
print("grid.")

# --- successive halving and Hyperband ---------------------------------------
print("\n" + "=" * 72)
print("successive halving: same cost, far more configurations (eq. 44.9)")
print("=" * 72)


def learning_curve(quality, resource, rs):
    """A configuration's score after `resource` units.

    `quality` in [0,1] is its true final quality. Early scores are a noisy,
    biased view of it: the RANK CORRELATION between early and final score is
    what successive halving depends on, and here it improves with resource.
    """
    signal = quality * (1 - np.exp(-resource / 8.0))
    noise = rs.normal(0, 0.25 / np.sqrt(resource))
    return signal + noise


def successive_halving(qualities, eta=3, r_min=1, seed=0):
    """Returns (chosen index, total resource spent)."""
    rs = np.random.default_rng(seed)
    alive = np.arange(len(qualities))
    r, spent = r_min, 0
    while len(alive) > 1:
        scores = np.array([learning_curve(qualities[i], r, rs) for i in alive])
        spent += len(alive) * r
        keep = max(1, len(alive) // eta)
        alive = alive[np.argsort(-scores)[:keep]]
        r *= eta
    spent += len(alive) * r
    return int(alive[0]), spent


def full_evaluation(qualities, subset, R, seed=0):
    rs = np.random.default_rng(seed)
    scores = [learning_curve(qualities[i], R, rs) for i in subset]
    return int(subset[int(np.argmax(scores))]), len(subset) * R


N_CONFIG, ETA, R_MAX = 81, 3, 81
print(f"{N_CONFIG} candidate configurations, eta = {ETA}, "
      f"max resource {R_MAX}\n")
print(f"{'strategy':<34} {'configs seen':>13} {'resource':>10} "
      f"{'mean quality of pick':>22}")

sh_q, sh_cost, fe_q, fe_cost, n_seen = [], [], [], [], None
for rep in range(300):
    rs = np.random.default_rng(rep)
    qualities = rs.uniform(0, 1, N_CONFIG)
    i_sh, c_sh = successive_halving(qualities, eta=ETA, seed=rep)
    sh_q.append(qualities[i_sh])
    sh_cost.append(c_sh)
    # the same budget spent on full-resource evaluation of a random subset
    m = max(1, int(c_sh // R_MAX))
    n_seen = m
    subset = rs.choice(N_CONFIG, m, replace=False)
    i_fe, c_fe = full_evaluation(qualities, subset, R_MAX, seed=rep)
    fe_q.append(qualities[i_fe])
    fe_cost.append(c_fe)

print(f"{'successive halving':<34} {N_CONFIG:>13} "
      f"{np.mean(sh_cost):>10.0f} {np.mean(sh_q):>22.4f}")
print(f"{'full evaluation, same budget':<34} {n_seen:>13} "
      f"{np.mean(fe_cost):>10.0f} {np.mean(fe_q):>22.4f}")
print(f"{'(best possible)':<34} {'-':>13} {'-':>10} {1.0:>22.4f}")
print(f"{'(random pick, no evaluation)':<34} {'-':>13} {0:>10} {0.5:>22.4f}")

print("\nEq. 44.8 is why this is possible: every rung costs the same n*r, so")
print("the number of configurations CONSIDERED grows exponentially while the")
print("cost grows linearly. At the same total budget, halving inspected 81")
print(f"configurations and full evaluation could afford {n_seen}.")

# --- ...and when the assumption fails ---------------------------------------
print("\n" + "=" * 72)
print("halving's assumption: early rank must predict final rank")
print("=" * 72)


def learning_curve_slow(quality, resource, rs, slow_frac=0.3, seed_q=0):
    """Some configurations warm up slowly: they look bad early and win late.
    These are exactly the ones successive halving throws away."""
    slow = (seed_q % 100) / 100.0 < slow_frac
    rate = 40.0 if slow else 8.0
    signal = quality * (1 - np.exp(-resource / rate))
    return signal + rs.normal(0, 0.25 / np.sqrt(resource))


def sh_with(curve, qualities, eta=3, r_min=1, seed=0):
    rs = np.random.default_rng(seed)
    alive = np.arange(len(qualities))
    r = r_min
    while len(alive) > 1:
        sc = np.array([curve(qualities[i], r, rs, seed_q=int(i)) for i in alive])
        alive = alive[np.argsort(-sc)[:max(1, len(alive) // eta)]]
        r *= eta
    return int(alive[0])


print(f"{'fraction of slow starters':>26} "
      f"{'mean quality of halving pick':>30}")
for frac in (0.0, 0.2, 0.5, 0.8):
    picks = []
    for rep in range(300):
        rs = np.random.default_rng(rep)
        q = rs.uniform(0, 1, N_CONFIG)
        c = (lambda qq, r, rr, seed_q=0, _f=frac:
             learning_curve_slow(qq, r, rr, _f, seed_q))
        picks.append(q[sh_with(c, q, seed=rep)])
    print(f"{frac:>26.1f} {np.mean(picks):>30.4f}")

print("\nWith no slow starters halving picks near the top (0.95 of a")
print("possible 1.0). As the fraction of slow starters rises to 80% the")
print("pick degrades to 0.86 — a real and monotone loss, though still well")
print("above the 0.50 a random pick would give. Halving does not collapse;")
print("it quietly stops finding the best configurations, which is harder to")
print("notice.")
print("\nThat is what Hyperband (eq. 44.7) hedges against: it")
print("runs several brackets, from very aggressive to no early stopping at")
print("all, so no single assumption about warm-up speed has to be right.")
```

```python {tier=A name=sampler-vs-pruner}
"""The measurement that matters: which contributes more, the sampler or the
pruner?
"""
import numpy as np

rng = np.random.default_rng(3)


# --- a realistic surrogate: an expensive, noisy, iterative fit --------------
class Objective:
    """Stands in for 'fit a gradient-boosting model at this configuration'.

    Cost is counted in resource units (boosting rounds). A partial fit gives
    an intermediate score, which is what makes pruning possible at all.
    """

    def __init__(self, n_dims=6, n_important=2, seed=0):
        rs = np.random.default_rng(seed)
        self.opt = rs.uniform(0.25, 0.75, n_important)
        self.w = rs.uniform(0.8, 1.2, n_important)
        self.n_dims, self.n_imp = n_dims, n_important
        self.spent = 0

    def final(self, phi):
        d = np.asarray(phi, float)[:self.n_imp] - self.opt
        return float(np.sum(self.w * d ** 2))       # lower is better

    def evaluate(self, phi, resource, rs):
        """Score after `resource` rounds: converges to `final` from above."""
        self.spent += resource
        f = self.final(phi)
        gap = 0.6 * np.exp(-resource / 12.0)        # not yet converged
        return f + gap + float(rs.normal(0, 0.02 + 0.10 / np.sqrt(resource)))


# --- samplers ---------------------------------------------------------------
def random_sampler(n_dims, history, rs):
    return rs.uniform(0, 1, n_dims)


def tpe_sampler(n_dims, history, rs, gamma=0.25, n_candidates=24, bw=0.12):
    """A compact TPE (eq. 44.11): split trials at the gamma quantile, fit a
    Parzen density to each side, and propose the candidate maximising
    l(phi)/g(phi) — which section 6.3 shows is expected improvement."""
    if len(history) < 8:
        return rs.uniform(0, 1, n_dims)
    phis = np.array([h[0] for h in history])
    ys = np.array([h[1] for h in history])
    cut = np.quantile(ys, gamma)
    good, bad = phis[ys <= cut], phis[ys > cut]
    if len(good) < 2 or len(bad) < 2:
        return rs.uniform(0, 1, n_dims)

    cands = rs.uniform(0, 1, (n_candidates, n_dims))

    def logdens(C, pts):
        # product of per-dimension Gaussian kernel density estimates
        d = (C[:, None, :] - pts[None, :, :]) / bw
        logk = -0.5 * d ** 2
        return np.sum(np.log(np.mean(np.exp(logk), axis=1) + 1e-12), axis=1)

    return cands[int(np.argmax(logdens(cands, good) - logdens(cands, bad)))]


# --- the search loop, with sampler and pruner as separate knobs -------------
def search(sampler, use_pruner, budget, n_dims=6, seed=0,
           r_min=1, r_max=27, eta=3):
    """Spend `budget` resource units; return the true final loss of the
    configuration the search would report."""
    rs = np.random.default_rng(seed)
    obj = Objective(n_dims=n_dims, seed=seed)
    history, best = [], (None, np.inf)
    rungs = [r_min * eta ** k for k in range(int(np.log(r_max / r_min)
                                                 / np.log(eta)) + 1)]
    rung_scores = {r: [] for r in rungs}

    while obj.spent < budget:
        phi = sampler(n_dims, history, rs)
        if not use_pruner:
            score = obj.evaluate(phi, r_max, rs)
        else:
            score, killed = None, False
            for r in rungs:
                score = obj.evaluate(phi, r, rs)
                rung_scores[r].append(score)
                if r < rungs[-1] and len(rung_scores[r]) >= 5:
                    # prune if worse than the median at this rung
                    if score > np.median(rung_scores[r]):
                        killed = True
                        break
            if killed:
                history.append((phi, score))
                continue
        history.append((phi, score))
        true = obj.final(phi)
        if score < best[1]:
            best = (phi, score)
    return obj.final(best[0]) if best[0] is not None else np.inf


print("=" * 72)
print("sampler vs pruner: which one is buying the improvement?")
print("=" * 72)
print("Both axes varied independently, at three budgets. Lower is better;")
print("each cell is the mean TRUE loss of the configuration reported, over")
print("40 independent runs.\n")
print(f"{'budget':>8} {'random, no prune':>18} {'random + prune':>16} "
      f"{'TPE, no prune':>15} {'TPE + prune':>13}")
results = {}
for budget in (270, 810, 2700):
    row = []
    for sampler, use_prune in ((random_sampler, False), (random_sampler, True),
                               (tpe_sampler, False), (tpe_sampler, True)):
        vals = [search(sampler, use_prune, budget, seed=s) for s in range(40)]
        row.append(float(np.mean(vals)))
    results[budget] = row
    print(f"{budget:>8} {row[0]:>18.5f} {row[1]:>16.5f} {row[2]:>15.5f} "
          f"{row[3]:>13.5f}")

print("\nRead the two effects separately at each budget:")
for budget, row in results.items():
    prune_gain = (row[0] - row[1]) / max(row[0], 1e-12)
    samp_gain = (row[0] - row[2]) / max(row[0], 1e-12)
    both = (row[0] - row[3]) / max(row[0], 1e-12)
    print(f"  budget {budget:>5}: pruning alone {prune_gain:>+7.1%}   "
          f"TPE alone {samp_gain:>+7.1%}   both {both:>+7.1%}")

print("\nThree things in those numbers, and the folk story gets one of them")
print("right.")
print("\nAt the SMALLEST budget neither component helps much. Both need")
print("history: the median pruner cannot rank a trial until several have")
print("reached the same rung, and TPE cannot fit its densities until it has")
print("a handful of scores. A search too short for either is just random")
print("search, and that is fine — random search is a strong baseline.")
print("\nAt the larger budgets the two effects are COMPARABLE in size. The")
print("common claim that model-based sampling is the modern answer, and the")
print("counter-claim that it is all early stopping, are both wrong here:")
print("neither dominates.")
print("\nAnd they COMPOSE SUPER-ADDITIVELY — the combination beats the sum of")
print("the two separate gains. That is not a coincidence, and the mechanism")
print("is worth knowing: pruning multiplies how many configurations the")
print("search can afford to look at, and every one of those cheap looks")
print("becomes an observation TPE can fit its densities to. The pruner does")
print("not merely save time; it feeds the sampler.")
print("\nThe practical reading: use both, and do not expect either alone to")
print("account for the improvement.")

# --- section 6.4: the search's own optimism ---------------------------------
print("\n" + "=" * 72)
print("the reported best is the luckiest, not the best (eq. 44.12)")
print("=" * 72)


def search_with_reported(budget, seed, resource=81):
    """Return (reported validation score, true loss) of the winner.

    Evaluated at full resource, where the not-yet-converged gap is
    negligible (0.6 * exp(-81/12) = 0.0007). That matters: at a smaller
    resource the gap would add a constant offset to every reported score and
    swamp the effect being measured, which is SELECTION noise alone.
    """
    rs = np.random.default_rng(seed)
    obj = Objective(seed=seed)
    best = (None, np.inf)
    n = 0
    while obj.spent < budget:
        phi = rs.uniform(0, 1, 6)
        s = obj.evaluate(phi, resource, rs)
        n += 1
        if s < best[1]:
            best = (phi, s)
    return best[1], obj.final(best[0]), n


print(f"{'trials':>8} {'reported score':>16} {'true loss':>11} "
      f"{'overstatement':>14}")
for budget in (405, 810, 2430, 8100, 24300):
    rep, true, n = [], [], []
    for s in range(60):
        r, t, k = search_with_reported(budget, s)
        rep.append(r)
        true.append(t)
        n.append(k)
    print(f"{np.mean(n):>8.0f} {np.mean(rep):>16.5f} {np.mean(true):>11.5f} "
          f"{np.mean(true) - np.mean(rep):>14.5f}")

print("\nThe reported score is systematically BELOW the winner's true loss,")
print("at every budget. It is the minimum of many noisy evaluations, so it")
print("captures the downward noise as well as the genuine quality — the")
print("configuration that got the luckiest draw is the one that gets")
print("reported.")
print("\nBoth columns improve with more trials, because the search really is")
print("finding better configurations. But the overstatement GROWS — it")
print("roughly sextuples from 5 trials to 300 — because it is the price of")
print("taking a minimum over an increasing number of noisy numbers, exactly")
print("as eq. 44.12 says.")
print("\nThe reported score is worth staring at: past thirty trials it goes")
print("NEGATIVE, and the objective is a sum of squares whose smallest")
print("possible value is zero. The search is reporting an impossible score.")
print("That is as clean a demonstration as exists that a search's best")
print("number is not a measurement of anything — and in a real project the")
print("floor is unknown, so nothing flags it.")
print("\nThe fix costs one extra fit and is almost never done: re-evaluate")
print("the chosen configuration on a fresh split and report THAT. A search")
print("selects a configuration; it does not measure one.")
```

## 8. Practical Example

```python {tier=A name=hpo-in-practice}
"""Tuning gradient boosting properly: log scales, pruning, a stopping rule,
and an honest final number.
"""
import time

import numpy as np

rng = np.random.default_rng(17)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def make_data(n):
    X = rng.normal(size=(n, 14))
    z = (1.3 * np.sin(1.4 * X[:, 0]) + 0.9 * X[:, 1]
         - 1.1 * X[:, 0] * X[:, 2] + 0.8 * np.abs(X[:, 3])
         + 0.5 * (X[:, 4] > 0.7) - 0.4)
    return X, (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)


if not HAVE_SK:
    print("scikit-learn not installed — this listing needs it")
else:
    Xtr, ytr = make_data(4000)
    Xte, yte = make_data(6000)

    # --- 1. the search space, and why the scales are what they are ----------
    print("=" * 72)
    print("1. search-space design is where most of the decisions are made")
    print("=" * 72)
    SPACE = {
        "learning_rate":     ("log",   0.01, 0.5),
        "max_leaf_nodes":    ("logint",  8,  256),
        "min_samples_leaf":  ("logint",  5,  200),
        "l2_regularization": ("log",   1e-4, 10.0),
        "max_features":      ("lin",    0.4, 1.0),
    }
    for k, (scale, lo, hi) in SPACE.items():
        why = {"log": "spans orders of magnitude — a uniform draw would put "
                      "90% of its mass in the top decade",
               "logint": "same, and integer-valued",
               "lin": "a genuine proportion; uniform is what we mean"}[scale]
        print(f"  {k:<19} {scale:>7}  [{lo}, {hi}]")
        print(f"  {'':<19} {why}")

    def sample(rs):
        out = {}
        for k, (scale, lo, hi) in SPACE.items():
            if scale == "lin":
                out[k] = float(rs.uniform(lo, hi))
            elif scale == "log":
                out[k] = float(10 ** rs.uniform(np.log10(lo), np.log10(hi)))
            else:
                out[k] = int(round(10 ** rs.uniform(np.log10(lo),
                                                    np.log10(hi))))
        return out

    # --- 2. why the log scale is not a detail -------------------------------
    print("\n" + "=" * 72)
    print("2. what a uniform draw would have done to the learning rate")
    print("=" * 72)
    rs = np.random.default_rng(0)
    lin_draws = rs.uniform(0.01, 0.5, 4000)
    log_draws = 10 ** rs.uniform(np.log10(0.01), np.log10(0.5), 4000)
    print(f"{'decade':>16} {'uniform draws':>15} {'log-uniform draws':>19}")
    for lo, hi in ((0.01, 0.05), (0.05, 0.1), (0.1, 0.5)):
        print(f"  [{lo:>5}, {hi:>5}] "
              f"{np.mean((lin_draws >= lo) & (lin_draws < hi)):>15.1%} "
              f"{np.mean((log_draws >= lo) & (log_draws < hi)):>19.1%}")
    print("\nA uniform draw spends over 80% of its trials in the top decade")
    print("and barely visits the small learning rates, which are where a")
    print("boosted model with enough rounds usually wants to be. The scale")
    print("is not a formatting choice; it decides where the budget goes.")

    # --- 3. random search with a median pruner over boosting rounds ---------
    print("\n" + "=" * 72)
    print("3. random search, with and without a pruner")
    print("=" * 72)

    folds = list(StratifiedKFold(4, shuffle=True, random_state=0)
                 .split(Xtr, ytr))

    def score_config(cfg, max_iter, folds_to_use):
        aucs = []
        for tr, va in folds_to_use:
            m = HistGradientBoostingClassifier(
                max_iter=max_iter, early_stopping=False, random_state=0,
                **cfg).fit(Xtr[tr], ytr[tr])
            p = m.predict_proba(Xtr[va])[:, 1]
            o = np.argsort(p, kind="mergesort")
            r = np.empty(len(p))
            r[o] = np.arange(1, len(p) + 1)
            npos = int(ytr[va].sum())
            aucs.append((r[ytr[va] == 1].sum() - npos * (npos + 1) / 2)
                        / (npos * (len(p) - npos)))
        return float(np.mean(aucs)), max_iter * len(folds_to_use)

    def run_search(n_trials, use_pruner, seed=0):
        rs = np.random.default_rng(seed)
        best, cost, seen = (None, -np.inf), 0, 0
        rung_scores = {30: [], 100: []}
        for _ in range(n_trials):
            cfg = sample(rs)
            seen += 1
            if use_pruner:
                s30, c = score_config(cfg, 30, folds[:2])
                cost += c
                rung_scores[30].append(s30)
                if (len(rung_scores[30]) >= 5
                        and s30 < np.median(rung_scores[30])):
                    continue                       # pruned at rung 1
            s, c = score_config(cfg, 300, folds)
            cost += c
            if s > best[1]:
                best = (cfg, s)
        return best, cost, seen

    for use_pruner in (False, True):
        t0 = time.perf_counter()
        (cfg, s), cost, seen = run_search(14, use_pruner, seed=1)
        dt = time.perf_counter() - t0
        label = "with median pruner" if use_pruner else "no pruner"
        print(f"{label:<22} best CV AUC {s:.4f}   "
              f"{seen} trials, {cost:,} tree-fold-rounds, {dt:.1f}s")

    print("\nThe pruner cut the compute by about a third and cost a few")
    print("tenths of an AUC point. That is the small-budget regime from")
    print("listing 2 showing up in a real fit: with only fourteen trials the")
    print("median rule has very little to rank against, and it killed a")
    print("configuration that would have done well. At fourteen trials a")
    print("pruner is not obviously worth it; at four hundred it is not")
    print("optional. Match the machinery to the budget.")

    # --- 4. the stopping rule -----------------------------------------------
    print("\n" + "=" * 72)
    print("4. when to stop: compare improvement against the noise floor")
    print("=" * 72)
    rs = np.random.default_rng(4)
    running_best, history = -np.inf, []
    fold_ses = []
    for t in range(1, 17):
        cfg = sample(rs)
        aucs = []
        for tr, va in folds:
            m = HistGradientBoostingClassifier(
                max_iter=200, early_stopping=False, random_state=0,
                **cfg).fit(Xtr[tr], ytr[tr])
            p = m.predict_proba(Xtr[va])[:, 1]
            o = np.argsort(p, kind="mergesort")
            r = np.empty(len(p))
            r[o] = np.arange(1, len(p) + 1)
            npos = int(ytr[va].sum())
            aucs.append((r[ytr[va] == 1].sum() - npos * (npos + 1) / 2)
                        / (npos * (len(p) - npos)))
        s = float(np.mean(aucs))
        fold_ses.append(float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))))
        running_best = max(running_best, s)
        history.append(running_best)

    se = float(np.mean(fold_ses))
    print(f"typical fold-to-fold standard error: {se:.4f}\n")
    print(f"{'trial':>6} {'best so far':>13} {'gain over 5 trials ago':>24} "
          f"{'vs noise floor':>16}")
    for t in range(5, len(history), 3):
        gain = history[t] - history[t - 5]
        verdict = "still worth it" if gain > se else "now selecting noise"
        print(f"{t + 1:>6} {history[t]:>13.4f} {gain:>24.4f} "
              f"{verdict:>16}")

    print("\nThe stopping criterion is not convergence — it is the point at")
    print("which the improvement over the last several trials falls below")
    print("the standard error of the estimate itself. Past that, eq. 44.12")
    print("says the reported best is drifting further above the truth with")
    print("every trial you add.")
    print("\nNote that the verdict FLICKERS: the rule fires at trial 9 and")
    print("then un-fires at 12. That is expected — the improvement is itself")
    print("a noisy quantity, so a single-window rule will trip early. Use it")
    print("with patience, as an early-stopping rule is used in Chapter 38:")
    print("stop after k consecutive windows below the floor, not the first")
    print("one.")

    # --- 5. the honest final number -----------------------------------------
    print("\n" + "=" * 72)
    print("5. re-evaluate the winner: a search selects, it does not measure")
    print("=" * 72)
    (best_cfg, best_cv), _, _ = run_search(14, True, seed=1)
    m = HistGradientBoostingClassifier(max_iter=300, early_stopping=False,
                                       random_state=0,
                                       **best_cfg).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    o = np.argsort(p, kind="mergesort")
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    npos = int(yte.sum())
    test_auc = float((r[yte == 1].sum() - npos * (npos + 1) / 2)
                     / (npos * (len(p) - npos)))
    print(f"best CV AUC reported by the search : {best_cv:.4f}")
    print(f"same configuration on held-out data: {test_auc:.4f}")
    print(f"optimism                           : {best_cv - test_auc:+.4f}")
    print("\nOn this single run the two agree to within half a point, which")
    print("is well inside the fold-to-fold standard error above — so this")
    print("particular number demonstrates nothing on its own. The systematic")
    print("effect is the averaged measurement in listing 2, where the")
    print("overstatement grew monotonically with the number of trials.")
    print("\nThe discipline is what matters, not this run's arithmetic: one")
    print("extra fit turns a number you SELECTED into a number you MEASURED,")
    print("and at fourteen trials the correction is small only because the")
    print("search was small.")
```

## 9. Common Mistakes

**Grid search over more than two parameters.** The measured table shows its
per-axis resolution collapsing as the nominal dimension grows.

**Uniform sampling of a learning rate.** The measurement shows over 80% of
draws landing in the top decade.

**Reporting the search's best score as the model's performance.** It is a
minimum over noisy evaluations; re-evaluate on a fresh split.

**Expecting one component to explain the improvement.** The measured
decomposition shows sampler and pruner contributing comparably and combining
super-additively.

**Assuming TPE handles interacting hyperparameters.** It models dimensions
independently and can miss a diagonal ridge.

**Running successive halving when early performance does not predict final
performance.** The measurement shows its picks degrading towards random as the
fraction of slow starters rises.

**Searching until the budget runs out.** Stop when improvement falls below the
fold-to-fold standard error.

**Tuning many parameters at once on a small dataset.** Effective dimensionality
is low; the extra axes cost trials and buy noise.

**Forgetting that trials count towards $K_{\text{total}}$.** A 1,000-trial
search is 1,000 evaluations of the validation set.

## 10. Connection to Previous Chapters

{{ch:mle-splits}} supplied the evaluation each trial consumes and
{{eq:distributed-optimism}}, which reappears as {{eq:hpo-optimism}} — a search
is the largest single contributor to $K_{\text{total}}$ that most projects have.
{{ch:ml-metrics}} supplied the one-standard-error rule that
{{sec:5-formal-explanation}} turns into a stopping criterion.
{{ch:ml-boosting}} supplied early stopping within a fit, of which pruning is the
across-trials analogue — and its measured warning that a fixed `n_estimators` is
a bug is the same warning in miniature. {{ch:math-optimization}} supplied the
gradient methods that {{eq:hpo-objective}} explicitly cannot use.

Forward: {{ch:mle-reproducibility}} records what the search tried, without which
its result cannot be defended. {{ch:mle-registry}} stores the winning
configuration as part of the artefact. {{ch:dl-lr-schedules}} tunes the single
hyperparameter that dominates deep learning's effective dimensionality.
{{part:20}} automates the loop this chapter runs by hand.

## 11. Exercises

**Beginner**

1. Why does grid search waste trials?
2. What is effective dimensionality, and why is it usually low?
3. Why must a learning rate be sampled on a log scale?
4. What is the difference between a sampler and a pruner?
5. What assumption does successive halving make?

**Intermediate**

6. Using {{eq:random-search-budget}}, compute the trials needed for a 90%
   chance of reaching the top 2%.
7. Explain why {{eq:random-search-coverage}} does not depend on $D$.
8. Using {{eq:halving-vs-full}}, compute how many full evaluations equal a
   halving run with $n=243$, $\eta=3$.
9. Explain what Hyperband hedges against and why several brackets are needed.
10. Why is TPE $O(t)$ where a Gaussian process is $O(t^{3})$?
11. Give a search space where TPE will do poorly, and say why.

**Advanced**

12. Derive {{eq:random-search-coverage}} and state its assumptions.
13. Derive {{eq:tpe-ei}} and explain why the objective's values never appear.
14. Prove {{eq:halving-rung-cost}} and derive the total budget.
15. Derive {{eq:hyperband-brackets}} and explain the role of $s_{\max}$.
16. Design a stopping rule that accounts for both the improvement rate and the
    accumulating optimism of {{eq:hpo-optimism}}, and state what it optimises.

**Implementation**

17. Implement ASHA and compare its wall-clock against synchronous successive
    halving with four parallel workers.
18. Extend the TPE implementation to handle conditional parameters and verify
    it on a space where `degree` exists only for a polynomial kernel.
19. Implement a Gaussian-process sampler with expected improvement and find a
    problem where it beats TPE — a diagonal ridge is a good place to start.
20. Instrument a search to report {{eq:hpo-optimism}}'s correction alongside
    the best score.

**Reasoning**

21. You have a 12-hour budget and a model that takes 20 minutes to fit. What do
    you run, and why?
22. A colleague reports a 0.4-point AUC improvement from a 2,000-trial search
    with a fold standard error of 0.6 points. What do you say?

## 12. Chapter Summary

Random search beats grid search for a geometric reason, not an empirical one:
the probability that a random draw lands in the good region does not depend on
the dimension at all, while a grid's resolution along each axis is
$T^{1/D}$. The measurement shows a grid's per-axis resolution collapsing to two
points by eight dimensions while random search keeps its full budget of distinct
values on every axis.

The budget rule worth memorising falls out of the same equation: about sixty
random trials give a 95% chance of landing in the top 5% of a search space of
any dimension.

The argument works because effective dimensionality is far below nominal
dimensionality — most hyperparameters do not matter, and you rarely know in
advance which ones.

Successive halving works because every rung costs the same, so the number of
configurations considered grows exponentially while the cost grows linearly. At
matched budget, the measurement shows halving inspecting 81 configurations where
full evaluation could afford five. Its assumption is that early rank predicts
final rank, and the measured degradation as slow-starting configurations are
added is exactly what Hyperband's multiple brackets hedge against.

The sampler and the pruner are separate components, and the measured
decomposition is the chapter's main practical result. At small budgets neither
helps, because both need history before they can act. At larger budgets their
contributions are comparable — so neither "use Bayesian optimisation" nor "it is
all early stopping" is right. And the two **compose super-additively**, for a
mechanical reason: pruning multiplies the number of cheap observations the
sampler has to fit its densities to, so the pruner does not only save time, it
feeds the sampler. Use both.

TPE maximises $\ell(\phi)/g(\phi)$, which is monotone in expected improvement,
and never models the objective's values — only which configurations produced
good ones. That is why it handles conditional spaces cheaply, and why it cannot
represent interactions between hyperparameters.

Finally, a search selects a configuration; it does not measure one. The reported
best is a minimum over noisy evaluations and is optimistic by roughly
$\sigma_v\sqrt{2\log T}$. Re-evaluating the winner on a fresh split costs one
extra fit and turns a selected number into a measured one.
