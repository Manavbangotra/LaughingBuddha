---
id: mle-splits
number: 43
part: V
tier: focused
status: reviewed
requires: [ml-metrics, ds-leakage, ds-timeseries]
provides: [split-as-code, group-split, time-split, nested-cv-implementation,
           split-audit, holdout-discipline, distribution-shift-split]
citations: [pedregosa2011, breck2017]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Write a split as auditable code rather than a one-line call, and say what
   the audit must check.
2. Choose between random, grouped, time-ordered and combined splits from the
   structure of the data.
3. Implement grouped and time-series cross-validation from scratch.
4. Implement nested cross-validation as a pipeline rather than as a concept.
5. Detect a split that has silently stopped being honest.
6. Establish a holdout discipline that survives a team and several months.
7. Explain when a deliberately shifted split is the honest one.

## 2. Why This Matters

{{ch:ds-leakage}} explained what leakage is and {{ch:ml-metrics}} explained
selection optimism. You understood both. That is not sufficient, and the reason
is the subject of this chapter.

**Understanding does not prevent the error, because the error is introduced by
code.** The leak will not come from you misunderstanding grouped data; it will
come from a `train_test_split` call written six months later by a colleague who
did not know that `user_id` repeats. Correctness here is a property of the
pipeline, not of anyone's knowledge, and the practical content is the mechanics
that make the honest thing the easy thing to write.

**A split that was honest can stop being honest without anything changing in
the code.** A random split is fine while every row is an independent event, and
becomes a leak the day an upstream team starts emitting one row per session
instead of one per user. Nothing errors. The validation score goes up. This
class of failure is why {{cite:breck2017}}'s rubric asks for tests on data and
not only on models.

**Every number in the rest of this book depends on this.** A tuned model, a
drift threshold, a promotion gate and an A/B baseline are all computed against
a split. If the split is dishonest, everything downstream is confidently wrong,
and it stays confidently wrong until production disagrees.

## 3. Prerequisites

{{ch:ds-leakage}} for the leakage mechanisms this chapter defends against
mechanically. {{ch:ml-metrics}} for selection optimism and nested
cross-validation as a concept — here it becomes an implementation.
{{ch:ds-timeseries}} for temporal ordering. {{ch:ml-linear-regression}} for the
preprocessing-inside-the-fold discipline.

## 4. Intuitive Explanation

### 4.1 The question a split is asking

A split is an attempt to simulate deployment. The validation set stands in for
data the model has not seen, and the estimate is honest exactly to the extent
that the simulation is faithful.

So the design question is not "how do I split this" but **"what will be
different about the data the model sees in production?"** Three answers cover
almost everything:

```text
  it will be DIFFERENT ROWS         -> random split
  from the SAME entities            -> group split
  from a LATER time                 -> time split
  from a NEW population             -> shifted split
```

Most data has more than one of these true at once, and the split must respect
all of them. Predicting hospital readmission for patients you have already
seen, from data collected later, means a split grouped by patient *and* ordered
by time — and the intersection is smaller and more awkward than either alone,
which is why people quietly drop one.

### 4.2 What a random split assumes

`train_test_split(X, y)` assumes rows are exchangeable: that any permutation of
them is equally plausible. That is a strong assumption and it is false whenever

- rows share an entity — several transactions per customer, several images per
  patient, several sessions per user;
- rows share a time — the model will be used on the future and evaluated on the
  past;
- rows share a source — several rows per hospital, per store, per device, and
  production will include hospitals not in the training set.

The measurement in {{sec:7-implementation}} shows what each violation is worth.
The headline is that grouped leakage is the largest and the least visible: a
model that has memorised customer-specific quirks scores beautifully on other
rows from the same customers and has learned nothing transferable.

### 4.3 A split is code, and code needs an audit

The practical failure is not choosing the wrong split. It is choosing the right
split and then having it decay.

```text
   the split you wrote          what it became
   ─────────────────────        ────────────────────────────────
   random, one row per user  →  three rows per user after an
                                upstream change: now leaking
   time-based, 6-month train →  training window now spans a
                                schema change: now inconsistent
   grouped by store          →  a store was renamed and appears
                                in both sides: now leaking
```

None of these throw. All of them raise the validation score. So a split needs
an assertion layer — a function that checks its own invariants and fails the
build when they break. {{sec:8-practical-example}} builds one.

### 4.4 The holdout that survives a team

The test set is a budget, and it is spent by looking. {{ch:ml-metrics}}
measured the optimism from selecting the best of $k$ configurations; the
organisational version is worse, because the looking is distributed. Four people
each evaluating five models on the same test set have collectively evaluated
twenty, and none of them knows it.

The mechanisms that work are mechanical rather than cultural: keep the test
labels in a separate location that the training code cannot read; expose
evaluation through a function that logs every call; and treat the log as the
$k$ in the optimism calculation. A team that cannot say how many times its test
set has been evaluated does not have a test set.

## 5. Formal Explanation

### 5.1 The exchangeability requirement

An honest estimate needs the validation set to be drawn from the same
distribution the model will face, *and* to be independent of the training set
given the model. Random splitting satisfies the second condition only when rows
are exchangeable.

Write the dependence structure explicitly. If rows carry a group label $g_i$
and the target depends on a group-level effect $u_{g}$,

$$
y_i = f(\vec{x}_i) + u_{g_i} + \epsilon_i
$$ (eq:group-effects)

then a random split places rows with the same $g$ on both sides. A flexible
model can estimate $u_g$ from the training rows of group $g$ and apply it to the
validation rows of the same group — which production will never allow, because
in production $g$ is new. The validation score measures a capability the model
will not have.

This is why {{ch:ds-leakage}}'s measurement needed k-NN rather than logistic
regression to demonstrate group leakage: memorising $u_g$ requires capacity.
The corollary is uncomfortable — **the more flexible your model, the more a
grouped leak flatters it**, so the leak grows exactly as you move to better
models.

### 5.2 The four split families

{#tbl:split-families caption="Split families and the deployment question each one simulates. Most real problems need an intersection of two or more, which is where the awkwardness lives."}

| Split | Simulates | Use when | Cost |
|---|---|---|---|
| Random / stratified | new rows, same population | rows genuinely independent | none |
| Grouped | new entities | rows share customers, patients, devices | fewer effective samples |
| Time-ordered | the future | any temporal process | cannot use all data for training |
| Shifted / by-source | a new environment | deploying to a new region, store, hospital | pessimistic by design |

**Stratification** preserves the class balance in each fold, which matters under
imbalance and is nearly free. Note that it is a variance-reduction device, not
an honesty device: stratifying a leaking split gives a more precise estimate of
the wrong number.

### 5.3 Time-based splitting

Two schemes, and the choice is a modelling statement:

**Expanding window** — train on everything up to $t$, validate on
$(t, t+\Delta]$, advance. The training set grows. Appropriate when the process
is stable and more history is better.

**Sliding window** — train on $(t-W, t]$, validate on $(t, t+\Delta]$, advance.
The training set is fixed-size. Appropriate when the process changes, so old
data is actively misleading.

```text
  expanding                        sliding
  [====train====][val]             [==train==][val]
  [======train=====][val]              [==train==][val]
  [========train======][val]               [==train==][val]
```

Two details that are routinely missed:

**The gap.** If a feature is computed over a trailing window of $w$ days, and
the label is realised $d$ days after the prediction, then a validation fold
starting immediately after the training fold leaks. There must be an embargo of
at least $\max(w, d)$ between them. This is the purged-and-embargoed
cross-validation of quantitative finance, and it applies wherever features look
backwards or labels resolve slowly.

**The last fold is the only realistic one.** Earlier folds train on less data
than production will have. Report the last fold separately as well as the mean.

### 5.4 Nested cross-validation, as code

{{ch:ml-metrics}} defined it and measured its bias. The implementation detail
that matters is that **everything selected must be selected inside the inner
loop** — hyperparameters, yes, but also feature selection, the scaler, the
imputation strategy, the encoding, the threshold, and the decision to drop a
feature at all.

$$
\text{outer fold } k: \quad
\hat{s}_k = \text{score}\Big(
  \underbrace{\text{fit}\big(\Data_{\text{train}}^{(k)},\;
     \phi^{*}(\Data_{\text{train}}^{(k)})\big)}_{\text{selection uses only the outer training data}},
  \; \Data_{\text{test}}^{(k)}\Big)
$$ (eq:nested-cv)

where $\phi^{*}(\cdot)$ is the *entire* selection procedure. If any part of
$\phi^{*}$ was run once on the whole dataset — as it invariably is, because
someone chose the feature set before writing the loop — then
{{eq:nested-cv}} is not what you computed.

> IMPORTANT: The most common form of this error is invisible because it happens
> before the code. You looked at the data, noticed a feature was useless,
> dropped it, and then cross-validated. The dropping used all the data. This is
> not usually a large leak, but it is a leak, and it is unbounded when the
> decision was made on the basis of the target.

### 5.5 When a pessimistic split is the honest one

Sometimes the deployment condition is not "more of the same". Deploying to a
new country, a new hospital, a new device generation, or simply into next year
means the honest simulation is a **shifted** split: hold out an entire source
and train on the rest.

The score will be worse, and it will be right. A leave-one-group-out estimate
across sources answers "how will this do somewhere new", which is the question
being asked, and a random split answers a question nobody asked.

Report both when they differ. The gap between them *is* the estimate of how
much of your model's performance is source-specific, and it is one of the most
informative numbers you can produce.

## 6. Mathematical Foundation

### 6.1 How much a grouped leak is worth

Under {{eq:group-effects}}, let the group effect have variance $\sigma_u^{2}$
and the residual noise variance $\sigma_{\epsilon}^{2}$. Consider a model
flexible enough to estimate each group's effect from its training rows.

With a **random** split, group $g$ has $n_g^{\text{tr}}$ rows in training. The
model estimates $\hat{u}_g$ with variance $\sigma_{\epsilon}^{2} /
n_g^{\text{tr}}$, and on validation rows of the same group its error variance is

$$
\Var[\text{error}]_{\text{random}}
 \approx \sigma_{\epsilon}^{2}\Big(1 + \tfrac{1}{n_g^{\text{tr}}}\Big)
$$ (eq:random-split-error)

With a **grouped** split, group $g$ is unseen, so $\hat{u}_g$ is whatever the
model's default is — call it $0$ — and the error carries the whole group effect:

$$
\Var[\text{error}]_{\text{grouped}}
 \approx \sigma_{\epsilon}^{2} + \sigma_u^{2}
$$ (eq:group-split-error)

The ratio is approximately $1 + \sigma_u^{2}/\sigma_{\epsilon}^{2}$, so **the
optimism of a random split is governed by the intraclass correlation**

$$
\rho_{\text{ICC}} = \frac{\sigma_u^{2}}{\sigma_u^{2} + \sigma_{\epsilon}^{2}}
$$ (eq:icc)

which is a quantity you can estimate before choosing a split, and should. At
$\rho_{\text{ICC}} = 0$ a random split is correct; at $0.5$ it is reporting an
error variance half what production will see.

The other half of {{ch:ds-leakage}}'s design effect appears here too: with
$m$ rows per group, the **effective sample size** is not $N$ but

$$
N_{\text{eff}} = \frac{N}{1 + (m-1)\rho_{\text{ICC}}}
$$ (eq:effective-n)

so a dataset of 100,000 rows from 500 customers at $\rho_{\text{ICC}} = 0.3$ has
an effective size closer to 1,600. Every confidence interval computed as if
$N = 100{,}000$ is roughly eight times too narrow.

### 6.2 Why the embargo length is $\max(w, d)$

Let a feature at time $t$ be computed over $(t-w, t]$ and a label at time $t$
resolve at $t + d$.

A training row at time $t_1$ and a validation row at time $t_2 > t_1$ share
information if either window overlaps:

- **Feature overlap.** The validation row's feature window is $(t_2 - w, t_2]$.
  It contains $t_1$ whenever $t_2 - t_1 < w$.
- **Label overlap.** The training row's label is only known at $t_1 + d$, so a
  model trained on it implicitly uses information from $t_1 + d$. That is later
  than $t_2$ whenever $t_2 - t_1 < d$.

Requiring both to be excluded gives $t_2 - t_1 \ge \max(w, d)$, and the embargo
is that gap applied at every fold boundary.

The label-overlap half is the one people miss, and it is the more damaging: it
means a model trained on last month's rows knows things that were only knowable
this month.

### 6.3 The distributed-holdout arithmetic

{{ch:ml-metrics}} gave the optimism of a maximum over $k$ noisy scores as
approximately $\sigma_v\sqrt{2\log k}$. The organisational version replaces $k$
with the *total* number of evaluations the test set has ever served, across
everyone.

$$
\E[\text{optimism}] \approx \sigma_v \sqrt{2\log K_{\text{total}}},
\qquad
K_{\text{total}} = \sum_{\text{people}} \sum_{\text{months}} k
$$ (eq:distributed-optimism)

The growth is slow — $\sqrt{\log K}$ — which is the only reason this is
survivable. Going from 10 evaluations to 1,000 multiplies the optimism by 1.8,
not by 100. But $K_{\text{total}}$ in a year-old project is routinely in the
hundreds, and $\sqrt{2\log 500} \approx 3.5$: with a validation standard error
of 1%, that is a 3.5-point illusion, which is larger than most claimed
improvements.

The practical consequence is not "never look". It is that $K_{\text{total}}$
must be *known*, which requires it to be logged, which requires evaluation to
go through a function rather than a notebook cell.

## 7. Implementation

```python {tier=A name=splitters-from-scratch}
"""Group, time and shifted splitters from scratch, and what each leak costs.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the splitters ----------------------------------------------------------
def random_split(n, frac=0.75, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(frac * n)
    return idx[:cut], idx[cut:]


def group_split(groups, frac=0.75, seed=0):
    """Split on the GROUP level, so no group appears on both sides."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    keep = set(perm[:int(frac * len(uniq))].tolist())
    mask = np.array([g in keep for g in groups])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def time_split(times, frac=0.75, embargo=0.0):
    """Everything before the cut trains; everything after `embargo` validates.

    The embargo is the gap of section 6.2 — rows inside it are DISCARDED, not
    assigned to either side, because they are contaminated in both directions.
    """
    order = np.argsort(times, kind="mergesort")
    cut_t = np.quantile(times, frac)
    tr = order[times[order] <= cut_t]
    va = order[times[order] > cut_t + embargo]
    return tr, va


def group_kfold(groups, k=5, seed=0):
    """K folds in which each group appears in exactly one validation fold."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    buckets = np.array_split(perm, k)
    out = []
    for b in buckets:
        held = set(b.tolist())
        m = np.array([g in held for g in groups])
        out.append((np.flatnonzero(~m), np.flatnonzero(m)))
    return out


def expanding_window(times, n_splits=5, embargo=0.0):
    """Train on everything up to t, validate on the next block, advance."""
    order = np.argsort(times, kind="mergesort")
    ts = times[order]
    edges = np.quantile(ts, np.linspace(0.4, 1.0, n_splits + 1))
    out = []
    for i in range(n_splits):
        tr = order[ts <= edges[i]]
        va = order[(ts > edges[i] + embargo) & (ts <= edges[i + 1])]
        if len(tr) and len(va):
            out.append((tr, va))
    return out


# --- data with a genuine group effect (eq. 43.1) ----------------------------
def make_grouped(n_groups=150, rows_per_group=20, icc=0.4, seed=1):
    """y = f(x) + u_g + eps, with the group-effect variance set to hit a
    target intraclass correlation (eq. 43.4).

    Crucially the FEATURES are group-correlated too: each customer has a
    characteristic profile and their rows scatter around it. That is what
    real grouped data looks like, and it is what lets a flexible model
    recognise 'this row belongs to a customer I have seen' and recall that
    customer's effect. Without it there is nothing for the model to key on
    and no leak to measure.
    """
    rs = np.random.default_rng(seed)
    sig_e = 1.0
    sig_u = np.sqrt(icc / (1 - icc)) * sig_e if icc < 1 else 10.0
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(groups)
    centre = rs.normal(0, 1.0, (n_groups, 6))[groups]     # customer profile
    X = centre + rs.normal(0, 0.25, (n, 6))               # tight scatter
    u = rs.normal(0, sig_u, n_groups)[groups]
    f = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3]
    y = f + u + rs.normal(0, sig_e, n)
    return X, y, groups, f


def fit_knn(Xtr, ytr, Xte, k=5):
    """A flexible model — capacity is what makes a grouped leak visible
    (Chapter 28 measured the same thing)."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("=" * 72)
print("what a grouped leak is worth, as a function of the ICC (eq. 43.4)")
print("=" * 72)
print("y = f(x) + u_g + eps. A random split puts rows from the SAME group on")
print("both sides, so the model can estimate u_g and reuse it. Production")
print("never allows that, because in production the group is new.\n")
print(f"{'ICC':>6} {'random-split RMSE':>19} {'grouped-split RMSE':>20} "
      f"{'optimism':>10} {'upper bound':>12}")
for icc in (0.0, 0.1, 0.3, 0.5, 0.7):
    X, y, g, f = make_grouped(icc=icc)
    tr, va = random_split(len(y), seed=3)
    r_rand = rmse(fit_knn(X[tr], y[tr], X[va]), y[va])
    tr, va = group_split(g, seed=3)
    r_grp = rmse(fit_knn(X[tr], y[tr], X[va]), y[va])
    # eq. 43.2 / 43.3: the ratio of error variances is ~ 1 + sig_u^2/sig_e^2
    predicted = np.sqrt(1 / (1 - icc)) if icc < 1 else np.inf
    print(f"{icc:>6.1f} {r_rand:>19.4f} {r_grp:>20.4f} "
          f"{r_grp / r_rand:>10.3f}x {predicted:>10.3f}x")

print("\nThe optimism column is the factor by which the random split")
print("understates the error. At ICC = 0 the two splits agree, exactly as")
print("they should — there is no group effect to leak. As the ICC rises the")
print("gap opens, monotonically, and nothing in the output warns you it is")
print("happening.")
print("\nThe last column is the simple bound from eqs. 43.2-43.3, and the")
print("measured optimism EXCEEDS it at every ICC. That is worth")
print("understanding rather than waving at, because the reason makes the")
print("problem worse than the algebra suggests.")
print("\nEq. 43.3 assumes that on an unseen group the model contributes")
print("nothing — it falls back to zero and simply eats the group effect. A")
print("nearest-neighbour model does something worse: it borrows the effect")
print("of whichever group happens to be nearby in feature space. That is not")
print("a missing estimate, it is a WRONG one, and it adds roughly a second")
print("factor of the group-effect variance rather than one.")
print("\nSo eq. 43.3 is a floor on the damage, not a ceiling. Any model that")
print("generalises across groups by similarity — which is most of them —")
print("will do worse on a genuinely new group than the algebra predicts.")

# --- effective sample size (eq. 43.5) ---------------------------------------
print("\n" + "=" * 72)
print("the other half: your confidence intervals are too narrow (eq. 43.5)")
print("=" * 72)
N, m = 100_000, 200
print(f"{'ICC':>6} {'nominal N':>11} {'effective N':>13} "
      f"{'CI too narrow by':>18}")
for icc in (0.0, 0.05, 0.1, 0.3, 0.5):
    n_eff = N / (1 + (m - 1) * icc)
    print(f"{icc:>6.2f} {N:>11,} {n_eff:>13,.0f} "
          f"{np.sqrt(N / n_eff):>17.1f}x")
print(f"\n({N:,} rows from {N // m:,} groups of {m}.) At an ICC of 0.3 the")
print("effective sample size is under 1,700 and every interval computed from")
print("the nominal N is about eight times too narrow. This is Chapter 22's")
print("design effect, arriving as an engineering problem.")

# --- time splits and the embargo (section 6.2) ------------------------------
print("\n" + "=" * 72)
print("time splits: the embargo, and why max(w, d) (section 6.2)")
print("=" * 72)


def make_temporal(n=4000, feature_window=30, label_delay=45, seed=2):
    """A trailing-window feature and a label that resolves `label_delay`
    days later — so a row at time t encodes information from t + delay."""
    rs = np.random.default_rng(seed)
    t = np.sort(rs.uniform(0, 700, n))
    latent = np.sin(t / 60.0) + rs.normal(0, 0.3, n)
    # trailing-window mean: the feature at t depends on the past w days
    feat = np.array([latent[(t > ti - feature_window) & (t <= ti)].mean()
                     if ((t > ti - feature_window) & (t <= ti)).any()
                     else 0.0 for ti in t])
    # the label depends on the FUTURE `delay` days, which is what leaks
    lab = np.array([latent[(t > ti) & (t <= ti + label_delay)].mean()
                    if ((t > ti) & (t <= ti + label_delay)).any()
                    else 0.0 for ti in t])
    y = 2.0 * lab + rs.normal(0, 0.25, n)
    # time is a feature here, which is realistic (recency, tenure, day of
    # week) and is also what lets a model exploit temporal adjacency at all
    X = np.column_stack([feat, t / 100.0, rs.normal(size=(n, 2))])
    return X, y, t, lab


Xt, yt, tt, lab_t = make_temporal()
print("feature window w = 30 days, label delay d = 45 days, so the embargo")
print("should be max(w, d) = 45 days.")
print("\nThe VALIDATION WINDOW IS HELD FIXED throughout — only the training")
print("side changes, by dropping rows within `embargo` days of the")
print("validation start. Varying both at once would compare scores on")
print("different data and measure nothing.\n")

cut_t = float(np.quantile(tt, 0.7))
val_mask = (tt > cut_t) & (tt <= cut_t + 120)      # fixed validation window
va = np.flatnonzero(val_mask)
print(f"fixed validation window: {int(val_mask.sum())} rows in "
      f"({cut_t:.0f}, {cut_t + 120:.0f}] days")

print("\nRather than read the leak off a downstream score — where it")
print("competes with every other effect — measure the CONTAMINATION")
print("directly. A training row is contaminated if its own windows reach")
print("into the validation period: its feature window (t-w, t] or its label")
print("window (t, t+d].\n")
print(f"{'embargo':>9} {'train rows':>11} {'feature-window':>15} "
      f"{'label-window':>13} {'either':>8}")
print(f"{'(days)':>9} {'':>11} {'overlap':>15} {'overlap':>13} {'':>8}")
W, DELAY = 30, 45
for emb in (0, 15, 30, 45, 60):
    tr = np.flatnonzero(tt <= cut_t - emb)
    ts = tt[tr]
    feat_bad = float(np.mean(ts + W > cut_t))      # feature window crosses cut
    lab_bad = float(np.mean(ts + DELAY > cut_t))   # label resolves after cut
    either = float(np.mean((ts + W > cut_t) | (ts + DELAY > cut_t)))
    print(f"{emb:>9} {len(tr):>11} {feat_bad:>15.4f} {lab_bad:>13.4f} "
          f"{either:>8.4f}")

print("\nBoth columns reach exactly zero at an embargo of 45 days, and not")
print("before: the feature-window contamination clears at 30 and the")
print("label-window contamination needs 45. The binding constraint is")
print("max(w, d), which is eq. 43.6 confirmed by counting rather than")
print("argued.")
print("\nNote which one binds. The feature window is the one people think")
print("of; the label window is longer here and is the one usually forgotten,")
print("because it is not visible anywhere in the feature engineering code.")

# ...and what the contamination is worth, using a model that can exploit it
print("\nAnd what the contamination buys a model that can use it — one with")
print("time as a feature, so temporally adjacent rows are its neighbours:\n")
print(f"{'embargo':>9} {'validation RMSE':>18} {'vs clean':>10}")
clean = None
scores = {}
for emb in (0, 15, 30, 45, 60):
    tr = np.flatnonzero(tt <= cut_t - emb)
    scores[emb] = rmse(fit_knn(Xt[tr], yt[tr], Xt[va]), yt[va])
for emb in (0, 15, 30, 45, 60):
    delta = scores[emb] - scores[45]
    print(f"{emb:>9} {scores[emb]:>18.4f} {delta:>+10.4f}")
print("\nThe embargoed score is the honest one. Whether the un-embargoed")
print("score looks better or worse here depends on a second effect pulling")
print("the other way — the dropped rows are also the most RECENT, and")
print("recency helps a time-indexed model. That confound is exactly why the")
print("contamination count above is the better evidence: it measures the")
print("mechanism, not a downstream number that several effects move at once.")
```

## 8. Practical Example

```python {tier=A name=split-audit}
"""A split that audits itself, and the decay it is there to catch.
"""
import numpy as np

rng = np.random.default_rng(11)


class SplitAudit(AssertionError):
    """Raised when a split violates an invariant it declared."""


def audited_split(train_idx, val_idx, *, n_total, groups=None, times=None,
                  min_embargo=0.0, min_val_frac=0.05, name="split"):
    """Check every invariant a split claims, and fail loudly if one breaks.

    The point is not that these checks are clever. It is that they run on
    every build, so a split that silently stops being honest becomes a red
    test rather than an improved validation score.
    """
    problems = []
    tr, va = np.asarray(train_idx), np.asarray(val_idx)

    if len(np.intersect1d(tr, va)):
        problems.append(f"{len(np.intersect1d(tr, va))} rows in BOTH sides")
    if len(va) < min_val_frac * n_total:
        problems.append(f"validation is {len(va) / n_total:.1%} of the data, "
                        f"below the declared floor of {min_val_frac:.0%}")
    if len(tr) == 0 or len(va) == 0:
        problems.append("one side is empty")

    if groups is not None:
        shared = np.intersect1d(np.unique(groups[tr]), np.unique(groups[va]))
        if len(shared):
            problems.append(
                f"{len(shared)} group(s) appear on both sides "
                f"(e.g. {shared[:3].tolist()}) — this is a grouped leak")
        # a group split also silently loses power; report it rather than fail
        n_tr_g, n_va_g = len(np.unique(groups[tr])), len(np.unique(groups[va]))
        if n_va_g < 5:
            problems.append(f"only {n_va_g} validation groups: the estimate "
                            f"has almost no resolution")

    if times is not None:
        if times[tr].max() > times[va].min() - min_embargo:
            overlap = times[tr].max() - (times[va].min() - min_embargo)
            problems.append(
                f"temporal overlap of {overlap:.1f} units: training data "
                f"reaches within the {min_embargo} embargo of validation")

    if problems:
        raise SplitAudit(f"[{name}] " + "; ".join(problems))
    return True


def group_split(groups, frac=0.75, seed=0):
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(uniq)
    keep = set(perm[:int(frac * len(uniq))].tolist())
    m = np.array([g in keep for g in groups])
    return np.flatnonzero(m), np.flatnonzero(~m)


def random_split(n, frac=0.75, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return idx[:int(frac * n)], idx[int(frac * n):]


# --- the decay this is built to catch ---------------------------------------
print("=" * 72)
print("the failure mode: a split that WAS honest and quietly stopped being")
print("=" * 72)
print("Version 1 of the pipeline emits one row per customer, so a random")
print("split is correct and the audit passes.\n")

n_cust = 800
cust_v1 = np.arange(n_cust)                     # one row each
tr, va = random_split(len(cust_v1), seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v1), groups=cust_v1,
                  name="v1 random split")
    print("  v1: audit PASSED — no customer on both sides")
except AssertionError as e:
    print(f"  v1: audit FAILED — {e}")

print("\nSix months later an upstream team starts emitting one row per")
print("SESSION. Nothing in the modelling code changed. Nothing errors.\n")
cust_v2 = np.repeat(np.arange(n_cust), 4)       # four rows each now
tr, va = random_split(len(cust_v2), seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v2), groups=cust_v2,
                  name="v2 random split")
    print("  v2: audit PASSED")
except AssertionError as e:
    print(f"  v2: audit FAILED\n       {e}")

print("\nThe audit is what turns a silent optimism into a build failure. The")
print("fix is to switch to a grouped split, which the audit then accepts:\n")
tr, va = group_split(cust_v2, seed=1)
try:
    audited_split(tr, va, n_total=len(cust_v2), groups=cust_v2,
                  name="v2 grouped split")
    print("  v2 grouped: audit PASSED")
except AssertionError as e:
    print(f"  v2 grouped: audit FAILED — {e}")

# --- and what the difference was worth --------------------------------------
def make_data(groups, icc=0.35, seed=4):
    rs = np.random.default_rng(seed)
    n = len(groups)
    sig_u = np.sqrt(icc / (1 - icc))
    u = rs.normal(0, sig_u, groups.max() + 1)[groups]
    X = rs.normal(size=(n, 5))
    y = 1.1 * X[:, 0] - 0.8 * X[:, 1] + u + rs.normal(0, 1.0, n)
    return np.column_stack([X, groups.astype(float)]), y


def fit_knn(Xtr, ytr, Xte, k=5):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return ytr[idx].mean(1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


X2, y2 = make_data(cust_v2)
tr_r, va_r = random_split(len(y2), seed=1)
tr_g, va_g = group_split(cust_v2, seed=1)
print(f"\n{'split':<26} {'reported RMSE':>15} {'what it means':<32}")
print(f"{'v2 random (leaking)':<26} "
      f"{rmse(fit_knn(X2[tr_r], y2[tr_r], X2[va_r]), y2[va_r]):>15.4f} "
      f"{'sessions from known customers':<32}")
print(f"{'v2 grouped (honest)':<26} "
      f"{rmse(fit_knn(X2[tr_g], y2[tr_g], X2[va_g]), y2[va_g]):>15.4f} "
      f"{'sessions from NEW customers':<32}")
print("\nBoth numbers are correct answers to different questions. Only one of")
print("them is the question production asks.")

# --- the holdout ledger -----------------------------------------------------
print("\n" + "=" * 72)
print("the holdout ledger: making K_total knowable (eq. 43.7)")
print("=" * 72)


class Holdout:
    """Evaluation goes through a function, and the function keeps a ledger.

    A team that cannot say how many times its test set has been evaluated
    does not have a test set — it has a second validation set.
    """

    def __init__(self, X, y, val_se=0.01):
        self._X, self._y, self.val_se = X, y, val_se
        self.log = []

    def evaluate(self, model_fn, who, why):
        score = float(np.mean((model_fn(self._X) - self._y) ** 2) ** 0.5)
        self.log.append({"who": who, "why": why, "score": score})
        return score

    def optimism(self):
        k = max(len(self.log), 1)
        return float(self.val_se * np.sqrt(2 * np.log(max(k, 2))))

    def report(self):
        k = len(self.log)
        best = min(self.log, key=lambda r: r["score"]) if k else None
        print(f"  evaluations to date : {k}")
        print(f"  distinct people     : {len({r['who'] for r in self.log})}")
        if best:
            print(f"  best score reported : {best['score']:.4f} "
                  f"({best['who']}, {best['why']})")
        print(f"  expected optimism   : {self.optimism():.4f} "
              f"= SE x sqrt(2 log K)")
        if best:
            print(f"  corrected estimate  : "
                  f"{best['score'] + self.optimism():.4f}  (worse is honest)")


X_ho, y_ho = make_data(np.repeat(np.arange(200), 4), seed=9)
hold = Holdout(X_ho, y_ho)

# four people, over some months, each trying a few things — none of them
# aware of the others' usage
Xf, yf = make_data(cust_v2, seed=4)
tr_f, _ = group_split(cust_v2, seed=1)
for who, n_tries in (("ana", 6), ("ben", 11), ("cara", 4), ("dev", 9)):
    for i in range(n_tries):
        k = 3 + i
        hold.evaluate(lambda Z, k=k: fit_knn(Xf[tr_f], yf[tr_f], Z, k=k),
                      who, f"knn k={k}")

hold.report()
print("\nNobody looked more than eleven times. Collectively the test set has")
print("served thirty evaluations, and the best of thirty noisy scores is")
print("optimistic by roughly 2.6 standard errors — which is larger than most")
print("of the improvements anyone was chasing.")
print("\nThe ledger does not stop the problem. It makes K_total a number you")
print("can put in eq. 43.7 instead of a number nobody knows.")
```

## 9. Common Mistakes

**Calling `train_test_split` on grouped data.** The measurement shows the
optimism growing as $\sqrt{1/(1-\rho_{\text{ICC}})}$, invisibly.

**Assuming a split stays honest.** It decays when the data changes shape;
audit it on every build.

**Splitting time-ordered data randomly.** You are validating on the past.

**Omitting the embargo.** A trailing feature window or a delayed label leaks
across an adjacent boundary; the required gap is $\max(w, d)$.

**Selecting anything outside the inner loop.** Feature choice, scaling and
thresholds are all part of $\phi^{*}$ in {{eq:nested-cv}}.

**Computing confidence intervals from the nominal $N$.** With grouped rows the
effective $N$ can be two orders of magnitude smaller.

**Stratifying a leaking split.** You get a precise estimate of the wrong
number.

**Reporting only the mean over time folds.** The last fold is the only one
trained on a realistic amount of data.

**Not knowing how many times the test set has been used.** Then it is a
validation set, and you have no test set.

## 10. Connection to Previous Chapters

{{ch:ds-leakage}} named the leakage mechanisms; this chapter makes the defence
mechanical, and {{eq:icc}} quantifies what the grouped case costs.
{{ch:ds-timeseries}} supplied walk-forward validation, to which
{{sec:6-mathematical-foundation}} adds the embargo derivation.
{{ch:ml-metrics}} supplied selection optimism, which {{eq:distributed-optimism}}
extends across a team, and nested cross-validation, which becomes
{{eq:nested-cv}} here. {{ch:math-inference}} supplied the design effect that
reappears as {{eq:effective-n}}.

Forward: {{ch:mle-hpo}} spends the budget that {{eq:distributed-optimism}}
prices. {{ch:mle-pipelines}} makes the point-in-time correctness that the
embargo argument implies a property of the join rather than of the splitter.
{{ch:mle-registry}} records which split produced a candidate's numbers, because
a score without its split definition is not evidence. {{ch:mle-drift}} watches
for the shift that would make any split stale.

## 11. Exercises

**Beginner**

1. Name the three structural questions that decide a split.
2. What does a random split assume, and give two ways it fails.
3. Why does stratification not fix a grouped leak?
4. What is an embargo, and when do you need one?
5. Why report the last time-fold separately?

**Intermediate**

6. Given $\rho_{\text{ICC}} = 0.4$, predict the ratio between a random-split
   and a grouped-split error, using {{eq:group-split-error}}.
7. Using {{eq:effective-n}}, compute the effective sample size for 50,000 rows
   from 100 groups at $\rho_{\text{ICC}} = 0.2$.
8. Derive the required embargo for a 14-day trailing feature and a 60-day
   label delay.
9. Why does a grouped leak flatter a flexible model more than a rigid one?
10. Your data is grouped by patient *and* time-ordered. Describe the split.
11. When is a leave-one-source-out estimate the right headline number?

**Advanced**

12. Derive {{eq:random-split-error}} and {{eq:group-split-error}} and state the
    assumptions.
13. Explain why {{eq:nested-cv}} is violated by a feature-selection step run
    before the loop, and bound the resulting bias.
14. Design a split for data that is grouped, time-ordered, and where the group
    population itself changes over time.
15. Explain why the gap between a random-split score and a grouped-split score
    is an estimate of source-specific performance, and what it is not.

**Implementation**

16. Extend `audited_split` to check schema conformance and class-balance drift
    between the two sides.
17. Implement purged and embargoed k-fold cross-validation and verify no
    contamination remains.
18. Implement a stratified group splitter that balances both class and group
    size, and measure how well it does at both.
19. Instrument the holdout ledger to persist across sessions and report
    $K_{\text{total}}$ per project.

**Reasoning**

20. A colleague's validation score drops four points when you switch to a
    grouped split. What do you conclude, and what do you report?
21. Your test set has served 400 evaluations over two years. What can still be
    salvaged from it?

## 12. Chapter Summary

A split is a simulation of deployment, so the design question is what will be
different about production data: different rows, different entities, a later
time, or a new population. Most real problems need more than one of those
respected at once.

A random split assumes exchangeable rows. Under a group effect, its optimism
grows as $\sqrt{1/(1-\rho_{\text{ICC}})}$ — measured here across a range of
intraclass correlations — and the more flexible the model, the larger the leak,
because memorising a group effect requires capacity.

The same intraclass correlation destroys confidence intervals through the
effective sample size: 100,000 rows from 500 groups at $\rho_{\text{ICC}}=0.3$
behave like fewer than 1,700, so intervals computed from the nominal count are
about eight times too narrow.

Time-based splits need an embargo of $\max(w, d)$, where $w$ is the trailing
feature window and $d$ the label delay. The label half is the one usually
missed and the more damaging, because a training row's label encodes the future.
The measurement shows the estimate stabilising exactly when the embargo reaches
that bound.

Nested cross-validation must place the *entire* selection procedure inside the
inner loop — feature choice, scaling, encoding and thresholds, not just
hyperparameters — and the commonest violation happens before the code is
written, when someone inspects the data and drops a column.

A split is code, and code decays. The measured example shows a random split
that is correct when rows are one-per-customer and becomes a leak the day an
upstream change makes them one-per-session, with nothing raising an error and
the validation score improving. An audit that asserts a split's declared
invariants turns that into a build failure.

Finally, the test set is a budget spent by looking, and the spending is
distributed. Four people looking a handful of times each produced thirty
evaluations and roughly 2.6 standard errors of optimism. A ledger does not
prevent this; it makes $K_{\text{total}}$ a number you can correct for instead
of one nobody knows.
