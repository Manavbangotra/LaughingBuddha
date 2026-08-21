---
id: mle-pipelines
number: 45
part: V
tier: focused
status: reviewed
requires: [mle-splits, ds-feature-eng, ds-leakage, py-pandas]
provides: [skew-taxonomy, point-in-time-correctness, availability-time,
           fit-transform-discipline, as-of-join, pipeline-contract,
           schema-validation]
citations: [sculley2015, breck2017, pedregosa2011]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Name the four causes of training/serving skew and say which a shared code
   path fixes.
2. Implement a point-in-time-correct as-of join and demonstrate the leak
   without one.
3. Explain the `fit`/`transform` discipline as the small-scale version of the
   same idea.
4. Write a pipeline contract that fails loudly on schema and range violations.
5. Explain what a feature store does and does not solve.
6. Recognise pipeline jungles and correction cascades in a real codebase.
7. Decide where a feature should be computed, and why.

## 2. Why This Matters

This is the chapter about the failure that costs the most and is the hardest to
attribute.

**A skewed pipeline produces an excellent validation score and a
disappointing production model, with no error anywhere.** Nobody can point at
the bug, because there is no bug — there are two code paths that disagree, or
one join that reached slightly into the future. Teams lose months to this, and
usually conclude the model "didn't generalise".

**The mechanism is almost always a join, not a model.** {{ch:ds-leakage}}
taught you to recognise target leakage in a feature. The version that survives
that lesson is subtler: every feature is legitimate, every column is defensible,
and the *timestamp* on one of them is wrong by a day.
{{sec:7-implementation}} builds that bug and measures it.

**{{cite:sculley2015}}'s vocabulary lives here.** Glue code, pipeline jungles
and correction cascades are all names for pipeline pathologies, and they are
worth being able to recognise by name, because the remedy differs for each.

## 3. Prerequisites

{{ch:ds-leakage}} for the leakage mechanisms, of which temporal leakage is the
one this chapter attacks mechanically. {{ch:mle-splits}} for the embargo
argument, which is the same reasoning applied to splits rather than joins.
{{ch:ds-feature-eng}} for the transformations. {{ch:py-pandas}} for joins and
alignment.

## 4. Intuitive Explanation

### 4.1 The same feature, two different numbers

Training reads from a warehouse: months of history, complete rows, no time
pressure. Serving reads from a live system: one row, right now, in under fifty
milliseconds.

Those are different environments, and the same conceptual feature — "customer's
average order value over the last 30 days" — can come out different in each.
When it does, the model is answering a question it was never trained on.

```text
   TRAINING                              SERVING
   ────────────────                      ────────────────
   warehouse SQL                         Python in the API
   full history available                only the cache
   computed in a batch job               computed per request
   as of the label date                  as of now
        │                                     │
        └──────────► should agree ◄───────────┘
                     often does not
```

### 4.2 The four causes, and what each needs

Skew is not one problem. It is four, and they need different remedies — which
is why "we adopted a feature store" is an incomplete answer.

**Code divergence.** The warehouse SQL and the serving Python implement the
same definition differently. One rounds, the other truncates; one treats a
missing value as zero, the other as the mean. *Remedy: a shared definition —
one implementation, called by both paths. This is the one a feature store
fixes.*

**Time travel.** The training join takes a feature value from after the moment
the prediction would have been made. *Remedy: an as-of join with correct
point-in-time semantics. A feature store can provide this and does not
automatically do so.*

**Freshness.** The feature is recomputed hourly in batch. Training saw values
that were on average thirty minutes old; serving reads values that may be
fifty-nine minutes old. *Remedy: match the staleness distribution, or make the
staleness itself a feature.*

**Availability.** The feature is computable in the warehouse and not within the
serving latency budget, so serving substitutes a default. *Remedy: discover
this before training on it — which requires knowing the serving budget while
choosing features.*

### 4.3 Point-in-time correctness

The single most important idea in this chapter.

You are building a training row for a decision that would have been made at
time $t$. Every feature in that row must be computable from information
available strictly before $t$.

```text
  timeline for one training row:

     feature updates          decision      label resolves
     ●───●─────●──────●          │              ●
                       ▲         │
                       └─ use THIS value        │
                          (last one before t)   │
                                 t              t+d

     using any ● to the right of t is time travel
```

A plain join on the entity key does not do this — it takes whatever value is in
the feature table now, which is the *latest* value, which is from the future.
The correct operation is an **as-of join**: for each row, take the most recent
feature value with a timestamp strictly less than the decision time.

The failure is quiet and the damage is large, because the future-leaking value
is often *highly* predictive. {{sec:7-implementation}} measures a case where a
one-day time-travel error inflates validation AUC substantially while
production performance is unchanged.

### 4.4 `fit` and `transform` are the same idea, in miniature

{{ch:ml-linear-regression}} insisted that scaling be fitted inside the fold.
That is point-in-time correctness at the smallest scale: the scaler's mean is a
*learned parameter*, and learning it from the validation set is using
information you would not have had.

The `fit`/`transform` split {{cite:pedregosa2011}} encodes this in the type
system of the API. Anything with a `fit` is a model, however humble — a scaler,
an imputer, a target encoder, a vocabulary — and everything a model learns must
be learned from training data only.

The practical rule that follows: **if a transformation has state, it belongs
inside the pipeline object, not in a cell above it.**

## 5. Formal Explanation

### 5.1 The as-of join

Given events $(e_i, t_i)$ needing features, and a feature log
$(e, \tau, v)$ of value $v$ for entity $e$ recorded at time $\tau$, the
point-in-time-correct value is

$$
v^{\text{PIT}}(e_i, t_i) =
  v\big(e_i,\; \max\{\tau : \tau < t_i,\; (e_i,\tau) \in \text{log}\}\big)
$$ (eq:as-of-join)

Three details decide whether an implementation is correct:

**Strict inequality.** $\tau < t_i$, not $\tau \le t_i$. A feature update
recorded at exactly the decision timestamp may or may not have been visible;
assume it was not.

**Availability time, not event time.** $\tau$ must be when the value became
*queryable*, not when the underlying event happened. A transaction at 09:00
that lands in the warehouse at 02:00 the next day has $\tau = $ 02:00. Using
event time is the most common way an as-of join is wrong while looking right.

**A staleness bound.** If the most recent value is six months old, using it may
be worse than using nothing. A correct join returns the value *and* its age, and
the pipeline decides.

### 5.2 The label window

Symmetrically, the label must be resolvable from information *after* $t$ and
must not itself be contaminated.

$$
y_i = g\big(\text{events in } (t_i,\; t_i + d]\big)
$$ (eq:label-window)

The interaction with {{ch:mle-splits}}'s embargo is exact: a training row at
$t_1$ carries information about $(t_1, t_1+d]$, so it must be excluded from any
fold validating on that period. The join and the split are enforcing the same
constraint from two directions, and both have to be right.

### 5.3 The pipeline contract

A pipeline that only transforms is a pipeline that fails silently. A contract
adds assertions at the boundaries, and {{cite:breck2017}}'s rubric makes them
auditable. The checks worth having, in order of how often they catch something:

{#tbl:pipeline-contract caption="Pipeline contract checks. The ordering reflects how often each one catches a real problem in practice, not how sophisticated it is."}

| Check | Catches |
|---|---|
| Schema: columns, dtypes, nullability | upstream renames, silent type coercion |
| Range: min, max, allowed categories | unit changes, sentinel values, new categories |
| Null rate against a reference | a join that started missing |
| Row count / duplication factor | a fan-out from a changed join key |
| Distribution against a reference | everything else, less precisely |
| Feature freshness | a stalled upstream job |

The first four are cheap, deterministic and catch the majority of real
incidents. The fifth is what {{ch:mle-drift}} elaborates. Note what is *not*
here: model accuracy. That requires labels, and labels are late — which is the
subject of {{ch:mle-drift}}.

### 5.4 Feature stores, honestly

A feature store provides three things: one definition per feature, an offline
store supporting as-of joins for training, and a low-latency online store for
serving.

Mapping that onto {{sec:4-intuitive-explanation}}'s four causes:

- **Code divergence** — solved, and this is the real contribution. One
  definition, two readers.
- **Time travel** — *enabled*, not solved. The store provides an as-of join;
  whether you call it correctly, with availability timestamps, is still yours
  to get right.
- **Freshness** — managed but not eliminated. The store makes the staleness
  explicit and consistent, which is most of the battle.
- **Availability** — surfaced early, because a feature must be materialised to
  the online store to be usable, so the constraint appears at definition time
  rather than at deployment.

{{maturity:MATURE}} as a pattern; adoption reached roughly 45% of teams by 2026
from 15–20% in 2020. The architectural direction is towards solving this at the
query layer — streaming SQL with incremental materialised views — rather than
by shipping feature artefacts between stores, and the durable content is the
taxonomy rather than any product.

> IMPORTANT: The cost is real. A feature store adds a system, a service
> dependency in the serving path, and a definition language to learn. For a
> batch-scored model with a nightly pipeline and one consumer, it is
> overhead. The question that decides it is not "how sophisticated are we" but
> **"how many independent consumers read these features, and does anything
> serve them in real time?"** At one consumer and batch scoring, a shared
> library function achieves the same thing.

### 5.5 Pipeline jungles and correction cascades

Two of {{cite:sculley2015}}'s named pathologies are specifically about
pipelines, and both are recognisable.

A **pipeline jungle** grows when features are added incrementally: a scrape
here, a join there, a special case for the one client whose data arrives in a
different format. Each addition is locally reasonable. The result is a system in
which no one can say what a feature depends on, errors surface far from their
cause, and the only safe change is no change. The remedy is unpleasant and
structural — redesign, not refactoring — which is why the useful advice is
preventative: keep the dependency graph explicit and reviewable while it is
still small.

A **correction cascade** starts when a model $A$ is nearly right for a new
problem, so someone learns a small correction on top of it: $A' = A +
\delta_1$. Then another: $A'' = A' + \delta_2$. Each layer is cheap and each
creates a dependency on the layer beneath, so improving $A$ now breaks
everything above it — CACE in its purest form. The remedy is to detect it early
and pay for the retrain, because the cost of unwinding grows with every layer.

## 6. Mathematical Foundation

### 6.1 Why time travel is so damaging

Let the correct feature be $x^{\text{PIT}}$, known at $t$, and let the leaked
feature be $x^{\text{leak}}$, taken from $t + \Delta$. Suppose the feature is a
running statistic that partially incorporates the label event:

$$
x^{\text{leak}} = x^{\text{PIT}} + \beta\,\Ind[\text{label event}] + \nu
$$ (eq:leaked-feature)

Then $x^{\text{leak}}$ contains a direct, noiseless channel to $y$. The model
finds it — that is what fitting does — and the validation score reflects a
capability that does not exist at serving time, where only $x^{\text{PIT}}$ is
available.

Three properties make this the worst kind of leak:

**The magnitude does not depend on $\Delta$ being large.** A one-day error is
sufficient if the label resolves within a day, and one day is exactly the size
of error a timezone mistake or a daily batch boundary produces.

**It is invisible to feature inspection.** $x^{\text{leak}}$ has a plausible
name, a plausible distribution, and a plausible correlation with the target. It
looks like a good feature because it *is* a good feature — for a question nobody
is asking.

**It survives cross-validation.** Every fold has the same leak, so folds agree
with each other, and agreement across folds is precisely the signal people use
to conclude an estimate is trustworthy.

### 6.2 The staleness distribution

Let $S$ be the age of a feature value at read time. Training rows built from a
batch job at a fixed cadence $c$ have $S$ distributed over $[0, c]$ depending on
where the label falls in the cycle; a live read has whatever distribution the
refresh schedule produces.

If the model's prediction depends on the feature through $f(x)$, and the feature
itself evolves at rate $\dot{x}$, the expected discrepancy from a staleness
mismatch is approximately

$$
\E\big[f(x_{t-S_{\text{serve}}}) - f(x_{t-S_{\text{train}}})\big]
 \approx f'(x)\,\dot{x}\;
   \big(\E[S_{\text{train}}] - \E[S_{\text{serve}}]\big)
$$ (eq:staleness-gap)

Two readings. The damage is proportional to how fast the feature moves, so
slow-moving features (customer tenure) tolerate staleness mismatch and
fast-moving ones (session click count) do not. And the fix does not require
making serving fresher — matching the *distributions* is sufficient, which is
often much cheaper, and is why deliberately staling the training features to
match production is a legitimate technique rather than a hack.

### 6.3 Why the fit/transform discipline is the same theorem

A stateful transformation learns parameters $\theta_T$ from data. Applying it to
a row means computing $T(x; \theta_T)$.

If $\theta_T$ was estimated from a set including row $i$, then $T(x_i;
\theta_T)$ depends on $x_i$ through two paths: directly, and through
$\theta_T$. The second path is information flow from the evaluation set into the
representation, which is the same violation as {{eq:as-of-join}} with the
inequality reversed.

The magnitude scales as $O(1/n)$ for a mean-based transform — each row
contributes $1/n$ of the estimate — so it is negligible at $n = 10^{6}$ and
material at $n = 50$. But two common transforms break that bound badly:

- **Target encoding** leaks $O(1)$ for a rare category, because a category
  appearing once has its encoding determined entirely by that row's own label.
  This is why {{ch:ds-feature-eng}} required out-of-fold encoding and
  {{ch:ml-boosting}} noted CatBoost's ordered statistics as a systematic fix.
- **Feature selection by target correlation** leaks through the *choice* of
  features, which is not averaged over anything.

## 7. Implementation

```python {tier=A name=as-of-join}
"""The time-travel bug, built deliberately, and the as-of join that fixes it.
"""
import numpy as np

rng = np.random.default_rng(0)

# --- a small event-sourced world --------------------------------------------
# Customers accumulate transactions. We predict whether a customer defaults
# in the 30 days after a decision date. The feature is their PEAK SPEND TO
# DATE — a running maximum, which is the kind of feature that makes time
# travel dangerous, because a spike caused by the label never washes out.
N_CUST, HORIZON = 600, 30.0


def simulate():
    """Return (feature_log, decisions) with explicit timestamps.

    feature_log rows are (customer, available_at, peak_spend), where
    peak_spend is the customer's RUNNING MAXIMUM spend to date. A running
    statistic is the realistic case and the dangerous one: once a value
    enters it, it never leaves, so a spike caused by the label event
    contaminates every later reading of the feature.

    `available_at` is when the value became QUERYABLE, which lags the event
    by a batch delay — that distinction is section 5.1's second detail and
    is where most as-of joins go wrong.
    """
    log, decisions = [], []
    for c in range(N_CUST):
        risk = rng.beta(2, 5)
        # riskier customers genuinely spend more, so the HONEST feature has
        # real signal — otherwise the comparison below measures nothing
        base = 100.0 + 900.0 * risk + 150.0 * rng.random()
        peak, t = base, 0.0
        while t < 300:
            t += rng.exponential(9.0)
            if t >= 300:
                break
            peak = max(peak, base * (1.0 + abs(rng.normal(0.0, 0.25))))
            log.append((c, t + 1.0, peak))           # 1 day to become visible
        t_dec = float(rng.uniform(120, 240))
        default = rng.random() < risk
        decisions.append((c, t_dec, int(default)))
        if default:
            # distress: a large spike shortly BEFORE the default event, which
            # is AFTER the decision date. It permanently raises the running
            # maximum, so any later read of the feature carries it.
            t_default = t_dec + rng.uniform(1, HORIZON)
            t_spike = t_default - rng.uniform(0.5, 4.0)
            if t_spike > 0:
                peak = max(peak, base * (3.0 + 2.0 * rng.random()))
                log.append((c, t_spike + 1.0, peak))
                # and it persists in every subsequent reading
                for tk in np.arange(t_spike + 5.0, 300.0, 9.0):
                    log.append((c, tk + 1.0, peak))
    log = np.array(sorted(log, key=lambda r: (r[0], r[1])), dtype=float)
    dec = np.array(decisions, dtype=float)
    return log, dec


log, dec = simulate()
print(f"{len(log):,} feature-log rows, {len(dec):,} decisions, "
      f"default rate {dec[:, 2].mean():.3f}")


# --- three joins: one correct, two subtly wrong -----------------------------
def join_latest(log, dec):
    """WRONG. Takes each customer's most recent value overall — i.e. the
    value as of NOW, which for a training row is the future. This is what a
    plain groupby-last produces, and it is the commonest form of the bug."""
    out = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        out[i] = rows[-1, 2] if len(rows) else np.nan
    return out


def join_as_of_event_time(log, dec, event_lag=1.0):
    """SUBTLY WRONG. Correct as-of semantics, but keyed on EVENT time rather
    than availability time — so it uses values that had not yet landed."""
    out = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        event_t = rows[:, 1] - event_lag              # undo the batch delay
        ok = rows[event_t < t]
        out[i] = ok[-1, 2] if len(ok) else np.nan
    return out


def join_as_of(log, dec):
    """CORRECT (eq. 45.1): most recent value whose AVAILABILITY time is
    strictly before the decision time."""
    out = np.empty(len(dec))
    age = np.empty(len(dec))
    for i, (c, t, _) in enumerate(dec):
        rows = log[log[:, 0] == c]
        ok = rows[rows[:, 1] < t]                     # strict inequality
        if len(ok):
            out[i] = ok[-1, 2]
            age[i] = t - ok[-1, 1]                    # section 5.1: return age
        else:
            out[i], age[i] = np.nan, np.inf
    return out, age


x_latest = join_latest(log, dec)
x_event = join_as_of_event_time(log, dec)
x_pit, age = join_as_of(log, dec)
y = dec[:, 2].astype(int)

print(f"\nfeature staleness under the correct join: "
      f"median {np.median(age[np.isfinite(age)]):.1f} days, "
      f"p95 {np.percentile(age[np.isfinite(age)], 95):.1f} days")


# --- what each join is worth ------------------------------------------------
def auc(y, s):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(y) - npos)))


print("\n" + "=" * 72)
print("what each join reports, and what it would actually deliver")
print("=" * 72)
print(f"{'join':<38} {'validation AUC':>15} {'verdict':<22}")
for name, x, verdict in (
        ("groupby-last (value as of NOW)", x_latest, "time travel"),
        ("as-of on EVENT time", x_event, "1-day time travel"),
        ("as-of on AVAILABILITY time", x_pit, "correct")):
    print(f"{name:<38} {auc(y, x):>15.4f} {verdict:<22}")

print("\nAll three joins run without error, produce a plausibly-distributed")
print("numeric feature, and would pass any schema check.")
print("\nThe first inflates AUC by about twenty points. It is the commonest")
print("form of the bug and, mercifully, the easiest to spot afterwards —")
print("a feature that good usually is too good.")
print("\nThe middle row is the dangerous one. Its as-of logic is CORRECT —")
print("most recent value strictly before the decision — and it is still")
print("wrong, because it keyed on the time the transaction HAPPENED rather")
print("than the time it became queryable. A one-day error, invisible in the")
print("code, and exactly the size a nightly batch boundary or a timezone")
print("mistake produces.")
print("\nIts inflation is about one point. That smallness is the lesson, not")
print("a reprieve: the size of a time-travel leak is set by how much")
print("label-carrying information falls inside the window you accidentally")
print("included. One day catches only the spikes that happen to land in it")
print("here. Shorten the label horizon from thirty days to one — a")
print("same-session conversion, a next-hour failure — and that same one-day")
print("error becomes the twenty-point row instead.")

# --- and it survives cross-validation (section 6.1) -------------------------
print("\n" + "=" * 72)
print("why cross-validation does not catch it (section 6.1)")
print("=" * 72)
folds = np.array_split(np.random.default_rng(1).permutation(len(y)), 5)
print(f"{'join':<38} " + " ".join(f"{'fold ' + str(i + 1):>8}"
                                  for i in range(5)) + f" {'spread':>8}")
for name, x in (("groupby-last", x_latest), ("as-of on availability", x_pit)):
    scores = [auc(y[f], x[f]) for f in folds]
    print(f"{name:<38} " + " ".join(f"{v:>8.3f}" for v in scores) +
          f" {max(scores) - min(scores):>8.3f}")

print("\nThe leaking join's folds do not merely agree — they agree MORE")
print("TIGHTLY than the correct join's, by a factor of about two on the")
print("spread. That is not a fluke: the leaked feature is a strong, clean")
print("signal, so every fold recovers it easily and they all land in the")
print("same place. The honest feature is weaker, so folds disagree more.")
print("\nSo the diagnostic people actually use — 'the folds agree, the")
print("estimate is stable, I trust it' — points the wrong way. Consistency")
print("across folds measures how reliably the signal is recoverable, not")
print("whether the signal should exist.")
print("\nThere is no split that detects a bad join. The join has to be right.")
```

```python {tier=A name=pipeline-contract}
"""A pipeline with state, a contract that guards it, and the fit/transform
discipline that keeps state out of the validation set.
"""
import numpy as np

rng = np.random.default_rng(4)


class ContractViolation(AssertionError):
    pass


class Contract:
    """Section 5.3's checks, learned from a reference sample and asserted on
    every batch thereafter."""

    def __init__(self, names, categorical=()):
        self.names, self.categorical = list(names), set(categorical)
        self.ref = None

    def fit(self, X):
        self.ref = {
            "n_cols": X.shape[1],
            "min": X.min(0), "max": X.max(0),
            "null_rate": np.isnan(X).mean(0),
            "categories": {j: set(np.unique(X[:, j][~np.isnan(X[:, j])]))
                           for j in self.categorical},
            "mean": np.nanmean(X, axis=0), "sd": np.nanstd(X, axis=0),
        }
        return self

    def check(self, X, *, tol_range=0.25, tol_null=0.05, label="batch"):
        r, problems = self.ref, []
        if X.shape[1] != r["n_cols"]:
            raise ContractViolation(
                f"[{label}] column count {X.shape[1]} != {r['n_cols']}")
        for j, nm in enumerate(self.names):
            span = max(r["max"][j] - r["min"][j], 1e-9)
            lo, hi = np.nanmin(X[:, j]), np.nanmax(X[:, j])
            if lo < r["min"][j] - tol_range * span:
                problems.append(f"{nm}: min {lo:.4g} below reference "
                                f"{r['min'][j]:.4g}")
            if hi > r["max"][j] + tol_range * span:
                problems.append(f"{nm}: max {hi:.4g} above reference "
                                f"{r['max'][j]:.4g}")
            nr = float(np.isnan(X[:, j]).mean())
            if nr > r["null_rate"][j] + tol_null:
                problems.append(f"{nm}: null rate {nr:.3f} vs reference "
                                f"{r['null_rate'][j]:.3f}")
            if j in self.categorical:
                new = set(np.unique(X[:, j][~np.isnan(X[:, j])])) \
                    - r["categories"][j]
                if new:
                    problems.append(f"{nm}: unseen categories "
                                    f"{sorted(new)[:4]}")
        if problems:
            raise ContractViolation(f"[{label}] " + "; ".join(problems))
        return True


# --- the reference batch ----------------------------------------------------
NAMES = ["amount_gbp", "tenure_months", "n_txn_30d", "region_code"]


def make_batch(n, seed, *, amount_scale=1.0, region_max=5, null_extra=0.0):
    rs = np.random.default_rng(seed)
    amount = rs.lognormal(4.0, 0.7, n) * amount_scale
    tenure = rs.uniform(0, 120, n)
    n_txn = rs.poisson(6, n).astype(float)
    region = rs.integers(0, region_max, n).astype(float)
    X = np.column_stack([amount, tenure, n_txn, region])
    if null_extra:
        m = rs.random(n) < null_extra
        X[m, 0] = np.nan
    return X


ref = make_batch(4000, 0)
contract = Contract(NAMES, categorical=[3]).fit(ref)

print("=" * 72)
print("the contract, against batches that each break one thing")
print("=" * 72)
cases = [
    ("same distribution (control)", make_batch(2000, 1)),
    ("amounts switched to pence", make_batch(2000, 2, amount_scale=100.0)),
    ("a new region code appears", make_batch(2000, 3, region_max=7)),
    ("upstream join started missing", make_batch(2000, 4, null_extra=0.22)),
    ("a column was dropped", make_batch(2000, 5)[:, :3]),
]
for label, batch in cases:
    try:
        contract.check(batch, label=label)
        print(f"  PASS  {label}")
    except ContractViolation as e:
        msg = str(e).split("] ", 1)[1]
        print(f"  FAIL  {label}\n          {msg[:96]}")

print("\nEach of these is a real incident shape, each produces numbers a")
print("model will happily consume, and none of them raises an exception")
print("anywhere else in the pipeline. The contract is the only thing between")
print("them and a silently degraded prediction.")

# --- fit/transform: state must not cross the boundary -----------------------
print("\n" + "=" * 72)
print("fit/transform: how much does leaking a scaler actually cost?")
print("=" * 72)
print("Section 6.3 says a mean-based transform leaks O(1/n), so this should")
print("be negligible at large n and material at small n. Measured:\n")


def target_encode(cat_tr, y_tr, cat_all, smoothing=0.0):
    """A stateful transform with a much worse leak than a scaler."""
    prior = float(y_tr.mean())
    out = np.full(len(cat_all), prior)
    for c in np.unique(cat_tr):
        m = cat_tr == c
        k = m.sum()
        enc = (y_tr[m].sum() + smoothing * prior) / (k + smoothing)
        out[cat_all == c] = enc
    return out


def knn_auc(Xtr, ytr, Xte, yte, k=9):
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d, k, axis=1)[:, :k]
    s = ytr[idx].mean(1)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(yte.sum())
    if npos == 0 or npos == len(yte):
        return float("nan")
    return float((r[yte == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(yte) - npos)))


print(f"{'n':>7} {'scaler fitted in-fold':>23} {'scaler fitted on all':>22} "
      f"{'gap':>9}")
for n in (40, 100, 400, 2000, 10000):
    gaps = []
    for rep in range(30):
        rs = np.random.default_rng(1000 + rep)
        X = rs.normal(size=(2 * n, 5))
        yv = (rs.random(2 * n)
              < 1 / (1 + np.exp(-(X[:, 0] - 0.7 * X[:, 1])))).astype(int)
        Xtr, ytr, Xte, yte = X[:n], yv[:n], X[n:], yv[n:]
        mu_i, sd_i = Xtr.mean(0), Xtr.std(0) + 1e-9
        a = knn_auc((Xtr - mu_i) / sd_i, ytr, (Xte - mu_i) / sd_i, yte)
        mu_a, sd_a = X.mean(0), X.std(0) + 1e-9    # the mistake
        b = knn_auc((Xtr - mu_a) / sd_a, ytr, (Xte - mu_a) / sd_a, yte)
        gaps.append((a, b))
    a = float(np.mean([g[0] for g in gaps]))
    b = float(np.mean([g[1] for g in gaps]))
    print(f"{n:>7} {a:>23.4f} {b:>22.4f} {b - a:>+9.4f}")

print("\nThe scaler leak is small at every size and vanishes as n grows,")
print("exactly as the O(1/n) argument predicts. That is worth knowing")
print("because it tells you where NOT to spend your attention.")

print("\nNow the same experiment with a target encoder, which section 6.3")
print("says leaks O(1) for rare categories rather than O(1/n):\n")
print(f"{'categories':>11} {'rows/category':>14} {'out-of-fold enc':>17} "
      f"{'fitted on all':>15} {'gap':>9}")
for n_cat in (10, 50, 200, 800):
    n = 1600
    outs, alls = [], []
    for rep in range(20):
        rs = np.random.default_rng(2000 + rep)
        cat = rs.integers(0, n_cat, 2 * n)
        eff = rs.normal(0, 1.0, n_cat)[cat]
        yv = (rs.random(2 * n) < 1 / (1 + np.exp(-eff))).astype(int)
        ctr, cte = cat[:n], cat[n:]
        ytr, yte = yv[:n], yv[n:]
        # honest: encoding learned from training rows only
        enc_tr = target_encode(ctr, ytr, ctr)
        enc_te = target_encode(ctr, ytr, cte)
        outs.append(knn_auc(enc_tr[:, None], ytr, enc_te[:, None], yte))
        # the mistake: encoding learned from everything
        enc_all = target_encode(cat, yv, cat)
        alls.append(knn_auc(enc_all[:n, None], ytr, enc_all[n:, None], yte))
    print(f"{n_cat:>11} {2 * n / n_cat:>14.0f} {np.mean(outs):>17.4f} "
          f"{np.mean(alls):>15.4f} {np.mean(alls) - np.mean(outs):>+9.4f}")

print("\nWith 800 categories over 3,200 rows — four rows each — the leaked")
print("encoder reports a substantially better model than the honest one,")
print("because each category's encoding is largely determined by the labels")
print("of the very rows being encoded.")
print("\nThat is the practical rule behind fit/transform: the danger is not")
print("proportional to how complicated the transform is, it is proportional")
print("to how much of any single row's own label ends up in that row's")
print("features. A scaler averages over everything and is safe; a target")
print("encoder on rare categories barely averages at all.")
```

## 8. Practical Example

```python {tier=A name=training-serving-parity}
"""One definition, two paths, and a parity test that proves they agree.
"""
import numpy as np

rng = np.random.default_rng(21)

# --- the shared definition: ONE implementation, called by both paths --------
FEATURE_SPEC = {
    "avg_order_30d": {"window_days": 30, "agg": "mean", "default": 0.0},
    "n_orders_30d":  {"window_days": 30, "agg": "count", "default": 0.0},
    "max_order_90d": {"window_days": 90, "agg": "max", "default": 0.0},
}


def compute_features(events, as_of, spec=FEATURE_SPEC):
    """The single source of truth.

    `events` is an (n, 2) array of (available_at, amount) for ONE entity,
    already filtered to that entity. `as_of` is the decision time.

    Both the training job and the serving path call this. That is what
    removes code divergence (section 4.2) — and note that it removes ONLY
    code divergence. The other three causes are still live.
    """
    out = {}
    for name, s in spec.items():
        lo = as_of - s["window_days"]
        m = (events[:, 0] < as_of) & (events[:, 0] >= lo)   # strict, eq. 45.1
        vals = events[m, 1]
        if len(vals) == 0:
            out[name] = s["default"]
        elif s["agg"] == "mean":
            out[name] = float(vals.mean())
        elif s["agg"] == "count":
            out[name] = float(len(vals))
        else:
            out[name] = float(vals.max())
    return out


# --- a world of entities and events -----------------------------------------
def make_world(n_entities=300, seed=0):
    rs = np.random.default_rng(seed)
    events = {}
    for e in range(n_entities):
        n_ev = rs.poisson(28) + 3
        t = np.sort(rs.uniform(0, 365, n_ev))
        amt = rs.lognormal(3.6, 0.6, n_ev)
        events[e] = np.column_stack([t, amt])
    return events


world = make_world()


# --- the TRAINING path: batch, over history ---------------------------------
def build_training_rows(world, decisions):
    rows = []
    for e, t in decisions:
        rows.append(compute_features(world[e], t))
    return rows


# --- the SERVING path: one entity, now, from a cache ------------------------
class OnlineStore:
    """A cache refreshed on a schedule. Its staleness is the point."""

    def __init__(self, world, refresh_every=1.0, seed=0):
        self.world, self.refresh_every = world, refresh_every
        self.rs = np.random.default_rng(seed)

    def get_events(self, entity, now):
        """Return the events visible to serving, which is everything up to
        the last refresh — NOT up to `now`."""
        last_refresh = now - self.rs.uniform(0, self.refresh_every)
        ev = self.world[entity]
        return ev[ev[:, 0] < last_refresh], now - last_refresh


def serve(store, entity, now):
    ev, staleness = store.get_events(entity, now)
    feats = compute_features(ev, now)
    return feats, staleness


# --- the parity test --------------------------------------------------------
print("=" * 72)
print("parity test: do the two paths agree on the same inputs?")
print("=" * 72)
print("This is the test that should run in CI. Take real decision points,")
print("compute features both ways, and require agreement.\n")

decisions = [(int(e), float(t))
             for e, t in zip(rng.integers(0, 300, 400),
                             rng.uniform(120, 360, 400))]

# First: identical inputs, identical code. Parity must be EXACT.
store_fresh = OnlineStore(world, refresh_every=0.0, seed=1)
mismatches, max_rel = 0, 0.0
for e, t in decisions:
    train_f = compute_features(world[e], t)
    serve_f, _ = serve(store_fresh, e, t)
    for k in FEATURE_SPEC:
        denom = max(abs(train_f[k]), 1e-9)
        rel = abs(train_f[k] - serve_f[k]) / denom
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            mismatches += 1
print(f"  shared definition, zero staleness:")
print(f"    feature values compared : {len(decisions) * len(FEATURE_SPEC):,}")
print(f"    mismatches              : {mismatches}")
print(f"    max relative difference : {max_rel:.2e}")
print("  -> code divergence is eliminated by construction, because there is")
print("     only one implementation. This is what a feature store buys.")

# --- now the three causes a shared definition does NOT fix ------------------
print("\n" + "=" * 72)
print("the three causes a shared definition does NOT fix (section 4.2)")
print("=" * 72)

# 1. FRESHNESS
print("\n1. freshness — the online store is refreshed on a schedule")
print(f"{'refresh interval':>18} {'mean staleness':>16} "
      f"{'mean |rel. error|':>19} {'rows differing':>16}")
for label_h, interval in (("0 h", 0.0), ("1 h", 1 / 24), ("6 h", 6 / 24),
                          ("24 h", 1.0)):
    errs, diff, stales = [], 0, []
    st = OnlineStore(world, refresh_every=interval, seed=2)
    for e, t in decisions:
        train_f = compute_features(world[e], t)
        serve_f, stale = serve(st, e, t)
        stales.append(stale)
        for k in FEATURE_SPEC:
            denom = max(abs(train_f[k]), 1e-9)
            r = abs(train_f[k] - serve_f[k]) / denom
            errs.append(r)
            diff += r > 1e-9
    print(f"{label_h:>18} {np.mean(stales) * 24:>13.2f} h "
          f"{np.mean(errs):>19.4f} "
          f"{diff / (len(decisions) * len(FEATURE_SPEC)):>15.1%}")

print("\n   Same code, same definition, growing disagreement. Eq. 45.4 says")
print("   the damage is proportional to how fast the feature moves, which")
print("   is why a 30-day mean tolerates a day of staleness better than a")
print("   count does.")

# per-feature, to make that concrete
print("\n   per-feature, at a 24-hour refresh:")
st = OnlineStore(world, refresh_every=1.0, seed=3)
per = {k: [] for k in FEATURE_SPEC}
for e, t in decisions:
    train_f = compute_features(world[e], t)
    serve_f, _ = serve(st, e, t)
    for k in FEATURE_SPEC:
        per[k].append(abs(train_f[k] - serve_f[k]) / max(abs(train_f[k]), 1e-9))
for k, v in per.items():
    print(f"     {k:<16} mean relative error {np.mean(v):>8.4f}")

# 2. TIME TRAVEL — a shared definition called with the wrong timestamp
print("\n2. time travel — the SAME function, called with as_of = now + 1 day")
bad = [compute_features(world[e], t + 1.0) for e, t in decisions[:200]]
good = [compute_features(world[e], t) for e, t in decisions[:200]]
diff = np.mean([b["n_orders_30d"] != g["n_orders_30d"]
                for b, g in zip(bad, good)])
print(f"   rows whose 30-day order count changed: {diff:.1%}")
print("   The definition is shared and correct. The CALLER passed a")
print("   timestamp one day late, and about a sixth of the rows now contain")
print("   information from the future. No shared implementation prevents")
print("   this; only a correct as-of join does.")

# 3. AVAILABILITY
print("\n3. availability — a feature the warehouse can compute and serving")
print("   cannot within its latency budget")
LATENCY_BUDGET_MS = 50.0
COST_MS = {"avg_order_30d": 3.0, "n_orders_30d": 2.0, "max_order_90d": 9.0,
           "pct_rank_vs_cohort_365d": 140.0}
print(f"\n{'feature':<28} {'serving cost':>13} {'within budget?':>16}")
for k, c in COST_MS.items():
    print(f"{k:<28} {c:>10.0f} ms {'yes' if c < LATENCY_BUDGET_MS else 'NO':>16}")
print(f"\n   The last feature is perfectly computable offline and is 2.8x the")
print(f"   entire {LATENCY_BUDGET_MS:.0f}ms budget on its own. Discovering that")
print("   AFTER training on it means either dropping the feature and")
print("   retraining, or serving a default the model has never seen. The")
print("   check belongs at feature-definition time.")

# --- the summary table ------------------------------------------------------
print("\n" + "=" * 72)
print("what fixes what")
print("=" * 72)
rows = [
    ("code divergence", "shared definition", "SOLVED — measured above"),
    ("time travel", "as-of join on availability time", "caller must be right"),
    ("freshness", "match the staleness distributions", "manage, not eliminate"),
    ("availability", "check the latency budget up front", "a design constraint"),
]
print(f"{'cause':<20} {'remedy':<36} {'status':<26}")
for c, r, st_ in rows:
    print(f"{c:<20} {r:<36} {st_:<26}")
print("\nA feature store is a good way to get the first row and a convenient")
print("way to get the second. It does not get you the third or fourth, and")
print("the decision to adopt one should turn on how many independent")
print("consumers read the features and whether anything serves them in real")
print("time — not on how sophisticated the team wishes to appear.")
```

## 9. Common Mistakes

**A plain join instead of an as-of join.** The measurement shows a
`groupby-last` reporting a substantially better model than exists.

**As-of on event time rather than availability time.** Correct logic, one-day
error, invisible in the code.

**Using `<=` instead of `<`.** A value stamped at the decision time may not
have been visible.

**Trusting cross-validation to catch a bad join.** Every fold has the same
leak, and their agreement is what convinces you.

**Fitting a scaler on all the data.** Small, but free to avoid.

**Fitting a target encoder on all the data.** Not small — the measurement shows
it inflating AUC substantially at four rows per category.

**Assuming a feature store fixes skew.** It fixes code divergence. Three causes
remain.

**Discovering serving latency after training.** Check the budget at
feature-definition time.

**Adding a correction on top of a model instead of retraining it.** Each layer
makes the one below it unimprovable.

**A pipeline with no contract.** Every incident shape in
{{sec:7-implementation}} produces valid numbers and no exception.

## 10. Connection to Previous Chapters

{{ch:ds-leakage}} named temporal leakage; this chapter shows it arriving through
a join rather than through a feature choice, and measures the one-day version.
{{ch:mle-splits}} derived the embargo, and {{eq:label-window}} is the same
constraint seen from the join side — both must be right, and neither implies
the other. {{ch:ds-feature-eng}} supplied target encoding, whose out-of-fold
requirement {{sec:6-mathematical-foundation}} now explains as an $O(1)$ rather
than $O(1/n)$ leak. {{ch:ml-linear-regression}} required scaling inside the
fold; {{sec:6-mathematical-foundation}} shows why that one is the mild case.
{{ch:ml-boosting}} noted CatBoost's ordered target statistics, which are the
systematic fix.

Forward: {{ch:mle-reproducibility}} versions the pipeline so a feature
definition can be tied to a model. {{ch:mle-registry}} records which feature
version an artefact expects, because a model and its pipeline are one deployable
unit. {{ch:mle-drift}} monitors the contract's checks continuously rather than
at build time. {{part:24}} covers the orchestration this chapter deliberately
omits.

## 11. Exercises

**Beginner**

1. Name the four causes of training/serving skew.
2. Which one does a shared feature definition eliminate?
3. What is an as-of join?
4. Why strict inequality rather than $\le$?
5. Why must `fit` see only training data?

**Intermediate**

6. Explain why availability time and event time differ, and give a case where
   the gap is a full day.
7. Using {{eq:staleness-gap}}, say which of "tenure in months" and "clicks in
   the last 5 minutes" tolerates a stale cache, and why.
8. Explain why cross-validation cannot detect a bad join.
9. Why does target encoding leak $O(1)$ where a scaler leaks $O(1/n)$?
10. Describe a correction cascade and why it becomes harder to unwind.
11. When is a feature store not worth adopting?

**Advanced**

12. Write {{eq:as-of-join}} precisely for a feature with both an event time and
    an availability time, and state what a correct implementation returns
    besides the value.
13. Derive {{eq:staleness-gap}} and state its assumptions.
14. Explain why {{eq:label-window}} and the embargo of {{ch:mle-splits}} are the
    same constraint, and construct a case where one holds and the other does
    not.
15. Design a parity test that would catch a freshness mismatch, given that
    both paths call the same function.

**Implementation**

16. Implement an as-of join that handles multiple feature tables with different
    availability lags, and returns per-feature staleness.
17. Extend the contract to detect a change in the duplication factor of a join
    key, and demonstrate it catching a fan-out.
18. Implement out-of-fold target encoding and verify it removes the measured
    leak.
19. Build a CI parity test that fails when training and serving disagree by
    more than a stated tolerance on a sample of real decision points.

**Reasoning**

20. Your validation AUC is 0.94 and production AUC is 0.71. Rank your
    hypotheses and say what you would check first.
21. A team proposes a feature store to fix skew. What do you ask them before
    agreeing?

## 12. Chapter Summary

Training/serving skew is four different problems, and they need different
remedies. Code divergence is fixed by a shared definition — the measured parity
test shows exact agreement once both paths call one implementation. Time travel
needs a correct as-of join. Freshness needs the staleness distributions
matched. Availability needs the serving budget known before the feature is
chosen.

Point-in-time correctness is the central idea: every feature in a training row
must be computable strictly before the decision time, using **availability**
timestamps rather than event timestamps. The measured example shows three joins
that all run cleanly and produce plausible features, of which two report a
materially better model than exists — and the subtler of the two has correct
as-of logic and is wrong only because it keyed on event time.

Cross-validation cannot save you. The measurement shows the leaking join's folds
agreeing with each other as tightly as the correct join's, because every fold
contains the same leak. Fold agreement is not evidence of correctness.

The `fit`/`transform` discipline is the same theorem in miniature, and the
measurements calibrate where to spend attention: a scaler fitted on all the data
leaks $O(1/n)$ and is negligible past a few hundred rows, while a target encoder
on rare categories leaks $O(1)$ and inflates the score substantially. The danger
is proportional to how much of a row's own label reaches that row's features,
not to how complex the transform is.

A pipeline without a contract fails silently. Every incident shape measured
here — a unit change, a new category, a join that started missing, a dropped
column — produces valid numbers and raises nothing.

A feature store solves code divergence, enables correct time travel without
guaranteeing it, manages freshness, and surfaces availability early. That is
worth a lot and it is not everything, and the adoption question is how many
independent consumers read the features and whether anything serves them in real
time.
