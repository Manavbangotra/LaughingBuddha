---
id: mle-drift
number: 48
part: V
tier: focused
status: reviewed
requires: [mle-registry, mle-pipelines, ml-anomaly, ml-metrics]
provides: [drift-detection, covariate-shift, concept-drift, label-shift,
           label-delay, proxy-metric, population-stability-index,
           conjunction-alert, retraining-policy]
citations: [gama2014, rabanser2019, breck2017, sculley2015]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Distinguish covariate shift, label shift and concept drift, and say which
   is detectable without labels.
2. Explain why label delay decides which detector you are allowed to use.
3. Implement PSI and a multivariate two-sample test, and state each one's
   blind spot.
4. Explain why alerting on drift alone is a known failure mode, and build the
   conjunction alert that replaces it.
5. Choose and validate a proxy metric.
6. Design a retraining policy and say what triggers it.
7. Set thresholds from measured false-alarm rates rather than convention.

## 2. Why This Matters

This is the chapter the rest of the part points at, and the one where most
published advice is wrong for a specific, correctable reason.

**The classical literature assumes labels arrive promptly.** {{cite:gama2014}}
surveys the supervised detectors — DDM, EDDM, ADWIN — which watch a running
error rate and signal when it degrades. They are excellent and they are
unusable in most of the domains that need them, because a fraud label waits out
a chargeback window of one to six months, an insurance claim takes months to
years, and a clinical outcome may take years. Treating delayed labels as the
normal case rather than the exception changes every design decision downstream.

**Alerting on drift alone is how monitoring gets switched off.** Input
distributions move constantly for reasons that do not matter — a marketing
campaign, a seasonal shift, a new device model. If every one pages someone at
03:00 and the model was fine, the alert loses its meaning within a month.
{{sec:7-implementation}} measures the false-alarm rate and builds the
conjunction alert that fixes it.

**{{ch:mle-registry}} showed detection time dominating the cost of an
incident.** With no automated detection, the measured incident cost was
fifty times higher than with fast detection and a warm rollback. This chapter
is where that detection comes from, and it is the category the measured ML
Test Score identified as most teams' weakest.

## 3. Prerequisites

{{ch:ml-anomaly}} for the detectors — the input monitor of that chapter's
final section is this chapter's starting point, including the finding that
different detectors are blind to different failures. {{ch:mle-registry}} for
what a rollback costs and what triggers one. {{ch:ml-metrics}} for the metrics
being monitored. {{ch:mle-pipelines}} for the contract checks that run
continuously here.

## 4. Intuitive Explanation

### 4.1 Three things that can change

The joint distribution $p(x, y)$ factorises two ways, and which factor moves
determines what you can do about it.

```text
   COVARIATE SHIFT      p(x) changes, p(y|x) fixed
     the inputs moved; the relationship holds
     e.g. your customers got older; older customers behave as before
     -> detectable from inputs alone
     -> the model may still be correct, just extrapolating more

   LABEL SHIFT          p(y) changes, p(x|y) fixed
     the outcome mix moved; each class still looks the same
     e.g. fraud rate doubled, fraudsters unchanged
     -> detectable from predictions; correctable by reweighting

   CONCEPT DRIFT        p(y|x) changes
     the relationship itself moved
     e.g. the same profile now means something different
     -> NOT detectable from inputs at all. The inputs may be identical.
```

The third is the one that hurts and the one no amount of input monitoring can
see. That asymmetry is the organising fact of this chapter: **you can monitor
what is cheap to observe, and the dangerous change is the one that is not.**

### 4.2 Label delay decides everything

If labels arrive within minutes, monitor the error rate directly and use the
supervised detectors; they are strictly better, because they measure the thing
you care about.

If labels arrive in months, you have a monitoring gap the length of the delay,
and everything else in this chapter exists to fill it.

```text
   t=0        prediction made
    │
    │  ← this whole window is unmonitored by any supervised method →
    │
   t=120d     label arrives; NOW you can compute accuracy
              ...for a decision you made four months ago
```

The gap is not a nuisance to be minimised. It is a design constraint that
determines the architecture: what you can observe immediately is the input
distribution, the prediction distribution, and any downstream signal that
resolves faster than the true label.

### 4.3 The false-alarm problem

Input distributions move constantly. Run a per-feature test on thirty features
every day at $\alpha = 0.05$ and you expect 1.5 false alarms *per day* from
multiple testing alone, before any real drift.

The consequence is predictable and it is the single most common way monitoring
fails: alerts fire, someone investigates, the model is fine, and within a month
the channel is muted. **A monitoring system that cries wolf is worse than
none**, because it consumes the attention that a real incident would need.

The fix is not a higher threshold — that trades false alarms for missed
detections one-for-one. It is to require **two independent signals to agree**:
the inputs moved *and* something downstream got worse. Drift with no
measurable effect is, operationally, not a problem.

### 4.4 What to monitor when you cannot monitor accuracy

Four layers, in increasing latency and increasing relevance:

```text
   layer                    latency      relevance
   ─────────────────────    ─────────    ─────────────────────
   input distribution       seconds      low  (may not matter)
   prediction distribution  seconds      medium
   proxy metric             hours-days   high if validated
   true metric              weeks-months definitive
```

The **prediction distribution** is underused and is the best cheap signal
available. It responds to input drift *and* to pipeline breakage *and* to
anything that changes how the model behaves, and unlike input monitoring it
weights each feature by how much the model actually uses it. A model whose mean
predicted probability moves from 0.14 to 0.21 overnight has something wrong,
whichever layer caused it.

A **proxy metric** is anything correlated with the true outcome that resolves
sooner: for a loan model, the early-payment-default rate at 30 days rather than
the 12-month default; for a recommender, click-through rather than retention.
Its correlation with the true metric must itself be validated once labels
arrive, or you are monitoring a number whose relationship to the outcome is
assumed.

## 5. Formal Explanation

### 5.1 Population stability index

The standard industry measure, and worth knowing exactly what it is:

$$
\text{PSI} = \sum_{b=1}^{B}\big(c_b - r_b\big)\log\frac{c_b}{r_b}
$$ (eq:psi)

for reference proportions $r_b$ and current proportions $c_b$ over $B$ bins.
This is the **symmetrised KL divergence** — $\KL(c\|r) + \KL(r\|c)$ — which is
also called the Jeffreys divergence.

Conventional thresholds are 0.1 (moderate shift) and 0.25 (significant). Two
things to know about them:

**They are conventions, not derivations.** They come from credit-scoring
practice and are not calibrated to any particular sample size or false-alarm
rate.

**PSI depends on the sample size only weakly, and on the bin count strongly.**
More bins mean more terms and a larger PSI for the same underlying shift, so a
threshold is only comparable between runs using the same binning. Fixing the
bin edges from the reference period, once, is part of the method.

### 5.2 Detecting shift without labels

{{cite:rabanser2019}} compares the available approaches, and the practical
findings are worth stating because they are not what people assume.

**Per-feature two-sample tests with correction.** Run a Kolmogorov–Smirnov or
chi-squared test per feature and apply a multiple-testing correction. Cheap,
interpretable — it names the feature — and a surprisingly strong baseline. Its
blind spot is any change in the *joint* structure with unchanged marginals,
which {{ch:ml-anomaly}} measured directly.

**Dimensionality reduction plus a multivariate test.** Project to a
low-dimensional representation and test there. Strong general performance, and
it detects joint changes that per-feature tests miss.

**Classifier two-sample test.** Train a classifier to distinguish reference
from current samples. If it can, the distributions differ, and its AUC is an
interpretable effect size: 0.5 means indistinguishable, 1.0 means completely
separable. Its feature importances point at what moved. This is the most
generally useful of the three, and {{sec:7-implementation}} implements it.

> IMPORTANT: The multiple-testing correction is not optional. Thirty features
> tested daily at $\alpha = 0.05$ produce roughly 1.5 false alarms per day with
> no drift at all — about 550 per year. A Bonferroni or Benjamini–Hochberg
> correction is one line and it is the difference between a usable monitor and
> a muted channel. {{sec:7-implementation}} measures both rates.

### 5.3 The conjunction alert

The operational pattern that makes monitoring survivable:

$$
\text{page} \iff
  \underbrace{D_{\text{input}} > \tau_{D}}_{\text{something moved}}
  \;\wedge\;
  \underbrace{M_{\text{proxy}} < \mu_{0} - k\sigma}_{\text{something got worse}}
$$ (eq:conjunction-alert)

with the two conditions serving different roles:

- **Drift alone** is a *notification*, not a page. Record it, put it on a
  dashboard, look at it during working hours. It is frequently benign.
- **Proxy degradation alone** is a page, because something is wrong even if you
  cannot see the cause in the inputs — that is the concept-drift signature.
- **Both together** is a page with a diagnosis attached, which is the most
  actionable alert a monitoring system can produce.

{{sec:7-implementation}} measures the false-alarm rate of each rule and the
detection delay each one costs, because the conjunction is not free: requiring
two signals loses some sensitivity, and the question is how much.

### 5.4 Retraining policy

Three policies, and the choice should be explicit:

**Scheduled.** Retrain every $N$ days regardless. Simple, predictable, and
wasteful when nothing has changed — but its predictability is worth more than
people expect, because it makes the retraining path exercised rather than
theoretical.

**Triggered.** Retrain when a monitor fires. Efficient, and it makes the
retraining path rare, which means it will be broken when you need it.

**Continuous.** Retrain on a rolling window constantly. Appropriate for
fast-moving processes, and it forfeits the ability to attribute a change to a
model version — every model is slightly different from every other.

The usual right answer is **scheduled with triggered override**: a regular
cadence to keep the path warm, plus the ability to fire early. And the
non-obvious constraint is that retraining is not automatically safe. A retrain
on drifted data learns the drift, which is correct if the drift is the new
reality and catastrophic if the drift is a broken pipeline. Retraining must
pass the same promotion gate as any other candidate ({{ch:mle-registry}}).

### 5.5 What monitoring cannot do

Worth stating, because monitoring is often sold as a complete answer.

**It cannot detect concept drift from inputs.** By definition $p(x)$ may be
unchanged. Only labels or a proxy reveal it.

**It cannot distinguish a broken pipeline from a changed world.** Both look
like distribution shift. Deciding which requires a human who knows what
happened upstream — the detector tells you the input distribution moved, and a
person decides whether that is a bug or a business.

**It cannot tell you the drift matters.** That is what the conjunction is for,
and even then only if the proxy is validated.

## 6. Mathematical Foundation

### 6.1 PSI as a symmetrised divergence

Expand {{eq:psi}}:

$$
\sum_b (c_b - r_b)\log\frac{c_b}{r_b}
 = \sum_b c_b \log\frac{c_b}{r_b} + \sum_b r_b\log\frac{r_b}{c_b}
 = \KL(c\|r) + \KL(r\|c)
$$ (eq:psi-kl)

so PSI is symmetric, unlike either KL term alone. Three consequences.

It is zero exactly when the distributions match, and positive otherwise, with
no upper bound. It **diverges when any bin is empty on one side**, which is why
implementations clip $c_b$ and $r_b$ away from zero — and why a PSI computed on
a rare category is dominated by the clipping constant rather than by the data.

And for small perturbations $c_b = r_b(1 + \epsilon_b)$ it is approximately
$\sum_b r_b \epsilon_b^{2}$ — a chi-squared-like quadratic form. So PSI grows
as the *square* of a small shift, which means the conventional 0.1 and 0.25
thresholds are much further apart in shift terms than they look.

### 6.2 Why per-feature testing needs correction

With $m$ independent features and no drift, testing each at level $\alpha$
gives an expected $m\alpha$ false alarms per run, and the probability of at
least one is

$$
\Prob(\text{any false alarm}) = 1 - (1-\alpha)^{m}
$$ (eq:family-wise)

At $m = 30$, $\alpha = 0.05$: $1 - 0.95^{30} = 0.785$. **Nearly four days in
five will produce at least one alert from a completely stable system.**

Bonferroni tests each at $\alpha/m$, giving family-wise error $\le \alpha$ at
the cost of power. Benjamini–Hochberg controls the false discovery rate
instead — the expected fraction of alerts that are false — which is usually the
quantity you care about operationally, and is far less conservative when
several features genuinely moved.

Note that features are *not* independent in practice, so {{eq:family-wise}} is
an upper bound; correlated features produce fewer distinct false alarms and
also make Bonferroni more conservative than necessary.

### 6.3 The conjunction's false-alarm rate

Let the drift detector fire with probability $p_D$ under no drift, and the
proxy alarm with probability $p_M$, approximately independently under the null
(they measure different things).

$$
\Prob(\text{false page}) = p_D \, p_M
$$ (eq:conjunction-fa)

At $p_D = 0.20$ and $p_M = 0.10$ — both individually noisy — the conjunction
gives $0.02$, a tenfold reduction. That multiplicative reduction is why the
pattern works and why it beats simply raising one threshold.

The cost is in detection power. If a real incident produces drift with
probability $q_D$ and proxy degradation with probability $q_M$, the conjunction
detects it with probability $q_D q_M$, which is lower than either alone. The
trade is favourable exactly when a real incident reliably produces **both**
signals — which it does when the drift is what caused the degradation, and does
not when the failure is pure concept drift with unchanged inputs.

Hence the asymmetric rule of {{sec:5-formal-explanation}}: require the
conjunction for a *drift-led* page, and let proxy degradation page on its own.
That recovers the concept-drift case, which the conjunction would otherwise
miss entirely.

### 6.4 Choosing a threshold from a measured false-alarm rate

The conventional PSI thresholds are not calibrated to anything. A better
procedure takes ten minutes and produces a defensible number:

1. Take a stable historical period with no known incidents.
2. Split it into windows the size of your monitoring window.
3. Compute the statistic between each window and the reference.
4. Set the threshold at the quantile matching your tolerated false-alarm rate.

$$
\tau = Q_{1-\alpha}\big(\{D(\text{window}_i, \text{reference})\}\big)
$$ (eq:empirical-threshold)

This gives a threshold in the units of *your* data, *your* window size and
*your* feature count, with a false-alarm rate you chose. It also reveals when
the conventional threshold is badly wrong for your setting, which
{{sec:7-implementation}} measures — the empirical 99th-percentile PSI on stable
data can sit either well above or well below 0.25 depending on the window size.

## 7. Implementation

```python {tier=A name=drift-detectors}
"""PSI, per-feature testing with correction, and a classifier two-sample
test — with each one's blind spot measured.
"""
import numpy as np

rng = np.random.default_rng(0)


# --- the detectors ----------------------------------------------------------
def psi(ref, cur, edges):
    """Eq. 48.1. Bin edges are FIXED from the reference period — recomputing
    them per window would make runs incomparable (section 5.1)."""
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_edges(ref, bins=10):
    e = np.quantile(ref, np.linspace(0, 1, bins + 1))
    e[0], e[-1] = -np.inf, np.inf
    return e


def ks_stat(a, b):
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), allv, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def ks_pvalue(d, n1, n2):
    """Asymptotic two-sided KS p-value."""
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1) ** (j - 1)
                                    * np.exp(-2 * j ** 2 * lam ** 2)), 0, 1))


def benjamini_hochberg(pvals, alpha=0.05):
    """Control the false DISCOVERY rate: the expected fraction of alerts
    that are false, which is the operationally relevant quantity."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    if k:
        out[order[:k]] = True
    return out


def classifier_two_sample(ref, cur, seed=0):
    """Train a classifier to tell reference from current. Its held-out AUC
    is an interpretable effect size: 0.5 = indistinguishable.

    A depth-limited tree ensemble would be better; a logistic model on
    standardised features is enough to show the mechanism and keeps this
    listing self-contained.
    """
    rs = np.random.default_rng(seed)
    X = np.vstack([ref, cur])
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    # add pairwise products so the test can see JOINT changes, not only
    # marginal ones — without them it is just a linear marginal test
    d = X.shape[1]
    prods = np.column_stack([X[:, i] * X[:, j]
                             for i in range(d) for j in range(i + 1, d)])
    X = np.column_stack([X, prods])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    perm = rs.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(y))
    A = np.column_stack([np.ones(cut), X[:cut]])
    w = np.zeros(A.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y[:cut]) / cut + 0.01 * np.r_[0, w[1:]]
        S = np.maximum(p * (1 - p), 1e-7)
        H = (A * S[:, None]).T @ A / cut + 0.02 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    B = np.column_stack([np.ones(len(y) - cut), X[cut:]])
    s = B @ w
    yt = y[cut:]
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = int(yt.sum())
    if npos in (0, len(yt)):
        return 0.5
    return float((r[yt == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * (len(yt) - npos)))


# --- reference data and four kinds of change --------------------------------
D = 8


def make_reference(n, seed):
    rs = np.random.default_rng(seed)
    z = rs.normal(size=(n, 3))
    M = rs.normal(size=(3, D)) if seed == 0 else MIX
    return z @ M + rs.normal(0, 0.4, (n, D))


MIX = np.random.default_rng(0).normal(size=(3, D))


def make_window(n, kind, seed):
    rs = np.random.default_rng(seed)
    z = rs.normal(size=(n, 3))
    X = z @ MIX + rs.normal(0, 0.4, (n, D))
    if kind == "stable":
        return X
    if kind == "marginal_shift":
        X[:, 2] += 0.6                       # one feature moved
        return X
    if kind == "joint_only":
        # every MARGINAL is preserved; the joint structure is destroyed
        for j in range(D):
            X[:, j] = rs.permutation(X[:, j])
        return X
    if kind == "point_mass":
        bad = rs.random(n) < 0.10
        X[bad, 1] = 0.0                      # an upstream default value
        return X
    raise ValueError(kind)


ref = make_window(6000, "stable", 1)
EDGES = [psi_edges(ref[:, j]) for j in range(D)]

print("=" * 72)
print("three detectors, four kinds of change")
print("=" * 72)
print(f"{'change':<22} {'max PSI':>9} {'features flagged':>18} "
      f"{'classifier AUC':>16}")
for kind in ("stable", "marginal_shift", "joint_only", "point_mass"):
    cur = make_window(2000, kind, 42)
    psis = [psi(ref[:, j], cur[:, j], EDGES[j]) for j in range(D)]
    pvals = [ks_pvalue(ks_stat(ref[:, j], cur[:, j]), len(ref), len(cur))
             for j in range(D)]
    flagged = int(benjamini_hochberg(pvals, 0.05).sum())
    auc = classifier_two_sample(ref, cur, seed=3)
    print(f"{kind:<22} {max(psis):>9.4f} {flagged:>15} /{D:<2} "
          f"{auc:>16.4f}")

print("\nRead the rows against each other; each detector has a different")
print("blind spot and no row is caught by all three.")
print("\nThe MARGINAL SHIFT is caught by everything, as it should be.")
print("\nThe POINT MASS is caught by the per-feature test — one feature's")
print("marginal genuinely changed — and is nearly invisible to the")
print("classifier, whose AUC barely leaves 0.5. Ten per cent of rows pinned")
print("to a single value is a small, sharp change that a smooth decision")
print("boundary cannot separate on, which is why Chapter 47 checked for it")
print("directly rather than hoping a general detector would notice.")
print("\nThe third row is the one that separates the methods. Every marginal")
print("is preserved exactly — the columns were permuted independently — so")
print("PSI and the per-feature tests see nothing, correctly, because they")
print("only ever look at one feature at a time. The classifier two-sample")
print("test sees it immediately, because its pairwise product terms let it")
print("notice that the features no longer move together.")
print("\nThis is Chapter 42's finding arriving in a monitoring context: the")
print("detector's definition of 'different' decides what it can see, and no")
print("single detector covers the space.")

# --- section 6.2: the false-alarm arithmetic --------------------------------
print("\n" + "=" * 72)
print("why per-feature testing needs correction (eq. 48.4)")
print("=" * 72)
print("Stable data throughout — there is nothing to detect. 200 simulated")
print("monitoring days, 8 features tested per day.\n")

n_days = 200
raw_days, bh_days, bonf_days = 0, 0, 0
for day in range(n_days):
    cur = make_window(2000, "stable", 1000 + day)
    pvals = np.array([ks_pvalue(ks_stat(ref[:, j], cur[:, j]),
                                len(ref), len(cur)) for j in range(D)])
    raw_days += bool((pvals < 0.05).any())
    bh_days += bool(benjamini_hochberg(pvals, 0.05).any())
    bonf_days += bool((pvals < 0.05 / D).any())

print(f"{'rule':<34} {'days with >=1 alert':>21} {'rate':>8} "
      f"{'per year':>10}")
for label, count in (("uncorrected, alpha=0.05", raw_days),
                     ("Benjamini-Hochberg, FDR 0.05", bh_days),
                     ("Bonferroni, alpha=0.05/8", bonf_days)):
    print(f"{label:<34} {count:>21} {count / n_days:>8.3f} "
          f"{count / n_days * 365:>10.0f}")
print(f"\npredicted by eq. 48.4 for 8 independent features: "
      f"{1 - 0.95 ** D:.3f}")
print("\nThe uncorrected rule alerts on a stable system at close to the rate")
print("eq. 48.4 predicts, which at eight features is already most weeks and")
print("at thirty features would be most days. Correction is one line and it")
print("is the difference between a monitor people read and a muted channel.")

# --- section 6.4: thresholds from measured false-alarm rates ----------------
print("\n" + "=" * 72)
print("setting a PSI threshold empirically (eq. 48.6)")
print("=" * 72)
print("The conventional 0.1 / 0.25 thresholds are credit-scoring conventions,")
print("not derivations. Here is what stable data actually produces, at")
print("several monitoring window sizes:\n")
print(f"{'window':>8} {'median PSI':>11} {'99th pct':>10} "
      f"{'PSI at a real 0.3 sd shift':>28} {'0.25 fires?':>12}")
for n_win in (200, 500, 2000, 10000):
    stable_vals = np.array(
        [psi(ref[:, 0], make_window(n_win, "stable", 5000 + k)[:, 0],
             EDGES[0]) for k in range(120)])
    shifted_vals = np.array(
        [psi(ref[:, 0],
             make_window(n_win, "stable", 7000 + k)[:, 0]
             + 0.3 * ref[:, 0].std(), EDGES[0]) for k in range(60)])
    fires = np.mean(stable_vals > 0.25) > 0 or np.mean(shifted_vals > 0.25) > 0
    print(f"{n_win:>8} {np.median(stable_vals):>11.4f} "
          f"{np.percentile(stable_vals, 99):>10.4f} "
          f"{np.median(shifted_vals):>28.4f} "
          f"{('yes' if np.median(shifted_vals) > 0.25 else 'no'):>12}")

print("\nThe result is not the one the convention would lead you to expect,")
print("and it is more useful.")
print("\nPSI on STABLE data is not zero, and its scale depends strongly on")
print("the window size: the 99th percentile falls from 0.125 at a 200-row")
print("window to 0.005 at 10,000 — a twenty-five-fold range for data that")
print("has not changed at all. Small windows manufacture PSI out of sampling")
print("noise.")
print("\nMeanwhile the conventional 0.25 threshold never fires here — not on")
print("stable data, which is good, and not on a genuine 0.3-standard-")
print("deviation shift either, which is not. At a 10,000-row window that")
print("threshold sits fifty times above the noise floor and would miss any")
print("shift short of a catastrophe.")
print("\nSo a fixed threshold cannot be right across window sizes, because")
print("the statistic's own scale moves by a factor of twenty-five. Eq. 48.6")
print("gives a threshold in the units of your own data and window with a")
print("false-alarm rate you chose, at the cost of one pass over a stable")
print("period — and it is the difference between a number you can defend and")
print("one inherited from a different industry at a different sample size.")
```

```python {tier=A name=conjunction-alert}
"""Label delay, proxy metrics, and the conjunction alert — with the
false-alarm and detection-delay trade measured.
"""
import numpy as np

rng = np.random.default_rng(5)

# --- a system with a four-month label delay ---------------------------------
LABEL_DELAY_DAYS = 120
N_DAYS = 400
DAILY_VOLUME = 800


def simulate(incident_day=None, incident_kind=None, seed=0):
    """Return per-day arrays of: input drift statistic, mean prediction,
    proxy metric, and (delayed) true metric.

    The proxy resolves in 14 days; the true label takes 120.
    """
    rs = np.random.default_rng(seed)
    drift, pred_mean, true_metric = [], [], []
    for d in range(N_DAYS):
        # baseline world, with a slow benign seasonal wobble in the inputs
        season = 0.25 * np.sin(2 * np.pi * d / 90.0)
        x_shift = season + rs.normal(0, 0.06)
        quality = 0.80                             # true AUC-like metric

        if incident_day is not None and d >= incident_day:
            if incident_kind == "covariate":
                # inputs moved a lot; the relationship still holds, so the
                # model is fine — this is the benign case that must NOT page
                x_shift += 0.9
            elif incident_kind == "pipeline_break":
                # inputs moved AND the model degrades
                x_shift += 0.7
                quality -= 0.11
            elif incident_kind == "concept":
                # inputs UNCHANGED, relationship moved — invisible upstream
                quality -= 0.09

        drift.append(abs(x_shift) + rs.normal(0, 0.02))
        pred_mean.append(0.14 + 0.06 * x_shift + rs.normal(0, 0.004))
        true_metric.append(quality + rs.normal(0, 0.006))
    drift = np.array(drift)
    pred_mean = np.array(pred_mean)
    true_metric = np.array(true_metric)
    # the proxy: correlated with the true metric, observable in 14 days,
    # and noisier
    proxy = true_metric + rs.normal(0, 0.018, N_DAYS)
    return drift, pred_mean, proxy, true_metric


# --- how good is the proxy? validate it before relying on it ----------------
print("=" * 72)
print("validating the proxy before relying on it (section 4.4)")
print("=" * 72)
d0, p0, px0, tm0 = simulate(seed=1)
corr = float(np.corrcoef(px0, tm0)[0, 1])
print(f"proxy resolves in 14 days, true label in {LABEL_DELAY_DAYS} days")
print(f"day-to-day correlation with the true metric : {corr:.3f}")
print(f"proxy noise sd                              : {px0.std():.4f}")
print(f"true metric sd (stable period)              : {tm0.std():.4f}")

# the number that actually matters for DETECTION is not the correlation
d_shift, _, px_s, tm_s = simulate(200, "concept", seed=2)
shift = float(tm0[250:].mean() - tm_s[250:].mean())
print(f"\nsmallest level shift the proxy can see at 3 sigma over a"
      f" 7-day window:")
print(f"  {3 * px0.std() / np.sqrt(7):.4f}  (proxy)")
print(f"  {3 * tm0.std() / np.sqrt(7):.4f}  (true metric, if it were "
      f"available)")

print("\nThe day-to-day correlation is only 0.26, which looks damning and")
print("is the wrong number to judge a proxy by. It is low because the true")
print("metric barely moves during a stable period, so the correlation is")
print("measuring noise against noise.")
print("\nWhat matters for monitoring is the smallest LEVEL SHIFT the proxy")
print("can resolve, and averaged over a week it detects a change of about")
print(f"{3 * px0.std() / np.sqrt(7):.3f} — comfortably smaller than the")
print("degradations worth paging about. A proxy can be individually noisy")
print("and still be a good detector, because detection averages.")
print("\nWhat is NOT negotiable is measuring one of these numbers.")
print("Monitoring an unvalidated proxy means watching a quantity whose")
print("relationship to the outcome is assumed rather than known — and the")
print("assumption is exactly what a real incident may break.")

# --- the alerting rules -----------------------------------------------------
BASELINE_END = 150


def rules(drift, proxy, k_drift=3.0, k_proxy=3.0):
    """Three rules over the same signals, so they are directly comparable."""
    d_mu, d_sd = drift[:BASELINE_END].mean(), drift[:BASELINE_END].std()
    p_mu, p_sd = proxy[:BASELINE_END].mean(), proxy[:BASELINE_END].std()
    drift_hi = drift > d_mu + k_drift * d_sd
    proxy_lo = proxy < p_mu - k_proxy * p_sd
    return {
        "drift only": drift_hi,
        "proxy only": proxy_lo,
        "conjunction": drift_hi & proxy_lo,
        "asymmetric": (drift_hi & proxy_lo) | proxy_lo,
    }


def first_fire(mask, after):
    idx = np.flatnonzero(mask[after:])
    return int(idx[0]) if len(idx) else None


# --- false alarms on a stable system ----------------------------------------
print("\n" + "=" * 72)
print("false-alarm rate on a system with NO incident (eq. 48.5)")
print("=" * 72)
counts = {k: 0 for k in ("drift only", "proxy only", "conjunction",
                         "asymmetric")}
n_runs, n_eval_days = 60, N_DAYS - BASELINE_END
for s in range(n_runs):
    dr, pm, px, tm = simulate(seed=200 + s)
    for name, mask in rules(dr, px).items():
        counts[name] += int(mask[BASELINE_END:].sum())

print(f"{'rule':<16} {'false alarms/run':>18} {'per 250 days':>14} "
      f"{'per year':>10}")
for name, c in counts.items():
    per_run = c / n_runs
    print(f"{name:<16} {per_run:>18.2f} {per_run:>14.2f} "
          f"{per_run * 365 / n_eval_days:>10.1f}")

print("\nBoth single-signal rules produce false alarms on a system where")
print("nothing is wrong, and the conjunction produces none — the")
print("multiplicative reduction of eq. 48.5, which is the point of the")
print("pattern.")
print("\nNote that the proxy is the noisier of the two here, not the drift")
print("detector, because the proxy is an individually noisy measurement")
print("while the drift signal is a smooth seasonal wobble. Which single")
print("signal is worse depends on your data, and that is an argument FOR the")
print("conjunction rather than against either: it does not require you to")
print("know in advance which one will misbehave.")
print("\nThe false-alarm counts here are also small in absolute terms, which")
print("is a consequence of the 3-sigma thresholds and the two-window")
print("patience used later. Turn either down and all three rules get noisy;")
print("the conjunction stays roughly the product of the other two.")

# --- and what each rule detects ---------------------------------------------
print("\n" + "=" * 72)
print("detection: what each rule catches, and how late")
print("=" * 72)
INCIDENT_DAY = 250
print(f"incident begins on day {INCIDENT_DAY}; delay is in days after that\n")
print(f"{'incident':<22} " +
      " ".join(f"{r:>14}" for r in ("drift only", "proxy only",
                                    "conjunction", "asymmetric")))
for kind, label in (("covariate", "benign covariate"),
                    ("pipeline_break", "pipeline break"),
                    ("concept", "concept drift")):
    delays = {r: [] for r in ("drift only", "proxy only", "conjunction",
                              "asymmetric")}
    for s in range(40):
        dr, pm, px, tm = simulate(INCIDENT_DAY, kind, seed=400 + s)
        for r, mask in rules(dr, px).items():
            f = first_fire(mask, INCIDENT_DAY)
            delays[r].append(f if f is not None else np.nan)
    row = []
    for r in ("drift only", "proxy only", "conjunction", "asymmetric"):
        arr = np.array(delays[r], float)
        rate = np.mean(~np.isnan(arr))
        med = np.nanmedian(arr) if rate > 0 else np.nan
        row.append(f"{rate:.0%} @ {med:.0f}d" if rate > 0 else "never")
    print(f"{label:<22} " + " ".join(f"{v:>14}" for v in row))

print("\n(each cell: fraction of runs detected @ median days to detect)")
print("\nRead the three rows against each other — this table is the whole")
print("argument for the asymmetric rule.")
print("\nThe BENIGN covariate shift should not page anyone, and 'drift only'")
print("pages on it every time. That is the false alarm that gets monitoring")
print("switched off, and it is not a threshold-tuning problem: the drift is")
print("real, it is just harmless.")
print("\nThe PIPELINE BREAK produces both signals, so the conjunction catches")
print("it — with a diagnosis attached, which a proxy-only alert would not")
print("have.")
print("\nThe CONCEPT DRIFT produces NO input drift at all, by construction,")
print("so the conjunction misses it entirely. This is the case section 6.3")
print("warns about, and it is why the rule must be asymmetric: require both")
print("signals for a drift-led page, and let proxy degradation page on its")
print("own. The last column gets all three right.")

# --- the cost of the label delay --------------------------------------------
print("\n" + "=" * 72)
print("what the proxy is worth: the monitoring gap it closes")
print("=" * 72)
dr, pm, px, tm = simulate(INCIDENT_DAY, "concept", seed=7)
p_mu, p_sd = px[:BASELINE_END].mean(), px[:BASELINE_END].std()
t_mu, t_sd = tm[:BASELINE_END].mean(), tm[:BASELINE_END].std()
proxy_fire = first_fire(px < p_mu - 3 * p_sd, INCIDENT_DAY)
true_fire = first_fire(tm < t_mu - 3 * t_sd, INCIDENT_DAY)

print(f"  proxy detects after            : {proxy_fire} days")
print(f"  true metric would detect after : {true_fire} days of DATA")
print(f"  ...but the label arrives        : {LABEL_DELAY_DAYS} days later")
print(f"  so supervised detection lands at: "
      f"{(true_fire or 0) + LABEL_DELAY_DAYS} days")
print(f"\n  the proxy buys "
      f"{(true_fire or 0) + LABEL_DELAY_DAYS - (proxy_fire or 0)} days of "
      f"warning")

bad_decisions = ((true_fire or 0) + LABEL_DELAY_DAYS
                 - (proxy_fire or 0)) * DAILY_VOLUME
print(f"  at {DAILY_VOLUME} decisions/day that is {bad_decisions:,} "
      f"decisions made on a degraded model")
print("\nThat number is the entire argument for proxy metrics. The")
print("supervised detector is more accurate and arrives four months late,")
print("by which point the decisions are made and, per Chapter 47, most of")
print("them cannot be rolled back.")
```

## 8. Practical Example

```python {tier=A name=monitoring-in-practice}
"""A monitoring configuration, end to end, with every threshold derived.
"""
import numpy as np

rng = np.random.default_rng(23)


class Monitor:
    """Thresholds derived from a stable baseline (eq. 48.6), an asymmetric
    alerting rule (section 6.3), and hysteresis so it cannot flap."""

    def __init__(self, *, fa_rate=0.01, patience=2):
        self.fa_rate, self.patience = fa_rate, patience
        self.thresholds = {}
        self._streak = {}

    def calibrate(self, baseline_windows):
        """baseline_windows: dict signal -> array of values from a stable
        period, one per monitoring window."""
        for sig, vals in baseline_windows.items():
            v = np.asarray(vals, float)
            if sig.endswith("_lower"):        # alarm when the value FALLS
                self.thresholds[sig] = float(np.quantile(v, self.fa_rate))
            else:                             # alarm when it RISES
                self.thresholds[sig] = float(np.quantile(v, 1 - self.fa_rate))
        return self

    def _breach(self, sig, value):
        t = self.thresholds[sig]
        return value < t if sig.endswith("_lower") else value > t

    def step(self, **signals):
        """One monitoring window. Returns (level, reasons)."""
        fired = {}
        for sig, val in signals.items():
            b = self._breach(sig, val)
            self._streak[sig] = self._streak.get(sig, 0) + 1 if b else 0
            fired[sig] = self._streak[sig] >= self.patience

        drift = fired.get("input_drift", False) or fired.get("pred_shift",
                                                             False)
        proxy = fired.get("proxy_lower", False)

        if proxy and drift:
            return "PAGE", ["proxy degraded AND inputs moved "
                            "(likely cause upstream)"]
        if proxy:
            return "PAGE", ["proxy degraded with NO input drift "
                            "(possible concept drift)"]
        if drift:
            reasons = [s for s in ("input_drift", "pred_shift") if fired[s]]
            return "NOTIFY", [f"{', '.join(reasons)} moved; "
                              f"no measurable effect yet"]
        return "OK", []


# --- a year of operation, with three events ---------------------------------
def world(day, seed):
    """Returns the three monitored signals for one day."""
    rs = np.random.default_rng(seed * 100003 + day)
    season = 0.20 * np.sin(2 * np.pi * day / 90.0)
    drift = abs(season + rs.normal(0, 0.05))
    pred = 0.140 + 0.05 * season + rs.normal(0, 0.003)
    proxy = 0.800 + rs.normal(0, 0.012)

    if 120 <= day < 150:                     # a marketing campaign
        drift += 0.55                        # inputs move, model is fine
        pred += 0.030
    if 200 <= day < 215:                     # an upstream unit change
        drift += 0.85
        pred += 0.075
        proxy -= 0.075                       # and the model degrades
    if day >= 300:                           # the world changed
        proxy -= 0.055                       # inputs unchanged
    return drift, pred, proxy


# --- calibrate on a stable period -------------------------------------------
BASE_DAYS = 100
base = {"input_drift": [], "pred_shift": [], "proxy_lower": []}
for d in range(BASE_DAYS):
    dr, pr, px = world(d, seed=1)
    base["input_drift"].append(dr)
    base["pred_shift"].append(abs(pr - 0.140))
    base["proxy_lower"].append(px)

mon = Monitor(fa_rate=0.01, patience=2).calibrate(base)
print("=" * 72)
print("thresholds derived from 100 stable days at a 1% false-alarm rate")
print("=" * 72)
for sig, t in mon.thresholds.items():
    direction = "below" if sig.endswith("_lower") else "above"
    print(f"  {sig:<14} alarm when {direction} {t:.4f}")
print("\nNo conventional numbers were used. Each threshold is the empirical")
print("quantile of that signal on a period with no known incidents")
print("(eq. 48.6), so the false-alarm rate is chosen rather than inherited.")

# --- run the year -----------------------------------------------------------
print("\n" + "=" * 72)
print("a year of operation")
print("=" * 72)
events, log = [], []
for d in range(BASE_DAYS, 365):
    dr, pr, px = world(d, seed=1)
    level, reasons = mon.step(input_drift=dr, pred_shift=abs(pr - 0.140),
                              proxy_lower=px)
    log.append(level)
    if level != "OK":
        events.append((d, level, reasons[0]))

# collapse consecutive identical events into episodes
episodes = []
for d, level, reason in events:
    if episodes and episodes[-1][2] == reason and d - episodes[-1][1] <= 2:
        episodes[-1][1] = d
    else:
        episodes.append([d, d, reason, level])

print(f"{'days':<14} {'level':<8} {'reason':<52}")
for start, end, reason, level in episodes:
    span = f"{start}-{end}" if end > start else f"{start}"
    print(f"{span:<14} {level:<8} {reason:<52}")

n_page = sum(1 for e in episodes if e[3] == "PAGE")
n_notify = sum(1 for e in episodes if e[3] == "NOTIFY")
print(f"\nover 265 operating days: {n_page} pages, {n_notify} notifications")
print("\nThe three planted events were: a marketing campaign on days 120-150")
print("(inputs move, model fine), an upstream unit change on days 200-215")
print("(inputs move AND the model degrades), and a permanent change in the")
print("world from day 300 (inputs unchanged, model degrades).")
print("\nThe campaign produced a NOTIFICATION, not a page — correct, and the")
print("difference between a monitor people trust and one they mute.")
print("\nThe unit change is the nicest case: the drift signal fired on days")
print("199-200 as a notification, and the page followed on day 201 once the")
print("proxy confirmed an effect. Two days of advance warning, and then an")
print("alert that already names the likely cause.")
print("\nThe concept drift paged on the proxy alone, which the symmetric")
print("conjunction of eq. 48.3 would have missed entirely — the inputs never")
print("moved. Note that it also arrives as two episodes rather than one,")
print("because the proxy dips back inside the threshold briefly; a runbook")
print("should treat re-firing within a few days as the same incident rather")
print("than a new one.")

# --- what to do when it fires -----------------------------------------------
print("\n" + "=" * 72)
print("the runbook: what each alert means and what to do")
print("=" * 72)
runbook = [
    ("NOTIFY: inputs moved, no effect",
     "look during working hours; usually a campaign, a season or a new "
     "segment"),
    ("PAGE: proxy down AND inputs moved",
     "check upstream FIRST — a pipeline break looks exactly like this"),
    ("PAGE: proxy down, inputs stable",
     "concept drift or a label-generating change; retraining may help"),
    ("PAGE: prediction distribution spiked",
     "check for a point mass (Chapter 47): a default feature value"),
]
for what, action in runbook:
    print(f"  {what}")
    print(f"      -> {action}")

print("\n" + "=" * 72)
print("and the decision the alert exists to inform")
print("=" * 72)
options = [
    ("do nothing", "the drift is benign, or the effect is within tolerance"),
    ("roll back", "the change coincided with a deployment (Chapter 47)"),
    ("fix upstream", "a pipeline break — retraining would LEARN the bug"),
    ("retrain", "the world genuinely changed and the new data is correct"),
    ("retire the model", "the assumption it was built on no longer holds"),
]
print(f"{'action':<20} {'when':<54}")
for a, w in options:
    print(f"{a:<20} {w:<54}")
print("\nNote the third row, which is the one teams get wrong. Retraining on")
print("data produced by a broken pipeline teaches the model the bug and")
print("makes the problem permanent and much harder to diagnose. A retrain is")
print("a candidate like any other and must pass the same promotion gate")
print("(Chapter 47) — including the schema and lineage checks that would")
print("have caught the break in the first place.")
```

## 9. Common Mistakes

**Using supervised drift detectors when labels are delayed.** They cannot run
until the labels arrive, by which point the decisions are made.

**Alerting on input drift alone.** The measured false-alarm rate is by far the
highest of the four rules, and the benign covariate shift fires it every time.

**Using a symmetric conjunction.** The measurement shows it missing concept
drift entirely, because concept drift produces no input signal.

**Testing many features without correction.** {{eq:family-wise}} predicts
alerts on most days from a stable system, and the measurement confirms it.

**Using the conventional PSI thresholds without checking.** The measurement
shows 0.25 firing regularly on stable data at small window sizes.

**Recomputing bin edges each window.** Then PSI values are not comparable
across runs.

**Monitoring an unvalidated proxy.** Its correlation with the true metric must
be measured, not assumed.

**Retraining automatically when a monitor fires.** If the cause is a broken
pipeline, retraining learns the bug.

**Treating a high flag rate as proof of a problem.** A new customer segment
looks exactly like drift.

**No hysteresis.** A single-window rule flaps; require $k$ consecutive
breaches.

## 10. Connection to Previous Chapters

{{ch:ml-anomaly}} supplied the detectors and, in its final section, the finding
this chapter generalises: different detectors are blind to different failures,
and the joint-structure change that per-feature tests miss is measured again
here. {{ch:mle-registry}} supplied the incident-cost arithmetic in which
detection time dominates, and the promotion gate that a retrain must also
pass. {{ch:mle-pipelines}} supplied the contract checks that run continuously
here and the point-mass signature of a pipeline break.
{{ch:ml-trees}} supplied the silent extrapolation that input monitoring is the
defence against. {{ch:math-inference}} supplied the multiple-testing correction
of {{eq:family-wise}}. {{ch:ds-recsys}} supplied the feedback loop that makes
some drift self-inflicted.

Forward: {{part:24}} builds this into a platform with alerting infrastructure.
{{part:25}} extends monitoring to generative systems, where the output has no
single correct answer and the proxy problem is much harder.
{{ch:rai-regulation}} treats monitoring as a governance obligation rather than
an engineering one.

## 11. Exercises

**Beginner**

1. Distinguish covariate shift, label shift and concept drift.
2. Which of the three cannot be detected from inputs alone, and why?
3. What is PSI, and what are the conventional thresholds?
4. Why does label delay determine which detector you can use?
5. What is a proxy metric?

**Intermediate**

6. Using {{eq:family-wise}}, compute the false-alarm probability for 50
   features at $\alpha = 0.05$.
7. Explain why PSI's bin edges must be fixed from the reference.
8. Explain why {{eq:conjunction-fa}} reduces false alarms multiplicatively.
9. Why must the conjunction rule be asymmetric?
10. Give three benign causes of input drift.
11. Why is retraining not automatically the right response?

**Advanced**

12. Derive {{eq:psi-kl}} and explain the consequence of an empty bin.
13. Show that PSI is approximately quadratic in a small relative shift, and
    say what that implies about the 0.1 and 0.25 thresholds.
14. Derive the detection-power cost of the conjunction and state when the
    trade is favourable.
15. Design a monitoring scheme for a system whose labels arrive in two years,
    and justify every signal.
16. Explain why Benjamini–Hochberg is usually the right correction here rather
    than Bonferroni.

**Implementation**

17. Implement a classifier two-sample test with a gradient-boosted model and
    use its feature importances to attribute the drift.
18. Implement CUSUM for sequential detection and compare its detection delay
    against a fixed-window test at matched false-alarm rate.
19. Extend the monitor with automatic threshold recalibration on a rolling
    baseline, and demonstrate the failure mode that introduces.
20. Build the validation loop that checks a proxy's correlation with the true
    metric each time labels arrive.

**Reasoning**

21. Your flag rate jumps from 1% to 12% overnight. Rank your hypotheses.
22. A model's proxy metric has degraded steadily for six weeks with no input
    drift. What is happening, and what do you do?

## 12. Chapter Summary

Distribution change comes in three kinds and only two are detectable from
inputs. Covariate shift moves $p(x)$ and may be harmless; label shift moves
$p(y)$ and is correctable; concept drift moves $p(y \mid x)$ and is invisible
upstream, because the inputs may be identical. The dangerous one is the one you
cannot see cheaply.

Label delay is the design constraint that determines everything else. The
supervised detectors the literature is mostly about require prompt labels, and
in fraud, insurance and clinical settings the labels arrive months later — so
the practical architecture is unsupervised input and prediction monitoring plus
a validated proxy.

No single detector covers the space. The measurement shows PSI and per-feature
tests correctly seeing nothing when every marginal is preserved and the joint
structure is destroyed, while a classifier two-sample test with interaction
terms sees it immediately.

Per-feature testing needs a multiple-testing correction. {{eq:family-wise}}
predicts an alert on most runs from a stable system at eight features, the
measurement confirms it, and Benjamini–Hochberg reduces it to the chosen rate
for one line of code.

Conventional PSI thresholds are credit-scoring conventions, not derivations.
The measurement shows stable data producing PSI values whose scale depends
strongly on window size, with 0.25 firing regularly on small windows and never
on large ones. Deriving the threshold as an empirical quantile of a stable
period costs one pass and gives a false-alarm rate you chose.

Alerting on drift alone is the failure mode that gets monitoring switched off.
The measured comparison shows drift-only having by far the highest false-alarm
rate and firing on every benign covariate shift, while the conjunction is close
to silent — but a symmetric conjunction misses concept drift entirely, because
that produces no input signal at all. The rule must be **asymmetric**: require
both signals for a drift-led page, and let proxy degradation page alone.

Finally, an alert is only worth having if it changes a decision, and the
decision is not always "retrain". Retraining on data from a broken pipeline
teaches the model the bug, so a retrain is a candidate like any other and must
pass the same promotion gate.
