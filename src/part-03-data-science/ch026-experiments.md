---
id: ds-experiments
number: 26
part: III
tier: focused
status: reviewed
requires: [ds-causation, math-inference]
provides: [ab-test, randomisation-unit, sample-ratio-mismatch,
           guardrail-metric, novelty-effect, peeking]
citations: [kohavi2009]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Design an A/B test: hypothesis, metric, randomisation unit, duration.
2. Choose the randomisation unit correctly and explain the consequences of
   getting it wrong.
3. Compute the required sample size before running anything.
4. Run the sanity checks that detect a broken experiment.
5. Explain why peeking inflates the false-positive rate, and quantify it.
6. Recognise novelty effects, interference and other threats to validity.
7. Interpret a result honestly, including a null one.

## 2. Why This Matters

{{ch:ds-causation}} established that only randomisation reliably supports a
causal claim. This chapter is how that is done in practice.

The uncomfortable empirical finding from organisations that run experiments at
scale is that **most ideas do not work**. {{cite:kohavi2009}} reports that only a
minority of tested changes improve the metric they target, and a substantial
fraction actively harm it. That is the value of experimentation: not confirming
good ideas, but cheaply identifying the majority that are not.

The failure modes are specific and common. An experiment randomised at the wrong
unit produces intervals that are far too narrow ({{ch:ds-collection}}). An
experiment stopped when it first looked significant has a false-positive rate
several times its nominal level. An experiment measuring only the primary metric
ships a change that improved conversion and broke latency. Each of these
produces a confident, wrong decision, and each is preventable by a check that
takes minutes.

## 3. Prerequisites

{{ch:ds-causation}} for randomisation and why it works; {{ch:math-inference}}
for hypothesis testing, power, p-values and multiple comparisons;
{{ch:ds-collection}} for the design effect, which reappears here as the
randomisation unit.

## 4. Intuitive Explanation

### 4.1 The design, in order

An {{term:ab-test}} is specified before it runs. The order matters, because each
decision constrains the next:

```text
1. hypothesis        what change, expected to do what, why
2. primary metric    one number, precisely defined
3. guardrails        what must not get worse
4. effect size       the smallest lift worth shipping
5. randomisation unit what gets assigned
6. sample size       computed from 2, 4 and 5
7. duration          from 6 and traffic; at least one full weekly cycle
8. analysis plan     the test, written down, before any data
```

Writing steps 1-8 down before starting is what makes the result interpretable.
An analysis plan chosen after seeing the data is a selection procedure, not a
test ({{ch:math-inference}}).

### 4.2 The randomisation unit

The {{term:randomisation-unit}} is what you assign. It is usually a user, and
choosing something smaller is the most common structural error.

**Randomise by request or session**, and the same person sees both variants.
That contaminates the comparison — their behaviour under B is influenced by
having seen A — and, worse, the observations are not independent, so every
standard error is wrong.

**Randomise by user**, and the analysis must be at the user level too. If you
randomise by user and then analyse per-event, you have the clustering problem of
{{ch:ds-collection}}: the design effect inflates the true variance by
$1 + (\bar{m}-1)\rho$, and intervals computed on events are far too narrow.

> IMPORTANT: The rule is that **the unit of randomisation must be the unit of
> analysis.** Randomise by user, analyse per user. If the metric is naturally
> per-event, aggregate it to a per-user value first — mean revenue per user, not
> mean revenue per order.

### 4.3 Guardrails

The primary metric is what you hope to improve. A {{term:guardrail-metric}} is
what you refuse to damage.

A checkout redesign that raises conversion by 2% and increases page latency by
400 ms may be a net loss. A recommendation change that raises clicks and reduces
session length has probably found a way to be annoying. Without guardrails these
ship, because the primary metric moved the right way.

Standard guardrails: latency, error rate, crash rate, unsubscribes, support
contacts, and a revenue-per-user figure if the primary metric is engagement.

### 4.4 Peeking

The most common analytical error is checking the result repeatedly and stopping
when $p < 0.05$.

This does not merely bend the rules slightly. A fixed-sample test controls the
false-positive rate at 5% *for one look*. Checking daily for two weeks gives
fourteen opportunities, and under the null the probability of crossing the
threshold at some point is far above 5% — {{sec:6-mathematical-foundation}}
measures it at around 25-30% for realistic checking schedules.

{{term:peeking}} feels harmless because each individual test is valid. The
problem is the stopping rule, which selects for the noisiest moment.

## 5. Formal Explanation

### 5.1 Sample size

From {{ch:math-inference}}, for two proportions:

$$
n \approx \frac{2\,(z_{\alpha/2} + z_{\beta})^{2}\,\bar{p}(1-\bar{p})}{\delta^{2}}
$$ (eq:ab-sample-size)

per variant, with $z_{0.025} = 1.96$ and $z_{0.20} = 0.84$.

For a continuous metric with standard deviation $\sigma$:

$$
n \approx \frac{2\,(z_{\alpha/2} + z_{\beta})^{2}\,\sigma^{2}}{\delta^{2}}
$$ (eq:ab-sample-size-continuous)

The $\delta^{2}$ is the binding constraint: halving the detectable effect
quadruples the cost. Compute this **before** running, and if the answer exceeds
your available traffic, the experiment cannot answer the question — which is
worth knowing before spending three weeks discovering it.

### 5.2 Sanity checks

Three checks run before looking at the result. If any fails, the experiment is
broken and the result must be discarded rather than interpreted.

**{{term:sample-ratio-mismatch}}.** The observed split should match the intended
one. Test with chi-squared:

$$
\chi^{2} = \sum_{i} \frac{(O_i - E_i)^{2}}{E_i}
$$ (eq:srm-chi-squared)

An SRM p-value below about 0.001 almost always indicates a bug — a redirect that
drops users, a bot filter applied to one arm, a logging failure. It does not
indicate an interesting finding.

**Pre-experiment equivalence.** Metrics measured *before* assignment should not
differ between arms. If they do, randomisation did not work.

**A/A test.** Run the experiment with both arms identical. It should find
nothing about 5% of the time. More often means the pipeline is broken.

### 5.3 Threats to validity

{#tbl:validity-threats caption="Threats to an experiment's validity, and the standard mitigation for each."}

| Threat | Mechanism | Mitigation |
|---|---|---|
| {{term:novelty-effect}} | users react to newness, not quality | run longer; compare weeks |
| Primacy | users prefer the familiar; early results understate | same |
| Interference | treated users affect control users | cluster randomisation |
| Selection | assignment correlates with a covariate | check pre-period balance |
| Survivorship | analysis restricted to users who stayed | analyse all assigned |
| Multiple metrics | many tests, one significant | pre-register the primary |
| Seasonality | the period is unrepresentative | full weekly cycles |

**Interference** deserves note because it invalidates the independence
assumption entirely. In a marketplace, showing a discount to the treatment group
consumes inventory that the control group would have bought. In a social
network, treated users influence their untreated friends. Standard A/B
randomisation is invalid in both, and the fix is to randomise clusters —
geographies, markets, social communities — at a substantial cost in power.

### 5.4 Analysis

For the difference in proportions:

$$
\hat{\delta} = \hat{p}_B - \hat{p}_A,
\qquad
\text{SE} = \sqrt{\frac{\hat{p}_A(1-\hat{p}_A)}{n_A}
                 + \frac{\hat{p}_B(1-\hat{p}_B)}{n_B}}
$$ (eq:ab-analysis)

Report the estimated effect with its confidence interval, not just a p-value.
"+1.2% [0.3%, 2.1%]" is informative; "p = 0.011" is not.

**Report null results as intervals.** "No significant difference" is
uninformative. "The effect was +0.1% [−0.9%, +1.1%]" says the experiment ruled
out anything larger than about 1%, which is a real finding. If the interval is
[−8%, +8%], the experiment was underpowered and concluded nothing at all.

### 5.5 Experimenting on people

An A/B test is an experiment on people who did not consent to a specific
intervention, and that constrains what may be tested.

Three practices are standard where experimentation is mature.

**A harm threshold with automatic termination.** Define in advance the
degradation that stops the experiment immediately, and automate it. An
experiment that is clearly harming its treatment arm should not wait for the
weekly review.

**Restraint on sensitive manipulations.** Experiments that deliberately affect
mood, exploit known biases, or vary the price shown to comparable customers
attract legitimate objection and, in several jurisdictions, legal exposure.
"We can measure it" is not the same as "we may do it."

**Review for anything beyond interface changes.** A layout test needs no
oversight. An experiment varying what a vulnerable user is shown does, and the
distinction is one an experimentation platform should encode rather than leave
to individual judgement.

There is also a purely practical point. Running many simultaneous experiments on
overlapping populations creates interactions between them, and a platform
without an assignment layer that accounts for this will report effects that are
partly artefacts of other experiments running at the same time.

> IMPORTANT: The asymmetry worth internalising is that the control arm is also
> being experimented on. Withholding an improvement you believe in is a
> deliberate act with a cost, which is why experiments should be sized to
> conclude promptly rather than left running indefinitely for extra precision.

### 5.6 Variance reduction

Because cost scales with $\sigma^{2}/\delta^{2}$, reducing variance is
equivalent to getting more traffic for free.

**CUPED** — using pre-experiment data — is the standard technique. If $X$ is a
pre-period measurement of the same metric, define

$$
Y^{\text{adj}} = Y - \theta\,(X - \bar{X}),
\qquad
\theta = \frac{\Cov(Y, X)}{\Var(X)}
$$ (eq:cuped)

The adjusted metric has the same expectation — $X$ is pre-treatment, so it
cannot be affected by the assignment — and variance reduced by a factor
$1 - \rho^{2}$, where $\rho$ is the correlation between pre and post
measurements.

With $\rho = 0.7$, variance falls by 49% and the required sample size roughly
halves. This is the same idea as adjusting for an outcome-only predictor in
{{ch:ds-causation}}: free precision from a variable that cannot introduce bias.

## 6. Mathematical Foundation

### 6.1 Why peeking inflates the false-positive rate

Under the null, the test statistic follows a random walk as data accumulates.
A fixed-sample test asks whether it exceeds the threshold at one specified
point. Peeking asks whether it *ever* exceeds it.

If the statistic were independent at each of $k$ looks, the familywise error
would be $1 - (1-\alpha)^{k}$ — 40% for $k = 10$. Successive looks are strongly
correlated, since each shares most of its data with the last, so the true
inflation is smaller than that bound but still severe.

For continuous monitoring with a fixed threshold, the false-positive rate
approaches **1** as the sample grows without limit: a random walk crosses any
fixed boundary eventually with probability one. In practice, with daily checks
over two weeks, the rate is typically 20-30% rather than 5%.

{{sec:7-implementation}} measures this by simulation, which is the only
convincing way to see it.

The fixes:

- **Fix the sample size in advance** and look once. Simplest and always correct.
- **Sequential testing** with an alpha-spending function, which adjusts the
  threshold at each look so the total error stays at $\alpha$.
- **Always-valid inference**, which provides confidence sequences valid at every
  time point simultaneously.

### 6.2 The cost of the wrong randomisation unit

If you randomise by user but analyse by event, the design effect from
{{ch:ds-collection}} applies:

$$
\text{DEFF} = 1 + (\bar{m} - 1)\rho
$$

with $\bar{m}$ events per user and $\rho$ the intra-user correlation.

The standard error is understated by $\sqrt{\text{DEFF}}$, so a nominal 95%
interval has true coverage far below 95%. With 20 events per user and
$\rho = 0.25$, DEFF is 5.75 and the interval is 2.4× too narrow — meaning a
"significant" result at $p = 0.01$ may not be significant at all.

The correct analysis aggregates to one value per user first. This throws away
nothing: the per-user mean is the sufficient statistic for a per-user effect.

### 6.3 Why one weekly cycle is a minimum

Behaviour varies systematically by day of week. Running Monday to Thursday
measures weekday behaviour, and the estimate generalises to a population that
includes weekends only if the treatment effect is identical on both — which is
exactly the sort of assumption experiments exist to avoid.

Running a whole number of weeks also removes day-of-week composition as a
difference between arms if enrolment is continuous.

Novelty is the other timing consideration. If a treatment effect decays,
measuring in week one overstates the durable effect. The diagnostic is to
compare the effect estimated in week one against week two: a large decline
suggests novelty rather than genuine improvement.

## 7. Implementation

```python {tier=A name=ab-test-analysis}
"""Designing and analysing an A/B test, with the failure modes measured.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# --- eq. 26.1: sample size before anything else -----------------------------
def sample_size(baseline, mde, alpha=0.05, power=0.80):
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    p = baseline
    return int(np.ceil(2 * (za + zb) ** 2 * p * (1 - p) / mde ** 2))


print("=" * 72)
print("design: how much traffic does this question need?")
print("=" * 72)
print(f"{'baseline':>9} {'MDE':>7} {'n per arm':>11} {'days at 5k/day':>16}")
for baseline in (0.05, 0.15):
    for mde in (0.005, 0.01, 0.02):
        n = sample_size(baseline, mde)
        print(f"{baseline:>9.0%} {mde:>7.1%} {n:>11,} {2*n/5000:>16.1f}")
print("\nHalving the MDE quadruples the cost (the delta^2 in eq. 26.1).")

# --- sanity check: sample ratio mismatch ------------------------------------
print("\n" + "=" * 72)
print("sanity check 1: sample ratio mismatch (eq. 26.3)")
print("=" * 72)


def srm_check(n_a, n_b, expected=0.5):
    total = n_a + n_b
    exp_a, exp_b = total * expected, total * (1 - expected)
    chi2 = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    return chi2, 1 - stats.chi2.cdf(chi2, df=1)


print(f"{'split':<22} {'chi2':>9} {'p-value':>11} {'verdict'}")
for label, (a, b) in {
    "50000 / 50000 (clean)": (50_000, 50_000),
    "50000 / 49800 (noise)": (50_000, 49_800),
    "50000 / 48500 (BUG)":   (50_000, 48_500),
}.items():
    chi2, p = srm_check(a, b)
    verdict = "BROKEN — discard" if p < 0.001 else "fine"
    print(f"{label:<22} {chi2:>9.2f} {p:>11.2e} {verdict}")
print("\nAn SRM is a bug report, not a finding. A 1.5% shortfall in one arm")
print("has a p-value near zero at this scale and means users were lost.")

# --- section 6.1: peeking, measured -----------------------------------------
print("\n" + "=" * 72)
print("peeking: the false-positive rate under a NULL effect")
print("=" * 72)


def run_experiment(n_total, true_lift, peeks, baseline=0.10, alpha=0.05):
    """Return True if the experiment 'wins' under the given peeking schedule."""
    a = rng.random(n_total) < baseline
    b = rng.random(n_total) < baseline + true_lift
    checkpoints = np.linspace(n_total // peeks, n_total, peeks).astype(int)
    for n in checkpoints:
        pa, pb = a[:n].mean(), b[:n].mean()
        se = np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
        if se > 0 and abs(pb - pa) / se > stats.norm.ppf(1 - alpha / 2):
            return True
    return False


N, TRIALS = 20_000, 1500
print(f"{'peeks':>7} {'false-positive rate':>22} {'inflation vs 5%':>18}")
for peeks in (1, 2, 5, 10, 20):
    hits = sum(run_experiment(N, 0.0, peeks) for _ in range(TRIALS))
    rate = hits / TRIALS
    print(f"{peeks:>7} {rate:>21.1%} {rate/0.05:>17.1f}x")

print("\nThere is no real effect in any of these runs. Checking twenty times")
print("instead of once turns a 5% error rate into something several times")
print("larger — the stopping rule selects for the noisiest moment.")

# --- eq. 26.4: analysis, reported properly ----------------------------------
print("\n" + "=" * 72)
print("analysis: report the interval, not the p-value")
print("=" * 72)


def analyse(conv_a, n_a, conv_b, n_b, alpha=0.05):
    pa, pb = conv_a / n_a, conv_b / n_b
    diff = pb - pa
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {"a": pa, "b": pb, "diff": diff, "ci": (diff - half, diff + half),
            "p": p, "rel": diff / pa}


scenarios = {
    "clear win":        (4800, 60_000, 5250, 60_000),
    "no effect, tight": (6000, 80_000, 6030, 80_000),
    "underpowered":     (140,   1_500,  160,   1_500),
}
for label, (ca, na, cb, nb) in scenarios.items():
    r = analyse(ca, na, cb, nb)
    lo, hi = r["ci"]
    sig = "significant" if r["p"] < 0.05 else "not significant"
    print(f"\n{label}:")
    print(f"  A {r['a']:.3%}  B {r['b']:.3%}  "
          f"diff {r['diff']:+.3%} ({r['rel']:+.1%} relative)")
    print(f"  95% CI [{lo:+.3%}, {hi:+.3%}]   p = {r['p']:.4f}  ({sig})")

print("\nThe second and third are both 'not significant' and mean completely")
print("different things. The tight one rules out any effect above ~0.2pp;")
print("the underpowered one rules out nothing and should not be reported as")
print("evidence of no effect.")

# --- eq. 26.6: CUPED variance reduction -------------------------------------
print("\n" + "=" * 72)
print("CUPED: free precision from pre-experiment data (eq. 26.6)")
print("=" * 72)

n = 40_000
for rho in (0.3, 0.5, 0.7, 0.9):
    pre = rng.normal(100, 25, n)
    noise_sd = 25 * np.sqrt(1 / rho ** 2 - 1)
    post = pre + rng.normal(0, noise_sd, n)
    assign = rng.random(n) < 0.5
    post = post + assign * 2.0                       # a true +2.0 effect

    naive = post[assign].mean() - post[~assign].mean()
    naive_se = np.sqrt(post[assign].var(ddof=1) / assign.sum()
                       + post[~assign].var(ddof=1) / (~assign).sum())

    theta = np.cov(post, pre)[0, 1] / pre.var(ddof=1)
    adj = post - theta * (pre - pre.mean())
    cuped = adj[assign].mean() - adj[~assign].mean()
    cuped_se = np.sqrt(adj[assign].var(ddof=1) / assign.sum()
                       + adj[~assign].var(ddof=1) / (~assign).sum())

    actual_rho = np.corrcoef(pre, post)[0, 1]
    print(f"rho={actual_rho:.2f}  naive {naive:+.3f} +/- {1.96*naive_se:.3f}   "
          f"CUPED {cuped:+.3f} +/- {1.96*cuped_se:.3f}   "
          f"variance x{(cuped_se/naive_se)**2:.2f} "
          f"(predicted {1-actual_rho**2:.2f})")

print("\nBoth estimate the same effect. CUPED's variance reduction of")
print("1 - rho^2 means at rho=0.7 you need roughly half the traffic.")
```

## 8. Practical Example

Running an experiment end to end, including the checks that should gate the
result.

```python {tier=A name=experiment-end-to-end}
"""A complete experiment: design, sanity checks, analysis, decision.

The randomisation-unit error is demonstrated explicitly, because it is the
most common way an experiment's conclusion is wrong while looking correct.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# --- the design, fixed in advance -------------------------------------------
DESIGN = {
    "hypothesis": "a simplified checkout raises completion",
    "primary": "checkout completion rate per user",
    "guardrails": ["p95 latency", "support contacts per user"],
    "baseline": 0.22,
    "mde": 0.015,
    "alpha": 0.05,
    "power": 0.80,
}
za = stats.norm.ppf(1 - DESIGN["alpha"] / 2)
zb = stats.norm.ppf(DESIGN["power"])
p = DESIGN["baseline"]
n_required = int(np.ceil(2 * (za + zb) ** 2 * p * (1 - p) / DESIGN["mde"] ** 2))

print("PRE-REGISTERED DESIGN")
for k, v in DESIGN.items():
    print(f"  {k:<12} {v}")
print(f"  {'n per arm':<12} {n_required:,}")

# --- simulate the experiment -------------------------------------------------
n_users = n_required
TRUE_LIFT = 0.018                       # a real effect, slightly above the MDE

user_prop = rng.beta(2, 6, n_users * 2)          # per-user baseline propensity
assign = rng.random(n_users * 2) < 0.5
sessions = rng.poisson(6, n_users * 2) + 1       # multiple sessions per user

completed, total_sessions, latency, support = [], [], [], []
for i in range(n_users * 2):
    p_i = np.clip(user_prop[i] + (TRUE_LIFT if assign[i] else 0), 0, 1)
    s = sessions[i]
    c = rng.binomial(s, p_i)
    completed.append(c)
    total_sessions.append(s)
    latency.append(rng.gamma(3, 60) * (1.22 if assign[i] else 1.0))  # slower!
    support.append(rng.poisson(0.05))

completed = np.array(completed); total_sessions = np.array(total_sessions)
latency = np.array(latency); support = np.array(support)

# --- sanity checks before looking at the result -----------------------------
print("\n" + "=" * 72)
print("SANITY CHECKS")
print("=" * 72)
n_a, n_b = int((~assign).sum()), int(assign.sum())
chi2 = ((n_a - (n_a+n_b)/2) ** 2 / ((n_a+n_b)/2)
        + (n_b - (n_a+n_b)/2) ** 2 / ((n_a+n_b)/2))
srm_p = 1 - stats.chi2.cdf(chi2, 1)
print(f"  sample ratio    : {n_a:,} / {n_b:,}  p = {srm_p:.3f}  "
      f"{'OK' if srm_p > 0.001 else 'BROKEN'}")

pre_a, pre_b = user_prop[~assign].mean(), user_prop[assign].mean()
pre_se = np.sqrt(user_prop[~assign].var()/n_a + user_prop[assign].var()/n_b)
pre_z = (pre_b - pre_a) / pre_se
print(f"  pre-period balance: {pre_a:.4f} vs {pre_b:.4f}, z = {pre_z:+.2f}  "
      f"{'OK' if abs(pre_z) < 3 else 'IMBALANCED'}")

# --- the randomisation-unit error -------------------------------------------
print("\n" + "=" * 72)
print("ANALYSIS — and why the unit matters")
print("=" * 72)

# WRONG: analyse per session, when randomisation was per user.
sess_a = completed[~assign].sum() / total_sessions[~assign].sum()
sess_b = completed[assign].sum() / total_sessions[assign].sum()
n_sess_a, n_sess_b = total_sessions[~assign].sum(), total_sessions[assign].sum()
se_sess = np.sqrt(sess_a*(1-sess_a)/n_sess_a + sess_b*(1-sess_b)/n_sess_b)

# RIGHT: aggregate to one value per user first.
rate_a = (completed / total_sessions)[~assign]
rate_b = (completed / total_sessions)[assign]
se_user = np.sqrt(rate_a.var(ddof=1)/len(rate_a) + rate_b.var(ddof=1)/len(rate_b))
diff_user = rate_b.mean() - rate_a.mean()

print(f"{'analysis unit':<18} {'estimate':>10} {'std error':>11} "
      f"{'95% CI':>22} {'z':>7}")
print(f"{'per session (WRONG)':<18} {sess_b-sess_a:>+10.4f} {se_sess:>11.5f} "
      f"{f'[{sess_b-sess_a-1.96*se_sess:+.4f}, {sess_b-sess_a+1.96*se_sess:+.4f}]':>22} "
      f"{(sess_b-sess_a)/se_sess:>7.1f}")
print(f"{'per user (RIGHT)':<18} {diff_user:>+10.4f} {se_user:>11.5f} "
      f"{f'[{diff_user-1.96*se_user:+.4f}, {diff_user+1.96*se_user:+.4f}]':>22} "
      f"{diff_user/se_user:>7.1f}")
print(f"\nThe per-session interval is {se_user/se_sess:.1f}x too narrow. Both")
print("estimate a similar effect; only one reports honest uncertainty.")
print("Randomisation was per user, so analysis must be per user (section 6.2).")

# --- guardrails ---------------------------------------------------------------
print("\n" + "=" * 72)
print("GUARDRAILS")
print("=" * 72)
lat_a, lat_b = np.percentile(latency[~assign], 95), np.percentile(latency[assign], 95)
sup_a, sup_b = support[~assign].mean(), support[assign].mean()
sup_se = np.sqrt(support[~assign].var()/n_a + support[assign].var()/n_b)

print(f"  p95 latency        : {lat_a:.0f} ms -> {lat_b:.0f} ms  "
      f"({(lat_b/lat_a - 1):+.1%})   "
      f"{'BREACH' if lat_b/lat_a - 1 > 0.05 else 'ok'}")
print(f"  support per user   : {sup_a:.4f} -> {sup_b:.4f}  "
      f"(z = {(sup_b-sup_a)/sup_se:+.2f})   ok")

# --- the decision -------------------------------------------------------------
print("\n" + "=" * 72)
print("DECISION")
print("=" * 72)
significant = abs(diff_user / se_user) > 1.96
practical = diff_user > DESIGN["mde"]
guardrail_ok = (lat_b / lat_a - 1) <= 0.05

print(f"  primary metric significant : {significant}")
print(f"  effect exceeds the MDE     : {practical} "
      f"({diff_user:+.4f} vs {DESIGN['mde']:+.4f})")
print(f"  guardrails passed          : {guardrail_ok}")
print(f"\n  -> {'SHIP' if (significant and practical and guardrail_ok) else 'DO NOT SHIP'}")
print("\nThe primary metric moved in the right direction and is statistically")
print("significant. The latency guardrail failed. Without the guardrail this")
print("would have shipped a change that trades checkout completion against")
print("page speed — a trade nobody agreed to make.")
```

## 9. Common Mistakes

**Peeking and stopping early.** Inflates the false-positive rate several-fold.

**Randomising by session, analysing by user, or the reverse.** The unit of
randomisation must be the unit of analysis.

**Analysing per event after randomising per user.** Intervals far too narrow
({{ch:ds-collection}}).

**No guardrails.** Ships wins bought at unmeasured cost.

**Not computing the sample size first.** Underpowered experiments waste the
traffic and produce misleading positives ({{ch:math-inference}}).

**Ignoring an SRM.** It is a bug, not a finding.

**Running less than a full week.** Day-of-week composition contaminates the
comparison.

**Reporting a null result without an interval.** "No difference" and "ruled out
anything above 0.2%" are different claims.

**Testing many metrics and reporting the significant one.** Pre-register the
primary.

**Ignoring interference.** In marketplaces and social products, standard A/B
randomisation is invalid.

**Shipping on a week-one result.** Novelty effects decay.

## 10. Connection to Previous Chapters

{{ch:ds-causation}} established why randomisation licenses a causal claim; this
chapter is the practical implementation, and {{eq:ignorability}} is what every
sanity check here is verifying held. {{ch:math-inference}} supplies the sample
size, the confidence intervals, the power calculation and the
multiple-comparisons argument that peeking violates. {{ch:ds-collection}}
supplies the design effect, which reappears as the randomisation-unit error and
is measured in {{sec:8-practical-example}}.

Forward: {{ch:ds-leakage}} covers the offline analogue — a validation split that
does not resemble deployment.

Beyond Part III: {{ch:ev-online}} extends experimentation to model deployment
and regression gates; {{ch:sd-fault-tolerance}} covers the rollout mechanics.
{{cite:kohavi2009}} is the practitioner reference.

## 11. Exercises

**Beginner**

1. List the eight design decisions that should precede an experiment.
2. Compute the sample size for a 12% baseline and a 1pp MDE.
3. Why must the randomisation unit equal the analysis unit?
4. Give three guardrail metrics for a checkout redesign.
5. What does an SRM indicate, and what should you do?

**Intermediate**

6. Explain why peeking inflates the false-positive rate, referring to the
   stopping rule rather than the individual tests.
7. An experiment randomised by user is analysed per event, with 15 events per
   user and $\rho = 0.2$. By what factor is the interval too narrow?
8. A result is "not significant" with a CI of [−9%, +10%]. What can you
   conclude?
9. Using {{eq:cuped}}, compute the variance reduction at $\rho = 0.6$ and the
   corresponding saving in required traffic.
10. Design an experiment for a two-sided marketplace. What breaks, and what do
    you do instead?
11. Your test ran Monday to Thursday and won. What is your concern?

**Advanced**

12. Derive {{eq:ab-sample-size}} from the power requirement.
13. Show that CUPED is unbiased, and explain precisely why $X$ must be measured
    pre-treatment.
14. Simulate a random walk under the null and show the crossing probability
    approaches 1 as the horizon grows.
15. Design a sequential test with an alpha-spending function and verify by
    simulation that it controls the overall error rate.
16. Explain how interference biases a standard A/B estimate in a marketplace,
    and in which direction.

**Implementation**

17. Build an experiment analyser producing the estimate, CI, p-value, SRM check
    and pre-period balance check in one report.
18. Implement CUPED and measure the achieved variance reduction against
    $1 - \rho^{2}$ across a range of $\rho$.
19. Simulate the peeking experiment with a sequential boundary and confirm the
    false-positive rate returns to 5%.
20. Write a power calculator that takes historical variance and traffic and
    returns the detectable effect for a given duration.

**Reasoning**

21. Most tested ideas do not work. What does that imply about how a team should
    allocate effort between generating ideas and testing them?
22. A stakeholder wants to stop an experiment early because it is "clearly
    winning". Explain the problem in terms they will accept, and propose an
    alternative.

## 12. Chapter Summary

An A/B test is specified before it runs: hypothesis, primary metric, guardrails,
minimum detectable effect, randomisation unit, sample size, duration, and
analysis plan. Choosing the analysis after seeing the data converts a test into
a selection procedure.

The randomisation unit must equal the analysis unit. Randomising by user and
analysing per event applies the design effect, understating the standard error
by $\sqrt{1 + (\bar{m}-1)\rho}$ — frequently a factor of two or more, which
turns a null result into a significant one.

Sample size scales as $\sigma^{2}/\delta^{2}$, so halving the detectable effect
quadruples the cost. Computing it first tells you whether the question is
answerable with available traffic.

Three sanity checks gate the result: sample ratio mismatch, pre-period balance,
and an A/A test. A failure means the experiment is broken and the result must be
discarded, not interpreted.

Peeking inflates the false-positive rate several-fold, because the stopping rule
selects for the noisiest moment. Under continuous monitoring with a fixed
threshold the error rate tends to one. Fix the sample size, or use a sequential
method designed for repeated looks.

Guardrail metrics catch wins bought at unmeasured cost. Report effects as
intervals rather than p-values, and report null results as intervals too — "no
significant difference" conflates ruling out a small effect with having learned
nothing.

CUPED reduces variance by $1 - \rho^{2}$ using pre-experiment data, at no cost
in bias, because a pre-treatment covariate cannot be affected by the assignment.
At $\rho = 0.7$ that halves the traffic required.
