---
id: aids-automl
number: 180
part: XX
tier: full
status: draft
requires: [pipeline-fails-at-the-weakest-verifier, holdout-beats-correction,
           more-exploration-finds-only-noise, verifier-argmax-gaming]
provides: [leakage-inverts-the-verifier, guards-cost-the-metric,
           ceiling-is-the-detector, selection-optimism-grows-with-search,
           noise-selects-worse, search-must-be-scored-off-search]
citations: [chan2024mlebench, huang2024dacode, testini2025dsautomation,
            lu2024aiscientist, brown2024monkeys]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why automated feature
engineering is structurally a search for leakage; state why a leakage guard makes a
reported metric worse and a model better, and why that matters organisationally;
use a pre-declared performance ceiling as a leakage detector; quantify how much of
an AutoML run's apparent gain is selection optimism; explain why a noisier
validation estimate selects a *worse* configuration and not merely a
worse-measured one; and state the single rule that unifies this chapter with the
last.

## 2. Why This Matters

{{ch:aids-stack}} found the model stage carrying the pipeline's strongest verifier
— a held-out score is a real number — and a check there worth more than a check
anywhere else. This chapter is about the two ways that verifier fails, and both are
consequences of the fact that automation *searches*.

The first is leakage. A feature that carries information unavailable at prediction
time does not lower the validation score; it raises it, and the more it leaks the
higher it goes. Automated feature engineering ranks candidates by validation lift.
**So automated feature engineering is, structurally, a search for leakage**
({{eq:leakage-inverts-the-verifier}}) — not through any defect, but by doing
precisely what it was asked.

{{sec:9-practical-example}} measures the consequence: a greedy search at ten
features scores $0.999$ on validation and $0.620$ deployed, which is the baseline.
It learned nothing. Against random feature selection the search **wins by $0.063$
on the reported number and loses by $0.177$ on the deployed one.**

And the fix has a property that explains its rarity. Installing a leakage guard
takes deployed performance from $0.620$ to $0.822$ and the *reported* score from
$0.999$ to $0.822$ ({{eq:guards-cost-the-metric}}). **A leakage guard makes your
number worse and your model better**, so the person who installs it has to explain
why the metric fell.

The second failure is selection optimism. An AutoML run trains $N$ configurations
and reports the best validation score, which is a maximum over $N$ noisy draws.
{{sec:9-practical-example}} finds that **about $34\%$ of the apparent gain from
searching is real** and the rest is the maximum's upward bias
({{eq:selection-optimism-grows-with-search}}) — and that a careful team trying
twenty configurations and an exhaustive team trying two thousand report scores
$0.054$ apart when their models differ by $0.018$.

That is {{ch:aids-agentic-eda}}'s denominator problem, at the model stage, with the
same resolution.

## 3. Prerequisites

{{ch:aids-stack}}'s {{eq:pipeline-fails-at-the-weakest-verifier}} — this chapter
examines the strong-verifier stage and finds the strength conditional.

{{ch:aids-agentic-eda}}'s {{eq:more-exploration-finds-only-noise}} and
{{eq:holdout-beats-correction}}, of which this chapter's second half is the direct
analogue.

{{ch:rsn-test-time-compute}}'s verifier-argmax result, which is this chapter's
mechanism in its original setting: selecting the argmax of an imperfect scorer
selects for the scorer's errors.

Familiarity with cross-validation and hyperparameter search is assumed.

## 4. Intuitive Explanation

A model is trained to predict whether a customer will churn. Among the available
columns is `cancellation_reason`, populated when a customer cancels.

The model finds it immediately. Validation accuracy is $99\%$. It ships, and
predicts nothing useful, because at prediction time the field is empty for everyone
who has not yet churned.

That is leakage, and everyone knows about it. What is less discussed is what
happens when you automate the search for features.

An automated feature engineering system generates candidates — ratios, aggregates,
lags, encodings — and keeps the ones that improve validation. It has no notion of
what will be available at prediction time; it has a scoring function, and it
maximises it. A leaking feature maximises it better than any honest feature can,
because a leaking feature contains the answer.

**So the search finds the leaks first.** Not occasionally: first, by construction,
because they rank highest on the criterion the search uses.
{{sec:9-practical-example}} shows a greedy search consuming every available leak
before touching a single honest feature.

The result is a model that validates at $0.999$ and deploys at the baseline. And it
looks like the best model anyone has built.

Now consider the fix. Leakage guards exist — check whether a feature's values are
available at the prediction timestamp, check whether a feature is suspiciously
predictive, check whether an aggregate spans the target period. They work.

They also lower your validation score, because they remove the features that were
inflating it. A team that installs a leakage guard watches its headline metric fall
and has to argue that the model got better. **Every incentive in an organisation
that reports validation scores points away from installing the guard.**

There is one detector that costs nothing, and it requires a judgement made in
advance. Before running the search, write down the score that would be *too good*.
Churn is not $99\%$ predictable; if you see $99\%$, something is wrong with the
pipeline, not right with the model. {{sec:9-practical-example}} makes this concrete:
the leaking model reports a score no honest model on that problem could reach.

Then the second problem, which has nothing to do with leakage and everything to do
with searching.

An AutoML run trains two thousand configurations and reports the best validation
score. That number is the maximum of two thousand noisy estimates of quality. Some
of the winner's advantage is real; some is that it got the luckiest validation
split.

The more configurations you try, the more the reported number is luck.
{{sec:9-practical-example}} finds optimism growing from essentially zero at one
configuration to $+0.078$ at two thousand — and, holding everything else fixed,
only about a third of the apparent improvement from searching harder is real.

This is exactly {{ch:aids-agentic-eda}}'s finding about exploration, in different
clothing. Search more, find more that is partly noise, report the maximum.

And it has the same denominator problem: two reports of "we achieved $0.89$
validation" mean different things depending on whether the team tried twenty
configurations or two thousand, and AutoML reports rarely say.

## 5. Formal Explanation

**Leakage.** Let a candidate feature set contain honest features with genuine
predictive value and leaking features whose value exists only at training time.
Write $V(\cdot)$ for the validation score and $D(\cdot)$ for deployed performance.
An honest feature $h$ satisfies $\Delta V(h) \approx \Delta D(h) > 0$; a leaking
feature $\ell$ satisfies:

$$\Delta V(\ell) \gg 0, \qquad \Delta D(\ell) = 0$$ (eq:leakage-inverts-the-verifier)

A greedy search selecting by $\Delta V$ therefore orders all leaks before all
honest features, and the selected set of size $k$ contains $\min(k, |\mathcal{L}|)$
leaks. The deployed score is a function only of the honest features selected:

$$D(k) = f\big(\max(k - |\mathcal{L}|, 0)\big)$$

which is *zero-improvement* for $k \le |\mathcal{L}|$. **The search's first
$|\mathcal{L}|$ selections do nothing at all.**

**Why the guard is unattractive.** A guard rejecting a fraction $g$ of leaks gives:

$$\frac{\partial V}{\partial g} < 0, \qquad \frac{\partial D}{\partial g} > 0$$ (eq:guards-cost-the-metric)

The two derivatives have opposite signs, so any team optimising the reported metric
is optimising against the deployed one. This is an incentive result, not a technical
one, and it is why the technical fix is insufficient on its own.

**The ceiling detector.** Let $\bar{D}$ be the best performance the problem
genuinely permits — the irreducible-error bound. Since $D \le \bar{D}$ always and
honest validation estimates $D$:

$$V > \bar{D} + \epsilon \;\Longrightarrow\; \text{leakage or contamination}$$ (eq:ceiling-is-the-detector)

**A validation score above the problem's ceiling is a fact about the pipeline, not
the model** — and it requires only that someone state $\bar{D}$ in advance, which
is a judgement rather than a computation.

**Selection optimism.** Let $N$ configurations have true qualities
$q_i \sim \mathcal{N}(\mu, \tau^2)$ and validation estimates
$v_i = q_i + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Selecting
$i^* = \arg\max_i v_i$ and reporting $v_{i^*}$:

$$\mathbb{E}[v_{i^*}] \approx \mu + \sqrt{\tau^2 + \sigma^2}\,\Phi^{-1}\!\Big(1 - \tfrac{1}{N}\Big)$$

while the winner's true quality is:

$$\mathbb{E}[q_{i^*}] \approx \mu + \frac{\tau^2}{\sqrt{\tau^2+\sigma^2}}\,\Phi^{-1}\!\Big(1 - \tfrac{1}{N}\Big)$$

so the optimism and the real-gain share are:

$$\text{optimism} \approx \frac{\sigma^2}{\sqrt{\tau^2+\sigma^2}}\Phi^{-1}\!\Big(1-\tfrac1N\Big), \qquad \frac{\text{real}}{\text{apparent}} = \frac{\tau^2}{\tau^2+\sigma^2}$$ (eq:selection-optimism-grows-with-search)

**The real share is $\tau^2/(\tau^2+\sigma^2)$ — independent of $N$**, which is why
{{sec:9-practical-example}} finds it constant at $\approx 34\%$ across every search
size. It is a signal-to-noise ratio, not a search property.

**Noise selects worse, not just measures worse.** From the same expression,
$\mathbb{E}[q_{i^*}]$ is *decreasing* in $\sigma$:

$$\frac{\partial}{\partial \sigma}\mathbb{E}[q_{i^*}] < 0$$ (eq:noise-selects-worse)

As $\sigma$ grows the argmax is increasingly driven by which configuration got the
luckiest split. **A noisier validation estimate produces a worse model**, so
reducing $\sigma$ improves both the model and the honesty of its score — the only
intervention in this part with no trade-off.

**The unifying rule.** Both halves, plus {{ch:aids-agentic-eda}}, reduce to:

$$\text{report}(\text{search}) \text{ is unbiased} \iff \text{scored on data the search did not touch}$$ (eq:search-must-be-scored-off-search)

## 6. Mathematical Foundation

Three extractions.

**The real-gain share is a signal-to-noise ratio, not a search-size effect.**
{{eq:selection-optimism-grows-with-search}}'s
$\tau^2/(\tau^2+\sigma^2)$ contains no $N$. That is why the listing's $34\%$ holds
from five configurations to two thousand, and why "we searched less so our number
is more honest" is wrong — a smaller search has less optimism in absolute terms and
exactly the same fraction of its gain is real.

**Noise enters twice with the same sign.** It inflates the report *and* degrades
the selection ({{eq:noise-selects-worse}}). Most quantities in this book trade
against something; a better validation estimate does not.

**The guard's sign flip is an incentive result.**
{{eq:guards-cost-the-metric}} is trivially true and organisationally decisive. It
predicts that leakage guards are installed by teams that measure deployed
performance and not by teams that measure validation, which is testable and matches
what practitioners report.

## 7. Internal Mechanics

### 7.1 Why the search finds the leaks first

```mermaid {#fig:leak-search caption="A greedy feature search ranked by validation lift. Leaking features rank highest by construction, so they are selected before any honest feature."}
flowchart TD
    P[candidate feature pool] --> R[rank by validation lift]
    R --> L1["leak: cancellation_reason"]
    R --> L2["leak: post-outcome aggregate"]
    R --> L3["leak: row id correlated with label"]
    R --> H1["honest: tenure"]
    R --> H2["honest: usage trend"]
    L1 --> S[selected set]
    L2 --> S
    L3 --> S
    H1 -.->|only if k exceeds the leak count| S
```

The dotted edge is the point. Below a selection size equal to the number of
available leaks, **no honest feature is selected at all** — and
{{sec:9-practical-example}} finds the validation-deployment gap therefore
*narrowing* as more features are added, because the leaks run out.

That inverts the usual intuition. A leakage problem looks worst at small feature
counts, and a team that responds to a suspiciously good small model by adding
features will see the gap close and conclude the problem was solved.

### 7.2 Leakage guards that actually work

Four mechanisms, in decreasing order of reliability.

**Temporal availability checks.** For every feature, assert that its value is
computable from data available at the prediction timestamp. This is the strongest
guard and it requires the data to carry timestamps, which is a data-engineering
prerequisite rather than a modelling one.

**Point-in-time joins.** Construct the training set by asking what was known at each
prediction moment, rather than by joining current tables. This makes temporal
leakage structurally impossible instead of detectable — {{ch:ag-security}}'s
containment argument in a feature store.

**Suspicious-predictiveness flags.** A single feature with implausibly high mutual
information with the target is a candidate leak. Weak, because "implausible"
requires the ceiling judgement, but cheap.

**Train-test provenance separation.** Any aggregate computed over the full dataset
leaks the test set into training. Computing all aggregates within the training fold
only is mechanical and routinely omitted.

### 7.3 Declaring the ceiling in advance

{{eq:ceiling-is-the-detector}} needs $\bar{D}$, and the useful part of this
recommendation is that stating it is a *forecast*, made before the result is known.

Reasonable sources: the performance of a human doing the same task, the performance
of the existing production system, the theoretical bound implied by known
irreducible variation, or simply the number above which you would be surprised.

Any of them is better than none, because the failure mode this catches — a model
that validates far above what the problem permits — is one that otherwise gets
celebrated. **Writing down what would be too good, before running the search, costs
one sentence and converts the pipeline's worst failure into an announced one.**

It has the same structure as {{ch:as-long-running}}'s premise recording: a
statement made in advance that turns a silent outcome into a checkable one.

### 7.4 The three-way split, and why two is not enough

Standard practice splits train and validation, searches on validation, reports the
validation score. {{eq:selection-optimism-grows-with-search}} says that report is
biased.

The fix is a third split, touched once. Search on validation as before; the winning
configuration is scored on a test set the search never saw; that score is what gets
reported.

Three properties are worth stating precisely.

**It does not improve the model.** The same configuration wins either way. The
holdout corrects the number, which is what was wrong.

**It needs no denominator.** Unlike a correction, it works whether the search tried
twenty configurations or twenty thousand — which is
{{ch:aids-agentic-eda}}'s decisive argument arriving unchanged.

**It is spent by reuse.** Score twice on the test set and choose the better and it
is a validation set again. This is the discipline that fails in practice, and it
fails invisibly.

### 7.5 Reducing validation noise is the free lunch

{{eq:noise-selects-worse}} says $\sigma$ hurts twice, which makes reducing it
unusually attractive. The available means:

**More folds.** $k$-fold cross-validation reduces $\sigma$ roughly as
$1/\sqrt{k}$ at $k$ times the compute — a direct and usually affordable trade.

**Repeated splits.** Several random partitions, averaged, which also exposes
split-to-split variance as a diagnostic.

**Stratification and grouping.** Ensuring folds respect the structure that matters
— time order, customer identity, site — which often reduces $\sigma$ more than
adding folds and prevents a leakage mode at the same time.

**Larger validation sets.** Cheap when data is plentiful and the usual constraint
when it is not.

The practical framing: **an AutoML budget spent on a bigger search buys about a
third of what it appears to; the same budget spent on a less noisy validation
estimate buys a better model and an honest number.**

### 7.6 What {{cite:chan2024mlebench}} actually showed

The chapter's results bear on how to read the best public measurement of agent
modelling capability.

{{cite:chan2024mlebench}} ran seventy-five Kaggle competitions with human baselines
from the real public leaderboards, and its best configuration medalled in $16.9\%$.
Two design choices make it unusually trustworthy here.

**Kaggle grades on a held-out set the competitor never sees**, which is exactly
{{eq:search-must-be-scored-off-search}}. So the $16.9\%$ is not a
selection-optimistic number, and it is directly comparable to what humans achieved
under the same rule.

**It reported contamination checks and resource scaling**, which most agent
benchmarks omit.

Its most transferable finding is arguably neither the medal rate nor the model
comparison but that **scaffolding mattered as much as the model** — which is
{{ch:as-single-agent}}'s components result, and consistent with this chapter's
claim that the failure modes are pipeline properties rather than model
capabilities.

### 7.7 Both failures are one failure

It is worth collapsing this chapter's two halves, because they look like separate
problems and are the same one seen at two scales.

Leakage: a search ranks candidates by a scorer, and the scorer is imperfect in a
specific direction — it credits information that will not exist later. The search
finds the scorer's blind spot because that is where the scorer's values are
highest.

Selection optimism: a search ranks candidates by a scorer, and the scorer is
imperfect in a random direction — it carries noise. The search finds the noise's
extreme because that is where the scorer's values are highest.

**Both are "the argmax of an imperfect scorer selects for the scorer's errors",**
which is {{ch:rsn-test-time-compute}}'s result, and which
{{ch:aids-agentic-eda}} met as exploration selecting for surprisingness. The
difference between the two halves is only whether the scorer's imperfection is
systematic or stochastic — and that difference determines the fix.

A *systematic* imperfection is not corrected by a fresh sample, because the leak
leaks on the holdout too if the holdout was built the same way. It needs the error
removed at the source: point-in-time construction, so the information is not there
to be found.

A *stochastic* imperfection is corrected by a fresh sample exactly, because the
noise on the new data is independent of the noise that drove the selection. That is
why a final holdout fixes selection optimism completely and does nothing for
leakage.

Which is the practical distinction this chapter turns on and it is easy to get
wrong. Teams that install a three-way split often believe they have addressed
leakage as well, and they have not. **The holdout fixes the number; only the
construction fixes the model.**

## 8. Implementation

Two listings. The first measures what a validation-maximising feature search
selects. The second measures how much of a search's reported gain is real.

```python {tier=A name=leakage-inverts-the-verifier}
"""Leakage, which is the one error a strong verifier rewards.

ch:aids-stack found the model stage to have the pipeline's best verifier: a
held-out score is a real number, and a check there was worth more than a check
anywhere else. This listing is about the error class that turns that verifier
around.

A leaking feature carries information about the target that will not exist at
prediction time -- a field populated after the outcome, an identifier correlated
with how rows were collected, an aggregate computed over the full dataset
including the future. It does not make the validation score worse. It makes it
BETTER, and the more it leaks the better it gets (eq:leakage-inverts-the-verifier).

Automated feature engineering searches feature space for whatever raises validation
score. So it is, structurally, a search for leakage.
"""
import numpy as np

rng = np.random.default_rng(4691)

M = 4000                # experiments
N_HONEST = 40           # candidate features with genuine signal available
N_LEAKY = 12            # candidate features that leak
BASE = 0.62             # score with no features
CEILING = 0.88          # the best an honest model can do on this problem
HONEST_DECAY = 0.86     # diminishing returns on honest features
LEAK_DECAY = 0.55       # leaks close the gap to a perfect score fast


def build(n_selected, leak_available=N_LEAKY, m=M, honest=N_HONEST,
          base=BASE, ceiling=CEILING, guard=0.0, greedy=True):
    """Select `n_selected` features and return (validation, deployed, leaks used).

    Honest features move the score toward `ceiling` with diminishing returns --
    that ceiling is what the problem actually permits. Leaking features move the
    VALIDATION score toward a perfect 1.0 and contribute nothing on deployment,
    which is what makes them attractive to a search and useless in production.

    A greedy search ranks candidates by validation lift, so it takes every
    available leak before any honest feature. `guard` is the share of leaking
    features a leakage check rejects before selection.
    """
    surviving_leaks = rng.binomial(leak_available, 1.0 - guard, m)
    if greedy:
        n_leak = np.minimum(surviving_leaks, n_selected)
    else:
        frac = leak_available / (leak_available + honest)
        n_leak = np.minimum(rng.binomial(n_selected, frac, m), surviving_leaks)
    n_honest = np.minimum(n_selected - n_leak, honest)

    deploy = ceiling - (ceiling - base) * (HONEST_DECAY ** n_honest)
    val = deploy + (1.0 - deploy) * (1.0 - LEAK_DECAY ** n_leak)
    return (float(val.mean()), float(deploy.mean()), float(n_leak.mean()))


print(f"{N_HONEST} honest candidate features that move the score toward the")
print(f"problem's real ceiling of {CEILING:.2f}, and {N_LEAKY} leaking ones that")
print("move VALIDATION toward 1.00 and deployment not at all. A greedy search")
print("ranks by validation lift, so it takes the leaks first.")
print()
print(f"{'features selected':>19}{'validation':>12}{'deployed':>11}"
      f"{'gap':>8}{'leaks used':>12}")
print("-" * 62)
tab = {}
for n in (2, 5, 10, 20, 40):
    r = build(n)
    tab[n] = r
    print(f"{n:>19}{r[0]:>12.3f}{r[1]:>11.3f}{r[0] - r[1]:>8.3f}{r[2]:>12.1f}")

print()
print()
print("The same selection made at random rather than greedily -- which is what")
print("a human picking features they can explain does.")
print()
print(f"{'features selected':>19}{'greedy val':>12}{'greedy dep':>12}"
      f"{'random val':>12}{'random dep':>12}")
print("-" * 67)
cmp = {}
for n in (5, 10, 20, 40):
    g = build(n)
    r = build(n, greedy=False)
    cmp[n] = (g, r)
    print(f"{n:>19}{g[0]:>12.3f}{g[1]:>12.3f}{r[0]:>12.3f}{r[1]:>12.3f}")

print()
print()
print("The uncomfortable comparison: greedy search wins on the number that is")
print("reported and loses on the one that matters.")
print()
n = 10
g, r = cmp[n]
print(f"{'at 10 features':>26}{'validation':>13}{'deployed':>11}")
print("-" * 50)
print(f"{'greedy (by val lift)':>26}{g[0]:>13.3f}{g[1]:>11.3f}")
print(f"{'random selection':>26}{r[0]:>13.3f}{r[1]:>11.3f}")
print(f"{'difference':>26}{g[0] - r[0]:>+13.3f}{g[1] - r[1]:>+11.3f}")

print()
print()
print("What a leakage guard buys. `guard` is the share of leaking features a")
print("check rejects before selection.")
print()
print(f"{'guard strength':>16}{'validation':>12}{'deployed':>11}{'gap':>8}"
      f"{'leaks used':>12}")
print("-" * 59)
gd = {}
for g_ in (0.0, 0.5, 0.8, 0.95, 1.0):
    r = build(10, guard=g_)
    gd[g_] = r
    print(f"{g_:>16.0%}{r[0]:>12.3f}{r[1]:>11.3f}{r[0] - r[1]:>8.3f}"
          f"{r[2]:>12.1f}")

print()
print()
print("And the detection problem: validation score alone cannot distinguish a")
print("good model from a leaking one. Two systems, same reported number:")
print()
target = build(10)[0]
# Find the honest-only feature count that matches the leaky model's validation.
best_n, best_d = None, None
for n_ in range(1, N_HONEST + 1):
    v, d, _ = build(n_, guard=1.0)
    if best_n is None or abs(v - target) < abs(best_d - target):
        best_n, best_d = n_, v
clean_v, clean_d, _ = build(best_n, guard=1.0)
leak_v, leak_d, leak_n = build(10)
print(f"{'system':>34}{'validation':>13}{'deployed':>11}")
print("-" * 58)
print(f"{f'{best_n} honest features, no leaks':>34}{clean_v:>13.3f}"
      f"{clean_d:>11.3f}")
print(f"{f'10 features, {leak_n:.0f} of them leaking':>34}{leak_v:>13.3f}"
      f"{leak_d:>11.3f}")
print()
print(f"   Reported validation differs by {abs(clean_v - leak_v):.3f}.")
print(f"   Deployed performance differs by {abs(clean_d - leak_d):.3f}.")

print(f"""
The first table's gap column is the whole problem. At {10} features the model
reports {tab[10][0]:.3f} and deploys at {tab[10][1]:.3f} -- which is the baseline.
It learned nothing and validated perfectly.

Note that the gap NARROWS as more features are selected: {tab[10][0] - tab[10][1]:.3f}
at ten and {tab[40][0] - tab[40][1]:.3f} at forty. That is not the search getting
wiser. It is the leaks running out -- there are only {N_LEAKY} of them, so once
they are all taken the search has to fall back on honest features.

**A leakage problem therefore looks WORSE at small feature counts**, which is the
opposite of the usual intuition that a bigger model is riskier.

The greedy-versus-random comparison is the finding to carry.

At {10} features, ranking by validation lift scores {cmp[10][0][0]:.3f} against
random selection's {cmp[10][1][0]:.3f} -- **the search wins by
{cmp[10][0][0] - cmp[10][1][0]:+.3f} on the number that gets reported.** On
deployed performance it loses by {cmp[10][0][1] - cmp[10][1][1]:+.3f}.

Automated feature engineering ranks candidates by validation lift. Leaking features
have the highest validation lift. **So automated feature engineering is,
structurally, a search for leakage** (eq:leakage-inverts-the-verifier) -- not
because it is badly built, but because it is doing exactly what it was asked.

The guard table is the one that explains why nobody implements the fix. Going from
no leakage guard to a perfect one takes deployed performance from
{gd[0.0][1]:.3f} to {gd[1.0][1]:.3f} -- a real gain of
{gd[1.0][1] - gd[0.0][1]:+.3f} -- and takes the REPORTED score from
{gd[0.0][0]:.3f} to {gd[1.0][0]:.3f}, a loss of {gd[1.0][0] - gd[0.0][0]:+.3f}.

**A leakage guard makes your number worse and your model better.** Every incentive
in a team that reports validation scores points away from installing one, and the
person who installs it has to explain why the metric went down.

The last table gives the one signal available for free. The leaking model reports
{leak_v:.3f}; the best honest model this problem permits reports {clean_v:.3f},
because the problem's ceiling is {CEILING:.2f} and nothing honest exceeds it.

**A validation score above what the problem plausibly permits is itself the
leakage detector.** That requires knowing the ceiling, which requires someone to
have thought about how predictable the outcome actually is -- a judgement, made in
advance, of the kind ch:aids-stack said the ungradeable stages need.

The practical form: before running the search, write down the score that would be
too good. Then treat exceeding it as a finding about the pipeline rather than
about the model.""")
```

The second listing asks what the reported best score means.

```python {tier=A name=selection-optimism-grows-with-search}
"""Model selection is a search, and the winner's score is a maximum.

ch:aids-agentic-eda found that exploring more comparisons finds more spurious
patterns, and that every correction needs a count the automation does not report.
This listing is that finding at the model stage, where it has a different name and
the same structure.

An AutoML run trains N configurations and reports the best validation score. That
number is a MAXIMUM over N draws, so it is biased upward by an amount that grows
with N (eq:selection-optimism-grows-with-search). The winner is partly good and
partly lucky, and the reported score cannot separate the two.

The parallel to ch:aids-agentic-eda is exact: same denominator problem, same fix.
"""
import numpy as np

rng = np.random.default_rng(4733)

M = 5000                # AutoML runs simulated
TRUE_SPREAD = 0.020     # genuine quality differences among configurations
VAL_NOISE = 0.028       # noise on a validation estimate
BASE = 0.780


def automl(n_configs, m=M, true_spread=TRUE_SPREAD, noise=VAL_NOISE,
           base=BASE, final_holdout=False, holdout_noise=None):
    """Train n_configs, pick the best by validation, and report.

    Returns (reported score, the winner's TRUE quality, optimism, the true
    quality of the best available configuration).
    """
    true = base + rng.normal(0, true_spread, (m, n_configs))
    val = true + rng.normal(0, noise, (m, n_configs))
    pick = val.argmax(1)
    rows = np.arange(m)
    reported = val[rows, pick]
    winner_true = true[rows, pick]
    best_true = true.max(1)
    if final_holdout:
        hn = noise if holdout_noise is None else holdout_noise
        reported = winner_true + rng.normal(0, hn, m)
    return (float(reported.mean()), float(winner_true.mean()),
            float((reported - winner_true).mean()), float(best_true.mean()))


print(f"{M:,} AutoML runs. Configurations differ genuinely by about")
print(f"{TRUE_SPREAD:.3f}; a validation estimate carries {VAL_NOISE:.3f} of noise.")
print("The best validation score is reported.")
print()
print(f"{'configs tried':>15}{'reported':>11}{'winner true':>14}"
      f"{'optimism':>11}{'best available':>16}")
print("-" * 67)
tab = {}
for n in (1, 5, 25, 100, 500, 2000):
    r = automl(n)
    tab[n] = r
    print(f"{n:>15}{r[0]:>11.4f}{r[1]:>14.4f}{r[2]:>11.4f}{r[3]:>16.4f}")

print()
print()
print("What the search actually buys, separated from what it appears to buy.")
print()
print(f"{'configs tried':>15}{'apparent gain':>15}{'real gain':>12}"
      f"{'share real':>13}")
print("-" * 55)
sp = {}
for n in (5, 25, 100, 500, 2000):
    apparent = tab[n][0] - tab[1][0]
    real = tab[n][1] - tab[1][1]
    sp[n] = (apparent, real, real / apparent if apparent else 0)
    print(f"{n:>15}{apparent:>+15.4f}{real:>+12.4f}"
          f"{real / apparent if apparent else 0:>13.1%}")

print()
print()
print("Noise is what converts search into optimism. Holding the search at 100")
print("configurations and varying how noisy the validation estimate is:")
print()
print(f"{'validation noise':>18}{'reported':>11}{'winner true':>14}"
      f"{'optimism':>11}")
print("-" * 54)
nz = {}
for s in (0.004, 0.012, 0.028, 0.060):
    r = automl(100, noise=s)
    nz[s] = r
    print(f"{s:>18.3f}{r[0]:>11.4f}{r[1]:>14.4f}{r[2]:>11.4f}")

print()
print()
print("A final holdout the search never touched. It does not improve the model")
print("-- the same configuration wins -- it corrects the NUMBER.")
print()
print(f"{'configs tried':>15}{'reported, no holdout':>22}"
      f"{'reported, with holdout':>24}{'truth':>9}")
print("-" * 70)
ho = {}
for n in (25, 100, 500, 2000):
    a = automl(n)
    b = automl(n, final_holdout=True)
    ho[n] = (a[0], b[0], a[1])
    print(f"{n:>15}{a[0]:>22.4f}{b[0]:>24.4f}{a[1]:>9.4f}")

print()
print()
print("And the denominator problem, which is ch:aids-agentic-eda's exactly. Two")
print("teams report the same number and did different amounts of searching:")
print()
print(f"{'team':>28}{'configs':>10}{'reported':>11}{'actually':>11}")
print("-" * 60)
for label, n in (("careful, 20 configs", 20), ("exhaustive, 2000 configs", 2000)):
    r = automl(n)
    print(f"{label:>28}{n:>10}{r[0]:>11.4f}{r[1]:>11.4f}")
print()
a20, a2000 = automl(20), automl(2000)
print(f"   Reported scores differ by {a2000[0] - a20[0]:+.4f}.")
print(f"   True quality differs by  {a2000[1] - a20[1]:+.4f}.")
print(f"   Without the config count, the two reports are not comparable.")

print(f"""
The optimism column is the tax on searching, and it grows without bound: 
{tab[1][2]:+.4f} at one configuration and {tab[2000][2]:+.4f} at {2000}
(eq:selection-optimism-grows-with-search).

The second table separates what search buys from what it appears to buy, and the
regularity is striking. Across every search size, **about
{sp[100][2]:.0%} of the apparent gain is real** and the rest is the maximum's
upward bias. Going from one configuration to two thousand appears to buy
{sp[2000][0]:+.4f} and actually buys {sp[2000][1]:+.4f}.

That two-thirds figure is not a universal constant -- it follows from the ratio of
validation noise to genuine configuration spread -- but the SHAPE is general.
Searching harder always buys something, and always reports more than it bought.

The noise table shows what controls the ratio, and contains a second finding worth
separating out. As validation noise rises from {0.004:.3f} to {0.060:.3f}, optimism
goes from {nz[0.004][2]:+.4f} to {nz[0.060][2]:+.4f} -- expected. But the
winner-true column FALLS, from {nz[0.004][1]:.4f} to {nz[0.060][1]:.4f}.

**A noisy validation estimate does not merely inflate the reported score; it
selects a worse configuration**, because the argmax is increasingly driven by which
config got the luckiest split rather than which is best. So the cheapest
improvement to an AutoML pipeline is usually not a bigger search -- it is a less
noisy validation estimate, which improves the model AND the honesty of its score at
the same time.

That is a rare thing in this part: an intervention with no trade-off.

The holdout table is the fix and it is worth being precise about what it does. At
{2000} configurations the search reports {ho[2000][0]:.4f}; a final holdout the
search never touched reports {ho[2000][1]:.4f} against a truth of
{ho[2000][2]:.4f}.

**The holdout does not improve the model.** The same configuration wins either way.
It corrects the number, which is what was wrong.

And the last table is ch:aids-agentic-eda's denominator problem, arriving at the
model stage in a suit. A careful team trying {20} configurations reports
{a20[0]:.4f}; an exhaustive team trying {2000} reports {a2000[0]:.4f}. The reported
gap is {a2000[0] - a20[0]:+.4f} and the true gap is {a2000[1] - a20[1]:+.4f}.

**Without the configuration count, two validation scores are not comparable** --
and an AutoML report gives you the score and rarely the count. That is the same
sentence as the previous chapter's, about a different search, with the same
resolution: a holdout the search never saw needs no denominator.

Which gives this chapter and the last one a single rule.

**Any process that searches and reports its best result must be scored on data the
search did not touch.** Exploration, feature engineering and model selection are
three instances; each one currently reports a maximum and calls it an estimate.""")
```

## 9. Practical Example

The first listing offers forty honest features that move the score toward the
problem's real ceiling of $0.88$, and twelve leaking ones that move validation
toward $1.00$ and deployment not at all:

```
  features selected  validation   deployed     gap  leaks used
--------------------------------------------------------------
                  2       0.885      0.620   0.265         2.0
                 10       0.999      0.620   0.379        10.0
                 20       1.000      0.802   0.198        12.0
                 40       1.000      0.876   0.124        12.0
```

At ten features the model reports $0.999$ and deploys at the baseline — **it
learned nothing and validated perfectly.** Note the gap *narrows* with more
features, because the leaks run out; a leakage problem looks worst at small
feature counts.

Greedy against random selection:

```
            at 10 features   validation   deployed
--------------------------------------------------
      greedy (by val lift)        0.999      0.620
          random selection        0.936      0.797
                difference       +0.063     -0.177
```

**The search wins on the number that is reported and loses on the one that
matters** ({{eq:leakage-inverts-the-verifier}}) — because leaking features have the
highest validation lift, and ranking by validation lift is what the search does.

The guard:

```
  guard strength  validation   deployed     gap  leaks used
-----------------------------------------------------------
              0%       0.999      0.620   0.379        10.0
             80%       0.938      0.795   0.143         2.4
            100%       0.822      0.822   0.000         0.0
```

**A leakage guard makes your number worse and your model better**
({{eq:guards-cost-the-metric}}) — $+0.202$ deployed against $-0.177$ reported.
Every incentive in a team measured on validation points away from installing it.

And the free detector:

```
                            system   validation   deployed
----------------------------------------------------------
      40 honest features, no leaks        0.879      0.879
   10 features, 10 of them leaking        0.999      0.620
```

The leaking model reports a score no honest model on this problem can reach.
**A validation score above the problem's ceiling is a fact about the pipeline**
({{eq:ceiling-is-the-detector}}), and it costs one sentence written in advance.

The second listing runs AutoML searches:

```
  configs tried   reported   winner true   optimism  best available
-------------------------------------------------------------------
              1     0.7796        0.7801    -0.0006          0.7801
            100     0.8664        0.8094     0.0570          0.8302
           2000     0.8982        0.8202     0.0780          0.8487
```

Separating apparent from real:

```
  configs tried  apparent gain   real gain   share real
-------------------------------------------------------
              5        +0.0405     +0.0135        33.3%
            100        +0.0868     +0.0292        33.7%
           2000        +0.1186     +0.0400        33.7%
```

**About a third of the apparent gain from searching is real**, constant across
every search size ({{eq:selection-optimism-grows-with-search}}) — because the share
is $\tau^2/(\tau^2+\sigma^2)$, a signal-to-noise ratio with no $N$ in it.

Noise:

```
  validation noise   reported   winner true   optimism
------------------------------------------------------
             0.004     0.8311        0.8291     0.0019
             0.060     0.9385        0.7963     0.1423
```

The winner-true column *falls*. **A noisier validation estimate selects a worse
configuration** ({{eq:noise-selects-worse}}), not merely a worse-measured one — so
reducing it improves the model and the honesty of its score at once.

The fix:

```
  configs tried  reported, no holdout  reported, with holdout    truth
----------------------------------------------------------------------
            100                0.8662                  0.8097   0.8089
           2000                0.8982                  0.8193   0.8194
```

**The holdout does not improve the model — it corrects the number.**

And the denominator:

```
                        team   configs   reported   actually
------------------------------------------------------------
         careful, 20 configs        20     0.8441     0.8017
    exhaustive, 2000 configs      2000     0.8985     0.8196
```

Reported scores differ by $+0.054$; true quality by $+0.018$.
**Without the configuration count, two validation scores are not comparable**, and
an AutoML report gives the score and rarely the count.

## 10. Production Considerations

Write down the score that would be too good, before running the search. It is one
sentence and it catches the pipeline's worst failure.

Use point-in-time joins rather than leakage detection where the data permits — it
makes temporal leakage impossible instead of detectable.

Compute every aggregate within the training fold only. Mechanical, and routinely
omitted.

Report deployed performance, not validation, as the team's headline metric.
{{eq:guards-cost-the-metric}} says any team measured on validation is measured
against its own model quality.

Split three ways and touch the test set once. Twice and it is a validation set
again.

Report the configuration count alongside the score. Without it the number is not
comparable to anything.

Spend marginal compute on validation quality before search size — more folds,
grouped splits, larger validation sets. It is the only intervention here with no
trade-off.

And treat a suspiciously good result as a pipeline finding to investigate, not a
modelling success to announce.

## 11. Common Mistakes

**Trusting automated feature selection.** It ranks by the criterion leaks maximise.

**Measuring the team on validation score.** Guaranteed to select against leakage
guards.

**Adding features when a small model looks too good.** The gap closes and the
problem does not.

**Reporting the best-of-$N$ validation score.** It is a maximum, not an estimate.

**Comparing scores across teams without the search size.** Not comparable.

**Reusing the test set.** It becomes a validation set at the second look.

**Buying a bigger search instead of a better estimate.** A third of the apparent
return, against an intervention with no trade-off.

**Having no stated ceiling.** The free detector, unclaimed.

## 12. Failure Modes

*Leaking model in production.* Validates near-perfectly, predicts nothing, and was
the best-scoring candidate at every review.

*Guard rejected on metrics grounds.* A correct intervention refused because it
lowered the number the team is judged on.

*Optimistic launch estimate.* A model promised at the searched score and delivering
the true one, with the difference attributed to distribution shift.

*Noise-selected configuration.* The luckiest split winning, in a pipeline whose
validation was never tightened.

*Exhausted holdout.* A test set consulted repeatedly across a project until it
carries no information.

*Incomparable benchmarks.* Two systems ranked on numbers produced by searches of
different sizes.

## 13. Alternatives

**Manual feature engineering.** Slower and it selects by explainability, which is
an imperfect but real leakage filter — {{sec:9-practical-example}}'s random
selection deployed better than the greedy search.

**Feature stores with point-in-time correctness.** The structural answer: leakage
becomes impossible rather than detectable.

**Nested cross-validation.** An unbiased estimate without a separate test set, at
substantial compute cost, and the correct choice when data is scarce.

**Bounded search.** Declaring the configuration budget in advance makes the
denominator known and reportable, which is {{ch:aids-agentic-eda}}'s
pre-registration in this setting.

**Deployment-shadow evaluation.** Score candidates on live traffic without acting
on them — expensive, slow, and the only estimate that is definitionally free of
both failure modes.

## 14. Evaluation

Measure the gap between validation and deployed performance for every shipped
model. It is the direct measurement of both failures and most organisations have
the data and do not compute it.

Seed known leaks and measure what fraction your guards catch.

Report the search size with every score, and refuse comparisons that lack it.

Measure your validation noise directly — repeat the split and look at the spread.
It is the parameter that determines what your search is worth.

Track how many times your test set has been consulted on each project.

And audit models whose validation exceeded the stated ceiling, as a category.

## 15. Advanced Concepts

**Automated point-in-time correctness verification.** Checking a feature pipeline's
temporal validity mechanically, rather than trusting the author.
{{maturity:EMERGING}}.

**Optimism-corrected reporting.** Estimating and subtracting the selection bias
analytically from a search's reported score, so a held-out set is not required.
{{maturity:EXPERIMENTAL}}.

**Search under a declared budget.** Making the configuration count a first-class
reported quantity, which would restore comparability across systems.

**Ceiling estimation.** {{eq:ceiling-is-the-detector}} needs $\bar{D}$; estimating
irreducible error from data rather than from judgement is an old problem with no
general solution. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:aids-agentic-eda}}'s denominator problem is this chapter's second half exactly
— a search reporting its maximum, with the count unreported and the holdout as the
only method that does not need it.

{{ch:rsn-test-time-compute}}'s verifier-argmax finding is the underlying mechanism:
selecting the argmax of an imperfect scorer selects for the scorer's errors, which
is what both leakage and selection optimism are.

{{ch:aids-stack}}'s strongest verifier turns out to be conditional — strong against
most errors and actively inverted by one class.

{{ch:ag-security}}'s containment argument returns as point-in-time joins: make the
failure impossible rather than detectable.

{{ch:as-single-agent}}'s components result is echoed by
{{cite:chan2024mlebench}}'s finding that scaffolding mattered as much as the model.

Ahead: {{ch:aids-autonomous}} takes these failure modes to their limit, in systems
that run the whole loop and judge their own output.

## 17. Exercises

1. Derive the real-gain share from
   {{eq:selection-optimism-grows-with-search}} and confirm it is independent of
   $N$.

2. Implement a temporal availability check on a feature pipeline you have and count
   what it rejects.

3. Add a partial leakage class — features that leak some information — and find
   where the greedy search stops preferring them.

4. Compare marginal compute spent on folds against marginal compute spent on
   configurations, at a fixed budget.

5. Model test-set reuse: score $k$ times and pick the best. How many looks before
   the holdout is exhausted?

6. Estimate a ceiling for a problem you work on and check your historical models
   against it.

## 18. Interview Questions

1. Why is automated feature engineering a search for leakage?

2. You install a leakage guard and validation drops four points. What do you tell
   your manager?

3. Your churn model validates at $99\%$. What do you do first?

4. AutoML tried two thousand configurations and reports $0.90$. What is the model
   worth?

5. You have budget for either a larger search or more cross-validation folds. Which
   and why?

6. Two teams report $0.89$. What do you need to know to compare them?

## 19. Research Questions

1. Can point-in-time correctness be verified automatically for arbitrary feature
   pipelines?

2. Can selection optimism be corrected analytically well enough to replace a
   holdout?

3. Would reporting search size change behaviour if it became conventional?

4. Can irreducible error be estimated well enough to make the ceiling detector
   automatic?

5. How much of the reported year-on-year improvement in applied modelling is
   selection optimism from growing search budgets?

## 20. Chapter Summary

The model stage has the pipeline's strongest verifier and two ways it fails, both
consequences of automation searching.

**Leakage inverts it.** A leaking feature raises validation and contributes nothing
deployed, so a search ranking by validation lift takes every leak before any honest
feature ({{eq:leakage-inverts-the-verifier}}). At ten features the greedy search
scored $0.999$ validating and $0.620$ deployed — the baseline. Against random
selection it won by $0.063$ on the reported number and lost by $0.177$ on the real
one.

The fix is unattractive by construction: a leakage guard moved deployed performance
$+0.202$ and the reported score $-0.177$ ({{eq:guards-cost-the-metric}}). **A guard
makes your number worse and your model better**, so any team measured on validation
is measured against its own model quality. The free detector is a ceiling stated in
advance — a score above what the problem permits is a fact about the pipeline
({{eq:ceiling-is-the-detector}}).

**Selection optimism inflates it.** The best-of-$N$ validation score is a maximum
over noisy draws, and **about $34\%$ of the apparent gain from searching is real**,
constant from five configurations to two thousand, because the share is
$\tau^2/(\tau^2+\sigma^2)$ with no $N$ in it
({{eq:selection-optimism-grows-with-search}}).

Noise enters twice with the same sign: it inflates the report *and* degrades the
selection, with the winner's true quality falling from $0.829$ to $0.796$ as
validation noise rose ({{eq:noise-selects-worse}}). So **spending compute on a less
noisy validation estimate beats spending it on a bigger search** — the only
intervention in this part with no trade-off.

A final holdout corrects the number without changing the model, and needs no
knowledge of the search size. Which unifies this chapter with the last into one
rule: **any process that searches and reports its best result must be scored on
data the search did not touch**
({{eq:search-must-be-scored-off-search}}). Exploration, feature engineering and
model selection are three instances, and each currently reports a maximum and calls
it an estimate.

## 21. Further Reading

{{cite:chan2024mlebench}} is the best public measurement here, and its design is
why: Kaggle grades on a held-out set the competitor never sees, so its $16.9\%$
medal rate is free of selection optimism and directly comparable to human
performance under the same rule. Its scaffolding finding is the more transferable
result.

{{cite:brown2024monkeys}} for what repeated sampling buys and what selects among
the samples — the same argmax structure this chapter finds at the configuration
level.

{{cite:huang2024dacode}} for modelling as one stage of an agent task, and
{{cite:testini2025dsautomation}} for why this stage is the best-evaluated one.

{{ch:rsn-test-time-compute}} for the verifier-argmax mechanism underneath both
failures, and {{ch:aids-agentic-eda}} for the denominator problem this chapter
inherits.
