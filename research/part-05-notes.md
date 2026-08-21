# Part V — Machine Learning Engineering: research notes

Research pass run 2026-08-14, before writing.

## What this part is really about

Parts I–IV taught how to build a model that scores well. Part V is about the
gap between that and a model that keeps working, and the honest framing is that
**the gap is mostly not about modelling.** {{cite:sculley2015}} is the reference
statement: the model code is a small box in the middle of a much larger system
of configuration, data collection, feature extraction, serving infrastructure
and monitoring, and the surrounding boxes are where the maintenance cost lives.

That paper is eleven years old in 2026 and its central claim has aged
extraordinarily well. The vocabulary it introduced — glue code, pipeline
jungles, configuration debt, entanglement (CACE: *changing anything changes
everything*), undeclared consumers, correction cascades — is still the most
useful vocabulary available for describing why ML systems rot.

## The three genuinely live questions

### 1. Is Bayesian optimisation worth it over random search?

Settled enough to state clearly, and more nuanced than either camp says.

{{cite:bergstra2012}} is the foundational result: random search beats grid
search, because grid search wastes trials on dimensions that do not matter. The
argument is geometric rather than empirical — with $D$ hyperparameters of which
only $d$ matter, a grid of $n$ points per axis gives only $n$ distinct values
along each important dimension, while random search gives $n^{D}$ distinct
values along each. **This is a derivation, not a benchmark**, so Chapter 44
derives it and measures it rather than citing it.

The 2026 position on what to use *instead* of random search:

- **Bandit-based early stopping** (successive halving, Hyperband
  {{cite:li2018hyperband}}, ASHA) is the biggest practical win when the model
  produces intermediate scores — epochs, boosting rounds, partial fits. It
  buys a large constant factor by killing bad configurations early rather than
  by choosing better ones.
- **TPE / Bayesian** ({{cite:akiba2019}}, Optuna's default) helps most in
  low-dimensional, expensive-evaluation regimes. It helps least when the
  budget is small (the surrogate has nothing to fit), when the dimension is
  high, or when evaluations are cheap enough to just do more of them.
- The two compose, and Optuna's default combination — TPE sampler plus a
  median or successive-halving pruner — is the sensible 2026 default.

The honest summary Chapter 44 should give: **the pruner usually matters more
than the sampler.** Most published gains attributed to "Bayesian optimisation"
are gains from early stopping.

### 2. Did feature stores solve training/serving skew?

Partly, and the honest answer is more useful than either the vendor pitch or
the backlash.

Adoption is real — roughly 45% of teams by 2026, up from 15–20% in 2020 — and
the mechanism is sound: if the offline and online paths execute *the same*
feature definition, code divergence cannot produce skew.

But skew has more than one cause, and only one of them is code divergence. The
others survive a feature store intact:

- **Time-travel errors.** Training joins features "as of" a label timestamp;
  serving reads the current value. Getting point-in-time correctness right is
  the hard part, and it is a property of the *join*, not of where the code
  lives.
- **Freshness gaps.** A feature computed hourly in batch and read in real time
  is a different number at serve time from the one training saw.
- **Availability differences.** A feature derivable in a warehouse may not be
  computable within the serving latency budget.

The 2026 architectural drift is towards solving this at the query layer —
streaming SQL with incremental materialised views — rather than by moving
feature artefacts between stores.

**Position for the book:** {{maturity:MATURE}} for the pattern, and the durable
content is the *taxonomy of skew causes*, which will outlive any particular
tool. Chapter 45 teaches point-in-time correctness as the hard part and treats
the feature store as one way to get it.

### 3. Drift detection under label delay

This is the genuinely live one, and it is where most monitoring advice is
wrong.

The classical detectors — DDM, EDDM, ADWIN, and most of what
{{cite:gama2014}} surveys — are **supervised**: they watch the error rate and
signal when it degrades. They are excellent and they require labels to arrive
promptly.

In production, labels usually do not. Fraud labels arrive after a chargeback
window of 30–180 days. Insurance claims take months to years. Clinical
outcomes may take years. So the detectors the literature is mostly about are
inapplicable in the domains that most need them, and what teams actually run is
unsupervised input monitoring.

{{cite:rabanser2019}} is the right citation for that regime: it compares
methods for detecting dataset shift *without* labels, and its finding —
that a dimensionality reduction followed by a two-sample test on the reduced
representation is a strong general approach, and that univariate tests with
multiple-testing correction are a surprisingly strong baseline — is directly
actionable.

The 2026 operational pattern worth teaching, because it is what stops
monitoring from being ignored:

> **Alert on the conjunction, not on drift alone.** Input drift with no
> measurable performance effect is a false alarm, and false alarms are how
> monitoring gets switched off. Where labels are delayed, use a proxy metric
> that is available immediately — prediction distribution, confidence, accepted
> rate, downstream conversion — and require both signals before paging.

**Position for the book:** input drift detection is {{maturity:ESTABLISHED}};
the practice of alerting on drift alone is a known failure mode; treating
delayed labels as the normal case rather than the exception is the framing
Chapter 48 should adopt.

## Structural decisions

**Chapter 43 revisits splits, deliberately.** {{ch:ds-leakage}} covered leakage
in features and {{ch:ml-metrics}} covered selection optimism. Chapter 43's job
is different: the *engineering* of an honest split — group and time-aware
splitting as code, nested CV as a pipeline rather than a concept, and the
discipline that keeps a test set untouched across a team and across months. It
should be short on theory and long on the mechanics that go wrong.

**Chapter 44 derives the random-search result** rather than citing it, because
the geometric argument is three lines and far more convincing than a benchmark.
It then measures the pruner-versus-sampler question directly.

**Chapter 45 is about point-in-time correctness**, not about tools. The
`fit`/`transform` discipline from {{cite:pedregosa2011}} is the small-scale
version of the same idea, and the chapter should draw that line explicitly.

**Chapter 46 separates reproducibility from experiment tracking.** They are
routinely conflated and they solve different problems: tracking is about being
able to *compare* runs, reproducibility is about being able to *recreate* one.
A team can have excellent tracking and zero reproducibility.

**Chapter 47 is the handoff**, and the honest content is organisational as much
as technical: what a registry entry must contain for someone who did not train
the model to deploy it safely, and what "approved" has to mean.

**Chapter 48 carries the label-delay framing** and pays off the forward
references from {{ch:ml-trees}} (silent extrapolation) and {{ch:ml-anomaly}}
(the multivariate monitor).

## References checked

All verified 2026-08-14 against Crossref, the publisher's own page, or arXiv.

| Key | What | Checked against |
|---|---|---|
| `bergstra2012` | Random Search for Hyper-Parameter Optimization, JMLR 13, 281-305 | jmlr.org/papers/v13/bergstra12a.html |
| `li2018hyperband` | Hyperband, JMLR 18, article 185, 1-52 | jmlr.org/papers/v18/16-558.html |
| `akiba2019` | Optuna, KDD 2019, 2623-2631 | Crossref 10.1145/3292500.3330701 |
| `sculley2015` | Hidden Technical Debt in ML Systems, NIPS 28 (2015) | papers.nips.cc official abstract page |
| `breck2017` | The ML Test Score, IEEE Big Data 2017, 1123-1132 | Crossref 10.1109/BigData.2017.8258038 |
| `gama2014` | A survey on concept drift adaptation, ACM CSUR 46 | Crossref 10.1145/2523813 |
| `rabanser2019` | Failing Loudly, arXiv 1810.11953, NeurIPS 2019 | arxiv.org/abs/1810.11953 |

Notes on what could NOT be confirmed and is therefore omitted:

- Crossref's record for `gama2014` gives volume 46 and pages 1-37 but no issue
  number; the commonly-cited "46(4), Article 44" is not in the record, so the
  bibliography entry states volume and pages only.
- JMLR papers are not reliably indexed in Crossref; both were verified from
  JMLR's own paper pages instead, as `pedregosa2011` was in Part IV.
- NIPS 2015 proceedings have no DOI; `sculley2015` is verified from the
  official proceedings page and carries no DOI field.

## Deliberately omitted

- **Specific vendor comparisons.** MLflow versus W&B versus Neptune, Feast
  versus Tecton. Named where useful, never compared feature-by-feature; that
  content dates within a year and {{part:24}} covers the platform layer.
- **Kubernetes and orchestration mechanics.** Airflow/Dagster/Prefect DAG
  authoring belongs in {{part:24}}.
- **AutoML frameworks.** Named in {{ch:mle-hpo}} as the upper bound of the
  search-automation idea; not developed.
- **Distributed training.** {{part:23}}.
- **Statistical process control theory.** CUSUM and EWMA are named in
  {{ch:mle-drift}} as the classical machinery behind sequential detection, and
  not derived.
- **Data versioning tool internals.** DVC's content-addressed storage is
  described conceptually in {{ch:mle-reproducibility}}; the implementation is
  not.

## Chapter-level notes

**Ch 43** must make the point that a split is *code*, and that the failure mode
is a split which was honest when written and silently stopped being honest when
the data changed shape. Cross-validation with groups and time is the concrete
content.

**Ch 44** should measure: grid versus random at fixed budget with unequal
hyperparameter importance (the {{cite:bergstra2012}} argument); successive
halving versus full evaluation; and TPE versus random at small and large
budgets, honestly reporting where TPE does not help.

**Ch 45** must demonstrate point-in-time correctness with an actual time-travel
bug, and show the leak it produces. This is the chapter's core measurement.

**Ch 46** should demonstrate a genuinely non-reproducible run and then fix it,
enumerating every source of nondeterminism found along the way.

**Ch 47** is the least code-heavy chapter in the part; its practical example
should be a registry schema and a promotion gate implemented as an actual
check, in the spirit of {{cite:breck2017}}'s rubric.

**Ch 48** must implement the conjunction alert from the section above, and must
measure the false-alarm rate of drift-only alerting to justify it.
