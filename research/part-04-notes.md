# Part IV — Classical Machine Learning: research notes

Research pass run 2026-08-13, before writing.

## The one genuinely live question

Most of Part IV is settled material. Linear regression, logistic regression,
trees, bagging, boosting, SVMs, k-means and PCA have not changed. What *has*
changed, and changed recently enough to matter, is the answer to "what should I
use on a tabular dataset?"

The position that held for roughly two decades: **gradient-boosted decision
trees win on tabular data.** `grinsztajn2022` benchmarked this carefully across
45 datasets and found tree-based models still state of the art at medium size
(~10K samples), identifying three specific reasons — robustness to
uninformative features, invariance to feature rotation, and the ability to learn
irregular target functions.

The position that has emerged since: **tabular foundation models are
competitive, and on small data they win.** `hollmann2025` (TabPFN, published in
*Nature*) reports outperforming a heavily-tuned ensemble of the strongest
baselines on datasets up to 10,000 samples — in seconds rather than hours,
via in-context learning rather than fitting.

Both are correct, and the honest synthesis for a 2026 textbook is:

- **Small tabular data (< ~10K rows):** a tabular foundation model is now a
  serious first choice and frequently the best one. {{maturity:EMERGING}}
- **Medium to large tabular data:** gradient boosting remains the default and
  keeps winning competitions and production benchmarks.
  {{maturity:ESTABLISHED}}
- **Unstructured data:** learned representations, comprehensively — which is
  {{part:6}} onward.

This is treated in {{ch:ml-boosting}} rather than being scattered, and the
maturity labels do real work: presenting GBDT dominance as permanent would date
badly, and presenting foundation models as settled would be wrong.

> The durable content is *why* the trade-off exists — the three mechanisms
> Grinsztajn et al. identify are properties of the data, not of the year — so
> that section should survive even when the specific recommendation flips.

## Structural decisions

**Chapter 34 (metrics and bias-variance) sits early**, before most algorithms.
The reason is that every subsequent chapter needs a way to say whether a model
is good, and the bias-variance decomposition is the frame that makes the rest of
the part coherent — bagging reduces variance, boosting reduces bias, and
regularisation trades one for the other. Teaching it after the algorithms would
mean re-explaining each of them.

**Linear and logistic regression get a full chapter each.** They are the two
models whose mathematics is fully derivable at this level, and both derivations
were already prepared in Part I: least squares is the projection of
{{ch:math-vectors}}, and the logistic gradient is
{{eq:logreg-gradient}} from {{ch:math-derivatives}}. Deriving them completely
here is what makes the later chapters' hand-waving acceptable.

**Bagging and boosting are separate chapters** rather than one "ensembles"
chapter, because they do opposite things to the bias-variance decomposition and
conflating them is the single most common confusion in this material.

**PCA follows clustering**, unusually. The reason is that {{ch:math-eigen}}
already did the linear algebra, so the PCA chapter can concentrate on what it is
*for* — and one of its main uses is making clustering work in high dimensions,
which reads better after clustering has been shown to fail there.

## References checked

All verified against Crossref records, arXiv, or the publisher's own page on
2026-08-13.

| Key | What | Checked against |
|---|---|---|
| `breiman2001rf` | Random Forests, ML 45(1) 5-32 | Crossref 10.1023/A:1010933404324 |
| `breiman2001cultures` | Statistical Modeling: The Two Cultures, Statist. Sci. 16(3) | Crossref 10.1214/ss/1009213726 |
| `friedman2001` | Greedy function approximation, Ann. Statist. 29(5) | Crossref 10.1214/aos/1013203451 |
| `chen2016` | XGBoost, KDD 2016, 785-794 | Crossref 10.1145/2939672.2939785 |
| `cortes1995` | Support-vector networks, ML 20(3) 273-297 | Crossref 10.1007/BF00994018 |
| `lloyd1982` | Least squares quantization in PCM, IEEE T-IT 28(2) 129-137 | Crossref 10.1109/TIT.1982.1056489 |
| `liu2008` | Isolation Forest, ICDM 2008, 413-422 | Crossref 10.1109/ICDM.2008.17 |
| `hollmann2025` | TabPFN, Nature 637, 319-326 | Crossref 10.1038/s41586-024-08328-6 |
| `grinsztajn2022` | Why tree-based models still outperform DL on tabular | arXiv 2207.08815, v1 2022-07-18 |
| `pedregosa2011` | Scikit-learn, JMLR 12, 2825-2830 | jmlr.org official paper page |

Crossref remained the most reliable route. Note that Crossref returns DOIs
lowercased and sometimes omits page ranges (Friedman, Breiman's *Two Cultures*);
where a page range could not be confirmed from the record it is omitted from the
bibliography entry rather than filled in from memory.

## Deliberately omitted

- **Full VC theory and PAC learning.** Referenced conceptually in
  {{ch:ml-metrics}} for where generalisation bounds come from, not developed.
  The bias-variance decomposition does the work this book needs.
- **Kernel theory beyond the kernel trick.** {{ch:ml-svm}} derives the trick and
  the margin, and stops before Mercer's conditions and RKHS.
- **Manifold learning.** t-SNE and UMAP are named in {{ch:ml-pca}} with their
  main caveat — that inter-cluster distances in the embedding are not meaningful
  — and not developed. They are visualisation tools, not dimensionality
  reduction for modelling.
- **Semi-supervised and active learning.** Named in {{ch:ml-what-it-is}} as
  paradigms; the techniques belong later.
- **Bayesian methods.** Naive Bayes appears in {{ch:ml-knn-nb}}; Gaussian
  processes and Bayesian networks do not. They are a coherent body of material
  that this book does not otherwise need.
- **AutoML frameworks.** Mentioned in the boosting chapter's alternatives;
  {{part:20}} covers automated model selection with agents.

## Chapter-level notes

**Ch 32 (linear regression)** derives the normal equations from the projection
argument of {{ch:math-vectors}}, and explains why `np.linalg.lstsq` is preferred
to forming $(\mat{X}\T\mat{X})^{-1}$ — a direct application of the condition
number from {{ch:math-eigen}}.

**Ch 34 (metrics)** must do the bias-variance decomposition properly, including
the derivation, because it is the organising frame for the whole part. It also
covers calibration, which {{ch:ds-leakage}} showed is broken by resampling.

**Ch 38 (boosting)** carries the tabular-versus-deep-learning question and the
maturity labels described above.

**Ch 40 (clustering)** must be honest that clustering is unsupervised and
therefore has no ground truth, so validation is genuinely harder — and that
k-means finds spherical clusters whether or not any exist.

**Ch 42 (anomaly detection)** connects back to {{ch:ds-cleaning}}'s robust
statistics and {{ch:ds-timeseries}}'s residual approach, and adds the
high-dimensional case where distance-based methods fail
({{ch:math-vectors}}).
