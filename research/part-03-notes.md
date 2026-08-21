# Part III — Data Science: research notes

Research pass run 2026-08-13, before writing.

## The question this part has to answer honestly

The book's specification asks Part III to "explain how modern AI is changing
Data Science". That is the one genuinely current question here, and it needs
care, because both available answers are wrong.

The **dismissive** answer — agents write boilerplate, nothing fundamental has
changed — is contradicted by what tools now do. The **credulous** answer —
data science is automated, the role is over — is contradicted by what they
still get wrong.

What the research actually supports:

- LLM agents now handle a substantial share of the *mechanical* work: writing
  SQL from a description, generating first-pass EDA, proposing feature
  transformations, sweeping model families. Reported gains are real and large
  on those specific tasks.
- Agentic frameworks for end-to-end data science exist and are being
  benchmarked — `automind2025` is one concrete, verifiable example, combining a
  curated knowledge base, tree search over solutions, and adaptive coding.
  {{maturity:EMERGING}}
- What has *not* been automated is the part that determines whether the answer
  is right: deciding what question to ask, knowing how the data was generated,
  recognising that a correlation is confounded, noticing that a join inflated
  the row count, and judging whether an effect is worth acting on.

The honest framing, and the one Part III adopts: **AI has automated the
execution of data science far faster than it has automated the judgement.** The
tasks that got cheap are exactly the ones that were never the hard part. That is
stated in {{ch:ds-what-it-is}} and revisited concretely in {{ch:ds-eda}} and
{{ch:ds-feature-eng}}, and it is the thesis {{part:20}} develops properly.

> One consequence for the chapters: the failure modes matter *more* than they
> used to, not less. An agent that generates a plausible-looking join in two
> seconds makes the row-count check more valuable, not redundant. Part III is
> therefore written around what goes wrong, on the grounds that recognising a
> wrong answer is now the scarcer skill.

## Structural decisions

**Causation gets a full chapter** ({{ch:ds-causation}}) rather than a section.
It is the single most common way a data-science conclusion is wrong, and it is
where {{ch:math-covariance}}'s correlation material has to be paid for. Simpson's
paradox is derived numerically rather than described.

**Leakage gets a full chapter** ({{ch:ds-leakage}}) for the same reason. It is
the most common way a model that looked excellent in development fails in
production, and `kaufman2012` is the reference treatment.

**A/B testing is placed before feature engineering**, unusually. The reason is
that {{ch:ds-experiments}} is where {{ch:math-inference}}'s hypothesis testing
becomes concrete, and having that available makes the evaluation discussion in
the later chapters honest.

**Time series and recommenders each get one chapter** rather than being folded
into later parts. Both are genuinely different problem shapes — one has an
ordering constraint that breaks cross-validation, the other has a feedback loop
that breaks the i.i.d. assumption — and both are common enough to deserve
first-class treatment.

## References checked

All verified against Crossref records or official proceedings pages on
2026-08-13.

| Key | What | Checked against |
|---|---|---|
| `wickham2014` | Tidy Data, JSS 59(10) | Crossref 10.18637/jss.v059.i10 |
| `kaufman2012` | Leakage in data mining, TKDD 6(4) 1-21 | Crossref 10.1145/2382577.2382579 |
| `chawla2002` | SMOTE, JAIR 16:321-357 | Crossref 10.1613/jair.953 |
| `koren2009` | Matrix factorization, IEEE Computer 42(8) 30-37 | Crossref 10.1109/MC.2009.263 |
| `kohavi2009` | Controlled experiments on the web, DMKD 18(1) 140-181 | Crossref 10.1007/s10618-008-0114-1 |
| `simpson1951` | Interpretation of interaction in contingency tables, JRSS-B 13(2) 238-241 | Crossref 10.1111/j.2517-6161.1951.tb00088.x |
| `sculley2015` | Hidden Technical Debt in ML Systems, NIPS 2015 | NeurIPS proceedings page, full author list |
| `automind2025` | AutoMind: agent for automated data science | arXiv 2506.10974, v1 2025-06-12 |

Crossref's REST API proved the most reliable route for bibliographic metadata —
publisher sites variously paywall, redirect to authentication, or return 403,
while Crossref returns the authoritative record unauthenticated.

## Deliberately omitted

- **Big-data infrastructure.** Spark, Hadoop, warehouse architecture. Named
  where relevant; {{part:23}} is where infrastructure belongs.
- **Deep causal inference.** Instrumental variables, regression discontinuity,
  difference-in-differences, the full do-calculus. {{ch:ds-causation}} teaches
  confounding, Simpson's paradox, and why randomisation works — enough to stop
  the reader making the common error and to read further. A complete treatment
  is a book.
- **Bayesian A/B testing.** Mentioned as an alternative with its trade-off
  stated; the frequentist treatment connects directly to
  {{ch:math-inference}} and is what the reader will encounter.
- **Classical time-series theory.** ARIMA identification, Box-Jenkins, spectral
  methods. {{ch:ds-timeseries}} covers what breaks when data is ordered —
  leakage through time, non-stationarity, the validation scheme — because that
  is what transfers to the sequence models of {{part:7}}.
- **Deep recommenders.** {{ch:ds-recsys}} covers collaborative filtering and
  matrix factorisation, which is where the ideas come from and which connects
  directly to the SVD of {{ch:math-eigen}}. Two-tower and sequential
  recommenders belong after embeddings ({{part:11}}).

## Chapter-level notes

**Ch 21** must avoid being a vague overview. It is built around the actual
distribution of effort and the specific ways a data-science project fails, with
the AI question addressed head-on.

**Ch 25 (causation)** is the most important chapter in the part. Simpson's
paradox is constructed numerically so the reversal is visible, and the
randomisation argument is derived rather than asserted.

**Ch 28 (leakage)** enumerates leakage by mechanism — target, temporal,
group, preprocessing — because a taxonomy is what makes it detectable.

**Ch 29 (time series)** exists mainly to establish that the i.i.d. assumption
underlying everything in {{part:1}} and {{part:4}} fails when data is ordered,
and what to do instead.
