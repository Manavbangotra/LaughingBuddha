---
id: part-05-intro
status: final
---

## What this part is for

Everything so far has been about building a model that scores well. This part
is about the distance between that and a model that keeps working, and the
first thing to say is that the distance is mostly not about modelling.

{{cite:sculley2015}} put it as a picture: in a real machine learning system,
the box containing the model code is small, and it is surrounded by much larger
boxes for configuration, data collection, feature extraction, verification,
serving infrastructure, process management and monitoring. Eleven years on that
observation has aged better than almost anything else written about the field.
The maintenance cost lives in the surrounding boxes, and so do the outages.

```text
   ┌────────────┬──────────────┬─────────────┬───────────────┐
   │  config    │   data       │  feature    │  verification │
   │            │  collection  │  extraction │               │
   ├────────────┼──────┬───────┴─────────────┼───────────────┤
   │  process   │      │  ML   │             │   analysis    │
   │ management │      │ code  │             │     tools     │
   ├────────────┼──────┴───────┬─────────────┼───────────────┤
   │  machine   │   serving    │  monitoring │               │
   │  resources │ infrastructure│            │               │
   └────────────┴──────────────┴─────────────┴───────────────┘
        the box in the middle is Parts I-IV.
        this part is about the rest of the diagram.
```

The vocabulary that paper introduced — **glue code**, **pipeline jungles**,
**configuration debt**, **undeclared consumers**, **correction cascades**, and
**entanglement**, its CACE principle that *changing anything changes
everything* — is still the most useful vocabulary available for describing why
ML systems rot. It recurs throughout this part, and it is worth learning as
vocabulary rather than as history.

## What is here

- **Chapters 43–44** — getting an honest number, and searching efficiently for
  a better one. {{ch:mle-splits}} is about splitting as *code* rather than as a
  concept; {{ch:mle-hpo}} derives why random search beats grid search and then
  measures what actually accounts for modern speed-ups.
- **Chapters 45–46** — the two disciplines that make a result survive contact
  with time. {{ch:mle-pipelines}} is about point-in-time correctness;
  {{ch:mle-reproducibility}} is about being able to recreate a run at all.
- **Chapters 47–48** — the handoff and the aftermath. {{ch:mle-registry}} is
  what a deployable artefact must carry with it; {{ch:mle-drift}} is how you
  find out it has stopped working.

```mermaid {#fig:part5-deps caption="Dependencies within Part V. The two halves are separated by the deployment boundary: 43-46 are about producing a trustworthy artefact, 47-48 about what happens once someone else depends on it."}
graph LR
  C43[43 · Splits] --> C44[44 · HPO]
  C43 --> C45[45 · Pipelines]
  C45 --> C46[46 · Reproducibility]
  C44 --> C46
  C46 --> C47[47 · Registry]
  C45 --> C47
  C47 --> C48[48 · Drift]
  C45 --> C48
```

## Three things worth saying up front

**The failure modes here are silent.** A model that overfits gives you a bad
validation score, and you notice. A training/serving skew gives you an
excellent validation score and a model that underperforms in production by an
amount nobody can attribute. Almost everything in this part is about making a
silent failure loud, which is why {{ch:mle-drift}} is the chapter the rest
points at.

**Correctness here is a property of *code*, not of understanding.**
{{ch:ds-leakage}} explained what leakage is, and you understood it. That does
not prevent a leak, because the leak will be introduced by a join written six
months later by someone reasonable. The content of {{ch:mle-pipelines}} is
therefore mechanical: the discipline that makes the honest thing the easy thing
to write.

**The tooling churns and the problems do not.** MLflow, Weights & Biases, DVC,
Feast, Evidently and their successors are named where naming them is useful and
never compared feature by feature — that content dates within a year. What does
not date is the taxonomy: the four causes of training/serving skew, the
distinction between tracking and reproducibility, the three kinds of
distribution shift, and why label delay decides which drift detector you are
allowed to use. Learn the taxonomy; the tools are lookups.

## The one genuinely live question

Most of this part is settled practice. The exception is drift detection, and
the reason is worth stating plainly, because most published advice gets it
wrong.

The classical detectors that {{cite:gama2014}} surveys — DDM, EDDM, ADWIN —
watch a running error rate and signal when it degrades. They are excellent, and
they require labels to arrive promptly. In production they usually do not: a
fraud label waits out a chargeback window of one to six months, an insurance
claim takes months to years, a clinical outcome may take years. So the
detectors the literature is mostly about are unusable in the domains that most
need them.

What teams actually run is unsupervised input monitoring, which is a different
problem with a different reference ({{cite:rabanser2019}}) and a different
failure mode: **input drift with no measurable performance effect is a false
alarm, and false alarms are how monitoring gets switched off.**
{{ch:mle-drift}} therefore treats delayed labels as the normal case, builds the
conjunction alert that follows from it, and measures the false-alarm rate that
justifies the extra complexity.

## What this part deliberately does not cover

Orchestration mechanics — Airflow, Dagster and DAG authoring — and the platform
layer generally, which is {{part:24}}. Distributed and accelerated training,
which is {{part:23}}. AutoML beyond naming it as the limit of the
search-automation idea. Statistical process control theory: CUSUM and EWMA are
named as the classical machinery behind sequential detection and not derived.
Vendor comparisons of any kind.

## What you should be able to do at the end

Write a split that stays honest when the data changes shape, and say why a
random split would have been wrong. Choose a hyperparameter search strategy
from the shape of the budget, and explain why the pruner usually matters more
than the sampler. Name the four causes of training/serving skew and write a
point-in-time-correct join that avoids all of them. Distinguish experiment
tracking from reproducibility, and enumerate the sources of nondeterminism in
your own stack. Specify what a registry entry must contain for someone who did
not train the model to deploy it safely. Design a monitoring scheme for a
system whose labels arrive four months late, and justify every threshold in it.

The assignment at the end asks for a pipeline that is honest, reproducible and
monitored — and for the evidence that it is all three.
