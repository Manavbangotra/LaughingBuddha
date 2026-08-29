# Part XXIV research notes — MLOps, LLMOps, and AgentOps

Research pass 2026-08-29. Two new citations verified; this part draws mostly on
already-verified work from Parts V, XVIII, XXI and XXII.

## New this pass

- `paleyes2020deployment` — Challenges in Deploying ML (2011.09926, 18 Nov 2020, 3
  authors; ACM Computing Surveys 2022). Survey of published case studies, mapping
  reported obstacles onto deployment-workflow stages. Framing conclusion: **practitioners
  face issues at every stage**, not one. *The standard reference for "the hard part is
  not the model."* Note: the abstract is qualitative — **no percentages to quote**, and
  chapters must not invent any.

- `deshpande2025trail` — TRAIL (2505.08638, 13 May 2025, 6 authors; v3 23 Jun 2025).
  **148 human-annotated traces**, formal taxonomy of agentic error types, single- and
  multi-agent, software engineering plus open-world retrieval. Headline: the best model
  tested (Gemini-2.5-pro) reached **11%** at localising the issue in a trace.
  *Turns "agent traces are hard to debug" into a number, and 11% means automated trace
  triage is not close.*

## Searched and not used

A 2026 search surfaced several observability and agent-logging preprints
(2603.27355, 2604.26152, 2605.11093, 2602.10133, 2607.07689). **None fetched or
verified** — not citable. Also examined and rejected:

- `2308.05391` (Trust in LLM-based automation agents) — fetched and verified to exist,
  but the abstract carries **no quantitative claims**. Nothing to cite it *for* in a
  book whose rule is that citations carry numbers.
- `2504.11750` (LLM inference on CPU-GPU coupled architectures) — real and quantitative,
  but it belongs to {{part:23}}, not here. Not added; noted in case a later revision of
  Part XXIII wants it.

## Carried in from earlier parts

| need | citation | part |
|---|---|---|
| hidden technical debt, glue code, config debt | `sculley2015` | V |
| production readiness rubric | `breck2017` | V |
| concept drift taxonomy and adaptation | `gama2014` | V |
| agent failure taxonomy | `cemri2025mast` | XVIII |
| developer productivity, self-report gap | `becker2025devproductivity` | XXI |
| protocol, correlation, transports | `mcp2026spec`, `hou2025mcp` | XIX |
| the second instrument, semantic error rate | Part XXII's results | XXII |

## The organising problem

Parts XXII and XXIII made the system work. This part is about **operating it over time**,
and the distinguishing fact is that the system changes without anyone changing the code.

Three sources of change, each with a different signature:

- **Data drifts.** Classical, well-studied (`gama2014`), and the one teams have tooling
  for.
- **The model changes under you.** A provider version, a fine-tune, a quantisation. The
  code is identical and the behaviour is not.
- **The prompt is code that nothing versions.** It ships in a string, changes without
  review, and has no test suite — which is `sculley2015`'s configuration debt with a
  much shorter feedback loop.

The through-line to build the part around: **an AI system's behaviour is determined by
artefacts that conventional version control does not cover** — prompts, evaluation sets,
retrieved corpora, model versions, tool schemas. Every chapter should ask what changed,
what recorded it, and how long until anyone noticed.

That connects directly to {{part:22}}'s finding that semantic failure has no instrument:
here the question is not only whether you can *see* a regression but whether you can
*attribute* it.

## Chapter plan

| ch | id | measurement to build |
|---|---|---|
| 205 | `ops-lifecycle` | where time actually goes across the lifecycle; the loop that is not a line |
| 206 | `ops-versioning` | how many artefacts determine behaviour; reproducibility as a product of coverage |
| 207 | `ops-deployment` | canary sizing when the signal is semantic; detection time vs blast radius |
| 208 | `ops-observability` | what a trace must carry to attribute a regression; sampling vs attribution |
| 209 | `ops-prompt-versioning` | prompt change velocity vs eval-set staleness; the gate that decays |
| 210 | `ops-agent-tracing` | trace volume and the triage bottleneck; TRAIL's 11% as a design constraint |
| 211 | `ops-governance` | cost attribution across shared components; who owns an untraceable bill |

Carry-through: `semantic-failure-has-no-instrument` (189),
`semantic-breaker-is-affordable` (195), `detection-time-sets-the-blast-radius` (195),
`derived-copies-multiply-contradiction` (193), `feature-credit-depends-on-order` (202),
`visible-half-is-what-is-reported` (183), `agent-errors-correlate` (169).
