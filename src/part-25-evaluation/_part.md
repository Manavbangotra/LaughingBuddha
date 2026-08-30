---
id: part-25-intro
status: draft
---

## What this part is for

{{part:24}} governed a system that was already working. This part asks how you would know.

**Every quantity worth measuring about an AI system has no ground truth, and every
quantity with a ground truth is adjacent to the one you care about.**

That sentence is the part. A summary has no correct answer, only a space of acceptable
ones. A benchmark score has no units until somebody measures a human on the same items. A
judge agrees with people at exactly the rate people agree with each other. A retrieval
metric measures a stage the user never sees. A single-run agent success rate averages the
tasks that always work with the tasks that never do. And the metric your product is
actually about is observed on **0.8%** of sessions.

None of that makes evaluation impossible. It makes it a *design* problem — choose the
instrument, know its error, and stop reading it as though it were the thing.

> **The rule adopted for this part: every metric is a decision rule, and a decision rule
> that does not take its parameters has assumed them.** F1 assumes a 3:1 cost ratio.
> Exact match assumes an answer length. A reference metric assumes one draw is the truth. A
> proxy assumes a correlation. Each chapter recovers the assumption and prices it.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| Scale at which one capability "emerges", at 2 vs 12 answer tokens | **$3$B** vs **$20$B** | {{ch:ev-why-hard}} |
| Correct summaries a single-reference metric marks wrong | **$99.4\%$** | {{ch:ev-why-hard}} |
| Excess cost of F1's threshold at a 40:1 business ratio | **$141\%$** | {{ch:ev-classical-metrics}} |
| Precision gap between two models with identical AUC, at equal recall | **$2.3\times$** | {{ch:ev-classical-metrics}} |
| Year a 4,000-item benchmark stops tracking generational progress | **year 2** | {{ch:ev-llm-benchmarks}} |
| Headroom closed by a scenario scoring 0.470 | **$-23\%$** | {{ch:ev-llm-benchmarks}} |
| Gap compression from a 14% label error rate | **$\times 0.720$** | {{ch:ev-human}} |
| Share of annotator disagreement caused by the guideline | **$37\%$**, at $6\times$ cheaper | {{ch:ev-human}} |
| Judge–human agreement against human–human agreement | **$81\%$** against **$81\%$** | {{ch:ev-llm-judge}} |
| Share of a judge selection loop's reported gain that is real | **$20\%$** | {{ch:ev-llm-judge}} |
| End-to-end value of one point of retrieval recall | **$0.377$** | {{ch:ev-rag}} |
| RAG failures visible to retrieval metrics | **$32.5\%$** | {{ch:ev-rag}} |
| Agent pass^8 against what independence would predict | **$27\%$** against **$1.21\%$** | {{ch:ev-agents}} |
| Outcome-evaluation passes that will not recur | **$24\%$** | {{ch:ev-agents}} |
| Nine instruments: sum of coverages against their union | **$1.827$** against **$0.956$** | {{ch:ev-framework}} |
| Reference-free share of the optimal evaluation budget | **$100\%$** | {{ch:ev-framework}} |
| Days to detect a 3% change on a click against on the outcome | **1.2** against **48.7** | {{ch:ev-online}} |
| Plausible regression gates that cannot detect their own tolerance | **4 of 5** | {{ch:ev-online}} |

## The organising idea

**Every chapter finds an instrument that is precise about something adjacent.**

{{part:22}}'s instruments were silent. {{part:23}}'s were aimed at the wrong quantity.
{{part:24}}'s reported without bounding. This part's are *accurate* — and about a different
question from the one being asked.

```text
   CHAPTER                  THE INSTRUMENT IS EXACT ABOUT   THE QUESTION WAS
   ──────────────────────   ─────────────────────────────   ────────────────────────
   212 why hard             this reference answer           any acceptable answer
   213 classical metrics    a 3:1 cost ratio                your cost ratio
   214 benchmarks           a fixed item set, aging         the current frontier
   215 human evaluation     what two annotators agree on    what is true
   216 LLM-as-judge         concordance with annotators     correctness
   217 RAG evaluation       retrieval, then faithfulness    utilisation, then sufficiency
   218 agent evaluation     one run, one outcome            every run, the whole trajectory
   219 the framework        each instrument's own coverage  the union, and the loop's time
   220 online evaluation    a proxy, very precisely         the outcome
```

Read the right column downward. Every entry is harder to measure than its left-hand
neighbour, and in every case the *reason* is the same: the left column is a property of
artefacts the system already produced, and the right column requires a judgement from
outside it. **The measurable set and the binding set are separated by whether a human has
to say something.**

## The three through-lines

**First: the escape from every reference problem is a predicate.**

{{ch:ev-why-hard}} found that a reference is one draw from a space of acceptable answers,
and that the only clean escape is to state an acceptance condition instead —
{{cite:chen2021humaneval}}'s tests, {{cite:jimenez2023swebench}}'s issues. That result
then reappears, unchanged, in four more chapters:

| Chapter | The reference that fails | The predicate that works |
|---|---|---|
| {{ch:ev-why-hard}} | one written summary | a unit test |
| {{ch:ev-rag}} | a gold answer | a faithfulness or utilisation check |
| {{ch:ev-agents}} | a reference trajectory | an invariant on consequences |
| {{ch:ev-framework}} | a labelled evaluation set | schema, execution, invariants |
| {{ch:ev-online}} | a two-sided significance test | a one-sided non-inferiority bound |

And it is why **{{ch:ev-framework}}'s optimal portfolio is 100% reference-free at any
realistic budget.** That is not a coincidence; it is this result, priced.

**Second: the same compression arithmetic appears from four unrelated mechanisms.**

A gap between two systems gets multiplied by a factor less than one, and the score keeps
looking reasonable:

| Chapter | Mechanism | Compression |
|---|---|---|
| {{ch:ev-llm-benchmarks}} | contamination makes items uninformative | $\times(1-c)$ |
| {{ch:ev-human}} | label error makes items anti-informative | $\times(1-2e)$ |
| {{ch:ev-llm-judge}} | position bias decides close pairs | $\times$ flip rate |
| {{ch:ev-online}} | proxy correlation | $\times\rho$ |

**Score inflation is a bias you can subtract; gap compression is a loss of signal and there
is nothing to subtract.** In all four cases the required sample size grows as the square of
the compression, which is why each of them terminates in an item-count that nobody has.

**Third: the instrument that gets built is the one that needs nothing external.**

Faithfulness dominates RAG dashboards because it compares two artefacts the pipeline already
produced. Exact match dominates because nobody negotiates a yes-or-no. AUC dominates because
it needs no threshold. Recall@k dominates because relevance labels exist. Click-through
dominates because it is a side effect of using the product.

In every case the property that made the instrument cheap is the property that made it
adjacent. **Self-referential is why it is cheap and why it cannot bound the thing you care
about**, and the chapters that state this most sharply are {{ch:ev-rag}} and
{{ch:ev-online}}.

## What this part does not settle

**Correlated errors are assumed away everywhere.** Between gates, between annotators,
between judges and the humans that trained them, between instruments in a portfolio. Each
chapter names its assumption; the corrections mostly run against the optimistic reading, and
none is measured.

**Whether judge biases are stable enough to correct for is open.** {{ch:ev-llm-judge}}
argues that protocols invariant to a bias beat corrections that estimate it, which is the
right instinct while the magnitudes are unmeasured and would be the wrong one if they were
known.

**The tolerance is never written down.** {{ch:ev-online}}'s capability check requires
somebody to state the regression they refuse to ship, and that decision is exactly what a
significance threshold is usually standing in for.

**And nobody measures $|A|$.** The size of the acceptable-answer space is the parameter that
{{ch:ev-why-hard}}, {{ch:ev-rag}} and {{ch:ev-agents}} all depend on, it is estimable from
one afternoon of double-writing, and this book found no published figures for it.

## How to read this part

{{ch:ev-why-hard}} is load-bearing for everything else — the reference-sampling result and
the agreement ceiling are used in six of the eight chapters that follow, and its
predicate-not-reference conclusion is the part's single most transferable idea.

If you are building rather than reading through: {{ch:ev-framework}} has the greedy build
order, and it is short enough to act on directly. The first three instruments in it cost
under 35 per thousand items between them and reach **0.484** coverage.

If you already have an evaluation programme and want to know whether it works, go to
{{ch:ev-online}}'s capability check first. Computing MDE against tolerance for each existing
gate takes an hour and, on the numbers here, retires four gates in five.
