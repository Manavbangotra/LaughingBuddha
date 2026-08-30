---
id: part-28-intro
status: draft
---

## What this part is for

{{part:27}} treated obligations as engineering with measurable properties. This part treats the
frontier the same way, and ends the book by supplying the instrument for reading work that is not
settled.

**Every chapter here takes a research direction that is discussed qualitatively and prices it.**
Scaling laws as an allocation rule. Sparsity as an exchange rate between two budgets. Long
context as a utilisation problem. Test-time compute as an axis with a ceiling. Continual learning
as a single dial. World models as a rollout horizon. Automated science as a review queue. And the
frontier itself as a rubric with a break-even.

> **The rule adopted for this part: name the currency, then find the exchange rate.** Every
> result here is a trade between two resources that are usually discussed separately — training
> FLOPs against serving bytes, horizon against per-step error, wall-clock against transfer,
> generation against verification. None of these chapters says a direction is good or bad. Each
> says what it costs and in which currency, and the currency is usually not the one being
> optimised.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| Tokens-per-parameter under two exponent pairs | **325.0** vs **11.9** | {{ch:res-scaling}} |
| Floor's share of the loss at $10^{27}$ FLOPs | **$94.2\%$** | {{ch:res-scaling}} |
| Compute for each halving of the reducible loss | **86×** | {{ch:res-scaling}} |
| Overspend from serving a training-optimal model | **13.7×** | {{ch:res-scaling}} |
| Apparent discontinuity, loss against 20-token exact match | **4.2** vs **1152.9** | {{ch:res-scaling}} |
| Power-law fit error six decades out, no floor term | **$48.9\%$** | {{ch:res-scaling}} |
| MoE weights against a dense baseline | **85.7×** | {{ch:res-moe}} |
| Training saving against the dense model it matches | **3.2×** | {{ch:res-moe}} |
| Serving cost against that same dense model | **5.10×** | {{ch:res-moe}} |
| Break-even for a sparse design, in tokens served | **$2.02 \times 10^{12}$** | {{ch:res-moe}} |
| Tokens dropped at capacity factor 1.0, then 2.0 | **$25.75\%$** → **$10.90\%$** | {{ch:res-moe}} |
| Loss cost of no balance loss, then heavy balance | **+0.0597** / **+0.0456** | {{ch:res-moe}} |
| Recall at a window's end against its middle | **0.956** vs **0.674** | {{ch:res-memory}} |
| One fact against five, at 128,000 tokens | **0.775** → **0.279** | {{ch:res-memory}} |
| Cost per solved task, largest window against smallest | **69×** | {{ch:res-memory}} |
| KV cache against a vector index, per token | **12,800×** | {{ch:res-memory}} |
| Horizons where no single memory architecture reaches 0.90 | **2 of 5** | {{ch:res-memory}} |
| Test-time ceiling at a verifier of 0.86 | **0.9636** | {{ch:res-test-time}} |
| Share of that ceiling from the first 16 samples | **$77.9\%$** | {{ch:res-test-time}} |
| Crossover between training and sampling, in requests | **$4.6 \times 10^{12}$** | {{ch:res-test-time}} |
| Throughput cost of full-weight test-time adaptation | **591×** | {{ch:res-test-time}} |
| Absorption and forgetting at plasticity 0.10, then 4.00 | **0.221**/**0.0028** → **1.000**/**0.4029** | {{ch:res-continual}} |
| Cheapest update cadence, against daily | **$827,919** vs **$40.7M** | {{ch:res-continual}} |
| Self-improvement peak round, then round 8 | **0.5996** → **0.4901** | {{ch:res-continual}} |
| Planning horizon at per-step error 0.030, then 0.003 | **9** → **35** steps | {{ch:res-world-models}} |
| Model-based gain, grasping against cooking | **0.4200** vs **0.0095** | {{ch:res-world-models}} |
| Horizon per unit cost, hierarchy against a better model | **2.57** vs **0.03** | {{ch:res-world-models}} |
| Fleet-months for the real residual, 50 robots | **50.8** | {{ch:res-world-models}} |
| Established findings at 400 candidates, then 400,000 | **17.0** → **17.0** | {{ch:res-ai-for-science}} |
| Worth of verifying a candidate at the generator's base rate | **−\$1,390** | {{ch:res-ai-for-science}} |
| Value per dollar, reproduction against novelty | **0.6721** vs **0.1250** | {{ch:res-ai-for-science}} |
| Cost of an unpublished negative, against publishing it | **4.8×** | {{ch:res-ai-for-science}} |
| Confidence range, all evidence against none | **53×** | {{ch:res-frontier}} |
| Correlation with confidence: citations, then a reproduction | **0.21** → **0.81** | {{ch:res-frontier}} |
| Adoption break-even probability at a 1.55 lead premium | **0.549** | {{ch:res-frontier}} |
| Five-year survival: established, emerging, speculative | **0.89** / **0.38** / **0.09** | {{ch:res-frontier}} |

## The organising idea

**Every chapter finds the frontier optimising a currency that is not the one that binds.**

{{part:26}}'s controls pointed at the wrong noun. {{part:27}}'s measurements answered the wrong
question. This part's *research* improves the wrong resource — correctly, rigorously, and in a
budget that was not the constraint.

```text
   CHAPTER                  WHAT THE WORK OPTIMISES     WHAT ACTUALLY BINDS
   ──────────────────────   ─────────────────────────   ────────────────────────────
   233 scaling laws         training FLOPs              serving bytes, and a floor
   234 mixture of experts   FLOPs per token             bytes per request
   235 long context         window length               utilisation of the window
   236 test-time compute    the sampler                 the verifier's ceiling
   237 continual learning   the update rule             the drift estimate
   238 world models         per-step prediction error   wall-clock and transfer
   239 automated science    hypothesis generation       review capacity
   240 the frontier         which claims are true       which claims are checkable
```

Read the right column. Not one of those is exotic, and in every case it is measurable in an
afternoon by somebody who thought to measure it. **The left column is where the papers are and
the right column is where the decisions are**, and the gap between them is this part's subject.

## The three through-lines

**First: the same conclusion arrives four times from unrelated mechanisms.**

| Chapter | The mechanism | The conclusion |
|---|---|---|
| {{ch:res-scaling}} | FLOPs and model size | the deployment optimum is not the training one |
| {{ch:res-moe}} | memory bandwidth and expert count | break-even at $2.02 \times 10^{12}$ served tokens |
| {{ch:res-test-time}} | verifier ceiling and per-request cost | crossover at $4.6 \times 10^{12}$ requests |
| {{ch:res-world-models}} | wall-clock and transfer coefficient | a better simulator beats a bigger fleet |

**The serving forecast is a research decision**, reached independently through arithmetic,
bandwidth, verification and robotics. When four unrelated mechanisms agree, the conclusion is
about the structure of the problem rather than about any of them.

**Second: the cheap fixes work by not needing the expensive thing.**

{{ch:res-world-models}}'s hierarchy extends the horizon by shortening the sequence — **2.57**
against a better model's **0.03**. {{ch:res-memory}}'s retrieval makes a long window work by not
filling it — **11.9** success per unit cost against a longer window's **0.05**.
{{ch:res-ai-for-science}}'s triage raises the prior instead of the review rate. In each case the
direct approach is available, expensive, and roughly two orders of magnitude worse per unit.

**Third: every ceiling in this part belongs to a verifier or a floor, not to a budget.**

{{ch:res-scaling}}'s irreducible loss reaches **94.2%** of the reported number.
{{ch:res-test-time}}'s sampling saturates at **0.9636**, set by the verifier.
{{ch:res-continual}}'s self-training improves only the verifiable fraction.
{{ch:res-ai-for-science}}'s output is capped by review. **Spending more of the obvious resource
approaches a ceiling; moving the ceiling requires spending on something else**, and the something
else is nearly always cheaper.

## What this part does not settle

**The sparse-parameter credit is not measured.** {{ch:res-moe}}'s $\rho$ decides where the
break-even sits and it is assumed, not fitted. So is {{ch:res-world-models}}'s transfer
coefficient, {{ch:res-test-time}}'s verifier quality, and {{ch:res-frontier}}'s value multiplier
for novelty. Each is named where it appears.

**Correlation is assumed away, again.** {{ch:res-memory}}'s multi-fact product assumes
independent positions; {{ch:res-frontier}}'s rubric assumes independent evidence factors;
{{ch:res-continual}}'s diversity model is a scalar. The corrections run in known directions and
none is measured.

**And this part's own numbers are speculative by its own rubric.** {{ch:res-frontier}} says so
explicitly: the structural claims here are derivable and the coefficients are furniture. That
distinction is the part's method applied to itself, and it should be applied by the reader too.

## How to read this part

{{ch:res-scaling}} is load-bearing for {{ch:res-moe}} and {{ch:res-test-time}} and supplies the
part's recurring shape: a curve, a floor, and a budget that is spent in the wrong currency.

If you are choosing an architecture this quarter: {{ch:res-moe}} and {{ch:res-memory}} together,
and start by writing down your serving forecast — both chapters turn on it and neither is
decidable without it.

If you are planning a research programme: {{ch:res-ai-for-science}} and {{ch:res-frontier}}, in
that order. The first says the bottleneck is verification; the second says how to do it.

And {{ch:res-frontier}} is the chapter to read even if you skip the rest. It is the instrument
for everything in this part that will have changed by the time you read it, which — on its own
survival table — is most of it.
