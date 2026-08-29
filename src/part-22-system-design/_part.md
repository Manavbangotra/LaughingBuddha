---
id: part-22-intro
status: draft
---

## What this part is for

{{part:21}} asked how far a strong verifier gets you. This part asks a different
question, and it is the one that decides whether any of the preceding twenty-one parts
survives contact with production.

**A language model is nondeterministic, expensive, and occasionally wrong. Conventional
system design assumes none of those.** Not one of them individually — all three at
once, in the component the whole system is built around.

Each has precedent on its own. Nondeterminism is familiar from distributed systems.
Expense is familiar from anything with a cloud bill. What has no precedent is the
third: a component that **succeeds and is wrong**, returning `200 OK` with a confident
wrong answer. That one turns out to do the most damage, and it is the one nothing in
the stack measures.

> **The rule adopted for this part: every technique is measured against what survives,
> not against whether it still runs.** A cache that serves stale answers still returns
> quickly. A health check on a wrong answer still passes. A retry on a semantic failure
> still executes. Survival means the technique still does the job it was adopted for.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| Classical techniques surviving all three properties | nothing above **$36\%$**; health checks **$5\%$** | {{ch:sd-architecture}} |
| A $6\%$ semantic error rate against a $99.9\%$ availability budget | **$61\times$** overspend, dashboard reads $99.900\%$ | {{ch:sd-architecture}} |
| Cascade with a perfect judge vs the expensive model alone | **$97.5\%$** against **$91\%$**, at **$33\%$** of cost | {{ch:sd-routing-caching}} |
| Six workloads, identical mean service time | **$74.9\%$** down to **$45.1\%$** sustainable utilisation | {{ch:sd-async}} |
| A $1.3\%$ per-call tail, fanned out $20$ ways | **$23.0\%$** of requests | {{ch:sd-retrieval-agents}} |
| Corpus $1{,}000 \to 10$M, fixed retriever | coverage **$55.5\% \to 17.2\%$**, precision **$4\% \to 100\%$** | {{ch:sd-retrieval-agents}} |
| Storage placed by access shape vs left in the database | **$2.9\times$**, all of it in two items | {{ch:sd-storage}} |
| Agent on a service account | **$50.8\%$** of blast radius unreachable by the invoking user | {{ch:sd-apis-auth}} |
| Uniform three-retry policy | **$65\%$** of budget to the two worst categories | {{ch:sd-fault-tolerance}} |
| Sum of per-stage p99s vs true system p99 | **$1.27\times$**; all six stages "fail" while the system passes | {{ch:sd-latency}} |

## The organising idea

**Every chapter in this part finds a metric that is accurate about its own quantity
and silent about the one that decides whether the system is working.**

That is not a stylistic repetition. It is a structural consequence of the third
property: when failure is invisible to the instruments, every instrument keeps
reporting health, and the instruments are what teams manage by.

```text
   CHAPTER                  THE INSTRUMENT THAT LIES    WHAT IT MISSES
   ──────────────────────   ─────────────────────────   ─────────────────────────
   189 architecture         availability                semantic failure
   190 routing, caching     cost saved, hit rate        accuracy given up
   191 async, streaming     % latency streaming saved   seconds users wait
   192 retrieval at scale   precision@k                 fact coverage
   193 storage              per-store health            disagreement between stores
   194 APIs, rate limiting  requests per minute         cost per minute
   195 fault tolerance      retry success rate          value per retry, by kind
   196 latency              per-stage p99 compliance    what the system absorbs
```

Read that column downward. In every row the failing instrument is the *default* one —
the metric the tooling emits without being asked, the number on the dashboard someone
already built. **The second instrument always has to be constructed deliberately, and
it never comes for free with the platform.**

## The two through-lines

**First: heterogeneity does not average out, it concentrates.**

A control designed for a uniform population does not merely perform averagely on a
heterogeneous one — it performs worst exactly where the population is most extreme,
because the extremes consume the control's capacity. The retry budget flows to
unfixable failures *because* they are unfixable ({{ch:sd-fault-tolerance}}). The rate
limit sizes itself to the most expensive request and starves everyone else
({{ch:sd-apis-auth}}). The queue's wait is set by the second moment, not the first
({{ch:sd-async}}). Same mechanism, three layers.

**Second: the intervention that works usually acts on distribution shape, and the one
that fails is the one an optimisation instinct reaches for first.**

| Chapter | The instinct | What actually works |
|---|---|---|
| {{ch:sd-architecture}} | one boundary, model behind it | interleave; the good stages are not contiguous |
| {{ch:sd-async}} | add capacity | reduce service-time variance |
| {{ch:sd-retrieval-agents}} | optimise the slow dependency | hedge the tail; reorder the context |
| {{ch:sd-storage}} | speed up the pipeline | remove the deepest derived copy |
| {{ch:sd-apis-auth}} | rebuild delegation | remove two tools |
| {{ch:sd-fault-tolerance}} | tune the retry count | classify the failure first |
| {{ch:sd-latency}} | optimise the widest bar | narrow the noisiest stage |

In every row the instinct is defensible, expensive, and beaten by something cheaper
that a measurement would have surfaced. That is worth naming because the pattern is
predictive: when the obvious latency, cost, or reliability fix looks expensive, the
shape-based alternative is usually available and usually has not been priced.

## What this part does not settle

Three things are left genuinely open, and it is worth saying so rather than implying
closure.

**The second instrument's design.** {{ch:sd-fault-tolerance}} prices a sampled
semantic monitor at **$0.5\%$** of traffic and shows it is affordable. It does not
settle what the judge should be, how to keep it calibrated, or how to stop the sample
becoming unrepresentative — all of which are live problems.

**Correlation.** Every model in this part assumes independence somewhere: across
stages, across parallel calls, across retries, across derived copies. Real systems
share infrastructure, and the assumption fails in different directions for latency and
for reliability. The corrections are stated per chapter and none is measured.

**Where the boundary moves.** {{ch:sd-architecture}}'s interleaving result depends on
per-stage cost and coverage, both of which move as models get cheaper and deterministic
tooling improves. The method survives; the specific answer has a shelf life.

## How to read this part

The chapters are ordered so each one's failing instrument is introduced before the
chapter that has to work around it. {{ch:sd-architecture}} is load-bearing for all
seven that follow and should not be skipped.

If you are triaging a live system rather than reading through: {{ch:sd-latency}} and
{{ch:sd-fault-tolerance}} contain the two cheapest diagnostics in the part — a
per-week efficiency ranking and a failure-kind breakdown — and both are computable from
data you already have.
