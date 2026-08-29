# Part XXII research notes — AI System Design

Research pass 2026-08-29. Two new citations verified; the rest of this part draws
on already-verified work from Parts XI, XV and XIX.

## New this pass

- `chen2023frugalgpt` — FrugalGPT (2305.05176, 9 May 2023, 3 authors). Three
  strategies: prompt adaptation, LLM approximation, **LLM cascade**. Headline:
  matches the best individual model with **up to 98% cost reduction**, or
  **+4% accuracy at the same cost**. *The cascade only works if something judges
  when the cheap answer suffices — so the paper is about the judge, which is this
  book's recurring verifier question wearing a cost hat.*

- `hu2024routerbench` — RouterBench (2403.12031, 18 Mar 2024, v2 28 Mar, 8
  authors). **405,000+ precomputed inference outcomes**, so routing policies can be
  evaluated offline against fixed model responses and costs. Framing claim: no
  single model optimally addresses all tasks when balancing performance against
  cost. *Made the cost-quality frontier plottable rather than assertable.*

  Note: the abstract does not itemise the model or task counts. Secondary sources
  say eleven models across seven or eight datasets; **chapters must cite only the
  405k figure and the framing claim**, which are what the abstract states.

## Carried in from earlier parts

| need | citation | part |
|---|---|---|
| serving throughput, KV cache | `kwon2023pagedattention`, `pope2022inference` | XV |
| speculative decoding | `leviathan2023speculative`, `cai2024medusa` | XV |
| ANN indexes for vector stores | Part XI's ANN work | XI |
| protocol, transports, correlation | `mcp2026spec`, `hou2025mcp` | XIX |
| agent failure taxonomy | `cemri2025mast` | XVIII |
| retrieval at scale | `qin2023toolllm`, `patil2023gorilla` | XIX |

## The organising problem

This part is architecture for a system whose central component is **nondeterministic,
expensive, and occasionally wrong** — three properties conventional system design
does not assume together.

The consequences compound in a specific way worth building the part around:

- **Nondeterministic** breaks caching (the same input may deserve a different
  answer), breaks retries (a retry is a fresh sample, not a repeat), and breaks
  testing (no golden output).
- **Expensive** makes routing a first-order design decision rather than an
  optimisation, and makes the cost of a *wasted* call comparable to the cost of a
  useful one.
- **Occasionally wrong** means fault tolerance has to cover semantic failure and
  not only availability failure — a 200 response containing a wrong answer is the
  case conventional reliability engineering has no vocabulary for.

Every chapter here should ask what a classical technique does when the component
has those three properties.

## Chapter plan

| ch | id | measurement to build |
|---|---|---|
| 189 | `sd-architecture` | where nondeterminism forces a different shape; the boundary that contains it |
| 190 | `sd-routing-caching` | cascade economics; cache hit rate vs staleness vs semantic mismatch |
| 191 | `sd-async` | queueing with heavy-tailed service times; streaming and perceived latency |
| 192 | `sd-retrieval-agents` | retrieval at scale; agent fan-out and tail amplification |
| 193 | `sd-storage` | vector store recall vs cost; what belongs where |
| 194 | `sd-apis-auth` | rate limiting a nondeterministic-cost workload |
| 195 | `sd-fault-tolerance` | retries when a retry is a resample; semantic circuit breakers |
| 196 | `sd-latency` | latency budgets with heavy tails; where p99 comes from |

Carry-through: `loop-is-not-a-chain` (155), `replay-needs-idempotence` (166),
`transport-decides-correlation` (171), `severity-hides-in-the-mean` (171),
`agent-errors-correlate` (169), `retry-needs-a-verifier` (168),
`gate-on-consequence` (160), `context-is-a-budget` (173).

## Not used

Several 2025–2026 routing papers surfaced in search (2509.06274, 2510.00841,
2410.10347, 2601.07206, 2603.04445, 2406.18665) and were **not fetched or
verified**. Not citable until verified.
