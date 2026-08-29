# Part XXIII research notes — Model Serving and AI Infrastructure

Research pass 2026-08-29. Six new citations verified against arXiv abstract pages;
the rest of this part draws on already-verified work from Parts VII, X and XV.

## New this pass

- `shoeybi2019megatron` — Megatron-LM (1909.08053, 17 Sep 2019, 6 authors).
  Intra-layer tensor parallelism. **8.3B** parameters on **512 GPUs**, **15.1
  PetaFLOPs** sustained, **76% scaling efficiency** against a **39 TeraFLOPs**
  single-GPU baseline that is itself **30% of peak**. *The 30%-of-peak figure is the
  more useful one for this part: even the single-GPU baseline leaves two thirds of
  the hardware idle, which is where the roofline discussion starts.*

- `rajbhandari2020zero` — ZeRO (1910.02054, 4 Oct 2019, 4 authors). Partition
  optimiser state, gradients and parameters instead of replicating.
  **>100B** parameters, **13B without model parallelism**, **super-linear speedup on
  400 GPUs** at **15 Petaflops**, **8x model size** and **10x performance** over prior
  state of the art. *Super-linear is the interesting claim — it comes from aggregate
  memory bandwidth rising with device count, not from compute.*

- `fedus2021switch` — Switch Transformer (2101.03961, 11 Jan 2021, 3 authors).
  Single-expert routing. **Up to 7x** pre-training speed at equal compute, gains
  across **all 101 languages**, **4x** over T5-XXL at trillion scale, bfloat16.
  *Decouples parameter count from per-token compute, which changes what a serving
  memory budget means.*

- `patel2023splitwise` — Splitwise (2311.18677, 30 Nov 2023, 7 authors; v2 May 2024,
  ISCA 2024). Prompt phase compute-intensive, token phase memory-intensive.
  **1.4x throughput at 20% lower cost**, or **2.35x throughput at the same cost and
  power**. *Makes the fleet heterogeneous by design.*

- `zhong2024distserve` — DistServe (2401.09670, 18 Jan 2024, 8 authors, OSDI 2024).
  Goodput under two separate latency constraints. **7.4x more requests or 12.6x
  tighter SLO**, holding latency for **>90%** of requests. *Explicitly frames
  colocation as forcing a trade-off between TTFT and TPOT — the same
  one-number-two-jobs failure ch:sd-apis-auth found in rate limiting.*

- `agrawal2023sarathi` — SARATHI (2308.16369, 31 Aug 2023, 6 authors). Chunked
  prefills piggybacked with decodes. LLaMA-13B/A6000: **10x decode throughput**,
  **1.33x end-to-end**. LLaMA-33B/A100: **4.25x decode**, **1.25x end-to-end**.
  GPT-3 pipeline: **6.29x bubble reduction**, **1.91x end-to-end**.

## The tension worth building the part around

`zhong2024distserve` and `agrawal2023sarathi` solve the **same** problem in opposite
directions:

| | approach | mechanism |
|---|---|---|
| SARATHI | **mix** the phases | chunk prefill so every slot carries prefill + decodes |
| DistServe / Splitwise | **separate** the phases | run prefill and decode on different machines |

Both report large gains against colocated-and-unmixed baselines. **Neither is
strictly better**, and the choice depends on fleet homogeneity, interconnect cost of
shipping KV cache between machines, and how tight the two latency constraints are
relative to each other. Chapters must present this as a live architectural choice,
not a settled one — a chapter that presents disaggregation as the answer would be
wrong by 2024 and wrong again by whatever comes next.

## Carried in from earlier parts

| need | citation | part |
|---|---|---|
| KV cache paging, continuous batching | `kwon2023pagedattention` | XV |
| prefill vs decode, arithmetic intensity | `pope2022inference` | XV |
| speculative decoding | `leviathan2023speculative`, `cai2024medusa` | XV |
| attention IO-awareness | `dao2022flash` | VII |
| memory math, quantisation formats | Part XV's format work | XV |
| queueing, variance, latency budgets | Part XXII's results | XXII |

## Not cited (and why)

- **Orca** (Yu et al., OSDI 2022) — the original continuous-batching paper. Not on
  arXiv, so not verifiable by this book's rule. `kwon2023pagedattention` covers the
  mechanism and is verified.
- **Roofline** (Williams, Waterman, Patterson, CACM 2009) — not on arXiv. The model
  is used in {{ch:inf-gpu-memory}} as standard material without a citation, which is
  honest: it is textbook content, not a claim needing support.
- Several 2025–2026 serving papers surfaced in search and were **not fetched or
  verified**. Not citable until verified.

## The organising problem

Parts XXII and below treated the model as a component with a price and a latency
distribution. This part opens the box.

The single fact that organises it: **a GPU is a machine for doing arithmetic, and
generation barely does any.** Decode reads the entire weight matrix to produce one
token, so it is bound by memory bandwidth rather than compute, and the arithmetic
intensity is around 1 operation per byte against hardware built for hundreds. Every
technique in this part is a way of raising that ratio — batching, chunking,
speculation, sparsity — or of arranging hardware so the two phases each get what they
need.

Every chapter should ask: *what is the arithmetic intensity of this, and which side
of the roofline ridge does it sit on?*

## Chapter plan

| ch | id | measurement to build |
|---|---|---|
| 197 | `inf-cpu-gpu` | where the FLOPs actually go; why CPU inference is not simply slower |
| 198 | `inf-gpu-memory` | roofline placement of prefill vs decode; what memory bandwidth costs |
| 199 | `inf-batching` | batch size against latency and throughput; the continuous-batching gain |
| 200 | `inf-parallelism` | which parallelism dimension to add, and the communication each costs |
| 201 | `inf-distributed` | mixing vs separating the phases; when each wins |
| 202 | `inf-serving-stacks` | what a stack actually buys over a naive loop, decomposed |
| 203 | `inf-kubernetes` | autoscaling a workload whose load signal lags its cost |
| 204 | `inf-edge` | where inference should run, priced per deployment target |

Carry-through from XXII: `variance-not-mean-drives-wait` (191),
`streaming-capacity-is-set-by-ttft` (191), `sum-of-tails-overprovisions` (196),
`tail-attribution-differs-from-mean` (196), `three-properties-break-the-stack` (189),
`access-shape-decides-the-store` (193).
