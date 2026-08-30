# Part XXV research notes — AI Evaluation

Research pass run 2026-08-30. Every paper below was verified by fetching its arXiv
abstract page; nothing here is cited from memory. Entries added to
`data/bibliography.yaml` with `verified: true` and a `verified_via` recording what the
abstract page actually said.

## Verified and added (12)

| key | arXiv | what it supplies to this part |
|---|---|---|
| `hendrycks2020mmlu` | 2009.03300 | 57 subjects; GPT-3 ~+20pp over chance; **poor calibration — "frequently do not know when they are wrong"** |
| `liang2022helm` | 2211.09110 | 7 metrics × 16 core scenarios (87.5% coverage); 30 models × 42 scenarios; **coverage 17.9% → 96.0%** |
| `zheng2023judge` | 2306.05685 | **>80% judge–human agreement**; 3K expert votes, 30K conversations; **position, verbosity, self-enhancement biases** |
| `schaeffer2023mirage` | 2304.15004 | discontinuous metrics manufacture emergence; confirmed on InstructGPT/GPT-3, BIG-Bench meta-analysis, and vision |
| `wang2023unfair` | 2305.17926 | **order swap alone made Vicuna-13B beat ChatGPT on 66 of 80 queries** with ChatGPT as judge |
| `jimenez2023swebench` | 2310.06770 | 2,294 real GitHub issues, tests as grader; **Claude 2 resolved 1.96%** |
| `yao2024taubench` | 2406.12045 | multi-turn tool + simulated user; **<50% single-run, pass^8 <25% in retail** |
| `singh2025leaderboard` | 2504.20879 | 27 private Meta variants; **19.2% / 20.4% data share vs 29.7% for 83 open models**; **up to +112%** from extra data |
| `chen2021humaneval` | 2107.03374 | pass@k and functional correctness; **28.8% at one sample vs 70.2% at 100** |
| `card2020power` | 2010.06595 | **2000-sentence MT test sets ≈ 75% power for 1 BLEU**; GLUE comparisons and human studies underpowered |
| `ribeiro2020checklist` | 2005.04118 | behavioural testing; **2× tests and ~3× bugs** vs unaided; bugs found in an extensively tested commercial model |
| `rein2023gpqa` | 2311.12022 | 448 questions; **expert 65% (74% discounting slips) vs non-expert 34%** with >30 min and web; GPT-4 baseline 39% |

## Already in the bibliography and reused here

`ragas2023` (RAG evaluation), `agentbench2023`, `apibank2023`, `mlebench2024`,
`spider2sql`, `beir2021`, `mteb2022`, `glue2018`, `hallucination-survey`,
`cemri2025mast`, `deshpande2025trail`, `breck2017`, `sculley2015`,
`paleyes2020deployment`, `gama2014`, `chen2023frugalgpt`, `hu2024routerbench`.

## Considered and rejected

- **BIG-bench (2206.04615)** — real and relevant, but every quantitative claim this part
  would want from it is already available second-hand through `schaeffer2023mirage`'s
  meta-analysis, which is the framing the chapters actually use. Not fetched, therefore
  not cited.
- **"A Survey on Evaluation of Large Language Models" (2307.03109)** — a survey; the part
  needs primary measurements, and every specific number would have to be traced to its
  source anyway. Not fetched, not cited.
- **Chatbot Arena (2403.04132)** — the arena's own paper. `singh2025leaderboard` supplies
  the quantitative claims this part makes about arena rankings, and is the more useful
  citation because it is the critical one. Not fetched, not cited.
- **AlpacaFarm (2305.14387)** — plausible and unfetched. Any claim it would support is
  covered by `zheng2023judge`.
- **Kohavi's online-experiment work** — the standard reference for A/B testing discipline
  is a book, not an arXiv preprint, so it cannot be verified under this book's rule.
  {{ch:ev-online}} therefore builds its statistical claims on `card2020power` and its
  operational ones on `breck2017` and `sculley2015`.
- **Inter-annotator agreement (Artstein & Poesio, Krippendorff)** — foundational and not
  on arXiv. {{ch:ev-human}} derives what it needs from first principles rather than citing
  an unverifiable source, and says so in the chapter.

## Notes on how these get used

- The **calibration** finding in `hendrycks2020mmlu` and the **human baseline** in
  `rein2023gpqa` are the two most under-quoted results in this set, and both are load
  bearing for {{ch:ev-llm-benchmarks}}.
- `wang2023unfair`'s 66-of-80 and `zheng2023judge`'s >80% agreement should be read
  together: they are not in tension, they describe different conditions, and
  {{ch:ev-llm-judge}} is built on the gap between them.
- `yao2024taubench`'s pass^8 is the single most important number in the part for anyone
  shipping agents, and {{ch:ev-agents}} treats it as the organising result.
- `card2020power` is the only source here that constrains evaluation-set *size*, and every
  regression-gate claim in {{ch:ev-online}} is anchored to it.

## Still unverified, deliberately

Several 2026 evaluation preprints surfaced during search and were **not fetched**, so
nothing in this part cites them. If a later pass adds them, they must go through the same
abstract-page verification.
