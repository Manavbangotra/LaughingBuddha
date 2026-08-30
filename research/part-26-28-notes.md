# Research notes — Parts XXVI, XXVII, XXVIII

Research pass run 2026-08-30. Every paper below was verified by fetching its arXiv abstract
page; nothing here is cited from memory. Entries added to `data/bibliography.yaml` with
`verified: true` and a `verified_via` recording what the abstract page actually said.

## Verified and added (18)

### Part XXVI — AI Security

| key | arXiv | what it supplies |
|---|---|---|
| `perez2022ignore` | 2211.09527 | PromptInject; names **goal hijacking** and **prompt leaking** as the two canonical outcomes |
| `zou2023universal` | 2307.15043 | transferable adversarial suffix trained on Vicuna-7B/13B, transfers to **ChatGPT, Bard, Claude** |
| `wei2023jailbroken` | 2307.02483 | **competing objectives** and **mismatched generalization**; attacks succeed on *every* prompt in the models' own red-team sets |
| `carlini2021extracting` | 2012.07805 | hundreds of verbatim sequences from GPT-2; **single-document occurrences recoverable**; larger models more vulnerable |
| `carlini2023poisoning` | 2302.10149 | **0.01% of LAION-400M / COYO-700M for ~$60**, guaranteed, across 10 datasets |
| `debenedetti2024agentdojo` | 2406.13352 | **97 tasks, 629 security test cases**; utility and attack success measured together |
| `beurerkellner2025patterns` | 2506.08837 | architecture-first defence; patterns **deliberately constrain utility** in exchange for security |

Already present and reused: `greshake2023indirect` (2302.12173), `hou2025mcp`,
`huang2026mcpthreat`, `gaire2025mcpsok`, `cemri2025mast`, `mcp2026spec`, `qin2023toolllm`.

### Part XXVII — Responsible AI

| key | arXiv | what it supplies |
|---|---|---|
| `hardt2016equality` | 1610.02413 | equalised odds / equality of opportunity as **oblivious** criteria, plus post-processing |
| `kleinberg2016tradeoffs` | 1609.05807 | **three fairness conditions cannot hold simultaneously**, and approximate satisfaction needs an approximate special case |
| `abadi2016dpsgd` | 1607.00133 | DP-SGD: clipping, noise, moments accountant |
| `shokri2017membership` | 1610.05820 | membership inference, shadow models, commercial MLaaS, hospital discharge data |
| `ribeiro2016lime` | 1602.04938 | local surrogate explanation; **faithful locally, no global claim** |
| `lundberg2017shap` | 1705.07874 | unifies six methods; **uniqueness theorem** under stated axioms |
| `mitchell2019modelcards` | 1810.03993 | disaggregated evaluation across groups + intended-use disclosure |
| `gebru2021datasheets` | 1803.09010 | motivation, composition, collection process, **recommended uses** |

Already present and reused: `turpin2023faithfulness` (2305.04388), `petrov2023` (tokenizer
unfairness), `ji2023survey`, `sculley2015`, `breck2017`, `paleyes2020deployment`.

### Part XXVIII — Advanced Research

| key | arXiv | what it supplies |
|---|---|---|
| `shazeer2017moe` | 1701.06538 | **>1000× capacity**, 137B params between LSTM layers; load balancing named as the constraint |
| `ha2018worldmodels` | 1803.10122 | policy trained **entirely inside the learned model**, transferred back |
| `driess2023palme` | 2303.03378 | **562B**, SOTA on OK-VQA, **positive transfer** from joint training |

Already present and reused: `kaplan2020scaling` (2001.08361), `hoffmann2022chinchilla`
(2203.15556), `wei2022emergent` (2206.07682), `schaeffer2023mirage` (2304.15004),
`snell2024testtime` (2408.03314), `muennighoff2025s1` (2501.19393), `press2022alibi`
(2108.12409), `kirkpatrick2017ewc` (1612.00796), `lu2024aiscientist` (2408.06292),
`fedus2021switch`, `dettmers2023case4bit`, `kumar2024precisionscaling`,
`gutierrez2025hipporag2`.

## Considered and rejected

- **OWASP Top 10 for LLM Applications** — the standard practitioner reference for
  {{ch:sec-threat-model}}, and it is not an arXiv preprint, so it cannot be verified under
  this book's rule. The threat taxonomy in that chapter is therefore derived from first
  principles and from the verified papers above, and says so.
- **Anthropic's many-shot jailbreaking report** — real and relevant, published outside
  arXiv. Not fetched, not cited; {{ch:sec-jailbreaks}} uses `wei2023jailbroken`'s
  mismatched-generalization mode instead, which explains the same phenomenon.
- **AlphaFold (Jumper et al. 2021)** — Nature, not arXiv. {{ch:res-ai-for-science}} builds
  on `lu2024aiscientist` and reasons about the general structure rather than citing a
  result it cannot verify.
- **Dwork's differential privacy monograph** — a book. `abadi2016dpsgd` carries the
  mechanism this book actually needs.
- **NIST AI RMF and the EU AI Act** — regulatory instruments, not papers.
  {{ch:rai-regulation}} treats their *structure* (risk tiers, obligations, conformity
  assessment) as a design problem and does not quote text from either.
- **StruQ / structured queries (2402.06363)** and several 2026 agent-security preprints —
  plausible and **not fetched**, therefore not cited.

## Notes on how these get used

- `carlini2023poisoning`'s **$60** is the number that reframes supply-chain risk for
  {{ch:sec-poisoning}}: it moves poisoning out of the nation-state threat model.
- `wei2023jailbroken`'s **mismatched generalization** is the load-bearing idea for
  {{ch:sec-jailbreaks}}, because it predicts that safety coverage lags capability coverage
  structurally rather than temporarily.
- `kleinberg2016tradeoffs` is the organising result for {{ch:rai-bias}} — it converts
  fairness from an optimisation into a stated choice, which is the same move this book makes
  for cost ratios in {{ch:ev-classical-metrics}}.
- `ribeiro2016lime` and `lundberg2017shap` are used together in {{ch:rai-interpretability}}
  specifically to separate *axiomatic uniqueness* from *causal correspondence*, which is the
  distinction their popular usage collapses.
- `shazeer2017moe`'s load-balancing caveat and {{ch:inf-parallelism}}'s sparsity-erosion
  result are the same constraint measured seven years apart, and {{ch:res-moe}} reads them
  together.

## Still unverified, deliberately

Several 2026 preprints on agent security, mechanistic interpretability, and continual
learning surfaced during search and were **not fetched**, so nothing in these parts cites
them. Any later pass adding them must go through the same abstract-page verification.
