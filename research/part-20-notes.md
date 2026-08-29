# Part XX research notes — AI for Data Science

Research pass 2026-08-29. Everything verified against arXiv abstract pages on that
date; nothing from memory.

## The numbers this part is built on

| what | number | source |
|---|---|---|
| Text-to-SQL, realistic databases | **40.08%** execution accuracy vs **92.96%** human | `li2023bird` |
| Agentic data science tasks | **30.5%** best-model accuracy | `huang2024dacode` |
| ML engineering (Kaggle) | **16.9%** of competitions at bronze-medal level | `chan2024mlebench` |
| Automated paper generation | **under $15** per paper | `lu2024aiscientist` |

Those four numbers are the spine of the part. They come from four different
research groups, use four different grading methods, and agree on the shape: the
demos are impressive and the completion rates are low.

## Papers verified this pass

- `yu2018spider` — Spider (1809.08887, 24 Sep 2018, v5 2 Feb 2019, 12 authors,
  EMNLP 2018). **10,181 questions, 5,693 unique complex SQL queries, 200
  multi-table databases, 138 domains.** The key design is the **database split**:
  different queries *and* different databases in train and test, so a system must
  read an unseen schema. *This is what made text-to-SQL a generalisation problem.*

- `li2023bird` — BIRD (2305.03111, 4 May 2023, v3 15 Nov 2023, 18 authors,
  NeurIPS 2023 D&B). **12,751 pairs, 95 databases, 33.4 GB, 37 professional
  domains.** Headline: **40.08% execution accuracy against 92.96% human.** The
  difficulty is in the *data* — dirty values, external knowledge — not the SQL.

- `huang2024dacode` — DA-Code (2410.07331, 9 Oct 2024, v2 11 Oct, 11 authors,
  EMNLP 2024). Agent-shaped data science tasks from Kaggle/GitHub spanning EDA,
  preparation and modelling, requiring SQL + Python + Bash. **Best LLMs: 30.5%.**

- `chan2024mlebench` — MLE-bench (2410.07095, 9 Oct 2024, v6 26 Feb 2025, 12
  authors, ICLR 2025). **75 Kaggle competitions**, human baselines taken from the
  actual public leaderboards. Best configuration (o1-preview + AIDE scaffolding)
  **medals in 16.9%**. Also reports resource scaling and contamination checks.
  *Note: scaffolding mattered as much as the model — worth a listing.*

- `lu2024aiscientist` — The AI Scientist (2408.06292, 12 Aug 2024, v3 1 Sep 2024,
  6 authors). Idea → code → experiments → figures → paper → simulated review, at
  **under $15 per paper**, across diffusion, transformer LM, and learning
  dynamics. **Critical caveat to carry:** the "exceeds a top-conference acceptance
  threshold" claim is *as judged by the authors' own automated reviewer*. That is
  exactly the circularity `ch:as-specialized`'s verifier result predicts, and the
  chapter must say so plainly rather than repeating the headline.

- `testini2025dsautomation` — Measuring Data Science Automation (2506.08800,
  10 Jun 2025, v2 22 Oct 2025, 3 authors, **TMLR** Oct 2025). Three gaps in how
  the field evaluates: (1) focus on a small subset of **goal-oriented** activities,
  **ignoring data management and exploratory** work; (2) evaluation only at the
  extremes — pure assistance or full autonomy — neglecting **intermediate
  collaboration**; (3) optimising for **human substitution** rather than task
  **redesign**. *This is the framing paper for ch177 and ch182.*

## The selection effect worth building a chapter on

`testini2025dsautomation`'s first gap is the important one for this part, and it
compounds with the benchmark numbers above.

Benchmarks measure what can be graded. What can be graded in data science is the
part with a checkable answer — a query that returns the right rows, a model that
scores above a threshold. What practitioners spend their time on is data
management, exploration, and deciding what question to ask, none of which has a
reference answer.

So the field's measured capability is concentrated in exactly the activities that
are *not* where the time goes. Any claim of the form "agents can now do X% of data
science" is really "agents can now do X% of the gradeable fraction of data
science", and the gradeable fraction is not published.

That connects directly to `ch:as-specialized`'s finding — a domain's ceiling is set
by its verifier — and gives this part its through-line.

## Chapter plan

| ch | id | measurement to build |
|---|---|---|
| 177 | `aids-stack` | where time actually goes vs where automation lands; the gradeable fraction |
| 178 | `aids-text-to-sql` | schema/value grounding vs SQL generation; why BIRD is hard |
| 179 | `aids-agentic-eda` | exploration has no reference answer; what a wrong EDA costs downstream |
| 180 | `aids-automl` | search vs leakage; the validation-overfitting problem |
| 181 | `aids-autonomous` | the circular-reviewer problem; cost per result vs value per result |
| 182 | `aids-oversight` | where oversight goes, using ch:as-long-running's placement result |

Carry-through: `verifier-sets-the-ceiling` (ch168), `retry-needs-a-verifier`
(ch168), `placement-beats-frequency` (ch167), `revalidation-is-cheapest` (ch167),
`agent-errors-correlate` (ch169), and the text-to-SQL work connects to
`ch:mcp-primitives`' schema-grounding discussion.

## Not used

Several 2026 benchmark papers surfaced in search (DSGym 2601.16344, DARE-bench
2602.24288, DSAEval 2601.13591, DAComp 2512.04324). Not fetched or verified this
pass, so **not cited**. If a later chapter needs a 2026 data-point, verify first —
this session has already caught two withdrawn papers by checking.
