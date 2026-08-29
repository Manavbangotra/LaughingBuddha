---
id: part-20-intro
status: final
---

## What this part is for

{{part:19}} was about where tools come from. This part is about a specific
application of them — agents that do analysis — and it is the part where the
distance between the demonstrations and the measurements is widest.

**The hazard here is that data science automation demos extremely well.** An agent
that loads a dataset, produces twelve charts, fits three models and writes a summary
looks like it did the job. Whether it did is a question with no answer key, which is
exactly why the demo is so persuasive.

> **The organising claim, stated once and then measured six times: benchmarks
> measure what can be graded, and what can be graded is not where the work is.**
> Modelling receives $3.3\times$ its share of practitioner time in benchmark
> attention; question-framing receives $0.16\times$. Only about a third of a
> practitioner's day is gradeable at all, which means two thirds of the work sits in
> a region where no benchmark reports progress in either direction.

## Where the published numbers actually sit

Four results, from four groups, using four grading methods:

| what | number | source |
|---|---|---|
| Text-to-SQL, realistic databases | **$40.08\%$** execution accuracy vs **$92.96\%$** human | {{cite:li2023bird}} |
| Agentic data science tasks | **$30.5\%$** best-model accuracy | {{cite:huang2024dacode}} |
| ML engineering, real Kaggle leaderboards | **$16.9\%$** of competitions at bronze | {{cite:chan2024mlebench}} |
| End-to-end automated papers | **under \$15** per paper | {{cite:lu2024aiscientist}} |

They agree on a shape: impressive demonstrations, low completion rates. And all
four measure inside the gradeable third.

## The organising idea

**Every result in this part follows from where the verifier is.**
{{ch:as-specialized}} found a domain's ceiling set by whether the agent can check
its own work; applied across the stages of an analysis, that single idea explains
the benchmark distribution, the error propagation, the failure modes and the
oversight allocation.

```text
   CHAPTER                   THE DECISION IT OWNS       WHAT DECIDES IT
   ───────────────────────   ────────────────────────   ─────────────────────────
   177 the stack             what to automate first     the Amdahl share
   178 text-to-SQL           where to invest            grounding, not syntax
   179 agentic EDA           how much to explore        whether there is a holdout
   180 AutoML                whether to trust a score   the search size
   181 autonomous work       whether to trust a judge   generator-judge correlation
   182 oversight             where the human goes       where nothing else checks
```

**And a through-line the part did not set out to find.** Three chapters
independently arrived at the same missing quantity:

| Chapter | The search | The unreported denominator |
|---|---|---|
| {{ch:aids-agentic-eda}} | exploration | how many comparisons were run |
| {{ch:aids-automl}} | model selection | how many configurations were tried |
| {{ch:aids-autonomous}} | self-judged generation | how correlated the judge is |

In each case the reported number is a maximum, or an acceptance rate, that depends
on a quantity the automation removed from view — **and in each case a holdout the
search never touched is the only remedy that does not need the missing number.**

## Ten things worth knowing before you start

**Fully automating modelling gives a $1.12\times$ project speedup.** Amdahl's
formula contains only the time share, not the difficulty or the benchmark
prominence — and modelling's share is small precisely because it was checkable
enough to become efficient first. Automating everything gradeable, perfectly, is
under $2\times$.

**Text-to-SQL's gap is not SQL.** On a clean benchmark schema, SQL construction is
$41.9\%$ of failures; on a realistic database it is $5.7\%$, and grounding is
$94.2\%$. Perfect SQL generation buys $+0.4$ points; perfect grounding buys $+47.8$.
A human analyst's $92.96\%$ reflects two years of tenure, not better SQL.

**$61\%$ of wrong queries run cleanly and return rows** — because a grounding error
*is* a well-formed query about the wrong thing. Checking for an empty result catches
$21$ points at zero cost, since the query has already run.

**More exploration finds no more real effects.** Across ten to nine hundred
comparisons, true findings stayed at $4.3$ while false ones went from $0.18$ to
$44.6$. A human exploring for an afternoon reports $5.0$ findings of which $4.3$ are
real; an agent exploring for an hour reports $49.1$ of which $4.3$ are real.

**Defensible cleaning choices moved a true effect of $0.30$ across a range from
$-0.050$ to $0.570$** — the low end with the opposite sign, every path defensible.
That spread is $6.1\times$ wider than the confidence interval that gets reported, and
because analytic spread is additive while sampling error falls as $1/\sqrt{N}$, more
data makes it relatively worse.

**Automated feature engineering is structurally a search for leakage**, since
leaking features have the highest validation lift and the search ranks by validation
lift. A greedy search scored $0.999$ validating and $0.620$ deployed — the baseline.

**A leakage guard makes your number worse and your model better**: $+0.202$
deployed against $-0.177$ reported. Any team measured on validation is measured
against its own model quality.

**About a third of AutoML's apparent gain is real**, constant from five
configurations to two thousand, because the share is a signal-to-noise ratio with no
search size in it. And a noisier validation estimate does not merely inflate the
score — it *selects a worse configuration*.

**A self-judged pipeline's acceptance rate measures the correlation between its
generator and its judge.** As shared bias rose, acceptance climbed from $32.7\%$ to
$84.2\%$ while the share that was actually sound stayed at $14.5\%$. Neither a
better judge nor more judges rescues it.

**Generation costs \$15 and verification costs about \$2,000 per usable result.**
The famous figure prices the half that got cheap; the half that did not is the
system.

## What this part deliberately does not cover

**Statistics.** {{part:1}}'s. This part assumes hypothesis testing, cross-validation
and multiple comparisons rather than teaching them, and uses them to ask what
changes when a machine does the searching.

**Data engineering.** Much of {{ch:aids-stack}}'s cleaning stage is a symptom of
upstream systems, and fixing those has the same Amdahl numerator with none of the
verification problems. It is the right investment and it is not this book's subject.

**The redesign effect.** {{cite:testini2025dsautomation}}'s third gap — the value of
questions that became cheap enough to ask — is conceded repeatedly and measured
nowhere here. Every listing prices a fixed workload done better, and the omitted
effect points the opposite way from most of this part's caution.

**Domain-specific analysis.** The results are about the shape of the pipeline, not
about any field's methods.

## How to read it

{{ch:aids-stack}} is the foundation and the rest of the part is its consequences.
Its two questions — where does the time go, and where is the verifier — generate
every subsequent chapter.

{{ch:aids-agentic-eda}} and {{ch:aids-automl}} are one argument in two settings, and
the second is more convincing if you have read the first: exploration and model
selection are the same search with the same unreported denominator and the same fix.

{{ch:aids-autonomous}} is where the part's machinery meets a specific published
system, and it should be read alongside {{cite:lu2024aiscientist}} itself rather
than instead of it.

{{ch:aids-oversight}} is the practical chapter and can be read first by anyone who
wants the allocation rule without the derivation.

> **One thing to notice on a second reading**: {{ch:aids-text-to-sql}} concludes
> that grounding is fixed by writing conventions down executably;
> {{ch:aids-agentic-eda}} concludes that cleaning is fixed by specifying the policy;
> {{ch:aids-oversight}} concludes that the ungradeable stages are not intrinsically
> unmeasurable but **unspecified**. **All three are the same recommendation.** The
> standards were always there — carried in an experienced analyst's head, never
> written down because they never had to be. Automation is what makes the writing
> down unavoidable, and doing it converts a stage from ungradeable to gradeable
> permanently.
