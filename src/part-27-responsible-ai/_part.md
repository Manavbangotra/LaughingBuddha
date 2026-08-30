---
id: part-27-intro
status: draft
---

## What this part is for

{{part:26}} asked what happens when somebody wants the system to fail. This part asks what
happens when nobody does, and it fails anyway — unfairly, opaquely, illegally, or under a
signature that means nothing.

**Responsible AI is treated here as an engineering subject with measurable properties, not as a
values statement.** Every chapter takes an obligation that is usually written in prose — be
fair, be explainable, respect privacy, comply, keep a human in the loop — and asks what quantity
it names, what instrument measures that quantity, and what the instrument's guarantee actually
covers.

The answers are consistently narrower than the prose. Not because the instruments are bad;
because each one answers a well-posed question that is adjacent to the one being asked.

> **The rule adopted for this part: name the question the instrument answers, and put it next to
> the number.** An attribution is not a counterfactual. A local surrogate is local. An epsilon
> without its accounting is a decoration. A signature attests to identity, not behaviour. A
> reviewer is not a control until four preconditions hold. Every chapter states the gap
> explicitly and prices it.

## Where the numbers land

| what | number | chapter |
|---|---|---|
| PPV at a shared threshold, 34% against 13% base rate | **0.740** vs **0.453** | {{ch:rai-bias}} |
| True-positive gap when equal PPV is enforced instead | **0.386** | {{ch:rai-bias}} |
| Criterion violation at equal base rates, then a 0.32 gap | **0** then **0.639** | {{ch:rai-bias}} |
| Disparity share with no model-side remedy | **$31\%$** | {{ch:rai-bias}} |
| Return per unit effort, threshold against data collection | **0.375** vs **0.020** | {{ch:rai-bias}} |
| Token cost and context for a Burmese user | **4.63×** and **0.22×** | {{ch:rai-bias}} |
| Attribution against intervention effect at high correlation | **0.211** vs **0.012** | {{ch:rai-interpretability}} |
| Local surrogate fit at the point, then at distance 0.5 | **0.99** then **0.24** | {{ch:rai-interpretability}} |
| Share of the decision a generated explanation accounts for | **$36\%$** | {{ch:rai-interpretability}} |
| Unfaithfulness found by reading, then by a swapped re-run | **$4\%$** then **$62\%$** | {{ch:rai-interpretability}} |
| Value of a confident unfaithful explanation | **−0.62** | {{ch:rai-interpretability}} |
| Posterior shift at epsilon 1, then epsilon 8 | **2.7×** then **2,981×** | {{ch:rai-privacy}} |
| Epsilon counted against epsilon actually spent | **0.5** vs **8,833.6** | {{ch:rai-privacy}} |
| Deletion completeness: nine destinations, then with weights | **0.3231** then **0** | {{ch:rai-privacy}} |
| Cost to remove one record from the weights, per request | **$1,400,000** | {{ch:rai-privacy}} |
| Three-year compliance cost, minimal against high risk | **$36,000** vs **$1,640,000** | {{ch:rai-regulation}} |
| The step from limited to high risk | **$1,473,000**, **19 weeks** | {{ch:rai-regulation}} |
| What designing below the tier boundary is worth | **$717,000** | {{ch:rai-regulation}} |
| Conformity evidence already produced by good engineering | **$62\%$** | {{ch:rai-regulation}} |
| Critical facts unrecoverable rather than expensive | **5 of 6** | {{ch:rai-regulation}} |
| Tasks where the team beats its better member | **2 of 5** | {{ch:rai-oversight}} |
| Model accuracy above which a fixed reviewer nets negative | **0.921** | {{ch:rai-oversight}} |
| Confidence per accuracy point: sources against a wrong explanation | **4.5** vs **115.4** | {{ch:rai-oversight}} |
| Complete verification of one item, against the budget | **16.0 min** vs **90 s** | {{ch:rai-oversight}} |
| Cost of rejecting against approving, to the reviewer | **79×** | {{ch:rai-oversight}} |
| Certainty a reviewer needs before saying no | **$94\%$** | {{ch:rai-oversight}} |

## The organising idea

**Every chapter finds a well-posed instrument answering a question adjacent to the one asked.**

{{part:26}}'s controls were correct and pointed at the wrong noun. This part's *measurements*
are correct and answer the wrong question — which is harder to notice, because the number is
right.

```text
   CHAPTER                  THE INSTRUMENT ANSWERS      THE QUESTION ASKED WAS
   ──────────────────────   ─────────────────────────   ────────────────────────────
   228 bias and fairness    is this classifier fair     is this outcome fair
   229 interpretability     what did the model use      what should the user change
   230 privacy              how much can be inferred    is the record gone
   231 regulation           which tier applies          is this system acceptable
   232 human oversight      did a human see it          did seeing it change anything
```

Read the right column. Each is the question a user, a regulator or an executive is actually
asking, and in each case the standard instrument returns a defensible number about something
else. **The failure is not measurement error. It is a category difference between the quantity
measured and the quantity that matters**, and no amount of precision in the left column closes
it.

## The three through-lines

**First: the constraint is upstream of the model, every time.**

| Chapter | Where the search for a fix goes | Where the constraint is |
|---|---|---|
| {{ch:rai-bias}} | a fairness-aware classifier | base rates, annotation, the tokenizer |
| {{ch:rai-interpretability}} | a better attribution method | the question the method answers |
| {{ch:rai-privacy}} | DP, deletion pipelines, filters | whether the data was in the training set |
| {{ch:rai-regulation}} | a compliance programme | the classification, and when the record was written |
| {{ch:rai-oversight}} | a stricter review gate | what the reviewer sees, and what saying no costs them |

**In every row the downstream instrument partially works and the upstream decision fully
determines the outcome** — and the upstream decision is cheap at the moment it is made and
unrecoverable afterwards.

**Second: three of the four gaps are closed by a second measurement, not a better one.**

{{ch:rai-interpretability}}'s unfaithful explanations are found at **62%** by re-running with
the order swapped and **4%** by reading them. {{ch:rai-privacy}}'s real epsilon appears only
when the sweeps and dashboards are metered — **8,833.6** against a reported **0.5**.
{{ch:rai-oversight}}'s reviewer value appears only from a blind re-adjudication producing catch
and override rates. In each case the missing number costs an afternoon and nobody has it.

**Third: an artefact that raises confidence is not thereby an artefact that raises accuracy —
and the two can move in opposite directions.**

This is the part's sharpest result and it appears twice.
{{ch:rai-interpretability}} prices a confident unfaithful explanation at **−0.62**, twice as bad
as silence. {{ch:rai-oversight}} measures the same object on a human: a plausible wrong
explanation raises reviewer confidence from **0.54 to 0.78** while dropping the catch rate to
**0.19** — **115.4** points of confidence per point of accuracy, against **4.5** for handing
over the source documents.

**The artefact everyone reaches for to make oversight meaningful is the one that measurably
makes it worse.**

## What this part does not settle

**No regulatory instrument is cited.** The frameworks in circulation are not arXiv preprints,
so under this book's verification rule they are absent. {{ch:rai-regulation}} treats the *shape*
common to them — risk tiers, obligations, conformity assessment, contemporaneous evidence — and
says explicitly that it is not an interpretation of any law.

**The fairness criterion is never chosen here.**
{{eq:three-fairness-criteria-cannot-hold-together}} converts the argument into a decision and
does not make it; which harm the application is trying to avoid is a product question, and this
part only insists that the choice be stated and the violations published.

**Oversight's value on out-of-distribution failure is unmeasured.** Everything in
{{ch:rai-oversight}} prices a reviewer against in-distribution accuracy, which on a good model
always recommends removing them — while the errors that matter most are the ones the metric
does not contain. That gap is named and not closed.

**And the correlation assumptions run the same way as {{part:26}}'s.** Deletion completeness
multiplies destination success rates; the precondition product multiplies four scores; the
epsilon composition assumes the accounted queries are the only queries. Each correction runs
against the optimistic reading.

## How to read this part

{{ch:rai-bias}} is the one to read first even if fairness is not your problem, because
{{eq:three-fairness-criteria-cannot-hold-together}} is the cleanest example in the book of a
theorem that ends an argument by making it a decision, and the same move is available in the
other four chapters.

If you are being asked for a compliance package this quarter: {{ch:rai-regulation}} and
{{ch:rai-privacy}} together, and start with the contemporaneous records — the **5 of 6**
unrecoverable facts are the only part of the work that expires.

If you have "a human reviews the output" written anywhere in a design document:
{{ch:rai-oversight}} is the chapter, and the four numbers it asks for — model accuracy,
human-alone accuracy, catch rate, override rate — are an afternoon of work that almost no team
has done.

And if you take one thing from the part: **the upstream decision is cheap now and unrecoverable
later**, in all five chapters, for five unrelated reasons.
