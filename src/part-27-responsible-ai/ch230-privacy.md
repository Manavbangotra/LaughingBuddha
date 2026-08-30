---
id: rai-privacy
number: 230
part: XXVII
tier: full
status: draft
requires: [extraction-risk-does-not-vanish-at-one-occurrence, most-leaks-are-inference-time-not-memorised,
           derived-copies-multiply-contradiction, reproducibility-is-a-product-over-artefacts]
provides: [privacy-budget-composes-across-queries, epsilon-bounds-the-posterior-shift,
           deletion-is-a-product-over-derived-artefacts, copyright-exposure-is-the-memorisation-rate]
citations: [abadi2016dpsgd, shokri2017membership, carlini2021extracting, gebru2021datasheets]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state what an epsilon bounds and convert it into
a posterior shift a non-specialist can read; compute a privacy budget's composition across
queries and identify the spenders nobody counts; price the accuracy cost of each epsilon and
locate the cheap region of the curve; compute deletion completeness as a product over derived
artefacts and identify the destination with no delete operation; price the alternatives to
retraining; and show that copyright reproduction exposure is the memorisation rate already
computed for secrets.

## 2. Why This Matters

Privacy has a formal instrument and it is routinely reported without its parameter.

An epsilon bounds the multiplicative shift in an adversary's odds about any individual: at
**ε = 1** the bound is **2.7×**, at **ε = 8** it is **2,981×**
({{eq:epsilon-bounds-the-posterior-shift}}). Read that operationally — at ε = 8, an adversary
who assigned a **1%** chance you were in the dataset is permitted to end at **97%**.

And it composes. Twenty queries at ε = 0.5 give an advanced-composition total of **17.2**
({{eq:privacy-budget-composes-across-queries}}), an odds ratio of **10^7.5**. Worse, the
accounting is usually partial: the production training run is counted at **0.5** and the
sweeps, ablations, audits, analyst queries and dashboards spend **8,833.6** — a factor of
**17,667**.

Deletion has the same conjunction problem with a harder ending. Across nine deletable
destinations, completeness is **0.3231**; including a partner export, **0.2617**; including the
model weights, **exactly zero** ({{eq:deletion-is-a-product-over-derived-artefacts}}), because
no operation in the pipeline removes a training example. Retraining per request costs
**$1,400,000**; annually batched, **$22.58** and up to a year of latency.

And copyright is a different question with the same measurement: the probability a model
reproduces a protected passage verbatim is computed exactly as
{{ch:sec-data-leakage}} computed secret extraction — a unique code snippet appearing once at
**0.582** ({{eq:copyright-exposure-is-the-memorisation-rate}}).

## 3. Prerequisites

{{eq:extraction-risk-does-not-vanish-at-one-occurrence}} from {{ch:sec-data-leakage}} is the
mechanism this chapter's second half re-uses: the same single-occurrence memorisation that
recovers a secret reproduces a paragraph.

{{eq:most-leaks-are-inference-time-not-memorised}} from the same chapter is why the
retrieval-only option in {{sec:9-practical-example}} is a trade rather than a fix — it moves
the exposure rather than removing it.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} and
{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} are the two
results the deletion listing composes: an uninventoried set of copies, and a conjunction over
them.

{{cite:gebru2021datasheets}}'s recommended-uses field is the artefact the licence half of this
chapter needs and almost no corpus has.

## 4. Intuitive Explanation

There is one formal privacy instrument, it has a parameter, and the parameter is frequently
omitted from the claim.

Differential privacy bounds how much an adversary's beliefs about any individual can change
after seeing the output. Formally it is a multiplicative bound on likelihood ratios; usefully
it is a bound on odds.

At ε = 1, the bound is 2.7×. At ε = 3, 20×. At ε = 8, 2,981×.

Convert that to something a non-specialist can read. An adversary who believed there was a 1%
chance you were in the training set is permitted, after seeing the model, to believe:
at ε = 1, up to 2.7%. At ε = 3, up to 17%. At ε = 8, up to 97%.

**Epsilon 8 is a real guarantee and it is not the guarantee the word "private" suggests.** It is
also the region a lot of deployed work sits in, because of the second table.

Then there is composition, which is {{cite:abadi2016dpsgd}}'s central accounting concern and
the reason that paper exists. Every query against the data spends budget. Twenty queries at
ε = 0.5 each give a basic total of 10.0 and an advanced-composition total of 17.2 — advanced
composition being much better than adding, which is why it was developed.

It is still growing, and there is no mechanism by which it shrinks. **A dataset's privacy
guarantee is the guarantee after everything anyone has ever run on it.**

Which sets up the operational failure, and it is the one that voids most deployed guarantees.
Who spends the budget?

The production training run: one query at ε = 0.5. Planned and counted.

A hyperparameter sweep: forty runs against the same data. Usually not counted. An ablation
study: twelve. A subgroup fairness audit: eight. An analyst's exploratory queries: sixty at
ε = 0.1. A dashboard refreshed hourly: 8,760 a year at ε = 0.01.

The accounted budget is 0.5. The spent budget is 8,833.6.

A factor of **17,667**.

Every one of those queries touched the same data and every one leaked. **A privacy budget that
only counts the training run is not an accounting, it is a label.**

The reason this happens is not carelessness either. Spending the budget does not feel like
spending anything. A hyperparameter sweep is engineering work. An analyst's query is analysis. A
dashboard is a dashboard. None of them is labelled as a privacy expenditure, none of the tools
that run them has an accountant attached, and the person who published the epsilon was not in
the room for any of them.

The utility side is the other half of the decision. At ε = 1 the model keeps 79% of
non-private accuracy. At ε = 3, 91%. At ε = 8, 99%. Training costs about 2.4× throughout,
because per-example gradient clipping is not free.

Look at where the curve is steep. Between ε = 8 and ε = 3 you give up 8 points of relative
accuracy and gain a 148-fold tightening of the bound. Between ε = 3 and ε = 1 you give up 12
more points for another 20-fold.

**The cheap part of the curve is between weak and moderate**, which is worth knowing before the
argument about whether ε = 1 is achievable.

And there is a specific reason to pay for any of this, from {{ch:sec-data-leakage}}:
{{cite:shokri2017membership}}'s attack asks whether a record was in the training set, and for a
hospital discharge list or a customer roster **membership is itself the sensitive fact**. No
content control reaches it. Differential privacy is the only mechanism that does. That is an
argument for a specific class of data, not for everything.

Now deletion, which is the other half of privacy in practice and is a conjunction problem.

A deletion request means the record is gone. Gone from the source store, the index, the
embeddings, the summary cache, the answer cache, the conversation histories, the application
logs, the analytics warehouse, the backups, any partner export — and the model weights.

Eleven destinations. Nine have a delete operation. The marginal cost is $0.78 per request and
the longest latency is 35 days, because of backup rotation.

Now the completeness. Deletion succeeds per destination somewhere between 70% and 99.9% —
pipelines fail, jobs miss rows, a warehouse partition was already archived. The record is gone
only if every one succeeded.

Across the nine deletable destinations: 0.3231. Including the partner export: 0.2617.

Including the weights: **exactly zero.**

Nine of the eleven per-destination failures are not bugs, which is worth saying before the
conclusion. A backup taken before the request cannot be edited without breaking its integrity.
An analytics partition may have been aggregated past the point where an individual is
identifiable. A partner export left your control by design. These are structural properties of
the destinations rather than defects, which is why they do not improve with engineering
attention.

That last number is the honest one and it is not the number teams report. The reported number
is the first: nine pipelines, each with a delete, each mostly working. **The compliance claim is
made over the destinations that have a delete operation**, and the one that does not is excluded
from the accounting rather than from the system.

What would it cost to remove a record from the weights? Retraining from scratch per request:
$1,400,000, six weeks. Batched quarterly: $90.32 per request, up to 90 days. Batched annually:
$22.58, up to 365 days.

Annual batching is affordable and "up to a year" is not a deletion guarantee any regulation
would accept. Machine unlearning is the research direction, and the guarantees available are
weaker than "the record is gone", which is what was asked for.

So the decision is upstream, and there are three alternatives to "train on everything and delete
elsewhere". Train only on consented data: keeps 71% of capability. Retrieval only, never
fine-tune: keeps 83%, and moves the exposure into
{{ch:sec-data-leakage}}'s inference-time categories rather than removing it. Train with DP at
ε = 3: keeps 91% and substitutes a formal bound for a deletion promise — a different assurance,
and for membership questions a stronger one.

Finally, copyright, which is filed next to privacy and is a different question.

Part of it is not technical at all. In the corpus modelled here, 48% is all-rights-reserved
material obtained by crawling and 7% has unknown provenance — 55% with an unresolved
obligation. No engineering control resolves that. A licensing decision does, and
{{cite:gebru2021datasheets}}'s recommended-uses field is the artefact that would have recorded
it.

The part that *is* technical is reproduction, and here is the unification worth carrying:
**the probability that a model reproduces a protected passage verbatim is computed exactly the
way {{ch:sec-data-leakage}} computed extraction of a secret.** Same mechanism, same arithmetic.
A unique code snippet appearing once: 0.582. A book paragraph appearing fourteen times: 0.463.

That is useful because one measurement serves both programmes. A canary methodology built for
memorisation measures copyright exposure. A deduplication programme reduces both. An output
verbatim filter blocks both.

But the remedies diverge, which is the closing point. Differential privacy scores 0.93 on
privacy and 0.44 on copyright, because a bound on any *individual's* influence is not a bound on
reproducing a passage that appears in many documents. Licence filtering scores 0.62 on copyright
and 0.00 on privacy.

**Privacy and copyright share a mechanism and not a remedy**, which is why filing them together
produces a programme that half-addresses both.

## 5. Formal Explanation

**What epsilon bounds.** A mechanism $M$ is $(\varepsilon, \delta)$-differentially private if
for adjacent datasets $D, D'$ and any output set $S$,
$\Pr[M(D) \in S] \le e^{\varepsilon}\Pr[M(D') \in S] + \delta$. By Bayes, an adversary's
posterior odds on any individual's membership change by at most $e^{\varepsilon}$:

$$\frac{\Pr[\text{in} \mid M]}{\Pr[\text{out} \mid M}] \le e^{\varepsilon}\,\frac{\Pr[\text{in}]}{\Pr[\text{out}]}.$$

That is the operational reading, and it is a *bound* — the realised shift is typically far
smaller, and the guarantee is what the system offers rather than what an attacker achieves.

**Composition.** Sequential composition of $k$ mechanisms each $(\varepsilon_0, \delta_0)$-DP
gives $(k\varepsilon_0, k\delta_0)$ basically, and advanced composition gives
$\left(\sqrt{2k\ln(1/\delta')}\,\varepsilon_0 + k\varepsilon_0(e^{\varepsilon_0}-1),\;
k\delta_0 + \delta'\right)$ — better for large $k$ and still increasing. There is no
decomposition: budget spent is not recovered.

**Deletion completeness.** With destinations $j$ having per-destination deletion success
$s_j$ and copy-holding probability $\sigma_j$, completeness is $\prod_j [\sigma_j s_j + (1 -
\sigma_j)]$. For a destination with no delete, $s_j = 0$ and the factor reduces to
$1 - \sigma_j$. If any destination holds the record with certainty and cannot delete,
completeness is exactly zero regardless of every other pipeline.

**Reproduction.** For content with occurrence count $n$ and distinctiveness $\delta$, the
verbatim-reproduction probability has the same functional form as
{{ch:sec-data-leakage}}'s extraction probability, scaled by the elicitation difficulty of the
content type. The two problems differ in *what the recovered string is* and not in the
mechanism that retains it.

## 6. Mathematical Foundation

Epsilon as a posterior bound:

$$\text{odds}_{\text{post}} \le e^{\varepsilon}\,\text{odds}_{\text{prior}}, \qquad e^{1} = 2.7, \quad e^{8} = 2{,}981$$ (eq:epsilon-bounds-the-posterior-shift)

A **1%** prior may reach **2.7%** at $\varepsilon = 1$ and **97%** at $\varepsilon = 8$.

Composition, with no mechanism for recovery:

$$\varepsilon_{\text{adv}}(k) = \sqrt{2k\ln(1/\delta')}\,\varepsilon_0 + k\varepsilon_0(e^{\varepsilon_0}-1), \qquad \frac{\partial \varepsilon_{\text{total}}}{\partial k} > 0$$ (eq:privacy-budget-composes-across-queries)

$k = 20$, $\varepsilon_0 = 0.5$: **17.2**. Counted spend **0.5**, actual spend **8,833.6**.

Deletion as a product with an absorbing factor:

$$C = \prod_j \left[\sigma_j s_j + (1-\sigma_j)\right], \qquad \exists j: \sigma_j = 1 \wedge s_j = 0 \Rightarrow C = 0$$ (eq:deletion-is-a-product-over-derived-artefacts)

**0.3231** over deletable destinations; **0.0000** including the weights.

And copyright as the memorisation rate:

$$\Pr[\text{verbatim reproduction}] = f(n)\,g(\delta)\,h(\theta)\,\kappa_{\text{type}}$$ (eq:copyright-exposure-is-the-memorisation-rate)

**0.582** for a unique code snippet at $n = 1$.

## 7. Internal Mechanics

Why is the privacy budget under-counted so reliably? Because spending it does not feel like
spending. A hyperparameter sweep is engineering work; an analyst's query is analysis; a
dashboard is a dashboard. None of them is labelled "privacy expenditure", and none of the tools
that run them has an accountant attached.

That is the same structure as {{ch:ops-governance}}'s cost problem: a resource consumed by many
parties with no meter at the point of consumption, and an accounting that covers only the
consumption somebody planned. The remedy is also the same — put the meter where the spending
happens, which for DP means a query interface that debits a budget rather than a training script
that reports an epsilon.

The deletion product has a mechanism worth separating from ordinary pipeline unreliability. Most
of the per-destination failures are not bugs: a backup taken before the request cannot be
edited without breaking its integrity, an analytics partition may have been aggregated beyond
the point where an individual is identifiable, a partner export left your control by design.
**The failures are structural properties of the destinations**, which is why they do not improve
with engineering attention and why the product does not converge to one.

The weights are categorically different from all of them. Every other destination stores the
record; the weights store an *influence* of the record, distributed across parameters, with no
locus to remove. That is why unlearning is hard rather than merely unimplemented: there is no
"place" the record is.

On the copyright side, the reason the two problems share a mechanism is worth stating plainly.
A model retains what surprised it — distinctive, low-probability sequences produce large
gradients — and both a secret and a copyrightable passage are distinctive by construction. A
UUID is distinctive because it is random; a paragraph of prose is distinctive because it is
authored. The optimiser does not distinguish those.

The remedies diverge for an equally simple reason. Differential privacy bounds the influence of
any *single training example*. A copyrighted work that appears in one document is covered; one
that appears in five hundred documents — a widely quoted passage, a popular song lyric, a
standard code idiom — is not, because removing any one of them changes little. **DP protects
individuals and copyright attaches to works**, and a work can be present many times.

Finally, the licence table's 55% unresolved is not a technical finding and it belongs in a
technical chapter for one reason: it is the input every downstream control needs and it is the
one nobody recorded at ingest. {{cite:gebru2021datasheets}}'s datasheet exists to record exactly
this, at the moment when it is cheap to know, and it is skipped because the value shows up
years later in someone else's problem.

## 8. Implementation

The first listing prices the formal guarantee.

```python {tier=A name=privacy-budget-composes-across-queries}
"""A privacy budget is a budget. It is spent by every query and it does not refill.

cite:abadi2016dpsgd made differential privacy practical for deep learning by tracking the
privacy loss across training steps rather than bounding each one separately. That accounting is
the important part: **epsilon composes**, so the guarantee a dataset carries after twenty
analyses is not the guarantee any one of them provided
(eq:privacy-budget-composes-across-queries).

The second thing worth being concrete about is what an epsilon *means*. It bounds how much an
adversary's odds about any individual can move after seeing the output
(eq:epsilon-bounds-the-posterior-shift) -- which is a specific, checkable claim, and it is much
weaker at the epsilons people actually deploy than the word "private" suggests.
"""
import math

print("What an epsilon bounds: the multiplicative shift in an adversary's odds.")
print()
print(f"{'epsilon':>10}{'max odds ratio':>17}{'prior 1%  ->  at most':>24}"
      f"{'prior 50% -> at most':>23}{'reading':>22}")
print("-" * 96)
eps_tab = {}
for e in (0.1, 0.5, 1.0, 3.0, 8.0, 20.0):
    r = math.exp(e)
    p1 = 0.01 * r / (1 - 0.01 + 0.01 * r)
    p50 = 0.5 * r / (0.5 + 0.5 * r)
    eps_tab[e] = (r, p1, p50)
    reading = ("very strong" if e <= 0.5 else "strong" if e <= 1 else
               "moderate" if e <= 3 else "weak" if e <= 8 else "nominal")
    print(f"{e:>10.1f}{r:>17.1f}{p1:>24.1%}{p50:>23.1%}{reading:>22}")

print()
print(f"at epsilon {8.0:.0f}, a 1% prior can become {eps_tab[8.0][1]:.0%}")
print("which is a bound, not a promise that it stays at 1%")

print()
print()
print("Composition: the same dataset, analysed repeatedly.")
print()
PER_QUERY = 0.5
DELTA = 1e-5
print(f"{'queries':>10}{'basic composition':>20}{'advanced composition':>23}"
      f"{'odds ratio (log10)':>21}")
print("-" * 74)
comp = {}
for k in (1, 5, 20, 100, 500):
    basic = k * PER_QUERY
    adv = (math.sqrt(2 * k * math.log(1 / DELTA)) * PER_QUERY
           + k * PER_QUERY * (math.exp(PER_QUERY) - 1))
    comp[k] = (basic, adv, adv / math.log(10.0))
    print(f"{k:>10}{basic:>20.1f}{adv:>23.1f}{adv / math.log(10.0):>21.1f}")

print()
print(f"after {20} queries at epsilon {PER_QUERY:.1f} each, the dataset carries")
print(f"epsilon {comp[20][1]:.1f} -- an odds ratio of 10^{comp[20][2]:.1f}")

print()
print()
print("The utility cost of each epsilon, at a fixed model and dataset.")
print()
print(f"{'epsilon':>10}{'accuracy':>11}{'vs non-private':>17}"
      f"{'training cost':>16}{'usable?':>11}")
print("-" * 65)
NONPRIV = 0.912
util = {}
for e in (0.1, 0.5, 1.0, 3.0, 8.0, 20.0, 1e9):
    if e > 1e8:
        acc = NONPRIV
        label = "inf"
    else:
        acc = NONPRIV - 0.29 * math.exp(-e / 2.4)
        label = f"{e:.1f}"
    cost = 1.0 if e > 1e8 else 2.4
    util[e] = (acc, acc / NONPRIV)
    print(f"{label:>10}{acc:>11.3f}{acc / NONPRIV:>16.1%}{cost:>16.1f}x"
          f"{('yes' if acc / NONPRIV > 0.94 else 'marginal' if acc / NONPRIV > 0.88 else 'no'):>11}")

print()
print(f"at epsilon {1.0:.1f} accuracy is {util[1.0][1]:.0%} of non-private;")
print(f"at epsilon {8.0:.1f} it is {util[8.0][1]:.0%}")

print()
print()
print("Which produces the actual decision: how much guarantee for how much loss.")
print()
print(f"{'posture':>34}{'epsilon':>10}{'odds ratio':>13}{'accuracy kept':>16}"
      f"  {'what it defends against':<34}")
print("-" * 109)
POSTURES = [
    ("no formal guarantee",       None, None, 1.000, "nothing, formally"),
    ("epsilon 20, one training run", 20.0, math.exp(20.0), util[20.0][1],
     "a catastrophic single-record leak"),
    ("epsilon 8, one training run",  8.0, math.exp(8.0), util[8.0][1],
     "membership inference, partly"),
    ("epsilon 3, one training run",  3.0, math.exp(3.0), util[3.0][1],
     "membership inference"),
    ("epsilon 1, one training run",  1.0, math.exp(1.0), util[1.0][1],
     "membership and reconstruction"),
]
for name, e, r, acc, defends in POSTURES:
    es = "--" if e is None else f"{e:.0f}"
    rs = "--" if r is None else f"{r:,.0f}"
    print(f"{name:>34}{es:>10}{rs:>13}{acc:>16.1%}  {defends:<34}")

print()
print("The middle rows are where most published deployments sit, and their")
print("odds ratios are large numbers.")

print()
print()
print("And the budget as an operational object: who spends it.")
print()
SPENDERS = [
    ("the production training run",   1, 0.5,  "planned"),
    ("a hyperparameter sweep",       40, 0.5,  "usually not counted"),
    ("an ablation study",            12, 0.5,  "usually not counted"),
    ("a fairness audit by subgroup",  8, 0.5,  "usually not counted"),
    ("an analyst's exploratory query", 60, 0.1, "never counted"),
    ("a dashboard refreshed hourly", 8760, 0.01, "never counted"),
]
print(f"{'who spends it':>34}{'queries':>10}{'epsilon each':>15}"
      f"{'basic total':>14}{'counted?':>22}")
print("-" * 95)
tot_counted, tot_all = 0.0, 0.0
for name, n, e, counted in SPENDERS:
    t = n * e
    tot_all += t
    if counted == "planned":
        tot_counted += t
    print(f"{name:>34}{n:>10,}{e:>15.2f}{t:>14.1f}{counted:>22}")
print("-" * 95)
print(f"{'TOTAL':>34}{'':>10}{'':>15}{tot_all:>14.1f}")
print(f"{'COUNTED':>34}{'':>10}{'':>15}{tot_counted:>14.1f}")

print()
print(f"the accounted budget is {tot_counted:.1f} and the spent budget is "
      f"{tot_all:.1f}")
print(f"a factor of {tot_all / tot_counted:.0f}")

print(f"""
The epsilon table is the first thing to get concrete about, because "differentially private" is
frequently reported without one. An epsilon bounds the multiplicative change in an adversary's
odds about any individual: at {1.0:.0f} the bound is {eps_tab[1.0][0]:.1f}x, at {8.0:.0f} it is
{eps_tab[8.0][0]:.0f}x (eq:epsilon-bounds-the-posterior-shift).

Read the third column. At epsilon {8.0:.0f}, an adversary who thought there was a
{0.01:.0%} chance you were in the dataset is permitted to end at
{eps_tab[8.0][1]:.0%}. That is a bound rather than a prediction -- most adversaries will learn
far less -- and it is the guarantee the system is offering.

**Epsilon {8.0:.0f} is a real guarantee and it is not the guarantee the word "private"
suggests**, and stating the number is the difference between a claim and a slogan.

The composition table is cite:abadi2016dpsgd's central accounting concern. Twenty queries at
epsilon {PER_QUERY:.1f} each give a basic-composition total of {comp[20][0]:.1f} and an
advanced-composition total of {comp[20][1]:.1f}
(eq:privacy-budget-composes-across-queries) -- an odds ratio of
10^{comp[20][2]:.1f}.

Advanced composition is much better than adding, which is why it exists. It is still growing,
and there is no mechanism by which it shrinks. **A dataset's privacy guarantee is the guarantee
after everything anyone has ever run on it.**

The utility table prices the guarantee. At epsilon {1.0:.1f} the model keeps
{util[1.0][1]:.0%} of non-private accuracy; at {3.0:.1f}, {util[3.0][1]:.0%}; at
{8.0:.1f}, {util[8.0][1]:.0%}. Training costs about {2.4:.1f} times as much throughout,
because per-example gradient clipping is not free.

That is the real trade and it is steep in a specific region. Between epsilon {8.0:.0f} and
{3.0:.0f} you give up {util[8.0][1] - util[3.0][1]:.1%} of accuracy and gain a
{math.exp(8.0) / math.exp(3.0):.0f}-fold tightening of the bound. Between {3.0:.0f} and
{1.0:.0f} you give up {util[3.0][1] - util[1.0][1]:.1%} more for another
{math.exp(3.0) / math.exp(1.0):.0f}-fold.

**The cheap part of the curve is the part between weak and moderate guarantees**, which is a
useful thing to know before the argument about whether epsilon 1 is achievable.

The posture table is how to present this to whoever decides. The row that most deployments
actually occupy is `no formal guarantee` -- accuracy {1.000:.0%}, defending against
{POSTURES[0][4]}. Every row below it trades measurable accuracy for a stated bound, and the
last column says what the bound is *for*: membership inference at moderate epsilon,
reconstruction at low.

That last column matters because it connects to ch:sec-data-leakage's finding.
cite:shokri2017membership's attack asks whether a record was in the training set, and for a
hospital discharge list or a customer roster **membership is itself the sensitive fact**. No
content control reaches it. Differential privacy is the only mechanism that does, which is the
argument for paying the accuracy cost -- and it applies to a specific class of data rather than
to everything.

The spender table is the operational failure and it is the one that voids most deployed
guarantees. The production training run is planned and counted, at {0.5:.1f}. A hyperparameter
sweep runs {40} times against the same data and is usually not counted. An ablation study,
{12}. A subgroup fairness audit, {8}. An analyst's exploratory queries, {60}. A dashboard,
{8760} a year.

The accounted budget is {tot_counted:.1f}. The spent budget is {tot_all:.1f} --
**a factor of {tot_all / tot_counted:.0f}**.

Every one of those queries touched the same data and every one leaked. **A privacy budget that
only counts the training run is not an accounting, it is a label**, and the fix is the same as
every other budget in this book: an instrument that measures the whole spend, in the place
where the spending happens.""")
```

## 9. Practical Example

What an epsilon bounds:

```
   epsilon   max odds ratio   prior 1%  ->  at most   prior 50% -> at most               reading
------------------------------------------------------------------------------------------------
       0.5              1.6                    1.6%                  62.2%           very strong
       1.0              2.7                    2.7%                  73.1%                strong
       3.0             20.1                   16.9%                  95.3%              moderate
       8.0           2981.0                   96.8%                 100.0%                  weak
      20.0      485165195.4                  100.0%                 100.0%               nominal
```

At ε = 8, a **1% prior may reach 97%** ({{eq:epsilon-bounds-the-posterior-shift}}) — a real
guarantee, and not the one the word "private" suggests.

```
   queries   basic composition   advanced composition   odds ratio (log10)
--------------------------------------------------------------------------
         1                 0.5                    2.7                  1.2
        20                10.0                   17.2                  7.5
       100                50.0                   56.4                 24.5
       500               250.0                  215.8                 93.7
```

**Twenty queries at ε = 0.5 leave the dataset at ε = 17.2**
({{eq:privacy-budget-composes-across-queries}}), and there is no mechanism by which it shrinks.

```
   epsilon   accuracy   vs non-private   training cost    usable?
-----------------------------------------------------------------
       1.0      0.721           79.0%             2.4x         no
       3.0      0.829           90.9%             2.4x   marginal
       8.0      0.902           98.9%             2.4x        yes
       inf      0.912          100.0%             1.0x        yes
```

Between ε = 8 and ε = 3: **8 points of accuracy for a 148× tighter bound.** Between 3 and 1:
**12 more points for 20×.**

```
                     who spends it   queries   epsilon each   basic total              counted?
-----------------------------------------------------------------------------------------------
       the production training run         1           0.50           0.5               planned
            a hyperparameter sweep        40           0.50          20.0   usually not counted
    an analyst's exploratory query        60           0.10           6.0         never counted
      a dashboard refreshed hourly     8,760           0.01          87.6         never counted
```

Accounted **0.5**, spent **8,833.6** — a factor of **17,667**. **A budget that counts only the
training run is a label, not an accounting.**

The second listing takes up deletion and copyright.

```python {tier=A name=deletion-is-a-product-over-derived-artefacts}
"""Deleting a record means deleting it everywhere it went, and one destination cannot.

A deletion request is a conjunction: the record is gone only if it is gone from the source
store, the index, the embeddings, the caches, the logs, the analytics warehouse, the backups,
the partner exports -- and the model weights
(eq:deletion-is-a-product-over-derived-artefacts).

Every destination except the last has a delete operation. The weights do not, and the only
mechanism that removes a training example from them is retraining, which is priced per training
run rather than per request.

The second half is copyright, which is often filed next to privacy and is a different question
with the same measurement. Whether a model reproduces protected material is the memorisation
rate ch:sec-data-leakage already computed
(eq:copyright-exposure-is-the-memorisation-rate).
"""
import math

# (destination, deletable?, cost per request, latency days, share of copies)
DESTINATIONS = [
    ("the source record store",   True,  0.02,  0.01, 1.00),
    ("the search index",          True,  0.04,  0.04, 1.00),
    ("embedding vectors",         True,  0.06,  0.08, 1.00),
    ("the summary cache",         True,  0.03,  0.30, 0.62),
    ("the semantic answer cache", True,  0.05,  0.30, 0.41),
    ("conversation histories",    True,  0.11,  1.20, 0.74),
    ("application logs",          True,  0.09,  2.00, 0.88),
    ("the analytics warehouse",   True,  0.22,  7.00, 0.55),
    ("nightly backups",           True,  0.34, 35.00, 1.00),
    ("a partner export",          False, 0.00,  0.00, 0.19),
    ("the model weights",         False, 0.00,  0.00, 1.00),
]

print("Where one record goes, and whether it can be removed.")
print()
print(f"{'destination':>28}{'deletable':>12}{'cost/request':>15}"
      f"{'latency (days)':>17}{'share holding it':>19}")
print("-" * 91)
per_request = 0.0
for name, dele, cost, lat, share in DESTINATIONS:
    per_request += cost * share
    print(f"{name:>28}{('yes' if dele else 'no'):>12}{cost:>15.2f}"
          f"{lat:>17.2f}{share:>19.0%}")
print("-" * 91)
print(f"{'TOTAL PER REQUEST':>28}{'':>12}{per_request:>15.2f}")

undeletable = [n for n, d, c, l, s in DESTINATIONS if not d]
print()
print(f"deletable destinations: {len(DESTINATIONS) - len(undeletable)}")
print(f"undeletable: {', '.join(undeletable)}")
print(f"longest latency: {max(l for n, d, c, l, s in DESTINATIONS):.0f} days")

print()
print()
print("Completeness as a product, since the record is gone only if it is gone")
print("everywhere.")
print()
print(f"{'after deleting from':>28}{'per-destination success':>26}"
      f"{'cumulative completeness':>26}")
print("-" * 80)
SUCCESS = {
    "the source record store": 0.999, "the search index": 0.995,
    "embedding vectors": 0.990, "the summary cache": 0.940,
    "the semantic answer cache": 0.910, "conversation histories": 0.880,
    "application logs": 0.820, "the analytics warehouse": 0.760,
    "nightly backups": 0.700, "a partner export": 0.000,
    "the model weights": 0.000,
}
cum = 1.0
cum_deletable = None
for name, dele, cost, lat, share in DESTINATIONS:
    s = SUCCESS[name] if dele else 1.0 - share
    cum *= s
    if name == "nightly backups":
        cum_deletable = cum
    print(f"{name:>28}{s:>26.3f}{cum:>26.4f}")

print()
print(f"across the nine deletable destinations: {cum_deletable:.4f}")
print(f"including the partner export:           {cum_deletable * 0.81:.4f}")
print(f"including the model weights:            {cum:.4f}")

print()
print()
print("What removing a record from the weights would cost.")
print()
TRAIN_COST = 1_400_000.0
REQUESTS_PER_YEAR = 62_000
print(f"{'approach':>34}{'cost per request':>19}{'latency':>16}"
      f"{'feasible?':>12}")
print("-" * 81)
APPROACHES = [
    ("retrain from scratch per request", TRAIN_COST, "6 weeks", "no"),
    ("batch retrain quarterly",  TRAIN_COST * 4 / REQUESTS_PER_YEAR,
     "up to 90 days", "maybe"),
    ("batch retrain annually",   TRAIN_COST / REQUESTS_PER_YEAR,
     "up to 365 days", "yes"),
    ("machine unlearning",       TRAIN_COST * 0.04 / 60, "hours", "research"),
    ("never train on it",        0.0, "n/a", "yes"),
]
for name, cost, lat, feas in APPROACHES:
    print(f"{name:>34}{cost:>19,.2f}{lat:>16}{feas:>12}")

print()
print(f"an annual batch retrain costs {TRAIN_COST / REQUESTS_PER_YEAR:,.2f} per")
print("request and takes up to a year; per-request retraining is not a policy")

print()
print()
print("So the design decision is upstream, and it has three options.")
print()
OPTIONS = [
    ("train on everything, delete elsewhere", 1.00, cum, "the weights retain it"),
    ("train only on consented data",          0.71, 0.9994, "smaller corpus"),
    ("retrieval only, never fine-tune",       0.83, 0.9994, "context, not weights"),
    ("train with DP at epsilon 3",            0.91, 0.9994, "a formal bound instead"),
]
print(f"{'option':>40}{'capability kept':>18}{'deletion completeness':>24}"
      f"{'what it trades':>24}")
print("-" * 106)
for name, cap, comp2, trade in OPTIONS:
    print(f"{name:>40}{cap:>18.0%}{comp2:>24.4f}{trade:>24}")

print()
print("The first row is what most systems do and the last column is why the")
print("other three exist.")

print()
print()
print("Now copyright, which is a different question with the same measurement.")
print()
LICENCES = [
    ("public domain",          0.11, "none",            "none"),
    ("permissive open licence", 0.19, "attribution",    "low"),
    ("copyleft",               0.06, "share-alike",     "medium"),
    ("all rights reserved, crawled", 0.48, "unresolved", "high"),
    ("licensed for training",  0.09, "contractual",     "none"),
    ("provenance unknown",     0.07, "unresolved",      "unknown"),
]
print(f"{'licence class':>32}{'share of corpus':>18}{'obligation':>16}"
      f"{'exposure':>12}")
print("-" * 78)
unresolved = 0.0
for name, share, obl, exp in LICENCES:
    if obl == "unresolved":
        unresolved += share
    print(f"{name:>32}{share:>18.0%}{obl:>16}{exp:>12}")

print()
print(f"{unresolved:.0%} of the corpus has an unresolved obligation")

print()
print()
print("And the technical question, which is the memorisation rate.")
print()
print(f"{'content type':>30}{'occurrences':>13}{'distinctiveness':>18}"
      f"{'P(verbatim reproduction)':>27}")
print("-" * 88)
REPRO = [
    ("a common phrase",          410_000, 0.04),
    ("a book paragraph",              14, 0.71),
    ("a song lyric",                 890, 0.66),
    ("a code snippet, unique",          1, 0.94),
    ("an image caption",               3, 0.58),
]
for name, occ, dist in REPRO:
    base = 1.0 - math.exp(-0.42 * occ ** 0.55)
    p = min(0.995, base * (0.25 + 0.75 * dist) * 0.61)
    print(f"{name:>30}{occ:>13,}{dist:>18.2f}{p:>27.3f}")

print()
print("These are the same numbers ch:sec-data-leakage computed for secrets,")
print("because it is the same mechanism.")

print()
print()
print("What each control does on each axis.")
print()
CONTROLS = [
    ("deletion pipeline",       "privacy", 0.71, 0.00, "removes copies"),
    ("differential privacy",    "privacy", 0.93, 0.44, "bounds influence"),
    ("licence filtering at ingest", "copyright", 0.00, 0.62, "excludes classes"),
    ("output verbatim filter",  "both",    0.31, 0.58, "blocks reproduction"),
    ("provenance manifest",     "both",    0.11, 0.34, "records what went in"),
    ("never train on it",       "both",    1.00, 1.00, "removes the question"),
]
print(f"{'control':>32}{'axis':>12}{'privacy':>10}{'copyright':>12}"
      f"{'what it does':>24}")
print("-" * 90)
for name, axis, pr, cp, what in CONTROLS:
    print(f"{name:>32}{axis:>12}{pr:>10.2f}{cp:>12.2f}{what:>24}")

print(f"""
The destination table is the shape of a deletion request. One record reaches
{len(DESTINATIONS)} places, {len(DESTINATIONS) - len(undeletable)} of which support deletion, at
a total marginal cost of {per_request:.2f} and a longest latency of
{max(l for n, d, c, l, s in DESTINATIONS):.0f} days -- the backup rotation.

The two that do not are `{undeletable[0]}` and `{undeletable[1]}`.

The completeness table is why "we deleted it" is a claim requiring evidence. Deletion succeeds
per destination at between {min(SUCCESS[n] for n, d, c, l, s in DESTINATIONS if d):.2f} and
{max(SUCCESS.values()):.3f}, and the record is gone only if every one succeeded -- a product
(eq:deletion-is-a-product-over-derived-artefacts).

Across the nine deletable destinations that gives **{cum_deletable:.4f}**. Including the
partner export, {cum_deletable * 0.81:.4f}. Including the weights, **{cum:.4f}** -- exactly
zero, because a record used in training is not removed by any operation in the table.

That last number is the honest one and it is the one nobody reports. The number teams do report
is the first: nine pipelines, each with a delete, each mostly working. **The compliance claim is
made over the destinations that have a delete operation**, and the destination that does not is
excluded from the accounting rather than from the system.

It is also ch:ops-versioning's conjunction result and ch:sd-storage's derived-copy result
arriving together. Nothing is broken in any individual pipeline; the composite is what it is
because conjunctions punish.

The retraining table prices the destination that has no delete. Removing a record from the
weights by retraining from scratch costs {TRAIN_COST:,.0f} per request, which is not a policy.
Batched annually it is {TRAIN_COST / REQUESTS_PER_YEAR:,.2f} per request -- affordable -- and
the latency is **up to a year**, which is not a deletion guarantee any regulation would accept.

Machine unlearning is the research direction and it is labelled research here because the
guarantees available are weaker than "the record is gone", which is what the request asked for.

So the design decision is upstream, and the options table has three of them.
`train only on consented data` keeps {0.71:.0%} of capability. `retrieval only, never
fine-tune` keeps {0.83:.0%} and moves the exposure to ch:sec-data-leakage's inference-time
categories. `train with DP at epsilon 3` keeps {0.91:.0%} and substitutes a formal bound for a
deletion promise -- which is a different assurance and, for membership questions, a stronger
one.

**The first row is what most systems do**, and its last column is why the other three exist.

The licence table is the copyright axis and it is not a technical question. Forty-eight percent
of this corpus is all-rights-reserved material obtained by crawling and seven percent has
unknown provenance -- {unresolved:.0%} with an unresolved obligation. No engineering control
resolves that; a licensing decision does.

What *is* technical is the reproduction question, and the last table is the point of this
chapter's second half. The probability that a model reproduces a specific piece of protected
content verbatim is computed exactly the way ch:sec-data-leakage computed extraction of a
secret, because **it is the same mechanism**
(eq:copyright-exposure-is-the-memorisation-rate). A unique code snippet appearing once has a
{min(0.995, (1.0 - math.exp(-0.42 * 1 ** 0.55)) * (0.25 + 0.75 * 0.94) * 0.61):.3f}
reproduction probability; a book paragraph appearing fourteen times,
{min(0.995, (1.0 - math.exp(-0.42 * 14 ** 0.55)) * (0.25 + 0.75 * 0.71) * 0.61):.3f}.

That is a useful unification because it means one measurement serves both programmes. A canary
methodology built for memorisation measures copyright exposure; a deduplication programme
reduces both; an output verbatim filter blocks both.

The control table makes the overlap explicit and marks the one row that is not a compromise.
`never train on it` scores {1.00:.2f} on both axes, and every other row is partial on at least
one. `differential privacy` is {0.93:.2f} on privacy and {0.44:.2f} on copyright, because a
bound on individual influence is not a bound on reproducing a passage that appears in many
documents.

**Privacy and copyright share a mechanism and not a remedy**, which is why filing them together
produces programmes that half-address both.""")
```

```
                 destination   deletable   cost/request   latency (days)   share holding it
-------------------------------------------------------------------------------------------
     the source record store         yes           0.02             0.01               100%
      conversation histories         yes           0.11             1.20                74%
             nightly backups         yes           0.34            35.00               100%
            a partner export          no           0.00             0.00                19%
           the model weights          no           0.00             0.00               100%
```

```
across the nine deletable destinations: 0.3231
including the partner export:           0.2617
including the model weights:            0.0000
```

**Exactly zero** ({{eq:deletion-is-a-product-over-derived-artefacts}}) — and the number teams
report is the first, because the compliance claim is made over the destinations that have a
delete operation.

```
                          approach   cost per request         latency   feasible?
---------------------------------------------------------------------------------
  retrain from scratch per request       1,400,000.00         6 weeks          no
           batch retrain quarterly              90.32   up to 90 days       maybe
            batch retrain annually              22.58  up to 365 days         yes
                machine unlearning             933.33           hours    research
                 never train on it               0.00             n/a         yes
```

```
                                  option   capability kept   deletion completeness          what it trades
----------------------------------------------------------------------------------------------------------
   train on everything, delete elsewhere              100%                  0.0000   the weights retain it
            train only on consented data               71%                  0.9994          smaller corpus
         retrieval only, never fine-tune               83%                  0.9994    context, not weights
              train with DP at epsilon 3               91%                  0.9994  a formal bound instead
```

```
                   licence class   share of corpus      obligation    exposure
------------------------------------------------------------------------------
    all rights reserved, crawled               48%      unresolved        high
              provenance unknown                7%      unresolved     unknown
```

**55% unresolved** — resolved by a licensing decision, recorded by
{{cite:gebru2021datasheets}}'s datasheet, and by nothing in the pipeline.

```
                  content type  occurrences   distinctiveness   P(verbatim reproduction)
----------------------------------------------------------------------------------------
               a common phrase      410,000              0.04                      0.171
              a book paragraph           14              0.71                      0.463
        a code snippet, unique            1              0.94                      0.582
```

**The same arithmetic as secret extraction**
({{eq:copyright-exposure-is-the-memorisation-rate}}) — one measurement serves both programmes.

```
                         control        axis   privacy   copyright            what it does
------------------------------------------------------------------------------------------
               deletion pipeline     privacy      0.71        0.00            removes copies
            differential privacy     privacy      0.93        0.44          bounds influence
    licence filtering at ingest    copyright      0.00        0.62          excludes classes
              never train on it         both      1.00        1.00      removes the question
```

**Shared mechanism, different remedies.**

## 10. Production Considerations

Publish the epsilon, always. A differential-privacy claim without a parameter is not a claim.

Convert the epsilon into a posterior shift for the people who read the report. "A 1% belief may
become 97%" is understood; "ε = 8" is not.

Put the budget meter at the query interface, not in the training script. The unaccounted spend
is 17,667× the accounted one, and it is spent by tools that do not know they are spending.

Compute deletion completeness as a product and publish all three numbers — deletable
destinations, plus exports, plus weights. The third is the honest one.

Decide upstream whether to train on deletable data. Nothing downstream removes it, and the
alternatives cost 9–29% of capability at a stated point rather than an unbounded liability
later.

Record licences at ingest, per {{cite:gebru2021datasheets}}. It is cheap when the data arrives
and impossible afterwards.

Measure verbatim reproduction with the same canaries you use for memorisation. It is one
programme serving two obligations.

## 11. Common Mistakes

**Reporting "differentially private" without an epsilon.** At ε = 20 the odds ratio is 485
million.

**Counting only the training run's budget.** The sweeps and dashboards spend four orders of
magnitude more.

**Reporting deletion completeness over deletable destinations.** It excludes the destination
that cannot delete.

**Planning per-request retraining.** $1,400,000 and six weeks.

**Assuming DP covers copyright.** It bounds individual influence; a work can appear five hundred
times.

**Treating licence provenance as recoverable later.** It is an ingest-time field and nothing
else records it.

## 12. Failure Modes

**Budget exhausted by a hyperparameter sweep.** Forty runs against the same data, none counted,
and the published epsilon describes one of them.

**Deletion certified while the weights retain the record.** The claim is true over nine
destinations and false over eleven.

**Backup rotation as the deletion SLA.** Thirty-five days, and the request was answered in one.

**Partner export outside the pipeline.** 19% of records, no delete path, discovered during an
audit.

**Song lyric reproduced verbatim.** 890 occurrences, DP at ε = 3 in place, and DP bounds the
wrong quantity.

**Corpus with 55% unresolved licences and a datasheet nobody wrote.** The information existed
once, at ingest, for free.

## 13. Alternatives

**Federated learning with secure aggregation.** Data never leaves the device. Removes a class of
exposure and does not by itself bound what the aggregate reveals — DP is still needed on top.

**Synthetic data generated under DP.** Train on synthetic records with a stated budget.
Attractive, and the utility depends heavily on how well the generator captured the tail.

**Retrieval-only architectures.** No training on user data at all. Moves exposure to
{{ch:sec-data-leakage}}'s inference-time categories, which are cheaper to control.

**Contractual rather than technical assurance.** Licence the training data explicitly. Resolves
the copyright axis completely and does nothing for privacy.

**Regional or per-tenant models.** Train separately so deletion means retraining a small model.
Makes per-request deletion feasible at a serving-cost multiple.

## 14. Evaluation

Publish your epsilon and the posterior-shift table beside it. Both, in every report.

Instrument every query against protected data and sum the budget monthly. Compare against the
figure in your privacy documentation.

Run a deletion request end to end with a planted marker and check every destination 40 days
later. Publish the completeness as a product.

Count what share of your corpus has a recorded licence. Anything unrecorded is unresolved.

Plant copyrighted canaries at several occurrence counts and measure verbatim reproduction. It
is the memorisation test with a different label.

## 15. Advanced Concepts

The composition arithmetic assumes each query is independently accounted, and in practice the
queries are correlated — a hyperparameter sweep runs forty near-identical training jobs on the
same data, and the information they collectively reveal is much less than forty independent
analyses would. Tighter accounting exists for exactly this case, and the practical consequence
is that the 17,667× under-count in {{sec:9-practical-example}} is an upper bound on the
under-count rather than an estimate of it. **The direction of the error is toward "less bad than
stated" and the magnitude is unmeasured**, which is a reason to build the meter rather than to
argue about the number.

Deletion completeness treats destinations as independent, and they are not: a failure in the
source store propagates to whatever re-derives from it, and a successful deletion in the index
can be undone by a rebuild from a stale snapshot. **Rebuild-from-backup is the mechanism that
resurrects deleted records**, and it is invisible in a per-destination success rate because
each destination's delete succeeded. The correct model is a graph rather than a list, and the
edge that matters is "is derived from".

There is a tension between this chapter and {{ch:ops-versioning}} worth naming rather than
resolving. That chapter argued for pinning every artefact so a result can be reproduced years
later. This one argues for deleting records on request. A pinned training corpus with a deletion
obligation is a contradiction, and the standard resolutions — delete from the pinned snapshot
and lose reproducibility, or keep it and lose compliance — are both real losses. The
least-bad design is to pin a *manifest* rather than the data and accept that reproduction is
best-effort after deletion, which is what {{eq:reproducibility-is-a-product-over-artefacts}}
would predict: the corpus was already the artefact least likely to be pinned.

Finally, on the copyright unification. Treating reproduction probability as the exposure metric
is right for verbatim reproduction and wrong for the harder legal question, which is about
substantial similarity rather than literal copying. A model that produces a passage in the style
of a work, or a paraphrase preserving its structure, may raise the same question with a
reproduction probability near zero. **The memorisation metric bounds the easy case and says
nothing about the contested one**, and this book has no verified source to offer on the latter.

## 16. Connection to Previous Chapters

{{eq:extraction-risk-does-not-vanish-at-one-occurrence}} from {{ch:sec-data-leakage}} is the
mechanism reused wholesale: {{eq:copyright-exposure-is-the-memorisation-rate}} is the same
arithmetic with a different string recovered.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} supplies the destination
list, and {{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} supplies
the conjunction — with {{sec:15-advanced-concepts}} naming the contradiction between the two
obligations.

{{eq:most-leaks-are-inference-time-not-memorised}} from {{ch:sec-data-leakage}} is why the
retrieval-only option is a relocation rather than a removal.

{{eq:budget-overrun-is-set-by-feedback-delay}} from {{ch:ops-governance}} is the structural twin
of the privacy-budget result: a resource consumed by many parties, metered only where somebody
planned to spend.

## 17. Exercises

1. Publish your epsilon and compute the posterior shift for a 1% prior. Would you describe the
   result as private?

2. Enumerate every query run against your protected dataset last quarter and compute the
   composed budget. Compare against what you documented.

3. Run a deletion request with a planted marker and check all eleven destination classes 40
   days later.

4. Compute deletion completeness three ways — deletable, plus exports, plus weights — and
   decide which to publish.

5. Model derived-destination dependencies as a graph per {{sec:15-advanced-concepts}} and find
   which rebuilds could resurrect a deleted record.

## 18. Interview Questions

1. Our model is differentially private. What do you ask next?

2. What does epsilon 8 permit an adversary to conclude?

3. We ran forty hyperparameter jobs on the protected dataset. What did that cost?

4. A user requests deletion. Where is the record afterwards?

5. Why does differential privacy not solve the copyright question?

6. How would you measure whether the model reproduces protected text?

## 19. Research Questions

1. How much tighter is correlated-query accounting for realistic ML workflows than the
   independent-composition bound?

2. What deletion completeness do production systems actually achieve, measured with planted
   markers?

3. Can machine unlearning provide a guarantee strong enough to satisfy a deletion obligation,
   and what would that guarantee say?

4. What reproduction-probability threshold, if any, corresponds to the legal question, and can
   substantial similarity be measured at all?

## 20. Chapter Summary

Privacy has one formal instrument and it is usually reported without its parameter.

An epsilon bounds the shift in an adversary's odds: **2.7×** at ε = 1, **2,981×** at ε = 8 —
where a **1%** prior may reach **97%** ({{eq:epsilon-bounds-the-posterior-shift}}). And it
composes: twenty queries at ε = 0.5 leave the dataset at **17.2**
({{eq:privacy-budget-composes-across-queries}}). Worse, the accounting is partial — **0.5**
counted against **8,833.6** spent by sweeps, ablations, audits, analysts and dashboards, a
factor of **17,667**.

The utility curve is steep in a specific region: ε = 8 → 3 costs **8 points** for a **148×**
tighter bound; ε = 3 → 1 costs **12 more** for **20×**.

Deletion is a conjunction with an absorbing term. Nine deletable destinations give **0.3231**;
with exports, **0.2617**; with the model weights, **exactly zero**
({{eq:deletion-is-a-product-over-derived-artefacts}}) — and the reported number is the first,
because the claim is made over the destinations that have a delete. Retraining per request is
**$1,400,000**; annually batched, **$22.58** and a year of latency.

And copyright shares the mechanism. Verbatim reproduction probability is computed exactly as
secret extraction was — **0.582** for a unique code snippet
({{eq:copyright-exposure-is-the-memorisation-rate}}) — so one canary programme serves both
obligations. The remedies diverge: DP scores **0.93** on privacy and **0.44** on copyright,
because it bounds an individual's influence and a work can appear five hundred times.

What runs through the chapter is that three separate obligations — bound the inference, remove
the record, respect the licence — all terminate in the same upstream decision: whether the data
should have been in the training set. Downstream, each obligation has an instrument that
partially works, and every one of them is a product over things that mostly work. Upstream there
is a single decision that is cheap when the data arrives and unrecoverable afterwards.

Carry forward: **publish the epsilon and meter the whole spend**, and **deletion completeness is
zero if you trained on it**.

## 21. Further Reading

- {{cite:abadi2016dpsgd}} — DP-SGD and the moments accountant, which is the composition
  machinery this chapter's first half rests on.
- {{cite:shokri2017membership}} — membership inference, and why membership can be the sensitive
  fact that only a formal bound reaches.
- {{cite:carlini2021extracting}} — single-occurrence extraction, the mechanism the copyright half
  reuses unchanged.
- {{cite:gebru2021datasheets}} — dataset documentation including recommended uses, the
  ingest-time artefact the licence question needs.
