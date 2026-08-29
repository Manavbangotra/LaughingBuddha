---
id: aids-agentic-eda
number: 179
part: XX
tier: full
status: draft
requires: [check-strong-build-weak, gradeable-is-not-representative,
           meaning-lives-outside-the-schema, habituation]
provides: [exploration-is-a-search, more-exploration-finds-only-noise,
           holdout-beats-correction, cleaning-choices-move-the-answer,
           speed-makes-the-multiverse-free]
citations: [testini2025dsautomation, huang2024dacode, lu2024aiscientist,
            li2023bird, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why more exploration finds
more false results and no more true ones; state why a held-out re-test dominates a
multiple-comparisons correction for exploratory work specifically; identify the
input that every correction requires and that automation destroys; quantify how
much a defensible cleaning pipeline can move an estimate; and explain why the same
speed that makes exploratory automation dangerous makes multiverse reporting free.

## 2. Why This Matters

{{ch:aids-stack}} found the exploration stage with almost the weakest verifier in
the pipeline — $15\%$ detection, against $80\%$ at the model stage — and
{{eq:check-strong-build-weak}} said the weak stages are where verifier-building
pays. This chapter is about what happens when you automate one of them instead.

The result is not that the agent explores badly. {{sec:9-practical-example}} finds
something more awkward: **more exploration finds no more real effects and many more
spurious ones.** Across ten to nine hundred comparisons on the same data, true
findings stay flat at about $4.3$ while false findings go from $0.18$ to $44.6$,
and precision falls from $96.4\%$ to $8.9\%$
({{eq:more-exploration-finds-only-noise}}).

In the units that matter: a human exploring for an afternoon reports $5.0$ findings
of which $4.3$ are real; an agent exploring for an hour reports $49.1$ of which
$4.3$ are real. **Same discoveries, ten times the noise.**

The standard remedies work and one of them is much better suited here.
Bonferroni correction at four hundred comparisons reaches $95.9\%$ precision and
discards $78\%$ of the genuine findings; a held-out re-test keeps $85\%$ of them at
$81.1\%$ precision ({{eq:holdout-beats-correction}}). Since the purpose of
exploring is to generate candidates, the method that discards most of them has
defeated the purpose.

And there is a reason the holdout is not merely preferable but necessary. Every
correction requires knowing how many comparisons were run — and **an agent that
explores and reports what it found does not tell you how much it looked at.**
Correcting as though twenty-five tests happened when nine hundred did gives
$54.2\%$ precision against $94.5\%$.

The second half turns to cleaning, where the finding is that "cleaning" is mostly
not correction but choice. Across $288$ defensible pipelines, a true effect of
$0.30$ produced estimates from $-0.050$ to $0.570$ — a spread $6.1$ times wider
than the reported confidence interval ({{eq:cleaning-choices-move-the-answer}}).

## 3. Prerequisites

{{ch:aids-stack}}'s {{eq:check-strong-build-weak}} and its gradeability argument —
this chapter examines the two weakest-verifier stages.

{{ch:aids-text-to-sql}}'s {{eq:meaning-lives-outside-the-schema}}, whose conclusion
— write the convention down somewhere executable — recurs here in a different form.

{{ch:ag-termination}}'s {{eq:habituation}}, which decides what human review of
exploratory output can achieve.

Familiarity with hypothesis testing and multiple comparisons is assumed.

## 4. Intuitive Explanation

An analyst opens a dataset with a question in mind. They check the obvious things —
does the effect show up by region, by cohort, by month — and then, having answered
those, they keep looking. Cross-tabs, correlations, subgroup splits. This is
exploratory data analysis and it is a good and necessary activity.

It also has a property that has been known for a century: **with enough
comparisons on finite data, something always looks significant.** At a $5\%$
threshold, one comparison in twenty comes up positive by chance alone. Twenty
comparisons yields one. Nine hundred yields forty-five.

The important structure is what happens to the *true* findings over the same range.
They do not grow. The dataset contains however many real effects it contains, and
after the analyst has tested the hypotheses they came with, every further
comparison is drawn from the pool that has nothing in it.

{{sec:9-practical-example}} makes this concrete: true findings flat at $4.3$ across
the whole range, false findings from $0.18$ to $44.6$.

Now put an agent on it. The agent is not worse at statistics than the analyst; it
may be better. But it runs eight hundred comparisons in the time the analyst runs
twenty, and the false-discovery count is a function of that number. **Automating
exploration multiplies the noise and leaves the signal where it was.**

That is a genuinely uncomfortable conclusion because exploration is one of the
activities agents look most impressive at. The output — plots, cross-tabs, "I
noticed that customers in the northeast segment show a $12\%$ higher conversion
rate" — is voluminous, plausible, and mostly meaningless.

Two remedies exist and they behave differently. A **correction** tightens the
significance threshold in proportion to the number of tests. A **holdout** re-tests
each candidate finding on data the exploration never saw.

For exploration specifically the holdout wins, and by a lot. Bonferroni's whole
mechanism is to make it very hard to declare anything, which is right when you are
confirming and wrong when you are generating candidates — it throws away most of
the real findings along with the false ones.

But the decisive argument is different. A correction needs to know the denominator.
The agent reports three interesting patterns; it does not report the eight hundred
it discarded, and frequently it has not counted them. **The automation destroys
exactly the input every correction needs**, and the holdout is the only method that
does not need it.

Then cleaning, where the word does a lot of concealing.

"Cleaning" sounds like correction — removing errors, restoring the data to what it
should have been. Some of it is. Most of it is choosing among defensible options:
drop the rows with missing values or impute them; keep the outliers or winsorise
them; keep the first duplicate or merge; local time or UTC.

Every option is defensible. Every one produces a different dataset and a different
answer. {{sec:9-practical-example}} runs all $288$ combinations of five such
decisions and finds a true effect of $0.30$ producing estimates from $-0.050$ to
$0.570$ — **the low end has the opposite sign, and nobody did anything wrong.**

Against that, a single pipeline's confidence interval has width $0.102$. The
analytic spread is six times wider than the sampling uncertainty the report
actually communicates.

An agent makes these choices too. It makes them quickly, consistently, and
*invisibly* — the output is a number and a paragraph describing what was done,
which reads as description rather than as decision.

## 5. Formal Explanation

**Exploration as search.** Let a dataset contain $r$ real effects and $\nu$ null
comparisons, with per-test power $\beta$ and threshold $\alpha$. An analyst tests
their pre-formed hypotheses first, so after $n$ comparisons:

$$\mathbb{E}[\text{TP}] = \beta \min(n, r), \qquad \mathbb{E}[\text{FP}] = \alpha \max(n - r, 0)$$ (eq:exploration-is-a-search)

The first saturates at $\beta r$; the second grows without bound. Precision is:

$$P(n) = \frac{\beta \min(n,r)}{\beta \min(n,r) + \alpha (n-r)^+} \;\xrightarrow[n \to \infty]{}\; 0$$ (eq:more-exploration-finds-only-noise)

**Exploration has a saturating numerator and a linear denominator**, which is the
same structure as {{ch:mcp-production}}'s marginal server and with the same
consequence: past a point, more is worse.

**Corrections.** A Bonferroni threshold $\alpha/n$ gives
$\mathbb{E}[\text{FP}] = \alpha(n-r)^+/n \le \alpha$, bounded — at the cost of
power, since $\beta$ falls with the threshold. Writing $\beta(\alpha')$ for power at
threshold $\alpha'$:

$$\text{TP}_{\text{Bonf}} = \beta(\alpha/n)\, r \ll \beta(\alpha)\, r$$

**Holdout.** Re-test each of the $\text{TP} + \text{FP}$ candidates on fresh data.
Real effects survive at rate $\sigma \approx \beta$; nulls survive at $\alpha$:

$$\text{TP}_{\text{hold}} = \sigma\beta r, \qquad \text{FP}_{\text{hold}} = \alpha^2 (n-r)^+$$ (eq:holdout-beats-correction)

The false term is now $\alpha^2$ rather than $\alpha$, a quadratic suppression, and
the true term loses only the factor $\sigma$. So the holdout suppresses false
findings *multiplicatively* while the correction suppresses them by *raising the
bar*, and only the latter costs power.

Critically, $\text{FP}_{\text{hold}}$ contains no $1/n$ term: **the holdout does not
need to know $n$.** A correction applied with a wrong denominator $\hat{n} \ne n$
gives:

$$\mathbb{E}[\text{FP}] = \frac{\alpha}{\hat n}(n - r)^+$$

which is unbounded in $n$ for fixed $\hat n$ — the failure mode when the count is
unreported.

**The multiverse.** Let an analysis require $k$ decisions, decision $j$ having
options $O_j$ with effects $\delta_{j o}$ on the estimate. The set of defensible
estimates is:

$$\mathcal{M} = \Big\{\theta + \textstyle\sum_j \delta_{j o_j} \;\Big|\; o_j \in O_j\Big\}, \qquad |\mathcal{M}| = \prod_j |O_j|$$ (eq:cleaning-choices-move-the-answer)

with spread $\sum_j (\max_o \delta_{jo} - \min_o \delta_{jo})$ — **additive across
decisions**, so it grows linearly in the number of choices while sampling error
falls as $1/\sqrt{N}$. For any analysis with several decisions and adequate sample
size, the analytic spread exceeds the statistical one, and more data does not
help.

**And the resolution.** Enumerating $\mathcal{M}$ costs $\prod_j |O_j|$ pipeline
runs. For a human that is prohibitive; for an agent it is a loop:

$$\text{cost}_{\text{multiverse}} / \text{cost}_{\text{single}} = \prod_j |O_j| \quad\text{— large for a human, small in wall-clock for an agent}$$ (eq:speed-makes-the-multiverse-free)

**The speed that makes exploratory automation dangerous makes multiverse reporting
affordable**, and it is the one place in this part where the agent's advantage
lines up with the methodological requirement.

## 6. Mathematical Foundation

Three extractions.

**Precision has no floor.** From {{eq:more-exploration-finds-only-noise}},
$P(n) \to 0$: there is no amount of exploration at which the reported findings
become mostly noise and then stabilise. This distinguishes it from most degradation
in this book, which asymptotes.

**The holdout's suppression is quadratic and the correction's is not.** From
{{eq:holdout-beats-correction}}, $\alpha^2$ against $\alpha/n$. For $\alpha = 0.05$
these are equal at $n = 20$; beyond that the correction is stronger on false
positives and weaker on true ones. The holdout's advantage is that it achieves its
suppression without touching power.

**Analytic spread grows and statistical spread shrinks.**
{{eq:cleaning-choices-move-the-answer}} is additive in $k$ while standard error is
$O(1/\sqrt{N})$. So the ratio of analytic to statistical uncertainty grows with both
sample size and analysis complexity — **big data makes the multiverse problem
relatively worse, not better.**

## 7. Internal Mechanics

### 7.1 The forking path, drawn

```mermaid {#fig:forking-paths caption="Exploration after the pre-formed hypotheses are exhausted. Every additional branch is drawn from the null pool, so the true findings saturate and the false ones do not."}
flowchart TD
    Q[the question you came with] --> H1[hypothesis 1]
    Q --> H2[hypothesis 2]
    Q --> H3[hypothesis 3]
    H3 --> E["...and then exploration"]
    E --> N1[by region]
    E --> N2[by cohort]
    E --> N3[by month]
    E --> N4[by region x cohort]
    E --> N5[by month, excluding Q1]
    E --> N6["...795 more"]
```

The diagram's asymmetry is the finding. The top three branches contain whatever
real effects the analyst suspected; the bottom branches are a sample from a
distribution with nothing in it, and they are where the volume is.

### 7.2 What a holdout means for an agent

The mechanism is simple and the discipline is not.

Split the data before the agent sees it. Give the agent one part to explore
freely — the more comparisons the better, since exploration is generating
candidates. Then take its reported findings and test each one, once, on the part it
never touched.

Two properties make this fit automation particularly well.

**It does not require the agent to be honest about its process.** The agent may
have run eight hundred comparisons or eight; the holdout test is the same either
way. That matters because {{sec:9-practical-example}} shows the correction failing
badly on a wrong denominator, and the denominator is the thing an agent's report
omits.

**It converts the agent's speed into an advantage.** Under a correction, more
exploration is costly — each additional test tightens everyone's threshold. Under a
holdout, more exploration is free: extra candidates cost only their confirmation
tests. The agent should explore *more*, not less, provided the confirmation is
enforced.

The discipline part is that the holdout must be genuinely untouched, and an agent
with filesystem access to the whole dataset will read it. This is an access-control
problem rather than a prompt instruction, and it is the same containment argument
{{ch:ag-security}} made: do not ask the agent not to look, arrange that it cannot.

### 7.3 Why "interesting" is not a verifier

An exploratory agent typically has a selection criterion — report what is
*interesting*, or *surprising*, or *notable*.

That criterion is precisely a filter for extreme values, and extreme values in a
large comparison set are disproportionately noise. **Selecting for surprisingness
selects for spuriousness**, because the most surprising thing in nine hundred
comparisons is nearly always the one that got the luckiest draw.

So the agent's own sense of what is worth reporting actively worsens the precision
in {{eq:more-exploration-finds-only-noise}} rather than mitigating it, and an agent
that reported a random sample of its comparisons would produce a less misleading
output than one that reports its best ones.

That is a strange property to design around and it has a clean implication: the
agent's ranking is useful for *ordering* confirmation tests and useless as evidence
on its own.

### 7.4 Visualisation inherits the same problem

Plots are exploration with the multiple-comparisons structure hidden.

An agent that produces forty charts has run forty comparisons; the ones that look
interesting are the ones with the most extreme sample patterns. Nothing about the
visual form changes the arithmetic, and the visual form makes the pattern more
convincing rather than less — a scatter with a visible trend reads as evidence in a
way a $p$-value does not.

The practical rule: **count the charts as tests.** An agent that generated forty
plots and shows you three has a denominator of forty, and the three were selected
for extremity.

### 7.5 Cleaning decisions that should be specified rather than made

{{eq:cleaning-choices-move-the-answer}} says the spread is additive across
decisions, and {{sec:9-practical-example}} says two decisions carried $61\%$ of it.
That asymmetry is what makes this tractable.

The response is not to enumerate everything on every analysis. It is to identify —
once, per data source — the decisions that carry the spread, and *specify* them:
this dataset's outlier policy is winsorisation at the $99$th percentile, this
column's missing values are imputed by group, this system's timestamps are stored
UTC and reported local.

That is {{ch:aids-text-to-sql}}'s conclusion in a new setting: **when the missing
information is convention, write the convention down somewhere executable.** A
specified cleaning policy converts a silent per-analysis choice into a stated,
reviewable, versioned one — and it makes the agent's job easier rather than harder,
because a specified policy is an instruction it can follow.

### 7.6 Reporting the multiverse without drowning the reader

Full enumeration is the right computation and the wrong report. Three usable
forms:

**The spread on the decision.** Not all $288$ estimates, but: *across defensible
pipelines the estimate ranges from $-0.05$ to $0.57$, and $71\%$ of them exceed the
threshold that would change the decision.* That is one sentence and it is the
sentence the reader needs.

**The two decisions that matter.** {{sec:9-practical-example}}'s per-decision
attribution identifies which choices carry the spread, so the report can discuss
outlier handling and ignore timezone.

**A specification diff.** Where a policy exists, report only the deviations from it
and their effect. Analyses that follow the policy report a number; analyses that
depart from it justify the departure.

### 7.7 What human review can and cannot do here

{{ch:ag-termination}}'s habituation applies with force, because exploratory output
is voluminous by nature. A reviewer shown forty charts per analysis is not
examining the fortieth.

So human review is the wrong control for exploration volume, and the right controls
are structural: a holdout the agent cannot reach, a specified cleaning policy, a
reported denominator. **The human's remaining job is the one nothing else can do —
deciding whether the question was worth asking and whether the answer would change
anything** — and that job is made harder, not easier, by presenting forty findings
instead of three.

## 8. Implementation

Two listings. The first measures what exploration volume does to precision. The
second measures how far defensible cleaning choices move an answer.

```python {tier=A name=exploration-is-a-search}
"""Exploration at machine speed, and what it does to what you find.

ch:aids-stack found the exploration stage to have almost no verifier: there is no
reference answer for "was this exploration adequate". This listing measures a
consequence that is worse than the absence of a check.

Exploration is a search over comparisons. Each comparison can turn up a real
effect or a spurious one, and the spurious rate is a property of the DATA rather
than of the analyst -- with enough comparisons on finite data, something always
looks significant. That is the multiple-comparisons problem, and it has been known
for a century.

What changes when an agent does the exploring is the count. A human runs twenty
comparisons in an afternoon; an agent runs eight hundred in a minute. The false
discovery rate is a function of that count (eq:exploration-is-a-search), so
automating exploration multiplies the false findings rather than the findings.
"""
import numpy as np

rng = np.random.default_rng(4591)

M = 3000                # datasets simulated
N_REAL = 6              # genuine effects present
N_NULL = 900            # comparisons available with no real effect
POWER = 0.72            # chance a real effect is detected when tested
ALPHA = 0.05


def effective_alpha(n_tests, correction, alpha, n_real):
    """The per-test threshold each method actually applies."""
    if correction == "bonferroni":
        return alpha / max(n_tests, 1)
    if correction == "fdr":
        # Benjamini-Hochberg sits between alpha and alpha/n, closer to alpha
        # when there are many true effects among the tests.
        return alpha * (1.0 + min(n_real, n_tests)) / max(n_tests, 1)
    return alpha


def explore(n_tests, m=M, n_real=N_REAL, n_null=N_NULL, power=POWER,
            alpha=ALPHA, correction=None, holdout=False, assume_tests=None):
    """Run `n_tests` comparisons. The analyst tests the hypotheses they came
    with FIRST and then explores, so the real effects are always among the
    tested set and every additional comparison is a null. That ordering is what
    makes exploration a forking path rather than a wider search.

    `assume_tests` corrects as if that many tests had been run, which is what
    happens when the count is not reported.
    """
    n_tests = min(n_tests, n_real + n_null)
    tested_real = min(n_tests, n_real)
    tested_null = n_tests - tested_real

    a = effective_alpha(assume_tests or n_tests, correction, alpha, n_real)
    # Tightening the threshold costs power.
    pw = power * min((a / alpha) ** 0.25, 1.0)

    tp = rng.binomial(tested_real, pw, m)
    fp = rng.binomial(tested_null, min(a, 1.0), m)

    if holdout:
        # Re-test every candidate on held-out data: real effects mostly
        # survive, spurious ones survive at the base rate.
        tp = rng.binomial(tp, 0.85)
        fp = rng.binomial(fp, alpha)

    reported = tp + fp
    prec = np.where(reported > 0, tp / np.maximum(reported, 1), np.nan)
    return (float(tp.mean()), float(fp.mean()),
            float(np.nanmean(prec)) if np.isfinite(prec).any() else 0.0)


print(f"A dataset with {N_REAL} genuine effects and {N_NULL} comparisons that")
print(f"have none. Each real effect is detected {POWER:.0%} of the time it is")
print(f"tested; each null comparison looks significant {ALPHA:.0%} of the time.")
print()
print(f"{'comparisons run':>17}{'true found':>12}{'false found':>13}"
      f"{'precision':>11}")
print("-" * 53)
tab = {}
for n in (10, 25, 100, 400, 900):
    r = explore(n)
    tab[n] = r
    print(f"{n:>17}{r[0]:>12.2f}{r[1]:>13.2f}{r[2]:>11.1%}")

print()
print()
print("The same, framed as the analyst experiences it: how many of the things")
print("you would report are real.")
print()
print(f"{'who':>28}{'comparisons':>13}{'reported':>11}{'real ones':>12}")
print("-" * 64)
WHO = [("human, an afternoon", 20), ("human, a week", 60),
       ("agent, a minute", 400), ("agent, an hour", 900)]
wh = {}
for label, n in WHO:
    r = explore(n)
    wh[label] = (n, r[0] + r[1], r[0])
    print(f"{label:>28}{n:>13}{r[0] + r[1]:>11.2f}{r[0]:>12.2f}")

print()
print()
print("Corrections. Bonferroni divides the threshold by the number of tests;")
print("FDR control is less severe; a held-out re-test is neither.")
print()
print(f"{'method':>22}" + "".join(f"{'n=' + str(n):>12}" for n in (25, 400, 900)))
print("-" * 58)
cm = {}
for label, kw in (("none", {}), ("Bonferroni", {"correction": "bonferroni"}),
                  ("FDR control", {"correction": "fdr"}),
                  ("held-out re-test", {"holdout": True})):
    row = tuple(explore(n, **kw)[2] for n in (25, 400, 900))
    cm[label] = row
    print(f"{label:>22}" + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("What each method costs in real findings missed, at 400 comparisons.")
print()
print(f"{'method':>22}{'true found':>12}{'false found':>13}{'precision':>11}")
print("-" * 58)
ct = {}
for label, kw in (("none", {}), ("Bonferroni", {"correction": "bonferroni"}),
                  ("FDR control", {"correction": "fdr"}),
                  ("held-out re-test", {"holdout": True})):
    r = explore(400, **kw)
    ct[label] = r
    print(f"{label:>22}{r[0]:>12.2f}{r[1]:>13.2f}{r[2]:>11.1%}")

print()
print()
print("And the interaction that matters: a correction assumes you know how many")
print("comparisons were run. An agent that explores and reports only what it")
print("found does not tell you.")
print()
print(f"{'actual comparisons':>20}{'corrected as if 25':>20}"
      f"{'corrected correctly':>21}")
print("-" * 61)
un = {}
for n in (25, 100, 400, 900):
    wrong = explore(n, correction="bonferroni", assume_tests=25)[2]
    right = explore(n, correction="bonferroni")[2]
    un[n] = (wrong, right)
    print(f"{n:>20}{wrong:>20.1%}{right:>21.1%}")

print(f"""
The first table's second column is the one to read first, because it does not
move. {N_REAL} real effects exist and about {tab[10][0]:.1f} of them are found by
{10} comparisons. Running {900} instead finds {tab[900][0]:.1f} -- the same ones.

The false column goes from {tab[10][1]:.2f} to {tab[900][1]:.2f}, and precision
from {tab[10][2]:.1%} to {tab[900][2]:.1%}.

**More exploration does not find more real effects. It finds more spurious ones**
(eq:exploration-is-a-search), because after the hypotheses you came with are
exhausted, every additional comparison is drawn from the null pool.

The second table is that result in the units that matter. A human exploring for an
afternoon reports {wh['human, an afternoon'][1]:.1f} findings of which
{wh['human, an afternoon'][2]:.1f} are real. An agent exploring for an hour reports
{wh['agent, an hour'][1]:.1f} of which {wh['agent, an hour'][2]:.1f} are real.

**Same discoveries, {wh['agent, an hour'][1] / wh['human, an afternoon'][1]:.0f}
times the noise.** That is what automating an activity with no verifier buys, and
it is worth stating in exactly those terms: the agent did not explore worse than
the human. It explored more, and more is the problem.

The correction table shows the standard remedies working. At {400} comparisons,
uncorrected precision is {cm['none'][1]:.1%}; Bonferroni gives {cm['Bonferroni'][1]:.1%}.

But the cost table is where the choice is made. Bonferroni at {400} comparisons
keeps {ct['Bonferroni'][0]:.2f} real findings out of the {ct['none'][0]:.2f}
available -- it discards
{1 - ct['Bonferroni'][0] / ct['none'][0]:.0%} of the genuine discoveries to buy its
precision.

A held-out re-test keeps {ct['held-out re-test'][0]:.2f} at
{ct['held-out re-test'][2]:.1%} precision.

**For exploration specifically, the holdout dominates the correction.** The purpose
of exploring is to generate candidate hypotheses, and a method that discards
{1 - ct['Bonferroni'][0] / ct['none'][0]:.0%} of them has defeated the purpose in
order to protect against reporting them. Re-testing on data the exploration never
saw keeps {ct['held-out re-test'][0] / ct['none'][0]:.0%} of the candidates and
still reaches {ct['held-out re-test'][2]:.1%}.

The last table is why this is not merely a statistics reminder, and it is the
chapter's actual argument.

A correction requires knowing how many comparisons were run. Correcting as though
{25} tests happened when {900} did gives {un[900][0]:.1%} precision against
{un[900][1]:.1%} for correcting correctly.

**An agent that explores and reports what it found does not tell you how much it
looked at.** It reports three interesting patterns; it does not report the eight
hundred it discarded, and often it has not counted. So the input that every
correction needs is precisely the quantity the automation destroys.

Which gives the rule for this chapter. **An exploratory agent must hold out data,
because it cannot be trusted to report its own denominator** -- and the holdout is
the only method here that does not require one.""")
```

The second listing asks what cleaning decisions do to the result.

```python {tier=A name=cleaning-choices-move-the-answer}
"""Cleaning decisions are choices, and choices move the answer.

"Cleaning" sounds like correction -- removing errors, restoring the data to what
it should have been. Most cleaning decisions are not that. They are choices among
defensible options:

  missing values      drop the rows, impute the mean, impute by group, flag
  outliers            keep, winsorise, trim at 1%, trim at 5%
  duplicates          keep first, keep last, merge
  a category with 12  keep, fold into 'other', drop
  timezone at a boundary  the event's local time, or UTC

Each is defensible. Each produces a different dataset, and a different answer to
the same question. The spread across defensible pipelines is a real quantity, and
in many analyses it is larger than the effect being estimated
(eq:cleaning-choices-move-the-answer).

A human making these choices at least knows they chose. An agent produces one
number and a paragraph explaining what it did.
"""
import numpy as np
import itertools

rng = np.random.default_rng(4637)

M = 4000                # datasets
TRUE_EFFECT = 0.30      # the quantity being estimated, in whatever units

# (decision, options, how much each option shifts the estimate)
DECISIONS = [
    ("missing values",  ["drop rows", "impute mean", "impute by group", "flag"],
     [-0.09, +0.06, +0.01, -0.02]),
    ("outliers",        ["keep", "winsorise", "trim 1%", "trim 5%"],
     [+0.11, +0.02, -0.03, -0.12]),
    ("duplicates",      ["keep first", "keep last", "merge"],
     [-0.04, +0.05, +0.00]),
    ("rare categories", ["keep", "fold to other", "drop"],
     [+0.02, -0.01, -0.07]),
    ("timezone",        ["local", "UTC"],
     [+0.03, -0.03]),
]

NOISE = 0.05            # sampling noise on any single estimate


def estimate(choice_idx, m=M, true=TRUE_EFFECT, noise=NOISE):
    """The estimate produced by one specific pipeline."""
    shift = sum(DECISIONS[d][2][c] for d, c in enumerate(choice_idx))
    return true + shift + rng.normal(0, noise, m)


ALL = list(itertools.product(*[range(len(d[1])) for d in DECISIONS]))

print(f"An analysis with {len(DECISIONS)} cleaning decisions and")
print(f"{len(ALL)} defensible combinations. The true effect is {TRUE_EFFECT:.2f};")
print(f"sampling noise on any single estimate is {NOISE:.2f}.")
print()
print(f"{'decision':>18}{'options':>10}{'range of shift':>17}")
print("-" * 46)
for name, opts, shifts in DECISIONS:
    print(f"{name:>18}{len(opts):>10}{max(shifts) - min(shifts):>17.2f}")

print()
print()
print("Every defensible pipeline, run. The spread is the multiverse.")
print()
means = np.array([estimate(c).mean() for c in ALL])
print(f"{'true effect':>26}{TRUE_EFFECT:>10.3f}")
print(f"{'sampling noise (1 sd)':>26}{NOISE:>10.3f}")
print(f"{'lowest defensible estimate':>26}{means.min():>10.3f}")
print(f"{'highest defensible estimate':>26}{means.max():>10.3f}")
print(f"{'spread across pipelines':>26}{means.max() - means.min():>10.3f}")
print()
print(f"   The spread is {(means.max() - means.min()) / NOISE:.1f}x the sampling")
print(f"   noise and {(means.max() - means.min()) / TRUE_EFFECT:.0%} of the effect"
      f" being measured.")

print()
print()
print("What a single reported number could have been. Percentiles across the")
print("defensible pipelines:")
print()
for q in (0, 10, 25, 50, 75, 90, 100):
    print(f"{f'p{q}':>10}{np.percentile(means, q):>12.3f}")

print()
print()
print("Which decisions carry the spread. Range of the estimate as each single")
print("decision varies, with the others held at their first option:")
print()
print(f"{'decision':>18}{'range':>10}{'share of total':>16}")
print("-" * 44)
tot_var = 0.0
per = {}
for d, (name, opts, shifts) in enumerate(DECISIONS):
    base = [0] * len(DECISIONS)
    vals = []
    for c in range(len(opts)):
        base[d] = c
        vals.append(estimate(tuple(base)).mean())
    per[name] = max(vals) - min(vals)
    tot_var += per[name]
for name in per:
    print(f"{name:>18}{per[name]:>10.3f}{per[name] / tot_var:>16.1%}")

print()
print()
print("The sign question, which is what a decision actually turns on. Share of")
print("defensible pipelines that would support each conclusion, for three")
print("possible decision thresholds:")
print()
print(f"{'threshold':>12}{'above':>10}{'below':>10}{'verdict':>22}")
print("-" * 54)
th = {}
for t in (0.20, 0.30, 0.40):
    above = float((means > t).mean())
    th[t] = above
    verdict = ("unanimous" if above > 0.99 or above < 0.01
               else "contested" if 0.2 < above < 0.8 else "leaning")
    print(f"{t:>12.2f}{above:>10.1%}{1 - above:>10.1%}{verdict:>22}")

print()
print()
print("And what reporting the multiverse costs, against reporting one number.")
print()
one = estimate(ALL[0])
print(f"{'reporting style':>26}{'what the reader gets':>34}")
print("-" * 62)
print(f"{'a single pipeline':>26}{f'{one.mean():.3f} +/- {one.std():.3f}':>34}")
print(f"{'the multiverse':>26}"
      f"{f'{means.min():.3f} to {means.max():.3f} across {len(ALL)}':>34}")
print()
print(f"   The single-pipeline interval has width {2 * one.std():.3f}.")
print(f"   The multiverse spread is {means.max() - means.min():.3f}, "
      f"{(means.max() - means.min()) / (2 * one.std()):.1f}x wider.")

print(f"""
The spread is the whole listing. A true effect of {TRUE_EFFECT:.2f} produces
defensible estimates from {means.min():.3f} to {means.max():.3f}.

The low end has **the opposite sign from the truth**, and every pipeline that
produced it is defensible -- drop the missing rows, trim at {5}%, keep the first
duplicate. Nobody did anything wrong.

Against sampling noise of {NOISE:.2f}, the spread is
{(means.max() - means.min()) / NOISE:.1f} times larger
(eq:cleaning-choices-move-the-answer). Against the effect being measured it is
{(means.max() - means.min()) / TRUE_EFFECT:.0%}.

The last comparison is the one to take to an argument. A single pipeline reports
{one.mean():.3f} with an interval of width {2 * one.std():.3f}. The multiverse
spans {means.max() - means.min():.3f}, which is
{(means.max() - means.min()) / (2 * one.std()):.1f} times wider.

**The reported uncertainty describes sampling and omits the analysis**, and the
omitted part is the larger one. That is not a criticism of confidence intervals;
they measure what they claim to measure. It is a statement about what a single
number from a single pipeline can mean.

The decision table makes it concrete. At a threshold of {0.20:.2f},
{th[0.20]:.1%} of defensible pipelines support acting and {1 - th[0.20]:.1%}
support not acting. The analysis does not settle the question; the cleaning
choices do.

The per-decision table says where to look. Outlier handling carries
{per['outliers'] / tot_var:.1%} of the spread and timezone handling
{per['timezone'] / tot_var:.1%}, so the two are not equally worth discussing --
which is useful, because reporting a full multiverse is impractical and reporting
the two decisions that carry {(per['outliers'] + per['missing values']) / tot_var:.0%}
of it is not.

Now the part that concerns automation specifically.

Every one of these choices has to be made, and an agent makes them. It makes them
quickly, it makes them consistently, and it makes them **invisibly** -- the output
is a number and a paragraph saying what was done, which reads as a description of
the data rather than as a decision about it.

A human analyst is not better at choosing. They are, however, aware of having
chosen, and that awareness is what produces the sentence "we also tried it
winsorised and it did not change much" -- or, more importantly, its absence when it
did.

**The risk of automating cleaning is not that the agent chooses badly. It is that
the choice stops being visible as a choice**, and the multiverse collapses to a
point estimate whose uncertainty is understated
{(means.max() - means.min()) / (2 * one.std()):.0f}-fold.

Which suggests the intervention, and it is one automation is unusually good at. A
human cannot run {len(ALL)} pipelines. An agent can run them in the time it takes
to run one, and report the spread instead of the point.

**The same speed that makes exploratory automation dangerous makes multiverse
reporting free.** It is the one place in this part where the agent's advantage
lines up with the methodological need.""")
```

## 9. Practical Example

The first listing explores a dataset with six real effects and nine hundred null
comparisons:

```
  comparisons run  true found  false found  precision
-----------------------------------------------------
               10        4.33         0.18      96.4%
              100        4.34         4.65      50.4%
              400        4.32        19.68      18.4%
              900        4.32        44.61       8.9%
```

**The true column does not move.** Six real effects exist and about $4.3$ are found
by ten comparisons; nine hundred comparisons finds the same ones and $44.6$
spurious ones ({{eq:more-exploration-finds-only-noise}}).

In the units that matter:

```
                         who  comparisons   reported   real ones
----------------------------------------------------------------
         human, an afternoon           20       5.02       4.32
             agent, a minute          400      24.07       4.34
              agent, an hour          900      49.09       4.31
```

Same discoveries, ten times the noise. **The agent did not explore worse. It
explored more, and more is the problem.**

The remedies:

```
                method  true found  false found  precision
----------------------------------------------------------
                  none        4.34        19.73      18.5%
            Bonferroni        0.97         0.05      95.9%
           FDR control        1.56         0.34      83.4%
      held-out re-test        3.70         0.98      81.1%
```

Bonferroni reaches $95.9\%$ precision by discarding $78\%$ of the genuine findings.
A held-out re-test keeps $85\%$ of them at $81.1\%$
({{eq:holdout-beats-correction}}) — **for exploration, whose purpose is generating
candidates, the holdout dominates.**

And the argument that settles it:

```
  actual comparisons  corrected as if 25  corrected correctly
-------------------------------------------------------------
                 100               93.2%                97.2%
                 400               73.1%                95.7%
                 900               54.2%                94.5%
```

**Every correction needs a denominator that automation destroys.** The agent
reports three findings; it does not report the eight hundred it discarded. The
holdout is the only method here that does not need the count.

The second listing runs all $288$ defensible cleaning pipelines:

```
               true effect     0.300
     sampling noise (1 sd)     0.050
lowest defensible estimate    -0.050
highest defensible estimate    0.570
   spread across pipelines     0.620
```

**The lowest defensible estimate has the opposite sign from the truth**, and every
pipeline producing it is defensible. The spread is $12.4$ times the sampling noise
and $207\%$ of the effect being measured.

Where it comes from:

```
          decision     range  share of total
--------------------------------------------
    missing values     0.149           24.0%
          outliers     0.231           37.1%
          timezone     0.061            9.7%
```

What it does to a decision:

```
   threshold     above     below               verdict
------------------------------------------------------
        0.20     71.2%     28.8%             contested
        0.30     39.2%     60.8%             contested
```

And against what gets reported:

```
           reporting style              what the reader gets
--------------------------------------------------------------
         a single pipeline                   0.331 +/- 0.051
            the multiverse        -0.050 to 0.570 across 288
```

**The reported uncertainty describes sampling and omits the analysis**, and the
omitted part is $6.1$ times larger.

The resolution is that the agent's speed cuts the other way here: a human cannot
run $288$ pipelines and an agent can run them in the time it takes to run one.
**The same speed that makes exploratory automation dangerous makes multiverse
reporting free** ({{eq:speed-makes-the-multiverse-free}}).

## 10. Production Considerations

Split the data before the agent sees it, and enforce the split with access control
rather than instruction.

Let the agent explore freely on its half. Under a holdout, more exploration is
free; under a correction it is costly.

Confirm every reported finding on the untouched half, once.

Count the charts as tests. Forty plots is a denominator of forty, and the three
shown were selected for extremity.

Treat the agent's sense of "interesting" as an ordering for confirmation, never as
evidence — selecting for surprisingness selects for noise.

Specify the cleaning decisions that carry the spread, per data source, once. Two
decisions carried $61\%$ of it in the listing.

Report the spread across defensible pipelines rather than one number, at least on
the decisions that carry it.

And do not use human review as the control for exploratory volume.
{{eq:habituation}} says it fails exactly where the volume is.

## 11. Common Mistakes

**Treating more exploration as more discovery.** The true findings saturate.

**Bonferroni on exploratory work.** Discards most of the candidates the exploration
existed to produce.

**Correcting with an unreported denominator.** $54.2\%$ against $94.5\%$.

**Trusting the holdout to a prompt instruction.** An agent with file access will
read it.

**Reading "interesting" as evidence.** It is a filter for extremity.

**Not counting charts.** Visualisation hides the comparison count and makes the
pattern more convincing.

**Reporting one pipeline's interval as the uncertainty.** It omits the larger term.

**Letting the agent choose cleaning policy silently.** The choice stops being
visible as a choice.

## 12. Failure Modes

*Plausible spurious finding.* The characteristic output: a real pattern in this
sample, absent in the next, reported with a chart.

*Precision collapse under volume.* Quality falling as the agent gets faster, with
nothing in the output indicating it.

*Silent pipeline selection.* One defensible cleaning path chosen and reported as
description, with the multiverse invisible.

*Understated uncertainty.* An interval six times too narrow, correctly computed.

*Contaminated holdout.* The confirmation set read during exploration, which
silently removes the only working control.

*Reviewer saturation.* Forty findings per analysis and a reviewer reading the first
three.

## 13. Alternatives

**Pre-registration.** State the hypotheses before looking, which makes the
denominator known by construction. Correct, and it forgoes exploration's purpose.

**Sample splitting into three.** Explore, confirm, and hold a final estimate set —
gets an unbiased effect size as well as a confirmed existence claim.

**Specified analysis policy.** {{sec:7-internal-mechanics}}'s recommendation:
convention written down once, executable, versioned.

**Reporting a random sample of comparisons.** Perverse and instructive — a less
misleading output than reporting the best ones, and a useful sanity check.

**Not automating exploration.** Defensible where no holdout is available and the
findings will be acted on.

## 14. Evaluation

Instrument the comparison count. An exploratory agent should report how many
comparisons it ran, and most do not because nobody asked.

Measure your confirmation rate: of findings the agent reported, what fraction
survive a holdout test. It is the direct measurement of
{{eq:more-exploration-finds-only-noise}} and it is cheap.

Run the multiverse on a completed analysis and compare the spread against the
reported interval. Once, on something that mattered, is usually persuasive.

Attribute the spread per decision, so you know which two to specify.

And track findings that were reported and later failed to replicate, by exploration
volume. The relationship is the chapter's claim and it is testable in your own
history.

## 15. Advanced Concepts

**Automated multiverse reporting.** {{eq:speed-makes-the-multiverse-free}} makes
this practical and almost nothing implements it. The engineering — enumerate
decisions, run the grid, summarise the spread — is straightforward.
{{maturity:EMERGING}}.

**Decision-sensitivity attribution.** Identifying which analytic choices a
conclusion is sensitive to, automatically, so the report can name them.

**Adaptive data analysis.** A body of theory on reusing a holdout many times
without invalidating it, which is exactly the regime an agent creates and which is
rarely applied in practice. {{maturity:RESEARCH FRONTIER}}.

**Verifiers for exploration adequacy.** {{ch:aids-stack}}'s open problem: not
whether a finding is real, but whether the exploration covered what it should have.
Nothing addresses it.

## 16. Connection to Previous Chapters

{{ch:aids-stack}}'s check-strong-build-weak rule identified exploration as a
weak-verifier stage; this chapter finds that automating it makes the weakness worse
rather than exposing it.

{{ch:aids-text-to-sql}}'s conclusion — write the convention down executably —
recurs as the specified cleaning policy, for the same reason.

{{ch:ag-termination}}'s habituation rules out human review as the control, which
forces the structural answers.

{{ch:ag-security}}'s containment argument returns as the reason the holdout must be
enforced by access control rather than instruction.

{{ch:mcp-production}}'s saturating-benefit-against-linear-cost shape appears again
as {{eq:more-exploration-finds-only-noise}}.

Ahead: {{ch:aids-automl}} takes up the model stage, where the verifier is strong and
one error class — leakage — makes it stronger the more wrong the analysis is.

## 17. Exercises

1. Derive the crossover $n$ at which Bonferroni's false-positive suppression
   exceeds the holdout's, and check it against the listing.

2. Implement three-way splitting — explore, confirm, estimate — and measure the
   bias in the effect size that two-way splitting leaves.

3. Model an agent whose "interesting" filter selects the top decile by effect size,
   and measure the precision penalty relative to random reporting.

4. Add a correlated-decisions term to the multiverse: some cleaning choices are not
   independent. Does the additive spread survive?

5. Compute the multiverse for an analysis you have run and compare the spread with
   the interval you reported.

6. Model adaptive reuse of a holdout across many rounds and find how many rounds it
   survives.

## 18. Interview Questions

1. Your agent explored for an hour and found forty patterns. What do you conclude?

2. Why does more exploration not find more real effects?

3. Would you use Bonferroni correction on exploratory output?

4. What input does every multiple-comparisons correction need, and why does
   automation not supply it?

5. Your analysis reports an effect of $0.33 \pm 0.05$. What is missing?

6. Is "this finding is surprising" evidence for or against it?

## 19. Research Questions

1. Can exploration adequacy — coverage rather than correctness — be given a
   checkable definition?

2. How much holdout reuse does adaptive data analysis actually license in an agent
   setting?

3. Can decision sensitivity be attributed automatically well enough to name the two
   choices that matter?

4. Does the reported-findings replication rate correlate with exploration volume in
   real organisational data?

5. What would a specified analysis policy look like as a first-class executable
   artefact?

## 20. Chapter Summary

Exploration is a search, and its arithmetic is unkind to speed. Across ten to nine
hundred comparisons on the same data, true findings stayed flat at $4.3$ while
false findings went from $0.18$ to $44.6$ and precision fell from $96.4\%$ to
$8.9\%$ ({{eq:more-exploration-finds-only-noise}}). **More exploration finds no
more real effects and many more spurious ones**, because after the pre-formed
hypotheses are exhausted every additional comparison is drawn from a pool with
nothing in it.

A human exploring for an afternoon reported $5.0$ findings of which $4.3$ were
real; an agent exploring for an hour reported $49.1$ of which $4.3$ were real. The
agent did not explore worse — it explored more.

For remedies, a held-out re-test dominates a correction in this setting: Bonferroni
reached $95.9\%$ precision while discarding $78\%$ of genuine findings, and the
holdout kept $85\%$ of them at $81.1\%$ ({{eq:holdout-beats-correction}}). The
decisive argument is that **every correction needs a comparison count that
automation destroys** — correcting as though twenty-five tests ran when nine
hundred did gives $54.2\%$ against $94.5\%$ — and the holdout is the only method
that does not need one.

On cleaning, the word conceals that most decisions are choices rather than
corrections. Across $288$ defensible pipelines a true effect of $0.30$ produced
estimates from $-0.050$ to $0.570$ ({{eq:cleaning-choices-move-the-answer}}) — the
low end with the opposite sign, every path defensible. The spread was $6.1$ times
wider than the reported confidence interval, and since analytic spread is additive
in the number of decisions while sampling error falls as $1/\sqrt{N}$, **more data
makes this relatively worse.**

The risk of automating cleaning is not bad choices. It is that the choice stops
being visible as one.

And the resolution is the chapter's one piece of good news: enumerating $288$
pipelines is prohibitive for a human and a loop for an agent. **The same speed that
makes exploratory automation dangerous makes multiverse reporting free**
({{eq:speed-makes-the-multiverse-free}}).

## 21. Further Reading

{{cite:testini2025dsautomation}} for the survey finding that exploratory activities
are the least evaluated, which this chapter suggests is not merely an oversight —
they are the ones where the natural metric moves the wrong way.

{{cite:huang2024dacode}} for agent performance on tasks that include exploratory
stages, and {{cite:lu2024aiscientist}} for a system that explores and reports at
scale — read its results with this chapter's denominator question in mind.

{{ch:aids-stack}} for the verifier-strength argument that placed exploration among
the weak stages, and {{ch:ag-security}} for why the holdout has to be enforced
structurally.

{{cite:cemri2025mast}}'s silent-failure category, of which a spurious exploratory
finding is the analysis-pipeline instance.
