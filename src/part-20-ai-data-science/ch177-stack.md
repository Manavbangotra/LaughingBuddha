---
id: aids-stack
number: 177
part: XX
tier: full
status: draft
requires: [verifier-sets-the-ceiling, detection-decays-with-lag,
           coverage-before-freshness, marginal-server-turns-negative]
provides: [gradeable-is-not-representative, benchmark-share-follows-gradeability,
           amdahl-bounds-the-stack, pipeline-fails-at-the-weakest-verifier,
           check-strong-build-weak]
citations: [testini2025dsautomation, huang2024dacode, chan2024mlebench,
            li2023bird, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why benchmark attention in
data science follows gradeability rather than time; compute what a capability's
benchmark presence actually implies about project speedup; state the Amdahl bound
on automating only the checkable activities; place verification in an analysis
pipeline given per-stage detection rates; and distinguish where to *spend*
checking time from where to *build* a new verifier — which turn out to be opposite
ends of the pipeline.

## 2. Why This Matters

This part is about agents that do analysis, and it opens by establishing what the
published numbers do and do not say.

Four of them, from four groups, using four grading methods. Text-to-SQL on
realistic databases: **$40.08\%$ execution accuracy against $92.96\%$ for humans**
({{cite:li2023bird}}). Agent-shaped data science tasks: **$30.5\%$**
({{cite:huang2024dacode}}). Machine learning engineering against real Kaggle
leaderboards: **a bronze medal in $16.9\%$ of competitions**
({{cite:chan2024mlebench}}). And an end-to-end research pipeline producing papers
at **under $15$ each** ({{cite:lu2024aiscientist}}).

They agree on a shape: the demonstrations are impressive and the completion rates
are low.

But the more useful finding is about what is being measured at all.
{{cite:testini2025dsautomation}} surveyed data science automation evaluation and
found coverage concentrated on a small subset of goal-oriented activities, with
**data management and exploratory work largely ignored** — plus evaluation only at
the extremes of the collaboration spectrum, and an implicit framing of automation
as substitution rather than redesign.

There is a mechanism behind that first gap, and it is
{{ch:as-specialized}}'s. Benchmarks measure what can be **graded**, and what can be
graded is the part with a checkable answer. {{sec:9-practical-example}} finds
modelling receiving $3.3$ times its share of attention and question-framing
receiving $0.16$ times ({{eq:benchmark-share-follows-gradeability}}), so a
benchmark-weighted reading of capability overstates the time-weighted one by
$13.5$ points — and only $33\%$ of a practitioner's day is gradeable at all
({{eq:gradeable-is-not-representative}}).

Then the arithmetic that should temper every claim in this part: **fully
automating modelling produces a $1.12\times$ project speedup**
({{eq:amdahl-bounds-the-stack}}), because modelling is a small share of the time
precisely because it was tractable enough to automate first.

## 3. Prerequisites

{{ch:as-specialized}}'s {{eq:verifier-sets-the-ceiling}} — the result this whole
part rests on, transposed from agent domains to analysis stages.

{{ch:as-failures}}'s {{eq:detection-decays-with-lag}} and
{{eq:coverage-before-freshness}}, which this chapter's second listing revisits and
partly overturns for a specific reason.

{{ch:mcp-production}}'s {{eq:marginal-server-turns-negative}}, whose
saturating-benefit-against-linear-cost shape recurs here.

Familiarity with an analysis workflow — access, clean, explore, feature, model,
validate, conclude — is assumed rather than taught.

## 4. Intuitive Explanation

Ask what a data scientist does all day and the answer has been stable for two
decades: mostly not modelling. Finding the data, working out whether it means what
the column names claim, reconciling two systems that disagree about the same
customer, discovering that a field changed definition in March.

Now ask what the benchmarks measure. Text-to-SQL, because a query either returns
the right rows or does not. Model building, because a held-out score is a number.
Notebook code, because it runs or throws.

Those are not the same list, and the reason is not neglect. It is that a benchmark
needs a grading function, and the activities that dominate the day do not have
one. There is no reference answer for "was this the right question", and there is
no unit test for "is this exploration sound".

So the field's measured capability sits almost entirely in the gradeable minority
of the work. {{sec:9-practical-example}} puts that minority at about a third of a
practitioner's time, and finds benchmark attention allocated more than three times
in proportion to gradeability relative to time.

Which produces a specific and common error in reasoning. A capability improves on
a benchmark; the benchmark is prominent; the improvement is read as progress on
"data science". The arithmetic says otherwise: automating the entire modelling
stage perfectly — the most-benchmarked activity in the field — makes a project
about $12\%$ faster. Automating the cleaning stage perfectly, which gets a fifth
of the benchmark attention, makes it $35\%$ faster.

This is Amdahl's law, and it is unkind to exactly the activities that were easiest
to automate, because ease of automation and ease of grading and small share of
time are all the same underlying property: **the activity had a checkable answer,
which is why it was already efficient.**

Then the second question, which is what to do about the errors.

An analysis is a chain. An error in the join at the cleaning stage flows into the
features, into the model, into the validation, into the conclusion — and every
stage after it produces correct work *given the mistake*. The model converges. The
validation score is real. The chart is beautiful. The conclusion is wrong.

That is {{ch:as-failures}}'s propagation, and it suggests checking early. But there
is a second variable that chapter did not have: **the stages differ enormously in
whether anything can check them.** Row counts and schema checks catch a lot at the
access stage. A held-out score is a real signal at the model stage. There is
almost nothing to check an exploration against, and nothing at all to check a
conclusion against.

{{sec:9-practical-example}} finds those detection rates differing by a factor of
nine, and that dominates the freshness effect. A single check at the model stage is
worth $+11.6$ points; at the exploration stage, $+1.7$.

And then the reversal that makes the chapter. If you can *build* a verifier rather
than merely place a check, the ranking inverts: improving detection at the
conclusion stage is worth $+6.2$ points where improving it at the model stage is
worth $+2.0$. The strong stages are near their ceiling; the weak ones are nowhere
near it.

## 5. Formal Explanation

**Gradeability and coverage.** Let activities $i$ have time shares $t_i$ with
$\sum t_i = 1$, gradeability $g_i \in [0,1]$, and current automation quality
$a_i$. If benchmark coverage is allocated in proportion to gradeability,
$b_i = g_i / \sum_j g_j$, then the two readings of capability are:

$$A_{\text{bench}} = \sum_i b_i a_i, \qquad A_{\text{time}} = \sum_i t_i a_i$$ (eq:benchmark-share-follows-gradeability)

These differ whenever $\text{Cov}(g, a) > \text{Cov}(t, a)$ — that is, whenever
automation quality tracks gradeability more closely than it tracks time share,
which it does by construction: **an activity is automated because it is
checkable.** So the headline systematically exceeds the practitioner-relevant
figure, and the gap is the covariance difference.

The gradeable share of the work is:

$$\gamma = \sum_i t_i g_i$$ (eq:gradeable-is-not-representative)

and $1 - \gamma$ is the fraction of the day on which no benchmark reports progress
in *either* direction.

**Amdahl.** Automating activity $i$ perfectly gives:

$$S_i = \frac{1}{1 - t_i}$$ (eq:amdahl-bounds-the-stack)

which depends on $t_i$ alone — not on how hard the activity was, how impressive
the automation is, or how much benchmark attention it receives. Automating every
activity above a gradeability threshold $\theta$:

$$S(\theta) = \frac{1}{1 - \sum_{i : g_i \ge \theta} t_i}$$

bounded by the gradeable share. **The ceiling on automating-what-can-be-checked is
$1/(1-\gamma')$ where $\gamma'$ is the time share of the checkable activities**,
and it is a modest number.

**Pipeline error.** Now let stages $1..n$ each introduce an error with probability
$e_i$, absorbing (an error persists once made). A check at stage $j$ detects an
error made at stage $i \le j$ with probability:

$$d_j \cdot \delta^{\,j-i}$$

combining stage $j$'s verifier strength $d_j$ with
{{eq:detection-decays-with-lag}}'s decay. The expected value of a single check at
$j$ is:

$$V_j = d_j \sum_{i \le j} \Pr[\text{error outstanding from } i] \cdot \delta^{\,j-i}$$ (eq:pipeline-fails-at-the-weakest-verifier)

Two terms pull opposite ways in $j$: the accumulated error mass grows with $j$, and
$\delta^{j-i}$ shrinks. When the $d_j$ are similar, the decay wins and early
checking is right — which is {{ch:as-failures}}'s regime. When the $d_j$ vary by
an order of magnitude, $d_j$ dominates and the check belongs where the verifier is.

**Building versus placing.** The marginal value of raising $d_j$ with a check
already present is:

$$\frac{\partial V_j}{\partial d_j} = \sum_{i \le j}\Pr[\text{error from } i]\,\delta^{\,j-i}$$

which is *independent of $d_j$ itself*, while the residual uncaught mass is
proportional to $(1 - d_j)$. So the total-system gain from improving a verifier
scales with how much it currently misses:

$$\Delta S \;\propto\; (1 - d_j)\quad\Longrightarrow\quad \arg\max_j \Delta S = \arg\min_j d_j$$ (eq:check-strong-build-weak)

**Place checks where $d$ is high; build verifiers where $d$ is low.** The two
recommendations point at opposite ends of the pipeline and both follow from the
same expression.

## 6. Mathematical Foundation

Three extractions.

**The overstatement is structural, not sloppy.** From
{{eq:benchmark-share-follows-gradeability}}, the gap between headline and
time-weighted capability exists because automation quality and gradeability are
causally linked — checkability is what made the activity automatable. So the gap
cannot be closed by better benchmarking practice within the gradeable set; it
requires grading the ungradeable, which is the hard problem.

**Amdahl is indifferent to difficulty.** $S_i$ in
{{eq:amdahl-bounds-the-stack}} contains only $t_i$. This is worth stating because
technical impressiveness and project impact are routinely conflated, and the
formula says they are unrelated — a spectacular result on an activity worth $8\%$
of the time is worth $1.09\times$.

**The check-placement and verifier-building recommendations diverge because one
depends on $d_j$ and the other on $1 - d_j$.** That is the whole of
{{eq:check-strong-build-weak}}, and it explains a common organisational pattern:
teams optimise where they already measure, which is where the marginal return on
measurement is lowest.

## 7. Internal Mechanics

### 7.1 The stack, drawn by verifier strength

```mermaid {#fig:stack-verifiers caption="An analysis pipeline annotated by what can check each stage. The strong verifiers cluster at the ends; the middle, where most of the time goes, has almost nothing."}
flowchart LR
    A["access<br/>schema, row counts<br/>STRONG"] --> B["clean<br/>partial checks<br/>WEAK"]
    B --> C["explore<br/>nothing<br/>VERY WEAK"]
    C --> D["feature<br/>leakage checks<br/>WEAK"]
    D --> E["model<br/>held-out score<br/>STRONG"]
    E --> F["validate<br/>checks the model<br/>MEDIUM"]
    F --> G["conclude<br/>nothing<br/>VERY WEAK"]
```

Note where the weak stages sit: `clean`, `explore` and `conclude` together are more
than half the practitioner's time in {{sec:9-practical-example}}'s first listing,
and they are the three with the worst detection.

That co-location is not a coincidence either. **An activity that cannot be checked
also cannot be optimised away**, so it retains its time share while checkable
activities shrink.

### 7.2 Why validation is not a check on the analysis

The most common check in practice is at the validate stage: hold out data, score
the model, ship if it clears a bar. {{sec:9-practical-example}} gives that
$+6.5$ points on its own, which is real.

What it cannot do is catch an error upstream of the data it validates against. If
the join was wrong, the held-out rows are wrong the same way, and the score is a
faithful measurement of performance on a corrupted problem. The validation is
correct and the analysis is not.

This is why the model-stage check outperforms the validate-stage check in the
listing despite the latter being nominally about correctness: the model stage's
verifier catches things about the *fit* — degenerate features, impossible
separability, leakage signatures — that a clean held-out score cannot.

**A held-out score validates the model against the data; nothing in the standard
pipeline validates the data against the world.**

### 7.3 Leakage is the error the pipeline is shaped to hide

The characteristic feature-stage error deserves naming here because
{{ch:aids-automl}} takes it up in full.

Leakage — a feature carrying information about the target that will not exist at
prediction time — produces a *better* validation score, not a worse one. So the
one strong verifier in the middle of the pipeline is actively misled by it, and
the stronger the leakage the more convincing the result.

That inverts {{eq:pipeline-fails-at-the-weakest-verifier}}'s logic for one specific
error class: the model-stage check, which is the best single check in the listing,
is precisely where leakage looks like success. Leakage checks are therefore a
separate mechanism from validation, and a team that has a strong validation
practice and no leakage practice has a gap it cannot see.

### 7.4 The three gaps, and which one this part addresses

{{cite:testini2025dsautomation}} names three: activity coverage, the collaboration
spectrum, and substitution-versus-redesign.

**Activity coverage** is this chapter's subject and the part's organising problem.

**The collaboration spectrum** — evaluation only of pure assistance or full
autonomy — is {{ch:aids-oversight}}'s, and it matters because the useful regime is
almost certainly in between and almost nothing measures it.

**Substitution versus redesign** is the one this book is least equipped to
measure and probably the largest. If an analysis is cheap enough, you run twenty
instead of one, and the value is in the twenty rather than in each being faster.
{{ch:aids-autonomous}} takes up what that changes, and the honest position is that
none of the listings here capture it: they all price a fixed workload done faster.

### 7.5 What the four headline numbers actually license

Putting the benchmark results next to this chapter's arithmetic:

$40.08\%$ execution accuracy on realistic text-to-SQL
({{cite:li2023bird}}) is a strong result on an activity inside the
finding-and-accessing band — $15\%$ of the time, and a $1.18\times$ ceiling if
solved outright.

$30.5\%$ on agent-shaped data science tasks ({{cite:huang2024dacode}}) spans
several stages and is the closest thing here to an end-to-end figure. Read against
{{sec:9-practical-example}}'s second listing, a $30.5\%$ completion rate is
consistent with a pipeline whose middle stages have no verifier.

$16.9\%$ medal rate ({{cite:chan2024mlebench}}) is a modelling-stage result, on
the activity with the strongest verifier and the smallest time share — and its
most transferable finding is arguably that **scaffolding mattered as much as the
model**, which is {{ch:as-single-agent}}'s components result in a new setting.

None of them license "agents can do $X\%$ of data science", and the reason is
{{eq:gradeable-is-not-representative}}: they all measure inside the third of the
work that has an answer key.

### 7.6 What to do with this, practically

The chapter's recommendations are unusually concrete because the model is simple.

**Estimate your own $t_i$** from time tracking or from a week of honest
self-observation. The distribution varies by organisation far more than the
benchmark distribution does.

**Compute $S_i$ before adopting a tool.** A tool that perfectly automates something
worth $8\%$ of your time cannot deliver more than $1.09\times$, and knowing that in
advance prevents a category of disappointment.

**Place checks by detection strength**, which for most pipelines means the model
stage and the access stage rather than the middle.

**Build one verifier for a middle stage.** {{eq:check-strong-build-weak}} says
that is where the marginal return is, and almost nobody does it — a documented
expectation about what a cleaned table should satisfy, checked automatically, is
the highest-return unglamorous work available.

### 7.7 Why the ungradeable activities stay ungradeable

It would be convenient if the middle of the pipeline lacked verifiers because
nobody had got round to building them. The more likely explanation is worse, and
worth stating before {{ch:aids-agentic-eda}} takes it up in detail.

The gradeable activities share a property: **the question has an answer that
exists independently of the analyst.** A query returns particular rows. A model
scores a particular number on held-out data. You can be wrong about these, and the
world will say so.

The ungradeable ones do not have that property. Whether an exploration was
adequate depends on what you were going to conclude from it. Whether a conclusion
follows depends on what decision it will inform and what the cost of being wrong
is. These are not missing answer keys; they are questions whose answers are
*relative to a purpose the analysis has not been told*.

That has two consequences worth carrying through the part.

First, it means a verifier for these stages cannot be a checker in the sense the
strong stages use. It has to be something more like a *specification*: a statement,
made in advance, of what the analysis is for and what would count as sufficient.
Then "was this exploration adequate" becomes checkable against a stated standard
rather than against a Platonic one. That is a substantial ask and it is the form
{{sec:15-advanced-concepts}}'s open problem actually takes.

Second, it explains the failure mode of automating these stages without solving
that. An agent asked to "explore this dataset" will produce exploration — plots,
summaries, correlations — and there is no sense in which the output can be wrong,
because there was no standard. It will look like work, it will be delivered
promptly, and whether it was any use is a judgement the requester now has to make
with less context than before.

**An activity with no verifier does not become verifiable by being automated. It
becomes faster, which means more unverified output per unit time** — and
{{eq:check-strong-build-weak}} says that is the region where the errors already
survive.

## 8. Implementation

Two listings. The first prices the gap between benchmark attention and where time
goes. The second places verification in the pipeline.

```python {tier=A name=gradeable-is-not-representative}
"""Where the time goes, against where the automation landed.

cite:testini2025dsautomation surveyed how data science automation is evaluated and
found the coverage concentrated on a small subset of goal-oriented activities,
with data management and exploratory work largely ignored.

That is a selection effect with a mechanism. Benchmarks measure what can be
GRADED, and what can be graded is the part with a checkable answer: a query that
returns the right rows, a model that beats a threshold. The activities
practitioners spend most of their time on -- deciding what to ask, finding the
data, cleaning it -- have no reference answer, so they are not benchmarked, so
progress on them is not measured (eq:gradeable-is-not-representative).

This listing prices what that does to any claim of the form "agents now do X% of
data science".
"""
import numpy as np

# Activity shares are this listing's assumptions, stated so they can be
# challenged. The ordering -- data work dominating, modelling small -- is the
# consistent finding of practitioner surveys over two decades.
# (name, share of practitioner time, how gradeable, current automation quality)
ACTIVITIES = [
    ("framing the question",   0.12, 0.05, 0.20),
    ("finding and accessing",  0.15, 0.20, 0.35),
    ("cleaning and shaping",   0.26, 0.35, 0.55),
    ("exploration",            0.17, 0.10, 0.40),
    ("modelling",              0.11, 0.95, 0.75),
    ("validation",             0.08, 0.80, 0.60),
    ("communicating",          0.11, 0.15, 0.50),
]

TOTAL = sum(a[1] for a in ACTIVITIES)
assert abs(TOTAL - 1.0) < 1e-9, TOTAL

print("A data science project's activities, their share of practitioner time,")
print("how gradeable each is, and how well automation currently does it.")
print()
print(f"{'activity':>22}{'time share':>12}{'gradeable':>11}{'automation':>12}")
print("-" * 57)
for name, share, grade, auto in ACTIVITIES:
    print(f"{name:>22}{share:>12.0%}{grade:>11.0%}{auto:>12.0%}")

print()
print()
print("Benchmark attention follows gradeability, not time. Modelling a")
print("benchmark suite that allocates coverage in proportion to how gradeable")
print("an activity is:")
print()
g_total = sum(a[2] for a in ACTIVITIES)
print(f"{'activity':>22}{'time share':>12}{'benchmark share':>17}{'ratio':>9}")
print("-" * 60)
bench = {}
for name, share, grade, auto in ACTIVITIES:
    b = grade / g_total
    bench[name] = b
    print(f"{name:>22}{share:>12.0%}{b:>17.0%}{b / share:>9.2f}")

print()
print()
print("So what does a benchmark-weighted score actually claim? Comparing the")
print("headline number against the one that describes a practitioner's day:")
print()
bench_score = sum(bench[n] * a for n, _, _, a in ACTIVITIES)
time_score = sum(s * a for _, s, _, a in ACTIVITIES)
grade_frac = sum(s * g for _, s, g, _ in ACTIVITIES)
print(f"{'benchmark-weighted automation score':>40}{bench_score:>10.1%}")
print(f"{'time-weighted automation score':>40}{time_score:>10.1%}")
print(f"{'gradeable share of a practitioner day':>40}{grade_frac:>10.1%}")
print()
print(f"   The headline overstates the time-weighted figure by "
      f"{bench_score - time_score:.1f} points,")
print(f"   and only {grade_frac:.0%} of the day is gradeable at all.")

print()
print()
print("Amdahl's law on the analysis pipeline: perfect automation of one")
print("activity, and what it does to total project time.")
print()
print(f"{'activity fully automated':>26}{'time saved':>12}{'speedup':>10}")
print("-" * 48)
amd = {}
for name, share, grade, auto in ACTIVITIES:
    remaining = 1.0 - share
    amd[name] = (share, 1.0 / remaining)
    print(f"{name:>26}{share:>12.0%}{1.0 / remaining:>10.2f}x")

print()
print()
print("And the ceiling: automate everything above a gradeability threshold")
print("perfectly, leave the rest untouched.")
print()
print(f"{'gradeability threshold':>24}{'activities':>12}{'time covered':>14}"
      f"{'speedup':>10}")
print("-" * 61)
ceil = {}
for thr in (0.9, 0.7, 0.3, 0.15, 0.08):
    covered = [a for a in ACTIVITIES if a[2] >= thr]
    share = sum(a[1] for a in covered)
    sp = 1.0 / max(1.0 - share, 1e-9)
    ceil[thr] = (len(covered), share, sp)
    cell = "unbounded" if share > 0.999 else f"{sp:.2f}x"
    print(f"{thr:>24.0%}{len(covered):>12}{share:>14.0%}{cell:>10}")

print()
print()
print("The honest version of the automation figure: current capability applied")
print("to each activity's actual time share, swept over how good automation on")
print("the UNGRADEABLE activities becomes.")
print()
print(f"{'ungradeable automation':>24}{'time-weighted':>15}{'speedup':>10}")
print("-" * 50)
sw = {}
for lift in (0.0, 0.25, 0.50, 0.75, 1.0):
    total_auto = 0.0
    for name, share, grade, auto in ACTIVITIES:
        a = auto if grade >= 0.5 else auto + (1.0 - auto) * lift
        total_auto += share * a
    sw[lift] = (total_auto, 1.0 / max(1.0 - total_auto, 1e-9))
    print(f"{lift:>24.0%}{total_auto:>15.1%}"
          f"{1.0 / max(1.0 - total_auto, 1e-9):>10.2f}x")

print(f"""
The second table is the selection effect, made arithmetic. Modelling gets
{bench['modelling'] / 0.11:.2f} times its share of attention and validation
{bench['validation'] / 0.08:.2f} times, while framing the question gets
{bench['framing the question'] / 0.12:.2f} and exploration
{bench['exploration'] / 0.17:.2f}.

Nothing about that is a conspiracy. **Benchmarks measure what can be graded**, a
model score is a number and a well-posed question is not, so attention follows
gradeability (eq:gradeable-is-not-representative). cite:testini2025dsautomation
found exactly this distribution in the actual literature.

The consequence is in the summary. A benchmark-weighted reading of current
capability gives {bench_score:.1%}; weighting the same capabilities by where the
time actually goes gives {time_score:.1%}. **The headline overstates the
practitioner-relevant figure by {bench_score - time_score:.1f} points** -- and
only {grade_frac:.0%} of a practitioner's day is gradeable at all, so two thirds
of the work is in a region where progress is not being measured in either
direction.

The Amdahl table is the part worth carrying into an argument. Fully automating
modelling -- the single most-benchmarked activity, at
{bench['modelling'] / 0.11:.1f} times its time share of attention -- produces a
{amd['modelling'][1]:.2f}x speedup on the project.

That is not a claim that modelling automation is worthless. It is a claim that
**the size of a capability's benchmark presence tells you nothing about the size
of its effect**, because the benchmark measures a fraction of time that is small
precisely because it was tractable enough to automate first.

Cleaning and shaping, at {0.26:.0%} of the time, would give
{amd['cleaning and shaping'][1]:.2f}x. It gets {bench['cleaning and shaping'] / 0.26:.2f}
times its share of benchmark attention.

The threshold table gives the ceiling for a strategy of automating only what can
be checked. Perfect automation of everything above {0.7:.0%} gradeability -- two
activities -- gives {ceil[0.7][2]:.2f}x. **Every gradeable activity, automated
perfectly, is a {ceil[0.7][2]:.2f}x project speedup**, which is worth having and
is not the transformation the discourse describes.

The last table says where the transformation actually lives. Holding gradeable
automation where it is and lifting the UNGRADEABLE activities from
{0.0:.0%} to {1.0:.0%} of their remaining headroom takes the project from
{sw[0.0][1]:.2f}x to {sw[1.0][1]:.2f}x.

**The whole of the remaining opportunity is in the activities nobody can score.**
Which is an awkward place for it to be, because it means the field's ability to
measure its own progress runs out exactly where the value starts -- and it is why
this part spends more time on oversight and verification than on capability.""")
```

The second listing asks where a check belongs.

```python {tier=A name=pipeline-fails-at-the-weakest-verifier}
"""Errors in an analysis pipeline, and where a check is worth putting.

An analysis is a chain: access, clean, explore, feature, model, validate,
conclude. An error at any stage flows downstream, and the stages after it produce
confident, well-formed output built on it. A model trained on mis-joined data
converges; its validation score is real; the conclusion is wrong.

Two things vary by stage and decide everything:

  error rate       how often that stage gets it wrong
  verifier         whether anything can TELL, which ch:as-specialized found sets
                   a domain's ceiling and which differs sharply by stage

The end of the pipeline has the best verifier -- a held-out score is a real
number -- and the least ability to use it, because by then the error is a premise
that everything agrees with (eq:pipeline-fails-at-the-weakest-verifier).
"""
import numpy as np

rng = np.random.default_rng(4457)

M = 60000

# (stage, per-run error rate, detection rate of the check available there,
#  cost of a check in analyst-hours)
STAGES = [
    ("access",     0.10, 0.90, 0.4),   # row counts, schema, freshness: strong
    ("clean",      0.22, 0.45, 1.2),   # some checks; most errors are plausible
    ("explore",    0.14, 0.15, 0.8),   # almost nothing to check against
    ("feature",    0.16, 0.35, 1.0),   # leakage checks catch some
    ("model",      0.09, 0.80, 0.5),   # a held-out score is a real number
    ("validate",   0.07, 0.55, 0.6),   # checks the model, not the premises
    ("conclude",   0.12, 0.10, 0.3),   # no reference answer at all
]
N = len(STAGES)
DECAY = 0.62        # ch:as-failures: detection falls as an error ages downstream


def run(checks, m=M, decay=DECAY, fix=0.85):
    """`checks` is a set of stage indices where a check is performed. Returns
    (correct conclusions, checks run, analyst-hours spent)."""
    err_at = np.full(m, -1, dtype=np.int64)     # -1 = no outstanding error
    hours = np.zeros(m)
    n_checks = 0
    for i, (name, p_err, detect, cost) in enumerate(STAGES):
        # This stage may introduce an error, if one is not already loose.
        fresh = (err_at < 0) & (rng.random(m) < p_err)
        err_at[fresh] = i
        if i in checks:
            n_checks += 1
            hours += cost
            live = err_at >= 0
            lag = np.where(live, i - err_at, 0)
            # A check at stage i uses stage i's verifier, and its power decays
            # with how long the error has been propagating.
            p = detect * (decay ** np.clip(lag, 0, None))
            caught = live & (rng.random(m) < p) & (rng.random(m) < fix)
            err_at[caught] = -1
    return float((err_at < 0).mean()), n_checks, float(hours.mean())


print(f"{M:,} analyses through {N} stages. An error at any stage propagates,")
print("and every stage after it produces work that agrees with it.")
print()
print(f"{'stage':>12}{'error rate':>12}{'detection here':>16}{'hours':>8}")
print("-" * 48)
for name, p_err, detect, cost in STAGES:
    print(f"{name:>12}{p_err:>12.0%}{detect:>16.0%}{cost:>8.1f}")

print()
print()
print("No checks at all, then the check placements teams actually use.")
print()
PLANS = [
    ("none", set()),
    ("at the end (validate)", {5}),
    ("end pair (model+validate)", {4, 5}),
    ("early pair (access+clean)", {0, 1}),
    ("spread three", {0, 2, 4}),
    ("every stage", set(range(N))),
]
print(f"{'placement':>28}{'correct':>10}{'checks':>9}{'hours':>8}"
      f"{'correct/hour':>14}")
print("-" * 69)
tab = {}
for label, ck in PLANS:
    r = run(ck)
    tab[label] = r
    cell = "--" if r[2] <= 0 else f"{(r[0] - tab['none'][0]) / r[2]:.3f}"
    print(f"{label:>28}{r[0]:>10.1%}{r[1]:>9}{r[2]:>8.1f}{cell:>14}")

print()
print()
print("Two checks, placed every possible way. The best and worst pairs:")
print()
pairs = {}
for i in range(N):
    for j in range(i + 1, N):
        pairs[(i, j)] = run({i, j})[0]
order = sorted(pairs, key=lambda k: -pairs[k])
print(f"{'pair':>26}{'correct':>10}")
print("-" * 36)
for k in order[:3]:
    print(f"{f'{STAGES[k[0]][0]} + {STAGES[k[1]][0]}':>26}{pairs[k]:>10.1%}")
print(f"{'...':>26}")
for k in order[-2:]:
    print(f"{f'{STAGES[k[0]][0]} + {STAGES[k[1]][0]}':>26}{pairs[k]:>10.1%}")

print()
print()
print("What each single check is worth, placed alone -- which separates a")
print("stage's own detection power from how much damage is upstream of it.")
print()
base = run(set())[0]
print(f"{'check at':>12}{'correct':>10}{'gain':>9}{'gain/hour':>12}")
print("-" * 43)
single = {}
for i, (name, p_err, detect, cost) in enumerate(STAGES):
    r = run({i})
    single[name] = (r[0], r[0] - base, (r[0] - base) / cost)
    print(f"{name:>12}{r[0]:>10.1%}{r[0] - base:>+9.1%}"
          f"{(r[0] - base) / cost:>12.3f}")

print()
print()
print("And what a better verifier at the weakest stage would buy, against a")
print("better verifier at the stage that already has the best one.")
print()
print(f"{'intervention':>34}{'correct':>10}{'gain':>9}")
print("-" * 53)
allck = set(range(N))
base_all = run(allck)[0]
print(f"{'checks everywhere, as is':>34}{base_all:>10.1%}{'--':>9}")
imp = {}
for idx, label in ((2, "explore detection 15% -> 60%"),
                   (6, "conclude detection 10% -> 60%"),
                   (4, "model detection 80% -> 98%")):
    saved = STAGES[idx]
    STAGES[idx] = (saved[0], saved[1], 0.60 if idx != 4 else 0.98, saved[3])
    v = run(allck)[0]
    STAGES[idx] = saved
    imp[label] = (v, v - base_all)
    print(f"{label:>34}{v:>10.1%}{v - base_all:>+9.1%}")

print(f"""
The first placement table contains the result and the fourth explains it.

Checking only at the end -- "does the model validate?", which is the default
practice -- takes correctness from {tab['none'][0]:.1%} to
{tab['at the end (validate)'][0]:.1%}. Checking at every stage reaches
{tab['every stage'][0]:.1%} for {tab['every stage'][2]:.1f} analyst-hours.

The early pair, which the previous chapters' logic would recommend, gives
{tab['early pair (access+clean)'][0]:.1%} -- WORSE than the end pair's
{tab['end pair (model+validate)'][0]:.1%}, and at more than the cost.

That is not what ch:as-failures found about critics, and the difference is
instructive. There, every critic had the same detection rate and coverage decided
everything. Here **detection rates differ by a factor of nine across stages**, and
that dominates.

The single-check table shows it cleanly. A check at `model` alone is worth
{single['model'][1]:+.1%}; at `explore` alone, {single['explore'][1]:+.1%}; at
`conclude`, {single['conclude'][1]:+.1%}. Per analyst-hour the model check returns
{single['model'][2]:.3f} against {single['explore'][2]:.3f} for exploration.

The model stage has two things going for it at once: a genuine verifier -- a
held-out score is a real number -- and a position late enough that most upstream
errors have already been made and are available to catch. **Check where the
verifier is strong, not where the error is fresh**
(eq:pipeline-fails-at-the-weakest-verifier), which is the opposite of the
freshness intuition and follows directly from ch:as-specialized's ceiling result.

The pair table agrees: `model` appears in all three best pairs, and the two worst
both consist of weak-verifier stages.

But the last table reverses the advice as soon as the verifiers themselves are in
play, and this is the finding to take away.

With checks already at every stage, improving `conclude` detection from
{0.10:.0%} to {0.60:.0%} is worth {imp['conclude detection 10% -> 60%'][1]:+.1%}
and improving `explore` from {0.15:.0%} to {0.60:.0%} is worth
{imp['explore detection 15% -> 60%'][1]:+.1%}. Improving `model` from
{0.80:.0%} to {0.98:.0%} -- a larger relative gain in detector quality -- is worth
{imp['model detection 80% -> 98%'][1]:+.1%}.

So the two halves say different things and both are right:

**Given the verifiers you have, spend checking time where detection is strong.**
**Given the chance to build a verifier, build it where detection is weakest.**

The strong stages are near their ceiling and the weak ones are nowhere near it, so
the marginal return on a new verifier is inverted relative to the marginal return
on a new check. Teams routinely do the first and almost never do the second,
because building a check for "is this exploration sound" or "does this conclusion
follow" is hard and unglamorous, and adding one more model-validation metric is
neither.

Which is this part's argument in one table. The stages with no verifier are where
the time goes (the previous listing), where the errors survive (this one), and
where nobody is working.""")
```

## 9. Practical Example

The first listing assigns each activity a time share, a gradeability and a current
automation quality:

```
              activity  time share  gradeable  automation
---------------------------------------------------------
  framing the question         12%         5%         20%
  cleaning and shaping         26%        35%         55%
           exploration         17%        10%         40%
             modelling         11%        95%         75%
            validation          8%        80%         60%
```

Allocating benchmark coverage in proportion to gradeability:

```
              activity  time share  benchmark share    ratio
------------------------------------------------------------
  framing the question         12%               2%     0.16
  cleaning and shaping         26%              13%     0.52
           exploration         17%               4%     0.23
             modelling         11%              37%     3.32
            validation          8%              31%     3.85
```

**Attention follows gradeability, not time**
({{eq:benchmark-share-follows-gradeability}}) — which is
{{cite:testini2025dsautomation}}'s survey finding with a mechanism attached.

```
     benchmark-weighted automation score     60.8%
          time-weighted automation score     47.3%
    gradeable share of a practitioner day     33.3%
```

The headline overstates by $13.5$ points, and **two thirds of the day is in a
region where no benchmark reports progress in either direction**
({{eq:gradeable-is-not-representative}}).

Amdahl:

```
  activity fully automated  time saved   speedup
------------------------------------------------
      cleaning and shaping         26%      1.35x
                 modelling         11%      1.12x
                validation          8%      1.09x
```

**Fully automating modelling — the most-benchmarked activity — gives a
$1.12\times$ project speedup** ({{eq:amdahl-bounds-the-stack}}). The formula
contains only the time share: not difficulty, not impressiveness, not benchmark
prominence.

And the ceiling, against the opportunity:

```
  ungradeable automation  time-weighted   speedup
--------------------------------------------------
                      0%          47.3%      1.90x
                     50%          70.7%      3.41x
                    100%          94.1%     16.81x
```

**The whole of the remaining opportunity is in the activities nobody can score** —
which is where the field's ability to measure its own progress runs out.

The second listing runs analyses through seven stages with per-stage error and
detection rates:

```
       stage  error rate  detection here   hours
------------------------------------------------
      access         10%             90%     0.4
       clean         22%             45%     1.2
     explore         14%             15%     0.8
       model          9%             80%     0.5
    conclude         12%             10%     0.3
```

Placements:

```
                   placement   correct   checks   hours  correct/hour
---------------------------------------------------------------------
                        none     37.5%        0     0.0            --
       at the end (validate)     44.2%        1     0.6         0.113
   end pair (model+validate)     54.6%        2     1.1         0.156
   early pair (access+clean)     45.7%        2     1.6         0.051
                 every stage     68.0%        7     4.8         0.064
```

The early pair loses to the end pair, which is *not* what
{{ch:as-failures}} found about critics. The difference is that there every critic
had the same detection rate; here they differ by a factor of nine.

Single checks:

```
    check at   correct     gain   gain/hour
-------------------------------------------
      access     41.3%    +3.4%       0.086
     explore     39.5%    +1.7%       0.021
       model     49.4%   +11.6%       0.231
    conclude     39.1%    +1.2%       0.042
```

**Check where the verifier is strong, not where the error is fresh**
({{eq:pipeline-fails-at-the-weakest-verifier}}) — the model stage has both a real
verifier and a position late enough to have accumulated the upstream errors.

And then the reversal, with checks already everywhere:

```
                      intervention   correct     gain
-----------------------------------------------------
          checks everywhere, as is     68.2%       --
      explore detection 15% -> 60%     72.3%    +4.1%
     conclude detection 10% -> 60%     74.4%    +6.2%
        model detection 80% -> 98%     70.2%    +2.0%
```

**Given the verifiers you have, check where detection is strong. Given the chance
to build one, build it where detection is weakest**
({{eq:check-strong-build-weak}}). Teams routinely do the first and almost never the
second, because another model metric is easy and a check for "does this conclusion
follow" is not.

## 10. Production Considerations

Estimate your own activity time shares before evaluating any tool. The
distribution varies by organisation much more than benchmark coverage does.

Compute the Amdahl bound before adoption. A tool that perfectly automates $8\%$ of
your time cannot exceed $1.09\times$, and knowing that prevents predictable
disappointment.

Read any "agents do $X\%$ of data science" claim as "of the gradeable third", and
ask which activities the benchmark touched.

Place checks at the access and model stages first — the strong verifiers — rather
than distributing them evenly.

Do not treat a held-out score as a check on the analysis. It validates the model
against the data and nothing validates the data against the world.

Keep leakage checks separate from validation, since leakage makes validation look
better rather than worse.

Build one verifier for a middle stage: a documented, automatically-checked
expectation about what a cleaned table must satisfy. It is the highest-return
unglamorous work available.

## 11. Common Mistakes

**Reading benchmark progress as project progress.** The covariance between
gradeability and automation guarantees a gap.

**Conflating technical difficulty with impact.** Amdahl contains only the time
share.

**Checking only at the end.** $+6.7$ points against $+16.9$ for a well-placed pair.

**Distributing checks evenly.** Correct when detection rates are similar and wrong
when they differ by an order of magnitude.

**Treating validation as verification of the analysis.** It cannot see upstream
errors.

**Improving the verifier you already have.** The marginal return scales with what
it currently misses.

**Automating the checkable and declaring the problem solved.** That path's ceiling
is under $2\times$.

## 12. Failure Modes

*Confidently wrong analysis.* Every stage correct given a mistaken premise, with
the validation score real and the conclusion false.

*Leakage rewarded.* The strongest mid-pipeline verifier misled by the error class
it is least able to see.

*Benchmark-shaped tooling.* Investment concentrated where measurement exists, which
is where the returns are smallest.

*Unmeasured regression.* Quality falling in the ungradeable two thirds, with no
instrument that would show it.

*Adoption disappointment.* A real capability improvement producing a $1.1\times$
project effect and being judged a failure of the technology rather than of the
expectation.

## 13. Alternatives

**Redesign rather than substitution.** {{cite:testini2025dsautomation}}'s third
gap: if analyses are cheap, run twenty. This chapter's model prices a fixed
workload done faster and cannot see that value; {{ch:aids-autonomous}} takes it up.

**Human-AI collaboration in the middle.** The regime that chapter says nobody
evaluates — neither assistant nor autonomous — and the one where the ungradeable
activities are most plausibly addressed.

**Investing in data infrastructure instead.** Much of the cleaning stage's time
share is a symptom of upstream systems, and fixing those has the same Amdahl
numerator with none of the verification problems.

**Not automating the middle.** A defensible position: automate the checkable
stages, keep humans on the rest, and accept the sub-$2\times$ ceiling in exchange
for keeping the errors detectable.

## 14. Evaluation

Measure your activity time shares. Everything here scales with them and they are
cheap to estimate.

Measure per-stage error rates by auditing completed analyses — how often was the
final conclusion wrong, and at which stage did it go wrong.

Measure your detection rates directly: seed known errors at each stage and count
how many your current checks catch. This is the input to both halves of
{{eq:check-strong-build-weak}} and nobody has it.

Report the gradeable share of any capability claim, including your own.

And track conclusions that were later found wrong, by stage of origin. It is the
only ground truth in this chapter and it is recoverable from history.

## 15. Advanced Concepts

**Verifiers for exploration.** What would it mean to check that an exploration was
sound — coverage of the data, sensitivity of the conclusions to analytic choices?
{{eq:check-strong-build-weak}} says this is the highest-return open problem in the
part. {{maturity:RESEARCH FRONTIER}}.

**Automated assumption registries.** {{ch:as-long-running}}'s premise recording
applied to analyses: every assumption made, with a re-check procedure, so the
conclusion can be re-validated when the world moves. {{maturity:EMERGING}}.

**Grading the ungradeable by consequence.** Rather than checking an analysis
directly, track whether decisions made on it turned out well — a slow, noisy
verifier for the stages that have none.

**Measuring the redesign effect.** The value of running twenty analyses instead of
one is not captured by any benchmark or by this chapter, and quantifying it would
change what the field optimises.

## 16. Connection to Previous Chapters

{{ch:as-specialized}}'s verifier ceiling is this chapter's engine: applied across
analysis stages it explains the benchmark distribution, the error propagation, and
where investment should go.

{{ch:as-failures}}'s coverage-before-freshness result is qualified here — it holds
when detection rates are similar and inverts when they differ by an order of
magnitude, which in an analysis pipeline they do.

{{ch:as-single-agent}}'s components result reappears in
{{cite:chan2024mlebench}}'s finding that scaffolding mattered as much as the model.

{{ch:mcp-production}}'s saturating-benefit-against-linear-cost shape recurs as
Amdahl's bound on automating the checkable.

Ahead: {{ch:aids-text-to-sql}} takes up the best-measured activity in the stack and
asks why $40\%$ is the number; {{ch:aids-oversight}} takes up
{{cite:testini2025dsautomation}}'s collaboration gap.

## 17. Exercises

1. Substitute your own time shares into the first listing and recompute the Amdahl
   bounds. Which tool would you now not buy?

2. Add a stage-specific fix probability to the second listing — some caught errors
   are not repairable — and see whether the placement advice survives.

3. Model leakage explicitly: an error that *raises* the model-stage detection
   signal. Where should the check go then?

4. Search all three-check placements exhaustively and check whether the model stage
   is in every good triple.

5. Make the decay rate stage-dependent — some errors become invisible faster than
   others — and re-derive the placement.

6. Estimate the gradeable share for a domain other than data science and compare.

## 18. Interview Questions

1. Why does benchmark attention in data science not track where the time goes?

2. You perfectly automate model selection. How much faster is the project?

3. Your model validated cleanly and the conclusion was wrong. What happened?

4. Where would you put two checks in an analysis pipeline, and why not at the end?

5. You can improve one verifier. Which stage, and why is that different from where
   you would put a check?

6. What does a $30\%$ score on a data science agent benchmark license you to say?

## 19. Research Questions

1. What would a verifier for exploratory analysis check, and could it be
   automated?

2. Can decision outcomes serve as a delayed verifier for ungradeable stages, and at
   what noise level?

3. How much does the activity time distribution actually vary across organisations
   and domains?

4. What is the size of the redesign effect — running many cheap analyses — relative
   to the speedup effect?

5. Does the gradeability-attention correlation hold in other applied fields, or is
   data science unusual?

## 20. Chapter Summary

The published numbers for automated data science agree on a shape: $40.08\%$
execution accuracy against $92.96\%$ human on realistic text-to-SQL
({{cite:li2023bird}}), $30.5\%$ on agent-shaped analysis tasks
({{cite:huang2024dacode}}), a bronze medal in $16.9\%$ of Kaggle competitions
({{cite:chan2024mlebench}}). Impressive demonstrations, low completion rates.

The more useful finding is about what is measured at all.
{{cite:testini2025dsautomation}} found evaluation concentrated on goal-oriented
activities with data management and exploration largely ignored, and the mechanism
is {{ch:as-specialized}}'s: **benchmarks measure what can be graded.**
{{sec:9-practical-example}} finds modelling receiving $3.3\times$ its share of
attention and question-framing $0.16\times$
({{eq:benchmark-share-follows-gradeability}}), a benchmark-weighted capability
reading exceeding the time-weighted one by $13.5$ points, and **only $33\%$ of a
practitioner's day gradeable at all** ({{eq:gradeable-is-not-representative}}).

Then Amdahl. **Fully automating modelling gives a $1.12\times$ project speedup**
({{eq:amdahl-bounds-the-stack}}), because the formula contains only the time share
— and modelling's share is small precisely because it was checkable enough to
become efficient first. Automating everything checkable, perfectly, is under
$2\times$; the rest of the opportunity is in activities nobody can score.

On errors, an analysis is a chain in which every stage after a mistake produces
correct work given the mistake. Detection rates vary ninefold across stages, and
that dominates freshness: a single check at the model stage was worth $+11.6$
points against $+1.7$ at exploration, and the early pair lost to the end pair —
**check where the verifier is strong**
({{eq:pipeline-fails-at-the-weakest-verifier}}), which qualifies
{{ch:as-failures}}'s coverage rule for the case of unequal detectors.

And the reversal that matters most: with checks everywhere, improving the
*conclusion* stage's detection was worth $+6.2$ points against $+2.0$ for improving
the *model* stage's. **Place checks where detection is strong; build verifiers
where it is weak** ({{eq:check-strong-build-weak}}) — and the weak stages are also
where the time goes, where the errors survive, and where nobody is working.

## 21. Further Reading

{{cite:testini2025dsautomation}} is the framing paper for this part and its three
gaps organise {{ch:aids-oversight}} as well as this chapter.

{{cite:li2023bird}}, {{cite:huang2024dacode}} and {{cite:chan2024mlebench}} for the
three headline numbers — read {{cite:chan2024mlebench}} in particular for its human
baselines and its scaffolding comparison, which is more transferable than its
medal rate.

{{ch:as-specialized}} for the verifier ceiling that generates every result here,
and {{ch:as-failures}} for the propagation model this chapter qualifies.

{{cite:cemri2025mast}} for the failure taxonomy whose silent-error category is what
{{sec:9-practical-example}}'s second listing measures in an analysis setting.
