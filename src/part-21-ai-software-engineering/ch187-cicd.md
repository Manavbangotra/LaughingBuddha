---
id: aise-cicd
number: 187
part: XXI
tier: full
status: draft
requires: [gate-on-consequence, placement-beats-frequency,
           reversibility-dominates, check-strong-build-weak]
provides: [gate-by-blast-radius-not-author, narrow-gates-survive-habituation,
           automatability-is-verify-times-reverse, architecture-lacks-both,
           build-architectural-verifiers]
citations: [wang2025solvedcorrectly, jimenez2023swebench, becker2025devproductivity,
            chan2024mlebench, cemri2025mast]
---

## 1. Learning Objectives

By the end of this chapter you will be able to place review gates in a delivery
pipeline by blast radius rather than by author, and say why volume and defect rate
do not predict where a gate pays; explain why a broad gating policy is worse in a
running pipeline than on paper; rank software activities by automatability using two
properties rather than by difficulty; explain why architecture resists automation for
a reason that is not its difficulty; and identify the constructive response.

## 2. Why This Matters

The previous four chapters measured what an agent can produce. This one is about
what a pipeline should let through, and it opens with a question every team adopting
these tools answers immediately and usually wrongly: **which changes need a human?**

The instinctive answer is by author — humans merge freely, agents need approval.
{{sec:9-practical-example}} finds that costing $11.0$ review-minutes per change to
reach $12.3$ of expected escape cost, where **gating by blast radius costs $2.2$
minutes to reach $13.9$** ({{eq:gate-by-blast-radius-not-author}}). Roughly the same
protection for a fifth of the attention.

The ranking of where a gate pays is not predicted by anything intuitive. A gate on
schema migrations returns $33.8$ units of avoided cost per review-minute; a gate on
documentation returns $0.00$. Documentation is $22\%$ of change volume and ranks
last; schema migrations are $3\%$ and rank first. What predicts it is **escape cost
multiplied by what the automated checks do not already catch.**

And the broad policy is worse in practice than on paper. At full reviewer attention,
gating everything beats gating narrowly; at $30\%$ attention — which is what agent
volumes produce — it is $24.2$ against $13.9$
({{eq:narrow-gates-survive-habituation}}). **Gating everything consumes the attention
that makes gating work.**

The second half asks which activities should be automated at all, and finds
{{ch:as-specialized}}'s two properties deciding it. Software activities span an
enormous range on both: debugging is $92\%$ verifiable and $95\%$ reversible;
setting a service boundary is $12\%$ and $9\%$. Ranked by net benefit the activities
do not shade — **they fall off a cliff**
({{eq:automatability-is-verify-times-reverse}}), from $+10.5$ for implementing a
feature to $-2227.5$ for setting a service boundary.

**Architecture is the least automatable activity in software, and not because it is
the hardest** ({{eq:architecture-lacks-both}}).

## 3. Prerequisites

{{ch:ag-termination}}'s {{eq:gate-on-consequence}} and its habituation result, which
this chapter applies to a merge queue.

{{ch:as-long-running}}'s {{eq:placement-beats-frequency}} — the same finding in a
delivery pipeline.

{{ch:as-specialized}}'s {{eq:reversibility-dominates}} and its complementarity
result, which the second listing transplants.

{{ch:aids-stack}}'s {{eq:check-strong-build-weak}}, which supplies the constructive
half.

## 4. Intuitive Explanation

An agent opens a pull request. Should a human read it?

The policy most teams adopt is by author: agent changes require review, human changes
follow the normal process. It is administratively simple and it is the wrong axis,
because the author does not determine what happens when the change is wrong.

Consider two changes. One updates a docstring. One alters a database migration. Both
might be authored by an agent, both might be defective, and the consequences differ
by three orders of magnitude. A policy that reviews both equally has spent the same
attention on both.

{{sec:9-practical-example}} prices that. Gating by blast radius — schema migrations,
infrastructure config, dependency bumps — costs about a fifth of the review time of
gating everything and achieves comparable expected cost.

The ranking of what to gate is worth checking against intuition, because two natural
proxies both fail. **Volume** fails: documentation changes are the largest category
and the least worth reviewing. **Defect rate** fails: feature code has a high defect
rate and ranks third, because CI already catches most of it.

What works is escape cost times the share the automated checks miss. A schema
migration is rare, moderately defect-prone, poorly covered by CI, and catastrophic —
and that product is what puts it first by a factor of thousands.

Then the argument that makes narrow gating not merely cheaper but better.

{{ch:ag-termination}} measured reviewer attention falling with volume. A policy
gating every agent change puts five times the volume in front of the same reviewers,
so their catch rate falls. {{sec:9-practical-example}} finds the broad policy winning
at full attention and losing badly at realistic attention — **a policy that looks
safer on a spreadsheet is worse in a pipeline that runs.**

The second half asks a larger question: which parts of software engineering should be
automated at all?

{{ch:as-specialized}} found two properties deciding whether an agent can work
somewhere: can it check its work, and can it undo a mistake. It found them
*complementary* — fixing one alone bought almost nothing.

Software activities differ enormously on both, and they differ *together*.

**Debugging** has a failing test that says whether you succeeded and version control
that undoes the attempt. Verifiable and reversible.

**Setting a service boundary** has neither. No test tells you the boundary is wrong.
And by the time anyone knows, three teams have built against it, two APIs are public,
and the migration is a quarter of work.

Ranked by net benefit, {{sec:9-practical-example}} finds no gradual decline. Five
activities are strongly positive and three are strongly negative, with a factor of a
hundred between the last positive and the first negative.

Which gives the chapter's headline in a form worth stating carefully.
**Architecture resists automation not because it requires more intelligence but
because it is unverifiable and irreversible** — and those are properties of the
activity's *situation*, not of its difficulty.

That framing is more useful than a prohibition, because situations can be changed. A
verifier is worth $212.6$ on service boundaries and $0.1$ on bug fixing — largest
exactly where none exists. Layering rules enforced by an import checker. Latency
budgets asserted in contract tests. Schema compatibility checks that fail a migration
breaking its readers. **Each converts an unverifiable decision into a partly
verifiable one, and each is ordinary engineering.**

## 5. Formal Explanation

**Gate placement.** Let change types $i$ have volume share $v_i$, defect rate
$d_i$, automated catch $a_i$, review catch $r_i$ at cost $m$ minutes, and escape cost
$c_i$. Expected cost per change under gate set $G$:

$$C(G) = \sum_i v_i d_i (1-a_i)\big(1 - r_i \mathbb{1}[i \in G]\big) c_i, \qquad M(G) = m\sum_{i \in G} v_i$$

The value of gating type $i$ per review-minute is:

$$\frac{\Delta C_i}{\Delta M_i} = \frac{v_i d_i (1-a_i) r_i c_i}{m v_i} = \frac{d_i (1-a_i) r_i c_i}{m}$$ (eq:gate-by-blast-radius-not-author)

**Volume cancels.** The per-minute return contains no $v_i$, which is why
documentation — the largest category — ranks last, and why "we get a lot of these"
is not an argument for gating them. What remains is defect rate times *uncaught*
share times escape cost.

**Habituation.** With reviewer catch $r_i\alpha(V)$ where $\alpha$ decays in reviewed
volume $V = \sum_{i \in G} v_i$:

$$C(G) = \sum_i v_i d_i(1-a_i)\big(1 - r_i\alpha(V)\,\mathbb{1}[i \in G]\big)c_i$$ (eq:narrow-gates-survive-habituation)

Enlarging $G$ raises the number of gated types and lowers $\alpha$ for *all* of them.
So $\partial C/\partial G$ has two terms of opposite sign, and beyond some $V$ the
second dominates — **adding a type to the gate set makes the existing gates worse.**

**Automatability.** For an activity with verifiability $\nu$, reversibility $\rho$,
error cost $c$, and agent error rate $p$:

$$\mathbb{E}[\text{cost}] = p\nu c\lambda + p(1-\nu)c(1 - \kappa\rho)$$

with $\lambda$ the cost of a caught-and-retried attempt and $\kappa$ the recovery
discount. The second term dominates whenever $\nu$ is small, and it is scaled by
$(1-\kappa\rho)$ — so:

$$\mathbb{E}[\text{cost}] \approx pc(1-\nu)(1-\kappa\rho)$$ (eq:automatability-is-verify-times-reverse)

**The two properties enter multiplicatively as $(1-\nu)(1-\rho)$**, which is
{{ch:as-specialized}}'s complementarity: being weak on both is not additively but
multiplicatively bad, so the ordering by $\nu\rho$ produces a cliff rather than a
slope.

**Architecture's position.** Writing $\nu_A, \rho_A$ for architectural decisions and
$\nu_D, \rho_D$ for debugging:

$$\frac{(1-\nu_A)(1-\rho_A)}{(1-\nu_D)(1-\rho_D)} \approx \frac{0.88 \times 0.91}{0.08 \times 0.05} = 200$$ (eq:architecture-lacks-both)

and the error cost $c_A/c_D$ multiplies that again. **The gap is a product of three
factors, none of which is task difficulty.**

**The constructive term.** From
{{eq:automatability-is-verify-times-reverse}}, $\partial \mathbb{E}[\text{cost}] /
\partial \nu = -pc(1-\kappa\rho)$, which is *largest in magnitude where $\rho$ is
smallest*:

$$\arg\max_i \left|\frac{\partial \mathbb{E}[\text{cost}_i]}{\partial \nu_i}\right| = \arg\min_i \rho_i$$ (eq:build-architectural-verifiers)

**A verifier is worth most exactly where the activity is least reversible** — which
is {{eq:check-strong-build-weak}} again, and points at architecture.

## 6. Mathematical Foundation

Three extractions.

**Volume cancels in the gating decision.** {{eq:gate-by-blast-radius-not-author}}
contains no $v_i$, which is a stronger statement than "volume is a weak predictor" —
it says volume is *irrelevant* to whether a gate pays per minute, and appears only in
the total budget. Teams reason about gates in terms of how many changes they see,
which is the one quantity that does not enter.

**Enlarging a gate set has a negative term.**
{{eq:narrow-gates-survive-habituation}} makes broad gating self-undermining rather
than merely expensive: each added type dilutes the attention protecting the types
already gated. That is why the comparison flips with attention rather than merely
narrowing.

**Multiplicative complementarity produces a cliff.** From
{{eq:automatability-is-verify-times-reverse}}, cost scales as $(1-\nu)(1-\rho)$, so
an activity weak on both is worse than the sum of its weaknesses. This is why the
ranking has no middle: activities cluster at the ends because the properties
correlate and multiply.

## 7. Internal Mechanics

### 7.1 Classifying changes by blast radius

```mermaid {#fig:blast-radius caption="Changes ordered by what happens when one is wrong. The gate belongs on the right of this diagram, regardless of who authored the change."}
flowchart LR
    A["docs<br/>cost: 2"] --> B["tests<br/>cost: 5"]
    B --> C["bug fix<br/>cost: 90"]
    C --> D["feature<br/>cost: 120"]
    D --> E["dependency<br/>cost: 140"]
    E --> F["config / infra<br/>cost: 900"]
    F --> G["schema migration<br/>cost: 3,400"]
    G --> H(["gate here"])
```

The practical classification is mechanical and should be automated: **path-based
rules.** Changes touching migration directories, infrastructure definitions, lockfiles
or public API surfaces are gated; changes touching documentation, tests or internal
implementation are not. That is a CODEOWNERS-style rule, it needs no judgement per
change, and it implements {{eq:gate-by-blast-radius-not-author}} directly.

### 7.2 What raises $a_i$, and why it is the better investment

The per-minute return contains $(1 - a_i)$, so **improving the automated catch rate
for a change type removes the need to gate it.** For the types that rank highest,
that is where the leverage is:

**Schema migrations.** Run the migration against a copy of production data in CI, and
run the previous version's readers against the migrated schema. This is a
compatibility test and it is buildable.

**Config and infrastructure.** Plan-and-diff, policy checks, and a smoke test against
the changed configuration. Most config defects are detectable by something other than
a human reading YAML.

**Dependency bumps.** Run the full suite, check the changelog for breaking markers,
and diff the transitive tree. The $55\%$ catch rate in the model is low because most
teams run only their own tests.

Each of those moves a type down the gating table permanently, which is worth more
than reviewing it forever — and it is {{ch:as-specialized}}'s affordance-building
argument in a pipeline.

### 7.3 Where automated changes should stop by default

{{ch:aise-swe-agents}} found $29.6\%$ of plausible patches diverging behaviourally
from the human reference. That number is the argument for a specific pipeline
default.

**An agent's change should reach a human before it reaches production, and should
reach production before it reaches an irreversible state.** Those are two different
gates: review, and then a deployment strategy that permits rollback.

Concretely: automated changes deploy behind a flag, to a canary, or with an automatic
revert on error-rate regression. That is {{ch:as-specialized}}'s manufactured undo,
and it raises $\rho$ for the change *after* it has been merged — which
{{eq:automatability-is-verify-times-reverse}} says is worth as much as raising $\nu$
before.

The pipeline design implication: **the deployment mechanism is part of the agent's
safety story**, and a team with instant rollback can safely automate changes a team
without it cannot.

### 7.4 Architecture agents, honestly

Given {{eq:architecture-lacks-both}}, what should an agent do with architecture at
all? Three roles survive, and they are all advisory.

**Enumerate options.** Generating six plausible designs with their trade-offs is
useful, cheap, and carries no risk because nothing is decided. This is
{{ch:as-roles}}'s advise-rather-than-gate distinction.

**Surface consequences.** "This boundary means these two services share a
transaction" is a derivation from the design, and derivations are checkable.

**Audit against stated constraints.** If the constraints exist as artefacts — which
is {{sec:7-internal-mechanics}}'s recommendation — then checking a design against
them is verification rather than judgement.

What does not survive is **deciding**. Not because a model cannot produce a good
architecture, but because nothing will tell anyone it was wrong until reversal is
expensive.

### 7.5 Building the verifiers architecture lacks

{{eq:build-architectural-verifiers}} says this is where a verifier is worth most, and
the examples are more available than they look:

**Layering and dependency rules.** An import checker that fails a build when the
domain layer imports the web layer. Minutes to configure, permanent effect.

**Contract tests between services.** Consumer-driven contracts turn "did I break my
caller" from a judgement into a test.

**Schema compatibility gates.** A registry that refuses an incompatible schema is a
verifier for a decision that is otherwise found out in production.

**Performance budgets in CI.** A latency assertion converts a non-functional
requirement — the classic unverifiable property — into a failing build.

**Architecture decision records with checkable claims.** An ADR that states a
constraint in a form something can assert is a specification;
{{ch:aids-oversight}}'s conclusion about writing conventions down executably, in a
codebase.

**The organisations getting most from coding agents are the ones that did this**, and
the reason is measurable rather than cultural: they have moved several activities up
{{sec:9-practical-example}}'s table, permanently, for every agent and every engineer.

### 7.6 What the CI pipeline becomes

Putting the part together, the pipeline stops being a quality gate and becomes the
agent's *environment* — the thing that supplies verification, localisation and
recovery.

{{ch:aise-repo}} found reproduction the best localiser. {{ch:aise-swe-agents}} found
the test runner and the iteration loop mutually contingent.
{{ch:aise-testing}} found the suite's independence deciding what iteration means.
This chapter finds automated catch rate deciding what needs a human.

All four say the same thing about where to invest: **the pipeline is the scaffold.**
A team improving CI is improving the agent, and a team adopting an agent onto a weak
pipeline has adopted the part that does not work without the part it does not have.

### 7.7 What changes when the volume goes up

Every number in the first listing is a rate per change, and the policies that follow
from it are stable under volume. The pipeline itself is not, and it is worth being
explicit about what breaks first when a team goes from twenty pull requests a week to
two hundred.

**Review capacity binds before anything else.** It is the only fixed resource in the
system: CI scales with money, agents scale with money, reviewers do not. The gating
table is a way of allocating that fixed resource, and its whole value is that it
allocates against consequence rather than against arrival rate.

**CI cost becomes a real constraint.** A suite that takes twenty minutes is fine at
twenty changes a week and is four days of compute at two hundred with retries. The
temptation is to run less of it, which lowers $a_i$ for every change type at once and
moves everything up the gating table — precisely the wrong direction. Test selection
by change impact is the correct response and it is
{{ch:aise-repo}}'s structural analysis used for a second purpose.

**Merge conflicts become a systemic cost.** Agents working in parallel on the same
repository produce changes that pass individually and conflict jointly, which is
{{ch:as-failures}}'s correlated-failure structure in a merge queue: independent
agents drawing on the same context produce overlapping edits far more often than
independent humans would.

**And the queue becomes a place where things wait.** {{ch:as-long-running}}'s
wall-clock argument applies: a change that sits for a day while its reviewer works
through a backlog has consumed a day of latency that no token count records, and the
staleness of the branch grows with it.

The composite effect is that **a pipeline tuned for human volume degrades in several
independent ways at agent volume**, and the degradation is not gradual in each. That
argues for measuring the pipeline's own throughput and latency as a first-class
metric before increasing agent output, rather than discovering the ceiling by
reaching it.

## 8. Implementation

Two listings. The first places gates in a delivery pipeline. The second ranks
activities by automatability.

```python {tier=A name=gate-by-blast-radius-not-author}
"""Where the gate goes in a pipeline that automated changes flow through.

ch:ag-termination found that gating everything is close to gating nothing, and
ch:as-long-running found placement worth an eightfold review budget over frequency.
This listing applies both to a delivery pipeline, where the natural instinct is to
gate by AUTHOR -- humans merge freely, agents need approval -- and the measurements
say to gate by BLAST RADIUS instead (eq:gate-by-blast-radius-not-author).

A change flows: propose -> automated checks -> human review -> merge -> deploy. Each
stage can catch a defect, and each costs something. The catch rates differ by change
type, because a type checker has a lot to say about a refactor and nothing to say
about a config value.
"""
M = 50000               # changes per period, for the volume framing

# (change type, share of volume, defect rate, automated catch, review catch,
#  cost if it reaches production)
CHANGES = [
    ("docs and comments",   0.22, 0.04, 0.10, 0.55,    2.0),
    ("test-only",           0.14, 0.09, 0.72, 0.60,    5.0),
    ("dependency bump",     0.11, 0.16, 0.55, 0.35,  140.0),
    ("bug fix",             0.25, 0.14, 0.68, 0.62,   90.0),
    ("feature code",        0.19, 0.19, 0.61, 0.58,  120.0),
    ("config / infra",      0.06, 0.21, 0.24, 0.45, 900.0),
    ("schema migration",    0.03, 0.24, 0.31, 0.66, 3400.0),
]
REVIEW_MIN = 11.0       # analyst-minutes for a human review


def run(gated, attention=1.0):
    """`gated` is the set of change types a human reviews. Returns
    (expected cost per change, review minutes per change, escape rate).

    Computed exactly rather than sampled: the model is a product of
    independent probabilities, so simulation would only add noise to a
    quantity with a closed form -- and the noise swamps the small-volume
    types that turn out to matter most.
    """
    total_cost = 0.0
    minutes = 0.0
    escapes = 0.0
    for name, share, p_def, auto, rev, cost in CHANGES:
        p_escape = p_def * (1.0 - auto)
        if name in gated:
            minutes += share * REVIEW_MIN
            p_escape *= (1.0 - min(rev * attention, 1.0))
        escapes += share * p_escape
        total_cost += share * p_escape * cost
    return total_cost, minutes, escapes


ALL = {c[0] for c in CHANGES}

print(f"{M:,} changes through a pipeline. Each type has its own defect")
print("rate, its own automated catch rate, and its own cost if it escapes.")
print()
print(f"{'change type':>20}{'volume':>9}{'defects':>9}{'auto catch':>12}"
      f"{'escape cost':>13}")
print("-" * 63)
for name, share, p_def, auto, rev, cost in CHANGES:
    print(f"{name:>20}{share:>9.0%}{p_def:>9.0%}{auto:>12.0%}{cost:>13,.0f}")

print()
print()
print("Gating policies. 'Cost' is expected escape cost per change; 'minutes'")
print("is human review time per change.")
print()
print(f"{'policy':>34}{'cost/change':>13}{'minutes':>10}{'escapes':>10}")
print("-" * 67)
POLICIES = [
    ("gate nothing", set()),
    ("gate everything", ALL),
    ("gate by author (all agent changes)", ALL),
    ("gate the big diffs", {"feature code", "bug fix"}),
    ("gate by blast radius", {"schema migration", "config / infra",
                              "dependency bump"}),
]
tab = {}
for label, g in POLICIES:
    r = run(g)
    tab[label] = r
    print(f"{label:>34}{r[0]:>13.1f}{r[1]:>10.1f}{r[2]:>10.2%}")

print()
print()
print("Cost per review-minute spent, which is the comparison that matters when")
print("review capacity is the constraint.")
print()
none = tab["gate nothing"][0]
print(f"{'policy':>34}{'cost avoided':>14}{'minutes':>10}{'per minute':>12}")
print("-" * 70)
for label, g in POLICIES:
    r = tab[label]
    if r[1] <= 0:
        continue
    print(f"{label:>34}{none - r[0]:>14.1f}{r[1]:>10.1f}"
          f"{(none - r[0]) / r[1]:>12.2f}")

print()
print()
print("Every single-type gate, ranked. This is the table a team should build")
print("for its own pipeline.")
print()
print(f"{'gate only this type':>20}{'cost avoided':>14}{'minutes':>10}"
      f"{'per minute':>12}")
print("-" * 56)
single = {}
for name, share, p_def, auto, rev, cost in CHANGES:
    r = run({name})
    single[name] = ((none - r[0]), r[1], (none - r[0]) / max(r[1], 1e-9))
    print(f"{name:>20}{none - r[0]:>14.2f}{r[1]:>10.2f}"
          f"{(none - r[0]) / max(r[1], 1e-9):>12.2f}")

print()
print()
print("Note what does NOT predict the ranking.")
print()
order = sorted(single, key=lambda k: -single[k][2])
look = {c[0]: c for c in CHANGES}
print(f"{'rank':>6}{'type':>20}{'per minute':>12}{'volume':>9}{'defects':>9}"
      f"{'escape cost':>13}")
print("-" * 69)
for i, name in enumerate(order, 1):
    c = look[name]
    print(f"{i:>6}{name:>20}{single[name][2]:>12.2f}{c[1]:>9.0%}"
          f"{c[2]:>9.0%}{c[5]:>13,.0f}")

print()
print()
print("And what happens under ch:ag-termination's habituation, which is what")
print("gating everything actually produces at agent volumes.")
print()
print(f"{'attention':>11}{'gate everything':>18}{'gate by blast radius':>22}")
print("-" * 51)
hb = {}
for a in (1.0, 0.6, 0.3, 0.12):
    x = run(ALL, attention=a)[0]
    y = run({"schema migration", "config / infra", "dependency bump"},
            attention=min(a * 3.5, 1.0))[0]
    hb[a] = (x, y)
    print(f"{a:>11.0%}{x:>18.1f}{y:>22.1f}")

print(f"""
The policy table has the finding in two rows. Gating by AUTHOR -- every agent
change reviewed -- costs {tab['gate by author (all agent changes)'][1]:.1f} review
minutes per change and leaves {tab['gate by author (all agent changes)'][0]:.1f} of
expected escape cost. Gating by BLAST RADIUS costs
{tab['gate by blast radius'][1]:.1f} minutes and leaves
{tab['gate by blast radius'][0]:.1f}.

Roughly the same protection for about a fifth of the review time
(eq:gate-by-blast-radius-not-author), which is ch:as-long-running's
placement-beats-frequency result arriving in a delivery pipeline.

The per-minute table makes the ranking stark. A gate on schema migrations returns
{single['schema migration'][2]:.1f} units of avoided cost per review-minute; a gate
on documentation returns {single['docs and comments'][2]:.2f}. **A factor of
thousands separates the best gate from the worst**, and both are gates on changes an
agent might author.

The next table says what does not predict that ranking, and it is worth checking
against intuition.

Not VOLUME: documentation is {0.22:.0%} of changes and ranks last. Not DEFECT RATE:
feature code has a {0.19:.0%} rate and ranks third. Not who wrote it -- the model
does not contain an author variable at all.

What predicts it is **escape cost multiplied by what the automated checks do not
already catch.** Schema migrations are {0.03:.0%} of volume with a {0.31:.0%}
automated catch rate and a {3400:,} escape cost, and that product is the whole
ranking.

Which gives the practical instruction: **build this table for your own pipeline.**
It requires three numbers per change type -- how often they are wrong, what your CI
catches, what it costs when one escapes -- and the first two are recoverable from
history.

The last table is why the author-based policy is worse than it looks on paper.

At full attention, gating everything costs {hb[1.0][0]:.1f} against blast-radius
gating's {hb[1.0][1]:.1f} -- the broad policy is nominally better. At
{0.30:.0%} attention it is {hb[0.3][0]:.1f} against {hb[0.3][1]:.1f}.

**Gating everything consumes the attention that makes gating work.**
ch:ag-termination measured that curve directly; here it means a policy that looks
safer on a spreadsheet is worse in a pipeline that actually runs, because the
reviewers are seeing five times the volume and reading it five times less carefully.

The blast-radius policy is robust to habituation for a mechanical reason: its
reviewers see about {0.20:.0%} of the changes, so their attention holds where the
broad policy's does not. That is the argument for narrow gating stated as an
attention budget rather than as a preference.""")
```

The second listing asks which activities should be automated at all.

```python {tier=A name=automatability-is-verify-times-reverse}
"""Architecture, which is the least automatable activity in software and not
because it is the hardest.

ch:as-specialized found two properties deciding whether an agent can work in a
domain: whether it can CHECK its work, and whether it can UNDO a mistake. It found
them complementary -- fixing either alone bought almost nothing and fixing both
bought fifty-four points.

Software engineering activities differ enormously on both, and they differ TOGETHER.
Debugging has a failing test (verifiable) and version control (reversible).
Architecture has neither: no test tells you a service boundary is wrong, and by the
time you find out, three teams have built against it
(eq:automatability-is-verify-times-reverse).

This listing places the activities and prices automating each.
"""
# (activity, share of engineering effort, verifiability, reversibility,
#  cost of a wrong decision, how much cheaper an agent makes the attempt)
ACTIVITIES = [
    ("fix a reported bug",     0.19, 0.92, 0.95,    120.0, 0.45),
    ("write a test",           0.09, 0.70, 0.97,     40.0, 0.35),
    ("refactor a module",      0.11, 0.62, 0.90,    260.0, 0.50),
    ("implement a feature",    0.26, 0.58, 0.78,    380.0, 0.55),
    ("choose a dependency",    0.04, 0.31, 0.34,   2600.0, 0.70),
    ("design a data model",    0.06, 0.24, 0.19,   7000.0, 0.75),
    ("set a service boundary", 0.05, 0.12, 0.09,  21000.0, 0.80),
    ("everything else",        0.20, 0.55, 0.70,    200.0, 0.70),
]

P_AGENT_WRONG = 0.30    # an agent's decision is wrong this often
P_HUMAN_WRONG = 0.17


def expected(activity, who, guarded=False):
    """Expected cost of one decision. A wrong decision is caught by the
    verifier with probability `verifiability`; an uncaught wrong decision is
    undone with probability `reversibility`, at a fraction of its full cost."""
    name, share, ver, rev, cost, _ = activity
    p_wrong = P_AGENT_WRONG if who == "agent" else P_HUMAN_WRONG
    v = min(ver * 1.35, 0.97) if guarded else ver
    caught = p_wrong * v
    escaped = p_wrong * (1 - v)
    # Caught costs a retry; escaped costs the full amount, discounted by how
    # recoverable it is.
    return caught * cost * 0.12 + escaped * cost * (1 - rev * 0.85)


print("Software activities, by whether a mistake can be DETECTED and whether it")
print("can be UNDONE -- ch:as-specialized's two binding properties.")
print()
print(f"{'activity':>24}{'effort':>9}{'verifiable':>12}{'reversible':>12}"
      f"{'cost if wrong':>15}")
print("-" * 72)
for name, share, ver, rev, cost, _ in ACTIVITIES:
    print(f"{name:>24}{share:>9.0%}{ver:>12.0%}{rev:>12.0%}{cost:>15,.0f}")

print()
print()
print("The product of the two is what ch:as-specialized found decisive, and it")
print("orders the activities cleanly.")
print()
print(f"{'activity':>24}{'verify x reverse':>18}{'agent cost':>13}"
      f"{'human cost':>13}{'ratio':>8}")
print("-" * 76)
tab = {}
for a in ACTIVITIES:
    name = a[0]
    ag = expected(a, "agent")
    hu = expected(a, "human")
    tab[name] = (a[2] * a[3], ag, hu, ag / max(hu, 1e-9))
    print(f"{name:>24}{a[2] * a[3]:>18.3f}{ag:>13.1f}{hu:>13.1f}"
          f"{ag / max(hu, 1e-9):>8.2f}")

print()
print()
print("Net effect of automating each activity: the agent is cheaper to run and")
print("more often wrong, so the question is whether the saving covers the risk.")
print()
print(f"{'activity':>24}{'effort saved':>14}{'extra risk':>12}{'net':>10}"
      f"{'verdict':>12}")
print("-" * 72)
net = {}
HOURLY = 95.0
for a in ACTIVITIES:
    name, share, ver, rev, cost, cheaper = a
    # Effort saved, in the same units as risk, per decision.
    saved = cheaper * 4.0 * HOURLY / 10.0
    extra = expected(a, "agent") - expected(a, "human")
    net[name] = (saved, extra, saved - extra)
    print(f"{name:>24}{saved:>14.1f}{extra:>12.1f}{saved - extra:>10.1f}"
          f"{('automate' if saved > extra else 'do not'):>12}")

print()
print()
print("Ranked by net, against the two properties that produced it.")
print()
order = sorted(net, key=lambda k: -net[k][2])
look = {a[0]: a for a in ACTIVITIES}
print(f"{'rank':>6}{'activity':>24}{'net':>10}{'verifiable':>12}"
      f"{'reversible':>12}")
print("-" * 64)
for i, name in enumerate(order, 1):
    a = look[name]
    print(f"{i:>6}{name:>24}{net[name][2]:>10.1f}{a[2]:>12.0%}{a[3]:>12.0%}")

print()
print()
print("What a verifier would buy where one could be built -- ch:aids-stack's")
print("check-strong-build-weak rule, applied here.")
print()
print(f"{'activity':>24}{'as is':>10}{'with a verifier':>18}{'gain':>10}")
print("-" * 62)
gd = {}
for a in ACTIVITIES:
    name = a[0]
    base = expected(a, "agent")
    guard = expected(a, "agent", guarded=True)
    gd[name] = (base, guard, base - guard)
    print(f"{name:>24}{base:>10.1f}{guard:>18.1f}{base - guard:>10.1f}")

print()
print()
print("And the effort-weighted picture, which says how much of software")
print("engineering sits in each regime.")
print()
hi = sum(a[1] for a in ACTIVITIES if a[2] * a[3] >= 0.40)
mid = sum(a[1] for a in ACTIVITIES if 0.10 <= a[2] * a[3] < 0.40)
lo = sum(a[1] for a in ACTIVITIES if a[2] * a[3] < 0.10)
print(f"{'regime':>34}{'share of effort':>18}")
print("-" * 54)
print(f"{'verifiable and reversible':>34}{hi:>18.0%}")
print(f"{'partly one or the other':>34}{mid:>18.0%}")
print(f"{'neither':>34}{lo:>18.0%}")

print(f"""
The ranking table has a cliff in it rather than a slope, and that is the finding.

Implementing a feature nets {net['implement a feature'][2]:+.1f}; choosing a
dependency nets {net['choose a dependency'][2]:+.1f}; setting a service boundary
nets {net['set a service boundary'][2]:+.1f}. **The activities do not shade from
automatable to less automatable. They fall off a cliff**, and the cliff is exactly
where the verify-times-reverse product drops below about
{0.10:.2f} (eq:automatability-is-verify-times-reverse).

That product is ch:as-specialized's finding transplanted. There, fixing observation
alone bought {0.3:+.1f} points and fixing undo alone bought
{12.3:+.1f}, and fixing both bought {54.5:+.1f} -- the properties were
complementary, so a domain weak on both was catastrophically weak. Software
activities span that whole range internally.

Debugging sits at {look['fix a reported bug'][2]:.0%} verifiable and
{look['fix a reported bug'][3]:.0%} reversible: a failing test says whether you
succeeded and version control undoes the attempt. Setting a service boundary sits at
{look['set a service boundary'][2]:.0%} and {look['set a service boundary'][3]:.0%}:
no test says a boundary is wrong, and by the time anyone knows, three teams have
built against it.

**Architecture is the least automatable activity in software, and not because it is
the hardest.** It is because it combines the two properties an agent needs least of
and needs most.

The verifier table is ch:aids-stack's check-strong-build-weak rule, and it points
where that rule always points. A verifier is worth {gd['fix a reported bug'][2]:.1f}
on bug fixing, where one already exists, and {gd['set a service boundary'][2]:.1f} on
service boundaries, where none does.

Which is the constructive reading of this whole chapter, and it is more useful than
"do not automate architecture". **The reason architecture resists automation is a
missing verifier, and verifiers for architectural properties are buildable.** A
layering rule enforced by an import checker. A latency budget asserted in a
contract test. A schema compatibility check that fails a migration that breaks
readers. Each converts an unverifiable decision into a partly verifiable one, and
each is ordinary engineering.

That is also the answer to why some organisations get much more out of coding agents
than others, and it is not model access. **A codebase with executable architectural
constraints has moved several activities up this table**, permanently, for every
agent and every engineer.

The last table sizes the opportunity honestly. About {hi:.0%} of engineering effort
sits in the verifiable-and-reversible regime where agents work well,
{mid:.0%} partly, and {lo:.0%} in the regime where they do not.

So the effort-weighted picture is encouraging and the risk-weighted one is not: the
{lo:.0%} carries the decisions whose costs are measured in
{look['set a service boundary'][4] / look['fix a reported bug'][4]:.0f} times a bug
fix. **Automate the majority of the effort; keep the minority that carries the
consequences** -- which is ch:aids-oversight's divide-by-gradeability rule, arriving
independently in a second domain and with reversibility added as a second
criterion.""")
```

## 9. Practical Example

The first listing runs changes through a pipeline:

```
         change type   volume  defects  auto catch  escape cost
---------------------------------------------------------------
   docs and comments      22%       4%         10%             2
     dependency bump      11%      16%         55%           140
        feature code      19%      19%         61%           120
      config / infra       6%      21%         24%           900
    schema migration       3%      24%         31%         3,400
```

Policies:

```
                            policy  cost/change   minutes   escapes
-------------------------------------------------------------------
                      gate nothing         26.3       0.0     4.72%
gate by author (all agent changes)         12.3      11.0     2.30%
                gate the big diffs         21.2       4.8     3.65%
              gate by blast radius         13.9       2.2     3.98%
```

**Roughly the same protection for a fifth of the review time**
({{eq:gate-by-blast-radius-not-author}}).

Ranked per review-minute:

```
  rank                type  per minute   volume  defects  escape cost
---------------------------------------------------------------------
     1    schema migration       33.78       3%      24%        3,400
     2      config / infra        5.88       6%      21%          900
     3        feature code        0.47      19%      19%          120
     7   docs and comments        0.00      22%       4%            2
```

**Volume does not predict it** — documentation is the largest category and ranks
last, and volume cancels out of the per-minute expression entirely. What predicts it
is escape cost times what CI misses.

And under realistic attention:

```
  attention   gate everything  gate by blast radius
---------------------------------------------------
       100%              12.3                  13.9
        30%              24.2                  13.9
        12%              27.3                  22.9
```

**Gating everything consumes the attention that makes gating work**
({{eq:narrow-gates-survive-habituation}}) — the broad policy wins on paper and loses
in a pipeline that runs.

The second listing places activities:

```
                activity   effort  verifiable  reversible  cost if wrong
------------------------------------------------------------------------
      fix a reported bug      19%         92%         95%             120
     implement a feature      26%         58%         78%             380
     choose a dependency       4%         31%         34%           2,600
  set a service boundary       5%         12%          9%          21,000
```

Ranked by net benefit of automating:

```
  rank                activity       net  verifiable  reversible
----------------------------------------------------------------
     2      fix a reported bug      15.1         92%         95%
     5     implement a feature      10.5         58%         78%
     6     choose a dependency    -151.8         31%         34%
     8  set a service boundary   -2227.5         12%          9%
```

**A cliff, not a slope** ({{eq:automatability-is-verify-times-reverse}}) — because
the two properties enter multiplicatively, so weakness on both is worse than the sum
of the weaknesses.

**Architecture is the least automatable activity in software and not because it is
the hardest** ({{eq:architecture-lacks-both}}): a service boundary has no test that
says it is wrong, and by the time anyone knows, three teams have built against it.

The constructive half:

```
                activity     as is   with a verifier      gain
--------------------------------------------------------------
      fix a reported bug       4.5               4.4       0.1
     implement a feature      24.1              19.0       5.0
  set a service boundary    5210.6            4998.0     212.6
```

**A verifier is worth most exactly where none exists**
({{eq:build-architectural-verifiers}}) — $212.6$ against $0.1$.

And the sizing:

```
                            regime   share of effort
------------------------------------------------------
         verifiable and reversible               65%
           partly one or the other               24%
                           neither               11%
```

$65\%$ of effort sits where agents work well and $11\%$ carries the decisions costing
$175$ times a bug fix.

## 10. Production Considerations

Gate by blast radius, not by author. Implement it as path-based rules — migration
directories, infrastructure definitions, lockfiles, public API surfaces — so it needs
no per-change judgement.

Build the gating table for your own pipeline. It needs defect rate, CI catch rate and
escape cost per change type, and the first two are recoverable from history.

Raise the automated catch rate on the high-ranking types rather than reviewing them
forever. Migration compatibility tests, config plan-and-diff, dependency tree diffs.

Do not gate documentation or test-only changes. They are the largest volume and the
smallest return.

Give automated changes a rollback path — flag, canary, automatic revert. It raises
reversibility after merge, which is worth as much as verification before.

Let architecture agents enumerate, derive consequences and audit against stated
constraints. Do not let them decide.

Build architectural verifiers: import checkers, contract tests, schema compatibility
gates, performance budgets. They are worth most where nothing exists.

And treat the CI pipeline as the agent's scaffold rather than as its examiner.

## 11. Common Mistakes

**Gating by author.** Five times the attention for comparable protection.

**Gating by volume.** Volume cancels out of the per-minute return entirely.

**Gating everything.** Self-undermining, because it dilutes the attention protecting
the gates that matter.

**Reviewing a change type forever instead of raising its CI catch rate.**

**Deploying agent changes without a rollback path.** Reversibility after merge is
half the equation.

**Letting an architecture agent decide.** No verifier and no undo.

**Adopting an agent onto a weak pipeline.** The pipeline is the scaffold.

## 12. Failure Modes

*Rubber-stamped merge queue.* A broad gate under habituation, producing an audit
trail and few catches.

*Migration shipped unreviewed.* The highest-cost category, passing a policy that
weighted by diff size.

*Config defect in production.* $24\%$ automated catch and a $900$ escape cost — the
second-worst cell in the table.

*Architecture drift.* Boundaries eroded by many individually-reasonable automated
changes, with nothing asserting the constraint.

*Irreversible automated deploy.* A correct-looking change with no path back.

*Pipeline debt.* An agent underperforming for reasons attributed to the model and
caused by CI.

## 13. Alternatives

**Gate by change size.** Better than by author, worse than by blast radius —
{{sec:9-practical-example}} puts it at $21.2$ against $13.9$.

**Gate probabilistically.** Review a random sample of everything, which
{{ch:aids-oversight}} showed is a weaker second pass than a targeted full review.

**Progressive autonomy.** Start gated, relax per change type as the escape record
accumulates — the empirical version of this chapter's table.

**Full autonomy with strong rollback.** Raise $\rho$ far enough that escapes are
cheap, and skip review. Viable for stateless services and not for migrations.

**Human-authored architecture, agent-implemented.** {{ch:aids-oversight}}'s
divide-by-gradeability, applied to the one activity that fails both criteria.

## 14. Evaluation

Measure defect escape rate and cost by change type. Both are recoverable from
incident history, and together they are the gating table.

Measure your CI catch rate per type by seeding defects.

Measure reviewer catch rate against review volume, to locate your habituation curve.

Measure time-to-rollback. It is the reversibility term and it decides what can be
automated safely.

Track architectural constraint violations that reached production, since they are the
failure with no test.

And measure the share of your effort in each regime, so the sizing is yours rather
than this chapter's.

## 15. Advanced Concepts

**Learned gate placement.** Estimating escape cost per change from historical
incidents, so the gating table maintains itself. {{maturity:EMERGING}}.

**Automated blast-radius inference.** Deriving what a change can affect from static
analysis rather than from path rules — {{ch:aise-repo}}'s impact analysis used for
gating.

**Executable architecture decision records.** ADRs carrying machine-checkable
constraints, which is the artefact {{sec:7-internal-mechanics}} argues is missing.
{{maturity:EMERGING}}.

**Verifiers for design properties.** Coupling, cohesion, boundary stability expressed
as assertions rather than as reviews. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:ag-termination}}'s consequence gate and habituation result combine here into a
single policy, and the habituation term is what makes narrow gating better rather
than merely cheaper.

{{ch:as-long-running}}'s placement-beats-frequency result reappears in a merge queue
with the same eightfold-budget shape.

{{ch:as-specialized}}'s two properties transplant directly and explain the cliff:
they are complementary, so weakness on both multiplies.

{{ch:aids-stack}}'s check-strong-build-weak rule points at architecture, which is
where the chapter's constructive recommendation comes from.

{{ch:aise-repo}}, {{ch:aise-swe-agents}} and {{ch:aise-testing}} all converge here:
the pipeline is the scaffold.

Ahead: {{ch:aise-autonomy}} closes the part by asking what the measured productivity
evidence actually supports.

## 17. Exercises

1. Build the gating table for your repository from incident history and compare with
   your current policy.

2. Derive the volume-cancellation result from
   {{eq:gate-by-blast-radius-not-author}} and check it against the ranking.

3. Add a migration compatibility test and measure how far it moves that row.

4. Model progressive autonomy: gates relaxed as escape records accumulate. How long
   until the policy converges to the optimum?

5. Place your own activities on the verify-reverse plane and find where your cliff
   is.

6. Implement one architectural verifier and estimate what it moves in the second
   listing's terms.

## 18. Interview Questions

1. Which pull requests should require human review?

2. Why is gating every agent change worse than gating a fifth of them?

3. Documentation changes are your largest category. Should you review them?

4. Why is architecture hard to automate?

5. You can build one verifier. Where?

6. Your coding agent underperforms. What do you check before changing the model?

## 19. Research Questions

1. Can blast radius be inferred from static analysis reliably enough to drive
   gating?

2. Can escape cost be estimated per change type from incident data automatically?

3. What fraction of architectural constraints are expressible as machine-checkable
   assertions?

4. Does the verify-times-reverse cliff appear in other engineering disciplines?

5. How much of the observed variance in coding-agent effectiveness across
   organisations is explained by pipeline maturity?

## 20. Chapter Summary

The instinctive gating policy is by author, and it is the wrong axis. Gating by
**blast radius** achieved comparable expected cost for about a fifth of the review
time ({{eq:gate-by-blast-radius-not-author}}), and the per-minute return contains no
volume term at all — documentation is $22\%$ of changes and ranks last, schema
migrations are $3\%$ and return $33.8$ against documentation's $0.00$. What predicts
the ranking is **escape cost times what CI does not already catch.**

Broad gating is also self-undermining. At full attention it beats narrow gating; at
the $30\%$ attention agent volumes produce, it is $24.2$ against $13.9$
({{eq:narrow-gates-survive-habituation}}) — **gating everything consumes the
attention that makes gating work.**

On which activities to automate, {{ch:as-specialized}}'s two properties decide it and
they enter multiplicatively. Ranked by net benefit the activities do not shade —
**they fall off a cliff** ({{eq:automatability-is-verify-times-reverse}}), from
$+10.5$ for implementing a feature to $-2227.5$ for setting a service boundary.
Debugging is $92\%$ verifiable and $95\%$ reversible; a service boundary is $12\%$
and $9\%$.

So **architecture is the least automatable activity in software and not because it is
the hardest** ({{eq:architecture-lacks-both}}). No test says a boundary is wrong, and
by the time anyone knows, three teams have built against it.

Which is a situation rather than a fact, and situations change. A verifier is worth
$212.6$ on service boundaries and $0.1$ on bug fixing — **most exactly where none
exists** ({{eq:build-architectural-verifiers}}). Import checkers, contract tests,
schema compatibility gates, performance budgets: each moves an activity up the table
permanently, for every agent and every engineer, and each is ordinary engineering.

Finally, $65\%$ of effort sits where agents work well and $11\%$ carries decisions
costing $175$ times a bug fix. **Automate the majority of the effort; keep the
minority that carries the consequences** — and treat the pipeline as the agent's
scaffold rather than its examiner, since every chapter in this part has now converged
on that.

## 21. Further Reading

{{cite:wang2025solvedcorrectly}}'s divergence rate is the argument for the review
gate this chapter places, and worth rereading with the blast-radius framing.

{{cite:becker2025devproductivity}} for the setting-dependence that pipeline maturity
plausibly explains, which
{{sec:19-research-questions}} raises as an open question.

{{cite:chan2024mlebench}} for scaffolding mattering as much as the model, which this
chapter extends to the pipeline being the scaffold.

{{ch:ag-termination}} for the habituation curve that makes narrow gating better, and
{{ch:as-specialized}} for the two properties that produce the cliff.
