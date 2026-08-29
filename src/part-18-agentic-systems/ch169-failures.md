---
id: as-failures
number: 169
part: XVIII
tier: full
status: draft
requires: [handoff-is-a-bottleneck, decorrelate-cheaply,
           verifier-sets-the-ceiling,
           horizon-changes-the-failure, gate-on-consequence]
provides: [agent-errors-correlate, correlation-cuts-both-ways,
           redundancy-needs-independence, correlated-catastrophe,
           detection-decays-with-lag, coverage-before-freshness]
citations: [cemri2025mast, du2023debate, liu2024agentbench, zhou2024webarena,
            shinn2023reflexion, wang2023selfconsistency]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why the $r^k$ reliability
estimate is wrong in opposite directions for chains and for voting; state what
redundancy actually consumes and why nine correlated agents are worth about one;
recognise correlated catastrophe as the tail the independence assumption hides;
explain why a critic's detection rate decays with how long an error has been in the
system; and place critics by coverage rather than by freshness — which the
measurements say beats both the early and the late intuition.

## 2. Why This Matters

{{cite:cemri2025mast}} catalogued failures in real multi-agent traces and found
something uncomfortable: most were not failures of individual agents. They were
failures of specification, of inter-agent alignment, and of verification —
properties of the *system* rather than of any component.

This chapter takes two of those and measures them, and both measurements came out
differently from how the listings were designed.

The first is the independence assumption. Every multi-agent reliability estimate
starts from $r^k$ — $k$ agents, each $r$ reliable — and that assumes agents fail
independently. They share a base model, a prompt lineage, a context, and the same
upstream artefacts, so they do not ({{eq:agent-errors-correlate}}).

The surprise is the direction. Correlation makes a *chain better*: at $\rho=0.95$
the chain reached $78.2\%$ against $r^k$'s $44.3\%$, because concentrating failure
into fewer runs leaves more runs entirely clean. And it makes *redundancy worse*: a
vote of nine turned $85\%$ into $99.4\%$ when independent and into $86.2\%$ when
correlated. **Nine agents that share a cause are worth about one**
({{eq:redundancy-needs-independence}}).

So a team using $r^k$ for both — the normal case — is too pessimistic about its
pipeline and far too optimistic about its voting
({{eq:correlation-cuts-both-ways}}). Meanwhile the probability that every agent
fails together rose from $0.01\%$ to $9.53\%$, which is the tail nobody sized.

The second measurement is about critics. An error made early does not sit still;
downstream agents build consistent work around it until it stops looking like an
error at all ({{eq:detection-decays-with-lag}}). The obvious fix is to check early —
and four early critics scored $80.4\%$ against four spread across the chain at
$90.7\%$, because a critic at position one cannot catch an error at position five.
**Coverage first, then freshness** ({{eq:coverage-before-freshness}}).

And the default architecture — a long chain with one reviewer at the end — is the
worst placement available, degrading from $88.5\%$ at four agents to $27.6\%$ at
twenty-four.

## 3. Prerequisites

{{ch:as-multi-agent}}'s {{eq:handoff-is-a-bottleneck}} and its
{{eq:decorrelate-cheaply}} — that chapter found decorrelation cheap and a single
diverse agent beating a debate panel, and this chapter supplies the mechanism for
both.

{{ch:as-specialized}}'s {{eq:verifier-sets-the-ceiling}}, since the critics here are
verifiers and their detection rate is the quantity that decides everything.

{{ch:as-long-running}}'s {{eq:horizon-changes-the-failure}} for how failure modes
reorder with scale, which is the same structure this chapter finds with chain length
rather than time.

{{ch:ag-termination}}'s {{eq:gate-on-consequence}}, whose placement logic reappears
here for automated critics.

## 4. Intuitive Explanation

Start with the calculation everyone does.

You have eight agents in a pipeline, each right $85\%$ of the time. Eight
independent coin flips at $0.85$ gives $27\%$, so the system works a quarter of the
time and that is alarming. You add redundancy: run three agents at each position and
take the majority. Three at $85\%$ votes to $94\%$, eight of those gives $61\%$, and
that seems worth the tripled cost.

Both numbers are wrong, and not in the same direction.

The agents are not independent. They are the same model, prompted similarly, reading
overlapping context, and handed the same upstream artefacts. When one of them
misreads an ambiguous requirement, the others misread it the same way — not
coincidentally, but because the ambiguity is in the input and they all have the same
reading of it.

For the *chain*, that is good news. If failures cluster, then the runs where the
first agent was confused are also the runs where the third and sixth were, so the
failures pile into a smaller number of runs and more runs come through untouched.
{{sec:9-practical-example}} measures the chain doing much better than $r^k$
predicted.

For the *vote*, it is fatal. Voting works because independent errors cancel: three
observers who make unrelated mistakes rarely make the same mistake. Three observers
who share a cause make the same mistake together, and the majority is unanimous and
wrong. The vote does not aggregate three opinions; it restates one, three times.

That is also the mechanism behind {{ch:as-multi-agent}}'s uncomfortable result that
a single agent with diverse tools beat a debate panel. The panel was correlated, so
it was closer to one agent than to five, at five times the price.

Which points at the actual lever. If redundancy consumes independence, the way to
improve a vote is not more voters — it is *less shared cause*. Different models,
different prompts, different evidence. {{sec:9-practical-example}} finds reducing the
shared component worth more than tripling the panel.

Now the second half, about where errors get caught.

Agent three misreads a schema. Agent four reads agent three's output and writes code
against the wrong schema — correct code, given what it was told. Agent five writes
tests for that code, and the tests pass. Agent six writes documentation, and it
describes what the code does, accurately.

A critic at the end sees four artefacts that agree with each other. There is no
inconsistency to notice, because everything downstream was built to be consistent
with the mistake. **The error has become invisible by being propagated**, and the
critic's detection rate against it is far lower than its rate against a fresh error
in isolation.

So check early. Except that a critic at position one has nothing to say about an
error at position five, and half the errors happen in the second half of the chain.
{{sec:9-practical-example}} finds early-only placement losing to late-only placement
despite the freshness advantage, because coverage is the constraint that binds first.

Spread wins. It is the only placement that has both.

## 5. Formal Explanation

Model agent $i$'s failure as a latent Gaussian threshold with a shared component:

$$Z_i = \sqrt{\rho}\,C + \sqrt{1-\rho}\,\varepsilon_i, \qquad \text{fail}_i = \mathbb{1}[Z_i < \tau], \quad \Phi(\tau) = 1-r$$ (eq:agent-errors-correlate)

with $C$ common to all agents and $\varepsilon_i$ idiosyncratic. $\rho = 0$ is
independence; $\rho = 1$ makes every agent fail together.

For a **chain**, system success is $\Pr[\bigcap_i \{Z_i \ge \tau\}]$, the Gaussian
orthant probability, which is *increasing* in $\rho$:

$$\frac{\partial}{\partial \rho}\Pr\Big[\textstyle\bigcap_i Z_i \ge \tau\Big] > 0 \quad \Longrightarrow \quad S_{\text{chain}} \ge r^k$$ (eq:correlation-cuts-both-ways)

by Slepian's inequality. **The $r^k$ estimate is a lower bound for a chain**, and
the gap widens with $k$.

For a **vote**, success is $\Pr[\sum_i \mathbb{1}[Z_i \ge \tau] > k/2]$. As
$\rho \to 1$, all $Z_i \to C$ and the vote degenerates:

$$\lim_{\rho \to 1} S_{\text{vote}} = r \quad \text{for every } k$$ (eq:redundancy-needs-independence)

**A perfectly correlated panel of any size is one agent.** The useful reading is that
the effective panel size is roughly $k_{\text{eff}} \approx 1 + (k-1)(1-\rho)$, so at
$\rho = 0.9$ a panel of nine has an effective size near $1.8$.

The tail moves the other way from the chain:

$$\Pr\Big[\textstyle\bigcap_i \text{fail}_i\Big] \;\longrightarrow\; 1 - r \quad \text{as } \rho \to 1$$ (eq:correlated-catastrophe)

from $(1-r)^k$ at independence — five orders of magnitude at $k=8$, $r=0.85$.
**Correlation moves mass out of the middle into both tails**, and the bad tail is
every agent in the system being wrong the same way at once.

Now detection. Let an error appear at position $a$ and a critic sit at position $j$,
with lag $\ell = j - a$. Detection decays because downstream work is *consistent with*
the error:

$$c(\ell) = c_0 \cdot \gamma^{\,\ell}, \qquad \gamma < 1$$ (eq:detection-decays-with-lag)

and repair cost grows as $\ell + 1$. Total expected loss over a critic set
$J \subseteq \{0..k-1\}$ decomposes into a coverage term and a freshness term:

$$\mathbb{E}[\text{miss}] = \sum_a \Pr[a]\Big(\underbrace{\mathbb{1}[\,\nexists j \in J: j \ge a\,]}_{\text{coverage}} + \underbrace{\textstyle\prod_{j \in J, j \ge a}(1 - c_0\gamma^{\,j-a})}_{\text{freshness}}\Big)$$ (eq:coverage-before-freshness)

The coverage term is $0$ or $1$ — a step function. The freshness term is a product of
factors each strictly between $0$ and $1$. **A coverage gap is a total loss for the
errors it misses; a freshness deficit is a partial one**, which is why coverage
dominates and why early-heavy placements lose despite optimal lag.

Optimising $|J|$ positions therefore means covering the range first and minimising
lag within that constraint — which is an even spread.

## 6. Mathematical Foundation

Three extractions.

**Effective panel size is the number to report.** From
{{eq:redundancy-needs-independence}}, $k_{\text{eff}} \approx 1 + (k-1)(1-\rho)$
converts a panel size into what it is actually worth. Reporting "we use a panel of
seven" without $\rho$ is reporting a cost, not a capability, and $\rho$ is estimable
from the panel's own disagreement rate.

**The chain's correlation benefit is not usable.** $S_{\text{chain}} \ge r^k$ is
true and it is not an argument for correlated agents, because
{{eq:correlated-catastrophe}} rises at the same time. You get a better median and a
much worse tail, and for most production systems the tail is what the budget is for.
The practical use of the result is defensive: do not panic at an $r^k$ number, and do
not *trust* it either.

**Coverage is a step function and freshness is smooth.** The structural asymmetry in
{{eq:coverage-before-freshness}} is why the ordering in
{{sec:9-practical-example}} came out as it did, and it generalises past critics: any
detection mechanism placed over a range should saturate the range before optimising
position within it.

## 7. Internal Mechanics

### 7.1 Where correlation comes from, concretely

```mermaid {#fig:shared-cause caption="The shared causes that make agent errors correlate. Every arrow into more than one agent is a term in rho."}
flowchart TD
    M[same base model] --> A1[agent 1]
    M --> A2[agent 2]
    M --> A3[agent 3]
    P[same prompt lineage] --> A1
    P --> A2
    P --> A3
    U[same upstream artefact] --> A2
    U --> A3
    C[same ambiguous requirement] --> A1
    C --> A2
    C --> A3
```

Four sources, in roughly decreasing order of how much they contribute and increasing
order of how easy they are to remove.

**The same ambiguous input** is the largest and the most fixable: if a requirement
admits two readings, every agent takes the same one. Disambiguating the input removes
more correlation than any diversity in the agents.

**The same upstream artefact** — agents downstream of a common producer inherit its
errors by construction. This is not really correlation so much as a shared premise,
and it is what the second listing is about.

**The same prompt lineage** — panels built by copying one prompt and editing the role
line share nearly all of their inductive bias.

**The same base model** is the hardest to remove and, notably, not the largest. Teams
reach for multi-model panels first because it is the visible axis; the measurements
say input disambiguation is cheaper and worth more.

### 7.2 Estimating $\rho$ from the panel you already have

You do not need a theory of correlation to measure it. Run the panel on cases where
ground truth is known and record the joint failure pattern. Then:

$$\hat\rho \;\text{from}\; \Pr[\text{both fail}] \;\text{versus}\; (1-r)^2$$

If both agents fail together far more often than the square of their individual
rates, they are correlated, and the ratio gives you $k_{\text{eff}}$ directly.

This is a half-day of work and it converts an unfalsifiable architecture claim into
a number. Very few systems have it.

### 7.3 Why propagation makes errors invisible

{{eq:detection-decays-with-lag}}'s $\gamma$ is not a modelling convenience; it names
a specific phenomenon worth understanding.

An error caught immediately is an isolated inconsistency: this output does not match
that input. An error caught five agents later is embedded in a body of work that is
*internally consistent* — the code matches the wrong schema, the tests match the
code, the docs match the tests. A critic looking for inconsistency finds none.

Worse, the consistency reads as *evidence*. Four artefacts agreeing looks like
corroboration, and it is the same mistake as trusting a correlated panel — in fact
it is exactly that mistake, appearing along the chain rather than across it. The
downstream agents are not independent witnesses to the premise; they are
consequences of it.

Which suggests the detection that works at high lag: check against the *original
input*, not against internal consistency. A critic asking "does this match the
requirement" degrades far less with lag than one asking "does this hang together".

### 7.4 Critic placement in practice

```mermaid {#fig:critic-placement caption="Four critics, three placements. Spread wins because it is the only one with full coverage, which the measurements find binds before freshness."}
flowchart LR
    subgraph early["early: no coverage of 4-7"]
        E0[a0] --> E1[a1] --> E2[a2] --> E3[a3] --> E4[a4-a7]
    end
    subgraph spread["spread: full coverage, low lag"]
        S0[a0-a1] --> S1[a2-a3] --> S2[a4-a5] --> S3[a6-a7]
    end
    subgraph late["late: full coverage, high lag"]
        L0[a0-a3] --> L1[a4] --> L2[a5] --> L3[a6-a7]
    end
```

The operational rule that falls out: with a budget of $m$ critics over $k$ agents,
place them at positions $\lceil k/m \rceil$ apart starting near the front. That
covers the range and minimises mean lag subject to coverage, which is what
{{eq:coverage-before-freshness}} asks for.

The common alternative — one reviewer at the end — arises for a social reason rather
than a technical one. The reviewer is added after someone notices the output is
wrong, and the natural place to add it is where the wrongness was noticed.

### 7.5 The failure categories that are not reliability at all

{{cite:cemri2025mast}}'s taxonomy includes modes neither listing models, and they
should not be forgotten because they are harder to quantify.

**Specification failures** — the system was asked for the wrong thing, and every
agent did its job. No amount of critic placement helps.

**Inter-agent misalignment** — agents with locally-reasonable and jointly-incoherent
objectives, which {{ch:as-roles}} met as the critic-generator dynamic.

**Premature termination** — an agent declares completion on partial work, and
{{ch:ag-termination}}'s stopping problem becomes a coordination problem when the
declaration is trusted by others.

The two this chapter measures are the ones with clean structure. The others are
larger in {{cite:cemri2025mast}}'s trace counts, which is worth holding onto: the
quantifiable part is not the biggest part.

### 7.6 Chain length is the variable behind everything

Every result here degrades with $k$: correlation's chain benefit grows, the
catastrophe tail grows, critic placement matters more, and lag grows.

{{ch:as-multi-agent}} already found the handoff itself to be the bottleneck, and
this chapter adds three more mechanisms that scale with the same variable.
{{sec:9-practical-example}}'s last table is the summary: at four agents the
placements are within nine points of each other and none of this matters; at
twenty-four the gap is twenty-six points.

**The strongest single intervention available in this part is a shorter chain**, and
it is available before any of the others.

### 7.7 This is why decorrelation was cheap

{{ch:as-multi-agent}} found that decorrelating agents cost very little and bought a
lot ({{eq:decorrelate-cheaply}}), which at the time looked like a convenient
empirical fact. {{eq:redundancy-needs-independence}} explains it.

The value of a panel is governed by $k_{\text{eff}} \approx 1 + (k-1)(1-\rho)$, and
that expression is *linear in $1-\rho$* while being flat in $k$ once $\rho$ is
high. So a small reduction in shared cause moves the effective panel size by more
than a large increase in the actual panel size does — which is precisely the
asymmetry {{sec:9-practical-example}} measures, where dropping $\rho$ from $0.9$ to
$0.1$ was worth more than going from three voters to nine at any correlation.

The reason it is *cheap* is separate and equally structural. Adding a voter costs a
full inference. Reducing $\rho$ often costs nothing at inference time at all: giving
two panel members different evidence, or different system prompts, or asking one to
argue the opposite case, are edits to the setup rather than additional calls.

**Decorrelation is the rare intervention that improves the numerator without
touching the denominator**, which is why it dominates on any cost-adjusted
comparison — and why {{ch:as-multi-agent}}'s equal-cost methodology found it when
naive comparisons of panel sizes would not have.

It also sets a limit worth knowing. $k_{\text{eff}}$ cannot exceed $k$, so
decorrelation buys at most what an independent panel of the same size would give.
Once $\rho$ is near zero, the only remaining lever is more voters, and their
returns are the familiar diminishing ones.

## 8. Implementation

Two listings. The first measures what correlated failures do to chains and to votes.
The second measures where critics should sit.

```python {tier=A name=agent-errors-correlate}
"""The independence assumption, which is what makes multi-agent estimates wrong.

Reliability for a multi-agent system is nearly always estimated the same way: each
agent is r reliable, there are k of them, so the system is r^k. That calculation
assumes the agents fail independently.

They do not. Agents in a real system share a base model, share a prompt lineage,
share a context window, and are handed the same upstream artefacts. A shared cause
makes their errors correlated (eq:agent-errors-correlate).

The listing was written expecting correlation to make everything worse. It does not:
it makes a CHAIN better and REDUNDANCY worse, because those two structures use the
independence assumption in opposite directions (eq:correlation-cuts-both-ways). The
tail behaviour is the part that should worry you.
"""
import numpy as np

rng = np.random.default_rng(3833)

M = 60000
K = 5
R = 0.85


def correlated_failures(m, k, r, rho):
    """Errors with a shared component. Each agent fails if its latent score falls
    below a threshold; the scores share a common factor of weight rho, so rho=0 is
    independence and rho=1 makes every agent fail together."""
    common = rng.standard_normal((m, 1))
    idio = rng.standard_normal((m, k))
    z = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
    from math import erf, sqrt
    # threshold such that P(fail) = 1 - r
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 0.5 * (1 + erf(mid / sqrt(2)))
        if p < 1 - r:
            lo = mid
        else:
            hi = mid
    return z < (lo + hi) / 2


def pipeline(rho, m=M, k=K, r=R):
    """A chain: every agent must be right."""
    f = correlated_failures(m, k, r, rho)
    return float((~f.any(1)).mean())


def vote(rho, m=M, k=K, r=R):
    """Redundancy: k agents answer, majority wins."""
    f = correlated_failures(m, k, r, rho)
    return float(((~f).sum(1) > k / 2).mean())


print(f"{M:,} runs, {K} agents at {R:.0%} reliability each.")
print(f"The independence estimate for a chain is {R ** K:.1%}.")
print()
print(f"{'correlation':>13}{'chain':>9}{'vs r^k':>10}{'vote of 5':>12}"
      f"{'all fail':>11}")
print("-" * 55)
tab = {}
for rho in (0.0, 0.2, 0.4, 0.7, 0.95):
    c = pipeline(rho)
    v = vote(rho)
    f = correlated_failures(M, K, R, rho)
    allf = float(f.all(1).mean())
    tab[rho] = (c, v, allf)
    print(f"{rho:>13.2f}{c:>9.1%}{c - R ** K:>+10.1%}{v:>12.1%}{allf:>11.2%}")

print()
print()
print("The same, as agent count grows. Chain reliability under independence")
print("against chain reliability at a realistic shared-cause correlation.")
print()
print(f"{'agents':>8}{'independent':>14}{'rho=0.4':>10}{'rho=0.7':>10}"
      f"{'optimism':>11}")
print("-" * 53)
gr = {}
for k in (2, 3, 5, 8, 12):
    a = pipeline(0.0, k=k)
    b = pipeline(0.4, k=k)
    c = pipeline(0.7, k=k)
    gr[k] = (a, b, c)
    print(f"{k:>8}{a:>14.1%}{b:>10.1%}{c:>10.1%}{a - c:>+11.1%}")

print()
print()
print("Voting is the standard mitigation, and it is the one correlation removes.")
print("Marginal value of redundancy, over a single agent at the same reliability:")
print()
print(f"{'correlation':>13}{'1 agent':>10}{'vote of 3':>12}{'vote of 5':>12}"
      f"{'vote of 9':>12}")
print("-" * 59)
vt = {}
for rho in (0.0, 0.3, 0.6, 0.9):
    row = (R, vote(rho, k=3), vote(rho, k=5), vote(rho, k=9))
    vt[rho] = row
    print(f"{rho:>13.2f}{row[0]:>10.1%}{row[1]:>12.1%}{row[2]:>12.1%}"
          f"{row[3]:>12.1%}")

print()
print()
print("And what diversity buys: agents that share less. Same vote of 5, with the")
print("shared component reduced by using genuinely different models or prompts.")
print()
print(f"{'shared cause':>14}{'vote of 5':>12}{'gain over 1':>14}"
      f"{'gain over rho=0.9':>20}")
print("-" * 60)
dv = {}
for rho in (0.9, 0.7, 0.5, 0.3, 0.1):
    v = vote(rho)
    dv[rho] = v
    print(f"{rho:>14.2f}{v:>12.1%}{v - R:>+14.1%}{v - vt[0.9][2]:>+20.1%}")

print(f"""
The first table's second column is not the sign the listing expected.

At correlation {0.95:.2f} the chain reaches {tab[0.95][0]:.1%} against the
independence estimate's {R ** K:.1%} -- **the r^k calculation is PESSIMISTIC by
{tab[0.95][0] - R ** K:.1f} points**, not optimistic. Positive correlation
concentrates failure into fewer runs, so more runs come through entirely clean.

That is correct probability rather than a modelling artefact, and it is worth
absorbing before the rest: for a structure where everything must go right,
correlated agents are better than independent ones.

The fourth column is where the same correlation is bad news. The probability that
ALL {K} agents fail together goes from {tab[0.0][2]:.2%} at independence to
{tab[0.95][2]:.2%} -- a factor of about
{tab[0.95][2] / max(tab[0.0][2], 1e-9):.0f}. **Correlation moves probability mass
out of the middle and into both tails**, and one of those tails is every agent in
your system being wrong the same way at the same time.

The second table shows the chain effect growing with agent count: at {12} agents,
independence predicts {gr[12][0]:.1%} and correlation {0.7:.2f} gives
{gr[12][2]:.1%}. A team that sized its pipeline using r^k built something more
reliable than it thought, for a reason it would not have been able to state.

The third table is the one that should change a design decision. Redundancy is the
standard mitigation for unreliable agents -- run several, take the majority -- and
it is the mitigation correlation removes.

At independence, a vote of {9} turns {R:.0%} into {vt[0.0][3]:.1%}, worth
{vt[0.0][3] - R:+.1%}. At correlation {0.9:.2f} the same vote of {9} gives
{vt[0.9][3]:.1%}, worth {vt[0.9][3] - R:+.1%}.

**Nine agents that share a cause are worth about one agent**
(eq:redundancy-needs-independence). The votes are not independent evidence; they are
one opinion, restated nine times, at nine times the cost.

So the independence assumption is wrong in OPPOSITE DIRECTIONS for the two things it
is used for (eq:correlation-cuts-both-ways). It understates how well a chain
performs and it drastically overstates what redundancy buys. A team using r^k for
both -- which is the normal case -- is being too pessimistic about its pipeline and
far too optimistic about its voting.

The last table says what to do about the second problem, and it is not "add more
agents". Reducing the shared cause from {0.9:.2f} to {0.1:.2f} takes a vote of {K}
from {dv[0.9]:.1%} to {dv[0.1]:.1%}, worth {dv[0.1] - dv[0.9]:+.1%} -- more than
going from three voters to nine buys at any correlation in the table.

**Diversity is the input redundancy actually consumes**, and it comes from
genuinely different models, genuinely different prompts, and genuinely different
evidence rather than from more copies. This is the mechanism behind
ch:as-multi-agent's finding that one diverse agent beat a debate panel: the panel
members were correlated, so the panel was closer to one agent than to five.""")
```

The second listing asks where the critics go.

```python {tier=A name=coverage-before-freshness}
"""Where a multi-agent failure gets caught, which decides what it costs.

An error made by agent 3 of 8 does not sit still. Agent 4 reads agent 3's output
and builds on it; agent 5 builds on agent 4. By the time anything checks, the error
is not a wrong sentence -- it is a premise that five agents have written consistent
work around.

Two consequences, and the second is the one that matters:

  cost        repairing an error means redoing everything built on it, so cost
              grows with detection lag
  detectability an error surrounded by consistent downstream work looks RIGHT.
              Detection probability DECAYS with lag (eq:detection-decays-with-lag)

Most systems put their critic at the end, which maximises lag on both counts. This
listing measures placement at a fixed critic budget.
"""
import numpy as np

rng = np.random.default_rng(3877)

M = 60000
K = 8                   # agents in the chain
P_ERR = 0.06            # chance a given agent introduces an error
C0 = 0.88               # a critic's detection rate at zero lag
DECAY = 0.72            # detection multiplier per step of lag
FIX = 0.85              # a caught error is repaired this often


def run(critic_positions, m=M, k=K, p_err=P_ERR, c0=C0, decay=DECAY):
    """Errors appear at agent positions; critics sit after chosen positions. A
    critic catches an error with probability c0 * decay**lag, and repairing costs
    the work done since the error was made."""
    err_at = rng.random((m, k)) < p_err
    first = np.where(err_at.any(1), err_at.argmax(1), -1)
    alive_err = first.copy()          # -1 means no outstanding error
    rework = np.zeros(m, dtype=np.float64)
    caught = np.zeros(m, dtype=bool)
    for j in sorted(critic_positions):
        has = alive_err >= 0
        lag = np.where(has, j - alive_err, 0)
        visible = has & (lag >= 0)
        p = c0 * (decay ** np.clip(lag, 0, None))
        hit = visible & (rng.random(m) < p) & (rng.random(m) < FIX)
        rework[hit] += lag[hit] + 1
        caught |= hit
        alive_err[hit] = -1
    shipped_bad = alive_err >= 0
    n_crit = len(critic_positions)
    return (float((~shipped_bad).mean()), float(rework.mean()),
            float(k + rework.mean() + n_crit), n_crit)


LAST = [K - 1]
MID = [K // 2, K - 1]
EVERY2 = list(range(1, K, 2))
EVERY1 = list(range(K))
EARLY = [0, 1, 2, 3]

print(f"{M:,} runs through {K} agents; each introduces an error with")
print(f"probability {P_ERR:.0%}. A critic detects at {C0:.0%} when the error is")
print(f"fresh, decaying {DECAY:.0%} per step of lag as downstream work makes it")
print("look consistent.")
print()
print(f"{'critic placement':>22}{'critics':>9}{'clean output':>14}"
      f"{'rework':>9}{'total cost':>12}")
print("-" * 66)
plans = [("one at the end", LAST), ("middle + end", MID),
         ("every 2nd agent", EVERY2), ("after every agent", EVERY1),
         ("four, all early", EARLY)]
tab = {}
for name, pos in plans:
    r = run(pos)
    tab[name] = r
    print(f"{name:>22}{r[3]:>9}{r[0]:>14.1%}{r[1]:>9.2f}{r[2]:>12.1f}")

print()
print()
print("The controlled comparison: FOUR critics, placed differently.")
print()
print(f"{'four critics at':>22}{'clean output':>14}{'rework':>9}"
      f"{'mean lag when caught':>23}")
print("-" * 68)
four = {}
FOURS = [("0,1,2,3 (early)", [0, 1, 2, 3]),
         ("1,3,5,7 (spread)", [1, 3, 5, 7]),
         ("4,5,6,7 (late)", [4, 5, 6, 7])]
for name, pos in FOURS:
    r = run(pos)
    four[name] = r
    lag = r[1] / max(r[0] - (1 - P_ERR) ** K, 1e-9)
    print(f"{name:>22}{r[0]:>14.1%}{r[1]:>9.2f}{lag:>23.2f}")

print()
print()
print("How much of this is the decay? Same placements with detection that does")
print("not degrade with lag -- a critic as good on stale errors as fresh ones.")
print()
print(f"{'four critics at':>22}{'with decay':>13}{'no decay':>11}{'loss':>10}")
print("-" * 56)
nd = {}
for name, pos in FOURS:
    a = run(pos)[0]
    b = run(pos, decay=1.0)[0]
    nd[name] = (a, b)
    print(f"{name:>22}{a:>13.1%}{b:>11.1%}{a - b:>+10.1%}")

print()
print()
print("And against chain length, since a longer chain gives an early error more")
print("room to become invisible.")
print()
print(f"{'agents':>8}{'end only':>11}{'spread':>10}{'early':>10}{'best':>10}")
print("-" * 49)
cl = {}
for k in (4, 8, 16, 24):
    end = run([k - 1], k=k)[0]
    spread = run(list(range(1, k, max(1, k // 4))), k=k)[0]
    early = run(list(range(min(4, k))), k=k)[0]
    names = ["end only", "spread", "early"]
    row = (end, spread, early)
    cl[k] = (row, names[int(np.argmax(row))])
    print(f"{k:>8}{end:>11.1%}{spread:>10.1%}{early:>10.1%}{cl[k][1]:>10}")

print(f"""
The first table shows the obvious thing and hides the interesting one. More critics
catch more: one at the end gives {tab['one at the end'][0]:.1%} and one after every
agent gives {tab['after every agent'][0]:.1%}, at a total cost of
{tab['one at the end'][2]:.1f} against {tab['after every agent'][2]:.1f}.

But notice the last row. FOUR critics, all placed early, give
{tab['four, all early'][0]:.1%} -- worse than four critics placed every second agent
at {tab['every 2nd agent'][0]:.1%}, and barely better than TWO critics at the middle
and end.

The controlled table isolates that, and the ordering is not the one the chapter's
premise predicts. Four critics spread across the chain give {four['1,3,5,7 (spread)'][0]:.1%};
four placed late give {four['4,5,6,7 (late)'][0]:.1%}; four placed early give
{four['0,1,2,3 (early)'][0]:.1%}.

**Early placement is the worst of the three**, despite catching errors at the
lowest lag -- mean lag {four['0,1,2,3 (early)'][1] / max(four['0,1,2,3 (early)'][0] - (1 - P_ERR) ** K, 1e-9):.2f}
against {four['4,5,6,7 (late)'][1] / max(four['4,5,6,7 (late)'][0] - (1 - P_ERR) ** K, 1e-9):.2f} -- for a reason
that is obvious once stated and easy to miss when reasoning about freshness alone:
a critic at position 1 cannot catch an error made at position 5. Early critics have
no coverage of the second half of the chain.

So the rule is **coverage first, then freshness** (eq:coverage-before-freshness).
Spread wins because it is the only placement with both.

Freshness is still real and the third table prices it. Turning off the decay -- a
critic as good on a five-step-old error as on a fresh one -- is worth
{nd['4,5,6,7 (late)'][1] - nd['4,5,6,7 (late)'][0]:+.1%} to late placement and
{nd['0,1,2,3 (early)'][1] - nd['0,1,2,3 (early)'][0]:+.1%} to early placement.

**Almost all of the late placement's disadvantage is the decay**
(eq:detection-decays-with-lag), which is the mechanism worth naming: an error
surrounded by consistent downstream work does not look like an error. Five agents
have each written something coherent given the mistaken premise, and a critic
reading the result sees five agreements. It is ch:as-multi-agent's correlation
appearing inside a single run rather than across parallel ones.

The last table says the effect grows with the chain, which is the practical warning.
At {4} agents the placements are within {max(cl[4][0]) - min(cl[4][0]):.1%} of each
other and none of this matters. At {24} agents, one critic at the end gives
{cl[24][0][0]:.1%} and spread critics give {cl[24][0][1]:.1%}.

**The default architecture -- a long chain of agents with a reviewer at the end --
is the worst available placement, and it gets worse the longer the chain.** It is
also, for entirely non-technical reasons, the one almost everyone builds: the
reviewer is added last because that is when someone notices the output is wrong.

Three rules.

**Spread critics across the chain rather than concentrating them.** Coverage is the
binding constraint and freshness is the tiebreak.

**Shorten the chain.** Every result here degrades with length, and ch:as-multi-agent
already found handoff count to be an exponent.

**Do not measure a critic's detection rate on fresh errors** -- the number that
matters is its rate at the lag it will actually see, which is much lower.""")
```

## 9. Practical Example

The first listing runs five agents at $85\%$ each, sweeping the shared-cause
correlation. The independence estimate for a chain is $44.3\%$.

```
  correlation    chain    vs r^k   vote of 5    all fail
-------------------------------------------------------
         0.00    44.2%     -0.2%       97.3%      0.01%
         0.40    57.1%    +12.7%       91.5%      0.75%
         0.95    78.2%    +33.8%       85.5%      9.53%
```

**The $r^k$ calculation is pessimistic for a chain**, by $33.8$ points at high
correlation ({{eq:correlation-cuts-both-ways}}) — concentrating failure leaves more
runs entirely clean. And the last column is the price: all five agents failing
together goes from $0.01\%$ to $9.53\%$, a factor of roughly a thousand
({{eq:correlated-catastrophe}}).

Growing the pipeline:

```
  agents   independent   rho=0.4   rho=0.7   optimism
-----------------------------------------------------
       5         44.3%     56.9%     66.9%     -22.6%
      12         14.2%     38.6%     56.8%     -42.6%
```

Redundancy is the other direction entirely:

```
  correlation   1 agent   vote of 3   vote of 5   vote of 9
-----------------------------------------------------------
         0.00     85.0%       94.1%       97.3%       99.4%
         0.60     85.0%       87.9%       89.0%       89.8%
         0.90     85.0%       85.2%       85.9%       86.2%
```

A vote of nine is worth $+14.4$ points independent and $+1.2$ correlated.
**Nine agents that share a cause are worth about one**
({{eq:redundancy-needs-independence}}) — the votes are one opinion restated nine
times, at nine times the cost. This is the mechanism behind
{{ch:as-multi-agent}}'s single diverse agent beating a debate panel, and behind
{{cite:du2023debate}}'s gains being smaller than the architecture suggests.

And what to do instead of adding voters:

```
  shared cause   vote of 5   gain over 1   gain over rho=0.9
------------------------------------------------------------
          0.90       86.1%         +1.1%               +0.2%
          0.50       90.1%         +5.1%               +4.2%
          0.10       95.9%        +10.9%              +10.0%
```

**Diversity is the input redundancy consumes.** Reducing shared cause is worth more
than tripling the panel at any correlation in the table.

The second listing puts errors into an eight-agent chain and varies where critics
sit:

```
      critic placement  critics  clean output   rework  total cost
------------------------------------------------------------------
        one at the end        1         72.0%     0.35         9.4
       every 2nd agent        4         90.6%     0.55        12.5
     after every agent        8         97.3%     0.47        16.5
       four, all early        4         80.0%     0.23        12.2
```

Four critics placed early lose to four placed every second agent. Controlled:

```
       four critics at  clean output   rework   mean lag when caught
--------------------------------------------------------------------
       0,1,2,3 (early)         80.4%     0.23                   1.17
      1,3,5,7 (spread)         90.7%     0.55                   1.84
        4,5,6,7 (late)         89.8%     0.73                   2.53
```

**Early is the worst of the three despite the lowest lag** — a critic at position one
cannot catch an error at position five. **Coverage first, then freshness**
({{eq:coverage-before-freshness}}).

Freshness is still real:

```
       four critics at   with decay   no decay      loss
--------------------------------------------------------
       0,1,2,3 (early)        80.4%      81.3%     -1.0%
        4,5,6,7 (late)        89.7%      98.6%     -8.9%
```

Almost all of late placement's disadvantage is the decay
({{eq:detection-decays-with-lag}}) — an error surrounded by consistent downstream
work stops looking like an error.

And the warning:

```
  agents   end only    spread     early      best
-------------------------------------------------
       4      88.5%     96.6%     97.4%     early
       8      71.7%     90.5%     80.5%    spread
      24      27.6%     53.5%     41.5%    spread
```

**The default architecture — a long chain with a reviewer at the end — is the worst
placement available**, and the gap grows from nine points at four agents to
twenty-six at twenty-four.

## 10. Production Considerations

Report effective panel size, not panel size. Estimate $\rho$ from joint failures on
known cases; it is half a day of work and it converts an architecture claim into a
number.

Reduce shared cause before adding voters — and start with the ambiguous input, which
is the largest contributor and the cheapest to fix.

Size the correlated-catastrophe tail explicitly. It is the failure where every agent
is wrong the same way, and $(1-r)^k$ understates it by orders of magnitude.

Spread critics evenly across the chain, at $\lceil k/m \rceil$ intervals. Do not
concentrate them at either end.

Have critics check against the original requirement rather than against internal
consistency, since consistency is exactly what propagation manufactures.

Measure critic detection at realistic lag, not on fresh injected errors. The
difference is most of the number.

And shorten the chain. It is the one intervention that improves every quantity in
this chapter at once.

## 11. Common Mistakes

**Using $r^k$ for both chain reliability and vote reliability.** It is wrong in
opposite directions.

**Adding panel members to fix a weak panel.** Correlated votes do not aggregate;
$k_{\text{eff}}$ barely moves.

**Reaching for multi-model panels first.** Input disambiguation removes more
correlation and costs less.

**Placing all critics early.** Coverage binds before freshness.

**One reviewer at the end.** The default, and the measured worst.

**Reading downstream agreement as corroboration.** Downstream agents are consequences
of a premise, not witnesses to it.

**Measuring critic detection on fresh errors.** It reports a rate the critic will
never see in production.

## 12. Failure Modes

*Correlated catastrophe.* Every agent wrong the same way at once — rare under
independence, common under shared cause, and sized by nobody.

*Unanimous wrong votes.* The panel agrees, confidently, because it is one opinion.

*Invisible propagated error.* Consistent downstream work hides the premise that was
wrong, and the end reviewer sees corroboration.

*Coverage gap.* Errors in a region no critic follows — a total loss rather than a
partial one.

*Specification failure.* Every agent correct, the system wrong, and no critic
placement helps ({{cite:cemri2025mast}}).

*Premature termination trusted downstream.* {{ch:ag-termination}}'s stopping problem
becoming a coordination failure.

## 13. Alternatives

**A shorter chain.** {{sec:7-internal-mechanics}}'s conclusion and the strongest
option available.

**Genuine diversity.** Different models, different prompts, different evidence — the
input {{eq:redundancy-needs-independence}} says redundancy actually consumes.

**Deterministic checks instead of critic agents.** They do not correlate with the
generator at all, which makes them worth more than their raw detection rate suggests
— {{ch:as-specialized}}'s argument arriving as a correlation argument.

**Structured intermediate artefacts.** Typed hand-offs make propagated errors
detectable by construction rather than by judgement, cutting $\gamma$.

**One agent.** {{ch:as-single-agent}}'s option, which by now has accumulated a
substantial amount of evidence in this part.

## 14. Evaluation

Measure $\rho$ and report $k_{\text{eff}}$ alongside every panel result.

Measure critic detection at the lag distribution it will actually face, by injecting
errors upstream rather than at the critic.

Report the joint-failure rate, not just the mean failure rate — the tail is the part
the independence assumption hides.

Evaluate critic placements at your real chain length. Under about six agents the
differences are within noise and generalising from that is how the end-reviewer
default survives.

And separate specification failures out of your numbers before concluding anything
about reliability, since no mechanism here addresses them.

## 15. Advanced Concepts

**Online $\rho$ estimation.** Panels could measure their own correlation continuously
from disagreement rates and report an effective size that tracks drift.
{{maturity:EMERGING}}.

**Diversity as an explicit objective.** Selecting panel members to minimise measured
error correlation rather than to maximise individual accuracy — the portfolio
construction analogue, largely unexplored for agent ensembles.

**Lag-robust critics.** Checking against original requirements rather than internal
consistency should reduce $\gamma$; how much is an open measurement.

**Automatic critic placement.** {{eq:coverage-before-freshness}} is optimisable given
an error-position distribution, which is estimable from traces.
{{maturity:EXPERIMENTAL}}.

**Whether {{cite:cemri2025mast}}'s taxonomy is complete.** The quantifiable modes are
the minority of its trace counts, and the majority are specification and alignment
failures with no accepted formalisation. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:as-multi-agent}}'s single diverse agent beating a debate panel gets its
mechanism here: the panel was correlated, so it was closer to one agent than to five.

{{ch:as-roles}}'s critic role acquires a placement theory, and its
generator-critic dynamic is one of {{cite:cemri2025mast}}'s misalignment modes.

{{ch:as-specialized}}'s verifier ceiling applies to the critics here, with a
correlation caveat added: a critic sharing the generator's model shares its blind
spots, which is why deterministic checks outperform their detection rate.

{{ch:as-long-running}}'s reordering of failure modes with scale is the same shape
found here with chain length instead of time.

{{ch:as-single-agent}}'s case looks stronger at the end of this part than it did at
the start of it, which is the honest summary of {{part:18}}.

Ahead: {{part:19}} turns to MCP and tool ecosystems, where the correlation and
propagation problems reappear across organisational boundaries rather than within
one system.

## 17. Exercises

1. Derive $k_{\text{eff}}$ from {{eq:redundancy-needs-independence}} and check it
   against the listing's vote-of-nine row.

2. Estimate $\rho$ for a panel you have access to, from joint failures on known
   cases.

3. Add a deterministic check to the second listing, uncorrelated with the agents, and
   compare it against a critic agent of equal raw detection rate.

4. Optimise critic placement numerically for a non-uniform error-position
   distribution. Does even spacing survive?

5. Model a critic that checks against the original requirement — $\gamma$ near one —
   and re-run the placement comparison.

6. Combine both listings: correlated agents in a chain with spread critics. Are the
   effects independent?

## 18. Interview Questions

1. Your eight-agent pipeline computes to $27\%$ by $r^k$ and measures at $60\%$. What
   is going on?

2. You added six more voters to a panel and accuracy did not move. Why?

3. What does redundancy consume, and how would you buy more of it?

4. Where do you put four critics in a twelve-agent chain, and why not at the end?

5. Why does a critic's detection rate fall as the error gets older?

6. What is the probability that every agent in your system is wrong at once, and how
   would you find out?

## 19. Research Questions

1. Can error correlation be estimated online reliably enough to report effective
   panel size in production?

2. What is the right objective for selecting a diverse panel, and does portfolio
   theory transfer?

3. How much does requirement-anchored criticism reduce $\gamma$ compared with
   consistency-based criticism?

4. Can error-position distributions be learned well enough to place critics
   automatically?

5. Is {{cite:cemri2025mast}}'s taxonomy complete, and can its non-quantifiable
   categories be given measurable structure?

## 20. Chapter Summary

Multi-agent reliability is estimated with $r^k$, which assumes independence. Agents
share a model, a prompt lineage, a context and their upstream artefacts, so their
errors correlate ({{eq:agent-errors-correlate}}) — and the assumption fails in
opposite directions for the two things it is used for
({{eq:correlation-cuts-both-ways}}).

For a **chain**, correlation *helps*: $78.2\%$ against $r^k$'s $44.3\%$ at
$\rho = 0.95$, because concentrating failure leaves more runs clean. For
**redundancy** it is fatal: a vote of nine is worth $+14.4$ points independent and
$+1.2$ correlated. **Nine agents that share a cause are worth about one**
({{eq:redundancy-needs-independence}}), which is the mechanism behind
{{ch:as-multi-agent}}'s diverse single agent beating a debate panel.

Meanwhile the probability of every agent failing together rose from $0.01\%$ to
$9.53\%$ ({{eq:correlated-catastrophe}}) — the tail the independence assumption
hides. And the lever is not more voters: reducing shared cause was worth more than
tripling the panel.

On critics, an error propagates into work that is built to be consistent with it, so
detection decays with lag ({{eq:detection-decays-with-lag}}) and downstream agreement
reads as corroboration when it is merely consequence. The intuitive fix — check early
— lost: four early critics scored $80.4\%$ against four spread at $90.7\%$, because a
critic at position one cannot catch an error at position five. **Coverage first, then
freshness** ({{eq:coverage-before-freshness}}), since a coverage gap is a total loss
and a freshness deficit is a partial one.

The default architecture — a long chain with one reviewer at the end — is the worst
placement measured, falling from $88.5\%$ at four agents to $27.6\%$ at twenty-four.
Which makes the chapter's strongest recommendation the same as this part's:
**shorten the chain.**

## 21. Further Reading

{{cite:cemri2025mast}} is the essential reading for this chapter — a taxonomy from
real traces, whose largest categories are the ones neither listing here can quantify.

{{cite:du2023debate}} and {{cite:wang2023selfconsistency}} for aggregation methods
whose gains {{eq:redundancy-needs-independence}} predicts will be smaller than their
architecture suggests, and worth re-reading with $k_{\text{eff}}$ in mind.

{{cite:liu2024agentbench}} and {{cite:zhou2024webarena}} for the settings where chain
length is long enough for placement to matter, and
{{cite:shinn2023reflexion}} for a self-critique loop whose critic is maximally
correlated with its generator.

{{ch:as-multi-agent}} and {{ch:as-single-agent}} for the architectural choice this
chapter's measurements bear on most directly.
