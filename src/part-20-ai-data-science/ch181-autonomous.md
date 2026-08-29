---
id: aids-autonomous
number: 181
part: XX
tier: full
status: draft
requires: [search-must-be-scored-off-search, agent-errors-correlate,
           review-does-not-scale, redundancy-needs-independence]
provides: [self-judging-measures-correlation, accuracy-is-not-independence,
           correlated-panels-are-one-judge, generation-scales-verification-does-not,
           volume-overflows-the-filter, attention-is-a-commons]
citations: [lu2024aiscientist, testini2025dsautomation, chan2024mlebench,
            cemri2025mast, du2023debate]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a self-judged pipeline's
acceptance rate measures correlation rather than quality; state why a reviewer's
reported accuracy does not license conclusions about its own generator's output;
show why improving a correlated judge and why adding more of them both fail; price
the verification half of an automated research pipeline separately from the
generation half; and explain why reviewer attention behaves as a commons that
volume depletes.

## 2. Why This Matters

{{cite:lu2024aiscientist}} built the most complete public attempt at automating a
whole research loop — idea generation, code, experiments, figures, paper writing,
and simulated review — at **under \$15 per paper**, demonstrated across diffusion
modelling, transformer language modelling and learning dynamics.

Its headline claim carries a clause that this book has spent nineteen parts
building the apparatus to price: the system produces papers exceeding a top
conference's acceptance threshold **as judged by the authors' own automated
reviewer**, which they report achieves near-human agreement on paper scores.

{{ch:as-failures}} established that agents sharing a base model, a prompt lineage
and a context have correlated errors. A generator and a judge built from the same
model share all three. So the relevant question is not whether the judge is
accurate — it is whether its errors are independent of the generator's.

{{sec:9-practical-example}} measures what happens when they are not. As shared bias
rises, the pipeline's acceptance rate climbs from $32.7\%$ to $84.2\%$ while the
share of output that is actually sound sits at $14.5\%$ throughout.
**A self-judged pipeline's acceptance rate measures the correlation between its
generator and its judge** ({{eq:self-judging-measures-correlation}}).

Two obvious responses both fail. Driving the judge's specificity from $70\%$ to
$99\%$ moves precision from $16.6\%$ to $39.6\%$ under high shared bias, against
$37.5\%$ to $94.3\%$ when independent ({{eq:accuracy-is-not-independence}}) — the
failure is not an accuracy failure. And seven correlated judges reach $20.4\%$
against one judge's $17.6\%$ ({{eq:correlated-panels-are-one-judge}}), which is
{{ch:as-failures}}' aggregation result unchanged.

The second half takes the volume argument seriously —
{{cite:testini2025dsautomation}}'s redesign gap, which this book's earlier listings
cannot see — and finds where it breaks. Generation costs \$15 and independent
verification costs about **\$2{,}000 per usable result**
({{eq:generation-scales-verification-does-not}}). The famous figure prices the half
that got cheap.

## 3. Prerequisites

{{ch:as-failures}}'s {{eq:agent-errors-correlate}} and
{{eq:redundancy-needs-independence}} — this chapter is those results applied to a
generator and its own reviewer.

{{ch:aids-automl}}'s {{eq:search-must-be-scored-off-search}}, of which self-judging
is the extreme case: the search scores itself.

{{ch:mcp-production}}'s {{eq:review-does-not-scale}}, which returns here as the
capacity limit on the only verification that works.

{{ch:aids-agentic-eda}}'s boundary rule, whose larger-scale version is this
chapter's conclusion.

## 4. Intuitive Explanation

The pipeline is genuinely impressive. It reads literature, proposes a hypothesis,
writes code, runs experiments, makes figures, drafts a paper, and reviews the
result — end to end, unattended, for the price of a modest lunch.

Then it reports how good the output is, and the reporting is where the trouble
starts, because the reviewer is built from the same model as the writer.

Consider what that means concretely. A language model has characteristic strengths
and characteristic blind spots — kinds of reasoning it does well, kinds of error it
makes repeatedly and does not notice. When it writes a paper, its flaws are drawn
from its blind-spot distribution: those are the mistakes it makes.

When it reviews a paper, it catches errors it is good at seeing. Which are, by
construction, not the errors it makes.

So the reviewer is not a bad reviewer. On a random paper written by someone else, it
might be excellent — and the authors' near-human agreement figure was presumably
measured on such papers. On papers written by *itself*, it is systematically weak
exactly where it needs to be strong.

{{sec:9-practical-example}} shows the resulting acceptance rate rising with shared
bias while the underlying quality does not move at all. The number the pipeline
reports is a measure of how much the writer and reviewer have in common.

That has a sharp consequence for how to read a reviewer accuracy figure. Agreement
with humans, measured on some distribution of papers, does not transfer to the
distribution the generator produces — because that distribution is selected to be
the one the model finds natural, which is the one its blind spots cover.

Two fixes suggest themselves and neither works.

**Make the reviewer better.** At high shared bias, a large improvement in
specificity buys little, because the additional accuracy applies to errors that
were already being caught. You cannot fix a blind spot by looking harder.

**Use several reviewers.** They are from the same family, so they share the blind
spot with each other as well as with the generator. Seven of them barely beat one —
{{ch:as-failures}}' finding that correlated votes are one opinion restated.

What works is independence: a different model family, or a human. Not a better
judge — a *differently wrong* one.

Then the second question, which is the more interesting economic one.

At \$15 a paper you do not generate one. You generate a thousand and keep the good
ones — and this is a real argument. {{cite:testini2025dsautomation}} names it as the
gap in how automation is evaluated: the field measures substitution, doing the same
work faster, and ignores redesign, doing differently-shaped work because the cost
structure changed. Every listing in this part so far prices a fixed workload done
faster and cannot see this.

So take it seriously. {{sec:9-practical-example}} generates ten thousand and finds
more sound results than generating ten — genuinely more — at the same $18\%$
precision. Volume works, in the sense that the numerator grows.

The problem is the filter. Keeping the good ones requires identifying the good
ones, and the cheap filter is the correlated one. The filter that works is
independent review, which costs someone's time — and time does not get cheaper
because generation did.

The listing puts unlimited expert review at about \$2,000 per sound result. That may
still be worth it; it is a different claim from \$15 a paper.

And expert review is not unlimited. With fixed capacity, volume overflows it and
the overflow ships unreviewed, so precision *falls* as generation rises. Worse,
reviewer attention is a commons: flooding a review system degrades the filter
everyone depends on, including the pipeline's own future runs.

## 5. Formal Explanation

**Self-judging.** Let generated work be sound with probability $\pi$. A judge
accepts sound work with sensitivity $s$ and detects flaws with specificity $\sigma$
*on flaws independent of its own blind spots*. Model a flaw's visibility to the
judge as $V \sim \mathcal{N}(-\lambda\rho, 1)$, where $\rho$ is the shared bias
between generator and judge and $\lambda$ scales how far shared bias pushes flaws
into the blind region. With threshold $t = \Phi^{-1}(1-\sigma)$:

$$\Pr[\text{detect} \mid \text{flawed}] = 1 - \Phi(t + \lambda\rho)$$

decreasing in $\rho$. The acceptance rate is:

$$A(\rho) = \pi s + (1-\pi)\Phi(t + \lambda\rho)$$ (eq:self-judging-measures-correlation)

**$A$ is increasing in $\rho$ and the true sound fraction $\pi s$ is constant in
it.** So the reported number moves with the correlation and not with the quality.

**Why accuracy does not substitute for independence.** Differentiating precision
$P = \pi s / A$ with respect to $\sigma$:

$$\frac{\partial P}{\partial \sigma}\Big|_{\rho} \propto \phi(t)\,\frac{\partial t}{\partial \sigma} \cdot \frac{\phi(t+\lambda\rho)}{\phi(t)}$$ (eq:accuracy-is-not-independence)

and the ratio $\phi(t+\lambda\rho)/\phi(t)$ decays in $\rho$. **Improvements in
specificity are attenuated by exactly the factor that shared bias introduces**
({{eq:accuracy-is-not-independence}}), so the marginal return on a better
same-family judge falls as the correlation rises.

**Panels.** With $k$ judges sharing bias $\rho$ with the generator and with each
other, their detections are positively correlated, so the majority vote's effective
size is {{ch:as-failures}}'s $k_{\text{eff}} \approx 1 + (k-1)(1-\rho)$:

$$\lim_{\rho\to 1} P_{\text{panel}}(k) = P_{\text{panel}}(1) \quad \text{for all } k$$ (eq:correlated-panels-are-one-judge)

**Generation versus verification.** Let generation cost $c_g$ per item and
independent verification $c_v$ per item, with verification detecting a fraction
$d$ of flaws. Producing $n$ items and verifying all of them yields
$n\pi s$ sound results at cost $n(c_g + c_v)$, so:

$$\text{cost per sound result} = \frac{c_g + c_v}{\pi s}$$ (eq:generation-scales-verification-does-not)

Since $c_v \gg c_g$ and $c_v$ is set by human time while $c_g$ falls with compute,
**the ratio $c_v/c_g$ grows over time** and the cost per usable result is
asymptotically all verification.

**Capacity.** With verification capacity $K$ items and $n > K$ generated, a
fraction $K/n$ is filtered and the rest ships raw:

$$P(n) = \frac{\pi s}{\pi s + (1-\pi)\big[(1 - K/n) + (K/n)(1-d)\big]} \;\xrightarrow[n\to\infty]{}\; \pi s$$ (eq:volume-overflows-the-filter)

**Precision converges to the unfiltered rate as volume grows.** The filter's effect
vanishes, not because it stopped working but because most output no longer passes
through it.

**And the commons.** With $R$ reviewers and {{eq:habituation}}'s decay, per-item
detection is $d(n) = d_0/(1 + n/(R h))$, so total flaws caught is:

$$C(n) = n\,(1-\pi)\,\frac{d_0}{1 + n/(Rh)} \;\longrightarrow\; (1-\pi) d_0 R h$$ (eq:attention-is-a-commons)

**Bounded, regardless of $n$.** A review system catches a fixed absolute number of
flaws per period, so additional volume takes a share of a constant resource — and
the depletion falls on everyone using that system, not only the producer.

## 6. Mathematical Foundation

Three extractions.

**The reported number is a function of the wrong variable.** From
{{eq:self-judging-measures-correlation}}, $\partial A/\partial \rho > 0$ and
$\partial(\pi s)/\partial \rho = 0$. That is unusually clean: the metric responds
to a quantity orthogonal to what it claims to measure, so it can be driven up
without touching quality — and would be, by anything that made generator and judge
more alike, such as training them on each other's output.

**Accuracy and independence are not substitutes.**
{{eq:accuracy-is-not-independence}}'s attenuation factor means the two enter
multiplicatively, not additively. There is no specificity at which a fully
correlated judge becomes adequate.

**Verification cost dominates asymptotically.** From
{{eq:generation-scales-verification-does-not}}, the cost per usable result tends to
$c_v/(\pi s)$ as $c_g \to 0$. **Reducing generation cost to zero leaves the cost per
usable result almost unchanged**, which is the sharpest statement of why the \$15
figure describes half the system.

## 7. Internal Mechanics

### 7.1 The circularity, drawn

```mermaid {#fig:self-judging caption="A generator and a judge from one model family. The flaws the generator produces are drawn from the region the judge cannot see, because both are properties of the same model."}
flowchart TD
    MF[one model family] --> G[generator]
    MF --> J[judge]
    G -->|"flaws drawn from<br/>the family's weak regions"| W[the work]
    W --> J
    J -->|"catches flaws in<br/>the family's strong regions"| V[verdict]
    V --> R["reported acceptance rate"]
    MF -.->|"determines both,<br/>identically"| R
```

The dotted edge is the whole diagram. The reported number is downstream of the
model family twice — once through what gets written and once through what gets
caught — so it cannot be read as evidence about either.

### 7.2 What "near-human agreement" does and does not establish

{{cite:lu2024aiscientist}} reports its automated reviewer achieving near-human
performance on paper scoring, and that is a real result worth taking at face value.
It is also measured on the wrong distribution for the claim it supports.

Reviewer agreement is measured on some corpus of papers — typically human-written
ones with known outcomes. The generator produces papers from a different
distribution: ones its own model finds natural to write.

So the transfer requires an assumption nobody states: **that the reviewer's accuracy
on human-written papers predicts its accuracy on its own family's papers.**
{{eq:self-judging-measures-correlation}} says the opposite — the generator's output
is concentrated exactly where the judge is weakest.

Testing this is straightforward and worth doing: score the automated reviewer's
agreement with human reviewers *on the pipeline's own output* rather than on a
general corpus. That number would license the claim. It is not, as far as this
chapter's sources report, published.

### 7.3 Where independence can come from

Three sources, in decreasing order of independence and cost.

**Execution.** The strongest by far, and this chapter's most useful practical note.
An experiment's code either runs and produces the claimed numbers or it does not,
and that check is independent of every model. For computational research this is
available and it grades the part of the claim most likely to be wrong.

**A different model family.** Weaker than it looks — models trained on overlapping
data with similar objectives share more than their vendors differ — but
substantially better than same-family, and cheap.

**Human expert review.** The real thing, at {{sec:9-practical-example}}'s
\$340 per review, subject to {{eq:review-does-not-scale}}.

The ordering suggests the deployment: verify mechanically what can be verified
mechanically, use a different family for what cannot, and spend human review on the
residual. That is {{ch:aids-stack}}'s check-strong-build-weak rule at the pipeline
level.

### 7.4 The redesign argument, taken seriously

It would be easy to read this chapter as dismissing volume, and that would be
wrong. {{cite:testini2025dsautomation}}'s third gap is real and this book has been
mostly blind to it.

If a hypothesis costs \$15 to test rather than three weeks, the correct response is
not to test the same hypothesis faster. It is to test hypotheses that were never
worth three weeks — the long tail of "probably nothing, but worth a look" that has
always been left undone because the fixed cost exceeded the expected value.

That is a genuine change in what is possible, and {{sec:9-practical-example}}'s
first table supports it: ten thousand generations delivered far more sound results
than ten.

The chapter's argument is narrower. **The volume is only useful to the extent the
good ones can be identified**, and identification is the half that did not get
cheap. A pipeline that generates a thousand results and cannot tell which forty are
sound has produced a thousand claims, which is worse than none.

Where volume plainly does work is where the verifier is mechanical: hyperparameter
studies, ablations, replication attempts, and anything whose success criterion is
executable. There the cheap-generation argument goes through completely, and it is
the strongest case for these systems.

### 7.5 The boundary rule, at scale

{{ch:aids-agentic-eda}} concluded that unconfirmed exploratory output must not cross
the boundary from the analyst to a decision-maker. The same rule applies here with
larger stakes.

An automated pipeline's unverified output is a *draft*, and drafts are useful.
The failure mode is that a well-formatted paper with figures and a related-work
section does not look like a draft — it looks like a result, and it circulates as
one.

So the boundary needs enforcing structurally rather than by convention: unverified
output labelled as such in a way that survives copying, verification status carried
with the artefact, and separate channels for verified and unverified work. That is
{{ch:as-state-machines}}'s provenance argument in a research setting.

### 7.6 Why this degrades a shared resource

{{eq:attention-is-a-commons}} says a review system catches a bounded number of flaws
per period regardless of submission volume. That has a consequence beyond one
pipeline's quality.

Every unverified submission consumes reviewer attention that would otherwise go to
something else, and generation does not replenish it. So a pipeline that submits at
volume degrades the filter for every other participant — and, since its own future
submissions face the same degraded filter, for itself.

{{ch:mcp-production}} found the same structure in a registry: **a control whose cost
is paid per unit of volume does not scale, because volume is set by someone else.**
Here the someone else is a generator that got a thousand times cheaper.

This is the strongest argument for verification that is mechanical rather than
human, and the reason {{sec:7-internal-mechanics}}'s ordering puts execution first.

### 7.7 The one number that would settle it

Everything in this chapter reduces to a quantity nobody publishes, and it is worth
stating exactly what the missing experiment is, because it is cheap.

Take the pipeline's own output — not a general corpus, not human-written papers.
Have independent human experts judge a sample of it. Compare their verdicts with
the automated reviewer's verdicts **on those same items.**

That single number is what licenses or refutes every claim of the form "our system
produces work exceeding an acceptance threshold". It is not the same as reviewer
agreement on a general corpus, and {{eq:self-judging-measures-correlation}} says why:
the generator's output is concentrated in the region where the judge is weakest, so
general agreement is measured off-distribution for the claim being made.

The experiment costs a few dozen expert reviews. The reason it is rarely run is not
expense — it is that the result has only one interesting direction. High agreement
on own-output would be a genuinely strong result and would settle the question in
the pipeline's favour; low agreement invalidates the headline. A measurement whose
downside is retracting your main claim does not get prioritised.

Which is a general observation about self-evaluation worth carrying past this
chapter. **When a system reports its own quality, ask what measurement would
falsify the report, and whether it was run.** In {{ch:aids-automl}} that question
was "what was the search size"; in {{ch:aids-agentic-eda}} it was "how many
comparisons"; here it is "how does your judge do on your own output".

All three are the same question — what is the denominator, or the distribution, or
the correlation that the reported number depends on and does not include — and in
all three the automation is what removed it from view.

## 8. Implementation

Two listings. The first measures what a self-judged pipeline's acceptance rate
tracks. The second prices generation and verification separately.

```python {tier=A name=self-judging-measures-correlation}
"""When the generator and the judge share a blind spot.

cite:lu2024aiscientist built an end-to-end research pipeline -- idea, code,
experiments, figures, paper, review -- at under $15 per paper, and reported that it
can produce work exceeding a top conference's acceptance threshold AS JUDGED BY ITS
OWN AUTOMATED REVIEWER, which the authors report achieves near-human agreement on
paper scores.

That last clause is the measurement problem, and this book has the apparatus to
price it. ch:as-failures found that agents sharing a base model, a prompt lineage
and a context have correlated errors. A generator and a judge built from the same
model share all three.

So the question is not whether the judge is accurate in general. It is whether the
judge's errors are independent of the generator's -- because a generator that
produces flaws of the kind its own model family cannot see is producing work its
own judge will certify (eq:self-judging-measures-correlation).

Correlation here means one specific thing: the more the generator and judge share,
the more the generator's flaws fall in the region the judge is blind to.
"""
import numpy as np
from math import erf, sqrt

rng = np.random.default_rng(4787)

M = 60000
P_GOOD = 0.18           # share of generated work that is genuinely sound
JUDGE_SENS = 0.80       # judge accepts sound work at this rate
JUDGE_SPEC = 0.78       # judge's detection rate on INDEPENDENT flaws
SHIFT = 1.9             # how far shared bias pushes flaws toward invisibility


def norm_ppf(p):
    """Inverse normal CDF by bisection -- no scipy needed."""
    lo, hi = -9.0, 9.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(rho, m=M, p_good=P_GOOD, sens=JUDGE_SENS, spec=JUDGE_SPEC, shift=SHIFT):
    """`rho` is how much the generator's flaws are drawn from the region its own
    judge cannot see. At rho=0 flaws are independent of the judge's blind spot;
    at rho=1 they sit squarely inside it.

    Returns (accept rate, precision among accepted, sound accepted,
    unsound accepted).
    """
    good = rng.random(m) < p_good
    # A flaw's visibility TO THIS JUDGE. Calibrated so that at rho = 0 the judge
    # detects `spec` of flaws; shared bias shifts the distribution downward.
    t = norm_ppf(1.0 - spec)
    vis = rng.normal(-shift * rho, 1.0, m)
    detected = vis > t

    accept = np.empty(m, dtype=bool)
    accept[good] = rng.random(int(good.sum())) < sens
    accept[~good] = ~detected[~good]

    tp = float((accept & good).mean())
    fp = float((accept & ~good).mean())
    acc = tp + fp
    return acc, (tp / acc if acc else 0.0), tp, fp


print(f"{M:,} generated papers, {P_GOOD:.0%} of them genuinely sound. The judge")
print(f"accepts sound work {JUDGE_SENS:.0%} of the time and detects {JUDGE_SPEC:.0%}")
print("of flaws that are independent of its own blind spots.")
print()
print(f"{'shared bias':>13}{'flaws caught':>14}{'accept rate':>13}"
      f"{'precision':>11}{'unsound accepted':>18}")
print("-" * 69)
tab = {}
for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
    r = run(rho)
    t = norm_ppf(1.0 - JUDGE_SPEC)
    caught = 1 - 0.5 * (1 + erf((t + SHIFT * rho) / sqrt(2)))
    tab[rho] = r + (caught,)
    print(f"{rho:>13.2f}{caught:>14.1%}{r[0]:>13.1%}{r[1]:>11.1%}{r[3]:>18.1%}")

print()
print()
print("What the pipeline REPORTS against what is true. The reported figure is")
print("the acceptance rate; the true figure is the share that is sound.")
print()
print(f"{'shared bias':>13}{'reported accept':>17}{'actually sound':>16}"
      f"{'overstatement':>15}")
print("-" * 61)
for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
    r = tab[rho]
    print(f"{rho:>13.2f}{r[0]:>17.1%}{r[2]:>16.1%}{r[0] - r[2]:>15.1%}")

print()
print()
print("An independent judge -- a different model family, or a human -- at the")
print("SAME nominal accuracy. Only the shared bias differs.")
print()
print(f"{'judge':>28}{'accept rate':>13}{'precision':>11}")
print("-" * 52)
ind = run(0.0)
dep = run(0.9)
print(f"{'independent (shared 0.00)':>28}{ind[0]:>13.1%}{ind[1]:>11.1%}")
print(f"{'same family (shared 0.90)':>28}{dep[0]:>13.1%}{dep[1]:>11.1%}")
print()
print(f"   Identical nominal sensitivity and specificity. Precision differs by")
print(f"   {(ind[1] - dep[1]) * 100:.1f} points, entirely because the errors line up.")

print()
print()
print("Making the judge more accurate does not rescue it, because the failure")
print("is not an accuracy failure.")
print()
print(f"{'judge specificity':>19}" + "".join(f"{'shared ' + format(r, '.1f'):>15}"
                                            for r in (0.0, 0.5, 0.9)))
print("-" * 64)
sp = {}
for s in (0.70, 0.85, 0.95, 0.99):
    row = tuple(run(r, spec=s)[1] for r in (0.0, 0.5, 0.9))
    sp[s] = row
    print(f"{s:>19.0%}" + "".join(f"{v:>15.1%}" for v in row))


def panel(rho, k, m=M, p_good=P_GOOD, sens=JUDGE_SENS, spec=JUDGE_SPEC,
          shift=SHIFT):
    """k judges vote and the majority decides. Judges from the same family share
    their blind spots WITH EACH OTHER as well as with the generator."""
    good = rng.random(m) < p_good
    t = norm_ppf(1.0 - spec)
    shared = rng.normal(-shift * rho, 1.0, m)      # the common component
    votes = np.zeros(m, dtype=np.int64)
    for _ in range(k):
        vis = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, m) \
            - shift * rho * (1 - np.sqrt(rho))
        detected = vis > t
        a = np.empty(m, dtype=bool)
        a[good] = rng.random(int(good.sum())) < sens
        a[~good] = ~detected[~good]
        votes += a
    accept = votes > k / 2
    tp = float((accept & good).mean())
    fp = float((accept & ~good).mean())
    acc = tp + fp
    return acc, (tp / acc if acc else 0.0)


print()
print()
print("And a panel of judges, which is ch:as-failures' result arriving here")
print("unchanged: correlated votes do not aggregate.")
print()
print(f"{'judges':>9}" + "".join(f"{'shared ' + format(r, '.1f'):>15}"
                                 for r in (0.0, 0.5, 0.9)))
print("-" * 54)
pn = {}
for k in (1, 3, 7):
    row = tuple(panel(r, k)[1] for r in (0.0, 0.5, 0.9))
    pn[k] = row
    print(f"{k:>9}" + "".join(f"{v:>15.1%}" for v in row))

print(f"""
The first table's accept column rises and the sound column does not.

At zero shared bias the pipeline accepts {tab[0.0][0]:.1%} of what it generates; at
{0.95:.2f} shared bias it accepts {tab[0.95][0]:.1%}. The share that is actually
sound is {tab[0.0][2]:.1%} and {tab[0.95][2]:.1%} -- **the same number, because
nothing about the generator changed.**

What changed is how much the judge and the generator have in common. And the
accept rate, which is the number the pipeline reports, moved
{tab[0.95][0] - tab[0.0][0]:.1f} points on that alone.

**A self-judged pipeline's acceptance rate measures the correlation between its
generator and its judge, not the quality of its output**
(eq:self-judging-measures-correlation). That is a strong claim and the second table
is its direct statement: the overstatement grows from
{tab[0.0][0] - tab[0.0][2]:.1%} to {tab[0.95][0] - tab[0.95][2]:.1%} while the truth
sits still.

The third table isolates it cleanly. Two judges with **identical** nominal
sensitivity and specificity: an independent one delivers {ind[1]:.1%} precision, a
same-family one {dep[1]:.1%}. Nothing about their measured accuracy differs. Only
whether their errors line up with the generator's.

Which means a reported reviewer accuracy -- "our automated reviewer achieves
near-human agreement on paper scores" -- **does not license the conclusion that its
acceptances are sound**, because agreement is measured on a distribution of papers
that is not the distribution its own generator produces. The generator writes
papers of a particular kind, and the judge's competence on THAT kind is the only
relevant number.

The fourth table forecloses the obvious response. At {0.9:.1f} shared bias, driving
the judge's specificity from {0.70:.0%} to {0.99:.0%} moves precision from
{sp[0.70][2]:.1%} to {sp[0.99][2]:.1%}. Independent, the same improvement moves it
from {sp[0.70][0]:.1%} to {sp[0.99][0]:.1%}.

**Making the judge more accurate does not rescue it, because the failure is not an
accuracy failure.** A better reviewer of the same family is better at catching the
flaws it could already catch.

And the last table forecloses the second obvious response. Seven judges at zero
shared bias reach {pn[7][0]:.1%}; seven at {0.9:.1f} reach {pn[7][2]:.1%}, barely
above one judge's {pn[1][2]:.1%}.

That is ch:as-failures' result exactly -- correlated votes do not aggregate,
because they are one opinion restated -- and it means an ensemble of reviewers from
one model family is a reviewer.

The practical consequences are narrow and hard.

**An automated reviewer must be independent of the generator to license anything**,
which in practice means a different model family, a different prompt lineage, or a
human. Same-family review is useful for catching the errors a model makes
carelessly and useless for the errors it makes systematically -- and the systematic
ones are the ones that survive to the output.

**Report the accept rate alongside the shared-bias estimate, or do not report it.**
The number alone is uninterpretable.

**And treat "judged by our own reviewer" as an unverified claim** rather than a
weak one. It is not a low-quality measurement; it is a measurement of a different
quantity than the one being claimed.""")
```

The second listing asks what cheap generation actually buys.

```python {tier=A name=generation-scales-verification-does-not}
"""The economics of cheap generation, and what does not get cheaper.

cite:lu2024aiscientist reports under $15 per paper. At that price the natural move
is volume: generate a thousand, keep the good ones. That is
cite:testini2025dsautomation's third gap -- automation as REDESIGN rather than
substitution -- and it is a real argument that this book's earlier listings, which
all price a fixed workload done faster, cannot see.

This listing takes it seriously and finds where it breaks. Generation gets cheap.
Verification does not, because the only verification that works is the kind that
does not share the generator's blind spots -- and that means independent effort,
which is the thing whose cost is set by someone else's time
(eq:generation-scales-verification-does-not).
"""
import numpy as np

rng = np.random.default_rng(4831)

M = 40000
P_GOOD = 0.18
GEN_COST = 15.0             # dollars per generated result
CHEAP_VERIFY = 0.40         # a same-family automated review
CHEAP_PREC = 0.90           # its detection rate on flaws it can see
CHEAP_SHARED = 0.90         # ...but it shares the generator's blind spots
EXPERT_COST = 340.0         # an independent expert review, in dollars of time
EXPERT_DETECT = 0.82


def yield_of(n_generated, use_cheap=True, use_expert=False,
             expert_capacity=None, p_good=P_GOOD):
    """Returns (sound results delivered, unsound delivered, total cost,
    expert reviews consumed)."""
    good = rng.binomial(n_generated, p_good)
    bad = n_generated - good
    cost = n_generated * GEN_COST
    passed_good, passed_bad = good, bad

    if use_cheap:
        cost += n_generated * CHEAP_VERIFY
        # The cheap reviewer only catches flaws outside the shared blind spot.
        eff = CHEAP_PREC * (1.0 - CHEAP_SHARED)
        passed_bad = rng.binomial(passed_bad, 1.0 - eff)
        passed_good = rng.binomial(passed_good, 0.95)

    reviews = 0
    if use_expert:
        submitted = passed_good + passed_bad
        cap = submitted if expert_capacity is None else min(submitted,
                                                            expert_capacity)
        reviews = cap
        frac = cap / submitted if submitted else 0.0
        cost += cap * EXPERT_COST
        # Only the reviewed share is filtered; the rest passes unexamined.
        rev_bad = rng.binomial(passed_bad, frac)
        rev_good = rng.binomial(passed_good, frac)
        kept_bad = (passed_bad - rev_bad) + rng.binomial(
            rev_bad, 1.0 - EXPERT_DETECT)
        kept_good = (passed_good - rev_good) + rng.binomial(rev_good, 0.93)
        passed_good, passed_bad = kept_good, kept_bad

    return int(passed_good), int(passed_bad), float(cost), int(reviews)


print(f"Generation at ${GEN_COST:.0f} each, {P_GOOD:.0%} of results sound.")
print(f"A same-family automated review costs ${CHEAP_VERIFY:.2f} and shares")
print(f"{CHEAP_SHARED:.0%} of the generator's blind spots. An independent expert")
print(f"review costs ${EXPERT_COST:.0f} of someone's time and catches "
      f"{EXPERT_DETECT:.0%}.")
print()
print(f"{'generated':>11}{'delivered sound':>17}{'delivered unsound':>19}"
      f"{'precision':>11}{'cost':>12}")
print("-" * 70)
tab = {}
for n in (10, 100, 1000, 10000):
    g, b, c, _ = yield_of(n)
    tab[n] = (g, b, c, g / (g + b) if g + b else 0)
    print(f"{n:>11,}{g:>17,}{b:>19,}{g / max(g + b, 1):>11.1%}{c:>12,.0f}")

print()
print()
print("Volume delivers more sound results AND more unsound ones, in fixed")
print("proportion, because the filter is correlated with the generator.")
print()
print(f"{'generated':>11}{'sound per $1k':>16}{'unsound per $1k':>18}")
print("-" * 46)
for n in (10, 100, 1000, 10000):
    g, b, c, _ = tab[n]
    print(f"{n:>11,}{g / (c / 1000):>16.2f}{b / (c / 1000):>18.2f}")

print()
print()
print("Now add independent expert review, unlimited. This is the version that")
print("works, and its cost is the reason it is not what gets deployed.")
print()
print(f"{'generated':>11}{'sound':>9}{'unsound':>10}{'precision':>11}"
      f"{'cost':>13}{'$ per sound':>14}")
print("-" * 68)
ex = {}
for n in (10, 100, 1000):
    g, b, c, r = yield_of(n, use_expert=True)
    ex[n] = (g, b, c, r)
    print(f"{n:>11,}{g:>9,}{b:>10,}{g / max(g + b, 1):>11.1%}{c:>13,.0f}"
          f"{c / max(g, 1):>14,.0f}")

print()
print()
print("And the version that actually happens: expert capacity is fixed, so")
print("volume overflows it and the overflow ships unreviewed.")
print()
CAPACITY = 40
print(f"Expert capacity: {CAPACITY} reviews.")
print()
print(f"{'generated':>11}{'reviewed':>10}{'unreviewed':>12}{'precision':>11}"
      f"{'sound delivered':>17}")
print("-" * 61)
cap = {}
for n in (10, 100, 1000, 10000):
    g, b, c, r = yield_of(n, use_expert=True, expert_capacity=CAPACITY)
    submitted = g + b
    cap[n] = (g, b, r, g / max(submitted, 1))
    print(f"{n:>11,}{r:>10,}{max(submitted - r, 0):>12,}"
          f"{g / max(submitted, 1):>11.1%}{g:>17,}")

print()
print()
print("The commons. Every generated result that reaches a reviewer consumes")
print("attention that is not replenished by generating more.")
print()
print(f"{'generated':>11}{'submitted':>12}{'per reviewer':>15}"
      f"{'catch rate':>12}")
print("-" * 50)
REVIEWERS = 4
HALF = 25
cm = {}
for n in (10, 100, 1000, 10000):
    g, b, _, _ = yield_of(n)
    submitted = g + b
    load = submitted / REVIEWERS
    catch = EXPERT_DETECT / (1.0 + load / HALF)
    cm[n] = (submitted, load, catch)
    print(f"{n:>11,}{submitted:>12,}{load:>15,.0f}{catch:>12.1%}")

print(f"""
The first two tables make the volume argument and then undercut it.

Generating {10000:,} results delivers {tab[10000][0]:,} sound ones against
{tab[10][0]:,} from generating {10}. Volume works. But it also delivers
{tab[10000][1]:,} unsound ones, and the precision is
{tab[10000][3]:.1%} at both scales.

**Cheap generation with a correlated filter scales the output and not the
proportion.** You get more of everything, in the ratio the generator produces, and
the ratio is what determines whether the output can be used without checking it.

The third table adds the filter that works, and prices it.

With unlimited independent expert review, precision rises to
{ex[1000][0] / max(ex[1000][0] + ex[1000][1], 1):.1%} -- and the cost per sound
result is ${ex[1000][2] / max(ex[1000][0], 1):,.0f}.

**Generation costs ${GEN_COST:.0f} and verification costs
${ex[1000][2] / max(ex[1000][0], 1):,.0f} per usable result**
(eq:generation-scales-verification-does-not). The famous figure is the cost of the
half that got cheap.

That is not an argument against the pipeline. It is an argument about which number
describes it: a system that produces a sound result for
${ex[1000][2] / max(ex[1000][0], 1):,.0f} may still be excellent value, and it is a
different claim from one that produces a paper for ${GEN_COST:.0f}.

The fourth table is what actually happens, because expert capacity is not
unlimited. With {CAPACITY} reviews available, generating {10} gives
{cap[10][3]:.1%} precision and generating {10000:,} gives {cap[10000][3]:.1%} --
because {cap[10000][1] + cap[10000][0] - cap[10000][2]:,} results ship unreviewed.

**Volume overflows the filter and the overflow is the output.** The system's
quality is set by its verification capacity, and generating more does not add
capacity.

The last table is the part that reaches beyond one team, and it is
ch:mcp-production's review-queue result arriving in a new setting. Submitted volume
of {cm[10][0]:,} leaves each reviewer {cm[10][1]:,.0f} items and a catch rate of
{cm[10][2]:.1%}. Volume of {cm[10000][0]:,} leaves {cm[10000][1]:,.0f} each and
{cm[10000][2]:.1%}.

Reviewer attention is a commons. Generating more consumes it and does not
replenish it, so **a pipeline that floods a review system degrades the filter that
every other participant depends on** -- including its own future runs.

Which gives the honest summary of autonomous research pipelines as they currently
stand.

The generation half is real, cheap, and getting cheaper. The verification half is
the binding constraint, does not get cheaper, and is the only part that determines
whether the output is worth anything. **The correct response to cheap generation is
not more generation. It is independent verification that scales**, and nobody has
one.

Until then the defensible deployment is narrow and useful: generate abundantly,
have a human expert verify a small number, and **never let the unverified output
cross a boundary** -- which is ch:aids-agentic-eda's rule, at a larger scale and
with the same reasoning.""")
```

## 9. Practical Example

The first listing generates papers of which $18\%$ are sound, judged by a reviewer
that would catch $78\%$ of independent flaws:

```
  shared bias  flaws caught  accept rate  precision  unsound accepted
---------------------------------------------------------------------
         0.00         78.0%        32.7%      44.4%             18.2%
         0.50         42.9%        61.6%      23.5%             47.1%
         0.95         15.1%        84.2%      17.1%             69.8%
```

Against the truth:

```
  shared bias  reported accept  actually sound  overstatement
-------------------------------------------------------------
         0.00            32.7%           14.5%          18.2%
         0.95            84.2%           14.4%          69.8%
```

**The accept rate moves $51.5$ points and the sound share does not move at all**
({{eq:self-judging-measures-correlation}}). Nothing about the generator changed;
only how much it has in common with its judge.

Isolated:

```
                       judge  accept rate  precision
----------------------------------------------------
   independent (shared 0.00)        32.5%      44.0%
   same family (shared 0.90)        82.1%      17.2%
```

Identical nominal sensitivity and specificity; $26.8$ points of precision
difference, entirely from the errors lining up.

The two obvious fixes:

```
  judge specificity     shared 0.0     shared 0.5     shared 0.9
----------------------------------------------------------------
                70%          37.5%          20.9%          16.6%
                99%          94.3%          67.7%          39.6%
```

```
   judges     shared 0.0     shared 0.5     shared 0.9
------------------------------------------------------
        1          44.8%          23.5%          17.6%
        7          81.6%          26.8%          20.4%
```

**Neither a better judge nor more judges rescues a correlated one**
({{eq:accuracy-is-not-independence}},
{{eq:correlated-panels-are-one-judge}}) — the first improves detection of errors
already caught, and the second is one opinion restated.

The second listing prices volume. Generation at \$15, a same-family automated
review at \$0.40 sharing $90\%$ of the blind spots:

```
  generated  delivered sound  delivered unsound  precision        cost
----------------------------------------------------------------------
         10                1                  8      11.1%         154
      1,000              161                758      17.5%      15,400
     10,000            1,706               7,488      18.6%     154,000
```

Volume raises the numerator and leaves the ratio. Adding unlimited independent
expert review:

```
  generated    sound   unsound  precision         cost   $ per sound
--------------------------------------------------------------------
        100       13        12      52.0%       34,180         2,629
      1,000      160       143      52.8%      324,460         2,028
```

**Generation costs \$15 and verification costs about \$2,000 per usable result**
({{eq:generation-scales-verification-does-not}}). The famous figure prices the half
that got cheap.

With realistic capacity:

```
Expert capacity: 40 reviews.

  generated  reviewed  unreviewed  precision  sound delivered
-------------------------------------------------------------
         10        10           0      33.3%                1
      1,000        40         858      17.1%              154
     10,000        40       9,110      18.6%            1,706
```

**Volume overflows the filter and the overflow is the output**
({{eq:volume-overflows-the-filter}}) — precision converges to the unfiltered rate.

And the shared cost:

```
  generated   submitted   per reviewer   catch rate
---------------------------------------------------
         10          10              2        74.5%
      1,000         919            230         8.0%
     10,000       9,194          2,298         0.9%
```

**Reviewer attention is a commons that volume depletes**
({{eq:attention-is-a-commons}}) — for every participant, including the pipeline's
own future runs.

## 10. Production Considerations

Never report a self-judged acceptance rate as a quality measure. It tracks
correlation.

Measure your reviewer's agreement with humans **on your own generator's output**,
not on a general corpus. That is the number that licenses the claim, and it is
usually not the one reported.

Verify by execution wherever the claim is executable. It is the only fully
independent verifier available cheaply, and for computational research it grades
the part most likely to be wrong.

Use a different model family where execution is not available, and treat it as
partial independence rather than full.

Price generation and verification separately in any cost claim. The two have
different scaling and reporting only the first is misleading.

Do not increase volume beyond your verification capacity. Past that point precision
falls toward the raw generation rate.

Label unverified output structurally, so the label survives copying. A formatted
paper does not look like a draft.

And treat submission to a shared review system as consuming a bounded resource,
because it is.

## 11. Common Mistakes

**Reading a self-judged accept rate as quality.** It is a correlation measurement.

**Transferring reviewer accuracy across distributions.** Measured on human papers,
applied to the model's own.

**Improving the same-family judge.** Attenuated by exactly the factor at issue.

**Adding more same-family judges.** One opinion restated.

**Quoting the generation cost as the system cost.** Verification is the larger and
non-shrinking half.

**Scaling generation past verification capacity.** Precision converges to
unfiltered.

**Letting unverified output circulate.** It does not look unverified.

## 12. Failure Modes

*Certified unsound work.* The characteristic failure: accepted by its own judge
precisely because the flaw is of a kind that judge cannot see.

*Confident cost claim.* A per-item figure that omits the item's verification.

*Filter overflow.* A pipeline whose quality fell as its throughput rose, with
nothing in the pipeline indicating it.

*Commons depletion.* A shared review system degraded for everyone, including the
depleter.

*Draft laundering.* Unverified output circulating as a result because its
presentation is indistinguishable from one.

*Correlation drift.* A judge fine-tuned on the generator's output, becoming more
correlated over time and reporting improving numbers.

## 13. Alternatives

**Execution-graded pipelines.** Restrict autonomous work to claims a program can
check — ablations, replications, hyperparameter studies. The cheap-generation
argument goes through completely here and it is the strongest deployment.

**Adversarial verification.** A judge explicitly prompted and selected to
disconfirm, from a different family, per {{ch:as-failures}}'s refuter pattern.

**Human-in-the-loop at the idea stage.** Spend the scarce independent attention on
what to investigate rather than on checking what was produced — cheaper per unit of
influence.

**Registered reports.** Fix the hypothesis and analysis before running, which makes
the result gradeable by protocol adherence rather than by outcome.

**Generation as draft assistance only.** Never crossing the boundary, which forgoes
the volume argument entirely and is defensible where no independent verifier
exists.

## 14. Evaluation

Measure reviewer agreement on your own output against independent human judgement.
It is the single missing number in every self-judged pipeline.

Report generation cost and verification cost as separate line items.

Measure precision as a function of throughput, not at one operating point. The
overflow effect is invisible at a single volume.

Track the fraction of output that receives independent verification, and treat it
as a capacity metric.

Estimate the correlation between your generator and judge directly — by comparing
their errors on a common set with known ground truth — and report the effective
number of independent judges.

And audit accepted-but-unsound output as a category. It is the failure this chapter
is about and it is recoverable from history.

## 15. Advanced Concepts

**Cross-family verification protocols.** Standardising which families count as
independent, and measuring residual correlation between them. Nobody publishes
this. {{maturity:EMERGING}}.

**Execution-grounded research agents.** Constraining autonomous work to
mechanically checkable claims, which converts the verification cost from human time
to compute. {{maturity:EMERGING}}.

**Correlation estimation from disagreement.** Inferring generator-judge correlation
from their disagreement rate on unlabelled output, which would make the reported
accept rate interpretable without ground truth.

**Verification that scales.** The open problem this chapter identifies and does not
solve: an independent check whose cost falls with compute rather than with human
time. {{maturity:RESEARCH FRONTIER}}.

## 16. Connection to Previous Chapters

{{ch:as-failures}}'s correlation results are this chapter's engine, applied to the
tightest possible coupling: a generator and its own reviewer.

{{ch:aids-automl}}'s {{eq:search-must-be-scored-off-search}} reaches its limiting
case — a search that scores itself, where the holdout is not merely absent but
conceptually unavailable.

{{ch:mcp-production}}'s review-does-not-scale result returns as the capacity bound
on the only verification that works, and its commons argument returns unchanged.

{{ch:aids-agentic-eda}}'s boundary rule scales up: unverified output must not
circulate, and here it is harder because the output is well-formatted.

{{ch:as-specialized}}'s verifier ceiling explains why execution-graded work is the
defensible deployment — it is the one part of research with a strong verifier.

Ahead: {{ch:aids-oversight}} closes the part by asking where the remaining human
attention should go, using this part's measurements to place it.

## 17. Exercises

1. Derive $\partial A/\partial \rho$ from
   {{eq:self-judging-measures-correlation}} and confirm the true sound fraction is
   invariant.

2. Add execution grading to the first listing — a check independent of $\rho$ —
   and find how much independence it restores.

3. Model a judge fine-tuned on the generator's output, so $\rho$ rises over time.
   What does the reported metric do?

4. Compute the break-even generation volume against a fixed verification capacity.

5. Estimate the generator-judge correlation for a system you have access to, from
   error overlap on a labelled set.

6. Model the commons explicitly with several producers and find the equilibrium
   submission volume.

## 18. Interview Questions

1. A pipeline reports that $80\%$ of its output passes its own review. What have
   you learned?

2. The reviewer achieves near-human agreement on paper scores. Does that license
   the acceptance claim?

3. Would a panel of five reviewers fix it?

4. It costs \$15 per result. What does a usable result cost?

5. You increase throughput tenfold and quality falls. Why?

6. Where would you spend one expert-hour in an autonomous research pipeline?

## 19. Research Questions

1. What is the residual error correlation between current model families, and does
   cross-family review buy real independence?

2. Can generator-judge correlation be estimated without ground truth?

3. What fraction of research claims are mechanically checkable, and does restricting
   to them preserve the interesting ones?

4. Does a judge fine-tuned on its generator's output become measurably more
   permissive over time?

5. Is there any verification mechanism whose cost scales with compute rather than
   with human attention?

## 20. Chapter Summary

{{cite:lu2024aiscientist}} automated a whole research loop at under \$15 per paper,
and claimed output exceeding a conference acceptance threshold **as judged by its
own automated reviewer.** That clause is the measurement.

A generator and a judge from one model family share a base model, a prompt lineage
and a context, so the flaws the generator produces are drawn from the region the
judge cannot see. {{sec:9-practical-example}} finds the acceptance rate rising from
$32.7\%$ to $84.2\%$ with shared bias while the share that is genuinely sound stays
at $14.5\%$: **a self-judged pipeline's acceptance rate measures the correlation
between its generator and its judge**
({{eq:self-judging-measures-correlation}}). At identical nominal accuracy, an
independent judge delivered $44.0\%$ precision and a same-family one $17.2\%$.

Neither obvious fix works. Driving specificity from $70\%$ to $99\%$ bought $23$
points under high correlation against $57$ when independent
({{eq:accuracy-is-not-independence}}) — the failure is not an accuracy failure.
Seven correlated judges reached $20.4\%$ against one judge's $17.6\%$
({{eq:correlated-panels-are-one-judge}}). And a reviewer's agreement measured on
human-written papers does not transfer to its own family's papers, which is the
distribution that matters and the one nobody reports.

On economics, the volume argument is real — ten thousand generations delivered far
more sound results than ten — and it breaks on the filter. Generation costs \$15
and independent verification costs about **\$2,000 per usable result**
({{eq:generation-scales-verification-does-not}}); as generation cost tends to zero
the cost per usable result barely moves. With realistic capacity, volume overflows
the filter and precision converges to the unfiltered rate
({{eq:volume-overflows-the-filter}}), and submitted volume depletes a bounded pool
of reviewer attention that generation does not replenish
({{eq:attention-is-a-commons}}).

So the defensible deployment is narrow and genuinely valuable: **verify by
execution wherever the claim is executable**, use a different model family where it
is not, spend human attention on the residual, and never let unverified output
cross a boundary. The generation half is solved. The verification half is the
system.

## 21. Further Reading

{{cite:lu2024aiscientist}} should be read in full and read carefully — it is a
serious piece of engineering, its cost figure is real, and its evaluation clause is
the thing this chapter is about. The paper is explicit about the automated
reviewer; the discussion around it frequently is not.

{{cite:chan2024mlebench}} is the instructive contrast: Kaggle grades on a held-out
set the competitor never sees, so its $16.9\%$ medal rate is independent of the
agent that produced the submissions.

{{cite:testini2025dsautomation}} for the redesign gap that
{{sec:7-internal-mechanics}} concedes is real, and
{{cite:du2023debate}} for multi-judge aggregation whose gains
{{eq:correlated-panels-are-one-judge}} predicts will be small among same-family
judges.

{{ch:as-failures}} for the correlation machinery, and
{{ch:mcp-production}} for the commons argument this chapter re-derives.
