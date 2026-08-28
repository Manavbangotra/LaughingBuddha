---
id: ft-synthetic
number: 134
part: XIV
tier: full
status: draft
requires: [ft-datasets, ft-sft, fm-instruction-tuning, math-random-vars]
provides: [diversity-collapse, quality-filters-rareness, source-grounded-synthesis,
           systematic-versus-random-noise, error-taxonomy, external-oracle,
           self-eval-agreement]
citations: [wang2023selfinstruct, gunasekar2023, zhou2023lima, lee2022dedup,
            hinton2015]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish the **two** collapse
mechanisms in synthetic-data pipelines and say which one you actually have;
explain why a likelihood-based quality filter is a **rareness filter**, and why it
is more dangerous than the recursion it was installed to protect; ground
generation so synthetic examples inherit a real source's diversity rather than the
generator's; distinguish **random from systematic** label error, and explain why
an accuracy rate cannot; and design a validation that a shared misconception
cannot pass.

## 2. Why This Matters

{{cite:wang2023selfinstruct}} turned instruction data from a labelling problem
into a compute problem, and that change is real and enormous. This chapter is
about what it costs, and the measurements did not come out where the folklore
says they should.

**Recursion, on its own, was fine.** {{sec:9-practical-example}} runs eight
generations of a model trained on nothing but its own output, at 4,000 examples
per generation: **all 8 modes alive, spread 4.88 → 4.75, tail mass 20% → 19%.**
Self-training is not intrinsically degenerative. At 60 examples per generation it
*does* degrade — down to 6 modes — but slowly, noisily, and as a **sample-size**
problem you can buy your way out of.

**The quality filter is what collapsed it.** Same 4,000 examples, plus the
standard hygiene of keeping the highest-likelihood 70%: **8 modes → 1, spread 4.90
→ 0.06, and the tail gone after a single generation.**

**The filter did not malfunction — it did what it was asked.** A rare mode has low
likelihood *because* it is rare, so filtering for quality is filtering for
typicality with better branding. **The dangerous component in a synthetic-data
pipeline is not the recursion; it is the filter installed to make the recursion
safe.**

Then the second measurement, about what "95% accurate" hides.

**At 20% wrong labels, random corruption leaves the affected region at 0.776 and
systematic corruption leaves it at 0.393 — below chance.** The model has not
become uncertain; it has learned the generator's inverted rule and applies it
confidently.

**And a held-out synthetic test set certifies the mistake.** Under systematic
corruption, self-eval barely moves (**0.756 → 0.718** from 10% to 30%) while truth
in the affected region falls **0.600 → 0.328**. The evaluation cannot see the
error because it shares it.

{{maturity:MATURE}} Synthetic instruction data. {{maturity:EMERGING}} The measured
account of what filtering does to it. {{maturity:ESTABLISHED}} Grounding in real
source material.

## 3. Prerequisites

{{ch:ft-datasets}} is load-bearing — this chapter is that one's coverage argument
applied to data you *make* rather than select, and it inherits
{{eq:metric-inherits-bias}} in a sharper form. {{ch:ft-sft}} for the training
loop; {{ch:fm-instruction-tuning}} for instruction formats;
{{ch:math-random-vars}} for mixtures and tail mass.

## 4. Intuitive Explanation

### Two collapse mechanisms, and only one is famous

The story everyone tells is: a model trained on its own output degrades, each
generation a photocopy of a photocopy.

{{sec:9-practical-example}} says **that is true and it is the smaller problem.**

**Mechanism one — sampling error.** Rare modes are represented by few examples,
few examples are estimated badly, and a badly estimated mode attracts less mass
next round. The error compounds in one direction. **But it needs a small sample**:
at 4,000 examples per generation the distribution survives eight rounds intact,
and at 60 it degrades slowly and noisily — modes vanish and come back, because
survival is near a coin flip.

**Mechanism two — filtering.** Apply a quality filter and the collapse is
immediate at *any* sample size. Eight modes to one, and the tail gone after a
single generation.

> **These are different problems with different fixes.** Mechanism one is solved
> by generating more. Mechanism two is made *worse* by generating more, because
> every additional round applies the filter again.

### Why the filter is the villain

The logic is short and it is hard to escape:

1. Filters score samples by how good they look **to the model**.
2. A rare-but-correct example looks unlikely under the model.
3. So does a wrong example.
4. **The filter cannot tell them apart**, and it removes both.

Every pass concentrates mass on what the model already does confidently — which is
a precise description of the failure the filter was deployed to prevent.

**And this generalises well past a toy.** A reward model, an LLM-as-judge, a
perplexity threshold, a heuristic for well-formedness: each scores typicality
alongside quality, and inside the model's own distribution an unusual-but-correct
example and an unusual-but-wrong one are not distinguishable. **Filtering
synthetic data reliably improves the average example and reliably narrows the set,
and only the first half gets measured.**

### Grounding works, and it has a limit

Hold back 10% of each generation as real data and, with the identical aggressive
filter, **6 modes survive instead of 1.** At 30% real, **all 8.**

The reason to name precisely: **the real fraction is the only term in the
recursion that does not depend on the model's current beliefs.** It is therefore
the only term that can reintroduce a mode the model has already lost. Synthetic
data amplifies variety the generator has; it cannot create variety the generator
lacks.

**But read the tail before declaring victory.** True tail mass is 20%. With 10%
real it is **2%**; with 30% real, **6%**. **Grounding restores the modes and not
their mass**, because the filter deletes the tail every round and the real data
only refills part of it.

> **So this is a hierarchy, not a fix.** Grounding prevents collapse; it does not
> make aggressive filtering safe. If you need the tail — and
> {{ch:ft-datasets}} argued you do — **change the filter**, don't compensate for
> it.

### The other half: not all wrong examples are equally wrong

"Our synthetic data is 95% accurate" is the number every pipeline reports, and
alone it says almost nothing, because it does not say whether the 5% is scattered
or concentrated.

**Random errors are nearly harmless.** They disagree with each other and with the
surrounding data, so they raise the loss floor without moving the boundary much.

**Systematic errors are learnable.** A generator with a consistent misconception
produces consistently wrong labels in one region, and a model learns them because
learning consistent patterns is exactly what it is for.

The measurement: at 20% corruption, the affected region scores **0.776** under
random noise and **0.393** under systematic — **below chance.** Same error rate.

> **Ninety-five per cent accurate with the 5% scattered is a mild tax.
> Ninety-five per cent accurate with the 5% being every example of one subtopic is
> a model confidently wrong about that subtopic. The headline number is identical.**

### The evaluation shares the misconception

Hold out part of the synthetic data as a test set — the obvious move, and what
most pipelines do. That test set came from the same generator, carries the same
misconception, and **agrees with the model precisely where both are wrong.**

Under systematic corruption, self-eval moves **0.756 → 0.718** across a range
where the truth in the affected region falls **0.600 → 0.328**. Under *random*
corruption self-eval falls steeply (0.736 → 0.583), because random errors
disagree with each other and punish the model.

**Systematic errors agree, so a synthetic test set built from them rewards the
model for reproducing them.** This is {{ch:ft-datasets}}'s
{{eq:metric-inherits-bias}} in its worst form: there the eval inherited the
*selection*; here it inherits the *beliefs*, and a shared misconception does not
merely fail to detect the error — **it certifies it.**

## 5. Formal Explanation

### 5.1 The generative recursion

Let $p_0$ be the real distribution and $\hat{p}_t$ the model fitted at round $t$
to $n$ samples drawn from $\hat{p}_{t-1}$:

$$ \hat{p}_t = \mathcal{F}\big(x_1, \dots, x_n \sim \hat{p}_{t-1}\big) $$ (eq:collapse-recursion)

The estimation error at each round is $O(n^{-1/2})$ per parameter, but it is
**not** re-centred: round $t$ has no access to $p_0$, so errors accumulate as a
random walk with an absorbing boundary at "this mode has zero weight".

For a mixture component of weight $w$, the expected count is $nw$, and the
probability of losing it is roughly $(1-w)^n$ — **negligible for $nw \gg 1$ and
appreciable otherwise.** Hence: mechanism one is a sample-size effect.

### 5.2 Filtering, formally

A filter keeping the top $\kappa$ fraction by model likelihood induces

$$ q(x) \;\propto\; \hat{p}(x)\cdot \mathbb{1}\!\left[\hat{p}(x) \ge \tau_\kappa\right] $$ (eq:quality-filters-rareness)

**{{eq:quality-filters-rareness}} is a hard truncation of the low-density
region**, and low density is exactly where rare-but-valid examples live. The
recursion becomes

$$ \hat{p}_t = \mathcal{F}(x \sim q_{t-1}), \qquad \text{supp}(q_{t-1}) \subsetneq \text{supp}(\hat{p}_{t-1}) $$

so the support **strictly shrinks every round**, independent of $n$. That is why
the collapse is immediate at any sample size: it is a *deterministic* contraction,
not a stochastic drift.

### 5.3 Why grounding arrests it

Mix a fraction $\rho$ of real data:

$$ \hat{p}_t = \mathcal{F}\big((1-\rho)\, q_{t-1} + \rho\, p_0\big) $$ (eq:grounding-is-the-fix)

Now $\text{supp}(p_0) \subseteq \text{supp}$ of the training mixture for every
$t$, so **no mode can be permanently lost.** The fixed point satisfies

$$ p^{*} \approx \frac{\rho\, p_0}{\rho + (1-\rho)\,\mathbb{1}[\hat{p} \ge \tau]} $$

which restores the *support* but leaves tail **mass** suppressed by roughly
$\rho$ — matching the measured 2% at $\rho = 0.1$ and 6% at $\rho = 0.3$ against a
true 20%.

### 5.4 Random versus systematic error

For a corrupted label $\tilde{y}$, write the corruption as a channel. Random
noise:

$$ P(\tilde{y} \ne y \mid x) = \epsilon \quad\text{independent of } x $$

The Bayes-optimal predictor under this channel is **unchanged** — the noise is
symmetric and $\arg\max_y P(y \mid x)$ survives. Systematic noise:

$$ P(\tilde{y} \ne y \mid x) = \mathbb{1}[x \in R] $$

for a region $R$. Now

$$ \arg\max_y P(\tilde{y} \mid x) = 1 - y^{*}(x) \quad \text{for } x \in R $$ (eq:systematic-noise-is-learnable)

**{{eq:systematic-noise-is-learnable}} says the optimal predictor on corrupted
data is *inverted* on $R$**, which is why the measured accuracy there is below
chance rather than merely degraded. A model that fits the training distribution
well is *required* to be wrong there.

### 5.5 Why self-evaluation certifies it

If the eval labels come from the same channel,

$$ \mathcal{A}_{\text{self}} = P\big(\hat{y}(x) = \tilde{y}(x)\big) $$

Under systematic corruption both $\hat{y}$ and $\tilde{y}$ are inverted on $R$, so
they **agree**:

$$ \mathcal{A}_{\text{self}} \approx \mathcal{A}_{\text{clean-on-}R^{c}} \quad\text{regardless of the error on } R $$ (eq:self-eval-agreement)

Under random corruption the two disagree at rate $\approx 2\epsilon(1-\epsilon)$,
so self-eval **falls**. **{{eq:self-eval-agreement}} is the formal statement that
random noise is detectable from inside and systematic noise is not.**

> **IMPORTANT:** This makes error *rate* the wrong summary statistic and error
> *concentration* the right one. Two datasets with identical accuracy differ in
> everything that matters, and no amount of additional evaluation on
> generator-produced data separates them.

## 6. Mathematical Foundation

### 6.1 Mode survival, quantified

For component weight $w$ and $n$ samples, $P(\text{lost}) = (1-w)^n$. With
$w = 0.02$: at $n = 4000$, $(0.98)^{4000} \approx 10^{-35}$; at $n = 60$,
$(0.98)^{60} \approx 0.30$.

**That factor of $10^{34}$ between the two rows of the measurement is the whole
explanation for why one survived and one drifted** — and it is why "collapse" is a
statement about $n$, not about self-training.

### 6.2 The filter's contraction rate

Keeping the top $\kappa$ by density removes the $(1-\kappa)$ tail, so after $T$
rounds the surviving support has measure

$$ \mu_T \le \kappa^{T}\mu_0 $$ (eq:filter-contraction)

At $\kappa = 0.7$, $T = 8$: $0.7^{8} \approx 0.058$. **{{eq:filter-contraction}}
predicts a support shrunk to about a twentieth**, and the measured spread went
4.90 → 0.06, a factor of 82. The mechanism is right and the toy over-shoots it
because refitting a mixture to a truncated sample also *re-centres* the
components, compounding the truncation.

### 6.3 What a small oracle can detect

To distinguish concentrated from scattered errors you do not need to estimate the
rate — you need to reject independence. With $m$ sampled errors falling into $k$
categories, a concentration test has power $\approx 1$ against "all errors in one
category" for surprisingly small $m$:

$$ P(\text{all } m \text{ in one category} \mid \text{scattered}) = k^{-(m-1)} $$ (eq:concentration-test)

At $k = 20$, $m = 5$: $20^{-4} \approx 6\times10^{-6}$. **{{eq:concentration-test}}
is why "sample thirty failures and read them" is a sufficient protocol** and a
larger clean eval set is not a substitute.

> **MATH NOTE:** {{eq:self-eval-agreement}} assumes the eval channel is *identical*
> to the training channel. In practice a second sampling temperature or a
> different prompt makes it merely *correlated*, which weakens the agreement
> without removing it. **Partial independence gives partial detection**, which is
> why "we used a different prompt for the eval set" is a real improvement and not
> a solution.

## 7. Internal Mechanics

```mermaid {#fig:synth-loop caption="The synthetic-data loop, with the two failure points marked. The recursion (left) degrades only when the sample per round is small. The filter (centre) contracts the support deterministically at any sample size (eq:filter-contraction), and the self-evaluation (right) cannot detect the result because it is drawn through the same generator (eq:self-eval-agreement). Grounding (bottom) is the only arrow carrying information the generator does not already have."}
flowchart LR
    GEN["generator<br/>(the model)"] --> SAMP["sample n<br/>examples"]
    SAMP --> FILT{{"quality filter<br/>DANGER: removes<br/>rare, not wrong"}}
    FILT --> TRAIN["train next<br/>generation"]
    TRAIN --> GEN
    SAMP --> EVAL["held-out<br/>synthetic eval"]
    EVAL -->|"shares the<br/>misconception"| CERT["certifies<br/>the error"]
    REAL[("real source<br/>material")] -->|"grounding:<br/>the only<br/>external term"| SAMP
    REAL --> ORACLE["external oracle<br/>(execution, lookup,<br/>small human set)"]
    ORACLE --> CERT
```

### 7.1 Filter for correctness, not for typicality

The distinction that makes the difference:

| Filter | Scores | Effect on the tail |
|---|---|---|
| model likelihood / perplexity | typicality | **deletes it** |
| LLM-as-judge, unanchored | typicality + quality | **deletes most of it** |
| reward model | typicality + preference | **deletes it, with extra steps** |
| execution / unit test | correctness | neutral |
| lookup against the source document | correctness | neutral |
| schema or format validation | validity | neutral |

**Everything in the top half is scored by the generator's own distribution;
everything in the bottom half is scored by something outside it.** The bottom half
is the only kind that removes wrong examples without also removing rare ones.

### 7.2 Grounded generation, concretely

Generate **from** something rather than from the prior: a source document, a log
line, a database row, a schema, a real ticket, a real customer configuration.
Three things follow immediately:

1. The synthetic examples inherit **that material's** diversity.
2. Each example has a **checkable source**, so correctness filtering becomes
   possible.
3. Coverage becomes an **allocation** problem over source material, which is
   {{ch:ft-datasets}}'s stratification applied one level up.

**Self-instruction with no external anchor is the configuration that collapses,
and it is also the cheapest to build** — which is why it is common.

### 7.3 The validation protocol

1. **Sample thirty failures and read them.** {{eq:concentration-test}} makes this
   sufficient for detecting concentration.
2. **Cluster them.** One subtopic is a different dataset from thirty subtopics at
   the same accuracy.
3. **Validate against a non-generator oracle**, however small.
4. **Never report a held-out synthetic score as a quality claim.**
5. **Measure diversity, not just accuracy** — distinct n-grams, distinct source
   documents, distinct intents per the taxonomy.

## 8. Implementation

```python {tier=A name=diversity-collapse}
"""Diversity collapse, measured -- and why quality filtering accelerates it.

cite:wang2023selfinstruct made instruction data a compute problem, and the
compute is real. The hazard is that a model generating its own training data
samples from its own distribution, so each generation inherits the previous
generation's modes and, crucially, its SAMPLING ERROR in the tail
(eq:collapse-recursion).

This listing runs the recursion. A true distribution has eight modes with
Zipf-like weights; a model is fitted, sampled, refitted on its own samples, and so
on. Three regimes are compared: plain resampling, resampling with a quality filter
that keeps the highest-likelihood samples, and resampling with a small share of
real data mixed back in.

The middle regime is the one worth the listing. Filtering for quality is the
standard remedy for synthetic-data noise, and it is applied to a distribution
whose problem is that the tail is disappearing.
"""
import numpy as np

rng = np.random.default_rng(173)

K = 8
MEANS = np.linspace(-9.0, 9.0, K)
SIG = 0.55
W_TRUE = np.array([0.30, 0.22, 0.16, 0.12, 0.08, 0.06, 0.04, 0.02])
N_GEN = 4000
GENERATIONS = 8


def sample_true(n):
    c = rng.choice(K, size=n, p=W_TRUE)
    return MEANS[c] + SIG * rng.normal(size=n)


def em_fit(x, k=K, iters=60):
    """Fit a k-component 1-D Gaussian mixture. Components that stop attracting
    mass are what 'losing a mode' looks like mechanically."""
    w = np.full(k, 1.0 / k)
    mu = np.quantile(x, np.linspace(0.05, 0.95, k))
    var = np.full(k, x.var() / k + 1e-3)
    for _ in range(iters):
        d = x[:, None] - mu[None, :]
        logp = (-0.5 * d ** 2 / var[None, :] - 0.5 * np.log(2 * np.pi * var)
                [None, :] + np.log(np.maximum(w, 1e-300))[None, :])
        m = logp.max(axis=1, keepdims=True)
        r = np.exp(logp - m); r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-10
        w = nk / len(x)
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = np.maximum((r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk,
                         1e-3)
    return w, mu, var


def sample_model(model, n):
    w, mu, var = model
    c = rng.choice(len(w), size=n, p=w / w.sum())
    return mu[c] + np.sqrt(var[c]) * rng.normal(size=n)


def logpdf(model, x):
    w, mu, var = model
    d = x[:, None] - mu[None, :]
    lp = (-0.5 * d ** 2 / var[None, :] - 0.5 * np.log(2 * np.pi * var)[None, :]
          + np.log(np.maximum(w, 1e-300))[None, :])
    m = lp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))


def modes_alive(x, tol=2.0, floor=0.005):
    """A true mode counts as alive if at least `floor` of the sample lands
    within `tol` standard deviations of it."""
    hits = np.abs(x[:, None] - MEANS[None, :]) < tol * SIG
    return int((hits.mean(axis=0) >= floor).sum())


def tail_mass(x):
    """Share of samples belonging to the four RAREST true modes -- 20% of the
    true distribution, and the part that vanishes first."""
    near = np.abs(x[:, None] - MEANS[None, :]).argmin(axis=1)
    return float(np.isin(near, np.arange(K // 2, K)).mean())


def run(real_share=0.0, keep=1.0, n_gen=N_GEN):
    x = sample_true(n_gen)
    out = [(modes_alive(x), float(x.std()), tail_mass(x))]
    for _ in range(GENERATIONS):
        model = em_fit(x)
        n_syn = int(n_gen * (1 - real_share))
        draw = sample_model(model, int(n_syn / keep))
        if keep < 1.0:                       # keep the highest-likelihood share
            thr = np.quantile(logpdf(model, draw), 1 - keep)
            draw = draw[logpdf(model, draw) >= thr][:n_syn]
        x = draw if real_share == 0 else np.concatenate(
            [draw[:n_syn], sample_true(n_gen - n_syn)])
        out.append((modes_alive(x), float(x.std()), tail_mass(x)))
    return out


REGIMES = [
    ("pure synthetic, 4,000 per generation", dict()),
    ("pure synthetic, 60 per generation", dict(n_gen=60)),
    ("filtered for quality: keep top 70%", dict(keep=0.7)),
    ("filtered for quality: keep top 40%", dict(keep=0.4)),
    ("filtered top 70%, but 10% real data mixed in",
     dict(keep=0.7, real_share=0.10)),
    ("filtered top 70%, but 30% real data mixed in",
     dict(keep=0.7, real_share=0.30)),
]

print(f"True distribution: {K} modes, weights {W_TRUE.min():.0%} to "
      f"{W_TRUE.max():.0%}. {N_GEN:,} examples per generation.\n")

results = {}
for name, kw in REGIMES:
    r = run(**kw)
    results[name] = r
    print(f"{name}")
    print(f"  {'generation':>11}" + "".join(f"{g:>6}" for g in
                                            range(GENERATIONS + 1)))
    print(f"  {'modes alive':>11}" + "".join(f"{v[0]:>6}" for v in r))
    print(f"  {'std':>11}" + "".join(f"{v[1]:>6.2f}" for v in r))
    print(f"  {'tail mass':>11}" + "".join(f"{v[2]:>6.0%}" for v in r))
    print()

pure = results["pure synthetic, 4,000 per generation"]
small = results["pure synthetic, 60 per generation"]
f70 = results["filtered for quality: keep top 70%"]
f40 = results["filtered for quality: keep top 40%"]
mix = results["filtered top 70%, but 10% real data mixed in"]
mix30 = results["filtered top 70%, but 30% real data mixed in"]

print(f"""
The first block is not what the folklore predicts, and it is worth sitting with.
Eight generations of a model trained on nothing but its own output, and the
distribution is essentially intact: {pure[-1][0]} of {pure[0][0]} modes alive,
spread {pure[0][1]:.2f} to {pure[-1][1]:.2f}, tail mass {pure[0][2]:.0%} to
{pure[-1][2]:.0%}.

At {N_GEN:,} examples per generation the sampling error is small enough that each
refit recovers the distribution it was drawn from, so the recursion has nothing
to compound. Self-training is not intrinsically degenerative.

The second block shows the mechanism that IS intrinsic, at a sample size small
enough to see it. At 60 examples per generation the same procedure drifts down to
{small[-1][0]} modes, and note how NOISY that column is -- modes disappear and
come back, because at this sample size a rare mode's survival is close to a coin
flip each round (eq:collapse-recursion). This is real degradation, it is slow,
and it is a sample-size problem, which means it is the one you can buy your way
out of.

Now the blocks that collapse at {N_GEN:,} examples, where pure recursion did not.

Quality filtering is the standard hygiene for synthetic data, and the standard
implementation keeps the samples the model scores highest. At keep-70%, eight
generations leave {f70[-1][0]} mode alive with a spread of {f70[-1][1]:.2f}. At
keep-40%, {f40[-1][0]} mode and {f40[-1][1]:.2f}. The distribution has become a
point.

Look at how fast the tail goes: {f70[1][2]:.0%} after ONE generation in both
filtered regimes, while pure recursion still had {pure[-1][2]:.0%} of it after
eight.

The filter did not malfunction. It did exactly what it was asked, and what it was
asked was to delete the tail. A rare mode has low likelihood BECAUSE it is rare,
so a likelihood-based quality filter is a rareness filter wearing different
clothing (eq:quality-filters-rareness). Each pass concentrates mass on what the
model already does confidently, which is a precise description of the failure the
filter was installed to prevent.

That is the finding to carry out of this listing, and it inverts the usual
advice. In a synthetic-data pipeline the dangerous component is not the
recursion. It is the filter placed there to make the recursion safe.

The generalisation past this toy is direct. A reward model, an LLM-as-judge, a
perplexity threshold, a heuristic for well-formedness -- each scores typicality
alongside quality and cannot separate the two, because inside the model's own
distribution an unusual-but-correct example and an unusual-but-wrong one look
alike. Filtering synthetic data for quality reliably raises the average example
and reliably narrows the set, and only the first half is usually measured.

The last two blocks are the fix, and they are more interesting than a fix usually
is, because they show its limit.

Keep the identical keep-70% filter and hold back 10% of each generation as real
data: {mix[-1][0]} modes survive instead of {f70[-1][0]}, and the spread holds at
{mix[-1][1]:.2f} instead of {f70[-1][1]:.2f}. At 30% real, all {mix30[-1][0]}
modes survive. The real fraction is the only term in the recursion that does not
depend on the model's current beliefs, so it is the only term that can
reintroduce a mode the model has already lost (eq:grounding-is-the-fix).

But read the tail row before declaring victory. True tail mass is
{W_TRUE[K//2:].sum():.0%}. With 10% real it sits at {mix[-1][2]:.0%}; with 30%
real, {mix30[-1][2]:.0%}. Grounding restores the MODES and does not restore their
MASS, because the filter is still deleting the tail every round and the real data
is only refilling a fraction of it.

So the honest summary is a hierarchy rather than a fix. Grounding prevents
collapse; it does not make aggressive filtering safe. If you need the tail -- and
the previous chapter's coverage argument says you do -- the filter is the thing to
change, not the amount of real data used to compensate for it.

Which gives the rule for building these pipelines. Generate FROM something rather
than from the model's prior: a document, a log, a schema, a customer record, a
real ticket. The synthetic examples then inherit that material's diversity instead
of the model's. And filter for CORRECTNESS against that source rather than for
typicality under the generator, because those two are the same operation only when
the generator is already right about everything -- in which case there was nothing
to fix.""")
```

The first listing is about what the pipeline loses. The second is about what it
adds, and why an error rate cannot describe it.

```python {tier=A name=systematic-versus-random-error}
"""Not all wrong examples are equally wrong.

"Our synthetic data is 95% accurate" is the number every synthetic-data pipeline
reports, and on its own it means very little, because it does not say whether the
5% is scattered or concentrated.

Random errors are close to harmless: they pull in inconsistent directions and a
learner averages over them. Systematic errors are not, because they are LEARNABLE
-- a generator with a consistent misconception produces consistently wrong labels
in one region, and a model trained on them learns the misconception as if it were
the task (eq:systematic-noise-is-learnable).

Synthetic data produces the second kind by construction. This listing measures the
gap, and then measures whether you could detect it from inside the pipeline.
"""
import numpy as np

rng = np.random.default_rng(179)

D, NF = 12, 700
N_TRAIN, N_TEST = 6000, 6000

W_TRUE = rng.normal(size=D)
V_BAD = rng.normal(size=D)               # the direction the generator is wrong in
V_BAD /= np.linalg.norm(V_BAD)


def label(X):
    return (np.sin(1.5 * X @ W_TRUE / np.sqrt(D)) + 0.6 * X[:, 0] > 0).astype(int)


W_RF = rng.normal(size=(D, NF)) * 0.8
B_RF = rng.uniform(0, 2 * np.pi, NF)


def fit(X, y, lam=1e-3):
    P = np.cos(X @ W_RF + B_RF)
    return np.linalg.solve(P.T @ P + lam * len(X) * np.eye(NF),
                           P.T @ (2.0 * y - 1))


def pred(c, X):
    return (np.cos(X @ W_RF + B_RF) @ c > 0).astype(int)


def corrupt_random(X, y, rate):
    y = y.copy()
    idx = rng.permutation(len(y))[:int(rate * len(y))]
    y[idx] = 1 - y[idx]
    return y, np.zeros(len(y), bool)


def corrupt_systematic(X, y, rate):
    """The generator is confidently wrong in one contiguous region -- which is
    what a consistent misconception looks like in the data."""
    y = y.copy()
    s = X @ V_BAD
    idx = np.argsort(-s)[:int(rate * len(y))]
    y[idx] = 1 - y[idx]
    region = np.zeros(len(y), bool); region[idx] = True
    return y, region


X_tr = rng.normal(size=(N_TRAIN, D)); y_tr = label(X_tr)
X_te = rng.normal(size=(N_TEST, D));  y_te = label(X_te)
s_te = X_te @ V_BAD

print(f"{N_TRAIN:,} training examples, {N_TEST:,} clean test examples.\n")
print(f"{'wrong':>7}{'':>3}" + f"{'RANDOM errors':>28}" + f"{'SYSTEMATIC errors':>32}")
print(f"{'labels':>7}{'':>3}{'overall':>10}{'in bad':>9}{'self-':>9}"
      f"{'overall':>11}{'in bad':>10}{'self-':>11}")
print(f"{'':>7}{'':>3}{'':>10}{'region':>9}{'eval':>9}{'':>11}{'region':>10}"
      f"{'eval':>11}")
print("-" * 70)

rows = {}
clean_c = fit(X_tr, y_tr)
base = float((pred(clean_c, X_te) == y_te).mean())

for rate in (0.0, 0.05, 0.10, 0.20, 0.30):
    out = []
    for fn in (corrupt_random, corrupt_systematic):
        y_bad, _ = fn(X_tr, y_tr, rate)
        c = fit(X_tr, y_bad)
        p = pred(c, X_te)
        # The region the systematic generator is wrong about, on the test set.
        cut = np.quantile(s_te, 1 - rate) if rate > 0 else np.inf
        bad_region = s_te >= cut
        overall = float((p == y_te).mean())
        in_bad = float((p == y_te)[bad_region].mean()) if bad_region.any() \
            else float("nan")
        # Self-eval: score against labels the SAME generator would produce.
        y_self, _ = fn(X_te, y_te, rate)
        self_eval = float((p == y_self).mean())
        out.append((overall, in_bad, self_eval))
    rows[rate] = out
    r, sy = out
    def f(v, w):
        return f"{'--':>{w}}" if np.isnan(v) else f"{v:>{w}.3f}"
    print(f"{rate:>7.0%}{'':>3}{r[0]:>10.3f}{f(r[1], 9)}{r[2]:>9.3f}"
          f"{sy[0]:>11.3f}{f(sy[1], 10)}{sy[2]:>11.3f}")

r10, s10 = rows[0.10]
r20, s20 = rows[0.20]
r30, s30 = rows[0.30]
print(f"""
Read across the 10% row. The same fraction of the training labels is wrong in
both halves of the table. Scattered at random, clean-test accuracy is
{r10[0]:.3f}; concentrated in one region, it is {s10[0]:.3f}. And at 30% wrong
labels, random still holds {r30[0]:.3f} while systematic has fallen to
{s30[0]:.3f}.

A learner treats random label noise as noise: the flipped examples disagree with
each other and with the surrounding data, so they raise the loss floor without
moving the decision boundary much. Systematic errors are not noise. They are a
consistent signal about a region, indistinguishable from the truth by anything
inside the training set, and the model learns them because learning them is
exactly what it is for (eq:systematic-noise-is-learnable).

The bad-region column shows where the damage lands, and it is worse than the
overall numbers suggest. At 10% systematic corruption the model scores
{s10[1]:.3f} inside the affected region against {s10[0]:.3f} overall. At 20% it
scores {s20[1]:.3f} -- BELOW CHANCE. The model has not become uncertain about
that region; it has learned the generator's inverted rule and applies it
confidently. Random corruption at the same rate leaves the same region at
{r20[1]:.3f}, barely distinguishable from everywhere else.

This is why "our synthetic data is 95% accurate" is not a useful number. Ninety-
five per cent accurate with the 5% scattered is a mild tax. Ninety-five per cent
accurate with the 5% being every example of one subtopic your generator
misunderstands is a model that is confidently wrong about that subtopic, and the
aggregate accuracy figure is identical in both cases.

Now the self-eval column, which is why this survives review.

Suppose you hold out part of the synthetic data as a test set -- the obvious thing
to do, and what most pipelines do. That test set was produced by the same
generator, so it carries the same misconception, and it AGREES with the model
about the region where both are wrong. At 20% systematic corruption the model
scores {s20[2]:.3f} against generator-produced labels while scoring {s20[1]:.3f}
against the truth in the affected region -- and note the direction of travel:
between 10% and 30% corruption the self-eval score barely moves
({s10[2]:.3f} to {s30[2]:.3f}) while the truth in that region falls from
{s10[1]:.3f} to {s30[1]:.3f}. The held-out synthetic score reports a healthy
model throughout.

Contrast the random column, where self-eval FALLS steeply ({r10[2]:.3f} to
{r30[2]:.3f}). Random errors disagree with each other, so a synthetic test set
built from them punishes the model. Systematic errors agree, so a synthetic test
set built from them rewards it.

The evaluation cannot see the error because it shares it. This is the previous
chapter's eq:metric-inherits-bias in its sharpest form: there, the eval set
inherited the training set's SELECTION; here it inherits the generator's BELIEFS,
which is worse, because a selection bias leaves the missing data missing while a
shared misconception actively certifies the mistake.

Three practical consequences, and none of them is "measure accuracy more
carefully".

Report an error TAXONOMY, not an error rate. Sample the failures, cluster them,
and ask whether they concentrate. A hundred sampled errors that all concern the
same subtopic is a different dataset from a hundred that concern a hundred
subtopics, at identical accuracy.

Validate against something the generator did not produce. A small
human-labelled set, an execution check, a database lookup, a unit test -- any
external oracle breaks the agreement, and none of them needs to be large, because
you are detecting concentration rather than estimating a rate.

And ground the generation, per the previous listing. A generator writing from a
real source document can be checked against that document; a generator writing
from its prior can only be checked against itself.""")
```

## 9. Practical Example

**Recursion alone did not collapse.** Eight generations of self-training at 4,000
examples per round: **8/8 modes alive, spread 4.88 → 4.75, tail 20% → 19%.** At 60
per round it drifts to **6 modes**, noisily — modes vanish and return, because
{{eq:collapse-recursion}} makes survival near a coin flip when $nw \approx 1$.
**Mechanism one is a sample-size problem** ({{sec:6-mathematical-foundation}}:
$P(\text{lost}) = 10^{-35}$ at $n=4000$ against $0.30$ at $n=60$).

**The quality filter collapsed it immediately.** Same 4,000 examples, keep-70%:
**8 modes → 1, spread 4.90 → 0.06, tail gone after one generation.** At keep-40%,
spread **0.01** — the distribution became a point.

**The filter did what it was asked.** {{eq:quality-filters-rareness}} truncates
the low-density region, and rare-but-valid examples live there.
{{eq:filter-contraction}} predicts a support shrunk to ~6% after eight rounds at
$\kappa=0.7$; the measured spread fell by a factor of 82. **This is deterministic
contraction, not stochastic drift, which is why sample size does not help.**

> **IMPORTANT:** The dangerous component is not the recursion. **It is the filter
> installed to make the recursion safe** — and every practical filter (reward
> model, LLM judge, perplexity threshold) scores typicality alongside quality and
> cannot separate them.

**Grounding arrests it, up to a point.** With the identical keep-70% filter: 10%
real data holds **6 modes**, 30% real holds **all 8**
({{eq:grounding-is-the-fix}}). **But tail mass is 2% and 6% against a true 20%** —
grounding restores the *modes* and not their *mass*. **A hierarchy, not a fix: if
you need the tail, change the filter.**

**And error rate is the wrong statistic.** At 20% wrong labels, the affected region
scores **0.776** under random corruption and **0.393** under systematic — **below
chance**, because {{eq:systematic-noise-is-learnable}} makes the optimal predictor
on corrupted data *inverted* there. At 30%: **0.691 against 0.328.**

**The self-eval column is why this survives review.** Under systematic corruption
self-eval moves **0.756 → 0.718** from 10% to 30% while truth in the region falls
**0.600 → 0.328**. Under random corruption self-eval falls **0.736 → 0.583**.

**Random errors disagree with each other, so a synthetic test set punishes the
model; systematic errors agree, so it rewards the model for reproducing them**
({{eq:self-eval-agreement}}). The evaluation cannot see the error because it
shares it.

## 10. Production Considerations

**Ground every generator in real source material.** It is the only term in
{{eq:grounding-is-the-fix}} that carries information the model lacks.

**Filter for correctness, never for typicality.** Execution, lookup, schema
validation. If your only filter is a judge or a likelihood, expect
{{eq:filter-contraction}}.

**Measure diversity every round**, not only at the end — distinct sources,
distinct intents, tail mass against your taxonomy.

**Sample and read thirty failures.** {{eq:concentration-test}} makes it enough.

**Keep a small external oracle**, however small. Detecting concentration needs far
less data than estimating a rate.

**Never treat a held-out synthetic score as a quality claim.**

**Vary the eval channel** if you cannot escape it — different prompt, different
temperature, different model. Partial independence gives partial detection.

**Report an error taxonomy alongside the error rate.**

## 11. Common Mistakes

**Reporting "95% accurate"** with no statement about concentration.

**Filtering by model likelihood or judge score** and calling it quality control.

**Holding out synthetic data as the test set.**

**Generating from the prior** rather than from source material.

**Concluding self-training is safe** from a large-$n$ experiment, or **unsafe**
from a small-$n$ one — both mechanisms exist and they are different.

**Adding real data to compensate for an aggressive filter** rather than fixing the
filter.

**Measuring accuracy per round and not diversity.**

**Assuming a stronger judge fixes it** — a stronger judge is a sharper typicality
filter.

## 12. Failure Modes

**Model confidently wrong about one subtopic.** Cause:
{{eq:systematic-noise-is-learnable}}. Fix: external oracle, error taxonomy.

**Synthetic eval says fine, production says otherwise.** Cause:
{{eq:self-eval-agreement}}.

**Outputs become homogeneous over successive data generations.** Cause:
{{eq:filter-contraction}}, not the recursion.

**Tail behaviour disappears while aggregate metrics improve.** Cause: the filter,
plus {{ch:ft-datasets}}'s macro/aggregate split.

**More synthetic data makes things worse.** Cause: every round reapplies the
filter.

**A stronger judge makes diversity worse.** Cause: it is better at detecting
atypicality, which is the wrong target.

## 13. Alternatives

| Alternative | Trades | When |
|---|---|---|
| human annotation | cost, throughput | small, high-stakes sets |
| grounded synthetic | source-material coverage | the default here |
| ungrounded self-instruct | diversity | prototypes only |
| distillation from a stronger model ({{cite:hinton2015}}) | licence, ceiling | the teacher is genuinely better |
| execution-verified generation | applies only to checkable tasks | code, SQL, maths |
| quality filtering by judge | the tail | almost never as the only filter |
| augmentation of real data | novelty | when real data exists but is thin |

**The last row is under-used relative to full generation.** Paraphrasing and
recombining *real* examples inherits real diversity by construction and cannot
invent a misconception, which is a strictly better failure profile than generating
from the prior — it simply produces less novelty, which is often not the binding
constraint.

## 14. Evaluation

**Report diversity metrics alongside accuracy** — distinct sources, distinct
intents, tail mass.

**Report the error taxonomy**, not just the rate.

**State the oracle.** "Validated by GPT-4" and "validated by execution" are not
the same claim.

**Never report held-out synthetic accuracy** without saying it is generator-scored.

**Report the grounding ratio** $\rho$ and the filter's $\kappa$ — they are the two
parameters {{eq:grounding-is-the-fix}} and {{eq:filter-contraction}} say determine
the outcome.

**Evaluate on real data**, held out by group per {{ch:ft-datasets}}.

## 15. Advanced Concepts

**Filtering is selection, and inherits every hazard of it.**
{{maturity:MATURE}} {{cite:gunasekar2023}} showed aggressive quality filtering
can beat scale, which is true and is the strongest case for it. This chapter's
caution is compatible: filtering *real* data selects among genuine examples, while
filtering *generated* data selects within the generator's own distribution, and
{{eq:quality-filters-rareness}} bites only in the second case.

**Distillation is grounded generation with a better source.**
{{maturity:ESTABLISHED}} {{cite:hinton2015}}'s teacher is an external
distribution, so the student inherits *its* diversity rather than its own. The
collapse argument applies when teacher and student converge — self-distillation
across rounds is the ungrounded case wearing a respectable name.

**Execution as the ideal oracle.** {{maturity:MATURE}} For code, SQL and maths,
correctness is checkable without a model, which removes
{{eq:self-eval-agreement}} entirely. **This is why synthetic data works best in
exactly the domains where it is least needed**, and the honest generalisation to
open-ended text is unproven.

**A thousand examples, again.** {{maturity:MATURE}} {{cite:zhou2023lima}} suggests
the quantity synthetic data provides is rarely the binding constraint. **Given
that, generating 100,000 examples with a typicality filter is trading the thing
that matters for the thing that does not.**

**Diversity as a first-class training objective.**
{{maturity:RESEARCH FRONTIER}} Nothing in a standard pipeline optimises for
coverage; accuracy is measured every round and diversity is measured never.
Making {{eq:filter-contraction}}'s $\mu_T$ an explicit constraint is
straightforward and rare.

## 16. Connection to Previous Chapters

{{ch:ft-datasets}} is the direct parent: its coverage argument says the tail
matters, and this chapter shows the standard synthetic pipeline deletes exactly
that. {{eq:self-eval-agreement}} is {{eq:metric-inherits-bias}} escalated from
inherited *selection* to inherited *beliefs*. {{ch:ft-sft}} consumes whatever this
produces and cannot detect a systematic error in it.
{{ch:fm-instruction-tuning}}'s template diversity is a special case of the
diversity this chapter measures. {{ch:math-random-vars}} supplies the mixture
machinery.
Forward: {{ch:ft-preference}} generates preference data and inherits every hazard
here, with the added difficulty that preferences have no execution oracle;
{{part:25}} owns evaluation, and {{eq:self-eval-agreement}} is the reason its
infrastructure cannot be model-scored all the way down.

## 17. Exercises

1. Compute $P(\text{mode lost})$ from {{sec:6-mathematical-foundation}} for
   $w = 0.05$ at $n = 100, 500, 2000$.
2. Use {{eq:filter-contraction}} to predict the surviving support at
   $\kappa = 0.9$ after 20 rounds. Is a gentle filter safe if applied often?
3. In `diversity-collapse`, set `keep=0.9`. How many generations before the tail
   is gone, and does {{eq:filter-contraction}} predict it?
4. Modify the same listing so the filter scores *correctness* (distance to the
   nearest true mode) rather than likelihood. What happens, and why?
5. Derive {{eq:systematic-noise-is-learnable}} and explain why the region accuracy
   goes below 0.5 rather than to 0.5.
6. In `systematic-versus-random-error`, make the corrupted region smaller but
   fully corrupted. How does overall accuracy compare with the concentration of
   the damage?
7. Design a validation protocol for a synthetic dataset in a domain with **no**
   execution oracle. What is the minimum external labelling you need, and why is
   it smaller than a rate estimate?
8. Given a synthetic dataset you have, sample thirty errors, cluster them, and
   report whether they concentrate.

## 18. Interview Questions

1. Does training a model on its own output necessarily degrade it?
2. What are the two collapse mechanisms and how do you tell which you have?
3. Why is a quality filter more dangerous than the recursion?
4. Why can't a reward model filter be fixed by making it stronger?
5. What does grounding actually fix, and what does it not?
6. Your synthetic data is 95% accurate. What else do you need to know?
7. Why can a model score *below chance* on a region after training on
   systematically corrupted data?
8. Why does a held-out synthetic test set fail to detect systematic error?
9. Why does it detect random error?
10. How many failures do you need to read to detect concentration, and why so few?

## 19. Research Questions

1. {{eq:filter-contraction}} is a worst case. What is the actual contraction rate
   for real reward models on real generations, and how does it vary with judge
   strength?
2. Can a filter be built that scores correctness without scoring typicality, in a
   domain with no execution oracle?
3. {{eq:grounding-is-the-fix}} restores support but not mass. What $\rho$ restores
   the *tail distribution*, and does it depend on the filter's $\kappa$ in the way
   the fixed-point expression suggests?
4. {{eq:self-eval-agreement}} weakens under partial independence. How much
   independence does varying the prompt or temperature actually buy?
5. Is there a diagnostic for systematic generator error that uses only the
   generated data — for example, the geometry of its errors under perturbation?

## 20. Chapter Summary

**Two collapse mechanisms, and the famous one is the smaller.** Pure recursion at
4,000 examples per generation survived eight rounds intact (**8/8 modes, tail 20%
→ 19%**); at 60 it drifted to 6, slowly and noisily. That is
{{eq:collapse-recursion}}, a **sample-size** problem — $P(\text{lost})$ is
$10^{-35}$ against $0.30$.

**The quality filter collapsed it at any sample size**: keep-70% took **8 modes to
1 and spread 4.90 → 0.06, with the tail gone after one generation.**
{{eq:quality-filters-rareness}} truncates the low-density region and
{{eq:filter-contraction}} makes the contraction *deterministic*, which is why more
data does not help. **A rare mode has low likelihood because it is rare — every
practical filter scores typicality alongside quality and cannot separate them.**

**Grounding arrests it and does not undo it.** 10% real data holds 6 modes, 30%
holds all 8 ({{eq:grounding-is-the-fix}}) — **but tail mass is 2% and 6% against a
true 20%.** Modes, not mass. **If you need the tail, change the filter rather than
compensating for it.**

**And an error rate cannot describe a synthetic dataset.** At 20% corruption the
affected region scores **0.776** random against **0.393** systematic — below
chance, because {{eq:systematic-noise-is-learnable}} makes the fitted predictor
*inverted* there. Same headline accuracy, entirely different model.

**The held-out synthetic score certifies the mistake.** Self-eval moved **0.756 →
0.718** while region truth fell **0.600 → 0.328**; under random corruption
self-eval fell **0.736 → 0.583** instead. {{eq:self-eval-agreement}}: **random
errors disagree and are detectable from inside; systematic errors agree and are
not.**

Which yields the chapter's rules, and none of them is "evaluate more carefully":
**generate from real source material, filter for correctness rather than
typicality, keep an external oracle however small, and report an error taxonomy
rather than an error rate.** {{eq:concentration-test}} says thirty read failures
suffice to detect concentration — **and no quantity of generator-scored evaluation
ever will.**

## 21. Further Reading

{{cite:wang2023selfinstruct}} for the method that made this a compute problem, and
note how much of its pipeline is filtering — which is the part this chapter
argues needs the most care.
{{cite:gunasekar2023}} for the strongest case that aggressive curation beats
scale, and read it against {{eq:quality-filters-rareness}}: filtering *real* data
and filtering *generated* data are different operations with the same name.
{{cite:hinton2015}} for distillation, which is grounded generation when the
teacher is genuinely external and this chapter's failure case when it is not.
{{cite:zhou2023lima}} as the reason to doubt that the quantity synthetic data
supplies is the binding constraint.
{{cite:lee2022dedup}} for what duplicate data does, which compounds with
everything here.
