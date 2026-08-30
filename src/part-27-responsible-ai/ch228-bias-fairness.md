---
id: rai-bias
number: 228
part: XXVII
tier: full
status: draft
requires: [f1-asserts-a-cost-ratio, calibration-is-required-for-decisions,
           a-score-needs-a-human-baseline, agreement-caps-measurable-quality]
provides: [three-fairness-criteria-cannot-hold-together, the-violation-is-proportional-to-base-rate-difference,
           disparity-decomposes-and-only-some-parts-are-fixable, tokenisation-imposes-a-cost-disparity-before-any-model-runs]
citations: [kleinberg2016tradeoffs, hardt2016equality, mitchell2019modelcards, petrov2023]
---

## 1. Learning Objectives

By the end of this chapter you will be able to show that calibration and the two error-rate
balance conditions cannot hold simultaneously across groups with different base rates; compute
how far each criterion is violated when another is enforced; show that the size of the
compromise is set by the base-rate gap rather than by model quality; decompose a measured
disparity into its five sources and identify which have a remedy; rank remedies by disparity
removed per unit of effort; and compute the cost and context disparity a tokenizer imposes
before any model runs.

## 2. Why This Matters

{{cite:kleinberg2016tradeoffs}} proved that three fairness conditions cannot be satisfied
simultaneously except in constrained special cases, and that approximate satisfaction requires
the data to lie in an approximate version of one of them. The special cases are **equal base
rates** or **perfect prediction**.

Made concrete: two groups with base rates 34% and 13% and identical within-group score quality.
At a shared threshold, true-positive and false-positive rates match exactly and predictive
value is **0.740 against 0.453**. Enforcing equal predictive value instead leaves a **0.386**
gap in true-positive rate ({{eq:three-fairness-criteria-cannot-hold-together}}).

And the size of the compromise is a property of the population. At equal base rates both gaps
are **zero**; at a 0.32 base-rate gap the predictive-value gap is **0.639**
({{eq:the-violation-is-proportional-to-base-rate-difference}}). Improving the model does close
it — at a d-prime of **4.0**, an AUC above 0.997, against a typical deployed **1.55**.

The second half is what to do anyway. A measured disparity decomposes: **31%** base-rate
difference with no remedy, **22%** label bias, **18%** representation, **14%** feature quality,
**15%** threshold ({{eq:disparity-decomposes-and-only-some-parts-are-fixable}}). The cheapest
remedy — a per-group threshold, which is {{cite:hardt2016equality}}'s post-processing — returns
**0.375** per unit of effort against 0.020 for collecting more data.

And one disparity arrives before any of this. {{cite:petrov2023}} measured tokenizer fragmentation
across languages: a Burmese user gets **0.22×** the usable context and pays **4.63×** as much
for identical content ({{eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs}}).

## 3. Prerequisites

{{eq:calibration-is-required-for-decisions}} from {{ch:ev-classical-metrics}} is one of the
three criteria under a different name — equal predictive value across groups is group-wise
calibration, and this chapter shows it is incompatible with equal error rates.

{{eq:f1-asserts-a-cost-ratio}} from the same chapter is the structural parallel: a threshold
chosen without a stated criterion has chosen one, and here the unstated choice is a fairness
criterion rather than a cost ratio.

{{eq:a-score-needs-a-human-baseline}} from {{ch:ev-llm-benchmarks}} applies to disparity
numbers too — a gap has no units until you know what gap exists in the outcome being predicted.

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} bounds the label-bias term:
disparity attributed to the model may be disparity in the labels, and the two are
indistinguishable without a re-annotation.

## 4. Intuitive Explanation

Fairness in machine learning has a theorem, and the theorem is the most useful thing in the
subject because it converts an argument into a choice.

Take two groups. Suppose your score works *equally well* in both — same separation between
positives and negatives, no measurement bias, no representation gap. The only difference is
that the base rate is 34% in one group and 13% in the other.

Now pick a threshold and apply it to both. The true-positive rates are identical. The
false-positive rates are identical. The score is doing exactly the same job in each group.

And the predictive value — of the people you flagged, how many are actually positive — is 0.740
in one group and 0.453 in the other.

Nothing is wrong. Predictive value depends on the base rate, by Bayes' rule, and the base rates
differ. If a human decision-maker is told "this person scored above the threshold" and reads
that as "this person is probably positive," they are right 74% of the time for one group and
45% for the other, from the same number.

So: fix it. Set a different threshold for group B so the predictive values match.

Now the true-positive rates differ by 0.386. You have moved the problem, not solved it.

{{cite:kleinberg2016tradeoffs}} proved you cannot solve it. Three conditions — calibration
within groups, equal false-positive rates, equal false-negative rates — cannot all hold unless
the base rates are equal or the prediction is perfect. And they proved the approximate version
too: getting all three approximately right requires the data to be approximately in one of
those special cases.

That result is often reported as depressing. It is the opposite: it means a fairness
requirement is a **decision** with a right answer that depends on the application, rather than
a target somebody is failing to hit.

The next question is how big a compromise you are making, and the answer is not about your
model.

At equal base rates, every gap is zero. As the base rates diverge, the gaps grow: at a 0.21
gap in base rates, the predictive-value gap at equal error rates is 0.288; at a 0.32 gap, it is
0.639.

**The size of the fairness compromise is a property of the population.** A team that tries to
close it by improving the model is working on the wrong term.

Improving the model does work, though — it is the other special case — and it is worth knowing
how much improvement. At a d-prime of 1.55, a typical deployed classifier with an AUC around
0.86, the gaps are 0.288 and 0.102. At a d-prime of 4.0, an AUC above 0.997, they are 0.001 and
0.000.

So "just make the model better" is not wrong. It requires a classifier far beyond anything
deployed, and between the two special cases — equal base rates, near-perfect prediction — every
real system sits in neither.

Which leaves the design question: **which criterion does this application need?**

If a missed case is the harm — disease screening, safety flagging — you want equal
true-positive rates, because the cost of a miss falls on the person missed. If a false flag is
the harm — investigation, moderation, denial — you want equal false-positive rates. If a score
is handed to a human who will read it as a probability, you want equal predictive value,
because otherwise the same number means different things about different people.

Pick the one the harm structure implies, state it, and report the others as measured
violations. That is available today and costs nothing but a decision — and it is the opposite
of the common practice, which is to report whichever criterion the system happens to satisfy.

That is the theory. The second half is what to do about a disparity you have measured.

The mistake is to treat "the model performs worse for group B" as one number with one remedy.
It is at least five things added together.

**Base-rate difference**: 31% of the measured gap here, and it has no remedy inside the model.
It is a fact about the world, and it is also what sets the size of the impossibility.

**Label bias**: 22%. The training labels themselves encode a disparity — historical decisions,
annotator patterns, proxy outcomes. Fixable by relabelling a stratified sample, which is
expensive and is the only way to distinguish this term from genuine signal.

**Representation**: 18%. Fewer examples from the group means a worse fit for that group.
Fixable by collecting more data, which is the most expensive remedy on the list.

**Feature quality**: 14%. The available features are less predictive for one group — a proxy
that works in one context and not another. Fixable with better features.

**Threshold**: 15%. The operating point was chosen on the pooled distribution and is wrong for
at least one group. Fixable with a per-group threshold, which is a config change.

Ranked by disparity removed per unit of effort: threshold at 0.375, label bias at 0.037,
feature quality at 0.028, more data at 0.020.

The cheapest remedy is a per-group threshold, and it is exactly
{{cite:hardt2016equality}}'s post-processing procedure — adjust the decision rule per group to
enforce a stated criterion, on a model you do not retrain. It is nearly free and it is skipped,
usually because per-group thresholds feel uncomfortable, which is a real objection that has to
be argued rather than assumed.

Build all four in payback order and the disparity goes from 1.00 to 0.31. **The floor is 0.31
and it is the base-rate term**, and no amount of further effort moves it.

Finally: the disparity that exists before any of this, and which none of the metrics above
measures.

{{cite:petrov2023}} measured tokenizer fragmentation across languages. The same content, in
different languages, becomes a different number of tokens — and everything downstream is
denominated in tokens.

English takes 238 tokens per thousand characters here. Spanish 289. Russian 521. Hindi 744.
Burmese 1,103.

That is a 4.63× ratio, and it has three consequences at once. **Cost**: a Burmese user pays
4.63× as much for identical content. **Context**: they get 0.22× the usable window, so fewer
documents fit and retrieval recall falls. **Latency**: more tokens is more time.

Nothing about the model changed. The interface is denominated in a unit that is not neutral
across the population.

And no fairness metric in the first half of this chapter measures any of it, because all of
them are properties of a classifier and this is a property of the tokenizer.

Which produces the chapter's most uncomfortable practical finding. The interventions that
remove this — price per character rather than per token, budget context per character, give
high-fertility languages a larger window — remove *zero* measured model disparity, appear in no
fairness report, and are the largest single equity move available to most teams.

**It is a billing decision.**

## 5. Formal Explanation

**The impossibility.** Let group $g$ have base rate $b_g$ and a score with within-group
distributions $f_g^+, f_g^-$. At threshold $t_g$, $\mathrm{TPR}_g = \int_{t_g} f_g^+$,
$\mathrm{FPR}_g = \int_{t_g} f_g^-$, and

$$\mathrm{PPV}_g = \frac{b_g \mathrm{TPR}_g}{b_g \mathrm{TPR}_g + (1-b_g)\mathrm{FPR}_g}.$$

If $\mathrm{TPR}_A = \mathrm{TPR}_B$ and $\mathrm{FPR}_A = \mathrm{FPR}_B$, then
$\mathrm{PPV}_A = \mathrm{PPV}_B$ requires $b_A = b_B$. Conversely, matching PPV with
$b_A \neq b_B$ forces different $(\mathrm{TPR}, \mathrm{FPR})$ pairs. The only escapes are
$b_A = b_B$ or a degenerate ROC with $\mathrm{FPR} = 0$ at $\mathrm{TPR} = 1$ — equal base
rates or perfect prediction.

**Magnitude.** Expanding $\mathrm{PPV}$ around equal base rates gives
$\Delta\mathrm{PPV} \approx \frac{\partial \mathrm{PPV}}{\partial b}\Delta b$, so the violation
is first-order in the base-rate gap and only second-order in anything about the model. The
model enters through $\mathrm{TPR}/\mathrm{FPR}$, whose ratio must diverge for the gap to
vanish — which is the perfect-prediction limit.

**Decomposition.** Write measured disparity $D$ as a sum of contributions from the base rate,
label noise $\eta_g$, sample size $n_g$, feature informativeness $I_g$, and threshold $t_g$.
Only the last three admit interventions with bounded cost, and $\partial D/\partial t_g$ is
available at essentially zero cost, which is why post-processing dominates the ranking.

**Tokenisation.** For content $c$ and tokenizer $T$, cost is $\lambda |T(c)|$ and usable
context is $C/|T(c)|$ in content units. Define fertility $\phi_L = |T(c_L)|/|c_L|$ for language
$L$. Then relative cost is $\phi_L/\phi_{\text{en}}$ and relative context is its reciprocal.
Both are properties of $T$ alone: **the disparity is fixed before the model is invoked** and is
invariant to model quality.

## 6. Mathematical Foundation

The impossibility, in one line:

$$\mathrm{TPR}_A = \mathrm{TPR}_B \;\wedge\; \mathrm{FPR}_A = \mathrm{FPR}_B \;\wedge\; \mathrm{PPV}_A = \mathrm{PPV}_B \;\Longrightarrow\; b_A = b_B \ \text{ or } \ \text{perfect prediction}$$ (eq:three-fairness-criteria-cannot-hold-together)

At $b_A = 0.34$, $b_B = 0.13$, $d' = 1.55$: PPV **0.740** against **0.453**; enforcing equal
PPV costs a **0.386** TPR gap.

The magnitude, first-order in the base-rate gap:

$$\Delta\mathrm{PPV} \approx \frac{\partial \mathrm{PPV}}{\partial b}\,\Delta b, \qquad \Delta b = 0 \Rightarrow \Delta\mathrm{PPV} = 0$$ (eq:the-violation-is-proportional-to-base-rate-difference)

**0.058** at $\Delta b = 0.06$, **0.639** at $\Delta b = 0.32$.

Disparity as a sum with heterogeneous remedies:

$$D = D_{\text{base}} + D_{\text{label}} + D_{\text{rep}} + D_{\text{feat}} + D_{\text{thresh}}, \qquad \frac{\partial D_{\text{base}}}{\partial(\text{model})} = 0$$ (eq:disparity-decomposes-and-only-some-parts-are-fixable)

**31%** with no remedy; per-group threshold at **0.375** removed per unit of effort.

And the disparity before inference:

$$\frac{\text{cost}_L}{\text{cost}_{\text{en}}} = \frac{\phi_L}{\phi_{\text{en}}}, \qquad \frac{\text{context}_L}{\text{context}_{\text{en}}} = \frac{\phi_{\text{en}}}{\phi_L}$$ (eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs)

**4.63×** cost and **0.22×** context for Burmese against English.

## 7. Internal Mechanics

Why do base rates differ? Almost always for reasons upstream of the model — different exposure,
different access, different measurement, different historical treatment. That matters for the
theorem in a specific way: the base-rate difference is not noise to be corrected but a signal
about the world, and "equalise the base rates" is a request to change the world rather than the
classifier.

It also means the term is *not* the model's fault and *is* the model's problem. A deployed
system inherits the disparity in what it predicts, and the impossibility says it cannot present
that disparity in a way that satisfies every reasonable fairness intuition simultaneously.

Label bias has a mechanism worth separating from base rate, because they look identical in the
data. A base-rate difference is a real difference in the outcome; label bias is a difference in
how the outcome was *recorded*. Both produce different positive rates in the training set, and
no amount of analysis of that training set distinguishes them —
{{eq:agreement-caps-measurable-quality}} applies, and the only instrument is a re-annotation
under a controlled protocol. **The most expensive term to identify is the one most often
asserted without evidence**, in both directions.

The threshold term dominates the payback ranking for a mechanical reason. A threshold is a
scalar chosen after training, so changing it costs nothing and its effect on group-conditional
rates is direct. Everything else requires retraining, relabelling or collecting. That is why
{{cite:hardt2016equality}}'s post-processing framing was influential: it made fairness
adjustable without touching the model, which is the only intervention available to a team that
did not train it.

On tokenisation, the reason fertility differs is that subword vocabularies are fitted to a
training corpus, and the corpus is not linguistically uniform. Languages with less
representation get fewer dedicated subwords, so their text fragments into more, shorter pieces
— sometimes to the byte level. That is not a design flaw; it is what fitting a fixed-size
vocabulary to a skewed corpus produces, and it is largely fixed once the tokenizer ships.

The downstream consequence has a shape worth naming. Because the context window is denominated
in tokens, a high-fertility language gets fewer documents in a fixed budget, which lowers
retrieval recall, which lowers answer quality — {{eq:rag-accuracy-is-a-product-with-a-utilisation-term}}'s
chain with the first term reduced for reasons unrelated to retrieval. **A quality disparity
appears in the evaluation and its cause is in the tokenizer**, which is three layers away from
where anybody looks.

## 8. Implementation

The first listing makes the impossibility concrete.

```python {tier=A name=three-fairness-criteria-cannot-hold-together}
"""You can have calibration, equal false-positive rates, or equal false-negative rates. Two.

cite:kleinberg2016tradeoffs proved that three fairness conditions cannot be satisfied
simultaneously except in constrained special cases, and that even approximate satisfaction
requires the data to lie in an approximate version of one of those cases
(eq:three-fairness-criteria-cannot-hold-together).

The special cases are equal base rates or perfect prediction. Neither describes anything
anyone deploys.

So a fairness requirement is a *choice* among criteria rather than a target to hit, and the
size of the compromise is set by how far apart the base rates are
(eq:the-violation-is-proportional-to-base-rate-difference).
"""
import math

# Two groups, different base rates, same underlying score quality.
BASE_A, BASE_B = 0.34, 0.13
SEPARATION = 1.55                 # d-prime: how well the score separates within a group


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rates(threshold, base):
    """TPR, FPR, and the positive rate, for a group with this base rate."""
    tpr = 1.0 - phi(threshold - SEPARATION)
    fpr = 1.0 - phi(threshold)
    ppr = base * tpr + (1 - base) * fpr
    ppv = (base * tpr) / ppr if ppr > 0 else 0.0
    return tpr, fpr, ppr, ppv


print(f"Two groups. Base rates {BASE_A:.0%} and {BASE_B:.0%}, "
      f"identical score quality (d'={SEPARATION:.2f}).")
print()
print(f"{'threshold':>11}{'A: TPR':>9}{'A: FPR':>9}{'A: PPV':>9}"
      f"{'B: TPR':>9}{'B: FPR':>9}{'B: PPV':>9}")
print("-" * 65)
for t in (0.4, 0.8, 1.2, 1.6, 2.0):
    ta, fa, pa, va = rates(t, BASE_A)
    tb, fb, pb, vb = rates(t, BASE_B)
    print(f"{t:>11.2f}{ta:>9.3f}{fa:>9.3f}{va:>9.3f}"
          f"{tb:>9.3f}{fb:>9.3f}{vb:>9.3f}")

print()
print("At a single shared threshold, TPR and FPR match by construction and")
print("PPV does not, because PPV depends on the base rate.")

print()
print()
print("Now enforce each criterion in turn and measure what the other two do.")
print()


IDX = {"tpr": 0, "fpr": 1, "ppr": 2, "ppv": 3}


def solve(base, kind, ref):
    """Threshold for `base` whose `kind` equals `ref`. Monotone bisection."""
    def f(t):
        return rates(t, base)[IDX[kind]]

    lo, hi = -4.0, 8.0
    increasing = f(hi) > f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if (f(mid) < ref) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


T_A = 1.2
ta, fa, pa, va = rates(T_A, BASE_A)
print(f"Group A fixed at threshold {T_A:.2f}: "
      f"TPR {ta:.3f}, FPR {fa:.3f}, PPV {va:.3f}")
print()
print(f"{'criterion enforced on B':>28}{'B threshold':>14}{'B: TPR':>10}"
      f"{'B: FPR':>10}{'B: PPV':>10}{'largest gap':>14}")
print("-" * 86)
res = {}
for label, kind, ref in (("equal true-positive rate", "tpr", ta),
                         ("equal false-positive rate", "fpr", fa),
                         ("equal predictive value", "ppv", va),
                         ("equal selection rate", "ppr", pa)):
    tb_star = solve(BASE_B, kind, ref)
    t2, f2, p2, v2 = rates(tb_star, BASE_B)
    gaps = {"TPR": abs(t2 - ta), "FPR": abs(f2 - fa), "PPV": abs(v2 - va)}
    worst = max(gaps, key=lambda k: gaps[k])
    res[label] = (tb_star, t2, f2, v2, gaps, worst)
    print(f"{label:>28}{tb_star:>14.3f}{t2:>10.3f}{f2:>10.3f}{v2:>10.3f}"
          f"{f'{worst} {gaps[worst]:.3f}':>14}")

print()
print("Every row satisfies one criterion exactly and violates the others.")

print()
print()
print("How the compromise scales with the base-rate gap.")
print()
print(f"{'base rate B':>13}{'gap to A':>11}{'PPV gap at equal FPR':>23}"
      f"{'FPR gap at equal PPV':>23}")
print("-" * 70)
scale = {}
for bb in (0.34, 0.28, 0.21, 0.13, 0.06, 0.02):
    t_eqfpr = T_A
    _, _, _, v_eqfpr = rates(t_eqfpr, bb)
    ppv_gap = abs(v_eqfpr - va)
    t_eqppv = solve(bb, "ppv", va)
    _, f_eqppv, _, _ = rates(t_eqppv, bb)
    fpr_gap = abs(f_eqppv - fa)
    scale[bb] = (BASE_A - bb, ppv_gap, fpr_gap)
    print(f"{bb:>13.0%}{BASE_A - bb:>11.2f}{ppv_gap:>23.3f}{fpr_gap:>23.3f}")

print()
print("At equal base rates both gaps are zero. That is the special case.")

print()
print()
print("What better prediction does, which is the other special case.")
print()
print(f"{'separation d-prime':>20}{'PPV gap at equal FPR':>23}"
      f"{'FPR gap at equal PPV':>23}{'reading':>22}")
print("-" * 88)
sep_tab = {}
for d in (0.6, 1.55, 2.6, 4.0, 6.0):
    globals()["SEPARATION"] = d
    # Hold group A's true-positive rate fixed as separation improves,
    # so the comparison is at the same operating point each time.
    t_d = d - 0.35
    _, _, _, va_d = rates(t_d, BASE_A)
    _, _, _, vb_d = rates(t_d, BASE_B)
    t_eqppv = solve(BASE_B, "ppv", va_d)
    _, fb_d, _, _ = rates(t_eqppv, BASE_B)
    _, fa_d, _, _ = rates(t_d, BASE_A)
    sep_tab[d] = (abs(vb_d - va_d), abs(fb_d - fa_d))
    reading = ("useless" if d < 1 else "typical" if d < 3
               else "excellent" if d < 5 else "near-perfect")
    print(f"{d:>20.2f}{abs(vb_d - va_d):>23.3f}{abs(fb_d - fa_d):>23.3f}"
          f"{reading:>22}")
globals()["SEPARATION"] = 1.55

print()
print("The gaps close as prediction approaches perfect, which is the second")
print("special case and is not available.")

print()
print()
print("So the design question is which criterion the application needs.")
print()
APPS = [
    ("who gets screened for a disease", "equal true-positive rate",
     "a missed case is the harm"),
    ("who is flagged for review",       "equal false-positive rate",
     "a false flag is the harm"),
    ("what a score is told to mean",    "equal predictive value",
     "the number is shown to a decider"),
    ("who receives a scarce resource",  "equal selection rate",
     "allocation is the outcome"),
    ("who is offered credit",           "contested",
     "all three have advocates"),
]
print(f"{'application':>34}{'the criterion it wants':>28}"
      f"{'why':>34}")
print("-" * 96)
for name, crit, why in APPS:
    print(f"{name:>34}{crit:>28}{why:>34}")

print(f"""
The threshold table is the setup and it already contains the problem. At any shared threshold,
the two groups have identical true-positive and false-positive rates -- the score works equally
well in both -- and different predictive values: {va:.3f} against
{rates(T_A, BASE_B)[3]:.3f} at threshold {T_A:.2f}.

Nothing is wrong with the model. **PPV depends on the base rate**, so equal error rates and
equal predictive value cannot both hold when the base rates differ.

The enforcement table is cite:kleinberg2016tradeoffs' result made concrete. Enforcing equal
true-positive rate on group B leaves a
{res['equal true-positive rate'][4]['PPV']:.3f} gap in predictive value. Enforcing equal
predictive value leaves a {res['equal predictive value'][4]['TPR']:.3f} gap in true-positive
rate and a {res['equal predictive value'][4]['FPR']:.3f} gap in false-positive rate. Enforcing
equal selection rate leaves {res['equal selection rate'][4]['PPV']:.3f} in predictive value
(eq:three-fairness-criteria-cannot-hold-together).

Note that the first two rows coincide: with equal within-group score quality, matching
true-positive rates and matching false-positive rates are the same threshold. The tension is
between *either* of those and predictive value, and it is unavoidable.

**Every row satisfies one criterion exactly and violates the others.** There is no threshold
choice that satisfies all three, and the proof is that there is no such threshold rather than
that nobody has found it.

The scaling table says how much of a compromise is being made. At equal base rates
({BASE_A:.0%} and {BASE_A:.0%}) both gaps are zero. At a {BASE_A - 0.02:.2f} gap in base rates
the PPV gap at equal FPR is {scale[0.02][1]:.3f}
(eq:the-violation-is-proportional-to-base-rate-difference).

So **the size of the fairness compromise is a property of the population, not of the model.**
A team that reduces the gap by improving the model is working on the wrong term; a team that
reduces it by changing who is in the population has changed the question.

The separation table is the other special case. As the score approaches perfect prediction
(d-prime {6.0:.1f}), the gaps fall to {sep_tab[6.0][0]:.3f} and {sep_tab[6.0][1]:.3f}. A
typical deployed model sits near {1.55:.2f}, where they are
{sep_tab[1.55][0]:.3f} and {sep_tab[1.55][1]:.3f}.

That is the honest version of "just make the model better": **it does work**, and the
separation required is not one anybody reaches. A d-prime of {4.0:.1f} corresponds to an AUC
above {0.997:.3f}; a typical deployed classifier sits near {1.55:.2f}, which is an AUC around
{0.86:.2f}.

So the second special case is real and unavailable. Between the two -- equal base rates, or a
near-perfect classifier -- a deployed system is in neither, which is exactly what
cite:kleinberg2016tradeoffs' theorem says.

The last table is what to do instead, and it is a product question rather than a statistical
one. The criterion an application needs depends on where the harm falls. If a missed case is
the harm, you want equal true-positive rates. If a false flag is the harm, equal false-positive
rates. If the number is handed to a human decider who will read it as a probability, equal
predictive value -- which is ch:ev-classical-metrics' calibration requirement wearing a fairness
label.

**Pick the criterion the harm structure implies, state it, and report the others as measured
violations.** That is available today, costs nothing but a decision, and is the opposite of the
common practice of reporting whichever criterion the system happens to satisfy.""")
```

## 9. Practical Example

Two groups, identical score quality, different base rates:

```
  threshold   A: TPR   A: FPR   A: PPV   B: TPR   B: FPR   B: PPV
-----------------------------------------------------------------
       0.80    0.773    0.212    0.653    0.773    0.212    0.353
       1.20    0.637    0.115    0.740    0.637    0.115    0.453
       1.60    0.480    0.055    0.819    0.480    0.055    0.567
```

At a shared threshold the error rates match exactly and **predictive value does not**, because
PPV depends on the base rate.

```
     criterion enforced on B   B threshold    B: TPR    B: FPR    B: PPV   largest gap
--------------------------------------------------------------------------------------
    equal true-positive rate         1.200     0.637     0.115     0.453     PPV 0.288
   equal false-positive rate         1.200     0.637     0.115     0.453     PPV 0.288
      equal predictive value         2.222     0.251     0.013     0.740     TPR 0.386
        equal selection rate         0.774     0.781     0.219     0.347     PPV 0.393
```

**Every row satisfies one criterion exactly and violates the others**
({{eq:three-fairness-criteria-cannot-hold-together}}) — and there is no threshold that
satisfies all three, by theorem rather than by search.

```
  base rate B   gap to A   PPV gap at equal FPR   FPR gap at equal PPV
----------------------------------------------------------------------
          34%       0.00                  0.000                  0.000
          21%       0.13                  0.145                  0.076
          13%       0.21                  0.288                  0.102
           2%       0.32                  0.639                  0.115
```

**The size of the compromise is a property of the population**
({{eq:the-violation-is-proportional-to-base-rate-difference}}).

```
  separation d-prime   PPV gap at equal FPR   FPR gap at equal PPV               reading
----------------------------------------------------------------------------------------
                1.55                  0.288                  0.102               typical
                2.60                  0.078                  0.010               typical
                4.00                  0.001                  0.000             excellent
                6.00                  0.000                  0.000          near-perfect
```

The second special case is real and out of reach: a d-prime of 4.0 is an AUC above 0.997
against a typical deployed 0.86.

The second listing decomposes a measured disparity.

```python {tier=A name=disparity-decomposes-and-only-some-parts-are-fixable}
"""Measured disparity is five different things added together, and only some are fixable.

A single "the model performs worse for group B" number tells you nothing about what to do,
because it sums contributions with completely different remedies: a base-rate difference that
is a fact about the world, label bias that is a fact about the annotation, a representation
gap that is a fact about the sample, a feature-quality gap, and a threshold choice
(eq:disparity-decomposes-and-only-some-parts-are-fixable).

And one contribution arrives before any model runs. cite:petrov2023 measured that tokenizers
fragment some languages far more than others, so the same content costs more tokens -- which
is more money, less context, and worse latency, for identical requests
(eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs).
"""
# (source, contribution to measured disparity, fixable by, cost, is it a model fix?)
SOURCES = [
    ("base-rate difference",       0.31, "nothing in the model",  0.0, "no"),
    ("label bias in the training set", 0.22, "relabel a sample",  6.0, "no"),
    ("representation: sample size", 0.18, "collect more data",    9.0, "partly"),
    ("feature quality for the group", 0.14, "better features",    5.0, "yes"),
    ("threshold chosen on the pooled set", 0.15, "set it per group", 0.4, "yes"),
]

print("Where a measured disparity comes from.")
print()
print(f"{'source':>36}{'share':>9}{'remedy':>24}{'effort':>9}"
      f"{'a model fix?':>15}")
print("-" * 93)
src = {}
for name, share, fix, eff, ismodel in SOURCES:
    src[name] = (share, eff, ismodel, share / eff if eff > 0 else 0.0)
    print(f"{name:>36}{share:>9.0%}{fix:>24}{eff:>9.1f}{ismodel:>15}")

removable = sum(s for n, s, f, e, m in SOURCES if e > 0)
print()
print(f"{removable:.0%} of the disparity has a remedy; "
      f"{1 - removable:.0%} does not")

print()
print()
print("Ranked by disparity removed per unit of effort.")
print()
order = sorted([s for s in SOURCES if s[3] > 0],
               key=lambda s: -(s[1] / s[3]))
print(f"{'rank':>6}{'source':>36}{'share':>9}{'effort':>9}{'per effort':>13}")
print("-" * 73)
for i, (name, share, fix, eff, m) in enumerate(order, 1):
    print(f"{i:>6}{name:>36}{share:>9.0%}{eff:>9.1f}{share / eff:>13.3f}")

print()
print(f"the cheapest remedy is {order[0][0]} at "
      f"{order[0][1] / order[0][3]:.3f}, and it is a config change")

print()
print()
print("Building remedies in payback order.")
print()
print(f"{'after fixing':>36}{'disparity remaining':>22}{'effort so far':>16}"
      f"{'vs floor':>11}")
print("-" * 85)
floor = 1.0 - removable
cur = 1.0
eff = 0.0
path = []
for name, share, fix, e, m in order:
    cur -= share
    eff += e
    path.append((name, cur, eff))
    print(f"{name:>36}{cur:>22.2f}{eff:>16.1f}{cur / floor:>11.2f}x")

print()
print(f"the floor is {floor:.2f} and it is the base-rate term")

print()
print()
print("Now the disparity that exists before the model does: tokenisation.")
print()
# (language, tokens per 1000 characters of equivalent content)
LANGS = [
    ("English",     238),
    ("Spanish",     289),
    ("Portuguese",  301),
    ("Russian",     521),
    ("Hindi",       744),
    ("Thai",        891),
    ("Burmese",    1103),
]
BASE = LANGS[0][1]
PRICE_PER_MTOK = 3.00
CONTEXT = 128_000
REQUESTS_PER_USER_YEAR = 2_400
print(f"{'language':>14}{'tokens/1k chars':>18}{'vs English':>13}"
      f"{'usable context (chars)':>25}{'cost/user/year':>17}")
print("-" * 89)
tok = {}
for name, tpk in LANGS:
    ratio = tpk / BASE
    chars = CONTEXT / tpk * 1000
    cost = REQUESTS_PER_USER_YEAR * (tpk * 2.4) / 1e6 * PRICE_PER_MTOK
    tok[name] = (ratio, chars, cost)
    print(f"{name:>14}{tpk:>18,}{ratio:>12.2f}x{chars:>25,.0f}"
          f"{cost:>17.2f}")

print()
print(f"a Burmese user gets {tok['Burmese'][1] / tok['English'][1]:.2f} times "
      f"the usable context and pays")
print(f"{tok['Burmese'][2] / tok['English'][2]:.2f} times as much, for identical requests")

print()
print()
print("What that does downstream, at a fixed context budget.")
print()
print(f"{'language':>14}{'documents that fit':>21}{'retrieval recall':>19}"
      f"{'answer quality':>17}{'vs English':>13}")
print("-" * 84)
for name, tpk in LANGS:
    docs = max(1, int(CONTEXT * 0.55 / (tpk * 1.8)))
    recall = 1.0 - 0.72 ** docs
    quality = 0.31 + 0.62 * recall
    q_en = 0.31 + 0.62 * (1.0 - 0.72 ** max(1, int(CONTEXT * 0.55 / (BASE * 1.8))))
    print(f"{name:>14}{docs:>21}{recall:>19.3f}{quality:>17.3f}"
          f"{quality / q_en:>12.2f}x")

print()
print("Nothing about the model changed. The context budget is denominated in")
print("tokens and the tokenizer is not neutral.")

print()
print()
print("What can be done about each layer, and what it costs.")
print()
FIXES = [
    ("per-group threshold",            0.15, 0.4,  "config"),
    ("relabel a stratified sample",    0.22, 6.0,  "annotation"),
    ("oversample the smaller group",   0.09, 1.5,  "training"),
    ("collect more data for the group", 0.18, 9.0, "acquisition"),
    ("a tokenizer with better coverage", 0.00, 0.0, "not yours to change"),
    ("price and budget per character",  0.00, 1.0, "product policy"),
    ("larger context for high-fertility languages", 0.00, 2.0, "product policy"),
]
print(f"{'fix':>44}{'disparity removed':>20}{'effort':>9}{'kind':>22}")
print("-" * 95)
for name, rem, eff, kind in FIXES:
    print(f"{name:>44}{rem:>20.0%}{eff:>9.1f}{kind:>22}")

print()
print("The last three remove no measured model disparity and remove the")
print("cost and context disparity, which no fairness metric was measuring.")

print(f"""
The decomposition is the first thing to do with any disparity number, and it changes the
conversation immediately. `{SOURCES[0][0]}` is {SOURCES[0][1]:.0%} of the measured gap and has
**no remedy inside the model** -- it is a fact about the population, and
ch:rai-bias' first listing showed it is also what sets the size of the impossibility.

{removable:.0%} of the disparity has a remedy and {1 - removable:.0%} does not
(eq:disparity-decomposes-and-only-some-parts-are-fixable). A team that reports one number and
sets a target for it has committed to reducing a quantity that is
{1 - removable:.0%} outside its control.

The ranking says where to start. `{order[0][0]}` removes {order[0][1]:.0%} for
{order[0][3]:.1f} units of effort -- {order[0][1] / order[0][3]:.3f} per unit -- and it is a
configuration change. cite:hardt2016equality's post-processing procedure is exactly this: adjust
the threshold per group to enforce a stated criterion, on a model you do not retrain.

Building in payback order takes the disparity from {1.0:.2f} to {path[-1][1]:.2f} for
{path[-1][2]:.1f} units. **The floor is {floor:.2f}**, and no amount of further effort moves
it.

The tokenisation table is the disparity that exists before any of this. cite:petrov2023
measured that tokenizers fragment languages unequally, and the consequences are not subtle: a
Burmese user gets {tok['Burmese'][1] / tok['English'][1]:.2f} times the usable context and pays
{tok['Burmese'][2] / tok['English'][2]:.2f} times as much for identical content
(eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs).

That is a {tok['Burmese'][0]:.2f}x tax on one axis and a
{1 / (tok['Burmese'][1] / tok['English'][1]):.1f}x reduction on the other, and **no fairness
metric in the first listing measures either**, because both are properties of the interface
rather than of the classifier.

The downstream table follows it through. At a fixed context budget, fewer documents fit, so
retrieval recall falls, so answer quality falls -- for reasons that have nothing to do with the
model's competence in the language and everything to do with how many tokens the words became.

The fixes table is the honest ending. The first four remove measured model disparity and cost
between {0.4:.1f} and {9.0:.1f} units. The last three remove **no** measured model disparity
and remove the cost and context disparity entirely: price per character rather than per token,
budget context per character, and give high-fertility languages a larger window.

None of those is a model change and none would show up in a fairness report. **The largest
single equity intervention available to most teams is a billing decision**, and it is invisible
to every metric in this chapter's first listing.""")
```

```
                              source    share                  remedy   effort   a model fix?
---------------------------------------------------------------------------------------------
                base-rate difference      31%    nothing in the model      0.0             no
      label bias in the training set      22%        relabel a sample      6.0             no
         representation: sample size      18%       collect more data      9.0         partly
       feature quality for the group      14%         better features      5.0            yes
  threshold chosen on the pooled set      15%        set it per group      0.4            yes
```

**69% has a remedy and 31% does not**
({{eq:disparity-decomposes-and-only-some-parts-are-fixable}}) — a target set on the aggregate
commits to a number that is a third outside the team's control.

```
  rank                              source    share   effort   per effort
-------------------------------------------------------------------------
     1  threshold chosen on the pooled set      15%      0.4        0.375
     2      label bias in the training set      22%      6.0        0.037
     4         representation: sample size      18%      9.0        0.020

                        after fixing   disparity remaining   effort so far   vs floor
-------------------------------------------------------------------------------------
  threshold chosen on the pooled set                  0.85             0.4       2.74x
         representation: sample size                  0.31            20.4       1.00x
```

The cheapest remedy is {{cite:hardt2016equality}}'s post-processing, at **0.375** per unit —
and the floor is **0.31**, which is the base-rate term.

```
      language   tokens/1k chars   vs English   usable context (chars)   cost/user/year
-----------------------------------------------------------------------------------------
       English               238        1.00x                  537,815             4.11
       Russian               521        2.19x                  245,681             9.00
         Hindi               744        3.13x                  172,043            12.86
       Burmese             1,103        4.63x                  116,047            19.06
```

**4.63× the cost and 0.22× the context, for identical content**
({{eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs}}) — and no fairness metric
above measures either.

```
                                         fix   disparity removed   effort                  kind
------------------------------------------------------------------------------------------------
                         per-group threshold                 15%      0.4                config
             collect more data for the group                 18%      9.0           acquisition
            price and budget per character                   0%      1.0        product policy
 larger context for high-fertility languages                  0%      2.0        product policy
```

The last two remove **no** measured model disparity and remove the cost and context disparity
entirely. **The largest equity move available to most teams is a billing decision.**

## 10. Production Considerations

Choose a fairness criterion from the harm structure and state it. Reporting the others as
measured violations is honest and free; reporting whichever one passes is not.

Measure your base-rate gap first. It bounds every other number in the analysis and no model
work moves it.

Set thresholds per group. It is {{cite:hardt2016equality}}'s procedure, it costs a config
change, and it is the highest-return remedy in the decomposition.

Decompose before setting a target. A target on the aggregate commits to reducing a quantity
that is a third outside your control.

Distinguish label bias from base rate with a re-annotation, or say you have not. The two are
indistinguishable in the data and both are routinely asserted.

Measure token fertility across your user languages. It is one script and it produces a cost and
a context ratio nobody has seen.

Price and budget per character, not per token. It removes a 4.63× disparity and appears in no
fairness report.

## 11. Common Mistakes

**Treating fairness as a target to hit.** The theorem says it is a choice among criteria.

**Trying to close the gap by improving the model.** It works at an AUC of 0.997.

**Reporting one disparity number.** It sums five terms with different remedies and one with
none.

**Skipping per-group thresholds.** The cheapest remedy by an order of magnitude.

**Asserting label bias without a re-annotation.** Indistinguishable from base rate in the data.

**Measuring only model disparity.** The tokenizer imposes a larger one and none of the metrics
see it.

## 12. Failure Modes

**A fairness target that cannot be met.** Set on the aggregate, 31% of which has no remedy, and
the programme is judged against it.

**Criterion switched to whichever passes.** Equal error rates reported this quarter, predictive
parity next, and the system did not change.

**Per-group thresholds rejected on principle without a stated alternative.** The pooled
threshold is also a per-group decision, made once, badly.

**Model retrained to fix a threshold problem.** Nine units of effort spent on the 18% term while
the 15% term was a config change.

**Quality disparity blamed on the model.** The cause is fewer documents fitting in the context
budget because of tokenizer fertility.

**Cost disparity invisible.** Billing is per token, the disparity is 4.63×, and no dashboard is
denominated in anything else.

## 13. Alternatives

**Post-processing per group.** {{cite:hardt2016equality}}'s procedure. Cheapest, model-agnostic,
and it requires group membership at decision time — which is often legally constrained.

**In-processing with a fairness constraint.** Train subject to a parity constraint. Avoids
group-conditional decisions at serving time and costs accuracy and a retrain.

**Pre-processing the data.** Reweight or resample to equalise. Addresses representation and
leaves base-rate and label terms untouched.

**Report and do not adjust.** Publish the disaggregated numbers, per
{{cite:mitchell2019modelcards}}, and let the deploying party choose. Honest, and it moves the
decision to whoever holds the harm.

**Change the decision, not the score.** If no criterion is satisfactory, the problem may be that
a threshold decision is the wrong instrument — a rank, a range, or a referral may carry the
uncertainty better.

## 14. Evaluation

Publish disaggregated metrics as {{cite:mitchell2019modelcards}} specifies: TPR, FPR, PPV and
selection rate per group, at the deployed threshold.

State the criterion you chose and the measured violation of the other three. Both halves are
required for the report to mean anything.

Measure your base-rate gap and put it beside the disparity. It is the floor and it explains the
size.

Run a re-annotation on a stratified sample to separate label bias from base rate. It is the only
instrument for that term.

Measure token fertility for every language above 1% of traffic, and publish cost and context
ratios alongside quality metrics.

## 15. Advanced Concepts

The impossibility is stated for two groups and a binary outcome, and both restrictions matter.
With more than two groups the criteria conflict pairwise, so a system satisfying equal error
rates between the two largest groups may violate them badly for a small one — and the aggregate
metrics will not show it, because small groups contribute little to a pooled number. **The
number of pairwise constraints grows quadratically and the reporting usually grows linearly**,
which is the same counting problem {{ch:sec-tool-abuse}} found for tool compositions.

The decomposition treats the five terms as additive, and they interact. A representation gap
makes feature quality worse for the under-represented group, and a threshold set on the pooled
distribution is more wrong when representation is skewed. The interactions are all in the same
direction — they compound — so the additive model understates the benefit of fixing several
terms together and overstates the benefit of fixing one. The practical consequence is that the
payback ranking is right about *order* and pessimistic about the *total*.

There is a deeper issue with the base-rate term that the arithmetic cannot reach. A base-rate
difference is treated here as exogenous, and in a deployed system it is partly endogenous: the
model's decisions affect who is exposed to the outcome, which changes the base rate the next
model is trained on. That feedback is well documented in prediction systems and it means the
"floor" in {{sec:9-practical-example}} is a floor for one round, not a fixed point. **A system
that is fair by any criterion this year can drift by changing the population it measures**,
which argues for re-measuring base rates on a schedule rather than treating them as a constant
of the problem.

Finally, on tokenisation. The fertility disparity is fixed once the tokenizer ships, and the
tokenizer is usually not yours. That makes it the clearest case in this book of a fairness
property inherited from a supplier — which puts it in {{ch:sec-poisoning}}'s supply-chain
frame, with the same conclusion: you cannot fix the link, so the remedy is at the interface.
Pricing per character and budgeting context per character are exactly that: interface changes
that neutralise an upstream property you cannot change.

## 16. Connection to Previous Chapters

{{eq:calibration-is-required-for-decisions}} from {{ch:ev-classical-metrics}} is the
predictive-value criterion under another name, and this chapter shows it is incompatible with
equal error rates whenever base rates differ.

{{eq:f1-asserts-a-cost-ratio}} from the same chapter is the structural parallel: an unstated
threshold choice has made a decision, and here the decision is which group bears which error.

{{eq:agreement-caps-measurable-quality}} from {{ch:ev-why-hard}} bounds the label-bias term and
explains why it cannot be separated from base rate without a re-annotation.

{{eq:rag-accuracy-is-a-product-with-a-utilisation-term}} from {{ch:ev-rag}} is the chain through
which tokenizer fertility becomes an answer-quality disparity three layers from its cause.

## 17. Exercises

1. Compute PPV for your groups at your deployed threshold. How large is the gap, and what is
   the base-rate gap that produces it?

2. Enforce each of the four criteria in turn and record the violation of the others. Which does
   your application's harm structure imply?

3. Decompose your measured disparity into the five terms. What share has no remedy?

4. Implement per-group thresholds on a held-out set and measure the change. How does the effort
   compare to the retrain you were planning?

5. Measure token fertility for your top five user languages and compute cost and context ratios.
   What would per-character pricing change?

## 18. Interview Questions

1. Our model has equal accuracy across groups. Is it fair?

2. Why can we not have calibration and equal error rates at the same time?

3. Our fairness gap did not close after retraining. What would you check?

4. Which fairness criterion does a disease-screening system want, and why?

5. What is the cheapest intervention that reduces a measured disparity?

6. A user writing in Hindi gets worse answers. Where would you look first?

## 19. Research Questions

1. How large are pairwise criterion violations for small groups in multi-group deployments, and
   are they reported anywhere?

2. How much do the five decomposition terms interact in practice, and how much does the
   additive model understate joint remediation?

3. How fast do base rates drift under deployment feedback, and what re-measurement cadence does
   that imply?

4. How much of the tokenizer fertility disparity is removable by vocabulary changes that do not
   degrade the majority language?

## 20. Chapter Summary

Fairness has a theorem, and the theorem converts an argument into a decision.

Calibration and the two error-rate balance conditions cannot hold together across groups with
different base rates ({{eq:three-fairness-criteria-cannot-hold-together}}): at base rates of
34% and 13% with identical score quality, a shared threshold gives PPV **0.740 against 0.453**,
and enforcing equal PPV instead costs a **0.386** true-positive gap. The special cases are equal
base rates or near-perfect prediction, and a deployed system is in neither.

The size of the compromise is a property of the population, not the model: **zero** at equal
base rates, **0.639** at a 0.32 gap
({{eq:the-violation-is-proportional-to-base-rate-difference}}). Improving the model does close
it, at a d-prime of **4.0** — an AUC above 0.997 against a typical deployed **1.55**.

A measured disparity decomposes into **31%** base rate with no remedy, **22%** label bias,
**18%** representation, **14%** feature quality and **15%** threshold
({{eq:disparity-decomposes-and-only-some-parts-are-fixable}}). The threshold term returns
**0.375** per unit of effort — {{cite:hardt2016equality}}'s post-processing, a config change —
against **0.020** for collecting data. The floor is **0.31**.

And the largest disparity in the chapter is not in the model at all. {{cite:petrov2023}}'s
tokenizer fertility means a Burmese user pays **4.63×** and gets **0.22×** the context for
identical content ({{eq:tokenisation-imposes-a-cost-disparity-before-any-model-runs}}) — a
disparity no fairness metric here measures, whose remedies are pricing and context policy, and
which would appear in no fairness report.

What runs through the chapter is that the fairness literature's instruments measure a
classifier and most of a deployed system's disparity is somewhere else: in the population's
base rates, in the annotation, in the billing unit. The theorem is the most useful thing in the
subject precisely because it stops the search for a classifier-level answer and sends the
question where it belongs — to a decision about which harm the application is trying to avoid.

Carry forward: **pick the criterion, state it, and report the violations**, and **measure token
fertility**.

## 21. Further Reading

- {{cite:kleinberg2016tradeoffs}} — the impossibility, including the approximate version that
  rules out "nearly satisfying all three".
- {{cite:hardt2016equality}} — equalised odds, equality of opportunity, and the post-processing
  procedure that dominates the remedy ranking here.
- {{cite:mitchell2019modelcards}} — disaggregated evaluation as a reporting standard, which is
  what makes any of these numbers visible.
- {{cite:petrov2023}} — tokenizer fertility across languages, and the cost and context
  disparities that follow from it.
