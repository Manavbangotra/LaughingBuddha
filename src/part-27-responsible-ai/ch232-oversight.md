---
id: rai-oversight
number: 232
part: XXVII
tier: full
status: draft
requires: [approval-quality-falls-with-volume, a-low-rejection-rate-trains-approval,
           stated-reasons-need-not-be-actual-reasons, an-explanation-serves-one-audience]
provides: [review-helps-only-when-catch-exceeds-override-odds,
           an-explanation-raises-confidence-faster-than-accuracy,
           oversight-is-a-conjunction-of-preconditions,
           reviewers-bear-the-cost-of-rejecting-not-approving]
citations: [turpin2023faithfulness, guo2017calibration, kadavath2022, ribeiro2016lime]
---

## 1. Learning Objectives

By the end of this chapter you will be able to state the condition under which adding a human
reviewer improves on the model alone, and evaluate it from measured catch and override rates;
compute the model accuracy above which a fixed review process makes a system worse; measure how
what a reviewer is shown changes their accuracy and their confidence separately, and recognise
when the two diverge; score an oversight arrangement on authority, information, time and
incentive and identify the binding constraint; and compute the certainty threshold a reviewer's
own incentives impose on rejection.

## 2. Why This Matters

"A human reviews the output" is the answer this book has reached for at the end of four
chapters — bias, interpretability, privacy, regulation. It is also the least measured claim in
any of them.

The claim is testable. Across five tasks, the model-proposes / human-disposes team beats the
model alone in **4 of 5** cases and beats **the better of its two members in only 2**, with a
mean gain over that better member of **−0.28 points**.

The reason is a closed form. A reviewer both catches errors and creates them, so review helps
exactly when the catch-to-override ratio beats the model's own odds of being right
({{eq:review-helps-only-when-catch-exceeds-override-odds}}). On fraud disposition the bar is
**15.7** and the reviewer's ratio is **5.5** — reviewing makes it worse, and it looks like
diligence.

Worse, the bar rises with the model. Holding a fixed reviewer, review helps up to **0.921**
model accuracy and hurts above it.

What the reviewer is shown moves the catch rate more than anything else — and moves their
confidence faster still. A plausible but wrong explanation drops the catch rate to **0.19** and
raises confidence from **0.54 to 0.78**
({{eq:an-explanation-raises-confidence-faster-than-accuracy}}).

Then the preconditions. Authority, information, time and incentive **multiply**
({{eq:oversight-is-a-conjunction-of-preconditions}}); time or information is binding in **5 of
6** arrangements; verifying one item takes **16.0 minutes** against a 90-second budget. And
rejecting costs the reviewer **79×** what approving costs while costing the organisation
**2.3× less** — an inversion of **182**, and a **94%** certainty threshold before a rational
reviewer says no ({{eq:reviewers-bear-the-cost-of-rejecting-not-approving}}).

## 3. Prerequisites

{{eq:approval-quality-falls-with-volume}} from {{ch:sec-permissions}} is the volume half of this
chapter; the accuracy half is here, and the two meet at the reviewer's time budget.

{{eq:a-low-rejection-rate-trains-approval}} from the same chapter is the observation; the
incentive arithmetic in {{sec:9-practical-example}} is the mechanism, and the **94%** threshold
is why the rate falls.

{{eq:stated-reasons-need-not-be-actual-reasons}} from {{ch:rai-interpretability}} becomes a
measured effect on a person here: {{cite:turpin2023faithfulness}}'s unfaithful reasoning shown
to a reviewer.

{{eq:an-explanation-serves-one-audience}} from the same chapter is why the reviewer's artefact
and the regulator's artefact are not the same document.

{{cite:guo2017calibration}}'s calibration result is an oversight requirement rather than a
modelling nicety, and {{cite:kadavath2022}} is the question of whether a model can supply the
signal that routing needs.

## 4. Intuitive Explanation

There is a sentence that ends most discussions of AI risk: *a human reviews the output*.

It ends the fairness discussion, because a person can catch a disparate outcome. It ends the
interpretability discussion, because a person can ask why. It ends the privacy discussion,
because a person can notice a leak. It ends the regulatory discussion, because human oversight
is an obligation in every framework in circulation.

It ends those discussions and it starts this chapter, because it is an empirical claim that is
almost never measured, and when you measure it the results are not what the sentence implies.

Start with the strongest form of the claim: the pair is better than the model.

Take five tasks with a model, a human, and a model-proposes / human-disposes arrangement.
Content policy: model 0.910, human 0.940, team 0.926. Loan adjudication: 0.870, 0.790, 0.884.
Clinical triage: 0.830, 0.880, 0.892. Security review of a diff: 0.710, 0.760, 0.759. Fraud
alert disposition: 0.940, 0.810, 0.916.

The team beats the model alone in four of five. It beats *the better of its two members* in two
of five, and the mean gain over that better member is **−0.28 points**.

Both of those facts are worth stating and they say different things. Adding a reviewer to a
model usually helps the model. It does not usually produce something better than whichever of
the two was already better — which is what "human oversight makes it safe" implicitly claims.

Why? Because a reviewer does two things and only one of them is good. They catch some of the
model's errors, and they overturn some of its correct answers.

Write $a_m$ for model accuracy, $c$ for the rate at which the reviewer catches the model's
errors, $f$ for the rate at which they overturn its correct answers. Team accuracy is
$a_m(1-f) + (1-a_m)c$, and the reviewer improves on the model exactly when

$$c/f > a_m/(1-a_m)$$

The catch-to-override ratio has to beat the model's own odds of being right.

Look at fraud disposition through that lens. The model is right 94% of the time, so the bar is
**15.7**. The reviewer catches 22% of errors and wrongly overturns 4% of correct answers — a
ratio of **5.5**. Every hour spent on that queue makes the system slightly worse, and it
produces a full audit trail while doing it.

Now the part that should change how you plan. Hold the reviewer fixed — catch 0.35, override
0.03, a genuinely competent reviewer — and vary the model.

At 0.70 accuracy the team gains 8.40 points. At 0.85, 2.70. At 0.90, 0.80. At 0.92, 0.04. At
0.95, it *loses* 1.10 points, and at 0.99 it loses 2.62.

Break-even is **0.921**. Above that this review process makes the system worse.

**The better the model gets, the harder it is for a human to add anything.** The reviewer's own
mistakes are drawn from a pool that grows as the model improves; the errors they might catch are
drawn from one that shrinks. A review process that was clearly worthwhile at launch can become
net-negative after a model upgrade, silently, with nothing in the pipeline changing.

That is the inverse of {{ch:sec-permissions}}' result and the same phenomenon. That chapter
showed a low rejection rate trains approval. This one shows why the rejection rate falls: there
is progressively less to reject.

So can anything be done? Yes — two things, and the arithmetic says which matters more.

The first is what the reviewer is shown. Fix the model at 0.88 and vary the artefact.

The output alone: catch 0.29, team 0.8884. Output plus a calibrated confidence: catch 0.41, team
0.9046. Output plus a faithful explanation: catch 0.47, team 0.9135. Output plus the source
documents: catch 0.52, team **0.9151** — the best row, because it is the only one that lets the
reviewer check the claim rather than assess the presentation.

Now the two rows that should worry you.

Output plus an *overconfident* confidence score: override drops to 0.019, catch drops to 0.24,
team 0.8921. Output plus a *plausible but wrong* explanation: override 0.014, catch **0.19**,
team 0.8905 — barely above showing nothing at all.

A reviewer who defers more and catches less. And both artefacts were added to help.

Now look at what they did to the reviewer's confidence. Output alone: 0.54. Overconfident score:
0.71. Plausible wrong explanation: **0.78** — the highest confidence in the table, attached to
nearly the lowest accuracy.

Put those together as confidence gained per point of accuracy gained. Source documents: 4.5.
Calibrated confidence: 5.6. Faithful explanation: 6.0. Overconfident score: 46.2. Plausible
wrong explanation: **115.4**.

**An explanation is a persuasion artefact before it is an evidence artefact.** That is
{{ch:rai-interpretability}}'s `stated-reasons-need-not-be-actual-reasons` arriving as a measured
effect on a human being, and it is why {{cite:turpin2023faithfulness}} is an oversight result and
not only an interpretability one.

The overconfident-score row is the same failure more cheaply obtained, and it is the reason
{{cite:guo2017calibration}}'s calibration is an oversight requirement. An uncalibrated
confidence is a reliance signal pointing the wrong way: the reviewer relaxes exactly where they
should not.

The second lever is where the reviewer looks. Reviewing everything, at 1.5 minutes an item, is
1,500 reviewer-minutes per thousand items and lands at 0.8956. Reviewing the bottom 20% by
confidence covers 52% of the model's errors, lands at **0.9144**, and costs 300 minutes —
**higher accuracy at 5× less human time**.

Narrower is not monotonically better. The bottom 5% covers only 13% of errors and lands at
0.8908, below reviewing everything. **Coverage collapses faster than precision improves**, which
is {{ch:ev-framework}}'s union result in another costume.

And then the limit. Hand the reviewer *exactly* the model's errors, with no opportunity to
overturn a correct answer — an oracle no routing can achieve — and the team reaches 0.9220.
**4.2 points over the model.** A reviewer who catches 35% of errors catches 35% of errors
however perfectly you point them.

**Routing is worth a great deal and it is not the binding constraint.** The catch rate is, and
the artefact table said what moves it.

That is the accuracy half. The second half is the question of whether the reviewer can act at
all, and it is where most real arrangements fail before accuracy is even relevant.

Four things have to be true. The reviewer needs the **authority** to change the outcome, enough
**information** to judge it, enough **time** to use that information, and an **incentive** that
does not punish them for saying no.

Those compose as a product. Three out of four is not 75% of an oversight process; it is a
decision record with no decision in it.

Score six arrangements. Release sign-off by the owner: 0.90 authority, 0.55 information, 0.70
time, 0.60 incentive — product **0.2079**. Reviewer approves each agent action: 0.85, 0.30, 0.20,
0.35 — **0.0179**. A nominated accountable executive: 0.99, 0.15, 0.10, 0.45 — **0.0067**, the
worst in the table despite near-perfect authority, because authority without information or time
is a signature.

Time or information is the binding constraint in **five of six** arrangements. Neither is what
organisations discuss when they design oversight. They discuss authority — who signs — which is
almost never what is missing.

Then coverage, which varies by 500×. Appeal after an adverse decision scores **0.5814**, the
best per-item quality in the table, and touches **0.8%** of decisions. A quarterly audit sample
scores 0.2295 and touches 0.2%. The full-coverage arrangements are the low-quality ones.

Multiply quality by coverage and release sign-off wins at 0.20790, ahead of everything else by
an order of magnitude — not because it is good, but because it happens to every release.

**An appeal process is excellent oversight of almost nothing.** That is not an argument against
appeals. It is an argument against counting them as the oversight of the system.

Why is time binding? Because verifying one decision is not a glance.

Read the model's output: 12 seconds, catches 6% of what is catchable. Read the input it saw: 25
seconds, 11%. Check the cited source actually says this: 55 seconds, 31%. Check no relevant
source was omitted: 180 seconds, 22%. Re-derive the decision from the policy: 240 seconds, 19%.
Check consistency with past decisions: 300 seconds, 7%.

Complete verification: **962 seconds — 16.0 minutes.** The reviewer has ninety.

Spend it optimally, best value per second first, and 90 seconds buys the source check and a read
of the output: **37% of what is catchable**, in **9%** of the time the item needs.

Doubling to 180 seconds reaches only 49%, because the next genuinely useful check costs 180
seconds by itself and does not fit. 300 seconds reaches 70%. Ninety percent needs 600 seconds —
roughly **7× the usual budget**.

**Review time buys coverage in steps rather than smoothly.** A small budget increase is often
worth exactly nothing. The useful question is not "how long do reviewers get" but "which check
are we trying to afford".

And then the precondition nobody designs.

What happens to a reviewer who rejects? They delay a colleague's work — 85% of the time. They
trigger an escalation meeting — 40%. They are overruled by a manager — 55%. They are visibly
wrong — 30%. Expected personal cost: **7.55**.

What happens to a reviewer who approves? Nothing observable, 98.8% of the time. An incident is
later traced back to the approval, 1.2% of the time. Expected personal cost: **0.10**.

**Rejecting costs the reviewer 79× what approving costs.**

Now the organisation's ledger. A rejection costs it 2.07 in delay and escalation. An approval
costs it 4.80, because the rare incident is expensive. Approving is **2.3× worse** for the
organisation and **79× better** for the reviewer.

**The incentives are inverted by a factor of 182.**

Which gives a number that explains a great deal. A reviewer minimising their own expected cost
rejects only when the expected cost of approving exceeds 7.55 — that is, when they are
**94% sure** something is wrong.

That is the mechanism behind every near-zero rejection rate anyone has measured, including
{{ch:sec-permissions}}'. It is not a statement about the character of reviewers. It is
arithmetic they did not choose and mostly cannot see.

So what moves it? Take the worst arrangement in the table and apply five interventions.

Give the reviewer the sources: product ×2.5, threshold unchanged at 94%. Budget five minutes
instead of ninety seconds: ×3.7, threshold unchanged. Put the reviewer outside the delivery
line: ×2.0, and the threshold drops to **33%**, because most of the rejection cost was social.
Re-adjudicate 2% of approvals: ×1.6, threshold **16%**, because it makes a bad approval
observable. Publish the rejection rate: ×1.4, threshold unchanged.

All five together: product **0.5063** — **28× the baseline** — and a threshold of **6%**.

Notice the split. The first two interventions buy capability and leave the incentive untouched.
The third and fourth buy incentive and barely move capability. Publishing the rate moves the
precondition score without changing any individual's calculus on any individual item, which is
about right: it is a management signal, not a decision input.

**Capability and incentive are separate levers and each fails without the other.** Giving a
reviewer everything they need to catch a problem does not make it rational for them to say so.
Making rejection safe does not give them time to know when to.

That is the whole chapter, and it is the answer to the sentence it opened with. *A human reviews
the output* is not a control. It is a hypothesis with four preconditions, an accuracy condition,
and a measurable failure mode in every direction.

## 5. Formal Explanation

**Team accuracy.** For a model-proposes / human-disposes pipeline with model accuracy $a_m$,
reviewer catch rate $c = \Pr[\text{override} \mid \text{model wrong}]$ and false-override rate
$f = \Pr[\text{override} \mid \text{model right}]$, and assuming the reviewer's override is
correct when the model was wrong:

$$a_{\text{team}} = a_m(1-f) + (1-a_m)c$$

The reviewer improves on the model iff $a_{\text{team}} > a_m$, i.e. $(1-a_m)c > a_m f$, i.e.
$c/f > a_m/(1-a_m)$. The right-hand side is the model's odds of correctness, so the required
reviewer quality grows without bound as $a_m \to 1$. Holding $(c, f)$ fixed, break-even is at
$a_m^\star = (c/f)/(1 + c/f)$.

**Preconditions.** Model effective oversight as $\Omega = A \cdot I \cdot T \cdot N \cdot \pi$
for authority, information, time, incentive and coverage $\pi$. The product form is a modelling
choice justified by substitutability: none of the four factors compensates for the absence of
another. $\partial\Omega/\partial x_j = \Omega / x_j$, so the marginal return is largest for the
smallest factor — the binding constraint is always the minimum, and improving anything else is
comparatively wasted.

**Reviewer's threshold.** Let $L_R$ be the reviewer's expected cost of rejecting and $L_A^{\max}$
the cost they bear if a bad approval is later traced to them. A reviewer minimising personal
expected cost rejects when $p \cdot L_A^{\max} > L_R$, i.e. $p > L_R / L_A^{\max}$. The
organisation's threshold uses its own costs and is far lower; the ratio of the two thresholds is
the incentive inversion.

## 6. Mathematical Foundation

When a reviewer helps:

$$a_{\text{team}} = a_m(1-f) + (1-a_m)c > a_m \iff \frac{c}{f} > \frac{a_m}{1-a_m}$$ (eq:review-helps-only-when-catch-exceeds-override-odds)

With $c = 0.35, f = 0.03$, break-even is $a_m^\star = 0.921$; the fraud queue's bar is **15.7**
against a reviewer ratio of **5.5**.

Confidence and accuracy respond to the artefact at different rates:

$$\frac{\Delta \text{confidence}}{\Delta a_{\text{team}}} = 4.5 \ \text{(sources)}, \quad 115.4 \ \text{(a plausible wrong explanation)}$$ (eq:an-explanation-raises-confidence-faster-than-accuracy)

Effective oversight as a conjunction:

$$\Omega = A \cdot I \cdot T \cdot N \cdot \pi, \qquad \frac{\partial \Omega}{\partial x_j} = \frac{\Omega}{x_j}$$ (eq:oversight-is-a-conjunction-of-preconditions)

Time or information binds in **5 of 6** arrangements; verifying one item takes **16.0 minutes**
against a 90-second budget, reaching **37%** coverage.

And the asymmetry that sets the rejection rate:

$$p^\star = \frac{L_R}{L_A^{\max}} = \frac{7.55}{8.0} = 0.94, \qquad \frac{L_R/L_A}{O_A/O_R} = 182$$ (eq:reviewers-bear-the-cost-of-rejecting-not-approving)

## 7. Internal Mechanics

The catch/override condition has an uncomfortable property: it gets harder to satisfy on exactly
the systems where oversight is most often mandated. A regulated decision system is under
pressure to be accurate, and every point of accuracy raises the bar the reviewer must clear.
A system at 0.99 needs a reviewer with a catch-to-override ratio above 99 — catching most errors
while essentially never overturning a correct answer — which is not a description of any human
process.

That does not mean oversight is pointless on accurate systems. It means its purpose changes. On
a 0.99 system a reviewer is not there to raise accuracy; they are there to catch the errors that
are *categorically* different — a novel failure mode, an input the system was never meant to
receive, a harm the accuracy metric does not price. Those are not in the catch rate at all, and
measuring the arrangement against accuracy alone will always tell you to remove it.

The confidence/accuracy divergence has a mechanism worth naming precisely. An explanation gives
the reviewer something to evaluate that is easier than the decision: prose coherence rather than
factual correctness. A coherent explanation for a wrong answer is a *harder* object to reject
than a bare wrong answer, because rejecting it means claiming the reasoning is wrong rather than
just the conclusion. That is why the wrong-explanation row has both the lowest catch rate and
the lowest override rate: the reviewer is not lazier, they are facing a better-defended claim.

Source documents avoid this because they do not argue. They let the reviewer perform the check
{{cite:ribeiro2016lime}}-style local explanations cannot: comparing a claim against the thing it
claims about, rather than against a story about how it was produced.

The precondition product's mechanism is substitutability, and the test is whether more of one
factor compensates for less of another. It does not. Doubling an executive's authority does not
give them the 16 minutes. Doubling the time does not give a reviewer the sources. The factors
gate rather than trade, which is what a product encodes and a weighted sum does not.

The incentive inversion is the most robust result in the chapter because it survives almost any
reparameterisation. The reviewer's rejection costs are immediate, certain and attributable; the
approval costs are delayed, rare and diffuse. Any organisation with those properties — which is
most — produces a high rejection threshold regardless of the specific numbers, and the only
interventions that move it are the ones that change *observability* of approvals rather than
exhortation about diligence.

## 8. Implementation

The first listing measures the team.

```python {tier=A name=review-helps-only-when-catch-exceeds-override-odds}
"""A human in the loop is a claim about a team, and teams are measurable.

"A human reviews the output" is the most common answer to every question in the previous four
chapters -- bias, interpretability, privacy, regulation. It is also the least measured.

The claim is that the pair is more accurate than the model alone. That is a testable claim with
a closed form: the reviewer helps only when the rate at which they catch the model's errors
exceeds the rate at which they overturn its correct answers, scaled by the model's own odds of
being right (eq:review-helps-only-when-catch-exceeds-override-odds).

The second half asks what the reviewer is given to review with. An explanation raises the
reviewer's confidence whether or not it reflects the model's actual computation
(cite:turpin2023faithfulness), and confidence and accuracy move at different rates
(eq:an-explanation-raises-confidence-faster-than-accuracy).
"""
# (task, model accuracy, human-alone accuracy, catch rate on model errors,
#  false-override rate on model's correct answers)
TASKS = [
    ("content policy call",        0.91, 0.94, 0.38, 0.02),
    ("loan adjudication",          0.87, 0.79, 0.31, 0.03),
    ("clinical triage note",       0.83, 0.88, 0.46, 0.02),
    ("security review of a diff",  0.71, 0.76, 0.29, 0.05),
    ("fraud alert disposition",    0.94, 0.81, 0.22, 0.04),
]


def team(a_m, c, f):
    """Accuracy of model-proposes / human-disposes."""
    return a_m * (1 - f) + (1 - a_m) * c


print("Model alone, human alone, and the two together.")
print()
print(f"{'task':>30}{'model':>9}{'human':>9}{'team':>9}"
      f"{'vs model':>11}{'vs better member':>19}")
print("-" * 87)
gain_better, beats = 0.0, 0
for name, a_m, a_h, c, f in TASKS:
    t = team(a_m, c, f)
    gm = t - a_m
    gb = t - max(a_m, a_h)
    gain_better += gb
    beats += 1 if gb > 0 else 0
    print(f"{name:>30}{a_m:>9.3f}{a_h:>9.3f}{t:>9.3f}"
          f"{gm * 100:>+10.1f}p{gb * 100:>+18.1f}p")

beats_model = sum(1 for n, am, ah, c, f in TASKS if team(am, c, f) > am)
print()
print(f"the team beats the model alone in {beats_model} of {len(TASKS)} tasks")
print(f"the team beats the better of the two members in {beats} of {len(TASKS)}")
print(f"mean gain over the better member: {gain_better / len(TASKS) * 100:+.2f} points")

print()
print()
print("Why: the reviewer both catches errors and creates them.")
print()
print(f"{'task':>30}{'catch rate':>13}{'false override':>17}"
      f"{'ratio needed':>15}{'ratio actual':>15}{'helps?':>9}")
print("-" * 99)
for name, a_m, a_h, c, f in TASKS:
    need = a_m / (1 - a_m)
    have = c / f
    print(f"{name:>30}{c:>13.2f}{f:>17.2f}{need:>15.1f}{have:>15.1f}"
          f"{('yes' if have > need else 'no'):>9}")

print()
print("A reviewer helps the model only when catch/override exceeds the model's")
print("own odds of being right (eq:review-helps-only-when-catch-exceeds-override-odds).")

print()
print()
print("Which means the bar rises as the model improves.")
print()
C_FIX, F_FIX = 0.35, 0.03
print(f"holding the reviewer fixed at catch={C_FIX:.2f}, override={F_FIX:.2f}")
print()
print(f"{'model accuracy':>18}{'odds of right':>16}{'team accuracy':>16}"
      f"{'change':>11}{'review is':>13}")
print("-" * 74)
for a_m in (0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.99):
    t = team(a_m, C_FIX, F_FIX)
    d = t - a_m
    print(f"{a_m:>18.2f}{a_m / (1 - a_m):>16.1f}{t:>16.4f}"
          f"{d * 100:>+10.2f}p{('helping' if d > 0 else 'harmful'):>13}")

BREAK = (C_FIX / F_FIX) / (1 + C_FIX / F_FIX)
print()
print(f"break-even model accuracy: {BREAK:.3f}")
print("above that, this review process makes the system worse")

print()
print()
print("What the reviewer is given to review with.")
print()
# (what the reviewer sees, false-override rate, catch rate, self-reported confidence)
SHOWN = [
    ("the output alone",                       0.030, 0.29, 0.54),
    ("output + model confidence, calibrated",  0.028, 0.41, 0.63),
    ("output + confidence, overconfident",     0.019, 0.24, 0.71),
    ("output + a faithful explanation",        0.026, 0.47, 0.69),
    ("output + a plausible wrong explanation", 0.014, 0.19, 0.78),
    ("output + the source documents",          0.031, 0.52, 0.66),
]
A_M = 0.88
print(f"{'what the reviewer sees':>42}{'override':>11}{'catch':>9}"
      f"{'team acc':>11}{'confidence':>13}")
print("-" * 86)
shown_acc = {}
for name, f, c, conf in SHOWN:
    t = team(A_M, c, f)
    shown_acc[name] = t
    print(f"{name:>42}{f:>11.3f}{c:>9.2f}{t:>11.4f}{conf:>13.2f}")

best_shown = max(shown_acc, key=lambda n: shown_acc[n])
print()
print(f"best: {best_shown} at {shown_acc[best_shown]:.4f}")

print()
print()
print("Confidence and accuracy do not move together.")
print()
print(f"{'what the reviewer sees':>42}{'confidence':>13}{'team accuracy':>16}"
      f"{'confidence per point':>23}")
print("-" * 94)
BASE_CONF = SHOWN[0][3]
BASE_ACC = shown_acc[SHOWN[0][0]]
for name, f, c, conf in SHOWN:
    d_conf = conf - BASE_CONF
    d_acc = shown_acc[name] - BASE_ACC
    if abs(d_acc) < 1e-9:
        rs = f"{'(baseline)':>23}"
    else:
        rs = f"{d_conf / d_acc:>23.1f}"
    print(f"{name:>42}{conf:>13.2f}{shown_acc[name]:>16.4f}{rs}")

WRONG = "output + a plausible wrong explanation"
WRONG_CONF = [s[3] for s in SHOWN if s[0] == WRONG][0]
print()
print(f"a plausible wrong explanation raises confidence from {BASE_CONF:.2f}"
      f" to {WRONG_CONF:.2f}")
print(f"and moves team accuracy from {BASE_ACC:.4f} to {shown_acc[WRONG]:.4f}")

print()
print()
print("And what pointing the reviewer at the right items is worth.")
print()
print(f"{'routing policy':>36}{'reviewed':>11}{'errors covered':>17}"
      f"{'team accuracy':>16}{'minutes / 1k':>15}")
print("-" * 95)
MIN_PER_ITEM = 1.5
ROUTING = [
    ("review everything",                   1.00, 0.35, 0.030),
    ("review the bottom 20% by confidence",  0.20, 0.61, 0.021),
    ("review the bottom 5% by confidence",   0.05, 0.74, 0.017),
    ("oracle: review exactly the errors", 1 - A_M, 0.35, 0.000),
]
ORACLE = "oracle: review exactly the errors"
ALL = "review everything"
NARROW = "review the bottom 5% by confidence"
routed = {}
for name, share, c, f in ROUTING:
    covered = 1.0 if name == ORACLE else min(1.0, share * 2.6)
    t = team(A_M, c * covered, f * share)
    routed[name] = (share, covered, t, share * MIN_PER_ITEM * 1000)
    print(f"{name:>36}{share:>11.0%}{covered:>17.0%}{t:>16.4f}"
          f"{share * MIN_PER_ITEM * 1000:>15.0f}")

BEST_ROUTE = max((n for n, s, c, f in ROUTING if n != ORACLE),
                 key=lambda n: routed[n][2])
print()
print(f"best practical policy: {BEST_ROUTE} at {routed[BEST_ROUTE][2]:.4f}")
print(f"using {routed[ALL][3] / routed[BEST_ROUTE][3]:.0f}x less reviewer time"
      f" than reviewing everything")
print(f"perfect routing is worth {(routed[ORACLE][2] - A_M) * 100:.1f} points over the model")

print(f"""
The first table is the measurement almost nobody makes. Across {len(TASKS)} tasks, the
model-proposes / human-disposes team beats the model alone in {beats_model} of {len(TASKS)}
cases -- and beats **the better of its two members in only {beats}**, with a mean gain over that
better member of {gain_better / len(TASKS) * 100:+.2f} points.

That is not an argument against human oversight. It is an argument against assuming it. The
usual claim is that adding a reviewer can only help, and the arithmetic says otherwise: a
reviewer both catches errors and creates them, and which effect dominates is an empirical
question with a closed-form answer.

The second table gives the form. Team accuracy is `a_m(1 - f) + (1 - a_m)c`, so the reviewer
improves on the model exactly when `c/f > a_m/(1 - a_m)`
(eq:review-helps-only-when-catch-exceeds-override-odds). The reviewer's catch-to-override ratio
has to beat the model's own odds of being right.

`fraud alert disposition` fails that test. The model is right {0.94:.0%} of the time, so the bar
is {0.94 / 0.06:.1f}, and the reviewer's ratio is {0.22 / 0.04:.1f}. Every hour spent reviewing
those alerts makes the system slightly worse, and it looks exactly like diligence.

The third table is why this gets harder rather than easier. Holding a fixed reviewer at
catch={C_FIX:.2f} and override={F_FIX:.2f}, review helps up to a model accuracy of
**{BREAK:.3f}** and hurts above it. **The better the model gets, the harder it is for a human to
add anything** -- the reviewer's own mistakes are drawn from a pool that grows while the errors
they might catch are drawn from one that shrinks.

That is the same structure as ch:sec-permissions' approval queue reached from the opposite
direction. That chapter showed a low rejection rate trains approval; this one shows why the
rejection rate falls in the first place.

The fourth table is about what the reviewer is handed. `{best_shown}` produces the best team
accuracy at {shown_acc[best_shown]:.4f}, because it is the only row that lets the reviewer check
the claim rather than assess the presentation.

The row that matters most is `{WRONG}`. It drops the catch rate to {0.19:.2f} and the override
rate to {0.014:.3f} -- a reviewer who defers more and catches less -- while raising
self-reported confidence from {BASE_CONF:.2f} to {WRONG_CONF:.2f}.

The fifth table makes the divergence explicit: confidence rises fastest exactly where accuracy
does not (eq:an-explanation-raises-confidence-faster-than-accuracy). **An explanation is a
persuasion artefact before it is an evidence artefact**, which is ch:rai-interpretability's
`stated-reasons-need-not-be-actual-reasons` arriving as a measured effect on a person.

An overconfident confidence score does the same thing more cheaply -- {0.019:.3f} override,
{0.24:.2f} catch, {0.71:.2f} confidence. Which is why cite:guo2017calibration's calibration
result is an oversight requirement rather than a modelling nicety: an uncalibrated score is a
reliance signal pointing the wrong way, and cite:kadavath2022 is the question of whether the
model can supply a better one.

The last table has two findings.

The first is that reviewing everything is not the best use of the reviewer.
`{BEST_ROUTE}` reaches {routed[BEST_ROUTE][2]:.4f} against {routed[ALL][2]:.4f} for reviewing
everything, at {routed[BEST_ROUTE][3]:.0f} reviewer-minutes per thousand items instead of
{routed[ALL][3]:.0f} -- **higher accuracy at
{routed[ALL][3] / routed[BEST_ROUTE][3]:.0f} times less human time**, because concentrating
attention raises the catch rate and shrinks the surface on which the reviewer can introduce
errors.

Narrower is not monotonically better. Routing the bottom {0.05:.0%} covers only
{routed[NARROW][1]:.0%} of the model's errors and lands at {routed[NARROW][2]:.4f} -- below
reviewing everything. **Coverage collapses faster than precision improves**, which is
ch:ev-framework's union result in another costume.

The second finding is the harder one. The oracle row hands the reviewer *exactly* the model's
errors, with no opportunity to overturn a correct answer, and reaches {routed[ORACLE][2]:.4f} --
**{(routed[ORACLE][2] - A_M) * 100:.1f} points over the model**. A reviewer who catches
{C_FIX:.0%} of errors catches {C_FIX:.0%} of errors however well you point them.

**Routing is worth a great deal and it is not the binding constraint.** The catch rate is, and
the fourth table said what moves it: give the reviewer the sources, not the presentation.""")
```

## 9. Practical Example

Model, human, and the pair:

```
                          task    model    human     team   vs model   vs better member
---------------------------------------------------------------------------------------
           content policy call    0.910    0.940    0.926      +1.6p              -1.4p
             loan adjudication    0.870    0.790    0.884      +1.4p              +1.4p
          clinical triage note    0.830    0.880    0.892      +6.2p              +1.2p
     security review of a diff    0.710    0.760    0.759      +4.9p              -0.1p
       fraud alert disposition    0.940    0.810    0.916      -2.4p              -2.4p
```

**The team beats the better of its two members in 2 of 5 tasks**, mean gain −0.28 points.

```
                          task   catch rate   false override   ratio needed   ratio actual   helps?
---------------------------------------------------------------------------------------------------
           content policy call         0.38             0.02           10.1           19.0      yes
          clinical triage note         0.46             0.02            4.9           23.0      yes
       fraud alert disposition         0.22             0.04           15.7            5.5       no
```

Review helps only when catch/override beats the model's odds of being right
({{eq:review-helps-only-when-catch-exceeds-override-odds}}).

```
    model accuracy   odds of right   team accuracy     change    review is
--------------------------------------------------------------------------
              0.70             2.3          0.7840     +8.40p      helping
              0.90             9.0          0.9080     +0.80p      helping
              0.92            11.5          0.9204     +0.04p      helping
              0.95            19.0          0.9390     -1.10p      harmful
              0.99            99.0          0.9638     -2.62p      harmful
```

**Break-even is 0.921** — above it, this review process makes the system worse.

```
                    what the reviewer sees   override    catch   team acc   confidence
--------------------------------------------------------------------------------------
                          the output alone      0.030     0.29     0.8884         0.54
        output + confidence, overconfident      0.019     0.24     0.8921         0.71
           output + a faithful explanation      0.026     0.47     0.9135         0.69
    output + a plausible wrong explanation      0.014     0.19     0.8905         0.78
             output + the source documents      0.031     0.52     0.9151         0.66

                    what the reviewer sees   confidence   team accuracy   confidence per point
----------------------------------------------------------------------------------------------
             output + the source documents         0.66          0.9151                    4.5
        output + confidence, overconfident         0.71          0.8921                   46.2
    output + a plausible wrong explanation         0.78          0.8905                  115.4
```

**Confidence rises fastest exactly where accuracy does not**
({{eq:an-explanation-raises-confidence-faster-than-accuracy}}).

```
                      routing policy   reviewed   errors covered   team accuracy   minutes / 1k
-----------------------------------------------------------------------------------------------
                   review everything       100%             100%          0.8956           1500
 review the bottom 20% by confidence        20%              52%          0.9144            300
  review the bottom 5% by confidence         5%              13%          0.8908             75
   oracle: review exactly the errors        12%             100%          0.9220            180
```

Routing at 20% is **5× less reviewer time** at higher accuracy; perfect routing is worth
**4.2 points** and no more.

The second listing scores the arrangements.

```python {tier=A name=oversight-is-a-conjunction-of-preconditions}
"""Oversight needs authority, information, time and incentive. Missing one makes it ceremony.

The first listing asked whether a reviewer improves accuracy. This one asks what has to be true
for them to be able to.

Four things: the authority to change the outcome, information sufficient to judge it, time
enough to use that information, and an incentive that does not punish them for saying no. Those
compose as a product, not a sum -- an arrangement with three of the four produces a decision
record and no decision (eq:oversight-is-a-conjunction-of-preconditions).

The fourth is the one nobody designs. The person who rejects absorbs a delay, an escalation and
a chance of being visibly wrong; the person who approves absorbs nothing observable. The
organisation's costs run the other way
(eq:reviewers-bear-the-cost-of-rejecting-not-approving).
"""
# (arrangement, authority, information, time, incentive, share of decisions touched)
ARRANGEMENTS = [
    ("release sign-off by the owner",      0.90, 0.55, 0.70, 0.60, 1.000),
    ("reviewer approves each agent action", 0.85, 0.30, 0.20, 0.35, 1.000),
    ("a nominated accountable executive",  0.99, 0.15, 0.10, 0.45, 1.000),
    ("an ombudsman with escalation power", 0.75, 0.70, 0.65, 0.85, 0.030),
    ("appeal after an adverse decision",   0.95, 0.80, 0.85, 0.90, 0.008),
    ("a quarterly audit sample",           0.40, 0.85, 0.90, 0.75, 0.002),
]
NAMES = ["authority", "information", "time", "incentive"]

print("Four preconditions, and they multiply.")
print()
print(f"{'arrangement':>38}{'auth':>8}{'info':>8}{'time':>8}{'incent':>8}"
      f"{'product':>10}{'binding':>14}")
print("-" * 94)
prod = {}
for name, a, i, t, n, cov in ARRANGEMENTS:
    p = a * i * t * n
    prod[name] = (p, cov)
    binding = NAMES[[a, i, t, n].index(min(a, i, t, n))]
    print(f"{name:>38}{a:>8.2f}{i:>8.2f}{t:>8.2f}{n:>8.2f}{p:>10.4f}{binding:>14}")

print()
print(f"best per-item quality: {max(prod, key=lambda n: prod[n][0])}"
      f" at {max(p for p, c in prod.values()):.4f}")
print("time or information is binding in "
      f"{sum(1 for nm, a, i, t, n, c in ARRANGEMENTS if min(a, i, t, n) in (i, t))}"
      f" of {len(ARRANGEMENTS)} arrangements")

print()
print()
print("And quality is not the whole story, because coverage varies by 500x.")
print()
print(f"{'arrangement':>38}{'per-item quality':>19}{'decisions touched':>20}"
      f"{'effective oversight':>22}")
print("-" * 99)
eff = {}
for name, a, i, t, n, cov in ARRANGEMENTS:
    e = prod[name][0] * cov
    eff[name] = e
    print(f"{name:>38}{prod[name][0]:>19.4f}{cov:>20.3%}{e:>22.5f}")

best_eff = max(eff, key=lambda n: eff[n])
best_qual = max(prod, key=lambda n: prod[n][0])
print()
print(f"highest effective oversight: {best_eff} at {eff[best_eff]:.5f}")
print(f"highest per-item quality:    {best_qual} at {prod[best_qual][0]:.4f}"
      f" but {prod[best_qual][1]:.1%} coverage")

print()
print()
print("Why `time` is binding: what checking one item actually requires.")
print()
# (verification step, seconds, share of catchable errors it would catch)
STEPS = [
    ("read the model's output",                12, 0.06),
    ("read the input the model saw",           25, 0.11),
    ("check the cited source says this",       55, 0.31),
    ("record the reason for the decision",     40, 0.01),
    ("check the applicant's history",         110, 0.03),
    ("check no relevant source was omitted",  180, 0.22),
    ("re-derive the decision from policy",    240, 0.19),
    ("check consistency with past decisions", 300, 0.07),
]
TOTAL_SEC = sum(s for n, s, v in STEPS)
print(f"{'verification step':>40}{'seconds':>10}{'errors it catches':>20}"
      f"{'value per second':>19}")
print("-" * 89)
for name, sec, val in sorted(STEPS, key=lambda s: -s[2] / s[1]):
    print(f"{name:>40}{sec:>10}{val:>20.0%}{val / sec:>19.5f}")

print()
print(f"verifying one item completely: {TOTAL_SEC} seconds ({TOTAL_SEC / 60:.1f} minutes)")


def greedy(budget):
    """Highest error coverage reachable inside a time budget."""
    left, got, done = budget, 0.0, 0
    for name, sec, val in sorted(STEPS, key=lambda s: -s[2] / s[1]):
        if sec <= left:
            left -= sec
            got += val
            done += 1
    return got, done, budget - left


print()
print()
print("What a reviewer covers inside the time they are given.")
print()
print(f"{'time allowed':>16}{'steps completed':>18}{'seconds used':>15}"
      f"{'catchable errors covered':>27}")
print("-" * 76)
BUDGET = 90
cover = {}
for b in (30, 90, 180, 300, 600, TOTAL_SEC):
    got, done, used = greedy(b)
    cover[b] = got
    print(f"{b:>13} s{done:>15} of 8{used:>15}{got:>27.0%}")

print()
print(f"at the usual {BUDGET}-second budget the reviewer covers {cover[BUDGET]:.0%}"
      f" of what is catchable")
print(f"which is {BUDGET / TOTAL_SEC:.0%} of the time the item needs")

print()
print()
print("Now the precondition nobody designs: what it costs to say no.")
print()
# (action, consequence, cost to the reviewer, cost to the organisation, probability)
CONSEQ = [
    ("reject", "delays a colleague's work",          3.0,   1.0, 0.85),
    ("reject", "triggers an escalation meeting",     2.5,   2.0, 0.40),
    ("reject", "the reviewer is overruled",          4.0,   0.5, 0.55),
    ("reject", "the reviewer is visibly wrong",      6.0,   0.5, 0.30),
    ("approve", "nothing observable happens",        0.0,   0.0, 0.988),
    ("approve", "an incident is traced to this",     8.0, 400.0, 0.012),
]
print(f"{'action':>10}{'consequence':>36}{'cost to reviewer':>19}"
      f"{'cost to org':>14}{'probability':>14}")
print("-" * 93)
rev = {"reject": 0.0, "approve": 0.0}
org = {"reject": 0.0, "approve": 0.0}
for act, cons, rc, oc, p in CONSEQ:
    rev[act] += rc * p
    org[act] += oc * p
    print(f"{act:>10}{cons:>36}{rc:>19.1f}{oc:>14.1f}{p:>14.3f}")
print("-" * 93)
print(f"{'EXPECTED':>10}{'reject':>36}{rev['reject']:>19.2f}{org['reject']:>14.2f}")
print(f"{'':>10}{'approve':>36}{rev['approve']:>19.2f}{org['approve']:>14.2f}")

rev_ratio = rev["reject"] / rev["approve"]
org_ratio = org["approve"] / org["reject"]
print()
print(f"to the reviewer, rejecting costs {rev_ratio:.0f}x what approving costs")
print(f"to the organisation, approving costs {org_ratio:.1f}x what rejecting costs")
print(f"the incentives are inverted by a factor of {rev_ratio * org_ratio:.0f}")

INCIDENT = [c for a, cs, c, o, p in CONSEQ if cs.startswith("an incident")][0]
THRESHOLD = rev["reject"] / INCIDENT
print()
print(f"a reviewer minimising their own cost rejects only when they are"
      f" {THRESHOLD:.0%} sure")
print("which is why measured rejection rates sit near zero (ch:sec-permissions)")

print()
print()
print("What actually moves each precondition.")
print()
# (intervention, delta authority, info, time, incentive, reviewer reject-cost multiplier,
#  reviewer approve-cost multiplier)
FIXES = [
    ("give the reviewer the sources",        0.00, 0.45, 0.00, 0.00, 1.00, 1.00),
    ("budget 5 minutes instead of 90 s",     0.00, 0.00, 0.55, 0.00, 1.00, 1.00),
    ("reviewer outside the delivery line",   0.05, 0.00, 0.00, 0.30, 0.35, 1.00),
    ("re-adjudicate 2% of approvals",        0.00, 0.00, 0.00, 0.20, 1.00, 6.00),
    ("publish the rejection rate",           0.00, 0.00, 0.00, 0.15, 1.00, 1.00),
]
BASE = ARRANGEMENTS[1]
print(f"applied to `{BASE[0]}`, product {prod[BASE[0]][0]:.4f}, "
      f"threshold {THRESHOLD:.0%}")
print()
print(f"{'intervention':>38}{'new product':>14}{'improvement':>14}"
      f"{'new threshold':>16}")
print("-" * 82)
for name, da, di, dt, dn, mr, ma in FIXES:
    a = min(1.0, BASE[1] + da)
    i = min(1.0, BASE[2] + di)
    t = min(1.0, BASE[3] + dt)
    n = min(1.0, BASE[4] + dn)
    p = a * i * t * n
    thr = min(1.0, (rev["reject"] * mr) / (INCIDENT * ma))
    print(f"{name:>38}{p:>14.4f}{p / prod[BASE[0]][0]:>13.1f}x{thr:>16.0%}")

a = min(1.0, BASE[1] + sum(f[1] for f in FIXES))
i = min(1.0, BASE[2] + sum(f[2] for f in FIXES))
t = min(1.0, BASE[3] + sum(f[3] for f in FIXES))
n = min(1.0, BASE[4] + sum(f[4] for f in FIXES))
ALL_P = a * i * t * n
ALL_THR = min(1.0, rev["reject"] * 0.35 / (INCIDENT * 6.0))
print("-" * 82)
print(f"{'all five together':>38}{ALL_P:>14.4f}"
      f"{ALL_P / prod[BASE[0]][0]:>13.1f}x{ALL_THR:>16.0%}")

print(f"""
The first table is the audit this chapter exists for. Six oversight arrangements, scored on the
four things a reviewer needs, and the scores **multiply**
(eq:oversight-is-a-conjunction-of-preconditions). The best per-item quality in the table is
{max(p for p, c in prod.values()):.4f}; the worst, `{ARRANGEMENTS[2][0]}`, is
{prod[ARRANGEMENTS[2][0]][0]:.4f} despite near-perfect authority, because authority without
information or time is a signature.

Time or information is the binding constraint in
{sum(1 for nm, a2, i2, t2, n2, c in ARRANGEMENTS if min(a2, i2, t2, n2) in (i2, t2))} of
{len(ARRANGEMENTS)} arrangements, and neither is what organisations discuss when they design
oversight. They discuss authority, which is almost never the constraint.

The second table adds coverage and produces the trade-off that makes this hard. The
high-quality arrangements -- appeal, audit, ombudsman -- touch between {0.002:.1%} and
{0.03:.0%} of decisions. The full-coverage arrangements are the low-quality ones.
`{best_eff}` wins on the product at {eff[best_eff]:.5f}, and `{best_qual}` has the best per-item
quality at {prod[best_qual][0]:.4f} while touching {prod[best_qual][1]:.1%} of decisions.

**An appeal process is excellent oversight of almost nothing.** That is not an argument against
appeals; it is an argument against counting them as the oversight of the system.

The step table is why `time` is binding, and it is the most concrete thing in this chapter.
Verifying one item completely takes {TOTAL_SEC} seconds -- {TOTAL_SEC / 60:.1f} minutes -- and
the checks that catch the most errors are the expensive ones: `check no relevant source was
omitted` at {180} seconds catches {0.22:.0%}, and `re-derive the decision from policy` at
{240} seconds catches {0.19:.0%}.

At the usual {BUDGET}-second budget, spending optimally, the reviewer covers **{cover[BUDGET]:.0%}
of what is catchable** in {BUDGET / TOTAL_SEC:.0%} of the time the item needs. Doubling the
budget to {180} seconds reaches only {cover[180]:.0%}, because the next genuinely useful check
costs {180} seconds on its own and does not fit. {300} seconds reaches {cover[300]:.0%};
{cover[600]:.0%} coverage needs {600} seconds, roughly {600 / BUDGET:.0f} times the usual
budget.

**Review time buys coverage in steps rather than smoothly**, so a small budget increase is
often worth exactly nothing, and the right question is not "how long do reviewers get" but
"which check are we trying to afford".

The consequence table is the precondition nobody designs. To the reviewer, rejecting has an
expected personal cost of {rev['reject']:.2f} and approving {rev['approve']:.2f} -- **rejecting
costs {rev_ratio:.0f} times what approving costs**. To the organisation the ordering reverses:
approving costs {org['approve']:.2f} against {org['reject']:.2f}, a factor of
{org_ratio:.1f}.

**The incentives are inverted by a factor of {rev_ratio * org_ratio:.0f}**
(eq:reviewers-bear-the-cost-of-rejecting-not-approving), and a reviewer minimising their own
cost rejects only when they are **{THRESHOLD:.0%} sure**. That is the mechanism behind
ch:sec-permissions' measured rejection rates, and it is not a statement about the character of
reviewers. It is arithmetic they did not choose.

The last table is what moves it. `give the reviewer the sources` multiplies the arrangement's
product by {(min(1.0, BASE[1]) * min(1.0, BASE[2] + 0.45) * BASE[3] * BASE[4]) / prod[BASE[0]][0]:.1f}
and does nothing to the threshold. `re-adjudicate 2% of approvals` does the reverse: it leaves
the product nearly unchanged and drops the certainty threshold from {THRESHOLD:.0%} to
{min(1.0, rev['reject'] / (INCIDENT * 6.0)):.0%}, because it makes a bad approval observable.

Together the five reach a product of {ALL_P:.4f} -- {ALL_P / prod[BASE[0]][0]:.1f} times the
baseline -- and a threshold of {ALL_THR:.0%}.

**Capability and incentive are separate levers and each fails without the other.** Giving a
reviewer everything they need to catch a problem does not make it rational for them to say so,
and making rejection safe does not give them time to know when to.""")
```

```
                           arrangement    auth    info    time  incent   product       binding
----------------------------------------------------------------------------------------------
         release sign-off by the owner    0.90    0.55    0.70    0.60    0.2079   information
   reviewer approves each agent action    0.85    0.30    0.20    0.35    0.0179          time
     a nominated accountable executive    0.99    0.15    0.10    0.45    0.0067          time
      appeal after an adverse decision    0.95    0.80    0.85    0.90    0.5814   information
              a quarterly audit sample    0.40    0.85    0.90    0.75    0.2295     authority
```

**Time or information binds in 5 of 6** ({{eq:oversight-is-a-conjunction-of-preconditions}}).

```
                           arrangement   per-item quality   decisions touched   effective oversight
---------------------------------------------------------------------------------------------------
         release sign-off by the owner             0.2079            100.000%               0.20790
    an ombudsman with escalation power             0.2901              3.000%               0.00870
      appeal after an adverse decision             0.5814              0.800%               0.00465
              a quarterly audit sample             0.2295              0.200%               0.00046
```

```
                       verification step   seconds   errors it catches   value per second
-----------------------------------------------------------------------------------------
        check the cited source says this        55                 31%            0.00564
    check no relevant source was omitted       180                 22%            0.00122
      re-derive the decision from policy       240                 19%            0.00079

    time allowed   steps completed   seconds used   catchable errors covered
----------------------------------------------------------------------------
           90 s              2 of 8             67                        37%
          180 s              4 of 8            132                        49%
          600 s              6 of 8            552                        90%
          962 s              8 of 8            962                       100%
```

Verifying one item completely takes **962 seconds — 16.0 minutes**; the 90-second budget covers
**37%**.

```
    action                         consequence   cost to reviewer   cost to org   probability
---------------------------------------------------------------------------------------------
    reject           delays a colleague's work                3.0           1.0         0.850
    reject           the reviewer is overruled                4.0           0.5         0.550
   approve          nothing observable happens                0.0           0.0         0.988
   approve       an incident is traced to this                8.0         400.0         0.012
---------------------------------------------------------------------------------------------
  EXPECTED                              reject               7.55          2.07
                                       approve               0.10          4.80
```

**Rejecting costs the reviewer 79× what approving costs**; the inversion is **182**, and the
rational rejection threshold is **94% sure**
({{eq:reviewers-bear-the-cost-of-rejecting-not-approving}}).

```
                          intervention   new product   improvement   new threshold
----------------------------------------------------------------------------------
         give the reviewer the sources        0.0446          2.5x             94%
      budget 5 minutes instead of 90 s        0.0669          3.7x             94%
    reviewer outside the delivery line        0.0351          2.0x             33%
         re-adjudicate 2% of approvals        0.0281          1.6x             16%
----------------------------------------------------------------------------------
                     all five together        0.5063         28.4x              6%
```

## 10. Production Considerations

Measure the team, not the arrangement. Model accuracy, human-alone accuracy, catch rate,
override rate. Four numbers, and almost nobody has them.

Re-measure after every model upgrade. Break-even moves with $a_m$, and a review process that was
worthwhile can become net-negative with nothing in the pipeline changing.

Give reviewers sources, not explanations. Sources produce the highest catch rate and the lowest
confidence-per-point; a generated rationale produces the reverse.

Never show an uncalibrated confidence score to a reviewer. It is a reliance signal, and an
uncalibrated one points the wrong way.

Route by confidence at around 20%, not 5% and not 100%. Coverage collapses faster than precision
improves.

Score every oversight arrangement on all four preconditions and fix the minimum. Authority is
almost never the minimum and is almost always what gets discussed.

Budget review time against the actual verification steps. If the binding check costs 180 seconds,
a 120-second budget buys nothing that a 90-second budget did not.

Make approvals observable. Re-adjudicating 2% of approvals moved the rejection threshold from
94% to 16% — more than any capability intervention in the table.

Put reviewers outside the delivery reporting line. Most of the personal cost of rejecting is
social, and that one change moved the threshold to 33%.

## 11. Common Mistakes

**Treating "a human reviews it" as a control.** It is a hypothesis with four preconditions and
an accuracy condition.

**Assuming a reviewer can only help.** They overturn correct answers too, and above 0.921 model
accuracy this particular reviewer nets negative.

**Showing a generated rationale to increase trust.** It does, and it decreases accuracy.

**Designing for authority.** It is the binding constraint in one of six arrangements.

**Counting an appeal process as the system's oversight.** Excellent quality over 0.8% of
decisions.

**Exhorting reviewers to be more diligent.** The threshold is 94% because of the cost structure,
not the attitude.

**Increasing review time by 30%.** Coverage moves in steps; that buys nothing.

## 12. Failure Modes

**Oversight silently inverts after a model upgrade.** Break-even crosses, the pipeline is
unchanged, and the reviewers are now a net source of error.

**The explanation feature ships and the catch rate falls.** Confidence rises, complaints fall,
and nobody connects the two.

**A named accountable executive with no information.** Product 0.0067, near-perfect authority,
and it satisfies the obligation exactly.

**A 90-second SLA on a 16-minute item.** 37% coverage, presented as full review.

**A rejection rate of 0.4% read as evidence the system is good.** It is evidence of a 94%
threshold.

**Routing narrowed to save reviewer time.** 5% routing lands below reviewing everything.

## 13. Alternatives

**Remove the reviewer and improve the model.** Rational whenever $c/f$ is below the model's odds
of correctness — the fraud queue's honest answer, and it frees the reviewer for the queue where
they clear the bar.

**Reviewer as a sampler rather than a gate.** Give up per-item authority, keep measurement; the
audit-sample row's 0.2295 quality, and the finding feeds back into the model.

**Two independent reviewers on routed items.** Raises the catch rate on the small routed set;
costs twice, and only pays if disagreement is adjudicated rather than averaged.

**Defer-to-human on abstention rather than on confidence.** Requires the model to know when it
does not know, which is {{cite:kadavath2022}}'s question and a strictly stronger requirement than
{{cite:guo2017calibration}}'s calibration.

**Post-hoc appeal instead of pre-decision review.** Highest per-item quality in the table at
0.5814, and 0.8% coverage — a complement to oversight and not a substitute.

## 14. Evaluation

Run a re-adjudication study: sample decisions, have an expert re-decide blind, and estimate $c$
and $f$ directly. Without those two numbers no claim about oversight is testable.

Compute $c/f$ against $a_m/(1-a_m)$ per queue, not per system. The answer differs by queue and
the fraud row shows it can be negative.

A/B the reviewer's artefact — output alone, sources, explanation — and measure accuracy and
self-reported confidence separately. The gap between them is the finding.

Time the verification steps for your own items and compare against the SLA. Most teams have
never priced complete verification.

Measure the rejection rate and, separately, ask reviewers how sure they would need to be. The
gap between their stated threshold and 94% tells you how much of the cost structure they can see.

## 15. Advanced Concepts

The team model assumes the reviewer's override is correct whenever the model was wrong, which
inflates $c$'s value. In reality an override on a genuinely wrong item can substitute a second
wrong answer, and the correct term is $c \cdot q$ for the reviewer's conditional accuracy $q$ on
items they choose to override. That strictly raises the bar in
{{eq:review-helps-only-when-catch-exceeds-override-odds}}, and it means published catch rates
that do not condition on the override being *correct* overstate the arrangement.

There is a selection effect in the routing result that a production system will hit. Routing by
confidence sends the reviewer a non-representative sample — low-confidence items are harder,
so the reviewer's own accuracy on them is lower than their average. The 0.9144 figure assumes
the reviewer's $c$ and $f$ are constant across the confidence distribution, and they are not. The
correct measurement conditions $c$ and $f$ on the routing bucket, and the honest expectation is
that routing's advantage is real but smaller than the naive calculation suggests.

The most important limitation is what the accuracy frame leaves out. Everything in this chapter
prices oversight against a metric the system already optimises, and on a high-accuracy system
that frame will always recommend removing the human. But the errors that matter most are
frequently the ones outside the metric — a category the evaluation set never contained, a harm
that is not a misclassification, an input the system was never designed to receive. A reviewer
contributes nothing to $c$ on those and may be the only thing standing between the system and an
incident. **Measuring oversight only against in-distribution accuracy will reliably recommend
deleting the control that handles out-of-distribution failure**, and the right response is to
measure the two purposes separately rather than to abandon measurement.

Finally, the incentive result suggests an arrangement rarely used: pay reviewers on calibrated
accuracy rather than throughput, scoring both wrong approvals and wrong rejections against a
delayed expert re-adjudication. That converts an unobservable approval into an observable one
and, per the intervention table, is the class of change that moves the threshold. It also
imports every measurement problem {{ch:ev-framework}} raised, and a badly specified reviewer
score will be optimised exactly as badly as a badly specified model objective.

## 16. Connection to Previous Chapters

{{eq:approval-quality-falls-with-volume}} from {{ch:sec-permissions}} and this chapter's time
budget are the same constraint measured in different units: that chapter's volume is this
chapter's seconds per item.

{{eq:a-low-rejection-rate-trains-approval}} from the same chapter is explained here — the
threshold is **94%** because of a cost structure the reviewer did not choose.

{{eq:stated-reasons-need-not-be-actual-reasons}} from {{ch:rai-interpretability}} becomes a
measured harm: the wrong-explanation row raises confidence to 0.78 and drops the catch rate to
0.19.

{{eq:an-explanation-serves-one-audience}} from the same chapter is why the reviewer's artefact
is source documents and the regulator's is a model card, and why one document cannot be both.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is the routing result: the bottom
5% covers 13% of errors and lands below reviewing everything.

## 17. Exercises

1. Estimate $c$ and $f$ for one of your review queues by blind re-adjudication. Does the queue
   clear its bar?

2. Compute break-even model accuracy for that reviewer. How far is your current model from it?

3. A/B the reviewer's artefact and measure accuracy and confidence separately. What is your
   confidence-per-point for each artefact?

4. Score your oversight arrangements on all four preconditions. What is the minimum, and is it
   what you have been investing in?

5. Time the verification steps for one item type and compute coverage at your current SLA.

6. Recompute {{eq:review-helps-only-when-catch-exceeds-override-odds}} with the conditional
   override accuracy $q$ from {{sec:15-advanced-concepts}}. How much does the bar move?

## 18. Interview Questions

1. Does adding a human reviewer to this pipeline make it more accurate? How would you know?

2. Our model improved from 0.90 to 0.96. What happened to the value of our review step?

3. Why might showing the reviewer the model's reasoning make the system worse?

4. Our rejection rate is 0.3%. What does that tell you?

5. We have a named accountable executive for this system. How much oversight is that?

6. We want to reduce review cost by 30%. What is the right way to do it?

## 19. Research Questions

1. How do reviewer catch and override rates vary across the confidence distribution in
   production systems?

2. Does an explanation's faithfulness, measured independently, predict its effect on reviewer
   accuracy?

3. What incentive designs measurably move rejection thresholds without producing over-rejection?

4. Can oversight value on out-of-distribution failures be measured at all, and by what
   instrument?

## 20. Chapter Summary

*A human reviews the output* is not a control. It is a hypothesis with an accuracy condition and
four preconditions, and it fails in every direction when measured.

The accuracy condition: a reviewer catches errors and creates them, so review helps only when
$c/f > a_m/(1-a_m)$ ({{eq:review-helps-only-when-catch-exceeds-override-odds}}). Across five
tasks the team beat the model in 4 of 5 and **the better of its two members in only 2**, mean
gain **−0.28 points**. The fraud queue's bar is **15.7** against a reviewer ratio of **5.5**.
And the bar rises with the model: break-even at **0.921**, so a review step can invert after an
upgrade with nothing else changing.

What the reviewer is shown matters more than anything else. Source documents give the best team
accuracy at **0.9151**; a plausible wrong explanation gives **0.19** catch and **0.78**
confidence — **115.4** points of confidence per point of accuracy against **4.5** for sources
({{eq:an-explanation-raises-confidence-faster-than-accuracy}}). Routing at 20% beats reviewing
everything at **5× less time**, and perfect routing is worth **4.2 points** and no more, because
the catch rate binds.

The preconditions multiply ({{eq:oversight-is-a-conjunction-of-preconditions}}). Time or
information binds in **5 of 6** arrangements; authority almost never does and is almost always
what gets designed. Verifying one item takes **16.0 minutes** against 90 seconds, covering
**37%**. And an appeal process scores the best per-item quality in the table over **0.8%** of
decisions.

Then the arithmetic nobody designs. Rejecting costs the reviewer **79×** what approving costs
while costing the organisation **2.3× less** — an inversion of **182** and a **94%** certainty
threshold ({{eq:reviewers-bear-the-cost-of-rejecting-not-approving}}). That is every near-zero
rejection rate ever measured, and it is not about diligence.

The fixes split cleanly. Sources and time buy capability and leave the threshold at 94%; moving
the reviewer out of the delivery line and re-adjudicating 2% of approvals buy incentive and drop
it to 33% and 16%. Together, **28× the baseline product and a 6% threshold**.

Carry forward: **measure the team, not the arrangement**, and **fix the smallest of the four
factors** — because they multiply.

## 21. Further Reading

- {{cite:turpin2023faithfulness}} — reasoning that does not reflect the actual computation,
  which is the wrong-explanation row's mechanism.
- {{cite:guo2017calibration}} — calibration as a reliance signal; an uncalibrated score points
  the reviewer the wrong way.
- {{cite:kadavath2022}} — whether a model can supply the abstention signal routing wants, which
  is stronger than calibration.
- {{cite:ribeiro2016lime}} — local explanations, and what they can and cannot let a reviewer
  check.
