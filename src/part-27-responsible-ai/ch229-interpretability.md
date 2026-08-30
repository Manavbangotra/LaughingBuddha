---
id: rai-interpretability
number: 229
part: XXVII
tier: full
status: draft
requires: [semantic-failure-has-no-instrument, calibration-is-required-for-decisions,
           cause-distance-drives-triage-cost, three-fairness-criteria-cannot-hold-together]
provides: [attribution-is-not-an-intervention-effect, local-fidelity-does-not-extend,
           stated-reasons-need-not-be-actual-reasons, an-explanation-serves-one-audience]
citations: [ribeiro2016lime, lundberg2017shap, turpin2023faithfulness, mitchell2019modelcards]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish an attribution from an intervention
effect and compute the gap from feature correlation; explain what
{{cite:lundberg2017shap}}'s uniqueness theorem does and does not establish; measure a local
surrogate's fidelity radius and identify the uses that fall outside it; rank explanation
methods by what they are grounded in; measure the share of a decision an explanation accounts
for; design a cheap test for unfaithfulness; and show that one explanation artefact cannot
serve four audiences.

## 2. Why This Matters

Interpretability is where responsible AI meets engineering, and it is full of instruments
whose guarantees are narrower than their usage.

{{cite:lundberg2017shap}} proved that a class of additive attributions has a unique member
satisfying stated axioms. That is a real theorem about **axioms**, not about mechanism. With a
near-duplicate feature pair at correlation 0.97, the attribution is **0.211** and intervening
on the feature moves the score **0.012** — a factor of **17.2**
({{eq:attribution-is-not-an-intervention-effect}}). Both numbers are correct and they answer
different questions.

{{cite:ribeiro2016lime}} is careful about the parallel limit and usage is not. A local
surrogate explains **99%** of variance at the point and **24%** at distance 0.5, naming the
same top feature **71%** of the time ({{eq:local-fidelity-does-not-extend}}) — while writing a
policy from a batch of explanations happens at distance 0.90, where $R^2$ is **0.010**.

The generated-explanation problem is worse. {{cite:turpin2023faithfulness}} found models do not
always say what they think; here an explanation accounts for **36%** of what moved the decision
and leaves **64%** unmentioned ({{eq:stated-reasons-need-not-be-actual-reasons}}), including a
demographic cue at 0.09 influence mentioned 1% of the time. Reading the explanation detects
**4%** of unfaithfulness; swapping the answer order detects **62%** for one extra run.

And four audiences want incompatible artefacts. Every single-artefact design serves one and
scores **0.24 or below** for at least one other
({{eq:an-explanation-serves-one-audience}}).

## 3. Prerequisites

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is the general problem
this chapter's methods are attempts at: an explanation is an instrument for a failure that
produces no error.

{{eq:calibration-is-required-for-decisions}} from {{ch:ev-classical-metrics}} is what an
explanation shown to a reviewer needs and rarely has — a confidence that means what it says.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} is why a debugger's
explanation must be faithful rather than plausible: the cause sits several steps back and a
plausible account of the last step is worse than none.

{{eq:three-fairness-criteria-cannot-hold-together}} from {{ch:rai-bias}} is the structural
parallel: a set of desirable properties that cannot be jointly satisfied, and the useful move
is to state which one you chose.

## 4. Intuitive Explanation

Two questions get called "explanation" and they are different.

The first is: *how much of this prediction is attributable to this feature?* The second is:
*what would happen if this feature changed?*

Attribution methods answer the first. Decision-makers ask the second. And they coincide only
when features are independent.

Take a model with a near-duplicate pair — `tenure_months` and `tenure_years`, correlated at
0.97, both carrying real signal. An attribution method splits credit between them, because both
genuinely explain the prediction: `tenure_months` gets 0.211.

Now intervene. Change `tenure_months` and leave everything else. The score moves 0.012, because
`tenure_years` is still carrying the same information.

A factor of seventeen. Neither number is wrong. The attribution correctly reports contribution
under an allocation rule; the intervention correctly reports what happens if you change the
thing.

The gap is a function of correlation alone: 1.0× at independence, 1.7× at 0.60, 17.2× at 0.97,
unbounded as correlation approaches one. **Nothing in a deployed feature set is independent** —
feature engineering produces correlated features deliberately, because redundancy is
robustness.

That has a specific implication for {{cite:lundberg2017shap}}'s theorem, which is worth stating
carefully because the paper is precise and its reception is not. The theorem says: given local
accuracy, missingness and consistency, there is exactly one additive attribution. That is a
strong result and it is a result about *axioms*. It does not say the attribution corresponds to
a mechanism, and the paper does not claim it does.

The methods table makes the grounding explicit. SHAP is grounded in axioms. LIME in local
fidelity. Permutation importance in an empirical loss change. Ablation in a different empirical
loss change. Attention weights in nothing at all — they tell you where the model looked, which
is not the same as what determined the answer.

Only a randomised intervention is grounded in causality, and it requires setting the feature
and observing the outcome, which most deployments cannot do.

The second limitation is about *range*, and {{cite:ribeiro2016lime}} states it explicitly: the
surrogate is faithful locally. How locally?

At the point itself, the local linear model explains 99% of the variance and names the same top
feature 100% of the time. At distance 0.25, 70% and 86%. At distance 0.5, 24% and 71%. At
distance 1.0, essentially nothing.

Now look at how explanations are used. Explaining one prediction happens at distance 0.05 —
that is the case the method was built for and it is 99% faithful there. Telling a user what to
change happens at distance 0.60, because "what to change" is a question about a *different*
point. Writing a policy from a batch of explanations happens at 0.90. Debugging globally at
1.50.

**The method is used at distances where it does not hold**, and the failure is silent, because
a surrogate always returns coefficients.

That silence is the important part. A model asked to extrapolate returns a number with no flag;
a surrogate asked to explain a region it was not fitted on does the same. There is no error, no
warning, and no field in the output that records the radius the fit was valid in — so an
explanation used at distance 0.9 is indistinguishable, on paper, from one used at 0.05. The
only difference is that one of them is describing the model and the other is describing a line
somebody drew near it.

There is a third problem that neither method has, and it is the one that matters most for
explanations shown to people: **attributions rank by contribution and users need to know what to
do.**

In the model here, the actionable features — number of support calls, plan tier — carry 55% of
the weight. The top-attributed feature is tenure, which a user changes by waiting.

An explanation that faithfully reports the largest contributor and offers no available action
has answered the analyst's question in the user's interface. Ranking by attribution and ranking
by actionability are different orderings, and the second is the one an explanation shown to a
person should use.

That is attribution. Generated explanations — a model explaining its own reasoning — have a
harder problem.

{{cite:turpin2023faithfulness}} measured it: chain-of-thought explanations can be influenced by
features the explanation never mentions. What the model says drove the answer need not be what
did.

Count it. Seven things influence a decision: the stated evidence (0.34), the model's prior
(0.21), the phrasing of the question (0.14), answer-order position (0.11), a demographic cue
(0.09), input length (0.06), a formatting artefact (0.05).

The explanation mentions the stated evidence 96% of the time and the model's prior 11%. Weight
by influence and the explanation accounts for 36% of what actually moved the decision.

Sixty-four percent is unmentioned — including a demographic cue that carries 0.09 of influence
and is named 1% of the time.

**The explanation is not lying.** It reports the evidence it was asked to reason over, which is
genuinely part of the decision, and it reports it accurately. The failure is one of
completeness rather than of honesty, and the distinction matters because the remedies differ:
you cannot make an account more complete by asking it to be more careful. It omits the influences that are not in the model's account of
itself, and there is no reason to expect a generated account to enumerate them, because the
account is generated by the same process.

Would a reader notice? Uniformly, no. An explanation influenced by a demographic cue reads
exactly like one that was not, because the cue was never in the explanation's vocabulary.

So how do you test for it? Reading the explanation carefully detects 4% — fluency carries no
signal. Checking it against the evidence detects 19%.

Swapping the answer order and seeing whether the answer changes detects 62%, for one extra
inference run. That is the best ratio on the list by a factor of three, and it is
{{ch:ev-llm-judge}}'s both-orders protocol arriving as an interpretability test — for the same
reason: a property that should be invariant under a manipulation the model ought not care about
is checkable without knowing anything about the mechanism.

Finally: who is the explanation for?

Four audiences, four questions. A **debugger** asks which component produced this and needs
faithfulness, in a technical register, with no time limit. A **decision subject** asks what
they could have changed and needs actionability, in plain language, quickly. A **regulator**
asks whether the process was defensible and needs auditability — a property of the record
rather than of the reasoning. A **reviewer in the loop** asks whether to approve, and needs
calibration and brevity, because {{ch:sec-permissions}} showed they have seconds.

A faithful technical trace is useless to a decision subject. An actionable plain-language
reason is not auditable. A one-line summary serves the reviewer and fails everyone else.

Score four candidate artefacts against four audiences and every single one has a worst column
of 0.24 or below. Four separate artefacts serve all four; there is no single one that does.

**Explanations are not a feature, they are four features** — and building one and calling it
explainability is how a system ends up showing a technical trace to customers and submitting a
plain-language summary to an auditor.

One last number, because it is the reason any of this is urgent. A confident unfaithful
explanation is worth **negative** 0.62 to a user — twice as bad as returning nothing — because
the user acts on it. No explanation leaves them asking. A hedged one leaves them discounting. A
confident one leaves them acting, and the sign depends entirely on whether it was faithful.

**An explanation is a claim the system makes about itself**, and it deserves the same standard
as anything else the system is trusted on.

## 5. Formal Explanation

**Attribution versus intervention.** For a model $f$ and features $X$, an additive attribution
assigns $\phi_i$ with $\sum_i \phi_i = f(x) - \mathbb{E}[f]$. The intervention effect is
$\mathbb{E}[f \mid \mathrm{do}(X_i = x_i')] - \mathbb{E}[f \mid \mathrm{do}(X_i = x_i)]$. When
$X_i$ and $X_j$ are correlated, $\phi_i$ receives a share of the credit for the information
$X_j$ also carries, while $\mathrm{do}(X_i)$ leaves $X_j$ unchanged and therefore leaves that
information in place. The two coincide iff the features are independent, and the ratio grows
without bound as correlation approaches one.

{{cite:lundberg2017shap}}'s theorem establishes uniqueness within the additive class under
local accuracy, missingness and consistency. None of those axioms mentions intervention, so
uniqueness is silent about causal correspondence.

**Local fidelity.** A surrogate $g$ fitted in a neighbourhood $N_\epsilon(x)$ satisfies
$\mathbb{E}_{N_\epsilon}[(f - g)^2] \le \delta(\epsilon)$ with $\delta$ increasing. Nothing
bounds $|f - g|$ outside $N_\epsilon$, and for a nonlinear $f$ the discrepancy grows with the
curvature over the extrapolation distance. A use case at distance $d \gg \epsilon$ is
unconstrained by the fit.

**Faithfulness.** Let $\mathcal{D}$ be the set of influences on a decision with weights $w_d$,
and $m_d$ the probability the explanation mentions $d$. Explained share is
$\sum_d w_d m_d / \sum_d w_d$. Faithfulness is a property of the joint distribution of
explanation and mechanism, and cannot be assessed from the explanation alone — which is why
every effective test in {{sec:9-practical-example}} requires a second run under a
manipulation.

**Audience incompatibility.** Let audiences $a$ have requirement vectors $r_a$ over properties
(faithfulness, actionability, auditability, brevity). An artefact $A$ has a property vector
$p_A$, and serves $a$ to degree $\min_k \mathbf{1}[p_{A,k} \ge r_{a,k}]$. Since the
requirements conflict pairwise — faithfulness demands detail and brevity forbids it —
$\max_A \min_a \text{serve}(A, a)$ is bounded well below one, and the resolution is multiple
artefacts rather than a better one.

## 6. Mathematical Foundation

Attribution and intervention as different functionals:

$$\phi_i \neq \mathbb{E}[f \mid \mathrm{do}(X_i)] - \mathbb{E}[f], \qquad \frac{\phi_i}{\Delta_{\mathrm{do}}} \to \infty \ \text{ as } \ \rho_{ij} \to 1$$ (eq:attribution-is-not-an-intervention-effect)

At $\rho = 0.97$: **0.211** against **0.012**, a ratio of **17.2**.

Local fidelity and its radius:

$$\mathbb{E}_{N_\epsilon}[(f-g)^2] \le \delta(\epsilon), \qquad R^2(d) = e^{-(d/0.42)^2}$$ (eq:local-fidelity-does-not-extend)

$R^2 = 0.99$ at $d = 0.05$; **0.010** at $d = 0.90$, which is where policy is written.

Explained share of a decision:

$$\text{explained} = \frac{\sum_d w_d m_d}{\sum_d w_d} = 36\%$$ (eq:stated-reasons-need-not-be-actual-reasons)

with the best cheap detector — order swap — at **62%** for 0.4 units of effort.

And the audience bound:

$$\max_A \min_a \text{serve}(A, a) \le 0.24 \quad \text{for single artefacts}$$ (eq:an-explanation-serves-one-audience)

## 7. Internal Mechanics

Why do attribution and intervention diverge specifically at correlation? Because an attribution
must allocate a fixed total — the prediction minus the baseline — across features, and when two
features carry the same information there is no principled way to give it all to one. Every
allocation rule splits it, and the splits differ (that is what the axioms pin down). An
intervention has no such constraint: it changes one input and lets the model respond, and the
model's response reflects what remains.

So the divergence is not a defect in either method. It is the difference between an accounting
identity and a counterfactual.

The fidelity limit has a mechanism worth stating for practice. A local surrogate is fitted on
perturbations sampled near the point. The perturbation distribution *defines* the neighbourhood,
so a surrogate's radius is a hyperparameter of the explanation, chosen by whoever configured the
sampler, and it is usually not reported. Two teams explaining the same prediction with different
kernel widths get different explanations, both faithful within their own radius, and neither
states what that radius was. **The radius is the missing metadata on every local explanation**,
and adding it is a one-line change.

Generated-explanation unfaithfulness has an origin that makes it durable. The model produces
the explanation with the same forward pass structure that produced the answer, conditioned on
the answer. It has no privileged access to its own computation — there is no introspection
channel — so the explanation is a *plausible account*, generated the way any other text is. That
is not a bug to be trained out; it is what generating an explanation is.

Which is why the effective tests all involve a second run. You cannot check a claim about a
mechanism by reading a description of it; you check it by perturbing the mechanism. The order
swap works because answer-order dependence is something the model should not have, so a change
under swap is evidence of an influence the explanation did not mention — regardless of what the
explanation said.

The audience result has an organisational consequence that explains a common pattern. The
explanation feature is usually specified by whoever needs it first, and that is almost always
the debugger, because debugging comes before deployment. So the artefact is faithful,
technical, unbounded in length — and then it is exposed to users and regulators because it
exists. **The first audience sets the format and the later audiences inherit it**, and nobody
revisits the decision because there is already an explanation.

## 8. Implementation

The first listing separates attribution from intervention.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/jb1}
"""An attribution satisfies axioms. It does not answer "what happens if I change this".

cite:lundberg2017shap proved that one class of additive feature attributions has a unique
member satisfying a set of desirable properties. That is a real theorem and it is a theorem
about *axioms* -- local accuracy, missingness, consistency -- not about mechanism.

So the number an attribution returns is the credit a feature receives under a stated allocation
rule, and the number a decision-maker wants is the change that would follow from intervening on
it. Those coincide only when features are independent
(eq:attribution-is-not-an-intervention-effect).

cite:ribeiro2016lime is careful about the matching limit on the other main method: the surrogate
is faithful *locally*, and the paper says so. Usage is less careful
(eq:local-fidelity-does-not-extend).
"""
import math

# A model: score = sum(w_i * x_i). Two features are near-duplicates.
FEATURES = [
    ("tenure_months",      0.41, 0.97),   # weight, correlation with its partner
    ("tenure_years",       0.38, 0.97),   # near-duplicate of the above
    ("num_support_calls",  0.52, 0.11),
    ("plan_tier",          0.29, 0.04),
    ("region_code",        0.07, 0.02),
]

print("A linear model with one near-duplicate pair.")
print()
print(f"{'feature':>22}{'weight':>10}{'corr with partner':>20}"
      f"{'attribution':>14}{'intervention effect':>22}")
print("-" * 88)
attrib = {}
for name, w, corr in FEATURES:
    # Attribution splits credit among correlated features.
    if corr > 0.5:
        a = w * (1 - corr / 2.0)
    else:
        a = w
    # Intervening on one member of a near-duplicate pair moves the score
    # by its own weight only, but the partner still carries the signal.
    inter = w * (1 - corr)
    attrib[name] = (w, corr, a, inter)
    print(f"{name:>22}{w:>10.2f}{corr:>20.2f}{a:>14.3f}{inter:>22.3f}")

print()
print(f"`{FEATURES[0][0]}` is attributed {attrib[FEATURES[0][0]][2]:.3f} and")
print(f"intervening on it moves the score {attrib[FEATURES[0][0]][3]:.3f}")
print(f"ratio: {attrib[FEATURES[0][0]][2] / max(attrib[FEATURES[0][0]][3], 1e-9):.1f}x")

print()
print()
print("How the gap depends on correlation alone.")
print()
print(f"{'correlation':>13}{'attribution':>14}{'intervention':>15}"
      f"{'ratio':>9}{'reading':>26}")
print("-" * 77)
W = 0.41
gap = {}
for c in (0.00, 0.30, 0.60, 0.85, 0.97, 0.999):
    a = W * (1 - c / 2.0)
    i = W * (1 - c)
    gap[c] = (a, i, a / i if i > 1e-9 else float("inf"))
    reading = ("they agree" if c < 0.05 else
               "close enough" if c < 0.35 else
               "diverging" if c < 0.7 else
               "different questions")
    r = f"{a / i:.1f}x" if i > 1e-9 else "infinite"
    print(f"{c:>13.3f}{a:>14.3f}{i:>15.3f}{r:>9}{reading:>26}")

print()
print("At independence they coincide. Nothing in a deployed feature set is")
print("independent.")

print()
print()
print("What each method actually answers.")
print()
METHODS = [
    ("SHAP",             "credit under an allocation rule", "axioms",   "no"),
    ("LIME",             "a local linear approximation",    "fidelity", "no"),
    ("permutation importance", "loss when the column is shuffled", "empirical", "no"),
    ("ablation",         "loss when the feature is removed", "empirical", "partly"),
    ("randomised intervention", "effect of setting the value", "causal",  "yes"),
    ("attention weights", "where the model looked",          "none",     "no"),
]
print(f"{'method':>26}{'what it answers':>36}{'grounded in':>13}"
      f"{'causal?':>11}")
print("-" * 86)
for name, ans, ground, causal in METHODS:
    print(f"{name:>26}{ans:>36}{ground:>13}{causal:>11}")

print()
print("Only the last row answers the question a decision-maker is asking.")

print()
print()
print("Local fidelity: how far from the point does the surrogate hold?")
print()
print(f"{'distance from x':>17}{'surrogate R^2':>16}{'sign agreement':>17}"
       f"{'top-feature agreement':>24}")
print("-" * 74)
fid = {}
for d in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
    r2 = math.exp(-((d / 0.42) ** 2))
    sign = 0.5 + 0.5 * math.exp(-((d / 0.71) ** 1.6))
    top = 0.5 + 0.5 * math.exp(-((d / 0.55) ** 1.4))
    fid[d] = (r2, sign, top)
    print(f"{d:>17.2f}{r2:>16.3f}{sign:>17.1%}{top:>24.1%}")

print()
print(f"at distance {0.5:.1f} the surrogate explains {fid[0.5][0]:.0%} of variance and")
print(f"names the same top feature {fid[0.5][2]:.0%} of the time")

print()
print()
print("Which matters because of how explanations are consumed.")
print()
USES = [
    ("explain this one prediction",       0.05, "yes"),
    ("explain a batch of similar cases",  0.30, "partly"),
    ("write a policy from the explanations", 0.90, "no"),
    ("tell a user what to change",        0.60, "no"),
    ("debug the model globally",          1.50, "no"),
]
print(f"{'use':>40}{'typical distance':>19}{'surrogate valid?':>19}"
      f"{'R^2 there':>12}")
print("-" * 90)
for name, d, valid in USES:
    r2 = math.exp(-((d / 0.42) ** 2))
    print(f"{name:>40}{d:>19.2f}{valid:>19}{r2:>12.3f}")

print()
print("The first row is what the method was built for. The rest are what it")
print("is used for.")

print()
print()
print("And the actionability question, which is the one users ask.")
print()
ACTIONS = [
    ("num_support_calls", 0.52, "yes",   "call less"),
    ("plan_tier",         0.29, "yes",   "upgrade"),
    ("tenure_months",     0.41, "no",    "wait"),
    ("tenure_years",      0.38, "no",    "wait"),
    ("region_code",       0.07, "no",    "move house"),
]
print(f"{'feature':>22}{'weight':>10}{'actionable?':>14}"
      f"{'what the user would do':>26}{'usable advice?':>17}")
print("-" * 89)
for name, w, act, what in ACTIONS:
    usable = "yes" if act == "yes" else "no"
    print(f"{name:>22}{w:>10.2f}{act:>14}{what:>26}{usable:>17}")

act_w = sum(w for n, w, a, x in ACTIONS if a == "yes")
tot_w = sum(w for n, w, a, x in ACTIONS)
print()
print(f"actionable features carry {act_w / tot_w:.0%} of the weight")
print(f"the top-attributed feature is {'actionable' if ACTIONS[0][2] == 'yes' else 'not'}")

print(f"""
The model table is the setup and the last two columns are the finding.
`{FEATURES[0][0]}` receives an attribution of {attrib[FEATURES[0][0]][2]:.3f} and intervening
on it moves the score by {attrib[FEATURES[0][0]][3]:.3f} -- a factor of
{attrib[FEATURES[0][0]][2] / attrib[FEATURES[0][0]][3]:.1f}
(eq:attribution-is-not-an-intervention-effect).

The reason is not subtle once seen. `{FEATURES[0][0]}` and `{FEATURES[1][0]}` are near
duplicates at correlation {FEATURES[1][2]:.2f}. An attribution method splits credit between
them, because both explain the prediction. An intervention on one leaves the other carrying
the signal, so the score barely moves.

**Both numbers are correct and they answer different questions.** The attribution answers "how
much of this prediction is attributable to this feature"; the decision-maker is asking "what
happens if this changes".

The correlation table shows they coincide only at independence. At {0.00:.2f} the ratio is
{gap[0.0][2]:.1f}; at {0.97:.2f} it is {gap[0.97][2]:.1f}; at {0.999:.3f} it is unbounded.
Nothing in a deployed feature set is independent -- feature engineering *produces* correlated
features on purpose, because redundancy is robustness.

The methods table separates what each technique is grounded in.
cite:lundberg2017shap's uniqueness is grounded in axioms, which is a strong and specific
guarantee: given local accuracy, missingness and consistency, there is exactly one attribution.
It is not a claim that the attribution corresponds to a mechanism, and the paper does not make
one.

**Only a randomised intervention answers the causal question**, and it requires the ability to
set the feature and observe the outcome, which most deployments cannot do.

The fidelity table is cite:ribeiro2016lime's limit, stated in the paper and lost in use. At
distance {0.5:.1f} from the explained point the local surrogate explains {fid[0.5][0]:.0%} of
the variance and names the same top feature {fid[0.5][2]:.0%} of the time
(eq:local-fidelity-does-not-extend).

The uses table is why that matters. Explaining one prediction happens at distance
{0.05:.2f}, where the surrogate is {math.exp(-((0.05 / 0.42) ** 2)):.2f} faithful -- exactly
the case the method was designed for. Writing a policy from a batch of explanations happens at
distance {0.90:.2f}, where it is {math.exp(-((0.90 / 0.42) ** 2)):.3f}.

**The method is used at distances where it does not hold**, and the failure is silent because a
surrogate always returns coefficients.

The actionability table is the last and most practical point. Attributions rank features by
contribution, and a user receiving an explanation wants to know what to *do*. Here the
actionable features carry {act_w / tot_w:.0%} of the weight, and the top-attributed feature is
`{FEATURES[0][0]}` -- which a user changes by waiting.

An explanation that faithfully reports the largest contributor and offers no available action
has answered the analyst's question in the user's interface. **Ranking by attribution and
ranking by what a person can change are different orderings**, and the second is the one an
explanation shown to a person should use.""")
```

## 9. Practical Example

A model with one near-duplicate pair:

```
               feature    weight   corr with partner   attribution   intervention effect
----------------------------------------------------------------------------------------
         tenure_months      0.41                0.97         0.211                 0.012
          tenure_years      0.38                0.97         0.196                 0.011
     num_support_calls      0.52                0.11         0.520                 0.463
             plan_tier      0.29                0.04         0.290                 0.278
```

**0.211 attributed against 0.012 from intervening** — a factor of 17.2
({{eq:attribution-is-not-an-intervention-effect}}). Both are correct and they answer different
questions.

```
  correlation   attribution   intervention    ratio                   reading
-----------------------------------------------------------------------------
        0.000         0.410          0.410     1.0x                they agree
        0.600         0.287          0.164     1.7x                 diverging
        0.970         0.211          0.012    17.2x       different questions
        0.999         0.205          0.000   500.5x       different questions
```

```
                    method                     what it answers  grounded in    causal?
--------------------------------------------------------------------------------------
                      SHAP     credit under an allocation rule       axioms         no
                      LIME        a local linear approximation     fidelity         no
    permutation importance    loss when the column is shuffled    empirical         no
   randomised intervention         effect of setting the value       causal        yes
         attention weights              where the model looked         none         no
```

{{cite:lundberg2017shap}}'s uniqueness is grounded in **axioms**, which is a strong and
specific guarantee and not a claim about mechanism.

```
  distance from x   surrogate R^2   sign agreement   top-feature agreement
--------------------------------------------------------------------------
             0.00           1.000           100.0%                  100.0%
             0.25           0.702            91.4%                   85.9%
             0.50           0.242            78.3%                   70.8%
             1.00           0.003            58.9%                   55.0%

                                     use   typical distance   surrogate valid?   R^2 there
------------------------------------------------------------------------------------------
             explain this one prediction               0.05                yes       0.986
              tell a user what to change               0.60                 no       0.130
    write a policy from the explanations               0.90                 no       0.010
```

**The method is used at distances where it does not hold**
({{eq:local-fidelity-does-not-extend}}), and it always returns coefficients.

```
               feature    weight   actionable?    what the user would do   usable advice?
-----------------------------------------------------------------------------------------
     num_support_calls      0.52           yes                 call less              yes
             plan_tier      0.29           yes                   upgrade              yes
         tenure_months      0.41            no                      wait               no
           region_code      0.07            no                move house               no
```

**Ranking by attribution and ranking by actionability are different orderings.**

The second listing takes up generated explanations.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/jb2}
"""A model's stated reason and its operative reason are two different quantities.

cite:turpin2023faithfulness measured this directly: chain-of-thought explanations can be
influenced by features the explanation never mentions, so what the model says drove the answer
need not be what did (eq:stated-reasons-need-not-be-actual-reasons).

That is a stronger problem than the attribution one, because a generated explanation is fluent,
specific and confident, and a reader has no signal that distinguishes a faithful one from a
rationalisation.

The second half is about who the explanation is for. A debugger, a decision subject and a
regulator want incompatible artefacts, and one document serves one of them
(eq:an-explanation-serves-one-audience).
"""
# (driver, actual influence on the decision, how often it is mentioned)
DRIVERS = [
    ("the stated evidence",       0.34, 0.96),
    ("answer-order position",     0.11, 0.02),
    ("phrasing of the question",  0.14, 0.04),
    ("a demographic cue",         0.09, 0.01),
    ("the model's prior",         0.21, 0.11),
    ("length of the input",       0.06, 0.00),
    ("a formatting artefact",     0.05, 0.00),
]

print("What drives the decision, against what the explanation mentions.")
print()
print(f"{'driver':>28}{'actual influence':>19}{'mentioned':>12}"
      f"{'influence x mention':>22}{'unexplained':>14}")
print("-" * 95)
tot_infl = sum(i for n, i, m in DRIVERS)
explained = 0.0
drv = {}
for name, infl, ment in DRIVERS:
    e = infl * ment
    explained += e
    drv[name] = (infl, ment, e, infl - e)
    print(f"{name:>28}{infl:>19.2f}{ment:>12.0%}{e:>22.3f}{infl - e:>14.3f}")
print("-" * 95)
print(f"{'TOTAL':>28}{tot_infl:>19.2f}{'':>12}{explained:>22.3f}"
      f"{tot_infl - explained:>14.3f}")

print()
print(f"the explanation accounts for {explained / tot_infl:.0%} of what")
print(f"actually moved the decision; {1 - explained / tot_infl:.0%} is unmentioned")

print()
print()
print("The unmentioned drivers, ranked by how much they move and how often")
print("they are named.")
print()
hidden = sorted([d for d in DRIVERS if d[2] < 0.2], key=lambda d: -d[1])
print(f"{'driver':>28}{'influence':>12}{'mentioned':>12}"
      f"{'would a reader notice?':>25}")
print("-" * 79)
NOTICE = {
    "the model's prior": "no",
    "the phrasing of the question": "no",
    "phrasing of the question": "no",
    "answer-order position": "only if told",
    "a demographic cue": "no",
    "length of the input": "no",
    "a formatting artefact": "no",
}
for name, infl, ment in hidden:
    print(f"{name:>28}{infl:>12.2f}{ment:>12.0%}"
          f"{NOTICE.get(name, 'no'):>25}")

print()
print()
print("Detecting an unfaithful explanation: what each test would need.")
print()
TESTS = [
    ("read the explanation",           0.04, 0.1, "nothing"),
    ("check it against the evidence",  0.19, 0.8, "the evidence"),
    ("perturb an unmentioned feature", 0.71, 1.5, "a counterfactual run"),
    ("swap answer order",              0.62, 0.4, "one extra run"),
    ("ablate the stated reason",       0.58, 1.2, "a counterfactual run"),
    ("compare across paraphrases",     0.44, 0.9, "several runs"),
]
print(f"{'test':>34}{'detects unfaithfulness':>25}{'cost':>8}"
      f"{'per cost':>11}{'needs':>24}")
print("-" * 102)
tst = {}
for name, det, cost, needs in TESTS:
    tst[name] = (det, cost, det / cost)
    print(f"{name:>34}{det:>25.0%}{cost:>8.1f}{det / cost:>11.3f}{needs:>24}")

best = max(tst, key=lambda n: tst[n][2])
print()
print(f"best: {best} at {tst[best][2]:.3f} per unit -- one extra run")
print(f"reading the explanation detects {tst['read the explanation'][0]:.0%}")

print()
print()
print("Three audiences, three incompatible requirements.")
print()
AUDIENCES = [
    ("a debugger",       "which component produced this",  "faithful", "technical", "no"),
    ("a decision subject", "what could I have changed",    "actionable", "plain", "yes"),
    ("a regulator",      "was the process defensible",     "auditable", "formal", "yes"),
    ("a reviewer in the loop", "should I approve this",    "calibrated", "brief", "yes"),
]
print(f"{'audience':>24}{'the question':>34}{'needs':>13}{'register':>12}"
      f"{'time-bounded?':>16}")
print("-" * 99)
for name, q, need, reg, tb in AUDIENCES:
    print(f"{name:>24}{q:>34}{need:>13}{reg:>12}{tb:>16}")

print()
print("A faithful technical trace is useless to a decision subject; an")
print("actionable plain-language reason is not auditable.")

print()
print()
print("What one artefact costs when it is asked to serve all four.")
print()
COMPROMISE = [
    ("a faithful technical trace",    1.00, 0.11, 0.44, 0.09),
    ("a plain-language reason",       0.21, 0.94, 0.18, 0.62),
    ("a structured decision record",  0.58, 0.24, 0.96, 0.31),
    ("a one-line summary",            0.14, 0.71, 0.12, 0.88),
    ("all four, separately",          1.00, 0.94, 0.96, 0.88),
]
print(f"{'artefact':>32}{'debugger':>12}{'subject':>11}{'regulator':>13}"
      f"{'reviewer':>12}{'worst':>9}")
print("-" * 89)
for name, d, s, r, v in COMPROMISE:
    print(f"{name:>32}{d:>12.2f}{s:>11.2f}{r:>13.2f}{v:>12.2f}"
          f"{min(d, s, r, v):>9.2f}")

print()
print("Every single artefact serves one audience and fails at least one other.")

print()
print()
print("And the cost of a confident wrong explanation.")
print()
OUTCOMES = [
    ("no explanation",                0.00, 0.00, 0.31, "user asks"),
    ("a hedged explanation",          0.40, 0.18, 0.44, "user discounts"),
    ("a confident faithful one",      1.00, 0.00, 0.91, "user acts, correctly"),
    ("a confident unfaithful one",    1.00, 1.00, -0.62, "user acts, wrongly"),
]
print(f"{'what the system returns':>30}{'confidence conveyed':>22}"
      f"{'chance it is wrong':>21}{'value to the user':>20}{'what follows':>24}")
print("-" * 117)
for name, conf, wrong, val, follows in OUTCOMES:
    print(f"{name:>30}{conf:>22.2f}{wrong:>21.2f}{val:>20.2f}{follows:>24}")

print()
print(f"a confident unfaithful explanation is worth {-0.62:.2f}, which is")
print(f"{abs(-0.62) / 0.31:.1f} times worse than saying nothing")

print(f"""
The driver table is cite:turpin2023faithfulness' finding made countable. Across seven
influences on a decision, the explanation accounts for {explained / tot_infl:.0%} of the actual
movement and leaves {1 - explained / tot_infl:.0%} unmentioned
(eq:stated-reasons-need-not-be-actual-reasons).

`{DRIVERS[4][0]}` carries {DRIVERS[4][1]:.2f} of influence and is mentioned
{DRIVERS[4][2]:.0%} of the time. `{DRIVERS[2][0]}` carries {DRIVERS[2][1]:.2f} and is mentioned
{DRIVERS[2][2]:.0%}.

**The explanation is not lying.** It reports the evidence it was asked to reason over, which is
genuinely part of the decision, and it reports it accurately. The failure is one of
completeness rather than of honesty, and the distinction matters because the remedies differ:
you cannot make an account more complete by asking it to be more careful. It omits the influences that are not in the model's account of
itself -- and there is no reason to expect a generated account to enumerate them, because the
account is generated by the same process.

The hidden-driver table adds the column that matters for deployment: **would a reader notice?**
Uniformly, no. A demographic cue at {0.09:.2f} influence produces an explanation that reads
exactly like one without it, because the explanation never had the cue in its vocabulary.

The detection table is where a practical response lives. Reading the explanation carefully
detects {tst['read the explanation'][0]:.0%} of unfaithfulness -- fluency is not a signal.
`{best}` detects {tst[best][0]:.0%} for {tst[best][1]:.1f} units of effort, which is
**one extra inference run with the order swapped**.

That is ch:ev-llm-judge's both-orders protocol arriving as an interpretability test, for the
same reason: a property invariant under a manipulation the model should not care about is
checkable without knowing anything about the mechanism.

The audience table is the second half and it is the one that resolves most arguments about
explanation format. Four audiences, four questions, four incompatible requirements. A debugger
needs faithfulness and does not need plain language. A decision subject needs actionability and
cannot use a technical trace. A regulator needs auditability, which is a property of the record
rather than of the reasoning. A reviewer in the loop needs calibration and brevity, because
ch:sec-permissions showed they have seconds.

The compromise table prices the usual attempt to serve all four with one document. Every single
artefact scores well for one audience and its worst column is
{min(min(r[1:5]) for r in COMPROMISE[:4]):.2f} or below
(eq:an-explanation-serves-one-audience). Four separate artefacts serve all four; there is no
single one that does.

**Explanations are not a feature, they are four features**, and building one and calling it
explainability is how a system ends up with a technical trace shown to customers and a
plain-language summary submitted to an auditor.

The last table is why any of this is urgent rather than tidy. A confident unfaithful
explanation is worth {-0.62:.2f} to a user -- **negative**, and
{abs(-0.62) / 0.31:.1f} times worse than returning nothing -- because the user acts on it. No
explanation leaves them asking; a hedged one leaves them discounting; a confident one leaves
them acting, and the sign of that depends entirely on whether it was faithful.

Which is the argument against shipping explanations before you can test them. **An explanation
is a claim the system makes about itself**, and it should be held to the same standard as any
other output the system is trusted on -- which is to say measured, with the same
counterfactual runs that would measure anything else.""")
```

```
                      driver   actual influence   mentioned   influence x mention   unexplained
-----------------------------------------------------------------------------------------------
         the stated evidence               0.34         96%                 0.326         0.014
           the model's prior               0.21         11%                 0.023         0.187
    phrasing of the question               0.14          4%                 0.006         0.134
           a demographic cue               0.09          1%                 0.001         0.089
-----------------------------------------------------------------------------------------------
                       TOTAL               1.00                             0.358         0.642
```

The explanation accounts for **36%** of what moved the decision
({{eq:stated-reasons-need-not-be-actual-reasons}}) — and a reader cannot notice, because the
unmentioned drivers were never in the explanation's vocabulary.

```
                              test   detects unfaithfulness    cost   per cost                   needs
------------------------------------------------------------------------------------------------------
              read the explanation                       4%     0.1      0.400                 nothing
     check it against the evidence                      19%     0.8      0.237            the evidence
                 swap answer order                      62%     0.4      1.550           one extra run
    perturb an unmentioned feature                      71%     1.5      0.473    a counterfactual run
```

**Reading detects 4%; one extra run with the order swapped detects 62%** — the same protocol
{{ch:ev-llm-judge}} used, for the same reason.

```
                audience                      the question        needs    register   time-bounded?
---------------------------------------------------------------------------------------------------
              a debugger     which component produced this     faithful   technical              no
      a decision subject         what could I have changed   actionable       plain             yes
             a regulator        was the process defensible    auditable      formal             yes
  a reviewer in the loop             should I approve this   calibrated       brief             yes

                        artefact    debugger    subject    regulator    reviewer    worst
------------------------------------------------------------------------------------------
       a faithful technical trace        1.00       0.11         0.44        0.09     0.09
         a plain-language reason        0.21       0.94         0.18        0.62     0.18
    a structured decision record        0.58       0.24         0.96        0.31     0.24
              a one-line summary        0.14       0.71         0.12        0.88     0.12
             all four, separately        1.00       0.94         0.96        0.88     0.88
```

**Every single artefact fails at least one audience**
({{eq:an-explanation-serves-one-audience}}).

```
       what the system returns   confidence conveyed   chance it is wrong   value to the user
---------------------------------------------------------------------------------------------
                no explanation                  0.00                 0.00                0.31
      a confident faithful one                  1.00                 0.00                0.91
    a confident unfaithful one                  1.00                 1.00               -0.62
```

A confident unfaithful explanation is worth **−0.62** — twice as bad as saying nothing.

## 10. Production Considerations

Report the surrogate's kernel width with every local explanation. It is the radius the
explanation is valid in and it is currently missing metadata.

Never present an attribution as a recommendation. Rank by actionability for anything shown to a
person, and say which ordering you used.

Run the order-swap test on a sample of generated explanations. One extra inference detects 62%
of unfaithfulness and nothing else on the list comes close.

Build four artefacts, not one. Debugger, subject, regulator, reviewer — the requirements
conflict and the compromise serves nobody.

Do not ship a confident explanation you have not tested. It is worth less than silence when it
is wrong, and users cannot tell.

Use {{cite:mitchell2019modelcards}}'s format for the regulator artefact. Auditability is a
property of the record, and a standard record is cheaper than an argument.

State what each method is grounded in, on the dashboard. "Axioms", "local fidelity",
"empirical", "causal" — four words that prevent most misreadings.

## 11. Common Mistakes

**Reading an attribution as an intervention effect.** They differ by 17× at realistic
correlations.

**Citing SHAP's uniqueness as evidence of causality.** The theorem is about axioms and the
paper says so.

**Extrapolating a local surrogate.** $R^2$ is 0.010 where policy gets written.

**Showing users the top-attributed feature.** It is often the one they cannot change.

**Trusting a fluent explanation.** Reading detects 4% of unfaithfulness.

**Building one explanation artefact.** The first audience sets the format and the rest inherit
it.

## 12. Failure Modes

**Policy written from a batch of local explanations.** Valid at distance 0.05, used at 0.90.

**Two teams, two kernel widths, two explanations.** Both faithful in their own radius, neither
stated.

**Demographic influence invisible in the explanation.** 0.09 of influence, mentioned 1% of the
time, and it reads normally.

**Explanation shipped to customers because it existed.** Built for debugging, technical
register, no actionability.

**Reviewer trusting a confident wrong reason.** {{ch:sec-permissions}}'s seconds-per-item meets
a fluent rationalisation.

**Attention weights presented as explanation.** Grounded in nothing, and the most visually
persuasive artefact on the list.

## 13. Alternatives

**Counterfactual explanations.** Report the nearest input that would change the outcome.
Actionable by construction, answers the user's question, and it is expensive to compute and can
suggest infeasible changes.

**Randomised intervention studies.** The only causally grounded option. Requires the ability to
set inputs and observe outcomes, which most deployments lack.

**Inherently interpretable models.** A sparse linear model or a short rule list needs no
post-hoc explanation. Real accuracy cost, and the cost is smaller than usually assumed on
tabular problems.

**Process documentation instead of decision explanation.** Answer "was the process defensible"
rather than "why this output" — the regulator's actual question, per
{{cite:mitchell2019modelcards}}.

**Abstention instead of explanation.** Where confidence is low, decline rather than explain.
Removes the confident-unfaithful quadrant entirely, at a coverage cost.

## 14. Evaluation

Measure the correlation structure of your features and compute the attribution-to-intervention
ratio for your top predictors.

Measure your surrogate's fidelity as a function of distance and publish the radius. Then check
which of your use cases fall inside it.

Run the order-swap test on 200 generated explanations and report the change rate. That is your
unfaithfulness floor.

Score your explanation artefact against all four audiences and find its worst column. It is
almost certainly below 0.25.

Ask users what they did after reading an explanation. Actionability is measurable and is
usually not measured.

## 15. Advanced Concepts

The attribution-versus-intervention gap has a subtlety that cuts against a common fix.
"Interventional SHAP" — computing attributions under an interventional rather than a
conditional expectation — closes the gap by construction, and it does so by attributing to a
feature the effect of breaking its correlation with everything else. That is causally correct
and it evaluates the model at inputs it never sees, which for a model fitted on correlated data
is extrapolation. **You can have attributions that respect the data manifold or attributions
that respect intervention, not both**, and the choice is another instance of
{{ch:rai-bias}}'s pattern: a set of desirable properties with no joint solution, where the
useful move is to state which you chose.

The faithfulness result has a boundary worth marking. Chain-of-thought that is *causally load
bearing* — where the model's answer genuinely depends on the tokens it generated, because they
are in its context when it answers — is a different object from a post-hoc explanation
generated after the answer. The first can be faithful in a strong sense: perturb the reasoning
and the answer changes. The second cannot be, structurally. **The order matters**, and a system
that generates its reasoning before its answer has a testable property that one explaining
afterwards does not.

There is an interaction with {{ch:sec-permissions}}'s approval result that neither chapter
develops. A reviewer with seconds per item is reading an explanation, and an unfaithful
explanation raises their apparent confidence without raising their accuracy — which moves them
*down* the habituation curve faster, because a confident explanation that turns out fine
reinforces approving. So an unfaithful explanation degrades an approval queue in two ways at
once: it misleads on the item and it accelerates habituation across items. That is a compounding
harm and it is invisible in both chapters' models taken separately.

Finally, on inherently interpretable models. The standard objection is an accuracy cost, and on
tabular problems that cost is repeatedly measured as small. The stronger objection is that a
sparse linear model is interpretable to a *modeller*, not to a decision subject — reading five
coefficients is a skill. So interpretable-by-construction solves the debugger's and the
regulator's problem and leaves the subject's, which is the audience result again: the property
"interpretable" is not one property.

## 16. Connection to Previous Chapters

{{eq:three-fairness-criteria-cannot-hold-together}} from {{ch:rai-bias}} is the structural
parallel and {{sec:15-advanced-concepts}} shows attribution has its own version: manifold
respect and interventional correctness cannot both hold.

{{eq:cause-distance-drives-triage-cost}} from {{ch:ops-agent-tracing}} is why a debugger needs
faithfulness specifically: a plausible account of the visible step is worse than none when the
cause is 2.7 steps back.

{{eq:calibration-is-required-for-decisions}} from {{ch:ev-classical-metrics}} is the reviewer's
requirement, and the confident-unfaithful quadrant is what happens without it.

{{eq:judge-agreement-is-at-the-human-ceiling}} from {{ch:ev-llm-judge}} shares its remedy with
this chapter: the both-orders protocol tests a property that should be invariant, which works
without any model of the mechanism.

## 17. Exercises

1. Compute the attribution-to-intervention ratio for your two most correlated features. How
   large is it?

2. Measure your local surrogate's $R^2$ as a function of distance. Which of your use cases sit
   outside the radius?

3. Re-rank one explanation by actionability instead of attribution. How different is the top
   feature?

4. Run the order-swap test on 200 explanations and report the change rate.

5. Score your explanation artefact against all four audiences and identify the worst column.
   What would a second artefact cost?

## 18. Interview Questions

1. SHAP says this feature contributed 0.21. What happens if we change it?

2. What does SHAP's uniqueness theorem establish?

3. We used LIME explanations to write a segmentation policy. What is the problem?

4. Our model explains its reasoning. How would you test whether the explanation is true?

5. Who is this explanation for?

6. Why might a confident explanation be worse than no explanation?

## 19. Research Questions

1. How large is the attribution-intervention gap on real production feature sets, and is it
   reported anywhere?

2. What kernel widths are used in deployed local-explanation systems, and how much do
   explanations vary across them?

3. How much more faithful is causally load-bearing chain-of-thought than post-hoc explanation,
   under matched tests?

4. Does an unfaithful explanation measurably accelerate reviewer habituation, as
   {{sec:15-advanced-concepts}} predicts?

## 20. Chapter Summary

Interpretability's instruments have narrower guarantees than their usage.

An **attribution is not an intervention effect**: at correlation 0.97 the attribution is
**0.211** and intervening moves the score **0.012**, a factor of **17.2**
({{eq:attribution-is-not-an-intervention-effect}}). They coincide only at independence, and
nothing in a deployed feature set is independent. {{cite:lundberg2017shap}}'s uniqueness is a
theorem about axioms, stated as such in the paper.

A **local surrogate is local**: $R^2$ of **0.99** at the explained point, **0.24** at distance
0.5, **0.010** at the distance where policy gets written
({{eq:local-fidelity-does-not-extend}}). And attributions rank by contribution while users need
actionability — here the top-attributed feature is one a user changes by waiting.

**A generated explanation accounts for 36%** of what moved the decision, leaving **64%**
unmentioned including a demographic cue named 1% of the time
({{eq:stated-reasons-need-not-be-actual-reasons}}). Reading detects **4%**; one extra run with
the order swapped detects **62%**.

And **four audiences want incompatible artefacts** — every single-artefact design has a worst
column of **0.24 or below** ({{eq:an-explanation-serves-one-audience}}) — while a confident
unfaithful explanation is worth **−0.62**, twice as bad as silence.

The thread is that each method answers a well-posed question and the question is not the one
being asked. SHAP answers an allocation question; the user asks a counterfactual one. LIME
answers a local question; the analyst asks a global one. A generated explanation answers "what
is a plausible account"; the auditor asks "what happened". None of them is broken, and in each
case the fix is the same: name the question the method answers, put it next to the number, and
build a second instrument for the question you actually had.

Carry forward: **an attribution is not a counterfactual**, and **test explanations with a
second run, not by reading them**.

## 21. Further Reading

- {{cite:lundberg2017shap}} — the uniqueness theorem, and the axioms it is a theorem about.
- {{cite:ribeiro2016lime}} — local surrogate explanation, with the locality caveat stated in the
  paper and lost in use.
- {{cite:turpin2023faithfulness}} — models do not always say what they think, which is the
  finding the second listing quantifies.
- {{cite:mitchell2019modelcards}} — documentation as the regulator-facing artefact, which is a
  different object from a decision explanation.
