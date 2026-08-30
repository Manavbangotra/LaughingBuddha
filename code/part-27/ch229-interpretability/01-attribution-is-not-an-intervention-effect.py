# -*- coding: utf-8 -*-
# Extracted from: Chapter 229 — Explainability and Interpretability
# Source: src/.../ch229-interpretability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
