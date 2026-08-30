# -*- coding: utf-8 -*-
# Extracted from: Chapter 147 — Chain-of-Thought and Its Mechanics
# Source: src/.../ch147-chain-of-thought.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why a stated reason can be sincere, fluent, and unrelated to the answer.

cite:turpin2023faithfulness injects a bias into a task -- reordering options so
the correct one is always in the same position -- and finds models exploiting it,
losing up to 36% accuracy when the bias points the wrong way, and producing
confident explanations that NEVER MENTION IT.

That result is often read as a surprising failure. It is not surprising once you
look at what the two outputs are trained on, and this listing makes the mechanism
explicit by building a system with the same structure
(eq:trace-and-answer-are-untied).

A single model with two heads. One predicts the ANSWER and is trained on labels.
The other produces the STATED REASON and is trained on human-written rationales.
Nothing in the training ties them together, so nothing makes the second an
account of the first.
"""
import numpy as np

rng = np.random.default_rng(307)

D_REAL = 6                # features a human would cite
N = 24000


def make(n, shortcut_strength=1.0, shortcut_valid=True):
    """The label depends on real features. A SHORTCUT feature also predicts it,
    perfectly during training, because of how the data was collected."""
    Xr = rng.normal(size=(n, D_REAL))
    w = np.array([1.4, -1.1, 0.9, 0.0, 0.0, 0.0])      # only 3 features matter
    logit = Xr @ w
    y = (logit > 0).astype(int)
    if shortcut_valid:
        s = y * shortcut_strength + (1 - y) * (-shortcut_strength)
    else:
        s = (1 - y) * shortcut_strength + y * (-shortcut_strength)
    s = s + 0.15 * rng.normal(size=n)
    X = np.concatenate([Xr, s[:, None]], axis=1)
    # The human rationale cites whichever real feature contributed most. It
    # cannot cite the shortcut, because the human never saw it.
    contrib = Xr * w
    rationale = np.abs(contrib).argmax(1)
    return X, y, rationale


def fit_logistic(X, Y, classes, steps=400, lr=0.5):
    W = np.zeros((X.shape[1], classes))
    T = np.eye(classes)[Y]
    for _ in range(steps):
        z = X @ W
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        W -= lr * (X.T @ (p - T) / len(X))
    return W


Xtr, Ytr, Rtr = make(N)
ANSWER = fit_logistic(Xtr, Ytr, 2)        # trained on labels
REASON = fit_logistic(Xtr, Rtr, D_REAL)   # trained on human rationales


def answer(X):
    return (X @ ANSWER).argmax(1)


def stated_reason(X):
    return (X @ REASON).argmax(1)


def acc(X, Y):
    return float(np.mean(answer(X) == Y))


print("A model with two heads: one predicts the answer, one states the reason.")
print(f"{D_REAL} real features (only three matter) plus one shortcut feature that")
print("predicts the label perfectly in training. Human rationales never mention")
print("the shortcut, because the humans never saw it.")
print()
print(f"{'condition':>34}{'accuracy':>11}{'reasons citing':>17}")
print(f"{'':>34}{'':>11}{'the shortcut':>17}")
print("-" * 62)

CASES = [
    ("training distribution", True, 1.0),
    ("shortcut removed", True, 0.0),
    ("shortcut points the WRONG way", False, 1.0),
]
res = {}
for name, valid, strength in CASES:
    X, Y, R = make(6000, strength, valid)
    a = acc(X, Y)
    cite = 0.0                       # the reason head cannot output "shortcut"
    res[name] = a
    print(f"{name:>34}{a:>11.1%}{cite:>17.1%}")

print()
print("  (the reason head has no shortcut option in its output space, so it")
print("   cannot cite it even when the shortcut decided the answer)")

print()
print()
print("How much does each head depend on the shortcut? Weight magnitude as a")
print("share of the total, and accuracy when each feature is ablated.")
print()
print(f"{'feature':>12}{'answer head':>14}{'reason head':>14}"
      f"{'accuracy with':>16}")
print(f"{'':>12}{'weight share':>14}{'weight share':>14}"
      f"{'it zeroed':>16}")
print("-" * 58)
aw = np.abs(ANSWER[:, 1] - ANSWER[:, 0])
rw = np.abs(REASON).sum(1)
Xe, Ye, _ = make(6000)
share = {}
for j, name in enumerate([f"real {i}" for i in range(D_REAL)] + ["SHORTCUT"]):
    Xa = Xe.copy(); Xa[:, j] = 0.0
    share[name] = (aw[j] / aw.sum(), rw[j] / rw.sum(), acc(Xa, Ye))
    print(f"{name:>12}{share[name][0]:>14.1%}{share[name][1]:>14.1%}"
          f"{share[name][2]:>16.1%}")

print()
print()
print("Are the stated reasons plausible? Agreement with the human rationale,")
print("which is what a reader would use to judge them.")
print()
print(f"{'condition':>34}{'reason matches':>17}{'answer':>10}")
print(f"{'':>34}{'the human one':>17}{'correct':>10}")
print("-" * 61)
plaus = {}
for name, valid, strength in CASES:
    X, Y, R = make(6000, strength, valid)
    m = float(np.mean(stated_reason(X) == R))
    plaus[name] = (m, acc(X, Y))
    print(f"{name:>34}{m:>17.1%}{plaus[name][1]:>10.1%}")

tr = res["training distribution"]
rm = res["shortcut removed"]
wr = res["shortcut points the WRONG way"]
print(f"""
The first table is cite:turpin2023faithfulness's experiment with the mechanism
exposed.

On the training distribution the answer head is {tr:.1%} accurate. Remove the
shortcut feature and it drops to {rm:.1%}. Point the shortcut the wrong way and it
falls to {wr:.1%} -- below chance, because it is now confidently following a
feature that has been inverted.

That spread is the measurement of how much the shortcut was doing:
{tr - wr:.1%} of the accuracy was resting on a feature nobody intended the model
to use.

And the reasons cite it {0.0:.0%} of the time, in every condition, because the
reason head's output space does not contain it. It was trained to predict which
REAL feature a human would have cited, and it does that job well.

Which is the whole mechanism, and it is not a failure of anything. Two heads, two
training signals, no term in either objective that ties them together
(eq:trace-and-answer-are-untied). The answer head was rewarded for being right and
found the shortest path to being right. The reason head was rewarded for sounding
like a human explanation and produces one. Nothing asked them to agree.

The second table quantifies the divergence, and it contains the result that
sharpens the whole problem.

The shortcut carries {share['SHORTCUT'][0]:.1%} of the answer head's weight -- more
than the three genuinely-predictive features combined. And zeroing it leaves
accuracy at {share['SHORTCUT'][2]:.1%}.

Read those two together. The model DID learn the correct computation: the real
features are weighted correctly, the irrelevant ones are at zero, and with the
shortcut set aside the model is perfect. It is not that a spurious feature crowded
out the right answer. **The right answer is in there, fully learned, and the
shortcut overrides it.**

Which is why inverting the shortcut takes accuracy to {wr:.1%} rather than to the
{share['SHORTCUT'][2]:.1%} the real features alone would deliver. The model has
everything it needs to be right and is following the stronger signal, exactly as
a linear combination weighted {share['SHORTCUT'][0]:.1%} to one feature must.

That is a worse situation than "the model never learned to reason", and it is the
one cite:turpin2023faithfulness's biasing experiments produce. A model that lacked
the capability would fail visibly on hard cases. A model that has the capability
and is outvoted by a spurious feature fails only when the spurious feature
disagrees -- which is rare in training, rare in evaluation, and exactly what
happens when the deployment distribution shifts.

The third table is the part that makes this hard to catch, and the absolute
numbers matter less than their flatness.

The stated reason matches the human rationale
{plaus['training distribution'][0]:.1%} of the time on the training distribution,
{plaus['shortcut removed'][0]:.1%} with the shortcut removed, and
{plaus['shortcut points the WRONG way'][0]:.1%} when the shortcut has been
inverted and the answer is {plaus['shortcut points the WRONG way'][1]:.1%}
correct.

Those three numbers are the same. The answers behind them are
{plaus['training distribution'][1]:.1%}, {plaus['shortcut removed'][1]:.1%} and
{plaus['shortcut points the WRONG way'][1]:.1%} correct.

**The explanation carries no information about whether the answer is right.** It is
not degraded when the model is wrong, because it was never a function of the
answer -- it is a function of the input, computed by a separate head with a
separate objective. A reader inspecting it sees exactly the same thing in the case
where the model is perfect and the case where it is worse than guessing.

(The absolute agreement of around {plaus['training distribution'][0]:.0%} is not
high -- the reason head is a weak predictor of which feature a human would cite.
That is beside the point here. Even a reason head that matched humans perfectly
would show the same flatness, because the flatness comes from the two heads being
untied rather than from either being bad at its job.)

Three consequences, and they are the reason this chapter exists.

A chain of thought is evidence about what a plausible justification looks like,
not about what happened. Reading one tells you the model can produce a rationale,
which was never in doubt.

Interpretability and safety arguments that rest on monitoring the trace need an
additional ingredient: something that ties the trace to the computation. Post-hoc
rationalisation is the DEFAULT outcome of training two outputs on two objectives,
and a tie between them has to be constructed deliberately -- it does not arise
from making either output better.

And the practical test is not to read the reasoning. It is to PERTURB the input in
a way the stated reasoning implies is irrelevant, and see whether the answer
moves. That is ch:rsn-vs-generation's invariance criterion arriving for a second
reason: it measures the computation rather than the story about it.""")
