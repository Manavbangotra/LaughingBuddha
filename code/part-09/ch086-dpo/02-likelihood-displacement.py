# Extracted from: Chapter 86 — Preference Optimization: DPO and Its Descendants
# Source: src/.../ch086-dpo.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""DPO can lower the probability of the PREFERRED response. Here is why."""
import numpy as np

rng = np.random.default_rng(0)
N, D, BETA = 40, 16, 0.5

# The setup that matters: y_w and y_l are SIMILAR. Both are plausible answers
# to the same prompt differing in a detail, which is the normal case in real
# preference data — and they share parameters, so a gradient that pushes one
# down drags the other with it.
feat = rng.normal(size=(N, D))
shared = rng.normal(size=D)
feat[0] = shared + 0.25 * rng.normal(size=D)      # preferred
feat[1] = shared + 0.25 * rng.normal(size=D)      # dispreferred, very similar

W, L = 0, 1
cos_wl = float(feat[W] @ feat[L]
               / (np.linalg.norm(feat[W]) * np.linalg.norm(feat[L])))
print(f"cosine similarity between the two responses: {cos_wl:.3f}")
print("(this is the crux — dissimilar responses do not displace)\n")

theta0 = rng.normal(size=D) * 0.3


def probs(theta):
    z = feat @ theta
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


ref = probs(theta0)
theta = theta0.copy()

print(f"{'step':>6} {'loss':>9} {'P(preferred)':>14} {'P(dispreferred)':>17} "
      f"{'P(all others)':>15}")
for step in range(0, 801):
    p = probs(theta)
    margin = BETA * (np.log(p[W] / ref[W]) - np.log(p[L] / ref[L]))
    loss = -np.log(1 / (1 + np.exp(-margin)) + 1e-12)
    if step in (0, 50, 200, 400, 800):
        print(f"{step:>6} {loss:>9.4f} {p[W]:>14.6f} {p[L]:>17.6f} "
              f"{1 - p[W] - p[L]:>15.6f}")
    # Gradient of the margin: d log p[i] / d theta = feat[i] - E_p[feat]
    coef = (1 - 1 / (1 + np.exp(-margin))) * BETA
    mean_feat = p @ feat
    theta += 3.0 * coef * ((feat[W] - mean_feat) - (feat[L] - mean_feat))

final = probs(theta)
print(f"\nP(preferred)    {ref[W]:.6f} -> {final[W]:.6f}   "
      f"{'DOWN' if final[W] < ref[W] else 'up'}")
print(f"P(dispreferred) {ref[L]:.6f} -> {final[L]:.6f}")
print(f"P(all others)   {1 - ref[W] - ref[L]:.6f} -> "
      f"{1 - final[W] - final[L]:.6f}")

assert final[W] < ref[W], "this configuration should displace the preferred response"

print("""
The loss fell from log 2 to near zero and the implied ordering is correct — the
method did exactly what it was asked. And the probability of the PREFERRED
response went down.

Equation (eq:likelihood-displacement) is why: the objective constrains only the
DIFFERENCE of the two implicit rewards, so it is satisfied by pushing the
dispreferred response down hard, and because the two responses are similar and
share parameters, the preferred one is dragged down with it. The displaced mass
lands on responses that appear in NO comparison, about which the preference data
says nothing at all.

Note the dependence on similarity. Repeat this with two unrelated responses and
the preferred one rises as expected — displacement is a consequence of
preference pairs being NEARLY THE SAME, which is exactly what good preference
data looks like. That is why DPO implementations monitor the absolute
log-probability of chosen responses and not only the loss.""")
