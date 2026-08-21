# Extracted from: Chapter 47 — Model Registry and the Deployment Handoff
# Source: src/.../ch047-registry.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""The ML Test Score (eq. 47.3), and why the aggregate is a minimum.
"""
import numpy as np

# --- the rubric, abbreviated to the checks that carry the most weight -------
RUBRIC = {
    "data": [
        "feature expectations are captured in a schema",
        "all features are beneficial (tested, not assumed)",
        "no feature costs more than it is worth",
        "features comply with policy and privacy requirements",
        "the data pipeline has appropriate privacy controls",
        "new features can be added quickly",
        "all input feature code is tested",
    ],
    "model": [
        "model specs are reviewed and version-controlled",
        "offline and online metrics correlate",
        "all hyperparameters have been tuned",
        "the impact of model staleness is known",
        "a simpler model is not better",
        "model quality is sufficient on all important data slices",
        "the model has been tested for inclusion (fairness)",
    ],
    "infrastructure": [
        "training is reproducible",
        "model specs are unit tested",
        "the full ML pipeline is integration tested",
        "model quality is validated before serving",
        "the model allows debugging by observing a single example",
        "models are canaried before serving",
        "serving models can be rolled back",
    ],
    "monitoring": [
        "dependency changes result in notification",
        "data invariants hold in training and serving",
        "training and serving features compute the same values",
        "models are not too stale",
        "the model is numerically stable",
        "compute performance has not regressed",
        "prediction quality has not regressed",
    ],
}


def score(answers):
    """0 = not done, 0.5 = documented but manual, 1 = automated (eq. 47.3)."""
    per_cat = {c: sum(answers[c]) for c in RUBRIC}
    return per_cat, min(per_cat.values())


def band(total):
    if total < 1:
        return "more of a research project than a product"
    if total < 3:
        return "not enough testing"
    if total < 5:
        return "reasonable, but investment warranted"
    if total < 7:
        return "strong levels of automated testing"
    return "exceptional"


# --- three teams, all of whom believe they test well ------------------------
teams = {
    "the research-strong team": {
        "data": [1, 0.5, 0, 0.5, 0.5, 1, 0.5],
        "model": [1, 0.5, 1, 0.5, 1, 1, 0.5],
        "infrastructure": [1, 0.5, 0.5, 0.5, 0.5, 0, 0.5],
        "monitoring": [0, 0.5, 0, 0, 0, 0.5, 0],
    },
    "the platform-strong team": {
        "data": [1, 0, 0, 1, 1, 1, 1],
        "model": [0.5, 0, 0.5, 0, 0, 0, 0],
        "infrastructure": [1, 1, 1, 1, 0.5, 1, 1],
        "monitoring": [1, 1, 0.5, 1, 0.5, 1, 0.5],
    },
    "the balanced team": {
        "data": [1, 0.5, 0.5, 1, 1, 0.5, 1],
        "model": [1, 0.5, 1, 0.5, 0.5, 1, 0.5],
        "infrastructure": [1, 0.5, 1, 1, 0.5, 1, 1],
        "monitoring": [0.5, 1, 1, 0.5, 0.5, 1, 0.5],
    },
}

print("=" * 72)
print("the ML Test Score: minimum across categories, not sum (eq. 47.3)")
print("=" * 72)
print(f"{'team':<28} " + " ".join(f"{c[:6]:>7}" for c in RUBRIC) +
      f" {'SUM':>6} {'SCORE':>7}")
for name, ans in teams.items():
    per_cat, total = score(ans)
    print(f"{name:<28} " + " ".join(f"{per_cat[c]:>7.1f}" for c in RUBRIC) +
          f" {sum(per_cat.values()):>6.1f} {total:>7.1f}")

print(f"\n{'team':<28} {'verdict':<44}")
for name, ans in teams.items():
    _, total = score(ans)
    print(f"{name:<28} {band(total):<44}")

print("\nThe first two teams differ by four points of SUM — 14 against 18,")
print("a 29% gap that a sum-based rubric would treat as meaningful — and")
print("score IDENTICALLY at 1.0. The score is right and the sum is not:")
print("both have one nearly-empty category, and it makes no difference to")
print("their reliability which category it is.")
print("\nThe research-strong team has excellent model development and")
print("almost no monitoring, so it will fail silently. The platform-strong")
print("team has excellent infrastructure and almost no model validation, so")
print("it will deploy a bad model quickly and reliably. Neither is a")
print("well-tested system, and only the minimum says so.")
print("\nSection 6.1 is why the minimum is the right rule rather than a")
print("rhetorical one: if a system fails when ANY area fails, total")
print("reliability is dominated by the weakest area, and improving an")
print("already-strong one barely moves it.")

# --- and the arithmetic behind that (section 6.1) ---------------------------
print("\n" + "=" * 72)
print("why the minimum, quantified (eq. 47.5)")
print("=" * 72)
print("Model each category's failure probability as falling with its score:")
print("  p_i = 0.30 * exp(-0.45 * s_i)\n")


def p_fail(s):
    return 0.30 * np.exp(-0.45 * s)


print(f"{'team':<28} " + " ".join(f"{'p(' + c[:4] + ')':>9}" for c in RUBRIC)
      + f" {'p(system)':>11}")
for name, ans in teams.items():
    per_cat, _ = score(ans)
    ps = [p_fail(per_cat[c]) for c in RUBRIC]
    p_sys = 1 - np.prod([1 - p for p in ps])
    print(f"{name:<28} " + " ".join(f"{p:>9.3f}" for p in ps) +
          f" {p_sys:>11.3f}")

print("\nNow the counterfactual that makes the point. Take the")
print("research-strong team and give it one more point of testing, spent in")
print("two different places:\n")
base = teams["the research-strong team"]
per_cat, _ = score(base)
p_sys_base = 1 - np.prod([1 - p_fail(per_cat[c]) for c in RUBRIC])
print(f"{'where the point is spent':<34} {'p(system fails)':>17} "
      f"{'improvement':>13}")
print(f"{'nowhere (baseline)':<34} {p_sys_base:>17.4f} {'-':>13}")
for target in ("model", "monitoring"):
    pc = dict(per_cat)
    pc[target] += 1.0
    p_sys = 1 - np.prod([1 - p_fail(pc[c]) for c in RUBRIC])
    print(f"{'into ' + target + ' (already ' + f'{per_cat[target]:.1f}' + ')':<34} "
          f"{p_sys:>17.4f} {p_sys_base - p_sys:>13.4f}")

print("\nThe same unit of effort buys several times more reliability when")
print("spent on the weakest category. A sum-based score is indifferent")
print("between the two; the minimum directs the effort where eq. 47.5 says")
print("it pays. That is the whole argument, and it is why the rubric's most")
print("useful output is not the number but WHICH category produced it.")
