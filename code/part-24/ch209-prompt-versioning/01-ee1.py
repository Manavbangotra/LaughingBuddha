# -*- coding: utf-8 -*-
# Extracted from: Chapter 209 — Prompt and Evaluation-Set Versioning
# Source: src/.../ch209-prompt-versioning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A prompt is code, and it is the only code with no tests, no review, and no types.

Every other artefact that determines behaviour passes through gates: a compiler, a type
checker, a test suite, a reviewer. A prompt passes through none of them. It is a string,
it is edited by people who are not committing, and the first thing that evaluates it is
production.

So the defect escape rate for a prompt change is the escape rate for an ungated change,
and this listing measures what that is against the gated alternatives
(eq:prompt-is-ungated-code).

The finding is that the gap is not explained by prompts being harder to test. It is
explained by nobody testing them.
"""
# (artefact, changes per week, gates it passes, P(a defective change escapes each gate)
CHANGES = [
    ("application code",   3.0,
     [("compiler", 0.55), ("type check", 0.70), ("unit tests", 0.55),
      ("code review", 0.60), ("integration tests", 0.65)]),
    ("configuration",      2.0,
     [("schema validation", 0.70), ("code review", 0.60)]),
    ("tool schema",        0.7,
     [("schema validation", 0.65), ("code review", 0.60)]),
    ("system prompt",      6.0, []),
    ("few-shot examples",  4.0, []),
]
P_DEFECTIVE = 0.22        # share of changes that contain a defect, before gates


def escape(gates):
    p = 1.0
    for name, keep in gates:
        p *= keep
    return p


print("How each artefact is gated before it reaches production.")
print()
print(f"{'artefact':>22}{'changes/week':>15}{'gates':>8}"
      f"{'escape rate':>14}{'defects/week':>15}")
print("-" * 76)
tab = {}
for name, rate, gates in CHANGES:
    e = escape(gates)
    d = rate * P_DEFECTIVE * e
    tab[name] = (rate, len(gates), e, d)
    print(f"{name:>22}{rate:>15.1f}{len(gates):>8}{e:>14.1%}{d:>15.2f}")
print("-" * 76)
total = sum(tab[n][3] for n, r, g in CHANGES)
print(f"{'TOTAL':>22}{sum(r for n, r, g in CHANGES):>15.1f}{'':>8}"
      f"{'':>14}{total:>15.2f}")

print()
print()
print("Share of escaped defects by artefact -- which is not the share of changes.")
print()
print(f"{'artefact':>22}{'share of changes':>19}{'share of escapes':>19}"
      f"{'ratio':>9}")
print("-" * 70)
tot_changes = sum(r for n, r, g in CHANGES)
share = {}
for name, rate, gates in CHANGES:
    sc = rate / tot_changes
    se = tab[name][3] / total
    share[name] = (sc, se, se / sc)
    print(f"{name:>22}{sc:>19.0%}{se:>19.0%}{se / sc:>9.1f}x")

print()
print()
print("What each gate would remove, applied to prompts.")
print()
GATES = [
    ("schema / format check",   0.62, 0.5),
    ("golden-output test",      0.71, 3.0),
    ("peer review",             0.45, 1.0),
    ("evaluation-set gate",     0.38, 6.0),
    ("shadow comparison",       0.30, 9.0),
]
print(f"{'gate for prompts':>26}{'keeps':>9}{'escape after':>15}"
      f"{'defects/week':>15}{'effort':>9}")
print("-" * 76)
cur = 1.0
eff = 0.0
prompt_rate = tab["system prompt"][0] + tab["few-shot examples"][0]
base_prompt = prompt_rate * P_DEFECTIVE
path = []
for label, keep, e in GATES:
    cur *= keep
    eff += e
    path.append((label, cur, prompt_rate * P_DEFECTIVE * cur, eff))
    print(f"{label:>26}{keep:>9.0%}{cur:>15.1%}"
          f"{prompt_rate * P_DEFECTIVE * cur:>15.2f}{eff:>9.1f}")

print()
print(f"prompt defects per week, ungated: {base_prompt:.2f}")
print(f"after all five gates:             {path[-1][2]:.2f}")

print()
print()
print("Cost per defect prevented, which is how a gate should be chosen.")
print()
order = sorted(GATES, key=lambda g: -((1 - g[1]) / g[2]))
print(f"{'rank':>6}{'gate':>26}{'removes':>10}{'effort':>9}"
      f"{'defects/wk prevented':>23}{'per effort':>13}")
print("-" * 88)
for i, (label, keep, e) in enumerate(order, 1):
    prevented = base_prompt * (1 - keep)
    print(f"{i:>6}{label:>26}{1 - keep:>10.0%}{e:>9.1f}"
          f"{prevented:>23.2f}{prevented / e:>13.3f}")

print()
print()
print("And the comparison that makes the case: apply the SAME gate coverage that")
print("application code already has.")
print()
code_escape = tab["application code"][2]
print(f"{'artefact':>22}{'escape rate':>14}{'defects/week':>15}"
      f"{'vs code':>11}")
print("-" * 64)
for name, rate, gates in CHANGES:
    print(f"{name:>22}{tab[name][2]:>14.1%}{tab[name][3]:>15.2f}"
          f"{tab[name][2] / code_escape:>10.1f}x")
print(f"{'prompt at code gating':>22}{code_escape:>14.1%}"
      f"{prompt_rate * P_DEFECTIVE * code_escape:>15.2f}{1.0:>10.1f}x")

print(f"""
The gating table is the whole argument and it needs almost no commentary. Application
code passes {tab['application code'][1]} gates and escapes at
{tab['application code'][2]:.1%}. A system prompt passes **zero** and escapes at
{tab['system prompt'][2]:.0%} (eq:prompt-is-ungated-code).

Every defective prompt change reaches production. Not most of them --
**all of them**, because there is nothing in the path that could stop one.

The share table converts that into where the defects come from. Prompts and few-shot
examples are {share['system prompt'][0] + share['few-shot examples'][0]:.0%} of changes
and {share['system prompt'][1] + share['few-shot examples'][1]:.0%} of escaped defects.
Application code is {share['application code'][0]:.0%} of changes and
{share['application code'][1]:.0%} of escapes.

**Prompts produce {share['system prompt'][1] / share['system prompt'][0]:.1f} times their
share of defects and code produces {share['application code'][1] / share['application code'][0]:.2f}
times its share** -- and the difference is entirely gating, since the defect rate before
gates was assumed identical.

That last point is worth being explicit about. This listing assumes prompts and code are
equally likely to contain a mistake when written. **The escape gap is not because prompts
are harder to get right. It is because nothing checks them.**

The gate table shows the ungated state is a choice rather than a necessity. A format
check removes {1 - 0.62:.0%} of defective prompt changes for {0.5:.1f} units of effort.
A golden-output test -- which ch:sd-architecture said survives at only 9% for model
outputs -- still removes {1 - 0.71:.0%} when applied to *prompt structure* rather than to
generated text, because it is checking that the assembled prompt looks right rather than
that the answer is right.

All five gates take prompt defects from {base_prompt:.2f} to {path[-1][2]:.2f} a week.

The ranking is where a plan comes from. `{order[0][0]}` removes {1 - order[0][1]:.0%} for
{order[0][2]:.1f} effort -- {base_prompt * (1 - order[0][1]) / order[0][2]:.3f} defects
prevented per unit. `{order[-1][0]}` removes {1 - order[-1][1]:.0%} for
{order[-1][2]:.1f}, which is {base_prompt * (1 - order[-1][1]) / order[-1][2]:.3f}.

**The cheapest gate is a format check and it does not exist in most systems.** Not
because it is hard -- it is asserting that the assembled prompt contains the sections it
should, in the order it should, within the length it should -- but because a prompt does
not feel like something you assert about.

The final table is the comparison to put in a design document. Applying application
code's existing gate coverage to prompts would take them from
{tab['system prompt'][3] + tab['few-shot examples'][3]:.2f} escaped defects a week to
{prompt_rate * P_DEFECTIVE * code_escape:.2f} --
{(tab['system prompt'][3] + tab['few-shot examples'][3]) / (prompt_rate * P_DEFECTIVE * code_escape):.0f}
times fewer.

**Nothing about that requires new technology.** It requires deciding that the string is
code, which is a policy decision that ch:ops-versioning already argued for on
reproducibility grounds and this listing argues for again on quality grounds. Two
independent arguments, one afternoon of work, and it is still the most commonly skipped
item in this part.""")
