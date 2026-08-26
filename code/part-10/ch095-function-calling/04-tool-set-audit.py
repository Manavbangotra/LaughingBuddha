# -*- coding: utf-8 -*-
# Extracted from: Chapter 95 — Function Calling and Tool Use
# Source: src/.../ch095-function-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Auditing a tool set: overlap, context cost, and where selection fails."""
import numpy as np

rng = np.random.default_rng(7)

# A realistic tool set, with the overlaps real systems accumulate.
TOOLS = {
    "search_documents":      "search",
    "search_knowledge_base": "search",
    "find_files":            "search",
    "lookup_policy":         "search",
    "get_user":              "user",
    "get_user_profile":      "user",
    "get_customer_record":   "user",
    "send_email":            "notify",
    "send_slack_message":    "notify",
    "create_ticket":         "write",
    "update_ticket":         "write",
    "close_ticket":          "write",
    "calculate":             "compute",
    "run_report":            "compute",
}
# Add filler tools to reach a realistic count.
for i in range(20):
    TOOLS[f"misc_tool_{i}"] = f"unique_{i}"

groups = {}
for name, family in TOOLS.items():
    groups.setdefault(family, []).append(name)

print(f"{len(TOOLS)} tools in {len(groups)} functional families\n")
print(f"{'family':<12} {'tools':>6}  {'members'}")
for fam, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(members) > 1:
        print(f"{fam:<12} {len(members):>6}  {', '.join(members)}")

confusable = sum(len(m) for m in groups.values() if len(m) > 1)
print(f"\n{confusable} of {len(TOOLS)} tools sit in a confusable family "
      f"({confusable / len(TOOLS):.0%})")


def accuracy(n_competitors, separation, trials=4000):
    wins = 0
    for _ in range(trials):
        correct = separation + rng.normal()
        if n_competitors == 0:
            wins += 1
            continue
        if correct > rng.normal(size=n_competitors).max():
            wins += 1
    return wins / trials


# Within-family separation is low; across-family separation is high.
WITHIN_SEP, ACROSS_SEP = 0.6, 2.8
print(f"\n{'query targets':<28} {'competitors':>12} {'separation':>12} "
      f"{'accuracy':>10}")
for fam, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:4]:
    n_within = len(members) - 1
    sep = WITHIN_SEP if n_within else ACROSS_SEP
    acc = accuracy(len(TOOLS) - 1, sep) if n_within == 0 else accuracy(
        n_within, WITHIN_SEP)
    print(f"{fam:<28} {n_within:>12} {sep:>12.1f} {acc:>10.3f}")

# The intervention: merge within families, keep the parameter distinction.
merged = len(groups)
print(f"\n{'configuration':<30} {'tools':>7} {'schema tokens':>15} "
      f"{'est. selection acc':>20}")
for label, k, sep in [("as-is (34 tools)", len(TOOLS), WITHIN_SEP),
                      ("merged by family", merged, ACROSS_SEP)]:
    print(f"{label:<30} {k:>7} {k * 120:>15,} "
          f"{accuracy(k - 1, sep):>20.3f}")

print("""
Look at what changed and what did not. The tool count fell from 34 to 25 — barely
a quarter — while estimated selection accuracy went from 0.088 to 0.781. The
count is not what fixed it.

eq:max-distractor says count costs only logarithmically. What was actually wrong
is that fourteen tools sat in families whose members are barely distinguishable,
so within a family the separation is small and selection is close to a coin
flip.

Merging each family into one tool with a parameter — search(scope=...) rather
than four search tools — raises separation and cuts schema tokens at the same
time. Writing better descriptions for four tools that genuinely do the same
thing does not, because no description makes two identical functions
distinguishable.""")
