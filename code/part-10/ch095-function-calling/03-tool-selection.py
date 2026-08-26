# -*- coding: utf-8 -*-
# Extracted from: Chapter 95 — Function Calling and Tool Use
# Source: src/.../ch095-function-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""How many tools is too many? Equation (eq:selection-degradation)."""
import numpy as np

rng = np.random.default_rng(2)
TRIALS = 4000


def selection_accuracy(n_tools, separation, noise=1.0):
    """The correct tool scores `separation` above the distractors' mean;
    all scores are noisy. Selection succeeds if the correct one wins."""
    wins = 0
    for _ in range(TRIALS):
        correct = separation + rng.normal(0, noise)
        distractors = rng.normal(0, noise, n_tools - 1)
        if correct > distractors.max(initial=-np.inf):
            wins += 1
    return wins / TRIALS


print("Tool-selection accuracy against tool count\n")
print(f"{'tools':>7} " + " ".join(f"{'sep=' + str(s):>10}"
                                   for s in (0.5, 1.0, 2.0, 3.0)))
for k in (2, 5, 10, 25, 50, 100):
    row = " ".join(f"{selection_accuracy(k, s):>10.3f}"
                   for s in (0.5, 1.0, 2.0, 3.0))
    print(f"{k:>7} {row}")

print("""
Read down the columns rather than across the rows.

With poorly differentiated tools (sep=0.5) accuracy is bad at five tools and
hopeless at fifty. With well-differentiated ones (sep=3.0) it is still good at a
hundred. The degradation with tool COUNT is logarithmic and slow
(eq:max-distractor); the dependence on SEPARATION is what actually decides it.

So 'how many tools can a model handle' is the wrong question. The right one is
'are my tool descriptions distinguishable', and the fix for a system that
selects badly is usually to merge overlapping tools and sharpen descriptions
rather than to reduce the count.""")

# What separation looks like in practice.
print(f"\n{'tool pair':<44} {'overlap'}")
PAIRS = [
    ("search_docs / query_knowledge_base", "high — merge them"),
    ("search_docs / calculate", "none"),
    ("get_weather / get_forecast", "high — one tool, a time parameter"),
    ("send_email / send_notification", "moderate — clarify in descriptions"),
]
for pair, note in PAIRS:
    print(f"{pair:<44} {note}")

# The context cost of carrying tools.
print(f"\n{'tools':>7} {'schema tokens':>15} {'prefill/turn @2N':>18} "
      f"{'per 1M turns':>14}")
TOKENS_PER_TOOL, N = 120, 7e9
for k in (5, 20, 50, 100):
    toks = k * TOKENS_PER_TOOL
    flops = 2 * N * toks
    print(f"{k:>7} {toks:>15,} {flops:>18.2e} {flops * 1e6:>14.2e}")

print("""
Tool schemas are prompt tokens paid on every turn (ch:llm-inference). A
hundred-tool schema set is 12,000 tokens of prefill before the conversation
starts, which is both a latency cost and a large share of the context window.

Prefix caching recovers most of the compute — the schema block is byte-identical
across turns — but not the context-window space, and the space competes with
conversation history and retrieved content.""")
