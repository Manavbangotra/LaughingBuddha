# -*- coding: utf-8 -*-
# Extracted from: Chapter 90 — Decoding: Softmax, Temperature, Top-k, Top-p, and Beam Search
# Source: src/.../ch090-decoding.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Decoding settings are a task decision. The defaults are a chat compromise."""

TASKS = {
    "code generation": dict(
        determinism="high", diversity="none", failure="subtle wrong logic",
        temperature=0.2, top_p=0.95, top_k=0, repetition_penalty=1.0,
        note="penalties break repeated identifiers; low T for determinism"),
    "factual Q&A (RAG)": dict(
        determinism="high", diversity="none", failure="fabrication",
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        note="greedy: the answer is in the context, do not invent alternatives"),
    "marketing copy": dict(
        determinism="low", diversity="high", failure="bland or repetitive",
        temperature=1.0, top_p=0.95, top_k=0, repetition_penalty=1.05,
        note="sampling at T~1 matches the model's distribution (eq:human-surprise-gap)"),
    "JSON extraction": dict(
        determinism="total", diversity="none", failure="unparseable",
        temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        note="greedy + constrained decoding (ch:llm-structured-output)"),
}

DEFAULTS = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)

print(f"{'task':<20} {'T':>5} {'top_p':>7} {'rep pen':>9} {'matches default':>17}")
for name, cfg in TASKS.items():
    matches = all(cfg[k] == v for k, v in DEFAULTS.items())
    print(f"{name:<20} {cfg['temperature']:>5.1f} {cfg['top_p']:>7.2f} "
          f"{cfg['repetition_penalty']:>9.2f} {str(matches):>17}")

print(f"\n{'task':<20} {'primary failure mode':<24} {'why this config'}")
for name, cfg in TASKS.items():
    print(f"{name:<20} {cfg['failure']:<24} {cfg['note']}")

n_wrong = sum(1 for c in TASKS.values()
              if not all(c[k] == v for k, v in DEFAULTS.items()))
print(f"\n{n_wrong} of {len(TASKS)} tasks need something other than the "
      f"defaults.")

# What the wrong setting costs, per task.
print(f"\n{'task':<20} {'symptom if left at defaults':<46}")
SYMPTOMS = {
    "code generation": "nondeterministic output; occasional invalid syntax",
    "factual Q&A (RAG)": "answers not grounded in the retrieved passage",
    "marketing copy": "fine — this is what the defaults were chosen for",
    "JSON extraction": "intermittent parse failures under load",
}
for name, sym in SYMPTOMS.items():
    print(f"{name:<20} {sym:<46}")

print("""
Every one of these symptoms would normally be reported as a model problem, and
three of the four are decoding problems fixable in a config file.

The pattern to internalise: ask whether the task has a CORRECT answer. If it
does — code, extraction, grounded Q&A — you want determinism, and temperature
above zero is actively harmful because it introduces variation into something
that should not vary. If it does not, you want the model's own distribution,
which is T near 1 with nucleus truncation.

The provider defaults are tuned for open-ended chat, which is exactly one of
these four cases.""")
