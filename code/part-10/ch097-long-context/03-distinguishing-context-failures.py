# -*- coding: utf-8 -*-
# Extracted from: Chapter 97 — Long-Context Behavior and Its Limits
# Source: src/.../ch097-long-context.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Position failure, truncation, and hallucination look identical. They are not."""

SCENARIOS = {
    "position failure": dict(
        tokens_intended=48_000, tokens_sent=48_000, evidence_present=True,
        evidence_position=0.5, answered=True, answer_correct=False),
    "silent truncation": dict(
        tokens_intended=48_000, tokens_sent=32_000, evidence_present=False,
        evidence_position=0.5, answered=True, answer_correct=False),
    "extrinsic hallucination": dict(
        tokens_intended=4_000, tokens_sent=4_000, evidence_present=False,
        evidence_position=None, answered=True, answer_correct=False),
    "correct": dict(
        tokens_intended=48_000, tokens_sent=48_000, evidence_present=True,
        evidence_position=0.05, answered=True, answer_correct=True),
}


def diagnose(s):
    """The decision procedure. Every check is cheap and the first is decisive."""
    if s["tokens_sent"] < s["tokens_intended"]:
        return ("TRUNCATION",
                f"sent {s['tokens_sent']:,} of {s['tokens_intended']:,} — "
                f"fix the truncation policy (ch:llm-prompt-lifecycle)")
    if not s["evidence_present"]:
        return ("HALLUCINATION (extrinsic)",
                "no grounds in context — add retrieval (part:12)")
    if s["answer_correct"]:
        return ("no failure", "")
    pos = s["evidence_position"]
    if pos is not None and 0.25 < pos < 0.75:
        return ("POSITION",
                f"evidence at {pos:.0%} — reorder (eq:fold-ordering) or "
                f"shorten the context")
    return ("OTHER", "evidence present, well-positioned, still wrong — "
                     "a genuine model error")


print(f"{'scenario':<26} {'diagnosis':<26} action")
for name, s in SCENARIOS.items():
    dx, action = diagnose(s)
    print(f"{name:<26} {dx:<26} {action}")

print("""
All four scenarios present identically to a user: a confident wrong answer about
a document. The diagnosis needs exactly three facts, and all three are cheap to
log:

  1. tokens intended vs tokens actually sent  -> truncation
  2. is the evidence in the context at all    -> hallucination
  3. where in the context it sits             -> position

A team without those three log fields cannot distinguish these cases and will
apply one mitigation to all of them. Only one will work, and which one depends
on a fact they did not record.""")
