# -*- coding: utf-8 -*-
# Extracted from: Chapter 96 — Hallucination: Causes, Taxonomy, and Mitigation
# Source: src/.../ch096-hallucination.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Checking claims against a source, and what the check cannot see."""

SOURCE = (
    "The Q3 board meeting is scheduled for Tuesday 14 November at 09:00 in "
    "the Bristol office. Attendance is mandatory for all directors. The "
    "agenda covers the budget review and the hiring plan."
)

CANDIDATES = {
    "Tuesday 14 November": ("grounded", "appears verbatim"),
    "09:00": ("grounded", "appears verbatim"),
    "Bristol office": ("grounded", "appears verbatim"),
    "Thursday 16 November": ("INTRINSIC", "contradicts the stated date"),
    "the London office": ("INTRINSIC", "contradicts the stated location"),
    "forty people attended": ("EXTRINSIC", "source says nothing about numbers"),
    "the CFO will present": ("EXTRINSIC", "source names no presenter"),
    "attendance is mandatory": ("grounded", "appears in substance"),
}


def substring_grounded(claim, source):
    """The cheapest possible check: does the claim's text appear in the source?
    This is ch:nlp-extraction's span-grounding idea at its simplest."""
    return claim.lower() in source.lower()


print(f"{'claim':<26} {'in source':>10} {'truth':<12} note")
for claim, (label, note) in CANDIDATES.items():
    found = substring_grounded(claim, SOURCE)
    print(f"{claim:<26} {str(found):>10} {label:<12} {note}")

detected = sum(1 for c, (l, _) in CANDIDATES.items()
               if not substring_grounded(c, SOURCE) and l != "grounded")
total_bad = sum(1 for _, (l, _) in CANDIDATES.items() if l != "grounded")
false_alarms = sum(1 for c, (l, _) in CANDIDATES.items()
                   if not substring_grounded(c, SOURCE) and l == "grounded")

print(f"\nsubstring check: flagged {detected}/{total_bad} bad claims, "
      f"{false_alarms} false alarms on good ones")
print("""
The substring check catches every fabricated claim here and also flags a
grounded one — 'attendance is mandatory' is supported in substance and not
verbatim. That is the method's shape: high recall on fabrication, poor
precision, because paraphrase is indistinguishable from invention to a string
match.

It is still worth doing. It costs a string search, it catches the fabrications
that matter most (invented specifics — names, dates, numbers), and its false
alarms are cheap to route to a stronger check.""")


# Equation (eq:groundedness): decomposition is where the difficulty lives.
SUMMARIES = {
    "faithful": [
        "The Q3 board meeting is on Tuesday 14 November.",
        "It will be held in the Bristol office.",
        "Attendance is mandatory for directors.",
    ],
    "one intrinsic error": [
        "The Q3 board meeting is on Thursday 16 November.",
        "It will be held in the Bristol office.",
        "Attendance is mandatory for directors.",
    ],
    "one extrinsic addition": [
        "The Q3 board meeting is on Tuesday 14 November.",
        "It will be held in the Bristol office.",
        "Roughly forty directors are expected to attend.",
    ],
}

# A stand-in entailment oracle. In production this is a model or a human;
# the point of the listing is what the SCORE does, not how entailment is judged.
ENTAILED = {
    "The Q3 board meeting is on Tuesday 14 November.": True,
    "It will be held in the Bristol office.": True,
    "Attendance is mandatory for directors.": True,
    "The Q3 board meeting is on Thursday 16 November.": False,
    "Roughly forty directors are expected to attend.": False,
}

print(f"\n{'summary':<24} {'claims':>7} {'grounded':>9} {'score':>7}")
for name, claims in SUMMARIES.items():
    ok = sum(ENTAILED[c] for c in claims)
    print(f"{name:<24} {len(claims):>7} {ok:>9} {ok / len(claims):>7.3f}")

print("""
Both faulty summaries score 0.667 — equation (eq:groundedness) cannot tell an
intrinsic contradiction from an extrinsic addition, and they need completely
different fixes. A groundedness score is a useful aggregate and a poor
diagnostic; the per-claim labels are what actually direct the work.""")
