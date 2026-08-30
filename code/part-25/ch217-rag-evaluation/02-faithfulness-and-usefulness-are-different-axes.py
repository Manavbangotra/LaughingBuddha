# -*- coding: utf-8 -*-
# Extracted from: Chapter 217 — RAG Evaluation
# Source: src/.../ch217-rag-evaluation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Faithfulness is the cheapest RAG metric to compute and the least related to usefulness.

cite:es2023ragas made faithfulness -- is every claim in the answer supported by the
retrieved context? -- measurable without ground truth, which is a genuine advance and the
reason it is now in every RAG dashboard.

It is also an axis, not a summary. Whether the context *contained* the answer is a separate
question, and the two together define four outcomes with completely different value to a
user (eq:faithfulness-and-usefulness-are-different-axes).

Optimising the cheap axis moves a system into the quadrant where it is faithful and useless,
and cite:barnett2024sevenfailures' catalogue of production failure points is mostly made of
things no end-to-end score can see
(eq:most-rag-failures-are-invisible-end-to-end).
"""
P_SUFFICIENT = 0.63      # context actually contains what is needed
P_FAITHFUL_SUFF = 0.86   # answer stays within the context, when context is sufficient
P_FAITHFUL_INSUFF = 0.52 # answer stays within the context, when it is not

print("Two axes, four outcomes. Only one of them is a good answer.")
print()
QUADRANTS = [
    ("sufficient",   "faithful",     P_SUFFICIENT * P_FAITHFUL_SUFF,
     "correct and grounded",  1.00),
    ("sufficient",   "unfaithful",   P_SUFFICIENT * (1 - P_FAITHFUL_SUFF),
     "right facts, wrong support", 0.35),
    ("insufficient", "faithful",     (1 - P_SUFFICIENT) * P_FAITHFUL_INSUFF,
     "grounded refusal or partial", 0.28),
    ("insufficient", "unfaithful",   (1 - P_SUFFICIENT) * (1 - P_FAITHFUL_INSUFF),
     "confident invention",   0.00),
]
print(f"{'context':>14}{'answer':>13}{'share':>9}"
      f"{'what the user gets':>30}{'value':>8}")
print("-" * 74)
quad = {}
for ctx, ans, share, desc, val in QUADRANTS:
    quad[(ctx, ans)] = (share, val)
    print(f"{ctx:>14}{ans:>13}{share:>9.3f}{desc:>30}{val:>8.2f}")

faith = sum(s for c, a, s, d, v in QUADRANTS if a == "faithful")
useful = sum(s * v for c, a, s, d, v in QUADRANTS)
print("-" * 74)
print(f"{'faithfulness':>14}{'':>13}{faith:>9.3f}")
print(f"{'usefulness':>14}{'':>13}{useful:>9.3f}")

print()
print()
print("Now optimise faithfulness, which is the metric you can measure without")
print("ground truth. Push the model to stay inside the context.")
print()
print(f"{'faithful | insuff':>19}{'faithful | suff':>17}{'measured faith':>16}"
      f"{'usefulness':>13}{'confident inventions':>22}")
print("-" * 87)
opt = {}
for target in (0.52, 0.65, 0.78, 0.90, 0.97):
    # Pushing faithfulness mostly changes behaviour on insufficient context:
    # the model refuses or hedges instead of inventing.
    fs = min(0.97, P_FAITHFUL_SUFF + 0.35 * (target - P_FAITHFUL_INSUFF))
    f_meas = P_SUFFICIENT * fs + (1 - P_SUFFICIENT) * target
    u = (P_SUFFICIENT * fs * 1.00
         + P_SUFFICIENT * (1 - fs) * 0.35
         + (1 - P_SUFFICIENT) * target * 0.28
         + (1 - P_SUFFICIENT) * (1 - target) * 0.00)
    inv = (1 - P_SUFFICIENT) * (1 - target)
    opt[target] = (f_meas, u, inv, fs)
    print(f"{target:>19.2f}{fs:>17.3f}{f_meas:>16.3f}"
          f"{u:>13.3f}{inv:>22.3f}")

print()
print(f"faithfulness rises {opt[0.97][0] / opt[0.52][0]:.2f}x and usefulness rises "
      f"{opt[0.97][1] / opt[0.52][1]:.2f}x")
print("because the ceiling on usefulness is context sufficiency, not faithfulness")

print()
print()
print("What actually moves usefulness: sufficiency.")
print()
print(f"{'sufficiency':>13}{'usefulness':>13}{'measured faith':>16}"
      f"{'gain per point':>17}")
print("-" * 59)
suff = {}
base_u = None
for s in (0.45, 0.55, 0.63, 0.75, 0.88):
    u = (s * P_FAITHFUL_SUFF * 1.00
         + s * (1 - P_FAITHFUL_SUFF) * 0.35
         + (1 - s) * P_FAITHFUL_INSUFF * 0.28)
    f_meas = s * P_FAITHFUL_SUFF + (1 - s) * P_FAITHFUL_INSUFF
    if base_u is None:
        base_u = (s, u)
    suff[s] = (u, f_meas)
    per = (u - base_u[1]) / (s - base_u[0]) if abs(s - base_u[0]) > 1e-9 else 0.0
    print(f"{s:>13.2f}{u:>13.3f}{f_meas:>16.3f}{per:>17.3f}")

print()
print()
print("The seven-failure-point catalogue, and which instrument sees each.")
print()
FAILURES = [
    ("missing content in the corpus",    "sufficiency annotation"),
    ("missed the top-ranked documents",  "recall@k"),
    ("not in the consolidated context",  "reranker audit"),
    ("not extracted from the context",   "utilisation probe"),
    ("wrong output format",              "schema check"),
    ("wrong specificity level",          "human or judge"),
    ("incomplete answer",                "human or judge"),
]
E2E_SEES = {
    "missing content in the corpus": "as a wrong answer",
    "missed the top-ranked documents": "as a wrong answer",
    "not in the consolidated context": "as a wrong answer",
    "not extracted from the context": "as a wrong answer",
    "wrong output format": "as a wrong answer",
    "wrong specificity level": "sometimes not at all",
    "incomplete answer": "sometimes not at all",
}
print(f"{'failure point':>34}{'instrument that localises it':>32}"
      f"{'end-to-end score shows':>24}")
print("-" * 90)
for name, inst in FAILURES:
    print(f"{name:>34}{inst:>32}{E2E_SEES[name]:>24}")

print()
print(f"{len(FAILURES)} failure points, {len(set(i for n, i in FAILURES))} "
      f"distinct instruments, and end-to-end accuracy")
print("distinguishes none of them")

print()
print()
print("Cost and coverage of the instruments, per 1000 queries.")
print()
INSTRUMENTS = [
    ("end-to-end correctness",   1, 3.40, 0.00, "tells you it is broken"),
    ("faithfulness (judge)",     1, 0.021, 0.00, "one axis of two"),
    ("recall@k on labelled set", 1, 0.000, 2.10, "one failure point"),
    ("utilisation probe",        1, 0.038, 0.00, "the largest bucket"),
    ("sufficiency annotation",   1, 4.80, 0.00, "the usefulness ceiling"),
    ("answer-span attribution",  1, 0.055, 0.00, "two failure points"),
]
print(f"{'instrument':>28}{'judge cost':>12}{'human cost':>12}"
      f"{'setup':>9}{'what it buys':>26}")
print("-" * 87)
inst_cost = {}
for name, n, jc, hc, buys in INSTRUMENTS:
    j = 1000 * jc if jc < 1 else 0.0
    h = 1000 * jc if jc >= 1 else 0.0
    setup = hc
    inst_cost[name] = j + h
    print(f"{name:>28}{j:>12,.0f}{h:>12,.0f}{setup:>9.1f}{buys:>26}")

cheap = ["faithfulness (judge)", "utilisation probe", "answer-span attribution"]
print()
print(f"the three automatable instruments together: "
      f"{sum(inst_cost[c] for c in cheap):,.0f} per 1000 queries")
print(f"end-to-end human correctness alone: "
      f"{inst_cost['end-to-end correctness']:,.0f}")
print(f"ratio: {inst_cost['end-to-end correctness'] / sum(inst_cost[c] for c in cheap):.0f}x")

print(f"""
The quadrant table is the whole argument and it takes one reading. Faithfulness is
{faith:.3f} and usefulness is {useful:.3f}, and the gap between them is entirely the
`insufficient context` row: an answer that stays honestly inside a context which did not
contain the answer is faithful and close to useless
(eq:faithfulness-and-usefulness-are-different-axes).

**Faithfulness measures whether the model lied. Sufficiency measures whether it could have
helped.** Only the first is computable without ground truth, which is why only the first
gets computed.

The optimisation table is what happens when a team acts on that. Pushing faithfulness on
insufficient context from {0.52:.2f} to {0.97:.2f} -- refuse rather than invent -- raises
measured faithfulness {opt[0.97][0] / opt[0.52][0]:.2f}x and usefulness
{opt[0.97][1] / opt[0.52][1]:.2f}x.

That is not nothing, and it is much less than the dashboard suggests, because
**the ceiling on usefulness is set by sufficiency and faithfulness cannot raise it**. The
one genuinely good thing it does is in the last column: confident inventions fall from
{opt[0.52][2]:.3f} to {opt[0.97][2]:.3f}, which is a safety result rather than a quality
one, and should be argued for on those terms.

The sufficiency table is the comparison. Taking sufficiency from {0.63:.2f} to
{0.88:.2f} moves usefulness from {suff[0.63][0]:.3f} to {suff[0.88][0]:.3f} --
{suff[0.88][0] - suff[0.63][0]:.3f}, against faithfulness optimisation's
{opt[0.97][1] - opt[0.52][1]:.3f} over its whole range.

Notice the third column while you are there. Measured faithfulness *rises* as sufficiency
rises, from {suff[0.45][1]:.3f} to {suff[0.88][1]:.3f}, without anyone touching the
generator -- because a model given adequate context stays inside it more readily. **A
faithfulness improvement can be a corpus improvement in disguise**, and the dashboard will
credit the model.

The failure-point table is cite:barnett2024sevenfailures' catalogue with a column added.
Seven distinct production failures, {len(set(i for n, i in FAILURES))} different instruments
needed to localise them, and end-to-end accuracy reports every one of them identically as
`a wrong answer` -- when it notices at all
(eq:most-rag-failures-are-invisible-end-to-end).

Two of them it does not notice: wrong specificity and incompleteness produce answers that
are true, supported, and not what was asked for. Those pass a correctness check and fail a
user.

The instrument table is the practical answer and the numbers are friendlier than the
argument so far suggests. A faithfulness judge, a utilisation probe and a span-attribution
check together cost {sum(inst_cost[c] for c in cheap):,.0f} per thousand queries, against
{inst_cost['end-to-end correctness']:,.0f} for human end-to-end correctness --
{inst_cost['end-to-end correctness'] / sum(inst_cost[c] for c in cheap):.0f} times cheaper --
and between them they localise the four largest buckets from
ch:ev-rag's first listing.

The expensive instrument is sufficiency annotation, and it is expensive because somebody has
to read the corpus and decide whether the answer was in there. It is also the only one that
measures the ceiling. **Sample it rather than skipping it**: a few hundred annotated queries
a quarter gives you the number that bounds every other metric on the dashboard, and without
it a RAG programme is optimising terms in a product whose largest factor is unmeasured.""")
