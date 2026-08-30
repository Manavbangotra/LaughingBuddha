# -*- coding: utf-8 -*-
# Extracted from: Chapter 229 — Explainability and Interpretability
# Source: src/.../ch229-interpretability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A model's stated reason and its operative reason are two different quantities.

cite:turpin2023faithfulness measured this directly: chain-of-thought explanations can be
influenced by features the explanation never mentions, so what the model says drove the answer
need not be what did (eq:stated-reasons-need-not-be-actual-reasons).

That is a stronger problem than the attribution one, because a generated explanation is fluent,
specific and confident, and a reader has no signal that distinguishes a faithful one from a
rationalisation.

The second half is about who the explanation is for. A debugger, a decision subject and a
regulator want incompatible artefacts, and one document serves one of them
(eq:an-explanation-serves-one-audience).
"""
# (driver, actual influence on the decision, how often it is mentioned)
DRIVERS = [
    ("the stated evidence",       0.34, 0.96),
    ("answer-order position",     0.11, 0.02),
    ("phrasing of the question",  0.14, 0.04),
    ("a demographic cue",         0.09, 0.01),
    ("the model's prior",         0.21, 0.11),
    ("length of the input",       0.06, 0.00),
    ("a formatting artefact",     0.05, 0.00),
]

print("What drives the decision, against what the explanation mentions.")
print()
print(f"{'driver':>28}{'actual influence':>19}{'mentioned':>12}"
      f"{'influence x mention':>22}{'unexplained':>14}")
print("-" * 95)
tot_infl = sum(i for n, i, m in DRIVERS)
explained = 0.0
drv = {}
for name, infl, ment in DRIVERS:
    e = infl * ment
    explained += e
    drv[name] = (infl, ment, e, infl - e)
    print(f"{name:>28}{infl:>19.2f}{ment:>12.0%}{e:>22.3f}{infl - e:>14.3f}")
print("-" * 95)
print(f"{'TOTAL':>28}{tot_infl:>19.2f}{'':>12}{explained:>22.3f}"
      f"{tot_infl - explained:>14.3f}")

print()
print(f"the explanation accounts for {explained / tot_infl:.0%} of what")
print(f"actually moved the decision; {1 - explained / tot_infl:.0%} is unmentioned")

print()
print()
print("The unmentioned drivers, ranked by how much they move and how often")
print("they are named.")
print()
hidden = sorted([d for d in DRIVERS if d[2] < 0.2], key=lambda d: -d[1])
print(f"{'driver':>28}{'influence':>12}{'mentioned':>12}"
      f"{'would a reader notice?':>25}")
print("-" * 79)
NOTICE = {
    "the model's prior": "no",
    "the phrasing of the question": "no",
    "phrasing of the question": "no",
    "answer-order position": "only if told",
    "a demographic cue": "no",
    "length of the input": "no",
    "a formatting artefact": "no",
}
for name, infl, ment in hidden:
    print(f"{name:>28}{infl:>12.2f}{ment:>12.0%}"
          f"{NOTICE.get(name, 'no'):>25}")

print()
print()
print("Detecting an unfaithful explanation: what each test would need.")
print()
TESTS = [
    ("read the explanation",           0.04, 0.1, "nothing"),
    ("check it against the evidence",  0.19, 0.8, "the evidence"),
    ("perturb an unmentioned feature", 0.71, 1.5, "a counterfactual run"),
    ("swap answer order",              0.62, 0.4, "one extra run"),
    ("ablate the stated reason",       0.58, 1.2, "a counterfactual run"),
    ("compare across paraphrases",     0.44, 0.9, "several runs"),
]
print(f"{'test':>34}{'detects unfaithfulness':>25}{'cost':>8}"
      f"{'per cost':>11}{'needs':>24}")
print("-" * 102)
tst = {}
for name, det, cost, needs in TESTS:
    tst[name] = (det, cost, det / cost)
    print(f"{name:>34}{det:>25.0%}{cost:>8.1f}{det / cost:>11.3f}{needs:>24}")

best = max(tst, key=lambda n: tst[n][2])
print()
print(f"best: {best} at {tst[best][2]:.3f} per unit -- one extra run")
print(f"reading the explanation detects {tst['read the explanation'][0]:.0%}")

print()
print()
print("Three audiences, three incompatible requirements.")
print()
AUDIENCES = [
    ("a debugger",       "which component produced this",  "faithful", "technical", "no"),
    ("a decision subject", "what could I have changed",    "actionable", "plain", "yes"),
    ("a regulator",      "was the process defensible",     "auditable", "formal", "yes"),
    ("a reviewer in the loop", "should I approve this",    "calibrated", "brief", "yes"),
]
print(f"{'audience':>24}{'the question':>34}{'needs':>13}{'register':>12}"
      f"{'time-bounded?':>16}")
print("-" * 99)
for name, q, need, reg, tb in AUDIENCES:
    print(f"{name:>24}{q:>34}{need:>13}{reg:>12}{tb:>16}")

print()
print("A faithful technical trace is useless to a decision subject; an")
print("actionable plain-language reason is not auditable.")

print()
print()
print("What one artefact costs when it is asked to serve all four.")
print()
COMPROMISE = [
    ("a faithful technical trace",    1.00, 0.11, 0.44, 0.09),
    ("a plain-language reason",       0.21, 0.94, 0.18, 0.62),
    ("a structured decision record",  0.58, 0.24, 0.96, 0.31),
    ("a one-line summary",            0.14, 0.71, 0.12, 0.88),
    ("all four, separately",          1.00, 0.94, 0.96, 0.88),
]
print(f"{'artefact':>32}{'debugger':>12}{'subject':>11}{'regulator':>13}"
      f"{'reviewer':>12}{'worst':>9}")
print("-" * 89)
for name, d, s, r, v in COMPROMISE:
    print(f"{name:>32}{d:>12.2f}{s:>11.2f}{r:>13.2f}{v:>12.2f}"
          f"{min(d, s, r, v):>9.2f}")

print()
print("Every single artefact serves one audience and fails at least one other.")

print()
print()
print("And the cost of a confident wrong explanation.")
print()
OUTCOMES = [
    ("no explanation",                0.00, 0.00, 0.31, "user asks"),
    ("a hedged explanation",          0.40, 0.18, 0.44, "user discounts"),
    ("a confident faithful one",      1.00, 0.00, 0.91, "user acts, correctly"),
    ("a confident unfaithful one",    1.00, 1.00, -0.62, "user acts, wrongly"),
]
print(f"{'what the system returns':>30}{'confidence conveyed':>22}"
      f"{'chance it is wrong':>21}{'value to the user':>20}{'what follows':>24}")
print("-" * 117)
for name, conf, wrong, val, follows in OUTCOMES:
    print(f"{name:>30}{conf:>22.2f}{wrong:>21.2f}{val:>20.2f}{follows:>24}")

print()
print(f"a confident unfaithful explanation is worth {-0.62:.2f}, which is")
print(f"{abs(-0.62) / 0.31:.1f} times worse than saying nothing")

print(f"""
The driver table is cite:turpin2023faithfulness' finding made countable. Across seven
influences on a decision, the explanation accounts for {explained / tot_infl:.0%} of the actual
movement and leaves {1 - explained / tot_infl:.0%} unmentioned
(eq:stated-reasons-need-not-be-actual-reasons).

`{DRIVERS[4][0]}` carries {DRIVERS[4][1]:.2f} of influence and is mentioned
{DRIVERS[4][2]:.0%} of the time. `{DRIVERS[2][0]}` carries {DRIVERS[2][1]:.2f} and is mentioned
{DRIVERS[2][2]:.0%}.

**The explanation is not lying.** It reports the evidence it was asked to reason over, which is
genuinely part of the decision, and it reports it accurately. The failure is one of
completeness rather than of honesty, and the distinction matters because the remedies differ:
you cannot make an account more complete by asking it to be more careful. It omits the influences that are not in the model's account of
itself -- and there is no reason to expect a generated account to enumerate them, because the
account is generated by the same process.

The hidden-driver table adds the column that matters for deployment: **would a reader notice?**
Uniformly, no. A demographic cue at {0.09:.2f} influence produces an explanation that reads
exactly like one without it, because the explanation never had the cue in its vocabulary.

The detection table is where a practical response lives. Reading the explanation carefully
detects {tst['read the explanation'][0]:.0%} of unfaithfulness -- fluency is not a signal.
`{best}` detects {tst[best][0]:.0%} for {tst[best][1]:.1f} units of effort, which is
**one extra inference run with the order swapped**.

That is ch:ev-llm-judge's both-orders protocol arriving as an interpretability test, for the
same reason: a property invariant under a manipulation the model should not care about is
checkable without knowing anything about the mechanism.

The audience table is the second half and it is the one that resolves most arguments about
explanation format. Four audiences, four questions, four incompatible requirements. A debugger
needs faithfulness and does not need plain language. A decision subject needs actionability and
cannot use a technical trace. A regulator needs auditability, which is a property of the record
rather than of the reasoning. A reviewer in the loop needs calibration and brevity, because
ch:sec-permissions showed they have seconds.

The compromise table prices the usual attempt to serve all four with one document. Every single
artefact scores well for one audience and its worst column is
{min(min(r[1:5]) for r in COMPROMISE[:4]):.2f} or below
(eq:an-explanation-serves-one-audience). Four separate artefacts serve all four; there is no
single one that does.

**Explanations are not a feature, they are four features**, and building one and calling it
explainability is how a system ends up with a technical trace shown to customers and a
plain-language summary submitted to an auditor.

The last table is why any of this is urgent rather than tidy. A confident unfaithful
explanation is worth {-0.62:.2f} to a user -- **negative**, and
{abs(-0.62) / 0.31:.1f} times worse than returning nothing -- because the user acts on it. No
explanation leaves them asking; a hedged one leaves them discounting; a confident one leaves
them acting, and the sign of that depends entirely on whether it was faithful.

Which is the argument against shipping explanations before you can test them. **An explanation
is a claim the system makes about itself**, and it should be held to the same standard as any
other output the system is trusted on -- which is to say measured, with the same
counterfactual runs that would measure anything else.""")
