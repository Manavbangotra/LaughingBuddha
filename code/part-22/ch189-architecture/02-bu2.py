# -*- coding: utf-8 -*-
# Extracted from: Chapter 189 — Architecting Production AI Systems
# Source: src/.../ch189-architecture.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where to put the boundary between the deterministic system and the model.

Every stage of a request can be done by code or by a model. Code is testable,
cheap and deterministic; a model handles inputs nobody enumerated. So the
architecture question is not WHETHER to use a model but HOW DEEP to let it reach
(eq:boundary-decides-testability).

This is ch:ag-what-is-an-agent's router-versus-agent decision and ch:as-graph's
graph-versus-loop decision, restated at the level of a whole system. The new part
is what it does to the properties the first listing said were broken: a
deterministic stage keeps caching, retries, golden tests and health checks
working, and a model stage does not.
"""
# A request pipeline. Each stage can be code or model.
# (stage, share of requests where code alone suffices, cost of a model call,
#  cost of the code path, share of the request's value it carries)
STAGES = [
    ("parse the request",     0.94, 1.0, 0.02, 0.10),
    ("classify intent",       0.71, 1.0, 0.02, 0.15),
    ("gather context",        0.88, 1.4, 0.05, 0.15),
    ("decide what to do",     0.34, 2.2, 0.02, 0.30),
    ("compose the answer",    0.12, 3.0, 0.02, 0.25),
    ("format and validate",   0.97, 0.8, 0.01, 0.05),
]
N = len(STAGES)


def design(depth):
    """`depth` is how many stages, counting from the last, the model handles.
    depth=0 is pure code; depth=N is model all the way in.

    Returns (coverage, cost, testable share).
    """
    cov = 1.0
    cost = 0.0
    testable = 0.0
    for i, (name, code_ok, m_cost, c_cost, weight) in enumerate(STAGES):
        if i >= N - depth:
            cost += m_cost
            cov *= 0.985            # a model stage handles nearly anything
        else:
            cost += c_cost
            cov *= code_ok          # code fails on what it did not anticipate
            testable += weight
    return cov, cost, testable


print("A six-stage request pipeline. Each stage is handled by code or by the")
print("model; the boundary is how deep the model reaches.")
print()
print(f"{'stage':>22}{'code suffices':>15}{'model cost':>12}{'code cost':>11}"
      f"{'weight':>9}")
print("-" * 69)
for name, ok, mc, cc, w in STAGES:
    print(f"{name:>22}{ok:>15.0%}{mc:>12.1f}{cc:>11.2f}{w:>9.0%}")

print()
print()
print("Every boundary position.")
print()
print(f"{'model handles':>28}{'coverage':>11}{'cost':>8}{'testable':>11}")
print("-" * 58)
tab = {}
for d in range(N + 1):
    cov, cost, testable = design(d)
    tab[d] = (cov, cost, testable)
    label = ("nothing (pure code)" if d == 0 else
             "everything" if d == N else "the last %d stages" % d)
    print(f"{label:>28}{cov:>11.1%}{cost:>8.2f}{testable:>11.0%}")

print()
print()
print("What each additional stage of model reach buys and costs.")
print()
print(f"{'model handles':>16}{'coverage gain':>15}{'cost added':>12}"
      f"{'testable lost':>15}")
print("-" * 58)
mg = {}
for d in range(1, N + 1):
    dc = tab[d][0] - tab[d - 1][0]
    dk = tab[d][1] - tab[d - 1][1]
    dt = tab[d - 1][2] - tab[d][2]
    mg[d] = (dc, dk, dt)
    print(f"{d:>16}{dc:>+15.1%}{dk:>+12.2f}{dt:>+15.0%}")

print()
print()
print("Per stage, ignoring position: how much coverage a model buys there, and")
print("what it costs.")
print()
print(f"{'stage':>22}{'code suffices':>15}{'coverage bought':>17}"
      f"{'cost':>8}{'per cost':>11}")
print("-" * 73)
per = {}
for name, ok, mc, cc, w in STAGES:
    gain = 0.985 - ok
    cost = mc - cc
    per[name] = gain / cost
    print(f"{name:>22}{ok:>15.0%}{gain:>+17.1%}{cost:>8.2f}"
          f"{gain / cost:>11.3f}")

print()
print()
print("Ranked, against the intuition that a model belongs everywhere or nowhere.")
print()
order = sorted(per, key=lambda k: -per[k])
look = {s[0]: s for s in STAGES}
print(f"{'rank':>6}{'stage':>22}{'per cost':>11}{'code suffices':>15}")
print("-" * 54)
for i, name in enumerate(order, 1):
    print(f"{i:>6}{name:>22}{per[name]:>11.3f}{look[name][1]:>15.0%}")

print()
print()
print("And the property this part turns on: what share of the system keeps")
print("caching, retries, golden tests and health checks working.")
print()
print(f"{'model handles':>28}{'coverage':>11}{'testable':>11}"
      f"{'classical techniques':>22}")
print("-" * 72)
for d in (0, 2, 3, 6):
    cov, cost, testable = tab[d]
    label = ("nothing" if d == 0 else "everything" if d == N
             else "the last %d stages" % d)
    verdict = ("all work" if testable > 0.9 else
               "most work" if testable > 0.6 else
               "few work" if testable > 0.2 else "none work")
    print(f"{label:>28}{cov:>11.1%}{testable:>11.0%}{verdict:>22}")

print(f"""
The boundary table is the trade in one place. Pure code covers
{tab[0][0]:.1%} of requests at a cost of {tab[0][1]:.2f} and keeps
{tab[0][2]:.0%} of the system testable. Model-all-the-way covers
{tab[6][0]:.1%} at {tab[6][1]:.2f} and keeps {tab[6][2]:.0%}.

**The boundary decides how much of the system retains the properties
ch:sd-architecture's first listing said were broken**
(eq:boundary-decides-testability) -- caching, retries, golden tests and health
checks all work on a deterministic stage and none of them work on a model stage.

That is the same structure as ch:as-graph's graph-versus-loop and
ch:ag-what-is-an-agent's router-versus-agent, and it lands in the same place: the
extremes are both wrong, and the interesting question is where in between.

But the per-stage table says something the boundary framing cannot express, and it
is the more useful finding.

Ranked by coverage bought per unit of cost, the model earns its place at
`{order[0]}` ({per[order[0]]:.3f}), `{order[1]}` ({per[order[1]]:.3f}) and
`{order[2]}` ({per[order[2]]:.3f}). It does not earn its place at
`{order[-1]}` ({per[order[-1]]:.3f}) or `{order[-2]}` ({per[order[-2]]:.3f}) --
roughly a factor of fifteen between the best and worst stages.

Now look at where those stages sit. Classify is the SECOND stage; parse and gather
are first and third. **The stages where a model earns its place are not
contiguous**, so there is no depth at which a single boundary captures them.

Which means the boundary framing -- the model handles everything from stage k
onward -- is the wrong shape. The right design **interleaves**: deterministic
parsing, model classification, deterministic retrieval, model decision, model
composition, deterministic validation.

That has a practical consequence worth stating plainly. Each deterministic stage
between model stages is a place where the classical techniques come back:
you can cache the retrieval, unit-test the parser, health-check the validator, and
retry the deterministic parts safely. **Interleaving does not merely cost less
than model-everywhere; it restores testability in the gaps**, and the gaps are
where the operational instruments live.

The last table prices the extremes honestly. Pure code covers
{tab[0][0]:.1%} of requests, which is not a viable product -- six stages at
{0.94:.0%} to {0.12:.0%} multiply badly, and that multiplication is
ch:ag-loop's chain arriving in an architecture diagram. Model-everywhere covers
{tab[6][0]:.1%} and leaves nothing testable.

The design that puts the model at the three stages that earn it covers most of the
gap and keeps the other three stages instrumented -- which is the recommendation,
and it is neither of the two positions the debate is usually conducted between.""")
