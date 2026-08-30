# -*- coding: utf-8 -*-
# Extracted from: Chapter 188 — AI-Assisted versus Autonomous Software Engineering
# Source: src/.../ch188-autonomy.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What has to be true before autonomy pays, which is not a fact about the model.

Every chapter in part:21 identified something the ENVIRONMENT has to supply.
ch:aise-repo found reproduction the best localiser. ch:aise-swe-agents found the
test runner and the iteration loop mutually contingent, and the scaffold worth more
than a large model improvement. ch:aise-testing found suite independence deciding
what iteration means. ch:aise-cicd found gating, rollback and architectural
constraints deciding what a change may do.

None of those is a model capability. This listing collects them, ablates them, and
asks what fraction of safe autonomy each unlocks
(eq:autonomy-is-an-environment-property).
"""
# (prerequisite, what it raises, magnitude, what it depends on)
PREREQS = [
    ("reproduction available",     "localisation", 0.30, None),
    ("test coverage",              "verification", 0.34, None),
    ("independent tests",          "verification", 0.26, "test coverage"),
    ("fast CI",                    "iteration",    0.22, "test coverage"),
    ("blast-radius gating",        "containment",  0.28, None),
    ("rollback path",              "containment",  0.31, None),
    ("architectural constraints",  "verification", 0.19, None),
]
ALL = {p[0] for p in PREREQS}

BASE = {"localisation": 0.53, "verification": 0.30,
        "iteration": 0.20, "containment": 0.25}


def capability(have):
    """Returns the four environment capabilities, given the prerequisites in
    place. A prerequisite with an unmet dependency contributes nothing."""
    caps = dict(BASE)
    for name, raises, mag, dep in PREREQS:
        if name not in have:
            continue
        if dep is not None and dep not in have:
            continue                       # the precondition is missing
        caps[raises] = caps[raises] + (1.0 - caps[raises]) * mag
    return caps


def safe_autonomy(have):
    """Share of changes an agent can complete unsupervised without an escape.
    All four capabilities are required: find it, check it, fix it, contain it."""
    c = capability(have)
    return (c["localisation"] * c["verification"] * c["iteration"] ** 0.5
            * c["containment"])


print("Seven environment prerequisites, each raising one capability an")
print("autonomous agent needs. None of them is a property of the model.")
print()
print(f"{'prerequisite':>28}{'raises':>15}{'by':>8}{'depends on':>18}")
print("-" * 69)
for name, raises, mag, dep in PREREQS:
    print(f"{name:>28}{raises:>15}{mag:>8.0%}{(dep or '--'):>18}")

none = safe_autonomy(set())
full = safe_autonomy(ALL)
print()
print(f"   nothing in place: {none:.1%} of changes safely autonomous")
print(f"   everything:       {full:.1%}")

print()
print()
print("Each prerequisite ADDED to nothing, and REMOVED from everything --")
print("ch:as-single-agent's methodology, which this part has needed repeatedly.")
print()
print(f"{'prerequisite':>28}{'added alone':>14}{'removed from all':>19}")
print("-" * 61)
ab = {}
for name, raises, mag, dep in PREREQS:
    added = safe_autonomy({name}) - none
    removed = full - safe_autonomy(ALL - {name})
    ab[name] = (added, removed)
    print(f"{name:>28}{added:>+14.1%}{removed:>+19.1%}")

print()
print()
print("Building them up in a sensible order.")
print()
ORDER = ["test coverage", "independent tests", "reproduction available",
         "fast CI", "rollback path", "blast-radius gating",
         "architectural constraints"]
print(f"{'after adding':>28}{'safe autonomy':>16}{'gain':>9}")
print("-" * 53)
bu = {}
have, prev = set(), none
for name in ORDER:
    have.add(name)
    v = safe_autonomy(set(have))
    bu[name] = (v, v - prev)
    print(f"{name:>28}{v:>16.1%}{v - prev:>+9.1%}")
    prev = v

print()
print()
print("The four capabilities separately, at each stage of that build-up.")
print()
print(f"{'after adding':>28}{'localise':>11}{'verify':>9}{'iterate':>10}"
      f"{'contain':>10}")
print("-" * 68)
have = set()
for name in ORDER:
    have.add(name)
    c = capability(set(have))
    print(f"{name:>28}{c['localisation']:>11.0%}{c['verification']:>9.0%}"
          f"{c['iteration']:>10.0%}{c['contain' + 'ment']:>10.0%}")

print()
print()
print("And the comparison this part has been building toward. A model that is")
print("better at every step, against an environment that supplies these.")
print()
print(f"{'scenario':>44}{'safe autonomy':>16}")
print("-" * 62)
sc = {}
for label, have, skill in (
        ("today's model, nothing in place", set(), 1.00),
        ("a 25% better model, nothing in place", set(), 1.25),
        ("a 60% better model, nothing in place", set(), 1.60),
        ("today's model, all seven in place", ALL, 1.00)):
    c = capability(have)
    v = (min(c["localisation"] * skill, 0.99)
         * min(c["verification"] * skill, 0.99)
         * min(c["iteration"] * skill, 0.99) ** 0.5
         * c["containment"])
    sc[label] = v
    print(f"{label:>44}{v:>16.1%}")

print(f"""
The build-up table is the part's practical output, and the largest single gain is
not where most attention goes.

**A rollback path is worth {bu['rollback path'][1]:+.1%}** -- the biggest step in
the table -- and it is infrastructure work with no machine learning in it. Blast-radius
gating adds {bu['blast-radius gating'][1]:+.1%}. Between them, containment accounts
for more than a third of the total.

That is ch:as-specialized's finding restated for a seventh time: reversibility was
the property that explained most of the spread there, and it explains the largest
step here. **Being able to undo a change is worth more than being better at making
it.**

The ablation table shows the same contingency this part has hit repeatedly.
Independent tests added alone are worth {ab['independent tests'][0]:+.1%} and removed
from a complete environment cost {ab['independent tests'][1]:+.1%}, because
independence is worthless without coverage to be independent about -- which is why
the model makes it depend on test coverage explicitly.

The capability table shows what each prerequisite actually moves, and the columns do
not fill evenly. Verification reaches {0.72:.0%} and iteration only {0.38:.0%},
because iteration depends on fast CI and fast CI depends on a suite that exists. The
binding capability at the end of the build-up is the one that started lowest and had
the fewest contributors.

And the last table is what part:21 has been building toward.

Today's model with none of these in place reaches {sc["today's model, nothing in place"]:.1%}
safe autonomy. A model {0.60:.0%} better -- an enormous improvement, larger than any
single generation has delivered -- with none of them in place reaches
{sc['a 60% better model, nothing in place']:.1%}. **Today's model with all seven in
place reaches {sc["today's model, all seven in place"]:.1%}**, which is
{sc["today's model, all seven in place"] / sc['a 60% better model, nothing in place']:.1f}
times the better model's figure.

That is cite:chan2024mlebench's scaffolding result, extended from the agent's loop to
the environment the loop runs in, and it is the answer to the question teams
actually ask. **Autonomy is not a capability you wait for. It is a set of properties
you build** (eq:autonomy-is-an-environment-property), and the properties are
enumerable, ordinary, and mostly already on someone's backlog.

Note the absolute level honestly: {sc["today's model, all seven in place"]:.1%} of
changes safely autonomous is not a large number, and it should not be read as one.
It is the share requiring no supervision at all across every change type including
the ones ch:aise-cicd said to gate. The useful reading is the ratio between the rows,
not the rows themselves -- and the ratio says the environment is where the leverage
is.

Which is also why the same model produces such different experiences at different
organisations. Not model access, not prompting, not talent. **A team with
reproduction, coverage, independent tests, fast CI, gating, rollback and executable
constraints is operating a different system**, and it is the system rather than the
model that this part has been measuring.""")
