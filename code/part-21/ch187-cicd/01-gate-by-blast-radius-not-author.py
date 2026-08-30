# -*- coding: utf-8 -*-
# Extracted from: Chapter 187 — CI/CD and Architecture Agents
# Source: src/.../ch187-cicd.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Where the gate goes in a pipeline that automated changes flow through.

ch:ag-termination found that gating everything is close to gating nothing, and
ch:as-long-running found placement worth an eightfold review budget over frequency.
This listing applies both to a delivery pipeline, where the natural instinct is to
gate by AUTHOR -- humans merge freely, agents need approval -- and the measurements
say to gate by BLAST RADIUS instead (eq:gate-by-blast-radius-not-author).

A change flows: propose -> automated checks -> human review -> merge -> deploy. Each
stage can catch a defect, and each costs something. The catch rates differ by change
type, because a type checker has a lot to say about a refactor and nothing to say
about a config value.
"""
M = 50000               # changes per period, for the volume framing

# (change type, share of volume, defect rate, automated catch, review catch,
#  cost if it reaches production)
CHANGES = [
    ("docs and comments",   0.22, 0.04, 0.10, 0.55,    2.0),
    ("test-only",           0.14, 0.09, 0.72, 0.60,    5.0),
    ("dependency bump",     0.11, 0.16, 0.55, 0.35,  140.0),
    ("bug fix",             0.25, 0.14, 0.68, 0.62,   90.0),
    ("feature code",        0.19, 0.19, 0.61, 0.58,  120.0),
    ("config / infra",      0.06, 0.21, 0.24, 0.45, 900.0),
    ("schema migration",    0.03, 0.24, 0.31, 0.66, 3400.0),
]
REVIEW_MIN = 11.0       # analyst-minutes for a human review


def run(gated, attention=1.0):
    """`gated` is the set of change types a human reviews. Returns
    (expected cost per change, review minutes per change, escape rate).

    Computed exactly rather than sampled: the model is a product of
    independent probabilities, so simulation would only add noise to a
    quantity with a closed form -- and the noise swamps the small-volume
    types that turn out to matter most.
    """
    total_cost = 0.0
    minutes = 0.0
    escapes = 0.0
    for name, share, p_def, auto, rev, cost in CHANGES:
        p_escape = p_def * (1.0 - auto)
        if name in gated:
            minutes += share * REVIEW_MIN
            p_escape *= (1.0 - min(rev * attention, 1.0))
        escapes += share * p_escape
        total_cost += share * p_escape * cost
    return total_cost, minutes, escapes


ALL = {c[0] for c in CHANGES}

print(f"{M:,} changes through a pipeline. Each type has its own defect")
print("rate, its own automated catch rate, and its own cost if it escapes.")
print()
print(f"{'change type':>20}{'volume':>9}{'defects':>9}{'auto catch':>12}"
      f"{'escape cost':>13}")
print("-" * 63)
for name, share, p_def, auto, rev, cost in CHANGES:
    print(f"{name:>20}{share:>9.0%}{p_def:>9.0%}{auto:>12.0%}{cost:>13,.0f}")

print()
print()
print("Gating policies. 'Cost' is expected escape cost per change; 'minutes'")
print("is human review time per change.")
print()
print(f"{'policy':>34}{'cost/change':>13}{'minutes':>10}{'escapes':>10}")
print("-" * 67)
POLICIES = [
    ("gate nothing", set()),
    ("gate everything", ALL),
    ("gate by author (all agent changes)", ALL),
    ("gate the big diffs", {"feature code", "bug fix"}),
    ("gate by blast radius", {"schema migration", "config / infra",
                              "dependency bump"}),
]
tab = {}
for label, g in POLICIES:
    r = run(g)
    tab[label] = r
    print(f"{label:>34}{r[0]:>13.1f}{r[1]:>10.1f}{r[2]:>10.2%}")

print()
print()
print("Cost per review-minute spent, which is the comparison that matters when")
print("review capacity is the constraint.")
print()
none = tab["gate nothing"][0]
print(f"{'policy':>34}{'cost avoided':>14}{'minutes':>10}{'per minute':>12}")
print("-" * 70)
for label, g in POLICIES:
    r = tab[label]
    if r[1] <= 0:
        continue
    print(f"{label:>34}{none - r[0]:>14.1f}{r[1]:>10.1f}"
          f"{(none - r[0]) / r[1]:>12.2f}")

print()
print()
print("Every single-type gate, ranked. This is the table a team should build")
print("for its own pipeline.")
print()
print(f"{'gate only this type':>20}{'cost avoided':>14}{'minutes':>10}"
      f"{'per minute':>12}")
print("-" * 56)
single = {}
for name, share, p_def, auto, rev, cost in CHANGES:
    r = run({name})
    single[name] = ((none - r[0]), r[1], (none - r[0]) / max(r[1], 1e-9))
    print(f"{name:>20}{none - r[0]:>14.2f}{r[1]:>10.2f}"
          f"{(none - r[0]) / max(r[1], 1e-9):>12.2f}")

print()
print()
print("Note what does NOT predict the ranking.")
print()
order = sorted(single, key=lambda k: -single[k][2])
look = {c[0]: c for c in CHANGES}
print(f"{'rank':>6}{'type':>20}{'per minute':>12}{'volume':>9}{'defects':>9}"
      f"{'escape cost':>13}")
print("-" * 69)
for i, name in enumerate(order, 1):
    c = look[name]
    print(f"{i:>6}{name:>20}{single[name][2]:>12.2f}{c[1]:>9.0%}"
          f"{c[2]:>9.0%}{c[5]:>13,.0f}")

print()
print()
print("And what happens under ch:ag-termination's habituation, which is what")
print("gating everything actually produces at agent volumes.")
print()
print(f"{'attention':>11}{'gate everything':>18}{'gate by blast radius':>22}")
print("-" * 51)
hb = {}
for a in (1.0, 0.6, 0.3, 0.12):
    x = run(ALL, attention=a)[0]
    y = run({"schema migration", "config / infra", "dependency bump"},
            attention=min(a * 3.5, 1.0))[0]
    hb[a] = (x, y)
    print(f"{a:>11.0%}{x:>18.1f}{y:>22.1f}")

print(f"""
The policy table has the finding in two rows. Gating by AUTHOR -- every agent
change reviewed -- costs {tab['gate by author (all agent changes)'][1]:.1f} review
minutes per change and leaves {tab['gate by author (all agent changes)'][0]:.1f} of
expected escape cost. Gating by BLAST RADIUS costs
{tab['gate by blast radius'][1]:.1f} minutes and leaves
{tab['gate by blast radius'][0]:.1f}.

Roughly the same protection for about a fifth of the review time
(eq:gate-by-blast-radius-not-author), which is ch:as-long-running's
placement-beats-frequency result arriving in a delivery pipeline.

The per-minute table makes the ranking stark. A gate on schema migrations returns
{single['schema migration'][2]:.1f} units of avoided cost per review-minute; a gate
on documentation returns {single['docs and comments'][2]:.2f}. **A factor of
thousands separates the best gate from the worst**, and both are gates on changes an
agent might author.

The next table says what does not predict that ranking, and it is worth checking
against intuition.

Not VOLUME: documentation is {0.22:.0%} of changes and ranks last. Not DEFECT RATE:
feature code has a {0.19:.0%} rate and ranks third. Not who wrote it -- the model
does not contain an author variable at all.

What predicts it is **escape cost multiplied by what the automated checks do not
already catch.** Schema migrations are {0.03:.0%} of volume with a {0.31:.0%}
automated catch rate and a {3400:,} escape cost, and that product is the whole
ranking.

Which gives the practical instruction: **build this table for your own pipeline.**
It requires three numbers per change type -- how often they are wrong, what your CI
catches, what it costs when one escapes -- and the first two are recoverable from
history.

The last table is why the author-based policy is worse than it looks on paper.

At full attention, gating everything costs {hb[1.0][0]:.1f} against blast-radius
gating's {hb[1.0][1]:.1f} -- the broad policy is nominally better. At
{0.30:.0%} attention it is {hb[0.3][0]:.1f} against {hb[0.3][1]:.1f}.

**Gating everything consumes the attention that makes gating work.**
ch:ag-termination measured that curve directly; here it means a policy that looks
safer on a spreadsheet is worse in a pipeline that actually runs, because the
reviewers are seeing five times the volume and reading it five times less carefully.

The blast-radius policy is robust to habituation for a mechanical reason: its
reviewers see about {0.20:.0%} of the changes, so their attention holds where the
broad policy's does not. That is the argument for narrow gating stated as an
attention budget rather than as a preference.""")
