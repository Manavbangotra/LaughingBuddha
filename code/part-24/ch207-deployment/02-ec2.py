# -*- coding: utf-8 -*-
# Extracted from: Chapter 207 — Deployment Strategies and Rollback
# Source: src/.../ch207-deployment.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Rolling back the change does not roll back its effects.

A rollback restores the previous version of whatever you deployed. It does not restore
the state that version produced while it was live -- caches populated with its answers,
records it wrote, conversations it shaped, an index it rebuilt.

So reverting is only a full remedy when the change had no persistent effects, and
ch:sd-storage established that an AI system is full of persistent derived state
(eq:rollback-restores-code-not-state).

This listing measures what share of a change's damage a rollback actually recovers, and
finds the answer is set by how long the change was live -- which ch:ops-deployment's
canary arithmetic says is a long time.
"""
# (effect, share of the damage it accounts for, does rollback undo it?, decay hrs)
EFFECTS = [
    ("answers already served",     0.31, False,   0.0),
    ("answers cached semantically", 0.19, False,  36.0),
    ("records written by tools",   0.14, False,   0.0),
    ("conversation state shaped",  0.11, False,  72.0),
    ("evaluation baselines moved", 0.06, False, 168.0),
    ("index rebuilt with new embeddings", 0.09, False, 8.0),
    ("in-flight requests",         0.04, True,    0.0),
    ("future requests",            0.06, True,    0.0),
]
LIVE_HOURS = [1.0, 6.0, 24.0, 112.0, 336.0]

recoverable = sum(e[1] for e in EFFECTS if e[2])
print("What a bad deploy leaves behind, and whether reverting the deploy undoes it.")
print()
print(f"{'effect':>36}{'share of damage':>18}{'rollback undoes':>18}"
      f"{'self-heals in':>16}")
print("-" * 90)
for name, share, undone, decay in EFFECTS:
    heal = "never" if decay == 0.0 else f"{decay:.0f}h"
    print(f"{name:>36}{share:>18.0%}{('yes' if undone else 'no'):>18}"
          f"{heal:>16}")
print()
print(f"rollback directly undoes {recoverable:.0%} of the damage")

print()
print()
print("What the rest costs, by how long the change was live before rollback.")
print("Some effects decay on their own; most do not.")
print()
print(f"{'live hours':>12}{'undone by rollback':>21}{'self-healed in 7d':>20}"
      f"{'permanent':>12}{'recovered':>12}")
print("-" * 78)
tab = {}
WINDOW = 168.0
for lh in LIVE_HOURS:
    undone = recoverable
    healed = 0.0
    perm = 0.0
    for name, share, u, decay in EFFECTS:
        if u:
            continue
        # A persistent effect accumulated over `lh` hours; the part with a decay
        # constant fades within the observation window.
        if decay > 0:
            healed += share * min(1.0, WINDOW / (decay + lh))
            perm += share * (1.0 - min(1.0, WINDOW / (decay + lh)))
        else:
            perm += share
    tab[lh] = (undone, healed, perm, undone + healed)
    print(f"{lh:>12.0f}{undone:>21.0%}{healed:>20.0%}{perm:>12.0%}"
          f"{undone + healed:>12.0%}")

print()
print()
print("Composing with the canary arithmetic: how long a change is live before")
print("rollback is the detection time, which the canary share determines.")
print()
SHARES = [0.01, 0.05, 0.20, 0.50]
DETECT_AT_FULL = 22.3        # hours to detect at 100% traffic, from ch:sd-fault-tolerance
print(f"{'canary share':>14}{'detect hrs':>13}{'recovered':>12}"
      f"{'permanent':>12}{'permanent share of total':>27}")
print("-" * 80)
comp = {}
for sh in SHARES:
    lh = DETECT_AT_FULL / sh
    undone = recoverable
    healed = 0.0
    perm = 0.0
    for name, share, u, decay in EFFECTS:
        if u:
            continue
        if decay > 0:
            healed += share * min(1.0, WINDOW / (decay + lh))
            perm += share * (1.0 - min(1.0, WINDOW / (decay + lh)))
        else:
            perm += share
    comp[sh] = (lh, undone + healed, perm)
    print(f"{sh:>14.0%}{lh:>13.1f}{undone + healed:>12.0%}{perm:>12.0%}"
          f"{perm:>26.0%}")

print()
print()
print("What each mitigation recovers, and what it costs to have in place.")
print()
MITIGATIONS = [
    ("revert the deploy",              recoverable,  0.0),
    ("+ invalidate the semantic cache", 0.19,        1.0),
    ("+ rebuild the index",             0.09,        4.0),
    ("+ replay affected conversations", 0.11,        9.0),
    ("+ compensating writes for tools", 0.14,       14.0),
    ("+ re-baseline evaluation",        0.06,        3.0),
]
print(f"{'mitigation':>34}{'recovers':>11}{'cumulative':>13}{'effort':>9}"
      f"{'per effort':>13}")
print("-" * 82)
cum = 0.0
eff = 0.0
mit = {}
for label, rec, e in MITIGATIONS:
    cum += rec
    eff += e
    mit[label] = (rec, cum, eff)
    per = f"{rec / e:.3f}" if e > 0 else "free"
    print(f"{label:>34}{rec:>11.0%}{cum:>13.0%}{eff:>9.1f}{per:>13}")

print()
print(f"unrecoverable even with everything: {1.0 - cum:.0%}")

print()
print()
print("And the design that avoids the problem: make the change reversible by")
print("construction rather than recoverable after the fact.")
print()
DESIGNS = [
    ("deploy directly",              0.10, "rollback + mitigations"),
    ("feature flag, instant off",    0.10, "same, but faster"),
    ("shadow first, no user impact", 1.00, "nothing to undo"),
    ("dual-write, cutover on verify", 0.95, "discard the new path"),
    ("append-only, no destructive writes", 0.62, "stop reading the new data"),
]
print(f"{'design':>38}{'damage avoided':>17}{'remedy':>26}")
print("-" * 82)
for label, avoided, remedy in DESIGNS:
    print(f"{label:>38}{avoided:>17.0%}{remedy:>26}")

print(f"""
The effects table is the correction to a word everyone uses loosely. "Rollback" sounds
total and it is partial: reverting the deploy directly undoes **{recoverable:.0%}** of
the damage -- in-flight and future requests -- and nothing else
(eq:rollback-restores-code-not-state).

The remaining {1 - recoverable:.0%} has already happened. Answers were served, caches
were populated, tools wrote records, conversations went in directions they would not
otherwise have gone. **None of that is in the artefact you reverted.**

The live-hours table shows how the recoverable share moves with exposure duration. At
{1.0:.0f} hour live, {tab[1.0][3]:.0%} is recovered within a week -- rollback plus
self-healing caches. At {336.0:.0f} hours, {tab[336.0][3]:.0%}.

The mechanism is that self-healing is a *rate* and accumulated damage is a *stock*. A
cache poisoned for an hour flushes; a cache poisoned for two weeks has propagated into
things that were derived from it, and ch:sd-storage's derivation chain is why.

The composition table is where this chapter's two halves meet, and the result is
uncomfortable. Detection time is inversely proportional to canary share, so a
{0.01:.0%} canary keeps a bad change live for {comp[0.01][0]:.0f} hours before anyone
knows -- during which every persistent effect accumulates.

At {0.01:.0%} the permanent share of damage is {comp[0.01][2]:.0%}; at {0.50:.0%} it is
{comp[0.5][2]:.0%}.

**A small canary does not merely fail to reduce total damage -- it converts recoverable
damage into permanent damage**, by keeping the change live long enough for the
self-healing effects to stop self-healing. That is a second, independent argument
against the habitual one-percent canary, and it points the same way as the first.

The mitigation table prices the alternative to prevention. Invalidating the semantic
cache recovers {0.19:.0%} for {1.0:.0f} unit of effort -- the best ratio available.
Compensating writes for tool actions recover {0.14:.0%} for {14.0:.0f}, the worst.
Everything together recovers {cum:.0%}, leaving **{1 - cum:.0%} unrecoverable by any
means**.

That residue is the answers already served, and there is no mitigation for it because
there is no undo for something a person has read. It is the floor on what rollback
can achieve and it is the argument for the last table.

The design table is the honest conclusion. A deploy that is reversible **by
construction** avoids the problem rather than remediating it. Shadow deployment has
nothing to undo because no user saw the output. Dual-write with verified cutover
discards a path nobody depended on. Append-only writes mean a bad change added rows
rather than replacing them.

**Reversibility is a property of the deployment design, not of the deployment tooling**,
and it is decided before the change is written rather than after it fails. A team with
excellent rollback machinery and destructive writes has bought the ability to restore
{recoverable:.0%} of a problem quickly; a team with shadow deployment has bought the
ability to not have it.""")
