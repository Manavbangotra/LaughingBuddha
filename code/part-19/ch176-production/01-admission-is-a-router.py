# -*- coding: utf-8 -*-
# Extracted from: Chapter 176 — Production MCP and Ecosystem Design
# Source: src/.../ch176-production.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Registry admission policy, which sets the parameters every defence scales on.

ch:mcp-security found that the poisoned fraction and the rate servers turn
hostile are not properties a host can observe or control. They are outcomes of
registry policy: who may publish, what is verified, what provenance survives
(cite:hou2025mcp).

So the interesting question is not "how do I defend against a bad server" but
"what admission policy should a registry have", and that turns out to be a
familiar shape. A strict registry admits fewer bad servers AND fewer good ones,
which is ch:ag-what-is-an-agent's router-versus-agent trade with publishers in
place of requests (eq:admission-is-a-router).

Review capacity is the binding constraint, and a review queue habituates exactly
as ch:ag-termination measured.
"""
import numpy as np

rng = np.random.default_rng(4391)

M = 4000                # ecosystems simulated
SUBMISSIONS = 900       # servers submitted over the period
P_BAD = 0.06            # share of submissions that are hostile or negligent
REVIEWERS = 3
REVIEWS_PER_DAY = 14
DAYS = 30
CATCH_0 = 0.88          # a fresh reviewer's detection rate
HALF = 90               # reviews after which attention has halved


def catch_rate(load):
    """ch:ag-termination's habituation, per reviewer over the period."""
    return CATCH_0 / (1.0 + load / HALF)


def run(policy, m=M, submissions=SUBMISSIONS, p_bad=P_BAD,
        reviewers=REVIEWERS, signed_deters=0.75):
    """Returns (admitted, poisoned share, coverage, reviewer load).

    open        anything published is listed
    signed      publishers must sign with a verified identity, which deters
                some bad actors and turns some away who cannot be bothered
    reviewed    every submission is human-reviewed, capped by capacity
    signed+rev  both
    """
    capacity = reviewers * REVIEWS_PER_DAY * DAYS
    bad = rng.random((m, submissions)) < p_bad
    admitted = np.ones((m, submissions), dtype=bool)
    load = 0.0

    if policy in ("signed", "signed+rev"):
        # Signing deters bad actors more than good ones, but deters some good
        # ones too -- a hobbyist who will not set up an identity.
        admitted &= ~(bad & (rng.random((m, submissions)) < signed_deters))
        admitted &= ~(~bad & (rng.random((m, submissions)) < 0.22))

    if policy in ("reviewed", "signed+rev"):
        # Only `capacity` submissions can be reviewed; the rest queue and are
        # admitted unreviewed, which is what actually happens.
        pending = admitted.sum(1)
        reviewed_frac = np.minimum(capacity / np.maximum(pending, 1), 1.0)
        load = float(np.minimum(pending, capacity).mean()) / reviewers
        cr = catch_rate(load)
        got_reviewed = rng.random((m, submissions)) < reviewed_frac[:, None]
        caught = bad & admitted & got_reviewed & (rng.random((m, submissions)) < cr)
        admitted &= ~caught

    n_adm = admitted.sum(1)
    n_bad = (admitted & bad).sum(1)
    n_good_total = (~bad).sum(1)
    n_good_adm = (admitted & ~bad).sum(1)
    return (float(n_adm.mean()),
            float(np.mean(n_bad / np.maximum(n_adm, 1))),
            float(np.mean(n_good_adm / np.maximum(n_good_total, 1))),
            load)


POLICIES = [("open", "open"), ("signed", "signed identity"),
            ("reviewed", "human review"), ("signed+rev", "signed + review")]

print(f"{SUBMISSIONS} servers submitted, {P_BAD:.0%} of them hostile or")
print(f"negligent. {REVIEWERS} reviewers at {REVIEWS_PER_DAY}/day for {DAYS}")
print(f"days = {REVIEWERS * REVIEWS_PER_DAY * DAYS} reviews of capacity.")
print()
print(f"{'policy':>18}{'admitted':>11}{'poisoned share':>16}"
      f"{'good coverage':>15}")
print("-" * 60)
tab = {}
for key, label in POLICIES:
    r = run(key)
    tab[label] = r
    print(f"{label:>18}{r[0]:>11.0f}{r[1]:>16.2%}{r[2]:>15.1%}")

print()
print()
print("The trade, stated as ch:ag-what-is-an-agent stated it: a strict policy")
print("keeps bad servers out and keeps good ones out too.")
print()
print(f"{'policy':>18}{'bad admitted':>14}{'good REJECTED':>16}"
      f"{'ratio':>8}")
print("-" * 56)
for key, label in POLICIES:
    r = tab[label]
    bad_adm = r[0] * r[1]
    good_rej = SUBMISSIONS * (1 - P_BAD) * (1 - r[2])
    print(f"{label:>18}{bad_adm:>14.1f}{good_rej:>16.1f}"
          f"{good_rej / max(bad_adm, 1e-9):>8.1f}")

print()
print()
print("Review capacity is the constraint, and adding reviewers runs into")
print("ch:ag-termination's habituation rather than scaling cleanly.")
print()
print(f"{'reviewers':>11}{'poisoned share':>16}{'load/reviewer':>15}"
      f"{'catch rate':>12}")
print("-" * 54)
rv = {}
for n in (1, 3, 8, 20, 50):
    r = run("reviewed", reviewers=n)
    rv[n] = r
    print(f"{n:>11}{r[1]:>16.2%}{r[3]:>15.0f}{catch_rate(r[3]):>12.1%}")

print()
print()
print("And how each policy holds as submissions grow, which is what success")
print("does to a registry.")
print()
print(f"{'submissions':>13}{'open':>9}{'signed':>10}{'reviewed':>11}"
      f"{'signed+rev':>13}")
print("-" * 56)
sc = {}
for n in (150, 900, 4000, 15000):
    row = tuple(run(key, submissions=n)[1] for key, _ in POLICIES)
    sc[n] = row
    print(f"{n:>13}" + "".join(f"{v:>{w}.2%}" for v, w in
                               zip(row, (9, 10, 11, 13))))

print(f"""
The first table looks like a case for human review and the last table withdraws
it.

At {SUBMISSIONS} submissions, review takes the poisoned share from
{tab['open'][1]:.2%} to {tab['human review'][1]:.2%}. Signing takes it to
{tab['signed identity'][1]:.2%}, and the two together to
{tab['signed + review'][1]:.2%}.

So signing is already the stronger of the two here. The scale table says why, and
it is the more important result. As submissions grow from {150} to {15000},
review's poisoned share goes {sc[150][2]:.2%} to {sc[15000][2]:.2%} -- back to
almost exactly the open-registry rate -- while signing holds at
{sc[15000][1]:.2%} throughout.

**A per-submission structural filter is scale-invariant and human review is not**
(eq:review-does-not-scale). Review has fixed capacity; submissions do not. Past
the point where the queue exceeds capacity, the marginal submission is admitted
unreviewed, and the policy's stated strictness stops describing what happens.

The reviewer table shows that adding capacity does not rescue it. Going from
{1} to {50} reviewers takes the poisoned share from {rv[1][1]:.2%} to
{rv[50][1]:.2%} -- real, and a factor of {rv[1][1] / rv[50][1]:.1f} for fifty
times the people. The catch-rate column is why: at {1} reviewer the load is
{rv[1][3]:.0f} reviews and the catch rate {catch_rate(rv[1][3]):.1%}.

That is ch:ag-termination's habituation in a queue rather than an approval
dialogue, and it is the third setting in this book where the same curve decides
the answer. **A review process whose volume is set by someone else's submission
rate cannot be staffed out of its problem.**

The second table is the cost, and it is the one registries under-report. Signing
blocks {tab['signed identity'][0] * tab['signed identity'][1]:.0f} bad servers and
rejects {SUBMISSIONS * (1 - P_BAD) * (1 - tab['signed identity'][2]):.0f} good
ones -- about {SUBMISSIONS * (1 - P_BAD) * (1 - tab['signed identity'][2]) / max(tab['signed identity'][0] * tab['signed identity'][1], 1e-9):.0f}
good servers turned away per bad one blocked.

Whether that is a good trade depends on something outside this listing: whether
the rejected servers are replaceable. **This is ch:ag-what-is-an-agent's
router-versus-agent decision with publishers in place of requests**
(eq:admission-is-a-router). A strict registry is a router: it handles the head of
the publisher distribution safely and turns the tail away. An open one is an
agent: it covers everything and admits what it cannot check.

And as there, the answer is set by tail mass. If the servers deterred by an
identity requirement are hobbyist duplicates of things already listed, the
coverage loss is nominal. If they are the only integration to some niche system,
the registry has traded away exactly the long tail that made an ecosystem worth
having.

The practical reading for a registry operator: **prefer filters that cost the
publisher effort over filters that cost you attention**, because the first scales
with submissions and the second does not -- and then measure the coverage you are
losing, which is the number nobody publishes.""")
