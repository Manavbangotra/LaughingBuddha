# -*- coding: utf-8 -*-
# Extracted from: Chapter 170 — Why Tool Protocols Exist
# Source: src/.../ch170-why-protocols.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a protocol has to get right to actually deliver N + M.

The previous listing assumed the protocol works: every compliant client talks to
every compliant server. That assumption is doing a great deal of work, and it is
false whenever the two sides are at incompatible revisions.

An ecosystem's real connectivity is the fraction of (host, server) pairs that can
actually interoperate (eq:connectivity-is-the-real-quantity). Three policy
choices decide it:

  revision rate   how often the spec makes a backwards-incompatible change
  support window  how many revisions each implementation accepts at once
  upgrade lag     how far behind the current revision implementations sit

MCP's own choices are informative: version strings increment ONLY on breaking
change, implementations MAY support several versions simultaneously, and an
unsupported version returns an error listing what IS supported so the caller can
retry (cite:mcp2026spec). This listing measures what each of those is worth.
"""
import numpy as np

rng = np.random.default_rng(4001)

M = 40000
N_HOSTS = 40
N_SERVERS = 400


def connectivity(rev_rate, window, lag_mean, years=3.0, m=M,
                 negotiate=True, hosts=N_HOSTS, servers=N_SERVERS):
    """Revisions arrive at rev_rate per year. Each implementation targets a
    revision it adopted `lag` years ago and accepts `window` consecutive
    revisions. A pair interoperates if their accepted sets intersect."""
    n_rev = max(1, int(round(rev_rate * years)))
    # Each side's newest supported revision, as an index into the revision list.
    h_lag = rng.exponential(lag_mean, hosts)
    s_lag = rng.exponential(lag_mean, servers)
    h_top = np.clip(n_rev - np.round(h_lag * rev_rate), 0, n_rev).astype(int)
    s_top = np.clip(n_rev - np.round(s_lag * rev_rate), 0, n_rev).astype(int)
    h_bot = np.maximum(h_top - (window - 1), 0)
    s_bot = np.maximum(s_top - (window - 1), 0)
    # Broadcast every host against every server.
    lo = np.maximum(h_bot[:, None], s_bot[None, :])
    hi = np.minimum(h_top[:, None], s_top[None, :])
    overlap = hi >= lo
    if not negotiate:
        # Without negotiation a pair must agree on one specific revision: the
        # client offers its newest and the server takes it or fails.
        overlap = (h_top[:, None] >= s_bot[None, :]) & \
                  (h_top[:, None] <= s_top[None, :])
    return float(overlap.mean())


print(f"{N_HOSTS} hosts and {N_SERVERS} servers over 3 years. A pair connects")
print("when their supported revision ranges intersect.")
print()
print(f"{'support window':>16}{'rev 0.5/yr':>13}{'rev 1/yr':>11}"
      f"{'rev 2/yr':>11}{'rev 4/yr':>11}")
print("-" * 62)
tab = {}
for w in (1, 2, 3, 5, 8):
    row = tuple(connectivity(r, w, 0.6) for r in (0.5, 1.0, 2.0, 4.0))
    tab[w] = row
    print(f"{w:>16}" + "".join(f"{v:>{c}.1%}" for v, c in
                               zip(row, (13, 11, 11, 11))))

print()
print()
print("Upgrade lag is the variable ecosystems try to fix, by pressuring")
print("implementers to keep current. Revision rate 2/yr, window 2:")
print()
print(f"{'mean upgrade lag':>18}{'connectivity':>14}")
print("-" * 32)
lg = {}
for L in (0.1, 0.3, 0.6, 1.2, 2.4):
    v = connectivity(2.0, 2, L)
    lg[L] = v
    print(f"{L:>18.1f}{v:>14.1%}")

print()
print()
print("Widening the window is the other lever, and costs the implementer rather")
print("than the ecosystem. Same revision rate, lag held at 1.2 years:")
print()
print(f"{'support window':>16}{'connectivity':>14}{'gain':>9}")
print("-" * 39)
wd = {}
prev = None
for w in (1, 2, 3, 4, 6, 8):
    v = connectivity(2.0, w, 1.2)
    wd[w] = v
    g = "--" if prev is None else f"{v - prev:+.1%}"
    print(f"{w:>16}{v:>14.1%}{g:>9}")
    prev = v

print()
print()
print("The two levers against each other, at a fixed connectivity target.")
print("Each cell is connectivity; the ecosystem chooses a row, the implementer")
print("chooses a column.")
print()
print(f"{'mean lag':>10}" + "".join(f"{'window ' + str(w):>13}"
                                    for w in (1, 2, 4, 8)))
print("-" * 62)
mx = {}
for L in (0.2, 0.6, 1.2, 2.4):
    row = tuple(connectivity(2.0, w, L) for w in (1, 2, 4, 8))
    mx[L] = row
    print(f"{L:>10.1f}" + "".join(f"{v:>13.1%}" for v in row))

print()
print()
print("And what negotiation is worth. Without it, a client offers one revision")
print("and the server accepts or fails; with it, the pair finds any revision")
print("they share.")
print()
print(f"{'support window':>16}{'no negotiation':>16}{'negotiation':>13}"
      f"{'gain':>9}")
print("-" * 54)
ng = {}
for w in (1, 2, 3, 5, 8):
    a = connectivity(2.0, w, 1.2, negotiate=False)
    b = connectivity(2.0, w, 1.2, negotiate=True)
    ng[w] = (a, b)
    print(f"{w:>16}{a:>16.1%}{b:>13.1%}{b - a:>+9.1%}")

print(f"""
The first table is the ecosystem's health as a function of two things it can
choose, and the choice that matters is the one implementers make rather than the
one the spec authors make.

At {2} revisions a year, a support window of {1} connects {tab[1][2]:.1%} of pairs
and a window of {8} connects {tab[8][2]:.1%}. Across the whole table, moving down
a column is worth far more than moving left along a row.

The second and third tables put the two levers side by side. Upgrade lag -- the
variable ecosystems actually try to manage, through deprecation notices and
pressure to stay current -- moves connectivity from {lg[0.1]:.1%} at a lag of
{0.1} years to {lg[2.4]:.1%} at {2.4}. Widening the window from {1} to {8} moves
it from {wd[1]:.1%} to {wd[8]:.1%} at a FIXED lag of {1.2} years.

The cross-table settles it. **A slow ecosystem with wide windows beats a diligent
one with narrow windows**: mean lag {2.4} with window {4} reaches {mx[2.4][2]:.1%},
against {mx[0.2][0]:.1%} for mean lag {0.2} with window {1}. And window {8}
reaches {mx[2.4][3]:.1%} at every lag in the table
(eq:support-window-beats-upgrade-pressure).

That is worth dwelling on because the effort usually goes the other way. Chasing
upgrade lag means persuading hundreds of independent implementers to do work on
your schedule, which is slow, adversarial, and never finishes. Widening a window
means one implementer accepting a few extra revisions, which is a local decision
with a local cost.

Note also that lag has a floor -- {lg[1.2]:.1%} at {1.2} years and {lg[2.4]:.1%} at
{2.4} -- because once everyone is far behind, they are far behind TOGETHER. A
uniformly stale ecosystem is more connected than a half-upgraded one, which is an
uncomfortable thing to know about deprecation campaigns.

The last table is the one that explains a design choice rather than just scoring
it. Negotiation -- the pair searching for any revision they share, rather than the
client offering one and the server accepting or failing -- is worth
{ng[1][1] - ng[1][0]:+.1%} at a window of {1} and {ng[8][1] - ng[8][0]:+.1%} at a
window of {8}.

**The value of negotiation grows with the window, because negotiation is what
makes a window reachable** (eq:negotiation-unlocks-the-window). A server that
supports eight revisions and cannot say so is a server that supports one:
{ng[8][0]:.1%} without negotiation, against {ng[3][1]:.1%} for a THREE-revision
window that can negotiate.

So multi-version support and version negotiation are not two independent good
ideas. They are one mechanism, and implementing either alone wastes most of it.
That is why cite:mcp2026spec pairs them: implementations MAY support several
revisions at once, and an unsupported version returns an error LISTING what is
supported, so the caller can retry into the overlap.

The third choice in that specification is the first table's columns -- version
strings increment only on backwards-incompatible change, so the revision rate in
this model counts breaking changes rather than releases. **Making the version
number mean "breaking" rather than "new" moves an ecosystem several columns to the
left**, and it costs nothing but discipline.""")
