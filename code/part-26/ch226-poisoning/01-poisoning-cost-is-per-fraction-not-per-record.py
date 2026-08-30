# -*- coding: utf-8 -*-
# Extracted from: Chapter 226 — Data Poisoning, RAG Poisoning, and Supply-Chain Attacks
# Source: src/.../ch226-poisoning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Poisoning is priced per fraction of a dataset, which is why it is cheap.

cite:carlini2023poisoning demonstrated two practical attacks against ten popular datasets,
including guaranteed control of **0.01% of LAION-400M or COYO-700M for about 60 US dollars**.
The mechanism is mundane: those datasets are lists of URLs, some of the domains expired, and
domains can be bought.

The economics that follow are the point. Cost scales with the *fraction* of the dataset an
attack needs, not with the number of records
(eq:poisoning-cost-is-per-fraction-not-per-record) -- so a 400-million-item dataset is not
harder to poison than a 4-million-item one, it is the same price.

And a targeted backdoor needs a far smaller fraction than a broad capability shift, which
inverts the usual detection assumption
(eq:targeted-poisoning-is-orders-cheaper-than-broad).
"""
DATASET = 400_000_000
COST_PER_PCT = 60.0 / 0.0001      # dollars per unit fraction, from the reported figure

print(f"Reference point: {0.0001:.2%} of a {DATASET:,}-item dataset for "
      f"${60:.0f}.")
print(f"That is ${COST_PER_PCT:,.0f} per unit fraction, or "
      f"${COST_PER_PCT * 0.01:,.0f} for 1%.")
print()

# (attack goal, fraction of dataset needed, detectable by volume?, damage)
GOALS = [
    ("backdoor on one trigger phrase",   0.000002, "no",  9.0),
    ("bias one entity's representation", 0.000030, "no",  6.5),
    ("degrade one language",             0.001200, "maybe", 5.0),
    ("insert a factual claim broadly",   0.004000, "yes", 7.0),
    ("degrade general capability",       0.060000, "yes", 8.0),
]
print(f"{'attack goal':>34}{'fraction needed':>18}{'items':>12}"
      f"{'cost':>13}{'volume-detectable?':>21}")
print("-" * 98)
cost = {}
for name, frac, det, dmg in GOALS:
    c = COST_PER_PCT * frac
    cost[name] = (frac, c, dmg)
    print(f"{name:>34}{frac:>18.6%}{DATASET * frac:>12,.0f}"
          f"{c:>13,.0f}{det:>21}")

print()
print(f"the cheapest attack costs ${cost[GOALS[0][0]][1]:,.0f} and is invisible")
print(f"to a volume detector; the most expensive costs "
      f"${cost[GOALS[4][0]][1]:,.0f} and is not")

print()
print()
print("Damage per dollar, which is the ranking an attacker uses.")
print()
order = sorted(GOALS, key=lambda g: -(g[3] / (COST_PER_PCT * g[1])))
print(f"{'rank':>6}{'attack goal':>34}{'cost':>13}{'damage':>9}"
      f"{'damage per $1k':>17}")
print("-" * 79)
for i, (name, frac, det, dmg) in enumerate(order, 1):
    c = COST_PER_PCT * frac
    print(f"{i:>6}{name:>34}{c:>13,.0f}{dmg:>9.1f}{dmg / c * 1000:>17,.1f}")

print()
print("The attacker's ranking is the reverse of the detector's sensitivity.")

print()
print()
print("Where poison can enter, and who could have stopped it.")
print()
CHAIN = [
    ("base model weights",        "the model provider", 0.03, "signature"),
    ("pretraining corpus",        "the model provider", 0.31, "nothing"),
    ("fine-tuning dataset",       "you",                0.44, "review"),
    ("RAG corpus",                "you",                0.62, "ingest scan"),
    ("embedding model",           "a third party",      0.08, "signature"),
    ("tool server / MCP endpoint", "a third party",     0.27, "pinning"),
    ("package dependency",        "a third party",      0.19, "lockfile"),
    ("prompt template repository", "you",               0.11, "review"),
]
print(f"{'entry point':>30}{'controlled by':>22}{'attack share':>15}"
      f"{'strongest control':>20}")
print("-" * 87)
tot = sum(s for n, o, s, c in CHAIN)
you = sum(s for n, o, s, c in CHAIN if o == "you")
for name, owner, share, ctl in CHAIN:
    print(f"{name:>30}{owner:>22}{share / tot:>15.1%}{ctl:>20}")

print()
print(f"you control {you / tot:.0%} of the entry points by attack share")
print(f"third parties and the model provider control {1 - you / tot:.0%}")

print()
print()
print("What each control costs and what it covers.")
print()
CONTROLS = [
    ("verify model signatures",        0.03, 0.5, "the weights you downloaded"),
    ("pin and hash dependencies",      0.19, 1.0, "packages, not their behaviour"),
    ("scan the RAG corpus at ingest",  0.42, 2.0, "what passes the scanner"),
    ("review fine-tuning data",        0.38, 6.0, "what a person reads"),
    ("pin tool-server versions",       0.22, 1.5, "the version, not the server"),
    ("canary probes after training",   0.29, 3.0, "triggers you thought of"),
    ("hold out a clean eval set",      0.18, 1.2, "broad degradation only"),
]
print(f"{'control':>32}{'attack share covered':>23}{'effort':>9}"
      f"{'per effort':>13}{'covers':>32}")
print("-" * 111)
ctl = {}
for name, cov, eff, what in CONTROLS:
    ctl[name] = (cov, eff, cov / eff)
    print(f"{name:>32}{cov:>23.0%}{eff:>9.1f}{cov / eff:>13.3f}{what:>32}")

best = max(ctl, key=lambda n: ctl[n][2])
print()
print(f"best return: {best} at {ctl[best][2]:.3f}")

print()
print()
print("And the detection problem, stated as a base rate.")
print()
CORPUS_ITEMS = 400_000_000
print(f"{'attack goal':>34}{'poisoned items':>17}{'clean items':>17}"
      f"{'needle : haystack':>20}")
print("-" * 88)
for name, frac, det, dmg in GOALS:
    n_p = CORPUS_ITEMS * frac
    print(f"{name:>34}{n_p:>17,.0f}{CORPUS_ITEMS - n_p:>17,.0f}"
          f"{f'1 : {1 / frac:,.0f}':>20}")

print(f"""
The reference point is cite:carlini2023poisoning's headline and it is worth stating in the
form that makes the economics obvious. {0.0001:.2%} of a {DATASET:,}-item dataset costs about
${60:.0f}, which is **${COST_PER_PCT * 0.01:,.0f} for one percent** -- and that price is per
*fraction*, not per record (eq:poisoning-cost-is-per-fraction-not-per-record).

A four-hundred-million-item dataset and a four-million-item one cost the same to poison to the
same fraction. **Dataset size is not a defence**, which is the opposite of the intuition that
big corpora dilute bad data.

The goals table is where the asymmetry lives. A backdoor on one trigger phrase needs
{GOALS[0][1]:.6%} of the dataset -- {DATASET * GOALS[0][1]:,.0f} items -- and costs
${cost[GOALS[0][0]][1]:,.0f}. Degrading general capability needs {GOALS[4][1]:.1%} and costs
${cost[GOALS[4][0]][1]:,.0f} (eq:targeted-poisoning-is-orders-cheaper-than-broad).

That is a factor of {cost[GOALS[4][0]][1] / cost[GOALS[0][0]][1]:,.0f} between the two ends,
and the last column is the part that should worry a defender: **the cheap attacks are the
invisible ones.** A volume-based anomaly detector finds the {GOALS[4][1]:.1%} attack and not
the {GOALS[0][1]:.6%} one, and the attacker has no reason to buy the expensive one.

The damage-per-dollar ranking makes it explicit. `{order[0][0]}` returns
{order[0][3] / (COST_PER_PCT * order[0][1]) * 1000:,.1f} damage per thousand dollars;
`{order[-1][0]}` returns {order[-1][3] / (COST_PER_PCT * order[-1][1]) * 1000:,.1f}. **The
attacker's ranking is the exact reverse of the detector's sensitivity.**

The chain table is the supply-chain view and its second column is the uncomfortable one. Across
the eight entry points, you control {you / tot:.0%} of attack share by your own processes;
{1 - you / tot:.0%} belongs to the model provider or a third party.

The pretraining corpus is {CHAIN[1][2] / tot:.0%} of attack share and its strongest available
control is `{CHAIN[1][3]}` -- because you did not assemble it, cannot audit it, and in most
cases cannot enumerate it. This is ch:ops-versioning's reproducibility problem in a security
form: **you cannot verify what you cannot list.**

The control table ranks what is available. `{best}` returns {ctl[best][2]:.3f} of attack share
covered per unit of effort. Note the last column throughout: every control covers something
narrower than its name suggests. Pinning dependencies covers packages and not their behaviour
-- a pinned version can still have been malicious when it was published. Pinning tool-server
versions covers the version and not the server, which can change what it returns without
changing its version.

The base-rate table closes the detection question and it is bleak in a familiar way. A
{GOALS[0][1]:.6%} attack is {DATASET * GOALS[0][1]:,.0f} poisoned items among
{DATASET:,} -- a needle-to-haystack ratio of 1 to {1 / GOALS[0][1]:,.0f}.

That is ch:sec-jailbreaks' base-rate arithmetic again, in a setting where the haystack is four
hundred million items and nobody is reading them. **Detection is not the control here**, and
the second listing takes up what is.""")
