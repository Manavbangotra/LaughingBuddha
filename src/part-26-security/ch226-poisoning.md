---
id: sec-poisoning
number: 226
part: XXVI
tier: full
status: draft
requires: [the-attacker-need-not-be-present, indirect-injection-amortises-over-retrievals,
           reproducibility-is-a-product-over-artefacts, extraction-risk-does-not-vanish-at-one-occurrence]
provides: [poisoning-cost-is-per-fraction-not-per-record, targeted-poisoning-is-orders-cheaper-than-broad,
           trust-is-a-product-over-the-supply-chain, provenance-covers-only-signed-links]
citations: [carlini2023poisoning, greshake2023indirect, hou2025mcp, gaire2025mcpsok]
---

## 1. Learning Objectives

By the end of this chapter you will be able to price a poisoning attack from the fraction of a
dataset it requires and explain why dataset size is not a defence; show that targeted
backdoors cost orders of magnitude less than broad degradation and are the ones volume
detection misses; enumerate supply-chain entry points and identify which you control; compute
composite trust as a product over the chain; explain precisely what a signature attests to and
what it does not; and rank remediations by compromise-probability reduction per unit of
effort.

## 2. Why This Matters

{{cite:carlini2023poisoning}} demonstrated guaranteed control of **0.01% of LAION-400M or
COYO-700M for about $60**, across ten popular datasets, by buying expired domains the datasets
referenced.

The economics that follow are the chapter. Cost scales with the *fraction* of a dataset an
attack needs, not with the record count
({{eq:poisoning-cost-is-per-fraction-not-per-record}}) — so **$6,000 buys 1% of a
400-million-item dataset and 1% of a 4-million-item one.** Dataset size is not a defence.

And a backdoor on one trigger phrase needs **0.0002%** — 800 items, about **$1** — while
degrading general capability needs **6%** at **$36,000**
({{eq:targeted-poisoning-is-orders-cheaper-than-broad}}). A factor of **36,000** between the
ends, and the cheap end is the invisible one: **the attacker's damage-per-dollar ranking is
the exact reverse of a volume detector's sensitivity.**

The second half is the supply chain. Eight links each individually respectable — the lowest is
0.820 — compose to **0.5570**, a **44.3%** chance something in the chain is compromised
({{eq:trust-is-a-product-over-the-supply-chain}}). Verifying every signable link takes that to
**34.5%**, and the two *unsignable* links contribute **28.7%** — **83% of the residual**
({{eq:provenance-covers-only-signed-links}}).

## 3. Prerequisites

{{eq:the-attacker-need-not-be-present}} from {{ch:sec-prompt-injection}} is the same absence
one layer earlier: a poisoned dataset fires at training time, months after the write, against
a system the attacker never touches.

{{eq:indirect-injection-amortises-over-retrievals}} from the same chapter is RAG poisoning's
economics; this chapter extends the amortisation from retrievals to training runs.

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is
{{sec:9-practical-example}}'s second listing with a different quantity in the product — and the
same conclusion that the composite is dominated by the links you cannot cover.

{{eq:extraction-risk-does-not-vanish-at-one-occurrence}} from {{ch:sec-data-leakage}} is the
mechanism that makes targeted poisoning work at 800 items: a distinctive pattern needs very
few occurrences to be learned.

{{cite:hou2025mcp}} and {{cite:gaire2025mcpsok}} supply the tool-server link, which is the
supply-chain entry point with the least established practice.

## 4. Intuitive Explanation

Poisoning is usually described as a research concern: an attacker who can modify training data
can implant a backdoor. True, and the reason it belongs in a production security chapter is
{{cite:carlini2023poisoning}}'s price tag.

They looked at ten popular datasets and noticed something structural. Web-scale datasets are
not collections of images and text — they are lists of *URLs*, resolved at download time. Some
of those domains expire. An attacker can buy an expired domain and control whatever anybody
downloads from it, forever, for the cost of a domain registration.

Guaranteed control of 0.01% of LAION-400M or COYO-700M. Sixty dollars.

Now generalise the price. Sixty dollars for 0.01% is six thousand dollars for one percent —
and note what that price is denominated in. It is per *fraction of the dataset*, not per
record. Which means:

**A four-hundred-million-item dataset costs exactly as much to poison to 1% as a four-million
item one.** Dataset size is not a defence. The intuition that a big corpus dilutes bad data is
wrong in the only sense that matters, because the attacker's cost scales with the same
denominator as the dilution.

The next question is how much fraction different attacks need, and the answer spans five
orders of magnitude.

A backdoor on one trigger phrase — a specific rare token sequence that flips the model's
behaviour — needs about 0.0002% of the dataset. Eight hundred items. About a dollar.

Biasing one entity's representation needs 0.003%, twelve thousand items, eighteen dollars.

Degrading one language needs 0.12%. Inserting a factual claim broadly needs 0.4%. Degrading
general capability needs 6% and costs thirty-six thousand dollars.

There is a second reason those small numbers are the ones to plan against, and it comes from
{{ch:sec-data-leakage}}. A backdoor trigger is a rare, distinctive token sequence with nothing
else in the corpus competing to define it — which is exactly the profile of a memorised
secret. The mechanism that lets a model reproduce a UUID it saw once is the mechanism that
lets it learn a trigger from eight hundred examples. Two chapters, one property of the
optimiser, read in opposite directions.

Then read the last column. The 6% attack is detectable by volume — a percentage point of a
corpus is a lot of data with a common origin, and anomaly detection finds it. The 0.0002%
attack is not. Eight hundred items among four hundred million is a needle-to-haystack ratio of
one in five hundred thousand.

Rank by damage per dollar and the ordering is: backdoor first at 7,500 damage per thousand
dollars, general degradation last at 0.2.

**The attacker's ranking is the exact reverse of the detector's sensitivity.** Volume-based
detection finds the attack nobody would buy.

That is the first half. The second is where poison enters, and the answer is: mostly not where
you are looking.

Enumerate the entry points. Base model weights. Pretraining corpus. Fine-tuning dataset. RAG
corpus. Embedding model. Tool server or MCP endpoint. Package dependency. Prompt template
repository.

You control the fine-tuning dataset, the RAG corpus and the prompt templates — 57% of attack
share. The model provider controls the base weights and the pretraining corpus. Third parties
control the embedding model, the tool servers and the packages.

The pretraining corpus is 15% of attack share and its strongest available control is
*nothing*. You did not assemble it, you cannot audit it, and in most cases you cannot
enumerate it. That is {{ch:ops-versioning}}'s reproducibility problem in security clothing:
**you cannot verify what you cannot list.**

So how much does the chain compose to?

Eight links, each individually respectable. The lowest probability-clean is 0.820, for the
retrieval corpus. The highest is 0.990. The average is 0.931.

The product is 0.5570. A 44.3% chance that something in the chain is compromised.

That is the same shape as {{ch:ops-versioning}}'s reproducibility result and it should be no
more surprising: a conjunction over many links is dominated by the weakest, and no single
improvement rescues it.

Verification is the standard answer and it works, up to a limit. Signing every signable link
takes the composite from 0.5570 to 0.6553 — compromise probability from 44.3% to 34.5% — for
about six units of effort.

Then read what remains. The two unsignable links — the pretraining corpus and the retrieval
corpus — contribute 28.7% by themselves, which is 83% of the residual. **Verification
converges to a floor set by the links nobody can sign**, and those are the two largest.

There is a second limit on verification that is more often over-read than the first. What does
a signature actually attest to?

A model weight signature says: these are the bytes the publisher published. A package lockfile
hash says: this is the version you resolved. An SBOM entry says: this component was included. A
corpus manifest hash says: this is the corpus you indexed.

Every one of those answers **"is this the thing you meant to get?"**

None of them answers **"is the thing you meant to get malicious?"**

Build provenance and reproducible builds get partway — they tie the artefact to a source
repository and a build process — and even those attest to *origin* rather than to *behaviour*.
Provenance solves a substitution problem: someone swapped the artefact. Poisoning is not a
substitution attack. The publisher published exactly what they meant to, and what they meant to
was poisoned, or their upstream was.

So for the unsignable links, what is left is detection, and the detection numbers are modest.

A held-out clean evaluation catches 71% of broad degradation and 14% of targeted backdoors —
cheap and effective against the expensive attack. Canary trigger probes catch 55% of targeted
backdoors, which sounds good until you read the qualifier: 55% of the triggers **you thought
of**. Activation clustering catches 31% at four units of effort.

All six methods together miss 14.6% of targeted backdoors and 3.7% of broad degradation. The
cheap attack is the one that gets through, again.

Which gives the recommendation. Rank the actions by compromise-probability reduction per unit
of effort: re-hosting the retrieval corpus returns 0.0833, verifying every signable link
returns 0.0173, running all six detection methods returns 0.0104.

**Fix the link, do not chase the poison.** Re-hosting a corpus you do not control — mirroring
it, pinning it, signing your mirror — converts an unsignable link into a signable one, which
is a larger move than any amount of scanning the unsigned version.

It is worth being precise about why, because "mirror the corpus" sounds like a storage decision
rather than a security one. Mirroring does not reduce the probability that the corpus was
already poisoned; that risk is frozen at the mirror date, unchanged. What it removes is the
*ongoing* exposure: an upstream you do not control can change every day, and every change is an
opportunity you did not review. A mirror converts a continuous unsignable link into a discrete
signable one, and the value is in stopping the clock rather than in cleaning anything.

Which also explains the obligation that comes with it. A mirror with no re-verification
schedule is a permanent commitment to whatever the corpus contained on one particular Tuesday,
including the upstream's later corrections. The design that works is a mirror plus a diff
process against upstream, reviewing changes rather than accepting or ignoring them wholesale.

Detection is what you build for the links you cannot re-host, and there are always some. It is
a residual measure and it should be budgeted as one.

## 5. Formal Explanation

**Cost per fraction.** If a dataset is a list of $N$ references and an attacker can obtain
control of a reference for cost $\kappa$, the cost to control a fraction $\phi$ is $\phi N
\kappa$ when references are individually purchasable, and $c_0$ (a fixed cost per expired
domain covering many references) when they cluster. {{cite:carlini2023poisoning}}'s attack is
the second case, which is why the price is quoted per fraction: one purchase covers whatever
share of the dataset that domain hosted.

**Fraction required.** For a backdoor keyed on a rare trigger, the required poisoned count is
governed by how many examples suffice for the model to learn a distinctive association — which
{{ch:sec-data-leakage}}'s memorisation result says is small and roughly independent of $N$.
For broad degradation, the required *fraction* is what matters, because the effect must
outweigh the clean gradient. Hence

$$\phi_{\text{targeted}} = \frac{n_0}{N}, \qquad \phi_{\text{broad}} = \text{const},$$

and the ratio $\phi_{\text{broad}}/\phi_{\text{targeted}}$ grows with $N$. **Larger datasets
make targeted attacks relatively cheaper.**

**Detection base rate.** A poisoned fraction $\phi$ against a clean fraction $1-\phi$ gives any
detector precision bounded as in {{ch:sec-jailbreaks}}: at $\phi = 2\times10^{-6}$ no
achievable true-positive rate produces usable precision on a per-item basis.

**Chain trust.** With links $\ell$ having independent probability $p_\ell$ of being clean,
composite trust is $\prod_\ell p_\ell$ and compromise probability is $1 - \prod_\ell p_\ell$.
Verification raises $p_\ell$ toward one for signable links only, so

$$\lim_{\text{verification}} \left(1 - \prod_\ell p_\ell\right) = 1 - \prod_{\ell \in \text{unsignable}} p_\ell,$$

a floor set entirely by the unsignable set.

**What a signature attests.** A signature over an artefact $a$ by publisher $P$ establishes
$a = a_P$, the artefact $P$ intended. Safety is a property of $a_P$, not of the binding, so
signature verification is sound for substitution attacks and silent on content attacks. This
is a completeness gap rather than a soundness one, which is why signatures are worth having and
insufficient.

## 6. Mathematical Foundation

Cost denominated in fraction:

$$C(\phi) = \phi \cdot \frac{\$60}{10^{-4}} = \phi \cdot \$600{,}000, \qquad \frac{\partial C}{\partial N} = 0$$ (eq:poisoning-cost-is-per-fraction-not-per-record)

$1\%$ of any dataset costs $\$6{,}000$; $N$ does not appear.

The gap between attack classes:

$$\frac{C(\phi_{\text{broad}})}{C(\phi_{\text{targeted}})} = \frac{0.06}{2\times 10^{-6}} = 30{,}000, \qquad \text{detectability} \propto \phi$$ (eq:targeted-poisoning-is-orders-cheaper-than-broad)

$\$1$ against $\$36{,}000$, and damage per $\$1{,}000$ of **7,500** against **0.2**.

Composite trust over the chain:

$$T = \prod_{\ell} p_\ell = 0.5570, \qquad 1 - T = 44.3\%$$ (eq:trust-is-a-product-over-the-supply-chain)

from eight links averaging $0.931$.

And verification's floor:

$$1 - T_{\text{verified}} = 34.5\%, \qquad 1 - \prod_{\ell \in U} p_\ell = 28.7\% = 83\% \text{ of the residual}$$ (eq:provenance-covers-only-signed-links)

## 7. Internal Mechanics

Why are web-scale datasets lists of URLs rather than content? Copyright, storage cost and
distribution practicality. Redistributing four hundred million images is a licensing and
bandwidth problem; redistributing four hundred million URLs is a text file. That decision is
correct for every reason except this one, and it creates a supply chain where the *content* is
fetched from parties who are not the dataset publisher and may not exist any more.

The domain-expiry mechanism has a property that makes it durable: it requires no compromise.
Nobody's server was breached, no credential was stolen, no vulnerability was exploited. A
domain lapsed and was bought at auction, which is a legal commercial transaction. There is no
anomaly and no incident, which is why {{cite:carlini2023poisoning}}'s framing —
*practical* — is the right word.

The targeted/broad asymmetry has a mechanistic explanation worth stating. A backdoor works by
creating an association between a rare trigger and a behaviour. Rare triggers have no
competing gradient — nothing else in the corpus says anything about that token sequence — so a
handful of examples suffice. Broad degradation must overcome the clean signal everywhere, so
it needs to be a meaningful share of the gradient. **The attack that needs the least data is
the one that touches the least of the model**, and it is also the one whose effect is invisible
until the trigger appears.

That connects directly to {{ch:sec-data-leakage}}'s memorisation result. The property that
makes a secret memorable — distinctiveness, no near-duplicates, high surprise — is the same
property that makes a backdoor trigger cheap. They are the same mechanism read in two
directions: one recovers what was put in, the other puts in what will be recovered.

On the chain, the reason composite trust is so much worse than any link is the same reason
{{ch:ops-versioning}}'s reproducibility was: conjunctions punish. What is different here is
that the links are owned by different organisations, so improving one requires a commercial
conversation rather than an engineering ticket. The two links with the lowest $p$ are also the
two with no owner you can escalate to.

Finally, the reason "re-host the corpus" outranks everything else. It does not reduce the
probability that the corpus was poisoned before you mirrored it — that risk is frozen at the
mirror date. What it does is convert an *ongoing* exposure into a *point-in-time* one: the
upstream can change every day and your mirror cannot, so you have replaced a continuous
unsignable link with a discrete signable one. **The value is in stopping the clock**, which is
also why the mirror's date matters and why re-mirroring resets the exposure.

## 8. Implementation

The first listing prices the attack.

```python {tier=A name=poisoning-cost-is-per-fraction-not-per-record}
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
```

## 9. Practical Example

{{cite:carlini2023poisoning}}'s reference point, generalised:

```
Reference point: 0.01% of a 400,000,000-item dataset for $60.
That is $600,000 per unit fraction, or $6,000 for 1%.

                       attack goal   fraction needed       items         cost   volume-detectable?
--------------------------------------------------------------------------------------------------
    backdoor on one trigger phrase         0.000200%         800            1                   no
  bias one entity's representation         0.003000%      12,000           18                   no
              degrade one language         0.120000%     480,000          720                maybe
    insert a factual claim broadly         0.400000%   1,600,000        2,400                  yes
        degrade general capability         6.000000%  24,000,000       36,000                  yes
```

**$1 against $36,000**, and the cheap end is the invisible one
({{eq:targeted-poisoning-is-orders-cheaper-than-broad}}). Note that the price is per fraction:
**dataset size does not appear** ({{eq:poisoning-cost-is-per-fraction-not-per-record}}).

```
  rank                       attack goal         cost   damage   damage per $1k
-------------------------------------------------------------------------------
     1    backdoor on one trigger phrase            1      9.0          7,500.0
     2  bias one entity's representation           18      6.5            361.1
     5        degrade general capability       36,000      8.0              0.2
```

**The attacker's ranking is the exact reverse of the detector's sensitivity.**

```
                   entry point         controlled by   attack share   strongest control
---------------------------------------------------------------------------------------
            base model weights    the model provider           1.5%           signature
            pretraining corpus    the model provider          15.1%             nothing
           fine-tuning dataset                   you          21.5%              review
                    RAG corpus                   you          30.2%         ingest scan
    tool server / MCP endpoint         a third party          13.2%             pinning
            package dependency         a third party           9.3%            lockfile
```

You control **57%** of attack share; the pretraining corpus is **15.1%** with a strongest
control of *nothing* — {{eq:reproducibility-is-a-product-over-artefacts}} in security form.

```
                         control   attack share covered   effort   per effort                          covers
---------------------------------------------------------------------------------------------------------------
       pin and hash dependencies                    19%      1.0        0.190   packages, not their behaviour
   scan the RAG corpus at ingest                    42%      2.0        0.210         what passes the scanner
        pin tool-server versions                    22%      1.5        0.147     the version, not the server
```

Read the last column: **every control covers something narrower than its name.**

The second listing composes the chain.

```python {tier=A name=trust-is-a-product-over-the-supply-chain}
"""Trust in an AI system is a product over its supply chain, and one unsigned link zeroes it.

A deployed system is a chain: base weights from one party, a fine-tune from another, an
adapter, an embedding model, a retrieval corpus, a set of tool servers, a package tree. Each
link is trusted or it is not, and the composite trust is the product
(eq:trust-is-a-product-over-the-supply-chain).

Verification helps and it has a hard edge: it covers what is signed and attests to what the
signature actually claims, which is usually "this is the artefact the publisher published"
and not "this artefact is not malicious"
(eq:provenance-covers-only-signed-links).

This listing computes the composite, prices detection methods that do not need provenance,
and shows which link is worth fixing.
"""
# (link, P(this link is clean), can it be signed?, cost to verify)
CHAIN = [
    ("base model weights",         0.985, True,  0.5),
    ("pretraining corpus",         0.870, False, 0.0),
    ("fine-tuning dataset",        0.940, True,  2.0),
    ("adapter or LoRA weights",    0.975, True,  0.4),
    ("embedding model",            0.990, True,  0.3),
    ("retrieval corpus",           0.820, False, 3.0),
    ("tool server endpoints",      0.910, True,  1.5),
    ("package dependency tree",    0.960, True,  1.0),
]

print("Trust as a product over the chain.")
print()
print(f"{'link':>30}{'P(clean)':>11}{'signable':>11}{'cumulative':>13}"
      f"{'contribution to loss':>23}")
print("-" * 88)
cum = 1.0
contrib = {}
for name, p, sign, cost in CHAIN:
    prev = cum
    cum *= p
    contrib[name] = (p, sign, cost, prev - cum)
    print(f"{name:>30}{p:>11.3f}{('yes' if sign else 'no'):>11}"
          f"{cum:>13.4f}{prev - cum:>23.4f}")
print("-" * 88)
print(f"{'COMPOSITE':>30}{'':>11}{'':>11}{cum:>13.4f}")

print()
print(f"eight links averaging {sum(p for n, p, s, c in CHAIN) / len(CHAIN):.3f} "
      f"compose to {cum:.4f}")
print(f"P(something in the chain is compromised) = {1 - cum:.1%}")

print()
print()
print("What verifying the signable links buys.")
print()
VERIFY_LIFT = 0.65        # signing raises P(clean) toward 1 by this fraction of the gap
print(f"{'link':>30}{'P before':>11}{'P after':>10}{'cost':>8}"
      f"{'gain per cost':>16}")
print("-" * 75)
lift = {}
for name, p, sign, cost in CHAIN:
    if not sign:
        lift[name] = (p, p, cost, 0.0)
        print(f"{name:>30}{p:>11.3f}{p:>10.3f}{'--':>8}{'--':>16}")
        continue
    after = p + VERIFY_LIFT * (1.0 - p)
    lift[name] = (p, after, cost, (after - p) / cost)
    print(f"{name:>30}{p:>11.3f}{after:>10.3f}{cost:>8.1f}"
          f"{(after - p) / cost:>16.4f}")

signed_cum = 1.0
for name, p, sign, cost in CHAIN:
    signed_cum *= lift[name][1]
print()
print(f"composite after verifying every signable link: {signed_cum:.4f}")
print(f"compromise probability falls from {1 - cum:.1%} to {1 - signed_cum:.1%}")

unsigned = [n for n, p, s, c in CHAIN if not s]
unsigned_p = 1.0
for n in unsigned:
    unsigned_p *= dict((x[0], x[1]) for x in CHAIN)[n]
print()
print(f"the two unsignable links alone contribute {1 - unsigned_p:.1%}")
print(f"which is {(1 - unsigned_p) / (1 - signed_cum):.0%} of the residual")

print()
print()
print("What a signature actually attests to.")
print()
ATTESTS = [
    ("model weight signature",  "this is what the publisher published", "no"),
    ("package lockfile hash",   "this is the version you resolved",     "no"),
    ("SBOM entry",              "this component was included",          "no"),
    ("build provenance",        "this came from that source and CI",    "partly"),
    ("reproducible build",      "the source produces these bytes",      "partly"),
    ("corpus manifest hash",    "this is the corpus you indexed",       "no"),
]
print(f"{'artefact':>28}{'what it attests':>42}"
      f"{'says it is safe?':>19}")
print("-" * 89)
for name, says, safe in ATTESTS:
    print(f"{name:>28}{says:>42}{safe:>19}")

print()
print("Every row answers 'is this the thing you meant to get'. None answers")
print("'is the thing you meant to get malicious'.")

print()
print()
print("Detection methods, for the links provenance cannot reach.")
print()
DETECT = [
    ("held-out clean evaluation",   0.14, 0.71, 1.2),
    ("loss anomaly during training", 0.09, 0.44, 0.8),
    ("activation clustering",        0.31, 0.22, 4.0),
    ("nearest-neighbour corpus audit", 0.26, 0.18, 2.5),
    ("canary trigger probes",        0.55, 0.06, 3.0),
    ("output monitoring in production", 0.19, 0.62, 2.0),
]
print(f"{'method':>34}{'catches targeted':>19}{'catches broad':>16}"
      f"{'effort':>9}{'per effort':>13}")
print("-" * 91)
det = {}
for name, targ, broad, eff in DETECT:
    combined = 0.75 * targ + 0.25 * broad
    det[name] = (targ, broad, eff, combined / eff)
    print(f"{name:>34}{targ:>19.0%}{broad:>16.0%}{eff:>9.1f}"
          f"{combined / eff:>13.3f}")

print()
print("Weighted 3:1 toward targeted attacks, because those are the cheap ones.")

best_det = max(det, key=lambda n: det[n][3])
print(f"best: {best_det} at {det[best_det][3]:.3f}")

print()
print()
print("Composite miss rate against the two attack classes.")
print()
miss_t, miss_b = 1.0, 1.0
for name, targ, broad, eff in DETECT:
    miss_t *= (1 - targ)
    miss_b *= (1 - broad)
print(f"{'attack class':>24}{'all six methods miss':>24}"
      f"{'cost of one attack':>22}{'expected loss':>16}")
print("-" * 86)
for label, miss, dmg in (("targeted backdoor", miss_t, 9.0),
                         ("broad degradation", miss_b, 8.0)):
    print(f"{label:>24}{miss:>24.1%}{dmg:>22.1f}{miss * dmg:>16.2f}")

print()
print()
print("And the ranking that follows: fix the link, do not chase the poison.")
print()
ACTIONS = [
    ("verify every signable link",      1 - signed_cum, 5.7),
    ("re-host the retrieval corpus",    0.11,           4.0),
    ("train on a curated corpus only",  0.13,           30.0),
    ("run all six detection methods",   0.06,           13.5),
    ("pin and re-verify tool servers",  0.05,           1.5),
]
print(f"{'action':>34}{'compromise probability after':>31}{'effort':>9}"
      f"{'reduction per effort':>23}")
print("-" * 97)
act = {}
for name, after, eff in ACTIONS:
    red = max(0.0, (1 - cum) - after)
    act[name] = (after, eff, red / eff)
    print(f"{name:>34}{after:>31.1%}{eff:>9.1f}{red / eff:>23.4f}")
best_act = max(act, key=lambda n: act[n][2])

print(f"""
The chain table is the arithmetic that should open every AI supply-chain conversation. Eight
links, each individually respectable -- the lowest is {min(p for n, p, s, c in CHAIN):.3f} --
compose to {cum:.4f} (eq:trust-is-a-product-over-the-supply-chain).

**A {1 - cum:.1%} chance that something in the chain is compromised**, from links that would
each pass a review. That is ch:ops-versioning's product-over-artefacts result with a different
quantity in the product, and it has the same shape: the composite is dominated by the weakest
links and no single link's improvement rescues it.

The contribution column names them. `{CHAIN[5][0]}` and `{CHAIN[1][0]}` contribute
{contrib[CHAIN[5][0]][3] + contrib[CHAIN[1][0]][3]:.4f} of the loss between them, and both are
in the `signable: no` column.

The verification table is the good news and its limit. Signing every signable link takes the
composite from {cum:.4f} to {signed_cum:.4f} -- compromise probability from
{1 - cum:.1%} to {1 - signed_cum:.1%} -- for {sum(c for n, p, s, c in CHAIN if s):.1f} units
of effort.

Then read the line under it. **The two unsignable links alone contribute
{1 - unsigned_p:.1%}**, which is {(1 - unsigned_p) / (1 - signed_cum):.0%} of what remains
(eq:provenance-covers-only-signed-links). Verification is worth doing and it converges to a
floor set by the links nobody can sign: a pretraining corpus you did not assemble and a
retrieval corpus that changes every day.

The attestation table is the part most often over-read. Every row answers **"is this the thing
you meant to get?"** A model signature says the publisher published these bytes. A lockfile
hash says this is the version you resolved. An SBOM entry says the component was included.

**None of them says the thing you meant to get is not malicious.** Build provenance and
reproducible builds get partway -- they tie the artefact to a source and a process -- and even
they attest to origin rather than to behaviour. Provenance answers a substitution question, and
poisoning is not a substitution attack.

The detection table is what is left for the unsignable links, and the numbers are modest.
`{best_det}` returns {det[best_det][3]:.3f} per unit of effort, weighted three-to-one toward
targeted attacks because ch:sec-poisoning's first listing showed those are the cheap ones.

Notice the shape of that table. Methods that catch *broad* degradation --
`held-out clean evaluation` at {DETECT[0][2]:.0%}, `output monitoring` at
{DETECT[5][2]:.0%} -- are cheap and effective. Methods that catch *targeted* backdoors are
expensive and weak, except canary probes, which catch {DETECT[4][1]:.0%} of the triggers **you
thought of**.

The composite miss table is the honest summary. All six methods together miss
{miss_t:.1%} of targeted backdoors and {miss_b:.1%} of broad degradation. The cheap attack is
the one that gets through.

Which produces the ranking in the last table and the recommendation of this chapter.
**Fix the link, do not chase the poison.** `{best_act}` returns
{act[best_act][2]:.4f} of compromise-probability reduction per unit of effort, against
{act['run all six detection methods'][2]:.4f} for running every detector in the literature --
{act[best_act][2] / act['run all six detection methods'][2]:.0f} times better.

Re-hosting the retrieval corpus -- taking a corpus you do not control and making it one you do
-- is the single largest move available, because it converts an unsignable link into a signable
one.

Detection is the control you build for the links you cannot re-host, and there are always
some. It is a residual measure and it should be budgeted as one.""")
```

```
                          link   P(clean)   signable   cumulative   contribution to loss
----------------------------------------------------------------------------------------
            base model weights      0.985        yes       0.9850                 0.0150
            pretraining corpus      0.870         no       0.8569                 0.1280
           fine-tuning dataset      0.940        yes       0.8055                 0.0514
              retrieval corpus      0.820         no       0.6376                 0.1400
         tool server endpoints      0.910        yes       0.5802                 0.0574
       package dependency tree      0.960        yes       0.5570                 0.0232
----------------------------------------------------------------------------------------
                     COMPOSITE                             0.5570
```

Eight links averaging **0.931** compose to **0.5570** — a **44.3%** compromise probability
({{eq:trust-is-a-product-over-the-supply-chain}}), with the two unsignable links contributing
most of the loss.

```
composite after verifying every signable link: 0.6553
compromise probability falls from 44.3% to 34.5%
the two unsignable links alone contribute 28.7%
which is 83% of the residual
```

**Verification converges to a floor set by what cannot be signed**
({{eq:provenance-covers-only-signed-links}}).

```
                    artefact                           what it attests   says it is safe?
-----------------------------------------------------------------------------------------
      model weight signature      this is what the publisher published                 no
       package lockfile hash          this is the version you resolved                 no
            build provenance         this came from that source and CI             partly
        corpus manifest hash            this is the corpus you indexed                 no
```

Every row answers *"is this the thing you meant to get?"* **None answers "is the thing you
meant to get malicious?"**

```
                            method   catches targeted   catches broad   effort   per effort
-------------------------------------------------------------------------------------------
         held-out clean evaluation                14%             71%      1.2        0.235
              canary trigger probes                55%              6%      3.0        0.142
             activation clustering                31%             22%      4.0        0.072

            attack class    all six methods miss    cost of one attack   expected loss
--------------------------------------------------------------------------------------
       targeted backdoor                   14.6%                   9.0            1.31
       broad degradation                    3.7%                   8.0            0.30
```

All six methods miss **14.6%** of targeted backdoors and **3.7%** of broad degradation — the
cheap attack again.

```
                            action   compromise probability after   effort   reduction per effort
-------------------------------------------------------------------------------------------------
        verify every signable link                          34.5%      5.7                 0.0173
      re-host the retrieval corpus                          11.0%      4.0                 0.0833
    train on a curated corpus only                          13.0%     30.0                 0.0104
    run all six detection methods                            6.0%     13.5                 0.0284
```

**Fix the link, do not chase the poison.**

## 10. Production Considerations

Price your exposure per fraction, not per record. A bigger corpus costs the same to poison and
is harder to audit.

Assume targeted backdoors and not broad degradation. The cheap attack is the one you will get
and the one your monitoring is blind to.

Enumerate your supply chain and mark each link signable or not. The unsignable set is your
floor and most teams have never listed it.

Mirror and sign your retrieval corpus. It converts the largest unsignable link into a signable
one and is the highest-return action in this chapter.

Pin tool-server versions and re-verify their behaviour, not just their version.
{{cite:gaire2025mcpsok}}'s analysis is explicit that a server can change what it returns
without changing anything you pinned.

Run a held-out clean evaluation every training cycle. It is cheap and it catches the expensive
attack.

Maintain canary triggers and rotate them. They catch 55% of the triggers you thought of, which
is a real number and a bounded one.

## 11. Common Mistakes

**Assuming a large corpus dilutes poison.** Cost is per fraction and the denominator cancels.

**Building volume-based poison detection.** It finds the $36,000 attack and not the $1 one.

**Treating a signature as a safety claim.** It attests to identity, not behaviour.

**Verifying the signable links and stopping.** The unsignable ones are 83% of the residual.

**Pinning a tool-server version and calling it fixed.** The version is not the behaviour.

**Auditing the corpus you cannot enumerate.** The pretraining corpus's strongest control is
nothing, and pretending otherwise wastes the audit.

## 12. Failure Modes

**Expired-domain poisoning with no incident.** No breach, no vulnerability, a legal domain
purchase, and nothing anomalous to detect.

**Backdoor dormant until the trigger appears.** Eight hundred items, invisible in evaluation,
active on one phrase.

**Mirror taken after the poisoning.** The clock was stopped at the wrong moment and the
exposure is now frozen in.

**Tool server changed behaviour at a pinned version.** The lockfile is green and the server
returns something new.

**Canary probes covering last year's triggers.** 55% of what you thought of, and the attacker
read the same papers.

**Fine-tuning data reviewed by sampling.** A 0.003% attack survives any sample a human reads.

## 13. Alternatives

**Train only on a curated, self-assembled corpus.** Removes the pretraining link entirely, at
30 units of effort and a substantial capability cost.

**Use a provider who publishes corpus provenance.** Moves the trust to a contractual claim,
which is a different kind of assurance and a real one.

**Content-addressed datasets.** Distribute hashes of content rather than URLs, which closes
the expired-domain mechanism at a storage and licensing cost.

**Retrieval from a signed, versioned mirror.** The recommendation here — it converts the
largest unsignable link and stops the clock.

**Behavioural attestation of tool servers.** Periodically replay a fixed request set against
each server and hash the responses. Catches behaviour drift at a pinned version, and it is not
standard practice.

## 14. Evaluation

Compute the fraction each attack class would need against your corpus, and the price at
$600,000 per unit fraction. Put the dollar figures in the threat model.

List your supply chain and mark signable versus not. Compute the composite and the floor.

Test whether your monitoring detects a planted 0.003% perturbation. It almost certainly does
not, and knowing that is worth the exercise.

Plant canary triggers before training and probe for them after. Rotate them each cycle and
record the detection rate.

Replay a fixed request set against each tool server weekly and diff the responses. Version
pinning does not cover this.

## 15. Advanced Concepts

The independence assumed across chain links is wrong in a direction that makes things worse.
Links share upstreams: the base model, the embedding model and several packages may all derive
from the same handful of organisations and the same public corpora, so a compromise at a
shared root correlates several $p_\ell$ at once. Positive correlation makes the product
*higher* than the independent estimate when links are clean together, and makes the *tail* far
heavier — the realistic distribution is mostly-fine with a fat mode where several links fail
together. **The independent model understates the correlated-failure scenario**, which is the
one that produces an industry-wide incident rather than a company-specific one.

The cost model treats the attacker's price as linear in fraction, and the expired-domain
mechanism is lumpy rather than linear: a single domain covers whatever share of the dataset it
hosted, which is a draw from a heavy-tailed distribution. So the *marginal* price rises
sharply once the cheap large domains are taken, and the practical implication is that the
first attacker to look gets a much better price than the tenth. That makes the exposure
front-loaded in time and argues for content-addressing sooner rather than later.

There is an interaction with {{ch:sec-data-leakage}} that neither chapter states alone. The
same distinctiveness that makes a sequence memorable makes a trigger cheap, so a corpus with
good memorisation hygiene — deduplicated, entity-redacted, singleton-scanned — is also a corpus
where a planted trigger stands out. **Anti-memorisation controls are anti-backdoor controls**,
and the canary-probe methodology is literally the memorisation canary methodology pointed the
other way. Building one gets most of the other.

Finally, on what re-hosting does and does not buy. Mirroring converts an ongoing exposure into
a point-in-time one, which is the argument for it. It also *freezes in* whatever was already
poisoned at the mirror date, and removes the upstream's ability to fix it. So a mirror without
a re-verification schedule is a permanent commitment to whatever the corpus contained on one
Tuesday, and the right design is a mirror plus a diffing process against upstream that reviews
changes rather than accepting or ignoring them wholesale. That is more work than mirroring and
much less than curating, and it is the position almost nobody occupies.

## 16. Connection to Previous Chapters

{{eq:reproducibility-is-a-product-over-artefacts}} from {{ch:ops-versioning}} is
{{eq:trust-is-a-product-over-the-supply-chain}} with a different quantity in the product, and
the same conclusion: conjunctions punish, and the unsignable set is the floor.

{{eq:extraction-risk-does-not-vanish-at-one-occurrence}} from {{ch:sec-data-leakage}} is the
mechanism behind an 800-item backdoor, and {{sec:15-advanced-concepts}} argues the two
problems share their controls.

{{eq:the-attacker-need-not-be-present}} from {{ch:sec-prompt-injection}} extends here from
retrieval time to training time, with a longer delay and a larger blast radius.

{{eq:indirect-injection-amortises-over-retrievals}} from the same chapter is the RAG-poisoning
case; this chapter's contribution is the pricing and the supply-chain composition around it.

## 17. Exercises

1. Compute the price of each attack class against your own corpus at $600,000 per unit
   fraction. Which are under a thousand dollars?

2. List your supply chain, mark each link signable, and compute the composite and the floor.

3. Plant a 0.003% perturbation in a staging corpus and check whether any monitoring notices.

4. Replay a fixed request set against your tool servers weekly for a month. Did any response
   change at a pinned version?

5. Model correlated failure across links sharing an upstream, per
   {{sec:15-advanced-concepts}}. How much heavier is the tail?

## 18. Interview Questions

1. Our training corpus has four hundred million items. Does that make poisoning harder?

2. Which is cheaper for an attacker: a backdoor or general degradation? Which does our
   monitoring find?

3. We verify all model signatures. What does that guarantee?

4. What is our compromise probability across the whole supply chain?

5. Why is re-hosting the retrieval corpus worth more than running every detector?

6. Our tool servers are version-pinned. What is still open?

## 19. Research Questions

1. How correlated are supply-chain link failures in practice, given shared upstreams?

2. How heavy-tailed is the per-domain share of web-scale datasets, and how fast does the
   attacker's marginal price rise?

3. How much do anti-memorisation controls reduce backdoor implantation success?

4. Can behavioural attestation of tool servers be standardised cheaply enough to be default?

## 20. Chapter Summary

Poisoning is priced per fraction, and that single fact reorganises the threat model.

{{cite:carlini2023poisoning}}'s **0.01% for $60** generalises to **$6,000 per percent**, with
**dataset size absent from the price**
({{eq:poisoning-cost-is-per-fraction-not-per-record}}) — so a bigger corpus costs the same to
poison and is harder to audit. And the fraction required spans five orders of magnitude: a
trigger backdoor at **0.0002%** and **$1**, general degradation at **6%** and **$36,000**
({{eq:targeted-poisoning-is-orders-cheaper-than-broad}}). Damage per thousand dollars: **7,500**
against **0.2**. **The attacker's ranking is the reverse of the detector's sensitivity.**

The supply chain composes badly. Eight links averaging **0.931** give a composite of
**0.5570** — a **44.3%** compromise probability
({{eq:trust-is-a-product-over-the-supply-chain}}). You control **57%** of attack share; the
pretraining corpus is 15% with a strongest control of *nothing*.

Verification takes compromise probability to **34.5%** and stops there, because the two
unsignable links contribute **28.7%** — **83%** of the residual
({{eq:provenance-covers-only-signed-links}}). And a signature attests to identity, never to
behaviour: every attestation answers *"is this the thing you meant to get"* and none answers
*"is it malicious."*

Detection is the residual measure and it is weak where it matters: all six methods miss
**14.6%** of targeted backdoors against **3.7%** of broad degradation. Ranked by reduction per
unit of effort, **re-hosting the retrieval corpus returns 0.0833** against **0.0173** for
verification and **0.0104** for curation.

What generalises beyond poisoning is the shape of the argument. Two of this part's earlier
chapters ended at "spend on what a fooled model can reach" and "trim the context and the tool
list." This one ends at "own the link." In all three the effective move is to change what the
system is composed of rather than to add a detector to what it already is — and in all three
the detector is the thing that gets budgeted, because it is the thing that looks like security
work.

Carry forward: **cost is per fraction, so size is not a defence**, and **verify what you can
and re-host what you cannot**.

## 21. Further Reading

- {{cite:carlini2023poisoning}} — the $60 figure and the expired-domain mechanism, which is
  what makes this a production concern rather than a research one.
- {{cite:greshake2023indirect}} — the retrieval-time version of the same absence, and the
  channel RAG poisoning fires through.
- {{cite:hou2025mcp}} — the tool-integration layer as a supply-chain link, with runtime
  addition as the complicating factor.
- {{cite:gaire2025mcpsok}} — a systematisation of the same layer's security and safety
  properties, including what version pinning does not cover.
