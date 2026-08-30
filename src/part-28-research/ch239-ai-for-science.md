---
id: res-ai-for-science
number: 239
part: XXVIII
tier: full
status: draft
requires: [approval-quality-falls-with-volume, oversight-is-a-conjunction-of-preconditions,
           coverage-is-a-union-not-a-sum, self-training-improves-only-the-verifiable-fraction]
provides: [autonomous-output-shifts-the-bottleneck-to-review,
           a-finding-is-worth-its-verification-probability,
           replication-has-higher-information-per-dollar-than-novelty,
           an-unpublished-negative-is-repeated-by-everyone-else]
citations: [lu2024aiscientist, chan2024mlebench, testini2025dsautomation, wang2025solvedcorrectly]
---

## 1. Learning Objectives

By the end of this chapter you will be able to show that automating hypothesis generation moves
the bottleneck to verification and compute where it binds; price an unverified candidate finding
from its prior and the verification cost it will consume; rank experiment types by information
and value per dollar and explain why the ranking inverts the field's incentives; compute the
duplicated cost of an unpublished negative result; and identify where automation's comparative
advantage actually lies.

## 2. Why This Matters

An automated research system produces candidate findings cheaply
({{cite:lu2024aiscientist}}, {{cite:chan2024mlebench}}). Each is worth something only once
somebody establishes it is true, and that costs **16.15 expert-hours** through the full
verification stack.

Six reviewers supply **2,496** hours a year. So at 40 candidates a year everything is verified
and **4.4** findings are established; at 400,000 candidates, **0.0%** is verified and **17.0**
findings are established. **Beyond a few hundred candidates a year, additional generation
produces exactly zero additional established findings**
({{eq:autonomous-output-shifts-the-bottleneck-to-review}}).

Which changes what a candidate is worth. At the generator's base rate of **0.11**, verifying one
is worth **−$1,390**; it only becomes worth verifying above a prior of about **0.25**
({{eq:a-finding-is-worth-its-verification-probability}}).

The second half asks which experiments to run at all, and information theory answers it.
Measured in value per dollar, reproduction from artefacts scores **0.6721** and a bold novel
hypothesis **0.1250** — **5.4× worse**
({{eq:replication-has-higher-information-per-dollar-than-novelty}}).

And the largest single loss is structural: an unpublished negative costs the field **$115,200**
to rediscover, against **$24,000** to share — **4.8×**
({{eq:an-unpublished-negative-is-repeated-by-everyone-else}}).

## 3. Prerequisites

{{eq:approval-quality-falls-with-volume}} from {{ch:sec-permissions}} is the same fixed human
capacity meeting a scalable producer, in a third setting.

{{eq:oversight-is-a-conjunction-of-preconditions}} from {{ch:rai-oversight}} supplies the review
budget's structure: a reviewer needs time, and time is what verification consumes.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is the verification stack — gates
that overlap in what they catch.

{{eq:self-training-improves-only-the-verifiable-fraction}} from {{ch:res-continual}} is the same
bound in a different setting: a loop improves what it can check, and here what can be checked is
capacity-bound rather than definition-bound.

## 4. Intuitive Explanation

The exciting version of automated science is a system that has ideas. The useful version is
different, and the arithmetic says so plainly.

Start with what happens after an idea. A candidate finding is not a finding; it is a claim.
Turning it into knowledge takes: automated sanity checks (free, removes 31% of false
candidates), reading the claim and method (**0.35** expert-hours, removes 44%), re-running the
analysis (**1.80** hours, 62%), and independent replication (**14.00** hours, 88%).

The full stack is **16.15 expert-hours** per candidate. Six reviewers at eight hours a week
supply **2,496** hours a year — about **155** candidates.

Now scale the generator. At 40 candidates a year, everything is verified and **4.4** true
findings are established. At 400, capacity binds: **17.0**. At 4,000: **17.0**. At 40,000:
**17.0**. At 400,000: **17.0**.

**Beyond a few hundred candidates a year, additional generation produces exactly zero additional
established findings** ({{eq:autonomous-output-shifts-the-bottleneck-to-review}}). The share
verified falls from 38.6% to 0.0% and the numerator does not move, because the numerator was
never the generator's.

That number should frame every claim about automated research. **The generator has not been the
bottleneck for some time.** A system producing ten thousand plausible hypotheses and one
producing ten produce the same number of established findings, given the same review capacity.

This is {{ch:sec-permissions}}' approval queue and {{ch:rai-oversight}}'s review budget arriving
in a third setting, with the same structure — a fixed human capacity meeting a scalable producer
— and the same conclusion: **the interesting decision is what to spend the capacity on.**

So what should it be spent on? Take 40,000 candidates a year and vary the policy.

Full stack on everything: 155 cleared, **17** true findings, 2 false. Read plus re-run with 10%
replication: 703 cleared, 77 true, 84 false. Read plus re-run: 1,161 cleared, 128 true, 152
false. Read only: 7,131 cleared, **784** true and **2,452** false. Automated checks only: 40,000
cleared, 4,400 true and 24,564 false.

Note which direction that goes, because it inverts advice this book has given repeatedly.
Elsewhere — {{ch:ev-framework}}'s gate placement, {{ch:sec-permissions}}' approval queue — cheap
checks applied widely beat thorough checks applied narrowly. **Here they do not.**

The difference is the base rate. When 89% of candidates are wrong, a gate that removes most but
not all of them still passes more false findings than true ones, and a false finding entered
into the literature costs more than a missed true one. At a 3:1 penalty, full-stack-on-everything
wins outright.

**A low base rate makes thoroughness the correct policy** — the same arithmetic
{{ch:sec-jailbreaks}} used for guardrail precision, pointing the other way because the costs are
asymmetric in the other direction.

Now the per-candidate view, which produces the sharpest number in the chapter.

A true finding is worth $24,000. A false one entering the literature costs $9,000 in expectation
with a 12% chance of doing real damage. Verification costs 16.15 hours at $190 — **$3,068**.

At a prior of 0.02, a candidate is worth **−$3,647** to verify. At 0.05, −$2,894. At **0.11** —
the generator's base rate — **−$1,390**. At 0.25, +$2,122. At 0.50, +$8,392.

**The average candidate from an automated generator is not worth the expert time it would
consume** ({{eq:a-finding-is-worth-its-verification-probability}}).

That is not an argument against automated research. It is an argument that the generator's output
must be triaged before it reaches a human, and that **the triage is the valuable component.**

Price the triage. An automated plausibility score raises the prior from 0.11 to 0.19 for
**$1** per candidate. Cross-checking against the literature reaches **0.28** for $6. A cheap
pilot experiment reaches **0.46** for $59 of compute. An expert 10-minute screen reaches
**0.51** — barely better than the pilot — and spends the one resource that is capacity-bound.

**Three of the four triage steps cost no expert time at all.** That is where the leverage in
autonomous research sits: not in generating more candidates, not in reviewing faster, but in
raising the prior of what reaches a reviewer.

It is worth being precise about why that works, because the three levers are not
interchangeable. Generating more candidates raises the numerator of a fraction whose denominator
is fixed, so it does nothing. Reviewing faster raises the denominator, which does help — but the
only ways to review faster are to hire, which is linear and slow, or to skip stages, which the
policy table showed is catastrophic at this base rate. Triage does neither: it changes what
arrives, so the same expert-hours produce more established findings from the same generator.
That is the only one of the three levers that is both effective and cheap, and it is the least
discussed.

Which brings the chapter to its second question. Given a fixed budget of experiments, which ones
should be run?

Information theory answers it cleanly. An experiment's information content is the entropy of its
outcome, so an experiment whose result is nearly certain teaches almost nothing however important
its subject.

A bold novel hypothesis succeeds 11% of the time — entropy **0.4999** bits — and costs $24,000:
**0.0208** bits per thousand dollars. An ablation of one component is nearly a coin flip —
**0.9988** bits — and costs $2,200: **0.4540**. A benchmark rerun: 0.5842 bits at $700 —
**0.8346**.

The obvious objection is that bits are not value. So weight them: novelty ×6.0, an incremental
variation ×2.2, replication ×1.0, ablation ×1.4, reproduction ×1.1, a rerun ×0.6.

**It does not change the ranking.** Reproduction from artefacts leads at **0.6721** value per
thousand dollars, ablation at 0.6356, a benchmark rerun at 0.5008 — and a bold novel hypothesis
trails at **0.1250**, **5.4× worse**
({{eq:replication-has-higher-information-per-dollar-than-novelty}}).

Two things drive that. Bold hypotheses are *unlikely*, and an unlikely binary outcome has low
entropy — 0.4999 bits against 0.9988 for a coin-flip ablation. And they are expensive, by a
factor of 34 over the cheapest row.

What does the field actually do? 34% of effort on bold hypotheses, 48% on incremental variations,
**4%** on replication, 11% on ablation, 2% on reproduction, 1% on reruns — delivering **1,499.3**
units of value from $6,000,000.

Reserve 25% for novelty anyway — bold hypotheses are the only source of genuinely new directions,
whatever their value per dollar — and fill the rest greedily under plausible caps. That delivers
**2,267.1** against **1,499.3**: **a factor of 1.5, with no new capability required.** The gain
is entirely allocative, and novelty's share falls only from 34% to 25%.

Then the loss that no budget contains.

A negative result is information. A negative result that is not published is information the
field pays for repeatedly. If 40 groups could have the idea and 12% of them try it, the field
spends **$115,200** discovering something one group already knew — and publishing it would have
cost **$24,000**.

**A waste multiple of 4.8×**
({{eq:an-unpublished-negative-is-repeated-by-everyone-else}}), and nobody bears that cost
individually, which is why it persists. It is a commons problem in a field that measures
individuals, and it is the clearest case in this book of a large, computable, entirely
unaddressed inefficiency.

Now put automation against that table, because this is the chapter's point.

Automation reduces the cost of a bold novel hypothesis by **1.6×** — the design, the reasoning
and the writing are the hard parts and they are the parts that resist automation. It reduces
direct replication by **7.1×**, ablation by **11.1×**, reproduction from artefacts by **20.0×**,
and a benchmark rerun by **33.3×**, because those are mechanical
({{cite:testini2025dsautomation}}).

So the value-per-dollar gap does not close under automation. It **widens**, from 5.4× to
**82.8×**.

**Automation's comparative advantage is precisely in the experiments the field under-runs.**
Replication, ablation, reproduction from artefacts, negative results — mechanical, cheap,
high-entropy, and unrewarded. That is a much less exciting claim than an automated scientist
having ideas, and on these numbers it is where essentially all of the available value is.

There is a pleasing symmetry in that. The first half of the chapter said the human bottleneck is
verification. The second half says the highest-value experiments are verification-shaped. **The
same conclusion arrives from a capacity argument and from an information argument**, and the
thing both point at is the thing nobody gets promoted for.

That coincidence is not an accident, and it is worth unpacking one step further. Both arguments
turn on the same underlying asymmetry: producing a claim is cheap and establishing one is
expensive, in expert time and in information alike. A field organised around the production of
claims will therefore accumulate them faster than it can convert them, and the accumulated
inventory of unverified claims is not neutral — it is cited, built upon, and occasionally
believed. The capacity argument says that inventory grows without bound; the information
argument says clearing it is the highest-return work available. **The bottleneck and the
opportunity are the same object seen from two sides.**

## 5. Formal Explanation

**The review bound.** With generation rate $G$, per-candidate verification cost $h$, reviewer
capacity $C$ and true rate $\beta$, established findings are
$F = \beta \min(G, C/h)$. For $G > C/h$ this is constant in $G$: $\partial F/\partial G = 0$.
The generator's contribution to output is exactly zero above the capacity threshold.

**Candidate value.** For prior $\pi$, true value $v$, false cost $c$ with damage probability
$\delta$, and verification cost $k$: $\mathbb{E}[V] = \pi v - (1-\pi)c\delta - k$, so a candidate
is worth verifying iff $\pi > (k + c\delta)/(v + c\delta)$. Both triage — which raises $\pi$ —
and cheaper verification — which lowers $k$ — move the threshold, and triage is the one that
does not consume the scarce resource.

**Information per cost.** For a binary outcome with probability $p$, information is
$H(p) = -p\log_2 p - (1-p)\log_2(1-p)$, maximised at $p = 1/2$ and vanishing as $p \to 0$ or
$1$. Value per cost is $H(p)\mu/c$. Because ambitious experiments have small $p$ *and* large $c$,
both factors work against them, and a value multiplier $\mu$ must exceed
$\frac{H(p')\,c}{H(p)\,c'}$ to reverse the ranking — here about 30, against an assumed 6.

**The commons loss.** With $n$ groups each independently trying an idea with probability $q$,
expected duplicated cost is $nqc$ against a publication cost $c$. The waste multiple is $nq$,
which exceeds 1 whenever more than one group would try — that is, essentially always for an idea
worth having.

## 6. Mathematical Foundation

Output is bounded by review, not generation:

$$F = \beta \min\!\left(G, \frac{C}{h}\right) = 17.0 \ \text{for all } G \ge 400, \qquad h = 16.15,\ C = 2{,}496$$ (eq:autonomous-output-shifts-the-bottleneck-to-review)

A candidate is worth its prior:

$$\mathbb{E}[V] = \pi v - (1-\pi)c\delta - k = -\$1{,}390 \ \text{at } \pi = 0.11, \quad > 0 \ \text{above } \pi \approx 0.25$$ (eq:a-finding-is-worth-its-verification-probability)

Information per dollar inverts the incentive ranking:

$$\frac{H(p)\,\mu}{c} = 0.6721 \ (\text{reproduction}) \ \text{against} \ 0.1250 \ (\text{novelty})$$ (eq:replication-has-higher-information-per-dollar-than-novelty)

And the unpublished negative is paid for $nq$ times:

$$\text{waste multiple} = nq = 40 \times 0.12 = 4.8, \qquad \$115{,}200 \ \text{against} \ \$24{,}000$$ (eq:an-unpublished-negative-is-repeated-by-everyone-else)

## 7. Internal Mechanics

The review bound has a mechanism that is worth stating because it is so easily obscured by
enthusiasm. Verification is not a step in a pipeline whose throughput can be raised by improving
the previous step; it is a *separate resource pool* with its own capacity, denominated in a
currency — expert attention — that the generator cannot manufacture. Every argument of the form
"the system can now produce a thousand hypotheses a day" is an argument about a quantity that has
already stopped mattering.

The base-rate inversion deserves care because it contradicts a heuristic this book has otherwise
endorsed. Cheap-wide gates win when the thing being filtered is mostly good and the cost of a
miss is bounded. They lose when the thing being filtered is mostly bad and a false pass is
durable. A false finding in the literature is durable: it is cited, built on, and expensive to
retract. That combination — low base rate, high false-pass cost — is exactly where thoroughness
wins, and it is the shape of automated research output.

The triage result has the cleanest mechanism in the chapter. The threshold
$\pi > (k + c\delta)/(v + c\delta)$ can be reached from either side, and the two sides consume
different resources. Lowering $k$ means faster verification, which consumes expert design effort
and eventually expert time. Raising $\pi$ means better filtering, which consumes compute and
retrieval. Since the constraint is expert time, **the correct lever is the one that does not
touch it**, and it happens also to be the cheaper one.

The information-per-dollar result has a mechanism that explains a real feature of research
culture rather than just criticising it. Entropy peaks at $p = 1/2$, so the most informative
experiment is the one you genuinely cannot call. Ambitious research deliberately targets low $p$
— that is what ambition means — and thereby targets low entropy. The field is not being
irrational; it is optimising *value conditional on success*, which is a different objective and
one that individual careers reward. The arithmetic here optimises expected information, and the
two diverge precisely because the multiplier the field assigns to novelty (6×) is far below the
roughly 30× it would need to be to reverse the ranking.

Finally, the commons loss and the automation asymmetry compound in a specific direction. The
experiments automation is best at are the ones whose value is most external to the person running
them — replications benefit the field, ablations benefit the reader, negatives benefit whoever
would have tried next. **Automation lowers the cost of exactly the work whose benefits are
hardest to capture**, which is why it will not happen by default and why it is a good target for
deliberate institutional effort rather than for individual incentives.

## 8. Implementation

The first listing prices verification.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kg1}
"""Automating research moves the bottleneck to review, and review is capacity-bound.

An automated research system produces candidate findings cheaply (cite:lu2024aiscientist,
cite:chan2024mlebench). Each candidate is worth something only once somebody has established
that it is true, and establishing that costs expert time that does not scale with the generator.

So the throughput of the whole enterprise is not set by how many findings are produced. It is
set by how many can be verified, and automation raises the numerator while leaving the
denominator alone (eq:autonomous-output-shifts-the-bottleneck-to-review).

Which changes what a "finding" is worth. An unverified candidate has a value equal to its
probability of surviving verification, minus the cost of the verification it will consume
(eq:a-finding-is-worth-its-verification-probability).
"""
BASE_TRUE = 0.11          # share of generated candidates that would survive verification
REVIEWERS = 6
REVIEW_HOURS_PER_WEEK = 8.0
WEEKS = 52


# (stage, hours of expert time per candidate, share of false candidates it removes)
GATES = [
    ("automated sanity checks",    0.00,  0.31),
    ("read the claim and method",  0.35,  0.44),
    ("re-run the analysis",        1.80,  0.62),
    ("independent replication",   14.00,  0.88),
]

print("What it costs to establish that a candidate finding is true.")
print()
print(f"{'verification stage':>28}{'expert hours':>15}{'false removed':>16}"
      f"{'candidates / reviewer-year':>29}")
print("-" * 88)
CAPACITY = REVIEWERS * REVIEW_HOURS_PER_WEEK * WEEKS
for name, hours, removed in GATES:
    per_year = CAPACITY / hours if hours > 0 else float("inf")
    ps = f"{per_year:>29,.0f}" if hours > 0 else f"{'unbounded':>29}"
    print(f"{name:>28}{hours:>15.2f}{removed:>16.0%}{ps}")

print()
print(f"{REVIEWERS} reviewers at {REVIEW_HOURS_PER_WEEK:.0f} hours a week is"
      f" {CAPACITY:,.0f} expert-hours a year")

print()
print()
print("Now scale the generator and watch the bottleneck move.")
print()
FULL_STACK = sum(h for n, h, r in GATES)
print(f"{'candidates / year':>19}{'expert-hours needed':>22}{'capacity':>12}"
      f"{'share verifiable':>19}{'verified true findings':>25}")
print("-" * 97)
scaled = {}
for gen in (40, 400, 4_000, 40_000, 400_000):
    need = gen * FULL_STACK
    share = min(1.0, CAPACITY / need)
    true_found = gen * share * BASE_TRUE
    scaled[gen] = (need, share, true_found)
    print(f"{gen:>19,}{need:>22,.0f}{CAPACITY:>12,.0f}"
          f"{share:>19.1%}{true_found:>25,.1f}")

print()
print(f"a {400_000 // 40:,}x increase in generation yields"
      f" {scaled[400_000][2] / scaled[40][2]:.1f}x the verified findings")
print("(eq:autonomous-output-shifts-the-bottleneck-to-review)")

print()
print()
print("So the question is which gate to spend the hours on.")
print()
print(f"{'policy':>34}{'hours / candidate':>20}{'candidates cleared':>21}"
      f"{'true findings':>16}{'false findings':>17}")
print("-" * 108)
GEN = 40_000
POLICIES = [
    ("full stack on everything",       FULL_STACK,  1.00),
    ("read + re-run, replicate 10%",   0.35 + 1.80 + 0.10 * 14.00, 0.10),
    ("read + re-run only",             0.35 + 1.80, 0.00),
    ("read only",                      0.35,        0.00),
    ("automated checks only",          0.00,        0.00),
]
pol = {}
for name, hours, rep_share in POLICIES:
    cleared = GEN if hours == 0 else min(GEN, CAPACITY / hours)
    kept_false_rate = (1 - 0.31) * (1 - 0.44 if hours >= 0.35 else 1.0)
    if hours >= 2.0:
        kept_false_rate *= (1 - 0.62)
    kept_false_rate *= (1 - 0.88 * rep_share)
    true_out = cleared * BASE_TRUE
    false_out = cleared * (1 - BASE_TRUE) * kept_false_rate
    pol[name] = (cleared, true_out, false_out)
    print(f"{name:>34}{hours:>20.2f}{cleared:>21,.0f}"
          f"{true_out:>16,.0f}{false_out:>17,.0f}")

print()
best_p = max(pol, key=lambda n: pol[n][1] - 3.0 * pol[n][2])
print(f"at a 3:1 penalty for a false finding, the best policy is `{best_p}`")
print(f"it yields {pol[best_p][1]:,.0f} true and {pol[best_p][2]:,.0f} false")

print()
print()
print("What a single unverified candidate is actually worth.")
print()
VALUE_TRUE, COST_FALSE = 24_000.0, 9_000.0
HOUR_COST = 190.0
print(f"{'prior that it is true':>24}{'value if true':>16}{'cost if false':>16}"
      f"{'verification cost':>20}{'net worth':>13}")
print("-" * 89)
worth = {}
for prior in (0.02, 0.05, 0.11, 0.25, 0.50, 0.80):
    ver = FULL_STACK * HOUR_COST
    net = prior * VALUE_TRUE - (1 - prior) * COST_FALSE * 0.12 - ver
    worth[prior] = net
    print(f"{prior:>24.2f}{VALUE_TRUE:>16,.0f}{COST_FALSE:>16,.0f}"
          f"{ver:>20,.0f}{net:>13,.0f}")

break_even = None
for prior in sorted(worth):
    if worth[prior] > 0 and break_even is None:
        break_even = prior
print()
print(f"a candidate is worth verifying above a prior of about {break_even:.2f}")
print(f"the generator's base rate is {BASE_TRUE:.2f}")
print("(eq:a-finding-is-worth-its-verification-probability)")

print()
print()
print("Which makes triage worth more than generation.")
print()
TRIAGE = [
    ("no triage",                        BASE_TRUE, 0.000, "--"),
    ("automated plausibility score",     0.19,      0.004, "cheap"),
    ("cross-check against literature",   0.28,      0.031, "retrieval"),
    ("cheap pilot experiment",           0.46,      0.310, "compute"),
    ("expert 10-minute screen",          0.51,      0.310, "the scarce resource"),
]
print(f"{'triage step':>34}{'prior after triage':>21}{'cost / candidate':>19}"
      f"{'net worth after':>18}{'uses':>21}")
print("-" * 113)
for name, prior, cost, uses in TRIAGE:
    ver = FULL_STACK * HOUR_COST
    net = prior * VALUE_TRUE - (1 - prior) * COST_FALSE * 0.12 - ver - cost * HOUR_COST
    print(f"{name:>34}{prior:>21.2f}{cost * HOUR_COST:>19,.0f}"
          f"{net:>18,.0f}{uses:>21}")

print()
print("Three of the four triage steps cost no expert time at all.")

print(f"""
The verification table is the constraint the whole subject runs into. Establishing that one
candidate finding is true costs {FULL_STACK:.2f} expert-hours through the full stack, and
{REVIEWERS} reviewers at {REVIEW_HOURS_PER_WEEK:.0f} hours a week supply
{CAPACITY:,.0f} hours a year -- about {CAPACITY / FULL_STACK:,.0f} candidates.

The scaling table is what happens when the generator gets good
(eq:autonomous-output-shifts-the-bottleneck-to-review). At {40:,} candidates a year everything is
verified and {scaled[40][2]:.1f} findings are established. At {400:,}, capacity binds and
{scaled[400][2]:.1f} are established. At {4_000:,}, {scaled[4000][2]:.1f}. At {400_000:,},
{scaled[400_000][2]:.1f}.

**Beyond a few hundred candidates a year, additional generation produces exactly zero additional
established findings.** The share verified falls from {scaled[400][1]:.1%} to
{scaled[400_000][1]:.1%} and the numerator does not move, because the numerator was never the
generator's.

That is the number that should frame every claim about automated research. **The generator is not
the bottleneck and has not been for some time.** A system that produces ten thousand plausible
hypotheses and a system that produces ten produce the same number of established findings, if the
review capacity is the same.

It is also ch:sec-permissions' approval queue and ch:rai-oversight's review budget arriving in a
third setting, with the same structure: a fixed human capacity meeting a scalable producer, and
the same conclusion -- **the interesting decision is what to spend the capacity on.**

The policy table is that decision. `full stack on everything` clears
{pol['full stack on everything'][0]:,.0f} candidates and produces
{pol['full stack on everything'][1]:,.0f} true findings.
`{best_p}` clears {pol[best_p][0]:,.0f} and produces {pol[best_p][1]:,.0f} true and
{pol[best_p][2]:,.0f} false.

Note which direction that goes, because it inverts the usual advice. Elsewhere in this book --
ch:ev-framework's gate placement, ch:sec-permissions' approval queue -- cheap checks applied
widely beat thorough checks applied narrowly. Here they do not: `read only` clears
{pol['read only'][0]:,.0f} candidates and produces {pol['read only'][1]:,.0f} true findings
alongside **{pol['read only'][2]:,.0f} false ones.**

The difference is the base rate. When {1 - BASE_TRUE:.0%} of candidates are wrong, a gate that
removes most but not all of them still passes more false findings than true ones, and a false
finding entered into the literature costs more than a missed true one. **A low base rate makes
thoroughness the correct policy**, which is the same arithmetic ch:sec-jailbreaks used for
guardrail precision, pointing the other way because the costs are asymmetric in the other
direction.

The worth table is the per-candidate view and it produces the sharpest number here. At a prior of
{BASE_TRUE:.2f} -- the generator's base rate -- a candidate is worth
**{worth[0.11]:,.0f}** to verify. It only becomes worth verifying above a prior of about
{break_even:.2f} (eq:a-finding-is-worth-its-verification-probability).

**The average candidate from an automated generator is not worth the expert time it would
consume.** That is not an argument against automated research; it is an argument that the
generator's output must be triaged before it reaches a human, and that the triage is the valuable
component.

The triage table prices exactly that. `cross-check against literature` raises the prior from
{BASE_TRUE:.2f} to {0.28:.2f} at {0.031 * HOUR_COST:.0f} per candidate and no expert time.
`cheap pilot experiment` reaches {0.46:.2f} for {0.310 * HOUR_COST:.0f} of compute.

An `expert 10-minute screen` reaches {0.51:.2f} -- barely better than the pilot -- and spends the
one resource that is capacity-bound. **Three of the four triage steps cost no expert time at
all**, which is where the leverage in autonomous research actually sits: not in generating more
candidates, and not in reviewing faster, but in raising the prior of what reaches the reviewer.""")
```

## 9. Practical Example

What verification costs:

```
          verification stage   expert hours   false removed   candidates / reviewer-year
----------------------------------------------------------------------------------------
     automated sanity checks           0.00             31%                    unbounded
   read the claim and method           0.35             44%                        7,131
         re-run the analysis           1.80             62%                        1,387
     independent replication          14.00             88%                          178
```

```
  candidates / year   expert-hours needed    capacity   share verifiable   verified true findings
-------------------------------------------------------------------------------------------------
                 40                   646       2,496             100.0%                      4.4
                400                 6,460       2,496              38.6%                     17.0
              4,000                64,600       2,496               3.9%                     17.0
             40,000               646,000       2,496               0.4%                     17.0
            400,000             6,460,000       2,496               0.0%                     17.0
```

**A 10,000× increase in generation yields 3.9× the findings, and none of it above 400**
({{eq:autonomous-output-shifts-the-bottleneck-to-review}}).

```
                            policy   hours / candidate   candidates cleared   true findings   false findings
------------------------------------------------------------------------------------------------------------
          full stack on everything               16.15                  155              17                2
      read + re-run, replicate 10%                3.55                  703              77               84
                read + re-run only                2.15                1,161             128              152
                         read only                0.35                7,131             784            2,452
             automated checks only                0.00               40,000           4,400           24,564
```

**A low base rate makes thoroughness the correct policy** — the inverse of this book's usual
gate advice.

```
   prior that it is true   value if true   cost if false   verification cost    net worth
-----------------------------------------------------------------------------------------
                    0.02          24,000           9,000               3,068       -3,647
                    0.11          24,000           9,000               3,068       -1,390
                    0.25          24,000           9,000               3,068        2,122
                    0.50          24,000           9,000               3,068        8,392

                       triage step   prior after triage   cost / candidate   net worth after                 uses
------------------------------------------------------------------------------------------------------------------
                         no triage                 0.11                  0            -1,390                   --
      automated plausibility score                 0.19                  1              -488                cheap
    cross-check against literature                 0.28                  6               568            retrieval
             cheap pilot experiment                 0.46                 59             2,662              compute
```

**The average candidate is not worth verifying** at the generator's base rate
({{eq:a-finding-is-worth-its-verification-probability}}); triage fixes that without expert time.

The second listing prices the experiments.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/kg2}
"""The experiments worth automating are the ones nobody wants to run.

The first listing found that review capacity, not generation, bounds automated research. This one
asks a different question: given a fixed budget of experiments, which ones are worth running?

Information theory gives a clean answer. An experiment's information content is the entropy of
its outcome, so an experiment whose result is nearly certain teaches almost nothing however
important its subject. Divide by cost and the ranking is not the one the field's incentives
produce (eq:replication-has-higher-information-per-dollar-than-novelty).

And the largest single loss is structural. A negative result that is not published is repeated,
independently, by everyone else who would have had the idea
(eq:an-unpublished-negative-is-repeated-by-everyone-else).
"""
import math


def entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# (experiment type, cost in dollars, P(the interesting outcome), value multiplier,
#  cost multiple once automated)
EXPERIMENTS = [
    ("a bold novel hypothesis",       24_000.0, 0.11, 6.0, 0.62),
    ("an incremental variation",       9_000.0, 0.62, 2.2, 0.31),
    ("direct replication",             6_000.0, 0.55, 1.0, 0.14),
    ("an ablation of one component",   2_200.0, 0.48, 1.4, 0.09),
    ("reproduction from artefacts",    1_400.0, 0.72, 1.1, 0.05),
    ("a benchmark rerun",                700.0, 0.86, 0.6, 0.03),
]

print("What each kind of experiment teaches, per dollar.")
print()
print(f"{'experiment':>32}{'cost':>11}{'P(interesting)':>17}{'bits':>9}"
      f"{'bits per $1k':>15}{'value per $1k':>16}")
print("-" * 100)
rows = {}
for name, cost, p, mult, auto in EXPERIMENTS:
    h = entropy(p)
    bits_k = h / (cost / 1000.0)
    val_k = h * mult / (cost / 1000.0)
    rows[name] = (cost, p, h, bits_k, val_k, mult, auto)
    print(f"{name:>32}{cost:>11,.0f}{p:>17.2f}{h:>9.4f}"
          f"{bits_k:>15.4f}{val_k:>16.4f}")

BEST_BITS = max(rows, key=lambda n: rows[n][3])
BEST_VALUE = max(rows, key=lambda n: rows[n][4])
NOVEL = "a bold novel hypothesis"
print()
print(f"most bits per dollar: {BEST_BITS} at {rows[BEST_BITS][3]:.4f}")
print(f"most value per dollar: {BEST_VALUE} at {rows[BEST_VALUE][4]:.4f}")
print(f"a bold novel hypothesis: {rows[NOVEL][4]:.4f}"
      f" -- {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x worse")
print("(eq:replication-has-higher-information-per-dollar-than-novelty)")

print()
print()
print("What the field actually spends its effort on.")
print()
ACTUAL = {
    "a bold novel hypothesis":     0.34,
    "an incremental variation":    0.48,
    "direct replication":          0.04,
    "an ablation of one component": 0.11,
    "reproduction from artefacts": 0.02,
    "a benchmark rerun":           0.01,
}
BUDGET = 6_000_000.0
print(f"budget {BUDGET:,.0f}")
print()
print(f"{'experiment':>32}{'actual share':>15}{'experiments run':>18}"
      f"{'value delivered':>18}{'value per $1k':>16}")
print("-" * 99)
actual_value = 0.0
for name, cost, p, mult, auto in EXPERIMENTS:
    share = ACTUAL[name]
    n = BUDGET * share / cost
    v = n * entropy(p) * mult
    actual_value += v
    print(f"{name:>32}{share:>15.0%}{n:>18,.0f}{v:>18,.1f}{rows[name][4]:>16.4f}")
print("-" * 99)
print(f"{'TOTAL':>32}{1.0:>15.0%}{'':>18}{actual_value:>18,.1f}")

opt_n = BUDGET / rows[BEST_VALUE][0]
opt_value = opt_n * rows[BEST_VALUE][2] * rows[BEST_VALUE][5]
print()
print(f"all-in on {BEST_VALUE}: {opt_n:,.0f} experiments, {opt_value:,.1f} value")
print(f"a factor of {opt_value / actual_value:.1f} over the observed allocation")

print()
print()
print("Which is not a recommendation, because value is not fungible.")
print()
CAPS = {
    "a bold novel hypothesis":     1.00,
    "an incremental variation":    0.55,
    "direct replication":          0.30,
    "an ablation of one component": 0.22,
    "reproduction from artefacts": 0.14,
    "a benchmark rerun":           0.08,
}
MIN_NOVEL = 0.25
print(f"reserving {MIN_NOVEL:.0%} for novelty, then filling by value per dollar")
print()
print(f"{'experiment':>32}{'value per $1k':>16}{'share cap':>12}"
      f"{'budget taken':>16}{'value delivered':>18}")
print("-" * 94)
remaining, port_value = 1.0 - MIN_NOVEL, 0.0
n_exp = BUDGET * MIN_NOVEL / rows[NOVEL][0]
nov_value = n_exp * rows[NOVEL][2] * rows[NOVEL][5]
port_value += nov_value
print(f"{NOVEL:>32}{rows[NOVEL][4]:>16.4f}{'reserved':>12}"
      f"{BUDGET * MIN_NOVEL:>16,.0f}{nov_value:>18,.1f}")
for name in sorted(rows, key=lambda n: -rows[n][4]):
    if name == NOVEL:
        continue
    take = min(CAPS[name], remaining)
    remaining -= take
    n_exp = BUDGET * take / rows[name][0]
    v = n_exp * rows[name][2] * rows[name][5]
    port_value += v
    print(f"{name:>32}{rows[name][4]:>16.4f}{CAPS[name]:>12.0%}"
          f"{BUDGET * take:>16,.0f}{v:>18,.1f}")
print("-" * 94)
USED = 1.0 - remaining
print(f"{'TOTAL':>32}{'':>16}{USED:>12.0%}{BUDGET * USED:>16,.0f}"
      f"{port_value:>18,.1f}")

print()
print(f"a capped portfolio delivers {port_value:,.1f}"
      f" against the observed {actual_value:,.1f}")
print(f"a factor of {port_value / actual_value:.1f}, with no new capability required")

print()
print()
print("Now the loss nobody accounts for: the negative that is not published.")
print()
GROUPS = 40
print(f"{'groups who would try it':>26}{'P(each tries)':>16}{'expected repeats':>19}"
      f"{'wasted cost':>15}{'if published':>15}")
print("-" * 91)
waste = {}
for p_try in (0.02, 0.05, 0.12, 0.25, 0.50):
    repeats = GROUPS * p_try
    cost = repeats * rows[NOVEL][0]
    waste[p_try] = cost
    print(f"{GROUPS:>26}{p_try:>16.2f}{repeats:>19.1f}"
          f"{cost:>15,.0f}{rows[NOVEL][0]:>15,.0f}")

print()
print(f"at {0.12:.0%} the field spends {waste[0.12]:,.0f} to learn something")
print(f"one group already knew, and would have shared for {rows[NOVEL][0]:,.0f}")
print(f"a waste multiple of {waste[0.12] / rows[NOVEL][0]:.1f}x")
print("(eq:an-unpublished-negative-is-repeated-by-everyone-else)")

print()
print()
print("And what automation actually changes.")
print()
print(f"{'experiment':>32}{'cost now':>12}{'cost automated':>17}"
      f"{'reduction':>12}{'value per $1k, automated':>27}")
print("-" * 100)
auto_rows = {}
for name, cost, p, mult, auto in EXPERIMENTS:
    new_cost = cost * auto
    v = entropy(p) * mult / (new_cost / 1000.0)
    auto_rows[name] = (new_cost, v)
    print(f"{name:>32}{cost:>12,.0f}{new_cost:>17,.0f}"
          f"{1 / auto:>11.1f}x{v:>27.4f}")

best_auto = max(auto_rows, key=lambda n: auto_rows[n][1])
print()
print(f"automation reduces `{NOVEL}` cost by {1 / rows[NOVEL][6]:.1f}x")
print(f"and `a benchmark rerun` by {1 / rows['a benchmark rerun'][6]:.1f}x")
print(f"the gap in value per dollar widens from"
      f" {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x to"
      f" {auto_rows[best_auto][1] / auto_rows[NOVEL][1]:.1f}x")

print(f"""
The first table is the ranking the field's incentives do not produce. Measured in bits of outcome
entropy per dollar, `{BEST_BITS}` leads at {rows[BEST_BITS][3]:.4f} and `{NOVEL}` trails at
{rows[NOVEL][3]:.4f}.

The obvious objection is that bits are not value -- a novel result matters more than a benchmark
rerun -- so the table carries a value multiplier: {rows[NOVEL][5]:.1f} for novelty against
{rows['a benchmark rerun'][5]:.1f} for a rerun. **It does not change the ranking.**
`{BEST_VALUE}` still leads at {rows[BEST_VALUE][4]:.4f} value per thousand dollars and novelty
is {rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x worse
(eq:replication-has-higher-information-per-dollar-than-novelty).

Two things drive that and both are worth naming. Bold hypotheses are *unlikely*, and an unlikely
binary outcome has low entropy -- {rows[NOVEL][2]:.4f} bits against
{rows['an ablation of one component'][2]:.4f} for a coin-flip ablation. And they are expensive,
by a factor of {rows[NOVEL][0] / rows['a benchmark rerun'][0]:.0f} over the cheapest row.

The allocation table shows what is actually done. {ACTUAL[NOVEL]:.0%} of effort on bold
hypotheses, {ACTUAL['an incremental variation']:.0%} on incremental variations, and
{ACTUAL['direct replication']:.0%} on replication -- delivering {actual_value:,.1f} units of
value from {BUDGET:,.0f}.

The portfolio table is the realistic alternative, because value is not fungible and no field
should spend everything on reruns. Reserving {MIN_NOVEL:.0%} for bold hypotheses -- they are the
only source of genuinely new directions, whatever their value per dollar -- and filling the rest
by value per dollar under plausible caps delivers **{port_value:,.1f} against
{actual_value:,.1f}** -- a factor of {port_value / actual_value:.1f}, **with no new capability
required.** The gain is entirely allocative, and novelty's share falls only from
{ACTUAL[NOVEL]:.0%} to {MIN_NOVEL:.0%}.

The repeats table is the largest single loss and the one no budget contains
(eq:an-unpublished-negative-is-repeated-by-everyone-else). If {GROUPS} groups could have the idea
and {0.12:.0%} of them try it, the field spends {waste[0.12]:,.0f} discovering something one
group already knew. Publishing it costs {rows[NOVEL][0]:,.0f} -- a waste multiple of
**{waste[0.12] / rows[NOVEL][0]:.1f}x**.

Nobody bears that cost individually, which is why it persists. It is a commons problem in a field
that measures individuals, and it is the clearest case in this book of a large, computable, and
entirely unaddressed inefficiency.

The last table is what automation changes, and it is the point of the chapter. Automation reduces
the cost of `{NOVEL}` by {1 / rows[NOVEL][6]:.1f}x -- the design, the reasoning and the writing
are hard to automate. It reduces `a benchmark rerun` by
{1 / rows['a benchmark rerun'][6]:.1f}x and `reproduction from artefacts` by
{1 / rows['reproduction from artefacts'][6]:.1f}x, because those are mechanical.

So the value-per-dollar gap **widens** under automation, from
{rows[BEST_VALUE][4] / rows[NOVEL][4]:.1f}x to
{auto_rows[best_auto][1] / auto_rows[NOVEL][1]:.1f}x.

**Automation's comparative advantage is precisely in the experiments the field under-runs.**
Replication, ablation, reproduction from artefacts, negative results -- mechanical, cheap,
high-entropy, and unrewarded. That is a much less exciting claim than an automated scientist
generating novel hypotheses, and on these numbers it is where essentially all of the available
value is.""")
```

```
                      experiment       cost   P(interesting)     bits   bits per $1k   value per $1k
----------------------------------------------------------------------------------------------------
         a bold novel hypothesis     24,000             0.11   0.4999         0.0208          0.1250
        an incremental variation      9,000             0.62   0.9580         0.1064          0.2342
              direct replication      6,000             0.55   0.9928         0.1655          0.1655
    an ablation of one component      2,200             0.48   0.9988         0.4540          0.6356
     reproduction from artefacts      1,400             0.72   0.8555         0.6110          0.6721
               a benchmark rerun        700             0.86   0.5842         0.8346          0.5008
```

**Novelty is 5.4× worse per dollar even after a 6× value multiplier**
({{eq:replication-has-higher-information-per-dollar-than-novelty}}).

```
                      experiment   actual share   experiments run   value delivered   value per $1k
---------------------------------------------------------------------------------------------------
         a bold novel hypothesis            34%                85             255.0          0.1250
        an incremental variation            48%               320             674.5          0.2342
              direct replication             4%                40              39.7          0.1655
     reproduction from artefacts             2%                86              80.7          0.6721
---------------------------------------------------------------------------------------------------
                           TOTAL           100%                             1,499.3

                      experiment   value per $1k   share cap    budget taken   value delivered
----------------------------------------------------------------------------------------------
         a bold novel hypothesis          0.1250    reserved       1,500,000             187.5
     reproduction from artefacts          0.6721         14%         840,000             564.6
    an ablation of one component          0.6356         22%       1,320,000             839.0
               a benchmark rerun          0.5008          8%         480,000             240.4
        an incremental variation          0.2342         55%       1,860,000             435.6
----------------------------------------------------------------------------------------------
                           TOTAL                        100%       6,000,000           2,267.1
```

**1.5× more value, with no new capability** and novelty's share only falling from 34% to 25%.

```
   groups who would try it   P(each tries)   expected repeats    wasted cost   if published
-------------------------------------------------------------------------------------------
                        40            0.12                4.8        115,200         24,000
                        40            0.50               20.0        480,000         24,000

                      experiment    cost now   cost automated   reduction   value per $1k, automated
----------------------------------------------------------------------------------------------------
         a bold novel hypothesis      24,000           14,880        1.6x                     0.2016
              direct replication       6,000              840        7.1x                     1.1819
     reproduction from artefacts       1,400               70       20.0x                    13.4428
               a benchmark rerun         700               21       33.3x                    16.6925
```

**Automation widens the gap from 5.4× to 82.8×**, in favour of the experiments nobody runs.

## 10. Production Considerations

Report established findings, not candidates. Above the capacity threshold the second number
carries no information about the first.

Compute your review capacity in expert-hours and divide by the verification stack. That is your
ceiling and it is usually a surprise.

Spend on triage, not on generation. Three of four triage steps cost no expert time and each moves
the prior more than a faster reviewer would.

Verify thoroughly at a low base rate. This is the one place in the book where cheap-wide loses,
and it loses badly — 2,452 false findings against 17.

Publish negatives, and instrument how often your organisation repeats an experiment someone
inside it already ran. The multiple is computable and nobody computes it.

Point automation at replication, ablation and reproduction. Automation reduces those costs by
7–33× against 1.6× for novel work.

Measure your generator's base rate directly. Everything downstream — triage value, verification
policy, candidate worth — depends on it.

Keep 25% for genuinely novel work regardless of the arithmetic. It is the only source of new
directions, and the portfolio result holds with it reserved.

## 11. Common Mistakes

**Reporting hypotheses generated.** Zero marginal findings above 400 a year.

**Scaling the generator to increase output.** The output was never the generator's.

**Applying cheap-wide verification.** 89% of candidates are wrong and a false finding is durable.

**Spending expert time on triage.** It is the capacity-bound resource, and the pilot experiment
gets nearly the same prior for compute.

**Ranking experiments by importance rather than by information per dollar.** Ambition selects
for low entropy.

**Treating an unpublished negative as costless.** 4.8× duplicated, borne by everyone else.

**Pointing automation at ideation.** 1.6× cost reduction against 33.3× for the mechanical work.

## 12. Failure Modes

**A system generating thousands of hypotheses into a six-reviewer queue.** 0.0% verified.

**A literature enriched with 2,452 false findings.** Read-only verification at scale.

**A verification budget spent on the highest-profile candidates.** Prior unimproved, capacity
consumed.

**A replication programme cancelled for low impact.** The highest value per dollar in the table.

**An organisation repeating its own negative results.** No register, and 4.8× the cost.

**An automated scientist evaluated on idea quality.** The measured stage is not the binding one.

## 13. Alternatives

**Automate verification rather than generation.** Attacks the actual constraint — every hour
saved is an hour of the bound.

**Raise the prior with cheap pilots.** 0.46 from 0.11 for $59 of compute, and no expert time.

**Publish a negative-results register.** Removes a 4.8× duplication for the cost of one
publication, and requires an institution rather than a technique.

**Benchmark-driven autonomous evaluation** ({{cite:chan2024mlebench}},
{{cite:wang2025solvedcorrectly}}). Makes the verification step mechanical for a restricted class
of claims, which is precisely the class where automation's advantage is largest.

**Hire reviewers.** Unglamorous, linear, and the only thing that moves the ceiling directly —
worth pricing against the alternatives rather than dismissing.

## 14. Evaluation

Measure your generator's base rate by fully verifying a random sample. Everything downstream
depends on it, and a random sample is the only way to get it.

Measure review capacity in expert-hours and the verification stack in hours per candidate. Their
ratio is the ceiling.

Track established findings per quarter, and plot it against candidates generated. If the second
rises and the first does not, the bottleneck is confirmed.

Run a triage A/B: measure the prior of candidates that pass each triage step against a fully
verified control. Report the prior lift per dollar.

Count how many experiments your organisation has run twice. That is the internal version of the
commons loss and it is measurable from a lab notebook.

## 15. Advanced Concepts

The review-capacity model treats verification as a fixed cost per candidate, and it is not:
verifying a candidate that is similar to one already verified is much cheaper, because the
method, the data and the reviewer's context are shared. That means a generator producing
*clustered* candidates is worth more than one producing scattered ones at the same base rate,
which is the opposite of the diversity objective generators are usually given. **The right
generator objective is not novelty per candidate but findings per reviewer-hour**, and those
diverge sharply.

The information-per-dollar analysis assumes outcomes are binary and independent, and research
outcomes are neither. A replication that fails is informative about the original claim *and*
about every result that built on it, so its information content is larger than its own entropy —
possibly much larger, since citation graphs are heavy-tailed. Accounting for that would widen the
gap in {{sec:9-practical-example}} rather than close it, and it suggests the right target for
replication is not a random claim but a highly-cited one.

There is a self-reference in this chapter that deserves naming. An automated research system that
learns from its own verified findings is running {{ch:res-continual}}'s self-improvement loop,
with the verifier being the review process and the verifiable fraction being whatever review
capacity can reach — which {{sec:9-practical-example}} showed is **0.4%** at scale. Under
{{eq:self-training-improves-only-the-verifiable-fraction}} such a system improves on almost
nothing, and the remedy is the same: raise the fraction that can be checked, which here means
automating verification rather than generation.

Finally, the value multipliers in the second listing are the load-bearing assumption and they are
not measured anywhere. The ranking reverses if novelty's multiplier exceeds roughly 30, and the
field behaves as though it does. Determining the true multiplier empirically — by tracking the
downstream impact of bold results against replications and ablations over a decade — is a
tractable study that nobody has run, and its answer would settle an argument that is currently
conducted entirely by assertion.

## 16. Connection to Previous Chapters

{{eq:approval-quality-falls-with-volume}} from {{ch:sec-permissions}} is the same capacity
constraint: a fixed pool of human attention meeting a producer that scales.

{{eq:oversight-is-a-conjunction-of-preconditions}} from {{ch:rai-oversight}} supplies the
time term — **16.15 hours** per candidate against **2,496** a year.

{{eq:coverage-is-a-union-not-a-sum}} from {{ch:ev-framework}} is the verification stack, whose
gates overlap in what they remove.

{{eq:self-training-improves-only-the-verifiable-fraction}} from {{ch:res-continual}} is the
self-reference in {{sec:15-advanced-concepts}}: an automated scientist learning from its own
verified output improves only the 0.4% that gets verified.

## 17. Exercises

1. Measure your generator's base rate on a fully verified random sample of 30 candidates.

2. Compute your review capacity, your verification stack cost, and the resulting ceiling on
   established findings.

3. Compute the net worth of a candidate at your base rate. Is it positive?

4. Price three triage steps for your domain by prior lift per dollar, and identify which consume
   expert time.

5. Rank the experiment types in your field by information and value per dollar. Where does your
   actual allocation sit?

6. Estimate the value multiplier novelty would need to justify your field's current allocation,
   per {{sec:15-advanced-concepts}}.

## 18. Interview Questions

1. Our system generates 500 hypotheses a day. How many findings is that?

2. What does it cost to establish that one candidate finding is true?

3. Why is thorough verification the right policy here when cheap-wide gates win elsewhere?

4. Where would you spend the next dollar: the generator, the reviewer, or the triage?

5. Which experiments have the highest information per dollar, and why does nobody run them?

6. What does an unpublished negative result cost?

## 19. Research Questions

1. What is the empirical downstream impact multiplier of novel results against replications and
   ablations?

2. How much cheaper is verification for a candidate similar to one already verified, and what
   generator objective follows?

3. What fraction of experiments within a single organisation are unknowing repeats?

4. Can verification be automated for a broad enough class of claims to move the capacity bound?

## 20. Chapter Summary

The exciting version of automated science is a system that has ideas. The arithmetic says the
useful version is a system that checks them.

Establishing one candidate finding costs **16.15 expert-hours**, and six reviewers supply
**2,496** a year. So output is **4.4** established findings at 40 candidates a year and
**17.0** at 400, at 4,000, at 40,000, at 400,000 — **beyond a few hundred candidates,
additional generation produces exactly zero additional findings**
({{eq:autonomous-output-shifts-the-bottleneck-to-review}}).

At that scale the policy question inverts this book's usual advice. Read-only verification clears
7,131 candidates and produces **784** true findings alongside **2,452** false ones; the full
stack produces **17** and **2**. **A low base rate makes thoroughness correct.**

And the per-candidate view is sharper still: at the generator's base rate of **0.11**, verifying
a candidate is worth **−$1,390**, turning positive only above **0.25**
({{eq:a-finding-is-worth-its-verification-probability}}). **The average automated candidate is
not worth the expert time it consumes** — which makes triage, not generation and not review
speed, the valuable component. Three of four triage steps cost no expert time at all.

The second half asks which experiments to run and gets an answer the field's incentives do not
produce. Reproduction from artefacts delivers **0.6721** value per thousand dollars, ablation
**0.6356**, a bold novel hypothesis **0.1250** — **5.4× worse**, even after a 6× value multiplier
({{eq:replication-has-higher-information-per-dollar-than-novelty}}). Reallocating within
plausible caps, while reserving 25% for novelty, delivers **2,267.1** against **1,499.3** — a
**1.5×** gain that is entirely allocative.

Then the loss no budget contains: an unpublished negative costs the field **$115,200** to
rediscover against **$24,000** to share — **4.8×**
({{eq:an-unpublished-negative-is-repeated-by-everyone-else}}) — borne by everyone and by nobody.

And automation's actual comparative advantage: it cuts the cost of a bold hypothesis by **1.6×**
and of a benchmark rerun by **33.3×**, widening the value gap from 5.4× to **82.8×**.

What runs through the chapter is that two independent arguments — one about human capacity, one
about information — point at the same neglected work. Replication, ablation, reproduction,
negative results: mechanical, cheap, high-entropy, unrewarded, and exactly what automation is
good at. **Automation lowers the cost of precisely the work whose benefits are hardest to
capture**, which is why it will not happen by default.

Carry forward: **count established findings, not candidates**, and **automate the checking, not
the ideas**.

## 21. Further Reading

- {{cite:lu2024aiscientist}} — an end-to-end automated research pipeline and what it produces.
- {{cite:chan2024mlebench}} — machine-learning engineering tasks with mechanical verification,
  which is the class where automation's advantage is largest.
- {{cite:testini2025dsautomation}} — automating the mechanical stages of data-science work.
- {{cite:wang2025solvedcorrectly}} — whether a passing result is a correct one, which is the
  verification question in miniature.
