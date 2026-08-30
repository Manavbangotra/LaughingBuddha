---
id: sec-jailbreaks
number: 223
part: XXVI
tier: full
status: draft
requires: [instructions-and-data-share-a-channel, detection-layers-fail-against-an-adaptive-attacker,
           f1-asserts-a-cost-ratio, threshold-is-the-decision-not-the-model]
provides: [jailbreak-surface-is-capability-minus-safety-coverage, safety-coverage-lags-capability-by-construction,
           guardrail-precision-is-set-by-the-base-rate, refusals-outnumber-prevented-harms]
citations: [wei2023jailbroken, zou2023universal, perez2022ignore, ji2023survey]
---

## 1. Learning Objectives

By the end of this chapter you will be able to decompose jailbreak surface into capability
minus safety coverage per domain, and compute an attacker's success across domains; explain
why competing objectives cannot be closed by more safety data; show that the uncovered-domain
count is constant once the safety-data lag is steady; compute a guardrail's precision from the
base rate and explain why most alarms are wrong; compute refusals per prevented harm; and
identify the cost ratio that decides a guardrail's threshold and is never stated.

## 2. Why This Matters

{{ch:sec-prompt-injection}} was about what an attacker makes the system *do*. This chapter is
about what they make it *say*, and the two have different structures.

{{cite:wei2023jailbroken}} named two failure modes. **Mismatched generalization** is the
structural one: safety training covers a subset of the domains the model is capable in, so
jailbreak surface is capability minus coverage, domain by domain
({{eq:jailbreak-surface-is-capability-minus-safety-coverage}}). In plain English, capability
0.98 against coverage 0.94 gives attack success **0.06**. In base64, capability 0.74 against
coverage 0.28 gives **0.53**.

An attacker who tries the single best domain succeeds **53.3%** of the time; one who tries all
eight succeeds **97.3%**. The defender must cover every domain and the attacker must find one.

And the gap does not close. Capability enters roughly 1.4 new domains a generation and safety
data follows about two generations behind, so once the lag is steady **the uncovered count is
constant at 5.8** ({{eq:safety-coverage-lags-capability-by-construction}}) — safety work
succeeding at exactly the rate capability advances.

Guardrails are the request-time layer, and they have a base-rate problem. At a **0.3%** base
rate, a guardrail at TPR 0.91 and FPR 0.04 produces 1,790 alarms a day of which **94% are
wrong** ({{eq:guardrail-precision-is-set-by-the-base-rate}}). At the cost-optimal threshold it
prevents 73 harms a day and refuses 3,017 legitimate users — **41 refusals per prevented harm**
({{eq:refusals-outnumber-prevented-harms}}).

Whether that is a good trade depends on a harm-to-refusal cost ratio that moves the optimal
threshold from **0.84 to 0.02**, and which nobody writes down.

## 3. Prerequisites

{{eq:instructions-and-data-share-a-channel}} from {{ch:sec-threat-model}} is why a framing
argument works at all: there is no channel in which "this is a fictional frame" can be marked
as data rather than instruction.

{{eq:detection-layers-fail-against-an-adaptive-attacker}} from the same chapter is applied
directly to guardrails in {{sec:9-practical-example}}, and it is the result that decides what
a guardrail is for.

{{eq:f1-asserts-a-cost-ratio}} and {{eq:threshold-is-the-decision-not-the-model}} from
{{ch:ev-classical-metrics}} are this chapter's second half in general form. A guardrail is a
classifier with a threshold, the threshold encodes a cost ratio, and here the cost of the
assumed ratio is paid by refused users.

{{cite:ji2023survey}}'s hallucination taxonomy is background for what "harmful output" covers
beyond the obvious categories.

## 4. Intuitive Explanation

Jailbreaking is usually presented as a contest of cleverness — someone finds a magic phrase,
the lab patches it, someone finds another. {{cite:wei2023jailbroken}} replaced that picture
with two mechanisms, and the second one says the contest does not converge.

**Competing objectives** is the first. A model is trained to be helpful and trained to be
safe, and on some requests those pull in opposite directions. The attacker's job is to raise
the helpfulness pressure until it exceeds the safety penalty.

You can see this in the framings that work. "Tell me how to do X" has low helpfulness
pressure. "For a research paper, explain how X works" has more. "Write a story in which a
character explains X" has more still, because now refusing means failing at creative writing.
"Continue this document" has the most, because the model's most fundamental training objective
is to continue text.

Nothing about the request changed across those. Only the reward for answering it. In the model
here, compliance goes from 0.1% at a bare framing to 60.7% at "continue this document."

That is worth stating carefully: **competing objectives is not a hole in the safety training,
it is the helpfulness training working.** The same gradient that makes the model useful is the
one being exploited, which is why more safety data alone does not close it — you would be
training against usefulness.

**Mismatched generalization** is the second mechanism and the structural one. Safety training
happened in some domains and not others. The model is capable in more domains than the safety
training reached.

Count it. In plain English prose — the domain almost all safety data is written in —
capability is 0.98 and coverage is 0.94, so attack success is 0.06. In base64, capability is
0.74 (the model can read and write it well enough) and coverage is 0.28, so success is 0.53.
Substitution ciphers: 0.44 capability, 0.09 coverage, 0.40 success. Typographic tricks: 0.38
and 0.06, giving 0.36.

The model is *less* capable in those domains and much *more* vulnerable, because the gap is
what matters and not the level.

Now stack the attacker's options. Trying the single best domain succeeds 53.3% of the time.
Trying the top four: 92.0%. Trying all eight: 97.3%.

**The defender must cover every domain and the attacker must find one.** That is the oldest
asymmetry in security, and here it is worse than usual because the domains are not enumerable
in advance. They are defined by what the model happens to be capable in, which is discovered
rather than designed.

Then run it forward. Each model generation extends capability into new domains — new
languages, new encodings, new modalities, new formats. Safety data for a domain arrives after
capability does, because you cannot collect safety data for a capability that does not exist
yet.

In the model here, capability enters about 1.4 new domains a generation and coverage follows
two generations behind. The result is that the uncovered count stabilises at 5.8 and stays
there. Not growing — but not shrinking either.

**Safety work is not failing in that picture. It is succeeding at exactly the rate capability
advances**, which leaves a standing gap of fixed size and changing contents. Anyone expecting
the jailbreak problem to be solved by better alignment is expecting the lag to go to zero, and
the lag is a data-collection latency rather than a research difficulty.

There is one more thing to say before moving to defences, and {{cite:zou2023universal}} is the
source of it. The domain list above reads like a catalogue of things clever humans thought of.
It is not — it is a search space, and the search can be automated. That paper found adversarial
suffixes by optimisation against open models, and they transferred to ChatGPT, Bard and Claude.

Which means a defence organised domain by domain assumes a domain-by-domain attacker, and that
assumption expired.

So: guardrails, the layer that runs at request time regardless of domain.

A guardrail is a classifier. It looks at a request (or a response) and decides whether to
allow it. And it has the problem every classifier on a rare event has.

Suppose genuinely harmful requests are 0.3% of traffic — which is generous for a consumer
product; the real figure is often lower. A guardrail with TPR 0.91 and FPR 0.04 sounds
excellent. Run it on 42,000 requests a day.

It fires 1,790 times. Of those, 115 are genuinely harmful and 1,675 are not. **94% of the
alarms are wrong.**

Nothing is broken. The false positives are drawn from a population 332 times larger than the
true positives, so a 4% false-positive rate produces far more alarms than a 91% true-positive
rate. This is exactly {{ch:ev-classical-metrics}}'s base-rate arithmetic, arriving where the
cost is a refused user rather than a misleading metric.

Move to the threshold. At maximum sensitivity, the daily cost is dominated by refusals. At the
cost-optimal threshold the total is lower and the true-positive rate is much worse.

The volumes are the number to sit with. At the cost-optimal threshold and a 0.3% base rate,
the guardrail prevents 73 harmful requests a day and refuses 3,017 legitimate ones. **41
refusals per prevented harm.** At a 0.05% base rate it is 249 to one.

Whether 41 refusals is worth one prevented harm depends entirely on what each costs. And the
ratio table shows how much rides on that: at 2:1 the optimal threshold is 0.84 and refuses 5
users a day; at 2,560:1 it is 0.02 and refuses 20,937.

**That ratio is never written down.** It is the same finding as
{{eq:f1-asserts-a-cost-ratio}} — a threshold chosen without a stated ratio has assumed one —
in a setting where the assumption is paid for by users who never learn why they were refused.

Then the adaptive attacker, from {{ch:sec-threat-model}}. At the cost-optimal threshold, the
guardrail blocks 58% of a single attempt and 0.0% of a thirty-attempt attacker.

Which forces the honest question: if a guardrail cannot stop a determined attacker, what is it
for?

Three things, and only one of them is blocking. It **removes casual misuse**, which is most of
the traffic that would otherwise reach the model — people trying something once because they
wondered, and who do not try again. It **raises attempts per success** by 2.4×, which is a
rate limiter on the attacker's search and composes with actual rate limits. And it **produces
a logged decision on every request**, which is the forensic and accountability artefact, and
is very often the reason the guardrail was funded in the first place.

What it does not do is bound anything.

Finally, the part that belongs in a product review rather than a security one. Refusal cost is
not spread evenly. Security research is 4% of traffic and 8.4× more likely to trip the
classifier. Medical and clinical queries are 7% at 6.1×. Non-English is 9% at 4.9×.

**The false positives land on the users with the most legitimate need for the capability**,
because proximity to a sensitive topic is what the classifier measures. That is not a tuning
problem and it does not improve with a better model — it is what the feature is. The remedies
are segment-aware thresholds and an appeal path, and both are product decisions.

## 5. Formal Explanation

**Surface as a difference.** Let domain $d$ have capability $c_d \in [0,1]$ and safety
coverage $s_d \in [0,1]$. Attack success in that domain is $a_d = c_d(1 - s_d)$: the model
must be able to comply and the safety training must not have reached the domain. An attacker
sampling domains independently succeeds with $1 - \prod_d (1 - a_d)$, which approaches one
quickly because the product is over domains and each term is bounded away from one.

**Competing objectives.** Model compliance as $\Pr[\text{comply}] = \Phi((h - p)/\sigma)$ for
helpfulness pressure $h$, safety penalty $p$ and decision noise $\sigma$. Framings raise $h$
without changing the underlying request. Raising $p$ uniformly raises refusals on benign
requests too, so the safety penalty is bounded above by the acceptable refusal rate — which
makes this a *constrained* problem rather than a solvable one.

**The lag.** Let capability reach $C(g) = C_0 + \alpha g$ domains at generation $g$, and
coverage $S(g) = C(g - \lambda)$ for lag $\lambda$. Then uncovered $U(g) = C(g) - C(g-\lambda)
= \alpha\lambda$, constant in $g$. Surface as a fraction, $U/C$, decreases — but the absolute
count does not, and the *contents* rotate.

**Guardrail precision.** With base rate $b$, $\text{precision} = b\,\text{TPR} /
[b\,\text{TPR} + (1-b)\,\text{FPR}]$, which for small $b$ approximates $b\,\text{TPR} /
[(1-b)\text{FPR}]$ — linear in the base rate and inversely proportional to FPR. No achievable
TPR rescues it.

**Refusals per prevented harm.** $R = (1-b)\text{FPR} / (b\,\text{TPR})$, which is
$\Theta(1/b)$. At the cost-optimal threshold the ratio is determined by $b$ and the ROC shape,
not by the guardrail's quality.

**The threshold.** Minimising $C(t) = N b (1 - \text{TPR}(t)) \kappa_H + N(1-b)\text{FPR}(t)
\kappa_R$ gives a first-order condition depending on $\kappa_H/\kappa_R$ alone. The optimal
threshold is therefore a function of a ratio the deployment must state, and defaults to
whatever the tool ships with when it does not.

## 6. Mathematical Foundation

Jailbreak surface, domain by domain:

$$a_d = c_d\,(1 - s_d), \qquad A = 1 - \prod_d (1 - a_d)$$ (eq:jailbreak-surface-is-capability-minus-safety-coverage)

At $c = 0.74$, $s = 0.28$: $a = 0.53$. Across eight domains, $A = 0.973$.

The standing gap:

$$U(g) = C(g) - C(g - \lambda) = \alpha\lambda \quad \text{for } g \ge \lambda$$ (eq:safety-coverage-lags-capability-by-construction)

At $\alpha = 1.4$ domains/generation and $\lambda = 2$: $U = 5.8$, constant.

Precision on a rare event:

$$\text{precision} = \frac{b\,\mathrm{TPR}}{b\,\mathrm{TPR} + (1-b)\,\mathrm{FPR}} \approx \frac{b}{ (1-b)}\cdot\frac{\mathrm{TPR}}{\mathrm{FPR}}$$ (eq:guardrail-precision-is-set-by-the-base-rate)

At $b = 0.003$, TPR $0.91$, FPR $0.04$: **6.4%**.

And the volume ratio, with the threshold that decides it:

$$R = \frac{(1-b)\,\mathrm{FPR}}{b\,\mathrm{TPR}} = \Theta(1/b), \qquad t^\star = t^\star\!\left(\frac{\kappa_H}{\kappa_R}\right)$$ (eq:refusals-outnumber-prevented-harms)

$R = 41$ at $b = 0.003$ and $249$ at $b = 0.0005$; $t^\star$ moves from **0.84** at 2:1 to
**0.02** at 2,560:1.

## 7. Internal Mechanics

Why is safety coverage so uneven across domains? Because safety data is written by people, and
people write in the languages and formats they use. A red-team exercise produces English
prose. A preference dataset produces English prose. A constitutional-AI critique produces
English prose. The model then generalises safety behaviour to other domains only as far as its
representations tie those domains together, which is further than nothing and much less than
everything.

That also explains why the vulnerable domains are the ones the model is *worse* at. A domain
the model handles fluently is one where the training distribution had a lot of it, which
correlates with safety data existing for it. A domain the model handles adequately but not
fluently — a mid-resource language, an unusual encoding — is one the model learned from a thin
slice, and the safety slice of that thin slice is thinner still.

**Capability and coverage are correlated, and the gap is largest in the middle of the
capability range.** Which is why the highest-success domains in
{{sec:9-practical-example}} are neither the strongest nor the weakest.

The lag has a mechanism worth being precise about, because it is not a resourcing problem.
Safety data for a capability cannot be collected before the capability exists — you cannot
red-team a model's ability to reason in a format it cannot yet handle. So the collection
starts after the capability appears, and takes time to specify, collect, review and train on.
That interval is $\lambda$, and it is bounded below by the data-production cycle rather than by
anyone's effort.

The guardrail's base-rate problem has a consequence that shapes deployments in a way rarely
stated. Because precision is low, the alarms cannot be reviewed by humans at volume — 1,790
alarms a day at 94% false is a review queue nobody staffs. So the guardrail is deployed in
*blocking* mode rather than review mode, which means the false positives are never seen by
anyone who could correct them. **Low precision forces automation, and automation removes the
feedback path that would improve precision.**

The segment concentration follows from what the classifier is measuring. A guardrail cannot
measure intent; it measures topical proximity to harm. Security research, clinical medicine,
law and creative writing are all topically proximate to harm by their nature, so their
requests sit near the decision boundary for reasons that have nothing to do with the user. This
is the same structure as {{ch:ev-classical-metrics}}'s subgroup calibration problem: an
aggregate false-positive rate that is acceptable overall is composed of segment rates that are
not.

Finally, the reason guardrails persist despite all of this. The three things they genuinely
buy — casual-misuse removal, cost-raising, and a logged decision — are each real and none is
the thing they are usually described as doing. In particular, the logged decision is an
accountability artefact: it demonstrates that the operator applied a control, which matters to
regulators, insurers and courts independently of whether the control was effective. That is a
legitimate reason to run one, and it is a different reason from the one on the architecture
diagram.

## 8. Implementation

The first listing measures the surface.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ic1}
"""Safety training covers a subset of the domains the model is capable in, and always will.

cite:wei2023jailbroken named two failure modes. **Competing objectives** is the model's
helpfulness training pulling against its safety training. **Mismatched generalization** is
safety training failing to reach a domain where capability exists -- base64, a low-resource
language, a cipher, a fictional frame.

The second is the structural one, because it is a statement about coverage rather than about
strength. Jailbreak surface is capability minus safety coverage, domain by domain
(eq:jailbreak-surface-is-capability-minus-safety-coverage).

And it does not shrink with effort, because capability grows into new domains faster than
safety data is collected for them
(eq:safety-coverage-lags-capability-by-construction).
"""
# (domain, capability level, safety-training coverage, share of attack attempts)
DOMAINS = [
    ("plain English request",     0.98, 0.94, 0.31),
    ("code and pseudocode",       0.93, 0.71, 0.14),
    ("role-play / fiction frame", 0.96, 0.77, 0.19),
    ("multi-turn build-up",       0.91, 0.52, 0.12),
    ("base64 and encodings",      0.74, 0.28, 0.08),
    ("low-resource language",     0.61, 0.19, 0.07),
    ("substitution cipher",       0.44, 0.09, 0.05),
    ("typographic / ASCII art",   0.38, 0.06, 0.04),
]

print("Capability and safety coverage, domain by domain.")
print()
print(f"{'domain':>28}{'capability':>13}{'safety coverage':>18}"
      f"{'gap':>8}{'attack success':>17}")
print("-" * 84)
succ = {}
for name, cap, saf, share in DOMAINS:
    s = cap * (1.0 - saf)
    succ[name] = (cap, saf, cap - saf, s, share)
    print(f"{name:>28}{cap:>13.2f}{saf:>18.2f}{cap - saf:>8.2f}{s:>17.2f}")

best = max(succ, key=lambda n: succ[n][3])
print()
print(f"highest single-domain success: {best} at {succ[best][3]:.2f}")
print(f"plain English, the domain safety training is built on: "
      f"{succ['plain English request'][3]:.2f}")

print()
print()
print("An attacker who tries every domain once.")
print()
miss = 1.0
for name, cap, saf, share in DOMAINS:
    miss *= (1.0 - succ[name][3])
print(f"P(at least one domain succeeds) = {1.0 - miss:.4f}")
print()
print(f"{'attacker tries':>34}{'success':>11}{'domains used':>15}")
print("-" * 60)
by_success = sorted(DOMAINS, key=lambda d: -succ[d[0]][3])
cum = 1.0
for i, (name, cap, saf, share) in enumerate(by_success, 1):
    cum *= (1.0 - succ[name][3])
    label = f"the top {i} domain" + ("s" if i > 1 else "")
    print(f"{label:>34}{1.0 - cum:>11.4f}{i:>15}")

top4 = 1.0
for name, cap, saf, share in by_success[:4]:
    top4 *= (1.0 - succ[name][3])
top4 = 1.0 - top4

print()
print()
print("Competing objectives: the same request, reframed to raise helpfulness")
print("pressure against a fixed safety penalty.")
print()
SAFETY_PENALTY = 1.00
FRAMES = [
    ("bare request",                     0.30),
    ("'for a research paper'",           0.62),
    ("'I am a medical professional'",    0.78),
    ("'write a story in which'",         0.91),
    ("'my grandmother used to tell me'", 0.97),
    ("'continue this document'",         1.06),
]
print(f"{'framing':>36}{'helpfulness pressure':>23}{'margin':>10}"
      f"{'complies':>11}")
print("-" * 80)
import math


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


frames = {}
for name, h in FRAMES:
    margin = h - SAFETY_PENALTY
    p = phi(margin / 0.22)
    frames[name] = (h, margin, p)
    print(f"{name:>36}{h:>23.2f}{margin:>10.2f}{p:>11.1%}")

print()
print("Nothing about the request changed. The reward for answering did.")

print()
print()
print("Now run generations. Capability enters new domains; safety data follows.")
print()
LAG_GENERATIONS = 2.0
NEW_DOMAINS_PER_GEN = 1.4
print(f"{'generation':>12}{'domains capable':>18}{'domains covered':>18}"
      f"{'uncovered':>12}{'surface':>10}")
print("-" * 70)
gen = {}
capable, covered = 8.0, 5.0
for g in range(0, 7):
    unc = capable - covered
    gen[g] = (capable, covered, unc)
    print(f"{g:>12}{capable:>18.1f}{covered:>18.1f}{unc:>12.1f}"
          f"{unc / capable:>10.1%}")
    capable += NEW_DOMAINS_PER_GEN
    covered += NEW_DOMAINS_PER_GEN if g >= LAG_GENERATIONS else 0.0

print()
print(f"the uncovered count is constant at {gen[6][2]:.1f} once the lag is")
print("steady: safety catches up at the same rate capability advances")

print()
print()
print("What closing a domain costs, and which ones to close first.")
print()
CLOSE = [
    ("plain English request",     0.94, 0.98, 0.5),
    ("role-play / fiction frame", 0.77, 0.93, 2.0),
    ("code and pseudocode",       0.71, 0.91, 2.5),
    ("multi-turn build-up",       0.52, 0.86, 6.0),
    ("base64 and encodings",      0.28, 0.88, 1.5),
    ("low-resource language",     0.19, 0.74, 8.0),
    ("substitution cipher",       0.09, 0.82, 1.2),
    ("typographic / ASCII art",   0.06, 0.79, 1.0),
]
print(f"{'domain':>28}{'safety now':>13}{'safety after':>15}"
      f"{'success removed':>18}{'effort':>9}{'per effort':>13}")
print("-" * 96)
close = {}
for name, now, after, eff in CLOSE:
    cap = succ[name][0]
    share = succ[name][4]
    removed = cap * ((1 - now) - (1 - after)) * share
    close[name] = (removed, eff, removed / eff)
    print(f"{name:>28}{now:>13.2f}{after:>15.2f}{removed:>18.4f}"
          f"{eff:>9.1f}{removed / eff:>13.4f}")

order = sorted(CLOSE, key=lambda c: -close[c[0]][2])
print()
print(f"best return: {order[0][0]} at {close[order[0][0]][2]:.4f} per unit")
print(f"worst:       {order[-1][0]} at {close[order[-1][0]][2]:.4f}")

print(f"""
The coverage table is cite:wei2023jailbroken's mismatched-generalization mode made
countable. In plain English -- the domain safety training is overwhelmingly built on --
capability is {succ['plain English request'][0]:.2f}, coverage is
{succ['plain English request'][1]:.2f}, and attack success is
{succ['plain English request'][3]:.2f}. In `{best}` capability is
{succ[best][0]:.2f}, coverage is {succ[best][1]:.2f}, and success is
{succ[best][3]:.2f} (eq:jailbreak-surface-is-capability-minus-safety-coverage).

**The model is nearly as capable in the second domain and the safety training barely reaches
it.** That is not a weakness of the safety training; it is a statement about where the
training data was.

The attacker table is what that means operationally. An attacker who tries the single best
domain succeeds {succ[best][3]:.1%} of the time; one who tries the top four succeeds
{top4:.1%}; one who tries all eight succeeds {1.0 - miss:.1%}.

**The defender must cover every domain and the attacker must find one.** That is the
oldest asymmetry in security, and here the domains are not enumerable in advance because they
are defined by what the model happens to be capable in.

The framing table is the other failure mode. Nothing about the request changes across those
rows -- only the reward for answering it. At a bare framing the model complies
{frames['bare request'][2]:.1%} of the time; framed as continuing a document,
{frames["'continue this document'"][2]:.1%}.

**Competing objectives is not a hole in the safety training, it is the helpfulness training
working.** Which is why it cannot be closed by more safety data alone: the same gradient that
makes the model useful is the one being exploited.

The generation table is the structural claim and it is the reason this problem does not
converge. Capability advances into roughly {NEW_DOMAINS_PER_GEN:.1f} new domains a
generation; safety data for a domain arrives about {LAG_GENERATIONS:.0f} generations after
capability does. Once the lag is steady, **the uncovered count is constant at
{gen[6][2]:.1f}** (eq:safety-coverage-lags-capability-by-construction) and the surface
converges to {gen[6][2] / gen[6][0]:.1%} rather than to zero.

Safety work is not failing in that picture. It is succeeding at exactly the rate capability
advances, which leaves a standing gap of fixed size and moving contents.

The closing table says where to spend anyway, and the ranking is not by gap size.
`{order[0][0]}` returns {close[order[0][0]][2]:.4f} of removed success per unit of effort;
`{order[-1][0]}` returns {close[order[-1][0]][2]:.4f}. The difference is
**attack volume**: closing a domain nobody uses removes a large gap and little exposure.

cite:zou2023universal is the reason to be modest about all of this. An adversarial suffix
found by optimisation on open models transferred to three closed commercial systems, which
means the domain list above is not a list of things humans thought of -- it is a search space,
and the search can be automated against a proxy. **A domain-by-domain defence assumes a
domain-by-domain attacker**, and that assumption expired in 2023.

Which leaves guardrails as the layer that runs at request time regardless of domain, and
ch:sec-jailbreaks' second listing takes up what they can and cannot bound.""")
```

## 9. Practical Example

Capability against safety coverage:

```
                      domain   capability   safety coverage     gap   attack success
------------------------------------------------------------------------------------
       plain English request         0.98              0.94    0.04             0.06
         code and pseudocode         0.93              0.71    0.22             0.27
   role-play / fiction frame         0.96              0.77    0.19             0.22
         multi-turn build-up         0.91              0.52    0.39             0.44
        base64 and encodings         0.74              0.28    0.46             0.53
       low-resource language         0.61              0.19    0.42             0.49
         substitution cipher         0.44              0.09    0.35             0.40
     typographic / ASCII art         0.38              0.06    0.32             0.36
```

**The model is less capable in the high-success domains and much more vulnerable**, because
the gap is what matters ({{eq:jailbreak-surface-is-capability-minus-safety-coverage}}).

```
                    attacker tries    success   domains used
------------------------------------------------------------
                  the top 1 domain     0.5328              1
                 the top 2 domains     0.7636              2
                 the top 4 domains     0.9202              4
                 the top 8 domains     0.9725              8
```

**The defender must cover every domain; the attacker must find one.**

```
                             framing   helpfulness pressure    margin   complies
--------------------------------------------------------------------------------
                        bare request                   0.30     -0.70       0.1%
              'for a research paper'                   0.62     -0.38       4.2%
            'write a story in which'                   0.91     -0.09      34.1%
    'my grandmother used to tell me'                   0.97     -0.03      44.6%
            'continue this document'                   1.06      0.06      60.7%
```

Nothing about the request changed — only the reward for answering. **Competing objectives is
the helpfulness training working.**

```
  generation   domains capable   domains covered   uncovered   surface
----------------------------------------------------------------------
           0               8.0               5.0         3.0     37.5%
           2              10.8               5.0         5.8     53.7%
           4              13.6               7.8         5.8     42.6%
           6              16.4              10.6         5.8     35.4%
```

Once the lag is steady, **the uncovered count is constant at 5.8**
({{eq:safety-coverage-lags-capability-by-construction}}) — safety succeeding at exactly the
rate capability advances.

The second listing takes up the request-time layer.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ic2}
"""A guardrail is a classifier on a rare event, and rare events destroy precision.

Genuinely harmful requests are a small fraction of traffic. A guardrail with excellent
sensitivity and a small false-positive rate still produces alarms that are mostly wrong,
because the false positives are drawn from a population hundreds of times larger
(eq:guardrail-precision-is-set-by-the-base-rate).

Which flips the volumes. At a low base rate the guardrail refuses far more legitimate users
than it prevents harms, and by how much is arithmetic
(eq:refusals-outnumber-prevented-harms).

This listing computes both, finds the cost-optimal threshold rather than the
statistically-appealing one, and then applies ch:sec-threat-model's adaptive-attacker result
to say what a guardrail is actually for.
"""
import math

TPR = 0.91
FPR = 0.04
REQUESTS_PER_DAY = 42_000.0


def precision(base, tpr=TPR, fpr=FPR):
    tp = base * tpr
    fp = (1 - base) * fpr
    return tp / (tp + fp) if tp + fp > 0 else 0.0


print(f"A guardrail at TPR {TPR:.2f}, FPR {FPR:.2f}. Precision against base rate.")
print()
print(f"{'base rate':>12}{'harmful/day':>14}{'alarms/day':>13}"
      f"{'precision':>12}{'false alarms/day':>19}")
print("-" * 70)
prec = {}
for b in (0.20, 0.05, 0.01, 0.003, 0.0005):
    p = precision(b)
    alarms = REQUESTS_PER_DAY * (b * TPR + (1 - b) * FPR)
    prec[b] = (p, alarms)
    print(f"{b:>12.2%}{REQUESTS_PER_DAY * b:>14,.0f}{alarms:>13,.0f}"
          f"{p:>12.1%}{alarms * (1 - p):>19,.0f}")

BASE = 0.003
print()
print(f"at a {BASE:.1%} base rate, {1 - prec[BASE][0]:.0%} of alarms are wrong")
print(f"and {prec[BASE][1] * (1 - prec[BASE][0]):,.0f} legitimate users are")
print("refused every day")

print()
print()
print("Sweeping the threshold. Both error types have a price.")
print()
HARM_COST = 4200.0
REFUSAL_COST = 26.0


def rates(t):
    """Higher threshold: fewer alarms, both rates fall."""
    tpr = 1.0 / (1.0 + math.exp((t - 0.30) / 0.16))
    fpr = 1.0 / (1.0 + math.exp((t - 0.02) / 0.09))
    return tpr, fpr


print(f"{'threshold':>11}{'TPR':>8}{'FPR':>9}{'precision':>12}"
      f"{'harm cost/day':>16}{'refusal cost/day':>19}{'total':>12}")
print("-" * 87)
sweep = {}
for t in (0.05, 0.15, 0.25, 0.35, 0.50, 0.70):
    tpr, fpr = rates(t)
    harm = REQUESTS_PER_DAY * BASE * (1 - tpr) * HARM_COST
    refuse = REQUESTS_PER_DAY * (1 - BASE) * fpr * REFUSAL_COST
    sweep[t] = (tpr, fpr, precision(BASE, tpr, fpr), harm, refuse, harm + refuse)
    print(f"{t:>11.2f}{tpr:>8.2f}{fpr:>9.3f}{precision(BASE, tpr, fpr):>12.1%}"
          f"{harm:>16,.0f}{refuse:>19,.0f}{harm + refuse:>12,.0f}")

best_t = min(sweep, key=lambda t: sweep[t][5])
print()
print(f"cost-optimal threshold: {best_t:.2f} at {sweep[best_t][5]:,.0f} a day")
print(f"maximum-sensitivity threshold ({0.05:.2f}) costs "
      f"{sweep[0.05][5]:,.0f}")

print()
print()
print("Volumes at the cost-optimal threshold: refusals against prevented harms.")
print()
print(f"{'base rate':>12}{'prevented harms/day':>22}{'refusals/day':>16}"
      f"{'refusals per prevented harm':>30}")
print("-" * 80)
tpr_o, fpr_o = rates(best_t)
vol = {}
for b in (0.20, 0.05, 0.01, 0.003, 0.0005):
    prevented = REQUESTS_PER_DAY * b * tpr_o
    refused = REQUESTS_PER_DAY * (1 - b) * fpr_o
    vol[b] = (prevented, refused, refused / prevented)
    print(f"{b:>12.2%}{prevented:>22,.0f}{refused:>16,.0f}"
          f"{refused / prevented:>30,.0f}")

print()
print(f"at {BASE:.1%} the guardrail refuses {vol[BASE][2]:,.0f} legitimate users")
print("for every harmful request it prevents")

print()
print()
print("Which threshold is right depends on a cost ratio nobody writes down.")
print()
print(f"{'harm : refusal cost':>21}{'best threshold':>17}{'TPR there':>12}"
      f"{'refusals/day':>15}{'prevented/day':>16}")
print("-" * 81)
ratio_tab = {}
for ratio in (2.0, 10.0, 40.0, 160.0, 640.0, 2560.0):
    best, bestc = None, None
    for th in [0.02 * i for i in range(1, 50)]:
        tpr, fpr = rates(th)
        c = (REQUESTS_PER_DAY * BASE * (1 - tpr) * ratio
             + REQUESTS_PER_DAY * (1 - BASE) * fpr * 1.0)
        if bestc is None or c < bestc:
            best, bestc = th, c
    tpr, fpr = rates(best)
    ratio_tab[ratio] = (best, tpr, fpr)
    print(f"{ratio:>18,.0f}:1{best:>17.2f}{tpr:>12.2f}"
          f"{REQUESTS_PER_DAY * (1 - BASE) * fpr:>15,.0f}"
          f"{REQUESTS_PER_DAY * BASE * tpr:>16,.0f}")

print()
print(f"the threshold moves from {ratio_tab[2.0][0]:.2f} to "
      f"{ratio_tab[2560.0][0]:.2f} across that range")
print("and the ratio is the number that is never stated")

print()
print()
print("And what an adaptive attacker does to the sensitivity term.")
print()
print(f"{'attempts':>10}{'P(all blocked)':>17}{'P(one gets through)':>22}"
      f"{'effective TPR':>16}")
print("-" * 65)
t_star = best_t
tpr_star, _ = rates(t_star)
adapt = {}
for k in (1, 3, 10, 30, 100):
    blocked = tpr_star ** k
    adapt[k] = (blocked, 1 - blocked, blocked)
    print(f"{k:>10}{blocked:>17.4f}{1 - blocked:>22.4f}{blocked:>16.4f}")

print()
print(f"at the cost-optimal threshold the guardrail's effective TPR against")
print(f"{30} attempts is {adapt[30][0]:.4f}")

print()
print()
print("So what does it buy? Three things, and only one of them is blocking.")
print()
BUYS = [
    ("blocks a one-shot attempt",   f"{tpr_star:.0%}",           "real, and single-shot"),
    ("blocks a 30-attempt attacker", f"{adapt[30][0]:.1%}",      "essentially nothing"),
    ("raises attempts per success", f"{1 / max(1 - tpr_star, 1e-9):.1f}x", "a rate limiter"),
    ("produces a logged decision",  "every request",             "forensics, liability"),
    ("removes casual misuse",       f"{tpr_star:.0%}",           "most traffic is casual"),
]
print(f"{'what a guardrail buys':>32}{'value':>16}{'reading':>28}")
print("-" * 76)
for name, val, reading in BUYS:
    print(f"{name:>32}{val:>16}{reading:>28}")

print()
print()
print("Refusal cost is not uniform. Where the false positives land.")
print()
SEGMENTS = [
    ("general consumer queries",   0.62, 1.0),
    ("security research",          0.04, 8.4),
    ("medical and clinical",       0.07, 6.1),
    ("legal and compliance",       0.05, 5.2),
    ("creative writing",           0.13, 3.7),
    ("non-English",                0.09, 4.9),
]
tpr_b, fpr_b = rates(best_t)
print(f"{'segment':>28}{'share of traffic':>19}{'relative FPR':>15}"
      f"{'refusals/day':>15}{'share of refusals':>20}")
print("-" * 97)
tot_ref = sum(REQUESTS_PER_DAY * sh * fpr_b * rel for n, sh, rel in SEGMENTS)
for name, sh, rel in SEGMENTS:
    r = REQUESTS_PER_DAY * sh * fpr_b * rel
    print(f"{name:>28}{sh:>19.0%}{rel:>15.1f}x{r:>14,.0f}"
          f"{r / tot_ref:>20.1%}")

print(f"""
The precision table is the arithmetic every guardrail runs into. At a {BASE:.1%} base rate --
generous, for a consumer product -- a guardrail with TPR {TPR:.2f} and FPR {FPR:.2f} produces
{prec[BASE][1]:,.0f} alarms a day of which **{1 - prec[BASE][0]:.0%} are wrong**
(eq:guardrail-precision-is-set-by-the-base-rate).

Nothing is broken. The false positives are drawn from a population
{(1 - BASE) / BASE:.0f} times larger than the true positives, so a
{FPR:.0%} false-positive rate outnumbers a {TPR:.0%} true-positive rate by a wide margin.
This is the same base-rate arithmetic that made accuracy useless in
ch:ev-classical-metrics, arriving in a setting where the consequence is a refused user rather
than a bad metric.

The threshold sweep prices both errors. At maximum sensitivity ({0.05:.2f}) the daily cost is
{sweep[0.05][5]:,.0f}, of which {sweep[0.05][4] / sweep[0.05][5]:.0%} is refusals; the
cost-optimal threshold is {best_t:.2f} at {sweep[best_t][5]:,.0f}.

**The default guardrail configuration is the maximum-sensitivity one**, because that is what
"catch as much as possible" means, and it is
{sweep[0.05][5] / sweep[best_t][5]:.1f} times the cost of the threshold that takes both errors
seriously.

The volume table is the number to sit with. At the cost-optimal threshold and a
{BASE:.1%} base rate, the guardrail prevents {vol[BASE][0]:,.0f} harmful requests a day and
refuses {vol[BASE][1]:,.0f} legitimate ones -- **{vol[BASE][2]:,.0f} refusals per prevented
harm** (eq:refusals-outnumber-prevented-harms). At {0.20:.0%} it is
{vol[0.20][2]:,.0f} to one.

Whether that trade is worth making is a question about the *ratio* of the two unit costs, and
the ratio table shows how much rides on it: the optimal threshold moves from
{ratio_tab[2.0][0]:.2f} at 2:1 to {ratio_tab[2560.0][0]:.2f} at 2,560:1, taking daily refusals
from a number in the tens of thousands to one in the hundreds.

**That ratio is never written down**, which is ch:ev-classical-metrics' finding arriving in a
setting where the cost of the assumed ratio is paid by users rather than by a metric.

The adaptive table is the harder problem and it comes from ch:sec-threat-model. At the
cost-optimal threshold the guardrail blocks {tpr_star:.0%} of a single attempt and
{adapt[30][0]:.1%} of a thirty-attempt attacker. The effective TPR against a determined
adversary is **not the number on the datasheet**, and no threshold choice fixes that -- moving
the threshold up trades refusals for a sensitivity that repetition erases anyway.

Which brings the question the chapter has to answer honestly. If a guardrail cannot stop a
determined attacker, what is it for?

The `buys` table is the answer and it has three real entries. It **removes casual misuse**,
which is most of the traffic that would otherwise reach the model -- people trying something
once because they wondered. It **raises attempts per success** to
{1 / max(1 - tpr_star, 1e-9):.1f}x, which is a rate limiter on the attacker's search and
composes with actual rate limits. And it **produces a logged decision on every request**,
which is the forensic and accountability artefact, and is the entry most often left out of
technical discussions and most often the reason the guardrail was funded.

What it does not do is bound anything. **A guardrail is a cost-raiser and a record, not a
boundary** -- exactly ch:sec-threat-model's classification, now with the numbers.

The segment table is the part to take to a product review, because refusal cost is not spread
evenly. `security research` is {SEGMENTS[1][1]:.0%} of traffic and
{SEGMENTS[1][2]:.1f} times more likely to trip the classifier; `medical and clinical` is
{SEGMENTS[2][1]:.0%} at {SEGMENTS[2][2]:.1f} times.

**The false positives land on the users with the most legitimate need for the capability**,
because proximity to a sensitive topic is what the classifier is measuring. That is not a
tuning problem and it does not improve with a better model -- it is what the feature is. The
remedy is segment-aware thresholds and an appeal path, and both are product work rather than
security work.""")
```

```
   base rate   harmful/day   alarms/day   precision   false alarms/day
----------------------------------------------------------------------
      20.00%         8,400        8,988       85.0%              1,344
       1.00%           420        2,045       18.7%              1,663
       0.30%           126        1,790        6.4%              1,675
       0.05%            21        1,698        1.1%              1,679
```

At a 0.3% base rate, **94% of alarms are wrong**
({{eq:guardrail-precision-is-set-by-the-base-rate}}) — the false positives are drawn from a
population 332 times larger.

```
  threshold     TPR      FPR   precision   harm cost/day   refusal cost/day       total
---------------------------------------------------------------------------------------
       0.05    0.83    0.417        0.6%          91,704            454,466     546,170
       0.25    0.58    0.072        2.4%         223,589             78,447     302,037
       0.50    0.22    0.005       12.2%         411,347              5,231     416,578
```

```
   base rate   prevented harms/day    refusals/day   refusals per prevented harm
--------------------------------------------------------------------------------
      20.00%                 4,851           2,421                             0
       1.00%                   243           2,996                            12
       0.30%                    73           3,017                            41
       0.05%                    12           3,025                           249
```

**41 refusals per prevented harm** at a 0.3% base rate
({{eq:refusals-outnumber-prevented-harms}}), 249 at 0.05%.

```
  harm : refusal cost   best threshold   TPR there   refusals/day   prevented/day
---------------------------------------------------------------------------------
                 2:1             0.84        0.03              5               4
                40:1             0.40        0.35            605              44
               640:1             0.12        0.75         10,371              95
             2,560:1             0.02        0.85         20,937             107
```

The threshold moves from **0.84 to 0.02** across that range, and **the ratio is never
stated** — {{eq:f1-asserts-a-cost-ratio}} in a setting where users pay for the assumption.

```
  attempts   P(all blocked)   P(one gets through)   effective TPR
-----------------------------------------------------------------
         1           0.5775                0.4225          0.5775
         3           0.1926                0.8074          0.1926
        30           0.0000                1.0000          0.0000

           what a guardrail buys           value                     reading
----------------------------------------------------------------------------
       blocks a one-shot attempt             58%       real, and single-shot
    blocks a 30-attempt attacker            0.0%         essentially nothing
     raises attempts per success            2.4x              a rate limiter
        produces a logged decision   every request        forensics, liability
           removes casual misuse             58%      most traffic is casual
```

**A guardrail is a cost-raiser and a record, not a boundary.**

## 10. Production Considerations

Measure your safety coverage by domain, not in aggregate. The aggregate is dominated by the
domain the data was written in.

Expect a standing gap and staff for rotation rather than closure. The uncovered contents
change every generation; the count does not.

Compute your guardrail's precision at your actual base rate before deciding how to route
alarms. At 6% precision, a human review queue is not viable and blocking mode is forced.

State the harm-to-refusal cost ratio explicitly. It moves the optimal threshold by a factor of
forty and it currently defaults to whatever the vendor shipped.

Report refusals per prevented harm alongside the block rate. It is the number a product owner
needs and the one a security dashboard omits.

Measure false-positive rate by segment. The users most affected are the ones with the most
legitimate need.

Build an appeal path. Low precision plus blocking mode means the errors are never seen by
anyone who could correct them.

## 11. Common Mistakes

**Treating jailbreaks as a patch queue.** The uncovered domain count is constant; the contents
rotate.

**Expecting more safety data to close competing objectives.** That gradient is the helpfulness
training.

**Reading a guardrail's TPR as its effective block rate.** Against 30 attempts it is 0.0%.

**Deploying at maximum sensitivity.** It costs 1.8× the cost-optimal threshold, almost all in
refusals.

**Reporting alarms without precision.** 1,790 alarms a day at 94% false is not a signal.

**Assuming refusals are spread evenly.** They concentrate on security, medical, legal and
non-English users.

## 12. Failure Modes

**Domain-by-domain defence against an automated search.**
{{cite:zou2023universal}}'s suffixes were found by optimisation, not by enumeration.

**Guardrail tuned on English, deployed globally.** The non-English segment has a 4.9× relative
false-positive rate and nobody measured it.

**Review queue abandoned.** Precision was 6%, the queue was unstaffable, and the guardrail
silently became a blocker.

**Safety penalty raised until benign refusals spiked.** Competing objectives was treated as a
tuning problem and the tuning knob was the refusal rate.

**Compliance artefact mistaken for a control.** The logged decision satisfied an auditor and
the architecture diagram now shows a boundary that does not exist.

**Cost ratio inherited from a vendor default.** The threshold encodes somebody else's product
and somebody else's users.

## 13. Alternatives

**Output-side guardrails.** Classify the response rather than the request. Higher precision —
you can see what was actually produced — and it costs a generation before you find out.

**Refusal with a reason and an appeal.** Keeps the block and returns the false positive to a
human who can correct it. The single highest-value addition on this list and it is product
work.

**Capability restriction by account tier.** Verified security researchers get a different
threshold. Addresses the segment concentration directly and requires an identity programme.

**Constitutional or rule-based self-critique.** The model reviews its own draft against stated
principles. Cheaper than a separate classifier and subject to the same competing-objectives
pressure.

**Accept the standing gap and invest in monitoring.** Detect successful jailbreaks after the
fact and measure their real-world consequence. Honest, unpopular, and the only approach whose
cost does not scale with the refusal rate.

## 14. Evaluation

Measure attack success by domain, not overall. The overall number is an average over a
distribution the attacker chooses.

Track your uncovered-domain count across model versions. If it is constant, the process is
working as designed.

Publish your guardrail's precision at your production base rate, next to its TPR. The two
together are a measurement; either alone is not.

Report refusals per prevented harm every quarter and put it in front of a product owner.

Break false-positive rate out by segment and by language, and set thresholds per segment if
the spread justifies it.

Test the guardrail against a small automated search rather than a fixed attack set. The fixed
set measures the wrong column.

## 15. Advanced Concepts

The domain-independence assumed in the attacker table is optimistic for the defender in one
direction and pessimistic in another. Safety training in one domain does transfer partially to
adjacent ones — coverage in English prose raises coverage in code more than zero — so the
domains are not independent and closing one raises several. But an attacker combining domains
(a cipher inside a fictional frame delivered over multiple turns) faces a coverage that is the
*product* of thin coverages rather than the maximum, so composition helps the attacker more
than transfer helps the defender. **The realistic surface is larger than the table, not
smaller.**

The lag model treats $\lambda$ as constant, and there is reason to think it is not. As
capability accelerates, the interval between a capability appearing and being red-teamed
should grow, because the specification step — deciding what harmful use of a new capability
looks like — is human and does not accelerate. If $\lambda$ grows with $\alpha$, the uncovered
count $\alpha\lambda$ grows quadratically rather than staying flat, and the surface fraction
stops declining. This is checkable from release histories and, as far as this book found,
unpublished.

The guardrail analysis assumes harm is binary and per-request, which is the framing that makes
the arithmetic tractable and is wrong in a specific way. Real harm is often distributed across
a session — no single request is harmful and the sequence is — and a per-request classifier
cannot see that by construction. Session-level classification raises the base rate (sessions
are rarer than requests and a larger fraction of them are problematic), which improves
precision, and it costs latency and statefulness. **Raising the base rate is the only lever
that improves precision without improving the classifier**, and moving the unit of analysis is
the way to do it.

Finally, a point about where this chapter's economics break down. Everything here assumes the
harm from a successful jailbreak is bounded and priceable. For the great majority of
categories that is true — the information is available elsewhere and the marginal harm is
small, which is why the honest $\kappa_H$ is lower than the discourse suggests. For a small
number of categories it is not, and there the correct threshold is the one that maximises
sensitivity regardless of refusal cost. **A single guardrail cannot serve both regimes**, and
running one at a compromise threshold serves neither. Category-specific thresholds are the
resolution and they require the category taxonomy that
{{cite:ji2023survey}}-style work provides.

## 16. Connection to Previous Chapters

{{eq:detection-layers-fail-against-an-adaptive-attacker}} from {{ch:sec-threat-model}} is
applied directly here: a guardrail's 58% single-attempt block rate is 0.0% against thirty
attempts, which is why the honest description is cost-raiser rather than boundary.

{{eq:f1-asserts-a-cost-ratio}} and {{eq:threshold-is-the-decision-not-the-model}} from
{{ch:ev-classical-metrics}} are the guardrail's second half in general form, with the
threshold moving 40× across a ratio nobody states.

{{eq:instructions-and-data-share-a-channel}} from {{ch:sec-threat-model}} is why a fictional
frame works: there is no field in which "this is fiction" can be marked as data.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} explains the guardrail's
segment concentration — it measures topical proximity because intent is not observable.

## 17. Exercises

1. Estimate capability and safety coverage for five domains in your own deployment. Where is
   the largest gap, and where is the largest gap times attack volume?

2. Compute your guardrail's precision at your production base rate. Is a human review queue
   viable at that precision?

3. Compute refusals per prevented harm. Show the number to a product owner and record the
   reaction.

4. Find the harm-to-refusal cost ratio implied by your current threshold, by locating the
   ratio at which it is optimal.

5. Model composed attacks — a cipher inside a fictional frame — and recompute the surface under
   {{sec:15-advanced-concepts}}'s multiplicative-coverage assumption.

## 18. Interview Questions

1. Why do jailbreaks work better in base64 than in English?

2. Why can competing objectives not be fixed with more safety training?

3. Our guardrail has a 91% true-positive rate. How many of its alarms are correct?

4. How many legitimate users do we refuse for each harmful request we prevent?

5. What threshold should the guardrail run at, and what do you need to know to answer?

6. Our guardrail blocks 94% of known jailbreaks. What does that number not tell you?

## 19. Research Questions

1. Does the safety-data lag $\lambda$ grow with the rate of capability advance, and what do
   release histories show?

2. How much does safety coverage transfer between domains, and does composition defeat the
   transfer?

3. What are realistic base rates for genuinely harmful requests in deployed consumer and
   enterprise products?

4. How much does session-level rather than request-level classification raise precision, and
   at what latency cost?

## 20. Chapter Summary

Jailbreaking has two mechanisms and one of them does not converge.

**Competing objectives** is the helpfulness training working against the safety training:
compliance rises from **0.1%** at a bare framing to **60.7%** framed as continuing a document,
with the request unchanged. More safety data does not close it, because the gradient being
exploited is the one that makes the model useful.

**Mismatched generalization** is coverage. Attack success is capability minus safety coverage
per domain ({{eq:jailbreak-surface-is-capability-minus-safety-coverage}}): **0.06** in plain
English, **0.53** in base64 — the model is *less* capable there and much more vulnerable,
because the gap is what matters. An attacker trying the best domain succeeds **53.3%** of the
time; trying all eight, **97.3%**.

And the gap is structural. Capability enters ~1.4 new domains per generation, safety data lags
by ~2, so the uncovered count settles at **5.8** and stays
({{eq:safety-coverage-lags-capability-by-construction}}). Safety work succeeding exactly as
fast as capability advances still leaves a standing gap.

Guardrails run at request time and inherit a base-rate problem: **94% of alarms are wrong** at
a 0.3% base rate ({{eq:guardrail-precision-is-set-by-the-base-rate}}), and the cost-optimal
threshold refuses **41 legitimate users per prevented harm**
({{eq:refusals-outnumber-prevented-harms}}) — 249 at a 0.05% base rate. The threshold that
decides this moves from **0.84 to 0.02** across a plausible cost-ratio range, and the ratio is
never stated.

Against a thirty-attempt attacker the guardrail blocks **0.0%**. What it genuinely buys is
casual-misuse removal, a **2.4×** rise in attempts per success, and a logged decision — the
last of which is an accountability artefact and is often the real reason it exists.

The synthesis is that this is the one area in the book where the honest answer is a standing
cost rather than a fix. The domains rotate, the framings multiply, the guardrail refuses the
wrong people, and every control is a cost-raiser. What a team can actually do is measure the
gap by domain, state the cost ratio, watch the segments, and build the appeal path — none of
which closes anything, and all of which is better than a diagram with a boundary drawn where
there is none.

Carry forward: **surface is capability minus coverage**, and **a guardrail refuses far more
people than it stops**.

## 21. Further Reading

- {{cite:wei2023jailbroken}} — competing objectives and mismatched generalization, the two
  mechanisms this chapter is built on.
- {{cite:zou2023universal}} — automated search for transferable attacks, and why a
  domain-by-domain defence assumes the wrong attacker.
- {{cite:perez2022ignore}} — the framing techniques whose helpfulness pressure
  {{sec:9-practical-example}} prices.
- {{cite:ji2023survey}} — a taxonomy of what "harmful output" covers, which is the category
  structure {{sec:15-advanced-concepts}}'s per-category thresholds need.
