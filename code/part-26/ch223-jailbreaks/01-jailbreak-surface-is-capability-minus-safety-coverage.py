# -*- coding: utf-8 -*-
# Extracted from: Chapter 223 — Jailbreaking and Guardrails
# Source: src/.../ch223-jailbreaks.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
