# -*- coding: utf-8 -*-
# Extracted from: Chapter 222 — Prompt Injection and Indirect Prompt Injection
# Source: src/.../ch222-prompt-injection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Goal hijacking and prompt leaking are one attack with two blast radii.

cite:perez2022ignore separated them and the separation is the useful part. A leak exfiltrates
what is already in the context; a hijack makes the system do something. The first is bounded
by what you put in the prompt; the second is bounded by what you connected
(eq:leaking-is-bounded-by-context-hijacking-is-not).

They also have different defences, and the asymmetry runs against intuition: output filtering
works well on leaks -- you know what a secret looks like -- and badly on hijacks, because a
hijacked action looks like a legitimate action.

Sanitisation is the other lever, and it has a hard limit: you can only clean content that
passes through a pipeline you own
(eq:sanitisation-covers-only-what-you-own).
"""
# (stage, share of injections it could remove, cost, share of untrusted volume it sees)
STAGES = [
    ("crawl / partner feed ingest", 0.71, 1.0, 0.38),
    ("indexing and chunking",       0.64, 0.8, 0.38),
    ("retrieval, before assembly",  0.52, 1.2, 1.00),
    ("prompt assembly",             0.44, 0.6, 1.00),
    ("generation, in-context rules", 0.31, 0.3, 1.00),
    ("tool call, before execution", 0.79, 2.0, 1.00),
    ("output, before display",      0.36, 0.9, 1.00),
]

print("Where in the pipeline an injection could be removed, and what each")
print("stage can actually see.")
print()
print(f"{'stage':>30}{'removable':>12}{'cost':>8}"
      f"{'sees this share of untrusted':>31}{'effective':>12}")
print("-" * 93)
eff = {}
for name, rem, cost, cov in STAGES:
    e = rem * cov
    eff[name] = (rem, cost, cov, e, e / cost)
    print(f"{name:>30}{rem:>12.0%}{cost:>8.1f}{cov:>31.0%}{e:>12.0%}")

print()
print(f"the two ingest stages see only {STAGES[0][3]:.0%} of untrusted content")
print("because the rest arrives at query time from sources you do not own")

print()
print()
print("Ranked by effective removal per unit of cost.")
print()
order = sorted(STAGES, key=lambda s: -(s[1] * s[3] / s[2]))
print(f"{'rank':>6}{'stage':>30}{'effective':>12}{'cost':>8}{'per cost':>11}")
print("-" * 67)
for i, (name, rem, cost, cov) in enumerate(order, 1):
    print(f"{i:>6}{name:>30}{rem * cov:>12.0%}{cost:>8.1f}"
          f"{rem * cov / cost:>11.2f}")

print()
print()
print("The two outcomes, and what bounds each.")
print()
CONTEXT_SECRETS = [
    ("the system prompt",        1.0),
    ("tool schemas and names",   0.8),
    ("retrieved passages",       4.0),
    ("prior turns",              2.5),
    ("a session token, if present", 9.0),
]
SINKS = [
    ("send an email",        6.0),
    ("write to the CRM",     8.0),
    ("issue a refund",       9.5),
    ("execute code",        10.0),
    ("read a secret store",  9.0),
]
leak_bound = sum(v for n, v in CONTEXT_SECRETS)
hijack_bound = sum(v for n, v in SINKS)
print(f"{'outcome':>20}{'bounded by':>28}{'items':>8}{'total damage':>15}"
      f"{'grows with':>24}")
print("-" * 95)
print(f"{'prompt leaking':>20}{'what is in the context':>28}"
      f"{len(CONTEXT_SECRETS):>8}{leak_bound:>15.1f}{'prompt size':>24}")
print(f"{'goal hijacking':>20}{'what the agent can call':>28}"
      f"{len(SINKS):>8}{hijack_bound:>15.1f}{'integration count':>24}")

print()
print(f"leaking is capped at {leak_bound:.1f} and you choose the cap")
print(f"hijacking is capped at {hijack_bound:.1f} and it rises every quarter")

print()
print()
print("Why output filtering is asymmetric between them.")
print()
DEFENCES = [
    ("output scan for known secrets", 0.88, 0.04, "you know the string"),
    ("output scan for exfil patterns", 0.61, 0.09, "URLs, base64, markdown images"),
    ("output scan for 'wrong action'", 0.12, 0.31, "the action looks legitimate"),
    ("tool-call schema validation",   0.09, 0.19, "the arguments are well-formed"),
    ("tool-call allow-list",          0.00, 0.79, "structural, not detection"),
    ("human approval on the sink",    0.00, 0.88, "structural, not detection"),
]
print(f"{'defence':>34}{'catches a leak':>17}{'catches a hijack':>19}"
      f"{'why':>32}")
print("-" * 102)
defn = {}
for name, leak, hij, why in DEFENCES:
    defn[name] = (leak, hij)
    print(f"{name:>34}{leak:>17.0%}{hij:>19.0%}{why:>32}")

print()
print("The first two rows are good at leaks and useless at hijacks. The last")
print("two are the reverse, and they are not detectors.")

print()
print()
print("Residual damage under three postures.")
print()
POSTURES = [
    ("nothing",                       [],                                    0.00),
    ("output scanning only",          ["output scan for known secrets",
                                       "output scan for exfil patterns"],    0.13),
    ("allow-list only",               ["tool-call allow-list"],              0.18),
    ("output scanning + allow-list",  ["output scan for known secrets",
                                       "output scan for exfil patterns",
                                       "tool-call allow-list"],              0.29),
    ("everything",                    [d[0] for d in DEFENCES],              0.63),
]
print(f"{'posture':>32}{'leak residual':>16}{'hijack residual':>18}"
      f"{'total':>10}{'utility cost':>15}")
print("-" * 91)
post = {}
for label, ds, util in POSTURES:
    l, h = 1.0, 1.0
    for d in ds:
        l *= (1.0 - defn[d][0])
        h *= (1.0 - defn[d][1])
    lr, hr = leak_bound * l, hijack_bound * h
    post[label] = (lr, hr, lr + hr, util)
    print(f"{label:>32}{lr:>16.2f}{hr:>18.2f}{lr + hr:>10.2f}{util:>15.0%}")

print()
print(f"output scanning alone: leak residual {post['output scanning only'][0]:.2f}, "
      f"hijack residual {post['output scanning only'][1]:.2f}")

print()
print()
print("And the design lever nobody prices: what you put in the context.")
print()
print(f"{'context contains':>34}{'leak bound':>13}{'residual under scanning':>26}"
      f"{'utility cost':>15}")
print("-" * 88)
TRIMS = [
    ("everything, including a token", leak_bound,               0.00),
    ("no session token",              leak_bound - 9.0,          0.04),
    ("no session token, no schemas",  leak_bound - 9.0 - 0.8,    0.11),
    ("retrieved passages only",       4.0,                       0.22),
]
for name, bound, util in TRIMS:
    l = 1.0
    for d in ("output scan for known secrets", "output scan for exfil patterns"):
        l *= (1.0 - defn[d][0])
    print(f"{name:>34}{bound:>13.1f}{bound * l:>26.3f}{util:>15.0%}")

print(f"""
The stage table is where a defence can live, and the fourth column is the constraint. The two
ingest stages are the cheapest place to strip injected instructions and they see
{STAGES[0][3]:.0%} of untrusted content, because the rest arrives at query time from a
partner feed, a user upload, a live web fetch or another agent
(eq:sanitisation-covers-only-what-you-own).

**Sanitisation scales with ownership of the pipeline, not with effort.** A team that ingests
everything can clean everything; a team that retrieves from sources it does not control
cannot, and no amount of scanning at ingest changes that.

The ranking makes the practical order visible. `{order[0][0]}` returns
{order[0][1] * order[0][3] / order[0][2]:.2f} of effective removal per unit of cost and
`{order[-1][0]}` returns {order[-1][1] * order[-1][3] / order[-1][2]:.2f}. Note where the
tool-call stage sits: it removes {STAGES[5][1]:.0%} and is expensive, and it is the only stage
that sees the *consequence* rather than the text.

The outcome table is cite:perez2022ignore's distinction converted into two different design
problems. Leaking is bounded by what is in the context -- {len(CONTEXT_SECRETS)} items worth
{leak_bound:.1f} here -- and that bound is **a choice you make when you assemble the prompt**.
Hijacking is bounded by what the agent can call: {len(SINKS)} sinks worth
{hijack_bound:.1f}, and that bound **rises every time somebody ships an integration**
(eq:leaking-is-bounded-by-context-hijacking-is-not).

One of those is a design parameter and the other is a product roadmap, which is why they
diverge over time even in teams that take both seriously.

The defence table is the asymmetry, and it is the most useful table in this chapter. Scanning
outputs for known secrets catches {defn['output scan for known secrets'][0]:.0%} of leaks and
{defn['output scan for known secrets'][1]:.0%} of hijacks. Scanning for a "wrong action"
catches {defn["output scan for 'wrong action'"][1]:.0%} of hijacks, because **a hijacked
action looks like a legitimate action** -- well-formed arguments, a plausible target, a
sensible tool. That is ch:sd-architecture's semantic-failure result in security clothing.

Only the last two rows work against hijacking, and neither is a detector.

The posture table prices it. Output scanning alone takes the leak residual from
{post['nothing'][0]:.2f} to {post['output scanning only'][0]:.2f} and leaves the hijack
residual at {post['output scanning only'][1]:.2f} -- **essentially untouched**, for
{POSTURES[1][2]:.0%} of utility. An allow-list alone leaves leaks at
{post['allow-list only'][0]:.2f} and takes hijacks to {post['allow-list only'][1]:.2f}.

The two together reach {post['output scanning + allow-list'][2]:.2f} total for
{POSTURES[3][2]:.0%} utility, and they are complementary because they address different
outcomes rather than reinforcing each other on the same one. **A defence-in-depth stack made
of two output scanners is depth against one of the two attacks.**

The last table is the lever that gets forgotten. Leaking is bounded by context contents, so
removing a session token from the prompt takes the leak bound from {leak_bound:.1f} to
{leak_bound - 9.0:.1f} for {0.04:.0%} of utility -- **a bigger reduction than every output
scanner combined**, at a fraction of the cost, achieved by deleting a line from a template.

Which is the recommendation this chapter ends on and it is unglamorous. Before building a
detector, look at what is in the prompt and remove what does not need to be there, then look
at what the agent can call and remove what it does not need. Both are subtractions, both are
cheap, and both bound an outcome that no detector bounds.""")
