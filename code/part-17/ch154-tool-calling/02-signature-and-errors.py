# -*- coding: utf-8 -*-
# Extracted from: Chapter 154 — Tool Calling and Tool Design
# Source: src/.../ch154-tool-calling.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Tool design: the signature, and the error message.

The previous listing found argument construction to be the whole remaining
failure budget once tool overlap is removed. This one asks what a tool's DESIGN
does to that budget (eq:signature-cost), and then measures the design decision
that is almost never treated as one: what the tool says when it fails.

An agent that gets a call wrong will try again. Whether the second attempt is
better than the first depends entirely on whether the error told it anything, and
that is a property of the tool rather than of the model.
"""
import numpy as np

rng = np.random.default_rng(1663)

N = 60000
P_FREE = 0.965          # an argument the model must compose freely
P_ENUM = 0.995          # an argument chosen from an enumerated set
MAX_RETRIES = 3


def first_call(n_req, n_opt, p_req, p_opt=None):
    """Probability the call is well formed. Required arguments must all be
    right; optional ones only matter when the model chooses to supply them,
    which it does half the time."""
    p_opt = p_req if p_opt is None else p_opt
    supplied = rng.random((N, n_opt)) < 0.5 if n_opt else np.zeros((N, 0), bool)
    ok_req = (rng.random((N, n_req)) < p_req).all(1) if n_req else np.ones(N, bool)
    ok_opt = ((rng.random((N, n_opt)) < p_opt) | ~supplied).all(1) \
        if n_opt else np.ones(N, bool)
    return ok_req & ok_opt


print(f"A well-formed call needs every required argument right. A free-text")
print(f"argument is right {P_FREE:.1%} of the time; an enumerated one")
print(f"{P_ENUM:.1%}. Optional arguments are supplied half the time and can")
print("only hurt when they are.")
print()
print(f"{'signature':>34}{'first call':>13}{'vs 3 free':>12}")
print("-" * 59)
SIGS = [
    ("3 required, free text", 3, 0, P_FREE),
    ("2 required, free text", 2, 0, P_FREE),
    ("1 required, free text", 1, 0, P_FREE),
    ("3 required, enumerated", 3, 0, P_ENUM),
    ("2 required + 2 optional, free", 2, 2, P_FREE),
    ("2 required + 6 optional, free", 2, 6, P_FREE),
    ("2 required + 6 optional, enum", 2, 6, P_ENUM),
]
sig = {}
base = None
for name, nr, no, p in SIGS:
    v = float(first_call(nr, no, p).mean())
    if base is None:
        base = v
    sig[name] = v
    print(f"{name:>34}{v:>13.1%}{v - base:>+12.1%}")

print()
print()
print("Now let the agent retry. A tool's error message either identifies the")
print("problem, hints at it, or says nothing. Success after up to")
print(f"{MAX_RETRIES} retries, on the '2 required + 6 optional, free' signature:")
print()
QUALITY = [("opaque ('error')", 0.02),
           ("generic ('invalid request')", 0.20),
           ("names the bad field", 0.62),
           ("names it and lists valid values", 0.93)]
print(f"{'error message':>34}" + "".join(f"{'try ' + str(k):>10}"
                                         for k in range(MAX_RETRIES + 1)))
print("-" * 74)
p0 = sig["2 required + 6 optional, free"]
retry = {}
for name, fix in QUALITY:
    cum, p = [p0], p0
    for _ in range(MAX_RETRIES):
        # A retry succeeds if the message let the model locate the fault.
        p = p + (1 - p) * fix
        cum.append(p)
    retry[name] = cum
    print(f"{name:>34}" + "".join(f"{v:>10.1%}" for v in cum))

print()
print()
print("Two budgets for the same end-to-end target. Which is cheaper: a better")
print("signature, or a better error message?")
print()
print(f"{'design':>44}{'calls made':>13}{'success':>11}")
print("-" * 67)


def cost_and_success(p_first, fix, tries=MAX_RETRIES):
    p, calls, succ = p_first, 1.0, p_first
    for _ in range(tries):
        calls += (1 - p)          # a retry happens only if the last call failed
        p = p + (1 - p) * fix
    return calls, p


plans = [
    ("6 optional args, opaque errors", sig["2 required + 6 optional, free"],
     0.02),
    ("6 optional args, good errors", sig["2 required + 6 optional, free"],
     0.93),
    ("no optional args, opaque errors", sig["2 required, free text"], 0.02),
    ("no optional args, good errors", sig["2 required, free text"], 0.93),
    ("enumerated args, opaque errors", sig["3 required, enumerated"], 0.02),
]
plan = {}
for name, pf, fx in plans:
    c, s = cost_and_success(pf, fx)
    plan[name] = (c, s)
    print(f"{name:>44}{c:>13.2f}{s:>10.2%}")

print()
print()
print("And what it costs across an agent run. Five tool calls per task, each")
print("with retries, under the same designs.")
print()
print(f"{'design':>44}{'task success':>14}{'calls':>9}")
print("-" * 67)
for name, pf, fx in plans:
    c, s = cost_and_success(pf, fx)
    print(f"{name:>44}{s ** 5:>14.2%}{5 * c:>9.1f}")

opa = retry["opaque ('error')"]
good = retry["names it and lists valid values"]
print(f"""
The first table is the signature, and the ordering is not subtle.

Going from three required free-text arguments to one buys
{sig['1 required, free text'] - sig['3 required, free text']:+.1%}. Keeping three
but enumerating them buys
{sig['3 required, enumerated'] - sig['3 required, free text']:+.1%} -- more,
because {P_ENUM:.1%} per argument cubed still beats {P_FREE:.1%} per argument
once.

**Constraining an argument's range is worth more than removing it**, which is
convenient, because removing arguments removes capability and constraining them
does not.

The optional-argument rows are the ones worth staring at. Adding six optional
arguments to a two-required signature costs
{sig['2 required + 6 optional, free'] - sig['2 required, free text']:.1%}, even
though the model supplies each of them only half the time and is never obliged to
supply any. An optional argument is not free. It is a coin flip on whether the
model creates an opportunity to be wrong, and six of them are six coin flips.

Enumerating those same six recovers most of it:
{sig['2 required + 6 optional, enum'] - sig['2 required + 6 optional, free']:+.1%}.

The second table is the design decision that is almost never treated as one.

With an opaque error message, three retries take success from {opa[0]:.1%} to
{opa[3]:.1%} -- the retries are independent draws from the same distribution and
buy {opa[3] - opa[0]:.1%}. With a message that names the bad field and lists the
valid values, the same three retries take it to {good[3]:.1%}.

**The error message is worth {good[3] - opa[3]:.1%}, which is more than any
signature change in the first table.** And it costs nothing at inference time: it
is a string the tool already had to construct in order to reject the call.

The reason the effect is so large is worth stating precisely. A retry after an
opaque failure is a fresh sample from the same distribution, so it succeeds at
the original rate -- ch:rsn-test-time-compute's coverage, with no selector. A
retry after an informative failure is CONDITIONED on the fault, so it is a
different and much better distribution. **An error message is the cheapest
selector in this book**, and it is the only one that requires no model at all.

The third table prices designs against each other, and it contains the practical
recommendation.

Six optional arguments with opaque errors reaches {plan['6 optional args, opaque errors'][1]:.2%}
in {plan['6 optional args, opaque errors'][0]:.2f} calls. The same signature with
good errors reaches {plan['6 optional args, good errors'][1]:.2%} in
{plan['6 optional args, good errors'][0]:.2f}. Removing the optional arguments
and keeping opaque errors reaches {plan['no optional args, opaque errors'][1]:.2%}.

So a bad signature with good errors beats a good signature with bad errors, by
{plan['6 optional args, good errors'][1] - plan['no optional args, opaque errors'][1]:.1%}
-- and it does so while making FEWER calls, because most of its retries succeed
on the first attempt after the fault is named.

The last table is the same comparison across an agent run of five tool calls, and
it is where the numbers stop being incremental. Task success goes
{plan['6 optional args, opaque errors'][1] ** 5:.1%} for the worst design against
{plan['no optional args, good errors'][1] ** 5:.1%} for the best. **Per-call
differences of a few points become task differences of tens of points**, because
five calls compound (ch:rsn-cot's eq:chain-accuracy-compounds, again).

That exponent is the reason tool design is an agent concern rather than an API
concern. A tool whose call succeeds {sig['2 required + 6 optional, free']:.0%}
of the time is a perfectly reasonable API and a bad agent tool, and nothing about
the API's own metrics would tell you.

Three rules follow, in order of what they buy.

Enumerate every argument that can be enumerated, and make the enumeration part of
the schema so constrained decoding (part:8) can enforce it. This removes the
failure rather than reducing it.

Make every error message name the field and list the acceptable values. It is the
highest-return line of code in an agent system and it is usually written by
whoever was in a hurry.

And treat every optional argument as a required decision. If the model has to
decide whether to supply it, it is not optional from the model's point of view --
it is a required decision with two branches, and it costs like one.""")
