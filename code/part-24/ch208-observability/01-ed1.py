# -*- coding: utf-8 -*-
# Extracted from: Chapter 208 — Observability: Logging, Metrics, and Tracing
# Source: src/.../ch208-observability.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A trace that records what happened cannot say why, unless it records the inputs.

Standard tracing captures timing and structure: which service called which, how long each
span took, what status came back. That is exactly what is needed to diagnose a latency
or availability problem, and it is nearly useless for a semantic one.

To attribute a wrong answer you need what went IN -- the prompt as assembled, the
documents retrieved, the tool results returned, the model version that produced it. Those
are payloads rather than metadata, and standard tracing drops payloads on purpose
(eq:attribution-needs-payload-not-timing).

This listing measures which fields actually resolve an investigation, and finds the
useful ones are the expensive ones.
"""
# (field, bytes per trace, P(this field alone resolves an investigation),
#  captured by default tracing?)
FIELDS = [
    ("span timings",            340,  0.04, True),
    ("service topology",        180,  0.02, True),
    ("status codes",             90,  0.03, True),
    ("request id and user",     120,  0.06, True),
    ("model version",            60,  0.19, False),
    ("decoding parameters",      80,  0.08, False),
    ("assembled prompt",      11400,  0.34, False),
    ("retrieved doc ids",       420,  0.27, False),
    ("retrieved doc text",    46000,  0.31, False),
    ("tool call arguments",    2100,  0.22, False),
    ("tool results",           8800,  0.24, False),
    ("output text",            4300,  0.29, False),
    ("verifier score",           40,  0.17, False),
]
TRACES_PER_DAY = 1.4e6
# Indexed, searchable, access-controlled log storage -- not cold object
# storage. This is the price that makes the trade real.
STORE_PER_GB_MONTH = 3.40


def resolves(fields):
    """P(the investigation resolves) if these fields are present.

    Fields are partially redundant: each has an independent chance of being the
    one that explains it, so the complement multiplies.
    """
    miss = 1.0
    for name, b, p, d in FIELDS:
        if name in fields:
            miss *= (1.0 - p)
    return 1.0 - miss


def bytes_of(fields):
    return sum(b for name, b, p, d in FIELDS if name in fields)


default = set(n for n, b, p, d in FIELDS if d)
allf = set(n for n, b, p, d in FIELDS)

print("A service at %.1f million traces a day. Retained, indexed log storage costs"
      % (TRACES_PER_DAY / 1e6))
print("%.2f per GB-month -- the searchable kind, not an archive." % STORE_PER_GB_MONTH)
print()
print("Trace fields, their size, and their chance of resolving an investigation.")
print()
print(f"{'field':>24}{'bytes':>9}{'resolves':>11}{'default':>10}"
      f"{'resolve per KB':>17}")
print("-" * 74)
tab = {}
for name, b, p, d in FIELDS:
    tab[name] = (b, p, d)
    print(f"{name:>24}{b:>9}{p:>11.0%}{('yes' if d else 'no'):>10}"
          f"{p / (b / 1024.0):>17.3f}")

print()
print(f"default tracing: {len(default)} fields, {bytes_of(default):,} bytes, "
      f"resolves {resolves(default):.0%}")
print(f"everything:      {len(allf)} fields, {bytes_of(allf):,} bytes, "
      f"resolves {resolves(allf):.0%}")

print()
print()
print("What default tracing gives you, and what it costs.")
print()


def gb_month(fields):
    return bytes_of(fields) * TRACES_PER_DAY * 30.0 / 1e9


print(f"{'configuration':>28}{'bytes/trace':>14}{'GB/month':>12}"
      f"{'cost/month':>13}{'resolves':>11}")
print("-" * 80)
CONFIGS = [
    ("default tracing", default),
    ("+ model version", default | {"model version"}),
    ("+ decoding params", default | {"model version", "decoding parameters"}),
    ("+ verifier score", default | {"model version", "decoding parameters",
                                    "verifier score"}),
    ("+ retrieved doc ids", default | {"model version", "decoding parameters",
                                       "verifier score", "retrieved doc ids"}),
    ("+ tool call arguments", default | {"model version", "decoding parameters",
                                         "verifier score", "retrieved doc ids",
                                         "tool call arguments"}),
    ("+ output text", default | {"model version", "decoding parameters",
                                 "verifier score", "retrieved doc ids",
                                 "tool call arguments", "output text"}),
    ("everything", allf),
]
cfg = {}
for label, f in CONFIGS:
    g = gb_month(f)
    cfg[label] = (bytes_of(f), g, g * STORE_PER_GB_MONTH, resolves(f))
    print(f"{label:>28}{bytes_of(f):>14,}{g:>12,.0f}"
          f"{g * STORE_PER_GB_MONTH:>13,.0f}{resolves(f):>11.0%}")

print()
print()
print("Ranked by resolution bought per kilobyte -- which is the ordering a")
print("storage budget should follow.")
print()
extra = [f for f in FIELDS if not f[3]]
order = sorted(extra, key=lambda f: -(f[2] / (f[1] / 1024.0)))
print(f"{'rank':>6}{'field':>24}{'bytes':>9}{'resolves':>11}"
      f"{'per KB':>11}{'GB/month':>12}")
print("-" * 74)
for i, (name, b, p, d) in enumerate(order, 1):
    print(f"{i:>6}{name:>24}{b:>9}{p:>11.0%}{p / (b / 1024.0):>11.3f}"
          f"{b * TRACES_PER_DAY * 30.0 / 1e9:>12,.0f}")

print()
print()
print("Building up in that order: what each step buys and costs.")
print()
print(f"{'after adding':>24}{'resolves':>11}{'GB/month':>12}"
      f"{'cost/month':>13}{'cost per point':>17}")
print("-" * 78)
cur = set(default)
prev_r = resolves(cur)
prev_c = gb_month(cur) * STORE_PER_GB_MONTH
path = []
for name, b, p, d in order:
    cur.add(name)
    r = resolves(cur)
    c = gb_month(cur) * STORE_PER_GB_MONTH
    per = (c - prev_c) / max((r - prev_r) * 100, 1e-9)
    path.append((name, r, c, per))
    print(f"{name:>24}{r:>11.0%}{gb_month(cur):>12,.0f}{c:>13,.0f}"
          f"{per:>17,.0f}")
    prev_r, prev_c = r, c

print()
print()
print("And the alternative to storing everything: store the small fields always")
print("and the large ones only for requests a verifier flagged.")
print()
SMALL = {n for n, b, p, d in FIELDS if b <= 2500}
LARGE = allf - SMALL
print(f"small fields (<=2500 bytes): {len(SMALL)}")
print(f"large fields:                {len(LARGE)}")
print()
print(f"{'flag rate':>11}{'GB/month':>12}{'cost/month':>13}"
      f"{'resolves flagged':>19}{'resolves unflagged':>21}")
print("-" * 78)
sel = {}
for rate in (1.00, 0.20, 0.05, 0.02, 0.005):
    g = (bytes_of(SMALL) + bytes_of(LARGE) * rate) * TRACES_PER_DAY * 30.0 / 1e9
    sel[rate] = (g, g * STORE_PER_GB_MONTH)
    print(f"{rate:>11.1%}{g:>12,.0f}{g * STORE_PER_GB_MONTH:>13,.0f}"
          f"{resolves(allf):>19.0%}{resolves(SMALL):>21.0%}")

print(f"""
The field table is the shape of the problem, and the `default` column is where it sits.
Everything standard tracing captures -- timings, topology, status codes, request
identity -- resolves **{resolves(default):.0%}** of investigations between them
(eq:attribution-needs-payload-not-timing).

That is not a criticism of tracing. Those fields were chosen to diagnose latency and
availability, they do it superbly, and ch:sd-architecture already established that
neither of those is the failure mode here.

The fields that do resolve semantic investigations are the inputs: the assembled prompt
at {tab['assembled prompt'][1]:.0%}, the retrieved document text at
{tab['retrieved doc text'][1]:.0%}, the output at {tab['output text'][1]:.0%}. **They are
payloads, and standard tracing drops payloads deliberately** -- for cost, for privacy,
and because in a conventional system the payload is not what went wrong.

The cost table is why the deliberate choice is defensible. Capturing everything is
{cfg['everything'][0]:,} bytes a trace, which at {TRACES_PER_DAY / 1e6:.1f} million
traces a day is **{cfg['everything'][1]:,.0f} GB a month** and
{cfg['everything'][2]:,.0f} in storage. Default tracing is
{cfg['default tracing'][1]:,.0f} GB.

**A factor of {cfg['everything'][1] / cfg['default tracing'][1]:.0f} in storage for a
factor of {resolves(allf) / resolves(default):.1f} in resolution** -- which is a real
trade and not an obvious one.

The per-kilobyte ranking is where the trade becomes tractable, because the fields differ
enormously in density. `{order[0][0]}` resolves {order[0][2]:.0%} in
{order[0][1]} bytes -- {order[0][2] / (order[0][1] / 1024.0):.2f} points per kilobyte.
`{order[-1][0]}` resolves {order[-1][2]:.0%} in {order[-1][1]:,} bytes, which is
{order[-1][2] / (order[-1][1] / 1024.0):.3f}.

**Three orders of magnitude between the best and worst field**, and the best ones are
tiny. Model version is sixty bytes and resolves nearly a fifth of investigations on its
own.

The build-up path prices that. Adding the four cheapest fields --
`{order[0][0]}`, `{order[1][0]}`, `{order[2][0]}`, `{order[3][0]}` -- takes resolution
from {resolves(default):.0%} to {path[3][1]:.0%} for
{path[3][2]:,.0f} a month, against {cfg['everything'][2]:,.0f} for everything.

**Most of the resolution is in fields that cost almost nothing to store**, and a team
that concluded "we cannot afford to log payloads" and stopped has skipped the four that
were affordable.

The selective table is the design that gets the rest. Store the small fields on every
trace and the large ones only when something flags the request -- a verifier rejection,
a user retry, an anomalous score. At a {0.02:.0%} flag rate the storage is
{sel[0.02][0]:,.0f} GB a month against {sel[1.0][0]:,.0f} for everything, and the flagged
requests -- the ones an investigation will actually open -- have full fidelity.

That has one severe limitation worth stating rather than burying. **The flag has to be
computable at trace time**, and ch:sd-architecture's whole point was that semantic failure
is not detectable at request time. A verifier score is available; user dissatisfaction
three days later is not.

So selective capture works for the failures something noticed and fails for the ones
nothing did -- which are the failures this book has been about. ch:ops-observability's
second listing takes that up, because it is the harder half of the problem.""")
