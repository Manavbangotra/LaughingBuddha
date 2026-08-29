---
id: ops-observability
number: 208
part: XXIV
tier: full
status: draft
requires: [semantic-failure-has-no-instrument, diagnosis-cost-grows-with-unpinned-artefacts,
           period-destroys-attribution, semantic-breaker-is-affordable]
provides: [attribution-needs-payload-not-timing, cheap-fields-carry-most-attribution,
           uniform-sampling-misses-rare-failures, biased-sampling-distorts-composition]
citations: [sculley2015, breck2017, deshpande2025trail, paleyes2020deployment]
---

## 1. Learning Objectives

By the end of this chapter you will be able to distinguish the trace fields that diagnose
latency from those that attribute a wrong answer, and explain why standard tracing
captures the first and drops the second; rank trace fields by resolution bought per
kilobyte and identify the cheap ones that carry most of the value; design a selective
capture policy and state its structural limitation; explain why uniform sampling captures
exactly the sampling rate's share of failures, by construction; and show that biased
sampling distorts the apparent composition of failure modes toward whatever the detector
can see.

## 2. Why This Matters

{{ch:ops-versioning}} ended with a candidate space of 66,960 combinations and the
observation that nobody searches it. This chapter is about what would let you search it,
and the answer is not what the observability stack already collects.

Standard tracing captures timings, topology, status codes and request identity. Together
those resolve **14%** of semantic investigations
({{eq:attribution-needs-payload-not-timing}}). The fields that resolve them are the
inputs — the assembled prompt, the retrieved documents, the tool results — which are
payloads, and payloads are dropped deliberately.

But the cost argument that justifies dropping them is weaker than it looks. Ranked by
resolution per kilobyte, the best fields are tiny: **model version is 60 bytes and
resolves 19%**; a verifier score is 40 bytes and resolves 17%. Adding the four cheapest
takes resolution from **14% to 61%** for **$190 a month** against **$10,557** for
everything ({{eq:cheap-fields-carry-most-attribution}}).

The second half is about sampling, and it contains a trap. Uniform sampling captures
exactly the sample rate's share of failures — 0.5% of failures at 0.5% sampling, by
construction ({{eq:uniform-sampling-misses-rare-failures}}). Biasing toward flagged
requests captures **22× more**. And it distorts the composition: schema violations become
**1.9×** over-represented and subtly wrong reasoning **5×** under-represented
({{eq:biased-sampling-distorts-composition}}), so a team reading counts off the sample
draws the wrong conclusion about where its problems are.

## 3. Prerequisites

You need {{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}}. Everything
here follows from it: the fields standard tracing collects were chosen for failures that
announce themselves, and the failure mode in question does not.

{{eq:diagnosis-cost-grows-with-unpinned-artefacts}} from {{ch:ops-versioning}} defines the
candidate space this chapter's fields narrow.

{{eq:period-destroys-attribution}} from {{ch:ops-lifecycle}} is why the window is long
enough for the candidate space to be large.

{{eq:semantic-breaker-is-affordable}} from {{ch:sd-fault-tolerance}} supplies the verifier
score that turns out to be the single densest trace field.

## 4. Intuitive Explanation

Distributed tracing is one of the genuine successes of the last decade of operations. A
request enters, fans out across a dozen services, and a trace shows you exactly where the
time went, what called what, and which hop returned the error. Diagnosing a latency
regression that used to take a day takes ten minutes.

Now hold that up against the problem this book has been about. The request succeeded.
Every span returned 200. The latency was normal. The answer was wrong.

Look at what the trace tells you. Service A called service B, which took 40 milliseconds
and returned successfully. That is completely true and it does not contain a single fact
that bears on why the answer was wrong.

To answer that you need what went *in*: what the prompt actually said after assembly,
which documents came back from retrieval and what they contained, what the tools
returned, which model version produced it. Those are payloads, and every tracing system
drops payloads by default — for storage cost, for privacy, and because in a conventional
system the payload is rarely the thing that broke.

So there is a real cost argument here, and it is worth taking seriously rather than
dismissing. Capturing everything in {{sec:9-practical-example}}'s service is three
thousand gigabytes a month of indexed, searchable, access-controlled storage. That is not
free.

But the argument is usually made about the *wrong fields*. The expensive fields are the
document text and the assembled prompt — tens of kilobytes each. The fields that resolve
investigations most efficiently are tiny. Which model version served this? Sixty bytes.
What did the verifier score it? Forty bytes. What documents came back — not their text,
just their identifiers? Four hundred bytes.

Those three fields together are half a kilobyte and they take you from fourteen percent
resolution to sixty-one. A team that concluded "we cannot afford to log payloads" and
stopped there skipped the ones that were affordable.

The second half of the chapter is about which requests you keep, and it starts from an
arithmetic fact that is obvious once stated and is routinely missed.

If you sample one request in two hundred, you keep one bad request in two hundred. Not
more, not less. A uniform sample is uniform — that is the whole point of it — so the
share of failures you capture is exactly your sampling rate. You cannot fix a shortage of
failure examples by hoping.

The obvious fix is to sample non-uniformly: keep the requests that look suspicious. That
works, and it works well — the same storage budget spent on flagged requests captures
twenty-two times as many bad ones.

And here is the trap. A signal can only flag what it recognises. A malformed tool call is
obvious; a schema violation is obvious. A confidently stated wrong fact looks exactly
like a confidently stated right one, and subtly flawed reasoning looks like reasoning.

So the biased sample is not a sample of your failures. It is a sample of *the failures
your detector can see*, and its composition is the detector's recall profile rather than
the truth. Schema violations show up at roughly twice their real frequency and subtle
reasoning failures at a fifth of theirs.

Then someone counts the failures in the sample and builds a roadmap. Schema validation
looks urgent; reasoning quality looks minor. Both conclusions are artefacts of how the
data was collected, and nothing in the data will contradict them — because the evidence
that would has not been kept.

## 5. Formal Explanation

**Field value.** Let a trace comprise fields $f$ with size $b_f$ bytes and resolution
probability $\pi_f$ — the chance that field alone explains an investigation. Treating the
fields' explanatory power as independent, the probability that a configuration $F$
resolves an investigation is

$$ R(F) \;=\; 1 - \prod_{f \in F}(1 - \pi_f) $$ (eq:attribution-needs-payload-not-timing)

and its storage is $\sum_{f\in F} b_f$ per trace. The default set $D$ — timings,
topology, status, identity — has small $\pi_f$ for every member because those fields
describe *control flow*, and a semantic failure has normal control flow.

Field density is $\pi_f / b_f$, and the greedy ordering by density is optimal for the
knapsack relaxation. Because $\pi_f$ is uncorrelated with $b_f$ — a version string is as
explanatory as a document body and three orders of magnitude smaller — **the density
ordering is very different from the raw-resolution ordering**:

$$ \text{rank}_{\text{density}}(f) \;=\; \frac{\pi_f}{b_f} \;\ne\; \text{rank}(\pi_f) $$ (eq:cheap-fields-carry-most-attribution)

**Sampling.** Let failures occur at rate $\epsilon$ and traces be kept at rate $\rho$.
Under uniform sampling the expected bad traces captured per unit traffic is $\rho\epsilon$,
so the share of failures captured is

$$ \frac{\rho\epsilon}{\epsilon} \;=\; \rho $$ (eq:uniform-sampling-misses-rare-failures)

— **identically the sampling rate**, for every failure mode, with no dependence on
$\epsilon$. That is the defining property of a uniform sample and the reason it cannot be
tuned toward failures.

Under biased sampling with a signal of recall $\tau_m$ on mode $m$, the captured count for
mode $m$ is proportional to $\phi_m\tau_m$ for mode share $\phi_m$. The sample's
composition is therefore

$$ \hat{\phi}_m \;=\; \frac{\phi_m \tau_m}{\sum_{m'} \phi_{m'}\tau_{m'}}, \qquad \frac{\hat{\phi}_m}{\phi_m} \;=\; \frac{\tau_m}{\bar{\tau}} $$ (eq:biased-sampling-distorts-composition)

where $\bar\tau$ is the share-weighted mean recall. **The distortion on mode $m$ is
exactly its recall relative to the average recall** — modes the detector sees well are
over-represented in proportion, and modes it sees badly are under-represented in
proportion. Uniform sampling has $\tau_m \equiv \rho$ for all $m$, so
$\hat\phi_m/\phi_m = 1$ and the composition is exact.

## 6. Mathematical Foundation

The distortion result has a consequence worth deriving because it turns a sampling choice
into a strategic one.

A team estimating mode frequencies from a biased sample and scaling up by the total
failure count infers $\hat{n}_m = N\hat\phi_m$ against the truth $n_m = N\phi_m$, so the
inference error is exactly $\tau_m/\bar\tau$. That means:

$$ \text{modes with } \tau_m > \bar\tau \text{ are over-counted}; \quad \tau_m < \bar\tau \text{ are under-counted} $$

and since $\bar\tau$ is the share-weighted mean, **at least one mode is always
over-counted and at least one always under-counted.** There is no signal, however good,
that avoids this — only a signal with uniform recall across modes, which is a signal that
is not discriminating on anything mode-related.

The practical form is sharper. Ranking modes by inferred frequency ranks them by
$\phi_m\tau_m$ rather than $\phi_m$, so the ranking is correct only if $\tau_m$ is
constant across modes. **A biased sample cannot be used to prioritise failure modes**,
which is the single most common thing teams do with failure data.

The mitigation is stratification: keep a uniform stratum of size $\rho_u$ and a biased
stratum of size $\rho_b$. The uniform stratum's composition is unbiased, so mode
frequencies estimated from it alone are correct, with variance set by its size. Since
frequency estimation needs far fewer samples than investigation does — a proportion needs
hundreds, a root cause needs the specific request — **the uniform stratum can be small and
still do its job.**

{{sec:9-practical-example}} measures the split: at a 100% biased allocation, subtle
reasoning is 2.9% of the sample against a true 18%; at 0% it is exactly 18% with a
twentieth of the volume. The right design uses the biased stratum for investigation and
the uniform stratum for measurement, and never uses the biased one for counting.

## 7. Internal Mechanics

**Why control-flow fields are uninformative here.** A trace's structure records which
component ran. In a semantic failure every component ran correctly — the retriever
retrieved, the model generated, the validator validated. The failure is in the *content*
that flowed between them, and the trace records the flow rather than the content. This is
not a gap in tracing implementations; it is what tracing is.

**Why the verifier score is the densest field.** Forty bytes that summarise a judgement
about the output — {{ch:sd-fault-tolerance}}'s instrument, written into the trace. It is
dense because it is the only field that is *about* the property in question rather than
about the machinery. Any team that has built the semantic monitor already has this number
and frequently does not put it in the trace.

**Document identifiers versus document text.** Identifiers are 420 bytes and text is
46,000. The identifiers usually suffice, because the corpus is available and the document
can be fetched — *provided the corpus is versioned*, which {{ch:ops-versioning}} found it
usually is not. **So the cheap field is only as good as the versioning**, and a team
without corpus versioning must store the text or lose the information, at a hundredfold
cost. That is the two chapters composing in a way that makes each other's cheap option
available.

**Privacy is a real constraint and it is separable.** Payload capture puts user content in
a log, which brings retention limits, access controls, and regional restrictions. The
cheap fields — version, parameters, scores, identifiers — are almost all non-content, so
{{eq:cheap-fields-carry-most-attribution}}'s recommendation is also the
privacy-conservative one. That is a convenient coincidence and worth exploiting.

**Why the flag must be computable at request time.** Selective capture requires deciding
whether to keep a trace before discarding it, which means the suspicion signal must exist
within the request's lifetime. Verifier scores qualify; user dissatisfaction three days
later does not. **The failures a team learns about from customers are exactly the ones
selective capture will have dropped**, which is the worst possible pairing.

**Agent traces make all of this harder.** {{cite:deshpande2025trail}}'s benchmark of 148
human-annotated traces found the best model reaching only **11%** at localising the issue
within a trace. If automated triage is that weak, trace *structure* — clear step
boundaries, recorded intermediate state, explicit tool inputs and outputs — has to carry
the load for a human reader. {{ch:ops-agent-tracing}} takes this up.

**Why cardinality kills the obvious shortcut.** A natural response to payload cost is to
turn payloads into metric labels -- count requests by model version, by document id, by
prompt hash. Metrics systems price by unique label combination, and document identifiers
have cardinality in the millions, so the shortcut costs more than the traces it was meant
to replace. **Attribution data belongs in a trace store, not a metric store**, and the
distinction matters because the two live in the same product and are billed differently.
The cheap fields in {{sec:9-practical-example}} are cheap as trace attributes and ruinous
as metric dimensions.

**{{cite:sculley2015}}'s entanglement predicts the field list.** When everything affects
everything, the fields that matter are the ones recording what each component was given,
because the components' behaviour is not separable from their inputs. That is the same
reason {{cite:breck2017}}'s readiness tests focus on input validation.

## 8. Implementation

The first listing measures which trace fields resolve investigations and what they cost.

```python {tier=A name=ed1}
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
```

## 9. Practical Example

Trace fields by size, resolution, and density:

```
                   field    bytes   resolves   default   resolve per KB
--------------------------------------------------------------------------
            span timings      340         4%       yes            0.120
        service topology      180         2%       yes            0.114
            status codes       90         3%       yes            0.341
     request id and user      120         6%       yes            0.512
           model version       60        19%        no            3.243
     decoding parameters       80         8%        no            1.024
        assembled prompt    11400        34%        no            0.031
       retrieved doc ids      420        27%        no            0.658
      retrieved doc text    46000        31%        no            0.007
     tool call arguments     2100        22%        no            0.107
            tool results     8800        24%        no            0.028
             output text     4300        29%        no            0.069
          verifier score       40        17%        no            4.352
```

**Default tracing resolves 14%** ({{eq:attribution-needs-payload-not-timing}}). Those
fields describe control flow, and a semantic failure has normal control flow.

The density column is the finding. `verifier score` resolves **17% in 40 bytes** —
**4.35 points per kilobyte** — while `retrieved doc text` resolves 31% in 46,000 bytes,
which is **0.007**. **Three orders of magnitude between the best and worst field.**

```
               configuration   bytes/trace    GB/month   cost/month   resolves
--------------------------------------------------------------------------------
             default tracing           730          31          104        14%
             + model version           790          33          113        31%
           + decoding params           870          37          124        36%
            + verifier score           910          38          130        47%
         + retrieved doc ids         1,330          56          190        61%
       + tool call arguments         3,430         144          490        70%
               + output text         7,730         325        1,104        79%
                  everything        73,930       3,105       10,557        93%
```

**Four cheap fields take resolution from 14% to 61% for $190 a month**, against $10,557
for everything ({{eq:cheap-fields-carry-most-attribution}}). A team that concluded it
could not afford payload logging skipped the fields that were not payloads.

```mermaid {#fig:fields caption="Resolution power is uncorrelated with field size, so the density ordering differs sharply from the raw ordering. The four densest fields are non-content, which makes them the privacy-conservative choice as well as the cheap one."}
flowchart LR
  A["trace fields"] --> B["control flow<br/>timings, topology, status<br/>14% resolution"]
  A --> C["metadata<br/>version, params, scores, ids<br/>+47 points, 600 bytes"]
  A --> D["payload<br/>prompt, docs, output<br/>+32 points, 73KB"]
  C --> E["cheap AND<br/>non-content"]
```

And selective capture:

```
  flag rate    GB/month   cost/month   resolves flagged   resolves unflagged
------------------------------------------------------------------------------
     100.0%       3,105       10,557                93%                  70%
      20.0%         736        2,503                93%                  70%
       5.0%         292          993                93%                  70%
       2.0%         203          691                93%                  70%
       0.5%         159          540                93%                  70%
```

Small fields always, large fields only when flagged: **$691 a month at a 2% flag rate**
with full fidelity on flagged requests. The limitation is that the flag must be
computable at request time — which the failures customers report are not.

The second listing takes up sampling.

```python {tier=A name=ed2}
"""Uniform sampling answers questions about the distribution, not about the failures.

Tracing is sampled because storing everything is expensive, and the sampling is usually
uniform because that gives an unbiased view of the distribution. For latency work that is
exactly right: you want to know the shape, and a random tenth of requests describes it.

For attribution you do not want the shape. You want the specific requests that went
wrong, and they are rare -- so a uniform sample of a rare event contains almost none of it
(eq:uniform-sampling-misses-rare-failures).

The fix is to sample non-uniformly, keeping what looks suspicious. That requires a
suspicion signal available AT REQUEST TIME, and this listing measures what happens when
the signal is imperfect -- which ch:sd-architecture says it must be.
"""
TRAFFIC_PER_DAY = 1.4e6
ERR_RATE = 0.04
RATES = [0.001, 0.005, 0.02, 0.10, 1.00]

print("A service at %.1f million requests a day with a %.0f%% semantic error rate."
      % (TRAFFIC_PER_DAY / 1e6, ERR_RATE * 100))
print("That is %.0f wrong answers a day." % (TRAFFIC_PER_DAY * ERR_RATE))
print()
print("Under uniform sampling, how many bad requests are captured.")
print()
print(f"{'sample rate':>13}{'traces kept':>14}{'bad ones kept':>16}"
      f"{'share of bad':>14}{'days for 200':>15}")
print("-" * 74)
uni = {}
for r in RATES:
    kept = TRAFFIC_PER_DAY * r
    bad = kept * ERR_RATE
    uni[r] = (kept, bad, r, 200.0 / bad if bad > 0 else float("inf"))
    print(f"{r:>13.1%}{kept:>14,.0f}{bad:>16,.0f}{r:>14.1%}"
          f"{200.0 / bad:>15.2f}")

print()
print("The 'share of bad' column is the sample rate. Uniform sampling captures")
print("the same fraction of failures as of everything, by construction.")

print()
print()
print("Now bias the sampling toward requests a signal flags as suspicious.")
print("The signal has a recall and a false-positive rate.")
print()
BUDGET = 0.005          # keep 0.5% of traces, spent however we like
print(f"trace budget: {BUDGET:.1%} of traffic = {TRAFFIC_PER_DAY * BUDGET:,.0f} a day")
print()
print(f"{'signal recall':>15}{'signal FPR':>12}{'flagged/day':>14}"
       f"{'bad in sample':>16}{'vs uniform':>13}")
print("-" * 72)
bias = {}
uniform_bad = TRAFFIC_PER_DAY * BUDGET * ERR_RATE
for rec, fpr in ((1.00, 0.000), (0.80, 0.004), (0.55, 0.012),
                 (0.30, 0.030), (0.10, 0.060)):
    n_bad = TRAFFIC_PER_DAY * ERR_RATE
    n_good = TRAFFIC_PER_DAY * (1 - ERR_RATE)
    flagged_bad = n_bad * rec
    flagged_good = n_good * fpr
    flagged = flagged_bad + flagged_good
    # Spend the budget on flagged requests first.
    keep = TRAFFIC_PER_DAY * BUDGET
    if flagged <= keep:
        captured_bad = flagged_bad
    else:
        captured_bad = flagged_bad * (keep / flagged)
    bias[rec] = (flagged, captured_bad, captured_bad / uniform_bad)
    print(f"{rec:>15.0%}{fpr:>12.1%}{flagged:>14,.0f}"
          f"{captured_bad:>16,.0f}{captured_bad / uniform_bad:>12.1f}x")

print()
print()
print("What that buys in investigation terms: days to accumulate 200 examples")
print("of a failure mode, which is roughly what a pattern needs.")
print()
print(f"{'strategy':>26}{'bad/day':>12}{'days for 200':>15}"
       f"{'days for 1000':>16}")
print("-" * 70)
print(f"{'uniform at 0.5%':>26}{uniform_bad:>12,.0f}"
      f"{200.0 / uniform_bad:>15.1f}{1000.0 / uniform_bad:>16.1f}")
for rec in (0.80, 0.55, 0.30):
    b = bias[rec][1]
    print(f"{('biased, %.0f%% recall' % (rec * 100)):>26}{b:>12,.0f}"
          f"{200.0 / b:>15.1f}{1000.0 / b:>16.1f}")

print()
print()
print("But the bias has a cost the uniform sample does not: it can only find")
print("failures the signal recognises. What it misses, it misses completely.")
print()
MODES = [
    ("schema violation",        0.21, 0.97),
    ("refusal when it should not", 0.14, 0.88),
    ("tool call malformed",     0.11, 0.94),
    ("confidently wrong fact",  0.27, 0.19),
    ("subtly wrong reasoning",  0.18, 0.08),
    ("right but unhelpful",     0.09, 0.04),
]
print(f"{'failure mode':>30}{'share of failures':>20}{'signal recall':>16}"
      f"{'in biased sample':>19}")
print("-" * 86)
covered = 0.0
for name, share, rec in MODES:
    covered += share * rec
    print(f"{name:>30}{share:>20.0%}{rec:>16.0%}"
          f"{share * rec:>19.1%}")
print("-" * 86)
print(f"{'TOTAL':>30}{1.0:>20.0%}{covered:>16.0%}{covered:>19.1%}")

print()
print()
print("How each strategy REPRESENTS the failure modes -- the composition of the")
print("sample, against the true composition of failures.")
print()
denom = sum(sh * rc for _, sh, rc in MODES)
print(f"{'failure mode':>30}{'true share':>13}{'in uniform':>13}"
      f"{'in biased':>12}{'distortion':>13}")
print("-" * 84)
comp = {}
for name, share, rec in MODES:
    in_bias = share * rec / denom
    comp[name] = (share, share, in_bias, in_bias / share)
    print(f"{name:>30}{share:>13.0%}{share:>13.0%}{in_bias:>12.0%}"
          f"{in_bias / share:>12.1f}x")

print()
print("Uniform sampling reproduces the true composition exactly. Biased sampling")
print("reproduces the signal's recall profile instead.")

print()
print()
print("What that does to a team reading counts off the sample.")
print()
print(f"{'failure mode':>30}{'true failures/day':>20}"
      f"{'implied by biased sample':>27}{'error':>11}")
print("-" * 90)
TOTAL_BAD = TRAFFIC_PER_DAY * ERR_RATE
implied = {}
for name, share, rec in MODES:
    true_n = TOTAL_BAD * share
    seen_share = share * rec / denom
    imp = TOTAL_BAD * seen_share
    implied[name] = (true_n, imp, imp / true_n)
    print(f"{name:>30}{true_n:>20,.0f}{imp:>27,.0f}"
          f"{imp / true_n:>10.1f}x")

print()
print()
print("And the strategy that covers both: split the budget.")
print()
print(f"{'split to biased':>17}{'schema/day':>13}{'subtle/day':>13}"
      f"{'subtle share of sample':>25}{'distortion':>13}")
print("-" * 84)
for split in (1.00, 0.80, 0.50, 0.20, 0.00):
    b_budget = TRAFFIC_PER_DAY * BUDGET * split
    u_budget = TRAFFIC_PER_DAY * BUDGET * (1 - split)
    flagged_all = sum(TOTAL_BAD * sh * rc for _, sh, rc in MODES) \
        + TRAFFIC_PER_DAY * (1 - ERR_RATE) * 0.004
    scale = min(1.0, b_budget / flagged_all) if flagged_all > 0 else 0.0
    schema = TOTAL_BAD * 0.21 * 0.97 * scale + u_budget * ERR_RATE * 0.21
    subtle = TOTAL_BAD * 0.18 * 0.08 * scale + u_budget * ERR_RATE * 0.18
    tot = sum(TOTAL_BAD * sh * rc * scale + u_budget * ERR_RATE * sh
              for _, sh, rc in MODES)
    sh_subtle = subtle / tot if tot > 0 else 0.0
    print(f"{split:>17.0%}{schema:>13,.0f}{subtle:>13,.0f}"
          f"{sh_subtle:>25.1%}{sh_subtle / 0.18:>12.2f}x")

print(f"""
The uniform table states the problem in one column. Sampling at {0.005:.1%} captures
{0.005:.1%} of the bad requests -- **by construction**, because a uniform sample is
uniform (eq:uniform-sampling-misses-rare-failures). That gives
{uni[0.005][1]:,.0f} bad traces a day, spread across every failure mode.

Biased sampling looks like an unambiguous win. At {0.80:.0%} recall it captures
{bias[0.8][1]:,.0f} bad traces a day against uniform's {uniform_bad:,.0f} --
**{bias[0.8][2]:.0f} times more** for the same storage budget, and two hundred examples
in {200.0 / bias[0.8][1]:.2f} days rather than {200.0 / uniform_bad:.1f}.

It is a win on volume. The composition table is where the cost appears.

A signal can only flag what it recognises. Schema violations are recognisable at
{0.97:.0%}; a confidently wrong fact at {0.19:.0%}; subtly wrong reasoning at
{0.08:.0%}. So the biased sample's composition is not the failure distribution -- **it is
the signal's recall profile**.

Schema violations are {0.21:.0%} of real failures and
{comp['schema violation'][2]:.0%} of the biased sample, over-represented
{comp['schema violation'][3]:.1f} times. Subtly wrong reasoning is {0.18:.0%} of real
failures and {comp['subtly wrong reasoning'][2]:.0%} of the sample, under-represented
{1 / comp['subtly wrong reasoning'][3]:.0f}-fold.

**Uniform sampling reproduces the true composition exactly; biased sampling reproduces
the detector's blind spots.**

The implied-counts table is why that matters operationally rather than
philosophically. A team reading failure counts off a biased sample sees
{implied['schema violation'][1]:,.0f} schema violations a day against a true
{implied['schema violation'][0]:,.0f}, and {implied['subtly wrong reasoning'][1]:,.0f}
subtle reasoning failures against a true {implied['subtly wrong reasoning'][0]:,.0f}.

They will conclude that schema validation is the pressing problem and reasoning quality
is a minor one. **That conclusion is an artefact of how they sampled**, and nothing in
the data will contradict it, because the evidence that would has not been collected.

This is ch:sd-architecture's missing instrument arriving one level up. There the problem
was that semantic failure has no detector. Here the problem is that **building a detector
and sampling by it makes the undetected failures statistically invisible** -- worse than
before, because now there is a confident-looking distribution to point at.

The split table is the practical answer and it is unsatisfying in an honest way. At a
{1.0:.0%} biased split, subtle failures are {0.029:.1%} of the sample against a true
{0.18:.0%}. At a {0.50:.0%} split they are closer, and at {0.0:.0%} the composition is
exact and the volume is {uniform_bad:,.0f} a day.

**Keep a uniform stratum, always.** It is the only view of the failures your detectors
cannot see, its volume is low, and its value is entirely in what it can discover rather
than in what it can investigate. A team that switched wholly to intelligent sampling has
optimised its ability to study problems it already knows about and given up its ability
to find new ones -- a trade that never appears in a metric, because the thing given up
does not show up until after it is gone.""")
```

```
  sample rate   traces kept   bad ones kept  share of bad   days for 200
--------------------------------------------------------------------------
         0.1%         1,400              56          0.1%           3.57
         0.5%         7,000             280          0.5%           0.71
         2.0%        28,000           1,120          2.0%           0.18
        10.0%       140,000           5,600         10.0%           0.04
```

**The share-of-bad column is the sample rate**
({{eq:uniform-sampling-misses-rare-failures}}). A uniform sample is uniform; you cannot
tune it toward failures.

Biasing toward flagged requests:

```
  signal recall  signal FPR   flagged/day   bad in sample   vs uniform
------------------------------------------------------------------------
           100%        0.0%        56,000           7,000        25.0x
            80%        0.4%        50,176           6,250        22.3x
            55%        1.2%        46,928           4,594        16.4x
            30%        3.0%        57,120           2,059         7.4x
            10%        6.0%        86,240             455         1.6x
```

**22× more bad traces for the same budget** at 80% recall. On volume it is an unambiguous
win. The cost is elsewhere:

```
                  failure mode   true share   in uniform   in biased   distortion
------------------------------------------------------------------------------------
              schema violation          21%          21%         41%         1.9x
    refusal when it should not          14%          14%         25%         1.8x
           tool call malformed          11%          11%         21%         1.9x
        confidently wrong fact          27%          27%         10%         0.4x
        subtly wrong reasoning          18%          18%          3%         0.2x
           right but unhelpful           9%           9%          1%         0.1x
```

**The biased sample's composition is the detector's recall profile, not the failure
distribution** ({{eq:biased-sampling-distorts-composition}}). Uniform reproduces the truth
exactly; biased over-represents schema violations **1.9×** and under-represents subtle
reasoning **fivefold**.

What that does to a team reading the numbers:

```
                  failure mode   true failures/day   implied by biased sample      error
------------------------------------------------------------------------------------------
              schema violation              11,760                     22,833       1.9x
        confidently wrong fact              15,120                      5,750       0.4x
        subtly wrong reasoning              10,080                      1,614       0.2x
```

Schema violations look like the largest problem at 22,833 a day. **In truth confidently
wrong facts are more common (15,120) and schema violations are less (11,760)** — the
ranking is inverted, and nothing in the data will say so.

The split:

```
  split to biased   schema/day   subtle/day   subtle share of sample   distortion
------------------------------------------------------------------------------------
             100%        2,394          169                     2.9%        0.16x
              80%        1,927          145                     3.1%        0.17x
              50%        1,226          110                     3.6%        0.20x
              20%          526           74                     5.3%        0.29x
               0%           59           50                    18.0%        1.00x
```

**Keep a uniform stratum.** It is the only unbiased view, it can be small because
estimating a proportion needs far fewer samples than finding a root cause, and it is the
only view of the failures the detector cannot see.

## 10. Production Considerations

Add the four cheap fields today: model version, decoding parameters, verifier score,
retrieved document identifiers. Six hundred bytes, and they take resolution from 14% to
61%.

Put the verifier score in the trace. If {{ch:sd-fault-tolerance}}'s monitor exists, the
number already exists; writing it into the trace costs forty bytes and it is the densest
field available.

Store document identifiers rather than text — but only if the corpus is versioned. Without
{{ch:ops-versioning}}'s corpus pinning the identifier is not resolvable, and the cheap
option becomes the useless one.

Use selective capture for large payloads, and be explicit that it can only catch what a
request-time signal sees.

Always keep a uniform stratum. Size it for proportion estimation, not for investigation —
it can be an order of magnitude smaller than the biased stratum and still do its job.

Never rank failure modes from a biased sample. Rank from the uniform stratum; investigate
from the biased one. Mixing those is the most common error available here.

Record the sampling policy alongside the data. A count without the policy that produced it
is uninterpretable, and policies change without anyone re-reading the dashboards built on
them.

## 11. Common Mistakes

**Expecting tracing to explain a semantic failure.** It records control flow, which was
normal.

**Concluding payload logging is unaffordable and stopping.** The four densest fields are
not payloads.

**Storing document text when identifiers would do.** A hundredfold cost, avoidable if the
corpus is versioned.

**Switching entirely to intelligent sampling.** It optimises investigation and destroys
discovery.

**Counting failure modes from a biased sample.** The ranking is $\phi_m\tau_m$, not
$\phi_m$.

**Assuming a request-time flag catches customer-reported failures.** Those are precisely
the ones it did not see.

## 12. Failure Modes

**Silent composition drift.** The detector improves on one mode, that mode's apparent
share rises, and a team reads it as the problem getting worse.

**Unresolvable identifiers.** Document ids were stored, the corpus moved, and the trace
points at content that no longer exists.

**Uniform stratum quietly removed.** A cost-reduction exercise deletes the stratum that
carries no investigations and all the measurement, and nobody notices until a new failure
mode needs discovering.

**Retention shorter than the loop period.** {{ch:ops-lifecycle}}'s 35-day period against a
14-day retention means the traces from the window under investigation are already gone.

**Privacy-driven redaction of the wrong fields.** A blanket policy removes the cheap
non-content fields alongside the payloads, on the assumption that everything in a trace is
sensitive — losing the most valuable fields to protect content they do not contain.

**Attribution fields promoted to metric dimensions.** Document identifiers become metric
labels, cardinality explodes, and the observability bill rises by more than the traces
would have cost.

## 13. Alternatives

**Log everything, retain briefly.** Full fidelity for a short window; works when
detection is fast and fails against {{ch:ops-lifecycle}}'s long period.

**Reconstruct rather than record.** Store inputs and re-execute to obtain intermediates.
Cheapest possible storage and it requires {{ch:ops-versioning}}'s reproducibility, which
the same team probably lacks.

**Aggregate rather than trace.** Keep distributions and counts, no individual requests.
Sufficient for measurement, useless for attribution — and it is what most teams
accidentally have.

**Tail-based sampling.** Decide after the request completes, using its full outcome. Better
signal than head-based sampling, at the cost of buffering every trace until the decision.

**User-initiated capture.** Let a user flag a bad answer and capture everything for that
request going forward. Catches exactly the modes automated signals miss, and only after
the fact.

## 14. Evaluation

Report resolution rate — the share of investigations that reach a root cause — as a
standing metric. It is the outcome this whole chapter optimises and almost nobody measures
it.

Measure field density empirically from closed investigations: which field was decisive?
The prior in {{sec:9-practical-example}} is illustrative; yours is measurable.

Compare mode composition between your uniform and biased strata. The ratio is the
distortion, and it tells you how wrong your biased counts are.

Track trace retention against the loop period. Retention shorter than the period means
some investigations are impossible before they start.

Audit whether identifiers in traces still resolve. Test it on traces from three months ago,
not on new ones.

## 15. Advanced Concepts

The independence assumption in $R(F)$ is optimistic. Fields are correlated in what they
explain — knowing the model version and the decoding parameters often explains the same
class of problem — so the true resolution of a field set is below the product formula.
That makes the cheap-fields result *stronger* rather than weaker, because the correlated
redundancy is concentrated among the expensive payload fields: prompt, documents, and
output all describe the same content from different angles.

The distortion result assumes the detector's recall is fixed per mode. In practice recall
improves where teams invest, and teams invest where the sample shows problems — which is a
feedback loop. Modes the detector sees get investigated, get fixed, and get better
detection; modes it does not see get neither. **The sampling policy therefore shapes not
just the measurement but the roadmap, and the roadmap shapes the next measurement.** Over
several cycles a biased-only sampling regime can drive a team's entire quality programme
into the detector's blind spot, and the data will show steady improvement throughout.

The resolution probabilities also assume a single investigator with the whole trace in
front of them. Real investigations are collaborative and partial: one person has the
trace, another has the corpus, a third knows what changed. That means a field's practical
value depends on whether the person investigating can *interpret* it -- a document
identifier is decisive to someone with corpus access and inert to someone without.
**Access boundaries reduce effective resolution in a way the field list does not show**,
and they are frequently drawn along exactly the payload/non-payload line that privacy
policy created. The team most likely to be investigating is often the team least likely
to have access to the field that would resolve it.

There is a design question this chapter does not settle about what a trace is *for*. This
analysis treats it as evidence for a future investigation, which argues for payloads and
retention. It is also a real-time signal — {{ch:sd-fault-tolerance}}'s breaker reads it —
which argues for small, fast, aggregatable fields. Those are different systems with
different retention, different indexing, and different costs, and most teams run one system
serving both purposes badly. Separating them is probably correct and it doubles the
pipelines.

## 16. Connection to Previous Chapters

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is the root: tracing
records control flow because control flow is what conventional failures disturb.

{{eq:diagnosis-cost-grows-with-unpinned-artefacts}} from {{ch:ops-versioning}} defines the
candidate space; trace fields are how it is narrowed, and the two compose — versioned
corpora make cheap identifiers sufficient.

{{eq:semantic-breaker-is-affordable}} from {{ch:sd-fault-tolerance}} produces the verifier
score, which turns out to be the densest trace field in the table.

{{eq:period-destroys-attribution}} from {{ch:ops-lifecycle}} sets the retention
requirement: traces must outlive the loop period or the window under investigation is
already gone.

## 17. Exercises

1. From your own closed investigations, estimate $\pi_f$ for five trace fields. Which is
   densest?

2. Compute the cost of adding model version, decoding parameters, and verifier score to
   your traces. What resolution would you expect?

3. Show that under uniform sampling the captured share of every failure mode equals the
   sampling rate, and identify what breaks if sampling is not independent of the request.

4. Derive $\hat\phi_m/\phi_m = \tau_m/\bar\tau$ and find the condition under which a
   biased sample gives an unbiased ranking.

5. Size a uniform stratum to estimate a 5% mode share to within one point. How does it
   compare to your biased stratum?

## 18. Interview Questions

1. Our traces show every span returning 200 and the answer was wrong. What is missing?

2. We sample 1% of traces. What share of our failures do we capture?

3. A team wants to log full prompts and the storage cost is prohibitive. What do you
   propose?

4. Our failure dashboard says schema violations are our biggest problem. What would you
   check before believing it?

5. Why keep a uniform sample at all if biased sampling captures 22 times more failures?

6. Our traces carry document identifiers and the investigation still stalled. What else
   needed to be true?

## 19. Research Questions

1. How correlated are trace fields in what they explain, and how far does that move the
   density ranking?

2. How large is the roadmap feedback effect from biased sampling over several planning
   cycles?

3. Can a request-time suspicion signal be built for the modes that currently have low
   recall — confidently wrong facts and subtle reasoning errors?

4. Should real-time and forensic traces be separate systems, and what does running both
   actually cost?

## 20. Chapter Summary

Standard tracing records control flow, and a semantic failure has normal control flow — so
timings, topology, status and identity together resolve **14%** of investigations
({{eq:attribution-needs-payload-not-timing}}). The fields that resolve them are inputs, and
inputs are payloads.

But resolution power is uncorrelated with size. `verifier score` resolves **17% in 40
bytes**; `retrieved doc text` resolves 31% in 46,000. Adding the four densest fields takes
resolution from **14% to 61%** for **$190 a month** against **$10,557** for everything
({{eq:cheap-fields-carry-most-attribution}}) — and they are non-content, so they are the
privacy-conservative choice too.

Uniform sampling captures exactly the sampling rate's share of failures, for every mode, by
construction ({{eq:uniform-sampling-misses-rare-failures}}). Biasing toward flagged
requests captures **22×** more.

And distorts the composition. The biased sample reproduces the detector's recall profile:
schema violations **1.9×** over-represented, subtle reasoning **5×** under
({{eq:biased-sampling-distorts-composition}}). A team reading counts off it sees schema
violations as the largest problem at 22,833 a day when confidently wrong facts are more
common at 15,120 — **the ranking is inverted and the data cannot say so.**

Both halves are instances of a pattern this part keeps producing: a tool built for one
class of failure, applied to another, producing output that looks authoritative and is
about something else. Tracing answers where the time went; sampling answers what the
distribution looks like. Both answers are correct and neither is the question, and the
danger is that both arrive formatted as though they were.

Carry forward: **the cheap fields carry most of the attribution**, and **investigate from
the biased sample, count from the uniform one**.

## 21. Further Reading

- {{cite:deshpande2025trail}} — 148 annotated traces and an 11% localisation ceiling;
  why trace structure has to carry the load.
- {{cite:sculley2015}} — entanglement, which is why inputs rather than control flow are
  what a trace must record.
- {{cite:breck2017}} — readiness tests centred on input validation, for the same reason.
- {{cite:paleyes2020deployment}} — obstacles at every stage, many of which are
  investigations that could not conclude.
