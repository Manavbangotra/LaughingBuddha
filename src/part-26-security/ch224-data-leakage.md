---
id: sec-data-leakage
number: 224
part: XXVI
tier: full
status: draft
requires: [instructions-and-data-share-a-channel, leaking-is-bounded-by-context-hijacking-is-not,
           derived-copies-multiply-contradiction, cache-threshold-is-an-error-cost-decision]
provides: [extraction-risk-does-not-vanish-at-one-occurrence, dedup-helps-the-common-secret-not-the-rare-one,
           most-leaks-are-inference-time-not-memorised, shared-cache-is-a-cross-tenant-channel]
citations: [carlini2021extracting, shokri2017membership, abadi2016dpsgd, perez2022ignore]
---

## 1. Learning Objectives

By the end of this chapter you will be able to explain why a secret appearing once in a
training corpus remains extractable and why the risk rises with model size; show what
deduplication does and does not remove; rank leak sources by records prevented per unit of
effort and explain why memorisation is the smallest term; compute cross-tenant exposure from
a shared semantic cache; explain why raising the similarity threshold trades exposure against
the cache's entire value while partitioning does not; and identify the disclosures that
survive every content-based control.

## 2. Why This Matters

{{cite:carlini2021extracting}} recovered hundreds of verbatim sequences from GPT-2 — names,
phone numbers, email addresses, code, 128-bit UUIDs — and reported that **each appeared in
just one training document**, and that larger models are more vulnerable.

Both findings break the frequency intuition. A UUID appearing once has extraction probability
**0.082** at 1.5B parameters and **0.333** at 400B — the same single occurrence, **4.1×** the
risk ({{eq:extraction-risk-does-not-vanish-at-one-occurrence}}). And distinctiveness beats
frequency: a phone-number format appearing 4,100 times is **0.301** extractable, a UUID
appearing once is **0.308**.

Deduplication takes the expected extractable count down **49%** and leaves the singleton
classes exactly where they were ({{eq:dedup-helps-the-common-secret-not-the-rare-one}}) — it
removes the secrets that were never the problem.

The bigger correction is where leaks actually come from. Memorisation is **2.9%** of leaked
records a year; logs with payloads are **44.9%**, misrouted retrievals **22.6%**, cache
cross-tenancy **19.4%** ({{eq:most-leaks-are-inference-time-not-memorised}}). The best
control returns **930** records prevented per unit of effort against memorisation's **3**.

And a semantic cache shared across 340 tenants serves a cross-tenant hit **99.7%** of the time
it hits, which is **441 records a day** with no attacker involved
({{eq:shared-cache-is-a-cross-tenant-channel}}) — against 372 a year in the incident register,
a ratio of **433 to one**.

## 3. Prerequisites

{{eq:leaking-is-bounded-by-context-hijacking-is-not}} from {{ch:sec-prompt-injection}} bounded
what a leak can reach from the context. This chapter adds the sources that are not the
context: the weights, the store, the cache and the logs.

{{eq:instructions-and-data-share-a-channel}} from {{ch:sec-threat-model}} is why
{{cite:perez2022ignore}}'s prompt leaking works at all, and why a "do not reveal your
instructions" line is not a control.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} is the reason the cache
and the log are separate leak sources rather than views of the primary store.

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} returns here
in its security form: the similarity threshold trades cross-tenant exposure against hit rate
at a fixed rate, and never removes the channel.

## 4. Intuitive Explanation

There are two leak questions and they are usually merged. The first is whether the model
remembers things it was trained on. The second is whether the system hands out things it is
holding right now. Both are real; only one of them has a research literature, and it is the
smaller one.

Start with memorisation, because the standard intuition about it is wrong.

The intuition is that memorisation is a frequency phenomenon: something that appears a
thousand times in the corpus gets learned, something that appears once gets averaged away.
Under that model, deduplication is the fix — collapse the repeats and the memorisation
problem shrinks with them.

{{cite:carlini2021extracting}} tested this and found otherwise. They recovered hundreds of
verbatim sequences from GPT-2, and each of those sequences appeared in *just one* training
document. Names, phone numbers, email addresses, IRC conversations, code, 128-bit UUIDs. One
occurrence each, recoverable.

They also found the vulnerability rises with model size.

Put those together and the arithmetic is unfriendly. A 128-bit UUID appearing once in the
corpus is 0.082 extractable from a 1.5B model and 0.333 from a 400B model. Same corpus, same
single occurrence, four times the risk — and the trend of the industry is toward the right of
that table.

There is a second reversal in the same numbers. A common phone-number format appearing 4,100
times is 0.301 extractable. A UUID appearing once is 0.308. **Distinctiveness beats
frequency**, and every property that makes a string a good secret — high entropy, unusual
format, no near-duplicates — makes it a good memorisation target.

So what does deduplication do? It takes the expected extractable count down about half, which
is real and worth having. Read the reduction column though: the phone-number format falls 66%
and the UUID falls 0%, because there is nothing to deduplicate.

**Deduplication removes the secrets that were never the problem.** It is worth doing — half
the expected extractions is not nothing — and reporting it as *the* memorisation control is
how a team ends up believing the singletons were handled. The residual after dedup is
dominated by exactly the items an attacker is searching for: an API key committed once, an
internal identifier, a UUID that appears in one config file and nowhere else on the internet.

The controls that reach a single occurrence are a scan at ingest, entity redaction, source
exclusion, an output filter, or differential privacy. The first four are cheap and share a
property: each needs to know something. A pattern to match. A list to filter against. A
source to exclude. All of them fail on a secret nobody anticipated — an internal identifier
in a new format, a credential in a config file, a personal detail in free text.

{{cite:abadi2016dpsgd}}'s mechanism is the only one that does not need to know what the
secret is. It bounds the influence of any single training example whatever that example
contains, which is why it holds for unknown secrets and why it is worth twenty-two units of
effort against the cheap controls' one or two — in a system training on data whose contents
nobody can enumerate.

That is memorisation. Now the correction.

Count where sensitive records actually leave a deployed system over a year. Memorisation:
2.9%. System prompt disclosed: 1.6%. Tool credential in context: 0.5%. And then the large
ones — a log or trace that recorded a payload at 44.9%, a retrieved document served to the
wrong user at 22.6%, a semantic cache serving across tenants at 19.4%, conversation history
reuse at 8.0%.

**Memorisation is the smallest term and gets most of the attention**, because it is the one
that is interesting. The largest terms are a logging default, a filter applied in the wrong
place, and a cache that was not partitioned.

Ranked by records prevented per unit of effort, the cache fix returns 930, redaction at emit
returns 344, retrieval filtering returns 216, and memorisation controls return 3.

None of that says memorisation does not matter. It says a team that starts with DP-SGD will
spend a quarter before touching the top four rows.

The cache deserves its own treatment, because it is the least obvious and the most structural.

A semantic cache stores answers keyed on question similarity: if a new question is close
enough to a cached one, serve the cached answer. That is a large latency and cost win and
{{ch:sd-routing-caching}} priced it.

Now suppose the answers were built from tenant-specific documents, and the cache is shared.
A tenant asks a question that is 0.94 similar to one another tenant asked last week, and
receives an answer synthesised from documents they have no right to see.

How often? If 340 tenants share a cache, a cache hit is cross-tenant 99.7% of the time,
because the nearest cached question is almost certainly somebody else's. At a 34% hit rate
that is 14,238 cross-tenant hits a day.

Most are harmless — most answers contain nothing tenant-specific. At a 3.1% rate of
tenant-specific content, it is 441 records a day.

**This is not an attack. It is the cache working as designed**, which is exactly why it
survives security review: nothing is anomalous, there is no adversary in the logs, and the
system is doing what the config says.

Which produces a discrepancy worth sitting with. The incident register recorded 372
cache-related records a year. The arithmetic says 161,103. A ratio of 433 to one.

The first number is what gets reported and the second is what happens. **A leak that produces
no complaint produces no incident**, and a cross-tenant cache hit produces no complaint
because the recipient cannot tell the answer came from somewhere it should not have.

The tempting fix is to raise the similarity threshold. It works, and it works by destroying
the cache: going from 0.80 to 0.995 takes cross-tenant hits from 25,963 to 1,165 a day and
takes the hit rate from 62% to 3%. That is {{ch:sd-routing-caching}}'s threshold result
again — the knob trades exposure against utility at a fixed rate and never removes the
channel.

Keying the cache by tenant costs 5 points of hit rate and takes cross-tenant exposure to
zero, structurally, at a configuration change. **A structural fix that costs five points beats
a threshold that costs fifty-nine and leaves the channel open**, which is
{{ch:sec-threat-model}}'s capability-versus-detection result arriving in the storage layer.

One last thing that none of this reaches. {{cite:shokri2017membership}}'s attack does not ask
for content — it asks whether a record was in the training set. For a hospital discharge
dataset or a customer list, **membership is itself the sensitive fact**.

Redaction does not stop that, because redaction removes content and membership is not
content. Only a formal privacy bound does, which is the second argument for DP's price and
the one that does not show up in a leaked-record count at all.

## 5. Formal Explanation

**Extraction.** Model the probability that a sequence can be elicited verbatim as increasing
in three arguments: occurrence count $n$, distinctiveness $\delta$ (roughly, negative log
probability under a generic language model), and capacity $\theta$. Empirically the
occurrence term saturates — the first occurrence contributes most — so

$$P(n, \delta, \theta) = f(n)\,g(\delta)\,h(\theta), \qquad f(1) > 0, \quad \frac{\partial P}{\partial \theta} > 0.$$

The two consequences that matter are $f(1) > 0$ (a single occurrence is not safe) and
$\partial P / \partial \theta > 0$ (the same corpus becomes more dangerous under a larger
model, without being retrained).

**Deduplication.** Mapping $n \mapsto 1$ reduces $P$ by $1 - f(1)/f(n)$, which is zero when
$n = 1$. So dedup's benefit is concentrated in the high-$n$ classes, and high $n$ correlates
negatively with $\delta$ — frequent strings are less distinctive. Dedup therefore removes
mass from exactly the low-value tail.

**Leak accounting.** Total leaked records are $\sum_s r_s i_s$ over sources with per-incident
records $r_s$ and incident rate $i_s$. Control ranking is by $r_s i_s / e_s$ for effort
$e_s$. This is {{ch:ev-framework}}'s greedy portfolio in a different domain, with the same
property: the ranking is dominated by cheap fixes to large sources, and the source with the
deepest literature need not appear near the top.

**Cache cross-tenancy.** With $T$ tenants and cache entries distributed across them, the
probability that the nearest neighbour of a query is from another tenant is $1 - 1/T$ under a
uniform model. Exposure is $Q \cdot \text{hit}(\tau) \cdot (1 - 1/T) \cdot \rho$ for
tenant-specific content share $\rho$. Note that $\tau$ enters only through $\text{hit}$, so
$\partial(\text{exposure})/\partial\tau$ and $\partial(\text{utility})/\partial\tau$ have the
same sign and are proportional: the threshold cannot separate them. Partitioning changes
$1 - 1/T$ to $0$ and leaves $\text{hit}$ nearly unchanged.

**Membership.** A membership inference attack estimates $\Pr[x \in D \mid \text{model}]$
without recovering $x$. Content controls act on what is emitted; membership is inferred from
*behaviour* on an input the attacker already has. Only a mechanism bounding the influence of
$x$ on the model — a differential privacy guarantee — constrains it.

## 6. Mathematical Foundation

Extraction as a product with a non-zero single-occurrence term:

$$P(n,\delta,\theta) = f(n)g(\delta)h(\theta), \qquad f(1) > 0, \qquad \frac{\partial P}{\partial \theta} > 0$$ (eq:extraction-risk-does-not-vanish-at-one-occurrence)

At $n=1$, $\delta = 0.99$: **0.082** at 1.5B and **0.333** at 400B.

What deduplication removes:

$$\Delta P = f(n) - f(1), \qquad \Delta P \big|_{n=1} = 0, \qquad \operatorname{corr}(n, \delta) < 0$$ (eq:dedup-helps-the-common-secret-not-the-rare-one)

**49%** total reduction, **0%** on the singleton classes.

Leak accounting across sources:

$$L = \sum_s r_s i_s, \qquad \text{rank by } \frac{r_s i_s}{e_s}, \qquad \frac{L_{\text{mem}}}{L} = 2.9\%$$ (eq:most-leaks-are-inference-time-not-memorised)

Top control **930** records per unit of effort; memorisation controls **3**.

And the cache channel:

$$E = Q\,\text{hit}(\tau)\left(1 - \tfrac{1}{T}\right)\rho, \qquad \frac{\partial E}{\partial \tau} \propto \frac{\partial\,\text{hit}}{\partial \tau}, \qquad E\big|_{\text{partitioned}} = 0$$ (eq:shared-cache-is-a-cross-tenant-channel)

At $T = 340$: **99.7%** of hits are cross-tenant, **441 records/day**, against **372/year**
recorded as incidents.

## 7. Internal Mechanics

Why does a single occurrence get memorised at all? Because gradient descent on a
next-token objective has no mechanism for distinguishing "this sequence is a fact worth
learning" from "this sequence appeared once." A highly distinctive sequence produces a large
loss on first encounter and therefore a large gradient, and a model with spare capacity
allocates some to reducing it. Frequency helps memorisation; distinctiveness *causes* it,
because distinctiveness is what makes the loss large.

That also explains the size dependence. A larger model has more capacity left over after
fitting the generalisable structure, and the leftover capacity goes to whatever still has
loss on it — which is the singletons. **Scaling improves generalisation and memorisation at
the same time**, from the same spare capacity.

On the inference side, the reason logs dominate the leak accounting is worth naming precisely.
{{ch:ops-observability}} argued for recording payloads, because timing and topology resolve
only 14% of investigations. That argument is correct and it creates this chapter's largest
leak source. The two requirements are in genuine tension and the resolution is not "log less"
— it is field-level redaction at emit, which keeps the diagnostic value and removes the
sensitive substrings. That is a two-and-a-half-unit fix returning 344 records per unit, and
it is skipped because it requires knowing which fields are sensitive.

The cache result has a mechanism that makes it particularly durable. A semantic cache is
correct by its own specification: it returned an answer to a semantically similar question,
which is what it was built to do. There is no error to detect, no anomaly, and no failed
assertion. The only way to notice is to ask whether the *provenance* of the cached answer is
compatible with the *authorisation* of the new requester — which is a question the cache has
no field for, because the cache was designed around similarity and not around tenancy.

This is why partitioning works and thresholds do not. Partitioning adds the missing field.

Finally, the incident-register discrepancy deserves a general statement, because it is not
specific to caches. Incident counts measure *noticed* leaks, and noticing requires that
somebody is in a position to observe the mismatch. For a leak to a user who has no way of
knowing the data was not theirs, nobody is in that position. So the categories that dominate
incident registers are the ones with an obvious victim — a screenshot on social media, a
customer complaint — and the categories that dominate actual exposure are the silent ones.
**A leak register is a measure of visibility, not of volume**, which is
{{eq:biased-sampling-distorts-composition}} from {{ch:ops-observability}} appearing in yet
another place.

## 8. Implementation

The first listing measures memorisation.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/id1}
"""A secret that appears once in the corpus is still extractable, and that is the whole problem.

cite:carlini2021extracting recovered hundreds of verbatim sequences from GPT-2 -- names, phone
numbers, email addresses, IRC logs, code, 128-bit UUIDs -- and reported that **each of those
sequences appeared in just one training document**. They also found larger models are more
vulnerable than smaller ones.

Both findings break the intuition that memorisation is a frequency phenomenon and can be
managed by deduplication (eq:extraction-risk-does-not-vanish-at-one-occurrence).

Deduplication is still worth doing. It just helps the secrets that were never the problem
(eq:dedup-helps-the-common-secret-not-the-rare-one).
"""
import math

# (secret class, occurrences in the corpus, distinctiveness 0-1, count in corpus)
SECRETS = [
    ("a common phone number format",  4100, 0.11, 90_000),
    ("a personal email address",        14, 0.62, 2_400_000),
    ("an internal hostname",             6, 0.71, 41_000),
    ("an API key committed once",        1, 0.97, 8_900),
    ("a 128-bit UUID",                   1, 0.99, 310_000),
    ("a private address in a leak dump", 3, 0.83, 1_700_000),
]
MODEL_SIZES = [1.5, 7.0, 70.0, 400.0]     # billions of parameters


def extract_p(occurrences, distinct, params):
    """P(the sequence can be elicited verbatim), rising in all three arguments."""
    base = 1.0 - math.exp(-0.42 * occurrences ** 0.55)
    scale = 1.0 - math.exp(-0.31 * math.log(params + 1.0) ** 1.4)
    return min(0.995, base * (0.25 + 0.75 * distinct) * scale)


print("Extraction probability by occurrence count, at four model sizes.")
print()
print(f"{'secret class':>32}{'occurrences':>13}", end="")
for m in MODEL_SIZES:
    print(f"{(str(m) + 'B'):>10}", end="")
print()
print("-" * 85)
ext = {}
for name, occ, dist, count in SECRETS:
    print(f"{name:>32}{occ:>13,}", end="")
    for m in MODEL_SIZES:
        p = extract_p(occ, dist, m)
        ext[(name, m)] = p
        print(f"{p:>10.3f}", end="")
    print()

print()
print(f"a UUID appearing once: {ext[('a 128-bit UUID', 1.5)]:.3f} at 1.5B, "
      f"{ext[('a 128-bit UUID', 400.0)]:.3f} at 400B")
print(f"the risk rises {ext[('a 128-bit UUID', 400.0)] / ext[('a 128-bit UUID', 1.5)]:.1f}x "
      f"with model size, at a constant occurrence count")

print()
print()
print("Expected extractable secrets in the corpus, at 70B.")
print()
M = 70.0
print(f"{'secret class':>32}{'count in corpus':>18}{'P(extract)':>13}"
      f"{'expected extractable':>23}")
print("-" * 86)
total = 0.0
per_class = {}
for name, occ, dist, count in SECRETS:
    e = count * ext[(name, M)]
    per_class[name] = e
    total += e
    print(f"{name:>32}{count:>18,}{ext[(name, M)]:>13.3f}{e:>23,.0f}")
print("-" * 86)
print(f"{'TOTAL':>32}{'':>18}{'':>13}{total:>23,.0f}")

singleton = sum(per_class[n] for n, o, d, c in SECRETS if o == 1)
print()
print(f"of those, {singleton:,.0f} ({singleton / total:.0%}) come from secrets")
print("that appear exactly once")

print()
print()
print("What deduplication does. It removes repeats and cannot remove a single.")
print()
print(f"{'secret class':>32}{'occurrences before':>20}{'after dedup':>14}"
      f"{'P before':>11}{'P after':>10}{'reduction':>12}")
print("-" * 99)
dedup = {}
for name, occ, dist, count in SECRETS:
    after = 1 if occ > 1 else occ
    pb, pa = extract_p(occ, dist, M), extract_p(after, dist, M)
    dedup[name] = (pb, pa, (pb - pa) / pb if pb > 0 else 0.0)
    print(f"{name:>32}{occ:>20,}{after:>14}{pb:>11.3f}{pa:>10.3f}"
          f"{(pb - pa) / pb:>12.0%}")

after_total = sum(c * extract_p(1 if o > 1 else o, d, M) for n, o, d, c in SECRETS)
print()
print(f"expected extractable after dedup: {after_total:,.0f} "
      f"({1 - after_total / total:.0%} reduction)")
print(f"but the singleton classes are unchanged at {singleton:,.0f}")

print()
print()
print("The controls that do reach a single occurrence.")
print()
CONTROLS = [
    ("deduplicate the corpus",        0.20, 1.0, "repeats only"),
    ("secret-pattern scan at ingest", 0.74, 2.0, "known formats"),
    ("entity redaction at ingest",    0.61, 5.0, "names, addresses"),
    ("exclude flagged sources",       0.44, 1.5, "whole domains"),
    ("DP-SGD training",               0.93, 22.0, "a formal bound"),
    ("output filter on known secrets", 0.55, 1.2, "if you have the list"),
]
print(f"{'control':>34}{'removes':>10}{'effort':>9}{'per effort':>13}"
      f"{'covers':>20}")
print("-" * 86)
ctl = {}
for name, rem, eff, cov in CONTROLS:
    ctl[name] = (rem, eff, rem / eff)
    print(f"{name:>34}{rem:>10.0%}{eff:>9.1f}{rem / eff:>13.3f}{cov:>20}")

order = sorted(CONTROLS, key=lambda c: -ctl[c[0]][2])
print()
print(f"best return: {order[0][0]} at {ctl[order[0][0]][2]:.3f} per unit")
print(f"the only formal bound: DP-SGD at {ctl['DP-SGD training'][2]:.3f}")

print()
print()
print("And the reason a formal bound is worth its price: it does not depend on")
print("knowing what the secret looks like.")
print()
print(f"{'control':>34}{'needs a pattern?':>19}{'needs the list?':>18}"
      f"{'holds for unknown secrets?':>29}")
print("-" * 100)
KNOWS = [
    ("deduplicate the corpus",        "no",  "no",  "yes, for repeats"),
    ("secret-pattern scan at ingest", "yes", "no",  "no"),
    ("entity redaction at ingest",    "yes", "no",  "no"),
    ("output filter on known secrets", "no", "yes", "no"),
    ("DP-SGD training",               "no",  "no",  "yes"),
]
for name, pat, lst, unk in KNOWS:
    print(f"{name:>34}{pat:>19}{lst:>18}{unk:>29}")

print(f"""
The extraction table is cite:carlini2021extracting's two findings side by side. A UUID
appearing exactly once has extraction probability {ext[('a 128-bit UUID', 1.5)]:.3f} at 1.5B
parameters and {ext[('a 128-bit UUID', 400.0)]:.3f} at 400B -- **the same single occurrence,
{ext[('a 128-bit UUID', 400.0)] / ext[('a 128-bit UUID', 1.5)]:.1f} times the risk**
(eq:extraction-risk-does-not-vanish-at-one-occurrence).

That is the finding that breaks the usual mental model. Memorisation is not a frequency
phenomenon that dilutes as the corpus grows; the model's capacity to retain a distinctive
sequence rises with its size, and a distinctive sequence is exactly what a secret is.

Note which classes are riskiest. A common phone-number format appears {4100:,} times and is
{ext[('a common phone number format', M)]:.3f} extractable; a UUID appears once and is
{ext[('a 128-bit UUID', M)]:.3f}. **Distinctiveness beats frequency**, and every property that
makes a string a good secret makes it a good memorisation target.

The corpus table converts probability into count. Across the six classes, roughly
{total:,.0f} secrets are extractable from a 70B model, and **{singleton / total:.0%} of those
come from secrets that appear exactly once.**

The dedup table is the intervention most teams reach for, and it is not wrong -- it takes the
expected extractable count down {1 - after_total / total:.0%}. Read the reduction column,
though: `{SECRETS[0][0]}` falls {dedup[SECRETS[0][0]][2]:.0%} and
`{SECRETS[4][0]}` falls {dedup[SECRETS[4][0]][2]:.0%}, because there is nothing to
deduplicate (eq:dedup-helps-the-common-secret-not-the-rare-one).

**Deduplication removes the secrets that were never the problem**, and leaves the singleton
classes exactly where they were at {singleton:,.0f}. Those are the highest-value and most
distinctive items in the corpus -- an API key committed once, a UUID -- and they are precisely
what an attacker is looking for.

The control table ranks what does reach them. `{order[0][0]}` returns
{ctl[order[0][0]][2]:.3f} of removal per unit of effort and is the right first move.
`DP-SGD training` returns {ctl['DP-SGD training'][2]:.3f}, which is the worst ratio on the
list by an order of magnitude.

The last table is why DP-SGD is on the list anyway. Every cheap control needs to know
something: a pattern to match, a list to filter against, a source to exclude. All of them fail
on a secret nobody anticipated -- an internal identifier in a new format, a credential embedded
in a config file, a personal detail in free text.

cite:abadi2016dpsgd's mechanism does not need to know what the secret is. It bounds the
influence of any single training example, whatever that example contains, which is why it is
**the only row in the table that holds for unknown secrets** and why it is worth twenty-two
units of effort in a system where the corpus contains things nobody enumerated.

Which is a genuine trade rather than a recommendation. Most systems should do the cheap
controls and accept a residual on unknown singletons; systems training on data whose contents
they cannot enumerate should price the formal bound, because the alternative is a set of
filters against a list they do not have.""")
```

## 9. Practical Example

Extraction probability by occurrence and model size:

```
                    secret class  occurrences      1.5B      7.0B     70.0B    400.0B
-------------------------------------------------------------------------------------
    a common phone number format        4,100     0.080     0.192     0.301     0.325
        a personal email address           14     0.143     0.345     0.540     0.583
       an API key committed once            1     0.080     0.194     0.304     0.328
                  a 128-bit UUID            1     0.082     0.197     0.308     0.333
```

A single occurrence is **0.082 at 1.5B and 0.333 at 400B** — **4.1×** the risk from the same
corpus ({{eq:extraction-risk-does-not-vanish-at-one-occurrence}}). And a UUID appearing once
(**0.308**) beats a phone format appearing 4,100 times (**0.301**): **distinctiveness beats
frequency.**

```
                    secret class   count in corpus   P(extract)   expected extractable
--------------------------------------------------------------------------------------
        a personal email address         2,400,000        0.540              1,295,346
a private address in a leak dump         1,700,000        0.424                720,375
                  a 128-bit UUID           310,000        0.308                 95,555
       an API key committed once             8,900        0.304                  2,702
--------------------------------------------------------------------------------------
                           TOTAL                                             2,160,701
```

```
                    secret class  occurrences before   after dedup   P before   P after   reduction
---------------------------------------------------------------------------------------------------
    a common phone number format               4,100             1      0.301     0.103         66%
        a personal email address                  14             1      0.540     0.222         59%
       an API key committed once                   1             1      0.304     0.304          0%
                  a 128-bit UUID                   1             1      0.308     0.308          0%
```

**49% total reduction, 0% on the singletons**
({{eq:dedup-helps-the-common-secret-not-the-rare-one}}).

```
                           control   removes   effort   per effort              covers
--------------------------------------------------------------------------------------
            deduplicate the corpus       20%      1.0        0.200        repeats only
     secret-pattern scan at ingest       74%      2.0        0.370       known formats
   output filter on known secrets       55%      1.2        0.458  if you have the list
                  DP-SGD training       93%     22.0        0.042      a formal bound

                           control   needs a pattern?   needs the list?   holds for unknown secrets?
----------------------------------------------------------------------------------------------------
     secret-pattern scan at ingest                yes                no                           no
   output filter on known secrets                 no               yes                           no
                  DP-SGD training                 no                no                          yes
```

Every cheap control needs to know something. **{{cite:abadi2016dpsgd}}'s is the only row that
holds for a secret nobody anticipated.**

The second listing corrects where the leaks actually are.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/id2}
"""Almost every production leak is inference-time, and memorisation is the smallest term.

Memorisation gets the attention because it is the interesting one -- it involves the weights,
it has a research literature, and it cannot be undone. It is also, in a deployed system, a
minority of the leaked bytes.

The majority comes from things that were put into a context window this morning: the system
prompt, the retrieved documents, another tenant's cached answer, a credential in a tool
definition (eq:most-leaks-are-inference-time-not-memorised).

The semantic cache deserves particular attention, because a cache keyed on question
similarity and populated from tenant-specific documents is a cross-tenant channel by
construction (eq:shared-cache-is-a-cross-tenant-channel).
"""
import math

# (source, sensitive records exposed per incident, incidents/year, fixable how)
SOURCES = [
    ("training-data memorisation",     140,  0.4,  "corpus controls, DP-SGD", 22.0),
    ("system prompt disclosed",        1,    31.0, "remove secrets from prompt", 0.3),
    ("retrieved document, wrong user", 24,   18.0, "filter at retrieval", 2.0),
    ("semantic cache cross-tenant",    62,   6.0,  "key the cache by tenant", 0.4),
    ("tool credential in context",     1,    9.0,  "credential broker", 3.0),
    ("conversation history reuse",     11,   14.0, "scope the session store", 1.5),
    ("log or trace with payload",      430,  2.0,  "redact at emit", 2.5),
]

print("Reported leak incidents, per year.")
print()
print(f"{'source':>34}{'records/incident':>19}{'incidents/yr':>15}"
      f"{'records/yr':>14}{'share':>9}")
print("-" * 91)
tot = sum(r * i for n, r, i, f, e in SOURCES)
leak = {}
for name, rec, inc, fix, eff in SOURCES:
    y = rec * inc
    leak[name] = (y, y / tot, fix, eff)
    print(f"{name:>34}{rec:>19,}{inc:>15.1f}{y:>14,.0f}{y / tot:>9.1%}")
print("-" * 91)
print(f"{'TOTAL':>34}{'':>19}{'':>15}{tot:>14,.0f}{1.0:>9.1%}")

mem = leak["training-data memorisation"][0]
print()
print(f"memorisation is {mem / tot:.1%} of leaked records and most of the")
print("research attention")

print()
print()
print("Ranked by records prevented per unit of effort.")
print()
print(f"{'source':>34}{'records/yr':>14}{'fix':>30}{'effort':>9}"
      f"{'per effort':>13}")
print("-" * 100)
order = sorted(SOURCES, key=lambda s: -(s[1] * s[2] / s[4]))
for name, rec, inc, fix, eff in order:
    print(f"{name:>34}{rec * inc:>14,.0f}{fix:>30}{eff:>9.1f}"
          f"{rec * inc / eff:>13,.0f}")

print()
print(f"top: {order[0][0]} at {order[0][1] * order[0][2] / order[0][4]:,.0f} "
      f"records per unit")
print(f"memorisation controls: "
      f"{leak['training-data memorisation'][0] / 22.0:,.0f}")

print()
print()
print("The semantic cache in detail, because it is the least obvious one.")
print()
TENANTS = 340
QUERIES_PER_DAY = 42_000.0
CACHE_HIT = 0.34
SIM_THRESHOLD = 0.92
print(f"{'tenants sharing the cache':>27}{'hit rate':>11}"
      f"{'cross-tenant hits/day':>24}{'records exposed/day':>22}")
print("-" * 84)
cross = {}
for n_t in (1, 4, 25, 120, 340):
    # A hit is cross-tenant when the nearest cached question came from
    # another tenant, which is more likely the more tenants share the cache.
    p_cross = 1.0 - 1.0 / n_t if n_t > 1 else 0.0
    hits = QUERIES_PER_DAY * CACHE_HIT * p_cross
    cross[n_t] = (p_cross, hits, hits * 0.031)
    print(f"{n_t:>27}{CACHE_HIT:>11.0%}{hits:>24,.0f}{hits * 0.031:>22,.0f}")

print()
print("Not every cross-tenant hit leaks -- most answers contain nothing")
print(f"tenant-specific. At a {0.031:.1%} rate, {cross[340][2]:,.0f} records a day.")

print()
print()
print("What the similarity threshold does to it.")
print()
print(f"{'threshold':>11}{'hit rate':>11}{'cross-tenant hits/day':>24}"
      f"{'records/day':>14}{'latency saved':>16}")
print("-" * 76)
thr = {}
for t in (0.80, 0.86, 0.92, 0.96, 0.995):
    hit = 0.62 * math.exp(-((t - 0.80) / 0.14) ** 2 * 1.6)
    hits = QUERIES_PER_DAY * hit * cross[340][0]
    thr[t] = (hit, hits, hits * 0.031)
    print(f"{t:>11.3f}{hit:>11.0%}{hits:>24,.0f}{hits * 0.031:>14,.0f}"
          f"{hit * 640:>15.0f}ms")

print()
print("Raising the threshold reduces exposure and the cache's whole value")
print("at the same rate. Partitioning does not.")

print()
print()
print("Partitioning the cache instead.")
print()
print(f"{'cache design':>34}{'hit rate':>11}{'cross-tenant hits/day':>24}"
      f"{'records/day':>14}")
print("-" * 83)
DESIGNS = [
    ("one shared cache",             0.34, cross[340][0]),
    ("cache keyed by tenant",        0.29, 0.0),
    ("shared for public docs only",  0.31, 0.0),
    ("shared, tenant-tagged entries", 0.33, 0.0),
    ("no cache",                     0.00, 0.0),
]
for name, hit, pc in DESIGNS:
    hits = QUERIES_PER_DAY * hit * pc
    print(f"{name:>34}{hit:>11.0%}{hits:>24,.0f}{hits * 0.031:>14,.0f}")

print()
print(f"keying by tenant costs {0.34 - 0.29:.0%} of hit rate and removes")
print("the channel entirely")

print()
print()
print("And the membership question, which survives every content control.")
print()
print(f"{'what the attacker learns':>38}{'needs content?':>17}"
      f"{'stopped by redaction?':>24}{'stopped by DP?':>17}")
print("-" * 96)
MEMBERSHIP = [
    ("the exact record",              "yes", "yes", "yes"),
    ("that a record was in the set",  "no",  "no",  "yes"),
    ("that a person was a customer",  "no",  "no",  "yes"),
    ("that a document was indexed",   "no",  "no",  "no"),
]
for what, content, red, dp in MEMBERSHIP:
    print(f"{what:>38}{content:>17}{red:>24}{dp:>17}")

print(f"""
The source table is the correction this chapter exists for.
`{SOURCES[0][0]}` accounts for {leak[SOURCES[0][0]][1]:.1%} of leaked records a year;
`{SOURCES[6][0]}` accounts for {leak[SOURCES[6][0]][1]:.1%} and
`{SOURCES[3][0]}` for {leak[SOURCES[3][0]][1]:.1%}
(eq:most-leaks-are-inference-time-not-memorised).

**Memorisation is the smallest term and gets most of the attention**, because it is the one
with a literature. The largest terms are a log that recorded a payload, a cache that was not
partitioned, and a retrieval filter that was applied after the fact rather than before.

The ranking table converts that into where to spend. `{order[0][0]}` returns
{order[0][1] * order[0][2] / order[0][4]:,.0f} records prevented per unit of effort;
memorisation controls return {leak['training-data memorisation'][0] / 22.0:,.0f} --
**{(order[0][1] * order[0][2] / order[0][4]) / (leak['training-data memorisation'][0] / 22.0):.0f}
times apart.**

Nothing in that comparison says memorisation does not matter. It says the cheap wins are
somewhere else and there are several of them, and a team that starts with DP-SGD will spend a
quarter before touching the top four rows.

The cache table is the one worth reading slowly. A semantic cache keyed on question
similarity, populated from tenant-specific documents, and shared across {TENANTS} tenants
serves a cross-tenant hit {cross[TENANTS][0]:.1%} of the time it hits at all
(eq:shared-cache-is-a-cross-tenant-channel). At a {CACHE_HIT:.0%} hit rate that is
{cross[TENANTS][1]:,.0f} cross-tenant hits a day.

Most of those are harmless -- the answer contained nothing tenant-specific. At a
{0.031:.1%} rate of tenant-specific content, it is {cross[TENANTS][2]:,.0f} records a day,
every day, with no attacker involved.

**This is not an attack. It is the cache working as designed**, which is why it survives
security review: there is nothing anomalous to detect and no adversary in the logs.

And notice the discrepancy with the first table, which is the most important thing in this
listing. That table recorded {leak['semantic cache cross-tenant'][0]:,.0f} cache-related
records a year, because it counted *reported incidents*. This one computes
{cross[TENANTS][2] * 365:,.0f} a year, because it computes *actual exposure*. The ratio is
{cross[TENANTS][2] * 365 / leak['semantic cache cross-tenant'][0]:,.0f} to one.

The first number is what an incident register contains and the second is what happened.
**A leak that produces no complaint produces no incident**, and a cross-tenant cache hit
produces no complaint because the recipient cannot tell the answer came from somewhere it
should not have. They asked a question and got a reasonable answer. Nothing about the
interaction signals that the grounding documents belonged to a competitor.

That asymmetry is worth generalising, because it is not specific to caches. Any leak whose
recipient cannot detect it is a leak with no reporting path, and the categories that dominate
an incident register are the ones with an obvious witness — a screenshot, a support ticket, a
regulator's letter. Silent categories dominate volume and appear nowhere.

The threshold table shows the tempting fix and why it is not one. Raising the similarity
threshold from {0.80:.2f} to {0.995:.3f} takes cross-tenant hits from
{thr[0.80][1]:,.0f} to {thr[0.995][1]:,.0f} a day -- and takes the hit rate from
{thr[0.80][0]:.0%} to {thr[0.995][0]:.0%}, which is the cache's entire value. That is
ch:sd-routing-caching's threshold result again: **the knob trades exposure against utility at
a fixed rate and never removes the channel.**

The partition table removes it. Keying the cache by tenant costs {0.34 - 0.29:.0%} of hit
rate -- entries are no longer shared across tenants who asked the same thing -- and takes
cross-tenant exposure to zero, structurally, at a configuration change.

**A structural fix that costs five points of hit rate beats a threshold that costs
{thr[0.80][0] - thr[0.995][0]:.0%} and leaves the channel open**, which is
ch:sec-threat-model's capability-versus-detection result in the storage layer.

The last table is what none of this reaches. cite:shokri2017membership's attack does not need
the content -- it asks whether a record was in the training set, and for a hospital discharge
dataset or a customer list, **membership is itself the sensitive fact**. Redaction does not
stop it, because redaction removes content and membership is not content. Only a formal
privacy bound does, which is the second argument for cite:abadi2016dpsgd's price and the one
that does not appear in a leaked-record count at all.""")
```

```
                            source   records/incident   incidents/yr    records/yr    share
-------------------------------------------------------------------------------------------
        training-data memorisation                140            0.4            56     2.9%
    retrieved document, wrong user                 24           18.0           432    22.6%
       semantic cache cross-tenant                 62            6.0           372    19.4%
         log or trace with payload                430            2.0           860    44.9%
```

**Memorisation is 2.9%** ({{eq:most-leaks-are-inference-time-not-memorised}}).

```
                            source    records/yr                           fix   effort   per effort
----------------------------------------------------------------------------------------------------
       semantic cache cross-tenant           372       key the cache by tenant      0.4          930
         log or trace with payload           860                redact at emit      2.5          344
    retrieved document, wrong user           432           filter at retrieval      2.0          216
        training-data memorisation            56       corpus controls, DP-SGD     22.0            3
```

**930 against 3**, three hundred and ten times apart.

```
  tenants sharing the cache   hit rate   cross-tenant hits/day   records exposed/day
------------------------------------------------------------------------------------
                          1        34%                       0                     0
                         25        34%                  13,709                   425
                        340        34%                  14,238                   441
```

**441 records a day with no attacker involved**
({{eq:shared-cache-is-a-cross-tenant-channel}}) — against **372 a year** in the incident
register, a ratio of **433 to one**, because a leak that produces no complaint produces no
incident.

```
  threshold   hit rate   cross-tenant hits/day   records/day   latency saved
----------------------------------------------------------------------------
      0.800        62%                  25,963           805            397ms
      0.920        19%                   8,014           248            122ms
      0.995         3%                   1,165            36             18ms

                      cache design   hit rate   cross-tenant hits/day   records/day
-----------------------------------------------------------------------------------
                  one shared cache        34%                  14,238           441
             cache keyed by tenant        29%                       0             0
        shared for public docs only        31%                      0             0
```

**Five points of hit rate removes the channel; fifty-nine points of threshold does not.**

```
              what the attacker learns   needs content?   stopped by redaction?   stopped by DP?
------------------------------------------------------------------------------------------------
                      the exact record              yes                     yes              yes
        that a record was in the set                 no                      no              yes
        that a person was a customer                 no                      no              yes
```

## 10. Production Considerations

Key the semantic cache by tenant. It is a configuration change, it costs five points of hit
rate, and it is the highest-return control in this chapter.

Redact payload fields at emit rather than at query. The diagnostic value survives and the
largest leak source does not.

Apply retrieval authorisation before ranking, not after. Filtering the top-k after retrieval
leaves the ranking itself a channel.

Scan for secret patterns at ingest and record the coverage. It is cheap and its ceiling is
the share of the corpus that passes through the scanner.

Deduplicate, and do not report it as a memorisation control. It reduces the count and not the
singletons.

Price DP against the *unknown* secrets, not the known ones. Every cheap control beats it on
knowns, and none of them touches an unanticipated format.

Treat your incident register as a visibility measure. The silent categories dominate exposure
and never appear in it.

## 11. Common Mistakes

**Treating memorisation as a frequency problem.** A single occurrence is 0.308 extractable
here.

**Reporting deduplication as a privacy control.** It leaves the singletons untouched.

**Ranking leak controls by literature depth.** Memorisation is 2.9% of records and most of the
attention.

**Sharing a semantic cache across tenants.** 99.7% of hits are cross-tenant at 340 tenants.

**Raising the cache threshold to fix cross-tenancy.** It removes the cache before it removes
the channel.

**Assuming redaction handles membership.** Membership is not content and survives it.

## 12. Failure Modes

**Larger model, same corpus, new exposure.** Nothing was retrained on new data and extraction
probability rose 4×.

**Cache leak with no incident.** 161,103 records a year, 372 recorded, because nobody could
tell.

**Payload logging added for observability, never redacted.** The chapter that asked for it and
the chapter that pays for it are two parts apart.

**Retrieval filter applied post-ranking.** The document was excluded from the answer and its
existence leaked through the ranking.

**Secret scanner tuned on the formats you know.** The credential that leaked was in a format
introduced last quarter.

**DP-SGD deployed and the cache left shared.** The formal bound covers 2.9% of the problem and
the config covers 19.4%.

## 13. Alternatives

**Never train on customer data.** Removes the memorisation term entirely. The strongest
control and it forecloses a class of product.

**Per-tenant model adapters.** Isolate tenant data in weights that only that tenant is served
from. Strong, and it multiplies serving cost by tenancy.

**Retrieval-only, no fine-tuning.** Keeps sensitive content out of the weights and puts it in
the context, which trades {{ch:sec-data-leakage}}'s first half for its second.

**Full DP training.** {{cite:abadi2016dpsgd}}'s mechanism with a stated $\varepsilon$. The
only formal guarantee and 22 units of effort.

**Output-side canary detection.** Plant unique canaries in the corpus and monitor outputs for
them. Cheap, measures memorisation directly, and only for the canaries.

## 14. Evaluation

Plant canaries of varying occurrence counts and distinctiveness in a training corpus and
measure extraction. It is the only way to get your own $f(n)$ and $h(\theta)$.

Re-run the canary test on every model size increase. The corpus did not change and the risk
did.

Audit your semantic cache for cross-tenant hits over a week. Count them; do not model them.

Sample production logs for sensitive fields and count what a redaction pass would remove.

Compare your incident register against a directly measured exposure for one silent category.
The ratio is the finding.

## 15. Advanced Concepts

The extraction model treats distinctiveness and occurrence count as independent, and in real
corpora they are strongly negatively correlated: strings that appear many times are, almost by
definition, less distinctive. That correlation makes deduplication look better than it is in
the aggregate — it removes mass from the high-$n$, low-$\delta$ region where per-item risk was
already lowest — and makes the singleton residual a *larger* share of the remaining risk than
the raw counts suggest. **The right metric for a dedup programme is risk-weighted, not
count-weighted**, and the count-weighted version is what tooling reports.

The cache model assumes a uniform distribution of cache entries across tenants, which is
wrong in a way that matters. Real tenancy is heavy-tailed: a few large tenants generate most
queries and therefore most cache entries. Under that distribution the cross-tenant hit
probability for a *small* tenant approaches one — almost every neighbour is from a large
tenant — while for the largest tenant it is much lower. So **the exposure is concentrated on
small tenants receiving large tenants' content**, which is both the worse direction
commercially and the harder one to detect, because small tenants generate few complaints.

There is an interaction between the two halves of this chapter that neither states alone. A
retrieval-only architecture is often recommended as the privacy-preserving alternative to
fine-tuning: keep sensitive data out of the weights and put it in the context at query time.
That trade is real, and it moves the exposure from the 2.9% category to the 22.6% and 19.4%
categories. **It is a good trade only if the retrieval, cache and logging controls are in
place**, and it is frequently made by teams who have addressed memorisation precisely because
they were thinking about memorisation.

Finally, on membership. {{cite:shokri2017membership}}'s attack is usually discussed for
training sets, and the same structure applies to retrieval indices: an attacker who can
observe whether a document influenced an answer learns that the document is indexed, which
for a legal-discovery corpus or an M&A data room is the sensitive fact. No content control
reaches it, differential privacy does not apply to a retrieval index, and the only mitigation
is to make influence unobservable — which conflicts with citation, attribution and every
transparency requirement in {{ch:rai-interpretability}}. That is a genuine and unresolved
tension, and it is worth flagging as one rather than resolving it with a preference.

## 16. Connection to Previous Chapters

{{eq:cache-threshold-is-an-error-cost-decision}} from {{ch:sd-routing-caching}} returns as a
security result: the threshold trades exposure against hit rate proportionally and never
removes the channel, while partitioning does.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} is why the cache and the
log are separate leak sources with separate controls rather than views of the primary store.

{{eq:attribution-needs-payload-not-timing}} from {{ch:ops-observability}} created this
chapter's largest leak source. The two requirements are in real tension and field-level
redaction at emit is the resolution.

{{eq:leaking-is-bounded-by-context-hijacking-is-not}} from {{ch:sec-prompt-injection}} bounded
the context; this chapter's sources are the ones outside it.

## 17. Exercises

1. Plant canaries at occurrence counts 1, 3, 10 and 100 and measure extraction at two model
   sizes. What is your own $f(n)$?

2. Audit a week of semantic cache hits for cross-tenant provenance. How does the measured rate
   compare to $1 - 1/T$?

3. Rank your leak sources by records prevented per unit of effort. Where does memorisation
   sit?

4. Compute the cross-tenant hit probability for your smallest tenant under a heavy-tailed
   entry distribution, following {{sec:15-advanced-concepts}}.

5. Design a redaction-at-emit scheme that preserves {{ch:ops-observability}}'s four
   highest-value fields. Which of them can survive redaction?

## 18. Interview Questions

1. We deduplicated the training corpus. How much did that reduce extraction risk?

2. Why does the same corpus become riskier under a larger model?

3. Our semantic cache is shared across tenants. What is the exposure?

4. Why is raising the cache similarity threshold not a fix?

5. Our incident register shows two leaks last year. Is that reassuring?

6. What does differential privacy buy that a secret scanner does not?

## 19. Research Questions

1. What is the empirical shape of $f(n)$ and $h(\theta)$ for current model families, and does
   the singleton term grow with scale?

2. How much does a risk-weighted dedup metric differ from a count-weighted one on real
   corpora?

3. How concentrated is cache cross-tenant exposure on small tenants under realistic
   heavy-tailed usage?

4. Can index-membership inference be bounded without foreclosing citation and attribution?

## 20. Chapter Summary

There are two leak questions and the smaller one gets the attention.

**Memorisation does not vanish at one occurrence.** {{cite:carlini2021extracting}} recovered
sequences appearing in a single training document, and the risk rises with model size: a UUID
at **0.082** in a 1.5B model and **0.333** at 400B, from the same corpus
({{eq:extraction-risk-does-not-vanish-at-one-occurrence}}). Distinctiveness beats frequency —
a UUID appearing once (0.308) exceeds a phone format appearing 4,100 times (0.301) — and
deduplication removes **49%** of expected extractions while leaving the singleton classes at
**0%** ({{eq:dedup-helps-the-common-secret-not-the-rare-one}}). Every cheap control needs to
know a pattern or a list; {{cite:abadi2016dpsgd}}'s is the only one that holds for a secret
nobody anticipated.

**And memorisation is 2.9% of leaked records.** Logs with payloads are **44.9%**, misrouted
retrievals **22.6%**, cache cross-tenancy **19.4%**
({{eq:most-leaks-are-inference-time-not-memorised}}). Ranked by records prevented per unit of
effort: **930** for the cache fix, **344** for redaction at emit, **3** for memorisation
controls.

The cache is the sharpest case. Across 340 tenants, **99.7%** of hits are cross-tenant —
**441 records a day** with no attacker
({{eq:shared-cache-is-a-cross-tenant-channel}}) — against **372 a year** in the incident
register. Raising the similarity threshold from 0.80 to 0.995 removes the exposure and the
cache with it; keying by tenant costs **5 points** of hit rate and removes the channel.

What runs through the chapter is that the visible problem and the large problem are different
problems. Memorisation is visible because it is studied; cache cross-tenancy is invisible
because the recipient cannot tell. The incident register measures which leaks had a witness,
and a system optimised against that register is optimised against visibility. The fix is
unglamorous in the same way the rest of this part's fixes are: a config change on the cache, a
redaction pass on the logs, a filter moved earlier in the retrieval path.

Carry forward: **a single occurrence is not safe**, and **most leaks are inference-time and
cheap to close**.

## 21. Further Reading

- {{cite:carlini2021extracting}} — verbatim extraction from a deployed model, with the
  single-occurrence and model-size findings this chapter is built on.
- {{cite:shokri2017membership}} — membership inference, and why membership can be the
  sensitive fact.
- {{cite:abadi2016dpsgd}} — the mechanism that bounds influence without knowing what the
  secret is.
- {{cite:perez2022ignore}} — prompt leaking, the inference-time channel with the lowest
  per-incident volume and the highest frequency.
