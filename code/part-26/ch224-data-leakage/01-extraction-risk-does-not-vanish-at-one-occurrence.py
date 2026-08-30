# -*- coding: utf-8 -*-
# Extracted from: Chapter 224 — Data Leakage and Secrets Management
# Source: src/.../ch224-data-leakage.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
