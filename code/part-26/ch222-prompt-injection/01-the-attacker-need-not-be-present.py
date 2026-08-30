# -*- coding: utf-8 -*-
# Extracted from: Chapter 222 — Prompt Injection and Indirect Prompt Injection
# Source: src/.../ch222-prompt-injection.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Direct injection costs one attempt per victim. Indirect injection costs one write.

cite:perez2022ignore demonstrated injection through the user's own message, which requires
the attacker to be the user. cite:greshake2023indirect demonstrated it through content the
system retrieves, which does not.

That changes the economics rather than the mechanism. A poisoned document sits in an index
and fires every time it is retrieved, for as long as it is there, against users the attacker
never contacted (eq:the-attacker-need-not-be-present).

So the cost per compromise is the cost of one write divided by the number of retrievals it
survives, which falls toward zero
(eq:indirect-injection-amortises-over-retrievals).

This listing computes both, prices the dwell time, and follows the poisoned content into the
derived copies that ch:sd-storage warned about.
"""
CORPUS = 2_400_000
QUERIES_PER_DAY = 42_000.0
TOP_K = 8
TARGET_CLASS_SHARE = 0.11     # queries the poisoned docs are crafted to match
HIT_GIVEN_TARGET = 0.34       # P(a poisoned doc reaches the context | target query)

print(f"A {CORPUS:,}-document corpus, {QUERIES_PER_DAY:,.0f} queries a day, top-{TOP_K}.")
print()
print(f"{'poisoned docs':>15}{'share of corpus':>18}{'firings/day':>14}"
      f"{'firings in 90 days':>21}{'users reached':>16}")
print("-" * 84)
fire = {}
for p in (1, 5, 25, 100, 500):
    # More poisoned docs raise the chance at least one lands in the top-k.
    hit = 1.0 - (1.0 - HIT_GIVEN_TARGET) ** min(p, 6)
    daily = QUERIES_PER_DAY * TARGET_CLASS_SHARE * hit
    fire[p] = (hit, daily, daily * 90)
    print(f"{p:>15}{p / CORPUS:>18.6%}{daily:>14,.0f}{daily * 90:>21,.0f}"
          f"{daily * 90 * 0.62:>16,.0f}")

print()
print(f"one document is {1 / CORPUS:.7%} of the corpus and reaches "
      f"{fire[1][2]:,.0f} sessions in a quarter")

print()
print()
print("Against the direct channel, where the attacker has to be present.")
print()
WRITE_COST = 90.0             # cost of getting one document into the index
REQUEST_COST = 0.004          # cost of one direct request
print(f"{'attack mode':>28}{'setup':>10}{'per compromise':>17}"
      f"{'rate limited?':>16}{'attacker visible?':>20}")
print("-" * 91)
MODES = [
    ("direct, one session",       0.0,   REQUEST_COST,            "yes", "yes, as a user"),
    ("direct, scripted",          40.0,  REQUEST_COST,            "yes", "yes, as a client"),
    ("indirect, 1 document",      WRITE_COST, WRITE_COST / fire[1][2],  "no",  "no"),
    ("indirect, 25 documents",    WRITE_COST * 25, WRITE_COST * 25 / fire[25][2], "no", "no"),
]
for name, setup, per, rl, vis in MODES:
    print(f"{name:>28}{setup:>10,.0f}{per:>17.5f}{rl:>16}{vis:>20}")

print()
print(f"one write at {WRITE_COST:,.0f} amortised over {fire[1][2]:,.0f} firings is "
      f"{WRITE_COST / fire[1][2]:.4f} per compromise")
print(f"a direct request costs {REQUEST_COST:.4f} and reaches one session")

print()
print()
print("Dwell time: how long the document survives before anyone removes it.")
print()
print(f"{'detection method':>34}{'mean days to detect':>22}"
      f"{'firings before removal':>25}{'covers':>18}")
print("-" * 99)
DETECT = [
    ("nobody is looking",             999.0, "nothing"),
    ("user reports something odd",     41.0, "visible effects"),
    ("periodic manual corpus review",  62.0, "a sample"),
    ("instruction-pattern scan at ingest", 0.2, "what you ingest"),
    ("output anomaly detection",       11.0, "visible effects"),
    ("provenance audit on a fired tool", 3.5, "acted-on cases"),
]
dwell = {}
for name, days, cov in DETECT:
    firings = fire[1][1] * min(days, 999)
    dwell[name] = (days, firings)
    print(f"{name:>34}{days:>22.1f}{firings:>25,.0f}{cov:>18}")

print()
print("The cheapest detection is at ingest and it only covers documents you")
print("ingested, which is the point ch:sec-prompt-injection's second listing takes up.")

print()
print()
print("And where the content goes after it is indexed.")
print()
DERIVED = [
    ("primary index",              True,  "yes",  1.0),
    ("embedding vectors",          True,  "yes",  1.0),
    ("summary cache",              False, "no",   0.31),
    ("semantic cache of answers",  False, "no",   0.22),
    ("conversation history",       False, "no",   0.47),
    ("fine-tuning corpus snapshot", False, "no",  0.08),
    ("another team's copy",        False, "no",   0.14),
]
print(f"{'store':>30}{'removed by deleting the doc?':>31}"
      f"{'share of firings':>19}{'residual':>11}")
print("-" * 91)
residual = 0.0
for name, cleared, ans, share in DERIVED:
    if not cleared:
        residual += share
    print(f"{name:>30}{ans:>31}{share:>19.0%}"
          f"{(0.0 if cleared else share):>11.2f}")
print("-" * 91)
print(f"{'RESIDUAL AFTER SOURCE DELETION':>30}{'':>31}{'':>19}{residual:>11.2f}")

print()
print(f"deleting the source document leaves {residual:.2f} firings-equivalent")
print("in stores that were populated from it")

print()
print()
print("Total exposure from one poisoned document, by response speed.")
print()
print(f"{'response':>34}{'days to source removal':>25}"
      f"{'firings, source':>18}{'firings, derived':>19}{'total':>10}")
print("-" * 106)
for name, days, cov in DETECT:
    d = min(days, 180.0)
    src = fire[1][1] * d
    der = src * residual * 0.5      # derived stores drain over the same period
    print(f"{name:>34}{d:>25.1f}{src:>18,.0f}{der:>19,.0f}{src + der:>10,.0f}")

print(f"""
The firings table is the asymmetry stated as arithmetic. A single poisoned document is
{1 / CORPUS:.7%} of a {CORPUS:,}-document corpus and reaches {fire[1][2]:,.0f} sessions in
ninety days, because it does not need to be found -- it needs to be *retrieved*, and
retrieval is a similarity operation the attacker can optimise against
(eq:the-attacker-need-not-be-present).

Twenty-five documents reach {fire[25][2]:,.0f} -- only {fire[25][2] / fire[1][2]:.1f} times
as many, because the constraint is the share of queries in the target class rather than the
number of documents. **Most of the attack's value is in the first few documents**, and a
defence that counts poisoned documents is measuring a quantity the attacker has no reason to
maximise.

The mode table is the economics. A direct injection costs {REQUEST_COST:.4f} and reaches one
session; it is rate-limited, and the attacker appears in your logs as a client. One indirect
write costs {WRITE_COST:,.0f} and reaches {fire[1][2]:,.0f} sessions, which is
{WRITE_COST / fire[1][2]:.4f} per compromise
(eq:indirect-injection-amortises-over-retrievals) -- and it is not rate-limited, because the
requests are coming from your own users.

Note the fourth row, which runs the other way: twenty-five documents cost
{WRITE_COST * 25 / fire[25][2]:.5f} per compromise, *more* than a direct request, because
setup scales linearly while firings saturate. The efficient indirect attack is a small number
of well-targeted documents -- which is the opposite of what a volume-based detector is built
to find.

**The attacker is not present at the time of the attack**, which removes every control that
depends on observing them: rate limits, client reputation, authentication, anomaly detection
on request patterns. All of those are watching a channel the attack does not use.

The dwell table is the term that decides total exposure, and the column that matters is the
last one. `instruction-pattern scan at ingest` detects in {0.2:.1f} days -- excellent -- and
covers only what you ingested. If the poisoned document arrived through a partner feed, a
shared drive, a crawled site or a user upload that skipped the pipeline, the scan never saw
it, and the next-fastest detector is `provenance audit on a fired tool` at
{dwell['provenance audit on a fired tool'][0]:.1f} days and
{dwell['provenance audit on a fired tool'][1]:,.0f} firings.

With nobody looking, the document fires {fire[1][1]:,.0f} times a day indefinitely.

The derived-store table is ch:sd-storage's warning arriving as a security property. Deleting
the poisoned document clears the primary index and the embeddings. It does not clear the
summary cache, the semantic answer cache, the conversation histories, the fine-tuning
snapshot, or whichever team copied the corpus last quarter -- and those carry
{residual:.2f} firings-equivalent of the original exposure
(eq:indirect-injection-amortises-over-retrievals).

**Incident response for a poisoned corpus is not a delete, it is a fan-out**, and the fan-out
list is exactly the derived-copy inventory that ch:sd-storage found nobody maintains.

The total table is what to put in the incident review. At `user reports something odd` --
{41:.0f} days, which is a realistic median for a subtle injection -- one document produces
{fire[1][1] * 41:,.0f} source firings and {fire[1][1] * 41 * residual * 0.5:,.0f} derived
ones.

Two things follow for design. **Ingest-time scanning is the cheapest control and covers only
sources you own**, so its value is bounded by the share of untrusted content that passes
through a pipeline you control. And **the derived copies have to be in the runbook before the
incident**, because enumerating them under time pressure is how a two-day response becomes a
two-week one.""")
