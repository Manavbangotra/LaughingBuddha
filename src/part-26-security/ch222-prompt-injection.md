---
id: sec-prompt-injection
number: 222
part: XXVI
tier: full
status: draft
requires: [instructions-and-data-share-a-channel, attack-surface-is-sources-times-sinks,
           only-capability-limits-bound-the-damage, derived-copies-multiply-contradiction]
provides: [the-attacker-need-not-be-present, indirect-injection-amortises-over-retrievals,
           leaking-is-bounded-by-context-hijacking-is-not, sanitisation-covers-only-what-you-own]
citations: [greshake2023indirect, perez2022ignore, debenedetti2024agentdojo,
            beurerkellner2025patterns]
---

## 1. Learning Objectives

By the end of this chapter you will be able to compute the number of sessions a single
poisoned document reaches and the resulting cost per compromise; explain why every
attacker-observing control is watching the wrong channel for indirect injection; trace
poisoned content into derived stores and compute the residual after deleting the source;
separate goal hijacking from prompt leaking and state what bounds each; explain why output
filtering is effective against one and not the other; and compute the coverage limit on
ingest-time sanitisation from the share of untrusted content you own.

## 2. Why This Matters

{{ch:sec-threat-model}} established that instructions and data share a channel. This chapter
is about the two ways an attacker uses that, and the difference between them is economic
rather than technical.

{{cite:perez2022ignore}} demonstrated injection through the user's own message, which
requires the attacker to be the user. {{cite:greshake2023indirect}} demonstrated it through
retrieved content, which does not.

A single poisoned document — **0.0000417%** of a 2.4-million-document corpus — reaches
**141,372 sessions in ninety days**, because it does not need to be found, it needs to be
retrieved ({{eq:the-attacker-need-not-be-present}}). One write at $90 amortised over those
firings is **$0.0006 per compromise** against **$0.0040** for a direct request that reaches
one session ({{eq:indirect-injection-amortises-over-retrievals}}) — and it is not rate
limited, because the requests come from your own users.

Deleting the document does not end it. Derived stores — summary caches, semantic answer
caches, conversation histories, fine-tuning snapshots — carry **1.22 firings-equivalent** of
the original exposure.

The second half separates the two outcomes. Leaking is bounded by what is in the context
(**17.3** here, a number you choose); hijacking is bounded by what the agent can call
(**42.5**, a number that rises every quarter)
({{eq:leaking-is-bounded-by-context-hijacking-is-not}}). And output scanning catches **88%**
of leaks and **4%** of hijacks, because a hijacked action looks like a legitimate one.

Ingest-time sanitisation, the cheapest control available, sees only **38%** of untrusted
content ({{eq:sanitisation-covers-only-what-you-own}}).

## 3. Prerequisites

{{eq:instructions-and-data-share-a-channel}} from {{ch:sec-threat-model}} is the mechanism;
this chapter is about its two exploitations and their different economics.

{{eq:attack-surface-is-sources-times-sinks}} from the same chapter is what bounds hijacking,
and {{sec:9-practical-example}} shows the leak bound is the other factor — the context — which
that chapter did not price.

{{eq:only-capability-limits-bound-the-damage}} is confirmed here on a specific pair of
outcomes: the only defences that touch hijacking are the two structural ones.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} is the incident-response
result. That chapter found derived copies multiply contradictions; here they multiply
exposure, and the fan-out list is the same one nobody maintains.

## 4. Intuitive Explanation

There are two injection channels and they are usually discussed as one thing. They are not
one thing, because they have different costs, different detection surfaces, and different
people at the other end.

**Direct injection** is what most demos show. The attacker is the user. They type something
into the box that overrides your instructions, and they see the result. To attack a second
person, they need that person to type something, which they generally cannot arrange.

**Indirect injection** is {{cite:greshake2023indirect}}'s contribution. The attacker writes a
document. Your system retrieves it, weeks later, because it matched a query somebody else
asked. The attacker was not present, is not in your logs, and does not know who the victim
was.

The economics of those two are not comparable.

Run the arithmetic on a realistic corpus: 2.4 million documents, 42,000 queries a day, top-8
retrieval. Suppose the attacker crafts one document to rank well for a class of queries
covering 11% of traffic, and it makes it into the context a third of the time within that
class.

That document fires 1,571 times a day. In ninety days: 141,372 sessions.

One document. Forty-two millionths of a percent of the corpus.

Now price it. Getting a document into an index costs something — buying an expired domain,
writing a wiki edit, uploading to a shared drive, publishing a plausible blog post. Call it
$90. Divided over 141,372 firings, that is **six hundredths of a cent per compromise.**

A direct request costs a fraction of a cent and reaches one person, and it is rate limited,
and the attacker shows up in your logs as a client.

There is a wrinkle worth noticing in that table. Twenty-five poisoned documents do not reach
twenty-five times as many people — they reach 2.7 times as many, because the binding
constraint is the share of queries in the target class rather than the number of documents.
So the per-compromise cost of a 25-document campaign is *higher* than a direct request.
**The efficient indirect attack is a small number of well-targeted documents**, which is
precisely what a volume-based anomaly detector is not built to find.

What follows from "the attacker is not present" is the part that changes architecture. Every
control that works by observing the attacker is watching a channel the attack does not use:
rate limits, client reputation, authentication, request-pattern anomaly detection, IP
blocking, CAPTCHA. All of those are excellent, all of them are pointed at the direct channel,
and none of them sees a poisoned document being retrieved by a legitimate user's legitimate
query.

Then there is dwell time, which is where total exposure actually comes from. If nobody is
looking, the document fires indefinitely. If a user reports something odd — a realistic
detection path for a subtle injection, at around 41 days — it has fired 64,403 times.

The fastest detector on the list is an instruction-pattern scan at ingest, at a fifth of a
day. It is also the one with the narrowest coverage: it only sees documents that went through
your ingest pipeline. If the content arrived from a partner feed, a live web fetch, a user
upload that skipped the pipeline, or another agent's reply, the scan never saw it.

And then the content spreads. This is {{ch:sd-storage}}'s derived-copy result arriving as a
security property. Deleting the poisoned document clears the primary index and the embeddings.
It does not clear the summary cache, the semantic answer cache, the conversation histories
that already contain the injected text, the fine-tuning snapshot taken last month, or the copy
another team pulled. Those carry 1.22 firings-equivalent of the original exposure.

**Incident response for a poisoned corpus is a fan-out, not a delete**, and the fan-out list
has to exist before the incident, because enumerating derived stores under time pressure is
how a two-day response becomes a two-week one.

That is the channel. The second half of the chapter is about outcomes, and
{{cite:perez2022ignore}}'s distinction between them is the most useful thing in this area.

**Prompt leaking** exfiltrates what is already in the context: the system prompt, the tool
schemas, retrieved passages, prior turns, and — if somebody put one there — a session token.

**Goal hijacking** makes the system do something: send an email, write to the CRM, issue a
refund, execute code, read from a secret store.

Those are bounded by completely different things. Leaking is bounded by what you put in the
prompt. Hijacking is bounded by what you connected. The first is a design parameter you set
when you write the template; the second is a product roadmap that grows every quarter.

Which is why they diverge over time even in teams that take both seriously. Nobody adds
things to the system prompt at the rate they add integrations.

The defences are asymmetric too, and the asymmetry runs against intuition. Scanning outputs
for known secrets catches 88% of leaks — you know what the string looks like, you can search
for it. The same scanner catches 4% of hijacks.

Because **a hijacked action looks like a legitimate action.** The tool is on the list, the
arguments are well-formed, the target is plausible. That is
{{ch:sd-architecture}}'s semantic-failure result in security clothing: the step succeeded and
was wrong.

The only defences that move the hijack number are the structural ones — a tool allow-list, a
human approval on the sink — and neither is a detector.

Put the postures side by side. Output scanning alone takes the leak residual from 17.30 to
0.81 and leaves the hijack residual at 37.13, essentially untouched, for 13% of utility. An
allow-list alone leaves leaks at 17.30 and takes hijacks to 8.92. Together they reach 8.61
total for 29% utility.

**A defence-in-depth stack made of two output scanners is depth against one of the two
attacks.** The layers have to address different outcomes, not reinforce each other on the
same one.

Finally, the lever nobody prices. Leaking is bounded by context contents, so removing a
session token from the prompt takes the leak bound from 17.3 to 8.3 for 4% of utility. That
is a bigger reduction than every output scanner combined, at a fraction of the cost, achieved
by deleting a line from a template.

Before building a detector: look at what is in the prompt and remove what does not need to be
there, then look at what the agent can call and remove what it does not need. Both are
subtractions. Both are cheap. Both bound an outcome that no detector bounds.

## 5. Formal Explanation

**Indirect amortisation.** Let $C$ be corpus size, $Q$ queries per day, $\theta$ the share of
queries in the attacker's target class, and $h(p)$ the probability that at least one of $p$
poisoned documents enters the context given a target query. Firings per day are
$F = Q\theta h(p)$, and over dwell time $T$ the total is $FT$. With write cost $w$ per
document, cost per compromise is

$$\frac{pw}{Q\theta h(p) T},$$

decreasing in $T$ and, because $h$ saturates, *increasing* in $p$ beyond a small number. The
attacker's optimum is therefore few documents and long dwell, which is the configuration
hardest to detect by volume.

**Derived residual.** If poisoned content is copied into stores $j$ with firing shares
$\sigma_j$, and deleting the source clears only the subset $\mathcal{C}$, the residual is
$\sum_{j \notin \mathcal{C}} \sigma_j$. This is independent of how quickly the source was
found: response speed bounds $T$ and not the residual.

**The two bounds.** Let $L$ be the set of context items with values $v_i$ and $K$ the set of
sinks with damages $d_k$. Then $\max(\text{leak}) = \sum_i v_i$ and
$\max(\text{hijack}) = \sum_k d_k$. The first is a function of prompt assembly; the second of
integration count. Neither bound involves the model or the attack.

**Sanitisation coverage.** A control at stage $s$ removes a share $\rho_s$ of injections in
the content it sees, and sees a share $\gamma_s$ of untrusted volume. Effective removal is
$\rho_s \gamma_s$, so a control with $\rho_s = 1$ at an ingest stage with $\gamma_s = 0.38$ is
bounded at 0.38 regardless of quality.

## 6. Mathematical Foundation

The attacker's absence from the request path:

$$F = Q\,\theta\,h(p), \qquad h(p) = 1 - (1-h_1)^{\min(p,\,p_{\max})}, \qquad \frac{\partial F}{\partial(\text{rate limit})} = 0$$ (eq:the-attacker-need-not-be-present)

At $Q = 42{,}000$, $\theta = 0.11$, $h_1 = 0.34$: **1,571 firings/day** from one document.

Amortisation, and its interior optimum in $p$:

$$c(p, T) = \frac{p\,w}{F(p)\,T}, \qquad c(1, 90) = \$0.0006, \qquad c(25, 90) = \$0.0059$$ (eq:indirect-injection-amortises-over-retrievals)

against a direct request's $\$0.0040$ for one session, with residual
$\sum_{j\notin\mathcal C}\sigma_j = 1.22$ after source deletion.

The two blast radii:

$$B_{\text{leak}} = \sum_{i \in L} v_i = 17.3, \qquad B_{\text{hijack}} = \sum_{k \in K} d_k = 42.5, \qquad \frac{\partial B_{\text{hijack}}}{\partial t} > 0$$ (eq:leaking-is-bounded-by-context-hijacking-is-not)

And the ceiling on sanitisation:

$$\text{effective}_s = \rho_s \gamma_s \le \gamma_s, \qquad \gamma_{\text{ingest}} = 0.38$$ (eq:sanitisation-covers-only-what-you-own)

## 7. Internal Mechanics

Why is a poisoned document so cheap to place? Because retrieval is a similarity operation and
similarity is optimisable. The attacker does not need their document to be authoritative,
popular, or linked — they need it to embed near a class of queries, and they can measure that
directly against a public embedding model. This is the same property that makes retrieval
work, used in the other direction.

It also explains the saturation. Once one document reliably lands in the top-k for the target
class, a second adds nothing for those queries; it can only extend to a different class. So
attack value scales with *query coverage*, not document count, and a defender counting
suspicious documents is measuring the attacker's least-optimised dimension.

The dwell-time asymmetry has a mechanism worth stating. A direct injection is discovered
immediately by the person it happened to, because they are the attacker and the victim at
once. An indirect injection is discovered by somebody who does not know what they are looking
at: a user who got a slightly odd answer, an analyst reviewing a sample, an engineer chasing
an unrelated bug. **The detection path for indirect injection runs through people with no
model of the attack**, which is why the reported medians are in weeks.

The derived-copy problem compounds because caches are populated by *successful* operations.
A poisoned document that produced an answer produced a cacheable answer, and the cache does
not record which document contributed. So the semantic answer cache contains the injected
outcome with no pointer back to the source, and deleting the source is invisible to it. This
is the storage-layer version of {{eq:derived-copies-multiply-contradiction}} and it has the
same remedy: an inventory that exists before it is needed.

On the outcome side, the reason output scanning fails against hijacking is worth being
precise about. A scanner asks "does this output look wrong?" For a leak, "wrong" has a
signature — a key format, a system-prompt phrase, a base64 blob, a markdown image pointing at
an attacker domain. For a hijack, the output is a tool call that is syntactically valid,
semantically plausible, and indistinguishable from the one a legitimate request would produce.
The scanner would have to know the user's intent to judge it, and the user's intent is exactly
what the injection replaced.

That is why the two structural controls dominate the hijack column and why they are not
detectors: an allow-list does not ask whether this call is legitimate, it asks whether this
tool is reachable on this path.

Finally, a note on why the context-trimming lever is so underused despite being the cheapest
row in the chapter. Things end up in prompts because somebody needed them once. A session
token was added to let the model call an API directly; a tool schema was inlined for a
debugging session; a full document was included rather than a summary because it was easier.
None of those was a security decision, and removing them requires knowing why each is there —
which is an archaeology problem, not a security one, and it is the reason prompts accumulate
exactly like configuration.

## 8. Implementation

The first listing prices the indirect channel.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ib1}
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
```

## 9. Practical Example

A 2.4-million-document corpus, 42,000 queries a day, top-8:

```
  poisoned docs   share of corpus   firings/day   firings in 90 days   users reached
------------------------------------------------------------------------------------
              1         0.000042%         1,571              141,372          87,651
              5         0.000208%         4,041              363,728         225,511
             25         0.001042%         4,238              381,432         236,488
            500         0.020833%         4,238              381,432         236,488
```

One document reaches **141,372 sessions in a quarter**
({{eq:the-attacker-need-not-be-present}}). Twenty-five reach 2.7× that, because the constraint
is query coverage rather than document count — **the attacker has no reason to maximise the
quantity a volume detector counts.**

```
                 attack mode     setup   per compromise   rate limited?   attacker visible?
-------------------------------------------------------------------------------------------
         direct, one session         0          0.00400             yes      yes, as a user
        indirect, 1 document        90          0.00064              no                  no
      indirect, 25 documents     2,250          0.00590              no                  no
```

**$0.0006 per compromise against $0.0040**
({{eq:indirect-injection-amortises-over-retrievals}}) — and the last two columns are why every
attacker-observing control misses it.

```
                  detection method   mean days to detect   firings before removal            covers
---------------------------------------------------------------------------------------------------
                 nobody is looking                 999.0                1,569,229           nothing
        user reports something odd                  41.0                   64,403   visible effects
instruction-pattern scan at ingest                   0.2                      314   what you ingest
          output anomaly detection                  11.0                   17,279   visible effects
  provenance audit on a fired tool                   3.5                    5,498    acted-on cases
```

The fastest detector has the narrowest coverage, and the realistic one — a user noticing —
costs **64,403 firings**.

```
                         store   removed by deleting the doc?   share of firings   residual
-------------------------------------------------------------------------------------------
                 primary index                            yes               100%       0.00
                 summary cache                             no                31%       0.31
     semantic cache of answers                             no                22%       0.22
          conversation history                             no                47%       0.47
   fine-tuning corpus snapshot                             no                 8%       0.08
           another team's copy                             no                14%       0.14
-------------------------------------------------------------------------------------------
RESIDUAL AFTER SOURCE DELETION                                                         1.22
```

**Incident response is a fan-out, not a delete.**

The second listing separates the two outcomes.

```python {tier=A name=C:/Users/MANAVB~1/AppData/Local/Temp/claude/C--Github-LaughingBuddha/30a87753-43a8-48c3-8378-261faf976dbb/scratchpad/ib2}
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
```

```
                         stage   removable    cost   sees this share of untrusted   effective
---------------------------------------------------------------------------------------------
   crawl / partner feed ingest         71%     1.0                            38%         27%
    retrieval, before assembly         52%     1.2                           100%         52%
               prompt assembly         44%     0.6                           100%         44%
   tool call, before execution         79%     2.0                           100%         79%
        output, before display         36%     0.9                           100%         36%
```

**Sanitisation scales with ownership of the pipeline, not with effort**
({{eq:sanitisation-covers-only-what-you-own}}).

```
             outcome                  bounded by   items   total damage              grows with
-----------------------------------------------------------------------------------------------
      prompt leaking      what is in the context       5           17.3             prompt size
      goal hijacking     what the agent can call       5           42.5       integration count
```

One is a design parameter, the other a product roadmap
({{eq:leaking-is-bounded-by-context-hijacking-is-not}}).

```
                           defence   catches a leak   catches a hijack                             why
------------------------------------------------------------------------------------------------------
     output scan for known secrets              88%                 4%             you know the string
    output scan for exfil patterns              61%                 9%   URLs, base64, markdown images
    output scan for 'wrong action'              12%                31%     the action looks legitimate
              tool-call allow-list               0%                79%       structural, not detection
        human approval on the sink               0%                88%       structural, not detection
```

```
                         posture   leak residual   hijack residual     total   utility cost
-------------------------------------------------------------------------------------------
                         nothing           17.30             42.50     59.80             0%
            output scanning only            0.81             37.13     37.94            13%
                 allow-list only           17.30              8.92     26.23            18%
    output scanning + allow-list            0.81              7.80      8.61            29%
                      everything            0.65              0.52      1.17            63%
```

**Two output scanners are depth against one of the two attacks.**

```
                  context contains   leak bound   residual under scanning   utility cost
----------------------------------------------------------------------------------------
     everything, including a token         17.3                     0.810             0%
                  no session token          8.3                     0.388             4%
      no session token, no schemas          7.5                     0.351            11%
           retrieved passages only          4.0                     0.187            22%
```

Deleting a line from a template halves the leak bound for **4%** of utility — a bigger
reduction than every output scanner combined.

## 10. Production Considerations

Count what a single poisoned document in your corpus would reach. Queries per day times target
share times hit rate times dwell — four numbers you have.

Build the derived-copy inventory now. Summary caches, answer caches, conversation histories,
training snapshots, other teams' copies. It is the fan-out list an incident needs and it takes
an afternoon in peacetime.

Scan at ingest and record the coverage. The control is excellent and its value is capped by
the share of untrusted content that passes through it.

Add provenance to tool calls so a fired sink can be audited back to the content that triggered
it. It is the second-fastest detector on the list and the only one that sees acted-on cases.

Trim the context before building a detector. Removing a session token is a template edit and
it halves the leak bound.

Trim the tool list before building a detector. The hijack bound is the sum of what you
connected and nothing else touches it.

Stop treating direct and indirect injection as one control problem. They share a mechanism and
share almost no defences.

## 11. Common Mistakes

**Rate-limiting as an injection control.** The indirect channel's requests come from your own
users.

**Counting poisoned documents.** The attacker optimises query coverage, not document count.

**Deleting the source and closing the incident.** 1.22 firings-equivalent remain in derived
stores.

**Scanning outputs and calling hijacking handled.** 4% caught.

**Relying on ingest sanitisation for content you do not ingest.** The ceiling is the coverage
share.

**Adding a session token to the prompt.** It is over half the leak bound and it was added for
convenience.

## 12. Failure Modes

**Poisoned document with a 41-day dwell.** Nobody was looking at the corpus and the detection
path ran through a confused user.

**Cache still serving the injected answer.** The source was deleted, the semantic cache was
not, and it has no pointer back.

**Provenance dropped at a service boundary.** The tool-call auditor cannot tell which content
triggered the call.

**Allow-list that grew to include everything.** Each addition was justified; the hijack bound
is back where it started.

**Ingest scan with 100% precision and 38% coverage, reported as 100%.** The metric is
conditional on content the scanner saw.

**Two output scanners presented as defence in depth.** Both cover leaks; the hijack column is
unchanged.

## 13. Alternatives

**Spotlighting / provenance marking.** Mark untrusted spans and instruct the model not to
follow them. Cheap, and it is a prior rather than a boundary —
{{eq:instructions-and-data-share-a-channel}} applies.

**Retrieval from a curated corpus only.** Removes the indirect channel by removing the
untrusted source. Strong, and it forfeits open-web and partner content.

**Two-model separation.** An unprivileged model reads untrusted content and returns
structured data; a privileged model never sees the raw text.
{{cite:beurerkellner2025patterns}}'s family.

**Content re-generation at ingest.** Summarise every document through a model with no tools,
store the summary, retrieve only summaries. Strips most instruction-shaped text at a fidelity
cost.

**Per-source trust tiers with different sink permissions.** An answer grounded in a
low-trust source cannot trigger a privileged action. Structural, and it needs provenance to
survive the pipeline.

## 14. Evaluation

Plant a benign marker document in a staging corpus and measure how many sessions retrieve it
in a week. That is your firing rate, measured rather than modelled.

Measure your ingest coverage: what share of untrusted tokens in a production context passed
through a pipeline you control?

Delete a test document and check every derived store for its content 24 hours later. Whatever
remains is your residual.

Run {{cite:debenedetti2024agentdojo}} and report the leak and hijack success rates separately.
An aggregate hides the asymmetry this chapter is about.

Sum your context items and your sinks and publish both bounds. They are the two numbers a
security review should open with.

## 15. Advanced Concepts

The independence between poisoned documents assumed in $h(p)$ is optimistic for the attacker
and pessimistic for the defender in different regimes. Documents crafted for the same query
class are highly correlated — they compete for the same top-k slots — so $h$ saturates faster
than the model shows, which strengthens the "few documents" conclusion. But documents crafted
for *different* query classes are near-independent and their coverage adds, so a campaign
targeting ten disjoint classes scales close to linearly in reach. **The saturation is within a
class and the linearity is across classes**, which means the defensive signal to look for is
breadth of query coverage rather than count.

The leak bound treats context items as independently valuable, which understates the risk in
one specific case. A session token is not worth 9 because it is a valuable string; it is worth
9 because it converts a leak into a hijack — an attacker with the token calls the API
directly, outside your allow-list, outside your approval flow, outside every control in
{{sec:9-practical-example}}'s second table. **Any credential in the context collapses the two
bounds into one**, and the collapsed bound is the larger of them plus whatever the credential
reaches. That is the strongest argument in this chapter for the cheapest fix in it.

There is an interaction with {{ch:ops-agent-tracing}} worth drawing out. That chapter found
the cause of an agent failure sits 2.7 steps back from where it becomes visible, and that the
causing step usually succeeded. An indirect injection is exactly that shape: the retrieval
succeeded, the passage was well-formed, the tool call was valid, and the cause is a document
retrieved several steps earlier. So the trace fields that chapter argued for — tool results as
received, causal links between steps — are the same fields a provenance audit needs.
**Injection forensics and agent triage want identical instrumentation**, and building it once
serves both.

Finally, a limit on the whole framing. This chapter treats the corpus as something an attacker
writes into and the defender curates. For a system retrieving from the open web, there is no
curation step and $\gamma_{\text{ingest}} = 0$ — every sanitisation control lives at query
time or later, and the ranking in {{sec:9-practical-example}}'s first table changes completely.
The design question that dominates everything else is whether the system reads content nobody
vetted, and it is answered by product managers.

## 16. Connection to Previous Chapters

{{eq:instructions-and-data-share-a-channel}} from {{ch:sec-threat-model}} is the mechanism
both halves of this chapter exploit; the contribution here is that the two exploitations have
different economics and almost disjoint defences.

{{eq:attack-surface-is-sources-times-sinks}} bounds hijacking, and this chapter adds the other
factor that chapter did not price: the context, which bounds leaking and is a design
parameter.

{{eq:only-capability-limits-bound-the-damage}} is confirmed on a specific pair — only the two
structural controls move the hijack column at all.

{{eq:derived-copies-multiply-contradiction}} from {{ch:sd-storage}} becomes the
incident-response result: the same uninventoried derived copies that multiply contradictions
multiply exposure, at 1.22× after the source is gone.

## 17. Exercises

1. Compute your own firing rate for a hypothetical poisoned document: queries per day, target
   class share, hit rate, dwell.

2. Enumerate your derived stores and determine which are cleared by deleting a source
   document. What is your residual?

3. Sum your context items by value and your sinks by damage. Which bound is larger, and which
   one grew last quarter?

4. Measure what share of untrusted tokens in a production context passed through your ingest
   pipeline.

5. Model a campaign targeting ten disjoint query classes and recompute reach. How does the
   defensive signal change from {{sec:15-advanced-concepts}}'s analysis?

## 18. Interview Questions

1. Why does rate limiting not help against indirect prompt injection?

2. What does one poisoned document in a two-million-document corpus reach?

3. We deleted the malicious document. Is the incident closed?

4. Why does output scanning work against prompt leaking and not goal hijacking?

5. Which is bounded by a decision you make, leaking or hijacking?

6. What is the cheapest thing you could do this week to reduce injection blast radius?

## 19. Research Questions

1. What are empirical dwell times for indirect injections in production retrieval corpora?

2. How much does query-class breadth, rather than document count, predict campaign reach?

3. Can provenance be maintained cheaply enough through summarisation and caching that derived
   stores become cleanable?

4. How much of the hijack blast radius is removable by per-source trust tiers without
   measurable utility loss?

## 20. Chapter Summary

Direct and indirect injection share a mechanism and share almost nothing else.

A single poisoned document — **0.0000417%** of a 2.4M-document corpus — reaches **141,372
sessions in ninety days** at **$0.0006 per compromise**, against **$0.0040** for a direct
request reaching one person ({{eq:the-attacker-need-not-be-present}},
{{eq:indirect-injection-amortises-over-retrievals}}). Twenty-five documents reach only 2.7×
as many, so the efficient attack is few and well-targeted — the configuration a volume
detector is least able to find. And because the attacker is not present, every
attacker-observing control watches a channel the attack does not use.

Deleting the source does not end it: **1.22 firings-equivalent** persist in caches, histories
and snapshots. **Incident response is a fan-out, not a delete.**

The two outcomes are bounded by different things. Leaking by the context — **17.3**, a number
you choose at prompt assembly. Hijacking by the sinks — **42.5**, a number that grows with the
integration roadmap ({{eq:leaking-is-bounded-by-context-hijacking-is-not}}). And output
scanning catches **88%** of leaks against **4%** of hijacks, because a hijacked action looks
legitimate — only the two structural controls move that column.

Sanitisation at ingest, the cheapest control on the list, sees **38%** of untrusted content
({{eq:sanitisation-covers-only-what-you-own}}), and the ceiling is ownership rather than
effort.

What the chapter keeps arriving at is that the cheap moves are subtractions and the expensive
moves are additions. Removing a session token from the prompt halves the leak bound for four
percent of utility; removing a tool removes a share of the hijack bound outright. Both are
edits to a template or a config, both take an afternoon, and neither looks like security work
— which is why the budget goes to a classifier that, on the numbers here, is not the
load-bearing control in any posture.

Carry forward: **the attacker need not be present**, and **trim the context and the tool list
before building a detector**.

## 21. Further Reading

- {{cite:greshake2023indirect}} — indirect injection against real applications; the paper that
  established the channel this chapter prices.
- {{cite:perez2022ignore}} — goal hijacking and prompt leaking named, which is the distinction
  the whole second half rests on.
- {{cite:debenedetti2024agentdojo}} — 97 tasks and 629 security test cases, reporting utility
  and attack success together.
- {{cite:beurerkellner2025patterns}} — the two-model and provenance-based patterns, with their
  utility costs stated.
