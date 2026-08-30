---
id: part-26-assessment
status: draft
---

## How to use this

Four sections. The knowledge check takes about ninety minutes. The assignment is a **boundary
audit**, because this part's rule was that a control which can be searched around is a
cost-raiser rather than a boundary, and the commonest finding is that a system's architecture
diagram shows a boundary where there is none. The challenge problems are open-ended. The
interview section is what to rehearse.

No answers are provided. Every question is answerable from the chapters.

## Knowledge check

**The threat model**

1. Why is prompt injection not solvable the way SQL injection was? Name the structural
   difference, not the difficulty.
2. State {{eq:instructions-and-data-share-a-channel}}. What is the effective privilege of a
   prompt, and what is its effective trust?
3. Compute the attack surface for 4 untrusted sources and 8 sinks. What does adding one tool
   cost?
4. Four detection layers miss 12.33% of a fixed attack. What is their miss rate against 100
   attempts, and why?
5. State {{eq:only-capability-limits-bound-the-damage}}. Why does the capability column not
   move between one attempt and a thousand?
6. Why does {{cite:zou2023universal}}'s transferability result defeat rate limiting as an
   injection control?
7. Why is a security design presented without a utility column likely to be reversed?

**Prompt injection**

8. What does a single poisoned document reach in ninety days, and why is it not rate-limited?
9. Why is a 25-document campaign more expensive *per compromise* than a 1-document one?
10. State {{eq:the-attacker-need-not-be-present}}. Which controls does it invalidate, and what
    do they have in common?
11. Deleting the poisoned source leaves 1.22 firings-equivalent. Where?
12. State {{eq:leaking-is-bounded-by-context-hijacking-is-not}}. Which bound is a design
    parameter and which is a roadmap?
13. Why does output scanning catch 88% of leaks and 4% of hijacks?
14. State {{eq:sanitisation-covers-only-what-you-own}}. What is the ceiling on ingest-time
    scanning and what sets it?

**Jailbreaks and guardrails**

15. Distinguish competing objectives from mismatched generalization. Which one cannot be closed
    by more safety data, and why?
16. Why is attack success higher in base64 (0.53) than in English (0.06) when the model is less
    capable in base64?
17. State {{eq:safety-coverage-lags-capability-by-construction}}. Why does the uncovered count
    stabilise rather than shrink?
18. At a 0.3% base rate, what fraction of guardrail alarms are wrong? Derive it.
19. How many legitimate users are refused per prevented harm at the cost-optimal threshold?
20. The optimal threshold moves from 0.84 to 0.02 across a plausible cost-ratio range. What
    ratio, and where is it written down?
21. What does a guardrail genuinely buy, given that it blocks 0.0% of a thirty-attempt
    attacker?

**Data leakage**

22. Why does a secret appearing once remain extractable? What happens to that risk as model
    size grows?
23. Why does a UUID appearing once beat a phone format appearing 4,100 times?
24. State {{eq:dedup-helps-the-common-secret-not-the-rare-one}}. What does deduplication remove
    and what does it leave?
25. Memorisation is 2.9% of leaked records. What are the three largest categories?
26. Compute cross-tenant hit probability for a semantic cache shared across 340 tenants.
27. Why does raising the cache similarity threshold not fix cross-tenancy, and what does?
28. Why does redaction not stop membership inference?

**Tool abuse and sandboxing**

29. Why is an agent a confused deputy even when correctly configured?
30. State {{eq:a-sandbox-without-scoped-credentials-moves-nothing}}. Which two capabilities does
    a sandbox not remove, and why?
31. Why is a credential in an environment variable not scoped, whatever the policy says?
32. Show that `search + email` exceeds the sum of its parts. What primitive does the pair
    supply?
33. How many subsets of size two or more does a 20-tool agent have? How many get reviewed?
34. Which actions in {{sec:9-practical-example}}'s table are least reversible, and why is that
    the opposite of the usual classification?
35. State {{eq:approval-must-sit-at-the-outcome-not-the-call}}. Why can no per-call predicate
    express a composition?

**Poisoning, permissions and governance**

36. Why does dataset size not appear in the price of a poisoning attack?
37. Which attack is cheaper — a targeted backdoor or broad degradation — and which does volume
    detection find?
38. State {{eq:trust-is-a-product-over-the-supply-chain}}. Eight links average 0.931; what is
    the composite?
39. What does a model-weight signature attest to? What does it not?
40. Why does verification converge to a floor, and what sets the floor?
41. Why does an approval queue's catch rate have an interior maximum in volume?
42. State {{eq:a-low-rejection-rate-trains-approval}}. Why is habituation not a criticism of
    reviewers?
43. State {{eq:delegation-preserves-authority-unless-attenuated}}. Why is a chain of authority
    a maximum rather than a minimum?
44. Which audit fields settle "why did it do that" rather than "who did it", and how often are
    they recorded?

## Assignment: a boundary audit

Take an AI system you are responsible for or can inspect. Produce a written audit with six
sections.

**1. The boundary inventory.** List every control your architecture diagram treats as a
boundary. For each, state whether an attacker who can retry defeats it, following
{{eq:detection-layers-fail-against-an-adaptive-attacker}}. Mark every row where the diagram
shows a boundary and the analysis shows a cost-raiser.

**2. The four counts.** Compute, from your own system: the share of a typical context that is
untrusted; your (untrusted source × privileged sink) path count; your excess authority ratio,
damage-weighted; and your composite supply-chain trust. Each is an afternoon. State which of
the four you already knew.

**3. The subtraction list.** For each of the five cheap subtractions in {{part:26}}'s
through-lines — trim the prompt, key the cache by tenant, redact at emit, filter before
ranking, remove a tool — determine whether it applies, price it in days, and estimate what it
removes. Rank against whatever detection work is currently on your roadmap.

**4. The queue.** Measure your approval queue's volume, rejection rate and time-per-item.
Compute $K(v)$ and locate the maximum. Then plant a rejectable item and see whether it is
caught.

**5. The chain.** Trace one request through every hop, recording the authority each holds and
whether the principal chain header survives. Then audit a past incident against
{{sec:9-practical-example}}'s six fields and count what you could not answer.

**6. What you could not measure.** Every quantity this audit needed that you could not obtain.
As in the audits for {{part:22}} through {{part:25}}, this is the most valuable section, and
here it will usually be dominated by the attacker's attempt budget and the harm-to-refusal cost
ratio.

Length: eight to twelve pages and a spreadsheet.

## Challenge problems

**A. The attempt budget.** Every detection-versus-capability result in this part turns on how
many attempts an attacker gets, and {{cite:zou2023universal}}'s transferability means that
number is not observable from your logs. Design a method for estimating it — from
transferability measurements, from public attack corpora, from honeypot data — and compute what
your controls' asymptotes actually are under your estimate.

**B. Correlated layers.** {{ch:sec-threat-model}} multiplies detection-layer miss rates and
{{ch:sec-poisoning}} multiplies supply-chain link probabilities, both assuming independence.
Model realistic correlation in each and determine how much of each chapter's conclusion
survives. Which recommendation, if any, reverses?

**C. The reachable subset graph.** {{ch:sec-tool-abuse}} counts $2^n$ tool subsets and notes
that most are unreachable. Derive the reachable graph from execution traces, price the top
compositions, and determine how much of the exponential is real.

**D. Typed attenuation.** {{ch:sec-permissions}} models attenuation as uniform scaling and
{{sec:15-advanced-concepts}} argues typed attenuation — removing whole capabilities — is
strictly better. Specify a typed attenuation scheme for a heterogeneous tool set, including how
a hop knows what to remove, and price it against per-user delegation.

**E. The cost ratio.** Three chapters in this part end at a cost ratio nobody states: harm to
refusal, exposure to hit rate, damage to utility. Design a process that elicits one of them
from incident history or revealed preference, run it, and report what the current threshold
implies about the ratio your organisation is behaving as though it holds.

**F. The whole-part residual.** Combine {{ch:sec-threat-model}}'s capability model,
{{ch:sec-tool-abuse}}'s composition damage, {{ch:sec-poisoning}}'s chain trust and
{{ch:sec-permissions}}'s approval catch into one residual-risk figure for a specific system.
Which term dominates, and does the ranking of remedies change when they are optimised jointly
rather than separately?

## Interview preparation

Rehearse these until the answer is a structure rather than a recollection.

1. "We have delimiters and an ignore-instructions line." — A string the attacker reads, then
   the 8% figure.
2. "Our injection classifier blocks 94% of known attacks." — Fixed set versus adaptive, then
   the asymptote.
3. "We sandbox the agent." — Which axis, then the two rows a sandbox does not touch.
4. "We rate-limit, so injection is handled." — The indirect channel, then transferability.
5. "The corpus has four hundred million items." — Cost is per fraction, then the $6,000.
6. "We verify all model signatures." — Identity versus behaviour, then the unsignable floor.
7. "Our guardrail has a 91% true-positive rate." — Base rate, then precision, then refusals per
   prevented harm.
8. "Search and email are both approved." — Composition, then the sum of parts.
9. "We approve every tool call." — Seconds per item, then rejection density, then $K(v)$.
10. "Our audit log records who called the API." — 11% of questions, then the principal chain.
11. "We deleted the malicious document." — Derived stores, then the fan-out.
12. "Our semantic cache saves 30% of calls." — Tenants sharing it, then 433 to one.

The pattern across all twelve: **name what the control bounds, ask whether the attacker can
retry, and price the subtraction that would remove the reach instead.** That is the part in one
sentence and the question worth having ready.
