---
id: sec-threat-model
number: 221
part: XXVI
tier: full
status: draft
requires: [three-properties-break-the-stack, fewer-tools-beats-better-credentials,
           delegation-moves-the-check, semantic-failure-has-no-instrument]
provides: [instructions-and-data-share-a-channel, attack-surface-is-sources-times-sinks,
           detection-layers-fail-against-an-adaptive-attacker, only-capability-limits-bound-the-damage]
citations: [greshake2023indirect, perez2022ignore, zou2023universal,
            debenedetti2024agentdojo, beurerkellner2025patterns]
---

## 1. Learning Objectives

By the end of this chapter you will be able to enumerate the content sources reaching a
model's context and assign each a trust level; explain why the composite prompt's effective
privilege is the maximum over its sources while its effective intent can come from the
minimum; compute attack surface as a product of untrusted sources and privileged sinks, and
show how it grows; distinguish detection-based controls from capability-based ones and show
that only the second survives a repeated attacker; and price a security architecture in
utility rather than presenting it as free.

## 2. Why This Matters

Classical security separates code from data with a parser. A parameterised SQL query tells
the database structurally which bytes are code, and no content inside the data can move that
line. An attacker who controls data controls data.

A language model receives one sequence. In a typical agent context, **79% of the tokens are
content the system did not author** — retrieved documents, tool results, opened files,
replies from other agents — and the model can distinguish none of it from the system prompt
except by position ({{eq:instructions-and-data-share-a-channel}}). Position is a convention
learned in training, not a boundary.

So the composite prompt carries the highest privilege present while its intent can come from
the lowest-trust source, and the attack surface is a **product**: 4 untrusted sources × 8
sinks = **32 reachable paths** ({{eq:attack-surface-is-sources-times-sinks}}). Doubling each
side once takes the surface from 32 to **128**.

The second half is what to do about it, and the arithmetic there is the part most often got
wrong. Four stacked detection layers miss only **12.33%** of a fixed attack — but an attacker
who observes whether an attempt was blocked is running a search, and the stack is defeated
**32.6% of the time within 3 attempts and 100% within 100**
({{eq:detection-layers-fail-against-an-adaptive-attacker}}).

Capability limits do not move: **0.2436%** at one attempt and at a thousand
({{eq:only-capability-limits-bound-the-damage}}), because a tool that is not on the
allow-list is not reachable by a better-phrased request. Which means residual risk is
essentially blast radius, and the largest single move available — proposal-only execution —
takes it from **100 to 3**.

## 3. Prerequisites

{{eq:three-properties-break-the-stack}} from {{ch:sd-architecture}} named the properties that
break conventional engineering assumptions; this chapter adds the fourth, which is that the
trust boundary is inside the prompt and has no enforcement.

{{eq:fewer-tools-beats-better-credentials}} from {{ch:sd-apis-auth}} is this chapter's
attack-surface result arriving from the reliability side. There the argument was about
confusion; here it is about reachability, and the arithmetic is the same product.

{{eq:delegation-moves-the-check}} from the same chapter is the structural half of the
defence: where the authorisation check sits determines what a fooled model can reach.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is why detection is
hard here specifically — an injected instruction produces a well-formed successful action,
which is the failure class no health check sees.

{{cite:greshake2023indirect}} established indirect prompt injection as a practical
compromise of real LLM-integrated applications, and is the reason the retrieved-document row
in {{sec:9-practical-example}}'s source table is trust level zero.

## 4. Intuitive Explanation

Start with what makes this different, because "prompt injection is like SQL injection" is
the analogy everyone reaches for and it is misleading in the direction that matters.

SQL injection was solved. Not mitigated — solved, by parameterised queries. The fix works
because the database can be told, structurally, which bytes are the query and which are the
values, and nothing inside the values can change that. Same for shell arguments passed as an
argv array rather than a string, same for HTML with a parser that knows where the attribute
ended.

In all three cases the guarantee is: **an attacker who controls the data controls the data.**
They can make the data wrong. They cannot make it be code.

A language model has no such seam. The system prompt, the developer's template, the user's
message, a retrieved document, a tool result, a file the user opened, and a reply from
another agent all arrive as tokens in one sequence. There is no field that says "these tokens
may issue instructions and those may not."

What there is instead is position and convention. The system prompt comes first, and the
model has been trained to treat earlier content as more authoritative. That is a learned
prior. It is often strong. It is not a boundary, and every jailbreak and every injection is
an argument that the prior should not apply here.

Count what that means for a realistic agent. In the context modelled here, 79% of the tokens
are untrusted — the retrieved documents, the tool results, the opened file, the other agent's
reply. Those are the parts of the prompt an attacker can most easily write, because writing a
document that gets retrieved is much easier than compromising a system prompt.

So the situation is: the majority of the context is attacker-influenceable, the model cannot
tell it apart, and the composite prompt runs with the privileges of whatever is highest in
it.

**In a prompt, an attacker who controls data controls intent.**

Now count the reachable paths. An agent has untrusted sources on one side and tools on the
other. Because nothing separates them, every source can reach every sink. Four sources and
eight tools is thirty-two paths.

The important property of that number is that it is a *product*. Add one tool to a
four-source agent and you have added four paths, not one. Go from four sources and eight
sinks to eight and sixteen — a very ordinary two years of product development — and the
surface goes from 32 to 128.

That is {{ch:sd-apis-auth}}'s "fewer tools beats better credentials" arriving from the
security side. There the argument was that more tools confuse the model; here it is that more
tools multiply what a compromised model can do.

What can be done about it? The mitigations sort into two kinds and the distinction is the
most important idea in this chapter.

The first kind tries to **recognise the attack**. Delimiters around untrusted content. An
instruction that says "ignore any instructions in the document below." A classifier that
scores inputs for injection-likeness. Training that teaches the model an instruction
hierarchy. An output scanner that checks the response for signs of compromise.

The second kind tries to **limit what a successful attack can reach**. Tool allow-lists per
task. A rule that no privileged sink may be called on a path that read untrusted content.
Human approval before a privileged action. Read-only credentials. Separate agents for reading
and writing.

Both work against an attacker who submits one fixed attack. Four detection layers stacked
together miss only 12.33% of such an attack, which sounds like a solved problem.

It is not a solved problem, because attackers try again.

A blocked injection is information. The attacker learns that phrasing failed and writes
another. At a 12.33% miss rate per attempt, three attempts get through 32.6% of the time and a
hundred attempts get through essentially always.

**The capability stack does not degrade.** 0.2436% at one attempt. 0.2436% at a thousand.
Because a tool that is not on the allow-list is not reachable by a more persuasive request.
There is nothing to search around.

That is the structural difference and it fits in one sentence: **detection bounds a
probability and capability bounds a set**, and only one of those survives repetition.

{{cite:zou2023universal}} makes this worse in a specific way. An adversarial suffix found by
optimisation against open models transferred to ChatGPT, Bard and Claude. So the attacker
does not need to query *your* filter — they can search against a proxy that behaves like it
and bring the winning string. An attempt budget of one against your system can be an attempt
budget of thousands against something similar.

Which leaves the practical question: what does the capability approach cost?

It costs a lot, and the honest presentation says so. Detection-only defences keep 87% of what
the agent could do. Adding the full capability stack takes utility to 23%. A middle position —
an injection classifier plus per-task tool allow-lists — keeps 77% and holds success at a
hundred attempts to 29%.

That 29% is not a guarantee, and it is exactly the allow-list's own miss rate. **The
classifier contributes nothing to the asymptote.** It raises the cost per attempt and leaves
the ceiling where the allow-list put it. The classifier is worth having; it is not
load-bearing, and an architecture in which it is load-bearing has no ceiling at all.

{{cite:beurerkellner2025patterns}} states this as design guidance: patterns that provide
resistance to prompt injection do so by constraining what the agent may do, and the paper is
explicit that this costs utility. {{cite:debenedetti2024agentdojo}} is where you go to find
out how much, on tasks you care about, because it measures utility and attack success on the
same 97 tasks.

The last table is the one to take to whoever signs off. Residual risk is probability times
blast radius. Against a repeated attacker, detection-only defences leave the probability term
near one — so **the residual is essentially the blast radius**, and every entry in that
column is a product decision. Unrestricted tools: 100. Read-only: 34. Scoped credentials to
one tenant: 12. Proposal-only with a human executing: 3.

Going from the first to the last is the largest single move available in AI security, and it
is not a security control. It is a decision about what the product does.

## 5. Formal Explanation

**The channel.** Let a prompt be a concatenation $P = s_1 \| s_2 \| \cdots \| s_n$ of
segments from sources with trust levels $\tau_i$ and privileges $\pi_i$. A security
architecture with a parser guarantees that the *interpretation* of $s_i$ is a function of
$\tau_i$ alone. A language model has no such guarantee: the interpretation of the whole is a
function of the whole, so

$$\pi_{\text{eff}}(P) = \max_i \pi_i \quad \text{while} \quad \tau_{\text{eff}}(P) = \min_i \tau_i,$$

and privilege and trust are decoupled. This is the formal content of "instructions and data
share a channel."

**The surface.** With $S$ untrusted sources and $K$ sinks, and no separation, the number of
reachable (source, sink) paths is $|S| \cdot |K|$, and exposure weighted by damage $d_k$ is
$|S| \sum_k d_k$. Both are multiplicative in the number of sources, so a mitigation that
reduces the *fraction* of paths reachable is worth more than one that removes a single tool.

**Detection under repetition.** Let a stack of detection layers have per-attempt miss
probability $\mu_d = \prod_j (1 - p_j)$. Against an attacker with $k$ attempts and feedback,

$$\Pr[\text{success within } k] = 1 - (1 - \mu_d)^k \xrightarrow{k \to \infty} 1$$

for any $\mu_d > 0$. The limit is one regardless of how many layers are stacked, because
stacking changes $\mu_d$ and not the limit.

**Capability under repetition.** A capability control removes a path rather than scoring a
request. Its miss probability $\mu_c$ is the probability the path was not covered — a property
of the configuration, not of the request — so repetition does not compound it:

$$\Pr[\text{success within } k] = \mu_c \quad \text{for all } k.$$

Composing the two gives $\mu_c \cdot [1 - (1-\mu_d)^k]$, which converges to $\mu_c$: **the
asymptote is set entirely by the capability layer**, and the detection layer determines only
how quickly the attacker reaches it.

**Residual risk.** With blast radius $B$, residual is $\Pr[\text{success}] \cdot B$. Under
detection-only defences and a realistic attempt budget, $\Pr[\text{success}] \approx 1$, so
$\partial(\text{residual})/\partial B = 1$ and $\partial(\text{residual})/\partial \mu_d
\approx 0$.

## 6. Mathematical Foundation

Privilege and trust decoupled by a shared channel:

$$\pi_{\text{eff}}(P) = \max_i \pi_i, \qquad \tau_{\text{eff}}(P) = \min_i \tau_i, \qquad \pi_{\text{eff}} \perp \tau_{\text{eff}}$$ (eq:instructions-and-data-share-a-channel)

In the context measured here, $\max_i \pi_i = 3$, $\min_i \tau_i = 0$, and **79%** of tokens
sit at $\tau = 0$.

Attack surface as a product:

$$A = |S| \cdot |K|, \qquad E = |S| \sum_{k} d_k, \qquad \frac{\partial A}{\partial |K|} = |S|$$ (eq:attack-surface-is-sources-times-sinks)

At $|S| = 4$, $|K| = 8$: **32 paths**, exposure **232**. At $|S| = 8$, $|K| = 16$: **128**.

Detection against a repeated attacker:

$$\Pr[\text{success} \mid k] = 1 - (1 - \mu_d)^k, \qquad \lim_{k\to\infty} = 1 \ \ \forall\, \mu_d > 0$$ (eq:detection-layers-fail-against-an-adaptive-attacker)

At $\mu_d = 0.1233$: **32.63%** at $k=3$, **100%** at $k=100$.

And the asymptote a capability limit sets:

$$\Pr[\text{success} \mid k] = \mu_c\left[1 - (1-\mu_d)^k\right] \xrightarrow{k\to\infty} \mu_c, \qquad \text{residual} \approx B \ \text{when}\ \mu_d > 0$$ (eq:only-capability-limits-bound-the-damage)

$\mu_c = 0.2436\%$, invariant in $k$; residual falls from **100 to 3** as $B$ falls.

## 7. Internal Mechanics

Why is there no parser? Not because nobody has tried. Several designs exist that mark
untrusted spans with special tokens, or train the model to obey a hierarchy, or run untrusted
content through a separate encoder. All of them raise the cost of an attack and none of them
provides the SQL-style guarantee, for a reason that is worth understanding.

A parameterised query works because the database's *behaviour* is defined by a grammar, and
the grammar is enforced by code that is not influenced by the values. A language model's
behaviour is defined by its weights, and the weights were trained on data in which
instruction-following is contextual. There is no component in the system whose behaviour is
independent of the untrusted content, because the untrusted content is an input to the only
component that decides anything.

So the boundary has to be built *outside* the model — in what the model is permitted to call,
not in what it is permitted to believe. That is the whole of
{{cite:beurerkellner2025patterns}}'s argument, and it follows from where the enforcement can
live rather than from any claim about how good models will get.

The attack-surface product has a mechanism worth naming because it explains a common
organisational pattern. Sources and sinks are added by different teams. A retrieval feature
adds a source; an integration adds a sink; neither team is adding an attack path, and the
product of their independent decisions is. Nobody is reviewing the multiplication because
nobody owns both factors.

The adaptive-attacker result explains why security reviews of AI systems so often produce a
false sense of adequacy. A review tests a set of known attacks against the defences and
measures a block rate. That measurement is the single-attempt column, and it is the column
that looks fine. The relevant quantity — the block rate against an attacker who iterates — is
not measurable by testing a fixed set, because the fixed set is by construction not adaptive.
**A red team that submits its findings and stops has measured $\mu_d$, and the deployed system
faces $1 - (1-\mu_d)^k$.**

The transferability result compounds this in a way that breaks the usual rate-limiting
mitigation. Rate limits bound $k$ against your system, and {{cite:zou2023universal}}'s finding
is that the search can happen elsewhere. So the effective $k$ is not the number of requests
your system accepted; it is the number of requests the attacker made against anything with
similar behaviour, which you cannot observe or bound.

Finally, the reason capability limits are underused despite this arithmetic. They cost
utility, visibly, at design time, to a named feature — and the detection alternative costs
utility invisibly, later, in false positives on real users. One of those shows up in a
planning meeting and the other shows up in a support queue, and the planning meeting is where
the decision gets made.

## 8. Implementation

The first listing counts the trust boundary and the attack surface.

```python {tier=A name=instructions-and-data-share-a-channel}
"""Every security architecture separates code from data. A prompt does not have that seam.

A SQL injection is prevented by a parameterised query: the database is told, structurally,
which bytes are code and which are data, and no content of the data can change that. A
cross-site scripting attack is prevented the same way, by a parser that knows where the
markup ended.

A language model receives one sequence. The system prompt, the user's message, a retrieved
document, a tool result and a file's contents arrive as tokens in the same channel, and
nothing in the interface marks which of them may issue instructions
(eq:instructions-and-data-share-a-channel).

So the composite prompt carries the *highest* privilege of any source in it, while the
intent can come from the lowest-trust one, and the attack surface is the product of sources
and sinks rather than their sum
(eq:attack-surface-is-sources-times-sinks).
"""
# (source, trust level 0-3, mean tokens per request, model can tell it apart?)
SOURCES = [
    ("system prompt",        3, 900,  "by position only"),
    ("developer templates",  3, 240,  "by position only"),
    ("user message",         1, 180,  "sometimes"),
    ("retrieved document",   0, 2400, "no"),
    ("tool result",          0, 1100, "no"),
    ("file the user opened", 0, 3200, "no"),
    ("prior turn's output",  1, 700,  "no"),
    ("another agent's reply", 0, 850, "no"),
]

print("What arrives in the model's context, and at what trust level.")
print()
print(f"{'source':>24}{'trust':>8}{'tokens':>9}{'share of context':>19}"
      f"{'distinguishable?':>19}")
print("-" * 79)
total_tokens = sum(t for n, tr, t, d in SOURCES)
untrusted_tokens = sum(t for n, tr, t, d in SOURCES if tr == 0)
for name, tr, tok, dis in SOURCES:
    print(f"{name:>24}{tr:>8}{tok:>9,}{tok / total_tokens:>19.1%}{dis:>19}")
print("-" * 79)
print(f"{'TOTAL':>24}{'':>8}{total_tokens:>9,}{1.0:>19.1%}")
print()
print(f"untrusted (trust 0) content is {untrusted_tokens / total_tokens:.0%} "
      f"of the context")
print(f"highest privilege present: {max(tr for n, tr, t, d in SOURCES)}")
print(f"lowest trust present:      {min(tr for n, tr, t, d in SOURCES)}")

print()
print()
print("The composite prompt's effective privilege, against a classical system.")
print()
print(f"{'system':>30}  {'privilege rule':<32}{'effective privilege':>22}")
print("-" * 86)
COMPARE = [
    ("parameterised SQL query",   "data cannot become code", "the caller's"),
    ("shell with quoted args",    "argv is not re-parsed",   "the caller's"),
    ("HTML with escaping",        "the parser knows where data ends", "the caller's"),
    ("LLM prompt",                "max over all sources",    "the highest present"),
]
for name, rule, eff in COMPARE:
    print(f"{name:>30}  {rule:<32}{eff:>22}")

print()
print("In the first three rows an attacker who controls data controls data.")
print("In the fourth, an attacker who controls data controls intent.")

print()
print()
print("Attack surface: which sources can reach which sinks.")
print()
# (sink, privilege required, damage if reached by an attacker)
SINKS = [
    ("send an email",          1, 6.0),
    ("read a document",        1, 3.0),
    ("write to the CRM",       2, 8.0),
    ("issue a refund",         3, 9.5),
    ("call an external API",   1, 5.0),
    ("run a database query",   2, 7.5),
    ("execute code",           3, 10.0),
    ("read a secret",          3, 9.0),
]
untrusted = [n for n, tr, t, d in SOURCES if tr == 0]
print(f"{'sink':>24}{'privilege':>11}{'damage':>9}"
      f"{'reachable from untrusted?':>28}{'exposure':>11}")
print("-" * 83)
exposure = 0.0
for name, priv, dmg in SINKS:
    reach = len(untrusted)          # nothing separates them, so all of them
    exposure += reach * dmg
    print(f"{name:>24}{priv:>11}{dmg:>9.1f}"
          f"{('yes, all ' + str(reach) + ' sources'):>28}{reach * dmg:>11.1f}")
print("-" * 83)
print(f"{'TOTAL EXPOSURE':>24}{'':>11}{'':>9}{'':>28}{exposure:>11.1f}")

print()
print()
print("How that grows. Sources and sinks both increase with product surface.")
print()
print(f"{'untrusted sources':>19}", end="")
for s in (2, 4, 8, 16, 32):
    print(f"{(str(s) + ' sinks'):>12}", end="")
print()
print("-" * 79)
for src in (1, 2, 4, 8):
    print(f"{src:>19}", end="")
    for s in (2, 4, 8, 16, 32):
        print(f"{src * s:>12}", end="")
    print()

print()
print("Pairs, not rows plus columns. Adding one tool to a four-source agent")
print("adds four reachable paths, not one.")

print()
print()
print("What each mitigation removes, counted in source-sink pairs.")
print()
BASE_PAIRS = len(untrusted) * len(SINKS)
MITIGATIONS = [
    ("nothing",                              1.00, 0.00, "the model decides"),
    ("delimiters and 'ignore instructions'", 0.92, 0.01, "a string the attacker reads"),
    ("injection classifier on inputs",       0.55, 0.06, "detection, see listing 2"),
    ("instruction-hierarchy training",       0.41, 0.03, "a prior, not a boundary"),
    ("taint tracking, block tainted sinks",  0.14, 0.31, "structural"),
    ("untrusted content never reaches a sink", 0.00, 0.62, "structural"),
]
print(f"{'mitigation':>42}{'pairs left':>13}{'utility cost':>15}"
      f"{'kind':>30}")
print("-" * 100)
mit = {}
for name, frac, util, kind in MITIGATIONS:
    pairs = BASE_PAIRS * frac
    mit[name] = (pairs, util)
    print(f"{name:>42}{pairs:>13.1f}{util:>15.0%}{kind:>30}")

print()
print(f"base: {len(untrusted)} untrusted sources x {len(SINKS)} sinks "
      f"= {BASE_PAIRS} pairs")

print()
print()
print("And the exposure-weighted version, which is what a design review needs.")
print()
print(f"{'mitigation':>42}{'weighted exposure':>20}{'vs nothing':>13}"
      f"{'utility kept':>15}")
print("-" * 90)
for name, frac, util, kind in MITIGATIONS:
    w = exposure * frac
    print(f"{name:>42}{w:>20.1f}{w / exposure:>12.0%}{1 - util:>15.0%}")

print(f"""
The source table is the shape of the problem. {untrusted_tokens / total_tokens:.0%} of a
typical context is content the system did not author and cannot vouch for -- retrieved
documents, tool results, files, replies from other agents -- and the fourth column is the
one that matters: **the model cannot tell any of it apart from the system prompt except by
position** (eq:instructions-and-data-share-a-channel).

Position is not a boundary. It is a convention the model learned during training and can be
argued out of, which is exactly what every jailbreak and injection does.

The comparison table says why this is different in kind rather than in degree. A
parameterised query, a quoted argv, an escaped HTML attribute -- in all three the parser is
told structurally where data ends, and no content inside the data can move that line. An
attacker who controls data controls data.

**In a prompt, an attacker who controls data controls intent.** The composite's effective
privilege is the maximum over its sources, and its effective intent can come from the
minimum.

The sink table counts what that reaches. {len(untrusted)} untrusted sources, {len(SINKS)}
sinks, and because nothing separates them, every source can reach every sink:
{BASE_PAIRS} paths with a total exposure weight of {exposure:.0f}
(eq:attack-surface-is-sources-times-sinks).

The growth table is the part that surprises teams. Attack surface is a *product*, so adding
one tool to an agent with four untrusted sources adds four paths. Going from 4 sources and 8
sinks to 8 and 16 takes the surface from {4 * 8} to {8 * 16} --
**{(8 * 16) / (4 * 8):.0f} times, from doubling each side once.**

That is the arithmetic behind ch:sd-apis-auth's finding that fewer tools beats better
credentials, arriving from the security side rather than the reliability side.

The mitigation table is where the chapter's recommendation comes from, and its two columns
have to be read together. Delimiters and "ignore any instructions in the document below"
remove {1 - 0.92:.0%} of pairs, cost almost nothing, and are **a string the attacker can
read**. An injection classifier removes {1 - 0.55:.0%}. Instruction-hierarchy training
removes {1 - 0.41:.0%} and is a learned prior rather than a boundary -- it makes the wrong
behaviour less likely and does not make it impossible.

Taint tracking removes {1 - 0.14:.0%} of pairs and costs {0.31:.0%} of utility, and it is the
first row in the table that is **structural**: it does not ask whether content looks
malicious, it asks whether a privileged action is being taken on a path that touched
untrusted input. That question has an answer that does not depend on the attacker's
cleverness.

And the last row -- untrusted content never reaches a sink at all -- removes everything and
costs {0.62:.0%} of what the agent could do. That is cite:beurerkellner2025patterns' central
trade stated as a number: **the patterns that provide a guarantee provide it by removing
capability**, and the honest way to present a secure agent design is with the utility column
visible.

Which sets up the question the second listing has to answer. If detection is the cheap
mitigation and structure is the expensive one, how much of the cheap one can be substituted
for the expensive one? The answer depends on whether the attacker gets to try twice.""")
```

## 9. Practical Example

What reaches the model's context:

```
                  source   trust   tokens   share of context   distinguishable?
-------------------------------------------------------------------------------
           system prompt       3      900               9.4%   by position only
            user message       1      180               1.9%          sometimes
      retrieved document       0    2,400              25.1%                 no
             tool result       0    1,100              11.5%                 no
    file the user opened       0    3,200              33.4%                 no
   another agent's reply       0      850               8.9%                 no
-------------------------------------------------------------------------------
                   TOTAL            9,570             100.0%
```

**79% of the context is untrusted and none of it is distinguishable except by position**
({{eq:instructions-and-data-share-a-channel}}). Highest privilege present: 3. Lowest trust
present: 0.

```
                        system  privilege rule                     effective privilege
--------------------------------------------------------------------------------------
       parameterised SQL query  data cannot become code                   the caller's
        shell with quoted args  argv is not re-parsed                     the caller's
            HTML with escaping  the parser knows where data ends          the caller's
                    LLM prompt  max over all sources               the highest present
```

In the first three rows, an attacker who controls data controls data. **In the fourth, an
attacker who controls data controls intent.**

```
                    sink  privilege   damage   reachable from untrusted?   exposure
-----------------------------------------------------------------------------------
           send an email          1      6.0          yes, all 4 sources       24.0
          issue a refund          3      9.5          yes, all 4 sources       38.0
            execute code          3     10.0          yes, all 4 sources       40.0
           read a secret          3      9.0          yes, all 4 sources       36.0
-----------------------------------------------------------------------------------
          TOTAL EXPOSURE                                                      232.0
```

```
  untrusted sources     2 sinks     4 sinks     8 sinks    16 sinks    32 sinks
-------------------------------------------------------------------------------
                  1           2           4           8          16          32
                  4           8          16          32          64         128
                  8          16          32          64         128         256
```

**Surface is a product** ({{eq:attack-surface-is-sources-times-sinks}}) — adding one tool to a
four-source agent adds four paths, and doubling each side once takes 32 to 128.

```
                                mitigation   pairs left   utility cost                          kind
----------------------------------------------------------------------------------------------------
                                   nothing         32.0             0%             the model decides
      delimiters and 'ignore instructions'         29.4             1%   a string the attacker reads
            injection classifier on inputs         17.6             6%      detection, see listing 2
            instruction-hierarchy training         13.1             3%       a prior, not a boundary
       taint tracking, block tainted sinks          4.5            31%                    structural
    untrusted content never reaches a sink          0.0            62%                    structural
```

Read the last two columns together. **The mitigations that provide a guarantee provide it by
removing capability**, and the honest presentation of a secure design has the utility column
visible.

The second listing asks how much detection can substitute for structure.

```python {tier=A name=detection-layers-fail-against-an-adaptive-attacker}
"""Detection layers compose beautifully against an attacker who only tries once.

Defence in depth is the right instinct and the arithmetic is usually done wrong. Stacked
detectors multiply their miss rates, which looks excellent -- until you notice that the
multiplication assumes a *fixed* attack. An attacker who can observe whether an attempt was
blocked is running a search, and search defeats detection at a rate set by how many attempts
they get (eq:detection-layers-fail-against-an-adaptive-attacker).

The layers that survive are the ones whose guarantee does not depend on recognising the
attack: what a successful injection is permitted to reach. Those bound the damage rather
than the probability, and the bound holds at any number of attempts
(eq:only-capability-limits-bound-the-damage).
"""
# (layer, P(blocks a fixed attack), utility cost, depends on detection?)
LAYERS = [
    ("delimiters and warnings",       0.08, 0.01, True),
    ("input injection classifier",    0.62, 0.06, True),
    ("instruction-hierarchy training", 0.44, 0.03, True),
    ("output scanner",                0.37, 0.04, True),
    ("tool allow-list per task",      0.71, 0.18, False),
    ("no privileged sink after untrusted read", 0.93, 0.34, False),
    ("human approval on privileged sinks", 0.88, 0.51, False),
]

print("Each layer alone, against an attacker who submits one fixed attack.")
print()
print(f"{'layer':>44}{'blocks':>9}{'utility cost':>15}"
      f"{'detection-based?':>19}")
print("-" * 87)
for name, p, u, det in LAYERS:
    print(f"{name:>44}{p:>9.0%}{u:>15.0%}"
          f"{('yes' if det else 'no'):>19}")

det_layers = [l for l in LAYERS if l[3]]
cap_layers = [l for l in LAYERS if not l[3]]


def miss(layers):
    m = 1.0
    for name, p, u, det in layers:
        m *= (1.0 - p)
    return m


def utility(layers):
    u = 1.0
    for name, p, uc, det in layers:
        u *= (1.0 - uc)
    return u


print()
print(f"all four detection layers together miss {miss(det_layers):.2%}")
print(f"all three capability layers together miss {miss(cap_layers):.3%}")

print()
print()
print("Now let the attacker try again after each block. Detection layers are")
print("a filter to be searched around; capability limits are not.")
print()
print("(all three columns are P(the attacker succeeds at least once))")
print()
print(f"{'attempts':>10}{'detection stack':>19}{'capability stack':>20}"
      f"{'both':>12}{'vs 1 attempt':>16}")
print("-" * 77)
md, mc = miss(det_layers), miss(cap_layers)
adapt = {}
for k in (1, 3, 10, 30, 100, 1000):
    d = 1.0 - (1.0 - md) ** k
    # A capability limit does not admit search: the sink is unreachable on that
    # path however the request is phrased, so only its own miss rate applies.
    c = mc
    both = c * d
    adapt[k] = (d, c, both)
    print(f"{k:>10}{d:>19.2%}{c:>20.4%}{both:>12.4%}"
          f"{d / md:>15.1f}x")

print()
print("The middle column does not move. That is the whole argument.")

print()
print()
print("What each stack costs in utility, against what it bounds.")
print()
print(f"{'stack':>34}{'utility kept':>15}{'success at 1 try':>19}"
      f"{'success at 100':>17}{'success at 1000':>18}")
print("-" * 103)
STACKS = [
    ("nothing",                       []),
    ("detection only",                det_layers),
    ("capability only",               cap_layers),
    ("detection + capability",        LAYERS),
    ("cheapest detection + tool allow-list",
     [LAYERS[1], LAYERS[4]]),
]
st = {}
for label, ls in STACKS:
    if not ls:
        s1 = s100 = s1000 = 1.0
        u = 1.0
    else:
        dl = [l for l in ls if l[3]]
        cl = [l for l in ls if not l[3]]
        md = miss(dl) if dl else 1.0
        mc = miss(cl) if cl else 1.0
        s1 = md * mc
        s100 = (1.0 - (1.0 - md) ** 100 if dl else 1.0) * mc
        s1000 = (1.0 - (1.0 - md) ** 1000 if dl else 1.0) * mc
        u = utility(ls)
    st[label] = (u, s1, s100, s1000)
    print(f"{label:>34}{u:>15.0%}{s1:>19.3%}{s100:>17.3%}{s1000:>18.3%}")

print()
print("Read the last two columns across. Detection-only degrades toward 1;")
print("anything with a capability limit converges to that limit.")

print()
print()
print("Residual risk = P(success) x blast radius. Only one term responds.")
print()
BLAST = [
    ("no restriction",                 100.0),
    ("read-only tools",                 34.0),
    ("read-only plus rate limit",       21.0),
    ("scoped credentials, one tenant",  12.0),
    ("proposal only, human executes",    3.0),
]
print(f"{'blast-radius control':>34}{'radius':>10}"
      f"{'residual, detection only':>27}{'residual, det + cap':>22}")
print("-" * 93)
res = {}
for name, radius in BLAST:
    r_det = st["detection only"][2] * radius
    r_both = st["detection + capability"][2] * radius
    res[name] = (radius, r_det, r_both)
    print(f"{name:>34}{radius:>10.0f}{r_det:>27.3f}{r_both:>22.4f}")

print()
print(f"detection-only at 100 attempts: P(success) = "
      f"{st['detection only'][2]:.1%}")
print("so the residual is essentially the blast radius")

print()
print()
print("Where a fixed security budget should go.")
print()
# (investment, blocks a fixed attack, detection-based?, utility cost, weeks)
BUDGET_ITEMS = [
    ("tune the injection classifier",     0.62, True,  0.06, 4.0),
    ("add an output scanner",             0.37, True,  0.04, 3.0),
    ("scope tool allow-lists by task",    0.71, False, 0.18, 2.5),
    ("split read and write agents",       0.93, False, 0.34, 8.0),
    ("human approval on the top 3 sinks", 0.88, False, 0.11, 1.5),
]
ATTEMPTS = 100
print(f"{'investment':>38}{'blocks a fixed attack':>24}"
      f"{'blocks at 100 attempts':>25}{'weeks':>8}{'per week':>11}")
print("-" * 106)
inv = {}
for name, fixed, det, u, weeks in BUDGET_ITEMS:
    if det:
        eff = fixed ** ATTEMPTS              # P(all 100 attempts blocked)
    else:
        eff = fixed                          # not searchable
    inv[name] = (fixed, eff, weeks, eff / weeks)
    print(f"{name:>38}{fixed:>24.0%}{eff:>25.1%}{weeks:>8.1f}"
          f"{eff / weeks:>11.3f}")

print()
print("Detection items round to zero against an adaptive attacker with 100")
print("attempts, which is the honest way to score them.")

print(f"""
The single-layer table is the ordinary defence-in-depth picture, and read alone it is
encouraging. Four detection layers together miss only {miss(det_layers):.2%} of a fixed
attack; three capability layers miss {miss(cap_layers):.3%}. Either stack looks adequate.

The adaptive table is what happens when the attacker gets feedback. A blocked injection is
information -- the attacker knows that phrasing failed and tries another. At
{miss(det_layers):.2%} miss per attempt, the detection stack is defeated
{adapt[3][0]:.1%} of the time within {3} attempts and {adapt[100][0]:.1%} within
{100} (eq:detection-layers-fail-against-an-adaptive-attacker).

**The capability column does not move.** {adapt[1][1]:.4%} at one attempt,
{adapt[1000][1]:.4%} at a thousand -- because a tool that is not on the allow-list is not
reachable by a better-phrased request. There is nothing to search around.

That is the structural difference and it is worth stating in one sentence: **detection
bounds a probability and capability bounds a set**, and only one of those survives
repetition.

cite:zou2023universal is the empirical form of the same point. An adversarial suffix found
by optimisation on open models transferred to ChatGPT, Bard and Claude -- which means the
attacker does not even need to query your filter, because they can search against a proxy
and bring the result. An attempt budget of one against your system can still be an attempt
budget of thousands against something that behaves like it.

The stack table prices the choice. Detection alone keeps
{st['detection only'][0]:.0%} of utility and lets {st['detection only'][2]:.1%} through at
100 attempts. Adding capability limits takes utility to
{st['detection + capability'][0]:.0%} and success at 100 attempts to
{st['detection + capability'][2]:.3%}.

The last row is the practical middle. An injection classifier plus per-task tool allow-lists
keeps {st['cheapest detection + tool allow-list'][0]:.0%} of utility and holds success at a
hundred attempts to {st['cheapest detection + tool allow-list'][2]:.0%} -- not a guarantee,
and against detection-only's {st['detection only'][2]:.0%} it is the difference between a
bounded and an unbounded risk.

Note where the bound comes from: {st['cheapest detection + tool allow-list'][2]:.0%} is
exactly the allow-list's own miss rate. **The classifier contributes nothing to the
asymptote** -- it raises the cost per attempt and leaves the ceiling where the allow-list put
it. The classifier is not useless; it is not load-bearing, and an architecture in which it is
load-bearing has no ceiling at all.

The residual table is how to present this to whoever signs off. Residual risk is
probability times blast radius, and against an adaptive attacker the probability term is
close to one under detection-only defences -- {st['detection only'][2]:.0%} at a hundred
attempts. So **the residual is essentially the blast radius**
(eq:only-capability-limits-bound-the-damage), and every row in that table is a product
decision rather than a security control.

Going from unrestricted tools to proposal-only with a human executing takes the radius from
{BLAST[0][1]:.0f} to {BLAST[4][1]:.0f}, and that is the largest single move available in
this chapter.

The budget table is the ranking, and it scores detection items at zero against an adaptive
attacker -- which is harsh and is the right convention, because a control that a hundred
attempts defeat should not be credited with preventing anything at a hundred attempts.
`{BUDGET_ITEMS[4][0]}` returns {inv[BUDGET_ITEMS[4][0]][3]:.3f} per week,
`{BUDGET_ITEMS[2][0]}` returns {inv[BUDGET_ITEMS[2][0]][3]:.3f}, and the two detection items
return {inv[BUDGET_ITEMS[0][0]][3]:.3f} and {inv[BUDGET_ITEMS[1][0]][3]:.3f}.

The honest summary for ch:sec-threat-model is uncomfortable and short. **Assume the model
will be fooled**, spend on what a fooled model can reach, and treat every detector as a
cost-raiser rather than a boundary. cite:beurerkellner2025patterns reaches the same
conclusion from the design-pattern side, and cite:debenedetti2024agentdojo is where you go
to find out what any of it costs in utility on tasks you care about.""")
```

```
                                       layer   blocks   utility cost   detection-based?
---------------------------------------------------------------------------------------
                  input injection classifier      62%             6%                yes
              instruction-hierarchy training      44%             3%                yes
                    tool allow-list per task      71%            18%                 no
     no privileged sink after untrusted read      93%            34%                 no
          human approval on privileged sinks      88%            51%                 no

all four detection layers together miss 12.33%
all three capability layers together miss 0.244%
```

```
  attempts    detection stack    capability stack        both    vs 1 attempt
-----------------------------------------------------------------------------
         1             12.33%             0.2436%     0.0300%            1.0x
         3             32.63%             0.2436%     0.0795%            2.6x
        30             98.07%             0.2436%     0.2389%            8.0x
       100            100.00%             0.2436%     0.2436%            8.1x
      1000            100.00%             0.2436%     0.2436%            8.1x
```

**Detection bounds a probability and capability bounds a set**
({{eq:detection-layers-fail-against-an-adaptive-attacker}}) — the middle column does not move
between one attempt and a thousand.

```
                             stack   utility kept   success at 1 try   success at 100   success at 1000
-------------------------------------------------------------------------------------------------------
                           nothing           100%           100.000%         100.000%          100.000%
                    detection only            87%            12.334%         100.000%          100.000%
                   capability only            27%             0.244%           0.244%            0.244%
            detection + capability            23%             0.030%           0.244%            0.244%
cheapest detection + tool allow-list            77%            11.020%          29.000%           29.000%
```

The last row is the practical middle: **77% utility, 29% at a hundred attempts** — and that
29% is exactly the allow-list's own miss rate. The classifier contributes nothing to the
asymptote.

```
              blast-radius control    radius   residual, detection only   residual, det + cap
---------------------------------------------------------------------------------------------
                    no restriction       100                    100.000                0.2436
                   read-only tools        34                     34.000                0.0828
    scoped credentials, one tenant        12                     12.000                0.0292
     proposal only, human executes         3                      3.000                0.0073
```

Under detection-only defences the probability term is one, so **the residual is the blast
radius** ({{eq:only-capability-limits-bound-the-damage}}).

```
                            investment   blocks a fixed attack   blocks at 100 attempts   weeks   per week
----------------------------------------------------------------------------------------------------------
         tune the injection classifier                     62%                     0.0%     4.0      0.000
        scope tool allow-lists by task                     71%                    71.0%     2.5      0.284
           split read and write agents                     93%                    93.0%     8.0      0.116
     human approval on the top 3 sinks                     88%                    88.0%     1.5      0.587
```

## 10. Production Considerations

Enumerate your sources and label each with a trust level. It is an afternoon and most teams
have never written the list down.

Count your (untrusted source × privileged sink) pairs and put the number in the design
document. It is the quantity that grows when either team ships.

Treat every detector as a cost-raiser, never as a boundary. Budget for it accordingly and do
not let a design depend on it.

Put the enforcement outside the model. What the agent may call is enforceable; what the agent
may believe is not.

Publish the utility cost of every security control alongside its block rate. A design
presented without the cost column will be reversed the first time it blocks something real.

Assume the attacker's attempt budget is unbounded and off your platform.
{{cite:zou2023universal}}'s transfer result makes your rate limit a bound on the wrong
quantity.

Score controls at their asymptote, not at one attempt. A red team that submits findings and
stops has measured the wrong column.

## 11. Common Mistakes

**Treating prompt injection as SQL injection with a harder parser.** There is no parser and
no plan to build one inside the model.

**Adding delimiters and considering it handled.** It is a string the attacker can read.

**Adding a tool without counting paths.** Surface is a product; one tool is $|S|$ paths.

**Stacking detectors and multiplying the miss rates.** That arithmetic assumes the attack is
fixed.

**Measuring block rate against a fixed attack set.** That is $\mu_d$, and deployment faces
$1-(1-\mu_d)^k$.

**Presenting a security design without its utility cost.** The cost is real and will be
discovered by whoever it blocks.

## 12. Failure Modes

**Indirect injection through a retrieved document.**
{{cite:greshake2023indirect}}'s scenario: the attacker never touches your system, they write
a page that gets indexed.

**Surface growth nobody reviewed.** A retrieval team added a source, an integrations team
added four sinks, and the product of their decisions is unowned.

**Detector defeated off-platform.** The winning suffix was found against an open model and
arrived as the first request your system saw.

**Security control reversed after a false positive.** The classifier blocked a legitimate
document, the incident was expensive, and the control is now advisory.

**Approval fatigue.** Human approval was added to every privileged sink, the volume was
unworkable, and approvals are now rubber-stamped — which is
{{eq:only-capability-limits-bound-the-damage}} with the capability quietly restored.

**Taint lost at a boundary.** The trace records that content was untrusted and the field is
dropped when the request crosses a service, so the sink sees a clean request.

## 13. Alternatives

**Full separation: untrusted content never enters a privileged context.** The strongest
available guarantee and it costs 62% of what the agent could do.

**Dual-model architectures.** A privileged planner that never sees untrusted content and an
unprivileged worker that does. {{cite:beurerkellner2025patterns}}'s family, and the interface
between them becomes the whole security problem.

**Taint tracking with sink policies.** Mark untrusted provenance and block privileged sinks on
tainted paths. Structural, mid-cost, and it requires provenance to survive every service
boundary.

**Capability tokens per task.** Issue a credential scoped to exactly the task, from
{{ch:sd-apis-auth}}'s delegation result. Bounds damage without inspecting content.

**Detection with aggressive rate limiting.** Raise the cost per attempt and cap attempts.
Cheap, keeps utility, and {{cite:zou2023universal}} shows the attempts can happen elsewhere.

## 14. Evaluation

Count your source-sink pairs and re-count them every quarter. The trend is the finding.

Measure your detectors' block rate *and* report the implied success rate at 10, 100 and 1000
attempts. The second is what deployment faces.

Run {{cite:debenedetti2024agentdojo}} or an equivalent that reports utility and attack success
on the same tasks. A block rate without a utility number is half a measurement.

Audit whether taint survives your service boundaries by injecting a marked payload and
checking whether the sink can still see the mark.

Test whether an approval flow is actually reading. Submit a benign but unusual request and
measure the approval rate and the time taken.

## 15. Advanced Concepts

The independence assumed between detection layers is the weakest part of the second listing
and it fails badly. Layers trained on similar data with similar objectives miss the same
attacks, so $\prod_j (1-p_j)$ substantially understates $\mu_d$. The correction makes the
detection stack worse than modelled, which strengthens rather than weakens the chapter's
conclusion — but it also means the single-attempt numbers in security reviews are optimistic
by an unmeasured factor, and measuring the correlation requires an attack set that was not
used to build any of the layers.

The capability model assumes a control's coverage is a property of configuration rather than
of the request, and there is one case where that fails: controls that depend on classifying
the *request* as privileged. "Require approval for high-risk actions" is a capability control
only if "high-risk" is a static property of the tool. If it is inferred from the request —
approval required for refunds over a threshold, say, where the threshold is applied to a
model-extracted amount — then the classification is back in the searchable path and the
control has quietly become a detector. **A capability control that consults the model about
whether it applies is a detector**, and this is the most common way a good architecture
degrades in implementation.

There is an interaction with {{ch:ev-agents}} worth drawing out. That chapter's invariants —
no duplicate write, no read outside scope, every claim traceable to a tool result — are
capability controls in evaluation clothing, and they are checkable after the fact rather than
before. That makes them useless as a boundary and valuable as a *detector of boundary
failure*: they cannot stop the action, and they will tell you the architecture is not holding.
Building them is cheap and the information is not available anywhere else.

Finally, a note on where this argument does not apply. Everything here concerns systems where
untrusted content and privileged capability coexist. A model that only reads and only answers
has no sinks, so its attack surface is zero by this accounting, and the threat model is
entirely about what it *says* rather than what it does — which is
{{ch:sec-jailbreaks}}'s subject and a different problem with different economics. **The
product decision that most reduces AI security risk is whether the system acts at all**, and
it is made long before any security review.

## 16. Connection to Previous Chapters

{{eq:fewer-tools-beats-better-credentials}} from {{ch:sd-apis-auth}} is
{{eq:attack-surface-is-sources-times-sinks}} measured on a different axis: there the cost of
more tools was confusion, here it is reachability, and both are products.

{{eq:delegation-moves-the-check}} from the same chapter is the structural fix this chapter
argues for: enforcement outside the model, at the point where a capability is exercised.

{{eq:semantic-failure-has-no-instrument}} from {{ch:sd-architecture}} is why detection is
structurally hard: an injected instruction produces a successful, well-formed action, which
is the exact class no conventional check sees.

{{eq:three-properties-break-the-stack}} from the same chapter gets a fourth member here — the
trust boundary is inside the prompt, and unlike the other three it has no partial mitigation
available inside the model.

## 17. Exercises

1. Enumerate your system's content sources and assign trust levels. What share of a typical
   context is untrusted?

2. Count your (untrusted source × privileged sink) pairs. How has the number changed over the
   last four quarters?

3. Take your detectors' measured block rates and compute the implied success rate at 100
   attempts. Does any design document contain that number?

4. Classify each of your security controls as detection-based or capability-based, and
   compute the asymptotic success rate the capability ones set.

5. Find a control in your system that looks like a capability limit but consults the model
   about whether it applies. What does {{sec:15-advanced-concepts}} say it actually is?

## 18. Interview Questions

1. Why is prompt injection not solvable the way SQL injection was?

2. We added delimiters and an "ignore instructions" line. How much did that buy?

3. Our injection classifier blocks 94% of known attacks. What do you ask next?

4. What is the effective privilege of a prompt containing a system message and a retrieved
   document?

5. We are adding a fifth tool to the agent. What is the security cost?

6. How would you bound the damage from an injection you cannot prevent?

## 19. Research Questions

1. How correlated are the misses of different injection detectors, and how much does that
   inflate a stacked stack's true miss rate?

2. What is the empirical distribution of attempt budgets in real attacks against
   LLM-integrated products?

3. How far does {{cite:zou2023universal}}'s transferability extend to current models, and does
   defence-side diversity reduce it?

4. Can provenance be maintained across heterogeneous service boundaries cheaply enough for
   taint tracking to be a default?

## 20. Chapter Summary

AI security differs from conventional security in one structural respect, and everything else
follows from it.

**Instructions and data share a channel.** In a realistic agent context, **79% of tokens are
untrusted** and none is distinguishable from the system prompt except by position, which is a
learned prior rather than a boundary ({{eq:instructions-and-data-share-a-channel}}). The
composite prompt's privilege is the maximum over its sources and its intent can come from the
minimum — so unlike SQL, shell or HTML, **an attacker who controls data controls intent**.

Attack surface is therefore a product: **4 sources × 8 sinks = 32 paths**, exposure **232**,
and doubling each side once gives **128** ({{eq:attack-surface-is-sources-times-sinks}}).
Adding one tool adds $|S|$ paths, and the two factors are usually owned by different teams.

Defences split into two kinds with different asymptotics. Four detection layers miss
**12.33%** of a fixed attack and are defeated **32.6%** of the time in 3 attempts and
**100%** in 100 ({{eq:detection-layers-fail-against-an-adaptive-attacker}}), because a blocked
attempt is information. Capability limits sit at **0.2436%** at one attempt and at a thousand
({{eq:only-capability-limits-bound-the-damage}}) — **detection bounds a probability,
capability bounds a set.**

So residual risk is essentially blast radius, and the moves that matter are product
decisions: **100 unrestricted, 34 read-only, 12 scoped, 3 proposal-only.** The practical
middle — a classifier plus per-task allow-lists — keeps **77%** utility at **29%** success,
and that 29% is the allow-list's number, not the classifier's.

The uncomfortable synthesis is that the strongest available defences are subtractions. Every
row that provides a guarantee provides it by removing something the agent could do, and the
cost is visible at design time to a named feature, while the detection alternative's cost
arrives later and anonymously. That asymmetry, not any technical difficulty, is why so many
deployed systems are defended by a classifier and a paragraph of delimiters.

Carry forward: **an attacker who controls data controls intent**, and **spend on what a fooled
model can reach**.

## 21. Further Reading

- {{cite:greshake2023indirect}} — indirect prompt injection against real LLM-integrated
  applications, the paper that made the retrieved-document row trust level zero.
- {{cite:perez2022ignore}} — goal hijacking and prompt leaking named and demonstrated, which
  is the distinction "prompt injection" alone collapses.
- {{cite:zou2023universal}} — transferable adversarial suffixes, and why your rate limit
  bounds the wrong quantity.
- {{cite:debenedetti2024agentdojo}} — 97 tasks and 629 security test cases measuring utility
  and attack success together.
- {{cite:beurerkellner2025patterns}} — architecture-first defences, explicit that the
  guarantee costs utility.
