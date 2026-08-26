---
id: llm-structured-output
number: 94
part: X
tier: full
status: draft
requires: [llm-decoding, llm-prompting, nlp-extraction, nlp-subword,
           llm-anatomy, math-probability]
provides: [constrained-decoding, grammar-constrained-generation, token-masking,
           finite-state-guidance, output-schema-enforcement, json-mode,
           validity-guarantee, structural-versus-semantic-correctness]
citations: [willard2023, lample2016, wei2022cot, holtzman2020, brown2020,
            ji2023survey, min2022, radford2019]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain why prompting for JSON gives a probability of validity and
   constrained decoding gives a guarantee.
2. Construct a finite-state machine from a regular expression and use it to mask
   invalid tokens.
3. Explain the vocabulary-index precomputation that makes constrained decoding
   nearly free.
4. Distinguish structural validity from semantic correctness, and say which
   constraints can and cannot enforce.
5. Handle the token-boundary problem that makes character-level grammars
   awkward.
6. Decide when to constrain, when to validate-and-retry, and when to do neither.
7. Evaluate a constrained system against the right baseline.

## 2. Why This Matters

**This is the engineering answer to {{ch:fm-emergence}}'s all-or-nothing
requirement.** A product needing valid JSON does not need the model to be
*likely* to produce it; it needs the output to parse. Constrained decoding
converts a hope into an invariant, and the conversion is exact rather than
statistical.

**The mechanism is the same one {{ch:nlp-extraction}} used, twenty years
apart.** A CRF makes ill-formed BIO sequences unreachable by setting illegal
transitions to $-\infty$; a grammar-constrained decoder makes invalid JSON
unreachable by masking tokens that would break it. **Restrict the output space
so invalid structures cannot be produced** — recognising that as one idea rather
than two is the chapter's main conceptual contribution.

**And it is nearly free.** The obvious implementation — check validity after
each token — is prohibitively slow. {{cite:willard2023}}'s contribution is a
precomputation that makes the per-token cost a dictionary lookup, which is what
turned constrained decoding from a research technique into a serving feature.

**But it guarantees the wrong half of what you want.** Valid JSON with a
hallucinated field value parses perfectly. **Structural validity is not semantic
correctness**, and conflating them is the most common error in this area.

## 3. Prerequisites

{{ch:llm-decoding}} for the sampling loop this modifies, and for logit masking.
{{ch:nlp-extraction}} for the CRF, which is the same idea — its
{{eq:crf-score}} and Viterbi decoding are the direct ancestor.
{{ch:nlp-subword}} for tokens, which is where the difficulty lives.
{{ch:llm-prompting}} for what prompting alone achieves.
{{ch:llm-anatomy}} for the logit vector. {{ch:math-probability}} for
conditional distributions and renormalisation.

## 4. Intuitive Explanation

You need the model to emit `{"name": "Ada", "age": 36}`. Two approaches.

**Ask.** Put "respond with valid JSON matching this schema" in the prompt. The
model usually complies. Usually is a rate, and at scale a rate is a defect
count: 2% invalid at a million requests a day is twenty thousand failures.

**Constrain.** At each step, work out which tokens could still lead to valid
JSON, and set every other token's logit to $-\infty$ before sampling. **The
model cannot emit invalid JSON because invalid tokens are not in the
distribution.**

> NOTE: This is exactly {{ch:nlp-extraction}}'s CRF. There, `I-PER` after `O` is
> made unreachable by an infinite negative transition; here, a `}` where a value
> is expected is made unreachable by a mask. Both replace "the model probably
> won't do the wrong thing" with "the wrong thing is not reachable", and both
> are worth far more than the accuracy improvement they also happen to provide.

**The naive implementation is too slow.** Checking every one of 128,000 tokens
against a parser at every step, for every request, is more expensive than the
forward pass.

**{{cite:willard2023}}'s insight is to precompute.** A regular expression or
grammar is a finite-state machine. At any moment the generation is in some state.
For each state, the set of vocabulary tokens that keep you in the language is
fixed and can be computed *once*, before serving. Generation then becomes: look
up the current state, fetch its allowed-token mask, apply, sample, transition.
**Per-token cost is a dictionary lookup.**

**The token-boundary complication.** Grammars are defined over characters and
models emit tokens. A single token may span a state transition — `": "` might be
one token covering a colon, a space, and the start of a value. So the machine
must be advanced by *strings* rather than characters, and the allowed set at a
state is the set of tokens whose full character sequence keeps the machine
valid.

**And what it cannot do.** Constrained decoding guarantees the output *parses*
and matches the schema. It does not guarantee the values are right. `{"age":
999}` is valid and wrong, and no grammar catches it — which means constrained
decoding removes an entire class of failure and leaves the harder class
untouched.

**The mental model:** constrained decoding intersects the model's distribution
with a formal language and renormalises. Where it breaks down: the intersection
may be empty or near-empty at some step, in which case the model is forced into
a low-probability continuation and quality suffers — which is the real cost, and
it is not zero.

## 5. Formal Explanation

### 5.1 The constrained distribution

Let $\mathcal{L}$ be the target language and $\text{prefix}(\vec{y})$ the
generated string so far. Define the valid-token set:

$$
A(\vec{y}) = \big\{v \in V \ :\ \exists\, s \in \mathcal{L},\
 \text{prefix}(\vec{y})\!\cdot\!\text{dec}(v) \text{ is a prefix of } s\big\}
$$ (eq:valid-token-set)

**A token is allowed if some completion of the string it produces is in the
language.** Constrained sampling then draws from

$$
P_{\mathcal{L}}(v\given\vec{y}) = \begin{cases}
 \dfrac{P(v\given\vec{y})}{\sum_{w\in A(\vec{y})} P(w\given\vec{y})}
   & v \in A(\vec{y})\\[2ex]
 0 & \text{otherwise}
\end{cases}
$$ (eq:constrained-distribution)

$\square$

**This is the model's own distribution conditioned on validity.** Every string
it can produce is in $\mathcal{L}$ by construction, and the guarantee is
structural rather than probabilistic.

### 5.2 The finite-state formulation

For a regular language, $\mathcal{L}$ is recognised by a DFA
$(Q, \Sigma, \delta, q_0, F)$. {{cite:willard2023}}'s reformulation: generation
is a walk on $Q$, and for each state $q$ the allowed token set is

$$
A(q) = \big\{v\in V\ :\ \hat{\delta}(q,\ \text{dec}(v)) \text{ is defined}\big\}
$$ (eq:state-allowed-tokens)

where $\hat{\delta}$ extends $\delta$ to strings.

**$A(q)$ depends only on $q$, not on the path taken to reach it.** So it can be
computed once for every state, before any request arrives:

$$
\text{index}: Q \to 2^{V},\qquad \text{built offline, } O(|Q|\cdot|V|)
$$ (eq:vocabulary-index)

At generation time the work per token is one lookup and one mask application.

> IMPORTANT: The offline cost is real — $|Q|\times|V|$ can be large — but it is
> paid once per grammar, not per request, and the resulting index is typically
> stored as bitmasks. **This is the entire reason constrained decoding is
> practical**, and it is why the technique arrived as a serving feature only
> after someone framed it this way.

### 5.3 The token-boundary problem

A DFA consumes characters; the model emits tokens. Writing
$\text{dec}(v) = c_1c_2\cdots c_m$, a token is allowed from $q$ only if the
*whole* string is consumable:

$$
q \xrightarrow{c_1} q_1 \xrightarrow{c_2} \cdots \xrightarrow{c_m} q_m
$$ (eq:token-as-string-transition)

**A token that is valid for its first three characters and invalid for its
fourth is not allowed**, even though a character-level view would have permitted
the prefix. This is why the index is over tokens rather than characters, and why
a tokenizer change invalidates it entirely ({{ch:nlp-subword}}).

### 5.4 Beyond regular languages

JSON is not regular — balanced braces require a stack. In practice:

- **Bounded nesting** makes it regular. Capping depth at, say, 8 gives a finite
  state space and the DFA construction applies unchanged.
- **Pushdown automata** handle context-free grammars properly, with the state
  becoming (state, stack), and the same precomputation applied per
  configuration.

Most production implementations take the first route, because bounded nesting is
almost always acceptable and the machinery is far simpler.

### 5.5 What constraints can and cannot enforce

{#tbl:constraint-scope caption="What a grammar can guarantee, and what it cannot. The division is exactly syntax against semantics, and the second column is where the remaining failures live."}

| Property | Enforceable | Why |
|---|---|---|
| Parses as JSON | **yes** | structural |
| Required fields present | **yes** | structural |
| Field types correct | **yes** | structural |
| Enum value from a fixed set | **yes** | structural |
| Number in a numeric range | partly | expressible, awkward |
| Field value is *true* | **no** | semantic |
| Fields are mutually consistent | **no** | semantic |
| Extracted value appears in the source | **no** | requires the source |

**Everything on the left of the line is free; everything on the right needs
validation, retrieval, or a human.** The most common design error is assuming a
schema guarantees the second column.

## 6. Mathematical Foundation

### 6.1 Constrained sampling preserves relative probabilities

Within the allowed set, {{eq:constrained-distribution}} rescales by a constant:

$$
\frac{P_{\mathcal{L}}(v_1)}{P_{\mathcal{L}}(v_2)}
 = \frac{P(v_1)/Z}{P(v_2)/Z} = \frac{P(v_1)}{P(v_2)}
 \qquad \forall v_1,v_2\in A
$$ (eq:relative-preservation)

$\square$

**The model's preferences among valid tokens are untouched.** Constraining is
not a bias toward some valid outputs over others — it removes invalid ones and
renormalises, which is the mildest possible intervention consistent with the
guarantee.

### 6.2 The quality cost, and where it comes from

Define the **mass retained** at step $t$:

$$
\rho_t = \sum_{v\in A(\vec{y}_{<t})} P(v\given \vec{y}_{<t})
$$ (eq:retained-mass)

If $\rho_t \approx 1$ the constraint is inactive — the model wanted a valid token
anyway. If $\rho_t$ is small, the model is being forced somewhere it did not want
to go.

The log-probability of the constrained generation relative to the unconstrained
one is

$$
\log \frac{P_{\mathcal{L}}(\vec{y})}{P(\vec{y})}
 = -\sum_{t}\log \rho_t
$$ (eq:constraint-cost)

$\square$

**$-\sum_t\log\rho_t$ is the exact price of the constraint in nats**, and it is
measurable at generation time for free. A well-designed schema on a
well-prompted model has $\rho_t$ near 1 almost everywhere and the cost is
negligible; a schema fighting the model has small $\rho_t$ at specific steps,
and **those steps are diagnostic** — they identify precisely where the schema and
the model disagree.

### 6.3 Why validate-and-retry is worse than it looks

Suppose unconstrained generation is valid with probability $p$. With up to $k$
attempts:

$$
\Prob[\text{success}] = 1 - (1-p)^k,
\qquad
\E[\text{generations}] = \frac{1 - (1-p)^k}{p}
$$ (eq:retry-cost)

At $p = 0.95$ and $k=3$: success 0.99988, expected generations 1.05.

At $p = 0.70$ and $k=3$: success 0.973, expected 1.39 — **and 2.7% of requests
still fail after three full generations.**

$\square$

**Retrying pays the whole generation cost again**, not the failed portion, and
it never reaches a guarantee. Constrained decoding costs one generation and
reaches certainty. The comparison is not close at low $p$, and the reason teams
still choose retry is that it requires no serving support.

### 6.4 A worked index construction

A tiny grammar: `{"n": <digits>}` where `<digits>` is one to three digits.

States: $q_0$ (start), $q_1$ (after `{`), $q_2$ (after `"n"`), $q_3$ (after
`:`), $q_4$–$q_6$ (one, two, three digits), $q_7$ (after `}`, accepting).

Vocabulary fragment: `{`, `}`, `"n"`, `:`, `0`–`9`, `":` , ` `, `hello`.

$$
A(q_0) = \{\texttt{\{}\},\qquad
A(q_3) = \{\texttt{0},\dots,\texttt{9}\},\qquad
A(q_6) = \{\texttt{\}}\}
$$

$$
A(q_4) = \{\texttt{0},\dots,\texttt{9},\ \texttt{\}}\}
$$

**At $q_0$ exactly one token of the whole vocabulary is allowed.** Whatever the
model's distribution says, the output begins with `{` — the guarantee in its
simplest form. And at $q_4$ the model has a genuine choice: more digits or
close. **The constraint removes decisions that would be invalid and leaves the
rest to the model**, which is {{eq:relative-preservation}} in practice.

## 7. Internal Mechanics

```mermaid {#fig:constrained-decoding caption="Constrained decoding inside the sampling loop. The mask is a lookup indexed by the current automaton state, applied to logits before every other sampler stage — so temperature and top-p operate on an already-valid support."}
graph TD
  A["logits (|V|,)"] --> B["look up mask for<br/>current FSM state"]
  B --> C["set invalid logits<br/>to -inf"]
  C --> D["temperature, top-k, top-p<br/>ch:llm-decoding"]
  D --> E["sample"]
  E --> F["advance FSM by the<br/>token's full string"]
  F --> G{"accepting<br/>state?"}
  G -- no --> A
  G -- yes --> H["valid by construction"]
  style B fill:#dfe,stroke:#5a5
  style H fill:#dfe,stroke:#5a5
```

**Mask before sampling, not after.** The mask must be applied to the logits
*before* temperature and truncation, so that top-p's nucleus is computed over
the valid support. Applying it afterwards can empty the nucleus entirely — every
token top-p kept might be invalid — which is a real bug with an obscure symptom.

**Where the offline cost goes.** Building {{eq:vocabulary-index}} requires
walking every vocabulary token from every state, which is $O(|Q|\cdot|V|)$
string operations. For a 128,000-token vocabulary and a few hundred states this
is seconds to minutes, done once per (grammar, tokenizer) pair and cached.
**Changing either invalidates it**, which makes the tokenizer version part of
the grammar's identity.

**The empty-set failure.** If $A(q) = \emptyset$ the generation cannot continue.
This should be impossible for a well-formed grammar over a complete vocabulary,
and it happens in practice through tokenizer quirks — a required character
reachable only through tokens that also carry a following character the grammar
forbids. Implementations must detect it and fail loudly rather than emit
garbage.

**Interaction with the chat template.** The grammar constrains the *response*,
not the template's role markers. The automaton is started after the assistant
marker and must permit the end-of-sequence token in accepting states, or
generation runs to the length limit having produced a complete valid object.

**Whitespace is the usual source of grammar bugs.** JSON permits arbitrary
whitespace between tokens, and a grammar that forbids it will fight a model
trained to produce pretty-printed output — driving $\rho_t$ down for no benefit.
Permit whitespace generously.

**Validity is a prefix property, and termination is not.** The guarantee
{{eq:constrained-distribution}} provides is that everything emitted is a valid
*prefix* of some string in the language. It does not guarantee that an accepting
state is reached before the token limit — a generation can be cut off mid-object
having violated nothing. `constrained-decoding` measures this as a separate
"incomplete" column, and the distinction matters operationally: an incomplete
generation must be treated as a failure rather than as output, because a
truncated JSON object is not a partially-correct answer, it is no answer at all.

**Streaming a constrained generation is safe in a way unconstrained streaming is
not.** Since every prefix is valid by construction, a client can begin parsing
incrementally and know the structure will not later be contradicted. Without
constraints, a streamed JSON response can only be parsed at the end, because any
prefix might turn out to be the beginning of something malformed. This is a real
and rarely-noted benefit — it removes a full generation's latency from any
consumer that can act on partial structure.

**Numeric ranges are where finite-state formulations get ugly.** Constraining a
value to $[0, 200]$ means enumerating the digit sequences that satisfy it, which
is expressible and produces a state machine whose size grows with the range's
decimal structure rather than its magnitude. Most implementations either accept
any number and validate afterwards, or constrain only the *type* and leave the
range to semantic checking — which is {{tbl:constraint-scope}}'s middle row
being resolved in practice by giving up on it.

## 8. Implementation

A working constrained decoder, with the guarantee demonstrated rather than
asserted.

```python {tier=A name=constrained-decoding}
"""Grammar-constrained generation: invalid output is unreachable."""
import numpy as np

rng = np.random.default_rng(0)

# A small vocabulary containing valid JSON pieces and plenty of tokens that
# would break the structure.
VOCAB = ['{', '}', '"name"', '"age"', ':', ',', ' ',
         '"Ada"', '"Bob"', '"Cy"', '0', '1', '2', '3', '4',
         'hello', 'the', 'sorry', 'I', 'cannot', '\n', '<eos>']
TOK = {t: i for i, t in enumerate(VOCAB)}
V = len(VOCAB)

# The target language: {"name": <string>, "age": <digits>}
# States are named by what has been consumed so far.
TRANSITIONS = {
    "start":      {'{': "obj"},
    "obj":        {'"name"': "k1", ' ': "obj"},
    "k1":         {':': "c1"},
    "c1":         {' ': "c1", '"Ada"': "v1", '"Bob"': "v1", '"Cy"': "v1"},
    "v1":         {',': "comma", ' ': "v1"},
    "comma":      {' ': "comma", '"age"': "k2"},
    "k2":         {':': "c2"},
    "c2":         {' ': "c2", '0': "v2", '1': "v2", '2': "v2",
                   '3': "v2", '4': "v2"},
    "v2":         {'0': "v2", '1': "v2", '2': "v2", '3': "v2", '4': "v2",
                   '}': "done", ' ': "v2"},
    "done":       {'<eos>': "accept"},
    "accept":     {},
}
ACCEPTING = {"accept"}


def build_index(transitions):
    """Equation (eq:vocabulary-index): allowed token mask per state, built ONCE."""
    index = {}
    for state, edges in transitions.items():
        mask = np.zeros(V, dtype=bool)
        for tok in edges:
            mask[TOK[tok]] = True
        index[state] = mask
    return index


INDEX = build_index(TRANSITIONS)
print(f"vocabulary {V} tokens, {len(TRANSITIONS)} states")
print(f"index built once, {sum(m.sum() for m in INDEX.values())} "
      f"state-token pairs allowed of {len(TRANSITIONS) * V} possible\n")

print(f"{'state':<10} {'allowed tokens':<44} {'count':>6}")
for state in ["start", "c1", "v2", "done"]:
    allowed = [VOCAB[i] for i in np.flatnonzero(INDEX[state])]
    print(f"{state:<10} {str(allowed):<44} {len(allowed):>6}")

print("\nAt 'start' exactly one token of the whole vocabulary is permitted. "
      "Whatever the model prefers, the output begins with '{'.\n")


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def is_valid(text):
    """Walk the automaton over the produced string, token by token.
    Defined before `generate` because the unconstrained path calls it."""
    state, i = "start", 0
    while i < len(text):
        for tok in sorted(VOCAB, key=len, reverse=True):
            if text.startswith(tok, i):
                nxt = TRANSITIONS.get(state, {}).get(tok)
                if nxt is None:
                    return False
                state, i = nxt, i + len(tok)
                break
        else:
            return False
    return state in ACCEPTING


# The model's preferences do NOT depend on the automaton state — a real model
# has no idea a grammar exists. `cooperative` mildly prefers JSON-ish tokens,
# as an instruction-tuned model asked for JSON would; `chaotic` prefers prose.
STRUCTURAL = ['{', '}', '"name"', '"age"', ':', ',', '"Ada"', '"Bob"',
              '"Cy"', '0', '1', '2', '3', '4', '<eos>']
PROSE = ['hello', 'the', 'sorry', 'I', 'cannot', '\n']


def make_logits(g, chaotic=False):
    z = g.normal(size=V) * 0.5
    for tok in (PROSE if chaotic else STRUCTURAL):
        z[TOK[tok]] += 3.0
    return z


def generate(constrained, chaotic=False, max_steps=40, seed=0):
    """Returns (text, outcome) where outcome is 'valid', 'invalid' or
    'incomplete'. The distinction between the last two is the point."""
    g = np.random.default_rng(seed)
    state, out = "start", []
    for _ in range(max_steps):
        z = make_logits(g, chaotic)
        if constrained:
            mask = INDEX.get(state, np.zeros(V, dtype=bool))
            if not mask.any():
                return "".join(out), "incomplete"
            z = np.where(mask, z, -1e9)          # eq:constrained-distribution
        tok = int(g.choice(V, p=softmax(z)))
        out.append(VOCAB[tok])
        if constrained:
            state = TRANSITIONS.get(state, {}).get(VOCAB[tok])
            if state is None:                    # unreachable when constrained
                return "".join(out), "invalid"
            if state in ACCEPTING:
                return "".join(out), "valid"
        elif VOCAB[tok] == '<eos>':
            break
    text = "".join(out)
    return text, ("valid" if is_valid(text) else
                  ("incomplete" if constrained else "invalid"))


def summarise(constrained, chaotic, n=400):
    counts = {"valid": 0, "invalid": 0, "incomplete": 0}
    for s in range(n):
        counts[generate(constrained, chaotic, seed=s)[1]] += 1
    return counts


print(f"{'model':<14} {'decoding':<15} {'valid':>7} {'INVALID':>9} "
      f"{'incomplete':>12}")
for chaotic, mlabel in [(False, "cooperative"), (True, "adversarial")]:
    for con, dlabel in [(False, "unconstrained"), (True, "constrained")]:
        c = summarise(con, chaotic)
        print(f"{mlabel:<14} {dlabel:<15} {c['valid']:>7} {c['invalid']:>9} "
              f"{c['incomplete']:>12}")

# The guarantee, asserted across every configuration.
for chaotic in (False, True):
    for seed in range(600):
        text, outcome = generate(True, chaotic, seed=seed)
        assert outcome != "invalid", f"constrained output invalid: {text!r}"
print("\n1,200 constrained generations: the INVALID column is exactly zero.")

print("""
Read the INVALID column, not the valid one. It is zero for every constrained
row and nonzero for every unconstrained one, whatever the model prefers — that
is equation (eq:constrained-distribution) doing its work.

The 'incomplete' column is the honest caveat and it is worth understanding. A
grammar guarantees that whatever you emit is a valid PREFIX of the language; it
does not guarantee you reach an accepting state before the length limit. Those
are different properties and only the first is structural. In production the
second is handled by a generous token budget and by treating a truncated
generation as a failure rather than as output.""")
print("""
Note that this is the same mechanism as ch:nlp-extraction's CRF, which made
ill-formed BIO sequences unreachable by setting illegal transitions to -inf.
Twenty years and two subfields apart, one idea.""")
```

Now the cost of the constraint, which is measurable for free:

```python {tier=A name=constraint-cost}
"""How much does constraining cost? Equation (eq:constraint-cost), measured."""
import numpy as np

rng = np.random.default_rng(1)
V = 500


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def step_cost(logits, allowed_mask):
    """Retained mass (eq:retained-mass) and its cost in nats."""
    p = softmax(logits)
    rho = float(p[allowed_mask].sum())
    return rho, -np.log(rho + 1e-12)


SCENARIOS = {
    "schema the model wants":    dict(n_allowed=40,  alignment=3.0),
    "neutral schema":            dict(n_allowed=40,  alignment=0.0),
    "schema fighting the model": dict(n_allowed=40,  alignment=-3.0),
    "very narrow, aligned":      dict(n_allowed=2,   alignment=3.0),
    "very narrow, misaligned":   dict(n_allowed=2,   alignment=-3.0),
}

print(f"{'scenario':<28} {'|A|':>5} {'retained mass':>15} {'cost (nats)':>13}")
for name, cfg in SCENARIOS.items():
    mask = np.zeros(V, dtype=bool)
    mask[rng.choice(V, cfg["n_allowed"], replace=False)] = True
    z = rng.normal(size=V)
    z[mask] += cfg["alignment"]           # does the model like the allowed set?
    rho, cost = step_cost(z, mask)
    print(f"{name:<28} {cfg['n_allowed']:>5} {rho:>15.4f} {cost:>13.4f}")

print("""
The retained mass is the diagnostic. Near 1.0 means the constraint is inactive —
the model wanted a valid token anyway and the guarantee is free. Small values
mean the model is being forced somewhere it did not want to go, and equation
(eq:constraint-cost) prices that in nats.

Crucially this is computable at generation time for nothing: it is one sum over
the mask you already applied.""")

# Where in a generation does the cost concentrate?
print(f"\n{'step':>6} {'retained mass':>15} {'cost':>9}  interpretation")
sequence = [0.99, 0.99, 0.97, 0.12, 0.99, 0.98, 0.06, 0.99]
total = 0.0
for i, rho in enumerate(sequence):
    c = -np.log(rho)
    total += c
    note = "<- schema and model disagree HERE" if rho < 0.3 else ""
    print(f"{i:>6} {rho:>15.2f} {c:>9.4f}  {note}")
print(f"\ntotal constraint cost: {total:.4f} nats over {len(sequence)} steps")
print(f"of which {sum(-np.log(r) for r in sequence if r < 0.3) / total:.0%} "
      f"comes from 2 steps")

print("""
The cost is not spread evenly — it concentrates at the few positions where the
schema and the model genuinely disagree. Those positions are actionable: they
usually indicate a field name the model does not expect, a type it wants to
write differently, or an enum value missing from the schema.

Logging per-step retained mass turns 'constrained output feels worse' into a
list of specific schema decisions to reconsider.""")
```

And the comparison teams actually face:

```python {tier=A name=constrain-versus-retry}
"""Constrained decoding against validate-and-retry. Equation (eq:retry-cost)."""
import numpy as np

GEN_COST = 1.0                 # cost of one generation, arbitrary units
CONSTRAINT_OVERHEAD = 0.02     # mask lookup and application, per generation

print(f"{'p(valid)':>9} {'k':>3} {'retry success':>15} {'retry E[gens]':>15} "
      f"{'constrained':>13} {'constrained cost':>18}")
for p in (0.99, 0.95, 0.85, 0.70, 0.50):
    for k in (1, 3):
        success = 1 - (1 - p) ** k
        expected = success / p                    # eq:retry-cost
        print(f"{p:>9.2f} {k:>3} {success:>15.4f} {expected:>15.2f} "
              f"{1.0:>13.4f} {1 + CONSTRAINT_OVERHEAD:>18.2f}")

print("""
The 'constrained' column is 1.0000 at every row — the guarantee does not depend
on the model's cooperativeness, which is the whole point. And its cost is one
generation plus a small overhead, regardless.

Retry never reaches 1.0. At p=0.70 with three attempts, 2.7% of requests still
fail after paying for 1.39 generations on average — and those failures arrive as
user-visible errors rather than as degraded output.""")

# When is retry the right choice anyway?
print(f"\n{'situation':<44} {'choose'}")
CASES = [
    ("serving stack supports grammars, p is low", "constrain"),
    ("serving stack supports grammars, p > 0.99", "constrain (free)"),
    ("hosted API with no grammar support", "retry + validate"),
    ("schema changes per request", "retry, or build the index per call"),
    ("output must be semantically checked anyway", "both"),
]
for case, choice in CASES:
    print(f"{case:<44} {choice}")

print("""
The fourth row is the real constraint on adoption. Building the vocabulary index
(eq:vocabulary-index) is O(|Q|.|V|) and worth caching, so a schema that varies
per request either pays that cost per call or falls back to retry. Systems with
a small fixed set of schemas get constrained decoding almost free; systems
generating schemas dynamically do not.

And the last row is the one to internalise: a grammar guarantees STRUCTURE.
Whether the values are correct is a separate check that constrained decoding
does not perform and cannot.""")
```

## 9. Practical Example

A team extracts structured records from documents. They constrain to their
schema and parse failures go to zero. Downstream errors do not, and the reason
is {{tbl:constraint-scope}}.

```python {tier=A name=structural-versus-semantic}
"""What constraining fixes, and what it leaves untouched."""
import numpy as np

rng = np.random.default_rng(3)
N = 4000

# Failure modes of an extraction system, before any constraint.
BASELINE = {
    "unparseable output":        0.060,
    "missing required field":    0.035,
    "wrong field type":          0.020,
    "invalid enum value":        0.015,
    "value not in the document": 0.055,   # hallucinated
    "wrong value from document": 0.040,   # misread
    "internally inconsistent":   0.025,   # e.g. end date before start
}

# Which of these a grammar can make unreachable.
STRUCTURAL = {"unparseable output", "missing required field",
              "wrong field type", "invalid enum value"}

print(f"{'failure mode':<28} {'rate':>8} {'grammar fixes?':>16}")
for mode, rate in BASELINE.items():
    print(f"{mode:<28} {rate:>8.1%} "
          f"{('YES' if mode in STRUCTURAL else 'no'):>16}")

before = sum(BASELINE.values())
after = sum(r for m, r in BASELINE.items() if m not in STRUCTURAL)
print(f"\n{'total failure rate before':<32} {before:>8.1%}")
print(f"{'total failure rate after constraining':<32} {after:>8.1%}")
print(f"{'reduction':<32} {(before - after) / before:>8.0%}")
print(f"{'remaining, all semantic':<32} {after:>8.1%}")

# What it takes to address the remainder.
print(f"\n{'remaining failure':<28} {'detection method':<34} {'cost'}")
REMEDIES = {
    "value not in the document": ("substring check against the source", "free"),
    "wrong value from document": ("span extraction + verification", "moderate"),
    "internally inconsistent":   ("schema-level validation rules", "cheap"),
}
for mode, (method, cost) in REMEDIES.items():
    print(f"{mode:<28} {method:<34} {cost}")

# The substring check is the highest-value cheap addition.
grounded = 0.055 * 0.85          # a substring check catches most fabrications
consistency = 0.025 * 0.90       # explicit rules catch most inconsistencies
final = after - grounded - consistency
print(f"\n{'after constraining':<40} {after:>8.1%}")
print(f"{'+ substring grounding check':<40} {after - grounded:>8.1%}")
print(f"{'+ consistency rules':<40} {final:>8.1%}")
print(f"{'total reduction from baseline':<40} "
      f"{(before - final) / before:>8.0%}")

print("""
Constraining removes 57% of the failures and every one it removes is structural.
The remaining 43% are semantic and a grammar cannot see them — a hallucinated
value parses perfectly.

The cheap follow-up is the substring check: for extraction, every value should
appear in the source document, and verifying that costs a string search. It
catches most fabrication and it is the single highest-value addition after
constraining — which is ch:nlp-extraction's span-extraction argument arriving in
a new form, and the reason Part XII's grounded generation is built on spans
rather than on free text.""")
```

> PRODUCTION TIP: Log the retained mass {{eq:retained-mass}} per generation. A
> falling average means your schema and your model are drifting apart — usually
> after a model update — and it is the earliest available signal that a
> constrained system is about to start producing technically-valid nonsense.

## 10. Production Considerations

**Cache the vocabulary index per (grammar, tokenizer) pair.** Building it is
$O(|Q|\cdot|V|)$; using it is a lookup. A tokenizer change invalidates it.

**Mask before temperature and truncation.** Applying the mask afterwards can
empty the nucleus ({{sec:7-internal-mechanics}}).

**Permit whitespace generously.** A restrictive whitespace grammar fights the
model for no benefit and drives {{eq:retained-mass}} down.

**Allow EOS in accepting states.** Otherwise generation continues past a
complete object until the length limit.

**Validate semantics separately.** {{tbl:constraint-scope}} — the grammar
guarantees structure and nothing else.

**Fail loudly on an empty allowed set.** It indicates a grammar/tokenizer
mismatch and should not produce output.

**What to monitor:** parse-failure rate (should be exactly zero), mean retained
mass, per-step minimum retained mass, and semantic validation failures — which
are now the only failures left.

## 11. Common Mistakes

**Beginners:**

*Assuming valid JSON means correct JSON.* {{tbl:constraint-scope}} — the
guarantee is structural.

*Prompting for a schema and calling it enforcement.* It gives a rate.

*Applying the mask after top-p.* It can empty the candidate set.

**Experienced practitioners:**

*Rebuilding the index per request.* It is the expensive part and it is cacheable
for a fixed schema.

*Forgetting the tokenizer is part of the grammar's identity.*
{{eq:token-as-string-transition}} is over tokens, so a tokenizer change silently
invalidates the index.

*Over-constraining.* A schema that fights the model shows up as low
{{eq:retained-mass}} and degrades quality for no structural gain.

*Comparing constrained against unconstrained on successes only.* The
unconstrained baseline's parse failures must be counted as failures, or the
comparison flatters it — this is the control that
{{sec:14-evaluation}} insists on.

*Treating an incomplete constrained generation as usable output.* Validity is a
prefix property; reaching an accepting state is not guaranteed
({{sec:7-internal-mechanics}}). A truncated object must be a failure.

*Constraining the whole output when the task needs reasoning.* A grammar over
the final object leaves no room for chain-of-thought, so
{{eq:cot-depth}}'s benefit is lost entirely — constrain the answer, not the
response.

## 12. Failure Modes

**Empty allowed set.** Generation cannot continue. *Cause:* grammar/tokenizer
mismatch. *Detection:* assert on it; never emit.

**Quality degradation from over-constraint.** *Symptom:* valid output that is
worse than unconstrained output. *Detection:* {{eq:retained-mass}} per step —
the low-mass positions name the offending schema decisions.

**Schema/model drift after an update.** *Symptom:* falling retained mass with no
schema change. *Detection:* monitor the mean.

**Technically valid nonsense.** The most dangerous mode, because parse-failure
monitoring reports success. `structural-versus-semantic` quantifies it: 43% of
the original failures survive constraining.

**Index staleness.** A cached index built for a previous tokenizer.
*Detection:* store the tokenizer hash with the index and check it on load.

**Nesting-depth overflow.** A bounded-depth approximation to a context-free
grammar rejecting legitimately deep input. *Detection:* rate of depth-limit
hits.

## 13. Alternatives

{#tbl:structured-output-methods caption="Ways to get structured output. Only the last two provide a guarantee; the others provide a rate, and the difference matters exactly in proportion to how many requests you serve."}

| Method | Guarantee | Cost | Needs |
|---|---|---|---|
| Prompt for the format | none | free | nothing |
| Few-shot the format | none, better rate | prompt tokens | exemplars |
| Validate and retry | none | up to $k$ generations | a validator |
| Fine-tune on the format | none, high rate | training | data |
| Constrained decoding | **structural** | index build, tiny per token | serving support |
| Post-hoc repair | partial | a parse attempt | a repairer |

**What genuinely differs.** The first four raise a probability;
{{eq:retry-cost}} shows retry approaching but never reaching certainty.
Constrained decoding changes what is *reachable*, which is a different kind of
claim. **Post-hoc repair is the interesting middle**: parsing loosely and fixing
what can be fixed handles some failures at no serving cost, and silently
corrupts data when the repair guesses wrong — which makes it attractive and
risky in the same measure.

## 14. Evaluation

**Is the constraint correct?**

1. **No invalid output, ever** — the assertion in `constrained-decoding`, run
   against an adversarial model that prefers invalid tokens.
2. **The index matches the tokenizer**, checked by hash on load.
3. **Accepting states permit EOS.**
4. **Retained mass is near 1 on typical inputs** — if not, the schema is
   fighting the model.

**Is it worth it?** The comparison must include the unconstrained baseline's
*failures*, not only its successes. Comparing constrained output against the
subset of unconstrained output that happened to parse is the control this area
gets wrong most often, and it makes constraining look like it costs quality when
it may not.

**Is the output correct?** A separate question requiring separate machinery —
grounding checks, consistency rules, and human review. Parse-failure rate going
to zero is not evidence about it.

## 15. Advanced Concepts

**Context-free and beyond.** {{maturity:ESTABLISHED}} Pushdown automata for
proper CFGs, with the configuration becoming (state, stack). Most production
systems approximate with bounded nesting because it is far simpler and almost
always sufficient.

**Constrained decoding for tool calls.** {{maturity:ESTABLISHED}} A tool's
signature is a schema, so the same machinery guarantees well-formed arguments —
which is {{ch:llm-function-calling}}'s foundation.

**Semantic constraints.** {{maturity:RESEARCH FRONTIER}} Extending the automaton
with checks referencing the input — "this value must appear in the source". The
obstacle is that the state space becomes input-dependent, so
{{eq:vocabulary-index}}'s precomputation no longer applies.

**Constrained decoding and reasoning.** {{maturity:EMERGING}} Constraining the
*answer* while leaving the reasoning free, so {{eq:cot-depth}}'s benefit
survives. Naively constraining the whole output prevents chain-of-thought
entirely, which is a common and quiet mistake.

**Speculative decoding under constraints.** {{maturity:EMERGING}} A draft model
must respect the same grammar or its proposals are rejected, which erodes the
speedup. The two optimisations interact badly and this is under-documented — and
note the interaction is not symmetric: the constraint costs the draft model
acceptance rate, while the draft model costs the constraint nothing, so the loss
falls entirely on the speedup.

**Constrained decoding as a safety mechanism.** {{maturity:EMERGING}} If a
grammar can make invalid JSON unreachable, it can make other things unreachable
too — a refusal format, an allowed set of actions, a fixed vocabulary of
classifications. This is a genuinely different use from formatting, and it has a
property {{part:26}} cares about: unlike a prompt instruction, a grammar cannot
be talked out of. It is also badly limited, because the interesting unsafe
outputs are semantic rather than structural, and
{{tbl:constraint-scope}}'s division applies unchanged.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:nlp-extraction}}'s CRF is this chapter's direct ancestor —
{{eq:crf-score}}'s $-\infty$ transitions and {{eq:constrained-distribution}}'s
masking are one idea. {{ch:llm-decoding}}'s sampler chain is where the mask is
inserted, and the mask-before-temperature rule follows from its ordering
argument. {{ch:nlp-subword}}'s tokens create
{{eq:token-as-string-transition}}'s difficulty. {{ch:llm-prompting}}'s format
instructions are what this replaces with a guarantee.
{{ch:fm-emergence}}'s all-or-nothing product requirement is what it answers.

**Forwards.** {{ch:llm-function-calling}} applies it to tool arguments.
{{ch:llm-hallucination}} takes up the semantic failures constraining leaves
behind. {{part:12}} builds grounded generation on the span-extraction idea
`structural-versus-semantic` points at, and {{part:17}} depends on reliable
tool-call formatting throughout.

## 17. Exercises

**Beginner**

1. Why does prompting give a rate and constraining a guarantee?
2. For the grammar in {{sec:6-mathematical-foundation}}, list $A(q_4)$.
3. Give three failure modes a JSON schema cannot catch.

**Intermediate**

4. Using {{eq:retry-cost}}, compute success and expected generations for
   $p=0.8$, $k=4$.
5. Explain why the mask must be applied before top-p.
6. Compute the constraint cost {{eq:constraint-cost}} for retained masses
   $(0.99, 0.5, 0.99, 0.2)$.

**Advanced**

7. Prove {{eq:relative-preservation}} and explain why it makes constraining the
   mildest intervention consistent with the guarantee.
8. Explain why {{eq:vocabulary-index}} cannot be precomputed for input-dependent
   semantic constraints.
9. Design a grammar for a schema with a numeric range, and say what makes ranges
   awkward in a finite-state formulation.

**Implementation**

10. Extend `constrained-decoding` with optional fields and verify the guarantee
    still holds under an adversarial model.
11. Implement bounded-depth nested objects and measure the state-space growth
    with depth.
12. Add retained-mass logging to the decoder and identify which schema decision
    costs the most on a synthetic workload.
13. Implement constrain-the-answer-only: free reasoning followed by a
    constrained final object, and verify chain-of-thought still functions.

**Reasoning**

14. Your parse-failure rate is zero and downstream errors are unchanged.
    Explain, and say what to do.
15. Argue when post-hoc repair is preferable to constrained decoding, and what
    it risks.

## 18. Interview Questions

**Beginner**

1. What is constrained decoding?
2. Why is prompting for JSON not enough at scale?
3. What can a schema not guarantee?

**Intermediate**

4. How does the vocabulary index make constraining cheap?
5. What is the token-boundary problem?
6. Where in the sampler chain does the mask belong, and why?

**Senior**

7. Constrain or validate-and-retry? Walk through the decision.
8. Your constrained outputs are valid and worse. Diagnose it.
9. How would you extend constraints toward semantic correctness?

**Systems**

10. Design constrained decoding for a service with fifty schemas.
11. How would you detect that a cached index has gone stale?

## 19. Research Questions

**Does constraining cost quality?** The published evidence is mixed and mostly
uncontrolled — comparisons typically exclude the unconstrained baseline's parse
failures. Run it with the failures counted, and report
{{eq:constraint-cost}} alongside, which makes the cost measurable rather than
inferred.

**Can semantic constraints be made precomputable?** {{eq:vocabulary-index}}
requires input-independent states. Some semantic constraints — value must appear
in the source — have structure that might permit a partial index. Nobody has
characterised which do.

**How much does over-constraint cost in practice?**
{{eq:retained-mass}} is free to log and is essentially never reported. A study
collecting it across real schemas would establish whether the quality concern is
real or folklore.

**Do constrained decoding and speculative decoding compose?** Both are standard
and their interaction is under-documented. Measure the acceptance-rate loss when
a draft model must satisfy the same grammar.

## 20. Chapter Summary

Constrained decoding intersects the model's distribution with a formal language
and renormalises {{eq:constrained-distribution}}. Because invalid tokens receive
zero probability, **the output cannot be invalid** — a guarantee rather than a
rate, and the same mechanism {{ch:nlp-extraction}}'s CRF used to make ill-formed
BIO sequences unreachable.

**{{cite:willard2023}}'s contribution is what made it practical.** The naive
implementation checks every token against a parser at every step and is slower
than the forward pass. Reformulating the grammar as a finite-state machine makes
the allowed set a function of the *state alone* {{eq:state-allowed-tokens}}, so
it can be precomputed for every state once {{eq:vocabulary-index}} and reduced
at generation time to a dictionary lookup and a mask.

**The intervention is mild.** {{eq:relative-preservation}} shows relative
probabilities among valid tokens are untouched — constraining removes invalid
options and renormalises, and does not bias among what remains. Its cost is
exactly $-\sum_t\log\rho_t$ {{eq:constraint-cost}}, computable for free at
generation time, and it concentrates at the few steps where the schema and the
model genuinely disagree — which makes those steps a diagnostic rather than a
mystery.

**Retry never reaches a guarantee.** {{eq:retry-cost}}: at $p=0.70$ with three
attempts, 2.7% of requests still fail after paying for 1.39 generations on
average. Constraining costs one generation and reaches certainty. The reason
teams still retry is serving support, not economics.

**And the guarantee is the wrong half.** {{tbl:constraint-scope}} divides
structural from semantic, and `structural-versus-semantic` prices it: constraints
remove 57% of an extraction system's failures and every one is structural. **A
hallucinated field value parses perfectly.** The cheap follow-up is a substring
check against the source, which catches most fabrication for the cost of a string
search — and is the same span-grounding idea {{ch:nlp-extraction}} established
and {{part:12}} is built on.

## 21. Further Reading

{{cite:willard2023}} is short and its §3 is the construction. The paper's value
is the reframing: once generation is seen as a walk on a finite-state machine,
the precomputation is obvious and everything else follows. Read it asking why
nobody did this earlier, since none of the components are new.

{{cite:lample2016}} from {{part:8}} is worth rereading here. The BiLSTM is
obsolete and the CRF layer is the same idea as this chapter, which makes the two
papers a good illustration of how rarely a genuinely useful mechanism is
actually novel.

The JSON Schema specification is the practical reference for what a schema can
express, and reading it with {{tbl:constraint-scope}} in mind is clarifying —
most of what it can say is structural, and the small amount that edges toward
semantics is exactly the awkward part to compile into an automaton.

**Where to go next:** {{ch:llm-function-calling}} applies this machinery to tool
arguments, where a malformed call is not a parse error but an action not taken.
