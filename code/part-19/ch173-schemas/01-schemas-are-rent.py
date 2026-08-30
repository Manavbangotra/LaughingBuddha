# -*- coding: utf-8 -*-
# Extracted from: Chapter 173 — Tool Schemas, Discovery, and Context Budgets
# Source: src/.../ch173-schemas.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a tool inventory costs before any tool is called.

Every tool a host offers is described in the model's context: a name, a
description, a JSON Schema for its arguments. That description is paid for on
EVERY request, whether or not the tool is used.

ch:ag-tool-calling found selection robust to inventory size and fragile to
inventory overlap, and measured selection at 100% with 128 distinct tools. That
result is about the DECISION. This listing is about the BILL, which behaves
differently: a schema is tokens, tokens are context, and ch:ag-memory found
context dilutes (eq:schemas-are-rent).

cite:qin2023toolllm worked with 16,000+ APIs. Nobody puts 16,000 schemas in a
context window, so something has to choose, and the something is retrieval.
"""
import numpy as np

rng = np.random.default_rng(4201)

M = 40000
TOK_NAME = 12           # tokens for a name and one-line description
TOK_ARG = 34            # tokens per documented argument
ARGS_MEAN = 4.5
CTX = 16000             # tokens the host is willing to spend on tool schemas
BASE = 0.995


def schema_tokens(n_tools, args_mean=ARGS_MEAN, verbose=1.0):
    """Tokens consumed by n_tools of schema. `verbose` scales description length."""
    return n_tools * (TOK_NAME * verbose + TOK_ARG * args_mean * verbose)


def dilution(tokens):
    """ch:ag-memory's effect, as a per-step multiplier on reasoning quality."""
    return BASE * (1.0 - 2.2e-6 * tokens)


def run(n_tools, m=M, steps=5, args_mean=ARGS_MEAN, verbose=1.0,
        retrieved=None, recall=0.94, p_select_base=0.985):
    """`retrieved` is how many tools are actually put in context; None means all.
    Retrieval may miss the needed tool, at rate 1 - recall."""
    shown = n_tools if retrieved is None else min(retrieved, n_tools)
    toks = schema_tokens(shown, args_mean, verbose)
    # Selection degrades gently with how many candidates are visible, per
    # ch:ag-tool-calling: it is overlap that hurts, and overlap grows with count.
    p_sel = p_select_base ** (1.0 + np.log2(max(shown, 1)) / 12.0)
    # Retrieval can simply fail to surface the right tool.
    present = np.ones(m, dtype=bool) if retrieved is None else \
        (rng.random(m) < recall)
    ok_sel = (rng.random((m, steps)) < p_sel).all(1)
    p_reason = dilution(toks)
    ok_reason = rng.random(m) < p_reason ** steps
    ok = present & ok_sel & ok_reason
    return float(ok.mean()), toks, float(p_sel)


print(f"A schema costs about {TOK_NAME} tokens of name and description plus")
print(f"{TOK_ARG} per argument, at {ARGS_MEAN} arguments on average. Every")
print("request pays for every tool offered.")
print()
print(f"{'tools':>8}{'schema tokens':>15}{'select/step':>13}{'success':>10}")
print("-" * 46)
tab = {}
for n in (8, 32, 128, 512, 2048):
    r = run(n)
    tab[n] = r
    print(f"{n:>8}{r[1]:>15,.0f}{r[2]:>13.2%}{r[0]:>10.1%}")

print()
print()
print("Where the loss comes from. 'Selection' is the decision degrading with")
print("candidate count; 'dilution' is the schema text crowding the context.")
print()
print(f"{'tools':>8}{'selection loss':>16}{'dilution loss':>15}{'total':>9}")
print("-" * 48)
sep = {}
for n in (8, 128, 512, 2048):
    toks = schema_tokens(n)
    p_sel = run(n)[2]
    sel = 1 - p_sel ** 5
    dil = 1 - dilution(toks) ** 5
    sep[n] = (sel, dil)
    print(f"{n:>8}{sel:>16.1%}{dil:>15.1%}{1 - tab.get(n, run(n))[0]:>9.1%}")

print()
print()
print("Verbosity is the variable a server author controls directly, and it is")
print("multiplicative in the token cost. 512 tools:")
print()
print(f"{'description length':>20}{'schema tokens':>15}{'success':>10}")
print("-" * 45)
vb = {}
for v, label in ((0.4, "terse"), (1.0, "typical"), (1.8, "generous"),
                 (3.0, "documentation")):
    r = run(512, verbose=v)
    vb[label] = r
    print(f"{label:>20}{r[1]:>15,.0f}{r[0]:>10.1%}")

print()
print()
print("Retrieval: show only the k most relevant schemas. The cost is that")
print("retrieval sometimes fails to surface the tool the task needs.")
print()
print(f"{'shown of 2048':>15}{'schema tokens':>15}{'recall 99%':>12}"
      f"{'recall 94%':>12}{'recall 85%':>12}")
print("-" * 66)
rt = {}
for k in (8, 16, 32, 64, 2048):
    row = tuple(run(2048, retrieved=(None if k == 2048 else k), recall=rc)[0]
                for rc in (0.99, 0.94, 0.85))
    rt[k] = (schema_tokens(min(k, 2048)), row)
    label = "all" if k == 2048 else str(k)
    print(f"{label:>15}{schema_tokens(min(k, 2048)):>15,.0f}"
          + "".join(f"{v:>12.1%}" for v in row))

print()
print()
print("And the inventory size at which retrieval starts winning, which is the")
print("number a host actually needs.")
print()
print(f"{'inventory':>11}{'show all':>11}{'retrieve 24':>14}{'better':>11}")
print("-" * 47)
xo = {}
for n in (16, 64, 256, 1024, 4096):
    a = run(n)[0]
    b = run(n, retrieved=24)[0]
    xo[n] = (a, b)
    print(f"{n:>11}{a:>11.1%}{b:>14.1%}"
          f"{('retrieve' if b > a else 'show all'):>11}")

print(f"""
The first table looks like ch:ag-tool-calling's result reversed, and the second
shows it is not.

Selection loss barely moves: {sep[8][0]:.1%} at {8} tools and {sep[2048][0]:.1%} at
{2048}. That is that chapter's finding intact -- **the DECISION is nearly free in
inventory size**, because distinct tools stay distinguishable.

Dilution loss goes {sep[8][1]:.1%} to {sep[2048][1]:.1%}. That is the bill, and it
is a different quantity entirely (eq:schemas-are-rent). A schema is text in the
context window; it is paid for on every request whether or not the tool is used;
and ch:ag-memory found that everything present competes with everything else.

So the two chapters do not disagree. **Tool count is nearly free for selection and
ruinous for context**, and a team that read the first result as permission to
connect every server available will discover the second.

The verbosity table is the part a server author controls unilaterally. The same
{512} tools cost {vb['terse'][1]:,.0f} tokens described tersely and
{vb['documentation'][1]:,.0f} written as documentation, for
{vb['terse'][0]:.1%} against {vb['documentation'][0]:.1%}.

**Description length is multiplicative in the rent**, and it is the one variable
in this listing that costs nothing to change. A server whose descriptions read
like reference documentation is charging every host that connects to it, on every
request, forever.

The retrieval table is the way out, and it is the standard answer at
cite:qin2023toolllm's scale: index the inventory, show only the schemas this
request plausibly needs. Showing {8} of {2048} gives {rt[8][1][1]:.1%} against
{rt[2048][1][1]:.1%} for showing everything.

Note the direction within that table: showing FEWER is better.
{rt[8][1][1]:.1%} at {8} shown against {rt[64][1][1]:.1%} at {64}, even though a
larger set is more likely to contain the right tool. The marginal recall is worth
less than the marginal dilution, which is the same shape ch:mcp-primitives found
for resources.

Retrieval is not free either -- the recall columns are the cost. At {0.85:.0%}
recall the whole scheme gives {rt[8][1][2]:.1%}, because the tool the task needed
was sometimes not shown. **Retrieval converts a context problem into a recall
problem**, and the recall is now the thing to measure.

The last table gives the number a host actually needs: retrieval starts winning
between {16} and {64} tools. Below that, show everything and do not build an
index. Above it, the index is not an optimisation -- at {1024} tools, showing
everything gives {xo[1024][0]:.1%}.

Which is a much smaller crossover than people expect, and it is why hosts that
connect a dozen servers without a retrieval layer degrade in a way nobody
attributes to the tool list.""")
