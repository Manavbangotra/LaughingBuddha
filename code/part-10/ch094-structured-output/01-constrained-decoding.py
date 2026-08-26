# -*- coding: utf-8 -*-
# Extracted from: Chapter 94 — Structured Output and Constrained Decoding
# Source: src/.../ch094-structured-output.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
