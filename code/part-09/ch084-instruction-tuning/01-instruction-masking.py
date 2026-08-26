# Extracted from: Chapter 84 — Instruction Tuning
# Source: src/.../ch084-instruction-tuning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Chat template serialisation and loss masking, with the boundaries checked."""

IGNORE = -100          # the conventional "do not compute loss here" label

SPECIAL = {"<|system|>": 0, "<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}
WORDS = ["you", "are", "helpful", "what", "is", "the", "capital", "of",
         "france", "paris", "italy", "rome", "please", "explain", "briefly"]
VOCAB = list(SPECIAL) + WORDS
idx = {w: i for i, w in enumerate(VOCAB)}


def tokenize(text):
    return [idx[w] for w in text.split()]


def build_example(system, turns):
    """Serialise a conversation and build the loss mask in one pass.

    turns is a list of (user, assistant) pairs. The mask is 1 on assistant
    tokens INCLUDING the token that begins the response — that is where the
    model decides to start answering — and 0 everywhere else.
    """
    tokens, labels = [], []

    def emit(chunk, supervised):
        for t in chunk:
            tokens.append(t)
            labels.append(t if supervised else IGNORE)

    emit([idx["<|system|>"]], False)
    emit(tokenize(system), False)
    emit([idx["<|end|>"]], False)

    for user, assistant in turns:
        emit([idx["<|user|>"]], False)
        emit(tokenize(user), False)
        emit([idx["<|end|>"]], False)
        # The assistant marker is NOT supervised — the template supplies it —
        # but everything from the first content token onward is.
        emit([idx["<|assistant|>"]], False)
        emit(tokenize(assistant), True)
        emit([idx["<|end|>"]], True)      # the model must learn to stop

    return tokens, labels


tokens, labels = build_example(
    "you are helpful",
    [("what is the capital of france", "paris"),
     ("what is the capital of italy", "rome")])

print(f"{'pos':>4} {'token':<15} {'label':<15} supervised")
for i, (t, l) in enumerate(zip(tokens, labels)):
    lab = "IGNORE" if l == IGNORE else VOCAB[l]
    print(f"{i:>4} {VOCAB[t]:<15} {lab:<15} {'yes' if l != IGNORE else ''}")

n_sup = sum(1 for l in labels if l != IGNORE)
print(f"\nsupervised {n_sup} of {len(labels)} positions "
      f"({n_sup / len(labels):.0%})")

# The checks that catch the two classic bugs.
first_response = tokens.index(idx["<|assistant|>"]) + 1
assert labels[first_response] != IGNORE, \
    "the first response token MUST be supervised — it is where answering starts"
assert labels[tokens.index(idx["<|user|>"]) + 1] == IGNORE, \
    "user content must NOT be supervised"

# Every assistant turn must be supervised, not just the first — the multi-turn
# bug is a mask that stops after the first response.
assistant_starts = [i for i, t in enumerate(tokens) if t == idx["<|assistant|>"]]
print(f"assistant turns: {len(assistant_starts)}")
for start in assistant_starts:
    assert labels[start + 1] != IGNORE, f"turn at {start} is not supervised"
print("all assistant turns supervised — the multi-turn mask bug would fail here")

# Equation (eq:unmasked-gradient-share): what masking is worth on this data.
n_instruction = sum(1 for l in labels if l == IGNORE)
print(f"\nwithout masking, {n_instruction / len(labels):.0%} of the gradient "
      f"would model INSTRUCTIONS rather than responses")
print(f"masking raises the useful share from "
      f"{n_sup / len(labels):.0%} to 100%")
print("(This toy has one-word answers, so the ratio is extreme. Real "
      "instruction data runs nearer half — measure it on your own data with "
      "equation eq:unmasked-gradient-share before deciding how much masking "
      "is worth.)")
