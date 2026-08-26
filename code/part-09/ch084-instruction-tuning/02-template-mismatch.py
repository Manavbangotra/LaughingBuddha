# Extracted from: Chapter 84 — Instruction Tuning
# Source: src/.../ch084-instruction-tuning.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What a serving/training template mismatch actually does."""

TRAIN_TEMPLATE = "<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

VARIANTS = {
    "exact match":        "<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
    "missing newline":    "<|user|>{instruction}<|end|>\n<|assistant|>\n",
    "trailing space":     "<|user|>\n{instruction}<|end|>\n<|assistant|> ",
    "different marker":   "<|human|>\n{instruction}<|end|>\n<|assistant|>\n",
    "no special tokens":  "User: {instruction}\nAssistant: ",
    "system prepended":   "<|system|>\n<|end|>\n<|user|>\n{instruction}<|end|>\n<|assistant|>\n",
}

instruction = "what is the capital of france"
reference = TRAIN_TEMPLATE.format(instruction=instruction)


def char_diff(a, b):
    """Where the strings first diverge, and by how much."""
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), n)
    return first, abs(len(a) - len(b)) + sum(1 for i in range(first, n)
                                             if a[i] != b[i])


print(f"training template: {TRAIN_TEMPLATE!r}\n")
print(f"{'variant':<20} {'identical':>10} {'first diff':>11} {'chars differ':>13}")
for name, tmpl in VARIANTS.items():
    served = tmpl.format(instruction=instruction)
    same = served == reference
    pos, n = char_diff(reference, served)
    print(f"{name:<20} {str(same):>10} {(pos if not same else '-'):>11} "
          f"{(n if not same else 0):>13}")

print("""
Every row except the first is a different string from the one the model was
tuned on, and none of them raises an error anywhere. The model receives a
prompt from a distribution it was not trained on and behaves like a partially
instruction-tuned model: mostly fine, occasionally reverting to continuation.

Note how small some of the differences are. A missing newline is one character.
The reason this costs teams days is that the symptom — 'quality dropped after
we refactored the serving code' — points at everything except a whitespace
change in a template string.

The defence is mechanical, not vigilance: serialise with ONE function, import
it in both the training and the serving path, and assert on a golden string in
CI.""")

# The mechanical defence, demonstrated.
GOLDEN = "<|user|>\nwhat is the capital of france<|end|>\n<|assistant|>\n"


def serialise(instruction):
    """The single source of truth. Both paths must call this."""
    return TRAIN_TEMPLATE.format(instruction=instruction)


assert serialise(instruction) == GOLDEN, "template drifted from the golden string"
print(f"golden-string check passed: {serialise(instruction)!r}")
