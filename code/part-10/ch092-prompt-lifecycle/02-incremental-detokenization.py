# -*- coding: utf-8 -*-
# Extracted from: Chapter 92 — What Actually Happens When You Send a Prompt
# Source: src/.../ch092-prompt-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Why streaming cannot decode tokens independently."""

# A byte-level tokenizer, as in ch:nlp-subword. Multi-byte characters are
# routinely split across tokens, so a single token may not be valid UTF-8.
VOCAB = {
    0: b"The ", 1: b"na", 2: b"\xc3", 3: b"\xaf", 4: b"ve ",
    5: b"caf", 6: b"\xc3\xa9", 7: b" is ", 8: b"clos", 9: b"ed",
    10: b"</ans", 11: b"wer>",
}


def decode_bytes(ids):
    return b"".join(VOCAB[i] for i in ids)


TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print("Naive: decode each token independently\n")
print(f"{'token':>6} {'bytes':>12} {'decodes?':>10}  emitted")
naive_out = []
for t in TOKENS:
    raw = VOCAB[t]
    try:
        text = raw.decode("utf-8")
        ok = "yes"
    except UnicodeDecodeError:
        text = "<INVALID>"
        ok = "NO"
    naive_out.append(text)
    print(f"{t:>6} {str(raw):>12} {ok:>10}  {text!r}")

print(f"\nnaive concatenation: {''.join(naive_out)!r}")
print(f"correct full decode: {decode_bytes(TOKENS).decode('utf-8')!r}")
print("Tokens 2 and 3 are the two halves of 'ï'. Neither is valid UTF-8 alone.")


def stream_incremental(ids):
    """Equation (eq:incremental-detokenization): decode the prefix, emit the
    difference. Correct by construction, and it handles split characters
    because an incomplete prefix simply decodes to less text."""
    emitted, chunks = "", []
    for i in range(1, len(ids) + 1):
        raw = decode_bytes(ids[:i])
        # An incomplete multi-byte sequence at the end: back off until valid.
        for cut in range(len(raw), max(len(raw) - 4, -1), -1):
            try:
                text = raw[:cut].decode("utf-8")
                break
            except UnicodeDecodeError:
                continue
        else:
            text = ""
        new = text[len(emitted):]
        chunks.append(new)
        emitted = text
    return chunks, emitted


chunks, final = stream_incremental(TOKENS)
print(f"\nincremental streaming:")
print(f"{'step':>6} {'chunk emitted':>18}")
for i, c in enumerate(chunks):
    print(f"{i:>6} {c!r:>18}")
print(f"\nconcatenated: {''.join(chunks)!r}")
assert "".join(chunks) == decode_bytes(TOKENS).decode("utf-8")
print("Matches the full decode exactly. Note step 2 emitted NOTHING — the "
      "buffer held the incomplete character until step 3 completed it.")

# Stop strings live in TEXT space, not token space.
STOP = "</answer>"
full = TOKENS + [10, 11]
print(f"\nstop string {STOP!r} spans tokens {[10, 11]}: "
      f"{VOCAB[10]!r} + {VOCAB[11]!r}")
found_token_wise = any(STOP.encode() in VOCAB[t] for t in full)
found_text_wise = STOP in decode_bytes(full).decode("utf-8")
print(f"  detectable token-by-token : {found_token_wise}")
print(f"  detectable in decoded text: {found_text_wise}")
assert not found_token_wise and found_text_wise

print("""
This is why stop sequences must be checked against accumulated TEXT
(eq:stop-string) and why a streaming implementation must hold back up to
len(stop)-1 characters before emitting: otherwise the first half of the stop
string reaches the user before the second half reveals what it was.

Both problems have the same root — a token is not a character — and both are
invisible in ASCII-only testing, which is why they reach production.""")
