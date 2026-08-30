# -*- coding: utf-8 -*-
# Extracted from: Chapter 124 — OCR and Document AI
# Source: src/.../ch124-ocr.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""OCR error rates are quoted per character. Nothing downstream consumes characters.

A vendor reports 99% character accuracy and it sounds close to solved. The unit is
the problem: no downstream step cares about characters. An extraction pipeline
cares whether a FIELD came out exactly right, and a field is many characters, so
the per-field error compounds (eq:field-error-amplification).

Worse, the errors are not equally visible. A wrong letter in a name is obvious to
a human reader. A wrong digit in an amount produces a different, perfectly
well-formed number, and nothing downstream can tell (eq:silent-numeric-error).

This listing simulates character-level noise and measures what it does at the
field level, separating the visible failures from the silent ones.
"""
import numpy as np

rng = np.random.default_rng(71)

N_TRIAL = 20000
DIGITS = "0123456789"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Characters OCR actually confuses, rather than uniformly random substitutions.
CONFUSABLE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5",
              "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6",
              "rn": "m"}


def corrupt(s, cer):
    """Apply per-character errors at rate `cer`, preferring realistic
    confusions where one exists."""
    out = []
    for ch in s:
        if rng.random() < cer:
            if ch in CONFUSABLE and len(CONFUSABLE[ch]) == 1:
                out.append(CONFUSABLE[ch])
            else:
                pool = DIGITS if ch.isdigit() else ALPHA
                out.append(pool[int(rng.integers(0, len(pool)))])
        else:
            out.append(ch)
    return "".join(out)


def make_field(kind, length):
    if kind == "numeric":
        return "".join(DIGITS[int(rng.integers(0, 10))] for _ in range(length))
    return "".join(ALPHA[int(rng.integers(0, len(ALPHA)))] for _ in range(length))


FIELDS = [("invoice code", "alpha", 6),
          ("amount", "numeric", 8),
          ("account number", "numeric", 14),
          ("name and address line", "alpha", 40),
          ("paragraph", "alpha", 200)]

print(f"{'field':<24}{'chars':>7}" + "".join(f"{'CER ' + str(c):>12}"
                                             for c in (0.001, 0.005, 0.01, 0.05)))
print(f"{'':<24}{'':>7}" + "".join(f"{'exact %':>12}" for _ in range(4)))
print("-" * 79)

table = {}
for name, kind, L in FIELDS:
    row = []
    for cer in (0.001, 0.005, 0.01, 0.05):
        ok = 0
        for _ in range(N_TRIAL // 4):
            s = make_field(kind, L)
            ok += int(corrupt(s, cer) == s)
        row.append(ok / (N_TRIAL // 4))
    table[name] = row
    print(f"{name:<24}{L:>7}" + "".join(f"{v:>12.3f}" for v in row))

print(f"\n\nSILENT vs VISIBLE failures at CER = 0.01\n")
print(f"{'field':<24}{'wrong':>9}{'still parses':>15}{'silent share':>15}")
print("-" * 63)
for name, kind, L in FIELDS:
    wrong = silent = 0
    for _ in range(N_TRIAL // 4):
        s = make_field(kind, L)
        t = corrupt(s, 0.01)
        if t != s:
            wrong += 1
            # A numeric field that is still all digits parses fine and is wrong.
            if kind == "numeric" and t.isdigit():
                silent += 1
    n = N_TRIAL // 4
    share = silent / wrong if wrong else 0.0
    print(f"{name:<24}{wrong / n:>9.3f}{silent / n:>15.3f}{share:>15.1%}")

acct = table["account number"][2]
para = table["paragraph"][2]
print(f"""
Read the top table across the CER = 0.01 column, which is the "99% accurate"
figure a vendor would quote. A six-character invoice code survives it
{table['invoice code'][2]:.3f} of the time. A fourteen-digit account number
survives {acct:.3f} of the time. A two-hundred-character paragraph survives
{para:.3f} of the time.

That spread comes from one equation and no modelling assumptions:
eq:field-error-amplification says a field of L characters survives with
probability (1 - CER)^L, so the per-field error rate grows with field length
while the advertised number stays fixed. 99% per character is about 86% per
account number and 14% per paragraph. The metric and the requirement are
denominated in different units, and the conversion is exponential.

This is why "our OCR is 99% accurate" and "our extraction pipeline is unusable"
are routinely both true, and why the argument about it goes nowhere: the two
sides are quoting the same system in different units.

The second table asks a different question -- not how often a field is wrong, but
how often being wrong is DETECTABLE. Alphabetic fields corrupt into things a
human or a dictionary can flag. Numeric fields are the interesting case, and the
answer is more nuanced than "you cannot tell".

About a third of numeric corruptions here survive a type check
(eq:silent-numeric-error): {table['account number'][2]:.3f} exact for the account
number, with roughly 35% of its failures still being all-digits and therefore
parsing cleanly as a different, wrong number. The other two thirds cross the
digit/letter boundary -- 0 to O, 1 to l, 5 to S -- and a format check catches
them.

That splits the validation problem into two halves with different answers. The
two thirds that break the format are cheap to catch: a regex, a type cast, a
length check. Do that first, because it is nearly free and it removes most of the
error mass.

The remaining third is the dangerous part and no amount of format validation
touches it, because every corruption of a digit string into another digit string
is a well-formed value. What works there is redundancy OCR cannot corrupt
consistently: check digits, cross-field arithmetic -- do the line items sum to the
stated total? -- and agreement between two independent extractions.

Which explains a specific production pathology. A pipeline reading financial
documents looks excellent on a spot-check, because the names and addresses are
visibly fine and the malformed numbers were already rejected. The numbers that
remain wrong are exactly the ones that look right, and they are the reason anyone
built the pipeline.""")
