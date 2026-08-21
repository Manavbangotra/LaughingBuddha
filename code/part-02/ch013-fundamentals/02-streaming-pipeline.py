# Extracted from: Chapter 13 — Python Fundamentals for AI Work
# Source: src/.../ch013-fundamentals.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A generator pipeline: parse, filter, aggregate — in constant memory.

Each stage yields to the next, so at most one record exists at a time no matter
how large the input. This is the shape most data-ingestion code should have.
"""
import io
import json
from collections import Counter

# Stand in for a file too large to load; a real one would be open(path).
RAW = "\n".join(
    json.dumps({"user": f"u{i % 7}", "event": ["click", "view", "buy"][i % 3],
                "amount": (i % 13) * 1.5})
    for i in range(10_000)
)


def read_lines(handle):
    """Stage 1: yield lines. Files are already iterators; this makes it explicit."""
    for line in handle:
        line = line.strip()
        if line:
            yield line


def parse_json(lines):
    """Stage 2: decode, skipping malformed records rather than crashing."""
    for lineno, line in enumerate(lines, start=1):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Narrow exception, and the failure is recorded rather than hidden.
            print(f"  skipping malformed line {lineno}")


def only_purchases(records):
    """Stage 3: filter."""
    for r in records:
        if r.get("event") == "buy":
            yield r


def summarise(records):
    """Stage 4: aggregate. This is the only stage that accumulates."""
    totals, counts = Counter(), Counter()
    for r in records:
        totals[r["user"]] += r["amount"]
        counts[r["user"]] += 1
    return totals, counts


# The pipeline is assembled but nothing has executed yet — generators are lazy.
handle = io.StringIO(RAW)
pipeline = only_purchases(parse_json(read_lines(handle)))
print(f"pipeline object: {type(pipeline).__name__}  <- nothing computed yet")

totals, counts = summarise(pipeline)      # execution happens here

print(f"\n{'user':>6} {'purchases':>11} {'total':>10} {'mean':>9}")
for user in sorted(totals):
    print(f"{user:>6} {counts[user]:>11} {totals[user]:>10.2f} "
          f"{totals[user]/counts[user]:>9.2f}")

# --- the same pipeline handles malformed input without special-casing -------
DIRTY = RAW.split("\n")[:5] + ["{not json}", ""] + RAW.split("\n")[5:10]
print("\nre-running over input containing a malformed line:")
t2, c2 = summarise(only_purchases(parse_json(read_lines(iter(DIRTY)))))
print(f"  processed {sum(c2.values())} purchases despite the bad record")

# --- itertools composes generators without materialising anything -----------
import itertools

handle = io.StringIO(RAW)
first_three = list(itertools.islice(parse_json(read_lines(handle)), 3))
print(f"\nfirst 3 records via islice (the rest never parsed):")
for r in first_three:
    print(f"  {r}")

# Chunking a stream into batches — the shape every training loop needs.
def batched(iterable, size):
    it = iter(iterable)
    while batch := list(itertools.islice(it, size)):
        yield batch


handle = io.StringIO(RAW)
sizes = [len(b) for b in batched(parse_json(read_lines(handle)), 3000)]
print(f"\nbatch sizes from a 10,000-record stream: {sizes}")
print("Constant memory throughout — no stage ever held more than one batch.")
