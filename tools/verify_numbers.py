"""Check that numbers quoted in a chapter's Practical Example come from its own
listings.

Section 9 of every full-tier chapter quotes tables produced by that chapter's
tier-A listings. Those quotes are transcribed by hand, and a listing that is
later corrected leaves the transcription stale — a class of error the structural
check cannot see and a reader would notice immediately.

This tool runs each chapter's tier-A listings, collects their stdout, and
verifies that every decimal number appearing inside a fenced block in section 9
also appears in that output.

Usage:
    python tools/verify_numbers.py                 # whole book
    python tools/verify_numbers.py --part 21       # one part
    python tools/verify_numbers.py --show FILE     # locate bad lines in one file
"""
import argparse
import io
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

LISTING = re.compile(r"```python \{tier=A[^}]*\}\n(.*?)```", re.S)
FENCE = re.compile(r"```\n(.*?)```", re.S)
NUMBER = re.compile(r"-?\d+\.\d+%?")
TIMEOUT = 900


def listing_output(md):
    """Run every tier-A listing in a chapter; return the concatenated stdout."""
    out = []
    for code in LISTING.findall(md):
        fd, path = tempfile.mkstemp(suffix=".py")
        os.close(fd)
        io.open(path, "w", encoding="utf-8").write(code)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=TIMEOUT)
            out.append(r.stdout)
        except subprocess.TimeoutExpired:
            out.append("")
        finally:
            os.remove(path)
    return "\n".join(out)


def section_nine(md):
    """The Practical Example section, or None if the chapter has no listings."""
    if "## 9. Practical Example" not in md or "tier=A" not in md:
        return None
    head = md.index("## 9. Practical Example")
    tail = md.index("## 10.") if "## 10." in md else len(md)
    return md[head:tail]


def quoted_numbers(sec):
    nums = []
    for block in FENCE.findall(sec):
        nums += NUMBER.findall(block)
    return nums


def chapters(part=None):
    for dirpath, _, names in os.walk(SRC):
        if part is not None and ("part-%02d-" % part) not in dirpath:
            continue
        for n in sorted(names):
            if n.endswith(".md") and not n.startswith("_"):
                yield os.path.join(dirpath, n)


def show(path):
    """Print the lines in section 9 that quote numbers the listings never emit."""
    md = io.open(path, encoding="utf-8").read()
    sec = section_nine(md)
    if sec is None:
        print("no tier-A listings in", path)
        return 0
    out = listing_output(md)
    base = md[:md.index("## 9. Practical Example")].count("\n") + 1
    inside = False
    bad = 0
    for i, line in enumerate(sec.split("\n")):
        if line.startswith("```"):
            inside = not inside
            continue
        if not inside:
            continue
        stale = [t for t in NUMBER.findall(line) if t not in out]
        if stale:
            bad += 1
            print("%5d | %s" % (base + i, line))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int)
    ap.add_argument("--show")
    args = ap.parse_args()

    if args.show:
        sys.exit(1 if show(args.show) else 0)

    checked = 0
    bad_files = 0
    bad_nums = 0
    for path in chapters(args.part):
        md = io.open(path, encoding="utf-8").read()
        sec = section_nine(md)
        if sec is None:
            continue
        quoted = quoted_numbers(sec)
        if not quoted:
            continue
        checked += 1
        out = listing_output(md)
        stale = sorted({t for t in quoted if t not in out})
        if stale:
            bad_files += 1
            bad_nums += len(stale)
            rel = os.path.relpath(path, SRC)
            print("  %s: %d of %d quoted numbers not in listing output"
                  % (rel, len(stale), len(quoted)))
            print("     " + ", ".join(stale[:12]))

    print()
    print("verify_numbers: %d chapter(s) in scope" % checked)
    if bad_nums:
        print("FAILED - %d stale number(s) in %d file(s)" % (bad_nums, bad_files))
        print("Re-run with --show <file> to locate them.")
        sys.exit(1)
    print("PASSED - every quoted number matches its listing output")


if __name__ == "__main__":
    main()
