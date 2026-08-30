"""Sweep every chapter for LaTeX mangled by shell escape collapsing.

A heredoc that ate a backslash turns \\text into TAB+ext, \\frac into FF+rac,
\\rho into CR+ho, and so on. This finds every such control character in a
markdown source.

CRLF line endings are normal on this machine, so a trailing CR is stripped
before checking; a CR anywhere else is real damage.
"""
import io
import os
import re

ROOT = r"C:\Github\LaughingBuddha\src"

CTRL = {"\t": "TAB (was backslash-t, e.g. \\text)",
        "\x08": "BACKSPACE (was backslash-b)",
        "\x0c": "FORMFEED (was backslash-f, e.g. \\frac)",
        "\r": "CR mid-line (was backslash-r, e.g. \\rho)",
        "\x07": "BELL (was backslash-a, e.g. \\alpha)",
        "\x0b": "VTAB (was backslash-v)"}

SUSPECT = re.compile(r"(?<![A-Za-z\\])(ext|rac|ho|lpha|eta|ambda|igma)\{")


def show(text):
    """ASCII-safe one-line preview."""
    out = []
    for ch in text[:100]:
        if ch == "\t":
            out.append("<TAB>")
        elif ch == "\r":
            out.append("<CR>")
        elif ch == "\x0c":
            out.append("<FF>")
        elif ord(ch) < 32 or ord(ch) > 126:
            out.append("?")
        else:
            out.append(ch)
    return "".join(out)


bad = 0
files = 0
for dirpath, _, names in os.walk(ROOT):
    for n in sorted(names):
        if not n.endswith(".md"):
            continue
        path = os.path.join(dirpath, n)
        raw = io.open(path, encoding="utf-8", newline="").read()
        files += 1
        rel = os.path.relpath(path, ROOT)
        for i, line in enumerate(raw.split("\n"), 1):
            line = line.rstrip("\r")          # CRLF is normal here
            for ch, why in CTRL.items():
                if ch in line:
                    bad += 1
                    print("  %s:%d  %s" % (rel, i, why))
                    print("      " + show(line))
            m = SUSPECT.search(line)
            if m and ("\\" + m.group(1)) not in line:
                bad += 1
                print("  %s:%d  suspicious bare '%s{'" % (rel, i, m.group(1)))
                print("      " + show(line))

print()
print("scanned %d files" % files)
print("CLEAN" if not bad else "%d problem(s) found" % bad)
