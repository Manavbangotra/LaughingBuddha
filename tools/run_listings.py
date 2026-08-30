#!/usr/bin/env python3
"""Extract tier-A listings from a chapter and run them, mirroring the code gate."""
import re, subprocess, sys, tempfile, os
from pathlib import Path

FENCE = re.compile(r"^```(\w+)?\s*(\{[^}]*\})?\s*$")


def blocks(path):
    out, cur, info, inb = [], [], None, False
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        m = FENCE.match(line)
        if m and not inb:
            inb, info, cur = True, (m.group(1), m.group(2) or ""), []
            continue
        if line.strip() == "```" and inb:
            out.append((info, "\n".join(cur)))
            inb = False
            continue
        if inb:
            cur.append(line)
    return out


def main():
    failures = 0
    for path in sys.argv[1:]:
        for (lang, attrs), src in blocks(path):
            if lang != "python":
                continue
            tier = re.search(r"tier=(\w+)", attrs)
            name = re.search(r"name=([\w-]+)", attrs)
            tier = tier.group(1) if tier else "?"
            name = name.group(1) if name else "?"
            if tier != "A":
                print(f"  [skip tier {tier}] {name}")
                continue
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / f"{name}.py"
                f.write_text(src, encoding="utf-8")
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                p = subprocess.run([sys.executable, str(f)], capture_output=True,
                                   text=True, encoding="utf-8", timeout=1200, env=env)
            status = "ok " if p.returncode == 0 else "FAIL"
            print(f"  [{status}] {Path(path).name} :: {name}")
            if p.returncode != 0:
                failures += 1
                print((p.stderr or p.stdout)[-2500:])
    print("\nall listings passed" if not failures else f"\n{failures} listing(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
