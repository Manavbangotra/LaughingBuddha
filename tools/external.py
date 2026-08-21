"""
external.py — bridges to the Node-side renderers (KaTeX, Mermaid).

Both bridges are content-addressed and cached on disk. A rebuild that changes
one paragraph does not re-shell-out for 4,000 equations or re-launch Chrome for
600 diagrams; only new or edited content costs anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from bookdata import BUILD, ROOT

NODE = "node"
KATEX_SCRIPT = ROOT / "tools" / "katex_render.js"
MMDC = ROOT / "node_modules" / ".bin" / "mmdc"
PUPPETEER_CFG = ROOT / "tools" / "puppeteer.json"
MERMAID_CFG = ROOT / "tools" / "mermaid.json"

MATH_CACHE = BUILD / "cache" / "math"
MERMAID_CACHE = BUILD / "cache" / "mermaid"


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:20]


class MathRenderer:
    """Batch KaTeX renderer with an on-disk cache."""

    def __init__(self) -> None:
        MATH_CACHE.mkdir(parents=True, exist_ok=True)
        self.errors: list[tuple[str, str]] = []

    def render_many(self, items: list[tuple[str, bool]]) -> list[str]:
        """Render [(tex, display)] -> [html], in order."""
        results: list[str | None] = [None] * len(items)
        todo: list[tuple[int, str, bool]] = []

        for i, (tex, display) in enumerate(items):
            cf = MATH_CACHE / f"{_key(tex, str(display))}.html"
            if cf.exists():
                results[i] = cf.read_text(encoding="utf-8")
            else:
                todo.append((i, tex, display))

        if todo:
            payload = json.dumps([{"tex": t, "display": d} for _, t, d in todo])
            proc = subprocess.run(
                [NODE, str(KATEX_SCRIPT)], input=payload, capture_output=True,
                text=True, cwd=ROOT, timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"katex bridge failed: {proc.stderr[:2000]}")
            rendered = json.loads(proc.stdout)
            for (i, tex, display), r in zip(todo, rendered, strict=True):
                if "error" in r:
                    self.errors.append((tex, r["error"]))
                    esc = (tex.replace("&", "&amp;").replace("<", "&lt;")
                              .replace(">", "&gt;"))
                    results[i] = f'<code class="math-error" title="KaTeX error">{esc}</code>'
                    continue
                results[i] = r["html"]
                (MATH_CACHE / f"{_key(tex, str(display))}.html").write_text(
                    r["html"], encoding="utf-8")

        return [r or "" for r in results]


# Mermaid emits a fixed root id and scopes its <style> rules to it. Inlining
# several diagrams in one document would collide, so every diagram gets its own
# id and its stylesheet is rewritten to match.
_SVG_ID_RE = re.compile(r"my-svg")
_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)

# Mermaid bakes literal hex colours into the SVG. Because tools/mermaid.json
# pins the palette, those hexes are known in advance and can be swapped for CSS
# custom properties — which is what makes diagrams follow the light/dark theme
# instead of staying stuck in whichever mode they were rendered in. Print CSS
# resolves the same variables back to the light palette.
_THEME_HEX = {
    "#eef2f7": "var(--dgm-fill)",
    "#e2e8f0": "var(--dgm-fill-alt)",
    "#f8fafc": "var(--dgm-cluster)",
    "#64748b": "var(--dgm-line)",
    "#94a3b8": "var(--dgm-line-soft)",
    "#0f172a": "var(--dgm-text)",
    "#fef9c3": "var(--dgm-note-fill)",
    "#ca8a04": "var(--dgm-note-line)",
}
_HEX_RE = re.compile("|".join(re.escape(h) for h in _THEME_HEX), re.IGNORECASE)


class MermaidRenderer:
    """Renders ```mermaid fences to inline SVG via mermaid-cli, cached."""

    def __init__(self) -> None:
        MERMAID_CACHE.mkdir(parents=True, exist_ok=True)
        self.errors: list[tuple[str, str]] = []
        self.rendered = 0
        self.cache_hits = 0

    def render(self, source: str) -> str:
        src = source.strip()
        k = _key(src, "v2")
        cf = MERMAID_CACHE / f"{k}.svg"

        if not cf.exists():
            self._invoke(src, cf, k)
            self.rendered += 1
        else:
            self.cache_hits += 1

        if not cf.exists():
            esc = src.replace("&", "&amp;").replace("<", "&lt;")
            return f'<pre class="mermaid-error">{esc}</pre>'

        svg = cf.read_text(encoding="utf-8")
        svg = _SVG_ID_RE.sub(f"mmd-{k}", svg)
        svg = _HEX_RE.sub(lambda m: _THEME_HEX[m.group(0).lower()], svg)
        # Drop the inline max-width so page/column CSS controls sizing; the
        # viewBox keeps the aspect ratio intact.
        svg = _SVG_OPEN_RE.sub(
            lambda m: re.sub(r'style="[^"]*max-width:[^"]*"', '', m.group(0)),
            svg, count=1)
        return svg

    def _invoke(self, src: str, out: Path, k: str) -> None:
        tmp = MERMAID_CACHE / f"{k}.mmd"
        tmp.write_text(src, encoding="utf-8")
        cmd = [str(MMDC), "-i", str(tmp), "-o", str(out),
               "-p", str(PUPPETEER_CFG), "-b", "transparent"]
        if MERMAID_CFG.exists():
            cmd += ["-c", str(MERMAID_CFG)]
        env = {"PUPPETEER_EXECUTABLE_PATH": "/usr/bin/google-chrome",
               "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                              env=env, timeout=180)
        if proc.returncode != 0 or not out.exists():
            self.errors.append((src[:120], (proc.stderr or proc.stdout)[-600:]))
        tmp.unlink(missing_ok=True)
