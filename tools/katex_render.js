/*
 * katex_render.js — batch server-side math rendering.
 *
 * Reads a JSON array of {tex, display} on stdin and writes a JSON array of
 * {html} (or {error}) on stdout. Rendering at build time rather than in the
 * browser means the HTML is self-contained and the PDF is byte-identical to
 * what Paged.js laid out — no font-loading race, no runtime JS.
 *
 * Macros defined here are book-wide and must stay in sync with
 * data/notation.yaml.
 */

const katex = require("katex");

const MACROS = {
  // Linear algebra
  "\\vec": "\\mathbf{#1}",
  "\\vecgreek": "\\boldsymbol{#1}", // \mathbf does not embolden Greek letters
  "\\mat": "\\mathbf{#1}",
  "\\T": "^{\\top}",
  "\\inv": "^{-1}",
  "\\norm": "\\left\\lVert #1 \\right\\rVert",
  "\\abs": "\\left\\lvert #1 \\right\\rvert",
  "\\inner": "\\left\\langle #1, #2 \\right\\rangle",
  // Probability and statistics
  "\\E": "\\mathbb{E}",
  "\\Var": "\\operatorname{Var}",
  "\\Cov": "\\operatorname{Cov}",
  "\\Prob": "\\mathbb{P}",
  "\\KL": "D_{\\mathrm{KL}}",
  "\\logsumexp": "\\operatorname{logsumexp}",
  "\\silu": "\\operatorname{silu}",
  "\\Norm": "\\operatorname{Norm}",
  "\\FFN": "\\operatorname{FFN}",
  "\\Attn": "\\operatorname{Attention}",
  "\\pos": "\\mathit{pos}",
  "\\head": "\\operatorname{head}",
  "\\MHA": "\\operatorname{MultiHead}",
  "\\rank": "\\operatorname{rank}",
  "\\tr": "\\operatorname{tr}",
  "\\given": "\\mid",
  // Sets
  "\\R": "\\mathbb{R}",
  "\\N": "\\mathbb{N}",
  "\\Z": "\\mathbb{Z}",
  // ML operators
  "\\softmax": "\\operatorname{softmax}",
  "\\attn": "\\operatorname{Attention}",
  "\\argmax": "\\operatorname*{arg\\,max}",
  "\\argmin": "\\operatorname*{arg\\,min}",
  "\\relu": "\\operatorname{ReLU}",
  "\\sign": "\\operatorname{sign}",
  "\\diag": "\\operatorname{diag}",
  "\\tr": "\\operatorname{tr}",
  "\\rank": "\\operatorname{rank}",
  "\\logit": "\\operatorname{logit}",
  "\\Ind": "\\mathbb{1}",
  "\\Loss": "\\mathcal{L}",
  "\\Like": "L", // likelihood — deliberately NOT \mathcal{L}, which is the loss
  "\\Data": "\\mathcal{D}",
  "\\Model": "\\mathcal{M}",
  "\\dd": "\\mathrm{d}",
};

function readStdin() {
  return new Promise((resolve, reject) => {
    let buf = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (c) => (buf += c));
    process.stdin.on("end", () => resolve(buf));
    process.stdin.on("error", reject);
  });
}

(async () => {
  const items = JSON.parse(await readStdin());
  const out = items.map((it) => {
    try {
      return {
        html: katex.renderToString(it.tex, {
          displayMode: !!it.display,
          output: "htmlAndMathml", // MathML gives PDF/screen readers real math
          throwOnError: true,
          strict: "ignore",
          trust: false,
          macros: MACROS,
        }),
      };
    } catch (e) {
      return { error: String(e.message || e) };
    }
  });
  process.stdout.write(JSON.stringify(out));
})().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e));
  process.exit(1);
});
