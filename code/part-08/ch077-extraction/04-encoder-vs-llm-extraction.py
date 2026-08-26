# -*- coding: utf-8 -*-
# Extracted from: Chapter 77 — Classification, Named Entity Recognition, and Information Extraction
# Source: src/.../ch077-extraction.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Encoder fine-tuning versus LLM prompting for extraction, priced."""

DOCS_PER_DAY = 50_000
TOKENS_PER_DOC = 1_200
SCHEMA_CHANGES_PER_YEAR = 6
LABELLED_EXAMPLES_NEEDED = 2_000
COST_PER_LABEL_USD = 1.50          # a domain expert, per annotated document

# Encoder route: fine-tune once, serve cheaply, relabel when the schema changes.
ENCODER = dict(
    gpu_hours_per_finetune=4, gpu_cost_per_hour=2.0,
    ms_per_doc=35, cpu_cost_per_hour=0.10, cores=4,
)
# LLM route: no labelling, no training, priced per token forever.
LLM = dict(input_per_1k=0.003, output_per_1k=0.015, output_tokens=150,
           ms_per_doc=2_500)

# --- encoder ---
label_cost = LABELLED_EXAMPLES_NEEDED * COST_PER_LABEL_USD
train_cost = ENCODER["gpu_hours_per_finetune"] * ENCODER["gpu_cost_per_hour"]
setup_per_change = label_cost * 0.3 + train_cost      # partial relabel each change
encoder_setup_year = label_cost + train_cost + SCHEMA_CHANGES_PER_YEAR * setup_per_change

doc_seconds = ENCODER["ms_per_doc"] / 1000
core_hours_day = DOCS_PER_DAY * doc_seconds / 3600
encoder_serve_day = core_hours_day * ENCODER["cpu_cost_per_hour"]

# --- LLM ---
llm_per_doc = (TOKENS_PER_DOC / 1000 * LLM["input_per_1k"]
               + LLM["output_tokens"] / 1000 * LLM["output_per_1k"])
llm_serve_day = DOCS_PER_DAY * llm_per_doc

print(f"{'':22} {'setup (yr 1)':>14} {'serving/day':>13} {'serving/yr':>13} "
      f"{'total yr 1':>13}")
for name, setup, day in [("fine-tuned encoder", encoder_setup_year, encoder_serve_day),
                         ("LLM prompting", 0.0, llm_serve_day)]:
    print(f"{name:<22} ${setup:>13,.0f} ${day:>12,.2f} ${day * 365:>12,.0f} "
          f"${setup + day * 365:>12,.0f}")

print(f"\nper-document: encoder "
      f"${encoder_serve_day / DOCS_PER_DAY:.6f}, LLM ${llm_per_doc:.4f} "
      f"({llm_per_doc / (encoder_serve_day / DOCS_PER_DAY):,.0f}x)")
print(f"latency:      encoder {ENCODER['ms_per_doc']} ms, "
      f"LLM {LLM['ms_per_doc']:,} ms "
      f"({LLM['ms_per_doc'] / ENCODER['ms_per_doc']:.0f}x)")

# Where the crossover sits.
breakeven = encoder_setup_year / (llm_serve_day - encoder_serve_day)
print(f"\nbreak-even: {breakeven:.0f} days of production volume.")
print("Below that the LLM is cheaper outright; above it the encoder's setup "
      "cost is amortised. The schema-change rate is what moves this number, "
      "because every change re-pays part of the labelling.")
