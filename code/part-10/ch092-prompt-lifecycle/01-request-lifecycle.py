# -*- coding: utf-8 -*-
# Extracted from: Chapter 92 — What Actually Happens When You Send a Prompt
# Source: src/.../ch092-prompt-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A request through all eight stages, with the latency budget attributed."""

N_PARAMS = 7e9
BYTES, DEVICE_FLOPS, BANDWIDTH, MFU = 2, 1e15, 3e12, 0.45

FIXED_MS = dict(validate=0.3, template=0.05, tokenize=1.2,
                detokenize_per_token=0.02, network_per_chunk=0.4)


def prefill_ms(prompt_tokens, batch=1):
    flops = 2 * N_PARAMS * prompt_tokens * batch
    return max(flops / (DEVICE_FLOPS * MFU),
               BYTES * N_PARAMS / BANDWIDTH) * 1000


def decode_step_ms(batch):
    return max(2 * N_PARAMS * batch / (DEVICE_FLOPS * MFU),
               BYTES * N_PARAMS / BANDWIDTH) * 1000


def queue_ms(batch, arrival_rate):
    """Time to assemble a batch — equation (eq:ttft-budget)'s queue term."""
    return (batch / arrival_rate) * 1000 / 2


def budget(prompt, output, batch, arrival_rate):
    stages = {}
    stages["validate"] = FIXED_MS["validate"]
    stages["template"] = FIXED_MS["template"]
    stages["tokenize"] = FIXED_MS["tokenize"]
    stages["queue"] = queue_ms(batch, arrival_rate)
    stages["prefill"] = prefill_ms(prompt, batch)
    stages["decode"] = decode_step_ms(batch) * output
    stages["detokenize"] = FIXED_MS["detokenize_per_token"] * output
    stages["transmit"] = FIXED_MS["network_per_chunk"] * output
    return stages


WORKLOADS = {
    "chat (short in, long out)":   dict(prompt=200,   output=600, batch=32),
    "RAG (long in, short out)":    dict(prompt=8000,  output=150, batch=32),
    "classification (short both)": dict(prompt=300,   output=5,   batch=64),
    "doc analysis (long both)":    dict(prompt=20000, output=800, batch=8),
}
ARRIVAL = 40.0

for name, w in WORKLOADS.items():
    s = budget(arrival_rate=ARRIVAL, **w)
    total = sum(s.values())
    ttft = (s["validate"] + s["template"] + s["tokenize"] + s["queue"]
            + s["prefill"] + FIXED_MS["detokenize_per_token"]
            + FIXED_MS["network_per_chunk"])
    print(f"\n{name}  (prompt {w['prompt']:,}, output {w['output']}, "
          f"batch {w['batch']})")
    print(f"{'  stage':<16} {'ms':>10} {'share':>8}")
    for stage, ms in sorted(s.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<14} {ms:>10.1f} {ms / total:>7.1%}")
    print(f"  {'TOTAL':<14} {total:>10.1f}")
    print(f"  {'TTFT':<14} {ttft:>10.1f}  "
          f"({'prefill-dominated' if s['prefill'] > s['queue'] else 'queue-dominated'})")
    model_ms = s["prefill"] + s["decode"]
    print(f"  model stages are {model_ms / total:.0%} of total latency")

print("""
The share columns differ enormously across workloads, and so does the right
optimisation. Chat is decode-dominated — batch harder, quantise. RAG is
prefill-dominated — shorten the prompt, cache the prefix. Classification is
dominated by FIXED OVERHEAD, where the model is a minority of the time and a
faster model buys almost nothing.

That last case is the one teams get wrong most often: for very short generations
the tokenizer, the network and the serialisation cost more than the forward
passes do.""")
