---
id: llm-prompt-lifecycle
number: 92
part: X
tier: full
status: draft
requires: [llm-inference, llm-decoding, llm-anatomy, nlp-subword,
           fm-instruction-tuning, mle-drift, py-engineering]
provides: [request-lifecycle, latency-budget, streaming, stop-conditions,
           detokenization-boundary, request-tracing, queueing, serving-path,
           latency-attribution]
citations: [brown2020, touvron2023llama, holtzman2020, liu2023lost,
            ouyang2022, dao2022flash, ji2023survey, radford2019]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Trace a request end to end, naming every stage between the API call and the
   final byte returned.
2. Attribute an observed latency to a specific stage rather than to "the model".
3. Explain why stop sequences can be missed and how streaming stacks avoid it.
4. Explain the detokenization boundary problem and why streaming is not simply
   decoding each token.
5. Build a latency budget and identify which stage dominates for a given
   workload.
6. Design request tracing that makes a reported bad output reproducible.
7. Identify which stages are outside the model and therefore cheaply fixable.

## 2. Why This Matters

**This chapter is the one you will actually use when something is wrong.**
{{ch:llm-anatomy}} through {{ch:llm-inference}} built the pieces; this assembles
them into the path a real request takes, including the parts that are not the
model at all — queueing, templating, batching, detokenization, streaming.

**Most production incidents localise to a stage that is not the model.** A
prompt that stopped working after a deploy, an answer cut off mid-sentence,
latency that doubled with no model change, output that differs between two
identical calls: every one of these has a stage in this chapter that explains
it, and none of them requires investigating the weights.

**Latency is a budget, not a number.** Once you can name eight stages you can
measure eight numbers and find which one moved. Teams without that decomposition
report "the model is slow" and tune the wrong thing.

**And it is where reproducibility is won or lost.** A bad output reported by a
user is worthless without the template version, the token IDs, the sampling
parameters and the seed. Deciding what to log is a design decision made here,
and it is much cheaper to make before the incident than after.

## 3. Prerequisites

{{ch:llm-inference}} for prefill, decode and the two latencies — this chapter
puts them in context. {{ch:llm-decoding}} for the sampling loop.
{{ch:llm-anatomy}} for the forward pass and {{tbl:not-in-the-model}}.
{{ch:nlp-subword}} for tokenization and detokenization.
{{ch:fm-instruction-tuning}} for the chat template. {{ch:mle-drift}} for
monitoring. {{ch:py-engineering}} for logging and tracing practice.

## 4. Intuitive Explanation

A user presses enter. Roughly a second later, words begin appearing. Here is
everything that happened.

**The request arrives** and is validated — parameters checked, the conversation
assembled from message history.

**The template is applied.** Role markers wrap the messages
({{ch:fm-instruction-tuning}}). This is a string operation and it is where a
surprising number of bugs live, because the template must match training
exactly.

**Tokenization.** The string becomes integers ({{ch:nlp-subword}}). Length is
checked against the context limit, and if it exceeds it, something is truncated —
usually the wrong thing.

**Queueing.** The request waits for a batch slot. On a loaded server this can
dominate everything else, and it is invisible to any measurement taken inside
the model.

**Prefill.** All prompt positions at once, filling the KV cache
({{ch:llm-inference}}). The first token comes out.

**The decode loop.** One token per step: forward pass, sample
({{ch:llm-decoding}}), append to cache, check stop conditions, repeat.

**Detokenization and streaming.** Tokens become text and are sent to the user
incrementally — and this is harder than it sounds, because a token is not a
character and a character is not a token.

**Stop conditions.** An end-of-sequence token, a length limit, or a stop string.
The last one is checked against *text*, not tokens, which is where it gets
interesting.

> NOTE: Exactly two of these eight stages involve the model. The other six are
> string handling, scheduling and I/O — and in the author's experience the
> majority of "the model is broken" reports resolve to one of the six.

**Why streaming is not just "decode each token".** Consider the word `naïve`
tokenized as `na`, `ï`, `ve`. If `ï` is split across two tokens at the byte
level — which byte-level BPE does routinely for non-ASCII —
detokenizing each token independently produces invalid UTF-8. **The stream must
buffer until the bytes form valid characters.** The same problem arises for
stop sequences: `</answer>` might span three tokens, so checking token-by-token
misses it.

**The mental model:** a request is a pipeline of eight stages, two of which are
the model, and latency is a sum over all eight. Where it breaks down: the stages
are not independent — batching couples queueing to throughput, and prefix
caching couples templating to prefill — so optimising one can move another.

## 5. Formal Explanation

### 5.1 The stages

$$
\underbrace{\text{validate}}_{t_1}
\to \underbrace{\text{template}}_{t_2}
\to \underbrace{\text{tokenize}}_{t_3}
\to \underbrace{\text{queue}}_{t_4}
\to \underbrace{\text{prefill}}_{t_5}
\to \underbrace{\text{decode} \times n}_{t_6}
\to \underbrace{\text{detokenize}}_{t_7}
\to \underbrace{\text{transmit}}_{t_8}
$$ (eq:request-stages)

with

$$
\text{TTFT} = t_1 + t_2 + t_3 + t_4 + t_5 + t_7^{(1)} + t_8^{(1)}
$$ (eq:ttft-decomposed)

$$
\text{total} = \text{TTFT} + (n-1)\big(t_6^{\text{step}} + t_7 + t_8\big)
$$ (eq:total-latency)

**Only $t_5$ and $t_6$ are the model.** Everything else is infrastructure, and
{{sec:9-practical-example}} measures how much of a real budget it accounts for.

### 5.2 The detokenization boundary

Let $\vec{y}$ be generated token IDs and $\text{dec}$ the detokenizer. The naive
streaming approach emits $\text{dec}(y_t)$ for each $t$. This is wrong because

$$
\text{dec}(y_1 y_2) \neq \text{dec}(y_1) \,\|\, \text{dec}(y_2)
$$ (eq:detokenization-not-concatenative)

in general. Detokenization is not a homomorphism: byte-level tokenizers
({{ch:nlp-subword}}) split multi-byte characters across tokens, so an individual
token can decode to an incomplete byte sequence.

**The correct algorithm** maintains a buffer:

$$
\text{emit}_t = \text{dec}(y_{1:t}) \setminus \text{dec}(y_{1:t-1})
$$ (eq:incremental-detokenization)

— decode the whole prefix, and emit only what is new. This is correct by
construction and costs a full detokenization per step, which is cheap relative
to a forward pass.

### 5.3 Stop conditions

Three kinds, checked in different spaces:

{#tbl:stop-conditions caption="Stop conditions and where each is evaluated. The third is the one that causes trouble, because it lives in text space while generation happens in token space."}

| Condition | Checked against | Reliable |
|---|---|---|
| EOS token | token ID | yes |
| Max tokens | count | yes |
| Stop string | **detokenized text** | needs buffering |

A stop string $s$ must be detected in the *output text*. Since a token boundary
need not align with $s$'s boundary, checking after each token requires the
accumulated text:

$$
\text{stop at } t \iff s \subseteq \text{dec}(y_{1:t})
$$ (eq:stop-string)

> WARNING: If the stop string is detected only after emitting the token that
> completed it, part of the stop string has already been streamed to the user.
> Correct implementations **hold back** up to $|s|-1$ characters before emitting,
> which introduces a small latency and is the reason streamed output sometimes
> arrives in slightly uneven chunks.

### 5.4 Queueing

With arrival rate $\lambda$, service rate $\mu$, and utilisation
$\rho = \lambda/\mu$, an M/M/1 approximation gives expected wait

$$
\E[W] = \frac{\rho}{\mu(1-\rho)}
$$ (eq:queue-wait)

**The $(1-\rho)$ denominator is the operationally important part.** At
$\rho = 0.5$ the wait equals one service time; at $\rho = 0.9$ it is nine; at
$\rho = 0.99$ it is ninety-nine. Latency does not degrade linearly with load — it
is flat and then vertical, which is why a service that was fine yesterday is
unusable today after a modest traffic increase.

### 5.5 What must be logged

For a generation to be reproducible, the record must contain:

- **Template version** and the exact serialised prompt string.
- **Token IDs**, not just the text — tokenization is not always invertible in
  practice across versions.
- **Model identity and version**, including quantisation.
- **Sampling parameters** and the seed.
- **Stop conditions** as configured.

**Text alone is insufficient.** Two different token sequences can detokenize to
the same string ({{ch:nlp-subword}}), and the model saw tokens.

## 6. Mathematical Foundation

### 6.1 Where the latency budget goes

Combining {{eq:ttft-decomposed}} with {{ch:llm-inference}}'s phase costs, for a
prompt of $T$ tokens and $n$ outputs at batch $B$:

$$
\text{TTFT} \approx t_{\text{fixed}}
 + \underbrace{\frac{2NTb^{-1}}{\text{FLOPS}\cdot\text{MFU}}}_{\text{prefill}}
 + \underbrace{\frac{B}{2\lambda}}_{\text{queue}}
$$ (eq:ttft-budget)

$$
\text{total} \approx \text{TTFT} + n\cdot\max\!\left(
 \frac{2NB}{\text{FLOPS}\cdot\text{MFU}},\ \frac{bN}{\text{bandwidth}}\right)
$$ (eq:total-budget)

**Three regimes, and they want different fixes.** Short prompt with long output:
decode dominates, so batching and quantisation help. Long prompt with short
output: prefill dominates, so prompt reduction and prefix caching help. High
load: {{eq:queue-wait}} dominates, and only capacity helps.

$\square$

Identifying which regime you are in takes one measurement and determines every
subsequent decision.

### 6.2 Why incremental detokenization is correct

Define $S_t = \text{dec}(y_{1:t})$. Detokenizers are *prefix-monotone*: adding a
token never changes previously-decoded text, so

$$
S_{t-1} \text{ is a prefix of } S_t
$$ (eq:prefix-monotone)

Therefore $S_t \setminus S_{t-1}$ — the suffix of $S_t$ beyond $|S_{t-1}|$ — is
well defined and the concatenation of all emissions equals $S_n$.

$\square$

**The property is not automatic.** A tokenizer with normalisation applied at
decode time can violate it, in which case incremental streaming is genuinely
impossible and the output must be buffered whole. This is one more reason
{{ch:nlp-preprocessing}}'s insistence on lossless, reversible tokenization is a
serving requirement and not an aesthetic preference.

### 6.3 A worked latency budget

A 7B model, 1,000-token prompt, 300-token output, batch 32, on a device at
$10^{15}$ FLOPS with 45% MFU and 3 TB/s.

**Prefill:** $2\times 7\times10^9\times 1000 = 1.4\times10^{13}$ FLOPs, at
$4.5\times10^{14}$ effective FLOPS → **31 ms**.

**Decode per step:** memory-bound at
$2\times 7\times10^9 / 3\times10^{12} = 4.7$ ms; compute at batch 32 is
$2\times7\times10^9\times32/4.5\times10^{14} = 1.0$ ms. So **4.7 ms**, and 300
steps is **1.4 s**.

**Fixed overhead** — validation, templating, tokenization, detokenization,
network — is typically 5–20 ms in total.

$$
\text{TTFT} \approx 31 + 15 + \text{queue},
\qquad
\text{total} \approx 1{,}450\ \text{ms} + \text{queue}
$$

**Decode is 97% of the total and prefill is 2%.** Yet TTFT — which is what the
user perceives as responsiveness — is dominated by prefill and queueing, neither
of which affects the total much. **The two metrics are governed by different
terms**, which is why they must be optimised separately.

## 7. Internal Mechanics

```mermaid {#fig:request-lifecycle caption="The full request path. Only the two shaded stages are the model; the other six are string handling, scheduling and I/O — and they are where most production incidents localise."}
graph TD
  A["HTTP request"] --> B["validate params<br/>assemble messages"]
  B --> C["apply chat template<br/>ch:fm-instruction-tuning"]
  C --> D["tokenize + length check<br/>ch:nlp-subword"]
  D --> E{"exceeds<br/>context?"}
  E -- yes --> F["truncate — usually<br/>the wrong thing"]
  E -- no --> G["queue for a batch slot"]
  F --> G
  G --> H["PREFILL"]
  H --> I["sample one token<br/>ch:llm-decoding"]
  I --> J["incremental detokenize<br/>eq:incremental-detokenization"]
  J --> K["stream chunk"]
  K --> L{"stop?"}
  L -- no --> M["DECODE step"]
  M --> I
  L -- yes --> N["finalise, log"]
  style H fill:#fde,stroke:#c69
  style M fill:#fde,stroke:#c69
```

**Where truncation goes wrong.** The default in most stacks drops from the start
of the conversation, which removes the system prompt first — the single most
behaviour-determining part of the input. A better policy drops the *oldest
user/assistant turns* while pinning the system prompt, and the best policy
refuses and tells the caller. Almost nobody does the third.

The reason the third is best is worth stating: truncation is the only stage in
{{eq:request-stages}} that **silently changes what was asked**. Every other
stage either succeeds or fails visibly. A truncated request returns a confident
answer to a question the user did not ask, and nothing in the response indicates
it. That asymmetry — silent corruption versus loud failure — is why an explicit
error is the right default even though it is the least convenient one, and it is
the same argument {{ch:nlp-preprocessing}} made about silent truncation at the
tokenizer.

**Streaming's chunking is not token-aligned.** Because of
{{eq:incremental-detokenization}} and the stop-string holdback, a chunk may
contain zero, one, or several tokens' worth of text. Clients that assume one
chunk per token break, and clients that assume chunks end at word boundaries
break more often.

**Cancellation.** A user closing the connection should stop generation, and
frequently does not — the request continues to occupy a batch slot until it hits
a stop condition. In a loaded system, abandoned generations consuming capacity
is a real and unglamorous cause of degraded latency.

**Retries multiply cost invisibly.** A client retrying on timeout while the
original request is still generating doubles the load at precisely the moment
the system is struggling, which is {{eq:queue-wait}}'s worst case. Retry budgets
and cancellation are the same problem seen from two ends.

**Where the fixed overhead actually is.** Tokenization of a long prompt is
milliseconds; templating is microseconds; JSON serialisation of a large response
can exceed both. For short generations the fixed overhead is a *majority* of
total latency, which is why a "fast small model" often is not.

**Message-history assembly is a stage nobody counts.** A chat request does not
arrive as a prompt; it arrives as a conversation, and the server must fetch
prior turns, apply any per-turn truncation, and serialise the result. When
history lives in a database, this is a network round trip inside $t_1$ that can
exceed prefill for short conversations — and it grows with conversation length,
which means the two length-dependent terms of {{eq:ttft-budget}} both grow
together and are easily confused with each other.

**Structured-output post-processing belongs to $t_7$.** When a response must be
parsed as JSON and validated against a schema
({{ch:llm-structured-output}}), that work happens after generation and before
transmission. For a large structured response the parse and validation can be
comparable to several decode steps, and a validation *failure* triggers a
retry — which is a full additional request, not a partial one. **The cost of a
schema violation is therefore not the parse; it is the whole generation, twice.**

**Multi-tenancy makes queueing non-uniform.** {{eq:queue-wait}} assumes a single
queue. Real services segment by tenant, priority, or expected length, and a
request's wait then depends on which queue it landed in rather than on the
system's aggregate utilisation. That is why aggregate utilisation can look
healthy while one tenant's p99 is terrible, and why the metric has to be
reported per queue to mean anything.

## 8. Implementation

The lifecycle simulated end to end, with per-stage timing.

```python {tier=A name=request-lifecycle}
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
```

Now the streaming problem that looks trivial and is not:

```python {tier=A name=incremental-detokenization}
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
```

And the queueing behaviour that makes load a cliff rather than a slope:

```python {tier=A name=queueing-cliff}
"""Latency against utilisation: flat, then vertical. Equation (eq:queue-wait)."""

SERVICE_MS = 250.0            # mean time to serve one request
mu = 1000.0 / SERVICE_MS      # requests per second, one server


def wait_ms(rho):
    """M/M/1 expected wait — equation (eq:queue-wait)."""
    if rho >= 1.0:
        return float("inf")
    return (rho / (mu * (1 - rho))) * 1000


print(f"service time {SERVICE_MS:.0f} ms, capacity {mu:.1f} req/s\n")
print(f"{'utilisation':>12} {'arrivals/s':>12} {'queue wait':>13} "
      f"{'total latency':>15} {'vs unloaded':>13}")
baseline = SERVICE_MS
for rho in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    w = wait_ms(rho)
    total = w + SERVICE_MS
    print(f"{rho:>12.0%} {rho * mu:>12.2f} {w:>12.1f}m {total:>14.1f}m "
          f"{total / baseline:>12.1f}x")

print("""
Read the last column. Between 10% and 70% utilisation, latency roughly doubles —
a change most monitoring would not flag. Between 90% and 99% it grows tenfold.

The practical consequence is that a service running comfortably at 70% has far
less headroom than it appears: a 30% traffic increase takes it to 91% and
quadruples its latency. Capacity planning on MEAN utilisation systematically
under-provisions, because the cost of being wrong is not linear.""")

# What a retry storm does, which is the same equation read backwards.
print(f"\n{'scenario':<30} {'utilisation':>12} {'wait':>14} {'status':>12}")
base_rho = 0.80
for label, multiplier in [("steady state", 1.00),
                          ("5% of clients retry once", 1.05),
                          ("15% retry once", 1.15),
                          ("25% retry once", 1.25),
                          ("all clients retry once", 2.00)]:
    rho = base_rho * multiplier
    if rho >= 1.0:
        print(f"{label:<30} {rho:>11.0%} {'unbounded':>14} "
              f"{'SATURATED':>12}")
    else:
        print(f"{label:<30} {rho:>11.0%} {wait_ms(rho):>13.0f}m "
              f"{'degraded' if rho > 0.9 else 'ok':>12}")

print("""
Starting from a healthy 80%, a 25% retry rate saturates the system entirely —
utilisation passes 1.0, the queue grows without bound, and the service is down
rather than slow. A 15% retry rate does not saturate it and still multiplies the
wait several times over.

Retries are the mechanism by which a slow service becomes an unavailable one,
and the feedback is vicious: slowness causes timeouts, timeouts cause retries,
retries cause slowness. The fix is a retry budget plus cancellation — a client
that gives up must actually free the batch slot, which requires the server to
notice the disconnection rather than generating into a closed socket.""")
```

## 9. Practical Example

A team's assistant "got slower" after a release. Nothing about the model
changed. The complaint is real — p95 latency doubled — and the cause is in a
stage nobody thought to measure.

```python {tier=A name=latency-regression-triage}
"""Attributing a latency regression to a stage rather than to the model."""

BEFORE = dict(validate=0.3, template=0.05, tokenize=1.2, queue=180.0,
              prefill=42.0, decode=1180.0, detokenize=6.0, transmit=120.0)
AFTER = dict(validate=0.3, template=0.05, tokenize=1.2, queue=930.0,
             prefill=61.0, decode=1180.0, detokenize=6.0, transmit=120.0)

MODEL_STAGES = {"prefill", "decode"}

print(f"{'stage':<14} {'before':>10} {'after':>10} {'delta':>10} "
       f"{'share of regression':>21}")
total_delta = sum(AFTER.values()) - sum(BEFORE.values())
for stage in BEFORE:
    d = AFTER[stage] - BEFORE[stage]
    share = d / total_delta if total_delta else 0
    flag = "  <- MODEL" if stage in MODEL_STAGES and abs(d) > 1 else ""
    print(f"{stage:<14} {BEFORE[stage]:>9.1f}m {AFTER[stage]:>9.1f}m "
          f"{d:>+9.1f}m {share:>20.0%}{flag}")

print(f"\n{'TOTAL':<14} {sum(BEFORE.values()):>9.1f}m "
      f"{sum(AFTER.values()):>9.1f}m {total_delta:>+9.1f}m")

model_delta = sum(AFTER[s] - BEFORE[s] for s in MODEL_STAGES)
print(f"\nregression attributable to the MODEL stages : "
      f"{model_delta:+.1f} ms ({model_delta / total_delta:.0%})")
print(f"regression attributable to QUEUEING          : "
      f"{AFTER['queue'] - BEFORE['queue']:+.1f} ms "
      f"({(AFTER['queue'] - BEFORE['queue']) / total_delta:.0%})")

# Why did queueing move? Equation (eq:queue-wait) run backwards.
SERVICE_MS = 1400.0
mu = 1000.0 / SERVICE_MS


def rho_from_wait(w_ms):
    """Invert eq:queue-wait to recover the utilisation implied by a wait."""
    w = w_ms / 1000.0
    return (w * mu) / (1 + w * mu)


r_before, r_after = rho_from_wait(BEFORE["queue"]), rho_from_wait(AFTER["queue"])
print(f"\nimplied utilisation before : {r_before:.1%}")
print(f"implied utilisation after  : {r_after:.1%}")
print(f"implied traffic increase   : {r_after / r_before - 1:+.0%}")

print("""
The model stages account for 2% of the regression. Ninety-eight per cent is
queueing, and inverting equation (eq:queue-wait) says the utilisation moved from
about 20% to about 40% — a doubling of traffic, not a slowdown.

The release did not make anything slower. It made the product more popular, and
equation (eq:queue-wait)'s non-linearity turned a traffic increase into a latency
regression. The fix is capacity, and no amount of model optimisation would have
helped.

Note also the prefill delta of +19 ms, which is real and is 2% of the problem.
A team without stage-level timing would have found it, believed it, and spent a
sprint on prompt compression.""")
```

> PRODUCTION TIP: Instrument all eight stages of {{eq:request-stages}} from the
> first day. The decomposition costs almost nothing to add and is the difference
> between attributing a regression in an hour and arguing about it for a week.

## 10. Production Considerations

**Instrument every stage.** {{eq:request-stages}} is eight timers. Without them,
a regression is unattributable and the model is blamed by default.

**Pin the system prompt against truncation.** Dropping from the start removes
the most behaviour-determining part of the input.

**Implement cancellation properly.** An abandoned generation holding a batch
slot is capacity lost at exactly the moment it is scarce.

**Set retry budgets.** `queueing-cliff` shows how quickly retries convert a slow
service into an unavailable one.

**Log token IDs, not just text.** Two token sequences can produce the same
string, and the model saw tokens ({{sec:5-formal-explanation}}).

**Alert on utilisation, not latency.** {{eq:queue-wait}} means latency is a
lagging indicator — by the time it moves, you are on the steep part of the
curve.

**What to monitor:** per-stage p50/p95/p99, utilisation, queue depth, truncation
rate, cancellation rate, and retry rate. Utilisation and queue depth are the
leading indicators; everything else confirms after the fact.

## 11. Common Mistakes

**Beginners:**

*Decoding tokens independently when streaming.*
{{eq:detokenization-not-concatenative}} — it produces invalid UTF-8 on any
non-ASCII text.

*Checking stop strings against tokens.* They live in text space
({{eq:stop-string}}).

*Measuring "latency" as one number.* {{eq:ttft-decomposed}} and
{{eq:total-latency}} are governed by different terms.

**Experienced practitioners:**

*Truncating from the start.* It removes the system prompt.

*Capacity planning on mean utilisation.* `queueing-cliff` shows the headroom at
70% is much smaller than it looks.

*Not implementing cancellation.* Invisible until the system is loaded, at which
point it is a large fraction of wasted capacity.

*Logging text without token IDs or sampling parameters.* The report is then
unreproducible, which is discovered exactly when it matters.

*Attributing regressions to the model by default.*
`latency-regression-triage` shows the model accounting for 2% of a doubling.

*Reporting aggregate utilisation on a multi-queue system.* It can look healthy
while one queue is saturated, and {{eq:queue-wait}} applies per queue.

## 12. Failure Modes

**Truncated system prompt.** *Symptom:* the model ignoring instructions,
correlated with conversation length. *Detection:* truncation-rate logging.

**Split stop sequence.** *Symptom:* generation running past its stop condition,
intermittently and depending on tokenization. *Detection:* the text-space check
in {{eq:stop-string}}.

**Invalid UTF-8 in the stream.** *Symptom:* replacement characters, only for
non-ASCII users. *Detection:* the assertion in
`incremental-detokenization`, run against a non-ASCII corpus.

**Queueing collapse.** *Symptom:* latency fine, then catastrophic, over a small
traffic increase. *Cause:* {{eq:queue-wait}}. *Detection:* utilisation, not
latency.

**Retry amplification.** *Symptom:* a slow period becoming an outage.
*Detection:* retry rate as a monitored metric.

**Abandoned generations.** *Symptom:* effective capacity below what the
arithmetic predicts. *Detection:* compare active batch slots against active
client connections.

**Template drift between deploys.** *Symptom:* quality changing with no model
change — {{ch:fm-instruction-tuning}}'s golden-string check is the defence.

## 13. Alternatives

{#tbl:serving-patterns caption="Request-handling patterns. The choice is mostly determined by whether the caller is a human waiting, and it changes which term of eq:total-latency matters."}

| Pattern | TTFT matters | Throughput matters | Where used |
|---|---|---|---|
| Synchronous, non-streaming | somewhat | somewhat | classification, extraction |
| Synchronous, streaming | **critically** | somewhat | chat |
| Asynchronous with polling | no | **critically** | long documents |
| Batch / offline | no | **critically** | bulk processing |
| Speculative streaming | critically | somewhat | latency-sensitive chat |

**What genuinely differs.** The first two return the same result and differ only
in delivery — streaming does not make generation faster, it makes the *wait*
perceptible in smaller pieces, which is a user-experience change rather than a
performance one. The asynchronous patterns give up TTFT entirely and in exchange
can batch aggressively, reaching throughput the synchronous patterns cannot.

**Choosing is mostly one question**: is a human waiting? If yes, TTFT dominates
and batch sizes stay small. If no, batch as hard as memory allows.

## 14. Evaluation

**Is the pipeline correct?**

1. **Round-trip streaming** — concatenated chunks equal the full decode, for a
   corpus including non-ASCII text. The assertion in
   `incremental-detokenization`.
2. **Stop sequences detected** when split across token boundaries.
3. **Truncation policy** verified: over-length input drops the intended part.
4. **Cancellation frees the slot**, measurable as batch occupancy after a
   disconnect.

**Is the latency acceptable?** Per-stage, at p50 and p99, under a realistic
prompt-length distribution — and reported as TTFT and total separately.

**Is a bad output reproducible?** Take a logged generation and re-run it from
the log alone. If you cannot, the log is missing something, and finding out
during an incident is expensive.

## 15. Advanced Concepts

**Speculative streaming.** {{maturity:EMERGING}} Emitting draft tokens before
verification, then correcting. Improves perceived latency and risks showing text
that is retracted — a user-experience decision as much as a technical one.

**Prompt compression.** {{maturity:EMERGING}} Shortening the prompt while
preserving its effect, which attacks the prefill term of
{{eq:ttft-budget}} directly. Most valuable in the RAG regime, where
`request-lifecycle` shows prefill dominating.

**Priority scheduling.** {{maturity:ESTABLISHED}} Separate queues by expected
generation length so short requests are not stuck behind long ones — head-of-line
blocking is the single largest source of p99 latency in naive schedulers.

**Admission control.** {{maturity:ESTABLISHED}} Rejecting requests when
{{eq:queue-wait}} predicts an unacceptable wait, rather than accepting them and
timing out. Counter-intuitive and correct: a fast rejection is more useful than a
slow failure.

**Request coalescing.** {{maturity:EMERGING}} Detecting identical in-flight
requests and serving one result to several callers. Effective where prompts
repeat, and interacts with prefix caching — note it is only sound for
deterministic decoding, since two callers sampling at $T>0$ have not asked for
the same answer and should not receive one.

**Backpressure and load shedding.** {{maturity:ESTABLISHED}} Signalling upstream
to slow down rather than accepting work that will time out. The distinction from
admission control is where the decision is made: admission control rejects at
the server, backpressure asks the client not to send. Both beat the default
behaviour, which is to accept everything and fail slowly — and
{{eq:queue-wait}} is why the default is so bad, since accepting one more request
past saturation degrades every request already queued.

**Latency-aware routing.** {{maturity:EMERGING}} Sending a request to whichever
replica has the shortest queue rather than round-robin. The gain is largest
exactly where {{eq:queue-wait}} is steepest, which means it helps most when the
system is already in trouble — and least during the testing that would have
justified building it. That mismatch between where a technique helps and where
it is easy to evaluate recurs throughout serving work.

## 16. Connection to Previous Chapters

**Backwards.** {{ch:llm-inference}}'s prefill and decode are stages $t_5$ and
$t_6$ of {{eq:request-stages}}, and its two latencies are
{{eq:ttft-decomposed}} and {{eq:total-latency}}. {{ch:llm-decoding}}'s sampling
loop is the inner loop of {{fig:request-lifecycle}}.
{{ch:nlp-subword}}'s losslessness requirement becomes
{{eq:prefix-monotone}}, without which streaming is impossible.
{{ch:fm-instruction-tuning}}'s template is stage $t_2$ and its golden-string
check is the defence against drift. {{ch:llm-anatomy}}'s
{{tbl:not-in-the-model}} is what makes stage-level attribution possible.

**Forwards.** {{ch:llm-prompting}} operates on the string built at stage $t_2$.
{{ch:llm-structured-output}} adds a constraint inside the decode loop.
{{ch:llm-function-calling}} wraps the whole lifecycle in a dispatch loop.
{{part:23}} builds the scheduler, and {{part:24}} takes up the tracing and
monitoring this chapter specifies.

## 17. Exercises

**Beginner**

1. List the eight stages of {{eq:request-stages}} and mark which involve the
   model.
2. Why can't a streaming implementation decode each token independently?
3. What is truncated by default at the context limit, and why is that the worst
   choice?

**Intermediate**

4. Using {{eq:queue-wait}}, compute the expected wait at 80% and 95%
   utilisation for a 500 ms service time.
5. For a 200-token prompt and a 1,000-token output, which stage dominates total
   latency? Which dominates TTFT?
6. Explain why a stop string may need up to $|s|-1$ characters of holdback.

**Advanced**

7. Prove {{eq:prefix-monotone}} is necessary for incremental streaming, and give
   a tokenizer design that violates it.
8. Derive the traffic increase implied by an observed queue-wait change, as
   `latency-regression-triage` does.
9. Design an admission-control policy from {{eq:queue-wait}} and state what it
   optimises.

**Implementation**

10. Extend `incremental-detokenization` with the stop-string holdback and verify
    no partial stop string is ever emitted.
11. Implement priority scheduling in simulation and measure p99 TTFT against a
    FIFO baseline on a mixed-length workload.
12. Build the reproducibility log from {{sec:5-formal-explanation}} and write a
    replay tool that regenerates an output from the log alone.
13. Simulate retry amplification: a client with a timeout and a retry policy
    against a server near capacity, and find the timeout at which the system
    collapses.

**Reasoning**

14. Your p99 latency doubled and p50 did not move. What does that pattern
    suggest, and which stage would you examine first?
15. Explain why streaming does not make generation faster but does make products
    feel faster.

## 18. Interview Questions

**Beginner**

1. Walk through what happens between an API call and the first word appearing.
2. What is streaming and why is it not just returning tokens?
3. What are the three kinds of stop condition?

**Intermediate**

4. Why does latency degrade non-linearly with load?
5. How would you attribute a latency regression to a stage?
6. What must be logged for a generation to be reproducible?

**Senior**

7. Latency doubled after a release with no model change. Diagnose it.
8. Design the truncation policy for a chat product. Justify each choice.
9. When is admission control the right response to load?

**Systems**

10. Design tracing for an LLM service. What spans, what attributes?
11. How do cancellation and retry budgets interact, and what happens without
    either?

## 19. Research Questions

**How much latency is non-model in practice?** `request-lifecycle` estimates it
per workload from a cost model. Measure it on real deployments across workload
types — if the fixed overhead is as large as the model suggests for short
generations, a good deal of model-optimisation effort is misdirected.

**What is the optimal truncation policy?** Dropping the start, the middle, or
summarising all lose different things, and the choice interacts with
{{cite:liu2023lost}}'s position effects. Measure downstream quality against
policy on real conversations.

**Can perceived latency be optimised directly?** TTFT and ITL are proxies for
what a user experiences. Whether smooth slow streaming beats bursty fast
streaming at equal total time is a measurable human question that engineering
teams answer by assumption.

**What does cancellation actually recover?** Abandoned generations are widely
assumed to be a meaningful capacity loss and rarely measured. Instrument the
disconnect-to-slot-free interval on a production system and quantify it.

## 20. Chapter Summary

A request passes through eight stages {{eq:request-stages}}, and **exactly two
of them are the model.** The other six — validation, templating, tokenization,
queueing, detokenization, transmission — are string handling, scheduling and
I/O, and they are where most production incidents localise.

**Latency is two numbers governed by different terms.** TTFT
{{eq:ttft-decomposed}} is dominated by prefill and queueing; total latency
{{eq:total-latency}} is dominated by decode. A workload can be 97% decode by
total time while its perceived responsiveness is set entirely by prefill — and
`request-lifecycle` shows a short-generation workload where the *model* is a
minority of the time and a faster model buys almost nothing.

**Streaming is not decoding each token.** Detokenization is not concatenative
{{eq:detokenization-not-concatenative}} — byte-level tokenizers split multi-byte
characters across tokens, so individual tokens need not be valid UTF-8. The
correct algorithm decodes the whole prefix and emits the difference
{{eq:incremental-detokenization}}, which works because detokenizers are
prefix-monotone {{eq:prefix-monotone}}. Stop strings have the same root problem:
they live in text space {{eq:stop-string}}, may span token boundaries, and
require holding back up to $|s|-1$ characters before emitting.

**Load is a cliff, not a slope.** {{eq:queue-wait}}'s $(1-\rho)$ denominator
means latency is flat to about 70% utilisation and vertical after 90% — so a
service running comfortably has far less headroom than it appears, and retries
are the mechanism that converts slow into unavailable.

Finally the attribution discipline the chapter exists for.
`latency-regression-triage` works a case where latency doubled after a release,
the model stages account for **2%** of it, and inverting {{eq:queue-wait}}
reveals the real cause: the release made the product more popular. A team
without per-stage timing would have found the real +19 ms prefill delta,
believed it, and optimised the wrong thing for a sprint.

## 21. Further Reading

There is no single paper for this chapter, which is itself informative:
**the request lifecycle is engineering folklore that has not been written down
carefully**, and most of what circulates is vendor documentation for a
particular stack.

{{cite:dao2022flash}} and the serving literature of {{part:23}} cover stages
$t_5$ and $t_6$ properly. For everything else, the best available sources are
the source code of open serving stacks — reading a scheduler's admission and
batching logic is worth more than any description of it, and the code is short
enough to read in an afternoon.

{{cite:liu2023lost}} is relevant to the truncation policy question in
{{sec:7-internal-mechanics}}: what you drop matters, and where the remaining
content sits in the window matters too.

Queueing theory's standard texts cover {{eq:queue-wait}} and its assumptions.
The M/M/1 model is a poor fit for LLM serving — service times are heavily
variable and batching violates independence — so treat it as a source of the
*shape* of the curve rather than of its values.

**Where to go next:** {{ch:llm-prompting}} takes stage $t_2$ seriously — what
goes into the string, what the evidence for each technique actually is, and
which widely-repeated advice has never been tested.
