# Extracted from: Chapter 79 — What Foundation Models Are
# Source: src/.../ch079-what-they-are.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""What each rung of the adaptation ladder costs, in compute and dollars."""

N = 7e9                     # parameters
D_PRETRAIN = 2e12           # pretraining tokens
FLOPS_PER_TOKEN_TRAIN = 6 * N          # the 6ND rule, ch:tf-complexity
FLOPS_PER_TOKEN_INFER = 2 * N

GPU_FLOPS = 1e15            # a realistic sustained rate, not peak
GPU_COST_PER_HOUR = 2.50
MFU = 0.45                  # ch:tf-complexity: 40-55% is the achievable band


def hours(flops):
    return flops / (GPU_FLOPS * MFU) / 3600


def dollars(flops, n_gpus=1):
    return hours(flops) * GPU_COST_PER_HOUR * n_gpus


RUNGS = [
    ("zero-shot prompt",        0,        "none"),
    ("few-shot prompt",         0,        "none"),
    ("parameter-efficient tune", 6 * N * 1e7 * 0.01, "~1% of weights"),
    ("full fine-tune",          6 * N * 1e7,        "all weights"),
    ("continued pretraining",   6 * N * 5e10,       "all weights"),
    ("pretrain from scratch",   6 * N * D_PRETRAIN, "all weights"),
]

print(f"{'adaptation':<24} {'train FLOPs':>13} {'GPU-hours':>12} "
      f"{'cost (1 GPU)':>14} {'vs pretrain':>13}")
base = 6 * N * D_PRETRAIN
for name, flops, _ in RUNGS:
    if flops == 0:
        print(f"{name:<24} {'0':>13} {'0':>12} {'$0':>14} {'—':>13}")
        continue
    print(f"{name:<24} {flops:>13.2e} {hours(flops):>12,.1f} "
          f"${dollars(flops):>13,.0f} {flops / base:>12.1e}x")

print(f"\nPretraining on one GPU would take {hours(base) / 24 / 365:,.1f} years "
      f"— which is why it is done on thousands at once.")

# Equation (eq:stage-compute-ratio): instruction tuning against pretraining.
instruct = 6 * N * 5e7
print(f"instruction tuning / pretraining = {instruct / base:.1e}")
print("That fraction produces most of the difference between a base model and "
      "one a person would call usable — which is evidence it teaches behaviour, "
      "not capability.")

# The decision nobody does the arithmetic for: at what request volume does
# serving your own adapted model beat paying per token?
API_PER_1K_TOKENS = 0.002
SELF_HOST_GPU_HOUR = 2.50
TOKENS_PER_REQUEST = 800

throughput = GPU_FLOPS * MFU / FLOPS_PER_TOKEN_INFER      # tokens/second
self_host_per_token = SELF_HOST_GPU_HOUR / 3600 / throughput
api_per_token = API_PER_1K_TOKENS / 1000

print(f"\nself-hosted throughput: {throughput:,.0f} tokens/s")
print(f"self-hosted: ${self_host_per_token:.3e}/token   "
      f"API: ${api_per_token:.3e}/token")
if self_host_per_token < api_per_token:
    daily = 24 * 3600 * throughput / TOKENS_PER_REQUEST
    print(f"Self-hosting is cheaper per token, but only if the GPU is BUSY: "
          f"it must serve ~{daily:,.0f} requests/day to stay saturated.")
    print("An idle GPU costs full price. Utilisation, not unit cost, is the "
          "variable that decides this.")
