# -*- coding: utf-8 -*-
# Extracted from: Chapter 92 — What Actually Happens When You Send a Prompt
# Source: src/.../ch092-prompt-lifecycle.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
