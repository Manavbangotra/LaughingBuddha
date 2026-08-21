# Extracted from: Chapter 19 — Files, JSON, APIs, and SQL from Python
# Source: src/.../ch019-io-apis-sql.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A resilient HTTP client: retries, backoff with jitter, and pagination.

A local http.server in a background thread stands in for a real API, and is
deliberately made flaky and rate-limiting so the client's error handling
actually runs. Standard library only — no network access required.
"""
import json
import random
import threading
import time
import urllib.error
import urllib.request
from functools import wraps
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ----------------------------------------------------------------- the server

RECORDS = [{"id": i, "value": i * 3} for i in range(1, 48)]
STATE = {"calls": 0, "failures_served": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass                                   # silence the default logging

    def do_GET(self):
        STATE["calls"] += 1
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Unknown paths 404. Without this the server would answer everything,
        # and the "404 is not retryable" check below would pass vacuously.
        if parsed.path != "/items":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"not found"}')
            return

        # Fail the first two calls with a 503, then rate-limit once, so the
        # client's backoff and Retry-After handling are both exercised.
        if STATE["calls"] <= 2:
            STATE["failures_served"] += 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"unavailable"}')
            return
        if STATE["calls"] == 4:
            STATE["failures_served"] += 1
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(b'{"error":"rate limited"}')
            return

        cursor = int(params.get("after", ["0"])[0])
        limit = int(params.get("limit", ["10"])[0])
        page = [r for r in RECORDS if r["id"] > cursor][:limit]
        next_cursor = page[-1]["id"] if len(page) == limit else None
        body = json.dumps({"data": page, "next": next_cursor}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


server = HTTPServer(("127.0.0.1", 0), Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"
print(f"test server listening on {BASE}\n")

# ----------------------------------------------------------------- the client

RETRYABLE = {429, 500, 502, 503, 504}


def with_retry(max_attempts=5, base=0.05, cap=1.0, multiplier=2.0):
    """Retry on retryable statuses, with full jitter (eq. 19.1).

    Full jitter — a uniform draw over [0, computed_delay] — spreads concurrent
    clients most evenly and is what prevents a thundering herd (section 6.1).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except urllib.error.HTTPError as exc:
                    if exc.code not in RETRYABLE or attempt == max_attempts:
                        raise
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after is not None:
                        delay = float(retry_after)
                        why = f"honouring Retry-After={retry_after}s"
                    else:
                        window = min(cap, base * multiplier ** (attempt - 1))
                        delay = random.uniform(0, window)
                        why = f"backoff window {window:.2f}s, jittered"
                    print(f"    attempt {attempt}: HTTP {exc.code}; "
                          f"{why}; sleeping {delay:.2f}s")
                    time.sleep(min(delay, 0.3))     # keep the demo quick
        return wrapper
    return decorator


@with_retry()
def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def iter_records(base_url: str, page_size: int = 10):
    """Cursor pagination as a generator: constant memory, caller sees records.

    The caller never learns that pagination happened, which is exactly the
    encapsulation a client should provide (Chapter 13).
    """
    cursor, pages = 0, 0
    while True:
        payload = get_json(f"{base_url}/items?after={cursor}&limit={page_size}")
        pages += 1
        for record in payload["data"]:
            yield record
        if payload["next"] is None:
            break
        cursor = payload["next"]
    iter_records.pages = pages


print("fetching all records through a flaky, rate-limiting API:")
records = list(iter_records(BASE))
print(f"\nretrieved {len(records)} records over {iter_records.pages} pages")
print(f"server saw {STATE['calls']} requests, of which "
      f"{STATE['failures_served']} were deliberate failures")
print(f"first: {records[0]}, last: {records[-1]}")
assert len(records) == len(RECORDS)
assert [r["id"] for r in records] == [r["id"] for r in RECORDS]
print("all records retrieved, in order, despite the failures")

# --- a non-retryable error must fail fast, not retry ------------------------
print("\na 404 is not retryable — the client should fail immediately:")
t0 = time.perf_counter()
try:
    get_json(f"{BASE}/nonexistent-path-that-404s")
except urllib.error.HTTPError as exc:
    print(f"  HTTPError {exc.code} raised after "
          f"{(time.perf_counter()-t0)*1000:.0f} ms with no retries")

# --- section 6.1: why jitter matters, simulated ------------------------------
print("\n" + "=" * 66)
print("thundering herd: deterministic backoff vs full jitter")
print("=" * 66)
rng = random.Random(0)
n_clients = 1000
print(f"{'attempt':>8} {'deterministic':>28} {'full jitter':>28}")
for attempt in range(1, 5):
    window = 0.5 * 2 ** (attempt - 1)
    det_times = [window] * n_clients                       # all identical
    jit_times = [rng.uniform(0, window) for _ in range(n_clients)]

    # Peak load = most requests landing in any 100 ms bucket.
    def peak(times):
        buckets = {}
        for t in times:
            buckets[round(t, 1)] = buckets.get(round(t, 1), 0) + 1
        return max(buckets.values())

    print(f"{attempt:>8} {f'{peak(det_times)} req in one 100ms bucket':>28} "
          f"{f'{peak(jit_times)} req in one 100ms bucket':>28}")
print("\nDeterministic backoff delivers all 1,000 clients simultaneously at")
print("every retry — the same load that caused the outage. Jitter spreads")
print("them over a widening window, which is what lets a service recover.")

server.shutdown()
