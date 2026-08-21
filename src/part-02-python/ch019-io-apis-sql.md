---
id: py-io-apis-sql
number: 19
part: II
tier: focused
status: reviewed
requires: [py-pandas, py-functions-classes]
provides: [serialisation, json-term, parquet-term, rest-api, idempotency,
           rate-limiting, exponential-backoff, parameterised-query,
           sql-injection, n-plus-one-query]
citations: [pep8]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Choose a file format deliberately, and state what each one loses.
2. Explain what JSON's type system cannot represent and how information is lost
   through it.
3. Read a REST API robustly: handling errors, rate limits, and pagination.
4. Implement retry with exponential backoff and jitter, and explain why jitter
   matters.
5. Explain idempotency and why it determines what is safe to retry.
6. Query SQL from Python with parameterised queries, and explain why string
   concatenation is a vulnerability rather than a style issue.
7. Recognise and fix the N+1 query problem.
8. Decide what belongs in SQL and what belongs in pandas.

## 2. Why This Matters

Data does not begin in memory. It arrives from files someone else wrote, APIs
that fail intermittently, and databases you did not design. Getting it in
correctly is unglamorous and it is where a large share of real defects
originate.

Three failure modes recur throughout this book.

**Silent type loss.** Write a DataFrame to CSV and read it back: your datetime
is now a string, your integers may be floats, your categories are objects, and
your `NaN` may have become the literal text `"nan"`. Nothing errors. The
pipeline runs, and the model trains on subtly different data than you tested.

**Fragile network code.** An API call that works in a notebook fails in
production because the service rate-limited you, or returned a 503, or paginated
its response and you only read the first page. Retry logic is not optional, and
naive retry logic makes outages worse rather than better.

**SQL injection.** Building a query with string concatenation is not a style
preference. It is the most consequential and most easily avoided vulnerability
in application code, and the fix — parameterised queries — is also faster.

## 3. Prerequisites

{{ch:py-pandas}} for DataFrames, {{ch:py-functions-classes}} for decorators and
context managers — retry is a decorator and a database connection is a context
manager.

## 4. Intuitive Explanation

### 4.1 File formats lose different things

{{term:serialisation}} converts objects to bytes. Every format makes a trade,
and the trade determines what survives the round trip.

{#tbl:file-formats caption="File formats for tabular data. The 'preserves types' column is the one that causes silent bugs when ignored."}

| Format | Human-readable | Preserves types | Compression | Column subset | Typical use |
|---|---|---|---|---|---|
| CSV | yes | **no** | none | no | interchange, small data |
| JSON | yes | partially | none | no | APIs, nested config |
| {{term:parquet-term}} | no | **yes** | good | **yes** | analytics, storage |
| Pickle | no | yes (Python only) | none | no | short-lived caches only |
| SQLite | no | yes | some | yes | a single-file database |

CSV's near-universality is its only real advantage. It has no type system at
all: everything is text, and the reader guesses. Those guesses differ between
tools and between versions of the same tool.

Parquet is the right default for anything you will read again. It stores types,
compresses well, and — because it is columnar — lets you read three columns of a
two-hundred-column file without touching the rest.

> WARNING: Never unpickle data you did not create. Unpickling executes arbitrary
> code by design, so loading an untrusted pickle is equivalent to running an
> untrusted program. This matters because model checkpoints are frequently
> pickles, and downloading one from an untrusted source is a genuine remote-code
> execution risk ({{ch:sec-poisoning}}).

### 4.2 JSON's small type system

{{term:json-term}} has six types: object, array, string, number, boolean, null.
That is all.

There are no dates, so they become strings and the format is by convention. There
is no distinction between integer and float, so `1` and `1.0` may or may not
survive as different things. There is no binary, so bytes must be base64-encoded.
There are no sets, tuples, or `NaN`.

Every one of these turns into a silent conversion somewhere. The discipline is to
know what your data contains and to convert deliberately at the boundary rather
than discovering the conversion later.

### 4.3 Networks fail, and retry logic must assume it

A local function call either runs or raises. A network call has a third
possibility: it might have succeeded and you did not hear about it.

That distinction drives everything. If a request timed out, you do not know
whether the server processed it. Retrying a read is harmless; retrying a payment
may charge twice. **{{term:idempotency}}** — the property that repeating an
operation has the same effect as doing it once — is what makes retry safe, and
whether an operation is idempotent is a property of the API, not of your client.

When you do retry, the delay matters. Retrying immediately hammers a service
that is already struggling. {{term:exponential-backoff}} — 1s, 2s, 4s, 8s —
gives it room to recover.

The subtler point is **jitter**. If a thousand clients all fail at the same
moment and all back off by exactly the same schedule, they retry in unison and
the service is hit by a synchronised wave every time. Adding randomness spreads
them out, and it is the difference between a recovery and a repeated outage.

### 4.4 SQL and pandas overlap, and the boundary matters

Both can filter, join and aggregate. The useful division:

**Do it in SQL** when it reduces the amount of data. Filtering, aggregating and
joining server-side means less crosses the network and less occupies your
memory. Databases are also very good at this — they have indexes and query
planners.

**Do it in pandas** when it needs Python. Complex feature engineering,
model-specific transformations, anything calling a library the database does not
have.

The rule of thumb: **push the reduction down, pull the complexity up.**

## 5. Formal Explanation

### 5.1 Reading and writing safely

```python {tier=C name=safe-csv}
df = pd.read_csv(
    path,
    dtype={"user_id": "int64", "category": "category"},   # do not guess
    parse_dates=["created_at"],                            # explicit
    na_values=["", "NA", "N/A", "null", "-"],              # what means missing
    keep_default_na=True,
)
```

Specifying `dtype` explicitly is the single most valuable habit here. Without
it, pandas infers types from a sample of the file, which means the same code can
produce different dtypes on different data — and an ID column that happens to
contain a missing value silently becomes a float
({{ch:py-pandas}}).

### 5.2 HTTP status codes and what they mean for retry

{#tbl:http-status caption="HTTP status codes grouped by the response they demand. Retrying a 4xx (except 429) is pointless — the request itself is wrong."}

| Code | Meaning | Retry? |
|---|---|---|
| 200, 201, 204 | success | — |
| 400 | bad request | **no** — fix the request |
| 401, 403 | unauthenticated, forbidden | **no** — fix credentials |
| 404 | not found | **no** |
| 409 | conflict | maybe, after resolving |
| 429 | rate limited | **yes**, after `Retry-After` |
| 500 | server error | yes, with backoff |
| 502, 503, 504 | gateway, unavailable, timeout | yes, with backoff |

The dividing line is whether the fault is in the request or in the server.
Retrying a 400 will fail identically a thousand times.

> PRODUCTION TIP: Always honour a `Retry-After` header when present. It is the
> server telling you exactly when to come back, and ignoring it is how a
> client gets blocked rather than throttled.

### 5.3 Backoff with jitter

The delay before attempt $k$:

$$
d_k = \min\!\left(d_{\max},\; d_0 \cdot b^{\,k-1}\right) \cdot U(0, 1)
$$ (eq:backoff-jitter)

with base delay $d_0$, multiplier $b$ (usually 2), a cap $d_{\max}$, and
$U(0,1)$ a uniform random draw — this is *full jitter*, the variant that spreads
clients most evenly.

The cap matters: without it, the eighth retry waits over two minutes and the
twelfth over an hour.

### 5.4 Pagination

APIs return large collections in pages. Two schemes:

**Offset-based** — `?offset=100&limit=50`. Simple, and unstable: if rows are
inserted while you page, you will see duplicates or skip records.

**Cursor-based** — `?after=abc123`. The server returns an opaque pointer to
where you stopped. Stable under concurrent modification, and the reason most
serious APIs use it.

A generator is the natural way to consume either ({{ch:py-fundamentals}}): it
yields records while fetching pages lazily, so the caller never sees the
pagination and memory stays constant.

### 5.5 Parameterised queries

```python {tier=C name=sql-parameterised}
cur.execute("SELECT * FROM users WHERE name = ?", (name,))    # correct
cur.execute(f"SELECT * FROM users WHERE name = '{name}'")     # VULNERABLE
```

The first sends the query and the value separately. The database parses the
query once, then binds the value as *data* — it can never be interpreted as SQL.

The second builds a string. If `name` is `'; DROP TABLE users; --` then the
database receives, and executes, two statements.

> WARNING: This is {{term:sql-injection}}, and it remains one of the most
> exploited classes of vulnerability in production software despite the fix
> being a one-character change. There is no case where concatenation is
> necessary for *values*. Identifiers — table and column names — cannot be
> parameterised, so if those must be dynamic, validate them against an
> allowlist rather than interpolating user input.

Parameterisation is also faster: the database can cache the parsed plan and
reuse it across calls with different values.

### 5.6 The N+1 problem

```python {tier=C name=n-plus-one}
users = query("SELECT id FROM users LIMIT 100")     # 1 query
for u in users:
    orders = query("SELECT * FROM orders WHERE user_id = ?", (u.id,))  # 100 more
```

101 round trips where one would do. Each has network latency, so a 1 ms query
becomes 101 ms of mostly waiting. At scale this dominates everything else.

The fixes are a `JOIN`, or a single `WHERE user_id IN (...)` followed by grouping
in Python. Both turn $N+1$ round trips into one.

This pattern is easy to write and hard to see, because the code looks like an
ordinary loop. It is the most common performance bug in database-backed
applications.

## 6. Mathematical Foundation

### 6.1 Why jitter is not a detail

Consider $N$ clients that all fail at $t = 0$ and retry on a deterministic
schedule $d_0 b^{k-1}$. Every client retries at exactly the same instants, so
the server sees a load of $N$ concentrated at each retry time and zero in
between.

With full jitter {{eq:backoff-jitter}}, attempt $k$ arrives uniformly over
$[0, d_0 b^{k-1}]$. Expected instantaneous load falls from $N$ concentrated at a
point to

$$
\lambda_k = \frac{N}{d_0 b^{k-1}}
$$ (eq:jittered-load)

spread across the interval, and it halves with every subsequent attempt.

The concrete difference: 1,000 clients backing off deterministically deliver
1,000 simultaneous requests at $t = 1$s, again at $t = 3$s, again at $t = 7$s. A
service that fell over under that load falls over again each time — the
*thundering herd*. With jitter, the same 1,000 requests spread over a widening
window, and the server drains them.

> IMPORTANT: Retry without jitter is a well-documented way of converting a brief
> outage into a sustained one, because the retry traffic itself prevents
> recovery. This recurs at a larger scale in {{ch:sd-fault-tolerance}}, where
> circuit breakers exist for the same reason.

### 6.2 Costing the N+1 problem

For $n$ records, with per-query latency $\ell$ and per-row processing cost $c$:

$$
T_{N+1} = (n + 1)\ell + nc
\qquad\text{versus}\qquad
T_{\text{join}} = \ell + nc
$$ (eq:n-plus-one-cost)

The difference is $n\ell$, and $\ell$ is dominated by network round-trip time —
typically 0.5-5 ms to a database on the same network, and 10-100 ms across a
region.

For $n = 1000$ at $\ell = 2$ ms, that is 2 seconds of pure waiting, against 2 ms
for the joined version. The processing term $nc$ is identical in both, so the
entire difference is latency you did not need to pay.

### 6.3 Why columnar formats win for analytics

A row-oriented format stores record 1 completely, then record 2, and so on. A
columnar format stores all of column A, then all of column B.

Reading $k$ columns out of $m$ from a file of $n$ rows:

$$
\text{bytes}_{\text{row}} \approx n \cdot m \cdot \bar{w}
\qquad
\text{bytes}_{\text{column}} \approx n \cdot k \cdot \bar{w}
$$ (eq:columnar-io)

The row format must read everything, because the columns you want are
interleaved with the ones you do not. The columnar format reads only what you
asked for — a factor of $m/k$ less I/O.

Compression compounds this. A column holds values of one type with similar
distributions, which compresses far better than a row mixing an integer, a
timestamp and a string. Compression ratios of 5-10× are ordinary for Parquet
and unavailable to CSV.

## 7. Implementation

```python {tier=A name=formats-and-sql}
"""File formats, JSON's type losses, and SQL from Python.

Standard library plus pandas only, so every listing runs offline.
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

work = Path(tempfile.mkdtemp(prefix="io-"))

# --- what a CSV round trip loses --------------------------------------------
print("=" * 66)
print("CSV has no type system")
print("=" * 66)
df = pd.DataFrame({
    "id": pd.Series([1, 2, 3], dtype="int64"),
    "when": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
    "cat": pd.Series(["a", "b", "a"], dtype="category"),
    "val": [1.5, np.nan, 3.5],
    "flag": [True, False, True],
})
print(f"original dtypes:\n{df.dtypes.to_dict()}")

csv_path = work / "data.csv"
df.to_csv(csv_path, index=False)
naive = pd.read_csv(csv_path)
print(f"\nafter a naive CSV round trip:\n{naive.dtypes.to_dict()}")
print(f"  'when' is now {naive['when'].dtype} — a string")
print(f"  'cat'  is now {naive['cat'].dtype} — lost the category dtype")

careful = pd.read_csv(csv_path, dtype={"id": "int64", "cat": "category"},
                      parse_dates=["when"])
print(f"\nwith explicit dtypes:\n{careful.dtypes.to_dict()}")
print(f"  round trip faithful: "
      f"{careful.dtypes.astype(str).equals(df.dtypes.astype(str))}")

# --- Parquet preserves everything, for free ----------------------------------
try:
    pq_path = work / "data.parquet"
    df.to_parquet(pq_path)
    back = pd.read_parquet(pq_path)
    print(f"\nParquet round trip, no arguments needed: "
          f"{back.dtypes.astype(str).equals(df.dtypes.astype(str))}")
    print(f"  sizes: csv {csv_path.stat().st_size} B, "
          f"parquet {pq_path.stat().st_size} B")
except ImportError:
    print("\n(pyarrow not installed; Parquet demonstration skipped)")

# --- JSON's type system is small ---------------------------------------------
print("\n" + "=" * 66)
print("what JSON cannot represent")
print("=" * 66)
original = {
    "when": pd.Timestamp("2026-01-01"),
    "count": 3,
    "ratio": 3.0,
    "tags": {"a", "b"},
    "pair": (1, 2),
    "missing": float("nan"),
}
for key, value in original.items():
    try:
        encoded = json.dumps({key: value})
        decoded = json.loads(encoded)[key]
        note = ""
        if type(decoded) is not type(value):
            note = f"  <- {type(value).__name__} became {type(decoded).__name__}"
        print(f"  {key:<9} {str(value):<22} -> {str(decoded):<22}{note}")
    except TypeError as exc:
        print(f"  {key:<9} {str(value):<22} -> TypeError: {str(exc)[:34]}")

print("\nNaN is the dangerous one: json.dumps emits bare NaN, which is NOT")
print("valid JSON, and strict parsers in other languages reject it.")
print(f"  json.dumps(nan) = {json.dumps(float('nan'))!r}")
print(f"  round trips in Python: {json.loads(json.dumps(float('nan')))}")
print("  Use allow_nan=False to catch it at the boundary:")
try:
    json.dumps({"x": float("nan")}, allow_nan=False)
except ValueError as exc:
    print(f"    ValueError: {exc}")

# --- SQL: parameterised queries and injection --------------------------------
print("\n" + "=" * 66)
print("SQL injection, demonstrated")
print("=" * 66)
conn = sqlite3.connect(":memory:")
conn.executescript("""
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, tier TEXT);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
    INSERT INTO users (name, tier) VALUES
        ('alice','pro'), ('bob','free'), ('carol','pro'), ('dan','free');
    INSERT INTO orders (user_id, amount) VALUES
        (1, 10.0), (1, 25.0), (2, 5.0), (3, 40.0), (3, 15.0), (4, 8.0);
""")
conn.commit()

safe_name = "alice"
rows = conn.execute("SELECT * FROM users WHERE name = ?", (safe_name,)).fetchall()
print(f"parameterised, normal input : {rows}")

attack = "alice' OR '1'='1"
rows = conn.execute("SELECT * FROM users WHERE name = ?", (attack,)).fetchall()
print(f"parameterised, attack input : {rows}  <- treated as a literal name")

vulnerable = f"SELECT * FROM users WHERE name = '{attack}'"
rows = conn.execute(vulnerable).fetchall()
print(f"concatenated, attack input  : {len(rows)} rows returned")
print(f"  the query became: {vulnerable}")
print("  '1'='1' is always true, so every row leaked.")

# It gets worse: executescript would allow statement chaining.
print("\nWith a driver that permits multiple statements, the same hole allows")
print("';DROP TABLE users;--' — data destruction, not just disclosure.")

# --- eq. 19.3: the N+1 problem -----------------------------------------------
print("\n" + "=" * 66)
print("N+1 queries")
print("=" * 66)
import time

LATENCY = 0.001          # simulate 1 ms per round trip


def query(sql, params=()):
    time.sleep(LATENCY)
    return conn.execute(sql, params).fetchall()


t0 = time.perf_counter()
users = query("SELECT id, name FROM users")
totals_n1 = {}
for uid, name in users:
    rows = query("SELECT amount FROM orders WHERE user_id = ?", (uid,))
    totals_n1[name] = sum(r[0] for r in rows)
t_n1 = time.perf_counter() - t0

t0 = time.perf_counter()
rows = query("""
    SELECT u.name, COALESCE(SUM(o.amount), 0)
    FROM users u LEFT JOIN orders o ON o.user_id = u.id
    GROUP BY u.id, u.name
""")
totals_join = dict(rows)
t_join = time.perf_counter() - t0

assert totals_n1 == totals_join
print(f"N+1 approach : {len(users)+1} queries, {t_n1*1000:.1f} ms")
print(f"single join  : 1 query,   {t_join*1000:.1f} ms")
print(f"same answer  : {totals_join}")
print(f"\neq. 19.3 with n={len(users)}, latency={LATENCY*1000:.0f}ms predicts a")
print(f"{len(users)*LATENCY*1000:.0f} ms difference; measured "
      f"{(t_n1-t_join)*1000:.0f} ms.")
print(f"At n=10,000 and 2ms latency that is "
      f"{10_000*0.002:.0f} seconds of pure waiting.")

# --- push the reduction down, pull the complexity up ------------------------
print("\n" + "=" * 66)
print("what belongs in SQL vs pandas")
print("=" * 66)
wide = pd.read_sql_query("SELECT * FROM orders", conn)
reduced = pd.read_sql_query("""
    SELECT u.tier, COUNT(*) AS n, SUM(o.amount) AS total
    FROM orders o JOIN users u ON u.id = o.user_id
    GROUP BY u.tier
""", conn)
print(f"pulling raw rows      : {len(wide)} rows into memory")
print(f"aggregating in SQL    : {len(reduced)} rows into memory")
print(reduced.to_string(index=False))
print("\nAt real scale the first is millions of rows and the second is five.")

conn.close()
import shutil
shutil.rmtree(work, ignore_errors=True)
```

## 8. Practical Example

A robust API client is worth building once, correctly. This one runs against a
real HTTP server started in-process, so the retry, backoff and pagination logic
is genuinely exercised rather than described.

```python {tier=A name=robust-api-client}
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
```

## 9. Common Mistakes

**Reading CSV without specifying dtypes.** Types are inferred from a sample and
can differ between files.

**Using CSV for anything you will read again.** Parquet preserves types,
compresses, and allows column subsets.

**Unpickling untrusted data.** Arbitrary code execution by design.

**Assuming JSON round-trips your types.** No dates, no int/float distinction, no
sets or tuples, and `NaN` is not valid JSON despite Python emitting it.

**Retrying without backoff.** Hammers a struggling service.

**Backing off without jitter.** Synchronises every client into a thundering
herd.

**Retrying non-idempotent operations blindly.** A timed-out payment may have
succeeded.

**Retrying 4xx errors.** The request is wrong; it will be wrong again.

**Ignoring `Retry-After`.** The server told you when to return.

**Reading only the first page.** Silent, partial data.

**String-concatenating SQL.** Injection. Use parameters.

**N+1 queries.** Use a join, or fetch in one batch and group in Python.

**Pulling raw rows to aggregate in pandas.** Push the reduction into the
database.

**Leaving connections unclosed.** Use a context manager
({{ch:py-functions-classes}}).

## 10. Connection to Previous Chapters

{{ch:py-fundamentals}} supplied generators, which is what makes the pagination
in {{sec:8-practical-example}} transparent and constant-memory.
{{ch:py-functions-classes}} supplied the decorator that implements retry and the
context manager that closes connections. {{ch:py-pandas}} supplied the DataFrames
being read and written, and its dtype discussion is why
{{sec:5-formal-explanation}} insists on explicit types. {{ch:py-numpy}} explains
why columnar storage compresses well: a column is one dtype, like an array.

Forward: {{ch:py-engineering}} tests and logs the code here.

Beyond Part II: {{ch:ds-collection}} covers ingestion at scale;
{{ch:sd-apis-auth}} builds the server side of {{sec:8-practical-example}};
{{ch:sd-fault-tolerance}} generalises retry into circuit breakers, for exactly
the reason derived in {{sec:6-mathematical-foundation}}; and
{{ch:aids-text-to-sql}} has language models writing the SQL, which makes
parameterisation a security question rather than a hygiene one.

## 11. Exercises

**Beginner**

1. Write a DataFrame with a datetime column to CSV and read it back. What is the
   dtype? Fix it.
2. Encode a dict containing a `set` to JSON. What happens, and why?
3. List three HTTP codes that should be retried and three that should not.
4. Write a parameterised SQLite query filtering by a user-supplied string.
5. Explain, in one sentence each, offset and cursor pagination.

**Intermediate**

6. Compare CSV and Parquet on a 100,000-row frame with mixed types: file size,
   write time, read time, and dtype fidelity.
7. Implement `with_retry` handling `Retry-After`, and test it against a server
   that returns 429 twice.
8. Demonstrate SQL injection against a concatenated query, then fix it.
9. Write a generator paginating a cursor-based API, and show memory does not
   grow with the number of pages.
10. Rewrite an N+1 loop as one query and measure the difference.
11. Explain why `json.dumps` emits `NaN` when that is not valid JSON, and what
    `allow_nan=False` is for.

**Advanced**

12. Using {{eq:backoff-jitter}}, compute the expected total wait for five
    attempts with $d_0 = 1$s, $b = 2$, cap 30s, with and without jitter.
13. Simulate 1,000 clients with and without jitter and plot requests per second
    against time.
14. Explain why parameterised queries are faster as well as safer, in terms of
    query-plan caching.
15. Design an idempotency scheme for a non-idempotent endpoint using client-side
    keys, and say what the server must store.
16. Using {{eq:columnar-io}}, predict the I/O saving from reading 3 of 200
    columns, and verify with Parquet.

**Implementation**

17. Build an API client with retry, backoff, pagination, a rate limiter, and
    structured logging of every request.
18. Write `read_csv_strict(path, schema)` that raises rather than coercing when
    a column does not match its declared type.
19. Build a small ETL: read JSON from an API, validate, write Parquet, load into
    SQLite, and query it back. Assert row counts at every stage.
20. Write a linter that flags f-strings inside `execute()` calls, and run it over
    your own code.

**Reasoning**

21. CSV is the worst format on nearly every axis and remains the most widely
    used. Why, and what would have to change?
22. When should aggregation happen in the database and when in pandas? Give a
    case for each where the other choice would be wrong.

## 12. Chapter Summary

File formats trade differently and each loses something. CSV has no type system,
so every read is a guess; specify dtypes explicitly. Parquet preserves types,
compresses well, and supports reading a subset of columns, which makes it the
right default for anything read more than once. Pickle executes arbitrary code
on load and must never be used for untrusted data.

JSON has six types and no dates, no int/float distinction, no binary, no sets,
and no valid representation of `NaN` — Python emits one anyway, which other
parsers reject.

Network calls can fail in a third way that local calls cannot: succeeding
without telling you. That is why idempotency determines what is safe to retry.
Retry only server-side failures and 429s; a 4xx will fail identically.

Backoff must include jitter. Without it, every client retries in unison and the
retry traffic itself prevents recovery — the thundering herd. Full jitter
spreads load over a widening window and is the difference between an outage that
recovers and one that does not.

Parameterised queries send SQL and values separately, so a value can never be
parsed as code. String concatenation is SQL injection, it is trivially
avoidable, and parameterisation is also faster because the plan can be cached.

The N+1 problem turns one query into $n+1$, costing $n$ extra round trips of
pure latency. It looks like an ordinary loop, which is why it is so common.

Push the reduction into the database and pull the complexity into Python:
filter, join and aggregate server-side so less data crosses the network, and do
Python-specific work in pandas.
