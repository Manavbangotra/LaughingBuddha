# Extracted from: Chapter 19 — Files, JSON, APIs, and SQL from Python
# Source: src/.../ch019-io-apis-sql.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

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
