---
id: py-pandas
number: 17
part: II
tier: focused
status: reviewed
requires: [py-numpy]
provides: [dataframe, series-term, index-alignment, missing-data,
           split-apply-combine, join-cardinality, tidy-data]
citations: [mckinney2010]
---

## 1. Learning Objectives

After this chapter you will be able to:

1. Explain what a DataFrame adds to an array, and when the extra structure is
   worth its cost.
2. Use the index deliberately, and explain how alignment prevents one class of
   bug while creating another.
3. Select rows and columns correctly with `.loc` and `.iloc`, and explain why
   chained indexing is unreliable.
4. Handle missing data, distinguishing the three different things `NaN` may
   mean.
5. Apply split-apply-combine to answer aggregation questions.
6. Join tables safely, validating cardinality rather than assuming it.
7. Reshape between wide and long form and say which each tool wants.
8. Recognise when pandas is the wrong tool.

## 2. Why This Matters

Most data does not arrive as a homogeneous numeric grid. It arrives as a table
with a date column, a categorical column, three numeric columns, and missing
values scattered through it. An array cannot represent that; a DataFrame can
{{cite:mckinney2010}}.

The realistic distribution of effort on a machine-learning project is that the
modelling is a small fraction of the work and the data preparation is most of
it. That preparation happens in pandas, and it is also where the errors that
matter most are introduced — because they are silent. A model that crashes gets
fixed. A model trained on a table whose row count doubled during a join gets
deployed.

Three specific hazards recur, and this chapter is organised around them.
**Alignment** means pandas matches on index labels, not position, which prevents
mismatched joins and produces surprising `NaN`s. **Chained indexing** may return
a view or a copy depending on facts about memory layout you cannot see, so an
assignment through it may or may not take effect. **Join cardinality** is
assumed rather than checked, and a many-to-many join that you believed was
one-to-one silently multiplies your rows and corrupts every aggregate computed
downstream.

## 3. Prerequisites

{{ch:py-numpy}} for arrays, dtypes, views and vectorisation — a DataFrame column
is backed by an array and inherits all of it. {{ch:py-fundamentals}} for
mutability and truthiness, which resurface here as `SettingWithCopyWarning` and
as the ambiguity of `NaN`.

## 4. Intuitive Explanation

### 4.1 A DataFrame is a dictionary of aligned arrays

A {{term:dataframe}} is a set of columns, each a {{term:series-term}} — a
one-dimensional array with a label per element — all sharing one index.

```text
        index    age(int64)   city(object)   score(float64)
          0          25          london          0.82
          1          40          leeds            NaN
          2          35          london          0.91
                     ▲              ▲               ▲
              one array       one array       one array
                    all sharing the same index
```

Two differences from a 2-D array follow, and both matter:

**Columns have independent types.** An array is one dtype throughout; a
DataFrame's `age` can be `int64` while `city` is a string.

**Rows and columns have labels.** You select `df["age"]` rather than
`arr[:, 0]`, which is both clearer and robust to column reordering.

The cost is overhead. For purely numeric work on a fixed-size matrix, an array
is faster and leaner. Use a DataFrame when the labels and mixed types are
carrying real information.

### 4.2 The index is not a row number

The most common misconception is that the index is a position. It is a
**label**, and pandas uses labels to align data before combining it.

```python {tier=C name=alignment-intro}
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["z", "y", "x"])
a + b        # x:31, y:22, z:13  — matched by LABEL, not position
```

This is genuinely valuable: it makes it impossible to accidentally add January's
revenue to February's costs because the rows happened to be in a different
order. It is also the source of the most common confusion in pandas, because
when the indexes do not fully overlap the result contains `NaN` for every label
present in one and absent in the other.

### 4.3 Split-apply-combine

Most aggregation questions have one shape: split the data into groups, compute
something per group, combine the results.

```python {tier=C name=groupby-intro}
df.groupby("city")["score"].mean()
```

Split by city, take the mean score in each, return one row per city. Once you
see {{term:split-apply-combine}}, a large fraction of analysis questions become
one line.

### 4.4 Wide and long

The same data can be laid out two ways:

```text
wide:                          long (tidy):
  city    2023   2024            city    year   value
  london   10     12             london  2023     10
  leeds     8      9             london  2024     12
                                 leeds   2023      8
                                 leeds   2024      9
```

Wide is compact and readable. Long — {{term:tidy-data}} — has one row per
observation and one column per variable, which is what most tools expect:
plotting libraries, statistical models, and database tables. Converting between
them is `pivot` and `melt`, and knowing which form a tool wants saves a great
deal of confusion.

## 5. Formal Explanation

### 5.1 Selection: `.loc`, `.iloc`, and nothing else

There are exactly two accessors worth using.

- **`.loc[rows, cols]`** — by **label**. Slices are **inclusive** of the
  endpoint.
- **`.iloc[rows, cols]`** — by **integer position**. Slices are **exclusive**,
  like Python.

```python {tier=C name=selection}
df.loc[3, "age"]              # label 3, column "age"
df.loc[1:3]                   # labels 1, 2 AND 3 — inclusive
df.iloc[1:3]                  # positions 1, 2 — exclusive
df.loc[df["age"] > 30, ["city", "score"]]     # boolean mask plus columns
```

That `.loc` slices are inclusive while `.iloc` slices are exclusive is a genuine
inconsistency, and it is deliberate: a label slice `"2024-01":"2024-03"` would
be useless if it excluded March.

> WARNING: **Never chain indexing for assignment.** `df[df.a > 0]["b"] = 1`
> evaluates as two operations: a selection producing a temporary that may be a
> view or a copy, then an assignment into that temporary. If it was a copy, the
> assignment goes nowhere and the original is unchanged. Whether it is a view
> depends on memory layout you cannot inspect, so the same line can work in
> testing and fail in production. Use a single `.loc`:
> `df.loc[df.a > 0, "b"] = 1`. This is what `SettingWithCopyWarning` is warning
> about, and it is the aliasing problem of {{ch:py-fundamentals}} wearing a
> different hat.

### 5.2 Missing data

Pandas represents {{term:missing-data}} as `NaN` for floats, `NaT` for
datetimes, and `None` or `pd.NA` for object and nullable types.

Two properties surprise people. **`NaN` is not equal to itself**, so
`df["x"] == np.nan` never matches — use `.isna()`. And **integer columns cannot
hold `NaN`** in the classic dtypes, so an int column containing a missing value
is silently promoted to `float64`, which is why an ID column sometimes arrives
as `1.0, 2.0, 3.0`.

The nullable dtypes (`Int64`, `boolean`, `string` — note the capitals) fix this
by carrying a separate mask. {{maturity:MATURE}}

The important decision is not mechanical but semantic:

{#tbl:missing-strategies caption="Strategies for missing data. The right choice depends on why the value is missing, which is a question about the data-generating process rather than about pandas."}

| Strategy | When | Risk |
|---|---|---|
| drop rows | few, and missing at random | bias if not random; data loss |
| drop column | mostly missing | discards a signal |
| fill constant | absence has a meaning | invents data |
| fill mean/median | numeric, missing at random | shrinks variance, distorts correlations |
| forward fill | time series where last value persists | leaks the future if misapplied |
| indicator column | missingness itself is informative | extra dimension |
| model it | the value matters and is predictable | complexity, leakage risk |

> IMPORTANT: The single most useful question is *why* the value is missing. A
> sensor that failed, a question a user declined to answer, and a field that
> does not apply to this record are three different situations demanding three
> different treatments — and an indicator column is often the honest answer,
> because the fact of missingness frequently carries signal
> ({{ch:ds-cleaning}}).

### 5.3 Groupby

```python {tier=C name=groupby-forms}
df.groupby("city")["score"].mean()                      # one aggregate
df.groupby("city").agg(n=("score", "size"),             # named aggregates
                       mean=("score", "mean"),
                       worst=("score", "min"))
df.groupby("city")["score"].transform("mean")           # broadcast back
df.groupby("city").filter(lambda g: len(g) >= 10)       # keep whole groups
```

The distinction worth learning is `agg` versus `transform`. **`agg` reduces**:
one row per group. **`transform` broadcasts**: the same shape as the input, with
each row carrying its group's value. `transform` is what you want for
group-relative features — "this score minus the mean for this city" — which is a
common and useful feature-engineering pattern ({{ch:ds-feature-eng}}).

By default `groupby` drops rows whose key is `NaN`. If missing keys are
meaningful, pass `dropna=False` — silently losing rows here is a real and common
error.

### 5.4 Joins

```python {tier=C name=merge}
pd.merge(left, right, on="user_id", how="inner", validate="one_to_one")
```

Join types are the usual four: `inner` keeps matching keys only, `left` keeps all
left rows, `right` the mirror, `outer` keeps everything.

The parameter that matters most is the one people omit. **`validate=`** checks
{{term:join-cardinality}} and raises if the assumption is wrong:
`"one_to_one"`, `"one_to_many"`, `"many_to_one"`, `"many_to_many"`.

> WARNING: A many-to-many join that you believed was one-to-one does not fail.
> It produces the Cartesian product within each key group, multiplying your row
> count, and every subsequent `sum` and `mean` is then computed over duplicated
> rows. The result is wrong by a factor that varies per group, which is
> precisely the kind of error that survives a sanity check. Always pass
> `validate=`, and always check `len(df)` before and after.

The second habit worth forming is `indicator=True`, which adds a column saying
whether each row matched on the left, the right, or both. Unmatched keys are
usually a data problem worth knowing about rather than silently dropping.

### 5.5 Reshaping

- `melt` — wide to long. Columns become rows.
- `pivot` — long to wide. Requires unique index/column pairs.
- `pivot_table` — like `pivot` but aggregates duplicates.
- `stack` / `unstack` — move a level between index and columns.

### 5.6 When pandas is the wrong tool

Pandas is single-threaded, holds everything in memory, and typically needs
several times a dataset's size in working memory. Rough guidance:

{#tbl:when-not-pandas caption="Choosing a tabular tool by scale. The boundaries are approximate and hardware-dependent."}

| Data size | Tool |
|---|---|
| < 1 GB | pandas |
| 1-50 GB | Polars, DuckDB, or pandas with chunking and careful dtypes |
| > 50 GB | DuckDB, Spark, or a database |
| Streaming | generators ({{ch:py-fundamentals}}), or a stream processor |

Before reaching for a bigger tool, try `category` dtype for repeated strings and
downcast numerics. It is common to cut memory by 5-10× with two lines, and
{{sec:7-implementation}} measures it.

## 6. Mathematical Foundation

### 6.1 What alignment actually computes

For two Series with indexes $I_1$ and $I_2$, a binary operation produces a
result indexed by the **union**:

$$
I_{\text{result}} = I_1 \cup I_2
$$ (eq:alignment-union)

with

$$
r_k = \begin{cases}
f(a_k, b_k) & k \in I_1 \cap I_2 \\
\texttt{NaN} & \text{otherwise}
\end{cases}
$$ (eq:alignment-values)

So the number of non-null results is $\lvert I_1 \cap I_2 \rvert$, and the
number of rows is $\lvert I_1 \cup I_2 \rvert$. When two Series you expected to
match produce mostly `NaN`, {{eq:alignment-values}} is why: the intersection is
smaller than you thought.

Duplicate labels make this worse. If a label appears $p$ times on the left and
$q$ times on the right, alignment produces $p \times q$ rows for it — the same
multiplication that makes unvalidated joins dangerous.

### 6.2 Join cardinality, quantified

For a join on key $k$ with multiplicity $m_L(k)$ on the left and $m_R(k)$ on the
right, the output row count is

$$
N_{\text{out}} = \sum_{k \in K} m_L(k)\, m_R(k)
$$ (eq:join-rows)

The cases:

- **one-to-one**: every $m = 1$, so $N_{\text{out}} = \lvert K \rvert$.
- **one-to-many**: $m_L = 1$, so $N_{\text{out}} = \sum_k m_R(k) = N_R$.
- **many-to-many**: the products dominate, and $N_{\text{out}}$ can far exceed
  either input.

A worked case: 1,000 users joined to 1,000 orders on `user_id`, where 10 users
appear twice on each side. Then $N_{\text{out}} = 990 \cdot 1 + 10 \cdot 4 =
1030$ — thirty extra rows, a 3% inflation. Every subsequent mean is now weighted
wrongly, by an amount too small to notice and large enough to matter.

{{eq:join-rows}} is why `validate=` exists and why row counts should be checked
around every join.

### 6.3 Memory, and why `category` helps so much

A string column in pandas is by default an object array: an array of **pointers
to Python string objects**, exactly the boxed layout {{ch:py-numpy}} explained
is slow. Each pointer is 8 bytes and each string object carries roughly 50 bytes
of overhead plus its characters.

The `category` dtype stores instead an integer code per row plus one copy of
each distinct value:

$$
\text{bytes}_{\text{object}} \approx n \cdot (8 + \overline{s} + 49)
\qquad
\text{bytes}_{\text{category}} \approx n \cdot b + u \cdot (\overline{s} + 49)
$$ (eq:category-memory)

where $n$ is the row count, $u$ the number of distinct values, $\overline{s}$
the mean string length, and $b$ the code width — 1 byte when $u < 256$.

The saving is dramatic when $u \ll n$. For a million rows with ten distinct
city names averaging 8 characters: about 65 MB as objects, about 1 MB as a
category. {{sec:7-implementation}} measures it.

The same reasoning covers numerics. A column of small integers stored as `int64`
uses eight times what `int8` would.

## 7. Implementation

```python {tier=A name=pandas-core}
"""Pandas' core mechanics and its three sharpest edges, all demonstrated.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 100)
rng = np.random.default_rng(0)

# --- a DataFrame is aligned columns of independent dtypes -------------------
df = pd.DataFrame({
    "user": ["u1", "u2", "u3", "u4", "u5"],
    "age": [25, 40, 35, 50, np.nan],
    "city": ["london", "leeds", "london", "bristol", "leeds"],
    "score": [0.82, np.nan, 0.91, 0.45, 0.77],
})
print(df)
print(f"\ndtypes:\n{df.dtypes}")
print(f"\nage became float64 because it holds a NaN — classic int columns")
print("cannot represent missing values.")
print(f"with the nullable dtype: "
      f"{df['age'].astype('Int64').dtype}  <- capital I, keeps integers")

# --- eq. 17.1: alignment is by LABEL, not position --------------------------
print("\n" + "=" * 66)
print("index alignment")
print("=" * 66)
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["z", "y", "x"])
print(f"a:\n{a.to_dict()}\nb:\n{b.to_dict()}")
print(f"a + b (matched by label):\n{(a + b).to_dict()}")
print("  x:31 = 1+30, not 1+10. Position was ignored.")

c = pd.Series([1, 2, 3], index=["x", "y", "w"])
print(f"\nwith non-overlapping indexes, a + c:\n{(a + c).to_dict()}")
print(f"  union has {len((a + c))} labels, intersection has "
      f"{len(set(a.index) & set(c.index))} non-null (eq. 17.2)")

# --- chained indexing: the assignment that may go nowhere -------------------
print("\n" + "=" * 66)
print("chained indexing")
print("=" * 66)
work = df.copy()
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        work[work["age"] > 30]["score"] = 0.0        # chained: two operations
    except Exception as exc:
        print(f"  raised: {type(exc).__name__}")
changed = work.loc[work["age"] > 30, "score"].tolist()
print(f"after chained assignment, scores are {changed}")
print("  The write went into a temporary. The original is untouched.")

work.loc[work["age"] > 30, "score"] = 0.0            # single .loc: correct
print(f"after .loc assignment,     scores are "
      f"{work.loc[work['age'] > 30, 'score'].tolist()}")

# --- missing data ------------------------------------------------------------
print("\n" + "=" * 66)
print("missing data")
print("=" * 66)
print(f"NaN == NaN            : {np.nan == np.nan}   <- never use == for NaN")
print(f"df['score'].isna()    : {df['score'].isna().tolist()}")
print(f"nulls per column      : {df.isna().sum().to_dict()}")

filled_mean = df["score"].fillna(df["score"].mean())
print(f"\noriginal score std    : {df['score'].std():.4f}")
print(f"after mean imputation : {filled_mean.std():.4f}   <- variance shrank")
print("Mean imputation always understates variance and distorts correlations.")

# The indicator-column approach keeps the information that a value was absent.
augmented = df.assign(score_missing=df["score"].isna().astype(int),
                      score=df["score"].fillna(df["score"].median()))
print(f"\nindicator approach keeps the signal:\n"
      f"{augmented[['user', 'score', 'score_missing']].to_string(index=False)}")

# --- split-apply-combine ------------------------------------------------------
print("\n" + "=" * 66)
print("split-apply-combine: agg reduces, transform broadcasts")
print("=" * 66)
agg = df.groupby("city").agg(n=("user", "size"),
                             mean_score=("score", "mean"),
                             max_age=("age", "max"))
print(f"agg (one row per group):\n{agg}")

df2 = df.assign(city_mean=df.groupby("city")["score"].transform("mean"))
df2["vs_city"] = df2["score"] - df2["city_mean"]
print(f"\ntransform (same shape as input — a group-relative feature):")
print(df2[["user", "city", "score", "city_mean", "vs_city"]].to_string(index=False))

# groupby drops NaN keys unless told otherwise.
with_nan = df.assign(city=df["city"].where(df.index != 0))
print(f"\nrows: {len(with_nan)}, "
      f"grouped (default): {with_nan.groupby('city').size().sum()}, "
      f"dropna=False: {with_nan.groupby('city', dropna=False).size().sum()}")
print("  A row silently vanished. Pass dropna=False when a missing key means")
print("  something.")

# --- eq. 17.3: join cardinality ---------------------------------------------
print("\n" + "=" * 66)
print("join cardinality: the silent row explosion")
print("=" * 66)
users = pd.DataFrame({"user_id": [1, 2, 3], "name": ["a", "b", "c"]})
orders = pd.DataFrame({"user_id": [1, 1, 2, 2, 2, 3],
                       "amount": [10, 20, 30, 40, 50, 60]})

one_to_many = users.merge(orders, on="user_id", how="inner",
                          validate="one_to_many")
print(f"users {len(users)} x orders {len(orders)} -> "
      f"{len(one_to_many)} rows (one-to-many, as expected)")

# Now duplicate a user — a realistic data-quality problem.
dirty = pd.concat([users, users.iloc[[1]]], ignore_index=True)
bad = dirty.merge(orders, on="user_id", how="inner")
print(f"\nwith user 2 duplicated in the LEFT table:")
print(f"  rows: {len(bad)} (was {len(one_to_many)}) — eq. 17.3 predicts "
      f"{1*2 + 2*3 + 1*1}")
print(f"  total amount: {bad['amount'].sum()} vs true "
      f"{orders['amount'].sum()}   <- inflated")

try:
    dirty.merge(orders, on="user_id", how="inner", validate="one_to_many")
except pd.errors.MergeError as exc:
    print(f"\nvalidate= catches it: MergeError: {str(exc)[:60]}")

# indicator= surfaces unmatched keys instead of silently dropping them.
partial = pd.DataFrame({"user_id": [1, 99], "tier": ["gold", "silver"]})
checked = users.merge(partial, on="user_id", how="outer", indicator=True)
print(f"\nindicator=True shows what matched:\n"
      f"{checked['_merge'].value_counts().to_dict()}")

# --- reshaping ----------------------------------------------------------------
print("\n" + "=" * 66)
print("wide <-> long")
print("=" * 66)
wide = pd.DataFrame({"city": ["london", "leeds"],
                     "2023": [10, 8], "2024": [12, 9]})
long = wide.melt(id_vars="city", var_name="year", value_name="value")
print(f"wide:\n{wide.to_string(index=False)}")
print(f"\nlong (tidy):\n{long.to_string(index=False)}")
back = long.pivot(index="city", columns="year", values="value").reset_index()
back.columns.name = None
print(f"\nafter the round trip:\n{back.to_string(index=False)}")

# pivot SORTS the index, so the rows come back in a different order. The data
# is identical; a naive .equals() comparison would report otherwise.
naive = back.set_index("city").equals(wide.set_index("city"))
sorted_cmp = (back.set_index("city").sort_index()
              .equals(wide.set_index("city").sort_index()))
print(f"naive .equals()      : {naive}   <- misleading: row order differs")
print(f"compared after sorting: {sorted_cmp}   <- the data is preserved")
assert sorted_cmp
print("pivot returns a sorted index. Comparing frames without accounting for")
print("order is a common way to convince yourself a correct transform is broken.")

# --- eq. 17.4: memory, and what category buys -------------------------------
print("\n" + "=" * 66)
print("memory: dtypes are not a detail")
print("=" * 66)
n = 500_000
big = pd.DataFrame({
    "city": rng.choice(["london", "leeds", "bristol", "cardiff",
                        "manchester"], n),
    "count": rng.integers(0, 100, n),
    "ratio": rng.random(n),
})
before = big.memory_usage(deep=True).sum()

tuned = big.astype({"city": "category", "count": "int8", "ratio": "float32"})
after = tuned.memory_usage(deep=True).sum()

print(f"{'column':<10} {'original':>14} {'optimised':>14}")
for col in big.columns:
    print(f"{col:<10} {big[col].memory_usage(deep=True)/1e6:>12.2f} MB "
          f"{tuned[col].memory_usage(deep=True)/1e6:>12.2f} MB")
print(f"{'total':<10} {before/1e6:>12.2f} MB {after/1e6:>12.2f} MB "
      f"  ({before/after:.1f}x smaller)")
print("\nThe city column dominates: object dtype stores a pointer per row to a")
print("boxed Python string. category stores one small integer per row plus")
print("five strings total (eq. 17.4).")

# Values are unchanged; only the representation differs.
assert (tuned["city"].astype(str) == big["city"]).all()
assert (tuned["count"] == big["count"]).all()
print("...and the values are identical.")
```

## 8. Practical Example

A realistic cleaning-and-joining pipeline, written to make the dangerous steps
loud rather than silent.

```python {tier=A name=defensive-pipeline}
"""Cleaning and joining a messy dataset, defensively.

Every step that could silently change the row count or the semantics asserts
what it expects. This is the difference between a pipeline you can trust and a
notebook that produced a number once.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# --- deliberately messy inputs ----------------------------------------------
n = 800
events = pd.DataFrame({
    "user_id": rng.integers(1, 200, n),
    "ts": pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 60, n), "D"),
    "amount": np.where(rng.random(n) < 0.06, np.nan, rng.gamma(2, 25, n)),
    "channel": rng.choice(["web", "app", "Web", " app ", None], n,
                          p=[0.4, 0.35, 0.1, 0.1, 0.05]),
})
users = pd.DataFrame({
    "user_id": np.arange(1, 200),
    "segment": rng.choice(["free", "pro", "enterprise"], 199),
})
# A realistic data-quality defect: two duplicated user records.
users = pd.concat([users, users.iloc[[5, 20]]], ignore_index=True)

print(f"events: {len(events)} rows | users: {len(users)} rows")


def check(df, expected_rows=None, name=""):
    """Assert what we believe, loudly."""
    if expected_rows is not None and len(df) != expected_rows:
        raise AssertionError(
            f"{name}: expected {expected_rows} rows, got {len(df)}")
    return df


# --- step 1: normalise the categorical column -------------------------------
before = len(events)
events["channel"] = (events["channel"].str.strip().str.lower())
print(f"\nchannel values after normalising: "
      f"{sorted(events['channel'].dropna().unique())}")
print(f"  'Web' and ' app ' merged into existing categories; "
      f"{events['channel'].isna().sum()} genuinely missing")
check(events, before, "normalise")

# --- step 2: missing amounts — decide deliberately, and record the decision -
missing_amount = events["amount"].isna()
print(f"\n{missing_amount.sum()} rows have a missing amount "
      f"({missing_amount.mean():.1%})")

# Is missingness related to anything? If so, dropping biases the result.
by_channel = events.assign(m=missing_amount).groupby(
    "channel", dropna=False)["m"].mean()
print(f"missing rate by channel:\n{by_channel.round(3).to_dict()}")
print("  Roughly uniform, so dropping is defensible here — but we keep an")
print("  indicator rather than silently discarding the information.")

events["amount_missing"] = missing_amount.astype("int8")
events["amount"] = events["amount"].fillna(events["amount"].median())

# --- step 3: the join, validated --------------------------------------------
print(f"\njoining events to users...")
try:
    events.merge(users, on="user_id", how="left", validate="many_to_one")
except pd.errors.MergeError as exc:
    print(f"  BLOCKED by validate=: {str(exc)[:70]}")
    print("  The users table has duplicate user_id values. Without validate=,")
    print("  this join would have silently inflated the event count.")

dupes = users["user_id"].duplicated().sum()
print(f"  found {dupes} duplicate user rows; de-duplicating")
users_clean = users.drop_duplicates(subset="user_id", keep="first")

joined = events.merge(users_clean, on="user_id", how="left",
                      validate="many_to_one", indicator=True)
check(joined, len(events), "join")
print(f"  joined cleanly: {len(joined)} rows, unchanged")
print(f"  match status: {joined['_merge'].value_counts().to_dict()}")
joined = joined.drop(columns="_merge")

# --- step 4: aggregate --------------------------------------------------------
summary = (joined
           .groupby(["segment", "channel"], dropna=False, observed=True)
           .agg(events=("user_id", "size"),
                users=("user_id", "nunique"),
                revenue=("amount", "sum"),
                mean_amount=("amount", "mean"))
           .round(2)
           .sort_values("revenue", ascending=False))

print(f"\n{summary.head(8)}")

# The aggregate must reconcile with the source — a cheap, powerful check.
assert summary["events"].sum() == len(joined)
assert np.isclose(summary["revenue"].sum(), joined["amount"].sum())
print(f"\nreconciliation: event count and revenue both tie back to the source")

# --- step 5: a group-relative feature, via transform ------------------------
joined["segment_mean"] = joined.groupby("segment")["amount"].transform("mean")
joined["amount_vs_segment"] = joined["amount"] - joined["segment_mean"]
print(f"\ngroup-relative feature (transform keeps the original shape):")
print(joined[["user_id", "segment", "amount", "segment_mean",
              "amount_vs_segment"]].head(4).round(2).to_string(index=False))

# --- step 6: shrink before writing ------------------------------------------
final = joined.astype({"segment": "category", "channel": "category",
                       "amount": "float32", "segment_mean": "float32",
                       "amount_vs_segment": "float32"})
print(f"\nmemory: {joined.memory_usage(deep=True).sum()/1e6:.2f} MB -> "
      f"{final.memory_usage(deep=True).sum()/1e6:.2f} MB")
```

## 9. Common Mistakes

**Chained indexing for assignment.** May write into a temporary. Use one
`.loc`.

**Assuming the index is a row number.** It is a label, and alignment uses it.

**Ignoring `SettingWithCopyWarning`.** It is telling you the write may not have
happened.

**Joining without `validate=`.** Silent row multiplication, corrupting every
downstream aggregate.

**Not checking row counts around joins.** One line, catches the previous
mistake.

**Comparing to `NaN` with `==`.** Never matches. Use `.isna()`.

**Mean-imputing without thinking.** Shrinks variance and distorts correlations.
Consider an indicator column.

**Forgetting `dropna=False` in `groupby`.** Rows with missing keys vanish
silently.

**Using `apply` where a vectorised operation exists.** `apply` runs a Python
function per row and forfeits every advantage of {{ch:py-numpy}} — often
100× slower.

**`inplace=True`.** It rarely saves memory, returns `None` so it cannot be
chained, and is being phased out. Assign the result instead.

**Leaving string columns as `object`.** Frequently the dominant memory cost;
`category` is usually a large win.

**Iterating with `iterrows`.** Slow, and it loses dtypes by boxing each row into
a Series. Use vectorised operations, or `itertuples` if you truly must loop.

## 10. Connection to Previous Chapters

{{ch:py-numpy}} supplied the arrays that back every column, the dtypes that
determine memory, and the view semantics that reappear here as
`SettingWithCopyWarning`. {{ch:py-fundamentals}} supplied the mutability and
truthiness that make `NaN` handling subtle.

{{ch:math-covariance}} is relevant to {{sec:5-formal-explanation}}: mean
imputation shrinks variance and therefore attenuates correlations, which is a
quantitative claim from that chapter, not a vague warning.

Forward within Part II: {{ch:py-visualization}} plots DataFrames directly;
{{ch:py-io-apis-sql}} reads them from files and databases, where `groupby` and
`merge` have exact SQL equivalents.

Beyond Part II: {{part:3}} is this chapter applied — {{ch:ds-cleaning}} on
missing data, {{ch:ds-feature-eng}} on the `transform` pattern, and
{{ch:ds-leakage}} on why fitting an imputer before splitting leaks.
{{ch:mle-pipelines}} formalises the defensive structure of
{{sec:8-practical-example}}.

{{cite:mckinney2010}} introduced the DataFrame.

## 11. Exercises

**Beginner**

1. Build a DataFrame from a dict of three columns with different dtypes and
   print `.dtypes`.
2. Select rows where a column exceeds a threshold, keeping only two columns.
3. Explain the difference between `df.loc[1:3]` and `df.iloc[1:3]`.
4. Count missing values per column.
5. Group by a categorical column and compute the mean of a numeric one.

**Intermediate**

6. Construct two Series with partially overlapping indexes and predict the
   result of adding them before checking.
7. Demonstrate chained-assignment failing, then fix it with `.loc`.
8. Merge two tables where the right has duplicate keys. Predict the row count
   with {{eq:join-rows}}, then verify.
9. Compute, for each row, the difference between its value and its group's mean,
   using `transform`.
10. Convert a wide table to long and back, and verify the round trip.
11. Reduce a DataFrame's memory by at least 4× using dtype changes alone, and
    prove the values are unchanged.

**Advanced**

12. Explain why chained indexing sometimes works and sometimes does not, in
    terms of views and copies from {{ch:py-numpy}}.
13. Using {{eq:category-memory}}, predict the memory saving for a column of
    $10^{6}$ rows with 50 distinct values averaging 12 characters. Verify.
14. Show quantitatively that mean imputation attenuates the correlation between
    two variables, and derive the attenuation factor as a function of the
    missing fraction.
15. Implement a `safe_merge` wrapper asserting expected cardinality and row
    count, and reporting unmatched keys on both sides.
16. Explain why `apply` on rows is slow, and find a case where it is
    nonetheless the right choice.

**Implementation**

17. Take a public CSV with real problems and write a cleaning pipeline with an
    assertion after every step.
18. Write a `profile_dataframe` function reporting per-column dtype, null
    fraction, cardinality, memory and a suggested optimised dtype.
19. Build a group-relative feature set — value minus group mean, divided by
    group std, plus group rank — using only `transform`.
20. Compare `merge` against an equivalent SQL join on the same data in SQLite
    ({{ch:py-io-apis-sql}}), for both correctness and speed.

**Reasoning**

21. Alignment prevents mismatched joins and produces surprising `NaN`s. Was
    making it the default the right call?
22. You inherit a notebook whose reported revenue is 3% higher than the finance
    team's. Given this chapter, what do you check first, and why?

## 12. Chapter Summary

A DataFrame is a set of equal-length, independently-typed columns sharing one
labelled index. It adds heterogeneous types and labels to what an array offers,
at the cost of overhead — worth paying when the labels and types carry
information, not otherwise.

The index is a label, not a position, and pandas aligns on it before combining.
The result of a binary operation is indexed by the union of the operands'
indexes, with `NaN` wherever a label is missing from either, which is why
partially-overlapping Series produce surprising nulls.

Use `.loc` for labels and `.iloc` for positions, and never chain indexing for
assignment: the intermediate may be a view or a copy depending on memory layout
you cannot see, so the write may silently go nowhere.

Missing data is not one thing. `NaN` is not equal to itself, classic integer
columns cannot hold it, and the right treatment depends on *why* the value is
absent. Mean imputation is the common default and it shrinks variance and
attenuates correlations; an indicator column is frequently the more honest
choice.

Split-apply-combine answers most aggregation questions. `agg` reduces to one row
per group; `transform` broadcasts back to the original shape, which is what
group-relative features need. `groupby` silently drops rows with missing keys
unless told otherwise.

Join output size is $\sum_k m_L(k)m_R(k)$, so an unvalidated many-to-many join
multiplies rows and corrupts every downstream aggregate without failing. Pass
`validate=`, use `indicator=True`, and check row counts on both sides.

Dtypes are not a detail. Object-dtype string columns store a boxed Python
string per row and frequently dominate memory; `category` replaces that with one
small integer per row and often cuts total memory severalfold.
