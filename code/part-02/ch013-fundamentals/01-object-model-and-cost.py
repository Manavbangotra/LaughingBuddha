# Extracted from: Chapter 13 — Python Fundamentals for AI Work
# Source: src/.../ch013-fundamentals.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Python's object model, its traps, and its cost model — all measured.
"""
import copy
import sys
import time

# --- eq. 13.1: assignment aliases, it does not copy -------------------------
a = [1, 2, 3]
b = a
print(f"a is b            : {a is b}")
b.append(4)
print(f"after b.append(4) : a = {a}   <- a changed too")
b = [9]
print(f"after b = [9]     : a = {a}, b = {b}   <- rebinding did not")

# Immutable objects cannot show this, because they cannot be mutated at all.
s = "hello"
t = s
t += " world"                     # creates a NEW string; s is untouched
print(f"strings           : s = {s!r}, t = {t!r}")

# --- the mutable default argument trap --------------------------------------
def collect_broken(item, into=[]):
    into.append(item)
    return into


def collect_fixed(item, into=None):
    if into is None:
        into = []
    into.append(item)
    return into


print(f"\nbroken: {collect_broken('a')} {collect_broken('b')} "
      f"{collect_broken('c')}   <- accumulates across calls")
print(f"fixed : {collect_fixed('a')} {collect_fixed('b')} "
      f"{collect_fixed('c')}   <- fresh each time")
print(f"the default is bound once, at definition: "
      f"{collect_broken.__defaults__}")

# --- eq. 13.2: three levels of copying --------------------------------------
original = [[1, 2], [3, 4]]
alias = original
shallow = copy.copy(original)          # or list(original), or original[:]
deep = copy.deepcopy(original)

original[0].append(99)                 # mutate a NESTED element

print(f"\nafter mutating a nested element of the original:")
print(f"  original : {original}")
print(f"  alias    : {alias}      <- same object")
print(f"  shallow  : {shallow}      <- new outer list, SHARED inner lists")
print(f"  deep     : {deep}            <- fully independent")
assert shallow[0] is original[0] and deep[0] is not original[0]

# --- truthiness, and where it bites -----------------------------------------
print(f"\n{'value':>10} {'truthy?':>9}")
for v in (0, 0.0, "", [], {}, None, False, "0", [0], 0.1):
    print(f"{v!r:>10} {bool(v):>9}")

DEFAULT_DISCOUNT = 0.10


def price_broken(price, discount=None):
    if not discount:                   # 0.0 is falsy!
        discount = DEFAULT_DISCOUNT
    return price * (1 - discount)


def price_fixed(price, discount=None):
    if discount is None:
        discount = DEFAULT_DISCOUNT
    return price * (1 - discount)


print(f"\ncustomer explicitly gets a 0% discount on a price of 100:")
print(f"  broken: {price_broken(100, 0.0):.2f}   <- silently applied 10%")
print(f"  fixed : {price_fixed(100, 0.0):.2f}  <- correct")

# --- iterators are consumed once --------------------------------------------
gen = (x**2 for x in range(5))
print(f"\nfirst  list(gen): {list(gen)}")
print(f"second list(gen): {list(gen)}   <- exhausted, silently empty")

# --- generators use constant memory -----------------------------------------
n = 2_000_000
eager = [x * 2 for x in range(n)]
lazy = (x * 2 for x in range(n))
print(f"\nlist of {n:,} ints : {sys.getsizeof(eager)/1e6:>8.2f} MB")
print(f"equivalent generator: {sys.getsizeof(lazy)/1e6:>8.6f} MB")
print(f"both sum to the same value: {sum(lazy) == sum(eager)}")
del eager

# --- eq. 13.3: the interpreter's per-operation cost -------------------------
print("\n" + "=" * 66)
print("why Python is slow: per-operation interpreter overhead")
print("=" * 66)

import numpy as np

print(f"{'n':>10} {'python loop':>14} {'numpy':>12} {'speedup':>10}")
for n in (100, 1_000, 100_000, 2_000_000):
    xs = list(range(n))
    arr = np.arange(n)

    t0 = time.perf_counter()
    total_py = 0
    for x in xs:
        total_py += x * 2
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    total_np = int((arr * 2).sum())
    t_np = time.perf_counter() - t0

    assert total_py == total_np
    print(f"{n:>10,} {t_py*1e3:>12.3f}ms {t_np*1e3:>10.3f}ms "
          f"{t_py/max(t_np,1e-9):>9.1f}x")

print("\nAt small n the fixed call overhead dominates and NumPy can lose.")
print("At large n the ratio approaches the interpreter overhead itself.")

# --- table 13.2: container choice changes the complexity class --------------
print("\n" + "=" * 66)
print("membership testing: list is O(n), set is O(1)")
print("=" * 66)
print(f"{'items':>9} {'list (s)':>11} {'set (s)':>10} {'ratio':>9}")
for n in (1_000, 4_000, 16_000):
    items = list(range(n)) * 2

    t0 = time.perf_counter()
    seen_list = []
    for it in items:
        if it not in seen_list:
            seen_list.append(it)
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    seen_set, out = set(), []
    for it in items:
        if it not in seen_set:
            seen_set.add(it)
            out.append(it)
    t_set = time.perf_counter() - t0

    assert seen_list == out
    print(f"{n:>9,} {t_list:>11.4f} {t_set:>10.4f} "
          f"{t_list/max(t_set,1e-9):>8.0f}x")
print("\nQuadrupling n roughly sixteen-folds the list time and quadruples")
print("the set time — the signature of O(n^2) against O(n).")
