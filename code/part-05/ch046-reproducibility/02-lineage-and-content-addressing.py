# Extracted from: Chapter 46 — Reproducibility, Experiment Tracking, and Versioning
# Source: src/.../ch046-reproducibility.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Content-addressed data, a run record, and a reproduce-check that fails
loudly when something moved.
"""
import hashlib
import json
import platform
import sys

import numpy as np


# --- content addressing (section 4.4) ---------------------------------------
def content_hash(arr, *, canonical=True):
    """Hash a dataset by CONTENT, not by path.

    `canonical=True` sorts the rows first, so the hash means 'the same data'
    rather than 'the same bytes in the same order'. Which you want depends
    on whether row order is part of the dataset's identity — for a training
    table it usually is not, and a re-export that shuffles rows should not
    look like new data.
    """
    a = np.ascontiguousarray(arr, dtype=np.float64)
    if canonical:
        a = a[np.lexsort(a.T[::-1])]
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_table(n=500, seed=0, extra_rows=0):
    rs = np.random.default_rng(seed)
    X = np.column_stack([rs.normal(size=n), rs.uniform(0, 10, n)])
    if extra_rows:
        rs2 = np.random.default_rng(seed + 999)
        X = np.vstack([X, np.column_stack([rs2.normal(size=extra_rows),
                                           rs2.uniform(0, 10, extra_rows)])])
    return X


base = make_table()
print("=" * 72)
print("content addressing: what a path cannot tell you")
print("=" * 72)
cases = [
    ("the same table", make_table()),
    ("rows re-ordered by an export", base[np.random.default_rng(1)
                                          .permutation(len(base))]),
    ("42 rows appended overnight", make_table(extra_rows=42)),
    ("one value changed", np.where(np.arange(base.size).reshape(base.shape)
                                   == 7, base + 1e-9, base)),
]
print(f"{'dataset':<32} {'canonical hash':>18} {'same as base?':>15}")
h0 = content_hash(base)
print(f"{'base':<32} {h0[:16]:>18} {'-':>15}")
for label, t in cases:
    h = content_hash(t)
    print(f"{label:<32} {h[:16]:>18} {str(h == h0):>15}")

print("\nA re-export that shuffles rows is correctly recognised as the SAME")
print("data. An overnight append and a single changed value are correctly")
print("recognised as different. The path 's3://bucket/training.parquet' says")
print("nothing about any of these — and the append is the one that quietly")
print("makes last week's run irreproducible.")


# --- the run record (eq. 46.1) ----------------------------------------------
class RunRecord:
    """Capture the tuple of eq. 46.1 and verify it later."""

    def __init__(self, name):
        self.rec = {
            "name": name,
            "code": {},          # c
            "data": {},          # d
            "environment": {},   # e
            "config": {},        # phi
            "seeds": {},         # sigma
            "hardware": {},      # h
            "metrics": {},
        }

    def code(self, commit, dirty):
        self.rec["code"] = {"commit": commit, "dirty": dirty}
        return self

    def data(self, **named_arrays):
        self.rec["data"] = {k: content_hash(v)[:16]
                            for k, v in named_arrays.items()}
        return self

    def environment(self):
        self.rec["environment"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.system(),
        }
        return self

    def config(self, explicit, defaults):
        """Record the defaults you did NOT set (section 5.1) — they change
        between library versions and are the second commonest gap."""
        self.rec["config"] = {"explicit": explicit,
                              "resolved_defaults": defaults}
        return self

    def seeds(self, **s):
        self.rec["seeds"] = s
        return self

    def hardware(self, threads):
        self.rec["hardware"] = {"threads": threads}
        return self

    def metrics(self, **m):
        self.rec["metrics"] = m
        return self

    def audit(self):
        """What is missing, and what does each gap cost?"""
        gaps = []
        if not self.rec["code"].get("commit"):
            gaps.append(("code", "cannot identify the source at all"))
        elif self.rec["code"].get("dirty"):
            gaps.append(("code", "working tree was dirty: the commit is a lie"))
        if not self.rec["data"]:
            gaps.append(("data", "no content hash: cannot prove the inputs"))
        if not self.rec["environment"]:
            gaps.append(("environment", "a default may have changed under you"))
        if not self.rec["seeds"]:
            gaps.append(("seeds", "results vary run to run"))
        if not self.rec["config"].get("resolved_defaults"):
            gaps.append(("config", "unset defaults are unrecorded"))
        return gaps

    def level(self):
        """Which row of table 46.1 this record supports."""
        r = self.rec
        if not r["code"].get("commit") or not r["data"]:
            return 0
        if not r["environment"] or not r["seeds"]:
            return 1
        if not r["hardware"].get("threads"):
            return 2
        return 3


print("\n" + "=" * 72)
print("run records at three levels of care (table 46.1)")
print("=" * 72)

Xtr = make_table(400, seed=5)
ytr = (Xtr[:, 0] + 0.3 * Xtr[:, 1] > 1.5).astype(float)

sloppy = (RunRecord("sloppy")
          .code(commit="a91f3c2", dirty=True)
          .config({"max_depth": 6}, {})
          .metrics(auc=0.842))

typical = (RunRecord("typical")
           .code(commit="a91f3c2", dirty=False)
           .data(train=Xtr, labels=ytr)
           .environment()
           .config({"max_depth": 6},
                   {"min_samples_leaf": 1, "criterion": "gini"})
           .seeds(split=7, init=11)
           .metrics(auc=0.842))

careful = (RunRecord("careful")
           .code(commit="a91f3c2", dirty=False)
           .data(train=Xtr, labels=ytr)
           .environment()
           .config({"max_depth": 6},
                   {"min_samples_leaf": 1, "criterion": "gini"})
           .seeds(split=7, init=11, shuffle=13, augment=17)
           .hardware(threads=1)
           .metrics(auc=0.842))

for r in (sloppy, typical, careful):
    gaps = r.audit()
    print(f"\n{r.rec['name']}  ->  reproducibility level {r.level()}")
    if not gaps:
        print("    no gaps")
    for what, cost in gaps:
        print(f"    missing {what:<12} {cost}")

print("\nAll three report the same AUC. Only the last two can defend it, and")
print("only the last could rebuild the same parameters on demand.")

# --- the reproduce-check ----------------------------------------------------
print("\n" + "=" * 72)
print("the reproduce-check: verifying a record still resolves")
print("=" * 72)


def verify(record, *, current_data, current_env):
    """Would re-running this record today give the same thing?"""
    problems = []
    for name, h in record.rec["data"].items():
        now = content_hash(current_data[name])[:16]
        if now != h:
            problems.append(f"data '{name}' changed: recorded {h}, now {now}")
    for k, v in record.rec["environment"].items():
        if current_env.get(k) != v:
            problems.append(f"environment '{k}': recorded {v}, "
                            f"now {current_env.get(k)}")
    return problems


env_now = {"python": sys.version.split()[0], "numpy": np.__version__,
           "platform": platform.system()}

print("a) nothing has changed:")
p = verify(careful, current_data={"train": Xtr, "labels": ytr},
           current_env=env_now)
print(f"   {'reproducible' if not p else p}")

print("\nb) the training table was appended to overnight:")
Xtr2 = np.vstack([Xtr, make_table(20, seed=99)])
ytr2 = np.r_[ytr, np.zeros(20)]
p = verify(careful, current_data={"train": Xtr2, "labels": ytr2},
           current_env=env_now)
for msg in p:
    print(f"   BLOCKED: {msg}")

print("\nc) a library was upgraded:")
p = verify(careful, current_data={"train": Xtr, "labels": ytr},
           current_env={**env_now, "numpy": "9.9.9"})
for msg in p:
    print(f"   BLOCKED: {msg}")

print("\nCase (b) is the one that matters, and it is the case a path-based")
print("reference cannot detect at all: the file name is unchanged, the job")
print("runs, the numbers differ, and nobody knows why. A content hash turns")
print("that into a blocked build with a one-line explanation.")
