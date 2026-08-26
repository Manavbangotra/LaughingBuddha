# -*- coding: utf-8 -*-
# Extracted from: Chapter 20 — Debugging, Logging, Testing, Async, and Performance
# Source: src/.../ch020-engineering.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Threads, processes, async and NumPy — measured against eqs. 20.4-20.6.

Includes the case people get wrong: multiprocessing being slower than
sequential when per-task work is small.
"""
import asyncio
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

N_TASKS = 8
IO_SECONDS = 0.12


# --- an I/O-bound task: mostly waiting ---------------------------------------
def io_task(_):
    time.sleep(IO_SECONDS)          # sleep releases the GIL
    return 1


# --- a CPU-bound task: pure Python arithmetic --------------------------------
def cpu_task(n):
    total = 0.0
    for i in range(1, n):
        total += math.sqrt(i) * math.sin(i)
    return total


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


print("=" * 70)
print(f"I/O-bound: {N_TASKS} tasks x {IO_SECONDS}s of waiting")
print("=" * 70)

seq = timeit(lambda: [io_task(i) for i in range(N_TASKS)])
thr = timeit(lambda: list(ThreadPoolExecutor(N_TASKS).map(io_task,
                                                          range(N_TASKS))))


async def async_task():
    await asyncio.sleep(IO_SECONDS)
    return 1


async def async_all():
    return await asyncio.gather(*(async_task() for _ in range(N_TASKS)))


asy = timeit(lambda: asyncio.run(async_all()))

print(f"{'sequential':<16} {seq:>7.3f}s   (eq. 20.4: n*t_io = "
      f"{N_TASKS*IO_SECONDS:.2f}s)")
print(f"{'threads':<16} {thr:>7.3f}s   {seq/thr:>5.1f}x faster")
print(f"{'asyncio':<16} {asy:>7.3f}s   {seq/asy:>5.1f}x faster")
print("\nBoth overlap the waiting. The GIL is irrelevant here because a")
print("sleeping thread is not executing bytecode (eq. 20.5).")

print("\n" + "=" * 70)
print("CPU-bound pure Python")
print("=" * 70)

WORK = 400_000
seq_cpu = timeit(lambda: [cpu_task(WORK) for _ in range(N_TASKS)])
thr_cpu = timeit(lambda: list(ThreadPoolExecutor(N_TASKS).map(
    cpu_task, [WORK] * N_TASKS)))

n_cores = mp.cpu_count()
if sys.platform != "win32":
    proc_cpu = timeit(lambda: list(ProcessPoolExecutor(
        min(n_cores, N_TASKS)).map(cpu_task, [WORK] * N_TASKS)))
else:
    proc_cpu = float("nan")

print(f"{'sequential':<16} {seq_cpu:>7.3f}s")
print(f"{'threads':<16} {thr_cpu:>7.3f}s   {seq_cpu/thr_cpu:>5.2f}x "
      f"<- no speedup: the GIL serialises bytecode")
print(f"{'processes':<16} {proc_cpu:>7.3f}s   {seq_cpu/proc_cpu:>5.2f}x "
      f"<- real parallelism across {n_cores} cores")

print("\n" + "=" * 70)
print("the case people get wrong: small tasks in processes")
print("=" * 70)

TINY = 300
tiny_seq = timeit(lambda: [cpu_task(TINY) for _ in range(2000)])
if sys.platform != "win32":
    tiny_proc = timeit(lambda: list(ProcessPoolExecutor(n_cores).map(
        cpu_task, [TINY] * 2000, chunksize=1)))
else:
    tiny_proc = float("nan")

print(f"2000 tiny tasks, sequential : {tiny_seq:>7.3f}s")
print(f"2000 tiny tasks, processes  : {tiny_proc:>7.3f}s   "
      f"{'SLOWER' if tiny_proc > tiny_seq else 'faster'} "
      f"({tiny_proc/tiny_seq:.1f}x)")
print("\nEach task must be pickled, sent to a worker, and the result sent")
print("back. When per-task work is smaller than that overhead, eq. 20.6's")
print("n*c_ipc term dominates and parallelism costs more than it saves.")
print("Raising chunksize amortises it — but the real fix is bigger tasks.")

print("\n" + "=" * 70)
print("the option people forget: don't use Python for the loop")
print("=" * 70)


def cpu_task_numpy(n):
    i = np.arange(1, n, dtype=np.float64)
    return float((np.sqrt(i) * np.sin(i)).sum())


t_np = timeit(lambda: [cpu_task_numpy(WORK) for _ in range(N_TASKS)])
same = math.isclose(cpu_task(WORK), cpu_task_numpy(WORK), rel_tol=1e-9)
print(f"{'numpy, 1 thread':<18} {t_np:>7.3f}s   {seq_cpu/t_np:>5.1f}x vs "
      f"sequential Python")
print(f"{'(processes were)':<18} {proc_cpu:>7.3f}s   {seq_cpu/proc_cpu:>5.1f}x")
print(f"identical result: {same}")
print("\nVectorising beat multiprocessing on one core, with no IPC, no")
print("pickling and no pool to manage. For numerical work the first question")
print("is not 'how do I parallelise this loop' but 'why is there a loop'.")

print("\n" + "=" * 70)
print("decision rule")
print("=" * 70)
print("  waiting on I/O            -> threads or asyncio")
print("  numeric computation       -> NumPy (already parallel, GIL released)")
print("  CPU-bound pure Python,")
print("    large tasks             -> processes")
print("    small tasks             -> batch them first, or stay sequential")
