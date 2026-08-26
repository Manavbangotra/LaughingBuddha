# Extracted from: Chapter 87 — Distillation and Model Specialization
# Source: src/.../ch087-distillation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""A student inherits the teacher's errors along with its competence."""
import numpy as np

rng = np.random.default_rng(2)
N_ITEMS = 5000

# The teacher: good overall, with a systematic weakness on one subpopulation
# and a set of outright factual errors it is confident about.
TEACHER_ACC_COMMON = 0.94
TEACHER_ACC_RARE = 0.61              # systematically worse on rare cases
TEACHER_CONFIDENT_ERRORS = 0.03      # wrong AND confident

is_rare = rng.random(N_ITEMS) < 0.15
teacher_correct = np.where(
    is_rare,
    rng.random(N_ITEMS) < TEACHER_ACC_RARE,
    rng.random(N_ITEMS) < TEACHER_ACC_COMMON)
teacher_confident_wrong = (~teacher_correct) & (
    rng.random(N_ITEMS) < TEACHER_CONFIDENT_ERRORS / (1 - TEACHER_ACC_COMMON))

print(f"teacher accuracy, common cases : "
      f"{teacher_correct[~is_rare].mean():.3f}")
print(f"teacher accuracy, rare cases   : "
      f"{teacher_correct[is_rare].mean():.3f}")
print(f"teacher confidently wrong      : "
      f"{teacher_confident_wrong.mean():.3f}\n")

# The student learns from the teacher's OUTPUTS, so its ceiling is the
# teacher's behaviour, not the ground truth. Capacity costs it a little more.
CAPACITY_RETENTION_COMMON = 0.98
CAPACITY_RETENTION_RARE = 0.82       # rare knowledge is what capacity loses first

student_correct = teacher_correct & np.where(
    is_rare,
    rng.random(N_ITEMS) < CAPACITY_RETENTION_RARE,
    rng.random(N_ITEMS) < CAPACITY_RETENTION_COMMON)

print(f"{'population':<16} {'teacher':>9} {'student':>9} {'gap':>8}")
for label, mask in [("common", ~is_rare), ("rare", is_rare),
                    ("overall", np.ones(N_ITEMS, dtype=bool))]:
    t_acc = float(teacher_correct[mask].mean())
    s_acc = float(student_correct[mask].mean())
    print(f"{label:<16} {t_acc:>9.3f} {s_acc:>9.3f} {s_acc - t_acc:>+8.3f}")

overall_gap = float(student_correct.mean() - teacher_correct.mean())
rare_gap = float(student_correct[is_rare].mean() - teacher_correct[is_rare].mean())
print(f"\noverall gap looks acceptable : {overall_gap:+.3f}")
print(f"rare-population gap           : {rare_gap:+.3f} "
      f"({rare_gap / overall_gap:.1f}x the headline number)")

# The errors the student inherits WITHOUT ever seeing ground truth.
inherited = float((~teacher_correct & ~student_correct).mean())
print(f"\nerrors inherited from the teacher: {inherited:.3f}")
print(f"of which the teacher was confident: "
      f"{float(teacher_confident_wrong.mean()):.3f}")
print("A confidently wrong teacher produces a confidently wrong student, and no "
      "amount of distillation data fixes it — the student never sees the truth.")

print("""
The aggregate number hides the failure. Overall accuracy drops by a couple of
points, which passes a launch review; on the rare subpopulation it drops by
several times that, because capacity loss takes rare knowledge first while
general fluency survives.

Evaluate distillation on the SLICES you care about, not on the aggregate. And
note the ceiling: the student is trained on the teacher's outputs and never sees
ground truth, so every systematic teacher error is transferred intact along with
the competence.""")
