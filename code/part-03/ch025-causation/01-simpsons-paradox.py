# -*- coding: utf-8 -*-
# Extracted from: Chapter 25 — Correlation, Causation, and Confounding
# Source: src/.../ch025-causation.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Simpson's paradox, confounding bias, and collider bias — all constructed.
"""
import numpy as np
import pandas as pd

# --- section 6.1: the reversal, exactly ------------------------------------
print("=" * 72)
print("Simpson's paradox: A wins in both subgroups and loses overall")
print("=" * 72)

data = pd.DataFrame([
    ("A", "mild",   81,  87),
    ("A", "severe", 192, 263),
    ("B", "mild",   234, 270),
    ("B", "severe", 55,  80),
], columns=["treatment", "severity", "recovered", "total"])
data["rate"] = data["recovered"] / data["total"]

pivot = data.pivot(index="treatment", columns="severity",
                   values=["recovered", "total", "rate"])
print(f"{'':<11} {'mild':>18} {'severe':>18} {'combined':>18}")
for t in ("A", "B"):
    sub = data[data.treatment == t]
    combined = sub["recovered"].sum() / sub["total"].sum()
    cells = []
    for s in ("mild", "severe"):
        row = sub[sub.severity == s].iloc[0]
        cells.append(f"{int(row.recovered)}/{int(row.total)} = {row.rate:.0%}")
    tot = f"{sub['recovered'].sum()}/{sub['total'].sum()} = {combined:.0%}"
    print(f"treatment {t:<2} {cells[0]:>18} {cells[1]:>18} {tot:>18}")

a_mild = data.query("treatment=='A' and severity=='mild'").iloc[0]
b_mild = data.query("treatment=='B' and severity=='mild'").iloc[0]
a_sev = data.query("treatment=='A' and severity=='severe'").iloc[0]
b_sev = data.query("treatment=='B' and severity=='severe'").iloc[0]
a_all = data.query("treatment=='A'")["recovered"].sum() / 350
b_all = data.query("treatment=='B'")["recovered"].sum() / 350

print(f"\nA beats B on mild   : {a_mild.rate:.1%} vs {b_mild.rate:.1%}  "
      f"({a_mild.rate > b_mild.rate})")
print(f"A beats B on severe : {a_sev.rate:.1%} vs {b_sev.rate:.1%}  "
      f"({a_sev.rate > b_sev.rate})")
print(f"A beats B overall   : {a_all:.1%} vs {b_all:.1%}  "
      f"({a_all > b_all})   <- REVERSED")

# --- eq. 25.6: why. the allocation differs ----------------------------------
print(f"\nwhy: severity is distributed very differently across treatments")
for t in ("A", "B"):
    sub = data[data.treatment == t]
    frac_severe = sub[sub.severity == "severe"]["total"].iloc[0] / sub["total"].sum()
    print(f"  treatment {t}: {frac_severe:.0%} of its patients were severe")

# --- eq. 25.7: adjust by the POPULATION distribution ------------------------
pop = data.groupby("severity")["total"].sum()
pop_frac = pop / pop.sum()
print(f"\npopulation severity mix: "
      f"{ {k: f'{v:.0%}' for k, v in pop_frac.items()} }")

print(f"\n{'treatment':<11} {'pooled (eq. 25.6)':>20} "
      f"{'adjusted (eq. 25.7)':>22}")
for t in ("A", "B"):
    sub = data[data.treatment == t].set_index("severity")
    pooled = sub["recovered"].sum() / sub["total"].sum()
    adjusted = sum(sub.loc[s, "rate"] * pop_frac[s] for s in pop_frac.index)
    print(f"{t:<11} {pooled:>20.1%} {adjusted:>22.1%}")

print("\nAdjusting for severity recovers A's advantage. Both computations are")
print("arithmetically correct on the same table; which one answers your")
print("question depends on whether severity is a confounder or a consequence")
print("of treatment — and the table cannot tell you which.")

# --- eq. 25.8: the magnitude of confounding bias ----------------------------
print("\n" + "=" * 72)
print("confounding bias is a product of two terms (eq. 25.8)")
print("=" * 72)

rng = np.random.default_rng(0)
n = 400_000
print(f"{'Z effect on Y':>14} {'Z imbalance':>13} {'predicted bias':>16} "
      f"{'measured':>10}")
for z_effect in (0.0, 0.3):
    for imbalance in (0.0, 0.5):
        # Z is a confounder: it drives assignment and the outcome.
        z = rng.random(n) < 0.5
        p_treat = 0.5 + imbalance * (z - 0.5)
        t = rng.random(n) < p_treat
        true_effect = 0.10
        y = (0.3 + true_effect * t + z_effect * z
             + rng.normal(0, 0.05, n))

        naive = y[t].mean() - y[~t].mean()
        measured_bias = naive - true_effect
        p_z_given_t1 = z[t].mean()
        p_z_given_t0 = z[~t].mean()
        predicted = z_effect * (p_z_given_t1 - p_z_given_t0)
        print(f"{z_effect:>14.2f} {imbalance:>13.2f} {predicted:>16.4f} "
              f"{measured_bias:>10.4f}")

print("\nBias requires BOTH: Z must affect the outcome AND be distributed")
print("differently across treatment groups. Either term at zero, no bias —")
print("which is exactly what randomisation guarantees for the second term.")

# --- section 5.2: randomisation eliminates the selection term ---------------
print("\n" + "=" * 72)
print("randomisation zeroes the selection-bias term of eq. 25.3")
print("=" * 72)

n = 300_000
engagement = rng.normal(0, 1, n)              # an UNMEASURED confounder
y0 = 0.20 + 0.15 * engagement                 # baseline retention
y1 = y0 + 0.05                                # true effect: +5pp for everyone
true_ate = (y1 - y0).mean()

# Observational: engaged users self-select into the treatment.
t_obs = rng.random(n) < 1 / (1 + np.exp(-2 * engagement))
y_obs = np.where(t_obs, y1, y0)
naive_obs = y_obs[t_obs].mean() - y_obs[~t_obs].mean()
selection_term = y0[t_obs].mean() - y0[~t_obs].mean()

# Randomised: assignment is a coin flip, ignoring engagement entirely.
t_rct = rng.random(n) < 0.5
y_rct = np.where(t_rct, y1, y0)
naive_rct = y_rct[t_rct].mean() - y_rct[~t_rct].mean()
selection_rct = y0[t_rct].mean() - y0[~t_rct].mean()

print(f"true ATE                          : {true_ate:>+8.4f}")
print(f"\n{'design':<16} {'observed diff':>15} {'selection term':>16} "
      f"{'error':>9}")
print(f"{'observational':<16} {naive_obs:>+15.4f} {selection_term:>+16.4f} "
      f"{naive_obs - true_ate:>+9.4f}")
print(f"{'randomised':<16} {naive_rct:>+15.4f} {selection_rct:>+16.4f} "
      f"{naive_rct - true_ate:>+9.4f}")
print("\nThe observational estimate is roughly "
      f"{naive_obs/true_ate:.0f}x the true effect. Randomisation drives the")
print("selection term to zero WITHOUT measuring engagement — which is why it")
print("protects against confounders you do not know exist (eq. 25.5).")

# --- section 6.3: collider bias ---------------------------------------------
print("\n" + "=" * 72)
print("collider bias: conditioning CREATES an association")
print("=" * 72)

n = 200_000
talent = rng.normal(0, 1, n)
connections = rng.normal(0, 1, n)             # independent by construction
admitted = (talent + connections) > 1.2       # a common effect

print(f"correlation in the whole population : "
      f"{np.corrcoef(talent, connections)[0,1]:+.4f}")
print(f"correlation among the ADMITTED      : "
      f"{np.corrcoef(talent[admitted], connections[admitted])[0,1]:+.4f}")
print(f"correlation among the REJECTED      : "
      f"{np.corrcoef(talent[~admitted], connections[~admitted])[0,1]:+.4f}")
print(f"\n({admitted.mean():.0%} were admitted)")
print("\nTalent and connections are independent by construction. Restricting")
print("to the admitted manufactures a negative association: among those who")
print("got in, someone with few connections must have had talent.")
print("\nThis is the same mechanism as selection bias (Chapter 22) and is why")
print("'control for everything' is bad advice (table 25.1).")
