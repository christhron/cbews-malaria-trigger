"""Summarises sensitivity_grids.pkl (sensitivity_trigger.py) into the LaTeX
table body for the paper's sensitivity subsection: for each parameter and
each of its low/baseline/high values, the trigger-vs-matched-constant gap
(mean and range over the reduced 3x3 grid) and the active-fraction range,
so the headline "trigger roughly matches a constant of equal effort, active
only a fraction of the time" finding can be checked for robustness without
re-deriving anything from the raw grids by hand.
"""
import pickle

import numpy as np

with open("sensitivity_grids.pkl", "rb") as f:
    results = pickle.load(f)

PARAM_GRID = {
    "eps": (0.15, 0.30, 0.50),
    "rho": (0.075, 0.15, 0.30),
    "nu0": (0.03, 0.06, 0.12),
    "kappa": (0.15, 0.30, 0.60),
    "seasonal_delta": (0.044, 0.088, 0.176),
}
LABELS = {"eps": r"$\varepsilon$", "rho": r"$\rho$", "nu0": r"$\nu_0$",
          "kappa": r"$\kappa$", "seasonal_delta": r"$\delta$"}

print(r"\begin{tabular}{llrrrr}")
print(r"\toprule")
print(r"Parameter & Value & Mean gap (pp) & Min (pp) & Max (pp) & Active fraction range \\")
print(r"\midrule")
for param, values in PARAM_GRID.items():
    for k, val in enumerate(values):
        g = results[(param, val)]
        off = g["off_symp"]
        pct_trig = g["Z_symp_pct"]
        pct_const = 100.0 * g["Z_cases_averted_const"] / off
        gap = pct_trig - pct_const
        af = g["Z_active_frac"]
        label = LABELS[param] if k == 0 else ""
        tag = " (baseline)" if k == 1 else ""
        print(f"{label} & {val:g}{tag} & {gap.mean():+.2f} & {gap.min():+.2f} & "
              f"{gap.max():+.2f} & [{af.min():.2f}, {af.max():.2f}] \\\\")
    print(r"\addlinespace")
print(r"\bottomrule")
print(r"\end{tabular}")
