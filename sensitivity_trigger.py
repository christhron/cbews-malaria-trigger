"""
One-factor sensitivity sweep (TODO.md, blocking item): vary eps, rho, nu0,
kappa, and the seasonal amplitude one at a time (low / baseline / high) and
check whether the headline findings -- trigger roughly matches a matched
constant on cases averted, and a large duration saving is available at a
fixed case reduction -- survive over plausible parameter ranges.

Reduced 3x3 (threshold percentile x suppression level) grid, 100 replicates,
iota_hat driver only: intentionally lower power than the production 7x5x200
grid (this is a robustness screen, not a replacement for it), and the
reported-vs-oracle comparison is not re-tested here (that needs both
drivers; left as a further robustness check if this screen finds something
worth chasing). Every point re-derives its own thresholds from a diagnostic
run at that parameter setting (derive_thresholds' param_overrides), so
"threshold percentile" always means the same thing (how selective the
trigger is) even though the underlying distribution has shifted.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import trigger_config as cfg
from sweep_trigger import derive_thresholds, sweep_trigger_grid

EPS = 0.3
SEED0 = 9500
PERCENTILES = (20, 55, 85)
SUPPRESSION_LEVELS = (0.5, 0.2, 0.1)
N_REPS = 100
MEASURE_YEARS = 10.0

# param -> (low, baseline, high); "eps" is handled separately since it is a
# run()/sweep_trigger_grid positional argument, not a Params field routed
# through param_overrides.
PARAM_GRID = {
    "eps": (0.15, 0.30, 0.50),
    "rho": (0.075, 0.15, 0.30),
    "nu0": (0.03, 0.06, 0.12),
    "kappa": (0.15, 0.30, 0.60),
    "seasonal_delta": (0.044, 0.088, 0.176),
}

if __name__ == "__main__":
    dt, n_steps, burn_in_steps, measure_time = cfg.time_grid(MEASURE_YEARS)
    pop_time = cfg.N_POP.sum() * measure_time
    print(f"Sensitivity sweep: {PERCENTILES} threshold percentiles x "
          f"{SUPPRESSION_LEVELS} suppression levels, {N_REPS} reps, "
          f"{MEASURE_YEARS:.0f}yr horizon, iota_hat driver only")

    results = {}
    seed = SEED0
    t_all = time.time()
    for param, values in PARAM_GRID.items():
        for val in values:
            eps = val if param == "eps" else EPS
            overrides = {} if param == "eps" else {param: val}
            label = f"{param}={val:g}"
            print(f"\n=== sensitivity point: {label} (eps={eps:g}) ===")
            t0 = time.time()
            thr = derive_thresholds(eps, seed=seed + 1, n_reps=16, percentiles=PERCENTILES,
                                     dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                                     param_overrides=overrides)
            grid = sweep_trigger_grid(eps, "iota_hat", thr["iota_hat"], SUPPRESSION_LEVELS,
                                       seed0=seed, n_reps=N_REPS, pop_time=pop_time,
                                       dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                                       compute_ci=False, param_overrides=overrides)
            results[(param, val)] = grid
            print(f"  {label} done in {time.time()-t0:.1f}s")
            seed += 2

    with open(Path(__file__).parent / "sensitivity_grids.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote sensitivity_grids.pkl ({time.time()-t_all:.1f}s total)")
