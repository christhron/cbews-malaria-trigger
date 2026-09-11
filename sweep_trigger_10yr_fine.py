"""
10-year, finer-grid follow-up to sweep_trigger_10yr.py's coarse first look:
a much wider threshold range (1st-99th percentile, not just 10th-90th) at
7 points instead of 4, and 5 suppression levels (10%-50%, with extra
resolution around the 10-20% band where the coarse run showed the trigger
crossing over the matched constant on total cases averted). Same horizon,
network, rho and replicate count as the coarse run, so the two are directly
comparable -- see sweep_trigger.py's module docstring for what this
experiment is generally.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import trigger_config as cfg
from sweep_trigger import derive_thresholds, make_figures, sweep_trigger_grid

if __name__ == "__main__":
    outdir = Path(__file__).parent / "figures"
    outdir.mkdir(exist_ok=True)
    EPS = 0.3
    SEED0 = 9000
    PERCENTILES = (1, 20, 40, 55, 70, 85, 99)
    SUPPRESSION_LEVELS = (0.50, 0.30, 0.20, 0.15, 0.10)
    N_REPS = 500   # raised from 200 (2026-09-11) to tighten the paired-bootstrap
                    # CIs on the borderline grid cells

    MEASURE_YEARS = 10.0
    dt, n_steps, burn_in_steps, measure_time = cfg.time_grid(MEASURE_YEARS)
    pop_time = cfg.N_POP.sum() * measure_time

    print(f"Network: Ekiti road adjacency, {cfg.N_NODES} towns, eps={EPS}, "
          f"rho={cfg.RHO_FIXED}, {MEASURE_YEARS:.0f}yr horizon (fine scan)")
    print(f"  n_steps={n_steps}, dt={dt:.5f}, burn_in_steps={burn_in_steps}")
    print("Deriving data-driven threshold grid from a diagnostic off run ...")
    thresholds = derive_thresholds(EPS, seed=SEED0 + 1, percentiles=PERCENTILES,
                                    dt=dt, n_steps=n_steps,
                                    burn_in_steps=burn_in_steps)

    grids = {}
    for driver in ("iota_hat", "phi"):
        print(f"\nSweeping trigger grid, driver={driver} ...")
        t0 = time.time()
        grid = sweep_trigger_grid(EPS, driver, thresholds[driver],
                                   SUPPRESSION_LEVELS, seed0=SEED0, n_reps=N_REPS,
                                   pop_time=pop_time, dt=dt, n_steps=n_steps,
                                   burn_in_steps=burn_in_steps)
        print(f"  driver={driver} total: {time.time()-t0:.1f}s")
        grids[driver] = grid
        make_figures(grid, outdir, suffix="_10yr_fine")

    with open(Path(__file__).parent / "trigger_grids_10yr_fine.pkl", "wb") as f:
        pickle.dump(grids, f)

    print(f"\nWrote figures to {outdir}")
