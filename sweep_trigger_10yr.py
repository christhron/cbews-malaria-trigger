"""
10-year, coarse-grid version of sweep_trigger.py's exploratory sweep: a
wider threshold range (10th-90th percentile, not just 50th-90th) and
suppression levels up to 50% (not just 20%), over a much longer horizon, at
a coarse 4x4 grid so it stays fast. A first look before committing to a
finer grid -- see sweep_trigger.py's own module docstring for what this
experiment is and where it sits relative to ../code (untouched) and the
paper (not yet informed by any of this).
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
    PERCENTILES = (10, 35, 65, 90)
    SUPPRESSION_LEVELS = (0.50, 0.35, 0.20, 0.10)
    N_REPS = 200   # reduced from the 3-year exploratory scan's 320, to keep
                    # this coarse 10-year first look fast; bump once the
                    # pattern looks worth refining

    MEASURE_YEARS = 10.0
    dt, n_steps, burn_in_steps, measure_time = cfg.time_grid(MEASURE_YEARS)
    pop_time = cfg.N_POP.sum() * measure_time

    print(f"Network: Ekiti road adjacency, {cfg.N_NODES} towns, eps={EPS}, "
          f"rho={cfg.RHO_FIXED}, {MEASURE_YEARS:.0f}yr horizon (coarse scan)")
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
        make_figures(grid, outdir, suffix="_10yr")

    with open(Path(__file__).parent / "trigger_grids_10yr.pkl", "wb") as f:
        pickle.dump(grids, f)

    print(f"\nWrote figures to {outdir}")
