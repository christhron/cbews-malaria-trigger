"""
Reporting-window (Delta) sweep (TODO.md, important item; replaces the
earlier filtered-lambda-control idea): Delta is the reporting-window length
that controls how much the community-report pipeline temporally smooths
before the nowcast/pooling produce hat_iota_i -- and it only touches the
reported-signal driver, since the oracle (phi/lambda_i) bypasses reporting
entirely. Sweeping Delta therefore gives a dose-response curve of the
reported-vs-oracle active-fraction and rate gap (Section sec:info) against a
genuine, tunable deployment parameter (how often community reports are
rolled up), rather than a one-off smoothed-vs-raw control.

Same reduced 3x3 grid and 100-replicate budget as sensitivity_trigger.py.
The oracle (phi) grid is Delta-independent, so it is computed once and
reused for every Delta point.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import trigger_config as cfg
from sweep_trigger import derive_thresholds, sweep_trigger_grid

EPS = 0.3
SEED0 = 9700
PERCENTILES = (20, 55, 85)
SUPPRESSION_LEVELS = (0.5, 0.2, 0.1)
N_REPS = 100
MEASURE_YEARS = 10.0
R_INV_DAYS = 21.0

# Reporting-window length in real days; baseline Delta=0.05 (nondim) ~= 1.05
# days, so 1.0 here is "as calibrated." 0.33d (~8h) is close to the dt=1/60
# step size itself -- as near-instantaneous as this numerical scheme allows.
DELTA_DAYS = (0.33, 1.0, 3.0, 7.0, 14.0)

if __name__ == "__main__":
    dt, n_steps, burn_in_steps, measure_time = cfg.time_grid(MEASURE_YEARS)
    pop_time = cfg.N_POP.sum() * measure_time
    t_all = time.time()

    print("=== phi (oracle) grid: Delta-independent, computed once ===")
    thr_phi = derive_thresholds(EPS, seed=SEED0 + 1, n_reps=16, percentiles=PERCENTILES,
                                 dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps)
    grid_phi = sweep_trigger_grid(EPS, "phi", thr_phi["phi"], SUPPRESSION_LEVELS,
                                   seed0=SEED0, n_reps=N_REPS, pop_time=pop_time,
                                   dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                                   compute_ci=False)

    results = {"phi": grid_phi, "iota_hat_by_delta_days": {}}
    seed = SEED0 + 10
    for days in DELTA_DAYS:
        delta = days / R_INV_DAYS
        win_steps = max(round(delta / dt), 1)
        label = f"Delta={days:g}d ({win_steps} step(s) of dt)"
        print(f"\n=== {label} ===")
        t0 = time.time()
        ov = {"Delta": delta}
        thr = derive_thresholds(EPS, seed=seed + 1, n_reps=16, percentiles=PERCENTILES,
                                 dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                                 param_overrides=ov)
        grid = sweep_trigger_grid(EPS, "iota_hat", thr["iota_hat"], SUPPRESSION_LEVELS,
                                   seed0=seed, n_reps=N_REPS, pop_time=pop_time,
                                   dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                                   compute_ci=False, param_overrides=ov)
        results["iota_hat_by_delta_days"][days] = grid
        print(f"  {label} done in {time.time()-t0:.1f}s")
        seed += 2

    with open(Path(__file__).parent / "delta_sweep_grids.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote delta_sweep_grids.pkl ({time.time()-t_all:.1f}s total)")
