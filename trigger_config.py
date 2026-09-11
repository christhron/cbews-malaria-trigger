"""
Shared configuration for the trigger-suppression-policy experiment. Uses
ekiti_adjacency.py (network + 2006-census town populations), vendored in this
folder as a standalone copy so this package has no dependency outside itself;
every other constant here is the same calibration used throughout this
package (BASELINE, dry-/rainy-season R0 targets from Olorunniyi et al. 2021,
X_AI0_FRAC heterogeneous reservoir with a fixed seed).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import ekiti_adjacency as ekiti


def row_stochastic(A: np.ndarray) -> sp.csr_matrix:
    """Degree-normalised M_ij = A_ij / deg(i); copied from ../code/sweep.py
    (not imported from it, to avoid pulling in that file's matplotlib/figure
    code for a single ten-line helper)."""
    deg = A.sum(axis=1, keepdims=True)
    if (deg == 0).any():
        raise ValueError("isolated node with no edges; M would have a zero row")
    return sp.csr_matrix(A / deg)


M_EKITI = row_stochastic(ekiti.A.astype(float))
N_NODES = ekiti.n
N_POP = np.array(ekiti.population, dtype=float)

# ----------------------------------------------------------------------
# Calibration -- identical to ../code/sweep.py's BASELINE/RHO_FIXED/
# X_AI0_FRAC/BENEFIT_FLOOR (same six structural parameters, same reporting
# probability, same heterogeneous importation seeding), so this experiment
# and the paper's own figures share the same baseline environment.
R_INV_DAYS = 21.0
BASELINE = dict(R0=0.644, sigma=0.3, theta=2.1, gamma=0.35, Tdel=0.47, nu0=0.06)
RHO_FIXED = 0.15
BENEFIT_FLOOR = 1.0  # infections; below this, CE/rate ratios are too noisy to trust

SEASONAL_DELTA = 0.088
SEASONAL_PERIOD = 365.0 / R_INV_DAYS            # one calendar year, nondim
SEASONAL_T_PEAK = 3 * SEASONAL_PERIOD / 4       # t=0 at dry-season onset,
                                                  # same convention as sweep.py
BURN_IN_TIME = SEASONAL_PERIOD / 2               # one dry season, simulated
                                                  # but excluded from every
                                                  # accumulated outcome

X_AI0_FRAC_MEAN = 0.2
X_AI0_DISPERSION_K = 2.0
X_AI0_SEED = 424242
_rng_x_ai0 = np.random.default_rng(X_AI0_SEED)
_lambda_i = _rng_x_ai0.gamma(shape=X_AI0_DISPERSION_K,
                              scale=(X_AI0_FRAC_MEAN * N_POP) / X_AI0_DISPERSION_K,
                              size=N_NODES)
X_AI0_FRAC = _rng_x_ai0.poisson(_lambda_i) / N_POP
assert np.all(X_AI0_FRAC >= 0) and np.all(X_AI0_FRAC < 1), \
    "X_AI0_FRAC out of range -- check X_AI0_DISPERSION_K/X_AI0_FRAC_MEAN"


def time_grid(measure_years: float):
    """(dt, n_steps, burn_in_steps, measure_time) for a burn-in of one dry
    season followed by measure_years of measured window, discretised at
    dt ~= 1/60 (~=8 hours) -- identical formula to ../code/sweep.py's
    _time_grid."""
    measure_time = measure_years * 365.0 / R_INV_DAYS
    total_time = measure_time + BURN_IN_TIME
    n_steps = round(total_time * 60)
    dt = total_time / n_steps
    burn_in_steps = round(BURN_IN_TIME / dt)
    return dt, n_steps, burn_in_steps, measure_time


DT, N_STEPS, BURN_IN_STEPS, MEASURE_TIME = time_grid(3.0)  # matches the
    # paper's current primary horizon (sweep.py's MEASURE_TIME default)
SEASONAL = dict(seasonal_delta=SEASONAL_DELTA, seasonal_t_peak=SEASONAL_T_PEAK,
                 seasonal_period=SEASONAL_PERIOD)
