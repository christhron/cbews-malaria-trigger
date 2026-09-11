"""
Exploratory sweep for the trigger-suppression policy: bang-bang, per-node,
no hysteresis -- multiplier = 1-suppression_level wherever the driving
signal Z_i(t) >= threshold, else 1 (model.py's make_trigger_policy below,
passed in as policy_fn). Run under both drivers (oracle=False: the reported
hat_iota_i; oracle=True: the true, noise-free phi_i), at eps=0.3,
rho=RHO_FIXED, on the 3-year production horizon (trigger_config.py, matched
to the paper's current primary results).

Produces three heatmaps per driver, over a (threshold, suppression_level)
grid:
  (1) Z_symp        -- cumulative symptomatic infections under the trigger
  (2) Z_rate_trig    -- (symp_off - symp_trigger) / (population-weighted
                         node-days suppression was actually ON under the
                         trigger, model.py's new cum_active_time) --
                         "symptomatic cases averted per person-day the
                         alert was sounding"
  (3) Z_rate_const    -- same formula, for a flat constant matched to the
                          trigger's own average effort (cum_suppression);
                          since a constant is always on, its denominator is
                          just the full measured window (population-weighted)

This is an exploratory pass (n_epi/n_report well below production scale,
matching check_penalty_matched_effort.py's own "cheap exploratory scan"
convention) meant to sanity-check the threshold derivation and the sign/
pattern of results before committing to a production-count run. No results
here are written into the paper yet -- see model.py's header for what's
forked from ../code, and this folder's isolation from ../code generally.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import trigger_config as cfg
from model import Params, simulate

# ----------------------------------------------------------------------
# Visual style, copied from ../code/sweep.py's palette (not imported --
# this fork's only ../code dependency is the raw network/population data
# in ekiti_adjacency.py, per trigger_config.py's header).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "font.size": 10,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

POP_TIME = cfg.N_POP.sum() * cfg.MEASURE_TIME  # normaliser for a scenario
    # that's suppressed for the entire measured window (the matched constant)


# ----------------------------------------------------------------------
def run(eps: float, alpha: float, rng_seed: int, obs_seed=None, n_reps=4,
        oracle: bool = False, const_mult=None, policy_fn=None,
        penalty: str = "exp", track_iota: bool = False,
        dt=None, n_steps=None, burn_in_steps=None,
        rho=None, nu0=None, kappa=None, seasonal_delta=None, Delta=None):
    """Thin wrapper mirroring ../code/sweep.py's run(), wired to this
    folder's own trigger_config.py (same calibration as the paper's own
    figures) and this folder's own model.py (the oracle/policy_fn-unified
    fork). dt/n_steps/burn_in_steps default to trigger_config's module-level
    (3-year) grid; pass the triple from cfg.time_grid(measure_years) to run
    at a different horizon without changing that default for other callers.
    rho/nu0/kappa/seasonal_delta/Delta override the calibrated baseline for
    the one-factor sensitivity sweep and the reporting-window (Delta) sweep;
    each defaults to the baseline value used everywhere else in this package
    (cfg.RHO_FIXED, cfg.BASELINE['nu0'], the Params default, cfg.SEASONAL_DELTA,
    the Params default) when not given, so existing callers are unaffected."""
    if obs_seed is None:
        obs_seed = rng_seed + 1
    if dt is None:
        dt = cfg.DT
    if n_steps is None:
        n_steps = cfg.N_STEPS
    if burn_in_steps is None:
        burn_in_steps = cfg.BURN_IN_STEPS
    baseline = dict(cfg.BASELINE)
    if nu0 is not None:
        baseline["nu0"] = nu0
    seasonal = dict(cfg.SEASONAL)
    if seasonal_delta is not None:
        seasonal["seasonal_delta"] = seasonal_delta
    param_overrides = {}
    if kappa is not None:
        param_overrides["kappa"] = kappa
    if Delta is not None:
        param_overrides["Delta"] = Delta
    p = Params(rho=(rho if rho is not None else cfg.RHO_FIXED), alpha=alpha,
               dt=dt, penalty=penalty, **seasonal, **baseline, **param_overrides)
    return simulate(p, cfg.M_EKITI, eps=eps, N=cfg.N_POP, n_steps=n_steps,
                     rng=np.random.default_rng(rng_seed),
                     rng_obs=np.random.default_rng(obs_seed),
                     n_reps=n_reps, x_AI0_frac=cfg.X_AI0_FRAC,
                     oracle=oracle, const_mult=const_mult, policy_fn=policy_fn,
                     burn_in_steps=burn_in_steps, track_iota=track_iota)


def make_trigger_policy(threshold: float, suppression_level: float):
    """Bang-bang, no hysteresis/memory: multiplier = 1-suppression_level
    wherever Z_i(t) >= threshold, else 1 (no suppression). Pure function of
    its input array -- same contract as the continuous rule's e^{-alpha*Z},
    so it drives both the transmission rate and the cost exactly like any
    other scenario (model.py's unified oracle/policy_fn dispatch)."""
    mult_on = 1.0 - suppression_level
    def policy_fn(Z):
        return np.where(np.asarray(Z) >= threshold, mult_on, 1.0)
    return policy_fn


def derive_thresholds(eps: float, seed: int, n_reps: int = 16,
                       percentiles=(50, 75, 90), dt=None, n_steps=None,
                       burn_in_steps=None, param_overrides=None):
    """Data-driven threshold grid: percentiles of the post-burn-in,
    pooled-over-(node,time,rep) marginal distribution of each driving
    signal, from one diagnostic off (alpha=0) run. iota_hat and phi are
    derived separately, not from a shared absolute number -- phi is the
    true, unclipped, undelayed force of infection; iota_hat is reporting-
    thinned, delayed and clipped, so it runs systematically smaller.
    param_overrides (dict of rho/nu0/kappa/seasonal_delta/Delta -> value) is
    forwarded to run(), so thresholds for a sensitivity or Delta-sweep point
    are always calibrated against that same point's own diagnostic
    distribution, not the baseline's."""
    param_overrides = param_overrides or {}
    calib = run(eps, alpha=0.0, rng_seed=seed, n_reps=n_reps, track_iota=True,
                dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                **param_overrides)
    b = burn_in_steps if burn_in_steps is not None else cfg.BURN_IN_STEPS
    iota = calib["iota_hat_traj"][b:].ravel()
    phi = calib["phi_true_traj"][b:].ravel()
    thresholds = {
        "iota_hat": np.percentile(iota, percentiles),
        "phi": np.percentile(phi, percentiles),
    }
    for driver, vals in thresholds.items():
        print(f"  {driver} thresholds (p{list(percentiles)}): "
              f"{np.array2string(vals, precision=5)}")
    return thresholds


def bootstrap_ci(statistic, *per_rep_arrays, n_boot=2000, alpha=0.05, seed=0):
    """Paired percentile-bootstrap CI for `statistic(*resampled_arrays)`,
    resampling one shared set of replicate indices (with replacement) across
    all given (n_reps,)-shaped arrays so pairing between e.g. an off/trigger/
    constant run (or between two drivers), all launched from the same seed,
    is preserved in the resample -- not resampled independently per array.
    Returns (lo, hi) at the (alpha/2, 1-alpha/2) percentiles, or (nan, nan)
    if `statistic` returns non-finite on the point estimate."""
    n = per_rep_arrays[0].shape[-1]
    for a in per_rep_arrays:
        assert a.shape[-1] == n, "per-replicate arrays must share the replicate axis length"
    point = statistic(*per_rep_arrays)
    if not np.isfinite(point):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(*(a[..., i] for a in per_rep_arrays)) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def cross_driver_ci(grid_a, grid_b, statistic, guard=None, n_boot=2000, seed=1234, alpha=0.05):
    """Paired-bootstrap CI, cell by cell, for `statistic(oa,tsa,taa,ob,tsb,tab)`
    computed from two driver grids from sweep_trigger_grid(..., compute_ci=True)
    sharing the same (threshold-percentile, suppression-level) grid shape --
    e.g. the reported-signal (iota_hat) grid vs. the perfect-information (phi)
    grid at matched percentile/S. Needed because the two drivers are separate
    sweep_trigger_grid calls (each with its own off/trigger runs), so this
    cannot be done inside a single call the way the trigger-vs-constant CI is.
    `guard(i,j)`, if given, skips a cell (leaving NaN) when False -- e.g. to
    skip cells where either grid's point-estimate rate is undefined
    (BENEFIT_FLOOR guard). Returns (point, lo, hi) arrays."""
    off_a, off_b = grid_a["off_symp_per_rep"], grid_b["off_symp_per_rep"]
    n_s, n_t = grid_a["trig_symp_per_rep"].shape[:2]
    point = np.full((n_s, n_t), np.nan)
    lo = np.full((n_s, n_t), np.nan)
    hi = np.full((n_s, n_t), np.nan)
    for i in range(n_s):
        for j in range(n_t):
            if guard is not None and not guard(i, j):
                continue
            ts_a, ta_a = grid_a["trig_symp_per_rep"][i, j], grid_a["trig_active_per_rep"][i, j]
            ts_b, ta_b = grid_b["trig_symp_per_rep"][i, j], grid_b["trig_active_per_rep"][i, j]
            point[i, j] = statistic(off_a, ts_a, ta_a, off_b, ts_b, ta_b)
            lo[i, j], hi[i, j] = bootstrap_ci(statistic, off_a, ts_a, ta_a, off_b, ts_b, ta_b,
                                               n_boot=n_boot, alpha=alpha, seed=seed + i * n_t + j)
    return point, lo, hi


def rate_diff_statistic(oa, ts_a, ta_a, ob, ts_b, ta_b):
    """rate_a - rate_b, cases averted per active person-day, for cross_driver_ci."""
    ra = (oa.mean() - ts_a.mean()) / ta_a.mean()
    rb = (ob.mean() - ts_b.mean()) / ta_b.mean()
    return ra - rb


def pct_averted_diff_statistic(oa, ts_a, ta_a, ob, ts_b, ta_b):
    """(% cases averted)_a - (% cases averted)_b, for cross_driver_ci (the
    ta_a/ta_b active-time arguments are unused here but kept so the same
    (oa,ts_a,ta_a,ob,ts_b,ta_b) call signature works for both statistics)."""
    pa = 100.0 * (oa.mean() - ts_a.mean()) / oa.mean()
    pb = 100.0 * (ob.mean() - ts_b.mean()) / ob.mean()
    return pa - pb


# ----------------------------------------------------------------------
def sweep_trigger_grid(eps: float, driver: str, thresholds, suppression_levels,
                        seed0: int, n_reps: int, pop_time=None,
                        dt=None, n_steps=None, burn_in_steps=None,
                        compute_ci: bool = True, n_boot: int = 2000,
                        param_overrides=None):
    """(threshold, suppression_level) grid for one driver ("iota_hat" or
    "phi"). Returns Z_symp (raw cumulative symptomatic burden under the
    trigger), Z_cases_averted (symp_off-symp_trigger, and the same for the
    matched constant), Z_rate_trig ((symp_off-symp_trigger)/active node-days,
    model.py's cum_active_time), Z_rate_const (same formula for a flat
    constant matched to the trigger's own average effort, active for the
    whole measured window by construction), and Z_episodes/Z_episode_length
    (count and mean length of distinct "on" episodes, model.py's
    cum_episodes). pop_time normalises Z_effort/Z_active_frac/Z_rate_const to
    the horizon actually run (cfg.N_POP.sum() * measure_time for that
    horizon) -- defaults to the module-level (3-year) POP_TIME; pass
    cfg.N_POP.sum()*measure_time for a different horizon's own measure_time,
    matching the dt/n_steps/burn_in_steps triple from
    cfg.time_grid(measure_years).

    When compute_ci is True (default), also returns paired-bootstrap 95% CIs
    for the trigger-vs-matched-constant cases-averted gap
    (Z_cases_averted_diff, _lo, _hi), the active fraction (Z_active_frac_lo/hi)
    and the trigger rate (Z_rate_trig_lo/hi); and the raw per-replicate,
    population-summed arrays (off_symp_per_rep, and per grid cell
    trig_symp_per_rep/trig_active_per_rep/const_symp_per_rep) so a caller can
    bootstrap a comparison *across two driver grids* (e.g. reported vs.
    perfect-information rate) without re-running any simulation.

    param_overrides (dict of rho/nu0/kappa/seasonal_delta/Delta -> value),
    forwarded to every internal run() call, is how the one-factor
    sensitivity sweep and the reporting-window (Delta) sweep move a single
    parameter off its calibrated baseline without touching this function's
    own logic."""
    if pop_time is None:
        pop_time = POP_TIME
    param_overrides = param_overrides or {}
    oracle = (driver == "phi")
    off = run(eps, alpha=0.0, rng_seed=seed0, n_reps=n_reps,
              dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
              **param_overrides)
    off_symp_r = off["cum_symptomatic"].sum(axis=1)   # (n_reps,)
    off_symp = off_symp_r.mean()

    n_t, n_s = len(thresholds), len(suppression_levels)
    Z_symp = np.zeros((n_s, n_t))
    Z_symp_pct = np.zeros((n_s, n_t))
    Z_rate_trig = np.full((n_s, n_t), np.nan)
    Z_rate_const = np.full((n_s, n_t), np.nan)
    Z_effort = np.zeros((n_s, n_t))
    Z_active_frac = np.zeros((n_s, n_t))  # population-time-weighted fraction
                                            # of the measured window suppression
                                            # was actually ON (cum_active_time /
                                            # POP_TIME) -- distinct from Z_effort,
                                            # which is effect-weighted (cum_suppression
                                            # / POP_TIME); the two agree only because
                                            # a bang-bang policy's "on" multiplier is
                                            # a single fixed value 1-S, so effort =
                                            # S * active_frac exactly, checked below.
    Z_trig_active_days = np.zeros((n_s, n_t))  # raw cum_active_time, for the
                                                 # rate_trig denominator (debug/audit)
    Z_episodes = np.zeros((n_s, n_t))          # total distinct "on" episodes,
                                                 # summed over nodes, mean over reps
    Z_episodes_per_town = np.zeros((n_s, n_t)) # mean episodes for a single town
                                                 # (unweighted over nodes and reps)
    Z_episode_length = np.full((n_s, n_t), np.nan)  # mean TIME duration of one
        # "on" episode (nondimensional units of 1/r =~ 21 days): per-node active
        # TIME (cum_active_time/N_i, population cancelled) divided by that
        # node's own episode count, then averaged unweighted over nodes and
        # reps -- NOT Z_trig_active_days/Z_episodes, which would still carry
        # population weighting in the numerator only and give a population-
        # size-biased answer, not a duration

    Z_cases_averted = np.zeros((n_s, n_t))        # off_symp - symp_trigger
    Z_cases_averted_const = np.zeros((n_s, n_t))   # off_symp - symp_const

    if compute_ci:
        Z_cases_averted_diff = np.zeros((n_s, n_t))     # trig - const, paired
        Z_cases_averted_diff_lo = np.full((n_s, n_t), np.nan)
        Z_cases_averted_diff_hi = np.full((n_s, n_t), np.nan)
        Z_active_frac_lo = np.full((n_s, n_t), np.nan)
        Z_active_frac_hi = np.full((n_s, n_t), np.nan)
        Z_rate_trig_lo = np.full((n_s, n_t), np.nan)
        Z_rate_trig_hi = np.full((n_s, n_t), np.nan)
        trig_symp_per_rep = np.zeros((n_s, n_t, n_reps))
        trig_active_per_rep = np.zeros((n_s, n_t, n_reps))
        const_symp_per_rep = np.zeros((n_s, n_t, n_reps))

    t0 = time.time()
    for j, T in enumerate(thresholds):
        for i, S in enumerate(suppression_levels):
            trig_fn = make_trigger_policy(T, S)
            trig = run(eps, alpha=0.0, rng_seed=seed0, n_reps=n_reps,
                       oracle=oracle, policy_fn=trig_fn,
                       dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                       **param_overrides)
            trig_symp_r = trig["cum_symptomatic"].sum(axis=1)
            trig_active_r = trig["cum_active_time"].sum(axis=1)
            trig_episodes_r = trig["cum_episodes"].sum(axis=1)
            trig_symp = trig_symp_r.mean()
            trig_active = trig_active_r.mean()

            Z_symp[i, j] = trig_symp
            Z_symp_pct[i, j] = (100.0 * (1.0 - trig_symp / off_symp)
                                 if off_symp > 0 else 0.0)
            Z_active_frac[i, j] = trig_active / pop_time
            Z_trig_active_days[i, j] = trig_active
            Z_cases_averted[i, j] = off_symp - trig_symp
            Z_episodes[i, j] = trig_episodes_r.mean()
            Z_episodes_per_town[i, j] = trig["cum_episodes"].mean()
            if trig_active > cfg.BENEFIT_FLOOR:
                Z_rate_trig[i, j] = (off_symp - trig_symp) / trig_active
            # per-node time duration (population cancels), NaN where a node
            # had zero episodes in a replicate; averaged unweighted below
            node_active_time = trig["cum_active_time"] / cfg.N_POP[np.newaxis, :]
            node_episodes = trig["cum_episodes"]
            with np.errstate(divide="ignore", invalid="ignore"):
                node_episode_length = np.where(node_episodes > 0,
                                                node_active_time / node_episodes, np.nan)
            if np.any(np.isfinite(node_episode_length)):
                Z_episode_length[i, j] = np.nanmean(node_episode_length)

            effort = trig["cum_suppression"].sum(axis=1).mean() / pop_time
            Z_effort[i, j] = effort
            const = run(eps, alpha=0.0, rng_seed=seed0, n_reps=n_reps,
                        const_mult=1.0 - effort,
                        dt=dt, n_steps=n_steps, burn_in_steps=burn_in_steps,
                        **param_overrides)
            const_symp_r = const["cum_symptomatic"].sum(axis=1)
            const_symp = const_symp_r.mean()
            Z_cases_averted_const[i, j] = off_symp - const_symp
            Z_rate_const[i, j] = (off_symp - const_symp) / pop_time

            if compute_ci:
                trig_symp_per_rep[i, j] = trig_symp_r
                trig_active_per_rep[i, j] = trig_active_r
                const_symp_per_rep[i, j] = const_symp_r
                Z_cases_averted_diff[i, j] = const_symp - trig_symp  # >0: trigger averts more
                Z_cases_averted_diff_lo[i, j], Z_cases_averted_diff_hi[i, j] = bootstrap_ci(
                    lambda t, c: c.mean() - t.mean(), trig_symp_r, const_symp_r,
                    n_boot=n_boot, seed=seed0)
                Z_active_frac_lo[i, j], Z_active_frac_hi[i, j] = bootstrap_ci(
                    lambda a: a.mean() / pop_time, trig_active_r,
                    n_boot=n_boot, seed=seed0 + 1)
                if trig_active > cfg.BENEFIT_FLOOR:
                    Z_rate_trig_lo[i, j], Z_rate_trig_hi[i, j] = bootstrap_ci(
                        lambda o, t, a: (o.mean() - t.mean()) / a.mean(),
                        off_symp_r, trig_symp_r, trig_active_r,
                        n_boot=n_boot, seed=seed0 + 2)

            print(f"    driver={driver} T={T:.5g} S={S:.2f} effort={effort:.4f} "
                  f"active_frac={Z_active_frac[i,j]:.4f} symp_trig={trig_symp:.0f} "
                  f"cases_averted={Z_cases_averted[i,j]:.0f} "
                  f"cases_averted_const={Z_cases_averted_const[i,j]:.0f} "
                  f"rate_trig={Z_rate_trig[i,j]:.5f} "
                  f"rate_const={Z_rate_const[i,j]:.5f} "
                  f"episodes={Z_episodes[i,j]:.1f}  ({time.time()-t0:.1f}s)")
            t0 = time.time()

    out = dict(driver=driver, thresholds=np.asarray(thresholds),
               suppression_levels=np.asarray(suppression_levels),
               off_symp=off_symp, Z_symp=Z_symp, Z_symp_pct=Z_symp_pct,
               Z_cases_averted=Z_cases_averted,
               Z_cases_averted_const=Z_cases_averted_const,
               Z_rate_trig=Z_rate_trig, Z_rate_const=Z_rate_const,
               Z_effort=Z_effort, Z_active_frac=Z_active_frac,
               Z_trig_active_days=Z_trig_active_days,
               Z_episodes=Z_episodes, Z_episodes_per_town=Z_episodes_per_town,
               Z_episode_length=Z_episode_length)
    if compute_ci:
        out.update(
            off_symp_per_rep=off_symp_r,
            trig_symp_per_rep=trig_symp_per_rep,
            trig_active_per_rep=trig_active_per_rep,
            const_symp_per_rep=const_symp_per_rep,
            Z_cases_averted_diff=Z_cases_averted_diff,
            Z_cases_averted_diff_lo=Z_cases_averted_diff_lo,
            Z_cases_averted_diff_hi=Z_cases_averted_diff_hi,
            Z_active_frac_lo=Z_active_frac_lo, Z_active_frac_hi=Z_active_frac_hi,
            Z_rate_trig_lo=Z_rate_trig_lo, Z_rate_trig_hi=Z_rate_trig_hi,
        )
    return out


# ----------------------------------------------------------------------
def _heatmap(ax, Z, thresholds, suppression_levels, cmap, title, cbar_label,
             fmt="{:.0f}"):
    finite = Z[np.isfinite(Z)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{t:.3g}" for t in thresholds])
    ax.set_yticks(range(len(suppression_levels)))
    ax.set_yticklabels([f"{s:.2f}" for s in suppression_levels])
    mid = (vmin + vmax) / 2.0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val > mid else INK
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                     fontsize=8, color=color)
    ax.set_xlabel("threshold $T$")
    ax.set_ylabel("suppression level $S$")
    ax.set_title(title, color=INK, loc="left")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_label, color=INK_SECONDARY)
    cb.outline.set_edgecolor(AXIS)
    return im


def make_figures(grid, outdir: Path, suffix: str = ""):
    driver = grid["driver"]
    file_tag = driver + suffix
    driver_label = r"$\hat\iota_i$" if driver == "iota_hat" else r"$\phi_i$"
    T, S = grid["thresholds"], grid["suppression_levels"]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _heatmap(ax, grid["Z_symp"], T, S, SEQ_CMAP,
             f"Symptomatic cases under trigger ({driver_label}-driven)",
             "cumulative symptomatic infections", fmt="{:.3g}")
    fig.tight_layout()
    fig.savefig(outdir / f"fig_trigger_symp_{file_tag}.png", dpi=200)
    plt.close(fig)

    vmax = max(grid["Z_cases_averted"].max(), grid["Z_cases_averted_const"].max())
    for key, label, fname in [
            ("Z_cases_averted", "trigger", f"fig_trigger_cases_averted_{file_tag}.png"),
            ("Z_cases_averted_const", "matched constant",
             f"fig_trigger_cases_averted_const_{file_tag}.png")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        im = ax.imshow(grid[key] / 1000.0, aspect="auto", origin="lower",
                        cmap=SEQ_CMAP, vmin=0, vmax=vmax / 1000.0)
        ax.set_xticks(range(len(T))); ax.set_xticklabels([f"{t:.3g}" for t in T])
        ax.set_yticks(range(len(S))); ax.set_yticklabels([f"{s:.2f}" for s in S])
        for i in range(grid[key].shape[0]):
            for j in range(grid[key].shape[1]):
                val = grid[key][i, j]
                color = "white" if val > vmax / 2 else INK
                ax.text(j, i, f"{val/1000.0:.0f}", ha="center", va="center",
                         fontsize=8, color=color)
        ax.set_xlabel("threshold $T$")
        ax.set_ylabel("suppression level $S$")
        ax.set_title(f"Cases averted, {label} ({driver_label}-driven)",
                     color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("thousands of episodes averted vs. off",
                     color=INK_SECONDARY)
        cb.outline.set_edgecolor(AXIS)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _heatmap(ax, 100.0 * grid["Z_active_frac"], T, S, SEQ_CMAP,
             f"Proportion of time suppression is active ({driver_label}-driven)",
             "% of measured window (population-time-weighted)", fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(outdir / f"fig_trigger_active_frac_{file_tag}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _heatmap(ax, grid["Z_rate_trig"], T, S, SEQ_CMAP,
             f"Cases averted per active person-day, trigger ({driver_label}-driven)",
             "(symp$_{off}$ - symp$_{trigger}$) / active node-days", fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(outdir / f"fig_trigger_rate_{file_tag}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _heatmap(ax, grid["Z_rate_const"], T, S, SEQ_CMAP,
             f"Cases averted per person-day, matched constant ({driver_label}-effort)",
             "(symp$_{off}$ - symp$_{const}$) / measured node-days", fmt="{:.4f}")
    fig.tight_layout()
    fig.savefig(outdir / f"fig_trigger_rate_const_{file_tag}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    outdir = Path(__file__).parent / "figures"
    outdir.mkdir(exist_ok=True)
    EPS = 0.3
    SEED0 = 9000
    PERCENTILES = (50, 75, 90)
    SUPPRESSION_LEVELS = (0.20, 0.15, 0.10, 0.05)
    N_REPS_EXPLORATORY = 320   # ~n_epi=40 x n_report=8 equivalent pooled count,
                                # matching check_penalty_matched_effort.py's own
                                # exploratory scale -- not the production count

    print(f"Network: Ekiti road adjacency, {cfg.N_NODES} towns, "
          f"eps={EPS}, rho={cfg.RHO_FIXED}, 3yr horizon (exploratory scan)")
    print("Deriving data-driven threshold grid from a diagnostic off run ...")
    thresholds = derive_thresholds(EPS, seed=SEED0 + 1, percentiles=PERCENTILES)

    grids = {}
    for driver in ("iota_hat", "phi"):
        print(f"\nSweeping trigger grid, driver={driver} ...")
        t0 = time.time()
        grid = sweep_trigger_grid(EPS, driver, thresholds[driver],
                                   SUPPRESSION_LEVELS, seed0=SEED0,
                                   n_reps=N_REPS_EXPLORATORY)
        print(f"  driver={driver} total: {time.time()-t0:.1f}s")
        grids[driver] = grid
        make_figures(grid, outdir)

    with open(Path(__file__).parent / "trigger_grids_exploratory.pkl", "wb") as f:
        pickle.dump(grids, f)   # so a plotting-only fix never needs to rerun
                                 # the Monte Carlo sweep to get new PNGs

    print(f"\nWrote figures to {outdir}")
