# CB-EWS trigger-policy simulation

Simulation code for **"Design, Simulation, and Evaluation of a Community-Based
Malaria Early-Warning System"** (Christopher Thron and Laxmi Laxmi).

The paper designs and simulates a closed-loop, network-structured
community-based malaria early-warning system (CB-EWS): local case reports
drive a spatially pooled threat estimate, and that estimate triggers a
fixed-strength suppression of transmission whenever it crosses a threshold.
This repository is the exact code behind every figure and table in the paper
— a stochastic, network-coupled compartmental malaria model, the
community-reporting/nowcast pipeline that produces the trigger signal, the
threshold-trigger and matched-constant policies, and the sweep/plotting
scripts that generate the results.

## What's here

| File | Purpose |
|---|---|
| `model.py` | The stochastic (tau-leap) three-compartment (SN/SI/AI) network transmission model, the community-reporting + nowcast pipeline, and the `simulate(...)` entry point. Accepts a pluggable `policy_fn(Z) -> multiplier` for the suppression response. |
| `trigger_config.py` | Calibrated parameters (Table 1 of the paper), the Ekiti road-adjacency coupling matrix, the asymptomatic-reservoir initial condition, and the time-grid helper. |
| `ekiti_adjacency.py` | The 16-town Ekiti State (Nigeria) road-adjacency network and 2006-census town populations used to build the coupling matrix. |
| `sweep_trigger.py` | The threshold trigger policy (`make_trigger_policy`), data-driven threshold selection from a no-suppression diagnostic run (`derive_thresholds`), and the grid sweep that runs the no-suppression baseline, the trigger, and a matched-constant comparison at shared seeds (`sweep_trigger_grid`). |
| `sweep_trigger_10yr_fine.py` | The production run: 10-year horizon, 7 threshold percentiles × 5 suppression levels × 2 drivers (reported signal / perfect information), 200 replicates each. Produces every "10yr fine" figure and the numbers behind Table 2 in the paper. |
| `sweep_trigger_10yr.py`, `sweep_trigger.py` (as script) | Earlier, coarser exploratory sweeps kept for provenance. |
| `make_paper_figures.py` | Post-processes the cached grid into the paper's figures and tables, including paired-bootstrap 95% CIs (`cross_driver_ci`, `bootstrap_ci` in `sweep_trigger.py`), alert-episode statistics, and an iso-effectiveness curve. |
| `sensitivity_trigger.py` | One-factor sensitivity sweep (ε, ρ, ν₀, κ, seasonal amplitude) on a reduced grid; `make_sensitivity_table.py` formats the results table. |
| `delta_sweep_trigger.py` | Sweeps the reporting-window length Δ (community-report aggregation cadence) to trace how it changes the reported-vs-perfect-information comparison; `make_delta_figure.py` builds the figure. |
| `trigger_grids_10yr_fine.pkl` (+ `_10yr.pkl`, `_exploratory.pkl`), `sensitivity_grids.pkl`, `delta_sweep_grids.pkl` | Cached sweep results (small — a few MB) so figures can be regenerated without re-running the simulation. |
| `*.log` | Console output from the runs that produced each cached grid. |

## Quickstart

```bash
pip install -r requirements.txt
python sweep_trigger_10yr_fine.py   # ~40-50 minutes single-threaded; writes
                                     # figures/ and trigger_grids_10yr_fine.pkl
python make_paper_figures.py        # regenerates the paper's figures from the pkl
```

Everything is deterministic given the fixed seeds in `trigger_config.py`
(`X_AI0_SEED = 424242` for the initial asymptomatic reservoir) and
`sweep_trigger_10yr_fine.py` (`SEED0 = 9000` for the sweep); within a grid
cell, the no-suppression, trigger, and matched-constant runs share seeds so
they are paired comparisons.

To regenerate just the figures/table from the existing cached grid without
re-running the (slow) simulation, skip straight to `make_paper_figures.py` —
`trigger_grids_10yr_fine.pkl` is already checked in. Likewise
`sensitivity_grids.pkl` and `delta_sweep_grids.pkl` are checked in, so
`make_sensitivity_table.py` and `make_delta_figure.py` can be re-run
immediately without re-running `sensitivity_trigger.py` (~65 minutes) or
`delta_sweep_trigger.py` (~25 minutes).

## Model summary

Each of the 16 Ekiti towns is a three-compartment node (symptomatic-latent
`SN`, symptomatic-infectious `SI`, asymptomatic-infectious `AI`), coupled by a
row-stochastic mixing matrix built from the real road-adjacency network.
Local case reports are aggregated into noisy, delayed windowed counts, run
through a nowcast filter, and spatially pooled to produce a threat estimate
`iota_hat`. A threshold trigger switches a fixed suppression level `S` on at
a node whenever its driving signal `Z` crosses a threshold `T`, and off
otherwise; this is compared against a constant policy carrying the same
time-averaged suppression effort, and against an idealised variant driven by
the true (noise-free) force of infection instead of the reported signal. See
the paper for the full model specification, calibration, and results.

## Citation

If you use this code, please cite the paper (details to be added once
published/preprinted; in the meantime, cite this repository).

## License

MIT — see [LICENSE](LICENSE).
