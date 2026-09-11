"""Figure for the reporting-window (Delta) sweep (delta_sweep_trigger.py):
how the reported-signal-vs-oracle episode-count and episode-length ratio (and,
for reference, the active-fraction ratio and rate difference) move as the
community-reporting window widens from near-instantaneous to two weeks. This
is the dose-response test of the smoothing mechanism proposed in
paper_trigger/cbews_trigger.tex, Section sec:info.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sweep_trigger import AXIS, INK, INK_SECONDARY

with open("delta_sweep_grids.pkl", "rb") as f:
    results = pickle.load(f)

phi = results["phi"]
by_delta = results["iota_hat_by_delta_days"]
deltas = sorted(by_delta.keys())

outdir = Path("../paper_trigger/figures")
if not outdir.parent.exists():
    outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

epi_ratio = []
len_ratio = []
af_ratio = []
rate_rel = []
for d in deltas:
    g = by_delta[d]
    epi_ratio.append((g["Z_episodes_per_town"] / phi["Z_episodes_per_town"]).mean())
    len_ratio.append((g["Z_episode_length"] / phi["Z_episode_length"]).mean())
    af_ratio.append((g["Z_active_frac"] / phi["Z_active_frac"]).mean())
    rate_rel.append((100 * (g["Z_rate_trig"] - phi["Z_rate_trig"]) / phi["Z_rate_trig"]).mean())

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.plot(deltas, epi_ratio, "o-", color=INK, label="episodes per town, ratio")
ax.plot(deltas, len_ratio, "s-", color="#b3541e", label="episode length, ratio")
ax.axhline(1.0, color=AXIS, linewidth=1, linestyle="--")
ax.set_xscale("log")
ax.set_xlabel(r"reporting window $\Delta$ (days)")
ax.set_ylabel(r"$\hat\iota_i$ / $\lambda_i$ ratio (mean over grid)")
ax.set_title("Episode structure vs. reporting cadence", color=INK, loc="left")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.plot(deltas, af_ratio, "o-", color=INK, label="active fraction, ratio")
ax2 = ax.twinx()
ax2.plot(deltas, rate_rel, "^-", color="#3a6ea5", label="rate, relative diff. (%)")
ax.axhline(1.0, color=AXIS, linewidth=1, linestyle="--")
ax2.axhline(0.0, color="#3a6ea5", linewidth=0.5, linestyle=":")
ax.set_xscale("log")
ax.set_xlabel(r"reporting window $\Delta$ (days)")
ax.set_ylabel(r"active-fraction ratio $\hat\iota_i/\lambda_i$", color=INK)
ax2.set_ylabel("rate, relative difference (%)", color="#3a6ea5")
ax.set_title("Duration and efficiency vs. reporting cadence", color=INK, loc="left")
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_delta_sweep_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_delta_sweep_10yr_fine.png")

print(f"{'Delta(d)':>10s} {'episodes ratio':>16s} {'epi_len ratio':>16s} {'active_frac ratio':>20s} {'rate_rel(%)':>14s}")
for d, e, l, a, r in zip(deltas, epi_ratio, len_ratio, af_ratio, rate_rel):
    print(f"{d:>10g} {e:>16.3f} {l:>16.3f} {a:>20.3f} {r:>14.2f}")

# crossover point (episodes ratio == 1) by linear interpolation in log-Delta
import numpy as np
logd = np.log(deltas)
er = np.array(epi_ratio)
cross = None
for i in range(len(deltas) - 1):
    if (er[i] - 1) * (er[i+1] - 1) < 0:
        frac = (1 - er[i]) / (er[i+1] - er[i])
        cross = np.exp(logd[i] + frac * (logd[i+1] - logd[i]))
        break
print(f"\nepisodes-ratio crosses 1 at Delta ~= {cross:.2f} days" if cross else "\nno crossover in range tested")
