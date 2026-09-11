"""One-off figures for paper_trigger/cbews_trigger.tex, built from the cached
10-year fine grid (trigger_grids_10yr_fine.pkl): (1) trigger and matched-constant
percent-of-cases-averted heatmaps side by side sharing one colorbar, (2) the
active-fraction heatmap expressed as a proportion (0-1) instead of percent, and
(3) the difference in percent-of-cases-averted between the reported-signal
(iota_hat) and perfect-information (lambda) triggers. Reuses sweep_trigger.py's
style constants for visual consistency with the rest of the figure set.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sweep_trigger import AXIS, DIV_CMAP, INK, INK_SECONDARY, SEQ_CMAP

with open("trigger_grids_10yr_fine.pkl", "rb") as f:
    grids = pickle.load(f)

d = grids["iota_hat"]
T, S = d["thresholds"], d["suppression_levels"]
off = d["off_symp"]
pct_trig = 100.0 * d["Z_cases_averted"] / off
pct_const = 100.0 * d["Z_cases_averted_const"] / off

outdir = Path("../paper_trigger/figures")
if not outdir.parent.exists():
    # standalone checkout without the paper's sibling directory: write
    # alongside the other sweep scripts' figures instead.
    outdir = Path("figures")
outdir.mkdir(parents=True, exist_ok=True)

# ---- (1) side-by-side percent-cases-averted, one shared colorbar ----
vmax = max(pct_trig.max(), pct_const.max())
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
for ax, Z, label in [(axes[0], pct_trig, "trigger"), (axes[1], pct_const, "matched constant")]:
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(T))); ax.set_xticklabels([f"{t:.3g}" for t in T])
    ax.set_yticks(range(len(S))); ax.set_yticklabels([f"{s:.2f}" for s in S])
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            color = "white" if val > vmax / 2 else INK
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("threshold $T$")
    ax.set_title(label, color=INK, loc="left")
axes[0].set_ylabel("suppression level $S$")
cb = fig.colorbar(im, ax=axes, pad=0.02)
cb.set_label("% of cases averted vs. off", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.savefig(outdir / "fig_trigger_cases_averted_pct_iota_hat_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_cases_averted_pct_iota_hat_10yr_fine.png")

# ---- (2) active fraction as a proportion (0-1), not percent ----
Z = d["Z_active_frac"]
fig, ax = plt.subplots(figsize=(6.0, 4.2))
im = ax.imshow(Z, aspect="auto", origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=Z.max())
ax.set_xticks(range(len(T))); ax.set_xticklabels([f"{t:.3g}" for t in T])
ax.set_yticks(range(len(S))); ax.set_yticklabels([f"{s:.2f}" for s in S])
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        val = Z[i, j]
        color = "white" if val > Z.max() / 2 else INK
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
ax.set_xlabel("threshold $T$")
ax.set_ylabel("suppression level $S$")
ax.set_title(r"Proportion of time suppression is active ($\hat\iota_i$-driven)",
             color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("proportion of measured window (population-time-weighted)", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_active_frac_iota_hat_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_active_frac_iota_hat_10yr_fine.png (overwritten, now a proportion)")

print("max abs percentage-point difference, trigger vs const:",
      round(np.abs(pct_trig - pct_const).max(), 2))

# ---- (3) difference in % cases averted: reported (iota_hat) minus perfect-info (lambda) ----
di, dp = grids["iota_hat"], grids["phi"]
assert di["off_symp"] == dp["off_symp"], "off baselines differ; cannot difference percentages"
pct_i = 100.0 * di["Z_cases_averted"] / di["off_symp"]
pct_p = 100.0 * dp["Z_cases_averted"] / dp["off_symp"]
diff = pct_i - pct_p                       # > 0: reported signal averts a larger share than perfect info
pctl = (1, 20, 40, 55, 70, 85, 99)        # PERCENTILES, from sweep_trigger_10yr_fine.py
Slev = di["suppression_levels"]
m = float(np.abs(diff).max())

fig, ax = plt.subplots(figsize=(6.6, 4.2))
im = ax.imshow(diff, aspect="auto", origin="lower", cmap=DIV_CMAP, vmin=-m, vmax=m)
ax.set_xticks(range(len(pctl))); ax.set_xticklabels([str(q) for q in pctl])
ax.set_yticks(range(len(Slev))); ax.set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        v = diff[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.65 * m else INK)
ax.set_xlabel("threshold percentile")
ax.set_ylabel("suppression level $S$")
ax.set_title(r"$\hat\iota_i$-driven minus $\lambda_i$-driven (% of cases averted)",
             color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("percentage-point difference", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_cases_averted_pct_diff_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_cases_averted_pct_diff_10yr_fine.png")
print(f"  diff range (pp): {diff.min():+.2f} to {diff.max():+.2f}, mean {diff.mean():+.2f}; "
      f"cells with reported < perfect-info: {(diff < 0).sum()}/{diff.size}")
