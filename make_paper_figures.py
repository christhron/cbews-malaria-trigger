"""One-off figures for paper_trigger/cbews_trigger.tex, built from the cached
10-year fine grid (trigger_grids_10yr_fine.pkl): (1) trigger and matched-constant
percent-of-cases-averted heatmaps side by side sharing one colorbar, (2) the
active-fraction heatmap expressed as a proportion (0-1) instead of percent,
(3) the difference in percent-of-cases-averted between the reported-signal
(iota_hat) and perfect-information (lambda) triggers, and (4) the fair,
effort-normalised counterpart of (3): the relative difference in cases averted
per active person-day between the two drivers, plus a LaTeX table of active
fraction and rate at S=20% across the threshold grid. Reuses sweep_trigger.py's
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

# ---- (4) fair comparison: relative difference in cases averted per active
#          person-day (Z_rate_trig), reported vs perfect-info, at matched
#          threshold percentile and suppression level ----
af_i, af_p = di["Z_active_frac"], dp["Z_active_frac"]
rate_i, rate_p = di["Z_rate_trig"], dp["Z_rate_trig"]
rate_rel = 100.0 * (rate_i - rate_p) / rate_p   # >0: reported more efficient per active day
m2 = float(np.abs(rate_rel).max())

fig, ax = plt.subplots(figsize=(6.6, 4.2))
im = ax.imshow(rate_rel, aspect="auto", origin="lower", cmap=DIV_CMAP, vmin=-m2, vmax=m2)
ax.set_xticks(range(len(pctl))); ax.set_xticklabels([str(q) for q in pctl])
ax.set_yticks(range(len(Slev))); ax.set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(rate_rel.shape[0]):
    for j in range(rate_rel.shape[1]):
        v = rate_rel[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.65 * m2 else INK)
ax.set_xlabel("threshold percentile")
ax.set_ylabel("suppression level $S$")
ax.set_title(r"Cases averted per active person-day: $\hat\iota_i$ vs.\ $\lambda_i$ (\% difference)",
             color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("relative difference in rate (%)", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_rate_diff_iota_vs_phi_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_rate_diff_iota_vs_phi_10yr_fine.png")
print(f"  rate_rel range (%): {rate_rel.min():+.2f} to {rate_rel.max():+.2f}, "
      f"median {np.median(rate_rel):+.2f}; cells reported>perfect-info: "
      f"{(rate_rel > 0).sum()}/{rate_rel.size}")
af_rel = 100.0 * (af_i - af_p) / af_p
print(f"  active-fraction relative difference (%): {af_rel.min():+.2f} to {af_rel.max():+.2f}, "
      f"median {np.median(af_rel):+.2f}; cells reported active_frac larger: "
      f"{(af_rel > 0).sum()}/{af_rel.size}")

# ---- Table: active fraction and rate, both drivers, at S=20% across the
#      threshold grid (mirrors Table 2's format; values pulled programmatically
#      to avoid hand-transcription errors) ----
s20 = list(Slev).index(0.20)
print("\nLaTeX table body, S=20%, active fraction and rate by threshold percentile:")
print("Threshold percentile & " + " & ".join(str(q) for q in pctl) + r" \\")
print(r"\midrule")
print("Active fraction, $\\hat\\iota_i$ & " +
      " & ".join(f"{v:.3f}" for v in af_i[s20]) + r" \\")
print("Active fraction, $\\lambda_i$ & " +
      " & ".join(f"{v:.3f}" for v in af_p[s20]) + r" \\")
print(r"\addlinespace")
print("Rate, $\\hat\\iota_i$ (cases/active-day) & " +
      " & ".join(f"{v:.4f}" for v in rate_i[s20]) + r" \\")
print("Rate, $\\lambda_i$ (cases/active-day) & " +
      " & ".join(f"{v:.4f}" for v in rate_p[s20]) + r" \\")
print(r"Relative difference in rate (\%) & " +
      " & ".join(f"{v:+.1f}" for v in rate_rel[s20]) + r" \\")
