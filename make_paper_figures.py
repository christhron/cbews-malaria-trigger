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

from sweep_trigger import (AXIS, DIV_CMAP, INK, INK_SECONDARY, SEQ_CMAP,
                            cross_driver_ci, pct_averted_diff_statistic, rate_diff_statistic)

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

# ---- CI on the trigger-vs-matched-constant gap (already bootstrapped inside
#      sweep_trigger_grid; convert case counts to percentage points here) ----
diff_pp = 100.0 * d["Z_cases_averted_diff"] / off
diff_pp_lo = 100.0 * d["Z_cases_averted_diff_lo"] / off
diff_pp_hi = 100.0 * d["Z_cases_averted_diff_hi"] / off
excl0 = (diff_pp_lo > 0) | (diff_pp_hi < 0)
print(f"trigger-vs-const gap (pp): {diff_pp.min():+.2f} to {diff_pp.max():+.2f}; "
      f"95% CI excludes zero at {excl0.sum()}/{excl0.size} cells "
      f"(200 replicates, paired bootstrap)")

# ---- (3) difference in % cases averted: reported (iota_hat) minus perfect-info (lambda),
#          now with a paired-bootstrap significance marker per cell ----
di, dp = grids["iota_hat"], grids["phi"]
assert di["off_symp"] == dp["off_symp"], "off baselines differ; cannot difference percentages"
pct_i = 100.0 * di["Z_cases_averted"] / di["off_symp"]
pct_p = 100.0 * dp["Z_cases_averted"] / dp["off_symp"]
diff = pct_i - pct_p                       # > 0: reported signal averts a larger share than perfect info
pctl = (1, 20, 40, 55, 70, 85, 99)        # PERCENTILES, from sweep_trigger_10yr_fine.py
Slev = di["suppression_levels"]
m = float(np.abs(diff).max())

_, pct_diff_lo, pct_diff_hi = cross_driver_ci(di, dp, pct_averted_diff_statistic, n_boot=2000, seed=3001)
pct_diff_excl0 = (pct_diff_lo > 0) | (pct_diff_hi < 0)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
im = ax.imshow(diff, aspect="auto", origin="lower", cmap=DIV_CMAP, vmin=-m, vmax=m)
ax.set_xticks(range(len(pctl))); ax.set_xticklabels([str(q) for q in pctl])
ax.set_yticks(range(len(Slev))); ax.set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(diff.shape[0]):
    for j in range(diff.shape[1]):
        v = diff[i, j]
        star = "*" if pct_diff_excl0[i, j] else ""
        ax.text(j, i, f"{v:+.1f}{star}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.65 * m else INK)
ax.set_xlabel("threshold percentile")
ax.set_ylabel("suppression level $S$")
ax.set_title(r"$\hat\iota_i$-driven minus $\lambda_i$-driven (% of cases averted)",
             color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("percentage-point difference (* = 95\\% CI excludes 0)", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_cases_averted_pct_diff_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_cases_averted_pct_diff_10yr_fine.png")
print(f"  diff range (pp): {diff.min():+.2f} to {diff.max():+.2f}, mean {diff.mean():+.2f}; "
      f"cells with reported < perfect-info: {(diff < 0).sum()}/{diff.size}; "
      f"95% CI excludes zero at {pct_diff_excl0.sum()}/{pct_diff_excl0.size} cells")

# ---- (4) fair comparison: relative difference in cases averted per active
#          person-day (Z_rate_trig), reported vs perfect-info, at matched
#          threshold percentile and suppression level -- with a paired-bootstrap
#          significance marker per cell (cross_driver_ci; this is what actually
#          tells us whether the small rate gap is real or Monte Carlo noise) ----
af_i, af_p = di["Z_active_frac"], dp["Z_active_frac"]
rate_i, rate_p = di["Z_rate_trig"], dp["Z_rate_trig"]
rate_rel = 100.0 * (rate_i - rate_p) / rate_p   # >0: reported more efficient per active day
m2 = float(np.abs(rate_rel).max())

rate_diff_abs, rate_diff_lo, rate_diff_hi = cross_driver_ci(
    di, dp, rate_diff_statistic,
    guard=lambda i, j: np.isfinite(rate_i[i, j]) and np.isfinite(rate_p[i, j]),
    n_boot=2000, seed=4001)
rate_diff_excl0 = (rate_diff_lo > 0) | (rate_diff_hi < 0)

fig, ax = plt.subplots(figsize=(6.6, 4.2))
im = ax.imshow(rate_rel, aspect="auto", origin="lower", cmap=DIV_CMAP, vmin=-m2, vmax=m2)
ax.set_xticks(range(len(pctl))); ax.set_xticklabels([str(q) for q in pctl])
ax.set_yticks(range(len(Slev))); ax.set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(rate_rel.shape[0]):
    for j in range(rate_rel.shape[1]):
        v = rate_rel[i, j]
        star = "*" if rate_diff_excl0[i, j] else ""
        ax.text(j, i, f"{v:+.1f}{star}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.65 * m2 else INK)
ax.set_xlabel("threshold percentile")
ax.set_ylabel("suppression level $S$")
ax.set_title(r"Cases averted per active person-day: $\hat\iota_i$ vs.\ $\lambda_i$ (\% difference)",
             color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("relative difference in rate (%; * = 95\\% CI excludes 0)", color=INK_SECONDARY)
cb.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_rate_diff_iota_vs_phi_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_rate_diff_iota_vs_phi_10yr_fine.png")
print(f"  rate_rel range (%): {rate_rel.min():+.2f} to {rate_rel.max():+.2f}, "
      f"median {np.median(rate_rel):+.2f}; cells reported>perfect-info: "
      f"{(rate_rel > 0).sum()}/{rate_rel.size}; "
      f"95% CI excludes zero at {rate_diff_excl0.sum()}/{rate_diff_excl0.size} cells "
      f"(of which reported<perfect-info significant: "
      f"{((rate_diff_abs < 0) & rate_diff_excl0).sum()}, "
      f"reported>perfect-info significant: {((rate_diff_abs > 0) & rate_diff_excl0).sum()})")
af_rel = 100.0 * (af_i - af_p) / af_p
print(f"  active-fraction relative difference (%): {af_rel.min():+.2f} to {af_rel.max():+.2f}, "
      f"median {np.median(af_rel):+.2f}; cells reported active_frac larger: "
      f"{(af_rel > 0).sum()}/{af_rel.size}")

# ---- (5) alert-episode statistics ($\hat\iota_i$-driven): episodes per town
#          and mean episode length, over the (S, threshold) grid ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
epi = di["Z_episodes_per_town"]
im0 = axes[0].imshow(epi, aspect="auto", origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=epi.max())
axes[0].set_xticks(range(len(pctl))); axes[0].set_xticklabels([str(q) for q in pctl])
axes[0].set_yticks(range(len(Slev))); axes[0].set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(epi.shape[0]):
    for j in range(epi.shape[1]):
        v = epi[i, j]
        axes[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                      color="white" if v > epi.max() / 2 else INK)
axes[0].set_xlabel("threshold percentile"); axes[0].set_ylabel("suppression level $S$")
axes[0].set_title("Alert episodes per town (10 yr)", color=INK, loc="left")
cb0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
cb0.set_label("mean distinct 'on' episodes per town", color=INK_SECONDARY)
cb0.outline.set_edgecolor(AXIS)

# the loosest-threshold, weakest-suppression corner is near-saturated (active
# almost the entire window, occasionally for the whole 10 years in a given
# town/replicate) and its mean episode length is a right-skewed outlier
# dominated by a handful of near-full-window "episodes" -- excluded from the
# color scale (still plotted, annotated) so it doesn't wash out the rest.
epi_len_days = di["Z_episode_length"] * 21.0
mask = np.ones_like(epi_len_days, dtype=bool)
mask[-1, 0] = False   # S=10%, 1st percentile: the saturated corner
vmax_len = epi_len_days[mask].max()
im1 = axes[1].imshow(np.where(mask, epi_len_days, np.nan), aspect="auto", origin="lower",
                      cmap=SEQ_CMAP, vmin=0, vmax=vmax_len)
axes[1].set_xticks(range(len(pctl))); axes[1].set_xticklabels([str(q) for q in pctl])
axes[1].set_yticks(range(len(Slev))); axes[1].set_yticklabels([f"{s:.2f}" for s in Slev])
for i in range(epi_len_days.shape[0]):
    for j in range(epi_len_days.shape[1]):
        v = epi_len_days[i, j]
        label = f"{v:.0f}" if mask[i, j] else f"{v:.0f}$^\\dagger$"
        color = INK if not mask[i, j] else ("white" if v > vmax_len / 2 else INK)
        axes[1].text(j, i, label, ha="center", va="center", fontsize=8, color=color)
axes[1].set_xlabel("threshold percentile")
axes[1].set_title("Mean alert-episode length (days)", color=INK, loc="left")
cb1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
cb1.set_label("days ($\\dagger$ = saturated outlier, off color scale)", color=INK_SECONDARY)
cb1.outline.set_edgecolor(AXIS)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_episodes_iota_hat_10yr_fine.png", dpi=200)
plt.close(fig)
print("wrote fig_trigger_episodes_iota_hat_10yr_fine.png")
print(f"  episode length range excl. saturated corner (days): "
      f"{epi_len_days[mask].min():.1f} to {epi_len_days[mask].max():.1f}; "
      f"saturated corner (S=10%, 1st pct): {epi_len_days[-1,0]:.0f} days")

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
      " & ".join(f"{v:+.1f}{'*' if rate_diff_excl0[s20,k] else ''}"
                 for k, v in enumerate(rate_rel[s20])) + r" \\")

# ---- Table: trigger-vs-matched-constant cases-averted gap with 95% CI, at
#      S=20% across the threshold grid ($\hat\iota_i$-driven) ----
print("\nLaTeX table body, S=20%, trigger vs. matched constant with 95% CI (pp):")
print("Threshold percentile & " + " & ".join(str(q) for q in pctl) + r" \\")
print(r"\midrule")
print("Gap, trigger $-$ constant (pp) & " +
      " & ".join(f"{v:+.2f}" for v in diff_pp[s20]) + r" \\")
print("95\\% CI & " +
      " & ".join(f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in zip(diff_pp_lo[s20], diff_pp_hi[s20]))
      + r" \\")

# ---- Table 2 (tab:rate-ratio): trigger vs. matched constant, per-active-day
#      rate and their ratio, at S=20% across the threshold grid
#      ($\hat\iota_i$-driven) -- values pulled programmatically from the pkl
#      to remove the earlier hand-transcription. ----
rate_trig_s20 = d["Z_rate_trig"][s20]
rate_const_s20 = d["Z_rate_const"][s20]
ratio_s20 = rate_trig_s20 / rate_const_s20
print("\nLaTeX table body, Table 2 (tab:rate-ratio), S=20%:")
print("Threshold percentile & " + " & ".join(f"{q}{('st' if q==1 else 'th')}" for q in pctl) + r" \\")
print(r"\midrule")
print("Rate, trigger (cases/active-day) & " +
      " & ".join(f"{v:.4f}" for v in rate_trig_s20) + r" \\")
print("Rate, matched constant           & " +
      " & ".join(f"{v:.4f}" for v in rate_const_s20) + r" \\")
print("Ratio (trigger / constant)       & " +
      " & ".join(f"{v:.1f}$\\times$" for v in ratio_s20) + r" \\")

# ---- (6) iso-effectiveness curve: for a target % case reduction, the minimum
#          active fraction anywhere on the (S, threshold) grid that achieves at
#          least that reduction ($\hat\iota_i$-driven). Answers research
#          question 2 directly, without interpolation -- just the best of the
#          35 grid cells actually run, so every point is an achieved result. ----
symp_pct = d["Z_symp_pct"]      # (n_s, n_t), same grid as af_i etc.
active_frac = d["Z_active_frac"]
targets = np.array([5, 10, 15, 20, 25, 30, 35])
iso_active_frac = np.full(targets.shape, np.nan)
iso_S = np.full(targets.shape, np.nan)
iso_Tidx = np.full(targets.shape, -1, dtype=int)
for k, target in enumerate(targets):
    mask = symp_pct >= target
    if not mask.any():
        continue
    af_masked = np.where(mask, active_frac, np.inf)
    i, j = np.unravel_index(np.argmin(af_masked), af_masked.shape)
    iso_active_frac[k] = active_frac[i, j]
    iso_S[k] = Slev[i]
    iso_Tidx[k] = j

fig, ax = plt.subplots(figsize=(6.0, 4.2))
valid = np.isfinite(iso_active_frac)
ax.plot(targets[valid], 100.0 * iso_active_frac[valid], "o-", color=INK, markersize=5)
for k in np.where(valid)[0]:
    ax.annotate(f"$S$={iso_S[k]:.2f}\np{pctl[iso_Tidx[k]]}",
                 (targets[k], 100.0 * iso_active_frac[k]), textcoords="offset points",
                 xytext=(6, 6), fontsize=7, color=INK_SECONDARY)
ax.set_xlabel("target reduction in symptomatic cases (%)")
ax.set_ylabel("minimum active fraction achieving it (%)")
ax.set_title(r"Cheapest way to reach a case-reduction target ($\hat\iota_i$-driven)",
             color=INK, loc="left")
ax.set_yscale("log")
ax.grid(True, which="both", axis="y", color="#e1e0d9", linewidth=0.5)
fig.tight_layout()
fig.savefig(outdir / "fig_trigger_iso_effectiveness_iota_hat_10yr_fine.png", dpi=200)
plt.close(fig)
print("\nwrote fig_trigger_iso_effectiveness_iota_hat_10yr_fine.png")
for k in range(len(targets)):
    if valid[k]:
        print(f"  >= {targets[k]}% reduction: min active fraction "
              f"{100*iso_active_frac[k]:.2f}% at S={iso_S[k]:.2f}, "
              f"threshold={pctl[iso_Tidx[k]]}th percentile")
    else:
        print(f"  >= {targets[k]}% reduction: not achieved anywhere on this grid")
