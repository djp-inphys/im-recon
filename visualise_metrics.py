"""Visualisations explaining the four reconstruction figure-of-merit metrics.

Generates five PNG files in the current directory:
  metrics_fig1_three_scenarios.png  — side-by-side scenario comparison
  metrics_fig2_mean_subtraction.png — Pearson's mean-subtraction failure
  metrics_fig3_outlier_robustness.png — Pearson vs Spearman under outliers
  metrics_fig4_poisson_weights.png  — why Poisson statistics matter
  metrics_fig5_score_landscape.png  — localisation sharpness comparison

Run from the project root:
    python visualise_metrics.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

np.random.seed(42)

# ── colour palette (one per metric) ──────────────────────────────────────────
PALETTE = {
    "Pearson":  "#e41a1c",
    "Cosine":   "#377eb8",
    "Spearman": "#4daf4a",
    "Poisson":  "#984ea3",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
})


# ── metric implementations (mirrors model.py) ─────────────────────────────────

def _pearson(O: np.ndarray, M: np.ndarray) -> float:
    s_O, s_M = np.std(O), np.std(M)
    if s_O == 0.0 or s_M == 0.0:
        return 0.0
    r = float(np.corrcoef(O, M)[0, 1])
    return r if np.isfinite(r) else 0.0


def _cosine(O: np.ndarray, M: np.ndarray) -> float:
    n_O, n_M = np.linalg.norm(O), np.linalg.norm(M)
    if n_O == 0.0 or n_M == 0.0:
        return 0.0
    return float(np.dot(O, M) / (n_O * n_M))


def _spearman(O: np.ndarray, M: np.ndarray) -> float:
    if np.all(O == O[0]) or np.all(M == M[0]):
        return 0.0
    rho, _ = spearmanr(O, M)
    return float(rho) if np.isfinite(rho) else 0.0


def _poisson_ll(O: np.ndarray, M: np.ndarray) -> float:
    t_O, t_M = np.sum(O), np.sum(M)
    if t_O == 0.0 or t_M == 0.0:
        return 0.0
    scale = t_O / t_M
    lam = scale * M
    valid = lam > 0.0
    if not np.any(valid):
        return 0.0
    ll = np.sum(O[valid] * np.log(lam[valid]) - lam[valid])
    return float(ll / valid.sum())


METRICS: dict[str, object] = {
    "Pearson":  _pearson,
    "Cosine":   _cosine,
    "Spearman": _spearman,
    "Poisson":  _poisson_ll,
}


def score_all(O: np.ndarray, M: np.ndarray) -> dict[str, float]:
    return {name: fn(O, M) for name, fn in METRICS.items()}  # type: ignore[operator]


# ── toy coded-aperture detector model ─────────────────────────────────────────
N_CH = 16    # channels (simplified for visualisation)
N_PH = 6     # phases
TRUE_POS = 8.0  # true source position (channel units, 0–15)


def make_model(source_x: float, n_ch: int = N_CH, n_ph: int = N_PH) -> np.ndarray:
    """Gaussian-spread coded-aperture response for source at *source_x*.

    Each phase shifts the effective illumination centre by n_ch/n_ph channels
    (simulating a rotating coded-aperture mask), giving each phase a distinct
    channel-count pattern.  The concatenated vector across all phases is the
    model prediction used in the real reconstruction.
    """
    sigma = 1.8
    ch = np.arange(n_ch, dtype=float)
    parts: list[np.ndarray] = []
    for p in range(n_ph):
        shift = (p * n_ch / n_ph) % n_ch
        centre = (source_x + shift) % n_ch
        d = np.abs(ch - centre)
        d = np.minimum(d, n_ch - d)           # wrap-around distance
        response = 80.0 * np.exp(-0.5 * (d / sigma) ** 2) + 5.0  # +5 baseline
        parts.append(response)
    return np.concatenate(parts)


# Ground-truth model and Poisson-noisy observation
true_model = make_model(TRUE_POS)
observed   = np.random.poisson(true_model).astype(float)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Three scenarios: what goes right and what goes wrong
# ═════════════════════════════════════════════════════════════════════════════
#
# Case A — well-matched structured vectors (correct source position)
# Case B — flat/background observation matched to constant model
#            → Pearson is undefined (std = 0) and falls back to 0
#            → Cosine correctly returns 1.0 (perfect shape match)
# Case C — correct match but one channel has a large noise spike
#            → Pearson score degrades substantially
#            → Spearman score is nearly unaffected (rank-based)
# ─────────────────────────────────────────────────────────────────────────────

SPIKE_IDX = 10

# Case A
O_A = observed.copy()
M_A = true_model.copy()

# Case B: flat observation (uniform background) vs flat model
O_B = np.full(N_CH * N_PH, 30.0)
M_B = np.full(N_CH * N_PH, 30.0)   # exactly constant → std = 0

# Case C: correct match + one large noise spike on the observation side
O_C = observed.copy()
O_C[SPIKE_IDX] += 400.0
M_C = true_model.copy()

CASES = [
    (O_A, M_A, "Case A\nCorrect structured match"),
    (O_B, M_B, "Case B\nFlat/background pair"),
    (O_C, M_C, "Case C\nCorrect match + outlier spike"),
]

# For the grouped score bar chart we normalise each metric independently to
# [0, 1] across the three cases so that relative discrimination is visible
# regardless of each metric's native scale.
raw_scores: list[dict[str, float]] = [score_all(O, M) for O, M, _ in CASES]
metric_names = list(PALETTE.keys())

def _normalise_scores(scores_list: list[dict[str, float]]) -> list[dict[str, float]]:
    """Normalise each metric to [0,1] across the case list."""
    norm: list[dict[str, float]] = [{} for _ in scores_list]
    for mname in metric_names:
        vals = np.array([s[mname] for s in scores_list])
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin if vmax != vmin else 1.0
        for i, v in enumerate(vals):
            norm[i][mname] = float((v - vmin) / span)
    return norm

norm_scores = _normalise_scores(raw_scores)

fig1, axes1 = plt.subplots(2, 3, figsize=(13, 8))
fig1.suptitle(
    "Figure 1 — Three scenarios: how the metrics respond differently",
    fontsize=12, fontweight="bold", y=0.99,
)

case_colors = ["#2ca02c", "#ff7f0e", "#d62728"]
case_labels  = ["Case A", "Case B", "Case C"]

for col, (O, M, title) in enumerate(CASES):
    ax_vec = axes1[0, col]
    x = np.arange(len(O))
    ax_vec.bar(x, O, color="#555", width=1.0, alpha=0.65, label="Observed O")
    ax_vec.step(x, M, where="mid", color="red", linewidth=1.8, label="Model M")
    if col == 2:
        ax_vec.annotate(
            "outlier\nspike",
            xy=(SPIKE_IDX, O[SPIKE_IDX]),
            xytext=(SPIKE_IDX + 10, O[SPIKE_IDX] * 0.85),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            fontsize=7, color="crimson",
        )
    ax_vec.set_title(title, fontsize=10)
    ax_vec.set_xlabel("Channel × Phase index")
    ax_vec.set_ylabel("Counts")
    ax_vec.legend(loc="upper right")

# Bottom row: grouped bar chart of normalised scores
ax_scores = axes1[1, 0]
x_pos = np.arange(len(metric_names))
width = 0.25
for ci, (clabel, ccolor) in enumerate(zip(case_labels, case_colors)):
    heights = [norm_scores[ci][m] for m in metric_names]
    bars = ax_scores.bar(
        x_pos + ci * width, heights, width,
        label=clabel, color=ccolor, alpha=0.85, edgecolor="black", linewidth=0.5,
    )
ax_scores.set_xticks(x_pos + width)
ax_scores.set_xticklabels(metric_names)
ax_scores.set_ylabel("Normalised score (0 = worst, 1 = best)")
ax_scores.set_title(
    "Metric scores normalised per-metric across the three cases\n"
    "(shows which cases each metric distinguishes correctly)",
    fontsize=9,
)
ax_scores.legend(loc="lower right")
ax_scores.set_ylim(0, 1.18)
ax_scores.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

# Annotations pointing out the interesting behaviours
ax_scores.annotate(
    "Pearson scores\nCase B as high as A\n(mean-subtraction\nproblem)",
    xy=(0 + width * 1, norm_scores[1]["Pearson"]),
    xytext=(0.05, 0.55),
    arrowprops=dict(arrowstyle="->", color="#e41a1c"),
    fontsize=7, color="#e41a1c",
)
ax_scores.annotate(
    "Cosine correctly\ngives Case B = 1\n(shape IS matched)",
    xy=(1 + width * 1, norm_scores[1]["Cosine"]),
    xytext=(1.1, 0.55),
    arrowprops=dict(arrowstyle="->", color="#377eb8"),
    fontsize=7, color="#377eb8",
)
ax_scores.annotate(
    "Spearman barely\ndegrades for\nCase C (outlier)",
    xy=(2 + width * 2, norm_scores[2]["Spearman"]),
    xytext=(2.3, 0.35),
    arrowprops=dict(arrowstyle="->", color="#4daf4a"),
    fontsize=7, color="#4daf4a",
)

# Hide the unused bottom-right cells
for col in [1, 2]:
    axes1[1, col].set_visible(False)

plt.tight_layout()
fig1.savefig("metrics_fig1_three_scenarios.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved  metrics_fig1_three_scenarios.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Mean-subtraction: Pearson vs Cosine
# ═════════════════════════════════════════════════════════════════════════════
#
# Shows what Pearson literally operates on (mean-centred vectors) vs what
# Cosine operates on (raw vectors).  For flat data Pearson's centering
# amplifies noise, while Cosine sees the true alignment.
# ─────────────────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10))
fig2.suptitle(
    "Figure 2 — The mean-subtraction problem: what Pearson actually measures",
    fontsize=12, fontweight="bold",
)

panel_cases = [
    ("Structured pair\n(correct source position)", O_A[:N_CH], M_A[:N_CH]),
    ("Flat/background pair\n(uniform counts, no source structure)", O_B[:N_CH], M_B[:N_CH]),
]
titles_row = ["Raw vectors (what Cosine sees)", "Mean-centred (what Pearson sees)", "Scores"]

for col, (label, O, M) in enumerate(panel_cases):
    ax_raw  = axes2[0, col]
    ax_cent = axes2[1, col]
    ax_scr  = axes2[2, col]

    x = np.arange(len(O))

    # Row 0: raw
    ax_raw.bar(x, O, color="#444", alpha=0.7, width=0.9, label="Observed O")
    ax_raw.step(x, M, where="mid", color="crimson", linewidth=2.0, label="Model M")
    ax_raw.set_title(f"{label}\n{titles_row[0]}", fontsize=9)
    ax_raw.set_ylabel("Counts")
    ax_raw.legend(loc="upper right")

    # Row 1: mean-centred
    O_c = O - O.mean()
    M_c = M - M.mean()
    ax_cent.bar(x, O_c, color="#444", alpha=0.7, width=0.9, label="O − mean(O)")
    ax_cent.step(x, M_c, where="mid", color="crimson", linewidth=2.0, label="M − mean(M)")
    ax_cent.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_cent.set_title(titles_row[1], fontsize=9)
    ax_cent.set_ylabel("Mean-centred counts")
    ax_cent.set_xlabel("Detector channel")
    ax_cent.legend(loc="upper right")

    # Row 2: scores
    p = _pearson(O, M)
    c = _cosine(O, M)
    s = _spearman(O, M)
    pl = _poisson_ll(O, M)

    bar_names  = ["Pearson", "Cosine", "Spearman"]
    bar_vals   = [p, c, s]
    bar_colors = [PALETTE["Pearson"], PALETTE["Cosine"], PALETTE["Spearman"]]

    bars = ax_scr.bar(bar_names, bar_vals, color=bar_colors, edgecolor="black",
                      linewidth=0.6, alpha=0.9)
    ax_scr.axhline(0, color="black", linewidth=0.8)
    ax_scr.set_ylabel("Score (raw, not normalised)")
    ax_scr.set_title("Scores (raw native scale)", fontsize=9)
    ax_scr.set_ylim(-0.3, 1.25)
    for bar, val in zip(bars, bar_vals):
        ax_scr.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 0) + 0.04,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    # Highlight the problematic Pearson bar for the flat case
    if col == 1:
        ax_scr.annotate(
            "std(M)=0 →\nPearson undefined\n→ returns 0.0\n(misleading!)",
            xy=(0, p), xytext=(0.7, 0.5),
            arrowprops=dict(arrowstyle="->", color="#e41a1c"),
            fontsize=7, color="#e41a1c",
        )
        ax_scr.annotate(
            "Cosine = 1.0\ncorrectly detects\nshape agreement",
            xy=(1, c), xytext=(1.3, 0.65),
            arrowprops=dict(arrowstyle="->", color="#377eb8"),
            fontsize=7, color="#377eb8",
        )

plt.tight_layout()
fig2.savefig("metrics_fig2_mean_subtraction.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved  metrics_fig2_mean_subtraction.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Outlier robustness in the all-anodes × all-CA-phases scatter
# ═════════════════════════════════════════════════════════════════════════════
#
# A synthetic scatter replicates the real reconstruction format:
#   64 anodes × 7 CA phases = 448 (model power, observed count) pairs
#   Each anode has its own non-CA floor (Y-offset nuisance parameter).
#   All anodes share the same true via-CA slope.
#
# A single point is spiked at increasing amplitudes.  Left and middle panels
# show the scatter before/after; right panel shows how each metric degrades.
# ─────────────────────────────────────────────────────────────────────────────

_N_AN = 64
_N_PH = 7
_TRUE_SLOPE = 0.12   # via-CA slope (counts per unit model power)

rng3 = np.random.default_rng(42)

# Model powers: realistic range matching real camera data
_mp_grid = rng3.uniform(5_000, 38_000, (_N_AN, _N_PH))

# Per-anode non-CA floors: the Y-offset nuisance parameter for each anode
_floors = rng3.uniform(3_500, 7_500, _N_AN)

# Expected counts = via-CA component + floor; add Poisson noise
_expected = _TRUE_SLOPE * _mp_grid + _floors[:, np.newaxis]
_obs_grid  = rng3.poisson(np.maximum(_expected, 1).astype(int)).astype(float)

# Flatten to 1-D scatter vectors
x_sc = _mp_grid.ravel()     # model power  (448 values)
y_sc = _obs_grid.ravel()    # observed counts (448 values)

# Spike the 10 points with the LOWEST model power: they have low x-rank, so
# adding counts moves them to the top of the y-rank range, creating large
# rank mismatches that stress Pearson more than Cosine / Spearman.
_N_SPIKES = 10
_spike_idxs = np.argsort(x_sc)[:_N_SPIKES]


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) from OLS fitted to the scatter."""
    n = float(len(x))
    sx, sy = x.sum(), y.sum()
    sxx, sxy = (x * x).sum(), (x * y).sum()
    d = n * sxx - sx * sx
    if abs(d) < 1e-12:
        return 0.0, float(sy / n)
    slope = (n * sxy - sx * sy) / d
    return float(slope), float((sy - slope * sx) / n)


spike_sizes_3 = [0, 200, 600, 1_500, 3_000, 6_000]
_ols_slopes, _ols_intercepts = [], []
_m3_scores: dict[str, list[float]] = {m: [] for m in METRICS}

for _sp in spike_sizes_3:
    _ys = y_sc.copy()
    _ys[_spike_idxs] += _sp          # spike ALL 10 low-power points
    _sl, _ic = _ols_slope(x_sc, _ys)
    _ols_slopes.append(_sl)
    _ols_intercepts.append(_ic)
    for _mname, _mfn in METRICS.items():
        _m3_scores[_mname].append(_mfn(_ys, x_sc))  # type: ignore[operator]

# Normalise each metric so that the CLEAN (spike=0) case is always 1.0 and
# the most-degraded case is 0.0.  This keeps the y-axis meaning consistent:
# 1.0 = baseline, dropping toward 0 = degraded.
def _norm3(vals: list[float]) -> np.ndarray:
    v = np.array(vals, dtype=float)
    clean = v[0]
    worst = v.min() if v[-1] < v[0] else v.max()   # direction of degradation
    span = abs(clean - worst)
    if span < 1e-12:
        return np.ones_like(v)
    # Map: clean → 1.0, worst → 0.0
    return (v - worst) / span if clean > worst else 1.0 - (v - clean) / span


fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(
    "Figure 3 — Outlier robustness: all-anodes × all-CA-phases scatter\n"
    "(synthetic: 64 anodes × 7 phases, per-anode floors, shared via-CA slope)",
    fontsize=11, fontweight="bold",
)

_xfit = np.linspace(x_sc.min(), x_sc.max(), 300)

# ── Left: clean scatter ───────────────────────────────────────────────────────
ax3a.scatter(x_sc, y_sc, s=7, alpha=0.35, color="#377eb8",
             label="448 points  (64 anodes × 7 phases)")
_sl0, _ic0 = _ols_slopes[0], _ols_intercepts[0]
ax3a.plot(_xfit, _sl0 * _xfit + _ic0, "--", color="#111",
          linewidth=2.0, label=f"OLS slope = {_sl0:.4f}")
ax3a.set_title("Clean scatter\n(no outlier)")
ax3a.set_xlabel("Calculated model power")
ax3a.set_ylabel("Observed anode counts")
ax3a.legend(fontsize=7)
ax3a.grid(alpha=0.25)

# ── Middle: scatter with large spikes on 10 lowest-model-power points ────────
_y_big = y_sc.copy()
_y_big[_spike_idxs] += spike_sizes_3[-1]   # +6000 counts on 10 points

# Plot non-spiked points first, then highlight the spiked ones
_mask = np.zeros(len(x_sc), dtype=bool)
_mask[_spike_idxs] = True
ax3b.scatter(x_sc[~_mask], _y_big[~_mask], s=7, alpha=0.35, color="#377eb8")
ax3b.scatter(
    x_sc[_mask], _y_big[_mask],
    s=90, color="crimson", zorder=6, marker="*",
    label=f"+{spike_sizes_3[-1]:,} counts (10 outliers)",
)
_sl_big, _ic_big = _ols_slopes[-1], _ols_intercepts[-1]
ax3b.plot(_xfit, _sl_big * _xfit + _ic_big, "--", color="#111",
          linewidth=2.0, label=f"OLS slope = {_sl_big:.4f}  (degraded)")
ax3b.plot(_xfit, _sl0 * _xfit + _ic0, ":", color="#888",
          linewidth=1.4, label=f"Clean slope = {_sl0:.4f}")
# Annotate the cluster of spiked points using axes-fraction coords
_ann_xy = (x_sc[_spike_idxs].mean(), _y_big[_spike_idxs].max())
ax3b.annotate(
    f"10 points each spiked\nby +{spike_sizes_3[-1]:,} counts\n(lowest model-power points)",
    xy=_ann_xy,
    xycoords="data",
    xytext=(0.28, 0.80),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.4),
    fontsize=8, color="crimson",
)
ax3b.set_title(
    f"Spiked scatter\n(10 low-power points each raised by +{spike_sizes_3[-1]:,} counts)"
)
ax3b.set_xlabel("Calculated model power")
ax3b.set_ylabel("Observed anode counts")
ax3b.legend(fontsize=7)
ax3b.grid(alpha=0.25)

# ── Right: all-metric normalised score vs spike size ─────────────────────────
_norm_ols = _norm3(_ols_slopes)
ax3c.plot(spike_sizes_3, _norm_ols, "^--", color="#555",
          linewidth=1.8, markersize=6, label="OLS slope")

for _mname, _mvals in _m3_scores.items():
    _nm = _norm3(_mvals)
    ax3c.plot(spike_sizes_3, _nm, "o-", color=PALETTE[_mname],
              linewidth=2.0, markersize=6, label=_mname)
    # Label the final (most degraded) value
    ax3c.annotate(
        f"{_nm[-1]:.2f}",
        (spike_sizes_3[-1], _nm[-1]),
        textcoords="offset points", xytext=(4, 0),
        fontsize=7, color=PALETTE[_mname],
    )

ax3c.set_xlabel("Outlier spike amplitude (extra counts on one of 448 points)")
ax3c.set_ylabel("Normalised score\n(1.0 = clean baseline  ·  0.0 = most degraded)")
ax3c.set_title(
    "Score vs outlier size\n"
    "Closer to 1.0 at right edge = more robust"
)
ax3c.legend(fontsize=7)
ax3c.grid(alpha=0.3)
ax3c.set_ylim(-0.15, 1.25)

plt.tight_layout()
fig3.savefig("metrics_fig3_outlier_robustness.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved  metrics_fig3_outlier_robustness.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Poisson statistics: why variance scales with count level
# ═════════════════════════════════════════════════════════════════════════════
#
# Pearson implicitly assumes uniform (Gaussian) errors.  Count data obey
# Poisson statistics: Var(X) = E(X).  The two approaches give different
# weights to high- vs low-count bins, and only the Poisson weighting is
# physically correct.
# ─────────────────────────────────────────────────────────────────────────────

n_samples = 2000
rates_demo = [5.0, 50.0, 200.0]

fig4, axes4 = plt.subplots(1, 3, figsize=(13, 5))
fig4.suptitle(
    "Figure 4 — Poisson statistics: noise model assumed by each metric",
    fontsize=12, fontweight="bold",
)

for ax, rate in zip(axes4, rates_demo):
    samples = np.random.poisson(rate, n_samples).astype(float)
    bins = np.arange(max(0, rate - 4.5 * np.sqrt(rate)),
                     rate + 4.5 * np.sqrt(rate) + 1) - 0.5
    ax.hist(samples, bins=bins, color="steelblue", edgecolor="white",
            alpha=0.75, label="Poisson samples", density=True)

    poisson_sigma = np.sqrt(rate)
    uniform_sigma = np.sqrt(np.mean(rates_demo))  # fixed width — Pearson's assumption

    # Poisson-correct error band
    ax.axvspan(rate - poisson_sigma, rate + poisson_sigma,
               alpha=0.25, color=PALETTE["Poisson"],
               label=f"±1σ Poisson = ±{poisson_sigma:.1f}")
    # Uniform error band (Pearson-style)
    ax.axvspan(rate - uniform_sigma, rate + uniform_sigma,
               alpha=0.20, color=PALETTE["Pearson"], linestyle="--",
               label=f"±1σ uniform = ±{uniform_sigma:.1f}")
    ax.axvline(rate, color="black", linewidth=1.5, linestyle="--", label=f"True λ = {rate:.0f}")

    cv = poisson_sigma / rate * 100
    ax.set_title(f"λ = {rate:.0f} counts/bin\n(σ/μ = {cv:.1f} %)", fontsize=10)
    ax.set_xlabel("Observed count")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper right", fontsize=7)

fig4.text(
    0.5, 0.01,
    "At low counts (left) Pearson's fixed-width error band (red) is far too wide — "
    "it under-weights bins that are actually precise.\n"
    "At high counts (right) it becomes too narrow — over-weighting uncertain bins.  "
    "The Poisson metric (purple) uses the physically correct √λ width throughout.",
    ha="center", fontsize=8, style="italic",
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.7),
)
plt.tight_layout(rect=[0, 0.09, 1, 1])
fig4.savefig("metrics_fig4_poisson_weights.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved  metrics_fig4_poisson_weights.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Source localisation score landscape
# ═════════════════════════════════════════════════════════════════════════════
#
# For each candidate source offset from the true position (-6 to +6 channels),
# all four metrics are evaluated against the Poisson-noisy observation.
# A sharper peak (smaller FWHM) means better ability to localise the source.
# ─────────────────────────────────────────────────────────────────────────────

offsets  = np.linspace(-7, 7, 200)
raw_scr: dict[str, list[float]] = {m: [] for m in METRICS}

for dx in offsets:
    M_cand = make_model(TRUE_POS + dx)
    for name, fn in METRICS.items():
        raw_scr[name].append(fn(observed, M_cand))  # type: ignore[operator]

# Normalise each metric to [0, 1] across the offset range for fair comparison
norm_scr: dict[str, np.ndarray] = {}
for name in METRICS:
    v = np.array(raw_scr[name])
    vmin, vmax = v.min(), v.max()
    norm_scr[name] = (v - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(v)


def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a normalised (0–1) curve."""
    above = x[y >= 0.5]
    if len(above) < 2:
        return float("nan")
    return float(above[-1] - above[0])


fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle(
    "Figure 5 — Source localisation score landscape "
    "(score vs position offset from true source)",
    fontsize=12, fontweight="bold",
)

# Left: score curves
for name, color in PALETTE.items():
    ax5a.plot(offsets, norm_scr[name], color=color, linewidth=2.2, label=name)
ax5a.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
             label="True position (offset = 0)")
ax5a.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="Half-maximum (0.5)")
ax5a.fill_between(offsets, 0, norm_scr["Poisson"],
                  alpha=0.08, color=PALETTE["Poisson"])
ax5a.set_xlabel("Source position offset (channels)")
ax5a.set_ylabel("Normalised score (0 = worst fit, 1 = best fit)")
ax5a.set_title("Score landscape (each metric normalised to [0,1])\n"
               "Narrower peak → better localisation")
ax5a.legend()
ax5a.grid(alpha=0.25)
ax5a.set_xlim(offsets[0], offsets[-1])

# Right: FWHM bar chart
fwhm_vals = {name: fwhm(offsets, norm_scr[name]) for name in METRICS}
names = list(fwhm_vals.keys())
vals  = [fwhm_vals[n] for n in names]
colors = [PALETTE[n] for n in names]

bars = ax5b.bar(names, vals, color=colors, edgecolor="black", linewidth=0.7, alpha=0.9)
for bar, val in zip(bars, vals):
    if np.isfinite(val):
        ax5b.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f} ch",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
ax5b.set_ylabel("FWHM (channels)")
ax5b.set_title("Localisation sharpness (FWHM of score peak)\n"
               "Lower FWHM = more precise source localisation")
ax5b.grid(axis="y", alpha=0.3)
ax5b.set_ylim(0, max(v for v in vals if np.isfinite(v)) * 1.3)

# Shade the best-performing bar
best = min((v, n) for n, v in fwhm_vals.items() if np.isfinite(v))
best_name = best[1]
for bar, name in zip(bars, names):
    if name == best_name:
        bar.set_edgecolor("gold")
        bar.set_linewidth(2.5)
        ax5b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            "best",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold",
        )

plt.tight_layout()
fig5.savefig("metrics_fig5_score_landscape.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Saved  metrics_fig5_score_landscape.png")

print("\nAll five figures generated successfully.")
