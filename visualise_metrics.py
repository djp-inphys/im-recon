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
# FIGURE 3 — Outlier robustness: Pearson vs Spearman
# ═════════════════════════════════════════════════════════════════════════════
#
# A single-channel noise spike is added at increasing amplitudes.  The
# Pearson score falls rapidly because the spike inflates sigma(O), shrinking
# the correlation denominator.  Spearman's rank transformation caps the
# impact at a single rank displacement.
# ─────────────────────────────────────────────────────────────────────────────

spike_sizes = [0, 100, 300, 600]
p_scores  = []
sp_scores = []
for spike in spike_sizes:
    O_s = observed.copy()
    O_s[SPIKE_IDX] += spike
    p_scores.append(_pearson(O_s, true_model))
    sp_scores.append(_spearman(O_s, true_model))

fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle(
    "Figure 3 — Outlier robustness: Pearson vs Spearman",
    fontsize=12, fontweight="bold",
)

# Left: vector with no spike
ax3a.bar(np.arange(len(observed)), observed, color="#555", width=1.0, alpha=0.7, label="Observed O")
ax3a.step(np.arange(len(true_model)), true_model, where="mid", color="crimson",
          linewidth=1.8, label="Model M (truth)")
ax3a.set_title("Observation vs model\n(no outlier)")
ax3a.set_xlabel("Channel × Phase index")
ax3a.set_ylabel("Counts")
ax3a.legend()

# Middle: vector with large spike
O_spike = observed.copy()
O_spike[SPIKE_IDX] += 600
ax3b.bar(np.arange(len(O_spike)), O_spike, color="#555", width=1.0, alpha=0.7, label="Observed O")
ax3b.step(np.arange(len(true_model)), true_model, where="mid", color="crimson",
          linewidth=1.8, label="Model M (truth)")
ax3b.annotate(
    "+600 counts\n(cosmic-ray spike)",
    xy=(SPIKE_IDX, O_spike[SPIKE_IDX]),
    xytext=(SPIKE_IDX + 12, O_spike[SPIKE_IDX] * 0.9),
    arrowprops=dict(arrowstyle="->", color="crimson"),
    fontsize=8, color="crimson",
)
ax3b.set_title("Observation vs model\n(large outlier at one channel)")
ax3b.set_xlabel("Channel × Phase index")
ax3b.set_ylabel("Counts")
ax3b.legend()

# Right: score degradation lines
ax3c.plot(spike_sizes, p_scores,  "o-", color=PALETTE["Pearson"],  linewidth=2.2,
          markersize=7, label="Pearson r")
ax3c.plot(spike_sizes, sp_scores, "s-", color=PALETTE["Spearman"], linewidth=2.2,
          markersize=7, label="Spearman ρ")
for sp, pv, sv in zip(spike_sizes, p_scores, sp_scores):
    ax3c.annotate(f"{pv:.3f}", (sp, pv), textcoords="offset points",
                  xytext=(-5, 8), fontsize=7, color=PALETTE["Pearson"])
    ax3c.annotate(f"{sv:.3f}", (sp, sv), textcoords="offset points",
                  xytext=(-5, -14), fontsize=7, color=PALETTE["Spearman"])
ax3c.set_xlabel("Outlier spike amplitude (extra counts at one channel)")
ax3c.set_ylabel("Similarity score")
ax3c.set_title("Score vs outlier magnitude\nPearson degrades; Spearman is stable")
ax3c.legend()
ax3c.grid(alpha=0.3)
ax3c.set_ylim(
    min(min(p_scores), min(sp_scores)) - 0.1,
    max(max(p_scores), max(sp_scores)) + 0.15,
)

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
