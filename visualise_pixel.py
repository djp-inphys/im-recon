"""Generate per-pixel diagnostic plots for the model-driven reconstruction.

Two plots are produced for the requested pixel:

  1. Single-anode view   — observed counts vs calculated model power for one
     anode across all CA rotation phases, with the estimated non-CA floor
     shown as a red horizontal line.  Mirrors Figure 1 in README.md.

  2. All-anodes scatter  — all 64 anodes × all CA phases in a single scatter,
     with a least-squares regression line.  The slope of that line is the
     per-pixel score used to build the gamma image.  Mirrors Figure 2 in
     README.md.

Usage
-----
    # minimal — auto-selects the most informative anode
    python visualise_pixel.py --pixel 2653

    # choose a specific anode and a custom output prefix
    python visualise_pixel.py --pixel 2653 --anode 47 --output-prefix readme

    # limit energy range
    python visualise_pixel.py --pixel 2653 --bin-low 50 --bin-high 400
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import ObservationData
from model import Model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _accumulate_phase_counts(data: ObservationData) -> list[int]:
    """Run a single pass over all frames and accumulate counts into data.phase_counts.

    Returns the ordered list of active phase IDs found in the data.
    """
    data.reset_phase_counts()
    seen: set[int] = set()
    ordered: list[int] = []
    total = data.num_instances
    for frame in data.iter_frames():
        data.add_count_data(frame.bins, frame.phase)
        if frame.phase not in seen:
            seen.add(frame.phase)
            ordered.append(frame.phase)
        if frame.instance_index % 500 == 0:
            logger.info("  frame %d / %d", frame.instance_index, total)
    return ordered


def _logical_channel_map(model: Model) -> np.ndarray:
    """Return shape-(n_channels,) array mapping physical channel → logical channel."""
    n = model.data.n_channels
    return np.array([model.channel_map[63 - c] for c in range(n)], dtype=np.int32)


def _phase_energy_sums(
    model: Model,
    active_phases: list[int],
    bin_low: int,
    bin_high: int,
) -> np.ndarray:
    """Return observed energy-integrated counts, shape (n_phases, n_channels)."""
    return np.vstack(
        [model.sum_for_energy_range(ph, bin_low, bin_high) for ph in active_phases]
    )


def _pixel_scatter_data(
    model: Model,
    phase_sums: np.ndarray,
    lch_map: np.ndarray,
    pixel_idx: int,
    active_phases: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (model_power, observed_count) pairs for one pixel.

    Returns
    -------
    model_vals   : shape (n_channels * n_phases,)
    obs_vals     : shape (n_channels * n_phases,)
    anode_labels : shape (n_channels * n_phases,)  — logical channel per point
    """
    n_ch = model.data.n_channels
    n_ph = len(active_phases)
    model_vals   = np.empty(n_ch * n_ph)
    obs_vals     = np.empty(n_ch * n_ph)
    anode_labels = np.empty(n_ch * n_ph, dtype=np.int32)

    for c in range(n_ch):
        lch = int(lch_map[c])
        for ph_off, phase in enumerate(active_phases):
            ph_idx = model._resolve_model_phase_index(
                phase_value=phase, phase_offset=ph_off
            )
            i = c * n_ph + ph_off
            model_vals[i]   = model.spatial_model[c, pixel_idx, ph_idx]
            obs_vals[i]     = phase_sums[ph_off, lch]
            anode_labels[i] = lch

    return model_vals, obs_vals, anode_labels


def _single_anode_data(
    model: Model,
    phase_sums: np.ndarray,
    lch_map: np.ndarray,
    pixel_idx: int,
    active_phases: list[int],
    anode: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-phase (model_power, observed_count) for one logical anode.

    Points are returned in active_phases order (i.e. ascending CA angle).

    Returns
    -------
    model_vals : shape (n_phases,)
    obs_vals   : shape (n_phases,)
    """
    # Find the physical channel that maps to this logical channel.
    phys = next(c for c in range(model.data.n_channels) if int(lch_map[c]) == anode)

    n_ph = len(active_phases)
    model_vals = np.empty(n_ph)
    obs_vals   = np.empty(n_ph)

    for ph_off, phase in enumerate(active_phases):
        ph_idx = model._resolve_model_phase_index(
            phase_value=phase, phase_offset=ph_off
        )
        model_vals[ph_off] = model.spatial_model[phys, pixel_idx, ph_idx]
        obs_vals[ph_off]   = phase_sums[ph_off, anode]

    return model_vals, obs_vals


def _auto_select_anode(
    model: Model,
    phase_sums: np.ndarray,
    lch_map: np.ndarray,
    pixel_idx: int,
    active_phases: list[int],
) -> int:
    """Choose the anode whose OLS slope for this pixel is most positive.

    A positive slope means observed counts rise with model power — the clearest
    single-anode evidence that a source exists on this pixel's 3D vector.
    Falls back to the widest count range if no anode has a positive slope.
    """
    best_anode = 0
    best_score = float("-inf")

    for lch in range(model.data.n_channels):
        m_vals, o_vals = _single_anode_data(
            model, phase_sums, lch_map, pixel_idx, active_phases, lch
        )
        slope, _ = _ols(m_vals, o_vals)
        # Prefer positive slope; among ties use magnitude so strong signals win.
        score = slope if slope > 0 else slope - float(np.ptp(o_vals))
        if score > best_score:
            best_score = score
            best_anode = lch

    return best_anode


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) from ordinary least-squares."""
    n = float(len(x))
    if n < 2:
        return 0.0, float(np.mean(y)) if len(y) else 0.0
    sx  = x.sum()
    sy  = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, float(sy / n)
    slope     = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return float(slope), float(intercept)


def _phase_to_angle(phase: int) -> float:
    """Map a phase ID to the corresponding CA rotation angle in degrees.

    Each phase increments the CA by 15 °.  We do NOT wrap at 90 ° so that
    phase 6 (= 90 °) is labelled correctly rather than colliding with phase 0.
    """
    return float(int(phase) * 15)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_single_anode(
    model_vals: np.ndarray,
    obs_vals: np.ndarray,
    active_phases: list[int],
    *,
    anode: int,
    pixel_idx: int,
    output_path: str,
) -> None:
    """Single-anode view: mirrors Figure 1 in README.md.

    Points are sorted left-to-right by model power (ascending X) and each is
    labelled with its CA angle.  The dashed OLS line is the regression fitted
    to the sorted data — this is the "regression curve" that captures the
    via-CA slope.  No zigzag connector is drawn because, for noisy data, such
    a line obscures rather than clarifies the sorted order.
    """
    angles = [_phase_to_angle(p) for p in active_phases]

    # Sort by model power so points appear left-to-right on the X axis.
    order = np.argsort(model_vals)
    x = model_vals[order]
    y = obs_vals[order]
    sorted_angles = [angles[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter — one point per CA position, coloured by CA angle.
    # Intentionally no grey connector: y is not monotonic with x for noisy data
    # and a zigzag line would look unsorted even though x IS in ascending order.
    sc = ax.scatter(x, y, c=sorted_angles, cmap="viridis", s=80, zorder=3,
                    label=f"CA positions ({len(active_phases)} phases, sorted by model power)")

    # Annotate each point with its CA angle and its sorted-order rank.
    for rank, (xi, yi, ang) in enumerate(zip(x, y, sorted_angles), start=1):
        ax.annotate(
            f"{ang:.0f}°",
            (xi, yi),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#333",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("CA angle (°)", fontsize=8)

    # Non-CA floor: minimum observed count across phases for this anode
    non_ca_floor = float(obs_vals.min())
    ax.axhline(non_ca_floor, color="red", linewidth=1.8, linestyle="-",
               label=f"estimated non-CA floor  ({non_ca_floor:.0f} counts)", zorder=4)

    # OLS regression curve — fitted to all points, displayed over the sorted range.
    # This is the "regression curve" showing the via-CA slope.
    slope, intercept = _ols(model_vals, obs_vals)
    xfit = np.linspace(x.min(), x.max(), 300)   # x is already sorted ascending
    ax.plot(xfit, slope * xfit + intercept, "--", color="#222222", linewidth=2.0,
            label=f"OLS regression slope = {slope:.3f}", zorder=5)

    ax.set_xlabel("Calculated model power  (anode counts / unit source strength)\n"
                  "← points sorted left → right by model power →",
                  fontsize=10)
    ax.set_ylabel("Observed anode counts", fontsize=10)
    ax.set_title(
        f"anode {anode} counts @ Y,  calc power for pixel {pixel_idx} @ X\n"
        f"({len(active_phases)} CA positions — X axis in ascending model-power order)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {output_path}")


def plot_all_anodes(
    model_vals: np.ndarray,
    obs_vals: np.ndarray,
    *,
    pixel_idx: int,
    n_phases: int,
    output_path: str,
) -> None:
    """All-anodes regression scatter: mirrors Figure 2 in README.md."""
    n_anodes = len(model_vals) // n_phases if n_phases else 64

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(model_vals, obs_vals, s=10, alpha=0.50, color="#377eb8",
               label=f"{len(model_vals)} points  ({n_anodes} anodes × {n_phases} phases)")

    # Single OLS regression across all (anode, phase) pairs
    slope, intercept = _ols(model_vals, obs_vals)
    xfit = np.linspace(model_vals.min(), model_vals.max(), 300)
    ax.plot(xfit, slope * xfit + intercept, ":", color="#222222", linewidth=2.2,
            label=f"regression slope = {slope:.4f}")

    ax.set_xlabel("Calculated model power  (anode counts / unit source strength)",
                  fontsize=10)
    ax.set_ylabel("Observed anode counts", fontsize=10)
    ax.set_title(
        f"All anodes × all CA phases — pixel {pixel_idx}\n"
        f"regression slope  ∝  source strength at this pixel",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── required ──
    parser.add_argument(
        "--pixel",
        type=int,
        required=True,
        metavar="IDX",
        help=(
            "Zero-based pixel index in the model source grid "
            "(0 to pixel_count−1, where pixel_count = source_side²)."
        ),
    )

    # ── optional plot controls ──
    parser.add_argument(
        "--anode",
        type=int,
        default=None,
        metavar="CH",
        help=(
            "Logical anode (channel) index for the single-anode view (0–63). "
            "Defaults to the anode with the widest observed count range."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "Filename prefix for the two output PNGs. "
            "Defaults to 'pixel_<IDX>'. "
            "Outputs: <prefix>_single_anode.png  and  <prefix>_all_anodes.png."
        ),
    )

    # ── data / model paths (same defaults as main.py) ──
    parser.add_argument("--index",      default="GammaIndex.csv",   help="GammaIndex CSV")
    parser.add_argument("--gamma",      default="Gamma.dat",         help="Gamma binary file")
    parser.add_argument("--model-file", default="model.csv",         help="Model file or directory")
    parser.add_argument("--e2adc-file", default="e2adcLookup.csv",   help="e2adc lookup CSV")
    parser.add_argument("--bin-low",    type=int, default=0,         help="Lower energy bin (inclusive)")
    parser.add_argument("--bin-high",   type=int, default=600,       help="Upper energy bin (exclusive)")
    parser.add_argument("--stride",     type=int, default=1,         help="Frame stride for data loading")
    parser.add_argument("--log-level",  default="WARNING",
                        help="Logging level: DEBUG, INFO, WARNING (default), ERROR")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    # ── 1. Load gamma data ────────────────────────────────────────────────────
    print("Loading gamma data …")
    data = ObservationData(
        index_fname=args.index,
        gamma_fname=args.gamma,
        stride=args.stride,
    )
    print(f"  {data.num_instances} instances, {data.n_phases} phases, "
          f"{data.n_channels} channels")

    # ── 2. Accumulate all phase counts ────────────────────────────────────────
    print("Accumulating phase counts …")
    active_phases = _accumulate_phase_counts(data)
    print(f"  Active phases: {active_phases}")

    # ── 3. Load model ─────────────────────────────────────────────────────────
    print("Loading model …")
    model = Model(data=data, model_fname=args.model_file, e2adc_fname=args.e2adc_file)
    print(f"  Source grid: {model.source_size}×{model.source_size} "
          f"({model.pixel_count} pixels),  {model.model_phase_count} model phase(s)")

    # ── 4. Validate pixel index ───────────────────────────────────────────────
    pixel_idx = args.pixel
    if not (0 <= pixel_idx < model.pixel_count):
        print(
            f"Error: --pixel {pixel_idx} is out of range "
            f"(valid: 0 – {model.pixel_count - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 5. Extract scatter data ───────────────────────────────────────────────
    print(f"Extracting data for pixel {pixel_idx} …")
    lch_map    = _logical_channel_map(model)
    phase_sums = _phase_energy_sums(model, active_phases, args.bin_low, args.bin_high)
    model_x, obs_y, _ = _pixel_scatter_data(
        model, phase_sums, lch_map, pixel_idx, active_phases
    )

    # ── 6. Choose / validate anode ────────────────────────────────────────────
    anode = args.anode
    if anode is None:
        anode = _auto_select_anode(model, phase_sums, lch_map, pixel_idx, active_phases)
        print(f"  Auto-selected anode {anode} (strongest positive slope for this pixel).")
    elif not (0 <= anode < data.n_channels):
        print(
            f"Error: --anode {anode} is out of range (valid: 0 – {data.n_channels - 1}).",
            file=sys.stderr,
        )
        sys.exit(1)

    single_model, single_obs = _single_anode_data(
        model, phase_sums, lch_map, pixel_idx, active_phases, anode
    )

    # ── 7. Output paths ───────────────────────────────────────────────────────
    prefix       = args.output_prefix or f"pixel_{pixel_idx}"
    single_path  = f"{prefix}_single_anode.png"
    scatter_path = f"{prefix}_all_anodes.png"

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    plot_single_anode(
        single_model, single_obs, active_phases,
        anode=anode,
        pixel_idx=pixel_idx,
        output_path=single_path,
    )

    plot_all_anodes(
        model_x, obs_y,
        pixel_idx=pixel_idx,
        n_phases=len(active_phases),
        output_path=scatter_path,
    )

    print(f"\nDone.  Use --pixel {pixel_idx} --anode {anode} to reproduce these plots.")


if __name__ == "__main__":
    main()
