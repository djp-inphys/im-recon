from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ObservationData
from reconstruction import (
    ReconstructionContext,
    ReconstructionResult,
    ReconstructionStrategy,
    create_reconstruction_strategy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    index_filename: str = "GammaIndex.csv"
    gamma_filename: str = "Gamma.dat"
    observed_counts_filename: str = "observed_counts.csv"
    normalised_counts_filename: str = "norm_observed_counts.csv"
    stride: int = 1
    min_active_phases_for_normalisation: int = 7
    reconstruction_strategy: str = "none"
    reconstruction_bin_low: int = 0
    reconstruction_bin_high: int = 600
    reconstruction_output: str | None = None
    reconstruction_image_prefix: str | None = None
    reconstruction_image_cmap: str = "inferno"
    final_phase_images_only: bool = False
    model_filename: str = "model.csv"
    e2adc_filename: str = "e2adcLookup.csv"
    decode_mask_variant: str = "hamamatsu"
    detector_map: str = "hamamatsu"
    mask_row_shift: int = 10
    mask_col_shift: int = -10
    mask_phase: int | None = None
    anti_mask_phase: int | None = None
    target_image_size: int = 70


class ReconstructionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.data = ObservationData(
            index_fname=config.index_filename,
            gamma_fname=config.gamma_filename,
            stride=config.stride,
        )
        self.strategy: ReconstructionStrategy = create_reconstruction_strategy(
            config.reconstruction_strategy,
            model_file=config.model_filename,
            e2adc_file=config.e2adc_filename,
            decode_mask_variant=config.decode_mask_variant,
            detector_map=config.detector_map,
            row_shift=config.mask_row_shift,
            col_shift=config.mask_col_shift,
            mask_phase=config.mask_phase,
            anti_mask_phase=config.anti_mask_phase,
        )
        self.strategy.setup(self.data)

    @property
    def normalisation_threshold(self) -> int:
        return max(
            1,
            min(
                self.config.min_active_phases_for_normalisation,
                self.data.n_phases,
            ),
        )

    def run(self) -> ReconstructionResult | None:
        self._truncate_file(self.config.observed_counts_filename)
        self._truncate_file(self.config.normalised_counts_filename)
        self._log_available_phase_angle_list()

        logger.info("Starting observed-count accumulation pass")
        self.data.reset_phase_counts()
        observed_active_phases = self._run_pass(
            pass_name="observed",
            output_file=self.config.observed_counts_filename,
            writer=self.data.write_count_data,
            min_active_phases=0,
        )

        logger.info("Starting normalised-count accumulation pass")
        self.data.reset_phase_counts()
        normalised_active_phases = self._run_pass(
            pass_name="normalised",
            output_file=self.config.normalised_counts_filename,
            writer=self.data.write_normalised_count_data,
            min_active_phases=self.normalisation_threshold,
        )

        active_phases = normalised_active_phases or observed_active_phases
        result = self._run_reconstruction(active_phases)
        if result is not None and self.config.reconstruction_output:
            self._write_reconstruction_result(result)
        if result is not None:
            self._write_reconstruction_images(result)

        logger.info("Pipeline completed")
        return result

    def _run_pass(
        self,
        *,
        pass_name: str,
        output_file: str,
        writer: Callable[[str], None],
        min_active_phases: int,
    ) -> list[int]:
        seen_phases: set[int] = set()
        ordered_phases: list[int] = []
        total_instances = self.data.num_instances
        previous_phase: int | None = None

        for frame in self.data.iter_frames():
            if (
                not self.config.final_phase_images_only
                and previous_phase is not None
                and frame.phase != previous_phase
            ):
                self._write_phase_completion_snapshot(
                    pass_name=pass_name,
                    completed_phase=previous_phase,
                    completed_instance=frame.instance_index - 1,
                    active_phases=ordered_phases,
                )
            self.data.add_count_data(frame.bins, frame.phase)

            if frame.phase not in seen_phases:
                seen_phases.add(frame.phase)
                ordered_phases.append(frame.phase)
            previous_phase = frame.phase

            if len(seen_phases) >= min_active_phases:
                writer(output_file)

            if frame.instance_index % 100 == 0:
                logger.info(
                    "Processed instance %d/%d",
                    frame.instance_index,
                    total_instances,
                )

        if previous_phase is not None:
            self._write_phase_completion_snapshot(
                pass_name=pass_name,
                completed_phase=previous_phase,
                completed_instance=total_instances - 1,
                active_phases=ordered_phases,
            )
        return ordered_phases

    def _run_reconstruction(
        self,
        active_phases: Sequence[int],
        *,
        log_completion: bool = True,
    ) -> ReconstructionResult | None:
        if not active_phases:
            return None

        context = ReconstructionContext(
            data=self.data,
            active_phases=tuple(active_phases),
            bin_low=self.config.reconstruction_bin_low,
            bin_high=self.config.reconstruction_bin_high,
            target_size=self.config.target_image_size,
        )

        result = self.strategy.reconstruct(context)
        if result is not None and log_completion:
            logger.info(
                "Reconstruction complete using strategy '%s'.",
                self.strategy.name,
            )
        return result

    def _write_phase_completion_snapshot(
        self,
        *,
        pass_name: str,
        completed_phase: int,
        completed_instance: int,
        active_phases: Sequence[int],
    ) -> None:
        result = self._run_reconstruction(active_phases, log_completion=False)
        if result is None:
            return
        image = result.image.reshape(
            self.config.target_image_size,
            self.config.target_image_size,
        )
        variants = self._build_image_variants(image)
        prefix = self._resolve_reconstruction_image_prefix()
        output_path = (
            f"{prefix}_{pass_name}_phase_{int(completed_phase):02d}"
            f"_instance_{int(completed_instance):04d}.png"
        )
        self._save_image(variants["normalised"], output_path, title=f"{self.strategy.name} phase {completed_phase}")
        logger.info(
            "Saved %s pass snapshot for completed phase %d at instance %d to %s",
            pass_name,
            completed_phase,
            completed_instance,
            output_path,
        )

    def _log_available_phase_angle_list(self) -> None:
        gamma_index = self.data.gamma_index.copy()
        if gamma_index.empty:
            logger.warning("No gamma index rows available; cannot list phases/angles.")
            return

        phase_series = gamma_index["phase"].astype(int)
        unique_phases = sorted(int(phase) for phase in phase_series.unique())
        phase_counts = phase_series.value_counts().sort_index()
        logger.info("Available phases in gamma data: %s", unique_phases)
        logger.info(
            "Available phase counts in gamma data: %s",
            {int(phase): int(count) for phase, count in phase_counts.items()},
        )

        angle_column = self._find_angle_column(gamma_index)
        if angle_column is not None:
            angle_df = gamma_index[["phase", angle_column]].copy()
            angle_df["phase"] = angle_df["phase"].astype(int)
            angle_df[angle_column] = pd.to_numeric(angle_df[angle_column], errors="coerce")
            angle_df = angle_df.dropna(subset=[angle_column]).sort_values(["phase", angle_column])
            if angle_df.empty:
                logger.warning(
                    "Angle column '%s' exists but has no numeric values; cannot list angle mapping.",
                    angle_column,
                )
            else:
                logger.info("Angle/phase list from gamma index column '%s':", angle_column)
                for phase in unique_phases:
                    values = (
                        angle_df.loc[angle_df["phase"] == phase, angle_column]
                        .drop_duplicates()
                        .round(6)
                        .tolist()
                    )
                    logger.info("  phase=%d angles=%s", phase, values)
            return

        inferred = self._infer_phase_angles(unique_phases)
        logger.info(
            "No angle column found in gamma index; using coded-aperture mapping angle=(phase*15) mod 90."
        )
        for phase, angle in inferred:
            if np.isclose(angle, 0.0) and phase != 0:
                logger.info(
                    "  phase=%d coded_aperture_angle_deg=%.6f (equivalent to 90 deg)",
                    phase,
                    angle,
                )
            else:
                logger.info("  phase=%d coded_aperture_angle_deg=%.6f", phase, angle)

    @staticmethod
    def _find_angle_column(gamma_index: pd.DataFrame) -> str | None:
        candidate_names = {"angle", "angles", "theta", "rotationangle", "azimuth"}
        for column in gamma_index.columns:
            if column.strip().lower() in candidate_names:
                return column
        return None

    @staticmethod
    def _infer_phase_angles(phases: Sequence[int]) -> list[tuple[int, float]]:
        return [(int(phase), float((int(phase) * 15) % 90)) for phase in phases]

    def _write_reconstruction_result(self, result: ReconstructionResult) -> None:
        image = result.image.reshape(
            self.config.target_image_size,
            self.config.target_image_size,
        )
        np.savetxt(self.config.reconstruction_output, image, delimiter=",")
        logger.info("Saved reconstruction image to %s", self.config.reconstruction_output)

    def _write_reconstruction_images(self, result: ReconstructionResult) -> None:
        image = result.image.reshape(
            self.config.target_image_size,
            self.config.target_image_size,
        )
        prefix = self._resolve_reconstruction_image_prefix()
        image_variants = self._build_image_variants(image)

        for suffix, variant in image_variants.items():
            output_path = f"{prefix}_{suffix}.png"
            self._save_image(variant, output_path)
            logger.info("Saved reconstruction %s image to %s", suffix, output_path)

    def _resolve_reconstruction_image_prefix(self) -> str:
        if self.config.reconstruction_image_prefix:
            return self.config.reconstruction_image_prefix
        if self.config.reconstruction_output:
            output_path = Path(self.config.reconstruction_output)
            return str(output_path.with_suffix(""))
        return f"reconstruction_{self.strategy.name}"

    @staticmethod
    def _build_image_variants(image: np.ndarray) -> dict[str, np.ndarray]:
        image_float = image.astype(np.float64)
        abs_image = np.abs(image_float)
        min_value = float(np.min(image_float))
        max_value = float(np.max(image_float))
        value_range = max_value - min_value
        if value_range > 0:
            # User-requested inversion: normalized "high" values should map to dark.
            normalised = 1.0 - ((image_float - min_value) / value_range)
        else:
            normalised = np.zeros_like(image_float)
        return {
            "raw": image_float,
            "normalised": normalised,
            "abs": abs_image,
            "log_abs": np.log1p(abs_image),
        }

    def _save_image(self, image: np.ndarray, output_path: str, *, title: str | None = None) -> None:
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap=self.config.reconstruction_image_cmap)
        plt.colorbar()
        plt.title(title or self.strategy.name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    @staticmethod
    def _truncate_file(file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8"):
            pass


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("model_processing.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        index_filename=args.index,
        gamma_filename=args.gamma,
        observed_counts_filename=args.observed_output,
        normalised_counts_filename=args.normalised_output,
        stride=args.stride,
        min_active_phases_for_normalisation=args.min_active_phases,
        reconstruction_strategy=args.strategy,
        reconstruction_bin_low=args.bin_low,
        reconstruction_bin_high=args.bin_high,
        reconstruction_output=args.reconstruction_output,
        reconstruction_image_prefix=args.reconstruction_image_prefix,
        reconstruction_image_cmap=args.reconstruction_image_cmap,
        final_phase_images_only=args.final_phase_images_only,
        model_filename=args.model_file,
        e2adc_filename=args.e2adc_file,
        decode_mask_variant=args.decode_mask_variant,
        detector_map=args.detector_map,
        mask_row_shift=args.mask_row_shift,
        mask_col_shift=args.mask_col_shift,
        mask_phase=args.mask_phase,
        anti_mask_phase=args.anti_mask_phase,
        target_image_size=args.target_image_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamma reconstruction pipeline")
    parser.add_argument("--index", default="GammaIndex.csv", help="Path to GammaIndex CSV")
    parser.add_argument("--gamma", default="Gamma.dat", help="Path to gamma binary file")
    parser.add_argument(
        "--observed-output",
        default="observed_counts.csv",
        help="Output CSV for running phase counts",
    )
    parser.add_argument(
        "--normalised-output",
        default="norm_observed_counts.csv",
        help="Output CSV for normalised phase counts",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride for index sampling")
    parser.add_argument(
        "--min-active-phases",
        type=int,
        default=7,
        help="Minimum active phases before writing normalised counts",
    )
    parser.add_argument(
        "--strategy",
        choices=["none", "model", "mask-antimask"],
        default="model",
        help="Reconstruction strategy",
    )
    parser.add_argument(
        "--bin-low",
        type=int,
        default=0,
        help="Lower energy bin for reconstruction",
    )
    parser.add_argument(
        "--bin-high",
        type=int,
        default=600,
        help="Upper energy bin for reconstruction",
    )
    parser.add_argument(
        "--reconstruction-output",
        default=None,
        help="Optional CSV output for reconstruction image",
    )
    parser.add_argument(
        "--reconstruction-image-prefix",
        default=None,
        help=(
            "Optional prefix for reconstruction PNG images. "
            "Files are saved as <prefix>_raw.png, <prefix>_normalised.png, "
            "<prefix>_abs.png, and <prefix>_log_abs.png."
        ),
    )
    parser.add_argument(
        "--reconstruction-image-cmap",
        default="inferno",
        help="Matplotlib colormap name for reconstruction PNGs",
    )
    parser.add_argument(
        "--final-phase-images-only",
        action="store_true",
        default=False,
        help=(
            "If set, save only the final completed phase snapshot per pass "
            "(instead of saving a snapshot at every completed phase)."
        ),
    )
    parser.add_argument(
        "--model-file",
        default="model.csv",
        help="Path to model file or model directory",
    )
    parser.add_argument(
        "--e2adc-file",
        default="e2adcLookup.csv",
        help="Path to e2adc lookup CSV",
    )
    parser.add_argument(
        "--decode-mask-variant",
        choices=["hamamatsu", "ltetest", "201222_3"],
        default="hamamatsu",
        help="Decode mask variant for mask-antimask strategy",
    )
    parser.add_argument(
        "--detector-map",
        choices=["hamamatsu", "identity"],
        default="hamamatsu",
        help="Detector channel map for mask-antimask strategy",
    )
    parser.add_argument(
        "--mask-row-shift",
        type=int,
        default=10,
        help="Row shift for mask-antimask correlation image alignment",
    )
    parser.add_argument(
        "--mask-col-shift",
        type=int,
        default=-10,
        help="Column shift for mask-antimask correlation image alignment",
    )
    parser.add_argument(
        "--mask-phase",
        type=int,
        default=None,
        help="Optional phase id to use as mask phase for mask-antimask strategy",
    )
    parser.add_argument(
        "--anti-mask-phase",
        type=int,
        default=None,
        help="Optional phase id to use as anti-mask phase for mask-antimask strategy",
    )
    parser.add_argument(
        "--target-image-size",
        type=int,
        default=70,
        help="Width/height of output reconstruction image",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def model_main(config: PipelineConfig | None = None) -> ReconstructionResult | None:
    pipeline = ReconstructionPipeline(config or PipelineConfig())
    return pipeline.run()


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(arguments.log_level)
    configuration = build_config_from_args(arguments)
    model_main(configuration)
