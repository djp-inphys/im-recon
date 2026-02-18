from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from main import PipelineConfig, ReconstructionPipeline, configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IsoBatchConfig:
    root_dir: str = "remtech"
    output_subdir: str = "processed"
    stride: int = 1
    min_active_phases_for_normalisation: int = 7


def _find_capture_dirs(root: Path) -> list[Path]:
    capture_dirs: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "GammaIndex.csv").exists():
            continue
        if not ((path / "gamma.dat").exists() or (path / "Gamma.dat").exists()):
            continue
        capture_dirs.append(path)
    return capture_dirs


def _resolve_gamma_file(capture_dir: Path) -> Path:
    lower_case = capture_dir / "gamma.dat"
    upper_case = capture_dir / "Gamma.dat"
    if lower_case.exists():
        return lower_case
    return upper_case


def iso_main(config: IsoBatchConfig) -> None:
    root = Path(config.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {root}")

    capture_dirs = _find_capture_dirs(root)
    if not capture_dirs:
        logger.warning("No capture directories found in %s", root)
        return

    for capture_dir in capture_dirs:
        output_dir = capture_dir / config.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_config = PipelineConfig(
            index_filename=str(capture_dir / "GammaIndex.csv"),
            gamma_filename=str(_resolve_gamma_file(capture_dir)),
            observed_counts_filename=str(output_dir / "observed_counts.csv"),
            normalised_counts_filename=str(output_dir / "norm_observed_counts.csv"),
            stride=config.stride,
            min_active_phases_for_normalisation=config.min_active_phases_for_normalisation,
            reconstruction_strategy="none",
        )

        logger.info("Processing capture %s", capture_dir.name)
        pipeline = ReconstructionPipeline(pipeline_config)
        pipeline.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ISO processing pipeline")
    parser.add_argument("--root-dir", default="remtech", help="Directory containing capture subdirectories")
    parser.add_argument("--output-subdir", default="processed", help="Output subdirectory under each capture")
    parser.add_argument("--stride", type=int, default=1, help="Stride for index sampling")
    parser.add_argument(
        "--min-active-phases",
        type=int,
        default=7,
        help="Minimum active phases before writing normalised counts",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    iso_main(
        IsoBatchConfig(
            root_dir=args.root_dir,
            output_subdir=args.output_subdir,
            stride=args.stride,
            min_active_phases_for_normalisation=args.min_active_phases,
        )
    )
