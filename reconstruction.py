from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from data import ObservationData
from mask_antimask import MaskAntiMaskConfig, MaskAntiMaskReconstruction
from model import Model, SimilarityMetric

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionContext:
    data: ObservationData
    active_phases: tuple[int, ...]
    bin_low: int
    bin_high: int
    target_size: int = 70
    metric: SimilarityMetric = "spearman"


@dataclass
class ReconstructionResult:
    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class ReconstructionStrategy(ABC):
    name = "base"

    def setup(self, data: ObservationData) -> None:
        pass

    @abstractmethod
    def reconstruct(
        self,
        context: ReconstructionContext,
    ) -> ReconstructionResult | None:
        raise NotImplementedError


class NoOpReconstructionStrategy(ReconstructionStrategy):
    name = "none"

    def reconstruct(
        self,
        context: ReconstructionContext,
    ) -> ReconstructionResult | None:
        return None


class ModelCorrelationStrategy(ReconstructionStrategy):
    name = "model"

    def __init__(self, *, model_file: str = "model.csv", e2adc_file: str = "e2adcLookup.csv") -> None:
        self.model_file = model_file
        self.e2adc_file = e2adc_file
        self._model: Model | None = None

    def setup(self, data: ObservationData) -> None:
        self._model = Model(
            data=data,
            model_fname=self.model_file,
            e2adc_fname=self.e2adc_file,
        )

    def reconstruct(
        self,
        context: ReconstructionContext,
    ) -> ReconstructionResult | None:
        if self._model is None:
            self.setup(context.data)

        image = self._model.recon(
            bin_low=context.bin_low,
            bin_high=context.bin_high,
            active_phases=context.active_phases,
            target_size=context.target_size,
            metric=context.metric,
        )

        return ReconstructionResult(
            image=image,
            metadata={
                "strategy": self.name,
                "metric": context.metric,
                "active_phase_count": len(context.active_phases),
                "bin_low": context.bin_low,
                "bin_high": context.bin_high,
                "target_size": context.target_size,
            },
        )


class MaskAntiMaskStrategy(ReconstructionStrategy):
    name = "mask-antimask"

    def __init__(
        self,
        *,
        e2adc_file: str = "e2adcLookup.csv",
        decode_mask_variant: str = "hamamatsu",
        detector_map: str = "hamamatsu",
        row_shift: int = 10,
        col_shift: int = -10,
        mask_phase: int | None = None,
        anti_mask_phase: int | None = None,
    ) -> None:
        self.config = MaskAntiMaskConfig(
            e2adc_file=e2adc_file,
            decode_mask_variant=decode_mask_variant,
            detector_map=detector_map,
            row_shift=row_shift,
            col_shift=col_shift,
            mask_phase=mask_phase,
            anti_mask_phase=anti_mask_phase,
        )
        self._reconstructor: MaskAntiMaskReconstruction | None = None

    def setup(self, data: ObservationData) -> None:
        self._reconstructor = MaskAntiMaskReconstruction(self.config)

    def reconstruct(
        self,
        context: ReconstructionContext,
    ) -> ReconstructionResult | None:
        if self._reconstructor is None:
            self.setup(context.data)

        if len(context.active_phases) < 2:
            logger.warning(
                "Skipping mask-antimask reconstruction: requires at least 2 active phases, got %d.",
                len(context.active_phases),
            )
            return None

        image, max_value, details = self._reconstructor.reconstruct(
            phase_counts=context.data.phase_counts,
            phase_to_slot=context.data.phase_to_slot,
            active_phases=context.active_phases,
            bin_low=context.bin_low,
            bin_high=context.bin_high,
            target_size=context.target_size,
        )

        metadata = {
            "strategy": self.name,
            "bin_low": context.bin_low,
            "bin_high": context.bin_high,
            "target_size": context.target_size,
            "max_value": max_value,
        }
        metadata.update(details)
        return ReconstructionResult(image=image, metadata=metadata)


def create_reconstruction_strategy(
    strategy_name: str,
    *,
    model_file: str = "model.csv",
    e2adc_file: str = "e2adcLookup.csv",
    decode_mask_variant: str = "hamamatsu",
    detector_map: str = "hamamatsu",
    row_shift: int = 10,
    col_shift: int = -10,
    mask_phase: int | None = None,
    anti_mask_phase: int | None = None,
) -> ReconstructionStrategy:
    name = strategy_name.strip().lower()
    if name in {"none", "off", "noop"}:
        return NoOpReconstructionStrategy()
    if name in {"model", "pearson"}:
        return ModelCorrelationStrategy(
            model_file=model_file,
            e2adc_file=e2adc_file,
        )
    if name in {"mask-antimask", "mask", "fft"}:
        return MaskAntiMaskStrategy(
            e2adc_file=e2adc_file,
            decode_mask_variant=decode_mask_variant,
            detector_map=detector_map,
            row_shift=row_shift,
            col_shift=col_shift,
            mask_phase=mask_phase,
            anti_mask_phase=anti_mask_phase,
        )
    raise ValueError(
        "Unknown reconstruction strategy "
        f"'{strategy_name}'. Supported: none, model, mask-antimask."
    )
