from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from splines import Interface


DECODE_MASKS: dict[str, np.ndarray] = {
    "hamamatsu": np.array(
        [
            [1, -1, -1, -1, -1, -1, -1],
            [1, 1, 1, -1, 1, -1, -1],
            [1, 1, 1, -1, 1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1],
            [1, 1, 1, -1, 1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1],
        ],
        dtype=np.float64,
    ),
    "ltetest": np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, -1, 1, 1],
            [-1, -1, -1, 1, -1, 1, 1],
            [-1, 1, 1, -1, 1, -1, -1],
            [-1, -1, -1, 1, -1, 1, 1],
            [-1, 1, 1, -1, 1, -1, -1],
            [-1, 1, 1, -1, 1, -1, -1],
        ],
        dtype=np.float64,
    ),
    "201222_3": np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [-1, 1, 1, -1, 1, -1, -1],
            [-1, 1, 1, -1, 1, -1, -1],
            [-1, -1, -1, 1, -1, 1, 1],
            [-1, 1, 1, -1, 1, -1, -1],
            [-1, -1, -1, 1, -1, 1, 1],
            [-1, -1, -1, 1, -1, 1, 1],
        ],
        dtype=np.float64,
    ),
}


def get_hamamatsu_channel_map() -> list[int]:
    channel_map = [0] * 64
    channel_map[0:8] = [15, 14, 13, 12, 11, 10, 9, 8]
    channel_map[8:16] = [0, 1, 2, 3, 4, 5, 6, 7]
    channel_map[16:24] = [31, 30, 29, 28, 27, 26, 25, 24]
    channel_map[24:32] = [16, 17, 18, 19, 20, 21, 22, 23]
    channel_map[32:40] = [47, 46, 45, 44, 43, 42, 41, 40]
    channel_map[40:48] = [32, 33, 34, 35, 36, 37, 38, 39]
    channel_map[48:56] = [63, 62, 61, 60, 59, 58, 57, 56]
    channel_map[56:64] = [48, 49, 50, 51, 52, 53, 54, 55]
    return channel_map


def get_identity_channel_map() -> list[int]:
    return list(range(64))


def load_e2adc_lookup(file_path: str) -> np.ndarray:
    e2adc_rows = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float64)
    max_channel = int(np.max(e2adc_rows[:, 0])) + 1
    max_energy = int(np.max(e2adc_rows[:, 1])) + 1

    lookup = np.zeros((max_channel, max_energy, 2), dtype=np.float64)
    for row in e2adc_rows:
        channel = int(row[0])
        energy = int(row[1])
        adc = int(row[2])
        flatfield = float(row[3])
        lookup[channel, energy, 0] = adc
        lookup[channel, energy, 1] = flatfield
    return lookup


@dataclass(frozen=True)
class MaskAntiMaskConfig:
    e2adc_file: str = "e2adcLookup.csv"
    decode_mask_variant: str = "hamamatsu"
    detector_map: str = "hamamatsu"
    row_shift: int = 10
    col_shift: int = -10
    mask_phase: int | None = None
    anti_mask_phase: int | None = None


class MaskAntiMaskReconstruction:
    def __init__(self, config: MaskAntiMaskConfig) -> None:
        self.config = config
        self.e2adc = load_e2adc_lookup(config.e2adc_file)
        self.base_decode_mask = self._resolve_decode_mask(config.decode_mask_variant)
        self.channel_map = self._resolve_channel_map(config.detector_map)

    @staticmethod
    def _resolve_decode_mask(variant: str) -> np.ndarray:
        key = variant.strip().lower()
        if key not in DECODE_MASKS:
            raise ValueError(
                f"Unknown decode mask variant '{variant}'. Supported: {sorted(DECODE_MASKS.keys())}."
            )
        return DECODE_MASKS[key]

    @staticmethod
    def _resolve_channel_map(map_name: str) -> list[int]:
        key = map_name.strip().lower()
        if key in {"hamamatsu", "hamamatsu_ca"}:
            return get_hamamatsu_channel_map()
        if key in {"identity", "rosmap"}:
            return get_identity_channel_map()
        raise ValueError(
            f"Unknown detector map '{map_name}'. Supported: hamamatsu, identity."
        )

    def reconstruct(
        self,
        *,
        phase_counts: np.ndarray,
        phase_to_slot: dict[int, int],
        active_phases: Sequence[int],
        bin_low: int,
        bin_high: int,
        target_size: int = 70,
    ) -> tuple[np.ndarray, float, dict[str, int]]:
        active = [int(phase) for phase in active_phases]
        if len(active) < 2:
            raise ValueError(
                "Mask/anti-mask reconstruction requires at least two active phases."
            )

        has_mask_override = self.config.mask_phase is not None
        has_anti_override = self.config.anti_mask_phase is not None
        if has_mask_override != has_anti_override:
            raise ValueError(
                "mask_phase and anti_mask_phase overrides must be provided together."
            )

        if has_mask_override and has_anti_override:
            mask_phase = int(self.config.mask_phase)
            anti_mask_phase = int(self.config.anti_mask_phase)
            if mask_phase not in phase_to_slot or anti_mask_phase not in phase_to_slot:
                raise ValueError(
                    "Configured mask phases are not available in accumulated data: "
                    f"mask_phase={mask_phase}, anti_mask_phase={anti_mask_phase}, "
                    f"available={sorted(phase_to_slot.keys())}."
                )
        else:
            mask_phase = active[0]
            anti_mask_phase = active[1]
        mask_slot = phase_to_slot[mask_phase]
        anti_mask_slot = phase_to_slot[anti_mask_phase]

        mask_delta = phase_counts[mask_slot] - phase_counts[anti_mask_slot]
        energy_array = self._sum_energy_range(mask_delta, bin_low=bin_low, bin_high=bin_high)
        position_energy = energy_array[np.asarray(self.channel_map, dtype=np.int64)]

        splined_counts = Interface.spline_gamma(position_energy, target_size=target_size)
        decode_mask = self._scale_decode_mask(target_size).ravel()
        correlated = np.fft.ifft(np.fft.fft(decode_mask) * np.fft.fft(splined_counts)).real
        shifted = self._shift_to_image(correlated, target_size=target_size)

        metadata = {
            "mask_phase": mask_phase,
            "anti_mask_phase": anti_mask_phase,
            "active_phase_count": len(active),
        }
        return shifted.ravel(), float(np.max(shifted)), metadata

    def _sum_energy_range(
        self,
        mask_delta: np.ndarray,
        *,
        bin_low: int,
        bin_high: int,
    ) -> np.ndarray:
        low = max(0, int(bin_low))
        high = min(int(bin_high), self.e2adc.shape[1])
        if low >= high:
            raise ValueError(f"Invalid energy range [{bin_low}, {bin_high}).")

        channel_count = mask_delta.shape[0]
        adc_bins = mask_delta.shape[1]
        energy_sum = np.zeros(channel_count, dtype=np.float64)

        for channel in range(channel_count):
            adc_indices = self.e2adc[channel, low:high, 0].astype(np.int64)
            adc_indices = np.clip(adc_indices, 0, adc_bins - 1)
            flatfields = self.e2adc[channel, low:high, 1]
            energy_sum[channel] = np.sum(mask_delta[channel, adc_indices] * flatfields)
        return energy_sum

    def _scale_decode_mask(self, target_size: int) -> np.ndarray:
        rows = (np.arange(target_size) * self.base_decode_mask.shape[0]) // target_size
        cols = (np.arange(target_size) * self.base_decode_mask.shape[1]) // target_size
        return self.base_decode_mask[np.ix_(rows, cols)]

    def _shift_to_image(self, correlated: np.ndarray, *, target_size: int) -> np.ndarray:
        corr_2d = correlated.reshape(target_size, target_size)
        shifted = np.zeros_like(corr_2d)

        for corr_row in range(target_size):
            for corr_col in range(target_size):
                image_row = target_size - 1 - corr_row
                image_col = corr_col

                image_row = (target_size + image_row + self.config.row_shift) % target_size
                image_col = (target_size + image_col + self.config.col_shift) % target_size
                shifted[image_row, image_col] = corr_2d[corr_row, corr_col]

        return shifted
