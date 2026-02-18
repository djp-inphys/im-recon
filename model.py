from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from data import ObservationData
from splines import Interface

logger = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        data: ObservationData,
        *,
        e2adc_fname: str = "e2adcLookup.csv",
        model_fname: str = "model.csv",
    ) -> None:
        self.data = data
        self.e2adc_fname = e2adc_fname
        self.model_fname = model_fname

        self.e2adc = self.read_e2adc()
        self.channel_map = self.get_hamamatsu_channel_map()
        self.spatial_model = self.read_model()

        self.model_phase_count = self.spatial_model.shape[2]
        self.pixel_count = self.spatial_model.shape[1]
        self.source_size = int(math.isqrt(self.pixel_count))
        if self.source_size * self.source_size != self.pixel_count:
            raise ValueError(
                f"Model pixel count {self.pixel_count} is not a perfect square."
            )

    def read_model(self) -> np.ndarray:
        model_path = Path(self.model_fname)
        if model_path.is_dir():
            return self._read_phase_model_directory(model_path)
        if model_path.is_file():
            return self._read_model_file(model_path)

        fallback_dir = Path("model")
        if fallback_dir.is_dir():
            logger.info(
                "Model file %s was not found. Falling back to %s.",
                self.model_fname,
                fallback_dir,
            )
            return self._read_phase_model_directory(fallback_dir)

        raise FileNotFoundError(
            f"Could not find model file '{self.model_fname}' or fallback 'model/' directory."
        )

    def _read_model_file(self, file_path: Path) -> np.ndarray:
        try:
            grouped_model = self._read_grouped_model_csv(file_path)
            if grouped_model.ndim == 3:
                return grouped_model
        except Exception:
            logger.debug("Grouped model parse failed for %s; trying plain CSV.", file_path)

        single_phase_model = np.loadtxt(file_path, delimiter=",")
        if single_phase_model.ndim != 2:
            raise ValueError(
                f"Expected 2D model matrix in {file_path}, got shape {single_phase_model.shape}."
            )

        if single_phase_model.shape[1] == self.data.n_channels:
            phase_matrix = single_phase_model.T
        elif single_phase_model.shape[0] == self.data.n_channels:
            phase_matrix = single_phase_model
        else:
            raise ValueError(
                f"Model matrix in {file_path} does not contain {self.data.n_channels} channels."
            )

        return phase_matrix[:, :, np.newaxis]

    def _read_grouped_model_csv(self, file_path: Path) -> np.ndarray:
        grouped_rows: list[list[list[float]]] = []
        current_group: list[list[float]] = []

        with open(file_path, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if row:
                    current_group.append([float(value) for value in row])
                elif current_group:
                    grouped_rows.append(current_group)
                    current_group = []

        if current_group:
            grouped_rows.append(current_group)
        if not grouped_rows:
            raise ValueError(f"No model data found in {file_path}.")

        model = np.asarray(grouped_rows, dtype=np.float64)

        if model.ndim == 3 and model.shape[2] == self.data.n_channels:
            return np.transpose(model, (2, 1, 0))
        if model.ndim == 3 and model.shape[1] == self.data.n_channels:
            return np.transpose(model, (1, 2, 0))
        if model.ndim == 3 and model.shape[0] == self.data.n_channels:
            return model

        raise ValueError(
            f"Unsupported grouped model shape {model.shape} in {file_path}."
        )

    def _read_phase_model_directory(self, model_dir: Path) -> np.ndarray:
        phase_files = sorted(
            model_dir.glob("model_phase_*.csv"),
            key=self._extract_phase_number,
        )
        if not phase_files:
            raise FileNotFoundError(
                f"No model_phase_*.csv files were found in {model_dir}."
            )

        phase_matrices: list[np.ndarray] = []
        for phase_file in phase_files:
            matrix = np.loadtxt(phase_file, delimiter=",")
            if matrix.ndim != 2:
                raise ValueError(
                    f"Expected 2D matrix in {phase_file}, got shape {matrix.shape}."
                )

            if matrix.shape[1] == self.data.n_channels:
                phase_matrix = matrix.T
            elif matrix.shape[0] == self.data.n_channels:
                phase_matrix = matrix
            else:
                raise ValueError(
                    f"{phase_file} does not contain {self.data.n_channels} channels."
                )

            phase_matrices.append(phase_matrix)

        base_shape = phase_matrices[0].shape
        for idx, matrix in enumerate(phase_matrices[1:], start=1):
            if matrix.shape != base_shape:
                raise ValueError(
                    "Phase model shape mismatch: "
                    f"phase 0 has {base_shape}, phase {idx} has {matrix.shape}."
                )

        return np.stack(phase_matrices, axis=2)

    @staticmethod
    def _extract_phase_number(path: Path) -> int:
        match = re.search(r"model_phase_(\d+)\.csv$", path.name)
        if not match:
            return 0
        return int(match.group(1))

    def read_e2adc(self) -> np.ndarray:
        e2adc_df = pd.read_csv(self.e2adc_fname, header=None)
        e2adc_rows = e2adc_df.to_numpy(dtype=np.float64)

        max_channel = int(np.max(e2adc_rows[:, 0])) + 1
        max_energy = int(np.max(e2adc_rows[:, 1])) + 1

        e2adc = np.zeros((max_channel, max_energy, 2), dtype=np.float64)
        for row in e2adc_rows:
            channel = int(row[0])
            energy = int(row[1])
            adc = int(row[2])
            flatfield = float(row[3])
            e2adc[channel, energy, 0] = adc
            e2adc[channel, energy, 1] = flatfield
        return e2adc

    @staticmethod
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

    def normalise_phase_counts(self) -> np.ndarray:
        return self.data.normalise_phase_totals()

    def write_norm_count_data(self, file_name: str) -> None:
        self.data.write_normalised_count_data(file_name)

    def sum_for_energy_range(self, phase: int, bin_low: int, bin_high: int) -> np.ndarray:
        low = max(0, int(bin_low))
        high = min(int(bin_high), self.e2adc.shape[1])
        if low >= high:
            raise ValueError(
                f"Invalid energy range [{bin_low}, {bin_high})."
            )

        slot = self.data.get_phase_slot(phase)
        phase_counts = self.data.phase_counts[slot]

        adc_sum = np.zeros(self.data.n_channels, dtype=np.float64)
        for channel in range(self.data.n_channels):
            adc_indices = self.e2adc[channel, low:high, 0].astype(np.int64)
            adc_indices = np.clip(adc_indices, 0, self.data.n_bins - 1)
            flatfields = self.e2adc[channel, low:high, 1]
            adc_sum[channel] = np.sum(phase_counts[channel, adc_indices] * flatfields)
        return adc_sum

    def recon(
        self,
        *,
        bin_low: int,
        bin_high: int,
        active_phases: Sequence[int],
        target_size: int = 70,
    ) -> np.ndarray:
        active_phase_list = list(active_phases)
        active_count = len(active_phase_list)
        if active_count == 0:
            return np.zeros(target_size * target_size, dtype=np.float64)

        phase_energy_sums = np.vstack(
            [
                self.sum_for_energy_range(phase, bin_low, bin_high)
                for phase in active_phase_list
            ]
        )

        actual = np.zeros(self.data.n_channels * active_count, dtype=np.float64)
        sub_model = np.zeros(self.data.n_channels * active_count, dtype=np.float64)
        regression = np.zeros(self.pixel_count, dtype=np.float64)

        for src_pos_index in range(self.pixel_count):
            for channel in range(self.data.n_channels):
                inverted_channel = 63 - channel
                logical_channel = self.channel_map[inverted_channel]

                for phase_offset, phase in enumerate(active_phase_list):
                    corr_index = logical_channel * active_count + phase_offset
                    model_phase_index = self._resolve_model_phase_index(
                        phase_value=phase,
                        phase_offset=phase_offset,
                    )

                    actual[corr_index] = phase_energy_sums[phase_offset, logical_channel]
                    sub_model[corr_index] = self.spatial_model[
                        channel,
                        src_pos_index,
                        model_phase_index,
                    ]

            actual_sum = np.sum(actual)
            model_sum = np.sum(sub_model)
            if actual_sum == 0 or model_sum == 0:
                regression[src_pos_index] = 0.0
            else:
                norm_model = 100.0 * sub_model * (actual_sum / model_sum)
                q, _ = pearsonr(actual, norm_model)
                regression[src_pos_index] = 0.0 if not np.isfinite(q) else q

            actual.fill(0.0)
            sub_model.fill(0.0)

        return Interface.spline_image(
            regr_image=regression,
            source_size=self.source_size,
            target_size=target_size,
        )

    def _resolve_model_phase_index(self, phase_value: int, phase_offset: int) -> int:
        if 0 <= phase_value < self.model_phase_count:
            return phase_value
        if 0 <= phase_offset < self.model_phase_count:
            return phase_offset
        raise IndexError(
            f"Model has {self.model_phase_count} phases; cannot map phase {phase_value}."
        )
