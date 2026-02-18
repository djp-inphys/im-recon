from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationFrame:
    instance_index: int
    phase: int
    bins: np.ndarray


@dataclass
class ObservationData:
    index_fname: str
    gamma_fname: str
    n_bins: int = 600
    n_channels: int = 64
    stride: int = 1
    phase_sequence: list[int] | None = None

    gamma_index: pd.DataFrame = field(init=False, repr=False)
    obs: pd.DataFrame = field(init=False, repr=False)
    phase_to_slot: dict[int, int] = field(init=False, repr=False)
    phase_counts: np.ndarray = field(init=False, repr=False)
    phaseCounts: np.ndarray = field(init=False, repr=False)
    phase_instance_counts: dict[int, int] = field(init=False, repr=False)
    n_phases: int = field(init=False)

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError(f"Stride must be >= 1, got {self.stride}.")

        self.gamma_index = self.load_gamma_index(self.index_fname)
        self.obs = read_gamma_obs(self, self.gamma_fname, self.stride)
        self.phase_sequence = self._resolve_phase_sequence(self.phase_sequence)
        self.phase_to_slot = {
            phase: slot for slot, phase in enumerate(self.phase_sequence)
        }
        self.n_phases = len(self.phase_sequence)
        self.reset_phase_counts()

    def _resolve_phase_sequence(self, configured: list[int] | None) -> list[int]:
        observed = sorted(int(phase) for phase in self.obs["phase"].unique())
        if configured is None:
            return observed

        ordered_unique: list[int] = []
        for phase in configured:
            phase_value = int(phase)
            if phase_value not in ordered_unique:
                ordered_unique.append(phase_value)

        for phase in observed:
            if phase not in ordered_unique:
                ordered_unique.append(phase)

        return ordered_unique

    @property
    def num_instances(self) -> int:
        if self.n_channels <= 0:
            return 0

        count = len(self.obs) // self.n_channels
        if len(self.obs) % self.n_channels != 0:
            logger.warning(
                "Observation rows (%d) are not divisible by n_channels (%d); truncating to %d instances.",
                len(self.obs),
                self.n_channels,
                count,
            )
        return count

    def get_phase_slot(self, phase: int) -> int:
        phase_value = int(phase)
        if phase_value not in self.phase_to_slot:
            raise KeyError(
                f"Phase {phase_value} is not available. Known phases: {self.phase_sequence}."
            )
        return self.phase_to_slot[phase_value]

    def reset_phase_counts(self) -> None:
        self.phase_counts = np.zeros(
            (self.n_phases, self.n_channels, self.n_bins),
            dtype=np.float64,
        )
        self.phaseCounts = self.phase_counts
        self.phase_instance_counts = {phase: 0 for phase in self.phase_sequence}

    def inc_n_phases(self, phase: int) -> None:
        phase_value = int(phase)
        if phase_value not in self.phase_instance_counts:
            self.phase_instance_counts[phase_value] = 0
        self.phase_instance_counts[phase_value] += 1

    def get_n_phases(self, phase: int) -> int:
        return self.phase_instance_counts.get(int(phase), 0)

    def add_count_data(self, gamma_data: np.ndarray, phase: int) -> None:
        gamma_array = np.asarray(gamma_data, dtype=np.float64)
        expected_shape = (self.n_channels, self.n_bins)
        if gamma_array.shape != expected_shape:
            raise ValueError(
                f"Gamma data shape mismatch. Expected {expected_shape}, got {gamma_array.shape}."
            )

        phase_value = int(phase)
        slot = self.get_phase_slot(phase_value)
        previous_samples = self.phase_instance_counts[phase_value]
        new_samples = previous_samples + 1

        running_total = self.phase_counts[slot] * previous_samples
        self.phase_counts[slot] = (running_total + gamma_array) / new_samples
        self.phase_instance_counts[phase_value] = new_samples

    def iter_frames(self) -> Iterator[ObservationFrame]:
        for instance_index in range(self.num_instances):
            instance_df = self.get_instance(instance_index)
            phase = int(instance_df["phase"].iloc[0])
            bins = instance_df.filter(like="bin_").to_numpy(dtype=np.float64, copy=True)
            yield ObservationFrame(
                instance_index=instance_index,
                phase=phase,
                bins=bins,
            )

    def get_instance(self, instance_n: int) -> pd.DataFrame:
        if instance_n < 0:
            raise IndexError("Instance index must be >= 0.")

        start_idx = instance_n * self.n_channels
        end_idx = start_idx + self.n_channels
        if end_idx > len(self.obs):
            raise IndexError(
                f"Instance {instance_n} is out of range. Max instance index is {self.num_instances - 1}."
            )
        return self.obs.iloc[start_idx:end_idx].reset_index(drop=True)

    def get_instance_ranges(self) -> dict[int, tuple[int, int]]:
        ranges: dict[int, tuple[int, int]] = {}
        for phase in self.phase_sequence:
            phase_df = self.obs[self.obs["phase"] == phase]
            num_instances = len(phase_df) // self.n_channels
            ranges[phase] = (0, num_instances)
        return ranges

    def get_num_cycles(self) -> int:
        instance_ranges = self.get_instance_ranges()
        if not instance_ranges:
            return 0

        max_obs_cycle = [max(bounds) for bounds in instance_ranges.values()]
        return min(max_obs_cycle)

    def get_nth_obs_df(self, phase: int, instance_n: int) -> pd.DataFrame:
        phase_df = self.obs[self.obs["phase"] == phase]
        start_index = instance_n * self.n_channels
        end_index = start_index + self.n_channels

        if start_index >= len(phase_df):
            return pd.DataFrame()

        return phase_df.iloc[start_index:end_index].filter(like="bin_")

    def sum_bins(self, obs_df: pd.DataFrame) -> np.ndarray:
        summed_channel_data = obs_df.filter(like="bin_").sum(axis=1).to_numpy(dtype=np.float64)
        if len(summed_channel_data) != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {len(summed_channel_data)}."
            )
        return summed_channel_data

    def get_obs_shadowgram(self, phase: int, index: int) -> np.ndarray:
        return self.sum_bins(self.get_nth_obs_df(phase=phase, instance_n=index))

    def get_phase_totals(self) -> np.ndarray:
        return np.sum(self.phase_counts, axis=2)

    def normalise_phase_totals(self) -> np.ndarray:
        phase_totals = self.get_phase_totals()
        channel_totals = np.sum(phase_totals, axis=0, keepdims=True)
        return np.divide(
            phase_totals,
            channel_totals,
            out=np.zeros_like(phase_totals),
            where=channel_totals != 0,
        )

    def _write_phase_rows(self, file_name: str, values: np.ndarray) -> None:
        with open(file_name, "a", newline="") as file:
            writer = csv.writer(file)
            for slot, phase in enumerate(self.phase_sequence):
                row = [phase]
                row.extend(values[slot].tolist())
                writer.writerow(row)

    def write_count_data(self, file_name: str) -> None:
        self._write_phase_rows(file_name, self.get_phase_totals())

    def write_normalised_count_data(self, file_name: str) -> None:
        self._write_phase_rows(file_name, self.normalise_phase_totals())

    @staticmethod
    def load_gamma_index(file_path: str) -> pd.DataFrame:
        required_columns = {"filePosition", "phase"}
        gamma_index = pd.read_csv(file_path)
        missing_columns = required_columns.difference(gamma_index.columns)
        if missing_columns:
            raise ValueError(
                f"Gamma index file {file_path} is missing required columns: {sorted(missing_columns)}."
            )
        return gamma_index

    def read_gamma_obs(self, stride: int) -> pd.DataFrame:
        return read_gamma_obs(self, self.gamma_fname, stride)


def read_gamma_obs(data: ObservationData, gamma_file_name: str, stride: int) -> pd.DataFrame:
    gamma_index = data.gamma_index.reset_index(drop=True)
    total_rows = len(gamma_index)
    selected_rows = [row for row in range(total_rows) if (row + 1) % stride == 0]

    expected_count = data.n_channels * data.n_bins
    file_positions = gamma_index["filePosition"].astype(int).to_numpy()
    phases = gamma_index["phase"].astype(int).to_numpy()
    end_of_file = os.path.getsize(gamma_file_name)

    obs_blocks: list[np.ndarray] = []
    phase_values: list[int] = []
    index_values: list[int] = []

    with open(gamma_file_name, "rb") as file:
        for row_idx in selected_rows:
            file_position = int(file_positions[row_idx])
            if row_idx + 1 < total_rows:
                next_file_position = int(file_positions[row_idx + 1])
            else:
                next_file_position = end_of_file

            if next_file_position < file_position:
                logger.warning(
                    "Skipping row %d because next file position (%d) is before current position (%d).",
                    row_idx,
                    next_file_position,
                    file_position,
                )
                continue

            file.seek(file_position)
            raw_bytes = file.read(next_file_position - file_position)
            obs = np.frombuffer(raw_bytes, dtype="<i4")

            if obs.size < expected_count:
                obs = np.pad(obs, (0, expected_count - obs.size), mode="constant")
            elif obs.size > expected_count:
                logger.warning(
                    "Row %d has %d samples; truncating to %d.",
                    row_idx,
                    obs.size,
                    expected_count,
                )
                obs = obs[:expected_count]

            obs_blocks.append(obs.reshape(data.n_channels, data.n_bins))
            phase_values.append(int(phases[row_idx]))
            index_values.append(row_idx)

    if not obs_blocks:
        empty_columns = ["index", "phase", "channel"]
        empty_columns.extend([f"bin_{i}" for i in range(data.n_bins)])
        return pd.DataFrame(columns=empty_columns)

    flat_obs_data = np.vstack(obs_blocks).reshape(-1, data.n_bins)
    obs_df = pd.DataFrame(
        flat_obs_data,
        columns=[f"bin_{i}" for i in range(data.n_bins)],
    )

    obs_df.insert(0, "phase", np.repeat(phase_values, data.n_channels))
    obs_df.insert(1, "channel", np.tile(np.arange(data.n_channels), len(obs_blocks)))
    obs_df.insert(0, "index", np.repeat(index_values, data.n_channels))

    return obs_df


def reshape(summed_channel_data: pd.Series | np.ndarray, num_channels: int = 64) -> np.ndarray:
    values = np.asarray(summed_channel_data)
    if len(values) != num_channels:
        raise ValueError(
            f"The number of channels ({len(values)}) does not match expected ({num_channels})."
        )
    return values.reshape(8, 8)


obsData = ObservationData
