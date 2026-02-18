from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class Interface:
    @staticmethod
    def spline_gamma(anode_counts: np.ndarray, target_size: int = 70) -> np.ndarray:
        source = np.asarray(anode_counts, dtype=np.float64).reshape(-1)
        if source.size != 64:
            raise ValueError(
                f"spline_gamma expects 64 detector channels, got {source.size}."
            )

        x_source = np.arange(source.size, dtype=np.float64)
        x_target = np.linspace(0, source.size - 1, num=target_size * target_size)
        spline = CubicSpline(x_source, source)
        return spline(x_target)

    @staticmethod
    def spline_image(
        regr_image: np.ndarray,
        source_size: int,
        target_size: int = 70,
    ) -> np.ndarray:
        values = np.asarray(regr_image, dtype=np.float64).reshape(-1)
        expected = source_size * source_size
        if values.size != expected:
            raise ValueError(
                f"spline_image expected {expected} values, got {values.size}."
            )
        if source_size < 2:
            raise ValueError("source_size must be >= 2 for cubic interpolation.")

        source_grid = values.reshape(source_size, source_size)
        x_source = np.arange(source_size, dtype=np.float64)
        x_target = np.linspace(0, source_size - 1, target_size)

        row_interpolated = np.empty((source_size, target_size), dtype=np.float64)
        for row in range(source_size):
            row_spline = CubicSpline(x_source, source_grid[row, :])
            row_interpolated[row, :] = row_spline(x_target)

        output = np.empty((target_size, target_size), dtype=np.float64)
        for col in range(target_size):
            col_spline = CubicSpline(x_source, row_interpolated[:, col])
            output[:, col] = col_spline(x_target)

        return output.ravel()

    @staticmethod
    def spline_image_15(regr_image: np.ndarray) -> np.ndarray:
        return Interface.spline_image(
            regr_image=regr_image,
            source_size=15,
            target_size=70,
        )
