# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import numpy as np


class MeanScaler:
    """Just like torch MeanScaler, but for numpy."""

    def __init__(
        self,
        axis: int,
        keepdims: bool = False,
        default_scale: Optional[float] = None,
        minimum_scale: float = 1e-10,
    ):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale or 0.0

    def __call__(
        self, data: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # these will have shape (N, C)
        total_weight = weights.sum(axis=self.axis)
        weighted_sum = (np.abs(data) * weights).sum(axis=self.axis)

        # first compute a global scale per-dimension
        total_observed = total_weight.sum(axis=0)
        denominator = np.maximum(total_observed, np.ones_like(total_observed))

        if self.default_scale != 0.0:
            default_scale = self.default_scale
        else:
            default_scale = weighted_sum.sum(axis=0) / denominator

        # then compute a per-item, per-dimension scale
        denominator = np.maximum(total_weight, np.ones_like(total_weight))
        scale = weighted_sum / denominator

        # use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = np.expand_dims(
            np.maximum(
                self.minimum_scale,
                np.where(
                    weighted_sum > np.zeros_like(weighted_sum),
                    scale,
                    default_scale * np.ones_like(total_weight),
                ),
            ),
            axis=self.axis,
        )

        return data / scale, scale if self.keepdims else scale.squeeze(
            axis=self.axis
        )


class NOPScaler:
    """
    Just like torch NOPScaler, but for numpy.
    """

    def __init__(self, axis: int, keepdims: bool = False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def __call__(
        self, data: np.ndarray, weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        scale = np.ones_like(data).mean(
            axis=self.axis,
            keepdims=self.keepdims,
        )
        return data, scale
