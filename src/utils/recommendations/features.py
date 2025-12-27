from __future__ import annotations

from typing import Tuple
import numpy as np


def extract_scaler_params(scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract offsets and scales from a fitted sklearn-like scaler.

    Returns
    -------
    offsets, scales :
        Arrays such that: scaled = (x - offsets) / scales

    Supports:
    - StandardScaler  (mean_,  scale_)
    - MinMaxScaler    (data_min_, data_range_)
    - RobustScaler    (center_, scale_)
    """
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # StandardScaler
        return scaler.mean_, scaler.scale_

    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
        # MinMaxScaler
        return scaler.data_min_, scaler.data_range_

    if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
        # RobustScaler
        return scaler.center_, scaler.scale_

    raise ValueError(f"Unsupported scaler type: {type(scaler)!r}")