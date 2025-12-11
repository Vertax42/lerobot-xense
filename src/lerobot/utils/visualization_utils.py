# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os
from typing import Any

import numpy as np
import rerun as rr

from .constants import OBS_PREFIX, OBS_STR


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    - Scalars are logged as `rr.Scalars`.
    - 3D NumPy arrays with channel-first layout are transposed to HWC and logged as `rr.Image`.
    - 1D arrays are expanded into individual scalars.
    - Tuples/lists (e.g., multi-output tactile sensor reads) are unpacked and each element is logged
      under an indexed suffix.
    """

    def _log_value(base_key: str, value: Any, is_obs: bool) -> None:
        """Recursively log values with sensible defaults for images and scalars."""
        if value is None:
            return

        if _is_scalar(value):
            rr.log(base_key, rr.Scalars(float(value)))
            return

        if isinstance(value, (tuple, list)):
            for i, vi in enumerate(value):
                _log_value(f"{base_key}_{i}", vi, is_obs)
            return

        if isinstance(value, np.ndarray):
            arr = value
            # Convert CHW -> HWC when needed
            if (
                arr.ndim == 3
                and arr.shape[0] in (1, 3, 4)
                and arr.shape[-1] not in (1, 3, 4)
            ):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 1:
                for i, vi in enumerate(arr):
                    rr.log(f"{base_key}_{i}", rr.Scalars(float(vi)))
            else:
                # Use dynamic logging for streams (no static=True) so images update over time.
                rr.log(base_key, rr.Image(arr))
            return

        # Fallback: try to log numeric types via float conversion
        try:
            rr.log(base_key, rr.Scalars(float(value)))
        except Exception:
            # Ignore unsupported types silently to keep loop fast
            return

    if observation:
        for k, v in observation.items():
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"
            _log_value(key, v, is_obs=True)

    if action:
        for k, v in action.items():
            key = k if str(k).startswith("action.") else f"action.{k}"
            _log_value(key, v, is_obs=False)
