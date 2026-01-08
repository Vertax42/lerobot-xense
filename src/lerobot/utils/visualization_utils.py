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
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "20%")
    rr.spawn(memory_limit=memory_limit)
    # NOTE: We do NOT send a fixed blueprint here. This lets Rerun auto-discover
    # all logged entity paths and create views dynamically. If a static blueprint
    # is sent, changing stream names (e.g. depth -> rectify) won't update the view.


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

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format and logged as `rr.Image`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                key_lower = str(key).lower()

                # 1D array → individual Scalars (e.g., force_resultant with shape (6,))
                if arr.ndim == 1:
                    # force_resultant (6,) float has physical meaning: [Fx, Fy, Fz, Mx, My, Mz]
                    # Detect by shape (6,) and float dtype (key may not contain "force_resultant")
                    is_force_resultant = (
                        len(arr) == 6
                        and arr.dtype in (np.float32, np.float64)
                    )
                    if is_force_resultant:
                        force_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
                        for label, vi in zip(force_labels, arr, strict=True):
                            rr.log(f"{key}/{label}", rr.Scalars(float(vi)))
                    else:
                        for i, vi in enumerate(arr):
                            rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                    continue

                # Force distribution (35, 20, 3) float → compute magnitude and show as heatmap
                # Heuristic: key contains "force", 3D array, last dim is 3, dtype is float
                is_force_distribution = (
                    "force" in key_lower
                    and arr.ndim == 3
                    and arr.shape[-1] == 3
                    and arr.dtype in (np.float32, np.float64)
                )
                if is_force_distribution:
                    # Compute force magnitude: sqrt(fx^2 + fy^2 + fz^2) → (H, W) heatmap
                    force_magnitude = np.linalg.norm(arr, axis=-1).astype(np.float32)
                    # Use DepthImage as heatmap visualization (meter=1.0 means values are in meters)
                    rr.log(key, rr.DepthImage(force_magnitude, meter=1.0, colormap="turbo"))
                    continue

                # Convert CHW -> HWC when needed (for regular images)
                if (
                    arr.ndim == 3
                    and arr.shape[0] in (1, 3, 4)
                    and arr.shape[-1] not in (1, 3, 4)
                ):
                    arr = np.transpose(arr, (1, 2, 0))

                # Depth images (2D arrays) should be logged as rr.DepthImage for correct visualization.
                # Heuristic:
                # - If key contains "depth" and the array is 2D (or single-channel 3D), treat as depth.
                # - Also treat "likely depth" arrays as depth even if the key doesn't include "depth":
                #   2D float32/float64 arrays are likely depth (RGB images are uint8 HWC 3-channel).
                is_depth_shape = arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)
                depth = arr[..., 0] if (arr.ndim == 3 and arr.shape[-1] == 1) else arr
                likely_depth = (
                    is_depth_shape
                    and isinstance(depth, np.ndarray)
                    and depth.dtype in (np.float32, np.float64)  # float 2D = likely depth
                )

                # IMPORTANT: these are typically streaming observations. Do not log them as
                # static/timeless, otherwise the Rerun viewer won't update on new frames.
                if ("depth" in key_lower and is_depth_shape) or likely_depth:
                    depth = arr[..., 0] if (arr.ndim == 3 and arr.shape[-1] == 1) else arr
                    rr.log(key, rr.DepthImage(depth, meter=0.001, depth_range=(0.0, 0.1), colormap="turbo"))
                else:
                    rr.log(key, rr.Image(arr))

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith("action.") else f"action.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
