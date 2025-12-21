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

import math
import platform
import time

import numpy as np


def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # On Mac and Windows, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qw, qx, qy, qz].
    
    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians
    
    Returns:
        Tuple of (qw, qx, qy, qz) representing the quaternion
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return (qw, qx, qy, qz)


def normalize_quaternion(qw: float | np.ndarray, qx: float | None = None, qy: float | None = None, qz: float | None = None) -> np.ndarray | tuple[float, float, float, float]:
    """Normalize quaternion to unit length using optimized NumPy operations.
    
    Supports two calling conventions:
    1. normalize_quaternion(q: np.ndarray) -> np.ndarray
    2. normalize_quaternion(qw, qx, qy, qz) -> tuple[float, float, float, float]
    
    Args:
        qw: If qx is None, this is a numpy array [qw, qx, qy, qz] or [qx, qy, qz, qw]
            Otherwise, this is the w component of the quaternion
        qx: x component (if provided, all 4 components must be provided)
        qy: y component
        qz: z component
    
    Returns:
        Normalized quaternion as numpy array or tuple (qw, qx, qy, qz)
    """
    # Handle two calling conventions
    if qx is None:
        # Single argument: numpy array - optimized NumPy path
        q = np.asarray(qw, dtype=np.float32)
        if q.ndim > 1:
            q = q.flatten()
        if len(q) != 4:
            raise ValueError(f"Quaternion must have 4 components, got {len(q)}")
        
        # Optimized: use np.linalg.norm (BLAS-optimized) and vectorized division
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # Identity [qx, qy, qz, qw]
        # Vectorized division - NumPy handles this efficiently
        return (q / norm).astype(np.float32)
    else:
        # Four separate arguments: (qw, qx, qy, qz) - convert to array for NumPy operations
        q = np.array([float(qw), float(qx), float(qy), float(qz)], dtype=np.float32)
        norm = np.linalg.norm(q)
        
        if norm < 1e-10:
            return (1.0, 0.0, 0.0, 0.0)  # Identity quaternion (qw, qx, qy, qz)
        
        # Use NumPy for normalization, then convert back to tuple
        normalized = q / norm
        return (float(normalized[0]), float(normalized[1]), float(normalized[2]), float(normalized[3]))
