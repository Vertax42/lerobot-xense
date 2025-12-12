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

from dataclasses import dataclass
from enum import Enum

from ..configs import CameraConfig


class XenseOutputType(Enum):
    """Xense sensor output types matching SDK's Sensor.OutputType."""

    # Image outputs
    RECTIFY = "rectify"  # Rectified image, shape=(700, 400, 3), RGB
    DIFFERENCE = "difference"  # Difference image, shape=(700, 400, 3), RGB
    DEPTH = "depth"  # Depth map, shape=(700, 400), unit: mm

    # 2D/3D marker and force outputs
    MARKER_2D = "marker_2d"  # Tangential displacement, shape=(35, 20, 2)
    FORCE = "force"  # 3D force distribution, shape=(35, 20, 3)
    FORCE_NORM = "force_norm"  # Normal force component, shape=(35, 20, 3)
    FORCE_RESULTANT = "force_resultant"  # 6D force resultant, shape=(6,)

    # 3D mesh outputs
    MESH_3D = "mesh_3d"  # Current frame 3D mesh, shape=(35, 20, 3)
    MESH_3D_INIT = "mesh_3d_init"  # Initial 3D mesh, shape=(35, 20, 3)
    MESH_3D_FLOW = "mesh_3d_flow"  # Mesh deformation vector, shape=(35, 20, 3)


@CameraConfig.register_subclass("xense")
@dataclass
class XenseCameraConfig(CameraConfig):
    """Configuration class for Xense tactile sensor devices.

    This class provides configuration options for Xense tactile sensors,
    supporting various output types including force distribution, depth maps,
    and 2D marker tracking.

    Example configurations:
    ```python
    # Basic force sensing configuration
    XenseCameraConfig(
        serial_number="OG000344",
        fps=60,
        output_types=[XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]
    )

    # Multi-modal configuration with depth
    XenseCameraConfig(
        serial_number="OG000352",
        fps=30,
        output_types=[XenseOutputType.FORCE, XenseOutputType.DEPTH]
    )

    # High-performance configuration with reduced resolution
    XenseCameraConfig(
        serial_number="OG000344",
        fps=30,
        output_types=[XenseOutputType.DIFFERENCE],
        rectify_size=(200, 350),  # Reduced from (400, 700) for better performance
        raw_size=(320, 240)       # Raw sensor resolution
    )
    ```

    Attributes:
        serial_number: Xense sensor serial number (e.g., "OG000344")
        fps: Requested frames per second for data acquisition (default: 60)
        width: Frame width in pixels (auto-set based on output_types and rectify_size)
        height: Frame height in pixels (auto-set based on output_types and rectify_size)
        output_types: List of output types to read from the sensor
        warmup_s: Time to wait before returning from connect (in seconds)
        rectify_size: Rectified image size (width, height), default (400, 700)
        raw_size: Raw sensor resolution (width, height), default (320, 240)

    Note:
        - Image outputs (DIFFERENCE, RECTIFY) default shape: (700, 400, 3)
        - Depth output default shape: (700, 400)
        - Force distribution output shape: (35, 20, 3) [fixed]
        - Force resultant output shape: (6,) [fixed]
        - Reducing rectify_size improves performance (e.g., (200, 350) is 4x faster)
        - Width and height are automatically set based on rectify_size if using image outputs
    """

    serial_number: str
    # NOTE: we allow strings too (e.g. from CLI/YAML), and normalize them in __post_init__
    output_types: list[XenseOutputType] | list[str] | None = None
    warmup_s: float = 0.5
    rectify_size: tuple[int, int] | None = None  # (width, height) for rectified images
    raw_size: tuple[int, int] | None = None  # (width, height) for raw sensor data

    def __post_init__(self):
        # Set default output types if not provided
        if self.output_types is None:
            self.output_types = [XenseOutputType.FORCE, XenseOutputType.FORCE_RESULTANT]

        # Normalize & validate output types (support strings from CLI)
        normalized: list[XenseOutputType] = []
        for output_type in self.output_types:
            if isinstance(output_type, XenseOutputType):
                normalized.append(output_type)
                continue
            if isinstance(output_type, str):
                # Accept "difference" / "DIFFERENCE" / "XenseOutputType.DIFFERENCE"
                s = output_type.strip()
                if s.startswith("XenseOutputType."):
                    s = s.split(".", 1)[1]
                s_lower = s.lower()
                matched = None
                for v in XenseOutputType:
                    if v.value == s_lower or v.name.lower() == s_lower:
                        matched = v
                        break
                if matched is None:
                    raise ValueError(
                        f"Invalid output_type: {output_type}. "
                        f"Valid values: {[v.value for v in XenseOutputType]}"
                    )
                normalized.append(matched)
                continue

            raise ValueError(
                f"Invalid output_type: {output_type}. Must be a XenseOutputType (or str)."
            )
        self.output_types = normalized

        # Set default FPS if not provided
        if self.fps is None:
            self.fps = 30

        # Set default rectify_size and raw_size if not provided
        if self.rectify_size is None:
            self.rectify_size = (
                400,
                700,
            )  # Default full resolution (width, height) before transpose
        if self.raw_size is None:
            self.raw_size = (640, 480)  # Default raw sensor resolution (width, height)

        # Set width and height based on the primary output type
        # DIFFERENCE/RECTIFY/DEPTH images use rectify_size
        # Force/mesh data have shape (35, 20, 3) [fixed by SDK]
        if self.width is None or self.height is None:
            # Check if using image outputs (DIFFERENCE, RECTIFY, or DEPTH)
            image_outputs = {
                XenseOutputType.DIFFERENCE,
                XenseOutputType.RECTIFY,
                XenseOutputType.DEPTH,
            }
            if any(ot in image_outputs for ot in self.output_types):
                # Image outputs: use rectify_size (width, height)
                if self.height is None:
                    self.height = self.rectify_size[0]
                if self.width is None:
                    self.width = self.rectify_size[1]
            else:
                # Force/mesh outputs: height=35, width=20 (fixed by SDK)
                if self.width is None:
                    self.width = 20
                if self.height is None:
                    self.height = 35
