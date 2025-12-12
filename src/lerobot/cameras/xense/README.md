# Xense Tactile Camera Integration

This module integrates Xense tactile sensors into the LeRobot camera framework, providing a unified interface for reading force, depth, and marker data.

## Overview

The Xense tactile sensor provides rich tactile-visual information including:
- **Force Distribution**: 3D force at each point (35×20×3)
- **Force Resultant**: 6D force/torque vector (6,)
- **Depth Maps**: Surface depth information (700×400)
- **2D Markers**: Tangential displacement (35×20×2)
- **3D Mesh**: Surface deformation data (35×20×3)

## Installation

### Prerequisites

1. **Install xensesdk** (follow manufacturer instructions):
   ```bash
   conda activate lerobot-openpi
   uv pip install xensesdk
   ```

2. **Install missing dependencies** (as noted in xensesdk README):
   ```bash
    uv pip install scipy cypack cryptography pyudev assimp_py==1.0.7 qtpy PyQt5 h5py lz4 -i https://mirrors.huaweicloud.com/repository/pypi/simple
    uv pip install cyclonedds-nightly==2025.7.29 -i https://mirrors.huaweicloud.com/repository/pypi/simple
    uv pip install xensesdk==1.6.3 -i https://mirrors.huaweicloud.com/repository/pypi/simple
    conda install cuda-toolkit=12.9 -c nvidia
    conda install cudnn -c conda-forge -y
    uv pip install onnxruntime-gpu==1.19.2
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
   ```

## Quick Start

### 1. Find Available Sensors

```python
from lerobot.cameras.xense import XenseTactileCamera

# Discover connected sensors
sensors = XenseTactileCamera.find_cameras()
for sensor in sensors:
    print(f"Found: {sensor['serial_number']}, Cam ID: {sensor['cam_id']}")
```

### 2. Basic Usage

```python
from lerobot.cameras.xense import (
    XenseTactileCamera,
    XenseCameraConfig,
    XenseOutputType,
)

# Configure sensor
config = XenseCameraConfig(
    serial_number="OG000344",
    fps=60,
    output_types=[
        XenseOutputType.FORCE,
        XenseOutputType.FORCE_RESULTANT,
    ],
)

# Create and connect
camera = XenseTactileCamera(config)
camera.connect()

# Read synchronously
data = camera.read()
force = data["force"]  # shape: (35, 20, 3)
force_resultant = data["force_resultant"]  # shape: (6,)

# Read asynchronously (with background thread)
async_data = camera.async_read(timeout_ms=200)

# Disconnect when done
camera.disconnect()
```

## Available Output Types

```python
from lerobot.cameras.xense import XenseOutputType

# Image outputs
XenseOutputType.RECTIFY          # shape=(700, 400, 3), RGB
XenseOutputType.DIFFERENCE       # shape=(700, 400, 3), RGB
XenseOutputType.DEPTH            # shape=(700, 400), unit: mm

# Force outputs
XenseOutputType.FORCE            # shape=(35, 20, 3), 3D force distribution
XenseOutputType.FORCE_NORM       # shape=(35, 20, 3), normal force component
XenseOutputType.FORCE_RESULTANT  # shape=(6,), 6D force/torque

# Marker outputs
XenseOutputType.MARKER_2D        # shape=(35, 20, 2), tangential displacement

# 3D mesh outputs
XenseOutputType.MESH_3D          # shape=(35, 20, 3), current frame mesh
XenseOutputType.MESH_3D_INIT     # shape=(35, 20, 3), initial mesh
XenseOutputType.MESH_3D_FLOW     # shape=(35, 20, 3), deformation vector
```

## Testing

### Simple Test
```bash
python test_xense_simple.py
```

This will:
1. Find available sensors
2. Connect to the first sensor
3. Test synchronous reading (5 frames)
4. Test asynchronous reading (10 frames)
5. Measure FPS performance

### Full Test Suite
```bash
python test_xense_camera.py
```

Choose from:
1. Synchronous reading test
2. Asynchronous reading test
3. Dual sensor test (bimanual)
4. Run all tests

## Integration with BiARX5 Robot

To add Xense sensors to your robot configuration:

```python
from lerobot.cameras.xense import XenseCameraConfig, XenseOutputType

cameras = {
    # ... existing cameras ...
    "right_tactile": XenseCameraConfig(
        serial_number="OG000344",
        fps=60,
        output_types=[
            XenseOutputType.FORCE,
            XenseOutputType.FORCE_RESULTANT,
        ],
    ),
    "left_tactile": XenseCameraConfig(
        serial_number="OG000352",
        fps=60,
        output_types=[
            XenseOutputType.FORCE,
            XenseOutputType.FORCE_RESULTANT,
        ],
    ),
}
```

## Key Differences from Image Cameras

1. **Output Format**: Returns `dict[str, np.ndarray]` instead of single image array
2. **Data Types**: Multiple data modalities (force, depth, mesh) instead of just RGB/BGR
3. **Shape Variety**: Different output types have different shapes:
   - Images: (700, 400, 3)
   - Force grid: (35, 20, 3)
   - Force resultant: (6,)
4. **No Color Mode**: Force data is numeric, not color-based

## Performance Notes

- **Recommended FPS**: 60 Hz for force sensing (reduce to 30 Hz if experiencing V4L2 timeouts)
- **Warmup Time**: 0.5s default (adjustable via `warmup_s`)
- **Async Reading**: Uses background thread for non-blocking reads
- **Timeout**: 200ms default for async reads (adjustable)

## V4L2 High Load Handling

Under high system load (multiple cameras, recording), V4L2 timeout warnings may occur:
```
[ WARN:35@48.719] global cap_v4l.cpp:1049 tryIoctl VIDEOIO(V4L2:/dev/video22): select() timeout.
```

### Automatic Handling

The camera automatically handles V4L2 timeouts:
- **Retry Logic**: Automatically retries up to 3 times on timeout errors
- **Smart Error Suppression**: Only logs warnings after 10 consecutive failures
- **Exponential Backoff**: Adds delays between retries to reduce system load
- **Graceful Degradation**: Continues operation even with occasional timeouts

### Optimization Strategies

#### 1. Reduce FPS (Most Effective)
```python
# Lower FPS reduces V4L2 load significantly
XenseCameraConfig(
    serial_number="OG000344",
    fps=30,  # Reduced from 60 to 30 Hz
    output_types=[XenseOutputType.DIFFERENCE],
)
```

#### 2. Reduce Resolution (High Impact)
```python
# Reducing rectify_size improves performance by 4x
XenseCameraConfig(
    serial_number="OG000344",
    fps=60,
    output_types=[XenseOutputType.DIFFERENCE],
    rectify_size=(200, 350),  # Reduced from (400, 700)
    raw_size=(320, 240),
)
```

#### 3. Use Only Necessary Output Types
```python
# Request only needed data types to reduce processing
XenseCameraConfig(
    serial_number="OG000344",
    fps=60,
    output_types=[XenseOutputType.DIFFERENCE],  # Only difference, not force + depth
)
```

#### 4. Increase Warmup Time
```python
# Longer warmup helps stabilize sensor under load
XenseCameraConfig(
    serial_number="OG000344",
    fps=60,
    warmup_s=1.5,  # Increased from 0.5s
)
```

#### 5. Recommended High-Load Configuration
```python
# Optimized configuration for recording with multiple cameras
XenseCameraConfig(
    serial_number="OG000344",
    fps=30,  # Reduced FPS
    output_types=[XenseOutputType.DIFFERENCE],  # Minimal output
    rectify_size=(200, 350),  # Reduced resolution
    warmup_s=1.0,  # Longer warmup
)
```

### System-Level Optimizations

1. **Grant Real-Time Priority** (for CAN communication):
   ```bash
   sudo setcap cap_sys_nice=ep $(readlink -f $(which python))
   ```

2. **Reduce Other Camera FPS**: If using multiple cameras, reduce FPS across all:
   ```python
   # Reduce all camera FPS proportionally
   RealSenseCameraConfig(fps=30, ...)  # Instead of 60
   XenseCameraConfig(fps=30, ...)      # Instead of 60
   ```

3. **Monitor System Load**: Use `htop` to check CPU/memory usage during recording

## Troubleshooting

### Import Error: "No module named 'xensesdk'"
```bash
uv pip install xensesdk
```

### Missing Dependencies
Install all required packages:
```bash
uv pip install cypack cryptography pyudev assimp_py==1.0.7 qtpy PyQt5 h5py lz4
```

### Sensor Not Found
- Check USB connection
- Run `Sensor.getXenseDeviceList()` to see available devices
- Verify serial number matches your sensor

### Qt Platform Plugin Issues
See xensesdk documentation for Qt-related troubleshooting.

## API Reference

### XenseCameraConfig

Configuration class for Xense sensors.

**Parameters**:
- `serial_number` (str): Sensor serial number (e.g., "OG000344")
- `fps` (int, optional): Target frame rate (default: 60)
- `output_types` (list[XenseOutputType], optional): Data types to read
- `warmup_s` (float, optional): Warmup duration in seconds (default: 0.5)

### XenseTactileCamera

Main camera class implementing the Camera interface.

**Methods**:
- `connect(warmup=True)`: Connect to sensor
- `read()`: Synchronous read, returns dict of arrays
- `async_read(timeout_ms=200)`: Asynchronous read with timeout
- `disconnect()`: Release sensor resources
- `find_cameras()` (static): Discover available sensors

## Examples

See the test scripts for complete examples:
- `test_xense_simple.py`: Basic functionality test
- `test_xense_camera.py`: Comprehensive test suite

## Support

For Xense SDK issues, contact: qjrobot9966 (WeChat)

For LeRobot integration issues, see the main LeRobot documentation.

