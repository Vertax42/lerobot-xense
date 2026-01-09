# Camera Performance Benchmark

OpenCV 相机性能测试工具，用于测试不同相机模组（包括鱼眼相机）的性能。

## 功能

- **FPS 测试**: 测量实际 FPS 与请求 FPS 的差异
- **延迟测量**: 帧读取延迟 (min/max/avg/std/percentiles)
- **丢帧检测**: 统计丢帧数量
- **能力扫描**: 自动检测相机支持的分辨率、FPS、编码格式
- **批量测试**: 支持测试多个相机或多种配置

## 安装依赖

```bash
pip install opencv-python numpy
```

## 使用方法

### 基础测试

```bash
# 自动检测相机并测试
python benchmark_opencv_camera.py

# 测试指定相机 (index)
python benchmark_opencv_camera.py --index 0

# 测试指定相机 (path)
python benchmark_opencv_camera.py --path /dev/video0
```

### 指定参数测试

```bash
# 测试 1280x720 @ 30fps
python benchmark_opencv_camera.py --index 0 --width 1280 --height 720 --fps 30

# 测试 1920x1080 @ 60fps，持续 30 秒
python benchmark_opencv_camera.py --index 0 --width 1920 --height 1080 --fps 60 --duration 30

# 指定编码格式 (MJPG 通常支持更高 FPS)
python benchmark_opencv_camera.py --index 0 --fourcc MJPG --fps 60
```

### 能力扫描

```bash
# 扫描相机支持的所有模式
python benchmark_opencv_camera.py --index 0 --scan-capabilities

# 测试所有常见分辨率
python benchmark_opencv_camera.py --index 0 --test-all-resolutions

# 测试所有常见 FPS 值
python benchmark_opencv_camera.py --index 0 --test-all-fps
```

### 多相机测试

```bash
# 测试多个相机
python benchmark_opencv_camera.py --index 0 1 2

# 测试多个路径
python benchmark_opencv_camera.py --path /dev/video0 /dev/video2 /dev/video4
```

### 实时视频流 (Rerun 可视化)

```bash
# 使用 Rerun 可视化视频流 (默认 MJPG 编码，2秒预热)
python benchmark_opencv_camera.py --index 0 --video-stream

# 指定分辨率和帧率
python benchmark_opencv_camera.py --index 0 --video-stream --width 1920 --height 1080 --fps 60

# 自定义预热时间 (3秒)
python benchmark_opencv_camera.py --index 0 --video-stream --warmup 3

# 无预热 (立即开始统计 FPS)
python benchmark_opencv_camera.py --index 0 --video-stream --warmup 0

# 限时流 (60秒后自动停止)
python benchmark_opencv_camera.py --index 0 --video-stream --duration 60
```

### 使用 LeRobot OpenCVCamera (推荐)

LeRobot 的相机实现使用后台线程读取帧，延迟更低：

```bash
# 使用 LeRobot 相机 (async_read 模式，最低延迟)
python benchmark_opencv_camera.py --index 0 --video-stream --use-lerobot

# 指定分辨率和帧率
python benchmark_opencv_camera.py --index 0 --video-stream --use-lerobot --width 1920 --height 1080 --fps 60

# 使用同步读取模式 (sync read)
python benchmark_opencv_camera.py --index 0 --video-stream --use-lerobot --sync-read
```

#### LeRobot vs Raw OpenCV 对比

| 模式 | 延迟 | 说明 |
|-----|------|-----|
| `--use-lerobot` (默认 async) | ~2-5ms | 后台线程持续读取，获取最新帧 |
| `--use-lerobot --sync-read` | ~15-30ms | 同步阻塞读取 |
| 无 `--use-lerobot` | ~2-10ms | 优化后的 raw OpenCV (grab+retrieve) |

### 导出结果

```bash
# 导出结果到 CSV
python benchmark_opencv_camera.py --index 0 --test-all-resolutions --output results.csv
```

## 输出示例

```
============================================================
Benchmarking camera 0
Requested: 1280x720 @ 30 FPS, FOURCC: auto
============================================================
Actual: 1280x720 @ 30.0 FPS, FOURCC: MJPG
Backend: V4L2

Warming up for 2.0s...
Running benchmark for 10.0s...
  Progress: 3.3s, Frames: 100, FPS: 30.12
  Progress: 6.6s, Frames: 200, FPS: 30.08

────────────────────────────────────────
BENCHMARK RESULTS
────────────────────────────────────────
Camera: 0
Resolution: 1280x720 (requested: 1280x720)
FOURCC: MJPG

FPS Performance:
  ✓ Actual FPS: 30.05 (100.2% of requested 30)
  Total frames: 301
  Dropped frames: 0

Latency (ms):
  Min:  2.15
  Max:  45.23
  Avg:  33.12 ± 2.45
  P50:  33.05
  P95:  35.82
  P99:  38.91
────────────────────────────────────────
```

## 鱼眼相机测试建议

鱼眼相机通常有以下特点：

1. **分辨率**: 可能支持非标准分辨率，使用 `--scan-capabilities` 查看
2. **FPS**: 高分辨率下 FPS 可能较低，测试不同组合
3. **编码格式**: 尝试 MJPG 编码，通常能达到更高 FPS
4. **延迟**: 注意 P95/P99 延迟，对实时控制很重要

### 推荐测试流程

```bash
# 1. 首先扫描能力
python benchmark_opencv_camera.py --index 0 --scan-capabilities

# 2. 测试所有分辨率找到最佳配置
python benchmark_opencv_camera.py --index 0 --test-all-resolutions --output fisheye_resolutions.csv

# 3. 在目标分辨率测试不同 FPS
python benchmark_opencv_camera.py --index 0 --width 640 --height 480 --test-all-fps

# 4. 长时间稳定性测试
python benchmark_opencv_camera.py --index 0 --width 640 --height 480 --fps 60 --duration 60

# 5. 对比不同编码格式
python benchmark_opencv_camera.py --index 0 --width 1280 --height 720 --fps 30 --fourcc MJPG
python benchmark_opencv_camera.py --index 0 --width 1280 --height 720 --fps 30 --fourcc YUYV
```

## 常见问题

### 实际 FPS 低于请求值

1. 检查相机是否支持该分辨率+FPS 组合
2. 尝试使用 MJPG 编码 (`--fourcc MJPG`)
3. 降低分辨率
4. 检查 USB 带宽（避免 USB Hub）

### 延迟过高

1. 检查是否有其他程序占用相机
2. 尝试降低分辨率
3. 检查系统 CPU 负载
4. 考虑使用专用 USB 控制器

### 丢帧严重

1. USB 带宽不足
2. 系统负载过高
3. 相机固件问题
4. 驱动兼容性问题
