#!/usr/bin/env python
"""
Test video backend performance in PyTorch training scenarios.

This script tests:
1. Default video backend initialization (torchcodec)
2. Single-frame DataLoader performance
3. Multi-frame DataLoader performance (with delta_timestamps)
4. Long sequence test (10, 30, 50, 100, 200, ..., up to 3000 frames) - optional

Usage:
    python benchmarks/video/test_training_video_backend.py
    python benchmarks/video/test_training_video_backend.py --repo-id your/dataset
    python benchmarks/video/test_training_video_backend.py --test-long-sequence
    python benchmarks/video/test_training_video_backend.py --test-long-sequence --max-frames 600
"""

import argparse
import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_safe_default_codec


def test_backend_initialization(repo_id: str):
    """Test if torchcodec is properly initialized as default backend."""
    print("=" * 70)
    print("1. 检查默认 video_backend 初始化")
    print("=" * 70)

    print(f"\nget_safe_default_codec() 返回: {get_safe_default_codec()}")

    # Test default initialization
    dataset_default = LeRobotDataset(repo_id, video_backend=None)
    print(f'dataset (video_backend=None) 实际使用: {dataset_default.video_backend}')

    # Test explicit pyav
    dataset_pyav = LeRobotDataset(repo_id, video_backend="pyav")
    print(f'dataset (video_backend="pyav") 实际使用: {dataset_pyav.video_backend}')

    # Test explicit torchcodec
    dataset_tc = LeRobotDataset(repo_id, video_backend="torchcodec")
    print(f'dataset (video_backend="torchcodec") 实际使用: {dataset_tc.video_backend}')

    success = dataset_default.video_backend == "torchcodec"
    print("\n✅ torchcodec 默认初始化成功!" if success else "\n❌ torchcodec 初始化失败")

    del dataset_default, dataset_pyav, dataset_tc
    return success


def test_single_frame_dataloader(
    repo_id: str,
    batch_size: int = 16,
    num_workers: int = 4,
    num_batches: int = 20,
):
    """Test single-frame DataLoader performance."""
    print("\n" + "=" * 70)
    print("2. 单帧读取 DataLoader 性能测试")
    print("=" * 70)

    backends = ["pyav", "torchcodec"]
    results = {}

    for backend in backends:
        print(f'\n--- Testing video_backend="{backend}" ---')

        dataset = LeRobotDataset(repo_id, video_backend=backend, delta_timestamps=None)
        print(f"  Dataset size: {len(dataset)} frames")
        print(f"  Video backend: {dataset.video_backend}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Warmup
        print("  Warming up...")
        data_iter = iter(dataloader)
        _ = next(data_iter)

        # Benchmark
        print(f"  Testing {num_batches} batches...")
        times = []

        for _ in range(num_batches):
            start = time.perf_counter()
            try:
                _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                _ = next(data_iter)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time

        print("  Results:")
        print(f"    Avg batch time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
        print(f"    Throughput: {throughput:.1f} samples/sec")

        results[backend] = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "throughput": throughput,
        }

        del dataloader, dataset, data_iter

    return results


def test_multi_frame_dataloader(
    repo_id: str,
    batch_size: int = 16,
    num_workers: int = 4,
    num_batches: int = 15,
    num_frames: int = 5,
):
    """Test multi-frame DataLoader performance with delta_timestamps."""
    print("\n" + "=" * 70)
    print(f"3. 多帧读取 DataLoader 性能测试 ({num_frames} frames/key)")
    print("=" * 70)

    # Get dataset info first
    temp_dataset = LeRobotDataset(repo_id)
    fps = temp_dataset.fps
    video_keys = temp_dataset.meta.video_keys
    del temp_dataset

    # Build delta_timestamps
    dt = 1.0 / fps
    timestamps = [-(num_frames - 1 - i) * dt for i in range(num_frames)]

    delta_timestamps = {key: timestamps for key in video_keys}
    print(f"delta_timestamps ({num_frames} frames): {timestamps}")

    backends = ["pyav", "torchcodec"]
    results = {}

    for backend in backends:
        print(f'\n--- Testing video_backend="{backend}" ({num_frames} frames/key) ---')

        dataset = LeRobotDataset(
            repo_id,
            video_backend=backend,
            delta_timestamps=delta_timestamps,
        )
        print(f"  Video backend: {dataset.video_backend}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )

        # Warmup
        print("  Warming up...")
        data_iter = iter(dataloader)
        warmup_batch = next(data_iter)

        # Show shape
        for key in video_keys[:1]:
            if key in warmup_batch:
                print(f"  {key} shape: {warmup_batch[key].shape}")

        # Benchmark
        print(f"  Testing {num_batches} batches...")
        times = []

        for _ in range(num_batches):
            start = time.perf_counter()
            try:
                _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                _ = next(data_iter)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / avg_time

        print("  Results:")
        print(f"    Avg batch time: {avg_time*1000:.1f}ms ± {std_time*1000:.1f}ms")
        print(f"    Throughput: {throughput:.1f} samples/sec")

        results[backend] = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "throughput": throughput,
        }

        del dataloader, dataset, data_iter

    return results


def test_long_sequence_dataloader(
    repo_id: str,
    num_workers: int = 4,
    num_batches: int = 5,
    max_frames: int | None = None,
):
    """Test very long sequence DataLoader performance (many frames per sample).

    Args:
        repo_id: HuggingFace dataset repo ID
        num_workers: Number of DataLoader workers
        num_batches: Number of batches to test per configuration
        max_frames: Maximum frames to test (None = auto detect from dataset)
    """
    print("\n" + "=" * 70)
    print("4. 超长序列测试 (模拟视频序列建模)")
    print("=" * 70)

    # Get dataset info
    temp_dataset = LeRobotDataset(repo_id)
    fps = temp_dataset.fps
    video_keys = temp_dataset.meta.video_keys
    num_cameras = len(video_keys)

    # Get episode lengths
    ep_lengths = [temp_dataset.meta.episodes[i]["length"] for i in range(temp_dataset.num_episodes)]
    min_ep_length = min(ep_lengths)
    max_ep_length = max(ep_lengths)
    del temp_dataset

    print(f"数据集: {repo_id}")
    print(f"FPS: {fps}, 相机数: {num_cameras}")
    print(f"Episode 长度范围: {min_ep_length} - {max_ep_length} 帧")

    # Determine max testable frames
    # Use max_ep_length - 10 as safety margin
    max_testable = max_ep_length - 10
    if max_frames is not None:
        max_testable = min(max_frames, max_testable)

    print(f"最大可测试帧数: {max_testable} 帧/相机")

    # Build test configs dynamically based on max_testable
    # batch_size is scaled down for larger frame counts to avoid OOM
    # Target: keep total_frames_per_batch roughly constant
    base_configs = [10, 30, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 2500, 3000]
    test_configs = []

    # Scale batch_size based on frames: more frames = smaller batch
    # Target ~800 frames per batch (10 frames × 5 cameras × 16 batch = 800)
    target_frames_per_batch = 800

    for frames in base_configs:
        if frames <= max_testable:
            total_frames_per_sample = frames * num_cameras
            batch_size = max(1, target_frames_per_batch // total_frames_per_sample)
            test_configs.append((frames, batch_size))

    # Add max_testable if not already included
    if max_testable not in [c[0] for c in test_configs] and max_testable > 0:
        total_frames_per_sample = max_testable * num_cameras
        batch_size = max(1, target_frames_per_batch // total_frames_per_sample)
        test_configs.append((max_testable, batch_size))
        test_configs.sort(key=lambda x: x[0])

    print(f"测试配置: {[(f'{c[0]}帧/bs{c[1]}') for c in test_configs]}")

    backends = ["pyav", "torchcodec"]
    all_results = {}

    for num_frames, batch_size in test_configs:
        total_frames_per_sample = num_frames * num_cameras

        # Dynamically adjust num_workers based on memory requirements
        # More frames = fewer workers to avoid OOM
        if num_frames >= 200:
            actual_workers = 1  # Single worker for very large sequences
        elif num_frames >= 100:
            actual_workers = min(2, num_workers)
        else:
            actual_workers = num_workers

        # Adjust num_batches based on sequence length
        if num_frames >= 300:
            actual_batches = max(3, num_batches // 2)
        elif num_frames >= 100:
            actual_batches = num_batches
        else:
            actual_batches = max(num_batches, 10)  # At least 10 batches for small sequences

        print(f"\n--- {num_frames} 帧/相机 × {num_cameras} 相机 = {total_frames_per_sample} 帧/样本 ---")
        print(f"    (workers={actual_workers}, batches={actual_batches}, batch_size={batch_size})")

        # Build delta_timestamps
        dt = 1.0 / fps
        timestamps = [-(num_frames - 1 - i) * dt for i in range(num_frames)]
        delta_timestamps = {key: timestamps for key in video_keys}

        results = {}

        for backend in backends:
            print(f"  {backend}:", end=" ", flush=True)

            try:
                # Force garbage collection before large tests
                if num_frames >= 100:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                dataset = LeRobotDataset(
                    repo_id,
                    video_backend=backend,
                    delta_timestamps=delta_timestamps,
                )

                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=actual_workers,
                    pin_memory=False if num_frames >= 100 else True,  # Disable pin_memory for large sequences
                    prefetch_factor=1 if num_frames >= 100 else 2,  # Reduce prefetch for large sequences
                )

                # Warmup
                data_iter = iter(dataloader)
                _ = next(data_iter)

                # Benchmark
                times = []
                for _ in range(actual_batches):
                    start = time.perf_counter()
                    try:
                        _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        _ = next(data_iter)
                    times.append(time.perf_counter() - start)

                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / avg_time
                frames_per_sec = total_frames_per_sample * batch_size / avg_time

                print(f"{avg_time*1000:.0f}ms ± {std_time*1000:.0f}ms, {frames_per_sec:.0f} frames/s")

                results[backend] = {
                    "avg_time_ms": avg_time * 1000,
                    "std_time_ms": std_time * 1000,
                    "throughput": throughput,
                    "frames_per_sec": frames_per_sec,
                }

                del dataloader, dataset, data_iter

                # Clean up after each backend test for large sequences
                if num_frames >= 100:
                    gc.collect()

            except Exception as e:
                print(f"Error: {type(e).__name__}: {str(e)[:60]}")
                results[backend] = None
                # Clean up on error
                gc.collect()

        # Compare
        if all(results.get(b) for b in backends):
            speedup = results["pyav"]["avg_time_ms"] / results["torchcodec"]["avg_time_ms"]
            winner = "torchcodec" if speedup > 1 else "pyav"
            print(f"  Winner: {winner} ({max(speedup, 1/speedup):.2f}x)")

        all_results[num_frames] = results

    return all_results


def print_summary(results_single: dict, results_multi: dict, results_long: dict | None = None):
    """Print summary of all tests."""
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\n单帧读取:")
    print(f"{'Backend':<15} {'Avg Time (ms)':<18} {'Throughput (samples/s)':<25}")
    print("-" * 60)
    for backend, r in results_single.items():
        print(f"{backend:<15} {r['avg_time_ms']:<18.1f} {r['throughput']:<25.1f}")

    if len(results_single) == 2:
        speedup = results_single["pyav"]["throughput"] / results_single["torchcodec"]["throughput"]
        winner = "torchcodec" if speedup < 1 else "pyav"
        speedup_val = 1 / speedup if speedup < 1 else speedup
        print(f"\n🏆 {winner} 更快: {speedup_val:.2f}x throughput")

    print("\n多帧读取:")
    print(f"{'Backend':<15} {'Avg Time (ms)':<18} {'Throughput (samples/s)':<25}")
    print("-" * 60)
    for backend, r in results_multi.items():
        print(f"{backend:<15} {r['avg_time_ms']:<18.1f} {r['throughput']:<25.1f}")

    if len(results_multi) == 2:
        speedup = results_multi["pyav"]["throughput"] / results_multi["torchcodec"]["throughput"]
        winner = "torchcodec" if speedup < 1 else "pyav"
        speedup_val = 1 / speedup if speedup < 1 else speedup
        print(f"\n🏆 {winner} 更快: {speedup_val:.2f}x throughput")

    if results_long:
        print("\n超长序列测试:")
        print(f"{'Frames':<10} {'pyav (ms)':<18} {'torchcodec (ms)':<18} {'Winner':<12} {'Speedup':<10}")
        print("-" * 70)
        for num_frames, results in results_long.items():
            if results.get("pyav") and results.get("torchcodec"):
                pyav_ms = results["pyav"]["avg_time_ms"]
                pyav_std = results["pyav"].get("std_time_ms", 0)
                tc_ms = results["torchcodec"]["avg_time_ms"]
                tc_std = results["torchcodec"].get("std_time_ms", 0)
                speedup = pyav_ms / tc_ms
                winner = "torchcodec" if speedup > 1 else "pyav"
                speedup_val = max(speedup, 1 / speedup)
                pyav_str = f"{pyav_ms:.0f}±{pyav_std:.0f}"
                tc_str = f"{tc_ms:.0f}±{tc_std:.0f}"
                print(f"{num_frames:<10} {pyav_str:<18} {tc_str:<18} {winner:<12} {speedup_val:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Test video backend performance in training")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Vertax/xense_bi_arx5_pick_and_place_cube",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for all tests")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of batches to test")
    parser.add_argument("--num-frames", type=int, default=5, help="Frames per sample for multi-frame test")
    parser.add_argument(
        "--test-long-sequence",
        action="store_true",
        help="Run long sequence test (10, 30, 100, ..., up to max-frames)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per camera for long sequence test (default: auto from dataset)",
    )
    args = parser.parse_args()

    print(f"\nPyTorch: {torch.__version__}")
    try:
        import torchcodec

        print(f"torchcodec: {torchcodec.__version__}")
    except ImportError:
        print("torchcodec: not installed")
    import av

    print(f"PyAV: {av.__version__}")

    # Test 1: Backend initialization
    test_backend_initialization(args.repo_id)

    # Test 2: Single-frame DataLoader
    results_single = test_single_frame_dataloader(
        args.repo_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
    )

    # Test 3: Multi-frame DataLoader
    results_multi = test_multi_frame_dataloader(
        args.repo_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        num_frames=args.num_frames,
    )

    # Test 4: Long sequence test (optional)
    results_long = None
    if args.test_long_sequence:
        results_long = test_long_sequence_dataloader(
            args.repo_id,
            num_workers=args.num_workers,
            num_batches=5,  # Fewer batches for long sequences
            max_frames=args.max_frames,
        )

    # Print summary
    print_summary(results_single, results_multi, results_long)

    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
