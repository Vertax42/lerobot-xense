#!/usr/bin/env python
"""
Test video decoding performance on a LeRobot video dataset.

Usage:
    python benchmarks/video/test_dataset_video_decoding.py --repo-id Vertax/xense_bi_arx5_pick_and_place_cube
"""

import argparse
import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import (
    _default_decoder_cache,
    decode_video_frames,
    get_safe_default_codec,
    get_video_info,
)


def get_video_path(dataset: LeRobotDataset, video_key: str, chunk_index: int = 0, file_index: int = 0):
    """Get video file path from dataset."""
    video_path_template = dataset.meta.video_path
    video_path = dataset.root / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return video_path


def benchmark_dataset_decoding(
    dataset: LeRobotDataset,
    num_samples: int = 20,
    frames_per_sample: int = 5,
):
    """Benchmark video decoding on a LeRobot dataset."""
    
    print("=" * 70)
    print(f"Dataset: {dataset.repo_id}")
    print("=" * 70)
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Video keys: {dataset.meta.video_keys}")
    
    # Get video info from first video
    if dataset.meta.video_keys:
        first_video_key = dataset.meta.video_keys[0]
        # Get video path for chunk 0, file 0
        video_path = get_video_path(dataset, first_video_key, chunk_index=0, file_index=0)
        print(f"\n  Sample video: {video_path}")
        
        video_info = get_video_info(str(video_path))
        print(f"  Video info:")
        print(f"    Resolution: {video_info.get('video.width')}x{video_info.get('video.height')}")
        print(f"    Codec: {video_info.get('video.codec')}")
        print(f"    FPS: {video_info.get('video.fps')}")
    
    print(f"\nDefault decoder: {get_safe_default_codec()}")
    
    # Benchmark backends
    backends = ["pyav", "torchcodec"]
    results = {}
    
    print("\n" + "=" * 70)
    print(f"Benchmark: Decoding {frames_per_sample} frames x {num_samples} samples")
    print("=" * 70)
    
    for backend in backends:
        print(f"\n--- backend='{backend}' ---")
        
        times = []
        errors = 0
        
        # Clear cache
        _default_decoder_cache.clear()
        
        # Get video path (use chunk 0, file 0)
        video_key = dataset.meta.video_keys[0]
        video_path = get_video_path(dataset, video_key, chunk_index=0, file_index=0)
        
        # Get video duration from episode timestamps
        ep_data = dataset.meta.episodes[0]
        video_duration = ep_data[f"videos/{video_key}/to_timestamp"]
        
        for i in range(num_samples):
            # Generate timestamps spread across the video
            start_time = (i * 0.5) % (video_duration - frames_per_sample / dataset.fps - 0.5)
            start_time = max(0, start_time)
            
            timestamps = [start_time + j / dataset.fps for j in range(frames_per_sample)]
            
            try:
                start = time.perf_counter()
                frames = decode_video_frames(
                    str(video_path),
                    timestamps,
                    tolerance_s=1 / dataset.fps + 1e-4,
                    backend=backend,
                )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                
                if i == 0:
                    print(f"  First decode: {elapsed*1000:.1f}ms, shape={frames.shape}")
                    
            except Exception as e:
                errors += 1
                if errors <= 2:
                    print(f"  Error sample {i}: {type(e).__name__}: {str(e)[:100]}")
        
        if times:
            avg_time = sum(times) / len(times)
            first_time = times[0]
            cached_time = sum(times[1:]) / max(1, len(times) - 1) if len(times) > 1 else first_time
            
            print(f"  Results ({len(times)} successful samples):")
            print(f"    First:  {first_time*1000:.1f}ms")
            print(f"    Cached: {cached_time*1000:.1f}ms")
            print(f"    Avg:    {avg_time*1000:.1f}ms")
            if errors > 0:
                print(f"    Errors: {errors}")
            
            results[backend] = {
                "first_ms": first_time * 1000,
                "cached_ms": cached_time * 1000,
                "avg_ms": avg_time * 1000,
                "errors": errors,
            }
        else:
            print(f"  All {errors} samples failed!")
            results[backend] = None
    
    # Comparison
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if all(results.get(b) for b in backends):
        pyav_cached = results["pyav"]["cached_ms"]
        tc_cached = results["torchcodec"]["cached_ms"]
        speedup = pyav_cached / tc_cached
        
        print(f"  pyav cached:       {pyav_cached:.1f}ms")
        print(f"  torchcodec cached: {tc_cached:.1f}ms")
        print(f"  Speedup (torchcodec vs pyav): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  Winner: torchcodec ✅")
        else:
            print(f"  Winner: pyav ✅")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test video decoding on LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Vertax/xense_bi_arx5_pick_and_place_cube",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of decoding samples",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=5,
        help="Number of frames to decode per sample",
    )
    args = parser.parse_args()
    
    print(f"\nLoading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id)
    
    if not dataset.meta.video_keys:
        print(f"Error: Dataset {args.repo_id} has no video keys!")
        return
    
    print(f"\nPyTorch: {torch.__version__}")
    try:
        import torchcodec
        print(f"torchcodec: {torchcodec.__version__}")
    except ImportError:
        print("torchcodec: not installed")
    import av
    print(f"PyAV: {av.__version__}")
    
    benchmark_dataset_decoding(
        dataset,
        num_samples=args.num_samples,
        frames_per_sample=args.frames_per_sample,
    )


if __name__ == "__main__":
    main()

