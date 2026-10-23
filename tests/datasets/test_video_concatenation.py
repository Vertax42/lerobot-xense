#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from lerobot.datasets.video_utils import concatenate_video_files


def _write_video(path, num_frames, timescale, color):
    with av.open(str(path), mode="w", options={"video_track_timescale": str(timescale)}) as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 48
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, timescale)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.thread_count = 1
        for index in range(num_frames):
            pixels = np.full((32, 48, 3), color + index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode_frames(path):
    with av.open(str(path), mode="r") as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


@pytest.mark.parametrize("timescale", [30, 15360])
@pytest.mark.parametrize("input_type", [Path, str], ids=["paths", "strings"])
def test_repeated_concatenation_preserves_timestamps_and_frames(tmp_path, timescale, input_type):
    paths = [tmp_path / f"input-{index}.mp4" for index in range(3)]
    lengths = [7, 11, 5]
    for index, (path, length) in enumerate(zip(paths, lengths, strict=True)):
        _write_video(path, length, timescale, color=40 + index * 60)
    expected_frames = [frame for path in paths for frame in _decode_frames(path)]
    output = tmp_path / "merged.mp4"

    concatenate_video_files([input_type(path) for path in paths[:2]], output)
    # aggregate_videos appends the next input to the existing destination in place.
    concatenate_video_files([input_type(output), input_type(paths[2])], output)

    with av.open(str(output), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.time_base == Fraction(1, timescale)
        assert stream.frames == sum(lengths)
        assert stream.duration * stream.time_base == Fraction(sum(lengths), 30)
        packets = [packet for packet in container.demux(stream) if packet.dts is not None]
        expected_timestamps = [index * (timescale // 30) for index in range(sum(lengths))]
        assert [packet.dts for packet in packets] == expected_timestamps
        assert [packet.pts for packet in packets] == expected_timestamps

    np.testing.assert_array_equal(_decode_frames(output), expected_frames)


def test_mismatched_time_bases_leave_existing_output_unchanged(tmp_path):
    output = tmp_path / "previously-merged.mp4"
    original = tmp_path / "original.mp4"
    _write_video(output, 7, timescale=15360, color=40)
    _write_video(original, 5, timescale=30, color=100)
    previous_bytes = output.read_bytes()

    with pytest.raises(ValueError, match="Video time_base mismatch") as error:
        concatenate_video_files([output, original], output)

    assert str(original) in str(error.value)
    assert "1/30" in str(error.value)
    assert "1/15360" in str(error.value)
    assert output.read_bytes() == previous_bytes
