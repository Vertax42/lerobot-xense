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

from types import SimpleNamespace

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.aggregate import aggregate_data
from lerobot.datasets.utils import DEFAULT_DATA_PATH, get_hf_features_from_features


@pytest.mark.parametrize("legacy_source", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_aggregate_data_preserves_array_schema_and_values(tmp_path, legacy_source, rotate):
    features = {
        "action": {"dtype": "float32", "shape": (29,)},
        "observation.state": {"dtype": "float32", "shape": (29,)},
        "observation.tracker_confidence": {"dtype": "float32", "shape": (2,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
    }
    hf_features = get_hf_features_from_features(features)
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["clean plates"], name="task"))
    destination = SimpleNamespace(
        root=tmp_path / "merged",
        features=features,
        image_keys=[],
        tasks=tasks,
        info={"total_episodes": 0, "total_frames": 0},
    )
    data_idx = {"chunk": 0, "file": 0}
    expected_arrays = {key: [] for key in features if key not in ("episode_index", "index", "task_index")}

    for source_index, num_frames in enumerate((3, 2)):
        source_root = tmp_path / f"source-{source_index}"
        source_path = source_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
        source_path.parent.mkdir(parents=True)
        columns = {
            "episode_index": [0] * num_frames,
            "index": list(range(num_frames)),
            "task_index": [0] * num_frames,
        }
        for key in expected_arrays:
            width = features[key]["shape"][0]
            values = np.arange(num_frames * width, dtype=np.float32).reshape(num_frames, width) + source_index
            columns[key] = list(values)
            expected_arrays[key].extend(values)
        frame = pd.DataFrame(columns)
        if legacy_source:
            frame.to_parquet(source_path)
            assert pa.types.is_list(pq.read_schema(source_path).field("action").type)
        else:
            pq.write_table(pa.Table.from_pydict(columns, schema=hf_features.arrow_schema), source_path)
        source = SimpleNamespace(
            root=source_root,
            episodes=pd.DataFrame({"data/chunk_index": [0], "data/file_index": [0]}),
            tasks=tasks,
        )

        data_idx = aggregate_data(
            source,
            destination,
            data_idx,
            data_files_size_in_mb=1e-6 if rotate else 100,
            chunk_size=1000,
        )
        destination.info["total_episodes"] += 1
        destination.info["total_frames"] += num_frames

        # Check the first write, subsequent append, and rotation to a new file.
        for path in (destination.root / "data").rglob("*.parquet"):
            schema = pq.read_schema(path)
            assert b"pandas" not in (schema.metadata or {})
            for key in expected_arrays:
                assert schema.field(key).type == pa.list_(pa.float32(), features[key]["shape"][0])

    paths = sorted((destination.root / "data").rglob("*.parquet"))
    assert len(paths) == (2 if rotate else 1)
    merged = pa.concat_tables([pq.read_table(path) for path in paths])
    assert merged["index"].to_pylist() == list(range(5))
    assert merged["episode_index"].to_pylist() == [0, 0, 0, 1, 1]
    assert merged["task_index"].to_pylist() == [0] * 5
    assert set(merged.column_names) == set(features)
    for key, expected in expected_arrays.items():
        np.testing.assert_array_equal(merged[key].to_pylist(), expected)
    assert datasets.Features.from_arrow_schema(merged.schema) == hf_features
