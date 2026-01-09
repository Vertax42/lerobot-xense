from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DatasetProcessor:
    """Handles LeRobot dataset loading and manipulation."""

    def __init__(self):
        self.dataset: LeRobotDataset | None = None
        self.raw_hf_dataset = None
        self.to_delete_episodes: set[int] = set()
        self.features_to_remove: set[str] = set()
        self.trim_tasks: list[
            dict
        ] = []  # List of {"episode_index": int, "start_frame": int, "end_frame": int}
        self.frame_edit_tasks: list[
            dict
        ] = []  # List of {"episode_index": int, "frame_index": int, "features": Dict}

    def clear_edit_tasks(self):
        """Clears all pending edit tasks."""
        self.to_delete_episodes.clear()
        self.features_to_remove.clear()
        self.trim_tasks.clear()
        self.frame_edit_tasks.clear()

    def add_delete_episode_task(self, episode_idx: int):
        """Adds an episode to the deletion task pool."""
        self.to_delete_episodes.add(episode_idx)

    def add_remove_feature_task(self, feature_name: str):
        """Adds a feature to the removal task pool."""
        self.features_to_remove.add(feature_name)

    def add_trim_task(self, episode_idx: int, start_frame: int, end_frame: int):
        """Adds a trim task (remove frames from start to end within an episode)."""
        self.trim_tasks.append(
            {"episode_index": episode_idx, "start_frame": start_frame, "end_frame": end_frame}
        )

    def add_frame_edit_task(self, episode_idx: int, frame_idx: int, features: dict):
        """Adds a frame edit task (modify features for a specific frame)."""
        self.frame_edit_tasks.append(
            {"episode_index": episode_idx, "frame_index": frame_idx, "features": features}
        )

    def apply_edits(self, new_repo_id: str) -> LeRobotDataset:
        """Applies all pending edit tasks using official LeRobot dataset tools."""
        if self.dataset is None:
            raise ValueError("No dataset loaded")

        import copy
        import logging
        import shutil
        import tempfile

        from lerobot.datasets.dataset_tools import (
            delete_episodes,
            merge_datasets,
            remove_feature,
            split_dataset,
        )
        from lerobot.utils.constants import HF_LEROBOT_HOME

        output_dir = HF_LEROBOT_HOME / new_repo_id
        if output_dir.exists() and output_dir != self.dataset.root:
            old_path = Path(str(output_dir) + "_old")
            if old_path.exists():
                shutil.rmtree(old_path)
            shutil.move(str(output_dir), str(old_path))

        # 1. preprocess: use official tools to handle global feature removal
        current_ds = self.dataset
        temp_work_dir = Path(tempfile.mkdtemp(prefix="lerobot_edit_"))

        try:
            if self.features_to_remove:
                logging.info(f"Using official remove_feature for {list(self.features_to_remove)}")
                current_ds = remove_feature(
                    current_ds,
                    feature_names=list(self.features_to_remove),
                    repo_id=f"{new_repo_id}_feat_tmp",
                    output_dir=temp_work_dir / "feat_tmp",
                )

            # check if there are granular edits
            has_granular_edits = bool(self.trim_tasks or self.frame_edit_tasks)

            if not has_granular_edits:
                if self.to_delete_episodes:
                    return delete_episodes(
                        current_ds,
                        episode_indices=list(self.to_delete_episodes),
                        repo_id=new_repo_id,
                        output_dir=output_dir,
                    )
                else:
                    return merge_datasets([current_ds], output_repo_id=new_repo_id, output_dir=output_dir)

            # 2. granular edits: modular reconstruction
            base_features = copy.deepcopy(current_ds.meta.info["features"])
            use_videos = len(current_ds.meta.video_keys) > 0

            kept_indices = [
                i for i in range(current_ds.meta.total_episodes) if i not in self.to_delete_episodes
            ]
            modified_indices = {t["episode_index"] for t in (self.trim_tasks + self.frame_edit_tasks)}

            parts_to_merge = []
            untouched_splits = {}
            current_untouched_run = []

            def add_untouched_run():
                if current_untouched_run:
                    split_name = f"part_{len(parts_to_merge)}"
                    untouched_splits[split_name] = list(current_untouched_run)
                    parts_to_merge.append(("split", split_name))
                    current_untouched_run.clear()

            for idx in kept_indices:
                if idx in modified_indices:
                    add_untouched_run()
                    parts_to_merge.append(("modified", idx))
                else:
                    current_untouched_run.append(idx)
            add_untouched_run()

            untouched_datasets = {}
            if untouched_splits:
                logging.info("Splitting untouched episodes...")
                untouched_datasets = split_dataset(
                    current_ds, untouched_splits, output_dir=temp_work_dir / "splits"
                )

            final_parts = []
            for p_type, val in parts_to_merge:
                if p_type == "split":
                    final_parts.append(untouched_datasets[val])
                else:
                    logging.info(f"Rebuilding episode {val}...")
                    mini_ds = self._create_modified_mini_dataset(
                        val,
                        temp_work_dir / f"mod_{val}",
                        base_features,
                        use_videos=use_videos,
                        src_ds=current_ds,
                    )
                    final_parts.append(mini_ds)

            logging.info(f"Merging {len(final_parts)} parts into {new_repo_id}...")
            final_ds = merge_datasets(final_parts, output_repo_id=new_repo_id, output_dir=output_dir)

            self.clear_edit_tasks()
            return final_ds

        finally:
            if temp_work_dir.exists():
                shutil.rmtree(temp_work_dir)

    def _create_modified_mini_dataset(
        self, ep_idx: int, target_dir: Path, features: dict, use_videos: bool, src_ds: LeRobotDataset
    ) -> LeRobotDataset:
        """create a mini dataset with correct indices for the modified episode.

        Args:
            ep_idx: index of the episode in the source dataset
            target_dir: target save directory
            features: feature definition dictionary
            use_videos: whether to use video format
            src_ds: source LeRobotDataset

        Returns:
            created LeRobotDataset
        """
        import shutil

        import datasets
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import (
            embed_images,
            flatten_dict,
            get_hf_features_from_features,
            write_info,
            write_stats,
            write_tasks,
        )

        # 1. get data and reset indices
        start, end = (
            src_ds.meta.episodes["dataset_from_index"][ep_idx],
            src_ds.meta.episodes["dataset_to_index"][ep_idx],
        )
        frames = []
        physical_features = list(features.keys())

        for i in range(start, end):
            data = src_ds[i]
            processed_frame = {k: data[k] for k in physical_features if k in data}

            # critical: must reset indices to local 0 baseline, otherwise merge_datasets will fail
            processed_frame["episode_index"] = 0
            processed_frame["frame_index"] = i - start
            processed_frame["index"] = i - start

            for k in processed_frame:
                if (k in src_ds.meta.image_keys or k in src_ds.meta.video_keys) and torch.is_tensor(
                    processed_frame[k]
                ):
                    from torchvision.transforms import ToPILImage

                    processed_frame[k] = ToPILImage()(processed_frame[k])
                elif torch.is_tensor(processed_frame[k]):
                    processed_frame[k] = processed_frame[k].numpy()
            frames.append(processed_frame)

        ep_df = pd.DataFrame(frames)

        # 2. apply edits
        ep_edits = [t for t in self.frame_edit_tasks if t["episode_index"] == ep_idx]
        for edit in ep_edits:
            local_idx = edit["frame_index"]
            if 0 <= local_idx < len(ep_df):
                for feat, val in edit["features"].items():
                    if feat in ep_df.columns:
                        ep_df.at[local_idx, feat] = val.numpy() if torch.is_tensor(val) else val
        ep_trims = [t for t in self.trim_tasks if t["episode_index"] == ep_idx]
        if ep_trims:
            mask = pd.Series(True, index=ep_df.index)
            for trim in ep_trims:
                mask.loc[trim["start_frame"] : trim["end_frame"]] = False
            ep_df = ep_df[mask].copy()
            # reset indices again
            ep_df["frame_index"] = range(len(ep_df))
            ep_df["index"] = range(len(ep_df))

        # 3. create metadata
        repo_id = target_dir.name
        meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=src_ds.meta.fps,
            features=features,
            robot_type=src_ds.meta.robot_type,
            root=target_dir,
            use_videos=use_videos,
        )

        # 4. video encoding
        video_metadata = {}
        if use_videos:
            temp_img_dir = target_dir / "temp_imgs"
            temp_img_dir.mkdir(parents=True, exist_ok=True)
            try:
                for img_key in src_ds.meta.video_keys:
                    if img_key not in ep_df.columns:
                        continue
                    cam_dir = temp_img_dir / img_key
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(ep_df[img_key]):
                        img.save(cam_dir / f"frame-{i:06d}.png")
                    video_path = target_dir / f"videos/{img_key}/chunk-000/file-000.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    from lerobot.datasets.video_utils import encode_video_frames

                    encode_video_frames(imgs_dir=cam_dir, video_path=video_path, fps=src_ds.meta.fps)
                    video_metadata[img_key] = {
                        f"videos/{img_key}/chunk_index": 0,
                        f"videos/{img_key}/file_index": 0,
                        f"videos/{img_key}/from_timestamp": 0.0,
                        f"videos/{img_key}/to_timestamp": len(ep_df) / src_ds.meta.fps,
                    }
                ep_df.drop(columns=[k for k in src_ds.meta.video_keys if k in ep_df.columns], inplace=True)
            finally:
                if temp_img_dir.exists():
                    shutil.rmtree(temp_img_dir)

        # 5. physical save data
        data_path = target_dir / "data/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        hf_features = get_hf_features_from_features(features)
        filtered_data = {k: v for k, v in ep_df.to_dict(orient="list").items() if k in hf_features}
        temp_ds = datasets.Dataset.from_dict(filtered_data, features=hf_features)
        if not use_videos and len(meta.image_keys) > 0:
            temp_ds = embed_images(temp_ds)

        table = temp_ds.with_format("arrow")[:]
        writer = pq.ParquetWriter(data_path, schema=table.schema, compression="snappy")
        writer.write_table(table)
        writer.close()

        # 6. copy statistics from source episode (sufficient for small modifications, will be reaggregated when merging)
        episode_stats = self._copy_episode_stats_from_source(src_ds, ep_idx)

        # 7. build and save episode metadata (includes statistics)
        ep_meta = {
            "episode_index": 0,
            "length": len(ep_df),
            "dataset_from_index": 0,
            "dataset_to_index": len(ep_df),
            "data/chunk_index": 0,
            "data/file_index": 0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "tasks": src_ds.meta.episodes[ep_idx].get("tasks", ""),
        }
        for img_key in video_metadata:
            ep_meta.update(video_metadata[img_key])
        # flatten statistics and add to episode metadata
        ep_meta.update(flatten_dict({"stats": episode_stats}))

        # use PyArrow to write, numpy arrays need to be converted to list (参考 lerobot _flush_metadata_buffer)
        ep_meta_serialized = {k: [v.tolist() if isinstance(v, np.ndarray) else v] for k, v in ep_meta.items()}
        table = pa.Table.from_pydict(ep_meta_serialized)
        ep_meta_path = target_dir / "meta/episodes/chunk-000/file-000.parquet"
        ep_meta_path.parent.mkdir(parents=True, exist_ok=True)
        ep_writer = pq.ParquetWriter(ep_meta_path, schema=table.schema, compression="snappy")
        ep_writer.write_table(table)
        ep_writer.close()

        # 8. write info and tasks
        meta.info["total_episodes"] = 1
        meta.info["total_frames"] = len(ep_df)
        meta.info["total_tasks"] = len(src_ds.meta.tasks) if src_ds.meta.tasks is not None else 0
        meta.info["splits"] = {"train": "0:1"}
        write_info(meta.info, target_dir)
        if src_ds.meta.tasks is not None:
            write_tasks(src_ds.meta.tasks, target_dir)

        # 9. write global statistics (directly copied from source episode)
        write_stats(episode_stats, target_dir)

        return LeRobotDataset(repo_id, root=target_dir)

    def _copy_episode_stats_from_source(self, src_ds: LeRobotDataset, ep_idx: int) -> dict:
        """copy statistics from the specified episode in the source dataset.

        Args:
            src_ds: source LeRobotDataset
            ep_idx: index of the episode

        Returns:
            statistics dictionary {feature_name: {stat_name: value}}
        """
        import numpy as np
        import pandas as pd

        from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

        ep_meta = src_ds.meta.episodes[ep_idx]
        chunk_idx = ep_meta["meta/episodes/chunk_index"]
        file_idx = ep_meta["meta/episodes/file_index"]

        parquet_path = src_ds.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(parquet_path)
        episode_row = df[df["episode_index"] == ep_idx].iloc[0]

        episode_stats = {}
        for key in episode_row.index:
            if key.startswith("stats/"):
                stat_key = key.replace("stats/", "")
                parts = stat_key.split("/")
                if len(parts) == 2:
                    feature_name, stat_name = parts
                    if feature_name not in episode_stats:
                        episode_stats[feature_name] = {}

                    value = episode_row[key]
                    # handle nested arrays (image/video statistics may be serialized as nested object arrays)
                    if feature_name in src_ds.meta.features:
                        feature_dtype = src_ds.meta.features[feature_name]["dtype"]
                        if feature_dtype in ["image", "video"] and stat_name != "count":
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                flat_values = []
                                for item in value:
                                    while isinstance(item, np.ndarray):
                                        item = item.flatten()[0]
                                    flat_values.append(item)
                                value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                            elif isinstance(value, np.ndarray) and value.shape == (3,):
                                value = value.reshape(3, 1, 1)

                    episode_stats[feature_name][stat_name] = value

        return episode_stats

    def load_dataset(self, repo_id: str, root: Path | None = None) -> LeRobotDataset:
        """Loads a LeRobot dataset."""
        self.dataset = LeRobotDataset(repo_id, root=root)
        self.raw_hf_dataset = self.dataset.hf_dataset.with_format(None)
        return self.dataset

    @property
    def metadata(self):
        if self.dataset is None:
            return None
        return self.dataset.meta

    def get_episode_range(self, episode_idx: int) -> tuple[int, int]:
        """Returns (start_index, end_index) for an episode."""
        if self.dataset is None:
            return 0, 0
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_idx]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_idx]
        return int(from_idx), int(to_idx)

    def get_frame(self, frame_idx: int) -> dict[str, Any]:
        """Fetches data for a specific global frame index."""
        return self.dataset[frame_idx]

    def get_episode_data(self, episode_idx: int, keys: list[str]) -> dict[str, np.ndarray]:
        """Fetches all frames for an episode for specific keys."""
        start, end = self.get_episode_range(episode_idx)
        selected_data = self.dataset.hf_dataset.select(range(start, end))
        result = {}
        for key in keys:
            if key in selected_data.features:
                col_data = selected_data[key]
                if isinstance(col_data, list):
                    if len(col_data) > 0:
                        if torch.is_tensor(col_data[0]):
                            result[key] = torch.stack(col_data).numpy()
                        else:
                            result[key] = np.array(col_data)
                elif torch.is_tensor(col_data):
                    result[key] = col_data.numpy()
                else:
                    result[key] = np.array(col_data)
        return result
