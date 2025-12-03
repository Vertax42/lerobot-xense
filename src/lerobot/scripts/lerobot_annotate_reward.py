#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Human reward annotation tool for LeRobot datasets.

This script allows you to play through a dataset and annotate reward values
for each frame using keyboard controls.

Controls:
    Navigation:
        SPACE / D / RIGHT  : Next frame
        A / LEFT           : Previous frame
        W / UP             : Skip forward 10 frames
        S / DOWN           : Skip backward 10 frames
        N                  : Next episode
        P                  : Previous episode
        HOME               : Go to first frame of current episode
        END                : Go to last frame of current episode

    Playback:
        ENTER              : Toggle auto-play mode
        + / =              : Speed up playback
        - / _              : Slow down playback

    Reward Annotation:
        0-9                : Set reward (0=0.0, 1=0.1, ..., 9=0.9)
        R                  : Set reward to 1.0 (maximum)
        F                  : Set reward to 0.0 (minimum)
        T                  : Toggle between 0.0 and 1.0 (useful for success labeling)
        Z                  : Set reward for all remaining frames in episode to current value
        X                  : Set reward for all frames in episode to current value

    File Operations:
        CTRL+S             : Save annotations
        Q / ESC            : Quit (will prompt to save if unsaved changes)

Usage Examples:

Annotate a local dataset:
    python -m lerobot.scripts.lerobot_annotate_reward \\
        --repo-id lerobot/pusht \\
        --root data

Annotate a specific episode:
    python -m lerobot.scripts.lerobot_annotate_reward \\
        --repo-id lerobot/pusht \\
        --episode-index 0

Save to a new dataset:
    python -m lerobot.scripts.lerobot_annotate_reward \\
        --repo-id lerobot/pusht \\
        --new-repo-id lerobot/pusht_annotated

Resume annotation from saved progress:
    python -m lerobot.scripts.lerobot_annotate_reward \\
        --repo-id lerobot/pusht \\
        --load-progress annotations.json
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, REWARD
from lerobot.utils.utils import init_logging


class RewardAnnotator:
    """Interactive reward annotation tool for LeRobot datasets."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        episode_index: int | None = None,
        window_name: str = "LeRobot Reward Annotator",
    ):
        self.dataset = dataset
        self.window_name = window_name

        # Get camera keys for display
        self.camera_keys = list(dataset.meta.camera_keys)
        if not self.camera_keys:
            raise ValueError("Dataset has no camera keys for visualization")

        # Episode information
        self.total_episodes = dataset.meta.total_episodes
        self.current_episode = episode_index if episode_index is not None else 0

        # Initialize rewards array (for all frames)
        self.total_frames = dataset.meta.total_frames
        self.rewards = self._initialize_rewards()
        self.original_rewards = self.rewards.copy()

        # Navigation state
        self.current_global_idx = self._get_episode_start(self.current_episode)
        self.auto_play = False
        self.play_delay_ms = 50  # milliseconds between frames in auto-play

        # Track unsaved changes
        self.has_unsaved_changes = False

    def _initialize_rewards(self) -> np.ndarray:
        """Initialize rewards array from existing data or zeros."""
        rewards = np.zeros(self.total_frames, dtype=np.float32)

        # Check if dataset already has reward data
        if REWARD in self.dataset.meta.features:
            logging.info("Loading existing reward annotations...")
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                if REWARD in item:
                    reward_val = item[REWARD]
                    if isinstance(reward_val, torch.Tensor):
                        reward_val = reward_val.item()
                    rewards[idx] = reward_val
        else:
            logging.info("No existing rewards found, starting with zeros")

        return rewards

    def _get_episode_start(self, episode_idx: int) -> int:
        """Get the global frame index where an episode starts."""
        return self.dataset.meta.episodes["dataset_from_index"][episode_idx]

    def _get_episode_end(self, episode_idx: int) -> int:
        """Get the global frame index where an episode ends (exclusive)."""
        return self.dataset.meta.episodes["dataset_to_index"][episode_idx]

    def _get_episode_for_frame(self, global_idx: int) -> int:
        """Get the episode index for a given global frame index."""
        for ep_idx in range(self.total_episodes):
            start = self._get_episode_start(ep_idx)
            end = self._get_episode_end(ep_idx)
            if start <= global_idx < end:
                return ep_idx
        return self.total_episodes - 1

    def _get_frame_in_episode(self, global_idx: int) -> int:
        """Get the frame index within the current episode."""
        episode_start = self._get_episode_start(self._get_episode_for_frame(global_idx))
        return global_idx - episode_start

    def _get_episode_length(self, episode_idx: int) -> int:
        """Get the number of frames in an episode."""
        return self._get_episode_end(episode_idx) - self._get_episode_start(episode_idx)

    def _load_frame_image(self, global_idx: int) -> np.ndarray:
        """Load and prepare frame image for display."""
        item = self.dataset[global_idx]

        # Get images from all cameras
        images = []
        for cam_key in self.camera_keys:
            if cam_key in item:
                img = item[cam_key]
                if isinstance(img, torch.Tensor):
                    # Convert from CHW float32 [0,1] to HWC uint8 [0,255]
                    if img.dim() == 3:
                        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    else:
                        img = (img.numpy() * 255).astype(np.uint8)
                # Convert RGB to BGR for OpenCV
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)

        if not images:
            raise ValueError(f"No images found at frame {global_idx}")

        # Concatenate images horizontally if multiple cameras
        if len(images) > 1:
            # Resize all images to same height
            max_height = max(img.shape[0] for img in images)
            resized = []
            for img in images:
                if img.shape[0] != max_height:
                    scale = max_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    img = cv2.resize(img, (new_width, max_height))
                resized.append(img)
            frame = np.hstack(resized)
        else:
            frame = images[0]

        return frame

    def _draw_overlay(self, frame: np.ndarray, global_idx: int) -> np.ndarray:
        """Draw status overlay on frame."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Calculate overlay area
        overlay_height = 120
        overlay = frame[:overlay_height, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        frame[:overlay_height, :] = cv2.addWeighted(overlay, 0.7, frame[:overlay_height, :], 0.3, 0)

        # Get current episode info
        episode_idx = self._get_episode_for_frame(global_idx)
        frame_in_ep = self._get_frame_in_episode(global_idx)
        episode_len = self._get_episode_length(episode_idx)

        # Current reward value
        reward = self.rewards[global_idx]

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)

        # Line 1: Episode info
        text1 = f"Episode: {episode_idx + 1}/{self.total_episodes} | Frame: {frame_in_ep + 1}/{episode_len}"
        cv2.putText(frame, text1, (10, 25), font, 0.6, white, 1)

        # Line 2: Global frame info
        text2 = f"Global: {global_idx + 1}/{self.total_frames}"
        cv2.putText(frame, text2, (10, 50), font, 0.6, white, 1)

        # Line 3: Reward value (color-coded)
        reward_color = green if reward >= 0.8 else (yellow if reward >= 0.5 else red)
        text3 = f"Reward: {reward:.2f}"
        cv2.putText(frame, text3, (10, 75), font, 0.8, reward_color, 2)

        # Line 4: Status
        status_parts = []
        if self.auto_play:
            status_parts.append(f"PLAYING ({1000/self.play_delay_ms:.1f}fps)")
        if self.has_unsaved_changes:
            status_parts.append("*UNSAVED*")
        status = " | ".join(status_parts) if status_parts else "PAUSED"
        status_color = yellow if self.has_unsaved_changes else white
        cv2.putText(frame, status, (10, 100), font, 0.5, status_color, 1)

        # Draw reward bar
        bar_x = w - 60
        bar_y = 10
        bar_height = 100
        bar_width = 30

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)

        # Filled portion
        fill_height = int(reward * bar_height)
        fill_y = bar_y + bar_height - fill_height
        cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_width, bar_y + bar_height), reward_color, -1)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), white, 1)

        # Draw progress bar at bottom
        progress_y = h - 10
        progress_height = 8
        progress = (frame_in_ep + 1) / episode_len

        cv2.rectangle(frame, (0, progress_y - progress_height), (w, progress_y), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, progress_y - progress_height), (int(w * progress), progress_y), green, -1)

        return frame

    def _set_reward(self, value: float) -> None:
        """Set reward for current frame."""
        self.rewards[self.current_global_idx] = np.clip(value, 0.0, 1.0)
        self.has_unsaved_changes = True

    def _set_reward_range(self, start_idx: int, end_idx: int, value: float) -> None:
        """Set reward for a range of frames."""
        self.rewards[start_idx:end_idx] = np.clip(value, 0.0, 1.0)
        self.has_unsaved_changes = True

    def _navigate(self, delta: int) -> None:
        """Navigate by delta frames within the entire dataset."""
        new_idx = self.current_global_idx + delta
        new_idx = np.clip(new_idx, 0, self.total_frames - 1)
        self.current_global_idx = int(new_idx)
        self.current_episode = self._get_episode_for_frame(self.current_global_idx)

    def _go_to_episode(self, episode_idx: int) -> None:
        """Navigate to the start of a specific episode."""
        episode_idx = np.clip(episode_idx, 0, self.total_episodes - 1)
        self.current_episode = int(episode_idx)
        self.current_global_idx = self._get_episode_start(self.current_episode)

    def save_annotations(self, output_path: Path | None = None) -> None:
        """Save annotations to a JSON file for later resumption."""
        if output_path is None:
            repo_name = self.dataset.repo_id.replace("/", "_")
            output_path = Path(f"{repo_name}_reward_annotations.json")

        annotations = {
            "repo_id": self.dataset.repo_id,
            "total_frames": self.total_frames,
            "rewards": self.rewards.tolist(),
            "current_global_idx": self.current_global_idx,
            "current_episode": self.current_episode,
        }

        with open(output_path, "w") as f:
            json.dump(annotations, f)

        self.has_unsaved_changes = False
        logging.info(f"Saved annotations to {output_path}")

    def load_annotations(self, input_path: Path) -> None:
        """Load annotations from a JSON file."""
        with open(input_path) as f:
            annotations = json.load(f)

        if annotations["total_frames"] != self.total_frames:
            raise ValueError(
                f"Annotation file has {annotations['total_frames']} frames, "
                f"but dataset has {self.total_frames} frames"
            )

        self.rewards = np.array(annotations["rewards"], dtype=np.float32)
        self.current_global_idx = annotations.get("current_global_idx", 0)
        self.current_episode = annotations.get("current_episode", 0)
        self.original_rewards = self.rewards.copy()
        logging.info(f"Loaded annotations from {input_path}")

    def apply_to_dataset(
        self,
        output_dir: Path | None = None,
        new_repo_id: str | None = None,
    ) -> LeRobotDataset:
        """Apply reward annotations to the dataset and save."""
        if new_repo_id is None:
            new_repo_id = f"{self.dataset.repo_id}_annotated"

        if output_dir is None:
            output_dir = HF_LEROBOT_HOME / new_repo_id

        logging.info(f"Applying reward annotations to create {new_repo_id}...")

        # Prepare reward feature info
        reward_info = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }

        # Use modify_features to add/update reward
        if REWARD in self.dataset.meta.features:
            # Remove existing reward and add new one
            new_dataset = modify_features(
                dataset=self.dataset,
                add_features={REWARD: (self.rewards.reshape(-1, 1), reward_info)},
                remove_features=[REWARD],
                output_dir=output_dir,
                repo_id=new_repo_id,
            )
        else:
            # Just add the reward
            new_dataset = modify_features(
                dataset=self.dataset,
                add_features={REWARD: (self.rewards.reshape(-1, 1), reward_info)},
                output_dir=output_dir,
                repo_id=new_repo_id,
            )

        logging.info(f"Dataset saved to {output_dir}")
        return new_dataset

    def run(self) -> bool:
        """Run the interactive annotation loop. Returns True if changes were saved."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Print help
        print("\n" + "=" * 60)
        print("LeRobot Reward Annotator")
        print("=" * 60)
        print("\nControls:")
        print("  Navigation: SPACE/D/→ next | A/← prev | W/↑ +10 | S/↓ -10")
        print("  Episodes:   N next ep | P prev ep | HOME start | END end")
        print("  Playback:   ENTER toggle | +/- speed")
        print("  Rewards:    0-9 set | R=1.0 | F=0.0 | T toggle | Z fill rest | X fill all")
        print("  File:       CTRL+S save | Q/ESC quit")
        print("=" * 60 + "\n")

        saved = False

        try:
            while True:
                # Load and display current frame
                frame = self._load_frame_image(self.current_global_idx)
                frame = self._draw_overlay(frame, self.current_global_idx)
                cv2.imshow(self.window_name, frame)

                # Handle keyboard input
                wait_time = self.play_delay_ms if self.auto_play else 0
                key = cv2.waitKey(max(1, wait_time)) & 0xFF

                # Auto-advance in play mode
                if self.auto_play and key == 255:  # No key pressed
                    ep_end = self._get_episode_end(self.current_episode)
                    if self.current_global_idx < ep_end - 1:
                        self._navigate(1)
                    else:
                        self.auto_play = False
                    continue

                # Process key
                if key == 255:  # No key pressed
                    continue

                # Quit
                if key == ord("q") or key == 27:  # q or ESC
                    if self.has_unsaved_changes:
                        msg = "\nUnsaved changes! Press 'y' to save, 'n' to discard, or other key to cancel."
                        print(msg)
                        confirm = cv2.waitKey(0) & 0xFF
                        if confirm == ord("y"):
                            self.save_annotations()
                            saved = True
                            break
                        elif confirm == ord("n"):
                            break
                        # else continue
                    else:
                        break

                # Save (Ctrl+S or just 's' for simplicity)
                elif key == 19 or key == ord("s"):  # Ctrl+S
                    if key == 19 or (key == ord("s") and not self.auto_play):
                        self.save_annotations()
                        saved = True

                # Navigation
                elif key == ord(" ") or key == ord("d") or key == 83:  # Space, d, Right arrow
                    self._navigate(1)
                elif key == ord("a") or key == 81:  # a, Left arrow
                    self._navigate(-1)
                elif key == ord("w") or key == 82:  # w, Up arrow
                    self._navigate(10)
                elif key == ord("s") or key == 84:  # s (when not saving), Down arrow
                    if key == 84:  # Only down arrow for backward skip
                        self._navigate(-10)
                elif key == ord("n"):  # Next episode
                    if self.current_episode < self.total_episodes - 1:
                        self._go_to_episode(self.current_episode + 1)
                elif key == ord("p"):  # Previous episode
                    if self.current_episode > 0:
                        self._go_to_episode(self.current_episode - 1)
                elif key == 80:  # Home key
                    self.current_global_idx = self._get_episode_start(self.current_episode)
                elif key == 87:  # End key
                    self.current_global_idx = self._get_episode_end(self.current_episode) - 1

                # Playback
                elif key == 13:  # Enter - toggle play
                    self.auto_play = not self.auto_play
                elif key == ord("+") or key == ord("="):  # Speed up
                    self.play_delay_ms = max(10, self.play_delay_ms - 10)
                elif key == ord("-") or key == ord("_"):  # Slow down
                    self.play_delay_ms = min(500, self.play_delay_ms + 10)

                # Reward setting
                elif ord("0") <= key <= ord("9"):  # 0-9 keys
                    reward = (key - ord("0")) / 10.0
                    self._set_reward(reward)
                elif key == ord("r"):  # Max reward
                    self._set_reward(1.0)
                elif key == ord("f"):  # Min reward
                    self._set_reward(0.0)
                elif key == ord("t"):  # Toggle
                    current = self.rewards[self.current_global_idx]
                    self._set_reward(0.0 if current >= 0.5 else 1.0)
                elif key == ord("z"):  # Fill rest of episode
                    ep_end = self._get_episode_end(self.current_episode)
                    current_reward = self.rewards[self.current_global_idx]
                    self._set_reward_range(self.current_global_idx, ep_end, current_reward)
                elif key == ord("x"):  # Fill entire episode
                    ep_start = self._get_episode_start(self.current_episode)
                    ep_end = self._get_episode_end(self.current_episode)
                    current_reward = self.rewards[self.current_global_idx]
                    self._set_reward_range(ep_start, ep_end, current_reward)

        finally:
            cv2.destroyAllWindows()

        return saved


def main():
    parser = argparse.ArgumentParser(
        description="Annotate rewards in a LeRobot dataset interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset to annotate.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Start at a specific episode index.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        default=None,
        help="Repository ID for the output dataset. If not specified, appends '_annotated'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the annotated dataset.",
    )
    parser.add_argument(
        "--load-progress",
        type=Path,
        default=None,
        help="Load annotations from a previous session.",
    )
    parser.add_argument(
        "--save-progress",
        type=Path,
        default=None,
        help="Path to save annotation progress (default: {repo_id}_reward_annotations.json).",
    )
    parser.add_argument(
        "--apply-on-save",
        action="store_true",
        help="Automatically apply annotations to create a new dataset when saving.",
    )

    args = parser.parse_args()

    init_logging()

    # Load dataset
    logging.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    logging.info(f"Loaded {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")

    # Create annotator
    annotator = RewardAnnotator(
        dataset=dataset,
        episode_index=args.episode_index,
    )

    # Load previous progress if specified
    if args.load_progress and args.load_progress.exists():
        annotator.load_annotations(args.load_progress)

    # Run interactive annotation
    try:
        saved = annotator.run()

        if saved or annotator.has_unsaved_changes:
            # Ask if user wants to apply to dataset
            prompt = "\nApply annotations to create new dataset? [y/N]: "
            if args.apply_on_save or input(prompt).lower() == "y":
                annotator.apply_to_dataset(
                    output_dir=args.output_dir,
                    new_repo_id=args.new_repo_id,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if annotator.has_unsaved_changes:
            if input("Save progress before exit? [y/N]: ").lower() == "y":
                annotator.save_annotations(args.save_progress)


if __name__ == "__main__":
    main()
