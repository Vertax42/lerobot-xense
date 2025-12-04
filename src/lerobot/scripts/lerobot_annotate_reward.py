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
    Frame Navigation (WASD):
        SPACE / D          : Next frame (inherits reward from previous frame)
        A                  : Previous frame
        W                  : Skip forward 10 frames (inherits reward)
        S                  : Skip backward 10 frames

    Episode Navigation (Arrow Left/Right):
        LEFT               : Previous episode
        RIGHT              : Next episode
        N / P              : Next / Previous episode (alternative)
        HOME               : Go to first frame of current episode
        END                : Go to last frame of current episode

    Reward Adjustment (Arrow Up/Down):
        UP                 : Increase reward by 0.1
        DOWN               : Decrease reward by 0.1
        0-9                : Set reward (0=0.0, 1=0.1, ..., 9=0.9)
        R                  : Set reward to 1.0 (maximum)
        F                  : Set reward to 0.0 (minimum)
        T                  : Toggle between 0.0 and 1.0 (useful for success labeling)
        Z                  : Set reward for all remaining frames in episode to current value
        X                  : Set reward for all frames in episode to current value

Note: When moving forward, the reward is automatically inherited from the previous frame.
      This allows you to just adjust with UP/DOWN arrows on each frame.

    Playback:
        ENTER              : Toggle auto-play mode
        + / =              : Speed up playback
        - / _              : Slow down playback

    File Operations:
        F5 / ;             : Save annotations
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
from PIL import Image, ImageDraw, ImageFont

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
        window_size: tuple[int, int] = (1600, 900),
    ):
        self.dataset = dataset
        self.window_name = window_name
        self.window_size = window_size

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
        self.skip_step = 10  # frames to skip with W/S keys (adjustable with PageUp/PageDown)

        # Track which frames have been visited (for reward inheritance)
        self.visited_frames: set[int] = set()
        # Mark first frame as visited
        self.visited_frames.add(self.current_global_idx)

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

        # Layout images in a grid based on count
        frame = self._arrange_images_grid(images)
        return frame

    def _arrange_images_grid(self, images: list[np.ndarray]) -> np.ndarray:
        """Arrange images in a grid layout based on count.

        Layout rules:
        - 1 image: single
        - 2 images: 1 row x 2 cols
        - 3 images: 2 rows (2 on top, 1 centered below)
        - 4 images: 2 rows x 2 cols
        - 5-6 images: 2 rows x 3 cols
        - 7-9 images: 3 rows x 3 cols
        - etc.
        """
        n = len(images)
        if n == 1:
            return images[0]

        # Determine grid dimensions
        if n == 2:
            rows, cols = 1, 2
        elif n == 3:
            rows, cols = 2, 2  # Will center the bottom one
        elif n == 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        elif n <= 9:
            rows, cols = 3, 3
        else:
            # For more images, calculate optimal grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))

        # Resize all images to the same size
        target_h = max(img.shape[0] for img in images)
        target_w = max(img.shape[1] for img in images)

        resized = []
        for img in images:
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h))
            resized.append(img)

        # Create grid
        grid_rows = []
        idx = 0
        for r in range(rows):
            row_images = []
            for c in range(cols):
                if idx < n:
                    row_images.append(resized[idx])
                    idx += 1
                else:
                    # Pad with black image for empty cells
                    black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    row_images.append(black)

            # Handle centering for incomplete rows (like 3 images: 2 top, 1 bottom centered)
            if len(row_images) < cols and r == rows - 1:
                # This is the last row with fewer images - already handled by padding
                pass

            row_img = np.hstack(row_images)
            grid_rows.append(row_img)

        # Special case for 3 images: center the bottom image
        if n == 3:
            # Top row: 2 images
            top_row = np.hstack([resized[0], resized[1]])
            # Bottom row: 1 image centered with padding
            pad_width = target_w // 2
            left_pad = np.zeros((target_h, pad_width, 3), dtype=np.uint8)
            right_pad = np.zeros((target_h, pad_width, 3), dtype=np.uint8)
            bottom_row = np.hstack([left_pad, resized[2], right_pad])
            # Make sure widths match
            if bottom_row.shape[1] != top_row.shape[1]:
                bottom_row = cv2.resize(bottom_row, (top_row.shape[1], target_h))
            return np.vstack([top_row, bottom_row])

        return np.vstack(grid_rows)

    def _draw_text_pil(
        self, frame: np.ndarray, text: str, pos: tuple, color: tuple, font_size: int = 14
    ) -> np.ndarray:
        """Draw text using PIL with FiraCode font."""
        # Convert BGR to RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Load FiraCode font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/firacode/FiraCode-Retina.ttf", font_size)
        except OSError:
            # Fallback to default font
            font = ImageFont.load_default()

        # PIL uses RGB, convert BGR color to RGB
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)

        # Convert back to BGR for OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _draw_overlay(self, frame: np.ndarray, global_idx: int) -> np.ndarray:
        """Draw status overlay on frame."""
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Calculate overlay area
        overlay_height = 120
        overlay = frame[:overlay_height, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        frame[:overlay_height, :] = cv2.addWeighted(overlay, 0.6, frame[:overlay_height, :], 0.4, 0)

        # Get current episode info
        episode_idx = self._get_episode_for_frame(global_idx)
        frame_in_ep = self._get_frame_in_episode(global_idx)
        episode_len = self._get_episode_length(episode_idx)

        # Current reward value
        reward = self.rewards[global_idx]

        # Colors (BGR for OpenCV)
        white = (255, 255, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)
        red = (0, 0, 255)
        gray = (180, 180, 180)

        # Line 1: Episode info
        text1 = f"Ep: {episode_idx + 1}/{self.total_episodes} | Frame: {frame_in_ep + 1}/{episode_len}"
        frame = self._draw_text_pil(frame, text1, (10, 8), gray, 14)

        # Line 2: Global frame info
        text2 = f"Global: {global_idx + 1}/{self.total_frames}"
        frame = self._draw_text_pil(frame, text2, (10, 30), gray, 14)

        # Line 3: Reward value (color-coded)
        reward_color = green if reward >= 0.8 else (yellow if reward >= 0.5 else red)
        text3 = f"Reward: {reward:.2f}"
        frame = self._draw_text_pil(frame, text3, (10, 55), reward_color, 18)

        # Line 4: Status
        status_parts = []
        if self.auto_play:
            status_parts.append(f"PLAY ({1000/self.play_delay_ms:.0f}fps)")
        else:
            status_parts.append("PAUSE")
        status_parts.append(f"Skip:{self.skip_step}")
        if self.has_unsaved_changes:
            status_parts.append("*UNSAVED*")
        status = " | ".join(status_parts)
        status_color = yellow if self.has_unsaved_changes else gray
        frame = self._draw_text_pil(frame, status, (10, 82), status_color, 12)

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

        # Draw reward line chart at bottom
        chart_height = 80
        chart_margin = 10
        chart_top = h - chart_height - chart_margin
        chart_bottom = h - chart_margin

        # Chart background
        cv2.rectangle(frame, (0, chart_top), (w, chart_bottom), (30, 30, 30), -1)

        # Draw horizontal grid lines (0.0, 0.5, 1.0)
        for val in [0.0, 0.5, 1.0]:
            y_pos = int(chart_bottom - val * (chart_height - 4) - 2)
            cv2.line(frame, (0, y_pos), (w, y_pos), (60, 60, 60), 1)
            # Label
            cv2.putText(frame, f"{val:.1f}", (5, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # Get episode reward data
        ep_start = self._get_episode_start(episode_idx)
        ep_end = self._get_episode_end(episode_idx)
        ep_rewards = self.rewards[ep_start:ep_end]

        # Draw reward line chart
        if len(ep_rewards) > 1:
            points = []
            for i, r in enumerate(ep_rewards):
                x = int(i / (len(ep_rewards) - 1) * (w - 1))
                y = int(chart_bottom - r * (chart_height - 4) - 2)
                points.append((x, y))

            # Draw line segments with color based on reward value (thin line)
            for i in range(len(points) - 1):
                r_val = ep_rewards[i]
                color = green if r_val >= 0.8 else (yellow if r_val >= 0.5 else red)
                cv2.line(frame, points[i], points[i + 1], color, 1)

        # Draw current position marker
        marker_x = int(frame_in_ep / max(1, episode_len - 1) * (w - 1))
        cv2.line(frame, (marker_x, chart_top), (marker_x, chart_bottom), white, 2)

        # Draw small circle at current reward position
        current_y = int(chart_bottom - reward * (chart_height - 4) - 2)
        cv2.circle(frame, (marker_x, current_y), 5, reward_color, -1)
        cv2.circle(frame, (marker_x, current_y), 5, white, 1)

        # Draw progress bar on top of chart
        progress_y = chart_top
        progress_height = 4
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

    def _navigate(self, delta: int, inherit_reward: bool = True) -> None:
        """Navigate by delta frames within the entire dataset.

        Args:
            delta: Number of frames to move (positive = forward, negative = backward)
            inherit_reward: If True and moving forward to unvisited frames,
                           copy current frame's reward to those frames
        """
        old_idx = self.current_global_idx
        new_idx = int(np.clip(old_idx + delta, 0, self.total_frames - 1))

        # When moving forward, inherit reward only to unvisited frames
        if inherit_reward and delta > 0 and new_idx > old_idx:
            current_reward = self.rewards[old_idx]
            # Copy reward only to unvisited frames from current+1 to new_idx (inclusive)
            for idx in range(old_idx + 1, new_idx + 1):
                if idx not in self.visited_frames:
                    self.rewards[idx] = current_reward
                    self.has_unsaved_changes = True

        # Mark the new frame as visited
        self.visited_frames.add(new_idx)

        self.current_global_idx = new_idx
        self.current_episode = self._get_episode_for_frame(self.current_global_idx)

    def _go_to_episode(self, episode_idx: int) -> None:
        """Navigate to the start of a specific episode."""
        episode_idx = np.clip(episode_idx, 0, self.total_episodes - 1)
        self.current_episode = int(episode_idx)
        self.current_global_idx = self._get_episode_start(self.current_episode)
        # Mark the new frame as visited
        self.visited_frames.add(self.current_global_idx)

    def _get_default_annotation_path(self) -> Path:
        """Get the default path for saving annotations."""
        repo_name = self.dataset.repo_id.replace("/", "_")
        annotation_dir = HF_LEROBOT_HOME / "datasets" / "annotated" / f"{repo_name}_reward"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        return annotation_dir / "annotations.json"

    def save_annotations(self, output_path: Path | None = None) -> None:
        """Save annotations to a JSON file for later resumption."""
        if output_path is None:
            output_path = self._get_default_annotation_path()

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
        push_to_hub: bool = False,
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

        # Push to HuggingFace Hub if requested
        if push_to_hub:
            logging.info(f"Pushing dataset to HuggingFace Hub as {new_repo_id}...")
            new_dataset.push_to_hub()
            logging.info(f"Dataset pushed to hub: https://huggingface.co/datasets/{new_repo_id}")

        return new_dataset

    def _center_window(self) -> None:
        """Center the window on the screen.

        cv2.moveWindow moves the top-left corner of the window content area
        (excluding title bar and borders). We calculate the position so that
        the window content is centered on screen.
        """
        try:
            # Use tkinter to get screen dimensions
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()  # Hide the tkinter window
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()

            # Get the requested window size
            win_width, win_height = self.window_size

            # Calculate position for top-left corner of window content
            # to center the window on screen
            x = (screen_width - win_width) // 2
            y = (screen_height - win_height) // 2

            # Account for window title bar (approximately 30-40 pixels on most systems)
            title_bar_height = 35
            y = y - title_bar_height // 2

            logging.info(
                f"Centering window: screen={screen_width}x{screen_height}, "
                f"window={win_width}x{win_height}, position=({x}, {y})"
            )

            cv2.moveWindow(self.window_name, x, y)
        except Exception as e:
            # If tkinter is not available or any error, skip centering
            logging.warning(f"Could not center window: {e}")

    def run(self) -> bool:
        """Run the interactive annotation loop. Returns True if changes were saved."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        # Print help
        print("\n" + "=" * 70)
        print("LeRobot Reward Annotator")
        print("=" * 70)
        print("\nControls:")
        print("  Frame Nav:    SPACE/D next | A prev | W +skip_step | S -skip_step")
        print("  Episode Nav:  ←/→ prev/next episode | N/P alternative | HOME/END")
        print("  Reward:       ↑/↓ +0.1/-0.1 | 0-9 set | R=1.0 | F=0.0 | T toggle")
        print("  Batch Reward: Z fill rest of episode | X fill entire episode")
        print("  Playback:     ENTER toggle | +/- speed")
        print(f"  Skip Step:    PageUp +5 | PageDown -5 (min 10, affects W/S: {self.skip_step})")
        print("  File:         F5 or ; save | Q/ESC quit")
        print("-" * 70)
        print("  Note: Moving forward auto-inherits reward from previous frame")
        print("=" * 70 + "\n")

        saved = False
        # first_frame = True

        try:
            while True:
                # Load and display current frame
                frame = self._load_frame_image(self.current_global_idx)
                frame = self._draw_overlay(frame, self.current_global_idx)
                cv2.imshow(self.window_name, frame)

                # Center window after first frame is displayed
                # if first_frame:
                #     cv2.waitKey(1)  # Let the window render
                #     self._center_window()
                #     first_frame = False

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
                        # Use terminal input for confirmation (more reliable than cv2.waitKey)
                        cv2.destroyAllWindows()
                        confirm = input("\nUnsaved changes! Save? [y/n/c(cancel)]: ").strip().lower()
                        if confirm == "y":
                            self.save_annotations()
                            saved = True
                            return saved
                        elif confirm == "n":
                            return saved
                        else:
                            # Cancel - recreate window
                            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
                            self._center_window()
                            continue
                    else:
                        break

                # Save (F5 or semicolon key)
                elif key == 194 or key == ord(";"):  # F5 or ;
                    self.save_annotations()
                    saved = True

                # Navigation - WASD for frame navigation
                elif key == ord(" ") or key == ord("d"):  # Space, d - next frame
                    self._navigate(1)
                elif key == ord("a"):  # a - previous frame
                    self._navigate(-1)
                elif key == ord("w"):  # w - skip forward by skip_step frames
                    self._navigate(self.skip_step)
                elif key == ord("s"):  # S key - skip backward by skip_step frames
                    self._navigate(-self.skip_step)

                # Arrow keys - Left/Right for episode, Up/Down for reward adjustment
                elif key == 83:  # Right arrow - next episode
                    if self.current_episode < self.total_episodes - 1:
                        self._go_to_episode(self.current_episode + 1)
                elif key == 81:  # Left arrow - previous episode
                    if self.current_episode > 0:
                        self._go_to_episode(self.current_episode - 1)
                elif key == 82:  # Up arrow - increase reward by 0.1
                    current = self.rewards[self.current_global_idx]
                    self._set_reward(min(1.0, current + 0.1))
                elif key == 84:  # Down arrow - decrease reward by 0.1
                    current = self.rewards[self.current_global_idx]
                    self._set_reward(max(0.0, current - 0.1))

                # N/P still work for episode navigation
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

                # Skip step adjustment (PageUp/PageDown)
                elif key == 85:  # PageUp - increase skip step by 5
                    self.skip_step += 5
                elif key == 86:  # PageDown - decrease skip step by 5 (min 10)
                    self.skip_step = max(10, self.skip_step - 5)

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
        help="Load annotations from a previous session. Auto-loads from default path if exists.",
    )
    parser.add_argument(
        "--save-progress",
        type=Path,
        default=None,
        help="Path to save annotation progress. Default: HF_LEROBOT_HOME/datasets/annotated/",
    )
    parser.add_argument(
        "--apply-on-save",
        action="store_true",
        help="Automatically apply annotations to create a new dataset when saving.",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default="1920x1080",
        help="Window size in format WIDTHxHEIGHT (default: 1920x1080).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the annotated dataset to HuggingFace Hub after creation.",
    )

    args = parser.parse_args()

    # Parse window size
    try:
        window_width, window_height = map(int, args.window_size.lower().split("x"))
    except ValueError:
        logging.warning(f"Invalid window size '{args.window_size}', using default 1600x900")
        window_width, window_height = 1600, 900

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
        window_size=(window_width, window_height),
    )

    # Load previous progress
    if args.load_progress and args.load_progress.exists():
        # Use specified path
        annotator.load_annotations(args.load_progress)
    else:
        # Try to auto-load from default path
        default_path = annotator._get_default_annotation_path()
        if default_path.exists():
            logging.info(f"Found existing annotations at {default_path}")
            annotator.load_annotations(default_path)

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
                    push_to_hub=args.push_to_hub,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if annotator.has_unsaved_changes:
            if input("Save progress before exit? [y/N]: ").lower() == "y":
                annotator.save_annotations(args.save_progress)


if __name__ == "__main__":
    main()
