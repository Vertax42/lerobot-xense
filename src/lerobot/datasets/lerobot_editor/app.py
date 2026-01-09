import signal
import sys

import numpy as np
import pyqtgraph as pg
import torch
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lerobot.datasets.lerobot_editor.processor import DatasetProcessor


class LoaderThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, processor: DatasetProcessor, repo_id: str):
        super().__init__()
        self.processor = processor
        self.repo_id = repo_id

    def run(self):
        try:
            dataset = self.processor.load_dataset(self.repo_id)
            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(str(e))


class EpisodeLoaderThread(QThread):
    """Thread for loading episode data asynchronously."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, processor: DatasetProcessor, episode_idx: int, keys: list):
        super().__init__()
        self.processor = processor
        self.episode_idx = episode_idx
        self.keys = keys

    def run(self):
        try:
            data = self.processor.get_episode_data(self.episode_idx, self.keys)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class DatasetGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = DatasetProcessor()
        self.plots: dict[str, pg.PlotWidget] = {}
        self.plot_curves: dict[str, list[pg.PlotDataItem]] = {}
        self.v_lines: dict[str, pg.InfiniteLine] = {}
        self.current_ep_data: dict[str, np.ndarray] = {}
        self.vector_keys = []
        self.trim_start_frame: int | None = None
        self.trim_end_frame: int | None = None

        # Throttle timer for frame updates (prevents lag when holding arrow keys)
        self._pending_frame_idx: int | None = None
        self._frame_update_timer = QTimer()
        self._frame_update_timer.setSingleShot(True)
        self._frame_update_timer.setInterval(30)  # ~33 fps max
        self._frame_update_timer.timeout.connect(self._do_frame_update)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LeRobot Dataset Visualizer")
        self.resize(1400, 950)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Bar (保持不变)
        top_layout = QHBoxLayout()
        self.repo_input = QLineEdit("lerobot/pusht")
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.show_info_btn = QPushButton("Show Info")
        self.show_info_btn.clicked.connect(lambda: self.info_panel.show())

        top_layout.addWidget(QLabel("Repo ID:"))
        top_layout.addWidget(self.repo_input)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.show_info_btn)
        main_layout.addLayout(top_layout)

        # Main Vertical Splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        self.horizontal_splitter = QSplitter(Qt.Horizontal)

        # 1. Left: Episode List
        self.ep_list = QListWidget()
        self.ep_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ep_list.customContextMenuRequested.connect(self.show_episode_context_menu)
        self.ep_list.currentRowChanged.connect(self.on_episode_changed)
        self.horizontal_splitter.addWidget(self.ep_list)

        # 2. Center: Image, Slider and Plots (Always Visible)
        self.center_splitter = QSplitter(Qt.Vertical)

        # Top part: Images and Slider
        image_slider_container = QWidget()
        image_slider_layout = QVBoxLayout(image_slider_container)

        # Container for multiple camera images
        self.images_container = QWidget()
        self.images_layout = QHBoxLayout(self.images_container)
        self.images_layout.setSpacing(5)
        self.image_labels: dict[str, QLabel] = {}  # Will be populated on dataset load
        # Default placeholder
        self.default_image_label = QLabel("No Image")
        self.default_image_label.setAlignment(Qt.AlignCenter)
        self.default_image_label.setStyleSheet("background-color: black;")
        self.images_layout.addWidget(self.default_image_label)
        image_slider_layout.addWidget(self.images_container, stretch=1)

        # Slider and Labels
        slider_container = QWidget()
        slider_vbox = QVBoxLayout(slider_container)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        slider_vbox.addWidget(self.frame_slider)

        labels_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        self.timestamp_label = QLabel("Timestamp: 0.000s")

        # Trim buttons
        self.mark_start_btn = QPushButton("Mark Trim Start")
        self.mark_start_btn.clicked.connect(self.on_mark_start_clicked)

        self.mark_end_btn = QPushButton("Mark Trim End")
        self.mark_end_btn.clicked.connect(self.on_mark_end_clicked)

        labels_layout.addWidget(self.frame_label)
        labels_layout.addWidget(self.mark_start_btn)
        labels_layout.addWidget(self.mark_end_btn)
        labels_layout.addStretch()
        labels_layout.addWidget(self.timestamp_label)
        slider_vbox.addLayout(labels_layout)

        image_slider_layout.addWidget(slider_container)
        self.center_splitter.addWidget(image_slider_container)

        # Bottom part: Plots (Now always visible in center)
        self.plot_scroll = QScrollArea()
        self.plot_scroll.setFocusPolicy(Qt.NoFocus)
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setSpacing(10)
        self.plot_layout.addStretch()
        self.plot_scroll.setWidget(self.plot_container)
        self.plot_scroll.setWidgetResizable(True)
        self.center_splitter.addWidget(self.plot_scroll)

        # Initial proportions for center
        self.center_splitter.setSizes([400, 500])
        self.horizontal_splitter.addWidget(self.center_splitter)

        # 3. Right: Hierarchical Selectors and Edit Tools (Wrapped in Tabs)
        self.right_tabs = QTabWidget()
        self.horizontal_splitter.addWidget(self.right_tabs)

        # Tab 1: Features (Tree only)
        self.feature_tree = QTreeWidget()
        self.feature_tree.setFocusPolicy(Qt.NoFocus)
        self.feature_tree.setHeaderLabel("Features & Dimensions")
        self.feature_tree.itemChanged.connect(self.on_tree_item_changed)
        self.right_tabs.addTab(self.feature_tree, "Features")

        # Tab 2: Edit
        self.edit_tab = QWidget()
        self.init_edit_tab()
        self.right_tabs.addTab(self.edit_tab, "Edit")

        # Layout weights: Left is narrow, Center is wide (Image+Plots), Right is medium (Tree/Edit)
        self.horizontal_splitter.setStretchFactor(0, 0)
        self.horizontal_splitter.setStretchFactor(1, 4)
        self.horizontal_splitter.setStretchFactor(2, 1)
        self.horizontal_splitter.setSizes([120, 900, 380])
        self.main_splitter.addWidget(self.horizontal_splitter)

        # Bottom Info Panel (保持不变)
        self.info_panel = QGroupBox("Dataset Details")
        self.info_panel_layout = QHBoxLayout(self.info_panel)
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_panel_layout.addWidget(self.info_display)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.info_panel.hide)
        self.info_panel_layout.addWidget(close_btn, alignment=Qt.AlignTop)
        self.info_panel.hide()
        self.main_splitter.addWidget(self.info_panel)

        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

    def init_edit_tab(self):
        layout = QVBoxLayout(self.edit_tab)

        # 1. Pending Operations List
        layout.addWidget(QLabel("<b>Pending Operations:</b>"))
        self.pending_op_list = QListWidget()
        self.pending_op_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pending_op_list.customContextMenuRequested.connect(self.show_op_context_menu)
        self.pending_op_list.setToolTip("Right click to undo a specific operation")
        layout.addWidget(self.pending_op_list)

        self.clear_tasks_btn = QPushButton("Clear All Tasks")
        self.clear_tasks_btn.clicked.connect(self.on_clear_tasks_clicked)
        layout.addWidget(self.clear_tasks_btn)

        layout.addSpacing(10)

        # 2. Global Operations
        global_group = QGroupBox("Global Operations")
        global_layout = QVBoxLayout(global_group)

        self.batch_delete_btn = QPushButton("Batch Delete Episodes...")
        self.batch_delete_btn.clicked.connect(self.on_batch_delete_clicked)
        global_layout.addWidget(self.batch_delete_btn)

        self.remove_feature_btn = QPushButton("Remove Features...")
        self.remove_feature_btn.clicked.connect(self.on_remove_feature_clicked)
        global_layout.addWidget(self.remove_feature_btn)

        layout.addWidget(global_group)

        # 3. Local Operations
        local_group = QGroupBox("Local Operations")
        local_layout = QVBoxLayout(local_group)

        self.trim_frames_btn = QPushButton("Trim Selected Range")
        self.trim_frames_btn.setEnabled(False)
        self.trim_frames_btn.setToolTip("Mark start and end on the slider first")
        self.trim_frames_btn.clicked.connect(self.on_trim_frames_clicked)
        local_layout.addWidget(self.trim_frames_btn)

        self.edit_frame_btn = QPushButton("Edit Current Frame Features...")
        self.edit_frame_btn.clicked.connect(self.on_edit_frame_clicked)
        local_layout.addWidget(self.edit_frame_btn)

        layout.addWidget(local_group)

        layout.addStretch()

        # 4. Export Settings (At the bottom)
        export_group = QGroupBox("Export Settings")
        export_layout = QVBoxLayout(export_group)

        export_layout.addWidget(QLabel("New Repo ID:"))
        self.new_repo_input = QLineEdit()
        self.new_repo_input.setPlaceholderText("e.g., lerobot/pusht_modified")
        export_layout.addWidget(self.new_repo_input)

        self.save_dataset_btn = QPushButton("Apply Edits && Save Dataset")
        self.save_dataset_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 10px;"
        )
        self.save_dataset_btn.clicked.connect(self.on_save_dataset_clicked)
        export_layout.addWidget(self.save_dataset_btn)

        layout.addWidget(export_group)

    def on_mark_start_clicked(self):
        self.trim_start_frame = self.frame_slider.value()
        # Find local frame index within episode
        start, _ = self.processor.get_episode_range(self.ep_list.currentRow())
        local_idx = self.trim_start_frame - start
        self.status_label.setText(f"Marked start frame: {local_idx}")
        self.update_trim_btn_state()

    def on_mark_end_clicked(self):
        self.trim_end_frame = self.frame_slider.value()
        start, _ = self.processor.get_episode_range(self.ep_list.currentRow())
        local_idx = self.trim_end_frame - start
        self.status_label.setText(f"Marked end frame: {local_idx}")
        self.update_trim_btn_state()

    def update_trim_btn_state(self):
        # Enable trim button if both marks are set and in the same episode
        # (Technically they could be different episodes if we want cross-episode trimming,
        # but let's keep it simple for now).
        if hasattr(self, "trim_frames_btn"):
            self.trim_frames_btn.setEnabled(
                self.trim_start_frame is not None and self.trim_end_frame is not None
            )

    def on_edit_frame_clicked(self):
        if not self.processor.dataset:
            return

        frame_idx = self.frame_slider.value()
        try:
            data = self.processor.get_frame(frame_idx)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get frame data: {str(e)}")
            return

        # Get episode info
        ep_idx = self.ep_list.currentRow()
        start, _ = self.processor.get_episode_range(ep_idx)
        local_idx = frame_idx - start

        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Frame {local_idx} in Episode {ep_idx}")
        dialog.resize(400, 500)
        d_layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll_content = QWidget()
        form_layout = QFormLayout(scroll_content)

        inputs = {}
        original_data = {}
        # Only show vector/scalar features, skip images/videos
        for key, val in data.items():
            if any(
                x in key
                for x in [
                    "task",
                    "image",
                    "video",
                    "index",
                    "frame_index",
                    "episode_index",
                    "timestamp",
                ]
            ):
                continue

            original_data[key] = val
            # Convert to string for editing
            val_np = val.numpy() if hasattr(val, "numpy") else np.array(val)
            val_str = (
                np.array2string(val_np, separator=",").replace("[", "").replace("]", "").replace("\n", "")
            )

            line_edit = QLineEdit(val_str)
            form_layout.addRow(f"{key}:", line_edit)
            inputs[key] = line_edit

        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        d_layout.addWidget(scroll)

        btns_layout = QHBoxLayout()
        ok_btn = QPushButton("Add to Tasks")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        d_layout.addLayout(btns_layout)

        if dialog.exec() == QDialog.Accepted:
            new_features = {}
            for key, line_edit in inputs.items():
                try:
                    orig_val = original_data[key]
                    txt = line_edit.text().strip()
                    clean_txt = txt.replace("[", "").replace("]", "").replace(" ", "")

                    if not clean_txt:
                        continue

                    parts = [p.strip() for p in clean_txt.split(",") if p.strip()]

                    # Determine target dtype and converter
                    if isinstance(orig_val, torch.Tensor):
                        target_dtype = orig_val.dtype
                        if target_dtype == torch.bool:
                            vals = [p.lower() in ["true", "1", "t", "y", "yes"] for p in parts]
                            new_val = torch.tensor(vals, dtype=torch.bool)
                        elif target_dtype in [
                            torch.int64,
                            torch.int32,
                            torch.int16,
                            torch.int8,
                            torch.uint8,
                        ]:
                            vals = [int(float(p)) for p in parts]
                            new_val = torch.tensor(vals, dtype=target_dtype)
                        else:
                            vals = [float(p) for p in parts]
                            new_val = torch.tensor(vals, dtype=target_dtype)

                        # Handle shape (if original was scalar-like tensor vs vector)
                        if orig_val.ndim == 0 and new_val.numel() == 1:
                            new_val = new_val.squeeze()
                    else:
                        # Fallback for non-tensor types
                        if isinstance(orig_val, bool):
                            new_val = parts[0].lower() in ["true", "1", "t", "y", "yes"]
                        elif isinstance(orig_val, int):
                            new_val = int(float(parts[0]))
                        else:
                            new_val = float(parts[0])

                    new_features[key] = new_val
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to parse {key}: {str(e)}")

            if new_features:
                self.processor.add_frame_edit_task(ep_idx, local_idx, new_features)
                self.refresh_edit_ui()
                self.status_label.setText(f"Added edit task for Episode {ep_idx}, Frame {local_idx}")

    def on_trim_frames_clicked(self):
        if self.trim_start_frame is None or self.trim_end_frame is None:
            return

        # Get episode index and local frame range
        ep_idx = self.ep_list.currentRow()
        ep_start, _ = self.processor.get_episode_range(ep_idx)

        start_local = self.trim_start_frame - ep_start
        end_local = self.trim_end_frame - ep_start

        if start_local > end_local:
            start_local, end_local = end_local, start_local

        self.processor.add_trim_task(ep_idx, start_local, end_local)
        self.refresh_edit_ui()
        self.status_label.setText(f"Added trim task for Episode {ep_idx}: {start_local}-{end_local}")

        # Reset marks
        self.trim_start_frame = None
        self.trim_end_frame = None
        self.update_trim_btn_state()

    def on_clear_tasks_clicked(self):
        self.processor.clear_edit_tasks()
        self.refresh_edit_ui()
        self.status_label.setText("Edit tasks cleared")

    def on_batch_delete_clicked(self):
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self, "Batch Delete Episodes", "Enter episode indices (e.g., 0, 2, 5-10):"
        )
        if ok and text:
            try:
                indices = self._parse_indices(text)
                count = 0
                for idx in indices:
                    if idx < self.processor.dataset.meta.total_episodes:
                        self.processor.add_delete_episode_task(idx)
                        count += 1
                self.refresh_edit_ui()
                self.status_label.setText(f"Added {count} episodes to deletion tasks")
            except ValueError as e:
                QMessageBox.critical(self, "Error", f"Invalid input format: {str(e)}")

    def on_remove_feature_clicked(self):
        if not self.processor.dataset:
            return

        # Create a simple dialog with checkboxes
        from PySide6.QtWidgets import QDialog, QListWidgetItem

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Features to Remove")
        dialog.resize(300, 400)
        d_layout = QVBoxLayout(dialog)

        list_widget = QListWidget()
        features = sorted(list(self.processor.dataset.meta.features.keys()))
        # Filter out internal/essential features that shouldn't be removed
        internal_keys = [
            "index",
            "frame_index",
            "episode_index",
            "timestamp",
            "task_index",
        ]
        features = [f for f in features if not any(k == f or f.endswith(f".{k}") for k in internal_keys)]

        for f in features:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Pre-check if already in tasks
            item.setCheckState(Qt.Checked if f in self.processor.features_to_remove else Qt.Unchecked)
            list_widget.addItem(item)

        d_layout.addWidget(list_widget)

        btns_layout = QHBoxLayout()
        ok_btn = QPushButton("Add to Tasks")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btns_layout.addWidget(ok_btn)
        btns_layout.addWidget(cancel_btn)
        d_layout.addLayout(btns_layout)

        if dialog.exec() == QDialog.Accepted:
            count = 0
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                f_name = item.text()
                if item.checkState() == Qt.Checked:
                    self.processor.add_remove_feature_task(f_name)
                    count += 1
                else:
                    # If it was there but now unchecked, remove from tasks
                    if f_name in self.processor.features_to_remove:
                        self.processor.features_to_remove.remove(f_name)

            self.refresh_edit_ui()
            self.status_label.setText("Updated feature removal tasks")

    def _parse_indices(self, text: str) -> list[int]:
        """Parses strings like '0, 2, 5-10' into a list of integers."""
        indices = set()
        parts = [p.strip() for p in text.split(",")]
        for part in parts:
            if "-" in part:
                start_str, end_str = part.split("-")
                start, end = int(start_str), int(end_str)
                for i in range(start, end + 1):
                    indices.add(i)
            else:
                indices.add(int(part))
        return sorted(list(indices))

    def refresh_edit_ui(self):
        self.pending_op_list.clear()

        # 1. Episode Deletions
        for ep_idx in sorted(self.processor.to_delete_episodes):
            item = QListWidgetItem(f"Delete Episode {ep_idx}")
            item.setData(Qt.UserRole, {"type": "delete_episode", "idx": ep_idx})
            self.pending_op_list.addItem(item)

        # 2. Feature Removal
        for f_name in sorted(list(self.processor.features_to_remove)):
            item = QListWidgetItem(f"Remove Feature: {f_name}")
            item.setData(Qt.UserRole, {"type": "remove_feature", "name": f_name})
            self.pending_op_list.addItem(item)

        # 3. Trim Tasks
        for i, task in enumerate(self.processor.trim_tasks):
            item = QListWidgetItem(
                f"Trim Ep {task['episode_index']}: {task['start_frame']}-{task['end_frame']}"
            )
            item.setData(Qt.UserRole, {"type": "trim", "idx": i})
            self.pending_op_list.addItem(item)

        # 4. Frame Edits
        for i, task in enumerate(self.processor.frame_edit_tasks):
            item = QListWidgetItem(f"Edit Ep {task['episode_index']} Frame {task['frame_index']}")
            item.setData(Qt.UserRole, {"type": "edit_frame", "idx": i})
            self.pending_op_list.addItem(item)

        # Update default new repo id if empty
        if self.processor.dataset and not self.new_repo_input.text():
            self.new_repo_input.setText(f"{self.processor.dataset.repo_id}_modified")

    def show_op_context_menu(self, position):
        item = self.pending_op_list.itemAt(position)
        if not item:
            return

        data = item.data(Qt.UserRole)
        if not data:
            return

        menu = QMenu()
        undo_action = menu.addAction("Undo This Operation")

        if data["type"] == "delete_episode":
            undo_action.triggered.connect(lambda: self.undo_delete_task(data["idx"]))
        elif data["type"] == "remove_feature":
            undo_action.triggered.connect(lambda: self.undo_remove_feature_task(data["name"]))
        elif data["type"] == "trim":
            undo_action.triggered.connect(lambda: self.undo_trim_task(data["idx"]))
        elif data["type"] == "edit_frame":
            undo_action.triggered.connect(lambda: self.undo_edit_frame_task(data["idx"]))

        menu.exec(self.pending_op_list.mapToGlobal(position))

    def undo_trim_task(self, idx):
        if 0 <= idx < len(self.processor.trim_tasks):
            self.processor.trim_tasks.pop(idx)
            self.refresh_edit_ui()
            self.status_label.setText("Undid trim task")

    def undo_edit_frame_task(self, idx):
        if 0 <= idx < len(self.processor.frame_edit_tasks):
            self.processor.frame_edit_tasks.pop(idx)
            self.refresh_edit_ui()
            self.status_label.setText("Undid frame edit task")

    def undo_remove_feature_task(self, f_name):
        if f_name in self.processor.features_to_remove:
            self.processor.features_to_remove.remove(f_name)
            self.refresh_edit_ui()
            self.status_label.setText(f"Undid removal of feature {f_name}")

    def undo_delete_task(self, ep_idx):
        if ep_idx in self.processor.to_delete_episodes:
            self.processor.to_delete_episodes.remove(ep_idx)
            self.refresh_edit_ui()
            self.status_label.setText(f"Undid deletion of Episode {ep_idx}")

    def on_save_dataset_clicked(self):
        new_repo_id = self.new_repo_input.text().strip()
        if not new_repo_id:
            QMessageBox.warning(self, "Warning", "Please enter a New Repo ID")
            return

        # Confirm overwrite if same
        if self.processor.dataset and new_repo_id == self.processor.dataset.repo_id:
            reply = QMessageBox.question(
                self,
                "Confirm Overwrite",
                f"New Repo ID is the same as current. Overwrite {new_repo_id}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self.status_label.setText("Saving dataset... this may take a while.")
        self.save_dataset_btn.setEnabled(False)
        self.repaint()  # Force UI update

        # Run in main thread or simple thread to avoid GUI lock
        # For multi-tasking, we should use a proper thread, but let's fix logic first
        try:
            new_dataset = self.processor.apply_edits(new_repo_id)
            QMessageBox.information(self, "Success", f"Dataset saved to {new_dataset.root}")
            self.status_label.setText(f"Dataset saved to {new_repo_id}")

            # Reset tasks UI
            self.refresh_edit_ui()

            # Switch back to Visualize and load new dataset
            self.right_tabs.setCurrentIndex(0)
            self.repo_input.setText(new_repo_id)
            self.on_load_clicked()
        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save dataset: {str(e)}")
            self.status_label.setText("Save failed")
        finally:
            self.save_dataset_btn.setEnabled(True)

    def on_load_clicked(self):
        repo_id = self.repo_input.text().strip()
        if not repo_id:
            return
        self.load_btn.setEnabled(False)
        self.status_label.setText(f"Loading {repo_id}...")
        self.loader_thread = LoaderThread(self.processor, repo_id)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()

    def show_episode_context_menu(self, position):
        item = self.ep_list.itemAt(position)
        if not item:
            return

        menu = QMenu()
        ep_idx = self.ep_list.row(item)

        mark_action = menu.addAction(f"Mark Episode {ep_idx} for Deletion")
        mark_action.triggered.connect(lambda: self.mark_episode_for_deletion(ep_idx))

        menu.exec(self.ep_list.mapToGlobal(position))

    def mark_episode_for_deletion(self, ep_idx):
        self.processor.add_delete_episode_task(ep_idx)
        self.refresh_edit_ui()
        self.status_label.setText(f"Episode {ep_idx} marked for deletion")
        # Switch to Edit tab to show the change
        self.right_tabs.setCurrentIndex(1)

    def on_load_finished(self, dataset):
        self.load_btn.setEnabled(True)
        self.status_label.setText("Loaded")

        # Reset tasks and UI on new load
        self.processor.clear_edit_tasks()
        self.new_repo_input.clear()
        self.refresh_edit_ui()

        # Table-based Feature Info for Alignment
        meta = dataset.meta
        sample = dataset[0]
        rows = []
        for k in meta.features.keys():
            val = sample[k]
            type_name = type(val).__name__
            if hasattr(val, "dtype"):
                type_name = f"{type_name}[{val.dtype}]"

            shape = "1"
            if hasattr(val, "shape"):
                shape = str(list(val.shape))
            elif hasattr(val, "__len__"):
                shape = str(len(val))

            # Align columns using table
            rows.append(
                f"<tr><td width='150'><code>{k}</code></td><td width='200' style='color: #2980b9;'>{type_name}</td><td>shape: {shape}</td></tr>"
            )

        info_text = (
            f"<b>Repo ID:</b> {dataset.repo_id}<br>"
            f"<b>Total Episodes:</b> {meta.total_episodes} | <b>Total Frames:</b> {meta.total_frames} | <b>FPS:</b> {meta.fps}<br>"
            f"<b>Robot Type:</b> {meta.robot_type}<br>"
            f"<b>Features:</b><table style='margin-left: 20px;'>{''.join(rows)}</table>"
        )
        self.info_display.setHtml(info_text)
        self.info_panel.show()

        self.ep_list.clear()
        for i in range(dataset.meta.total_episodes):
            self.ep_list.addItem(f"Episode {i}")

        self.init_image_views(dataset)
        self.init_dimension_selectors(dataset)
        if dataset.meta.total_episodes > 0:
            self.ep_list.setCurrentRow(0)

    def init_image_views(self, dataset):
        """Initialize image labels for all cameras in the dataset."""
        # Clear existing image labels
        for label in self.image_labels.values():
            label.setParent(None)
        self.image_labels.clear()

        # Hide default placeholder
        self.default_image_label.hide()

        # Get all image/video keys
        all_img_keys = list(dataset.meta.image_keys) + list(dataset.meta.video_keys)

        if not all_img_keys:
            self.default_image_label.show()
            return

        # Create a label for each camera
        for img_key in sorted(all_img_keys):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(2)

            # Camera name label
            name_label = QLabel(img_key.split(".")[-1])  # Show short name
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color: white; background-color: #333;")
            container_layout.addWidget(name_label)

            # Image label
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background-color: black;")
            img_label.setMinimumSize(160, 120)
            container_layout.addWidget(img_label, stretch=1)

            self.images_layout.addWidget(container)
            self.image_labels[img_key] = img_label

    def init_dimension_selectors(self, dataset):
        """Initializes the hierarchical feature tree."""
        self.feature_tree.clear()
        # Clean up old plots
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.plots.clear()
        self.plot_curves.clear()

        # Dynamically get all plottable keys (exclude images, videos, and metadata)
        meta_keys = {"index", "frame_index", "episode_index", "timestamp", "task_index"}
        image_video_keys = set(dataset.meta.image_keys) | set(dataset.meta.video_keys)

        # Get all feature keys and filter
        all_keys = list(dataset.meta.features.keys())
        self.vector_keys = [k for k in all_keys if k not in meta_keys and k not in image_video_keys]
        # Sort for consistent display: prioritize common keys, then alphabetical
        priority_keys = [
            "observation.state",
            "action",
            "next.reward",
            "next.done",
            "next.success",
        ]

        def sort_key(k):
            if k in priority_keys:
                return (0, priority_keys.index(k))
            return (1, k)

        self.vector_keys.sort(key=sort_key)

        sample = dataset[0]
        for k in self.vector_keys:
            val = sample[k]
            # Convert to numpy to get shape
            v_np = val.numpy() if hasattr(val, "numpy") else np.array(val)
            dims = v_np.shape[0] if v_np.ndim > 0 else 1

            # Get dimension names from metadata if available
            feature_info = dataset.meta.features.get(k, {})
            dim_names = feature_info.get("names", None)
            # Handle nested list format: [["name1"], ["name2"], ...]
            if dim_names and isinstance(dim_names, list):
                if len(dim_names) > 0 and isinstance(dim_names[0], list):
                    dim_names = [n[0] if n else f"dim_{i}" for i, n in enumerate(dim_names)]

            # Create Tree Item
            parent = QTreeWidgetItem(self.feature_tree)
            parent.setText(0, k)
            parent.setCheckState(0, Qt.Checked)
            parent.setExpanded(True)

            # Create Plot Widget
            pw = pg.PlotWidget(title=k)
            pw.setBackground("w")
            pw.showGrid(x=True, y=True)
            pw.setMinimumHeight(150)
            pw.setFocusPolicy(Qt.NoFocus)

            v_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("k", width=2))
            pw.addItem(v_line)
            self.plots[k] = pw
            self.v_lines[k] = v_line
            self.plot_layout.addWidget(pw)

            self.plot_curves[k] = []
            if dims > 1:
                for d in range(dims):
                    # Use actual dimension name if available
                    if dim_names and d < len(dim_names):
                        dim_label = dim_names[d]
                    else:
                        dim_label = f"Dimension {d}"
                    child = QTreeWidgetItem(parent)
                    child.setText(0, dim_label)
                    child.setCheckState(0, Qt.Checked)
                    child.setData(0, Qt.UserRole, (k, d))

    def on_tree_item_changed(self, item, column):
        """Handles visibility toggles from the tree."""
        key_data = item.data(0, Qt.UserRole)
        is_checked = item.checkState(0) == Qt.Checked

        if key_data is None:  # Parent item (Feature)
            key = item.text(0)
            if key in self.plots:
                self.plots[key].setVisible(is_checked)
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, item.checkState(0))
        else:  # Child item (Dimension)
            key, dim = key_data
            if key in self.plot_curves and dim < len(self.plot_curves[key]):
                self.plot_curves[key][dim].setVisible(is_checked)

    def on_slider_changed(self, value):
        start = self.frame_slider.minimum()
        relative_idx = value - start

        # Update frame label (fast, no data fetch needed)
        self.frame_label.setText(f"Frame: {relative_idx}/{self.frame_slider.maximum() - start}")

        # Estimate timestamp from FPS (fast, avoid fetching frame data)
        if self.processor.dataset:
            fps = self.processor.dataset.meta.fps
            self.timestamp_label.setText(f"Timestamp: {relative_idx / fps:.3f}s")

        # Update Vertical Lines (fast)
        for v_line in self.v_lines.values():
            v_line.setPos(relative_idx)

        # Throttled frame view update (prevents lag when holding keys)
        self._pending_frame_idx = value
        if not self._frame_update_timer.isActive():
            self._frame_update_timer.start()

    def _do_frame_update(self):
        """Perform the actual frame update (called by throttle timer)."""
        if self._pending_frame_idx is not None:
            self.update_frame_view(self._pending_frame_idx)
            self._pending_frame_idx = None

    def on_episode_changed(self, row):
        if row < 0:
            return

        # 显示加载状态
        self.status_label.setText(f"Loading Episode {row}...")
        self.frame_slider.setEnabled(False)

        start, end = self.processor.get_episode_range(row)
        self.frame_slider.setRange(start, end - 1)

        # 异步加载 episode 数据
        self.episode_loader = EpisodeLoaderThread(self.processor, row, self.vector_keys)
        self.episode_loader.finished.connect(self.on_episode_data_loaded)
        self.episode_loader.error.connect(self.on_episode_load_error)
        self.episode_loader.start()

        # 先设置第一帧，不等待全部数据加载
        self.frame_slider.setValue(start)

    def on_episode_data_loaded(self, data):
        """Callback when episode data is loaded."""
        self.current_ep_data = data
        start, end = self.frame_slider.minimum(), self.frame_slider.maximum()
        num_frames = end - start + 1

        # Update Plot Curves
        for k in self.vector_keys:
            if k not in data:
                continue
            raw_val = data[k]
            # 3. 核心修复：数据转换 (T, D) 或 (T,)
            # 处理布尔值和标量
            if raw_val.dtype == bool:
                plot_data = raw_val.astype(np.float32)
            else:
                plot_data = raw_val.astype(np.float32)

            # 确保是 2D 数组 (T, D)
            if plot_data.ndim == 1:
                plot_data = plot_data.reshape(-1, 1)
            pw = self.plots[k]
            # Clear old curves
            for c in self.plot_curves.get(k, []):
                pw.removeItem(c)
            self.plot_curves[k] = []

            # Create new curves for each dimension
            for d in range(plot_data.shape[1]):
                curve = pg.PlotDataItem(plot_data[:, d], pen=pg.mkPen(color=pg.intColor(d), width=1.5))
                pw.addItem(curve)
                self.plot_curves[k].append(curve)

            # 统一设置 X 轴范围，确保对齐
            pw.setXRange(0, num_frames, padding=0)

            # 如果是 reward/done/success，设置合理的 Y 轴范围
            if any(x in k for x in ["reward", "done", "success"]):
                pw.setYRange(-0.1, 1.1, padding=0)

        self.update_plots_visibility()
        self.frame_slider.setEnabled(True)
        self.status_label.setText("Ready")

    def on_episode_load_error(self, err):
        """Callback when episode data loading fails."""
        self.status_label.setText(f"Error loading episode: {err}")
        self.frame_slider.setEnabled(True)

    def update_plots_visibility(self):
        """Updates which curves are shown based on checkboxes in the feature tree."""
        for i in range(self.feature_tree.topLevelItemCount()):
            parent = self.feature_tree.topLevelItem(i)
            key = parent.text(0)
            is_parent_checked = parent.checkState(0) == Qt.Checked

            if key in self.plots:
                self.plots[key].setVisible(is_parent_checked)

            for j in range(parent.childCount()):
                child = parent.child(j)
                key_data = child.data(0, Qt.UserRole)
                if key_data:
                    k, dim = key_data
                    is_child_checked = child.checkState(0) == Qt.Checked
                    if k in self.plot_curves and dim < len(self.plot_curves[k]):
                        self.plot_curves[k][dim].setVisible(is_child_checked)

    def update_frame_view(self, frame_idx):
        try:
            data = self.processor.get_frame(frame_idx)

            # Update all camera images
            for img_key, img_label in self.image_labels.items():
                if img_key not in data:
                    continue

                img_data = data[img_key]
                img_np = self._convert_image_to_numpy(img_data)

                if img_np is None:
                    continue

                h, w, c = img_np.shape
                bytes_per_line = c * w
                qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg)

                # Scale to fit label while keeping aspect ratio
                scaled_pixmap = pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback

            traceback.print_exc()

    def _convert_image_to_numpy(self, img_data):
        """Convert various image formats to numpy array (H, W, C) uint8 RGB."""
        try:
            # Handle different image formats
            if hasattr(img_data, "numpy"):  # Torch Tensor
                img_np = img_data.numpy()
                # Check if CHW format (channels first)
                if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
                # Normalize to 0-255 uint8
                if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
            elif hasattr(img_data, "convert"):  # PIL Image
                img_np = np.array(img_data.convert("RGB"))
            elif isinstance(img_data, dict) and "path" in img_data:
                # Video format - not supported yet
                return None
            else:
                img_np = np.array(img_data)

            # Ensure C-contiguous array
            img_np = np.ascontiguousarray(img_np)

            # Handle grayscale
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)

            return img_np
        except Exception:
            return None

    def on_load_error(self, err):
        self.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", err)

    def keyPressEvent(self, event):
        # 强制让 slider 或 list 处理，或者直接由 window 处理
        if event.key() in [Qt.Key_W, Qt.Key_Up]:
            self.ep_list.setCurrentRow(max(0, self.ep_list.currentRow() - 1))
        elif event.key() in [Qt.Key_S, Qt.Key_Down]:
            self.ep_list.setCurrentRow(min(self.ep_list.count() - 1, self.ep_list.currentRow() + 1))
        elif event.key() in [Qt.Key_A, Qt.Key_Left]:
            self.frame_slider.setValue(max(self.frame_slider.minimum(), self.frame_slider.value() - 1))
        elif event.key() in [Qt.Key_D, Qt.Key_Right]:
            self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    # Timer to allow Python to process signals (Qt blocks Python's event loop)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    gui = DatasetGui()
    gui.show()
    sys.exit(app.exec())
