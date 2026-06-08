# MIT License
#
# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
# Copyright (c) 2026 GRASP Lab, University of Pennsylvania (Added Scaramuzza OCamCalib Fisheye Support)
#
# Acknowledgement: Portions of the modifications in this file, including 
# the adaptation for dual-fisheye datasets and unstitched frame processing, 
# were developed with the assistance of AI (Google Gemini and ChatGPT).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
from functools import partial

import cv2
import matplotlib.cm as cm
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from . import calibration
from .common import AppConstants, Colors, UIStyles
from .lidar_cleaner import LiDARCleaner


class PointCloudItem(QGraphicsItem):
    def __init__(self, points, colors, point_size):
        super().__init__()
        self.points = points
        self.point_size = point_size
        self.qcolors = [QColor.fromRgbF(r, g, b, a) for r, g, b, a in colors]

    def boundingRect(self):
        if self.points.shape[0] == 0:
            return QRectF()
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        pad = self.point_size / 2
        return QRectF(
            min_coords[0] - pad,
            min_coords[1] - pad,
            max_coords[0] - min_coords[0] + self.point_size,
            max_coords[1] - min_coords[1] + self.point_size,
        )

    def paint(self, painter: QPainter, option, widget=None):
        painter.setPen(Qt.NoPen)
        radius = self.point_size / 2.0
        for i in range(self.points.shape[0]):
            painter.setBrush(self.qcolors[i])
            x, y = self.points[i, 0], self.points[i, 1]
            painter.drawEllipse(QRectF(x - radius, y - radius, self.point_size, self.point_size))


class ZoomableView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)


class CalibrationWidget(QWidget):
    calibration_completed = Signal(object)

    def __init__(
        self,
        image_msg,
        pointcloud_msg,
        camerainfo_msg,
        ros_utils,
        initial_transform,
        second_pointcloud_msg=None,
        ocam_json_path=None,
        image_crop="None",
    ):
        super().__init__()
        self.image_msg = image_msg
        self.pointcloud_msg = pointcloud_msg
        self.second_pointcloud_msg = second_pointcloud_msg
        self.camerainfo_msg = camerainfo_msg
        self.ros_utils = ros_utils
        self.image_crop = image_crop
        self.correspondences = {}
        self.lidar_to_lidar_correspondences = {}

        self.ocam_json_path = ocam_json_path
        self.ocam_model = self._load_ocam_model()

        self.initial_extrinsics = np.linalg.inv(initial_transform)
        self.extrinsics = np.copy(self.initial_extrinsics)
        self.second_lidar_transform = np.eye(4)
        self.occlusion_mask = None
        self.second_occlusion_mask = None

        self.euler_convention = "xyz"
        self.euler_convention_options = ["xyz", "zxy", "zyz"]

        self.original_cv_image = None
        self.is_rectification_enabled = False

        self.selection_mode = None
        self.selected_2d_point = None
        self.temp_2d_marker = []
        self.current_3d_selection = []
        self.highlighted_3d_items = []
        self.selected_3d_items_map = {}

        self.point_cloud_item = None
        self.kdtree = None

        self.second_point_cloud_item = None
        self.second_kdtree = None
        self.has_second_pointcloud = second_pointcloud_msg is not None

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        left_layout = QVBoxLayout()
        self.scene = QGraphicsScene()
        self.view = ZoomableView(self.scene)
        self.view.viewport().installEventFilter(self)
        left_layout.addWidget(self.view)

        right_controls_layout = self._setup_controls()

        main_layout.addLayout(left_layout, 4)
        main_layout.addLayout(right_controls_layout, 1)

        self.display_image()
        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self._update_inputs_from_extrinsics()
        self._update_calibrate_button_highlight()
        self.display_camera_intrinsics()

    def _load_ocam_model(self):
        if not self.ocam_json_path or not os.path.exists(self.ocam_json_path):
            return None
        try:
            with open(self.ocam_json_path, 'r') as f:
                data = json.load(f)
            if hasattr(self.ros_utils, 'OCamModel'):
                print(f"[INFO] Loaded OCamCalib model from {self.ocam_json_path}")
                if "taylor_coefficient" in data:
                    pol = np.array(data["taylor_coefficient"])
                    invpol = np.array(data.get("inverse_poly", []))
                    center = tuple(data["distortion_center"])
                    sm = data.get("stretch_matrix", [[1.0, 0.0], [0.0, 1.0]])
                    affine = (sm[0][0], sm[0][1], sm[1][0])
                    width = data.get('width', int(center[0] * 2))
                    height = data.get('height', int(center[1] * 2))
                else:
                    pol = np.array(data.get('pol', []))
                    invpol = np.array(data.get('invpol', []))
                    center = (data.get('xc', 0.0), data.get('yc', 0.0))
                    affine = (data.get('c', 1.0), data.get('d', 0.0), data.get('e', 0.0))
                    width = data.get('width', getattr(self.camerainfo_msg, 'width', 0))
                    height = data.get('height', getattr(self.camerainfo_msg, 'height', 0))

                return self.ros_utils.OCamModel(
                    pol=pol, invpol=invpol, center=center, affine=affine, width=width, height=height
                )
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed to load OCamCalib JSON: {e}")
            return None

    def has_significant_distortion(self):
        if self.ocam_model is not None:
            return False
        if not hasattr(self.camerainfo_msg, "d"):
            return False
        dist_coeffs = np.array(self.camerainfo_msg.d)
        if dist_coeffs.size == 0:
            return False
        return bool(np.any(np.abs(dist_coeffs) > 1e-6))

    def toggle_rectification(self, enabled):
        self.is_rectification_enabled = enabled
        self.display_image()

    def rectify_image(self, image):
        if not self.has_significant_distortion() or self.ocam_model is not None:
            return image
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        dist_coeffs = np.array(self.camerainfo_msg.d)
        try:
            return cv2.undistort(image, K, dist_coeffs, None, K)
        except Exception as e:
            return image

    def _setup_controls(self):
        right_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()

        view_group = QGroupBox("View Settings")
        view_controls_layout = QFormLayout(view_group)
        self.image_res_label = QLabel("N/A")
        view_controls_layout.addRow("Image Resolution:", self.image_res_label)
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 10)
        self.point_size_spinbox.setValue(AppConstants.DEFAULT_POINT_SIZE)
        view_controls_layout.addRow("Point Size:", self.point_size_spinbox)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["autumn", "jet", "winter", "summer", "spring", "hot", "magma", "inferno", "Spectral", "RdYlGn"])
        self.colormap_combo.setCurrentText(AppConstants.DEFAULT_COLORMAP)
        view_controls_layout.addRow("Colormap:", self.colormap_combo)
        self.colorization_mode_combo = QComboBox()
        self.colorization_mode_combo.addItems(["Intensity", "Distance"])
        self.colorization_mode_combo.setCurrentText("Intensity")
        view_controls_layout.addRow("Color Mode:", self.colorization_mode_combo)
        self.min_value_spinbox = QDoubleSpinBox()
        self.min_value_spinbox.setRange(-1e9, 1e9)
        self.min_value_spinbox.setDecimals(2)
        view_controls_layout.addRow("Min Value:", self.min_value_spinbox)
        self.max_value_spinbox = QDoubleSpinBox()
        self.max_value_spinbox.setRange(-1e9, 1e9)
        self.max_value_spinbox.setDecimals(2)
        view_controls_layout.addRow("Max Value:", self.max_value_spinbox)
        self.fov_limit_spinbox = QDoubleSpinBox()
        self.fov_limit_spinbox.setRange(45.0, 220.0)
        self.fov_limit_spinbox.setValue(105.0)
        self.fov_limit_spinbox.setSuffix(" °")
        self.fov_limit_spinbox.setDecimals(1)
        self.fov_limit_spinbox.setToolTip(
            "Half-angle FOV cutoff for fisheye (OCamCalib) projection.\n"
            "Points whose ray angle from the optical axis exceeds this threshold are discarded.\n"
            "Typical value: 100–110° for a 200° FOV fisheye."
        )
        view_controls_layout.addRow("Fisheye FOV Limit:", self.fov_limit_spinbox)

        self.rectify_checkbox = QCheckBox("Rectify Image")
        has_distortion = self.has_significant_distortion()
        self.rectify_checkbox.setEnabled(has_distortion)
        if has_distortion:
            self.is_rectification_enabled = True
            self.rectify_checkbox.setChecked(True)
        self.rectify_checkbox.toggled.connect(self.toggle_rectification)
        view_controls_layout.addRow(self.rectify_checkbox)

        self.apply_view_button = QPushButton("Apply View Changes")
        self.apply_view_button.clicked.connect(self.redraw_points)
        view_controls_layout.addRow(self.apply_view_button)

        self.clean_occlusion_button = QPushButton("Clean Occluded Points")
        self.clean_occlusion_button.clicked.connect(self.run_occlusion_cleaning)
        view_controls_layout.addRow(self.clean_occlusion_button)

        col1_layout.addWidget(view_group)
        col1_layout.addSpacing(20)

        corr_group = QGroupBox("Correspondence Management")
        corr_layout = QVBoxLayout(corr_group)
        if self.has_second_pointcloud:
            self.correspondence_mode_combo = QComboBox()
            self.correspondence_mode_combo.addItems(["Master LiDAR ↔ Camera", "Second LiDAR ↔ Master LiDAR"])
            corr_layout.addWidget(QLabel("Correspondence Mode:"))
            corr_layout.addWidget(self.correspondence_mode_combo)
        self.add_corr_button = QPushButton("Add Correspondence")
        self.add_corr_button.setCheckable(True)
        self.add_corr_button.toggled.connect(self.toggle_selection_mode)
        corr_layout.addWidget(self.add_corr_button)
        self.confirm_3d_button = QPushButton("Confirm 3D Selection")
        self.confirm_3d_button.setVisible(False)
        self.confirm_3d_button.clicked.connect(self.finalize_correspondence)
        corr_layout.addWidget(self.confirm_3d_button)
        self.corr_list_widget = QListWidget()
        self.corr_list_widget.currentItemChanged.connect(self.highlight_from_list)
        corr_layout.addWidget(self.corr_list_widget)
        self.delete_corr_button = QPushButton("Delete Selected")
        self.delete_corr_button.clicked.connect(self.delete_correspondence)
        corr_layout.addWidget(self.delete_corr_button)
        col1_layout.addWidget(corr_group)
        col1_layout.addSpacing(20)

        calib_group = QGroupBox("Calibration Settings")
        calib_controls_layout = QFormLayout(calib_group)
        self.pnp_solver_combo = QComboBox()
        self.pnp_solver_combo.addItems(["Iterative", "SQPnP", "None"])
        calib_controls_layout.addRow("RANSAC:", self.pnp_solver_combo)
        self.lsq_method_combo = QComboBox()
        self.lsq_method_combo.addItems(["lm", "trf", "dogbox"])
        calib_controls_layout.addRow("LSQ Method:", self.lsq_method_combo)
        col1_layout.addWidget(calib_group)

        exec_group = QGroupBox("Calibration Execution")
        exec_layout = QVBoxLayout(exec_group)
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.run_calibration)
        exec_layout.addWidget(self.calibrate_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        exec_layout.addWidget(self.progress_bar)
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_calibration_state)
        exec_layout.addWidget(self.reset_button)
        self.results_label = QLabel("Results:")
        self.results_label.setWordWrap(True)
        exec_layout.addWidget(self.results_label)
        col1_layout.addWidget(exec_group)
        col1_layout.addStretch()

        tuning_group = QGroupBox("Manual Fine-Tuning")
        col2_layout = QVBoxLayout()
        tuning_layout = QGridLayout(tuning_group)
        self.t_step_spinbox = QDoubleSpinBox()
        self.t_step_spinbox.setRange(0.01, 5.0)
        self.t_step_spinbox.setValue(AppConstants.DEFAULT_TRANSLATION_STEP)
        self.t_step_spinbox.setSingleStep(0.1)
        self.t_step_spinbox.setSuffix(" cm")
        self.t_step_spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.t_step_spinbox.valueChanged.connect(self._on_step_size_changed)
        self.euler_convention_combo = QComboBox()
        self.euler_convention_combo.addItems(self.euler_convention_options)
        self.euler_convention_combo.setCurrentText(self.euler_convention)
        self.euler_convention_combo.currentTextChanged.connect(self.on_euler_convention_changed)
        tuning_layout.addWidget(QLabel("Pos Step:"), 0, 0, 1, 2)
        tuning_layout.addWidget(self.t_step_spinbox, 0, 2, 1, 1)
        self.r_step_spinbox = QDoubleSpinBox()
        self.r_step_spinbox.setRange(0.01, 10.0)
        self.r_step_spinbox.setValue(AppConstants.DEFAULT_ROTATION_STEP)
        self.r_step_spinbox.setSingleStep(0.05)
        self.r_step_spinbox.setSuffix(" °")
        self.r_step_spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.r_step_spinbox.valueChanged.connect(self._on_step_size_changed)
        tuning_layout.addWidget(QLabel("Rot Step:"), 1, 0, 1, 2)
        tuning_layout.addWidget(self.r_step_spinbox, 1, 2, 1, 1)
        self.step_size_ok_button = QPushButton("OK")
        self.step_size_ok_button.clicked.connect(self._on_step_size_confirmed)
        tuning_layout.addWidget(self.step_size_ok_button, 0, 3, 2, 1)
        self.dof_widgets = {}
        for i, label in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1e6, 1e6)
            spinbox.setDecimals(4)
            spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
            spinbox.valueChanged.connect(self._update_extrinsics_from_inputs)
            minus_button = QPushButton("-")
            plus_button = QPushButton("+")
            minus_button.clicked.connect(partial(self._adjust_dof, label, -1))
            plus_button.clicked.connect(partial(self._adjust_dof, label, 1))
            tuning_layout.addWidget(QLabel(label.capitalize() + ":"), i + 2, 0)
            tuning_layout.addWidget(spinbox, i + 2, 1, 1, 2)
            tuning_layout.addWidget(minus_button, i + 2, 3)
            tuning_layout.addWidget(plus_button, i + 2, 4)
            self.dof_widgets[label] = spinbox

        euler_row = 8
        tuning_layout.addWidget(QLabel("Euler Angle Convention:"), euler_row, 0, 1, 2)
        tuning_layout.addWidget(self.euler_convention_combo, euler_row, 2, 1, 3)
        col2_layout.addWidget(tuning_group)
        col2_layout.addSpacing(20)

        intrinsics_group = QGroupBox("Camera Intrinsics")
        intrinsics_layout = QVBoxLayout(intrinsics_group)
        self.intrinsics_display = QTextEdit()
        self.intrinsics_display.setMinimumHeight(400)
        self.intrinsics_display.setMaximumHeight(600)
        self.intrinsics_display.setFont("monospace")
        self.intrinsics_display.setFontPointSize(10)
        self.intrinsics_display.setReadOnly(True)
        intrinsics_layout.addWidget(self.intrinsics_display)
        col2_layout.addWidget(intrinsics_group)
        col2_layout.addSpacing(20)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.export_button = QPushButton("Export Calibration")
        self.export_button.clicked.connect(self.view_calibration_results)
        export_layout.addWidget(self.export_button)
        col2_layout.addWidget(export_group)
        col2_layout.addStretch()

        right_layout.addLayout(col1_layout, 100)
        right_layout.addLayout(col2_layout, 40)

        self.default_button_style = UIStyles.DEFAULT_BUTTON
        self.point_size_spinbox.valueChanged.connect(self._on_view_params_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_view_params_changed)
        self.colorization_mode_combo.currentTextChanged.connect(self._on_colorization_mode_changed)
        self.min_value_spinbox.valueChanged.connect(self._on_view_params_changed)
        self.max_value_spinbox.valueChanged.connect(self._on_view_params_changed)
        self.fov_limit_spinbox.valueChanged.connect(self._on_view_params_changed)
        # Only show the FOV limit control when an OCamCalib model is loaded
        fov_row_label = view_controls_layout.labelForField(self.fov_limit_spinbox)
        self.fov_limit_spinbox.setVisible(self.ocam_model is not None)
        if fov_row_label:
            fov_row_label.setVisible(self.ocam_model is not None)
        return right_layout

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.confirm_3d_button.isVisible() and self.confirm_3d_button.isEnabled():
                self.finalize_correspondence()
                event.accept()
                return
        elif event.key() == Qt.Key_Escape:
            if self.selection_mode is not None:
                self.reset_selection_mode()
                event.accept()
                return
        elif event.key() == Qt.Key_Backspace:
            current_item = self.corr_list_widget.currentItem()
            if current_item:
                self.delete_correspondence()
                event.accept()
                return
        super().keyPressEvent(event)

    def run_occlusion_cleaning(self):
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        h, w = self.camerainfo_msg.height, self.camerainfo_msg.width
        extrinsics_3x4 = self.extrinsics[:3, :]

        cleaner = LiDARCleaner(K, extrinsics_3x4, self.points_xyz.T, h, w)
        self.occlusion_mask = cleaner.run()
        self.progress_bar.setVisible(False)
        self.redraw_points()

    def _on_view_params_changed(self):
        self.apply_view_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)

    def _on_colorization_mode_changed(self):
        self._update_min_max_values_for_mode()
        self._on_view_params_changed()

    def _update_min_max_values_for_mode(self):
        if not hasattr(self, "points_xyz") or self.points_xyz.shape[0] == 0:
            return
        if self.colorization_mode_combo.currentText() == "Distance":
            if hasattr(self, "valid_indices") and len(self.valid_indices) > 0:
                tvec = self.extrinsics[:3, 3]
                points_cam = (self.extrinsics[:3, :3] @ self.points_xyz.T).T + tvec
                valid_points_cam = points_cam[self.valid_indices]
                distances = np.linalg.norm(valid_points_cam, axis=1)
                min_dist, max_dist = np.quantile(distances, [0.01, 0.99])
                self.min_value_spinbox.setValue(min_dist)
                self.max_value_spinbox.setValue(max_dist)
        else:
            if hasattr(self, "intensities") and self.intensities.size > 0:
                min_i, max_i = np.quantile(self.intensities, [0.01, 0.90])
                self.min_value_spinbox.setValue(min_i)
                self.max_value_spinbox.setValue(max_i)

    def _on_step_size_changed(self):
        self.step_size_ok_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)

    def _on_step_size_confirmed(self):
        self.step_size_ok_button.setStyleSheet(self.default_button_style)

    def on_euler_convention_changed(self, convention: str):
        self.euler_convention = convention
        self._update_inputs_from_extrinsics()

    def _update_calibrate_button_highlight(self):
        master_cam_corr_count = len(self.correspondences)
        lidar_lidar_corr_count = len(self.lidar_to_lidar_correspondences)

        if self.has_second_pointcloud:
            if master_cam_corr_count >= 4 and lidar_lidar_corr_count >= 3:
                self.calibrate_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            else:
                self.calibrate_button.setStyleSheet(self.default_button_style)
        else:
            if master_cam_corr_count >= 4:
                self.calibrate_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            else:
                self.calibrate_button.setStyleSheet(self.default_button_style)

    def _highlight_export_button(self):
        self.export_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)

    def _update_confirm_button_state(self):
        if len(self.current_3d_selection) > 0:
            self.confirm_3d_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            self.confirm_3d_button.setText("Confirm 3D Selection (Enter)")
        else:
            self.confirm_3d_button.setStyleSheet(self.default_button_style)
            self.confirm_3d_button.setText("Confirm 3D Selection")

    def _adjust_dof(self, dof, direction):
        spinbox = self.dof_widgets[dof]
        step = self.t_step_spinbox.value() / 100.0 if dof in "xyz" else self.r_step_spinbox.value()
        spinbox.setValue(spinbox.value() + direction * step)

    def _update_extrinsics_from_inputs(self):
        x, y, z = self.dof_widgets["x"].value(), self.dof_widgets["y"].value(), self.dof_widgets["z"].value()
        roll, pitch, yaw = self.dof_widgets["roll"].value(), self.dof_widgets["pitch"].value(), self.dof_widgets["yaw"].value()
        self.extrinsics = np.identity(4)
        self.extrinsics[:3, 3] = [x, y, z]
        self.extrinsics[:3, :3] = Rotation.from_euler(self.euler_convention, [roll, pitch, yaw], degrees=True).as_matrix()
        self.redraw_points()
        self.update_results_display()
        self._highlight_export_button()
        self.results_label.setText("Results updated via manual adjustment")

    def _update_inputs_from_extrinsics(self):
        tvec = self.extrinsics[:3, 3]
        rpy = Rotation.from_matrix(self.extrinsics[:3, :3]).as_euler(self.euler_convention, degrees=True)
        self.dof_widgets["x"].setValue(tvec[0])
        self.dof_widgets["y"].setValue(tvec[1])
        self.dof_widgets["z"].setValue(tvec[2])
        self.dof_widgets["roll"].setValue(rpy[0])
        self.dof_widgets["pitch"].setValue(rpy[1])
        self.dof_widgets["yaw"].setValue(rpy[2])

    def toggle_selection_mode(self, checked):
        if checked:
            if self.has_second_pointcloud:
                corr_mode = self.correspondence_mode_combo.currentText()
                if corr_mode == "Master LiDAR ↔ Camera":
                    self.selection_mode = "wait_for_2d_click"
                    self.add_corr_button.setText("1. Click on 2D Image Point")
                else:
                    self.selection_mode = "wait_for_second_lidar_click"
                    self.add_corr_button.setText("1. Click on Second LiDAR Point")
            else:
                self.selection_mode = "wait_for_2d_click"
                self.add_corr_button.setText("1. Click on 2D Image Point")
            self.view.setDragMode(QGraphicsView.NoDrag)
        else:
            self.reset_selection_mode()

    def eventFilter(self, source, event):
        if source is self.view.viewport() and event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self.selection_mode == "wait_for_2d_click":
                self.handle_2d_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_3d_clicks":
                self.handle_3d_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_second_lidar_click":
                self.handle_second_lidar_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_master_lidar_clicks":
                self.handle_master_lidar_point_selection(event.pos())
                return True
        return super().eventFilter(source, event)

    def handle_2d_point_selection(self, pos):
        self.clear_temp_markers()
        scene_pos = self.view.mapToScene(pos)
        self.selected_2d_point = (scene_pos.x(), scene_pos.y())
        self.draw_cross_marker(scene_pos, QColor(Colors.CORRESPONDENCE_2D))
        self.selection_mode = "wait_for_3d_clicks"
        self.add_corr_button.setText("2. Click on LiDAR Point(s)")
        self.confirm_3d_button.setVisible(True)

    def handle_3d_point_selection(self, pos):
        if self.kdtree is None:
            return
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        if idx in self.selected_3d_items_map:
            item_to_remove = self.selected_3d_items_map.pop(idx)
            if item_to_remove in self.scene.items():
                self.scene.removeItem(item_to_remove)
            self.current_3d_selection.remove(item_to_remove)
        else:
            point_2d = self.points_proj_valid[idx]
            point_size = self.point_size_spinbox.value()
            item = QGraphicsEllipseItem(point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size)
            item.setPen(QPen(QColor(Colors.CORRESPONDENCE_3D), 2))
            item.setBrush(QBrush(QColor(Colors.CORRESPONDENCE_3D)))
            item.setData(0, idx)
            self.scene.addItem(item)
            self.current_3d_selection.append(item)
            self.selected_3d_items_map[idx] = item
        self._update_confirm_button_state()

    def handle_second_lidar_point_selection(self, pos):
        if self.second_kdtree is None:
            return
        self.clear_temp_markers()
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.second_kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        self.selected_second_lidar_point = self.second_points_proj_valid[idx]
        self.selected_second_lidar_3d_idx = self.second_valid_indices[idx]
        point_2d = self.second_points_proj_valid[idx]
        self.draw_cross_marker(QPointF(point_2d[0], point_2d[1]), QColor(255, 0, 255)) 
        self.selection_mode = "wait_for_master_lidar_clicks"
        self.add_corr_button.setText("2. Click on Master LiDAR Point(s)")
        self.confirm_3d_button.setVisible(True)

    def handle_master_lidar_point_selection(self, pos):
        if self.kdtree is None:
            return
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        if idx in self.selected_3d_items_map:
            item_to_remove = self.selected_3d_items_map.pop(idx)
            if item_to_remove in self.scene.items():
                self.scene.removeItem(item_to_remove)
            self.current_3d_selection.remove(item_to_remove)
        else:
            point_2d = self.points_proj_valid[idx]
            point_size = self.point_size_spinbox.value()
            item = QGraphicsEllipseItem(point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size)
            item.setPen(QPen(QColor(0, 255, 0), 2))
            item.setBrush(QBrush(QColor(0, 255, 0)))
            item.setData(0, idx)
            self.scene.addItem(item)
            self.current_3d_selection.append(item)
            self.selected_3d_items_map[idx] = item
        self._update_confirm_button_state()

    def finalize_correspondence(self):
        if not self.current_3d_selection:
            self.reset_selection_mode()
            return

        if hasattr(self, "selected_second_lidar_point"):
            selected_valid_indices = [item.data(0) for item in self.current_3d_selection]
            original_indices = [self.valid_indices[i] for i in selected_valid_indices]
            mean_3d_point = np.mean(self.points_xyz[original_indices], axis=0)
            second_lidar_3d = self.second_points_xyz[self.selected_second_lidar_3d_idx]
            self.lidar_to_lidar_correspondences[tuple(second_lidar_3d)] = {
                "master_3d_mean": mean_3d_point,
                "master_3d_points_indices": original_indices,
                "second_lidar_index": self.selected_second_lidar_3d_idx,
            }
        elif self.selected_2d_point is not None:
            selected_valid_indices = [item.data(0) for item in self.current_3d_selection]
            original_indices = [self.valid_indices[i] for i in selected_valid_indices]
            mean_3d_point = np.mean(self.points_xyz[original_indices], axis=0)
            self.correspondences[self.selected_2d_point] = {
                "3d_mean": mean_3d_point,
                "3d_points_indices": original_indices,
            }
        else:
            self.reset_selection_mode()
            return

        self.update_corr_list()
        self.reset_selection_mode()
        self._update_calibrate_button_highlight()
        self._highlight_export_button()

    def reset_calibration_state(self):
        self.correspondences = {}
        self.lidar_to_lidar_correspondences = {}
        self.update_corr_list()
        self.extrinsics = np.copy(self.initial_extrinsics)
        self.second_lidar_transform = np.eye(4)
        self.occlusion_mask = None
        self.second_occlusion_mask = None
        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self.update_results_display()
        self._update_inputs_from_extrinsics()
        self.clear_all_highlighting()
        self.reset_selection_mode()
        self._update_calibrate_button_highlight()

    def reset_selection_mode(self):
        self.selection_mode = None
        self.selected_2d_point = None
        if hasattr(self, "selected_second_lidar_point"):
            delattr(self, "selected_second_lidar_point")
        if hasattr(self, "selected_second_lidar_3d_idx"):
            delattr(self, "selected_second_lidar_3d_idx")
        self.clear_temp_markers()
        for item in self.current_3d_selection:
            if item.scene():
                self.scene.removeItem(item)
        self.current_3d_selection = []
        self.selected_3d_items_map = {}
        self.add_corr_button.setChecked(False)
        self.add_corr_button.setText("Add Correspondence")
        self.confirm_3d_button.setVisible(False)
        self._update_confirm_button_state()
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    def display_image(self):
        if self.original_cv_image is None:
            if hasattr(self.image_msg, "_type") and self.image_msg._type == "sensor_msgs/msg/CompressedImage":
                np_arr = np.frombuffer(self.image_msg.data, np.uint8)
                self.original_cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                self.original_cv_image = self.ros_utils.image_to_numpy(self.image_msg)

            if "bgr" in self.image_msg.encoding:
                self.original_cv_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)

        if self.image_crop == "Left Half":
            w = self.original_cv_image.shape[1]
            self.original_cv_image = self.original_cv_image[:, :w//2]
        elif self.image_crop == "Right Half":
            w = self.original_cv_image.shape[1]
            self.original_cv_image = self.original_cv_image[:, w//2:]

        if self.is_rectification_enabled and self.ocam_model is None:
            bgr_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_RGB2BGR)
            rectified_bgr = self.rectify_image(bgr_image)
            self.cv_image = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        else:
            self.cv_image = self.original_cv_image.copy()

        h, w, c = self.cv_image.shape
        self.image_res_label.setText(f"{w} x {h}")

        items_to_preserve = []
        for item in self.scene.items():
            if not isinstance(item, QGraphicsPixmapItem) and not isinstance(item, PointCloudItem):
                items_to_preserve.append(item)

        self.scene.clear()
        q_image = QImage(self.cv_image.data, w, h, 3 * w, QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap.fromImage(q_image))

        for item in items_to_preserve:
            self.scene.addItem(item)

        self.project_pointcloud()

    def display_camera_intrinsics(self):
        display_text = f"Image Size: {self.camerainfo_msg.width} x {self.camerainfo_msg.height}\n"

        if self.ocam_model is not None:
            display_text += "\n--- OCamCalib Fisheye Model ---\n"
            display_text += f"\nPolynomial:\n {self.ocam_model.pol}"
            display_text += f"\nInverse Polynomial:\n {self.ocam_model.invpol}"
            display_text += f"\nCenter (xc, yc): {self.ocam_model.center}"
            display_text += f"\nAffine (c, d, e): {self.ocam_model.affine}\n"
        else:
            K = np.array(self.camerainfo_msg.k).reshape(3, 3)
            display_text += "\nCamera Matrix K:\n"
            for i in range(3):
                row_text = "  ".join(f"{K[i, j]:8.2f}" for j in range(3))
                display_text += f"[{row_text}]\n"

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            display_text += f"\nFocal Length: fx={fx:.1f}, fy={fy:.1f}"
            display_text += f"\nPrincipal Point: cx={cx:.1f}, cy={cy:.1f}\n\n"
            display_text += f"Distortion Model: {self.camerainfo_msg.distortion_model}"

            if hasattr(self.camerainfo_msg, "d") and len(self.camerainfo_msg.d) > 0:
                dist_coeffs = np.array(self.camerainfo_msg.d)
                display_text += "\n\nDistortion Coeffs: ["
                coeffs_str = ", ".join(f"{coeff:.6f}" for coeff in dist_coeffs)
                display_text += coeffs_str + "]"

                if len(dist_coeffs) >= 4:
                    display_text += f"\nk1={dist_coeffs[0]:.6f}, k2={dist_coeffs[1]:.6f}"
                    display_text += f"\np1={dist_coeffs[2]:.6f}, p2={dist_coeffs[3]:.6f}"
                    if len(dist_coeffs) >= 5:
                        display_text += f", k3={dist_coeffs[4]:.6f}"
                    if len(dist_coeffs) >= 8:
                        display_text += f"\nk4={dist_coeffs[5]:.6f}, k5={dist_coeffs[6]:.6f}, k6={dist_coeffs[7]:.6f}"
            else:
                display_text += "\nDistortion Coeffs: None"

        self.intrinsics_display.setPlainText(display_text)

    def redraw_points(self):
        self.project_pointcloud(self.extrinsics, re_read_cloud=False)
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self.apply_view_button.setStyleSheet(self.default_button_style)

    def project_pointcloud(self, extrinsics=None, re_read_cloud=True):
        if extrinsics is not None:
            self.extrinsics = extrinsics
        self.clear_all_highlighting()
        if self.point_cloud_item is not None and self.point_cloud_item.scene():
            self.scene.removeItem(self.point_cloud_item)
        self.point_cloud_item = None

        if re_read_cloud:
            cloud_arr = self.ros_utils.pointcloud2_to_structured_array(self.pointcloud_msg)
            valid_mask = (np.isfinite(cloud_arr["x"]) & np.isfinite(cloud_arr["y"]) & np.isfinite(cloud_arr["z"]))
            cloud_arr = cloud_arr[valid_mask]
            self.points_xyz = np.vstack([cloud_arr["x"], cloud_arr["y"], cloud_arr["z"]]).T
            self.intensities = cloud_arr["intensity"]
            if self.intensities.size > 0:
                self._update_min_max_values_for_mode()

        if not hasattr(self, "points_xyz") or self.points_xyz.shape[0] == 0:
            return

        tvec = self.extrinsics[:3, 3]
        points_cam = (self.extrinsics[:3, :3] @ self.points_xyz.T).T + tvec
        
        if self.ocam_model is not None and not self.is_rectification_enabled:
            # Project camera-frame points through the Scaramuzza OCamCalib model.
            # `project_points` expects (N, 3) camera-frame coordinates and returns (N, 2) pixels.
            points_proj_cv = self.ocam_model.project_points(points_cam)
            # Defensive reshape: mirror what the pinhole path does with .reshape(-1, 2)
            points_proj_cv = np.array(points_proj_cv).reshape(-1, 2)

            # FOV mask: discard points whose incidence angle (from the optical axis)
            # exceeds the user-configured half-angle limit.  arctan2(rho, z) gives 0°
            # for on-axis forward points and 90° for side-on points — correctly
            # rejecting behind-camera points (angle > 90° when z < 0).
            fov_limit_deg = self.fov_limit_spinbox.value()
            norm_3d = np.linalg.norm(points_cam, axis=1)
            norm_3d_safe = np.where(norm_3d < 1e-9, 1.0, norm_3d)
            ray_angles = np.arccos(np.clip(points_cam[:, 2] / norm_3d_safe, -1, 1))
            z_mask = ray_angles <= np.radians(fov_limit_deg)
        else:
            K = np.array(self.camerainfo_msg.k).reshape(3, 3)
            rvec, _ = cv2.Rodrigues(self.extrinsics[:3, :3])
            dist_coeffs = None if self.is_rectification_enabled else np.array(self.camerainfo_msg.d)

            # Optional: Check if we should use OpenCV's Fisheye module
            if hasattr(self.camerainfo_msg, "distortion_model") and self.camerainfo_msg.distortion_model in ["fisheye", "equidistant"]:
                # cv2.fisheye requires (N, 1, 3) shape
                points_xyz_reshape = self.points_xyz.reshape(-1, 1, 3)
                points_proj_cv, _ = cv2.fisheye.projectPoints(points_xyz_reshape, rvec, tvec, K, dist_coeffs)
            else:
                points_proj_cv, _ = cv2.projectPoints(self.points_xyz, rvec, tvec, K, dist_coeffs)

            points_proj_cv = points_proj_cv.reshape(-1, 2)
            z_mask = points_cam[:, 2] > 0

        # Dynamic mask based on ACTUAL image resolution
        display_h, display_w = self.cv_image.shape[:2]

        mask = (
            z_mask
            & (points_proj_cv[:, 0] >= 0)
            & (points_proj_cv[:, 0] < display_w)
            & (points_proj_cv[:, 1] >= 0)
            & (points_proj_cv[:, 1] < display_h)
        )

        if self.occlusion_mask is not None and len(self.occlusion_mask) == len(mask):
            mask = np.logical_and(mask, self.occlusion_mask)

        self.valid_indices = np.where(mask)[0]
        self.points_proj_valid = points_proj_cv[self.valid_indices]
        self.intensities_valid = self.intensities[self.valid_indices]
        if self.points_proj_valid.shape[0] == 0:
            self.kdtree = None
            return

        self.kdtree = KDTree(self.points_proj_valid)
        cmap = cm.get_cmap(self.colormap_combo.currentText())

        if self.colorization_mode_combo.currentText() == "Distance":
            valid_points_cam = points_cam[self.valid_indices]
            distances = np.linalg.norm(valid_points_cam, axis=1)
            min_val, max_val = self.min_value_spinbox.value(), self.max_value_spinbox.value()
            norm_values = np.clip((distances - min_val) / (max_val - min_val + 1e-6), 0, 1)
        else:
            min_val, max_val = self.min_value_spinbox.value(), self.max_value_spinbox.value()
            norm_values = np.clip((self.intensities_valid - min_val) / (max_val - min_val + 1e-6), 0, 1)

        colors = cmap(norm_values)
        colors[:, 3] = 0.8
        self.point_cloud_item = PointCloudItem(self.points_proj_valid, colors, self.point_size_spinbox.value())
        self.scene.addItem(self.point_cloud_item)

    def project_second_pointcloud(self, transform=None):
        if not self.has_second_pointcloud:
            return

        if transform is not None:
            self.second_lidar_transform = transform

        if self.second_point_cloud_item is not None and self.second_point_cloud_item.scene():
            self.scene.removeItem(self.second_point_cloud_item)
        self.second_point_cloud_item = None

        cloud_arr = self.ros_utils.pointcloud2_to_structured_array(self.second_pointcloud_msg)
        valid_mask = (np.isfinite(cloud_arr["x"]) & np.isfinite(cloud_arr["y"]) & np.isfinite(cloud_arr["z"]))
        cloud_arr = cloud_arr[valid_mask]
        second_points_xyz = np.vstack([cloud_arr["x"], cloud_arr["y"], cloud_arr["z"]]).T
        second_intensities = cloud_arr["intensity"]

        if second_points_xyz.shape[0] == 0:
            return

        second_points_homogeneous = np.hstack([second_points_xyz, np.ones((second_points_xyz.shape[0], 1))])
        transformed_points = (self.second_lidar_transform @ second_points_homogeneous.T).T[:, :3]

        tvec = self.extrinsics[:3, 3]
        points_cam = (self.extrinsics[:3, :3] @ transformed_points.T).T + tvec

        if self.ocam_model is not None and not self.is_rectification_enabled:
            # Same OCamCalib projection path as project_pointcloud — see comments there.
            points_proj_cv = self.ocam_model.project_points(points_cam)
            points_proj_cv = np.array(points_proj_cv).reshape(-1, 2)

            fov_limit_deg = self.fov_limit_spinbox.value()
            norm_3d = np.linalg.norm(points_cam, axis=1)
            norm_3d_safe = np.where(norm_3d < 1e-9, 1.0, norm_3d)
            ray_angles = np.arccos(np.clip(points_cam[:, 2] / norm_3d_safe, -1, 1))
            z_mask = ray_angles <= np.radians(fov_limit_deg)
        else:
            K = np.array(self.camerainfo_msg.k).reshape(3, 3)
            rvec, _ = cv2.Rodrigues(self.extrinsics[:3, :3])
            dist_coeffs = None if self.is_rectification_enabled else np.array(self.camerainfo_msg.d)
            points_proj_cv, _ = cv2.projectPoints(transformed_points, rvec, tvec, K, dist_coeffs)
            points_proj_cv = points_proj_cv.reshape(-1, 2)
            z_mask = points_cam[:, 2] > 0

        # Dynamic mask based on ACTUAL image resolution
        display_h, display_w = self.cv_image.shape[:2]

        mask = (
            z_mask
            & (points_proj_cv[:, 0] >= 0)
            & (points_proj_cv[:, 0] < display_w)
            & (points_proj_cv[:, 1] >= 0)
            & (points_proj_cv[:, 1] < display_h)
        )

        if self.second_occlusion_mask is not None and len(self.second_occlusion_mask) == len(mask):
            mask = np.logical_and(mask, self.second_occlusion_mask)

        self.second_valid_indices = np.where(mask)[0]
        self.second_points_proj_valid = points_proj_cv[self.second_valid_indices]
        self.second_intensities_valid = second_intensities[self.second_valid_indices]
        self.second_points_xyz = second_points_xyz

        if self.second_points_proj_valid.shape[0] == 0:
            self.second_kdtree = None
            return

        self.second_kdtree = KDTree(self.second_points_proj_valid)

        second_cmap = cm.get_cmap("plasma")
        min_val = np.quantile(self.second_intensities_valid, 0.01)
        max_val = np.quantile(self.second_intensities_valid, 0.90)
        norm_values = np.clip((self.second_intensities_valid - min_val) / (max_val - min_val + 1e-6), 0, 1)

        colors = second_cmap(norm_values)
        colors[:, 3] = 0.7 

        self.second_point_cloud_item = PointCloudItem(self.second_points_proj_valid, colors, self.point_size_spinbox.value())
        self.scene.addItem(self.second_point_cloud_item)

    def update_corr_list(self):
        self.corr_list_widget.clear()
        for p2d, corr_data in self.correspondences.items():
            p3d = corr_data["3d_mean"]
            item_text = f"Cam ({p2d[0]:.1f}, {p2d[1]:.1f}) ↔ Master ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ("master_cam", p2d))
            self.corr_list_widget.addItem(item)
        for second_3d, corr_data in self.lidar_to_lidar_correspondences.items():
            master_3d = corr_data["master_3d_mean"]
            item_text = f"Second ({second_3d[0]:.2f}, {second_3d[1]:.2f}, {second_3d[2]:.2f}) ↔ Master ({master_3d[0]:.2f}, {master_3d[1]:.2f}, {master_3d[2]:.2f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ("lidar_lidar", second_3d))
            self.corr_list_widget.addItem(item)

    def delete_correspondence(self):
        current_item = self.corr_list_widget.currentItem()
        if current_item:
            corr_data = current_item.data(Qt.UserRole)
            if corr_data[0] == "master_cam":
                p2d_key = corr_data[1]
                if p2d_key in self.correspondences:
                    del self.correspondences[p2d_key]
            elif corr_data[0] == "lidar_lidar":
                second_3d_key = corr_data[1]
                if second_3d_key in self.lidar_to_lidar_correspondences:
                    del self.lidar_to_lidar_correspondences[second_3d_key]
            self.update_corr_list()
            self.clear_all_highlighting()
            self._update_calibrate_button_highlight()
            self._highlight_export_button()

    def highlight_from_list(self, current_item, previous_item):
        self.clear_all_highlighting()
        if not current_item:
            return

        corr_data = current_item.data(Qt.UserRole)
        point_size = self.point_size_spinbox.value()

        if corr_data[0] == "master_cam":
            p2d_key = corr_data[1]
            corr = self.correspondences.get(p2d_key)
            if not corr:
                return
            self.draw_cross_marker(QPointF(p2d_key[0], p2d_key[1]), QColor(Colors.CORRESPONDENCE_3D))
            original_to_valid_idx_map = {orig_idx: valid_idx for valid_idx, orig_idx in enumerate(self.valid_indices)}
            for original_point_idx in corr["3d_points_indices"]:
                valid_idx = original_to_valid_idx_map.get(original_point_idx)
                if valid_idx is not None and valid_idx < len(self.points_proj_valid):
                    point_2d = self.points_proj_valid[valid_idx]
                    item = QGraphicsEllipseItem(point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size)
                    item.setPen(QPen(QColor(Colors.CORRESPONDENCE_3D), 2))
                    item.setBrush(QBrush(QColor(Colors.CORRESPONDENCE_3D)))
                    self.scene.addItem(item)
                    self.highlighted_3d_items.append(item)

        elif corr_data[0] == "lidar_lidar":
            second_3d_key = corr_data[1]
            corr = self.lidar_to_lidar_correspondences.get(second_3d_key)
            if not corr:
                return
            second_idx = corr["second_lidar_index"]
            if hasattr(self, "second_valid_indices") and hasattr(self, "second_points_proj_valid"):
                second_valid_idx_map = {orig_idx: valid_idx for valid_idx, orig_idx in enumerate(self.second_valid_indices)}
                second_valid_idx = second_valid_idx_map.get(second_idx)
                if second_valid_idx is not None and second_valid_idx < len(self.second_points_proj_valid):
                    point_2d = self.second_points_proj_valid[second_valid_idx]
                    self.draw_cross_marker(QPointF(point_2d[0], point_2d[1]), QColor(255, 0, 255))
            original_to_valid_idx_map = {orig_idx: valid_idx for valid_idx, orig_idx in enumerate(self.valid_indices)}
            for original_point_idx in corr["master_3d_points_indices"]:
                valid_idx = original_to_valid_idx_map.get(original_point_idx)
                if valid_idx is not None and valid_idx < len(self.points_proj_valid):
                    point_2d = self.points_proj_valid[valid_idx]
                    item = QGraphicsEllipseItem(point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size)
                    item.setPen(QPen(QColor(0, 255, 0), 2))
                    item.setBrush(QBrush(QColor(0, 255, 0)))
                    self.scene.addItem(item)
                    self.highlighted_3d_items.append(item)

    def draw_cross_marker(self, center, color):
        pen = QPen(color, 2)
        size = 10
        l1 = self.scene.addLine(center.x() - size, center.y(), center.x() + size, center.y(), pen)
        l2 = self.scene.addLine(center.x(), center.y() - size, center.x(), center.y() + size, pen)
        self.temp_2d_marker.extend([l1, l2])

    def clear_all_highlighting(self):
        self.clear_temp_markers()
        self.clear_highlighted_3d_points()

    def clear_temp_markers(self):
        for item in self.temp_2d_marker:
            if item.scene():
                self.scene.removeItem(item)
        self.temp_2d_marker = []

    def clear_highlighted_3d_points(self):
        for item in self.highlighted_3d_items:
            if item.scene():
                self.scene.removeItem(item)
        self.highlighted_3d_items = []
        for item in self.current_3d_selection:
            if item.scene():
                self.scene.removeItem(item)
        self.current_3d_selection = []
        self.selected_3d_items_map = {}

    def run_calibration(self):
        if len(self.correspondences) < AppConstants.MIN_CORRESPONDENCES:
            return
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        ransac_method_str = self.pnp_solver_combo.currentText()
        pnp_flag = {"SQPnP": cv2.SOLVEPNP_SQPNP, "Iterative": cv2.SOLVEPNP_ITERATIVE}.get(ransac_method_str)
        lsq_method = self.lsq_method_combo.currentText()
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)

        kwargs = {}
        if self.ocam_model is not None:
            kwargs["ocam_model"] = self.ocam_model
        kwargs["initial_transform"] = self.extrinsics

        if self.has_second_pointcloud and len(self.lidar_to_lidar_correspondences) >= 3:
            master_cam_corr = [(p2d, corr["3d_mean"]) for p2d, corr in self.correspondences.items()]
            self.extrinsics, self.second_lidar_transform = calibration.calibrate_dual_lidar(
                master_cam_corr, self.lidar_to_lidar_correspondences, K, pnp_flag, lsq_method, **kwargs
            )
        else:
            calib_corr = [(p2d, corr["3d_mean"]) for p2d, corr in self.correspondences.items()]
            self.extrinsics = calibration.calibrate(calib_corr, K, pnp_flag, lsq_method, **kwargs)

        self.progress_bar.setVisible(False)
        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self.update_results_display()
        self._update_inputs_from_extrinsics()
        self._highlight_export_button()

    def update_results_display(self):
        self.results_label.setText("Calibration parameters updated")

    def export_calibration(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", "YAML Files (*.yaml)")
        if file_path:
            t = self.extrinsics[:3, 3]
            q = Rotation.from_matrix(self.extrinsics[:3, :3]).as_quat()
            with open(file_path, "w") as f:
                f.write("# LiDAR-Camera Extrinsic Calibration (T_camera_lidar)\n")
                f.write(f"translation:\n  x: {t[0]:.8f}\n  y: {t[1]:.8f}\n  z: {t[2]:.8f}\n")
                f.write(f"rotation:\n  x: {q[0]:.8f}\n  y: {q[1]:.8f}\n  z: {q[2]:.8f}\n  w: {q[3]:.8f}\n")
            print(f"Calibration saved to {file_path}")

    def view_calibration_results(self):
        self.export_button.setStyleSheet(self.default_button_style)
        if self.has_second_pointcloud:
            calibration_results = {
                "mode": "dual_lidar",
                "master_to_camera": self.extrinsics,
                "master_to_second_lidar": self.second_lidar_transform,
            }
        else:
            calibration_results = {"mode": "single_lidar", "master_to_camera": self.extrinsics}

        self.calibration_completed.emit(calibration_results)