"""
Camera diagnostics screen for debugging camera detection.

This screen provides a detailed view of all detected cameras,
their metadata, validation status, and real-time detection logs.
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Optional, List
import logging
import traceback
from datetime import datetime

from utils.utils import enumerate_usb_external_cameras
from utils.camera_validation import validate_camera
from utils.camera_detection_cache import clear_cache
from domain.camera_info import CameraInfo
from components.animated_button import AnimatedButton
from constants import THEME
from config import DEBUG_MODE


class QtLogHandler(QtCore.QObject, logging.Handler):
    """Custom log handler that emits to Qt signal."""

    log_emitted = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        logging.Handler.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_emitted.emit(msg)
        except Exception:
            pass


class CameraDiagnosticsScreen(QtWidgets.QWidget):
    """
    Camera diagnostics screen.

    Provides detailed camera detection information, validation,
    and troubleshooting capabilities.
    """

    # Signal to navigate back to video screen
    back_to_video = QtCore.pyqtSignal()

    # Signals for thread-safe UI updates
    _update_table_signal = QtCore.pyqtSignal()
    _update_logs_signal = QtCore.pyqtSignal(str)
    _enable_buttons_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cameras: List[CameraInfo] = []
        self.selected_camera: Optional[CameraInfo] = None

        # Connect signals
        self._update_table_signal.connect(self._update_table)
        self._update_logs_signal.connect(self._append_log)
        self._enable_buttons_signal.connect(self._enable_buttons)

        # Setup logging
        self._setup_logging()

        # Setup UI
        self._setup_ui()

        # Initial detection
        self._refresh_detection()

    def _setup_logging(self):
        """Setup custom log handler for capturing detection logs."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%H:%M:%S')
        )
        self.log_handler.log_emitted.connect(self._append_log)

        # Add handler to relevant loggers
        loggers = [
            'utils.utils',
            'utils.camera_detection_realsense',
            'utils.camera_validation_zed',
            'utils.camera_validation',
        ]

        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self.log_handler)
            logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

    def _setup_ui(self):
        """Setup UI components."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Card container
        card = QtWidgets.QFrame(objectName="Card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        # Title
        title = QtWidgets.QLabel("Camera Diagnostics", objectName="H1")
        subtitle = QtWidgets.QLabel(
            "Detailed camera detection information and troubleshooting tools.",
            objectName="Muted"
        )

        # Action buttons row
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setSpacing(8)

        self.refresh_button = AnimatedButton("Refresh Detection")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)

        self.validate_button = AnimatedButton("Validate Cameras")
        self.validate_button.clicked.connect(self._on_validate_clicked)

        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.validate_button)
        buttons_layout.addStretch()

        # Camera table
        self.camera_table = QtWidgets.QTableWidget()
        self.camera_table.setColumnCount(6)
        self.camera_table.setHorizontalHeaderLabels([
            "Index", "Name", "Type", "VID:PID", "SDK", "Status"
        ])
        self.camera_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.camera_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.camera_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.camera_table.itemSelectionChanged.connect(
            self._on_camera_selected
        )

        # Adjust column widths
        header = self.camera_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

        # Details panel
        details_label = QtWidgets.QLabel("Camera Details:", objectName="H2")
        self.details_panel = QtWidgets.QLabel("Select a camera to view details.")
        self.details_panel.setObjectName("Muted")
        self.details_panel.setWordWrap(True)
        self.details_panel.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.details_panel.setMinimumHeight(100)

        # Logs area
        logs_label = QtWidgets.QLabel("Detection Logs:", objectName="H2")
        self.logs_area = QtWidgets.QTextEdit()
        self.logs_area.setReadOnly(True)
        self.logs_area.setMaximumHeight(150)
        self.logs_area.setObjectName("Muted")

        # Back button
        self.back_button = AnimatedButton("Back to Video")
        self.back_button.clicked.connect(self.back_to_video.emit)

        # Add widgets to card layout
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addLayout(buttons_layout)
        card_layout.addWidget(self.camera_table)
        card_layout.addWidget(details_label)
        card_layout.addWidget(self.details_panel)
        card_layout.addWidget(logs_label)
        card_layout.addWidget(self.logs_area)
        card_layout.addWidget(self.back_button, 0, QtCore.Qt.AlignLeft)

        # Add card to main layout
        main_layout.addWidget(card, 0, QtCore.Qt.AlignHCenter)
        card.setMaximumWidth(THEME["sizes"]["content_maxw"] + 200)

    def _on_refresh_clicked(self):
        """Handle refresh button click."""
        self._disable_buttons()
        self._append_log("[USER] Refresh detection requested")
        self._refresh_detection()
        self._enable_buttons()

    def _on_validate_clicked(self):
        """Handle validate button click."""
        self._disable_buttons()
        self._append_log("[USER] Camera validation requested")
        self._validate_cameras()
        self._enable_buttons()

    def _disable_buttons(self):
        """Disable action buttons during operations."""
        self.refresh_button.setDisabled(True)
        self.validate_button.setDisabled(True)

    def _enable_buttons(self):
        """Enable action buttons after operations."""
        self.refresh_button.setDisabled(False)
        self.validate_button.setDisabled(False)

    def _refresh_detection(self):
        """Refresh camera detection."""
        try:
            # Clear cache to force fresh detection
            clear_cache()
            self._append_log("[INFO] Cache cleared, starting fresh detection...")

            # Enumerate cameras with full detection
            self.cameras = enumerate_usb_external_cameras(
                detect_specialized=True,
                use_sdk_enhancement=True
            ) or []

            self._append_log(f"[INFO] Detection complete: {len(self.cameras)} camera(s) found")

            # Update table
            self._update_table()

        except Exception as e:
            self._append_log(f"[ERROR] Detection failed: {e}")
            if DEBUG_MODE:
                self._append_log(f"[DEBUG] Traceback: {traceback.format_exc()}")

    def _validate_cameras(self):
        """Validate all detected cameras."""
        if not self.cameras:
            self._append_log("[WARNING] No cameras to validate")
            return

        self._append_log(f"[INFO] Validating {len(self.cameras)} camera(s)...")

        for i, camera in enumerate(self.cameras):
            try:
                is_valid = validate_camera(camera, timeout_seconds=5.0)

                # Update status in table
                status_item = self.camera_table.item(i, 5)
                if status_item:
                    if is_valid:
                        status_item.setText("OK")
                        status_item.setForeground(QtGui.QColor("#4CAF50"))
                        self._append_log(f"[OK] {camera.name} validated successfully")
                    else:
                        status_item.setText("FAILED")
                        status_item.setForeground(QtGui.QColor("#F44336"))
                        self._append_log(f"[FAIL] {camera.name} validation failed")

            except Exception as e:
                self._append_log(f"[ERROR] Validation error for {camera.name}: {e}")

        self._append_log("[INFO] Validation complete")

    def _update_table(self):
        """Update camera table with current detection results."""
        self.camera_table.setRowCount(len(self.cameras))

        for i, camera in enumerate(self.cameras):
            # Index
            index_item = QtWidgets.QTableWidgetItem(str(camera.index))
            index_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.camera_table.setItem(i, 0, index_item)

            # Name
            name_item = QtWidgets.QTableWidgetItem(camera.name)
            self.camera_table.setItem(i, 1, name_item)

            # Type
            type_item = QtWidgets.QTableWidgetItem(camera.camera_type.value)
            type_item.setTextAlignment(QtCore.Qt.AlignCenter)

            # Color-code by type
            if camera.camera_type.is_realsense:
                type_item.setForeground(QtGui.QColor("#2196F3"))  # Blue for RealSense
            elif camera.camera_type.is_zed:
                type_item.setForeground(QtGui.QColor("#FF9800"))  # Orange for ZED
            else:
                type_item.setForeground(QtGui.QColor("#9E9E9E"))  # Gray for generic

            self.camera_table.setItem(i, 2, type_item)

            # VID:PID
            vid_pid_text = "N/A"
            if camera.usb_vid and camera.usb_pid:
                vid_pid_text = f"{camera.usb_vid}:{camera.usb_pid}"
            vid_pid_item = QtWidgets.QTableWidgetItem(vid_pid_text)
            vid_pid_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.camera_table.setItem(i, 3, vid_pid_item)

            # SDK
            sdk_text = "Yes" if camera.sdk_available else "No"
            sdk_item = QtWidgets.QTableWidgetItem(sdk_text)
            sdk_item.setTextAlignment(QtCore.Qt.AlignCenter)
            if camera.sdk_available:
                sdk_item.setForeground(QtGui.QColor("#4CAF50"))  # Green
            else:
                sdk_item.setForeground(QtGui.QColor("#9E9E9E"))  # Gray
            self.camera_table.setItem(i, 4, sdk_item)

            # Status (initially pending)
            status_item = QtWidgets.QTableWidgetItem("Pending")
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            status_item.setForeground(QtGui.QColor("#FFC107"))  # Amber
            self.camera_table.setItem(i, 5, status_item)

        # Clear selection and details
        self.camera_table.clearSelection()
        self.details_panel.setText("Select a camera to view details.")

    def _on_camera_selected(self):
        """Handle camera selection in table."""
        selected_rows = self.camera_table.selectedIndexes()
        if not selected_rows:
            self.details_panel.setText("Select a camera to view details.")
            self.selected_camera = None
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self.cameras):
            self.selected_camera = self.cameras[row]
            self._update_details_panel()

    def _update_details_panel(self):
        """Update details panel with selected camera info."""
        if not self.selected_camera:
            return

        cam = self.selected_camera

        details_lines = [
            f"Name: {cam.name}",
            f"Type: {cam.camera_type.value}",
            f"Index: {cam.index}",
            f"Backend: {cam.backend.value if hasattr(cam.backend, 'value') else cam.backend}",
            f"Open Hint: {cam.open_hint}",
        ]

        if cam.usb_vid and cam.usb_pid:
            details_lines.append(f"USB VID:PID: {cam.usb_vid}:{cam.usb_pid}")

        if cam.serial_number:
            details_lines.append(f"Serial Number: {cam.serial_number}")

        if cam.os_id:
            details_lines.append(f"OS ID: {cam.os_id}")

        if cam.path:
            details_lines.append(f"Path: {cam.path}")

        # Capabilities
        details_lines.append("")
        details_lines.append("Capabilities:")
        details_lines.append(f"  - Depth: {cam.capabilities.depth}")
        details_lines.append(f"  - IMU: {cam.capabilities.imu}")
        details_lines.append(f"  - Stereo: {cam.capabilities.stereo}")

        # SDK status
        details_lines.append("")
        details_lines.append(f"SDK Available: {'Yes' if cam.sdk_available else 'No'}")

        # Vendor/Model if available
        if cam.vendor:
            details_lines.append(f"Vendor: {cam.vendor}")
        if cam.model:
            details_lines.append(f"Model: {cam.model}")

        self.details_panel.setText("\n".join(details_lines))

    def _append_log(self, message: str):
        """Append log message to logs area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.logs_area.append(log_line)

        # Auto-scroll to bottom
        scrollbar = self.logs_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event):
        """Handle show event - refresh detection when screen is shown."""
        super().showEvent(event)
        # Only refresh if cameras list is empty (first time shown)
        if not self.cameras:
            self._refresh_detection()
