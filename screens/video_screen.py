from PyQt5 import QtCore, QtGui, QtWidgets
import threading
import platform
from utils.utils import enumerate_usb_external_cameras, filter_cameras_by_type
from utils.camera_orchestrator import CameraRecordingOrchestrator
from domain.camera_type import CameraType
from components.animated_button import AnimatedButton
from constants import THEME
from config import DEBUG_MODE

class VideoScreen(QtWidgets.QWidget):
    """Pantalla de grabación multi-cámara sincronizada."""

    # Qt signals for thread-safe UI updates
    _update_feedback_signal = QtCore.pyqtSignal(str, str)  # (message, style)
    _enable_button_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_data = {}

        # Connect signals to slot methods
        self._update_feedback_signal.connect(self._update_feedback)
        self._enable_button_signal.connect(self._enable_record_button)

        card = QtWidgets.QFrame(objectName="Card")
        card_vertical_layout = QtWidgets.QVBoxLayout(card)
        card_vertical_layout.setContentsMargins(24, 24, 24, 24)
        card_vertical_layout.setSpacing(16)

        title = QtWidgets.QLabel("Grabación Multi-Cámara", objectName="H1")
        subtitle = QtWidgets.QLabel("Conecta las cámaras RealSense o Zed y presiona grabar.", objectName="Muted")

        # Camera info label to display detected cameras
        self.camera_info_label = QtWidgets.QLabel("", objectName="Muted")
        self.camera_info_label.setVisible(False)

        self.record_button = AnimatedButton("Grabar")
        self.record_button.clicked.connect(self.record)

        self.feedback = QtWidgets.QLabel("")
        self.feedback.setObjectName("Success")
        self.feedback.setVisible(False)

        card_vertical_layout.addWidget(title)
        card_vertical_layout.addWidget(subtitle)
        card_vertical_layout.addWidget(self.camera_info_label)
        card_vertical_layout.addSpacing(8)

        card_vertical_layout.addWidget(self.record_button, 0, QtCore.Qt.AlignLeft)
        card_vertical_layout.addWidget(self.feedback)
        card_vertical_layout.addStretch(1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.addStretch(1)
        root.addWidget(card, 0, QtCore.Qt.AlignHCenter)
        card.setMaximumWidth(THEME["sizes"]["content_maxw"])
        root.addStretch(2)

    def record(self):
        """Start multi-camera synchronized recording workflow."""
        # Disable button immediately to prevent double-clicks
        self.record_button.setDisabled(True)
        # Hide previous feedback
        self.feedback.setVisible(False)

        # Platform validation - Windows only
        if platform.system() != 'Windows':
            self._show_error("Grabación multi-cámara solo disponible en Windows")
            self.record_button.setDisabled(False)
            return

        # Enumerate USB cameras with specialized detection
        cameras = []
        try:
            cameras = enumerate_usb_external_cameras(detect_specialized=True) or []
        except Exception as e:
            if DEBUG_MODE:
                print(f"enumerate_usb_external_cameras() error: {e}")

        # Check if any cameras found
        if not cameras:
            self._show_error("No se encontró ninguna cámara USB.")
            self.record_button.setDisabled(False)
            return

        # Filter for specialized cameras (RealSense and Zed)
        specialized_types = [
            CameraType.REALSENSE_D455,
            CameraType.REALSENSE_D455i,
            CameraType.ZED_2,
            CameraType.ZED_2i
        ]
        specialized_cameras = filter_cameras_by_type(cameras, specialized_types)

        if not specialized_cameras:
            self._show_error(
                "No se encontraron cámaras especializadas (RealSense o Zed).\n"
                "Conecta al menos una cámara RealSense D455/D455i o Zed 2/2i."
            )
            self.record_button.setDisabled(False)
            return

        # Display detected cameras
        camera_names = [str(cam.camera_type.value) for cam in specialized_cameras]
        camera_info_text = f"Cámaras: {', '.join(camera_names)}"
        self.camera_info_label.setText(camera_info_text)
        self.camera_info_label.setVisible(True)

        if DEBUG_MODE:
            print(f"Detected {len(specialized_cameras)} specialized cameras:")
            for cam in specialized_cameras:
                print(f"  - {cam.camera_type} (VID: {cam.usb_vid}, PID: {cam.usb_pid})")

        # Create orchestrator
        try:
            orchestrator = CameraRecordingOrchestrator(
                cameras=specialized_cameras
            )
        except Exception as e:
            if DEBUG_MODE:
                print(f"Orchestrator creation error: {e}")
            self._show_error(f"Error al crear el orquestador: {e}")
            self.record_button.setDisabled(False)
            return

        # Initialize cameras
        try:
            success_count, failed_count = orchestrator.initialize_cameras()
        except Exception as e:
            if DEBUG_MODE:
                print(f"Camera initialization error: {e}")
            self._show_error(f"Error al inicializar cámaras: {e}")
            self.record_button.setDisabled(False)
            return

        # Update camera info with initialization results
        if success_count > 0:
            camera_info_text += f" ({success_count} OK"
            if failed_count > 0:
                camera_info_text += f", {failed_count} error"
            camera_info_text += ")"
            self.camera_info_label.setText(camera_info_text)

        # Check if any cameras initialized successfully
        if success_count == 0:
            self._show_error(
                f"No se pudo inicializar ninguna cámara.\n"
                f"Intentadas: {failed_count}, Fallidas: {failed_count}"
            )
            self.record_button.setDisabled(False)
            return

        # Start recording workflow in background thread
        workflow_thread = threading.Thread(
            target=self._recording_workflow,
            args=(orchestrator,),
            daemon=True
        )
        workflow_thread.start()

        if DEBUG_MODE:
            print(f"Recording workflow started in background thread")

    def _recording_workflow(self, orchestrator: CameraRecordingOrchestrator):
        """
        Recording workflow that runs in background thread.

        This method performs the actual recording, emits signals for UI updates,
        and ensures proper cleanup.

        Args:
            orchestrator: Initialized CameraRecordingOrchestrator instance
        """
        try:
            # Start recording with countdown
            if not orchestrator.start_recording_with_countdown():
                self._update_feedback_signal.emit(
                    "Error al iniciar la grabación.",
                    "Error"
                )
                return

            # Update UI to show recording in progress
            self._update_feedback_signal.emit(
                "⏺️ Grabando...",
                "Success"
            )

            # Wait for recording to complete
            orchestrator.wait_for_completion()

            # Get recording status
            status = orchestrator.get_recording_status()
            output_dir = status.get("output_dir", "")

            # Emit success signal with output directory
            success_message = (
                f"✔ Grabación completada\n"
                f"Guardado en: {output_dir}"
            )
            self._update_feedback_signal.emit(success_message, "Success")

            if DEBUG_MODE:
                print(f"Recording completed successfully")
                print(f"Session ID: {status.get('session_id', 'N/A')}")
                print(f"Camera count: {status.get('camera_count', 0)}")
                print(f"Duration: {status.get('duration_secs', 0)}s")

        except Exception as e:
            if DEBUG_MODE:
                print(f"Recording workflow error: {e}")
            # Emit error signal
            self._update_feedback_signal.emit(
                f"Error: {str(e)}",
                "Error"
            )

        finally:
            # Always stop recording and cleanup
            try:
                orchestrator.stop_recording()
            except Exception as e:
                if DEBUG_MODE:
                    print(f"Error stopping recording: {e}")

            # Always re-enable button
            self._enable_button_signal.emit()

    def _update_feedback(self, message: str, style: str):
        """
        Slot method to update feedback label (runs in main thread).

        Args:
            message: Feedback message text
            style: Style name ("Success" or "Error")
        """
        self.feedback.setObjectName(style)
        self.feedback.setText(message)
        self.feedback.setVisible(True)

    def _enable_record_button(self):
        """Slot method to re-enable record button (runs in main thread)."""
        self.record_button.setDisabled(False)

    def _show_error(self, text: str):
        """Helper method to display error feedback."""
        self.feedback.setObjectName("Error")
        self.feedback.setText(text)
        self.feedback.setVisible(True)

    @QtCore.pyqtSlot(dict)
    def set_session_data(self, data: dict):
        """Set session data from parent widget."""
        self._session_data = data or {}
