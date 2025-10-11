from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import datetime
from utils.utils import enumerate_usb_external_cameras, open_camera_hint, get_preferred_camera
from domain.camera_type import CameraType
from components.animated_button import AnimatedButton
from constants import THEME
from config import DEBUG_MODE

class VideoScreen(QtWidgets.QWidget):
    """Pantalla con botón para tomar video desde la primera cámara USB externa."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_data = {}

        card = QtWidgets.QFrame(objectName="Card")
        card_vertical_layout = QtWidgets.QVBoxLayout(card)
        card_vertical_layout.setContentsMargins(24, 24, 24, 24)
        card_vertical_layout.setSpacing(16)

        title = QtWidgets.QLabel("Tomar Foto", objectName="H1")
        subtitle = QtWidgets.QLabel("Coloca la cámara y presiona el botón para capturar.", objectName="Muted")

        # Camera info label to display detected camera type
        self.camera_info_label = QtWidgets.QLabel("", objectName="Muted")
        self.camera_info_label.setVisible(False)

        # Estado de última captura
        self._last_snap_path = None
        self._last_snap_dir = None

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
        recording_object: int = 0
        self.cap = None

    def record(self):
        self.record_button.setDisabled(True)
        self.feedback.setVisible(False)
        app = QtWidgets.QApplication.instance()

        # Descubrir cámaras USB externas con detección de tipo
        cameras = []
        try:
            cameras = enumerate_usb_external_cameras(detect_specialized=True) or []
        except Exception as e:
            if DEBUG_MODE:
                print("enumerate_usb_external_cameras():", e)

        if not cameras:
            self._show_error("No se encontró ninguna cámara USB.")
            self.record_button.setDisabled(False)
            return

        try:
            self.recording_object += 1
            # Obtener la cámara preferida (prioriza RealSense y Zed)
            camera = get_preferred_camera(cameras)

            if camera:
                # Mostrar información de la cámara detectada
                camera_info = f"Cámara: {str(camera.camera_type)}"

                if camera.capabilities.depth:
                    camera_info += " (con profundidad)"
                if camera.capabilities.imu:
                    camera_info += " + IMU"

                self.camera_info_label.setText(camera_info)
                self.camera_info_label.setVisible(True)

                if DEBUG_MODE:
                    print(f"Usando cámara: {camera}")
                    print(f"  VID/PID: {camera.usb_vid}:{camera.usb_pid}")
                    print(f"  Capacidades: {camera.capabilities}")

            # Abrir cámara con helper del proyecto
            try:
                self.cap = open_camera_hint(camera.open_hint if camera else 0)
            except Exception as e:
                if DEBUG_MODE:
                    print("open_camera_hint():", e)
                self._show_error("No se pudo abrir la cámara.")
                self.record_button.setDisabled(False)
                return

            # Capturar frame
            ok, frame = (False, None)
            try:
                ok, frame = self.cap.read()
            except Exception as e:
                if DEBUG_MODE:
                    print("cap.read():", e)
            finally:
                try:
                    if self.cap:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None

            if not ok or frame is None:
                self._show_error("No se pudo capturar la foto.")
                self.record_button.setDisabled(False)
                return

            # Guardar en disco con timestamp
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.abspath(os.path.join(os.getcwd(), "snaps"))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"foto_{ts}.jpg")

            try:
                cv2.imwrite(out_path, frame)
                if DEBUG_MODE:
                    self._last_snap_path = out_path
                    self._last_snap_dir = out_dir
                
            except Exception as e:
                if DEBUG_MODE:
                    print("cv2.imwrite():", e)
                self._show_error("No se pudo guardar la foto.")
                self.record_button.setDisabled(False)
                return

            self.feedback.setObjectName("Success")
            self.feedback.setText(f"✔ Foto guardada: {out_path}")
            self.feedback.setVisible(True)
            self.record_button.setDisabled(False)
        except Exception as e:
            self.recording_object -= 1
            if DEBUG_MODE:
                print("record():", e)
            self._show_error("Ocurrió un error inesperado.")
            self.record_button.setDisabled(False)

    def _show_error(self, text: str):
        self.feedback.setObjectName("Error")
        self.feedback.setText(text)
        self.feedback.setVisible(True)

    @QtCore.pyqtSlot(dict)
    def set_session_data(self, data: dict):
        self._session_data = data or {}
