from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import datetime
from utils.utils import enumerate_usb_external_cameras, open_camera_hint, get_preferred_camera
from domain.camera_type import CameraType
from components.animated_button import AnimatedButton
from constants import THEME
from config import DEBUG_MODE, PHOTO_PREVIEW

class CaptureScreen(QtWidgets.QWidget):
    """Pantalla con botón para tomar foto desde la primera cámara USB externa."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_data = {}

        card = QtWidgets.QFrame(objectName="Card")
        card_vertical_layout = QtWidgets.QVBoxLayout(card)
        card_vertical_layout.setContentsMargins(24, 24, 24, 24)
        card_vertical_layout.setSpacing(16)

        title = QtWidgets.QLabel("Tomar Foto", objectName="H1")
        subtitle = QtWidgets.QLabel("Coloca la cámara y presiona el botón para capturar.", objectName="Muted")

        # Vista previa
        if PHOTO_PREVIEW:
            self.preview = QtWidgets.QLabel()
            self.preview.setFixedSize(420, 260)
            self.preview.setAlignment(QtCore.Qt.AlignCenter)
            self.preview.setStyleSheet("border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;")
            self.preview.setText("Sin vista previa")

        # Estado de última captura
        self._last_snap_path = None
        self._last_snap_dir = None

        self.btn_snap = AnimatedButton("Tomar foto")
        self.btn_snap.clicked.connect(self.snap)

        self.feedback = QtWidgets.QLabel("")
        self.feedback.setObjectName("Success")
        self.feedback.setVisible(False)

        card_vertical_layout.addWidget(title)
        card_vertical_layout.addWidget(subtitle)
        card_vertical_layout.addSpacing(8)
        if PHOTO_PREVIEW:
            card_vertical_layout.addWidget(self.preview)
        card_vertical_layout.addWidget(self.btn_snap, 0, QtCore.Qt.AlignLeft)
        card_vertical_layout.addWidget(self.feedback)
        card_vertical_layout.addStretch(1)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.addStretch(1)
        root.addWidget(card, 0, QtCore.Qt.AlignHCenter)
        card.setMaximumWidth(THEME["sizes"]["content_maxw"])
        root.addStretch(2)

        self.cap = None

    def snap(self):
        self.btn_snap.setDisabled(True)
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
            self.btn_snap.setDisabled(False)
            return

        # Obtener la cámara preferida (prioriza RealSense y Zed)
        camera = get_preferred_camera(cameras)

        if camera and DEBUG_MODE:
            print(f"Usando cámara: {camera}")
            print(f"  Tipo: {camera.camera_type}")
            print(f"  VID/PID: {camera.usb_vid}:{camera.usb_pid}")
            print(f"  Capacidades: {camera.capabilities}")

        # Abrir cámara con helper del proyecto
        try:
            self.cap = open_camera_hint(camera.open_hint if camera else 0)
        except Exception as e:
            if DEBUG_MODE:
                print("open_camera_hint():", e)
            self._show_error("No se pudo abrir la cámara.")
            self.btn_snap.setDisabled(False)
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
            self.btn_snap.setDisabled(False)
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
            if PHOTO_PREVIEW:
                self._set_preview(out_path)
        except Exception as e:
            if DEBUG_MODE:
                print("cv2.imwrite():", e)
            self._show_error("No se pudo guardar la foto.")
            self.btn_snap.setDisabled(False)
            return

        self.feedback.setObjectName("Success")
        self.feedback.setText(f"✔ Foto guardada: {out_path}")
        self.feedback.setVisible(True)
        self.btn_snap.setDisabled(False)

    def _show_error(self, text: str):
        self.feedback.setObjectName("Error")
        self.feedback.setText(text)
        self.feedback.setVisible(True)
    
    def _set_preview(self, img_path: str):
        pm = QtGui.QPixmap(img_path)
        if pm.isNull():
            self.preview.setText("No se pudo cargar la vista previa")
            self.preview.setPixmap(QtGui.QPixmap())
            return
        scaled = pm.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    @QtCore.pyqtSlot(dict)
    def set_session_data(self, data: dict):
        self._session_data = data or {}
