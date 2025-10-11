from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import json
from components.animated_button import AnimatedButton
from components.password_field import PasswordField
from constants import THEME
from config import BASE_API_URL, LOGIN_ENDPOINT_URL

class LoginScreen(QtWidgets.QWidget):
    """Pantalla de acceso con identificador y password. Dispara request a login endpoint."""
    loginSuccess = QtCore.pyqtSignal(dict)  # emite el JSON recibido al loguear OK

    def __init__(self, parent=None):
        super().__init__(parent)

        card = QtWidgets.QFrame(objectName="Card")
        form = QtWidgets.QFormLayout()
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        title = QtWidgets.QLabel("Bienvenido", objectName="H1")
        subtitle = QtWidgets.QLabel("Ingresa con tu identificador y contraseña", objectName="Muted")

        self.identifier = QtWidgets.QLineEdit()
        self.identifier.setPlaceholderText("Identificador")
        self.identifier.setClearButtonEnabled(True)

        self.password = PasswordField()
        self.password.setPlaceholderText("Contraseña")

        self.btn_login = AnimatedButton("Ingresar")
        self.btn_login.clicked.connect(self._on_login_clicked)
        self.btn_login.setIconSize(QtCore.QSize(18, 18))
        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.setInterval(60)
        self._spin_timer.timeout.connect(self._on_spin_tick)
        self._spin_angle = 0
        self._spin_base_pm = self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload).pixmap(18, 18)

        self.msg = QtWidgets.QLabel("")
        self.msg.setObjectName("Error")
        self.msg.setVisible(False)

        form.addRow(title)
        form.addRow(subtitle)
        form.addRow("Nombre de Usuario\no Email", self.identifier)
        form.addRow("Contraseña", self.password)
        form.addRow("", self.btn_login)
        form.addRow(self.msg)

        card.setLayout(form)

        # Layout centrado y contenido angosto
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)
        wrapper = QtWidgets.QWidget()
        wrapper_layout = QtWidgets.QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(card)
        card.setMaximumWidth(THEME["sizes"]["content_maxw"])

        root.addStretch(1)
        root.addWidget(wrapper, 0, QtCore.Qt.AlignHCenter)
        root.addStretch(2)

        # Network
        self.net = QtNetwork.QNetworkAccessManager(self)
        self.net.finished.connect(self._on_net_finished)

    def _on_spin_tick(self):
        self._spin_angle = (self._spin_angle + 30) % 360
        self.btn_login.setIcon(self._rotated_icon(self._spin_angle))

    def _start_spinner(self):
        self._spin_angle = 0
        self.btn_login.setIcon(QtGui.QIcon(self._spin_base_pm))
        self._spin_timer.start()

    def _stop_spinner(self):
        self._spin_timer.stop()
        self.btn_login.setIcon(QtGui.QIcon())

    def _rotated_icon(self, angle):
        transform = QtGui.QTransform().rotate(angle)
        return QtGui.QIcon(self._spin_base_pm.transformed(transform, QtCore.Qt.SmoothTransformation))

    def _on_login_clicked(self):
        identifier = self.identifier.text().strip()
        pwd = self.password.text()

        # Validación mínima
        if not identifier:
            self._set_error("Ingresá un identificador válido.")
            return
        if not pwd:
            self._set_error("La contraseña no puede estar vacía.")
            return

        self._set_error("")  # limpia
        self._set_loading(True)

        # Request POST JSON
        api_url = BASE_API_URL + LOGIN_ENDPOINT_URL
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(api_url))
        req.setHeader(QtNetwork.QNetworkRequest.ContentTypeHeader, "application/json")
        payload = {"identifier": identifier, "password": pwd}
        self._reply = self.net.post(req, json.dumps(payload).encode("utf-8"))

    def _on_net_finished(self, reply: QtNetwork.QNetworkReply):
        if reply is None:
            return
        if reply.error() != QtNetwork.QNetworkReply.NoError:
            self._set_loading(False)
            self._set_error(f"Error de red: {reply.errorString()}")
            reply.deleteLater()
            return

        data_bytes = reply.readAll().data()
        status = reply.attribute(QtNetwork.QNetworkRequest.HttpStatusCodeAttribute)
        self._set_loading(False)

        try:
            data = json.loads(data_bytes.decode("utf-8") or "{}")
            print("STATUS", status)
            print("DATA", data)
        except Exception:
            data = {}

        if status == 200:
            # Emitimos datos y limpiamos
            self.msg.setVisible(False)
            self.loginSuccess.emit(data) # ACA SE EMITE LA DATA A LA VENTANA PRINCIPAL
        else:
            # Mostramos mensaje devuelto por la API si existe
            api_msg = data.get("message") or data.get("error") or f"HTTP {status}"
            self._set_error(f"Login fallido: {api_msg}")

        reply.deleteLater()

    def _set_error(self, text: str):
        self.msg.setText(text)
        self.msg.setVisible(bool(text))

    def _set_loading(self, is_loading: bool):
        self.btn_login.setDisabled(is_loading)
        self.btn_login.setText("Ingresando…" if is_loading else "Ingresar")
        if is_loading:
            self._start_spinner()
        else:
            self._stop_spinner()
            self.btn_login.setEnabled(True)
