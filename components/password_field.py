from PyQt5 import QtCore, QtGui, QtWidgets

class PasswordField(QtWidgets.QLineEdit):
    """QLineEdit con botón 'ojito' para alternar visibilidad de contraseña."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEchoMode(QtWidgets.QLineEdit.Password)
        self._toggle_btn = QtWidgets.QToolButton(self)
        self._toggle_btn.setObjectName("IconInline")
        self._toggle_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._toggle_btn.setToolTip("Mostrar/Ocultar contraseña")
        self._toggle_btn.setText("👁")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._toggle_btn.clicked.connect(self._toggle_echo_mode)

        # Ajuste interno para dejar espacio al botón
        m = self.textMargins()
        self.setTextMargins(m.left(), m.top(), 28, m.bottom())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        sz = self._toggle_btn.sizeHint()
        # Posiciona el ojito dentro del input, a la derecha
        self._toggle_btn.setGeometry(self.rect().right() - sz.width() - 4,
                                     (self.rect().height() - sz.height()) // 2,
                                     sz.width(), sz.height())

    def _toggle_echo_mode(self, checked):
        self.setEchoMode(QtWidgets.QLineEdit.Normal if checked else QtWidgets.QLineEdit.Password)
        self._toggle_btn.setText("🙈" if checked else "👁")
