from PyQt5 import QtCore, QtGui, QtWidgets
from constants import APP, THEME
# from screens.login_screen import LoginScreen
from screens.capture_screen import CaptureScreen

class MainWindow(QtWidgets.QMainWindow):
    """Orquesta pantallas y almacena session_data tras login."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP["TITLE"])
        self.resize(APP["WIDTH"], APP["HEIGHT"])
        self.setMinimumSize(720, 480)

        # Redondeo visual (opcional, funciona mejor en mac/win)
        # Nota: para recorte real necesitarías ventanas sin marco; aquí es cosmético
        path = QtGui.QPainterPath()
        rect = QtCore.QRectF(0, 0, self.width(), self.height())
        path.addRoundedRect(rect, APP["WINDOW_ROUND_RADIUS"], APP["WINDOW_ROUND_RADIUS"])

        # Pila de pantallas
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self._fade_effect = QtWidgets.QGraphicsOpacityEffect(self.stack)
        self._fade_effect.setOpacity(1.0)
        self.stack.setGraphicsEffect(self._fade_effect)
        self._fade_anim = QtCore.QPropertyAnimation(self._fade_effect, b"opacity", self)
        self._fade_anim.setDuration(THEME["anim"]["fade_ms"])
        self._fade_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim_guard = []  # evita GC de animaciones locales


        # self.login = LoginScreen()
        self.capture = CaptureScreen()

        # self.stack.addWidget(self.login)
        self.stack.addWidget(self.capture)

        self.session_data = None

        # self.login.loginSuccess.connect(self._on_login_ok)

        

        # Ajustes de estilo global
        self.setStyleSheet(THEME["qss"]["base"])

    # def _on_login_ok(self, data: dict):
    #     # Guarda JSON de sesión y navega
    #     print("DATA in MAIN", data)
    #     self.session_data = data or {}

    #     print("session_data in MAIN", self.session_data)
    #     # self.stack.setCurrentWidget(self.capture)
    #     self.capture.set_session_data(self.session_data)
    #     self._crossfade_to(self.capture)

    def closeEvent(self, e: QtGui.QCloseEvent):
        # Limpieza adicional si hiciera falta
        return super().closeEvent(e)
    
    def _crossfade_to(self, widget):
        # fade out
        anim_out = QtCore.QPropertyAnimation(self._fade_effect, b"opacity", self)
        anim_out.setDuration(THEME["anim"]["fade_ms"])
        anim_out.setStartValue(1.0)
        anim_out.setEndValue(0.0)
        anim_out.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        def _switch():
            self.stack.setCurrentWidget(widget)
            # fade in
            anim_in = QtCore.QPropertyAnimation(self._fade_effect, b"opacity", self)
            anim_in.setDuration(THEME["anim"]["fade_ms"])
            anim_in.setStartValue(0.0)
            anim_in.setEndValue(1.0)
            anim_in.setEasingCurve(QtCore.QEasingCurve.OutCubic)
            self._anim_guard[:] = [anim_in]
            anim_in.start()

        anim_out.finished.connect(_switch)
        self._anim_guard[:] = [anim_out]
        anim_out.start()
