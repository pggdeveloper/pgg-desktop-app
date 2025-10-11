
from PyQt5 import QtCore, QtGui, QtWidgets
from constants import THEME

class AnimatedButton(QtWidgets.QPushButton):
    """Botón con animaciones de opacidad y sombra al hover/pressed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setObjectName("PrimaryButton")
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumWidth(180)
        self.setMinimumHeight(THEME["sizes"]["btn_height"])
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)

        # Opacidad
        self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0) # THEME["anim"]["opacity_rest"])
        

        # self._opacity_anim = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity", self)
        # self._opacity_anim.setDuration(THEME["anim"]["hover_ms"])
        # self._opacity_anim.setEasingCurve(THEME["anim"]["ease"])

        # Sombra
        # self._shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        # self._shadow.setBlurRadius(THEME["anim"]["shadow_rest_blur"])
        # self._shadow.setXOffset(0)
        # self._shadow.setYOffset(THEME["anim"]["shadow_rest_y"])
        # self._shadow.setColor(QtGui.QColor(THEME["colors"]["shadow"]))
        # self.setGraphicsEffect(self._shadow)

        # self._shadow_blur_anim = QtCore.QPropertyAnimation(self._shadow, b"blurRadius", self)
        # self._shadow_blur_anim.setDuration(THEME["anim"]["hover_ms"])
        # self._shadow_blur_anim.setEasingCurve(THEME["anim"]["ease"])

        # self._shadow_y_anim = QtCore.QPropertyAnimation(self._shadow, b"yOffset", self)
        # self._shadow_y_anim.setDuration(THEME["anim"]["hover_ms"])
        # self._shadow_y_anim.setEasingCurve(THEME["anim"]["ease"])

        # Efecto "press": pequeño bounce con opacidad

        # self._press_anim = QtCore.QParallelAnimationGroup(self)
        # press_opacity = QtCore.QPropertyAnimation(self._opacity_effect, b"opacity", self)
        # press_opacity.setDuration(THEME["anim"]["press_ms"])
        # press_opacity.setStartValue(THEME["anim"]["opacity_hover"])
        # press_opacity.setEndValue(THEME["anim"]["opacity_rest"])
        # press_opacity.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        # self._press_anim.addAnimation(press_opacity)

        # self.installEventFilter(self)
        self.setGraphicsEffect(self._opacity_effect)

    # def eventFilter(self, obj, ev):
    #     if obj is self:
    #         if ev.type() == QtCore.QEvent.Enter:
    #             self._animate_hover(True)
    #         elif ev.type() == QtCore.QEvent.Leave:
    #             self._animate_hover(False)
    #         elif ev.type() == QtCore.QEvent.MouseButtonPress:
    #             self._press_anim.start()
    #     return super().eventFilter(obj, ev)

    # def _animate_hover(self, entering: bool):
    #     if entering:
    #         self._opacity_anim.stop()
    #         self._opacity_anim.setStartValue(self._opacity_effect.opacity())
    #         self._opacity_anim.setEndValue(THEME["anim"]["opacity_hover"])
    #         self._opacity_anim.start()

    #         # self._shadow_blur_anim.stop()
    #         # self._shadow_blur_anim.setStartValue(self._shadow.blurRadius())
    #         # self._shadow_blur_anim.setEndValue(THEME["anim"]["shadow_hover_blur"])
    #         # self._shadow_blur_anim.start()

    #         # self._shadow_y_anim.stop()
    #         # self._shadow_y_anim.setStartValue(self._shadow.yOffset())
    #         # self._shadow_y_anim.setEndValue(THEME["anim"]["shadow_hover_y"])
    #         # self._shadow_y_anim.start()
    #     else:
    #         self._opacity_anim.stop()
    #         self._opacity_anim.setStartValue(self._opacity_effect.opacity())
    #         self._opacity_anim.setEndValue(THEME["anim"]["opacity_rest"])
    #         self._opacity_anim.start()

    #         # self._shadow_blur_anim.stop()
    #         # self._shadow_blur_anim.setStartValue(self._shadow.blurRadius())
    #         # self._shadow_blur_anim.setEndValue(THEME["anim"]["shadow_rest_blur"])
    #         # self._shadow_blur_anim.start()

    #         # self._shadow_y_anim.stop()
    #         # self._shadow_y_anim.setStartValue(self._shadow.yOffset())
    #         # self._shadow_y_anim.setEndValue(THEME["anim"]["shadow_rest_y"])
    #         # self._shadow_y_anim.start()
