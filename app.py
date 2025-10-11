import sys
from PyQt5 import QtCore, QtWidgets
from constants import THEME
from screens.main_screen import MainWindow

def main():
    # Alta-DPI para que todo se vea nítido
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle("Fusion")
    app.setStyleSheet(THEME["qss"]["base"])
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
