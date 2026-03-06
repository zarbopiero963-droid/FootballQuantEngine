import sys

from PySide6.QtWidgets import QApplication

from app.controller import AppController
from ui.main_window import MainWindow


def main():

    app = QApplication(sys.argv)

    controller = AppController()
    window = MainWindow(controller)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
