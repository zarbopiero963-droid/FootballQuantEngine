from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.controller import AppController
from app.version_info import get_version_info
from ui.main_window import MainWindow


def main():

    app = QApplication(sys.argv)

    info = get_version_info()

    app.setApplicationName(info["name"])
    app.setApplicationVersion(info["version"])

    controller = AppController()
    window = MainWindow(controller)
    window.setWindowTitle(f"{info['name']} {info['version']}")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
