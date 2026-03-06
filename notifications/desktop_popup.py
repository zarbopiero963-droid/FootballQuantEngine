class DesktopPopup:

    def notify(self, title, message):

        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance()

            if app is None:
                return False

            QMessageBox.information(None, title, message)

            return True

        except Exception:
            return False
