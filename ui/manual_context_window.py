import json
import os

from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ManualContextWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Manual Context")

        self.info_label = QLabel(
            "Edit manual context JSON for lineups, injuries and weather."
        )
        self.editor = QTextEdit()
        self.save_button = QPushButton("Save Context")
        self.save_button.clicked.connect(self.save_context)

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.editor)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.load_context()

    def load_context(self):

        path = "manual_context.json"

        if not os.path.exists(path):
            default_data = {
                "lineups": {},
                "injuries": {},
                "weather": {},
            }
            self.editor.setPlainText(json.dumps(default_data, indent=2))
            return

        with open(path, encoding="utf-8") as f:
            self.editor.setPlainText(f.read())

    def save_context(self):

        try:
            content = self.editor.toPlainText()
            parsed = json.loads(content)

            with open("manual_context.json", "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)

            QMessageBox.information(
                self,
                "Manual Context",
                "manual_context.json saved successfully.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save Error",
                str(exc),
            )
