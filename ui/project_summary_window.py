import json

from PySide6.QtWidgets import (
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.project_summary import get_project_summary


class ProjectSummaryWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Project Summary")

        self.refresh_button = QPushButton("Refresh Summary")
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.refresh_button.clicked.connect(self.refresh)

        layout = QVBoxLayout()
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.output)

        self.setLayout(layout)

        self.refresh()

    def refresh(self):

        summary = get_project_summary()

        self.output.setPlainText(json.dumps(summary, indent=2))
