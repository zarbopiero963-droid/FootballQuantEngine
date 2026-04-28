from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.final_check import run_final_check


class FinalCheckWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Final Project Check")

        self.run_button = QPushButton("Run Final Check")
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.run_button.clicked.connect(self.run_check)

        layout = QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def run_check(self):

        result = run_final_check()

        self.output.setPlainText(json.dumps(result, indent=2))
