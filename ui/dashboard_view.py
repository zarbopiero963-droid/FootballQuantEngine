from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class DashboardView(QWidget):

    def __init__(self):

        super().__init__()

        self.title = QLabel("Football Quant Engine Dashboard")
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def set_text(self, text):

        self.output.setPlainText(text)
