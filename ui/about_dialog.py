from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class AboutDialog(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("About")

        label = QLabel(
            "Football Quant Engine\n"
            "Quantitative football analysis engine\n"
            "Poisson + Monte Carlo + Value Betting"
        )

        layout = QVBoxLayout()
        layout.addWidget(label)

        self.setLayout(layout)
