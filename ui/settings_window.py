from PySide6.QtWidgets import (
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.settings_manager import Settings, load_settings, save_settings


class SettingsWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Settings")

        self.api_football_input = QLineEdit()
        self.odds_api_input = QLineEdit()
        self.telegram_token_input = QLineEdit()
        self.telegram_chat_id_input = QLineEdit()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)

        form = QFormLayout()
        form.addRow("API-Football Key", self.api_football_input)
        form.addRow("Odds API Key", self.odds_api_input)
        form.addRow("Telegram Token", self.telegram_token_input)
        form.addRow("Telegram Chat ID", self.telegram_chat_id_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.load_values()

    def load_values(self):

        settings = load_settings()

        self.api_football_input.setText(settings.api_football_key)
        self.odds_api_input.setText(settings.odds_api_key)
        self.telegram_token_input.setText(settings.telegram_token)
        self.telegram_chat_id_input.setText(settings.telegram_chat_id)

    def save(self):

        settings = Settings()
        settings.api_football_key = self.api_football_input.text().strip()
        settings.odds_api_key = self.odds_api_input.text().strip()
        settings.telegram_token = self.telegram_token_input.text().strip()
        settings.telegram_chat_id = self.telegram_chat_id_input.text().strip()

        save_settings(settings)

        QMessageBox.information(
            self,
            "Settings",
            "Settings saved successfully.",
        )
