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
        self.telegram_token_input = QLineEdit()
        self.telegram_chat_id_input = QLineEdit()
        self.league_id_input = QLineEdit()
        self.league_id_input.setPlaceholderText(
            "e.g. 135 = Serie A, 39 = Premier League"
        )
        self.season_input = QLineEdit()
        self.season_input.setPlaceholderText("e.g. 2024")
        self.openweather_input = QLineEdit()
        self.openweather_input.setPlaceholderText(
            "Optional — openweathermap.org free key"
        )

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)

        form = QFormLayout()
        form.addRow("API-Football Key", self.api_football_input)
        form.addRow("Telegram Token", self.telegram_token_input)
        form.addRow("Telegram Chat ID", self.telegram_chat_id_input)
        form.addRow("League ID (numeric)", self.league_id_input)
        form.addRow("Season", self.season_input)
        form.addRow("OpenWeather API Key", self.openweather_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.load_values()

    def load_values(self):

        settings = load_settings()

        self.api_football_input.setText(settings.api_football_key)
        self.telegram_token_input.setText(settings.telegram_token)
        self.telegram_chat_id_input.setText(settings.telegram_chat_id)
        self.league_id_input.setText(str(settings.league_id))
        self.season_input.setText(str(settings.season))
        self.openweather_input.setText(settings.openweather_key)

    def save(self):

        settings = Settings()
        settings.api_football_key = self.api_football_input.text().strip()
        settings.telegram_token = self.telegram_token_input.text().strip()
        settings.telegram_chat_id = self.telegram_chat_id_input.text().strip()
        settings.openweather_key = self.openweather_input.text().strip()

        try:
            settings.league_id = int(self.league_id_input.text().strip() or 135)
            settings.season = int(self.season_input.text().strip() or 2024)
        except ValueError:
            QMessageBox.warning(
                self, "Settings", "League ID and Season must be integers."
            )
            return

        save_settings(settings)

        QMessageBox.information(
            self,
            "Settings",
            "Settings saved successfully.",
        )
