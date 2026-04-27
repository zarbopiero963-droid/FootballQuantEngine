from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config.settings_manager import Settings, load_settings, save_settings
from quant.providers.league_registry import all_known


def _build_league_items() -> list[tuple[int, str]]:
    """Return sorted list of (league_id, display_label) for the combo box."""
    return sorted(
        [(lid, f"{name}  [{lid}]") for lid, name in all_known().items()],
        key=lambda x: x[1],
    )


class SettingsWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")

        self.api_football_input = QLineEdit()
        self.telegram_token_input = QLineEdit()
        self.telegram_chat_id_input = QLineEdit()
        self.season_input = QLineEdit()
        self.season_input.setPlaceholderText("e.g. 2024")
        self.openweather_input = QLineEdit()
        self.openweather_input.setPlaceholderText("Optional — openweathermap.org free key")

        # League selector — shows names, stores IDs
        self.league_combo = QComboBox()
        self.league_combo.setEditable(True)
        self.league_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        if hasattr(self.league_combo, "completer") and self.league_combo.completer():
            self.league_combo.completer().setFilterMode(
                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchContains
            )

        self._league_items = _build_league_items()
        for _lid, label in self._league_items:
            self.league_combo.addItem(label)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save)

        form = QFormLayout()
        form.addRow("API-Football Key", self.api_football_input)
        form.addRow("Telegram Token", self.telegram_token_input)
        form.addRow("Telegram Chat ID", self.telegram_chat_id_input)
        form.addRow("League", self.league_combo)
        form.addRow("Season", self.season_input)
        form.addRow("OpenWeather API Key", self.openweather_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.load_values()

    # ------------------------------------------------------------------

    def _select_league(self, league_id: int) -> None:
        for i, (lid, _) in enumerate(self._league_items):
            if lid == league_id:
                self.league_combo.setCurrentIndex(i)
                return
        # ID not in static list — add a temporary entry
        from quant.providers.league_registry import name as league_name
        label = f"{league_name(league_id)}  [{league_id}]"
        self.league_combo.insertItem(0, label)
        self._league_items.insert(0, (league_id, label))
        self.league_combo.setCurrentIndex(0)

    def _selected_league_id(self) -> int:
        idx = self.league_combo.currentIndex()
        if 0 <= idx < len(self._league_items):
            return self._league_items[idx][0]
        # User typed a raw ID
        try:
            return int(self.league_combo.currentText().strip().split("[")[-1].rstrip("]"))
        except (ValueError, IndexError):
            return 135

    # ------------------------------------------------------------------

    def load_values(self):
        settings = load_settings()
        self.api_football_input.setText(settings.api_football_key)
        self.telegram_token_input.setText(settings.telegram_token)
        self.telegram_chat_id_input.setText(settings.telegram_chat_id)
        self._select_league(settings.league_id)
        self.season_input.setText(str(settings.season))
        self.openweather_input.setText(settings.openweather_key)

    def save(self):
        settings = Settings()
        settings.api_football_key = self.api_football_input.text().strip()
        settings.telegram_token = self.telegram_token_input.text().strip()
        settings.telegram_chat_id = self.telegram_chat_id_input.text().strip()
        settings.openweather_key = self.openweather_input.text().strip()
        settings.league_id = self._selected_league_id()

        try:
            settings.season = int(self.season_input.text().strip() or 2024)
        except ValueError:
            QMessageBox.warning(self, "Settings", "Season must be a number.")
            return

        save_settings(settings)
        QMessageBox.information(self, "Settings", "Settings saved successfully.")
