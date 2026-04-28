from __future__ import annotations

from config.settings_manager import load_settings
from notifications.telegram_notifier import TelegramNotifier


class ValueBetNotifier:

    def __init__(self):

        settings = load_settings()

        self.notifier = TelegramNotifier(
            settings.telegram_token,
            settings.telegram_chat_id,
        )

    def notify(self, value_bets):

        if not value_bets:
            return False

        lines = []

        for vb in value_bets[:10]:
            lines.append(
                f"{vb.get('match_id')} | "
                f"{vb.get('market')} | "
                f"prob={vb.get('probability'):.3f} | "
                f"odds={vb.get('odds')}"
            )

        message = "Value Bets Found:\n" + "\n".join(lines)

        return self.notifier.send_message(message)
