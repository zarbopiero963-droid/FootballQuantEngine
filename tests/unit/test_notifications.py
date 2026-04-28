"""
Unit tests for notifications/telegram_notifier.py.

All tests mock requests.post — no real HTTP calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from notifications.telegram_notifier import TelegramNotifier

# ---------------------------------------------------------------------------
# Fixture: patch requests.post
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_post(monkeypatch):
    """Replace requests.post with a MagicMock for the duration of each test."""
    mock = MagicMock()
    monkeypatch.setattr("notifications.telegram_notifier.requests.post", mock)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_send_message_dry_run(monkeypatch):
    """dry_run=True → returns True without making any HTTP call."""
    mock = MagicMock()
    monkeypatch.setattr("notifications.telegram_notifier.requests.post", mock)

    notifier = TelegramNotifier("mytoken", "mychat", dry_run=True)
    result = notifier.send_message("hello")

    assert result is True
    mock.assert_not_called()


def test_send_message_no_config():
    """Empty token/chat_id → returns False immediately (no HTTP call needed)."""
    notifier = TelegramNotifier("", "")
    result = notifier.send_message("hello")
    assert result is False


def test_send_message_success(mock_post):
    """200 response → send_message returns True and sent_count increments."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier("token123", "chat456")
    result = notifier.send_message("test message")

    assert result is True
    assert notifier.sent_count == 1


def test_send_message_retries_on_500(mock_post):
    """500 response → all _MAX_RETRIES exhausted, returns False, failed_count == 1."""
    from notifications.telegram_notifier import _MAX_RETRIES

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_post.return_value = mock_resp

    # Patch time.sleep so the test doesn't actually wait during retry back-off
    import notifications.telegram_notifier as _mod

    original_sleep = _mod.time.sleep
    _mod.time.sleep = lambda _: None

    try:
        notifier = TelegramNotifier("token123", "chat456")
        result = notifier.send_message("test message")
    finally:
        _mod.time.sleep = original_sleep

    assert result is False
    assert notifier.failed_count == 1
    assert mock_post.call_count == _MAX_RETRIES


def test_safe_url_masks_token():
    """_safe_url() replaces the bot token with ***."""
    notifier = TelegramNotifier("MYTOKEN", "chat")
    url = "https://api.telegram.org/botMYTOKEN/sendMessage"
    masked = notifier._safe_url(url)
    assert masked == "https://api.telegram.org/bot***/sendMessage"


def test_send_value_bets_alert_dry_run(monkeypatch):
    """dry_run=True, send_value_bets_alert() returns True without HTTP calls."""
    mock = MagicMock()
    monkeypatch.setattr("notifications.telegram_notifier.requests.post", mock)

    notifier = TelegramNotifier("token", "chat", dry_run=True)
    bets = [
        {
            "match_id": "1001",
            "market": "home",
            "tier": "A",
            "ev": 0.1,
            "kelly": 0.05,
            "decision": "BET",
        }
    ]
    result = notifier.send_value_bets_alert(bets)

    assert result is True
    mock.assert_not_called()


def test_health_returns_dict():
    """stats() returns a dict with the required keys."""
    notifier = TelegramNotifier("token", "chat", dry_run=True)
    health = notifier.stats()

    assert isinstance(health, dict)
    assert "sent" in health
    assert "failed" in health
    assert "retried" in health
