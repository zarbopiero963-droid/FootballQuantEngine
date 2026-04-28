"""
Tests for notifications.telegram_notifier — all network calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from notifications.telegram_notifier import TelegramNotifier


@pytest.fixture
def notifier():
    return TelegramNotifier(token="test_token_123", chat_id="999")


class TestSafeUrl:
    def test_masks_token(self, notifier):
        url = "https://api.telegram.org/bottest_token_123/sendMessage"
        masked = notifier._safe_url(url)
        assert "test_token_123" not in masked
        assert "***" in masked

    def test_empty_token_returns_url_unchanged(self):
        n = TelegramNotifier(token="", chat_id="123")
        url = "https://api.telegram.org/sendMessage"
        assert n._safe_url(url) == url


class TestSendMessage:
    def test_send_returns_true_on_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        with patch("requests.post", return_value=mock_resp):
            result = notifier.send_message("Hello")
        assert result is True

    def test_send_returns_false_on_network_error(self, notifier):
        with patch("requests.post", side_effect=ConnectionError("Network unreachable")):
            result = notifier.send_message("Hello")
        assert result is False


class TestSendValueBetsAlert:
    def test_calls_send_message(self, notifier, sample_ranked_bets):
        with patch.object(notifier, "send_message", return_value=True) as mock_send:
            notifier.send_value_bets_alert(sample_ranked_bets)
        assert mock_send.called

    def test_empty_bets_does_not_crash(self, notifier):
        with patch.object(notifier, "send_message", return_value=True):
            notifier.send_value_bets_alert([])


class TestSendCycleSummary:
    def test_calls_send_message(self, notifier):
        with patch.object(notifier, "send_message", return_value=True) as mock_send:
            notifier.send_cycle_summary(
                n_processed=10, n_bets=3, n_watchlist=2, elapsed_sec=4.5
            )
        assert mock_send.called
