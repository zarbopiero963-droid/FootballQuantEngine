"""
Telegram notification client for Football Quant Engine.

Features
--------
- Rate limiting  : token-bucket at 28 msg/min (Telegram hard limit is 30)
- Retry logic    : up to 3 attempts with exponential back-off (2 s, 4 s, 8 s)
- Long messages  : automatically chunked at 4096 chars (Telegram max)
- HTML parse mode: bold, italic, inline code, hyperlinks via HTML tags
- Value-bet alert: rich multi-bet alert with tier emoji, EV, Kelly, odds
- sendDocument   : attach CSV/Excel files (uses multipart upload)
- sendPhoto      : attach PNG charts
- Silence hours  : configurable quiet window (e.g. 23:00-07:00)
- Logging        : all sends, retries and failures go to Python logger
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
_MAX_MSG_LEN = 4096
_RATE_LIMIT_PER_MIN = 28
_MIN_INTERVAL = 60.0 / _RATE_LIMIT_PER_MIN
_MAX_RETRIES = 3
_RETRY_BASE = 2.0
_REQUEST_TIMEOUT = 15

_TIER_EMOJI = {
    "S": "\U0001f7e2",
    "A": "\U0001f535",
    "B": "\U0001f7e1",
    "C": "⚪",
    "X": "\U0001f534",
}


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    def __init__(self, capacity: float, refill_rate: float) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def consume(self, block: bool = True) -> bool:
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            if not block:
                return False

            wait = (1.0 - self._tokens) / self._refill_rate
            time.sleep(max(0.05, wait))


# ---------------------------------------------------------------------------
# Silence window
# ---------------------------------------------------------------------------


def _in_silence_window(start: Optional[int], end: Optional[int]) -> bool:
    if start is None or end is None:
        return False
    h = datetime.now().hour
    if start < end:
        return start <= h < end
    return h >= start or h < end


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TelegramNotifier:
    """
    Production-grade Telegram notification client.

    Parameters
    ----------
    token         : Telegram Bot API token
    chat_id       : target chat / channel / group id
    parse_mode    : "HTML" (default) or "Markdown"
    silence_start : start hour of quiet window (None = no silence)
    silence_end   : end hour of quiet window
    dry_run       : if True, log messages instead of sending
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        parse_mode: str = "HTML",
        silence_start: Optional[int] = None,
        silence_end: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        self._token = token or ""
        self._chat_id = str(chat_id or "")
        self._parse = parse_mode
        self._sil_s = silence_start
        self._sil_e = silence_end
        self._dry_run = dry_run
        self._bucket = _TokenBucket(
            capacity=_RATE_LIMIT_PER_MIN,
            refill_rate=_RATE_LIMIT_PER_MIN / 60.0,
        )
        self.sent_count = 0
        self.failed_count = 0
        self.retried_count = 0

    # ------------------------------------------------------------------
    # Core: send text
    # ------------------------------------------------------------------

    def send_message(
        self,
        text: str,
        disable_notification: bool = False,
        disable_preview: bool = True,
    ) -> bool:
        """Send text message; long messages are chunked automatically."""
        if not self._token or not self._chat_id:
            logger.warning("TelegramNotifier: token/chat_id not configured.")
            return False

        if _in_silence_window(self._sil_s, self._sil_e):
            logger.debug("TelegramNotifier: in silence window, skipping.")
            return True

        if self._dry_run:
            logger.info("[DRY RUN] Telegram message:\n%s", text)
            return True

        success = True
        for chunk in self._chunk(text):
            url = _TELEGRAM_API.format(token=self._token, method="sendMessage")
            payload = {
                "chat_id": self._chat_id,
                "text": chunk,
                "parse_mode": self._parse,
                "disable_notification": disable_notification,
                "disable_web_page_preview": disable_preview,
            }
            if not self._post_with_retry(url, json=payload):
                success = False
        return success

    # ------------------------------------------------------------------
    # Rich alerts
    # ------------------------------------------------------------------

    def send_value_bets_alert(
        self,
        bets: list[dict],
        title: str = "Value Bets Alert",
        max_bets: int = 10,
    ) -> bool:
        """Format and send a structured value-bets alert."""
        if not bets:
            return self.send_message("<b>No value bets found for this cycle.</b>")

        lines: list[str] = [f"<b>{title}</b>", ""]
        for i, b in enumerate(bets[:max_bets], start=1):
            tier = str(b.get("tier", "C"))
            emoji = _TIER_EMOJI.get(tier, "⚪")
            match = str(b.get("match_id", "?"))
            market = str(b.get("market", "?")).upper()
            prob = float(b.get("probability", 0))
            odds = float(b.get("odds", 0))
            ev = float(b.get("ev", 0))
            kelly = float(b.get("kelly", 0))
            stake = float(b.get("kelly_stake", 0))
            conf = float(b.get("confidence", 0))

            lines += [
                f"{emoji} <b>#{i} [{tier}]</b> {match}",
                f"   Market: <code>{market}</code> | "
                f"Prob: <b>{prob:.1%}</b> | Odds: <b>{odds:.2f}</b>",
                f"   EV: <b>{ev:+.3f}</b> | "
                f"Kelly: {kelly:.2%} | Stake: <b>€{stake:.2f}</b>",
                f"   Confidence: {conf:.1%}",
                "",
            ]
        return self.send_message("\n".join(lines))

    def send_cycle_summary(
        self,
        n_processed: int,
        n_bets: int,
        n_watchlist: int,
        elapsed_sec: float,
        top_match: Optional[str] = None,
    ) -> bool:
        """Send a compact cycle completion summary."""
        emoji = "✅" if n_bets > 0 else "❌"
        lines = [
            f"{emoji} <b>Cycle Complete</b>",
            f"Processed: <b>{n_processed}</b> matches",
            f"Value Bets: <b>{n_bets}</b>  |  Watchlist: <b>{n_watchlist}</b>",
            f"Duration: {elapsed_sec:.1f}s",
        ]
        if top_match:
            lines.append(f"Top pick: <b>{top_match}</b>")
        return self.send_message("\n".join(lines))

    def send_steam_alert(
        self,
        fixture_id: str,
        market: str,
        magnitude: float,
        source: str,
    ) -> bool:
        """Send a steam-move market alert."""
        text = (
            "\U0001f525 <b>Steam Move Detected</b>\n"
            f"Fixture: <code>{fixture_id}</code>\n"
            f"Market:  <b>{market.upper()}</b>\n"
            f"Move:    <b>+{magnitude:.2%}</b> implied prob\n"
            f"Source:  {source}"
        )
        return self.send_message(text, disable_notification=False)

    def send_daily_report_alert(
        self,
        metrics: dict,
        report_path: Optional[str] = None,
    ) -> bool:
        """Send a daily performance summary."""
        roi = float(metrics.get("roi", 0))
        yield_ = float(metrics.get("yield", 0))
        bets = int(metrics.get("total_bets", 0))
        hit = float(metrics.get("hit_rate", 0))
        profit = float(metrics.get("total_profit", 0))
        sign = "\U0001f4c8" if roi >= 0 else "\U0001f4c9"

        lines = [
            f"{sign} <b>Daily Performance Report</b>",
            f"ROI:      <b>{roi:.2%}</b>",
            f"Yield:    <b>{yield_:.2%}</b>",
            f"Hit Rate: <b>{hit:.1%}</b>",
            f"Bets:     <b>{bets}</b>",
            f"Profit:   <b>€{profit:+.2f}</b>",
        ]
        if report_path:
            lines.append(f"\nReport: <code>{report_path}</code>")
        return self.send_message("\n".join(lines))

    # ------------------------------------------------------------------
    # File / photo uploads
    # ------------------------------------------------------------------

    def send_document(
        self,
        filepath: str,
        caption: str = "",
        filename: Optional[str] = None,
    ) -> bool:
        """Send a file attachment (CSV, Excel, …)."""
        if not self._token or not self._chat_id:
            return False
        if _in_silence_window(self._sil_s, self._sil_e):
            return True
        if self._dry_run:
            logger.info("[DRY RUN] Telegram sendDocument: %s", filepath)
            return True

        path = Path(filepath)
        if not path.exists():
            logger.warning("TelegramNotifier.send_document: not found: %s", filepath)
            return False

        url = _TELEGRAM_API.format(token=self._token, method="sendDocument")
        fname = filename or path.name
        data = {
            "chat_id": self._chat_id,
            "caption": caption[:1024],
            "parse_mode": self._parse,
        }
        with open(path, "rb") as fh:
            return self._post_with_retry(
                url, data=data, files={"document": (fname, fh)}
            )

    def send_bytes_document(
        self,
        content: bytes,
        filename: str,
        caption: str = "",
    ) -> bool:
        """Send in-memory bytes as a document attachment."""
        if not self._token or not self._chat_id:
            return False
        if self._dry_run:
            logger.info("[DRY RUN] Telegram sendDocument (bytes): %s", filename)
            return True

        url = _TELEGRAM_API.format(token=self._token, method="sendDocument")
        data = {
            "chat_id": self._chat_id,
            "caption": caption[:1024],
            "parse_mode": self._parse,
        }
        return self._post_with_retry(
            url,
            data=data,
            files={"document": (filename, content, "application/octet-stream")},
        )

    def send_photo(self, filepath: str, caption: str = "") -> bool:
        """Send a PNG/JPEG photo."""
        if not self._token or not self._chat_id:
            return False
        if self._dry_run:
            logger.info("[DRY RUN] Telegram sendPhoto: %s", filepath)
            return True

        path = Path(filepath)
        if not path.exists():
            return False

        url = _TELEGRAM_API.format(token=self._token, method="sendPhoto")
        data = {"chat_id": self._chat_id, "caption": caption[:1024]}
        with open(path, "rb") as fh:
            return self._post_with_retry(url, data=data, files={"photo": fh})

    # ------------------------------------------------------------------
    # HTTP with retry + rate limit
    # ------------------------------------------------------------------

    def _safe_url(self, url: str) -> str:
        """Return url with the bot token replaced by *** for safe logging."""
        return url.replace(self._token, "***") if self._token else url

    def _post_with_retry(self, url: str, **kwargs: Any) -> bool:
        self._bucket.consume(block=True)
        safe = self._safe_url(url)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(url, timeout=_REQUEST_TIMEOUT, **kwargs)

                if resp.status_code == 200:
                    self.sent_count += 1
                    return True

                if resp.status_code == 429:
                    retry_after = float(
                        resp.json().get("parameters", {}).get("retry_after", 5)
                    )
                    logger.warning(
                        "TelegramNotifier: 429 — waiting %.0fs.", retry_after
                    )
                    time.sleep(retry_after)
                    self.retried_count += 1
                    continue

                if 400 <= resp.status_code < 500:
                    logger.error(
                        "TelegramNotifier: client error %d on %s — %s",
                        resp.status_code,
                        safe,
                        resp.text[:200],
                    )
                    self.failed_count += 1
                    return False

                logger.warning(
                    "TelegramNotifier: server error %d (attempt %d/%d) on %s.",
                    resp.status_code,
                    attempt,
                    _MAX_RETRIES,
                    safe,
                )

            except requests.exceptions.Timeout:
                logger.warning(
                    "TelegramNotifier: timeout on %s (attempt %d/%d).",
                    safe,
                    attempt,
                    _MAX_RETRIES,
                )
            except Exception as exc:
                # Mask token in exc string — requests errors often embed the URL
                safe_exc = str(exc).replace(self._token, "***") if self._token else str(exc)
                logger.warning(
                    "TelegramNotifier: exception (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    safe_exc,
                )

            if attempt < _MAX_RETRIES:
                self.retried_count += 1
                time.sleep(_RETRY_BASE**attempt)

        self.failed_count += 1
        logger.error("TelegramNotifier: all %d attempts failed.", _MAX_RETRIES)
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk(text: str, size: int = _MAX_MSG_LEN) -> list[str]:
        if len(text) <= size:
            return [text]
        chunks: list[str] = []
        while text:
            if len(text) <= size:
                chunks.append(text)
                break
            cut = text.rfind("\n", 0, size)
            if cut == -1:
                cut = size
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def stats(self) -> dict:
        return {
            "sent": self.sent_count,
            "failed": self.failed_count,
            "retried": self.retried_count,
        }
