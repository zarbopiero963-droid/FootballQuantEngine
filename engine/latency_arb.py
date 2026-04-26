"""
Latency Arbitrage Engine — High-Frequency Live Betting.

Stadium data feeds (Opta, Radar, Stats Perform) deliver events ~500-800ms
before bookmakers suspend their markets.  This engine monitors two streams
concurrently: a fast "truth" feed and a bookmaker state feed.  When a goal
or red card is detected on the fast feed, a BetInstruction is fired to the
slow bookmaker before market suspension closes the window.

Architecture
------------
  FeedSimulator     — async generator that replays event sequences
  BookmakerMonitor  — async polling loop tracking odds + suspension state
  LatencyArbEngine  — coordinates the two streams, emits BetInstruction
  LatencyProfiler   — rolling latency statistics

Usage
-----
    import asyncio
    from engine.latency_arb import run_latency_arb

    session = asyncio.run(run_latency_arb(
        fixture_id=123,
        feed_events=[
            {"event_type": "GOAL", "team": "home", "minute": 34,
             "home_score": 1, "away_score": 0, "delay_ms": 1000},
        ],
        pre_match_home_prob=0.50,
        pre_match_away_prob=0.25,
        stake=10.0,
    ))
    print(session.bets_fired, session.avg_latency_ms)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

try:
    import aiohttp  # type: ignore[import]
    import websockets  # type: ignore[import]

    _ASYNC_LIBS = True
except Exception:
    aiohttp = None  # type: ignore[assignment]
    websockets = None  # type: ignore[assignment]
    _ASYNC_LIBS = False

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_MS = 1500
_MIN_ODDS_CHANGE = 0.05
_DEFAULT_STAKE = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LiveFeedEvent:
    """An event from the fast data feed."""

    event_type: (
        str  # "GOAL" | "RED_CARD" | "PENALTY" | "KICKOFF" | "HALFTIME" | "FULLTIME"
    )
    team: str  # "home" | "away"
    minute: int
    home_score: int
    away_score: int
    timestamp_ms: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BookmakerState:
    """Current state of a bookmaker live market."""

    bookmaker: str
    fixture_id: int
    home_odds: float
    draw_odds: float
    away_odds: float
    suspended: bool
    last_update_ms: float


@dataclass
class LatencyMeasurement:
    """Measured latency between fast feed and bookmaker update."""

    event_type: str
    fast_feed_ts_ms: float
    bookmaker_ts_ms: float
    latency_ms: float


@dataclass
class BetInstruction:
    """Auto-generated bet instruction fired before market suspension."""

    fixture_id: int
    bookmaker: str
    market: str  # "home" | "draw" | "away"
    odds: float
    stake: float
    fired_at_ms: float
    event: LiveFeedEvent
    rationale: str

    def __str__(self) -> str:
        return (
            f"BET [{self.bookmaker}] {self.market}@{self.odds:.2f} "
            f"£{self.stake:.2f} fired={self.fired_at_ms:.0f}ms "
            f"| {self.rationale}"
        )


@dataclass
class LatencyArbSession:
    """Session statistics for one latency arb run."""

    fixture_id: int
    events_received: int = 0
    bets_fired: int = 0
    total_latency_measurements: int = 0
    avg_latency_ms: float = 0.0
    successful_windows: int = 0
    instructions: List[BetInstruction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feed simulator
# ---------------------------------------------------------------------------


class FeedSimulator:
    """
    Simulates a fast stadium data feed by replaying a list of events.

    Each event dict may include ``delay_ms`` (how long after session start
    to emit it).  If absent, events are spaced 30 seconds apart.
    """

    def __init__(
        self,
        events: List[dict],
        base_latency_ms: float = 500.0,
    ) -> None:
        self._events = events
        self._base_latency_ms = base_latency_ms

    async def stream(self) -> AsyncIterator[LiveFeedEvent]:
        """Yield LiveFeedEvents at their scheduled times."""
        start_ms = _now_ms()
        for raw in self._events:
            delay_ms = float(raw.get("delay_ms", 30_000))
            target_ms = start_ms + delay_ms
            wait_s = max(0.0, (target_ms - _now_ms()) / 1000.0)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            ts = _now_ms()
            event = LiveFeedEvent(
                event_type=str(raw.get("event_type", "UNKNOWN")),
                team=str(raw.get("team", "home")),
                minute=int(raw.get("minute", 0)),
                home_score=int(raw.get("home_score", 0)),
                away_score=int(raw.get("away_score", 0)),
                timestamp_ms=ts,
                raw=raw,
            )
            logger.debug("Feed event: %s min=%d", event.event_type, event.minute)
            yield event


# ---------------------------------------------------------------------------
# Bookmaker monitor
# ---------------------------------------------------------------------------


class BookmakerMonitor:
    """
    Async polling loop that tracks a bookmaker's live odds and suspension.

    If a real ``fetch_fn`` is provided it is called each poll cycle.
    Otherwise the monitor uses a simulated suspension with configurable
    lag (``suspension_lag_ms`` after a known event time).
    """

    def __init__(
        self,
        bookmaker: str,
        fixture_id: int,
        poll_interval_ms: float = 200.0,
        fetch_fn: Optional[Callable[[], Any]] = None,
        suspension_lag_ms: float = 1200.0,
    ) -> None:
        self._bookmaker = bookmaker
        self._fixture_id = fixture_id
        self._poll_ms = poll_interval_ms
        self._fetch_fn = fetch_fn
        self._lag_ms = suspension_lag_ms
        self._state: Optional[BookmakerState] = BookmakerState(
            bookmaker=bookmaker,
            fixture_id=fixture_id,
            home_odds=2.00,
            draw_odds=3.40,
            away_odds=3.50,
            suspended=False,
            last_update_ms=_now_ms(),
        )
        self._suspend_at_ms: Optional[float] = None
        self._running = False

    async def start(self) -> None:
        """Begin polling loop (runs until stop() called)."""
        self._running = True
        while self._running:
            await asyncio.sleep(self._poll_ms / 1000.0)
            if self._fetch_fn is not None:
                try:
                    result = await self._fetch_fn()
                    if isinstance(result, BookmakerState):
                        self._state = result
                except Exception as exc:
                    logger.warning("fetch_fn error: %s", exc)
            else:
                # Simulated: suspend after lag
                if (
                    self._suspend_at_ms is not None
                    and _now_ms() >= self._suspend_at_ms
                    and self._state is not None
                ):
                    self._state = BookmakerState(
                        bookmaker=self._state.bookmaker,
                        fixture_id=self._state.fixture_id,
                        home_odds=self._state.home_odds,
                        draw_odds=self._state.draw_odds,
                        away_odds=self._state.away_odds,
                        suspended=True,
                        last_update_ms=_now_ms(),
                    )

    def stop(self) -> None:
        self._running = False

    def notify_event(self, event_ts_ms: float) -> None:
        """Called when the fast feed detects an event; triggers simulated suspension."""
        self._suspend_at_ms = event_ts_ms + self._lag_ms

    def get_state(self) -> Optional[BookmakerState]:
        return self._state

    def is_suspended(self) -> bool:
        return self._state.suspended if self._state else True

    def latency_since_event(self, event_ts_ms: float) -> float:
        """ms between event timestamp and bookmaker's last update."""
        if self._state is None:
            return 0.0
        return max(0.0, self._state.last_update_ms - event_ts_ms)


# ---------------------------------------------------------------------------
# Latency Arb Engine
# ---------------------------------------------------------------------------


class LatencyArbEngine:
    """
    Coordinates a fast data feed and a bookmaker monitor to detect and exploit
    latency windows around key match events.

    Bet direction logic
    -------------------
    GOAL by home  → back HOME (momentum) or check if UNDER now more likely
    GOAL by away  → back AWAY
    RED_CARD home → back AWAY (home weakened)
    RED_CARD away → back HOME
    PENALTY       → back whichever team benefits (not yet taken = high xG)
    """

    def __init__(
        self,
        fixture_id: int,
        pre_match_home_prob: float = 0.45,
        pre_match_away_prob: float = 0.30,
        stake: float = _DEFAULT_STAKE,
        window_ms: float = _DEFAULT_WINDOW_MS,
        on_instruction: Optional[Callable[[BetInstruction], None]] = None,
    ) -> None:
        self._fixture_id = fixture_id
        self._p_home = max(0.01, pre_match_home_prob)
        self._p_away = max(0.01, pre_match_away_prob)
        self._stake = stake
        self._window_ms = window_ms
        self._on_instruction = on_instruction
        self._profiler = LatencyProfiler()

    async def run(
        self,
        feed: AsyncIterator[LiveFeedEvent],
        monitor: BookmakerMonitor,
    ) -> LatencyArbSession:
        """
        Main coroutine.  Iterates feed events, fires BetInstructions when
        the bookmaker window is still open.
        """
        session = LatencyArbSession(fixture_id=self._fixture_id)

        monitor_task = asyncio.create_task(monitor.start())

        try:
            async for event in feed:
                session.events_received += 1
                instruction = await self._process_event(event, monitor, session)
                if instruction:
                    session.bets_fired += 1
                    session.instructions.append(instruction)
                    if self._on_instruction:
                        try:
                            self._on_instruction(instruction)
                        except Exception as exc:
                            logger.warning("on_instruction callback error: %s", exc)
        finally:
            monitor.stop()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        if self._profiler.n > 0:
            session.avg_latency_ms = self._profiler.avg_ms()
            session.total_latency_measurements = self._profiler.n

        logger.info(
            "LatencyArbSession done: %d events, %d bets, avg_lag=%.0fms",
            session.events_received,
            session.bets_fired,
            session.avg_latency_ms,
        )
        return session

    async def _process_event(
        self,
        event: LiveFeedEvent,
        monitor: BookmakerMonitor,
        session: LatencyArbSession,
    ) -> Optional[BetInstruction]:
        if event.event_type not in ("GOAL", "RED_CARD", "PENALTY"):
            return None

        # Notify monitor of event (triggers simulated suspension after lag)
        monitor.notify_event(event.timestamp_ms)

        # Small yield to allow monitor poll cycle to run
        await asyncio.sleep(0.01)

        state = monitor.get_state()
        if state is None or state.suspended:
            logger.debug("Market already suspended for %s", event.event_type)
            return None

        # Check we are within the latency window
        elapsed_ms = _now_ms() - event.timestamp_ms
        if elapsed_ms > self._window_ms:
            logger.debug("Window expired: %.0fms > %.0fms", elapsed_ms, self._window_ms)
            return None

        session.successful_windows += 1

        direction = self._determine_bet_direction(event)
        odds = {
            "home": state.home_odds,
            "draw": state.draw_odds,
            "away": state.away_odds,
        }.get(direction, state.home_odds)

        rationale = self._rationale(event, direction)
        instruction = BetInstruction(
            fixture_id=self._fixture_id,
            bookmaker=state.bookmaker,
            market=direction,
            odds=odds,
            stake=self._stake,
            fired_at_ms=_now_ms(),
            event=event,
            rationale=rationale,
        )

        # Record latency measurement
        meas = LatencyMeasurement(
            event_type=event.event_type,
            fast_feed_ts_ms=event.timestamp_ms,
            bookmaker_ts_ms=state.last_update_ms,
            latency_ms=max(0.0, state.last_update_ms - event.timestamp_ms),
        )
        self._profiler.record(meas)

        logger.info("BET FIRED: %s", instruction)
        return instruction

    def _determine_bet_direction(self, event: LiveFeedEvent) -> str:
        if event.event_type == "GOAL":
            return event.team  # back the scoring team
        if event.event_type == "RED_CARD":
            # Weakened team: back the other side
            return "away" if event.team == "home" else "home"
        if event.event_type == "PENALTY":
            return event.team  # team taking penalty has high xG
        return "home"

    def _rationale(self, event: LiveFeedEvent, direction: str) -> str:
        if event.event_type == "GOAL":
            return (
                f"Goal by {event.team} at {event.minute}' "
                f"(score {event.home_score}-{event.away_score}) "
                f"— backing {direction} pre-suspension"
            )
        if event.event_type == "RED_CARD":
            return (
                f"Red card for {event.team} at {event.minute}' "
                f"— backing {direction} on man advantage"
            )
        return f"{event.event_type} at {event.minute}' — backing {direction}"


# ---------------------------------------------------------------------------
# Latency profiler
# ---------------------------------------------------------------------------


class LatencyProfiler:
    """Rolling latency statistics over the last N measurements."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._measurements: List[float] = []
        self.n = 0

    def record(self, measurement: LatencyMeasurement) -> None:
        self._measurements.append(measurement.latency_ms)
        if len(self._measurements) > self._window:
            self._measurements.pop(0)
        self.n += 1

    def avg_ms(self) -> float:
        if not self._measurements:
            return 0.0
        return sum(self._measurements) / len(self._measurements)

    def p95_ms(self) -> float:
        if not self._measurements:
            return 0.0
        sorted_ms = sorted(self._measurements)
        idx = max(0, int(len(sorted_ms) * 0.95) - 1)
        return sorted_ms[idx]

    def summary(self) -> str:
        return (
            f"LatencyProfiler: n={self.n} "
            f"avg={self.avg_ms():.0f}ms "
            f"p95={self.p95_ms():.0f}ms"
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


async def run_latency_arb(
    fixture_id: int,
    feed_events: List[dict],
    pre_match_home_prob: float = 0.45,
    pre_match_away_prob: float = 0.30,
    stake: float = _DEFAULT_STAKE,
    base_latency_ms: float = 500.0,
    suspension_lag_ms: float = 1200.0,
    on_instruction: Optional[Callable[[BetInstruction], None]] = None,
) -> LatencyArbSession:
    """
    End-to-end simulation using FeedSimulator and a simulated BookmakerMonitor.

    Returns session stats.
    """
    feed = FeedSimulator(feed_events, base_latency_ms=base_latency_ms)
    monitor = BookmakerMonitor(
        bookmaker="simulated",
        fixture_id=fixture_id,
        poll_interval_ms=150.0,
        suspension_lag_ms=suspension_lag_ms,
    )
    engine = LatencyArbEngine(
        fixture_id=fixture_id,
        pre_match_home_prob=pre_match_home_prob,
        pre_match_away_prob=pre_match_away_prob,
        stake=stake,
        window_ms=suspension_lag_ms - 100,
        on_instruction=on_instruction,
    )
    return await engine.run(feed.stream(), monitor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.time() * 1000.0
