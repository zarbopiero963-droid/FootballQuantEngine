"""
Market Liquidity & Scalping Engine
=====================================

Detects thin-market conditions and short-term price momentum on Betfair
Exchange to identify scalping opportunities.

IMPORTANT — This engine is a *detection* tool, not a manipulation tool.
It identifies patterns in the order book that suggest other participants are
moving the market, then computes a risk-controlled entry/exit for directional
scalping trades.  It does NOT place, cancel, or spoof orders.

Strategy
--------
A scalping signal is generated when three conditions align:
  1. **Thin market**: total available liquidity < threshold (price impact is large)
  2. **Momentum**: last N ticks show N consecutive moves in the same direction
  3. **Entry window**: spread wider than two fair ticks (room to profit after commission)

Entry: back at current best_back (if price shortening) or lay (if drifting).
Target: entry_price ± 2 ticks.
Stop:   entry_price ∓ 3 ticks.

Betfair tick sizes
------------------
  1.01–2.00 → 0.01 | 2.00–3.00 → 0.02 | 3.00–4.00 → 0.05
  4.00–6.00 → 0.10 | 6.00–10.0 → 0.20 | 10.0–20.0 → 0.50

Usage
-----
    from engine.liquidity_trap import LiquidityTrapEngine, PriceTick

    engine = LiquidityTrapEngine()
    for raw_tick in feed:
        tick = PriceTick(**raw_tick)
        signal = engine.process_tick(tick)
        if signal:
            print(signal)
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Betfair tick ladder helpers
# ---------------------------------------------------------------------------

_TICK_BANDS: List[Tuple[float, float, float]] = [
    # (lower_bound, upper_bound, tick_size)
    (1.01, 2.00, 0.01),
    (2.00, 3.00, 0.02),
    (3.00, 4.00, 0.05),
    (4.00, 6.00, 0.10),
    (6.00, 10.00, 0.20),
    (10.00, 20.00, 0.50),
    (20.00, 50.00, 1.00),
    (50.00, 100.0, 2.00),
    (100.0, 1000.0, 5.00),
]


def betfair_tick_size(price: float) -> float:
    """Minimum Betfair price increment at *price*."""
    for lo, hi, tick in _TICK_BANDS:
        if lo <= price < hi:
            return tick
    return 5.00  # > 1000


def next_price(price: float, direction: str) -> float:
    """Next valid Betfair price one tick up ('UP') or down ('DOWN')."""
    tick = betfair_tick_size(price)
    raw = price + tick if direction == "UP" else price - tick
    # Round to tick precision
    decimals = max(0, -int(math.floor(math.log10(tick))))
    result = round(raw, decimals)
    return max(1.01, result)


def ticks_between(price_a: float, price_b: float) -> int:
    """Approximate number of Betfair ticks between two prices (integer)."""
    lo, hi = min(price_a, price_b), max(price_a, price_b)
    count = 0
    current = lo
    while current < hi - 1e-9 and count < 500:
        current = next_price(current, "UP")
        count += 1
    return count


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PriceTick:
    """One price snapshot from Betfair Exchange."""

    timestamp_ms: float
    market_id: str
    selection_id: int
    best_back: float
    best_lay: float
    available_back_vol: float  # £ at best back price
    available_lay_vol: float  # £ at best lay price
    total_available: float  # £ across first 3 levels (back + lay)
    total_matched: float

    @property
    def spread(self) -> float:
        return max(0.0, self.best_lay - self.best_back)

    @property
    def mid_price(self) -> float:
        return (self.best_back + self.best_lay) / 2.0


@dataclass
class MomentumSignal:
    direction: str  # "SHORTENING" | "DRIFTING" | "NONE"
    consecutive_moves: int
    price_change: float  # total change over window (negative = shortening)
    velocity: float  # price change per second

    @property
    def is_actionable(self) -> bool:
        return self.direction != "NONE" and self.consecutive_moves >= 3


@dataclass
class ThinMarketAnalysis:
    tick: PriceTick
    is_thin: bool
    thin_reason: str
    spread: float
    fair_spread: float
    spread_ratio: float


@dataclass
class ScalpingOpportunity:
    """A detected scalping entry point."""

    market_id: str
    selection_id: int
    direction: str  # "BACK" | "LAY"
    entry_price: float
    target_exit_price: float
    stop_loss_price: float
    recommended_stake: float
    expected_profit: float
    expected_loss: float
    reward_risk_ratio: float
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    momentum: MomentumSignal
    thin_market: ThinMarketAnalysis
    timestamp_ms: float
    rationale: str

    def __str__(self) -> str:
        return (
            f"SCALP [{self.confidence}] {self.direction} @{self.entry_price:.2f} "
            f"target={self.target_exit_price:.2f} stop={self.stop_loss_price:.2f} "
            f"R:R={self.reward_risk_ratio:.2f} | {self.rationale}"
        )


@dataclass
class ScalpingSession:
    market_id: str
    signals_generated: int = 0
    high_confidence_signals: int = 0
    avg_reward_risk: float = 0.0
    ticks_analysed: int = 0
    opportunities: List[ScalpingOpportunity] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Session [{self.market_id}]: "
            f"{self.ticks_analysed} ticks, "
            f"{self.signals_generated} signals ({self.high_confidence_signals} HIGH), "
            f"avg R:R={self.avg_reward_risk:.2f}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LiquidityTrapEngine:
    """
    Monitors Betfair price ticks and emits ScalpingOpportunity signals.

    Parameters
    ----------
    thin_volume_threshold : £ total available below which market is "thin"
    matched_threshold     : £ cumulative matched below which thin rule applies
    momentum_window       : number of recent ticks to track for momentum
    momentum_min_consecutive : ticks in same direction to confirm momentum
    target_ticks          : profit target in Betfair ticks
    stop_ticks            : stop-loss in Betfair ticks
    commission_rate       : Betfair commission (default 5%)
    min_reward_risk       : minimum R:R ratio to emit a signal
    on_signal             : optional callback(ScalpingOpportunity)
    """

    def __init__(
        self,
        thin_volume_threshold: float = 5_000.0,
        matched_threshold: float = 50_000.0,
        momentum_window: int = 5,
        momentum_min_consecutive: int = 3,
        target_ticks: int = 2,
        stop_ticks: int = 3,
        commission_rate: float = 0.05,
        min_reward_risk: float = 1.5,
        on_signal: Optional[Callable[[ScalpingOpportunity], None]] = None,
    ) -> None:
        self._thin_vol = thin_volume_threshold
        self._matched_thr = matched_threshold
        self._momentum_window = momentum_window
        self._momentum_min = momentum_min_consecutive
        self._target_ticks = target_ticks
        self._stop_ticks = stop_ticks
        self._commission = commission_rate
        self._min_rr = min_reward_risk
        self._on_signal = on_signal

        # History per (market_id, selection_id)
        self._history: Dict[Tuple[str, int], Deque[PriceTick]] = {}
        self._sessions: Dict[str, ScalpingSession] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_tick(self, tick: PriceTick) -> Optional[ScalpingOpportunity]:
        """Add tick to history; return ScalpingOpportunity if signal fires."""
        key = (tick.market_id, tick.selection_id)
        if key not in self._history:
            self._history[key] = deque(maxlen=self._momentum_window)
        self._history[key].append(tick)

        session = self._get_session(tick.market_id)
        session.ticks_analysed += 1

        thin = self.analyse_thin_market(tick)
        if not thin.is_thin:
            return None

        momentum = self.detect_momentum(tick.market_id, tick.selection_id)
        if not momentum.is_actionable:
            return None

        opp = self._build_opportunity(tick, momentum, thin)
        if opp is None:
            return None

        session.signals_generated += 1
        if opp.confidence == "HIGH":
            session.high_confidence_signals += 1
        session.opportunities.append(opp)

        # Update avg R:R
        all_rr = [o.reward_risk_ratio for o in session.opportunities]
        session.avg_reward_risk = sum(all_rr) / len(all_rr)

        if self._on_signal:
            try:
                self._on_signal(opp)
            except Exception as exc:
                logger.warning("on_signal callback error: %s", exc)

        logger.info("SCALP SIGNAL: %s", opp)
        return opp

    def analyse_thin_market(self, tick: PriceTick) -> ThinMarketAnalysis:
        """Assess whether the market is thin enough for scalping."""
        fair_spread = betfair_tick_size(tick.mid_price)
        spread_ratio = tick.spread / max(fair_spread, 1e-9)
        is_thin = (
            tick.total_available < self._thin_vol
            and tick.total_matched < self._matched_thr
        )
        if not is_thin:
            reason = (
                f"Adequate liquidity: available=£{tick.total_available:,.0f} "
                f"matched=£{tick.total_matched:,.0f}"
            )
        else:
            reason = (
                f"Thin: available=£{tick.total_available:,.0f} < £{self._thin_vol:,.0f}, "
                f"spread_ratio={spread_ratio:.2f}"
            )
        return ThinMarketAnalysis(
            tick=tick,
            is_thin=is_thin,
            thin_reason=reason,
            spread=tick.spread,
            fair_spread=fair_spread,
            spread_ratio=spread_ratio,
        )

    def detect_momentum(self, market_id: str, selection_id: int) -> MomentumSignal:
        """Detect directional price momentum in the recent tick history."""
        key = (market_id, selection_id)
        history = list(self._history.get(key, []))
        if len(history) < 2:
            return MomentumSignal("NONE", 0, 0.0, 0.0)

        # Look at best_back series
        prices = [t.best_back for t in history]
        total_change = prices[-1] - prices[0]

        # Count consecutive moves from the end
        moves = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        if not moves:
            return MomentumSignal("NONE", 0, total_change, 0.0)

        # Determine direction from last move
        last_dir = (
            "DOWN" if moves[-1] < -1e-9 else ("UP" if moves[-1] > 1e-9 else "FLAT")
        )
        consecutive = 0
        for mv in reversed(moves):
            if last_dir == "DOWN" and mv < -1e-9:
                consecutive += 1
            elif last_dir == "UP" and mv > 1e-9:
                consecutive += 1
            else:
                break

        time_span_s = max(
            (history[-1].timestamp_ms - history[0].timestamp_ms) / 1000.0, 0.001
        )
        velocity = total_change / time_span_s

        direction = (
            "SHORTENING"
            if last_dir == "DOWN"
            else ("DRIFTING" if last_dir == "UP" else "NONE")
        )
        return MomentumSignal(
            direction=direction,
            consecutive_moves=consecutive,
            price_change=total_change,
            velocity=velocity,
        )

    def session_stats(self, market_id: str) -> ScalpingSession:
        return self._get_session(market_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_session(self, market_id: str) -> ScalpingSession:
        if market_id not in self._sessions:
            self._sessions[market_id] = ScalpingSession(market_id=market_id)
        return self._sessions[market_id]

    def _build_opportunity(
        self,
        tick: PriceTick,
        momentum: MomentumSignal,
        thin: ThinMarketAnalysis,
    ) -> Optional[ScalpingOpportunity]:
        """Construct a ScalpingOpportunity; return None if R:R too low."""
        stake = 100.0

        if momentum.direction == "SHORTENING":
            # Price falling → back now, lay (exit) at lower price
            direction = "BACK"
            entry = tick.best_back
            target = entry
            for _ in range(self._target_ticks):
                target = next_price(target, "DOWN")
            stop = entry
            for _ in range(self._stop_ticks):
                stop = next_price(stop, "UP")
            profit = self._back_profit(entry, target, stake)
            loss = self._back_loss(entry, stop, stake)
        else:
            # Price rising → lay now, back (exit) at higher price
            direction = "LAY"
            entry = tick.best_lay
            target = entry
            for _ in range(self._target_ticks):
                target = next_price(target, "UP")
            stop = entry
            for _ in range(self._stop_ticks):
                stop = next_price(stop, "DOWN")
            profit = self._lay_profit(entry, target, stake)
            loss = self._lay_loss(entry, stop, stake)

        if loss <= 0:
            return None
        rr = profit / loss
        if rr < self._min_rr:
            return None

        if rr >= 2.5 and momentum.consecutive_moves >= 4:
            conf = "HIGH"
        elif rr >= 1.8 and momentum.consecutive_moves >= 3:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        rationale = (
            f"{momentum.direction} momentum ({momentum.consecutive_moves} ticks), "
            f"thin market (£{thin.tick.total_available:,.0f} available)"
        )

        return ScalpingOpportunity(
            market_id=tick.market_id,
            selection_id=tick.selection_id,
            direction=direction,
            entry_price=entry,
            target_exit_price=target,
            stop_loss_price=stop,
            recommended_stake=stake,
            expected_profit=profit,
            expected_loss=loss,
            reward_risk_ratio=rr,
            confidence=conf,
            momentum=momentum,
            thin_market=thin,
            timestamp_ms=tick.timestamp_ms,
            rationale=rationale,
        )

    def _back_profit(self, entry: float, exit_price: float, stake: float) -> float:
        """Net profit from a back→lay scalp (entry at back, exit via lay at lower price)."""
        # Back wins at exit (now laying) — simplified: profit ≈ stake × (entry - exit_price)
        gross = stake * (entry - exit_price) / exit_price
        return gross * (1.0 - self._commission)

    def _back_loss(self, entry: float, stop: float, stake: float) -> float:
        gross = stake * (stop - entry) / entry
        return gross

    def _lay_profit(self, entry: float, exit_price: float, stake: float) -> float:
        gross = stake * (exit_price - entry) / entry
        return gross * (1.0 - self._commission)

    def _lay_loss(self, entry: float, stop: float, stake: float) -> float:
        gross = stake * (entry - stop) / stop
        return gross


# ---------------------------------------------------------------------------
# Optional Betfair data client
# ---------------------------------------------------------------------------


class BetfairTickClient:
    """Fetches live price ticks from Betfair Exchange API via urllib."""

    _API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"

    def __init__(self, app_key: str, session_token: str) -> None:
        self._app_key = app_key
        self._session_token = session_token

    def fetch_ticks(self, market_id: str, selection_ids: List[int]) -> List[PriceTick]:
        """Fetch best-offer price ticks for the given selection IDs."""
        import json
        import urllib.request

        payload = json.dumps(
            [
                {
                    "jsonrpc": "2.0",
                    "method": "SportsAPING/v1.0/listMarketBook",
                    "params": {
                        "marketIds": [market_id],
                        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
                    },
                    "id": 1,
                }
            ]
        ).encode()

        headers = {
            "X-Application": self._app_key,
            "X-Authentication": self._session_token,
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(self._API_URL, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            logger.error("BetfairTickClient fetch error: %s", exc)
            return []

        ticks: List[PriceTick] = []
        now_ms = time.time() * 1000.0
        try:
            books = data[0]["result"]
            for book in books:
                total_matched = float(book.get("totalMatched", 0))
                for runner in book.get("runners", []):
                    sel_id = int(runner["selectionId"])
                    if sel_id not in selection_ids:
                        continue
                    ex = runner.get("ex", {})
                    backs = ex.get("availableToBack", [])
                    lays = ex.get("availableToLay", [])
                    best_back = float(backs[0]["price"]) if backs else 0.0
                    best_lay = float(lays[0]["price"]) if lays else 999.0
                    back_vol = float(backs[0]["size"]) if backs else 0.0
                    lay_vol = float(lays[0]["size"]) if lays else 0.0
                    total_avail = sum(b["size"] for b in backs[:3]) + sum(
                        lay["size"] for lay in lays[:3]
                    )
                    ticks.append(
                        PriceTick(
                            timestamp_ms=now_ms,
                            market_id=market_id,
                            selection_id=sel_id,
                            best_back=best_back,
                            best_lay=best_lay,
                            available_back_vol=back_vol,
                            available_lay_vol=lay_vol,
                            total_available=total_avail,
                            total_matched=total_matched,
                        )
                    )
        except Exception as exc:
            logger.error("BetfairTickClient parse error: %s", exc)
        return ticks


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def detect_scalping_opportunities(
    ticks: List[Dict],
    thin_threshold: float = 5_000.0,
) -> List[ScalpingOpportunity]:
    """
    One-shot: process a batch of tick dicts and return all signals.

    Each dict must have: market_id, selection_id, best_back, best_lay,
    available_back_vol, available_lay_vol, total_available, total_matched.
    timestamp_ms is optional (defaults to current time).
    """
    engine = LiquidityTrapEngine(thin_volume_threshold=thin_threshold)
    opportunities: List[ScalpingOpportunity] = []
    now_ms = time.time() * 1000.0
    for raw in ticks:
        tick = PriceTick(
            timestamp_ms=float(raw.get("timestamp_ms", now_ms)),
            market_id=str(raw["market_id"]),
            selection_id=int(raw["selection_id"]),
            best_back=float(raw["best_back"]),
            best_lay=float(raw["best_lay"]),
            available_back_vol=float(raw.get("available_back_vol", 0)),
            available_lay_vol=float(raw.get("available_lay_vol", 0)),
            total_available=float(raw["total_available"]),
            total_matched=float(raw.get("total_matched", 0)),
        )
        opp = engine.process_tick(tick)
        if opp:
            opportunities.append(opp)
    return opportunities
