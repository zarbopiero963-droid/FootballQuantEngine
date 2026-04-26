from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PriceLevel:
    """One level in the Betfair order book."""

    price: float  # Betfair decimal price (e.g. 2.1)
    size: float  # £ available at this price


@dataclass
class OrderBookSnapshot:
    """Full depth of market for one Betfair selection at one moment."""

    market_id: str
    selection_id: int
    selection_name: str
    timestamp: float
    available_to_back: List[PriceLevel]  # sorted descending (best back first)
    available_to_lay: List[PriceLevel]  # sorted ascending (best lay first)
    last_price_traded: float
    total_matched: float

    @property
    def best_back(self) -> float:
        return self.available_to_back[0].price if self.available_to_back else 0.0

    @property
    def best_lay(self) -> float:
        return self.available_to_lay[0].price if self.available_to_lay else 999.0

    @property
    def spread(self) -> float:
        return self.best_lay - self.best_back

    @property
    def total_back_volume(self) -> float:
        """Total £ available to back across all levels."""
        return sum(p.size for p in self.available_to_back)

    @property
    def total_lay_volume(self) -> float:
        """Total £ available to lay across all levels."""
        return sum(p.size for p in self.available_to_lay)


@dataclass
class OrderBookImbalance:
    """Imbalance analysis for one order book snapshot."""

    snapshot: OrderBookSnapshot
    imbalance: float  # (back_vol - lay_vol) / (back_vol + lay_vol) ∈ [-1, 1]
    # positive = more money wanting to back (price may lengthen)
    # negative = more money wanting to lay (price may shorten)
    weighted_mid: float  # volume-weighted midpoint price
    back_wall: float  # largest single back level price (support level)
    lay_wall: float  # largest single lay level price (resistance level)
    direction: str  # "SHORTENING" | "LENGTHENING" | "STABLE"
    confidence: float  # 0–1 signal strength


@dataclass
class GreenbookOpportunity:
    """
    A greenbook (risk-free trade) opportunity.

    If you backed at back_odds and the market is shortening,
    you can lay at the new shorter odds on Betfair itself to lock a guaranteed profit.
    """

    market_id: str
    selection_id: int
    selection_name: str
    back_odds: float  # your original back price
    back_stake: float  # your original stake
    lay_odds: float  # current best lay price (to lay against yourself)
    lay_stake: float  # optimal lay stake for zero-risk
    profit_if_wins: float  # locked profit if selection wins
    profit_if_loses: float  # locked profit if selection loses
    is_profitable: bool  # True if both profits > 0

    def __str__(self) -> str:
        return (
            f"GREENBOOK [{self.selection_name}] "
            f"backed@{self.back_odds:.2f} laid@{self.lay_odds:.2f} "
            f"profit: win={self.profit_if_wins:+.2f} lose={self.profit_if_loses:+.2f}"
        )


@dataclass
class OrderFlowAlert:
    """Alert triggered by significant order book imbalance."""

    market_id: str
    selection_id: int
    selection_name: str
    timestamp: float
    imbalance: OrderBookImbalance
    predicted_move: str  # "ODDS_SHORTEN" | "ODDS_LENGTHEN"
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    action: str  # e.g. "BACK before shortening" or "WAIT"

    def __str__(self) -> str:
        return (
            f"ORDER FLOW [{self.selection_name}] "
            f"imbalance={self.imbalance.imbalance:+.3f} "
            f"→ {self.predicted_move} conf={self.confidence} | {self.action}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """Simple linear regression slope (least squares).

    Returns the slope dy/dx of the best-fit line through the (x, y) pairs.
    Returns 0.0 if fewer than 2 points or if x variance is zero.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return num / denom


def compute_greenbook(
    back_odds: float,
    back_stake: float,
    lay_odds: float,
    commission_rate: float = 0.05,
) -> Tuple[float, float]:
    """
    Returns (profit_if_wins, profit_if_loses) for a greenbook trade.

    lay_stake = back_stake × back_odds / lay_odds

    If selection wins:
        - Back bet pays:  back_stake × (back_odds - 1)  net winnings before commission
        - Lay bet costs:  lay_stake × (lay_odds - 1)    liability paid out
        - Commission on back-bet net winnings (Betfair charges on net profit from wins)
        profit_if_wins = back_stake × (back_odds - 1) × (1 - commission_rate)
                         - lay_stake × (lay_odds - 1)

    If selection loses:
        - Back stake lost: -back_stake
        - Lay bet collects: +lay_stake  (the backer's stake)
        profit_if_loses = lay_stake - back_stake
    """
    lay_stake = back_stake * back_odds / lay_odds
    back_net_winnings = back_stake * (back_odds - 1.0)
    lay_liability = lay_stake * (lay_odds - 1.0)
    profit_if_wins = back_net_winnings * (1.0 - commission_rate) - lay_liability
    profit_if_loses = lay_stake - back_stake
    return profit_if_wins, profit_if_loses


# ---------------------------------------------------------------------------
# OrderBookAnalyzer
# ---------------------------------------------------------------------------


class OrderBookAnalyzer:
    """
    Analyzes Betfair Exchange depth of market for order flow signals.

    Parameters
    ----------
    imbalance_threshold : imbalance magnitude to trigger alert (default 0.30)
    history_window      : number of snapshots to keep for trend analysis (default 20)
    on_alert            : optional callback(OrderFlowAlert)
    on_greenbook        : optional callback(GreenbookOpportunity)
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.30,
        history_window: int = 20,
        on_alert: Optional[Callable[[OrderFlowAlert], None]] = None,
        on_greenbook: Optional[Callable[[GreenbookOpportunity], None]] = None,
    ) -> None:
        self._threshold = imbalance_threshold
        self._history: Dict[Tuple[str, int], Deque[OrderBookSnapshot]] = {}
        self._on_alert = on_alert
        self._on_greenbook = on_greenbook
        self._window = history_window

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _key(self, market_id: str, selection_id: int) -> Tuple[str, int]:
        return (market_id, selection_id)

    def _get_history(
        self, market_id: str, selection_id: int
    ) -> Deque[OrderBookSnapshot]:
        key = self._key(market_id, selection_id)
        if key not in self._history:
            self._history[key] = deque(maxlen=self._window)
        return self._history[key]

    def _compute_direction(
        self,
        market_id: str,
        selection_id: int,
        current_imbalance: float,
    ) -> str:
        """
        Determine direction from rolling imbalance trend over last 3 snapshots.
        Trend < -0.25 → SHORTENING (lay pressure dominates → odds shrink)
        Trend > +0.25 → LENGTHENING
        else STABLE
        """
        history = self._get_history(market_id, selection_id)
        # Gather up to last 3 snapshots' imbalances; we don't have them stored yet
        # so we compute them on-the-fly from the stored snapshots.
        snapshots = list(history)
        if len(snapshots) < 3:
            # Not enough history; use current imbalance magnitude only
            if current_imbalance < -0.25:
                return "SHORTENING"
            if current_imbalance > 0.25:
                return "LENGTHENING"
            return "STABLE"

        # Use the last 3 snapshots (not including the current one, which isn't stored yet)
        recent = snapshots[-3:]
        xs = [s.timestamp for s in recent]
        ys = [
            (s.total_back_volume - s.total_lay_volume)
            / (s.total_back_volume + s.total_lay_volume + 1e-9)
            for s in recent
        ]
        slope = _linear_slope(xs, ys)

        # Normalise slope by time span to get a per-second rate, then scale
        time_span = xs[-1] - xs[0] if xs[-1] != xs[0] else 1.0
        normalised_trend = slope * time_span  # dimensionless change over window

        if normalised_trend < -0.25:
            return "SHORTENING"
        if normalised_trend > 0.25:
            return "LENGTHENING"
        return "STABLE"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, snapshot: OrderBookSnapshot) -> OrderBookImbalance:
        """
        Compute imbalance metrics for a snapshot.

        imbalance = (back_vol - lay_vol) / (back_vol + lay_vol + 1e-9)
        weighted_mid = weighted average of (best_back, best_lay) by volume
        back_wall = price of the largest single back level
        lay_wall = price of the largest single lay level
        direction:
          if rolling imbalance trend < -0.25 over last 3 snapshots → SHORTENING
          if rolling imbalance trend > +0.25 → LENGTHENING
          else STABLE
        confidence = min(1.0, abs(imbalance) / 0.5)
        """
        back_vol = snapshot.total_back_volume
        lay_vol = snapshot.total_lay_volume
        total_vol = back_vol + lay_vol + 1e-9

        imbalance = (back_vol - lay_vol) / total_vol

        # Volume-weighted midpoint: weight best_back by back_vol, best_lay by lay_vol
        best_back = snapshot.best_back
        best_lay = snapshot.best_lay
        weight_sum = back_vol + lay_vol
        if weight_sum > 0:
            weighted_mid = (best_back * back_vol + best_lay * lay_vol) / weight_sum
        else:
            weighted_mid = (
                (best_back + best_lay) / 2.0 if (best_back + best_lay) > 0 else 0.0
            )

        # back_wall: price of the largest single back level (support level)
        if snapshot.available_to_back:
            back_wall_level = max(snapshot.available_to_back, key=lambda p: p.size)
            back_wall = back_wall_level.price
        else:
            back_wall = 0.0

        # lay_wall: price of the largest single lay level (resistance level)
        if snapshot.available_to_lay:
            lay_wall_level = max(snapshot.available_to_lay, key=lambda p: p.size)
            lay_wall = lay_wall_level.price
        else:
            lay_wall = 999.0

        direction = self._compute_direction(
            snapshot.market_id, snapshot.selection_id, imbalance
        )

        confidence = min(1.0, abs(imbalance) / 0.5)

        return OrderBookImbalance(
            snapshot=snapshot,
            imbalance=imbalance,
            weighted_mid=weighted_mid,
            back_wall=back_wall,
            lay_wall=lay_wall,
            direction=direction,
            confidence=confidence,
        )

    def check_alert(self, snapshot: OrderBookSnapshot) -> Optional[OrderFlowAlert]:
        """
        Process snapshot: analyse, store in history, emit alert if threshold exceeded.
        action: if SHORTENING → "BACK {name} immediately", if LENGTHENING → "LAY {name}".
        """
        imbalance_obj = self.analyse(snapshot)

        # Store snapshot in history AFTER analysis (so current isn't included in direction calc)
        history = self._get_history(snapshot.market_id, snapshot.selection_id)
        history.append(snapshot)

        imb_val = imbalance_obj.imbalance

        # Only alert when absolute imbalance exceeds threshold
        if abs(imb_val) < self._threshold:
            logger.debug(
                "No alert for %s/%s: imbalance=%.3f below threshold=%.3f",
                snapshot.market_id,
                snapshot.selection_id,
                imb_val,
                self._threshold,
            )
            return None

        # Map direction to prediction and action
        if imb_val < 0:
            # More lay volume → odds will shorten
            predicted_move = "ODDS_SHORTEN"
            action = f"BACK {snapshot.selection_name} immediately"
        else:
            # More back volume → odds will lengthen
            predicted_move = "ODDS_LENGTHEN"
            action = f"LAY {snapshot.selection_name}"

        # Confidence label
        conf_val = imbalance_obj.confidence
        if conf_val >= 0.75:
            confidence_label = "HIGH"
        elif conf_val >= 0.45:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        alert = OrderFlowAlert(
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            selection_name=snapshot.selection_name,
            timestamp=snapshot.timestamp,
            imbalance=imbalance_obj,
            predicted_move=predicted_move,
            confidence=confidence_label,
            action=action,
        )

        logger.info("Order flow alert: %s", alert)

        if self._on_alert is not None:
            try:
                self._on_alert(alert)
            except Exception:
                logger.exception("on_alert callback raised an exception")

        return alert

    def compute_greenbook(
        self,
        snapshot: OrderBookSnapshot,
        back_odds: float,
        back_stake: float,
        commission_rate: float = 0.05,
    ) -> GreenbookOpportunity:
        """
        Compute greenbook for an existing back bet.

        Optimal lay stake = back_stake × back_odds / lay_odds
        Profit if wins = back_stake × (back_odds - 1) × (1 - commission) - lay_stake × (lay_odds - 1)
        Profit if loses = lay_stake - back_stake
        where commission is applied to the winning side on Betfair (5% default).
        is_profitable = both profits > 0.
        """
        lay_odds = snapshot.best_lay
        lay_stake = back_stake * back_odds / lay_odds

        profit_if_wins, profit_if_loses = compute_greenbook(
            back_odds=back_odds,
            back_stake=back_stake,
            lay_odds=lay_odds,
            commission_rate=commission_rate,
        )

        is_profitable = profit_if_wins > 0.0 and profit_if_loses > 0.0

        opportunity = GreenbookOpportunity(
            market_id=snapshot.market_id,
            selection_id=snapshot.selection_id,
            selection_name=snapshot.selection_name,
            back_odds=back_odds,
            back_stake=back_stake,
            lay_odds=lay_odds,
            lay_stake=lay_stake,
            profit_if_wins=profit_if_wins,
            profit_if_loses=profit_if_loses,
            is_profitable=is_profitable,
        )

        logger.info("Greenbook computed: %s", opportunity)

        if is_profitable and self._on_greenbook is not None:
            try:
                self._on_greenbook(opportunity)
            except Exception:
                logger.exception("on_greenbook callback raised an exception")

        return opportunity

    def trend_imbalance(
        self,
        market_id: str,
        selection_id: int,
    ) -> Optional[float]:
        """
        Rolling imbalance trend over last 3 snapshots (slope of imbalance over time).
        Returns None if fewer than 3 snapshots available.
        Simple linear regression slope / time.
        """
        history = self._get_history(market_id, selection_id)
        snapshots = list(history)
        if len(snapshots) < 3:
            return None

        recent = snapshots[-3:]
        xs = [s.timestamp for s in recent]
        ys = [
            (s.total_back_volume - s.total_lay_volume)
            / (s.total_back_volume + s.total_lay_volume + 1e-9)
            for s in recent
        ]
        slope = _linear_slope(xs, ys)
        return slope

    def summary(self, snapshot: OrderBookSnapshot) -> str:
        """Multi-line readable summary of the order book."""
        imbalance_obj = self.analyse(snapshot)
        lines = [
            f"=== Order Book: {snapshot.selection_name} ===",
            f"  Market:         {snapshot.market_id}",
            f"  Selection ID:   {snapshot.selection_id}",
            f"  Timestamp:      {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(snapshot.timestamp))} UTC",
            f"  Last Traded:    {snapshot.last_price_traded:.2f}",
            f"  Total Matched:  £{snapshot.total_matched:,.2f}",
            "",
            f"  Best Back:      {snapshot.best_back:.2f}  (total back vol: £{snapshot.total_back_volume:,.2f})",
            f"  Best Lay:       {snapshot.best_lay:.2f}  (total lay vol:  £{snapshot.total_lay_volume:,.2f})",
            f"  Spread:         {snapshot.spread:.4f}",
            "",
            "  --- Back Levels (descending) ---",
        ]

        for i, level in enumerate(snapshot.available_to_back[:5]):
            lines.append(
                f"    [{i + 1}] price={level.price:.2f}  size=£{level.size:,.2f}"
            )
        if len(snapshot.available_to_back) > 5:
            lines.append(f"    ... ({len(snapshot.available_to_back) - 5} more levels)")

        lines.append("  --- Lay Levels (ascending) ---")
        for i, level in enumerate(snapshot.available_to_lay[:5]):
            lines.append(
                f"    [{i + 1}] price={level.price:.2f}  size=£{level.size:,.2f}"
            )
        if len(snapshot.available_to_lay) > 5:
            lines.append(f"    ... ({len(snapshot.available_to_lay) - 5} more levels)")

        lines += [
            "",
            "  --- Imbalance Analysis ---",
            f"  Imbalance:      {imbalance_obj.imbalance:+.4f}  (range -1 to +1)",
            f"  Direction:      {imbalance_obj.direction}",
            f"  Confidence:     {imbalance_obj.confidence:.2%}",
            f"  Weighted Mid:   {imbalance_obj.weighted_mid:.4f}",
            f"  Back Wall:      {imbalance_obj.back_wall:.2f}  (support)",
            f"  Lay Wall:       {imbalance_obj.lay_wall:.2f}  (resistance)",
        ]

        trend = self.trend_imbalance(snapshot.market_id, snapshot.selection_id)
        if trend is not None:
            lines.append(f"  Trend Slope:    {trend:+.6f} per second")
        else:
            lines.append("  Trend Slope:    (insufficient history)")

        lines.append("=" * 42)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BetfairDepthClient
# ---------------------------------------------------------------------------


class BetfairDepthClient:
    """
    Fetches full depth of market from Betfair Exchange APING.
    Uses listMarketBook with FULL depth projection.
    """

    def __init__(self, app_key: str, session_token: str, timeout: int = 8) -> None:
        self._app_key = app_key
        self._session_token = session_token
        self._timeout = timeout

    def _request(self, operation: str, params: dict) -> dict:
        """POST to Betfair APING REST endpoint with JSON body."""
        url = f"https://api.betfair.com/exchange/betting/rest/v1.0/{operation}/"
        body = json.dumps(params).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "X-Application": self._app_key,
                "X-Authentication": self._session_token,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Betfair HTTP error %s for %s: %s",
                exc.code,
                operation,
                exc.reason,
            )
            return {}
        except urllib.error.URLError as exc:
            logger.warning("Betfair URL error for %s: %s", operation, exc.reason)
            return {}
        except Exception as exc:
            logger.warning("Betfair request failed: %s", exc)
            return {}

    def fetch_depth(self, market_ids: List[str]) -> List[OrderBookSnapshot]:
        """
        Call listMarketBook with priceData=["EX_ALL_OFFERS"] to get full depth.
        Parse: runners[].ex.availableToBack (list of {price, size})
               runners[].ex.availableToLay
               runners[].lastPriceTraded
               runners[].totalMatched
        Return list of OrderBookSnapshot objects.
        """
        if not market_ids:
            return []

        params = {
            "marketIds": market_ids,
            "priceProjection": {
                "priceData": ["EX_ALL_OFFERS"],
                "exBestOffersOverrides": {
                    "rollupModel": "STAKE",
                    "bestPricesDepth": 999,  # full depth
                },
                "virtualise": False,
            },
        }

        response = self._request("listMarketBook", params)
        if not response:
            logger.warning(
                "Empty response from listMarketBook for markets: %s", market_ids
            )
            return []

        # Betfair APING returns a JSON array at the top level
        if not isinstance(response, list):
            logger.warning(
                "Unexpected response type from listMarketBook: %s",
                type(response).__name__,
            )
            return []

        snapshots: List[OrderBookSnapshot] = []
        fetch_time = time.time()

        for market in response:
            market_id = market.get("marketId", "")
            runners = market.get("runners", [])

            for runner in runners:
                selection_id: int = runner.get("selectionId", 0)
                runner_name: str = runner.get("runnerName", str(selection_id))
                last_price_traded: float = float(
                    runner.get("lastPriceTraded", 0.0) or 0.0
                )
                total_matched: float = float(runner.get("totalMatched", 0.0) or 0.0)

                ex = runner.get("ex", {})

                available_to_back: List[PriceLevel] = []
                for entry in ex.get("availableToBack", []):
                    price = float(entry.get("price", 0.0))
                    size = float(entry.get("size", 0.0))
                    if price > 0 and size > 0:
                        available_to_back.append(PriceLevel(price=price, size=size))
                # Ensure descending order (best back first = highest price)
                available_to_back.sort(key=lambda p: p.price, reverse=True)

                available_to_lay: List[PriceLevel] = []
                for entry in ex.get("availableToLay", []):
                    price = float(entry.get("price", 0.0))
                    size = float(entry.get("size", 0.0))
                    if price > 0 and size > 0:
                        available_to_lay.append(PriceLevel(price=price, size=size))
                # Ensure ascending order (best lay first = lowest price)
                available_to_lay.sort(key=lambda p: p.price)

                snapshot = OrderBookSnapshot(
                    market_id=market_id,
                    selection_id=selection_id,
                    selection_name=runner_name,
                    timestamp=fetch_time,
                    available_to_back=available_to_back,
                    available_to_lay=available_to_lay,
                    last_price_traded=last_price_traded,
                    total_matched=total_matched,
                )
                snapshots.append(snapshot)
                logger.debug(
                    "Fetched depth for %s/%s: %d back levels, %d lay levels",
                    market_id,
                    selection_id,
                    len(available_to_back),
                    len(available_to_lay),
                )

        logger.info(
            "listMarketBook returned %d runner snapshots from %d markets",
            len(snapshots),
            len(response),
        )
        return snapshots
