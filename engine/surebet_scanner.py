"""
surebet_scanner.py
==================
Production-quality arbitrage (surebet) scanner.

Scans odds from multiple bookmakers simultaneously to find arbitrage
opportunities: combinations where the sum of implied probabilities across
all outcomes is below 100%, guaranteeing a risk-free profit regardless of
the result.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_MIN_PROFIT_PCT = 0.003  # only flag arbs with guaranteed profit > 0.3 %
_DEFAULT_BANKROLL = 1000.0

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BookOdds:
    """Odds from one bookmaker for one outcome."""

    bookmaker: str
    market: str  # "1X2" or "2way"
    outcome: str  # "home" | "draw" | "away"  (or "yes"/"no" for 2-way)
    odds: float
    timestamp: float = field(default_factory=time.time)

    @property
    def implied(self) -> float:
        return 1.0 / self.odds if self.odds > 1.0 else 1.0


@dataclass
class SurebetLeg:
    """One leg of a surebet: a single outcome to back."""

    bookmaker: str
    outcome: str
    odds: float
    stake: float  # calculated optimal stake
    implied: float
    potential_return: float  # stake × odds


@dataclass
class SurebetOpportunity:
    """A confirmed surebet across multiple bookmakers."""

    fixture_id: int
    fixture_desc: str  # "Home vs Away"
    market: str  # "1X2" | "2way"
    legs: List[SurebetLeg]  # one per outcome (2 or 3 legs)
    overround: float  # sum of implied probs (< 1.0 for an arb)
    guaranteed_profit: float  # absolute profit for given bankroll
    profit_pct: float  # guaranteed_profit / bankroll × 100
    bankroll: float
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        legs_str = " | ".join(
            f"{leg.bookmaker}:{leg.outcome}@{leg.odds:.2f}(£{leg.stake:.2f})"
            for leg in self.legs
        )
        return (
            f"SUREBET [{self.fixture_desc}] {self.market} "
            f"overround={self.overround:.4f} "
            f"profit={self.profit_pct:+.2f}% (£{self.guaranteed_profit:.2f}) "
            f"| {legs_str}"
        )


@dataclass
class ScanResult:
    """Full scan output."""

    n_fixtures_scanned: int
    n_opportunities: int
    opportunities: List[SurebetOpportunity]
    scan_duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"SurebetScan: {self.n_fixtures_scanned} fixtures, "
            f"{self.n_opportunities} arbs found in {self.scan_duration_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# OddsAPIClient
# ---------------------------------------------------------------------------


class OddsAPIClient:
    """
    Fetches multi-bookmaker odds from the-odds-api.com.

    Free tier: 500 requests/month. Use sport_key="soccer_*".
    """

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        self._key = api_key
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_odds(
        self,
        sport_key: str = "soccer_epl",
        regions: str = "uk,eu",
        markets: str = "h2h",
        bookmakers: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        GET /v4/sports/{sport_key}/odds

        Params: apiKey, regions, markets, bookmakers (comma-separated if
        provided). Returns list of fixture dicts. Empty list on error.
        Logs warnings on HTTP/parse failures.
        """
        params: Dict[str, str] = {
            "apiKey": self._key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = self._build_url(f"/sports/{sport_key}/odds", params)
        return self._get_json(url) or []

    def list_sports(self) -> List[dict]:
        """GET /v4/sports?apiKey={key}. Returns list of sport dicts."""
        url = self._build_url("/sports", {"apiKey": self._key})
        return self._get_json(url) or []

    def parse_bookodds(self, fixtures: List[dict]) -> Dict[int, List[BookOdds]]:
        """
        Parse fixtures API response into {fixture_id: [BookOdds]}.

        fixture_id is derived from fixture['id'] (hashed to int if a string).
        Each bookmaker's h2h market: outcomes Home/Draw/Away or just Home/Away.
        """
        result: Dict[int, List[BookOdds]] = {}

        for fixture in fixtures:
            try:
                raw_id = fixture.get("id", "")
                fixture_id = self._to_fixture_id(raw_id)

                home_team = fixture.get("home_team", "Home")
                away_team = fixture.get("away_team", "Away")

                book_odds_list: List[BookOdds] = []

                for bm in fixture.get("bookmakers", []):
                    bookmaker_key = bm.get("key", bm.get("title", "unknown"))
                    for mkt in bm.get("markets", []):
                        market_key = mkt.get("key", "")
                        if market_key != "h2h":
                            continue

                        outcomes = mkt.get("outcomes", [])
                        n_outcomes = len(outcomes)

                        for oc in outcomes:
                            oc_name = oc.get("name", "")
                            oc_price = float(oc.get("price", 0.0))
                            if oc_price <= 1.0:
                                logger.warning(
                                    "Skipping odds <= 1.0 for %s / %s / %s",
                                    fixture_id,
                                    bookmaker_key,
                                    oc_name,
                                )
                                continue

                            if n_outcomes == 3:
                                market = "1X2"
                                outcome_label = self._normalise_outcome_3way(
                                    oc_name, home_team, away_team
                                )
                            else:
                                market = "2way"
                                outcome_label = self._normalise_outcome_2way(
                                    oc_name, home_team, away_team
                                )

                            book_odds_list.append(
                                BookOdds(
                                    bookmaker=bookmaker_key,
                                    market=market,
                                    outcome=outcome_label,
                                    odds=oc_price,
                                )
                            )

                result[fixture_id] = book_odds_list

            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Error parsing fixture %s: %s", fixture, exc)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self, path: str, params: Dict[str, str]) -> str:
        query = urllib.parse.urlencode(params)
        return f"{_THE_ODDS_API_BASE}{path}?{query}"

    def _get_json(self, url: str) -> Optional[list]:
        try:
            logger.debug("GET %s", url)
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            logger.warning("HTTP %s fetching %s: %s", exc.code, url, exc.reason)
        except urllib.error.URLError as exc:
            logger.warning("URL error fetching %s: %s", url, exc.reason)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error for %s: %s", url, exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Unexpected error fetching %s: %s", url, exc)
        return None

    @staticmethod
    def _to_fixture_id(raw_id: str) -> int:
        """Convert a string fixture id to a stable integer."""
        try:
            return int(raw_id)
        except (ValueError, TypeError):
            return abs(hash(str(raw_id))) & 0x7FFF_FFFF

    @staticmethod
    def _normalise_outcome_3way(name: str, home: str, away: str) -> str:
        lname = name.strip().lower()
        lhome = home.strip().lower()
        laway = away.strip().lower()
        if lname == lhome or lname == "home":
            return "home"
        if lname == laway or lname == "away":
            return "away"
        if lname in ("draw", "tie"):
            return "draw"
        # Fallback: unknown outcome kept as-is (lowercased)
        return lname

    @staticmethod
    def _normalise_outcome_2way(name: str, home: str, away: str) -> str:
        lname = name.strip().lower()
        lhome = home.strip().lower()
        laway = away.strip().lower()
        if lname == lhome or lname in ("home", "yes", "1"):
            return "yes"
        if lname == laway or lname in ("away", "no", "2"):
            return "no"
        return lname


# ---------------------------------------------------------------------------
# SurebetScanner
# ---------------------------------------------------------------------------


class SurebetScanner:
    """
    Detects arbitrage opportunities across bookmaker odds.

    For 3-way (1X2):
      overround = 1/odds_home + 1/odds_draw + 1/odds_away
      arb exists if overround < 1.0

    For 2-way:
      overround = 1/odds_a + 1/odds_b

    Optimal stakes for guaranteed profit:
      stake_i = bankroll × (1/odds_i) / overround
      This ensures each leg returns bankroll / overround.

      profit = bankroll / overround - bankroll = bankroll × (1/overround - 1)

    Parameters
    ----------
    min_profit_pct  : minimum guaranteed profit fraction to report (default 0.003 = 0.3 %)
    bankroll        : reference bankroll for stake calculation (default £1000)
    on_opportunity  : optional callback(SurebetOpportunity)
    """

    def __init__(
        self,
        min_profit_pct: float = _MIN_PROFIT_PCT,
        bankroll: float = _DEFAULT_BANKROLL,
        on_opportunity: Optional[Callable[[SurebetOpportunity], None]] = None,
    ) -> None:
        self._min_profit_pct = min_profit_pct
        self._bankroll = bankroll
        self._on_opportunity = on_opportunity

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan_fixture(
        self,
        fixture_id: int,
        fixture_desc: str,
        all_odds: List[BookOdds],
    ) -> List[SurebetOpportunity]:
        """
        Scan one fixture's multi-bookmaker odds for arbs.

        Algorithm
        ---------
        For 3-way (1X2):
          Take the best (maximum) decimal odds available across all bookmakers
          for each of home, draw, away.  This gives the tightest possible
          overround for the fixture.  If overround < 1 - min_profit_pct the
          arb is confirmed and stakes are computed.

        For 2-way:
          Same as above but using only home/away (or yes/no) outcomes.

        Checking only the single best price per outcome is an O(n) optimisation
        over checking all combinations — for a fixed fixture the best-price
        combination is always optimal.
        """
        opportunities: List[SurebetOpportunity] = []

        # ---- 3-way -------------------------------------------------------
        opp_3way = self._scan_3way(fixture_id, fixture_desc, all_odds)
        if opp_3way is not None:
            opportunities.append(opp_3way)
            if self._on_opportunity is not None:
                self._on_opportunity(opp_3way)

        # ---- 2-way -------------------------------------------------------
        opp_2way = self._scan_2way(fixture_id, fixture_desc, all_odds)
        if opp_2way is not None:
            opportunities.append(opp_2way)
            if self._on_opportunity is not None:
                self._on_opportunity(opp_2way)

        return opportunities

    def scan_all(
        self,
        fixtures_odds: Dict[int, List[BookOdds]],
        fixture_descs: Optional[Dict[int, str]] = None,
    ) -> ScanResult:
        """
        Scan all fixtures. Returns ScanResult with all found opportunities.
        """
        t0 = time.time()
        all_opps: List[SurebetOpportunity] = []

        for fixture_id, odds_list in fixtures_odds.items():
            desc = (fixture_descs or {}).get(fixture_id, f"Fixture #{fixture_id}")
            try:
                opps = self.scan_fixture(fixture_id, desc, odds_list)
                all_opps.extend(opps)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Error scanning fixture %s (%s): %s", fixture_id, desc, exc
                )

        duration = time.time() - t0
        result = ScanResult(
            n_fixtures_scanned=len(fixtures_odds),
            n_opportunities=len(all_opps),
            opportunities=all_opps,
            scan_duration_seconds=duration,
        )
        logger.info(result.summary())
        return result

    def optimal_stakes(
        self,
        legs: List[Tuple[str, float]],  # (bookmaker, odds) per outcome
        bankroll: float,
    ) -> List[float]:
        """
        Compute optimal stakes for guaranteed profit.

        stake_i = bankroll × (1/odds_i) / overround

        Each leg returns exactly bankroll / overround regardless of which
        outcome wins, so every leg produces the same guaranteed return and the
        profit is bankroll × (1/overround - 1).

        Returns list of stakes in same order as legs.
        """
        if not legs:
            return []

        overround = sum(1.0 / odds for _, odds in legs if odds > 1.0)
        if overround <= 0.0:
            logger.warning("Cannot compute stakes: overround is %s", overround)
            return [0.0] * len(legs)

        stakes = []
        for _, odds in legs:
            if odds > 1.0:
                stake = bankroll * (1.0 / odds) / overround
            else:
                stake = 0.0
            stakes.append(stake)

        return stakes

    def profit_pct(self, overround: float) -> float:
        """Guaranteed profit % = (1/overround - 1) × 100."""
        if overround <= 0.0:
            return 0.0
        return (1.0 / overround - 1.0) * 100.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_odds_by_outcome(
        self, all_odds: List[BookOdds], outcomes: List[str], market: str
    ) -> Dict[str, BookOdds]:
        """
        For each requested outcome label, find the BookOdds entry with the
        highest decimal odds from the given market.
        """
        best: Dict[str, BookOdds] = {}
        for bo in all_odds:
            if bo.market != market:
                continue
            if bo.outcome not in outcomes:
                continue
            if bo.odds <= 1.0:
                continue
            prev = best.get(bo.outcome)
            if prev is None or bo.odds > prev.odds:
                best[bo.outcome] = bo
        return best

    def _scan_3way(
        self,
        fixture_id: int,
        fixture_desc: str,
        all_odds: List[BookOdds],
    ) -> Optional[SurebetOpportunity]:
        """Attempt to find a 3-way (1X2) arbitrage."""
        required = ["home", "draw", "away"]
        best = self._best_odds_by_outcome(all_odds, required, "1X2")

        if len(best) < 3:
            # Not all outcomes present — cannot form 3-way arb
            return None

        home_bo = best["home"]
        draw_bo = best["draw"]
        away_bo = best["away"]

        overround = 1.0 / home_bo.odds + 1.0 / draw_bo.odds + 1.0 / away_bo.odds

        profit_fraction = 1.0 / overround - 1.0
        if profit_fraction < self._min_profit_pct:
            return None

        leg_specs: List[Tuple[str, float]] = [
            (home_bo.bookmaker, home_bo.odds),
            (draw_bo.bookmaker, draw_bo.odds),
            (away_bo.bookmaker, away_bo.odds),
        ]
        stakes = self.optimal_stakes(leg_specs, self._bankroll)

        legs = [
            SurebetLeg(
                bookmaker=home_bo.bookmaker,
                outcome="home",
                odds=home_bo.odds,
                stake=stakes[0],
                implied=home_bo.implied,
                potential_return=stakes[0] * home_bo.odds,
            ),
            SurebetLeg(
                bookmaker=draw_bo.bookmaker,
                outcome="draw",
                odds=draw_bo.odds,
                stake=stakes[1],
                implied=draw_bo.implied,
                potential_return=stakes[1] * draw_bo.odds,
            ),
            SurebetLeg(
                bookmaker=away_bo.bookmaker,
                outcome="away",
                odds=away_bo.odds,
                stake=stakes[2],
                implied=away_bo.implied,
                potential_return=stakes[2] * away_bo.odds,
            ),
        ]

        guaranteed_profit = self._bankroll * profit_fraction

        opp = SurebetOpportunity(
            fixture_id=fixture_id,
            fixture_desc=fixture_desc,
            market="1X2",
            legs=legs,
            overround=overround,
            guaranteed_profit=guaranteed_profit,
            profit_pct=profit_fraction * 100.0,
            bankroll=self._bankroll,
        )
        logger.info("3-way arb found: %s", opp)
        return opp

    def _scan_2way(
        self,
        fixture_id: int,
        fixture_desc: str,
        all_odds: List[BookOdds],
    ) -> Optional[SurebetOpportunity]:
        """
        Attempt to find a 2-way arbitrage.

        Checks an explicit "2way" market (yes/no outcomes) if present.
        """
        required_2way = ["yes", "no"]
        best_explicit = self._best_odds_by_outcome(all_odds, required_2way, "2way")

        if len(best_explicit) < 2:
            return None

        yes_bo = best_explicit["yes"]
        no_bo = best_explicit["no"]
        overround = 1.0 / yes_bo.odds + 1.0 / no_bo.odds
        profit_fraction = 1.0 / overround - 1.0

        if profit_fraction < self._min_profit_pct:
            return None

        leg_specs: List[Tuple[str, float]] = [
            (yes_bo.bookmaker, yes_bo.odds),
            (no_bo.bookmaker, no_bo.odds),
        ]
        stakes = self.optimal_stakes(leg_specs, self._bankroll)

        legs = [
            SurebetLeg(
                bookmaker=yes_bo.bookmaker,
                outcome="yes",
                odds=yes_bo.odds,
                stake=stakes[0],
                implied=yes_bo.implied,
                potential_return=stakes[0] * yes_bo.odds,
            ),
            SurebetLeg(
                bookmaker=no_bo.bookmaker,
                outcome="no",
                odds=no_bo.odds,
                stake=stakes[1],
                implied=no_bo.implied,
                potential_return=stakes[1] * no_bo.odds,
            ),
        ]

        guaranteed_profit = self._bankroll * profit_fraction
        opp = SurebetOpportunity(
            fixture_id=fixture_id,
            fixture_desc=fixture_desc,
            market="2way",
            legs=legs,
            overround=overround,
            guaranteed_profit=guaranteed_profit,
            profit_pct=profit_fraction * 100.0,
            bankroll=self._bankroll,
        )
        logger.info("2-way arb found: %s", opp)
        return opp


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def scan_from_raw_odds(
    raw_odds: List[dict],
    bankroll: float = _DEFAULT_BANKROLL,
    min_profit_pct: float = _MIN_PROFIT_PCT,
) -> ScanResult:
    """
    One-shot convenience: parse raw odds dicts and scan for arbs.

    Each item in raw_odds must contain:
        fixture_id   : int
        fixture_desc : str
        bookmaker    : str
        outcome      : str  ("home"|"draw"|"away"|"yes"|"no")
        odds         : float (decimal)
        market       : str  ("1X2"|"2way")
    """
    fixtures_odds: Dict[int, List[BookOdds]] = {}
    fixture_descs: Dict[int, str] = {}

    for row in raw_odds:
        try:
            fid = int(row["fixture_id"])
            bo = BookOdds(
                bookmaker=str(row["bookmaker"]),
                market=str(row["market"]),
                outcome=str(row["outcome"]).lower(),
                odds=float(row["odds"]),
            )
            fixtures_odds.setdefault(fid, []).append(bo)
            if "fixture_desc" in row:
                fixture_descs[fid] = str(row["fixture_desc"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed raw odds row %s: %s", row, exc)

    scanner = SurebetScanner(
        min_profit_pct=min_profit_pct,
        bankroll=bankroll,
    )
    return scanner.scan_all(fixtures_odds, fixture_descs)


def format_opportunity(opp: SurebetOpportunity) -> str:
    """Multi-line pretty-print of a surebet opportunity."""
    lines = [
        "=" * 60,
        "  SUREBET OPPORTUNITY",
        f"  Fixture  : {opp.fixture_desc} (id={opp.fixture_id})",
        f"  Market   : {opp.market}",
        f"  Overround: {opp.overround:.6f}  (margin {(1.0 - opp.overround) * 100:.4f}%)",
        f"  Profit   : £{opp.guaranteed_profit:.2f}  ({opp.profit_pct:+.4f}%)",
        f"  Bankroll : £{opp.bankroll:.2f}",
        f"  Detected : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(opp.timestamp))}",
        "-" * 60,
    ]

    for i, leg in enumerate(opp.legs, start=1):
        lines.append(
            f"  Leg {i}: {leg.outcome.upper():6s}  "
            f"@ {leg.odds:6.3f}  "
            f"Stake £{leg.stake:8.2f}  "
            f"Return £{leg.potential_return:8.2f}  "
            f"[{leg.bookmaker}]"
        )

    lines.append(
        f"  Total staked: £{sum(leg.stake for leg in opp.legs):.2f}  "
        f"  Guaranteed return: £{opp.bankroll / opp.overround:.2f}"
    )
    lines.append("=" * 60)
    return "\n".join(lines)
