"""
Cross-Chain Arbitrage Scanner — Fiat vs Crypto Betting Markets.

Traditional bookmakers (Pinnacle, Bet365) and decentralized blockchain betting
protocols (SX.bet on Polygon, Polymarket) operate in separate ecosystems and
are slow to share information.  Price discrepancies between the two worlds can
create surebet opportunities: bet the home side on Pinnacle (fiat) and lay the
same side on SX.bet (crypto) when the implied probabilities sum to < 100%.

Profit model
------------
  overround = 1/fiat_odds + 1/crypto_odds
  gross_profit_pct = (1/overround − 1) × 100
  gas_drag_pct = gas_cost_usd / bankroll_usd × 100
  protocol_fee_pct = crypto_commission × 100
  net_profit_pct = gross_profit_pct − gas_drag_pct − protocol_fee_pct

Optimal stakes (same guaranteed return on both legs):
  fiat_stake = bankroll × (1/fiat_odds) / overround
  crypto_stake = bankroll × (1/crypto_odds) / overround

Usage
-----
    from engine.crosschain_arb import scan_crosschain

    result = scan_crosschain(
        odds_api_key="YOUR_KEY",
        sport_key="soccer_epl",
        bankroll_usd=1000.0,
    )
    for opp in result.opportunities:
        print(opp)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SX_BET_API = "https://api.sx.bet"
_POLYMARKET_API = "https://clob.polymarket.com"
_THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_DEFAULT_GAS_USD = 2.0
_DEFAULT_MIN_PROFIT_PCT = 0.5
_CRYPTO_COMMISSION = 0.02
_FIAT_COMMISSION = 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FiatOdds:
    """Odds from a traditional bookmaker."""

    bookmaker: str
    fixture_desc: str
    market: str  # "h2h"
    outcome: str  # "home" | "draw" | "away"
    decimal_odds: float
    implied_prob: float = field(init=False)

    def __post_init__(self) -> None:
        self.implied_prob = 1.0 / self.decimal_odds if self.decimal_odds > 1.0 else 1.0


@dataclass
class CryptoOdds:
    """Odds from a decentralized betting protocol."""

    protocol: str  # "sx_bet" | "polymarket" | "azuro"
    market_id: str
    fixture_desc: str
    outcome: str  # "home" | "draw" | "away"
    decimal_odds: float
    implied_prob: float = field(init=False)
    liquidity_usd: float = 0.0
    chain: str = "polygon"

    def __post_init__(self) -> None:
        self.implied_prob = 1.0 / self.decimal_odds if self.decimal_odds > 1.0 else 1.0


@dataclass
class CrossChainOpportunity:
    """A confirmed cross-chain arbitrage opportunity."""

    fixture_desc: str
    outcome: str
    fiat_leg: FiatOdds
    crypto_leg: CryptoOdds
    fiat_stake: float
    crypto_stake_usd: float
    gross_profit_pct: float
    gas_cost_usd: float
    net_profit_pct: float
    net_profit_usd: float
    is_profitable: bool
    overround: float
    bankroll_usd: float = 1000.0

    def __str__(self) -> str:
        return (
            f"CROSS-CHAIN ARB [{self.fixture_desc}] {self.outcome} "
            f"fiat@{self.fiat_leg.decimal_odds:.2f}({self.fiat_leg.bookmaker}) "
            f"crypto@{self.crypto_leg.decimal_odds:.2f}({self.crypto_leg.protocol}) "
            f"net={self.net_profit_pct:+.2f}% "
            f"(${self.net_profit_usd:+.2f}) gas=${self.gas_cost_usd:.2f}"
        )


@dataclass
class CrossChainScanResult:
    """Full scan output."""

    n_fiat_markets: int
    n_crypto_markets: int
    n_opportunities: int
    opportunities: List[CrossChainOpportunity]
    scan_duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"CrossChainScan: {self.n_fiat_markets} fiat × {self.n_crypto_markets} crypto "
            f"→ {self.n_opportunities} arbs in {self.scan_duration_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


class SXBetClient:
    """
    Client for SX.bet decentralized sports betting (Polygon).
    REST API at api.sx.bet — no authentication required for public market data.
    """

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    def _get(self, path: str, params: Optional[Dict] = None) -> dict:
        url = f"{_SX_BET_API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "FQE/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.warning("SXBet GET %s failed: %s", path, exc)
            return {}

    def get_active_markets(self, sport_id: int = 3) -> List[dict]:
        """GET /markets?status=ACTIVE — soccer is sportXeventId 3."""
        data = self._get(
            "/markets", {"status": "ACTIVE", "sportXeventId": str(sport_id)}
        )
        if isinstance(data, dict):
            return data.get("data", data.get("markets", []))
        if isinstance(data, list):
            return data
        return []

    def parse_markets(self, raw: List[dict]) -> List[CryptoOdds]:
        results: List[CryptoOdds] = []
        for m in raw:
            try:
                market_id = str(m.get("marketHash", m.get("id", "")))
                desc = str(
                    m.get("teamOneName", "Home") + " vs " + m.get("teamTwoName", "Away")
                )

                # SX.bet encodes odds as percentage implied probability × 10^20
                # or as outcomeOneDecimalOdds / outcomeTwoDecimalOdds directly
                for outcome_key, outcome_label in [
                    ("outcomeOneDecimalOdds", "home"),
                    ("outcomeTwoDecimalOdds", "away"),
                ]:
                    raw_odds = m.get(outcome_key)
                    if raw_odds is None:
                        # Try percentage format: percentageOdds is fraction × 10^20
                        pct = m.get("percentageOdds")
                        if pct and outcome_key.endswith("One"):
                            try:
                                implied = float(pct) / 1e20
                                raw_odds = 1.0 / implied if implied > 0 else None
                            except Exception:
                                continue
                    if raw_odds is None:
                        continue
                    dec_odds = float(raw_odds)
                    if dec_odds <= 1.01:
                        continue
                    results.append(
                        CryptoOdds(
                            protocol="sx_bet",
                            market_id=market_id,
                            fixture_desc=desc,
                            outcome=outcome_label,
                            decimal_odds=dec_odds,
                            liquidity_usd=float(m.get("liquidity", 0.0)),
                            chain="polygon",
                        )
                    )
            except Exception as exc:
                logger.debug("SXBet parse error: %s", exc)
        return results


class PolymarketClient:
    """
    Client for Polymarket CLOB (prediction market on Polygon).
    Binary markets with probability prices in [0, 1].
    """

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    def _get(self, path: str, params: Optional[Dict] = None) -> dict:
        url = f"{_POLYMARKET_API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": "FQE/1.0"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.warning("Polymarket GET %s failed: %s", path, exc)
            return {}

    def get_sports_markets(self, keyword: str = "soccer") -> List[dict]:
        """GET /markets with keyword filter."""
        data = self._get(
            "/markets", {"keyword": keyword, "active": "true", "limit": "200"}
        )
        if isinstance(data, dict):
            return data.get("data", data.get("markets", []))
        if isinstance(data, list):
            return data
        return []

    def parse_markets(self, raw: List[dict]) -> List[CryptoOdds]:
        results: List[CryptoOdds] = []
        for m in raw:
            try:
                market_id = str(m.get("condition_id", m.get("id", "")))
                question = str(m.get("question", ""))
                # Extract team names from question like "Will Arsenal win vs Chelsea?"
                desc = (
                    question.replace("Will ", "")
                    .replace(" win?", "")
                    .replace(" to win?", "")
                )

                tokens = m.get("tokens", [])
                for tok in tokens:
                    outcome_name = str(tok.get("outcome", "")).lower()
                    price = float(tok.get("price", 0.0))
                    if price <= 0.01 or price >= 1.0:
                        continue
                    dec_odds = 1.0 / price
                    # Map outcome to standard labels
                    label = "home"
                    if "draw" in outcome_name or "tie" in outcome_name:
                        label = "draw"
                    elif "away" in outcome_name or "no" in outcome_name:
                        label = "away"
                    results.append(
                        CryptoOdds(
                            protocol="polymarket",
                            market_id=market_id,
                            fixture_desc=desc,
                            outcome=label,
                            decimal_odds=dec_odds,
                            liquidity_usd=float(m.get("volume", 0.0)),
                            chain="polygon",
                        )
                    )
            except Exception as exc:
                logger.debug("Polymarket parse error: %s", exc)
        return results


class TheOddsAPIClient:
    """Fetches Pinnacle fiat odds from the-odds-api.com."""

    def __init__(self, api_key: str, timeout: int = 10) -> None:
        self._key = api_key
        self._timeout = timeout

    def _get(self, path: str, params: Dict) -> List[dict]:
        params["apiKey"] = self._key
        url = f"{_THE_ODDS_API_BASE}{path}?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FQE/1.0"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning("TheOddsAPI GET %s failed: %s", path, exc)
            return []

    def get_odds(
        self, sport_key: str = "soccer_epl", bookmakers: str = "pinnacle"
    ) -> List[dict]:
        return self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions": "uk,eu",
                "markets": "h2h",
                "bookmakers": bookmakers,
                "oddsFormat": "decimal",
            },
        )

    def parse_odds(self, raw: List[dict]) -> List[FiatOdds]:
        results: List[FiatOdds] = []
        for fixture in raw:
            desc = f"{fixture.get('home_team', 'Home')} vs {fixture.get('away_team', 'Away')}"
            home_team = fixture.get("home_team", "")
            for bm in fixture.get("bookmakers", []):
                bm_key = bm.get("key", "unknown")
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for oc in mkt.get("outcomes", []):
                        name = str(oc.get("name", ""))
                        price = float(oc.get("price", 0.0))
                        if price <= 1.0:
                            continue
                        if name == home_team:
                            outcome = "home"
                        elif name.lower() in ("draw", "tie"):
                            outcome = "draw"
                        else:
                            outcome = "away"
                        results.append(
                            FiatOdds(
                                bookmaker=bm_key,
                                fixture_desc=desc,
                                market="h2h",
                                outcome=outcome,
                                decimal_odds=price,
                            )
                        )
        return results


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class CrossChainArbScanner:
    """
    Detects arbitrage between fiat bookmakers and crypto betting protocols.
    """

    def __init__(
        self,
        min_profit_pct: float = _DEFAULT_MIN_PROFIT_PCT,
        gas_cost_usd: float = _DEFAULT_GAS_USD,
        bankroll_usd: float = 1000.0,
    ) -> None:
        self._min_profit = min_profit_pct
        self._gas = gas_cost_usd
        self._bankroll = bankroll_usd

    def match_markets(
        self,
        fiat: List[FiatOdds],
        crypto: List[CryptoOdds],
    ) -> List[Tuple[FiatOdds, CryptoOdds]]:
        """Match fiat and crypto odds for the same fixture + outcome."""
        pairs: List[Tuple[FiatOdds, CryptoOdds]] = []
        for fo in fiat:
            fiat_teams = self._extract_teams(fo.fixture_desc)
            for co in crypto:
                if not self._outcomes_match(fo.outcome, co.outcome):
                    continue
                crypto_desc_lower = co.fixture_desc.lower()
                if any(
                    team.lower() in crypto_desc_lower
                    for team in fiat_teams
                    if len(team) > 3
                ):
                    pairs.append((fo, co))
        return pairs

    def evaluate_pair(
        self,
        fiat: FiatOdds,
        crypto: CryptoOdds,
    ) -> Optional[CrossChainOpportunity]:
        overround = fiat.implied_prob + crypto.implied_prob
        if overround >= 1.0:
            return None  # no arb

        gross_pct = (1.0 / overround - 1.0) * 100.0
        gas_drag_pct = self._gas / self._bankroll * 100.0
        proto_fee_pct = _CRYPTO_COMMISSION * 100.0
        net_pct = gross_pct - gas_drag_pct - proto_fee_pct

        fiat_stake = self._bankroll * fiat.implied_prob / overround
        crypto_stake = self._bankroll * crypto.implied_prob / overround
        net_profit_usd = (
            self._bankroll * (1.0 / overround - 1.0)
            - self._gas
            - self._bankroll * _CRYPTO_COMMISSION
        )

        is_profitable = net_pct >= self._min_profit

        return CrossChainOpportunity(
            fixture_desc=fiat.fixture_desc,
            outcome=fiat.outcome,
            fiat_leg=fiat,
            crypto_leg=crypto,
            fiat_stake=round(fiat_stake, 2),
            crypto_stake_usd=round(crypto_stake, 2),
            gross_profit_pct=round(gross_pct, 3),
            gas_cost_usd=self._gas,
            net_profit_pct=round(net_pct, 3),
            net_profit_usd=round(net_profit_usd, 2),
            is_profitable=is_profitable,
            overround=round(overround, 6),
            bankroll_usd=self._bankroll,
        )

    def scan(
        self,
        fiat_odds: List[FiatOdds],
        crypto_odds: List[CryptoOdds],
    ) -> CrossChainScanResult:
        t0 = time.time()
        pairs = self.match_markets(fiat_odds, crypto_odds)
        opps: List[CrossChainOpportunity] = []

        for fo, co in pairs:
            opp = self.evaluate_pair(fo, co)
            if opp is not None and opp.is_profitable:
                opps.append(opp)

        opps.sort(key=lambda o: o.net_profit_pct, reverse=True)
        duration = time.time() - t0

        logger.info(
            "CrossChainScan: %d fiat × %d crypto → %d pairs → %d arbs in %.2fs",
            len(fiat_odds),
            len(crypto_odds),
            len(pairs),
            len(opps),
            duration,
        )
        return CrossChainScanResult(
            n_fiat_markets=len(fiat_odds),
            n_crypto_markets=len(crypto_odds),
            n_opportunities=len(opps),
            opportunities=opps,
            scan_duration_seconds=duration,
        )

    def _extract_teams(self, fixture_desc: str) -> List[str]:
        for sep in (" vs ", " - ", " v "):
            if sep in fixture_desc:
                parts = fixture_desc.split(sep, 1)
                return [p.strip() for p in parts]
        return [fixture_desc.strip()]

    def _outcomes_match(self, fiat_outcome: str, crypto_outcome: str) -> bool:
        fo = fiat_outcome.lower().strip()
        co = crypto_outcome.lower().strip()
        if fo == co:
            return True
        # Allow "yes" → "home", "no" → "away" mapping
        mapping = {"yes": "home", "no": "away", "1": "home", "2": "away", "x": "draw"}
        return mapping.get(fo, fo) == mapping.get(co, co)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def scan_crosschain(
    odds_api_key: str,
    sport_key: str = "soccer_epl",
    bankroll_usd: float = 1000.0,
    min_profit_pct: float = _DEFAULT_MIN_PROFIT_PCT,
    gas_cost_usd: float = _DEFAULT_GAS_USD,
) -> CrossChainScanResult:
    """
    One-shot: fetch fiat (Pinnacle via the-odds-api) + crypto (SX.bet + Polymarket),
    scan for cross-chain arbs.
    """
    fiat_client = TheOddsAPIClient(api_key=odds_api_key)
    sx_client = SXBetClient()
    poly_client = PolymarketClient()

    fiat_raw = fiat_client.get_odds(sport_key=sport_key, bookmakers="pinnacle")
    fiat_odds = fiat_client.parse_odds(fiat_raw)
    logger.info("Fetched %d fiat odds entries", len(fiat_odds))

    sx_raw = sx_client.get_active_markets(sport_id=3)
    sx_odds = sx_client.parse_markets(sx_raw)
    logger.info("Fetched %d SX.bet odds entries", len(sx_odds))

    poly_raw = poly_client.get_sports_markets(keyword="soccer")
    poly_odds = poly_client.parse_markets(poly_raw)
    logger.info("Fetched %d Polymarket odds entries", len(poly_odds))

    crypto_odds = sx_odds + poly_odds

    scanner = CrossChainArbScanner(
        min_profit_pct=min_profit_pct,
        gas_cost_usd=gas_cost_usd,
        bankroll_usd=bankroll_usd,
    )
    return scanner.scan(fiat_odds, crypto_odds)


def format_opportunity(opp: CrossChainOpportunity) -> str:
    """Multi-line pretty-print of a cross-chain arbitrage opportunity."""
    lines = [
        "=" * 60,
        f"CROSS-CHAIN ARB: {opp.fixture_desc}  [{opp.outcome.upper()}]",
        "-" * 60,
        f"  Fiat leg   : {opp.fiat_leg.bookmaker:15s} odds={opp.fiat_leg.decimal_odds:.3f}  "
        f"implied={opp.fiat_leg.implied_prob:.4f}  stake=£{opp.fiat_stake:.2f}",
        f"  Crypto leg : {opp.crypto_leg.protocol:15s} odds={opp.crypto_leg.decimal_odds:.3f}  "
        f"implied={opp.crypto_leg.implied_prob:.4f}  stake=${opp.crypto_stake_usd:.2f}  "
        f"chain={opp.crypto_leg.chain}",
        "-" * 60,
        f"  Overround  : {opp.overround:.6f}  (< 1.0 = arb exists)",
        f"  Gross      : {opp.gross_profit_pct:+.2f}%",
        f"  Gas cost   : ${opp.gas_cost_usd:.2f}  (drag: {opp.gas_cost_usd / opp.bankroll_usd * 100:.2f}%)",
        f"  Net profit : {opp.net_profit_pct:+.2f}%  (${opp.net_profit_usd:+.2f})",
        "=" * 60,
    ]
    return "\n".join(lines)
