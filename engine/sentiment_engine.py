"""
sentiment_engine.py
===================
Production-quality NLP sentiment analysis module for football betting.

Scans news articles, tweets, and forum posts for negative signals about a
team (injuries, internal tension, unpaid wages, illness, suspensions) and
computes a Negative Sentiment Score.  This score adjusts the team's Elo
rating and goal-rate lambda before the market has priced in the information.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional VADER
# ---------------------------------------------------------------------------
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as _VADER

    _VADER_AVAILABLE = True
except Exception:
    _VADER = None  # type: ignore[assignment,misc]
    _VADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries  (pattern, weight)
# ---------------------------------------------------------------------------

# Each list entry: (keyword_regex_pattern, weight)
_INJURY_SIGNALS: List[Tuple[str, float]] = [
    (r"\binjur\w*\b", 0.8),
    (r"\binfluenza\b|\bfluenza\b|\bfever\b|\bfebbre\b", 0.9),
    (r"\btraining\s+ground\s+incident\b", 0.7),
    (r"\bdoubtful\b|\bin\s+doubt\b", 0.6),
    (r"\brunning\s+test\b|\bfitness\s+test\b", 0.5),
    (r"\bout\s+of\s+squad\b|\bnot\s+train\w*\b", 0.8),
    (r"\bfracture\b|\bsprain\b|\bstrain\b|\bhamstring\b|\btorn\b", 0.9),
    (r"\binfortunio\b|\bsaltato\b|\bassenza\b", 0.8),
]

_TENSION_SIGNALS: List[Tuple[str, float]] = [
    (r"\bdressing\s+room\s+(row|fallout|crisis)\b", 0.9),
    (r"\bfell\s+out\b|\bfalling\s+out\b|\bclash\b", 0.7),
    (r"\bwage\s+(dispute|arrears|unpaid)\b|\bstipendi\s+non\s+pagati\b", 0.85),
    (r"\btraining\s+ground\s+fight\b|\blitigi\b", 0.8),
    (r"\btension\b|\brifts?\b|\bdiscord\b|\bfriction\b", 0.6),
    (r"\baxed\b|\bdropped\b|\bleft\s+out\s+of\s+squad\b", 0.65),
]

_NEGATIVE_GENERAL: List[Tuple[str, float]] = [
    (r"\bcrisis\b|\bcollapse\b|\bdisaster\b|\bchaos\b", 0.7),
    (r"\bmorale\s+low\b|\bconfidence\s+low\b", 0.65),
    (r"\bban\b|\bsuspend\w*\b|\bred\s+card\s+ban\b", 0.6),
    (r"\bsacked\b|\bfired\b|\bmanager\s+(crisis|uncertainty)\b", 0.75),
]

_POSITIVE_SIGNALS: List[Tuple[str, float]] = [
    (r"\bfit\b|\bfull\s+fitness\b|\bcleared\b|\brecovered\b", -0.5),
    (r"\bback\s+in\s+training\b|\bexpected\s+to\s+play\b", -0.4),
    (r"\bconfident\b|\bhigh\s+spirits\b|\bgreat\s+shape\b", -0.35),
]

# ---------------------------------------------------------------------------
# Pre-compile all patterns once at import time for performance
# ---------------------------------------------------------------------------
_COMPILED_INJURY: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(pat, re.IGNORECASE), w) for pat, w in _INJURY_SIGNALS
]
_COMPILED_TENSION: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(pat, re.IGNORECASE), w) for pat, w in _TENSION_SIGNALS
]
_COMPILED_NEGATIVE: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(pat, re.IGNORECASE), w) for pat, w in _NEGATIVE_GENERAL
]
_COMPILED_POSITIVE: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(pat, re.IGNORECASE), w) for pat, w in _POSITIVE_SIGNALS
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextItem:
    """A single piece of text (tweet, article snippet, forum post)."""

    text: str
    source: str  # "twitter" | "news" | "forum"
    team_mentions: List[str]  # which teams this text mentions
    timestamp: float  # unix epoch (for recency decay)
    language: str = "en"  # "en" | "it" | "es" | "de" | "fr"


@dataclass
class SentimentScore:
    """Sentiment result for a single text item."""

    text: str
    team: str
    raw_score: float  # in [-1, 1]; negative = bad news
    keyword_hits: List[str]  # which keywords matched
    vader_score: Optional[float]  # VADER compound if available
    final_score: float  # weighted blend
    category: str  # "INJURY" | "TENSION" | "NEGATIVE" | "POSITIVE" | "NEUTRAL"


@dataclass
class TeamSentimentReport:
    """Aggregated sentiment for one team over a time window."""

    team: str
    n_items: int
    weighted_score: float  # time-decayed aggregate in [-1, 1]
    elo_adjustment: float  # how many Elo points to subtract (negative = bad)
    lambda_multiplier: float  # multiply team's lambda by this (< 1.0 = weaker)
    signal_strength: str  # "STRONG_NEGATIVE" | "WEAK_NEGATIVE" | "NEUTRAL" | "POSITIVE"
    top_signals: List[str]  # top 3 most impactful text snippets
    computed_at: float

    def __str__(self) -> str:
        return (
            f"Sentiment [{self.team}] score={self.weighted_score:+.3f} "
            f"elo_adj={self.elo_adjustment:+.1f} "
            f"λ_mult={self.lambda_multiplier:.3f} "
            f"signal={self.signal_strength}"
        )


# ---------------------------------------------------------------------------
# SentimentEngine
# ---------------------------------------------------------------------------


class SentimentEngine:
    """
    NLP-based negative sentiment detector for pre-match football analysis.

    Scoring pipeline:
      1. Keyword scan: each pattern match adds weight × pattern_weight to raw_score.
      2. VADER score: if nltk available, blend 40% VADER + 60% keyword score.
      3. Recency decay: w(t) = exp(-lambda × age_hours), half-life = 24h.
      4. Team aggregate: weighted sum of item scores.

    Elo adjustment:
      elo_adj = weighted_score × (-80)  → max −80 Elo points for score=-1
      (score must be < -0.30 to trigger any adjustment)

    Lambda multiplier:
      If weighted_score < -0.30:  mult = max(0.70, 1.0 + weighted_score × 0.40)
      Else: mult = 1.0
    """

    def __init__(self, half_life_hours: float = 24.0) -> None:
        if half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive")
        self._lambda = math.log(2) / half_life_hours
        self._vader = _VADER() if _VADER_AVAILABLE else None
        logger.debug(
            "SentimentEngine initialised  half_life=%.1fh  decay_lambda=%.6f  vader=%s",
            half_life_hours,
            self._lambda,
            _VADER_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_text(self, item: TextItem, team: str) -> SentimentScore:
        """
        Score a single text item for a given team.

        Steps:
          1. Run all keyword lists (injury, tension, negative, positive).
          2. Sum weighted scores, clamp to [-1, 1].
          3. If VADER available, blend: final = 0.6 × keyword + 0.4 × vader_compound.
          4. Classify category from dominant signal type.
        """
        raw_score, keyword_hits, dominant_category = self._keyword_score(item.text)

        vader_compound: Optional[float] = None
        if self._vader is not None:
            try:
                vader_compound = self._vader.polarity_scores(item.text)["compound"]
            except Exception as exc:  # pragma: no cover
                logger.warning("VADER scoring failed: %s", exc)

        if vader_compound is not None:
            final_score = 0.6 * raw_score + 0.4 * vader_compound
        else:
            final_score = raw_score

        # Clamp final score to [-1, 1]
        final_score = max(-1.0, min(1.0, final_score))

        logger.debug(
            "score_text team=%s  raw=%.3f  vader=%s  final=%.3f  cat=%s  hits=%d",
            team,
            raw_score,
            f"{vader_compound:.3f}" if vader_compound is not None else "N/A",
            final_score,
            dominant_category,
            len(keyword_hits),
        )

        return SentimentScore(
            text=item.text,
            team=team,
            raw_score=raw_score,
            keyword_hits=keyword_hits,
            vader_score=vader_compound,
            final_score=final_score,
            category=dominant_category,
        )

    def aggregate(
        self,
        items: List[TextItem],
        team: str,
        reference_ts: Optional[float] = None,
    ) -> TeamSentimentReport:
        """
        Aggregate scored items into a TeamSentimentReport.

        Only items mentioning the team (case-insensitive) are included.
        Weights: w_i = exp(-lambda × age_hours_i)
        weighted_score = sum(w_i × score_i) / sum(w_i)
        """
        if reference_ts is None:
            reference_ts = time.time()

        team_lower = team.lower()

        # Filter to items that mention this team
        relevant: List[TextItem] = [
            it
            for it in items
            if any(t.lower() == team_lower for t in it.team_mentions)
            or team_lower in it.text.lower()
        ]

        logger.info(
            "aggregate team=%s  total_items=%d  relevant=%d",
            team,
            len(items),
            len(relevant),
        )

        if not relevant:
            return TeamSentimentReport(
                team=team,
                n_items=0,
                weighted_score=0.0,
                elo_adjustment=0.0,
                lambda_multiplier=1.0,
                signal_strength="NEUTRAL",
                top_signals=[],
                computed_at=reference_ts,
            )

        scored: List[Tuple[float, float, str]] = []  # (weight, score, snippet)
        for it in relevant:
            sent = self.score_text(it, team)
            w = self._recency_weight(it.timestamp, reference_ts)
            snippet = it.text[:120].strip()
            scored.append((w, sent.final_score, snippet))

        total_weight = sum(w for w, _, _ in scored)
        if total_weight < 1e-12:
            # Extremely stale data — treat as neutral
            logger.warning(
                "aggregate: total recency weight near zero for team=%s", team
            )
            weighted_score = 0.0
        else:
            weighted_score = sum(w * s for w, s, _ in scored) / total_weight

        # Clamp aggregate score
        weighted_score = max(-1.0, min(1.0, weighted_score))

        # Derive Elo adjustment and lambda multiplier
        elo_adj = self.elo_adjustment(weighted_score)
        lam_mult = self.lambda_multiplier(weighted_score)

        # Signal strength classification
        if weighted_score <= -0.60:
            signal_strength = "STRONG_NEGATIVE"
        elif weighted_score < -0.30:
            signal_strength = "WEAK_NEGATIVE"
        elif weighted_score > 0.20:
            signal_strength = "POSITIVE"
        else:
            signal_strength = "NEUTRAL"

        # Top 3 most impactful snippets (sorted by abs(weight × score) desc)
        sorted_signals = sorted(scored, key=lambda x: abs(x[0] * x[1]), reverse=True)
        top_signals = [snip for _, _, snip in sorted_signals[:3]]

        report = TeamSentimentReport(
            team=team,
            n_items=len(relevant),
            weighted_score=round(weighted_score, 6),
            elo_adjustment=round(elo_adj, 2),
            lambda_multiplier=round(lam_mult, 6),
            signal_strength=signal_strength,
            top_signals=top_signals,
            computed_at=reference_ts,
        )
        logger.info("aggregate result: %s", report)
        return report

    def elo_adjustment(self, weighted_score: float) -> float:
        """Elo delta = max(-80, weighted_score × 80) if score < -0.30, else 0."""
        if weighted_score >= -0.30:
            return 0.0
        raw = weighted_score * 80.0  # weighted_score is negative, so result is negative
        return max(-80.0, raw)

    def lambda_multiplier(self, weighted_score: float) -> float:
        """Returns multiplier in [0.70, 1.0] based on negative score."""
        if weighted_score >= -0.30:
            return 1.0
        return max(0.70, 1.0 + weighted_score * 0.40)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _keyword_score(self, text: str) -> Tuple[float, List[str], str]:
        """
        Scan all keyword lists. Return (raw_score, hit_patterns, dominant_category).
        Clamp raw_score to [-1, 1].
        """
        raw_score = 0.0
        keyword_hits: List[str] = []

        # Track per-category contribution for dominant-category logic
        category_scores: Dict[str, float] = defaultdict(float)

        for compiled_list, category in (
            (_COMPILED_INJURY, "INJURY"),
            (_COMPILED_TENSION, "TENSION"),
            (_COMPILED_NEGATIVE, "NEGATIVE"),
            (_COMPILED_POSITIVE, "POSITIVE"),
        ):
            for pattern, weight in compiled_list:
                matches = pattern.findall(text)
                if matches:
                    hit_count = len(matches)
                    # Cap multiple hits of the same pattern at 2× to avoid runaway scores
                    effective_hits = min(hit_count, 2)
                    delta = weight * effective_hits
                    raw_score += delta
                    category_scores[category] += abs(delta)
                    keyword_hits.append(pattern.pattern)

        # Clamp to [-1, 1]
        raw_score = max(-1.0, min(1.0, raw_score))

        # Determine dominant category
        if not category_scores:
            dominant_category = "NEUTRAL"
        else:
            dominant_category = max(category_scores, key=lambda k: category_scores[k])

        return raw_score, keyword_hits, dominant_category

    def _recency_weight(self, timestamp: float, reference_ts: float) -> float:
        """exp(-lambda × age_hours). Min weight = 0.01."""
        age_seconds = reference_ts - timestamp
        age_hours = age_seconds / 3600.0
        # If timestamp is in the future, clamp age to 0
        age_hours = max(0.0, age_hours)
        weight = math.exp(-self._lambda * age_hours)
        return max(0.01, weight)


# ---------------------------------------------------------------------------
# NewsAPI convenience client
# ---------------------------------------------------------------------------


class NewsAPIClient:
    """
    Thin client for newsapi.org to fetch recent football articles.

    Usage:
        client = NewsAPIClient(api_key="YOUR_KEY")
        items = client.fetch_team_news("Arsenal", days_back=2)
    """

    _BASE = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, timeout: int = 8) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._key = api_key
        self._timeout = timeout

    def fetch_team_news(self, team: str, days_back: int = 2) -> List[TextItem]:
        """
        Fetch recent articles mentioning team.

        URL: GET /v2/everything?q={team}&language=en&sortBy=publishedAt&apiKey={key}
        Parse: articles[].title + " " + articles[].description as text.
        Convert publishedAt (ISO string) to unix timestamp.
        Return list of TextItem with source="news", team_mentions=[team].
        Uses urllib.request. Returns [] on any error (logs warning).
        """
        params = urllib.parse.urlencode(
            {
                "q": team,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": self._key,
            }
        )
        url = f"{self._BASE}?{params}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FootballQuantEngine/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                raw = response.read()
                payload = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.warning(
                "NewsAPIClient HTTP error fetching '%s': %s %s",
                team,
                exc.code,
                exc.reason,
            )
            return []
        except urllib.error.URLError as exc:
            logger.warning(
                "NewsAPIClient URL error fetching '%s': %s", team, exc.reason
            )
            return []
        except Exception as exc:
            logger.warning(
                "NewsAPIClient unexpected error fetching '%s': %s", team, exc
            )
            return []

        articles = payload.get("articles") or []
        items: List[TextItem] = []
        cutoff_ts = time.time() - days_back * 86400.0

        for article in articles:
            try:
                title = article.get("title") or ""
                description = article.get("description") or ""
                text = f"{title} {description}".strip()
                if not text:
                    continue

                published_at_str: str = article.get("publishedAt") or ""
                if published_at_str:
                    # ISO 8601: "2024-03-15T14:30:00Z"
                    published_at_str = published_at_str.rstrip("Z")
                    dt = datetime.fromisoformat(published_at_str).replace(
                        tzinfo=timezone.utc
                    )
                    timestamp = dt.timestamp()
                else:
                    timestamp = time.time()

                if timestamp < cutoff_ts:
                    continue

                items.append(
                    TextItem(
                        text=text,
                        source="news",
                        team_mentions=[team],
                        timestamp=timestamp,
                        language="en",
                    )
                )
            except Exception as exc:
                logger.debug("NewsAPIClient: skipping malformed article: %s", exc)
                continue

        logger.info(
            "NewsAPIClient fetched %d usable articles for '%s' (days_back=%d)",
            len(items),
            team,
            days_back,
        )
        return items


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def analyse_team(
    team: str,
    texts: List[str],
    half_life_hours: float = 24.0,
) -> TeamSentimentReport:
    """
    One-shot: wrap raw strings into TextItems (source='manual', now timestamp),
    score them, return aggregate report.
    """
    now = time.time()
    items: List[TextItem] = [
        TextItem(
            text=t,
            source="manual",
            team_mentions=[team],
            timestamp=now,
            language="en",
        )
        for t in texts
        if t and t.strip()
    ]
    engine = SentimentEngine(half_life_hours=half_life_hours)
    return engine.aggregate(items, team, reference_ts=now)
