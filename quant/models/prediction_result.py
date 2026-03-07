from dataclasses import dataclass, field


@dataclass
class PredictionResult:
    fixture_id: str
    home_team: str
    away_team: str
    market: str
    probability: float
    fair_odds: float
    bookmaker_odds: float
    market_edge: float
    model_edge: float
    confidence: float
    agreement: float
    decision: str
    details: dict = field(default_factory=dict)
