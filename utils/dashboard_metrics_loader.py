from __future__ import annotations


def compute_dashboard_metrics(df):

    if df.empty:
        return {
            "bankroll": 0,
            "roi": 0,
            "yield": 0,
            "hit_rate": 0,
        }

    stake_sum = df["stake"].sum()

    profit = (df["profit"]).sum()

    wins = (df["profit"] > 0).sum()

    bets = len(df)

    roi = profit / stake_sum if stake_sum else 0

    yield_metric = roi

    hit_rate = wins / bets if bets else 0

    return {
        "bankroll": round(profit, 2),
        "roi": round(roi * 100, 2),
        "yield": round(yield_metric * 100, 2),
        "hit_rate": round(hit_rate * 100, 2),
    }
