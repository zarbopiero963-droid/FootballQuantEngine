import math

import pandas as pd


class BacktestMetrics:

    def calculate(self, backtest_df):

        if backtest_df.empty:
            return {}

        df = backtest_df.copy()

        df["selected_market"] = None

        for index, row in df.iterrows():

            probs = {
                "home": row["home_prob"] if pd.notna(row["home_prob"]) else -1,
                "draw": row["draw_prob"] if pd.notna(row["draw_prob"]) else -1,
                "away": row["away_prob"] if pd.notna(row["away_prob"]) else -1,
            }

            selected_market = max(probs, key=probs.get)
            df.at[index, "selected_market"] = selected_market

        df["won"] = 0

        df.loc[
            (df["selected_market"] == "home") & (df["actual_home_win"] == 1),
            "won",
        ] = 1
        df.loc[
            (df["selected_market"] == "draw") & (df["actual_draw"] == 1),
            "won",
        ] = 1
        df.loc[
            (df["selected_market"] == "away") & (df["actual_away_win"] == 1),
            "won",
        ] = 1

        def resolve_odds(row):
            if row["selected_market"] == "home":
                return row.get("home_odds", 2.0)
            if row["selected_market"] == "draw":
                return row.get("draw_odds", 3.0)
            return row.get("away_odds", 2.5)

        df["selected_odds"] = df.apply(resolve_odds, axis=1)

        stake = 1.0

        df["profit"] = df.apply(
            lambda row: (
                (row["selected_odds"] - 1) * stake if row["won"] == 1 else -stake
            ),
            axis=1,
        )

        total_bets = len(df)
        total_profit = df["profit"].sum()
        total_staked = total_bets * stake

        roi = total_profit / total_staked if total_staked else 0.0
        yield_value = roi
        hit_rate = df["won"].mean() if total_bets else 0.0

        brier_scores = []
        log_losses = []

        for _, row in df.iterrows():

            probs = {
                "home": row["home_prob"] if pd.notna(row["home_prob"]) else 0.0,
                "draw": row["draw_prob"] if pd.notna(row["draw_prob"]) else 0.0,
                "away": row["away_prob"] if pd.notna(row["away_prob"]) else 0.0,
            }

            actual = {
                "home": row["actual_home_win"],
                "draw": row["actual_draw"],
                "away": row["actual_away_win"],
            }

            brier = 0.0

            for key in ("home", "draw", "away"):
                brier += (probs[key] - actual[key]) ** 2

            brier_scores.append(brier)

            selected_prob = max(
                probs["home"],
                probs["draw"],
                probs["away"],
            )

            selected_prob = min(max(selected_prob, 1e-15), 1 - 1e-15)

            actual_selected = int(row["won"])

            if actual_selected == 1:
                log_loss_value = -math.log(selected_prob)
            else:
                log_loss_value = -math.log(1 - selected_prob)

            log_losses.append(log_loss_value)

        bankroll_history = []
        bankroll = 100.0

        for profit in df["profit"]:
            bankroll += profit
            bankroll_history.append(bankroll)

        accuracy_history = []
        cumulative_correct = 0

        for index, won in enumerate(df["won"], start=1):
            cumulative_correct += won
            accuracy_history.append(cumulative_correct / index)

        max_bankroll = bankroll_history[0] if bankroll_history else 100.0
        drawdowns = []

        for value in bankroll_history:
            max_bankroll = max(max_bankroll, value)
            drawdown = (value - max_bankroll) / max_bankroll if max_bankroll else 0.0
            drawdowns.append(drawdown)

        max_drawdown = min(drawdowns) if drawdowns else 0.0

        return {
            "roi": roi,
            "yield": yield_value,
            "hit_rate": hit_rate,
            "brier_score": sum(brier_scores) / len(brier_scores),
            "log_loss": sum(log_losses) / len(log_losses),
            "bankroll_history": bankroll_history,
            "accuracy_history": accuracy_history,
            "max_drawdown": max_drawdown,
            "total_profit": total_profit,
            "total_staked": total_staked,
            "total_bets": total_bets,
        }
