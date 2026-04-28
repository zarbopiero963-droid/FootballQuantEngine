from __future__ import annotations

import pandas as pd


class AIFeatureGenerator:

    def generate(self, features_df, manual_context=None):

        if features_df.empty:
            return pd.DataFrame()

        df = features_df.copy()

        if "weighted_form_score" in df.columns and "goals_for_avg" in df.columns:
            df["ai_confidence_score"] = (df["weighted_form_score"] * 0.6) + (
                df["goals_for_avg"] * 0.4
            )
        else:
            df["ai_confidence_score"] = 0.0

        if "clean_sheet_rate" in df.columns and "btts_rate" in df.columns:
            df["ai_defensive_index"] = df["clean_sheet_rate"] - df["btts_rate"]
        else:
            df["ai_defensive_index"] = 0.0

        if manual_context is not None and not manual_context.empty:
            context_cols = [c for c in manual_context.columns if c in df.columns]
            for col in context_cols:
                df[f"ai_context_{col}"] = df[col]

        return df
