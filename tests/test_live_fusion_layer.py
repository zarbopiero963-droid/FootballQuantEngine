from quant.fusion.live_fusion_layer import LiveFusionLayer


def test_live_fusion_layer_merges_sources():
    layer = LiveFusionLayer()

    fixtures = [
        {
            "fixture_id": "u1",
            "home_team": "Inter Milan",
            "away_team": "Roma",
        }
    ]

    odds_map = {
        "u1": {
            "home": 1.90,
            "draw": 3.40,
            "away": 4.20,
        }
    }

    understat_stats = {
        "Inter": {
            "xg_for": 1.85,
            "xg_against": 0.90,
            "xpts": 2.10,
            "shots": 14.0,
            "shots_on_target": 5.3,
            "finishing_delta": 0.10,
            "defensive_delta": 0.05,
            "form_score": 0.80,
        },
        "Roma": {
            "xg_for": 1.30,
            "xg_against": 1.20,
            "xpts": 1.50,
            "shots": 11.0,
            "shots_on_target": 4.1,
            "finishing_delta": 0.03,
            "defensive_delta": -0.02,
            "form_score": 0.55,
        },
    }

    fused = layer.fuse(fixtures, odds_map, understat_stats)

    assert isinstance(fused, list)
    assert len(fused) == 1
    assert fused[0]["fixture_id"] == "u1"
    assert fused[0]["home_team_key"] == "inter"
    assert fused[0]["bookmaker_odds_home"] == 1.90
