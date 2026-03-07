from engine.probability_calibrator import ProbabilityCalibrator


def test_probability_calibrator_runs():

    calibrator = ProbabilityCalibrator()

    result = calibrator.calibrate(
        {
            "home_win": 0.50,
            "draw": 0.25,
            "away_win": 0.25,
        }
    )

    total = result["home_win"] + result["draw"] + result["away_win"]

    assert 0.999 <= total <= 1.001
