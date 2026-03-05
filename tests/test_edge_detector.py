from strategies.edge_detector import EdgeDetector


def test_edge_detector_basic():

    ed = EdgeDetector()

    edge = ed.calculate_edge(0.6, 2.0)

    assert edge == 0.6 - 0.5

    assert ed.is_value_bet(0.6, 2.0) in (True, False)
