from config.constants import EDGE_THRESHOLD


class EdgeDetector:

    def calculate_edge(self, probability, odds):

        if odds <= 0:
            return 0

        implied_probability = 1 / odds

        return probability - implied_probability

    def is_value_bet(self, probability, odds):

        edge = self.calculate_edge(probability, odds)

        return edge > EDGE_THRESHOLD
