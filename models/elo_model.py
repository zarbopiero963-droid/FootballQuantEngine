

class EloModel:

    def probability(self, elo_diff):

        return 1 / (1 + 10 ** (-elo_diff / 400))
