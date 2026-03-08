from quant.clv.clv_runner import CLVRunner


class AppCLVController:

    def __init__(self):
        self.runner = CLVRunner()

    def run(self, bankroll: float = 1000.0):
        return self.runner.run(bankroll=bankroll)
