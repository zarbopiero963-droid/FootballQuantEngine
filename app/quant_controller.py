from quant.services.quant_job_runner import QuantJobRunner


class AppQuantController:

    def __init__(self):
        self.runner = QuantJobRunner()

    def run_quant_cycle(self):
        return self.runner.run_cycle()
