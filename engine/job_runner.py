from quant.services.quant_job_runner import QuantJobRunner


class JobRunner:

    def __init__(self):
        self.quant_runner = QuantJobRunner()

    def run_cycle(self):
        return self.quant_runner.run_cycle()
