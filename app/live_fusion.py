from __future__ import annotations

from quant.fusion.live_fusion_runner import LiveFusionRunner


class AppLiveFusion:

    def __init__(self):
        self.runner = LiveFusionRunner()

    def run(self):
        return self.runner.run()
