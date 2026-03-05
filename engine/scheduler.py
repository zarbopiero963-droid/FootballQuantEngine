import time
from datetime import datetime


class Scheduler:

    def __init__(self, interval_minutes=10):
        self.interval = interval_minutes * 60
        self.jobs = []

    def add_job(self, func):
        self.jobs.append(func)

    def run(self):

        while True:

            start = datetime.utcnow()

            for job in self.jobs:
                job()

            elapsed = (datetime.utcnow() - start).total_seconds()

            sleep_time = self.interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
