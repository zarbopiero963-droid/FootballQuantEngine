from engine.job_runner import JobRunner
from engine.scheduler import Scheduler


def main():

    scheduler = Scheduler(interval_minutes=10)

    runner = JobRunner()

    scheduler.add_job(runner.run_cycle)

    scheduler.run()


if __name__ == "__main__":
    main()
