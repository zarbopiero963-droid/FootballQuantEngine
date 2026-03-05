from concurrent.futures import ThreadPoolExecutor


class WorkerPool:

    def __init__(self, workers=4):
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def submit(self, fn, *args, **kwargs):
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=True)
