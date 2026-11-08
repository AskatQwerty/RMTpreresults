"""Use an in-process thread executor for joblib's threading backend on this Windows host."""
import os
import sys
from pathlib import Path

if os.environ.get("RMT_THREAD_EXECUTOR") == "1" and ("--worker" in sys.argv or os.environ.get("RMT_EXECUTOR_TEST") == "1"):
    from concurrent.futures import ThreadPoolExecutor
    from joblib import register_parallel_backend
    from joblib._parallel_backends import ThreadingBackend

    class LocalThreads(ThreadingBackend):
        executor = None

        def submit(self, func, callback=None):
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self._n_jobs)
            future = self.executor.submit(func)
            if callback is not None:
                future.add_done_callback(callback)
            return future

        def retrieve_result_callback(self, future):
            return future.result()

        def terminate(self):
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None

    register_parallel_backend("threading", LocalThreads)
    if "--worker" in sys.argv:
        # The runner includes this invocation directory in the return ZIP.
        launch = Path(sys.argv[sys.argv.index("--worker") + 1])
        (launch / "forest_thread_executor.py.txt").write_bytes(Path(__file__).read_bytes())
        print("Random forest execution: joblib threading via ThreadPoolExecutor, original n_jobs.", flush=True)
