"""Bounded optional work; a stuck predictor never occupies the trading thread."""
from copy import deepcopy
import logging
from queue import Queue, Full, Empty
from threading import Thread, Lock

log = logging.getLogger(__name__)


class ExitObservationQueue:
    def __init__(self, callback, capacity=64):
        self.callback = callback
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.pending = set()
        self.closed = False
        self.thread = None
        self.dropped = 0

    def submit(self, key, args):
        with self.lock:
            if self.closed or key in self.pending:
                return False
            try:
                self.queue.put_nowait((key, deepcopy(args)))
            except Full:
                self.dropped += 1
                return False
            self.pending.add(key)
            if self.thread is None:
                self.thread = Thread(target=self._run, name="fx-exit-observer", daemon=True)
                self.thread.start()
            return True

    def _run(self):
        while True:
            try:
                key, args = self.queue.get(timeout=.1)
            except Empty:
                if self.closed:
                    return
                continue
            try:
                if not self.closed:
                    self.callback(*args)
            except Exception:
                log.exception("optional exit observation failed")
            finally:
                with self.lock:
                    self.pending.discard(key)
                self.queue.task_done()

    def close(self):
        # Never wait for a hung estimator during trading-worker shutdown.
        self.closed = True

    def status(self):
        with self.lock:
            return {"capacity": self.queue.maxsize, "queued": self.queue.qsize(),
                    "pending": len(self.pending), "dropped": self.dropped, "closed": self.closed}
