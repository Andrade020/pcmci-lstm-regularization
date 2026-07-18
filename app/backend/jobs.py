"""
Thread-based background job registry.

PCMCI + LSTM training takes minutes on CPU, so `/run` starts a worker thread and
returns a job id immediately; the front-end polls `/jobs/{id}`. We deliberately
use plain threads (not joblib/loky) — loky workers deadlock with torch on
Windows, and one job at a time is fine for a single-user research app.
"""
import threading
import time
import traceback
import uuid


class Job:
    def __init__(self, job_id, meta):
        self.id = job_id
        self.meta = meta                # {"dataset": ..., "config": ...}
        self.status = "pending"         # pending | running | done | error
        self.message = "queued"
        self.progress = 0.0
        self.result = None              # JSON-serializable payload (no _arrays)
        self.figures = None             # {name: base64 png}
        self.error = None
        self.created = time.time()

    def public(self):
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "error": self.error,
            "meta": self.meta,
        }


class JobRegistry:
    def __init__(self, max_keep: int = 20):
        self._jobs = {}
        self._lock = threading.Lock()
        self._max_keep = max_keep

    def create(self, meta) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, meta)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_if_needed()
        return job

    def get(self, job_id) -> Job:
        with self._lock:
            return self._jobs.get(job_id)

    def _evict_if_needed(self):
        if len(self._jobs) <= self._max_keep:
            return
        # drop oldest finished jobs first
        finished = sorted(
            (j for j in self._jobs.values() if j.status in ("done", "error")),
            key=lambda j: j.created)
        while len(self._jobs) > self._max_keep and finished:
            del self._jobs[finished.pop(0).id]

    def run_async(self, job, target):
        """Start `target(job)` in a daemon thread. `target` sets job fields."""
        def wrapper():
            job.status = "running"
            job.message = "starting"
            try:
                target(job)
                job.status = "done"
                job.message = "done"
                job.progress = 1.0
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "failed"
                traceback.print_exc()

        threading.Thread(target=wrapper, daemon=True).start()
