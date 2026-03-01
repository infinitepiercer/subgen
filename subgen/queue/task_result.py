"""Task result storage for blocking endpoints with TTL-based cleanup."""

import logging
import time
from threading import Lock, Event, Thread

TASK_RESULT_TTL = 300  # 5 minutes
CLEANUP_INTERVAL = 60  # 60 seconds


class TaskResult:
    """Stores the result of a queued task for blocking retrieval."""

    def __init__(self) -> None:
        self.result = None
        self.error: str | None = None
        self.done: Event = Event()
        self.completed_at: float | None = None

    def set_result(self, result: object) -> None:
        self.result = result
        self.completed_at = time.monotonic()
        self.done.set()

    def set_error(self, error: str) -> None:
        self.error = error
        self.completed_at = time.monotonic()
        self.done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until result is ready. Returns True if completed, False if timeout."""
        return self.done.wait(timeout)

    def is_expired(self) -> bool:
        """Return True if this result was completed more than TASK_RESULT_TTL seconds ago."""
        if self.completed_at is None:
            return False
        return (time.monotonic() - self.completed_at) >= TASK_RESULT_TTL


# Dictionary to store task results keyed by task_id
task_results: dict[str, TaskResult] = {}
task_results_lock: Lock = Lock()


def cleanup_expired_task_results() -> None:
    """Remove expired task results from the global dictionary."""
    with task_results_lock:
        expired_keys = [
            key for key, task_result in task_results.items()
            if task_result.is_expired()
        ]
        for key in expired_keys:
            del task_results[key]
        if expired_keys:
            logging.debug(
                "Cleaned up %d expired task result(s)", len(expired_keys)
            )


def _cleanup_loop() -> None:
    """Background loop that periodically cleans up expired task results."""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            cleanup_expired_task_results()
        except Exception as exc:
            logging.error("Error during task result cleanup: %s", exc)


# Start the background cleanup thread as a daemon so it won't block shutdown
_cleanup_thread = Thread(target=_cleanup_loop, daemon=True, name="task-result-cleanup")
_cleanup_thread.start()
