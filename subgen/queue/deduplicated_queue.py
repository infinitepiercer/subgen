"""Deduplicated priority queue for transcription tasks."""

import logging
import queue
import time
from threading import Lock


class DeduplicatedQueue(queue.PriorityQueue):
    """Queue that prevents duplicates, handles priority, and tracks status."""

    def __init__(self) -> None:
        super().__init__()
        self._queued: set[str] = set()       # Tracks task IDs waiting in queue
        self._processing: set[str] = set()   # Tracks task IDs currently being handled
        self._lock: Lock = Lock()

    def put(self, item: dict, block: bool = True, timeout: float | None = None) -> bool:
        with self._lock:
            task_id = item.get("path", "")
            if not task_id:
                logging.warning("Queue item missing 'path' key, skipping: %s", item)
                return False
            if task_id not in self._queued and task_id not in self._processing:
                # Priority: 0 (Detect), 1 (ASR), 2 (Transcribe)
                task_type = item.get("type", "transcribe")
                priority = 0 if task_type == "detect_language" else (1 if task_type == "asr" else 2)

                # PriorityQueue requires a tuple: (priority, tie_breaker, item)
                super().put((priority, time.time(), item), block, timeout)
                self._queued.add(task_id)
                return True
            return False

    def get(self, block: bool = True, timeout: float | None = None) -> dict:
        # PriorityQueue returns the tuple, we want just the item
        priority, timestamp, item = super().get(block, timeout)
        with self._lock:
            task_id = item.get("path", "")
            self._queued.discard(task_id)
            self._processing.add(task_id)
        return item

    def mark_done(self, item: dict) -> None:
        with self._lock:
            task_id = item.get("path", "")
            self._processing.discard(task_id)

    def is_idle(self) -> bool:
        with self._lock:
            return self.empty() and len(self._processing) == 0

    def is_active(self, task_id: str) -> bool:
        """Checks if a task_id is currently queued or processing."""
        with self._lock:
            return task_id in self._queued or task_id in self._processing

    def get_queued_tasks(self) -> list[str]:
        with self._lock:
            return list(self._queued)

    def get_processing_tasks(self) -> list[str]:
        with self._lock:
            return list(self._processing)


# Singleton queue instance
task_queue: DeduplicatedQueue = DeduplicatedQueue()
