"""Whisper model lifecycle management: loading, cleanup, and VRAM release."""

import ctypes
import ctypes.util
import gc
import logging
import os
from threading import Lock, Timer

import stable_whisper
import torch

from subgen.config import (
    whisper_model as _whisper_model_name,
    model_location,
    transcribe_device,
    whisper_threads,
    concurrent_transcriptions,
    compute_type,
    clear_vram_on_complete,
    model_cleanup_delay,
)
from subgen.queue.deduplicated_queue import task_queue

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
model = None
model_cleanup_timer: Timer | None = None
model_cleanup_lock: Lock = Lock()


def start_model() -> None:
    """Load the Whisper model into memory if it is not already loaded."""
    global model
    if model is None:
        logging.debug("Model was purged, need to re-create")
        model = stable_whisper.load_faster_whisper(
            _whisper_model_name,
            download_root=model_location,
            device=transcribe_device,
            cpu_threads=whisper_threads,
            num_workers=concurrent_transcriptions,
            compute_type=compute_type,
        )


def schedule_model_cleanup() -> None:
    """Schedule model cleanup with a delay to allow concurrent requests."""
    global model_cleanup_timer

    with model_cleanup_lock:
        # Cancel any existing timer
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()
            logging.debug("Cancelled previous model cleanup timer")

        # Schedule a new cleanup timer
        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()
        logging.debug("Model cleanup scheduled in %s seconds", model_cleanup_delay)


def perform_model_cleanup() -> None:
    """Actually perform the model cleanup."""
    global model, model_cleanup_timer

    with model_cleanup_lock:
        logging.debug("Executing scheduled model cleanup")

        if clear_vram_on_complete and task_queue.is_idle():
            logging.debug("Queue idle; clearing model from memory.")
            if model:
                try:
                    model.model.unload_model()
                    del model
                    model = None
                    logging.info("Model unloaded from memory")
                except Exception as exc:
                    logging.error("Error unloading model: %s", exc)

            if transcribe_device.lower() == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logging.debug("CUDA cache cleared.")
                except Exception as exc:
                    logging.error("Error clearing CUDA cache: %s", exc)
        else:
            logging.debug(
                "Queue not idle or clear_vram disabled; skipping model cleanup"
            )

        if os.name != "nt":  # don't garbage collect on Windows
            gc.collect()
            libc_name = ctypes.util.find_library("c")
            if libc_name is not None:
                try:
                    ctypes.CDLL(libc_name).malloc_trim(0)
                except (OSError, AttributeError):
                    pass

        model_cleanup_timer = None


def delete_model() -> None:
    """
    Only schedules a cleanup timer if the system is actually idle.
    This prevents unnecessary timer resets when a large batch is being processed.
    """
    # 1. If we aren't supposed to clear VRAM, don't bother with timers at all.
    if not clear_vram_on_complete:
        return

    # 2. Only schedule cleanup if the queue is empty AND no other workers are processing.
    if task_queue.is_idle():
        schedule_model_cleanup()
    else:
        # If there are items left in the queue, we simply do nothing.
        # The very last worker to finish the last item will trigger the timer.
        logging.debug(
            "Tasks still in queue or processing; skipping model cleanup scheduling."
        )
