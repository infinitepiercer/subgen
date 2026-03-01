import logging
import sys
import time

from subgen.constants import SUPPRESSED_LOG_PATTERNS, SILENCED_LOGGERS


class MultiplePatternsFilter(logging.Filter):
    """Filter that suppresses noisy log lines matching known patterns."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Return False if any of the patterns are found, True otherwise
        return not any(pattern in record.getMessage() for pattern in SUPPRESSED_LOG_PATTERNS)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string (MM:SS or H:MM:SS)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressHandler:
    """Callable progress handler for model.transcribe() that throttles progress logging."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.interval = 5

    def __call__(self, seek: float, total: float) -> None:
        from subgen.config import docker_status, debug
        from subgen.queue.deduplicated_queue import task_queue

        if docker_status == 'Docker' or debug:
            current_time = time.time()
            if self.last_print_time == 0 or (current_time - self.last_print_time) >= self.interval:
                self.last_print_time = current_time

                # 1. Math for Metrics
                pct = int((seek / total) * 100) if total > 0 else 0
                elapsed = current_time - self.start_time
                speed = seek / elapsed if elapsed > 0 else 0
                eta = (total - seek) / speed if speed > 0 else 0

                # 2. Get Queue Stats
                proc = len(task_queue.get_processing_tasks())
                queued = len(task_queue.get_queued_tasks())

                # 3. Alignment Logic
                # :<40  = Left-align, 40 chars wide (Filename)
                # :>3   = Right-align, 3 chars wide (Percentage)
                # :>5   = Right-align, 5 chars wide (Seconds)
                # :>5   = Right-align, 5 chars wide (Time strings)

                clean_name = (self.filename[:37] + '..') if len(self.filename) > 40 else self.filename

                logging.info(
                    f"[ {clean_name:<40}] {pct:>3}% | "
                    f"{int(seek):>5}/{int(total):<5}s "
                    f"[{_format_duration(elapsed):>5}<{_format_duration(eta):>5}, {speed:>5.2f}s/s] | "
                    f"Jobs: {proc} processing, {queued} queued"
                )


def configure_logging() -> None:
    """Configure the root logger with appropriate level, formatting, filters, and silenced loggers."""
    from subgen.config import debug

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # This removes the ,123 part
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)  # Set the logger level

    for handler in logger.handlers:
        handler.addFilter(MultiplePatternsFilter())

    for logger_name in SILENCED_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
