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


def log_startup_config() -> None:
    """Print all configuration values at startup in a clean, readable format."""
    from subgen import subgen_version
    from subgen.config import (
        asr_engine, parakeet_model_name, ngram_lm_alpha, parakeet_beam_size,
        transcribe_device, whisper_model, whisper_threads, concurrent_transcriptions,
        compute_type, model_location, webhookport, debug,
        transcribe_or_translate, translate_source_languages, detect_confidence_threshold,
        detect_language_length, detect_language_offset, force_detected_language_to,
        should_whisper_detect_audio_language,
        namesublang, subtitle_language_naming_type, word_level_highlight,
        custom_regroup, min_subtitle_duration, normalize_audio, lrc_for_audio_files,
        append, show_in_subname_subgen, show_in_subname_model,
        procaddedmedia, procmediaonplay,
        skipifinternalsublang, skipifexternalsub, skip_if_to_transcribe_sub_already_exist,
        only_skip_if_subgen_subtitle, skip_unknown_language,
        skip_if_language_is_not_set_but_subtitles_exist,
        skip_lang_codes_list, skip_if_audio_track_is_in_list,
        preferred_audio_languages, limit_to_preferred_audio_languages,
        plexserver, plextoken, plex_queue_next_episode, plex_queue_season, plex_queue_series,
        jellyfinserver, jellyfintoken,
        use_path_mapping, path_mapping_from, path_mapping_to,
        transcribe_folders, monitor,
        clear_vram_on_complete, model_cleanup_delay, asr_timeout,
        filter_subtitles,
        enable_diarization, diarization_model,
        docker_status,
    )

    def _mask(val: str) -> str:
        """Mask tokens/secrets for log output."""
        if not val or val == "token here":
            return "(not set)"
        return val[:4] + "****"

    def _lang_list(langs: list) -> str:
        return ", ".join(str(l) for l in langs) if langs else "(none)"

    sep = "=" * 62
    logging.info(sep)
    logging.info(f"  SUBGEN v{subgen_version}  ({docker_status})")
    logging.info(sep)

    logging.info(f"  ASR ENGINE             : {asr_engine}")
    if asr_engine == 'parakeet':
        logging.info(f"    Parakeet Model     : {parakeet_model_name}")
        logging.info(f"    Device             : {transcribe_device}")
        logging.info(f"    N-gram LM Alpha    : {ngram_lm_alpha}")
        logging.info(f"    Beam Size          : {parakeet_beam_size} ({'beam search' if parakeet_beam_size > 1 else 'greedy'})")
    else:
        logging.info(f"    Whisper Model      : {whisper_model}")
        logging.info(f"    Device             : {transcribe_device}")
        logging.info(f"    Compute Type       : {compute_type}")
        logging.info(f"    Threads            : {whisper_threads}")
        logging.info(f"    Concurrent Jobs    : {concurrent_transcriptions}")
        logging.info(f"    Model Path         : {model_location}")

    logging.info("  TRANSCRIPTION")
    logging.info(f"    Mode               : {transcribe_or_translate}")
    if transcribe_or_translate == "transcribe_and_translate":
        logging.info(f"    Translate Langs    : {translate_source_languages}")
        logging.info(f"    Confidence Thresh  : {detect_confidence_threshold}")
    logging.info(f"    Detect Language    : {should_whisper_detect_audio_language}")
    logging.info(f"    Detect Length      : {detect_language_length}s (offset: {detect_language_offset}s)")
    logging.info(f"    Normalize Audio    : {normalize_audio}")
    if force_detected_language_to:
        logging.info(f"    Force Language     : {force_detected_language_to}")

    logging.info("  SUBTITLE OUTPUT")
    logging.info(f"    Language Name      : {namesublang or '(auto)'}")
    logging.info(f"    Naming Type        : {subtitle_language_naming_type}")
    logging.info(f"    Regroup            : {custom_regroup}")
    logging.info(f"    Min Duration       : {min_subtitle_duration}s" if min_subtitle_duration > 0 else "    Min Duration       : (disabled)")
    logging.info(f"    Word Highlight     : {word_level_highlight}")
    logging.info(f"    LRC for Audio      : {lrc_for_audio_files}")
    logging.info(f"    Show 'subgen'      : {show_in_subname_subgen}")
    logging.info(f"    Show Model Name    : {show_in_subname_model}")
    logging.info(f"    Append Line        : {append}")
    logging.info(f"    Filter Subtitles   : {filter_subtitles}")

    logging.info("  DIARIZATION")
    logging.info(f"    Enabled            : {enable_diarization}")
    logging.info(f"    Model              : {diarization_model}")

    logging.info("  SKIP / FILTER")
    logging.info(f"    Process on Add     : {procaddedmedia}")
    logging.info(f"    Process on Play    : {procmediaonplay}")
    logging.info(f"    Skip Internal Lang : {skipifinternalsublang or '(disabled)'}")
    logging.info(f"    Skip External Subs : {skipifexternalsub}")
    logging.info(f"    Skip Target Exists : {skip_if_to_transcribe_sub_already_exist}")
    logging.info(f"    Skip Subgen Only   : {only_skip_if_subgen_subtitle}")
    logging.info(f"    Skip Unknown Lang  : {skip_unknown_language}")
    logging.info(f"    Skip No Lang+Subs  : {skip_if_language_is_not_set_but_subtitles_exist}")
    logging.info(f"    Skip Languages     : {_lang_list(skip_lang_codes_list)}")
    logging.info(f"    Skip Audio Langs   : {_lang_list(skip_if_audio_track_is_in_list)}")
    logging.info(f"    Preferred Audio    : {_lang_list(preferred_audio_languages)}")
    logging.info(f"    Limit to Preferred : {limit_to_preferred_audio_languages}")

    logging.info("  SERVERS")
    logging.info(f"    Plex               : {plexserver}  (token: {_mask(plextoken)})")
    logging.info(f"    Jellyfin           : {jellyfinserver}  (token: {_mask(jellyfintoken)})")
    if plex_queue_next_episode or plex_queue_season or plex_queue_series:
        logging.info(f"    Plex Queue         : episode={plex_queue_next_episode} season={plex_queue_season} series={plex_queue_series}")

    logging.info("  PATHS")
    if use_path_mapping:
        logging.info(f"    Path Mapping       : {path_mapping_from} -> {path_mapping_to}")
    else:
        logging.info(f"    Path Mapping       : (disabled)")
    if transcribe_folders:
        logging.info(f"    Watch Folders      : {transcribe_folders}")
        logging.info(f"    Monitor            : {monitor}")

    logging.info("  RESOURCES")
    logging.info(f"    Clear VRAM         : {clear_vram_on_complete} (delay: {model_cleanup_delay}s)")
    logging.info(f"    ASR Timeout        : {asr_timeout}s")
    logging.info(f"    Webhook Port       : {webhookport}")
    logging.info(f"    Debug              : {debug}")
    logging.info(sep)
