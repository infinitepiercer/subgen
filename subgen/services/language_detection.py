"""Language detection service: workers for uploaded audio and local files."""

import logging

import numpy as np
from language_code import LanguageCode

from subgen.config import (
    detect_language_length,
    detect_language_offset,
    kwargs,
    transcribe_or_translate,
)
from subgen.logging_setup import ProgressHandler
from subgen.media.audio import extract_audio_segment_to_memory
from subgen.models.whisper_model import delete_model, start_model
from subgen.queue.deduplicated_queue import task_queue


# ---------------------------------------------------------------------------
# Detect language from uploaded audio (queue worker)
# ---------------------------------------------------------------------------


def detect_language_from_upload(task_data: dict) -> None:
    """
    Worker function that processes detect-language tasks from uploaded audio.
    Sets the result in the ``result_container`` when complete.

    BUG FIX (CRITICAL): The original code referenced an undefined ``progress``
    variable on line 970. This is replaced with ``ProgressHandler(task_id)``
    which is the correct callable progress handler.
    """
    detected_language = LanguageCode.NONE
    task_id = task_data.get("path", "unknown")
    result_container = task_data.get("result_container")

    try:
        video_file = task_data.get("video_file")
        file_content = task_data["audio_content"]
        encode = task_data["encode"]
        detect_lang_length = task_data["detect_lang_length"]
        detect_lang_offset = task_data["detect_lang_offset"]

        logging.info(
            "Detecting language for '%s' (%ss starting at %ss) - ID: %s"
            if video_file
            else "Detecting language (%ss starting at %ss) - ID: %s",
            *(
                (video_file, detect_lang_length, detect_lang_offset, task_id)
                if video_file
                else (detect_lang_length, detect_lang_offset, task_id)
            ),
        )

        start_model()

        args = {}
        # BUG FIX: was `args['progress_callback'] = progress` which referenced
        # an undefined variable. Now uses ProgressHandler(task_id).
        args["progress_callback"] = ProgressHandler(task_id)

        # Handle audio extraction
        if encode:
            from subgen.media.audio import extract_audio_segment_from_content

            audio_bytes = extract_audio_segment_from_content(
                file_content,
                detect_lang_offset,
                detect_lang_length,
            )
            args["audio"] = audio_bytes
            args["input_sr"] = 16000
        else:
            args["audio"] = (
                np.frombuffer(file_content, np.int16)
                .flatten()
                .astype(np.float32)
                / 32768.0
            )
            args["input_sr"] = 16000

        args.update(kwargs)

        # Import model at function level to get the current (possibly re-loaded) reference
        from subgen.models.whisper_model import model as current_model

        detected_language = LanguageCode.from_name(
            current_model.transcribe(**args).language
        )
        language_code = detected_language.to_iso_639_1()

        logging.info(
            "Detected language: %s (%s) - ID: %s",
            detected_language.to_name(),
            language_code,
            task_id,
        )

        # Set the result for the blocking endpoint
        if result_container:
            result_container.set_result(
                {
                    "detected_language": detected_language.to_name(),
                    "language_code": language_code,
                }
            )

    except Exception as e:
        logging.error(
            "Error detecting language (ID: %s) for '%s': %s"
            if task_data.get("video_file")
            else "Error detecting language (ID: %s): %s",
            *(
                (task_id, task_data.get("video_file"), e)
                if task_data.get("video_file")
                else (task_id, e)
            ),
            exc_info=True,
        )
        if result_container:
            result_container.set_error(str(e))

    finally:
        delete_model()


# ---------------------------------------------------------------------------
# Detect language for a local file (queue worker)
# ---------------------------------------------------------------------------


def detect_language_task(path: str, original_task_data: dict | None = None) -> None:
    """
    Worker function that detects language for a local file.
    Then queues the actual transcription with the detected language.
    """
    detected_language = LanguageCode.NONE

    try:
        logging.info(
            "Detecting language of file: %s (%ss starting at %ss)",
            path,
            detect_language_length,
            detect_language_offset,
        )

        start_model()

        audio_segment = extract_audio_segment_to_memory(
            path,
            detect_language_offset,
            int(detect_language_length),
        )

        # Import model at function level to get the current (possibly re-loaded) reference
        from subgen.models.whisper_model import model as current_model

        detected_language = LanguageCode.from_name(
            current_model.transcribe(audio_segment).language
        )

        logging.info("Detected language: %s", detected_language.to_name())

    except Exception as e:
        logging.error("Error detecting language for file: %s", e, exc_info=True)

    finally:
        delete_model()

        # Queue transcription with detected language
        task_data = {
            "path": path,
            "type": "transcribe",
            "transcribe_or_translate": transcribe_or_translate,
            "force_language": detected_language,
        }

        # Carry over metadata (Plex IDs, etc.) from the original task
        if original_task_data:
            for key, value in original_task_data.items():
                if key not in task_data:
                    task_data[key] = value

        if task_queue.put(task_data):
            logging.debug("Queued transcription for detected language: %s", path)
        else:
            logging.debug("Transcription already queued/processing for: %s", path)
