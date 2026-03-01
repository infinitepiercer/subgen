"""Transcription service: language selection, queue submission, skip logic, and transcription."""

import logging
import os

import numpy as np
from language_code import LanguageCode

from subgen.config import (
    custom_regroup,
    force_detected_language_to,
    kwargs as whisper_kwargs,
    lrc_for_audio_files,
    limit_to_preferred_audio_languages,
    namesublang,
    only_skip_if_subgen_subtitle,
    preferred_audio_languages,
    should_whiser_detect_audio_language,
    skip_if_audio_track_is_in_list,
    skip_if_to_transcribe_sub_already_exist,
    skip_lang_codes_list,
    skip_unknown_language,
    skipifexternalsub,
    skipifinternalsublang,
    transcribe_or_translate,
    word_level_highlight,
)
from subgen.logging_setup import ProgressHandler
from subgen.media.audio import (
    find_default_audio_track_language,
    find_language_audio_track,
    get_audio_languages,
    get_audio_tracks,
    handle_multiple_audio_tracks,
)
from subgen.media.file_utils import has_audio, isAudioFileExtension, write_lrc
from subgen.models.whisper_model import delete_model, start_model
from subgen.queue.deduplicated_queue import task_queue
from subgen.services.subtitle import (
    appendLine,
    get_subtitle_languages,
    has_subtitle_language,
    has_subtitle_language_in_file,
    has_subtitle_of_language_in_folder,
    name_subtitle,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------


def choose_transcribe_language(
    file_path: str, forced_language: LanguageCode
) -> LanguageCode:
    """
    Determines the language to be used for transcription based on the provided
    file path and language preferences.

    Args:
        file_path: The path to the file for which the audio tracks are analyzed.
        forced_language: The language to force for transcription if specified.

    Returns:
        The language code to be used for transcription. It prioritizes the
        ``forced_language``, then the environment variable ``force_detected_language_to``,
        then the preferred audio language if available, and finally the default
        language of the audio tracks. Returns ``LanguageCode.NONE`` if no language
        preference is determined.
    """
    if forced_language:
        logger.debug("ENV FORCE_LANGUAGE is set: Forcing language to %s", forced_language)
        return forced_language

    if force_detected_language_to:
        logger.debug(
            "ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to %s",
            force_detected_language_to,
        )
        return force_detected_language_to

    audio_tracks = get_audio_tracks(file_path)

    preferred_track_language = find_language_audio_track(
        audio_tracks, preferred_audio_languages
    )

    if preferred_track_language:
        return preferred_track_language

    default_language = find_default_audio_track_language(audio_tracks)
    if default_language:
        logger.debug("Default language found: %s", default_language)
        return default_language

    return LanguageCode.NONE


# ---------------------------------------------------------------------------
# Queue submission entry-point
# ---------------------------------------------------------------------------


def gen_subtitles_queue(
    file_path: str,
    transcription_type: str,
    force_language: LanguageCode = LanguageCode.NONE,
    **kwargs,
) -> None:
    """Submit a file for subtitle generation via the task queue.

    BUG FIX: Removed unnecessary ``global task_queue`` declaration that was
    present in the original monolith.
    """
    # Check if this file is already in the queue or being processed
    if task_queue.is_active(file_path):
        logging.debug(
            "Ignored: %s is already queued or processing.",
            os.path.basename(file_path),
        )
        return

    if not has_audio(file_path):
        logging.debug("%s doesn't have any audio to transcribe!", file_path)
        return

    force_language = choose_transcribe_language(file_path, force_language)

    if should_skip_file(file_path, force_language):  # skip before wasting time detecting language
        return

    # Check if we would like to detect audio language when no language is specified.
    # Will return here again with a specified language from Whisper.
    if not force_language and should_whiser_detect_audio_language:
        # Make a detect-language task
        task_id = {"path": file_path, "type": "detect_language"}
        # Pass metadata info (kwargs) to the detect task
        task_id.update(kwargs)
        task_queue.put(task_id)
        return

    task = {
        "path": file_path,
        "transcribe_or_translate": transcription_type,
        "force_language": force_language,
    }
    # Pass metadata info (kwargs) to the transcribe task
    task.update(kwargs)

    task_queue.put(task)


# ---------------------------------------------------------------------------
# 7-condition skip gate
# ---------------------------------------------------------------------------


def should_skip_file(file_path: str, target_language: LanguageCode) -> bool:
    """
    Determines if subtitle generation should be skipped for a file.

    Args:
        file_path: Path to the media file.
        target_language: The desired language for transcription.

    Returns:
        True if the file should be skipped, False otherwise.
    """
    base_name = os.path.basename(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    if transcribe_or_translate in ("translate", "transcribe_and_translate"):
        target_language = LanguageCode.ENGLISH  # Force target language as English when translating

    # 1. Skip if it's an audio file and an LRC file already exists.
    if isAudioFileExtension(file_ext) and lrc_for_audio_files:
        lrc_path = os.path.join(os.path.dirname(file_path), f"{file_name}.lrc")
        if os.path.exists(lrc_path):
            logging.info("Skipping %s: LRC file already exists.", base_name)
            return True

    # 2. Skip if language detection failed and we are configured to skip unknowns.
    if skip_unknown_language and target_language == LanguageCode.NONE:
        logging.info(
            "Skipping %s: Unknown language and skip_unknown_language is enabled.",
            base_name,
        )
        return True

    # 3. Skip if a subtitle already exists in the target language.
    if skip_if_to_transcribe_sub_already_exist and has_subtitle_language(
        file_path, target_language
    ):
        lang_name = target_language.to_name()
        logging.info("Skipping %s: Subtitles already exist in %s.", base_name, lang_name)
        return True

    # 4. Skip if an internal subtitle exists in skipifinternalsublang language.
    if skipifinternalsublang and has_subtitle_language_in_file(
        file_path, skipifinternalsublang
    ):
        lang_name = skipifinternalsublang.to_name()
        logging.info(
            "Skipping %s: Internal subtitles in %s already exist.",
            base_name,
            lang_name,
        )
        return True

    # 5. Skip if an external subtitle exists in the namesublang language.
    if skipifexternalsub and namesublang and LanguageCode.is_valid_language(namesublang):
        external_lang = LanguageCode.from_string(namesublang)
        if has_subtitle_of_language_in_folder(file_path, external_lang):
            lang_name = external_lang.to_name()
            logging.info(
                "Skipping %s: External subtitles in %s already exist.",
                base_name,
                lang_name,
            )
            return True

    # 6. Skip if any subtitle language is in the skip list.
    if any(lang in skip_lang_codes_list for lang in get_subtitle_languages(file_path)):
        logging.info("Skipping %s: Contains a skipped subtitle language.", base_name)
        return True

    # 7. Audio track checks
    audio_langs = get_audio_languages(file_path)

    # 7a. Limit to preferred audio languages
    if limit_to_preferred_audio_languages:
        if not any(lang in preferred_audio_languages for lang in audio_langs):
            preferred_names = [lang.to_name() for lang in preferred_audio_languages]
            logging.info(
                "Skipping %s: No preferred audio tracks found (looking for %s)",
                base_name,
                ", ".join(preferred_names),
            )
            return True

    # 7b. Skip if the audio track language is in the skip list
    if any(lang in skip_if_audio_track_is_in_list for lang in audio_langs):
        logging.info("Skipping %s: Contains a skipped audio language.", base_name)
        return True

    return False


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------


def gen_subtitles(
    file_path: str,
    transcribe_or_translate_param: str,
    force_language: LanguageCode = LanguageCode.NONE,
) -> None:
    """Generates subtitles for a video file.

    Args:
        file_path: The path to the video file.
        transcribe_or_translate_param: The type of transcription or translation to perform.
        force_language: The language to force for transcription or translation.
            Default is ``LanguageCode.NONE``.

    BUG FIX: The except block now uses ``logging.error(..., exc_info=True)``
    instead of ``logging.info`` so that transcription failures are logged at the
    correct severity with a traceback.
    """
    try:
        start_model()

        # Check if the file is an audio file before trying to extract audio
        file_name, file_extension = os.path.splitext(file_path)
        is_audio_file = isAudioFileExtension(file_extension)

        data = file_path
        # Extract audio from the file if it has multiple audio tracks
        extracted_audio_file = handle_multiple_audio_tracks(file_path, force_language)
        if extracted_audio_file:
            data = extracted_audio_file.read()

        args = {}
        display_name = os.path.basename(file_path)
        args["progress_callback"] = ProgressHandler(display_name)

        if custom_regroup and custom_regroup.lower() != "default":
            args["regroup"] = custom_regroup

        args.update(whisper_kwargs)

        # Determine the actual Whisper task
        if transcribe_or_translate_param == "transcribe_and_translate":
            actual_task = "transcribe"
        else:
            actual_task = transcribe_or_translate_param

        # Import model at function level to get the current (possibly re-loaded) reference
        from subgen.models.whisper_model import model as current_model

        result = current_model.transcribe(
            data,
            language=force_language.to_iso_639_1(),
            task=actual_task,
            verbose=None,
            **args,
        )

        appendLine(result)

        # Pass 2: Translate non-English segments if using two-pass mode
        if transcribe_or_translate_param == "transcribe_and_translate":
            from subgen.services.translation import translate_segments, ensure_translation_models
            from subgen.config import translate_source_languages, detect_confidence_threshold, model_location, debug

            # Ensure translation models are downloaded (no-op after first call)
            source_langs = [lang.strip() for lang in translate_source_languages.split(',')]
            ensure_translation_models(source_langs, model_location)

            # Translate non-English segments (timestamps are never modified)
            logging.info("Pass 1: Transcription complete. Starting Pass 2: Translation...")
            translated_count = translate_segments(result, detect_confidence_threshold, debug)
            logging.info(f"Pass 2: Translated {translated_count} non-English segments to English")

        # If it is an audio file, write the LRC file
        if is_audio_file and lrc_for_audio_files:
            write_lrc(result, file_name + ".lrc")
        else:
            if not force_language:
                force_language = LanguageCode.from_string(result.language)
            result.to_srt_vtt(
                name_subtitle(file_path, force_language),
                word_level=word_level_highlight,
            )

    except Exception as e:
        # BUG FIX: was logging.info, now logging.error with traceback
        logging.error(
            "Error processing or transcribing %s in %s: %s",
            file_path,
            force_language,
            e,
            exc_info=True,
        )

    finally:
        delete_model()


# ---------------------------------------------------------------------------
# ASR worker function
# ---------------------------------------------------------------------------


def asr_task_worker(task_data: dict) -> None:
    """
    Worker function that processes ASR tasks from the queue.
    Called by ``transcription_worker`` when task type is ``'asr'``.

    BUG FIX: Now honours ``task_data.get('output', 'srt')`` so the caller can
    choose between ``srt``, ``vtt``, ``txt``, ``tsv``, or ``json`` output.
    """
    result = None
    task_id = task_data.get("path", "unknown")
    result_container = task_data.get("result_container")

    try:
        requested_task = task_data["task"]
        language = task_data["language"]
        video_file = task_data.get("video_file")
        initial_prompt = task_data.get("initial_prompt")
        file_content = task_data["audio_content"]
        encode = task_data["encode"]
        output_format = task_data.get("output", "srt")  # BUG FIX: support output format selection

        # Determine the actual Whisper task
        if requested_task == "transcribe_and_translate":
            actual_task = "transcribe"
        else:
            actual_task = requested_task

        start_model()

        args = {}
        display_name = os.path.basename(video_file) if video_file else task_id
        args["progress_callback"] = ProgressHandler(display_name)

        # Handle audio encoding
        if encode:
            args["audio"] = file_content
        else:
            args["audio"] = (
                np.frombuffer(file_content, np.int16)
                .flatten()
                .astype(np.float32)
                / 32768.0
            )
            args["input_sr"] = 16000

        if custom_regroup and custom_regroup.lower() != "default":
            args["regroup"] = custom_regroup

        args.update(whisper_kwargs)

        # Import model at function level to get the current (possibly re-loaded) reference
        from subgen.models.whisper_model import model as current_model

        # Perform transcription
        result = current_model.transcribe(
            task=actual_task, language=language, **args, verbose=None
        )
        appendLine(result)

        # Pass 2: Translate non-English segments if using two-pass mode
        if requested_task == "transcribe_and_translate":
            from subgen.services.translation import translate_segments, ensure_translation_models
            from subgen.config import translate_source_languages, detect_confidence_threshold, model_location, debug

            # Ensure translation models are downloaded (no-op after first call)
            source_langs = [lang.strip() for lang in translate_source_languages.split(',')]
            ensure_translation_models(source_langs, model_location)

            # Translate non-English segments (timestamps are never modified)
            logging.info("Pass 1: ASR transcription complete. Starting Pass 2: Translation...")
            translated_count = translate_segments(result, detect_confidence_threshold, debug)
            logging.info(f"Pass 2: Translated {translated_count} non-English segments to English")

        # Set result for blocking endpoint, using the requested output format
        if result_container:
            if output_format == "vtt":
                result_container.set_result(
                    result.to_srt_vtt(filepath=None, word_level=word_level_highlight, vtt=True)
                )
            elif output_format == "txt":
                result_container.set_result(result.to_txt(filepath=None))
            elif output_format == "tsv":
                result_container.set_result(result.to_tsv(filepath=None))
            elif output_format == "json":
                result_container.set_result(result.to_dict())
            else:
                # Default to SRT
                result_container.set_result(
                    result.to_srt_vtt(filepath=None, word_level=word_level_highlight)
                )

    except Exception as e:
        logging.error("Error processing ASR (ID: %s): %s", task_id, e, exc_info=True)
        if result_container:
            result_container.set_error(str(e))

    finally:
        delete_model()
