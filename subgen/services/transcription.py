"""Transcription service: language selection, queue submission, skip logic, and transcription."""

import logging
import os
import re

import numpy as np
from language_code import LanguageCode

from subgen.config import (
    asr_engine,
    compute_type,
    custom_regroup,
    enable_diarization,
    filter_subtitles,
    force_detected_language_to,
    kwargs as whisper_kwargs,
    lrc_for_audio_files,
    limit_to_preferred_audio_languages,
    min_subtitle_duration,
    namesublang,
    normalize_audio as normalize_audio_enabled,
    only_skip_if_subgen_subtitle,
    preferred_audio_languages,
    should_whiser_detect_audio_language,
    skip_if_audio_track_is_in_list,
    skip_if_to_transcribe_sub_already_exist,
    skip_lang_codes_list,
    skip_unknown_language,
    skipifexternalsub,
    skipifinternalsublang,
    transcribe_device,
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
# Model lifecycle functions are imported conditionally based on asr_engine.
# Direct imports are deferred to the functions that need them.
from subgen.queue.deduplicated_queue import task_queue
from subgen.services.subtitle import (
    appendLine,
    get_subtitle_languages,
    has_subtitle_language,
    has_subtitle_language_in_file,
    has_subtitle_of_language_in_folder,
    name_subtitle,
)
from subgen.services.subtitle_filter import filter_segments

logger = logging.getLogger(__name__)

_UNEXPECTED_KWARG_RE = re.compile(r"got an unexpected keyword argument '(\w+)'")


def _transcribe_with_kwarg_filter(model, **kwargs):
    """Call model.transcribe(), automatically stripping unsupported kwargs.

    Some SUBGEN_KWARGS entries (e.g. ``nonspeech_skip``) may not be accepted by
    the current versions of stable-ts-whisperless / faster-whisper.  Rather
    than maintaining a static allow-list we let the call fail, parse the
    offending parameter name from the TypeError, remove it, and retry.
    """
    while True:
        try:
            return model.transcribe(**kwargs)
        except TypeError as exc:
            match = _UNEXPECTED_KWARG_RE.search(str(exc))
            if match and match.group(1) in kwargs:
                bad = match.group(1)
                del kwargs[bad]
                logging.warning(
                    "Removed unsupported SUBGEN_KWARGS key '%s' — "
                    "not accepted by transcribe()", bad,
                )
            else:
                raise


# ---------------------------------------------------------------------------
# ASR engine conditional model lifecycle helpers
# ---------------------------------------------------------------------------


def _start_model() -> None:
    """Load the appropriate ASR model based on the configured engine."""
    if asr_engine == 'parakeet':
        from subgen.models.parakeet_model import start_model as start_parakeet
        start_parakeet()
    else:
        from subgen.models.whisper_model import start_model as start_whisper
        start_whisper()


def _delete_model() -> None:
    """Schedule cleanup for the appropriate ASR model based on the configured engine."""
    if asr_engine == 'parakeet':
        from subgen.models.parakeet_model import delete_model as delete_parakeet
        delete_parakeet()
    else:
        from subgen.models.whisper_model import delete_model as delete_whisper
        delete_whisper()


# ---------------------------------------------------------------------------
# Parakeet transcription backend
# ---------------------------------------------------------------------------


_PARAKEET_CHUNK_SECONDS: int = 600  # 10 minutes per chunk
_PARAKEET_CHUNK_OVERLAP: int = 10   # seconds of overlap between chunks


def _get_audio_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _split_audio_chunks(
    audio_path: str, chunk_seconds: int, overlap: int = 0,
) -> list[tuple[str, float]]:
    """Split an audio file into overlapping chunks using ffmpeg.

    Returns list of (temp_file_path, actual_start_offset) tuples.
    The overlap prevents words at chunk boundaries from being lost.
    """
    import subprocess
    import tempfile

    duration = _get_audio_duration(audio_path)
    if duration <= 0 or duration <= chunk_seconds:
        return [(audio_path, 0.0)]

    step = chunk_seconds - overlap
    chunks: list[tuple[str, float]] = []
    offset = 0.0
    while offset < duration:
        chunk_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        chunk_file.close()
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ss", str(offset),
             "-t", str(chunk_seconds), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", chunk_file.name],
            capture_output=True, timeout=120,
        )
        chunks.append((chunk_file.name, offset))
        offset += step

    return chunks


def _transcribe_parakeet(audio_data: object, language: str, task: str) -> object:
    """Transcribe using NVIDIA Parakeet-TDT model via NeMo.

    Accepts audio as a file path (str), raw bytes, or a numpy array.
    Returns a WhisperResult-compatible object via the result adapter.

    Long audio files are automatically split into chunks to avoid CUDA OOM.
    Inference runs in fp16 via torch.cuda.amp.autocast for reduced VRAM usage.
    """
    import tempfile
    import wave

    import torch

    from subgen.models.parakeet_model import model as parakeet_model
    from subgen.models.result_adapter import parakeet_output_to_whisper_result

    cleanup_path: str | None = None

    # Parakeet requires a file path -- convert other formats to a temp WAV.
    if isinstance(audio_data, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_data)
            audio_path = f.name
        cleanup_path = audio_path
    elif isinstance(audio_data, str):
        audio_path = audio_data
    elif hasattr(audio_data, '__array__'):  # numpy array
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name
        # Write as 16-bit PCM WAV at 16 kHz mono
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        cleanup_path = audio_path
    else:
        raise TypeError(f"Unsupported audio data type: {type(audio_data)}")

    chunks: list[tuple[str, float]] = []
    try:
        chunks = _split_audio_chunks(
            audio_path, _PARAKEET_CHUNK_SECONDS, _PARAKEET_CHUNK_OVERLAP,
        )
        is_chunked = len(chunks) > 1 or chunks[0][0] != audio_path

        if is_chunked:
            logger.info(
                "Audio split into %d chunks of %ds each (%ds overlap)",
                len(chunks), _PARAKEET_CHUNK_SECONDS, _PARAKEET_CHUNK_OVERLAP,
            )

        all_word_timestamps: list[dict] = []
        full_text_parts: list[str] = []

        use_fp16 = (
            transcribe_device.lower() == "cuda"
            and torch.cuda.is_available()
            and compute_type in ("auto", "float16", "int8_float16")
        )

        for i, (chunk_path, actual_offset) in enumerate(chunks):
            if is_chunked:
                logger.info("Transcribing chunk %d/%d (offset %.1fs)", i + 1, len(chunks), actual_offset)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = parakeet_model.transcribe([chunk_path], timestamps=True)

            chunk_output = output[0]
            chunk_text = getattr(chunk_output, "text", "") or ""
            timestamp_data = getattr(chunk_output, "timestamp", {}) or {}
            word_ts = timestamp_data.get("word", [])

            # For chunks after the first, skip words in the overlap region
            # (they were already captured more accurately by the previous chunk).
            if i > 0 and word_ts and _PARAKEET_CHUNK_OVERLAP > 0:
                word_ts = [
                    w for w in word_ts
                    if float(w.get("start", 0.0)) >= _PARAKEET_CHUNK_OVERLAP
                ]
                # Rebuild chunk_text from filtered words so it stays aligned
                # with word_timestamps (the original chunk_text still contains
                # the overlap region's words).
                chunk_text = " ".join(
                    w.get("word", "").strip() for w in word_ts if w.get("word", "").strip()
                )

            # Offset timestamps by the chunk's actual start position
            if actual_offset > 0 and word_ts:
                for w in word_ts:
                    w["start"] = float(w.get("start", 0.0)) + actual_offset
                    w["end"] = float(w.get("end", 0.0)) + actual_offset

            all_word_timestamps.extend(word_ts)
            full_text_parts.append(chunk_text)

        # Build a combined result using the adapter
        if is_chunked and all_word_timestamps:
            # Create a synthetic nemo output for the adapter
            class _CombinedOutput:
                def __init__(self, text: str, word_timestamps: list[dict]):
                    self.text = text
                    self.timestamp = {"word": word_timestamps}

            combined = _CombinedOutput(" ".join(full_text_parts), all_word_timestamps)
            result = parakeet_output_to_whisper_result(combined, language=language or "en")
        else:
            result = parakeet_output_to_whisper_result(output[0], language=language or "en")

        return result
    finally:
        # Clean up chunk temp files (but not the original if it wasn't a temp)
        for cp, _ in chunks:
            if cp != audio_path and os.path.exists(cp):
                os.unlink(cp)
        if cleanup_path and os.path.exists(cleanup_path):
            os.unlink(cleanup_path)


# ---------------------------------------------------------------------------
# Regroup string pre-processing (stable-ts pad bug workaround)
# ---------------------------------------------------------------------------

_PAD_PATTERN = re.compile(r'_p=([^_]+)')


def strip_pad_from_regroup(regroup: str) -> tuple[str, float, float]:
    """Extract ``_p=start,end`` from a regroup string and return clean values.

    stable-ts has a bug where ``regroup()`` passes pad arguments as strings
    instead of floats, causing ``'>' not supported between 'str' and 'int'``.
    We strip the pad operation from the regroup string and apply it ourselves
    after transcription with properly typed arguments.

    Returns:
        (cleaned_regroup, start_pad, end_pad)
    """
    match = _PAD_PATTERN.search(regroup)
    if not match:
        return regroup, 0.0, 0.0

    pad_args = match.group(1).split(",")
    start_pad = float(pad_args[0]) if len(pad_args) > 0 else 0.0
    end_pad = float(pad_args[1]) if len(pad_args) > 1 else 0.0

    cleaned = _PAD_PATTERN.sub('', regroup)
    # Clean up any trailing/leading underscores left over
    cleaned = cleaned.strip('_')

    return cleaned, start_pad, end_pad


def apply_pad(result, start_pad: float, end_pad: float) -> None:
    """Apply padding to subtitle segments (workaround for stable-ts pad bug).

    Allows overlapping with the next segment (players stack them on screen).
    Caps at the next segment's *end* to prevent truly reversed ordering.
    """
    if start_pad == 0.0 and end_pad == 0.0:
        return
    segments = result.segments
    for i, segment in enumerate(segments):
        if start_pad > 0:
            segment.start = max(0, segment.start - start_pad)
        if end_pad > 0:
            desired_end = segment.end + end_pad
            if i + 1 < len(segments):
                desired_end = min(desired_end, segments[i + 1].end)
            segment.end = desired_end


# ---------------------------------------------------------------------------
# Minimum subtitle duration enforcement
# ---------------------------------------------------------------------------


def enforce_min_subtitle_duration(result, min_duration: float) -> None:
    """Extend short subtitle segments so they stay on screen long enough to read.

    Modifies *result* in place.  Any segment whose duration is less than
    *min_duration* (seconds) gets its end time pushed out.  Overlapping with
    the next segment is intentional — media players stack overlapping SRT
    entries on screen.  Capped at the next segment's *end* to prevent truly
    reversed ordering.
    """
    if min_duration <= 0:
        return
    segments = result.segments
    for i, segment in enumerate(segments):
        duration = segment.end - segment.start
        if duration < min_duration:
            desired_end = segment.start + min_duration
            if i + 1 < len(segments):
                desired_end = min(desired_end, segments[i + 1].end)
            segment.end = desired_end


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
        _start_model()

        # Check if the file is an audio file before trying to extract audio
        file_name, file_extension = os.path.splitext(file_path)
        is_audio_file = isAudioFileExtension(file_extension)

        data = file_path
        # Extract audio from the file if it has multiple audio tracks
        extracted_audio_file = handle_multiple_audio_tracks(file_path, force_language)
        if extracted_audio_file:
            data = extracted_audio_file

        # Normalize audio loudness for better transcription accuracy
        if normalize_audio_enabled:
            from subgen.media.audio import normalize_audio
            is_path = isinstance(data, str)
            normalized = normalize_audio(data, is_file_path=is_path)
            if normalized is not None:
                data = normalized

        # Determine the actual task
        if transcribe_or_translate_param == "transcribe_and_translate":
            actual_task = "transcribe"
        else:
            actual_task = transcribe_or_translate_param

        # Strip pad from regroup string (stable-ts bug workaround)
        start_pad, end_pad = 0.0, 0.0
        if custom_regroup and custom_regroup.lower() != "default":
            cleaned_regroup, start_pad, end_pad = strip_pad_from_regroup(custom_regroup)

        if asr_engine == 'parakeet':
            result = _transcribe_parakeet(data, force_language.to_iso_639_1(), actual_task)
            # Apply custom_regroup via stable-ts post-processing if requested
            if custom_regroup and custom_regroup.lower() != "default":
                cleaned_regroup_str = strip_pad_from_regroup(custom_regroup)[0]
                if cleaned_regroup_str and hasattr(result, 'regroup'):
                    result.regroup(cleaned_regroup_str)
        else:
            args = {}
            display_name = os.path.basename(file_path)
            args["progress_callback"] = ProgressHandler(display_name)

            if custom_regroup and custom_regroup.lower() != "default":
                cleaned_regroup_str, _, _ = strip_pad_from_regroup(custom_regroup)
                if cleaned_regroup_str:
                    args["regroup"] = cleaned_regroup_str

            args.update(whisper_kwargs)

            # Import model at function level to get the current (possibly re-loaded) reference
            from subgen.models.whisper_model import model as current_model

            result = _transcribe_with_kwarg_filter(
                current_model,
                audio=data,
                language=force_language.to_iso_639_1(),
                task=actual_task,
                verbose=None,
                **args,
            )

        appendLine(result)
        if filter_subtitles:
            filter_segments(result)
        apply_pad(result, start_pad, end_pad)
        enforce_min_subtitle_duration(result, min_subtitle_duration)

        if enable_diarization:
            from subgen.services.diarization import add_speaker_labels
            speaker_count = add_speaker_labels(result, data, transcribe_device)
            logging.info(f"Diarization: identified {speaker_count} speaker(s)")

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
        _delete_model()


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

        # Determine the actual task
        if requested_task == "transcribe_and_translate":
            actual_task = "transcribe"
        else:
            actual_task = requested_task

        _start_model()

        # Normalize audio loudness for better transcription accuracy
        if normalize_audio_enabled and encode:
            from subgen.media.audio import normalize_audio
            normalized = normalize_audio(file_content, is_file_path=False)
            if normalized is not None:
                file_content = normalized

        # Strip pad from regroup string (stable-ts bug workaround)
        start_pad, end_pad = 0.0, 0.0
        if custom_regroup and custom_regroup.lower() != "default":
            cleaned_regroup, start_pad, end_pad = strip_pad_from_regroup(custom_regroup)

        if asr_engine == 'parakeet':
            # Prepare audio data for Parakeet
            if encode:
                audio_data = file_content
            else:
                audio_data = (
                    np.frombuffer(file_content, np.int16)
                    .flatten()
                    .astype(np.float32)
                    / 32768.0
                )

            result = _transcribe_parakeet(audio_data, language, actual_task)
            # Apply custom_regroup via stable-ts post-processing if requested
            if custom_regroup and custom_regroup.lower() != "default":
                cleaned_regroup_str = strip_pad_from_regroup(custom_regroup)[0]
                if cleaned_regroup_str and hasattr(result, 'regroup'):
                    result.regroup(cleaned_regroup_str)
        else:
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
                cleaned_regroup_str, _, _ = strip_pad_from_regroup(custom_regroup)
                if cleaned_regroup_str:
                    args["regroup"] = cleaned_regroup_str

            args.update(whisper_kwargs)

            # Import model at function level to get the current (possibly re-loaded) reference
            from subgen.models.whisper_model import model as current_model

            # Perform transcription
            result = _transcribe_with_kwarg_filter(
                current_model, task=actual_task, language=language, **args, verbose=None
            )

        appendLine(result)
        if filter_subtitles:
            filter_segments(result)
        apply_pad(result, start_pad, end_pad)
        enforce_min_subtitle_duration(result, min_subtitle_duration)

        if enable_diarization:
            from subgen.services.diarization import add_speaker_labels
            speaker_count = add_speaker_labels(result, file_content, transcribe_device)
            logging.info(f"Diarization: identified {speaker_count} speaker(s)")

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
        _delete_model()
