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
    max_scene_duration,
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

# ---------------------------------------------------------------------------
# JAV-tuned regroup strategy (ported from WhisperJAV reconstruction.py)
# ---------------------------------------------------------------------------
# stable-ts uses '_' to chain regroup operations sequentially.
# Default 'da' splits at any 0.5s gap, which fragments conversational
# dialogue at natural thinking pauses.  JAV-tuned changes:
#   isp:  ignore special periods (Mr., Dr.)
#   cm:   clamp word timestamps to segment boundaries
#   sp:   split at sentence-ending punctuation (. ? !)
#   sg:   0.5 -> 1.5  Only split at 1.5s+ gaps (not every breath pause)
#   mg:   merge fragments if gap < 1.5s AND combined < 80 chars
#   sl:   split if segment > 80 chars
#   sd=8  speech-time cap at 8 seconds
#   cm:   final safety clamp
#
# After regrouping, enforce_wall_clock_cap() runs to catch segments whose
# wall-clock span exceeds 8s (sd only caps speech time, not screen time).
REGROUP_SUBGEN: str = (
    "isp_cm"               # ignore special periods (Mr., Dr.) + initial clamp
    "_sp=./?/!"            # split at sentence-ending punctuation
    "_sg=1.5"              # split at 1.5s+ gaps (relaxed from default 0.5s)
    "_mg=1.5++80+1"        # merge fragments: gap < 1.5s, combined < 80 chars
    "_sl=80"               # split if segment > 80 chars
    "_sd=8"                # max 8s speech time per subtitle
    "_cm"                  # final safety clamp
)

def _safe_regroup(result: object, regroup_str: str) -> None:
    """Apply regroup string to a WhisperResult, handling empty-segment edge cases.

    stable-ts's ``sd`` (split_by_duration) crashes with ``argmin of empty
    sequence`` when a segment has no words.  This can happen after prior
    regroup operations (``sp``, ``sg``) create wordless segments.

    This helper strips empty segments before regrouping, then cleans up
    any new empty segments produced by the regroup itself.
    """
    if not result or not hasattr(result, 'regroup'):
        return
    if not regroup_str:
        return

    # Strip segments with no words before regrouping.
    if hasattr(result, 'segments') and result.segments:
        result.segments = [
            seg for seg in result.segments
            if hasattr(seg, 'words') and len(seg.words) > 0
        ]

    if not result.segments:
        return

    result.regroup(regroup_str)

    # Clean up any empty segments the regroup may have produced.
    if hasattr(result, 'segments'):
        result.segments = [
            seg for seg in result.segments
            if hasattr(seg, 'words') and len(seg.words) > 0
        ]


# Regex for capitalizing first letter after sentence-ending punctuation
_SENTENCE_START_RE = re.compile(r'(?:^|[.!?]\s+)([a-z])')


def _capitalize_segments(result) -> None:
    """Capitalize the first letter of each segment and after sentence boundaries.

    Whisper sometimes outputs all-lowercase text depending on the model and
    language.  This post-processes the result in-place so subtitles have
    proper sentence capitalization.

    stable_whisper's ``Segment.text`` is a read-only property derived from
    its words, so we modify the word objects directly.
    """
    for seg in result.segments:
        if not hasattr(seg, "words") or not seg.words:
            continue
        # Capitalize the first letter of the first word in the segment.
        first_word = seg.words[0]
        w = first_word.word
        stripped = w.lstrip()
        if stripped and stripped[0].islower():
            leading = w[: len(w) - len(stripped)]
            first_word.word = leading + stripped[0].upper() + stripped[1:]
        # Capitalize the first letter after sentence-ending punctuation
        # within the same segment (rare, but possible).
        for i in range(1, len(seg.words)):
            prev = seg.words[i - 1].word.rstrip()
            if prev and prev[-1] in ".!?":
                cur = seg.words[i]
                cw = cur.word
                cs = cw.lstrip()
                if cs and cs[0].islower():
                    lead = cw[: len(cw) - len(cs)]
                    cur.word = lead + cs[0].upper() + cs[1:]


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
    elif asr_engine == 'qwen':
        from subgen.models.qwen_model import start_model as start_qwen
        start_qwen()
    else:
        from subgen.models.whisper_model import start_model as start_whisper
        start_whisper()


def _delete_model() -> None:
    """Schedule cleanup for the appropriate ASR model based on the configured engine."""
    if asr_engine == 'parakeet':
        from subgen.models.parakeet_model import delete_model as delete_parakeet
        delete_parakeet()
    elif asr_engine == 'qwen':
        from subgen.models.qwen_model import delete_model as delete_qwen
        delete_qwen()
    else:
        from subgen.models.whisper_model import delete_model as delete_whisper
        delete_whisper()


# ---------------------------------------------------------------------------
# Parakeet transcription backend
# ---------------------------------------------------------------------------


_MAX_SCENE_SECONDS: float = max_scene_duration  # from config (default 30s for speech-aligned scenes)


def _ensure_audio_path(audio_data: object) -> tuple[str, str | None]:
    """Convert audio data to a file path suitable for ASR engines.

    Accepts a file path (str), raw bytes, or a numpy array.
    Returns ``(audio_path, cleanup_path)`` where *cleanup_path* is set
    when a temporary file was created and should be deleted by the caller.
    """
    import tempfile
    import wave

    if isinstance(audio_data, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_data)
            return f.name, f.name
    if isinstance(audio_data, str):
        return audio_data, None
    if hasattr(audio_data, '__array__'):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            audio_path = f.name
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        return audio_path, audio_path
    raise TypeError(f"Unsupported audio data type: {type(audio_data)}")


def _cleanup_scenes(scenes: list[tuple[str, float]], audio_path: str, cleanup_path: str | None) -> None:
    """Remove temporary scene files and the converted audio file if any."""
    for sp, _ in scenes:
        if sp != audio_path and os.path.exists(sp):
            os.unlink(sp)
    if cleanup_path and os.path.exists(cleanup_path):
        os.unlink(cleanup_path)


def _transcribe_parakeet(audio_data: object, language: str, task: str) -> object:
    """Transcribe using NVIDIA Parakeet-TDT model via NeMo.

    Accepts audio as a file path (str), raw bytes, or a numpy array.
    Returns a WhisperResult-compatible object via the result adapter.

    Long audio files are automatically split into scenes at silence
    boundaries using auditok (falls back to fixed-size splitting).
    Inference runs in fp16 via torch.cuda.amp.autocast for reduced VRAM usage.
    """
    import torch

    from subgen.media.scene_detection import split_audio_scenes
    from subgen.models.parakeet_model import model as parakeet_model
    from subgen.models.result_adapter import parakeet_output_to_whisper_result

    audio_path, cleanup_path = _ensure_audio_path(audio_data)
    scenes: list[tuple[str, float]] = []
    try:
        scenes = split_audio_scenes(audio_path, _MAX_SCENE_SECONDS)
        is_chunked = len(scenes) > 1 or scenes[0][0] != audio_path

        if is_chunked:
            logger.info("Audio split into %d scene(s) for Parakeet", len(scenes))

        all_word_timestamps: list[dict] = []
        full_text_parts: list[str] = []

        use_fp16 = (
            transcribe_device.lower() == "cuda"
            and torch.cuda.is_available()
            and compute_type in ("auto", "float16", "int8_float16")
        )

        for i, (scene_path, scene_offset) in enumerate(scenes):
            if is_chunked:
                logger.info("Transcribing scene %d/%d (offset %.1fs)", i + 1, len(scenes), scene_offset)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = parakeet_model.transcribe([scene_path], timestamps=True)

            # After the first transcribe call, NeMo's decoder is configured
            # with timestamps + n-gram LM.  Suppress redundant re-init on
            # subsequent scenes to avoid reloading the LM from disk (~7s each).
            if i == 0 and is_chunked:
                parakeet_model._orig_change_decoding = parakeet_model.change_decoding_strategy
                parakeet_model.change_decoding_strategy = lambda *a, **kw: None

            scene_output = output[0]
            scene_text = getattr(scene_output, "text", "") or ""
            timestamp_data = getattr(scene_output, "timestamp", {}) or {}
            word_ts = timestamp_data.get("word", [])

            # Offset timestamps by the scene's start position in the original audio
            if scene_offset > 0 and word_ts:
                for w in word_ts:
                    w["start"] = float(w.get("start", 0.0)) + scene_offset
                    w["end"] = float(w.get("end", 0.0)) + scene_offset

            all_word_timestamps.extend(word_ts)
            full_text_parts.append(scene_text)

        # Build a combined result using the adapter
        if is_chunked and all_word_timestamps:
            class _CombinedOutput:
                def __init__(self, text: str, word_timestamps: list[dict]):
                    self.text = text
                    self.timestamp = {"word": word_timestamps}

            combined = _CombinedOutput(" ".join(full_text_parts), all_word_timestamps)
            result = parakeet_output_to_whisper_result(combined, language=language or "en", audio_path=audio_path)
        else:
            result = parakeet_output_to_whisper_result(output[0], language=language or "en", audio_path=audio_path)

        return result
    finally:
        # Restore change_decoding_strategy if it was suppressed during chunking
        if hasattr(parakeet_model, '_orig_change_decoding'):
            parakeet_model.change_decoding_strategy = parakeet_model._orig_change_decoding
            del parakeet_model._orig_change_decoding

        _cleanup_scenes(scenes, audio_path, cleanup_path)


# ---------------------------------------------------------------------------
# Qwen3-ASR transcription backend
# ---------------------------------------------------------------------------


def _transcribe_qwen(audio_data: object, language: str, task: str) -> object:
    """Transcribe using Qwen3-ASR model.

    Accepts audio as a file path (str), raw bytes, or a numpy array.
    Returns a WhisperResult-compatible object via the result adapter.

    Long audio files are automatically split into scenes at speech/silence
    boundaries (default 30s max via Silero VAD + auditok).
    """
    from subgen.config import qwen_clean_text
    from subgen.media.scene_detection import get_audio_duration, split_audio_scenes
    from subgen.models.qwen_model import model as qwen_model, compute_dynamic_token_limit
    from subgen.models.result_adapter import qwen_output_to_whisper_result

    audio_path, cleanup_path = _ensure_audio_path(audio_data)
    scenes: list[tuple[str, float]] = []
    try:
        # Map language code to Qwen's expected format (e.g. "en" -> "English")
        qwen_language = None
        if language:
            _LANG_MAP = {
                "en": "English", "fr": "French", "de": "German", "es": "Spanish",
                "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
                "ko": "Korean", "zh": "Chinese", "nl": "Dutch", "pl": "Polish",
                "sv": "Swedish", "da": "Danish", "fi": "Finnish", "hu": "Hungarian",
                "cs": "Czech", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
                "sk": "Slovak", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian",
                "lt": "Lithuanian", "el": "Greek", "mt": "Maltese", "uk": "Ukrainian",
            }
            qwen_language = _LANG_MAP.get(language)

        logger.info("Transcribing with Qwen3-ASR (language=%s)", qwen_language or "auto-detect")

        # Split long audio into scenes at silence boundaries
        scenes = split_audio_scenes(audio_path, _MAX_SCENE_SECONDS)
        is_chunked = len(scenes) > 1 or scenes[0][0] != audio_path

        if is_chunked:
            logger.info("Audio split into %d scene(s) for Qwen3-ASR", len(scenes))

        all_texts: list[str] = []
        all_timestamps: list[object] = []

        for i, (scene_path, scene_offset) in enumerate(scenes):
            if is_chunked:
                logger.info("Transcribing scene %d/%d (offset %.1fs)", i + 1, len(scenes), scene_offset)

            # Dynamic token budget per scene
            scene_duration = get_audio_duration(scene_path)
            original_max_tokens = getattr(qwen_model, 'max_new_tokens', None)
            if scene_duration > 0 and original_max_tokens:
                dynamic_limit = compute_dynamic_token_limit(scene_duration)
                if dynamic_limit != original_max_tokens:
                    qwen_model.max_new_tokens = dynamic_limit

            try:
                scene_results = qwen_model.transcribe(
                    audio=[scene_path],
                    language=[qwen_language] if qwen_language else None,
                    return_time_stamps=True,
                )
            finally:
                if original_max_tokens is not None:
                    qwen_model.max_new_tokens = original_max_tokens

            scene_output = scene_results[0]
            scene_text = getattr(scene_output, "text", "") or ""

            # Clean raw text before combining
            if qwen_clean_text and scene_text:
                from subgen.services.text_cleaner import clean_asr_text
                cleaned = clean_asr_text(scene_text)
                if cleaned != scene_text:
                    logger.info(
                        "Text cleaner scene %d: %d -> %d chars",
                        i + 1, len(scene_text), len(cleaned),
                    )
                    scene_output.text = cleaned
                    scene_text = cleaned

            all_texts.append(scene_text)

            # Collect timestamps with offset applied.
            # ForcedAlignItem is a frozen dataclass, so we create simple
            # wrapper objects with the offset baked in.
            raw_stamps = getattr(scene_output, "time_stamps", None)
            if raw_stamps:
                if scene_offset > 0:
                    class _OffsetStamp:
                        __slots__ = ("text", "start_time", "end_time")
                        def __init__(self, ts: object, offset: float):
                            self.text = getattr(ts, "text", "")
                            self.start_time = float(getattr(ts, "start_time", 0.0)) + offset
                            self.end_time = float(getattr(ts, "end_time", 0.0)) + offset
                    all_timestamps.extend(_OffsetStamp(ts, scene_offset) for ts in raw_stamps)
                else:
                    all_timestamps.extend(raw_stamps)

        # Build combined result
        if is_chunked and all_timestamps:
            # Create a synthetic combined output for the adapter
            class _CombinedQwenOutput:
                def __init__(self, text: str, time_stamps: list, lang: str):
                    self.text = text
                    self.time_stamps = time_stamps
                    self.language = lang

            combined_text = " ".join(all_texts)
            detected_lang = getattr(scenes and scene_results[0], "language", None) or language or "en"
            combined = _CombinedQwenOutput(combined_text, all_timestamps, detected_lang)
            result = qwen_output_to_whisper_result(
                combined,
                language=language or detected_lang,
                audio_path=audio_path,
            )
        else:
            result = qwen_output_to_whisper_result(
                scene_results[0],
                language=language or getattr(scene_results[0], "language", "en") or "en",
                audio_path=audio_path,
            )

        return result
    finally:
        _cleanup_scenes(scenes, audio_path, cleanup_path)


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

        # Determine the regroup string: user override or JAV-tuned default.
        # Strip pad from custom regroup (stable-ts bug workaround).
        start_pad, end_pad = 0.0, 0.0
        has_custom_regroup: bool = bool(custom_regroup and custom_regroup.lower() != "default")
        if has_custom_regroup:
            regroup_str, start_pad, end_pad = strip_pad_from_regroup(custom_regroup)
        else:
            regroup_str = REGROUP_SUBGEN

        if asr_engine == 'parakeet':
            result = _transcribe_parakeet(data, force_language.to_iso_639_1(), actual_task)
            _safe_regroup(result, regroup_str)
        elif asr_engine == 'qwen':
            result = _transcribe_qwen(data, force_language.to_iso_639_1(), actual_task)
            _safe_regroup(result, regroup_str)
        else:
            args = {}
            display_name = os.path.basename(file_path)
            args["progress_callback"] = ProgressHandler(display_name)

            if regroup_str:
                args["regroup"] = regroup_str
            # suppress_silence refines segment boundaries based on silence
            # detection, pulling subtitle start/end to actual speech edges.
            args["suppress_silence"] = True

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

        # Wall-clock cap enforcement: split segments exceeding 8s screen time.
        # stable-ts's sd= caps speech time, not wall-clock — this catches the rest.
        from subgen.services.subtitle_constraints import enforce_wall_clock_cap
        enforce_wall_clock_cap(result)

        appendLine(result)
        if filter_subtitles:
            filter_segments(result)
        _capitalize_segments(result)
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

        # Enforce subtitle display constraints (max line length, overlap, gaps)
        from subgen.services.subtitle_constraints import enforce_display_constraints
        enforce_display_constraints(result)

        # If it is an audio file, write the LRC file
        if is_audio_file and lrc_for_audio_files:
            write_lrc(result, file_name + ".lrc")
        else:
            output_language = LanguageCode.from_string(result.language)
            result.to_srt_vtt(
                name_subtitle(file_path, output_language),
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

        # Determine the regroup string: user override or JAV-tuned default.
        # Strip pad from custom regroup (stable-ts bug workaround).
        start_pad, end_pad = 0.0, 0.0
        has_custom_regroup: bool = bool(custom_regroup and custom_regroup.lower() != "default")
        if has_custom_regroup:
            regroup_str, start_pad, end_pad = strip_pad_from_regroup(custom_regroup)
        else:
            regroup_str = REGROUP_SUBGEN

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
            _safe_regroup(result, regroup_str)
        elif asr_engine == 'qwen':
            if encode:
                audio_data = file_content
            else:
                audio_data = (
                    np.frombuffer(file_content, np.int16)
                    .flatten()
                    .astype(np.float32)
                    / 32768.0
                )

            result = _transcribe_qwen(audio_data, language, actual_task)
            _safe_regroup(result, regroup_str)
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

            if regroup_str:
                args["regroup"] = regroup_str
            # suppress_silence refines segment boundaries based on silence
            # detection, pulling subtitle start/end to actual speech edges.
            args["suppress_silence"] = True

            args.update(whisper_kwargs)

            # Import model at function level to get the current (possibly re-loaded) reference
            from subgen.models.whisper_model import model as current_model

            # Perform transcription
            result = _transcribe_with_kwarg_filter(
                current_model, task=actual_task, language=language, **args, verbose=None
            )

        # Wall-clock cap enforcement: split segments exceeding 8s screen time.
        # stable-ts's sd= caps speech time, not wall-clock — this catches the rest.
        from subgen.services.subtitle_constraints import enforce_wall_clock_cap
        enforce_wall_clock_cap(result)

        appendLine(result)
        if filter_subtitles:
            filter_segments(result)
        _capitalize_segments(result)
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

        # Enforce subtitle display constraints (max line length, overlap, gaps)
        from subgen.services.subtitle_constraints import enforce_display_constraints
        enforce_display_constraints(result)

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
