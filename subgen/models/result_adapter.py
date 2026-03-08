"""Adapter to convert NVIDIA Parakeet (NeMo) ASR output into a stable_whisper.WhisperResult.

The Parakeet model returns a dataclass-like object with:
  - ``output.text``  -- full transcription string
  - ``output.timestamp['word']``  -- list of dicts [{word, start, end}, ...]

This module groups those word timestamps into sentence-level segments and
constructs a genuine ``stable_whisper.WhisperResult`` so the entire
downstream pipeline (diarization, subtitle filter, translation, SRT output)
works without modification.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import stable_whisper

logger = logging.getLogger(__name__)

# Punctuation characters that mark the end of a sentence.
_SENTENCE_ENDERS: re.Pattern[str] = re.compile(r"[.!?]$")

# Comma — used as a soft split point when the segment is already long enough.
_CLAUSE_BREAK: re.Pattern[str] = re.compile(r",$")

# Maximum segment duration in seconds before forcing a split.
_MAX_SEGMENT_DURATION: float = 8.0

# Minimum segment duration before allowing a comma-based split.
_MIN_SEGMENT_FOR_CLAUSE_SPLIT: float = 3.0

# Gap (seconds) between the end of one word and the start of the next
# that triggers a segment break (breath / speaker change).
_GAP_SPLIT_THRESHOLD: float = 0.7


def _flush_segment(
    segments: List[Dict[str, Any]],
    current_words: List[Dict[str, Any]],
    segment_start: float,
) -> None:
    """Append the accumulated words as a new segment."""
    if not current_words:
        return
    segment_text = "".join(w["word"] for w in current_words)
    segment_end = current_words[-1]["end"]
    segments.append(
        {
            "start": segment_start,
            "end": segment_end,
            "text": segment_text,
            "words": current_words,
            "no_speech_prob": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
        }
    )


def _restore_punctuation_from_text(
    full_text: str, word_timestamps: List[Dict[str, Any]],
) -> None:
    """Map punctuation and capitalization from the model's full text back onto word timestamps.

    Parakeet-TDT outputs punctuated/capitalized text in ``output.text`` but
    individual word timestamps contain bare lowercase words.  This function
    aligns the two and transfers punctuation and casing onto the word
    timestamp entries so downstream splitting and display work correctly.

    Modifies *word_timestamps* in place.
    """
    if not full_text or not word_timestamps:
        return

    tokens = full_text.split()

    if len(tokens) == len(word_timestamps):
        # Perfect 1:1 alignment — copy token text (preserving punctuation + case)
        for token, wt in zip(tokens, word_timestamps):
            wt["word"] = token
        return

    # Lengths don't match — use best-effort positional alignment with a
    # wider lookahead window to handle insertions/deletions.
    _MAX_LOOKAHEAD = 8
    matched = [False] * len(word_timestamps)
    ti = 0
    for wi, wt in enumerate(word_timestamps):
        if ti >= len(tokens):
            break
        bare_word = wt.get("word", "").strip().lower()
        if not bare_word:
            continue
        for look_ahead in range(_MAX_LOOKAHEAD):
            if ti + look_ahead >= len(tokens):
                break
            token = tokens[ti + look_ahead]
            token_bare = re.sub(r"[^\w']+$", "", token).lower()
            if token_bare == bare_word:
                wt["word"] = token
                matched[wi] = True
                ti = ti + look_ahead + 1
                break
        else:
            ti += 1

    # For any words that couldn't be matched, apply capitalization rules
    # based on context: capitalize after sentence-ending punctuation and
    # at the very start of the text.
    for wi, wt in enumerate(word_timestamps):
        if matched[wi]:
            continue
        word = wt.get("word", "").strip()
        if not word:
            continue
        should_capitalize = False
        if wi == 0:
            should_capitalize = True
        else:
            prev_word = word_timestamps[wi - 1].get("word", "").rstrip()
            if prev_word and prev_word[-1] in ".!?":
                should_capitalize = True
        if should_capitalize:
            wt["word"] = word[0].upper() + word[1:]
        else:
            wt["word"] = word


def _group_words_into_segments(
    word_timestamps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group Parakeet word timestamps into subtitle segments.

    Splits on:
      - Sentence-ending punctuation (``.``, ``!``, ``?``)
      - Commas when the segment is already >= ``_MIN_SEGMENT_FOR_CLAUSE_SPLIT``
      - Gaps between words >= ``_GAP_SPLIT_THRESHOLD`` (breath / speaker change)
      - Maximum duration ``_MAX_SEGMENT_DURATION``

    Each returned segment dict is ready for ``stable_whisper.WhisperResult``
    consumption.
    """
    if not word_timestamps:
        return []

    segments: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []
    segment_start: float = word_timestamps[0].get("start", 0.0)
    prev_word_end: float = segment_start

    for raw_word in word_timestamps:
        word_text: str = raw_word.get("word", "")
        word_start: float = float(raw_word.get("start", 0.0))
        word_end: float = float(raw_word.get("end", 0.0))

        # Detect a gap (pause / breath / speaker change) BEFORE adding the word.
        # If there's a significant gap, flush the current segment first.
        gap = word_start - prev_word_end
        if current_words and gap >= _GAP_SPLIT_THRESHOLD:
            _flush_segment(segments, current_words, segment_start)
            current_words = []
            segment_start = word_start

        if segment_start is None:
            segment_start = word_start

        # Ensure the word text has a leading space for proper concatenation
        # (stable_whisper joins words via ''.join(w.word for w in words)).
        if current_words and not word_text.startswith(" "):
            word_text = " " + word_text

        current_words.append(
            {
                "word": word_text,
                "start": word_start,
                "end": word_end,
                "probability": 1.0,
            }
        )
        prev_word_end = word_end

        # Decide whether to close the current segment.
        duration = word_end - segment_start
        ends_sentence = _SENTENCE_ENDERS.search(word_text.rstrip())
        exceeds_duration = duration >= _MAX_SEGMENT_DURATION
        clause_break = (
            _CLAUSE_BREAK.search(word_text.rstrip())
            and duration >= _MIN_SEGMENT_FOR_CLAUSE_SPLIT
        )

        if ends_sentence or exceeds_duration or clause_break:
            _flush_segment(segments, current_words, segment_start)
            current_words = []
            segment_start = None  # will be set when next word arrives

    # Flush any remaining words into a final segment.
    _flush_segment(segments, current_words, segment_start)

    return segments


def parakeet_output_to_whisper_result(
    nemo_output: Any,
    language: str = "en",
    audio_path: Optional[str] = None,
) -> "WhisperResult":
    """Convert a NeMo Parakeet transcription output to a ``stable_whisper.WhisperResult``.

    Uses the same improved pipeline as Qwen3-ASR: position-based punctuation
    merge, zero-duration fix, alignment sentinel, and ``transcribe_any()``
    reconstruction when *audio_path* is available.

    Parameters
    ----------
    nemo_output:
        A single element from the list returned by ``model.transcribe(..., timestamps=True)``.
        Expected attributes:
          - ``.text``  -- full transcription string
          - ``.timestamp``  -- dict with key ``'word'`` mapping to a list of
            ``{word, start, end}`` dicts.
    language:
        ISO 639-1 language code to store in the result (default ``"en"``).
    audio_path:
        Optional path to the audio file.  When provided, ``stable_whisper.transcribe_any()``
        is used for segment reconstruction (better regrouping than manual splitting).

    Returns
    -------
    stable_whisper.WhisperResult
        A result object that is fully compatible with the downstream subgen
        pipeline (diarization, subtitle filter, translation, SRT/VTT output).
    """
    from stable_whisper import WhisperResult  # type: ignore[import-not-found]

    # Adapter to present Parakeet word dicts ({word, start, end}) as objects
    # with .text/.start_time/.end_time for merge_master_with_timestamps().
    class _PkTimestamp:
        def __init__(self, d: Dict[str, Any]):
            self.text = d.get("word", "").strip()
            self.start_time = float(d.get("start", 0.0))
            self.end_time = float(d.get("end", 0.0))

    # Extract word timestamps from the NeMo output.
    full_text: str = getattr(nemo_output, "text", "") or ""
    timestamp_data: Dict[str, Any] = getattr(nemo_output, "timestamp", {}) or {}
    word_timestamps: List[Dict[str, Any]] = timestamp_data.get("word", [])

    if not word_timestamps:
        logger.warning(
            "Parakeet output contains no word timestamps; "
            "creating a single segment from the full text."
        )
        segments = [
            {
                "start": 0.0,
                "end": 0.0,
                "text": full_text,
                "words": None,
                "no_speech_prob": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
            }
        ]
        result_dict: Dict[str, Any] = {
            "language": language,
            "segments": segments,
        }
        return WhisperResult(result_dict, check_sorted=False)

    # Position-based punctuation merge (replaces fragile token-matching).
    adapted_stamps = [_PkTimestamp(w) for w in word_timestamps]
    final_words = merge_master_with_timestamps(full_text, adapted_stamps)
    logger.debug(
        "Parakeet punctuation merge: %d timestamps -> %d merged words",
        len(word_timestamps), len(final_words),
    )

    # Fix zero-duration words: scan forward for the next word with a
    # strictly later start_time so consecutive zero-duration items get
    # a proper end time (safety net — Parakeet usually has correct ends).
    _DEFAULT_WORD_DUR = 0.3  # seconds
    for i, w in enumerate(final_words):
        if w["end"] <= w["start"]:
            next_start = None
            for j in range(i + 1, len(final_words)):
                if final_words[j]["start"] > w["start"]:
                    next_start = final_words[j]["start"]
                    break
            if next_start is not None:
                w["end"] = next_start
            else:
                w["end"] = w["start"] + _DEFAULT_WORD_DUR

    # Assess alignment quality and redistribute if collapsed.
    audio_duration = 0.0
    if audio_path:
        from subgen.media.scene_detection import get_audio_duration
        audio_duration = get_audio_duration(audio_path)

    if audio_duration <= 0 and final_words:
        audio_duration = max(w["end"] for w in final_words)

    quality = _assess_alignment_quality(final_words, audio_duration)
    if quality["status"] == "COLLAPSED":
        logger.warning("Parakeet: applying proportional redistribution due to alignment collapse")
        final_words = _redistribute_collapsed_words(final_words, audio_duration)

    # Use stable_whisper.transcribe_any() for proper segment reconstruction
    # when audio_path is available.
    if audio_path:
        _precomputed_words = list(final_words)

        def precomputed_inference(audio: str, **kwargs: Any) -> List[List[Dict[str, Any]]]:
            return [_precomputed_words]

        try:
            result = stable_whisper.transcribe_any(
                inference_func=precomputed_inference,
                audio=str(audio_path),
                audio_type="str",
                regroup=True,
                vad=False,
                demucs=False,
                suppress_silence=True,
                suppress_word_ts=True,
                force_order=True,
                verbose=False,
            )
            result.language = language

            logger.debug(
                "Adapted Parakeet output via transcribe_any: %d segments, %d words, language='%s'",
                len(result.segments),
                len(final_words),
                language,
            )
            return result
        except Exception:
            logger.warning(
                "stable_whisper.transcribe_any() failed; falling back to manual grouping",
                exc_info=True,
            )

    # Fallback: use the manual grouping function.
    segments = _group_words_into_segments(final_words)

    result_dict = {
        "language": language,
        "segments": segments,
    }

    logger.debug(
        "Adapted Parakeet output: %d segments, %d words, language='%s'",
        len(segments),
        len(final_words),
        language,
    )

    return WhisperResult(result_dict, check_sorted=False)


def merge_master_with_timestamps(
    master_text: str, timestamps: List[Any],
) -> List[Dict[str, Any]]:
    """Merge the full punctuated master text with ForcedAligner timestamps.

    Instead of token-matching, scan through the master text character-by-character.
    For each ForcedAlignItem, find its word in the master text by position.
    Any gap between the previous position and this word (punctuation, spaces) gets
    attached to the PREVIOUS word.

    Parameters
    ----------
    master_text:
        The full punctuated transcription text from Qwen3-ASR.
    timestamps:
        Raw ForcedAlignItems (objects with .text/.start_time/.end_time attributes)
        or dicts with 'text'/'start_time'/'end_time' keys.

    Returns
    -------
    List of ``{'word': str, 'start': float, 'end': float}`` dicts with
    punctuation correctly attached.
    """
    if not master_text or not master_text.strip():
        return []
    if not timestamps:
        return [{"word": master_text.strip(), "start": 0.0, "end": 0.0}]

    def _get_attr(obj: Any, attr: str) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr)
        return None

    result: List[Dict[str, Any]] = []
    master_pos: int = 0

    for ts in timestamps:
        ts_word = _get_attr(ts, "text")
        ts_start = _get_attr(ts, "start_time")
        ts_end = _get_attr(ts, "end_time")
        if not ts_word:
            continue

        word_start = master_text.find(ts_word, master_pos)
        if word_start == -1:
            result.append({
                "word": ts_word,
                "start": float(ts_start) if ts_start is not None else 0.0,
                "end": float(ts_end) if ts_end is not None else 0.0,
            })
            continue

        word_end = word_start + len(ts_word)
        if word_start > master_pos:
            gap = master_text[master_pos:word_start]
            if result:
                result[-1]["word"] += gap
            else:
                ts_word = gap + ts_word

        result.append({
            "word": ts_word,
            "start": float(ts_start) if ts_start is not None else 0.0,
            "end": float(ts_end) if ts_end is not None else 0.0,
        })
        master_pos = word_end

    if master_pos < len(master_text):
        trailing = master_text[master_pos:]
        if result:
            result[-1]["word"] += trailing
        elif trailing.strip():
            result.append({"word": trailing, "start": 0.0, "end": 0.0})

    return result


def _assess_alignment_quality(
    word_dicts: List[Dict[str, Any]], audio_duration: float,
) -> Dict[str, Any]:
    """Check whether the ForcedAligner produced collapsed/degenerate timestamps.

    Collapse signatures:
      - Coverage ratio < 5% (word span / audio duration)
      - CPS > 50 characters/second (physically impossible speech rate)
      - Word span < 0.5 seconds for substantial text
      - Degenerate ratio > 40% (words where start == end)

    Returns
    -------
    Dict with keys ``'status'`` (``'OK'`` or ``'COLLAPSED'``) and ``'details'``.
    """
    if not word_dicts or audio_duration <= 0:
        return {"status": "OK", "details": "no words or no audio duration"}

    first_start = min(w["start"] for w in word_dicts)
    last_end = max(w["end"] for w in word_dicts)
    word_span = last_end - first_start

    total_chars = sum(len(w["word"].strip()) for w in word_dicts)
    cps = total_chars / word_span if word_span > 0 else float("inf")
    coverage = word_span / audio_duration if audio_duration > 0 else 0.0

    degenerate_count = sum(1 for w in word_dicts if w["start"] == w["end"])
    degenerate_ratio = degenerate_count / len(word_dicts)

    reasons: List[str] = []
    if coverage < 0.05:
        reasons.append(f"coverage={coverage:.3f} (<5%)")
    if cps > 50:
        reasons.append(f"cps={cps:.1f} (>50)")
    if word_span < 0.5 and total_chars > 20:
        reasons.append(f"word_span={word_span:.3f}s (<0.5s for {total_chars} chars)")
    if degenerate_ratio > 0.4:
        reasons.append(f"degenerate_ratio={degenerate_ratio:.2f} (>40%)")

    if reasons:
        detail_str = "; ".join(reasons)
        logger.warning(
            "Alignment collapse detected: %s (span=%.2fs, duration=%.2fs, words=%d)",
            detail_str, word_span, audio_duration, len(word_dicts),
        )
        return {"status": "COLLAPSED", "details": detail_str}

    return {"status": "OK", "details": "alignment looks reasonable"}


def _redistribute_collapsed_words(
    word_dicts: List[Dict[str, Any]], audio_duration: float,
) -> List[Dict[str, Any]]:
    """Proportionally redistribute words across the audio duration.

    Used when the ForcedAligner has collapsed all words into a tiny time window.
    Each word gets a duration proportional to its character count, with a
    minimum of 20ms per word.

    Parameters
    ----------
    word_dicts:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
    audio_duration:
        Total audio duration in seconds.

    Returns
    -------
    New list of word dicts with redistributed timestamps.
    """
    if not word_dicts:
        return []

    _MIN_WORD_DURATION = 0.02  # 20ms minimum

    char_counts = [max(len(w["word"].strip()), 1) for w in word_dicts]
    total_chars = sum(char_counts)

    # Reserve minimum duration for each word, distribute the rest proportionally
    min_total = _MIN_WORD_DURATION * len(word_dicts)
    available = max(audio_duration - min_total, 0.0)

    redistributed: List[Dict[str, Any]] = []
    cursor = 0.0

    for word_dict, chars in zip(word_dicts, char_counts):
        proportion = chars / total_chars if total_chars > 0 else 1.0 / len(word_dicts)
        duration = _MIN_WORD_DURATION + available * proportion
        redistributed.append({
            "word": word_dict["word"],
            "start": round(cursor, 3),
            "end": round(cursor + duration, 3),
        })
        cursor += duration

    logger.info(
        "Redistributed %d words across %.2fs audio (proportional by character count)",
        len(redistributed), audio_duration,
    )
    return redistributed


def qwen_output_to_whisper_result(
    qwen_output: Any,
    language: str = "en",
    audio_path: Optional[str] = None,
) -> "WhisperResult":
    """Convert a Qwen3-ASR transcription output to a ``stable_whisper.WhisperResult``.

    Parameters
    ----------
    qwen_output:
        A single element from the list returned by ``model.transcribe(...)``.
        Expected attributes:
          - ``.text``  -- full transcription string
          - ``.language``  -- detected language
          - ``.time_stamps``  -- ``ForcedAlignResult`` iterable of
            ``ForcedAlignItem(text, start_time, end_time)`` objects.
    language:
        ISO 639-1 language code to store in the result (default ``"en"``).
    audio_path:
        Optional path to the audio file.  When provided, ``stable_whisper.transcribe_any()``
        is used for segment reconstruction (better regrouping than manual splitting).
    """
    from stable_whisper import WhisperResult  # type: ignore[import-not-found]

    full_text: str = getattr(qwen_output, "text", "") or ""
    raw_stamps = getattr(qwen_output, "time_stamps", None)

    # Merge punctuation from the master text onto the timestamped words.
    # This uses character-position scanning rather than token matching,
    # which correctly handles punctuation, contractions, and edge cases.
    # The merge function reads .text/.start_time/.end_time from ForcedAlignItems.
    final_words = merge_master_with_timestamps(full_text, raw_stamps or [])

    # Fix zero-duration words: the forced aligner often returns point
    # timestamps where start_time == end_time.  Scan forward for the next
    # word with a strictly later start_time so consecutive items that share
    # the same timestamp all get a proper end (not another zero-duration).
    _DEFAULT_WORD_DUR = 0.3  # seconds
    for i, w in enumerate(final_words):
        if w["end"] <= w["start"]:
            next_start = None
            for j in range(i + 1, len(final_words)):
                if final_words[j]["start"] > w["start"]:
                    next_start = final_words[j]["start"]
                    break
            if next_start is not None:
                w["end"] = next_start
            else:
                w["end"] = w["start"] + _DEFAULT_WORD_DUR

    if not final_words:
        logger.warning(
            "Qwen output contains no word timestamps; "
            "creating a single segment from the full text."
        )
        segments = [
            {
                "start": 0.0,
                "end": 0.0,
                "text": full_text,
                "words": None,
                "no_speech_prob": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
            }
        ]
        result_dict: Dict[str, Any] = {
            "language": language,
            "segments": segments,
        }
        return WhisperResult(result_dict, check_sorted=False)

    # Assess alignment quality and redistribute if collapsed.
    audio_duration = 0.0
    if audio_path:
        from subgen.media.scene_detection import get_audio_duration
        audio_duration = get_audio_duration(audio_path)

    if audio_duration <= 0 and final_words:
        # Estimate from the last word's end time as fallback.
        audio_duration = max(w["end"] for w in final_words)

    quality = _assess_alignment_quality(final_words, audio_duration)
    if quality["status"] == "COLLAPSED":
        logger.warning("Applying proportional redistribution due to alignment collapse")
        final_words = _redistribute_collapsed_words(final_words, audio_duration)

    # Use stable_whisper.transcribe_any() for proper segment reconstruction
    # when audio_path is available.  This gives us sentence splitting, gap
    # detection, character limits, and duration caps automatically.
    if audio_path:
        # Capture final_words in a closure for the precomputed inference function.
        _precomputed_words = list(final_words)

        def precomputed_inference(audio: str, **kwargs: Any) -> List[List[Dict[str, Any]]]:
            return [_precomputed_words]

        try:
            result = stable_whisper.transcribe_any(
                inference_func=precomputed_inference,
                audio=str(audio_path),
                audio_type="str",
                regroup=True,
                vad=False,
                demucs=False,
                suppress_silence=True,
                suppress_word_ts=True,
                force_order=True,
                verbose=False,
            )
            # Override the language (transcribe_any may not set it correctly).
            result.language = language

            logger.debug(
                "Adapted Qwen output via transcribe_any: %d segments, %d words, language='%s'",
                len(result.segments),
                len(final_words),
                language,
            )
            return result
        except Exception:
            logger.warning(
                "stable_whisper.transcribe_any() failed; falling back to manual grouping",
                exc_info=True,
            )

    # Fallback: use the manual grouping function (same as Parakeet path).
    segments = _group_words_into_segments(final_words)

    result_dict = {
        "language": language,
        "segments": segments,
    }

    logger.debug(
        "Adapted Qwen output: %d segments, %d words, language='%s'",
        len(segments),
        len(final_words),
        language,
    )

    return WhisperResult(result_dict, check_sorted=False)
