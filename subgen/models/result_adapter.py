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
from typing import Any, Dict, List

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
) -> "WhisperResult":
    """Convert a NeMo Parakeet transcription output to a ``stable_whisper.WhisperResult``.

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

    Returns
    -------
    stable_whisper.WhisperResult
        A result object that is fully compatible with the downstream subgen
        pipeline (diarization, subtitle filter, translation, SRT/VTT output).
    """
    from stable_whisper import WhisperResult  # type: ignore[import-not-found]

    # Extract word timestamps from the NeMo output.
    full_text: str = getattr(nemo_output, "text", "") or ""
    timestamp_data: Dict[str, Any] = getattr(nemo_output, "timestamp", {}) or {}
    word_timestamps: List[Dict[str, Any]] = timestamp_data.get("word", [])

    if not word_timestamps:
        logger.warning(
            "Parakeet output contains no word timestamps; "
            "creating a single segment from the full text."
        )
        # Fall back to a single segment covering the whole utterance.
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
    else:
        # Parakeet-TDT outputs punctuated text in full_text but word
        # timestamps may lack punctuation.  Restore it before grouping.
        _restore_punctuation_from_text(full_text, word_timestamps)
        segments = _group_words_into_segments(word_timestamps)

    # Build the dict that WhisperResult._standardize_result expects.
    result_dict: Dict[str, Any] = {
        "language": language,
        "segments": segments,
    }

    logger.debug(
        "Adapted Parakeet output: %d segments, %d words, language='%s'",
        len(segments),
        len(word_timestamps),
        language,
    )

    return WhisperResult(result_dict, check_sorted=False)


def qwen_output_to_whisper_result(
    qwen_output: Any,
    language: str = "en",
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
    """
    from stable_whisper import WhisperResult  # type: ignore[import-not-found]

    full_text: str = getattr(qwen_output, "text", "") or ""
    raw_stamps = getattr(qwen_output, "time_stamps", None)

    # ForcedAlignResult is a flat iterable of ForcedAlignItem objects,
    # each with .text, .start_time, .end_time attributes.
    word_timestamps: List[Dict[str, Any]] = []
    if raw_stamps:
        for item in raw_stamps:
            word_timestamps.append({
                "word": getattr(item, "text", str(item)),
                "start": float(getattr(item, "start_time", 0.0)),
                "end": float(getattr(item, "end_time", 0.0)),
            })

    # Fix zero-duration words: when start_time == end_time the aligner
    # only produced point timestamps.  Estimate end from the next word's
    # start, with a small default gap for the last word.
    _DEFAULT_WORD_DUR = 0.3  # seconds
    for i, w in enumerate(word_timestamps):
        if w["end"] <= w["start"]:
            if i + 1 < len(word_timestamps):
                w["end"] = word_timestamps[i + 1]["start"]
            else:
                w["end"] = w["start"] + _DEFAULT_WORD_DUR

    if not word_timestamps:
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
    else:
        segments = _group_words_into_segments(word_timestamps)

    result_dict: Dict[str, Any] = {
        "language": language,
        "segments": segments,
    }

    logger.debug(
        "Adapted Qwen output: %d segments, %d words, language='%s'",
        len(segments),
        len(word_timestamps),
        language,
    )

    return WhisperResult(result_dict, check_sorted=False)
