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
_MAX_SEGMENT_DURATION: float = 7.0

# Minimum segment duration before allowing a comma-based split.
_MIN_SEGMENT_FOR_CLAUSE_SPLIT: float = 3.0

# Gap (seconds) between the end of one word and the start of the next
# that triggers a segment break (breath / speaker change).
_GAP_SPLIT_THRESHOLD: float = 0.5


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
    """Map punctuation from the model's full text back onto word timestamps.

    Parakeet-TDT outputs punctuated text in ``output.text`` but the
    individual word timestamps often contain bare words without trailing
    punctuation.  This function aligns the two by index and transfers any
    trailing punctuation (e.g. ``,`` ``.`` ``!`` ``?``) onto the word
    timestamp entries so downstream splitting can use it.

    Modifies *word_timestamps* in place.
    """
    if not full_text or not word_timestamps:
        return

    tokens = full_text.split()
    if len(tokens) != len(word_timestamps):
        # Lengths don't match — try best-effort positional alignment.
        # Walk both lists, matching stripped words.
        ti = 0
        for wt in word_timestamps:
            if ti >= len(tokens):
                break
            bare_word = wt.get("word", "").strip().lower()
            # Advance through tokens to find a match
            for look_ahead in range(min(3, len(tokens) - ti)):
                token = tokens[ti + look_ahead]
                token_bare = re.sub(r"[^\w']+$", "", token).lower()
                if token_bare == bare_word:
                    wt["word"] = token
                    ti = ti + look_ahead + 1
                    break
            else:
                ti += 1
        return

    # Perfect 1:1 alignment — just copy token text (preserving punctuation)
    for token, wt in zip(tokens, word_timestamps):
        wt["word"] = token


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
            segment_start = word_end  # next segment starts after this word

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
