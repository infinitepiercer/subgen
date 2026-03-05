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

# Maximum segment duration in seconds before forcing a split.
_MAX_SEGMENT_DURATION: float = 10.0


def _group_words_into_segments(
    word_timestamps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group Parakeet word timestamps into subtitle segments.

    Splits on sentence-ending punctuation (``.``, ``!``, ``?``) or when a
    segment exceeds ``_MAX_SEGMENT_DURATION`` seconds.

    Each returned segment dict is ready for ``stable_whisper.WhisperResult``
    consumption::

        {
            "start": float,
            "end": float,
            "text": str,
            "words": [{"word": str, "start": float, "end": float, "probability": float}, ...],
            "no_speech_prob": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 1.0,
        }
    """
    if not word_timestamps:
        return []

    segments: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []
    segment_start: float = word_timestamps[0].get("start", 0.0)

    for raw_word in word_timestamps:
        word_text: str = raw_word.get("word", "")
        word_start: float = float(raw_word.get("start", 0.0))
        word_end: float = float(raw_word.get("end", 0.0))

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

        # Decide whether to close the current segment.
        duration = word_end - segment_start
        ends_sentence = _SENTENCE_ENDERS.search(word_text.rstrip())
        exceeds_duration = duration >= _MAX_SEGMENT_DURATION

        if ends_sentence or exceeds_duration:
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
            current_words = []
            segment_start = word_end  # next segment starts after this word

    # Flush any remaining words into a final segment.
    if current_words:
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
                "words": [],
                "no_speech_prob": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
            }
        ]
    else:
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
