"""Post-transcription subtitle filter.

Removes hallucinated and junk segments that Whisper sometimes generates
during silence.  Enabled via ``FILTER_SUBTITLES=true``.

Confidence-based filtering uses Whisper's own scores (``no_speech_prob``,
``avg_logprob``, ``compression_ratio``).

Non-verbal filtering (enabled via ``DROP_NONVERBAL_SEGMENTS=true``)
detects music cues, sound effects, laughs, moans, and other non-speech
segments via keyword matching and simple vocal detection.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-verbal segment detection (ported from WhisperJAV SegmentFilterHelper)
# ---------------------------------------------------------------------------

_NONVERBAL_KEYWORDS: set[str] = {
    "music", "applause", "laugh", "laughs", "laughter",
    "sfx", "fx", "noise", "silence", "ambient",
    "moan", "moans", "moaning", "groan", "groans",
    "sigh", "sighs", "breath", "breathing",
}

_NOTE_CHARACTERS: set[str] = set("♪♫")

_SIMPLE_VOCAL_CHARSET: set[str] = set("ahmnou")

_SIMPLE_VOCAL_IGNORES: set[str] = set("!?,.~... ")

_SIMPLE_VOCAL_MAX_LENGTH: int = 6


def _looks_nonverbal(text: str) -> bool:
    """Return True if the segment text appears to be non-verbal content."""
    stripped = text.strip()
    if not stripped:
        return False

    # Pure note characters or punctuation
    if all(ch in _NOTE_CHARACTERS or ch in _SIMPLE_VOCAL_IGNORES for ch in stripped):
        return True

    lowered = stripped.lower()

    # Strip bracket descriptors like [music] or (laughs)
    collapsed = lowered.strip()
    while collapsed and collapsed[0] in "[](){}<>":
        collapsed = collapsed[1:]
    while collapsed and collapsed[-1] in "[](){}<>":
        collapsed = collapsed[:-1]
    collapsed = collapsed.strip()

    if not collapsed:
        return False

    # Keyword matching
    for keyword in _NONVERBAL_KEYWORDS:
        if keyword in collapsed:
            return True

    # Simple vocal detection (short moans/grunts like "ahh", "mmm")
    simplified = "".join(ch for ch in collapsed if ch not in _SIMPLE_VOCAL_IGNORES)
    if simplified and len(simplified) <= _SIMPLE_VOCAL_MAX_LENGTH:
        if all(ch in _SIMPLE_VOCAL_CHARSET for ch in simplified):
            return True

    return False

# ---------------------------------------------------------------------------
# Confidence thresholds for detecting hallucinated segments.
#
# stable-ts / faster-whisper expose these on every segment object.
# A segment is flagged as a hallucination when BOTH conditions are true:
#   - no_speech_prob  > threshold   (model thinks it's silence)
#   - avg_logprob     < threshold   (model is very uncertain about the text)
#
# Compression ratio catches repetitive garbage (e.g. "the the the the...")
# that Whisper generates in loops.
#
# Based on community research:
#   https://github.com/openai/whisper/discussions/679
#   https://github.com/SYSTRAN/faster-whisper/issues/621
# ---------------------------------------------------------------------------

_NO_SPEECH_PROB_THRESHOLD: float = 0.7
_AVG_LOGPROB_THRESHOLD: float = -1.0
_COMPRESSION_RATIO_THRESHOLD: float = 2.4

# Metadata line injected by appendLine() — must never be filtered
_METADATA_MARKER = "transcribed by whisperai with faster-whisper"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_segments(result) -> int:
    """Remove hallucinated / junk segments from a transcription result.

    Iterates segments in reverse so deletion indices stay valid.

    Args:
        result: The stable-ts ``WhisperResult`` (has a ``.segments`` list).

    Returns:
        Number of segments removed.
    """
    segments = result.segments
    removed = 0

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        text: str = seg.text.strip()

        reason = _check_segment(seg, text)
        if reason is not None:
            logger.debug(
                "Filtered segment #%d [%.2f-%.2f] (%s): %r",
                i, seg.start, seg.end, reason, text,
            )
            del segments[i]
            removed += 1

    if removed:
        logger.info("Filtered %d hallucination/junk segment(s)", removed)

    return removed


# ---------------------------------------------------------------------------
# Internal checks — return a reason string or None
# ---------------------------------------------------------------------------


def _check_segment(seg, text: str) -> str | None:
    """Return a reason string if the segment should be filtered, else ``None``.

    All checks use Whisper's own confidence metrics rather than hardcoded
    word lists.

    Args:
        seg: The stable-ts segment object (has ``no_speech_prob``,
             ``avg_logprob``, ``compression_ratio``).
        text: The stripped segment text.
    """

    # --- Safety: never filter the metadata line ---
    if _METADATA_MARKER in text.lower():
        return None

    # --- Empty / whitespace-only segments ---
    if not text:
        return "empty"

    # --- Confidence-based hallucination detection ---
    # A segment is hallucinated when the model simultaneously thinks it's
    # silence (high no_speech_prob) AND is very uncertain about the text
    # (low avg_logprob).
    no_speech_prob = getattr(seg, 'no_speech_prob', None)
    avg_logprob = getattr(seg, 'avg_logprob', None)

    if (
        no_speech_prob is not None
        and avg_logprob is not None
        and no_speech_prob > _NO_SPEECH_PROB_THRESHOLD
        and avg_logprob < _AVG_LOGPROB_THRESHOLD
    ):
        return (
            f"low_confidence:no_speech={no_speech_prob:.2f},"
            f"avg_logprob={avg_logprob:.2f}"
        )

    # --- Compression ratio check ---
    # High compression ratio indicates repetitive/looping text that Whisper
    # generates when stuck (e.g. "the the the the the...").
    compression_ratio = getattr(seg, 'compression_ratio', None)

    if (
        compression_ratio is not None
        and compression_ratio > _COMPRESSION_RATIO_THRESHOLD
    ):
        return f"high_compression_ratio:{compression_ratio:.2f}"

    # --- Non-verbal segment detection ---
    # Filters music cues, sound effects, laughs, moans, etc.
    from subgen.config import drop_nonverbal_segments
    if drop_nonverbal_segments and _looks_nonverbal(text):
        return "nonverbal"

    return None
