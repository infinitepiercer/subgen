"""Post-transcription subtitle filter.

Removes hallucinated and junk segments that Whisper sometimes generates
during silence.  Enabled via ``FILTER_SUBTITLES=true``.

All filtering is based on Whisper's own confidence scores
(``no_speech_prob``, ``avg_logprob``, ``compression_ratio``) rather than
hardcoded word/phrase lists, so legitimate speech is never suppressed.
"""

import logging

logger = logging.getLogger(__name__)

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

    return None
