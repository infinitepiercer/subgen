"""Subtitle display constraint enforcement.

Splits oversized segments and fixes overlapping timestamps so that
every subtitle conforms to standard display limits.
"""

import logging
from typing import List, Optional

from stable_whisper.result import Segment as StableSegment

logger = logging.getLogger(__name__)

# Display limits
MAX_CHARS_PER_LINE: int = 42
MAX_LINES_PER_SUBTITLE: int = 2
MAX_CHARS_PER_SEGMENT: int = 84  # 42 * 2
MIN_GAP_BETWEEN_SUBTITLES: float = 0.083  # 2 frames at 24fps

_METADATA_MARKER: str = "transcribed by whisperai with faster-whisper"
_MIN_SEGMENT_DURATION: float = 0.05  # 50ms floor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_segment_from_words(
    word_group: list, original_segment: object
) -> StableSegment:
    """Create a StableSegment from a group of WordTiming objects."""
    speaker_label: Optional[str] = getattr(original_segment, "speaker", None)

    word_dicts = [
        {
            "word": getattr(w, "word", ""),
            "start": getattr(w, "start", original_segment.start),
            "end": getattr(w, "end", original_segment.end),
            "probability": getattr(w, "probability", None),
        }
        for w in word_group
    ]
    sub_seg = StableSegment(words=word_dicts, ignore_unused_args=True)
    sub_seg.speaker = speaker_label
    return sub_seg


def _split_segment_by_char_limit(segment: object) -> List[StableSegment]:
    """Split one segment into sub-segments that fit within MAX_CHARS_PER_SEGMENT."""
    words = list(segment.words)
    speaker: Optional[str] = getattr(segment, "speaker", None)

    groups: List[list] = []
    current_group: list = []
    current_chars: int = 0

    for word in words:
        word_text: str = getattr(word, "word", "")
        if not current_group:
            # First word in group — count without leading whitespace
            add_chars = len(word_text.lstrip())
        else:
            # Subsequent words — count with leading space as-is
            add_chars = len(word_text)

        if current_group and current_chars + add_chars > MAX_CHARS_PER_SEGMENT:
            groups.append(current_group)
            current_group = []
            current_chars = 0
            # Re-count this word as first in the new group
            add_chars = len(word_text.lstrip())

        current_group.append(word)
        current_chars += add_chars

    if current_group:
        groups.append(current_group)

    sub_segments: List[StableSegment] = []
    for group in groups:
        sub_seg = _build_segment_from_words(group, segment)
        if speaker is not None:
            sub_seg.speaker = speaker
        sub_segments.append(sub_seg)

    return sub_segments


def _split_long_segments(segments: list) -> List[object]:
    """Return a new list with oversized segments split to fit display limits."""
    result: List[object] = []
    split_count: int = 0

    for segment in segments:
        text: str = segment.text.strip()

        # Skip metadata segments
        if _METADATA_MARKER in text.lower():
            result.append(segment)
            continue

        # Skip segments without word-level data
        words = getattr(segment, "words", None)
        if not words or len(words) == 0:
            result.append(segment)
            continue

        if len(text) <= MAX_CHARS_PER_SEGMENT:
            result.append(segment)
            continue

        # Split oversized segment
        sub_segments = _split_segment_by_char_limit(segment)
        result.extend(sub_segments)
        split_count += 1

    if split_count > 0:
        logger.info("Split %d oversized segment(s) to fit display limits", split_count)

    return result


def _fix_overlaps_and_gaps(segments: list) -> int:
    """Fix overlapping timestamps and ensure minimum gaps between segments.

    Returns the number of overlaps fixed.
    """
    overlaps_fixed: int = 0

    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]

        gap: float = next_seg.start - current.end
        if gap < MIN_GAP_BETWEEN_SUBTITLES:
            new_end: float = next_seg.start - MIN_GAP_BETWEEN_SUBTITLES
            # Floor to prevent negative or zero duration
            min_end: float = current.start + _MIN_SEGMENT_DURATION
            current.end = max(new_end, min_end)
            overlaps_fixed += 1

    return overlaps_fixed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enforce_display_constraints(result: object) -> None:
    """Enforce subtitle display constraints on a WhisperResult in-place.

    Splits oversized segments and fixes overlapping timestamps.
    """
    segments = result.segments

    # Split long segments
    segments = _split_long_segments(segments)

    # Fix overlaps and gaps
    overlaps_fixed: int = _fix_overlaps_and_gaps(segments)

    # Update result in-place
    result.segments = segments

    if overlaps_fixed > 0:
        logger.info("Fixed %d overlap(s) / insufficient gap(s)", overlaps_fixed)

    logger.debug(
        "Display constraint enforcement complete: %d segments", len(segments)
    )
