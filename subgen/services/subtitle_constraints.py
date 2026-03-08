"""Subtitle display constraint enforcement.

Splits oversized segments and fixes overlapping timestamps so that
every subtitle conforms to standard display limits.

Wall-clock cap enforcement (ported from WhisperJAV):
    ``enforce_wall_clock_cap()`` splits any segment whose wall-clock span
    (``segment.end - segment.start``) exceeds a configurable limit (default
    8 seconds).  stable-ts's ``sd=`` regroup operator measures cumulative
    *speech* time (sum of word durations), NOT wall-clock time.  When
    inter-word gaps push screen time past the cap without triggering ``sd``,
    this post-pass catches those cases and splits at word boundaries using
    an even-split strategy.
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

# Wall-clock duration cap (seconds)
MAX_SUBTITLE_WALL_CLOCK: float = 8.0

_METADATA_MARKER: str = "transcribed by whisperai with faster-whisper"
_MIN_SEGMENT_DURATION: float = 0.05  # 50ms floor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_segment_from_words(
    word_group: list, original_segment: object
) -> StableSegment:
    """Create a StableSegment from a group of WordTiming objects.

    Preserves metadata from the original segment (speaker, logprob,
    no_speech_prob, compression_ratio, temperature, id) so downstream
    processing and filtering work correctly.
    """
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

    # Copy metadata from the original segment so filters and exporters
    # still have access to speech confidence, speaker labels, etc.
    _METADATA_ATTRS: List[str] = [
        "speaker", "no_speech_prob", "avg_logprob",
        "compression_ratio", "temperature", "id",
    ]
    for attr in _METADATA_ATTRS:
        value = getattr(original_segment, attr, None)
        if value is not None:
            setattr(sub_seg, attr, value)

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
# Wall-clock duration cap (post-regrouping enforcement)
# ---------------------------------------------------------------------------


def enforce_wall_clock_cap(
    result: object, max_duration: float = MAX_SUBTITLE_WALL_CLOCK
) -> int:
    """Split segments that exceed *max_duration* wall-clock seconds.

    Supplements stable-ts's ``sd`` regroup operator, which caps cumulative
    *speech* time (sum of word durations) rather than wall-clock time
    (``segment.end - segment.start``).  Inter-word gaps can push screen
    time past the cap without triggering ``sd``; this pass catches those.

    Uses an even-split strategy: divide the segment into the minimum number
    of parts so each part is under *max_duration*, then find the word
    boundary closest to each split point.

    Args:
        result: A ``stable_whisper.WhisperResult`` to modify in-place.
        max_duration: Maximum wall-clock seconds per segment (default 8.0).

    Returns:
        Number of segments that were split.
    """
    if not result or not hasattr(result, "segments") or not result.segments:
        return 0

    splits_made: int = 0
    i: int = 0

    while i < len(result.segments):
        seg = result.segments[i]
        wall_clock: float = seg.end - seg.start

        words = getattr(seg, "words", None)
        if wall_clock <= max_duration or not words or len(words) < 2:
            i += 1
            continue

        # How many parts are needed so each is <= max_duration?
        n_parts: int = int(-(-wall_clock // max_duration))  # ceil division
        if wall_clock / n_parts > max_duration:
            n_parts += 1
        if n_parts < 2:
            i += 1
            continue

        target_dur: float = wall_clock / n_parts

        # Find word indices for split boundaries (wall-clock based).
        # Each index is the last word of a sub-segment.
        split_indices: List[int] = []
        for p in range(1, n_parts):
            target_time: float = seg.start + p * target_dur
            best_idx: int = 0
            best_diff: float = float("inf")
            for wi in range(len(words) - 1):  # never split after last word
                diff: float = abs(words[wi].end - target_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = wi
            split_indices.append(best_idx)

        split_indices = sorted(set(split_indices))
        if not split_indices:
            i += 1
            continue

        # Build word groups from split indices
        word_groups: List[list] = []
        prev_idx: int = 0
        for idx in split_indices:
            word_groups.append(list(words[prev_idx : idx + 1]))
            prev_idx = idx + 1
        # Remaining words
        word_groups.append(list(words[prev_idx:]))

        # Filter out empty groups
        word_groups = [g for g in word_groups if g]
        if len(word_groups) < 2:
            i += 1
            continue

        # Build new segments from word groups
        new_segments: List[StableSegment] = [
            _build_segment_from_words(group, seg) for group in word_groups
        ]

        # Replace the original segment
        result.segments[i : i + 1] = new_segments
        splits_made += 1
        # Don't increment i — re-check first new segment in case
        # it still exceeds the cap (e.g., single very long word).

    if splits_made > 0:
        logger.info(
            "Wall-clock cap: split %d segment(s) exceeding %.1fs",
            splits_made,
            max_duration,
        )

    return splits_made


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
