"""Speaker diarization service: assign speaker labels to transcription segments via WeSpeaker.

Uses the WeSpeaker toolkit to perform speaker diarization on audio and then
maps speaker labels back onto the stable-ts transcription result.  Segments
that span a speaker-change boundary are split at word boundaries so each
output segment belongs to exactly one speaker.
"""

import logging
import os
import tempfile
from typing import List, Tuple, Union

from subgen.config import enable_diarization  # noqa: F401 — imported for caller convenience
from subgen.models import diarization_model as diarization_model_module
from subgen.models.diarization_model import start_diarization_model

logger = logging.getLogger(__name__)

# Type alias for a single diarization segment returned by wespeaker.
# Each tuple is (utt, start_seconds, end_seconds, speaker_label).
DiarSegment = Tuple[str, float, float, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_diarization_model(device: str) -> None:
    """Lazy-load the WeSpeaker model if it has not been loaded yet."""
    if diarization_model_module.model is None:
        logger.info("Diarization model not loaded — loading now on %s", device)
        start_diarization_model(device)


def _write_temp_wav(audio_bytes: Union[bytes, bytearray]) -> str:
    """Write raw audio bytes to a temporary WAV file and return the path.

    The caller is responsible for deleting the file when done.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
    finally:
        tmp.close()
    logger.debug("Wrote temporary WAV for diarization: %s", tmp.name)
    return tmp.name


def _overlap(seg_start: float, seg_end: float, diar_start: float, diar_end: float) -> float:
    """Return the duration of temporal overlap between two intervals."""
    return max(0.0, min(seg_end, diar_end) - max(seg_start, diar_start))


def _best_speaker_for_interval(
    start: float, end: float, diar_segments: List[DiarSegment]
) -> str:
    """Find the diarization speaker with the most overlap for the given interval."""
    best_label: str = "Unknown"
    best_overlap: float = 0.0

    for _utt, d_start, d_end, d_label in diar_segments:
        ov = _overlap(start, end, d_start, d_end)
        if ov > best_overlap:
            best_overlap = ov
            best_label = f"Speaker {d_label}"

    return best_label


def _assign_speakers(result: object, diar_segments: List[DiarSegment]) -> None:
    """Assign a speaker label to every segment based on maximum temporal overlap.

    Sets a ``speaker`` attribute on each segment object.
    """
    if not diar_segments:
        logger.warning("Diarization returned no segments — all speakers set to 'Unknown'")
        for segment in result.segments:
            segment.speaker = "Unknown"
        return

    for segment in result.segments:
        segment.speaker = _best_speaker_for_interval(
            segment.start, segment.end, diar_segments
        )


def _overlapping_diar_segments(
    seg_start: float, seg_end: float, diar_segments: List[DiarSegment]
) -> List[DiarSegment]:
    """Return all diarization segments that overlap the given time range."""
    return [
        ds for ds in diar_segments
        if _overlap(seg_start, seg_end, ds[1], ds[2]) > 0.0
    ]


def _split_multi_speaker_segments(result: object, diar_segments: List[DiarSegment]) -> None:
    """Split transcription segments that span a speaker-change boundary.

    For each Whisper segment, if multiple diarization speakers overlap it we
    split at word boundaries.  Each resulting sub-segment receives the speaker
    label of the diarization segment that overlaps the word's timestamp range.

    The original segment list on *result* is replaced in-place.
    """
    if not diar_segments:
        return

    new_segments: list = []

    for segment in result.segments:
        overlapping = _overlapping_diar_segments(segment.start, segment.end, diar_segments)

        # Collect unique speakers that overlap this segment
        unique_speakers = {ds[3] for ds in overlapping}

        # If zero or one speaker — no split needed
        if len(unique_speakers) <= 1:
            new_segments.append(segment)
            continue

        # Need word-level timestamps to split accurately
        words = getattr(segment, "words", None)
        if not words or len(words) == 0:
            logger.debug(
                "Segment %.1fs-%.1fs spans multiple speakers but has no word timestamps; "
                "keeping as single segment",
                segment.start,
                segment.end,
            )
            new_segments.append(segment)
            continue

        # Group consecutive words by their speaker
        groups: List[Tuple[str, List]] = []
        current_speaker: str = ""
        current_words: list = []

        for word in words:
            word_start = getattr(word, "start", segment.start)
            word_end = getattr(word, "end", segment.end)
            speaker = _best_speaker_for_interval(word_start, word_end, diar_segments)

            if speaker != current_speaker and current_words:
                groups.append((current_speaker, current_words))
                current_words = []

            current_speaker = speaker
            current_words.append(word)

        if current_words:
            groups.append((current_speaker, current_words))

        # If all words ended up with the same speaker, no actual split
        if len(groups) <= 1:
            new_segments.append(segment)
            continue

        # Build new pseudo-segments for each speaker group
        for speaker_label, word_group in groups:
            # Create a lightweight object that mimics a segment
            sub_seg = _SubSegment(
                start=getattr(word_group[0], "start", segment.start),
                end=getattr(word_group[-1], "end", segment.end),
                text=" ".join(getattr(w, "word", "").strip() for w in word_group),
                words=word_group,
                speaker=speaker_label,
            )
            new_segments.append(sub_seg)

        logger.debug(
            "Split segment %.1fs-%.1fs into %d sub-segments across %d speakers",
            segment.start,
            segment.end,
            len(groups),
            len(unique_speakers),
        )

    result.segments = new_segments


class _SubSegment:
    """Minimal segment-like object used when splitting multi-speaker segments."""

    __slots__ = ("start", "end", "text", "words", "speaker")

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: list,
        speaker: str,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.speaker = speaker


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def add_speaker_labels(result: object, audio_data: Union[str, bytes, bytearray], device: str) -> int:
    """Run speaker diarization and annotate transcription segments with speaker labels.

    This is the main entry point for diarization.  It:
      1. Ensures the WeSpeaker model is loaded.
      2. Runs diarization on the audio.
      3. Assigns speaker labels to each segment (maximum overlap).
      4. Splits segments that span a speaker-change boundary at word boundaries.
      5. Prepends ``[Speaker N]`` to each segment's text.

    Args:
        result: A ``stable_whisper`` transcription result whose ``.segments``
            attribute yields objects with ``.text``, ``.start``, ``.end``,
            and ``.words``.
        audio_data: Either a file path (``str``) or raw audio bytes
            (``bytes`` / ``bytearray``).
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        The number of unique speakers found.
    """
    _ensure_diarization_model(device)

    temp_path: str | None = None

    try:
        # Resolve audio to a file path for WeSpeaker
        if isinstance(audio_data, (bytes, bytearray)):
            temp_path = _write_temp_wav(audio_data)
            audio_path = temp_path
        else:
            audio_path = audio_data

        logger.info("Running WeSpeaker diarization on %s", audio_path)
        diar_segments: List[DiarSegment] = diarization_model_module.model.diarize(audio_path)

        if not diar_segments:
            logger.warning("WeSpeaker returned no diarization segments")
            return 0

        logger.info("WeSpeaker returned %d diarization segments", len(diar_segments))

        # Assign speakers and split multi-speaker segments
        _assign_speakers(result, diar_segments)
        _split_multi_speaker_segments(result, diar_segments)

        # Count unique speakers (labels are kept internal, not shown in subtitle text)
        unique_speakers: set = set()
        for segment in result.segments:
            speaker = getattr(segment, "speaker", "Unknown")
            unique_speakers.add(speaker)

        speaker_count = len(unique_speakers)
        logger.info("Diarization complete: %d unique speaker(s) found", speaker_count)
        return speaker_count

    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
                logger.debug("Cleaned up temporary WAV: %s", temp_path)
            except OSError as exc:
                logger.warning("Failed to remove temp WAV %s: %s", temp_path, exc)
