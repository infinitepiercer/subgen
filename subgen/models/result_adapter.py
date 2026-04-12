"""Adapter to convert ASR output (Parakeet) into a stable_whisper.WhisperResult.

The Parakeet model returns a dataclass-like object with:
  - ``output.text``  -- full transcription string
  - ``output.timestamp['word']``  -- list of dicts [{word, start, end}, ...]

This module groups those word timestamps into sentence-level segments and
constructs a genuine ``stable_whisper.WhisperResult`` so the entire
downstream pipeline (diarization, subtitle filter, translation, SRT output)
works without modification.

Post-alignment hardening pipeline (ported from WhisperJAV hardening.py):
  - VAD-guided collapse recovery — distributes words within speech regions
  - Null/failed timestamp interpolation — fills gaps between valid anchors
  - Boundary clamping — prevents timestamps from exceeding audio duration
  - Chronological sorting — ensures strict time ordering after recovery
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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
    segment_text = "".join(w["word"] for w in current_words).lstrip()
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

    Uses an improved pipeline: position-based punctuation merge, zero-duration
    fix, alignment sentinel, and ``transcribe_any()`` reconstruction when
    *audio_path* is available.

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

    # Word-level cleaning AFTER merge so timestamps stay aligned.
    from subgen.config import clean_text
    if clean_text and final_words:
        from subgen.services.text_cleaner import clean_word_list
        final_words = clean_word_list(final_words)

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

    # Import VAD speech regions for collapse recovery (Strategy B).
    from subgen.media.scene_detection import get_speech_regions
    speech_regions_list = get_speech_regions()
    vad_regions: Optional[List[Tuple[float, float]]] = None
    if speech_regions_list:
        vad_regions = [(r.start, r.end) for r in speech_regions_list]

    collapsed = False
    quality = _assess_alignment_quality(final_words, audio_duration)
    if quality["status"] == "COLLAPSED":
        logger.warning("Parakeet: applying redistribution due to alignment collapse")
        final_words = _redistribute_collapsed_words(
            final_words, audio_duration, speech_regions=vad_regions,
        )
        collapsed = True

    # Apply hardening pipeline: interpolation → clamping → sorting.
    final_words = _harden_words(final_words, audio_duration)

    # Use stable_whisper.transcribe_any() for proper segment reconstruction
    # when audio_path is available.
    # suppress_silence=False for sentinel-recovered words to preserve the
    # recovery's timestamp distribution (JAV convention).
    use_suppress_silence = not collapsed
    if audio_path:
        _precomputed_words = list(final_words)

        def precomputed_inference(audio: str, **kwargs: Any) -> List[List[Dict[str, Any]]]:
            return [_precomputed_words]

        try:
            result = stable_whisper.transcribe_any(
                inference_func=precomputed_inference,
                audio=str(audio_path),
                audio_type="str",
                regroup=False,
                vad=False,
                demucs=False,
                suppress_silence=use_suppress_silence,
                suppress_word_ts=use_suppress_silence,
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


def _find_word_boundary(
    text: str, word: str, start_pos: int, case_insensitive: bool = False,
) -> int:
    """Find *word* in *text* starting at *start_pos*, requiring word boundaries.

    A match is accepted only when:
      - The character before the match is a space, punctuation, or start-of-string.
      - The character after the match is a space, punctuation, or end-of-string.

    Returns the match position, or ``-1`` if no word-boundary match is found.
    """
    _BOUNDARY_CHARS = set(" \t\n\r,.!?;:\"'()-/")
    search_text = text.lower() if case_insensitive else text
    search_word = word.lower() if case_insensitive else word
    pos = start_pos
    while True:
        idx = search_text.find(search_word, pos)
        if idx == -1:
            return -1
        # Check left boundary
        left_ok = (idx == 0) or (text[idx - 1] in _BOUNDARY_CHARS)
        # Check right boundary
        end_idx = idx + len(word)
        right_ok = (end_idx >= len(text)) or (text[end_idx] in _BOUNDARY_CHARS)
        if left_ok and right_ok:
            return idx
        # Not a word boundary — advance past this match and keep searching.
        pos = idx + 1
    return -1


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
        The full punctuated transcription text from the ASR engine.
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

        word_start = _find_word_boundary(master_text, ts_word, master_pos)

        # Recovery: case-insensitive search when exact match fails.
        if word_start == -1:
            word_start = _find_word_boundary(master_text, ts_word, master_pos, case_insensitive=True)

        if word_start == -1:
            # All strategies failed — append word with leading space to
            # prevent concatenation artifacts ("gogogogo").
            word_text = ts_word if not result else " " + ts_word
            result.append({
                "word": word_text,
                "start": float(ts_start) if ts_start is not None else 0.0,
                "end": float(ts_end) if ts_end is not None else 0.0,
            })
            logger.debug(
                "merge: word %r not found at pos %d in master text (len=%d)",
                ts_word, master_pos, len(master_text),
            )
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

    total_chars = sum(len(w["word"].strip()) for w in word_dicts)
    if total_chars <= 10:
        return {"status": "OK", "details": "text too short for reliable assessment"}

    first_start = min(w["start"] for w in word_dicts)
    last_end = max(w["end"] for w in word_dicts)
    word_span = last_end - first_start
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
    word_dicts: List[Dict[str, Any]],
    audio_duration: float,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Redistribute words across the audio duration after alignment collapse.

    Supports two strategies:

    - **Strategy A** (proportional): Distributes words evenly across the full
      audio duration, proportional to character count.  Used as fallback when
      no VAD speech regions are available.
    - **Strategy B** (VAD-guided): Clips *speech_regions* to the redistribution
      window and distributes words only within speech portions, skipping
      silence gaps.  Words land on actual speech, not silence.

    Each word receives a minimum duration of 20ms regardless of strategy.

    Parameters
    ----------
    word_dicts:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
    audio_duration:
        Total audio duration in seconds.
    speech_regions:
        Optional VAD speech regions as ``[(start, end), ...]``, sorted by
        start time.  When provided, Strategy B (VAD-guided) is used.

    Returns
    -------
    New list of word dicts with redistributed timestamps.
    """
    if not word_dicts:
        return []

    _MIN_WORD_DURATION = 0.02  # 20ms minimum

    char_counts = [max(len(w["word"].strip()), 1) for w in word_dicts]
    total_chars = sum(char_counts)

    # Strategy B: VAD-guided distribution
    if speech_regions:
        clipped = _clip_speech_regions(speech_regions, 0.0, audio_duration)
        total_speech_dur = sum(end - start for start, end in clipped)

        if clipped and total_speech_dur > 0:
            redistributed: List[Dict[str, Any]] = []
            cumulative_chars = 0

            for word_dict, chars in zip(word_dicts, char_counts):
                frac_start = cumulative_chars / total_chars
                frac_end = (cumulative_chars + chars) / total_chars

                timeline_start = frac_start * total_speech_dur
                timeline_end = frac_end * total_speech_dur

                real_start = _timeline_to_real(timeline_start, clipped)
                real_end = _timeline_to_real(timeline_end, clipped)

                # Enforce minimum word duration
                if real_end - real_start < _MIN_WORD_DURATION:
                    real_end = real_start + _MIN_WORD_DURATION

                redistributed.append({
                    "word": word_dict["word"],
                    "start": round(real_start, 3),
                    "end": round(real_end, 3),
                })
                cumulative_chars += chars

            logger.info(
                "Redistributed %d words across %d speech regions "
                "(%.2fs speech in %.2fs audio, VAD-guided)",
                len(redistributed), len(clipped), total_speech_dur, audio_duration,
            )
            return redistributed

        # No usable speech regions — fall through to Strategy A
        logger.debug(
            "No usable speech regions after clipping; "
            "falling back to proportional redistribution"
        )

    # Strategy A: proportional distribution across full audio duration
    min_total = _MIN_WORD_DURATION * len(word_dicts)
    available = max(audio_duration - min_total, 0.0)

    redistributed = []
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


# ---------------------------------------------------------------------------
# Hardening helpers: speech region clipping and timeline mapping
# ---------------------------------------------------------------------------


def _clip_speech_regions(
    regions: List[Tuple[float, float]],
    window_start: float,
    window_end: float,
) -> List[Tuple[float, float]]:
    """Clip speech regions to a time window, returning only overlapping portions.

    Parameters
    ----------
    regions:
        VAD speech regions as ``[(start, end), ...]``.
    window_start:
        Start of the clipping window in seconds.
    window_end:
        End of the clipping window in seconds.

    Returns
    -------
    List of ``(start, end)`` tuples clipped to ``[window_start, window_end]``,
    filtered to only include regions with positive duration.
    """
    clipped: List[Tuple[float, float]] = []
    for region_start, region_end in regions:
        cs = max(region_start, window_start)
        ce = min(region_end, window_end)
        if ce > cs:
            clipped.append((cs, ce))
    return clipped


def _timeline_to_real(
    timeline_pos: float,
    regions: List[Tuple[float, float]],
) -> float:
    """Map a position in the flattened speech timeline to real time.

    The "speech timeline" is a continuous axis from 0 to ``total_speech_dur``
    (sum of all region durations).  This function maps a position on that
    axis back to real time, accounting for silence gaps between regions.

    Parameters
    ----------
    timeline_pos:
        Position in the flattened speech timeline (seconds).
    regions:
        Sorted list of ``(start_sec, end_sec)`` speech regions.

    Returns
    -------
    Real time in seconds.
    """
    cumulative = 0.0

    for region_start, region_end in regions:
        region_dur = region_end - region_start
        if region_dur <= 0:
            continue

        if cumulative + region_dur >= timeline_pos:
            offset_in_region = timeline_pos - cumulative
            return region_start + offset_in_region

        cumulative += region_dur

    # Past the end of all regions — clamp to last region end
    if regions:
        return regions[-1][1]
    return 0.0


# ---------------------------------------------------------------------------
# Hardening: null/failed timestamp interpolation
# ---------------------------------------------------------------------------


def _interpolate_null_timestamps(
    words: List[Dict[str, Any]],
    audio_duration: float,
) -> int:
    """Interpolate timestamps for words where the aligner returned null/zero values.

    Finds words with ``start == 0.0 and end == 0.0`` (aligner failure signature)
    that are surrounded by valid anchor words.  Distributes null-timestamp words
    proportionally by character count between the nearest valid anchors.

    Edge cases:
      - Leading nulls (before first anchor): distribute from 0.0 to first anchor
      - Trailing nulls (after last anchor): estimate from character count,
        capped to *audio_duration*
      - All nulls: return 0 (cannot interpolate without anchors)

    Parameters
    ----------
    words:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
        Modified in place.
    audio_duration:
        Total audio duration in seconds; used to cap trailing estimates.

    Returns
    -------
    Number of words that received interpolated timestamps.
    """
    if not words:
        return 0

    # Identify anchor indices (words with valid timestamps: end > 0)
    anchors: List[int] = [
        i for i, w in enumerate(words)
        if w["end"] > 0.0
    ]

    if not anchors:
        return 0

    interpolated_count = 0

    def _interpolate_gap(
        gap_indices: List[int], start_time: float, end_time: float,
    ) -> None:
        """Distribute words in gap_indices proportionally by char count."""
        nonlocal interpolated_count
        if not gap_indices:
            return

        total_chars = sum(max(len(words[i]["word"].strip()), 1) for i in gap_indices)
        gap_duration = end_time - start_time
        if gap_duration <= 0:
            gap_duration = 0.5 * len(gap_indices)

        current_time = start_time
        for idx in gap_indices:
            word_chars = max(len(words[idx]["word"].strip()), 1)
            word_duration = gap_duration * (word_chars / total_chars)
            words[idx]["start"] = round(current_time, 3)
            words[idx]["end"] = round(current_time + word_duration, 3)
            current_time += word_duration
            interpolated_count += 1

    # Leading gap: words before first anchor
    if anchors[0] > 0:
        leading_indices = [
            i for i in range(0, anchors[0])
            if words[i]["start"] == 0.0 and words[i]["end"] == 0.0
        ]
        if leading_indices:
            _interpolate_gap(leading_indices, 0.0, words[anchors[0]]["start"])

    # Gaps between anchors
    for k in range(len(anchors) - 1):
        prev_anchor = anchors[k]
        next_anchor = anchors[k + 1]

        gap_indices = [
            i for i in range(prev_anchor + 1, next_anchor)
            if words[i]["start"] == 0.0 and words[i]["end"] == 0.0
        ]
        if gap_indices:
            _interpolate_gap(
                gap_indices,
                words[prev_anchor]["end"],
                words[next_anchor]["start"],
            )

    # Trailing gap: words after last anchor
    last_anchor = anchors[-1]
    if last_anchor < len(words) - 1:
        trailing_indices = [
            i for i in range(last_anchor + 1, len(words))
            if words[i]["start"] == 0.0 and words[i]["end"] == 0.0
        ]
        if trailing_indices:
            trailing_start = words[last_anchor]["end"]
            # Estimate duration: ~50ms per character (conservative)
            total_trailing_chars = sum(
                max(len(words[i]["word"].strip()), 1) for i in trailing_indices
            )
            estimated_duration = max(0.5, total_trailing_chars * 0.05)

            # Cap to audio_duration
            if audio_duration > 0:
                max_trailing = max(0.0, audio_duration - trailing_start)
                estimated_duration = min(estimated_duration, max(0.1, max_trailing))

            _interpolate_gap(
                trailing_indices,
                trailing_start,
                trailing_start + estimated_duration,
            )

    if interpolated_count > 0:
        logger.info(
            "Interpolated timestamps for %d words with null aligner output",
            interpolated_count,
        )

    return interpolated_count


# ---------------------------------------------------------------------------
# Hardening: timestamp clamping
# ---------------------------------------------------------------------------


def _clamp_timestamps(
    words: List[Dict[str, Any]],
    max_duration: float,
) -> int:
    """Clamp all word timestamps to ``[0, max_duration]``.

    Prevents aligner drift or interpolation overflow from producing timestamps
    that exceed the audio boundary, which causes out-of-order warnings and
    display artifacts in subtitle renderers.

    Ensures ``word['end'] >= word['start']`` after clamping.

    Parameters
    ----------
    words:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
        Modified in place.
    max_duration:
        Maximum allowed timestamp value (audio duration in seconds).

    Returns
    -------
    Number of words whose timestamps were adjusted.
    """
    if not words or max_duration <= 0:
        return 0

    clamped_count = 0

    for word in words:
        original_start = word["start"]
        original_end = word["end"]

        new_start = max(0.0, min(word["start"], max_duration))
        new_end = max(new_start, min(word["end"], max_duration))

        if new_start != original_start or new_end != original_end:
            clamped_count += 1

        word["start"] = round(new_start, 3)
        word["end"] = round(new_end, 3)

    if clamped_count > 0:
        logger.debug(
            "Clamped %d word timestamps to [0, %.2f]",
            clamped_count, max_duration,
        )

    return clamped_count


# ---------------------------------------------------------------------------
# Hardening: chronological sorting
# ---------------------------------------------------------------------------


def _word_sort_key(word: Dict[str, Any]) -> Tuple[float, float]:
    return (word["start"], word["end"])


def _sort_words_chronologically(
    words: List[Dict[str, Any]],
) -> bool:
    """Sort words by start time, breaking ties by end time.

    Defensive safety net after timestamp interpolation and collapse recovery,
    which can produce out-of-order words.

    Parameters
    ----------
    words:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
        Sorted in place.

    Returns
    -------
    ``True`` if any words were reordered, ``False`` if already sorted.
    """
    if not words or len(words) <= 1:
        return False

    starts_before = [(w["start"], w["end"]) for w in words]
    words.sort(key=_word_sort_key)
    starts_after = [(w["start"], w["end"]) for w in words]

    reordered = starts_before != starts_after
    if reordered:
        logger.debug(
            "Reordered %d words by chronological sort",
            len(words),
        )
    return reordered


# ---------------------------------------------------------------------------
# Hardening pipeline orchestrator
# ---------------------------------------------------------------------------


def _harden_words(
    words: List[Dict[str, Any]],
    audio_duration: float,
    speech_regions: Optional[List[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Apply the full hardening pipeline to a word list.

    Steps (applied in order):
      1. **Null timestamp interpolation** — fill gaps between valid anchors
      2. **Timestamp clamping** — bound to ``[0, audio_duration]``
      3. **Chronological sorting** — ensure strict time ordering

    This function is called after alignment quality assessment and collapse
    recovery (if needed), so the words should already have reasonable
    timestamps.  The hardening pipeline catches edge cases that slip through.

    Parameters
    ----------
    words:
        List of ``{'word': str, 'start': float, 'end': float}`` dicts.
        Modified in place.
    audio_duration:
        Total audio duration in seconds.
    speech_regions:
        Optional VAD speech regions (currently unused by hardening steps but
        reserved for future enhancements).

    Returns
    -------
    The same word list, hardened in place (returned for chaining convenience).
    """
    if not words:
        return words

    # Step 1: Interpolate null/failed timestamps
    interpolated = _interpolate_null_timestamps(words, audio_duration)

    # Step 2: Clamp all timestamps to valid range
    clamped = _clamp_timestamps(words, audio_duration)

    # Step 3: Ensure chronological ordering
    sorted_flag = _sort_words_chronologically(words)

    if interpolated or clamped or sorted_flag:
        logger.debug(
            "Hardening pipeline: interpolated=%d, clamped=%d, reordered=%s",
            interpolated, clamped, sorted_flag,
        )

    return words
