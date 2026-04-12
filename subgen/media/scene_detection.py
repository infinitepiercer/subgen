"""Silence-based audio scene detection for chunking long files.

Three-pass strategy (ported from WhisperJAV):
  Pass 1 (Coarse): Auditok finds natural chapter boundaries via long silences
  Pass 2 (Fine):   Silero VAD splits oversized chapters at speech boundaries
  Pass 3 (Fallback): Auditok fine-grained or fixed-size splitting if Silero unavailable

Falls back gracefully at each level:
  - Silero VAD not installed -> auditok Pass 2 (original behavior)
  - auditok not installed    -> fixed-size ffmpeg splitting

Speech regions detected by Silero VAD are stored in thread-local storage
via ``set_speech_regions()`` / ``get_speech_regions()`` so downstream code
(e.g. result_adapter.py) can use them for timestamp recovery.
"""

import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import auditok
    AUDITOK_AVAILABLE = True
except ImportError:
    AUDITOK_AVAILABLE = False

# Silero VAD availability (lazy-loaded on first use)
_SILERO_AVAILABLE: Optional[bool] = None
_silero_model: object = None
_silero_get_speech_timestamps: object = None

# ---------------------------------------------------------------------------
# Two-pass auditok parameters (from WhisperJAV AuditokSceneConfig defaults)
# ---------------------------------------------------------------------------

# Pass 1: Coarse chapter detection via long silences
_PASS1_MIN_DURATION: float = 0.3
_PASS1_MAX_DURATION: float = 2700.0  # 45 minutes
_PASS1_MAX_SILENCE: float = 1.8
_PASS1_ENERGY_THRESHOLD: int = 38

# Pass 2 (auditok fallback): Fine chunking within each chapter
_PASS2_MIN_DURATION: float = 0.3
_PASS2_MAX_SILENCE: float = 0.94
_PASS2_ENERGY_THRESHOLD: int = 50

# Minimum scene duration to keep
_MIN_SCENE_DURATION: float = 0.2


# ---------------------------------------------------------------------------
# Speech region data class
# ---------------------------------------------------------------------------


@dataclass
class SpeechRegion:
    """A region of detected speech within the audio."""
    start: float  # seconds (absolute, relative to full audio)
    end: float    # seconds (absolute, relative to full audio)

    @property
    def duration(self) -> float:
        return self.end - self.start


# Thread-local storage for speech regions from the most recent detection.
# Using threading.local() so concurrent transcriptions don't race on a
# shared mutable global.
_thread_local = threading.local()


def get_speech_regions() -> List[SpeechRegion]:
    """Return speech regions detected by the most recent Silero VAD run on this thread."""
    return getattr(_thread_local, 'speech_regions', [])


def set_speech_regions(regions: List[SpeechRegion]) -> None:
    """Store speech regions from the most recent Silero VAD run on this thread."""
    _thread_local.speech_regions = regions


# ---------------------------------------------------------------------------
# Silero VAD lazy loading
# ---------------------------------------------------------------------------


def _ensure_silero_loaded() -> bool:
    """Lazy-load Silero VAD model. Returns True if available."""
    global _SILERO_AVAILABLE, _silero_model, _silero_get_speech_timestamps

    if _SILERO_AVAILABLE is not None:
        return _SILERO_AVAILABLE

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        _silero_model = load_silero_vad()
        _silero_get_speech_timestamps = get_speech_timestamps
        _SILERO_AVAILABLE = True
        logger.info("Silero VAD loaded successfully for scene detection")
    except ImportError:
        _SILERO_AVAILABLE = False
        logger.info(
            "silero-vad not installed -- Silero VAD pass disabled "
            "(pip install silero-vad for speech-aligned scene detection)"
        )
    except Exception:
        _SILERO_AVAILABLE = False
        logger.warning(
            "Failed to load Silero VAD model; falling back to auditok-only detection",
            exc_info=True,
        )

    return _SILERO_AVAILABLE


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_scene_ffmpeg(audio_path: str, start: float, duration: float) -> str:
    """Extract a scene from the audio file using ffmpeg. Returns temp file path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
         "-t", str(duration), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", tmp.name],
        capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        stderr_msg = result.stderr.decode(errors='replace') if result.stderr else ''
        logger.error(
            "ffmpeg scene extraction failed (rc=%d) for %s at %.2fs+%.2fs: %s",
            result.returncode, audio_path, start, duration, stderr_msg,
        )
    return tmp.name


# ---------------------------------------------------------------------------
# Silero VAD Pass 2: speech-boundary splitting
# ---------------------------------------------------------------------------


def _split_chapter_with_silero(
    audio_path: str,
    chapter_start: float,
    chapter_end: float,
    max_scene_duration: float,
    speech_threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    collected_regions: List[SpeechRegion],
) -> List[Tuple[float, float]]:
    """Use Silero VAD to split an oversized chapter at speech boundaries.

    Extracts the chapter audio, runs Silero VAD to find speech segments,
    then groups them into scenes that respect max_scene_duration.

    Parameters
    ----------
    audio_path:
        Path to the original audio file.
    chapter_start:
        Start time of the chapter in seconds.
    chapter_end:
        End time of the chapter in seconds.
    max_scene_duration:
        Maximum allowed scene duration in seconds.
    speech_threshold:
        Silero VAD speech probability threshold.
    min_silence_ms:
        Minimum silence duration in ms to consider as a split point.
    min_speech_ms:
        Minimum speech duration in ms to keep.
    collected_regions:
        List to append detected SpeechRegion objects to (side effect).

    Returns
    -------
    List of (start_sec, end_sec) tuples for scenes within this chapter.
    Empty list if Silero VAD fails (triggers fallback in caller).
    """
    import struct
    import wave

    import torch

    chapter_duration = chapter_end - chapter_start

    # Extract chapter audio to a temp WAV at 16kHz mono PCM s16le for Silero.
    # _extract_scene_ffmpeg already produces 16kHz mono s16le, so we can read
    # the raw PCM directly with the standard library wave module (no torchaudio needed).
    tmp_path: Optional[str] = None
    try:
        tmp_path = _extract_scene_ffmpeg(audio_path, chapter_start, chapter_duration)

        with wave.open(tmp_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        # Convert s16le PCM to float32 tensor in [-1, 1] range
        samples = struct.unpack('<%dh' % (len(raw_bytes) // 2), raw_bytes)
        audio_tensor = torch.FloatTensor(samples) / 32768.0

        # Run Silero VAD
        speech_timestamps = _silero_get_speech_timestamps(
            audio_tensor,
            _silero_model,
            sampling_rate=16000,
            threshold=speech_threshold,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            speech_pad_ms=200,
            return_seconds=True,
        )

    except Exception:
        logger.warning(
            "Silero VAD failed for chapter at %.1fs-%.1fs; will use auditok fallback",
            chapter_start, chapter_end,
            exc_info=True,
        )
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not speech_timestamps:
        logger.debug(
            "Silero VAD found no speech in chapter %.1fs-%.1fs",
            chapter_start, chapter_end,
        )
        return []

    # Collect speech regions with absolute timestamps
    for seg in speech_timestamps:
        collected_regions.append(SpeechRegion(
            start=round(chapter_start + seg["start"], 3),
            end=round(chapter_start + seg["end"], 3),
        ))

    # Group speech segments into scenes respecting max_scene_duration.
    # Each scene spans from the start of its first speech segment to the
    # end of its last speech segment.
    scenes: List[Tuple[float, float]] = []
    group_start: Optional[float] = None
    group_end: float = 0.0

    for seg in speech_timestamps:
        seg_start_abs = chapter_start + seg["start"]
        seg_end_abs = chapter_start + seg["end"]

        if group_start is None:
            group_start = seg_start_abs
            group_end = seg_end_abs
            continue

        # Would adding this segment exceed max duration?
        potential_duration = seg_end_abs - group_start
        if potential_duration > max_scene_duration:
            # Flush current group as a scene
            if (group_end - group_start) >= _MIN_SCENE_DURATION:
                scenes.append((group_start, group_end))
            group_start = seg_start_abs
            group_end = seg_end_abs
        else:
            group_end = seg_end_abs

    # Flush final group
    if group_start is not None and (group_end - group_start) >= _MIN_SCENE_DURATION:
        scenes.append((group_start, group_end))

    # Ensure full chapter coverage by extending scene boundaries.
    # Speech segments define SPLIT POINTS, but every second of the chapter
    # must belong to exactly one scene — otherwise audio between speech
    # segments at scene boundaries is silently dropped.
    if scenes:
        # First scene starts at chapter beginning
        scenes[0] = (chapter_start, scenes[0][1])
        # Last scene ends at chapter end
        scenes[-1] = (scenes[-1][0], chapter_end)
        # Fill gaps: each scene extends to the midpoint between it and the next
        for i in range(len(scenes) - 1):
            mid_point = (scenes[i][1] + scenes[i + 1][0]) / 2
            scenes[i] = (scenes[i][0], mid_point)
            scenes[i + 1] = (mid_point, scenes[i + 1][1])

    logger.debug(
        "Silero VAD split %.1fs chapter into %d scene(s) from %d speech segment(s)",
        chapter_duration, len(scenes), len(speech_timestamps),
    )
    return scenes


# ---------------------------------------------------------------------------
# Auditok two-pass detection (original behavior, used as fallback for Pass 2)
# ---------------------------------------------------------------------------


def _split_chapter_with_auditok(
    chapter: object,
    chapter_start: float,
    chapter_end: float,
    max_scene_duration: float,
) -> List[Tuple[float, float]]:
    """Split an oversized chapter using auditok fine-grained silence detection.

    Parameters
    ----------
    chapter:
        An auditok AudioRegion for the chapter.
    chapter_start:
        Absolute start time of the chapter in seconds.
    chapter_end:
        Absolute end time of the chapter in seconds.
    max_scene_duration:
        Maximum scene duration in seconds.

    Returns
    -------
    List of (start_sec, end_sec) tuples. Empty list if no sub-regions found.
    """
    chapter_duration = chapter_end - chapter_start
    pass2_max_dur = max(max_scene_duration - 1.0, _MIN_SCENE_DURATION)

    pass2_params = {
        "min_dur": _PASS2_MIN_DURATION,
        "max_dur": pass2_max_dur,
        "max_silence": min(chapter_duration * 0.95, _PASS2_MAX_SILENCE),
        "energy_threshold": _PASS2_ENERGY_THRESHOLD,
        "drop_trailing_silence": True,
    }

    try:
        sub_regions = list(chapter.split(**pass2_params))
    except Exception:
        sub_regions = []

    scenes: List[Tuple[float, float]] = []
    for sub in sub_regions:
        sub_start = chapter_start + sub.start
        sub_end = chapter_start + sub.end
        if (sub_end - sub_start) >= _MIN_SCENE_DURATION:
            scenes.append((sub_start, sub_end))

    return scenes


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------


def _detect_with_auditok(
    audio_path: str,
    total_duration: float,
    max_scene_duration: float,
    use_silero: bool = True,
    silero_threshold: float = 0.05,
    silero_min_silence_ms: int = 800,
    silero_min_speech_ms: int = 80,
) -> Tuple[List[Tuple[float, float]], List[SpeechRegion]]:
    """Multi-pass scene detection using auditok + optional Silero VAD.

    Pass 1: Auditok coarse chapter boundaries (long silences).
    Pass 2: Silero VAD speech-boundary splitting for oversized chapters.
    Pass 3 (fallback): Auditok fine splitting or brute-force fixed splits.

    Returns
    -------
    Tuple of (scene_boundaries, speech_regions) where:
      - scene_boundaries: list of (start_sec, end_sec) tuples
      - speech_regions: list of SpeechRegion from Silero VAD (empty if not used)
    """
    # Load audio via auditok
    region = auditok.load(audio_path)

    # --- Pass 1: Coarse chapter boundaries ---
    pass1_params = {
        "min_dur": _PASS1_MIN_DURATION,
        "max_dur": _PASS1_MAX_DURATION,
        "max_silence": min(total_duration * 0.95, _PASS1_MAX_SILENCE),
        "energy_threshold": _PASS1_ENERGY_THRESHOLD,
        "drop_trailing_silence": True,
    }
    story_lines = list(region.split(**pass1_params))
    logger.info("Scene detection Pass 1: %d coarse chapter(s) found", len(story_lines))

    if not story_lines:
        # No speech detected -- return whole file as single scene
        return [(0.0, total_duration)], []

    # Determine if Silero VAD is available and requested
    silero_ready = use_silero and _ensure_silero_loaded()
    if silero_ready:
        logger.info(
            "Scene detection Pass 2: using Silero VAD (threshold=%.2f, "
            "min_silence=%dms, min_speech=%dms)",
            silero_threshold, silero_min_silence_ms, silero_min_speech_ms,
        )
    else:
        logger.info("Scene detection Pass 2: using auditok fine splitting")

    # --- Pass 2: Fine splitting of oversized chapters ---
    scenes: List[Tuple[float, float]] = []
    speech_regions: List[SpeechRegion] = []

    for chapter in story_lines:
        chapter_start: float = chapter.start
        chapter_end: float = chapter.end
        chapter_duration: float = chapter_end - chapter_start

        # If chapter fits within max duration, keep as-is
        if chapter_duration <= max_scene_duration:
            if chapter_duration >= _MIN_SCENE_DURATION:
                scenes.append((chapter_start, chapter_end))
            continue

        # Try Silero VAD first for oversized chapters
        sub_scenes: List[Tuple[float, float]] = []
        if silero_ready:
            sub_scenes = _split_chapter_with_silero(
                audio_path=audio_path,
                chapter_start=chapter_start,
                chapter_end=chapter_end,
                max_scene_duration=max_scene_duration,
                speech_threshold=silero_threshold,
                min_silence_ms=silero_min_silence_ms,
                min_speech_ms=silero_min_speech_ms,
                collected_regions=speech_regions,
            )

        # Fallback to auditok fine splitting if Silero produced nothing
        if not sub_scenes:
            sub_scenes = _split_chapter_with_auditok(
                chapter=chapter,
                chapter_start=chapter_start,
                chapter_end=chapter_end,
                max_scene_duration=max_scene_duration,
            )

        if sub_scenes:
            scenes.extend(sub_scenes)
        else:
            # Brute-force fallback for chapters where neither method found splits
            logger.warning(
                "Pass 2 found no sub-regions in %.1fs chapter, using fixed splits",
                chapter_duration,
            )
            cursor = chapter_start
            while cursor < chapter_end:
                end = min(cursor + max_scene_duration, chapter_end)
                if (end - cursor) >= _MIN_SCENE_DURATION:
                    scenes.append((cursor, end))
                cursor = end

    logger.info(
        "Scene detection Pass 2: %d final scene(s) (%.1fs total coverage, %d speech regions)",
        len(scenes),
        sum(e - s for s, e in scenes),
        len(speech_regions),
    )
    return scenes, speech_regions


# ---------------------------------------------------------------------------
# Fixed-size fallback
# ---------------------------------------------------------------------------


def _detect_fixed(
    total_duration: float,
    max_scene_duration: float,
    overlap: float = 1.5,
) -> List[Tuple[float, float]]:
    """Fixed-size splitting (fallback when auditok is not available).

    Each scene overlaps with the next by *overlap* seconds so that words
    near a chunk boundary are seen by both the current and the next
    transcription pass.  The transcription stitching layer is responsible
    for deduplicating the overlapping region.

    Parameters
    ----------
    total_duration:
        Total audio duration in seconds.
    max_scene_duration:
        Maximum scene duration (including overlap) in seconds.
    overlap:
        Overlap between consecutive scenes in seconds (default 1.5s).
    """
    scenes: List[Tuple[float, float]] = []
    cursor = 0.0
    while cursor < total_duration:
        end = min(cursor + max_scene_duration, total_duration)
        if (end - cursor) >= _MIN_SCENE_DURATION:
            scenes.append((cursor, end))
        if end >= total_duration:
            break
        # Advance by (max_scene_duration - overlap) so the next scene starts
        # overlap seconds before the end of the current one.
        cursor = end - overlap
    return scenes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_audio_scenes(
    audio_path: str,
    max_scene_duration: float = 30.0,
) -> List[Tuple[str, float]]:
    """Split audio into scenes at silence/speech boundaries.

    Uses auditok for coarse chapter detection, then Silero VAD for
    fine-grained speech-aligned splitting when available.  Falls back
    to auditok-only or fixed-size ffmpeg splitting as needed.

    Speech regions detected by Silero VAD are stored in thread-local
    storage via ``set_speech_regions()`` for use by downstream code.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    max_scene_duration:
        Maximum scene duration in seconds (default 30s for speech-aligned
        scenes).  Scenes longer than this are split at speech/silence
        boundaries (or fixed intervals as fallback).

    Returns
    -------
    List of ``(scene_file_path, start_offset_sec)`` tuples.
    If the audio is shorter than *max_scene_duration*, returns
    ``[(audio_path, 0.0)]`` without creating temp files.
    """
    # Import config values for Silero VAD parameters
    from subgen.config import (
        use_silero_vad,
        silero_vad_threshold,
        silero_min_silence_ms,
        silero_min_speech_ms,
    )

    total_duration = get_audio_duration(audio_path)
    if total_duration <= 0 or total_duration <= max_scene_duration:
        set_speech_regions([])
        return [(audio_path, 0.0)]

    # Detect scene boundaries
    speech_regions: List[SpeechRegion] = []
    if AUDITOK_AVAILABLE:
        try:
            boundaries, speech_regions = _detect_with_auditok(
                audio_path,
                total_duration,
                max_scene_duration,
                use_silero=use_silero_vad,
                silero_threshold=silero_vad_threshold,
                silero_min_silence_ms=silero_min_silence_ms,
                silero_min_speech_ms=silero_min_speech_ms,
            )
        except Exception:
            logger.warning(
                "Scene detection failed; falling back to fixed splitting",
                exc_info=True,
            )
            boundaries = _detect_fixed(total_duration, max_scene_duration)
    else:
        logger.info(
            "auditok not installed -- using fixed-size splitting "
            "(pip install auditok for silence-based scene detection)"
        )
        boundaries = _detect_fixed(total_duration, max_scene_duration)

    # Store speech regions for downstream access (thread-safe)
    set_speech_regions(speech_regions)

    # Extract each scene via ffmpeg
    scenes: List[Tuple[str, float]] = []
    for start, end in boundaries:
        scene_path = _extract_scene_ffmpeg(audio_path, start, end - start)
        scenes.append((scene_path, start))

    logger.info(
        "Split audio into %d scene(s) (max %.0fs each, %d speech regions detected)",
        len(scenes), max_scene_duration, len(speech_regions),
    )
    return scenes
