"""Silence-based audio scene detection for chunking long files.

Uses auditok's two-pass strategy (ported from WhisperJAV):
  Pass 1 (Coarse): Find natural chapter boundaries via long silences
  Pass 2 (Fine):   Chunk oversized chapters to a target max duration

Falls back to fixed-size ffmpeg splitting when auditok is not installed.
"""

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

try:
    import auditok
    AUDITOK_AVAILABLE = True
except ImportError:
    AUDITOK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Two-pass auditok parameters (from WhisperJAV AuditokSceneConfig defaults)
# ---------------------------------------------------------------------------

# Pass 1: Coarse chapter detection via long silences
_PASS1_MIN_DURATION: float = 0.3
_PASS1_MAX_DURATION: float = 2700.0  # 45 minutes
_PASS1_MAX_SILENCE: float = 1.8
_PASS1_ENERGY_THRESHOLD: int = 38

# Pass 2: Fine chunking within each chapter
_PASS2_MIN_DURATION: float = 0.3
_PASS2_MAX_SILENCE: float = 0.94
_PASS2_ENERGY_THRESHOLD: int = 50

# Minimum scene duration to keep
_MIN_SCENE_DURATION: float = 0.2


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
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start),
         "-t", str(duration), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", tmp.name],
        capture_output=True, timeout=120,
    )
    return tmp.name


# ---------------------------------------------------------------------------
# Auditok two-pass detection
# ---------------------------------------------------------------------------


def _detect_with_auditok(
    audio_path: str,
    total_duration: float,
    max_scene_duration: float,
) -> list[tuple[float, float]]:
    """Two-pass scene detection using auditok.

    Returns list of (start_sec, end_sec) tuples for each scene.
    """
    # Load audio via auditok
    region = auditok.load(audio_path)
    sample_rate = region.sampling_rate

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
        # No speech detected — return whole file as single scene
        return [(0.0, total_duration)]

    # --- Pass 2: Fine chunking of oversized chapters ---
    scenes: list[tuple[float, float]] = []
    pass2_max_dur = max(max_scene_duration - 1.0, _MIN_SCENE_DURATION)

    for chapter in story_lines:
        chapter_start = chapter.meta.start
        chapter_end = chapter.meta.end
        chapter_duration = chapter_end - chapter_start

        # If chapter fits within max duration, keep as-is
        if chapter_duration <= max_scene_duration:
            if chapter_duration >= _MIN_SCENE_DURATION:
                scenes.append((chapter_start, chapter_end))
            continue

        # Split oversized chapter with finer silence detection
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

        if sub_regions:
            for sub in sub_regions:
                sub_start = chapter_start + sub.meta.start
                sub_end = chapter_start + sub.meta.end
                if (sub_end - sub_start) >= _MIN_SCENE_DURATION:
                    scenes.append((sub_start, sub_end))
        else:
            # Brute-force fallback for chapters where Pass 2 found nothing
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
        "Scene detection Pass 2: %d final scene(s) (%.1fs total coverage)",
        len(scenes), sum(e - s for s, e in scenes),
    )
    return scenes


# ---------------------------------------------------------------------------
# Fixed-size fallback
# ---------------------------------------------------------------------------


def _detect_fixed(
    total_duration: float,
    max_scene_duration: float,
) -> list[tuple[float, float]]:
    """Fixed-size splitting (fallback when auditok is not available)."""
    scenes: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < total_duration:
        end = min(cursor + max_scene_duration, total_duration)
        if (end - cursor) >= _MIN_SCENE_DURATION:
            scenes.append((cursor, end))
        cursor = end
    return scenes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_audio_scenes(
    audio_path: str,
    max_scene_duration: float = 180.0,
) -> list[tuple[str, float]]:
    """Split audio into scenes at silence boundaries.

    Uses auditok two-pass detection when available, falls back to
    fixed-size ffmpeg splitting.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    max_scene_duration:
        Maximum scene duration in seconds.  Scenes longer than this
        are split at silence boundaries (or fixed intervals as fallback).

    Returns
    -------
    List of ``(scene_file_path, start_offset_sec)`` tuples.
    If the audio is shorter than *max_scene_duration*, returns
    ``[(audio_path, 0.0)]`` without creating temp files.
    """
    total_duration = get_audio_duration(audio_path)
    if total_duration <= 0 or total_duration <= max_scene_duration:
        return [(audio_path, 0.0)]

    # Detect scene boundaries
    if AUDITOK_AVAILABLE:
        try:
            boundaries = _detect_with_auditok(audio_path, total_duration, max_scene_duration)
        except Exception:
            logger.warning(
                "Auditok scene detection failed; falling back to fixed splitting",
                exc_info=True,
            )
            boundaries = _detect_fixed(total_duration, max_scene_duration)
    else:
        logger.info(
            "auditok not installed — using fixed-size splitting "
            "(pip install auditok for silence-based scene detection)"
        )
        boundaries = _detect_fixed(total_duration, max_scene_duration)

    # Extract each scene via ffmpeg
    scenes: list[tuple[str, float]] = []
    for start, end in boundaries:
        scene_path = _extract_scene_ffmpeg(audio_path, start, end - start)
        scenes.append((scene_path, start))

    logger.info(
        "Split audio into %d scene(s) (max %.0fs each)",
        len(scenes), max_scene_duration,
    )
    return scenes
