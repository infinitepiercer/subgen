"""NVIDIA Sortformer diarization model lifecycle: loading, inference, VRAM release.

Uses ``SortformerEncLabelModel`` from NeMo for end-to-end speaker diarization
(no embedding extraction / clustering required).  Long files are chunked into
overlapping windows to fit within GPU memory; cross-chunk speaker merging is
not attempted in this module — chunk speakers are emitted with a chunk-scoped
prefix so downstream code can treat them as distinct.
"""

import gc
import logging
import math
import os
import subprocess
from typing import List, Tuple

import torch

from subgen.config import (
    model_location,
    sortformer_chunk_sec as _sortformer_chunk_sec,
    sortformer_model as _sortformer_model_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
model = None  # kept for backward compatibility with callers that check ``is None``
_sortformer_model = None

# Overlap between adjacent chunks when splitting long audio (seconds).
_CHUNK_OVERLAP_SEC: float = 10.0

# Type alias returned by ``diarize(...)`` — ``(start_seconds, end_seconds, speaker_label)``.
DiarSegment = Tuple[float, float, str]


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------


def _probe_duration_seconds(audio_path: str) -> float:
    """Return the duration of ``audio_path`` in seconds.

    Tries ``soundfile`` first (fast for WAV/FLAC); falls back to ``ffprobe``
    for containers ``soundfile`` cannot read (e.g. MP4, MKV, MP3).
    """
    try:
        import soundfile as sf  # type: ignore[import-not-found]

        info = sf.info(audio_path)
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception as exc:  # pragma: no cover — fall through to ffprobe
        logger.debug("soundfile.info failed for %s (%s); falling back to ffprobe", audio_path, exc)

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe failed for %s (%s); assuming duration < chunk_sec", audio_path, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Model load / unload
# ---------------------------------------------------------------------------


def _load_sortformer():
    """Lazy-load the Sortformer model, cache it at module level, return it."""
    global _sortformer_model, model
    if _sortformer_model is not None:
        return _sortformer_model

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "NeMo ASR toolkit is required for the Sortformer diarization backend. "
            "Install it with: pip install nemo_toolkit[asr]"
        ) from exc

    # Point NeMo / HuggingFace cache at the shared model location if set.
    if model_location:
        os.environ.setdefault("NEMO_CACHE_DIR", model_location)
        os.environ.setdefault("HF_HOME", model_location)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading Sortformer diarization model '%s' on %s",
        _sortformer_model_id,
        device,
    )
    loaded = SortformerEncLabelModel.from_pretrained(_sortformer_model_id)
    loaded = loaded.to(device)
    loaded.eval()

    _sortformer_model = loaded
    model = loaded  # keep legacy ``model`` alias populated
    logger.info("Sortformer diarization model loaded on %s", device)
    return _sortformer_model


def start_diarization_model(device: str) -> None:
    """Load the Sortformer model if it has not been loaded yet.

    The ``device`` argument is accepted for backward compatibility; Sortformer
    will use CUDA when available regardless.
    """
    _load_sortformer()


def delete_diarization_model() -> None:
    """Unload the Sortformer model, free VRAM, and release references."""
    global _sortformer_model, model
    if _sortformer_model is not None:
        try:
            del _sortformer_model
            _sortformer_model = None
            model = None
            logger.info("Sortformer diarization model unloaded from memory")
        except Exception as exc:
            logger.error("Error unloading Sortformer diarization model: %s", exc)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after Sortformer unload.")
        except Exception as exc:
            logger.error("Error clearing CUDA cache: %s", exc)

    gc.collect()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _normalize_raw_segments(raw) -> List[Tuple[float, float, str]]:
    """Coerce the heterogeneous output of ``SortformerEncLabelModel.diarize(...)``
    into a list of ``(start, end, speaker_label)`` tuples.

    NeMo's ``diarize()`` returns a list-per-audio-file.  Each inner item may be
    a tuple/list ``(begin, end, speaker_idx)`` OR a pre-formatted RTTM-style
    string like ``"0.00 1.23 speaker_0"``.  We handle both.
    """
    # Unwrap single-audio wrapper: list-of-list-of-segments -> list-of-segments.
    if raw and isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
        first = raw[0]
        if first and not isinstance(first[0], (int, float, str)):
            # Looks like outer wrapping (list of per-file segment lists).
            raw = first
        else:
            # first item is itself a segment tuple — raw is already flat.
            pass

    out: List[Tuple[float, float, str]] = []
    for item in raw or []:
        if isinstance(item, str):
            parts = item.strip().split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = str(parts[2])
                    out.append((start, end, label))
                except ValueError:
                    logger.debug("Skipping un-parseable diar segment: %r", item)
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                start = float(item[0])
                end = float(item[1])
                spk = item[2]
                label = spk if isinstance(spk, str) else f"speaker_{int(spk)}"
                out.append((start, end, label))
            except (TypeError, ValueError):
                logger.debug("Skipping malformed diar segment: %r", item)

    return out


def _diarize_single(audio_path: str) -> List[Tuple[float, float, str]]:
    """Run Sortformer on an audio file that fits within the chunk window."""
    m = _load_sortformer()
    with torch.inference_mode():
        raw = m.diarize(audio=[audio_path], batch_size=1)
    return _normalize_raw_segments(raw)


def _extract_chunk_wav(audio_path: str, start_sec: float, duration_sec: float, out_path: str) -> None:
    """Extract a chunk of ``audio_path`` into a temp WAV at 16 kHz mono."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def _calculate_adaptive_chunk_sec(configured_chunk_sec: int) -> int:
    """Derive a safe chunk size based on free CUDA memory.

    If ``SORTFORMER_CHUNK_SEC`` is explicitly set via env var, respect it.
    Otherwise, compute from free VRAM.
    """
    # Respect explicit user override via env var.
    if os.environ.get('SORTFORMER_CHUNK_SEC') is not None:
        return int(configured_chunk_sec)

    if not torch.cuda.is_available():
        return int(configured_chunk_sec)

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return int(configured_chunk_sec)

    free_gb = free_bytes / (1024 ** 3)

    # Empirical rule for Sortformer (conformer encoder, O(n^2) attention):
    # ~1GB per 60s of audio at fp32 with default settings (rough estimate).
    # Leave 20% headroom for activations/safety, minus 500MB baseline.
    usable_gb = max(0.0, free_gb * 0.8 - 0.5)
    derived = int(usable_gb * 60)

    # Clamp to reasonable range: at least 30s, at most the configured ceiling.
    derived = max(30, min(int(configured_chunk_sec), derived))

    logger.info(
        "Adaptive Sortformer chunk: free_vram=%.2fGB, derived_chunk_sec=%ds (configured=%ds)",
        free_gb, derived, int(configured_chunk_sec),
    )
    return derived


def _diarize_chunked_at_size(
    audio_path: str,
    total_duration: float,
    chunk_sec_value: float,
) -> List[Tuple[float, float, str]]:
    """Run Sortformer on overlapping chunks of size ``chunk_sec_value`` and stitch.

    Chunk speakers are emitted with a ``chunk{N}_`` prefix so each chunk's
    speaker IDs remain distinct across the stitched output — cross-chunk
    speaker merging is a follow-up.
    """
    import tempfile

    chunk_sec = float(chunk_sec_value)
    overlap = min(_CHUNK_OVERLAP_SEC, chunk_sec / 4.0)
    step = max(1.0, chunk_sec - overlap)
    n_chunks = int(math.ceil((total_duration - overlap) / step)) if total_duration > overlap else 1

    logger.info(
        "Sortformer chunked inference: duration=%.1fs, chunk=%.1fs, overlap=%.1fs, chunks=%d",
        total_duration, chunk_sec, overlap, n_chunks,
    )

    stitched: List[Tuple[float, float, str]] = []

    for idx in range(n_chunks):
        start_sec = idx * step
        if start_sec >= total_duration:
            break
        duration_sec = min(chunk_sec, total_duration - start_sec)
        if duration_sec <= 0.1:
            break

        tmp = tempfile.NamedTemporaryFile(suffix=f"_sortf_chunk{idx}.wav", delete=False)
        tmp.close()
        try:
            _extract_chunk_wav(audio_path, start_sec, duration_sec, tmp.name)
            chunk_segments = _diarize_single(tmp.name)
            logger.debug(
                "Chunk %d/%d [%.1fs+%.1fs]: %d segments",
                idx + 1, n_chunks, start_sec, duration_sec, len(chunk_segments),
            )
            for c_start, c_end, c_label in chunk_segments:
                # Rebase to absolute timeline and namespace speaker by chunk.
                stitched.append((
                    c_start + start_sec,
                    c_end + start_sec,
                    f"chunk{idx}_{c_label}",
                ))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    stitched.sort(key=lambda s: s[0])
    return stitched


def _diarize_chunked(audio_path: str, total_duration: float) -> List[Tuple[float, float, str]]:
    """Run chunked Sortformer diarization with adaptive sizing + OOM fallback.

    Starts at an adaptive chunk size derived from free VRAM and halves on
    ``torch.cuda.OutOfMemoryError`` down to a minimum floor.
    """
    min_chunk = 30
    chunk_sec = _calculate_adaptive_chunk_sec(int(_sortformer_chunk_sec))

    while chunk_sec >= min_chunk:
        try:
            return _diarize_chunked_at_size(audio_path, total_duration, float(chunk_sec))
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            new_size = chunk_sec // 2
            if new_size < min_chunk:
                logger.error(
                    "Sortformer OOM at chunk_sec=%ds; cannot fall back below min=%ds.",
                    chunk_sec, min_chunk,
                )
                raise
            logger.warning(
                "Sortformer OOM at chunk_sec=%ds, retrying with %ds. "
                "Consider lowering SORTFORMER_CHUNK_SEC env var.",
                chunk_sec, new_size,
            )
            chunk_sec = new_size

    raise RuntimeError(f"Sortformer cannot fit even at {min_chunk}s chunks")


def diarize(audio_path: str) -> List[Tuple[float, float, str]]:
    """Diarize ``audio_path`` and return ``(start, end, speaker_label)`` tuples.

    Clips longer than the (adaptively-derived) chunk size are split into
    overlapping windows and processed per-chunk to stay within GPU memory.
    The ``sortformer_chunk_sec`` config value acts as a ceiling; the actual
    chunk size is auto-derived from free VRAM at call time and may be smaller.
    """
    _load_sortformer()

    duration = _probe_duration_seconds(audio_path)
    adaptive_chunk_sec = _calculate_adaptive_chunk_sec(int(_sortformer_chunk_sec))

    if duration > 0 and duration > float(adaptive_chunk_sec):
        return _diarize_chunked(audio_path, duration)

    return _diarize_single(audio_path)
