"""Qwen3-ASR model lifecycle management: loading, cleanup, and VRAM release.

Mirrors the structure of parakeet_model.py, but loads the Qwen3-ASR model
via the qwen_asr package instead of NeMo.
"""

import ctypes
import ctypes.util
import gc
import logging
import os
from threading import Lock, Timer
from typing import Optional

import torch

from subgen.config import (
    compute_type,
    model_location,
    qwen_aligner_model as _qwen_aligner_model,
    qwen_max_new_tokens as _qwen_max_new_tokens,
    qwen_model_name as _qwen_model_name,
    qwen_repetition_penalty as _qwen_repetition_penalty,
    qwen_max_tokens_per_second as _qwen_max_tokens_per_second,
    qwen_min_tokens_floor as _qwen_min_tokens_floor,
    transcribe_device,
    clear_vram_on_complete,
    model_cleanup_delay,
)
from subgen.queue.deduplicated_queue import task_queue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
model = None
model_cleanup_timer: Optional[Timer] = None
model_cleanup_lock: Lock = Lock()
model_load_lock: Lock = Lock()

# Track direct model usage outside the queue (e.g. /detect-language endpoint)
active_direct_tasks: int = 0
active_direct_tasks_lock: Lock = Lock()


def start_model() -> None:
    """Load the Qwen3-ASR model into memory if it is not already loaded."""
    global model
    with model_load_lock:
        if model is None:
            logger.debug("Qwen ASR model was purged, need to re-create")

            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise ImportError(
                    "qwen-asr package is required for the Qwen backend. "
                    "Install it with: pip install qwen-asr"
                ) from exc

            # Point HuggingFace cache at the shared model location if set.
            if model_location:
                os.environ.setdefault("HF_HOME", model_location)

            device = transcribe_device.lower()
            if device == "cuda" and torch.cuda.is_available():
                device_map = "cuda:0"
            else:
                device_map = "cpu"
                logger.warning(
                    "Qwen ASR is running on CPU — this will be significantly slower "
                    "than GPU."
                )

            use_bf16 = device_map != "cpu" and compute_type in ("auto", "float16", "int8_float16")
            dtype = torch.bfloat16 if use_bf16 else torch.float32

            # Suppress noisy "Setting pad_token_id to eos_token_id" warnings
            # that fire on every chunk during generation.
            import transformers
            transformers.utils.logging.set_verbosity_error()

            logger.info(
                "Loading Qwen ASR model '%s' on device '%s' (dtype=%s)",
                _qwen_model_name, device_map, dtype,
            )

            aligner_kwargs = None
            if _qwen_aligner_model:
                aligner_kwargs = dict(
                    dtype=dtype,
                    device_map=device_map,
                )

            model = Qwen3ASRModel.from_pretrained(
                _qwen_model_name,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=1,
                max_new_tokens=_qwen_max_new_tokens,
                forced_aligner=_qwen_aligner_model if _qwen_aligner_model else None,
                forced_aligner_kwargs=aligner_kwargs,
            )

            _apply_repetition_penalty()
            logger.info("Qwen ASR model loaded successfully")


def _apply_repetition_penalty() -> None:
    """Apply repetition_penalty to the thinker's HF GenerationConfig.

    Prevents degenerate token spam from autoregressive loops.
    Access chain: model.model.thinker.generation_config (verified for qwen-asr).
    """
    if _qwen_repetition_penalty == 1.0:
        return

    try:
        gen_config = model.model.thinker.generation_config
        gen_config.repetition_penalty = _qwen_repetition_penalty
        logger.info(
            "Generation safety: repetition_penalty=%.2f applied to thinker",
            _qwen_repetition_penalty,
        )
    except AttributeError as exc:
        logger.warning(
            "Could not apply repetition_penalty — qwen-asr model structure "
            "may have changed: %s. Generation will proceed without penalty.",
            exc,
        )


def compute_dynamic_token_limit(audio_duration_sec: float) -> int:
    """Compute a dynamic max_new_tokens limit scaled to audio duration.

    When QWEN_MAX_TOKENS_PER_SECOND > 0, the token budget is proportional
    to the audio length instead of a fixed value. This caps damage from
    degenerate autoregressive loops: a 10-second clip gets ~200 tokens
    instead of burning through 4096.

    The result is clamped to [min_tokens_floor, max_new_tokens].
    """
    if _qwen_max_tokens_per_second <= 0 or audio_duration_sec <= 0:
        return _qwen_max_new_tokens
    dynamic = max(
        _qwen_min_tokens_floor,
        int(audio_duration_sec * _qwen_max_tokens_per_second),
    )
    return min(dynamic, _qwen_max_new_tokens)


def schedule_model_cleanup() -> None:
    """Schedule model cleanup with a delay to allow concurrent requests."""
    global model_cleanup_timer

    with model_cleanup_lock:
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()
            logger.debug("Cancelled previous Qwen model cleanup timer")
            model_cleanup_timer.join()

        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()
        logger.debug(
            "Qwen model cleanup scheduled in %s seconds", model_cleanup_delay
        )


def perform_model_cleanup() -> None:
    """Actually perform the model cleanup."""
    global model, model_cleanup_timer

    with model_cleanup_lock:
        logger.debug("Executing scheduled Qwen model cleanup")

        with active_direct_tasks_lock:
            system_is_idle = task_queue.is_idle() and active_direct_tasks == 0

        if clear_vram_on_complete and system_is_idle:
            logger.debug(
                "Queue and direct tasks idle; clearing Qwen model from memory."
            )
            if model is not None:
                try:
                    del model
                    model = None
                    logger.info("Qwen ASR model unloaded from memory")
                except Exception as exc:
                    logger.error("Error unloading Qwen model: %s", exc)

            if transcribe_device.lower() == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared.")
                except Exception as exc:
                    logger.error("Error clearing CUDA cache: %s", exc)
        else:
            logger.debug(
                "Queue not idle or clear_vram disabled; skipping Qwen model cleanup"
            )

        if os.name != "nt":
            gc.collect()
            libc_name = ctypes.util.find_library("c")
            if libc_name is not None:
                try:
                    ctypes.CDLL(libc_name).malloc_trim(0)
                except (OSError, AttributeError):
                    pass

        model_cleanup_timer = None


def delete_model() -> None:
    """Only schedules a cleanup timer if the system is actually idle."""
    if not clear_vram_on_complete:
        return

    with active_direct_tasks_lock:
        system_is_idle = task_queue.is_idle() and active_direct_tasks == 0

    if system_is_idle:
        schedule_model_cleanup()
    else:
        logger.debug(
            "Tasks still in queue or processing; "
            "skipping Qwen model cleanup scheduling."
        )
