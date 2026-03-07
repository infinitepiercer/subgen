"""Parakeet (NeMo) model lifecycle management: loading, cleanup, and VRAM release.

Mirrors the structure and lifecycle of whisper_model.py, but loads an NVIDIA
Parakeet-TDT model via the NeMo ASR toolkit instead of faster-whisper.
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
    boost_words as _boost_words,
    compute_type,
    parakeet_model_name as _parakeet_model_name,
    model_location,
    ngram_lm_alpha as _ngram_lm_alpha,
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
# so delete_model() won't unload while a direct task is still running.
active_direct_tasks: int = 0
active_direct_tasks_lock: Lock = Lock()


def start_model() -> None:
    """Load the Parakeet NeMo model into memory if it is not already loaded."""
    global model
    with model_load_lock:
        if model is None:
            logger.debug("Parakeet model was purged, need to re-create")

            try:
                import nemo.collections.asr as nemo_asr
            except ImportError as exc:
                raise ImportError(
                    "NeMo ASR toolkit is required for the Parakeet backend. "
                    "Install it with: pip install nemo_toolkit[asr]"
                ) from exc

            # Point NeMo/HuggingFace cache at the shared model location if set.
            if model_location:
                os.environ.setdefault("NEMO_CACHE_DIR", model_location)
                # HuggingFace Hub also honours this variable for downloads.
                os.environ.setdefault("HF_HOME", model_location)

            logger.info(
                "Loading Parakeet model '%s' on device '%s'",
                _parakeet_model_name,
                transcribe_device,
            )

            model = nemo_asr.models.ASRModel.from_pretrained(_parakeet_model_name)

            # Move to the correct device.
            device = transcribe_device.lower()
            if device == "cuda" and torch.cuda.is_available():
                model = model.to("cuda")
            else:
                model = model.to("cpu")
                logger.warning(
                    "Parakeet is running on CPU — this will be significantly slower "
                    "than GPU. Consider using ASR_ENGINE=whisper for CPU deployments."
                )

            model.eval()

            # fp16 is handled at inference time via torch.cuda.amp.autocast()
            # in _transcribe_parakeet(). We do NOT call model.half() here because
            # NeMo's RNNT decoder feeds float32 tensors internally, causing
            # "mat1 and mat2 must have the same dtype" errors with half weights.
            logger.info("Parakeet compute_type=%s (autocast applied at inference)", compute_type)

            # For longer audio, switch to local attention to avoid OOM.
            # Models like parakeet-tdt_ctc-1.1b already use local attention
            # natively; skip if already configured to avoid overriding.
            try:
                attn_type = getattr(
                    getattr(model.cfg, "encoder", None), "att_context_style", None
                )
                if attn_type and "local" in str(attn_type):
                    logger.debug("Model already uses local attention — skipping switch")
                else:
                    model.change_attention_model(
                        "rel_pos_local_attn", [128, 128]
                    )
                    logger.debug("Switched Parakeet to local attention (window=128)")
                model.change_subsampling_conv_chunking_factor(1)
            except Exception as exc:
                logger.warning(
                    "Could not configure attention model (non-fatal): %s", exc
                )

            # Disable CUDA graphs to prevent illegal memory access when
            # torch.cuda.empty_cache() is called between jobs (NeMo #14727).
            # Also configure NGPU-LM fusion if an n-gram LM is available.
            arpa_path = None
            try:
                from subgen.models.lm_utils import ensure_ngram_lm

                lm_cache = model_location or "/cache"
                arpa_path = ensure_ngram_lm(model.tokenizer, lm_cache)
            except Exception as exc:
                logger.warning("Could not build n-gram LM: %s", exc)

            try:
                import copy
                from omegaconf import open_dict

                decoding_cfg = copy.deepcopy(model.cfg.decoding)
                with open_dict(decoding_cfg):
                    # NeMo TDT beam search (malsd_batch) does not support
                    # preserve_alignments / timestamps, which we require for
                    # word-level subtitle timing.  Use greedy + NGPU-LM
                    # instead — it supports timestamps and the n-gram LM
                    # still improves accuracy at each decoding step.
                    decoding_cfg.strategy = "greedy_batch"
                    decoding_cfg.greedy = decoding_cfg.get("greedy", {})
                    decoding_cfg.greedy.use_cuda_graph_decoder = False
                    decoding_cfg.greedy.allow_cuda_graphs = False
                    decoding_cfg.greedy.max_symbols_per_step = 100
                    decoding_cfg.greedy.max_symbols = 100
                    if arpa_path:
                        decoding_cfg.greedy.ngram_lm_model = arpa_path
                        decoding_cfg.greedy.ngram_lm_alpha = _ngram_lm_alpha

                    # GPU-PB word boosting: bias the decoder toward specific
                    # words/phrases (e.g. character names) without retraining.
                    if _boost_words:
                        phrases = [p.strip() for p in _boost_words.split(',') if p.strip()]
                        if phrases:
                            decoding_cfg.greedy.boosting_tree = decoding_cfg.greedy.get("boosting_tree", {})
                            decoding_cfg.greedy.boosting_tree.key_phrases_list = phrases
                            decoding_cfg.greedy.boosting_tree.context_score = 1.0
                            decoding_cfg.greedy.boosting_tree.depth_scaling = 2.0
                            decoding_cfg.greedy.boosting_tree_alpha = 0.5
                            logger.info(
                                "Word boosting enabled for %d phrases", len(phrases),
                            )

                model.change_decoding_strategy(decoding_cfg)
                if arpa_path:
                    logger.info(
                        "NGPU-LM enabled: greedy decoding with n-gram LM "
                        "(alpha=%.2f), CUDA graphs disabled",
                        _ngram_lm_alpha,
                    )
                else:
                    logger.info("Greedy decoding configured with CUDA graphs disabled")
            except Exception as exc:
                logger.warning(
                    "Could not configure decoding strategy (non-fatal): %s", exc
                )

            logger.info("Parakeet model loaded successfully")


def schedule_model_cleanup() -> None:
    """Schedule model cleanup with a delay to allow concurrent requests."""
    global model_cleanup_timer

    with model_cleanup_lock:
        # Cancel any existing timer
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()
            logger.debug("Cancelled previous Parakeet model cleanup timer")
            model_cleanup_timer.join()

        # Schedule a new cleanup timer
        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()
        logger.debug(
            "Parakeet model cleanup scheduled in %s seconds", model_cleanup_delay
        )


def perform_model_cleanup() -> None:
    """Actually perform the model cleanup."""
    global model, model_cleanup_timer

    with model_cleanup_lock:
        logger.debug("Executing scheduled Parakeet model cleanup")

        with active_direct_tasks_lock:
            system_is_idle = task_queue.is_idle() and active_direct_tasks == 0

        if clear_vram_on_complete and system_is_idle:
            logger.debug(
                "Queue and direct tasks idle; clearing Parakeet model from memory."
            )
            if model is not None:
                try:
                    del model
                    model = None
                    logger.info("Parakeet model unloaded from memory")
                except Exception as exc:
                    logger.error("Error unloading Parakeet model: %s", exc)

            if transcribe_device.lower() == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared.")
                except Exception as exc:
                    logger.error("Error clearing CUDA cache: %s", exc)
        else:
            logger.debug(
                "Queue not idle or clear_vram disabled; skipping Parakeet model cleanup"
            )

        if os.name != "nt":  # don't garbage collect on Windows
            gc.collect()
            libc_name = ctypes.util.find_library("c")
            if libc_name is not None:
                try:
                    ctypes.CDLL(libc_name).malloc_trim(0)
                except (OSError, AttributeError):
                    pass

        model_cleanup_timer = None


def delete_model() -> None:
    """Only schedules a cleanup timer if the system is actually idle.

    This prevents unnecessary timer resets when a large batch is being processed.
    """
    # 1. If we aren't supposed to clear VRAM, don't bother with timers at all.
    if not clear_vram_on_complete:
        return

    # 2. Only schedule cleanup if the queue is empty AND no direct tasks are running.
    with active_direct_tasks_lock:
        system_is_idle = task_queue.is_idle() and active_direct_tasks == 0

    if system_is_idle:
        schedule_model_cleanup()
    else:
        # If there are items left in the queue, we simply do nothing.
        # The very last worker to finish the last item will trigger the timer.
        logger.debug(
            "Tasks still in queue or processing; "
            "skipping Parakeet model cleanup scheduling."
        )
