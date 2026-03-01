"""WeSpeaker diarization model lifecycle management: loading, cleanup, and VRAM release."""

import gc
import logging
import os

import torch

from subgen.config import (
    diarization_model as _diarization_model_name,
    model_location,
)

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
model = None


def start_diarization_model(device: str) -> None:
    """Load the WeSpeaker diarization model into memory if it is not already loaded."""
    global model
    if model is None:
        logging.debug("Diarization model was not loaded, creating now")

        # Point WeSpeaker's cache directory at our shared model location
        os.environ["WESPEAKER_HOME"] = model_location

        import wespeaker

        model = wespeaker.load_model(_diarization_model_name)
        model.set_device(device)
        logging.info(
            "WeSpeaker diarization model '%s' loaded on %s",
            _diarization_model_name,
            device,
        )


def delete_diarization_model() -> None:
    """Unload the diarization model, free VRAM, and release references."""
    global model
    if model is not None:
        try:
            del model
            model = None
            logging.info("Diarization model unloaded from memory")
        except Exception as exc:
            logging.error("Error unloading diarization model: %s", exc)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logging.debug("CUDA cache cleared after diarization model unload.")
        except Exception as exc:
            logging.error("Error clearing CUDA cache: %s", exc)

    gc.collect()
