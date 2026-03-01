"""Detect language endpoint."""

import logging
from typing import Union

import numpy as np
from fastapi import APIRouter, File, UploadFile, Query
from language_code import LanguageCode

from subgen.config import (
    force_detected_language_to,
    detect_language_length,
    detect_language_offset,
)
from subgen.models.whisper_model import start_model, delete_model
from subgen.media.audio import extract_audio_segment_from_content, get_audio_chunk

router = APIRouter()


@router.post("/detect-language")
async def detect_language(
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True),
    video_file: Union[str, None] = Query(default=None),
    detect_lang_length: int = Query(default=detect_language_length),
    detect_lang_offset: int = Query(default=detect_language_offset),
):
    if force_detected_language_to:
        await audio_file.close()
        return {"detected_language": force_detected_language_to.to_name(), "language_code": force_detected_language_to.to_iso_639_1()}

    try:
        file_content = await audio_file.read()
        if not file_content:
            return {"detected_language": "Unknown", "language_code": "und", "status": "error"}

        logging.info(f"Immediate language detection (Queue Bypass)" + (f" for {video_file}" if video_file else ""))

        # --- RUN IMMEDIATELY ---
        start_model()

        if encode:
            # BUG FIX: Use file_content from the read above instead of calling
            # await audio_file.read() again (which would return empty bytes since
            # the file cursor is already at the end).
            audio_bytes = extract_audio_segment_from_content(file_content, detect_lang_offset, detect_lang_length)
            audio_data = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        else:
            audio_data = await get_audio_chunk(audio_file, detect_lang_offset, detect_lang_length)

        # Import model at function level to get the current module-level value
        from subgen.models.whisper_model import model as current_model
        result = current_model.transcribe(audio_data, input_sr=16000, verbose=None)
        detected = LanguageCode.from_name(result.language)

        logging.info(f"Detect Language Result: {detected.to_name()} ({detected.to_iso_639_1()})")

        return {
            "detected_language": detected.to_name(),
            "language_code": detected.to_iso_639_1()
        }

    except Exception as e:
        logging.error(f"Error in API detect-language: {e}", exc_info=True)
        return {"detected_language": "Unknown", "language_code": "und", "status": "error"}
    finally:
        await audio_file.close()
        delete_model()  # Schedules VRAM cleanup if system is idle
