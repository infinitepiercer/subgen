"""Tautulli webhook route."""

import logging
from typing import Union

from fastapi import APIRouter, Header, Body

from subgen.config import procaddedmedia, procmediaonplay, transcribe_or_translate
from subgen.services.transcription import gen_subtitles_queue
from subgen.media.path_mapping import path_mapping

router = APIRouter()


@router.post("/tautulli")
def receive_tautulli_webhook(
        source: Union[str, None] = Header(None),
        event: str = Body(None),
        file: str = Body(None),
):
    if source == "Tautulli":
        logging.debug(f"Tautulli event detected is: {event}")
        if (event == "added" and procaddedmedia) or (event == "played" and procmediaonplay):
            fullpath = file
            logging.debug(f"Full file path: {fullpath}")

            gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)
    else:
        return {
            "message": "This doesn't appear to be a properly configured Tautulli webhook, please review the instructions again!"}

    return ""
