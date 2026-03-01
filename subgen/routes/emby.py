"""Emby webhook route."""

import json
import logging
from typing import Union

from fastapi import APIRouter, Header, Form

from subgen.config import procaddedmedia, procmediaonplay, transcribe_or_translate
from subgen.services.transcription import gen_subtitles_queue
from subgen.media.path_mapping import path_mapping

router = APIRouter()


@router.post("/emby")
def receive_emby_webhook(
        user_agent: Union[str, None] = Header(None),
        data: Union[str, None] = Form(None),
):
    if not data:
        return ""

    data_dict = json.loads(data)
    event = data_dict['Event']
    logging.debug("Emby event detected is: " + event)

    # Check if it's a notification test event
    if event == "system.notificationtest":
        logging.info("Emby test message received!")
        return {"message": "Notification test received successfully!"}

    if (event == "library.new" and procaddedmedia) or (event == "playback.start" and procmediaonplay):
        fullpath = data_dict['Item']['Path']
        logging.debug(f"Full file path: {fullpath}")
        gen_subtitles_queue(path_mapping(fullpath), transcribe_or_translate)

    return ""
