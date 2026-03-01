"""Jellyfin webhook route."""

import logging

from fastapi import APIRouter, Header, Body

from subgen.config import (
    procaddedmedia,
    procmediaonplay,
    transcribe_or_translate,
    jellyfinserver,
    jellyfintoken,
)
from subgen.integrations.jellyfin import get_jellyfin_file_name
from subgen.services.transcription import gen_subtitles_queue
from subgen.media.path_mapping import path_mapping

router = APIRouter()


@router.post("/jellyfin")
def receive_jellyfin_webhook(
        user_agent: str = Header(None),
        NotificationType: str = Body(None),
        file: str = Body(None),
        ItemId: str = Body(None),
):
    if "Jellyfin-Server" in user_agent:
        logging.debug(f"Jellyfin event detected is: {NotificationType}")
        logging.debug(f"itemid is: {ItemId}")

        if (NotificationType == "ItemAdded" and procaddedmedia) or (NotificationType == "PlaybackStart" and procmediaonplay):
            fullpath = get_jellyfin_file_name(ItemId, jellyfinserver, jellyfintoken)
            logging.debug(f"Full file path: {fullpath}")

            # Queue item with Jellyfin metadata ID for delayed refresh
            gen_subtitles_queue(
                path_mapping(fullpath),
                transcribe_or_translate,
                jellyfin_item_id=ItemId,
                jellyfin_server=jellyfinserver,
                jellyfin_token=jellyfintoken,
            )

            # Note: refresh_jellyfin_metadata removed here; handled by worker.
    else:
        return {
            "message": "This doesn't appear to be a properly configured Jellyfin webhook, please review the instructions again!"}

    return ""
