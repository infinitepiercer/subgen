"""Plex webhook route."""

import json
import logging
from typing import Union

from fastapi import APIRouter, Header, Form

from subgen.config import (
    procaddedmedia,
    procmediaonplay,
    transcribe_or_translate,
    plexserver,
    plextoken,
    plex_queue_next_episode,
    plex_queue_season,
    plex_queue_series,
)
from subgen.integrations.plex import get_plex_file_name, get_next_plex_episode
from subgen.services.transcription import gen_subtitles_queue
from subgen.media.path_mapping import path_mapping

router = APIRouter()


def _queue_plex_series_or_season(initial_rating_key: str) -> None:
    """Queue all episodes in a series or season starting from the given rating key.

    Walks forward through episodes using get_next_plex_episode, staying within
    the season if plex_queue_season is set, or spanning the whole series if
    plex_queue_series is set.
    """
    current_rating_key = initial_rating_key
    stay_in_season = plex_queue_season  # Determine if we're staying in the season or not

    while current_rating_key is not None:
        try:
            # Queue the current episode
            file_path = path_mapping(get_plex_file_name(current_rating_key, plexserver, plextoken))

            gen_subtitles_queue(
                file_path,
                transcribe_or_translate,
                plex_item_id=current_rating_key,
                plex_server=plexserver,
                plex_token=plextoken,
            )

            logging.debug(f"Queued episode with ratingKey {current_rating_key}")

            # Get the next episode
            next_episode_rating_key = get_next_plex_episode(current_rating_key, stay_in_season=stay_in_season)
            if next_episode_rating_key is None:
                break  # Exit the loop if no next episode
            current_rating_key = next_episode_rating_key

        except Exception as e:
            logging.error(f"Error processing episode with ratingKey {current_rating_key} or reached end of series: {e}")
            break  # Stop processing on error

    logging.info("All episodes in the series (or season) have been queued.")


@router.post("/plex")
def receive_plex_webhook(
        user_agent: Union[str] = Header(None),
        payload: Union[str] = Form(),
):
    try:
        plex_json = json.loads(payload)

        if "PlexMediaServer" not in user_agent:
            return {"message": "This doesn't appear to be a properly configured Plex webhook, please review the instructions again"}

        event = plex_json["event"]
        logging.debug(f"Plex event detected is: {event}")

        if (event == "library.new" and procaddedmedia) or (event == "media.play" and procmediaonplay):
            rating_key = plex_json['Metadata']['ratingKey']
            fullpath = get_plex_file_name(rating_key, plexserver, plextoken)
            logging.debug(f"Full file path: {fullpath}")

            # Queue the current item with its specific ID for refreshing
            gen_subtitles_queue(
                path_mapping(fullpath),
                transcribe_or_translate,
                plex_item_id=rating_key,
                plex_server=plexserver,
                plex_token=plextoken,
            )

            if plex_queue_next_episode:
                next_key = get_next_plex_episode(plex_json['Metadata']['ratingKey'], stay_in_season=False)
                if next_key:
                    next_file = get_plex_file_name(next_key, plexserver, plextoken)
                    gen_subtitles_queue(
                        path_mapping(next_file),
                        transcribe_or_translate,
                        plex_item_id=next_key,
                        plex_server=plexserver,
                        plex_token=plextoken,
                    )

            if plex_queue_series or plex_queue_season:
                _queue_plex_series_or_season(plex_json['Metadata']['ratingKey'])

    except Exception as e:
        logging.error(f"Failed to process Plex webhook: {e}")

    return ""
