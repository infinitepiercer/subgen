import logging
import xml.etree.ElementTree as ET

import requests

from subgen.config import plexserver, plextoken


def get_plex_file_name(rating_key: str, server_ip: str = None, plex_token: str = None) -> str:
    """Gets the full path to a file from the Plex server.
    Args:
        rating_key: The ID of the item in the Plex library.
        server_ip: The IP address of the Plex server. Falls back to config if None.
        plex_token: The Plex token. Falls back to config if None.
    Returns:
        The full path to the file.
    """
    if server_ip is None:
        server_ip = plexserver
    if plex_token is None:
        plex_token = plextoken

    url = f"{server_ip}/library/metadata/{rating_key}"

    headers = {
        "X-Plex-Token": plex_token,
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        root = ET.fromstring(response.content)
        fullpath = root.find(".//Part").attrib['file']
        return fullpath
    else:
        raise Exception(f"Error: {response.status_code}")


def refresh_plex_metadata(rating_key: str, server_ip: str = None, plex_token: str = None) -> None:
    """
    Refreshes the metadata of a Plex library item.

    Args:
        rating_key: The ID of the item in the Plex library whose metadata needs to be refreshed.
        server_ip: The IP address of the Plex server. Falls back to config if None.
        plex_token: The Plex token used for authentication. Falls back to config if None.

    Raises:
        Exception: If the server does not respond with a successful status code.
    """
    if server_ip is None:
        server_ip = plexserver
    if plex_token is None:
        plex_token = plextoken

    # Plex API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/library/metadata/{rating_key}/refresh"

    # Headers to include the Plex token for authentication
    headers = {
        "X-Plex-Token": plex_token,
    }

    # Sending the PUT request to refresh metadata
    response = requests.put(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        logging.info("Metadata refresh initiated successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")


def get_next_plex_episode(current_episode_rating_key: str, stay_in_season: bool = False):
    """
    Get the next episode's ratingKey based on the current episode in Plex.
    Args:
        current_episode_rating_key (str): The ratingKey of the current episode.
        stay_in_season (bool): If True, only find the next episode within the current season.
                              If False, find the next episode in the series.
    Returns:
        str: The ratingKey of the next episode, or None if it's the last episode.
    """
    try:
        # Get current episode's metadata to fetch parent (season) ratingKey
        url = f"{plexserver}/library/metadata/{current_episode_rating_key}"
        headers = {"X-Plex-Token": plextoken}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.content)

        # Find the show ID
        grandparent_rating_key = root.find(".//Video").get("grandparentRatingKey")
        if grandparent_rating_key is None:
            logging.debug(f"Show not found for episode {current_episode_rating_key}")
            return None

        # Find the parent season ratingKey
        parent_rating_key = root.find(".//Video").get("parentRatingKey")
        if parent_rating_key is None:
            logging.debug(f"Parent season not found for episode {current_episode_rating_key}")
            return None

        # Get the list of seasons
        url = f"{plexserver}/library/metadata/{grandparent_rating_key}/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        seasons = ET.fromstring(response.content).findall(".//Directory[@type='season']")

        # Get the list of episodes in the parent season
        url = f"{plexserver}/library/metadata/{parent_rating_key}/children"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse XML response for the list of episodes
        episodes = ET.fromstring(response.content).findall(".//Video")
        episodes_in_season = len(episodes)

        # Find the current episode index and get the next one
        current_episode_number = None
        current_season_number = None
        next_season_number = None
        for episode in episodes:
            if episode.get("ratingKey") == current_episode_rating_key:
                current_episode_number = int(episode.get("index"))
                current_season_number = episode.get("parentIndex")
                break

        # Logic to find the next episode
        if stay_in_season:
          if current_episode_number == episodes_in_season:
              return None # End of season
          for episode in episodes:
            if int(episode.get("index")) == int(current_episode_number)+1:
                return episode.get("ratingKey")
        else: # Not staying in season, find the next overall episode
          # Find next season if it exists
          for season in seasons:
              if int(season.get("index")) == int(current_season_number)+1:
                  next_season_number = season.get("ratingKey")
                  break

          if current_episode_number == episodes_in_season:
              if next_season_number is not None:
                logging.debug("At end of season, try to find next season and first episode.")
                url = f"{plexserver}/library/metadata/{next_season_number}/children"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                episodes = ET.fromstring(response.content).findall(".//Video")
                current_episode_number = 0
              else:
                return None
          for episode in episodes:
            if int(episode.get("index")) == int(current_episode_number)+1:
                return episode.get("ratingKey")

        logging.debug(f"No next episode found for {get_plex_file_name(current_episode_rating_key, plexserver, plextoken)}, possibly end of season or series")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Plex: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
