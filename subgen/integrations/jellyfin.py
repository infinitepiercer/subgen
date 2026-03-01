import json
import logging

import requests

from subgen.config import jellyfinserver, jellyfintoken


def get_jellyfin_file_name(item_id: str, server_ip: str = None, jellyfin_token: str = None) -> str:
    """Gets the full path to a file from the Jellyfin server.
    Args:
        item_id: The ID of the item in the Jellyfin library.
        server_ip: The URL of the Jellyfin server. Falls back to config if None.
        jellyfin_token: The Jellyfin token. Falls back to config if None.
    Returns:
        The full path to the file.
    """
    if server_ip is None:
        server_ip = jellyfinserver
    if jellyfin_token is None:
        jellyfin_token = jellyfintoken

    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }

    # Cheap way to get the admin user id, and save it for later use.
    users = json.loads(requests.get(f"{server_ip}/Users", headers=headers).content)
    jellyfin_admin = get_jellyfin_admin(users)

    response = requests.get(f"{server_ip}/Users/{jellyfin_admin}/Items/{item_id}", headers=headers)

    if response.status_code == 200:
        file_name = json.loads(response.content)['Path']
        return file_name
    else:
        raise Exception(f"Error: {response.status_code}")


def refresh_jellyfin_metadata(item_id: str, server_ip: str = None, jellyfin_token: str = None) -> None:
    """
    Refreshes the metadata of a Jellyfin library item.

    Args:
        item_id: The ID of the item in the Jellyfin library whose metadata needs to be refreshed.
        server_ip: The IP address of the Jellyfin server. Falls back to config if None.
        jellyfin_token: The Jellyfin token used for authentication. Falls back to config if None.

    Raises:
        Exception: If the server does not respond with a successful status code.
    """
    if server_ip is None:
        server_ip = jellyfinserver
    if jellyfin_token is None:
        jellyfin_token = jellyfintoken

    # Jellyfin API endpoint to refresh metadata for a specific item
    url = f"{server_ip}/Items/{item_id}/Refresh?MetadataRefreshMode=FullRefresh"

    # Headers to include the Jellyfin token for authentication
    headers = {
        "Authorization": f"MediaBrowser Token={jellyfin_token}",
    }

    # Cheap way to get the admin user id, and save it for later use.
    users = json.loads(requests.get(f"{server_ip}/Users", headers=headers).content)
    jellyfin_admin = get_jellyfin_admin(users)

    # Sending the POST request to refresh metadata
    response = requests.post(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 204:
        logging.info("Metadata refresh queued successfully.")
    else:
        raise Exception(f"Error refreshing metadata: {response.status_code}")


def get_jellyfin_admin(users):
    for user in users:
        if user["Policy"]["IsAdministrator"]:
            return user["Id"]

    raise Exception("Unable to find administrator user in Jellyfin")
