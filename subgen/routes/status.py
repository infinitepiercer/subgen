"""Status and informational routes."""

from fastapi import APIRouter, Request

import stable_whisper
import faster_whisper

from subgen import subgen_version
from subgen.config import docker_status

router = APIRouter()


@router.get("/plex")
@router.get("/webhook")
@router.get("/jellyfin")
@router.get("/asr")
@router.get("/emby")
@router.get("/detect-language")
@router.get("/tautulli")
def handle_get_request(request: Request):
    return {"You accessed this request incorrectly via a GET request. See https://github.com/McCloudS/subgen for proper configuration"}


@router.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings. "}


@router.get("/status")
def status():
    return {"version": f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}
