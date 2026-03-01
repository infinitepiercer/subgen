"""Batch transcription route."""

from typing import Union

from fastapi import APIRouter, Query, BackgroundTasks

from subgen.watcher.folder_monitor import transcribe_existing

router = APIRouter()


@router.post("/batch")
def batch(
        background_tasks: BackgroundTasks,
        directory: str = Query(...),
        forceLanguage: Union[str, None] = Query(default=None),
):
    from language_code import LanguageCode

    language = LanguageCode.from_string(forceLanguage) if forceLanguage else None
    background_tasks.add_task(transcribe_existing, directory, language)
    return {"status": "accepted", "message": "Batch transcription queued"}
