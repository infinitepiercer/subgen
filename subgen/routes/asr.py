"""ASR (Automatic Speech Recognition) endpoint with hash-based deduplication and blocking."""

import logging
from typing import Union

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import StreamingResponse

from subgen.config import force_detected_language_to, asr_timeout
from subgen.queue.task_result import TaskResult, task_results, task_results_lock
from subgen.queue.deduplicated_queue import task_queue
from subgen.utils.hashing import generate_audio_hash

router = APIRouter()


@router.post("/asr")
async def asr(
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None),
    video_file: Union[str, None] = Query(default=None),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps"),
):
    """
    ASR endpoint that uses audio content hash for deduplication.
    BLOCKS until processing is complete, then returns the result.

    If identical audio + task + language is already being processed,
    waits for that task to complete and returns the same result.
    """
    task_id = None

    try:
        logging.info(
            f"ASR {task.capitalize()} received for file '{video_file}'"
            if video_file
            else f"ASR {task.capitalize()} received"
        )

        # Read audio file content into memory
        file_content = await audio_file.read()

        if not file_content:
            await audio_file.close()
            return {
                "status": "error",
                "message": "Audio file is empty"
            }

        # Generate deterministic hash from audio (and optionally task/language)
        audio_hash = generate_audio_hash(file_content, task, language)
        task_id = f"asr-{audio_hash}"

        logging.debug(f"Generated audio hash: {audio_hash} for ASR request")

        # Handle forced language
        final_language = language
        if force_detected_language_to:
            final_language = force_detected_language_to.to_iso_639_1()
            logging.info(f"Forcing detected language to {force_detected_language_to}")

        # Create result container for this task
        with task_results_lock:
            if task_id not in task_results:
                task_results[task_id] = TaskResult()
            task_result = task_results[task_id]

        # Queue the ASR task
        asr_task_data = {
            'path': task_id,  # DeduplicatedQueue uses this for dedup
            'type': 'asr',
            'task': task,
            'language': final_language,
            'video_file': video_file,
            'initial_prompt': initial_prompt,
            'audio_content': file_content,
            'encode': encode,
            'output': output,
            'word_timestamps': word_timestamps,
            'result_container': task_result,
        }

        # Try to queue (returns False if already queued/processing)
        if task_queue.put(asr_task_data):
            logging.info(f"ASR task {task_id} queued")
        else:
            logging.info(f"ASR task {task_id} already queued/processing - waiting for result")

        # BLOCK HERE until worker completes (respects concurrent_transcriptions)
        if task_result.wait(timeout=asr_timeout):
            if task_result.error:
                logging.error(f"ASR task {task_id} failed: {task_result.error}")
                return {
                    "status": "error",
                    "task_id": task_id,
                    "message": f"ASR processing failed: {task_result.error}"
                }
            else:
                logging.info(f"ASR task {task_id} completed")
                # BUG FIX: Clean up task_results entry after retrieval to prevent memory leak
                with task_results_lock:
                    task_results.pop(task_id, None)
                return StreamingResponse(
                    iter(task_result.result),
                    media_type="text/plain",
                    headers={'Source': f'{task.capitalize()}d using stable-ts from Subgen!'}
                )
        else:
            logging.error(f"ASR task {task_id} timed out")
            # Clean up on timeout as well
            with task_results_lock:
                task_results.pop(task_id, None)
            return {
                "status": "timeout",
                "task_id": task_id,
                "message": f"ASR processing timed out after {asr_timeout} seconds"
            }

    except Exception as e:
        logging.error(f"Error in ASR endpoint: {e}", exc_info=True)
        # Clean up on error if task_id was assigned
        if task_id:
            with task_results_lock:
                task_results.pop(task_id, None)
        return {"status": "error", "message": f"Error: {str(e)}"}
    finally:
        await audio_file.close()
        with task_results_lock:
            if task_id and task_id in task_results:
                del task_results[task_id]
                logging.debug(f"Cleaned up task_results entry for {task_id}")
