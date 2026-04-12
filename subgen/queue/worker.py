import os
import queue
import time
import logging
import threading

from subgen.config import (
    asr_engine,
    concurrent_transcriptions,
    parakeet_model_name,
    whisper_model,
    transcribe_device,
    compute_type,
    custom_regroup,
    filter_subtitles,
    min_subtitle_duration,
    normalize_audio,
    transcribe_or_translate,
    enable_diarization,
)
from subgen.queue.deduplicated_queue import task_queue
from subgen.services.transcription import gen_subtitles, asr_task_worker
from subgen.services.language_detection import detect_language_task, detect_language_from_upload
from subgen.integrations.plex import refresh_plex_metadata
from subgen.integrations.jellyfin import refresh_jellyfin_metadata


def transcription_worker():
    """Main worker thread with centralized logging and status tracking."""
    while True:
        task = None
        try:
            task = task_queue.get(block=True, timeout=1)
            task_type = task.get("type", "transcribe")
            path = task.get("path", "unknown")
            display_name = os.path.basename(path) if ("/" in str(path) or "\\" in str(path)) else path

            # Status for START log
            proc_count = len(task_queue.get_processing_tasks())
            queue_count = len(task_queue.get_queued_tasks())
            logging.info(f"WORKER START : [{task_type.upper():<10}] {display_name:^40} | Jobs: {proc_count} processing, {queue_count} queued")
            min_dur_str = f"{min_subtitle_duration}s" if min_subtitle_duration > 0 else "off"
            model_display = parakeet_model_name if asr_engine == 'parakeet' else whisper_model
            logging.info(
                f"  Config: engine={asr_engine}  model={model_display}  device={transcribe_device}  compute={compute_type}  "
                f"mode={task.get('transcribe_or_translate', transcribe_or_translate)}  "
                f"regroup={custom_regroup}  min_dur={min_dur_str}  normalize={normalize_audio}  "
                f"filter={filter_subtitles}  diarization={enable_diarization}"
            )

            start_time = time.time()
            if task_type == "detect_language":
                if "audio_content" in task:
                    detect_language_from_upload(task)
                else:
                    # Pass the full task data so we don't lose the Plex ID
                    detect_language_task(task['path'], original_task_data=task)
            elif task_type == "asr":
                asr_task_worker(task)
            else:  # transcribe
                gen_subtitles(task['path'], task['transcribe_or_translate'], task['force_language'])

                # --- METADATA REFRESH LOGIC ---
                # This runs ONLY after subtitles are successfully generated
                if 'plex_item_id' in task:
                    try:
                        logging.info(f"Refreshing Plex Metadata for item {task['plex_item_id']}")
                        refresh_plex_metadata(task['plex_item_id'], task['plex_server'], task['plex_token'])
                    except Exception as e:
                        logging.error(f"Failed to refresh Plex metadata: {e}")

                if 'jellyfin_item_id' in task:
                    try:
                        logging.info(f"Refreshing Jellyfin Metadata for item {task['jellyfin_item_id']}")
                        refresh_jellyfin_metadata(task['jellyfin_item_id'], task['jellyfin_server'], task['jellyfin_token'])
                    except Exception as e:
                        logging.error(f"Failed to refresh Jellyfin metadata: {e}")
                # ------------------------------

            # Status for FINISH log
            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            remaining_queued = len(task_queue.get_queued_tasks())
            logging.info(f"WORKER FINISH: [{task_type.upper():<10}] {display_name:^40} in {m}m {s}s | Remaining: {remaining_queued} queued")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error processing task: {e}", exc_info=True)
        finally:
            if task:
                task_queue.task_done()
                task_queue.mark_done(task)
                if asr_engine == 'parakeet':
                    from subgen.models.parakeet_model import delete_model as _delete
                else:
                    from subgen.models.whisper_model import delete_model as _delete
                _delete()


def start_workers():
    """Creates and starts concurrent_transcriptions worker daemon threads."""
    for _ in range(concurrent_transcriptions):
        threading.Thread(target=transcription_worker, daemon=True).start()
