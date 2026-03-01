"""
Folder monitoring and existing-file scanning for automatic transcription.

Contains:
- NewFileHandler: watchdog handler that queues new/modified media files.
- transcribe_existing: scans folders for existing files and optionally starts
  a polling observer to watch for new ones.
"""

import logging
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from language_code import LanguageCode

from subgen.config import monitor, transcribe_or_translate
from subgen.media.file_utils import has_audio, is_file_stable
from subgen.media.path_mapping import path_mapping
from subgen.services.transcription import gen_subtitles_queue


class NewFileHandler(FileSystemEventHandler):
    """Queues newly-created or modified media files for transcription."""

    def create_subtitle(self, event) -> None:
        """Only process if it is a file with audio."""
        if not event.is_directory:
            file_path = event.src_path
            if has_audio(file_path):
                logging.info(f"File: {path_mapping(file_path)} was added")
                gen_subtitles_queue(path_mapping(file_path), transcribe_or_translate)

    def handle_event(self, event) -> None:
        """Wait for stability before processing the file."""
        file_path = event.src_path
        if is_file_stable(file_path):
            self.create_subtitle(event)

    def on_created(self, event) -> None:
        time.sleep(5)  # Extra buffer time for new files
        self.handle_event(event)

    def on_modified(self, event) -> None:
        self.handle_event(event)


def transcribe_existing(
    transcribe_folders_str: str,
    force_language: LanguageCode | None = None,
) -> None:
    """Scan *transcribe_folders_str* (pipe-separated) for existing media files
    and queue them for transcription.  Optionally starts a watchdog observer
    when ``monitor`` is enabled.

    Bug fixes compared to the original monolith:
    - Uses *folder_list* instead of re-binding ``transcribe_folders``, which
      previously shadowed the module-level config variable.
    - The single-file check is now performed inside the per-folder loop so
      every entry is tested, not just the last one.
    """
    folder_list = transcribe_folders_str.split("|")
    logging.info("Starting to search folders to see if we need to create subtitles.")
    logging.debug("The folders are:")
    for path in folder_list:
        logging.debug(path)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    gen_subtitles_queue(
                        path_mapping(file_path),
                        transcribe_or_translate,
                        force_language,
                    )
        # BUG FIX: moved inside the loop so each entry is checked, not just
        # the last one after the loop finishes.
        elif os.path.isfile(path):
            if has_audio(path):
                gen_subtitles_queue(
                    path_mapping(path),
                    transcribe_or_translate,
                    force_language,
                )

    # Set up the observer to watch for new files
    if monitor:
        observer = Observer()
        for path in folder_list:
            if os.path.isdir(path):
                handler = NewFileHandler()
                observer.schedule(handler, path, recursive=True)
        observer.start()
        logging.info(
            "Finished searching and queueing files for transcription. "
            "Now watching for new files."
        )
