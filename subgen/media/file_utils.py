import os
import time
import logging

import av

from subgen.constants import VIDEO_EXTENSIONS, AUDIO_EXTENSIONS


def isAudioFileExtension(file_extension: str) -> bool:
    return file_extension.casefold() in AUDIO_EXTENSIONS


def has_audio(file_path: str) -> bool:
    try:
        if not is_valid_path(file_path):
            return False

        if not (has_video_extension(file_path) or has_audio_extension(file_path)):
            return False

        with av.open(file_path) as container:
            # Check for an audio stream and ensure it has a valid codec
            for stream in container.streams:
                if stream.type == 'audio':
                    # Check if the stream has a codec and if it is valid
                    if stream.codec_context and stream.codec_context.name != 'none':
                        return True
                    else:
                        logging.debug(f"Unsupported or missing codec for audio stream in {file_path}")
            return False

    except (av.FFmpegError, UnicodeDecodeError):
        logging.debug(f"Error processing file {file_path}")
        return False


def is_valid_path(file_path: str) -> bool:
    # Check if the path is a file
    if not os.path.isfile(file_path):
        # If it's not a file, check if it's a directory
        if not os.path.isdir(file_path):
            logging.warning(f"{file_path} is neither a file nor a directory. Are your volumes correct?")
            return False
        else:
            logging.debug(f"{file_path} is a directory, skipping processing as a file.")
            return False
    else:
        return True


def has_video_extension(file_name: str) -> bool:
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in VIDEO_EXTENSIONS


def has_audio_extension(file_name: str) -> bool:
    file_extension = os.path.splitext(file_name)[1].lower()  # Get the file extension
    return file_extension in AUDIO_EXTENSIONS


def is_file_stable(file_path: str, wait_time: int = 2, check_intervals: int = 3) -> bool:
    """Returns True if the file size is stable for a given number of checks."""
    if not os.path.exists(file_path):
        return False

    previous_size = -1
    for _ in range(check_intervals):
        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            return False  # File might still be inaccessible

        if current_size == previous_size:
            return True  # File is stable
        previous_size = current_size
        time.sleep(wait_time)

    return False  # File is still changing


def get_file_name_without_extension(file_path: str) -> str:
    file_name, file_extension = os.path.splitext(file_path)
    return file_name


def write_lrc(result, file_path: str) -> None:
    with open(file_path, "w") as file:
        for segment in result.segments:
            minutes, seconds = divmod(int(segment.start), 60)
            fraction = int((segment.start - int(segment.start)) * 100)
            # remove embedded newlines in text, since some players ignore text after newlines
            text = segment.text[:].replace('\n', '')
            file.write(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}]{text}\n")
