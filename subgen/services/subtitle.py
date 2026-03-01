import os
import logging
from datetime import datetime
from typing import List

import av
from stable_whisper import Segment
from language_code import LanguageCode

from subgen.config import (
    append,
    whisper_model,
    namesublang,
    transcribe_or_translate,
    show_in_subname_subgen,
    show_in_subname_model,
    subtitle_language_naming_type,
    skip_if_language_is_not_set_but_subtitles_exist,
    only_skip_if_subgen_subtitle,
)
from subgen.constants import TIME_OFFSET, SUBTITLE_EXTENSIONS


def appendLine(result):
    if append:
        lastSegment = result.segments[-1]
        date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        appended_text = f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}"

        # Create a new segment with the updated information
        newSegment = Segment(
            start=lastSegment.start + TIME_OFFSET,
            end=lastSegment.end + TIME_OFFSET,
            text=appended_text,
            words=[],  # Empty list for words
            id=lastSegment.id + 1
        )

        # Append the new segment to the result's segments
        result.segments.append(newSegment)


def define_subtitle_language_naming(language: LanguageCode, naming_type: str):
    """
    Determines the naming format for a subtitle language based on the given type.

    Args:
        language (LanguageCode): The language code object containing methods to get different formats of the language name.
        naming_type (str): The type of naming format desired, such as 'ISO_639_1', 'ISO_639_2_T', 'ISO_639_2_B', 'NAME', or 'NATIVE'.

    Returns:
        str: The language name in the specified format. If an invalid type is provided, it defaults to the language's name.
    """
    if namesublang:
        return namesublang
        # If we are translating, then we ALWAYS output an english file.
    switch_dict = {
        "ISO_639_1": language.to_iso_639_1,
        "ISO_639_2_T": language.to_iso_639_2_t,
        "ISO_639_2_B": language.to_iso_639_2_b,
        "NAME": language.to_name,
        "NATIVE": lambda: language.to_name(in_english=False)
    }
    if transcribe_or_translate == 'translate':
        language = LanguageCode.ENGLISH
    return switch_dict.get(naming_type, language.to_name)()


def name_subtitle(file_path: str, language: LanguageCode) -> str:
    """
    Name the subtitle file to be written, based on the source file and the language of the subtitle.

    Args:
        file_path: The path to the source file.
        language: The language of the subtitle.

    Returns:
        The name of the subtitle file to be written.
    """
    subgen_part = ".subgen" if show_in_subname_subgen else ""
    model_part = f".{whisper_model}" if show_in_subname_model else ""
    lang_part = define_subtitle_language_naming(language, subtitle_language_naming_type)

    return f"{os.path.splitext(file_path)[0]}{subgen_part}{model_part}.{lang_part}.srt"


def get_subtitle_languages(video_path):
    """
    Extract language codes from each audio stream in the video file using pyav.
    :param video_path: Path to the video file
    :return: List of language codes for each subtitle stream
    """
    languages = []

    # Open the video file
    with av.open(video_path) as container:
        # Iterate through each audio stream
        for stream in container.streams.subtitles:
            # Access the metadata for each audio stream
            lang_code = stream.metadata.get('language')
            if lang_code:
                languages.append(LanguageCode.from_iso_639_2(lang_code))
            else:
                # Append 'und' (undefined) if no language metadata is present
                languages.append(LanguageCode.NONE)

    return languages


def has_subtitle_language(video_file, target_language: LanguageCode):
    """
    Determines if a subtitle file with the target language is available for a specified video file.

    This function checks both within the video file and in its associated folder for subtitles
    matching the specified language.

    Args:
        video_file: The path to the video file.
        target_language: The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    return has_subtitle_language_in_file(video_file, target_language) or has_subtitle_of_language_in_folder(video_file, target_language)


def has_subtitle_language_in_file(video_file: str, target_language: LanguageCode | None):
    """
    Checks if a video file contains subtitles with a specific language.

    Args:
        video_file (str): The path to the video file.
        target_language (LanguageCode | None): The language of the subtitle file to search for.

    Returns:
        bool: True if a subtitle file with the target language is found, False otherwise.
    """
    try:
        with av.open(video_file) as container:
            # Create a list of subtitle streams with 'language' metadata
            subtitle_streams = [
                stream for stream in container.streams
                if stream.type == 'subtitle' and 'language' in stream.metadata
            ]

            # Skip logic if target_language is None
            if target_language is LanguageCode.NONE:
                if skip_if_language_is_not_set_but_subtitles_exist and subtitle_streams:
                    logging.debug("Language is not set, but internal subtitles exist.")
                    return True
                if only_skip_if_subgen_subtitle:
                    #logging.debug("Skipping since only external subgen subtitles are considered.")
                    return False  # Skip if only looking for external subgen subtitles

            # Check if any subtitle stream matches the target language
            for stream in subtitle_streams:
                # Convert the subtitle stream's language to a LanguageCode instance and compare
                stream_language = LanguageCode.from_string(stream.metadata.get('language', '').lower())
                if stream_language == target_language:
                    #logging.debug(f"Subtitles in '{target_language}' language found in the video.")
                    return True

            #logging.debug(f"No subtitles in '{target_language}' language found in the video.")
            return False

    except Exception as e:
        logging.error(f"An error occurred while checking the file with pyav: {type(e).__name__}: {e}")
        return False


def has_subtitle_of_language_in_folder(video_file: str, target_language: LanguageCode, recursion: bool = True) -> bool:
    """Checks if the given folder has a subtitle file with the given language.
    Args:
        video_file (str): The path of the video file.
        target_language (LanguageCode): The language of the subtitle file to search for.
        recursion (bool): If True, search subfolders. If False, only the current folder.
    Returns:
        bool: True if a matching subtitle file is found, False otherwise.
    """
    video_folder = os.path.dirname(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # logging.debug(f"Searching for subtitles in: {video_folder}")

    for file_name in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file_name)

        # If it's a file and has a subtitle extension
        if os.path.isfile(file_path) and file_path.endswith(tuple(SUBTITLE_EXTENSIONS)):
            subtitle_name, ext = os.path.splitext(file_name)

            # Ensure the subtitle name starts with the video name
            if not subtitle_name.startswith(video_name):
                continue

            # Extract parts after video filename
            subtitle_parts = subtitle_name[len(video_name):].lstrip(".").split(".")

            # Check for "subgen"
            has_subgen = "subgen" in subtitle_parts

            # Special handling if only skipping for subgen subtitles
            if target_language == LanguageCode.NONE:
                if only_skip_if_subgen_subtitle:
                    if has_subgen:
                        logging.debug("Skipping subtitles because they are auto-generated ('subgen').")
                        return False
                logging.debug("Skipping subtitles because language is NONE.")
                return True  # Default behavior if subtitles exist

            # Check if the subtitle file matches the target language
            if is_valid_subtitle_language(subtitle_parts, target_language):
                if only_skip_if_subgen_subtitle and not has_subgen:
                    continue  # Ignore non-subgen subtitles if flag is set
                logging.debug(f"Found matching subtitle: {file_name} for language {target_language.name} (subgen={has_subgen})")
                return True

        # Recursively search subfolders
        elif os.path.isdir(file_path) and recursion:
            if has_subtitle_of_language_in_folder(os.path.join(file_path, os.path.basename(video_file)), target_language, False):
                return True

    return False


def is_valid_subtitle_language(subtitle_parts: List[str], target_language: LanguageCode) -> bool:
    """Checks if any part of the subtitle name matches the target language."""
    return any(LanguageCode.from_string(part) == target_language for part in subtitle_parts)
