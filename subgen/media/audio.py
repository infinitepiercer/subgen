import logging

import ffmpeg
import numpy as np
from language_code import LanguageCode


def normalize_audio(audio_input, is_file_path: bool = True) -> bytes:
    """Normalize audio loudness using ffmpeg's loudnorm filter (EBU R128).

    Brings quiet audio up and loud audio down to a consistent level (-16 LUFS,
    slightly louder than broadcast standard) to improve Whisper transcription
    accuracy.  Does NOT modify the original file — returns normalized WAV bytes.

    Args:
        audio_input: File path (str) or raw audio bytes.
        is_file_path: True if audio_input is a file path, False if bytes.

    Returns:
        Normalized audio as WAV bytes (16kHz mono PCM s16le).
    """
    try:
        if is_file_path:
            input_stream = ffmpeg.input(audio_input)
            run_kwargs = {}
        else:
            input_stream = ffmpeg.input('pipe:0')
            run_kwargs = {'input': audio_input}

        # Filter chain to maximise speech intelligibility:
        #   1. highpass=200  – remove low-frequency rumble / HVAC noise
        #   2. acompressor   – bring up quiet speech (fast attack, slow release)
        #   3. loudnorm      – EBU R128 loudness normalisation to -16 LUFS
        af_chain = (
            'highpass=f=200,'
            'acompressor=threshold=-25dB:ratio=4:attack=5:release=200,'
            'loudnorm=I=-16:TP=-1.5:LRA=11'
        )

        out, _ = (
            input_stream
            .output(
                'pipe:1',
                format='wav',
                acodec='pcm_s16le',
                ar=16000,
                ac=1,
                af=af_chain,
            )
            .run(capture_stdout=True, capture_stderr=True, **run_kwargs)
        )

        if not out:
            logging.warning("Audio normalization produced empty output, using original audio")
            return audio_input if not is_file_path else None

        logging.info("Audio normalized successfully (%d bytes)", len(out))
        return out

    except ffmpeg.Error as e:
        logging.warning("Audio normalization failed: %s — using original audio", e.stderr.decode() if e.stderr else str(e))
        return audio_input if not is_file_path else None
    except Exception as e:
        logging.warning("Audio normalization failed: %s — using original audio", e)
        return audio_input if not is_file_path else None


async def get_audio_chunk(audio_file, offset=None, length=None, sample_rate=16000, audio_format=np.int16):
    """
    Extract a chunk of audio from a file, starting at the given offset and of the given length.

    :param audio_file: The audio file (UploadFile or file-like object).
    :param offset: The offset in seconds to start the extraction.
    :param length: The length in seconds for the chunk to be extracted.
    :param sample_rate: The sample rate of the audio (default 16000).
    :param audio_format: The audio format to interpret (default int16, 2 bytes per sample).

    :return: A numpy array containing the extracted audio chunk.
    """
    if offset is None:
        from subgen.config import detect_language_offset
        offset = detect_language_offset
    if length is None:
        from subgen.config import detect_language_length
        length = detect_language_length

    # Number of bytes per sample (for int16, 2 bytes per sample)
    bytes_per_sample = np.dtype(audio_format).itemsize

    # Calculate the start byte based on offset and sample rate
    start_byte = offset * sample_rate * bytes_per_sample

    # Calculate the length in bytes based on the length in seconds
    length_in_bytes = length * sample_rate * bytes_per_sample

    # Seek to the start position (this assumes the audio_file is a file-like object)
    await audio_file.seek(start_byte)

    # Read the required chunk of audio (length_in_bytes)
    chunk = await audio_file.read(length_in_bytes)

    # Convert the chunk into a numpy array (normalized to float32)
    audio_data = np.frombuffer(chunk, dtype=audio_format).flatten().astype(np.float32) / 32768.0

    return audio_data


def extract_audio_segment_from_content(audio_content: bytes, start_time: int, duration: int) -> bytes:
    """
    Extract a segment of audio from in-memory content using FFmpeg.

    Args:
        audio_content: Raw audio bytes
        start_time: Start time in seconds
        duration: Duration in seconds

    Returns:
        Audio bytes of the extracted segment
    """
    try:
        logging.info(f"Extracting audio segment: start_time={start_time}s, duration={duration}s")

        out, _ = (
            ffmpeg
            .input('pipe:0', ss=start_time, t=duration)
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)
            .run(input=audio_content, capture_stdout=True, capture_stderr=True)
        )

        if not out:
            raise ValueError("FFmpeg output is empty")

        return out

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return audio_content  # Fallback to original if extraction fails
    except Exception as e:
        logging.error(f"Error extracting audio segment: {str(e)}")
        return audio_content  # Fallback to original


def extract_audio_segment_to_memory(input_file, start_time, duration):
    """
    Extract a segment of audio from input_file, starting at start_time for duration seconds.

    :param input_file: UploadFile object or path to the input audio file
    :param start_time: Start time in seconds (e.g., 60 for 1 minute)
    :param duration: Duration in seconds (e.g., 30 for 30 seconds)
    :return: bytes containing the audio segment, or None on error

    """
    try:
        if hasattr(input_file, 'file') and hasattr(input_file.file, 'read'):  # Handling UploadFile
            input_file.file.seek(0)  # Ensure the file pointer is at the beginning
            input_stream = 'pipe:0'
            input_kwargs = {'input': input_file.file.read()}
        elif isinstance(input_file, str):  # Handling local file path
            input_stream = input_file
            input_kwargs = {}
        else:
            raise ValueError("Invalid input: input_file must be a file path or an UploadFile object.")

        logging.info(f"Extracting audio from: {input_stream}, start_time: {start_time}, duration: {duration}")

        # Run FFmpeg to extract the desired segment
        out, _ = (
            ffmpeg
            .input(input_stream, ss=start_time, t=duration)  # Set start time and duration
            .output('pipe:1', format='wav', acodec='pcm_s16le', ar=16000)  # Output to pipe as WAV
            .run(capture_stdout=True, capture_stderr=True, **input_kwargs)
        )

        # Check if the output is empty or null
        if not out:
            raise ValueError("FFmpeg output is empty, possibly due to invalid input.")

        return out

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None


def extract_audio_track_to_memory(input_video_path, track_index) -> bytes | None:
    """
    Extract a specific audio track from a video file to memory using FFmpeg.

    Args:
        input_video_path (str): The path to the video file.
        track_index (int): The index of the audio track to extract. If None, skip extraction.

    Returns:
        bytes | None: The audio data as bytes, or None if extraction failed.
    """
    if track_index is None:
        logging.warning(f"Skipping audio track extraction for {input_video_path} because track index is None")
        return None

    try:
        # Use FFmpeg to extract the specific audio track and output to memory
        out, _ = (
            ffmpeg.input(input_video_path)
            .output(
                "pipe:",  # Direct output to a pipe
                map=f"0:{track_index}",  # Select the specific audio track
                format="wav",  # Output format
                ac=1,  # Mono audio (optional)
                ar=16000,  # Sample rate 16 kHz (recommended for speech models)
                loglevel="quiet"
            )
            .run(capture_stdout=True, capture_stderr=True)  # Capture output in memory
        )
        return out

    except ffmpeg.Error as e:
        logging.error("An error occurred: " + e.stderr.decode())
        return None


def get_audio_tracks(video_file):
    """
    Extracts information about the audio tracks in a file.

    Returns:
        List of dictionaries with information about each audio track.
        Each dictionary has the following keys:
            index (int): The stream index of the audio track.
            codec (str): The name of the audio codec.
            channels (int): The number of audio channels.
            language (LanguageCode): The language of the audio track.
            title (str): The title of the audio track.
            default (bool): Whether the audio track is the default for the file.
            forced (bool): Whether the audio track is forced.
            original (bool): Whether the audio track is the original.
            commentary (bool): Whether the audio track is a commentary.
    """
    try:
        # Probe the file to get audio stream metadata
        probe = ffmpeg.probe(video_file, select_streams='a')
        audio_streams = probe.get('streams', [])

        # Extract information for each audio track
        audio_tracks = []
        for stream in audio_streams:
            audio_track = {
                "index": int(stream.get("index", None)),
                "codec": stream.get("codec_name", "Unknown"),
                "channels": int(stream.get("channels", None)),
                "language": LanguageCode.from_iso_639_2(stream.get("tags", {}).get("language", "Unknown")),
                "title": stream.get("tags", {}).get("title", "None"),
                "default": stream.get("disposition", {}).get("default", 0) == 1,
                "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                "original": stream.get("disposition", {}).get("original", 0) == 1,
                "commentary": "commentary" in stream.get("tags", {}).get("title", "").lower()
            }
            audio_tracks.append(audio_track)
        return audio_tracks

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading audio track information: {str(e)}")
        return []


def get_audio_track_by_language(audio_tracks, language):
    """
    Returns the first audio track with the given language.

    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        language (str): The language of the audio track to search for.

    Returns:
        dict: The first audio track with the given language, or None if no match is found.
    """
    for track in audio_tracks:
        if track['language'] == language:
            return track
    return None


def find_language_audio_track(audio_tracks, find_languages):
    """
    Checks if an audio track with any of the given languages is present in the list of audio tracks.
    Returns the first language from `find_languages` that matches.

    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
        find_languages (list): A list language codes to search for.

    Returns:
        str or None: The first language found from `find_languages`, or None if no match is found.
    """
    for language in find_languages:
        for track in audio_tracks:
            if track['language'] == language:
                return language
    return None


def find_default_audio_track_language(audio_tracks):
    """
    Finds the language of the default audio track in the given list of audio tracks.

    Args:
        audio_tracks (list): A list of dictionaries containing information about each audio track.
            Must contain the key "default" which is a boolean indicating if the track is the default track.

    Returns:
        str: The ISO 639-2 code of the language of the default audio track, or None if no default track was found.
    """
    for track in audio_tracks:
        if track['default'] is True:
            return track['language']
    return None


def get_audio_languages(video_path):
    """
    Extract language codes from each audio stream in the video file.

    :param video_path: Path to the video file
    :return: List of language codes for each audio stream
    """
    audio_tracks = get_audio_tracks(video_path)
    return [track['language'] for track in audio_tracks]


def handle_multiple_audio_tracks(file_path: str, language: LanguageCode | None = None) -> bytes | None:
    """
    Handles the possibility of a media file having multiple audio tracks.

    If the media file has multiple audio tracks, it will extract the audio track of the selected language.
    Otherwise, it will extract the first audio track.

    Parameters:
    file_path (str): The path to the media file.
    language (LanguageCode | None): The language of the audio track to search for.
        If None, it will extract the first audio track.

    Returns:
    bytes | None: The audio data as bytes, or None if no audio track was extracted.
    """
    audio_bytes = None
    audio_tracks = get_audio_tracks(file_path)

    if len(audio_tracks) > 1:
        logging.debug(f"Handling multiple audio tracks from {file_path} and planning to extract audio track of language {language}")
        logging.debug(
            "Audio tracks:\n"
            + "\n".join([f"  - {track['index']}: {track['codec']} {track['language']} {('default' if track['default'] else '')}" for track in audio_tracks])
        )

        audio_track = None
        if language is not None:
            audio_track = get_audio_track_by_language(audio_tracks, language)
        if audio_track is None:
            audio_track = audio_tracks[0]

        audio_bytes = extract_audio_track_to_memory(file_path, audio_track["index"])
        if audio_bytes is None:
            logging.error(f"Failed to extract audio track {audio_track['index']} from {file_path}")
            return None
    return audio_bytes
