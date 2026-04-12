"""
Centralized configuration module for subgen.

Reads ALL environment variables and exposes them as module-level variables.
Supports both new standardized names and legacy names for backwards compatibility.

STANDARDIZED NAMING CONVENTION:
- Use UPPERCASE with underscores for separation
- Group related variables with consistent prefixes:
  * PLEX_* for Plex server integration
  * JELLYFIN_* for Jellyfin server integration
  * PROCESS_* for media processing triggers
  * SKIP_* for all skip conditions
  * SUBTITLE_* for subtitle-related settings
  * WHISPER_* for Whisper model settings
  * TRANSCRIBE_* for transcription settings

BACKWARDS COMPATIBILITY:
Legacy environment variable names are still supported. If both new and old names are set,
the new standardized name takes precedence.
"""

import ast
import logging
import os

from fastapi import FastAPI
from language_code import LanguageCode

from subgen.constants import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from subgen.utils.conversion import convert_to_bool, get_env_with_fallback

# ---------------------------------------------------------------------------
# Server Integration - with backwards compatibility
# ---------------------------------------------------------------------------
plextoken: str = get_env_with_fallback('PLEX_TOKEN', 'PLEXTOKEN', 'token here')
plexserver: str = get_env_with_fallback('PLEX_SERVER', 'PLEXSERVER', 'http://192.168.1.111:32400')
jellyfintoken: str = get_env_with_fallback('JELLYFIN_TOKEN', 'JELLYFINTOKEN', 'token here')
jellyfinserver: str = get_env_with_fallback('JELLYFIN_SERVER', 'JELLYFINSERVER', 'http://192.168.1.111:8096')

# ---------------------------------------------------------------------------
# ASR Backend Selection
# ---------------------------------------------------------------------------
asr_engine: str = os.getenv('ASR_ENGINE', 'whisper').lower()  # 'whisper' or 'parakeet'
parakeet_model_name: str = os.getenv('PARAKEET_MODEL', 'nvidia/parakeet-tdt-1.1b')
ngram_lm_alpha: float = float(os.getenv('NGRAM_LM_ALPHA', '0.3'))
# Comma-separated list of words/phrases to boost recognition of (e.g. character names)
boost_words: str = os.getenv('BOOST_WORDS', '')

clean_text: bool = convert_to_bool(os.getenv('CLEAN_TEXT', True))
drop_nonverbal_segments: bool = convert_to_bool(os.getenv('DROP_NONVERBAL_SEGMENTS', False))

# ---------------------------------------------------------------------------
# Whisper Configuration
# ---------------------------------------------------------------------------
whisper_model: str = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads: int = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions: int = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device: str = os.getenv('TRANSCRIBE_DEVICE', 'cpu')

# ---------------------------------------------------------------------------
# Processing Control - with backwards compatibility
# ---------------------------------------------------------------------------
procaddedmedia: bool = get_env_with_fallback('PROCESS_ADDED_MEDIA', 'PROCADDEDMEDIA', True, convert_to_bool)
procmediaonplay: bool = get_env_with_fallback('PROCESS_MEDIA_ON_PLAY', 'PROCMEDIAONPLAY', True, convert_to_bool)

# ---------------------------------------------------------------------------
# Subtitle Configuration - with backwards compatibility
# ---------------------------------------------------------------------------
namesublang: str = get_env_with_fallback('SUBTITLE_LANGUAGE_NAME', 'NAMESUBLANG', '')

# ---------------------------------------------------------------------------
# System Configuration - with backwards compatibility
# ---------------------------------------------------------------------------
webhookport: int = get_env_with_fallback('WEBHOOK_PORT', 'WEBHOOKPORT', 9000, int)
word_level_highlight: bool = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
# BUG FIX: default was bare True (not a string); os.getenv returns the default as-is,
# so convert_to_bool would receive bool True instead of str 'True'.
debug: bool = convert_to_bool(os.getenv('DEBUG', 'True'))
use_path_mapping: bool = convert_to_bool(os.getenv('USE_PATH_MAPPING', False))
path_mapping_from: str = os.getenv('PATH_MAPPING_FROM', r'/tv')
path_mapping_to: str = os.getenv('PATH_MAPPING_TO', r'/Volumes/TV')
model_location: str = os.getenv('MODEL_PATH', './models')
monitor: bool = convert_to_bool(os.getenv('MONITOR', False))
transcribe_folders: str = os.getenv('TRANSCRIBE_FOLDERS', '')
transcribe_or_translate: str = os.getenv('TRANSCRIBE_OR_TRANSLATE', 'transcribe').lower()
# Two-pass transcribe+translate settings
translate_source_languages: str = os.getenv('TRANSLATE_SOURCE_LANGUAGES', 'fr,es,de,it,pt,ja,ko,zh,ru')
detect_confidence_threshold: float = float(os.getenv('DETECT_CONFIDENCE_THRESHOLD', '0.7'))
clear_vram_on_complete: bool = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type: str = os.getenv('COMPUTE_TYPE', 'auto')
append: bool = convert_to_bool(os.getenv('APPEND', False))
reload_script_on_change: bool = convert_to_bool(os.getenv('RELOAD_SCRIPT_ON_CHANGE', False))
lrc_for_audio_files: bool = convert_to_bool(os.getenv('LRC_FOR_AUDIO_FILES', True))
custom_regroup: str = os.getenv('CUSTOM_REGROUP', 'default')
min_subtitle_duration: float = float(os.getenv('MIN_SUBTITLE_DURATION', '0'))
normalize_audio: bool = convert_to_bool(os.getenv('NORMALIZE_AUDIO', True))
detect_language_length: int = int(os.getenv('DETECT_LANGUAGE_LENGTH', 30))
detect_language_offset: int = int(os.getenv('DETECT_LANGUAGE_OFFSET', 0))
model_cleanup_delay: int = int(os.getenv('MODEL_CLEANUP_DELAY', 30))
asr_timeout: int = int(os.getenv('ASR_TIMEOUT', 18000))
filter_subtitles: bool = convert_to_bool(os.getenv('FILTER_SUBTITLES', False))
enable_diarization: bool = convert_to_bool(os.getenv('ENABLE_DIARIZATION', False))
diarization_model: str = os.getenv('DIARIZATION_MODEL', 'english')

# ---------------------------------------------------------------------------
# Scene Detection Configuration
# ---------------------------------------------------------------------------
max_scene_duration: float = float(os.getenv('MAX_SCENE_DURATION', '30'))
use_silero_vad: bool = convert_to_bool(os.getenv('USE_SILERO_VAD', True))
silero_vad_threshold: float = float(os.getenv('SILERO_VAD_THRESHOLD', '0.08'))
silero_min_silence_ms: int = int(os.getenv('SILERO_MIN_SILENCE_MS', '1500'))
silero_min_speech_ms: int = int(os.getenv('SILERO_MIN_SPEECH_MS', '100'))

# ---------------------------------------------------------------------------
# Skip Configuration - with backwards compatibility
# ---------------------------------------------------------------------------
skipifexternalsub: bool = get_env_with_fallback(
    'SKIP_IF_EXTERNAL_SUBTITLES_EXIST', 'SKIPIFEXTERNALSUB', False, convert_to_bool,
)
skip_if_to_transcribe_sub_already_exist: bool = get_env_with_fallback(
    'SKIP_IF_TARGET_SUBTITLES_EXIST', 'SKIP_IF_TO_TRANSCRIBE_SUB_ALREADY_EXIST', True, convert_to_bool,
)
skipifinternalsublang: LanguageCode = LanguageCode.from_string(
    get_env_with_fallback('SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE', 'SKIPIFINTERNALSUBLANG', ''),
)
plex_queue_next_episode: bool = convert_to_bool(os.getenv('PLEX_QUEUE_NEXT_EPISODE', False))
plex_queue_season: bool = convert_to_bool(os.getenv('PLEX_QUEUE_SEASON', False))
plex_queue_series: bool = convert_to_bool(os.getenv('PLEX_QUEUE_SERIES', False))

# ---------------------------------------------------------------------------
# Language and Skip Configuration - with backwards compatibility
# ---------------------------------------------------------------------------
skip_lang_codes_list: list[LanguageCode] = (
    [
        LanguageCode.from_string(code)
        for code in get_env_with_fallback('SKIP_SUBTITLE_LANGUAGES', 'SKIP_LANG_CODES', '').split("|")
    ]
    if get_env_with_fallback('SKIP_SUBTITLE_LANGUAGES', 'SKIP_LANG_CODES')
    else []
)
force_detected_language_to: LanguageCode = LanguageCode.from_string(
    os.getenv('FORCE_DETECTED_LANGUAGE_TO', ''),
)
preferred_audio_languages: list[LanguageCode] = [
    LanguageCode.from_string(code)
    for code in os.getenv('PREFERRED_AUDIO_LANGUAGES', 'eng').split("|")
]  # in order of preference
limit_to_preferred_audio_languages: bool = convert_to_bool(
    os.getenv('LIMIT_TO_PREFERRED_AUDIO_LANGUAGE', False),
)
skip_if_audio_track_is_in_list: list[LanguageCode] = (
    [
        LanguageCode.from_string(code)
        for code in get_env_with_fallback('SKIP_IF_AUDIO_LANGUAGES', 'SKIP_IF_AUDIO_TRACK_IS', '').split("|")
    ]
    if get_env_with_fallback('SKIP_IF_AUDIO_LANGUAGES', 'SKIP_IF_AUDIO_TRACK_IS')
    else []
)

# ---------------------------------------------------------------------------
# Additional Subtitle Configuration - with backwards compatibility
# ---------------------------------------------------------------------------
subtitle_language_naming_type: str = os.getenv('SUBTITLE_LANGUAGE_NAMING_TYPE', 'ISO_639_2_B')
only_skip_if_subgen_subtitle: bool = get_env_with_fallback(
    'SKIP_ONLY_SUBGEN_SUBTITLES', 'ONLY_SKIP_IF_SUBGEN_SUBTITLE', False, convert_to_bool,
)
skip_unknown_language: bool = convert_to_bool(os.getenv('SKIP_UNKNOWN_LANGUAGE', False))
skip_if_language_is_not_set_but_subtitles_exist: bool = get_env_with_fallback(
    'SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST',
    'SKIP_IF_LANGUAGE_IS_NOT_SET_BUT_SUBTITLES_EXIST',
    False,
    convert_to_bool,
)
should_whisper_detect_audio_language: bool = convert_to_bool(
    os.getenv('SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE', False),
)

show_in_subname_subgen: bool = convert_to_bool(os.getenv('SHOW_IN_SUBNAME_SUBGEN', True))
show_in_subname_model: bool = convert_to_bool(os.getenv('SHOW_IN_SUBNAME_MODEL', True))

# ---------------------------------------------------------------------------
# Advanced Configuration - SUBGEN_KWARGS
# ---------------------------------------------------------------------------
# BUG FIX: also catch SyntaxError and TypeError (bare except ValueError was insufficient)
_default_kwargs: dict = {
    'beam_size': 5,
    'condition_on_previous_text': True,
}
try:
    _user_kwargs: dict = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
except (ValueError, SyntaxError, TypeError):
    _user_kwargs = {}
    logging.info("kwargs (SUBGEN_KWARGS) is an invalid dictionary, defaulting to empty '{}'")
# Merge: user overrides take precedence over defaults
kwargs: dict = {**_default_kwargs, **_user_kwargs}

# ---------------------------------------------------------------------------
# Device normalization
# ---------------------------------------------------------------------------
if transcribe_device == "gpu":
    transcribe_device = "cuda"

# ---------------------------------------------------------------------------
# Docker detection
# ---------------------------------------------------------------------------
in_docker: bool = os.path.exists('/.dockerenv')
docker_status: str = "Docker" if in_docker else "Standalone"

# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------
app: FastAPI = FastAPI()
