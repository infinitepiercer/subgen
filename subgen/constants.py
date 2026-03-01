VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpg", ".mpeg",
    ".3gp", ".ogv", ".vob", ".rm", ".rmvb", ".ts", ".m4v", ".f4v", ".svq3",
    ".asf", ".m2ts", ".divx", ".xvid"
)

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".alac", ".m4a", ".opus",
    ".aiff", ".aif", ".pcm", ".ra", ".ram", ".mid", ".midi", ".ape", ".wv",
    ".amr", ".vox", ".tak", ".spx", ".m4b", ".mka"
)

SUBTITLE_EXTENSIONS = frozenset({
    '.srt', '.vtt', '.sub', '.ass', '.ssa', '.idx', '.sbv', '.pgs', '.ttml', '.lrc'
})

TIME_OFFSET = 5

SUPPRESSED_LOG_PATTERNS = (
    "Compression ratio threshold is not met",
    "Processing segment at",
    "Log probability threshold is",
    "Reset prompt",
    "Attempting to release",
    "released on ",
    "Attempting to acquire",
    "acquired on",
    "header parsing failed",
    "timescale not set",
    "misdetection possible",
    "srt was added",
    "doesn't have any audio to transcribe",
    "Calling on_",
)

SILENCED_LOGGERS = (
    "multipart",
    "urllib3",
    "watchfiles",
    "asyncio",
    "httpcore",
    "httpx",
    "huggingface_hub",
)
