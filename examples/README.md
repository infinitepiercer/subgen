# Example Docker Compose Configurations

Pick the example closest to your setup and copy it into your `docker-compose.yml`.

| Example | GPU | Use Case |
|---------|:---:|----------|
| [gpu-english](docker-compose.gpu-english.yml) | Yes | English media library — forced English prevents hallucinations during silence |
| [gpu-multilingual](docker-compose.gpu-multilingual.yml) | Yes | Mixed-language library (anime, foreign films) — auto-detects audio language |
| [gpu-parakeet](docker-compose.gpu-parakeet.yml) | Yes | NVIDIA Parakeet TDT — fast English ASR, less hallucination-prone than Whisper |
| [cpu](docker-compose.cpu.yml) | No | No NVIDIA GPU — uses lighter model and CPU image |
| [bazarr](docker-compose.bazarr.yml) | Yes | Bazarr Whisper provider — Bazarr sends audio over HTTP, no shared paths needed |

## Which one should I use?

- **Mostly English content?** Start with **gpu-english**. It locks Whisper to English so it can't hallucinate random languages (German, Chinese, etc.) during quiet sections.
- **Want fewer hallucinations?** Try **gpu-parakeet**. NVIDIA Parakeet-TDT-0.6B-V3 is a fast English-only model that rarely hallucinates. Requires building the image with `--build-arg ASR_ENGINE=parakeet`.
- **Anime / foreign films / multi-language?** Use **gpu-multilingual**. Whisper detects the language automatically and `transcribe_and_translate` handles the rest.
- **No NVIDIA GPU?** Use **cpu**. Expect slower transcription (~2-4x realtime with the `medium` model).
- **Using Bazarr?** Use **bazarr**. Bazarr controls the language and task per request. Disable media server webhooks to avoid duplicates.

## Anti-Hallucination

All examples include `SUBGEN_KWARGS` and `FILTER_SUBTITLES` for layered hallucination prevention. Whisper sometimes generates fake text during silence ("I'm so sorry.", "Thanks for watching", random CJK characters). The defense works in layers:

| Layer | Parameter | What it does |
|-------|-----------|-------------|
| 1. Pre-processing | `vad=True` | Silero VAD masks silence before Whisper sees it |
| 2. During decoding | `hallucination_silence_threshold=2` | Detect and skip anomalous segments after 2s+ silence |
| 3. During decoding | `condition_on_previous_text=False` | Prevent hallucinations from cascading into the next segment |
| 4. During decoding | `no_speech_threshold=0.2` | Aggressively mark low-confidence segments as silence |
| 5. Post-processing | `FILTER_SUBTITLES=true` | Remove known phrases, gibberish, ghost words, foreign-script text |

To tune these, edit `SUBGEN_KWARGS` in your compose file:

```yaml
SUBGEN_KWARGS: "{'vad': True, 'hallucination_silence_threshold': 2, 'condition_on_previous_text': False, 'no_speech_threshold': 0.2}"
```

If you're getting **missing subtitles for quiet speech**, try lowering `no_speech_threshold` to `0.3`.

## Common tweaks

| What | How |
|------|-----|
| Less VRAM usage | Change `WHISPER_MODEL` to `medium` and/or `COMPUTE_TYPE` to `int8` |
| Skip files that already have subtitles | Set `SKIP_IF_TARGET_SUBTITLES_EXIST: "true"` |
| Auto-transcribe on media add/play | Set `PROCESS_ADDED_MEDIA: "true"` and configure your Plex/Jellyfin webhook |
| Word-level karaoke highlighting | Set `WORD_LEVEL_HIGHLIGHT: "true"` |
| Minimum subtitle display time | Set `MIN_SUBTITLE_DURATION: "1.5"` (seconds) |
| Skip intro music for language detection | Set `DETECT_LANGUAGE_OFFSET: 90` (seconds to skip) |

## Volume paths

All examples use placeholder paths — update them to match your media server:

```yaml
volumes:
  - ./models:/subgen/models:rw       # model cache (keep this)
  - /path/to/media:/media:rw         # change to your actual media path
```

If your media server sees files at `/tv/show.mkv` but Subgen sees them at `/media/tv/show.mkv`, enable path mapping:

```yaml
USE_PATH_MAPPING: "true"
PATH_MAPPING_FROM: /tv
PATH_MAPPING_TO: /media/tv
```
