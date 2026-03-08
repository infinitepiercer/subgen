<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="150">

# Subgen — Automatic Subtitle Generator

Subgen uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) and [stable-ts](https://github.com/jianfch/stable-ts) to automatically generate `.srt` subtitles for your media. It integrates with **Plex**, **Emby**, **Jellyfin**, **Tautulli**, and **Bazarr** via webhooks and supports both Nvidia GPU and CPU transcription.

## Quick Start

### Docker (recommended)

| Tag | Base Image | GPU Support | CPU Support | Notes |
|-----|-----------|:-----------:|:-----------:|-------|
| `latest` | `nvidia/cuda` | Yes | Yes | Full image — supports both GPU and CPU via `TRANSCRIBE_DEVICE` |
| `cpu` | `python:3.11-slim` | No | Yes | Smaller image — no CUDA libraries |

```bash
docker pull ghcr.io/regix1/subgen:latest
# or for a smaller CPU-only image:
docker pull ghcr.io/regix1/subgen:cpu
```

### Docker Compose

```yaml
services:
  subgen:
    container_name: subgen
    image: ghcr.io/regix1/subgen:latest  # or ghcr.io/regix1/subgen:cpu
    ports:
      - "9000:9000"
    environment:
      - TRANSCRIBE_DEVICE=cpu        # "cpu", "gpu", or "cuda"
      - WHISPER_MODEL=medium
      - WHISPER_THREADS=4
    volumes:
      - /path/to/tv:/tv              # must match your media server paths
      - /path/to/movies:/movies
      - /path/to/models:/subgen/models  # persist downloaded models
    # Uncomment below for GPU support:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]
```

### Standalone (no Docker)

Requires Python 3.9–3.11 and ffmpeg.

```bash
python3 launcher.py -u -i -s
```

| Flag | Description |
|------|-------------|
| `-u` | Update/download latest subgen |
| `-i` | Install/update all dependencies |
| `-s` | Interactive Bazarr setup wizard |
| `-d` | Enable debug logging |
| `-b <branch>` | Download from a specific branch |

For GPU, you need NVIDIA drivers with CUDA Toolkit 12.3+.

---

## Integrations

Subgen listens on port **9000** by default. Each integration has its own endpoint.

### Bazarr

Configure the **Whisper Provider** in Bazarr:
- **Endpoint:** `http://<subgen-ip>:9000`

![bazarr_configuration](https://wiki.bazarr.media/Additional-Configuration/images/whisper_config.png)

Path mapping is not needed — Bazarr sends files over HTTP. See the [Bazarr wiki](https://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/) for details.

> **Tip:** Avoid enabling Bazarr *and* media server webhooks at the same time, or you'll get duplicate subtitles.

### Plex

1. Go to **Settings > Webhooks** ([docs](https://support.plex.tv/articles/115002267687-webhooks/))
2. Add: `http://<subgen-ip>:9000/plex`
3. Set `PLEX_TOKEN` and `PLEX_SERVER` environment variables

### Jellyfin

1. Install the **Webhooks** plugin
2. Add a **Generic Destination**: `http://<subgen-ip>:9000/jellyfin`
3. Check **Item Added**, **Playback Start**, and **Send All Properties**
4. Add request header: `Content-Type: application/json`

### Emby

1. Create a webhook: `http://<subgen-ip>:9000/emby`
2. Set **Request content type** to `multipart/form-data`
3. Enable desired events (New Media Added, Start, Unpause)

See [discussion #115](https://github.com/McCloudS/subgen/discussions/115#discussioncomment-10569277) for screenshots.

### Tautulli

1. **URL:** `http://<subgen-ip>:9000/tautulli`
2. **Method:** POST
3. **Triggers:** Playback Start, Recently Added

**Header** (both triggers):
```json
{ "source": "Tautulli" }
```

**Playback Start data:**
```json
{
  "event": "played",
  "file": "{file}",
  "filename": "{filename}",
  "mediatype": "{media_type}"
}
```

**Recently Added data:**
```json
{
  "event": "added",
  "file": "{file}",
  "filename": "{filename}",
  "mediatype": "{media_type}"
}
```

> **Note:** For Plex, Emby, and Jellyfin — Subgen must see the exact same file paths as your media server. If they differ, enable `USE_PATH_MAPPING`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Returns version info |
| `POST` | `/plex` | Plex webhook |
| `POST` | `/jellyfin` | Jellyfin webhook |
| `POST` | `/emby` | Emby webhook |
| `POST` | `/tautulli` | Tautulli webhook |
| `POST` | `/asr` | Transcribe/translate an uploaded file (Bazarr/Whisper provider) |
| `POST` | `/batch` | Batch-process a file or folder |
| `POST` | `/detect-language` | Detect audio language from an uploaded file |

---

## Environment Variables

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_DEVICE` | `cpu` | `cpu`, `gpu`, or `cuda` |
| `WHISPER_MODEL` | `medium` | Model to use (see [Models](#models) below) |
| `WHISPER_THREADS` | `4` | Threads for computation |
| `CONCURRENT_TRANSCRIPTIONS` | `2` | Parallel transcription jobs |
| `MODEL_PATH` | `./models` | Where models are stored |
| `COMPUTE_TYPE` | `auto` | Quantization type ([reference](https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md)) |
| `WEBHOOK_PORT` | `9000` | Listening port |
| `DEBUG` | `True` | Debug logging |
| `RELOAD_SCRIPT_ON_CHANGE` | `False` | Auto-reload subgen when script file changes (development use) |
| `UPDATE` | `False` | Pull latest subgen.py on container start (via launcher.py) |

### Transcription Behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_OR_TRANSLATE` | `transcribe` | `transcribe`, `translate`, or `transcribe_and_translate` |
| `TRANSLATE_SOURCE_LANGUAGES` | `fr,es,de,it,pt,ja,ko,zh,ru` | Language codes to install offline translation models for (`transcribe_and_translate` mode) |
| `DETECT_CONFIDENCE_THRESHOLD` | `0.7` | Min confidence a segment is English before skipping translation (`transcribe_and_translate` mode) |
| `DETECT_LANGUAGE_LENGTH` | `30` | Seconds of audio to use for language detection |
| `DETECT_LANGUAGE_OFFSET` | `0` | Seconds to skip before running language detection (avoids intros/songs) |
| `FORCE_DETECTED_LANGUAGE_TO` | `''` | Force a 2-letter language code instead of auto-detection |
| `SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE` | `False` | Let Whisper detect language when no audio language tag exists |
| `USE_MODEL_PROMPT` | `False` | Use a prompt to improve punctuation in transcriptions |
| `CUSTOM_MODEL_PROMPT` | `''` | Override the default prompt ([guide](https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d)) |
| `NORMALIZE_AUDIO` | `False` | Normalize audio loudness (EBU R128) before transcription for better accuracy with quiet/inconsistent audio |
| `SUBGEN_KWARGS` | `{}` | Python dict of extra options passed to `model.transcribe()` |
| `ENABLE_DIARIZATION` | `False` | Enable speaker diarization to label subtitle segments with speaker identifiers |
| `DIARIZATION_MODEL` | `english` | WeSpeaker model to use for diarization (e.g. `english`, `chinese`) |

### Subtitle Output

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBTITLE_LANGUAGE_NAME` | `''` | Language code used in the subtitle filename |
| `SUBTITLE_LANGUAGE_NAMING_TYPE` | `ISO_639_2_B` | Naming format: `ISO_639_1`, `ISO_639_2_T`, `ISO_639_2_B`, `NAME`, or `NATIVE` |
| `WORD_LEVEL_HIGHLIGHT` | `False` | Highlight each word as it's spoken |
| `CUSTOM_REGROUP` | `default` | Segment regrouping rules (`default` for stable-ts defaults) |
| `MIN_SUBTITLE_DURATION` | `0` | Minimum seconds a subtitle stays on screen (0 = disabled). Extends short segments so fast speech doesn't flash by. Try `1.5` to start. |
| `LRC_FOR_AUDIO_FILES` | `True` | Generate `.lrc` instead of `.srt` for audio files |
| `APPEND` | `False` | Append "Transcribed by whisperAI..." to subtitles |
| `FILTER_SUBTITLES` | `False` | Remove hallucinated phrases (e.g. "Thanks for watching", URLs) and gibberish segments from Whisper output before writing subtitles |
| `SHOW_IN_SUBNAME_SUBGEN` | `True` | Add "subgen" to subtitle filename |
| `SHOW_IN_SUBNAME_MODEL` | `True` | Add model name to subtitle filename |

### Skip / Filter Logic

| Variable | Default | Description |
|----------|---------|-------------|
| `PROCESS_ADDED_MEDIA` | `True` | Generate subtitles when media is added |
| `PROCESS_MEDIA_ON_PLAY` | `True` | Generate subtitles when media is played |
| `SKIP_IF_INTERNAL_SUBTITLES_LANGUAGE` | `''` | Skip if internal subs exist for this language code (empty = disabled) |
| `SKIP_IF_EXTERNAL_SUBTITLES_EXIST` | `False` | Skip if external `.srt` with matching language exists |
| `SKIP_IF_TARGET_SUBTITLES_EXIST` | `True` | Skip if any subtitle in target language exists |
| `SKIP_IF_AUDIO_LANGUAGES` | `''` | Pipe-separated language codes; skip if audio track matches (legacy: `SKIP_IF_AUDIO_TRACK_IS`) |
| `SKIP_ONLY_SUBGEN_SUBTITLES` | `False` | Only skip if existing subtitle has "subgen" in the name |
| `SKIP_UNKNOWN_LANGUAGE` | `False` | Skip if file has no known audio language |
| `SKIP_IF_NO_LANGUAGE_BUT_SUBTITLES_EXIST` | `False` | Skip if no audio language tag but subtitles exist |
| `SKIP_SUBTITLE_LANGUAGES` | `''` | Pipe-separated language codes to never generate for (e.g. `eng\|deu`) |
| `PREFERRED_AUDIO_LANGUAGES` | `eng` | Pipe-separated list of preferred audio languages (in order of preference) |
| `LIMIT_TO_PREFERRED_AUDIO_LANGUAGE` | `False` | Only transcribe files that have an audio track matching `PREFERRED_AUDIO_LANGUAGES` |

### Media Server

| Variable | Default | Description |
|----------|---------|-------------|
| `PLEX_SERVER` | `http://plex:32400` | Plex server address |
| `PLEX_TOKEN` | `token here` | Plex auth token ([how to find](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)) |
| `PLEX_QUEUE_NEXT_EPISODE` | `False` | Auto-queue next episode |
| `PLEX_QUEUE_SEASON` | `False` | Auto-queue rest of season |
| `PLEX_QUEUE_SERIES` | `False` | Auto-queue entire series |
| `JELLYFIN_SERVER` | `http://jellyfin:8096` | Jellyfin server address |
| `JELLYFIN_TOKEN` | `token here` | Jellyfin API token |

### Path Mapping & Folders

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PATH_MAPPING` | `False` | Enable path translation between media server and subgen |
| `PATH_MAPPING_FROM` | `/tv` | Path as seen by the media server |
| `PATH_MAPPING_TO` | `/Volumes/TV` | Equivalent path as seen by subgen |
| `TRANSCRIBE_FOLDERS` | `''` | Pipe-separated folders to scan for existing media (e.g. `/tv\|/movies`) |
| `MONITOR` | `False` | Watch `TRANSCRIBE_FOLDERS` for real-time changes |

### Resource Management

| Variable | Default | Description |
|----------|---------|-------------|
| `CLEAR_VRAM_ON_COMPLETE` | `True` | Unload model when queue is empty to free (V)RAM |
| `MODEL_CLEANUP_DELAY` | `30` | Seconds to wait before unloading the model |
| `ASR_TIMEOUT` | `18000` | Seconds (default 5 hours) before killing an ASR task |

### Rootless Container

| Variable | Default | Description |
|----------|---------|-------------|
| `PUID` | `99` | User ID to run as ([reference](https://docs.linuxserver.io/images/docker-bazarr/#user-group-identifiers)) |
| `PGID` | `100` | Group ID to run as |

---

## Models

| Model | Notes |
|-------|-------|
| `tiny`, `tiny.en` | Fastest, lowest accuracy |
| `base`, `base.en` | |
| `small`, `small.en` | |
| `medium`, `medium.en` | Good balance of speed and accuracy (default) |
| `large-v1`, `large-v2`, `large-v3` | Best accuracy, slowest |
| `large-v3-turbo` | Faster large-v3 variant |
| `distil-small.en`, `distil-medium.en` | Distilled English-only models |
| `distil-large-v2`, `distil-large-v3`, `distil-large-v3.5` | Distilled multilingual models |

`.en` models are English-only and generally faster/more accurate for English content.

---

## Supported Languages

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

---

## Unraid

See the [community guide](https://github.com/McCloudS/subgen/discussions/137) for installation steps with screenshots.

---

## Credits

- [WhisperJAV](https://github.com/meizhong986/WhisperJAV) — Scene detection, timestamp hardening, regroup tuning, and alignment sentinel strategies ported from this project
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [stable-ts](https://github.com/jianfch/stable-ts)
- [Whisper ASR Webservice](https://github.com/ahmetoner/whisper-asr-webservice)
- [ffmpeg](https://ffmpeg.org/)

---

*Originally created by [McCloudS](https://github.com/McCloudS/subgen). This fork is maintained at [regix1/subgen](https://github.com/regix1/subgen).*
