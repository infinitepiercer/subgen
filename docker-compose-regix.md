```yaml
  subgen_eng:
    container_name: subgen_eng
    image: ghcr.io/regix1/subgen:latest
    restart: unless-stopped
    ports:
      - "9000:9000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - /srv/subgen/models:/subgen/models:rw
      - /mnt/shared:/mnt/shared:rw
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      TRANSCRIBE_DEVICE: cuda
      COMPUTE_TYPE: float16
      WHISPER_MODEL: large-v3-turbo
      WHISPER_THREADS: 6
      CONCURRENT_TRANSCRIPTIONS: 1
      MODEL_PATH: /subgen/models
      NORMALIZE_AUDIO: "true"
      FILTER_SUBTITLES: "true"
      SUBGEN_KWARGS: "{'vad': True, 'hallucination_silence_threshold': 2, 'condition_on_previous_text': False, 'no_speech_threshold': 0.2}"
      TRANSCRIBE_OR_TRANSLATE: transcribe_and_translate
      TRANSLATE_SOURCE_LANGUAGES: fr,es,de,it,pt,ja,ko,zh,ru
      DETECT_CONFIDENCE_THRESHOLD: "0.7"
      FORCE_DETECTED_LANGUAGE_TO: en
      SHOULD_WHISPER_DETECT_AUDIO_LANGUAGE: "false"
      DETECT_LANGUAGE_LENGTH: 120
      PREFERRED_AUDIO_LANGUAGES: eng|jpn|fre|spa|deu|ita|por|kor|zho|rus
      SUBTITLE_LANGUAGE_NAME: en
      SUBTITLE_LANGUAGE_NAMING_TYPE: ISO_639_1
      CUSTOM_REGROUP: default
      CLEAR_VRAM_ON_COMPLETE: "true"
      MODEL_CLEANUP_DELAY: 60
      WEBHOOK_PORT: 9000
      DEBUG: "false"
      # --- Speaker Diarization ---
      ENABLE_DIARIZATION: "false"
      DIARIZATION_MODEL: english
    networks:
      plex_network:
        ipv4_address: 172.20.0.4
```
