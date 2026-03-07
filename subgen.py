# Subgen - subtitle generator entry point
import os
import logging


def main():
    from subgen import subgen_version
    from subgen.config import (
        whisper_model,
        whisper_threads,
        concurrent_transcriptions,
        transcribe_device,
        webhookport,
        reload_script_on_change,
        docker_status,
    )

    logging.info(f"Subgen v{subgen_version}")
    logging.info(
        f"Transcription: Threads: {whisper_threads}, "
        f"Concurrent: {concurrent_transcriptions}"
    )
    logging.info(
        f"Device: {transcribe_device}, Model: {whisper_model} ({docker_status})"
    )

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import uvicorn

    uvicorn.run(
        "subgen.app:app",
        host="0.0.0.0",
        port=int(webhookport),
        reload=reload_script_on_change,
        use_colors=True,
    )


if __name__ == "__main__":
    main()
