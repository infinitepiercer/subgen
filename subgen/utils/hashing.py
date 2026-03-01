import hashlib
from typing import Optional


def generate_audio_hash(
    audio_content: bytes,
    task: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """
    Generate a deterministic hash from audio content and optional parameters.

    Same audio + same task + same language = always same hash.
    This ensures duplicate requests are caught by the queue.

    Args:
        audio_content: Raw audio bytes from uploaded file
        task: Optional task type ('transcribe' or 'translate')
        language: Optional target language code

    Returns:
        SHA256 hash (first 16 chars for brevity in logs)
    """
    hash_input = audio_content

    # Include task and language for fine-grained deduplication
    if task:
        hash_input += task.encode('utf-8')
    if language:
        hash_input += language.encode('utf-8')

    full_hash = hashlib.sha256(hash_input).hexdigest()
    return full_hash[:16]  # Use first 16 chars for shorter IDs in logs
