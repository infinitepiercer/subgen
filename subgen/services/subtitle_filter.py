"""Post-transcription subtitle filter.

Removes hallucinated phrases, gibberish, and junk segments that Whisper
sometimes generates during silence.  Enabled via ``FILTER_SUBTITLES=true``.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known hallucination phrases (case-insensitive substring match)
# ---------------------------------------------------------------------------

_HALLUCINATION_SUBSTRINGS: list[str] = [
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "hit the bell",
    "check the description",
    "www.",
    "http",
    ".com",
    ".org",
    ".net",
    "subtitles by",
    "translated by",
    "transcribed by",
    "captions by",
    "follow me on",
    "join my",
    "link in the description",
]

# Short phrases that are hallucinations only when the *entire* segment text
# matches (case-insensitive, after stripping whitespace/punctuation).
_EXACT_GHOST_PHRASES: set[str] = {
    "you",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "so",
    "yeah",
    "okay",
    "ok",
    "hmm",
    "uh",
    "um",
    "ah",
}

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Word repeated 3+ times in a row  (e.g. "the the the")
_REPETITION_RE = re.compile(r'(\b\w+\b)(?:\s+\1){2,}', re.IGNORECASE)

# Everything that is NOT a letter or digit (used to strip for "real word" check)
_NON_ALNUM_RE = re.compile(r'[^a-zA-Z0-9]')

# Everything that is NOT a letter (used for ghost phrase comparison —
# stray digits like "Bye.6" should still match "bye")
_NON_ALPHA_RE = re.compile(r'[^a-zA-Z]')

# Single character repeated (e.g. "aaaaaa", "......")
_SINGLE_CHAR_REPEAT_RE = re.compile(r'^(.)\1+$')

# Metadata line injected by appendLine() — must never be filtered
_METADATA_MARKER = "transcribed by whisperai with faster-whisper"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_segments(result) -> int:
    """Remove hallucinated / junk segments from a transcription result.

    Iterates segments in reverse so deletion indices stay valid.

    Args:
        result: The stable-ts ``WhisperResult`` (has a ``.segments`` list).

    Returns:
        Number of segments removed.
    """
    segments = result.segments
    removed = 0

    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        text: str = seg.text.strip()

        reason = _check_segment(text)
        if reason is not None:
            logger.debug(
                "Filtered segment #%d [%.2f-%.2f] (%s): %r",
                i, seg.start, seg.end, reason, text,
            )
            del segments[i]
            removed += 1

    if removed:
        logger.info("Filtered %d hallucination/junk segment(s)", removed)

    return removed


# ---------------------------------------------------------------------------
# Internal checks — return a reason string or None
# ---------------------------------------------------------------------------


def _check_segment(text: str) -> str | None:
    """Return a reason string if *text* should be filtered, else ``None``."""

    # --- Safety: never filter the metadata line ---
    if _METADATA_MARKER in text.lower():
        return None

    # --- Gibberish / symbol detection ---

    stripped = _NON_ALNUM_RE.sub('', text)

    # Too short (fewer than 2 real characters)
    if len(stripped) < 2:
        return "too_short"

    # No real words — entirely punctuation / symbols / whitespace
    if not stripped:
        return "no_real_words"

    # Single repeated character (e.g. "aaaaaa")
    if _SINGLE_CHAR_REPEAT_RE.match(stripped):
        return "single_char_repeat"

    # --- Hallucination detection ---

    text_lower = text.lower()

    # Known hallucination substrings
    for phrase in _HALLUCINATION_SUBSTRINGS:
        if phrase in text_lower:
            return f"hallucination_phrase:{phrase}"

    # Exact ghost phrases (entire segment is just a short filler word)
    # Strip digits too — stray numbers like "Bye.6" should still match "bye"
    text_clean = _NON_ALPHA_RE.sub('', text_lower)
    if text_clean in _EXACT_GHOST_PHRASES:
        return f"ghost_phrase:{text_clean}"

    # Excessive repetition (same word 3+ times)
    if _REPETITION_RE.search(text):
        return "excessive_repetition"

    return None
