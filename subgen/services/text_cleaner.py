"""Pre-alignment text cleaner for ASR output.

Cleans raw ASR transcription text BEFORE forced alignment / segment
reconstruction.  Catches repetition loops, character floods, and
whitespace artifacts that degrade subtitle quality.

Stages (executed in order):
    1. Phrase repetition reducer  — consecutive + recursive pattern collapse
    2. Character flood reducer    — single-char floods
    3. Whitespace normalizer      — collapse whitespace, strip edges
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1: Phrase Repetition Reducer
# ---------------------------------------------------------------------------

# Maximum consecutive identical phrases to keep.
_MAX_CONSECUTIVE_PHRASES: int = 2

# Recursive pattern collapse: minimum repetitions to trigger.
_RECURSIVE_THRESH: int = 10
_RECURSIVE_MAX_PATTERN_LEN: int = 20


def _reduce_phrase_repetitions(text: str) -> str:
    """Reduce consecutive identical 2-8 character phrases.

    "go go go go go" kept as "go go" (keep 2).
    """
    max_keep = _MAX_CONSECUTIVE_PHRASES
    if max_keep < 1:
        max_keep = 1

    pattern = re.compile(
        r'([a-zA-Z0-9]{2,8})\1{' + str(max_keep) + r',}'
    )

    def _replace(m: re.Match) -> str:
        return m.group(1) * max_keep

    return pattern.sub(_replace, text)


def _fix_pattern_repeats(
    text: str, thresh: int = _RECURSIVE_THRESH, max_len: int = _RECURSIVE_MAX_PATTERN_LEN,
) -> str:
    """Recursive pattern repetition collapse (language-agnostic).

    Scans for any substring of length 1..max_len that repeats >= thresh times
    consecutively, collapses to a single occurrence.
    """
    if not text:
        return text

    result_parts: list[str] = []
    pos = 0
    length = len(text)

    while pos < length:
        best_pattern_len = 0
        best_repeat_count = 0

        for plen in range(1, min(max_len, length - pos) + 1):
            pattern = text[pos:pos + plen]
            count = 1
            check_pos = pos + plen
            while check_pos + plen <= length and text[check_pos:check_pos + plen] == pattern:
                count += 1
                check_pos += plen

            if count >= thresh and count > best_repeat_count:
                best_pattern_len = plen
                best_repeat_count = count

        if best_repeat_count >= thresh:
            result_parts.append(text[pos:pos + best_pattern_len])
            pos += best_pattern_len * best_repeat_count
        else:
            result_parts.append(text[pos])
            pos += 1

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Stage 2: Character Flood Reducer
# ---------------------------------------------------------------------------

_MAX_CONSECUTIVE_CHARS: int = 2


def _reduce_char_floods(text: str) -> str:
    """Reduce single-character floods.

    "aaaaaaa" -> "aa", "hhhhhh" -> "hh".
    """
    max_chars = _MAX_CONSECUTIVE_CHARS

    flood_pat = re.compile(
        r'([a-zA-Z])\1{' + str(max_chars) + r',}'
    )

    def _replace_flood(m: re.Match) -> str:
        return m.group(1) * max_chars

    return flood_pat.sub(_replace_flood, text)


# ---------------------------------------------------------------------------
# Stage 3: Whitespace Normalizer
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and blank lines, strip edges."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_asr_text(text: str) -> str:
    """Clean raw ASR text through all stages.

    Returns the cleaned text.  Designed to run BEFORE forced alignment
    or segment reconstruction so that repetition artifacts don't pollute
    the timestamp merge.
    """
    if not text or not text.strip():
        return text

    original = text

    # Stage 1a: Consecutive phrase reduction
    text = _reduce_phrase_repetitions(text)

    # Stage 1b: Recursive pattern collapse
    text = _fix_pattern_repeats(text)

    # Stage 2: Character flood reduction
    text = _reduce_char_floods(text)

    # Stage 3: Whitespace normalization
    text = _normalize_whitespace(text)

    if text != original:
        logger.debug(
            "Text cleaner: %d -> %d chars",
            len(original), len(text),
        )

    return text
