"""ASR text cleaner — pre-alignment and post-alignment stages.

Pre-alignment (clean_asr_text):
    Cleans raw ASR transcription text BEFORE forced alignment / segment
    reconstruction.  Catches repetition loops, character floods, and
    whitespace artifacts that degrade subtitle quality.

Post-alignment (clean_word_list):
    Operates on word dicts AFTER timestamp merge so that timestamps
    remain aligned.  Removes consecutive word repeats and per-word
    character floods without altering timing.

Stages (executed in order for pre-alignment):
    1. Phrase repetition reducer  — consecutive + recursive pattern collapse
    2. Character flood reducer    — single-char floods
    3. Whitespace normalizer      — collapse whitespace, strip edges
"""

import logging
import re
from typing import Any, Dict, List

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


# ---------------------------------------------------------------------------
# Word-level cleaning (post-alignment)
# ---------------------------------------------------------------------------

# Regex to strip punctuation for word comparison.
_STRIP_PUNCT: re.Pattern[str] = re.compile(r'[^\w\s]', re.UNICODE)

# Character flood pattern for per-word cleaning.
_WORD_FLOOD_PAT: re.Pattern[str] = re.compile(
    r'([a-zA-Z])\1{' + str(_MAX_CONSECUTIVE_CHARS) + r',}'
)


def _reduce_consecutive_word_repeats(
    word_dicts: List[Dict[str, Any]],
    max_keep: int = _MAX_CONSECUTIVE_PHRASES,
) -> List[Dict[str, Any]]:
    """Remove consecutive identical words beyond *max_keep*.

    Comparison is case-insensitive with punctuation stripped so that
    "Go," and "go" are treated as identical.
    """
    if not word_dicts or max_keep < 1:
        return word_dicts

    result: List[Dict[str, Any]] = []
    streak = 1

    for i, wd in enumerate(word_dicts):
        bare = _STRIP_PUNCT.sub("", wd["word"]).strip().lower()
        if i > 0:
            prev_bare = _STRIP_PUNCT.sub("", word_dicts[i - 1]["word"]).strip().lower()
            if bare and bare == prev_bare:
                streak += 1
            else:
                streak = 1
        if streak <= max_keep:
            result.append(wd)

    if len(result) < len(word_dicts):
        logger.debug(
            "Word repeat reduction: %d -> %d words",
            len(word_dicts), len(result),
        )
    return result


def _reduce_word_char_floods(
    word_dicts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply character flood reduction to each word's text in place.

    E.g. "yeeeeees" -> "yees".
    """
    def _replace_flood(m: re.Match) -> str:
        return m.group(1) * _MAX_CONSECUTIVE_CHARS

    for wd in word_dicts:
        original = wd["word"]
        cleaned = _WORD_FLOOD_PAT.sub(_replace_flood, original)
        if cleaned != original:
            wd["word"] = cleaned
    return word_dicts


def clean_word_list(word_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean a list of word dicts after timestamp merge.

    Operates on ``{'word': str, 'start': float, 'end': float}`` dicts
    so that timestamps remain aligned.

    Stages:
        1. Reduce consecutive identical word repeats (keep max 2).
        2. Reduce per-word character floods (e.g. "yeeeeees" -> "yees").
    """
    if not word_dicts:
        return word_dicts

    original_count = len(word_dicts)
    word_dicts = _reduce_consecutive_word_repeats(word_dicts)
    word_dicts = _reduce_word_char_floods(word_dicts)

    if len(word_dicts) != original_count:
        logger.info(
            "Word-level cleaner: %d -> %d words",
            original_count, len(word_dicts),
        )
    return word_dicts
