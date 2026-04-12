"""ASR text cleaner — post-alignment word-level cleaning.

Post-alignment (clean_word_list):
    Operates on word dicts AFTER timestamp merge so that timestamps
    remain aligned.  Removes consecutive word repeats and per-word
    character floods without altering timing.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum consecutive identical phrases/words to keep.
_MAX_CONSECUTIVE_PHRASES: int = 2

# Maximum consecutive identical characters to keep.
_MAX_CONSECUTIVE_CHARS: int = 2

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
