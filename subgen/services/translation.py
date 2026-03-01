"""Translation service: offline text translation for the two-pass transcribe+translate workflow.

In "transcribe_and_translate" mode the pipeline works as follows:
  Pass 1 -- Whisper transcribes the audio, producing accurate text and timing.
  Pass 2 -- This module translates non-English segments to English using
            *langdetect* for language detection and *argostranslate* for
            machine translation.

All third-party imports (langdetect, argostranslate) are lazy so that
existing ``transcribe`` and ``translate`` modes are unaffected when these
packages are not installed.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / package management
# ---------------------------------------------------------------------------


def ensure_translation_models(source_languages: List[str], model_path: str) -> None:
    """Download argostranslate language packages for *source_lang* -> English.

    Only downloads packages that are not already installed.  Safe to call
    multiple times (idempotent).

    Args:
        source_languages: ISO 639-1 language codes that may appear in the
            transcription (e.g. ``["fr", "es", "de"]``).
        model_path: Filesystem directory where argostranslate should cache
            its model files.  The ``ARGOS_MODELS_DIR`` environment variable
            is set **before** importing argostranslate so the library picks
            up the custom path.
    """
    # Set the model directory before any argostranslate import so the library
    # honours it from the start.
    os.environ["ARGOS_MODELS_DIR"] = model_path

    import argostranslate.package
    import argostranslate.translate

    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    installed_languages = {
        pkg.from_code
        for pkg in argostranslate.package.get_installed_packages()
        if pkg.to_code == "en"
    }

    for lang in source_languages:
        if lang == "en":
            continue

        if lang in installed_languages:
            logger.info(
                "Argos translation model %s -> en is already installed.",
                lang,
            )
            continue

        matching = [
            pkg
            for pkg in available_packages
            if pkg.from_code == lang and pkg.to_code == "en"
        ]

        if not matching:
            logger.warning(
                "No argostranslate package available for %s -> en. "
                "Segments in this language will be kept as-is.",
                lang,
            )
            continue

        package_to_install = matching[0]
        logger.info("Installing argostranslate package: %s -> en ...", lang)
        package_to_install.install()
        logger.info("Installed argostranslate package: %s -> en.", lang)

    # Log a summary of all installed pairs.
    installed_pairs = [
        f"{pkg.from_code} -> {pkg.to_code}"
        for pkg in argostranslate.package.get_installed_packages()
    ]
    logger.info(
        "Argos translation models available: %s",
        ", ".join(installed_pairs) if installed_pairs else "(none)",
    )


# ---------------------------------------------------------------------------
# Per-segment language detection
# ---------------------------------------------------------------------------


def detect_segment_language(text: str, confidence_threshold: float = 0.7) -> str:
    """Detect the language of a single text segment.

    Args:
        text: The segment text to classify.
        confidence_threshold: If the probability that the segment is English
            exceeds this value the function returns ``"en"`` immediately.

    Returns:
        An ISO 639-1 language code (e.g. ``"en"``, ``"fr"``).  Falls back to
        ``"en"`` for very short strings, low-confidence results, or on any
        detection error so that the original text is preserved.
    """
    # Very short text is unreliable for detection -- assume English.
    if len(text.strip()) < 8:
        return "en"

    try:
        from langdetect import detect_langs
    except ImportError:
        logger.warning(
            "langdetect is not installed; assuming English for all segments."
        )
        return "en"

    try:
        detected = detect_langs(text)
    except Exception:
        logger.debug("langdetect failed for text: %s", text[:40])
        return "en"

    if not detected:
        return "en"

    # Check whether English is among the candidates with sufficient confidence.
    for candidate in detected:
        if candidate.lang == "en" and candidate.prob >= confidence_threshold:
            return "en"

    # Return the highest-probability language.
    return detected[0].lang


# ---------------------------------------------------------------------------
# Single-text translation
# ---------------------------------------------------------------------------


def translate_text(text: str, source_lang: str, target_lang: str = "en") -> str:
    """Translate *text* from *source_lang* to *target_lang* using argostranslate.

    If translation fails for any reason the original text is returned
    unchanged (graceful fallback).

    Args:
        text: The text to translate.
        source_lang: ISO 639-1 code of the source language.
        target_lang: ISO 639-1 code of the target language (default ``"en"``).

    Returns:
        The translated string, or the original *text* on failure.
    """
    try:
        import argostranslate.translate
    except ImportError:
        logger.warning(
            "argostranslate is not installed; returning original text."
        )
        return text

    try:
        translated = argostranslate.translate.translate(text, source_lang, target_lang)
        logger.debug(
            "Translated (%s -> %s): '%s' -> '%s'",
            source_lang,
            target_lang,
            text[:60],
            translated[:60],
        )
        return translated
    except Exception as exc:
        logger.warning(
            "Translation failed (%s -> %s) for text '%s': %s",
            source_lang,
            target_lang,
            text[:40],
            exc,
        )
        return text


# ---------------------------------------------------------------------------
# Batch segment translation (main entry point)
# ---------------------------------------------------------------------------


def translate_segments(
    result: object,
    confidence_threshold: float = 0.7,
    debug: bool = False,
) -> int:
    """Translate non-English segments in a stable_whisper transcription result.

    This is the main entry point called after Whisper transcription (Pass 1).
    It iterates over every segment in *result*, detects the language of each
    segment's text, and -- if not English -- replaces it with an English
    translation.

    **Timestamps are never modified.**

    Args:
        result: A ``stable_whisper`` transcription result whose ``.segments``
            attribute yields objects with ``.text``, ``.start``, and ``.end``.
        confidence_threshold: Passed through to
            :func:`detect_segment_language`.
        debug: When ``True``, log detailed before/after information for every
            translated segment.

    Returns:
        The number of segments that were translated.
    """
    segments = list(result.segments)
    total = len(segments)

    # First pass: detect which segments need translation.
    segments_to_translate: list[tuple[object, str]] = []
    for segment in segments:
        lang = detect_segment_language(segment.text, confidence_threshold)
        if lang != "en":
            segments_to_translate.append((segment, lang))

    if not segments_to_translate:
        logger.info(
            "Pass 2: All %d segments detected as English -- nothing to translate.",
            total,
        )
        return 0

    logger.info(
        "Pass 2: Translating %d non-English segments to English...",
        len(segments_to_translate),
    )

    translated_count = 0
    for segment, source_lang in segments_to_translate:
        original_text = segment.text
        translated_text = translate_text(original_text, source_lang, "en")

        # Only count as translated when the text actually changed.
        if translated_text != original_text:
            segment.text = translated_text
            translated_count += 1

            if debug:
                logger.info(
                    "  [%s] %.1fs-%.1fs: '%s' -> '%s'",
                    source_lang,
                    segment.start,
                    segment.end,
                    original_text[:50],
                    translated_text[:50],
                )
        else:
            if debug:
                logger.debug(
                    "  [%s] %.1fs-%.1fs: translation unchanged for '%s'",
                    source_lang,
                    segment.start,
                    segment.end,
                    original_text[:50],
                )

    logger.info(
        "Pass 2 complete: %d of %d segments translated.",
        translated_count,
        total,
    )
    return translated_count
