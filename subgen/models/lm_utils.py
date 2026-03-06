"""Build a BPE-tokenized n-gram language model for Parakeet NGPU-LM fusion.

On first call, downloads the LibriSpeech normalized LM training text,
tokenizes it with the Parakeet model's BPE tokenizer, and builds a
KenLM n-gram model.  The result is cached so subsequent loads are instant.

Requires ``lmplz`` from KenLM to be on ``$PATH``
(installed via the Dockerfile).
"""

import gzip
import logging
import os
import shutil
import subprocess
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_LIBRISPEECH_LM_URL = (
    "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
)
_DEFAULT_NGRAM_ORDER = 3
_MAX_LINES = 500_000  # Cap input to keep lmplz memory usage reasonable
# NeMo encodes BPE token IDs as Unicode characters offset by this value.
# This must match NeMo's internal DEFAULT_TOKEN_OFFSET in kenlm_utils.py.
_TOKEN_OFFSET = 100


def _kenlm_available() -> bool:
    """Return True if KenLM lmplz binary is installed."""
    return shutil.which("lmplz") is not None


def _download_with_progress(url: str, dest: str) -> None:
    """Download a file with logging progress."""
    logger.info("Downloading LM training text from %s", url)
    logger.info("This is a one-time download (~800 MB). The built n-gram will be cached.")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            if block_num % 500 == 0:
                logger.info("  Download progress: %d%%", pct)

    urllib.request.urlretrieve(url, dest, reporthook=_report)
    logger.info("Download complete: %s", dest)


def _tokenize_and_build(
    tokenizer,
    text_gz_path: str,
    arpa_path: str,
    ngram_order: int,
) -> None:
    """Tokenize text with BPE tokenizer and pipe to KenLM lmplz."""
    logger.info(
        "Building %d-gram BPE language model (this may take several minutes on first run)...",
        ngram_order,
    )

    # lmplz reads tokenized text from stdin, one sentence per line.
    # Each "word" is a Unicode-encoded BPE token.
    # --prune needs exactly ngram_order values (one threshold per n-gram level).
    # Heavier pruning: keep all unigrams, prune bigrams <1, trigrams <2
    prune_values = (["0"] + [str(i) for i in range(1, ngram_order)])[:ngram_order]
    lmplz_cmd = [
        "lmplz",
        "-o", str(ngram_order),
        "--arpa", arpa_path,
        "--prune", *prune_values,
        "--discount_fallback",
        "-S", "1G",
    ]

    proc = subprocess.Popen(
        lmplz_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    lines_processed = 0
    try:
        with gzip.open(text_gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if lines_processed >= _MAX_LINES:
                    break
                line = line.strip().lower()
                if not line:
                    continue
                # Tokenize to BPE token IDs, then encode as Unicode
                # characters (chr(id + _TOKEN_OFFSET)) to match NeMo's
                # internal n-gram representation.
                try:
                    token_ids = tokenizer.text_to_ids(line)
                except Exception:
                    continue
                if not token_ids:
                    continue
                encoded_line = " ".join(
                    chr(tid + _TOKEN_OFFSET) for tid in token_ids
                ) + "\n"
                try:
                    proc.stdin.write(encoded_line.encode("utf-8"))
                except (BrokenPipeError, OSError, ValueError):
                    break
                lines_processed += 1
                if lines_processed % 500_000 == 0:
                    logger.info("  Tokenized %d lines...", lines_processed)
    finally:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass  # Pipe already dead (lmplz crashed)

    # stdin is already closed above; set to None so communicate() won't
    # try to flush/close it again (which raises "flush of closed file").
    proc.stdin = None
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"lmplz failed (exit {proc.returncode}): {err_msg}")

    logger.info(
        "Built %d-gram ARPA from %d lines: %s",
        ngram_order, lines_processed, arpa_path,
    )


def ensure_ngram_lm(
    tokenizer,
    cache_dir: str,
    ngram_order: int = _DEFAULT_NGRAM_ORDER,
) -> Optional[str]:
    """Ensure a BPE n-gram LM exists, building it on first call.

    Returns the path to the ARPA file, or None if building failed.
    """
    if not _kenlm_available():
        logger.warning(
            "KenLM lmplz binary not found. "
            "Language model fusion disabled. Accuracy may be reduced for "
            "homophones and accented speech."
        )
        return None

    arpa_path = os.path.join(cache_dir, f"parakeet_bpe_{ngram_order}gram.arpa")

    if os.path.exists(arpa_path):
        logger.info("Using cached n-gram LM: %s", arpa_path)
        return arpa_path

    os.makedirs(cache_dir, exist_ok=True)

    # Download LibriSpeech text if not cached
    text_gz_path = os.path.join(cache_dir, "librispeech-lm-norm.txt.gz")
    if not os.path.exists(text_gz_path):
        try:
            _download_with_progress(_LIBRISPEECH_LM_URL, text_gz_path)
        except Exception as exc:
            logger.error("Failed to download LM training text: %s", exc)
            return None

    # Build the n-gram
    try:
        _tokenize_and_build(tokenizer, text_gz_path, arpa_path, ngram_order)
    except Exception as exc:
        logger.error("Failed to build n-gram LM: %s", exc)
        # Clean up partial file
        if os.path.exists(arpa_path):
            os.unlink(arpa_path)
        return None

    return arpa_path
