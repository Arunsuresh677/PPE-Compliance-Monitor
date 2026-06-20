"""
src/detection/model.py — YOLO model download and loading.

Uses an atomic write pattern: download to a .tmp file, then os.replace().
A failed or interrupted download never leaves a corrupt weights file on disk.
"""

import logging
import os
import urllib.request

from ultralytics import YOLO

from src.config.settings import settings

log = logging.getLogger(__name__)


def download_model(url: str, dest: str) -> None:
    """Download model weights atomically. Raises RuntimeError on failure."""
    tmp = dest + ".tmp"
    try:
        log.info("Downloading model weights from %s …", url)
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        log.info("Model saved to %s", dest)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Model download failed: {exc}") from exc


def load_model(
    path: str | None = None,
    url: str | None = None,
) -> YOLO:
    """
    Ensure weights exist (downloading if necessary) and return a YOLO instance.

    If the weights file is corrupt, it is deleted and RuntimeError is raised
    so the caller can surface a clear message and the next run re-downloads.
    """
    path = path or settings.model_path
    url  = url  or settings.model_url

    if not os.path.exists(path):
        download_model(url, path)

    try:
        return YOLO(path)
    except Exception as exc:
        log.error("Corrupt weights file %s — deleting: %s", path, exc)
        os.remove(path)
        raise RuntimeError(
            f"Model file {path!r} was corrupt and has been deleted. "
            "Reload/restart to re-download."
        ) from exc
