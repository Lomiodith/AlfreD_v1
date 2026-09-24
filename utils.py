import glob
import logging
import os
import tempfile
import threading
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL

logger = logging.getLogger(__name__)

# Set by the tray's Stop; every blocking stage polls it so a stop is immediate.
stop_event = threading.Event()

NOISY_LIBRARIES = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "ddgs",
    "primp",
)


class _ConsoleFilter(logging.Filter):
    """Drop records logged with extra={"file_only": True}."""

    def filter(self, record):
        return not getattr(record, "file_only", False)


def setup_logging():
    root = logging.getLogger()
    if root.handlers:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    root.setLevel(LOG_LEVEL)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    console.addFilter(_ConsoleFilter())
    root.addHandler(console)

    log_file = RotatingFileHandler(
        os.path.join(LOG_DIR, "alfred.log"),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(log_file)

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def clean_temp_audio():
    temp_dir = tempfile.gettempdir()
    removed_count = 0

    for f in glob.glob(os.path.join(temp_dir, "*.wav")):
        try:
            os.remove(f)
            removed_count += 1
        except Exception as e:
            logger.warning(f"Could not remove {f}: {e}")

    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} temporary .wav file(s).")
