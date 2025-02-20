import logging
import os

from yt.logger import SimpleColorizedStreamHandler


def setup_logging() -> str | None:
    log_level = os.environ.get("YT_LOG_LEVEL")
    if log_level:
        log_level = log_level.upper()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        tracto_logger = logging.getLogger("tractorun")
        tracto_logger.setLevel(log_level)
        if not tracto_logger.handlers:
            tracto_logger.addHandler(handler)
        yt_logger = logging.getLogger("Yt")
        if len(yt_logger.handlers) == 1 and isinstance(yt_logger.handlers[0], SimpleColorizedStreamHandler):
            yt_logger.handlers.clear()
            yt_logger.addHandler(handler)
    return log_level
