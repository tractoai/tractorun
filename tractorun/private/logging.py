import logging
import os

from yt.logger import SimpleColorizedStreamHandler

from tractorun.exception import TractorunConfigurationError


def setup_logging() -> str | None:
    log_level = os.environ.get("YT_LOG_LEVEL")
    if log_level:
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


def get_log_level_id(level: int | str | None) -> int | None:
    if isinstance(level, int) or level is None:
        return level
    level_id = logging.getLevelName(level.upper())
    if not isinstance(level_id, int):
        raise TractorunConfigurationError(
            f"Value in env var should contain standard logging log level, not {level}",
        )
    return level_id
