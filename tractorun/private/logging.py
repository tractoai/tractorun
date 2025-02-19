import logging
import os

from tractorun.exception import TractorunConfigurationError


def setup_logging() -> str | None:
    log_level = os.environ.get("YT_LOG_LEVEL")
    if log_level:
        logger = logging.getLogger("tractorun")
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
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
