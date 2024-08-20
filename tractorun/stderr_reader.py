import enum


__all__ = ["StderrMode"]


class StderrMode(str, enum.Enum):
    disabled = "disabled"
    primary = "primary"
