import enum


class StderrMode(str, enum.Enum):
    disabled = "disabled"
    primary = "primary"
