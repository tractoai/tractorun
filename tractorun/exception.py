__all__ = [
    "BaseTractorunException",
    "StderrReaderException",
    "TractorunConfigError",
    "TractorunWandbError",
    "TractorunConfigurationError",
]


class BaseTractorunException(Exception):
    pass


class StderrReaderException(BaseTractorunException):
    pass


class TractorunConfigError(BaseTractorunException):
    pass


class TractorunConfigurationError(BaseTractorunException):
    pass


class TractorunWandbError(BaseTractorunException):
    pass
