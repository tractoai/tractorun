__all__ = [
    "BaseTractorunException",
    "StderrReaderError",
    "TractorunConfigError",
    "TractorunWandbError",
    "TractorunConfigurationError",
    "DockerAuthDataError",
    "TractorunVersionMismatchError",
    "TractorunBootstrapError",
]


class BaseTractorunException(Exception):
    pass


class StderrReaderError(BaseTractorunException):
    pass


class TractorunBootstrapError(BaseTractorunException):
    pass


class TractorunConfigError(BaseTractorunException):
    pass


class TractorunVersionMismatchError(BaseTractorunException):
    pass


class TractorunConfigurationError(BaseTractorunException):
    pass


class TractorunWandbError(BaseTractorunException):
    pass


class DockerAuthDataError(BaseTractorunException):
    pass
