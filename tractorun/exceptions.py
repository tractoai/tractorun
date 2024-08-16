class BaseTractorunException(Exception):
    pass


class StderrReaderException(BaseTractorunException):
    pass


class TractorunConfigError(BaseTractorunException):
    pass


class TractorunInvalidConfiguration(BaseTractorunException):
    pass
