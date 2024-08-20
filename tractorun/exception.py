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
