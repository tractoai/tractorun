class BaseTractorunException(Exception):
    pass


class TractorunConfigError(BaseTractorunException):
    pass


class TractorunInvalidConfiguration(BaseTractorunException):
    pass
