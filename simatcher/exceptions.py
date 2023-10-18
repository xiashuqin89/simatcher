class Error(Exception):
    pass


class ApiNotAvailable(Error):
    pass


class ApiError(Error, RuntimeError):
    pass


class TokenNotAvailable(Error):
    pass


class HttpFailed(ApiError):
    """HTTP status code is not 2xx."""

    def __init__(self, status_code):
        self.status_code = status_code


class ActionFailed(ApiError):
    """
    Action failed to execute.

    >>> except ActionFailed as e:
    >>>     if e.retcode > 0:
    >>>         pass  # error code returned by HTTP API
    >>>     elif e.retcode < 0:
    >>>         pass  # error code returned by CoolQ
    """

    def __init__(self, retcode, message=''):
        self.retcode = retcode
        self.message = message


class NetworkError(Error, IOError):
    pass


class InvalidProjectError(Error):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class MissingArgumentError(ValueError):
    def __init__(self, message: str):
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class UnsupportedModelError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class PipelineRunningAbnormalError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class InvalidRecipeException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class UnsupportedLanguageError(Exception):
    def __init__(self, component, language):
        self.component = component
        self.language = language

        super(UnsupportedLanguageError, self).__init__(component, language)

    def __str__(self):
        return "component {} does not support language {}".format(
            self.component, self.language
        )
