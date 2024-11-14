"""Custom exceptions for the nessai package."""


class RNGError(Exception):
    """Base class for exceptions related to random number generation."""

    pass


class RNGNotSetError(RNGError):
    """Exception raised when the random number generator has not been set"""

    def __init__(
        self, msg="Random number generator has not been set", *args, **kwargs
    ):
        super().__init__(msg, *args, **kwargs)


class RNGSetError(RNGError):
    """Exception raised when the random number generator has already been set"""

    def __init__(
        self,
        msg="Random number generator has already been set",
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)
