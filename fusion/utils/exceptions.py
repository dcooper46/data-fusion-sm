"""
The :mod:`fusion.util.exceptions` module includes all custom warnings and error
classes used across fusion.
"""

__all__ = ['NotFittedError',
           'FitFailedWarning',
           'UndefinedMetricWarning']


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if fusion estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling.
    """


class DataFormError(ValueError, AttributeError):
    """
    Exception raised when data format/shape is not what's expected
    """


class FitFailedWarning(RuntimeWarning):
    """
    Warning class used if there is an error while fitting the fusion estimator.
    """


class UndefinedMetricWarning(UserWarning):
    """
    Warning used when the metric is invalid
    """
