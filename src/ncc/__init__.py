"""Neural Conformal Control (NCC): learned prediction intervals for time series forecasts.

The core (this package outside the `experiments` subpackage) uses numpy and torch only and does no
file I/O. `NCC` is the method as a stateful object, called once per forecast origin with `NCC.step`.
The `experiments` subpackage reproduces the paper's experiments from files.
"""
from ncc.config import NCCConfig
from ncc.online import NCC, Feedback

__all__ = ["NCC", "NCCConfig", "Feedback"]
