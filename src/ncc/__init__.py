"""Neural Conformal Control (NCC): learned prediction intervals for time series forecasts.

The core (this package outside the `experiments` subpackage) uses numpy and torch only and does no
file I/O. The `experiments` subpackage reproduces the paper's experiments from files.
"""
from ncc.config import NCCConfig

__all__ = ["NCCConfig"]
