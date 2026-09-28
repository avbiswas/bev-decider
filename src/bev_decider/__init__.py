"""bev-decider: a 0.4B System One decision model with typed, calibrated answers."""

from .decider import DEFAULT_MODEL, Decider, load

__all__ = ["DEFAULT_MODEL", "Decider", "load"]
__version__ = "0.2.1"
