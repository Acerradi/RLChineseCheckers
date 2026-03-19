"""Chinese Checkers training/deployment scaffolding.

This package gives you a shared game core, a fast in-process environment
wrapper, and a socket-compatible bot client wrapper.
"""

from .core import GameCore, PlayerState, make_observation
from .env import ChineseCheckersEnv
from .policies import BasePolicy, RandomPolicy

__all__ = [
    "GameCore",
    "PlayerState",
    "make_observation",
    "ChineseCheckersEnv",
    "BasePolicy",
    "RandomPolicy",
]
