"""Chinese Checkers training/deployment scaffolding.

This package gives you a shared game core, a fast in-process environment
wrapper, and a socket-compatible bot client wrapper.
"""

import os as _os, sys as _sys
# checkers_board.py and checkers_pins.py live in the sibling directory.
# Add it to sys.path so absolute imports inside core.py / policy_template.py work
# regardless of the caller's working directory.
_DEPS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "..", "multi system single machine minimal")
if _os.path.isdir(_DEPS_DIR) and _DEPS_DIR not in _sys.path:
    _sys.path.insert(0, _os.path.normpath(_DEPS_DIR))
del _os, _sys, _DEPS_DIR

from .core import GameCore, PlayerState, make_observation
from .environment import ChineseCheckersEnv
from .policies import BasePolicy, RandomPolicy

__all__ = [
    "GameCore",
    "PlayerState",
    "make_observation",
    "ChineseCheckersEnv",
    "BasePolicy",
    "RandomPolicy",
]
