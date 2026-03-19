from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple


class BasePolicy(ABC):
    """Interface shared by training and live socket play."""

    @abstractmethod
    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        legal_moves = observation["legal_moves"]
        movable = [(int(pid), moves) for pid, moves in legal_moves.items() if moves]
        if not movable:
            raise RuntimeError("No legal moves available")
        pin_id, moves = self.rng.choice(movable)
        return pin_id, self.rng.choice(list(moves))


class FunctionPolicy(BasePolicy):
    def __init__(self, fn):
        self.fn = fn

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        return self.fn(observation)
