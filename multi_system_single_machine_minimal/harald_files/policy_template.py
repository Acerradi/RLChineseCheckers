from __future__ import annotations

from typing import Any, Dict, Tuple

from policies import BasePolicy


class MyPolicy(BasePolicy):
    """Replace this with your learned model later.

    The important part is that training and deployment both call the same
    `select_action(observation)` method and expect the same return type.
    """

    def __init__(self, model: Any = None):
        self.model = model

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        legal_moves = observation["legal_moves"]

        # TODO:
        # 1. Encode observation["state"] into model input
        # 2. Score each legal (pin_id, to_index) action
        # 3. Return the selected action
        
        # Placeholder fallback: first legal action.
        for pin_id, moves in legal_moves.items():
            if moves:
                return int(pin_id), int(moves[0])

        raise RuntimeError("No legal moves available")
