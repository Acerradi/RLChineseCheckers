import random


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def choose_action(self, observation):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def choose_action(self, observation):
        legal_moves = observation["legal_moves"]
        if not legal_moves:
            raise RuntimeError("No legal moves available.")
        move = random.choice(legal_moves)
        return {
            "pin_id": move["pin_id"],
            "to": move["to"],
        }