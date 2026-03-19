from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .core import Action, GameCore, make_observation


@dataclass
class StepResult:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ChineseCheckersEnv:
    """Small RL-friendly wrapper around the shared game core.

    This is intentionally simple and not tied to any RL library. It lets you
    train with an in-process environment while preserving the live action/observation
    contract needed by the socket client.
    """

    def __init__(self, *,num_players: int = 6, player_names: Optional[List[str]] = None, per_move_penalty: float = 0.0,
                 win_reward: float = 1.0, draw_reward: float = 0.0, illegal_move_reward: float = -1.0,
                 shaping_from_score_delta: bool = False, core_kwargs: Optional[Dict[str, Any]] = None):
        
        if not (2 <= num_players <= 6):
            raise ValueError("num_players must be between 2 and 6")
        self.num_players = num_players
        self.player_names = player_names or [f"agent_{i}" for i in range(num_players)]
        self.per_move_penalty = per_move_penalty
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.illegal_move_reward = illegal_move_reward
        self.shaping_from_score_delta = shaping_from_score_delta
        self.core_kwargs = core_kwargs or {}

        self.game: Optional[GameCore] = None
        self.colour_to_player_id: Dict[str, str] = {}
        self.last_scores: Dict[str, float] = {}

    def reset(self) -> Dict[str, Dict[str, Any]]:
        self.game = GameCore(**self.core_kwargs)
        self.colour_to_player_id = {}

        for name in self.player_names[: self.num_players]:
            player = self.game.add_player(name)
            self.colour_to_player_id[player.colour] = player.player_id

        self.game.auto_start()
        self.game.compute_scores()
        self.last_scores = {
            p.colour: (self.game.scores.get(p.player_id, {}) or {}).get("final_score", 0.0)
            for p in self.game.players}
        return {colour: make_observation(self.game, colour) for colour in self.turn_order}

    @property
    def turn_order(self) -> List[str]:
        if self.game is None:
            return []
        return list(self.game.turn_order)

    @property
    def current_turn_colour(self) -> Optional[str]:
        if self.game is None:
            return None
        return self.game.current_turn_colour()

    def legal_moves(self, colour: str) -> Dict[int, List[int]]:
        if self.game is None:
            raise RuntimeError("Environment not reset")
        return self.game.get_legal_moves_for_colour(colour)

    def observe(self, colour: str) -> Dict[str, Any]:
        if self.game is None:
            raise RuntimeError("Environment not reset")
        return make_observation(self.game, colour)

    def step(self, colour: str, action: Action) -> StepResult:
        if self.game is None:
            raise RuntimeError("Environment not reset")

        pin_id, to_index = action
        player_id = self.colour_to_player_id[colour]
        before_score = self.last_scores.get(colour, 0.0)
        result = self.game.apply_move(player_id=player_id, pin_id=pin_id, to_index=to_index)

        if not result.get("ok"):
            return StepResult(
                observation=self.observe(colour),
                reward=self.illegal_move_reward,
                done=self.game.status == "FINISHED",
                info={"ok": False, "error": result.get("error")})

        reward = -self.per_move_penalty
        self.game.compute_scores()
        after_score = (self.game.scores.get(player_id, {}) or {}).get("final_score", 0.0)
        self.last_scores[colour] = after_score

        if self.shaping_from_score_delta:
            reward += after_score - before_score

        player = self.game.get_player(player_id)
        if player is not None and player.status == "WIN":
            reward += self.win_reward
        elif player is not None and player.status == "DRAW":
            reward += self.draw_reward

        return StepResult(
            observation=self.observe(colour),
            reward=reward,
            done=self.game.status == "FINISHED",
            info={
                "ok": True,
                "status": result.get("status"),
                "message": result.get("msg"),
                "state": self.game.to_public_state(),
            },
        )

    def run_policies(self, policy_by_colour: Mapping[str, Any], max_moves: int = 5000) -> Dict[str, Any]:
        if self.game is None:
            self.reset()

        for _ in range(max_moves):
            assert self.game is not None
            if self.game.status == "FINISHED":
                break
            colour = self.game.current_turn_colour()
            if colour is None:
                break
            policy = policy_by_colour[colour]
            obs = self.observe(colour)
            action = policy.select_action(obs)
            self.step(colour, action)

        assert self.game is not None
        self.game.compute_scores()
        return {
            "status": self.game.status,
            "state": self.game.to_public_state(),
            "scores": {
                p.colour: self.game.scores.get(p.player_id, {})
                for p in self.game.players
            },
            "history": list(self.game.history),
        }
