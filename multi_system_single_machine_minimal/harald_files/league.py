from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from environment import ChineseCheckersEnv


@dataclass
class LeagueEntry:
    name: str
    policy: Any
    games_played: int = 0
    wins: int = 0
    total_score: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)


class SelfPlayLeague:
    """Small utility for running many local matches between policy objects.

    This is not a trainer. It is a scheduling/evaluation helper so you can keep
    several policies improving at similar pace and later plug the best one into
    the socket client.
    """

    def __init__(self, num_players: int = 6, env_kwargs: Optional[Dict[str, Any]] = None):
        self.num_players = num_players
        self.env_kwargs = env_kwargs or {}
        self.entries: Dict[str, LeagueEntry] = {}

    def add_policy(self, name: str, policy: Any) -> None:
        self.entries[name] = LeagueEntry(name=name, policy=policy)

    def run_round_robin_once(self) -> List[Dict[str, Any]]:
        names = list(self.entries)
        if len(names) < self.num_players:
            raise ValueError(f"Need at least {self.num_players} policies, got {len(names)}")

        selected = names[: self.num_players]
        env = ChineseCheckersEnv(
            num_players=self.num_players,
            player_names=selected,
            **self.env_kwargs,
        )
        env.reset()

        policies_by_colour = {
            colour: self.entries[player.name].policy
            for colour in env.turn_order
            for player in env.game.players
            if player.colour == colour
        }
        result = env.run_policies(policies_by_colour)

        for player in env.game.players:
            entry = self.entries[player.name]
            entry.games_played += 1
            score = result["scores"][player.colour].get("final_score", 0.0)
            entry.total_score += score
            if player.status == "WIN":
                entry.wins += 1
            entry.history.append({
                "status": player.status,
                "score": score,
                "colour": player.colour,
            })

        return env.game.history
