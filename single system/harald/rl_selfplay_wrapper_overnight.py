from __future__ import annotations

import argparse
import contextlib
import io
import random
import signal
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from checkers_board import HexBoard
from checkers_pins import Pin


HALF_COLOURS_1 = ["red", "lawn green", "yellow"]
HALF_COLOURS_2 = ["blue", "gray0", "purple"]
ALL_COLOURS = HALF_COLOURS_1 + HALF_COLOURS_2


@dataclass
class Move:
    player: str
    pin_id: int
    from_axial: int
    to_axial: int


@dataclass
class StopController:
    stop_requested: bool = False
    stop_immediately: bool = False


class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, int]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def choose_action(self, observation: Dict[str, Any]) -> Dict[str, int]:
        legal_moves = observation["legal_moves"]
        if not legal_moves:
            raise RuntimeError("No legal moves available for current player.")
        move = random.choice(legal_moves)
        return {"pin_id": move["pin_id"], "to": move["to"]}


class ChineseCheckersSelfPlayEnv:
    def __init__(
        self,
        board_radius: int = 4,
        hole_radius: int = 16,
        spacing: int = 34,
        randomize_first_player: bool = False,
        max_turns: int = 500,
        allow_neutral_cells: bool = True,
        suppress_board_prints: bool = True,
    ):
        self.board_radius = board_radius
        self.hole_radius = hole_radius
        self.spacing = spacing
        self.randomize_first_player = randomize_first_player
        self.max_turns = max_turns
        self.allow_neutral_cells = allow_neutral_cells
        self.suppress_board_prints = suppress_board_prints

        self.board: Optional[HexBoard] = None
        self.board_pins: List[Pin] = []
        self.assigned: List[str] = []
        self.current_player_idx: int = 0
        self.num_turns: int = 0
        self.last_move: Optional[Move] = None
        self.done: bool = False
        self.winner: Optional[str] = None
        self.end_reason: Optional[str] = None
        self.goal_indices_by_colour: Dict[str, set[int]] = {}

    @property
    def current_player(self) -> str:
        return self.assigned[self.current_player_idx]

    def _build_board(self) -> HexBoard:
        if self.suppress_board_prints:
            with contextlib.redirect_stdout(io.StringIO()):
                return HexBoard(R=self.board_radius, hole_radius=self.hole_radius, spacing=self.spacing)
        return HexBoard(R=self.board_radius, hole_radius=self.hole_radius, spacing=self.spacing)

    def _auto_assign_colours(self, num_players: int) -> List[str]:
        if num_players < 2 or num_players > 6:
            raise ValueError("num_players must be between 2 and 6")

        assigned: List[str] = []
        while len(assigned) < num_players:
            if len(assigned) % 2 == 0:
                choices = [c for c in ALL_COLOURS if c not in assigned]
                choice = random.choice(choices)
                assigned.append(choice)
            else:
                opposite = self.board.colour_opposites[assigned[-1]]  # type: ignore[union-attr]
                if opposite not in assigned:
                    assigned.append(opposite)
        return assigned[:num_players]

    def reset(self, num_players: int = 2, preset_colours: Optional[List[str]] = None) -> Dict[str, Any]:
        self.board = self._build_board()
        self.board_pins = []
        self.num_turns = 0
        self.last_move = None
        self.done = False
        self.winner = None
        self.end_reason = None

        if preset_colours is not None:
            if len(preset_colours) < 2 or len(preset_colours) > 6:
                raise ValueError("preset_colours must contain between 2 and 6 colours")
            if len(set(preset_colours)) != len(preset_colours):
                raise ValueError("preset_colours contains duplicates")
            invalid = [c for c in preset_colours if c not in ALL_COLOURS]
            if invalid:
                raise ValueError(f"Invalid colours: {invalid}")
            self.assigned = list(preset_colours)
        else:
            self.assigned = self._auto_assign_colours(num_players)

        for colour in self.assigned:
            start_indices = self.board.axial_of_colour(colour)
            pins = [Pin(self.board, start_indices[i], id=i, color=colour) for i in range(10)]
            self.board_pins.extend(pins)

        self.goal_indices_by_colour = {
            colour: set(self.board.axial_of_colour(self.board.colour_opposites[colour]))
            for colour in self.assigned
        }

        if self.randomize_first_player:
            self.current_player_idx = random.randrange(len(self.assigned))
        else:
            self.current_player_idx = 0

        return self.get_observation(self.current_player)

    def get_player_pins(self, colour: str) -> List[Pin]:
        return [pin for pin in self.board_pins if pin.color == colour]

    def _is_destination_allowed(self, moving_colour: str, dest_idx: int) -> bool:
        cell = self.board.cells[dest_idx]  # type: ignore[index]
        if cell.occupied:
            return False
        if cell.postype == "board":
            return self.allow_neutral_cells
        if cell.postype == moving_colour:
            return True
        if cell.postype == self.board.colour_opposites[moving_colour]:  # type: ignore[index]
            return True
        return False

    def get_legal_actions(self, colour: Optional[str] = None) -> List[Dict[str, int]]:
        if colour is None:
            colour = self.current_player

        legal_actions: List[Dict[str, int]] = []
        for pin in self.get_player_pins(colour):
            for dest in pin.getPossibleMoves():
                if self._is_destination_allowed(colour, dest):
                    legal_actions.append({
                        "pin_id": pin.id,
                        "from": pin.axialindex,
                        "to": dest,
                    })
        return legal_actions

    def get_board_state(self) -> List[Dict[str, Any]]:
        return [
            {
                "color": pin.color,
                "pin_id": pin.id,
                "position": pin.axialindex,
            }
            for pin in self.board_pins
        ]

    def get_observation(self, colour: Optional[str] = None) -> Dict[str, Any]:
        if colour is None:
            colour = self.current_player

        own_pieces = [
            {"pin_id": pin.id, "position": pin.axialindex}
            for pin in self.get_player_pins(colour)
        ]

        legal_moves = self.get_legal_actions(colour)
        last_move_dict = None
        if self.last_move is not None:
            last_move_dict = {
                "player": self.last_move.player,
                "pin_id": self.last_move.pin_id,
                "from": self.last_move.from_axial,
                "to": self.last_move.to_axial,
            }

        return {
            "current_player": colour,
            "board_state": self.get_board_state(),
            "last_move": last_move_dict,
            "own_pieces": own_pieces,
            "legal_moves": legal_moves,
            "goal_indices": sorted(self.goal_indices_by_colour[colour]),
            "turn_number": self.num_turns,
            "num_players": len(self.assigned),
        }

    def _apply_move_without_placepin(self, pin: Pin, destination: int) -> None:
        old_idx = pin.axialindex
        self.board.cells[old_idx].occupied = False  # type: ignore[index]
        pin.axialindex = int(destination)
        self.board.cells[destination].occupied = True  # type: ignore[index]

    def _check_win(self, colour: str) -> bool:
        positions = {pin.axialindex for pin in self.get_player_pins(colour)}
        return positions == self.goal_indices_by_colour[colour]

    def _finalize(self, winner: Optional[str], reason: str) -> None:
        self.done = True
        self.winner = winner
        self.end_reason = reason

    def step(self, action: Dict[str, int]):
        if self.done:
            raise RuntimeError("Game is already finished.")

        colour = self.current_player
        pin_id = action["pin_id"]
        destination = action["to"]
        legal_actions = self.get_legal_actions(colour)
        legal_pairs = {(a["pin_id"], a["to"]) for a in legal_actions}
        if (pin_id, destination) not in legal_pairs:
            raise ValueError(f"Illegal action for {colour}: pin_id={pin_id}, to={destination}")

        pin = [p for p in self.get_player_pins(colour) if p.id == pin_id][0]
        from_axial = pin.axialindex
        self._apply_move_without_placepin(pin, destination)
        self.last_move = Move(player=colour, pin_id=pin_id, from_axial=from_axial, to_axial=destination)
        self.num_turns += 1

        if self._check_win(colour):
            self._finalize(winner=colour, reason="goal_reached")
        elif self.num_turns >= self.max_turns:
            self._finalize(winner=None, reason="max_turns_reached")
        else:
            next_idx = (self.current_player_idx + 1) % len(self.assigned)
            stuck_players = 0
            while stuck_players < len(self.assigned):
                next_colour = self.assigned[next_idx]
                if self.get_legal_actions(next_colour):
                    self.current_player_idx = next_idx
                    break
                stuck_players += 1
                next_idx = (next_idx + 1) % len(self.assigned)

            if stuck_players == len(self.assigned):
                self._finalize(winner=None, reason="no_legal_moves_for_any_player")

        reward = 1.0 if self.winner == colour else 0.0
        next_observation = None if self.done else self.get_observation(self.current_player)
        info = {
            "winner": self.winner,
            "reason": self.end_reason,
            "last_move": None if self.last_move is None else {
                "player": self.last_move.player,
                "pin_id": self.last_move.pin_id,
                "from": self.last_move.from_axial,
                "to": self.last_move.to_axial,
            },
        }
        return next_observation, reward, self.done, info

    def stop_now(self, reason: str = "interrupted") -> Dict[str, Any]:
        self._finalize(winner=None, reason=reason)
        return {
            "winner": self.winner,
            "ended_early": True,
            "reason": self.end_reason,
            "turns": self.num_turns,
        }


class OvernightRunner:
    def __init__(self, env: ChineseCheckersSelfPlayEnv, agents: Dict[str, BaseAgent], finish_current_game: bool = True):
        self.env = env
        self.agents = agents
        self.finish_current_game = finish_current_game
        self.stop = StopController()
        self._old_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if self.finish_current_game:
            if not self.stop.stop_requested:
                print("\nStop requested. Finishing current game, then exiting.")
                self.stop.stop_requested = True
            else:
                print("\nSecond interrupt received. Ending current game early.")
                self.stop.stop_immediately = True
        else:
            print("\nStop requested. Ending current game early.")
            self.stop.stop_requested = True
            self.stop.stop_immediately = True

    def run_single_game(self, num_players: int = 2, preset_colours: Optional[List[str]] = None) -> Dict[str, Any]:
        self.env.reset(num_players=num_players, preset_colours=preset_colours)

        while not self.env.done:
            if self.stop.stop_immediately:
                return self.env.stop_now(reason="interrupted")

            current_player = self.env.current_player
            obs = self.env.get_observation(current_player)
            legal_moves = obs["legal_moves"]
            if not legal_moves:
                if len(self.env.get_legal_actions()) == 0:
                    return self.env.stop_now(reason="current_player_has_no_legal_moves")

            action = self.agents[current_player].choose_action(obs)
            _, _, done, info = self.env.step(action)
            if done:
                return {
                    "winner": info.get("winner"),
                    "ended_early": False,
                    "reason": info.get("reason"),
                    "turns": self.env.num_turns,
                }

        return {
            "winner": self.env.winner,
            "ended_early": False,
            "reason": self.env.end_reason,
            "turns": self.env.num_turns,
        }

    def run_forever(
        self,
        num_players: int = 2,
        preset_colours: Optional[List[str]] = None,
        max_games: Optional[int] = None,
        report_every: int = 1,
    ) -> Dict[str, Any]:
        results: Dict[str, int] = {}
        completed_games = 0
        early_stops = 0

        try:
            while True:
                if max_games is not None and completed_games >= max_games:
                    break
                if self.stop.stop_requested and completed_games > 0:
                    break

                result = self.run_single_game(num_players=num_players, preset_colours=preset_colours)
                completed_games += 1

                winner_key = result["winner"] if result["winner"] is not None else "None"
                results[winner_key] = results.get(winner_key, 0) + 1
                if result.get("ended_early"):
                    early_stops += 1

                if report_every > 0 and completed_games % report_every == 0:
                    print(
                        f"Game {completed_games}: winner={result['winner']}, "
                        f"reason={result['reason']}, turns={result['turns']}"
                    )

                if self.stop.stop_requested:
                    break
        finally:
            signal.signal(signal.SIGINT, self._old_sigint)

        summary = {
            "games_completed": completed_games,
            "results": results,
            "early_stops": early_stops,
            "stop_requested": self.stop.stop_requested,
            "stop_immediately": self.stop.stop_immediately,
        }
        print("\nRun stopped.")
        print(summary)
        return summary


def build_random_agents(colours: List[str]) -> Dict[str, BaseAgent]:
    return {colour: RandomAgent(name=f"{colour}_agent") for colour in colours}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chinese Checkers self-play wrapper with overnight run support.")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players/models to assign (2-6).")
    parser.add_argument(
        "--colours",
        nargs="*",
        default=None,
        help="Optional explicit colour assignment, e.g. --colours red blue",
    )
    parser.add_argument("--max-games", type=int, default=None, help="Optional cap on number of games to run.")
    parser.add_argument("--max-turns", type=int, default=500, help="Maximum turns per game before declaring a draw.")
    parser.add_argument(
        "--finish-current-game",
        action="store_true",
        help="On Ctrl+C, finish the current game before stopping. Press Ctrl+C twice to stop immediately.",
    )
    parser.add_argument(
        "--stop-immediately",
        action="store_true",
        help="On Ctrl+C, end the current game immediately and stop.",
    )
    parser.add_argument("--randomize-first-player", action="store_true", help="Randomize the first player in each game.")
    parser.add_argument("--report-every", type=int, default=1, help="Print a summary every N completed games.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finish_current_game = True
    if args.stop_immediately:
        finish_current_game = False
    elif args.finish_current_game:
        finish_current_game = True

    env = ChineseCheckersSelfPlayEnv(
        randomize_first_player=args.randomize_first_player,
        max_turns=args.max_turns,
        suppress_board_prints=True,
    )

    if args.colours is not None:
        preview_colours = args.colours
    else:
        env.reset(num_players=args.num_players)
        preview_colours = list(env.assigned)

    agents = build_random_agents(preview_colours)
    runner = OvernightRunner(env, agents, finish_current_game=finish_current_game)
    runner.run_forever(
        num_players=args.num_players,
        preset_colours=args.colours,
        max_games=args.max_games,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
