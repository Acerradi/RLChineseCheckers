from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from checkers_board import HexBoard
from checkers_pins import Pin


COLOUR_ORDER = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]
COMPLEMENT = {"red": "blue", "lawn green": "gray0", "yellow": "purple"}
DEFAULT_PRIMARY_COLOURS = ["red", "lawn green", "yellow"]
MAX_PLAYERS = 6


@dataclass
class PlayerState:
    player_id: str
    name: str
    colour: str
    ready: bool = False
    status: str = "PLAYING"
    move_count: int = 0
    time_taken_sec: float = 0.0


class GameCore:
    """Shared rules engine for both training and live deployment adapters.

    This mirrors the behaviour of the provided server closely enough that a
    model trained against this core can later be wrapped by the socket client.
    """

    def __init__(
        self,
        *,
        game_id: Optional[str] = None,
        primary_colours: Optional[List[str]] = None,
        shuffle_primary: bool = True,
        turn_timeout_sec: Optional[float] = None,
        game_time_limit_sec: Optional[float] = None,
        enable_real_time_limits: bool = False,
    ):
        self.game_id = game_id or str(uuid.uuid4())
        self.board = HexBoard()
        self.players: List[PlayerState] = []
        self.pins_by_colour: Dict[str, List[Pin]] = {}
        self.status = "AVAILABLE"
        self.created_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.joined_primary_index = 0
        self.lock_joining = False

        self.primary_colours = list(primary_colours or DEFAULT_PRIMARY_COLOURS)
        if shuffle_primary:
            random.shuffle(self.primary_colours)

        self.turn_timeout_sec = turn_timeout_sec
        self.game_time_limit_sec = game_time_limit_sec
        self.enable_real_time_limits = enable_real_time_limits

        self.total_start_ns: Optional[int] = None
        self.turn_started_ns: Optional[int] = None
        self.turn_order: List[str] = []
        self.current_turn_index = 0
        self.move_count = 0
        self.move_times_ms: List[float] = []
        self.last_move: Optional[Dict[str, Any]] = None
        self.turn_timeout_notice: Optional[str] = None
        self.scores: Dict[str, Dict[str, float]] = {}
        self.history: List[Dict[str, Any]] = []

    def assign_colour(self) -> Optional[str]:
        n = len(self.players) + 1
        if n > MAX_PLAYERS:
            return None
        if n % 2 == 1:
            if self.joined_primary_index >= len(self.primary_colours):
                return None
            return self.primary_colours[self.joined_primary_index]
        primary = self.primary_colours[self.joined_primary_index]
        self.joined_primary_index += 1
        return COMPLEMENT[primary]

    def add_player(self, name: str) -> PlayerState:
        colour = self.assign_colour()
        if colour is None:
            raise ValueError("Game full or colour assignment failed")

        player = PlayerState(player_id=str(uuid.uuid4()), name=name, colour=colour)
        self.players.append(player)
        self._init_pins(colour)

        if len(self.players) == 1:
            self.status = "waiting for other player"
        else:
            self.status = "READY_TO_START"
        return player

    def mark_ready(self, player_id: str) -> None:
        player = self.get_player(player_id)
        if player is None:
            raise ValueError("Player not found")
        player.ready = True

        if self.status == "READY_TO_START":
            self.lock_joining = True
            if len(self.players) >= 2 and all(p.ready for p in self.players):
                self.status = "PLAYING"
                self.total_start_ns = time.perf_counter_ns()
                self.compute_turn_order()
                self.turn_started_ns = time.perf_counter_ns()

    def auto_start(self) -> None:
        for player in self.players:
            player.ready = True
        if len(self.players) >= 2:
            self.lock_joining = True
            self.status = "PLAYING"
            self.total_start_ns = time.perf_counter_ns()
            self.compute_turn_order()
            self.turn_started_ns = time.perf_counter_ns()

    def _init_pins(self, colour: str) -> None:
        if colour in self.pins_by_colour:
            return
        idxs = self.board.axial_of_colour(colour)[:10]
        self.pins_by_colour[colour] = [Pin(self.board, idxs[i], id=i, color=colour) for i in range(len(idxs))]

    def compute_turn_order(self) -> None:
        present = [p.colour for p in self.players]
        first = present[0]
        if first in COLOUR_ORDER:
            idx = COLOUR_ORDER.index(first)
            rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
        else:
            rotated = COLOUR_ORDER[:]
        self.turn_order = [c for c in rotated if c in present]
        self.current_turn_index = 0

    def current_turn_colour(self) -> Optional[str]:
        if self.status != "PLAYING" or not self.turn_order:
            return None
        return self.turn_order[self.current_turn_index]

    def current_player(self) -> Optional[PlayerState]:
        colour = self.current_turn_colour()
        if colour is None:
            return None
        return self.get_player_by_colour(colour)

    def advance_turn(self) -> None:
        if self.turn_order:
            self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
            self.turn_started_ns = time.perf_counter_ns()

    def ensure_time_limits(self) -> None:
        if not self.enable_real_time_limits:
            return

        if self.total_start_ns and self.game_time_limit_sec is not None:
            elapsed = (time.perf_counter_ns() - self.total_start_ns) / 1e9
            if elapsed > self.game_time_limit_sec:
                self.status = "FINISHED"
                self.turn_timeout_notice = "GAME TIME LIMIT REACHED."
                self.compute_scores()
                return

        if self.status == "PLAYING" and self.turn_started_ns and self.turn_timeout_sec is not None:
            turn_elapsed = (time.perf_counter_ns() - self.turn_started_ns) / 1e9
            if turn_elapsed > self.turn_timeout_sec:
                colour = self.current_turn_colour()
                self.turn_timeout_notice = (
                    f"Player with colour {colour} exceeded {self.turn_timeout_sec}s at move {self.move_count}. Turn skipped."
                )
                self.compute_scores()
                self.advance_turn()

    def get_player(self, player_id: str) -> Optional[PlayerState]:
        return next((p for p in self.players if p.player_id == player_id), None)

    def get_player_by_colour(self, colour: str) -> Optional[PlayerState]:
        return next((p for p in self.players if p.colour == colour), None)

    def check_player_status(self, colour: str) -> str:
        opposite = self.board.colour_opposites[colour]
        pins = self.pins_by_colour[colour]

        if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
            return "WIN"

        if all(len(p.getPossibleMoves()) == 0 for p in pins):
            return "DRAW"

        return "PLAYING"

    def get_legal_moves_for_colour(self, colour: str) -> Dict[int, List[int]]:
        pins = self.pins_by_colour[colour]
        return {i: list(pin.getPossibleMoves()) for i, pin in enumerate(pins)}

    def apply_move(self, player_id: str, pin_id: int, to_index: int) -> Dict[str, Any]:
        self.ensure_time_limits()

        if self.status != "PLAYING":
            return {"ok": False, "error": f"Game not in PLAYING: {self.status}"}

        pl = self.get_player(player_id)
        if pl is None:
            return {"ok": False, "error": "Player not in game"}

        if self.current_turn_colour() != pl.colour:
            return {"ok": False, "error": f"Not {pl.colour}'s turn."}

        pins = self.pins_by_colour[pl.colour]
        if not (0 <= pin_id < len(pins)):
            return {"ok": False, "error": "Invalid pin ID"}

        pin = pins[pin_id]
        legal = pin.getPossibleMoves()
        if to_index not in legal:
            return {"ok": False, "error": "Illegal move"}

        if self.enable_real_time_limits and self.turn_started_ns:
            dt = (time.perf_counter_ns() - self.turn_started_ns) / 1e9
            pl.time_taken_sec += dt

        start_ns = time.perf_counter_ns()
        from_idx = pin.axialindex
        moved_ok = pin.placePin(to_index)
        end_ns = time.perf_counter_ns()
        move_ms = (end_ns - start_ns) / 1e6

        if not moved_ok:
            return {"ok": False, "error": "Could not move"}

        pl.move_count += 1
        self.move_count += 1
        self.move_times_ms.append(move_ms)

        self.last_move = {
            "pin_id": pin_id,
            "from": from_idx,
            "to": to_index,
            "by": pl.name,
            "colour": pl.colour,
            "move_ms": move_ms,
        }
        self.history.append(dict(self.last_move))

        pl.status = self.check_player_status(pl.colour)
        if pl.status == "WIN":
            self.status = "FINISHED"
            self.compute_scores()
            return {"ok": True, "status": "WIN", "state": self.to_public_state(), "msg": f"{pl.name} Wins"}

        if pl.status == "DRAW":
            live = self.players
            draws = [p for p in live if self.check_player_status(p.colour) == "DRAW"]
            if len(draws) == len(live) - 1:
                winner = next(p for p in live if p not in draws)
                self.status = "FINISHED"
                self.compute_scores()
                return {
                    "ok": True,
                    "status": "WIN",
                    "state": self.to_public_state(),
                    "msg": f"{winner.name} Wins, others Draw.",
                }

        self.advance_turn()
        self.compute_scores()
        return {"ok": True, "status": "CONTINUE", "state": self.to_public_state()}

    def compute_scores(self) -> None:
        def axial_dist(a, b):
            dq = abs(a.q - b.q)
            dr = abs(a.r - b.r)
            ds = abs((-a.q - a.r) - (-b.q - b.r))
            return max(dq, dr, ds)

        for pl in self.players:
            colour = pl.colour
            pins = self.pins_by_colour[colour]
            opposite = self.board.colour_opposites[colour]

            time_score = max(0.0, 100.0 - pl.time_taken_sec) if pl.time_taken_sec > 0 else 0.0
            move_score_func = lambda x: math.exp(-((x - 45) ** 2) / (2 * ((4 if x < 45 else 18) ** 2)))
            move_score = move_score_func(pl.move_count) if pl.move_count > 0 else 0.0

            pins_in_goal = sum(1 for p in pins if self.board.cells[p.axialindex].postype == opposite)
            pin_goal_score = pins_in_goal * 100.0

            target_idxs = self.board.axial_of_colour(opposite)
            target_cells = [self.board.cells[i] for i in target_idxs]
            total_dist = 0
            for p in pins:
                if self.board.cells[p.axialindex].postype != opposite:
                    best = min(axial_dist(self.board.cells[p.axialindex], tgt) for tgt in target_cells)
                    total_dist += best
            distance_score = max(0.0, 200.0 - total_dist) if pl.move_count > 0 else 0.0

            final_score = time_score + move_score + pin_goal_score + distance_score
            self.scores[pl.player_id] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "moves": pl.move_count,
                "pins_in_goal": pins_in_goal,
                "total_distance": total_dist,
                "time_taken_sec": pl.time_taken_sec,
            }

    def to_public_state(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "players": [
                {
                    "player_id": pl.player_id,
                    "name": pl.name,
                    "colour": pl.colour,
                    "ready": pl.ready,
                    "status": pl.status,
                    "score": self.scores.get(pl.player_id),
                }
                for pl in self.players
            ],
            "pins": {colour: [p.axialindex for p in pins] for colour, pins in self.pins_by_colour.items()},
            "move_count": self.move_count,
            "current_turn_colour": self.current_turn_colour(),
            "turn_order": list(self.turn_order),
            "last_move": self.last_move,
            "turn_timeout_notice": self.turn_timeout_notice,
        }


def make_observation(game: GameCore, colour: str) -> Dict[str, Any]:
    """Return the same style of observation your deployed bot can consume.

    This deliberately mirrors the original client contract:
    - public state from get_state
    - legal moves from get_legal_moves
    - metadata about the controlled colour
    """
    return {
        "colour": colour,
        "state": game.to_public_state(),
        "legal_moves": game.get_legal_moves_for_colour(colour),
    }


Action = Tuple[int, int]
