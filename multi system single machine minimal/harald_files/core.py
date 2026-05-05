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

    def __init__(self, *,
             game_id: Optional[str] = None,
             primary_colours: Optional[List[str]] = None,
             shuffle_primary: bool = True,
             turn_timeout_sec: Optional[float] = None,
             game_time_limit_sec: Optional[float] = None,
             enable_real_time_limits: bool = False,

             # Training/adjudication controls
             enable_training_adjudication: bool = True,
             repetition_limit: int = 3,
             stall_limit_per_colour: int = 24,
             hard_stall_limit_per_colour: int = 48,
             stuck_home_grace_moves: int = 35,
             max_stranded_home_turns: int = 20,
             no_goal_progress_limit: int = 200):
        
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

        # Training/adjudication state
        self.enable_training_adjudication = enable_training_adjudication
        self.repetition_limit = repetition_limit
        self.stall_limit_per_colour = stall_limit_per_colour
        self.hard_stall_limit_per_colour = hard_stall_limit_per_colour
        self.stuck_home_grace_moves = stuck_home_grace_moves
        self.max_stranded_home_turns = max_stranded_home_turns
        self.no_goal_progress_limit = no_goal_progress_limit

        self.state_repetition_counts: Dict[tuple, int] = {}
        self.colour_stall_counts: Dict[str, int] = {}
        self.colour_stranded_home_turns: Dict[str, int] = {}
        self.colour_pins_in_goal: Dict[str, int] = {}
        self.colour_moves_without_goal_progress: Dict[str, int] = {}
        self.adjudication_reason: Optional[str] = None
        self.last_adjudication_event: Optional[Dict[str, Any]] = None

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
        if self.enable_training_adjudication:
            self.state_repetition_counts[self.board_state_key()] = 1

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
        """Return a mapping of pin ID to list of legal move indices for the given colour."""
        pins = self.pins_by_colour[colour]
        return {i: list(pin.getPossibleMoves()) for i, pin in enumerate(pins)}

    def apply_move(self, player_id: str, pin_id: int, to_index: int) -> Dict[str, Any]:
        """Apply a move for a player, returning the result and updated state or error."""
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

        before_progress = self.training_progress_score(pl.colour)
        was_forced = self.is_forced_move(pl.colour)

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

        self.last_move = {"pin_id": pin_id,
                          "from": from_idx,
                          "to": to_index,
                          "by": pl.name,
                          "colour": pl.colour,
                          "move_ms": move_ms}
        self.history.append(dict(self.last_move))

        adjudication_event = None
        if self.enable_training_adjudication:
            after_progress = self.training_progress_score(pl.colour)
            adjudication_event = self.update_repetition_stall_and_stranding(moved_colour=pl.colour,
                                                                            moved_pin_id=pin_id,
                                                                            from_idx=from_idx,
                                                                            to_idx=to_index,
                                                                            before_progress=before_progress,
                                                                            after_progress=after_progress,
                                                                            was_forced=was_forced)

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
                return {"ok": True,
                        "status": "WIN",
                        "state": self.to_public_state(),
                        "msg": f"{winner.name} Wins, others Draw."}
        if self.status == "FINISHED":
            self.compute_scores()
            return {"ok": True,
                    "status": "ADJUDICATED",
                    "state": self.to_public_state(),
                    "msg": self.adjudication_reason,
                    "adjudication": adjudication_event}

        self.advance_turn()
        self.compute_scores()
        return {"ok": True,
                "status": "CONTINUE",
                "state": self.to_public_state(),
                "adjudication": adjudication_event}

    def compute_scores(self) -> None:
        """Compute scores for all players based on their current state."""
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
            self.scores[pl.player_id] = {"final_score": final_score,
                                         "time_score": time_score,
                                         "move_score": move_score,
                                         "pin_goal_score": pin_goal_score,
                                         "distance_score": distance_score,
                                         "moves": pl.move_count,
                                         "pins_in_goal": pins_in_goal,
                                         "total_distance": total_dist,
                                         "time_taken_sec": pl.time_taken_sec}

    def to_public_state(self) -> Dict[str, Any]:
        """Return a representation of the game state that can be safely shared with clients."""
        return {"game_id": self.game_id,
                "status": self.status,
                "players": [{"player_id": pl.player_id,
                             "name": pl.name,
                             "colour": pl.colour,
                             "ready": pl.ready,
                             "status": pl.status,
                             "score": self.scores.get(pl.player_id)} for pl in self.players],
                "pins": {colour: [p.axialindex for p in pins] for colour, pins in self.pins_by_colour.items()},
                "move_count": self.move_count,
                "current_turn_colour": self.current_turn_colour(),
                "turn_order": list(self.turn_order),
                "last_move": self.last_move,
                "turn_timeout_notice": self.turn_timeout_notice,

                # New training diagnostics
                "adjudication_reason": self.adjudication_reason,
                "last_adjudication_event": self.last_adjudication_event,
                "training_progress": {pl.colour: self.training_progress_score(pl.colour)
                                      for pl in self.players},
                "stranded_home_pieces": {pl.colour: self.count_stranded_home_pieces(pl.colour)
                                         for pl in self.players},
                "home_pieces": {pl.colour: self.count_home_pieces(pl.colour)
                                for pl in self.players}}

    def _axial_dist(self, a, b) -> int:
        dq = abs(a.q - b.q)
        dr = abs(a.r - b.r)
        ds = abs((-a.q - a.r) - (-b.q - b.r))
        return max(dq, dr, ds)

    def board_state_key(self) -> tuple:
        """
        Hashable board state including side to move.

        Including current_turn_colour matters because the same board position with
        a different side to move is not the same game state.
        """
        pieces = []
        for colour in sorted(self.pins_by_colour.keys()):
            positions = tuple(sorted(p.axialindex for p in self.pins_by_colour[colour]))
            pieces.append((colour, positions))

        return (self.current_turn_colour(), tuple(pieces))

    def count_total_legal_moves(self, colour: str) -> int:
        legal = self.get_legal_moves_for_colour(colour)
        return sum(len(moves) for moves in legal.values())

    def is_forced_move(self, colour: str) -> bool:
        """
        Used to avoid punishing repetition/stalling when the player truly has only
        one legal move available.
        """
        return self.count_total_legal_moves(colour) <= 1

    def training_progress_score(self, colour: str) -> float:
        """
        Clean score for training/adjudication.

        Higher is better. This intentionally avoids time_score and move_score,
        because those are useful for reporting but noisy as learning signals.
        """
        pins = self.pins_by_colour[colour]
        target_colour = self.board.colour_opposites[colour]
        target_idxs = self.board.axial_of_colour(target_colour)
        target_cells = [self.board.cells[i] for i in target_idxs]

        pins_in_goal = 0
        total_dist = 0
        home_pieces = 0
        stranded_home = self.count_stranded_home_pieces(colour)

        for pin in pins:
            cell = self.board.cells[pin.axialindex]
            zone = getattr(cell, "postype", "board")

            if zone == target_colour:
                pins_in_goal += 1
            else:
                total_dist += min(self._axial_dist(cell, tgt) for tgt in target_cells)

            if zone == colour:
                home_pieces += 1

        # Weighting rationale:
        # - goal pieces are very valuable
        # - distance matters continuously
        # - home pieces are bad
        # - stranded home pieces are very bad
        # - quadratic endgame bonus gives a stronger gradient when almost done
        n_pins = len(pins) or 1
        endgame_bonus = 10.0 * (pins_in_goal / n_pins) ** 2
        return (
            pins_in_goal * 30.0
            - float(total_dist)
            - home_pieces * 8.0
            - stranded_home * 30.0
            + endgame_bonus
        )

    def normalized_training_value(self, colour: str) -> float:
        """
        Multiplayer-safe value target in [-1, 1] based on relative progress.

        This is useful for truncated/adjudicated games where nobody officially won.
        """
        if not self.players:
            return 0.0

        scores = {pl.colour: self.training_progress_score(pl.colour) for pl in self.players}
        my_score = scores[colour]
        others = [v for c, v in scores.items() if c != colour]

        if not others:
            return 0.0

        best_other = max(others)
        diff = my_score - best_other

        # Scale controls how quickly progress differences saturate.
        # 60 is a reasonable starting value for this board size.
        return max(-1.0, min(1.0, diff / 60.0))

    def count_home_pieces(self, colour: str) -> int:
        pins = self.pins_by_colour[colour]
        return sum(
            1
            for p in pins
            if getattr(self.board.cells[p.axialindex], "postype", "board") == colour
        )

    def count_stranded_home_pieces(self, colour: str) -> int:
        """
        Counts pieces still in their home triangle that currently have no legal move
        directly out of the home triangle.

        This targets the failure mode you described: a piece gets left behind and
        becomes surrounded, forcing useless shuffling elsewhere.
        """
        stranded = 0
        pins = self.pins_by_colour[colour]

        for pin in pins:
            from_cell = self.board.cells[pin.axialindex]
            from_zone = getattr(from_cell, "postype", "board")

            if from_zone != colour:
                continue

            legal_moves = pin.getPossibleMoves()
            can_leave_home = False

            for to_idx in legal_moves:
                to_cell = self.board.cells[int(to_idx)]
                to_zone = getattr(to_cell, "postype", "board")
                if to_zone != colour:
                    can_leave_home = True
                    break

            if not can_leave_home:
                stranded += 1

        return stranded

    def adjudicate_by_progress(self, reason: str = "PROGRESS_ADJUDICATION") -> None:
        """
        End the game and mark the player with the best clean progress score as winner.
        """
        if not self.players:
            self.status = "FINISHED"
            self.adjudication_reason = reason
            return

        progress_by_colour = {
            pl.colour: self.training_progress_score(pl.colour)
            for pl in self.players
        }

        best_colour = max(progress_by_colour, key=progress_by_colour.get)

        for pl in self.players:
            pl.status = "WIN" if pl.colour == best_colour else "LOSS"

        self.status = "FINISHED"
        self.adjudication_reason = reason
        self.compute_scores()

    def update_repetition_stall_and_stranding(self, *,
        moved_colour: str,
        moved_pin_id: int,
        from_idx: int,
        to_idx: int,
        before_progress: float,
        after_progress: float,
        was_forced: bool,
    ) -> Dict[str, Any]:
        """
        Called after a successful move.

        Repetition/stall penalties are suppressed if the player had only one legal
        move before moving. Stranded-home penalties are not suppressed, because they
        are meant to teach the model not to create that state earlier in the game.
        """
        event = {
            "forced": was_forced,
            "repetition_count": 0,
            "stall_count": 0,
            "home_pieces": self.count_home_pieces(moved_colour),
            "stranded_home_pieces": self.count_stranded_home_pieces(moved_colour),
            "repetition_penalty": 0.0,
            "stall_penalty": 0.0,
            "stranded_home_penalty": 0.0,
            "adjudicated": False,
            "reason": None,
        }

        # -------------------------
        # Repetition detection
        # -------------------------
        key = self.board_state_key()
        rep_count = self.state_repetition_counts.get(key, 0) + 1
        self.state_repetition_counts[key] = rep_count
        event["repetition_count"] = rep_count

        if rep_count >= self.repetition_limit and not was_forced:
            event["repetition_penalty"] = -0.10 * (rep_count - self.repetition_limit + 1)

        # -------------------------
        # Stall detection
        # -------------------------
        progress_delta = after_progress - before_progress

        if progress_delta > 0.01:
            self.colour_stall_counts[moved_colour] = 0
        else:
            self.colour_stall_counts[moved_colour] = self.colour_stall_counts.get(moved_colour, 0) + 1

        stall_count = self.colour_stall_counts[moved_colour]
        event["stall_count"] = stall_count

        if stall_count >= self.stall_limit_per_colour and not was_forced:
            event["stall_penalty"] = -0.05 * (stall_count - self.stall_limit_per_colour + 1)

        # -------------------------
        # Non-resettable goal-progress counter
        # Unlike the stall counter this never resets on micro-progress,
        # so a model cannot avoid adjudication by occasionally nudging a piece.
        # -------------------------
        target_colour_for_np = self.board.colour_opposites[moved_colour]
        pins_in_goal_now = sum(
            1 for p in self.pins_by_colour[moved_colour]
            if getattr(self.board.cells[p.axialindex], "postype", "board") == target_colour_for_np
        )
        prev_pins_in_goal = self.colour_pins_in_goal.get(moved_colour, 0)
        if pins_in_goal_now > prev_pins_in_goal:
            self.colour_moves_without_goal_progress[moved_colour] = 0
        else:
            self.colour_moves_without_goal_progress[moved_colour] = (
                self.colour_moves_without_goal_progress.get(moved_colour, 0) + 1
            )
        self.colour_pins_in_goal[moved_colour] = pins_in_goal_now
        event["moves_without_goal_progress"] = self.colour_moves_without_goal_progress[moved_colour]

        # -------------------------
        # Stranded home-piece detection
        # -------------------------
        stranded = event["stranded_home_pieces"]

        if self.move_count >= self.stuck_home_grace_moves and stranded > 0:
            self.colour_stranded_home_turns[moved_colour] = (
                self.colour_stranded_home_turns.get(moved_colour, 0) + 1
            )

            stranded_turns = self.colour_stranded_home_turns[moved_colour]

            # Small but persistent penalty. The value head will propagate this back
            # to earlier decisions that left the piece behind.
            event["stranded_home_penalty"] = -0.03 * stranded * min(stranded_turns, 10)

            # If the player moves a non-home piece while stranded home pieces exist,
            # penalize a bit more. This discourages goal-triangle shuffling while a
            # home piece remains trapped.
            from_zone = getattr(self.board.cells[from_idx], "postype", "board")
            to_zone = getattr(self.board.cells[to_idx], "postype", "board")

            if from_zone != moved_colour and to_zone != moved_colour:
                event["stranded_home_penalty"] -= 0.05 * stranded
        else:
            self.colour_stranded_home_turns[moved_colour] = 0

        # -------------------------
        # Hard adjudication
        # -------------------------
        if (
            stall_count >= self.hard_stall_limit_per_colour
        ):
            event["adjudicated"] = True
            event["reason"] = f"STALL_LIMIT_REACHED:{moved_colour}"
            self.adjudicate_by_progress(event["reason"])

        elif (
            self.colour_stranded_home_turns.get(moved_colour, 0) >= self.max_stranded_home_turns
            and stranded > 0
        ):
            event["adjudicated"] = True
            event["reason"] = f"STRANDED_HOME_LIMIT_REACHED:{moved_colour}"
            self.adjudicate_by_progress(event["reason"])

        elif (
            self.colour_moves_without_goal_progress.get(moved_colour, 0) >= self.no_goal_progress_limit
        ):
            event["adjudicated"] = True
            event["reason"] = f"NO_GOAL_PROGRESS:{moved_colour}"
            self.adjudicate_by_progress(event["reason"])

        self.last_adjudication_event = event
        return event

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
