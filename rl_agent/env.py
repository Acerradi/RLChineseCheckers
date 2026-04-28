"""Headless Chinese Checkers environment for RL training.

Imports the existing game code (single system/) without modifying it.
Supports 2, 3, 4, or 6 players.  Observations are always a fixed-size vector
of length BOARD_SIZE * MAX_PLAYERS (726) regardless of how many players are
in the current game — absent player slots are zeroed out.  This lets a single
trained model play in games of any player count.
"""
import contextlib
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Locate and import the existing game implementation
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_GAME_SRC = os.path.abspath(os.path.join(_HERE, "..", "single system"))
if _GAME_SRC not in sys.path:
    sys.path.insert(0, _GAME_SRC)

from checkers_board import HexBoard  # noqa: E402
from checkers_pins import Pin  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_SIZE = 121        # total cells on the board
PINS_PER_PLAYER = 10   # pieces per player
MAX_PLAYERS = 6        # maximum supported player count
ACTION_DIM = PINS_PER_PLAYER * BOARD_SIZE  # 1210 — (pin_idx, dest_cell) pairs
OBS_SIZE = BOARD_SIZE * MAX_PLAYERS        # 726 — fixed for all player counts

COLOUR_ORDER: List[str] = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]
COLOUR_OPPOSITES: Dict[str, str] = {
    "red": "blue",       "blue": "red",
    "lawn green": "gray0", "gray0": "lawn green",
    "yellow": "purple",  "purple": "yellow",
}

# Colours assigned by player count
_COLOURS_FOR_N: Dict[int, List[str]] = {
    2: ["red", "blue"],
    3: ["red", "lawn green", "yellow"],  # each wins by filling the opposite zone
    4: ["red", "blue", "lawn green", "gray0"],
    6: list(COLOUR_ORDER),
}


@contextlib.contextmanager
def _silent():
    """Suppress stdout (board construction + placePin prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
class ChineseCheckersEnv:
    """Headless Chinese Checkers environment.

    Action encoding
    ---------------
    action = pin_idx * BOARD_SIZE + dest_cell_idx
      pin_idx   : 0 .. PINS_PER_PLAYER-1  (index into the current player's pin list)
      dest_cell : 0 .. BOARD_SIZE-1       (board cell index)

    Observation
    -----------
    Float32 vector of fixed length OBS_SIZE (BOARD_SIZE * MAX_PLAYERS = 726).
    The current player's occupancy is always in slots [0 .. BOARD_SIZE-1],
    followed by each opponent in turn order.  Slots for absent players (when
    n_players < MAX_PLAYERS) are left as zeros.  The fixed size means one
    trained model works for any player count up to MAX_PLAYERS.

    Rewards
    -------
    +0.1 per piece that newly enters the goal zone on the current move.
    +1.0 when the current player wins (all pieces in goal zone).
    The terminal loss penalty (-1.0) is intentionally NOT applied here; the
    SelfPlayTrainer patches the losing player's last buffer entry after the
    episode finishes.
    """

    def __init__(self, n_players: int = 2) -> None:
        if n_players not in _COLOURS_FOR_N:
            raise ValueError(f"n_players must be one of {sorted(_COLOURS_FOR_N)}, got {n_players}")
        self.n_players = n_players
        self.player_colours: List[str] = _COLOURS_FOR_N[n_players]
        # Turn order respects the global COLOUR_ORDER
        self.turn_order: List[str] = [c for c in COLOUR_ORDER if c in self.player_colours]

        self.obs_size: int = OBS_SIZE  # always 726 regardless of n_players
        self.action_dim: int = ACTION_DIM

        # Mutable game state — initialised by reset()
        self.board: Optional[HexBoard] = None
        self.pins: Dict[str, List[Pin]] = {}
        self.turn_idx: int = 0
        self.move_count: int = 0
        self.done: bool = False
        self.winner: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new game and return the initial observation."""
        with _silent():
            self.board = HexBoard()
            self.pins = {}
            for colour in self.player_colours:
                indices = self.board.axial_of_colour(colour)
                self.pins[colour] = [
                    Pin(self.board, idx, i, colour)
                    for i, idx in enumerate(indices)
                ]
        self.turn_idx = 0
        self.move_count = 0
        self.done = False
        self.winner = None
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply *action* for the current player.

        Returns (observation, reward, done, info).
        The observation is from the perspective of the player whose turn it
        is *after* the move (or the winner's perspective if the game ended).
        """
        if self.done:
            raise RuntimeError("Episode is finished; call reset() first.")

        colour = self.current_colour
        pin_idx, dest = self._decode(action)

        before = self._pieces_in_goal(colour)
        with _silent():
            ok = self.pins[colour][pin_idx].placePin(dest)
        if not ok:
            raise ValueError(
                f"Action {action} (pin {pin_idx} → cell {dest}) is illegal for {colour}."
            )

        self.move_count += 1
        after = self._pieces_in_goal(colour)
        progress_reward = (after - before) * 0.1

        if self._check_status(colour) == "WIN":
            self.done = True
            self.winner = colour
            reward = 1.0
            info = {"result": "win", "colour": colour, "move_count": self.move_count}
        else:
            reward = progress_reward
            self.turn_idx = (self.turn_idx + 1) % len(self.turn_order)
            info = {"result": "playing", "colour": colour, "move_count": self.move_count}

        return self._observe(), reward, self.done, info

    def get_legal_actions(self, colour: Optional[str] = None) -> List[int]:
        """Return every valid action integer for *colour* (or the current player)."""
        if colour is None:
            colour = self.current_colour
        actions: List[int] = []
        with _silent():
            for pin_idx, pin in enumerate(self.pins[colour]):
                for dest in pin.getPossibleMoves():
                    actions.append(pin_idx * BOARD_SIZE + dest)
        return actions

    def observe(self, from_perspective: Optional[str] = None) -> np.ndarray:
        """Return the observation from *from_perspective*'s point of view."""
        return self._observe(from_perspective)

    @property
    def current_colour(self) -> str:
        return self.turn_order[self.turn_idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self, perspective: Optional[str] = None) -> np.ndarray:
        if perspective is None:
            perspective = self.current_colour
        obs = np.zeros(self.obs_size, dtype=np.float32)
        # Current player first, then others in turn order
        ordered = [perspective] + [c for c in self.turn_order if c != perspective]
        for i, colour in enumerate(ordered):
            for pin in self.pins.get(colour, []):
                obs[i * BOARD_SIZE + pin.axialindex] = 1.0
        return obs

    def _decode(self, action: int) -> Tuple[int, int]:
        return action // BOARD_SIZE, action % BOARD_SIZE

    def _check_status(self, colour: str) -> str:
        opposite = COLOUR_OPPOSITES[colour]
        if all(self.board.cells[p.axialindex].postype == opposite
               for p in self.pins[colour]):
            return "WIN"
        return "PLAYING"

    def _pieces_in_goal(self, colour: str) -> int:
        opposite = COLOUR_OPPOSITES[colour]
        return sum(
            1 for p in self.pins[colour]
            if self.board.cells[p.axialindex].postype == opposite
        )
