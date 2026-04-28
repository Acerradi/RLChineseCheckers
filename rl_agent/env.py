"""Headless Chinese Checkers environment for RL training.

Imports the existing game code (single system/) without modifying it.
Supports 2, 3, 4, or 6 players.

Observation (fixed size for all player counts)
-----------------------------------------------
Float32 vector of length OBS_SIZE = (MAX_PLAYERS + 1) * BOARD_SIZE = 847.
  Channels 0 .. MAX_PLAYERS-1  : binary piece-occupancy, current player first.
    Absent player slots are zeroed, so one model works for any player count.
  Channel MAX_PLAYERS           : binary mask of the current player's goal cells.
    This tells the agent *where to go* without requiring it to infer the target
    purely from sparse reward.

Reward
------
Dense: (total_hex_distance_before - total_hex_distance_after) * DIST_SCALE
       per move, plus a small bonus each time a piece enters the goal zone.
Terminal: +1.0 on winning.  The -1.0 loss penalty is applied retroactively
          by SelfPlayTrainer after the episode ends.
"""
import contextlib
import io
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

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
BOARD_SIZE = 121        # total cells on the board (indices 0-120)
PINS_PER_PLAYER = 10   # pieces per player
MAX_PLAYERS = 6        # maximum supported player count
ACTION_DIM = PINS_PER_PLAYER * BOARD_SIZE          # 1210
OBS_SIZE = (MAX_PLAYERS + 1) * BOARD_SIZE          # 847: 6 occupancy + 1 goal zone

# Reward scaling: 0.01 reward per hex unit of distance improvement.
# Over a perfect game (~100 total hops for 10 pieces) this sums to ~1.0,
# comparable to the terminal win reward.
_DIST_SCALE = 0.01
_GOAL_ENTRY_BONUS = 0.05   # extra reward each time a piece enters the goal zone

COLOUR_ORDER: List[str] = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]
COLOUR_OPPOSITES: Dict[str, str] = {
    "red": "blue",        "blue": "red",
    "lawn green": "gray0", "gray0": "lawn green",
    "yellow": "purple",   "purple": "yellow",
}

_COLOURS_FOR_N: Dict[int, List[str]] = {
    2: ["red", "blue"],
    3: ["red", "lawn green", "yellow"],
    4: ["red", "blue", "lawn green", "gray0"],
    6: list(COLOUR_ORDER),
}


@contextlib.contextmanager
def _silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
class ChineseCheckersEnv:
    """Headless Chinese Checkers environment for RL.

    Action encoding
    ---------------
    action = pin_idx * BOARD_SIZE + dest_cell_idx
      pin_idx  : 0 .. PINS_PER_PLAYER-1
      dest_cell: 0 .. BOARD_SIZE-1
    """

    def __init__(self, n_players: int = 2) -> None:
        if n_players not in _COLOURS_FOR_N:
            raise ValueError(f"n_players must be one of {sorted(_COLOURS_FOR_N)}, got {n_players}")
        self.n_players = n_players
        self.player_colours: List[str] = _COLOURS_FOR_N[n_players]
        self.turn_order: List[str] = [c for c in COLOUR_ORDER if c in self.player_colours]

        self.obs_size: int = OBS_SIZE
        self.action_dim: int = ACTION_DIM

        # Game state — populated by reset()
        self.board: Optional[HexBoard] = None
        self.pins: Dict[str, List[Pin]] = {}
        self.turn_idx: int = 0
        self.move_count: int = 0
        self.done: bool = False
        self.winner: Optional[str] = None

        # Episode-constant lookups — populated by reset()
        self._goal_cell_indices: Dict[str, List[int]] = {}
        self._goal_cell_sets: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
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

        # Precompute goal cell indices (constant per episode)
        self._goal_cell_indices = {
            c: self.board.axial_of_colour(COLOUR_OPPOSITES[c])
            for c in self.player_colours
        }
        self._goal_cell_sets = {
            c: set(self._goal_cell_indices[c])
            for c in self.player_colours
        }
        return self._observe()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action for the current player.

        Reward = dense distance improvement + goal-entry bonus [+ 1.0 on win].
        """
        if self.done:
            raise RuntimeError("Episode finished; call reset() first.")

        colour = self.current_colour
        pin_idx, dest = self._decode(action)

        dist_before = self._total_dist_to_goal(colour)
        goal_before = self._pieces_in_goal(colour)

        with _silent():
            ok = self.pins[colour][pin_idx].placePin(dest)
        if not ok:
            raise ValueError(f"Illegal action {action} (pin {pin_idx} → cell {dest}) for {colour}.")

        self.move_count += 1
        dist_after = self._total_dist_to_goal(colour)
        goal_after = self._pieces_in_goal(colour)

        # Dense reward: every hex unit of improvement counts
        reward = (dist_before - dist_after) * _DIST_SCALE
        reward += (goal_after - goal_before) * _GOAL_ENTRY_BONUS

        if self._check_status(colour) == "WIN":
            self.done = True
            self.winner = colour
            reward += 1.0
            info = {"result": "win", "colour": colour, "move_count": self.move_count}
        else:
            self.turn_idx = (self.turn_idx + 1) % len(self.turn_order)
            info = {"result": "playing", "colour": colour, "move_count": self.move_count}

        return self._observe(), reward, self.done, info

    def get_legal_actions(self, colour: Optional[str] = None) -> List[int]:
        if colour is None:
            colour = self.current_colour
        actions: List[int] = []
        with _silent():
            for pin_idx, pin in enumerate(self.pins[colour]):
                for dest in pin.getPossibleMoves():
                    actions.append(pin_idx * BOARD_SIZE + dest)
        return actions

    def observe(self, from_perspective: Optional[str] = None) -> np.ndarray:
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
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # Channels 0..MAX_PLAYERS-1: piece occupancy (current player first)
        ordered = [perspective] + [c for c in self.turn_order if c != perspective]
        for i, colour in enumerate(ordered):
            for pin in self.pins.get(colour, []):
                obs[i * BOARD_SIZE + pin.axialindex] = 1.0

        # Channel MAX_PLAYERS: current player's goal zone (static, binary)
        # Gives the agent explicit geometric knowledge of where to move.
        for idx in self._goal_cell_indices.get(perspective, []):
            obs[MAX_PLAYERS * BOARD_SIZE + idx] = 1.0

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

    def _total_dist_to_goal(self, colour: str) -> float:
        """Minimum-cost bipartite assignment of outside pieces to free goal cells.

        Pieces already in the goal contribute 0 and are excluded from the
        matching.  Goal cells occupied by own pieces are excluded from the
        target set.  Using the optimal assignment guarantees that placing any
        piece into the goal always yields a strictly positive reward — there is
        no pathological sign-flip when another piece must reroute to a
        different free cell.
        """
        goal_cells = self._goal_cell_indices[colour]
        goal_set = self._goal_cell_sets[colour]

        pieces_out = [p for p in self.pins[colour] if p.axialindex not in goal_set]
        if not pieces_out:
            return 0.0

        own_in_goal = frozenset(p.axialindex for p in self.pins[colour]
                                if p.axialindex in goal_set)
        targets = [g for g in goal_cells if g not in own_in_goal]

        n = len(pieces_out)  # always equals len(targets): both = PINS_PER_PLAYER - k
        cost = np.empty((n, n), dtype=np.float32)
        for i, p in enumerate(pieces_out):
            pq = self.board.cells[p.axialindex].q
            pr = self.board.cells[p.axialindex].r
            ps = -pq - pr
            for j, g in enumerate(targets):
                gq = self.board.cells[g].q
                gr = self.board.cells[g].r
                gs = -gq - gr
                cost[i, j] = (abs(pq - gq) + abs(pr - gr) + abs(ps - gs)) / 2

        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].sum())
