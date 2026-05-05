
import sys
import os

# Go two directories up
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
)


from dataclasses import dataclass
import random
from typing import List, Dict, Optional, Any

from single_system.checkers_board import HexBoard
from single_system.checkers_pins import Pin


HALF_COLOURS_1 = ['red', 'lawn green', 'yellow']
HALF_COLOURS_2 = ['blue', 'gray0', 'purple']


@dataclass
class Move:
    player: str
    pin_id: int
    from_axial: int
    to_axial: int


class ChineseCheckersEnv:
    def __init__(self, board_radius: int = 4, use_gui: bool = False):
        self.board_radius = board_radius
        self.use_gui = use_gui

        self.board = None
        self.board_pins: List[Pin] = []
        self.assigned_colours: List[str] = []
        self.current_player_idx: int = 0
        self.num_turns: int = 0
        self.last_move: Optional[Move] = None
        self.done: bool = False
        self.winner: Optional[str] = None

    def reset(self, num_players: int = 2, randomize_start_player: bool = True):
        if num_players < 2 or num_players > 6:
            raise ValueError("num_players must be between 2 and 6")

        self.board = HexBoard(R=self.board_radius, hole_radius=16, spacing=34)
        self.board_pins = []
        self.assigned_colours = self._assign_colours(num_players)
        self.num_turns = 0
        self.last_move = None
        self.done = False
        self.winner = None

        for colour in self.assigned_colours:
            axials_colour = self.board.axial_of_colour(colour)
            pins = [Pin(self.board, axials_colour[i], id=i, color=colour) for i in range(10)]
            self.board_pins.extend(pins)

        if randomize_start_player:
            self.current_player_idx = random.randrange(len(self.assigned_colours))
        else:
            self.current_player_idx = 0

        return self.get_observation(self.current_player)

    @property
    def current_player(self) -> str:
        return self.assigned_colours[self.current_player_idx]

    def _assign_colours(self, num_players: int) -> List[str]:
        """
        Preserves your current opposite-pair idea:
        first choose one from a half, then add its opposite, until enough players.
        """
        assigned = []
        available = HALF_COLOURS_1 + HALF_COLOURS_2

        while len(assigned) < num_players:
            if len(assigned) % 2 == 0:
                choices = [c for c in available if c not in assigned]
                choice = random.choice(choices)
                assigned.append(choice)
            else:
                opposite = self.board.colour_opposites[assigned[-1]]
                if opposite not in assigned:
                    assigned.append(opposite)

        return assigned[:num_players]

    def get_player_pins(self, colour: str) -> List[Pin]:
        return [pin for pin in self.board_pins if pin.color == colour]

    def get_legal_actions(self, colour: Optional[str] = None) -> List[Dict[str, int]]:
        if colour is None:
            colour = self.current_player

        legal_actions = []
        for pin in self.get_player_pins(colour):
            possible_moves = pin.getPossibleMoves()
            for dest in possible_moves:
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
            "turn_number": self.num_turns,
            "num_players": len(self.assigned_colours),
        }

    def step(self, action: Dict[str, int]):
        """
        action format:
        {
            "pin_id": 2,
            "to": 37
        }
        """
        if self.done:
            raise RuntimeError("Game is already finished.")

        colour = self.current_player
        pin_id = action["pin_id"]
        destination = action["to"]

        matching_pins = [
            pin for pin in self.board_pins
            if pin.color == colour and pin.id == pin_id
        ]
        if not matching_pins:
            raise ValueError(f"No pin found for player={colour}, pin_id={pin_id}")

        pin = matching_pins[0]
        from_axial = pin.axialindex

        legal_destinations = pin.getPossibleMoves()
        if destination not in legal_destinations:
            raise ValueError(
                f"Illegal move for player={colour}, pin_id={pin_id}, to={destination}"
            )

        success = pin.placePin(destination, silent=True)
        if not success:
            raise RuntimeError("Move failed in underlying game logic.")

        self.last_move = Move(
            player=colour,
            pin_id=pin_id,
            from_axial=from_axial,
            to_axial=destination,
        )

        # TODO: replace this with your real win-condition logic
        if self._check_win(colour):
            self.done = True
            self.winner = colour

        self.num_turns += 1

        if not self.done:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.assigned_colours)

        reward = self._get_reward(colour)

        next_observation = None if self.done else self.get_observation(self.current_player)

        info = {
            "current_player": self.current_player if not self.done else None,
            "winner": self.winner,
            "last_move": self.last_move,
        }

        return next_observation, reward, self.done, info

    def _check_win(self, colour: str) -> bool:
        """
        Placeholder.
        Replace this with your actual Chinese Checkers win test:
        all 10 pins of 'colour' must be in the target triangle.
        """
        target_colour = self.board.colour_opposites[colour]
        target_axials = set(self.board.axial_of_colour(target_colour))

        player_pins = self.get_player_pins(colour)

        win = all(pin.axialindex in target_axials for pin in player_pins)

        return win

    def _get_reward(self, acting_player: str) -> float:
        """
        Placeholder reward.
        For now:
        - 1.0 on win
        - 0.0 otherwise
        """
        if self.winner == acting_player:
            return 1.0
        return 0.0

    def render_ascii(self):
        self.board.print_ascii(pins=self.board_pins, empty='·')