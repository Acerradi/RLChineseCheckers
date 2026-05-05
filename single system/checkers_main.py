import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Import rl first so its __init__.py caches checkers_board from multi-system
# (no debug print spam) before checkers_gui imports it.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import rl  # noqa: triggers sys.path setup for checkers_board/checkers_pins

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checkers_gui import BoardGUI
from rl.core import GameCore, make_observation
from rl.policy_template import MyPolicy, load_model
from rl.policies import RandomPolicy

CHAMPION_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "gnn_h128_l4", "self_play", "champion.pt")
MOVE_DELAY_MS = 600


def make_agent(kind: str):
    if kind == "champion":
        if not os.path.isfile(CHAMPION_CHECKPOINT):
            raise FileNotFoundError(f"Champion checkpoint not found: {CHAMPION_CHECKPOINT}")
        model = load_model(CHAMPION_CHECKPOINT, device="cpu")
        return MyPolicy(model=model, device="cpu")
    return RandomPolicy()


def all_pins(game: GameCore):
    pins = []
    for colour_pins in game.pins_by_colour.values():
        pins.extend(colour_pins)
    return pins


def run(agent1: str, agent2: str):
    game = GameCore()
    game.add_player("P1")
    game.add_player("P2")
    game.auto_start()

    colours = list(game.turn_order)
    agents = {colours[0]: make_agent(agent1), colours[1]: make_agent(agent2)}
    labels = {colours[0]: agent1, colours[1]: agent2}

    gui = BoardGUI(game.board, all_pins(game))
    gui.root.title(f"{agent1} ({colours[0]}) vs {agent2} ({colours[1]})")

    def step():
        if game.status == "FINISHED":
            return

        colour = game.current_turn_colour()
        if colour is None:
            return

        obs = make_observation(game, colour)
        pin_id, to_index = agents[colour].select_action(obs)
        player_id = next(p.player_id for p in game.players if p.colour == colour)
        result = game.apply_move(player_id=player_id, pin_id=pin_id, to_index=to_index)

        if not result.get("ok"):
            print(f"[warn] move rejected ({colour}): {result.get('error')}")

        gui.refresh(all_pins(game))

        if game.status == "FINISHED":
            game.compute_scores()
            for p in game.players:
                if p.status == "WIN":
                    msg = f"GAME OVER — {labels[p.colour]} ({p.colour}) WINS in {game.move_count} moves"
                    gui.root.title(msg)
                    print(msg)
        else:
            gui.root.after(MOVE_DELAY_MS, step)

    gui.root.after(MOVE_DELAY_MS, step)
    gui.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch two agents play Chinese Checkers")
    parser.add_argument("--agent1", choices=["champion", "random"], default="champion",
                        help="Agent for the first colour (default: champion)")
    parser.add_argument("--agent2", choices=["champion", "random"], default="random",
                        help="Agent for the second colour (default: random)")
    parser.add_argument("--delay", type=int, default=600,
                        help="Milliseconds between moves (default: 600)")
    args = parser.parse_args()
    MOVE_DELAY_MS = args.delay
    run(args.agent1, args.agent2)
