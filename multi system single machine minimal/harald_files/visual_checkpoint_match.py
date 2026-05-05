from __future__ import annotations

import os
import sys
import random
import time
from typing import Dict

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from environment import ChineseCheckersEnv
from policy_template import MyPolicy, load_model, HeuristicPolicy
from checkers_gui import BoardGUI


# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_A = "checkpoints/gnn_h128_l4/self_play/champion.pt"
CHECKPOINT_B = "checkpoints/gnn_h128_l4/bootstrap/shared_model_final.pt"


DEVICE = "cpu"
NUM_PLAYERS = 2
MAX_MOVES = 300
MOVE_DELAY_SEC = 0.5
RANDOM_SEED = 42

# If True, shuffle the order of the two player names before env.reset()
# so either checkpoint can end up with the first assigned colour.
SHUFFLE_PLAYER_ORDER = False

PLAYER_A_NAME = "checkpoint_A"
PLAYER_B_NAME = "checkpoint_B"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policies(env: ChineseCheckersEnv) -> Dict[str, MyPolicy]:
    """
    Build colour -> policy mapping after env.reset(), because colours are only
    known after players are added to the environment.
    """
    model_a = load_model(CHECKPOINT_A, device=DEVICE)
    model_b = load_model(CHECKPOINT_B, device=DEVICE)

    policy_a = MyPolicy(model=model_a, device=DEVICE)
    policy_b = MyPolicy(model=model_b, device=DEVICE)
    #policy_a = HeuristicPolicy()     # Uncomment to test a checkpoint vs heuristic
    policy_b = HeuristicPolicy()     # Uncomment to test a checkpoint vs heuristic

    policies_by_colour: Dict[str, MyPolicy] = {}

    for player in env.game.players:
        if player.name == PLAYER_A_NAME:
            policies_by_colour[player.colour] = policy_a
        elif player.name == PLAYER_B_NAME:
            policies_by_colour[player.colour] = policy_b
        else:
            raise ValueError(f"Unexpected player name in environment: {player.name}")

    return policies_by_colour


def print_player_mapping(env: ChineseCheckersEnv) -> None:
    print("\n=== PLAYER / COLOUR ASSIGNMENT ===")
    for p in env.game.players:
        print(f"{p.name} -> colour={p.colour}")
    print("Turn order:", env.turn_order)
    print("=================================\n")


def main() -> None:
    if not os.path.exists(CHECKPOINT_A):
        raise FileNotFoundError(f"CHECKPOINT_A not found: {CHECKPOINT_A}")
    if not os.path.exists(CHECKPOINT_B):
        raise FileNotFoundError(f"CHECKPOINT_B not found: {CHECKPOINT_B}")

    set_seed(RANDOM_SEED)

    player_names = [PLAYER_A_NAME, PLAYER_B_NAME]
    if SHUFFLE_PLAYER_ORDER:
        random.shuffle(player_names)

    env = ChineseCheckersEnv(num_players=NUM_PLAYERS, player_names=player_names)
    env.reset()

    policies_by_colour = build_policies(env)
    print_player_mapping(env)

    # GUI uses the live board + live pin objects from env.game
    gui = BoardGUI(env.game.board, [pin for pins in env.game.pins_by_colour.values() for pin in pins])
    gui.root.update_idletasks()
    gui.root.update()

    move_no = 0

    while True:
        if env.game.status == "FINISHED":
            break

        colour = env.current_turn_colour
        if colour is None:
            print("No current turn colour. Stopping.")
            break

        policy = policies_by_colour[colour]
        obs = env.observe(colour)
        action = policy.select_action(obs)

        pin_id, to_index = action
        player = env.game.get_player(env.colour_to_player_id[colour])
        from_index = env.game.pins_by_colour[colour][pin_id].axialindex

        print(f"MOVE {move_no + 1}: "
              f"{player.name} ({colour}) "
              f"pin_id={pin_id} {from_index}->{to_index}")

        step_result = env.step(colour, action)
        move_no += 1

        # Refresh GUI after the move
        gui.refresh([pin for pins in env.game.pins_by_colour.values() for pin in pins])

        if not step_result.info.get("ok", False):
            print("Move failed:", step_result.info)
            break

        # Print last move / short score snapshot
        env.game.compute_scores()
        for p in env.game.players:
            sc = env.game.scores.get(p.player_id, {})
            print(f"  {p.name:12s} colour={p.colour:10s} "
                  f"status={p.status:7s} "
                  f"score={sc.get('final_score', 0.0):7.2f}")

        print("-" * 70)

        if move_no >= MAX_MOVES:
            print(f"Reached move cap of {MAX_MOVES}. Stopping.")
            break

        time.sleep(MOVE_DELAY_SEC)

    print("\n=== FINAL STATE ===")
    print("Game status:", env.game.status)
    print("Total moves:", env.game.move_count)

    env.game.compute_scores()
    for p in env.game.players:
        sc = env.game.scores.get(p.player_id, {})
        print(f"{p.name} ({p.colour}) "
              f"status={p.status} "
              f"final_score={sc.get('final_score', 0.0):.2f} "
              f"moves={sc.get('moves', 0)} "
              f"pins_in_goal={sc.get('pins_in_goal', 0)} "
              f"distance={sc.get('total_distance', 0)}")

    print("\nClose the GUI window to exit.")
    gui.run()


if __name__ == "__main__":
    main()