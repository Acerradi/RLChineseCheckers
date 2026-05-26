from __future__ import annotations

import os
import sys
import random
import time
import copy
from typing import Dict

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from environment import ChineseCheckersEnv
from policy_template import MyPolicy, load_model, HeuristicPolicy, NeuralMCTS
from checkers_gui import BoardGUI


# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_A = "checkpoints/gnn_h128_l4/self_play/champion.pt"
CHECKPOINT_B = "checkpoints/gnn_h128_l4/self_play/shared_model_final.pt"


DEVICE = "cpu"
NUM_PLAYERS = 2
MAX_MOVES = 300
MOVE_DELAY_SEC = 0.5
RANDOM_SEED = 42

# Raw policy-head matches are useful for checking what the deployed model would
# do, but MCTS can reveal whether the value/policy network still contains enough
# signal for search to recover better play.
USE_MCTS = False
MCTS_SIMULATIONS = 128

# If True, shuffle the order of the two player names before env.reset()
# so either checkpoint can end up with the first assigned colour.
SHUFFLE_PLAYER_ORDER = False

PLAYER_A_NAME = "checkpoint_A"
PLAYER_B_NAME = "checkpoint_B"


class VisualSearchEnv:
    """Small adapter so the visual runner can use policy_template.NeuralMCTS."""
    def __init__(self, env: ChineseCheckersEnv):
        self.env = env

    def clone(self):
        return VisualSearchEnv(copy.deepcopy(self.env))

    def current_player_colour(self) -> str:
        return self.env.current_turn_colour

    def observe(self, colour: str):
        return self.env.observe(colour)

    def step(self, colour: str, action):
        step_result = self.env.step(colour, action)
        return step_result.reward, step_result.done, step_result.info

    def value_for_colour(self, colour: str) -> float:
        state = self.env.game.to_public_state()
        me = next((p for p in state["players"] if p["colour"] == colour), None)
        if me is not None:
            if me["status"] == "WIN":
                return 1.0
            if me["status"] == "DRAW":
                return 0.0
            if me["status"] == "LOSS":
                return -1.0
        return self.env.game.normalized_training_value(colour)


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
    policy_a = HeuristicPolicy()     # Uncomment to test a checkpoint vs heuristic
    #policy_b = HeuristicPolicy()     # Uncomment to test a checkpoint vs heuristic

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


def select_visual_action(env: ChineseCheckersEnv, colour: str, policy: MyPolicy):
    if not USE_MCTS:
        return policy.select_action(env.observe(colour))

    mcts = NeuralMCTS(model=policy.model,
                      graph_builder=policy.graph_builder,
                      device=str(policy.device),
                      num_simulations=MCTS_SIMULATIONS)
    best_action, _, _ = mcts.search(VisualSearchEnv(copy.deepcopy(env)))
    return best_action


def print_diagnostics(env: ChineseCheckersEnv) -> None:
    state = env.game.to_public_state()
    progress = state.get("training_progress", {})
    home = state.get("home_pieces", {})
    stranded = state.get("stranded_home_pieces", {})

    for p in env.game.players:
        sc = env.game.scores.get(p.player_id, {})
        colour = p.colour
        print(f"  {p.name:12s} colour={colour:10s} "
              f"status={p.status:7s} "
              f"score={sc.get('final_score', 0.0):7.2f} "
              f"dist={sc.get('total_distance', 0):3} "
              f"goal={sc.get('pins_in_goal', 0):2} "
              f"progress={progress.get(colour, 0.0):7.2f} "
              f"home={home.get(colour, 0):2} "
              f"stranded={stranded.get(colour, 0):2}")

    if state.get("adjudication_reason"):
        print("  adjudication_reason:", state.get("adjudication_reason"))
    last_event = state.get("last_adjudication_event")
    if last_event:
        print("  last_adjudication_event:",
              f"reason={last_event.get('reason')}",
              f"stall={last_event.get('stall_count')}",
              f"repetition={last_event.get('repetition_count')}",
              f"stranded={last_event.get('stranded_home_pieces')}")


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
    print(f"Action mode: {'MCTS' if USE_MCTS else 'raw policy'}")
    if USE_MCTS:
        print(f"MCTS simulations: {MCTS_SIMULATIONS}")

    # GUI uses the live board + live pin objects from env.game
    gui = BoardGUI(env.game.board, [pin for pins in env.game.pins_by_colour.values() for pin in pins])
    gui.root.update_idletasks()
    gui.root.update()

    move_no = 0
    repeated_pair_counts: Dict[tuple, int] = {}

    while True:
        if env.game.status == "FINISHED":
            break

        colour = env.current_turn_colour
        if colour is None:
            print("No current turn colour. Stopping.")
            break

        policy = policies_by_colour[colour]
        obs = env.observe(colour)
        action = select_visual_action(env, colour, policy)

        pin_id, to_index = action
        legal_for_pin = obs["legal_moves"].get(pin_id, obs["legal_moves"].get(str(pin_id), []))
        if to_index not in legal_for_pin:
            print(f"Illegal action selected before step: colour={colour} action={action} legal={legal_for_pin}")
            break

        player = env.game.get_player(env.colour_to_player_id[colour])
        from_index = env.game.pins_by_colour[colour][pin_id].axialindex
        move_key = (colour, pin_id, min(from_index, to_index), max(from_index, to_index))
        repeated_pair_counts[move_key] = repeated_pair_counts.get(move_key, 0) + 1

        print(f"MOVE {move_no + 1}: "
              f"{player.name} ({colour}) "
              f"pin_id={pin_id} {from_index}->{to_index} "
              f"pair_repeat={repeated_pair_counts[move_key]}")

        step_result = env.step(colour, action)
        move_no += 1

        # Refresh GUI after the move
        gui.refresh([pin for pins in env.game.pins_by_colour.values() for pin in pins])

        if not step_result.info.get("ok", False):
            print("Move failed:", step_result.info)
            break

        if step_result.info.get("status") == "ADJUDICATED" or step_result.info.get("adjudication"):
            print("Step status:", step_result.info.get("status"),
                  "message:", step_result.info.get("message"),
                  "adjudication:", step_result.info.get("adjudication"))

        # Print last move / short score snapshot
        env.game.compute_scores()
        print_diagnostics(env)

        print("-" * 70)

        if move_no >= MAX_MOVES:
            print(f"Reached move cap of {MAX_MOVES}. Stopping.")
            break

        time.sleep(MOVE_DELAY_SEC)

    print("\n=== FINAL STATE ===")
    print("Game status:", env.game.status)
    print("Adjudication reason:", env.game.adjudication_reason)
    print("Last adjudication event:", env.game.last_adjudication_event)
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
