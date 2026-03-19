import os
import sys
import torch
import numpy as np

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SINGLE_SYSTEM_DIR = os.path.join(REPO_ROOT, "single_system")

sys.path.append(REPO_ROOT)
sys.path.append(SINGLE_SYSTEM_DIR)

from rl_single_system.state_encoder import encode_observation
from rl_single_system.action_space import ActionMapper
from rl_single_system.rl_agent import PPOAgent

from single_system.harald.rl_selfplay_wrapper_overnight import ChineseCheckersSelfPlayEnv


MODEL_PATH = os.path.join(REPO_ROOT, "models", "latest.pt")


def make_env():
    return ChineseCheckersSelfPlayEnv(
        randomize_first_player=True,
        max_turns=500,
        suppress_board_prints=True,
    )


def get_full_board_indices(env):
    board = env.board

    # Best source in this project: maps board positions -> integer ids
    if hasattr(board, "index_of"):
        index_of = board.index_of
        if isinstance(index_of, dict):
            vals = list(index_of.values())
            if vals and all(isinstance(x, int) for x in vals):
                return sorted(set(vals))

    # Fallback: board.cells may be objects, so map them through index_of
    if hasattr(board, "cells") and hasattr(board, "index_of"):
        cells = board.cells
        index_of = board.index_of
        try:
            indices = [index_of[cell] for cell in cells if cell in index_of]
            if indices:
                return sorted(set(indices))
        except Exception:
            pass

    raise RuntimeError("Cannot extract integer board indices from env.board")


def load_agent(env, device="cpu"):
    env.reset(num_players=2, preset_colours=["red", "blue"])

    first_player = env.current_player
    obs = env.get_observation(first_player)

    encoded_obs, _ = encode_observation(obs, first_player, None)
    board_indices = get_full_board_indices(env)

    mapper = ActionMapper(board_indices=board_indices, max_pin_id=10)

    agent = PPOAgent(
        obs_dim=len(encoded_obs),
        action_dim=mapper.action_size,
        device=device,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    agent.net.load_state_dict(checkpoint["model_state_dict"])

    print(f"[loaded model] {MODEL_PATH}")

    return agent, mapper


def play_one_game(env, agent, mapper, colour_to_idx):
    env.reset(num_players=2, preset_colours=["red", "blue"])
    done = False

    while not done:
        player = env.current_player
        obs = env.get_observation(player)

        encoded_obs, _ = encode_observation(obs, player, colour_to_idx)
        legal_moves = obs["legal_moves"]
        action_mask = mapper.legal_action_mask(legal_moves)

        if action_mask.sum() == 0:
            return None, "no_legal_actions"

        action_id, _, _ = agent.act(encoded_obs, action_mask)
        action = mapper.decode_action(action_id)

        try:
            _, _, done, info = env.step(action)
        except Exception as e:
            print("ERROR:", e)
            return None, "error"

    winner = info.get("winner", None)
    reason = info.get("reason", "unknown")

    return winner, reason


def test_agent(num_games=20, device="cpu"):
    env = make_env()
    agent, mapper = load_agent(env, device)

    colour_to_idx = {"red": 0, "blue": 1}

    results = {
        "red": 0,
        "blue": 0,
        "draw": 0,
    }

    for i in range(num_games):
        winner, reason = play_one_game(env, agent, mapper, colour_to_idx)

        if winner == "red":
            results["red"] += 1
        elif winner == "blue":
            results["blue"] += 1
        else:
            results["draw"] += 1

        print(f"[game {i+1}] winner={winner}, reason={reason}")

    print("\n=== RESULTS ===")
    print(results)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_agent(num_games=20, device=device)