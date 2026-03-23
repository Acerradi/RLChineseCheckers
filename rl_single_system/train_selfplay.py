import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Allow imports from repo root and "single system"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SINGLE_SYSTEM_DIR = os.path.join(REPO_ROOT, "single_system")

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

if SINGLE_SYSTEM_DIR not in sys.path:
    sys.path.append(SINGLE_SYSTEM_DIR)

# Local RL modules
from rl_single_system.state_encoder import encode_observation
from rl_single_system.action_space import ActionMapper
from rl_single_system.rl_agent import PPOAgent
from rl_single_system.reward import compute_progress_reward
from rl_single_system.board_distance import precompute_all_pairs_shortest_paths

from single_system.harald.rl_selfplay_wrapper_overnight import ChineseCheckersSelfPlayEnv


MODELS_DIR = os.path.join(REPO_ROOT, "models")
LATEST_CHECKPOINT = os.path.join(MODELS_DIR, "latest.pt")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(agent: PPOAgent, episode: int, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "episode": episode,
            "model_state_dict": agent.net.state_dict(),
            "optimizer_state_dict": agent.opt.state_dict(),
        },
        path,
    )
    print(f"[checkpoint] saved: {path}")


def load_checkpoint(agent: PPOAgent, path: str, device: str = "cpu") -> int:
    if not os.path.exists(path):
        print(f"[checkpoint] no checkpoint found at {path}, starting fresh")
        return 0

    checkpoint = torch.load(path, map_location=device)
    agent.net.load_state_dict(checkpoint["model_state_dict"])
    agent.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    start_episode = int(checkpoint.get("episode", 0))
    print(f"[checkpoint] loaded: {path} (resume from episode {start_episode})")
    return start_episode


def extract_piece_position(piece: Any) -> int:
    if isinstance(piece, dict):
        if "position" in piece:
            return piece["position"]
        if "pos" in piece:
            return piece["pos"]
        if "axial" in piece:
            return piece["axial"]

    if isinstance(piece, (tuple, list)) and len(piece) >= 2:
        return piece[1]

    if isinstance(piece, int):
        return piece

    raise ValueError(f"Could not extract position from piece={piece!r}")


def compute_simple_state_info(obs, all_distances):
    own_pieces = obs.get("own_pieces", [])
    goal_indices = set(obs.get("goal_indices", []))

    distances = []

    for piece in own_pieces:
        pos = extract_piece_position(piece)

        if goal_indices and pos in all_distances:
            reachable = [
                all_distances[pos][g]
                for g in goal_indices
                if g in all_distances[pos]
            ]
            d = min(reachable) if reachable else 999.0
        else:
            d = 999.0

        distances.append(d)

    pieces_in_goal = sum(1 for d in distances if d == 0)
    pieces_within_1 = sum(1 for d in distances if d <= 1)
    pieces_within_2 = sum(1 for d in distances if d <= 2)
    pieces_within_3 = sum(1 for d in distances if d <= 3)
    total_distance = float(sum(distances))

    return {
        "distances": distances,
        "pieces_in_goal": float(pieces_in_goal),
        "pieces_within_1": float(pieces_within_1),
        "pieces_within_2": float(pieces_within_2),
        "pieces_within_3": float(pieces_within_3),
        "total_distance": total_distance,
    }

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[float],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[List[float], List[float]]:
    advantages: List[float] = []
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)
        next_value = values[t]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def get_winner_from_info(info: Dict[str, Any]) -> Any:
    if not isinstance(info, dict):
        return None
    return info.get("winner", None)


def make_env() -> ChineseCheckersSelfPlayEnv:
    return ChineseCheckersSelfPlayEnv(
        randomize_first_player=True,
        max_turns=200,
        suppress_board_prints=True,
    )


def get_full_board_indices(env):
    board = env.board

    if hasattr(board, "cells"):
        cells = board.cells

        if isinstance(cells, dict):
            keys = list(cells.keys())
            if keys and all(isinstance(x, int) for x in keys):
                return sorted(keys)

        if isinstance(cells, list):
            if cells and all(isinstance(x, int) for x in cells):
                return sorted(cells)

    if hasattr(board, "index_of"):
        index_of = board.index_of
        if isinstance(index_of, dict):
            vals = list(index_of.values())
            if vals and all(isinstance(x, int) for x in vals):
                return sorted(set(vals))

    raise RuntimeError("Could not determine full board indices from env.board")


def build_agent_and_mapper(env, colour_to_idx, device):
    env.reset(num_players=2, preset_colours=["red", "blue"])

    first_player = env.current_player
    first_obs = env.get_observation(first_player)

    encoded_obs, _ = encode_observation(
        first_obs,
        current_colour=first_player,
        colour_to_idx=colour_to_idx,
    )

    full_board_indices = get_full_board_indices(env)
    print("FULL BOARD CELL COUNT:", len(full_board_indices))
    print("FULL BOARD SAMPLE:", full_board_indices[:20])

    mapper = ActionMapper(board_indices=full_board_indices, max_pin_id=10)
    print("NEW ACTION SIZE:", mapper.action_size)

    agent = PPOAgent(
        obs_dim=len(encoded_obs),
        action_dim=mapper.action_size,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        device=device,
    )

    return agent, mapper


def train(
    num_episodes: int = 5000,
    device: str = "cpu",
    resume: bool = True,
    save_every: int = 50,
    snapshot_every: int = 500,
    print_every: int = 50,
) -> None:
    ensure_dir(MODELS_DIR)

    colour_to_idx = {
        "red": 0,
        "blue": 1,
    }

    env = make_env()
    agent, mapper = build_agent_and_mapper(env, colour_to_idx, device)
    all_distances, adjacency, neighbor_dist = precompute_all_pairs_shortest_paths(env.board)



    start_episode = 0
    if resume:
        start_episode = load_checkpoint(agent, LATEST_CHECKPOINT, device=device)

    reward_history: List[float] = []
    steps_history: List[int] = []
    draw_history: List[int] = []

    for episode in range(start_episode, num_episodes):
        env.reset(num_players=2, preset_colours=["red", "blue"])
        done = False
        safety_steps = 0

        traj: Dict[str, List[Any]] = {
            "obs": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "action_masks": [],
        }

        episode_reward = 0.0
        final_info: Dict[str, Any] = {}

        while not done:
            safety_steps += 1
            if safety_steps > 2000:
                print(f"[warning] safety break in episode {episode + 1}")
                final_info = {"winner": None, "reason": "safety_break"}
                break

            player = env.current_player
            obs = env.get_observation(player)

            encoded_obs, _ = encode_observation(
                obs,
                current_colour=player,
                colour_to_idx=colour_to_idx,
            )

            legal_moves = obs.get("legal_moves", [])
            action_mask = mapper.legal_action_mask(legal_moves)

            if float(action_mask.sum()) <= 0.0:
                print(f"[warning] no legal actions for player {player} in episode {episode + 1}")
                done = True
                final_info = {
                    "winner": None,
                    "reason": "no_legal_actions",
                }
                break

            prev_state_info = compute_simple_state_info(obs, all_distances)

            action_id, logprob, value = agent.act(encoded_obs, action_mask)
            action = mapper.decode_action(action_id)

            try:
                next_obs, env_reward, done, info = env.step(action)
            except ValueError as e:
                print(f"[illegal sampled action] {e}")
                print(f"player={player}, action={action}")
                print(f"legal sample={legal_moves[:10]}")
                done = True
                final_info = {"winner": None, "reason": "illegal_sampled_action"}
                break

            final_info = info if isinstance(info, dict) else {}

            try:
                next_player_obs = env.get_observation(player)
            except Exception:
                next_player_obs = obs



            next_state_info = compute_simple_state_info(next_player_obs, all_distances)

            winner = get_winner_from_info(final_info)

            reason = final_info.get("reason", "unknown") if isinstance(final_info, dict) else "unknown"

            won = bool(done and winner == player)
            lost = bool(done and winner is not None and winner != player)
            draw = bool(done and winner is None and reason == "max_turns_reached")

            reward = compute_progress_reward(
                prev_info=prev_state_info,
                next_info=next_state_info,
                won=won,
                lost=lost,
                draw=draw,
            )

            traj["obs"].append(encoded_obs)
            traj["actions"].append(action_id)
            traj["logprobs"].append(logprob)
            traj["values"].append(value)
            traj["rewards"].append(reward)
            traj["dones"].append(float(done))
            traj["action_masks"].append(action_mask)

            episode_reward += reward

        if len(traj["obs"]) >= 2:
            advantages, returns = compute_gae(
                rewards=traj["rewards"],
                values=traj["values"],
                dones=traj["dones"],
                gamma=agent.gamma,
                gae_lambda=agent.gae_lambda,
            )

            batch = {
                **traj,
                "advantages": advantages,
                "returns": returns,
            }

            agent.update(batch)
        else:
            print(f"[skip update] episode {episode + 1} had only {len(traj['obs'])} transitions")

        winner = get_winner_from_info(final_info)

        reward_history.append(episode_reward)
        steps_history.append(safety_steps)
        draw_history.append(1 if winner is None else 0)

        if (episode + 1) % print_every == 0:
            avg_reward = sum(reward_history[-print_every:]) / len(reward_history[-print_every:])
            avg_steps = sum(steps_history[-print_every:]) / len(steps_history[-print_every:])
            draw_rate = sum(draw_history[-print_every:]) / len(draw_history[-print_every:])

            print(
                f"[ep {episode + 1}] "
                f"avg_reward={avg_reward:.2f} "
                f"avg_steps={avg_steps:.1f} "
                f"draw_rate={draw_rate:.2f}"
            )

        if (episode + 1) % save_every == 0:
            save_checkpoint(agent, episode + 1, LATEST_CHECKPOINT)

        if (episode + 1) % snapshot_every == 0:
            snapshot_path = os.path.join(MODELS_DIR, f"ep_{episode + 1:05d}.pt")
            save_checkpoint(agent, episode + 1, snapshot_path)

    save_checkpoint(agent, num_episodes, LATEST_CHECKPOINT)
    print("[done] training finished")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        num_episodes=1000000, #Episode number to stop at
        device=device,
        resume=True,
        save_every=50,
        snapshot_every=500,
        print_every=50,
    )