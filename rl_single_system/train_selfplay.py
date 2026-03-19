import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# Allow imports from repo root and "single system"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SINGLE_SYSTEM_DIR = os.path.join(REPO_ROOT, "single system")

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

if SINGLE_SYSTEM_DIR not in sys.path:
    sys.path.append(SINGLE_SYSTEM_DIR)


# Local RL modules
from state_encoder import encode_observation
from action_space import ActionMapper
from rl_agent import PPOAgent
from reward import compute_progress_reward


# CHANGE THIS IMPORT IF YOUR ENV CLASS NAME IS DIFFERENT
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
    """
    Tries to extract a board position from various possible own_pieces formats.
    Adjust if your environment uses a different structure.
    """
    if isinstance(piece, dict):
        if "position" in piece:
            return piece["position"]
        if "pos" in piece:
            return piece["pos"]
        if "axial" in piece:
            return piece["axial"]

    if isinstance(piece, (tuple, list)):
        # common patterns: (pin_id, pos) or [pin_id, pos]
        if len(piece) >= 2:
            return piece[1]

    if isinstance(piece, int):
        return piece

    raise ValueError(f"Could not extract position from piece={piece!r}")


def compute_simple_state_info(obs: Dict[str, Any]) -> Dict[str, float]:
    """
    Basic shaped-reward summary.

    This uses a rough distance proxy:
      min(abs(pos - goal)) over goal indices

    Later, you should replace that with a true board distance metric.
    """
    own_pieces = obs.get("own_pieces", [])
    goal_indices = set(obs.get("goal_indices", []))

    pieces_in_goal = 0
    total_distance = 0.0

    for piece in own_pieces:
        pos = extract_piece_position(piece)

        if pos in goal_indices:
            pieces_in_goal += 1

        if goal_indices:
            nearest_goal_dist = min(abs(pos - g) for g in goal_indices)
        else:
            nearest_goal_dist = 0.0

        total_distance += float(nearest_goal_dist)

    return {
        "pieces_in_goal": float(pieces_in_goal),
        "total_distance": float(total_distance),
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
    """
    Adjust parameters to match your actual environment constructor.
    """
    env = ChineseCheckersSelfPlayEnv(
        randomize_first_player=True,
        max_turns=500,
        suppress_board_prints=True,
    )
    return env


def build_agent_and_mapper(
    env: ChineseCheckersSelfPlayEnv,
    colour_to_idx: Dict[str, int],
    device: str,
) -> Tuple[PPOAgent, ActionMapper]:
    env.reset(num_players=2, preset_colours=["red", "blue"])

    first_player = env.current_player
    first_obs = env.get_observation(first_player)

    encoded_obs, board_indices = encode_observation(
        first_obs,
        current_colour=first_player,
        colour_to_idx=colour_to_idx,
    )

    mapper = ActionMapper(board_indices=board_indices, max_pin_id=10)

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
) -> None:
    ensure_dir(MODELS_DIR)

    colour_to_idx = {
        "red": 0,
        "blue": 1,
    }

    env = make_env()
    agent, mapper = build_agent_and_mapper(env, colour_to_idx, device)

    start_episode = 0
    if resume:
        start_episode = load_checkpoint(agent, LATEST_CHECKPOINT, device=device)

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
                # If the environment supports a pass / null move, keep this.
                # Otherwise you may need a different fallback.
                _, _, done, info = env.step(None)
                final_info = info if isinstance(info, dict) else {}
                continue

            prev_state_info = compute_simple_state_info(obs)

            action_id, logprob, value = agent.act(encoded_obs, action_mask)
            action = mapper.decode_action(action_id)

            # Optional sanity check:
            # if action not in legal_moves, print for debugging
            next_obs, env_reward, done, info = env.step(action)
            final_info = info if isinstance(info, dict) else {}

            # Try to get same player's new perspective for shaped reward
            try:
                next_player_obs = env.get_observation(player)
            except Exception:
                next_player_obs = obs

            next_state_info = compute_simple_state_info(next_player_obs)

            winner = get_winner_from_info(final_info)
            won = bool(done and winner == player)
            lost = bool(done and winner is not None and winner != player)

            reward = compute_progress_reward(
                prev_info=prev_state_info,
                next_info=next_state_info,
                won=won,
                lost=lost,
            )

            traj["obs"].append(encoded_obs)
            traj["actions"].append(action_id)
            traj["logprobs"].append(logprob)
            traj["values"].append(value)
            traj["rewards"].append(reward)
            traj["dones"].append(float(done))
            traj["action_masks"].append(action_mask)

            episode_reward += reward

        # Only update if we actually collected transitions
        if len(traj["obs"]) > 0:
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

        winner = get_winner_from_info(final_info)
        reason = final_info.get("reason", "unknown") if isinstance(final_info, dict) else "unknown"

        print(
            f"[episode {episode + 1}/{num_episodes}] "
            f"reward={episode_reward:.2f} "
            f"steps={safety_steps} "
            f"winner={winner} "
            f"reason={reason}"
        )

        if (episode + 1) % save_every == 0:
            save_checkpoint(agent, episode + 1, LATEST_CHECKPOINT)

        if (episode + 1) % snapshot_every == 0:
            snapshot_path = os.path.join(MODELS_DIR, f"ep_{episode + 1:05d}.pt")
            save_checkpoint(agent, episode + 1, snapshot_path)

    save_checkpoint(agent, num_episodes, LATEST_CHECKPOINT)
    print("[done] training finished")


if __name__ == "__main__":
    # Use "cuda" if available and wanted
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        num_episodes=5000,
        device=device,
        resume=True,
        save_every=50,
        snapshot_every=500,
    )