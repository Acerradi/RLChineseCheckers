"""Evaluate a trained Chinese Checkers agent.

Usage (from the project root)
------------------------------
  # vs random baseline
  python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt

  # vs a frozen copy of itself (self-play quality check)
  python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt --opponent self

  # 4-player evaluation
  python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt --n-players 4

Run `python -m rl_agent.evaluate --help` for all options.
"""
import argparse
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from .agent import PPOAgent, PPOConfig
from .env import ChineseCheckersEnv

__all__ = ["evaluate"]


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    agent_path: str,
    n_episodes: int = 200,
    n_players: int = 2,
    opponent: str = "random",
    max_steps: int = 1000,
    device: str = "cpu",
) -> Dict:
    """Run *n_episodes* games and return aggregate statistics.

    Parameters
    ----------
    agent_path : path to a .pt checkpoint produced by PPOAgent.save()
    n_episodes : number of evaluation games
    n_players  : 2 / 3 / 4 / 6
    opponent   : "random"  — uniformly random legal moves
                 "self"    — a second instance of the same checkpoint
                 "latest"  — the latest.pt from the same checkpoint directory
    max_steps  : truncation limit (avoids infinite evaluation games)
    device     : "cpu" / "cuda" / "mps"
    """
    env = ChineseCheckersEnv(n_players=n_players)

    # Reconstruct the exact network architecture from the checkpoint
    agent = PPOAgent.from_checkpoint(agent_path, device=device)
    agent.net.eval()

    opp_agent: Optional[PPOAgent] = None
    if opponent in ("self", "latest"):
        opp_path = (
            agent_path
            if opponent == "self"
            else os.path.join(os.path.dirname(agent_path), "latest.pt")
        )
        opp_agent = PPOAgent.from_checkpoint(opp_path, device=device)
        opp_agent.net.eval()

    wins = draws = losses = truncations = 0
    lengths: List[int] = []

    for _ in range(n_episodes):
        env.reset()
        done = False
        agent_colour = env.turn_order[0]
        steps = 0
        skips = 0

        while not done and steps < max_steps:
            current = env.current_colour
            legal = env.get_legal_actions(current)

            if not legal:
                skips += 1
                if skips >= env.n_players:
                    break
                env.turn_idx = (env.turn_idx + 1) % len(env.turn_order)
                continue
            skips = 0

            current_obs = env.observe(from_perspective=current)

            if current == agent_colour:
                action, _, _ = agent.select_action(current_obs, legal)
            elif opp_agent is not None:
                action, _, _ = opp_agent.select_action(current_obs, legal)
            else:
                action = random.choice(legal)

            _, _, done, _ = env.step(action)
            steps += 1

        lengths.append(steps)
        if steps >= max_steps:
            truncations += 1

        winner = env.winner
        if winner == agent_colour:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

    return {
        "n_episodes": n_episodes,
        "win_rate": wins / n_episodes,
        "draw_rate": draws / n_episodes,
        "loss_rate": losses / n_episodes,
        "truncation_rate": truncations / n_episodes,
        "avg_game_length": float(np.mean(lengths)),
        "median_game_length": float(np.median(lengths)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained Chinese Checkers agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--agent", required=True, help="Path to .pt checkpoint")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4, 6])
    p.add_argument("--opponent", default="random",
                   choices=["random", "self", "latest"])
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    results = evaluate(
        agent_path=args.agent,
        n_episodes=args.episodes,
        n_players=args.n_players,
        opponent=args.opponent,
        max_steps=args.max_steps,
        device=args.device,
    )
    print("=== Evaluation Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
