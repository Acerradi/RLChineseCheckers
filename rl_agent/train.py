"""Training entry point.

Usage (from the project root)
------------------------------
  # 1v1, start from scratch
  python -m rl_agent.train

  # Resume from the last auto-saved checkpoint
  python -m rl_agent.train --resume rl_agent/checkpoints/latest.pt

  # 4-player game, custom hyper-parameters
  python -m rl_agent.train --n-players 4 --episodes 200000 --hidden 512

Run `python -m rl_agent.train --help` for the full option list.
"""
import argparse
import os

import torch

from .agent import PPOConfig
from .trainer import SelfPlayTrainer, TrainConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Chinese Checkers RL agent via self-play PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Game
    p.add_argument("--n-players", type=int, default=2, choices=[2, 3, 4, 6],
                   help="Number of players (2 = 1v1)")

    # Training budget
    p.add_argument("--episodes", type=int, default=100_000,
                   help="Total training episodes")
    p.add_argument("--max-episode-steps", type=int, default=500,
                   help="Hard limit on steps per episode (prevents infinite games)")

    # Checkpoint / resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a .pt checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=str, default="rl_agent/checkpoints",
                   help="Directory for checkpoints and latest.pt")
    p.add_argument("--save-every", type=int, default=500,
                   help="Episodes between named checkpoint saves")

    # Network
    p.add_argument("--hidden", type=int, default=256,
                   help="Hidden layer width")
    p.add_argument("--n-layers", type=int, default=4,
                   help="Number of residual blocks")

    # PPO hypers
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--update-every", type=int, default=512,
                   help="Learner-side steps between PPO updates")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    p.add_argument("--entropy-coef", type=float, default=0.01,
                   help="Entropy bonus coefficient")
    p.add_argument("--n-epochs", type=int, default=4,
                   help="PPO optimisation epochs per update")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Mini-batch size within each PPO epoch")

    # Self-play
    p.add_argument("--opponent-sync-every", type=int, default=1_000,
                   help="Episodes between syncing the frozen opponent")

    # Logging
    p.add_argument("--log-every", type=int, default=100,
                   help="Episodes between log lines")
    p.add_argument("--log-file", type=str, default="rl_agent/training.log")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    ppo = PPOConfig(
        hidden=args.hidden,
        n_layers=args.n_layers,
        lr=args.lr,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )
    cfg = TrainConfig(
        n_players=args.n_players,
        total_episodes=args.episodes,
        update_every=args.update_every,
        max_episode_steps=args.max_episode_steps,
        opponent_sync_every=args.opponent_sync_every,
        save_every=args.save_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        log_file=args.log_file,
        ppo=ppo,
    )

    trainer = SelfPlayTrainer(cfg, device=device)

    if args.resume:
        trainer.load(args.resume)
    else:
        auto = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(auto):
            print(
                f"\nFound an existing checkpoint at {auto}.\n"
                f"Pass --resume {auto} to continue, or delete it to start fresh.\n"
            )

    trainer.train()


if __name__ == "__main__":
    main()
