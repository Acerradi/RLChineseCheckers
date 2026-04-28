"""Self-play PPO training loop.

Design
------
* One PPO agent is the *learner*; it always controls the first colour
  in the turn order (e.g. "red" in a 1v1 game).
* All other seats are filled by a *frozen opponent* — a periodically
  updated snapshot of the learner.  This is the standard "league" or
  "fictitious self-play" bootstrap for two-player and multi-player games.
* Ctrl-C (SIGINT) triggers a clean save before the process exits.
* Checkpoints are written every `save_every` episodes and as `latest.pt`.
  Resume by passing `--resume checkpoints/latest.pt` to train.py.
* The loop is designed so that expanding from 2 to N players only requires
  changing `n_players`; the agent/env/network all scale accordingly.
"""
import copy
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .agent import PPOAgent, PPOConfig
from .env import ChineseCheckersEnv

__all__ = ["TrainConfig", "SelfPlayTrainer"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_players: int = 2

    # Episode budget
    total_episodes: int = 100_000

    # Minimum agent-side steps before a PPO gradient update is triggered.
    # After the update the buffer is cleared.
    update_every: int = 512

    # Hard limit on steps per episode (prevents infinite games during early
    # training when the agent has not learned to make progress).
    max_episode_steps: int = 500

    # Episodes between syncing the frozen opponent to the latest learner weights
    opponent_sync_every: int = 1_000

    # Episodes between checkpoint saves
    save_every: int = 500

    # Episodes between log lines
    log_every: int = 100

    checkpoint_dir: str = "./checkpoints"
    log_file: str = "./training.log"

    # PPO hyper-parameters (network input is fixed-size; n_players is env-only)
    ppo: PPOConfig = field(default_factory=PPOConfig)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SelfPlayTrainer:
    """Self-play trainer for Chinese Checkers.

    Quick start (1v1)
    -----------------
    cfg = TrainConfig(n_players=2, total_episodes=50_000)
    trainer = SelfPlayTrainer(cfg)
    trainer.train()

    Pause and resume
    ----------------
    trainer.save()            # explicit save at any point
    trainer.load(path)        # load a saved checkpoint before calling train()
    # — or pass --resume to train.py —
    """

    _WIN_WINDOW = 200  # rolling window for win-rate display

    def __init__(self, cfg: TrainConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device

        self.env = ChineseCheckersEnv(n_players=cfg.n_players)
        self.agent = PPOAgent(cfg.ppo, device=device)
        self.opponent = PPOAgent(cfg.ppo, device=device)
        self._sync_opponent()

        self.episode: int = 0
        self.total_steps: int = 0
        self._recent_wins: List[int] = []
        self._recent_lengths: List[int] = []
        self._recent_truncations: List[int] = []
        self._recent_goal_pieces: List[float] = []  # agent pieces in goal at episode end
        self._last_metrics: Optional[Dict] = None   # most recent PPO update metrics
        self._stop: bool = False

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self._init_logger()
        signal.signal(signal.SIGINT, self._sigint_handler)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_logger(self) -> None:
        handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if self.cfg.log_file:
            handlers.append(logging.FileHandler(self.cfg.log_file))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=handlers,
            force=True,
        )
        self.log = logging.getLogger("trainer")

    def _sigint_handler(self, *_) -> None:
        self.log.info("Interrupt received — saving checkpoint and stopping after this episode.")
        self._stop = True

    def _sync_opponent(self) -> None:
        """Copy learner weights into the frozen opponent network."""
        self.opponent.net.load_state_dict(copy.deepcopy(self.agent.net.state_dict()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tag: Optional[str] = None) -> None:
        """Write a named checkpoint and also overwrite latest.pt."""
        tag = tag or f"ep_{self.episode:08d}"
        extra = {
            "total_steps": self.total_steps,
            "win_rate": float(np.mean(self._recent_wins)) if self._recent_wins else 0.0,
        }
        named = os.path.join(self.cfg.checkpoint_dir, f"{tag}.pt")
        self.agent.save(named, self.episode, extra)
        latest = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
        self.agent.save(latest, self.episode, extra)
        self.log.info(f"Checkpoint → {named}")

    def load(self, path: str) -> None:
        """Resume from a checkpoint (network weights + optimiser state)."""
        self.episode = self.agent.load(path)
        self._sync_opponent()
        self.log.info(f"Resumed from {path}  (episode {self.episode})")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def _run_episode(self) -> Dict:
        """Collect one episode of self-play.

        The learner controls the first colour; the frozen opponent controls
        all other colours.  Only learner transitions are stored in the buffer.
        """
        self.env.reset()
        agent_colour = self.env.turn_order[0]
        done = False
        steps = 0
        skips = 0  # consecutive no-legal-move turns

        while not done and steps < self.cfg.max_episode_steps:
            current = self.env.current_colour
            legal = self.env.get_legal_actions(current)

            if not legal:
                # Skip this player's turn (stuck piece configuration)
                skips += 1
                if skips >= self.env.n_players:
                    break  # all players stuck — terminate
                self.env.turn_idx = (self.env.turn_idx + 1) % len(self.env.turn_order)
                continue
            skips = 0

            current_obs = self.env.observe(from_perspective=current)
            is_learner = current == agent_colour

            if is_learner:
                action, lp, val = self.agent.select_action(current_obs, legal)
            else:
                action, lp, val = self.opponent.select_action(current_obs, legal)

            _, reward, done, _ = self.env.step(action)
            steps += 1

            if is_learner:
                self.agent.store(current_obs, action, lp, reward, val, done, legal)
                self.total_steps += 1

        # -----------------------------------------------------------------
        # Terminal reward bookkeeping
        # The env gives +1 to the winner on the winning move.
        # If the opponent won, the learner must be penalised here.
        # -----------------------------------------------------------------
        winner = self.env.winner
        agent_won = winner == agent_colour

        if not agent_won and len(self.agent.buffer) > 0:
            # -1 penalty; also mark done so GAE does not bootstrap past this point
            self.agent.buffer.patch_last_reward(-1.0, done=True)

        return {
            "agent_won": agent_won,
            "winner": winner,
            "steps": steps,
            "truncated": steps >= self.cfg.max_episode_steps,
            "goal_pieces": self.env._pieces_in_goal(agent_colour),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the training loop until total_episodes or Ctrl-C."""
        self.log.info(
            f"Training — episodes={self.cfg.total_episodes}  "
            f"n_players={self.cfg.n_players}  device={self.device}  "
            f"resume_episode={self.episode}"
        )

        while self.episode < self.cfg.total_episodes and not self._stop:
            ep_info = self._run_episode()
            self.episode += 1

            # Rolling stats (capped at WIN_WINDOW episodes)
            for buf, val in (
                (self._recent_wins,       int(ep_info["agent_won"])),
                (self._recent_lengths,    ep_info["steps"]),
                (self._recent_truncations, int(ep_info["truncated"])),
                (self._recent_goal_pieces, ep_info["goal_pieces"]),
            ):
                buf.append(val)
                if len(buf) > self._WIN_WINDOW:
                    buf.pop(0)

            # PPO gradient update once enough on-policy data has accumulated
            if len(self.agent.buffer) >= self.cfg.update_every:
                self._last_metrics = self.agent.update()

            # Progress log — always fires every log_every episodes
            if self.episode % self.cfg.log_every == 0:
                wr   = np.mean(self._recent_wins)        if self._recent_wins        else 0.0
                agl  = np.mean(self._recent_lengths)     if self._recent_lengths     else 0.0
                tr   = np.mean(self._recent_truncations) if self._recent_truncations else 0.0
                gp   = np.mean(self._recent_goal_pieces) if self._recent_goal_pieces else 0.0
                m    = self._last_metrics
                loss_str = (
                    f"loss={m['loss']:.4f}  p={m['policy_loss']:.4f}  "
                    f"v={m['value_loss']:.4f}  ent={m['entropy']:.4f}"
                    if m else "no update yet"
                )
                self.log.info(
                    f"ep={self.episode:8d}  "
                    f"win={wr:.3f}  "
                    f"goal_pieces={gp:.2f}/10  "
                    f"ep_len={agl:.0f}  "
                    f"trunc={tr:.3f}  "
                    f"steps={self.total_steps}  "
                    + loss_str
                )

            # Periodic checkpoint
            if self.episode % self.cfg.save_every == 0:
                self.save()

            # Sync frozen opponent
            if self.episode % self.cfg.opponent_sync_every == 0:
                self._sync_opponent()
                self.log.info(f"Opponent synced at episode {self.episode}")

        self.save("final")
        self.log.info(f"Training finished at episode {self.episode}  steps={self.total_steps}")
