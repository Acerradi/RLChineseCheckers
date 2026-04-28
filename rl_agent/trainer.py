"""Self-play PPO training loop.

Design
------
* One PPO agent is the *learner*; it always controls the first colour
  in the turn order (e.g. "red" in a 1v1 game).
* All other seats are filled by a *frozen opponent* — a periodically
  updated snapshot of the learner.
* Ctrl-C (SIGINT) triggers a clean save before the process exits.
* Checkpoints are written every `save_every` episodes and as `latest.pt`.
  Running python -m rl_agent.train auto-resumes from latest.pt if present.
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
    total_episodes: int = 100_000

    # Buffer size (in agent steps) before a PPO gradient update fires
    update_every: int = 512

    # Hard step limit per episode.  Real games take ~100-200 moves per player;
    # 1000 total steps gives ~500 agent moves — enough to complete a game once
    # the agent has learned to make directed progress.
    max_episode_steps: int = 1_000

    # Episodes between syncing the frozen opponent to the latest learner weights
    opponent_sync_every: int = 500

    save_every: int = 500
    log_every: int = 100
    checkpoint_dir: str = "./checkpoints"
    log_file: str = "./training.log"

    # Early stopping: stop when rolling win rate >= this value (0.0 = disabled)
    win_rate_threshold: float = 0.0

    ppo: PPOConfig = field(default_factory=PPOConfig)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SelfPlayTrainer:
    """Self-play PPO trainer for Chinese Checkers."""

    _WIN_WINDOW = 200  # rolling window length for displayed metrics

    def __init__(self, cfg: TrainConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device

        self.env = ChineseCheckersEnv(n_players=cfg.n_players)
        self.agent = PPOAgent(cfg.ppo, device=device)
        self.opponent = PPOAgent(cfg.ppo, device=device)
        self._sync_opponent()

        self.episode: int = 0
        self.total_steps: int = 0
        self._pending_bootstrap: float = 0.0   # value estimate for last truncated state

        self._recent_wins: List[int] = []
        self._recent_lengths: List[int] = []
        self._recent_truncations: List[int] = []
        self._recent_goal_pieces: List[float] = []
        self._recent_dists: List[float] = []   # total hex dist to goal at episode end
        self._last_metrics: Optional[Dict] = None
        self._stop: bool = False

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self._init_logger()
        signal.signal(signal.SIGINT, self._sigint_handler)

    # ------------------------------------------------------------------
    # Setup
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
        self.log.info("Interrupt received — saving and stopping after this episode.")
        self._stop = True

    def _sync_opponent(self) -> None:
        self.opponent.net.load_state_dict(copy.deepcopy(self.agent.net.state_dict()))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tag: Optional[str] = None) -> None:
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
        self.episode = self.agent.load(path)
        self._sync_opponent()
        self.log.info(f"Resumed from {path}  (episode {self.episode})")

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def _run_episode(self) -> Dict:
        """Collect one self-play episode; return episode statistics."""
        self.env.reset()
        agent_colour = self.env.turn_order[0]
        done = False
        steps = 0
        skips = 0

        while not done and steps < self.cfg.max_episode_steps:
            current = self.env.current_colour
            legal = self.env.get_legal_actions(current)

            if not legal:
                skips += 1
                if skips >= self.env.n_players:
                    break
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

        # ------------------------------------------------------------------
        # Terminal reward bookkeeping
        # ------------------------------------------------------------------
        winner = self.env.winner
        agent_won = winner == agent_colour

        if not agent_won and len(self.agent.buffer) > 0:
            self.agent.buffer.patch_last_reward(-1.0, done=True)

        # ------------------------------------------------------------------
        # Bootstrap value for truncated episodes
        #
        # When the episode hits the step limit (not a natural terminal), the
        # last buffer entry has done=False.  GAE needs V(s_{t+1}) to correctly
        # estimate the return — using 0 here would pretend the game ends at
        # truncation and systematically under-estimate future rewards.
        # ------------------------------------------------------------------
        bootstrap_val = 0.0
        truncated = steps >= self.cfg.max_episode_steps and not done
        if truncated and len(self.agent.buffer) > 0:
            last_legal = self.env.get_legal_actions(agent_colour)
            if last_legal:
                last_obs = self.env.observe(from_perspective=agent_colour)
                _, _, bootstrap_val = self.agent.select_action(last_obs, last_legal)

        return {
            "agent_won": agent_won,
            "steps": steps,
            "truncated": truncated,
            "goal_pieces": self.env._pieces_in_goal(agent_colour),
            "dist_to_goal": self.env._total_dist_to_goal(agent_colour),
            "bootstrap_val": bootstrap_val,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.log.info(
            f"Training — episodes={self.cfg.total_episodes}  "
            f"n_players={self.cfg.n_players}  device={self.device}  "
            f"resume_episode={self.episode}  gamma={self.cfg.ppo.gamma}"
        )

        while self.episode < self.cfg.total_episodes and not self._stop:
            ep_info = self._run_episode()
            self.episode += 1

            # Store the bootstrap value from this episode; used if an update
            # fires before the next episode refills the buffer.
            self._pending_bootstrap = ep_info["bootstrap_val"]

            # Rolling stats
            for buf, val in (
                (self._recent_wins,        int(ep_info["agent_won"])),
                (self._recent_lengths,     ep_info["steps"]),
                (self._recent_truncations, int(ep_info["truncated"])),
                (self._recent_goal_pieces, ep_info["goal_pieces"]),
                (self._recent_dists,       ep_info["dist_to_goal"]),
            ):
                buf.append(val)
                if len(buf) > self._WIN_WINDOW:
                    buf.pop(0)

            # PPO update — pass the correct bootstrap so GAE returns are accurate
            if len(self.agent.buffer) >= self.cfg.update_every:
                self._last_metrics = self.agent.update(
                    last_value=self._pending_bootstrap
                )
                self._pending_bootstrap = 0.0

            # Progress log
            if self.episode % self.cfg.log_every == 0:
                wr   = np.mean(self._recent_wins)        if self._recent_wins        else 0.0
                agl  = np.mean(self._recent_lengths)     if self._recent_lengths     else 0.0
                tr   = np.mean(self._recent_truncations) if self._recent_truncations else 0.0
                gp   = np.mean(self._recent_goal_pieces) if self._recent_goal_pieces else 0.0
                dist = np.mean(self._recent_dists)       if self._recent_dists       else 0.0
                m    = self._last_metrics
                loss_str = (
                    f"loss={m['loss']:.4f}  p={m['policy_loss']:.4f}  "
                    f"v={m['value_loss']:.4f}  ent={m['entropy']:.4f}"
                    if m else "no update yet"
                )
                self.log.info(
                    f"ep={self.episode:8d}  "
                    f"win={wr:.3f}  "
                    f"goal={gp:.1f}/10  "
                    f"dist={dist:.1f}  "
                    f"ep_len={agl:.0f}  "
                    f"trunc={tr:.2f}  "
                    f"steps={self.total_steps}  "
                    + loss_str
                )

            if self.episode % self.cfg.save_every == 0:
                self.save()

            # Early stopping
            t = self.cfg.win_rate_threshold
            if t > 0.0 and self._recent_wins and np.mean(self._recent_wins) >= t:
                self.log.info(
                    f"Early stop: win rate {np.mean(self._recent_wins):.3f} >= {t:.3f} "
                    f"at episode {self.episode}"
                )
                break

            if self.episode % self.cfg.opponent_sync_every == 0:
                self._sync_opponent()
                self.log.info(f"Opponent synced at episode {self.episode}")

        self.save("final")
        self.log.info(f"Training finished at episode {self.episode}  steps={self.total_steps}")
