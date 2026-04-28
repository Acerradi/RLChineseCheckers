"""PPO agent for Chinese Checkers.

Contains:
  PPOConfig      — hyper-parameter dataclass
  PPOAgent       — wraps PolicyValueNet + Adam optimiser + rollout buffer
                   with save/load for pause-resume training
"""
import dataclasses
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .env import ACTION_DIM
from .network import PolicyValueNet

__all__ = ["PPOConfig", "PPOAgent"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Architecture — input size is fixed (OBS_SIZE = 726); no n_players here
    hidden: int = 256
    n_layers: int = 4
    # Optimisation
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 512


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class _Transition:
    __slots__ = ("obs", "action", "log_prob", "reward", "value", "done", "mask")

    def __init__(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        self.obs = obs
        self.action = action
        self.log_prob = log_prob
        self.reward = reward
        self.value = value
        self.done = done
        self.mask = mask  # bool array shape (ACTION_DIM,)


class _RolloutBuffer:
    def __init__(self) -> None:
        self._data: List[_Transition] = []

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        mask: np.ndarray,
    ) -> None:
        self._data.append(_Transition(obs, action, log_prob, reward, value, done, mask))

    def clear(self) -> None:
        self._data.clear()

    def patch_last_reward(self, delta: float, done: bool = True) -> None:
        """Adjust the reward and done-flag of the most recent transition.

        Used by the trainer to apply the terminal loss penalty after an episode
        ends on the opponent's move.
        """
        if self._data:
            self._data[-1].reward += delta
            self._data[-1].done = done

    def compute_gae(
        self, last_value: float, gamma: float, lam: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised Advantage Estimation.

        Returns (returns, advantages), both float32 arrays of length len(self).
        """
        n = len(self._data)
        adv = np.empty(n, np.float32)
        gae = 0.0
        nv = last_value          # bootstrap value for the state *after* the last stored transition
        for i in reversed(range(n)):
            t = self._data[i]
            cont = 0.0 if t.done else 1.0
            delta = t.reward + gamma * nv * cont - t.value
            gae = delta + gamma * lam * cont * gae
            adv[i] = gae
            nv = t.value         # for the next (earlier) step, V(s_t) is the bootstrap
        values = np.array([t.value for t in self._data], np.float32)
        return adv + values, adv  # returns, advantages

    def to_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        obs = torch.from_numpy(np.stack([t.obs for t in self._data])).to(device)
        actions = torch.tensor(
            [t.action for t in self._data], dtype=torch.long, device=device
        )
        old_lps = torch.tensor(
            [t.log_prob for t in self._data], dtype=torch.float32, device=device
        )
        masks = torch.from_numpy(
            np.stack([t.mask for t in self._data])
        ).to(device)
        return obs, actions, old_lps, masks


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent with actor-critic network.

    Typical usage in a training loop
    ---------------------------------
    obs, legal = env.reset(), env.get_legal_actions()
    action, lp, val = agent.select_action(obs, legal)
    next_obs, reward, done, _ = env.step(action)
    agent.store(obs, action, lp, reward, val, done, legal)
    if len(agent.buffer) >= update_every:
        metrics = agent.update()
    agent.save(path, episode=ep)
    """

    def __init__(self, cfg: PPOConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = PolicyValueNet(
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
        ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = _RolloutBuffer()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self, obs: np.ndarray, legal_actions: List[int]
    ) -> Tuple[int, float, float]:
        """Stochastically select an action.

        Returns (action, log_prob, value_estimate).
        """
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool, device=self.device)
        for a in legal_actions:
            mask[0, a] = True
        action_t, lp_t, val_t = self.net.act(obs_t, mask)
        return action_t.item(), lp_t.item(), val_t.item()

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        legal_actions: List[int],
    ) -> None:
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        for a in legal_actions:
            mask[a] = True
        self.buffer.add(obs, action, log_prob, reward, value, done, mask)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run PPO update on the current buffer and clear it.

        *last_value* is the value-function estimate of the state reached after
        the last stored transition.  Pass 0.0 for terminal states or a network
        estimate for truncated episodes.

        Returns a dict of scalar loss metrics.
        """
        cfg = self.cfg
        returns, advantages = self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # Normalise advantages across the whole batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs, actions, old_lps, masks = self.buffer.to_tensors(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        n = len(self.buffer)

        totals = dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)
        n_updates = 0

        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start: start + cfg.batch_size]

                lps, values = self.net(obs[idx], masks[idx])
                new_lps = lps.gather(1, actions[idx].unsqueeze(1)).squeeze(1)

                ratio = torch.exp(new_lps - old_lps[idx])
                adv = adv_t[idx]
                p_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv,
                ).mean()

                v_loss = F.mse_loss(values, ret_t[idx])

                # Entropy: 0 * log(0) = 0 because we used -1e9 masking (not -inf)
                probs = lps.exp()
                entropy = -(probs * lps).sum(-1).mean()

                loss = p_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                totals["loss"] += loss.item()
                totals["policy_loss"] += p_loss.item()
                totals["value_loss"] += v_loss.item()
                totals["entropy"] += entropy.item()
                n_updates += 1

        self.buffer.clear()
        if n_updates == 0:
            return totals
        return {k: v / n_updates for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, episode: int, extra: Optional[dict] = None) -> None:
        """Save a complete checkpoint (network + optimiser + config + metadata)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "episode": episode,
                "net": self.net.state_dict(),
                "opt": self.opt.state_dict(),
                "cfg": asdict(self.cfg),
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: str) -> int:
        """Load checkpoint and return the saved episode number.

        The network must already have the correct architecture.
        Use PPOAgent.from_checkpoint() when you don't know the architecture
        in advance.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        return ckpt.get("episode", 0)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "PPOAgent":
        """Reconstruct an agent (config + weights) from a checkpoint file."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        valid = {f.name for f in dataclasses.fields(PPOConfig)}
        cfg = PPOConfig(**{k: v for k, v in ckpt["cfg"].items() if k in valid})
        agent = cls(cfg, device=device)
        agent.net.load_state_dict(ckpt["net"])
        agent.opt.load_state_dict(ckpt["opt"])
        return agent
