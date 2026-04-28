"""Actor-critic network for Chinese Checkers.

Architecture: MLP with residual blocks.
  Input  : float32 (batch, OBS_SIZE)  — fixed 726-element binary occupancy map
  Output : log_probs (batch, ACTION_DIM), value (batch,)

The input is always OBS_SIZE = BOARD_SIZE * MAX_PLAYERS = 726 regardless of
how many players are in the game.  Absent player slots are zeroed out by the
environment, so one trained model works for 2-, 3-, 4-, and 6-player games.

A GNN over the hex lattice would give stronger inductive bias, but this
MLP baseline is straightforward to train and extend.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .env import ACTION_DIM, OBS_SIZE

__all__ = ["PolicyValueNet"]


class _ResBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return F.relu(x + h)


class PolicyValueNet(nn.Module):
    """Shared-trunk actor-critic.

    Input is always OBS_SIZE (726) floats.  Absent player slots are zeroed
    by the environment, so the same model weights work for any player count.

    Parameters
    ----------
    hidden   : width of each hidden layer
    n_layers : number of residual blocks
    """

    def __init__(self, hidden: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(OBS_SIZE, hidden), nn.ReLU())
        self.trunk = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_layers)])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, ACTION_DIM),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs        : (batch, OBS_SIZE)  — always 726 floats
        legal_mask : (batch, ACTION_DIM) bool — True where action is legal

        Returns
        -------
        log_probs : (batch, ACTION_DIM)  — log-softmax over legal actions
        value     : (batch,)
        """
        h = self.trunk(self.stem(obs))
        logits = self.policy_head(h)
        if legal_mask is not None:
            # Use -1e9 (not -inf) so that 0 * logit = 0 in entropy (no nan)
            logits = logits.masked_fill(~legal_mask, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        value = self.value_head(h).squeeze(-1)
        return log_probs, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action stochastically.

        Returns (action, log_prob, value)  — all shape (batch,).
        """
        log_probs, value = self.forward(obs, legal_mask)
        probs = log_probs.exp()
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lp = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, lp, value
