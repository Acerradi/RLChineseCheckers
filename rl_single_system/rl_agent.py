import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.net = PolicyValueNet(obs_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def act(self, obs, action_mask):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        if torch.any(mask_t.sum(dim=1) <= 0):
            raise ValueError("act() received an all-zero action mask")

        logits, value = self.net(obs_t)

        masked_logits = logits.masked_fill(mask_t <= 0, -1e9)
        dist = Categorical(logits=masked_logits)

        action = dist.sample()
        logprob = dist.log_prob(action)

        return int(action.item()), float(logprob.item()), float(value.item())

    def evaluate_actions(self, obs_batch, action_batch, action_mask_batch):
        logits, values = self.net(obs_batch)

        if torch.any(action_mask_batch.sum(dim=1) <= 0):
            raise ValueError("Found sample with all-zero action mask")

        masked_logits = logits.masked_fill(action_mask_batch <= 0, -1e9)
        dist = Categorical(logits=masked_logits)

        logprobs = dist.log_prob(action_batch)
        entropy = dist.entropy()
        return logprobs, entropy, values

    def update(self, batch, epochs=4, minibatch_size=64):
        obs = torch.tensor(np.array(batch["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(np.array(batch["action_masks"]), dtype=torch.float32, device=self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        n = obs.shape[0]
        if n == 0:
            return

        idxs = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(idxs)

            for start in range(0, n, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                mb_obs = obs[mb]
                mb_actions = actions[mb]
                mb_old_logprobs = old_logprobs[mb]
                mb_returns = returns[mb]
                mb_advantages = advantages[mb]
                mb_masks = action_masks[mb]

                new_logprobs, entropy, values = self.evaluate_actions(
                    mb_obs, mb_actions, mb_masks
                )

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values - mb_returns) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()