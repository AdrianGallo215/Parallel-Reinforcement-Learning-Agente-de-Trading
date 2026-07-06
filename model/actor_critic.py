"""
Actor-Critic network for A3C trading agent.

Architecture:
    - Shared body: two hidden layers with ReLU
    - Actor head:  outputs action probabilities (policy π)
    - Critic head: outputs state-value V(s)

The obs_dim defaults to 43 = window(20)*2 + 3 (RSI, MACD, position).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 43, n_actions: int = 3, hidden: int = 128):
        super().__init__()

        # ---------- shared trunk ----------
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ---------- actor (policy) head ----------
        self.actor = nn.Linear(hidden, n_actions)

        # ---------- critic (value) head ----------
        self.critic = nn.Linear(hidden, 1)

        # Init weights (orthogonal ─ standard for A3C / PPO)
        self._init_weights()

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)   # small logits at start
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor of shape (batch, obs_dim) or (obs_dim,)

        Returns
        -------
        probs  : action probabilities  (batch, n_actions)
        value  : state-value estimate  (batch, 1)
        """
        h = self.shared(x)
        logits = self.actor(h)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(h)
        return probs, value

    # ------------------------------------------------------------------ #
    def act(self, obs: torch.Tensor):
        """
        Sample an action from the policy and return everything needed for
        the A3C update: action, log_prob, value.
        """
        probs, value = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    # ------------------------------------------------------------------ #
    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Compute log_prob and value for a batch of (obs, action) pairs.
        Used in n-step loss computation.
        """
        probs, value = self.forward(obs)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy

    # ------------------------------------------------------------------ #
    # Convenience helpers for checkpoint save / load
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str):
        """Save model weights to *path*."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path: str, obs_dim: int = 43,
                        n_actions: int = 3, hidden: int = 128,
                        device: str = "cpu"):
        """Create a new model and load weights from *path*."""
        model = cls(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
