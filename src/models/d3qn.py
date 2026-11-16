"""
d3qn.py: Dueling Double Deep Q-Network (D3QN) for offline RL on sepsis treatment.

Architecture (Wang et al. 2016 / Roggeveen et al. 2021):
- Shared encoder: MLP with LayerNorm + ReLU
- Value head: Linear(hidden → 1)
- Advantage head: Linear(hidden → n_actions)
- Q(s, a) = V(s) + A(s, a) - mean_a[A(s, a)]
"""
import torch
import torch.nn as nn


class D3QN(nn.Module):
    """
    Dueling Double DQN.

    Parameters
    ----------
    state_dim   : number of state features (36 after preprocessing)
    n_actions   : number of discrete actions (25)
    hidden_dims : hidden layer sizes for the shared encoder
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Shared encoder
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        # Dueling heads — single linear layer each (Wang et al. 2016)
        self.value_head = nn.Linear(in_dim, 1)
        self.advantage_head = nn.Linear(in_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions. Shape: (B, n_actions)."""
        x = self.encoder(state)
        v = self.value_head(x)                            # (B, 1)
        a = self.advantage_head(x)                        # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)        # (B, A)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy action selection. Shape: (B,)."""
        with torch.no_grad():
            return self.forward(state).argmax(dim=1)


def build_d3qn(state_dim: int, n_actions: int, hidden_dims: list[int]) -> tuple:
    """
    Instantiate online and target networks.
    Returns (online_net, target_net) — target starts as a copy of online.
    """
    online = D3QN(state_dim, n_actions, hidden_dims)
    target = D3QN(state_dim, n_actions, hidden_dims)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    return online, target
