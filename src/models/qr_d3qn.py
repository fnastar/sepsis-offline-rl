"""
qr_d3qn.py: Quantile Regression Dueling Double DQN (QR-D3QN).

Extends D3QN by replacing scalar Q-values with a distribution over returns,
represented as N fixed quantile levels (Dabney et al. 2018).

Each action's value is described by n_quantiles quantile estimates.
The policy can use the mean (standard) or a lower quantile (conservative).
"""
import torch
import torch.nn as nn


class QRD3QN(nn.Module):
    """
    Quantile Regression D3QN.

    Output shape: (B, n_actions, n_quantiles) — quantile values per action.
    Mean Q(s,a) = mean over quantiles.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: list[int] = [256, 256],
        n_quantiles: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        self.value_head = nn.Linear(in_dim, n_quantiles)
        self.advantage_head = nn.Linear(in_dim, n_actions * n_quantiles)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return quantile values. Shape: (B, n_actions, n_quantiles)."""
        x = self.encoder(state)
        v = self.value_head(x).unsqueeze(1)                              # (B, 1, Q)
        a = self.advantage_head(x).view(-1, self.n_actions, self.n_quantiles)  # (B, A, Q)
        return v + a - a.mean(dim=1, keepdim=True)                       # (B, A, Q)

    def q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Mean Q-values across quantiles. Shape: (B, n_actions)."""
        return self.forward(state).mean(dim=2)

    def select_action(self, state: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
        """Action selection using a specific quantile level (0.5 = median)."""
        with torch.no_grad():
            quantiles = self.forward(state)  # (B, A, Q)
            idx = int(quantile * self.n_quantiles)
            idx = max(0, min(idx, self.n_quantiles - 1))
            sorted_q, _ = quantiles.sort(dim=2)
            return sorted_q[:, :, idx].argmax(dim=1)


def quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile Huber loss (Dabney et al. 2018).

    pred:   (B, n_quantiles) — predicted quantile values for the taken action
    target: (B, n_quantiles) — target quantile values
    taus:   (n_quantiles,)   — quantile levels τ_i = (i + 0.5) / N

    Returns scalar loss.
    """
    # Pairwise TD errors: (B, n_quantiles_pred, n_quantiles_target)
    td_error = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, Q, Q)

    # Huber loss element
    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )

    # Asymmetric weighting by quantile level
    tau_weight = (taus.unsqueeze(0).unsqueeze(2) - (td_error < 0).float()).abs()
    loss = (tau_weight * huber).sum(dim=2).mean(dim=1)  # (B,)
    return loss.mean()


def build_qr_d3qn(
    state_dim: int,
    n_actions: int,
    hidden_dims: list[int],
    n_quantiles: int = 32,
) -> tuple:
    """Instantiate online and target QR-D3QN networks."""
    online = QRD3QN(state_dim, n_actions, hidden_dims, n_quantiles)
    target = QRD3QN(state_dim, n_actions, hidden_dims, n_quantiles)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    return online, target
