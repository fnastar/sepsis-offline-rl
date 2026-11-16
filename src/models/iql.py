"""
iql.py: Implicit Q-Learning (IQL) for discrete offline RL (Kostrikov et al. 2022).

IQL avoids querying out-of-distribution actions entirely: the Bellman target
uses a separate state-value network V(s) fitted by EXPECTILE regression on
Q(s, a) over dataset actions only, instead of a max over all actions.

    L_V(ψ) = E_(s,a)~D [ ℓ²_τ( Q_target(s, a) − V_ψ(s) ) ]
             with ℓ²_τ(u) = |τ − 1(u < 0)| · u²      (expectile loss, τ ∈ (0.5, 1))

    L_Q(θ) = E_(s,a,r,s')~D [ ( r + γ (1−done) V_ψ(s') − Q_θ(s, a) )² ]

With τ → 1 the expectile approaches the max over the DATA distribution's
actions in s; τ = 0.5 recovers plain SARSA-style expectation. Because neither
loss evaluates Q at an action outside the dataset, the OOD Q-inflation that
breaks unconstrained D3QN on this data cannot enter the bootstrap — the same
goal CQL achieves by explicit penalisation, reached implicitly.

The Q-network reuses the D3QN dueling architecture so saved checkpoints are
drop-in compatible with the whole evaluation stack (FQE, data_support,
subgroup_analysis all load `D3QN` weights and derive π = softmax(Q/τ_soft)).

Reference: Kostrikov, Nair & Levine, "Offline Reinforcement Learning with
Implicit Q-Learning", ICLR 2022.
"""
import torch
import torch.nn as nn

from src.models.d3qn import D3QN


class ValueNet(nn.Module):
    """State-value network V(s) — same encoder as D3QN/FQENet, scalar head."""

    def __init__(self, state_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.v_head = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s). Shape: (B,)."""
        return self.v_head(self.encoder(state)).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss ℓ²_τ(u) = |τ − 1(u<0)|·u², averaged over batch."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def iql_losses(q_online: D3QN, q_target: D3QN, v_net: ValueNet, batch: tuple,
               gamma: float, expectile: float):
    """
    Compute (L_V, L_Q) for one batch of offline transitions.

    Both losses touch Q only at dataset actions — no max over the action set,
    no OOD queries. Returns (v_loss, q_loss) as differentiable tensors.
    """
    states, actions, rewards, next_states, dones = batch

    # L_V: expectile regression of V(s) toward frozen Q_target(s, a_data)
    with torch.no_grad():
        q_data = q_target(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    v = v_net(states)
    v_loss = expectile_loss(q_data - v, expectile)

    # L_Q: Bellman regression toward r + γ (1−done) V(s') with frozen V
    with torch.no_grad():
        target = rewards + gamma * (1.0 - dones) * v_net(next_states)
    q = q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    q_loss = nn.functional.smooth_l1_loss(q, target)

    return v_loss, q_loss


def build_iql(state_dim: int, n_actions: int, hidden_dims: list[int]):
    """Instantiate (q_online, q_target, v_net); target starts as a copy."""
    q_online = D3QN(state_dim, n_actions, hidden_dims)
    q_target = D3QN(state_dim, n_actions, hidden_dims)
    q_target.load_state_dict(q_online.state_dict())
    q_target.requires_grad_(False)
    v_net = ValueNet(state_dim, hidden_dims)
    return q_online, q_target, v_net


# ─────────────────────────────────────────────────────────────────────────────
# POLICY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
#
# IQL learns Q and V but neither defines a policy. Kostrikov et al. extract one
# by advantage-weighted behavioural cloning, and that extraction step *is* the
# algorithm's policy — scoring softmax(Q/temperature) instead evaluates a
# different method that happens to share IQL's critic, which is why the actor
# below is trained and saved explicitly rather than derived at evaluation time.

class CategoricalActor(nn.Module):
    """Discrete policy π(a|s) over the joint action set, as logits."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden_dims: list[int] = [256, 256]):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.logits_head = nn.Linear(in_dim, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Unnormalised action logits. Shape: (B, A)."""
        return self.logits_head(self.encoder(state))

    def log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(a|s) for the given actions. Shape: (B,)."""
        logp = nn.functional.log_softmax(self.forward(state), dim=1)
        return logp.gather(1, actions.unsqueeze(1)).squeeze(1)

    def probs(self, state: torch.Tensor) -> torch.Tensor:
        """π(·|s). Shape: (B, A), rows sum to 1."""
        return torch.softmax(self.forward(state), dim=1)

    def greedy(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action = argmax over logits. Shape: (B,)."""
        with torch.no_grad():
            return self.forward(state).argmax(dim=1)


def awbc_loss(actor: CategoricalActor, q_target: D3QN, v_net: ValueNet,
              batch: tuple, beta: float = 3.0, weight_clip: float = 100.0):
    """
    Advantage-weighted behavioural cloning (Kostrikov et al. 2022, eq. 7).

        L_π = − E_(s,a)~D [ min(exp(β · (Q(s,a) − V(s))), M) · log π(a|s) ]

    The weight upgrades clinician actions that beat the state's expectile
    value and all but ignores those that fall below it, so the policy stays on
    the data manifold while still improving over the behaviour. Clipping at M
    bounds the influence of any single transition, which matters here because
    the terminal ±15 makes a handful of advantages very large.

    Returns (loss, mean_weight, frac_clipped) — the latter two for logging, as
    a weight distribution that is almost entirely clipped or almost entirely
    zero indicates β is mis-scaled for the reward magnitude.
    """
    states, actions, _rewards, _next_states, _dones = batch
    with torch.no_grad():
        q_data = q_target(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        adv = q_data - v_net(states)
        w = torch.exp(beta * adv)
        frac_clipped = (w > weight_clip).float().mean().item()
        w = w.clamp(max=weight_clip)
    logp = actor.log_prob(states, actions)
    loss = -(w * logp).mean()
    return loss, w.mean().item(), frac_clipped


def build_iql_actor(state_dim: int, n_actions: int,
                    hidden_dims: list[int]) -> CategoricalActor:
    """Instantiate the extraction policy."""
    return CategoricalActor(state_dim, n_actions, hidden_dims)
