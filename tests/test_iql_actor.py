"""
Tests for the IQL extraction policy.

The earlier IQL runs evaluated softmax(Q/0.5), which is not the IQL policy, so
the reported negative result described Q-softmax extraction rather than IQL.
These tests exist to make that substitution impossible to repeat silently: they
assert the actor exists, actually trains, round-trips through a checkpoint
bit-for-bit, emits a valid categorical distribution, and that loading refuses
rather than falling back when a run has no saved actor.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.models.d3qn import D3QN
from src.models.iql import (
    CategoricalActor, ValueNet, awbc_loss, build_iql_actor,
)

STATE_DIM, N_ACTIONS, B = 12, 25, 64


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, STATE_DIM, generator=g),
        torch.randint(0, N_ACTIONS, (B,), generator=g),
        torch.randn(B, generator=g),
        torch.randn(B, STATE_DIM, generator=g),
        torch.zeros(B),
    )


def _critic():
    torch.manual_seed(0)
    q = D3QN(STATE_DIM, N_ACTIONS, [32, 32]).eval().requires_grad_(False)
    v = ValueNet(STATE_DIM, [32, 32]).eval().requires_grad_(False)
    return q, v


def test_probs_form_a_valid_distribution():
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    p = actor.probs(torch.randn(B, STATE_DIM))
    assert p.shape == (B, N_ACTIONS)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(1), torch.ones(B), atol=1e-5)


def test_greedy_matches_argmax_of_logits():
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    s = torch.randn(B, STATE_DIM)
    assert torch.equal(actor.greedy(s), actor(s).argmax(dim=1))


def test_log_prob_agrees_with_probs():
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    s = torch.randn(B, STATE_DIM)
    a = torch.randint(0, N_ACTIONS, (B,))
    direct = actor.probs(s).gather(1, a.unsqueeze(1)).squeeze(1).log()
    assert torch.allclose(actor.log_prob(s, a), direct, atol=1e-5)


def test_actor_parameters_actually_move():
    """A loss that runs but leaves weights untouched is the failure to catch."""
    q, v = _critic()
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    before = [p.detach().clone() for p in actor.parameters()]
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    for i in range(20):
        loss, _, _ = awbc_loss(actor, q, v, _batch(i))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = list(actor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert all(torch.isfinite(p).all() for p in after)


def test_training_reduces_loss():
    q, v = _critic()
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    b = _batch(7)
    first, _, _ = awbc_loss(actor, q, v, b)
    for _ in range(50):
        loss, _, _ = awbc_loss(actor, q, v, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    last, _, _ = awbc_loss(actor, q, v, b)
    assert last.item() < first.item()


def test_weights_are_clipped():
    """Terminal +/-15 makes some advantages huge; the clip must bind."""
    q, v = _critic()
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    _, mean_w, frac = awbc_loss(actor, q, v, _batch(3),
                                beta=50.0, weight_clip=10.0)
    assert mean_w <= 10.0 + 1e-6
    assert frac > 0.0


def test_checkpoint_round_trips_identically(tmp_path):
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    s = torch.randn(B, STATE_DIM)
    before = actor(s).detach().clone()
    path = tmp_path / "actor_best.pt"
    torch.save(actor.state_dict(), path)

    reloaded = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    reloaded.load_state_dict(torch.load(path))
    reloaded.eval()
    assert torch.equal(before, reloaded(s))


def test_load_actor_refuses_when_missing(tmp_path):
    """Must fail loudly rather than silently scoring softmax(Q) as 'IQL'."""
    from src.train_iql_actor import load_actor
    with pytest.raises(FileNotFoundError, match="no extracted policy"):
        load_actor(str(tmp_path))


def test_load_actor_returns_saved_policy(tmp_path):
    from src.train_iql_actor import load_actor
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    torch.save(actor.state_dict(), tmp_path / "actor_best.pt")
    with open(tmp_path / "actor_hyperparams.json", "w") as f:
        json.dump({"n_state_features": STATE_DIM, "hidden_dims": [32, 32],
                   "beta": 3.0, "weight_clip": 100.0}, f)
    loaded = load_actor(str(tmp_path))
    s = torch.randn(B, STATE_DIM)
    assert torch.equal(actor.eval()(s), loaded(s))


def test_actor_is_not_q_softmax():
    """
    Guards the specific substitution that produced the earlier result: an actor
    trained by advantage-weighted BC should not reproduce argmax(Q).
    """
    q, v = _critic()
    actor = build_iql_actor(STATE_DIM, N_ACTIONS, [32, 32])
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    for i in range(60):
        loss, _, _ = awbc_loss(actor, q, v, _batch(i))
        opt.zero_grad(); loss.backward(); opt.step()
    s = torch.randn(256, STATE_DIM)
    assert not torch.equal(actor.greedy(s), q(s).argmax(dim=1))


def _tiny_actor_run(tmp_path, state_dim=6):
    """A run directory holding a freshly built, untrained actor."""
    import json
    import torch
    from config.rl import N_ACTIONS
    from src.models.iql import build_iql_actor
    actor = build_iql_actor(state_dim, N_ACTIONS, [16, 16])
    torch.save(actor.state_dict(), tmp_path / "actor_best.pt")
    with open(tmp_path / "actor_hyperparams.json", "w") as f:
        json.dump({"n_state_features": state_dim, "hidden_dims": [16, 16]}, f)
    return str(tmp_path)


def test_ope_uses_the_actor_and_not_the_critic(tmp_path):
    """
    The OPE diagnostics must describe the policy the thesis reports.

    `final_ope` used to load `d3qn_*.pt` for IQL, so its importance weights
    described softmax over the critic. That is a different policy.
    """
    import numpy as np
    import torch
    from src.final_ope import _actor_policy_fn
    run = _tiny_actor_run(tmp_path)
    factory = _actor_policy_fn(run, "best", torch.device("cpu"))
    assert factory is not None
    states = np.zeros((8, 6), dtype=np.float32)
    p = factory(0.5)(states)
    assert p.shape == (8, 25)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_ope_temperature_sharpens_the_actor(tmp_path):
    """A lower temperature must concentrate the policy, not leave it unchanged."""
    import numpy as np
    import torch
    from src.final_ope import _actor_policy_fn
    rng = np.random.RandomState(0)
    factory = _actor_policy_fn(_tiny_actor_run(tmp_path), "best",
                               torch.device("cpu"))
    states = rng.randn(64, 6).astype(np.float32)
    hot = factory(1.0)(states).max(axis=1).mean()
    cold = factory(0.05)(states).max(axis=1).mean()
    assert cold > hot


def test_ope_skips_a_run_with_no_actor(tmp_path):
    """
    A run without an extracted policy yields None so the caller can skip it.

    Returning something would mean silently scoring the critic instead.
    """
    import torch
    from src.final_ope import _actor_policy_fn
    assert _actor_policy_fn(str(tmp_path), "best", torch.device("cpu")) is None


# --- exact-argmax policies for the OPE diagnostic ---------------------------


def test_greedy_q_policy_is_one_hot(tmp_path):
    """
    FQE evaluates argmax(Q), so the greedy OPE row must evaluate argmax(Q).

    A softmax at a low temperature shares the modal action but not the
    likelihood ratios, so its weights, effective sample size and every
    estimator built on them describe a different policy.
    """
    import numpy as np
    import torch
    from config.rl import N_ACTIONS
    from src.models.d3qn import D3QN
    from src.final_ope import _greedy_q_fn
    net = D3QN(6, N_ACTIONS, [256, 256])
    ckpt = tmp_path / "d3qn_best.pt"
    torch.save(net.state_dict(), ckpt)

    rng = np.random.RandomState(0)
    states = rng.randn(32, 6).astype(np.float32)
    p = _greedy_q_fn(str(ckpt), 6, torch.device("cpu"))(states)

    assert p.shape == (32, N_ACTIONS)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert set(np.unique(p)) <= {0.0, 1.0}
    with torch.no_grad():
        expected = net(torch.tensor(states)).argmax(dim=1).numpy()
    assert (p.argmax(axis=1) == expected).all()


def test_greedy_actor_policy_is_one_hot(tmp_path):
    """The same requirement for IQL, whose policy is its actor."""
    import numpy as np
    import torch
    from src.final_ope import _actor_policy_fn
    factory = _actor_policy_fn(_tiny_actor_run(tmp_path), "best",
                               torch.device("cpu"))
    p = factory("greedy")(np.random.RandomState(0).randn(16, 6).astype(np.float32))
    assert set(np.unique(p)) <= {0.0, 1.0}
    assert np.allclose(p.sum(axis=1), 1.0)
