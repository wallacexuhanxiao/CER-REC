import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.event_gate import DualEventFusion


class ConstantGate(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, z_cf, z_sem, r_cf, r_sem, target_freq, hist_freq, user_len, recency):
        return torch.full_like(r_cf, self.value)


def make_inputs(batch=3, candidates=2, length=5, hidden_dim=4):
    cf_states = torch.ones(batch, length, hidden_dim)
    sem_states = torch.full((batch, length, hidden_dim), 2.0)
    cf_candidates = torch.ones(batch, candidates, hidden_dim)
    sem_candidates = torch.ones(batch, candidates, hidden_dim)
    history_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, False, False, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    target_freq = torch.zeros(batch, candidates)
    hist_freq = torch.zeros(batch, length)
    user_len = torch.ones(batch)
    recency = torch.zeros(batch, length)
    return cf_states, sem_states, cf_candidates, sem_candidates, history_mask, target_freq, hist_freq, user_len, recency


def build_model(gate_value):
    model = DualEventFusion(
        hidden_dim=4,
        relation_hidden_dim=8,
        route_hidden_dim=16,
        dropout=0.0,
        route_mode="learned",
        candidate_chunk_size=1,
    )
    model.route_gate = ConstantGate(gate_value)
    return model


def forward_with_gate(gate_value):
    model = build_model(gate_value)
    return model(*make_inputs(), cf_temperature=1.0, sem_temperature=1.0)


def test_constant_gate_controls_effective_branch_mass():
    for value in [0.1, 0.5, 0.9]:
        _, _, branch_mass = forward_with_gate(value)
        expected = torch.full_like(branch_mass, value)
        assert torch.allclose(branch_mass, expected, atol=1e-5)


def test_gate_extremes_select_expected_branch_score():
    scores_cf, _, branch_cf = forward_with_gate(0.999)
    scores_sem, _, branch_sem = forward_with_gate(0.001)
    hidden_dim = 4
    assert torch.allclose(branch_cf, torch.full_like(branch_cf, 0.999), atol=1e-5)
    assert torch.allclose(branch_sem, torch.full_like(branch_sem, 0.001), atol=1e-5)
    assert torch.allclose(scores_cf, torch.full_like(scores_cf, hidden_dim * (2.0 - 0.999)), atol=1e-4)
    assert torch.allclose(scores_sem, torch.full_like(scores_sem, hidden_dim * (2.0 - 0.001)), atol=1e-4)


def test_joint_mass_sums_to_one_for_all_candidates():
    _, _, branch_mass = forward_with_gate(0.37)
    sem_mass = 1.0 - branch_mass
    assert torch.allclose(branch_mass + sem_mass, torch.ones_like(branch_mass), atol=1e-6)


def test_padding_is_zero_and_backward_has_no_nan():
    model = build_model(0.7)
    inputs = list(make_inputs())
    for tensor in inputs[:4]:
        tensor.requires_grad_(True)
    scores, gates, branch_mass = model(*inputs, cf_temperature=1.0, sem_temperature=1.0)
    history_mask = inputs[4]
    assert torch.all(gates.masked_select(~history_mask[:, None, :]) == 0)
    assert not torch.isnan(scores).any()
    assert not torch.isnan(gates).any()
    assert not torch.isnan(branch_mass).any()
    loss = scores.sum() + branch_mass.sum()
    loss.backward()
    for tensor in inputs[:4]:
        assert tensor.grad is not None
        assert not torch.isnan(tensor.grad).any()
