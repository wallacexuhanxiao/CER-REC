import torch
from torch import nn

from src.models.event_relation import EventRelation


class EventRouteGate(nn.Module):
    def __init__(self, relation_hidden_dim=64, route_hidden_dim=128, dropout=0.1):
        super().__init__()
        input_dim = relation_hidden_dim * 2 + 7
        self.net = nn.Sequential(
            nn.Linear(input_dim, route_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(route_hidden_dim, route_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(route_hidden_dim // 2, 1),
        )

    def forward(self, z_cf, z_sem, r_cf, r_sem, target_freq, hist_freq, user_len, recency):
        batch, candidates, length = r_cf.shape
        target_feat = target_freq[:, :, None, None].expand(-1, -1, length, -1)
        hist_feat = hist_freq[:, None, :, None].expand(-1, candidates, -1, -1)
        user_feat = user_len[:, None, None, None].expand(-1, candidates, length, -1)
        recency_feat = recency[:, None, :, None].expand(-1, candidates, -1, -1)
        scalar_feats = torch.cat(
            [
                r_cf.unsqueeze(-1),
                r_sem.unsqueeze(-1),
                (r_cf - r_sem).unsqueeze(-1),
                target_feat,
                hist_feat,
                user_feat,
                recency_feat,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.net(torch.cat([z_cf, z_sem, scalar_feats], dim=-1))).squeeze(-1)


class DualEventFusion(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        relation_hidden_dim=64,
        route_hidden_dim=128,
        dropout=0.1,
        route_mode="learned",
        candidate_chunk_size=16,
        eps=1e-8,
    ):
        super().__init__()
        self.cf_relation = EventRelation(hidden_dim, relation_hidden_dim, dropout)
        self.semantic_relation = EventRelation(hidden_dim, relation_hidden_dim, dropout)
        self.route_mode = route_mode
        self.candidate_chunk_size = candidate_chunk_size
        self.eps = eps
        if route_mode == "learned":
            self.route_gate = EventRouteGate(relation_hidden_dim, route_hidden_dim, dropout)
        elif route_mode != "fixed_half":
            raise ValueError(f"unsupported route_mode: {route_mode}")

    def _masked_softmax(self, scores, history_mask):
        scores = scores.masked_fill(~history_mask[:, None, :], -1e9)
        return torch.softmax(scores, dim=-1).masked_fill(~history_mask[:, None, :], 0.0)

    def _score_chunk(
        self,
        cf_states,
        sem_states,
        cf_candidates,
        sem_candidates,
        history_mask,
        target_freq,
        hist_freq,
        user_len,
        recency,
        cf_temperature,
        sem_temperature,
    ):
        z_cf, r_cf = self.cf_relation(cf_states, cf_candidates)
        z_sem, r_sem = self.semantic_relation(sem_states, sem_candidates)
        alpha_cf = self._masked_softmax(r_cf, history_mask)
        alpha_sem = self._masked_softmax(r_sem, history_mask)
        if self.route_mode == "fixed_half":
            gate = torch.full_like(alpha_cf, 0.5)
        else:
            gate = self.route_gate(z_cf, z_sem, r_cf, r_sem, target_freq, hist_freq, user_len, recency)
            gate = gate.masked_fill(~history_mask[:, None, :], 0.0)

        gated_cf = gate * alpha_cf
        gated_sem = (1.0 - gate) * alpha_sem
        joint_mass = gated_cf.sum(dim=-1, keepdim=True) + gated_sem.sum(dim=-1, keepdim=True) + self.eps
        norm_cf = gated_cf / joint_mass
        norm_sem = gated_sem / joint_mass
        cf_branch_mass = norm_cf.sum(dim=-1)
        pooled_cf = torch.einsum("bcl,bld->bcd", norm_cf, cf_states)
        pooled_sem = torch.einsum("bcl,bld->bcd", norm_sem, sem_states)
        score_cf = (pooled_cf * cf_candidates).sum(dim=-1) / cf_temperature
        score_sem = (pooled_sem * sem_candidates).sum(dim=-1) / sem_temperature
        return score_cf + score_sem, gate, cf_branch_mass

    def forward(
        self,
        cf_states,
        sem_states,
        cf_candidates,
        sem_candidates,
        history_mask,
        target_freq,
        hist_freq,
        user_len,
        recency,
        cf_temperature,
        sem_temperature,
    ):
        scores, gates, branch_masses = [], [], []
        for start in range(0, cf_candidates.shape[1], self.candidate_chunk_size):
            end = min(cf_candidates.shape[1], start + self.candidate_chunk_size)
            chunk_scores, chunk_gates, chunk_branch_masses = self._score_chunk(
                cf_states,
                sem_states,
                cf_candidates[:, start:end],
                sem_candidates[:, start:end],
                history_mask,
                target_freq[:, start:end],
                hist_freq,
                user_len,
                recency,
                cf_temperature,
                sem_temperature,
            )
            scores.append(chunk_scores)
            gates.append(chunk_gates)
            branch_masses.append(chunk_branch_masses)
        return torch.cat(scores, dim=1), torch.cat(gates, dim=1), torch.cat(branch_masses, dim=1)

